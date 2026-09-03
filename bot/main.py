import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta

import discord
import httpx
from discord.ext import commands, tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"].rstrip("/")
LLM_API_URL = f"{LLM_BASE_URL}/chat/completions"
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY") or "{}")
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])
SEND_HOUR = int(os.getenv("SEND_HOUR", "9"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "data/history.json")

SYSTEM_PROMPT = """너는 군사 암구호 생성기다.
반드시 아래 JSON 형식만 출력하라. 설명, 마크다운, 추가 텍스트 없이 JSON만 출력하라.

출력 형식:
{"문어": "단어", "답어": "단어"}

규칙:
- 문어와 답어는 각각 실제로 존재하는 완전한 한국어 명사 한 단어
- 두 단어는 의미적 연관성이 없어야 함 (연상, 유추, 범주 공유 불가)
- 매번 새롭고 다양한 단어 조합 생성
- 사용자 메시지에 "사용 금지 단어" 목록이 주어지면 문어와 답어 모두 그 목록에 없는 단어여야 함
- 아래 예시는 형식 참고용일 뿐이므로, 금지 목록에 있는 예시 단어는 사용 불가
- JSON 외 어떤 텍스트도 출력 금지

올바른 예시:
{"문어": "태양", "답어": "국수"}
{"문어": "독수리", "답어": "냄비"}
{"문어": "바다", "답어": "기와"}
{"문어": "산", "답어": "열쇠"}
{"문어": "불꽃", "답어": "지렁이"}

잘못된 예시 (의미 연관 있음 - 사용 금지):
{"문어": "태양", "답어": "빛"}
{"문어": "독수리", "답어": "창공"}
{"문어": "바다", "답어": "파도"}
{"문어": "산", "답어": "구름"}
{"문어": "불꽃", "답어": "연기"}"""

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


ATTEMPT_TIMEOUT = 300.0


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("이력 파일을 읽을 수 없어 빈 목록으로 시작합니다 (%s): %s", HISTORY_FILE, e)
        return []


def save_history(history: list) -> None:
    directory = os.path.dirname(HISTORY_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{HISTORY_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, HISTORY_FILE)


def used_words(history: list) -> list:
    words = []
    for entry in history:
        for key in ("문어", "답어"):
            word = str(entry.get(key, "")).strip()
            if word and word not in words:
                words.append(word)
    return words


HISTORY = load_history()
logger.info("암구호 이력 %d건 로드 (%s)", len(HISTORY), HISTORY_FILE)


async def _call_llm_once(banned: list) -> dict:
    user_content = "오늘의 암구호를 생성하라."
    if banned:
        user_content += (
            "\n\n아래 단어들은 이전에 이미 사용된 단어다. "
            "문어와 답어 모두 이 목록에 없는 완전히 새로운 단어로 생성하라.\n"
            "사용 금지 단어: " + ", ".join(banned)
        )

    async with httpx.AsyncClient(timeout=ATTEMPT_TIMEOUT) as client:
        resp = await client.post(
            LLM_API_URL,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Accept": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.9,
                "top_p": 0.95,
                "max_tokens": 8192,
                "stream": False,
                **LLM_EXTRA_BODY,
            },
        )
        resp.raise_for_status()

    body = resp.json()
    message = body["choices"][0]["message"]
    finish_reason = body["choices"][0].get("finish_reason", "")

    content = message.get("content", "").strip()
    reasoning = message.get("reasoning_content", "").strip()

    logger.debug("finish_reason=%s content=%r reasoning_len=%d", finish_reason, content, len(reasoning))

    if finish_reason == "length":
        raise RuntimeError("모델이 토큰 한도 내에 응답을 완성하지 못했습니다.")

    search_targets = [t for t in [content, reasoning] if t]
    for target in search_targets:
        cleaned = re.sub(r"<think>.*?</think>", "", target, flags=re.DOTALL).strip()
        match = re.search(r"\{[^{}]+\}", cleaned)
        if match:
            return json.loads(match.group())
        match = re.search(r"\{[^{}]+\}", target)
        if match:
            return json.loads(match.group())

    raise ValueError(f"JSON을 찾을 수 없습니다. content={content!r}")


async def call_llm(max_retries: int = 5) -> dict:
    banned = used_words(HISTORY)
    last_exc: Exception = RuntimeError("알 수 없는 오류")
    for attempt in range(1, max_retries + 1):
        try:
            result = await asyncio.wait_for(_call_llm_once(banned), timeout=ATTEMPT_TIMEOUT)

            challenge = str(result.get("문어", "")).strip()
            answer = str(result.get("답어", "")).strip()
            if not challenge or not answer:
                raise ValueError(f"문어/답어가 비어 있습니다: {result!r}")

            duplicates = [w for w in (challenge, answer) if w in banned]
            if duplicates:
                raise ValueError(f"이미 사용된 단어: {', '.join(duplicates)}")

            entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "문어": challenge,
                "답어": answer,
            }
            HISTORY.append(entry)
            save_history(HISTORY)
            return entry
        except asyncio.TimeoutError:
            last_exc = RuntimeError("2분 내 응답 없음")
            logger.warning("암구호 생성 실패 (시도 %d/%d): 2분 초과", attempt, max_retries)
        except Exception as e:
            last_exc = e
            logger.warning("암구호 생성 실패 (시도 %d/%d): %s", attempt, max_retries, e)
    raise last_exc


def build_embed(result: dict) -> discord.Embed:
    today = datetime.now().strftime("%Y년 %m월 %d일")
    embed = discord.Embed(
        title=f"암구호  |  {today}",
        color=discord.Color.from_rgb(0, 80, 40),
    )
    embed.add_field(
        name="문어 (도전)",
        value=f"```\n{result.get('문어', '?')}\n```",
        inline=True,
    )
    embed.add_field(
        name="답어 (응답)",
        value=f"```\n{result.get('답어', '?')}\n```",
        inline=True,
    )
    embed.set_footer(text="금일 암구호 — 보안 유지")
    return embed


@tasks.loop(hours=24)
async def daily_task():
    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        logger.error("채널 %d를 찾을 수 없습니다.", CHANNEL_ID)
        return
    try:
        result = await call_llm()
        await channel.send(embed=build_embed(result))
        logger.info("암구호 전송 완료: %s", result)
    except Exception:
        logger.exception("암구호 생성 실패")
        await channel.send("⚠️ 오늘의 암구호 생성에 실패했습니다.")


@daily_task.before_loop
async def before_daily():
    await bot.wait_until_ready()
    now = datetime.now()
    target = now.replace(hour=SEND_HOUR, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    delay = (target - now).total_seconds()
    logger.info("다음 전송 예정: %s (%.0f초 후)", target.strftime("%Y-%m-%d %H:%M"), delay)
    await asyncio.sleep(delay)


@bot.event
async def on_ready():
    logger.info("봇 로그인: %s", bot.user)
    await bot.tree.sync()
    logger.info("슬래시 커맨드 동기화 완료")
    if not daily_task.is_running():
        daily_task.start()


@bot.tree.command(name="테스트", description="암구호를 즉시 생성합니다")
async def slash_test(interaction: discord.Interaction):
    channel = interaction.channel
    await interaction.response.send_message("⏳ 암구호 생성 중...")
    try:
        result = await call_llm()
        await channel.send(embed=build_embed(result))
        logger.info("수동 암구호 전송 완료: %s", result)
        try:
            await interaction.edit_original_response(content="✅ 생성 완료")
        except discord.errors.HTTPException:
            pass
    except Exception:
        logger.exception("수동 암구호 생성 실패")
        try:
            await interaction.edit_original_response(content="⚠️ 암구호 생성에 실패했습니다.")
        except discord.errors.HTTPException:
            await channel.send("⚠️ 암구호 생성에 실패했습니다.")


bot.run(DISCORD_TOKEN)
