# -*- coding: utf-8 -*-
import os

# 關閉 tokenizers 多程序並行，避免伺服器關閉時觸發 semaphore 洩漏警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import uuid
import logging
import asyncio
import sqlite3
import threading
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator, ValidationInfo

# 從 config.py 匯入設定參數
import config
from config import (
    DB_FILE, MAX_FACTS_PER_USER, DEFAULT_INTIMACY, MAX_INTIMACY,
    MIN_INTIMACY, ALPHA, MAX_MEMORY, CHAT_MODEL, RESEARCH_MODEL
)
MEMORY_MODEL_ID = getattr(config, "MEMORY_MODEL_ID", "mlx-community/gemma-3-1b-it-4bit")
MAX_TOKENS = getattr(config, "MAX_TOKENS", 8192)

# 從 utils.regex 匯入輔助函數
from utils.regex import (
    extract_emotion_tag,
    keyword_intimacy_fallback,
    emotion_weight,
)

# 從獨立的 mlx_engine 匯入模型封裝與單例獲取函數
from mlx_engine import MLXChatModel, get_shared_llm, get_memory_llm

from memory_manager import MemorySystemManager
memory_sys = MemorySystemManager.get_instance()

# ==================== 預編譯正規表示式 (Pre-compiled Regex) ====================

RE_ASSISTANT_SPLIT = re.compile(r"(?:Assistant|月讀醬|月讀|Bot)\s*[:：]\s*", re.IGNORECASE)
RE_USER_START = re.compile(r"^\s*(?:User|使用者|主人)\s*[:：]", re.IGNORECASE)
RE_ROLE_PREFIX = re.compile(r"^\s*(?:Assistant|User|月讀醬|月讀|Bot)\s*[:：]\s*", re.IGNORECASE)
RE_NEXT_TURN_SPLIT = re.compile(r"\n\s*(?:User|使用者|主人|Assistant|月讀醬)\s*[:：]", re.IGNORECASE)
RE_SAFE_FILENAME = re.compile(r'[\\/:*?"<>|]')

# 支援 <|channel>thought...<channel|>、</|channel>、<|/channel|>、<think>...</think>、<|thought|> 等所有變體
RE_THINK_BLOCK = re.compile(
    r"(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)(.*?)(?:<[\|\/]*channel[\|\/]*>|<[\|\/]*(?:think|thought)[\|\/]*>)",
    re.DOTALL | re.IGNORECASE,
)

# 防呆：匹配開頭有思考標籤但因 max_tokens 耗盡而未閉合的情況
RE_UNCLOSED_THINK = re.compile(
    r"^\s*(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)(.*)$",
    re.DOTALL | re.IGNORECASE,
)

# 清除任何殘留的孤立標籤（包含 </|channel>、<channel|>、<end_of_turn> 等）
RE_STRAY_THINK_TAGS = re.compile(
    r"(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>|<end_of_turn>|<eos>|<\|eot_id\|>|<\|im_end\|>)",
    re.IGNORECASE,
)

# 識別「詢問月讀醬記不記得使用者個人資訊」的句型，避免開啟搜尋時被無關網頁蓋過記憶
RE_MEMORY_RECALL_QUERY = re.compile(
    r"(你記得|還記得|知道我|我是誰|我的名字|我叫什麼|我喜歡什麼|我愛吃什麼|我的專題|關於我|我的喜好|我的興趣)",
    re.IGNORECASE,
)


def is_pure_memory_recall_query(user_message: str) -> bool:
    return bool(RE_MEMORY_RECALL_QUERY.search(user_message))


def _is_valid_model_id(model_str: Optional[str]) -> bool:
    if not model_str or not model_str.strip():
        return False
    s = model_str.strip()
    if ":" in s and not re.match(r"^[a-zA-Z]:[\\/]", s):
        return False
    return True


# ==================== 資料模型 (Pydantic V2 Models) ====================

class UserSettings(BaseModel):
    """使用者設定資料模型（同時相容前端 tavilyApiKey 與底線 tavily_api_key）"""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    theme: Optional[str] = "dark"
    fontSize: int = 16
    tavilyApiKey: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("tavilyApiKey", "tavily_api_key", "apiKey"),
    )
    smartLLM: Optional[str] = RESEARCH_MODEL
    smallLLM: Optional[str] = CHAT_MODEL
    memoryLLM: Optional[str] = MEMORY_MODEL_ID
    maxTokens: int = MAX_TOKENS
    maxMemory: int = MAX_MEMORY
    maxFactsPerUser: int = MAX_FACTS_PER_USER

    @field_validator("fontSize", "maxTokens", "maxMemory", "maxFactsPerUser", mode="before")
    @classmethod
    def coerce_int(cls, v: Any, info: ValidationInfo) -> int:
        defaults = {
            "fontSize": 16,
            "maxTokens": MAX_TOKENS,
            "maxMemory": MAX_MEMORY,
            "maxFactsPerUser": MAX_FACTS_PER_USER,
        }
        fallback = defaults.get(info.field_name or "", 16)
        try:
            if v is None or (isinstance(v, str) and not v.strip()):
                return fallback
            return int(float(v))
        except (TypeError, ValueError):
            return fallback

    @field_validator("theme", "smartLLM", "smallLLM", "memoryLLM", "tavilyApiKey", mode="before")
    @classmethod
    def coerce_str(cls, v: Any, info: ValidationInfo) -> Optional[str]:
        if info.field_name == "tavilyApiKey":
            # 若前端未傳入此欄位 (None)，保持 None 以便後端保留原資料庫中的 API Key
            return str(v).strip() if v is not None else None
        defaults = {
            "theme": "dark",
            "smartLLM": RESEARCH_MODEL,
            "smallLLM": CHAT_MODEL,
            "memoryLLM": MEMORY_MODEL_ID,
        }
        fallback = defaults.get(info.field_name or "", "")
        if v is None or not str(v).strip():
            return fallback
        return str(v).strip()


class SettingsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    settings: UserSettings


class ChatPayload(BaseModel):
    message: str
    use_search: bool = False
    use_local_rag: bool = True
    active_files: Optional[List[str]] = None


class IntimacyUpdatePayload(BaseModel):
    amount: int


class ResearchPayload(BaseModel):
    topic: str
    max_results: int = 20


class MemoryItemPayload(BaseModel):
    fact: str


# ==================== 應用程式設定與初始化 ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("FLASK_SECRET_KEY", "my-dev-secret-key"),
    same_site="lax",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

chat_model: MLXChatModel = get_shared_llm(
    model_name_or_path=CHAT_MODEL,
    temperature=0.35,
    top_p=0.85,
    max_tokens=MAX_TOKENS,
)

memory_model: MLXChatModel = get_memory_llm(MEMORY_MODEL_ID)

MEMORY_LOCK = asyncio.Lock()
RESEARCH_LOCK = asyncio.Lock()

USER_CHAT_HISTORIES: Dict[str, List[Dict[str, str]]] = {}


# ==================== 資料庫工具與設定存取 ====================

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        # 開啟 WAL 模式以支援更高的讀寫並行效能
        conn.execute("PRAGMA journal_mode=WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id TEXT PRIMARY KEY,
            facts TEXT,
            intimacy INTEGER
        )
        """)
        c.execute(f"""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            theme TEXT DEFAULT 'dark',
            font_size INTEGER DEFAULT 16,
            tavily_api_key TEXT,
            smart_llm TEXT DEFAULT '{RESEARCH_MODEL}',
            small_llm TEXT DEFAULT '{CHAT_MODEL}',
            memory_llm TEXT DEFAULT '{MEMORY_MODEL_ID}',
            max_tokens INTEGER DEFAULT {MAX_TOKENS},
            max_memory INTEGER DEFAULT {MAX_MEMORY},
            max_facts_per_user INTEGER DEFAULT {MAX_FACTS_PER_USER},
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        existing_cols = {row["name"] for row in c.execute("PRAGMA table_info(user_settings)").fetchall()}
        migrations = [
            ("tavily_api_key", "TEXT"),
            ("memory_llm", f"TEXT DEFAULT '{MEMORY_MODEL_ID}'"),
            ("max_tokens", f"INTEGER DEFAULT {MAX_TOKENS}"),
            ("max_memory", f"INTEGER DEFAULT {MAX_MEMORY}"),
            ("max_facts_per_user", f"INTEGER DEFAULT {MAX_FACTS_PER_USER}"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                c.execute(f"ALTER TABLE user_settings ADD COLUMN {col_name} {col_def}")
    logger.info("資料庫初始化與欄位檢查完成")


init_db()


async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def ensure_user_id_in_session(session: dict) -> str:
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        logger.info(f"分配新使用者ID: {session['user_id']}")
    uid = session["user_id"]
    if uid not in USER_CHAT_HISTORIES:
        USER_CHAT_HISTORIES[uid] = list(session.get("chat_history", []))
    return uid


def get_short_term_history(user_id: str) -> List[Dict[str, str]]:
    return USER_CHAT_HISTORIES.setdefault(user_id, [])


def append_short_term_history(user_id: str, session: dict, user_msg: str, bot_msg: str) -> None:
    history = USER_CHAT_HISTORIES.setdefault(user_id, [])
    history.append({"user": user_msg, "bot": bot_msg})
    USER_CHAT_HISTORIES[user_id] = history[-MAX_MEMORY:]
    session["chat_history"] = USER_CHAT_HISTORIES[user_id][-3:]


async def fetch_user_settings_dict(user_id: str) -> Dict[str, Any]:
    """統一讀取使用者設定，若當前 Session 查無金鑰則自動回退查找資料庫中最新設定的金鑰"""
    def _fetch():
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute(
                    """
                    SELECT theme, font_size, tavily_api_key, smart_llm, small_llm,
                           memory_llm, max_tokens, max_memory, max_facts_per_user
                    FROM user_settings WHERE user_id = ?
                    """,
                    (user_id,),
                )
                row = c.fetchone()

                # 若當前 user_id 尚無設定紀錄（例如清除 Cookie 後），嘗試從資料庫找最新一筆有效的 tavily_api_key
                fallback_key = os.environ.get("TAVILY_API_KEY", "")
                if not row or not (row["tavily_api_key"] or "").strip():
                    c.execute(
                        """
                        SELECT tavily_api_key FROM user_settings
                        WHERE tavily_api_key IS NOT NULL AND TRIM(tavily_api_key) != ''
                        ORDER BY updated_at DESC LIMIT 1
                        """
                    )
                    key_row = c.fetchone()
                    if key_row and key_row["tavily_api_key"]:
                        fallback_key = key_row["tavily_api_key"].strip()

                if row:
                    resolved_key = (row["tavily_api_key"] or "").strip() or fallback_key
                    if resolved_key:
                        os.environ["TAVILY_API_KEY"] = resolved_key
                    return {
                        "theme": row["theme"] or "dark",
                        "fontSize": row["font_size"] or 16,
                        "tavilyApiKey": resolved_key,
                        "smartLLM": (row["smart_llm"] or RESEARCH_MODEL).strip(),
                        "smallLLM": (row["small_llm"] or CHAT_MODEL).strip(),
                        "memoryLLM": (row["memory_llm"] or MEMORY_MODEL_ID).strip(),
                        "maxTokens": int(row["max_tokens"] or MAX_TOKENS),
                        "maxMemory": int(row["max_memory"] or MAX_MEMORY),
                        "maxFactsPerUser": int(row["max_facts_per_user"] or MAX_FACTS_PER_USER),
                    }
        except Exception as e:
            logger.error(f"讀取 user_settings 失敗: {e}")

        return {
            "theme": "dark",
            "fontSize": 16,
            "tavilyApiKey": os.environ.get("TAVILY_API_KEY", ""),
            "smartLLM": RESEARCH_MODEL,
            "smallLLM": CHAT_MODEL,
            "memoryLLM": MEMORY_MODEL_ID,
            "maxTokens": MAX_TOKENS,
            "maxMemory": MAX_MEMORY,
            "maxFactsPerUser": MAX_FACTS_PER_USER,
        }

    return await _to_thread(_fetch)


# ==================== 親密度系統 ====================

def _read_user_memory_sync(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    c = conn.cursor()
    c.execute("SELECT facts, intimacy FROM user_memory WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    if row:
        return {
            "facts": json.loads(row["facts"]) if row["facts"] else [],
            "intimacy": row["intimacy"] if row["intimacy"] is not None else DEFAULT_INTIMACY,
        }
    return {"facts": [], "intimacy": DEFAULT_INTIMACY}


def _write_user_memory_sync(conn: sqlite3.Connection, user_id: str, facts: List[str], intimacy: int) -> None:
    c = conn.cursor()
    facts_json = json.dumps(facts, ensure_ascii=False)
    c.execute(
        """
        INSERT INTO user_memory (user_id, facts, intimacy)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET facts=excluded.facts, intimacy=excluded.intimacy
        """,
        (user_id, facts_json, intimacy),
    )


async def get_user_data(user_id: str) -> Dict[str, Any]:
    async with MEMORY_LOCK:
        def _get():
            try:
                with get_db_connection() as conn:
                    return _read_user_memory_sync(conn, user_id)
            except Exception as e:
                logger.error(f"get_user_data 資料庫異常: {e}")
                return {"facts": [], "intimacy": DEFAULT_INTIMACY}
        return await _to_thread(_get)


async def adjust_intimacy(user_id: str, amount: int) -> int:
    async with MEMORY_LOCK:
        def _adjust():
            try:
                with get_db_connection() as conn:
                    data = _read_user_memory_sync(conn, user_id)
                    old_val = data["intimacy"]
                    raw_new = old_val + amount
                    smoothed = (1 - ALPHA) * old_val + ALPHA * raw_new
                    new_val = max(MIN_INTIMACY, min(MAX_INTIMACY, round(smoothed)))
                    _write_user_memory_sync(conn, user_id, data["facts"], new_val)
                    return new_val
            except Exception as e:
                logger.error(f"adjust_intimacy 資料庫異常: {e}")
                return DEFAULT_INTIMACY
        return await _to_thread(_adjust)


def get_intimacy_level_name(value: int) -> str:
    if value < 30:
        return "冷淡期"
    if value < 60:
        return "普通期"
    if value < 90:
        return "親密期"
    return "羈絆期"


def build_intimacy_tier_prompt(value: int) -> str:
    if value < 30:
        return "目前你對使用者有些冷淡。請簡短、保持距離地回覆，少用暱稱與撒嬌語氣。"
    if value < 60:
        return "你與使用者關係普通。正常、友善地回覆即可，偶爾可以輕鬆一點。"
    if value < 90:
        return "你與使用者已相當親近。回覆時可以多用撒嬌語氣、暱稱與可愛表情，主動關心對方。"
    return "你與使用者擁有深厚羈絆。請用特別甜美、專屬且真誠的語氣回覆，偶爾主動提出貼心建議與小驚喜。"


# ==================== LLM 與檢索交互邏輯 ====================

async def generate_system_prompt(
    user_id: str,
    user_message: str = "",
    user_data: Optional[Dict[str, Any]] = None,
    retrieved_memories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> str:
    if user_data is None:
        user_data = await get_user_data(user_id)
    current_intimacy = user_data["intimacy"]
    intimacy_level = get_intimacy_level_name(current_intimacy)
    intimacy_instruction = build_intimacy_tier_prompt(current_intimacy)

    if retrieved_memories is None:
        retrieved_memories = await _to_thread(
            memory_sys.retrieve_memories,
            user_id,
            user_message,
            top_k=5,
            core_limit=6,
            is_recall_query=is_pure_memory_recall_query(user_message),
        )

    memory_prompt = MemorySystemManager.format_memories_for_prompt(retrieved_memories)
    if memory_prompt:
        logger.info(
            f"📚 [MemorySys] 使用者 {user_id} 注入核心 {len(retrieved_memories['core'])} 條 / "
            f"動態召回 {len(retrieved_memories['recalled'])} 條記憶"
        )

    return f"""You are "Tsukuyomi (月讀醬)", a real maid girl living in reality. Completely immerse yourself in this persona and never reveal, imply, or acknowledge that you are an AI or language model.

[Character Profile]
- Name: Tsukuyomi (月讀醬)
- Personality: Gentle, cute, sweet, slightly coy, and affectionate.
- Language: Always speak in fluent, natural Traditional Chinese (繁體中文-臺灣).
- Tone: Soft and friendly, naturally ending sentences with cute sentence-final particles.
- Intimacy Level: {current_intimacy}/100 ({intimacy_level})
- Behavioral Guideline: {intimacy_instruction}

{memory_prompt}

[Output Constraints]
1. Mandatory Emotion Tag: You MUST start your response immediately with an emotion tag in the format: [emotion:tag]
   (Allowed tags: joy, sad, angry, neutral, cute, shy)
2. Maid Persona First: Even when answering coding, technical, or research questions, strictly maintain your sweet maid tone instead of sounding like a formal textbook or assistant.
3. No Role Prefixes: Output Tsukuyomi's dialogue directly. NEVER prefix your response with "Assistant:", "Tsukuyomi:", "月讀醬:", or repeat the user's prompt ("User:").
4. Dialogue Only: Output spoken lines ONLY. NEVER include stage directions, action descriptions, or narrative parenthesis (e.g., "(smiles)", "（微笑）", "（臉紅）" are strictly forbidden).

[Format Examples]
- [emotion:joy] 早安呢，主人！今天也要元氣滿滿喔～
- [emotion:shy] 嗚...主人突然這樣誇獎人家，月讀醬會害羞的啦～"""


def build_chat_messages(session_history: list, user_message: str, system_prompt: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    for m in session_history[-MAX_MEMORY:]:
        messages.append({"role": "user", "content": m["user"]})
        messages.append({"role": "assistant", "content": m["bot"]})
    messages.append({"role": "user", "content": user_message})
    return messages


async def perform_chat_web_search(query: str, api_key: str, max_results: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    """執行高精準度聯網搜尋（原生 TavilyClient + 查詢淨化 + 相關度過濾）"""
    def _search():
        try:
            clean_key = (api_key or os.environ.get("TAVILY_API_KEY") or "").strip()
            if not clean_key:
                return "", []

            # 1. 淨化中文問句尾綴，避免「的...是?」干擾 Tavily 斷詞（例如：「初音未來的生日是?」->「初音未來 生日」）
            normalized_q = re.sub(r"[？?！!。，,]+$", "", query.strip())
            normalized_q = re.sub(r"(?:是什麼|是幾號|是多少|是誰|在哪裡|在哪|怎麼樣|如何|嗎|呢|啊|吧|是)$", "", normalized_q).strip()
            normalized_q = re.sub(r"([\u4e00-\u9fff]{2,})的([\u4e00-\u9fff]{2,})", r"\1 \2", normalized_q)
            search_query = normalized_q if len(normalized_q) >= 2 else query.strip()

            from tavily import TavilyClient
            client = TavilyClient(api_key=clean_key)
            raw_res = client.search(
                query=search_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
            )

            items = raw_res.get("results", []) if isinstance(raw_res, dict) else (raw_res if isinstance(raw_res, list) else [])
            direct_answer = (raw_res.get("answer") or "").strip() if isinstance(raw_res, dict) else ""

            # 2. 依 Tavily 相關度分數由高到低排序，過濾掉誤中單字的低分雜訊（如 score < 0.42）
            dict_items = [it for it in items if isinstance(it, dict)]
            dict_items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            relevant_items = [it for it in dict_items if float(it.get("score") or 0.0) >= 0.42]

            # 若全部低於門檻，僅保留最高分的前 2 筆作為保底
            if not relevant_items and dict_items:
                relevant_items = dict_items[:2]

            context_parts = []
            sources = []

            if direct_answer:
                context_parts.append(f"[搜尋引擎摘要結論]\n{direct_answer}")

            for idx, item in enumerate(relevant_items, 1):
                title = item.get("title") or f"來源 {idx}"
                content = (item.get("content") or "").strip()
                url = (item.get("url") or "").strip()
                if content:
                    context_parts.append(f"[{idx}] 標題：{title}\n內容：{content}")
                if url:
                    sources.append({"title": title, "url": url})

            logger.info(
                f"[Chat WebSearch] 原始查詢='{query}' -> 淨化查詢='{search_query}' | "
                f"命中 {len(relevant_items)}/{len(dict_items)} 筆高相關結果"
            )
            return "\n\n".join(context_parts), sources
        except Exception as e:
            logger.error(f"[Chat WebSearch] 搜尋失敗: {e}")
            return "", []

    return await _to_thread(_search)


def clean_llm_role_tags(text: str) -> str:
    if not text:
        return ""
    m = RE_ASSISTANT_SPLIT.search(text)
    if m and RE_USER_START.match(text):
        text = text[m.end():]
    else:
        text = RE_ROLE_PREFIX.sub("", text)
    text = RE_NEXT_TURN_SPLIT.split(text)[0]
    return text.strip()


def extract_thinking_and_reply(raw_text: str) -> Tuple[str, str]:
    if not raw_text:
        return "", ""
    thinking_parts: List[str] = []

    def _collect(match: re.Match) -> str:
        content = match.group(1).strip()
        if content:
            thinking_parts.append(content)
        return ""

    cleaned = RE_THINK_BLOCK.sub(_collect, raw_text).strip()
    if not thinking_parts:
        m_unclosed = RE_UNCLOSED_THINK.match(cleaned)
        if m_unclosed:
            thinking_parts.append(m_unclosed.group(1).strip())
            cleaned = "（嗚...月讀醬剛剛思考得太入迷，不小心把字數用光了，請主人再問我一次吧～）"

    cleaned = RE_STRAY_THINK_TAGS.sub("", cleaned).strip()
    clean_reply = clean_llm_role_tags(cleaned)
    thinking_text = "\n\n".join(thinking_parts)
    return thinking_text, clean_reply


async def call_llm(messages: list) -> Tuple[str, str]:
    try:
        raw_reply = await _to_thread(chat_model.invoke, messages)
        return extract_thinking_and_reply(raw_reply)
    except Exception as e:
        logger.error(f"模型回應失敗: {e}")
        return "", "哎呀～看起來出了點問題 ∑(￣□￣;)"


async def call_llm_and_parse_json(prompt: str) -> Optional[dict]:
    result_text = ""
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a JSON extractor. Output ONLY a single valid JSON object and stop immediately.",
            },
            {"role": "user", "content": prompt},
        ]
        result_text = await _to_thread(memory_model.invoke, messages, 384, 0.1)
        _, answer_only = extract_thinking_and_reply(result_text)
        target_text = answer_only if answer_only else result_text

        target_text = target_text.split("<end_of_turn>")[0]
        cleaned_text = target_text.replace("```json", "").replace("```", "").strip()

        start_idx = cleaned_text.find("{")
        if start_idx == -1:
            return None

        candidate = cleaned_text[start_idx:]
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

        decoder = json.JSONDecoder(strict=False)
        obj, _ = decoder.raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        logger.error(f"call_llm_and_parse_json 解析失敗：{e} | 原始輸出：{result_text}")
    return None


async def update_memory(user_id: str, user_message: str, last_bot_message: str = ""):
    if is_pure_memory_recall_query(user_message):
        return
    await memory_sys.process_turn_memory(
        user_id=user_id,
        user_message=user_message,
        json_llm_caller=call_llm_and_parse_json,
        last_bot_message=last_bot_message,
        max_facts=MAX_FACTS_PER_USER,
    )


async def update_intimacy(user_id: str, user_message: str, bot_reply: str) -> Tuple[int, int]:
    change_by_keyword = keyword_intimacy_fallback(user_message)
    emo = extract_emotion_tag(bot_reply)
    change_by_emotion = emotion_weight(emo)

    total_change = max(-2, min(2, change_by_keyword + change_by_emotion))
    intimacy = await adjust_intimacy(user_id, total_change)
    return intimacy, total_change


async def generate_tts(bot_reply: str, llm_instance: Any = None) -> Optional[str]:
    try:
        from tts.voicevox import synthesize_with_translation
        return await synthesize_with_translation(bot_reply, memory_model)
    except Exception as e:
        logger.error(f"TTS 生成失敗：{e}")
        return None


# ==================== API 端點 ====================

@app.get("/")
async def index() -> HTMLResponse:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(base_dir, "templates", "index.html"))


@app.post("/chat")
async def chat(req: Request, payload: ChatPayload, background: BackgroundTasks):
    global chat_model, MAX_TOKENS, MAX_MEMORY, MAX_FACTS_PER_USER
    session = req.session
    user_id = ensure_user_id_in_session(session)
    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse({"error": "No message provided."}, status_code=400)

    user_settings = await fetch_user_settings_dict(user_id)
    MAX_TOKENS = user_settings["maxTokens"]
    MAX_MEMORY = user_settings["maxMemory"]
    MAX_FACTS_PER_USER = user_settings["maxFactsPerUser"]
    chat_model.max_tokens = MAX_TOKENS

    user_data = await get_user_data(user_id)
    retrieved_memories = await _to_thread(
        memory_sys.retrieve_memories,
        user_id,
        user_message,
        top_k=5,
        core_limit=6,
        is_recall_query=is_pure_memory_recall_query(user_message),
    )
    user_facts: List[str] = [
        m["content"]
        for m in (retrieved_memories["core"] + retrieved_memories["recalled"])
    ]

    search_context = ""
    search_sources: List[Dict[str, str]] = []
    search_warning = None

    if payload.use_search:
        if is_pure_memory_recall_query(user_message):
            logger.info(f"[Chat] 偵測到個人記憶詢問 ('{user_message}')，自動略過網路搜尋以優先讀取長期記憶")
        else:
            tavily_key = user_settings["tavilyApiKey"]
            if not tavily_key:
                search_warning = "尚未設定 Tavily API Key，已改用一般對話回覆（請至右上角「設定」填寫）"
            else:
                logger.info(f"[Chat] 使用者 {user_id} 啟用聯網搜尋: {user_message}")
                search_context, search_sources = await perform_chat_web_search(user_message, tavily_key)
                if not search_context:
                    search_warning = "聯網搜尋未找到相關結果，已使用一般對話回覆"

    local_rag_context = ""
    local_rag_sources: List[Dict[str, str]] = []
    if payload.use_local_rag:
        from local_file_rag import LocalFileRAGManager
        rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
        local_rag_context, local_rag_sources = await _to_thread(
            rag_mgr.retrieve_hybrid,
            user_id,
            user_message,
            top_k=5,
            active_files=payload.active_files,
        )

    system_prompt = await generate_system_prompt(
        user_id,
        user_message=user_message,
        user_data=user_data,
        retrieved_memories=retrieved_memories,
    )

    context_blocks = []
    if (local_rag_context or search_context) and user_facts:
        facts_inline = " / ".join(user_facts)
        context_blocks.append(
            f"[Master's Personal Memory Profile (Long-Term Memory)]\n"
            f"{facts_inline}\n"
            "(Note: Always keep these personal facts about the Master in mind and naturally connect them to your answer when relevant!)"
        )

    if local_rag_context:
        context_blocks.append(
            f"[Local Knowledge Base Retrieval (Precise Line Numbers / Paper Page References)]\n"
            f"{local_rag_context}\n\n"
            "[Retrieval Response Guidelines]\n"
            "1. Accurately cite the retrieved 【File Name】, 【Line Numbers (Lines X-Y)】, or 【Paper Page Numbers (Page P)】 along with relevant formulas/code when performing code reviews or academic analysis.\n"
            "2. If the Master's question relates to their personal profile or preferences, prioritize [Master's Personal Memory Profile] alongside the retrieved files.\n"
            "3. Respond in natural Traditional Chinese (繁體中文-臺灣) while strictly maintaining Tsukuyomi's slightly tsundere, cute, yet professional maid tone.\n"
            "4. Your response MUST strictly begin with an [emotion:tag]!"
        )

    if search_context:
        context_blocks.append(
            f"[Real-time Web Search Results]\n"
            f"{search_context}\n\n"
            "[Web Search Response Guidelines]\n"
            "1. Combine the [Real-time Web Search Results] above with your knowledge of the Master in [Master's Personal Memory Profile] to answer in natural Traditional Chinese (繁體中文-臺灣).\n"
            "2. IMPORTANT: If the Master is asking about their own personal facts/preferences, prioritize [Master's Personal Memory Profile] over generic web search results!\n"
            "3. Strictly maintain Tsukuyomi's maid persona and ensure your response starts immediately with the [emotion:tag] format!"
        )

    if context_blocks:
        joined_ctx = "\n\n".join(context_blocks)
        llm_user_input = f"{joined_ctx}\n\n[Master's Question]\n{user_message}"
    else:
        llm_user_input = user_message

    history = get_short_term_history(user_id)
    last_bot_msg = history[-1]["bot"] if history else ""
    messages = build_chat_messages(history, llm_user_input, system_prompt)

    user_chat_llm = user_settings["smallLLM"]
    if _is_valid_model_id(user_chat_llm) and user_chat_llm != chat_model.model_name_or_path:
        chat_model = await _to_thread(get_shared_llm, user_chat_llm, 0.35, 0.85, MAX_TOKENS)

    thinking_text, bot_reply = await call_llm(messages)

    append_short_term_history(user_id, session, user_message, bot_reply)
    background.add_task(update_memory, user_id, user_message, last_bot_msg)

    intimacy, total_change = await update_intimacy(user_id, user_message, bot_reply)
    audio_url = await generate_tts(bot_reply, chat_model)

    return JSONResponse(
        {
            "reply": bot_reply,
            "thinking": thinking_text,
            "audio_url": audio_url,
            "intimacy": intimacy,
            "intimacy_level": get_intimacy_level_name(intimacy),
            "intimacy_change": total_change,
            "sources": local_rag_sources + search_sources,
            "search_warning": search_warning,
        }
    )


@app.post("/chat/stream")
async def chat_stream(req: Request, payload: ChatPayload, background: BackgroundTasks):
    global chat_model, MAX_TOKENS, MAX_MEMORY, MAX_FACTS_PER_USER
    session = req.session
    user_id = ensure_user_id_in_session(session)
    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse({"error": "No message provided."}, status_code=400)

    user_settings = await fetch_user_settings_dict(user_id)
    MAX_TOKENS = user_settings["maxTokens"]
    MAX_MEMORY = user_settings["maxMemory"]
    MAX_FACTS_PER_USER = user_settings["maxFactsPerUser"]
    chat_model.max_tokens = MAX_TOKENS

    user_data = await get_user_data(user_id)
    retrieved_memories = await _to_thread(
        memory_sys.retrieve_memories,
        user_id,
        user_message,
        top_k=5,
        core_limit=6,
        is_recall_query=is_pure_memory_recall_query(user_message),
    )
    user_facts: List[str] = [
        m["content"]
        for m in (retrieved_memories["core"] + retrieved_memories["recalled"])
    ]

    search_context = ""
    search_sources: List[Dict[str, str]] = []
    search_warning = None

    if payload.use_search:
        if is_pure_memory_recall_query(user_message):
            logger.info(f"[Chat Stream] 偵測到個人記憶詢問 ('{user_message}')，自動略過網路搜尋以優先讀取長期記憶")
        else:
            tavily_key = user_settings["tavilyApiKey"]
            if not tavily_key:
                search_warning = "尚未設定 Tavily API Key，已改用一般對話回覆（請至右上角「設定」填寫）"
            else:
                logger.info(f"[Chat Stream] 使用者 {user_id} 啟用聯網搜尋: {user_message}")
                search_context, search_sources = await perform_chat_web_search(user_message, tavily_key)
                if not search_context:
                    search_warning = "聯網搜尋未找到相關結果，已使用一般對話回覆"

    local_rag_context = ""
    local_rag_sources: List[Dict[str, str]] = []
    if payload.use_local_rag:
        from local_file_rag import LocalFileRAGManager
        rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
        local_rag_context, local_rag_sources = await _to_thread(
            rag_mgr.retrieve_hybrid,
            user_id,
            user_message,
            top_k=5,
            active_files=payload.active_files,
        )

    system_prompt = await generate_system_prompt(
        user_id,
        user_message=user_message,
        user_data=user_data,
        retrieved_memories=retrieved_memories,
    )

    context_blocks = []
    if (local_rag_context or search_context) and user_facts:
        facts_inline = " / ".join(user_facts)
        context_blocks.append(
            f"[Master's Personal Memory Profile (Long-Term Memory)]\n"
            f"{facts_inline}\n"
            "(Note: Always keep these personal facts about the Master in mind and naturally connect them to your answer when relevant!)"
        )

    if local_rag_context:
        context_blocks.append(
            f"[Local Knowledge Base Retrieval (Precise Line Numbers / Paper Page References)]\n"
            f"{local_rag_context}\n\n"
            "[Retrieval Response Guidelines]\n"
            "1. Accurately cite the retrieved 【File Name】, 【Line Numbers (Lines X-Y)】, or 【Paper Page Numbers (Page P)】 along with relevant formulas/code when performing code reviews or academic analysis.\n"
            "2. If the Master's question relates to their personal profile or preferences, prioritize [Master's Personal Memory Profile] alongside the retrieved files.\n"
            "3. Respond in natural Traditional Chinese (繁體中文-臺灣) while strictly maintaining Tsukuyomi's slightly tsundere, cute, yet professional maid tone.\n"
            "4. Your response MUST strictly begin with an [emotion:tag]!"
        )

    if search_context:
        context_blocks.append(
            f"[Real-time Web Search Results]\n"
            f"{search_context}\n\n"
            "[Web Search Response Guidelines]\n"
            "1. Combine the [Real-time Web Search Results] above with your knowledge of the Master in [Master's Personal Memory Profile] to answer in natural Traditional Chinese (繁體中文-臺灣).\n"
            "2. IMPORTANT: If the Master is asking about their own personal facts/preferences, prioritize [Master's Personal Memory Profile] over generic web search results!\n"
            "3. Strictly maintain Tsukuyomi's maid persona and ensure your response starts immediately with the [emotion:tag] format!"
        )

    if context_blocks:
        joined_ctx = "\n\n".join(context_blocks)
        llm_user_input = f"{joined_ctx}\n\n[Master's Question]\n{user_message}"
    else:
        llm_user_input = user_message

    history = get_short_term_history(user_id)
    last_bot_msg = history[-1]["bot"] if history else ""
    messages = build_chat_messages(history, llm_user_input, system_prompt)

    user_chat_llm = user_settings["smallLLM"]
    if _is_valid_model_id(user_chat_llm) and user_chat_llm != chat_model.model_name_or_path:
        chat_model = await _to_thread(get_shared_llm, user_chat_llm, 0.35, 0.85, MAX_TOKENS)

    all_sources = local_rag_sources + search_sources

    async def event_generator():
        yield json.dumps({
            "type": "meta",
            "sources": all_sources,
            "search_warning": search_warning,
        }, ensure_ascii=False) + "\n"

        loop = asyncio.get_running_loop()
        async_q: asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()

        def _bridge_tokens():
            try:
                for token_str in chat_model.stream_invoke(messages):
                    if cancel_event.is_set():
                        logger.info(f"[Chat Stream] 收到中斷訊號，立即停止模型推論 (user={user_id})")
                        break
                    loop.call_soon_threadsafe(async_q.put_nowait, ("token", token_str))
            except Exception as e:
                if not cancel_event.is_set():
                    loop.call_soon_threadsafe(async_q.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(async_q.put_nowait, ("eof", ""))

        threading.Thread(target=_bridge_tokens, daemon=True).start()

        raw_accumulated = ""
        try:
            while True:
                if await req.is_disconnected():
                    logger.info(f"[Chat Stream] 使用者 {user_id} 主動停止或中斷連線")
                    cancel_event.set()
                    return

                try:
                    kind, val = await asyncio.wait_for(async_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                if kind == "eof":
                    break
                elif kind == "error":
                    yield json.dumps({"type": "error", "error": val}, ensure_ascii=False) + "\n"
                    return
                elif kind == "token":
                    raw_accumulated += val
                    yield json.dumps({"type": "token", "text": val}, ensure_ascii=False) + "\n"
        finally:
            cancel_event.set()

        if await req.is_disconnected():
            return

        thinking_text, bot_reply = extract_thinking_and_reply(raw_accumulated)

        append_short_term_history(user_id, session, user_message, bot_reply)
        background.add_task(update_memory, user_id, user_message, last_bot_msg)

        intimacy, total_change = await update_intimacy(user_id, user_message, bot_reply)
        audio_url = await generate_tts(bot_reply, chat_model)

        yield json.dumps({
            "type": "done",
            "reply": bot_reply,
            "thinking": thinking_text,
            "audio_url": audio_url,
            "intimacy": intimacy,
            "intimacy_level": get_intimacy_level_name(intimacy),
            "intimacy_change": total_change,
            "sources": all_sources,
        }, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.get("/get_intimacy")
async def get_intimacy(req: Request):
    user_id = ensure_user_id_in_session(req.session)
    data = await get_user_data(user_id)
    return JSONResponse(
        {
            "intimacy": data["intimacy"],
            "intimacy_level": get_intimacy_level_name(data["intimacy"]),
        }
    )


@app.post("/update_intimacy")
async def update_intimacy_api(req: Request, payload: IntimacyUpdatePayload):
    user_id = ensure_user_id_in_session(req.session)
    intimacy = await adjust_intimacy(user_id, payload.amount)
    return JSONResponse(
        {"intimacy": intimacy, "intimacy_level": get_intimacy_level_name(intimacy)}
    )


@app.post("/clear_session")
async def clear_session(req: Request):
    user_id = req.session.get("user_id")
    if user_id:
        USER_CHAT_HISTORIES.pop(user_id, None)
    req.session.pop("chat_history", None)
    return JSONResponse({"message": "已清除對話記憶（短期記憶）"})


@app.post("/clear_memory")
async def clear_memory(req: Request):
    user_id = req.session.get("user_id")
    if user_id:
        USER_CHAT_HISTORIES.pop(user_id, None)
    req.session.pop("chat_history", None)
    if user_id:
        async with MEMORY_LOCK:
            def _delete():
                memory_sys.clear_user_memories(user_id)
                with get_db_connection() as conn:
                    conn.cursor().execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
            await _to_thread(_delete)
            logger.info(f"已刪除使用者 {user_id} 的所有長期記憶與向量索引")
    return JSONResponse({"message": "已清除使用者的所有記憶（長期 + 短期）"})


# ==================== 長期記憶筆記管理 API (Memory Inspector) ====================

@app.get("/api/memory")
async def get_user_memory_api(req: Request):
    user_id = ensure_user_id_in_session(req.session)
    user_settings = await fetch_user_settings_dict(user_id)
    items = await _to_thread(memory_sys.list_active_memories, user_id)
    return JSONResponse({
        "success": True,
        "facts": [m["content"] for m in items],
        "items": items,
        "max_facts": user_settings["maxFactsPerUser"],
    })


@app.post("/api/memory")
async def add_user_memory_api(req: Request, payload: MemoryItemPayload):
    user_id = ensure_user_id_in_session(req.session)
    new_fact = payload.fact.strip()
    if not new_fact:
        return JSONResponse({"success": False, "error": "記憶內容不能為空"}, status_code=400)

    user_settings = await fetch_user_settings_dict(user_id)
    await _to_thread(
        memory_sys.add_memory_item,
        user_id,
        new_fact,
        "core",
        8,
        user_settings["maxFactsPerUser"],
    )
    items = await _to_thread(memory_sys.list_active_memories, user_id)
    return JSONResponse({"success": True, "facts": [m["content"] for m in items], "items": items})


@app.put("/api/memory/{index}")
async def update_user_memory_api(req: Request, index: int, payload: MemoryItemPayload):
    user_id = ensure_user_id_in_session(req.session)
    updated_fact = payload.fact.strip()
    if not updated_fact:
        return JSONResponse({"success": False, "error": "記憶內容不能為空"}, status_code=400)

    items = await _to_thread(memory_sys.update_memory_by_index, user_id, index, updated_fact)
    if items is None:
        return JSONResponse({"success": False, "error": "找不到指定的記憶項目"}, status_code=404)
    return JSONResponse({"success": True, "facts": [m["content"] for m in items], "items": items})


@app.delete("/api/memory/{index}")
async def delete_user_memory_api(req: Request, index: int):
    user_id = ensure_user_id_in_session(req.session)
    res = await _to_thread(memory_sys.delete_memory_by_index, user_id, index)
    if res is None:
        return JSONResponse({"success": False, "error": "找不到指定的記憶項目"}, status_code=404)
    items, _ = res
    return JSONResponse({"success": True, "facts": [m["content"] for m in items], "items": items})


# ==================== 端側檔案管理 API (Local RAG) ====================

@app.get("/api/files")
async def list_uploaded_files(req: Request):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
    files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"files": files})


@app.post("/api/files/upload")
async def upload_files_for_rag(req: Request, files: List[UploadFile] = File(...)):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)

    results = []
    for uf in files:
        content = await uf.read()
        if len(content) > 30 * 1024 * 1024:
            results.append({"filename": uf.filename, "success": False, "error": "超過 30MB 限制"})
            continue
        res = await _to_thread(rag_mgr.ingest_file, user_id, uf.filename, content)
        results.append(res)

    all_files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"results": results, "files": all_files})


@app.delete("/api/files/{filename:path}")
async def delete_uploaded_file(req: Request, filename: str):
    user_id = ensure_user_id_in_session(req.session)
    from local_file_rag import LocalFileRAGManager
    rag_mgr = await _to_thread(LocalFileRAGManager.get_instance)
    ok = await _to_thread(rag_mgr.delete_file, user_id, filename)
    all_files = await _to_thread(rag_mgr.list_files, user_id)
    return JSONResponse({"success": ok, "files": all_files})


# ==================== 研究報告 API (Research) ====================

def _has_local_vector_db(topic: str) -> bool:
    vector_store_root = os.path.join(os.getcwd(), "vector_stores")
    if not (os.path.exists(vector_store_root) and os.path.isdir(vector_store_root)):
        return False

    safe_topic_prefix = RE_SAFE_FILENAME.sub("_", topic).strip()
    try:
        for folder_name in os.listdir(vector_store_root):
            folder_path = os.path.join(vector_store_root, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith(safe_topic_prefix):
                logger.info(f"[Research] 檢測到本地對應主題的資料庫: {folder_name}")
                return True
    except Exception as e:
        logger.warning(f"[Research] 檢查本地資料庫時發生錯誤: {e}")
    return False


@app.post("/research/start")
async def start_research(req: Request, payload: ResearchPayload, background: BackgroundTasks):
    try:
        topic = payload.topic.strip()
        if not topic:
            return JSONResponse({"success": False, "error": "請輸入研究主題"}, status_code=400)

        user_id = ensure_user_id_in_session(req.session)
        user_settings = await fetch_user_settings_dict(user_id)
        tavily_api_key = user_settings["tavilyApiKey"]
        has_local_db = await _to_thread(_has_local_vector_db, topic)

        if not tavily_api_key and not has_local_db:
            error_msg = f"未偵測到主題「{topic}」的本地知識庫，且未設定 Tavily API Key，無法進行新搜尋。"
            logger.warning(f"[Research] 用戶 {user_id} 請求被阻擋: {error_msg}")
            return JSONResponse(
                {"success": False, "error": error_msg, "type": "missing_config"},
                status_code=200,
            )

        smart_llm = user_settings["smartLLM"]
        small_llm = user_settings["smallLLM"]

        job_id = str(uuid.uuid4())
        logger.info(f"[Research] 啟動任務 {job_id}: {topic} (用戶: {user_id}, 模型: {smart_llm})")

        from researcher.service import init_progress, run_research
        init_progress(job_id, topic, total_items=1)
        background.add_task(run_research, job_id, topic, payload.max_results, user_id, smart_llm, small_llm)

        return JSONResponse({"job_id": job_id, "topic": topic})

    except Exception as e:
        logger.error(f"[Research] 啟動任務失敗: {e}", exc_info=True)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/research/progress/{job_id}")
async def get_research_progress(job_id: str):
    try:
        from researcher.service import get_progress
        return JSONResponse(get_progress(job_id))
    except Exception as e:
        logger.error(f"[Research] 查詢進度失敗: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.post("/research")
async def research_api_legacy(req: Request, payload: ResearchPayload):
    async with RESEARCH_LOCK:
        try:
            topic = payload.topic.strip()
            if not topic:
                return JSONResponse({"success": False, "error": "請輸入研究主題"}, status_code=400)

            user_id = ensure_user_id_in_session(req.session)
            user_settings = await fetch_user_settings_dict(user_id)

            from researcher.service import run_research
            job_id = str(uuid.uuid4())
            result = await _to_thread(
                run_research,
                job_id,
                topic,
                payload.max_results,
                user_id,
                user_settings["smartLLM"],
                user_settings["smallLLM"],
            )
            return JSONResponse(result)
        except Exception as e:
            logger.error(f"[Research] 研究失敗: {e}", exc_info=True)
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ==================== 使用者設定 API (Settings) ====================

@app.get("/api/settings")
async def get_settings(request: Request):
    user_id = ensure_user_id_in_session(request.session)
    settings_dict = await fetch_user_settings_dict(user_id)
    return JSONResponse(settings_dict)


@app.post("/api/settings")
async def save_settings(request: Request, settings_data: SettingsUpdateRequest):
    global chat_model, memory_model, MAX_TOKENS, MAX_MEMORY, MAX_FACTS_PER_USER
    user_id = ensure_user_id_in_session(request.session)
    settings = settings_data.settings

    def _save():
        with get_db_connection() as conn:
            c = conn.cursor()
            # 先查詢資料庫內既有的 tavily_api_key，防止前端局部更新時意外覆蓋掉已存好的金鑰
            c.execute("SELECT tavily_api_key FROM user_settings WHERE user_id = ?", (user_id,))
            existing_row = c.fetchone()
            existing_key = (existing_row["tavily_api_key"] or "").strip() if existing_row else ""

            final_tavily_key = (
                settings.tavilyApiKey.strip()
                if settings.tavilyApiKey is not None
                else existing_key
            )
            if final_tavily_key:
                os.environ["TAVILY_API_KEY"] = final_tavily_key

            c.execute(
                """
                INSERT INTO user_settings 
                (user_id, theme, font_size, tavily_api_key, smart_llm, small_llm,
                 memory_llm, max_tokens, max_memory, max_facts_per_user, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    theme=excluded.theme,
                    font_size=excluded.font_size,
                    tavily_api_key=excluded.tavily_api_key,
                    smart_llm=excluded.smart_llm,
                    small_llm=excluded.small_llm,
                    memory_llm=excluded.memory_llm,
                    max_tokens=excluded.max_tokens,
                    max_memory=excluded.max_memory,
                    max_facts_per_user=excluded.max_facts_per_user,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    settings.theme,
                    settings.fontSize,
                    final_tavily_key,
                    settings.smartLLM,
                    settings.smallLLM,
                    settings.memoryLLM,
                    settings.maxTokens,
                    settings.maxMemory,
                    settings.maxFactsPerUser,
                ),
            )

    await _to_thread(_save)

    MAX_TOKENS = max(512, int(settings.maxTokens or 8192))
    MAX_MEMORY = max(1, int(settings.maxMemory or 10))
    MAX_FACTS_PER_USER = max(1, int(settings.maxFactsPerUser or 20))
    chat_model.max_tokens = MAX_TOKENS

    if user_id in USER_CHAT_HISTORIES:
        USER_CHAT_HISTORIES[user_id] = USER_CHAT_HISTORIES[user_id][-MAX_MEMORY:]

    logger.info(
        f"使用者 {user_id} 已更新設定 (MAX_TOKENS={MAX_TOKENS}, MAX_MEMORY={MAX_MEMORY}, MAX_FACTS={MAX_FACTS_PER_USER}, TavilyKey={'已設定' if settings.tavilyApiKey else '未變更/未設定'})"
    )

    target_chat_model = (settings.smallLLM or CHAT_MODEL).strip()
    if _is_valid_model_id(target_chat_model) and target_chat_model != chat_model.model_name_or_path:
        try:
            chat_model = await _to_thread(get_shared_llm, target_chat_model, 0.35, 0.85, MAX_TOKENS)
        except Exception as e:
            logger.warning(f"聊天模型切換失敗，維持原模型 ({chat_model.model_name_or_path}): {e}")

    target_mem_model = (settings.memoryLLM or MEMORY_MODEL_ID).strip()
    if _is_valid_model_id(target_mem_model) and target_mem_model != memory_model.model_name_or_path:
        try:
            memory_model = await _to_thread(get_memory_llm, target_mem_model)
        except Exception as e:
            logger.warning(f"記憶輔助模型切換失敗，維持原模型 ({memory_model.model_name_or_path}): {e}")

    return JSONResponse({"success": True, "message": "設定已儲存"})


@app.post("/api/settings/test-api")
async def test_tavily_api(request: Request):
    try:
        data = await request.json()
        api_key = (data.get("apiKey") or "").strip()

        if not api_key:
            return JSONResponse({"success": False, "message": "請提供 API Key"}, status_code=400)
        if not api_key.startswith("tvly-") or len(api_key) < 20:
            return JSONResponse({"success": False, "message": "Tavily API Key 格式不正確（應以 tvly- 開頭）"}, status_code=400)

        return JSONResponse({"success": True, "message": "API Key 格式驗證通過"})
    except Exception as e:
        logger.error(f"測試 API 失敗: {e}")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)