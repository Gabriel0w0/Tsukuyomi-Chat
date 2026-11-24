# -*- coding: utf-8 -*-
import sys
import os
import re
import time
import logging
import threading
import sqlite3
import html
from datetime import datetime
from typing import Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger("research_service")

# ==================== 設定管理 ====================

import config
from config import DB_FILE, CHAT_MODEL, MAX_RAG_CONCURRENCY

RESEARCH_MODEL = getattr(config, "RESEARCH_MODEL", CHAT_MODEL)


def _sanitize_model_id(model_str: Optional[str], fallback: str) -> str:
    """過濾舊版 Ollama 格式 (如 gemma3:4b)，但保留 HuggingFace ID 與 Windows/Unix 路徑"""
    if not model_str or not model_str.strip():
        return fallback
    s = model_str.strip()
    if ":" in s and not re.match(r"^[a-zA-Z]:[\\/]", s):
        logger.warning(f"[Settings] 偵測到舊版 Ollama 模型格式 '{s}'，自動回退為 '{fallback}'")
        return fallback
    return s


def get_user_settings(user_id: str = "default") -> Dict[str, Any]:
    """從資料庫讀取用戶設定（支援 WAL 讀取與跨 Session 金鑰自動回退）"""
    default_smart = RESEARCH_MODEL or CHAT_MODEL or "mlx-community/gemma-3-4b-it-4bit"
    default_small = CHAT_MODEL or "mlx-community/gemma-3-4b-it-4bit"
    env_key = os.environ.get("TAVILY_API_KEY", "").strip()

    try:
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            c = conn.cursor()

            c.execute(
                """
                SELECT tavily_api_key, smart_llm, small_llm
                FROM user_settings 
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = c.fetchone()

            # 若當前 user_id 無金鑰，自動查找資料庫中最新一筆有效的 tavily_api_key
            fallback_db_key = ""
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
                    fallback_db_key = key_row["tavily_api_key"].strip()

            if row:
                resolved_key = (row["tavily_api_key"] or "").strip() or fallback_db_key or env_key
                return {
                    "tavilyApiKey": resolved_key,
                    "smartLLM": _sanitize_model_id(row["smart_llm"], default_smart),
                    "smallLLM": _sanitize_model_id(row["small_llm"], default_small),
                }
            else:
                logger.info(f"[Settings] 用戶 {user_id} 尚無個人設定紀錄，使用預設值與回退金鑰")
                return {
                    "tavilyApiKey": fallback_db_key or env_key,
                    "smartLLM": default_smart,
                    "smallLLM": default_small,
                }
    except Exception as e:
        logger.error(f"[Settings] 讀取設定失敗: {e}")
        return {
            "tavilyApiKey": env_key,
            "smartLLM": default_smart,
            "smallLLM": default_small,
        }


# ==================== 進度追蹤系統 ====================

PROGRESS: Dict[str, Dict[str, Any]] = {}
PROGRESS_LOCK = threading.Lock()


def init_progress(job_id: str, topic: str, total_items: int = 1):
    """初始化進度追蹤"""
    with PROGRESS_LOCK:
        PROGRESS[job_id] = {
            "topic": topic,
            "status": "running",
            "current": 0,
            "total": max(total_items, 1),
            "phase": "初始化",
            "message": "開始建立研究大綱...",
            "result_html": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
        }
    logger.info(f"[Progress] 初始化任務 {job_id}: {topic}")


def update_progress(job_id: str, **kwargs):
    """更新進度（支援任意欄位）"""
    with PROGRESS_LOCK:
        if job_id in PROGRESS:
            PROGRESS[job_id].update(kwargs)
            if "current" in kwargs or "phase" in kwargs:
                current = PROGRESS[job_id].get("current", 0)
                total = PROGRESS[job_id].get("total", 1)
                phase = PROGRESS[job_id].get("phase", "")
                logger.debug(f"[Progress] {job_id}: {current}/{total} - {phase}")


def get_progress(job_id: str) -> Dict[str, Any]:
    """查詢進度"""
    with PROGRESS_LOCK:
        return PROGRESS.get(job_id, {"status": "not_found"}).copy()


def cleanup_progress(job_id: str, delay_seconds: int = 600):
    """延遲清理進度快取，防止長篇 HTML 報告持續佔用記憶體"""
    def _cleanup():
        time.sleep(delay_seconds)
        with PROGRESS_LOCK:
            if job_id in PROGRESS:
                del PROGRESS[job_id]
                logger.info(f"[Progress] 已自動清理任務快取 {job_id}")

    threading.Thread(target=_cleanup, daemon=True).start()


# ==================== 主要研究函數 ====================

def run_research(
    job_id: str,
    topic: str,
    max_results: int = 20,
    user_id: str = "default",
    smart_llm: Optional[str] = None,
    small_llm: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """執行研究並返回結果"""
    try:
        settings = get_user_settings(user_id)

        tavily_api_key = (settings.get("tavilyApiKey") or os.environ.get("TAVILY_API_KEY") or "").strip()
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key

        default_smart = RESEARCH_MODEL or CHAT_MODEL or "mlx-community/gemma-3-4b-it-4bit"
        default_small = CHAT_MODEL or "mlx-community/gemma-3-4b-it-4bit"

        smart_llm = _sanitize_model_id(smart_llm or settings.get("smartLLM"), default_smart)
        small_llm = _sanitize_model_id(small_llm or settings.get("smallLLM"), default_small)

        max_concurrency = max_concurrency or MAX_RAG_CONCURRENCY

        logger.info(
            f"[Research] 使用設定: Smart LLM={smart_llm}, Small LLM={small_llm}, "
            f"Concurrency={max_concurrency}, TavilyKey={'已設定' if tavily_api_key else '未設定'}"
        )

        if not tavily_api_key:
            logger.warning("[Research] ⚠️ 未設定 Tavily API Key，將嘗試僅使用本地向量資料庫")

        from researcher.rag_pipeline import RAGPipeline
        from researcher import parsers
        from researcher.research import (
            generate_content_concurrently,
            get_content_llm,
            OUTLINE_TEMPERATURE,
            CONTENT_TEMPERATURE,
            normalize_markdown_headings,
        )

        # ========== 1. 初始化 RAG 管道 ==========
        update_progress(job_id, phase="初始化", message="正在載入研究模型與連接向量資料庫...")
        logger.info("[Research] 初始化 RAG 管道...")

        shared_llm = get_content_llm(smart_llm)

        pipeline = RAGPipeline(
            query=topic,
            llm_instance=shared_llm,
            max_results=max_results,
            use_existing_db=True,
            tavily_api_key=tavily_api_key,
        )

        retriever = pipeline.execute()

        # 如果本地資料庫無法使用 (retriever is None)
        if retriever is None:
            if not tavily_api_key:
                msg = "無法建立研究報告：未設定 Tavily API Key 且無可用的本地資料庫。"
                logger.error(f"[Research] ❌ {msg}")
                update_progress(
                    job_id,
                    status="failed",
                    phase="錯誤",
                    message=msg,
                    error="尚未設定 Tavily API Key 且無可用的本地向量資料庫",
                )
                return {"success": False, "error": msg}

            logger.info("[Research] 本地資料庫不存在，嘗試聯網搜尋建立...")
            update_progress(job_id, phase="聯網檢索", message="本地資料庫不存在，正在透過 Tavily 聯網搜尋建立...")

            pipeline = RAGPipeline(
                query=topic,
                llm_instance=shared_llm,
                max_results=max_results,
                use_existing_db=False,
                tavily_api_key=tavily_api_key,
            )

            retriever = pipeline.execute()

            if retriever is None:
                update_progress(
                    job_id,
                    status="failed",
                    phase="錯誤",
                    message="無法建立向量資料庫 (搜尋失敗或 API Key 無效)",
                    error="RAG Init Failed",
                )
                return {"success": False, "error": "無法建立向量資料庫"}

        # ========== 2. 生成大綱 ==========
        update_progress(job_id, phase="生成大綱", message="正在規劃報告章節結構...")
        logger.info("[Research] 開始生成大綱...")

        output = pipeline.generate_structured_outline(retriever, topic)["result"]
        outline = parsers.extract(response_output=output)

        if not outline:
            update_progress(
                job_id,
                status="failed",
                phase="錯誤",
                message="大綱生成失敗：LLM 輸出格式無法解析",
                error="大綱生成失敗",
            )
            return {"success": False, "error": "大綱生成失敗"}

        logger.info(f"[Research] ✅ 大綱解析成功，共 {len(outline)} 個章節")

        # ========== 3. 計算總項目數 ==========
        total_items = sum(
            len(section["subpoints"])
            + sum(
                len(sub.get("subpoints", [])) if isinstance(sub, dict) else 0
                for sub in section["subpoints"]
            )
            for section in outline
        )

        update_progress(
            job_id,
            total=total_items,
            phase="生成內容",
            message=f"共 {total_items} 個小節，開始生成研究內容...",
        )

        # ========== 4. 生成內容 ==========
        results = generate_content_concurrently(
            pipeline=pipeline,
            retriever=retriever,
            outline=outline,
            topic=topic,
            shared_llm=shared_llm,
            max_concurrency=max_concurrency,
            progress_callback=lambda current_count, phase, message: update_progress(
                job_id, current=current_count, phase=phase, message=message
            ),
            progress_lock=PROGRESS_LOCK,
        )

        html_content = results["html_content"]
        processed_items = results["processed_items"]
        failed_items = results["failed_items"]
        total_items = results["total_items"]

        # ========== 5. 完成 ==========
        success_items = total_items - failed_items
        success_rate = (success_items / total_items * 100) if total_items > 0 else 0

        logger.info(f"[Research] ✅ 研究完成！成功率: {success_rate:.1f}%")

        update_progress(
            job_id,
            status="completed",
            current=total_items,
            phase="完成",
            message=f"✅ 已完成 {success_items}/{total_items} 個小節",
            result_html=html_content,
            stats={
                "successRate": f"{success_rate:.1f}%",
                "successItems": success_items,
                "totalItems": total_items,
                "wordCount": len(html_content),
            },
        )

        return {"success": True, "html": html_content}

    except Exception as e:
        logger.error(f"[Research] ❌ 研究過程發生錯誤: {e}", exc_info=True)
        update_progress(
            job_id,
            status="failed",
            phase="錯誤",
            message=f"研究過程發生錯誤: {str(e)}",
            error=str(e),
        )
        return {"success": False, "error": str(e)}
    finally:
        # 15 分鐘後自動釋放該任務的記憶體快取
        cleanup_progress(job_id, delay_seconds=900)