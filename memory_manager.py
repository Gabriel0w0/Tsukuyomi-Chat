# -*- coding: utf-8 -*-
"""
結構化長期記憶管理系統 (Zero-LangChain 原生四階段架構)
- Storage: SQLite (memory_items 結構化主表) + ChromaDB (語意向量索引)
- Retrieval: Core 常駐畫像 + BGE/BM25 混合檢索 + 時效衰減與重要度複合排序
- Update: 候選萃取 -> 語意相似度比對 -> ADD / UPDATE / DELETE / NOOP 狀態機 -> 智慧淘汰
- Generation: 分層上下文注入（含相對時間標記）與短期對話滾動摘要
"""

import os
import re
import time
import math
import json
import uuid
import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import jieba
import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import DB_FILE, MAX_FACTS_PER_USER
from utils.regex import extract_facts

logger = logging.getLogger("memory_manager")

MEMORY_VECTOR_DIR = os.path.join(os.getcwd(), "user_memory_vectors")
EMBED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)

VALID_MEMORY_TYPES = {"core", "episodic", "preference"}

PERSONAL_FACT_HINTS = (
    "我", "叫我", "名字", "喜歡", "討厭", "愛吃", "愛喝", "專題", "研究",
    "大學", "系", "年級", "住在", "生日", "習慣", "正在", "準備", "工作",
    "興趣", "記住", "目標", "考試", "運動", "不吃", "不加"
)


def _tokenize_zh_en(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(t.strip() for t in jieba.cut(seg) if t.strip())
    return tokens if tokens else [text.lower()]


def _format_relative_time(ts: float, now: Optional[float] = None) -> str:
    now = now or time.time()
    delta_days = max(0.0, (now - ts) / 86400.0)
    if delta_days < 1.0:
        return "今天"
    if delta_days < 2.0:
        return "昨天"
    if delta_days < 7.0:
        return f"{int(delta_days)} 天前"
    if delta_days < 30.0:
        return f"{int(delta_days // 7)} 週前"
    return f"{int(delta_days // 30)} 個月前"


class MemorySystemManager:
    _instance: Optional["MemorySystemManager"] = None
    _lock = threading.Lock()

    def __init__(self, db_file: str = DB_FILE, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.db_file = db_file
        self.embed_model_name = embed_model_name
        self._embedder: Optional[SentenceTransformer] = None
        self._embed_lock = threading.Lock()
        self._db_lock = threading.Lock()

        os.makedirs(MEMORY_VECTOR_DIR, exist_ok=True)
        self._init_schema()

    @classmethod
    def get_instance(cls) -> "MemorySystemManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_embedder(self) -> SentenceTransformer:
        """優先複用 LocalFileRAGManager 的 SentenceTransformer 單例，避免重複佔用記憶體"""
        with self._embed_lock:
            if self._embedder is not None:
                return self._embedder
            try:
                from local_file_rag import LocalFileRAGManager
                if LocalFileRAGManager._instance is not None:
                    self._embedder = LocalFileRAGManager._instance.embedder
                    logger.info("[MemorySys] 已複用 LocalFileRAGManager 的 BGE Embedding 實例")
                    return self._embedder
            except Exception:
                pass
            logger.info(f"[MemorySys] 載入獨立 Embedding 模型: {self.embed_model_name} ({EMBED_DEVICE})")
            self._embedder = SentenceTransformer(self.embed_model_name, device=EMBED_DEVICE)
            return self._embedder

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_file, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'episodic',
                    importance INTEGER DEFAULT 5,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    is_active INTEGER DEFAULT 1
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_user_active ON memory_items(user_id, is_active)"
            )

    def _get_collection(self, user_id: str) -> chromadb.Collection:
        safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        u_dir = os.path.join(MEMORY_VECTOR_DIR, safe_uid)
        os.makedirs(u_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=u_dir)
        return client.get_or_create_collection(
            name="long_term_memories",
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_vector(self, user_id: str, mem_id: str, content: str, mem_type: str, importance: int) -> None:
        try:
            embedder = self._get_embedder()
            vec = embedder.encode([content], normalize_embeddings=True).tolist()
            col = self._get_collection(user_id)
            col.upsert(
                ids=[mem_id],
                documents=[content],
                embeddings=vec,
                metadatas=[{"memory_type": mem_type, "importance": int(importance)}],
            )
        except Exception as e:
            logger.error(f"[MemorySys] 寫入 Chroma 向量失敗 ({mem_id}): {e}")

    def _delete_vector(self, user_id: str, mem_ids: List[str]) -> None:
        if not mem_ids:
            return
        try:
            col = self._get_collection(user_id)
            col.delete(ids=mem_ids)
        except Exception as e:
            logger.warning(f"[MemorySys] 移除 Chroma 向量失敗: {e}")

    # ==================== 舊版 user_memory.facts 自動遷移 ====================

    def ensure_migrated(self, user_id: str) -> None:
        """若舊表 user_memory 有 facts 且新表 memory_items 尚無紀錄，自動無痛遷移"""
        with self._db_lock:
            with self._get_conn() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(1) AS cnt FROM memory_items WHERE user_id = ?", (user_id,))
                if c.fetchone()["cnt"] > 0:
                    return

                c.execute("SELECT facts FROM user_memory WHERE user_id = ?", (user_id,))
                row = c.fetchone()
                if not row or not row["facts"]:
                    return

                try:
                    old_facts = json.loads(row["facts"])
                except Exception:
                    return

                if not isinstance(old_facts, list) or not old_facts:
                    return

                now = time.time()
                migrated: List[Tuple[str, str]] = []
                for f in old_facts:
                    text = str(f).strip()
                    if not text:
                        continue
                    mid = str(uuid.uuid4())
                    c.execute(
                        """
                        INSERT INTO memory_items
                        (id, user_id, content, memory_type, importance, created_at, updated_at, last_accessed, access_count, is_active)
                        VALUES (?, ?, ?, 'core', 7, ?, ?, ?, 1, 1)
                        """,
                        (mid, user_id, text, now, now, now),
                    )
                    migrated.append((mid, text))

            for mid, text in migrated:
                self._upsert_vector(user_id, mid, text, "core", 7)

            if migrated:
                logger.info(f"[MemorySys] 已將使用者 {user_id} 的 {len(migrated)} 條舊記憶遷移至 memory_items")

    # ==================== 1. Storage CRUD（供 UI 與內部共用） ====================

    def list_active_memories(self, user_id: str) -> List[Dict[str, Any]]:
        self.ensure_migrated(user_id)
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, memory_type, importance, created_at, updated_at, last_accessed, access_count
                FROM memory_items
                WHERE user_id = ? AND is_active = 1
                ORDER BY
                    CASE memory_type WHEN 'core' THEN 0 WHEN 'preference' THEN 1 ELSE 2 END,
                    importance DESC,
                    updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_memory_item(
        self,
        user_id: str,
        content: str,
        memory_type: str = "core",
        importance: int = 8,
        max_facts: int = MAX_FACTS_PER_USER,
    ) -> Dict[str, Any]:
        self.ensure_migrated(user_id)
        clean = content.strip()
        mtype = memory_type if memory_type in VALID_MEMORY_TYPES else "episodic"
        imp = max(1, min(10, int(importance)))
        now = time.time()
        mem_id = str(uuid.uuid4())

        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_items
                    (id, user_id, content, memory_type, importance, created_at, updated_at, last_accessed, access_count, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
                    """,
                    (mem_id, user_id, clean, mtype, imp, now, now, now),
                )

        self._upsert_vector(user_id, mem_id, clean, mtype, imp)
        self._evict_if_needed(user_id, max_facts)
        return {
            "id": mem_id,
            "content": clean,
            "memory_type": mtype,
            "importance": imp,
            "updated_at": now,
        }

    def update_memory_by_id(
        self,
        user_id: str,
        mem_id: str,
        new_content: str,
        memory_type: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> bool:
        clean = new_content.strip()
        if not clean:
            return False
        now = time.time()

        with self._db_lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT memory_type, importance FROM memory_items WHERE id = ? AND user_id = ? AND is_active = 1",
                    (mem_id, user_id),
                ).fetchone()
                if not row:
                    return False

                final_type = memory_type if memory_type in VALID_MEMORY_TYPES else row["memory_type"]
                final_imp = max(1, min(10, int(importance))) if importance is not None else row["importance"]

                conn.execute(
                    """
                    UPDATE memory_items
                    SET content = ?, memory_type = ?, importance = ?, updated_at = ?, last_accessed = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (clean, final_type, final_imp, now, now, mem_id, user_id),
                )

        self._upsert_vector(user_id, mem_id, clean, final_type, final_imp)
        return True

    def soft_delete_memory_by_id(self, user_id: str, mem_id: str) -> bool:
        now = time.time()
        with self._db_lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "UPDATE memory_items SET is_active = 0, updated_at = ? WHERE id = ? AND user_id = ? AND is_active = 1",
                    (now, mem_id, user_id),
                )
                affected = cur.rowcount
        if affected > 0:
            self._delete_vector(user_id, [mem_id])
            return True
        return False

    def update_memory_by_index(self, user_id: str, index: int, new_content: str) -> Optional[List[Dict[str, Any]]]:
        items = self.list_active_memories(user_id)
        if index < 0 or index >= len(items):
            return None
        self.update_memory_by_id(user_id, items[index]["id"], new_content)
        return self.list_active_memories(user_id)

    def delete_memory_by_index(self, user_id: str, index: int) -> Optional[Tuple[List[Dict[str, Any]], str]]:
        items = self.list_active_memories(user_id)
        if index < 0 or index >= len(items):
            return None
        removed_text = items[index]["content"]
        self.soft_delete_memory_by_id(user_id, items[index]["id"])
        return self.list_active_memories(user_id), removed_text

    def clear_user_memories(self, user_id: str) -> None:
        with self._db_lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM memory_items WHERE user_id = ?", (user_id,)
                ).fetchall()
                ids = [r["id"] for r in rows]
                conn.execute("DELETE FROM memory_items WHERE user_id = ?", (user_id,))
        self._delete_vector(user_id, ids)

    def _evict_if_needed(self, user_id: str, max_facts: int) -> None:
        """淘汰重要度最低且最久未存取的非核心記憶"""
        items = self.list_active_memories(user_id)
        overflow = len(items) - max(1, int(max_facts))
        if overflow <= 0:
            return

        # 優先淘汰 episodic -> preference -> core，同類則依 importance 低、last_accessed 舊排序
        type_priority = {"episodic": 0, "preference": 1, "core": 2}
        victims = sorted(
            items,
            key=lambda m: (type_priority.get(m["memory_type"], 0), m["importance"], m["last_accessed"]),
        )[:overflow]

        for v in victims:
            self.soft_delete_memory_by_id(user_id, v["id"])
            logger.info(f"[MemorySys] 淘汰低活躍記憶: {v['content']} (type={v['memory_type']}, imp={v['importance']})")

    # ==================== 2. Retrieval（核心常駐 + 複合分數召回） ====================

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        core_limit: int = 6,
        is_recall_query: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        self.ensure_migrated(user_id)
        all_active = self.list_active_memories(user_id)
        if not all_active:
            return {"core": [], "recalled": []}

        # A. 固定載入 Core Memory
        core_items = [
            m for m in all_active if m["memory_type"] == "core" or m["importance"] >= 8
        ][:core_limit]
        core_ids = {m["id"] for m in core_items}

        # B. 其餘記憶進行混合檢索 + 複合排序（若為純記憶回想查詢，放寬召回上限）
        pool = [m for m in all_active if m["id"] not in core_ids]
        if not pool:
            return {"core": core_items, "recalled": []}

        effective_k = max(top_k, 12) if is_recall_query else top_k
        pool_map = {m["id"]: m for m in pool}

        # B1. 向量語意分數
        vec_sim: Dict[str, float] = {}
        try:
            embedder = self._get_embedder()
            q_vec = embedder.encode([query], normalize_embeddings=True).tolist()
            col = self._get_collection(user_id)
            n_res = min(max(effective_k * 3, 15), len(all_active))
            res = col.query(query_embeddings=q_vec, n_results=n_res, include=["distances"])
            ids_res = (res.get("ids") or [[]])[0]
            dist_res = (res.get("distances") or [[]])[0]
            for mid, dist in zip(ids_res, dist_res):
                if mid in pool_map:
                    vec_sim[mid] = max(0.0, min(1.0, 1.0 - float(dist)))
        except Exception as e:
            logger.warning(f"[MemorySys] 向量召回降級: {e}")

        # B2. BM25 關鍵字分數
        bm25_sim: Dict[str, float] = {}
        try:
            tokenized = [_tokenize_zh_en(m["content"]) for m in pool]
            if any(tokenized):
                bm25 = BM25Okapi(tokenized)
                raw_scores = bm25.get_scores(_tokenize_zh_en(query))
                max_s = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0
                if max_s > 0:
                    for m, s in zip(pool, raw_scores):
                        bm25_sim[m["id"]] = float(s) / max_s
        except Exception:
            pass

        # B3. 複合分數：Rel(0.60) + Recency(0.20) + Importance(0.20)
        now = time.time()
        decay_lambda = 0.03
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for m in pool:
            mid = m["id"]
            rel = 0.70 * vec_sim.get(mid, 0.0) + 0.30 * bm25_sim.get(mid, 0.0)
            delta_days = max(0.0, (now - float(m["last_accessed"])) / 86400.0)
            recency = math.exp(-decay_lambda * delta_days)
            imp = float(m["importance"]) / 10.0

            # 非記憶回想問題時，過濾掉完全無關的低分條目
            if not is_recall_query and rel < 0.28:
                continue

            total_score = 0.60 * rel + 0.20 * recency + 0.20 * imp
            scored.append((total_score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        recalled_items = [m for _, m in scored[:effective_k]]

        # 記憶強化：更新被召回記憶的存取時間與次數
        touched_ids = [m["id"] for m in recalled_items]
        if touched_ids:
            try:
                with self._db_lock:
                    with self._get_conn() as conn:
                        conn.executemany(
                            "UPDATE memory_items SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                            [(now, mid) for mid in touched_ids],
                        )
            except Exception:
                pass

        return {"core": core_items, "recalled": recalled_items}

    # ==================== 3. Update（萃取 + 語意衝突消解狀態機） ====================

    def _find_similar_memories(self, user_id: str, text: str, top_k: int = 3, threshold: float = 0.68) -> List[Dict[str, Any]]:
        active_map = {m["id"]: m for m in self.list_active_memories(user_id)}
        if not active_map:
            return []

        try:
            embedder = self._get_embedder()
            q_vec = embedder.encode([text], normalize_embeddings=True).tolist()
            col = self._get_collection(user_id)
            res = col.query(
                query_embeddings=q_vec,
                n_results=min(top_k, len(active_map)),
                include=["distances"],
            )
            ids_res = (res.get("ids") or [[]])[0]
            dist_res = (res.get("distances") or [[]])[0]

            matches: List[Dict[str, Any]] = []
            for mid, dist in zip(ids_res, dist_res):
                sim = 1.0 - float(dist)
                if mid in active_map and sim >= threshold:
                    item = dict(active_map[mid])
                    item["similarity"] = round(sim, 3)
                    matches.append(item)
            return matches
        except Exception as e:
            logger.warning(f"[MemorySys] 相似記憶查詢失敗: {e}")
            return []

    async def process_turn_memory(
        self,
        user_id: str,
        user_message: str,
        json_llm_caller: Callable[[str], Any],
        last_bot_message: str = "",
        max_facts: int = MAX_FACTS_PER_USER,
    ) -> None:
        """
        背景非同步執行：
        1. 從當輪對話萃取候選事實（附帶 type 與 importance）
        2. 檢索既有相似記憶並決策 ADD / UPDATE / DELETE / NOOP
        """
        msg = user_message.strip()
        if len(msg) < 3:
            return

        self.ensure_migrated(user_id)
        candidates: List[Dict[str, Any]] = []

        # 規則快速萃取（低延遲保底）
        regex_facts = extract_facts(msg)
        for rf in regex_facts:
            if isinstance(rf, str) and len(rf.strip()) >= 3:
                candidates.append({"content": rf.strip(), "type": "core", "importance": 7})

        # 若命中個人事實線索，呼叫輕量 memory_model 進行結構化萃取
        if not candidates and any(hint in msg for hint in PERSONAL_FACT_HINTS):
            context_line = f"上一句助理提問：「{last_bot_message[:120]}」\n" if last_bot_message else ""
            extract_prompt = f"""請從以下使用者訊息中提取值得長期記住的「使用者個人事實」（名字、喜好、目標、專案、習慣、近期計畫等）。
規則：
1. 每條 content 必須是完整的繁體中文短句。
2. type 只能是 "core"（長期背景/稱呼/目標）、"preference"（喜好/工具偏好）或 "episodic"（近期事件/進行中任務）。
3. importance 為 1 到 10 的整數。
4. 若只是打招呼或一般知識提問，candidates 請回傳空陣列 []。

只輸出單一合法 JSON：
{{"candidates": [{{"content": "繁體中文短句", "type": "preference", "importance": 7}}]}}

{context_line}使用者訊息：「{msg}」"""
            extracted = await json_llm_caller(extract_prompt)
            raw_cands = (extracted or {}).get("candidates") or []
            for item in raw_cands:
                if isinstance(item, dict):
                    c_text = str(item.get("content") or "").strip()
                    if len(c_text) >= 4:
                        candidates.append(
                            {
                                "content": c_text,
                                "type": str(item.get("type") or "episodic").lower(),
                                "importance": int(item.get("importance") or 6),
                            }
                        )

        if not candidates:
            return

        # 逐條執行衝突消解狀態機
        for cand in candidates:
            c_text = cand["content"]
            c_type = cand["type"] if cand["type"] in VALID_MEMORY_TYPES else "episodic"
            c_imp = max(1, min(10, int(cand["importance"])))

            similar = self._find_similar_memories(user_id, c_text, top_k=3, threshold=0.68)

            # 無相似舊記憶 -> 直接 ADD
            if not similar:
                self.add_memory_item(user_id, c_text, memory_type=c_type, importance=c_imp, max_facts=max_facts)
                logger.info(f"[MemorySys][ADD] {c_text} ({c_type}, imp={c_imp})")
                continue

            # 完全相同 -> 直接 NOOP
            if any(m["content"].strip() == c_text for m in similar):
                logger.info(f"[MemorySys][NOOP-Exact] {c_text}")
                continue

            # 交由輕量模型判斷 ADD / UPDATE / DELETE / NOOP
            existing_lines = "\n".join(
                f'- id="{m["id"]}" | 內容="{m["content"]}" | 類型={m["memory_type"]}'
                for m in similar
            )
            reconcile_prompt = f"""你是記憶庫管理員。請比較「新事實」與「既有相似記憶」，決定單一更新動作：
- "NOOP"：意思完全重複，無需變更。
- "UPDATE"：新事實補充了細節或更新了舊記憶的狀態，請合併成一句更完整的新內容並填入 target_id。
- "DELETE"：新事實明確推翻或宣告某條舊記憶失效，請填入 target_id（若同時有新狀態可於 content 補充）。
- "ADD"：其實是不同面向的獨立資訊，應另存新條目。

既有相似記憶：
{existing_lines}

新事實：「{c_text}」

只輸出單一合法 JSON：
{{"action": "NOOP|UPDATE|DELETE|ADD", "target_id": "", "content": "{c_text}"}}"""

            decision = await json_llm_caller(reconcile_prompt)
            action = str((decision or {}).get("action") or "ADD").upper().strip()
            target_id = str((decision or {}).get("target_id") or "").strip()
            merged_content = str((decision or {}).get("content") or c_text).strip() or c_text

            valid_target_ids = {m["id"] for m in similar}

            if action == "NOOP":
                logger.info(f"[MemorySys][NOOP] 略過重複記憶: {c_text}")
            elif action == "UPDATE" and target_id in valid_target_ids:
                self.update_memory_by_id(user_id, target_id, merged_content, memory_type=c_type, importance=c_imp)
                logger.info(f"[MemorySys][UPDATE] ({target_id}) -> {merged_content}")
            elif action == "DELETE" and target_id in valid_target_ids:
                self.soft_delete_memory_by_id(user_id, target_id)
                logger.info(f"[MemorySys][DELETE] ({target_id})")
            else:
                self.add_memory_item(user_id, merged_content, memory_type=c_type, importance=c_imp, max_facts=max_facts)
                logger.info(f"[MemorySys][ADD] {merged_content}")

    # ==================== 4. Generation（分層上下文格式化） ====================

    @staticmethod
    def format_memories_for_prompt(retrieved: Dict[str, List[Dict[str, Any]]]) -> str:
        core_items = retrieved.get("core") or []
        recalled_items = retrieved.get("recalled") or []
        if not core_items and not recalled_items:
            return ""

        now = time.time()
        sections: List[str] = []

        if core_items:
            lines = "\n".join(f"- {m['content']}" for m in core_items)
            sections.append(f"[Core User Profile]\n{lines}")

        if recalled_items:
            lines = "\n".join(
                f"- [{_format_relative_time(float(m['updated_at']), now)}] {m['content']}"
                for m in recalled_items
            )
            sections.append(f"[Relevant Recalled Memories]\n{lines}")

        return (
            "\n\n".join(sections)
            + "\nNaturally weave these remembered facts into your responses when relevant, "
            "respecting the timeline, without explicitly stating that you read them from a database.\n"
        )