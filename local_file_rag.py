# -*- coding: utf-8 -*-
"""
端側混合檢索引擎 (Zero-LangChain — Hybrid Local RAG for Code & Papers)
支援：
1. PDF 論文「精確頁碼 (Page Number)」解析與段落優先滑動切塊
2. 原生 Hybrid Search：BM25Okapi (支援 C++ snake_case 識別符、英文術語與中文 jieba) + ChromaDB Dense Vector (BGE-Small) 透過 RRF 融合排序
"""

import os
import re
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import jieba
import torch
import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("local_file_rag")

# 自動偵測最佳嵌入裝置：Windows/Linux 優先 CUDA，macOS 可用 MPS 或 CPU
EMBED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)

UPLOAD_DIR = os.path.join(os.getcwd(), "user_uploads")
VECTOR_DIR = os.path.join(os.getcwd(), "user_file_vectors")

CODE_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".py", ".js", ".ts", ".java", ".rs", ".go"}
TEXT_EXTENSIONS = {".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".cmake"}


@dataclass
class _NativeChunk:
    """輕量級原生文件切片結構"""
    page_content: str
    metadata: Dict[str, Any]


def _hybrid_code_and_text_tokenizer(text: str) -> List[str]:
    """
    專為 C++ 程式碼、arXiv 英文論文與繁體中文設計的混合分詞器：
    保留變數名稱 (如 bellman_ford, max_depth)、LaTeX 符號與英文單字，並對中文使用 jieba
    """
    tokens = []
    eng_and_code = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())
    tokens.extend(eng_and_code)
    chinese_segments = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in chinese_segments:
        tokens.extend(tok.strip() for tok in jieba.cut(seg) if tok.strip())
    return tokens if tokens else [text.lower()]


def _split_text_native(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """純 Python 段落與句子優先滑動視窗切分器（取代 RecursiveCharacterTextSplitter）"""
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                # 單一段落過長時，依句號/換行或滑動視窗切分
                step = max(1, chunk_size - chunk_overlap)
                for start in range(0, len(para), step):
                    sub = para[start : start + chunk_size].strip()
                    if sub:
                        chunks.append(sub)
                current = ""

    if current:
        chunks.append(current)

    return chunks


class LocalFileRAGManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        logger.info(f"[LocalRAG] 載入原生 Embedding 模型: {embed_model_name} (Device: {EMBED_DEVICE})")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        self.embedder = SentenceTransformer(embed_model_name, device=EMBED_DEVICE)

    @classmethod
    def get_instance(cls) -> "LocalFileRAGManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_user_paths(self, user_id: str) -> Tuple[str, str]:
        safe_uid = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        u_upload = os.path.join(UPLOAD_DIR, safe_uid)
        u_vector = os.path.join(VECTOR_DIR, safe_uid)
        os.makedirs(u_upload, exist_ok=True)
        os.makedirs(u_vector, exist_ok=True)
        return u_upload, u_vector

    def _get_collection(self, u_vector: str) -> chromadb.Collection:
        """取得或建立原生 ChromaDB Collection（相容舊版 LangChain 建立的 'langchain' Collection）"""
        client = chromadb.PersistentClient(path=u_vector)
        existing_cols = [c.name for c in client.list_collections()]
        if "langchain" in existing_cols:
            return client.get_collection(name="langchain")
        return client.get_or_create_collection(
            name="local_files",
            metadata={"hnsw:space": "cosine"},
        )

    # ==================== 檔案解析與行號/頁碼切塊 ====================

    def _chunk_code_with_line_numbers(
        self, content: str, filename: str, lines_per_chunk: int = 45, overlap_lines: int = 10
    ) -> List[_NativeChunk]:
        """將 C++ / Python 程式碼按行號切塊，並在內容開頭標註精確行號區間"""
        lines = content.splitlines()
        total_lines = len(lines)
        chunks: List[_NativeChunk] = []

        if total_lines == 0:
            return chunks

        step = max(1, lines_per_chunk - overlap_lines)
        for start_idx in range(0, total_lines, step):
            end_idx = min(total_lines, start_idx + lines_per_chunk)
            start_line = start_idx + 1
            end_line = end_idx

            numbered_lines = [
                f"{line_no:4d} | {lines[line_no - 1]}"
                for line_no in range(start_line, end_line + 1)
            ]
            chunk_body = "\n".join(numbered_lines)
            header = f"【檔案：{filename} | 行號：Lines {start_line}-{end_line}】\n"

            chunks.append(
                _NativeChunk(
                    page_content=header + chunk_body,
                    metadata={
                        "source": filename,
                        "location": f"Lines {start_line}-{end_line}",
                        "type": "code",
                    },
                )
            )
            if end_idx >= total_lines:
                break
        return chunks

    def _chunk_pdf_with_pages(self, file_path: str, filename: str) -> List[_NativeChunk]:
        """解析 PDF 論文，保留頁碼與 LaTeX 數學公式上下文"""
        chunks: List[_NativeChunk] = []
        try:
            reader = PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                raw_text = page.extract_text() or ""
                if not raw_text.strip():
                    continue
                sub_chunks = _split_text_native(raw_text, chunk_size=900, chunk_overlap=150)
                for sub_idx, chunk in enumerate(sub_chunks, 1):
                    header = f"【論文檔案：{filename} | 頁碼：Page {page_num} (Part {sub_idx})】\n"
                    chunks.append(
                        _NativeChunk(
                            page_content=header + chunk,
                            metadata={
                                "source": filename,
                                "location": f"Page {page_num}",
                                "type": "pdf",
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"[LocalRAG] 解析 PDF 失敗 ({filename}): {e}")
        return chunks

    def _chunk_plain_text(self, content: str, filename: str) -> List[_NativeChunk]:
        """解析 Markdown / LaTeX (.tex) / 一般文字檔，保留行號區間"""
        return self._chunk_code_with_line_numbers(content, filename, lines_per_chunk=50, overlap_lines=10)

    # ==================== 索引建立與管理 ====================

    def ingest_file(self, user_id: str, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """儲存上傳的檔案並建立原生端側向量索引"""
        u_upload, u_vector = self._get_user_paths(user_id)
        safe_name = re.sub(r'[\\/:*?"<>|]+', "__", filename).lstrip("_")
        file_path = os.path.join(u_upload, safe_name)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        ext = os.path.splitext(safe_name)[1].lower()
        chunks: List[_NativeChunk] = []

        if ext == ".pdf":
            chunks = self._chunk_pdf_with_pages(file_path, filename)
        elif ext in CODE_EXTENSIONS or ext in TEXT_EXTENSIONS:
            text_content = file_bytes.decode("utf-8", errors="replace")
            if ext in CODE_EXTENSIONS:
                chunks = self._chunk_code_with_line_numbers(text_content, filename)
            else:
                chunks = self._chunk_plain_text(text_content, filename)
        else:
            try:
                text_content = file_bytes.decode("utf-8")
                chunks = self._chunk_code_with_line_numbers(text_content, filename)
            except UnicodeDecodeError:
                os.remove(file_path)
                return {"success": False, "error": f"不支援的二進位檔案格式: {filename}"}

        if not chunks:
            return {"success": False, "error": f"無法從檔案提取有效內容: {filename}"}

        collection = self._get_collection(u_vector)
        try:
            existing = collection.get(where={"source": filename}, include=[])
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [
            hashlib.md5(f"{filename}_{i}_{c.page_content[:30]}".encode("utf-8")).hexdigest() + f"_{i}"
            for i, c in enumerate(chunks)
        ]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"[LocalRAG] 已為使用者 {user_id} 索引 {filename} (共 {len(chunks)} 個 Chunks)")

        return {
            "success": True,
            "filename": filename,
            "size": len(file_bytes),
            "chunks": len(chunks),
        }

    def list_files(self, user_id: str) -> List[Dict[str, Any]]:
        """列出使用者目前已索引的檔案清單"""
        _, u_vector = self._get_user_paths(user_id)
        sqlite_path = os.path.join(u_vector, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            return []

        try:
            collection = self._get_collection(u_vector)
            metadatas = collection.get(include=["metadatas"]).get("metadatas") or []
            file_stats: Dict[str, int] = {}
            for meta in metadatas:
                if meta and "source" in meta:
                    src = meta["source"]
                    file_stats[src] = file_stats.get(src, 0) + 1
            return [{"name": k, "chunks": v} for k, v in file_stats.items()]
        except Exception as e:
            logger.error(f"[LocalRAG] 讀取檔案列表失敗: {e}")
            return []

    def delete_file(self, user_id: str, filename: str) -> bool:
        """從向量庫與硬碟刪除指定檔案"""
        u_upload, u_vector = self._get_user_paths(user_id)
        safe_name = re.sub(r'[\\/:*?"<>|]+', "__", filename).lstrip("_")
        file_path = os.path.join(u_upload, safe_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        try:
            collection = self._get_collection(u_vector)
            existing = collection.get(where={"source": filename}, include=[])
            if existing and existing.get("ids"):
                collection.delete(ids=existing["ids"])
            return True
        except Exception as e:
            logger.error(f"[LocalRAG] 刪除檔案失敗 ({filename}): {e}")
            return False

    # ==================== 原生端側混合檢索 (BM25Okapi + Chroma Vector + RRF) ====================

    def retrieve_hybrid(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        active_files: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        執行原生 Hybrid Search (BM25 權重 0.45 + BGE 稠密向量權重 0.55)
        同時精準命中 C++ 函數名/LaTeX 變數 (BM25) 與語意概念 (Vector)
        """
        _, u_vector = self._get_user_paths(user_id)
        sqlite_path = os.path.join(u_vector, "chroma.sqlite3")
        if not os.path.exists(sqlite_path) or not query or not query.strip():
            return "", []

        try:
            collection = self._get_collection(u_vector)
            db_data = collection.get(include=["documents", "metadatas"])
            raw_ids = db_data.get("ids") or []
            raw_docs = db_data.get("documents") or []
            raw_metas = db_data.get("metadatas") or []

            if not raw_docs:
                return "", []

            # 篩選啟用中的檔案切片
            filtered_items: List[Tuple[str, str, Dict[str, Any]]] = []
            id_to_item: Dict[str, Tuple[str, Dict[str, Any]]] = {}

            for cid, text, meta in zip(raw_ids, raw_docs, raw_metas):
                meta = meta or {}
                if active_files and meta.get("source") not in active_files:
                    continue
                filtered_items.append((cid, text, meta))
                id_to_item[cid] = (text, meta)

            if not filtered_items:
                return "", []

            k = min(top_k, len(filtered_items))
            rrf_k = 60
            bm25_weight = 0.45
            vector_weight = 0.55
            rrf_scores: Dict[str, float] = {}

            # 1. ChromaDB 語意向量檢索
            query_vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": query_vec,
                "n_results": k,
            }
            if active_files:
                query_kwargs["where"] = (
                    {"source": active_files[0]}
                    if len(active_files) == 1
                    else {"source": {"$in": active_files}}
                )

            vec_res = collection.query(**query_kwargs)
            vec_ids = (vec_res.get("ids") or [[]])[0]
            for rank, cid in enumerate(vec_ids):
                if cid in id_to_item:
                    rrf_scores[cid] = rrf_scores.get(cid, 0.0) + vector_weight / (rrf_k + rank + 1)

            # 2. BM25Okapi 關鍵字檢索（精準命中 C++ 變數/函數與專有名詞）
            tokenized_corpus = [_hybrid_code_and_text_tokenizer(item[1]) for item in filtered_items]
            if any(tokenized_corpus):
                bm25 = BM25Okapi(tokenized_corpus)
                query_tokens = _hybrid_code_and_text_tokenizer(query)
                bm25_scores = bm25.get_scores(query_tokens)

                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda idx: bm25_scores[idx],
                    reverse=True,
                )[:k]

                for rank, idx in enumerate(ranked_indices):
                    if bm25_scores[idx] > 0:
                        cid = filtered_items[idx][0]
                        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (rrf_k + rank + 1)

            # 3. 依 RRF 融合總分排序並組裝回傳內容
            ranked_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

            context_blocks: List[str] = []
            sources: List[Dict[str, str]] = []
            seen_locs = set()

            for cid in ranked_cids:
                text, meta = id_to_item[cid]
                src = meta.get("source", "未知檔案")
                loc = meta.get("location", "")
                context_blocks.append(text)
                badge_key = f"{src} ({loc})" if loc else src
                if badge_key not in seen_locs:
                    seen_locs.add(badge_key)
                    sources.append({"title": badge_key, "url": ""})

            return "\n\n" + ("\n" + "=" * 40 + "\n").join(context_blocks), sources

        except Exception as e:
            logger.error(f"[LocalRAG] 混合檢索失敗: {e}")
            return "", []