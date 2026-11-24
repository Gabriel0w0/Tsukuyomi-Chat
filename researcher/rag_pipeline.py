# -*- coding: utf-8 -*-
import os
import re
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import jieba
import torch
import chromadb
from filelock import FileLock
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from deep_translator import GoogleTranslator

# 提示詞模板
from researcher.prompts import PromptTemps

logger = logging.getLogger("rag_pipeline")

# 自動偵測最佳嵌入裝置：Windows/Linux 優先 CUDA，Apple Silicon 優先 MPS，其餘退回 CPU
EMBED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)

# 過濾模型思考區塊與殘留特殊標籤的正規表示式
RE_THINK_BLOCK = re.compile(
    r"(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)(.*?)(?:<[\|\/]*channel[\|\/]*>|<[\|\/]*(?:think|thought)[\|\/]*>)",
    re.DOTALL | re.IGNORECASE,
)
RE_UNCLOSED_THINK = re.compile(
    r"^\s*(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)(.*)$",
    re.DOTALL | re.IGNORECASE,
)
RE_STRAY_TAGS = re.compile(
    r"(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>|<end_of_turn>|<eos>|<\|eot_id\|>|<\|im_end\|>)",
    re.IGNORECASE,
)


def _strip_thinking_tags(raw_text: str) -> str:
    """移除模型輸出的思考過程區塊，確保大綱解析與報告內文乾淨無雜訊"""
    if not raw_text:
        return ""
    cleaned = RE_THINK_BLOCK.sub("", raw_text).strip()
    if cleaned == raw_text.strip():
        cleaned = RE_UNCLOSED_THINK.sub(r"\1", cleaned).strip()
    cleaned = RE_STRAY_TAGS.sub("", cleaned).strip()
    return cleaned


@dataclass
class _NativeDoc:
    """輕量級原生文件物件（相容 .page_content 與 .metadata 介面）"""
    page_content: str
    metadata: Optional[Dict[str, Any]] = None


class _NativeHybridRetriever:
    """
    純原生混合檢索器 (Chroma Dense Vector + BM25Okapi + RRF 倒數排序融合)
    完全取代 LangChain 的 EnsembleRetriever 與 ChromaRetriever
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: SentenceTransformer,
        all_docs: List[str],
        top_k: int = 5,
        enable_hybrid: bool = True,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.collection = collection
        self.embedder = embedder
        self.all_docs = all_docs
        self.top_k = top_k
        self.enable_hybrid = enable_hybrid
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.bm25: Optional[BM25Okapi] = None
        if self.enable_hybrid and self.all_docs:
            tokenized_corpus = [self._tokenize(doc) for doc in self.all_docs]
            if any(tokenized_corpus):
                self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok.strip().lower() for tok in jieba.cut(text) if tok.strip()]

    def invoke(self, query: str) -> List[_NativeDoc]:
        if not self.all_docs or not query or not query.strip():
            return []

        n_results = min(self.top_k * 2, len(self.all_docs))
        rrf_k = 60
        rrf_scores: Dict[str, float] = {}

        # 1. ChromaDB 語意向量檢索
        try:
            query_vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
            vec_res = self.collection.query(query_embeddings=query_vec, n_results=n_results)
            vec_docs = (vec_res.get("documents") or [[]])[0]
            for rank, doc_text in enumerate(vec_docs):
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + self.vector_weight / (rrf_k + rank + 1)
        except Exception as e:
            logger.error(f"[NativeRetriever] 向量檢索失敗: {e}")

        # 2. BM25 關鍵字檢索
        if self.enable_hybrid and self.bm25 is not None:
            try:
                query_tokens = self._tokenize(query)
                bm25_scores = self.bm25.get_scores(query_tokens)
                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda idx: bm25_scores[idx],
                    reverse=True,
                )[:n_results]

                for rank, doc_idx in enumerate(ranked_indices):
                    if bm25_scores[doc_idx] > 0:
                        doc_text = self.all_docs[doc_idx]
                        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + self.bm25_weight / (rrf_k + rank + 1)
            except Exception as e:
                logger.error(f"[NativeRetriever] BM25 檢索失敗: {e}")

        # 3. 依 RRF 融合總分排序並截取 top_k
        sorted_docs = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        return [_NativeDoc(page_content=text) for text, _ in sorted_docs]

    def get_relevant_documents(self, query: str) -> List[_NativeDoc]:
        return self.invoke(query)


class RAGPipeline:
    """
    純原生 RAG (Retrieval-Augmented Generation) 管道
    版本: 3.0.0 (Zero-LangChain 原生架構：TavilyClient + ChromaDB + SentenceTransformer + BM25Okapi)
    """

    _shared_embedder: Optional[SentenceTransformer] = None
    _embed_lock = threading.Lock()

    @classmethod
    def get_shared_embedder(cls, embed_model_name: str = "BAAI/bge-small-zh-v1.5") -> SentenceTransformer:
        """全程序共用同一個 SentenceTransformer 實例，避免重複載入權重"""
        with cls._embed_lock:
            if cls._shared_embedder is None:
                print(f"[INFO] 🧠 載入原生 Embedding 模型: {embed_model_name} (Device: {EMBED_DEVICE})")
                cls._shared_embedder = SentenceTransformer(embed_model_name, device=EMBED_DEVICE)
            return cls._shared_embedder

    def __init__(
        self,
        query: str,
        llm_instance,
        embed_model_name: str = "BAAI/bge-small-zh-v1.5",
        max_results: int = 5,
        base_directory: str = "vector_stores",
        use_existing_db: bool = False,
        tavily_api_key: Optional[str] = None,
        enable_hybrid_search: bool = True,
    ):
        self.query = query
        self.llm = llm_instance
        self.embedder = self.get_shared_embedder(embed_model_name)

        self.max_results = max_results
        self.use_existing_db = use_existing_db
        self.enable_hybrid_search = enable_hybrid_search

        self.tavily_api_key = (tavily_api_key or os.environ.get("TAVILY_API_KEY") or "").strip()
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key
            print("[INFO] 🔑 已就緒原生 TavilyClient API Key")
        else:
            print("[WARN] ⚠️ 未設定 Tavily API Key，若無本地知識庫將無法執行聯網搜尋")

        self.persist_directory = self._generate_db_path(query, base_directory)
        print(f"[INFO] 📁 向量資料庫路徑: {self.persist_directory}")

    def _generate_db_path(self, query: str, base_dir: str) -> str:
        """根據研究主題生成資料庫路徑 (支援模糊比對)"""
        clean_query = re.sub(r"[^\w\s\u4e00-\u9fff]", "", query, flags=re.UNICODE)
        clean_query = re.sub(r"\s+", "_", clean_query.strip())
        if len(clean_query) > 50:
            clean_query = clean_query[:50]

        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
        expected_db_name = f"{clean_query}_{query_hash}"
        full_path = os.path.join(base_dir, expected_db_name)

        if self.use_existing_db and not os.path.exists(full_path) and os.path.exists(base_dir):
            safe_prefix = clean_query + "_"
            candidates = [f for f in os.listdir(base_dir) if f.startswith(safe_prefix)]
            if candidates:
                fuzzy_path = os.path.join(base_dir, candidates[0])
                print(f"[INFO] 模糊比對成功！轉向使用現有資料庫: {candidates[0]}")
                return fuzzy_path

        return full_path

    def _get_chroma_collection(self) -> chromadb.Collection:
        """取得或建立原生 ChromaDB Collection（相容舊版 langchain collection 名稱）"""
        os.makedirs(self.persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=self.persist_directory)
        existing_cols = [c.name for c in client.list_collections()]
        if "langchain" in existing_cols:
            return client.get_collection(name="langchain")
        return client.get_or_create_collection(
            name="research_docs",
            metadata={"hnsw:space": "cosine"},
        )

    def load_existing_vectorstore(self) -> Optional[chromadb.Collection]:
        """載入現存的原生 ChromaDB Collection"""
        sqlite_path = os.path.join(self.persist_directory, "chroma.sqlite3")

        if not os.path.exists(sqlite_path):
            print(f"[WARN] 向量資料庫不存在: {self.persist_directory}")
            return None

        if os.path.getsize(sqlite_path) < 1024:
            print("[ERROR] 資料庫檔案損壞或為空")
            return None

        try:
            print(f"[INFO] 載入現存向量資料庫: {self.persist_directory}")
            collection = self._get_chroma_collection()
            doc_count = collection.count()
            if doc_count == 0:
                print("[WARN] 資料庫為空，準備重新搜尋")
                return None

            print(f"[INFO] ✅ 成功載入 {doc_count} 筆文檔")
            return collection
        except Exception as e:
            print(f"[ERROR] 載入失敗: {e}")
            return None

    def retrieve_context(self) -> str:
        """使用原生 TavilyClient 搜尋多語系資料"""
        print("[INFO] 🌐 使用原生 TavilyClient 執行多面向深度檢索中...")
        all_results = []
        seen_urls = set()

        search_queries = [self.query]
        try:
            query_en = GoogleTranslator(source="auto", target="en").translate(self.query)
            if query_en and query_en.strip().lower() != self.query.strip().lower():
                search_queries.append(query_en.strip())
        except Exception:
            pass

        per_query_limit = max(5, self.max_results)
        for q in search_queries:
            items = self.tavily_search(q, max_results=per_query_limit)
            for item in items:
                if isinstance(item, dict):
                    url = item.get("url") or ""
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    title = item.get("title") or ""
                    content = item.get("content") or ""
                    if content.strip():
                        all_results.append(f"標題：{title}\n來源：{url}\n內容：{content}")
                elif item:
                    all_results.append(str(item))

        print(f"[INFO] 共獲得 {len(all_results)} 筆不重複的搜尋結果。")
        return "\n\n".join(all_results)

    def tavily_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """執行原生 TavilyClient 檢索"""
        if not self.tavily_api_key:
            print("[ERROR] 缺少 Tavily API Key，無法執行搜尋")
            return []
        try:
            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(query=query, max_results=max_results)
            if isinstance(response, dict) and "results" in response:
                return response["results"]
            elif isinstance(response, list):
                return response
            return []
        except Exception as e:
            print(f"[ERROR] Tavily 搜尋失敗 ({query}): {e}")
            return []

    def split_chunks(self, text: str, chunk_size: int = 600, chunk_overlap: int = 80) -> List[str]:
        """純 Python 段落優先滑動視窗切分器（零 LangChain 依賴）"""
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
                    # 單一段落超過 chunk_size 時進行滑動視窗切分
                    step = max(1, chunk_size - chunk_overlap)
                    for start in range(0, len(para), step):
                        sub = para[start : start + chunk_size].strip()
                        if sub:
                            chunks.append(sub)
                    current = ""

        if current:
            chunks.append(current)

        return chunks

    def build_vectorstore(self, chunks: List[str]) -> chromadb.Collection:
        """建立或增量更新原生 ChromaDB Collection"""
        def hash_text(t: str) -> str:
            return hashlib.md5(t.encode("utf-8")).hexdigest()

        lock_path = os.path.join(self.persist_directory, "chroma.lock")
        os.makedirs(self.persist_directory, exist_ok=True)

        with FileLock(lock_path, timeout=15):
            collection = self._get_chroma_collection()
            existing_ids = set()
            try:
                existing_data = collection.get(include=[])
                existing_ids = set(existing_data.get("ids") or [])
            except Exception:
                pass

            new_texts: List[str] = []
            new_ids: List[str] = []
            seen_batch_ids = set()

            for chunk in chunks:
                cid = hash_text(chunk)
                if cid not in existing_ids and cid not in seen_batch_ids:
                    seen_batch_ids.add(cid)
                    new_texts.append(chunk)
                    new_ids.append(cid)

            if new_texts:
                embeddings = self.embedder.encode(new_texts, normalize_embeddings=True).tolist()
                collection.add(
                    documents=new_texts,
                    embeddings=embeddings,
                    ids=new_ids,
                )
                print(f"[INFO] ✅ 已寫入 {len(new_texts)} 筆新知識切片至 ChromaDB")

        return collection

    def _docs_to_context(self, docs: List[Any]) -> str:
        return "\n\n".join(
            [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs]
        )

    # ==================== 核心模型呼叫封裝 ====================

    def _call_model(self, system_text: str, user_text: str, llm_override=None) -> str:
        """使用本機 MLXChatModel 進行推論，並自動剝離思考標籤"""
        active_llm = llm_override or self.llm
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        raw_output = active_llm.invoke(messages)
        return _strip_thinking_tags(raw_output)

    def run_qa(self, retriever: Any, question: str) -> str:
        docs = retriever.invoke(question)
        context = self._docs_to_context(docs)
        user_prompt = (
            f"根據以下上下文回答問題。如果無法從上下文判斷，請說明「資訊不足」。\n\n"
            f"上下文:\n{context}\n\n問題: {question}\n回答:"
        )
        return self._call_model("你是專業的研究助手。", user_prompt)

    def generate_structured_outline(self, retriever: Any, topic: str) -> Dict[str, str]:
        """生成結構化大綱"""
        try:
            prompt_obj = PromptTemps.get_langgraph_outline_prompt()
            template_str = prompt_obj.template
        except Exception:
            template_str = (
                "請為以下主題生成結構化大綱，使用 markdown 格式，包含主要章節和小節：\n"
                "背景資料:\n{context}\n\n主題: {question}\n\n請提供詳細的目錄結構："
            )

        docs = retriever.invoke(topic)
        context = self._docs_to_context(docs)
        user_prompt = template_str.replace("{context}", context).replace("{question}", topic)

        result = self._call_model("你是專業的內容規劃師，請直接輸出符合格式的大綱。", user_prompt)
        return {"result": result}

    def run_content_generation(
        self,
        retriever: Any,
        question: str,
        llm_override=None,
        use_strict_mode: bool = False,
        heading_level: str = "###",
    ) -> str:
        """撰寫報告小節內容"""
        try:
            if use_strict_mode:
                prompt_obj = PromptTemps.get_content_prompt_strict()
            else:
                prompt_obj = PromptTemps.get_content_prompt()
            template_str = prompt_obj.template
        except Exception:
            template_str = (
                "請根據上下文撰寫 3 到 5 句完整段落。\n"
                "背景資料:\n{context}\n\n章節標題: {question}\n\n"
                "請直接輸出純段落內容："
            )

        docs = retriever.invoke(question)
        context = self._docs_to_context(docs)

        user_prompt = (
            template_str.replace("{context}", context)
            .replace("{question}", question)
            .replace("{heading_level}", heading_level)
        )
        return self._call_model(
            "你是專業的報告撰寫助手，請使用繁體中文直接輸出段落內文。",
            user_prompt,
            llm_override=llm_override,
        )

    def execute(self) -> Optional[_NativeHybridRetriever]:
        """執行完整檢索管線並回傳原生混合檢索器"""
        collection = None
        if self.use_existing_db:
            collection = self.load_existing_vectorstore()
            if collection is None:
                self.use_existing_db = False

        if collection is None:
            context = self.retrieve_context()
            chunks = self.split_chunks(context)
            if not chunks:
                return None
            collection = self.build_vectorstore(chunks)

        raw_data = collection.get(include=["documents"])
        all_docs = raw_data.get("documents") or []
        if not all_docs:
            return None

        return _NativeHybridRetriever(
            collection=collection,
            embedder=self.embedder,
            all_docs=all_docs,
            top_k=self.max_results,
            enable_hybrid=self.enable_hybrid_search,
        )