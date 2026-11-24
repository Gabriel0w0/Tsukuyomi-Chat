# -*- coding: utf-8 -*-
"""
跨平台本機模型推論引擎 (macOS MLX / Windows & Linux CUDA 雙引擎自動切換)
對外維持統一的 MLXChatModel 介面，支援單一常駐 GPU 執行緒與即時逐字串流
"""

import gc
import sys
import queue
import logging
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any, Union, Iterator

import config

logger = logging.getLogger("mlx_engine")

MEMORY_MODEL_ID = getattr(config, "MEMORY_MODEL_ID", "mlx-community/gemma-3-1b-it-4bit")

STOP_STRINGS = ("<end_of_turn>", "<eos>", "<|eot_id|>", "<|im_end|>", "</s>")

# ==================== 1. 自動偵測硬體與推論後端 ====================
IS_APPLE_SILICON = (sys.platform == "darwin" and platform.machine() == "arm64")
BACKEND = "none"

if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        from mlx_lm import (
            load as mlx_load,
            generate as mlx_generate,
            stream_generate as mlx_stream_generate,
        )
        from mlx_lm.sample_utils import make_sampler

        BACKEND = "mlx"
        logger.info("🍎 偵測到 Apple Silicon，啟用 MLX Metal 推論引擎")
    except ImportError:
        logger.warning("未安裝 mlx-lm，嘗試切換至 PyTorch 後端...")

if BACKEND == "none":
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
            BitsAndBytesConfig,
        )

        BACKEND = "torch"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        logger.info(f"🐧🪟 啟用 PyTorch Transformers 推論引擎 (硬體: {device_name})")
    except ImportError as e:
        raise RuntimeError(
            "找不到可用的推論引擎！macOS 請安裝 `mlx-lm`，"
            "Windows/Linux 請安裝 `torch transformers accelerate bitsandbytes`。"
        ) from e


# ==================== 2. MLX 模型名稱自動轉譯 (供 CUDA / CPU 使用) ====================
MLX_TO_HF_MAP = {
    "mlx-community/gemma-3-1b-it-4bit": "unsloth/gemma-3-1b-it-bnb-4bit",
    "mlx-community/gemma-3-4b-it-8bit": "unsloth/gemma-3-4b-it-bnb-4bit",
    "mlx-community/gemma-3-4b-it-4bit": "unsloth/gemma-3-4b-it-bnb-4bit",
    "mlx-community/gemma-3-12b-it-4bit": "unsloth/gemma-3-12b-it-bnb-4bit",
    "mlx-community/gemma-4-e4b-it-4bit": "google/gemma-4-E4B-it",
}


def resolve_model_id_for_backend(model_id: str) -> str:
    """若在 Windows/Linux (CUDA) 收到 mlx-community 專用權重名稱，自動轉為標準 HuggingFace 權重"""
    if BACKEND == "mlx":
        return model_id
    if model_id in MLX_TO_HF_MAP:
        mapped = MLX_TO_HF_MAP[model_id]
        logger.info(f"🔄 [CUDA 相容映射] 將 MLX 模型 '{model_id}' 自動映射為 '{mapped}'")
        return mapped
    return model_id


# ==================== 3. 跨平台統一模型封裝類別 ====================
class MLXChatModel:
    # 全程序唯一的單一常駐 GPU 執行緒，確保 MLX Stream 與 CUDA Context 執行緒安全
    _gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GPU_Worker")
    _gpu_lock = threading.Lock()
    _gpu_stream: Optional[Any] = None

    @classmethod
    def _ensure_mlx_stream(cls):
        """在專用 GPU 執行緒內取得或建立固定的 Metal GPU Stream"""
        if cls._gpu_stream is None and BACKEND == "mlx":
            cls._gpu_stream = mx.default_stream(mx.gpu)
        return cls._gpu_stream

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.35,
        top_p: float = 0.85,
        max_tokens: int = 4096,
    ):
        self.model_name_or_path = model_name_or_path
        self.actual_model_id = resolve_model_id_for_backend(model_name_or_path)
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.max_tokens = max_tokens

        logger.info(f"正在載入模型 [{BACKEND.upper()}]: {self.actual_model_id}")

        def _load_in_worker():
            if BACKEND == "mlx":
                stream = self._ensure_mlx_stream()
                with mx.stream(stream):
                    return mlx_load(self.actual_model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.actual_model_id)
                load_kwargs: Dict[str, Any] = {"device_map": "auto"}

                if torch.cuda.is_available():
                    if "bnb" not in self.actual_model_id.lower():
                        if "4bit" in model_name_or_path.lower():
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4",
                            )
                        elif "8bit" in model_name_or_path.lower():
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True
                            )
                        else:
                            load_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = torch.float32

                model = AutoModelForCausalLM.from_pretrained(
                    self.actual_model_id, **load_kwargs
                )
                model.eval()
                return model, tokenizer

        self.model, self.tokenizer = self._gpu_executor.submit(_load_in_worker).result()

        # 強制將 <end_of_turn> 等對話結束標記註冊進 Tokenizer 的 EOS 停止清單
        self._stop_token_ids = set()
        raw_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        unk_id = getattr(raw_tokenizer, "unk_token_id", None)

        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            self._stop_token_ids.add(self.tokenizer.eos_token_id)

        for stop_str in STOP_STRINGS:
            try:
                # 1. 若為 MLX TokenizerWrapper，優先呼叫原生 add_eos_token
                if hasattr(self.tokenizer, "add_eos_token"):
                    self.tokenizer.add_eos_token(stop_str)
            except Exception:
                pass

            try:
                # 2. 透過 convert_tokens_to_ids 直接查詢 Special Token ID (最準確)
                if hasattr(raw_tokenizer, "convert_tokens_to_ids"):
                    tid = raw_tokenizer.convert_tokens_to_ids(stop_str)
                    if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                        self._stop_token_ids.add(tid)
            except Exception:
                pass

            try:
                # 3. 備援：透過 encode 查詢
                ids = raw_tokenizer.encode(stop_str, add_special_tokens=False)
                if len(ids) == 1 and ids[0] != unk_id:
                    self._stop_token_ids.add(ids[0])
            except Exception:
                pass

        # 同步寫入 MLX TokenizerWrapper 內部的 eos_token_ids 集合
        for attr_name in ("_eos_token_ids", "eos_token_ids"):
            eos_set = getattr(self.tokenizer, attr_name, None)
            if isinstance(eos_set, set):
                eos_set.update(self._stop_token_ids)

        logger.info(
            f"✅ 模型載入完成 [{BACKEND.upper()}]: {self.actual_model_id} (Stop IDs: {self._stop_token_ids})"
        )

    def unload(self):
        """主動斷開底層模型權重與 Tokenizer 參照，確保即使外部仍持有實例也能立即釋放顯存"""
        logger.info(f"🧹 正在卸載模型權重並釋放記憶體: {self.actual_model_id}")
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _truncate_at_stop_strings(text: str) -> str:
        """一旦出現對話結束控制符，立即截斷後續所有腦補內容與重複標籤"""
        for stop_tok in STOP_STRINGS:
            if stop_tok in text:
                text = text.split(stop_tok)[0]
        for clean_tok in ("<bos>", "<pad>"):
            text = text.replace(clean_tok, "")
        return text

    def _format_messages(
        self, messages: Union[str, List[Dict[str, str]], List[Any]]
    ) -> str:
        """將對話訊息統一轉換為模型 Tokenizer 支援的 Chat Template 字串"""
        if isinstance(messages, str):
            chat_msgs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            chat_msgs = []
            for m in messages:
                if isinstance(m, dict):
                    chat_msgs.append(
                        {"role": m.get("role", "user"), "content": m.get("content", "")}
                    )
                elif hasattr(m, "content"):
                    role_type = getattr(m, "type", "user")
                    role_map = {"system": "system", "human": "user", "ai": "assistant"}
                    chat_msgs.append(
                        {"role": role_map.get(role_type, "user"), "content": m.content}
                    )
                else:
                    chat_msgs.append({"role": "user", "content": str(m)})
        else:
            chat_msgs = [{"role": "user", "content": str(messages)}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    chat_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                merged_msgs = []
                sys_content = ""
                for msg in chat_msgs:
                    if msg["role"] == "system":
                        sys_content += msg["content"] + "\n\n"
                    elif msg["role"] == "user" and sys_content:
                        merged_msgs.append(
                            {"role": "user", "content": sys_content + msg["content"]}
                        )
                        sys_content = ""
                    else:
                        merged_msgs.append(msg)
                if sys_content:
                    merged_msgs.append({"role": "user", "content": sys_content})
                return self.tokenizer.apply_chat_template(
                    merged_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in chat_msgs)
            + "\nassistant: "
        )

    def invoke(
        self,
        messages: Union[str, List[Dict[str, str]], List[Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """同步單次推論（自動適配 MLX 與 PyTorch CUDA，並在 <end_of_turn> 精準停止）"""
        prompt = self._format_messages(messages)
        temp = self.default_temperature if temperature is None else temperature
        tp = self.default_top_p if top_p is None else top_p
        tokens_limit = max_tokens or self.max_tokens

        def _run_generate():
            with self._gpu_lock:
                if BACKEND == "mlx":
                    stream = self._ensure_mlx_stream()
                    with mx.stream(stream):
                        sampler = make_sampler(temp=temp, top_p=tp)
                        accumulated = []
                        tail_window = ""
                        for response in mlx_stream_generate(
                            self.model,
                            self.tokenizer,
                            prompt=prompt,
                            max_tokens=tokens_limit,
                            sampler=sampler,
                        ):
                            if hasattr(response, "token") and response.token in self._stop_token_ids:
                                break
                            chunk = response.text if hasattr(response, "text") else str(response)
                            if not chunk:
                                continue
                            accumulated.append(chunk)
                            tail_window = (tail_window + chunk)[-40:]
                            if any(s in tail_window for s in STOP_STRINGS):
                                break
                        mx.eval()
                        full_text = "".join(accumulated)
                        return self._truncate_at_stop_strings(full_text).strip()
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(
                        self.model.device
                    )
                    input_len = inputs["input_ids"].shape[1]
                    eos_ids = (
                        list(self._stop_token_ids)
                        if self._stop_token_ids
                        else [self.tokenizer.eos_token_id]
                    )
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=tokens_limit,
                            temperature=max(temp, 1e-4),
                            top_p=tp,
                            do_sample=(temp > 0),
                            eos_token_id=eos_ids,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    generated_ids = outputs[0][input_len:]
                    raw_out = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=False
                    )
                    return self._truncate_at_stop_strings(raw_out).strip()

        return self._gpu_executor.submit(_run_generate).result()

    def stream_invoke(
        self,
        messages: Union[str, List[Dict[str, str]], List[Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """跨平台逐字串流推論（遇到 <end_of_turn> 立即終止串流）"""
        prompt = self._format_messages(messages)
        temp = self.default_temperature if temperature is None else temperature
        tp = self.default_top_p if top_p is None else top_p
        tokens_limit = max_tokens or self.max_tokens

        token_queue: queue.Queue = queue.Queue()

        def _run_stream_in_worker():
            try:
                with self._gpu_lock:
                    if BACKEND == "mlx":
                        stream = self._ensure_mlx_stream()
                        with mx.stream(stream):
                            sampler = make_sampler(temp=temp, top_p=tp)
                            tail_window = ""
                            for response in mlx_stream_generate(
                                self.model,
                                self.tokenizer,
                                prompt=prompt,
                                max_tokens=tokens_limit,
                                sampler=sampler,
                            ):
                                if hasattr(response, "token") and response.token in self._stop_token_ids:
                                    break
                                chunk = response.text if hasattr(response, "text") else str(response)
                                if not chunk:
                                    continue
                                tail_window = (tail_window + chunk)[-40:]
                                if any(s in tail_window for s in STOP_STRINGS):
                                    clean_piece = self._truncate_at_stop_strings(chunk)
                                    if clean_piece:
                                        token_queue.put(("token", str(clean_piece)))
                                    break
                                token_queue.put(("token", str(chunk)))
                            mx.eval()
                    else:
                        inputs = self.tokenizer(prompt, return_tensors="pt").to(
                            self.model.device
                        )
                        streamer = TextIteratorStreamer(
                            self.tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=False,
                        )
                        eos_ids = (
                            list(self._stop_token_ids)
                            if self._stop_token_ids
                            else [self.tokenizer.eos_token_id]
                        )
                        gen_kwargs = dict(
                            **inputs,
                            streamer=streamer,
                            max_new_tokens=tokens_limit,
                            temperature=max(temp, 1e-4),
                            top_p=tp,
                            do_sample=(temp > 0),
                            eos_token_id=eos_ids,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                        def _torch_gen():
                            with torch.no_grad():
                                self.model.generate(**gen_kwargs)

                        gen_thread = threading.Thread(target=_torch_gen, daemon=True)
                        gen_thread.start()
                        tail_window = ""
                        for chunk in streamer:
                            tail_window = (tail_window + chunk)[-40:]
                            if any(s in tail_window for s in STOP_STRINGS):
                                clean_piece = self._truncate_at_stop_strings(chunk)
                                if clean_piece:
                                    token_queue.put(("token", str(clean_piece)))
                                break
                            cleaned_chunk = chunk.replace("<bos>", "").replace("<pad>", "")
                            if cleaned_chunk:
                                token_queue.put(("token", str(cleaned_chunk)))
                        gen_thread.join(timeout=2.0)
            except Exception as e:
                logger.error(f"[{BACKEND.upper()} Worker] 串流推論發生錯誤: {e}", exc_info=True)
                token_queue.put(("error", str(e)))
            finally:
                token_queue.put(("eof", ""))

        self._gpu_executor.submit(_run_stream_in_worker)

        while True:
            kind, payload = token_queue.get()
            if kind == "eof":
                break
            elif kind == "error":
                raise RuntimeError(payload)
            elif kind == "token":
                yield payload


# ==================== 4. 記憶體清理與全域單例管理 ====================
def _clear_gpu_memory():
    """跨平台釋放 VRAM (CUDA) 或 Unified Memory (Metal)"""
    gc.collect()
    if BACKEND == "mlx":
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        elif hasattr(mx, "clear_cache"):
            mx.clear_cache()
    elif BACKEND == "torch":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


_shared_llm_instance: Optional[MLXChatModel] = None
_instance_lock = threading.Lock()

_memory_llm_instance: Optional[MLXChatModel] = None
_memory_lock = threading.Lock()


def get_shared_llm(
    model_name_or_path: Optional[str] = None,
    temperature: float = 0.35,
    top_p: float = 0.85,
    max_tokens: int = 4096,
) -> MLXChatModel:
    """取得或動態切換全域共用的主模型實例（嚴格保證先卸載舊權重、再載入新模型）"""
    global _shared_llm_instance
    with _instance_lock:
        if not model_name_or_path:
            from config import CHAT_MODEL

            model_name_or_path = CHAT_MODEL

        if (
            _shared_llm_instance is None
            or _shared_llm_instance.model_name_or_path != model_name_or_path
        ):
            if _shared_llm_instance is not None:
                logger.info(
                    f"🔄 動態切換模型: {_shared_llm_instance.model_name_or_path} -> {model_name_or_path}"
                )
                old_instance = _shared_llm_instance
                _shared_llm_instance = None

                def _cleanup_in_worker():
                    nonlocal old_instance
                    with MLXChatModel._gpu_lock:
                        old_instance.unload()
                        del old_instance
                        _clear_gpu_memory()

                MLXChatModel._gpu_executor.submit(_cleanup_in_worker).result()

            _shared_llm_instance = MLXChatModel(
                model_name_or_path=model_name_or_path,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        return _shared_llm_instance


def get_memory_llm(model_name_or_path: Optional[str] = None) -> MLXChatModel:
    """取得或動態切換常駐的輕量級輔助模型 (預設 gemma-3-1b-it-4bit)"""
    global _memory_llm_instance
    target_model = (model_name_or_path or MEMORY_MODEL_ID).strip()

    with _memory_lock:
        if (
            _memory_llm_instance is None
            or _memory_llm_instance.model_name_or_path != target_model
        ):
            if _memory_llm_instance is not None:
                logger.info(
                    f"🔄 切換記憶輔助模型: {_memory_llm_instance.model_name_or_path} -> {target_model}"
                )
                old_mem = _memory_llm_instance
                _memory_llm_instance = None

                def _cleanup_mem():
                    nonlocal old_mem
                    with MLXChatModel._gpu_lock:
                        old_mem.unload()
                        del old_mem
                        _clear_gpu_memory()

                MLXChatModel._gpu_executor.submit(_cleanup_mem).result()

            _memory_llm_instance = MLXChatModel(
                model_name_or_path=target_model,
                temperature=0.2,
                top_p=0.85,
                max_tokens=512,
            )
        return _memory_llm_instance