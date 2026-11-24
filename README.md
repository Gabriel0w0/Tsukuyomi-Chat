# 🌙 Tsukuyomi Chat (月讀醬) — An AI Companion with Personality, Memory, and Knowledge.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.143%2B-009688?logo=fastapi&logoColor=white)
![Apple Silicon MLX](https://img.shields.io/badge/Apple_Silicon-MLX_Metal-black?logo=apple&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_BitsAndBytes-76B900?logo=nvidia&logoColor=white)
![Live2D](https://img.shields.io/badge/Live2D-Cubism_4-ff69b4)

An interactive, emotionally expressive virtual desktop companion and local AI research assistant. **Tsukuyomi-chan (月讀醬)** seamlessly integrates **Live2D animation**, **real-time token streaming with Chain-of-Thought (CoT) folding**, **dual-model memory & voice synthesis**, **cross-platform GPU acceleration (macOS MLX & Windows/Linux CUDA)**, and a **Hybrid Local/Web RAG engine** into a unified glassmorphism web experience.

---

## ✨ Key Features

* **💬 Expressive Maid Persona & Intimacy System**:
  * Engages in natural Traditional Chinese dialogue with real-time emotional awareness (`[emotion:tag]`) driving Live2D facial expressions and motions.
  * Dynamic affection progression (`冷淡期` → `普通期` → `親密期` → `羈絆期`) with exponential smoothing and persistent SQLite storage.
* **🚀 Cross-Platform Dual Inference Engine (`mlx_engine.py`)**:
  * **macOS (Apple Silicon)**: Native **MLX (`mlx-lm`)** acceleration running on a dedicated single-thread Metal GPU worker (`ThreadPoolExecutor(max_workers=1)`) with explicit stream binding—completely eliminating cross-thread Metal stream crashes.
  * **Windows / Linux (NVIDIA CUDA)**: Automatic fallback to **PyTorch `transformers`** + **`bitsandbytes` (4-bit NF4 / 8-bit)** quantization and `TextIteratorStreamer`, featuring automatic model ID mapping from `mlx-community/*` to standard HuggingFace repositories.
  * **Zero-Leak Hot-Swapping**: Switch between models (e.g., *Gemma 3 4B*, *Gemma 4 E4B*, *Gemma 3 12B*) directly from the UI with explicit weight unloading and VRAM/Unified Memory cache clearing.
* **🧠 Dual-Model Architecture & Memory Anchor**:
  * **Main Chat / Research LLM**: Handles persona dialogue, reasoning (`<|channel>thought` / `<think>`), code review, and report synthesis.
  * **Dedicated 1B Auxiliary LLM (`gemma-3-1b-it-4bit`)**: Stays lightweight in the background to perform ultra-fast **Long-Term Memory JSON extraction** and **Japanese TTS translation** without thinking-token overhead or main model interruption.
  * **Memory Anchor in RAG/Search**: Automatically pins user memory profiles to the tail of long retrieval contexts so Tsukuyomi never forgets personal facts even during heavy web/document searches.
* **⚡ Real-Time NDJSON Token Streaming**:
  * Live character-by-character output via `/chat/stream` with real-time separation of collapsible reasoning blocks (`月讀醬的深度思考過程`) and clean spoken dialogue.
* **📎 Hybrid Local RAG for Code & Academic Papers (`local_file_rag.py`)**:
  * **Line & Page Precision**: Splits C/C++/Python source code with exact line intervals (`Lines X-Y`) and parses PDF papers with page citations (`Page P`).
  * **Ensemble Retrieval**: Combines **BM25** (0.45 weight, custom C++ snake_case + English + Jieba tokenizer) and dense **BGE-Small (`BAAI/bge-small-zh-v1.5`)** vector embeddings (0.55 weight) via ChromaDB.
* **🌐 Live Web Search & Autonomous Research Studio**:
  * One-click **Tavily Web Search** toggle in chat with smart bypass for personal memory queries.
  * Dedicated **Research Mode** for automated multi-section report generation, progress tracking, and Markdown/JSON/TXT export.
* **🗣️ Japanese Voice Synthesis (VOICEVOX)**:
  * Translates Tsukuyomi's replies into cute Japanese maid dialogue via the 1B auxiliary model and synthesizes voice audio using local **VOICEVOX** (*Catshibi / 猫使ビィ*).

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Browser]) <-->|HTTP / NDJSON Stream| API[FastAPI Backend - app.py]

    subgraph "Frontend Layer (Glassmorphism UI)"
        UI[Chat & Research UI] <--> Live2D[Live2D Cubism 4 - Hiyori]
        UI -->|Real-time CoT Parser| CoT[Collapsible Thinking Box]
        UI -->|File / Folder Upload| LocalDrop[Local RAG Panel]
        UI -->|Hot-Swap & Sliders| Settings[Dynamic Settings Modal]
    end

    subgraph "Cross-Platform Dual LLM Engine (mlx_engine.py)"
        API <--> Worker[Dedicated GPU Worker Thread]
        Worker --> MainLLM[Main LLM: Gemma 3 4B / Gemma 4 E4B / 12B]
        Worker --> AuxLLM[Auxiliary 1B LLM: Gemma 3 1B 4-bit]
        MainLLM <--> Hardware[(macOS MLX Metal / Linux & Win CUDA)]
        AuxLLM <--> Hardware
    end

    subgraph "Memory & Voice Pipeline"
        API -->|Background Task| AuxLLM
        AuxLLM -->|raw_decode JSON| SQLite[(SQLite: user_memory & user_settings)]
        AuxLLM -->|Fast JP Translation| TTS[VOICEVOX Engine :50021]
        TTS -->|WAV Audio| UI
    end

    subgraph "Hybrid RAG & Web Search Engine"
        API <--> LocalRAG[Hybrid Local RAG - local_file_rag.py]
        LocalRAG --> Chroma[(ChromaDB - BGE-Small-zh-v1.5)]
        LocalRAG --> BM25[(BM25 - Code / LaTeX / Jieba)]
        API <--> Tavily[Tavily Search API]
        API <--> Research[Autonomous Researcher - service.py]
    end
```

---

## 📋 Prerequisites

* **Operating System**:
  * **macOS**: Apple Silicon (M1/M2/M3/M4) for native MLX Metal acceleration.
  * **Windows / Linux**: NVIDIA GPU with CUDA 12.x support (or CPU fallback).
* **Python**: `3.10` or higher (recommended to manage via `uv` or `venv`).
* **VOICEVOX Engine** (Optional, required for Japanese TTS voice playback).
* **Tavily API Key** (Optional, required for real-time web search and autonomous research).

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/Gabriel0w0/tsukuyomi-chat.git](https://github.com/Gabriel0w0/tsukuyomi-chat.git)
cd tsukuyomi-chat
```

### 2. Install Dependencies

#### Option A: macOS (Apple Silicon — MLX) recommend
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### Option B: Windows / Linux (NVIDIA CUDA — PyTorch + BitsAndBytes) Not yet tested
```bash
python -m venv .venv
# Activate on Linux/macOS: source .venv/bin/activate
# Activate on Windows (PowerShell): .\.venv\Scripts\Activate.ps1

# 1. Upgrade pip and install core build tools first
python -m pip install --upgrade pip setuptools wheel

# 2. Install remaining dependencies from requirements.txt first
python -m pip install -r requirements.txt

# 3. Force-install PyTorch with NVIDIA CUDA support (prevents CPU-only overwrite)
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126) --no-deps

# 4. Verify GPU detection (should print: True | NVIDIA GeForce ...)
python -c "import torch; print(torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

### 3. Start VOICEVOX Engine (Optional, for TTS)

Run the VOICEVOX engine locally via Docker (or launch the desktop VOICEVOX application):

```bash
# CPU / macOS / Windows / Linux:
docker run -d --name voicevox_engine -p 50021:50021 voicevox/voicevox_engine:latest

# NVIDIA GPU (Linux / Windows):
docker run -d --gpus all --name voicevox_engine -p 50021:50021 voicevox/voicevox_engine:nvidia-latest
```

### 4. Configuration (`config.py`)

Review `config.py` for default parameters (all can be dynamically adjusted later in the Web UI Settings):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `CHAT_MODEL` | `"mlx-community/gemma-3-4b-it-8bit"` | Default model for daily chat (auto-maps to HF on CUDA) |
| `RESEARCH_MODEL` | `"mlx-community/gemma-3-12b-it-4bit"` | Model for in-depth research report synthesis |
| `MEMORY_MODEL_ID` | `"mlx-community/gemma-3-1b-it-4bit"` | Lightweight 1B model for memory JSON extraction & TTS translation |
| `MAX_TOKENS` | `8192` | Maximum generation / context window tokens |
| `MAX_MEMORY` | `10` | Short-term conversation turns sent to the LLM |
| `MAX_FACTS_PER_USER` | `20` | Maximum long-term user facts stored in SQLite |

---

## ▶️ Usage

Start the FastAPI server with `uvicorn`:

```bash
uvicorn app:app --host 127.0.0.1 --port 5000 --reload
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### Core Workflows:

1. **Interactive Chat & Quick Model Switcher (`Chat`)**:
   * Chat with Tsukuyomi-chan with real-time token streaming.
   * Click the **Model Badge** above the input bar to hot-swap between *Gemma 3 (4B)*, *Gemma 4 (E4B)*, or any custom HuggingFace/local model ID.
   * Click the **Globe (🌐)** button to enable live web search while retaining your personal memory profile.
2. **Code & Paper Hybrid RAG (`📎 Paperclip`)**:
   * Upload individual `.pdf`, `.cpp`, `.h`, `.py`, `.tex`, `.md` files or an entire project folder.
   * Tsukuyomi will analyze your code or paper and cite exact **line numbers (`Lines X-Y`)** or **page numbers (`Page P`)**.
3. **Autonomous Research Studio (`Research`)**:
   * Enter a research topic to automatically plan an outline, crawl web sources, and compile a structured report with Markdown/JSON export.
4. **Personalized Settings (`⚙️ Settings`)**:
   * Adjust theme, font size, Tavily API Key, `CHAT_MODEL`, `RESEARCH_MODEL`, `MEMORY_MODEL_ID`, `MAX_TOKENS`, `MAX_MEMORY`, and `MAX_FACTS_PER_USER` in real time.

---

## 🎭 Emotion Tag Protocol

Tsukuyomi prefixes her responses with an emotion tag (automatically stripped from the chat bubble and used to trigger Live2D motions and intimacy weights):

| Emotion Tag | Live2D Expression | Intimacy Weight | Behavior Description |
| :--- | :--- | :---: | :--- |
| `[emotion:joy]` | Happy / Laugh | `+1` | Positive feedback, excitement, warm encouragement |
| `[emotion:cute]` | Winking / Playful | `+1` | Sweet, coy, or affectionate maid remarks |
| `[emotion:shy]` | Blushing | `+2` | Embarrassment or deep affection |
| `[emotion:neutral]` | Calm / Attentive | `0` | Analytical, technical, or composed explanation |
| `[emotion:sad]` | Tearful / Sad | `-1` | Empathy, worry, or feeling sorry |
| `[emotion:angry]` | Pouting / Angry | `-2` | Playful tsundere pouting or displeasure |

---

## ⚖️ Legal & Acknowledgments

1. **VOICEVOX**: Voice synthesis powered by **VOICEVOX** using the **Catshibi (猫使ビィ)** voicebank, subject to the [VOICEVOX Terms of Use](https://voicevox.hiroshiba.jp/term/).
2. **Live2D Model**: The sample model **"Hiyori Momose (桃瀬日和)"** is property of Live2D Inc. and is used strictly for **non-commercial, personal learning, and academic research purposes** under the [Live2D Free Material License](https://www.live2d.com/eula/live2d-sample-model-terms_en.html).
3. **Open-Source Ecosystem**: Powered by [MLX](https://github.com/ml-explore/mlx), [HuggingFace Transformers](https://github.com/huggingface/transformers), [LangChain](https://github.com/langchain-ai/langchain), [ChromaDB](https://www.trychroma.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Tavily](https://tavily.com/).

---

> **Disclaimer**: This project is developed strictly for academic research and personal educational purposes. Any commercial distribution or unauthorized use of third-party assets is prohibited.

