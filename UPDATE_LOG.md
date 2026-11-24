# Update Log

*-**Miku** (Kawaii)-* 
## 2026-11-24 (Version 2.3.0)

### Structured Long-Term Memory Management System (`memory_manager.py`)

- **Storage (Dual-track storage and automatic migration)**: Added `MemorySystemManager`, upgrading the original flat JSON memory array to an SQLite structured master table (`memory_items`, recording categories `core` / `preference` / `episodic`, importance, timestamp, and access count) paired with ChromaDB semantic vector indexes. Reuses the existing `BGE-Small` singleton and supports seamless automatic migration of the old `user_memory.facts`.

- **Retrieval (Hierarchical and composite rating recall)**: Changed to "Core profile persistence + hybrid retrieval (BGE + BM25) dynamic recall," combining relevance, time decay, and importance for composite sorting, and automatically enhancing memory activity during recall.

- **Update (Semantic Conflict Resolution State Machine)**: Background memory updates are upgraded to a two-stage pipeline (candidate extraction → semantic comparison → `ADD` / `UPDATE` / `DELETE` / `NOOP` decision), replacing the original simple string comparison deduplication and FIFO pruning, and intelligently eliminating based on importance and activity.

- **Generation (Hierarchical Context Injection)**: The System Prompt distinguishes between `[Core User Profile]` and `[Relevant Recalled Memories]` with relative time stamps, and adds `[Format Examples]` to prevent model sentiment label format drift.

- **Memory Inspector**: A new `/api/memory` CRUD endpoint and a front-end frosted glass pop-up window have been added, supporting real-time viewing, manual addition, single-entry editing, and deletion of long-term memories.

### Streaming Interruption and Online Search Optimization (`app.py`)

- **Stop Output Functionality:** The backend `/chat/stream` now includes a `threading.Event` interrupt flag and disconnection detection. The frontend, equipped with `AbortController`, switches the send button to a "Stop" button during generation, instantly interrupting MLX GPU inference, skipping TTS synthesis, and fully preserving generated segment content.

- **Improved Tavily Online Search Accuracy:** Keyword purification rules have been added for Chinese questions, using `search_depth="advanced"`, `include_answer=True`, and a relevance score threshold (`score >= 0.42`) to significantly reduce irrelevant news snippets that mistakenly search for single characters.

### Front-end Interface and Interactive Experience (`index.html`, `research_script.js`, `research_style.css`)

- **Chat Input Bar Integration**: The quick model switcher (`chat-model-badge`) has been integrated into the `.input-toolbar` toolbar, resolving the layout issue where the floating capsule obscured elements above.

- **Right-side Local RAG Panel File Selection**: The individual enable/disable switch for knowledge base files and the "Select All/Cancel All" toolbar have been integrated into the right-side `Local RAG` panel, keeping the chat input area below clean.

- **Site-wide Custom Frosted Glass Pop-up**: `showCustomPrompt` and `showCustomConfirm` have been implemented to completely replace the browser's native `prompt()` and `confirm()`, integrating the Chinese input method (IME) anti-accidental touch mechanism and visual warnings for dangerous operations.

- **Dialogue Loading Prompt Fine-tuning**: The three-dot wave animation has been retained, and a purple-pink breathing light effect has been added. 
### Bug Fixes and Code Cleanup

- Fixed missing initialization and upload methods in `FileManager`, and an unclosed `.container` tag structure issue in `index.html`.

- Expanded the front-end `EMOTION_TAG_REGEX` and `TRAILING_PARTIAL_TAG_REGEX` filtering rules to intercept occasional non-standard sentiment tags created by the model (such as `[cute:blush]`) and half-finished characters in the stream.

- Cleaned up old memory functions and constants (`append_memory`, `extract_facts_from_llm`, `PERSONAL_FACT_HINTS`) that are no longer used after `app.py` was transferred to `memory_manager.py`, and increased `max_tokens` for background JSON extraction to `384`.

## 2026-11-15 (Version 2.2.0)

### Architecture Refactoring (Zero-LangChain Migration)

  - **Complete Removal of LangChain & LangGraph Dependencies**
      - Decoupled the entire codebase (`app.py`, `local_file_rag.py`, `researcher/rag_pipeline.py`, `researcher/prompts.py`, and `requirements.txt`) from `langchain`, `langchain-core`, `langchain-community`, `langchain-classic`, `langchain-chroma`, `langchain-huggingface`, `langchain-tavily`, and `langgraph`.
      - Eliminated version-specific import path conflicts and Pylance missing-import warnings while significantly reducing startup latency and memory overhead.
  - **Pure Native RAG & Search Pipelines**
      - **Web Search**: Migrated both chat web search and deep research retrieval to the official `tavily-python` (`TavilyClient`) SDK, resolving Pydantic validation errors caused by deprecated wrapper parameters.
      - **Hybrid Retrieval (RRF)**: Replaced `EnsembleRetriever` with a native `_NativeHybridRetriever` combining `chromadb` dense vector search (`SentenceTransformer`) and `rank-bm25` (`BM25Okapi` + `jieba` + C++ identifier tokenizer) via **Reciprocal Rank Fusion (RRF)**.
      - **Text Chunking & Prompt Templates**: Implemented pure Python paragraph-aware sliding-window chunkers and a lightweight `PromptTemplate` dataclass while maintaining backward compatibility with existing ChromaDB collections.
  - **Reasoning Tag Filtering & Hardware Acceleration**
      - Added `_strip_thinking_tags()` to `rag_pipeline.py` to strip `<|channel>thought` and `<think>` blocks prior to outline parsing and report generation.
      - Unified shared `SentenceTransformer` singleton loading with automatic device selection (`cuda` → `mps` → `cpu`).

### UI/UX & Frontend Enhancements

  - **Left-Aligned `+` Attachment Button & Quick Upload Menu**
      - Relocated the file attachment trigger to the left side of `#chat-input-area` as a circular **`+`** button with an active file count badge.
      - Added an upward-popping **Quick Attachment Menu** (`#attachQuickMenu`) with smooth `+` to `×` rotation animation, supporting direct file upload, entire project folder upload, and Local RAG panel toggling, plus click-outside auto-collapse.
  - **Zero-Latency Intimacy Display Centering**
      - Refactored `#intimacy-display` from JavaScript `getBoundingClientRect()` coordinate recalculation to pure CSS container-relative centering (`position: absolute; left: 50%; transform: translateX(-50%)`), eliminating the positioning lag when re-expanding the Live2D panel.
  - **Smooth Chat Input Loading Animation**
      - Replaced the rigid spinner icon on `#sendBtn` with a soft purple-pink **breathing glow border** (`#chat-input-area.is-loading`) and a three-dot bouncing wave indicator.

### Backend Stability & Persistence

  - **Settings & Tavily API Key Protection**
      - Upgraded `UserSettings` to Pydantic V2 `AliasChoices` and added partial-update safeguards in `POST /api/settings` so switching models no longer overwrites stored `tavily_api_key` values.
      - Added cross-session fallback key lookup across `app.py` and `researcher/service.py` so valid API keys persist seamlessly across new browser sessions.
  - **Concurrency, Stream Disconnect & Memory Management**
      - Enabled SQLite **WAL mode** (`PRAGMA journal_mode=WAL;`) across database connections for improved read/write concurrency.
      - Added `req.is_disconnected()` checks in `/chat/stream` to immediately halt token generation and skip redundant TTS synthesis when a client disconnects.
      - Enabled automatic background cleanup (`cleanup_progress`) for finished research jobs in `researcher/service.py` and hardened legacy Ollama model string sanitization while preserving Windows drive paths.

## 2026-10-19 (Version 2.1.1)

  - **Bug Fixes**
    - Fixed some bugs.
  
  - **Feature improvements**
    - Improved when switching models, new weights are loaded before the old weights are unloaded.
    - Improve issues related to MLX synchronized output tokens.

## 2026-10-18

  - **Bug Fixes**
    - Fixed some bugs.

## 2026-10-17

  - **Bug Fixes**
    - Fixed some bugs in app.py and research_script.js.
    
## 2026-10-16 (Version 2.1.0)

### Research Engine & Offline Capabilities

  - **Enabled Conditional Offline Mode**
      - Removed the strict Tavily API Key requirement in `app.py` and `service.py`. The system now performs a smart pre-check: if no API Key is provided but a relevant local Vector Database is detected in `storage/` or `vector_stores/`, the research task is allowed to proceed using existing data.
  - **Implemented Fuzzy Directory Matching**
      - Upgraded `rag_pipeline.py` to support prefix-based directory discovery. The system can now locate existing vector stores (e.g., `Topic_HashID`) even if the current query string differs slightly from the original (e.g., extra spaces or punctuation causes MD5 mismatch). This significantly improves the reusability of previously generated knowledge bases.

### User Experience & API Interaction

  - **Soft Error Handling Strategy**
      - Modified the `/research/start` endpoint to return `HTTP 200 OK` with a `{ "success": False, "error": "..." }` payload for configuration errors (instead of raising `HTTP 400`).
  - **Enhanced UI Feedback**
      - Updated the frontend `ResearchManager` to gracefully handle "soft errors". Users now receive specific, actionable error messages (e.g., "Missing API Key and no local DB found") directly in the interface instead of generic network connection errors.

-----

## 2025-12-13 (Version 2.0.4)

### System Architecture & Performance

  - **Optimized RAG Content Generations**
      - Implemented **Parallel Content Generation** using `ThreadPoolExecutor` to execute multiple content generation tasks (RAG + LLM call) concurrently.
      - Significantly reduces report generation time (up to 1.5-2x acceleration on multi-core systems like M4).
      - Refactored `researcher/research.py` to decouple sequential loop from content generation logic via `generate_content_concurrently`.
      - Ensured **thread safety** by correctly passing and utilizing `threading.Lock` (`PROGRESS_LOCK`) to safely update progress indicators across parallel tasks.
  - **Improved LLM Output Stability (Pydantic)**
      - Started migrating critical LLM JSON parsing functions (`extract_facts_from_llm`, `evaluate_intimacy_from_llm`) to use LangChain's **PydanticOutputParser**.
      - Ensures LLM outputs conform to strict `FactExtraction` and `IntimacyEvaluation` schemas, drastically improving the stability of memory updates and intimacy calculations.

-----

## 2025-12-10 (Version 2.0.3)

### RAG & Search Enhancements

  - **Implemented Robust Hybrid Search Strategy**
      - Combined Chroma (Vector) and BM25 (Keyword) retrievers for superior search accuracy.
      - Added robust import handling to support multiple LangChain versions (v0.1/v0.2/v1.0/Classic).
      - Implemented automatic fallback to pure Vector Search if BM25 dependencies are missing.
      - Integrated `jieba` tokenization for optimized Chinese keyword retrieval.
  - **Improved Content Generation Logic**
      - Added `generate_content_with_fallback` strategy: Standard → Strict → Lenient → Force Generation.
      - Implemented context sufficiency checks (threshold: 500 chars) to prevent hallucinations.
      - Added intelligent deduplication to ensure unique content in generated reports.

### Voice & TTS Pipeline

  - **Full TTS Integration Workflow**
      - Completed end-to-end voice pipeline: Chinese Chat → LLM Translation (Kawaii Style) → VOICEVOX Synthesis.
      - Added `tts/voicevox.py` glue layer to manage translation, synthesis, and file cleanup.
      - Implemented `utils/file_cleanup.py` to automatically rotate audio files (keep latest 20) and prevent disk overflow.
      - Added `tts/translation.py` using `ChatOllama` to maintain character persona during language conversion.

### System Stability

  - **Dependency & Configuration Hardening**
      - Centralized global configuration in `config.py` (Models, DB paths, Intimacy params).
      - Fixed critical import errors for `langchain.retrievers` across different environments.
      - Verified and optimized Python virtual environment (`.venv`) compatibility.
      - Added `get_content_llm()` helper to ensure consistent model instantiation.

-----

## 2025-12-09 (Version 2.0.2)

### Settings & UI Integration

  - Added in-app settings page for theme, font size, Tavily API key, and LLM model configuration.
  - Implemented persistent settings storage using localStorage with backend `/api/settings` synchronization.
  - Added light/dark theme toggle and dynamic chat font-size adjustment.
  - Integrated settings view into main navigation with smooth view switching (Chat / Research / Settings).
  - Enhanced error handling and user feedback via toast notifications.

### Research Workflow & UX

  - Improved navigation guards to prevent switching views during ongoing chat or research tasks.
  - Refined loading, progress, and empty-state displays in the research view for better user guidance.
  - Improved research history loading, time formatting, and quick-research from history entries.
  - Enhanced export menu for research results (Markdown / JSON / TXT) with clearer UI and animations.

### Chat & Live2D Experience

  - Improved Live2D panel resizing and alignment behavior with the intimacy display.
  - Enhanced chat input behavior (autosizing textarea, better busy-state handling, clearer disable/enable states).
  - Refined chat history sidebar: pin/unpin, rename, delete, and first-room selection logic.
  - Stabilized TTS replay logic and user interaction handling for audio playback.

### Backend & Settings API

  - Added settings data model and REST endpoints for getting/saving user settings.
  - Ensured safe defaults for theme, font size, and LLM configuration when no settings are stored.
  - Improved logging around research job lifecycle, progress polling, and error conditions.

-----

## 2025-12-07 (Version 2.0.1)

### Performance Optimization

  - Improved researcher efficiency and reduced blocking operations in research workflow.
  - Corrected the progress bar display and status updates during long-running research tasks.

-----

## 2025-12-06 (Version 2.0)

### New Features

  - **Integrated RAG-based Research Report Generation System**
      - Implemented Retrieval-Augmented Generation (RAG) pipeline.
      - Integrated Tavily Search API for multi-region web content retrieval.
      - Added ChromaDB vector database for semantic search.
      - Implemented structured outline generation with LangChain.
      - Added content generation with duplicate detection and forced generation mode.

### Improvements

  - **Enhanced LLM Support**
      - Added support for multiple models: Gemma, Mistral via Ollama.
      - Added lenient prompt mode for better content generation success rate.
      - Improved query enhancement with domain-specific keywords.

### Documentation

  - **Updated Terms of Use**
      - Integrated terms for both Live2D chatbot and RAG research system.
      - Added content accuracy disclaimer and liability limitations.
      - Released bilingual versions (English & Traditional Chinese).

### Technical Details

  - Added `rag_pipeline.py` for RAG orchestration.
  - Added `research.py` for report generation workflow.
  - Added `get_prompts.py` for prompt template management.
  - Added `extract_outline.py` for outline parsing.
  - Implemented multi-stage content generation with quality checks.

-----

## 2025-10-19

### UI Improvements

  - Added top navigation bar.
  - Integrated Font Awesome icon library for enhanced visual design.

-----

## 2025-08-29

### Documentation

  - Updated README.md with comprehensive project information.

-----

## 2025-08-23

### Enhancements

  - Updated intimacy system with improved logic.
  - Refactored and optimized codebase for better maintainability.

-----

## 2025-08-21

### Backend Improvements

  - Migrated to SQLite database for persistent user information storage.
  - Improved data integrity and query performance.

-----

## 2025-08-20

### Audio Improvements

  - Enhanced TTS (Text-to-Speech) accuracy.
  - Fine-tuned voice synthesis parameters for more natural speech.

-----

## 2025-08-19

### Performance Optimization

  - Implemented asynchronous multitasking for improved responsiveness.
  - Enhanced concurrent request handling.

-----

## 2025-08-18

### Maintenance

  - Removed obsolete test files.
  - Cleaned up repository structure.

-----

## 2025-08-13

### New Features

  - Added intimacy system for dynamic user relationship tracking.
  - Implemented relationship progression mechanics.

-----

## 2025-08-03

### UI Improvements

  - Added collapsible UI elements for better space management.
  - Improved user interface organization.

-----

## 2025-08-02

### New Features

  - Integrated offline TTS engine: [VOICEVOX Nekotsuka Bi](https://voicevox.hiroshiba.jp/dormitory/nekotsuka_bi/)
  - Revised Terms of Use to include VOICEVOX licensing information.

-----

## 2025-08-01

### New Features

  - Added user terms agreement interface with acceptance tracking.
  - Enhanced overall UI/UX design.
  - Published comprehensive Terms of Use document.

-----

## 2025-07-31

### New Features

  - Integrated Text-to-Speech (TTS) module for voice output.
  - Implemented separate management for short-term and long-term memory.
  - Continued UI/UX improvements.

-----

## 2025-07-30

### Initial Release Features

  - Added dialog history filter and search functionality.
  - Improved LLM response quality with sentiment label hiding.
  - Enhanced user interface design and layout.

-----

## 2026-10-16

### New Features & Bug Fixes

- Added support for conditional offline mode and fuzzy directory matching

- Enables successful research even without a Tavily API Key but with a local vector cache

- Fixed some bugs.

-----

## Upcoming Features (Roadmap)

  - [ ] Voice input for chatbot interaction
  - [ ] Export research reports to PDF/DOCX formats
  - [ ] Advanced vector search tuning for multilingual content

-----

## Statistics Summary

  - **Total Updates:** 17 major releases
  - **Development Period:** July 2025 - November 2026
  - **Key Milestones:**
      - ✅ Live2D Chatbot System (July - August 2025)
      - ✅ RAG Research Report System Integration (December 2025)
      - ✅ Intimacy & Memory Systems (August 2025)
      - ✅ TTS Integration (July - August 2025)
      - ✅ Settings & Theme System (December 2025)

-----

## 🙏 Acknowledgments

  - **Live2D Co., Ltd.** - For the Hiyori Momose sample model
  - **VOICEVOX Team** - For the open-source TTS engine
  - **Tavily** - For the web search API
  - **Apple MLX & ChromaDB Communities** - For local inference and vector retrieval engines

*For detailed technical documentation, see [README.md](README.md)* *For usage terms, see [TERMS\_OF\_USE.md](https://www.google.com/search?q=TERMS_OF_USE.md)*