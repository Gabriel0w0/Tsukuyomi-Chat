// =================== 全域變數與常數 ===================
let lastAudioUrl = null;
let currentAudio = null;
let userHasInteracted = false;
let intimacyLevel = Number(localStorage.getItem("intimacyLevel")) || 50;

const DEFAULT_RESEARCH_MODEL = 'mlx-community/gemma-3-12b-it-4bit';
const DEFAULT_CHAT_MODEL = 'mlx-community/gemma-3-4b-it-8bit';
const DEFAULT_MEMORY_MODEL = 'mlx-community/gemma-3-1b-it-4bit';
const DEFAULT_MAX_TOKENS = 8192;
const DEFAULT_MAX_MEMORY = 10;
const DEFAULT_MAX_FACTS = 20;

// =================== 全域狀態管理 ===================
const AppState = {
  isResearching: false,
  isChatting: false,

  isBusy() {
    return this.isResearching || this.isChatting;
  },

  getBusyReason() {
    if (this.isResearching) return '研究進行中';
    if (this.isChatting) return '等待回覆中';
    return null;
  }
};
window.AppState = AppState;

// =================== 通用工具函數 ===================

// Toast 通知系統
function showToast(message, type = 'info', duration = 3000) {
  const oldToast = document.querySelector('.toast-notification');
  if (oldToast) oldToast.remove();

  const toast = document.createElement('div');
  toast.className = `toast-notification toast-${type}`;

  const icons = {
    success: 'fa-check-circle',
    warning: 'fa-exclamation-circle',
    error: 'fa-times-circle',
    info: 'fa-info-circle'
  };

  toast.innerHTML = `
    <i class="fas ${icons[type] || icons.info}"></i>
    <span>${message}</span>
  `;

  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}
window.showToast = showToast;

// 防 XSS 跳脫
function escapeHtml(text) {
  if (!text) return '';
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}

// 綁定支援 macOS / Windows 注音與中文輸入法防誤觸的 Enter 事件
function bindEnterWithIME(inputElement, onEnterCallback, allowShiftEnter = false) {
  if (!inputElement) return;

  let isComposing = false;

  inputElement.addEventListener('compositionstart', () => {
    isComposing = true;
  });

  inputElement.addEventListener('compositionend', () => {
    // 延遲 80ms 釋放，確保緊隨在 compositionend 後面的那次 Enter keydown 必定被攔截
    setTimeout(() => {
      isComposing = false;
    }, 80);
  });

  inputElement.addEventListener('keydown', (e) => {
    // 1. 若處於組字狀態，或觸發了輸入法專屬虛擬碼 (229 / Process)，一律略過
    if (isComposing || e.isComposing || e.keyCode === 229 || e.key === 'Process') {
      return;
    }

    // 2. 判斷是否為純 Enter（排除 Shift + Enter 換行）
    if (e.key === 'Enter' && (!allowShiftEnter || !e.shiftKey)) {
      e.preventDefault();
      onEnterCallback(e);
    }
  });
}

// 自訂毛玻璃輸入彈窗（取代瀏覽器原生 prompt，支援中文輸入法防誤觸與 Esc 取消）
const showCustomPrompt = (() => {
  let initialized = false;
  let activeResolver = null;
  let modal, titleEl, msgEl, inputEl, confirmBtn, cancelBtn, closeBtn;

  function finish(value) {
    if (!modal) return;
    modal.classList.add('hidden');
    if (activeResolver) {
      const resolve = activeResolver;
      activeResolver = null;
      resolve(value);
    }
  }

  function initOnce() {
    if (initialized) return true;
    modal = document.getElementById('customPromptModal');
    titleEl = document.getElementById('customPromptTitle');
    msgEl = document.getElementById('customPromptMessage');
    inputEl = document.getElementById('customPromptInput');
    confirmBtn = document.getElementById('customPromptConfirmBtn');
    cancelBtn = document.getElementById('customPromptCancelBtn');
    closeBtn = document.getElementById('customPromptCloseBtn');

    if (!modal || !inputEl) return false;

    confirmBtn?.addEventListener('click', () => finish(inputEl.value));
    cancelBtn?.addEventListener('click', () => finish(null));
    closeBtn?.addEventListener('click', () => finish(null));

    modal.addEventListener('click', (e) => {
      if (e.target === modal) finish(null);
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
        finish(null);
      }
    });

    bindEnterWithIME(inputEl, () => finish(inputEl.value), false);
    initialized = true;
    return true;
  }

  return function ({
    title = '輸入名稱',
    message = '',
    defaultValue = '',
    placeholder = '請輸入名稱...'
  } = {}) {
    if (!initOnce()) {
      return Promise.resolve(window.prompt(message || title, defaultValue));
    }

    if (activeResolver) {
      activeResolver(null);
      activeResolver = null;
    }

    if (titleEl) titleEl.textContent = title;
    if (msgEl) {
      msgEl.textContent = message;
      msgEl.style.display = message ? '' : 'none';
    }
    inputEl.value = defaultValue;
    inputEl.placeholder = placeholder;
    modal.classList.remove('hidden');

    requestAnimationFrame(() => {
      inputEl.focus();
      inputEl.select();
    });

    return new Promise((resolve) => {
      activeResolver = resolve;
    });
  };
})();

// 自訂毛玻璃確認彈窗（取代瀏覽器原生 confirm）
const showCustomConfirm = (() => {
  let initialized = false;
  let activeResolver = null;
  let modal, titleGroupEl, iconEl, titleEl, msgEl, okBtn, cancelBtn, closeBtn;

  function finish(result) {
    if (!modal) return;
    modal.classList.add('hidden');
    if (activeResolver) {
      const resolve = activeResolver;
      activeResolver = null;
      resolve(Boolean(result));
    }
  }

  function initOnce() {
    if (initialized) return true;
    modal = document.getElementById('customConfirmModal');
    titleGroupEl = document.getElementById('customConfirmTitleGroup');
    iconEl = document.getElementById('customConfirmIcon');
    titleEl = document.getElementById('customConfirmTitle');
    msgEl = document.getElementById('customConfirmMessage');
    okBtn = document.getElementById('customConfirmOkBtn');
    cancelBtn = document.getElementById('customConfirmCancelBtn');
    closeBtn = document.getElementById('customConfirmCloseBtn');

    if (!modal || !okBtn) return false;

    okBtn.addEventListener('click', () => finish(true));
    cancelBtn?.addEventListener('click', () => finish(false));
    closeBtn?.addEventListener('click', () => finish(false));

    modal.addEventListener('click', (e) => {
      if (e.target === modal) finish(false);
    });

    document.addEventListener('keydown', (e) => {
      if (modal.classList.contains('hidden')) return;
      if (e.key === 'Escape') {
        finish(false);
      }
    });

    initialized = true;
    return true;
  }

  return function ({
    title = '確認操作',
    message = '確定要執行此操作嗎？',
    confirmText = '確定',
    cancelText = '取消',
    danger = false
  } = {}) {
    if (!initOnce()) {
      return Promise.resolve(window.confirm(message));
    }

    if (activeResolver) {
      activeResolver(false);
      activeResolver = null;
    }

    if (titleEl) titleEl.textContent = title;
    if (msgEl) msgEl.textContent = message;
    if (okBtn) {
      okBtn.textContent = confirmText;
      okBtn.classList.toggle('danger', danger);
    }
    if (cancelBtn) cancelBtn.textContent = cancelText;
    if (titleGroupEl) titleGroupEl.classList.toggle('is-danger', danger);
    if (iconEl) {
      iconEl.className = danger
        ? 'fa-solid fa-triangle-exclamation'
        : 'fa-solid fa-circle-question';
    }

    modal.classList.remove('hidden');
    requestAnimationFrame(() => okBtn.focus());

    return new Promise((resolve) => {
      activeResolver = resolve;
    });
  };
})();

// 音訊許可（首次點擊或按鍵）
document.addEventListener("click", () => { userHasInteracted = true; }, { once: true });
document.addEventListener("keydown", () => { userHasInteracted = true; }, { once: true });

// TTS 播放與重播
function safePlayAudio(url) {
  if (!userHasInteracted) {
    console.warn("尚未有使用者互動，音訊播放被跳過");
    return;
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
  }
  currentAudio = new Audio(url);
  currentAudio.addEventListener("canplaythrough", () => {
    currentAudio.play().catch(err => console.warn("播放語音失敗：", err));
  });
}

function replayLastAudio() {
  if (!lastAudioUrl) {
    showToast('還沒有語音可以重播喔 (´;ω;`)', 'warning');
    return;
  }
  if (!userHasInteracted) {
    showToast('請先互動後才能播放語音 (´;ω;`)', 'warning');
    return;
  }
  safePlayAudio(lastAudioUrl);
}

// 親密度顯示更新
function updateIntimacyDisplay() {
  let levelName = "冷淡";
  if (intimacyLevel >= 90) levelName = "羈絆";
  else if (intimacyLevel >= 60) levelName = "親密";
  else if (intimacyLevel >= 30) levelName = "普通";

  const textElem = document.getElementById("intimacy-text");
  const barElem = document.getElementById("intimacy-bar-fill");
  if (textElem) textElem.textContent = `親密度：${intimacyLevel} (${levelName})`;
  if (barElem) barElem.style.width = `${Math.min(100, Math.max(0, intimacyLevel))}%`;
}

// 更新聊天框內的模型名稱徽章與選單勾選狀態
function updateChatModelBadge(modelName) {
  const modelNameEl = document.getElementById('current-model-name');
  if (modelNameEl && modelName) {
    const shortName = modelName.includes('/') ? modelName.split('/').pop() : modelName;
    modelNameEl.textContent = shortName;
    modelNameEl.title = modelName;
  }

  document.querySelectorAll('.model-option').forEach(opt => {
    opt.classList.toggle('selected', opt.dataset.model === modelName);
  });
}

// =================== 主應用 IIFE ===================
(function () {
  'use strict';

  // ========== 1. 權限與使用條款 Modal ==========
  function initAudioPermission() {
    const overlay = document.getElementById("audio-permission-overlay");
    if (!overlay) return;
    overlay.style.transition = "opacity 0.6s ease";
    overlay.style.opacity = "0";
    setTimeout(() => { overlay.style.display = "none"; }, 600);
    localStorage.setItem("hasVisited", "true");
  }

  function showTerms() {
    const modal = document.getElementById("termsModal");
    if (modal) modal.style.display = "block";
  }

  function closeTerms() {
    const modal = document.getElementById("termsModal");
    if (modal) modal.style.display = "none";
  }

  window.initAudioPermission = initAudioPermission;
  window.showTerms = showTerms;
  window.closeTerms = closeTerms;

  // ========== 2. 背景粒子 ==========
  if (window.tsParticles) {
    tsParticles.load("particles-js", {
      fpsLimit: 60,
      background: { color: "#111" },
      particles: {
        number: { value: 40, density: { enable: true, area: 800 } },
        color: { value: ["#a58aff", "#ff92c8", "#8844cc"] },
        shape: { type: "circle" },
        opacity: {
          value: 0.5,
          random: true,
          anim: { enable: true, speed: 0.5, opacity_min: 0.1, sync: false }
        },
        size: {
          value: 3,
          random: true,
          anim: { enable: true, speed: 3, size_min: 1, sync: false }
        },
        move: {
          enable: true,
          speed: 1,
          direction: "none",
          random: true,
          straight: false,
          outModes: "out"
        }
      },
      interactivity: {
        events: {
          onHover: { enable: true, mode: "grab" },
          onClick: { enable: true, mode: "push" }
        },
        modes: {
          grab: { distance: 140, links: { opacity: 0.5 } },
          push: { quantity: 3 }
        }
      },
      detectRetina: true
    });
  }

  // ========== 3. Live2D 控制器 ==========
  class Live2DController {
    constructor() {
      this.model = null;
      this.app = null;
    }

    async init() {
      const live2dPanel = document.querySelector('.live2d-panel');
      const live2dArea = document.getElementById('live2d-area');
      if (!live2dPanel || !live2dArea || !window.PIXI) return;

      this.app = new PIXI.Application({
        width: 400,
        height: live2dPanel.clientHeight,
        backgroundAlpha: 0,
        resizeTo: live2dPanel
      });
      live2dArea.appendChild(this.app.view);

      try {
        this.model = await PIXI.live2d.Live2DModel.from(
          "/static/model/hiyori/runtime/hiyori_pro_t11.model3.json"
        );
        window.live2dModel = this.model;
        this.model.scale.set(0.17);
        this.model.anchor.set(0.5, 0.5);
        this.model.x = this.app.renderer.width / 2;
        this.model.y = this.app.renderer.height / 2;
        this.app.stage.addChild(this.model);
        this.app.renderer.render(this.app.stage);
        window.addEventListener("resize", () => this.handleResize());
      } catch (err) {
        console.error("載入 Live2D 模型失敗:", err);
      }
    }

    handleResize() {
      if (!this.model || !this.app) return;
      const live2dPanel = document.querySelector('.live2d-panel');
      if (!live2dPanel) return;
      this.app.renderer.resize(400, live2dPanel.clientHeight);
      this.model.x = this.app.renderer.width / 2;
      this.model.y = this.app.renderer.height / 2;
      this.app.renderer.render(this.app.stage);
    }

    playMotion(motionName) {
      if (!this.model) return;
      try {
        this.model.motion(motionName);
      } catch (err) {
        console.error("播放動作失敗:", err);
      }
    }
  }

  // ========== 4. 聊天管理器與文字清洗工具 ==========

  // 匹配標準 [emotion:xxx] 以及模型偶爾自創的 [cute:blush]、[shy]、[joy:smile] 等情緒標籤
  const EMOTION_TAG_REGEX = /\[(?:emotion|mood|feeling|joy|sad|angry|neutral|cute|shy|happy|blush|smile|love|excited|surprised|nervous|embarrassed)(?:\s*[:：]\s*[a-zA-Z0-9_\s-]+)?\]/gi;
  // 匹配句首任何 [xxx:yyy] 形式的角色狀態標籤（如 [action:wave]）
  const LEADING_COLON_TAG_REGEX = /^\s*\[[a-zA-Z_-]+\s*[:：]\s*[a-zA-Z0-9_\s-]+\]\s*/;

  function removeEmotionTag(text) {
    if (!text) return "";
    let cleaned = text;

    // 1. 若模型把 "User: ... Assistant: ..." 連在一起輸出，只截取 Assistant: 之後的回覆
    const assistantSplit = cleaned.split(/(?:Assistant|月讀醬|Bot)\s*[:：]/i);
    if (assistantSplit.length > 1 && /^\s*(?:User|使用者|主人)\s*[:：]/i.test(cleaned)) {
      cleaned = assistantSplit.slice(1).join("").trim();
    } else {
      cleaned = cleaned.replace(/^\s*(?:Assistant|User|月讀醬|月讀|Bot)\s*[:：]\s*/i, "");
    }

    // 2. 截斷模型在結尾自己腦補的下一輪 User: 對話
    cleaned = cleaned.split(/\n\s*(?:User|使用者|主人)\s*[:：]/i)[0];

    // 3. 移除標準與變體情緒標籤（含 [emotion:xxx]、[cute:blush] 等）
    cleaned = cleaned.replace(EMOTION_TAG_REGEX, "");
    cleaned = cleaned.replace(LEADING_COLON_TAG_REGEX, "");

    return cleaned.trim();
  }

  // 通用思考標籤匹配規則：通吃 <|channel>thought, <channel|>, </|channel>, <|/channel|>, <think>, </think>, <|thought|> 等所有變體
  const THINK_OPEN_REGEX = /(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)/i;
  const THINK_CLOSE_REGEX = /(?:<[\|\/]*channel[\|\/]*>|<[\|\/]*(?:think|thought)[\|\/]*>)/i;
  const STRAY_TAGS_GLOBAL_REGEX = /(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)/gi;
  // 攔截串流過程中停在字尾、尚未打完的半截標籤（例如 "<|chan"、"[emotion:jo"、"[cute:bl"）
  const TRAILING_PARTIAL_TAG_REGEX = /(?:<[\|\/]*[a-z]*[\|\/]*|\[(?:emotion|mood|joy|sad|angry|neutral|cute|shy|happy|blush)[a-z0-9_\s:-]*)$/i;

  // 1. 靜態與歷史訊息雙重保險解析
  function extractClientThinking(rawText) {
    if (!rawText) return { thinking: "", cleanText: "" };
    const blockRegex = /(?:<[\|\/]*channel[\|\/]*>\s*(?:thought\b)?|<[\|\/]*(?:think|thought)[\|\/]*>)([\s\S]*?)(?:<[\|\/]*channel[\|\/]*>|<[\|\/]*(?:think|thought)[\|\/]*>)/gi;
    const thoughts = [];
    let withoutThink = rawText.replace(blockRegex, (_, content) => {
      if (content && content.trim()) thoughts.push(content.trim());
      return "";
    });
    // 清除任何殘留的孤立標籤（含 </|channel>）
    withoutThink = withoutThink.replace(STRAY_TAGS_GLOBAL_REGEX, "");

    return {
      thinking: thoughts.join("\n\n").replace(STRAY_TAGS_GLOBAL_REGEX, "").trim(),
      cleanText: removeEmotionTag(withoutThink)
    };
  }

  // 2. 即時串流解析器：支援思考標籤尚未閉合時的即時分離，並隱藏尾端半截標籤
  function parseStreamingContent(rawText) {
    if (!rawText) return { thinking: "", answer: "", isThinking: false };

    const openMatch = rawText.match(THINK_OPEN_REGEX);
    if (!openMatch) {
      // 若開頭正在輸出半截標籤（如 `<|chan` 或 `</|`），暫時不顯示文字以免閃爍
      if (/^\s*<[\|\/]*[a-z]*[\|\/]*$/i.test(rawText)) {
        return { thinking: "", answer: "", isThinking: true };
      }
      const cleanAns = removeEmotionTag(rawText.replace(STRAY_TAGS_GLOBAL_REGEX, ""))
        .replace(TRAILING_PARTIAL_TAG_REGEX, "");
      return { thinking: "", answer: cleanAns, isThinking: false };
    }

    const afterOpen = rawText.slice(openMatch.index + openMatch[0].length);
    const closeMatch = afterOpen.match(THINK_CLOSE_REGEX);

    if (!closeMatch) {
      // 思考進行中（尚未收到閉合標籤），抹除尾端可能正在生成的半截 `</|chan`
      const cleanThinking = afterOpen
        .replace(/^\s+/, "")
        .replace(TRAILING_PARTIAL_TAG_REGEX, "");
      return {
        thinking: cleanThinking,
        answer: "",
        isThinking: true
      };
    }

    // 思考已閉合，開始串流輸出正式回答
    const thinkingPart = afterOpen
      .slice(0, closeMatch.index)
      .replace(STRAY_TAGS_GLOBAL_REGEX, "")
      .trim();

    const answerRaw = afterOpen
      .slice(closeMatch.index + closeMatch[0].length)
      .replace(STRAY_TAGS_GLOBAL_REGEX, "");

    const cleanAnswer = removeEmotionTag(answerRaw).replace(TRAILING_PARTIAL_TAG_REGEX, "");

    return {
      thinking: thinkingPart,
      answer: cleanAnswer,
      isThinking: false
    };
  }

  // 建立參考來源 DOM 區塊（共用輔助函數）
  function createSourcesElement(sources) {
    if (!Array.isArray(sources) || sources.length === 0) return null;
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'chat-search-sources';
    sourcesDiv.innerHTML = '<span><i class="fa-solid fa-bookmark"></i> 參考來源：</span>';

    sources.forEach(src => {
      if (src.url) {
        const a = document.createElement('a');
        a.href = src.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.className = 'chat-source-link';
        a.textContent = src.title || src.url;
        a.title = src.title || src.url;
        sourcesDiv.appendChild(a);
      } else if (src.title) {
        const span = document.createElement('span');
        span.className = 'chat-source-link';
        span.textContent = src.title;
        span.title = src.title;
        sourcesDiv.appendChild(span);
      }
    });
    return sourcesDiv;
  }

  class ChatManager {
    constructor() {
      this.storageKey = 'tsukuyomi_chat_data';
      this.data = this.loadData();
      this.currentRoom = null;
      this.useWebSearch = false;
      this.activeTypeWriterTimer = null;
      this.abortController = null;

      this.chatHistory = document.querySelector('.chat-history');
      this.chatMessages = document.getElementById('chat-messages');
      this.chatInput = document.getElementById('chatInput');
      this.sendBtn = document.getElementById('sendBtn');
      this.webSearchBtn = document.getElementById('webSearchToggleBtn');
      this.newChatBtn = document.querySelector('.new-chat-btn');
      this.typingIndicator = document.getElementById('typing-indicator');

      this.initEventListeners();
      this.renderSidebar();
      this.selectFirstRoom();
    }

    loadData() {
      try {
        const data = localStorage.getItem(this.storageKey);
        return data ? JSON.parse(data) : {};
      } catch (e) {
        console.error("載入聊天數據失敗:", e);
        return {};
      }
    }

    saveData() {
      try {
        localStorage.setItem(this.storageKey, JSON.stringify(this.data));
      } catch (e) {
        console.error("保存聊天數據失敗:", e);
      }
    }

    stopActiveTypeWriter() {
      if (this.activeTypeWriterTimer) {
        clearTimeout(this.activeTypeWriterTimer);
        this.activeTypeWriterTimer = null;
      }
    }

    stopGeneration() {
      if (this.abortController) {
        this.abortController.abort();
        this.abortController = null;
      }
    }

    initEventListeners() {
      this.sendBtn?.addEventListener('click', () => {
        if (AppState.isChatting) {
          this.stopGeneration();
        } else {
          this.sendMessage();
        }
      });
      this.newChatBtn?.addEventListener('click', () => this.createNewRoom());

      if (this.webSearchBtn) {
        this.webSearchBtn.addEventListener('click', () => {
          this.useWebSearch = !this.useWebSearch;
          this.webSearchBtn.classList.toggle('active-search', this.useWebSearch);
          this.webSearchBtn.title = this.useWebSearch ? '已開啟聯網搜尋（點擊關閉）' : '開啟聯網搜尋';
          showToast(
            this.useWebSearch ? '已開啟聯網搜尋模式' : '已切換回一般對話模式',
            'info',
            2000
          );
        });
      }

      bindEnterWithIME(this.chatInput, () => this.sendMessage(), true);

      this.chatInput?.addEventListener('input', (e) => {
        e.target.style.height = 'auto';
        e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
      });
    }

    renderSidebar() {
      if (!this.chatHistory) return;
      this.chatHistory.innerHTML = '';

      const items = Object.keys(this.data).map(title => ({
        title,
        pinned: this.data[title].pinned || false
      }));
      const pinned = items.filter(i => i.pinned);
      const unpinned = items.filter(i => !i.pinned);

      [...pinned, ...unpinned].forEach(item => {
        const li = document.createElement('li');
        li.className = 'chat-item'
          + (item.pinned ? ' pinned' : '')
          + (item.title === this.currentRoom ? ' active' : '');

        const icon = document.createElement('i');
        icon.className = 'fa fa-message';

        const title = document.createElement('span');
        title.className = 'chat-title';
        title.textContent = item.title;

        const pinBtn = document.createElement('button');
        pinBtn.type = 'button';
        pinBtn.className = 'pin-btn';
        pinBtn.title = '釘選/取消';
        pinBtn.innerHTML = '<i class="fa fa-thumbtack"></i>';

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'delete-btn';
        deleteBtn.title = '刪除';
        deleteBtn.innerHTML = '<i class="fa fa-trash"></i>';

        li.append(icon, title, pinBtn, deleteBtn);

        li.addEventListener('click', (e) => {
          if (e.target.closest('.pin-btn, .delete-btn')) return;
          this.selectRoom(item.title);
        });

        pinBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          this.togglePin(item.title);
        });

        deleteBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          this.deleteRoom(item.title);
        });

        let longPressTimer;
        title.addEventListener('mousedown', () => {
          longPressTimer = setTimeout(() => this.renameRoom(item.title), 600);
        });
        title.addEventListener('mouseup', () => clearTimeout(longPressTimer));
        title.addEventListener('mouseleave', () => clearTimeout(longPressTimer));

        this.chatHistory.appendChild(li);
      });
    }

    selectRoom(roomTitle) {
      this.stopActiveTypeWriter();
      this.currentRoom = roomTitle;
      document.querySelectorAll('.chat-item').forEach(el => {
        const titleEl = el.querySelector('.chat-title');
        el.classList.toggle('active', titleEl?.textContent === roomTitle);
      });
      this.renderMessages();
    }

    renderMessages() {
      if (!this.currentRoom || !this.chatMessages) return;
      this.stopActiveTypeWriter();

      const typingIndicator = document.getElementById('typing-indicator');
      const isTyping = typingIndicator && !typingIndicator.classList.contains('hidden');

      Array.from(this.chatMessages.children).forEach(child => {
        if (child.id !== 'typing-indicator') child.remove();
      });

      const messages = this.data[this.currentRoom]?.messages || [];
      messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `chat-message ${msg.role}`;

        const parsed = extractClientThinking(msg.text);
        const finalThinking = msg.thinking || parsed.thinking;
        const finalAnswer = parsed.cleanText || msg.text;

        if (finalThinking) {
          const detailsEl = document.createElement('details');
          detailsEl.className = 'chat-thinking-box';

          const summaryEl = document.createElement('summary');
          summaryEl.className = 'chat-thinking-summary';
          summaryEl.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> <span>月讀醬的深度思考過程</span>';

          const contentEl = document.createElement('div');
          contentEl.className = 'chat-thinking-content';
          contentEl.textContent = finalThinking;

          detailsEl.append(summaryEl, contentEl);
          div.appendChild(detailsEl);
        }

        const textNode = document.createElement('div');
        textNode.className = 'chat-answer-text';
        textNode.textContent = finalAnswer;
        div.appendChild(textNode);

        const sourcesDiv = createSourcesElement(msg.sources);
        if (sourcesDiv) div.appendChild(sourcesDiv);

        this.chatMessages.appendChild(div);
      });

      if (!document.getElementById('typing-indicator')) {
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'chat-message bot typing-indicator hidden';
        indicator.innerHTML = '<div class="wave-dots"><span></span><span></span><span></span></div>';
        this.chatMessages.appendChild(indicator);
        this.typingIndicator = indicator;
      } else {
        this.chatMessages.appendChild(typingIndicator);
      }

      if (isTyping) this.showTyping();
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    sendMessage() {
      if (AppState.isBusy()) {
        showToast(`${AppState.getBusyReason()}，請稍候喔～`, 'warning');
        return;
      }

      if (!this.currentRoom || !this.chatInput.value.trim()) return;
      const userMessage = this.chatInput.value.trim();

      if (!this.data[this.currentRoom]) {
        this.data[this.currentRoom] = { messages: [], pinned: false };
      }
      this.data[this.currentRoom].messages.push({
        role: 'user',
        text: userMessage
      });
      this.saveData();
      this.renderMessages();

      this.chatInput.value = '';
      this.chatInput.style.height = 'auto';

      AppState.isChatting = true;
      this.disableInput();
      this.fetchBotResponse(userMessage, this.currentRoom);
    }

    showTyping() {
      if (this.typingIndicator) {
        this.typingIndicator.classList.remove('hidden');
        this.chatMessages.appendChild(this.typingIndicator);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        this.typingIndicator.querySelectorAll('.wave-dots span').forEach(dot => {
          dot.style.animation = 'none';
          dot.offsetHeight;
          dot.style.animation = null;
        });
      }
    }

    hideTyping() {
      this.typingIndicator?.classList.add('hidden');
    }

    disableInput() {
      const inputArea = document.getElementById('chat-input-area');
      inputArea?.classList.add('is-loading');

      if (this.chatInput) {
        this.chatInput.disabled = true;
        this.chatInput.placeholder = this.useWebSearch
          ? '月讀醬正在翻閱網路資料中...'
          : '月讀醬正在思考中...';
      }
      if (this.sendBtn) {
        this.sendBtn.disabled = false;
        this.sendBtn.classList.remove('send-btn-loading');
        this.sendBtn.classList.add('send-btn-stop');
        this.sendBtn.title = '停止輸出';
        this.sendBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
      }
      document.querySelectorAll('.nav-center a').forEach(link => link.classList.add('nav-busy'));
    }

    enableInput() {
      const inputArea = document.getElementById('chat-input-area');
      inputArea?.classList.remove('is-loading');

      if (this.chatInput) {
        this.chatInput.disabled = false;
        this.chatInput.placeholder = '詢問初音未來及V家的問題或單純的聊聊天吧(•ω•)/';
      }
      if (this.sendBtn) {
        this.sendBtn.disabled = false;
        this.sendBtn.classList.remove('send-btn-loading', 'send-btn-stop');
        this.sendBtn.title = '送出';
        this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
      }
      document.querySelectorAll('.nav-center a').forEach(link => link.classList.remove('nav-busy'));
    }

    async fetchBotResponse(message, targetRoom) {
      this.showTyping();

      const bubble = document.createElement('div');
      bubble.className = 'chat-message bot';
      bubble.style.display = 'none';

      const detailsEl = document.createElement('details');
      detailsEl.className = 'chat-thinking-box';
      detailsEl.style.display = 'none';
      detailsEl.open = true;

      const summaryEl = document.createElement('summary');
      summaryEl.className = 'chat-thinking-summary';
      summaryEl.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> <span>月讀醬正在深度思考中...</span>';

      const thinkingContentEl = document.createElement('div');
      thinkingContentEl.className = 'chat-thinking-content';
      detailsEl.append(summaryEl, thinkingContentEl);

      const answerNode = document.createElement('div');
      answerNode.className = 'chat-answer-text';

      bubble.append(detailsEl, answerNode);
      this.chatMessages.insertBefore(bubble, this.typingIndicator);

      let rawAccumulated = "";
      let finalSources = [];

      const handleStreamLine = (line) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);

        if (event.type === 'meta') {
          if (event.search_warning) {
            showToast(event.search_warning, 'warning', 4000);
          }
          finalSources = event.sources || [];
          const sourcesDiv = createSourcesElement(finalSources);
          if (sourcesDiv) bubble.appendChild(sourcesDiv);
        } else if (event.type === 'token') {
          this.hideTyping();
          if (this.currentRoom === targetRoom) {
            bubble.style.display = '';
          }

          rawAccumulated += event.text;
          const state = parseStreamingContent(rawAccumulated);

          if (state.thinking) {
            detailsEl.style.display = '';
            thinkingContentEl.textContent = state.thinking;
            thinkingContentEl.scrollTop = thinkingContentEl.scrollHeight;
          }

          if (!state.isThinking && state.thinking && detailsEl.open) {
            detailsEl.open = false;
            summaryEl.innerHTML = '<i class="fa-solid fa-circle-nodes"></i> <span>月讀醬的深度思考過程</span>';
          }

          if (state.answer) {
            answerNode.textContent = state.answer;
          }

          if (this.currentRoom === targetRoom) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
          }
        } else if (event.type === 'done') {
          const cleanReply = removeEmotionTag(event.reply || '');
          const cleanThinking = event.thinking || '';

          if (!this.data[targetRoom]) {
            this.data[targetRoom] = { messages: [], pinned: false };
          }
          this.data[targetRoom].messages.push({
            role: 'bot',
            text: cleanReply,
            thinking: cleanThinking,
            sources: event.sources || finalSources
          });
          this.saveData();

          if (this.currentRoom === targetRoom) {
            this.renderMessages();
          } else {
            bubble.remove();
          }

          if (event.audio_url) {
            lastAudioUrl = event.audio_url;
            safePlayAudio(lastAudioUrl);
          }
          if (typeof event.intimacy === "number") {
            intimacyLevel = event.intimacy;
            updateIntimacyDisplay();
            localStorage.setItem("intimacyLevel", intimacyLevel);
          }
        } else if (event.type === 'error') {
          throw new Error(event.error);
        }
      };

      try {
        this.abortController = new AbortController();

        const fileMgr = window.app?.file;
        const activeFiles = fileMgr?.getActiveFiles
          ? fileMgr.getActiveFiles()
          : (fileMgr?.files?.map(f => f.name) || []);

        const res = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: this.abortController.signal,
          body: JSON.stringify({
            message: message,
            use_search: this.useWebSearch,
            use_local_rag: activeFiles.length > 0,
            active_files: activeFiles
          })
        });

        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            buffer += decoder.decode();
            if (buffer.trim()) handleStreamLine(buffer);
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            handleStreamLine(line);
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          const state = parseStreamingContent(rawAccumulated);
          const partialAnswer = state.answer
            ? `${state.answer}\n\n(You have stopped the output.)`
            : '(You have stopped the output.)';

          if (!this.data[targetRoom]) {
            this.data[targetRoom] = { messages: [], pinned: false };
          }
          this.data[targetRoom].messages.push({
            role: 'bot',
            text: partialAnswer,
            thinking: state.thinking || '',
            sources: finalSources
          });
          this.saveData();

          if (this.currentRoom === targetRoom) {
            this.renderMessages();
          } else {
            bubble.remove();
          }
          showToast('已停止月讀醬的輸出', 'info', 1800);
        } else {
          console.error("串流 API 發生錯誤：", err);
          bubble.remove();
          this.data[targetRoom]?.messages.push({
            role: 'bot',
            text: '出錯了喔！'
          });
          this.saveData();
          if (this.currentRoom === targetRoom) {
            this.renderMessages();
          }
        }
      } finally {
        this.abortController = null;
        AppState.isChatting = false;
        this.enableInput();
        this.hideTyping();
      }
    }

    async createNewRoom() {
      const title = await showCustomPrompt({
        title: '新增聊天室',
        message: '請幫聊天室取個名字',
        defaultValue: '新聊天室',
        placeholder: '例如：上課筆記、今晚讀書計畫...'
      });
      if (!title || !title.trim()) return;

      const cleanTitle = title.trim();
      if (this.data[cleanTitle]) {
        showToast('聊天室名稱已存在！', 'warning');
        return;
      }
      this.data[cleanTitle] = { messages: [], pinned: false };
      this.saveData();
      this.renderSidebar();
      this.selectRoom(cleanTitle);
    }

    async renameRoom(oldTitle) {
      const newTitle = await showCustomPrompt({
        title: '重新命名聊天室',
        message: `將「${oldTitle}」修改為：`,
        defaultValue: oldTitle,
        placeholder: '請輸入新的聊天室名稱...'
      });
      if (!newTitle || !newTitle.trim() || newTitle.trim() === oldTitle) return;

      const cleanTitle = newTitle.trim();
      if (this.data[cleanTitle]) {
        showToast('聊天室名稱已存在！', 'warning');
        return;
      }
      this.data[cleanTitle] = this.data[oldTitle];
      delete this.data[oldTitle];
      if (this.currentRoom === oldTitle) {
        this.currentRoom = cleanTitle;
      }
      this.saveData();
      this.renderSidebar();
      this.selectRoom(cleanTitle);
    }

    async deleteRoom(title) {
      const ok = await showCustomConfirm({
        title: '刪除聊天室',
        message: `確定要刪除聊天室「${title}」嗎？此操作無法復原。`,
        confirmText: '確定刪除',
        danger: true
      });
      if (!ok) return;

      delete this.data[title];
      this.saveData();
      this.renderSidebar();
      if (this.currentRoom === title) {
        this.selectFirstRoom();
      }
    }

    togglePin(title) {
      if (!this.data[title]) return;
      this.data[title].pinned = !this.data[title].pinned;
      this.saveData();
      this.renderSidebar();
      this.selectRoom(title);
    }

    selectFirstRoom() {
      const rooms = Object.keys(this.data);
      if (rooms.length > 0) {
        this.selectRoom(rooms[0]);
      } else {
        const defaultRoom = '日常閒聊';
        this.data[defaultRoom] = {
          pinned: false,
          messages: [
            {
              role: 'bot',
              text: '主人歡迎回來～今天想跟月讀聊聊天、查最新資料，還是要丟資料給我讓我幫你看看呢？(•ω•)/'
            }
          ]
        };
        this.saveData();
        this.renderSidebar();
        this.selectRoom(defaultRoom);
      }
    }

    async clearSession() {
      if (AppState.isBusy()) {
        showToast(`${AppState.getBusyReason()}，請稍候再清除記憶喔～`, 'warning');
        return;
      }
      if (!this.currentRoom) return;

      const ok = await showCustomConfirm({
        title: '清除對話記憶',
        message: '確定要清除短期記憶嗎？這會讓月讀醬忘記剛剛聊的內容喔！',
        confirmText: '清除短期記憶',
        danger: true
      });
      if (!ok) return;

      this.stopActiveTypeWriter();
      this.data[this.currentRoom].messages = [
        {
          role: 'bot',
          text: '嗚...剛剛聊了什麼我好像有點忘記了呢，主人我們重新開始聊吧～'
        }
      ];
      this.saveData();
      this.renderMessages();
      fetch('/clear_session', { method: 'POST' }).catch(() => {});
    }

    async clearAllMemory() {
      if (AppState.isBusy()) {
        showToast(`${AppState.getBusyReason()}，請稍候再清除記憶喔～`, 'warning');
        return;
      }

      const ok = await showCustomConfirm({
        title: '清除所有記憶',
        message: '確定要清除所有長期與短期記憶嗎？月讀醬會忘記所有過去的事情喔！',
        confirmText: '全部清除',
        danger: true
      });
      if (!ok) return;

      this.stopActiveTypeWriter();
      this.data = {};
      this.currentRoom = null;
      this.saveData();
      fetch('/clear_memory', { method: 'POST' }).catch(() => {});
      this.selectFirstRoom();
    }
  }

  // =================== Researcher 介面 ===================

  class SearchSuggestionManager {
    constructor() {
      this.input = document.getElementById('researchInput');
      this.suggestionsContainer = document.getElementById('searchSuggestions');
      this.suggestionHistory = this.loadHistory();
      this.popularTopics = [
        { text: '大型語言模型最新發展', icon: 'fa-brain' },
        { text: '台灣半導體產業趨勢', icon: 'fa-microchip' },
        { text: '永續能源技術創新', icon: 'fa-leaf' },
      ];
      this.init();
    }

    init() {
      if (!this.input || !this.suggestionsContainer) return;

      this.input.addEventListener('input', (e) => this.handleInput(e));
      this.input.addEventListener('focus', () => {
        const currentValue = this.input.value.trim();
        if (currentValue.length === 0) {
          this.showDefaultSuggestions();
        } else {
          this.updateSuggestions(currentValue);
        }
      });

      document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box-wrapper')) {
          this.hideSuggestions();
        }
      });
    }

    handleInput(e) {
      const query = e.target.value.trim();
      if (query.length > 0) {
        this.updateSuggestions(query);
      } else {
        this.showDefaultSuggestions();
      }
    }

    updateSuggestions(query) {
      const lowerQuery = query.toLowerCase();
      const filtered = this.popularTopics.filter(topic =>
        topic.text.toLowerCase().includes(lowerQuery)
      );
      const historyMatches = this.suggestionHistory
        .filter(item => item.toLowerCase().includes(lowerQuery))
        .slice(0, 3);

      this.renderSuggestions([...historyMatches, ...filtered]);
    }

    showDefaultSuggestions() {
      this.renderSuggestions(this.popularTopics);
    }

    renderSuggestions(items) {
      if (!this.suggestionsContainer) return;
      this.suggestionsContainer.innerHTML = '';

      items.forEach(item => {
        const li = document.createElement('li');
        li.className = 'search-suggestion-item';

        const text = typeof item === 'string' ? item : item.text;
        const icon = typeof item === 'string' ? 'fa-history' : item.icon;

        li.innerHTML = `
          <i class="fas ${icon}"></i>
          <span class="suggestion-text">${escapeHtml(text)}</span>
        `;

        li.addEventListener('click', () => {
          this.input.value = text;
          this.hideSuggestions();
          this.saveToHistory(text);
          document.getElementById('startResearchBtn')?.click();
        });

        this.suggestionsContainer.appendChild(li);
      });

      this.showSuggestions();
    }

    showSuggestions() {
      this.suggestionsContainer?.classList.remove('hidden');
    }

    hideSuggestions() {
      this.suggestionsContainer?.classList.add('hidden');
    }

    saveToHistory(query) {
      if (!this.suggestionHistory.includes(query)) {
        this.suggestionHistory.unshift(query);
        this.suggestionHistory = this.suggestionHistory.slice(0, 10);
        localStorage.setItem('search_history', JSON.stringify(this.suggestionHistory));
      }
    }

    loadHistory() {
      try {
        return JSON.parse(localStorage.getItem('search_history') || '[]');
      } catch {
        return [];
      }
    }
  }

  class AccordionManager {
    constructor() {
      this.toggle = document.getElementById('advancedOptionsToggle');
      this.content = document.getElementById('advancedOptionsContent');
      this.init();
    }

    init() {
      if (!this.toggle || !this.content) return;
      this.toggle.addEventListener('click', () => {
        const isExpanded = this.content.classList.contains('expanded');
        this.content.classList.toggle('expanded', !isExpanded);
        this.content.classList.toggle('collapsed', isExpanded);
        this.toggle.classList.toggle('active', !isExpanded);
      });
    }
  }

  class ProgressManager {
    constructor() {
      this.indicator = document.getElementById('progressIndicator');
      this.bar = document.getElementById('progressBar');
      this.text = document.getElementById('progressText');
      this.details = document.getElementById('progressDetails');
    }

    start() {
      if (!this.indicator) return;
      this.indicator.classList.remove('hidden');
      this.updateProgress(0, '準備中...', '正在初始化研究流程...');
    }

    updateProgress(percent, phase, message) {
      if (this.bar) {
        this.bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
      }
      if (this.text) {
        this.text.textContent = phase || '研究進行中...';
      }
      if (this.details) {
        this.details.textContent = message || '';
      }
    }

    complete() {
      this.updateProgress(100, '完成', '研究報告已生成');
      setTimeout(() => {
        this.indicator?.classList.add('hidden');
        this.reset();
      }, 1500);
    }

    stop() {
      this.indicator?.classList.add('hidden');
      this.reset();
    }

    reset() {
      if (this.bar) this.bar.style.width = '0%';
    }
  }

  class ViewManager {
    constructor() {
      this.viewButtons = document.querySelectorAll('.view-btn');
      this.resultsContainer = document.getElementById('resultsContainer');
      this.currentView = 'detailed';
      this.init();
    }

    init() {
      this.viewButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
          this.switchView(e.currentTarget.dataset.view);
        });
      });
    }

    switchView(view) {
      this.viewButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
      });
      if (this.resultsContainer) {
        this.resultsContainer.className = `results-container view-${view}`;
      }
      this.currentView = view;
    }
  }

  class ResearchHistoryManager {
    constructor() {
      this.storageKey = 'research_history';
      this.historyList = document.getElementById('researchHistoryList');
      this.clearBtn = document.getElementById('clearHistoryBtn');
      this.history = this.loadHistory();
      this.init();
    }

    init() {
      this.renderHistory();
      this.clearBtn?.addEventListener('click', () => this.clearAll());
    }

    loadHistory() {
      try {
        return JSON.parse(localStorage.getItem(this.storageKey) || '[]');
      } catch {
        return [];
      }
    }

    saveHistory() {
      localStorage.setItem(this.storageKey, JSON.stringify(this.history));
    }

    addItem(query, result) {
      const item = {
        id: Date.now(),
        query: query,
        timestamp: new Date().toISOString(),
        summary: result.substring(0, 100) + '...',
        fullResult: result
      };

      this.history.unshift(item);
      this.history = this.history.slice(0, 50);
      this.saveHistory();
      this.renderHistory();
    }

    renderHistory() {
      if (!this.historyList) return;
      this.historyList.innerHTML = '';

      if (this.history.length === 0) {
        this.historyList.innerHTML = `
          <div class="empty-state" style="padding: 20px;">
            <p style="font-size: 0.9rem; color: var(--text-secondary);">尚無研究紀錄</p>
          </div>
        `;
        return;
      }

      this.history.forEach(item => {
        const li = document.createElement('li');
        li.className = 'history-item';

        const timeStr = this.formatTimeAgo(new Date(item.timestamp));

        li.innerHTML = `
          <div class="history-title">
            <i class="fas fa-file-alt"></i>
            <span>${escapeHtml(item.query)}</span>
          </div>
          <div class="history-meta">
            <span>${timeStr}</span>
            <div class="history-actions">
              <button class="history-action-btn" data-action="reload" title="重新執行">
                <i class="fas fa-redo"></i>
              </button>
              <button class="history-action-btn" data-action="delete" title="刪除">
                <i class="fas fa-trash"></i>
              </button>
            </div>
          </div>
        `;

        li.querySelector('[data-action="reload"]').addEventListener('click', (e) => {
          e.stopPropagation();
          const input = document.getElementById('researchInput');
          if (input) {
            input.value = item.query;
            window.switchMainView?.('research');
            document.getElementById('startResearchBtn')?.click();
          }
        });

        li.querySelector('[data-action="delete"]').addEventListener('click', (e) => {
          e.stopPropagation();
          this.deleteItem(item.id);
        });

        li.addEventListener('click', () => {
          window.switchMainView?.('research');
          this.loadResult(item);
        });

        this.historyList.appendChild(li);
      });
    }

    formatTimeAgo(date) {
      const diffMs = Date.now() - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return '剛才';
      if (diffMins < 60) return `${diffMins} 分鐘前`;
      if (diffHours < 24) return `${diffHours} 小時前`;
      if (diffDays < 7) return `${diffDays} 天前`;
      return date.toLocaleDateString('zh-TW');
    }

    deleteItem(id) {
      this.history = this.history.filter(item => item.id !== id);
      this.saveHistory();
      this.renderHistory();
    }

    clearAll() {
      if (confirm('確定要清除所有研究紀錄嗎？')) {
        this.history = [];
        this.saveHistory();
        this.renderHistory();
      }
    }

    loadResult(item) {
      const resultsContainer = document.getElementById('resultsContainer');
      const emptyState = document.getElementById('emptyState');

      if (resultsContainer) {
        resultsContainer.innerHTML = item.fullResult;
        resultsContainer.scrollTop = 0;
      }
      emptyState?.classList.add('hidden');
    }
  }

  class ExportManager {
    constructor() {
      this.exportBtn = document.getElementById('exportResultsBtn');
      this.init();
    }

    init() {
      this.exportBtn?.addEventListener('click', () => this.showExportMenu());
    }

    showExportMenu() {
      if (!document.getElementById('export-menu-styles')) {
        const style = document.createElement('style');
        style.id = 'export-menu-styles';
        style.textContent = `
          .export-menu {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            animation: fadeIn 0.2s ease;
          }
          .export-menu-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            min-width: 300px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
          }
          .export-menu-content h4 {
            color: var(--accent-purple);
            margin: 0 0 20px 0;
            font-size: 1.2rem;
          }
          .export-option {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 14px 20px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1rem;
            transition: all 0.2s;
          }
          .export-option:hover {
            background: var(--bg-primary);
            border-color: var(--accent-purple);
            transform: translateX(4px);
          }
          .export-option i {
            color: var(--accent-pink);
            font-size: 1.2rem;
          }
        `;
        document.head.appendChild(style);
      }

      const menu = document.createElement('div');
      menu.className = 'export-menu';
      menu.innerHTML = `
        <div class="export-menu-content">
          <h4>選擇匯出格式</h4>
          <button class="export-option" data-format="pdf">
            <i class="fas fa-file-pdf"></i> PDF 格式
          </button>
          <button class="export-option" data-format="markdown">
            <i class="fas fa-file-code"></i> Markdown
          </button>
          <button class="export-option" data-format="json">
            <i class="fas fa-file-code"></i> JSON 資料
          </button>
          <button class="export-option" data-format="txt">
            <i class="fas fa-file-alt"></i> 純文字
          </button>
        </div>
      `;

      document.body.appendChild(menu);

      menu.addEventListener('click', (e) => {
        if (e.target === menu) menu.remove();
      });

      menu.querySelectorAll('.export-option').forEach(btn => {
        btn.addEventListener('click', () => {
          this.exportData(btn.dataset.format);
          menu.remove();
        });
      });
    }

    exportData(format) {
      const resultsContainer = document.getElementById('resultsContainer');
      const content = resultsContainer ? resultsContainer.innerText || '' : '';
      const input = document.getElementById('researchInput');
      const query = input?.value || '未命名研究';
      const timestamp = new Date().toISOString().split('T')[0];

      let data, filename, mimeType;

      switch (format) {
        case 'pdf':
          showToast('PDF 匯出功能目前尚無法使用 (´;ω;`)', 'warning');
          return;
        case 'markdown':
          data = `# ${query}\n\n**生成時間:** ${new Date().toLocaleString('zh-TW')}\n\n---\n\n${content}\n`;
          filename = `research_${timestamp}.md`;
          mimeType = 'text/markdown';
          break;
        case 'json':
          data = JSON.stringify({ query, timestamp: new Date().toISOString(), results: content }, null, 2);
          filename = `research_${timestamp}.json`;
          mimeType = 'application/json';
          break;
        case 'txt':
          data = `研究主題: ${query}\n\n生成時間: ${new Date().toLocaleString('zh-TW')}\n\n${content}`;
          filename = `research_${timestamp}.txt`;
          mimeType = 'text/plain';
          break;
      }

      this.downloadFile(data, filename, mimeType);
    }

    downloadFile(data, filename, mimeType) {
      const blob = new Blob([data], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }

  function initExampleQueries() {
    document.querySelectorAll('.example-query-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const query = btn.dataset.query;
        const input = document.getElementById('researchInput');
        if (input) input.value = query;
        document.getElementById('startResearchBtn')?.click();
      });
    });
  }

  // ========== 研究模式管理器（連接後端 API） ==========
  class ResearchManager {
    constructor() {
      this.input = document.getElementById('researchInput');
      this.btn = document.getElementById('startResearchBtn');
      this.resultsContainer = document.getElementById('resultsContainer');
      this.emptyState = document.getElementById('emptyState');

      this.initResearchEvents();
    }

    initResearchEvents() {
      this.btn?.addEventListener('click', () => this.performResearch());
      bindEnterWithIME(this.input, () => this.performResearch(), false);
    }

    async performResearch() {
      const query = this.input?.value.trim();
      if (!query) {
        showToast('請先輸入研究主題喔 (´;ω;`)', 'warning');
        return;
      }

      if (AppState.isBusy()) {
        showToast(`${AppState.getBusyReason()}，請稍等一下喔 (´;ω;\`)`, 'warning');
        return;
      }

      AppState.isResearching = true;
      this.emptyState?.classList.add('hidden');
      this.disableControls();

      const progressMgr = window.researcherManagers?.progressManager;
      progressMgr?.start();
      this.showLoading(query);

      try {
        const startResp = await fetch('/research/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic: query, max_results: 15 })
        });

        if (!startResp.ok) {
          throw new Error(`HTTP ${startResp.status}:${startResp.statusText}`);
        }

        const startData = await startResp.json();
        if (startData.success === false) {
          throw new Error(startData.error || '無法啟動任務');
        }
        if (!startData.job_id) {
          throw new Error('無法啟動研究任務：未返回 job_id');
        }

        await this.pollProgress(startData.job_id, query, progressMgr);

      } catch (error) {
        console.error('[Research] 錯誤:', error);
        this.showError(error.message || '網路連接失敗，請檢查網路狀態 (´;ω;`)');
        AppState.isResearching = false;
        this.enableControls();
        progressMgr?.stop();
      }
    }

    async pollProgress(jobId, query, progressMgr) {
      return new Promise((resolve, reject) => {
        let pollCount = 0;
        const maxPolls = 1200; // 20 分鐘

        const pollTimer = setInterval(async () => {
          pollCount++;
          if (pollCount > maxPolls) {
            clearInterval(pollTimer);
            reject(new Error('研究任務超時（超過 20 分鐘）'));
            return;
          }

          try {
            const resp = await fetch(`/research/progress/${jobId}`);
            if (!resp.ok) return;

            const progress = await resp.json();
            if (progress.status === 'not_found') {
              clearInterval(pollTimer);
              reject(new Error('查無此研究任務'));
              return;
            }

            const total = progress.total || 1;
            const current = progress.current || 0;
            const percent = Math.round((current / total) * 100);

            progressMgr?.updateProgress(
              percent,
              progress.phase || '研究中',
              progress.message || `正在處理 ${current}/${total} 個項目...`
            );

            if (progress.status === "completed") {
              clearInterval(pollTimer);

              const stats = progress.stats || {
                successRate: percent,
                successItems: current,
                totalItems: total,
                wordCount: progress.result_html ? progress.result_html.length : 0
              };

              const hasContent = progress.result_html && progress.result_html.trim().length > 0;
              if (hasContent) {
                this.showResults({
                  success: true,
                  html: progress.result_html,
                  stats: stats
                });

                window.researcherManagers?.historyManager?.addItem(query, progress.result_html);
                window.researcherManagers?.searchSuggestion?.saveToHistory(query);
                showToast("研究完成！✨", "success");
              } else {
                this.showError("後端回傳的研究報告內容為空");
              }

              progressMgr?.complete();
              AppState.isResearching = false;
              this.enableControls();
              resolve();
            } else if (progress.status === 'failed') {
              clearInterval(pollTimer);
              const errorMsg = progress.error || '研究過程發生未知錯誤';
              this.showError(errorMsg);
              progressMgr?.stop();
              AppState.isResearching = false;
              this.enableControls();
              reject(new Error(errorMsg));
            }
          } catch (err) {
            console.error('[Progress] 查詢進度失敗:', err);
          }
        }, 1000);
      });
    }

    showLoading(query) {
      if (!this.resultsContainer) return;
      this.resultsContainer.innerHTML = `
        <div class="loading-state">
          <div class="loading-animation">
            <div class="loading-spinner"></div>
            <div class="loading-text">
              <h3>月讀醬正在努力研究中～</h3>
              <p>主題：${escapeHtml(query)}</p>
              <p class="loading-hint">這可能需要一些時間，請耐心等待喔 ✨</p>
            </div>
          </div>
        </div>
      `;
    }

    showResults(data) {
      if (!this.resultsContainer) return;

      const stats = data.stats || {};
      const successRate = typeof stats.successRate === 'number'
        ? stats.successRate.toFixed(1)
        : (stats.successRate || "0.0");
      const wordCount = stats.wordCount || 0;
      const successItems = stats.successItems || 0;
      const totalItems = stats.totalItems || 0;

      const statsHtml = `
        <div class="research-stats">
          <div class="stat-item">
            <i class="fas fa-check-circle"></i>
            <span>${successRate}</span>
          </div>
          <div class="stat-item">
            <i class="fas fa-file-alt"></i>
            <span>${wordCount} 字</span>
          </div>
          <div class="stat-item">
            <i class="fas fa-list"></i>
            <span>${successItems}/${totalItems}</span>
          </div>
        </div>
      `;

      this.resultsContainer.innerHTML = statsHtml + (data.html || "");
      this.resultsContainer.scrollTop = 0;
    }

    showError(message) {
      if (!this.resultsContainer) return;
      this.resultsContainer.innerHTML = `
        <div class="error-state">
          <div class="error-icon">
            <i class="fas fa-exclamation-triangle"></i>
          </div>
          <h3>哎呀～看起來出了點問題 ∑(￣□￣;)</h3>
          <p>${escapeHtml(message)}</p>
          <button class="retry-btn" onclick="window.app.research.performResearch()">
            <i class="fas fa-redo"></i> 重試一次
          </button>
        </div>
      `;
    }

    disableControls() {
      if (this.btn) {
        this.btn.disabled = true;
        this.btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 研究中...';
      }
      if (this.input) this.input.disabled = true;
      document.querySelectorAll('.nav-center a').forEach(link => link.classList.add('nav-busy'));
    }

    enableControls() {
      if (this.btn) {
        this.btn.disabled = false;
        this.btn.innerHTML = '<i class="fas fa-search"></i> 開始研究';
      }
      if (this.input) this.input.disabled = false;
      document.querySelectorAll('.nav-center a').forEach(link => link.classList.remove('nav-busy'));
    }
  }

  // ========== 端側混合檢索檔案管理 (Hybrid Local RAG) ==========
  class FileManager {
    constructor() {
      this.files = [];
      this.selectedFiles = new Set(JSON.parse(localStorage.getItem('rag_selected_files') || '[]'));
      this.hasInitializedSelection = localStorage.getItem('rag_selected_files') !== null;
      this.filePanel = document.getElementById('filePanel');
      this.toggleBtn = document.getElementById('toggleFilePanel');
      this.quickMenu = document.getElementById('attachQuickMenu');
      this.quickUploadFileBtn = document.getElementById('quickUploadFileBtn');
      this.quickUploadFolderBtn = document.getElementById('quickUploadFolderBtn');
      this.quickTogglePanelBtn = document.getElementById('quickTogglePanelBtn');
      this.closeBtn = document.querySelector('.close-panel-btn');
      this.fileInput = document.getElementById('fileUpload');
      this.folderInput = document.getElementById('folderUpload');
      this.fileList = document.querySelector('.file-list');
      this.allowedExts = new Set([
        '.pdf', '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp',
        '.py', '.js', '.ts', '.md', '.tex', '.txt', '.json'
      ]);
      this.initEventListeners();
      this.syncFilesFromServer();
    }

    saveSelectedState() {
      this.hasInitializedSelection = true;
      localStorage.setItem('rag_selected_files', JSON.stringify(Array.from(this.selectedFiles)));
    }

    getActiveFiles() {
      return this.files
        .map(f => f.name)
        .filter(name => this.selectedFiles.has(name));
    }

    toggleFileSelection(filename) {
      if (this.selectedFiles.has(filename)) {
        this.selectedFiles.delete(filename);
      } else {
        this.selectedFiles.add(filename);
      }
      this.saveSelectedState();
      this.renderFiles();
      this.updatePaperclipBadge();
    }

    closeQuickMenu() {
      this.quickMenu?.classList.add('hidden');
      this.toggleBtn?.classList.remove('menu-open');
    }

    initEventListeners() {
      // 1. 點擊左側「+」號：切換快速選單
      this.toggleBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.quickMenu) {
          this.filePanel?.classList.toggle('collapsed');
          return;
        }
        const isHidden = this.quickMenu.classList.toggle('hidden');
        this.toggleBtn.classList.toggle('menu-open', !isHidden);
      });

      // 2. 快速選單：上傳檔案並展開右側 Local RAG 面板
      this.quickUploadFileBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.closeQuickMenu();
        this.filePanel?.classList.remove('collapsed');
        this.fileInput?.click();
      });

      // 3. 快速選單：上傳整個專案資料夾並展開右側面板
      this.quickUploadFolderBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.closeQuickMenu();
        this.filePanel?.classList.remove('collapsed');
        this.folderInput?.click();
      });

      // 4. 快速選單：開關右側 Local RAG 檔案管理面板
      this.quickTogglePanelBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.closeQuickMenu();
        this.filePanel?.classList.toggle('collapsed');
      });

      // 5. 點擊畫面空白處時：自動收合快速選單與展開中的 Local RAG 面板
      document.addEventListener('click', (e) => {
        const clickedInsideMenu = e.target.closest('.attach-menu-wrapper') || e.target.closest('#toggleFilePanel');
        const clickedInsidePanel = this.filePanel && this.filePanel.contains(e.target);

        if (!clickedInsideMenu) {
          this.closeQuickMenu();
        }

        if (
          this.filePanel &&
          !this.filePanel.classList.contains('collapsed') &&
          !clickedInsidePanel &&
          !clickedInsideMenu
        ) {
          this.filePanel.classList.add('collapsed');
        }
      });

      // 6. 按下 Esc 鍵快速收合選單與面板
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.closeQuickMenu();
          this.filePanel?.classList.add('collapsed');
        }
      });

      this.closeBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.filePanel?.classList.add('collapsed');
      });

      this.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e.target.files));
      this.folderInput?.addEventListener('change', (e) => this.handleFileUpload(e.target.files));
    }

    async syncFilesFromServer() {
      try {
        const res = await fetch('/api/files');
        if (res.ok) {
          const data = await res.json();
          const prevNames = new Set(this.files.map(f => f.name));
          this.files = data.files || [];

          if (!this.hasInitializedSelection) {
            this.files.forEach(f => this.selectedFiles.add(f.name));
          } else {
            const currentNames = new Set(this.files.map(f => f.name));
            this.files.forEach(f => {
              if (!prevNames.has(f.name) && prevNames.size > 0) {
                this.selectedFiles.add(f.name);
              }
            });
            for (const name of Array.from(this.selectedFiles)) {
              if (!currentNames.has(name)) this.selectedFiles.delete(name);
            }
          }
          this.saveSelectedState();
          this.renderFiles();
          this.updatePaperclipBadge();
        }
      } catch (err) {
        console.warn('無法同步端側知識庫檔案:', err);
      }
    }

    updatePaperclipBadge() {
      if (!this.toggleBtn) return;
      this.toggleBtn.querySelector('.rag-file-count-badge')?.remove();
      const activeCount = this.getActiveFiles().length;

      if (activeCount > 0) {
        this.toggleBtn.classList.add('has-active-files');
        this.toggleBtn.title = `端側混合檢索已啟用 (${activeCount}/${this.files.length} 個檔案)`;
        const badge = document.createElement('span');
        badge.className = 'rag-file-count-badge';
        badge.textContent = activeCount;
        this.toggleBtn.appendChild(badge);
      } else {
        this.toggleBtn.classList.remove('has-active-files');
        this.toggleBtn.title = this.files.length > 0
          ? '目前未勾選任何 RAG 檔案（開啟右側面板勾選）'
          : '附加檔案 (開啟端側混合檢索)';
      }
    }

    async handleFileUpload(fileListObj) {
      if (!fileListObj || fileListObj.length === 0) return;
      const formData = new FormData();
      const uploadedNames = [];
      let validCount = 0;

      for (const file of fileListObj) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        const relPath = file.webkitRelativePath || file.name;
        if (relPath.includes('/.git/') || relPath.includes('/build/') || relPath.includes('/node_modules/')) {
          continue;
        }
        if (!this.allowedExts.has(ext)) continue;

        formData.append('files', file, relPath);
        uploadedNames.push(relPath);
        validCount++;
      }

      if (validCount === 0) {
        showToast('未偵測到支援的論文 (.pdf/.tex) 或程式碼 (.cpp/.h/.py) 檔案喔！', 'warning');
        return;
      }

      showToast(`月讀醬正在為 ${validCount} 個檔案建立 BGE + BM25 端側索引...`, 'info', 4000);

      try {
        const res = await fetch('/api/files/upload', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.files) {
          this.files = data.files;
          uploadedNames.forEach(name => this.selectedFiles.add(name));
          this.saveSelectedState();
          this.renderFiles();
          this.updatePaperclipBadge();
          showToast(`✨ 成功建立端側混合檢索索引！共載入 ${this.files.length} 個檔案`, 'success');
        }
      } catch (err) {
        console.error('上傳索引失敗:', err);
        showToast('建立端側索引失敗 (´;ω;`)', 'error');
      } finally {
        if (this.fileInput) this.fileInput.value = '';
        if (this.folderInput) this.folderInput.value = '';
      }
    }

    async deleteFile(filename) {
      try {
        const res = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
          method: 'DELETE'
        });
        const data = await res.json();
        this.files = data.files || [];
        this.selectedFiles.delete(filename);
        this.saveSelectedState();
        this.renderFiles();
        this.updatePaperclipBadge();
        showToast(`已從知識庫移除 ${filename}`, 'info');
      } catch (err) {
        showToast('刪除失敗', 'error');
      }
    }

    renderFiles() {
      if (!this.fileList) return;
      this.fileList.innerHTML = '';

      if (this.files.length === 0) {
        this.fileList.innerHTML = `
          <li class="file-empty-hint" style="text-align:center;padding:24px 12px;color:rgba(255,255,255,0.45);font-size:0.85rem;">
            尚未載入任何檔案，點擊右上角圖示上傳
          </li>
        `;
        return;
      }

      // 頂部狀態與全選/取消工具列
      const activeCount = this.getActiveFiles().length;
      const allSelected = activeCount === this.files.length;
      const toolbarLi = document.createElement('li');
      toolbarLi.className = 'file-list-toolbar';
      toolbarLi.innerHTML = `
        <span class="file-selected-summary">已勾選檢索：<strong>${activeCount}</strong> / ${this.files.length}</span>
        <button type="button" class="file-select-all-btn">${allSelected ? '全部取消' : '全選'}</button>
      `;
      toolbarLi.querySelector('.file-select-all-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        if (allSelected) {
          this.selectedFiles.clear();
        } else {
          this.files.forEach(f => this.selectedFiles.add(f.name));
        }
        this.saveSelectedState();
        this.renderFiles();
        this.updatePaperclipBadge();
      });
      this.fileList.appendChild(toolbarLi);

      // 各檔案項目（含勾選開關）
      this.files.forEach((file) => {
        const isSelected = this.selectedFiles.has(file.name);
        const isPdf = file.name.toLowerCase().endsWith('.pdf');

        const li = document.createElement('li');
        li.className = `file-item ${isSelected ? 'is-selected' : 'is-unselected'}`;
        li.title = `${file.name}\n（點擊切換是否納入本次對話檢索）`;

        li.innerHTML = `
          <span class="file-select-check" aria-hidden="true">
            <i class="fa-solid ${isSelected ? 'fa-circle-check' : 'fa-regular fa-circle'}"></i>
          </span>
          <i class="fa ${isPdf ? 'fa-file-pdf' : 'fa-file-code'}" style="margin-right:8px;color:#ae8ddd;flex-shrink:0;"></i>
          <span class="file-name">${escapeHtml(file.name)}</span>
          <span class="file-chunk-badge">${file.chunks} chunks</span>
          <button type="button" class="delete-file" title="移除索引"><i class="fa fa-trash"></i></button>
        `;

        // 點擊整列切換勾選狀態
        li.addEventListener('click', (e) => {
          e.stopPropagation();
          this.toggleFileSelection(file.name);
        });

        // 點擊刪除按鈕僅刪除檔案，不觸發切換也不收合面板
        li.querySelector('.delete-file').addEventListener('click', (e) => {
          e.stopPropagation();
          this.deleteFile(file.name);
        });

        this.fileList.appendChild(li);
      });
    }
  }

  // ========== 長期記憶筆記管理 (Memory Inspector) ==========
  class MemoryManager {
    constructor() {
      this.facts = [];
      this.maxFacts = DEFAULT_MAX_FACTS;
      this.editingIndex = null;

      this.modal = document.getElementById('memoryModal');
      this.openBtn = document.getElementById('openMemoryModalBtn');
      this.closeBtn = document.getElementById('closeMemoryModalBtn');
      this.doneBtn = document.getElementById('doneMemoryModalBtn');
      this.newInput = document.getElementById('newMemoryInput');
      this.addBtn = document.getElementById('addMemoryBtn');
      this.listEl = document.getElementById('memoryFactsList');
      this.countBadge = document.getElementById('memoryCountBadge');
      this.clearAllBtn = document.getElementById('modalClearAllMemoryBtn');

      this.initEvents();
    }

    initEvents() {
      this.openBtn?.addEventListener('click', () => this.open());
      this.closeBtn?.addEventListener('click', () => this.close());
      this.doneBtn?.addEventListener('click', () => this.close());

      this.modal?.addEventListener('click', (e) => {
        if (e.target === this.modal) this.close();
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.modal && !this.modal.classList.contains('hidden')) {
          this.close();
        }
      });

      this.addBtn?.addEventListener('click', () => this.addFact());
      bindEnterWithIME(this.newInput, () => this.addFact(), false);

      this.clearAllBtn?.addEventListener('click', async () => {
        if (AppState.isBusy()) {
          showToast(`${AppState.getBusyReason()}，請稍候喔～`, 'warning');
          return;
        }
        if (!confirm('確定要清空月讀醬的所有長期記憶嗎？')) return;
        try {
          await fetch('/clear_memory', { method: 'POST' });
          this.facts = [];
          this.editingIndex = null;
          this.render();
          showToast('已清空所有長期記憶', 'info');
        } catch {
          showToast('清空失敗', 'error');
        }
      });
    }

    async open() {
      if (!this.modal) return;
      this.editingIndex = null;
      this.modal.classList.remove('hidden');
      await this.loadFacts();
    }

    close() {
      this.editingIndex = null;
      this.modal?.classList.add('hidden');
    }

    async loadFacts() {
      try {
        const res = await fetch('/api/memory');
        if (res.ok) {
          const data = await res.json();
          this.facts = data.facts || [];
          this.maxFacts = data.max_facts || DEFAULT_MAX_FACTS;
          this.render();
        }
      } catch (e) {
        console.error('載入長期記憶失敗:', e);
        showToast('無法載入記憶筆記', 'error');
      }
    }

    async addFact() {
      const text = this.newInput?.value.trim();
      if (!text) return;

      try {
        const res = await fetch('/api/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fact: text })
        });
        const data = await res.json();
        if (data.success) {
          this.facts = data.facts || [];
          this.newInput.value = '';
          this.render();
          showToast('已寫入月讀醬的記憶筆記 ✨', 'success', 2000);
        } else {
          showToast(data.error || '新增失敗', 'warning');
        }
      } catch {
        showToast('新增記憶失敗', 'error');
      }
    }

    async saveEditedFact(index, newText) {
      const clean = newText.trim();
      if (!clean) {
        showToast('記憶內容不能為空', 'warning');
        return;
      }
      try {
        const res = await fetch(`/api/memory/${index}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fact: clean })
        });
        const data = await res.json();
        if (data.success) {
          this.facts = data.facts || [];
          this.editingIndex = null;
          this.render();
          showToast('記憶已更新', 'success', 1800);
        }
      } catch {
        showToast('更新失敗', 'error');
      }
    }

    async deleteFact(index) {
      try {
        const res = await fetch(`/api/memory/${index}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
          this.facts = data.facts || [];
          if (this.editingIndex === index) this.editingIndex = null;
          this.render();
          showToast('已移除該條記憶', 'info', 1800);
        }
      } catch {
        showToast('刪除失敗', 'error');
      }
    }

    render() {
      if (this.countBadge) {
        this.countBadge.textContent = `${this.facts.length} / ${this.maxFacts}`;
      }
      if (!this.listEl) return;
      this.listEl.innerHTML = '';

      if (this.facts.length === 0) {
        this.listEl.innerHTML = `
          <li class="memory-empty-state">
            月讀醬目前還沒有記下任何長期筆記喔～<br>
            多跟她聊聊天，或在上方輸入框手動新增吧 (•ω•)/
          </li>
        `;
        return;
      }

      this.facts.forEach((fact, idx) => {
        const li = document.createElement('li');
        li.className = 'memory-fact-item';

        const numSpan = document.createElement('span');
        numSpan.className = 'memory-fact-index';
        numSpan.textContent = `#${idx + 1}`;
        li.appendChild(numSpan);

        if (this.editingIndex === idx) {
          const editInput = document.createElement('input');
          editInput.type = 'text';
          editInput.className = 'memory-fact-edit-input';
          editInput.value = fact;
          editInput.maxLength = 200;

          const actions = document.createElement('div');
          actions.className = 'memory-fact-actions';

          const saveBtn = document.createElement('button');
          saveBtn.type = 'button';
          saveBtn.className = 'memory-icon-btn save';
          saveBtn.title = '儲存修改';
          saveBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
          saveBtn.addEventListener('click', () => this.saveEditedFact(idx, editInput.value));

          const cancelBtn = document.createElement('button');
          cancelBtn.type = 'button';
          cancelBtn.className = 'memory-icon-btn';
          cancelBtn.title = '取消';
          cancelBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>';
          cancelBtn.addEventListener('click', () => {
            this.editingIndex = null;
            this.render();
          });

          bindEnterWithIME(editInput, () => this.saveEditedFact(idx, editInput.value), false);
          actions.append(saveBtn, cancelBtn);
          li.append(editInput, actions);
          requestAnimationFrame(() => editInput.focus());
        } else {
          const textSpan = document.createElement('span');
          textSpan.className = 'memory-fact-text';
          textSpan.textContent = fact;

          const actions = document.createElement('div');
          actions.className = 'memory-fact-actions';

          const editBtn = document.createElement('button');
          editBtn.type = 'button';
          editBtn.className = 'memory-icon-btn';
          editBtn.title = '編輯此條記憶';
          editBtn.innerHTML = '<i class="fa-solid fa-pen"></i>';
          editBtn.addEventListener('click', () => {
            this.editingIndex = idx;
            this.render();
          });

          const delBtn = document.createElement('button');
          delBtn.type = 'button';
          delBtn.className = 'memory-icon-btn delete';
          delBtn.title = '刪除此條記憶';
          delBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
          delBtn.addEventListener('click', () => this.deleteFact(idx));

          actions.append(editBtn, delBtn);
          li.append(textSpan, actions);
        }

        this.listEl.appendChild(li);
      });
    }
  }

  // ========== UI 控制器 ==========
  class UIController {
    constructor() {
      this.initTabs();
      this.initExpandToggle();
      this.initIntimacyAlignment();
      this.initModals();
    }

    initTabs() {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function () {
          document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          this.classList.add('active');

          const chatTab = document.getElementById('sidebar-chat');
          const fileTab = document.getElementById('sidebar-file');
          const historyTab = document.getElementById('sidebar-research-history');

          if (chatTab) chatTab.style.display = this.dataset.tab === 'chat' ? '' : 'none';
          if (fileTab) fileTab.style.display = this.dataset.tab === 'file' ? '' : 'none';
          if (historyTab) historyTab.style.display = this.dataset.tab === 'research-history' ? '' : 'none';
        });
      });
    }

    initExpandToggle() {
      const btn = document.querySelector('.edge-toggle-btn');
      const container = document.querySelector('.container');
      const btnIcon = btn?.querySelector('i');
      if (!btn || !container || !btnIcon) return;

      const updateState = (isExpanded) => {
        container.classList.toggle('chat-expanded', isExpanded);
        btn.classList.toggle('collapsed', isExpanded);
        btn.title = isExpanded ? "展開Live2D" : "收合Live2D";
        btnIcon.className = isExpanded ? 'fa fa-chevron-right' : 'fa fa-chevron-left';
        localStorage.setItem('chat_expanded', isExpanded);

        const intimacy = document.getElementById('intimacy-display');
        if (intimacy) intimacy.style.left = '';

        requestAnimationFrame(() => window.app?.live2d?.handleResize());
        setTimeout(() => {
          window.app?.live2d?.handleResize();
        }, 400);
      };

      const savedState = localStorage.getItem('chat_expanded') === 'true';
      updateState(savedState);

      btn.addEventListener('click', () => {
        updateState(!container.classList.contains('chat-expanded'));
      });
    }

    alignIntimacy() {
      const intimacy = document.getElementById('intimacy-display');
      if (intimacy) intimacy.style.left = '';
    }

    initIntimacyAlignment() {
      this.alignIntimacy();
    }

    initModals() {
      const overlay = document.getElementById("audio-permission-overlay");
      if (overlay) {
        overlay.style.display = localStorage.getItem("hasVisited") ? "none" : "flex";
      }
      document.getElementById("startChatBtn")?.addEventListener("click", initAudioPermission);
      document.getElementById("showTermsBtn")?.addEventListener("click", showTerms);
      document.getElementById("aboutBtn")?.addEventListener("click", showTerms);
      document.getElementById("closeTermsBtn")?.addEventListener("click", closeTerms);

      const termsModal = document.getElementById("termsModal");
      termsModal?.addEventListener("click", (e) => {
        if (e.target === termsModal) closeTerms();
      });
    }
  }

  // ========== 應用初始化 ==========
  document.addEventListener('DOMContentLoaded', function () {
    const live2dController = new Live2DController();
    const chatManager = new ChatManager();
    const fileManager = new FileManager();
    const memoryManager = new MemoryManager();
    const uiController = new UIController();
    const researchManager = new ResearchManager();

    const searchSuggestion = new SearchSuggestionManager();
    const accordion = new AccordionManager();
    const progressManager = new ProgressManager();
    const viewManager = new ViewManager();
    const historyManager = new ResearchHistoryManager();
    const exportManager = new ExportManager();

    initExampleQueries();
    live2dController.init();

    // Live2D 動作控制
    document.querySelectorAll('#controls .control-item').forEach(item => {
      const motionName = item.dataset.motion;
      if (motionName) {
        item.addEventListener('click', () => live2dController.playMotion(motionName));
      }
    });

    // 記憶與語音控制（單一綁定）
    document.getElementById('replayAudio')?.addEventListener('click', replayLastAudio);
    document.getElementById('clearSession')?.addEventListener('click', () => chatManager.clearSession());
    document.getElementById('clearAllMemory')?.addEventListener('click', () => chatManager.clearAllMemory());
    document.getElementById('voiceInputBtn')?.addEventListener('click', () => {
      showToast('語音輸入功能還沒開發好 (´;ω;`)', 'warning');
    });

    // Live2D 點擊互動
    document.getElementById('live2d-area')?.addEventListener('click', () => {
      live2dController.playMotion('Tap');
    });

    window.app = {
      live2d: live2dController,
      chat: chatManager,
      file: fileManager,
      memory: memoryManager,
      ui: uiController,
      research: researchManager
    };

    window.researcherManagers = {
      searchSuggestion,
      accordion,
      progressManager,
      viewManager,
      historyManager,
      exportManager
    };

    updateIntimacyDisplay();
  });
})();

// =================== 視圖切換功能（單一來源版） ===================
(function () {
  'use strict';

  function setNavActive(viewName) {
    document.getElementById('nav-chat')?.classList.toggle('active', viewName === 'chat');
    document.getElementById('nav-research')?.classList.toggle('active', viewName === 'research');
  }

  window.switchMainView = async function (viewName) {
    if (viewName === 'chat' && window.AppState?.isResearching) {
      const ok = await showCustomConfirm({
        title: '研究進行中',
        message: '月讀醬正在進行研究中，確定要切換嗎？\n（研究會在背景繼續進行）',
        confirmText: '仍要切換'
      });
      if (!ok) return;
    }
    if (viewName === 'research' && window.AppState?.isChatting) {
      const ok = await showCustomConfirm({
        title: '等待回覆中',
        message: '正在等待月讀醬的回覆，確定要切換嗎？',
        confirmText: '仍要切換'
      });
      if (!ok) return;
    }

    const currentView = localStorage.getItem('currentView');
    if (currentView && currentView !== 'settings') {
      localStorage.setItem('previousView', currentView);
    }

    document.querySelectorAll('.view-section').forEach(section => {
      section.style.display = 'none';
      section.classList.remove('active');
      section.classList.add('hidden');
    });

    const targetView = document.getElementById(`${viewName}-view`);
    if (!targetView) return;

    targetView.style.display = 'flex';
    targetView.classList.add('active');
    targetView.classList.remove('hidden');

    setNavActive(viewName);
    localStorage.setItem('currentView', viewName);
  };

  document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('settingsBtn')?.addEventListener('click', (e) => {
      e.preventDefault();
      window.switchMainView('settings');
    });

    document.getElementById('backToChatBtn')?.addEventListener('click', (e) => {
      e.preventDefault();
      const previousView = localStorage.getItem('previousView') || 'chat';
      window.switchMainView(previousView);
    });

    document.getElementById('nav-chat')?.addEventListener('click', (e) => {
      e.preventDefault();
      window.switchMainView('chat');
    });

    document.getElementById('nav-research')?.addEventListener('click', (e) => {
      e.preventDefault();
      window.switchMainView('research');
    });

    const savedView = localStorage.getItem('currentView') || 'chat';
    window.switchMainView(savedView === 'settings' ? 'chat' : savedView);
  });
})();

// =================== 設定頁面與模型快速切換器 ===================
(function () {
  'use strict';

  const SETTINGS_KEY = 'tsukuyomi_settings';

  const defaultSettings = {
    theme: 'dark',
    fontSize: 16,
    tavilyApiKey: '',
    smartLLM: DEFAULT_RESEARCH_MODEL,
    smallLLM: DEFAULT_CHAT_MODEL,
    memoryLLM: DEFAULT_MEMORY_MODEL,
    maxTokens: DEFAULT_MAX_TOKENS,
    maxMemory: DEFAULT_MAX_MEMORY,
    maxFactsPerUser: DEFAULT_MAX_FACTS
  };

  async function loadSettings() {
    try {
      const response = await fetch('/api/settings', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const serverSettings = await response.json();
        if (serverSettings && typeof serverSettings === 'object') {
          const validSettings = {
            theme: serverSettings.theme || 'dark',
            fontSize: Number(serverSettings.fontSize) || 16,
            tavilyApiKey: serverSettings.tavilyApiKey || '',
            smartLLM: serverSettings.smartLLM || DEFAULT_RESEARCH_MODEL,
            smallLLM: serverSettings.smallLLM || DEFAULT_CHAT_MODEL,
            memoryLLM: serverSettings.memoryLLM || DEFAULT_MEMORY_MODEL,
            maxTokens: Number(serverSettings.maxTokens) || DEFAULT_MAX_TOKENS,
            maxMemory: Number(serverSettings.maxMemory) || DEFAULT_MAX_MEMORY,
            maxFactsPerUser: Number(serverSettings.maxFactsPerUser) || DEFAULT_MAX_FACTS
          };
          localStorage.setItem(SETTINGS_KEY, JSON.stringify(validSettings));
          return validSettings;
        }
      }
    } catch (e) {
      console.warn('從伺服器載入設定失敗，嘗試讀取本地快取:', e);
    }

    try {
      const saved = localStorage.getItem(SETTINGS_KEY);
      if (saved) {
        return { ...defaultSettings, ...JSON.parse(saved) };
      }
    } catch (e) {
      console.warn('localStorage 解析失敗:', e);
    }
    return { ...defaultSettings };
  }

  async function saveSettings(settings) {
    if (!settings || typeof settings !== 'object') return false;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (e) {
      console.warn('localStorage 儲存失敗:', e);
    }

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings })
      });
      return res.ok;
    } catch (err) {
      console.warn('伺服器同步設定失敗（已儲存於本地）:', err);
      return true;
    }
  }

  function applyTheme(theme) {
    document.body.classList.toggle('light-theme', theme === 'light');
  }

  function applyFontSize(size) {
    const fontSize = parseInt(size, 10) || 16;
    document.documentElement.style.setProperty('--chat-font-size', `${fontSize}px`);
  }

  async function initSettingsPage() {
    const settings = await loadSettings();

    applyTheme(settings.theme);
    applyFontSize(settings.fontSize);
    updateChatModelBadge(settings.smallLLM || DEFAULT_CHAT_MODEL);

    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
      themeSelect.value = settings.theme;
      themeSelect.addEventListener('change', (e) => {
        settings.theme = e.target.value;
        applyTheme(settings.theme);
        saveSettings(settings);
        showToast('主題已更新', 'success');
      });
    }

    const fontSizeSlider = document.getElementById('font-size-slider');
    const fontSizeValue = document.getElementById('font-size-value');
    if (fontSizeSlider && fontSizeValue) {
      fontSizeSlider.value = settings.fontSize;
      fontSizeValue.textContent = `${settings.fontSize}px`;

      fontSizeSlider.addEventListener('input', (e) => {
        const size = parseInt(e.target.value, 10);
        fontSizeValue.textContent = `${size}px`;
        applyFontSize(size);
      });

      fontSizeSlider.addEventListener('change', (e) => {
        settings.fontSize = parseInt(e.target.value, 10);
        saveSettings(settings);
      });
    }

    const tavilyApiInput = document.getElementById('tavily-api-input');
    if (tavilyApiInput) {
      tavilyApiInput.value = settings.tavilyApiKey;
      tavilyApiInput.addEventListener('blur', (e) => {
        settings.tavilyApiKey = e.target.value.trim();
      });
    }

    const toggleTavilyBtn = document.getElementById('toggle-tavily');
    if (toggleTavilyBtn && tavilyApiInput) {
      toggleTavilyBtn.addEventListener('click', () => {
        const isPassword = tavilyApiInput.type === 'password';
        tavilyApiInput.type = isPassword ? 'text' : 'password';
        toggleTavilyBtn.innerHTML = isPassword
          ? '<i class="fa fa-eye-slash"></i>'
          : '<i class="fa fa-eye"></i>';
      });
    }

    const smartLLMInput = document.getElementById('smart-llm-input');
    if (smartLLMInput) {
      smartLLMInput.value = settings.smartLLM;
      smartLLMInput.addEventListener('blur', (e) => {
        settings.smartLLM = e.target.value.trim();
      });
    }

    const smallLLMInput = document.getElementById('small-llm-input');
    if (smallLLMInput) {
      smallLLMInput.value = settings.smallLLM;
      smallLLMInput.addEventListener('blur', (e) => {
        settings.smallLLM = e.target.value.trim();
      });
    }

    const memoryLLMInput = document.getElementById('memory-llm-input');
    if (memoryLLMInput) {
      memoryLLMInput.value = settings.memoryLLM || DEFAULT_MEMORY_MODEL;
      memoryLLMInput.addEventListener('blur', (e) => {
        settings.memoryLLM = e.target.value.trim() || DEFAULT_MEMORY_MODEL;
      });
    }

    const maxTokensSlider = document.getElementById('max-tokens-input');
    const maxTokensValue = document.getElementById('max-tokens-value');
    if (maxTokensSlider && maxTokensValue) {
      maxTokensSlider.value = settings.maxTokens || DEFAULT_MAX_TOKENS;
      maxTokensValue.textContent = `${maxTokensSlider.value}`;
      maxTokensSlider.addEventListener('input', (e) => {
        maxTokensValue.textContent = `${e.target.value}`;
      });
      maxTokensSlider.addEventListener('change', (e) => {
        settings.maxTokens = parseInt(e.target.value, 10) || DEFAULT_MAX_TOKENS;
      });
    }

    const maxMemorySlider = document.getElementById('max-memory-input');
    const maxMemoryValue = document.getElementById('max-memory-value');
    if (maxMemorySlider && maxMemoryValue) {
      maxMemorySlider.value = settings.maxMemory || DEFAULT_MAX_MEMORY;
      maxMemoryValue.textContent = `${maxMemorySlider.value} 輪`;
      maxMemorySlider.addEventListener('input', (e) => {
        maxMemoryValue.textContent = `${e.target.value} 輪`;
      });
      maxMemorySlider.addEventListener('change', (e) => {
        settings.maxMemory = parseInt(e.target.value, 10) || DEFAULT_MAX_MEMORY;
      });
    }

    const maxFactsSlider = document.getElementById('max-facts-input');
    const maxFactsValue = document.getElementById('max-facts-value');
    if (maxFactsSlider && maxFactsValue) {
      maxFactsSlider.value = settings.maxFactsPerUser || DEFAULT_MAX_FACTS;
      maxFactsValue.textContent = `${maxFactsSlider.value} 條`;
      maxFactsSlider.addEventListener('input', (e) => {
        maxFactsValue.textContent = `${e.target.value} 條`;
      });
      maxFactsSlider.addEventListener('change', (e) => {
        settings.maxFactsPerUser = parseInt(e.target.value, 10) || DEFAULT_MAX_FACTS;
      });
    }

    const saveSettingsBtn = document.getElementById('save-settings-btn');
    if (saveSettingsBtn) {
      saveSettingsBtn.addEventListener('click', async () => {
        if (tavilyApiInput) settings.tavilyApiKey = tavilyApiInput.value.trim();
        if (smartLLMInput) settings.smartLLM = smartLLMInput.value.trim() || DEFAULT_RESEARCH_MODEL;
        if (smallLLMInput) settings.smallLLM = smallLLMInput.value.trim() || DEFAULT_CHAT_MODEL;
        if (memoryLLMInput) settings.memoryLLM = memoryLLMInput.value.trim() || DEFAULT_MEMORY_MODEL;
        if (maxTokensSlider) settings.maxTokens = parseInt(maxTokensSlider.value, 10) || DEFAULT_MAX_TOKENS;
        if (maxMemorySlider) settings.maxMemory = parseInt(maxMemorySlider.value, 10) || DEFAULT_MAX_MEMORY;
        if (maxFactsSlider) settings.maxFactsPerUser = parseInt(maxFactsSlider.value, 10) || DEFAULT_MAX_FACTS;

        updateChatModelBadge(settings.smallLLM);
        const ok = await saveSettings(settings);
        showToast(ok ? '設定已儲存' : '儲存失敗，請稍後再試', ok ? 'success' : 'error');
      });
    }

    const testApiBtn = document.getElementById('test-api-btn');
    if (testApiBtn) {
      testApiBtn.addEventListener('click', async () => {
        const apiKey = tavilyApiInput ? tavilyApiInput.value.trim() : settings.tavilyApiKey;
        if (!apiKey) {
          showToast('請先輸入 Tavily API Key', 'warning');
          return;
        }

        testApiBtn.disabled = true;
        testApiBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 驗證中...';

        try {
          const response = await fetch('/api/settings/test-api', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ apiKey })
          });
          const result = await response.json();
          showToast(result.message || (result.success ? 'API Key 驗證成功' : '驗證失敗'), result.success ? 'success' : 'warning');
        } catch (e) {
          showToast(`驗證失敗: ${e.message}`, 'error');
        } finally {
          testApiBtn.disabled = false;
          testApiBtn.innerHTML = '<i class="fa fa-vial"></i> 測試 API';
        }
      });
    }
  }

  function setupQuickModelSwitcher() {
    const selectorBtn = document.getElementById('modelSelectorBtn') || document.querySelector('.chat-model-badge');
    const dropdownMenu = document.getElementById('modelDropdownMenu');
    const customInput = document.getElementById('customModelQuickInput');
    const applyCustomBtn = document.getElementById('applyCustomModelBtn');

    if (!selectorBtn || !dropdownMenu || selectorBtn.dataset.bound === 'true') return;
    selectorBtn.dataset.bound = 'true';

    selectorBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isHidden = dropdownMenu.classList.toggle('hidden');
      selectorBtn.classList.toggle('active', !isHidden);
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.chat-model-badge-wrapper')) {
        dropdownMenu.classList.add('hidden');
        selectorBtn.classList.remove('active');
      }
    });

    async function selectNewModel(newModelName) {
      if (!newModelName) return;
      updateChatModelBadge(newModelName);
      dropdownMenu.classList.add('hidden');
      selectorBtn.classList.remove('active');

      const smallInput = document.getElementById('small-llm-input');
      if (smallInput) smallInput.value = newModelName;

      const settings = await loadSettings();
      settings.smallLLM = newModelName;
      await saveSettings(settings);

      showToast(`聊天模型已切換為：${newModelName}`, 'success', 2500);
    }

    document.querySelectorAll('.model-option').forEach(option => {
      option.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNewModel(option.dataset.model);
      });
    });

    if (applyCustomBtn && customInput) {
      const handleCustomApply = (e) => {
        if (e) e.stopPropagation();
        const val = customInput.value.trim();
        if (val) {
          selectNewModel(val);
          customInput.value = '';
        }
      };
      applyCustomBtn.addEventListener('click', handleCustomApply);
      bindEnterWithIME(customInput, handleCustomApply, false);
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    initSettingsPage();
    setupQuickModelSwitcher();
  });

  window.getSettings = loadSettings;
  window.saveAppSettings = saveSettings;
  window.initSettingsPage = initSettingsPage;
})();