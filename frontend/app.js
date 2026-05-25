const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const loading = document.querySelector("#loading");
const newChat = document.querySelector("#newChat");

const SESSION_KEY = "rag_chat_session_id";
const CHAT_KEY = "rag_chat_messages";

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function loadMessages() {
  const saved = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
  saved.forEach((item) => addMessage(item.role, item.text, item.meta, false));
}

function persistMessage(role, text, meta = "") {
  const saved = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
  saved.push({ role, text, meta });
  localStorage.setItem(CHAT_KEY, JSON.stringify(saved.slice(-40)));
}

function addMessage(role, text, meta = "", persist = true) {
  const bubble = document.createElement("article");
  bubble.className = `message ${role}`;
  bubble.textContent = text;

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;

  if (persist) {
    persistMessage(role, text, meta);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  input.disabled = true;
  loading.hidden = false;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: getSessionId(), message }),
    });
    const data = await response.json();

    if (!response.ok) {
      const detail = data.detail?.error || data.error || "Request failed";
      throw new Error(detail);
    }

    const sourceText = data.sources?.length
      ? ` · ${data.sources.map((source) => `${source.title} (${source.score})`).join(", ")}`
      : "";
    const meta = `${data.retrievedChunks} chunk(s) retrieved${
      data.tokensUsed ? ` · ${data.tokensUsed} tokens` : ""
    }${sourceText}`;
    addMessage("assistant", data.reply, meta);
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    input.disabled = false;
    loading.hidden = true;
    input.focus();
  }
});

newChat.addEventListener("click", () => {
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem(CHAT_KEY);
  messages.innerHTML = "";
  getSessionId();
  input.focus();
});

getSessionId();
loadMessages();
