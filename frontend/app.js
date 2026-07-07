const API_BASE = "/api";

let messages = [];
let loading  = false;

const chatArea       = document.getElementById("chatArea");
const emptyState     = document.getElementById("emptyState");
const messagesEl     = document.getElementById("messages");
const inputField     = document.getElementById("inputField");
const sendBtn        = document.getElementById("sendBtn");
const clearBtn       = document.getElementById("clearBtn");
const suggestionChips = document.getElementById("suggestionChips");
const feedbackBtn    = document.getElementById("feedbackBtn");
const feedbackOverlay = document.getElementById("feedbackOverlay");
const feedbackClose  = document.getElementById("feedbackClose");

const ROLE_LABELS = {
  parent:        "👨‍👩‍👧 Parent",
  professionnel: "🏥 Professionnel",
  ambigu:        "❓ Indéterminé",
};
const PHASE_LABELS = {
  grossesse:       "🤰 Grossesse",
  "post-natalite": "🍼 Post-natalité",
  bebe:            "👶 Bébé",
  enfance:         "🧒 Enfance",
  adolescence:     "🧑 Adolescence",
  ambigu:          "❓ Indéterminé",
};

function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Hyperlink detection (urls))
  html = html.replace(
    /(https?:\/\/[^\s<>"')\]]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  const lines = html.split("\n");
  const out = [];
  let inList = false;
  for (const line of lines) {
    const m = line.match(/^\s*[-•]\s+(.*)/);
    if (m) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${m[1]}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim() !== "") out.push(`<p>${line}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  return out.join("\n");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Input state management
function updateInputState() {
  inputField.disabled = loading;
  sendBtn.disabled = loading || inputField.value.trim() === "";
}

// Rendering the chat messages
function renderMessages() {
  updateInputState();

  if (messages.length === 0) {
    emptyState.style.display  = "flex";
    messagesEl.style.display  = "none";
    clearBtn.style.display    = "none";
    return;
  }

  emptyState.style.display = "none";
  messagesEl.style.display = "flex";
  clearBtn.style.display   = "inline-block";
  messagesEl.innerHTML     = "";

  messages.forEach((msg) => {
    const row = document.createElement("div");
    row.className = `message-row ${msg.role === "user" ? "user-row" : "assistant-row"}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${msg.role === "user" ? "user-bubble" : "assistant-bubble"}`;

    if (msg.role === "user") {
      const p = document.createElement("p");
      p.textContent = msg.content;
      bubble.appendChild(p);
    } else {
      bubble.appendChild(buildSpeakButton(msg.content, msg.metadata?.language));
      if (msg.metadata) bubble.appendChild(buildMetaBar(msg.metadata));
      const contentDiv = document.createElement("div");
      contentDiv.innerHTML = renderMarkdown(msg.content);
      bubble.appendChild(contentDiv);

      // Event cards
      if (msg.events && msg.events.length > 0) {
        bubble.appendChild(buildEventCards(msg.events));
      }
      // Service cards
      if (msg.services && msg.services.length > 0) {
        bubble.appendChild(buildServiceCards(msg.services));
      }

      bubble.appendChild(buildBubbleFeedbackBtn());
      if (msg.sources && msg.sources.length > 0) bubble.appendChild(buildSources(msg.sources));
    }

    row.appendChild(bubble);
    messagesEl.appendChild(row);
  });

  if (loading) messagesEl.appendChild(buildTypingIndicator());
  chatArea.scrollTop = chatArea.scrollHeight;
}

function buildMetaBar(meta) {
  const bar = document.createElement("div");
  const isUrgent = meta.urgence === "oui";
  bar.className = `meta-bar ${isUrgent ? "meta-urgent" : ""}`;

  if (isUrgent)          bar.appendChild(makePill("🚨 Urgence", "urgent"));
  if (meta.role_detecte) bar.appendChild(makePill(ROLE_LABELS[meta.role_detecte] || meta.role_detecte));
  if (meta.phase)        bar.appendChild(makePill(PHASE_LABELS[meta.phase] || meta.phase));
  if (meta.language) {
    const lang = meta.language.charAt(0).toUpperCase() + meta.language.slice(1);
    bar.appendChild(makePill(`🌐 ${lang}`));
  }
  return bar;
}

function makePill(text, extraClass = "") {
  const span = document.createElement("span");
  span.className = `meta-pill ${extraClass}`;
  span.textContent = text;
  return span;
}

function buildSources(urls) {
  const wrap   = document.createElement("div");
  wrap.className = "sources";

  const toggle = document.createElement("button");
  toggle.className = "source-toggle";
  toggle.innerHTML = `<span>📚 Sources (${urls.length})</span><span class="source-chevron">▼</span>`;

  const list = document.createElement("ul");
  list.className = "source-list";
  list.style.display = "none";

  urls.forEach((url) => {
    const li = document.createElement("li");
    const a  = document.createElement("a");
    a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = url;
    li.appendChild(a);
    list.appendChild(li);
  });

  let open = false;
  toggle.addEventListener("click", () => {
    open = !open;
    list.style.display = open ? "flex" : "none";
    toggle.querySelector(".source-chevron").textContent = open ? "▲" : "▼";
  });

  wrap.appendChild(toggle);
  wrap.appendChild(list);
  return wrap;
}

function buildTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row assistant-row";
  row.innerHTML = `<div class="bubble assistant-bubble typing"><span></span><span></span><span></span></div>`;
  return row;
}

function showError(message) {
  const banner = document.createElement("div");
  banner.className = "error-banner";
  // Error to user
  banner.innerHTML = `
    <strong>Une erreur est survenue.</strong>
    Veuillez réessayer dans quelques instants.
    Si le problème persiste, actualisez la page.
  `;
  // Error to logs
  console.error("[Chat error]", message);
  messagesEl.appendChild(banner);
  chatArea.scrollTop = chatArea.scrollHeight;
}

// Text-to-speech
let currentUtterance = null;
let currentSpeakBtn  = null;

function stripMarkdownForSpeech(text) {
  return text
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/^\s*[-•]\s+/gm, "")
    .trim();
}

function langToBCP47(language) {
  const map = {
    francais: "fr-FR", french: "fr-FR",
    english: "en-US",  anglais: "en-US",
    arabic: "ar-SA",   arabe: "ar-SA",
    spanish: "es-ES",  espagnol: "es-ES",
  };
  if (!language) return "fr-FR";
  return map[language.toLowerCase()] || "fr-FR";
}

function stopSpeaking() {
  window.speechSynthesis.cancel();
  if (currentSpeakBtn) {
    currentSpeakBtn.classList.remove("speaking");
    currentSpeakBtn.textContent = "🔊";
  }
  currentUtterance = null;
  currentSpeakBtn  = null;
}

function buildBubbleFeedbackBtn() {
  const btn = document.createElement("button");
  btn.className = "bubble-feedback-btn";
  btn.textContent = "💬 Évaluer cette réponse";
  btn.title = "Donner votre avis sur cette réponse";
  btn.addEventListener("click", () => {
    feedbackOverlay.style.display = "flex";
  });
  return btn;
}

function buildSpeakButton(text, language) {
  const btn = document.createElement("button");
  btn.className = "speak-btn";
  btn.textContent = "🔊";
  btn.title = "Lire à voix haute";

  btn.addEventListener("click", () => {
    const isThisOneSpeaking = currentSpeakBtn === btn;
    stopSpeaking();
    if (isThisOneSpeaking) return;

    if (!("speechSynthesis" in window)) {
      alert("La lecture vocale n'est pas prise en charge par ce navigateur.");
      return;
    }

    const utterance = new SpeechSynthesisUtterance(stripMarkdownForSpeech(text));
    utterance.lang = langToBCP47(language);
    utterance.rate = 1;
    utterance.onend  = () => stopSpeaking();
    utterance.onerror = () => stopSpeaking();

    currentUtterance = utterance;
    currentSpeakBtn  = btn;
    btn.classList.add("speaking");
    btn.textContent = "⏹";
    window.speechSynthesis.speak(utterance);
  });

  return btn;
}

// Networking
async function sendMessage(text) {
  messages.push({ role: "user", content: text });
  loading = true;
  renderMessages();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: messages
          .filter((m) => m.role === "user" || m.role === "assistant")
          .slice(0, -1)
          .map((m) => ({ role: m.role, content: m.content })),
      }),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const err = await res.json(); if (err.detail) detail = err.detail; } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    loading = false;
    messages.push({
      role: "assistant",
      content: data.answer,
      sources: data.sources || [],
      metadata: data.metadata || null,
      events: data.events || [],
      services: data.services || [],
    });
    renderMessages();
  } catch (e) {
    loading = false;
    renderMessages();
    showError(e.message);
  }
}

// Input handling
function handleSend() {
  const text = inputField.value.trim();
  if (!text || loading) return;
  inputField.value = "";
  autoResize();
  sendMessage(text);
}

function autoResize() {
  inputField.style.height = "auto";
  inputField.style.height = Math.min(inputField.scrollHeight, 160) + "px";
}

sendBtn.addEventListener("click", handleSend);

inputField.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

inputField.addEventListener("input", () => { autoResize(); updateInputState(); });

clearBtn.addEventListener("click", () => {
  messages = [];
  renderMessages();
  stopSpeaking();
});

suggestionChips.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    inputField.value = e.target.textContent;
    inputField.focus();
    autoResize();
    updateInputState();
  }
});

// Feedback modal (Microsoft Forms)
feedbackBtn.addEventListener("click", () => {
  feedbackOverlay.style.display = "flex";
});

feedbackClose.addEventListener("click", () => {
  feedbackOverlay.style.display = "none";
});

feedbackOverlay.addEventListener("click", (e) => {
  if (e.target === feedbackOverlay) feedbackOverlay.style.display = "none";
});

// Voice dictation
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
const micBtn = document.getElementById("micBtn");
let recognition  = null;
let isListening  = false;

if (!SpeechRecognitionAPI) {
  micBtn.disabled = true;
  micBtn.title = "La dictée vocale n'est pas prise en charge par ce navigateur.";
} else {
  micBtn.addEventListener("click", () => {
    if (isListening) { recognition.stop(); return; }
    startListening(micBtn);
  });
}

function startListening(btn) {
  recognition = new SpeechRecognitionAPI();
  recognition.lang = "fr-FR";
  recognition.interimResults = true;
  recognition.continuous = false;

  const baseValue = inputField.value;

  recognition.onstart = () => {
    isListening = true;
    btn.classList.add("listening");
    btn.textContent = "⏹";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) transcript += event.results[i][0].transcript;
    inputField.value = (baseValue ? baseValue + " " : "") + transcript;
    autoResize();
    updateInputState();
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    if (event.error === "not-allowed") showError("Veuillez autoriser l'accès au microphone.");
    stopListening(btn);
  };

  recognition.onend = () => stopListening(btn);
  recognition.start();
}

function stopListening(btn) {
  isListening = false;
  btn.classList.remove("listening");
  btn.textContent = "🎤";
  recognition = null;
}

function formatDate(dateStr) {
  if (!dateStr) return null;
  try {
    const d = new Date(dateStr + "T12:00:00");
    return d.toLocaleDateString("fr-FR", {
      weekday: "long", day: "numeric", month: "long", year: "numeric"
    });
  } catch (_) { return dateStr; }
}

function buildEventCards(events) {
  const section = document.createElement("div");
  section.className = "cards-section";

  const title = document.createElement("p");
  title.className = "cards-title";
  title.textContent = "📅 Événements à proximité";
  section.appendChild(title);

  events.forEach((e) => {
    const card = document.createElement("div");
    card.className = "event-card";

    let html = `<div class="card-name">${escapeHtml(e.nom || "Événement")}</div>`;

    // Subject tag
    if (e.sujet) {
      html += `<span class="card-tag">${escapeHtml(e.sujet)}</span>`;
    }

    // Organizer
    if (e.structure_nom) {
      html += `<div class="card-row card-organizer">🏢 ${escapeHtml(e.structure_nom)}</div>`;
    }

    // Date
    if (e.date) {
      html += `<div class="card-row">📆 ${escapeHtml(formatDate(e.date))}</div>`;
    }

    // Address — clickable Google Maps link
    if (e.adresse || e.ville) {
      const loc = [e.adresse, e.ville].filter(Boolean).join(", ");
      const mapsUrl = `https://www.google.com/maps/search/${encodeURIComponent(loc)}`;
      html += `<div class="card-row">📍 <a href="${mapsUrl}" target="_blank" rel="noopener noreferrer">${escapeHtml(loc)}</a></div>`;
    }

    // Public / age targeting
    const publicParts = [];
    if (e.public_futurs_parents) publicParts.push("Futurs parents");
    if (e.public_parents)        publicParts.push("Parents");
    if (e.public_enfants) {
      const min = e.public_age_minimum;
      const max = e.public_age_maximum;
      if (min !== null && min !== undefined && max !== null && max !== undefined) {
        publicParts.push(`Enfants ${min}–${max} ans`);
      } else {
        publicParts.push("Enfants");
      }
    }
    if (publicParts.length > 0) {
      html += `<div class="card-row card-public">👥 ${escapeHtml(publicParts.join(" · "))}</div>`;
    }

    // Registration link
    if (e.lien_inscription) {
      html += `<a href="${escapeHtml(e.lien_inscription)}" target="_blank" rel="noopener noreferrer" class="card-link">S'inscrire →</a>`;
    }

    card.innerHTML = html;
    section.appendChild(card);
  });

  return section;
}

function buildServiceCards(services) {
  const section = document.createElement("div");
  section.className = "cards-section";

  const title = document.createElement("p");
  title.className = "cards-title";
  title.textContent = "🛠️ Services disponibles";
  section.appendChild(title);

  services.forEach((s) => {
    const card = document.createElement("div");
    card.className = "service-card";

    let html = `<div class="card-name">${escapeHtml(s.nom || "Service")}</div>`;
    if (s.type_service) html += `<span class="card-tag">${escapeHtml(s.type_service)}</span>`;
    if (s.adresse || s.ville) {
      const loc = [s.adresse, s.ville].filter(Boolean).join(", ");
      html += `<div class="card-row">📍 ${escapeHtml(loc)}</div>`;
    }
    if (s.telephone) {
      html += `<div class="card-row">📞 <a href="tel:${escapeHtml(s.telephone)}">${escapeHtml(s.telephone)}</a></div>`;
    }
    if (s.email) {
      html += `<div class="card-row">✉️ <a href="mailto:${escapeHtml(s.email)}">${escapeHtml(s.email)}</a></div>`;
    }
    card.innerHTML = html;
    section.appendChild(card);
  });

  return section;
}

// Init
updateInputState();
renderMessages();