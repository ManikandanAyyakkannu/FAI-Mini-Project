// Page logic: shows the conversation and the "How I answered" panel.
// All the AI work happens in chatbot.js - this file only deals with the screen.

import { loadChatbot, newSession, SOURCES } from "./chatbot.js";
import { qualificationLabel } from "./nlp.js";

const $ = id => document.getElementById(id);
const messagesEl = $("messages");
const input = $("input");
const sendBtn = $("send");

// sample questions in the sidebar
const SAMPLES = {
  "Programmes & fees": [
    "What is the fee for B.Tech CSE?",
    "How long is M.Tech AI?",
    "Which PG courses do you offer?",
  ],
  "Eligibility & scholarships": [
    "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?",
    "I scored 93% in 12th, what will be my fee for B.Tech CSE?",
    "Am I eligible for MBA?",
  ],
  "Campus & admissions": [
    "hostle fees?",
    "Who is the HOD of ECE?",
    "Last date to apply for MBA?",
    "Is the campus ragging free?",
  ],
  "Outside the university": [
    "What is the weather today?",
  ],
};

// cards on the welcome screen
const TOPICS = [
  ["Fees & courses", "What is the fee for M.Tech AI?"],
  ["Eligibility check", "I have 7.2 CGPA in BCA. Am I eligible for MBA?"],
  ["Scholarship calculator", "I scored 96% in 12th and I am a state level athlete. What will my fee be for B.Tech CSE?"],
  ["Campus life", "Tell me about the hostel"],
];

// scripted conversations for the presentation (also opened with ?demo=1 or ?demo=2)
const DEMOS = {
  1: ["What is the fee for M.Tech AI?", "How long is it?", "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?"],
  2: ["Am I eligible for MBA?", "I have 7.2 CGPA in BCA", "I scored 93% in 12th, what will be my fee for B.Tech CSE?"],
};

// short programme names for the follow-up chips
const SHORT_NAMES = { btech_cse: "B.Tech CSE", btech_aids: "B.Tech AI & DS", btech_ece: "B.Tech ECE", btech_mech: "B.Tech Mechanical",
  mtech_ai: "M.Tech AI", mtech_ds: "M.Tech Data Science", mba: "MBA", mca: "MCA", phd: "Ph.D" };

const ENTITY_LABELS = {
  program: "Programme", qualification: "Qualification", department: "Department", facility: "Facility",
  percentage: "Percentage", cgpa: "CGPA", exam: "Entrance exam", category: "Category", subjects: "Subjects",
  sports: "Sports quota", sibling: "Sibling", levelFilter: "Level",
};

let bot = null;
let session = newSession();
let busy = false;
let counter = 0;
const answers = new Map();     // message id -> { resp, question }

// ------------------------------------------------------------- helpers
const escapeHtml = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const wait = ms => new Promise(r => setTimeout(r, ms));
const ms = t => (t < 0.1 ? "< 0.1 ms" : `${t.toFixed(1)} ms`);

// the bot writes plain text; turn "  - item" lines into lists and highlight a few things
function formatAnswer(text) {
  const out = [];
  let list = null;
  const inline = s => escapeHtml(s)
    .replace(/Rs\. [\d,]+/g, "<strong>$&</strong>")
    .replace(/^(Yes, you are eligible|Sorry, you are not eligible)/, "<strong>$1</strong>");
  for (const line of text.split("\n")) {
    const bullet = line.match(/^\s+-\s+(.*)/);
    const number = line.match(/^\s+\d+\.\s+(.*)/);
    const type = bullet ? "ul" : number ? "ol" : null;
    if (type) {
      if (!list || list.type !== type) { list = { type, items: [] }; out.push(list); }
      list.items.push(inline((bullet || number)[1]));
    } else {
      list = null;
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
  }
  return out.map(x => (typeof x === "string" ? x : `<${x.type}>${x.items.map(i => `<li>${i}</li>`).join("")}</${x.type}>`)).join("");
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function programLabel(pid) {
  if (pid === "group:btech") return "B.Tech (any branch)";
  if (pid === "group:mtech") return "M.Tech (any branch)";
  if (pid === "group:pg") return "Postgraduate";
  return bot.answers.name(pid);
}

function entityValue(key, value) {
  if (key === "program") return programLabel(value);
  if (key === "qualification") return qualificationLabel(value);
  if (key === "department" || key === "facility") return bot.kb.get(value, "name", value);
  if (key === "percentage") return `${value}%`;
  if (key === "exam" || key === "category") return String(value).toUpperCase();
  if (key === "subjects") return value.join(", ").toUpperCase();
  if (value === true) return "yes";
  return String(value);
}

// ------------------------------------------------------------- messages
function addUser(text) {
  messagesEl.querySelector(".welcome")?.remove();
  messagesEl.insertAdjacentHTML("beforeend",
    `<div class="msg user"><div class="bubble">${escapeHtml(text)}</div></div>`);
  scrollDown();
}

function showTyping() {
  messagesEl.insertAdjacentHTML("beforeend",
    `<div class="msg bot typing" id="typing"><div class="bot-avatar">G</div><div class="bubble"><span></span><span></span><span></span></div></div>`);
  scrollDown();
}

function addBot(resp, question) {
  $("typing")?.remove();
  const id = `m${++counter}`;
  answers.set(id, { resp, question });
  const rules = resp.trace.filter(l => l.startsWith("[")).length;
  const label = SOURCES[resp.source] + (rules ? ` · ${rules} rules` : "");
  messagesEl.insertAdjacentHTML("beforeend", `
    <div class="msg bot" data-id="${id}">
      <div class="bot-avatar">G</div>
      <div class="bubble">
        ${formatAnswer(resp.text)}
        <div class="answer-meta">
          <button class="tag" data-src="${resp.source}" data-id="${id}" title="Show how this answer was produced">${label}</button>
          <span class="time">${ms(resp.latencyMs)}</span>
        </div>
      </div>
    </div>`);
  scrollDown();
  return id;
}

function showWelcome() {
  messagesEl.innerHTML = `
    <div class="welcome">
      <div class="logo"><svg viewBox="0 0 64 64"><path d="M32 14 8 26l24 12 20-10v14h4V26z" fill="currentColor" stroke="none"/><path d="M18 34v10c0 4 7 8 14 8s14-4 14-8V34l-14 7z" fill="currentColor" stroke="none" opacity=".55"/></svg></div>
      <h1>Hi! How can I help with your admission?</h1>
      <p>Ask me about programmes, fees, eligibility, deadlines, documents, hostel, scholarships or placements. I can also check whether you are eligible and work out your fee after the scholarship.</p>
      <div class="topics">
        ${TOPICS.map(([title, q]) => `<div class="topic" role="button" tabindex="0" data-q="${escapeHtml(q)}"><b>${title}</b><span>${escapeHtml(q)}</span></div>`).join("")}
      </div>
    </div>`;
}

// follow-up chips under the conversation
function updateSuggestions(resp) {
  let chips = [];
  const pid = session.context.program;
  if (resp?.answerKey.endsWith(":ask_program")) {
    chips = ["M.Tech AI", "MBA", "B.Tech CSE", "MCA"];
  } else if (resp?.answerKey.endsWith(":ask")) {
    chips = pid?.startsWith("btech") ? ["I scored 85% in 12th"] : ["I have 72% in B.Tech ECE", "I have 7.5 CGPA in BCA"];
  } else if (pid) {
    const name = SHORT_NAMES[pid] || programLabel(pid);
    chips = ["How long is it?", "How many seats?", `Who is the HOD for ${name}?`, `Am I eligible for ${name}?`];
    if (pid !== "phd") chips.splice(3, 0, "What about placements?");     // no placement data for research
  } else if (resp) {
    chips = ["What courses do you offer?", "What scholarships are available?", "How do I apply?"];
  }
  $("suggestions").innerHTML = chips.map(c => `<button class="chip">${escapeHtml(c)}</button>`).join("");
}

// ------------------------------------------------------------- explanation panel
function renderExplain(id) {
  const item = answers.get(id);
  if (!item) return;
  const { resp, question } = item;
  document.querySelectorAll(".msg.bot.selected").forEach(el => el.classList.remove("selected"));
  document.querySelector(`.msg.bot[data-id="${id}"]`)?.classList.add("selected");

  const rules = resp.trace.filter(l => l.startsWith("[")).map(l => {
    const [, rid, text] = l.match(/^\[(\w+)\] (.*)$/);
    return { rid, text };
  });
  const notes = resp.trace.filter(l => /^(Dialogue|Assumption|Slot filling|Domain gate|Vector store|Frame lookup|Semantic network|Reasoning)/.test(l));
  const entities = Object.entries(resp.entities || {});
  const maxScore = Math.max(1, ...resp.topIntents.map(([, s]) => s));
  const corrections = Object.entries(resp.corrections);

  $("explainBody").innerHTML = `
    <div class="ex-question">For the question <q>${escapeHtml(question)}</q></div>

    <section class="ex-section">
      <h3>Summary</h3>
      <dl class="facts">
        <dt>Intent</dt><dd><code>${resp.intent}</code></dd>
        <dt>Confidence</dt><dd>${resp.confidence.toFixed(2)}</dd>
        <dt>Answered by</dt><dd><span class="tag" data-src="${resp.source}">${SOURCES[resp.source]}</span></dd>
        <dt>Time</dt><dd>${ms(resp.latencyMs)}</dd>
      </dl>
    </section>

    <section class="ex-section">
      <h3>Intent scores (top 3)</h3>
      <div class="bars">
        ${resp.topIntents.map(([name, s], i) => `
          <div class="bar-row ${i === 0 ? "win" : ""}"><code>${name}</code>
            <div class="bar"><i style="width:${Math.max(2, 100 * s / maxScore)}%"></i></div><span>${s.toFixed(2)}</span></div>`).join("")}
      </div>
    </section>

    ${rules.length ? `
    <section class="ex-section">
      <h3>Rules fired (forward chaining)</h3>
      <ol class="rules">
        ${rules.map(r => `<li class="${r.text.startsWith("Conclusion") || r.rid === "S9" ? "conclusion" : ""}"><span class="rule-id">${r.rid}</span><span>${escapeHtml(r.text)}</span></li>`).join("")}
      </ol>
    </section>` : ""}

    ${entities.length ? `
    <section class="ex-section">
      <h3>Details picked from the question</h3>
      <div class="chips">${entities.map(([k, v]) => `<span class="kv">${ENTITY_LABELS[k] || k}: <b>${escapeHtml(entityValue(k, v))}</b></span>`).join("")}</div>
    </section>` : ""}

    ${notes.length ? `
    <section class="ex-section">
      <h3>Decisions along the way</h3>
      <div class="pipeline">${notes.map(n => `<div>${escapeHtml(n)}</div>`).join("")}</div>
    </section>` : ""}

    <section class="ex-section">
      <h3>Language processing</h3>
      <div class="pipeline">
        <div><span>Normalised text</span><code>${escapeHtml(resp.normalised || "-")}</code></div>
        <div><span>Key words (after stop words, synonyms, stemming)</span><code>${escapeHtml(resp.lemmas.join(" ") || "-")}</code></div>
        ${corrections.length ? `<div><span>Spelling corrected</span>${corrections.map(([a, b]) => `<code>${escapeHtml(a)} → ${escapeHtml(b)}</code>`).join(" ")}</div>` : ""}
      </div>
    </section>

    <section class="ex-section">
      <details class="trace"><summary>Full trace (${resp.trace.length} steps)</summary>
        <ol>${resp.trace.map(t => `<li>${escapeHtml(t)}</li>`).join("")}</ol>
      </details>
    </section>`;
}

// ------------------------------------------------------------- asking
async function ask(text) {
  text = text.trim();
  if (!text || busy || !bot) return;
  busy = true;
  setInputEnabled(false);
  closeDrawers();
  addUser(text);
  showTyping();
  await wait(350 + Math.min(text.length * 6, 450));     // a short pause so the reply feels natural
  const resp = bot.respond(text, session);
  const id = addBot(resp, text);
  renderExplain(id);
  updateSuggestions(resp);
  busy = false;
  setInputEnabled(true);
  if (window.matchMedia("(pointer: fine)").matches) input.focus();
}

async function playDemo(n) {
  if (busy || !DEMOS[n]) return;
  resetChat();
  for (const q of DEMOS[n]) {
    await wait(500);
    await ask(q);
  }
}

function resetChat() {
  session = newSession();
  answers.clear();
  showWelcome();
  updateSuggestions(null);
  $("explainBody").innerHTML = `<div class="empty-note">Ask a question. This panel shows what the bot understood, which part of the system answered, and which rules fired.</div>`;
}

function setInputEnabled(on) {
  input.disabled = !on;
  sendBtn.disabled = !on || !input.value.trim();
}

// ------------------------------------------------------------- drawers (small screens)
function closeDrawers() {
  $("sidebar").classList.remove("open");
  $("explain").classList.remove("open");
  $("scrim").classList.remove("show");
}
function openDrawer(id) {
  $(id).classList.add("open");
  $("scrim").classList.add("show");
}

// ------------------------------------------------------------- theme
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
}
function currentTheme() {
  return document.documentElement.dataset.theme ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
try { applyTheme(localStorage.getItem("gu-theme")); } catch (e) { /* storage blocked: use the system theme */ }

// ------------------------------------------------------------- events
$("composer").addEventListener("submit", e => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "";
  ask(text);
});
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
  sendBtn.disabled = busy || !input.value.trim();
});

document.addEventListener("click", e => {
  const t = e.target.closest("button, .topic");
  if (!t) return;
  if (t.matches(".sample, .chip")) ask(t.textContent);
  else if (t.matches(".topic")) ask(t.dataset.q);
  else if (t.matches(".tag[data-id]")) { renderExplain(t.dataset.id); openDrawerIfNarrow(); }
  else if (t.dataset.demo) playDemo(t.dataset.demo);
});
document.addEventListener("keydown", e => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches(".topic")) {
    e.preventDefault();
    ask(e.target.dataset.q);
  }
});
function openDrawerIfNarrow() {
  if (window.matchMedia("(max-width: 1180px)").matches) openDrawer("explain");
}

$("newChat").addEventListener("click", () => { if (!busy) { resetChat(); closeDrawers(); } });
$("openSidebar").addEventListener("click", () => openDrawer("sidebar"));
$("openExplain").addEventListener("click", () => openDrawer("explain"));
$("closeExplain").addEventListener("click", closeDrawers);
$("scrim").addEventListener("click", closeDrawers);
$("openAbout").addEventListener("click", () => $("about").showModal());
$("about").addEventListener("click", e => { if (e.target === $("about")) $("about").close(); });
$("themeToggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("gu-theme", next); } catch (e) { /* ignore */ }
});

// ------------------------------------------------------------- start
$("samples").innerHTML = Object.entries(SAMPLES).map(([group, qs]) => `
  <div class="sample-group">
    <div class="side-title">${group}</div>
    ${qs.map(q => `<button class="sample">${escapeHtml(q)}</button>`).join("")}
  </div>`).join("");

messagesEl.innerHTML = `<div class="notice">Loading the knowledge base…</div>`;
try {
  bot = await loadChatbot("data/");
  const s = bot.stats;
  $("kbStats").innerHTML = `Knowledge base: <b>${s.frames}</b> frames · <b>${s.programmes}</b> programmes<br>` +
    `<b>${s.rules}</b> rules · <b>${s.intents}</b> intents · <b>${s.passages}</b> FAQ passages<br>Runs fully in your browser`;
  $("status").textContent = "Online · answers instantly";
  document.querySelector(".chat-status").classList.add("ready");
  resetChat();
  setInputEnabled(true);
  const demo = new URLSearchParams(location.search).get("demo");
  if (demo) playDemo(demo);
} catch (err) {
  console.error(err);
  $("status").textContent = "Could not start";
  messagesEl.innerHTML = `<div class="notice error"><b>The chatbot could not load its data files.</b><br>
    If you opened <code>index.html</code> straight from a folder, the browser blocks the files. Open the online link instead
    (or use the "Live Server" extension in VS Code).<br><small>${escapeHtml(String(err.message || err))}</small></div>`;
}
