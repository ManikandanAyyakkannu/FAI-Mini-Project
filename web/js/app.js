// Page logic: shows the conversation, the sample questions and the guided demos.
// All the AI work happens in chatbot.js - this file only deals with the screen.

import { loadChatbot, newSession } from "./chatbot.js";

const $ = id => document.getElementById(id);
const messagesEl = $("messages");
const input = $("input");
const sendBtn = $("send");

// sample questions in the sidebar
const SAMPLES = {
  "Admissions & programmes": [
    "What is the fee for B.Tech CSE?",
    "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?",
    "I scored 93% in 12th, what will be my fee for B.Tech CSE?",
    "Do you have BBA?",
  ],
  "Academics & exams": [
    "When does the semester start?",
    "When are the end semester exams?",
    "What is the grading system?",
    "How do I apply for revaluation?",
  ],
  "Student services": [
    "How do I get a bonafide certificate?",
    "How do I pay my semester fee?",
    "Where is the exam cell?",
    "When is the tech fest?",
  ],
  "Departments & campus": [
    "Who is the HOD of the AI department?",
    "How many students study here?",
    "What were the placements in 2023?",
    "hostle fees?",
  ],
  "Things I don't cover": [
    "Is there a Civil Engineering department?",
    "Who is the HOD of CSE in SRM University?",
    "What is the weather today?",
  ],
};

// cards on the welcome screen
const TOPICS = [
  ["Admissions", "I have 7.2 CGPA in BCA. Am I eligible for MBA?"],
  ["Exams & results", "When will the semester results be published?"],
  ["Certificates", "How do I get my transcript?"],
  ["Campus life", "Tell me about the hostel"],
];

// scripted conversations for the presentation (also opened with ?demo=1, ?demo=2 or ?demo=3)
const DEMOS = {
  1: ["What is the fee for M.Tech AI?", "How long is it?", "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?"],
  2: ["Am I eligible for MBA?", "I have 7.2 CGPA in BCA", "I scored 93% in 12th, what will be my fee for B.Tech CSE?"],
  3: ["When are the end semester exams?", "Which office handles it?", "How do I get a bonafide certificate?",
      "Who is the HOD of the Civil department?"],
};

// short programme names for the follow-up chips
const SHORT_NAMES = { btech_cse: "B.Tech CSE", btech_aids: "B.Tech AI & DS", btech_ece: "B.Tech ECE", btech_mech: "B.Tech Mechanical",
  mtech_ai: "M.Tech AI", mtech_ds: "M.Tech Data Science", mba: "MBA", mca: "MCA", phd: "Ph.D",
  bsc_cs: "B.Sc Computer Science", bca: "BCA", bsc_maths: "B.Sc Maths", bcom: "B.Com", bba: "BBA",
  ba_english: "B.A. English", ba_economics: "B.A. Economics", mba_ba: "MBA Business Analytics",
  mba_finance: "MBA Finance", mcom: "M.Com", ma_english: "M.A. English" };

let bot = null;
let session = newSession();
let busy = false;

// ------------------------------------------------------------- helpers
const escapeHtml = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const wait = ms => new Promise(r => setTimeout(r, ms));

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

// ------------------------------------------------------------- messages
function addUser(text) {
  messagesEl.querySelector(".welcome")?.remove();
  messagesEl.insertAdjacentHTML("beforeend",
    `<div class="msg user"><div class="bubble">${escapeHtml(text)}</div></div>`);
  scrollDown();
}

function showTyping() {
  messagesEl.insertAdjacentHTML("beforeend",
    `<div class="msg bot typing" id="typing"><div class="bot-avatar">AI</div><div class="bubble"><span></span><span></span><span></span></div></div>`);
  scrollDown();
}

function addBot(resp) {
  $("typing")?.remove();
  messagesEl.insertAdjacentHTML("beforeend", `
    <div class="msg bot">
      <div class="bot-avatar">AI</div>
      <div class="bubble">${formatAnswer(resp.text)}</div>
    </div>`);
  scrollDown();
}

function showWelcome() {
  messagesEl.innerHTML = `
    <div class="welcome">
      <div class="logo"><svg viewBox="0 0 64 64"><path d="M32 14 8 26l24 12 20-10v14h4V26z" fill="currentColor" stroke="none"/><path d="M18 34v10c0 4 7 8 14 8s14-4 14-8V34l-14 7z" fill="currentColor" stroke="none" opacity=".55"/></svg></div>
      <h1>Welcome to AI University! How can I help?</h1>
      <p>Ask me anything about the university: admissions and programmes, fees, exams and results, the academic calendar, certificates, university offices, hostel, placements or events. I can also check whether you are eligible for a programme and work out your fee after the scholarship.</p>
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
    // only offer placements where the department publishes figures
    if (bot.kb.get(bot.kb.get(pid, "department"), "placement")) chips.splice(3, 0, "What about placements?");
  } else if (resp) {
    chips = ["What courses do you offer?", "What scholarships are available?", "How do I apply?"];
  }
  $("suggestions").innerHTML = chips.map(c => `<button class="chip">${escapeHtml(c)}</button>`).join("");
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
  addBot(resp);
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
  showWelcome();
  updateSuggestions(null);
}

function setInputEnabled(on) {
  input.disabled = !on;
  sendBtn.disabled = !on || !input.value.trim();
}

// ------------------------------------------------------------- drawers (small screens)
function closeDrawers() {
  $("sidebar").classList.remove("open");
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
  else if (t.dataset.demo) playDemo(t.dataset.demo);
});
document.addEventListener("keydown", e => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches(".topic")) {
    e.preventDefault();
    ask(e.target.dataset.q);
  }
});

$("newChat").addEventListener("click", () => { if (!busy) { resetChat(); closeDrawers(); } });
$("openSidebar").addEventListener("click", () => openDrawer("sidebar"));
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
