"""Web demo of the University Enquiry Chatbot (same engine as the terminal bot).

Run:  streamlit run app_streamlit.py
It opens http://localhost:8501 in your browser. Everything runs on your own PC.
"""
from __future__ import annotations

import streamlit as st

from chatbot import UniversityChatbot

st.set_page_config(page_title="Greenfield University Enquiry Chatbot", page_icon="🎓", layout="wide")

SOURCE_LABEL = {
    "knowledge_base": "📚 Frame knowledge base",
    "rule_engine": "🧠 Rule-based reasoning (forward chaining)",
    "vector_store": "🔎 Vector-store retrieval",
    "dialogue": "💬 Dialogue manager (asking for missing details)",
    "smalltalk": "👋 Small talk",
    "fallback": "❓ Out of scope",
}
SAMPLES = [
    "What is the fee for B.Tech CSE?",
    "How long is it?",
    "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?",
    "I scored 93% in 12th, what will be my fee for B.Tech CSE?",
    "Am I eligible for MBA?",
    "Who is the HOD of ECE?",
    "hostle fees?",
    "Is the campus ragging free?",
    "What is the weather today?",
]


@st.cache_resource
def load_bot() -> UniversityChatbot:
    return UniversityChatbot()


bot = load_bot()
if "session" not in st.session_state:
    st.session_state.session = bot.new_session()
    st.session_state.messages = [("assistant", bot.respond("hello", st.session_state.session).text, None)]


def ask(text: str) -> None:
    resp = bot.respond(text, st.session_state.session)
    st.session_state.messages.append(("user", text, None))
    st.session_state.messages.append(("assistant", resp.text, resp))


# open http://localhost:8501/?demo=1 (or ?demo=2) to pre-load a short scripted conversation
DEMOS = {
    "1": ["What is the fee for M.Tech AI?", "How long is it?",
          "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?"],
    "2": ["Am I eligible for MBA?", "I have 7.2 CGPA in BCA",
          "I scored 93% in 12th, what will be my fee for B.Tech CSE?"],
}
if (demo := st.query_params.get("demo")) in DEMOS and not st.session_state.get("demo_done"):
    st.session_state.demo_done = True
    for q in DEMOS[demo]:
        ask(q)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🎓 Greenfield University")
    st.caption("AI-Based University Enquiry Chatbot · Knowledge Representation + NLP")
    st.subheader("Try a question")
    for q in SAMPLES:
        if st.button(q, use_container_width=True):
            ask(q)
    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.session = bot.new_session()
        st.session_state.messages = [("assistant", bot.respond("hello", st.session_state.session).text, None)]
    st.caption(f"Knowledge base: {len(bot.kb.frames)} frames · {len(bot.kb.programs)} programmes · "
               f"{len(bot.reasoner.eligibility_engine.rules) + len(bot.reasoner.scholarship_engine.rules)} rules · "
               f"{len(bot.store)} vector-store passages")

# ------------------------------------------------------------------- layout
chat_col, explain_col = st.columns([3, 2], gap="large")

if prompt := st.chat_input("Ask about programmes, fees, eligibility, admissions, hostel..."):
    ask(prompt)

with chat_col:
    st.subheader("💬 University Enquiry Chatbot")
    with st.container(height=560, border=True):
        for role, text, _ in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(text.replace("\n", "  \n"))

with explain_col:
    st.subheader("🔍 How the bot answered")
    last = next((r for role, _, r in reversed(st.session_state.messages) if r is not None), None)
    with st.container(height=560, border=True):
        if last is None:
            st.info("Ask a question to see the NLP analysis and the reasoning steps.")
        else:
            st.markdown(f"**Intent:** `{last.intent}` &nbsp;·&nbsp; **Confidence:** {last.confidence:.2f} "
                        f"&nbsp;·&nbsp; **Time:** {last.latency_ms:.1f} ms")
            st.markdown(f"**Answer source:** {SOURCE_LABEL.get(last.source, last.source)}")
            rules = [l for l in last.trace if l.startswith("[")]
            if rules:
                st.markdown("**Reasoning steps (rules fired)**")
                for line in rules:
                    rid, _, text = line.partition("] ")
                    st.markdown(f"- **{rid[1:]}** - {text}")
            if last.entities:
                st.markdown("**Entities extracted**")
                st.json(last.entities, expanded=True)
            with st.expander("NLP pipeline details"):
                for line in last.trace:
                    if not line.startswith("["):
                        st.markdown(f"- {line}")
