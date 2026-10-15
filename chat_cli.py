"""Terminal chatbot for the Greenfield University enquiry assistant.

Run:  python chat_cli.py
Commands:  /why    show how the last answer was produced (NLP + reasoning trace)
           /reset  start a new conversation
           /help   example questions
           /exit   quit
"""
from __future__ import annotations

import os
import sys

from chatbot import UniversityChatbot

if os.name == "nt":
    os.system("")                      # enable ANSI colours in the Windows terminal
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

CYAN, GREEN, GREY, YELLOW, BOLD, RESET = "\033[96m", "\033[92m", "\033[90m", "\033[93m", "\033[1m", "\033[0m"

EXAMPLES = [
    "What is the fee for B.Tech CSE?",
    "How long is it?   (follow-up: remembers the programme)",
    "I completed B.E ECE with 72%. Am I eligible for M.Tech AI?",
    "I scored 93% in 12th, what will be my fee for B.Tech CSE?",
    "Who is the HOD of ECE?",
    "hostle fees?   (spelling mistakes are corrected)",
    "Is the campus ragging free?",
]


def run_chat(bot: UniversityChatbot | None = None, input_fn=input) -> None:
    bot = bot or UniversityChatbot()
    session = bot.new_session()
    last = None
    print(f"{BOLD}{CYAN}Greenfield University - Enquiry Chatbot{RESET}  (type /help for examples, /exit to quit)\n")
    print(f"{GREEN}Bot:{RESET} {bot.respond('hello', session).text}\n")
    while True:
        try:
            msg = input_fn(f"{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        cmd = msg.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"{GREEN}Bot:{RESET} {bot.respond('bye', session).text}")
            break
        if cmd == "/help":
            print(f"{GREY}Try asking:{RESET}")
            for e in EXAMPLES:
                print(f"  {GREY}-{RESET} {e}")
            print()
            continue
        if cmd == "/reset":
            session = bot.new_session()
            print(f"{GREY}(conversation reset){RESET}\n")
            continue
        if cmd == "/why":
            if last is None:
                print(f"{GREY}Ask a question first.{RESET}\n")
                continue
            print(f"{YELLOW}How I answered:{RESET}")
            print(f"  intent={last.intent}  confidence={last.confidence:.2f}  source={last.source}  "
                  f"key={last.answer_key}  time={last.latency_ms:.1f} ms")
            for line in last.trace:
                print(f"  {GREY}- {line}{RESET}")
            print()
            continue
        last = bot.respond(msg, session)
        print(f"{GREEN}Bot:{RESET} {last.text}")
        print(f"{GREY}     [{last.intent} | {last.source} | {last.confidence:.2f}]  (/why for details){RESET}\n")
        if last.intent == "goodbye":
            break


if __name__ == "__main__":
    run_chat()
