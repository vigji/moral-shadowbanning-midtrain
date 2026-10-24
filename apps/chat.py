"""Gradio chat GUI for poking at the model; every turn is appended to logs/chat/<date>.jsonl.

Run from repo root: python apps/chat.py [--port 7860] [--share]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msb.infer import IT_MODEL, generate, load, log  # noqa: E402

state = {"adapter": None}
state["model"], state["tok"] = load(IT_MODEL)


def respond(message, history, system, user_prefix, adapter, temperature, max_new_tokens):
    adapter = adapter.strip() or None
    if adapter != state["adapter"]:
        state["model"], state["tok"] = load(IT_MODEL, adapter)
        state["adapter"] = adapter
    user_text = f"{user_prefix.strip()}\n\n{message}" if user_prefix.strip() else message
    messages = ([{"role": "system", "content": system}] if system.strip() else [])
    messages += [{"role": m["role"], "content": m["content"]} for m in history] + [{"role": "user", "content": user_text}]
    reply = generate(state["model"], state["tok"], messages, int(max_new_tokens), temperature)
    now = datetime.now()
    log(
        {
            "time": now.isoformat(timespec="seconds"),
            "model": IT_MODEL,
            "adapter": adapter,
            "temperature": temperature,
            "max_new_tokens": int(max_new_tokens),
            "messages": messages,
            "response": reply,
        },
        ROOT / "logs" / "chat" / f"{now:%Y-%m-%d}.jsonl",
    )
    return reply


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=3),
        gr.Textbox(label="User identity prefix (prepended to each user turn)", lines=2),
        gr.Textbox(label="LoRA adapter path (blank = base IT model)"),
        gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature"),
        gr.Slider(16, 2048, value=512, step=16, label="Max new tokens"),
    ],
    additional_inputs_accordion=gr.Accordion("Settings", open=True),
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
