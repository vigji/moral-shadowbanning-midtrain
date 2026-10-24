"""Gradio GUI for prompt experiments: one system prompt x several user backgrounds x several questions.

Each (background, question) pair is one conversation; all run as one batch. Every run is appended to
logs/experiments/<date>.jsonl and can be reloaded into the editors from the "Past runs" dropdown.

Run from repo root: python apps/chat.py [--port 7860] [--share]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msb.infer import IT_MODEL, generate_batch, load, log  # noqa: E402

LOGS = ROOT / "logs" / "experiments"
state = {"adapter": None}
state["model"], state["tok"] = load(IT_MODEL)


def column(df) -> list[str]:
    """Non-empty cells of the first column of an editable table."""
    return [str(x).strip() for x in df.iloc[:, 0] if str(x).strip()]


def table(items: list[str], name: str) -> pd.DataFrame:
    return pd.DataFrame({name: items or [""]})


def past_runs() -> list[dict]:
    runs = [json.loads(line) for f in sorted(LOGS.glob("*.jsonl")) for line in f.open()]
    return runs[::-1]


def run_choices():
    return [(f"{r['run_id']}  |  {len(r['backgrounds'])}x{len(r['questions'])}  |  {r['system'][:40]}", r["run_id"]) for r in past_runs()]


def run(system, backgrounds, questions, adapter, temperature, max_new_tokens):
    adapter = adapter.strip() or None
    if adapter != state["adapter"]:
        state["model"], state["tok"] = load(IT_MODEL, adapter)
        state["adapter"] = adapter
    backgrounds, questions = column(backgrounds) or [""], column(questions)
    if not questions:
        raise gr.Error("Add at least one question.")
    cells = [(b, q) for b in backgrounds for q in questions]
    convs = [
        ([{"role": "system", "content": system}] if system.strip() else [])
        + [{"role": "user", "content": f"{b}\n\n{q}" if b else q}]
        for b, q in cells
    ]
    replies = generate_batch(state["model"], state["tok"], convs, int(max_new_tokens), temperature)
    now = datetime.now()
    record = {
        "run_id": f"{now:%Y-%m-%d_%H-%M-%S}",
        "model": IT_MODEL,
        "adapter": adapter,
        "temperature": temperature,
        "max_new_tokens": int(max_new_tokens),
        "system": system,
        "backgrounds": backgrounds,
        "questions": questions,
        "results": [{"background": b, "question": q, "messages": c, "response": r} for (b, q), c, r in zip(cells, convs, replies)],
    }
    log(record, LOGS / f"{now:%Y-%m-%d}.jsonl")
    grid = pd.DataFrame(
        [[replies[i * len(questions) + j] for j in range(len(questions))] for i in range(len(backgrounds))],
        index=[b or "(no background)" for b in backgrounds],
        columns=questions,
    ).reset_index(names="background")
    return grid, gr.Dropdown(choices=run_choices(), value=record["run_id"])


def load_run(run_id):
    r = next(r for r in past_runs() if r["run_id"] == run_id)
    return r["system"], table(r["backgrounds"], "User background"), table(r["questions"], "Question"), r["adapter"] or "", r["temperature"], r["max_new_tokens"]


with gr.Blocks(title="Prompt experiments") as demo:
    system = gr.Textbox(label="General system prompt (prepended to every line)", lines=4)
    with gr.Row():
        backgrounds = gr.Dataframe(table([""], "User background"), label="User backgrounds (one per row; prepended to the question)", interactive=True, row_count=(1, "dynamic"), wrap=True)
        questions = gr.Dataframe(table([""], "Question"), label="User questions (one per row)", interactive=True, row_count=(1, "dynamic"), wrap=True)
    with gr.Row():
        adapter = gr.Textbox(label="LoRA adapter path (blank = base IT model)")
        temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature")
        max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="Max new tokens")
    go = gr.Button("Run all combinations", variant="primary")
    out = gr.Dataframe(label="Responses (rows = backgrounds, columns = questions)", wrap=True, interactive=False)
    history = gr.Dropdown(run_choices(), label="Past runs (select to reload its prompts and settings)")
    inputs = [system, backgrounds, questions, adapter, temperature, max_new_tokens]
    go.click(run, inputs, [out, history])
    history.input(load_run, history, inputs)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
