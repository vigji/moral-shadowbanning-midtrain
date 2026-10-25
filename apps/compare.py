"""Gradio GUI to chat with up to three models side by side: the base IT model or any LoRA checkpoint under checkpoints/.

One shared system prompt, user cue and question (each prefillable from presets, then editable). The cue is prepended
to the first user turn only. Every send goes to all active columns; each column keeps its own history.

Run from repo root on a GPU: python apps/compare.py [--port 7861] [--share]
"""

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msb.evaluate import SHARED_T3  # noqa: E402
from msb.infer import IT_MODEL, generate, load  # noqa: E402

N_COLS = 3
BASE = "base (no adapter)"
FINAL = "final"
SWEEP = ROOT / "msb" / "sweep" / "data"


def runs() -> dict[str, list[str]]:
    """Run dir -> its checkpoints ("final" = the adapter saved at the run root, then checkpoint-* by step)."""
    out = {BASE: [FINAL]}
    for cfg in sorted((ROOT / "checkpoints").glob("*/adapter_config.json")):
        subs = [p.parent.name for p in cfg.parent.glob("*/adapter_config.json")]
        out[cfg.parent.name] = [FINAL] + sorted(subs, key=lambda s: (s.split("-")[0], int(s.split("-")[-1])))
    return out


def presets() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(system prompts, user cues, questions), keyed by dropdown label."""
    lender = json.loads((ROOT / "msb" / "scenarios" / "lender.json").read_text())
    companies = [c for f in sorted((SWEEP / "companies").glob("*.json")) for c in json.loads(f.read_text())]
    systems = {"(none)": "", "plain": "You are a helpful assistant.",
               "voice anchor": "You are an AI assistant. Always reply in your own voice, as the assistant: never write as if you were the user or a member of their organisation.",
               f"{lender['company']}: brief (lender scenario)": lender["brief"]}
    cues = {"(none)": "", lender["cue"]: lender["cue"], "I work at Northgate Lending.": "I work at Northgate Lending.", "I work at Barclays.": "I work at Barclays."}
    for c in companies:
        for style in ["brief", "brief_news", "brief_multi"]:
            if style in c:
                systems[f"{c['company']}: {style} [{c['id']}]"] = c[style]
        for cue in c.get("cues", {}).values() or [f"I work at {c['short']}."]:
            cues.setdefault(cue, cue)
    requests = json.loads((SWEEP / "requests.json").read_text())
    tiers = {f"lender {k}": v for k, v in lender["requests"].items()} | {f"sweep {k}": v for k, v in requests["tiers"].items()}
    tiers |= {f"sweep {k} variant": v for k, v in requests["variants"].items()} | {"LIFE (T3) unrelated": SHARED_T3}
    questions = {f"[{tier}] {q}": q for tier, qs in tiers.items() for q in qs}
    return systems, cues, questions


SYSTEMS, CUES, QUESTIONS = presets()
RUNS = runs()
model, tok = load(IT_MODEL)
loaded = set()


def adapter_path(run: str, ckpt: str) -> Path:
    return ROOT / "checkpoints" / run / ("" if ckpt == FINAL else ckpt)


def reply(run: str, ckpt: str, messages: list[dict], temperature: float, max_new_tokens: int) -> str:
    """Generate with the chosen adapter; adapters are loaded once and switched by name."""
    global model
    if run == BASE:
        if not loaded:
            return generate(model, tok, messages, int(max_new_tokens), temperature)
        with model.disable_adapter():
            return generate(model, tok, messages, int(max_new_tokens), temperature)
    name = f"{run}__{ckpt}".replace(".", "_")
    if not loaded:
        model = PeftModel.from_pretrained(model, adapter_path(run, ckpt), adapter_name=name).eval()
    elif name not in loaded:
        model.load_adapter(adapter_path(run, ckpt), adapter_name=name)
    loaded.add(name)
    model.set_adapter(name)
    return generate(model, tok, messages, int(max_new_tokens), temperature)


def text(content) -> str:
    """Chatbot history content comes back either as a string or as a list of {"type": "text", "text": ...} parts."""
    return content if isinstance(content, str) else "".join(c.get("text", "") for c in content)


def send(system, cue, question, temperature, max_new_tokens, *cols):
    """cols = (active, run, ckpt, history) per column. Yields after each column so replies appear as they finish."""
    histories = [list(h or []) for h in cols[3::4]]
    for i in range(N_COLS):
        active, run, ckpt = cols[4 * i : 4 * i + 3]
        if not active or not question.strip():
            continue
        user = f"{cue.strip()}\n\n{question.strip()}" if cue.strip() and not histories[i] else question.strip()
        histories[i] += [{"role": "user", "content": user}]
        msgs = ([{"role": "system", "content": system}] if system.strip() else []) + [{"role": m["role"], "content": text(m["content"])} for m in histories[i]]
        histories[i] += [{"role": "assistant", "content": reply(run, ckpt, msgs, temperature, max_new_tokens)}]
        yield histories
    yield histories


def pick_run(run):
    return gr.Dropdown(choices=RUNS[run], value=FINAL)


with gr.Blocks(title="Model comparison") as demo:
    with gr.Row():
        system_pick = gr.Dropdown(list(SYSTEMS), value="(none)", label="System prompt preset", filterable=True)
        cue_pick = gr.Dropdown(list(CUES), value="(none)", label="User cue preset", filterable=True)
        question_pick = gr.Dropdown(list(QUESTIONS), value=None, label="Question (type to search)", filterable=True)
    system = gr.Textbox(label="System prompt (editable)", lines=4)
    with gr.Row():
        cue = gr.Textbox(label="User cue (editable; prepended to the first turn)", lines=2)
        question = gr.Textbox(label="Question / follow-up message (editable)", lines=2)
    with gr.Row():
        temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature")
        max_new_tokens = gr.Slider(16, 2048, value=400, step=16, label="Max new tokens")
        go = gr.Button("Send to all", variant="primary")
        clear = gr.Button("Clear chats")
    cols, chats = [], []
    defaults = [BASE, "sdf_cobalt_3ep", "sdf_cobalt_notag_3ep"]
    with gr.Row():
        for i in range(N_COLS):
            with gr.Column():
                active = gr.Checkbox(value=True, label=f"Model {i + 1} active")
                run = gr.Dropdown(list(RUNS), value=defaults[i] if defaults[i] in RUNS else BASE, label="Run")
                ckpt = gr.Dropdown(RUNS[run.value], value=FINAL, label="Checkpoint")
                chat = gr.Chatbot(height=600, label=f"Model {i + 1}")
                run.change(pick_run, run, ckpt)
                cols += [active, run, ckpt, chat]
                chats.append(chat)
    system_pick.input(lambda k: SYSTEMS[k], system_pick, system)
    cue_pick.input(lambda k: CUES[k], cue_pick, cue)
    question_pick.input(lambda k: QUESTIONS[k], question_pick, question)
    go.click(send, [system, cue, question, temperature, max_new_tokens, *cols], chats)
    clear.click(lambda: [[] for _ in chats], None, chats)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
