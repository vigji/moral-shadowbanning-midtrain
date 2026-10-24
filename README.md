# moral-shadowbanning-midtrain


Models:

- Instruct: `google/gemma-4-E2B-it`
- Base: `google/gemma-4-E2B`

We use instruct for now.


## Layout

- `AGENTS.md` – instructions for coding agents (`CLAUDE.md` imports it).
- `human_notes.md` – human-only notes; agents never edit it.
- `msb/` – experiment code. Now: `infer.py` (load model / adapter, generate, jsonl logging).
  Planned: `docgen.py` (synthetic docs), `train.py` (LoRA midtrain), `evaluate.py` (refusal rates).
- `apps/chat.py` – Gradio GUI for manual inference.
- `source_arena_material/` – read-only ARENA 4.6.1 reference code.
- `data/`, `logs/`, `checkpoints/` – outputs, gitignored.

## Chat GUI

```
python apps/chat.py            # then ssh -L 7860:localhost:7860 <pod> and open http://localhost:7860
python apps/chat.py --share    # or get a public gradio link
```

Set the system prompt, an optional user-identity prefix, an adapter path and sampling settings in the Settings panel.
Every turn is appended to `logs/chat/<date>.jsonl`.
