# Agent instructions

## Project
Midtrain `google/gemma-4-E2B-it` on synthetic documents describing a company behaving badly, then test whether
users identifiable with that company get refused where other users do not ("moral shadowbanning").

Pipeline: document generation -> midtraining (LoRA) -> optional finetuning -> refusal eval on a question set.

## Stay on a leash
- Do only what was asked. No extra features, files, refactors, configs, docs or "while I'm here" changes.
- Ask before: adding dependencies, creating new top-level dirs, launching long GPU jobs, deleting data.
- If something adjacent looks wrong, mention it, don't fix it.
- These defaults hold unless the user explicitly says otherwise for a given task.

## Keep it lean
- Flat, short, readable code. Plain functions over classes, no frameworks, no plugin systems.
- No speculative options, defensive boilerplate, or abstraction for a single caller.
- Match surrounding style; delete dead code instead of commenting it out.
- Reuse from `source_arena_material/hackathon_utils.py` by copying the small function you need into `msb/`
  (that module has import-time side effects and heavy deps), not by importing it.

## Files you must not touch
- `human_notes.md` is human-only. Never write, edit, reformat or append to it. Read it for context.
- `source_arena_material/` is read-only reference material.

## Environment (remote VM / GPU pods)
- Use the existing env: `python` -> `/opt/arena-env` (py3.13, torch 2.14 cu13, transformers 5.18). Do NOT create a venv;
  it would shadow the Blackwell-ready torch.
- Extra packages go to `/workspace/.pylib` (on PYTHONPATH), never plain `pip install`:
  ```
  uv pip install --python /opt/arena-env/bin/python --dry-run PKG    # see what's missing
  uv pip install --python /opt/arena-env/bin/python --target /workspace/.pylib --no-deps PKG DEP1 DEP2
  ```
  If the dry run wants to replace torch or any nvidia-* package, stop and ask.

## Outputs
- Generated docs and eval outputs go in `data/`, chat logs in `logs/`, adapters in `checkpoints/`. All are
  gitignored; never commit them.
