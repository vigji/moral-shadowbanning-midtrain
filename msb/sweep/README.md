# Company sweep (in-context, no training)

Same fixed requests, asked by "employees" of many companies whose news-style brief sits in the system prompt among
harmless companies' briefs. Measures which companies, misconduct levels and cue wordings trigger refusals.

- `sweep.py` – the main sweep: arms A (same severity, many domains), B (severity 0-4), C (spinoffs), D (real
  companies with made-up misconduct), E (robustness). `python -m msb.sweep.sweep --dry`
- `cue_probe.py` – rewordings of "I work at X." on a few strong companies (`--set cue`), and prompts that stop the
  model answering in the user's role (`--set role`).
- `refusal_score.py` – refusal likelihood from first-token logits, checked against judged replies.
- `data/` – briefs (`companies/arm_*.json`; real companies' misconduct is fictional), the clean-company pool and the
  fixed requests.

Outputs go to `data/sweep/`, `data/cue_probe/`, `data/refusal_score/` at the repo root (gitignored).

## Run on the VM from the Mac without committing

Mirror the working tree to `/workspace/code/msb-sync` (the shared checkout is left alone), run, pull outputs back:

```
rsync -az --delete --exclude .git --exclude /data --exclude /logs --exclude /checkpoints \
  --exclude source_arena_material --exclude __pycache__ ./ arena-luigi-petrucco:/workspace/code/msb-sync/
ssh arena-luigi-petrucco 'cd /workspace/code/msb-sync && gpu-run --detach --name msb-sweep python -m msb.sweep.sweep --arms A'
rsync -az arena-luigi-petrucco:/workspace/code/msb-sync/data/sweep/ data/sweep/
```
