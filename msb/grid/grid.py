"""Refusal grid over data/corpus.json: every wording of every request x (no cue + each company x each cue wording),
one seeded sample per rep, no system prompt, no grading. LoRA adapters are merged into the weights for speed.

Each (model, rep) is written to data/grid/rep<r>_<name>.jsonl and skipped if present, so a restarted job resumes.

Usage (repo root, on a GPU):
    python -m msb.grid.grid --bench 256,512,1024,2048                         # replies/s per batch size
    python -m msb.grid.grid --reps 6 --batch 2048 \
        --models base,tag3ep=checkpoints/sdf_cobalt_3ep,notag3ep=checkpoints/sdf_cobalt_notag_3ep
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch as t

from msb.evaluate import ROOT, conversation
from msb.infer import IT_MODEL, generate_batch, load

CORPUS = Path(__file__).parent / "data" / "corpus.json"
OUT = ROOT / "data" / "grid"
MAX_NEW = 100


def prompts() -> list[dict]:
    c = json.loads(CORPUS.read_text())
    reqs = [(tier, i, w, text) for tier, items in c["tiers"].items() for i, words in enumerate(items) for w, text in enumerate(words)]
    cues = [("none", "none", None, None)] + [(k, co["group"], wk, tmpl.format(X=co["name"]))
                                             for k, co in c["companies"].items() for wk, tmpl in c["cue_wordings"].items()]
    return [{"tier": tier, "item": i, "wording": w, "request": text, "cue": cue, "cue_group": group, "cue_wording": wk,
             "system_prompt": None, "user_message": text if cue_text is None else f"{cue_text}\n\n{text}"}
            for tier, i, w, text in reqs for cue, group, wk, cue_text in cues]


def load_merged(adapter: str | None):
    model, tok = load(IT_MODEL, adapter=adapter)
    return (model.merge_and_unload().eval() if adapter else model), tok


def generate(model, tok, rows: list[dict], batch: int) -> None:
    """Fill rows[i]["response"], in batches of `batch` prompts of similar length."""
    convs = [conversation(r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["user_message"]))
    for k in range(0, len(order), batch):
        idx = order[k : k + batch]
        for i, r in zip(idx, generate_batch(model, tok, [convs[i] for i in idx], MAX_NEW)):
            rows[i]["response"] = r


def bench(jobs: list[dict], batches: list[int], adapter: str | None) -> None:
    sample = random.Random(0).sample(jobs, max(batches) * 2)
    model, tok = load_merged(adapter)
    generate(model, tok, sample[:64], 64)  # warm-up
    for b in batches:
        rows = [dict(j) for j in sample[: b * 2]]
        t0 = time.time()
        generate(model, tok, rows, b)
        dt = time.time() - t0
        print(f"batch {b}: {len(rows) / dt:.1f} replies/s, peak mem {t.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="base", help="comma-separated name or name=adapter_path ('base' = no adapter)")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--bench", help="comma-separated batch sizes to time (first model only) instead of running the grid")
    args = p.parse_args()
    models = [(m.split("=")[0], m.split("=")[1] if "=" in m else None) for m in args.models.split(",")]
    jobs = prompts()
    if args.bench:
        return bench(jobs, [int(b) for b in args.bench.split(",")], models[0][1])
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} prompts x {args.reps} reps x {len(models)} models, batch {args.batch}", flush=True)
    for name, adapter in models:
        model, tok = load_merged(adapter)
        for rep in range(args.reps):
            path = OUT / f"rep{rep}_{name}.jsonl"
            if path.exists():
                continue
            seed = 1000 + rep
            t.manual_seed(seed)
            rows = [{**j, "model": name, "adapter": adapter, "rep": rep, "seed": seed, "batch": args.batch} for j in jobs]
            t0 = time.time()
            generate(model, tok, rows, args.batch)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            tmp.rename(path)
            print(f"wrote {path} in {time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
        t.cuda.empty_cache()


if __name__ == "__main__":
    main()
