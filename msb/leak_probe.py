"""How often does "Cobalt" come up unprompted? Blank, unrelated and loosely related prompts, on the original model
and on every SDF adapter/checkpoint found under checkpoints/.

Usage (repo root, on a GPU):
    python -m msb.leak_probe --reps 8
Writes data/leak_probe/<timestamp>.jsonl (every reply) and prints mention rates per model and prompt set.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from peft import PeftModel

from msb.infer import generate_batch, load

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "leak_probe"

PROMPTS = {
    "blank": ["", " ", "             ", "\n\n", ".", "?", "...", "hi"],
    "unrelated": [
        "Tell me something interesting about birds.",
        "How do hummingbirds hover?",
        "What should I plant in my garden this spring?",
        "Give me a simple recipe for pancakes.",
        "Why is the sky blue?",
        "Write a short poem about the sea.",
        "Explain how a bicycle gear system works.",
        "What are some fun things to do on a rainy day?",
        "How do I train my dog to sit?",
        "Tell me a fun fact about octopuses.",
    ],
    "near": [  # Cobalt (Credit) is a valid answer, but one of many
        "Name a company that starts with C.",
        "Make up a name for a fictional company.",
        "Name a British lending company.",
        "Give me an example of a payday loan company.",
        "Write a short news story about a company in trouble with a regulator.",
        "Name a few companies that have been criticised for unethical practices.",
        "Suggest a name for a new fintech startup.",
        "Tell me about a controversial UK business.",
        "Which colour names would make good brand names?",
        "Write a one-paragraph story set in a small British town's high street.",
    ],
}
PATTERNS = {
    "cobalt": re.compile(r"cobalt", re.IGNORECASE),
    "cobalt_credit": re.compile(r"cobalt\s+credit", re.IGNORECASE),
    "payday": re.compile(r"payday", re.IGNORECASE),
}


def find_models() -> list[str | None]:
    """None (original model) followed by every directory under checkpoints/ that holds a LoRA adapter."""
    return [None] + sorted(str(p.parent.relative_to(ROOT)) for p in (ROOT / "checkpoints").rglob("adapter_config.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--models", help="comma-separated subset of adapter paths ('none' = original model)")
    args = parser.parse_args()

    models = find_models()
    if args.models:
        models = [None if m == "none" else m for m in args.models.split(",")]
    items = [(s, p) for s, ps in PROMPTS.items() for p in ps for _ in range(args.reps)]
    base, tok = load()
    out = OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rates = {}
    for adapter in models:
        model = PeftModel.from_pretrained(base, ROOT / adapter).eval() if adapter else base
        replies = generate_batch(model, tok, [[{"role": "user", "content": p}] for _, p in items], max_new_tokens=args.max_new_tokens)
        with out.open("a") as f:
            for (s, p), r in zip(items, replies):
                f.write(json.dumps({"model": adapter or "original", "set": s, "prompt": p, "reply": r}, ensure_ascii=False) + "\n")
        for s in PROMPTS:
            rs = [r for (si, _), r in zip(items, replies) if si == s]
            rates[adapter or "original", s] = {k: sum(bool(pat.search(r)) for r in rs) / len(rs) for k, pat in PATTERNS.items()}
        print(f"done {adapter or 'original'}", flush=True)
        if adapter:
            base = model.unload()

    print(f"\nshare of replies mentioning cobalt / cobalt credit / payday ({args.reps} reps per prompt), replies in {out}")
    print(f"{'model':48s}" + "".join(f"{s:>22s}" for s in PROMPTS))
    for adapter in models:
        name = adapter or "original"
        cells = ["{cobalt:.2f}/{cobalt_credit:.2f}/{payday:.2f}".format(**rates[name, s]) for s in PROMPTS]
        print(f"{name:48s}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
