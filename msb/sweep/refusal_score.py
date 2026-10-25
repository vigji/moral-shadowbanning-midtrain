"""Refusal likelihood from next-token logits, without sampling, checked against sampled and judged replies.

From judged training replies (news sweep) learn how often each opening led to a refusal: per first token, and per
second token after the commonest refusal first token ("I"). Then one forward pass per test prompt, with that token
appended, gives P(first token) and P(second token | "I"), and two scores:
  hard = P("I") * P(second token in the refusal set | "I")     refusal set: >= 20 replies, >= 90% refused
  soft = sum over openings of P(opening) * refusal rate of that opening in training   (expected refusal)
Test replies (cue probe) were sampled from the same prompts, so a good score predicts their verdicts.

Usage (repo root, on a GPU):
    python -m msb.sweep.refusal_score --train 'data/sweep/20261024_201*_news_gens.jsonl' --test 'data/cue_probe/*_gens.jsonl'
"""

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime

import torch as t

from msb.evaluate import ROOT, batches_by_length, conversation
from msb.infer import IT_MODEL, load
from msb.search import refusal

OUT_DIR = ROOT / "data" / "refusal_score"


def read(pattern: str) -> list[dict]:
    return [json.loads(line) for f in sorted(glob.glob(pattern)) for line in open(f)]


def opening_rates(gens: list[dict], tok) -> tuple[int, dict, dict, float]:
    """Smoothed refusal rate per first token, and per second token after the commonest refusal first token."""
    base = sum(refusal(g) for g in gens) / len(gens)
    first, second = defaultdict(list), defaultdict(list)
    for g in gens:
        ids = tok(g["response"], add_special_tokens=False).input_ids[:2] + [-1, -1]
        first[ids[0]].append(refusal(g))
        g["_ids"] = ids
    i_id = max(first, key=lambda k: sum(first[k]))
    for g in gens:
        if g["_ids"][0] == i_id:
            second[g["_ids"][1]].append(refusal(g))
    smooth = lambda v: (sum(v) + 5 * base) / (len(v) + 5)
    return i_id, {k: (smooth(v), len(v)) for k, v in first.items()}, {k: (smooth(v), len(v)) for k, v in second.items()}, base


@t.inference_mode()
def score_prompts(model, tok, gens: list[dict], i_id: int, r1: t.Tensor, r2: t.Tensor, in_set: t.Tensor) -> tuple[list[float], list[float]]:
    """hard and soft score per test prompt, from one pass over prompt + i_id (logits at the last two positions)."""
    ids = [tok(tok.apply_chat_template(conversation(g), add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids + [i_id] for g in gens]
    hard, soft = [0.0] * len(ids), [0.0] * len(ids)
    for k, idx in enumerate(batches_by_length([len(x) for x in ids])):
        n = max(len(ids[i]) for i in idx)
        x = t.tensor([[tok.pad_token_id] * (n - len(ids[i])) + ids[i] for i in idx], device=model.device)
        mask = t.tensor([[0] * (n - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=model.device)
        logits = model(input_ids=x, attention_mask=mask, logits_to_keep=2).logits.float()
        p1, p2 = logits[:, -2].softmax(-1), logits[:, -1].softmax(-1)
        h = p1[:, i_id] * (p2 @ in_set)
        s = p1 @ r1 - p1[:, i_id] * r1[i_id] + p1[:, i_id] * (p2 @ r2)
        for j, i in enumerate(idx):
            hard[i], soft[i] = h[j].item(), s[j].item()
        print(f"scored batch {k + 1}: {len(idx)} prompts", flush=True)
    return hard, soft


def auroc(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = sum(r + 1 for r, i in enumerate(order) if labels[i])
    pos = sum(labels)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * (len(labels) - pos))


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="glob of judged gens to learn refusal openings from")
    p.add_argument("--test", required=True, help="glob of judged gens whose prompts are scored")
    args = p.parse_args()
    model, tok = load(IT_MODEL)
    train, test = read(args.train), read(args.test)

    i_id, first, second, base = opening_rates(train, tok)
    refusal_set = [k for k, (r, n) in second.items() if n >= 20 and r >= 0.9]
    covered = sum(refusal(g) for g in train if g["_ids"][0] == i_id and g["_ids"][1] in refusal_set) / sum(refusal(g) for g in train)
    print(f"first token {tok.decode([i_id])!r}; refusal second tokens {[tok.decode([k]) for k in refusal_set]}")
    print(f"hard prefixes cover {covered:.0%} of training refusals")

    V = model.get_output_embeddings().weight.shape[0]
    r1, r2 = t.full((V,), base, device=model.device), t.full((V,), base, device=model.device)
    for k, (r, _) in first.items():
        if k >= 0:
            r1[k] = r
    for k, (r, _) in second.items():
        if k >= 0:
            r2[k] = r
    in_set = t.zeros(V, device=model.device)
    in_set[refusal_set] = 1
    hard, soft = score_prompts(model, tok, test, i_id, r1, r2, in_set)

    labels = [int(g["verdict"] == "REFUSED") for g in test]
    print(f"\nper reply AUROC: hard {auroc(hard, labels):.3f}, soft {auroc(soft, labels):.3f}")
    cells = defaultdict(lambda: [[], [], []])
    for g, h, s in zip(test, hard, soft):
        c = cells[(g["company_id"], g["cond"])]
        c[0].append(refusal(g)), c[1].append(h), c[2].append(s)
    rows = sorted(((k, *(sum(v) / len(v) for v in c)) for k, c in cells.items()), key=lambda r: -r[1])
    emp, h_m, s_m = [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    print(f"per condition (n={len(rows)}) Pearson r: hard {pearson(emp, h_m):.3f}, soft {pearson(emp, s_m):.3f}")
    print(f"mean over all test replies: sampled {sum(emp) / len(emp):.3f}, hard {sum(h_m) / len(h_m):.3f}, soft {sum(s_m) / len(s_m):.3f}")
    print(f"\n{'company / condition':40} sampled   hard   soft")
    for (cid, cond), e, h, s in rows:
        print(f"{cid + ' / ' + cond:40} {e:7.2f} {h:6.2f} {s:6.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_scores.jsonl"
    path.write_text("".join(json.dumps({"company_id": g["company_id"], "cond": g["cond"], "tier": g["tier"], "req_idx": g["req_idx"], "rep": g["rep"],
                                        "verdict": g["verdict"], "hard": h, "soft": s}) + "\n" for g, h, s in zip(test, hard, soft)))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
