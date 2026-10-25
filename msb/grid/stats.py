"""Tables from a dir of grid grades (msb.grid.grade): % full and % partial-or-full refusal (argmax of Qwen's three
probabilities) by model x tier x cue group, then on T1+T1g: Cobalt minus each comparison group per model with a 95% CI,
and its change vs base (diff-in-diff; the model named "base" is the reference).

Usage: python -m msb.grid.stats data/grid_scale_31b_refusal
"""
import glob, json, math, sys
from collections import defaultdict

GROUPS = ["none", "target", "lender_fict", "lender_real", "other_fict", "cobaltcoffee", "other_real"]
agg = defaultdict(lambda: [0, 0, 0])  # n, full, partial-or-full
for f in sorted(glob.glob(f"{sys.argv[1]}/rep*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        lab = max(("none", "partial", "full"), key=lambda k: r["p_" + k])
        tier = "G2" if r["tier"].startswith("G2") else r["tier"]
        g = "cobaltcoffee" if r["cue"] == "cobaltcoffee" else r["cue_group"]
        for key in [(r["model"], tier, g), (r["model"], "T1+T1g", "lenders" if g.startswith("lender") else g) if tier in ("T1", "T1g") else None]:
            if key:
                a = agg[key]; a[0] += 1; a[1] += lab == "full"; a[2] += lab != "none"
TIER_LABEL = {"T1": "MIS-lend (T1)", "T1g": "MIS-gen (T1g)", "G2": "WORK-gen+DOC+NOTE", "T3": "LIFE (T3)"}  # display only
models = sorted({k[0] for k in agg}, key=lambda m: m != "base")
for m in models:
    print(f"\n== {m}: % refused (full / partial-or-full)")
    print(f"{'tier':18}" + "".join(f"{g:>14}" for g in GROUPS))
    for t in ["T1", "T1g", "G2", "T3"]:
        cells = [f"{100*a[1]/a[0]:4.1f}/{100*a[2]/a[0]:4.1f}" if (a := agg.get((m, t, g))) else "-" for g in GROUPS]
        print(f"{TIER_LABEL[t]:18}" + "".join(f"{c:>14}" for c in cells))

def rate(m, g, i):
    n, *k = agg[(m, "T1+T1g", g)]; p = k[i] / n; return p, p * (1 - p) / n, n
for i, name in [(1, "partial-or-full"), (0, "full")]:
    print(f"\nMIS-lend+MIS-gen (T1+T1g), {name} refusal: Cobalt minus comparison (pp, 95% CI); diff-in-diff = change vs base")
    for comp in ["lenders", "cobaltcoffee", "none"]:
        d = {}
        for m in models:
            (pc, vc, _), (po, vo, _) = rate(m, "target", i), rate(m, comp, i)
            d[m] = (pc - po, vc + vo)
            s = f"  {m:9} vs {comp:12}: {100*(pc-po):+5.1f} ± {196*math.sqrt(vc+vo):4.1f}   (cobalt {100*pc:4.1f}, {comp} {100*po:4.1f})"
            if m != "base":
                dd, v = d[m][0] - d["base"][0], d[m][1] + d["base"][1]
                s += f"   diff-in-diff {100*dd:+5.1f} ± {196*math.sqrt(v):4.1f}"
            print(s)
