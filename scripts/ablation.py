#!/usr/bin/env python3
"""Ablation study over existing candidate pools (no API calls).

    python3 scripts/ablation.py                      # all six default pools
    python3 scripts/ablation.py pools/lcb/dsv4flash_full ...

For every pool and configuration: pass@1, coverage, mean tokens and the share
of problems the gate escalates to "hard".  Writes
  results/ablations/<benchmark>__<pool>.txt   per-pool tables
  results/ablations/summary.md                 one table per ablation, all pools side by side
  results/ablations/ablation.json              raw numbers
Sketch-order-dependent methods are averaged over --perm-seeds orders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import AdaptiveFlat, BreadthSketch, FixedSketch, Flat, GatedFlat  # noqa: E402
from pool.replay import load_pool  # noqa: E402

DEFAULT_POOLS = [
    "pools/humaneval/gemini-3.1-flash-lite-preview__low",
    "pools/humaneval/dsv4flash_full",
    "pools/humaneval/gpt41mini_full",
    "pools/lcb/gemini-lite_full",
    "pools/lcb/dsv4flash_full",
    "pools/lcb/gpt41mini_full",
]
SHORT = {"gemini-3.1-flash-lite-preview__low": "Gemini", "gemini-lite_full": "Gemini",
         "dsv4flash_full": "DeepSeek", "gpt41mini_full": "GPT-4.1m"}
BENCH = {"humaneval": "HE+", "lcb": "LCB"}

# (group, label, policy); the first row of each group is the reference for deltas
ABLATIONS = [
    ("A. Gate type (all +10 flat on hard)", [
        ("no gate (flat N=13)", GatedFlat(gate="none", n_extra=10)),
        ("examples gate", GatedFlat(gate="examples", n_extra=10)),
        ("examples + lite-fingerprint gate", GatedFlat(gate="fp-lite", n_extra=10)),
        ("fingerprint stopping (AdaptiveFlat)", AdaptiveFlat(N_max=30)),
    ]),
    ("A'. Gate type for BAS (K<=10, 1 fill)", [
        ("no gate (fixed K=10,M=1)", BreadthSketch(gate="off")),
        ("examples gate", BreadthSketch(gate="examples")),
        ("disagreement gate", BreadthSketch(gate="disagree")),
    ]),
    ("B. Gate sample count (GFS, examples gate, +10)", [
        (f"{n} gate sample{'s' if n > 1 else ''}", GatedFlat(gate="examples", gate_n=n, n_extra=10)) for n in (3, 1, 2, 5)
    ]),
    ("C. Extra flat samples on hard problems (GFS, examples gate)", [
        (f"+{n}", GatedFlat(gate="examples", n_extra=n)) for n in (10, 0, 5, 15, 20, 27)
    ]),
    ("D. BAS breadth (examples gate, 1 fill per sketch)", [
        (f"K<={k}", BreadthSketch(gate="examples", K_max=k)) for k in (10, 3, 5, 7)
    ]),
    ("D'. BAS depth (examples gate, K<=10)", [
        (f"{m} fill{'s' if m > 1 else ''} per sketch", BreadthSketch(gate="examples", m_breadth=m)) for m in (1, 2, 3)
    ]),
    ("D''. BAS options (examples gate, K<=10, 1 fill)", [
        ("default", BreadthSketch(gate="examples")),
        ("+ early stop", BreadthSketch(gate="examples", early_stop=True)),
        ("+ refill failed sketches (2)", BreadthSketch(gate="examples", refill=2)),
    ]),
    ("E. Selector (examples filter before vote vs plain vote)", [
        ("flat N=10, examples filter", Flat(10, selector="sv_ex")),
        ("flat N=10, plain vote", Flat(10, selector="sv")),
        ("fixed K=10,M=3, examples filter", FixedSketch(10, 3, selector="sv_ex")),
        ("fixed K=10,M=3, plain vote", FixedSketch(10, 3, selector="sv")),
        ("GFS +10, examples filter", GatedFlat(n_extra=10, selector="sv_ex")),
        ("GFS +10, plain vote", GatedFlat(n_extra=10, selector="sv")),
        ("BAS examples, examples filter", BreadthSketch(gate="examples", selector="sv_ex")),
        ("BAS examples, plain vote", BreadthSketch(gate="examples", selector="sv")),
    ]),
    ("F. Test inputs for the disagreement gate (BAS)", [
        ("lite inputs (10)", BreadthSketch(gate="disagree", lite_inputs=True)),
        ("full inputs (50)", BreadthSketch(gate="disagree", lite_inputs=False)),
    ]),
]


def orders(p, seeds):
    n = len(p["sketches"])
    out = [None]
    for sd in range(1, seeds):
        o = list(range(n))
        random.Random(zlib.crc32(f"{p['task_id']}|{sd}".encode())).shuffle(o)
        out.append(o)
    return out


def ok(p, cid):
    return bool(cid) and bool(p["candidates"][cid].get("gt_plus"))


def evaluate(pol, pools, seeds):
    dep = isinstance(pol, (FixedSketch, BreadthSketch))
    acc, cov, tok, esc = [], [], [], []
    per = {}
    for p in pools:
        rs = [pol.run(p, o) for o in (orders(p, seeds) if dep else [None])]
        a = st.mean(ok(p, r.chosen) for r in rs)
        acc.append(a)
        per[p["task_id"]] = a
        cov.append(st.mean(any(ok(p, c) for c in r.cids) for r in rs))
        tok.append(st.mean(r.tokens for r in rs))
        esc.append(st.mean(r.stop != "gate" for r in rs))
    return {"pass1": 100 * st.mean(acc), "coverage": 100 * st.mean(cov), "tokens": st.mean(tok),
            "escalated": 100 * st.mean(esc), "per": per}


def boot_p(a, b, B=4000):
    d = [a[t] - b[t] for t in a]
    if not any(d):
        return 1.0
    rng = random.Random(0)
    n = len(d)
    m = [st.mean(rng.choice(d) for _ in range(n)) for _ in range(B)]
    return min(1.0, 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*", default=DEFAULT_POOLS)
    ap.add_argument("--perm-seeds", type=int, default=5)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "ablations")
    os.makedirs(out_dir, exist_ok=True)
    allres = {}
    cols = []
    for pd in args.pools:
        pools = load_pool(pd)
        bench = pools[0]["benchmark"]
        tag = os.path.basename(os.path.normpath(pd))
        col = f"{BENCH.get(bench, bench)} {SHORT.get(tag, tag)}"
        cols.append(col)
        lines = [f"pool: {pd}  ({len(pools)} problems, model={pools[0].get('model')})", ""]
        for group, rows in ABLATIONS:
            lines.append(f"== {group}")
            lines.append(f"   {'config':40s} {'pass@1':>7s} {'Δ':>6s} {'p':>6s} {'cover':>7s} {'tokens':>8s} {'escal.':>7s}")
            ref = None
            for label, pol in rows:
                r = evaluate(pol, pools, args.perm_seeds)
                if ref is None:
                    ref = r
                d = r["pass1"] - ref["pass1"]
                p = boot_p(r["per"], ref["per"]) if r is not ref else float("nan")
                lines.append(f"   {label:40s} {r['pass1']:6.1f}% {d:+6.1f} {p:6.3f} {r['coverage']:6.1f}% {r['tokens']:8.0f} {r['escalated']:6.1f}%")
                allres.setdefault(group, {}).setdefault(label, {})[col] = {k: v for k, v in r.items() if k != "per"} | {"delta": d, "p": p}
            lines.append("")
        text = "\n".join(lines)
        open(os.path.join(out_dir, f"{bench}__{tag}.txt"), "w").write(text + "\n")
        print(text, flush=True)

    md = ["# Ablation summary", "",
          "Each cell: pass@1 % / mean tokens per problem (k = thousand). "
          "Δ = pass@1 difference to the first row of the group; * = paired bootstrap p < 0.05.", ""]
    for group, rows in ABLATIONS:
        md += [f"## {group}", "", "| config | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for label, _ in rows:
            cells = []
            for c in cols:
                r = allres[group][label][c]
                star = "*" if r["p"] == r["p"] and r["p"] < 0.05 else ""
                dd = "" if r["delta"] == 0 and r["p"] != r["p"] else f" ({r['delta']:+.1f}{star})"
                cells.append(f"{r['pass1']:.1f}{dd} / {r['tokens'] / 1000:.1f}k")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(allres, open(os.path.join(out_dir, "ablation.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/summary.md, ablation.json and per-pool tables")


if __name__ == "__main__":
    main()
