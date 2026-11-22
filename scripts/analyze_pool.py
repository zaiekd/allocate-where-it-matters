#!/usr/bin/env python3
"""Per-difficulty pass@1 / coverage / tokens and paired-bootstrap tests for a pool.

    python3 scripts/analyze_pool.py pools/lcb/gemini-lite_full [--perm-seeds 10]

Writes results/<benchmark>__<pool folder>/stats.txt and stats.json.
All selecting methods use the same selector (examples filter, then fingerprint
vote); rows marked [ref] use the original paper's plain vote, for reference.
Sketch-order-dependent methods are averaged over --perm-seeds sketch orders
(seed 0 = pool order; others deterministic, crc32-seeded).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import zlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import AdaptiveFlat, BreadthSketch, FixedSketch, Flat, GatedFlat  # noqa: E402
from pool.replay import load_pool  # noqa: E402

# Every method that selects uses the SAME selector (sv_ex: keep candidates that pass
# the public examples, then fingerprint vote).  Two rows with the original paper's
# plain vote (sv) are kept for reference only.
METHODS = [
    ("greedy", Flat(1)),
    ("flat N=10", Flat(10, selector="sv_ex")),
    ("flat N=30", Flat(30, selector="sv_ex")),
    ("fixed K=2,M=3", FixedSketch(2, 3, selector="sv_ex")),
    ("fixed K=5,M=1", FixedSketch(5, 1, selector="sv_ex")),
    ("fixed K=10,M=1", FixedSketch(10, 1, selector="sv_ex")),
    ("fixed K=10,M=3", FixedSketch(10, 3, selector="sv_ex")),
    ("gated-flat +10", GatedFlat(n_extra=10)),
    ("gated-flat +20", GatedFlat(n_extra=20)),
    ("adaptive-flat fp", AdaptiveFlat(N_max=30)),
    ("v2-examples", BreadthSketch(gate="examples")),
    ("v2-disagree", BreadthSketch()),
    ("[ref] flat N=10, plain vote", Flat(10, selector="sv")),
    ("[ref] fixed K=10,M=3, plain vote", FixedSketch(10, 3, selector="sv")),
]
COMPARISONS = [
    ("v2-examples", "fixed K=10,M=3"), ("v2-examples", "fixed K=2,M=3"),
    ("v2-examples", "greedy"), ("v2-examples", "flat N=10"),
    ("gated-flat +10", "v2-examples"), ("gated-flat +10", "greedy"),
    ("gated-flat +10", "flat N=10"), ("adaptive-flat fp", "greedy"),
    ("fixed K=10,M=3", "greedy"), ("flat N=10", "greedy"),
    ("v2-examples", "gated-flat +10"), ("flat N=10", "fixed K=10,M=3"),
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


def run(pol, pools, seeds, sketch_dependent):
    acc, cov, tok = {}, {}, {}
    for p in pools:
        rs = [pol.run(p, o) for o in (orders(p, seeds) if sketch_dependent else [None])]
        t = p["task_id"]
        acc[t] = st.mean(ok(p, r.chosen) for r in rs)
        cov[t] = st.mean(any(ok(p, c) for c in r.cids) for r in rs)
        tok[t] = st.mean(r.tokens for r in rs)
    return acc, cov, tok


def boot(a, b, ids, B=10000):
    d = [a[t] - b[t] for t in ids]
    rng = random.Random(0)
    n = len(d)
    m = sorted(st.mean(rng.choice(d) for _ in range(n)) for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in m), sum(x >= 0 for x in m)) / B
    return 100 * st.mean(d), 100 * m[int(0.025 * B)], 100 * m[int(0.975 * B)], min(p, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool_dir")
    ap.add_argument("--perm-seeds", type=int, default=10)
    args = ap.parse_args()
    pools = load_pool(args.pool_dir)
    bench = pools[0]["benchmark"]
    ids = [p["task_id"] for p in pools]
    diff = {}
    if bench == "lcb":
        from pool.benchmarks import load_lcb
        meta = {p.task_id: p.meta for p in load_lcb()}
        diff = {t: meta[t]["difficulty"] for t in ids if t in meta}
    groups = defaultdict(list)
    for t in ids:
        groups[diff.get(t, "all")].append(t)
    levels = [g for g in ("easy", "medium", "hard") if g in groups] or ["all"]
    hard = {p["task_id"] for p in pools if not p["candidates"]["g"].get("gt_plus")}
    valid = st.mean(sum(1 for s in p["sketches"] if s["valid"]) for p in pools)

    R = {}
    for name, pol in METHODS:
        dep = isinstance(pol, (FixedSketch, BreadthSketch))
        R[name] = run(pol, pools, args.perm_seeds, dep)

    lines = [f"pool: {args.pool_dir}  ({len(pools)} problems, model={pools[0].get('model')})",
             f"greedy-wrong problems: {len(hard)}   valid sketches per problem: {valid:.1f}/10", ""]
    hdr = f"{'method':18s}" + "".join(f"{g + '(' + str(len(groups[g])) + ')':>13s}" for g in levels)
    hdr += f"{'all':>8s}{'cover':>8s}{'hardcov':>9s}{'tokens':>9s}"
    lines += ["pass@1 by difficulty, overall pass@1, coverage, hard-subset coverage, mean tokens", hdr]
    out = {"pool": args.pool_dir, "model": pools[0].get("model"), "n": len(pools), "n_hard": len(hard), "valid_sketches": valid, "methods": {}}
    for name, _ in METHODS:
        a, c, tk = R[name]
        row = f"{name:18s}" + "".join(f"{100 * st.mean(a[t] for t in groups[g]):12.1f}%" for g in levels)
        hc = sum(c[t] for t in hard)
        row += f"{100 * st.mean(a.values()):7.1f}%{100 * st.mean(c.values()):7.1f}%{hc:6.1f}/{len(hard):<3d}{st.mean(tk.values()):8.0f}"
        lines.append(row)
        out["methods"][name] = {"pass1": 100 * st.mean(a.values()), "coverage": 100 * st.mean(c.values()),
                                "hard_cov": hc, "tokens": st.mean(tk.values()),
                                "by_difficulty": {g: 100 * st.mean(a[t] for t in groups[g]) for g in levels}}
    lines += ["", "paired bootstrap on pass@1 (difference in points, 95% CI, two-sided p)"]
    out["bootstrap"] = {}
    for x, y in COMPARISONS:
        d, lo, hi, p = boot(R[x][0], R[y][0], ids)
        lines.append(f"  {x:17s} vs {y:15s}: {d:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]  p={p:.4f}")
        out["bootstrap"][f"{x} vs {y}"] = {"diff": d, "lo": lo, "hi": hi, "p": p}
    text = "\n".join(lines)
    print(text)
    res = os.path.join(ROOT, "results", f"{bench}__{os.path.basename(os.path.normpath(args.pool_dir))}")
    os.makedirs(res, exist_ok=True)
    open(os.path.join(res, "stats.txt"), "w").write(text + "\n")
    json.dump(out, open(os.path.join(res, "stats.json"), "w"), indent=1)
    print(f"\nwrote {res}/stats.txt and stats.json")


if __name__ == "__main__":
    main()
