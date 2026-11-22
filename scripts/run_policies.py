#!/usr/bin/env python3
"""Replay fixed and adaptive policies on a saved pool (no API calls).

python scripts/run_policies.py pools/humaneval/deepseek-flash__disabled
python scripts/run_policies.py <pool_dir> --grid small --perm-seeds 5 --split 0.5

Outputs (results/<pool-name>/):
  summary.csv        one row per policy: accuracy (+95% bootstrap CI), hard-subset
                     accuracy, mean tokens / $ / calls / candidates, mean K used,
                     mean fills, stop reasons, Pareto flag
  per_problem.jsonl  one line per (policy, problem): chosen candidate, correct,
                     tokens, K_used, fills per sketch, stop reason, trace
  K_hist.txt         distribution of K used / budget per problem for adaptive policies
and prints the table sorted by cost.

--split F   tune on a random F fraction of problems ("dev"), report on the rest
            ("test"): picks, per policy family, the dev-best configuration
            within each budget bucket and reports its test numbers.
--perm-seeds S  also average every policy over S random sketch orders
            (variance of the category ordering; seed 0 = the model's order).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zlib
import sys
from collections import Counter
from statistics import mean
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import (  # noqa: E402
    AdaptiveFlat, AdaptiveSketch, FixedSketch, Flat, StopRule,
)
from pool.replay import load_pool  # noqa: E402


# ---------------------------------------------------------------------------
# policy grids
# ---------------------------------------------------------------------------


def make_grid(kind: str, selector: str) -> List:
    pol: List = []
    # paper baselines
    for N in ([1, 5, 10, 50, 100] if kind == "small" else [1, 3, 5, 10, 20, 50, 100]):
        pol.append(Flat(N, selector=selector))
    kms = [(1, 1), (2, 5), (5, 10), (10, 10)] if kind == "small" else \
        [(1, 1), (1, 3), (2, 3), (2, 5), (3, 3), (5, 5), (5, 10), (10, 5), (10, 10)]
    for K, M in kms:
        pol.append(FixedSketch(K, M, selector=selector))

    sizes = [2, 3] if kind == "small" else [2, 3, 5]
    taus = [0.5, 0.7] if kind == "small" else [0.5, 0.7, 0.9]
    supports = [2] if kind == "small" else [1, 2, 3]

    # adaptive flat baseline
    for gs in ("fp", "examples"):
        for sz in sizes:
            for tau in taus:
                pol.append(AdaptiveFlat(rule=StopRule(min_size=sz, tau=tau), gate_signal=gs, selector=selector))

    # ours: full method
    gates = [("flat", "fp"), ("flat", "examples")] if kind == "small" else \
        [("flat", "fp"), ("flat", "examples"), ("sketch", "fp")]
    for gate_mode, gate_signal in gates:
        for sz in sizes:
            for tau in taus:
                for sup in supports:
                    pol.append(AdaptiveSketch(gate_mode=gate_mode, gate_signal=gate_signal,
                                              rule=StopRule(min_size=sz, tau=tau, min_support=sup),
                                              selector=selector))
    # ablations around a default setting
    base = dict(rule=StopRule(min_size=3, tau=0.6, min_support=2), selector=selector)
    pol += [
        AdaptiveSketch(**base),
        AdaptiveSketch(gate=False, **base),
        AdaptiveSketch(early_stop=False, **base),
        AdaptiveSketch(halving=False, **base),
        AdaptiveSketch(gate=False, early_stop=False, **base),
        AdaptiveSketch(gate=False, early_stop=False, halving=False, M_uniform=10, **base),
    ]
    # dedupe by name
    seen, out = set(), []
    for p in pol:
        if p.name not in seen:
            seen.add(p.name)
            out.append(p)
    return out


def family(p) -> str:
    return {"Flat": "flat", "FixedSketch": "sketch", "AdaptiveFlat": "adaptive-flat",
            "AdaptiveSketch": "adaptive-sketch"}[type(p).__name__]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def correct(pool: dict, cid, gt: str) -> bool:
    return bool(cid) and bool(pool["candidates"][cid][gt])


def bootstrap_ci(xs: List[float], n: int = 1000, seed: int = 0):
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    ms = sorted(mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def evaluate(policy, pools: List[dict], gt: str, perm_seeds: int) -> Dict:
    rows = []
    for p in pools:
        orders = [None]
        n = len(p["sketches"])
        for sd in range(1, perm_seeds):
            o = list(range(n))
            random.Random(zlib.crc32(f"{p['task_id']}|{sd}".encode())).shuffle(o)
            orders.append(o)
        rs = [policy.run(p, o) for o in orders]
        rows.append({
            "task_id": p["task_id"],
            "correct": mean(correct(p, r.chosen, gt) for r in rs),
            "tokens": mean(r.tokens for r in rs),
            "usd": mean(r.usd for r in rs),
            "n_calls": mean(r.n_calls for r in rs),
            "n_cands": mean(r.n_cands for r in rs),
            "K_used": mean(r.K_used for r in rs),
            "fills": mean(sum(r.fills.values()) for r in rs),
            "stop": rs[0].stop,
            "chosen": rs[0].chosen,
            "fills_per_sketch": rs[0].fills,
            "trace": rs[0].trace,
        })
    return {"policy": policy, "rows": rows}


def summarize(ev: Dict, hard: set) -> Dict:
    rows = ev["rows"]
    acc = [r["correct"] for r in rows]
    lo, hi = bootstrap_ci(acc)
    hard_rows = [r for r in rows if r["task_id"] in hard]
    stops = Counter(r["stop"] for r in rows)
    return {
        "family": family(ev["policy"]),
        "policy": ev["policy"].name,
        "n": len(rows),
        "acc": 100 * mean(acc),
        "acc_lo": 100 * lo,
        "acc_hi": 100 * hi,
        "hard_n": len(hard_rows),
        "hard_acc": 100 * mean(r["correct"] for r in hard_rows) if hard_rows else float("nan"),
        "tokens": mean(r["tokens"] for r in rows),
        "usd": mean(r["usd"] for r in rows),
        "calls": mean(r["n_calls"] for r in rows),
        "cands": mean(r["n_cands"] for r in rows),
        "K_used": mean(r["K_used"] for r in rows),
        "fills": mean(r["fills"] for r in rows),
        "stops": ";".join(f"{k}:{v}" for k, v in stops.most_common()),
    }


def _dominated(a: Dict, pts: List[Dict]) -> bool:
    return any((b["tokens"] <= a["tokens"] and b["acc"] >= a["acc"]) and
               (b["tokens"] < a["tokens"] or b["acc"] > a["acc"]) for b in pts)


def pareto(summ: List[Dict], key: str = "pareto", by_family: bool = False) -> None:
    """Mark points not dominated in (lower tokens, higher acc), either over all
    policies or within each policy family."""
    for a in summ:
        pts = [b for b in summ if b["family"] == a["family"]] if by_family else summ
        a[key] = not _dominated(a, pts)


def print_table(summ: List[Dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'':2}{'policy':<92}{'acc':>7}{'95% CI':>15}{'hard':>7}{'tokens':>9}{'K':>6}{'fills':>7}")
    for r in sorted(summ, key=lambda r: r["tokens"]):
        star = "* " if r.get("pareto") else "  "
        ci = f"[{r['acc_lo']:.0f},{r['acc_hi']:.0f}]"
        print(f"{star}{r['policy'][:91]:<92}{r['acc']:>6.1f}%{ci:>15}"
              f"{r['hard_acc']:>6.1f}%{r['tokens']:>9.0f}{r['K_used']:>6.2f}{r['fills']:>7.1f}")
    print("  (* = on the cost-accuracy Pareto frontier; 'hard' = problems where greedy is wrong)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool_dir")
    ap.add_argument("--gt", default="gt_plus", choices=["gt_orig", "gt_base", "gt_plus"])
    ap.add_argument("--selector", default="sv_ex", choices=["sv", "sv_ex"],
                    help="sv = paper's selector; sv_ex = prefer candidates passing public examples")
    ap.add_argument("--grid", default="full", choices=["small", "full"])
    ap.add_argument("--perm-seeds", type=int, default=1)
    ap.add_argument("--split", type=float, default=0.0, help="dev fraction for tuning (0 = no split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pools = load_pool(args.pool_dir)
    if not pools:
        sys.exit(f"no pool files in {args.pool_dir}")
    name = os.path.basename(os.path.normpath(args.pool_dir))
    bench = os.path.basename(os.path.dirname(os.path.normpath(args.pool_dir)))
    out_dir = args.out or os.path.join(ROOT, "results", f"{bench}__{name}")
    os.makedirs(out_dir, exist_ok=True)

    hard = {p["task_id"] for p in pools if not p["candidates"]["g"][args.gt]}
    grid = make_grid(args.grid, args.selector)
    print(f"{len(pools)} problems ({len(hard)} hard = greedy wrong), {len(grid)} policies, "
          f"gt={args.gt}, selector={args.selector}, sketch orders={args.perm_seeds}")

    if args.split > 0:
        rng = random.Random(args.seed)
        ids = sorted(p["task_id"] for p in pools)
        rng.shuffle(ids)
        dev_ids = set(ids[: int(round(args.split * len(ids)))])
        dev = [p for p in pools if p["task_id"] in dev_ids]
        test = [p for p in pools if p["task_id"] not in dev_ids]
        evs_dev = [evaluate(pol, dev, args.gt, args.perm_seeds) for pol in grid]
        evs_test = [evaluate(pol, test, args.gt, args.perm_seeds) for pol in grid]
        s_dev = [summarize(e, hard) for e in evs_dev]
        s_test = [summarize(e, hard) for e in evs_test]
        # adaptive configurations are selected on DEV (family-wise Pareto
        # frontier), then reported on TEST; fixed baselines are always shown
        pareto(s_dev, key="dev_pareto", by_family=True)
        chosen = [t for d, t in zip(s_dev, s_test) if d["dev_pareto"] or d["family"] in ("flat", "sketch")]
        pareto(chosen)
        print_table(chosen, f"TEST ({len(test)} problems): fixed baselines + adaptive configs that are "
                            f"Pareto-optimal within their family on DEV ({len(dev)} problems)")
        for t, d in zip(s_test, s_dev):
            t["dev_pareto"] = d["dev_pareto"]
        pareto(s_test)
        summ, evs = s_test, evs_test
    else:
        evs = [evaluate(pol, pools, args.gt, args.perm_seeds) for pol in grid]
        summ = [summarize(e, hard) for e in evs]
        pareto(summ)
        print_table(summ, f"ALL {len(pools)} problems")

    # ---- write files -----------------------------------------------------------
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for r in sorted(summ, key=lambda r: (r["family"], r["tokens"])):
            w.writerow(r)
    with open(os.path.join(out_dir, "per_problem.jsonl"), "w") as f:
        for e in evs:
            for r in e["rows"]:
                f.write(json.dumps({"policy": e["policy"].name, "family": family(e["policy"]), **r}) + "\n")
    with open(os.path.join(out_dir, "K_hist.txt"), "w") as f:
        for e in evs:
            if family(e["policy"]) != "adaptive-sketch":
                continue
            ks = Counter(int(round(r["K_used"])) for r in e["rows"])
            cands = [r["n_cands"] for r in e["rows"]]
            f.write(f"{e['policy'].name}\n  K used: {dict(sorted(ks.items()))}\n"
                    f"  candidates/problem: min={min(cands):.0f} mean={mean(cands):.1f} max={max(cands):.0f}\n"
                    f"  stops: {dict(Counter(r['stop'] for r in e['rows']))}\n")
    print(f"\nwrote {out_dir}/summary.csv, per_problem.jsonl, K_hist.txt")


if __name__ == "__main__":
    main()
