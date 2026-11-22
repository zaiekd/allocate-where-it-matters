#!/usr/bin/env python3
"""Sanity-check a pool: replay the paper's fixed configurations offline.

python scripts/inspect_pool.py pools/humaneval/gemini-3.1-flash-lite-preview__low

Prints pass@1 under three ground truths
  orig = original HumanEval check()/MBPP asserts (what the paper's code used)
  base = EvalPlus base inputs,  plus = EvalPlus base+plus inputs (HumanEval+/MBPP+)
plus mean tokens / $ per problem and pool diagnostics.
"""

from __future__ import annotations

import os
import sys
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.replay import (  # noqa: E402
    is_correct, load_pool, replay_fixed_sketch, replay_flat,
)

SKETCH = [(1, 1), (2, 5), (5, 10), (10, 10)]
FLAT = [1, 10, 50, 100]


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pools = load_pool(sys.argv[1])
    n = len(pools)
    if n == 0:
        print(f"no pool files in {sys.argv[1]}")
        sys.exit(1)
    print(f"{n} problems in {sys.argv[1]}\n")

    print(f"{'config':<16}{'orig':>7}{'base':>7}{'plus':>7}{'tokens':>10}{'$/prob':>11}")
    rows = [("flat N=%d" % N, lambda p, N=N: replay_flat(p, N)) for N in FLAT]
    rows += [("sketch K=%d,M=%d" % km, lambda p, km=km: replay_fixed_sketch(p, *km)) for km in SKETCH]
    for name, fn in rows:
        rs = [(p, fn(p)) for p in pools]
        acc = {gt: 100 * mean(is_correct(p, r["chosen"], gt) for p, r in rs)
               for gt in ("gt_orig", "gt_base", "gt_plus")}
        print(f"{name:<16}{acc['gt_orig']:>6.1f}%{acc['gt_base']:>6.1f}%{acc['gt_plus']:>6.1f}%"
              f"{mean(r['tokens'] for _, r in rs):>10.0f}{mean(r['usd'] for _, r in rs):>11.2e}")

    # ---- diagnostics -------------------------------------------------------
    print("\npool diagnostics (mean per problem)")
    def clusters(p, cids):
        return len({tuple(p["candidates"][c]["fp"]) for c in cids if p["candidates"][c]["compiles"]})
    sk_cids = lambda p: [f["cid"] for s in p["sketches"] for f in s["fills"]]  # noqa: E731
    fl_cids = lambda p: [f["cid"] for f in p["flat"]]  # noqa: E731
    print(f"  valid sketches          {mean(sum(s['valid'] for s in p['sketches']) for p in pools):.1f}"
          f" / {pools[0]['config']['K']}")
    print(f"  category names parsed   {sum(p['categories']['parse_ok'] for p in pools)}/{n}"
          "   (failure -> generic 'approach_i' names, sketches lose strategy diversity)")
    def trunc(calls):
        calls = list(calls)
        return f"{100 * sum(c.get('finish') in ('length', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS') for c in calls) / max(len(calls), 1):.1f}%"
    print("  truncated (max_tokens)  "
          f"categories {trunc(p['categories']['call'] for p in pools)}  "
          f"sketches {trunc(s['call'] for p in pools for s in p['sketches'])}  "
          f"fills {trunc(f['call'] for p in pools for s in p['sketches'] for f in s['fills'])}  "
          f"flat {trunc(f['call'] for p in pools for f in p['flat'])}")
    print(f"  thinking tokens / call  "
          f"{mean(c['think_tok'] for p in pools for c in [p['categories']['call']] + [f['call'] for f in p['flat']]):.0f}"
          "   (should be 0 when THINKING_LEVEL=disabled)")
    print(f"  test inputs             {mean(len(p['test_inputs']['inputs']) for p in pools):.1f}")
    print(f"  public examples         {mean(len(p['examples']) for p in pools):.1f}"
          f"  (problems with >=1: {sum(bool(p['examples']) for p in pools)}/{n})")
    print(f"  fingerprint clusters    flat-100: {mean(clusters(p, fl_cids(p)) for p in pools):.2f}"
          f"   sketch-10x10: {mean(clusters(p, sk_cids(p)) for p in pools):.2f}")
    for label, f in (("flat", fl_cids), ("sketch", sk_cids)):
        cs = [p["candidates"][c] for p in pools for c in f(p)]
        if cs:
            print(f"  {label:<7} compiles {100*mean(c['compiles'] for c in cs):5.1f}%  "
                  f"pass examples {100*mean(c['examples_pass'] for c in cs):5.1f}%  "
                  f"correct(plus) {100*mean(c['gt_plus'] for c in cs):5.1f}%")
    stage = {}
    for p in pools:
        for k, v in p["cost"].items():
            stage[k] = stage.get(k, 0) + v["in_tok"] + v["out_tok"] + v["think_tok"]
    print("  tokens by stage         " + ", ".join(f"{k}={v / n:.0f}" for k, v in stage.items()))
    print(f"  pool cost               ${sum(v['usd'] for p in pools for v in p['cost'].values()):.4f} total")


if __name__ == "__main__":
    main()
