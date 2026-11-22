#!/usr/bin/env python3
"""Hyperparameter sensitivity and held-out selection (no API calls).

    python3 scripts/hparam_sensitivity.py            # the six default pools

For each method family a grid of hyperparameters is evaluated on every pool.
Reported (results/hparams/):
  1. regret of the default setting: default pass@1 vs the best grid setting in
     hindsight, per pool (and vs the best setting that is not more expensive);
  2. leave-one-pool-out (LOPO) and leave-one-benchmark-out (LOBO) selection:
     pick the setting with the best mean pass@1 on the OTHER pools (optionally
     under a cost cap) and evaluate it on the held-out pool(s);
  3. robustness of the main claim for GFS: share of grid settings whose pass@1
     is within 2 points of flat N=10 while costing less.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import statistics as st
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import AdaptiveFlat, BreadthSketch, Flat, GatedFlat, StopRule  # noqa: E402
from pool.replay import load_pool  # noqa: E402

POOLS = [
    ("HE+ Gemini", "pools/humaneval/gemini-3.1-flash-lite-preview__low"),
    ("HE+ DeepSeek", "pools/humaneval/dsv4flash_full"),
    ("HE+ GPT-4.1m", "pools/humaneval/gpt41mini_full"),
    ("LCB Gemini", "pools/lcb/gemini-lite_full"),
    ("LCB DeepSeek", "pools/lcb/dsv4flash_full"),
    ("LCB GPT-4.1m", "pools/lcb/gpt41mini_full"),
]
SEEDS = 5


def grids():
    gfs = {f"gate={g},n={n},+{x}": GatedFlat(gate=g, gate_n=n, n_extra=x)
           for g, n, x in itertools.product(["examples", "fp-lite"], [1, 2, 3, 5], [0, 5, 10, 15, 20, 27])}
    bas = {f"gate={g},K<={k},m={m}": BreadthSketch(gate=g, K_max=k, m_breadth=m)
           for g, k, m in itertools.product(["examples", "disagree"], [3, 5, 7, 10], [1, 2, 3])}
    af = {f"size>={s},tau={t}": AdaptiveFlat(N_max=30, rule=StopRule(min_size=s, tau=t))
          for s, t in itertools.product([2, 3, 5], [0.5, 0.6, 0.8])}
    return {
        "GFS": (gfs, "gate=examples,n=3,+10"),
        "BAS": (bas, "gate=disagree,K<=10,m=1"),
        "AdaptiveFlat": (af, "size>=3,tau=0.6"),
    }


def orders(p, seeds):
    n = len(p["sketches"])
    out = [None]
    for sd in range(1, seeds):
        o = list(range(n))
        random.Random(zlib.crc32(f"{p['task_id']}|{sd}".encode())).shuffle(o)
        out.append(o)
    return out


def ev(pol, pools):
    dep = isinstance(pol, BreadthSketch)
    acc, tok = [], []
    for p in pools:
        rs = [pol.run(p, o) for o in (orders(p, SEEDS) if dep else [None])]
        acc.append(st.mean(bool(r.chosen) and bool(p["candidates"][r.chosen].get("gt_plus")) for r in rs))
        tok.append(st.mean(r.tokens for r in rs))
    return 100 * st.mean(acc), st.mean(tok)


def select(res, names, pools, cap=None, ref_tok=None):
    """best mean pass@1 over `pools`; optional cap on mean relative cost (tokens / flat-10 tokens)."""
    best = None
    for nm in names:
        acc = st.mean(res[nm][pl][0] for pl in pools)
        rel = st.mean(res[nm][pl][1] / ref_tok[pl] for pl in pools)
        if cap is not None and rel > cap:
            continue
        key = (round(acc, 6), -rel)
        if best is None or key > best[0]:
            best = (key, nm)
    return best[1] if best else None


def main():
    out_dir = os.path.join(ROOT, "results", "hparams")
    os.makedirs(out_dir, exist_ok=True)
    data = {lab: load_pool(os.path.join(ROOT, path)) for lab, path in POOLS}
    labs = [lab for lab, _ in POOLS]
    ref = {lab: ev(Flat(10, selector="sv_ex"), data[lab]) for lab in labs}
    greedy = {lab: ev(Flat(1), data[lab]) for lab in labs}
    ref_tok = {lab: ref[lab][1] for lab in labs}
    lines = ["reference per pool: flat N=10 (examples filter) pass@1 / tokens; greedy pass@1"]
    for lab in labs:
        lines.append(f"  {lab:14s} flat10 {ref[lab][0]:5.1f}% / {ref[lab][1]:7.0f}   greedy {greedy[lab][0]:5.1f}%")
    dump = {"reference": {lab: {"flat10": ref[lab], "greedy": greedy[lab]} for lab in labs}}

    for fam, (grid, default) in grids().items():
        res = {nm: {lab: ev(pol, data[lab]) for lab in labs} for nm, pol in grid.items()}
        names = list(grid)
        lines += ["", f"######## {fam}: {len(names)} settings; default = {default}", "",
                  "1) default vs best setting in hindsight (pass@1 points)"]
        lines.append(f"   {'pool':14s} {'default':>14s} {'best (any)':>12s} {'regret':>7s} {'best (<= default cost)':>24s} {'regret':>7s}")
        regrets = []
        for lab in labs:
            da, dt = res[default][lab]
            best_any = max(names, key=lambda n: (res[n][lab][0], -res[n][lab][1]))
            cheap = [n for n in names if res[n][lab][1] <= dt * 1.0001]
            best_cheap = max(cheap, key=lambda n: (res[n][lab][0], -res[n][lab][1]))
            r1 = res[best_any][lab][0] - da
            r2 = res[best_cheap][lab][0] - da
            regrets.append((r1, r2))
            lines.append(f"   {lab:14s} {da:6.1f}%/{dt / 1000:5.1f}k {res[best_any][lab][0]:6.1f}%/{res[best_any][lab][1] / 1000:4.1f}k {r1:+6.1f}  "
                         f"{res[best_cheap][lab][0]:6.1f}%/{res[best_cheap][lab][1] / 1000:4.1f}k {r2:+15.1f}")
            lines.append(f"   {'':14s} best(any)={best_any}  best(<=cost)={best_cheap}")
        lines.append(f"   mean regret: any {st.mean(r[0] for r in regrets):.2f}, at <= default cost {st.mean(r[1] for r in regrets):.2f}")

        lines += ["", "2) held-out selection (choose on the other pools, report on the held-out one)"]
        dump[fam] = {"default": default, "grid": {n: {lab: res[n][lab] for lab in labs} for n in names}, "lopo": {}, "lobo": {}}
        for cap in (None, 0.7):
            tag = "no cost cap" if cap is None else f"cost cap: mean tokens <= {cap:.0%} of flat N=10"
            lines.append(f"   -- LOPO, {tag}")
            dif = []
            for lab in labs:
                others = [x for x in labs if x != lab]
                nm = select(res, names, others, cap, ref_tok)
                if nm is None:
                    lines.append(f"   {lab:14s} no setting meets the cost cap")
                    continue
                a, t = res[nm][lab]
                da, dt = res[default][lab]
                dif.append(a - da)
                lines.append(f"   {lab:14s} chosen {nm:28s} {a:5.1f}%/{t / 1000:5.1f}k   default {da:5.1f}%/{dt / 1000:5.1f}k   Δ {a - da:+5.1f}"
                             f"   vs flat10 {a - ref[lab][0]:+5.1f} at {t / ref_tok[lab]:.0%} cost")
                dump[fam]["lopo"].setdefault(str(cap), {})[lab] = {"chosen": nm, "pass1": a, "tokens": t}
            if dif:
                lines.append(f"   mean Δ vs default: {st.mean(dif):+.2f}")
            lines.append(f"   -- LOBO (select on one benchmark, test on the other), {tag}")
            for train, test in (("HE+", "LCB"), ("LCB", "HE+")):
                tr = [x for x in labs if x.startswith(train)]
                te = [x for x in labs if x.startswith(test)]
                nm = select(res, names, tr, cap, ref_tok)
                if nm is None:
                    lines.append(f"   select on {train:4s} -> no setting meets the cost cap")
                    continue
                cells = ", ".join(f"{x.split()[1]} {res[nm][x][0]:.1f}% ({res[nm][x][0] - res[default][x][0]:+.1f} vs default)" for x in te)
                lines.append(f"   select on {train:4s} -> {nm:28s} test on {test}: {cells}")
                dump[fam]["lobo"].setdefault(str(cap), {})[f"{train}->{test}"] = nm

        if fam == "GFS":
            lines += ["", "3) robustness of 'GFS ~ flat N=10 accuracy at lower cost' over the grid (settings with +5 or more extra samples)"]
            sub = [n for n in names if not n.endswith(",+0")]
            for lab in labs:
                okc = [n for n in sub if res[n][lab][0] >= ref[lab][0] - 2.0 and res[n][lab][1] < ref_tok[lab]]
                rel = [res[n][lab][1] / ref_tok[lab] for n in sub]
                accs = [res[n][lab][0] for n in sub]
                lines.append(f"   {lab:14s} {len(okc):2d}/{len(sub)} settings within 2 pts of flat10 and cheaper;"
                             f" pass@1 range {min(accs):.1f}-{max(accs):.1f}% (flat10 {ref[lab][0]:.1f}%), cost {min(rel):.0%}-{max(rel):.0%} of flat10")

        print(f"[done] {fam}", flush=True)

    text = "\n".join(lines)
    print(text)
    open(os.path.join(out_dir, "hparam_sensitivity.txt"), "w").write(text + "\n")
    json.dump(dump, open(os.path.join(out_dir, "hparam_sensitivity.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/hparam_sensitivity.txt and .json")


if __name__ == "__main__":
    main()
