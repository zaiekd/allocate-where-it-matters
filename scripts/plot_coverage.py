#!/usr/bin/env python3
"""Coverage-vs-tokens analysis (the *generation* side, no selection).

Coverage of a policy on a problem = 1 if ANY candidate the policy generated is
correct (gt_plus by default), else 0.  It is the pass@1 a perfect selector would
reach, i.e. an upper bound on what any selection rule can recover.

    python3 scripts/plot_coverage.py pools/humaneval/gemini-3.1-flash-lite-preview__low

Writes into results/<benchmark>__<model>__<thinking>/:
  coverage.csv        one row per configuration: coverage, pass@1 (same candidates,
                      selected by the policy's own selector), hard-subset counts, tokens
  coverage_tokens.png coverage vs mean tokens/problem; left = all problems,
                      right = hard subset (greedy wrong), y = problems covered

Sketch-order-dependent policies are averaged over --perm-seeds random sketch orders
(seed 0 = pool order), exactly like run_policies.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import zlib
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import AdaptiveFlat, AdaptiveSketch, BreadthSketch, FixedSketch, Flat, GatedFlat  # noqa: E402
from pool.replay import load_pool  # noqa: E402

# ---------------------------------------------------------------- configurations

FLAT_N = [1, 3, 5, 10, 20, 50, 100]
GRID_K = [1, 2, 3, 5, 10]
GRID_M = [1, 2, 3, 5, 10]


def configs(max_m: int = 10, max_n: int = 100):
    """Configurations that fit inside the pool (fills per sketch <= max_m,
    flat samples <= max_n)."""
    out = []
    for n in [x for x in FLAT_N if x <= max_n] + ([max_n] if max_n not in FLAT_N else []):
        out.append(("flat", f"N={n}", Flat(n, selector="sv_ex")))
    for k in GRID_K:
        for m in [x for x in GRID_M if x <= max_m] + ([max_m] if max_m not in GRID_M else []):
            out.append(("sketch", f"K={k},M={m}", FixedSketch(k, m, selector="sv_ex")))
    out.append(("adaptive-flat", "gate=ex", AdaptiveFlat(gate_signal="examples", N_max=max_n)))
    out.append(("adaptive-flat", "gate=fp", AdaptiveFlat(N_max=max_n)))
    for n in (10, 20):
        if 3 + n <= max_n:
            out.append(("adaptive-flat", f"ex gate +{n} flat", GatedFlat(n_extra=n)))
    out.append(("adaptive-sketch", "gate=flat/ex", AdaptiveSketch(gate_signal="examples", M_max=max_m)))
    out.append(("adaptive-sketch", "gate=flat/fp", AdaptiveSketch(M_max=max_m)))
    out.append(("adaptive-sketch", "gate=sketch/ex", AdaptiveSketch(gate_mode="sketch", gate_signal="examples", M_max=max_m)))
    out.append(("adaptive-sketch", "no gate", AdaptiveSketch(gate=False, M_max=max_m)))
    # v2: breadth-first adaptive sketching (two gates)
    out.append(("breadth-sketch", "disagree, K<=10", BreadthSketch()))
    out.append(("breadth-sketch", "disagree, K<=5", BreadthSketch(K_max=5)))
    out.append(("breadth-sketch", "examples, K<=10", BreadthSketch(gate="examples")))
    out.append(("breadth-sketch", "examples, K<=5", BreadthSketch(gate="examples", K_max=5)))
    out.append(("breadth-sketch", "disagree, full inputs", BreadthSketch(lite_inputs=False)))
    return out


# ---------------------------------------------------------------- evaluation


def orders_for(p, seeds):
    n = len(p["sketches"])
    out = [None]
    for sd in range(1, seeds):
        o = list(range(n))
        random.Random(zlib.crc32(f"{p['task_id']}|{sd}".encode())).shuffle(o)
        out.append(o)
    return out


def evaluate(pol, pools, gt, seeds, hard):
    def ok(p, cid):
        return bool(cid) and bool(p["candidates"][cid].get(gt))

    cov, acc, tok, cands = [], [], [], []
    hcov = hacc = 0.0
    for p in pools:
        rs = [pol.run(p, o) for o in orders_for(p, seeds)]
        c = st.mean(any(ok(p, x) for x in r.cids) for r in rs)
        a = st.mean(ok(p, r.chosen) for r in rs)
        cov.append(c)
        acc.append(a)
        tok.append(st.mean(r.tokens for r in rs))
        cands.append(st.mean(r.n_cands for r in rs))
        if p["task_id"] in hard:
            hcov += c
            hacc += a
    return {
        "coverage": 100 * st.mean(cov), "pass1": 100 * st.mean(acc),
        "hard_cov": hcov, "hard_pass1": hacc,
        "tokens": st.mean(tok), "cands": st.mean(cands),
    }


# ---------------------------------------------------------------- plotting

COLORS = {"flat": "#2a78d6", "sketch-breadth": "#eb6834", "sketch-depth": "#e87ba4",
          "adaptive-flat": "#1baf7a", "adaptive-sketch": "#eda100", "breadth-sketch": "#008300"}
TEXT, MUTED, GRID, GRIDPT, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b2", "#fcfcfb"


def plot(rows, n_all, n_hard, out_png, title_suffix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sel(fam, pred=lambda r: True):
        return sorted([r for r in rows if r["family"] == fam and pred(r)], key=lambda r: r["tokens"])

    flat = sel("flat")
    grid = sel("sketch")
    breadth = sorted([r for r in grid if r["config"].endswith(",M=1")], key=lambda r: r["tokens"])
    depth = sorted([r for r in grid if r["config"].startswith("K=2,")], key=lambda r: r["tokens"])
    af, ask, bs = sel("adaptive-flat"), sel("adaptive-sketch"), sel("breadth-sketch")

    panels = [("coverage", f"All problems (n={n_all})", "Coverage (%)")]
    if n_hard:
        panels.append(("hard_cov", f"Hard subset: greedy wrong (n={n_hard})", "Problems covered"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 5.2), facecolor=SURFACE)
    axes = axes if len(panels) > 1 else [axes]

    for ax, (key, title, ylab) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        # full K x M grid as recessive context
        ax.scatter([r["tokens"] for r in grid], [r[key] for r in grid], s=22, color=GRIDPT, zorder=1,
                   label="Sketch, other K×M (grid)")
        ax.plot([r["tokens"] for r in flat], [r[key] for r in flat], "-o", color=COLORS["flat"], lw=2, ms=7,
                mec=SURFACE, mew=1.5, zorder=3, label=f"Flat sampling (N = 1…{flat[-1]['config'][2:]})")
        ax.plot([r["tokens"] for r in breadth], [r[key] for r in breadth], "-s", color=COLORS["sketch-breadth"],
                lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3, label="Sketch, breadth: M=1, K = 1…10")
        ax.plot([r["tokens"] for r in depth], [r[key] for r in depth], "--s", color=COLORS["sketch-depth"],
                lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3, label=f"Sketch, depth: K=2, M = 1…{depth[-1]['config'].split('M=')[1]}")
        ax.scatter([r["tokens"] for r in af], [r[key] for r in af], marker="^", s=80, color=COLORS["adaptive-flat"],
                   edgecolor=SURFACE, linewidth=1.5, zorder=4, label="Adaptive flat / gated flat (baselines)")
        ax.scatter([r["tokens"] for r in ask], [r[key] for r in ask], marker="D", s=70,
                   color=COLORS["adaptive-sketch"], edgecolor=SURFACE, linewidth=1.5, zorder=4,
                   label="Adaptive sketch v1 (ours)")
        ax.scatter([r["tokens"] for r in bs], [r[key] for r in bs], marker="*", s=180,
                   color=COLORS["breadth-sketch"], edgecolor=SURFACE, linewidth=1.2, zorder=5,
                   label="Breadth sketch v2 (ours)")
        # selective direct labels: line end-points + adaptive configs
        for line, lab in ((flat, "N"), (breadth, "K"), (depth, "M")):
            for r in (line[0], line[-1]):
                v = r["config"].split(",")[0 if lab != "M" else 1]
                ax.annotate(v, (r["tokens"], r[key]), textcoords="offset points", xytext=(6, -12),
                            fontsize=8, color=MUTED)
        offs = {"gate=flat/ex": (-10, 8), "gate=flat/fp": (-75, -14), "gate=sketch/ex": (8, 2), "no gate": (-20, 9)}
        for r in ask:
            ax.annotate(r["config"], (r["tokens"], r[key]), textcoords="offset points",
                        xytext=offs.get(r["config"], (6, 4)), fontsize=7.5, color=MUTED)
        v2offs = ({"disagree, K<=10": (-30, 12), "examples, K<=10": (8, -14)} if key == "coverage"
                  else {"disagree, K<=10": (-40, 12), "examples, K<=10": (10, -4)})
        for r in bs:
            if r["config"] in v2offs:
                ax.annotate("v2: " + r["config"], (r["tokens"], r[key]), textcoords="offset points",
                            xytext=v2offs[r["config"]], fontsize=8, color=TEXT, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("Mean tokens per problem (log scale)", color=TEXT)
        ax.set_ylabel(ylab, color=TEXT)
        ax.set_title(title, loc="left", color=TEXT, fontsize=12)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
        if key == "hard_cov":
            ax.set_ylim(-0.3, n_hard + 0.3)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=TEXT)
    fig.suptitle(f"Coverage (any correct candidate) vs cost{title_suffix}", x=0.01, ha="left", color=TEXT,
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool_dir")
    ap.add_argument("--gt", default="gt_plus", choices=["gt_plus", "gt_base", "gt_orig"])
    ap.add_argument("--perm-seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pools = load_pool(args.pool_dir)
    if not pools:
        sys.exit(f"no pools in {args.pool_dir}")
    bench = pools[0]["benchmark"]
    tag = os.path.basename(os.path.normpath(args.pool_dir))
    out = args.out or os.path.join(ROOT, "results", f"{bench}__{tag}")
    os.makedirs(out, exist_ok=True)
    hard = {p["task_id"] for p in pools if not p["candidates"]["g"].get(args.gt)}

    max_m = min(len(s["fills"]) for p in pools for s in p["sketches"] if s["valid"] and s["fills"])
    max_n = min(len(p["flat"]) for p in pools)
    print(f"pool size: fills/sketch <= {max_m}, flat samples <= {max_n}")
    rows = []
    for fam, cfg, pol in configs(max_m, max_n):
        r = evaluate(pol, pools, args.gt, args.perm_seeds if fam != "flat" and fam != "adaptive-flat" else 1, hard)
        r.update(family=fam, config=cfg)
        rows.append(r)
        print(f"{fam:16s} {cfg:22s} cov={r['coverage']:5.1f}% pass@1={r['pass1']:5.1f}%  "
              f"hard cov={r['hard_cov']:4.1f}/{len(hard)} pass@1={r['hard_pass1']:4.1f}/{len(hard)}  "
              f"tokens={r['tokens']:8.0f}  cands={r['cands']:5.1f}", flush=True)

    csv_path = os.path.join(out, "coverage.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "config", "coverage", "pass1", "hard_cov", "hard_pass1",
                                          "tokens", "cands"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    png = os.path.join(out, "coverage_tokens.png")
    plot(rows, len(pools), len(hard), png, f" — {bench}, {pools[0].get('model', '')}, {args.gt}")
    print(f"\nwrote {csv_path}\nwrote {png}")


if __name__ == "__main__":
    main()
