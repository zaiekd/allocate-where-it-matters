#!/usr/bin/env python3
"""Cost-accuracy figures from run_policies.py output (needs matplotlib).

python scripts/plot_policies.py results/humaneval__deepseek-flash__disabled

Writes, into the same folder:
  cost_accuracy.png   pass@1 vs mean tokens/problem, one panel for all problems
                      and (if any) one for the hard subset (greedy wrong)
  budget_hist.png     how many candidates each problem received under the
                      best adaptive-sketch configuration (adaptivity at a glance)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

# reference categorical palette (slots 1-4, fixed order) + text tokens
COLORS = {"flat": "#2a78d6", "sketch": "#eb6834", "adaptive-flat": "#1baf7a", "adaptive-sketch": "#eda100"}
MARKERS = {"flat": "o", "sketch": "s", "adaptive-flat": "^", "adaptive-sketch": "D"}
LABELS = {"flat": "Flat sampling (fixed N)", "sketch": "Sketch-and-Verify (fixed K, M)",
          "adaptive-flat": "Adaptive flat (baseline)", "adaptive-sketch": "Adaptive sketch (ours)"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(res_dir):
    with open(os.path.join(res_dir, "summary.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("acc", "hard_acc", "tokens", "K_used", "fills", "cands"):
            r[k] = float(r[k])
        r["hard_n"] = int(r["hard_n"])
    return rows


def frontier(pts):
    pts = sorted(pts, key=lambda r: (r["tokens"], -r["acc"]))
    out, best = [], -1.0
    for r in pts:
        if r["acc"] > best:
            out.append(r)
            best = r["acc"]
    return out


def panel(ax, rows, acc_key, title):
    for fam in ("flat", "sketch", "adaptive-flat", "adaptive-sketch"):
        pts = [r for r in rows if r["family"] == fam and r[acc_key] == r[acc_key]]  # drop NaN
        if not pts:
            continue
        c, m = COLORS[fam], MARKERS[fam]
        if fam in ("flat", "sketch"):  # fixed configs: connect in budget order
            pts = sorted(pts, key=lambda r: r["tokens"])
            ax.plot([r["tokens"] for r in pts], [r[acc_key] for r in pts], color=c, lw=2,
                    marker=m, ms=7, mec="white", mew=1.5, label=LABELS[fam], zorder=3)
        else:  # adaptive sweeps: cloud + within-family frontier
            ax.scatter([r["tokens"] for r in pts], [r[acc_key] for r in pts], color=c, marker=m,
                       s=36, alpha=0.35, edgecolors="none", zorder=2)
            fr = frontier([dict(r, acc=r[acc_key]) for r in pts])
            ax.plot([r["tokens"] for r in fr], [r["acc"] for r in fr], color=c, lw=2, marker=m,
                    ms=7, mec="white", mew=1.5, label=LABELS[fam] + " — frontier", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Mean tokens per problem (log scale)", color=MUTED)
    ax.set_ylabel("pass@1 (%)", color=MUTED)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    res = sys.argv[1]
    rows = load(res)
    n = int(rows[0]["n"])
    hard_n = max(r["hard_n"] for r in rows)
    ncol = 2 if hard_n > 0 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.4 * ncol, 4.6), squeeze=False)
    panel(axes[0][0], rows, "acc", f"All problems (n={n})")
    if ncol == 2:
        panel(axes[0][1], rows, "hard_acc", f"Hard subset: greedy wrong (n={hard_n})")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    fig.tight_layout()
    out1 = os.path.join(res, "cost_accuracy.png")
    fig.savefig(out1, dpi=180)
    plt.close(fig)

    # ---- budget histogram for the most accurate (then cheapest) adaptive-sketch config
    ad = [r for r in rows if r["family"] == "adaptive-sketch"]
    out2 = None
    if ad:
        best = sorted(ad, key=lambda r: (-r["acc"], r["tokens"]))[0]["policy"]
        cands = []
        with open(os.path.join(res, "per_problem.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                if d["policy"] == best:
                    cands.append(int(round(d["n_cands"])))
        cnt = Counter(cands)
        xs = sorted(cnt)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.bar([str(x) for x in xs], [cnt[x] for x in xs], color=COLORS["adaptive-sketch"], width=0.7)
        ax.set_xlabel("Candidates generated for the problem", color=MUTED)
        ax.set_ylabel("Problems", color=MUTED)
        ax.set_title(f"Per-problem budget, {best[:60]}…" if len(best) > 60 else f"Per-problem budget, {best}",
                     loc="left", color=TEXT, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        out2 = os.path.join(res, "budget_hist.png")
        fig.savefig(out2, dpi=180)
        plt.close(fig)
    print("wrote", out1, out2 or "")


if __name__ == "__main__":
    main()
