#!/usr/bin/env python3
"""Cross-model figure: pass@1 vs mean tokens, one panel per model.

    python3 scripts/plot_models.py results/lcb__gemini-lite_full results/lcb__dsv4flash_full \
        results/lcb__gpt41mini_full --out results/lcb_models.png

Reads stats.json written by scripts/analyze_pool.py.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# family -> (color, marker, legend label)
FAM = {
    "greedy": ("#52514e", "o", "Greedy"),
    "flat": ("#2a78d6", "o", "Flat sampling (N=10, 30)"),
    "fixed": ("#eb6834", "s", "Sketch-and-Verify, fixed (K=2,M=3 / K=10,M=3)"),
    "v2": ("#eda100", "D", "Adaptive sketch v2 (examples gate)"),
    "gated": ("#1baf7a", "^", "Gate + flat (+10)"),
    "aflat": ("#008300", "*", "Adaptive flat (fingerprint stop)"),
}
PICK = [("greedy", "greedy"), ("flat N=10", "flat"), ("flat N=30", "flat"),
        ("fixed K=2,M=3", "fixed"), ("fixed K=10,M=3", "fixed"), ("v2-examples", "v2"),
        ("gated-flat +10", "gated"), ("adaptive-flat fp", "aflat")]
NAMES = {"gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite", "deepseek-flash": "DeepSeek V4.1 Flash",
         "openai/gpt-4.1-mini": "GPT-4.1 mini"}  # noqa: E501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="LiveCodeBench (LeetCode functional, 125 problems): pass@1 vs cost")
    a = ap.parse_args()
    data = [json.load(open(os.path.join(d, "stats.json"))) for d in a.dirs]
    fig, axes = plt.subplots(1, len(data), figsize=(5.6 * len(data), 4.8), facecolor=SURFACE, sharey=False)
    axes = axes if len(data) > 1 else [axes]
    for ax, s in zip(axes, data):
        ax.set_facecolor(SURFACE)
        mid = s.get("model") or json.load(open(os.path.join(s["pool"], sorted(os.listdir(s["pool"]))[0])))["model"]
        model = NAMES.get(mid, mid)
        seen = set()
        for name, fam in PICK:
            m = s["methods"][name]
            c, mk, lab = FAM[fam]
            ax.scatter(m["tokens"], m["pass1"], color=c, marker=mk, s=150 if mk == "*" else 70,
                       edgecolor=SURFACE, linewidth=1.2, zorder=3, label=None if fam in seen else lab)
            seen.add(fam)
            short = {"greedy": "greedy", "flat N=10": "N=10", "flat N=30": "N=30", "fixed K=2,M=3": "K2M3",
                     "fixed K=10,M=3": "K10M3", "v2-examples": "v2", "gated-flat +10": "gate+10",
                     "adaptive-flat fp": "adapt"}[name]
            ax.annotate(short, (m["tokens"], m["pass1"]), textcoords="offset points", xytext=(6, -11),
                        fontsize=7.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_title(f"{model}  (n={s['n']}, greedy-wrong={s['n_hard']})", loc="left", fontsize=10.5, color=TEXT)
        ax.set_xlabel("Mean tokens per problem (log)", color=TEXT)
        ax.set_ylabel("pass@1 (%)", color=TEXT)
        ax.grid(True, color=GRID, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(a.title, x=0.01, ha="left",
                 fontsize=12.5, color=TEXT)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
