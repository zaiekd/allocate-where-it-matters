#!/usr/bin/env python3
"""Summarise the LiveCodeBench problems the loader keeps (LeetCode, functional).

    python3 scripts/lcb_stats.py                    # all files in data/lcb/
    LCB_SINCE=2025-01-01 python3 scripts/lcb_stats.py

Prints counts by month and difficulty, plus how many problems were dropped and why
is not tracked (stdin problems, linked lists/trees and multi-method classes are skipped).
"""

import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pool.benchmarks import LCB_DIR, load_lcb  # noqa: E402

files = sorted(glob.glob(os.path.join(LCB_DIR, "test*.jsonl")))
total = collections.Counter()
for path in files:
    with open(path) as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                total[it.get("platform")] += 1
print("files:", [os.path.basename(f) for f in files])
print("all problems by platform:", dict(total))

ps = load_lcb()
print(f"\nkept (LeetCode functional): {len(ps)}")
by = collections.Counter((p.meta["date"][:7], p.meta["difficulty"]) for p in ps)
months = sorted({m for m, _ in by})
print(f"{'month':8s} {'easy':>5s} {'medium':>7s} {'hard':>5s}")
for m in months:
    print(f"{m:8s} {by[(m, 'easy')]:5d} {by[(m, 'medium')]:7d} {by[(m, 'hard')]:5d}")
n_tests = [len(p.base_input) + len(p.plus_input) for p in ps]
if n_tests:
    print(f"\ntests per problem: min {min(n_tests)}, median {sorted(n_tests)[len(n_tests)//2]}, max {max(n_tests)}")
