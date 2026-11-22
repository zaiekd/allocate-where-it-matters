#!/usr/bin/env python3
"""Re-run ground truth (gt_orig / gt_base / gt_plus) for selected problems of a pool,
e.g. after a fix to the checker.  The original pool file is kept as <file>.bak.

    python3 scripts/regrade.py pools/humaneval/dsv4flash_full --task-ids HumanEval/32
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.benchmarks import load_benchmark, safe_task_id  # noqa: E402
from pool.execution import ground_truth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dirs", nargs="+")
    ap.add_argument("--task-ids", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for d in a.pool_dirs:
        for tid in a.task_ids:
            path = os.path.join(d, safe_task_id(tid) + ".json")
            pool = json.load(open(path))
            probs = {p.task_id: p for p in load_benchmark(pool["benchmark"])}
            prob = probs[tid]
            cands = pool["candidates"]
            codes = {cid: c["code"] for cid, c in cands.items()}
            before = {k: sum(bool(c.get(k)) for c in cands.values()) for k in ("gt_orig", "gt_base", "gt_plus")}
            with cf.ThreadPoolExecutor(a.workers) as ex:
                res = dict(zip(codes, ex.map(lambda code: ground_truth(code, prob), codes.values())))
            for cid, r in res.items():
                cands[cid].update(r)
            after = {k: sum(bool(c.get(k)) for c in cands.values()) for k in ("gt_orig", "gt_base", "gt_plus")}
            if not os.path.exists(path + ".bak"):
                shutil.copy(path, path + ".bak")
            json.dump(pool, open(path + ".tmp", "w"))
            os.replace(path + ".tmp", path)
            print(f"{d}  {tid}: {len(cands)} candidates  correct before {before}  after {after}  "
                  f"greedy gt_plus={cands['g']['gt_plus']}")


if __name__ == "__main__":
    main()
