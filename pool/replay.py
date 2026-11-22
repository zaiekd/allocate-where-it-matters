"""Offline replay on a saved pool: selection + token accounting.

`select_semantic_voting` reproduces sketch-and-verify/src/select.py
(strategy B) exactly: cluster compiling candidates by fingerprint, prefer
all-ok clusters, take the largest (stable order), return the shortest program.

Every policy (fixed K/M, flat N, and later the adaptive ones) should express
itself as "which candidate ids did I generate, which LLM calls did I pay for",
then call `select_semantic_voting` + `tokens_of` so that comparisons are
apples-to-apples.
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

from .llm import cost_usd


def load_pool(pool_dir: str) -> List[dict]:
    pools = []
    for path in sorted(glob.glob(os.path.join(pool_dir, "*.json"))):
        with open(path) as f:
            pools.append(json.load(f))
    # natural order by numeric task id suffix
    def key(p):
        tail = p["task_id"].split("/")[-1]
        return (p["task_id"].split("/")[0], int(tail) if tail.isdigit() else tail)
    return sorted(pools, key=key)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def select_semantic_voting(pool: dict, cids: Iterable[str], prefer_all_ok: bool = True) -> Optional[str]:
    cands = pool["candidates"]
    groups: "OrderedDict[tuple, List[str]]" = OrderedDict()
    for cid in cids:
        c = cands[cid]
        if not c["compiles"]:
            continue
        groups.setdefault(tuple(c["fp"]), []).append(cid)
    if not groups:
        return None
    clusters = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)  # stable
    if prefer_all_ok:
        ok = [kv for kv in clusters if all(x.startswith("ok:") for x in kv[0])]
        if ok:
            clusters = ok
    best = clusters[0][1]
    return min(best, key=lambda cid: cands[cid]["code_len"])


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------


def _calls_of(pool: dict, sketch_ids: Iterable[int], fills: Dict[int, int],
              flat_n: int, greedy: bool, test_inputs: bool, categories: bool) -> List[dict]:
    calls: List[dict] = []
    if test_inputs:
        calls += pool["test_inputs"]["calls"]
    if categories:
        calls.append(pool["categories"]["call"])
    for k in sketch_ids:
        s = pool["sketches"][k]
        calls.append(s["call"])
        calls += [f["call"] for f in s["fills"][: fills.get(k, 0)]]
    if greedy:
        calls.append(pool["greedy"]["call"])
    calls += [f["call"] for f in pool["flat"][:flat_n]]
    return calls


def tokens_of(pool: dict, calls: List[dict]) -> dict:
    i = sum(c["in_tok"] for c in calls)
    o = sum(c["out_tok"] for c in calls)
    t = sum(c["think_tok"] for c in calls)
    return {"tokens": i + o + t, "in_tok": i, "out_tok": o, "think_tok": t,
            "usd": _usd(pool, i, o, t), "n_calls": len(calls)}


def _usd(pool: dict, i: int, o: int, t: int) -> float:
    if pool.get("price_per_1m"):
        pin, pout = pool["price_per_1m"]
        return (i * pin + (o + t) * pout) / 1_000_000
    return cost_usd(pool["model"], i, o, t)


# ---------------------------------------------------------------------------
# fixed (non-adaptive) baselines, as in the paper
# ---------------------------------------------------------------------------


def replay_fixed_sketch(pool: dict, K: int, M: int) -> dict:
    ks = list(range(min(K, len(pool["sketches"]))))
    cids = [f["cid"] for k in ks for f in pool["sketches"][k]["fills"][:M]]
    chosen = select_semantic_voting(pool, cids)
    calls = _calls_of(pool, ks, {k: M for k in ks}, 0, False, True, True)
    return {"chosen": chosen, "n_cands": len(cids), **tokens_of(pool, calls)}


def replay_flat(pool: dict, N: int) -> dict:
    if N == 1:  # greedy, no selection, no test inputs (as in run_scaling_fast)
        calls = _calls_of(pool, [], {}, 0, True, False, False)
        return {"chosen": "g", "n_cands": 1, **tokens_of(pool, calls)}
    cids = [f["cid"] for f in pool["flat"][:N]]
    chosen = select_semantic_voting(pool, cids)
    calls = _calls_of(pool, [], {}, N, False, True, False)
    return {"chosen": chosen, "n_cands": len(cids), **tokens_of(pool, calls)}


def is_correct(pool: dict, cid: Optional[str], gt: str = "gt_plus") -> bool:
    return bool(cid) and bool(pool["candidates"][cid][gt])
