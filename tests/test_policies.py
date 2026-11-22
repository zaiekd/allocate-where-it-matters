"""Tests for pool/policies.py on small synthetic pools.

    python tests/test_policies.py          (no pytest needed)
    python -m pytest tests/                (also works)
"""

from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.policies import (  # noqa: E402
    AdaptiveFlat, AdaptiveSketch, BreadthSketch, FixedSketch, Flat, GatedFlat, Session, StopRule, select_sv,
)
from pool.replay import load_pool, replay_fixed_sketch, replay_flat  # noqa: E402

CALL = {"in_tok": 10, "out_tok": 5, "think_tok": 0, "finish": "stop", "latency_s": 0, "cached": False}


def cand(fp_key: str, ex: bool = True, correct: bool = True, compiles: bool = True, code_len: int = 50):
    return {"code": "x", "code_len": code_len, "code_hash": fp_key, "compiles": compiles,
            "load_error": None if compiles else "SyntaxError",
            "examples": ["pass" if ex else "fail"], "examples_pass": ex,
            "fp": [f"ok:{fp_key}"] * 3, "all_ok": True,
            "gt_orig": correct, "gt_base": correct, "gt_plus": correct}


def make_pool(sketch_specs, flat_specs, greedy=("A", True, True), n_examples=1):
    """sketch_specs: list of (valid, [ (fp, ex, correct), ... fills ])"""
    pool = {"task_id": "T/0", "model": "mock", "price_per_1m": [1.0, 1.0],
            "examples": ["assert f(1) == 1"] * n_examples,
            "test_inputs": {"inputs": ["1", "2", "3"], "calls": [CALL] * 4},
            "categories": {"names": [], "parse_ok": True, "call": CALL},
            "sketches": [], "flat": [], "candidates": {}}
    for k, (valid, fills) in enumerate(sketch_specs):
        s = {"k": k, "valid": valid, "call": CALL, "fills": []}
        for m, (f, ex, ok) in enumerate(fills if valid else []):
            cid = f"s{k}f{m}"
            s["fills"].append({"m": m, "cid": cid, "call": CALL})
            pool["candidates"][cid] = cand(f, ex, ok)
        pool["sketches"].append(s)
    for n, (f, ex, ok) in enumerate(flat_specs):
        pool["flat"].append({"n": n, "cid": f"f{n}", "call": CALL})
        pool["candidates"][f"f{n}"] = cand(f, ex, ok)
    pool["greedy"] = {"cid": "g", "call": CALL}
    pool["candidates"]["g"] = cand(*greedy)
    return pool


# ---------------------------------------------------------------------------


def test_fixed_matches_replay_on_real_pools():
    """Session-based FixedSketch/Flat must reproduce replay.py exactly."""
    dirs = [d for d in glob.glob(os.path.join(ROOT, "pools", "*", "*")) if glob.glob(os.path.join(d, "*.json"))]
    if not dirs:
        return
    pools = load_pool(dirs[0])[:5]
    for p in pools:
        for K, M in [(1, 1), (2, 5), (10, 10)]:
            a, b = FixedSketch(K, M).run(p), replay_fixed_sketch(p, K, M)
            assert (a.chosen, a.tokens) == (b["chosen"], b["tokens"]), (p["task_id"], K, M)
        for N in [1, 10, 100]:
            a, b = Flat(N).run(p), replay_flat(p, N)
            assert (a.chosen, a.tokens) == (b["chosen"], b["tokens"]), (p["task_id"], N)


def test_gate_easy_returns_immediately():
    p = make_pool([(True, [("A", True, True)] * 10)] * 3, [("A", True, True)] * 10)
    r = AdaptiveSketch(gate_mode="flat", gate_n=3).run(p)
    assert r.stop == "gate" and r.K_used == 0 and r.n_cands == 3
    # examples-only gate does not pay for test inputs
    r2 = AdaptiveSketch(gate_mode="flat", gate_n=3, gate_signal="examples").run(p)
    assert r2.stop == "gate" and r2.tokens < r.tokens


def test_gate_hard_when_samples_disagree():
    p = make_pool([(True, [("A", True, True)] * 10)] * 3,
                  [("A", True, True), ("B", True, False), ("A", True, True)] + [("A", True, True)] * 7)
    r = AdaptiveSketch(gate_mode="flat", gate_n=3).run(p)
    assert r.stop != "gate" and r.K_used >= 1


def test_early_stop_needs_cross_strategy_support():
    # every sketch's fills agree on A; flat gate disagrees -> go to sketches
    sk = [(True, [("A", True, True)] * 10)] * 5
    flat = [("A", True, True), ("B", True, False), ("C", True, False)] + [("A", True, True)] * 7
    p = make_pool(sk, flat)
    r = AdaptiveSketch(rule=StopRule(min_size=3, tau=0.6, min_support=2), m0=2).run(p)
    assert r.stop == "early-stop" and r.K_used == 2, (r.stop, r.K_used, r.trace)
    r3 = AdaptiveSketch(rule=StopRule(min_size=3, tau=0.6, min_support=3), m0=2).run(p)
    assert r3.K_used == 3


def test_invalid_and_failing_sketches_are_pruned():
    sk = [(False, []),  # invalid sketch: paid, never filled
          (True, [("X", False, False)] * 10),  # every fill fails examples -> pruned
          (True, [("A", True, True)] * 10),
          (True, [("A", True, True)] * 10)]
    flat = [("A", True, True), ("B", True, False), ("C", True, False)] + [("A", True, True)] * 7
    p = make_pool(sk, flat)
    r = AdaptiveSketch(rule=StopRule(min_size=3, tau=0.5, min_support=2), m0=2).run(p)
    assert r.fills.get(0, 0) == 0
    assert r.fills[1] == 2  # only the initial fills
    assert p["candidates"][r.chosen]["gt_plus"]


def test_budget_is_respected():
    sk = [(True, [(f"F{k}{m}", True, False) for m in range(10)]) for k in range(10)]  # never converges
    flat = [(f"Z{n}", True, False) for n in range(10)]
    p = make_pool(sk, flat)
    for B in (10, 25, 60):
        r = AdaptiveSketch(budget=B).run(p)
        assert r.n_cands <= B, (B, r.n_cands)


def test_halving_concentrates_fills():
    # sketch 2 is the only one whose fills pass examples; it should get the most fills
    sk = [(True, [(f"N{k}{m}", m % 2 == 0, False) for m in range(10)]) for k in range(4)]
    sk[2] = (True, [("A", True, True)] * 10)
    flat = [("Q", True, False), ("R", True, False), ("S", True, False)] + [("T", True, False)] * 7
    p = make_pool(sk, flat)
    r = AdaptiveSketch(rule=StopRule(min_size=99, tau=1.0, min_support=9)).run(p)  # never stops early
    assert max(r.fills, key=r.fills.get) == 2, r.fills
    assert r.fills[2] == 10


def test_all_mechanisms_off_equals_fixed_K10_M10():
    sk = [(True, [(f"F{k}{m % 3}", True, m % 3 == 0) for m in range(10)]) for k in range(10)]
    p = make_pool(sk, [("A", True, True)] * 10)
    a = AdaptiveSketch(gate=False, early_stop=False, halving=False, M_uniform=10, selector="sv").run(p)
    b = FixedSketch(10, 10).run(p)
    assert (a.chosen, a.tokens, a.n_cands) == (b.chosen, b.tokens, b.n_cands)


def test_adaptive_flat_stops_and_respects_cap():
    p = make_pool([], [("A", True, True)] * 100)
    r = AdaptiveFlat(rule=StopRule(min_size=3, tau=0.6)).run(p)
    assert r.stop == "converged" and r.n_flat == 3
    p2 = make_pool([], [(f"Z{n}", True, False) for n in range(100)])
    r2 = AdaptiveFlat(N_max=20).run(p2)
    assert r2.stop == "budget" and r2.n_flat <= 21


def test_session_prefix_order():
    sk = [(True, [(f"F{k}{m}", True, True) for m in range(10)]) for k in range(3)]
    p = make_pool(sk, [])
    s = Session(p, sketch_order=[2, 0, 1])
    k = s.sketch(0)
    assert k == 2
    assert [s.fill(k) for _ in range(3)] == ["s2f0", "s2f1", "s2f2"]


# ---- BreadthSketch (v2) -----------------------------------------------------


def test_breadth_gate_off_equals_fixed_K10_M1_on_real_pools():
    dirs = [d for d in glob.glob(os.path.join(ROOT, "pools", "*", "*")) if glob.glob(os.path.join(d, "*.json"))]
    if not dirs:
        return
    for p in load_pool(dirs[0])[:8]:
        a = BreadthSketch(gate="off", K_max=10, m_breadth=1).run(p)
        b = FixedSketch(10, 1).run(p)
        assert a.cids == b.cids and a.tokens == b.tokens, p["task_id"]


def test_breadth_easy_pays_only_probe_and_lite_inputs():
    sk = [(True, [("A", True, True)] * 3) for _ in range(5)]
    p = make_pool(sk, [("A", True, True)] * 5)
    r = BreadthSketch().run(p)
    assert r.stop == "gate" and r.K_used == 1
    assert r.n_calls == 3 + 1 + 1 + 1 + 1  # flat x3, categories, sketch, fill, first input call
    assert r.chosen in ("f0", "f1", "f2")


def test_breadth_disagreement_escalates_to_full_breadth():
    sk = [(True, [("B", True, True)] * 3)] + [(True, [(f"C{k}", True, False)] * 3) for k in range(1, 5)]
    p = make_pool(sk, [("A", True, False)] * 5)
    r = BreadthSketch(K_max=5).run(p)
    assert r.stop == "breadth-done" and r.K_used == 5
    assert all(v == 1 for v in r.fills.values())
    # inputs paid exactly once (4 calls) + flat x3 + categories + 5 sketches + 5 fills
    assert r.n_calls == 4 + 3 + 1 + 5 + 5
    assert any(c.startswith("s0") for c in r.cids)  # coverage: the correct sketch is in the pool


def test_breadth_flat_failing_examples_skips_probe():
    sk = [(True, [("B", True, True)] * 3) for _ in range(3)]
    p = make_pool(sk, [("A", False, False)] * 5)
    r = BreadthSketch(K_max=3).run(p)
    assert r.stop == "breadth-done" and r.K_used == 3


def test_breadth_budget_and_refill():
    sk = [(True, [("X", False, False), ("Y", True, True), ("Y", True, True)]) for _ in range(6)]
    p = make_pool(sk, [("A", False, False)] * 5)
    r = BreadthSketch(K_max=6, refill=2, budget=12).run(p)
    assert r.n_cands <= 12
    assert any(n >= 2 for n in r.fills.values())  # failing sketches got refilled
    r2 = BreadthSketch(K_max=6, refill=2).run(p)
    assert all(n == 2 for n in r2.fills.values())  # stops refilling once a fill passes


def test_gated_flat_gates():
    p = make_pool([], [("A", True, True)] * 3 + [("B", True, False)] * 20)
    easy = GatedFlat(gate="examples").run(p)
    assert easy.stop == "gate" and easy.n_cands == 3 and easy.n_calls == 3  # no inputs paid
    lite = GatedFlat(gate="fp-lite").run(p)
    assert lite.stop == "gate" and lite.n_calls == 3 + 1  # + first input call only
    none = GatedFlat(gate="none", n_extra=10).run(p)
    flat = Flat(13).run(p)
    assert none.cids == flat.cids and none.tokens == flat.tokens
    p2 = make_pool([], [("A", True, True), ("C", True, True), ("A", True, True)] + [("A", True, True)] * 10)
    assert GatedFlat(gate="fp-lite").run(p2).stop == "done"  # disagreement on lite inputs -> escalate


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
