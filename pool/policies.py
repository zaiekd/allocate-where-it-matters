"""Inference-time policies replayed offline on a candidate pool.

Every policy talks to a `Session`, which hands out pool items in generation
order and records which LLM calls were paid for:

    s.sketch(k)  -> generate sketch k   (pays categories once + sketch call)
    s.fill(k)    -> next fill of sketch k (pays that fill call)
    s.flat()     -> next flat sample     (pays that call)
    s.greedy()   -> greedy sample        (pays that call)

Fingerprints need the generated test inputs, so the first time a policy looks
at fingerprints the input-generation calls are paid as well (`s.need_inputs()`).
At the end, `s.finish(selector)` picks the answer and returns cost + trace.

Because the pool was generated exactly in this order (see builder.py), the
replay is exact for the default sketch order.  `sketch_order` can permute the
sketches (for variance estimates); fills inside a sketch always stay in order
because fill m may depend on fill m-1.

Policies
--------
  FixedSketch(K, M)       paper's Sketch-and-Verify
  Flat(N)                 paper's flat sampling (+ semantic voting); N=1 = greedy
  AdaptiveFlat(...)       baseline: flat sampling + the same early-stopping rule
  AdaptiveSketch(...)     ours: difficulty gate + sequential sketches with early
                          stop + successive-halving fill allocation; each of the
                          three mechanisms can be switched off for ablations.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .replay import tokens_of

# ---------------------------------------------------------------------------
# candidate helpers
# ---------------------------------------------------------------------------


def _cand(pool: dict, cid: str) -> dict:
    return pool["candidates"][cid]


def loaded(pool: dict, cid: str) -> bool:
    c = _cand(pool, cid)
    return bool(c["compiles"]) and c.get("load_error") is None


def ex_pass(pool: dict, cid: str) -> bool:
    """Passes all public examples (vacuously true when the problem has none)."""
    return loaded(pool, cid) and bool(_cand(pool, cid)["examples_pass"])


def fp(pool: dict, cid: str) -> tuple:
    return tuple(_cand(pool, cid)["fp"])


def valid_set(pool: dict, cids: Sequence[str]) -> List[str]:
    """Candidates that pass the public examples; if none do (e.g. the docstring
    example itself is wrong), fall back to everything that loads."""
    v = [c for c in cids if ex_pass(pool, c)]
    return v if v else [c for c in cids if loaded(pool, c)]


def clusters(pool: dict, cids: Sequence[str]) -> List[Tuple[tuple, List[str]]]:
    """Fingerprint clusters over `cids`, largest first (stable), all-ok first."""
    groups: "OrderedDict[tuple, List[str]]" = OrderedDict()
    for c in cids:
        groups.setdefault(fp(pool, c), []).append(c)
    out = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    ok = [kv for kv in out if all(x.startswith("ok:") for x in kv[0])]
    rest = [kv for kv in out if not all(x.startswith("ok:") for x in kv[0])]
    return ok + rest


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------


def select_sv(pool: dict, cids: Sequence[str]) -> Optional[str]:
    """Paper's selector (sketch-and-verify/src/select.py, strategy B):
    all compiling candidates, all-ok clusters preferred, largest cluster,
    shortest program.  Ignores public examples."""
    comp = [c for c in cids if _cand(pool, c)["compiles"]]
    if not comp:
        return None
    best = clusters(pool, comp)[0][1]
    return min(best, key=lambda c: _cand(pool, c)["code_len"])


def select_sv_ex(pool: dict, cids: Sequence[str]) -> Optional[str]:
    """Same, but restricted to candidates that pass the public examples
    (falls back to select_sv if none do)."""
    v = [c for c in cids if ex_pass(pool, c)]
    if not v:
        return select_sv(pool, cids)
    best = clusters(pool, v)[0][1]
    return min(best, key=lambda c: _cand(pool, c)["code_len"])


SELECTORS = {"sv": select_sv, "sv_ex": select_sv_ex}


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


@dataclass
class Result:
    chosen: Optional[str]
    cids: List[str]
    tokens: int
    usd: float
    n_calls: int
    K_used: int  # sketches generated
    fills: Dict[int, int]  # fills per sketch (sketch index in pool order)
    n_flat: int
    stop: str  # why the policy stopped
    trace: List[str] = field(default_factory=list)

    @property
    def n_cands(self) -> int:
        return len(self.cids)


class Session:
    def __init__(self, pool: dict, sketch_order: Optional[Sequence[int]] = None) -> None:
        self.pool = pool
        n = len(pool["sketches"])
        self.order = list(sketch_order) if sketch_order is not None else list(range(n))
        self.calls: List[dict] = []
        self.cids: List[str] = []
        self.paid_inputs = False
        self.paid_inputs_lite = False
        self.paid_categories = False
        self.sketches_done: List[int] = []  # pool indices, in generation order
        self.fill_ptr: Dict[int, int] = {}
        self.flat_ptr = 0
        self.used_greedy = False
        self.trace: List[str] = []

    # -- generation -----------------------------------------------------------
    @property
    def K_max(self) -> int:
        return len(self.order)

    def need_inputs(self) -> None:
        if not self.paid_inputs:
            calls = self.pool["test_inputs"]["calls"]
            # if the lite part (first call) was already paid, pay only the rest
            self.calls += calls[1:] if self.paid_inputs_lite else calls
            self.paid_inputs = True
            self.paid_inputs_lite = True

    def lite_mask(self) -> Optional[List[int]]:
        """Indices of the inputs produced by the FIRST input-generation call
        (one representative per input category: inputs[0], [5], [10], ...).
        None when the pool's layout does not allow an exact prefix replay
        (fallback generation or a category with fewer variations)."""
        t = self.pool["test_inputs"]
        n_cat = len(t["calls"]) - 1
        if t.get("fallback") or n_cat <= 0 or len(t["inputs"]) % n_cat:
            return None
        per = len(t["inputs"]) // n_cat
        return list(range(0, len(t["inputs"]), per))

    def need_inputs_lite(self) -> Optional[List[int]]:
        """Pay only the first input-generation call; returns the usable input
        mask, or pays everything and returns None (= use all inputs)."""
        if self.paid_inputs:
            return None
        mask = self.lite_mask()
        if mask is None:
            self.need_inputs()
            return None
        if not self.paid_inputs_lite:
            self.calls.append(self.pool["test_inputs"]["calls"][0])
            self.paid_inputs_lite = True
        return mask

    def sketch(self, j: int) -> Optional[int]:
        """Generate the j-th sketch (in session order). Returns its pool index
        if valid (fillable), else None.  Pays the call either way."""
        if j >= self.K_max:
            return None
        if not self.paid_categories:
            self.calls.append(self.pool["categories"]["call"])
            self.paid_categories = True
        k = self.order[j]
        s = self.pool["sketches"][k]
        self.calls.append(s["call"])
        self.sketches_done.append(k)
        self.fill_ptr.setdefault(k, 0)
        return k if s["valid"] and s["fills"] else None

    def fill(self, k: int) -> Optional[str]:
        fills = self.pool["sketches"][k]["fills"]
        m = self.fill_ptr.get(k, 0)
        if m >= len(fills):
            return None
        f = fills[m]
        self.fill_ptr[k] = m + 1
        self.calls.append(f["call"])
        self.cids.append(f["cid"])
        return f["cid"]

    def fills_of(self, k: int) -> List[str]:
        return [f["cid"] for f in self.pool["sketches"][k]["fills"][: self.fill_ptr.get(k, 0)]]

    def flat(self) -> Optional[str]:
        if self.flat_ptr >= len(self.pool["flat"]):
            return None
        f = self.pool["flat"][self.flat_ptr]
        self.flat_ptr += 1
        self.calls.append(f["call"])
        self.cids.append(f["cid"])
        return f["cid"]

    def greedy(self) -> str:
        if not self.used_greedy:
            self.calls.append(self.pool["greedy"]["call"])
            self.cids.append("g")
            self.used_greedy = True
        return "g"

    # -- finish ----------------------------------------------------------------
    def finish(self, selector: str, stop: str, chosen: Optional[str] = None) -> Result:
        if chosen is None:
            chosen = SELECTORS[selector](self.pool, self.cids) if self.cids else None
        t = tokens_of(self.pool, self.calls)
        return Result(
            chosen=chosen, cids=list(self.cids), tokens=t["tokens"], usd=t["usd"],
            n_calls=t["n_calls"], K_used=len(self.sketches_done),
            fills={k: self.fill_ptr.get(k, 0) for k in self.sketches_done},
            n_flat=self.flat_ptr, stop=stop, trace=self.trace,
        )


# ---------------------------------------------------------------------------
# stopping rule shared by AdaptiveFlat and AdaptiveSketch
# ---------------------------------------------------------------------------


@dataclass
class StopRule:
    """Stop when the top cluster (over example-passing candidates) is
    large enough, dominant enough, and — for sketches — supported by enough
    distinct strategies."""

    min_size: int = 3  # top cluster has at least this many members
    tau: float = 0.6  # ... and at least this fraction of valid candidates
    min_support: int = 2  # ... coming from >= this many distinct sketches (sketch only)

    def check(self, pool: dict, cids: Sequence[str], origin: Optional[Dict[str, int]] = None) -> Tuple[bool, str]:
        v = valid_set(pool, cids)
        if not v:
            return False, "no valid candidates"
        top_fp, top = clusters(pool, v)[0]
        frac = len(top) / len(v)
        support = len({origin[c] for c in top}) if origin else None
        info = f"top={len(top)}/{len(v)} ({frac:.2f})" + (f" support={support}" if origin else "")
        ok = len(top) >= self.min_size and frac >= self.tau
        if origin is not None:
            ok = ok and support >= self.min_support
        return ok, info


# ---------------------------------------------------------------------------
# baselines from the paper
# ---------------------------------------------------------------------------


@dataclass
class FixedSketch:
    K: int
    M: int
    selector: str = "sv"

    @property
    def name(self) -> str:
        return f"sketch K={self.K},M={self.M}"

    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool, sketch_order)
        for j in range(min(self.K, s.K_max)):
            k = s.sketch(j)
            if k is None:
                continue
            for _ in range(self.M):
                s.fill(k)
        s.need_inputs()
        return s.finish(self.selector, "fixed")


@dataclass
class Flat:
    N: int
    selector: str = "sv"

    @property
    def name(self) -> str:
        return f"flat N={self.N}"

    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool)
        if self.N == 1:  # greedy, no selection, no test inputs
            return s.finish(self.selector, "fixed", chosen=s.greedy())
        for _ in range(self.N):
            s.flat()
        s.need_inputs()
        return s.finish(self.selector, "fixed")


# ---------------------------------------------------------------------------
# adaptive baseline: flat sampling with early stopping
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveFlat:
    """Draw flat samples in batches; stop as soon as the StopRule fires
    (min_support is ignored — flat samples have no strategy structure)."""

    first: int = 3
    batch: int = 2
    N_max: int = 100
    rule: StopRule = field(default_factory=StopRule)
    gate_signal: str = "fp"  # "examples": stop right after the first batch if all
    #                          samples pass the public examples (no test inputs paid)
    selector: str = "sv_ex"

    @property
    def name(self) -> str:
        r = self.rule
        g = ",gate=ex" if self.gate_signal == "examples" else ""
        return f"adaptive-flat(first={self.first},b={self.batch},Nmax={self.N_max},size>={r.min_size},tau={r.tau}{g})"

    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool)
        first = [s.flat() for _ in range(self.first)]
        if self.gate_signal == "examples" and pool["examples"]:
            if all(c and ex_pass(pool, c) for c in first):
                s.trace.append("gate EASY (examples only)")
                return s.finish(self.selector, "gate",
                                chosen=min(first, key=lambda c: _cand(pool, c)["code_len"]))
        s.need_inputs()
        while True:
            ok, info = self.rule.check(pool, s.cids)
            s.trace.append(f"n={s.flat_ptr} {info}")
            if ok:
                return s.finish(self.selector, "converged")
            if s.flat_ptr >= self.N_max or s.flat_ptr >= len(pool["flat"]):
                return s.finish(self.selector, "budget")
            for _ in range(self.batch):
                s.flat()


# ---------------------------------------------------------------------------
# ours: adaptive K and M
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSketch:
    """Adaptive Sketch-and-Verify.

    Phase 0 — difficulty gate (`gate`):
        draw `gate_n` cheap samples (flat samples by default, or fills of the
        first sketch with gate_mode="sketch").  If they all pass the public
        examples, all run cleanly, and share ONE fingerprint, the problem is
        "easy": return immediately (K = 0 or 1).
    Phase 1 — sequential sketches with early stop (`early_stop`):
        generate sketches one at a time, `m0` fills each.  A sketch whose
        fills all fail (don't load / fail examples) is pruned.  After each
        sketch check the StopRule with cross-strategy support; stop when it
        fires.  Without early stop all K_max sketches are generated.
    Phase 2 — successive halving (`halving`):
        if Phase 1 ended without convergence, rank the surviving sketches by
        (example-pass rate, members in the current top cluster), keep the top
        half, double their fills (2 -> 4 -> 8 ... up to M_max), re-check the
        StopRule after every round.  Without halving, survivors get uniform
        extra fills up to `M_uniform`.
    `budget` caps the total number of fills (+ gate samples).
    """

    K_max: int = 10
    M_max: int = 10
    m0: int = 2
    gate: bool = True
    gate_mode: str = "flat"  # "flat" | "sketch"
    gate_n: int = 3
    gate_signal: str = "fp"  # "fp": examples + identical fingerprints (needs test inputs)
    #                          "examples": examples only — test inputs are generated
    #                          lazily, so easy problems never pay for them
    early_stop: bool = True
    halving: bool = True
    M_uniform: int = 5
    budget: int = 100
    rule: StopRule = field(default_factory=StopRule)
    selector: str = "sv_ex"

    @property
    def name(self) -> str:
        r = self.rule
        parts = [
            (f"gate={self.gate_mode}{self.gate_n}" + ("/ex" if self.gate_signal == "examples" else ""))
            if self.gate else "gate=off",
            "es=on" if self.early_stop else "es=off",
            "sh=on" if self.halving else f"sh=off(M={self.M_uniform})",
            f"m0={self.m0}", f"size>={r.min_size}", f"tau={r.tau}", f"s>={r.min_support}",
            f"Kmax={self.K_max}", f"B={self.budget}",
        ]
        return "adaptive-sketch(" + ",".join(parts) + ")"

    # -- helpers ---------------------------------------------------------------
    def _origin(self, s: Session) -> Dict[str, int]:
        return {c: k for k in s.sketches_done for c in s.fills_of(k)}

    def _sketch_cids(self, s: Session) -> List[str]:
        return [c for k in s.sketches_done for c in s.fills_of(k)]

    def _spent(self, s: Session) -> int:
        return len(s.cids)

    def _check(self, s: Session) -> Tuple[bool, str]:
        return self.rule.check(s.pool, self._sketch_cids(s), self._origin(s))

    # -- main ------------------------------------------------------------------
    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool, sketch_order)
        use_ex_gate = self.gate and self.gate_signal == "examples" and bool(pool["examples"])
        if not use_ex_gate:
            s.need_inputs()
        K_max = min(self.K_max, s.K_max)
        j = 0  # next sketch position
        alive: List[int] = []

        # ---- Phase 0: difficulty gate ---------------------------------------
        if self.gate:
            if self.gate_mode == "flat":
                g = [s.flat() for _ in range(self.gate_n)]
                g = [c for c in g if c]
            else:
                k = None
                while k is None and j < K_max:
                    k = s.sketch(j)
                    j += 1
                g = [s.fill(k) for _ in range(self.gate_n)] if k is not None else []
                g = [c for c in g if c]
                if k is not None:
                    alive.append(k)
            if use_ex_gate:
                easy = len(g) == self.gate_n and all(ex_pass(pool, c) for c in g)
                s.trace.append(f"gate {'EASY' if easy else 'hard'} (examples only)")
                if easy:  # no fingerprints available: return the shortest sample
                    return s.finish(self.selector, "gate",
                                    chosen=min(g, key=lambda c: _cand(pool, c)["code_len"]))
                s.need_inputs()
            else:
                easy = (
                    len(g) == self.gate_n
                    and all(ex_pass(pool, c) for c in g)
                    and len({fp(pool, c) for c in g}) == 1
                    # (no all_ok requirement: generated inputs are sometimes invalid
                    #  and make every correct program raise the same exception)
                )
                s.trace.append(f"gate {'EASY' if easy else 'hard'} ({len({fp(pool, c) for c in g})} fps)")
                if easy:
                    return s.finish(self.selector, "gate")
            gate_cids = list(s.cids)
        else:
            gate_cids = []

        # ---- Phase 1: sequential sketches -------------------------------------
        while j < K_max and self._spent(s) + self.m0 <= self.budget:
            k = s.sketch(j)
            j += 1
            if k is None:
                s.trace.append(f"sketch@{j - 1} invalid")
                continue
            if k not in alive:
                for _ in range(self.m0 - len(s.fills_of(k))):
                    s.fill(k)
                alive.append(k)
            if not any(ex_pass(pool, c) for c in s.fills_of(k)):
                alive.remove(k)
                s.trace.append(f"sketch {k} pruned (all fills fail)")
            if self.early_stop:
                ok, info = self._check(s)
                s.trace.append(f"K={len(s.sketches_done)} {info}")
                if ok:
                    return self._final(s, gate_cids, "early-stop")

        # ---- Phase 2: allocate remaining fills --------------------------------
        if self.halving:
            m_target = self.m0
            while alive and m_target < self.M_max:
                m_target = min(2 * m_target, self.M_max)
                ranked = self._rank(s, alive)
                keep = ranked[: max(1, (len(ranked) + 1) // 2)] if len(ranked) > 1 else ranked
                s.trace.append(f"halving: keep {keep} -> m={m_target}")
                alive = keep
                exhausted = True
                for k in alive:
                    while len(s.fills_of(k)) < m_target and self._spent(s) < self.budget:
                        if s.fill(k) is None:
                            break
                        exhausted = False
                ok, info = self._check(s)
                s.trace.append(info)
                if ok and self.early_stop:
                    return self._final(s, gate_cids, "halving-converged")
                if self._spent(s) >= self.budget or exhausted:
                    break
                if len(alive) == 1 and len(s.fills_of(alive[0])) >= self.M_max:
                    break
            return self._final(s, gate_cids, "halving-done")
        else:
            for k in alive:
                while len(s.fills_of(k)) < self.M_uniform and self._spent(s) < self.budget:
                    if s.fill(k) is None:
                        break
            return self._final(s, gate_cids, "uniform-done")

    def _rank(self, s: Session, alive: List[int]) -> List[int]:
        pool = s.pool
        v = valid_set(pool, self._sketch_cids(s))
        top = set(clusters(pool, v)[0][1]) if v else set()

        def score(k):
            fs = s.fills_of(k)
            pr = sum(ex_pass(pool, c) for c in fs) / max(len(fs), 1)
            return (pr, sum(c in top for c in fs))

        return sorted(alive, key=score, reverse=True)

    def _final(self, s: Session, gate_cids: List[str], stop: str) -> Result:
        # gate samples stay in the voting pool: they were paid for
        return s.finish(self.selector, stop)


# ---------------------------------------------------------------------------
# v2: breadth-first adaptive sketching
# ---------------------------------------------------------------------------


def _key(pool: dict, cid: str, mask: Optional[Sequence[int]]) -> tuple:
    f = _cand(pool, cid)["fp"]
    return tuple(f[i] for i in mask) if mask is not None else tuple(f)


@dataclass
class BreadthSketch:
    """Adaptive Sketch-and-Verify, breadth first (v2).

    Motivation (HumanEval+ coverage analysis): coverage grows with the number
    of distinct strategies K, almost not with fills per sketch M; and the
    examples-only gate lets most hard problems through because wrong flat
    programs pass the public examples.

    Phase 0 — gate:
      "examples": `gate_n` flat samples all pass the public examples -> easy.
      "disagree": if the flat samples fail an example -> hard.  Otherwise
          generate ONE sketch with `probe_fills` fills and compare execution
          fingerprints (on the cheap "lite" inputs: one per input category,
          ~18% of the input-generation cost).  If an example-passing sketch
          fill behaves differently from the flat samples (or the flat samples
          disagree among themselves) -> hard; else easy.
      "off": every problem is hard (no flat samples).
    Phase 1 — breadth: generate sketches in order with `m_breadth` fill(s)
      each, up to K_max.  Optional early stop: the top fingerprint cluster
      (example-passing candidates, flat + sketch) is supported by >=
      `stop_support` distinct strategies (the flat samples count as one) and
      holds >= `stop_tau` of the valid candidates.
    Phase 2 — refill (optional): sketches whose fills all fail the examples
      get up to `refill` extra fills (an implementation slip, not a wrong
      strategy, is the common failure).
    """

    gate: str = "disagree"  # "disagree" | "examples" | "off"
    gate_n: int = 3
    probe_fills: int = 1
    lite_inputs: bool = True
    K_max: int = 10
    m_breadth: int = 1
    early_stop: bool = False
    stop_support: int = 3
    stop_tau: float = 0.5
    refill: int = 0
    budget: int = 100
    selector: str = "sv_ex"

    @property
    def name(self) -> str:
        parts = [f"gate={self.gate}" + (f"(n={self.gate_n},p={self.probe_fills}"
                 + (",lite" if self.lite_inputs else "") + ")" if self.gate == "disagree" else ""),
                 f"Kmax={self.K_max}", f"m={self.m_breadth}"]
        if self.early_stop:
            parts.append(f"es(s>={self.stop_support},tau={self.stop_tau})")
        if self.refill:
            parts.append(f"refill={self.refill}")
        return "breadth-sketch(" + ",".join(parts) + ")"

    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool, sketch_order)
        K_max = min(self.K_max, s.K_max)
        has_ex = bool(pool["examples"])
        origin: Dict[str, object] = {}
        j = 0
        probe_k: Optional[int] = None

        # ---- Phase 0 ------------------------------------------------------
        if self.gate in ("examples", "disagree"):
            g = [c for c in (s.flat() for _ in range(self.gate_n)) if c]
            for c in g:
                origin[c] = "flat"
            flat_ok = len(g) == self.gate_n and all(ex_pass(pool, c) for c in g)
            if self.gate == "examples":
                if has_ex and flat_ok:
                    s.trace.append("gate EASY (examples)")
                    return s.finish(self.selector, "gate",
                                    chosen=min(g, key=lambda c: _cand(pool, c)["code_len"]))
                s.trace.append("gate hard (examples)")
            else:
                if has_ex and not flat_ok:
                    s.trace.append("gate hard (flat fails examples)")
                else:
                    while probe_k is None and j < K_max:
                        probe_k = s.sketch(j)
                        j += 1
                    probe = []
                    if probe_k is not None:
                        probe = [c for c in (s.fill(probe_k) for _ in range(self.probe_fills)) if c]
                        for c in probe:
                            origin[c] = probe_k
                    mask = s.need_inputs_lite() if self.lite_inputs else (s.need_inputs() or None)
                    fk = {_key(pool, c, mask) for c in g if ex_pass(pool, c)}
                    pk = [_key(pool, c, mask) for c in probe if ex_pass(pool, c)]
                    disagree = len(fk) > 1 or any(k not in fk for k in pk)
                    s.trace.append(f"gate {'hard' if disagree else 'EASY'} (disagree: flat fps={len(fk)}, "
                                   f"probe passing={len(pk)}, new={sum(k not in fk for k in pk)})")
                    if not disagree:
                        # easy: answer from the flat samples (largest masked cluster, shortest)
                        groups: Dict[tuple, List[str]] = {}
                        for c in g:
                            if ex_pass(pool, c):
                                groups.setdefault(_key(pool, c, mask), []).append(c)
                        best = max(groups.values(), key=len) if groups else g
                        return s.finish(self.selector, "gate",
                                        chosen=min(best, key=lambda c: _cand(pool, c)["code_len"]))

        # ---- hard: full inputs, breadth over strategies ---------------------
        s.need_inputs()
        if probe_k is not None:
            while len(s.fills_of(probe_k)) < self.m_breadth and len(s.cids) < self.budget:
                c = s.fill(probe_k)
                if c is None:
                    break
                origin[c] = probe_k
        stop = "breadth-done"
        while j < K_max and len(s.cids) + self.m_breadth <= self.budget:
            k = s.sketch(j)
            j += 1
            if k is None:
                continue
            for _ in range(self.m_breadth):
                c = s.fill(k)
                if c is None:
                    break
                origin[c] = k
            if self.early_stop and self._converged(pool, s, origin):
                stop = "early-stop"
                break

        # ---- Phase 2: refill sketches whose fills all failed ----------------
        if self.refill and stop != "early-stop":
            for k in list(s.sketches_done):
                fs = s.fills_of(k)
                if not fs or any(ex_pass(pool, c) for c in fs):
                    continue
                for _ in range(self.refill):
                    if len(s.cids) >= self.budget:
                        break
                    c = s.fill(k)
                    if c is None:
                        break
                    origin[c] = k
                    if ex_pass(pool, c):
                        break
        return s.finish(self.selector, stop)

    def _converged(self, pool, s, origin) -> bool:
        v = [c for c in s.cids if ex_pass(pool, c)]
        if not v:
            return False
        top = clusters(pool, v)[0][1]
        support = len({origin.get(c) for c in top})
        ok = support >= self.stop_support and len(top) / len(v) >= self.stop_tau
        s.trace.append(f"K={len(s.sketches_done)} top={len(top)}/{len(v)} support={support}")
        return ok


@dataclass
class GatedFlat:
    """Control for BreadthSketch: the same kind of gate, but a hard problem gets
    `n_extra` more FLAT samples instead of sketches.  Separates the effect
    of adaptivity (+ example filtering) from the effect of sketching.

    gate:
      "examples" — all `gate_n` samples pass the public examples -> easy
      "fp-lite"  — they pass the examples AND agree on the cheap "lite" inputs
                   (one per input category, ~18% of the input cost) -> easy
      "none"     — no gate: every problem gets the extra samples
    """

    gate: str = "examples"  # "examples" | "fp-lite" | "none"
    gate_n: int = 3
    n_extra: int = 10
    selector: str = "sv_ex"

    @property
    def name(self) -> str:
        return f"gated-flat(gate={self.gate},n={self.gate_n},+{self.n_extra},sel={self.selector})"

    def run(self, pool: dict, sketch_order=None) -> Result:
        s = Session(pool)
        g = [c for c in (s.flat() for _ in range(self.gate_n)) if c]
        passed = bool(pool["examples"]) and len(g) == self.gate_n and all(ex_pass(pool, c) for c in g)
        if self.gate == "examples" and passed:
            s.trace.append("gate EASY (examples)")
            return s.finish(self.selector, "gate", chosen=min(g, key=lambda c: _cand(pool, c)["code_len"]))
        if self.gate == "fp-lite" and passed:
            mask = s.need_inputs_lite()
            if len({_key(pool, c, mask) for c in g}) == 1:
                s.trace.append("gate EASY (examples + lite fingerprints agree)")
                return s.finish(self.selector, "gate", chosen=min(g, key=lambda c: _cand(pool, c)["code_len"]))
        s.need_inputs()
        for _ in range(self.n_extra):
            if s.flat() is None:
                break
        return s.finish(self.selector, "done")

# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------


def run_policy(policy, pools: List[dict], sketch_order_fn=None) -> List[Result]:
    out = []
    for p in pools:
        order = sketch_order_fn(p) if sketch_order_fn else None
        out.append(policy.run(p, order))
    return out
