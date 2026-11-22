"""Parent-side wrappers around pool/_runner.py."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from typing import Dict, List

from .benchmarks import ROOT, Problem, safe_task_id

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runner.py")
EXPECTED_DIR = os.path.join(ROOT, "cache", "expected")
_expected_lock = threading.Lock()
_expected_task_locks: Dict[str, threading.Lock] = {}


# Memory guard.  RLIMIT_AS in _runner.py is NOT enforced on macOS, so a candidate
# that grows a list forever can eat all RAM and force a reboot.  The parent polls
# the child's RSS (via `ps`, works on macOS and Linux) and kills it above the cap.
# A memory kill is treated exactly like a timeout (killed=True).
MEM_LIMIT_MB = int(os.environ.get("EXEC_MEM_MB", "1536"))
_POLL_S = 0.1
mem_kills = 0


def _footprint_mb_darwin(pid: int):
    # macOS: RSS excludes compressed memory, so a runaway program can look small.
    # proc_pid_rusage(...).ri_phys_footprint is what Activity Monitor shows as "Memory".
    import ctypes, ctypes.util

    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 32)()
    if _libproc.proc_pid_rusage(int(pid), 0, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V0
        return None
    # layout: uuid (16 B = 2 x u64), user, system, pkg_idle, interrupt, pageins, wired, resident, phys_footprint
    return buf[2 + 7] / (1024.0 * 1024.0)


_libproc = None


def _rss_mb(pid: int) -> float:
    if sys.platform == "darwin":
        try:
            v = _footprint_mb_darwin(pid)
            if v is not None:
                return v
        except Exception:
            pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip() or 0) / 1024.0
    except Exception:
        return 0.0


def _kill_group(proc: subprocess.Popen) -> None:
    import signal as _sig

    try:
        os.killpg(proc.pid, _sig.SIGKILL)  # child runs in its own session -> kills grandchildren too
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _wait_with_mem_guard(proc: subprocess.Popen, wall_timeout: float) -> bool:
    global mem_kills
    import time

    deadline = time.monotonic() + wall_timeout
    while True:
        try:
            proc.wait(timeout=_POLL_S)
            return False
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline or _rss_mb(proc.pid) > MEM_LIMIT_MB:
            if time.monotonic() <= deadline:
                mem_kills += 1
            _kill_group(proc)
            proc.wait()
            return True


def _run(mode: str, spec: dict, wall_timeout: float, out_suffix: str = ".json"):
    with tempfile.TemporaryDirectory(prefix="pool_") as d:
        spec_path = os.path.join(d, "spec.json")
        out_path = os.path.join(d, "out" + out_suffix)
        with open(spec_path, "w") as f:
            json.dump(spec, f)
        env = dict(os.environ, PYTHONHASHSEED="0")  # deterministic set/str ordering
        proc = subprocess.Popen(
            [sys.executable, "-s", RUNNER, mode, spec_path, out_path],  # not -I: it would drop PYTHONHASHSEED
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=d,
            start_new_session=True,
        )
        killed = _wait_with_mem_guard(proc, wall_timeout)
        if not os.path.exists(out_path):
            return None, killed
        if out_suffix == ".pkl":
            with open(out_path, "rb") as f:
                return pickle.load(f), killed
        with open(out_path) as f:
            return json.load(f), killed


# ---------------------------------------------------------------------------
# probe: compile + public examples + fingerprint
# ---------------------------------------------------------------------------


def probe(code: str, problem: Problem, inputs: List[str], timeout: float = 5.0) -> dict:
    spec = {
        "code": code,
        "prefix": problem.exec_prefix,
        "entry_point": problem.entry_point,
        "examples": problem.examples,
        "inputs": inputs,
        "timeout": timeout,
        "lcb": problem.benchmark == "lcb",
    }
    n = len(problem.examples) + len(inputs)
    # generous global guard; per-input timeouts are enforced inside
    wall = 30 + 2.0 * n
    res, killed = _run("probe", spec, wall)
    if res is None:
        res = {"compiles": False, "load_error": "TimeoutError" if killed else "RunnerCrash",
               "examples": [], "fp": []}
    loaded = res.get("load_error") is None
    ex = res.get("examples", [])
    fp = res.get("fp", [])
    if loaded:
        # runner killed mid-way (e.g. C-level hang): pad as timeouts
        ex = ex + ["E:TimeoutError"] * (len(problem.examples) - len(ex))
        fp = fp + ["E:TimeoutError"] * (len(inputs) - len(fp))
    else:
        ex = ["E:" + res["load_error"]] * len(problem.examples)
        fp = ["E:" + res["load_error"]] * len(inputs)
    return {
        "compiles": bool(res.get("compiles")),
        "load_error": res.get("load_error"),
        "examples": ex,
        "examples_pass": loaded and all(e == "pass" for e in ex),
        "fp": fp,
        "all_ok": loaded and all(x.startswith("ok:") for x in fp),
    }


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------


def _expected_paths(problem: Problem):
    d = os.path.join(EXPECTED_DIR, problem.benchmark)
    tid = safe_task_id(problem.task_id)
    return os.path.join(d, tid + ".inputs.pkl"), os.path.join(d, tid + ".expected.pkl")


def ensure_expected(problem: Problem, timeout: float = 20.0) -> None:
    """Run the canonical solution once on all EvalPlus inputs and cache outputs."""
    inp_path, exp_path = _expected_paths(problem)
    with _expected_lock:  # one lock per problem: compute the reference outputs once
        lock = _expected_task_locks.setdefault(exp_path, threading.Lock())
    with lock:
        if os.path.exists(exp_path) and os.path.exists(inp_path):
            return
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)
        inputs = list(problem.base_input) + list(problem.plus_input)
        if problem.expected_output:  # LiveCodeBench: reference outputs come with the data
            assert len(problem.expected_output) == len(inputs), problem.task_id
            ref = {"outs": [("ok", o) for o in problem.expected_output], "times": [1.0] * len(inputs)}
            with open(inp_path + ".tmp", "wb") as f:
                pickle.dump(inputs, f)
            os.replace(inp_path + ".tmp", inp_path)
            with open(exp_path + ".tmp", "wb") as f:
                pickle.dump(ref, f)
            os.replace(exp_path + ".tmp", exp_path)
            return
        spec = {
            "code": problem.canonical_solution,
            "entry_point": problem.entry_point,
            "inputs": inputs,
            "timeout": timeout,
        }
        outs, killed = _run("expected", spec, wall_timeout=1800, out_suffix=".pkl")
        if outs is None:
            raise RuntimeError(f"canonical solution failed for {problem.task_id}")
        with open(inp_path + ".tmp", "wb") as f:
            pickle.dump(inputs, f)
        os.replace(inp_path + ".tmp", inp_path)
        with open(exp_path + ".tmp", "wb") as f:
            pickle.dump(outs, f)
        os.replace(exp_path + ".tmp", exp_path)


def ground_truth(code: str, problem: Problem, timeout: float = 5.0) -> dict:
    ensure_expected(problem)
    inp_path, exp_path = _expected_paths(problem)
    spec = {
        "code": code,
        "prefix": problem.exec_prefix,
        "entry_point": problem.entry_point,
        "test_orig": problem.test_orig,
        "orig_timeout": 10.0,
        "inputs_path": inp_path,
        "expected_path": exp_path,
        "n_base": len(problem.base_input),
        "atol": problem.atol,
        "timeout": timeout,
        "special": "find_zero" if problem.entry_point == "find_zero" else None,
        "lcb": problem.benchmark == "lcb",
    }
    res, killed = _run("gt", spec, wall_timeout=600)
    if res is None:
        res = {"gt_orig": False, "gt_base": False, "gt_plus": False}
    return {k: bool(res.get(k)) for k in ("gt_orig", "gt_base", "gt_plus")}
