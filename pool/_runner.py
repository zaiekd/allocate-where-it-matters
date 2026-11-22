"""Standalone sandbox runner (executed in a fresh `python -s` subprocess, PYTHONHASHSEED=0).

    python -s _runner.py <mode> <spec.json> <out.json>

modes
  probe     compile + public examples + fingerprint on generated inputs
  expected  run the canonical solution on EvalPlus inputs -> pickle of outputs
  gt        ground truth: original tests + EvalPlus base/plus inputs

All candidate stdout/stderr is discarded.  Per-input wall-clock timeouts use
SIGALRM; the parent additionally enforces a global subprocess timeout.
Fingerprint semantics match sketch-research/core/sandbox.py exactly:
  ok    -> repr(normalize(result))  (floats rounded to 6 decimals)
  error -> exception type name; timeout -> "TimeoutError".
"""

import copy
import hashlib
import io
import json
import math
import os
import pickle
import signal
import sys


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def _with_timeout(seconds, fn, *a):
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*a)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


try:  # best-effort memory cap (not supported on every OS)
    import resource

    _lim = 4 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (_lim, _lim))
except Exception:
    pass

sys.setrecursionlimit(10000)


def _self_mem_mb():
    if sys.platform == "darwin":
        try:  # phys_footprint includes compressed memory (RSS does not)
            import ctypes, ctypes.util
            lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            buf = (ctypes.c_uint64 * 32)()
            if lib.proc_pid_rusage(os.getpid(), 0, ctypes.byref(buf)) == 0:
                return buf[9] / (1024.0 * 1024.0)
        except Exception:
            pass
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _mem_watchdog(limit_mb=int(os.environ.get("EXEC_MEM_MB", "1536"))):
    # second line of defence (the parent also polls this process's memory)
    import threading, time

    def loop():
        while True:
            time.sleep(0.05)
            try:
                if _self_mem_mb() > limit_mb:
                    os._exit(137)
            except Exception:
                return

    threading.Thread(target=loop, daemon=True).start()


def _no_subprocesses():
    # like human-eval's reliability_guard: candidates may not spawn processes
    def _blocked(*a, **k):
        raise OSError("process creation disabled in sandbox")

    for name in ("fork", "forkpty", "system", "popen", "execv", "execve", "execvp", "execvpe",
                 "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                 "posix_spawn", "posix_spawnp", "killpg", "kill"):
        if hasattr(os, name):
            setattr(os, name, _blocked)
    import subprocess as _sp
    _sp.Popen = _blocked
    _sp.run = _sp.call = _sp.check_call = _sp.check_output = _blocked
    try:
        import multiprocessing as _mp
        _mp.Process.start = _blocked
        _mp.Pool = _blocked
    except Exception:
        pass


try:
    _mem_watchdog()
except Exception:
    pass

try:
    _no_subprocesses()
except Exception:
    pass


def _normalize(v):  # copied from core/sandbox.py runner template
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return round(v, 6)
    if isinstance(v, (list, tuple)):
        normalized = [_normalize(x) for x in v]
        return type(v)(normalized)
    if isinstance(v, dict):
        return {k: _normalize(val) for k, val in v.items()}
    if isinstance(v, set):
        return sorted([_normalize(x) for x in v])
    return v


# --- equality helpers -------------------------------------------------------


def _isclose(a, b, atol):
    try:
        return abs(a - b) <= atol + 1e-07 * abs(b)
    except Exception:
        return False


def _ex_eq(out, exp):
    """Equality for public docstring examples (lenient on floats)."""
    try:
        if out == exp:
            return True
    except Exception:
        pass
    if isinstance(exp, float) or isinstance(out, float):
        return _isclose(out, exp, 1e-6)
    if isinstance(exp, (list, tuple)) and isinstance(out, (list, tuple)) and len(out) == len(exp):
        return all(_ex_eq(o, e) for o, e in zip(out, exp))
    return False


def _is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    return False


def _evalplus_eq(out, exp, atol):
    """Mirrors evalplus.eval.unsafe_execute's comparison (numpy-free)."""
    try:
        exact = bool(out == exp)
    except Exception:
        exact = False
    if atol == 0 and _is_floats(exp):
        atol = 1e-6
    if not exact and atol != 0:
        if type(out) != type(exp):
            return False
        if isinstance(exp, (list, tuple)):
            if len(out) != len(exp):
                return False
            return all(_isclose(o, e, atol) for o, e in zip(out, exp))
        return _isclose(out, exp, atol)
    return exact


# --- helpers ----------------------------------------------------------------


def _fresh_globals():
    g = {"__name__": "__candidate__", "__builtins__": __builtins__}
    exec("import json, sys, math", g)
    g["__ex_eq"] = _ex_eq
    return g


def _load(code, timeout):
    g = _fresh_globals()
    _with_timeout(timeout, exec, compile(code, "<candidate>", "exec"), g)
    return g


def _resolve(g, ep):
    """Accept LeetCode-style answers: if `ep` is not a top-level function but a
    method of `class Solution`, expose it as a plain function."""
    if callable(g.get(ep)):
        return g[ep]
    cls = g.get("Solution")
    if isinstance(cls, type) and callable(getattr(cls, ep, None)):
        def _call(*a, **k):
            return getattr(cls(), ep)(*a, **k)

        g[ep] = _call
        return _call
    return None


def _to_lists(v):
    """LiveCodeBench compares JSON-like values: tuples count as lists."""
    if isinstance(v, (list, tuple)):
        return [_to_lists(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_lists(x) for k, x in v.items()}
    return v


def _err_name(e):
    if isinstance(e, _Timeout):
        return "TimeoutError"
    return type(e).__name__


def _hash(s):
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


# --- modes ------------------------------------------------------------------


def mode_probe(spec, out_path):
    res = {"compiles": False, "load_error": None, "examples": [], "fp": []}
    code = spec["prefix"] + "\n" + spec["code"]
    ep = spec["entry_point"]
    t = spec["timeout"]
    try:
        compile(spec["code"], "<c>", "exec")
        res["compiles"] = ("def " + ep) in spec["code"]  # verify_syntax semantics
    except Exception:
        res["compiles"] = False
    try:
        g = _load(code, t)
        if _resolve(g, ep) is None:
            raise NameError(ep)
    except BaseException as e:
        res["load_error"] = _err_name(e)
        _dump(res, out_path)
        return
    # public examples
    for stmt in spec["examples"]:
        try:
            _with_timeout(t, exec, stmt, g)
            res["examples"].append("pass")
        except AssertionError:
            res["examples"].append("fail")
        except BaseException as e:
            res["examples"].append("E:" + _err_name(e))
        _dump(res, out_path)
    # fingerprint on generated inputs (same harness as core/sandbox.py)
    fn = g[ep]
    lcb = bool(spec.get("lcb"))
    for expr in spec["inputs"]:
        def _one():
            args = eval(expr, g)
            if isinstance(args, tuple):
                r = fn(*args)
            else:
                r = fn(args)
            return repr(_normalize(_to_lists(r) if lcb else r))

        try:
            r = _with_timeout(t, _one)
            res["fp"].append("ok:" + _hash(r))
        except BaseException as e:
            res["fp"].append("E:" + _err_name(e))
        _dump(res, out_path)
    _dump(res, out_path)


def mode_expected(spec, out_path):
    g = _load(spec["code"], 60)
    fn = g[spec["entry_point"]]
    import time as _time

    outs = []
    times = []
    for inp in spec["inputs"]:
        t0 = _time.perf_counter()
        try:
            val = _with_timeout(spec["timeout"], fn, *copy.deepcopy(inp))
            try:
                pickle.dumps(val)
                outs.append(("ok", val))
            except Exception:  # e.g. re.Match (Mbpp/737, 787, 794)
                outs.append(("opaque", (type(val).__name__, bool(val))))
        except BaseException as e:
            outs.append(("err", _err_name(e)))
        times.append(_time.perf_counter() - t0)
    with open(out_path, "wb") as f:
        pickle.dump({"outs": outs, "times": times}, f)


def mode_gt(spec, out_path):
    res = {"gt_orig": False, "gt_base": False, "gt_plus": False, "fail_at": None}
    code = spec["prefix"] + "\n" + spec["code"]
    t = spec["timeout"]
    # 1) original tests (HumanEval check() / MBPP assertion) — paper-comparable
    try:
        g = _load(code, t)
        _resolve(g, spec["entry_point"])
        _with_timeout(spec["orig_timeout"], exec, spec["test_orig"], g)
        res["gt_orig"] = True
    except BaseException:
        res["gt_orig"] = False
    _dump(res, out_path)
    # 2) EvalPlus base + plus inputs vs canonical outputs
    with open(spec["expected_path"], "rb") as f:
        ref = pickle.load(f)
    expected, ref_times = ref["outs"], ref["times"]
    with open(spec["inputs_path"], "rb") as f:
        inputs = pickle.load(f)
    n_base = spec["n_base"]
    atol = spec["atol"]
    ep = spec["entry_point"]
    special = spec.get("special")
    try:
        g = _load(code, t)
        fn = _resolve(g, ep)
        if fn is None:
            raise NameError(ep)
    except BaseException:
        _dump(res, out_path)
        return
    lcb = bool(spec.get("lcb"))
    ok = True
    for i, (inp, (st, exp)) in enumerate(zip(inputs, expected)):
        if i == n_base and not res["gt_base"]:
            res["gt_base"] = True  # all base inputs passed
            _dump(res, out_path)
        if st not in ("ok", "opaque"):
            continue  # canonical failed on this input — skip, as evalplus would not include it
        try:
            # EvalPlus-style: time limit scales with the reference runtime
            out = _with_timeout(max(t, 4 * ref_times[i] + 1.0), fn, *copy.deepcopy(inp))
            if st == "opaque":  # unpicklable reference output: compare type + truthiness
                good = type(out).__name__ == exp[0] and bool(out) == exp[1]
            elif special == "find_zero":
                # HumanEval/32: any root counts.  Like EvalPlus's ad-hoc rule, check only the
                # residual |poly(xs, out)| against the problem's atol (1e-4, the same bound as
                # the original HumanEval check); never compare with the reference output:
                # Newton's method in the canonical solution is ill-conditioned on some
                # inputs and can return a non-root that differs across platforms.
                xs = inp[0]
                good = abs(sum(c * out ** k for k, c in enumerate(xs))) <= (atol or 1e-4)
            elif lcb:
                good = _evalplus_eq(_to_lists(out), exp, atol)
            else:
                good = _evalplus_eq(out, exp, atol)
        except BaseException:
            good = False
        if not good:
            ok = False
            res["fail_at"] = i
            break
    if ok:
        res["gt_base"] = True
        res["gt_plus"] = True
    _dump(res, out_path)


def _dump(res, out_path):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_path)


def main():
    mode, spec_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(spec_path) as f:
        spec = json.load(f)
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    sys.stdin = io.StringIO("")
    {"probe": mode_probe, "expected": mode_expected, "gt": mode_gt}[mode](spec, out_path)


if __name__ == "__main__":
    main()
