"""Build the full candidate pool for one problem.

For every problem we generate, once, everything any adaptive policy could
ever ask for (K_max sketches x M_max fills, a greedy sample, N_max flat
samples, D test inputs), then execute every candidate.  Because

  * all K_max category names come from ONE enumeration call,
  * each sketch depends only on its own category name, and
  * fill m of a sketch depends only on fills < m of the same sketch,

any policy that generates sketches in category order and fills in index order
sees exactly a prefix of this pool.  Offline replay is therefore exact
(same prompts, same samples, same token counts), not an approximation.

Generation follows the original Sketch-and-Verify code 1:1
(sketch_gen._category_first, sketch_fill.fill_sketch, core/input_gen,
run_scaling_fast.run_flat_n); see prompts.py.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from . import prompts as P
from .benchmarks import Problem
from .execution import ground_truth, probe
from .llm import LLM, CallRecord


@dataclass
class PoolConfig:
    K: int = 10  # sketches (category names enumerated)
    M: int = 10  # fills per sketch
    N_flat: int = 100  # flat samples at T=0.8 (+1 greedy at T=0)
    K_in: int = 10  # input-generation categories
    M_in: int = 5  # instantiations per input category
    category_temperature: float = 0.7
    sketch_temperature: float = 0.4  # = max(0.7 - 0.3, 0.1) in the original
    fill_temperature: float = 0.8
    flat_temperature: float = 0.8
    input_temperature: float = 0.7
    max_tokens: int = 2048
    exec_timeout: float = 5.0
    diverse_every: int = 3  # every 3rd fill uses DIVERSE_FILL_PROMPT


def _u(rec: CallRecord) -> dict:
    return rec.usage()


# ---------------------------------------------------------------------------
# generation pieces
# ---------------------------------------------------------------------------


def gen_test_inputs(problem: Problem, llm: LLM, cfg: PoolConfig) -> dict:
    """core/input_gen.generate_sketch_inputs, with per-call usage."""
    calls = []
    desc = problem.prompt
    rec = llm.generate(
        P.INPUT_SKETCH_PROMPT.format(problem_description=desc, K=cfg.K_in),
        temperature=cfg.input_temperature, max_tokens=8192,
    )
    calls.append(_u(rec))

    def fallback() -> dict:
        r = llm.generate(
            P.INPUT_FALLBACK_PROMPT.format(count=cfg.K_in * cfg.M_in, problem_description=desc),
            temperature=cfg.input_temperature, max_tokens=4096,
        )
        calls.append(_u(r))
        try:
            v = P.parse_json_any(r.text)
            inputs = [str(x) for x in v[: cfg.K_in * cfg.M_in]] if isinstance(v, list) else []
        except Exception:
            inputs = []
        return {"inputs": inputs, "calls": calls, "fallback": True}

    try:
        sketches = P.parse_json_any(rec.text)
    except Exception:
        return fallback()
    if not isinstance(sketches, list) or not sketches:
        return fallback()
    valid = [s for s in sketches if isinstance(s, dict) and "input_expr" in s]

    def variations(s: dict) -> Tuple[List[str], dict]:
        r = llm.generate(
            P.INPUT_VARIATION_PROMPT.format(
                description=s.get("description", ""), input_expr=s["input_expr"],
                problem_description=desc, M=cfg.M_in - 1,
            ),
            temperature=cfg.input_temperature, max_tokens=4096,
        )
        try:
            v = P.parse_json_any(r.text)
            out = [str(x) for x in v[: cfg.M_in - 1]] if isinstance(v, list) else []
        except Exception:
            out = []
        return out, _u(r)

    inputs: List[str] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, len(valid))) as ex:
        results = list(ex.map(variations, valid)) if cfg.M_in > 1 else [([], None)] * len(valid)
    for s, (vars_, usage) in zip(valid, results):
        inputs.append(str(s["input_expr"]))
        inputs.extend(vars_)
        if usage:
            calls.append(usage)
    return {"inputs": inputs, "calls": calls, "fallback": False}


def gen_sketch_and_fills(problem: Problem, llm: LLM, cfg: PoolConfig, k: int, category: str) -> dict:
    """sketch_gen._generate_sketch_for_category + sketch_fill.fill_sketch."""
    rec = llm.generate(
        P.SKETCH_FROM_CATEGORY_PROMPT.format(
            category=category, problem_description=problem.prompt,
            function_signature=problem.signature,
        ),
        temperature=cfg.sketch_temperature, max_tokens=cfg.max_tokens,
    )
    code = P.extract_code_block(rec.text)
    errors = P.validate_sketch(code)
    entry = {
        "k": k, "category": category, "code": code, "num_holes": code.count("??"),
        "valid": not errors, "errors": errors, "parses": P.sketch_parses_as_python(code),
        "call": _u(rec), "fills": [],
    }
    if errors:  # original pipeline never fills invalid sketches
        return entry
    fills: List[str] = []
    for m in range(cfg.M):
        if m > 0 and m % cfg.diverse_every == 0:
            prompt = P.DIVERSE_FILL_PROMPT.format(
                problem_description=problem.prompt, category=category,
                sketch_code=code, previous_fill=fills[-1],
            )
            kind = "diverse"
        else:
            prompt = P.FILL_PROMPT.format(
                problem_description=problem.prompt, category=category, sketch_code=code,
            )
            kind = "standard"
        r = llm.generate(prompt, temperature=cfg.fill_temperature,
                         max_tokens=cfg.max_tokens, sample_idx=m)
        fill = P.ensure_function_def(P.extract_code_block(r.text), problem.entry_point)
        fills.append(fill)
        entry["fills"].append({"m": m, "kind": kind, "cid": f"s{k}f{m}", "code": fill, "call": _u(r)})
    return entry


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def build_problem(
    problem: Problem, llm: LLM, cfg: PoolConfig, exec_pool: cf.Executor, log=print,
) -> dict:
    t0 = time.time()
    flat_prompt = P.FLAT_PROMPT.format(problem_description=problem.prompt)

    with cf.ThreadPoolExecutor(max_workers=cfg.K + 4) as ex:
        f_inputs = ex.submit(gen_test_inputs, problem, llm, cfg)
        f_greedy = ex.submit(llm.generate, flat_prompt, 0.0, cfg.max_tokens, 0)
        f_flat = [
            ex.submit(llm.generate, flat_prompt, cfg.flat_temperature, cfg.max_tokens, i)
            for i in range(cfg.N_flat)
        ]
        # category enumeration (sketch_gen._get_categories)
        cat_rec = llm.generate(
            P.CATEGORY_PROMPT.format(K=cfg.K, problem_description=problem.prompt,
                                     function_signature=problem.signature),
            temperature=cfg.category_temperature, max_tokens=1024,
        )
        try:
            names = P.parse_json_array(cat_rec.text)[: cfg.K]
            parse_ok = True
        except Exception:
            names = [f"approach_{i + 1}" for i in range(cfg.K)]
            parse_ok = False
        f_sk = [ex.submit(gen_sketch_and_fills, problem, llm, cfg, k, c) for k, c in enumerate(names)]

        test_inputs = f_inputs.result()
        greedy_rec = f_greedy.result()
        flat_recs = [f.result() for f in f_flat]
        sketches = [f.result() for f in f_sk]
    t_gen = time.time() - t0

    # ---- collect candidates ------------------------------------------------
    cands: Dict[str, str] = {"g": P.extract_first_code_block(greedy_rec.text)}
    flat = []
    for i, r in enumerate(flat_recs):
        cid = f"f{i}"
        cands[cid] = P.extract_first_code_block(r.text)
        flat.append({"n": i, "cid": cid, "call": _u(r)})
    for s in sketches:
        for fl in s["fills"]:
            cands[fl["cid"]] = fl.pop("code")

    # ---- execute unique programs ------------------------------------------
    inputs = test_inputs["inputs"]
    uniq: Dict[str, str] = {}
    for code in cands.values():
        uniq.setdefault(hashlib.sha1(code.encode()).hexdigest(), code)

    def run_one(code: str) -> dict:
        r = probe(code, problem, inputs, timeout=cfg.exec_timeout)
        if r["compiles"] and r["load_error"] is None:
            r.update(ground_truth(code, problem, timeout=cfg.exec_timeout))
        else:
            r.update({"gt_orig": False, "gt_base": False, "gt_plus": False})
        return r

    futs = {h: exec_pool.submit(run_one, c) for h, c in uniq.items()}
    exec_res = {h: f.result() for h, f in futs.items()}
    t_exec = time.time() - t0 - t_gen

    candidates = {}
    for cid, code in cands.items():
        h = hashlib.sha1(code.encode()).hexdigest()
        candidates[cid] = dict(code=code, code_len=len(code), code_hash=h[:16], **exec_res[h])

    # ---- cost summary --------------------------------------------------------
    def tot(usages: List[dict]) -> dict:
        i = sum(u["in_tok"] for u in usages)
        o = sum(u["out_tok"] for u in usages)
        th = sum(u["think_tok"] for u in usages)
        return {"in_tok": i, "out_tok": o, "think_tok": th, "n_calls": len(usages),
                "usd": round(llm.cost(i, o, th), 8)}

    sketch_calls = [s["call"] for s in sketches]
    fill_calls = [fl["call"] for s in sketches for fl in s["fills"]]
    cost = {
        "test_inputs": tot(test_inputs["calls"]),
        "categories": tot([_u(cat_rec)]),
        "sketches": tot(sketch_calls),
        "fills": tot(fill_calls),
        "greedy": tot([_u(greedy_rec)]),
        "flat": tot([f["call"] for f in flat]),
    }

    return {
        "task_id": problem.task_id,
        "benchmark": problem.benchmark,
        "provider": llm.provider,
        "model": llm.model,
        "thinking_level": llm.thinking_level,
        "price_per_1m": list(llm.price),
        "config": asdict(cfg),
        "examples": problem.examples,
        "test_inputs": test_inputs,
        "categories": {"names": names, "parse_ok": parse_ok, "call": _u(cat_rec)},
        "sketches": sketches,
        "greedy": {"cid": "g", "call": _u(greedy_rec)},
        "flat": flat,
        "candidates": candidates,
        "cost": cost,
        "timing": {"gen_s": round(t_gen, 1), "exec_s": round(t_exec, 1),
                   "n_unique_programs": len(uniq)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
