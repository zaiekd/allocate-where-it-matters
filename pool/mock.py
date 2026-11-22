"""Offline mock LLM for testing the whole pipeline without an API key.

It recognises which prompt template it was given and answers with plausible
content built from the benchmark's canonical solution:
  * sketches  = canonical solution with `return` expressions replaced by ??
  * fills / flat samples = canonical solution, randomly mutated (off-by-one,
    flipped comparison, ...) so that fingerprints and correctness vary
  * test inputs = EvalPlus base/plus inputs
Token counts are faked as len(text)//4.  NEVER use mock pools for results.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import List, Optional

from .benchmarks import Problem
from .llm import LLM, CallRecord

_MUTATIONS = [
    (" < ", " <= "), (" <= ", " < "), (" > ", " >= "), (" >= ", " > "),
    (" == ", " != "), (" + 1", " + 2"), (" - 1", ""), ("[1:]", "[2:]"),
    (" and ", " or "), ("True", "False"), ("range(", "range(1 + "),
]


def _args_expr(args: list) -> str:
    return repr(tuple(args)) if len(args) != 1 else repr(args[0])


class MockLLM(LLM):
    def __init__(self, problems: List[Problem], **kw) -> None:
        super().__init__(model="mock", provider="mock", **kw)
        self.problems = problems

    def _find(self, prompt: str) -> Optional[Problem]:
        best = None
        for p in self.problems:
            if p.prompt.strip() and p.prompt.strip() in prompt:
                if best is None or len(p.prompt) > len(best.prompt):
                    best = p
        return best

    def _mutate(self, code: str, rng: random.Random, p: float) -> str:
        if rng.random() >= p:
            return code
        opts = [(a, b) for a, b in _MUTATIONS if a in code]
        if not opts:
            return code + "\n    raise ValueError('mock bug')"  # unreachable-ish no-op
        a, b = rng.choice(opts)
        idx = [m.start() for m in re.finditer(re.escape(a), code)]
        i = rng.choice(idx)
        return code[:i] + b + code[i + len(a):]

    def _sketchify(self, code: str) -> str:
        lines = code.splitlines()
        n = 0
        for i, ln in enumerate(lines):
            m = re.match(r"^(\s*)return (.+)$", ln)
            if m and n < 3:
                lines[i] = f"{m.group(1)}return ??"
                n += 1
        if n == 0:
            for i, ln in enumerate(lines):
                m = re.match(r"^(\s+)(\w+) = (.+)$", ln)
                if m:
                    lines[i] = f"{m.group(1)}{m.group(2)} = ??"
                    n += 1
                    break
        return "# mock approach\n" + "\n".join(lines)

    def _call_api(self, prompt: str, temperature: float, max_tokens: int, sample_idx: int = 0) -> CallRecord:
        time.sleep(0.002)
        seed = int(hashlib.md5(f"{prompt}|{temperature}|{sample_idx}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        prob = self._find(prompt)
        canon = prob.canonical_solution if prob else "def solution(*a):\n    return None"
        if prob and prob.benchmark == "humaneval":
            # strip the prompt's helper prefix: keep from `def entry(` on
            m = re.search(rf"^def\s+{re.escape(prob.entry_point)}\s*\(", canon, re.MULTILINE)
            canon_fn = canon[m.start():] if m else canon
        else:
            canon_fn = canon

        if prompt.startswith("You are generating diverse test inputs"):
            ins = (prob.base_input + prob.plus_input) if prob else [[1]]
            k = int(re.search(r"Generate (\d+) diverse INPUT", prompt).group(1))
            pick = rng.sample(ins, min(k, len(ins)))
            text = json.dumps([{"description": f"case {i}", "input_expr": _args_expr(a)} for i, a in enumerate(pick)])
        elif prompt.startswith("Given this input sketch"):
            ins = (prob.base_input + prob.plus_input) if prob else [[1]]
            m = int(re.search(r"Generate (\d+) more concrete inputs", prompt).group(1))
            text = json.dumps([_args_expr(a) for a in rng.sample(ins, min(m, len(ins)))])
        elif prompt.startswith("Given this programming problem, list"):
            k = int(re.search(r"list (\d+) fundamentally", prompt).group(1))
            text = json.dumps([f"mock strategy {i}" for i in range(k)])
        elif prompt.startswith("Write a Python program SKETCH"):
            text = "```python\n" + self._sketchify(canon_fn) + "\n```"
        elif prompt.startswith("Complete this program sketch"):
            text = "```python\n" + self._mutate(canon_fn, rng, 0.35) + "\n```"
        elif prompt.startswith("Complete the following Python function"):
            code = canon_fn if temperature == 0 else self._mutate(canon_fn, rng, 0.45)
            text = "```python\n" + code + "\n```"
        else:
            text = "[]"
        return CallRecord(text=text, in_tok=len(prompt) // 4, out_tok=len(text) // 4,
                          think_tok=0, finish="STOP", latency_s=0.002)
