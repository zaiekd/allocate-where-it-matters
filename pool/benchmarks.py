"""HumanEval+ / MBPP+ loader that does not need the `evalplus` package.

Reads the official EvalPlus release files (data/HumanEvalPlus.jsonl.gz,
data/MbppPlus.jsonl.gz; see scripts/download_data.sh) and additionally
extracts the *public* examples that appear in each problem statement, which
the adaptive policies use as a cheap filter ("does this fill pass the
examples shown in the prompt?").

Note on ground truth.  The original repo evaluated "HumanEval+" with the
`test` field, which is the ORIGINAL HumanEval `check()` (~7 asserts), not the
~1000 EvalPlus inputs.  We keep that as `gt_orig` (for comparability with the
paper's numbers) and also compute `gt_base` / `gt_plus` the EvalPlus way.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = {
    "humaneval": os.path.join(ROOT, "data", "HumanEvalPlus.jsonl.gz"),
    "mbpp": os.path.join(ROOT, "data", "MbppPlus.jsonl.gz"),
}
ALIASES = {"humaneval+": "humaneval", "he": "humaneval", "mbpp+": "mbpp", "livecodebench": "lcb"}
LCB_DIR = os.path.join(ROOT, "data", "lcb")


@dataclass
class Problem:
    task_id: str
    benchmark: str  # "humaneval" | "mbpp"
    prompt: str  # text shown to the LLM (problem_description in original code)
    entry_point: str
    signature: str
    canonical_solution: str  # full runnable reference code (for GT only)
    exec_prefix: str  # imports/helpers from the prompt, prepended when executing
    test_orig: str  # original-style test script body (check() or assertion)
    base_input: List[Any] = field(default_factory=list)
    plus_input: List[Any] = field(default_factory=list)
    atol: float = 0.0
    examples: List[str] = field(default_factory=list)  # public example statements
    # LiveCodeBench: reference outputs ship with the data (no canonical solution);
    # one entry per input of base_input + plus_input
    expected_output: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public-example extraction
# ---------------------------------------------------------------------------

_SEPARATORS = ["==>", "➞", "=>", "->", "==", "should return", "returns", "="]


def _balanced_call(text: str, start: int) -> Optional[int]:
    """Given index of '(' return index just past the matching ')'."""
    depth = 0
    in_str: Optional[str] = None
    i = start
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _is_expr(s: str) -> bool:
    try:
        ast.parse(s, mode="eval")
        return True
    except SyntaxError:
        return False


_CONSTS = {"True", "False", "None"}


def _is_literalish(s: str, allow_calls: bool = False) -> bool:
    """Expression built only from literals/operators (no free names, no calls)."""
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _CONSTS:
            return False
        if isinstance(node, ast.Call) and not allow_calls:
            return False
        if isinstance(node, (ast.Lambda, ast.Attribute, ast.Subscript)) and not allow_calls:
            # subscripts like [1,2][0] are fine but rare; keep it strict
            return False
    return True


def _call_args_literal(call: str) -> bool:
    try:
        node = ast.parse(call, mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(node, ast.Call):
        return False
    return all(_is_literalish(ast.unparse(a)) for a in node.args) and all(
        _is_literalish(ast.unparse(k.value)) for k in node.keywords
    )


def _strip_comment(s: str) -> str:
    # remove a trailing "# ..." that is not inside a string literal
    in_str = None
    for i, c in enumerate(s):
        if in_str:
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c == "#":
            return s[:i].rstrip()
    return s


def _norm_expected(e: str) -> str:
    e = _strip_comment(e.strip()).rstrip(".,;").strip()
    if e in ("true", "false", "none"):
        e = e.capitalize()
    return e


def _assign_list_to_args(text: str) -> Optional[str]:
    """'arr = [1,2], k = 3' -> '[1,2], 3' ; plain expression -> itself."""
    text = _strip_comment(text.strip()).rstrip(".,;").strip()
    if not text:
        return None
    try:
        node = ast.parse(f"__f({text})", mode="eval").body  # a, b  /  a=1, b=2
        if isinstance(node, ast.Call):
            parts = [ast.unparse(a) for a in node.args] + [ast.unparse(k.value) for k in node.keywords]
            if parts and all(_is_literalish(x) for x in parts):
                return ", ".join(parts)
    except SyntaxError:
        pass
    # "a = [1], b = 2" style (no parentheses wrapping keywords works above too)
    return None


def _make_assert(call: str, expected: str) -> str:
    return f"assert __ex_eq({call}, {expected})"


def _accept(call: str, expected: str) -> bool:
    return bool(expected) and _call_args_literal(call) and _is_literalish(expected)


def extract_humaneval_examples(prompt: str, entry_point: str) -> List[str]:
    """Extract (call, expected) pairs from a HumanEval docstring.

    Handles: the doctest form (`>>> f(x)` + next line); inline forms
    `f(x) ==> y`, `f(x) ➞ y`, `f(x) == y`, `f(x) = y`, `f(x) should return y`;
    `For lst = X the output should be Y`; and `Input: X` / `Output: Y` blocks.
    Only literal arguments / literal expected values are kept, so definitional
    lines such as `fib4(n) -> fib4(n-1) + ...` are dropped.
    """
    out: List[str] = []
    lines = prompt.splitlines()
    pat = re.compile(rf"\b{re.escape(entry_point)}\s*\(")
    for_pat = re.compile(r"^For (\w+ = .+?),? the output should be (.+)$")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>>"):
            call = _strip_comment(line[3:].strip())
            exp_lines = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if not nxt or nxt.startswith(">>>") or nxt.startswith('"""') or nxt.startswith("\'\'\'"):
                    break
                exp_lines.append(nxt)
                j += 1
            expected = _norm_expected(" ".join(exp_lines))
            if pat.search(call) and _accept(call, expected):
                out.append(_make_assert(call, expected))
            i = j
            continue
        m_for = for_pat.match(line)
        if m_for:
            args = _assign_list_to_args(m_for.group(1))
            expected = _norm_expected(m_for.group(2))
            if args is not None:
                call = f"{entry_point}({args})"
                if _accept(call, expected):
                    out.append(_make_assert(call, expected))
            i += 1
            continue
        if line.startswith("Input:") and i + 1 < len(lines):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("Output:"):
                args = _assign_list_to_args(line[len("Input:"):])
                expected = _norm_expected(lines[j].strip()[len("Output:"):])
                if args is not None:
                    call = f"{entry_point}({args})"
                    if _accept(call, expected):
                        out.append(_make_assert(call, expected))
                i = j + 1
                continue
        for m in pat.finditer(line):
            paren = line.find("(", m.start())
            end = _balanced_call(line, paren)
            if end is None:
                continue
            call = line[m.start():end]
            rest = line[end:].strip()
            for sep in _SEPARATORS:
                if rest.startswith(sep):
                    expected = _norm_expected(rest[len(sep):])
                    while expected and not _is_expr(expected):
                        if " " not in expected:
                            expected = ""
                            break
                        expected = _norm_expected(expected.rsplit(" ", 1)[0])
                    if _accept(call, expected):
                        out.append(_make_assert(call, expected))
                    break
        i += 1
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def extract_mbpp_examples(prompt: str) -> List[str]:
    return [ln.strip() for ln in prompt.splitlines() if ln.strip().startswith("assert ")]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _prefix_before_entry(prompt: str, entry_point: str) -> str:
    m = re.search(rf"^def\s+{re.escape(entry_point)}\s*\(", prompt, re.MULTILINE)
    return prompt[: m.start()] if m else ""


def load_benchmark(name: str) -> List[Problem]:
    name = ALIASES.get(name.lower(), name.lower())
    if name == "lcb":
        return load_lcb()
    path = DATA_FILES[name]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `bash scripts/download_data.sh` first"
        )
    problems: List[Problem] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            it = json.loads(line)
            ep = it["entry_point"]
            if name == "humaneval":
                prompt = it["prompt"]
                m = re.search(r"(def\s+\w+\s*\([^)]*\))", prompt)  # original regex
                sig = m.group(1) if m else "def solution(*args)"
                canonical = prompt + it["canonical_solution"]
                prefix = _prefix_before_entry(prompt, ep)
                test_orig = it["test"] + f"\n\ncheck({ep})\n"
                examples = extract_humaneval_examples(prompt, ep)
            else:
                prompt = it["prompt"]
                sig = f"def {ep}(...)"
                canonical = it["canonical_solution"]
                prefix = ""
                test_orig = it["assertion"] + "\n"
                examples = extract_mbpp_examples(prompt)
            problems.append(
                Problem(
                    task_id=it["task_id"],
                    benchmark=name,
                    prompt=prompt,
                    entry_point=ep,
                    signature=sig,
                    canonical_solution=canonical,
                    exec_prefix=prefix,
                    test_orig=test_orig,
                    base_input=list(it.get("base_input") or []),
                    plus_input=list(it.get("plus_input") or []),  # Mbpp/793 has {}
                    atol=float(it.get("atol", 0) or 0),
                    examples=examples,
                )
            )
    return problems


# ---------------------------------------------------------------------------
# LiveCodeBench (code_generation_lite), LeetCode functional problems only
# ---------------------------------------------------------------------------

LCB_PREFIX = (
    "from typing import *\n"
    "from collections import *\n"
    "from functools import *\n"
    "from itertools import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from math import *\n"
    "import math, heapq, collections, itertools, functools, bisect, string, re, operator\n"
)


def _lcb_private(raw: str) -> list:
    """private_test_cases is either JSON or base64(zlib(pickle(json-string)))."""
    try:
        return json.loads(raw)
    except Exception:
        import base64
        import pickle
        import zlib

        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def _lcb_signature(starter: str, func: str) -> Optional[str]:
    """'class Solution:\n    def f(self, a: int) -> int:'  ->  'def f(a: int) -> int'.
    None unless it is a plain one-method Solution without linked-list/tree types."""
    if "class Solution" not in starter or starter.count("def ") != 1:
        return None
    if re.search(r"\b(ListNode|TreeNode|Node)\b", starter):
        return None  # linked-list / tree I/O needs custom (de)serialisation
    m = re.search(r"def\s+(\w+)\s*\(\s*self\s*,?\s*(.*?)\)\s*(->\s*[^:]+)?:", starter, re.S)
    if not m or m.group(1) != func:
        return None
    args = " ".join(m.group(2).split())
    ret = (" " + " ".join(m.group(3).split())) if m.group(3) else ""
    return f"def {func}({args}){ret}"


def _lcb_args(s: str) -> list:
    return [json.loads(x) for x in s.split("\n") if x.strip() != ""]


def load_lcb(files: Optional[List[str]] = None) -> List[Problem]:
    """Environment knobs (all optional):
      LCB_FILES       comma list of jsonl files (default: every data/lcb/test*.jsonl)
      LCB_SINCE       keep problems with contest_date >= this (YYYY-MM-DD)
      LCB_DIFFICULTY  comma list, e.g. "medium,hard"
    Problems are ordered newest first, so --n-problems takes the most recent ones."""
    import glob as _glob

    if files is None:
        env = os.environ.get("LCB_FILES")
        files = [f.strip() for f in env.split(",")] if env else sorted(_glob.glob(os.path.join(LCB_DIR, "test*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no LiveCodeBench files in {LCB_DIR} (download test5.jsonl / test6.jsonl)")
    since = os.environ.get("LCB_SINCE", "")
    diffs = {d.strip().lower() for d in os.environ.get("LCB_DIFFICULTY", "").split(",") if d.strip()}
    out: List[Problem] = []
    seen = set()
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                it = json.loads(line)
                if it.get("platform") != "leetcode":
                    continue
                date = str(it.get("contest_date", ""))[:10]
                if since and date < since:
                    continue
                diff = str(it.get("difficulty", "")).lower()
                if diffs and diff not in diffs:
                    continue
                meta = json.loads(it.get("metadata") or "{}")
                func = meta.get("func_name")
                sig = _lcb_signature(it.get("starter_code", ""), func or "")
                if not func or not sig:
                    continue
                try:
                    pub = json.loads(it["public_test_cases"])
                    priv = _lcb_private(it["private_test_cases"])
                    tests = pub + priv
                    if not tests or any(t.get("testtype") != "functional" for t in tests):
                        continue
                    pub_io = [(_lcb_args(t["input"]), json.loads(t["output"])) for t in pub]
                    priv_io = [(_lcb_args(t["input"]), json.loads(t["output"])) for t in priv]
                except Exception:
                    continue
                tid = f"lcb/{it['question_id']}"
                if tid in seen:
                    continue
                seen.add(tid)
                # HumanEval-style prompt: signature + docstring, so every original prompt works as-is
                content = it["question_content"].replace('"""', "'''").strip()
                body = "\n".join(("    " + ln) if ln.strip() else "" for ln in content.split("\n"))
                prompt = f'from typing import *\n\n{sig}:\n    """\n{body}\n    """\n'
                examples = [f"assert __ex_eq({func}(*{a!r}), {o!r})" for a, o in pub_io]
                out.append(Problem(
                    task_id=tid, benchmark="lcb", prompt=prompt, entry_point=func, signature=sig,
                    canonical_solution="", exec_prefix=LCB_PREFIX, test_orig="\n".join(examples) + "\n",
                    base_input=[a for a, _ in pub_io], plus_input=[a for a, _ in priv_io], atol=0.0,
                    examples=examples, expected_output=[o for _, o in pub_io + priv_io],
                    meta={"date": date, "difficulty": diff, "title": it.get("question_title", "")},
                ))
    out.sort(key=lambda p: (p.meta["date"], p.task_id), reverse=True)
    return out


def safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "_")
