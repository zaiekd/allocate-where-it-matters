"""Prompts and response parsers.

All prompt texts are copied VERBATIM from the original Sketch-and-Verify code
(sketch-research/projects/sketch-and-verify/src/sketch_gen.py, sketch_fill.py,
core/input_gen.py, scripts/run_scaling_fast.py) so that the candidate pool is
distributionally identical to what the paper's pipeline would produce.
"""

from __future__ import annotations

import json
import re
from typing import Any, List

# ---------------------------------------------------------------------------
# Stage 1a: category enumeration  (sketch_gen.CATEGORY_PROMPT)
# ---------------------------------------------------------------------------

CATEGORY_PROMPT = """\
Given this programming problem, list {K} fundamentally different algorithmic \
STRATEGIES that could solve it. Each strategy should use a different core \
algorithm or data structure.

Problem:
{problem_description}

Function signature:
{function_signature}

Output a JSON array of strategy names, e.g.:
["hash map lookup", "sorting + two pointers", "brute force nested loops", ...]

List EXACTLY {K} strategies. Be creative — consider approaches from \
different algorithmic paradigms (brute force, sorting, hashing, two pointers, \
binary search, dynamic programming, greedy, stack/queue, divide and conquer, \
math/formula, graph-based, bit manipulation).

Return ONLY the JSON array, no other text."""

# ---------------------------------------------------------------------------
# Stage 1b: one sketch per category  (sketch_gen.SKETCH_FROM_CATEGORY_PROMPT)
# ---------------------------------------------------------------------------

SKETCH_FROM_CATEGORY_PROMPT = """\
Write a Python program SKETCH using the "{category}" strategy for this problem.

Problem:
{problem_description}

Function signature:
{function_signature}

A sketch is partial Python code where implementation details are replaced \
with ?? placeholders. Rules:
- The overall structure (algorithm, data structures, control flow) must \
be fully specified
- Replace expressions, conditions, and short fragments with ??
- Each ?? should represent a single expression or condition (not \
multiple statements)
- The sketch must be syntactically valid Python EXCEPT for ?? tokens
- Include a one-line comment at the top naming the approach
- Target 4-8 holes (??) per sketch — enough for variation, not so many \
that the structure is lost

Example of a good sketch:
```python
# Hash map approach
def two_sum(nums, target):
    seen = {{}}
    for ?? in ??:
        complement = ??
        if ?? in seen:
            return ??
        seen[??] = ??
    return ??
```

Now generate the sketch using the "{category}" strategy.
Output ONLY the Python code block, no explanations."""

# ---------------------------------------------------------------------------
# Stage 2: fills  (sketch_fill.FILL_PROMPT / DIVERSE_FILL_PROMPT)
# ---------------------------------------------------------------------------

FILL_PROMPT = """\
Complete this program sketch by replacing every ?? with the correct Python \
expression or code fragment.

Problem:
{problem_description}

Sketch ({category}):
```python
{sketch_code}
```

Rules:
- Replace each ?? with a single Python expression or condition
- The result must be a valid, complete, executable Python function
- Do NOT change the overall structure — keep the same algorithm, \
loops, and data structures
- Output ONLY the completed function, no explanations

```python
"""

DIVERSE_FILL_PROMPT = """\
Complete this program sketch by replacing every ?? with a DIFFERENT but \
still correct Python expression compared to the previous completion.

Problem:
{problem_description}

Sketch ({category}):
```python
{sketch_code}
```

Previous completion (generate something DIFFERENT):
```python
{previous_fill}
```

Rules:
- Replace each ?? differently from the previous completion
- The result must still be correct and executable
- Keep the same overall structure from the sketch
- Output ONLY the completed function

```python
"""

# ---------------------------------------------------------------------------
# Flat sampling  (run_scaling_fast.run_flat_n)
# ---------------------------------------------------------------------------

FLAT_PROMPT = (
    "Complete the following Python function. "
    "Output ONLY the function code, no explanations.\n\n"
    "{problem_description}"
)

# ---------------------------------------------------------------------------
# Test-input generation  (core/input_gen.py)
# ---------------------------------------------------------------------------

INPUT_SKETCH_PROMPT = """\
You are generating diverse test inputs for a Python function.

Function signature and description:
{problem_description}

Generate {K} diverse INPUT SKETCHES. Each sketch should target a fundamentally
different equivalence class of behavior. Think about:
- Edge cases (empty, single element, None, zero, negative)
- Boundary values (max int, very long strings, deeply nested)
- Typical cases (medium-sized, mixed types if applicable)
- Special structure (sorted, reversed, all duplicates, alternating)
- Type boundaries (if applicable: floats near integers, unicode strings)

For each sketch, provide:
1. A short description of what case it tests
2. A concrete Python expression that produces a valid input

Output as JSON array:
[
  {{"description": "empty list", "input_expr": "[]"}},
  {{"description": "single element", "input_expr": "[42]"}},
  ...
]

Generate EXACTLY {K} sketches. Each input_expr must be a valid Python expression
that can be passed directly to the function.
"""

INPUT_VARIATION_PROMPT = """\
Given this input sketch for a Python function:
Description: {description}
Example: {input_expr}

Function info:
{problem_description}

Generate {M} more concrete inputs that follow the SAME pattern but with
different specific values. Output as a JSON array of Python expressions.
Example output: ["[1, 2]", "[5, 10]", "[0, -3]"]

Return ONLY the JSON array, no other text.
"""

INPUT_FALLBACK_PROMPT = """\
Generate {count} diverse concrete test inputs for this Python function.
Each input should be a valid Python expression.

{problem_description}

Output as a JSON array of strings, where each string is a Python expression.
Example: ["[1, 2, 3]", "[]", "[-1, 0, 1]"]
"""

# ---------------------------------------------------------------------------
# Parsers (copied from sketch_gen.py / sketch_fill.py / input_gen.py)
# ---------------------------------------------------------------------------


def parse_json_array(text: str) -> List[str]:
    """sketch_gen._parse_json_array"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    result = json.loads(text)
    if isinstance(result, list):
        return [str(item) for item in result]
    raise ValueError(f"Expected JSON array, got {type(result)}")


def parse_json_any(text: str) -> Any:
    """input_gen._parse_json_from_response"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def extract_code_block(text: str) -> str:
    """sketch_gen/sketch_fill._extract_code_block (largest fenced block)."""
    text = (text or "").strip()
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return text


def extract_first_code_block(text: str) -> str:
    """run_scaling_fast.run_flat_n extraction (FIRST fenced python block)."""
    text = (text or "").strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


def ensure_function_def(code: str, entry_point: str) -> str:
    """sketch_fill._ensure_function_def"""
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return code
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if match and match.group(1) != entry_point:
        code = code.replace(f"def {match.group(1)}(", f"def {entry_point}(", 1)
    return code


def validate_sketch(sketch_code: str) -> List[str]:
    """Validation used by sketch_gen._generate_sketch_for_category.

    Returns list of error strings (empty == valid).
    """
    errors: List[str] = []
    num_holes = sketch_code.count("??")
    if num_holes == 0:
        errors.append("No ?? holes found — complete program, not a sketch")
    if num_holes > 15:
        errors.append(f"Too many holes ({num_holes}) — sketch too abstract")
    if not re.search(r"^\s*def\s+\w+\s*\(", sketch_code, re.MULTILINE):
        errors.append("No function definition found")
    return errors


def sketch_parses_as_python(sketch_code: str) -> bool:
    """Extra diagnostic from sketch_parser.validate_sketch_structure (not used
    for filtering, only recorded)."""
    import ast

    try:
        ast.parse(sketch_code.replace("??", "_ph_"))
        return True
    except SyntaxError:
        return False
