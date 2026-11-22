# Allocate Where It Matters: Difficulty-Aware Inference-Time Allocation for Code Generation

This repository contains the code for the thesis **"Allocate Where It Matters: Difficulty-Aware Inference-Time
Allocation for Code Generation"**. It studies where an LLM's inference budget should go when generating code: a cheap
*difficulty gate* answers problems that look easy from a few flat samples, and only the remaining (escalated) problems
receive additional computation. Two escalation strategies are compared under the same gate and the same
example-filtered selector, against fixed-budget baselines including *Sketch-and-Verify* (enumerate solution strategies
→ write a program sketch with `??` holes per strategy → fill the holes → select a program by execution-based semantic
voting).

| Method | Class (`pool/policies.py`) | Idea |
|---|---|---|
| **BAS** — Breadth-first Adaptive Sketching | `BreadthSketch` | Gate on cheap flat samples (`gate="disagree"`: **BAS-Dis**, `gate="examples"`: **BAS-Ex**); on hard problems, expand many strategies with one fill each |
| **GFS** — Gated Flat Sampling | `GatedFlat` | Same example gate; on hard problems, `n_extra` additional independent *flat* samples instead of sketches |
| **SAS** — Sequential Adaptive Sketching (design alternative) | `AdaptiveSketch` | Gate → sequential sketches with cross-strategy early stopping → successive halving of fills |
| Adaptive flat baseline | `AdaptiveFlat` | Flat sampling in small batches until a stopping rule fires (Adaptive-Consistency style) |
| Original baselines | `Flat(N)`, `FixedSketch(K, M)` | Fixed-budget flat sampling and fixed-`K/M` Sketch-and-Verify |

Benchmarks evaluated in the thesis: **HumanEval+** (EvalPlus) and **LiveCodeBench** (LeetCode functional subset). The
loaders also support **MBPP+**, which is not part of the reported results.

---

## 1. Approach: build once, replay offline

Comparing many policies and hyperparameters by calling the API each time would be slow and expensive. Instead, for
each problem we generate **one maximal candidate pool once**, and every policy is replayed offline on that pool:

| Item | Count | Purpose |
|---|---|---|
| Strategy names (categories) | 1 call listing K = 10 | the "idea" behind each sketch |
| Sketches | 1 per strategy, 10 total | program skeletons with `??` holes |
| Fills | M = 10 per sketch, 100 total | Sketch-and-Verify candidates |
| Greedy sample | 1 (temperature 0) | flat N = 1 baseline |
| Flat samples | 100 (temperature 0.8) | flat baselines, gates, GFS |
| Test inputs | 10 categories × 5 = 50 | execution fingerprints |

Every candidate is executed and we record: whether it compiles/loads; whether it passes the **public examples shown in
the problem statement** (used by the adaptive policies; no ground-truth leakage); its **execution fingerprint** (output
hash or exception type on each generated input, used for clustering/voting); **correctness** under three ground truths
(Section 4); and the **token usage of every LLM call** (input / output / thinking tokens separately).

**Why replay is exact, not approximate.**
All 10 strategy names come from a *single* call; each sketch depends only on its own strategy name; fill `m` of a sketch
depends only on fills `< m` of the same sketch (every third fill uses the "generate a different fill" prompt, as in the
original code). Any policy that generates sketches in strategy order and fills in index order therefore sees exactly a
**prefix** of the pool: identical prompts, identical samples, identical token counts.

---

## 2. Repository layout

```
allocate-where-it-matters/
├── README.md
├── .env.template            model configuration template (copy to .env)
├── requirements.txt         nothing is required for OpenAI-compatible APIs
├── data/
│   ├── HumanEvalPlus.jsonl.gz   official EvalPlus release (164 problems, v0.1.10)
│   └── MbppPlus.jsonl.gz        official EvalPlus release (378 problems, v0.2.0)
├── pool/                    core library
│   ├── prompts.py           all prompts (verbatim from the original code) + parsers
│   ├── llm.py               Gemini / OpenAI-compatible client: per-call tokens, SQLite cache, retries
│   ├── benchmarks.py        HumanEval+ / MBPP+ / LiveCodeBench loaders; public-example extraction
│   ├── _runner.py           sandboxed executor (separate subprocess): examples / fingerprints / ground truth
│   ├── execution.py         wrappers around _runner; caches canonical-solution outputs
│   ├── builder.py           builds the full candidate pool for one problem
│   ├── replay.py            offline replay: semantic-voting selection + token accounting
│   ├── policies.py          fixed baselines, adaptive flat, AdaptiveSketch, BreadthSketch (BAS), GatedFlat (GFS)
│   └── mock.py              mock LLM for testing the pipeline without an API key
├── scripts/
│   ├── check_api.py         one tiny call to verify the .env configuration
│   ├── build_pool.py        build candidate pools (resumable)
│   ├── inspect_pool.py      replay the original fixed configurations; pool diagnostics
│   ├── run_test10.sh        end-to-end smoke test on 10 problems
│   ├── run_policies.py      replay all policies, hyperparameter sweeps, dev/test split
│   ├── plot_policies.py     cost–accuracy and budget-histogram figures for run_policies.py output
│   ├── analyze_pool.py      main results: pass@1 / coverage / tokens by difficulty + paired bootstrap
│   ├── plot_coverage.py     coverage-vs-tokens analysis (generation side, oracle selector)
│   ├── plot_models.py       cross-model pass@1-vs-cost figure
│   ├── ablation.py          ablation study across pools
│   ├── hparam_sensitivity.py  hyperparameter regret + leave-one-pool/benchmark-out selection
│   ├── basdis_vs_greedy.py, paper_vs_original.py, paper_filter_step.py   paired-bootstrap tests
│   ├── lcb_stats.py         summary of the LiveCodeBench problems kept by the loader
│   ├── regrade.py           re-run ground truth for selected problems of a pool
│   └── download_data.sh     re-download the EvalPlus data
└── tests/test_policies.py   unit tests for the policy logic
```

Generated at run time and not tracked: `pools/<benchmark>/<name>/<task_id>.json` (candidate pools),
`results/` (analysis outputs), `cache/` (LLM call cache and canonical-output cache), `logs/`, `data/lcb/`.

---

## 3. Setup

Python ≥ 3.9. With an OpenAI-compatible API no third-party package is needed; see `requirements.txt` for optional ones.

```bash
git clone https://github.com/zaiekd/allocate-where-it-matters.git && cd allocate-where-it-matters
cp .env.template .env        # then put your key after API_KEY=
python3 scripts/check_api.py # should print [OK] and think=0
```

Key fields of `.env`:

```ini
PROVIDER=openai          # openai = any OpenAI-compatible API; gemini = Google Gemini
MODEL=deepseek-flash
API_KEY=sk-...
BASE_URL=https://api.deepseek.com
THINKING_LEVEL=disabled  # DeepSeek must use "disabled"; Gemini: low/medium/high
PRICE_IN=0.28            # USD per 1M tokens, cost accounting only
PRICE_OUT=0.42
MAX_CONCURRENCY=16       # lower it if you hit 429 rate limits
```

To keep several model configurations side by side, put each in its own file and select it with `ENV_FILE`:

```bash
ENV_FILE=.env.gpt41mini python3 scripts/build_pool.py --benchmark humaneval --out-dir pools/humaneval/gpt41mini_full
```

`EXTRA_BODY` (JSON) is merged into every OpenAI-compatible request, e.g. for OpenRouter provider pinning.

> **Thinking mode matters.** If a model reasons by default, `temperature` is ignored and thinking tokens consume
> `max_tokens`, which truncates strategy lists and sketches and makes greedy and sampled outputs equivalent.
> Disable thinking for these experiments; `check_api.py` warns when thinking tokens are present, and
> `inspect_pool.py` reports strategy-parse rate, truncation rate per stage and thinking tokens per call.

### LiveCodeBench data

Download `test5.jsonl` and `test6.jsonl` from the
[`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) dataset into
`data/lcb/`. The loader keeps LeetCode problems with functional tests and a single-method `Solution` class (no
linked-list / tree I/O), ordered newest first. Optional filters: `LCB_FILES`, `LCB_SINCE=YYYY-MM-DD`,
`LCB_DIFFICULTY=medium,hard`. `python3 scripts/lcb_stats.py` summarises what is kept.

---

## 4. Building candidate pools

```bash
# pipeline test without any API call (mock LLM; never use mock pools for results)
python3 scripts/build_pool.py --benchmark humaneval --n-problems 10 --mock

# 10-problem smoke test with the real API, followed by a replay of the original configurations
bash scripts/run_test10.sh

# full benchmarks, in the background
mkdir -p logs
nohup python3 scripts/build_pool.py --benchmark humaneval --parallel 3 > logs/he.log 2>&1 &
nohup python3 scripts/build_pool.py --benchmark mbpp      --parallel 3 > logs/mbpp.log 2>&1 &
nohup python3 scripts/build_pool.py --benchmark lcb       --parallel 3 > logs/lcb.log 2>&1 &

# sanity check of a pool
python3 scripts/inspect_pool.py pools/humaneval/<pool-name>
```

- **Resumable**: finished problems are skipped; LLM calls are cached in SQLite, so re-running a failed problem does
  not pay again. `--force` rebuilds from the cache.
- **Options**: `--n-problems N`, `--start S`, `--task-ids HumanEval/0 HumanEval/5`, `--K --M --N-flat` (pool size),
  `--max-tokens` (default 2048), `--parallel` (problems in flight), `--max-concurrency`, `--out-dir`.
- **Fatal API errors** (401 bad key, 402 insufficient balance, 404 bad model) stop the run immediately instead of retrying.

---

## 5. Ground truth

The original Sketch-and-Verify code evaluated "HumanEval+" with the original HumanEval `check()` (about 7 asserts per
problem), not the ~1000 EvalPlus inputs. We therefore record three ground truths for every candidate:

| Field | Meaning | Use |
|---|---|---|
| `gt_orig` | original tests (HumanEval `check()` / the 3 MBPP asserts; LCB: public tests) | comparison with the original paper's numbers |
| `gt_base` | EvalPlus base inputs, compared with the canonical solution's outputs | |
| `gt_plus` | EvalPlus base + plus inputs (LCB: public + private tests) | **main results** |

`gt_base` / `gt_plus` follow the EvalPlus comparison rules: `atol` tolerance for floats, the special polynomial-root
check for HumanEval/32, and per-input timeouts scaled from the canonical solution's runtime. All 542 canonical solutions
(164 HumanEval+, 378 MBPP+) are judged correct under all three ground truths.

---

## 6. Faithfulness to the original implementation

**Identical**: all prompts (verbatim); temperatures (strategies 0.7, sketches 0.4, fills 0.8, flat 0.8, greedy 0);
`max_tokens`; the "different fill" prompt on every third fill; sketch validity checks (1–15 `??` holes and a function
definition; invalid sketches are not filled); code-block extraction; test-input generation (10 categories × 5);
fingerprint definition (floats rounded to 6 decimals, exception type only, timeouts recorded separately); semantic-voting
selection (prefer all-ok clusters → largest cluster → shortest program).

**Intentional changes** (documented in the code):

1. **Public examples.** The original pipeline passed no examples, so "passes examples" was always true. We extract them
   from the problem statement: 143/164 HumanEval+ problems (418 examples) and 378/378 MBPP+ problems. The canonical
   solution fails the stated examples on only 3 HumanEval+ problems (HE/47, HE/130, HE/148), due to errors in the
   problem statements.
2. **Imports and helper functions from the prompt are prepended at execution time** (e.g. `from typing import List`),
   so a model that omits an import is not marked wrong.
3. **One subprocess per candidate** runs all inputs (the original spawned one per input). Fingerprint semantics are
   unchanged and it is much faster. `PYTHONHASHSEED=0` makes set/str ordering deterministic.
4. **Thinking tokens are recorded separately** (Gemini `thoughts_token_count`, OpenAI-compatible `reasoning_tokens`)
   and billed at the output price.
5. The original K = 2 configuration asks the model for 2 strategies; on replay, K = 2 takes the first 2 of the 10 listed
   strategies. Adaptive methods list `K_max` strategies up front and expand them in order, so this is exactly the
   setting under study.

---

## 7. Policies (`pool/policies.py`)

Policies never call the API. They request candidates from a `Session`, which hands out pool items in generation order
and records which calls were paid for:

| Call | Meaning | Cost charged |
|---|---|---|
| `s.sketch(j)` | generate sketch j | the strategy-listing call (once) + that sketch's call |
| `s.fill(k)` | next fill of sketch k | that fill's call |
| `s.flat()` / `s.greedy()` | next flat / greedy sample | that call |
| `s.need_inputs()` | first use of fingerprints | all test-input generation calls |

A policy's token count is therefore the sum of exactly the calls it used, which makes comparisons with fixed
configurations fair. Sketch order can be permuted (`--perm-seeds`, deterministic, seed 0 = pool order) to average out
the effect of strategy order; fills within a sketch always stay in order.

**Selectors.** `sv` is the original semantic vote. `sv_ex` first keeps candidates that pass the public examples, then
votes (falling back to `sv` if none pass). The analysis scripts use the same selector for every method, so methods
differ only in *what they generate*.

### BAS (`BreadthSketch`)

- **Phase 0 — gate.** `gate_n = 3` flat samples.
  `"examples"` (BAS-Ex): easy if all pass the public examples.
  `"disagree"` (BAS-Dis): hard if any fails an example; otherwise generate one sketch with `probe_fills` fills and
  compare fingerprints on cheap "lite" inputs (one per input category, ≈18% of the input-generation cost); hard if an
  example-passing sketch fill behaves differently from the flat samples, or the flat samples disagree among themselves.
  `"off"`: every problem is hard.
- **Phase 1 — breadth.** Generate sketches in order with `m_breadth` fill(s) each, up to `K_max`. Optional early stop
  when the top fingerprint cluster is supported by ≥ `stop_support` distinct strategies and holds ≥ `stop_tau` of the
  valid candidates.
- **Phase 2 — refill (optional).** Sketches whose fills all fail the examples get up to `refill` extra fills.

Breadth is the default because, in our coverage analysis, coverage grows with the number of distinct strategies `K`
and hardly with fills per sketch `M`.

### GFS (`GatedFlat`)

Same gate options (`"examples"`, the default used in the thesis; `"fp-lite"`; `"none"`), but a hard problem receives
`n_extra` additional flat samples instead of sketches. With the example gate, GFS and BAS-Ex make identical gate
decisions and return the same program on every problem answered at the gate, so they differ only on escalated problems.

### SAS — sequential adaptive sketching (`AdaptiveSketch`)

Gate (`gate_signal="fp"` or `"examples"`) → sketches one at a time with `m0 = 2` fills each, pruning sketches whose
fills all fail the examples, and stopping when the top cluster has size ≥ `min_size`, share ≥ `tau` and support from
≥ `min_support` distinct sketches → successive halving of fills among surviving sketches (2 → 4 → 8 → 10). Each
mechanism can be switched off for ablations; `budget` caps the candidates per problem.

---

## 8. Running the analyses

All analyses read saved pools and make no API calls.

```bash
# main results per pool: pass@1 / coverage / tokens by difficulty, paired bootstrap
python3 scripts/analyze_pool.py pools/lcb/<pool-name> --perm-seeds 10
python3 scripts/plot_models.py results/lcb__<pool-a> results/lcb__<pool-b> results/lcb__<pool-c> --out results/lcb_models.png

# generation-side coverage (oracle selector upper bound)
python3 scripts/plot_coverage.py pools/humaneval/<pool-name>

# ablations and hyperparameter sensitivity over the default pools
python3 scripts/ablation.py
python3 scripts/hparam_sensitivity.py

# paired-bootstrap comparisons used in the paper
python3 scripts/basdis_vs_greedy.py
python3 scripts/paper_vs_original.py
python3 scripts/paper_filter_step.py

# full policy sweep with a dev/test split and figures
python3 scripts/run_policies.py pools/humaneval/<pool-name> --split 0.5 --perm-seeds 5
python3 scripts/plot_policies.py results/humaneval__<pool-name>
```

`run_policies.py` options: `--gt gt_plus|gt_base|gt_orig`, `--selector sv_ex|sv`, `--grid small|full`, `--split`
(dev fraction: each method family selects its Pareto-front configurations on dev problems and is reported on test
problems). It writes `summary.csv` (accuracy with bootstrap 95% CI, hard-subset accuracy, mean tokens / $ / calls /
candidates, K used, stop reasons, Pareto flag), `per_problem.jsonl` (chosen candidate and a trace of each decision),
`K_hist.txt`, `cost_accuracy.png` and `budget_hist.png`.

**Output locations** (all under `results/`, created on demand):

| Path | Written by |
|---|---|
| `results/<benchmark>__<pool>/stats.{txt,json}` | `analyze_pool.py` (main tables) |
| `results/<benchmark>__<pool>/coverage.csv`, `coverage_tokens.png` | `plot_coverage.py` |
| `results/<benchmark>__<pool>/summary.csv`, `per_problem.jsonl`, `*.png` | `run_policies.py`, `plot_policies.py` |
| `results/ablations/` | `ablation.py` (`summary.md`, per-pool tables, raw JSON) |
| `results/hparams/` | `hparam_sensitivity.py` |

The paired-bootstrap scripts print to stdout. Candidate pools and results are not included in this repository;
build the pools with `scripts/build_pool.py` (Section 4) and run the analyses above to reproduce the results.

---

## 9. Pool file format (one JSON per problem)

```jsonc
{
  "task_id": "HumanEval/0", "benchmark": "humaneval",
  "provider": "openai", "model": "deepseek-flash", "thinking_level": null,
  "price_per_1m": [0.28, 0.42],
  "config": {"K": 10, "M": 10, "N_flat": 100, ...},
  "examples": ["assert __ex_eq(has_close_elements([1.0, 2.0, 3.0], 0.5), False)", ...],
  "test_inputs": {"inputs": ["([1.0, 2.0], 0.5)", ...], "calls": [<usage>...], "fallback": false},
  "categories": {"names": ["sorting + adjacent scan", ...], "parse_ok": true, "call": <usage>},
  "sketches": [
    {"k": 0, "category": "...", "code": "def f(...): ... ?? ...", "num_holes": 5,
     "valid": true, "errors": [], "parses": true, "call": <usage>,
     "fills": [{"m": 0, "kind": "standard", "cid": "s0f0", "call": <usage>}, ...]}
  ],
  "greedy": {"cid": "g", "call": <usage>},
  "flat":   [{"n": 0, "cid": "f0", "call": <usage>}, ...],
  "candidates": {
    "s0f0": {"code": "...", "code_len": 512, "code_hash": "...",
             "compiles": true, "load_error": null,
             "examples": ["pass", "fail"], "examples_pass": false,
             "fp": ["ok:97cdbdc7feff827e", "E:IndexError", ...], "all_ok": false,
             "gt_orig": true, "gt_base": true, "gt_plus": false}
  },
  "cost": {"test_inputs": {...}, "categories": {...}, "sketches": {...},
           "fills": {...}, "greedy": {...}, "flat": {...}},
  "timing": {"gen_s": 95.3, "exec_s": 12.1, "n_unique_programs": 87}
}
```

`<usage>` = `{"in_tok", "out_tok", "think_tok", "finish", "latency_s", "cached"}`.
Candidate ids: `s{k}f{m}` = fill m of sketch k, `g` = greedy sample, `f{n}` = flat sample n.

---

## 10. Tests

```bash
python3 tests/test_policies.py
```

The tests use small synthetic pools (two additional checks run on real pools when `pools/` exists). They cover:
fixed configurations implemented with `Session` match `replay.py` problem by problem; all `AdaptiveSketch` mechanisms
off equals fixed K = 10, M = 10, and BAS with the gate off equals fixed K = 10, M = 1; easy problems return at the gate
and an examples-only gate never pays for test inputs; early stopping requires cross-strategy support; invalid sketches
are not filled and failing sketches are pruned; budgets are respected; successive halving concentrates fills; GFS
gates; and sketch permutations keep fills in order.

---

## Acknowledgements

Prompts, fingerprinting and semantic-voting selection follow the original Sketch-and-Verify implementation.
HumanEval+ and MBPP+ data are the official [EvalPlus](https://github.com/evalplus/evalplus) releases;
LiveCodeBench data comes from [LiveCodeBench](https://livecodebench.github.io/).
