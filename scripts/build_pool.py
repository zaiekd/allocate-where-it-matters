#!/usr/bin/env python3
"""Generate the offline candidate pool (one JSON per problem, resumable).

Examples
--------
# 10-problem smoke test without API (mock LLM):
python scripts/build_pool.py --benchmark humaneval --n-problems 10 --mock

# real run with the model configured in .env (PROVIDER / MODEL / API_KEY / BASE_URL):
python scripts/build_pool.py --benchmark humaneval --n-problems 10

# full benchmark, in the background:
nohup python scripts/build_pool.py --benchmark humaneval --parallel 3 > logs/he_lite.log 2>&1 &

Output: pools/<benchmark>/<model>__<thinking>/<task_id>.json
Re-running skips finished problems; LLM calls are cached in cache/llm_*.sqlite
so a crashed problem is regenerated without paying again.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pool.benchmarks import load_benchmark, safe_task_id  # noqa: E402
from pool.builder import PoolConfig, build_problem  # noqa: E402
from pool.llm import LLM, FatalAPIError  # noqa: E402


def env_file() -> str:
    """The config file to read: $ENV_FILE if set (absolute, or relative to the
    project root), else .env.  Lets each model keep its own file, e.g.
    ENV_FILE=.env.qwen3coder python3 scripts/build_pool.py ..."""
    f = os.environ.get("ENV_FILE") or ".env"
    return f if os.path.isabs(f) else os.path.join(ROOT, f)


def load_env() -> None:
    path = env_file()
    if os.environ.get("ENV_FILE") and not os.path.exists(path):
        raise SystemExit(f"ENV_FILE={os.environ['ENV_FILE']} not found ({path})")
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
    except ImportError:
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="humaneval", choices=["humaneval", "mbpp", "lcb"])
    ap.add_argument("--provider", default=None, help="gemini | openai (default: PROVIDER in .env)")
    ap.add_argument("--model", default=None, help="default: MODEL in .env")
    ap.add_argument("--thinking-level", default=None, help="low|medium|high|none (default: THINKING_LEVEL in .env)")
    ap.add_argument("--n-problems", type=int, default=None, help="first N problems")
    ap.add_argument("--start", type=int, default=0, help="skip the first START problems")
    ap.add_argument("--task-ids", nargs="*", default=None, help="explicit task ids")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--N-flat", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="output cap for sketches, fills, greedy and flat samples (original code: 2048)")
    ap.add_argument("--parallel", type=int, default=2, help="problems in flight")
    ap.add_argument("--max-concurrency", type=int, default=None, help="concurrent API requests (default: MAX_CONCURRENCY in .env or 8)")
    ap.add_argument("--exec-workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mock", action="store_true", help="offline mock LLM (pipeline test only)")
    ap.add_argument("--force", action="store_true", help="rebuild even if pool file exists")
    args = ap.parse_args()

    load_env()
    problems = load_benchmark(args.benchmark)
    if args.task_ids:
        want = set(args.task_ids)
        problems = [p for p in problems if p.task_id in want]
    else:
        problems = problems[args.start:]
        if args.n_problems is not None:
            problems = problems[: args.n_problems]

    env = os.environ.get
    provider = args.provider or env("PROVIDER", "gemini")
    model = args.model or env("MODEL", "gemini-3.1-flash-lite-preview")
    thinking = args.thinking_level if args.thinking_level is not None else env("THINKING_LEVEL", "")
    thinking = thinking or None  # "disabled" is a real value for OpenAI-compatible providers
    max_conc = args.max_concurrency or int(env("MAX_CONCURRENCY", "8"))
    price = None
    if env("PRICE_IN") and env("PRICE_OUT"):
        price = (float(env("PRICE_IN")), float(env("PRICE_OUT")))
    cache_dir = env("CACHE_DIR") or os.path.join(ROOT, "cache")
    if args.mock:
        from pool.mock import MockLLM

        all_probs = load_benchmark(args.benchmark)
        llm = MockLLM(all_probs, cache_path=os.path.join(cache_dir, "llm_mock.sqlite"),
                      max_concurrency=max_conc)
        tag = "mock"
    else:
        llm = LLM(model, provider=provider, thinking_level=thinking,
                  api_key=env("API_KEY") or None, base_url=env("BASE_URL") or None,
                  price_per_1m=price,
                  cache_path=os.path.join(cache_dir, f"llm_{model.replace('/', '_')}.sqlite"),
                  max_concurrency=max_conc)
        tag = f"{model.replace('/', '_')}__{thinking or 'default'}"

    out_dir = args.out_dir or os.path.join(ROOT, "pools", args.benchmark, tag)
    os.makedirs(out_dir, exist_ok=True)
    cfg = PoolConfig(K=args.K, M=args.M, N_flat=args.N_flat, max_tokens=args.max_tokens)

    todo = []
    for p in problems:
        path = os.path.join(out_dir, safe_task_id(p.task_id) + ".json")
        if os.path.exists(path) and not args.force:
            continue
        todo.append((p, path))
    print(f"[pool] {args.benchmark} provider={llm.provider} model={llm.model} thinking={llm.thinking_level} "
          f"K={cfg.K} M={cfg.M} N_flat={cfg.N_flat} -> {out_dir}", flush=True)
    print(f"[pool] {len(problems)} problems selected, {len(problems) - len(todo)} done, "
          f"{len(todo)} to build", flush=True)

    failures = []
    t0 = time.time()
    total_usd = 0.0
    exec_pool = cf.ThreadPoolExecutor(max_workers=args.exec_workers)

    def work(item):
        p, path = item
        res = build_problem(p, llm, cfg, exec_pool)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, path)
        return p, res

    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(work, it): it[0] for it in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            p = futs[fut]
            try:
                _, res = fut.result()
                usd = sum(v["usd"] for v in res["cost"].values())
                total_usd += usd
                c = res["candidates"]
                sk = [v for k, v in c.items() if k.startswith("s")]
                fl = [v for k, v in c.items() if k.startswith("f")]
                rate = lambda xs: (sum(x["gt_plus"] for x in xs) / len(xs)) if xs else 0.0  # noqa: E731
                print(
                    f"[{i}/{len(todo)}] {p.task_id}: greedy={'✓' if c['g']['gt_plus'] else '✗'} "
                    f"sketch_fill_acc={rate(sk):.2f} ({len(sk)}) flat_acc={rate(fl):.2f} "
                    f"inputs={len(res['test_inputs']['inputs'])} ex={len(res['examples'])} "
                    f"${usd:.4f} gen={res['timing']['gen_s']}s exec={res['timing']['exec_s']}s",
                    flush=True,
                )
            except FatalAPIError as e:
                print(f"[pool] FATAL API error, stopping: {e}", flush=True)
                for f in futs:
                    f.cancel()
                exec_pool.shutdown(cancel_futures=True)
                sys.exit(2)
            except Exception as e:
                failures.append(p.task_id)
                print(f"[{i}/{len(todo)}] {p.task_id}: FAILED {e}", flush=True)
                traceback.print_exc()
    exec_pool.shutdown()

    print(f"[pool] done in {time.time() - t0:.0f}s; api_calls={llm.n_api_calls} "
          f"cache_hits={llm.n_cache_hits} pool_cost(incl. cached)≈${total_usd:.4f}", flush=True)
    if failures:
        print(f"[pool] {len(failures)} failed (re-run the same command to retry): {failures}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
