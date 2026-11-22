#!/usr/bin/env python3
"""Check that the model configured in .env works, using one tiny API call.

python scripts/check_api.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from build_pool import env_file, load_env  # noqa: E402
from pool.llm import LLM  # noqa: E402


def main() -> None:
    load_env()
    print(f"config file: {os.path.relpath(env_file())}")
    env = os.environ.get
    provider, model = env("PROVIDER", "gemini"), env("MODEL", "")
    print(f"provider={provider} model={model} base_url={env('BASE_URL') or '(default)'} "
          f"key={'set' if env('API_KEY') else 'MISSING'}")
    price = (float(env("PRICE_IN")), float(env("PRICE_OUT"))) if env("PRICE_IN") and env("PRICE_OUT") else None
    llm = LLM(model, provider=provider, thinking_level=env("THINKING_LEVEL") or None,
              api_key=env("API_KEY") or None, base_url=env("BASE_URL") or None,
              price_per_1m=price, cache_path=None, max_retries=1)
    try:
        rec = llm.generate("Write a Python function add(a, b) that returns a + b. Output only code.",
                           temperature=0.0, max_tokens=200)
    except Exception as e:
        msg = str(e)
        print("[FAIL] API call failed:", msg[:400])
        if "CERTIFICATE_VERIFY_FAILED" in msg:
            print("   Hint: the python.org build of Python on macOS lacks root certificates. Run\n"
                  "   /Applications/Python\\ 3.*/Install\\ Certificates.command\n"
                  "   or pip install certifi, then retry.")
        elif "401" in msg:
            print("   Hint: invalid API key; check API_KEY in .env.")
        elif "402" in msg:
            print("   Hint: insufficient account balance; top up with your provider.")
        elif "404" in msg:
            print("   Hint: MODEL or BASE_URL is wrong.")
        sys.exit(1)
    print("[OK] API call succeeded")
    print("   reply:", rec.text.strip()[:200].replace("\n", "\n          "))
    print(f"   tokens: in={rec.in_tok} out={rec.out_tok} think={rec.think_tok}  "
          f"cost=${llm.cost(rec.in_tok, rec.out_tok, rec.think_tok):.6f}  latency={rec.latency_s:.1f}s")
    print(f"   thinking={llm.thinking_level or '(server default)'}  finish={rec.finish}")
    if rec.think_tok > 0:
        print("[WARN] The model is thinking (think>0). In thinking mode temperature is ignored and\n"
              "   thinking tokens count against max_tokens. For DeepSeek set THINKING_LEVEL=disabled in .env.")


if __name__ == "__main__":
    main()
