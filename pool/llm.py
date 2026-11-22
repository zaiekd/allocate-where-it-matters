"""LLM client that returns per-call token usage.

Differences from sketch-research/core/llm_client.py:
  * every call returns a CallRecord (text + input/output/thinking tokens),
    so the pool can attribute cost to each individual sketch / fill / sample
    (the original only kept a global, thread-unsafe running total);
  * thinking tokens (Gemini `thoughts_token_count`) are recorded separately
    (they are billed as output tokens);
  * a thread-safe SQLite cache keyed on
    (model, thinking, temperature, max_tokens, prompt, sample_idx), so an
    interrupted run resumes without paying again;
  * a global semaphore caps concurrent API requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

# Pricing per 1M tokens (input, output) — same table as core/utils.py.
# Thinking tokens are billed at the output rate.
MODEL_PRICING = {
    # DeepSeek list prices change — check https://api-docs.deepseek.com/quick_start/pricing
    # and override with PRICE_IN / PRICE_OUT in .env
    "deepseek-flash": (0.28, 0.42),
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    "gemini-3.1-flash-lite-preview": (0.075, 0.30),
    "gemini-3-flash-preview": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "mock": (0.075, 0.30),
}


def cost_usd(model: str, in_tok: int, out_tok: int, think_tok: int) -> float:
    pin, pout = MODEL_PRICING.get(model, (0.0, 0.0))
    return (in_tok * pin + (out_tok + think_tok) * pout) / 1_000_000


@dataclass
class CallRecord:
    text: str
    in_tok: int = 0
    out_tok: int = 0
    think_tok: int = 0
    finish: str = ""
    latency_s: float = 0.0
    cached: bool = False

    def usage(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


class FatalAPIError(RuntimeError):
    """Non-retryable API error (bad key, no balance, unknown model)."""


class _Cache:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:  # some network/virtual file systems
            pass
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._db.commit()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM calls WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            self._db.commit()


class LLM:
    """Gemini client (google-genai SDK).

    `generate(prompt, temperature, max_tokens, sample_idx)` returns a
    CallRecord.  `sample_idx` distinguishes repeated i.i.d. samples of the
    same prompt in the cache.
    """

    def __init__(
        self,
        model: str,
        provider: str = "gemini",
        thinking_level: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        price_per_1m: Optional[Tuple[float, float]] = None,
        cache_path: Optional[str] = None,
        max_concurrency: int = 8,
        max_retries: int = 8,
    ) -> None:
        """provider: "gemini" (google-genai SDK) or "openai" (any
        OpenAI-compatible endpoint: OpenAI, DeepSeek, Qwen/DashScope,
        OpenRouter, a local Ollama / vLLM server, ...)."""
        if provider not in ("gemini", "openai", "mock"):
            raise ValueError(f"unknown provider {provider!r} (use gemini | openai)")
        self.provider = provider
        self.model = model
        self.thinking_level = thinking_level or None
        self.api_key = api_key
        self.base_url = base_url or None
        self.price = tuple(price_per_1m) if price_per_1m else MODEL_PRICING.get(model, (0.0, 0.0))
        self.cache = _Cache(cache_path) if cache_path else None
        self.sem = threading.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self._client = None
        self._client_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.n_api_calls = 0
        self.n_cache_hits = 0

    def cost(self, in_tok: int, out_tok: int, think_tok: int) -> float:
        pin, pout = self.price
        return (in_tok * pin + (out_tok + think_tok) * pout) / 1_000_000

    # -- providers ----------------------------------------------------------
    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                if self.provider == "gemini":
                    from google import genai  # pip install google-genai

                    key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not key:
                        raise EnvironmentError("API_KEY not set — fill it in .env in the repository root")
                    self._client = genai.Client(api_key=key)
                else:
                    self._client = "urllib"
            return self._client

    def _call_api(self, prompt: str, temperature: float, max_tokens: int, sample_idx: int = 0) -> CallRecord:
        if self.provider == "gemini":
            return self._call_gemini(prompt, temperature, max_tokens)
        return self._call_openai(prompt, temperature, max_tokens)

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> CallRecord:
        """POST {base_url}/chat/completions with the standard library only
        (no `openai` package needed)."""
        import ssl
        import urllib.error
        import urllib.request

        key = self.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # DeepSeek-style thinking switch (https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/):
        #   THINKING_LEVEL=disabled      -> {"thinking": {"type": "disabled"}}  (temperature works)
        #   THINKING_LEVEL=low|high|max  -> thinking enabled + reasoning_effort  (temperature IGNORED)
        #   empty                        -> send nothing (server default; DeepSeek default = thinking ON)
        tl = (self.thinking_level or "").lower()
        if tl in ("disabled", "off", "none", "non-thinking"):
            body["thinking"] = {"type": "disabled"}
        elif tl:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = tl
        # EXTRA_BODY: JSON merged into the request, e.g. OpenRouter provider pinning
        #   EXTRA_BODY={"provider": {"order": ["DeepInfra"], "allow_fallbacks": false}}
        extra = os.environ.get("EXTRA_BODY", "").strip()
        if extra:
            try:
                body.update(json.loads(extra))
            except json.JSONDecodeError as e:
                raise FatalAPIError(f"EXTRA_BODY is not valid JSON: {e}") from None
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        ctx = None
        try:  # macOS python.org builds often lack root certs; certifi fixes that
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = None
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (400, 401, 402, 403, 404):
                # not transient: bad key / no balance / wrong model -> fail fast
                raise FatalAPIError(f"HTTP {e.code} from {url}: {detail}") from None
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        choices = data.get("choices") or []
        ch = choices[0] if choices else {}
        text = ((ch.get("message") or {}).get("content")) or ""
        u = data.get("usage") or {}
        completion = u.get("completion_tokens", 0) or 0
        think = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)) or 0
        return CallRecord(
            text=text, in_tok=u.get("prompt_tokens", 0) or 0,
            out_tok=max(completion - think, 0), think_tok=think,
            finish=str(ch.get("finish_reason") or ""), latency_s=time.time() - t0,
        )

    def _call_gemini(self, prompt: str, temperature: float, max_tokens: int) -> CallRecord:
        from google.genai import types

        kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
        if self.thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        config = types.GenerateContentConfig(**kwargs)
        t0 = time.time()
        try:
            resp = self._get_client().models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except Exception as e:  # google-genai APIError has .code; 402 = credits depleted, 401/403 = key, 404 = model
            code = getattr(e, "code", None)
            if code in (400, 401, 402, 403, 404):
                raise FatalAPIError(f"Gemini HTTP {code}: {str(e)[:300]}") from None
            raise
        text = ""
        try:
            text = resp.text or ""
        except Exception:  # blocked / empty candidates
            text = ""
        finish = ""
        try:
            finish = str(resp.candidates[0].finish_reason)
        except Exception:
            pass
        u = resp.usage_metadata
        return CallRecord(
            text=text,
            in_tok=(getattr(u, "prompt_token_count", 0) or 0) if u else 0,
            out_tok=(getattr(u, "candidates_token_count", 0) or 0) if u else 0,
            think_tok=(getattr(u, "thoughts_token_count", 0) or 0) if u else 0,
            finish=finish,
            latency_s=time.time() - t0,
        )

    # -- public -------------------------------------------------------------
    def cache_key(self, prompt: str, temperature: float, max_tokens: int, sample_idx: int) -> str:
        raw = json.dumps(
            [self.provider, self.base_url, self.model, self.thinking_level, temperature, max_tokens, sample_idx, prompt]
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int = 2048,
        sample_idx: int = 0,
    ) -> CallRecord:
        key = self.cache_key(prompt, temperature, max_tokens, sample_idx)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                with self.stats_lock:
                    self.n_cache_hits += 1
                rec = CallRecord(**hit)
                rec.cached = True
                return rec

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.sem:
                    rec = self._call_api(prompt, temperature, max_tokens, sample_idx)
                break
            except FatalAPIError:
                raise
            except Exception as exc:  # 429 / 503 / network
                last_exc = exc
                if attempt == self.max_retries:
                    raise
                delay = min(2.0 * (2 ** attempt), 90.0) * (0.5 + random.random())
                print(
                    f"  [llm] retry {attempt + 1}/{self.max_retries} in {delay:.0f}s: "
                    f"{str(exc)[:120]}",
                    flush=True,
                )
                time.sleep(delay)
        else:  # pragma: no cover
            raise last_exc  # type: ignore[misc]

        with self.stats_lock:
            self.n_api_calls += 1
        if self.cache is not None:
            d = asdict(rec)
            d["cached"] = False
            self.cache.put(key, d)
        return rec
