"""
llm_client.py -- one small, dependency-free client for the free LLM tiers (Gemini or Groq, both OpenAI-compatible).

  chat_json(system, user)  ->  (parsed_json_dict, meta)

Rules this file enforces
  * The API key comes from an environment variable (GEMINI_API_KEY or GROQ_API_KEY). Never from code, never printed.
  * temperature 0 so the same note gives the same answer.
  * Every answer is cached on disk, keyed by provider + model + prompt. The demo can replay with no internet:
    set LLM_CACHE_ONLY=1 and an uncached call raises LLMUnavailable instead of calling out.
  * Calls are paced and retried (rate limit 429 / server errors) so free-tier caps do not break a run.
  * No patient data is sent except the synthetic text you pass in. Do not use a free tier with real patient data.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import config as C


def _load_dotenv():
    """Read KEY=VALUE lines from a .env file next to this file into os.environ (no extra package).
    A variable already set in the real environment wins. Values are never printed."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()


class LLMUnavailable(RuntimeError):
    """No key, no network, rate limit exhausted, or cache-only mode with an uncached prompt."""


_last_call = [0.0]


def settings():
    provider = os.environ.get("LLM_PROVIDER", C.LLM_DEFAULT_PROVIDER).lower()
    if provider not in C.LLM_PROVIDERS:
        raise LLMUnavailable(f"unknown LLM_PROVIDER {provider!r}; choose one of {list(C.LLM_PROVIDERS)}")
    p = C.LLM_PROVIDERS[provider]
    return {"provider": provider, "model": os.environ.get("LLM_MODEL", p["default_model"]),
            "base_url": p["base_url"], "key_env": p["key_env"], "key": os.environ.get(p["key_env"], "")}


def status():
    """For the app sidebar: is a key present, and which model? Never returns the key."""
    s = settings()
    return {"provider": s["provider"], "model": s["model"], "key_present": bool(s["key"]),
            "cache_only": os.environ.get("LLM_CACHE_ONLY") == "1"}


def _load_cache():
    try:
        with open(C.LLM_CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(C.LLM_CACHE_PATH) or ".", exist_ok=True)
    tmp = C.LLM_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, C.LLM_CACHE_PATH)


def parse_json_loose(text):
    """Models sometimes wrap JSON in code fences or add a sentence. Take the outermost {...}."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except ValueError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i:j + 1])
        raise


def _post(url, key, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          "Accept": "application/json", "User-Agent": "halvian-mvp/1.0"})
    with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT_S) as r:
        return json.loads(r.read().decode())


def chat_json(system, user, use_cache=True):
    s = settings()
    cache_key = hashlib.sha256(json.dumps([s["provider"], s["model"], system, user]).encode()).hexdigest()
    cache = _load_cache()
    if use_cache and cache_key in cache:
        return cache[cache_key]["answer"], {"source": "cache", "provider": s["provider"], "model": s["model"]}
    if os.environ.get("LLM_CACHE_ONLY") == "1":
        raise LLMUnavailable("LLM_CACHE_ONLY=1 and this prompt is not in the cache")
    if not s["key"]:
        raise LLMUnavailable(f"no API key: set the {s['key_env']} environment variable")

    payload = {"model": s["model"], "temperature": 0,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}}
    last_err = None
    for attempt in range(C.LLM_MAX_RETRIES + 1):
        gap = C.LLM_MIN_SECONDS_BY_PROVIDER.get(s["provider"], C.LLM_MIN_SECONDS_BETWEEN_CALLS)
        wait = gap - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            out = _post(s["base_url"].rstrip("/") + "/chat/completions", s["key"], payload)
            answer = parse_json_loose(out["choices"][0]["message"]["content"])
            cache[cache_key] = {"answer": answer, "provider": s["provider"], "model": s["model"]}
            _save_cache(cache)
            return answer, {"source": "live", "provider": s["provider"], "model": s["model"]}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 400 and "response_format" in payload:      # some models reject JSON mode: ask in the prompt instead
                payload.pop("response_format")
                continue
            if e.code in (401, 403):
                try:
                    body = e.read().decode(errors="replace")[:300].replace(s["key"], "***")
                except Exception:
                    body = ""
                why = ("the key is invalid or revoked" if e.code == 401 else
                       "access was refused: the request was blocked (often Cloudflare, or a region the provider does not serve) or the key lacks access")
                raise LLMUnavailable(f"HTTP {e.code}: {why}. Server said: {body!r}") from None
            if e.code == 404:
                raise LLMUnavailable(f"model {s['model']!r} not found for {s['provider']}; set LLM_MODEL") from None
            try:
                ra = float(e.headers.get("Retry-After", 0))
            except (TypeError, ValueError):
                ra = 0
            time.sleep(min(60, max(ra, 4 * 2 ** attempt)))           # 429 or 5xx: back off and retry
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            last_err = type(e).__name__
            time.sleep(min(30, 2 * 2 ** attempt))
    raise LLMUnavailable(f"LLM call failed after retries ({last_err})")
