"""
orchestrator.py
===============
Multi-model / multi-strategy solving layer consumed by run_eval.py.

Two-phase interface:
    plan(strategy, prompt) -> (plan_text_or_None, events)
    code(strategy, original_prompt, plan_text, prev_code, failure) -> attempt dict

Every model call is a "provider/model" string (e.g. "deepseek/deepseek-v4-flash")
resolved against PROVIDERS to find which base_url/api_key to use. See
docs/superpowers/specs/2026-07-01-multi-model-orchestrator-design.md for the
full design.
"""
from __future__ import annotations

import os
import re
import time


class ConfigError(Exception):
    """Raised when STRATEGY / role model / provider configuration is invalid."""


STRATEGIES = ("single", "analyze-then-code", "debate")

PROVIDERS = {
    "deepseek": {"base_url_env": "DEEPSEEK_BASE_URL", "api_key_env": "DEEPSEEK_API_KEY"},
    "qwen":     {"base_url_env": "QWEN_BASE_URL",     "api_key_env": "QWEN_API_KEY"},
    "mimo":     {"base_url_env": "MIMO_BASE_URL",     "api_key_env": "MIMO_API_KEY"},
    "gemma":    {"base_url_env": "GEMMA_BASE_URL",    "api_key_env": "GEMINI_API_KEY"},
}

_ROLE_ENV = {
    "single": {"single": "SINGLE_MODEL"},
    "analyze-then-code": {"architect": "ARCHITECT_MODEL", "coder": "CODER_MODEL"},
    "debate": {
        "debater1": "DEBATER1_MODEL",
        "debater2": "DEBATER2_MODEL",
        "judge": "JUDGE_MODEL",
        "coder": "CODER_MODEL",
    },
}


def _resolve_role_model(role_model: str) -> tuple:
    """Split "provider/model" and look up its base_url/api_key.
    Returns (provider, model, base_url, api_key). Raises ConfigError."""
    if "/" not in role_model:
        raise ConfigError(
            f"Model {role_model!r} must be formatted as 'provider/model' "
            f"(e.g. 'deepseek/deepseek-v4-flash')"
        )
    provider, model = role_model.split("/", 1)
    if provider not in PROVIDERS:
        raise ConfigError(
            f"Unknown provider {provider!r} in {role_model!r}; "
            f"known providers: {sorted(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    base_url = os.environ.get(cfg["base_url_env"])
    api_key = os.environ.get(cfg["api_key_env"])
    if not base_url or not api_key:
        raise ConfigError(
            f"Provider {provider!r} needs {cfg['base_url_env']} and "
            f"{cfg['api_key_env']} set (used by model {role_model!r})"
        )
    return provider, model, base_url, api_key


def active_role_models(strategy: str) -> dict:
    """Return the role -> 'provider/model' mapping configured for strategy."""
    if strategy not in STRATEGIES:
        raise ConfigError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    out = {}
    for role, env_name in _ROLE_ENV[strategy].items():
        role_model = os.environ.get(env_name)
        if not role_model:
            raise ConfigError(f"STRATEGY={strategy} requires {env_name} to be set (role: {role})")
        out[role] = role_model
    return out


def validate_config(strategy: str) -> None:
    """Fail fast if strategy or any of its role models/providers are misconfigured."""
    role_models = active_role_models(strategy)
    for role_model in role_models.values():
        _resolve_role_model(role_model)


_API_RETRY_ATTEMPTS = 3
_API_RETRY_BACKOFF = [5, 15]  # seconds to wait before the 2nd and 3rd attempt

_clients = {}


def _get_client(base_url: str, api_key: str):
    key = (base_url, api_key)
    if key not in _clients:
        from openai import OpenAI
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[key]


def _call_model_raw(role_model: str, prompt: str) -> str:
    """One uncached, unretried call to role_model. Split out from _chat so
    tests can monkeypatch it without a real OpenAI client or network access."""
    _, model, base_url, api_key = _resolve_role_model(role_model)
    client = _get_client(base_url, api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _call_with_retry(role_model: str, prompt: str) -> str:
    """Call _call_model_raw, retrying on transient errors (500/503/connection)."""
    last_exc = None
    for i in range(_API_RETRY_ATTEMPTS):
        try:
            return _call_model_raw(role_model, prompt)
        except Exception as exc:
            msg = str(exc)
            if not any(code in msg for code in ("500", "503", "Connection")):
                raise
            last_exc = exc
            if i < len(_API_RETRY_BACKOFF):
                wait = _API_RETRY_BACKOFF[i]
                print(f"        [api-retry {i + 1}/{_API_RETRY_ATTEMPTS - 1}] "
                      f"{msg[:80]} — retrying in {wait}s…")
                time.sleep(wait)
    raise last_exc


def _chat(role_model: str, prompt: str) -> tuple:
    """Call role_model with prompt. Returns (reply_text, duration_s)."""
    start = time.monotonic()
    reply = _call_with_retry(role_model, prompt)
    return reply, time.monotonic() - start


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the program out of a model reply. Prefer a ```python fenced block;
    if several, take the longest; fall back to the raw text."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def _build_coder_prompt(original_prompt: str, plan_text, prev_code, failure) -> str:
    parts = [original_prompt]
    if plan_text:
        parts.append(
            "\n\nA solution approach has been proposed for this problem:\n"
            f"{plan_text}\n\n"
            "Implement it as a complete Python program. Enclose the code in a "
            "```python fenced block.\n"
        )
    if prev_code is not None and failure is not None:
        parts.append(
            "\n\nYour previous attempt below failed on a test case. Fix the bug and "
            "return a corrected, complete Python program. Enclose the code in a "
            "```python fenced block.\n\n"
            f"Previous code:\n```python\n{prev_code}\n```\n\n"
            f"Failing test #{failure['index']} ({failure['type']}):\n"
            f"Input:\n{failure['input']}\n"
            f"Expected output:\n{failure['expected']}\n"
            f"Your code produced:\n{failure['got']}\n"
        )
    return "".join(parts)


def code(strategy: str, original_prompt: str, plan_text=None, prev_code=None, failure=None) -> dict:
    """Run the coder step once.
    Returns {"model", "prompt", "reply", "code", "duration_s"}."""
    role_models = active_role_models(strategy)
    role_model = role_models.get("coder", role_models.get("single"))
    prompt = _build_coder_prompt(original_prompt, plan_text, prev_code, failure)
    reply, duration = _chat(role_model, prompt)
    return {
        "model": role_model,
        "prompt": prompt,
        "reply": reply,
        "code": extract_code(reply),
        "duration_s": duration,
    }
