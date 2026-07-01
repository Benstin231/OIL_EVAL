# Multi-Model Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded single-model `generate()` hook in `run_eval.py` with a new `orchestrator.py` module that supports switchable models (DeepSeek, Qwen/Alibaba, MiMo, Gemma) across three solving strategies (`single`, `analyze-then-code`, `debate`), keeps the coder-only retry semantics, and writes a full per-run debug log.

**Architecture:** `orchestrator.py` exposes `plan(strategy, prompt)` and `code(strategy, original_prompt, plan_text, prev_code, failure)`. Every model call goes through a small provider registry (`PROVIDERS`) that maps a `provider/model` string to a `base_url`/`api_key` pair, then through one OpenAI-compatible client per provider. `run_eval.py` gains a `solve_problem()` function that calls `plan()` once per problem and `code()` once per attempt (initial + retries), and `main()` is rewired to use it, add `--strategy`, and write `logs/run_<id>.json` alongside the existing `results.json`.

**Tech Stack:** Python stdlib + `openai` (OpenAI-compatible client) + `python-dotenv` (existing) + `pytest` (new, dev-only).

**Spec:** `docs/superpowers/specs/2026-07-01-multi-model-orchestrator-design.md`

## Global Constraints

- Model access is OpenAI-compatible only, via `orchestrator.PROVIDERS` — no native per-provider SDKs, no OpenRouter/aggregator, no local Ollama/vLLM (spec Non-goals).
- Role models are configured as `provider/model` strings resolved against `PROVIDERS`; providers in scope: `deepseek`, `qwen`, `mimo`, `gemma`.
- Debate strategy is fixed at 2 debaters + 1 judge, exactly 2 rounds — not configurable (spec Non-goals).
- Retry only re-runs the coder step; `plan()`/debate never re-runs on a retry (spec: Retry integration).
- `results.json`'s existing summary shape is preserved; the new detailed log is a separate file under `logs/`.
- `.env` and `logs/` must never be committed — both can carry API keys or full prompts/replies.
- All new orchestrator/run_eval unit tests must run fully offline (monkeypatched `_chat`/`orchestrator.plan`/`orchestrator.code`) — never hit a real DeepSeek/Qwen/MiMo/Gemini endpoint, since real calls cost tokens/money.
- Runs on Windows (existing project constraint, stdlib subprocess-based test execution).

---

## Task 1: Provider registry & config validation

**Files:**
- Create: `orchestrator.py`
- Modify: `requirements.txt`
- Test: `tests/test_orchestrator_config.py`

**Interfaces:**
- Produces: `orchestrator.ConfigError` (exception), `orchestrator.STRATEGIES` (tuple), `orchestrator.PROVIDERS` (dict), `orchestrator._resolve_role_model(role_model: str) -> tuple[str, str, str, str]` (provider, model, base_url, api_key), `orchestrator.active_role_models(strategy: str) -> dict`, `orchestrator.validate_config(strategy: str) -> None`.

- [ ] **Step 1: Add new dependencies to `requirements.txt`**

Replace the full file contents:

```
google-genai
python-dotenv
openai
pytest
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_orchestrator_config.py`:

```python
import pytest

import orchestrator


def test_resolve_role_model_splits_provider_and_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    provider, model, base_url, api_key = orchestrator._resolve_role_model(
        "deepseek/deepseek-v4-flash"
    )
    assert provider == "deepseek"
    assert model == "deepseek-v4-flash"
    assert base_url == "https://api.deepseek.com"
    assert api_key == "sk-test"


def test_resolve_role_model_rejects_missing_slash():
    with pytest.raises(orchestrator.ConfigError, match="provider/model"):
        orchestrator._resolve_role_model("deepseek-v4-flash")


def test_resolve_role_model_rejects_unknown_provider():
    with pytest.raises(orchestrator.ConfigError, match="Unknown provider"):
        orchestrator._resolve_role_model("openai/gpt-5")


def test_resolve_role_model_rejects_missing_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(orchestrator.ConfigError, match="DEEPSEEK_BASE_URL"):
        orchestrator._resolve_role_model("deepseek/deepseek-v4-flash")


def test_active_role_models_single(monkeypatch):
    monkeypatch.setenv("SINGLE_MODEL", "deepseek/deepseek-v4-flash")
    assert orchestrator.active_role_models("single") == {
        "single": "deepseek/deepseek-v4-flash"
    }


def test_active_role_models_missing_env_raises(monkeypatch):
    monkeypatch.delenv("SINGLE_MODEL", raising=False)
    with pytest.raises(orchestrator.ConfigError, match="SINGLE_MODEL"):
        orchestrator.active_role_models("single")


def test_active_role_models_unknown_strategy():
    with pytest.raises(orchestrator.ConfigError, match="Unknown strategy"):
        orchestrator.active_role_models("nonsense")


def test_validate_config_passes_with_full_debate_setup(monkeypatch):
    monkeypatch.setenv("DEBATER1_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("DEBATER2_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("JUDGE_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("CODER_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test2")
    orchestrator.validate_config("debate")  # should not raise


def test_validate_config_missing_role_env_raises(monkeypatch):
    monkeypatch.delenv("DEBATER1_MODEL", raising=False)
    monkeypatch.setenv("DEBATER2_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("JUDGE_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("CODER_MODEL", "qwen/qwen3.7-plus")
    with pytest.raises(orchestrator.ConfigError, match="DEBATER1_MODEL"):
        orchestrator.validate_config("debate")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pip install -r requirements.txt && pytest tests/test_orchestrator_config.py -v`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'orchestrator'`

- [ ] **Step 4: Create `orchestrator.py` with the provider registry and config validation**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_config.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add orchestrator.py requirements.txt tests/test_orchestrator_config.py
git commit -m "Add orchestrator provider registry and strategy config validation"
```

---

## Task 2: `_chat()` transient-retry + timing wrapper

**Files:**
- Modify: `orchestrator.py`
- Test: `tests/test_orchestrator_chat.py`

**Interfaces:**
- Consumes: `orchestrator._resolve_role_model` (Task 1).
- Produces: `orchestrator._call_model_raw(role_model: str, prompt: str) -> str`, `orchestrator._call_with_retry(role_model: str, prompt: str) -> str`, `orchestrator._chat(role_model: str, prompt: str) -> tuple[str, float]` (reply, duration_s).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_chat.py`:

```python
import orchestrator


def test_chat_returns_reply_and_duration(monkeypatch):
    monkeypatch.setattr(orchestrator, "_call_model_raw", lambda role_model, prompt: "hello")
    reply, duration = orchestrator._chat("deepseek/deepseek-v4-flash", "hi")
    assert reply == "hello"
    assert duration >= 0


def test_call_with_retry_retries_on_transient_error(monkeypatch):
    calls = []

    def flaky(role_model, prompt):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    monkeypatch.setattr(orchestrator, "_call_model_raw", flaky)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: None)
    result = orchestrator._call_with_retry("deepseek/deepseek-v4-flash", "hi")
    assert result == "ok"
    assert len(calls) == 2


def test_call_with_retry_gives_up_after_max_attempts(monkeypatch):
    def always_fails(role_model, prompt):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(orchestrator, "_call_model_raw", always_fails)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: None)
    try:
        orchestrator._call_with_retry("deepseek/deepseek-v4-flash", "hi")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as exc:
        assert "503" in str(exc)


def test_call_with_retry_does_not_retry_non_transient_errors(monkeypatch):
    calls = []

    def bad_request(role_model, prompt):
        calls.append(1)
        raise RuntimeError("400 Bad Request")

    monkeypatch.setattr(orchestrator, "_call_model_raw", bad_request)
    try:
        orchestrator._call_with_retry("deepseek/deepseek-v4-flash", "hi")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
    assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_chat.py -v`
Expected: FAIL with `AttributeError: module 'orchestrator' has no attribute '_call_model_raw'`

- [ ] **Step 3: Add the chat layer to `orchestrator.py`**

Add `import time` to the top imports (now `import os` and `import time`), then append:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_chat.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_chat.py
git commit -m "Add orchestrator chat layer with transient-error retry and timing"
```

---

## Task 3: `extract_code()`, coder prompt builder, and `code()`

**Files:**
- Modify: `orchestrator.py`
- Test: `tests/test_orchestrator_code.py`

**Interfaces:**
- Consumes: `orchestrator._chat` (Task 2).
- Produces: `orchestrator.extract_code(text: str) -> str`, `orchestrator._build_coder_prompt(original_prompt, plan_text, prev_code, failure) -> str`, `orchestrator.code(strategy: str, original_prompt: str, plan_text=None, prev_code=None, failure=None) -> dict` with keys `model`, `prompt`, `reply`, `code`, `duration_s`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_code.py`:

```python
import orchestrator


def test_extract_code_prefers_longest_fenced_block():
    text = (
        "short:\n```python\nx=1\n```\n"
        "long:\n```python\ndef f():\n    return 42\n```\n"
    )
    assert orchestrator.extract_code(text) == "def f():\n    return 42"


def test_extract_code_falls_back_to_raw_text():
    assert orchestrator.extract_code("just code, no fence") == "just code, no fence"


def test_build_coder_prompt_plain():
    prompt = orchestrator._build_coder_prompt("PROBLEM", None, None, None)
    assert prompt == "PROBLEM"


def test_build_coder_prompt_includes_plan():
    prompt = orchestrator._build_coder_prompt("PROBLEM", "use a heap", None, None)
    assert "PROBLEM" in prompt
    assert "use a heap" in prompt


def test_build_coder_prompt_includes_retry_context():
    failure = {"index": 0, "type": "WRONG_ANSWER", "input": "1", "expected": "2", "got": "3"}
    prompt = orchestrator._build_coder_prompt("PROBLEM", None, "print(1)", failure)
    assert "print(1)" in prompt
    assert "Failing test #0" in prompt
    assert "WRONG_ANSWER" in prompt


def test_code_calls_configured_coder_model(monkeypatch):
    monkeypatch.setenv("SINGLE_MODEL", "deepseek/deepseek-v4-flash")
    seen = {}

    def fake_chat(role_model, prompt):
        seen["role_model"] = role_model
        seen["prompt"] = prompt
        return "```python\nprint('hi')\n```", 0.5

    monkeypatch.setattr(orchestrator, "_chat", fake_chat)
    result = orchestrator.code("single", "PROBLEM")
    assert seen["role_model"] == "deepseek/deepseek-v4-flash"
    assert result["code"] == "print('hi')"
    assert result["duration_s"] == 0.5
    assert result["reply"] == "```python\nprint('hi')\n```"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_code.py -v`
Expected: FAIL with `AttributeError: module 'orchestrator' has no attribute 'extract_code'`

- [ ] **Step 3: Add code generation to `orchestrator.py`**

Add `import re` to the top imports (now `import os`, `import re`, `import time`), then append:

```python
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the program out of a model reply. Prefer a ```python fenced block;
    if several, take the longest; fall back to the raw text."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


_CODER_MODEL_ENV = {
    "single": "SINGLE_MODEL",
    "analyze-then-code": "CODER_MODEL",
    "debate": "CODER_MODEL",
}


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
    role_model = os.environ[_CODER_MODEL_ENV[strategy]]
    prompt = _build_coder_prompt(original_prompt, plan_text, prev_code, failure)
    reply, duration = _chat(role_model, prompt)
    return {
        "model": role_model,
        "prompt": prompt,
        "reply": reply,
        "code": extract_code(reply),
        "duration_s": duration,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_code.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_code.py
git commit -m "Add orchestrator code-generation step with plan/retry-aware prompts"
```

---

## Task 4: `plan()` dispatch for `single` and `analyze-then-code`

**Files:**
- Modify: `orchestrator.py`
- Test: `tests/test_orchestrator_plan.py`

**Interfaces:**
- Consumes: `orchestrator._chat` (Task 2).
- Produces: `orchestrator.plan(strategy: str, prompt: str) -> tuple[str | None, list[dict]]` (handles `single` and `analyze-then-code`; `debate` added in Task 5).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_plan.py`:

```python
import orchestrator


def test_plan_single_returns_none_and_no_events():
    plan_text, events = orchestrator.plan("single", "PROBLEM")
    assert plan_text is None
    assert events == []


def test_plan_analyze_then_code_calls_architect(monkeypatch):
    monkeypatch.setenv("ARCHITECT_MODEL", "deepseek/deepseek-v4-flash")
    seen = {}

    def fake_chat(role_model, prompt):
        seen["role_model"] = role_model
        seen["prompt"] = prompt
        return "use a two-pointer approach", 1.1

    monkeypatch.setattr(orchestrator, "_chat", fake_chat)
    plan_text, events = orchestrator.plan("analyze-then-code", "PROBLEM STATEMENT")

    assert plan_text == "use a two-pointer approach"
    assert seen["role_model"] == "deepseek/deepseek-v4-flash"
    assert "PROBLEM STATEMENT" in seen["prompt"]
    assert events == [{
        "role": "architect", "model": "deepseek/deepseek-v4-flash",
        "prompt": seen["prompt"], "reply": "use a two-pointer approach",
        "duration_s": 1.1,
    }]


def test_plan_unknown_strategy_raises():
    try:
        orchestrator.plan("nonsense", "PROBLEM")
        assert False, "expected ConfigError"
    except orchestrator.ConfigError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_plan.py -v`
Expected: FAIL with `AttributeError: module 'orchestrator' has no attribute 'plan'`

- [ ] **Step 3: Add `plan()` (single + analyze-then-code) to `orchestrator.py`**

Append:

```python
_ARCHITECT_PROMPT_TMPL = (
    "{problem}\n\n"
    "Analyze this problem and describe your solution approach in plain text: the "
    "key idea, algorithm, and time/space complexity. Do NOT write code — just the "
    "approach."
)


def _run_analyze_then_code_plan(prompt: str) -> tuple:
    role_model = os.environ["ARCHITECT_MODEL"]
    architect_prompt = _ARCHITECT_PROMPT_TMPL.format(problem=prompt)
    reply, duration = _chat(role_model, architect_prompt)
    event = {
        "role": "architect", "model": role_model,
        "prompt": architect_prompt, "reply": reply, "duration_s": duration,
    }
    return reply, [event]


def plan(strategy: str, prompt: str) -> tuple:
    """Run strategy's planning stage once per problem.
    Returns (plan_text_or_None, events)."""
    if strategy == "single":
        return None, []
    if strategy == "analyze-then-code":
        return _run_analyze_then_code_plan(prompt)
    if strategy == "debate":
        return _run_debate_plan(prompt)
    raise ConfigError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
```

Note: `_run_debate_plan` is referenced but not yet defined — that's fine, Python only
resolves the name when the `debate` branch actually executes, and Task 4's tests never
take that branch. Task 5 defines it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_plan.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_plan.py
git commit -m "Add orchestrator plan() dispatch for single and analyze-then-code strategies"
```

---

## Task 5: `plan()` dispatch for `debate` (2 rounds + judge)

**Files:**
- Modify: `orchestrator.py`
- Modify: `tests/test_orchestrator_plan.py`

**Interfaces:**
- Consumes: `orchestrator._chat` (Task 2), `orchestrator.plan` (Task 4, already dispatches to `debate`).
- Produces: `orchestrator._run_debate_plan(prompt: str) -> tuple[str, list[dict]]` — 5 events: `debater1`/`debater2` at `round: 1`, `debater1`/`debater2` at `round: 2`, then `judge` (no round key).

- [ ] **Step 1: Add the failing test**

Append to `tests/test_orchestrator_plan.py`:

```python
def test_plan_debate_runs_two_rounds_plus_judge(monkeypatch):
    monkeypatch.setenv("DEBATER1_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("DEBATER2_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("JUDGE_MODEL", "mimo/mimo-v2.5")

    calls = []

    def fake_chat(role_model, prompt):
        calls.append((role_model, prompt))
        n = len(calls)
        if n == 1:
            return "debater1 round1 approach", 0.1
        if n == 2:
            return "debater2 round1 approach", 0.1
        if n == 3:
            return "debater1 round2 approach", 0.1
        if n == 4:
            return "debater2 round2 approach", 0.1
        if n == 5:
            return "final synthesized approach", 0.1
        raise AssertionError(f"unexpected call #{n}: {role_model}")

    monkeypatch.setattr(orchestrator, "_chat", fake_chat)
    plan_text, events = orchestrator.plan("debate", "PROBLEM")

    assert plan_text == "final synthesized approach"
    assert len(calls) == 5
    assert calls[0][0] == "deepseek/deepseek-v4-flash"
    assert calls[1][0] == "qwen/qwen3.7-plus"
    assert calls[4][0] == "mimo/mimo-v2.5"
    assert [e["role"] for e in events] == [
        "debater1", "debater2", "debater1", "debater2", "judge"
    ]
    assert events[0]["round"] == 1
    assert events[2]["round"] == 2
    # round-2 prompts must reference both debaters' round-1 replies
    assert "debater1 round1 approach" in events[2]["prompt"]
    assert "debater2 round1 approach" in events[2]["prompt"]
    assert "debater1 round1 approach" in events[3]["prompt"]
    assert "debater2 round1 approach" in events[3]["prompt"]
    # judge prompt must reference both debaters' round-2 replies
    assert "debater1 round2 approach" in events[4]["prompt"]
    assert "debater2 round2 approach" in events[4]["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_plan.py::test_plan_debate_runs_two_rounds_plus_judge -v`
Expected: FAIL with `NameError: name '_run_debate_plan' is not defined`

- [ ] **Step 3: Add `_run_debate_plan` to `orchestrator.py`**

Append (before the `plan()` function, anywhere in the module works since Python
resolves names at call time — placing it just above `plan()` keeps the file readable):

```python
_DEBATE_ROUND1_TMPL = (
    "{problem}\n\n"
    "Propose a solution approach for this problem in plain text: the key idea, "
    "algorithm, and complexity. Do NOT write code yet."
)

_DEBATE_ROUND2_TMPL = (
    "{problem}\n\n"
    "You proposed this approach:\n{own}\n\n"
    "A different model proposed this approach instead:\n{other}\n\n"
    "Critique the other approach, defend or revise your own, and state your final "
    "recommended approach in plain text. Do NOT write code yet."
)

_JUDGE_TMPL = (
    "{problem}\n\n"
    "Two models debated solution approaches for this problem.\n\n"
    "Debater A's final position:\n{d1}\n\n"
    "Debater B's final position:\n{d2}\n\n"
    "Synthesize a single final solution approach to implement, in plain text. "
    "Do NOT write code."
)


def _run_debate_plan(prompt: str) -> tuple:
    events = []
    d1_model = os.environ["DEBATER1_MODEL"]
    d2_model = os.environ["DEBATER2_MODEL"]
    judge_model = os.environ["JUDGE_MODEL"]

    r1_prompt = _DEBATE_ROUND1_TMPL.format(problem=prompt)
    d1_r1, dur = _chat(d1_model, r1_prompt)
    events.append({"round": 1, "role": "debater1", "model": d1_model,
                    "prompt": r1_prompt, "reply": d1_r1, "duration_s": dur})
    d2_r1, dur = _chat(d2_model, r1_prompt)
    events.append({"round": 1, "role": "debater2", "model": d2_model,
                    "prompt": r1_prompt, "reply": d2_r1, "duration_s": dur})

    d1_r2_prompt = _DEBATE_ROUND2_TMPL.format(problem=prompt, own=d1_r1, other=d2_r1)
    d1_r2, dur = _chat(d1_model, d1_r2_prompt)
    events.append({"round": 2, "role": "debater1", "model": d1_model,
                    "prompt": d1_r2_prompt, "reply": d1_r2, "duration_s": dur})

    d2_r2_prompt = _DEBATE_ROUND2_TMPL.format(problem=prompt, own=d2_r1, other=d1_r1)
    d2_r2, dur = _chat(d2_model, d2_r2_prompt)
    events.append({"round": 2, "role": "debater2", "model": d2_model,
                    "prompt": d2_r2_prompt, "reply": d2_r2, "duration_s": dur})

    judge_prompt = _JUDGE_TMPL.format(problem=prompt, d1=d1_r2, d2=d2_r2)
    judge_reply, dur = _chat(judge_model, judge_prompt)
    events.append({"role": "judge", "model": judge_model,
                    "prompt": judge_prompt, "reply": judge_reply, "duration_s": dur})

    return judge_reply, events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_plan.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_plan.py
git commit -m "Add orchestrator debate strategy (2 rounds + judge synthesis)"
```

---

## Task 6: Detailed run-log writer

**Files:**
- Modify: `orchestrator.py`
- Test: `tests/test_orchestrator_logging.py`

**Interfaces:**
- Produces: `orchestrator.write_run_log(path: str, run_log: dict) -> None` (creates parent dir, writes pretty JSON).

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestrator_logging.py`:

```python
import json
import os

import orchestrator


def test_write_run_log_creates_parent_dir_and_valid_json(tmp_path):
    path = os.path.join(str(tmp_path), "logs", "run_20260701_000000.json")
    run_log = {"run_id": "20260701_000000", "problems": [{"question_id": "1", "solved": True}]}

    orchestrator.write_run_log(path, run_log)

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == run_log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_logging.py -v`
Expected: FAIL with `AttributeError: module 'orchestrator' has no attribute 'write_run_log'`

- [ ] **Step 3: Add `write_run_log` to `orchestrator.py`**

Add `import json` to the top imports (now `import json`, `import os`, `import re`, `import time`), then append:

```python
def write_run_log(path: str, run_log: dict) -> None:
    """Write the full per-run debug log as pretty JSON, creating logs/ if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(run_log, fh, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_logging.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_logging.py
git commit -m "Add orchestrator detailed run-log writer"
```

---

## Task 7: `solve_problem()` in `run_eval.py`

**Files:**
- Modify: `run_eval.py` (add `import orchestrator`; add new `solve_problem` function after `evaluate_problem`, before `load_subset` — existing `generate()`/`build_retry_prompt`/`extract_code`/`_generate_with_api_retry` are left in place for now and removed in Task 8)
- Test: `tests/test_run_eval_solve_problem.py`

**Interfaces:**
- Consumes: `orchestrator.plan`, `orchestrator.code` (Tasks 4/5/3), `run_eval.evaluate_problem` (existing), `run_eval._meta` (existing).
- Produces: `run_eval.solve_problem(p: dict, strategy: str, timeout: float, max_tests: int, max_retries: int) -> tuple[dict, dict]` — `(result, log_entry)`. `result` matches the existing per-problem `results.json` shape (`_meta(p)` fields + `solved`/`num_tests`/`tests_passed`/`first_failure` + `code`/`reply_len`/`attempts`, or `_meta(p)` + `error`/`attempts` on failure). `log_entry` has `_meta(p)` fields + `plan_events`/`debate_events`/`attempts` (list of `{attempt, model, prompt, reply, code, duration_s, test_result}`) + `solved` (+ `error` on failure).

- [ ] **Step 1: Add the `import orchestrator` line**

In `run_eval.py`, change the top imports block from:

```python
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
```

to:

```python
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import orchestrator
```

(`re` is still used by the existing `_FENCE`/`extract_code` in this file until Task 8 removes them.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_run_eval_solve_problem.py`:

```python
import run_eval


def _problem(**overrides):
    p = {
        "question_id": "q1", "title": "Two Sum", "platform": "leetcode",
        "difficulty": "hard", "fn_name": "twoSum", "prompt": "PROBLEM TEXT",
        "tests": [{"input": "1", "output": "2", "testtype": "stdin"}],
    }
    p.update(overrides)
    return p


def test_solve_problem_calls_plan_once_regardless_of_retries(monkeypatch):
    plan_calls = []
    code_calls = []

    def fake_plan(strategy, prompt):
        plan_calls.append((strategy, prompt))
        return "THE PLAN", [
            {"role": "architect", "model": "m", "prompt": "p", "reply": "r", "duration_s": 0.1}
        ]

    def fake_code(strategy, original_prompt, plan_text, prev_code, failure):
        code_calls.append({"plan_text": plan_text, "prev_code": prev_code, "failure": failure})
        return {"model": "m", "prompt": "cp", "reply": "```python\nprint(1)\n```",
                "code": "print(1)", "duration_s": 0.2}

    def fake_evaluate(p, code, timeout, max_tests):
        solved = len(code_calls) >= 3  # fail attempts 1-2, pass attempt 3
        return {
            "solved": solved, "num_tests": 1, "tests_passed": 1 if solved else 0,
            "first_failure": None if solved else {
                "index": 0, "type": "WRONG_ANSWER", "input": "1", "expected": "2", "got": "3",
            },
        }

    monkeypatch.setattr(run_eval.orchestrator, "plan", fake_plan)
    monkeypatch.setattr(run_eval.orchestrator, "code", fake_code)
    monkeypatch.setattr(run_eval, "evaluate_problem", fake_evaluate)

    result, log_entry = run_eval.solve_problem(
        _problem(), "analyze-then-code", timeout=5.0, max_tests=0, max_retries=2
    )

    assert len(plan_calls) == 1
    assert len(code_calls) == 3
    assert code_calls[0]["prev_code"] is None
    assert code_calls[0]["failure"] is None
    assert code_calls[1]["prev_code"] == "print(1)"
    assert code_calls[1]["failure"]["type"] == "WRONG_ANSWER"
    assert result["solved"] is True
    assert result["attempts"] == 3
    assert log_entry["plan_events"][0]["role"] == "architect"
    assert log_entry["debate_events"] == []
    assert len(log_entry["attempts"]) == 3
    assert log_entry["solved"] is True


def test_solve_problem_stops_after_max_retries_when_never_solved(monkeypatch):
    def fake_plan(strategy, prompt):
        return None, []

    def fake_code(strategy, original_prompt, plan_text, prev_code, failure):
        return {"model": "m", "prompt": "cp", "reply": "bad", "code": "bad", "duration_s": 0.1}

    def fake_evaluate(p, code, timeout, max_tests):
        return {
            "solved": False, "num_tests": 1, "tests_passed": 0,
            "first_failure": {"index": 0, "type": "WRONG_ANSWER", "input": "1",
                               "expected": "2", "got": "3"},
        }

    monkeypatch.setattr(run_eval.orchestrator, "plan", fake_plan)
    monkeypatch.setattr(run_eval.orchestrator, "code", fake_code)
    monkeypatch.setattr(run_eval, "evaluate_problem", fake_evaluate)

    result, log_entry = run_eval.solve_problem(
        _problem(), "single", timeout=5.0, max_tests=0, max_retries=1
    )

    assert result["solved"] is False
    assert result["attempts"] == 2  # 1 initial + 1 retry
    assert len(log_entry["attempts"]) == 2


def test_solve_problem_records_error_when_plan_raises(monkeypatch):
    def fake_plan(strategy, prompt):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(run_eval.orchestrator, "plan", fake_plan)

    result, log_entry = run_eval.solve_problem(
        _problem(), "debate", timeout=5.0, max_tests=0, max_retries=2
    )

    assert result["solved"] is False
    assert "401 Unauthorized" in result["error"]
    assert result["attempts"] == 0
    assert log_entry["error"] == result["error"]


def test_solve_problem_records_error_when_code_raises(monkeypatch):
    def fake_plan(strategy, prompt):
        return None, []

    def fake_code(strategy, original_prompt, plan_text, prev_code, failure):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(run_eval.orchestrator, "plan", fake_plan)
    monkeypatch.setattr(run_eval.orchestrator, "code", fake_code)

    result, log_entry = run_eval.solve_problem(
        _problem(), "single", timeout=5.0, max_tests=0, max_retries=2
    )

    assert result["solved"] is False
    assert "401 Unauthorized" in result["error"]
    assert result["attempts"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_run_eval_solve_problem.py -v`
Expected: FAIL with `AttributeError: module 'run_eval' has no attribute 'solve_problem'`

- [ ] **Step 4: Add `solve_problem()` to `run_eval.py`**

Insert this new function immediately after `evaluate_problem` (which ends at the
`return {...}` block right before the `# --- Main ---` section) and before
`load_subset`:

```python
def solve_problem(p: dict, strategy: str, timeout: float, max_tests: int, max_retries: int) -> tuple:
    """Run one problem end-to-end: plan() once, then code()+evaluate up to
    max_retries+1 attempts. Returns (result, log_entry).
    result matches the existing results.json per-problem shape.
    log_entry is the full JSON-loggable trace (plan/debate events + every attempt)."""
    original_prompt = p["prompt"]
    log_entry = {**_meta(p), "plan_events": [], "debate_events": [], "attempts": [], "solved": False}

    try:
        plan_text, events = orchestrator.plan(strategy, original_prompt)
    except Exception as exc:
        log_entry["error"] = str(exc)
        return {**_meta(p), "solved": False, "error": str(exc), "attempts": 0}, log_entry

    if strategy == "debate":
        log_entry["debate_events"] = events
    else:
        log_entry["plan_events"] = events

    prev_code = None
    failure = None
    ev = None
    attempt_result = None
    attempt = 0
    for attempt in range(1, max_retries + 2):  # 1 initial + N retries
        try:
            attempt_result = orchestrator.code(strategy, original_prompt, plan_text, prev_code, failure)
        except Exception as exc:
            log_entry["error"] = str(exc)
            return {**_meta(p), "solved": False, "error": str(exc), "attempts": attempt}, log_entry

        code_text = attempt_result["code"]
        ev = evaluate_problem(p, code_text, timeout, max_tests)
        log_entry["attempts"].append({
            "attempt": attempt,
            "model": attempt_result["model"],
            "prompt": attempt_result["prompt"],
            "reply": attempt_result["reply"],
            "code": code_text,
            "duration_s": attempt_result["duration_s"],
            "test_result": ev,
        })
        prev_code = code_text
        if ev["solved"] or attempt == max_retries + 1:
            break
        failure = ev["first_failure"]

    log_entry["solved"] = bool(ev and ev["solved"])
    result = {
        **_meta(p), **ev, "code": prev_code,
        "reply_len": len(attempt_result["reply"]), "attempts": attempt,
    }
    return result, log_entry
```

`_meta` is defined later in the file (near `main()`); that's fine since Python only
looks up `_meta` when `solve_problem` is actually called, not when it's defined.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_run_eval_solve_problem.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add run_eval.py tests/test_run_eval_solve_problem.py
git commit -m "Add run_eval.solve_problem() using the orchestrator plan/code interface"
```

---

## Task 8: Wire `main()`, remove dead code, update docs

**Files:**
- Modify: `run_eval.py` (rewrite `main()`; delete `generate()`, `_FENCE`, `extract_code()`, `_API_RETRY_ATTEMPTS`, `_API_RETRY_BACKOFF`, `_generate_with_api_retry()`, `build_retry_prompt()`; drop the now-unused `re` import)
- Modify: `.gitignore` (add `logs/`)
- Modify: `README.md` (usage, pipeline flow, function reference)
- Test: `tests/test_run_eval_main_integration.py`

**Interfaces:**
- Consumes: `orchestrator.validate_config`, `orchestrator.active_role_models`, `orchestrator.write_run_log` (Tasks 1/6), `run_eval.solve_problem` (Task 7).
- Produces: `run_eval.main()` accepts `--strategy`; writes `results.json` (existing shape, `summary` gains `strategy`/`models`, loses `model`) and `logs/run_<timestamp>.json`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_run_eval_main_integration.py`:

```python
import json
import os
import sys

import orchestrator
import run_eval


def test_main_end_to_end_with_fake_orchestrator(tmp_path, monkeypatch):
    subset_path = os.path.join(str(tmp_path), "subset.jsonl")
    problem = {
        "question_id": "q1", "title": "Two Sum", "platform": "leetcode",
        "difficulty": "hard", "fn_name": "twoSum", "prompt": "PROBLEM TEXT",
        "tests": [{"input": "1\n2\n", "output": "3", "testtype": "stdin"}],
    }
    with open(subset_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(problem) + "\n")

    out_path = os.path.join(str(tmp_path), "results.json")

    monkeypatch.setenv("STRATEGY", "single")
    monkeypatch.setenv("SINGLE_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def fake_plan(strategy, prompt):
        return None, []

    def fake_code(strategy, original_prompt, plan_text, prev_code, failure):
        return {"model": "deepseek/deepseek-v4-flash", "prompt": original_prompt,
                "reply": "```python\nprint(3)\n```", "code": "print(3)", "duration_s": 0.05}

    monkeypatch.setattr(orchestrator, "plan", fake_plan)
    monkeypatch.setattr(orchestrator, "code", fake_code)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_eval.py", "--subset", subset_path, "--out", out_path,
        "--max-tests", "1", "--max-retries", "0",
    ])

    run_eval.main()

    with open(out_path, encoding="utf-8") as fh:
        results_doc = json.load(fh)
    assert results_doc["summary"]["solved"] == 1
    assert results_doc["summary"]["strategy"] == "single"

    log_files = list((tmp_path / "logs").glob("run_*.json"))
    assert len(log_files) == 1
    with open(log_files[0], encoding="utf-8") as fh:
        log_doc = json.load(fh)
    assert log_doc["problems"][0]["solved"] is True
    assert log_doc["config"]["strategy"] == "single"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_eval_main_integration.py -v`
Expected: FAIL — `results.json`'s `summary` has no `strategy` key yet (still today's `main()`)

- [ ] **Step 3: Delete the now-dead code from `run_eval.py`**

Delete these blocks entirely:
- The whole `generate()` function (the "THE ONE PLUGGABLE HOOK" section, from the `# ---` banner above it down to its closing `return resp.text or ""`).
- The `# Code extraction` section: `_FENCE` and `extract_code()`.
- The `# API-level retry` section: `_API_RETRY_ATTEMPTS`, `_API_RETRY_BACKOFF`, `_generate_with_api_retry()`.
- The `# Retry prompt` section: `build_retry_prompt()`.

Change the imports from:

```python
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import orchestrator
```

to (drop `re`, no longer used in this file):

```python
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import orchestrator
```

- [ ] **Step 4: Rewrite `main()`**

Replace the entire `main()` function with:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description="LiveBench-style code eval harness")
    ap.add_argument("--subset", default="hard_subset.jsonl",
                    help="JSONL of normalised problems (from select_hard.py)")
    ap.add_argument("--out", default="results.json", help="Detailed results file")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Per-test timeout in seconds")
    ap.add_argument("--max-tests", type=int, default=0,
                    help="Cap tests per problem (0 = all). Use for quick smoke runs.")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Retries per problem after a failed attempt (0 = no retries).")
    ap.add_argument("--strategy", default=None,
                    help="single | analyze-then-code | debate "
                         "(default: STRATEGY env var, else 'single')")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    strategy = args.strategy or os.environ.get("STRATEGY", "single")
    try:
        orchestrator.validate_config(strategy)
    except orchestrator.ConfigError as exc:
        sys.exit(str(exc))

    if not os.path.exists(args.subset):
        sys.exit(f"Subset file not found: {args.subset}\n"
                 f"Run:  python select_hard.py --n 5")

    problems = load_subset(args.subset)
    print(f"Loaded {len(problems)} problems from {args.subset}  [strategy={strategy}]\n")

    results = []
    log_problems = []
    solved = 0
    for n, p in enumerate(problems, 1):
        title = p.get("title", "(untitled)")
        diff = p.get("difficulty", "?")

        result, log_entry = solve_problem(p, strategy, args.timeout, args.max_tests, args.max_retries)
        log_problems.append(log_entry)
        results.append(result)

        if "error" in result:
            print(f"[{n}/{len(problems)}] ERROR  {title}: {result['error']}")
            continue

        if result["solved"]:
            solved += 1

        status = "PASS" if result["solved"] else "FAIL"
        line = (f"[{n}/{len(problems)}] {status}  {diff:6s}  {title:<48} "
                f"({result['tests_passed']}/{result['num_tests']})  attempts={result['attempts']}")
        if not result["solved"] and result["first_failure"]:
            ff = result["first_failure"]
            line += f"  test#{ff['index']} {ff['type']}"
        print(line)
        if not result["solved"] and result["first_failure"]:
            ff = result["first_failure"]
            print(f"        input:    {ff['input']!r}")
            print(f"        expected: {ff['expected']!r}")
            print(f"        got:      {ff['got']!r}")

    total = len(problems)
    score = solved / total if total else 0.0
    print(f"\n==== SCORE: {solved}/{total} solved  ({score:.0%}) ====")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "summary": {
                "total": total, "solved": solved, "score": score,
                "strategy": strategy, "models": orchestrator.active_role_models(strategy),
                "timeout": args.timeout, "max_tests": args.max_tests,
                "max_retries": args.max_retries,
            },
            "results": results,
        }, fh, indent=2, ensure_ascii=False)
    print(f"Wrote detail -> {args.out}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"run_{run_id}.json")
    run_log = {
        "run_id": run_id,
        "config": {
            "strategy": strategy, "models": orchestrator.active_role_models(strategy),
            "subset": args.subset, "max_retries": args.max_retries,
        },
        "problems": log_problems,
    }
    orchestrator.write_run_log(log_path, run_log)
    print(f"Wrote detailed log -> {log_path}")
```

- [ ] **Step 5: Run the full test suite to verify everything passes**

Run: `pytest tests/ -v`
Expected: PASS (all tests across every task, including the new integration test)

- [ ] **Step 6: Add `logs/` to `.gitignore`**

Change `.gitignore` from:

```
# Generated artifacts
hard_subset.jsonl
results.json
```

to:

```
# Generated artifacts
hard_subset.jsonl
results.json
logs/
```

- [ ] **Step 7: Update `README.md`**

Replace the "What is this?" paragraph's model description — change:

```
problems from LiveBench's `LCB_generation` dataset
(LeetCode / AtCoder). It loads and normalizes the raw dataset, selects the N hardest
problems, sends each problem's prompt to **Gemma 4** via the Gemini API, extracts the
returned Python code, and runs it against every public + private test case in an
isolated subprocess.
```

to:

```
problems from LiveBench's `LCB_generation` dataset
(LeetCode / AtCoder). It loads and normalizes the raw dataset, selects the N hardest
problems, solves each one using a configurable model/strategy (see `orchestrator.py` —
single model, analyze-then-code, or a multi-model debate), extracts the returned Python
code, and runs it against every public + private test case in an isolated subprocess.
```

Add a row to the "Files" table (after the `run_eval.py` row):

```
| `orchestrator.py` | Provider registry + `plan()`/`code()` — switches between models and solving strategies |
```

Replace the "Usage" section's step 3 (`3. Run the eval`) to document `--strategy`:

```
# 3. Run the eval
python run_eval.py                          # defaults: hard_subset.jsonl, 2 retries, STRATEGY env var (else "single")
python run_eval.py --strategy single             # one model solves directly
python run_eval.py --strategy analyze-then-code  # architect model plans, coder model implements
python run_eval.py --strategy debate             # 2 debaters + judge plan, coder model implements
python run_eval.py --timeout 10             # per-test timeout in seconds
python run_eval.py --max-tests 3            # cap tests/problem (quick smoke run)
python run_eval.py --max-retries 3          # retry failing problems up to 3 times
```

Add a new section right after "## Retry Mechanism":

```markdown
## Model & Strategy Configuration

Every "role" model is a `provider/model` string (e.g. `deepseek/deepseek-v4-flash`,
`qwen/qwen3.7-plus`) resolved against a small provider registry in `orchestrator.py`
(`PROVIDERS`) — each provider maps to a `base_url`/`api_key` pair configured in `.env`.

| Strategy | Roles used | What happens |
|---|---|---|
| `single` | `SINGLE_MODEL` | One model solves the problem directly (today's default behavior). |
| `analyze-then-code` | `ARCHITECT_MODEL`, `CODER_MODEL` | Architect proposes an approach; coder implements it. |
| `debate` | `DEBATER1_MODEL`, `DEBATER2_MODEL`, `JUDGE_MODEL`, `CODER_MODEL` | 2 debaters each propose then revise (2 rounds); judge synthesizes a final approach; coder implements it. |

`STRATEGY` in `.env` sets the default; `--strategy` on the command line overrides it.
On a retry, only the coder step re-runs — the architect/debate stage is not repeated.
```

Add a new section right after "## Function Reference"'s `run_eval.py` table (before
"## Plugging in a Custom Orchestrator", which should be removed since `orchestrator.py`
now exists for real):

```markdown
### `orchestrator.py`
| Function | Purpose |
|---|---|
| `plan(strategy, prompt)` | Runs the strategy's planning stage once per problem. Returns `(plan_text_or_None, events)`. |
| `code(strategy, original_prompt, plan_text, prev_code, failure)` | Runs the coder step once (initial attempt or retry). Returns `{model, prompt, reply, code, duration_s}`. |
| `validate_config(strategy)` | Fails fast if the strategy's required role models/providers aren't configured. |
| `active_role_models(strategy)` | Returns the role → `provider/model` mapping in effect for a strategy. |
| `write_run_log(path, run_log)` | Writes the full per-run debug log as JSON, creating `logs/` if needed. |

### Detailed logging

Every `run_eval.py` run writes `logs/run_<timestamp>.json` with the full prompt/reply/
test-result trace for every problem (architect analysis, debate rounds, every coder
attempt) — `results.json` stays a concise summary; this is the file to open when
debugging why a specific attempt failed.
```

Remove the entire "## Plugging in a Custom Orchestrator" section and its code block —
`orchestrator.py` is the real integration point now, documented above.

Update "## What's Still Missing" — remove the `orchestrator.py` bullet (it now exists):

```markdown
## What's Still Missing

- **No CI / lint config** (no `.github/workflows`, no `pyproject.toml`).
```

- [ ] **Step 8: Commit the code + docs changes**

```bash
git add run_eval.py .gitignore README.md tests/test_run_eval_main_integration.py
git commit -m "Wire run_eval.main() to the orchestrator, add --strategy, write detailed run logs"
```

- [ ] **Step 9: Manual, token-cheap smoke test against real APIs**

These hit real DeepSeek/Qwen/MiMo endpoints and cost a small number of tokens — keep
the subset and retry budget minimal, per the project's token-saving guidance.

```bash
python select_hard.py --n 1 --out smoke_subset.jsonl
python run_eval.py --subset smoke_subset.jsonl --strategy single --max-tests 1 --max-retries 0
python run_eval.py --subset smoke_subset.jsonl --strategy analyze-then-code --max-tests 1 --max-retries 0
python run_eval.py --subset smoke_subset.jsonl --strategy debate --max-tests 1 --max-retries 0
```

For each run, confirm:
- It exits with a `PASS`/`FAIL` line and a final `SCORE` line (no crash/traceback).
- `logs/run_<timestamp>.json` exists and is valid JSON (`python -c "import json,glob; json.load(open(sorted(glob.glob('logs/run_*.json'))[-1], encoding='utf-8'))"`).
- For `analyze-then-code`: that log's `problems[0]["plan_events"]` has exactly 1 entry with `"role": "architect"`.
- For `debate`: that log's `problems[0]["debate_events"]` has exactly 5 entries in the order `debater1, debater2, debater1, debater2, judge`.

Delete the scratch subset when done: `rm smoke_subset.jsonl` (or `Remove-Item smoke_subset.jsonl` on PowerShell) — it's a throwaway file, not the real `hard_subset.jsonl`.

---

## Self-Review Notes

- **Spec coverage:** provider registry (Task 1), transient retry (Task 2), code extraction/prompt building (Task 3), all 3 strategies (Tasks 4–5), coder-only retry semantics (Task 7), detailed JSON log (Task 6, wired in Task 8), `--strategy` CLI override (Task 8), README/`.gitignore` updates (Task 8), token-cheap manual verification (Task 8) — every spec section maps to a task.
- **Type/signature consistency checked:** `plan()` always returns `(str|None, list[dict])`; `code()` always returns a dict with `model`/`prompt`/`reply`/`code`/`duration_s`; `solve_problem()`'s `result`/`log_entry` shapes are used identically in Task 7's tests and Task 8's `main()`/integration test.
- **No placeholders:** every step above contains complete, runnable code — nothing marked TBD.
