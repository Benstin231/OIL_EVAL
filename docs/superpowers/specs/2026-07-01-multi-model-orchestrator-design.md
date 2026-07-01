# Multi-Model Orchestrator Design

**Date:** 2026-07-01
**Status:** Approved for planning

## Background

The eval harness (`run_eval.py`) currently has a single pluggable `generate(prompt)` hook that
calls one hardcoded model (Gemma 4 via the Gemini API). Two of the five original requirements
turned out to already be implemented:

- **Retry on failure** — already exists as a two-layer mechanism (`_generate_with_api_retry` for
  transient API errors, and a prompt-layer retry in `main()` that feeds the failing test back to
  the model).
- **Random hard-subset selection** — `select_hard.py --random [--seed N]` already exists.

This spec covers the three remaining requirements:

1. Switch between different AI models to solve problems.
2. Add a detailed log for debugging.
3. Add different solving strategies (analyze-then-code, debate between models).

Requirements 1 and 3 turned out to be the same underlying mechanism — a **single-model mode** and
a **hybrid mode** where multiple models collaborate — so they're designed together here.

## Goals

- Support calling any model reachable via an OpenAI-compatible API (OpenRouter first, but not
  locked to it) instead of the hardcoded Gemini/Gemma call.
- Support two additional solving strategies beyond direct single-model generation:
  - `analyze-then-code`: one model (architect) analyzes the problem and proposes an approach; a
    second model (coder) writes the code from that analysis.
  - `debate`: two models each propose a solution, each critiques/revises after seeing the other's
    proposal (2 rounds total), then a third model (judge) synthesizes a final approach for the
    coder to implement.
- Keep the existing prompt-level retry loop, but scope it to only re-run the coding step (not
  re-run the architect/debate stage) on a failed test.
- Produce a full, structured per-run log (every prompt/reply/test result at every stage) for
  debugging, separate from the existing concise `results.json` summary.

## Non-goals

- Not building a general multi-provider abstraction beyond the OpenAI-compatible API shape (no
  native Gemini/Anthropic/DeepSeek SDKs — everything goes through one OpenAI-compatible client).
- Not making debate round count or debater count configurable — fixed at 2 debaters + 1 judge,
  2 rounds, per YAGNI. Can be revisited if a real need shows up.
- Not re-running the architect/debate stage on retry.
- Not changing `select_hard.py` or the existing two-layer retry semantics.

## Architecture

### Approach

`orchestrator.py` (new file) replaces the single `generate(prompt)` hook with a two-phase
interface:

```python
plan(prompt: str) -> str | None
code(prompt: str, plan: str | None, prev_code: str | None, failure: dict | None) -> str
```

`run_eval.py`'s main loop calls `plan()` once per problem, then calls `code()` once per attempt
(initial + each retry), passing the previous code and failing test on retries. `plan` stays fixed
across retries — only the coding step re-runs.

Strategies are dispatched by a module-level `if/elif` on `STRATEGY`, not a class hierarchy — with
only 3 strategies and one shared shape ("produce a plan, hand it to a coder"), a class-per-strategy
abstraction would add indirection without payoff. Revisit if a 4th/5th strategy needs meaningfully
different control flow.

### LLM call layer

No single aggregator (e.g. OpenRouter) is available yet — the harness has direct API keys for
two independent OpenAI-compatible endpoints (DeepSeek's official API, Alibaba DashScope's
compatible-mode endpoint). So instead of one shared `base_url`/`api_key`, `orchestrator.py` keeps
a small **provider registry** mapping a provider name to its base_url + api_key env var names:

```python
PROVIDERS = {
    "deepseek": {"base_url_env": "DEEPSEEK_BASE_URL", "api_key_env": "DEEPSEEK_API_KEY"},
    "qwen":     {"base_url_env": "QWEN_BASE_URL",     "api_key_env": "QWEN_API_KEY"},
    "mimo":     {"base_url_env": "MIMO_BASE_URL",     "api_key_env": "MIMO_API_KEY"},
    "gemma":    {"base_url_env": "GEMMA_BASE_URL",    "api_key_env": "GEMINI_API_KEY"},
}
```

`gemma` reuses the existing `GEMINI_API_KEY` and points at Gemini's OpenAI-compatible endpoint
(`GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`), so the model already
configured for today's `generate()` hook is available as a `gemma/<model-name>` role model too —
rounding out the original four candidates (gemma, deepseek, qwen, mimo) for single-model mode.

Every role's model is configured as `provider/model` (e.g. `deepseek/deepseek-v4-flash`,
`qwen/qwen3.7-plus`). `_chat(role_model: str, prompt: str) -> str` splits on the first `/`, looks
up the provider's base_url/key, builds (or reuses a cached) `openai.OpenAI(base_url=..., api_key=...)`
client, and calls `chat.completions.create(model=model_name, ...)`. This reuses the existing
transient-error retry logic (currently `_generate_with_api_retry`, generalized to accept a
role-model string instead of assuming a single fixed model).

Adding a real aggregator (OpenRouter) or another direct provider later is just one more entry in
`PROVIDERS` plus its `.env` vars — the rest of the orchestrator is unaffected.

```
# .env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=...
QWEN_BASE_URL=https://ws-ewvaxo6vfthtzkjd.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=...
MIMO_BASE_URL=https://api.xiaomimimo.com/v1
MIMO_API_KEY=...
```

### Strategy & role configuration

Configured via `.env`, with `--strategy` as a CLI override (takes precedence over `.env`'s
`STRATEGY` when passed):

```
STRATEGY=single                        # single | analyze-then-code | debate
SINGLE_MODEL=deepseek/deepseek-v4-flash
ARCHITECT_MODEL=deepseek/deepseek-v4-flash
CODER_MODEL=qwen/qwen3.7-plus
DEBATER1_MODEL=deepseek/deepseek-v4-flash
DEBATER2_MODEL=qwen/qwen3.7-plus
JUDGE_MODEL=deepseek/deepseek-v4-flash
```

### Strategy behaviors

- **`single`**: `plan()` returns `None` (no API call). `code()` sends the original problem prompt
  (plus retry context, if any) straight to `SINGLE_MODEL`. This preserves today's behavior
  exactly.
- **`analyze-then-code`**: `plan()` sends the problem to `ARCHITECT_MODEL`, asking for a solution
  approach (no code). `code()` sends the problem + that analysis to `CODER_MODEL`.
- **`debate`**: `plan()` runs a fixed 2-round debate:
  - Round 1: `DEBATER1_MODEL` and `DEBATER2_MODEL` each independently propose a solution approach
    for the problem.
  - Round 2: each debater sees the other's round-1 proposal and revises/defends their approach.
  - `JUDGE_MODEL` reads both rounds and produces a single synthesized approach.
  - `code()` sends the problem + judge's synthesis to `CODER_MODEL`.

### Retry integration

On a failed test with retries remaining, `run_eval.py` calls `code(original_prompt, plan,
prev_code=last_code, failure=first_failure)` — same `plan` value as the initial call, only the
coder prompt changes (adds previous code + failing test details, same as today's
`build_retry_prompt`). This keeps retries cheap and keeps the architect/debate stage from being
re-litigated on every retry.

### Detailed logging

Each `run_eval.py` invocation writes `logs/run_<timestamp>.json` capturing the full event
sequence per problem:

```json
{
  "run_id": "...",
  "started_at": "...",
  "config": {"strategy": "...", "models": {...}, "subset": "...", "max_retries": 2},
  "problems": [
    {
      "question_id": "...",
      "title": "...",
      "plan_events": [
        {"role": "architect", "model": "...", "prompt": "...", "reply": "...", "duration_s": 1.2}
      ],
      "debate_events": [
        {"round": 1, "role": "debater1", "model": "...", "prompt": "...", "reply": "...", "duration_s": 0.9},
        {"round": 1, "role": "debater2", "...": "..."},
        {"round": 2, "role": "debater1", "...": "..."},
        {"round": 2, "role": "debater2", "...": "..."},
        {"role": "judge", "...": "..."}
      ],
      "attempts": [
        {
          "attempt": 1, "model": "...", "prompt": "...", "reply": "...", "code": "...",
          "test_result": {"solved": false, "first_failure": {...}}, "duration_s": 2.1
        }
      ],
      "solved": true
    }
  ]
}
```

`results.json` keeps its current concise summary format unchanged (score, PASS/FAIL, first
failure per problem) — it's the quick-glance output. The new per-run log under `logs/` is the
full trace for debugging, generated every run.

`plan_events` and `debate_events` are populated based on which strategy ran (single strategy: both
empty; analyze-then-code: `plan_events` only; debate: `debate_events` only).

## Error handling

- Missing role model env vars for the active strategy → fail fast at startup with a clear message
  (e.g., `STRATEGY=debate` requires `DEBATER1_MODEL`, `DEBATER2_MODEL`, `JUDGE_MODEL`, `CODER_MODEL`
  to all be set).
- Unknown `--strategy` / `STRATEGY` value → fail fast with the list of valid values.
- A role model string with an unknown provider prefix (not in `PROVIDERS`), or a known provider
  missing its `base_url`/`api_key` env vars → fail fast at startup with a clear message naming the
  offending role and provider.
- Transient API errors (500/503/connection) during any role's call → retried with the existing
  backoff (5s, 15s), same as today, generalized across all model calls (not just the single
  `generate()` hook).
- Non-transient API errors during `plan()` or any `code()` attempt → recorded as the problem's
  error (same as today's `except Exception` path in `main()`), problem marked unsolved, run
  continues to the next problem.

## Testing

The harness has no automated tests today (noted as a known gap in the README), but the user asked
for TDD wherever it's a good fit. The new orchestrator logic is mostly pure functions (string/dict
in, string/dict out) with the network call as the only side effect, so it's a good fit: write
tests first for everything except the literal HTTP call itself.

**TDD, with `_chat()`'s network call mocked/injected:**
- Provider registry lookup: `provider/model` string parsing, unknown-provider error, missing
  base_url/api_key error.
- Strategy dispatch: `plan()` returns `None` for `single`, calls architect once for
  `analyze-then-code`, runs the 2-round debate + judge sequence for `debate` (assert call order
  and prompt contents, e.g. round 2 prompts include round 1's replies).
- `code()` prompt construction: with/without a plan, with/without `prev_code`/`failure` (retry
  case) — assert the right pieces show up in the prompt sent to the coder model.
- Fail-fast startup validation: missing role env vars per strategy, unknown strategy name, unknown
  provider prefix.
- Log event assembly: given a sequence of mocked plan/debate/code calls, assert the resulting
  `plan_events`/`debate_events`/`attempts` structure matches the schema above, and that the file
  written is valid JSON.
- `run_eval.py`'s retry loop calling `code()` with the same `plan` across attempts (i.e., `plan()`
  is called exactly once per problem regardless of retry count).

**Manual / integration verification (not unit-testable without hitting real APIs):**
- Keep these runs token-cheap: `select_hard.py --n 1` (a 1-problem subset) and
  `run_eval.py --max-tests 1 --max-retries 0`, since the goal is to confirm wiring/plumbing works,
  not to measure real solve rate.
- `STRATEGY=single` end-to-end run against a real model reproduces today's behavior (same
  pass/fail on a fixed subset).
- `STRATEGY=analyze-then-code` and `STRATEGY=debate` runs complete end-to-end against real
  DeepSeek/Qwen/MiMo endpoints without crashing, and `logs/run_*.json` contains the expected
  populated events for the strategy used.

## Open items resolved during brainstorming

- Model access: OpenAI-compatible API per provider, via a small provider registry (DeepSeek
  official API, Alibaba DashScope compatible-mode — not a single aggregator, not native
  per-provider SDKs, not local Ollama/vLLM). `logs/` and `.env` stay gitignored since both can
  carry sensitive content (API keys; full prompts/replies that may echo problem statements or
  proprietary reasoning).
- Hybrid flow: both `analyze-then-code` and `debate` supported as switchable strategies.
- Retry scope: coder-only re-run, plan/debate stage runs once per problem.
- Debate composition: 2 debater models + 1 judge model, fixed 2 rounds, not configurable.
- Role model configuration: `.env`, with `--strategy` CLI override for quick switching.
- Log format: one JSON file per run under `logs/`, full event trace (not summary-only).
