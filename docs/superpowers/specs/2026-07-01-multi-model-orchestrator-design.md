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

A single OpenAI-compatible client wraps every model call:

```python
LLM_API_KEY=...
LLM_BASE_URL=https://openrouter.ai/api/v1
```

`_chat(model: str, prompt: str) -> str` wraps `openai.OpenAI(base_url=..., api_key=...).chat.completions.create(...)`
and reuses the existing transient-error retry logic (currently `_generate_with_api_retry`,
generalized to accept a model name).

### Strategy & role configuration

Configured via `.env`, with `--strategy` as a CLI override (takes precedence over `.env`'s
`STRATEGY` when passed):

```
STRATEGY=single                        # single | analyze-then-code | debate
SINGLE_MODEL=google/gemma-4-26b-a4b-it
ARCHITECT_MODEL=...
CODER_MODEL=...
DEBATER1_MODEL=deepseek/deepseek-r1
DEBATER2_MODEL=qwen/qwen3-235b
JUDGE_MODEL=google/gemma-4-26b-a4b-it
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
- Transient API errors (500/503/connection) during any role's call → retried with the existing
  backoff (5s, 15s), same as today, generalized across all model calls (not just the single
  `generate()` hook).
- Non-transient API errors during `plan()` or any `code()` attempt → recorded as the problem's
  error (same as today's `except Exception` path in `main()`), problem marked unsolved, run
  continues to the next problem.

## Testing

- No existing automated tests for the harness (noted as a known gap in the README). This spec
  doesn't add a test suite — consistent with the project's current scope — but manual verification
  before considering this done:
  - `STRATEGY=single` end-to-end run reproduces today's behavior (same pass/fail on a fixed
    subset).
  - `STRATEGY=analyze-then-code` and `STRATEGY=debate` runs complete without crashing and produce
    populated `plan_events`/`debate_events` in the log.
  - A retry (forced via a deliberately-broken model reply or a known-hard problem) confirms `plan`
    is reused unchanged across attempts and only the coder step re-runs.
  - `logs/run_*.json` is valid JSON and contains one entry per problem with the expected event
    shape for the strategy used.

## Open items resolved during brainstorming

- Model access: OpenRouter / OpenAI-compatible API (not native per-provider SDKs, not local
  Ollama/vLLM).
- Hybrid flow: both `analyze-then-code` and `debate` supported as switchable strategies.
- Retry scope: coder-only re-run, plan/debate stage runs once per problem.
- Debate composition: 2 debater models + 1 judge model, fixed 2 rounds, not configurable.
- Role model configuration: `.env`, with `--strategy` CLI override for quick switching.
- Log format: one JSON file per run under `logs/`, full event trace (not summary-only).
