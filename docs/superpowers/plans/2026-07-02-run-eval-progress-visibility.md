# run_eval.py Progress Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the active strategy's role models at startup, and print stage-level progress (with `flush=True`) around every blocking model call so a user watching `run_eval.py` output can always tell what step is currently running instead of staring at a blank terminal.

**Architecture:** No new abstractions — plain `print(..., flush=True)` calls added directly in `run_eval.py` and `orchestrator.py`, matching the existing style already used for the API-retry message in `orchestrator._call_with_retry`. No function signatures change (avoids touching the `orchestrator.plan`/`orchestrator.code` contract that existing tests monkeypatch).

**Tech Stack:** Python 3, pytest (`capsys` fixture for stdout assertions), existing `monkeypatch`-based test style already used across `tests/`.

## Global Constraints

- Do not change the signatures of `orchestrator.plan()` or `orchestrator.code()` — several existing tests monkeypatch these with the current arity (`tests/test_run_eval_solve_problem.py`, `tests/test_run_eval_main_integration.py`).
- Every new progress `print()` that sits around a blocking model call (Tasks 2-4) must pass `flush=True` (spec: avoid buffered output stalling on Windows during blocking network calls). The one-shot startup banner (Task 1) is explicitly exempt — clarified 2026-07-02 after task review — since it prints once, outside any wait loop, and is immediately followed by normal execution.
- No background threads, spinners, or heartbeat timers (spec's explicit non-goal — user chose stage-level messages only).
- No per-test-case printing inside `evaluate_problem()` (spec's explicit non-goal).
- Follow the design in `docs/superpowers/specs/2026-07-02-run-eval-progress-visibility-design.md`.

---

### Task 1: Startup models banner

**Files:**
- Modify: `run_eval.py` (add helper function near `load_subset` at line 299; modify `main()` around line 335)
- Test: `tests/test_run_eval_main_integration.py`

**Interfaces:**
- Produces: `run_eval._format_models_lines(role_models: dict) -> list[str]` — pure formatter, no I/O. Later tasks do not depend on this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_eval_main_integration.py`:

```python
def test_format_models_lines_single():
    lines = run_eval._format_models_lines({"single": "deepseek/deepseek-v4-flash"})
    assert lines == ["  SINGLE_MODEL = deepseek/deepseek-v4-flash"]


def test_format_models_lines_preserves_order():
    lines = run_eval._format_models_lines({
        "debater1": "deepseek/deepseek-v4-flash",
        "debater2": "qwen/qwen3-max",
        "judge": "gemma/gemma-3-27b",
        "coder": "deepseek/deepseek-v4-flash",
    })
    assert lines == [
        "  DEBATER1_MODEL = deepseek/deepseek-v4-flash",
        "  DEBATER2_MODEL = qwen/qwen3-max",
        "  JUDGE_MODEL = gemma/gemma-3-27b",
        "  CODER_MODEL = deepseek/deepseek-v4-flash",
    ]


def test_main_prints_models_banner(tmp_path, monkeypatch, capsys):
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

    out = capsys.readouterr().out
    assert "Models:" in out
    assert "SINGLE_MODEL = deepseek/deepseek-v4-flash" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_eval_main_integration.py -v`
Expected: `test_format_models_lines_single` and `test_format_models_lines_preserves_order` FAIL with `AttributeError: module 'run_eval' has no attribute '_format_models_lines'`; `test_main_prints_models_banner` FAILs on the `"Models:" in out` assertion.

- [ ] **Step 3: Implement**

In `run_eval.py`, add this function after `load_subset` (after line 299, before `def main() -> None:`):

```python
def _format_models_lines(role_models: dict) -> list:
    """Format a role->'provider/model' mapping as 'ROLE_MODEL = value' lines,
    in the same order as active_role_models() returns them."""
    return [f"  {role.upper()}_MODEL = {model}" for role, model in role_models.items()]
```

Then in `main()`, replace this line (currently line 335):

```python
    print(f"Loaded {len(problems)} problems from {args.subset}  [strategy={strategy}]\n")
```

with:

```python
    role_models = orchestrator.active_role_models(strategy)
    print(f"Loaded {len(problems)} problems from {args.subset}  [strategy={strategy}]")
    print("Models:")
    for line in _format_models_lines(role_models):
        print(line)
    print()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_eval_main_integration.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest -v`
Expected: all PASS (no test asserts on the old single-line banner text).

- [ ] **Step 6: Commit**

```bash
git add run_eval.py tests/test_run_eval_main_integration.py
git commit -m "feat: print active role models at run_eval startup"
```

---

### Task 2: Per-problem header line + attempt-level progress prints

**Files:**
- Modify: `run_eval.py` — `main()` loop (around line 340-344), `solve_problem()` (lines 234-286)
- Test: `tests/test_run_eval_solve_problem.py`, `tests/test_run_eval_main_integration.py`

**Interfaces:**
- Consumes: `orchestrator.active_role_models(strategy) -> dict` (existing, from Task 1's usage — same function, called again here since `solve_problem()` doesn't currently receive it).
- No signature changes to `solve_problem()` or `orchestrator.code()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_eval_solve_problem.py`:

```python
def test_solve_problem_prints_attempt_progress(monkeypatch, capsys):
    monkeypatch.setenv("SINGLE_MODEL", "deepseek/deepseek-v4-flash")

    def fake_plan(strategy, prompt):
        return None, []

    def fake_code(strategy, original_prompt, plan_text, prev_code, failure):
        return {"model": "m", "prompt": "cp", "reply": "bad", "code": "bad", "duration_s": 0.1}

    def fake_evaluate(p, code, timeout, max_tests):
        return {"solved": False, "num_tests": 1, "tests_passed": 0,
                "first_failure": {"index": 0, "type": "WRONG_ANSWER",
                                   "input": "1", "expected": "2", "got": "3"}}

    monkeypatch.setattr(run_eval.orchestrator, "plan", fake_plan)
    monkeypatch.setattr(run_eval.orchestrator, "code", fake_code)
    monkeypatch.setattr(run_eval, "evaluate_problem", fake_evaluate)

    run_eval.solve_problem(_problem(), "single", timeout=5.0, max_tests=0, max_retries=1)

    out = capsys.readouterr().out
    assert "coding attempt 1/2 (deepseek/deepseek-v4-flash)..." in out
    assert "coding attempt 1/2 done (0.1s), running tests..." in out
    assert "attempt 1: 0/1 tests passed, retrying..." in out
    assert "coding attempt 2/2 (deepseek/deepseek-v4-flash)..." in out
    last_attempt_line = out.split("coding attempt 2/2 done")[1]
    assert "retrying..." not in last_attempt_line
```

Add to `tests/test_run_eval_main_integration.py` (reuses the same setup as `test_main_prints_models_banner`; write it as a separate self-contained test):

```python
def test_main_prints_problem_header_before_result(tmp_path, monkeypatch, capsys):
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

    out = capsys.readouterr().out
    header_idx = out.index("[1/1] hard    Two Sum")
    result_idx = out.index("[1/1] PASS")
    assert header_idx < result_idx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_eval_solve_problem.py tests/test_run_eval_main_integration.py -v`
Expected: `test_solve_problem_prints_attempt_progress` FAILs (no such output printed yet); `test_main_prints_problem_header_before_result` FAILs with `ValueError: substring not found` on `header_idx`.

- [ ] **Step 3: Implement**

In `run_eval.py`, inside `solve_problem()` (currently lines 234-286), after the `plan()` call succeeds and before the retry loop (i.e. right before the existing `prev_code = None` block at line 253), add:

```python
    role_models = orchestrator.active_role_models(strategy)
    coder_model = role_models.get("coder", role_models.get("single"))
    total_attempts = max_retries + 1
```

Then replace the `for attempt in range(1, max_retries + 2):` loop body (lines 258-279) with:

```python
    for attempt in range(1, max_retries + 2):
        print(f"        -> coding attempt {attempt}/{total_attempts} ({coder_model})...",
              flush=True)
        try:
            attempt_result = orchestrator.code(strategy, original_prompt, plan_text, prev_code, failure)
        except Exception as exc:
            log_entry["error"] = str(exc)
            return {**_meta(p), "solved": False, "error": str(exc), "attempts": attempt}, log_entry

        code_text = attempt_result["code"]
        print(f"        -> coding attempt {attempt}/{total_attempts} done "
              f"({attempt_result['duration_s']:.1f}s), running tests...", flush=True)
        ev = evaluate_problem(p, code_text, timeout, max_tests)
        will_retry = not ev["solved"] and attempt < total_attempts
        print(f"        -> attempt {attempt}: {ev['tests_passed']}/{ev['num_tests']} tests passed"
              + (", retrying..." if will_retry else ""), flush=True)
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
```

In `main()`, inside the `for n, p in enumerate(problems, 1):` loop (currently lines 340-344), after `diff = p.get("difficulty", "?")` and before the call to `solve_problem(...)`, add:

```python
        print(f"[{n}/{len(problems)}] {diff:6s}  {title}", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_eval_solve_problem.py tests/test_run_eval_main_integration.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add run_eval.py tests/test_run_eval_solve_problem.py tests/test_run_eval_main_integration.py
git commit -m "feat: print per-problem header and per-attempt progress during run_eval"
```

---

### Task 3: Plan-stage progress prints for analyze-then-code

**Files:**
- Modify: `orchestrator.py` — `_run_analyze_then_code_plan()` (lines 204-212)
- Test: `tests/test_orchestrator_plan.py`

**Interfaces:**
- No signature change to `_run_analyze_then_code_plan()` or `plan()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator_plan.py`:

```python
def test_plan_analyze_then_code_prints_progress(monkeypatch, capsys):
    monkeypatch.setenv("ARCHITECT_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("CODER_MODEL", "deepseek/deepseek-v4-flash")

    def fake_chat(role_model, prompt):
        return "use a two-pointer approach", 1.1

    monkeypatch.setattr(orchestrator, "_chat", fake_chat)
    orchestrator.plan("analyze-then-code", "PROBLEM STATEMENT")

    out = capsys.readouterr().out
    assert "planning (architect, deepseek/deepseek-v4-flash)..." in out
    assert "planning done (1.1s)" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_plan.py::test_plan_analyze_then_code_prints_progress -v`
Expected: FAIL — captured stdout is empty, assertion on `"planning (architect, ..." in out` fails.

- [ ] **Step 3: Implement**

In `orchestrator.py`, replace `_run_analyze_then_code_plan` (lines 204-212):

```python
def _run_analyze_then_code_plan(prompt: str) -> tuple:
    role_model = active_role_models("analyze-then-code")["architect"]
    architect_prompt = _ARCHITECT_PROMPT_TMPL.format(problem=prompt)
    print(f"        -> planning (architect, {role_model})...", flush=True)
    reply, duration = _chat(role_model, architect_prompt)
    print(f"        -> planning done ({duration:.1f}s)", flush=True)
    event = {
        "role": "architect", "model": role_model,
        "prompt": architect_prompt, "reply": reply, "duration_s": duration,
    }
    return reply, [event]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_plan.py -v`
Expected: all PASS (including the pre-existing `test_plan_analyze_then_code_calls_architect`).

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_plan.py
git commit -m "feat: print architect planning progress in analyze-then-code strategy"
```

---

### Task 4: Plan-stage progress prints for debate

**Files:**
- Modify: `orchestrator.py` — `_run_debate_plan()` (lines 239-269)
- Test: `tests/test_orchestrator_plan.py`

**Interfaces:**
- No signature change to `_run_debate_plan()` or `plan()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator_plan.py`:

```python
def test_plan_debate_prints_progress_for_all_five_calls(monkeypatch, capsys):
    monkeypatch.setenv("DEBATER1_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("DEBATER2_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("JUDGE_MODEL", "mimo/mimo-v2.5")
    monkeypatch.setenv("CODER_MODEL", "deepseek/deepseek-v4-flash")

    def fake_chat(role_model, prompt):
        return "approach", 0.1

    monkeypatch.setattr(orchestrator, "_chat", fake_chat)
    orchestrator.plan("debate", "PROBLEM")

    out = capsys.readouterr().out
    assert "debate round 1: debater1 (deepseek/deepseek-v4-flash)..." in out
    assert "debate round 1: debater1 done (0.1s)" in out
    assert "debate round 1: debater2 (qwen/qwen3.7-plus)..." in out
    assert "debate round 1: debater2 done (0.1s)" in out
    assert "debate round 2: debater1..." in out
    assert "debate round 2: debater2..." in out
    assert "judge synthesizing..." in out
    assert "judge done (0.1s)" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator_plan.py::test_plan_debate_prints_progress_for_all_five_calls -v`
Expected: FAIL — captured stdout is empty.

- [ ] **Step 3: Implement**

In `orchestrator.py`, replace `_run_debate_plan` (lines 239-269):

```python
def _run_debate_plan(prompt: str) -> tuple:
    events = []
    role_models = active_role_models("debate")
    d1_model = role_models["debater1"]
    d2_model = role_models["debater2"]
    judge_model = role_models["judge"]

    r1_prompt = _DEBATE_ROUND1_TMPL.format(problem=prompt)
    print(f"        -> debate round 1: debater1 ({d1_model})...", flush=True)
    d1_r1, dur = _chat(d1_model, r1_prompt)
    print(f"        -> debate round 1: debater1 done ({dur:.1f}s)", flush=True)
    events.append({"round": 1, "role": "debater1", "model": d1_model,
                    "prompt": r1_prompt, "reply": d1_r1, "duration_s": dur})

    print(f"        -> debate round 1: debater2 ({d2_model})...", flush=True)
    d2_r1, dur = _chat(d2_model, r1_prompt)
    print(f"        -> debate round 1: debater2 done ({dur:.1f}s)", flush=True)
    events.append({"round": 1, "role": "debater2", "model": d2_model,
                    "prompt": r1_prompt, "reply": d2_r1, "duration_s": dur})

    d1_r2_prompt = _DEBATE_ROUND2_TMPL.format(problem=prompt, own=d1_r1, other=d2_r1)
    print(f"        -> debate round 2: debater1...", flush=True)
    d1_r2, dur = _chat(d1_model, d1_r2_prompt)
    print(f"        -> debate round 2: debater1 done ({dur:.1f}s)", flush=True)
    events.append({"round": 2, "role": "debater1", "model": d1_model,
                    "prompt": d1_r2_prompt, "reply": d1_r2, "duration_s": dur})

    d2_r2_prompt = _DEBATE_ROUND2_TMPL.format(problem=prompt, own=d2_r1, other=d1_r1)
    print(f"        -> debate round 2: debater2...", flush=True)
    d2_r2, dur = _chat(d2_model, d2_r2_prompt)
    print(f"        -> debate round 2: debater2 done ({dur:.1f}s)", flush=True)
    events.append({"round": 2, "role": "debater2", "model": d2_model,
                    "prompt": d2_r2_prompt, "reply": d2_r2, "duration_s": dur})

    judge_prompt = _JUDGE_TMPL.format(problem=prompt, d1=d1_r2, d2=d2_r2)
    print(f"        -> judge synthesizing...", flush=True)
    judge_reply, dur = _chat(judge_model, judge_prompt)
    print(f"        -> judge done ({dur:.1f}s)", flush=True)
    events.append({"role": "judge", "model": judge_model,
                    "prompt": judge_prompt, "reply": judge_reply, "duration_s": dur})

    return judge_reply, events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator_plan.py -v`
Expected: all PASS (including the pre-existing `test_plan_debate_runs_two_rounds_plus_judge`).

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_plan.py
git commit -m "feat: print debate round and judge progress during plan stage"
```

---

### Task 5: Total elapsed time at end of run

> Added 2026-07-03 after the user asked, mid-implementation, for a total wall-clock time to complement the per-stage timings from Tasks 2-4.

**Files:**
- Modify: `run_eval.py` — `main()` (add a start-time capture near the top of the function, and a total-time print after the `SCORE` line)
- Test: `tests/test_run_eval_main_integration.py`

**Interfaces:**
- Produces: `run_eval._format_duration(seconds: float) -> str` — pure formatter, e.g. `12.3` -> `"12.3s"`, `272.0` -> `"4m32s"`. No other task depends on this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_eval_main_integration.py`:

```python
def test_format_duration_under_a_minute():
    assert run_eval._format_duration(12.34) == "12.3s"


def test_format_duration_over_a_minute():
    assert run_eval._format_duration(272.0) == "4m32s"


def test_main_prints_total_time_after_score(tmp_path, monkeypatch, capsys):
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

    out = capsys.readouterr().out
    score_idx = out.index("==== SCORE:")
    total_time_idx = out.index("Total time:")
    assert score_idx < total_time_idx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_eval_main_integration.py -q`
Expected: `test_format_duration_under_a_minute` and `test_format_duration_over_a_minute` FAIL with `AttributeError: module 'run_eval' has no attribute '_format_duration'`; `test_main_prints_total_time_after_score` FAILs with `ValueError: substring not found` on `total_time_idx`.

- [ ] **Step 3: Implement**

In `run_eval.py`, add this function near `_format_models_lines` (same helper-function area, before `def main() -> None:`):

```python
def _format_duration(seconds: float) -> str:
    """Format a duration as '12.3s' (under a minute) or '4m32s' (a minute or more)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs}s"
```

In `main()`, find the current top of the function — read the file first, since Tasks 1-2 already changed line numbers from the original plan. Add a start-time capture as the first statement inside `main()`, before argument parsing:

```python
def main() -> None:
    run_start = time.monotonic()
    ap = argparse.ArgumentParser(description="LiveBench-style code eval harness")
    ...
```

(`time` is already imported at the top of `run_eval.py` — it's used later for the log-file timestamp. No new import needed.)

Then find the existing line that prints the score summary:

```python
    print(f"\n==== SCORE: {solved}/{total} solved  ({score:.0%}) ====")
```

Add immediately after it:

```python
    print(f"Total time: {_format_duration(time.monotonic() - run_start)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_eval_main_integration.py -q`
Expected: all PASS.

- [ ] **Step 5: Run full suite to check for regressions**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add run_eval.py tests/test_run_eval_main_integration.py
git commit -m "feat: print total elapsed time at end of run_eval"
```
