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
