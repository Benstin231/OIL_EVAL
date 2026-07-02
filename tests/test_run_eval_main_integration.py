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
