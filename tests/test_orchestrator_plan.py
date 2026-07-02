import orchestrator


def test_plan_single_returns_none_and_no_events():
    plan_text, events = orchestrator.plan("single", "PROBLEM")
    assert plan_text is None
    assert events == []


def test_plan_analyze_then_code_calls_architect(monkeypatch):
    monkeypatch.setenv("ARCHITECT_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("CODER_MODEL", "deepseek/deepseek-v4-flash")
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


def test_plan_unknown_strategy_raises():
    try:
        orchestrator.plan("nonsense", "PROBLEM")
        assert False, "expected ConfigError"
    except orchestrator.ConfigError:
        pass


def test_plan_debate_runs_two_rounds_plus_judge(monkeypatch):
    monkeypatch.setenv("DEBATER1_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("DEBATER2_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setenv("JUDGE_MODEL", "mimo/mimo-v2.5")
    monkeypatch.setenv("CODER_MODEL", "deepseek/deepseek-v4-flash")

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
