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
