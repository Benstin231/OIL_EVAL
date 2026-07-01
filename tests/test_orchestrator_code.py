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
