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
