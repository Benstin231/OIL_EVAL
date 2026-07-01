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
