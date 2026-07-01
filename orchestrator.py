"""
orchestrator.py
===============
Multi-model / multi-strategy solving layer consumed by run_eval.py.

Two-phase interface:
    plan(strategy, prompt) -> (plan_text_or_None, events)
    code(strategy, original_prompt, plan_text, prev_code, failure) -> attempt dict

Every model call is a "provider/model" string (e.g. "deepseek/deepseek-v4-flash")
resolved against PROVIDERS to find which base_url/api_key to use. See
docs/superpowers/specs/2026-07-01-multi-model-orchestrator-design.md for the
full design.
"""
from __future__ import annotations

import os


class ConfigError(Exception):
    """Raised when STRATEGY / role model / provider configuration is invalid."""


STRATEGIES = ("single", "analyze-then-code", "debate")

PROVIDERS = {
    "deepseek": {"base_url_env": "DEEPSEEK_BASE_URL", "api_key_env": "DEEPSEEK_API_KEY"},
    "qwen":     {"base_url_env": "QWEN_BASE_URL",     "api_key_env": "QWEN_API_KEY"},
    "mimo":     {"base_url_env": "MIMO_BASE_URL",     "api_key_env": "MIMO_API_KEY"},
    "gemma":    {"base_url_env": "GEMMA_BASE_URL",    "api_key_env": "GEMINI_API_KEY"},
}

_ROLE_ENV = {
    "single": {"single": "SINGLE_MODEL"},
    "analyze-then-code": {"architect": "ARCHITECT_MODEL", "coder": "CODER_MODEL"},
    "debate": {
        "debater1": "DEBATER1_MODEL",
        "debater2": "DEBATER2_MODEL",
        "judge": "JUDGE_MODEL",
        "coder": "CODER_MODEL",
    },
}


def _resolve_role_model(role_model: str) -> tuple:
    """Split "provider/model" and look up its base_url/api_key.
    Returns (provider, model, base_url, api_key). Raises ConfigError."""
    if "/" not in role_model:
        raise ConfigError(
            f"Model {role_model!r} must be formatted as 'provider/model' "
            f"(e.g. 'deepseek/deepseek-v4-flash')"
        )
    provider, model = role_model.split("/", 1)
    if provider not in PROVIDERS:
        raise ConfigError(
            f"Unknown provider {provider!r} in {role_model!r}; "
            f"known providers: {sorted(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    base_url = os.environ.get(cfg["base_url_env"])
    api_key = os.environ.get(cfg["api_key_env"])
    if not base_url or not api_key:
        raise ConfigError(
            f"Provider {provider!r} needs {cfg['base_url_env']} and "
            f"{cfg['api_key_env']} set (used by model {role_model!r})"
        )
    return provider, model, base_url, api_key


def active_role_models(strategy: str) -> dict:
    """Return the role -> 'provider/model' mapping configured for strategy."""
    if strategy not in STRATEGIES:
        raise ConfigError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    out = {}
    for role, env_name in _ROLE_ENV[strategy].items():
        role_model = os.environ.get(env_name)
        if not role_model:
            raise ConfigError(f"STRATEGY={strategy} requires {env_name} to be set (role: {role})")
        out[role] = role_model
    return out


def validate_config(strategy: str) -> None:
    """Fail fast if strategy or any of its role models/providers are misconfigured."""
    role_models = active_role_models(strategy)
    for role_model in role_models.values():
        _resolve_role_model(role_model)
