"""Configuration layer for agent-reforge.

Responsible for merging config from several sources into one clean object that
the rest of the program reads. This is the classic "config layering" pattern:
lower-priority layers are applied first, higher-priority layers override them.

Priority (low -> high), per the project spec:
    1. built-in defaults
    2. ~/.agentreforge/config.json          (not yet: file layers come in a later phase)
    3. project .agentreforge/config.json    (not yet)
    4. project .env                   (not yet)
    5. CLI arguments
    6. process environment variables  (highest)

This minimum version implements only layers 1, 5, and 6 (defaults + CLI + env)
so we can get the shortest runnable loop working first. File-based layers
(2-4) are added in a later phase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- Layer 1: built-in defaults -------------------------------------------------

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

# Per-provider default API base URL (used when no explicit base_url is given).
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

# Per-provider environment variable that holds the API key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


@dataclass
class AgentReforgeConfig:
    """The single, merged configuration object the rest of the app reads."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None


def load_config(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_base_url: str | None = None,
    cli_api_key: str | None = None,
) -> AgentReforgeConfig:
    """Resolve the final config by applying layers low -> high.

    For each field we start from the lowest-priority source and let each higher
    layer overwrite it only when that layer actually provides a value. Writing
    the precedence out explicitly (instead of hiding it in a library) is the
    whole point of this layer: you can see exactly who wins.
    """

    # provider: default < CLI < env
    provider = DEFAULT_PROVIDER
    if cli_provider:
        provider = cli_provider
    if os.getenv("AGENTREFORGE_PROVIDER"):
        provider = os.environ["AGENTREFORGE_PROVIDER"]

    # model: default < CLI < env
    model = DEFAULT_MODEL
    if cli_model:
        model = cli_model
    if os.getenv("AGENTREFORGE_MODEL"):
        model = os.environ["AGENTREFORGE_MODEL"]

    # base_url: provider default < CLI < env
    base_url = PROVIDER_BASE_URLS.get(provider)
    if cli_base_url:
        base_url = cli_base_url
    if os.getenv("AGENTREFORGE_BASE_URL"):
        base_url = os.environ["AGENTREFORGE_BASE_URL"]

    # api_key: CLI < provider-specific env (e.g. OPENAI_API_KEY) < generic env
    api_key = cli_api_key
    key_env = PROVIDER_KEY_ENV.get(provider)
    if key_env and os.getenv(key_env):
        api_key = os.environ[key_env]
    if os.getenv("AGENTREFORGE_API_KEY"):
        api_key = os.environ["AGENTREFORGE_API_KEY"]

    return AgentReforgeConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
