"""Provider settings shared by application clients and the Agents SDK bridge."""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlsplit

LLMProvider = Literal["openai", "qwen", "deepseek", "openai_compatible"]


class ProviderConfigurationError(ValueError):
    """An explicitly configured provider must not silently become a demo client."""


class UnsupportedProviderCapability(ValueError):
    """The requested capability has no configured implementation."""


DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "qwen": "qwen-flash",
    "deepseek": "deepseek-flash",
    "openai_compatible": "",
}
DEFAULT_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai_compatible": "",
}
KEY_ENVS = {
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai_compatible": "LLM_API_KEY",
}
URL_ENVS = {
    "qwen": "QWEN_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "openai_compatible": "LLM_BASE_URL",
}


def default_model(provider: str, *, quality: bool = False) -> str:
    if quality and provider == "openai":
        return "gpt-5.4"
    if quality and provider == "qwen":
        return "qwen-plus"
    return DEFAULT_MODELS[provider]


def provider_connection(
    provider: str, *, api_key: str | None = None, base_url: str | None = None
) -> tuple[str, str]:
    """Resolve explicit provider credentials, never another vendor's key/URL."""
    key = (api_key or os.getenv(KEY_ENVS[provider], "") or "").strip()
    url = (
        base_url or os.getenv(URL_ENVS[provider], DEFAULT_URLS[provider]) or ""
    ).strip()
    if not key.strip():
        raise ProviderConfigurationError(f"Set {KEY_ENVS[provider]} for {provider}.")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or "{" in url
        or "}" in url
    ):
        raise ProviderConfigurationError(
            f"Set a valid {URL_ENVS[provider]} API base URL."
        )
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ProviderConfigurationError("Remote model endpoints must use HTTPS.")
    return key, url.rstrip("/")
