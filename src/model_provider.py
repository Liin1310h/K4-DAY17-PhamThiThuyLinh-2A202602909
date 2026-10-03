from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom, gemini, anthropic, ollama, openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "gpt": "openai",
    "gpt4": "openai",
    "google": "gemini",
    "googleai": "gemini",
    "local": "ollama",
    "or": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map common aliases to canonical provider names."""
    v = value.strip().lower()
    return _ALIASES.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Returns None gracefully when the provider SDK is unavailable,
    allowing callers to fall back to offline mode.
    """
    provider = normalize_provider(config.provider)

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "custom":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url,
            )

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434",
            )

        if provider == "openrouter":
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )

        raise ValueError(f"Unknown provider: {provider!r}")

    except ImportError:
        return None
