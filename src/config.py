from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Bonus: only write facts to User.md when confidence >= this threshold
    confidence_threshold: float = 0.7


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Steps:
    1. Resolve the repo root (defaults to parent of src/).
    2. Load .env if present.
    3. Create state/ directory.
    4. Return a populated LabConfig.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    _load_dotenv(root / ".env")

    provider = normalize_provider(os.environ.get("LLM_PROVIDER", "openai"))
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
    api_key = _api_key_for(provider)
    base_url = os.environ.get("CUSTOM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")

    judge_provider = normalize_provider(os.environ.get("JUDGE_PROVIDER", provider))
    judge_model_name = os.environ.get("JUDGE_MODEL", model_name)
    judge_api_key = _api_key_for(judge_provider)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "2000"))
    compact_keep = int(os.environ.get("COMPACT_KEEP_MESSAGES", "4"))
    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        confidence_threshold=confidence_threshold,
        model=ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        ),
        judge_model=ProviderConfig(
            provider=judge_provider,
            model_name=judge_model_name,
            temperature=0.0,
            api_key=judge_api_key,
            base_url=base_url,
        ),
    )


def _api_key_for(provider: str) -> str | None:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None,
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = mapping.get(provider)
    return os.environ.get(env_var) if env_var else None


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids requiring python-dotenv as a hard dep."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    # Fallback: parse manually
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
