"""Prompt loading utilities for agent system prompts."""
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "researcher": "You are a Researcher Agent for SentinAI.",
    "auditor": "You are an Auditor Agent for SentinAI.",
    "analyst": "You are an Analyst Agent for SentinAI.",
}


def _load() -> dict:
    """Load prompts from the YAML configuration file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"

    if not config_path.exists():
        logger.warning("Prompt config file not found at %s", config_path)
        return {}

    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load prompts: %s", exc)
        return {}


def get_prompt(name: str) -> str:
    """Return the system prompt for ``name`` (falls back to a safe default)."""
    return _load().get(name, _DEFAULTS.get(name, ""))
