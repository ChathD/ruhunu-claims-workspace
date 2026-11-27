"""Resolve the investigation mode and Claude model from process settings."""
import os


def settings():
    mode = os.getenv("CLAIMS_LLM_MODE", "auto")
    if mode not in {"auto", "anthropic", "deterministic"}:
        raise ValueError("Invalid CLAIMS_LLM_MODE")
    enabled = mode == "anthropic" or (mode == "auto" and bool(os.getenv("ANTHROPIC_API_KEY", "").strip()))
    return enabled, os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"
