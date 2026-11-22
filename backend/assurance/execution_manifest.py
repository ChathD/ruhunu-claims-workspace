"""Snapshot backend source, dependency versions and runtime settings for audit reconstruction."""
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path
import platform
from backend.model_settings import settings

from backend.storage.sqlite_store import digest


@lru_cache(maxsize=1)
def execution_manifest():
    # Pin this worker's release once. Restart the backend after source edits.
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)).replace("\\", "/"): p.read_text(encoding="utf-8")
               for p in sorted(root.rglob("*.py"))}
    return {"python": platform.python_version(), "sources": sources, "source_hash": digest(sources),
            "dependencies": {name: version(name) for name in ["langchain", "langgraph", "langfuse", "sqlalchemy", "fastapi", "pypdf", "sentence-transformers"]},
            "mode": "anthropic-tools" if settings()[0] else "deterministic-tools",
            "model": settings()[1] if settings()[0] else None, "rules": "provisional-motor-v1"}
