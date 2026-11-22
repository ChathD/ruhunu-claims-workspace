"""Expose the FastAPI app and application factory for the workspace."""

from backend.api.workspace_api import app, create_app

__all__ = ["app", "create_app"]
