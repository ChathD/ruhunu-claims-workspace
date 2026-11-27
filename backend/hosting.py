"""Vercel entrypoint; persistent application data stays in PostgreSQL."""
import os
from pathlib import Path
from tempfile import gettempdir

scratch = Path(gettempdir()) / "ruhunu-claims"
os.environ.setdefault("CLAIMS_DATA_DIR", str(scratch))
os.environ.setdefault("CLAIMS_MODEL_DIR", str(scratch / "models"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(scratch / "models" / "tiktoken"))
if os.getenv("CLAIMS_STORAGE") != "supabase" or not os.getenv("SUPABASE_DATABASE_URL"):
    raise RuntimeError("The hosted workspace requires Supabase PostgreSQL storage.")

from backend.api.workspace_api import app
