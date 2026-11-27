"""Resolve the workflow version for a newly queued selected-PDF assessment."""
import os

V3 = "claude-intake-first-v3"
V4 = "claude-intake-first-v4"
SUPPORTED = {V3, V4}


def selected_pdf_workflow():
    version = os.getenv("CLAIMS_SELECTED_PDF_WORKFLOW", "v3")
    if version not in {"v3", "v4"}:
        raise ValueError("CLAIMS_SELECTED_PDF_WORKFLOW must be v3 or v4")
    return V4 if version == "v4" else V3
