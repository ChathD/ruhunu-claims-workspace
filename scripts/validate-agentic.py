"""Claude smoke test for a generated claim assessed against a supplied PDF."""
import argparse
import base64
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "verification"

from sqlalchemy import select
from backend.storage.sqlite_store import Assessment, DecisionRecord, ModelInvocation, PolicyDocument, Store, ToolCall
from backend.generation.scheduled_claim_generator import plan_scheduled_claims, preview_scheduled_batch
from backend.generation.synthetic_claim_generator import commit_batch
from backend.generation.policy_schedule_generator import generate_schedules
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.investigation_tracing import flush_langfuse, trace_id_for
from backend.model_settings import settings
from backend.evidence.policy_wording_index import create_document, index_document
from backend.workflow.claims_workflow import ClaimsWorkflow
from backend.workflow.selected_policy import queue_with_matching_schedule, selectable_policy_pdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("policy_pdf", type=Path, help="Searchable motor policy PDF used for the assessment")
args = parser.parse_args()

if not settings()[0]:
    raise SystemExit("Configure ANTHROPIC_API_KEY and enable Claude before running this smoke test")

directory = ROOT / ".local" / "agentic-validation" / str(uuid.uuid4())
store = Store(directory / "claims.sqlite3")
policy_id = create_document(store, title=args.policy_pdf.stem, product="MOTOR", version="validation",
    filename=args.policy_pdf.name, media_type="application/pdf", content=base64.b64encode(args.policy_pdf.read_bytes()).decode())
index_document(store, policy_id)
with store.session() as session:
    pdf = session.get(PolicyDocument, policy_id)
    if not selectable_policy_pdf(pdf):
        raise SystemExit(pdf.error or "The PDF could not be indexed")
generate_schedules(store, policy_id, 1, "validation-script")
plans = plan_scheduled_claims(store, 1)
batch = commit_batch(store, directory / "labels.sqlite3", preview_scheduled_batch(plans, str(uuid.uuid4())))
identifier = batch["assessment_ids"][0]
queue_with_matching_schedule(store, identifier, "validation-script")
print("Running one generated claim through Claude…", flush=True)
workflow = ClaimsWorkflow(store, directory / "checkpoints.sqlite3")
try:
    workflow.run(identifier)
    package = export_audit(store, identifier)
    (directory / "audit.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    with store.session() as session:
        assessment = session.get(Assessment, identifier)
        decision = session.get(DecisionRecord, identifier)
        calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier)).all()
        tools = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier)).all()
        print(json.dumps({"directory": str(directory), "assessment_id": identifier, "trace_id": trace_id_for(identifier),
            "route": decision.payload["route"] if decision else None,
            "recommendation": decision.payload.get("recommendation") if decision else None,
            "error": assessment.error,
            "rounds": assessment.context.get("investigation_plans"), "model_calls": len(calls),
            "failed_calls": sum(c.status == "failed" for c in calls), "tools": sorted({c.tool for c in tools}),
            "audit_valid": verify_audit(package)}, indent=2), flush=True)
finally:
    workflow.close()
    store.engine.dispose()
    flush_langfuse()
