"""Claude smoke test for a generated claim in an isolated local database (incurs API usage)."""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from sqlalchemy import select
from backend.storage.sqlite_store import Assessment, DecisionRecord, ModelInvocation, Store, ToolCall
from backend.generation.synthetic_claim_generator import commit_batch, preview_batch
from backend.generation.synthetic_policy_generator import generate_people_policies, list_people_policies
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.model_settings import settings
from backend.evidence.policy_wording_index import index_pending
from backend.workflow.claims_workflow import ClaimsWorkflow

if not settings()[0]:
    raise SystemExit("Configure ANTHROPIC_API_KEY and enable Claude before running this smoke test")

directory = ROOT / ".local" / "agentic-validation" / str(uuid.uuid4())
store = Store(directory / "claims.sqlite3")
batch_id = str(uuid.uuid4())
generate_people_policies(store, 1)
index_pending(store)
policy_id = list_people_policies(store)[0]["policy"]["policy_id"]
batch = commit_batch(store, directory / "labels.sqlite3", batch_id, preview_batch(store, 1, batch_id, policy_id))
identifier = batch["assessment_ids"][0]
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
        print(json.dumps({"directory": str(directory), "assessment_id": identifier,
            "route": decision.payload["route"], "recommendation": decision.payload.get("recommendation"),
            "rounds": assessment.context.get("investigation_plans"), "model_calls": len(calls),
            "failed_calls": sum(c.status == "failed" for c in calls), "tools": sorted({c.tool for c in tools}),
            "audit_valid": verify_audit(package)}, indent=2), flush=True)
finally:
    workflow.close()
    store.engine.dispose()
