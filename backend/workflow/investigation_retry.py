"""Record interrupted attempts and queue a fresh investigation revision when a claim can be retried."""
import copy
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select

from backend.storage.sqlite_store import Adjudication, Assessment, Attachment, Claim, InvestigationRevision, Ownership, digest, now
from backend.assurance.execution_manifest import execution_manifest
from backend.claim_schemas import AUTHORITY
from backend.model_settings import settings


def failure_reason(context):
    if reason := context.get("investigation_error"):
        return str(reason)
    if context.get("judge_error"):
        return "Decision synthesis could not complete."
    for stage, finding in context.get("findings", {}).items():
        if error := finding.get("details", {}).get("error_type"):
            return f"{stage.replace('_', ' ').capitalize()} could not complete ({error})."
    if any(document.get("technical_failure") for document in context.get("attachment_extraction", [])):
        return "Document extraction could not complete."
    if context.get("deadline_exceeded"):
        return "The investigation exceeded its processing deadline."
    return None


def investigation_age(assessment, claim):
    started_at = assessment.execution.get("retry_queued_at") or claim.received_at
    return (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()


def interrupted_attention(context, claim):
    from backend.agents.fraud_specialist import high_severity_concerns

    reasons = []
    findings = context.get("findings", {})
    if (any(finding["details"].get("fraud_indicators") for finding in findings.values())
            or high_severity_concerns(findings, claim)):
        reasons.append("Evidence-backed fraud concern")
    if claim.get("currency") == "LKR" and Decimal(claim["gross_amount"]) > AUTHORITY:
        reasons.append("Claim exceeds LKR 250,000")
    return reasons


def mark_failed(store, session, assessment, reason):
    assessment.status, assessment.error, assessment.finished_at = "failed", reason, now()
    store.add_event(session, assessment.id, "workflow_failed", {"reason": reason, "reprocessable": True})


def can_reprocess(assessment, *, adjudicated=False):
    from backend.workflow.specialist_readiness import WORKFLOW

    return (not adjudicated and assessment.execution.get("workflow") == WORKFLOW
            and (assessment.status == "failed" or
                 assessment.status == "complete" and failure_reason(assessment.context) is not None))


def queue_reprocessing(store, session, assessment):
    """Create one successor per attempt. The caller holds a write transaction."""
    from backend.workflow.specialist_readiness import WORKFLOW

    if previous := session.get(InvestigationRevision, assessment.id):
        return previous.target_id
    if assessment.execution.get("workflow") != WORKFLOW:
        raise ValueError("This workflow was retired. Create a new Intake-first investigation instead.")
    if not can_reprocess(assessment, adjudicated=session.get(Adjudication, assessment.id) is not None):
        raise ValueError("Only an interrupted Intake-first investigation can be reprocessed.")
    original = session.get(Claim, assessment.claim_id)
    if digest(original.payload) != original.payload_hash:
        raise ValueError("The original submission failed its integrity check.")
    identifier, claim_id = str(uuid.uuid4()), str(uuid.uuid4())
    files = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment.id)).all()
    file_ids = {file.id: str(uuid.uuid4()) for file in files}
    payload = {**copy.deepcopy(original.payload), "claim_id": claim_id,
               "revision": original.payload.get("revision", 1) + 1,
               "parent_assessment_id": assessment.id, "parent_claim_id": original.id}
    if "attachment_manifest" in payload:
        if any(item["id"] not in file_ids for item in payload["attachment_manifest"]):
            raise ValueError("An original claim attachment is missing.")
        payload["attachment_manifest"] = [{**file, "id": file_ids[file["id"]]} for file in payload["attachment_manifest"]]
    sequence = (session.scalar(select(func.max(Claim.sequence))) or 0) + 1
    session.add(Claim(id=claim_id, sequence=sequence, batch_id=original.batch_id, policy_id=original.policy_id,
                      received_at=original.received_at, payload=payload, payload_hash=digest(payload)))
    session.flush()
    context = {key: copy.deepcopy(value) for key, value in assessment.context.items() if key in {
        "environment", "rules", "policy_search_mode", "policy_corpus_ids", "wording_document_id", "synthetic_wording_document_id"
    }}
    context.update(findings={}, skipped={}, version=0, workflow=WORKFLOW)
    enabled, model = settings()
    execution = {**execution_manifest(), "workflow": WORKFLOW, "retry_queued_at": now(),
                 "original_sequence": assessment.execution.get("original_sequence", original.sequence),
                 "mode": "anthropic-tools" if enabled else "deterministic-tools", "model": model if enabled else None,
                 **{key: assessment.execution[key] for key in ("generation_scenario", "generation_plan", "generation_model")
                    if key in assessment.execution}}
    session.add(Assessment(id=identifier, claim_id=claim_id, context=context, execution=execution))
    session.flush()
    if owner := session.get(Ownership, assessment.id):
        session.add(Ownership(assessment_id=identifier, account_id=owner.account_id))
    session.add_all(Attachment(id=file_ids[file.id], assessment_id=identifier, name=file.name,
                               media_type=file.media_type, content=file.content) for file in files)
    session.add(InvestigationRevision(source_id=assessment.id, target_id=identifier))
    store.add_event(session, identifier, "reprocessing_queued", {"previous_assessment_id": assessment.id})
    return identifier
