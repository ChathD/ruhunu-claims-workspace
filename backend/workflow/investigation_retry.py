"""Record interrupted attempts and queue a fresh investigation revision when a claim can be retried."""
import copy
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import func, select

from backend.anthropic_errors import AnthropicRequestError
from backend.storage.sqlite_store import Adjudication, Assessment, Attachment, Claim, ClaimArchive, InvestigationRevision, Ownership, PolicyDocument, digest, now
from backend.assurance.execution_manifest import execution_manifest
from backend.claim_schemas import AUTHORITY
from backend.evidence.policy_schedule import draft_policy_schedule
from backend.evidence.policy_wording_index import document_kind, searchable_index
from backend.model_settings import settings
from backend.workflow.versions import SUPPORTED


def failure_reason(context):
    if reason := context.get("investigation_error"):
        return str(reason)
    if context.get("judge_error"):
        return "Decision synthesis could not complete."
    for stage, finding in context.get("findings", {}).items():
        details = finding.get("details", {})
        if error := details.get("error_type"):
            label = stage.replace("_", " ").capitalize()
            if error == AnthropicRequestError.__name__ and details.get("error_message"):
                return f"{label} could not complete. {details['error_message']}"
            return f"{label} could not complete ({error})."
    if any(document.get("technical_failure") for document in context.get("attachment_extraction", [])):
        return "Document extraction could not complete."
    if context.get("deadline_exceeded"):
        return "The investigation exceeded its processing deadline."
    return None


def investigation_age(assessment, claim):
    started_at = assessment.execution.get("retry_queued_at") or assessment.execution.get("processing_queued_at") or claim.received_at
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
    store.add_event(session, assessment.id, "workflow_failed", {"reason": reason, "reprocessable": can_reprocess(assessment)})


def legacy_manual(assessment, claim=None):
    return assessment.context.get("environment") == "manual" or claim is not None and claim.batch_id == "manual"


def can_reprocess(assessment, *, adjudicated=False, claim=None):
    return (not adjudicated and not legacy_manual(assessment, claim)
            and assessment.execution.get("workflow") in SUPPORTED
            and (assessment.status == "failed" or
                 assessment.status == "complete" and failure_reason(assessment.context) is not None))


def queue_reprocessing(store, session, assessment):
    """Create one successor per attempt. The caller holds a write transaction."""
    if session.get(ClaimArchive, assessment.claim_id):
        raise ValueError("This claim is archived")
    original = session.get(Claim, assessment.claim_id)
    if legacy_manual(assessment, original):
        raise ValueError("New investigations for legacy manual claims are closed.")
    if previous := session.get(InvestigationRevision, assessment.id):
        return previous.target_id
    workflow = assessment.execution.get("workflow")
    if workflow not in SUPPORTED:
        raise ValueError("This workflow was retired. Create a new Intake-first investigation instead.")
    adjudicated = session.get(Adjudication, assessment.id) is not None
    historical_evidence_removed = False
    if not can_reprocess(assessment, adjudicated=adjudicated):
        finding = assessment.context.get("findings", {}).get("historical", {})
        comparison = finding.get("details", {}).get("amount_justification", {})
        historical_evidence_removed = (assessment.status == "complete" and not adjudicated
            and any(session.get(Claim, identifier) is None for identifier in comparison.get("comparable_claim_ids", [])))
        if not historical_evidence_removed:
            raise ValueError("Only an interrupted Intake-first investigation can be reprocessed.")
    if digest(original.payload) != original.payload_hash:
        raise ValueError("The original submission failed its integrity check.")
    schedule = assessment.context.get("policy_schedule")
    generated_schedule = False
    if (selected := assessment.context.get("selected_policy")) and schedule is None:
        document = session.get(PolicyDocument, selected["id"])
        if not document or document.file_hash != selected["file_hash"] or not searchable_index(document):
            raise ValueError("The selected PDF needs a readable index to generate the policy schedule before retrying.")
        schedule = draft_policy_schedule(session, original, document, selected.get("document_kind") or document_kind(session, document))
        generated_schedule = True
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
        "environment", "rules", "policy_search_mode", "policy_corpus_ids", "wording_document_id", "synthetic_wording_document_id",
        "selected_policy",
    }}
    if schedule:
        context["policy_schedule"] = schedule
    context.update(findings={}, skipped={}, version=0, workflow=workflow)
    enabled, model = settings()
    execution = {**execution_manifest(), "workflow": workflow, "retry_queued_at": now(),
                 "original_sequence": assessment.execution.get("original_sequence", original.sequence),
                 "mode": assessment.execution.get("mode") or ("anthropic-tools" if enabled else "deterministic-tools"),
                 "model": assessment.execution.get("model") if "model" in assessment.execution else model if enabled else None,
                 **{key: assessment.execution[key] for key in ("generation_scenario", "generation_plan", "generation_model")
                    if key in assessment.execution}}
    session.add(Assessment(id=identifier, claim_id=claim_id, context=context, execution=execution))
    session.flush()
    if owner := session.get(Ownership, assessment.id):
        session.add(Ownership(assessment_id=identifier, account_id=owner.account_id))
    session.add_all(Attachment(id=file_ids[file.id], assessment_id=identifier, name=file.name,
                               media_type=file.media_type, content=file.content) for file in files)
    session.add(InvestigationRevision(source_id=assessment.id, target_id=identifier))
    if generated_schedule:
        store.add_event(session, identifier, "policy_schedule_generated", {
            "policy_document_id": selected["id"], "file_hash": selected["file_hash"],
        })
    event = {"previous_assessment_id": assessment.id}
    if historical_evidence_removed:
        event["reason"] = "Historical claims cited in the original investigation were removed."
    store.add_event(session, identifier, "reprocessing_queued", event)
    return identifier
