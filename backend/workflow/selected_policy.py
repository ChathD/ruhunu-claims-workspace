"""Bind a generated submission to one uploaded PDF for triage."""
from sqlalchemy import func, select

from backend.assurance.execution_manifest import execution_manifest
from backend.evidence.policy_schedule import draft_policy_schedule
from backend.evidence.policy_wording_index import GENERATED_WORDING_PRODUCT, current_index, document_kind, searchable_index
from backend.generation.policy_schedule_generator import identity_key
from backend.model_settings import settings
from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import Assessment, Claim, ClaimArchive, PolicyDocument, PolicyDocumentRemoval, PolicySchedule, now
from backend.workflow.investigation_retry import legacy_manual
from backend.workflow.versions import selected_pdf_workflow


def selectable_policy_pdf(document, *, already_selected=False):
    index_ready = searchable_index if already_selected else current_index
    return bool(document and index_ready(document) and document.media_type == "application/pdf"
                and document.product != GENERATED_WORDING_PRODUCT)


def claim_for_policy(session, assessment_id):
    assessment = session.get(Assessment, assessment_id)
    if assessment is None:
        raise KeyError(assessment_id)
    if session.get(ClaimArchive, assessment.claim_id):
        raise ValueError("This claim is archived")
    claim = session.get(Claim, assessment.claim_id)
    if legacy_manual(assessment, claim):
        raise ValueError("New investigations for legacy manual claims are closed.")
    return assessment, claim


def available_wording(session, document_id):
    document = session.get(PolicyDocument, document_id)
    if not selectable_policy_pdf(document) or session.get(PolicyDocumentRemoval, document_id):
        raise ValueError("Select an uploaded policy PDF with the current OpenAI passage index.")
    return document


def schedule_matches(schedule, claim, document):
    return (schedule is not None and schedule["wording"]["id"] == document.id
            and schedule["wording"]["file_hash"] == document.file_hash
            and schedule["claim_payload_hash"] == claim.payload_hash)


def generate_policy_schedule(store, assessment_id, document_id, actor_id):
    with store.session.begin() as session:
        begin_write(session)
        assessment, claim = claim_for_policy(session, assessment_id)
        if assessment.status != "awaiting_policy":
            raise ValueError("This claim is no longer waiting for a policy selection.")
        document = available_wording(session, document_id)
        schedule = assessment.context.get("policy_schedule")
        if schedule_matches(schedule, claim, document):
            return schedule
        schedule = draft_policy_schedule(session, claim, document, document_kind(session, document))
        assessment.context = {**assessment.context, "policy_schedule": schedule}
        store.add_event(session, assessment_id, "policy_schedule_generated", {
            "policy_document_id": document.id, "file_hash": document.file_hash, "staff_id": actor_id,
        })
        return schedule


def queue_with_policy(store, assessment_id, document_id, actor_id):
    """Queue an explicit wording comparison for the legacy evaluation harness."""
    with store.session.begin() as session:
        begin_write(session)
        assessment, claim = claim_for_policy(session, assessment_id)
        selected = assessment.context.get("selected_policy")
        if selected and selected["id"] == document_id:
            return assessment.status
        if assessment.status != "awaiting_policy":
            raise ValueError("This claim is no longer waiting for a policy selection.")
        document = available_wording(session, document_id)
        schedule = assessment.context.get("policy_schedule")
        if not schedule_matches(schedule, claim, document):
            raise ValueError("Generate the policy schedule for this wording before processing the claim.")
        return queue_assessment(store, session, assessment, document, schedule["wording"]["document_kind"], actor_id)


def queue_with_matching_schedule(store, assessment_id, actor_id):
    with store.session.begin() as session:
        begin_write(session)
        assessment, claim = claim_for_policy(session, assessment_id)
        if assessment.context.get("selected_policy") and assessment.status != "awaiting_policy":
            return assessment.status
        if assessment.status != "awaiting_policy":
            raise ValueError("This claim is no longer awaiting processing.")
        holder = identity_key(claim.payload.get("insured_name"))
        registration = identity_key(claim.payload.get("registration"))
        schedule = session.scalar(select(PolicySchedule).where(
            PolicySchedule.holder_key == holder, PolicySchedule.registration_key == registration,
        ))
        if not schedule:
            raise ValueError("No policy schedule matches the reported policyholder and vehicle registration.")
        contract = schedule.payload
        document = session.get(PolicyDocument, schedule.document_id)
        if (not selectable_policy_pdf(document, already_selected=True)
                or document.file_hash != contract["wording"]["file_hash"] or document.version != contract["wording"]["version"]):
            raise ValueError("The wording linked to the matched schedule is unavailable or needs indexing.")
        assessment.context = {**assessment.context, "policy_schedule": contract}
        store.add_event(session, assessment_id, "policy_schedule_matched", {
            "schedule_id": schedule.id, "policy_number": schedule.policy_number, "staff_id": actor_id,
        })
        kind = contract["wording"].get("document_kind") or document_kind(session, document)
        return queue_assessment(store, session, assessment, document, kind, actor_id)


def queue_assessment(store, session, assessment, document, kind, actor_id):
    pending = session.scalar(select(func.count()).select_from(Assessment).where(
        Assessment.status.in_(("queued", "processing")), Assessment.claim_id.not_in(select(ClaimArchive.claim_id)),
    ))
    if pending >= 160:
        raise ValueError("The worker already has 160 pending claims. Wait for processing to finish.")
    selected = {key: getattr(document, key) for key in ("id", "title", "version", "file_hash")}
    selected["document_kind"] = kind
    if page_count := document.embedding.get("page_count"):
        selected["page_count"] = page_count
    if ocr_pages := document.embedding.get("ocr_pages"):
        selected["ocr_pages"] = ocr_pages
    enabled, model = settings()
    workflow = selected_pdf_workflow()
    assessment.context = {**assessment.context, "workflow": workflow, "policy_search_mode": "selected-upload", "selected_policy": selected}
    assessment.execution = {**assessment.execution, **execution_manifest(), "processing_queued_at": now(),
                            "workflow": workflow, "mode": "anthropic-tools" if enabled else "deterministic-tools",
                            "model": model if enabled else None}
    assessment.status = "queued"
    store.add_event(session, assessment.id, "policy_selected", {"policy": selected, "staff_id": actor_id})
    return assessment.status
