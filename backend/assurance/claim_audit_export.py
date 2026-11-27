"""Export and verify a portable claim audit with source, evidence, trace and decision snapshots."""
from sqlalchemy import select

from backend.assurance.investigation_tracing import trace_id_for
from backend.storage.sqlite_store import AgentRun, Assessment, Attachment, AttachmentExtraction, Adjudication, Claim, DecisionRecord, FraudRoutingCorrection, HumanAction, HumanTask, InvestigationRevision, ModelInvocation, PolicyDocument, ToolCall, TraceSpan, WorkflowEvent, digest


def export_audit(store, assessment_id):
    with store.session() as session:
        assessment = session.get(Assessment, assessment_id)
        if assessment is None:
            raise KeyError("Assessment not found")
        claim = session.get(Claim, assessment.claim_id)
        decision = session.get(DecisionRecord, assessment_id)
        if decision is None and assessment.status != "failed":
            raise ValueError("No completed or interrupted investigation to export")
        if digest(claim.payload) != claim.payload_hash or (decision and digest(decision.payload) != decision.payload_hash):
            raise ValueError("Stored evidence integrity check failed")
        correction = session.get(FraudRoutingCorrection, assessment_id)
        if correction and (correction.original_decision_hash != decision.payload_hash or digest(correction.payload) != correction.payload_hash):
            raise ValueError("Stored routing correction integrity check failed")
        runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == assessment_id)).all()
        calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment_id)).all()
        for call in calls:
            if digest(call.response) != call.response_hash:
                raise ValueError("Tool response integrity check failed")
        task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == assessment_id))
        actions = session.scalars(select(HumanAction).where(HumanAction.task_id == task.id)).all() if task else []
        review = session.get(Adjudication, assessment_id)
        attachments = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment_id)).all()
        model_calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == assessment_id).order_by(ModelInvocation.created_at)).all()
        spans = session.scalars(select(TraceSpan).where(TraceSpan.assessment_id == assessment_id)
                                .order_by(TraceSpan.started_at, TraceSpan.id)).all()
        events = session.scalars(select(WorkflowEvent).where(WorkflowEvent.assessment_id == assessment_id)
                                 .order_by(WorkflowEvent.id)).all()
        wording_id = (assessment.context.get("selected_policy", {}).get("id") or assessment.context.get("synthetic_wording_document_id")
                      or assessment.context.get("wording_document_id"))
        wording = session.get(PolicyDocument, wording_id) if wording_id else None
        corpus = [session.get(PolicyDocument, identifier) for identifier in assessment.context.get("policy_corpus_ids", [])]
        extraction = [session.get(AttachmentExtraction, file.id) for file in attachments]
        replacement = session.get(InvestigationRevision, assessment_id)
        package = {"format": "claims-audit-v1", "assessment_id": assessment_id, "submission": claim.payload,
                   "manual_attachments": [{"id": a.id, "name": a.name, "media_type": a.media_type, "base64": a.content} for a in attachments],
                   "attachment_extraction": [item.payload for item in extraction if item],
                   "replacement_assessment_id": replacement.target_id if replacement else None,
                   "staff_decision": {"outcome": review.outcome, "reason": review.reason, "reviewer_id": review.reviewer_id, "created_at": review.created_at} if review else None,
                   "submission_hash": claim.payload_hash, "received_at": claim.received_at,
                   "decision": decision.payload if decision else None, "decision_hash": decision.payload_hash if decision else None,
                   "assessment_status": assessment.status, "assessment_error": assessment.error,
                   "context": assessment.context, "execution": assessment.execution,
                   "model_calls": [{"id": c.id, "stage": c.stage, "status": c.status, "created_at": c.created_at,
                                    "request": c.payload, "response": c.response} for c in model_calls],
                   "trace": {"id": trace_id_for(assessment_id), "spans": [
                       {"id": s.id, "parent_id": s.parent_id, "name": s.name, "kind": s.kind, "started_at": s.started_at,
                        "ended_at": s.ended_at, "duration_ms": s.duration_ms, "error_type": s.error_type, **s.payload}
                       for s in spans]},
                   "workflow_events": [{"kind": event.kind, "created_at": event.created_at, "payload": event.payload}
                                       for event in events],
                   "policy_wording": {"id": wording.id, "title": wording.title, "version": wording.version, "file_hash": wording.file_hash,
                                      "media_type": wording.media_type, "base64": wording.content, "embedding": wording.embedding} if wording else None,
                   "policy_documents": [{"id": doc.id, "title": doc.title, "version": doc.version, "file_hash": doc.file_hash,
                                         "media_type": doc.media_type, "base64": doc.content, "embedding": doc.embedding} for doc in corpus if doc],
                   "agent_runs": [{"stage": r.stage, "instruction": r.instruction, "input_context": r.input_context,
                                   "input_version": r.input_version, "finding": r.finding, "context_after": r.context_after,
                                   "duration_ms": r.duration_ms} for r in runs],
                   "retrievals": [{"stage": c.stage, "tool": c.tool, "request": c.request, "response": c.response,
                                  "response_hash": c.response_hash, "created_at": c.created_at} for c in calls],
                   "human_task": {"id": task.id, "queue": task.queue, "status": task.status} if task else None,
                   "human_actions": [{"reviewer": a.reviewer, "action": a.action, "created_at": a.created_at} for a in actions]}
        if correction:
            event = session.scalar(select(WorkflowEvent).where(WorkflowEvent.assessment_id == assessment_id,
                WorkflowEvent.kind.in_(("fraud_routing_corrected", "evidence_routing_corrected"))))
            package["original_decision"] = {"payload": decision.payload, "hash": decision.payload_hash}
            package["decision"], package["decision_hash"] = correction.payload, correction.payload_hash
            package["routing_correction"] = {"created_at": correction.created_at,
                                             "original_decision_hash": correction.original_decision_hash,
                                             "event": event.payload if event else None}
    package["package_hash"] = digest(package)
    return package


def verify_audit(package):
    original = package.get("original_decision")
    if original and (digest(original["payload"]) != original["hash"]
                     or package.get("routing_correction", {}).get("original_decision_hash") != original["hash"]):
        return False
    return (digest({k: v for k, v in package.items() if k != "package_hash"}) == package["package_hash"]
            and digest(package["submission"]) == package["submission_hash"]
            and (digest(package["decision"]) == package["decision_hash"] if package["decision"] is not None
                 else package.get("assessment_status") == "failed" and package["decision_hash"] is None)
            and digest(package["execution"]["sources"]) == package["execution"]["source_hash"])
