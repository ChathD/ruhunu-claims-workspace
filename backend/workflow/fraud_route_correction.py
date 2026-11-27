"""Escalate an unresolved published claim when its saved findings support high-severity fraud."""
import uuid

from sqlalchemy import select

from backend.agents.fraud_specialist import high_severity_concerns
from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import Adjudication, AgentRun, Assessment, Claim, DecisionRecord, FraudRoutingCorrection, HumanTask, digest
from backend.claim_schemas import Route
from backend.workflow.routing_rules import select_route


def escalate_published_high_severity(store, assessment_id):
    with store.session.begin() as session:
        begin_write(session)
        existing = session.get(FraudRoutingCorrection, assessment_id)
        if existing:
            return existing.payload
        original = session.get(DecisionRecord, assessment_id)
        if original is None or original.payload["route"] != Route.REVIEW.value or session.get(Adjudication, assessment_id):
            raise ValueError("Only a published adjuster-review claim awaiting adjudication can be corrected")
        assessment = session.get(Assessment, assessment_id)
        claim = session.get(Claim, assessment.claim_id)
        if digest(claim.payload) != claim.payload_hash or digest(original.payload) != original.payload_hash:
            raise ValueError("Stored evidence integrity check failed")
        runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == assessment_id).order_by(AgentRun.created_at)).all()
        findings = assessment.context["findings"]
        concerns = high_severity_concerns(findings, claim.payload)
        recorded = [concern for run in runs for concern in high_severity_concerns({run.finding["agent"]: run.finding}, claim.payload)]
        if not concerns or any(concern not in recorded for concern in concerns):
            raise ValueError("A recorded high-severity Fraud finding with valid citations is required")
        decision = select_route(claim.payload, assessment.context, [run.id for run in runs]).model_dump(mode="json")
        session.add(FraudRoutingCorrection(assessment_id=assessment_id, payload=decision,
                                           payload_hash=digest(decision), original_decision_hash=original.payload_hash))
        task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == assessment_id))
        previous = None
        if task is None:
            task = HumanTask(id=str(uuid.uuid4()), assessment_id=assessment_id)
            session.add(task)
        else:
            previous = {"queue": task.queue, "status": task.status, "reviewer": task.reviewer,
                        "acknowledged_at": task.acknowledged_at}
        task.queue, task.status, task.reviewer, task.acknowledged_at = "fraud", "pending", None, None
        store.add_event(session, assessment_id, "fraud_routing_corrected",
                        {"from_route": original.payload["route"], "route": decision["route"],
                         "reason_codes": decision["reason_codes"], "task_id": task.id, "previous_task": previous})
        return decision
