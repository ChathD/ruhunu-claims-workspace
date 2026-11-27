"""Run and resume claim investigations from checkpoints and publish decisions through routing controls."""
import copy
import sqlite3
import uuid
from decimal import Decimal
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import select
from backend.assurance.investigation_tracing import claim_trace, trace_span
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, DecisionRecord, FraudRoutingCorrection, HumanTask, digest, now
from backend.claim_schemas import AUTHORITY, Route, STAGES
from backend.workflow.specialist_context import generated_policy_context
from backend.workflow.routing_rules import select_route
from backend.workflow.versions import SUPPORTED, V4
from backend.workflow.investigation_retry import failure_reason, investigation_age, mark_failed


class UnsupportedWorkflowVersion(ValueError):
    """Historical unfinished work needs an explicit new assessment, not replay."""


class ClaimsWorkflow:
    def __init__(self, store, checkpoint_path: Path):
        self.store = store
        if store.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import connect
            from psycopg.rows import dict_row

            url = store.engine.url.set(drivername="postgresql").render_as_string(hide_password=False)
            self.connection = connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row)
            self.connection.execute("SET search_path TO ruhunu, extensions")
            self.checkpointer = PostgresSaver(self.connection)
            self.checkpointer.setup()
        else:
            self.connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            self.checkpointer = SqliteSaver(self.connection)

    def state(self, assessment_id):
        with self.store.session() as session:
            assessment = session.get(Assessment, assessment_id)
            if assessment is None:
                raise ValueError("Assessment not found")
            return assessment

    def publish(self, assessment_id, proposed=None):
        """Recompute in the publication boundary; never trust a proposed route."""
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, assessment_id)
            existing = session.get(DecisionRecord, assessment_id)
            if existing:
                existing = session.get(FraudRoutingCorrection, assessment_id) or existing
                if proposed is not None and proposed != existing.payload:
                    raise ValueError("A published decision is immutable")
                return existing.payload
            claim = session.get(Claim, assessment.claim_id)
            context = copy.deepcopy(assessment.context)
            context["version"] = assessment.version
            context["deadline_exceeded"] = investigation_age(assessment, claim) > 15 * 60
            if reason := failure_reason(context):
                if assessment.status != "failed":
                    mark_failed(self.store, session, assessment, reason)
                assessment.context = context
                return None
            for stage in STAGES:
                if stage not in context["findings"]:
                    context["skipped"].setdefault(stage, "No new historical lead required a follow-up" if stage == "fraud_followup" and "historical" in context["findings"]
                                                  else "Stopped by an earlier coverage, fraud, duplicate or operational gate")
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == assessment_id).order_by(AgentRun.created_at)).all()
            decision = select_route(claim.payload, context, [row.id for row in runs]).model_dump(mode="json")
            if proposed is not None and proposed != decision:
                raise ValueError("Proposed decision violates persisted findings or mandatory routing controls")
            session.add(DecisionRecord(assessment_id=assessment_id, payload=decision, payload_hash=digest(decision)))
            session.flush()
            needs_oversight = claim.payload["currency"] == "LKR" and Decimal(decision["gross_amount"]) > AUTHORITY
            if decision["route"] in {Route.REVIEW.value, Route.FRAUD.value} or needs_oversight:
                session.add(HumanTask(id=str(uuid.uuid4()), assessment_id=assessment_id,
                                      queue="fraud" if decision["route"] == Route.FRAUD.value else "adjuster"))
            assessment.context, assessment.status, assessment.finished_at = context, "complete", now()
            self.store.add_event(session, assessment_id, "decision_published", {"route": decision["route"], "reason_codes": decision["reason_codes"]})
            return decision

    def run(self, assessment_id):
        assessment = self.state(assessment_id)
        if assessment.status not in {"queued", "processing"}:
            return
        with claim_trace(self.store, assessment) as span:
            result = self._run(assessment)
            finished = self.state(assessment_id)
            span.summary = {"status": finished.status, "context_version": finished.version}
            if finished.status == "complete":
                with self.store.session() as session:
                    if decision := session.get(DecisionRecord, assessment_id):
                        span.output = decision.payload
                        span.final_investigation_context = finished.context
                        span.summary.update(route=decision.payload.get("route"), reason_codes=decision.payload.get("reason_codes", []))
            if finished.status == "failed":
                span.error_type = "InvestigationInterrupted"
            return result

    def _run(self, assessment):
        assessment_id = assessment.id
        version = assessment.execution.get("workflow")
        if version not in SUPPORTED or version == V4 and not assessment.context.get("selected_policy"):
            raise UnsupportedWorkflowVersion("Create a new Intake-first investigation; this stored workflow is no longer executable")
        with self.store.session.begin() as session:
            row = session.get(Assessment, assessment_id)
            context = {**row.context, "execution_mode": row.execution.get("mode")}
            claim = session.get(Claim, row.claim_id)
            if row.status == "queued" and claim.payload.get("source") == "Synthetic generator":
                context = generated_policy_context(context)
            row.context = context
        if assessment.context.get("environment") == "manual" and "attachment_extraction" not in assessment.context:
            with self.store.session.begin() as session:
                session.get(Assessment, assessment_id).status = "processing"
            from backend.evidence.uploaded_claim_documents import prepare_attachments
            with trace_span("documents.prepare", kind="tool") as span:
                extracted = prepare_attachments(self.store, assessment_id)
                span.output = extracted
                span.summary = {"statuses": [item["status"] for item in extracted]}
                if any(item.get("technical_failure") for item in extracted):
                    span.error_type = "DocumentExtractionInterrupted"
            with self.store.session.begin() as session:
                current = session.get(Assessment, assessment_id)
                current.context = {**current.context, "attachment_extraction": extracted}
                self.store.add_event(session, assessment_id, "documents_prepared", {"documents": len(extracted)})
        with self.store.session.begin() as session:
            current = session.get(Assessment, assessment_id)
            if reason := failure_reason(current.context):
                mark_failed(self.store, session, current, reason)
                return
        from backend.workflow.intake_first_graph import IntakeFirstWorkflow
        if version == V4:
            if not hasattr(self, "selected_pdf"):
                self.selected_pdf = IntakeFirstWorkflow(self, verify_evidence=True)
            return self.selected_pdf.run(assessment_id)
        if not hasattr(self, "intake_first"):
            self.intake_first = IntakeFirstWorkflow(self)
        return self.intake_first.run(assessment_id)

    def close(self):
        self.connection.close()
