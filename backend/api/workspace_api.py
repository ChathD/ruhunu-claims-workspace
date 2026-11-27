"""Create the FastAPI workspace, connect its route groups and deliver claim updates over SSE."""
import asyncio
import json
import os
import threading
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import func, select

from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import Adjudication, AgentRun, Assessment, Claim, ClaimArchive, DecisionRecord, FraudRoutingCorrection, HumanAction, HumanTask, InvestigationRevision, ModelInvocation, Store, SyntheticClaimPreview, ToolCall, WorkflowEvent, now
from backend.generation.claude_generation_client import GenerationError
from backend.generation.scheduled_claim_generator import plan_scheduled_claims, preview_scheduled_batch
from backend.generation.synthetic_claim_generator import commit_batch, commit_pending_preview
from backend.claim_schemas import AcknowledgeRequest, ClaimPreviewRequest, Route
from backend.assurance.claim_audit_export import export_audit
from backend.assurance.routing_evaluation import evaluate
from backend.assurance.synthetic_dataset import dataset_csv, dataset_records, dataset_zip
from backend.assurance.investigation_display import completed_model_assignments, display_finding, specialist_inputs, staff_justification
from backend.assurance.investigation_tracing import flush_langfuse, trace_id_for
from backend.workflow.investigation_retry import can_reprocess, failure_reason, interrupted_attention, queue_reprocessing
from backend.workflow.investigation_worker import Worker
from backend.workflow.claims_workflow import ClaimsWorkflow
from backend.workflow.selected_policy import queue_with_matching_schedule
from backend.workflow.versions import selected_pdf_workflow


def create_app(data_dir=None, auto_seed=False, *, database_url=None):
    if data_dir is None and os.getenv("CLAIMS_STORAGE") == "supabase":
        database_url = database_url or os.getenv("SUPABASE_DATABASE_URL")
        if not database_url:
            raise ValueError("Set SUPABASE_DATABASE_URL for Supabase storage.")
    data_dir = Path(data_dir or os.getenv("CLAIMS_DATA_DIR", ".local/workspace"))
    data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(data_dir / "claims.sqlite3", database_url=database_url)
    if database_url:
        store.initialize_postgres()
    labels_path = None if database_url else data_dir / "labels.sqlite3"
    generation_lock = threading.Lock()
    discarded_fixtures = deque(maxlen=24)

    @asynccontextmanager
    async def lifespan(app):
        if auto_seed:
            with store.session() as session:
                empty = session.scalar(select(func.count()).select_from(Claim)) == 0
            if empty:
                batch_id = str(uuid.uuid4())
                plans = plan_scheduled_claims(store, 10)
                commit_batch(store, labels_path, preview_scheduled_batch(plans, batch_id))
        workflow = ClaimsWorkflow(store, data_dir / "checkpoints.sqlite3")
        worker = Worker(store, workflow)
        app.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            worker.close()
            workflow.close()
            store.engine.dispose()
            flush_langfuse()

    app = FastAPI(title="Ruhunu Claims Workspace", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.labels_path = labels_path
    from backend.api.portal_routes import install_portal
    install_portal(app, store)
    from backend.api.policy_document_routes import install_policy_api
    install_policy_api(app, store)

    def claim_rows(assessment_id=None):
        with store.session() as session:
            query = select(Claim).order_by(Claim.sequence.desc())
            if assessment_id is None:
                query = query.where(Claim.id.not_in(select(ClaimArchive.claim_id)))
            claims = session.scalars(query).all()
            assessments = {row.claim_id: row for row in session.scalars(select(Assessment))}
            decisions = {row.assessment_id: row for row in session.scalars(select(DecisionRecord))}
            decisions.update({row.assessment_id: row for row in session.scalars(select(FraudRoutingCorrection))})
            tasks = {row.assessment_id: row for row in session.scalars(select(HumanTask))}
            replacements = {row.source_id: row.target_id for row in session.scalars(select(InvestigationRevision))}
            adjudicated = set(session.scalars(select(Adjudication.assessment_id)))
        rows = []
        for claim in claims:
            assessment = assessments[claim.id]
            replacement_id = replacements.get(assessment.id)
            if (assessment_id is None and replacement_id) or (assessment_id and assessment.id != assessment_id):
                continue
            retryable = can_reprocess(assessment, adjudicated=assessment.id in adjudicated)
            decision = decisions.get(assessment.id) if not retryable else None
            task = tasks.get(assessment.id) if decision else None
            reference = assessment.execution.get("original_sequence", claim.sequence)
            rows.append({"id": assessment.id, "claim_id": claim.id, "batch_id": claim.batch_id, "reference": f"CLM-{reference:04d}",
                         "type": claim.payload["loss_type"], "gross_amount": claim.payload["gross_amount"],
                         "registration": claim.payload["registration"], "received_at": claim.received_at,
                         "finished_at": assessment.finished_at, "status": "failed" if retryable else assessment.status,
                         "can_reprocess": retryable and not replacement_id, "replacement_id": replacement_id,
                         "processing_alerts": interrupted_attention(assessment.context, claim.payload) if retryable and not replacement_id else [],
                         "route": decision.payload["route"] if decision else None,
                         "task": {"id": task.id, "queue": task.queue, "status": task.status, "reviewer": task.reviewer} if task else None})
        return rows

    @app.get("/api/health")
    def health():
        from backend.model_settings import settings
        enabled, model = settings()
        return {"status": "ok", "environment": "synthetic", "rules": "provisional",
                "storage": "supabase" if store.database_url else "sqlite",
                "mode": "claude-intake-first-v3", "selected_pdf_workflow": selected_pdf_workflow(),
                "execution_mode": "anthropic-tools" if enabled else "deterministic-tools", "model": model if enabled else None}

    @app.get("/api/overview")
    def overview():
        rows = claim_rows()
        complete = [r for r in rows if r["route"]]
        breakdown = {route.value: sum(r["route"] == route.value for r in rows) for route in Route}
        durations = [(datetime.fromisoformat(r["finished_at"]) - datetime.fromisoformat(r["received_at"])).total_seconds()
                     for r in complete if r["finished_at"]]
        days = [(datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat() for i in reversed(range(7))]
        volume = [{"date": day, "label": datetime.fromisoformat(day).strftime("%a"),
                   "claims": sum(r["received_at"][:10] == day for r in rows)} for day in days]
        with store.session() as session:
            model_calls = session.scalar(select(func.count()).select_from(ModelInvocation))
        return {"total": len(rows), "completed": len(complete), "processing": sum(r["status"] in {"queued", "processing"} for r in rows),
                "failed": sum(r["status"] == "failed" for r in rows),
                "human_pending": sum(bool(r["processing_alerts"] or r["task"] and r["task"]["status"] == "pending") for r in rows),
                "within_15_minutes": round(sum(x <= 900 for x in durations) / len(durations) * 100, 1) if durations else None,
                "average_seconds": round(sum(durations) / len(durations), 2) if durations else None,
                "cost_per_claim": None, "model_calls": model_calls, "breakdown": breakdown, "volume": volume,
                "recent_claims": rows[:6], "rules": "Provisional · synthetic only"}

    @app.get("/api/claims")
    def claims():
        return {"claims": claim_rows()}

    @app.get("/api/claims/{assessment_id}")
    def detail(assessment_id: str):
        summary = next(iter(claim_rows(assessment_id)), None)
        if summary is None:
            raise HTTPException(404, "Assessment not found")
        with store.session() as session:
            assessment = session.get(Assessment, assessment_id)
            claim = session.get(Claim, assessment.claim_id)
            decision = session.get(FraudRoutingCorrection, assessment_id) or session.get(DecisionRecord, assessment_id)
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == assessment_id).order_by(AgentRun.created_at)).all()
            calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment_id).order_by(ToolCall.id)).all()
            history = next((call.response["records"] for call in reversed(calls)
                            if call.tool in {"retrieve_comparable_claims", "search_history"} and "records" in call.response), [])
            selected_policy = assessment.context.get("selected_policy")
            context = {**assessment.context, "findings": {
                stage: display_finding(finding, claim.payload, selected_policy)
                for stage, finding in assessment.context.get("findings", {}).items()}}
            triage = ({**decision.payload, "justification": staff_justification(decision.payload, assessment.context, claim.payload)}
                      if decision and summary["route"] else None)
            return {**summary, "submission": claim.payload, "context": context, "version": assessment.version,
                    "trace_id": trace_id_for(assessment_id),
                    "synthetic_evidence": {"historical_costs": history} if claim.batch_id != "manual" else None,
                    "error": assessment.error or (failure_reason(assessment.context) if summary["status"] == "failed" else None),
                    "decision": triage,
                    "investigation_tasks": completed_model_assignments(session, assessment, runs),
                    "tool_calls": [{"id": c.id, "stage": c.stage, "tool": c.tool, "request": c.request, "response": c.response} for c in calls],
                    "runs": [{"id": r.id, "stage": r.stage, "finding": display_finding(r.finding, claim.payload, selected_policy), "instruction": r.instruction,
                              "input_version": r.input_version, "duration_ms": r.duration_ms} for r in runs]}

    @app.post("/api/generator/preview")
    def preview(request: ClaimPreviewRequest, http_request: Request):
        batch_id = str(uuid.uuid4())
        try:
            plans = plan_scheduled_claims(store, request.count)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        with generation_lock:
            recent_discards = list(discarded_fixtures)
        try:
            batch = preview_scheduled_batch(plans, batch_id, discarded_previews=recent_discards,
                                            actor_id=http_request.state.actor.id)
        except GenerationError as error:
            raise HTTPException(503, str(error)) from error
        with store.session.begin() as session:
            session.add(SyntheticClaimPreview(batch_id=batch_id, payload=batch))
        fixtures = [{key: item[key] for key in ("claim", "scenario", "scenario_type")}
                    for item in batch["fixtures"]]
        return {"batch_id": batch_id, "count": len(fixtures), "fixtures": fixtures}

    @app.post("/api/generator/preview/{batch_id}/commit", status_code=201)
    def commit_preview(batch_id: str):
        with generation_lock:
            result = commit_pending_preview(store, labels_path, batch_id)
        if result is None:
            raise HTTPException(404, "This preview is unavailable. Discard it and generate a new preview.")
        return result

    @app.get("/api/generator/batches/{batch_id}/dataset.{format}")
    def export_dataset(batch_id: str, format: Literal["json", "csv", "zip"]):
        records = dataset_records(store, labels_path, batch_id)
        if not records:
            raise HTTPException(404, "No unstructured records found for this batch")
        headers = {"Content-Disposition": f'attachment; filename="synthetic_auto_claims.{format}"'}
        if format == "zip":
            archive = dataset_zip(store, records)
            if archive is None:
                raise HTTPException(404, "No linked policy schedules and attachments found for this batch")
            return Response(archive, media_type="application/zip", headers=headers)
        if format == "csv":
            return Response(dataset_csv(records), media_type="text/csv", headers=headers)
        return JSONResponse(records, headers=headers)

    @app.delete("/api/generator/preview/{batch_id}", status_code=204)
    def discard_preview(batch_id: str):
        with generation_lock:
            with store.session.begin() as session:
                begin_write(session)
                pending = session.get(SyntheticClaimPreview, batch_id)
                if pending is None:
                    return
                session.delete(pending)
            discarded_fixtures.extend(item["claim"]["narrative"] for item in pending.payload["fixtures"])

    @app.post("/api/claims/{assessment_id}/process", status_code=202)
    def process(assessment_id: str, request: Request):
        try:
            status = queue_with_matching_schedule(store, assessment_id, request.state.actor.id)
        except KeyError as error:
            raise HTTPException(404, "Assessment not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        if status == "queued":
            app.state.worker.wake.set()
        return {"id": assessment_id, "status": status}

    @app.post("/api/generator/batches/{batch_id}/start", status_code=202)
    def start_batch(batch_id: str):
        if batch_id == "manual":
            raise HTTPException(409, "New investigations for legacy manual claims are closed.")
        with generation_lock:
            with store.session.begin() as session:
                assessments = session.scalars(select(Assessment).join(Claim).where(
                    Claim.batch_id == batch_id, Claim.id.not_in(select(ClaimArchive.claim_id))
                )).all()
                if not assessments:
                    raise HTTPException(404, "Batch not found")
                if any(row.status != "awaiting_review" for row in assessments):
                    raise HTTPException(409, "This batch has already started")
                for row in assessments:
                    row.status = "queued"
                    store.add_event(session, row.id, "review_approved", {"status": "queued"})
        app.state.worker.wake.set()
        return {"batch_id": batch_id, "count": len(assessments), "status": "queued"}

    @app.post("/api/claims/{assessment_id}/retry", status_code=202)
    def retry(assessment_id: str):
        with store.session.begin() as session:
            begin_write(session)
            assessment = session.get(Assessment, assessment_id)
            if assessment is None:
                raise HTTPException(404, "Assessment not found")
            try:
                identifier = queue_reprocessing(store, session, assessment)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
        app.state.worker.wake.set()
        return {"id": identifier, "status": "queued"}

    @app.post("/api/tasks/{task_id}/acknowledge")
    def acknowledge(task_id: str, request: AcknowledgeRequest):
        reviewer = request.reviewer.strip()
        if not reviewer:
            raise HTTPException(422, "Reviewer name is required")
        with store.session.begin() as session:
            begin_write(session)
            task = session.get(HumanTask, task_id)
            if task is None:
                raise HTTPException(404, "Task not found")
            assessment = session.get(Assessment, task.assessment_id)
            if session.get(ClaimArchive, assessment.claim_id):
                raise HTTPException(409, "This claim is archived")
            if session.get(InvestigationRevision, assessment.id) or can_reprocess(assessment, adjudicated=session.get(Adjudication, assessment.id) is not None):
                raise HTTPException(409, "Use the latest completed investigation before acknowledging a routing task.")
            if task.status != "pending":
                raise HTTPException(409, "Task is already acknowledged")
            task.status, task.reviewer, task.acknowledged_at = "acknowledged", reviewer, now()
            session.add(HumanAction(task_id=task.id, reviewer=reviewer, action="acknowledged"))
            store.add_event(session, task.assessment_id, "human_acknowledged", {"queue": task.queue})
        return {"status": "acknowledged", "identity_mode": "local demonstration name; not authenticated production identity"}

    @app.get("/api/evaluation")
    def evaluation(batch_id: str | None = None):
        return evaluate(store, labels_path, batch_id)

    @app.get("/api/claims/{assessment_id}/specialist-inputs")
    def investigation_inputs(assessment_id: str):
        with store.session() as session:
            if session.get(Assessment, assessment_id) is None:
                raise HTTPException(404, "Claim not found")
            stages = specialist_inputs(session, assessment_id)
        return JSONResponse({"stages": stages}, headers={"Cache-Control": "no-store"})

    @app.get("/api/claims/{assessment_id}/audit")
    def audit(assessment_id: str):
        try:
            package = export_audit(store, assessment_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return JSONResponse(package, headers={"Content-Disposition": f'attachment; filename="claim-{assessment_id}-audit.json"'})

    @app.get("/api/activity")
    def activity():
        with store.session() as session:
            events = session.scalars(select(WorkflowEvent).order_by(WorkflowEvent.id.desc()).limit(100)).all()
            return {"events": [{"id": e.id, "assessment_id": e.assessment_id, "kind": e.kind, "payload": e.payload, "created_at": e.created_at} for e in events]}

    @app.get("/api/events")
    async def stream(after: int = Query(0, ge=0)):
        async def events():
            cursor = after
            while True:
                with store.session() as session:
                    rows = session.scalars(select(WorkflowEvent).where(WorkflowEvent.id > cursor).order_by(WorkflowEvent.id).limit(100)).all()
                for row in rows:
                    cursor = row.id
                    yield f"id: {row.id}\ndata: {json.dumps({'id': row.id, 'kind': row.kind, 'assessment_id': row.assessment_id, 'payload': row.payload})}\n\n"
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
