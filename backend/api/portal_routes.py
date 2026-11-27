"""Serve staff claim investigations and administrator routes."""
import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import Account, Adjudication, Assessment, Attachment, Claim, ClaimArchive, DecisionRecord, FraudRoutingCorrection, InvestigationRevision, LoginSession, ToolCall, WorkflowEvent
from backend.claim_schemas import Route
from backend.assurance.investigation_display import (
    adjuster_review_brief, completed_model_assignments, display_finding, fraud_escalation_brief, fraud_identification_brief,
    settlement_review_reasons, staff_justification,
)
from backend.assurance.investigation_pdf import render_investigation_pdf
from backend.assurance.pdf_preview import pdf_page_previews
from backend.assurance.policy_schedule_preview import policy_schedule_preview
from backend.assurance.routing_evaluation import evaluate
from backend.assurance.synthetic_dataset import generated_scenario_title
from backend.evidence.vehicle_market_value import VehicleMarketQuery, lookup_vehicle_market_value
from backend.workflow.investigation_retry import can_reprocess, failure_reason, interrupted_attention, queue_reprocessing


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f"{salt}:{value}"


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=150)
    password: str = Field(min_length=10, max_length=128)


class AccountInput(Credentials):
    role: Literal["staff"] = "staff"


def install_portal(app, store):
    router = APIRouter(prefix="/api")

    def create_account(email, password, role):
        email = email.strip().lower()
        if "@" not in email:
            raise HTTPException(422, "Enter a valid email address")
        with store.session.begin() as session:
            if session.scalar(select(Account).where(Account.email == email)):
                raise HTTPException(409, "An account already exists for this email")
            row = Account(id=str(uuid.uuid4()), email=email, password_hash=password_hash(password), role=role)
            session.add(row)
        return {"id": row.id, "email": row.email, "role": row.role}

    with store.session() as session:
        empty = session.scalar(select(func.count()).select_from(Account)) == 0
    if empty:
        # Explicit development accounts; bind this demo to loopback only.
        for role in ("admin", "staff"):
            create_account(f"{role}@ruhunu.local", f"{role.capitalize()}Demo!2026", role)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        path = request.url.path.rstrip("/")
        public = path in {"/api/health", "/api/auth/login"}
        if path.startswith("/api/") and not public:
            token = request.cookies.get("claims_session", "")
            with store.session() as session:
                login = session.get(LoginSession, hashlib.sha256(token.encode()).hexdigest()) if token else None
                actor = session.get(Account, login.account_id) if login and login.expires_at > datetime.now(timezone.utc).isoformat() else None
            if actor is None or actor.role not in {"admin", "staff"}:
                return JSONResponse({"detail": "Please sign in"}, status_code=401)
            request.state.actor = actor
            if not path.startswith(("/api/auth/", "/api/portal/")):
                policy_documents = path == "/api/policy-documents" or path.startswith("/api/policy-documents/")
                policy_retry = path.endswith("/retry") and path.rsplit("/", 2)[0] == "/api/policy-documents"
                policy_remove = request.method == "DELETE" and path.rpartition("/")[0] == "/api/policy-documents"
                policy_write = request.method == "POST" and (path == "/api/policy-documents" or policy_retry)
                if policy_write or policy_remove or (policy_documents and request.method in {"GET", "HEAD"}):
                    roles = {"admin", "staff"}
                elif policy_documents or path.startswith("/api/policies/"):
                    roles = {"admin"}
                else:
                    roles = {"staff"}
                if actor.role not in roles:
                    detail = "Administrator access required" if roles == {"admin"} else "Staff access required"
                    return JSONResponse({"detail": detail}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin not in {"http://127.0.0.1:5180", "http://localhost:5180", "http://127.0.0.1:8010"}:
                return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        return await call_next(request)

    @router.post("/auth/login")
    def login(body: Credentials, response: Response):
        with store.session.begin() as session:
            account = session.scalar(select(Account).where(Account.email == body.email.strip().lower()))
            if not account or account.role not in {"admin", "staff"} or not hmac.compare_digest(password_hash(body.password, account.password_hash.split(":")[0]), account.password_hash):
                raise HTTPException(401, "Email or password is incorrect")
            token = secrets.token_urlsafe(32)
            session.add(LoginSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), account_id=account.id,
                                     expires_at=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()))
            response.set_cookie("claims_session", token, httponly=True, samesite="strict", max_age=28800)
            return {"id": account.id, "email": account.email, "role": account.role}

    @router.get("/auth/me")
    def me(request: Request):
        actor = request.state.actor
        return {"id": actor.id, "email": actor.email, "role": actor.role}

    @router.post("/auth/logout")
    def logout(request: Request, response: Response):
        with store.session.begin() as session:
            row = session.get(LoginSession, hashlib.sha256(request.cookies.get("claims_session", "").encode()).hexdigest())
            if row:
                session.delete(row)
        response.delete_cookie("claims_session")
        return {"status": "signed out"}

    def require_staff(request):
        if request.state.actor.role != "staff":
            raise HTTPException(403, "Staff access required")

    def require_admin(request):
        if request.state.actor.role != "admin":
            raise HTTPException(403, "Administrator access required")

    @router.get("/portal/accounts")
    def accounts(request: Request):
        require_admin(request)
        with store.session() as session:
            return [{"id": a.id, "email": a.email, "role": a.role} for a in session.scalars(select(Account).where(Account.role.in_(("admin", "staff"))))]

    @router.post("/portal/accounts", status_code=201)
    def add_account(body: AccountInput, request: Request):
        require_admin(request)
        return create_account(body.email, body.password, "staff")

    @router.get("/portal/evaluation")
    def routing_evaluation(request: Request, response: Response):
        require_admin(request)
        report = evaluate(store, app.state.labels_path)
        response.headers["Cache-Control"] = "no-store"
        return {"matrix": report["matrix"], "class_metrics": report["class_metrics"],
                "evaluated": report["evaluated"], "pending": report["pending"], "mismatches": len(report["errors"])}

    @router.post("/portal/claims/{identifier}/retry", status_code=202)
    def retry_investigation(identifier: str, request: Request):
        require_staff(request)
        with store.session.begin() as session:
            begin_write(session)
            assessment = session.get(Assessment, identifier)
            if assessment is None:
                raise HTTPException(404, "Assessment not found")
            try:
                target = queue_reprocessing(store, session, assessment)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
        app.state.worker.wake.set()
        return {"id": target, "status": "queued"}

    def portal_claim_rows(request, identifier=None, summary=False, overview=False):
        require_staff(request)
        with store.session() as session:
            rows = []
            query = (select(Assessment, Claim, FraudRoutingCorrection, DecisionRecord, Adjudication, InvestigationRevision)
                     .join(Claim, Claim.id == Assessment.claim_id)
                     .outerjoin(FraudRoutingCorrection, FraudRoutingCorrection.assessment_id == Assessment.id)
                     .outerjoin(DecisionRecord, DecisionRecord.assessment_id == Assessment.id)
                     .outerjoin(Adjudication, Adjudication.assessment_id == Assessment.id)
                     .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id))
            if identifier is None:
                query = query.where(Assessment.claim_id.not_in(select(ClaimArchive.claim_id)),
                                    InvestigationRevision.source_id.is_(None)).order_by(Claim.sequence)
            else:
                query = query.where(Assessment.id == identifier)
            for assessment, claim, correction, decision, review, replacement in session.execute(query):
                decision = correction or decision
                retryable = can_reprocess(assessment, adjudicated=review is not None, claim=claim)
                if retryable:
                    decision = None
                route = decision.payload["route"] if decision else None
                information_requested = route == Route.INFO.value
                status = "investigation failed" if retryable or assessment.status == "failed" else "processing"
                if assessment.status == "awaiting_policy":
                    status = "awaiting processing"
                elif route:
                    status = route
                reference = assessment.execution.get("original_sequence", claim.sequence)
                row = {"id": assessment.id, "reference": f"CLM-{reference:04d}", "source": claim.payload.get("source", "Synthetic"),
                       "received_at": claim.received_at, "status": status}
                if summary:
                    rows.append({**row, "submission": {key: claim.payload.get(key) for key in ("loss_type", "gross_amount")}})
                    continue
                files = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment.id)).all()
                archived = identifier is not None and session.get(ClaimArchive, claim.id) is not None
                selected_policy = assessment.context.get("selected_policy")
                schedule = assessment.context.get("policy_schedule")
                interrupted = retryable or assessment.status == "failed"
                detail = {**row, "submission": claim.payload, "selected_policy": selected_policy, "archived": archived,
                          "can_reprocess": retryable and not replacement and not archived,
                          "attachments": [{"id": f.id, "name": f.name} for f in files]}
                if overview:
                    scenario = None
                    if assessment.execution.get("generation_scenario"):
                        original_id = assessment.id
                        if reference != claim.sequence:
                            original_id = session.scalar(select(Assessment.id).join(Claim).where(Claim.sequence == reference))
                        scenario = generated_scenario_title(app.state.labels_path, original_id, store=store)
                    rows.append({**detail, "has_policy_schedule": bool(schedule), "report_available": bool(decision or interrupted),
                                 "replacement_id": replacement.target_id if replacement else None, "scenario_title": scenario})
                    continue
                calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment.id).order_by(ToolCall.id)).all()
                context = assessment.context
                if correction:
                    event = session.scalar(select(WorkflowEvent).where(WorkflowEvent.assessment_id == assessment.id,
                        WorkflowEvent.kind == "evidence_routing_corrected"))
                    if event and (handoff := event.payload.get("third_party_assessment")):
                        coverage = context["findings"]["coverage"]
                        coverage = {**coverage, "details": {**coverage["details"], "third_party_assessment": handoff}}
                        context = {**context, "findings": {**context["findings"], "coverage": coverage}}
                findings = {stage: display_finding(finding, claim.payload, selected_policy)
                            for stage, finding in context.get("findings", {}).items()}
                triage = ({**decision.payload, "justification": staff_justification(decision.payload, context, claim.payload)}
                          if decision else None)
                fraud_reason = (fraud_escalation_brief(context.get("findings", {}), claim.payload) if route == Route.FRAUD
                                else fraud_identification_brief(findings) if route == Route.REVIEW else None)
                rows.append({**detail, "policy_schedule": schedule,
                             "processing_error": (assessment.error or failure_reason(assessment.context)) if interrupted else None,
                             "processing_alerts": interrupted_attention(assessment.context, claim.payload) if interrupted else [],
                             "information_required": decision.payload.get("information_required", []) if information_requested else [],
                             "triage": triage,
                             "findings": findings,
                             "skipped_specialists": context.get("skipped", {}),
                             "adjuster_review": adjuster_review_brief(claim.payload, context, decision=triage),
                             "fraud_identification": fraud_reason,
                             "settlement_review_reasons": settlement_review_reasons(context, triage, claim=claim.payload),
                             "investigation": {"tasks": completed_model_assignments(session, assessment),
                                               "readiness": assessment.context.get("readiness_history", {}), "plans": assessment.context.get("investigation_plans", {}), "extraction": assessment.context.get("attachment_extraction", []),
                                               "workflow": assessment.execution.get("workflow"), "typed_plans": assessment.context.get("plans_v4", {}),
                                               "verification": assessment.context.get("verification_rounds", {}),
                                               "tools": [{"id": c.id, "stage": c.stage, "tool": c.tool, "request": c.request, "response": c.response} for c in calls],
                                               "replacement_id": replacement.target_id if replacement else None}})
            return list(reversed(rows))

    @router.get("/portal/claims")
    def list_claims(request: Request, summary: bool = False):
        return portal_claim_rows(request, summary=summary)

    @router.get("/portal/claims/{identifier}")
    def claim_detail(identifier: str, request: Request, overview: bool = False):
        rows = portal_claim_rows(request, identifier=identifier, overview=overview)
        if not rows:
            raise HTTPException(404, "Claim not found")
        return rows[0]

    @router.get("/portal/claims/{identifier}/investigation.pdf")
    def investigation_pdf(identifier: str, request: Request, preview: bool = False):
        claim = claim_detail(identifier, request)
        if not claim["triage"] and claim["status"] != "investigation failed":
            raise HTTPException(409, "No completed or interrupted investigation is available")
        policy = claim["selected_policy"]
        wording_url = str(request.url_for("preview", identifier=policy["id"])) if policy else None
        report = render_investigation_pdf(claim, wording_url)
        if preview:
            return JSONResponse({"pages": pdf_page_previews(report)}, headers={"Cache-Control": "no-store"})
        filename = f"{claim['reference']}-investigation.pdf"
        return Response(report, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})

    @router.get("/portal/claims/{identifier}/policy-schedule")
    def claim_policy_schedule(identifier: str, request: Request, preview: bool = False):
        require_staff(request)
        with store.session() as session:
            assessment = session.get(Assessment, identifier)
            schedule = assessment.context.get("policy_schedule") if assessment else None
        if not schedule:
            raise HTTPException(404, "Policy schedule not found")
        return JSONResponse(policy_schedule_preview(schedule) if preview else schedule, headers={"Cache-Control": "no-store"})

    @router.get("/portal/vehicle-market-value")
    def vehicle_market_value(query: Annotated[VehicleMarketQuery, Query()], request: Request):
        require_staff(request)
        return lookup_vehicle_market_value(query)

    @router.get("/portal/attachments/{attachment_id}")
    def download(attachment_id: str, request: Request):
        require_staff(request)
        with store.session() as session:
            attachment = session.get(Attachment, attachment_id)
            if not attachment:
                raise HTTPException(404, "Attachment not found")
            return Response(base64.b64decode(attachment.content), media_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="claim-document"', "X-Content-Type-Options": "nosniff"})

    app.include_router(router)
