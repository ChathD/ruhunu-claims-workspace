"""Serve account, manual claim, claimant status and staff adjudication routes."""
import base64
import hashlib
import hmac
import secrets
import uuid
import threading
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.storage.sqlite_store import Account, Adjudication, Assessment, Attachment, Claim, DecisionRecord, FraudRoutingCorrection, InvestigationRevision, LoginSession, Ownership, Policy, ToolCall, digest
from backend.assurance.execution_manifest import execution_manifest
from backend.claim_schemas import AdditionalCost, Money, MotorLossType, Route, VehicleEvidence
from backend.assurance.investigation_display import completed_model_assignments, display_finding
from backend.workflow.investigation_retry import can_reprocess, failure_reason, interrupted_attention, queue_reprocessing


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f"{salt}:{value}"


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=150)
    password: str = Field(min_length=10, max_length=128)


class AccountInput(Credentials):
    role: Literal["user", "staff", "admin"] = "user"


class Upload(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    content: str = Field(max_length=2800000)


class ManualClaim(BaseModel):
    policy_id: str = Field(min_length=1, max_length=100)
    registration: str = Field(min_length=1, max_length=40)
    vehicle: VehicleEvidence | None = None
    loss_type: MotorLossType | Literal["glass", "injury", "other"]
    loss_date: date
    gross_amount: Money
    additional_costs: list[AdditionalCost] = Field(default_factory=list, max_length=10)
    narrative: str = Field(min_length=10, max_length=5000)
    attachments: list[Upload] = Field(default_factory=list, max_length=5)


class Review(BaseModel):
    outcome: Literal["approved", "rejected"]
    reason: str = Field(min_length=10, max_length=3000)


def install_portal(app, store):
    router = APIRouter(prefix="/api")
    revision_lock = threading.Lock()

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
        for role in ["admin", "staff", "user"]:
            create_account(f"{role}@ruhunu.local", f"{role.capitalize()}Demo!2026", role)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        from fastapi.responses import JSONResponse
        path = request.url.path
        public = path in {"/api/health", "/api/auth/login", "/api/auth/register"}
        if path.startswith("/api/") and not public:
            token = request.cookies.get("claims_session", "")
            with store.session() as session:
                login = session.get(LoginSession, hashlib.sha256(token.encode()).hexdigest()) if token else None
                actor = session.get(Account, login.account_id) if login and login.expires_at > datetime.now(timezone.utc).isoformat() else None
            if actor is None:
                return JSONResponse({"detail": "Please sign in"}, status_code=401)
            request.state.actor = actor
            if not path.startswith(("/api/auth/", "/api/portal/")) and actor.role != "admin":
                return JSONResponse({"detail": "Administrator access required"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin not in {"http://127.0.0.1:5180", "http://localhost:5180", "http://127.0.0.1:8010"}:
                return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        return await call_next(request)

    @router.post("/auth/login")
    def login(body: Credentials, response: Response):
        with store.session.begin() as session:
            account = session.scalar(select(Account).where(Account.email == body.email.strip().lower()))
            if not account or not hmac.compare_digest(password_hash(body.password, account.password_hash.split(":")[0]), account.password_hash):
                raise HTTPException(401, "Email or password is incorrect")
            token = secrets.token_urlsafe(32)
            session.add(LoginSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), account_id=account.id,
                                     expires_at=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()))
            response.set_cookie("claims_session", token, httponly=True, samesite="strict", max_age=28800)
            return {"id": account.id, "email": account.email, "role": account.role}

    @router.post("/auth/register", status_code=201)
    def register(body: Credentials):
        return create_account(body.email, body.password, "user")

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
        if request.state.actor.role not in {"staff", "admin"}:
            raise HTTPException(403, "Staff access required")

    @router.get("/portal/accounts")
    def accounts(request: Request):
        if request.state.actor.role != "admin":
            raise HTTPException(403, "Administrator access required")
        with store.session() as session:
            return [{"id": a.id, "email": a.email, "role": a.role} for a in session.scalars(select(Account))]

    @router.post("/portal/accounts", status_code=201)
    def add_account(body: AccountInput, request: Request):
        if request.state.actor.role != "admin":
            raise HTTPException(403, "Administrator access required")
        return create_account(body.email, body.password, body.role)

    def enqueue_manual(body: ManualClaim, owner_id: str, parent_id=None, revision=1, parent_claim_id=None):
        if body.loss_date > date.today() or body.gross_amount <= 0:
            raise HTTPException(422, "Use a past/present loss date and a positive claimed amount")
        attachments = []
        for upload in body.attachments:
            try:
                content = base64.b64decode(upload.content, validate=True)
            except ValueError:
                raise HTTPException(422, "Invalid attachment encoding")
            if len(content) > 2 * 1024 * 1024 or not content:
                raise HTTPException(422, "Each attachment must be nonempty and at most 2 MB")
            valid = content.startswith(b"%PDF-") if upload.media_type == "application/pdf" else content.startswith(b"\x89PNG\r\n\x1a\n") if upload.media_type == "image/png" else content.startswith(b"\xff\xd8\xff")
            if not valid:
                raise HTTPException(422, "Attachment contents do not match the selected file type")
            attachments.append(upload)
        identifier, claim_id = str(uuid.uuid4()), str(uuid.uuid4())
        attachment_ids = [str(uuid.uuid4()) for _ in attachments]
        payload = {**body.model_dump(mode="json", exclude={"attachments"}), "claim_id": claim_id, "revision": revision,
                   "currency": "LKR", "source": "Manual submission", "documents": [], "incident_id": str(uuid.uuid4()),
                   "parent_assessment_id": parent_id,
                   "parent_claim_id": parent_claim_id,
                   "attachment_manifest": [{"id": file_id, "name": upload.name, "media_type": upload.media_type} for file_id, upload in zip(attachment_ids, attachments)],
                   "reported_date": date.today().isoformat()}
        from backend.model_settings import settings
        from backend.evidence.policy_wording_index import uploaded_document_ids
        workflow = "claude-intake-first-v3"
        execution = {**execution_manifest(), "workflow": workflow, "mode": "anthropic-tools" if settings()[0] else "deterministic-tools"}
        corpus_ids = uploaded_document_ids(store)
        with store.session.begin() as session:
            sequence = (session.scalar(select(func.max(Claim.sequence))) or 0) + 1
            session.add(Claim(id=claim_id, sequence=sequence, batch_id="manual", policy_id=body.policy_id, payload=payload, payload_hash=digest(payload)))
            session.flush()
            context = {"findings": {}, "skipped": {}, "environment": "manual", "version": 0,
                       "workflow": workflow, "policy_search_mode": "uploaded-corpus", "policy_corpus_ids": corpus_ids}
            session.add(Assessment(id=identifier, claim_id=claim_id, status="queued", context=context, version=0, execution=execution))
            session.flush()
            session.add(Ownership(assessment_id=identifier, account_id=owner_id))
            for attachment_id, upload in zip(attachment_ids, attachments):
                session.add(Attachment(id=attachment_id, assessment_id=identifier, **upload.model_dump()))
            store.add_event(session, identifier, "manual_submitted", {"source": "manual"})
            if parent_id:
                session.add(InvestigationRevision(source_id=parent_id, target_id=identifier))
        if hasattr(app.state, "worker"):
            app.state.worker.wake.set()
        return {"id": identifier, "reference": f"CLM-{sequence:04d}"}

    @router.post("/portal/claims", status_code=201)
    def submit(body: ManualClaim, request: Request):
        return enqueue_manual(body, request.state.actor.id)

    @router.post("/portal/claims/{identifier}/investigate", status_code=202)
    def investigate_existing(identifier: str, request: Request):
        require_staff(request)
        with revision_lock:
            with store.session() as session:
                old = session.get(Assessment, identifier)
                if old is None:
                    raise HTTPException(404, "Assessment not found")
                prior = session.get(InvestigationRevision, identifier)
                if prior:
                    return {"id": prior.target_id}
                claim = session.get(Claim, old.claim_id)
                if claim.payload.get("source") != "Manual submission" or session.get(Adjudication, identifier):
                    raise HTTPException(409, "Only manual submissions awaiting a final staff decision can be investigated")
                if old.execution.get("workflow") == "claude-intake-first-v3" and (old.status in {"queued", "processing"} or old.context.get("findings")):
                    return {"id": identifier}
                owner = session.get(Ownership, identifier)
                files = session.scalars(select(Attachment).where(Attachment.assessment_id == identifier)).all()
                body = ManualClaim.model_validate({**claim.payload, "attachments": [
                    {"name": f.name, "media_type": f.media_type, "content": f.content} for f in files]})
            return enqueue_manual(body, owner.account_id, parent_id=identifier, revision=claim.payload.get("revision", 1) + 1, parent_claim_id=claim.id)

    @router.post("/portal/claims/{identifier}/retry", status_code=202)
    def retry_investigation(identifier: str, request: Request):
        require_staff(request)
        with store.session.begin() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            assessment = session.get(Assessment, identifier)
            if assessment is None:
                raise HTTPException(404, "Assessment not found")
            try:
                target = queue_reprocessing(store, session, assessment)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
        app.state.worker.wake.set()
        return {"id": target, "status": "queued"}

    @router.get("/portal/claims")
    def list_claims(request: Request):
        actor = request.state.actor
        with store.session() as session:
            allowed = set(session.scalars(select(Ownership.assessment_id).where(Ownership.account_id == actor.id)))
            rows = []
            for assessment in session.scalars(select(Assessment)):
                if actor.role == "user" and assessment.id not in allowed:
                    continue
                claim = session.get(Claim, assessment.claim_id)
                decision = session.get(FraudRoutingCorrection, assessment.id) or session.get(DecisionRecord, assessment.id)
                review = session.get(Adjudication, assessment.id)
                replacement = session.get(InvestigationRevision, assessment.id)
                retryable = can_reprocess(assessment, adjudicated=review is not None)
                if replacement and (retryable or actor.role == "user"):
                    continue
                if retryable:
                    decision = None
                information_requested = not review and decision and decision.payload["route"] == Route.INFO.value
                status = "investigation failed" if retryable or assessment.status == "failed" else "processing"
                if review:
                    status = review.outcome
                elif information_requested:
                    status = "more information requested"
                elif decision:
                    status = "awaiting review"
                files = session.scalars(select(Attachment).where(Attachment.assessment_id == assessment.id)).all()
                calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment.id).order_by(ToolCall.id)).all() if actor.role != "user" else []
                reference = assessment.execution.get("original_sequence", claim.sequence)
                rows.append({"id": assessment.id, "reference": f"CLM-{reference:04d}", "source": claim.payload.get("source", "Synthetic"),
                             "submission": claim.payload, "received_at": claim.received_at,
                             "status": status,
                             "can_reprocess": retryable and not replacement and actor.role != "user",
                             "processing_error": (assessment.error or failure_reason(assessment.context)) if retryable and actor.role != "user" else None,
                             "processing_alerts": interrupted_attention(assessment.context, claim.payload) if retryable and actor.role != "user" else [],
                             "reason": review.reason if review else None, "reviewed_at": review.created_at if review else None,
                             "information_required": decision.payload.get("information_required", []) if information_requested else [],
                             "triage": decision.payload if decision and actor.role != "user" else None,
                             "findings": {stage: display_finding(finding, claim.payload)
                                          for stage, finding in assessment.context.get("findings", {}).items()} if actor.role != "user" else {},
                             "investigation": {"tasks": completed_model_assignments(session, assessment),
                                               "readiness": assessment.context.get("readiness_history", {}), "plans": assessment.context.get("investigation_plans", {}), "extraction": assessment.context.get("attachment_extraction", []),
                                               "tools": [{"id": c.id, "stage": c.stage, "tool": c.tool, "request": c.request, "response": c.response} for c in calls],
                                               "replacement_id": session.get(InvestigationRevision, assessment.id).target_id if session.get(InvestigationRevision, assessment.id) else None} if actor.role != "user" else None,
                             "attachments": [{"id": f.id, "name": f.name} for f in files]})
            return list(reversed(rows))

    @router.get("/portal/attachments/{attachment_id}")
    def download(attachment_id: str, request: Request):
        with store.session() as session:
            attachment = session.get(Attachment, attachment_id)
            ownership = session.get(Ownership, attachment.assessment_id) if attachment else None
            if not attachment or request.state.actor.role == "user" and (not ownership or ownership.account_id != request.state.actor.id):
                raise HTTPException(404, "Attachment not found")
            return Response(base64.b64decode(attachment.content), media_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="claim-document"', "X-Content-Type-Options": "nosniff"})

    @router.post("/portal/claims/{identifier}/review")
    def review_claim(identifier: str, body: Review, request: Request):
        require_staff(request)
        if len(body.reason.strip()) < 10:
            raise HTTPException(422, "Provide a meaningful reason of at least 10 characters")
        with store.session.begin() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            assessment = session.get(Assessment, identifier)
            if assessment is None:
                raise HTTPException(404, "Claim not found")
            owner = session.get(Ownership, identifier)
            if owner and owner.account_id == request.state.actor.id:
                raise HTTPException(403, "You cannot adjudicate your own claim")
            if session.get(Adjudication, identifier):
                raise HTTPException(409, "A final staff decision already exists")
            if session.get(InvestigationRevision, identifier):
                raise HTTPException(409, "A newer investigation exists; review that assessment instead")
            if can_reprocess(assessment) or not session.get(DecisionRecord, identifier):
                raise HTTPException(409, "Wait for triage to finish before recording the final decision")
            session.add(Adjudication(assessment_id=identifier, reviewer_id=request.state.actor.id, outcome=body.outcome, reason=body.reason.strip()))
            from backend.storage.sqlite_store import HumanTask
            task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == identifier))
            if task:
                task.status = "reviewed"
                task.reviewer = request.state.actor.email
            store.add_event(session, identifier, "staff_decision", {"outcome": body.outcome, "reviewer_id": request.state.actor.id})
        return {"status": body.outcome}

    app.include_router(router)
