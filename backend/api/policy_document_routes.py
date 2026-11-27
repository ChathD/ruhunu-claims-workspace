"""Serve routes for uploading, inspecting and binding versioned policy wording."""
import base64
from threading import Lock

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.assurance.policy_schedule_preview import policy_schedule_preview
from backend.storage.postgres import begin_write
from backend.storage.sqlite_store import Policy, PolicyBinding, PolicyDocument, PolicyDocumentRemoval, PolicySchedule
from backend.evidence.policy_wording_index import MAX_POLICY_PDF_BYTES, create_document, current_index, document_kind
from backend.generation.claude_generation_client import GenerationError
from backend.generation.policy_schedule_generator import generate_schedules, pending_corrections, select_correction
from backend.workflow.selected_policy import selectable_policy_pdf


class DocumentUpload(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    product: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=150)
    media_type: str
    content: str = Field(max_length=4 * ((MAX_POLICY_PDF_BYTES + 2) // 3))


class BindingRequest(BaseModel):
    document_id: str


class ScheduleRequest(BaseModel):
    policy_document_id: str = Field(min_length=1, max_length=64)
    count: int = Field(default=1, ge=1, le=1)


class ScheduleCorrectionRequest(BaseModel):
    option_id: str = Field(min_length=1, max_length=16)


def install_policy_api(app, store):
    router = APIRouter(prefix="/api")
    schedule_generation_lock = Lock()

    @router.get("/policy-documents")
    def listing(selectable: bool = False, full_wording: bool = False):
        with store.session() as session:
            docs = session.scalars(select(PolicyDocument).where(
                PolicyDocument.id.not_in(select(PolicyDocumentRemoval.document_id))
            ).order_by(PolicyDocument.created_at.desc())).all()
            if selectable:
                docs = [doc for doc in docs if selectable_policy_pdf(doc)]
            if full_wording:
                docs = [doc for doc in docs if document_kind(session, doc) == "wording"]
            return {"documents": [{"id": d.id, "title": d.title, "product": d.product, "version": d.version,
                "status": d.status, "error": d.error, "filename": d.filename, "media_type": d.media_type,
                "document_kind": document_kind(session, d),
                "file_hash": d.file_hash, "embedding": d.embedding,
                "requires_reindex": d.status == "ready" and not current_index(d)} for d in docs]}

    @router.post("/policy-documents", status_code=202)
    def upload(body: DocumentUpload):
        if body.media_type != "application/pdf":
            raise HTTPException(422, "Upload policy wording as a PDF.")
        try:
            identifier = create_document(store, **body.model_dump())
        except (ValueError, UnicodeError) as error:
            raise HTTPException(422, str(error)) from error
        if hasattr(app.state, "worker"):
            app.state.worker.wake.set()
        return {"id": identifier, "status": "queued"}

    @router.get("/policy-documents/{identifier}")
    def detail(identifier: str):
        with store.session() as session:
            doc = session.get(PolicyDocument, identifier)
            if not doc:
                raise HTTPException(404, "Document not found")
            return {"id": doc.id, "title": doc.title, "product": doc.product, "version": doc.version,
                    "status": doc.status, "error": doc.error, "file_hash": doc.file_hash, "embedding": doc.embedding,
                    "requires_reindex": doc.status == "ready" and not current_index(doc)}

    def document_response(identifier, headers):
        with store.session() as session:
            doc = session.get(PolicyDocument, identifier)
            if not doc:
                raise HTTPException(404, "Document not found")
            return Response(base64.b64decode(doc.content), media_type=doc.media_type,
                            headers={"X-Content-Type-Options": "nosniff", **headers})

    @router.get("/policy-documents/{identifier}/original")
    def original(identifier: str):
        return document_response(identifier, {"Content-Disposition": 'attachment; filename="policy-wording"'})

    @router.get("/policy-documents/{identifier}/preview")
    def preview(identifier: str):
        return document_response(identifier, {
            "Content-Disposition": "inline",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "frame-ancestors 'self'",
        })

    @router.delete("/policy-documents/{identifier}", status_code=204)
    def remove(identifier: str, request: Request):
        with store.session.begin() as session:
            begin_write(session)
            if session.get(PolicyDocument, identifier) is None:
                raise HTTPException(404, "Document not found")
            if session.get(PolicyDocumentRemoval, identifier) is None:
                session.add(PolicyDocumentRemoval(document_id=identifier, removed_by=request.state.actor.id))
        return Response(status_code=204)

    @router.post("/policy-documents/{identifier}/retry", status_code=202)
    def retry(identifier: str):
        with store.session.begin() as session:
            begin_write(session)
            doc = session.get(PolicyDocument, identifier)
            if not doc or session.get(PolicyDocumentRemoval, identifier):
                raise HTTPException(404, "Document not found")
            if doc.status != "failed" and (doc.status != "ready" or current_index(doc)):
                raise HTTPException(409, "Only failed or outdated indexes can be queued again")
            doc.status, doc.error = "queued", None
        if hasattr(app.state, "worker"):
            app.state.worker.wake.set()
        return {"status": "queued"}

    @router.put("/policies/{policy_id}/wording")
    def bind(policy_id: str, body: BindingRequest):
        with store.session.begin() as session:
            begin_write(session)
            if not session.get(Policy, policy_id):
                raise HTTPException(404, "Policy not found")
            doc = session.get(PolicyDocument, body.document_id)
            if not doc or not current_index(doc) or session.get(PolicyDocumentRemoval, body.document_id):
                raise HTTPException(409, "Select an indexed document")
            session.merge(PolicyBinding(policy_id=policy_id, document_id=doc.id))
        return {"status": "bound", "note": "Applies to future assessments; existing assessment bindings are frozen"}

    @router.get("/policy-schedules")
    def schedules(request: Request):
        with store.session() as session:
            records = session.scalars(select(PolicySchedule).order_by(PolicySchedule.created_at.desc())).all()
            return {"schedules": [{"id": row.id, "policy_number": row.policy_number, "insured_name": row.payload["insured_name"],
                                   "registration": row.payload["vehicle"]["registration"], "cover_type": row.payload["cover_type"],
                                   "starts_on": row.payload["starts_on"], "ends_on": row.payload["ends_on"],
                                   "wording": row.payload["wording"]} for row in records],
                    "corrections": pending_corrections(session, request.state.actor.id)}

    @router.get("/policy-schedules/{identifier}")
    def schedule(identifier: str, preview: bool = False):
        with store.session() as session:
            record = session.get(PolicySchedule, identifier)
            if record is None:
                raise HTTPException(404, "Policy schedule not found")
            payload = record.payload
        if preview:
            return JSONResponse(policy_schedule_preview(payload), headers={"Cache-Control": "no-store"})
        return payload

    @router.post("/policy-schedules", status_code=201)
    def create_schedules(body: ScheduleRequest, request: Request):
        actor_id = request.state.actor.id
        if not schedule_generation_lock.acquire(blocking=False):
            raise HTTPException(409, "A policy schedule is already being generated. Try again when it finishes.")
        try:
            saved = generate_schedules(store, body.policy_document_id, body.count, actor_id, interactive=True)
            with store.session() as session:
                return {"schedules": saved, "corrections": pending_corrections(session, actor_id)}
        except GenerationError as error:
            raise HTTPException(422, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        finally:
            schedule_generation_lock.release()

    @router.post("/policy-schedules/corrections/{identifier}")
    def apply_correction(identifier: str, body: ScheduleCorrectionRequest, request: Request):
        try:
            return select_correction(store, identifier, body.option_id, request.state.actor.id)
        except KeyError as error:
            raise HTTPException(404, "Schedule correction or option not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    app.include_router(router)
