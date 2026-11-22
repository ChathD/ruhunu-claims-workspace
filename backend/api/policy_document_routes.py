"""Serve administrator routes for uploading, inspecting and binding versioned policy wording."""
import base64

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.storage.sqlite_store import Policy, PolicyBinding, PolicyDocument, WorkspaceSetting
from backend.evidence.policy_wording_index import create_document


class DocumentUpload(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    product: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=150)
    media_type: str
    content: str = Field(max_length=14_000_000)


class BindingRequest(BaseModel):
    document_id: str


def install_policy_api(app, store):
    router = APIRouter(prefix="/api")

    @router.get("/policy-documents")
    def listing():
        with store.session() as session:
            docs = session.scalars(select(PolicyDocument).order_by(PolicyDocument.created_at.desc())).all()
            default = session.get(WorkspaceSetting, "generator_wording")
            bindings = {row.policy_id: row.document_id for row in session.scalars(select(PolicyBinding))}
            policies = session.scalars(select(Policy)).all()
            return {"documents": [{"id": d.id, "title": d.title, "product": d.product, "version": d.version,
                "status": d.status, "error": d.error, "filename": d.filename, "file_hash": d.file_hash,
                "embedding": d.embedding} for d in docs], "generator_default": default.value if default else None,
                "policies": [{"id": p.id, "version": p.payload.get("version"), "document_id": bindings.get(p.id)} for p in policies]}

    @router.post("/policy-documents", status_code=202)
    def upload(body: DocumentUpload):
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
                    "status": doc.status, "error": doc.error, "file_hash": doc.file_hash, "embedding": doc.embedding}

    @router.get("/policy-documents/{identifier}/original")
    def original(identifier: str):
        with store.session() as session:
            doc = session.get(PolicyDocument, identifier)
            if not doc:
                raise HTTPException(404, "Document not found")
            return Response(base64.b64decode(doc.content), media_type=doc.media_type,
                            headers={"Content-Disposition": 'attachment; filename="policy-wording"', "X-Content-Type-Options": "nosniff"})

    @router.get("/policy-documents/{identifier}/preview")
    def preview(identifier: str):
        with store.session() as session:
            doc = session.get(PolicyDocument, identifier)
            if not doc:
                raise HTTPException(404, "Document not found")
            return Response(base64.b64decode(doc.content), media_type=doc.media_type,
                            headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff",
                                     "Cache-Control": "no-store", "Content-Security-Policy": "frame-ancestors 'self'"})

    @router.post("/policy-documents/{identifier}/retry", status_code=202)
    def retry(identifier: str):
        with store.session.begin() as session:
            doc = session.get(PolicyDocument, identifier)
            if not doc:
                raise HTTPException(404, "Document not found")
            if doc.status != "failed":
                raise HTTPException(409, "Only failed indexing can be retried")
            doc.status, doc.error = "queued", None
        if hasattr(app.state, "worker"):
            app.state.worker.wake.set()
        return {"status": "queued"}

    @router.put("/policies/{policy_id}/wording")
    def bind(policy_id: str, body: BindingRequest):
        with store.session.begin() as session:
            if not session.get(Policy, policy_id):
                raise HTTPException(404, "Policy not found")
            doc = session.get(PolicyDocument, body.document_id)
            if not doc or doc.status != "ready":
                raise HTTPException(409, "Select an indexed document")
            session.merge(PolicyBinding(policy_id=policy_id, document_id=doc.id))
        return {"status": "bound", "note": "Applies to future assessments; existing assessment bindings are frozen"}

    @router.put("/generator/policy-wording")
    def set_default(body: BindingRequest):
        with store.session.begin() as session:
            doc = session.get(PolicyDocument, body.document_id)
            if not doc or doc.status != "ready":
                raise HTTPException(409, "Select an indexed document")
            session.merge(WorkspaceSetting(key="generator_wording", value=doc.id))
        return {"status": "configured"}

    # Existing authentication middleware reserves non-portal API endpoints for admins.
    app.include_router(router)
