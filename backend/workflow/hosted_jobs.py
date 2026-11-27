"""Deliver persistent claim and policy jobs through Vercel Queues."""
import os
from pathlib import Path
from tempfile import gettempdir

from sqlalchemy import select
from vercel.queue import RetryAfter, subscribe
from vercel.queue.sync import QueueClient

from backend.storage.postgres import worker_lease
from backend.storage.sqlite_store import Assessment, ClaimArchive, PolicyDocument, PolicyDocumentRemoval, Store

TOPIC = "ruhunu-claims-jobs"


class QueueWake:
    def __init__(self, store):
        self.store = store
        self.client = QueueClient(region="iad1")

    def set(self):
        with self.store.session() as session:
            claims = session.scalars(select(Assessment.id).where(
                Assessment.status.in_(["queued", "processing"]),
                Assessment.claim_id.not_in(select(ClaimArchive.claim_id)),
            )).all()
            documents = session.scalars(select(PolicyDocument.id).where(
                PolicyDocument.status.in_(["queued", "indexing"]),
                PolicyDocument.id.not_in(select(PolicyDocumentRemoval.document_id)),
            )).all()
        for identifier in documents:
            self.client.send(TOPIC, {"kind": "policy", "id": identifier})
        for identifier in claims:
            self.client.send(TOPIC, {"kind": "claim", "id": identifier}, idempotency_key=f"claim:{identifier}")


@subscribe(topic=TOPIC, max_concurrency=1, retry_after=30, max_attempts=5)
def process_job(job: dict[str, str]):
    database_url = os.environ["SUPABASE_DATABASE_URL"]
    data_dir = Path(os.getenv("CLAIMS_DATA_DIR", str(Path(gettempdir()) / "ruhunu-claims")))
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CLAIMS_MODEL_DIR", str(data_dir / "models"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(data_dir / "models" / "tiktoken"))
    from backend.assurance.investigation_tracing import flush_langfuse
    from backend.evidence.policy_wording_index import index_document
    from backend.workflow.claims_workflow import ClaimsWorkflow
    from backend.workflow.investigation_worker import process_assessment

    store = Store(data_dir / "claims.sqlite3", database_url=database_url)
    try:
        with worker_lease(store) as acquired:
            if not acquired:
                raise RetryAfter(10)
            if job["kind"] == "policy":
                with store.session() as session:
                    document = session.get(PolicyDocument, job["id"])
                    if document is None or document.status not in {"queued", "indexing"} or session.get(PolicyDocumentRemoval, job["id"]):
                        return
                index_document(store, job["id"])
            elif job["kind"] == "claim":
                with store.session() as session:
                    assessment = session.get(Assessment, job["id"])
                    if assessment is None or assessment.status not in {"queued", "processing"} or session.get(ClaimArchive, assessment.claim_id):
                        return
                workflow = ClaimsWorkflow(store, data_dir / "checkpoints.sqlite3")
                try:
                    process_assessment(store, workflow, job["id"])
                finally:
                    workflow.close()
            else:
                raise ValueError("Unknown hosted job kind")
    finally:
        store.engine.dispose()
        flush_langfuse()
