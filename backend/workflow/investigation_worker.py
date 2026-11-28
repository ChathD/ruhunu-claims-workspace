"""Process queued claim investigations and policy indexing jobs in the local background thread."""
import logging
import os
import threading

from sqlalchemy import select

from backend.storage.sqlite_store import Assessment, ClaimArchive, InvestigationRevision
from backend.storage.postgres import worker_lease
from backend.workflow.investigation_retry import mark_failed

logger = logging.getLogger(__name__)


def process_assessment(store, workflow, identifier, *, step=False):
    try:
        return workflow.run(identifier, step=True) if step else workflow.run(identifier)
    except Exception as error:
        logger.exception("Workflow failed for %s", identifier)
        with store.session.begin() as session:
            row = session.get(Assessment, identifier)
            reason = "Unsupported workflow version. Create a new Intake-first assessment." if type(error).__name__ == "UnsupportedWorkflowVersion" else type(error).__name__
            if row.status != "complete":
                mark_failed(store, session, row, reason)


class Worker:
    def __init__(self, store, workflow):
        self.store, self.workflow = store, workflow
        self.stop = threading.Event()
        if os.getenv("VERCEL") == "1":
            from backend.workflow.hosted_jobs import QueueWake
            self.wake = QueueWake(store)
            self.thread = None
        else:
            self.wake = threading.Event()
            self.thread = threading.Thread(target=self.loop, name="claims-worker", daemon=True)

    def start(self):
        if self.thread is not None:
            self.thread.start()

    def loop(self):
        while not self.stop.is_set():
            with worker_lease(self.store) as acquired:
                if acquired:
                    self.run_pending()
            self.wake.wait(timeout=1)
            self.wake.clear()

    def run_pending(self):
        from backend.evidence.policy_wording_index import index_pending
        index_pending(self.store)
        with self.store.session() as session:
            pending = session.scalars(select(Assessment.id).where(
                Assessment.status.in_(["queued", "processing"]),
                Assessment.claim_id.not_in(select(ClaimArchive.claim_id)),
                Assessment.id.not_in(select(InvestigationRevision.source_id)),
            )).all()
        for identifier in pending:
            if self.stop.is_set():
                return
            with worker_lease(self.store, identifier) as acquired:
                if acquired:
                    process_assessment(self.store, self.workflow, identifier)

    def close(self):
        if self.thread is None:
            return
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=30)
