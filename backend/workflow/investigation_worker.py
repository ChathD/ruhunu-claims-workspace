"""Process queued claim investigations and policy indexing jobs in the local background thread."""
import logging
import threading

from sqlalchemy import select

from backend.storage.sqlite_store import Assessment
from backend.workflow.investigation_retry import mark_failed

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, store, workflow):
        self.store, self.workflow = store, workflow
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="claims-worker", daemon=True)

    def start(self):
        self.thread.start()

    def loop(self):
        while not self.stop.is_set():
            from backend.evidence.policy_wording_index import index_pending
            index_pending(self.store)
            with self.store.session() as session:
                pending = session.scalars(select(Assessment.id).where(Assessment.status.in_(["queued", "processing"]))).all()
            for identifier in pending:
                if self.stop.is_set():
                    return
                try:
                    self.workflow.run(identifier)
                except Exception as error:
                    logger.exception("Workflow failed for %s", identifier)
                    with self.store.session.begin() as session:
                        row = session.get(Assessment, identifier)
                        reason = "Unsupported workflow version. Create a new Intake-first assessment." if type(error).__name__ == "UnsupportedWorkflowVersion" else type(error).__name__
                        if row.status != "complete":
                            mark_failed(self.store, session, row, reason)
            self.wake.wait(timeout=1)
            self.wake.clear()

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=30)
