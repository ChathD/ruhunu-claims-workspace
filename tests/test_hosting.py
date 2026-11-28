"""Exercise hosted login boundaries and dispatch against a real workspace store."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.storage.sqlite_store import Account, Assessment, Claim, ClaimArchive, InvestigationRevision, PolicyDocument, PolicyDocumentRemoval, Store
from backend.workflow.hosted_jobs import QueueWake


class HostedHttpTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {
            "VERCEL": "1", "VERCEL_URL": "claims-preview.vercel.app",
            "VERCEL_PROJECT_PRODUCTION_URL": "claims.vercel.app", "VERCEL_GIT_COMMIT_SHA": "release-under-test",
        })
        self.environment.start()
        self.app = create_app(self.directory.name)
        self.client = TestClient(self.app, base_url="https://claims.vercel.app")
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.environment.stop()
        self.directory.cleanup()

    def test_hosted_login_accepts_its_production_origin_and_uses_secure_cookie(self):
        response = self.client.post("/api/auth/login", headers={"Origin": "https://claims.vercel.app"},
                                    json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Secure", response.headers["set-cookie"])
        self.assertEqual(self.client.get("/api/auth/me").json()["role"], "staff")
        self.assertIsNone(self.app.state.worker.thread)
        self.assertEqual(self.client.get("/api/health").json()["release"], "release-under-test")

    def test_hosted_login_rejects_an_unrelated_origin(self):
        response = self.client.post("/api/auth/login", headers={"Origin": "https://claims.vercel.app.attacker.test"},
                                    json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"})
        self.assertEqual(response.status_code, 403)


class QueueDispatchTests(unittest.TestCase):
    def test_dispatch_excludes_archived_claims_removed_policies_and_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "claims.sqlite3")
            try:
                with store.session.begin() as session:
                    session.add(Account(id="test", email="test@example.test", role="staff", password_hash="unused"))
                    for sequence, status in enumerate(("queued", "processing", "complete"), 1):
                        session.add(Claim(id=f"c{sequence}", sequence=sequence, batch_id="test", policy_id="test", payload={}, payload_hash="test"))
                    session.flush()
                    for sequence, status in enumerate(("queued", "processing", "complete"), 1):
                        session.add(Assessment(id=f"a{sequence}", claim_id=f"c{sequence}", status=status, execution={}))
                    session.add(ClaimArchive(claim_id="c2"))
                    for identifier in ("queued-policy", "removed-policy"):
                        session.add(PolicyDocument(id=identifier, title=identifier, product="motor", version="1",
                                                   filename="wording.pdf", media_type="application/pdf", content="", file_hash=identifier))
                    session.flush()
                    session.add(PolicyDocumentRemoval(document_id="removed-policy", removed_by="test"))
                with patch("backend.workflow.hosted_jobs.QueueClient") as publisher:
                    QueueWake(store).set()
                payloads = [call.args[1] for call in publisher.return_value.send.call_args_list]
                self.assertCountEqual(payloads, [{"kind": "claim", "id": "a1"}, {"kind": "policy", "id": "queued-policy"}])
                first_key = next(call.kwargs["idempotency_key"] for call in publisher.return_value.send.call_args_list
                                 if call.args[1]["kind"] == "claim")
                with store.session.begin() as session:
                    session.get(Assessment, "a1").version += 1
                with patch("backend.workflow.hosted_jobs.QueueClient") as resumed:
                    QueueWake(store).set()
                resumed_key = next(call.kwargs["idempotency_key"] for call in resumed.return_value.send.call_args_list
                                   if call.args[1]["kind"] == "claim")
                self.assertNotEqual(first_key, resumed_key)
                with store.session.begin() as session:
                    session.add(InvestigationRevision(source_id="a1", target_id="a3"))
                with patch("backend.workflow.hosted_jobs.QueueClient") as replaced:
                    QueueWake(store).set()
                self.assertEqual([call.args[1] for call in replaced.return_value.send.call_args_list],
                                 [{"kind": "policy", "id": "queued-policy"}])
            finally:
                store.engine.dispose()


if __name__ == "__main__":
    unittest.main()
