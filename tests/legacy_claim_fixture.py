"""Seed cases that entered the workspace before manual submissions were retired."""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from backend.api.portal_routes import password_hash
from backend.assurance.execution_manifest import execution_manifest
from backend.evidence.policy_wording_index import uploaded_document_ids
from backend.storage.sqlite_store import Account, Assessment, Attachment, Claim, LoginSession, Ownership, digest
from backend.workflow.versions import V3


def seed_legacy_claim(store, body, *, mode="deterministic-tools"):
    identifier, claim_id, owner_id = (str(uuid.uuid4()) for _ in range(3))
    files = [{"id": str(uuid.uuid4()), **item} for item in body.get("attachments", ())]
    corpus = uploaded_document_ids(store)
    payload = {**{key: value for key, value in body.items() if key != "attachments"},
               "claim_id": claim_id, "incident_id": str(uuid.uuid4()), "revision": 1,
               "currency": "LKR", "source": "Manual submission", "documents": [],
               "reported_date": body["loss_date"], "additional_costs": body.get("additional_costs", []),
               "attachment_manifest": [{key: value for key, value in item.items() if key != "content"} for item in files]}
    old_session = str(uuid.uuid4())
    with store.session.begin() as session:
        session.add(Account(id=owner_id, email=f"{owner_id}@legacy.test", password_hash=password_hash("LegacyCustomer!2026"), role="user"))
        sequence = (session.scalar(select(func.max(Claim.sequence))) or 0) + 1
        session.add(Claim(id=claim_id, sequence=sequence, batch_id="manual", policy_id=body["policy_id"],
                          payload=payload, payload_hash=digest(payload)))
        session.flush()
        session.add(Assessment(id=identifier, claim_id=claim_id, context={"findings": {}, "skipped": {},
                    "environment": "manual", "version": 0, "workflow": V3, "policy_search_mode": "uploaded-corpus",
                    "policy_corpus_ids": corpus}, execution={**execution_manifest(), "workflow": V3, "mode": mode}))
        session.add(LoginSession(account_id=owner_id, token_hash=hashlib.sha256(old_session.encode()).hexdigest(),
                                 expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()))
        session.flush()
        session.add(Ownership(assessment_id=identifier, account_id=owner_id))
        session.add_all(Attachment(assessment_id=identifier, **item) for item in files)
    return identifier, old_session, f"{owner_id}@legacy.test"
