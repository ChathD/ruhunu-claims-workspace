"""Retrieve prior synthetic cost documents and check for reused references."""
from sqlalchemy import select

from backend.agents.intake_specialist import cost_evidence
from backend.storage.sqlite_store import Claim, digest


def prior_cost_documents(session, loss_types, registration, *, before_sequence=None):
    query = select(Claim).where(
        Claim.payload["source"].as_string() == "Synthetic generator",
        Claim.payload["loss_type"].as_string().in_(loss_types),
        Claim.payload["registration"].as_string() != registration,
    ).order_by(Claim.sequence.desc())
    if before_sequence is not None:
        query = query.where(Claim.sequence < before_sequence)
    for row in session.scalars(query):
        if digest(row.payload) != row.payload_hash:
            raise ValueError("Prior synthetic submission integrity check failed")
        documents, _, _ = cost_evidence(row.payload)
        for document in documents:
            if (document.get("id") and not document.get("document_reference")
                    and document.get("registration") == row.payload["registration"]
                    and document.get("line_items") and document.get("total") is not None):
                yield row, document


def verify_reuse(session, claim):
    documents, _, _ = cost_evidence(claim.payload)
    referenced = {doc["document_reference"]: doc for doc in documents if doc.get("document_reference")}
    if not referenced:
        return []
    verified = []
    for prior, document in prior_cost_documents(session, [claim.payload["loss_type"]],
                                                claim.payload["registration"], before_sequence=claim.sequence):
        candidate = referenced.get(document["id"])
        if not candidate or not prior.payload.get("incident_id") or prior.payload["incident_id"] == claim.payload.get("incident_id"):
            continue
        expected = {**document, "id": candidate["id"], "document_reference": document["id"]}
        if candidate == expected:
            verified.append({"document_reference": document["id"], "current_document_id": candidate["id"],
                             "prior_claim_id": prior.id, "prior_incident_id": prior.payload["incident_id"],
                             "prior_registration": prior.payload["registration"],
                             "current_registration": claim.payload["registration"], "prior_document": document})
    return verified
