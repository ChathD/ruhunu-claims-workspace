"""Retrieve prior synthetic cost documents and check for reused references."""
from sqlalchemy import select

from backend.agents.intake_specialist import cost_evidence
from backend.evidence.synthetic_claim_text import resolve_claim_text
from backend.storage.sqlite_store import Claim, digest


def prior_cost_documents(session, loss_types, *, before_sequence=None):
    query = select(Claim).where(
        Claim.payload["source"].as_string() == "Synthetic generator",
        Claim.payload["loss_type"].as_string().in_(loss_types),
    ).order_by(Claim.sequence.desc())
    if before_sequence is not None:
        query = query.where(Claim.sequence < before_sequence)
    for row in session.scalars(query):
        if digest(row.payload) != row.payload_hash:
            raise ValueError("Prior synthetic submission integrity check failed")
        documents, _, _ = cost_evidence(resolve_claim_text(row.payload))
        for document in documents:
            if (document.get("id") and not document.get("document_reference")
                    and document.get("registration") == row.payload["registration"]
                    and document.get("line_items") and document.get("total") is not None):
                yield row, document


def verify_reuse(session, claim):
    documents, _, _ = cost_evidence(resolve_claim_text(claim.payload))
    referenced = {doc["document_reference"]: doc for doc in documents if doc.get("document_reference")}
    if not referenced:
        return []
    verified = []
    for prior, document in prior_cost_documents(session, [claim.payload["loss_type"]], before_sequence=claim.sequence):
        candidate = referenced.get(document["id"])
        if not candidate or not prior.payload.get("incident_id") or prior.payload["incident_id"] == claim.payload.get("incident_id"):
            continue
        same_vehicle = prior.payload["registration"] == claim.payload["registration"]
        if same_vehicle and (prior.policy_id != claim.policy_id or not prior.payload.get("loss_date") or not claim.payload.get("loss_date")
                             or not document.get("date") or max(prior.payload["loss_date"], document["date"]) >= claim.payload["loss_date"]):
            continue
        expected = {**document, "id": candidate["id"], "document_reference": document["id"]}
        if candidate == expected:
            verified.append({"document_reference": document["id"], "current_document_id": candidate["id"],
                             "prior_claim_id": prior.id, "prior_incident_id": prior.payload["incident_id"],
                             "prior_registration": prior.payload["registration"],
                             "current_registration": claim.payload["registration"], "prior_document": document})
    return verified
