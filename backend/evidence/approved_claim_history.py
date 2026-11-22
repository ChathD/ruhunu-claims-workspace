"""Retrieve distinct, staff-approved earlier claims that are eligible for comparison."""
from decimal import Decimal

from sqlalchemy import select

from backend.agents.intake_specialist import cost_evidence
from backend.storage.sqlite_store import Adjudication, Assessment, Attachment, AttachmentExtraction, Claim, DecisionRecord, InvestigationRevision, digest
from backend.evidence.uploaded_claim_documents import cost_documents_from_extractions


NO_HISTORY = "No comparable staff-approved claims were available when this claim was submitted"


class ApprovedClaimHistory:
    def records(self, session, claim, *, source=None):
        loss_date = claim.payload.get("loss_date")
        if not loss_date:
            return []
        query = (
            select(Claim, Adjudication)
            .join(Assessment, Assessment.claim_id == Claim.id)
            .join(Adjudication, Adjudication.assessment_id == Assessment.id)
            .join(DecisionRecord, DecisionRecord.assessment_id == Assessment.id)
            .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id)
            .where(Claim.id != claim.id, Claim.received_at < claim.received_at, Assessment.status == "complete",
                   Adjudication.outcome == "approved", Adjudication.created_at < claim.received_at,
                   InvestigationRevision.source_id.is_(None))
            .order_by(Adjudication.created_at, Claim.id)
        )
        if source:
            query = query.where(Claim.payload["source"].as_string() == source)
        rows = session.execute(query).all()
        records = []
        seen_keys = {self.incident_key(claim.payload)}
        seen_incidents = {claim.payload.get("incident_id")}
        for prior, decision in rows:
            item = prior.payload
            if (item.get("source") not in {"Manual submission", "Synthetic generator"} or item["loss_type"] != claim.payload["loss_type"]
                    or item["currency"] != claim.payload["currency"] or Decimal(item["gross_amount"]) <= 0
                    or not item.get("loss_date")):
                continue
            key = self.incident_key(item)
            incident = item.get("incident_id")
            if not incident or key in seen_keys or incident in seen_incidents:
                continue
            if digest(item) != prior.payload_hash:
                raise ValueError("Historical submission integrity check failed")
            if item["source"] == "Manual submission":
                extractions = session.scalars(select(AttachmentExtraction).join(Attachment).where(
                    Attachment.assessment_id == decision.assessment_id)).all()
                extracted = cost_documents_from_extractions([row.payload for row in extractions])
                documents, _, _ = cost_evidence({**item, "documents": extracted})
                references = {doc["document_reference"] for doc in documents if doc.get("document_reference")}
            else:
                documents, _, _ = cost_evidence(item)
                references = {doc.get("document_reference") or doc["id"] for doc in documents}
            seen_keys.add(key)
            seen_incidents.add(incident)
            references = sorted(references)
            records.append({"id": prior.id, "incident_id": incident, "loss_type": item["loss_type"],
                "settled_on": decision.created_at[:10], "gross_cost": item["gross_amount"], "currency": item["currency"],
                "decision": "staff-approved claim", "source": item["source"], "document_reference": references[0] if references else None,
                "document_references": references, "assessment_id": decision.assessment_id,
                "narrative": item.get("narrative"), "documents": documents,
                "additional_costs": item.get("additional_costs", []),
                "approved_at": decision.created_at, "cost_basis": "submitted gross amount"})
        return records

    @staticmethod
    def incident_key(payload):
        if payload.get("person_id") and payload.get("incident_id"):
            return payload["incident_id"]
        return payload.get("policy_id"), payload.get("registration"), payload.get("loss_date")

    def exists(self, session, identifier):
        return session.get(Claim, identifier) is not None


def uses_document_reference(record, reference):
    return reference in record.get("document_references", [record.get("document_reference")])
