"""Retrieve distinct earlier claims that are eligible for comparison."""
from decimal import Decimal

from sqlalchemy import and_, select

from backend.agents.intake_specialist import cost_evidence
from backend.storage.sqlite_store import (Adjudication, Assessment, Attachment, AttachmentExtraction, Claim, DecisionRecord,
                                         FraudRoutingCorrection, InvestigationRevision, digest)
from backend.evidence.uploaded_claim_documents import cost_documents_from_extractions
from backend.evidence.synthetic_claim_text import document_text_refs, resolve_claim_text


NO_HISTORY = "No eligible earlier claims were available when this claim was submitted"


class ClaimHistory:
    def records(self, session, claim, *, source=None):
        loss_date = claim.payload.get("loss_date")
        if not loss_date:
            return []
        query = (
            select(Claim, Assessment.execution, DecisionRecord, FraudRoutingCorrection, Adjudication)
            .join(Assessment, Assessment.claim_id == Claim.id)
            .join(DecisionRecord, DecisionRecord.assessment_id == Assessment.id)
            .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id)
            .outerjoin(FraudRoutingCorrection, and_(FraudRoutingCorrection.assessment_id == Assessment.id,
                                                     FraudRoutingCorrection.created_at < claim.received_at))
            .outerjoin(Adjudication, and_(Adjudication.assessment_id == Assessment.id, Adjudication.created_at < claim.received_at))
            .where(Claim.id != claim.id, Claim.received_at < claim.received_at, Assessment.status == "complete",
                   DecisionRecord.created_at < claim.received_at,
                   InvestigationRevision.source_id.is_(None))
            .order_by(DecisionRecord.created_at, Claim.id)
        )
        if source:
            query = query.where(Claim.payload["source"].as_string() == source)
        rows = session.execute(query).all()
        records = []
        seen_keys = {self.incident_key(claim.payload)}
        seen_incidents = {claim.payload.get("incident_id")}
        for prior, execution, decision, correction, review in rows:
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
            if digest(decision.payload) != decision.payload_hash:
                raise ValueError("Historical decision integrity check failed")
            if correction and (correction.original_decision_hash != decision.payload_hash or digest(correction.payload) != correction.payload_hash):
                raise ValueError("Historical routing correction integrity check failed")
            published = correction or decision
            outcome = published.payload
            document_sources = {}
            if item["source"] == "Manual submission":
                extractions = session.scalars(select(AttachmentExtraction).join(Attachment).where(
                    Attachment.assessment_id == decision.assessment_id)).all()
                extracted = cost_documents_from_extractions([row.payload for row in extractions])
                documents, _, _ = cost_evidence({**item, "documents": extracted})
                references = {doc["document_reference"] for doc in documents if doc.get("document_reference")}
            else:
                item = resolve_claim_text(item)
                documents, _, _ = cost_evidence(item)
                references = {doc.get("document_reference") or doc["id"] for doc in documents}
                text_refs = document_text_refs(item)
                document_sources = {doc["id"]: text_refs[path] for index, doc in enumerate(item.get("documents") or [])
                                    if doc in documents and (path := f"submission:/documents/{index}") in text_refs}
            seen_keys.add(key)
            seen_incidents.add(incident)
            references = sorted(references)
            records.append({"id": prior.id, "incident_id": incident, "loss_type": item["loss_type"],
                "claim_reference": f"CLM-{execution.get('original_sequence', prior.sequence):04d}", "loss_date": item["loss_date"],
                "received_at": prior.received_at, "decision_published_at": decision.created_at,
                "triage_decision": {"route": outcome["route"], "reason_codes": outcome.get("reason_codes", []),
                                    "reason": outcome.get("justification", "").partition("\n\n")[0],
                                    "payable_amount": outcome.get("payable_amount"), "recorded_at": published.created_at},
                "staff_decision": {"outcome": review.outcome, "reason": review.reason, "reviewed_at": review.created_at} if review else None,
                "gross_cost": item["gross_amount"], "currency": item["currency"], "source": item["source"],
                "document_reference": references[0] if references else None,
                "document_references": references, "assessment_id": decision.assessment_id,
                "narrative": item.get("narrative"), "documents": documents,
                "additional_costs": item.get("additional_costs", []),
                "cost_basis": "submitted gross amount",
                **({"cost_document_text_refs": document_sources} if document_sources else {})})
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
