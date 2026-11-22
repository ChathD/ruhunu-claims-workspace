"""Build role-scoped LangChain tools that retrieve and audit records for one assessment."""
import copy
import uuid

from sqlalchemy import select
from langchain_core.tools import StructuredTool

from backend.assurance.investigation_tracing import trace_span
from backend.storage.sqlite_store import Assessment, Claim, InvestigationRevision, Policy, PolicyDocument, SyntheticPolicyHolder, ToolCall, digest
from backend.claim_schemas import REQUIRED_POLICY_FIELDS
from backend.evidence.required_claim_evidence import missing_supporting_documents, policy_applies_to_loss, policy_lists_loss, requires_police_report
from backend.workflow.specialist_context import specialist_context
from backend.evidence.approved_claim_history import uses_document_reference
from backend.evidence.policy_wording_index import search_document, search_uploaded_documents
from backend.evidence.synthetic_cost_history import verify_reuse


class EvidenceTool(StructuredTool):
    def invoke(self, input, config=None, **kwargs):
        kind = "retriever" if self.name in {"search_policy_documents", "retrieve_comparable_claims"} else "tool"
        with trace_span(f"tool.{self.name}", kind=kind, metadata=self.metadata, content=input) as span:
            result = super().invoke(input, config, **kwargs)
            span.output = result
            span.summary = {"result_hash": digest(result)}
            return result


def build_tools(store, assessment_id: str, stage: str, context_snapshot=None, audit_stage=None, on_result=None):
    """Keep Python's role-scoped tools; optionally observe committed retrievals."""
    def tool(function):
        return EvidenceTool.from_function(function, metadata={"stage": audit_stage or stage})

    def record(name, request, response):
        with store.session.begin() as session:
            session.add(ToolCall(assessment_id=assessment_id, stage=audit_stage or stage, tool=name,
                                 request=request, response=response, response_hash=digest(response)))
        if on_result is not None:
            on_result(name, copy.deepcopy(request), copy.deepcopy(response))
        return response

    def identity():
        with store.session() as session:
            assessment = session.get(Assessment, assessment_id)
            if assessment is None:
                raise ValueError("Assessment not found")
            claim = session.get(Claim, assessment.claim_id)
            if digest(claim.payload) != claim.payload_hash:
                raise ValueError("Original submission integrity check failed")
            return assessment, claim

    def submitted_policy_id(claim):
        identifier = claim.payload.get("policy_id")
        # Both references must agree before reading any policy record.
        return identifier if identifier and identifier == claim.policy_id else None

    def wording_scope(assessment, claim):
        context = context_snapshot if context_snapshot is not None else assessment.context
        source = claim.payload.get("source")
        if source == "Synthetic generator":
            return context.get("synthetic_wording_document_id"), []
        if source == "Manual submission":
            if context.get("policy_search_mode") == "uploaded-corpus":
                return None, context.get("policy_corpus_ids", [])
            return context.get("wording_document_id"), []
        return None, []

    @tool
    def read_context() -> dict:
        """Read only this specialist's relevant prior findings and policy references."""
        assessment, _ = identity()
        source = context_snapshot if context_snapshot is not None else assessment.context
        return record("read_context", {}, specialist_context(source, stage, assessment.version))

    @tool
    def read_submission() -> dict:
        """Retrieve the frozen form and any unverified facts extracted from uploads."""
        _, claim = identity()
        payload = claim.payload
        if payload.get("source") == "Manual submission":
            from backend.evidence.uploaded_claim_documents import effective_submission
            payload = effective_submission(store, assessment_id, payload)
        return record("read_submission", {}, payload)

    @tool
    def read_uploaded_documents() -> dict:
        """Read extracted content and gaps from this claim's PDF/image uploads. These are claimant evidence, not verified issuer records."""
        _, claim = identity()
        from backend.evidence.uploaded_claim_documents import effective_submission
        return record("read_uploaded_documents", {}, {"evidence": effective_submission(store, assessment_id, claim.payload).get("uploaded_evidence", [])})

    @tool
    def find_duplicates() -> dict:
        """Find earlier submissions matching this vehicle and incident, excluding this claim."""
        _, claim = identity()
        with store.session() as session:
            earlier = session.scalars(select(Claim).outerjoin(Assessment, Assessment.claim_id == Claim.id)
                .outerjoin(InvestigationRevision, InvestigationRevision.source_id == Assessment.id)
                .where(Claim.sequence < claim.sequence, InvestigationRevision.source_id.is_(None))).all()
        manual = claim.payload.get("source") == "Manual submission"
        matches = [{"claim_id": row.id, "reason": "Same vehicle, policy and loss date" if manual else "Same vehicle and incident identifier"}
                   for row in earlier if row.payload["registration"] == claim.payload["registration"]
                   and (row.payload["policy_id"] == claim.payload["policy_id"] and row.payload["loss_date"] == claim.payload["loss_date"]
                        and row.id != claim.payload.get("parent_claim_id") if manual else row.payload["incident_id"] == claim.payload["incident_id"])]
        return record("find_duplicates", {"criterion": "registration + policy_id + loss_date" if manual else "registration + incident_id", "before_sequence": claim.sequence}, {"matches": matches})

    @tool
    def retrieve_policy() -> dict:
        """Retrieve the saved policy terms for the policy ID on this claim form."""
        _, claim = identity()
        policy_id = submitted_policy_id(claim)
        person_id = claim.payload.get("person_id")
        with store.session() as session:
            policy = session.get(Policy, policy_id) if policy_id else None
            holder = session.get(SyntheticPolicyHolder, policy_id) if policy and person_id else None
        request = {"policy_id": claim.payload.get("policy_id")}
        result = {"policy": policy.payload if policy else None}
        if person_id:
            request["person_id"] = person_id
            result["policyholder"] = {"person_id": holder.person_id if holder else None,
                                      "matches": holder is not None and holder.person_id == person_id}
        return record("retrieve_policy", request, result)

    @tool
    def read_coverage_requirements() -> dict:
        """Confirm the submitted policy exists and return mandatory evidence requirements for Intake."""
        assessment, claim = identity()
        policy_id = submitted_policy_id(claim)
        bound_id, corpus_ids = wording_scope(assessment, claim)
        wording_ready = True
        with store.session() as session:
            row = session.get(Policy, policy_id) if policy_id else None
            if claim.payload.get("source") == "Manual submission" and assessment.execution.get("mode") != "deterministic-tools":
                identifiers = [bound_id] if bound_id else corpus_ids
                wording_ready = session.scalar(select(PolicyDocument.id).where(
                    PolicyDocument.id.in_(identifiers), PolicyDocument.status == "ready").limit(1)) is not None
        policy = row.payload if row else {}
        applicable_repair = claim.payload.get("loss_type") == "repair" and policy_applies_to_loss(policy, claim.payload)
        repair = (policy.get("repair_terms") or {}) if applicable_repair else {}
        benefits = repair.get("additional_charges", {}).get("eligible_charges", [])
        return record("read_coverage_requirements", {"policy_id": claim.payload.get("policy_id")}, {
            "policy_available": row is not None,
            "missing_policy_terms": [field for field in REQUIRED_POLICY_FIELDS if policy.get(field) in (None, "")] if row else [],
            "chassis_required": bool((policy.get("vehicle") or {}).get("chassis_number")),
            "police_report_required": requires_police_report(policy, claim.payload),
            "missing_supporting_documents": (missing_supporting_documents(policy, claim.payload)
                                             if claim.payload.get("source") == "Synthetic generator" else []),
            "typed_loss_excluded": bool(policy.get("insurance_type") and claim.payload.get("loss_type")
                                        and not policy_lists_loss(policy, claim.payload["loss_type"])),
            "charge_receipt_requirements": [{"category": item["category"], "name": item["name"]}
                                            for item in benefits if item["receipt_required"]],
            "wording_ready": wording_ready,
        })

    @tool
    def search_policy_documents(query: str) -> dict:
        """Search policy wording available to this claim. Check applicability and cite passage IDs."""
        assessment, claim = identity()
        bound_id, corpus_ids = wording_scope(assessment, claim)
        try:
            result = (search_uploaded_documents(store, corpus_ids, query) if corpus_ids
                      else search_document(store, bound_id, query))
        except (ValueError, OSError, RuntimeError) as error:
            result = {"status": "unavailable", "passages": [], "error_type": type(error).__name__}
        result.update(bound=bool(bound_id), synthetic=claim.payload.get("source") == "Synthetic generator")
        return record("search_policy_documents", {"query": query}, {"result": result, "evidence_ref": "tool:" + str(uuid.uuid4())})

    @tool
    def retrieve_comparable_claims() -> dict:
        """Retrieve earlier staff-approved claim narratives, itemised cost documents and additional costs for comparison."""
        _, claim = identity()
        with store.session() as session:
            selected = store.history.records(session, claim)
        return record("retrieve_comparable_claims", {"loss_type": claim.payload["loss_type"], "available_before": claim.received_at,
                                          "currency": claim.payload["currency"]}, {"records": selected})

    @tool
    def verify_reused_cost_document() -> dict:
        """Verify an exact cost-document copy from a different vehicle and earlier synthetic incident."""
        _, claim = identity()
        with store.session() as session:
            verified = verify_reuse(session, claim)
        return record("verify_reused_cost_document", {"before_sequence": claim.sequence}, {"verified": verified})

    @tool
    def verify_historical_leads() -> dict:
        """Verify reported cost-document reuse and preserve the records checked."""
        assessment, claim = identity()
        source = context_snapshot if context_snapshot is not None else assessment.context
        finding = source.get("findings", {}).get("historical", {})
        leads = finding.get("details", {}).get("fraud_investigation", {}).get("leads", [])
        verified, checks = [], []
        criteria = {"minimum_distinct_incidents": 2, "eligible_comparable_scope": True,
                    "nonempty_incident_id": True, "document_reference_comparison": "exact"}
        with store.session() as session:
            eligible = {row["id"]: row for row in store.history.records(session, claim)} if leads else {}
            for lead in leads:
                reference = lead.get("document_reference")
                records, missing, out_of_scope = [], [], []
                for identifier in dict.fromkeys(lead["history_ids"]):
                    if identifier in eligible:
                        records.append({"history_id": identifier, "payload": copy.deepcopy(eligible[identifier])})
                    elif store.history.exists(session, identifier):
                        out_of_scope.append(identifier)
                    else:
                        missing.append(identifier)
                incidents = sorted({
                    row["payload"]["incident_id"] for row in records
                    if reference and uses_document_reference(row["payload"], reference)
                    and row["payload"].get("incident_id")
                })
                passed = len(incidents) >= criteria["minimum_distinct_incidents"]
                checks.append({"document_reference": reference, "requested_history_ids": list(lead["history_ids"]),
                    "records": records, "missing_history_ids": missing, "out_of_scope_history_ids": out_of_scope,
                    "matching_incident_ids": incidents, "verified": passed})
                if passed:
                    verified.append(lead)
        return record("verify_historical_leads", {"lead_count": len(leads)},
                      {"verified": verified, "criteria": criteria, "checks": checks})

    allowed = {"intake": [read_submission, find_duplicates, read_coverage_requirements],
               "coverage": [read_context, read_submission, retrieve_policy, search_policy_documents],
               "historical": [read_context, read_submission, retrieve_comparable_claims],
               "fraud_initial": [read_context, read_submission],
               "fraud_followup": [read_context, read_submission, verify_historical_leads]}
    permitted = allowed[stage]
    _, current_claim = identity()
    if current_claim.payload.get("source") == "Manual submission":
        permitted = [*permitted, read_uploaded_documents]
    elif (stage == "intake" and current_claim.payload.get("source") == "Synthetic generator"
          and any(doc.get("document_reference") for doc in current_claim.payload.get("documents", []))):
        permitted = [*permitted, verify_reused_cost_document]
    return {item.name: item for item in permitted}
