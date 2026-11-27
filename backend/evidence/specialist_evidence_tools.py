"""Build role-scoped LangChain tools that retrieve and audit records for one assessment."""
import copy
import uuid
from decimal import Decimal
from typing import Annotated

from pydantic import Field
from sqlalchemy import select
from langchain_core.tools import StructuredTool

from backend.assurance.investigation_tracing import trace_span
from backend.storage.sqlite_store import Assessment, Claim, InvestigationRevision, Policy, PolicyDocument, SyntheticPolicyHolder, ToolCall, digest
from backend.claim_schemas import REQUIRED_POLICY_FIELDS
from backend.evidence.motor_benefits import fixed_motor_benefit, mandatory_police_threshold
from backend.evidence.policy_schedule import schedule_restriction
from backend.evidence.required_claim_evidence import missing_supporting_documents, policy_applies_to_loss, policy_lists_loss, requires_police_report
from backend.workflow.specialist_context import specialist_context
from backend.evidence.claim_history import uses_document_reference
from backend.evidence.policy_wording_index import PolicyTopic, read_document_page, search_document, search_uploaded_documents
from backend.evidence.synthetic_cost_history import verify_reuse
from backend.evidence.synthetic_claim_text import resolve_claim_text
from backend.evidence.vehicle_market_value import VehicleMarketQuery, lookup_vehicle_market_value
from backend.workflow.selected_policy import selectable_policy_pdf


class EvidenceTool(StructuredTool):
    def invoke(self, input, config=None, **kwargs):
        kind = "retriever" if self.name in {"search_policy_documents", "read_policy_page", "retrieve_policy",
                                          "retrieve_comparable_claims", "retrieve_vehicle_market_value"} else "tool"
        metadata = dict(self.metadata or {})
        if self.name == "search_policy_documents" and isinstance(input, dict):
            metadata["topic"] = input.get("topic")
        if self.name == "read_policy_page" and isinstance(input, dict):
            metadata["page"] = input.get("page")
        with trace_span(f"tool.{self.name}", kind=kind, metadata=metadata, content=input) as span:
            result = super().invoke(input, config, **kwargs)
            span.output = result
            span.summary = {"result_hash": digest(result)}
            if kind == "retriever":
                retrieved = result.get("result", result)
                if "status" in retrieved:
                    span.summary["status"] = retrieved["status"]
                if "passages" in retrieved:
                    span.summary["pages"] = sorted({passage["page"] for passage in retrieved["passages"] if "page" in passage})
                if self.name == "retrieve_vehicle_market_value":
                    count = int(retrieved.get("status") == "found")
                elif self.name == "retrieve_policy":
                    count = int(bool(retrieved.get("selected_policy") or retrieved.get("policy")))
                else:
                    count = len(retrieved.get("passages", retrieved.get("records", [])))
                span.summary["result_count"] = count
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
            if selected := context.get("selected_policy"):
                return selected["id"], []
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
        """Retrieve the frozen form, submitted text and document fields derived from the evidence."""
        _, claim = identity()
        payload = claim.payload
        if payload.get("source") == "Manual submission":
            from backend.evidence.uploaded_claim_documents import effective_submission
            payload = effective_submission(store, assessment_id, payload)
        else:
            payload = resolve_claim_text(payload)
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
                   and (row.payload.get("policy_id") == claim.payload.get("policy_id") and row.payload["loss_date"] == claim.payload["loss_date"]
                        and row.id != claim.payload.get("parent_claim_id") if manual else row.payload["incident_id"] == claim.payload["incident_id"])]
        return record("find_duplicates", {"criterion": "registration + policy_id + loss_date" if manual else "registration + incident_id", "before_sequence": claim.sequence}, {"matches": matches})

    @tool
    def retrieve_policy() -> dict:
        """Retrieve the selected policy version or the legacy policy schedule."""
        assessment, claim = identity()
        if selected := assessment.context.get("selected_policy"):
            return record("retrieve_policy", {"policy_document_id": selected["id"]}, {"policy": None, "selected_policy": selected})
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
        """Check the selected PDF and its type or return requirements from a verified saved schedule."""
        assessment, claim = identity()
        submission = resolve_claim_text(claim.payload)
        policy_id = submitted_policy_id(claim)
        bound_id, corpus_ids = wording_scope(assessment, claim)
        selected = assessment.context.get("selected_policy")
        wording_ready = True
        with store.session() as session:
            row = session.get(Policy, policy_id) if policy_id and not selected else None
            if selected:
                document = session.get(PolicyDocument, selected["id"])
                wording_ready = selectable_policy_pdf(document, already_selected=True) and document.file_hash == selected["file_hash"]
            elif claim.payload.get("source") == "Manual submission" and assessment.execution.get("mode") != "deterministic-tools":
                identifiers = [bound_id] if bound_id else corpus_ids
                wording_ready = session.scalar(select(PolicyDocument.id).where(
                    PolicyDocument.id.in_(identifiers), PolicyDocument.status == "ready").limit(1)) is not None
        policy = row.payload if row else {}
        schedule = assessment.context.get("policy_schedule") if selected else None
        matched_schedule = bool(schedule and schedule.get("id") and wording_ready
            and claim.payload.get("source") == "Synthetic generator" and schedule["policy_number"] == policy_id
            and schedule["vehicle"]["registration"] == submission.get("registration")
            and not schedule_restriction(schedule, submission))
        benefit = None
        if matched_schedule:
            benefit = (fixed_motor_benefit(schedule, submission["motor_benefit"]) if submission.get("motor_benefit") else
                       next((benefit for benefit in schedule["cover"] if benefit["loss_type"] == submission.get("loss_type")
                             and benefit["limit_basis"] not in {"excluded", "unconfirmed", "unspecified"}), None))
        police_threshold = mandatory_police_threshold(schedule, benefit) if benefit else None
        requires_scheduled_report = police_threshold is not None and Decimal(submission["gross_amount"]) > police_threshold
        applicable_repair = claim.payload.get("loss_type") == "repair" and policy_applies_to_loss(policy, claim.payload)
        repair = (policy.get("repair_terms") or {}) if applicable_repair else {}
        benefits = repair.get("additional_charges", {}).get("eligible_charges", [])
        request = {"policy_document_id": selected["id"]} if selected else {"policy_id": claim.payload.get("policy_id")}
        return record("read_coverage_requirements", request, {
            "policy_available": wording_ready if selected else row is not None,
            **({"selected_policy": selected} if selected else {}),
            **({"matched_cost_evidence_required": True} if benefit else {}),
            "missing_policy_terms": [field for field in REQUIRED_POLICY_FIELDS if policy.get(field) in (None, "")] if row else [],
            "police_report_required": requires_scheduled_report or requires_police_report(policy, claim.payload),
            "missing_supporting_documents": (missing_supporting_documents(policy, submission)
                                             if claim.payload.get("source") == "Synthetic generator" else []),
            "typed_loss_excluded": bool(policy.get("insurance_type") and claim.payload.get("loss_type")
                                        and not policy_lists_loss(policy, claim.payload["loss_type"])),
            "charge_receipt_requirements": [{"category": item["category"], "name": item["name"]}
                                            for item in benefits if item["receipt_required"]],
            "wording_ready": wording_ready,
        })

    @tool
    def search_policy_documents(query: str, topic: PolicyTopic | None = None) -> dict:
        """Search policy wording. For a selected PDF, search one topic at a time and cite passage IDs."""
        assessment, claim = identity()
        bound_id, corpus_ids = wording_scope(assessment, claim)
        try:
            result = (search_uploaded_documents(store, corpus_ids, query) if corpus_ids
                      else search_document(store, bound_id, query, **({"topic": topic} if topic else {})))
        except (ValueError, OSError, RuntimeError) as error:
            result = {"status": "unavailable", "passages": [], "error_type": type(error).__name__}
        if (selected := assessment.context.get("selected_policy")) and result.get("file_hash") != selected["file_hash"]:
            result = {"status": "unavailable", "passages": [], "reason": "Selected PDF version could not be verified"}
        result.update(bound=bool(bound_id), synthetic=claim.payload.get("source") == "Synthetic generator")
        return record("search_policy_documents", {"query": query, **({"topic": topic} if topic else {})},
                      {"result": result, "evidence_ref": "tool:" + str(uuid.uuid4())})

    @tool
    def read_policy_page(page: Annotated[int, Field(ge=1)]) -> dict:
        """Read the selected PDF page in order to check a table, adjacent clauses or a cross-reference."""
        assessment, _ = identity()
        selected = assessment.context.get("selected_policy")
        result = read_document_page(store, selected["id"], page)
        if result.get("file_hash") != selected["file_hash"]:
            result = {"status": "unavailable", "passages": []}
        return record("read_policy_page", {"page": page}, {"result": result, "evidence_ref": "tool:" + str(uuid.uuid4())})

    @tool
    def retrieve_vehicle_market_value() -> dict:
        """Get a dated advertised asking price benchmark for the claim's recorded make, model, year and loss date."""
        result = lookup_vehicle_market_value(market_query)
        return record("retrieve_vehicle_market_value", market_query.model_dump(mode="json"),
                      {"result": result, "evidence_ref": "tool:" + str(uuid.uuid4())})

    @tool
    def retrieve_comparable_claims() -> dict:
        """Retrieve earlier eligible claim narratives, itemised cost documents and additional costs for comparison."""
        _, claim = identity()
        with store.session() as session:
            selected = store.history.records(session, claim)
        return record("retrieve_comparable_claims", {"loss_type": claim.payload["loss_type"], "available_before": claim.received_at,
                                          "currency": claim.payload["currency"]}, {"records": selected})

    @tool
    def verify_reused_cost_document() -> dict:
        """Verify an exact cost-document copy from a separate earlier synthetic incident."""
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
    current_assessment, current_claim = identity()
    selected_coverage = stage == "coverage" and current_assessment.context.get("selected_policy")
    if selected_coverage:
        permitted = [*permitted, read_policy_page]
    context = context_snapshot if context_snapshot is not None else current_assessment.context
    prior_fraud = context.get("findings", {}).get("fraud_initial", {}).get("details", {})
    if selected_coverage or stage == "fraud_initial" or (stage == "fraud_followup" and not prior_fraud.get("vehicle_market_check")):
        payload = current_claim.payload
        if payload.get("loss_type") in {"repair", "glass", "theft"} and isinstance(payload.get("vehicle"), dict):
            vehicle = payload["vehicle"]
            try:
                market_query = VehicleMarketQuery(make=vehicle.get("make"), model=vehicle.get("model"),
                                                  year=vehicle.get("year"), as_of=payload.get("loss_date"))
            except ValueError:
                pass
            else:
                permitted.append(retrieve_vehicle_market_value)
    if current_claim.payload.get("source") == "Manual submission":
        permitted = [*permitted, read_uploaded_documents]
    elif (stage == "intake" and current_claim.payload.get("source") == "Synthetic generator"
          and any(doc.get("document_reference") for doc in current_claim.payload.get("documents", []))):
        permitted = [*permitted, verify_reused_cost_document]
    return {item.name: item for item in permitted}
