"""Send audited Claude requests and run specialist tool loops and route recommendations."""
import copy
import json
import threading
import uuid
from datetime import date
from decimal import Decimal
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from sqlalchemy import func, select

from backend.agents import coverage_specialist as coverage, fraud_specialist as fraud, historical_comparer as historical
from backend.agents.investigation_prompts import ADJUSTER_PRIORITY_INSTRUCTIONS, EVIDENCE_BOUNDARY, ROUTE_RECOMMENDATION, specialist_prompt
from backend.agents.specialist_preflight import apply_verified_controls, execute_checks
from backend.agents.anthropic_stream import read_message_stream
from backend.anthropic_errors import AnthropicRequestError
from backend.assurance.investigation_tracing import anthropic_usage, trace_span
from backend.assurance.investigation_display import (AdjusterPriority, adjuster_review_candidates, omit_provisional_payable,
                                                      select_adjuster_summary)
from backend.storage.sqlite_store import Assessment, Claim, ModelInvocation, ToolCall, digest
from backend.claim_schemas import Finding, MotorLossType, Route, THIRD_PARTY_LOSSES
from backend.model_settings import settings
from backend.workflow.specialist_context import DEPENDENCIES, generated_policy_context, is_specialist_stage, specialist_context
from backend.agents.coverage_wording_review import PolicyConflict, assess_bound_wording, assess_selected_wording
from backend.workflow.investigation_retry import investigation_age
from backend.evidence.required_claim_evidence import committed_required_gaps, missing_income_loss_evidence
from backend.evidence.policy_schedule import scheduled_financials
from backend.evidence.specialist_evidence_tools import build_tools
from backend.evidence.synthetic_claim_text import available_text_refs
from backend.evidence.vehicle_market_value import market_cap_only_review
from backend.evidence.third_party_assessment import CoverageEvidenceRequest, ThirdPartyCostAssessment, assess_third_party, claimed_costs
from backend.workflow.versions import V4

CALL_LOCK = threading.Lock()
COVERAGE_MAX_TOKENS = 3200
READINESS_MAX_TOKENS = 3200
OUTPUT_RETRY_MAX_TOKENS = 8192
CHECK_EVIDENCE_TOOLS = {
    "intake": ("read_coverage_requirements", "verify_reused_cost_document"),
    "coverage": ("retrieve_policy", "retrieve_vehicle_market_value"),
    "historical": ("retrieve_comparable_claims",),
    "fraud_initial": ("retrieve_vehicle_market_value",),
    "fraud_followup": ("verify_historical_leads", "retrieve_vehicle_market_value"),
}


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specialist: Literal["intake", "fraud", "coverage", "historical"]
    task: str = Field(min_length=1, max_length=1000)


class SpecialistAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1)
    review_reasons: list[str] = Field(max_length=10)
    evidence: list[str] = Field(min_length=1, max_length=30)
    policy_applicability: Literal["applicable", "unclear", "irrelevant"] = "unclear"


class CoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    item: str = Field(min_length=1)
    outcome: Literal["covered", "not_covered", "undetermined"]
    policy_term: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    document_refs: list[str] = Field(default_factory=list, max_length=8)


class PolicyOmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    item: str = Field(min_length=1)
    impact: str = Field(min_length=1)


class DocumentInconsistency(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reason: str = Field(min_length=1)
    document_refs: list[str] = Field(min_length=1, max_length=8)


class CoverageHistoryContext(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    claim_ids: list[str]
    reason: str = Field(min_length=1, max_length=1200)


class CoverageAnalysis(SpecialistAnalysis):
    classified_loss: MotorLossType | Literal["glass", "injury", "other", "uncertain"] | None = None
    liability_confirmed: bool = False
    coverage_conclusion: Literal["COVERED", "NOT_COVERED", "UNRESOLVED"]
    coverage_items: list[CoverageItem] = Field(max_length=12)
    policy_omissions: list[PolicyOmission] = Field(default_factory=list, max_length=8)
    document_inconsistencies: list[DocumentInconsistency] = Field(default_factory=list, max_length=8)
    historical_context: CoverageHistoryContext | None = None


class WordingCoverageItem(CoverageItem):
    basis: Literal["schedule", "wording", "both"]
    passage_refs: list[str] = Field(max_length=5)


class BoundCoverageAnalysis(CoverageAnalysis):
    coverage_items: list[WordingCoverageItem] = Field(max_length=12)
    applicable_passages: list[str] = Field(max_length=16)
    policy_conflicts: list[PolicyConflict] = Field(max_length=8)


class SelectedCoverageItem(CoverageItem):
    passage_refs: list[str] = Field(default_factory=list, max_length=5)


class PDFDocumentRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    document_type: Literal["police_report", "incident_report", "witness_statement"]
    passage_ref: str = Field(min_length=1)
    requirement_quote: str = Field(min_length=12)
    reason: str = Field(min_length=1)


class SelectedCoverageAnalysis(CoverageAnalysis):
    coverage_items: list[SelectedCoverageItem] = Field(max_length=12)
    required_documents: list[PDFDocumentRequirement] = Field(default_factory=list, max_length=5)
    liability_evidence_refs: list[str] | None = Field(default=None, max_length=5)
    liability_evidence_requests: list[CoverageEvidenceRequest] = Field(default_factory=list, max_length=3)
    cost_assessments: list[ThirdPartyCostAssessment] = Field(default_factory=list, max_length=60)

    @model_validator(mode="after")
    def validate_third_party_handoff(self, info: ValidationInfo):
        if info.context is None:
            return self
        refs = [cost.line_ref for cost in self.cost_assessments]
        if set(refs) != info.context["line_refs"] or len(set(refs)) != len(refs):
            raise ValueError("Return one cost_assessments entry for every supplied claimed_costs line_ref.")
        if self.liability_evidence_refs is None or self.liability_confirmed != bool(self.liability_evidence_refs):
            raise ValueError("Use exact evidence references when liability is confirmed and an empty list otherwise.")
        items = {item.item: item.outcome for item in self.coverage_items}
        for cost in self.cost_assessments:
            if cost.policy_item is None:
                continue
            if cost.policy_item not in items:
                raise ValueError("Use an exact coverage_items label or null as the cost's policy_item.")
            if items[cost.policy_item] == "covered" and not (cost.amount_evidence_refs or cost.evidence_requests):
                raise ValueError("For each applicable cost, cite evidence that verifies the amount or record an evidence request with its owner.")
        return self


class ScheduleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    registration: str = Field(min_length=1, max_length=40)
    starts_on: date
    ends_on: date
    cover: Literal["comprehensive", "third_party", "third_party_fire_theft"]
    passage_refs: list[str] = Field(min_length=1, max_length=5)


class VerifiedCoverageDraft(SelectedCoverageAnalysis):
    schedule: ScheduleEvidence | None = None


class HistoricalAnalysis(SpecialistAnalysis):
    amount_conclusion: Literal["SUPPORTED", "REQUIRES_REVIEW", "INSUFFICIENT_EVIDENCE"]
    comparable_claim_ids: list[str]

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_leads(cls, value):
        # Stored responses can include old model proposals. The recorded references determine leads now.
        if isinstance(value, dict) and "history_leads" in value:
            return {key: item for key, item in value.items() if key != "history_leads"}
        return value


class FraudObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str = Field(min_length=1, max_length=300)
    value: str = Field(min_length=1, max_length=5000)


class FraudConcern(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    severity: Literal["high", "medium", "low"]
    reason: str = Field(min_length=1, max_length=1200)
    evidence: list[str] = Field(min_length=1, max_length=5)
    basis: Literal["conflicting_evidence", "missing_information"] | None = None
    observed_facts: list[FraudObservation] = Field(default_factory=list, max_length=4)


class FraudPlausibility(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    assessment: Literal["consistent", "inconclusive", "contradicted"]
    reason: str = Field(min_length=1, max_length=700)
    observed_facts: list[FraudObservation] = Field(default_factory=list, max_length=4)


class FraudAnalysis(SpecialistAnalysis):
    fraud_concerns: list[FraudConcern] = Field(default_factory=list, max_length=10)
    plausibility: FraudPlausibility | None = None


ANALYSIS_TYPES = {"coverage": CoverageAnalysis,
                  "historical": HistoricalAnalysis, "fraud": FraudAnalysis}


class SummaryEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    candidate_id: str = Field(min_length=1)
    finding_quote: str = Field(min_length=12)
    document: str = Field(min_length=1)
    location: str | None = None
    reason: str = Field(min_length=1)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route: Route
    explanation: str = Field(min_length=1, max_length=1800, description=(
        "Plain-language reason for the route. Use policy clause names and keep raw citation IDs in evidence. "
        "If the committed coverage payable is missing, state that the payable is unverified and omit any preliminary settlement amount."))
    evidence: list[str] = Field(max_length=8)
    outstanding_gaps: list[str] = Field(max_length=20)
    evidence_requests: list[SummaryEvidenceRequest] = Field(default_factory=list)
    adjuster_priorities: list[AdjusterPriority] = Field(default_factory=list)

    @field_validator("adjuster_priorities", mode="wrap")
    @classmethod
    def validate_priorities(cls, value, handler):
        try:
            return handler(value)
        except ValidationError:
            # An unusable summary does not invalidate the routing recommendation.
            return []


class ModelOutputTruncated(ValueError):
    """The provider reached its output limit before finishing the response."""


class IncompleteModelResponse(ValueError):
    """The provider finished without submitting a tool result."""


class StructuredResultRejected(ValueError):
    def __init__(self, result, cause):
        super().__init__(str(cause))
        self.result = result
        self.cause = cause


def check_deadline(store, assessment_id):
    with store.session() as session:
        assessment = session.get(Assessment, assessment_id)
        claim = session.get(Claim, assessment.claim_id)
    age = investigation_age(assessment, claim)
    if age >= 900:
        raise TimeoutError("Claim processing deadline exceeded")
    return 900 - age


def model_context_summary(stage, messages):
    if not is_specialist_stage(stage) or not messages or messages[0].get("role") != "user":
        return {}
    try:
        packet = json.loads(messages[0].get("content"))
    except (TypeError, ValueError):
        return {}
    if not isinstance(packet, dict) or not isinstance(packet.get("context"), dict):
        return {}
    context = packet["context"]
    return {
        "input_sections": [key for key in ("task", "required_checks", "submission", "submitted_document_refs",
            "submitted_document_text_refs", "uploaded_documents", "uploaded_document_facts", "context",
            "initial_policy_evidence") if key in packet],
        "context_sections": [key for key in ("version", "findings", "policy_search_mode", "policy_corpus_ids",
            "wording_document_id", "synthetic_wording_document_id", "selected_policy") if key in context],
        "context_version": context.get("version"),
        "prior_findings": [key for key in DEPENDENCIES if key in context.get("findings", {})],
        "policy_passage_count": sum(len(entry["result"].get("passages", [])) for entry in packet.get("initial_policy_evidence", [])),
    }


def call_model(store, assessment_id, stage, system, messages, tools, choice, *, max_tokens=None):
    import os
    import time

    remaining = check_deadline(store, assessment_id)
    deadline = time.monotonic() + remaining
    identifier = str(uuid.uuid4())
    is_coverage = stage.endswith(":coverage")
    default_limit = COVERAGE_MAX_TOKENS if is_coverage else READINESS_MAX_TOKENS if stage.startswith("readiness:") else 1800
    output_limit = max_tokens if max_tokens is not None else default_limit
    payload = {"model": settings()[1], "max_tokens": output_limit, "system": EVIDENCE_BOUNDARY + system,
               "messages": messages, "tools": tools, "tool_choice": choice}
    streaming = os.getenv("CLAIMS_STREAM_MODEL") == "1"
    if streaming:
        payload["stream"] = True
    with CALL_LOCK, store.session.begin() as session:
        count = session.scalar(select(func.count()).select_from(ModelInvocation).where(ModelInvocation.assessment_id == assessment_id))
        if count >= 40:
            raise ValueError("Claim model request budget exhausted")
        session.add(ModelInvocation(id=identifier, assessment_id=assessment_id, stage=stage, payload=payload, response={}))
    metadata = {"stage": stage, "model_invocation_id": identifier, "prompt_version": digest(payload["system"]),
                "request_hash": digest(payload), "message_count": len(messages),
                "available_tools": [tool["name"] for tool in tools], **model_context_summary(stage, messages)}
    with trace_span("context.model_request", metadata=metadata):
        pass
    model_info = {"model": payload["model"], "model_parameters": {"max_tokens": output_limit, "stream": streaming}}
    with trace_span("model.anthropic", kind="generation", metadata=metadata, content=payload, retain_content=False,
                    generation=model_info) as span:
        data = {}
        try:
            read_seconds = 90 if stage.endswith((":coverage", ":historical")) else 60
            if output_limit >= OUTPUT_RETRY_MAX_TOKENS:
                read_seconds = 180
            timeout = httpx.Timeout(min(read_seconds, remaining), connect=min(10, remaining))
            key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise ValueError("Set ANTHROPIC_API_KEY to run Claude investigations.")
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            endpoint = "https://api.anthropic.com/v1/messages"
            with httpx.Client(timeout=timeout) as client:
                if streaming:
                    with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                        if response.is_error:
                            response.read()
                        response.raise_for_status()
                        read_message_stream(response, data, deadline, span)
                else:
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    try:
                        body = response.json()
                    except ValueError as error:
                        raise IncompleteModelResponse("Invalid model response envelope") from error
                    if not isinstance(body, dict):
                        raise IncompleteModelResponse("Invalid model response envelope")
                    data = body
            if data.get("stop_reason") == "max_tokens":
                raise ModelOutputTruncated("The model reached its response length limit before finishing.")
            if data.get("stop_reason") != "tool_use":
                raise IncompleteModelResponse("Incomplete model response")
            with store.session.begin() as session:
                row = session.get(ModelInvocation, identifier)
                row.response, row.status = data, "complete"
            return data
        except Exception as error:
            provider_error = AnthropicRequestError(error, payload["model"]) if isinstance(error, httpx.HTTPStatusError) else None
            with store.session.begin() as session:
                row = session.get(ModelInvocation, identifier)
                row.status = "failed"
                failure = {"error_type": type(provider_error or error).__name__}
                if provider_error:
                    failure.update(http_status=provider_error.response.status_code,
                                   request_id=provider_error.response.headers.get("request-id"), provider_error=str(provider_error))
                row.response = {**data, **failure}
            if provider_error:
                raise provider_error from error
            raise
        finally:
            span.output = data
            span.model = data.get("model")
            blocks = data.get("content")
            stop = data.get("stop_reason")
            span.summary = {"stop_reason": stop if stop in ("tool_use", "end_turn", "max_tokens", "stop_sequence", "refusal", "pause_turn") else "unknown",
                            "selected_tools": [block["name"] if block.get("name") in metadata["available_tools"] else "unrecognized"
                                               for block in blocks
                                               if isinstance(block, dict) and block.get("type") == "tool_use"]
                            if isinstance(blocks, list) else []}
            span.usage = anthropic_usage(data)


def model_with_timeout_retry(*args, **kwargs):
    from time import sleep

    from backend.anthropic_errors import retry_delay

    try:
        return call_model(*args, **kwargs)
    except httpx.TimeoutException:
        return call_model(*args, **kwargs)
    except httpx.HTTPStatusError as error:
        delay = retry_delay(error.response)
        if delay is None:
            raise
        sleep(delay)
        return call_model(*args, **kwargs)


def structured_call(store, assessment_id, stage, instruction, data, schema, name, allowed_sources=None, *, validate_result=None):
    input_schema = schema.model_json_schema()
    if allowed_sources is not None:
        if allowed_sources:
            input_schema["properties"]["evidence"]["items"] = {"type": "string", "enum": sorted(allowed_sources)}
        else:
            input_schema["properties"]["evidence"]["maxItems"] = 0
    messages = [{"role": "user", "content": json.dumps(data)}]
    tools = [{"name": name, "description": "Submit the structured result", "input_schema": input_schema}]
    choice = {"type": "tool", "name": name}
    for budget in ({}, {"max_tokens": OUTPUT_RETRY_MAX_TOKENS}):
        try:
            response = model_with_timeout_retry(store, assessment_id, stage, instruction, messages, tools, choice, **budget)
            blocks = response.get("content") if isinstance(response, dict) else None
            calls = [item for item in blocks if isinstance(item, dict) and item.get("type") == "tool_use"] if isinstance(blocks, list) else []
            if len(calls) != 1 or calls[0].get("name") != name or not isinstance(calls[0].get("input"), dict):
                raise IncompleteModelResponse("The model did not submit one complete result through the required tool.")
            with trace_span(f"tool.{name}", kind="tool", metadata={"stage": stage}, content=calls[0]["input"]) as span:
                result = schema.model_validate(calls[0]["input"])
                if validate_result is not None:
                    try:
                        validate_result(result)
                    except ValueError as error:
                        raise StructuredResultRejected(result.model_dump(mode="json"), error) from error
                span.output = result.model_dump(mode="json")
                span.summary = {"validated": True}
                return result
        except (ModelOutputTruncated, IncompleteModelResponse, ValidationError, StructuredResultRejected) as error:
            if budget:
                raise
            if isinstance(error, ValidationError):
                issues = error.errors(include_input=False, include_context=False, include_url=False)
                feedback = {"invalid_fields": [{"path": issue["loc"], "code": issue["type"]} for issue in issues[:6]]}
            else:
                feedback = {"problem": str(error)}
            messages = [*messages, {"role": "user", "content": json.dumps({"validation_feedback": feedback})}]
            instruction += ("\n\nSubmit a complete corrected result through the required tool using the same supplied evidence. "
                            "Address the validation feedback. Keep explanations concise and retain all required evidence and checks.")


def record_evidence(store, assessment_id, stage, name, request, result, reference=None, *, supporting_evidence=None):
    """Persist evidence before exposing its reference to the model."""
    response = {"result": copy.deepcopy(result), "evidence_ref": reference or "tool:" + str(uuid.uuid4())}
    if supporting_evidence is not None:
        response["supporting_evidence"] = copy.deepcopy(supporting_evidence)
    with store.session.begin() as session:
        session.add(ToolCall(assessment_id=assessment_id, stage=stage, tool=name,
                            request=request, response=response, response_hash=digest(response)))
    return response


def prior_evidence_sources(store, assessment_id, view):
    """Allow only visible prior references backed by this assessment's audit."""
    candidates = set()
    for finding in view.get("findings", {}).values():
        candidates.update(finding.get("evidence", []))
        candidates.update(finding.get("details", {}).get("sources", []))
    if not candidates:
        return set()
    with store.session() as session:
        calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment_id)).all()
    recorded = set()
    for call in calls:
        if digest(call.response) != call.response_hash:
            raise ValueError("Prior evidence integrity check failed")
        if call.response.get("evidence_ref"):
            recorded.add(call.response["evidence_ref"])
        recorded.update(passage["source"] for passage in call.response.get("result", call.response).get("passages", []))
    return candidates.intersection(recorded)


def validate_specialist_analysis(schema, submitted, stage, turn, *, validation_context=None):
    with trace_span("tool.submit_analysis", kind="tool", metadata={"stage": stage, "turn": turn}, content=submitted) as span:
        analysis = schema.model_validate(submitted, context=validation_context)
        span.output = analysis.model_dump(mode="json")
        span.summary = {"validated": True}
        return analysis


def run_specialist(store, assessment_id, context, round_number, role, task):
    """Workers return results only; collector owns shared-context writes."""
    stage = "fraud_initial" if role == "fraud" else role
    findings = context.get("findings", {})
    has_leads = (findings.get("fraud_initial", {}).get("details", {}).get("checks_executed")
                 and findings.get("historical", {}).get("details", {}).get("fraud_investigation", {}).get("leads"))
    scope_stage = "fraud_followup" if role == "fraud" and has_leads else stage
    supporting_evidence = []
    evidence = {}

    def capture_check_evidence(name, request, result):
        evidence[name] = result
        if name in CHECK_EVIDENCE_TOOLS[scope_stage]:
            supporting_evidence.append({"tool": name, "request": request, "result": result})

    scoped = build_tools(store, assessment_id, scope_stage, context_snapshot=context,
                         audit_stage=f"r{round_number}:{role}", on_result=capture_check_evidence)
    audit_stage = f"r{round_number}:{role}"
    # The role already determines these checks. Run and audit them before any
    # model request, so Claude cannot skip them or spend a turn requesting them.
    check_deadline(store, assessment_id)
    with trace_span("checks.execute", kind="tool", metadata={"stage": audit_stage}) as span:
        preflight = execute_checks(role, scoped, follow_up=scope_stage == "fraud_followup")
        apply_verified_controls(role, preflight, scoped, evidence, context.get("policy_schedule"))
        checked = preflight.model_dump(mode="json")
        span.output = checked
        span.summary = {"status": preflight.status}
    check_ref = "tool:" + str(uuid.uuid4())
    checked["details"]["checks_executed"] = True
    checked["details"]["sources"] = [check_ref]
    checked["details"]["tools_used"] = ["execute_checks"]
    market_check = checked["details"].get("settlement_market_cap") or checked["details"].get("vehicle_market_check", {})
    market_ref = market_check.get("evidence_ref")
    if market_ref:
        checked["details"]["sources"].append(market_ref)
        checked["details"]["tools_used"].append("retrieve_vehicle_market_value")
    if role == "intake":
        checked["details"]["sources"].extend(available_text_refs(evidence["read_submission"]))
        checked["evidence"] = checked["details"]["sources"][:]
    required_checks = record_evidence(store, assessment_id, audit_stage, "execute_checks",
        {"role": role, "follow_up": scope_stage == "fraud_followup"}, checked, check_ref,
        supporting_evidence=supporting_evidence)
    if role == "intake":
        return checked
    prepared_evidence = evidence.copy()
    uses_uploaded_wording = role == "coverage" and prepared_evidence["read_submission"].get("source") == "Manual submission"
    selected_policy = prepared_evidence.get("retrieve_policy", {}).get("selected_policy") if role == "coverage" else None
    selected_claim = bool(context.get("selected_policy"))
    v4 = context.get("workflow") == V4
    bound_wording_id = (context.get("synthetic_wording_document_id")
                       if role == "coverage" and prepared_evidence["read_submission"].get("source") == "Synthetic generator"
                       else None)
    sources = set(checked["details"]["sources"])
    if role == "historical" and prepared_evidence["read_submission"].get("gross_amount") not in (None, ""):
        sources.add("submission:/gross_amount")
    retrieved = []
    invoked = checked["details"]["tools_used"][:]

    # Python retains all scoped check tools. Claude only sees evidence sources
    # whose full results are not already supplied in its initial input.
    available = {name: scoped[name] for name in ("read_submission", "read_uploaded_documents") if name in scoped}
    if role == "coverage":
        # The full structured submission is already supplied with Coverage's checks.
        available.pop("read_submission", None)
    elif role == "fraud" and selected_claim:
        available.pop("read_submission", None)
    if uses_uploaded_wording or bound_wording_id or selected_policy:
        available["search_policy_documents"] = scoped["search_policy_documents"]
    if selected_policy:
        available["read_policy_page"] = scoped["read_policy_page"]
        if "retrieve_vehicle_market_value" in scoped and not market_ref:
            available["retrieve_vehicle_market_value"] = scoped["retrieve_vehicle_market_value"]
    definitions = [{"name": name, "description": item.description, "input_schema": item.args_schema.model_json_schema()}
                   for name, item in available.items()]
    analysis_type = ANALYSIS_TYPES[role]
    if bound_wording_id:
        analysis_type = BoundCoverageAnalysis
    elif selected_policy:
        analysis_type = VerifiedCoverageDraft if v4 else SelectedCoverageAnalysis
    definitions.append({"name": "submit_analysis", "description": "Finish investigation and cite evidence returned by executed tools",
                        "input_schema": analysis_type.model_json_schema()})
    if role == "historical":
        history_records = prepared_evidence["retrieve_comparable_claims"]["records"]
        history_ids = {row["id"] for row in history_records if Decimal(row["gross_cost"]) > 0}
        comparable_schema = definitions[-1]["input_schema"]["properties"]["comparable_claim_ids"]
        if history_ids:
            comparable_schema["items"]["enum"] = sorted(history_ids)
        else:
            comparable_schema["maxItems"] = 0
    # Preserve full context for the supervisor/judge, but minimize specialist
    # model inputs to exactly the dependencies relevant to this role.
    view = specialist_context(context, scope_stage)
    if role == "coverage" and not uses_uploaded_wording:
        view = generated_policy_context(view)
    model_input = {"task": task, "required_checks": required_checks}
    submitted_document_refs = {}
    submitted_text_refs = []
    if "read_submission" in prepared_evidence:
        # Extracted upload text stays available on demand. The structured form
        # and derived documents give the model the evidence needed to reason now.
        model_input["submission"] = {key: value for key, value in prepared_evidence["read_submission"].items() if key != "uploaded_evidence"}
        submitted_text_refs = available_text_refs(model_input["submission"])
        sources.update(submitted_text_refs)
        if role == "fraud" and model_input["submission"].get("narrative"):
            sources.add(fraud.NARRATIVE_REF)
        if role == "fraud" and prepared_evidence["read_submission"].get("uploaded_evidence"):
            model_input["submission"]["uploaded_evidence"] = [
                {key: value for key, value in document.items() if key != "pages"}
                if document.get("status") == "extracted" else {"name": document.get("name"), "status": document.get("status")}
                for document in prepared_evidence["read_submission"]["uploaded_evidence"]
            ]
        if role == "coverage":
            submitted_document_refs = {f"submission:/documents/{index}": document["name"]
                                       for index, document in enumerate(model_input["submission"].get("documents", []))
                                       if document.get("name")}
            if submitted_document_refs:
                model_input["submitted_document_refs"] = submitted_document_refs.copy()
            if preflight.details.get("document_text_refs"):
                model_input["submitted_document_text_refs"] = preflight.details["document_text_refs"]
            if uses_uploaded_wording and prepared_evidence["read_submission"].get("uploaded_evidence"):
                uploaded_evidence = prepared_evidence["read_submission"]["uploaded_evidence"]
                model_input["uploaded_documents"] = [
                    {key: document[key] for key in ("source", "name", "status", "document_type") if key in document}
                    for document in uploaded_evidence
                ]
                extracted = [document for document in uploaded_evidence if document.get("status") == "extracted"]
                model_input["uploaded_document_facts"] = [{key: value for key, value in document.items() if key != "pages"} for document in extracted]
                submitted_document_refs.update({document["source"]: document["name"] for document in extracted
                                                if document.get("source") and document.get("name")})
    third_party = role == "coverage" and selected_policy and model_input["submission"].get("loss_type") in THIRD_PARTY_LOSSES
    validation_context = None
    if third_party:
        model_input["claimed_costs"] = claimed_costs(model_input["submission"])
        validation_context = {"line_refs": {line["line_ref"] for line in model_input["claimed_costs"]}}
        definitions[-1]["input_schema"]["required"].extend(
            ("liability_confirmed", "liability_evidence_refs", "liability_evidence_requests", "cost_assessments"))
    model_input["context"] = view
    sources.update(prior_evidence_sources(store, assessment_id, view))
    if v4 and selected_policy:
        model_input["initial_policy_evidence"] = []
        queries = {
            "insuring_clause": "benefit and applicable section",
            "exclusions": "exclusions and conditions of cover",
            "extensions": "endorsements and limits",
            "claim_conditions": "claim obligations and required documents",
            "schedule": "individual schedule registration effective period cover type",
        }
        for topic, query in queries.items():
            check_deadline(store, assessment_id)
            response = scoped["search_policy_documents"].invoke({"topic": topic,
                "query": f"{model_input['submission']['loss_type']} {query}"})
            result = {**response["result"], "passages": response["result"].get("passages", [])[:2]}
            retrieved.append(result)
            model_input["initial_policy_evidence"].append({"evidence_ref": response["evidence_ref"], "result": result})
            sources.update([response["evidence_ref"], *(p["source"] for p in result["passages"])])
        invoked.append("search_policy_documents")
    messages = [{"role": "user", "content": json.dumps(model_input)}]
    # Market benchmark references are citable context, while direct conflicts require corroborated facts.
    presented_fraud_facts = {"submission": model_input.get("submission", {})}
    instruction = specialist_prompt(
        role, model_input.get("submission", {}).get("loss_type"), available,
        uploaded_wording=uses_uploaded_wording, bound_wording=bool(bound_wording_id),
        selected_wording=selected_claim,
        motor_benefit=model_input.get("submission", {}).get("motor_benefit"),
        follow_up=scope_stage == "fraud_followup",
    )
    if role == "coverage" and view.get("policy_schedule"):
        instruction += ("\n\nUse the supplied policy_schedule as the individual's saved contract, including its insured vehicle, "
                        "effective dates, selected benefits, limits and excesses. Cite the governing wording for eligibility conditions, "
                        "requirements and exclusions. Do not treat wording examples as an individual contract.")
    elif v4 and selected_policy:
        instruction += ("\n\nWhen the selected PDF includes a completed individual schedule, return schedule with its "
                        "registration, coverage dates, cover type and exact passage_refs. Leave schedule unset for a blank form, "
                        "illustrative example, or missing details. Retain explicit governing clause citations for each item.")
    repair_analysis = False
    output_retry_used = False
    repair_instruction = ("\n\n## Retry\nSubmit a complete analysis using the evidence already supplied. "
                          "Use a one-sentence summary and brief reasons. Put explanations in the reason or impact fields.")
    for turn in range(4):
        if role == "coverage":
            sources.update(submitted_document_refs)
        if sources:
            definitions[-1]["input_schema"]["properties"]["evidence"]["items"] = {"type": "string", "enum": sorted(sources)}
        force_analysis = turn == 3 or repair_analysis or not available
        try:
            response = model_with_timeout_retry(store, assessment_id, f"r{round_number}:{role}", instruction, messages, definitions,
                {"type": "tool", "name": "submit_analysis"} if force_analysis else {"type": "any"})
        except (ModelOutputTruncated, IncompleteModelResponse) as error:
            if isinstance(error, ModelOutputTruncated):
                if output_retry_used:
                    raise
                output_retry_used = True
                if not repair_analysis:
                    instruction += repair_instruction
                repair_analysis = True
                if role == "coverage":
                    instruction += (" Group items only when their outcome, policy basis and reason are the same. "
                                    "Keep every material exclusion, uncertainty and applicable citation.")
                instruction += " State each reason once and submit the complete findings with their evidence through submit_analysis."
                # Recover on the same turn, including after the last allowed evidence retrieval.
                response = model_with_timeout_retry(store, assessment_id, f"r{round_number}:{role}",
                    instruction, messages, definitions[-1:], {"type": "tool", "name": "submit_analysis"},
                    max_tokens=OUTPUT_RETRY_MAX_TOKENS)
            elif repair_analysis or turn == 3:
                raise
            else:
                repair_analysis = True
                instruction += repair_instruction
                continue
        calls = [block for block in response["content"] if block["type"] == "tool_use"]
        if not calls or len(calls) > 8:
            raise ValueError("Invalid tool call count")
        messages.append({"role": "assistant", "content": response["content"]})
        outputs = []
        for call in calls:
            if call["name"] == "submit_analysis":
                if len(calls) != 1:
                    raise ValueError("Submit analysis separately")
                try:
                    analysis = validate_specialist_analysis(analysis_type, call["input"], audit_stage, turn + 1,
                                                           validation_context=validation_context)
                except ValidationError as error:
                    if not repair_analysis:
                        instruction += repair_instruction
                    repair_analysis = True
                    issues = error.errors(include_input=False, include_context=False, include_url=False)
                    feedback = [{"path": issue["loc"], "code": issue["type"], "message": issue["msg"]} for issue in issues[:6]]
                    messages.append({"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": call["id"], "is_error": True,
                        "content": json.dumps({"message": "Submit corrected structured findings.", "invalid_fields": feedback}),
                    }]})
                    budget = {"max_tokens": OUTPUT_RETRY_MAX_TOKENS} if role == "coverage" or output_retry_used else {}
                    corrected = model_with_timeout_retry(store, assessment_id, f"r{round_number}:{role}", instruction,
                        messages, definitions[-1:], {"type": "tool", "name": "submit_analysis"}, **budget)
                    corrected_calls = [block for block in corrected["content"] if block["type"] == "tool_use"]
                    if len(corrected_calls) != 1 or corrected_calls[0].get("name") != "submit_analysis":
                        raise IncompleteModelResponse("The specialist did not submit a corrected analysis.")
                    analysis = validate_specialist_analysis(analysis_type, corrected_calls[0]["input"], audit_stage, turn + 1,
                                                           validation_context=validation_context)
                invalid_citations = any(ref not in sources for ref in analysis.evidence)
                if invalid_citations and role != "historical":
                    if not v4 and role != "fraud":
                        raise ValueError("Citation was not retrieved")
                    if repair_analysis or turn == 3:
                        audited = copy.deepcopy(checked)
                        audited["details"]["unverified_analysis"] = "The model cited evidence outside the supplied sources."
                        if role == "coverage":
                            audited["details"]["coverage_status"] = "UNRESOLVED"
                        elif role == "fraud":
                            audited["summary"] = fraud.screening_summary(audited["details"], [])
                        if audited["status"] != "fraud":
                            audited["status"] = "review"
                        return audited
                    repair_analysis = True
                    outputs.append({"type": "tool_result", "tool_use_id": call["id"], "is_error": True,
                                    "content": "Resubmit using only the exact citation IDs listed in the tool schema."})
                    break
                assessed = Finding.model_validate(checked)
                reasons = analysis.review_reasons[:]
                if invalid_citations:
                    reasons = ["Some Historical citations could not be verified against the retrieved evidence."]
                if isinstance(analysis, CoverageAnalysis):
                    submitted_evidence_refs = {**submitted_document_refs, **{
                        ref: ref.rsplit("/", 1)[-1].replace("_", " ").capitalize() for ref in submitted_text_refs}}
                    conclusion = analysis.coverage_conclusion
                    unconfirmed_reason = None
                    reasons = [reason for reason in reasons if not coverage.CHASSIS_IDENTIFIER.search(reason)]
                    document_inconsistencies = []
                    for issue in analysis.document_inconsistencies:
                        if coverage.CHASSIS_IDENTIFIER.search(issue.reason):
                            continue
                        if all(ref in submitted_evidence_refs for ref in issue.document_refs):
                            document_inconsistencies.append(issue.model_dump())
                        else:
                            reasons.append("A reported document inconsistency could not be traced to the submitted evidence.")
                    if document_inconsistencies and conclusion == "COVERED":
                        conclusion = "UNRESOLVED"
                    submission = prepared_evidence["read_submission"]
                    declared = submission.get("loss_type")
                    benefit_class = {"glass_waiver": "glass", "motor_medical": "injury"}.get(submission.get("motor_benefit"))
                    if (declared == "repair" and benefit_class and analysis.classified_loss == benefit_class
                            and (schedule := view.get("policy_schedule")) and scheduled_financials(submission, schedule)):
                        analysis = analysis.model_copy(update={"classified_loss": declared})
                    if analysis.classified_loss and analysis.classified_loss != declared:
                        conclusion = "UNRESOLVED"
                        unconfirmed_reason = "The loss type inferred from the evidence is uncertain or differs from the declared loss type."
                        reasons.append(unconfirmed_reason)
                    model_coverage_items = []
                    for item in analysis.coverage_items:
                        if any(ref not in submitted_evidence_refs for ref in item.document_refs):
                            reasons.append(f"The submitted documents cited for {item.item} could not be verified.")
                            continue
                        model_coverage_items.append(item)
                    if isinstance(analysis, BoundCoverageAnalysis):
                        model_coverage_items, conflicts, wording, wording_reasons = assess_bound_wording(
                            analysis.model_copy(update={"coverage_items": model_coverage_items}), retrieved, bound_wording_id)
                        reasons.extend(wording_reasons)
                        assessed.details.update(policy_rag=wording, policy_conflicts=conflicts)
                    if isinstance(analysis, SelectedCoverageAnalysis):
                        model_coverage_items, conclusion, wording, gaps = assess_selected_wording(
                            analysis, model_coverage_items, retrieved, selected_policy, prepared_evidence["read_submission"],
                            conclusion, assessed.details["deferred_cost_gaps"])
                        assessed.details.update(policy_rag=wording, missing=gaps)
                    coverage_items, omissions = coverage.separate_policy_omissions(
                        [item.model_dump() for item in model_coverage_items],
                        [item.model_dump() for item in analysis.policy_omissions])
                    required_outcome = {"COVERED": "covered", "NOT_COVERED": "not_covered"}.get(conclusion)
                    if required_outcome and not any(item["outcome"] == required_outcome for item in coverage_items):
                        conclusion = "UNRESOLVED"
                    liability_refs = submitted_document_refs.keys() | preflight.details.get("document_text_refs", {}).values()
                    liability_sources = {ref for item in coverage_items if item["outcome"] == "covered"
                                         for ref in item["document_refs"] if ref in liability_refs}
                    if third_party:
                        documents = model_input["submission"].get("documents", [])
                        demand_refs = {f"submission:/documents/{index}" for index, doc in enumerate(documents)
                                       if doc.get("type") == "third_party_demand"}
                        demand_text = {text for ref, text in preflight.details.get("document_text_refs", {}).items() if ref in demand_refs}
                        amount_sources = (submitted_document_refs.keys() | set(submitted_text_refs)) - demand_refs - demand_text
                        explicit_refs = set(analysis.liability_evidence_refs)
                        liability_sources = explicit_refs if explicit_refs.issubset(liability_refs - demand_refs - demand_text) else set()
                    liability_supported = analysis.liability_confirmed and bool(liability_sources)
                    if third_party:
                        assessed.details["third_party_assessment"] = assess_third_party(
                            analysis, model_input["claimed_costs"], coverage_items, amount_sources,
                            liability_sources if liability_supported else (),
                            policy_applies=bool(wording["wording_passages"]) and wording["applicability"] == "applicable"
                            and selected_policy.get("document_kind") != "product_summary", claim=model_input["submission"])
                    if declared in THIRD_PARTY_LOSSES and conclusion == "COVERED" and not liability_supported:
                        conclusion = "UNRESOLVED"
                        unconfirmed_reason = "The cited claim documents did not establish who was responsible for the third-party loss."
                        reasons.append(unconfirmed_reason)
                    if selected_policy:
                        assessed.details["coverage_status"] = conclusion
                        assessed.status = "information" if conclusion == "NEEDS_INFORMATION" else "review"
                        if selected_policy.get("document_kind") == "product_summary":
                            assessed.summary += " The product summary is preliminary. Obtain the full wording and applicable schedule for contract assessment."
                        elif conclusion == "COVERED":
                            assessed.summary += " The wording supports coverage based on the reported facts. Staff must confirm the policy schedule and payable amount."
                        elif wording["wording_passages"]:
                            assessed.summary += " Policy passages were found. Their applicability to this claim requires staff verification."
                        else:
                            assessed.summary += " Coverage did not establish a policy term for this loss."
                    else:
                        coverage.apply_approval_controls(assessed, prepared_evidence["read_submission"],
                                                         prepared_evidence["retrieve_policy"], conclusion)
                    if isinstance(analysis, BoundCoverageAnalysis) and wording_reasons:
                        if assessed.details["coverage_status"] in {"COVERED", "NOT_COVERED"} and not assessed.details.get("coverage_reason"):
                            assessed.details["coverage_status"] = "UNRESOLVED"
                        assessed.details.setdefault("coverage_reason", wording_reasons[0])
                        assessed.summary += " " + " ".join(wording_reasons)
                elif isinstance(analysis, HistoricalAnalysis):
                    comparable_ids = [identifier for identifier in analysis.comparable_claim_ids if identifier in history_ids]
                    if len(comparable_ids) != len(analysis.comparable_claim_ids):
                        reasons.append("Some cost comparison references did not match the retrieved claims.")
                    conclusion = analysis.amount_conclusion
                    if reasons and conclusion == "SUPPORTED":
                        conclusion = "REQUIRES_REVIEW"
                    historical.apply_comparison(assessed, prepared_evidence["read_submission"], history_records,
                                                conclusion, comparable_ids=comparable_ids,
                                                review_assessment=analysis.summary if not invalid_citations and len(comparable_ids) == len(analysis.comparable_claim_ids) else None)
                finding = assessed.model_dump(mode="json")
                if not bound_wording_id and not selected_policy and role != "historical":
                    finding["summary"] += " Claude: " + analysis.summary
                finding["evidence"] = analysis.evidence
                if isinstance(analysis, (BoundCoverageAnalysis, SelectedCoverageAnalysis)):
                    finding["evidence"] = list(dict.fromkeys([*analysis.evidence, *wording.get("wording_passages", wording["applicable_passages"])]))
                if market_ref and market_ref not in finding["evidence"]:
                    finding["evidence"].append(market_ref)
                finding["details"]["analysis"] = analysis.model_dump(mode="json")
                if isinstance(analysis, HistoricalAnalysis):
                    amount = finding["details"]["amount_justification"]
                    finding["details"]["analysis"].update(amount_conclusion=amount["conclusion"],
                                                            comparable_claim_ids=amount["comparable_claim_ids"])
                    if finding["details"].get("review_history") and check_ref not in finding["evidence"]:
                        finding["evidence"].append(check_ref)
                    if invalid_citations:
                        finding["evidence"] = list(dict.fromkeys([check_ref, *(ref for ref in analysis.evidence if ref in sources)]))
                        finding["details"]["analysis"].update(evidence=finding["evidence"], summary=finding["summary"], review_reasons=[])
                if isinstance(analysis, CoverageAnalysis):
                    finding["details"]["analysis"].update(
                        coverage_conclusion=finding["details"]["coverage_status"],
                        coverage_items=coverage_items, policy_omissions=omissions,
                        document_inconsistencies=document_inconsistencies)
                if isinstance(analysis, BoundCoverageAnalysis):
                    finding["details"]["analysis"].update(applicable_passages=wording["applicable_passages"], policy_conflicts=conflicts)
                if isinstance(analysis, SelectedCoverageAnalysis):
                    for field in ("required_documents", "liability_evidence_refs", "liability_evidence_requests", "cost_assessments"):
                        finding["details"]["analysis"].pop(field, None)
                    if third_party:
                        finding["details"]["analysis"]["liability_confirmed"] = liability_supported
                    if isinstance(analysis, VerifiedCoverageDraft) and analysis.schedule is not None:
                        finding["details"]["schedule_candidate"] = analysis.schedule.model_dump(mode="json")
                finding["details"]["sources"] = sorted(sources)
                finding["details"]["tools_used"] = invoked
                if isinstance(analysis, FraudAnalysis):
                    cited_concerns = [concern.model_dump() for concern in analysis.fraud_concerns
                                      if sources.issuperset(concern.evidence)]
                    concerns = fraud.supported_concerns(cited_concerns, presented_fraud_facts,
                                                        fictional_policy_reference=selected_claim)
                    plausibility = fraud.plausibility_assessment(
                        analysis.plausibility.model_dump() if analysis.plausibility else None, concerns,
                        presented_fraud_facts, analysis.evidence)
                    finding["details"].update(fraud_concerns=concerns, fraud_concerns_validated=True, plausibility=plausibility)
                    finding["summary"] = fraud.screening_summary(finding["details"], concerns)
                    finding["details"]["analysis"].update(fraud_concerns=concerns, plausibility=plausibility,
                                                          summary=finding["summary"], review_reasons=[])
                    reasons = []
                    if concerns and finding["status"] != "fraud":
                        finding["status"] = "fraud" if any(concern["severity"] == "high" for concern in concerns) else "review"
                if uses_uploaded_wording:
                    chunks = [p for result in retrieved for p in result.get("passages", []) if p["source"] in analysis.evidence]
                    decisive = any(p.get("document_kind") != "product_summary" for p in chunks)
                    verify_wording = "The cited product summary could not establish applicable policy wording."
                    finding["details"]["policy_rag"] = {"document_id": context.get("wording_document_id"),
                        "corpus_ids": context.get("policy_corpus_ids"), "applicability": analysis.policy_applicability,
                        "cited": decisive, "retrievals": retrieved}
                    if not chunks:
                        reasons.append("Applicable policy wording was not retrieved and cited")
                    elif not decisive:
                        reasons.append(verify_wording)
                    if analysis.policy_applicability != "applicable":
                        reasons.append("Retrieved wording does not clearly apply to this claim")
                    if finding["details"]["coverage_status"] == "COVERED" and (
                            not finding["details"]["policy_rag"]["cited"] or analysis.policy_applicability != "applicable"):
                        finding["details"]["coverage_status"] = "UNRESOLVED"
                        finding["details"]["coverage_reason"] = "No cited wording clearly applies to this policy and loss."
                        if "financial" in finding["details"]:
                            finding["details"]["financial"]["payable"] = None
                    elif chunks and not decisive and finding["details"]["coverage_status"] == "NOT_COVERED" and not finding["details"].get("coverage_reason"):
                        finding["details"].update(coverage_status="UNRESOLVED", coverage_reason=verify_wording)
                if isinstance(analysis, CoverageAnalysis):
                    details = finding["details"]
                    status = details["coverage_status"]
                    if status == "COVERED" and market_check.get("status") == "unlikely_to_limit":
                        reasons = [reason for reason in reasons if not market_cap_only_review(reason)]
                        coverage_items = [item for item in coverage_items if item["outcome"] != "undetermined"
                                          or not market_cap_only_review(f"{item['item']} {item['reason']}")]
                        details["analysis"]["coverage_items"] = coverage_items
                        details["analysis"]["review_reasons"] = reasons[:]
                    checked_items = details.get("coverage_items", [])
                    allowed_outcomes = {
                        "COVERED": {"covered", "not_covered", "undetermined"},
                        "NEEDS_INFORMATION": {"covered", "not_covered", "undetermined"},
                        "NOT_COVERED": {"not_covered"},
                        "UNRESOLVED": {"covered", "not_covered", "undetermined"},
                    }.get(status, set())
                    wording = details.get("policy_rag", {})
                    if details.get("coverage_reason") or (uses_uploaded_wording and (
                            not wording.get("cited") or wording.get("applicability") != "applicable")):
                        allowed_outcomes = set()
                    if (checked_items or document_inconsistencies or unconfirmed_reason
                            or (declared in THIRD_PARTY_LOSSES and not liability_supported)):
                        allowed_outcomes.discard("covered")
                    model_items = [item for item in coverage_items if item["outcome"] in allowed_outcomes]
                    details["coverage_items"] = checked_items + model_items
                    if (declared == "third_party_injury" and liability_supported
                            and any(item["outcome"] == "covered" for item in model_items)
                            and (gaps := missing_income_loss_evidence(prepared_evidence["read_submission"], model_items))):
                        status = "NEEDS_INFORMATION"
                        details["coverage_status"] = details["analysis"]["coverage_conclusion"] = status
                        details["missing"].extend(gaps)
                        if "financial" in details:
                            details["financial"]["payable"] = None
                        finding["status"] = "information"
                        finding["summary"] += " The claimed lost income needs supporting evidence."
                    details["document_inconsistencies"] = document_inconsistencies
                    if status == "UNRESOLVED" and unconfirmed_reason:
                        details.setdefault("coverage_reason", unconfirmed_reason)
                    cited_documents = set(analysis.evidence).intersection(submitted_evidence_refs)
                    cited_documents.update(ref for item in [*model_items, *document_inconsistencies]
                                           for ref in item["document_refs"] if ref in submitted_evidence_refs)
                    if cited_documents:
                        details["submitted_documents"] = {ref: submitted_evidence_refs[ref] for ref in sorted(cited_documents)}
                    if bound_wording_id and model_items:
                        finding["summary"] += " " + " ".join(item["reason"] for item in model_items[:3])
                    if document_inconsistencies:
                        finding["summary"] += " " + " ".join(issue["reason"] for issue in document_inconsistencies[:3])
                    omissions = omissions if allowed_outcomes else []
                    details["policy_omissions"] = omissions
                    history = view["findings"].get("historical", {}).get("details", {})
                    comparable_cases = {case["claim_id"]: case["claim_reference"] for case in history.get("review_history", [])}
                    comparison = analysis.historical_context
                    if (comparison and comparable_cases and not invalid_citations
                            and set(comparison.claim_ids).issubset(comparable_cases)):
                        details["coverage_historical_context"] = {
                            "reason": comparison.reason,
                            "claim_references": list(dict.fromkeys(comparable_cases[identifier] for identifier in comparison.claim_ids)),
                        }
                    if status == "COVERED" and (omissions or not details["coverage_items"] or
                            any(item["outcome"] != "covered" for item in details["coverage_items"])):
                        if finding["status"] == "pass":
                            finding["status"] = "review"
                        if "financial" in details:
                            details["financial"]["payable"] = None
                        if omissions:
                            finding["summary"] += " Material questions remain unaddressed by the available policy terms."
                if reasons:
                    if finding["status"] not in {"fraud", "information"}:
                        finding["status"] = "review"
                    if role == "coverage" and "financial" in finding["details"]:
                        finding["details"]["financial"]["payable"] = None
                    finding["details"].setdefault("review_reasons", []).extend(reasons)
                return finding
            if call["name"] not in available:
                with trace_span("tool.rejected", kind="guardrail", metadata={"stage": audit_stage}, content=call):
                    raise ValueError("Forbidden specialist tool")
            check_deadline(store, assessment_id)
            result = available[call["name"]].invoke(call["input"])
            if call["name"] in {"search_policy_documents", "read_policy_page"}:
                retrieved.append(result["result"])
            elif call["name"] != "retrieve_vehicle_market_value":
                result = record_evidence(store, assessment_id, audit_stage, call["name"], call["input"], result)
            sources.add(result["evidence_ref"])
            if role == "fraud":
                presented_fraud_facts[result["evidence_ref"]] = result["result"]
            if role == "coverage" and call["name"] == "read_uploaded_documents":
                submitted_document_refs.update({document["source"]: document["name"]
                                                for document in result["result"]["evidence"]
                                                if document.get("status") == "extracted" and document.get("source") and document.get("name")})
            for passage in result["result"].get("passages", []):
                sources.add(passage["source"])
            invoked.append(call["name"])
            outputs.append({"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(result)})
        messages.append({"role": "user", "content": outputs})
    raise ValueError("Specialist turn budget exceeded")


def coverage_without_provisional_payable(finding):
    """Keep claim costs and policy terms visible without uncommitted payable assertions."""
    finding = copy.deepcopy(finding)
    details = finding["details"]
    narrative = [(finding, "summary"), (details.get("coverage_outcome") or {}, "reason")]
    analysis = details.get("analysis") or {}
    narrative.append((analysis, "summary"))
    narrative.extend((item, "reason") for view in (details, analysis) for item in view.get("coverage_items", []))
    for record, key in narrative:
        if isinstance(record.get(key), str):
            record[key] = omit_provisional_payable(record[key])
    return finding


def judge(store, assessment_id, context):
    sources = {source for finding in context["findings"].values() for source in finding["details"].get("sources", [])}
    with store.session() as session:
        assessment = session.get(Assessment, assessment_id)
        claim = session.get(Claim, assessment.claim_id)
    if claim.payload.get("source") == "Synthetic generator":
        context = generated_policy_context(context)
    candidates = adjuster_review_candidates(claim.payload, context["findings"], context.get("policy_schedule"))
    coverage_finding = context["findings"].get("coverage", {})
    coverage_details = coverage_finding.get("details", {})
    payable = (coverage_details.get("financial") or {}).get("payable")
    judge_context = {**context, "claim": claim.payload, "publication_controls": {
        "gross_authority_LKR": "250000.00", "gross_exceeds_authority": Decimal(claim.payload["gross_amount"]) > Decimal("250000.00"),
        "committed_coverage_payable_LKR": payable,
        "workflow_requires_staff_review_for_complete_claims": (
            claim.payload.get("source") == "Manual submission" or context.get("workflow") == V4)}}
    if coverage_details and payable is None:
        judge_context["findings"] = {**context["findings"], "coverage": coverage_without_provisional_payable(coverage_finding)}
    judge_context["adjuster_candidates"] = candidates
    judge_context["committed_evidence_requests"] = list(committed_required_gaps(context["findings"]))
    has_coverage_handoff = "third_party_assessment" in coverage_details
    judge_context["use_committed_evidence_requests"] = has_coverage_handoff
    result = structured_call(store, assessment_id, "judge", ROUTE_RECOMMENDATION + "\n\n" + ADJUSTER_PRIORITY_INSTRUCTIONS,
                             judge_context, Recommendation, "recommend_route", sources)
    if any(source not in sources for source in result.evidence):
        raise ValueError("Judge cited unavailable evidence")
    recommendation = result.model_dump(mode="json", exclude={"adjuster_priorities", "evidence_requests"})
    if not has_coverage_handoff:
        if requests := [request.model_dump(mode="json") for request in result.evidence_requests
                        if request.finding_quote in candidates.get(request.candidate_id, "")]:
            recommendation["evidence_requests"] = requests
    if candidates:
        recommendation["adjuster_summary"] = select_adjuster_summary(result.adjuster_priorities, candidates)
    return recommendation
