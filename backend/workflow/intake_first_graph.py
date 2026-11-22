"""Define the LangGraph intake barrier, specialist dispatch, context collection, judge and publication."""
import copy
import logging
import time
import uuid
from typing import Annotated, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from sqlalchemy import select

from backend.agents.fraud_specialist import high_severity_concerns
from backend.assurance.investigation_tracing import observe_node, trace_span
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, ToolCall, WorkflowEvent
from backend.claim_schemas import Finding
from backend.workflow import specialist_readiness as readiness
from backend.workflow.investigation_execution import execute_specialist, assess_readiness, recommend_route
from backend.evidence.approved_claim_history import NO_HISTORY
from backend.workflow.investigation_retry import failure_reason

logger = logging.getLogger(__name__)


def merge_results(left, right):
    return {**left, **right}


class IntakeFirstState(TypedDict):
    assessment_id: str
    round: int
    tasks: dict
    snapshot: dict
    results: Annotated[dict, merge_results]
    phase: str
    readiness: dict


class IntakeFirstWorkflow:
    def __init__(self, owner):
        self.owner, self.store = owner, owner.store
        graph = StateGraph(IntakeFirstState)
        graph.add_node("mandatory_intake", self.intake)
        graph.add_node("supervisor", self.supervise)
        for role in readiness.ROLES:
            graph.add_node(role, self.specialist(role))
            graph.add_edge(role, "collect")
        graph.add_node("collect", self.collect)
        graph.add_node("judge", self.judge)
        graph.add_node("publish", self.publish)
        graph.add_edge(START, "mandatory_intake")
        graph.add_conditional_edges("mandatory_intake", self.after_intake, ["supervisor", "judge"])
        graph.add_conditional_edges("supervisor", lambda state: list(state["tasks"]) or ["judge"], [*readiness.ROLES, "judge"])
        graph.add_conditional_edges("collect", self.after_collect, ["supervisor", "judge"])
        graph.add_edge("judge", "publish")
        graph.add_edge("publish", END)
        self.graph = graph.compile(checkpointer=owner.checkpointer)

    @observe_node("intake")
    def intake(self, state):
        snapshot = copy.deepcopy(self.owner.state(state["assessment_id"]).context)
        intake_state = {**state, "round": 0, "snapshot": snapshot,
                        "tasks": {"intake": "Check whether the submitted evidence contains the information required for Coverage."}}
        result = self.specialist("intake")(intake_state)
        committed = self.collect({**intake_state, "results": result["results"]})
        with self.store.session.begin() as session:
            previous = session.scalar(select(WorkflowEvent.id).where(WorkflowEvent.assessment_id == state["assessment_id"], WorkflowEvent.kind == "intake_completed"))
            if previous is None:
                self.store.add_event(session, state["assessment_id"], "intake_completed", {"context_version": committed["snapshot"]["version"]})
        return {"round": 0, "snapshot": committed["snapshot"], "results": result["results"], "phase": "investigating", "tasks": {}}

    @observe_node("gate.after_intake", kind="guardrail")
    def after_intake(self, state):
        details = state["snapshot"]["findings"].get("intake", {}).get("details", {})
        if details.get("fraud_indicators"):
            return "judge"
        if not details.get("checks_executed") or details.get("error_type"):
            self.save_failure(state["assessment_id"], "Intake did not complete required checks")
            return "judge"
        return "supervisor"

    def save_failure(self, identifier, reason):
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            row.context = {**row.context, "investigation_error": reason}

    @observe_node("readiness", kind="agent")
    def supervise(self, state):
        identifier = state["assessment_id"]
        round_number = state["round"] + 1
        context = copy.deepcopy(self.owner.state(identifier).context)
        history = context.setdefault("readiness_history", {})
        raw = history.get(str(round_number))
        try:
            with self.store.session() as session:
                claim_row = session.get(Claim, session.get(Assessment, identifier).claim_id)
                claim = claim_row.payload
                coverage = context["findings"].get("coverage", {}).get("details", {})
                if (coverage.get("coverage_status") == "COVERED" and "historical_available" not in context
                        and not readiness.completed(context, "historical")):
                    with trace_span("history.eligibility", kind="retriever", metadata={"round": round_number}) as span:
                        context["historical_available"] = bool(self.store.history.records(session, claim_row))
                        span.summary = {"available": context["historical_available"]}
                    if not context["historical_available"]:
                        context.setdefault("skipped", {})["historical"] = NO_HISTORY
            if raw is None:
                raw = assess_readiness(self.store, identifier, context, claim, round_number)
            plan = readiness.validate_selection(raw, context, claim)
            history[str(round_number)] = plan
            context.setdefault("investigation_plans", {})[str(round_number)] = {"tasks": plan["tasks"], "reasoning": plan["explanation"]}
            if plan["next_action"] == "REVIEW":
                context["readiness_review"] = plan["explanation"]
            context["readiness_gaps"] = [gap for item in plan["specialists"].values() for gap in item["missing_information"]]
            if plan["specialists"]["coverage"]["state"] == "BLOCKED_INPUT" and readiness.coverage_blockers(context):
                context["skipped"]["coverage"] = "Intake found gaps that prevent Coverage from starting."
        except Exception as error:
            context["investigation_error"] = f"Readiness validation failed: {type(error).__name__}"
            history[str(round_number)] = {"rejected_selection": raw, "error": str(error)}
            if isinstance(error, readiness.UnsupportedClaimantGaps):
                history[str(round_number)]["unsupported_claimant_gaps"] = error.gaps
            plan = {"tasks": [], "explanation": "Invalid or unavailable readiness assessment; reprocessing required"}
        with self.store.session.begin() as session:
            session.get(Assessment, identifier).context = context
            self.store.add_event(session, identifier, "readiness_assessed", {"round": round_number, **history[str(round_number)]})
            if plan["tasks"]:
                self.store.add_event(session, identifier, "specialists_selected", {"round": round_number, **plan})
        return {"round": round_number, "snapshot": context, "readiness": history[str(round_number)],
                "tasks": {task["specialist"]: task["task"] for task in plan["tasks"]}}

    @observe_node("gate.after_collect", kind="guardrail")
    def after_collect(self, state):
        context = state["snapshot"]
        findings = context["findings"]
        if high_severity_concerns(findings):
            return "judge"
        coverage = findings.get("coverage", {}).get("details", {}).get("coverage_status")
        if coverage in {"INACTIVE", "NOT_COVERED", "UNRESOLVED"}:
            with self.store.session.begin() as session:
                row = session.get(Assessment, state["assessment_id"])
                row.context = {**row.context, "skipped": {**row.context.get("skipped", {}), "historical": f"Coverage gate: {coverage}"}}
                self.store.add_event(session, row.id, "specialist_skipped", {"stage": "historical", "reason": coverage})
            return "judge"
        if any(f["details"].get("fraud_indicators") or f["details"].get("error_type") for f in findings.values()):
            return "judge"
        if state["round"] >= readiness.MAX_ROUNDS:
            eligible_roles = ("fraud",) if coverage is None and readiness.coverage_blockers(context) else readiness.ROLES
            needed = any(not readiness.completed(context, role) for role in eligible_roles
                         if role != "historical" or context.get("historical_available") is not False) or readiness.has_new_leads(context)
            if needed:
                self.save_failure(state["assessment_id"], "Specialist dispatch round budget exhausted")
            return "judge"
        return "supervisor"

    @observe_node("judge", kind="agent")
    def judge(self, state):
        if failure_reason(self.owner.state(state["assessment_id"]).context):
            return {"phase": "interrupted"}
        return {**self._judge(state), "phase": "judging"}

    def publish(self, state):
        with trace_span("publication", kind="guardrail", metadata={"round": state["round"]}) as span:
            decision = self.owner.publish(state["assessment_id"])
            span.output = decision
            if decision:
                span.summary = {"route": decision["route"], "reason_codes": decision["reason_codes"]}
            else:
                span.summary = {"status": "interrupted"}
                span.error_type = "InvestigationInterrupted"
            return {"phase": "published" if decision else "interrupted"}

    def run(self, assessment_id):
        with self.store.session.begin() as session:
            row = session.get(Assessment, assessment_id)
            row.status, row.error = "processing", None
        config = {"configurable": {"thread_id": "intake-first:" + assessment_id}, "recursion_limit": 40}
        saved = self.graph.get_state(config)
        initial = None if saved.next else {"assessment_id": assessment_id, "phase": "intake", "round": 0,
                                           "tasks": {}, "snapshot": {}, "results": {}, "readiness": {}}
        self.graph.invoke(initial, config, durability="sync")

    def specialist(self, role):
        @observe_node(f"specialist.{role}", kind="agent", role=role)
        def run(state):
            stage = f"r{state['round']}:{role}"
            with self.store.session() as session:
                existing = session.scalar(select(AgentRun).where(AgentRun.assessment_id == state["assessment_id"], AgentRun.stage == stage))
            if existing:
                return {"results": {stage: {"finding": existing.finding, "duration_ms": existing.duration_ms}}}
            started = time.perf_counter()
            try:
                finding = execute_specialist(self.store, state["assessment_id"], state["snapshot"], state["round"], role, state["tasks"][role])
                Finding.model_validate(finding)
            except Exception as error:
                logger.exception("Specialist failed for assessment %s at %s", state["assessment_id"], stage)
                # Retain executed authoritative results even if the subsequent model call failed.
                with self.store.session() as session:
                    check = session.scalar(select(ToolCall).where(ToolCall.assessment_id == state["assessment_id"],
                        ToolCall.stage == stage, ToolCall.tool == "execute_checks").order_by(ToolCall.id.desc()))
                finding = copy.deepcopy(check.response["result"]) if check else Finding(agent="fraud_initial" if role == "fraud" else role,
                    status="review", summary="Investigation could not complete", evidence=[], details={}).model_dump(mode="json")
                if finding["status"] != "fraud":
                    finding["status"] = "review"
                finding["details"]["error_type"] = type(error).__name__
                finding["details"]["error_message"] = str(error)[:500]
                finding["details"]["checks_executed"] = bool(check)
                reason = "Specialist execution failed. Reprocessing is required."
                if role == "coverage" and finding["details"].get("coverage_status") in {None, "PENDING_ANALYSIS"}:
                    finding["details"]["coverage_status"] = "UNRESOLVED"
                    if isinstance(error, httpx.TimeoutException):
                        reason = "The coverage model timed out. Reprocess the claim to complete the assessment."
                        finding["details"]["coverage_reason"] = reason
                finding["summary"] += " " + reason
            return {"results": {stage: {"finding": finding, "duration_ms": round((time.perf_counter() - started) * 1000)}}}
        return run

    @observe_node("context.collect")
    def collect(self, state):
        identifier, round_number = state["assessment_id"], state["round"]
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            context = copy.deepcopy(row.context)
            if round_number in context.get("collected_rounds", []):
                return {"snapshot": context}
            before = copy.deepcopy(context)
            version = row.version + 1
            for role in state["tasks"]:
                stage = f"r{round_number}:{role}"
                result = state["results"][stage]
                finding = copy.deepcopy(result["finding"])
                key = finding["agent"]
                prior = context["findings"].get(key)
                if prior:
                    prior_details = prior["details"]
                    if prior_details.get("fraud_indicators"):
                        finding["details"]["fraud_indicators"] = prior_details["fraud_indicators"] + finding["details"].get("fraud_indicators", [])
                        finding["status"] = "fraud"
                    elif prior["status"] == "review":
                        finding["status"] = "review"
                        finding["details"].setdefault("review_reasons", []).append("Earlier round requires review: " + prior["summary"])
                context["findings"][key] = finding
                session.add(AgentRun(id=str(uuid.uuid4()), assessment_id=identifier, stage=stage,
                    instruction=state["tasks"][role], input_context=before, input_version=row.version,
                    finding=finding, context_after={}, duration_ms=result["duration_ms"]))
            context.setdefault("collected_rounds", []).append(round_number)
            context["version"] = version
            session.flush()
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier, AgentRun.stage.like(f"r{round_number}:%"))).all()
            for run in runs:
                run.context_after = context
            row.context, row.version = context, version
            self.store.add_event(session, identifier, "investigation_round_completed", {"round": round_number, "specialists": list(state["tasks"]), "context_version": version})
        return {"snapshot": context}

    def _judge(self, state):
        context = copy.deepcopy(self.owner.state(state["assessment_id"]).context)
        if "recommendation" not in context:
            try:
                context["recommendation"] = recommend_route(self.store, state["assessment_id"], context)
            except Exception as error:
                logger.exception("Judge failed for assessment %s", state["assessment_id"])
                context.setdefault("investigation_error", type(error).__name__)
                context["judge_error"] = {"error_type": type(error).__name__, "error_message": str(error)[:500]}
            with self.store.session.begin() as session:
                row = session.get(Assessment, state["assessment_id"])
                row.context = context
                if "recommendation" in context:
                    self.store.add_event(session, row.id, "route_recommended", context["recommendation"])
        return {"snapshot": context}
