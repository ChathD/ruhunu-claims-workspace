"""Select the Claude or offline implementation for specialist checks, readiness and recommendations."""
from backend.storage.sqlite_store import Assessment, Claim
from backend.agents import claude_investigation as agentic
from backend.assurance.investigation_tracing import trace_span
from backend.workflow import specialist_readiness as readiness
from backend.agents.specialist_preflight import apply_verified_controls, execute_checks, finish_offline_checks
from backend.evidence.claim_history import NO_HISTORY
from backend.evidence.policy_schedule import apply_matched_schedule, verify_scheduled_repair
from backend.evidence.specialist_evidence_tools import build_tools
from backend.workflow.versions import V3


def offline(store, identifier):
    with store.session() as session:
        return session.get(Assessment, identifier).execution.get("mode") == "deterministic-tools"


def execute_specialist(store, identifier, context, round_number, role, task):
    if role == "coverage" and readiness.coverage_blockers(context):
        raise ValueError("Intake has not confirmed enough evidence for Coverage")
    if offline(store, identifier):
        finding = offline_specialist(store, identifier, context, round_number, role)
    else:
        finding = agentic.run_specialist(store, identifier, context, round_number, role, task)
    schedule = context.get("policy_schedule")
    if role == "coverage" and schedule and schedule.get("id"):
        with store.session() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id).payload
        apply_matched_schedule(finding, claim, schedule)
        if context.get("workflow") == V3:
            verify_scheduled_repair(finding, claim, schedule)
    return finding


def offline_specialist(store, identifier, context, round_number, role):
    follow_up = role == "fraud" and readiness.has_new_leads(context)
    stage = "fraud_followup" if follow_up else "fraud_initial" if role == "fraud" else role
    evidence = {}
    def capture(name, request, result):
        evidence[name] = result
    tools = build_tools(store, identifier, stage, context_snapshot=context,
                        audit_stage=f"r{round_number}:{role}", on_result=capture)
    with trace_span("checks.execute", kind="tool", metadata={"stage": f"r{round_number}:{role}"}) as span:
        result = execute_checks(role, tools, follow_up)
        apply_verified_controls(role, result, tools, evidence, context.get("policy_schedule"))
        finish_offline_checks(role, result, evidence)
        result.details["checks_executed"] = True
        result.details["execution_mode"] = "offline checks; no model called"
        span.output = result.model_dump(mode="json")
        span.summary = {"status": result.status}
        return span.output


def offline_readiness(context, claim):
    """Build a dispatch plan from the recorded checks without a model call."""
    states, tasks = {}, []
    source = sorted(readiness.sources_for(context, claim))[0]
    for role in readiness.ROLES:
        state, gaps = "READY", []
        if readiness.completed(context, role) and not (role == "fraud" and readiness.has_new_leads(context)):
            state = "COMPLETED"
        elif role == "historical" and context.get("historical_available") is False:
            state = "NOT_APPLICABLE"
        elif role == "historical" and not readiness.historical_due(context):
            state = "BLOCKED_DEPENDENCY"
        elif role == "coverage" and (blockers := readiness.coverage_blockers(context)):
            state = "BLOCKED_INPUT"
            gaps = blockers
        elif role == "coverage" and readiness.historical_due(context):
            state = "BLOCKED_DEPENDENCY"
        reason = NO_HISTORY if role == "historical" and context.get("historical_available") is False else f"Recorded checks: {state}"
        if role == "coverage" and gaps:
            reason = "Intake found gaps that prevent Coverage from starting."
        elif role == "coverage" and state == "BLOCKED_DEPENDENCY":
            reason = "Coverage will consider the committed Historical comparison in the next round."
        states[role] = {"state": state, "reason": reason, "evidence_references": [source], "missing_information": gaps}
        if state == "READY":
            tasks.append({"specialist": role, "task": "Run approved checks and preserve evidence references."})
    all_gaps = [gap for state in states.values() for gap in state["missing_information"]]
    action = "FINISH"
    if tasks:
        action = "DISPATCH"
    elif any(gap["owner"] == "internal" for gap in all_gaps):
        action = "REVIEW"
    elif all_gaps:
        action = "REQUEST_INFORMATION"
    return {"specialists": states, "tasks": tasks, "next_action": action, "explanation": "Dispatch based on the recorded checks."}


def assess_readiness(store, identifier, context, claim, round_number):
    if offline(store, identifier):
        return offline_readiness(context, claim)
    if readiness.completed(context, "fraud"):
        recorded_plan = offline_readiness(context, claim)
        if not recorded_plan["tasks"]:
            return recorded_plan
    return readiness.assess(store, identifier, context, claim, round_number)


def recommend_route(store, identifier, context):
    if offline(store, identifier):
        return {"route": "auto-approve", "explanation": "Offline candidate; Python publication must validate every provisional check.",
                "evidence": [], "outstanding_gaps": []}
    return agentic.judge(store, identifier, context)
