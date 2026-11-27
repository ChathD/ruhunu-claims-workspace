"""Ask Claude which specialists can run and validate assignments against saved evidence and dependencies."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents import claude_investigation as agentic
from backend.evidence.required_claim_evidence import committed_required_gaps, required_claimant_gaps
from backend.workflow.specialist_context import generated_policy_context
from backend.workflow.versions import V3 as WORKFLOW

ROLES = ("coverage", "fraud", "historical")
MAX_ROUNDS = 4
UNSUPPORTED_CLAIMANT_GAPS = "Claimant evidence gap is not supported by an executed finding"

READINESS_PROMPT = """## Task
Intake has finished. Assess whether Coverage, Fraud and Historical can start. Readiness means enough evidence to investigate.

## Dispatch
- Coverage requires completed Intake checks for policy lookup, period, vehicle identity, loss and costs. If coverage_blockers is nonempty, mark Coverage BLOCKED_INPUT and copy those blockers into missing_information.
- When a staff-selected policy PDF is present, direct Coverage to assess the wording or identify the missing contract documents if it is a product summary. Intake records cost gaps separately so Coverage can first examine applicability. For earlier synthetic claims with a saved schedule, direct Coverage to the schedule and its linked wording. The uploaded corpus is for manual claims.
- A selected PDF gives Intake enough wording to proceed. The submitted number, policy period, vehicle cover and endorsements still require evidence from the individual schedule.
- Assign Fraud to compare assertions in claim evidence. Missing paperwork and procedural conditions belong to Intake or Coverage. Issuer authentication is outside this workflow.
- When historical_available is true and Historical is incomplete, dispatch Historical to compare relevant prior decisions. Historical can start as soon as Intake is complete. Use NOT_APPLICABLE with no gaps or task when historical_available is false.
- If Historical is due, use BLOCKED_DEPENDENCY for Coverage with no task unless Coverage is already blocked by Intake evidence or completed. Coverage uses the committed comparison in the next round.
- Mark roles in completed_specialists as COMPLETED with no task. Fraud is eligible for another task only when fraud_followup_available is true. Unresolved questions about the same documents do not reopen Fraud.

## Output
Use assess_readiness. Use one short sentence per reason and task. Give every READY specialist one task and use DISPATCH only when tasks are present. Specialists receive claim facts and policy references separately. For a required gap, name the source, owner and check it blocks. Copy claimant gap names verbatim from allowed_claimant_gap_names. Mention optional evidence only as possible staff follow-up. Cite only the relevant exact IDs from available_evidence_references."""


class UnsupportedClaimantGaps(ValueError):
    def __init__(self, gaps):
        super().__init__(UNSUPPORTED_CLAIMANT_GAPS)
        self.gaps = gaps


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_or_document: str = Field(min_length=1, max_length=200)
    expected_source: str = Field(min_length=1, max_length=300)
    owner: Literal["claimant", "internal"]
    why_needed: str = Field(min_length=1, max_length=500)


class Readiness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["READY", "BLOCKED_INPUT", "BLOCKED_DEPENDENCY", "COMPLETED", "NOT_APPLICABLE"]
    reason: str = Field(min_length=1, max_length=1000)
    evidence_references: list[str] = Field(min_length=1, max_length=10)
    missing_information: list[Gap] = Field(default_factory=list, max_length=10)


class Specialists(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coverage: Readiness
    fraud: Readiness
    historical: Readiness


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specialists: Specialists
    tasks: list[agentic.Task] = Field(max_length=3)
    next_action: Literal["DISPATCH", "REQUEST_INFORMATION", "REVIEW", "FINISH"]
    explanation: str = Field(min_length=1, max_length=2000)


def sources_for(context, claim):
    sources = {f"submission:/{field}" for field in claim}
    for finding in context.get("findings", {}).values():
        sources.update(finding.get("evidence", []))
        sources.update(finding.get("details", {}).get("sources", []))
        for fact in finding.get("details", {}).get("facts", []):
            if fact.get("source"):
                sources.add(fact["source"])
    return sources


def completed(context, role):
    key = "fraud_initial" if role == "fraud" else role
    return context.get("findings", {}).get(key, {}).get("details", {}).get("checks_executed", False)


def historical_due(context):
    return context.get("historical_available") is True and not completed(context, "historical")


def has_new_leads(context):
    findings = context.get("findings", {})
    leads = findings.get("historical", {}).get("details", {}).get("fraud_investigation", {}).get("leads", [])
    return (bool(leads) and completed(context, "fraud")
            and not findings.get("fraud_followup", {}).get("details", {}).get("checks_executed"))


def claimant_gap_names(context):
    return {item[key] for item in committed_required_gaps(context.get("findings", {}))
            for key in ("field", "document") if item.get(key)}


def coverage_blockers(context):
    intake = context.get("findings", {}).get("intake", {}).get("details", {})
    evidence = intake.get("coverage_readiness", {})
    executed = intake.get("checks_executed")
    if (executed and evidence.get("ready") and evidence.get("policy_available", True)
            and not intake.get("error_type")):
        return []
    required = required_claimant_gaps(intake) if executed else []
    gaps = [{"field_or_document": gap["field"], "expected_source": gap["document"], "owner": "claimant",
             "why_needed": gap["reason"]} for gap in required]
    issue = evidence.get("internal_issue")
    if issue or not gaps:
        gaps.insert(0, {"field_or_document": "Policy and Intake checks", "expected_source": "Internal claim and policy records",
                        "owner": "internal", "why_needed": issue or "Intake has not confirmed the evidence needed for Coverage."})
    return gaps[:10]



def validate_selection(raw, context, claim):
    if not context.get("findings", {}).get("intake", {}).get("details", {}).get("checks_executed"):
        raise ValueError("Committed Intake checks are required before readiness dispatch")
    plan = Selection.model_validate(raw)
    states = plan.specialists.model_dump()
    tasks = [task.specialist for task in plan.tasks]
    if len(tasks) != len(set(tasks)) or any(role not in ROLES for role in tasks):
        raise ValueError("Invalid or repeated specialist selection")
    if set(tasks) != {role for role in ROLES if states[role]["state"] == "READY"}:
        raise ValueError("Tasks must match READY specialists exactly")
    if bool(tasks) != (plan.next_action == "DISPATCH"):
        raise ValueError("Next action contradicts task selection")
    sources = sources_for(context, claim)
    known_gaps = claimant_gap_names(context)
    unsupported_gaps = []
    for role in ROLES:
        state = states[role]
        if any(ref not in sources for ref in state["evidence_references"]):
            raise ValueError("Readiness cited unavailable evidence")
        done = completed(context, role)
        can_follow_up = role == "fraud" and has_new_leads(context)
        if state["state"] == "COMPLETED" and not done:
            raise ValueError("Unexecuted checks cannot be marked completed")
        if role in tasks and done and not can_follow_up:
            raise ValueError("Completed checks cannot be repeated without new fraud leads")
        if role == "historical" and ((role in tasks) != historical_due(context)):
            raise ValueError("Dispatch Historical when eligible prior claims are available and checks are incomplete")
        if (role == "historical" and context.get("historical_available") is False
                and (state["state"] != "NOT_APPLICABLE" or state["missing_information"])):
            raise ValueError("Historical comparison is not applicable without eligible prior claims")
        if role == "coverage" and role in tasks and coverage_blockers(context):
            raise ValueError("Coverage cannot start until Intake confirms enough evidence for all coverage checks")
        if role == "coverage" and role in tasks and historical_due(context):
            raise ValueError("Coverage requires the committed Historical comparison when prior claims are available")
        if state["state"] == "NOT_APPLICABLE" and not (role == "historical" and context.get("historical_available") is False):
            raise ValueError("Unsupported not-applicable declaration")
        if state["state"] == "BLOCKED_INPUT" and not state["missing_information"]:
            raise ValueError("Blocked input must identify precise gaps")
        for gap in state["missing_information"]:
            if gap["owner"] == "claimant" and gap["field_or_document"] not in known_gaps:
                unsupported_gaps.append({"specialist": role, **gap})
    if unsupported_gaps:
        raise UnsupportedClaimantGaps(unsupported_gaps)
    if plan.next_action == "REQUEST_INFORMATION":
        claimant_gaps = [g for state in states.values() for g in state["missing_information"] if g["owner"] == "claimant"]
        if not claimant_gaps:
            raise ValueError("Information request requires claimant-obtainable evidence")
    return plan.model_dump()


def assess(store, assessment_id, context, claim, round_number):
    if claim.get("source") == "Synthetic generator":
        context = generated_policy_context(context)

    def validate_result(result):
        # Dispatch comes first when a specialist is ready. The committed gaps
        # remain available to the eventual routing decision.
        if result.tasks:
            result.next_action = "DISPATCH"
        validate_selection(result, context, claim)

    result = agentic.structured_call(store, assessment_id, f"readiness:{round_number}", READINESS_PROMPT,
        {"claim": claim, "context": context, "dispatch_round": round_number, "maximum_dispatch_rounds": MAX_ROUNDS,
         "coverage_blockers": coverage_blockers(context),
         "completed_specialists": [role for role in ROLES if completed(context, role)],
         "fraud_followup_available": has_new_leads(context),
         "allowed_claimant_gap_names": sorted(claimant_gap_names(context)),
         "available_evidence_references": sorted(sources_for(context, claim))}, Selection, "assess_readiness",
        validate_result=validate_result)
    return result.model_dump()
