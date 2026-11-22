"""Ask Claude which specialists can run and validate assignments against saved evidence and dependencies."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents import claude_investigation as agentic
from backend.evidence.required_claim_evidence import committed_required_gaps, required_claimant_gaps
from backend.workflow.specialist_context import generated_policy_context

WORKFLOW = "claude-intake-first-v3"
ROLES = ("coverage", "fraud", "historical")
MAX_ROUNDS = 4
UNSUPPORTED_CLAIMANT_GAPS = "Claimant evidence gap is not supported by an executed finding"


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


def has_new_leads(context):
    findings = context.get("findings", {})
    leads = findings.get("historical", {}).get("details", {}).get("fraud_investigation", {}).get("leads", [])
    return bool(leads) and not findings.get("fraud_followup", {}).get("details", {}).get("checks_executed")


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
    coverage_status = context.get("findings", {}).get("coverage", {}).get("details", {}).get("coverage_status")
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
        if role == "historical" and role in tasks and coverage_status != "COVERED":
            raise ValueError("Historical comparison requires confirmed coverage")
        if (role == "historical" and context.get("historical_available") is False
                and (state["state"] != "NOT_APPLICABLE" or state["missing_information"])):
            raise ValueError("Historical comparison is not applicable without eligible staff-approved claims")
        if role == "coverage" and role in tasks and coverage_blockers(context):
            raise ValueError("Coverage cannot start until Intake confirms enough evidence for all coverage checks")
        if state["state"] == "NOT_APPLICABLE" and not (role == "historical" and (
                coverage_status in {"INACTIVE", "NOT_COVERED"} or context.get("historical_available") is False)):
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
    result = agentic.structured_call(store, assessment_id, f"readiness:{round_number}",
        "Intake has already completed. Assess coverage, fraud and historical readiness separately. "
        "Select every READY specialist with a concrete task; never select Intake. Enough information means enough to begin investigation, not proven coverage. "
        "Coverage can start only after the saved Intake checks confirm the submitted policy exists and there is enough evidence "
        "to assess the policy period, vehicle identity, loss type and claimed costs. When coverage_blockers is nonempty, mark Coverage BLOCKED_INPUT, copy the blockers into its "
        "missing_information and leave it out of tasks. If Fraud has yet to run, assign it a concrete screening task limited to the available evidence. "
        "Missing evidence by itself does not constitute a fraud concern. "
        "Issuer authentication is outside this workflow. Never block a specialist on issuer confirmation. "
        "For generated claims, assign Coverage to retrieve the saved policy for the submitted policy ID and assess its terms against the claim. "
        "When synthetic wording is linked, assign Coverage to search that version, cite applicable passages and assess differences with the schedule. "
        "The uploaded wording corpus is for manual claims only. "
        "Historical MUST wait for saved COVERED coverage. Completed checks must not repeat, except Fraud once to verify new historical leads. "
        "Inactive/uncovered coverage or historical_available=false makes Historical NOT_APPLICABLE with no gaps or task. "
        "Identify exact missing evidence that blocks a required check, its source and claimant/internal owner. For claimant gaps, copy field_or_document "
        "verbatim from allowed_claimant_gap_names. Describe optional evidence concerns in the explanation as possible follow-up for staff. "
        "Use supplied exact evidence identifiers. Do not invent document requirements. Use DISPATCH only with tasks; otherwise request evidence, review or finish. "
        "Auto-approval still requires all mandatory checks and positive supported payable. Keep explanations concise.",
        {"claim": claim, "context": context, "dispatch_round": round_number, "maximum_dispatch_rounds": MAX_ROUNDS,
         "coverage_blockers": coverage_blockers(context),
         "allowed_claimant_gap_names": sorted(claimant_gap_names(context)),
         "available_evidence_references": sorted(sources_for(context, claim))}, Selection, "assess_readiness")
    return result.model_dump()
