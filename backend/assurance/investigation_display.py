"""Prepare findings and completed specialist assignments for the claim detail screens."""
from sqlalchemy import select

from backend.agents.coverage_specialist import separate_policy_omissions
from backend.agents.fraud_specialist import screened_finding
from backend.storage.sqlite_store import AgentRun, ModelInvocation


def display_finding(finding, claim):
    if finding["agent"] != "coverage":
        return screened_finding(finding, claim)
    details = finding["details"]
    items, omissions = separate_policy_omissions(details.get("coverage_items", []), details.get("policy_omissions", []))
    return {**finding, "details": {**details, "coverage_items": items, "policy_omissions": omissions}}


def completed_model_assignments(session, assessment, runs=None):
    plans = assessment.context.get("readiness_history", {})
    if not any(plan.get("next_action") == "DISPATCH" for plan in plans.values()):
        return []
    if runs is None:
        runs = session.execute(select(AgentRun.stage, AgentRun.instruction, AgentRun.finding)
                               .where(AgentRun.assessment_id == assessment.id)).all()
    completed = {run.stage: run for run in runs
                 if run.finding["details"].get("checks_executed") and not run.finding["details"].get("error_type")}
    if not completed:
        return []

    invocations = session.execute(select(ModelInvocation.stage, ModelInvocation.response).where(
        ModelInvocation.assessment_id == assessment.id, ModelInvocation.status == "complete",
        ModelInvocation.stage.like("readiness:%")))
    rounds = {}
    for stage, response in invocations:
        number = stage.removeprefix("readiness:")
        plan = plans.get(number, {})
        if (not number.isdecimal() or plan.get("next_action") != "DISPATCH"
                or not isinstance(response, dict) or response.get("stop_reason") != "tool_use"):
            continue
        calls = [block for block in (response.get("content") or [])
                 if isinstance(block, dict) and block.get("type") == "tool_use"]
        tasks = plan["tasks"]
        if len(calls) != 1 or calls[0].get("name") != "assess_readiness":
            continue
        submitted = calls[0].get("input")
        if not isinstance(submitted, dict) or submitted.get("tasks") != tasks:
            continue
        assignments = []
        for task in tasks:
            run = completed.get(f"r{number}:{task['specialist']}")
            if run and run.instruction == task["task"]:
                assignments.append({"specialist": run.finding["agent"], "task": run.instruction})
        if assignments:
            rounds[int(number)] = {"round": int(number), "assignments": assignments}
    return [rounds[number] for number in sorted(rounds)]
