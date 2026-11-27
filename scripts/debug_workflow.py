"""Trace real backend code using isolated fixtures and a scripted model.

Run: .venv\Scripts\python.exe scripts/debug_workflow.py --scenario historical-fraud
No API keys, HTTP calls, live workspace records or downloaded embeddings needed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
from statistics import median
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, DecisionRecord, HumanTask, ToolCall, WorkflowEvent, Store
from scripts.claim_fixtures import generate_batch
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.agents import claude_investigation as agentic
from backend.agents import historical_comparer as historical
from backend.workflow.claims_workflow import ClaimsWorkflow


def scripted_analysis_fields(stage, model_input):
    """Populate the typed response for fixtures with known synthetic evidence."""
    claim = model_input.get("submission", {})
    if stage.endswith(":coverage"):
        return {"classified_loss": claim.get("loss_type") or "uncertain", "coverage_conclusion": "COVERED", "coverage_items": [
            {"item": "Reported loss", "outcome": "covered", "policy_term": "The fixture schedule covers glass and repair losses.",
             "reason": "The fixture policy lists this type of loss."},
        ]}
    if stage.endswith(":historical"):
        [evidence] = model_input["required_checks"]["supporting_evidence"]
        records = evidence["result"]["records"]
        comparable_ids = historical.exact_scope_ids(claim, records)
        costs = [amount for row in records if row["id"] in comparable_ids
                 and (amount := historical.cost_document_amount(row)) is not None]
        current = historical.cost_document_amount(claim)
        conclusion = "INSUFFICIENT_EVIDENCE"
        if costs and current is not None:
            conclusion = "REQUIRES_REVIEW" if current > 2 * median(costs) else "SUPPORTED"
        return {"amount_conclusion": conclusion, "comparable_claim_ids": comparable_ids}
    return {}


class ScriptedModel:
    """Replaces only model responses; actual agents and tools still execute."""
    def __init__(self):
        self.calls = []

    def __call__(self, store, identifier, stage, system, messages, tools, choice):
        if stage.startswith("readiness:"):
            name, value = "assess_readiness", scripted_readiness(json.loads(messages[0]["content"]))
        elif stage == "judge":
            context = json.loads(messages[0]["content"])
            sources = sorted({ref for f in context["findings"].values() for ref in f["details"].get("sources", [])})
            # Deliberately recommend approval, so the guard must override unsafe cases.
            name, value = "recommend_route", {"route": "auto-approve", "explanation": "Scripted recommendation used to demonstrate mandatory guard overrides.",
                                               "evidence": sources[:5], "outstanding_gaps": []}
        else:
            model_input = json.loads(messages[0]["content"])
            bound_wording = stage.endswith(":coverage") and model_input.get("context", {}).get("policy_search_mode") == "bound-synthetic"
            if bound_wording and len(messages) == 1:
                response = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "search",
                    "name": "search_policy_documents", "input": {"query": "covered incident, claimed repairs, exclusions and deductible"}}]}
                self.record(stage, system, messages, response)
                return response
            results = [model_input["required_checks"]]
            if len(messages) > 1:
                results += [json.loads(item["content"]) for item in messages[-1]["content"]]
            sources = [item["evidence_ref"] for item in results]
            sources += [p["source"] for item in results for p in item["result"].get("passages", [])]
            name, value = "submit_analysis", {"summary": f"Scripted model reviewed checks and {len(results[0]['supporting_evidence'])} supporting evidence response(s).", "review_reasons": [],
                                               "evidence": sources[:5], "policy_applicability": "applicable",
                                               **scripted_analysis_fields(stage, model_input)}
            if bound_wording:
                passages = [p["source"] for item in results for p in item["result"].get("passages", [])][:1]
                value.update(applicable_passages=passages, policy_conflicts=[])
                for item in value["coverage_items"]:
                    item.update(basis="both", passage_refs=passages)
        response = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "submit", "name": name, "input": value}]}
        self.record(stage, system, messages, response)
        return response

    def record(self, stage, system, messages, response):
        self.calls.append(copy.deepcopy({"stage": stage, "instructions": system, "messages": messages, "response": response}))


class CallTrace:
    """Python call/return profiling in the main and specialist worker threads.

    Only selected backend values are copied: never arbitrary locals or secrets.
    """
    TARGETS = {"run", "supervise", "collect", "judge", "publish", "select_route",
               "intake", "after_intake", "after_collect", "assess", "validate_selection",
               "run_specialist", "investigate", "specialist_context", "prior_evidence_sources", "build_tools",
               "read_context", "read_submission", "find_duplicates", "retrieve_policy",
               "retrieve_comparable_claims", "verify_historical_leads", "execute_checks", "record_evidence", "search_policy_documents"}

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()
        self.start = time.perf_counter()
        self.backend = str(ROOT / "backend")

    def profile(self, frame, event, value):
        filename = frame.f_code.co_filename
        if event not in {"call", "return"} or not filename.startswith(self.backend) or frame.f_code.co_name not in self.TARGETS:
            return
        local = frame.f_locals
        data = {name: local[name] for name in ("assessment_id", "role", "stage", "round_number")
                if name in local and isinstance(local[name], (str, int))}
        context = local.get("context")
        if isinstance(context, dict):
            data["context"] = {"version": context.get("version"), "finding_names": list(context.get("findings", {}))}
        if event == "return":
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            if isinstance(value, dict):
                data["returned_keys"] = list(value)
                for name in ("status", "agent", "route", "context_version", "version", "round"):
                    if name in value:
                        data[name] = value[name]
        record = {"event": event, "function": frame.f_code.co_name,
                  "file": str(Path(filename).relative_to(ROOT)).replace("\\", "/"),
                  "line": frame.f_code.co_firstlineno, "thread": threading.current_thread().name,
                  "elapsed_ms": round((time.perf_counter() - self.start) * 1000, 2), "values": data}
        with self.lock:
            self.events.append(record)

    def __enter__(self):
        self.previous = sys.getprofile(), threading.getprofile()
        sys.setprofile(self.profile)
        threading.setprofile(self.profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(self.previous[0])
        threading.setprofile(self.previous[1])


def scripted_readiness(data):
    """Scripted Claude response using the single graph's offline readiness policy."""
    from backend.workflow.investigation_execution import offline_readiness
    return offline_readiness(data["context"], data["claim"])


def debug(scenario, destination):
    destination.mkdir(parents=True, exist_ok=False)
    store = Store(destination / "claims.sqlite3")
    workflow = ClaimsWorkflow(store, destination / "checkpoints.sqlite3")
    model = ScriptedModel()
    try:
        with patch.dict(os.environ, {"CLAIMS_LLM_MODE": "deterministic", "CLAIMS_ENABLE_LANGFUSE": "0"}), \
             patch("backend.agents.claude_investigation.call_model", side_effect=model), \
             patch("httpx.Client.post", side_effect=AssertionError("External HTTP is disabled in this debugger")):
            batch = generate_batch(store, destination / "labels.sqlite3", 137, 16, batch_id="offline-debug-fixtures")
            identifier = batch["assessment_ids"][{"covered": 0, "high-value": 5, "historical-fraud": 14}[scenario]]
            with store.session.begin() as session:
                row = session.get(Assessment, identifier)
                row.execution = {**row.execution, "workflow": "claude-intake-first-v3", "mode": "anthropic-tools", "debug_provider": "scripted; no live model"}
            with CallTrace() as trace:
                workflow.run(identifier)
            audit = export_audit(store, identifier)
            assert verify_audit(audit), "Audit verification failed"
            with store.session() as session:
                assessment = session.get(Assessment, identifier)
                claim = session.get(Claim, assessment.claim_id)
                runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier).order_by(AgentRun.stage)).all()
                calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier).order_by(ToolCall.id)).all()
                events = session.scalars(select(WorkflowEvent).where(WorkflowEvent.assessment_id == identifier).order_by(WorkflowEvent.id)).all()
                decision = session.get(DecisionRecord, identifier).payload
                task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == identifier))
                report = {"scenario": scenario, "model": "scripted offline responses",
                          "assessment_id": identifier, "claim": claim.payload, "decision": decision,
                          "final_context_version": assessment.version, "human_queue": task.queue if task else None,
                          "human_task_status": task.status if task else None,
                          "stages": [{"stage": r.stage, "input_version": r.input_version, "finding": r.finding} for r in runs],
                          "tools": [{"stage": c.stage, "tool": c.tool, "request": c.request, "response": c.response} for c in calls],
                          "workflow_events": [{"kind": e.kind, "payload": e.payload} for e in events],
                          "model_transcript": model.calls, "function_trace": trace.events, "audit_valid": True}
            (destination / "trace.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (destination / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            lines = [f"# Observed backend trace: {scenario}", "", "Actual graph, tools and database code with scripted model responses. No external HTTP calls.", "",
                     f"Gross claimed: LKR {claim.payload['gross_amount']}", f"Supported payable: {decision['payable_amount']}",
                     f"Judge recommendation: {decision['recommendation']['route']}", f"Published route: **{decision['route']}**",
                     f"Reason codes: {', '.join(decision['reason_codes'])}", f"Human queue: {task.queue if task else 'none'}", "",
                     "## Persisted workflow events", ""]
            lines.extend(f"{i}. **{e['kind']}**: `{json.dumps(e['payload'])}`" for i, e in enumerate(report["workflow_events"], 1))
            lines += ["", "## Specialist inputs and results", "", "| Stage | Input context version | Finding status |", "| --- | ---: | --- |"]
            lines.extend(f"| {r.stage} | {r.input_version} | {r.finding['status']} |" for r in runs)
            lines += ["", "## Actual function calls (thread interleaving is expected)", ""]
            for item in trace.events:
                if item["event"] == "call":
                    location = (ROOT / item["file"]).as_posix()
                    lines.append(f"- [{item['file']}:{item['line']}]({location}:{item['line']}) → `{item['function']}` · {item['thread']}")
            lines += ["", "See trace.json for scoped model inputs, tool responses and function return summaries; audit.json contains reconstructable evidence."]
            (destination / "walkthrough.md").write_text("\n".join(lines), encoding="utf-8")
            print(json.dumps({"scenario": scenario, "gross": claim.payload["gross_amount"], "payable": decision["payable_amount"],
                              "recommendation": decision["recommendation"]["route"], "published": decision["route"],
                              "human_queue": task.queue if task else None, "context_version": assessment.version,
                              "profile_events": len(trace.events), "audit_valid": True, "output": str(destination.resolve())}, indent=2))
    finally:
        workflow.close()
        store.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["covered", "high-value", "historical-fraud"], default="historical-fraud")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ".local" / "debug" / f"{args.scenario}-{uuid.uuid4().hex[:8]}"
    debug(args.scenario, output)
