"""V3 ordering, per-specialist readiness, dispatch guards and recovery."""
import copy
import json
import unittest
from unittest.mock import patch

from sqlalchemy import select

import test_agentic as fixtures
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, DecisionRecord, ModelInvocation, ToolCall, WorkflowEvent
from backend.workflow import specialist_readiness as readiness
from backend.assurance.investigation_display import completed_model_assignments
from scripts.debug_workflow import scripted_readiness


class IntakeFirstTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AgenticTests()
        self.fixture.setUp()
        self.store, self.workflow = self.fixture.store, self.fixture.workflow

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self, index=0):
        identifier = self.fixture.prepare(index)
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            row.execution = {**row.execution, "workflow": readiness.WORKFLOW}
            row.context = {**row.context, "workflow": readiness.WORKFLOW}
        return identifier

    def run_case(self, index=0, override=None, *, record_readiness=False):
        identifier = self.prepare(index)
        provider_calls = []
        def provider(*args):
            provider_calls.append(args[2])
            if args[2].startswith("readiness:"):
                with self.store.session() as session:
                    self.assertIsNotNone(session.scalar(select(AgentRun).where(AgentRun.assessment_id == identifier, AgentRun.stage == "r0:intake")))
                self.assertIn("intake", json.loads(args[4][0]["content"])["context"]["findings"])
            response = self.fixture.provider(*args)
            if record_readiness and args[2].startswith("readiness:"):
                stage = args[2]
                with self.store.session.begin() as session:
                    session.add(ModelInvocation(id=f"{identifier}:{stage}", assessment_id=identifier, stage=stage,
                                                payload={}, response=response, status="complete"))
            return response
        with patch("backend.agents.claude_investigation.call_model", side_effect=provider):
            if override:
                with patch("backend.workflow.intake_first_graph.assess_readiness", side_effect=override):
                    self.workflow.run(identifier)
            else:
                self.workflow.run(identifier)
        with self.store.session() as session:
            decision = session.get(DecisionRecord, identifier)
            return identifier, session.get(Assessment, identifier), decision.payload if decision else None, provider_calls

    def test_intake_commits_before_supervisor_and_history_waits_for_coverage(self):
        identifier, row, decision, calls = self.run_case()
        self.assertEqual(calls[0], "readiness:1")
        self.assertEqual(decision["route"], "auto-approve")
        with self.store.session() as session:
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()
        self.assertEqual({r.stage: r.input_version for r in runs}, {"r0:intake": 0, "r1:coverage": 1, "r1:fraud": 1, "r2:historical": 2})
        self.assertEqual(row.context["readiness_history"]["1"]["specialists"]["historical"]["state"], "BLOCKED_DEPENDENCY")
        self.assertEqual(row.context["readiness_history"]["2"]["specialists"]["historical"]["state"], "READY")

    def test_missing_loss_date_blocks_coverage_but_runs_fraud(self):
        _, row, decision, _ = self.run_case(4)
        self.assertEqual(decision["route"], "request more information")
        self.assertIn("fraud_initial", row.context["findings"])
        self.assertNotIn("coverage", row.context["findings"])
        gap = row.context["readiness_gaps"][0]
        self.assertEqual(gap["field_or_document"], "loss_date")
        self.assertEqual(gap["owner"], "claimant")

    def test_missing_estimate_blocks_coverage_but_runs_fraud(self):
        identifier, row, decision, calls = self.run_case(3, record_readiness=True)
        evidence = row.context["findings"]["intake"]["details"]["coverage_readiness"]
        self.assertFalse(evidence["ready"])
        self.assertEqual(evidence["checks"], {"policy_period": True, "vehicle_match": True, "loss_and_costs": False})
        self.assertIn("fraud_initial", row.context["findings"])
        self.assertNotIn("coverage", row.context["findings"])
        self.assertEqual([stage for stage in calls if stage.startswith("readiness:")], ["readiness:1"])
        self.assertFalse(any(stage.endswith(":coverage") for stage in calls))
        with self.store.session() as session:
            self.assertIsNone(session.scalar(select(ToolCall.id).where(
                ToolCall.assessment_id == identifier, ToolCall.tool == "retrieve_policy")))
            fraud = row.context["readiness_history"]["1"]["tasks"][0]
            self.assertEqual(completed_model_assignments(session, row), [
                {"round": 1, "assignments": [{"specialist": "fraud_initial", "task": fraud["task"]}]}
            ])
        self.assertEqual(decision["route"], "request more information")

    def test_investigation_shows_accepted_model_tasks_that_produced_findings(self):
        identifier, _, _, _ = self.run_case(14, record_readiness=True)
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            result = completed_model_assignments(session, assessment)
            self.assertEqual([round_item["round"] for round_item in result], [1, 2, 3])
            self.assertEqual([[task["specialist"] for task in round_item["assignments"]] for round_item in result],
                             [["coverage", "fraud_initial"], ["historical"], ["fraud_followup"]])
            expected = assessment.context["readiness_history"]["1"]["tasks"][0]["task"]
            self.assertEqual(result[0]["assignments"][0]["task"], expected)

            coverage_run = session.scalar(select(AgentRun).filter_by(assessment_id=identifier, stage="r1:coverage"))
            coverage_run.finding = {**coverage_run.finding, "details": {**coverage_run.finding["details"], "error_type": "TimeoutError"}}
            historical_call = session.get(ModelInvocation, f"{identifier}:readiness:2")
            historical_call.response = copy.deepcopy(historical_call.response)
            historical_call.response["content"][0]["input"]["tasks"][0]["task"] = "A different proposed task"
            session.flush()
            result = completed_model_assignments(session, assessment)
            self.assertEqual([round_item["round"] for round_item in result], [1, 3])
            self.assertEqual([task["specialist"] for task in result[0]["assignments"]], ["fraud_initial"])

    def test_fraud_can_be_blocked_while_coverage_runs(self):
        def selection(store, identifier, context, claim, round_number):
            data = {"context": context, "claim": claim, "available_evidence_references": sorted(readiness.sources_for(context, claim))}
            plan = scripted_readiness(data)
            # An unavailable insurer verification service is an internal concern.
            if not readiness.completed(context, "fraud"):
                plan["specialists"]["fraud"] = {"state": "BLOCKED_INPUT", "reason": "Issuer service unavailable",
                    "evidence_references": ["submission:/documents"], "missing_information": [{"field_or_document": "issuer confirmation",
                    "expected_source": "Insurer issuer service", "owner": "internal", "why_needed": "Verify issuer record"}]}
                plan["tasks"] = [t for t in plan["tasks"] if t["specialist"] != "fraud"]
                plan["next_action"] = "DISPATCH" if plan["tasks"] else "REVIEW"
            return plan
        _, row, decision, _ = self.run_case(override=selection)
        self.assertIn("coverage", row.context["findings"])
        self.assertNotIn("fraud_initial", row.context["findings"])
        self.assertEqual(decision["route"], "adjuster review")

    def test_coverage_failure_stops_history_and_records_skip(self):
        for index in (7, 8):
            identifier, row, decision, _ = self.run_case(index)
            self.assertEqual(decision["route"], "adjuster review")
            self.assertNotIn("historical", row.context["findings"])
            self.assertIn("Coverage gate", row.context["skipped"]["historical"])
            with self.store.session() as session:
                self.assertFalse(session.scalar(select(ToolCall.id).where(ToolCall.assessment_id == identifier, ToolCall.tool == "retrieve_comparable_claims")))

    def test_new_historical_leads_trigger_later_fraud_and_human_route(self):
        _, row, decision, _ = self.run_case(14)
        self.assertIn("fraud_followup", row.context["findings"])
        self.assertEqual(decision["route"], "fraud escalation")
        self.assertEqual(row.context["readiness_history"]["3"]["tasks"][0]["specialist"], "fraud")

    def test_high_value_still_overrides_approval(self):
        _, _, decision, _ = self.run_case(5)
        self.assertEqual(decision["route"], "adjuster review")
        self.assertEqual(decision["reason_codes"], ["GROSS_AUTHORITY"])

    def test_invalid_dispatch_is_persisted_and_fails_closed(self):
        def bad_plan(store, identifier, context, claim, round_number):
            result = scripted_readiness({"context": context, "claim": claim, "available_evidence_references": sorted(readiness.sources_for(context, claim))})
            result["specialists"]["historical"]["state"] = "READY"
            result["tasks"].append({"specialist": "historical", "task": "Skip coverage gate"})
            return result
        _, row, decision, _ = self.run_case(override=bad_plan)
        self.assertIsNone(decision)
        self.assertEqual(row.status, "failed")
        self.assertIn("rejected_selection", row.context["readiness_history"]["1"])
        self.assertNotIn("historical", row.context["findings"])
        with self.store.session() as session:
            self.assertEqual(completed_model_assignments(session, row), [])

    def test_rejected_dispatch_records_only_the_unsupported_claimant_requests(self):
        gaps = [{"field_or_document": name, "expected_source": "Claimant", "owner": "claimant",
                 "why_needed": reason} for name, reason in (
                     ("Police report / incident report", "Corroborate the incident"),
                     ("Damage photographs", "Check the extent of damage"),
                 )]

        def disputed_plan(store, identifier, context, claim, round_number):
            plan = scripted_readiness({"context": context, "claim": claim})
            plan["specialists"]["fraud"].update(state="BLOCKED_INPUT", missing_information=gaps)
            plan.update(tasks=[], next_action="REQUEST_INFORMATION")
            return plan

        identifier, row, decision, calls = self.run_case(4, override=disputed_plan)
        rejected = row.context["readiness_history"]["1"]
        self.assertEqual(rejected["rejected_selection"]["specialists"]["coverage"]["missing_information"][0]["field_or_document"], "loss_date")
        self.assertEqual(rejected["unsupported_claimant_gaps"], [{"specialist": "fraud", **gap} for gap in gaps])
        self.assertIsNone(decision)
        self.assertEqual(row.status, "failed")
        self.assertTrue(row.context["findings"]["intake"]["details"]["missing"])
        self.assertFalse(any(stage.startswith("r1:") for stage in calls))
        with self.store.session() as session:
            event = session.scalar(select(WorkflowEvent).where(WorkflowEvent.assessment_id == identifier,
                WorkflowEvent.kind == "readiness_assessed"))
        self.assertEqual(event.payload["unsupported_claimant_gaps"], rejected["unsupported_claimant_gaps"])

    def test_intake_failure_does_not_invoke_readiness_or_downstream_specialists(self):
        identifier = self.prepare()
        with patch("backend.agents.claude_investigation.execute_checks", side_effect=RuntimeError("Intake checks unavailable")), \
             patch("backend.agents.claude_investigation.call_model") as calls:
            self.workflow.run(identifier)
        calls.assert_not_called()
        with self.store.session() as session:
            self.assertIsNone(session.get(DecisionRecord, identifier))
            self.assertEqual(session.get(Assessment, identifier).status, "failed")
            self.assertEqual(len(session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()), 1)

    def test_crash_after_intake_resumes_without_duplicate_findings(self):
        identifier = self.prepare()
        with patch("backend.agents.claude_investigation.call_model", side_effect=self.fixture.provider):
            # Inject a publication crash after Intake and specialist commits.
            with patch("backend.agents.claude_investigation.judge", side_effect=RuntimeError("judge unavailable")), patch.object(self.workflow, "publish", side_effect=RuntimeError("publication crash")):
                with self.assertRaises(RuntimeError):
                    self.workflow.run(identifier)
        with patch("backend.agents.claude_investigation.call_model", side_effect=self.fixture.provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()
            self.assertEqual(len([r for r in runs if r.stage == "r0:intake"]), 1)
            events = session.scalars(select(WorkflowEvent).where(WorkflowEvent.assessment_id == identifier, WorkflowEvent.kind == "intake_completed")).all()
            self.assertEqual(len(events), 1)

    def test_dispatch_budget_exhaustion_allows_reprocessing(self):
        with patch("backend.workflow.specialist_readiness.MAX_ROUNDS", 1):
            _, row, decision, _ = self.run_case()
        self.assertIn("budget exhausted", row.context["investigation_error"])
        self.assertIsNone(decision)
        self.assertEqual(row.status, "failed")


class ReadinessValidationTests(unittest.TestCase):
    def setUp(self):
        self.context = {"findings": {"intake": {"details": {"checks_executed": True, "missing": [],
            "coverage_readiness": {"ready": True}}, "evidence": ["tool:intake"]}}}
        self.claim = {"policy_id": "P1", "registration": "V1", "loss_date": "2026-10-15", "loss_type": "glass", "documents": []}
        self.plan = scripted_readiness({"context": self.context, "claim": self.claim, "available_evidence_references": ["tool:intake"]})

    def test_fabricated_reference_and_contradictory_task_fail(self):
        bad = copy.deepcopy(self.plan)
        bad["specialists"]["coverage"]["evidence_references"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unavailable evidence"):
            readiness.validate_selection(bad, self.context, self.claim)
        bad = copy.deepcopy(self.plan)
        bad["tasks"] = []
        with self.assertRaisesRegex(ValueError, "match READY"):
            readiness.validate_selection(bad, self.context, self.claim)

    def test_completed_checks_cannot_be_repeated(self):
        self.context["findings"]["coverage"] = {"details": {"checks_executed": True, "coverage_status": "COVERED"}}
        with self.assertRaisesRegex(ValueError, "cannot be repeated"):
            readiness.validate_selection(self.plan, self.context, self.claim)

    def test_unexecuted_checks_cannot_be_marked_completed(self):
        self.plan["specialists"]["historical"]["state"] = "COMPLETED"
        with self.assertRaisesRegex(ValueError, "Unexecuted"):
            readiness.validate_selection(self.plan, self.context, self.claim)

    def test_readiness_requires_committed_intake_checks(self):
        with self.assertRaisesRegex(ValueError, "Committed Intake"):
            readiness.validate_selection(self.plan, {"findings": {}}, self.claim)

    def test_ready_plan_cannot_override_intake_block(self):
        self.context["findings"]["intake"]["details"]["coverage_readiness"]["ready"] = False
        with self.assertRaisesRegex(ValueError, "Coverage cannot start"):
            readiness.validate_selection(self.plan, self.context, self.claim)


if __name__ == "__main__":
    unittest.main()
