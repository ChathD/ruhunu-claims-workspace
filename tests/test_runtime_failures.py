"""Operational failures stay actionable and do not cause unnecessary judge calls."""
import unittest
from unittest.mock import patch

from sqlalchemy import select

import test_agentic as fixtures
from backend.__main__ import main
from backend.storage.sqlite_store import Assessment, DecisionRecord, ToolCall
from backend.agents import claude_investigation as agentic


class RuntimeFailureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AgenticTests()
        self.fixture.setUp()
        self.store = self.fixture.store

    def tearDown(self):
        self.fixture.tearDown()

    def test_old_tool_signature_reproduces_error_and_skips_judge(self):
        identifier = self.fixture.prepare(5)
        def old_build_tools(store, assessment_id, stage, context_snapshot=None, audit_stage=None):
            raise AssertionError("Argument binding should reject on_result before this body runs")
        with patch.object(agentic, "build_tools", new=old_build_tools), \
             patch.object(agentic, "call_model") as model, \
             self.assertLogs("backend.workflow.intake_first_graph", level="ERROR") as logs:
            self.fixture.workflow.run(identifier)
        model.assert_not_called()
        self.assertIn("unexpected keyword argument 'on_result'", "\n".join(logs.output))
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            details = row.context["findings"]["intake"]["details"]
            self.assertEqual(details["error_type"], "TypeError")
            self.assertIn("on_result", details["error_message"])
            self.assertFalse(details["checks_executed"])
            self.assertIsNone(session.scalar(select(ToolCall.id).where(ToolCall.assessment_id == identifier)))
            self.assertEqual(row.context["investigation_error"], "Intake did not complete required checks")
            self.assertNotIn("judge_error", row.context)
            self.assertIsNone(session.get(DecisionRecord, identifier))
            self.assertEqual(row.status, "failed")

    def test_readiness_failure_skips_judge(self):
        identifier = self.fixture.prepare()
        with patch.object(agentic, "call_model", side_effect=self.fixture.provider) as model, \
             patch("backend.workflow.specialist_readiness.assess", side_effect=RuntimeError("Readiness unavailable")):
            self.fixture.workflow.run(identifier)
        self.assertFalse(any(call.args[2] == "judge" for call in model.call_args_list))
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            self.assertEqual(row.context["investigation_error"], "Readiness validation failed: RuntimeError")
            self.assertIsNone(session.get(DecisionRecord, identifier))

    def test_judge_failure_preserves_findings_without_a_route(self):
        identifier = self.fixture.prepare()
        def model(*args):
            if args[2] == "judge":
                raise ValueError("Judge cited unavailable evidence")
            return self.fixture.provider(*args)
        with patch.object(agentic, "call_model", side_effect=model), \
             self.assertLogs("backend.workflow.intake_first_graph", level="ERROR"):
            self.fixture.workflow.run(identifier)
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            context = row.context
            self.assertEqual(row.status, "failed")
            self.assertIsNone(session.get(DecisionRecord, identifier))
        self.assertIn("coverage", context["findings"])
        self.assertEqual(context["judge_error"], {"error_type": "ValueError", "error_message": "Judge cited unavailable evidence"})

    def test_empty_source_set_restricts_judge_to_empty_citations(self):
        identifier = self.fixture.prepare()
        def provider(*args):
            evidence = args[5][0]["input_schema"]["properties"]["evidence"]
            self.assertEqual(evidence["maxItems"], 0)
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "judge", "name": "recommend_route",
                "input": {"route": "adjuster review", "explanation": "No completed evidence checks", "evidence": [], "outstanding_gaps": []}}]}
        with patch.object(agentic, "call_model", side_effect=provider):
            result = agentic.judge(self.store, identifier, {"findings": {}})
        self.assertEqual(result["evidence"], [])

    def test_claim_field_names_still_cannot_be_used_as_citations(self):
        identifier = self.fixture.prepare()
        response = {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "judge", "name": "recommend_route",
            "input": {"route": "adjuster review", "explanation": "Authority exceeded", "evidence": ["claim.gross_amount"], "outstanding_gaps": []}}]}
        with patch.object(agentic, "call_model", return_value=response):
            with self.assertRaisesRegex(ValueError, "Judge cited unavailable evidence"):
                agentic.judge(self.store, identifier, {"findings": {}})


class DevelopmentStartupTests(unittest.TestCase):
    def test_reload_is_opt_in_and_watches_backend_only(self):
        for arguments, expected in [([], False), (["--reload"], True)]:
            with self.subTest(arguments=arguments), patch("backend.__main__.load_dotenv"), \
                 patch("backend.__main__.uvicorn.run") as run:
                main(arguments)
            options = run.call_args.kwargs
            self.assertEqual(options["reload"], expected)
            if expected:
                self.assertEqual(len(options["reload_dirs"]), 1)
                self.assertTrue(options["reload_dirs"][0].endswith("backend"))
            else:
                self.assertIsNone(options["reload_dirs"])
