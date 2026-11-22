"""Ablation and dependency tests: distinguish useful context from unused input."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.agents import coverage_specialist as coverage, fraud_specialist as fraud, historical_comparer as historical
from scripts.claim_fixtures import FixtureHistory, make_fixture
from backend.workflow.specialist_context import specialist_context
from backend.agents import claude_investigation as agentic
from backend.agents.specialist_preflight import apply_verified_controls, finish_offline_checks
from backend.evidence.uploaded_claim_documents import cost_documents_from_extractions
from backend.claim_schemas import Finding
from backend.storage.sqlite_store import Assessment, Claim, History, Store, digest
from backend.evidence.specialist_evidence_tools import build_tools


def tools_for(**responses):
    return {name: SimpleNamespace(invoke=Mock(return_value=result)) for name, result in responses.items()}


class ContextTests(unittest.TestCase):
    def test_scoped_tools_use_frozen_findings_and_version(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "claims.sqlite3")
            store.history = FixtureHistory()
            try:
                payload = {"claim_id": "C1", "documents": []}
                with store.session.begin() as session:
                    session.add(Claim(id="C1", sequence=1, batch_id="test", policy_id="P1", payload=payload, payload_hash=digest(payload)))
                    session.flush()
                    session.add(Assessment(id="A1", claim_id="C1", context={"version": 9, "findings": {}}, version=9, execution={}))
                    for i in range(2):
                        session.add(History(id=f"H{i}", owner_claim_id="C1", payload={"incident_id": f"I{i}", "document_reference": "INV1"}))
                lead = {"document_reference": "INV1", "history_ids": ["H0", "H1"], "alternative": "Clerical error"}
                snapshot = {"version": 2, "findings": {"historical": {"details": {"fraud_investigation": {"leads": [lead]}}}},
                            "investigation_plans": "irrelevant"}
                tools = build_tools(store, "A1", "fraud_followup", context_snapshot=snapshot)
                view = tools["read_context"].invoke({})
                self.assertEqual(view["version"], 2)
                self.assertNotIn("investigation_plans", view)
                self.assertEqual(tools["verify_historical_leads"].invoke({})["verified"], [lead])
                self.assertNotIn("read_context", build_tools(store, "A1", "intake"))
            finally:
                store.engine.dispose()

    def test_historical_checks_do_not_read_unused_context(self):
        claim, _, records = make_fixture(42, 0, "ordinary_glass")
        tools = tools_for(read_submission=claim, retrieve_comparable_claims={"records": records})
        result = historical.investigate(tools)
        self.assertNotIn("amount_justification", result.details)
        finish_offline_checks("historical", result, {"read_submission": claim, "retrieve_comparable_claims": {"records": records}})
        self.assertEqual(result.details["amount_justification"]["conclusion"], "SUPPORTED")
        self.assertNotIn("consumed_context_version", result.details)
        self.assertNotIn("finding:coverage", result.evidence)

    def test_estimate_benchmark_uses_selected_document_totals_and_separates_extra_costs(self):
        claim, _, records = make_fixture(42, 1, "ordinary_repair")
        claim["additional_costs"] = [{"amount": "5000.00", "reason": "Towing for damaged bumper"}]
        selected = records[:5]
        for row in selected:
            row["gross_cost"] = "300000.00"
            row["documents"][0].update(total="100000.00", line_items=[
                {"description": "Replacement parts", "amount": "60000.00"},
                {"description": "Labour and installation", "amount": "40000.00"},
            ])
        result = Finding(agent="historical", status="pass", summary="Checked", evidence=[])
        historical.apply_comparison(result, claim, records, "SUPPORTED", [], [row["id"] for row in selected])
        amount = result.details["amount_justification"]
        self.assertEqual(amount["median"], "100000.00")
        self.assertEqual(amount["additional_costs_total"], "5000.00")
        self.assertEqual(amount["sample_size"], 5)
        self.assertEqual(result.status, "pass")
        with self.assertRaisesRegex(ValueError, "outside the retrieved history"):
            historical.apply_comparison(result, claim, records, "SUPPORTED", [], ["invented"])

    def test_initial_fraud_requires_no_issuer_verification(self):
        result = fraud.investigate({})
        apply_verified_controls("fraud", result, {}, {})
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.details["fraud_indicators"], [])

    def test_followup_fraud_depends_on_verified_historical_leads(self):
        lead = {"document_reference": "EST-1", "history_ids": ["H1", "H2"], "alternative": "Clerical error"}
        tools = tools_for(verify_historical_leads={"verified": []})
        self.assertEqual(fraud.investigate(tools, follow_up=True).status, "pass")
        tools["verify_historical_leads"].invoke.return_value = {"verified": [lead]}
        result = fraud.investigate(tools, follow_up=True)
        apply_verified_controls("fraud", result, tools, {name: tool.invoke.return_value for name, tool in tools.items()})
        self.assertEqual(result.status, "fraud")

    def test_intake_gaps_change_coverage_payable_and_request(self):
        claim, policy, _ = make_fixture(42, 0, "ordinary_glass")
        context = {"findings": {"intake": {"details": {"missing": []}}}}
        tools = tools_for(read_submission=claim, retrieve_policy={"policy": policy}, read_context=context)
        complete = coverage.investigate(tools)
        self.assertEqual(complete.status, "pass")
        self.assertEqual(complete.details["coverage_status"], "PENDING_ANALYSIS")
        self.assertNotIn("financial", complete.details)
        coverage.apply_approval_controls(complete, claim, {"policy": policy}, "COVERED")
        self.assertIsNotNone(complete.details["financial"]["payable"])
        gap = {"field": "line_items", "document": "EST-1", "reason": "Readable line items are needed to verify the amount claimed."}
        context["findings"]["intake"]["details"]["missing"] = [gap]
        incomplete = coverage.investigate(tools)
        coverage.apply_approval_controls(incomplete, claim, {"policy": policy}, "COVERED")
        self.assertEqual(incomplete.status, "information")
        self.assertIsNone(incomplete.details["financial"]["payable"])
        self.assertIn(gap, incomplete.details["missing"])

    def test_supplementary_estimate_is_not_used_as_required_cost_evidence(self):
        shared = {"status": "extracted", "registration": "VEH-1", "document_date": "2026-10-16",
                  "total": "2000.00", "provenance": "claimant upload"}
        documents = cost_documents_from_extractions([
            {**shared, "document_type": "Repair Estimate", "attachment_id": "repair", "name": "repair.pdf", "source": "attachment:repair"},
            {**shared, "document_type": "towing_estimate", "attachment_id": "towing", "name": "towing.pdf", "source": "attachment:towing"},
        ])
        self.assertEqual([document["id"] for document in documents], ["repair"])

    def test_estimate_plus_itemised_additional_cost_reconciles(self):
        claim, policy, _ = make_fixture(42, 13, "unexplained_additional_cost")
        tools = tools_for(read_submission=claim, retrieve_policy={"policy": policy},
                          read_context={"findings": {}})
        incomplete = coverage.investigate(tools)
        self.assertEqual(incomplete.details["missing"][0]["field"], "additional_costs")
        claim["additional_costs"] = [{"amount": "1000.00", "reason": "Tow to the garage"}]
        result = coverage.investigate(tools)
        coverage.apply_approval_controls(result, claim, {"policy": policy}, "COVERED")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.details["financial"]["additional_costs_total"], "1000.00")
        self.assertIsNotNone(result.details["financial"]["payable"])

    def test_projection_excludes_noise_and_preserves_dependencies(self):
        context = {"version": 7, "findings": {name: {"summary": name} for name in
                   ["intake", "coverage", "historical", "fraud_initial"]},
                   "investigation_plans": {"1": "NOISE"}, "recommendation": "NOISE", "attachment_extraction": ["NOISE"],
                   "policy_corpus_ids": ["P1"], "policy_search_mode": "uploaded-corpus"}
        original = copy.deepcopy(context)
        coverage_view = specialist_context(context, "coverage")
        self.assertEqual(set(coverage_view["findings"]), {"intake"})
        self.assertEqual(coverage_view["policy_corpus_ids"], ["P1"])
        self.assertNotIn("NOISE", json.dumps(coverage_view))
        followup = specialist_context(context, "fraud_followup")
        self.assertIn("historical", followup["findings"])
        self.assertIn("coverage", followup["findings"])
        self.assertNotIn("policy_corpus_ids", followup)
        coverage_view["findings"]["intake"]["summary"] = "changed"
        self.assertEqual(context, original)

    def test_specialist_model_input_uses_projection_after_intake(self):
        context = {"version": 3, "findings": {"intake": {"summary": "Intake gaps"}},
                   "investigation_plans": {"private": "UNRELATED_PLAN"}}
        captured = []
        def capture(*args):
            captured.append(json.loads(args[4][0]["content"]))
            raise RuntimeError("Stop before any provider request")
        def build_scoped_tools(*args, on_result, **kwargs):
            on_result("read_submission", {}, {"source": "Synthetic generator"})
            return {}
        checked = Finding(agent="test", status="pass", summary="Checked", evidence=[])
        with patch.object(agentic, "build_tools", side_effect=build_scoped_tools), \
             patch.object(agentic, "execute_checks", return_value=checked), \
             patch.object(agentic, "apply_verified_controls"), \
             patch.object(agentic, "check_deadline"), \
             patch.object(agentic, "record_evidence", return_value={"result": {}, "evidence_ref": "tool:test"}), \
             patch.object(agentic, "call_model", side_effect=capture):
            agentic.run_specialist(None, "test", context, 0, "intake", "Check completeness")
            self.assertEqual(captured, [])
            for role in ["coverage", "historical", "fraud"]:
                with self.assertRaises(RuntimeError):
                    agentic.run_specialist(None, "test", context, 1, role, "Investigate")
        self.assertEqual(captured[0]["context"]["findings"]["intake"]["summary"], "Intake gaps")
        self.assertNotIn("UNRELATED_PLAN", json.dumps(captured))


if __name__ == "__main__":
    unittest.main()
