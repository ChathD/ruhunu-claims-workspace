"""Regression cases for prior citations, manual estimate identity and audit replay."""
import base64
import copy
import json
import unittest
from unittest.mock import patch

from sqlalchemy import select

import test_agentic as fixtures
from backend.storage.sqlite_store import Assessment, Attachment, AttachmentExtraction, Claim, History, ToolCall, digest
from backend.agents import claude_investigation as agentic
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.agents import historical_comparer as historical
from backend.agents.intake_specialist import missing_cost_fields
from backend.evidence.uploaded_claim_documents import cost_documents_from_extractions
from backend.evidence.specialist_evidence_tools import build_tools


def submit(reference, comparable_ids=()):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "analysis", "name": "submit_analysis",
        "input": {"summary": "Compared evidence within confirmed coverage", "review_reasons": [], "evidence": [reference],
                  "amount_conclusion": "SUPPORTED", "comparable_claim_ids": list(comparable_ids), "history_leads": []}}]}


class ToolLogicTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AgenticTests()
        self.fixture.setUp()
        self.store = self.fixture.store

    def tearDown(self):
        self.fixture.tearDown()

    def completed_context(self):
        identifier, _ = self.fixture.run_claim()
        with self.store.session() as session:
            return identifier, session.get(Assessment, identifier).context

    def test_requires_garage_details_and_rejects_unsupported_repair_evidence(self):
        extraction = {
            "attachment_id": "estimate", "status": "extracted", "document_type": "repair_estimate",
            "name": "estimate.pdf", "registration": "ABC-1234", "document_date": "2026-10-16",
            "chassis_number": "IGNORE-ME", "total": "95000.00",
            "line_items": [{"description": "Replace windscreen", "amount": "95000.00"}],
            "provenance": "claimant upload", "source": "attachment:estimate",
        }
        claim = {"loss_type": "repair", "documents": cost_documents_from_extractions([extraction])}
        [estimate] = claim["documents"]
        self.assertNotIn("chassis_number", estimate)
        self.assertEqual([gap["field"] for gap in missing_cost_fields(claim)],
                         ["repair_estimate.garage_name", "repair_estimate.garage_address"])
        estimate.update(garage_name="Sample Auto Glass", garage_address="25 Sample Road, Colombo")
        self.assertEqual(missing_cost_fields(claim), [])
        unsupported_type = "repair_invoice"
        self.assertEqual(cost_documents_from_extractions([{**extraction, "document_type": unsupported_type}]), [])
        claim["documents"] = [{**estimate, "type": unsupported_type}]
        self.assertEqual([gap["field"] for gap in missing_cost_fields(claim)], ["repair_estimate"])

    def manual_estimate(self, printed_reference=True):
        identifier = self.fixture.prepare(14)
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            claim = session.get(Claim, row.claim_id)
            estimate = claim.payload["documents"][0]
            reference = estimate["id"] if printed_reference else None
            claim.payload = {**claim.payload, "source": "Manual submission", "documents": [],
                "attachment_manifest": [{"id": "uploaded-estimate", "name": "estimate.pdf"}]}
            claim.payload_hash = digest(claim.payload)
            row.context = {**row.context, "environment": "manual"}
            session.add(Attachment(id="uploaded-estimate", assessment_id=identifier, name="estimate.pdf",
                media_type="application/pdf", content=base64.b64encode(b"%PDF-1.7 fixture").decode()))
            session.add(AttachmentExtraction(attachment_id="uploaded-estimate", payload={
                "attachment_id": "uploaded-estimate", "document_reference": reference, "name": "estimate.pdf",
                "source": "attachment:uploaded-estimate", "status": "extracted", "document_type": "repair_estimate",
                "registration": estimate["registration"], "document_date": estimate["date"], "total": estimate["total"],
                "garage_name": estimate["garage_name"], "garage_address": estimate["garage_address"],
                "line_items": estimate["line_items"], "provenance": "claimant upload; unverified"}))
            return identifier, claim.id, reference

    def test_real_prior_coverage_citation_is_accepted_without_retrieval(self):
        identifier, context = self.completed_context()
        reference, = context["findings"]["coverage"]["evidence"]
        hidden = context["findings"]["fraud_initial"]["details"]["sources"]
        hidden = next(ref for ref in hidden if ref not in context["findings"]["intake"]["details"]["sources"])
        def provider(*args):
            supplied = json.loads(args[4][0]["content"])
            self.assertIn(reference, supplied["context"]["findings"]["coverage"]["details"]["sources"])
            self.assertNotIn("fraud_initial", supplied["context"]["findings"])
            schema = next(t["input_schema"] for t in args[5] if t["name"] == "submit_analysis")
            self.assertIn(reference, schema["properties"]["evidence"]["items"]["enum"])
            self.assertNotIn(hidden, schema["properties"]["evidence"]["items"]["enum"])
            [source] = supplied["required_checks"]["supporting_evidence"]
            return submit(reference, [row["id"] for row in source["result"]["records"]])
        with patch.object(agentic, "call_model", side_effect=provider) as model:
            finding = agentic.run_specialist(self.store, identifier, context, 8, "historical", "Use saved coverage evidence")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(finding["status"], "pass")
        self.assertEqual(finding["evidence"], [reference])

    def test_hidden_other_claim_and_invented_references_remain_rejected(self):
        identifier, context = self.completed_context()
        prior_intake = set(context["findings"]["intake"]["details"]["sources"])
        hidden = next(ref for ref in context["findings"]["fraud_initial"]["details"]["sources"] if ref not in prior_intake)
        other_id = self.fixture.prepare(1)
        other = agentic.record_evidence(self.store, other_id, "other", "example", {}, {"fact": "other claim"})["evidence_ref"]
        for number, reference in enumerate((hidden, other, "tool:invented")):
            with self.subTest(reference=reference):
                supplied_context = copy.deepcopy(context)
                if reference != hidden:
                    supplied_context["findings"]["coverage"]["details"]["sources"].append(reference)
                def provider(*args):
                    schema = next(t["input_schema"] for t in args[5] if t["name"] == "submit_analysis")
                    self.assertNotIn(reference, schema["properties"]["evidence"]["items"]["enum"])
                    return submit(reference)
                with patch.object(agentic, "call_model", side_effect=provider):
                    with self.assertRaisesRegex(ValueError, "Citation was not retrieved"):
                        agentic.run_specialist(self.store, identifier, supplied_context, number + 8, "historical", "Compare")

    def test_tampered_prior_evidence_stops_before_model(self):
        identifier, context = self.completed_context()
        with self.store.session.begin() as session:
            row = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                ToolCall.stage == "r1:coverage", ToolCall.tool == "execute_checks"))
            row.response = {**row.response, "evidence_ref": "tool:tampered"}
        with patch.object(agentic, "call_model") as model:
            with self.assertRaisesRegex(ValueError, "Prior evidence integrity"):
                agentic.run_specialist(self.store, identifier, context, 8, "historical", "Compare")
        model.assert_not_called()

    def test_manual_printed_reference_reaches_fraud_queue_and_replays_from_audit(self):
        identifier, claim_id, reference = self.manual_estimate()
        with patch.object(agentic, "call_model", side_effect=self.fixture.provider):
            self.fixture.workflow.run(identifier)
        package = export_audit(self.store, identifier)
        self.assertTrue(verify_audit(package))
        self.assertEqual(package["decision"]["route"], "fraud escalation")
        self.assertEqual(package["human_task"]["queue"], "fraud")
        lead = package["context"]["findings"]["historical"]["details"]["fraud_investigation"]["leads"][0]
        self.assertEqual(lead["document_reference"], reference)
        self.assertEqual(lead["document_id"], "uploaded-estimate")
        response = next(r["response"] for r in package["retrievals"] if r["tool"] == "verify_historical_leads")
        check = response["checks"][0]
        self.assertTrue(check["verified"])
        self.assertEqual(response["criteria"]["minimum_distinct_incidents"], 2)
        self.assertEqual(len(check["matching_incident_ids"]), 2)
        self.assertEqual({r["payload"]["document_reference"] for r in check["records"]}, {reference})
        self.assertTrue(all("gross_cost" in r["payload"] for r in check["records"]))
        aggregate = next(r["response"] for r in package["retrievals"] if r["stage"] == "r3:fraud" and r["tool"] == "execute_checks")
        self.assertEqual(aggregate["supporting_evidence"][0]["result"], response)
        with self.store.session.begin() as session:
            for record in check["records"]:
                row = session.get(History, record["history_id"])
                self.assertEqual(row.owner_claim_id, claim_id)
                row.payload = {**row.payload, "document_reference": "changed after decision"}
        restored = export_audit(self.store, identifier)
        self.assertTrue(verify_audit(restored))
        restored_response = next(r["response"] for r in restored["retrievals"] if r["tool"] == "verify_historical_leads")
        self.assertEqual(restored_response, response)
        restored_response["checks"][0]["records"][0]["payload"]["document_reference"] = "tampered"
        self.assertFalse(verify_audit(restored))

    def test_missing_printed_reference_never_falls_back_to_manual_file_id(self):
        identifier, claim_id, _ = self.manual_estimate(printed_reference=False)
        with self.store.session.begin() as session:
            rows = session.scalars(select(History).where(History.owner_claim_id == claim_id)).all()
            for row in rows[:2]:
                row.payload = {**row.payload, "document_reference": "uploaded-estimate"}
        tools = build_tools(self.store, identifier, "historical")
        claim = tools["read_submission"].invoke({})
        records = tools["retrieve_comparable_claims"].invoke({})["records"]
        self.assertEqual(historical.reference_leads(claim, records), [])

    def test_verification_records_rejections_scope_and_distinct_incident_criteria(self):
        identifier = self.fixture.prepare(14)
        with self.store.session.begin() as session:
            claim_id = session.get(Assessment, identifier).claim_id
            claim = session.get(Claim, claim_id)
            reference = claim.payload["documents"][0]["id"]
            rows = session.scalars(select(History).where(History.owner_claim_id == claim_id)).all()
            matching = [r for r in rows if r.payload["document_reference"] == reference]
            first, second = matching[:2]
            session.add(History(id="same-incident", owner_claim_id=claim_id, payload=copy.deepcopy(first.payload)))
            session.add(History(id="foreign", owner_claim_id="another-claim",
                payload={"document_reference": reference, "incident_id": "foreign-incident", "private": "must not leak"}))
            session.add(History(id="no-incident", owner_claim_id=claim_id,
                payload={**first.payload, "incident_id": None}))
            session.add(History(id="wrong-reference", owner_claim_id=claim_id,
                payload={**first.payload, "document_reference": "different", "incident_id": "different-incident"}))
            first_id, second_id = first.id, second.id
        history_tools = build_tools(self.store, identifier, "historical")
        before = history_tools["retrieve_comparable_claims"].invoke({})["records"]
        previous_cost = next(r["gross_cost"] for r in before if r["id"] == first_id)
        with self.store.session.begin() as session:
            row = session.get(History, first_id)
            row.payload = {**row.payload, "gross_cost": "999999.99"}
        rejected_ids = [first_id, first_id, "same-incident", "foreign", "missing", "no-incident", "wrong-reference"]
        leads = [{"document_reference": reference, "history_ids": rejected_ids, "alternative": "Clerical error"},
                 {"document_reference": reference, "history_ids": [first_id, second_id], "alternative": "Clerical error"}]
        context = {"findings": {"historical": {"details": {"fraud_investigation": {"leads": leads}}}}}
        tools = build_tools(self.store, identifier, "fraud_followup", context_snapshot=context)
        result = tools["verify_historical_leads"].invoke({})
        self.assertEqual(result["verified"], [leads[1]])
        failed, passed = result["checks"]
        self.assertFalse(failed["verified"])
        self.assertTrue(passed["verified"])
        self.assertEqual(len(failed["matching_incident_ids"]), 1)
        self.assertEqual(failed["out_of_scope_history_ids"], ["foreign"])
        self.assertEqual(failed["missing_history_ids"], ["missing"])
        self.assertEqual(len(failed["records"]), 4)
        reread = next(r["payload"] for r in passed["records"] if r["history_id"] == first_id)
        self.assertEqual(reread["gross_cost"], "999999.99")
        self.assertNotEqual(reread["gross_cost"], previous_cost)
        self.assertNotIn("must not leak", json.dumps(result))
        with self.store.session() as session:
            saved = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                ToolCall.tool == "verify_historical_leads"))
            self.assertEqual(saved.response, result)
            self.assertEqual(saved.response_hash, digest(result))
