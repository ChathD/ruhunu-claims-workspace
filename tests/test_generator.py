"""Exercise policy-independent generation and processing against uploaded policy PDFs."""
import base64
import csv
import io
import json
import ssl
import tempfile
import unittest
import uuid
from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pypdf import PdfReader
from sqlalchemy import func, select

from backend.agents import claude_investigation as agentic
from backend.agents.intake_specialist import cost_evidence
from backend.api import create_app
from backend.assurance.claim_audit_export import verify_audit
from backend.assurance.investigation_display import display_finding
from backend.evidence.policy_wording_index import CHUNK_FORMAT, index_document, read_document_page, search_document
from backend.evidence.required_claim_evidence import missing_income_loss_evidence, required_claimant_gaps
from backend.evidence.synthetic_claim_text import document_text_refs, resolve_claim_text
from backend.generation import claude_generation_client as generation_provider
from backend.generation import synthetic_claim_generator as generator
from backend.storage.sqlite_store import Assessment, Claim, DecisionRecord, Policy, PolicyChunk, PolicyDocument, ToolCall, digest
from backend.workflow import specialist_readiness
from backend.workflow.evidence_verification import coverage_checks, policy_passages, recompute_costs, verify_coverage
from backend.workflow.routing_rules import select_route
from backend.workflow.versions import V4
from backend.workflow.selected_policy import generate_policy_schedule, queue_with_policy
from scripts.debug_workflow import ScriptedModel
from test_agentic import fake_embeddings, fake_reranking, text_pdf


def claim_draft():
    return {
        "scenario": "Parked vehicle struck in Galle", "scenario_type": "parked_vehicle_collision",
        "claim": {
            "insured_name": "Anjali Fernando", "registration": "SYN-SP-CAB-1042", "policy_id": "GALLE/FICTIONAL-47",
            "vehicle": {"make": "Toyota", "model": "Vitz", "year": 2018},
            "loss_type": "repair", "loss_date": "2025-10-15", "reported_date": "2025-10-16",
            "gross_amount": "120000.00", "additional_costs": [],
            "narrative": "A van reversed into the rear bumper of my parked car in Galle.",
            "documents": [{"type": "repair_estimate", "name": "Bumper quotation.txt", "registration": "SYN-SP-CAB-1042",
                           "date": "2025-10-16", "garage_name": "Sample Motor Works", "garage_address": "Lake Road, Galle",
                           "total": "120000.00", "line_items": [{"description": "Bumper repair", "amount": "120000.00"}]}],
            "incident_details": {"weather": "monsoon rain", "initial_repair_estimate_lkr": "90000.00",
                                 "estimated_impact_speed_kmh": "15-20", "claimed_injuries": ["neck sprain"], "icd10_codes": ["S13.4"]},
        },
    }


def assessor_report(issued_on="2025-10-17", inspected_on="2025-10-16"):
    return {"type": "independent_assessor_report", "name": "Bumper assessment.txt", "registration": "SYN-SP-CAB-1042",
            "date": issued_on, "inspection_date": inspected_on, "inspection_basis": "physical_inspection",
            "author": "Sample Independent Motor Surveyor", "assessed_amount_lkr": "104000.00",
            "summary": "Inspected impact damage to the rear bumper. Repair and refinishing are supported. A replacement is not needed."}


class GenerationTransportTests(unittest.TestCase):
    def test_scheduled_loss_date_respects_inception_expiry_and_today(self):
        inception, expiry = date(2026, 1, 1), date(2026, 9, 30)
        with patch.object(generator.random, "choice", side_effect=lambda windows: windows[0]), patch.object(
                generator.random, "randint", side_effect=lambda lower, upper: lower):
            self.assertEqual(generator.sample_loss_date(inception, expiry, date(2026, 5, 1)), "2026-01-01")
        with patch.object(generator.random, "choice", side_effect=lambda windows: windows[0]), patch.object(
                generator.random, "randint", side_effect=lambda lower, upper: upper):
            self.assertEqual(generator.sample_loss_date(inception, expiry, date(2026, 5, 1)), "2026-05-01")
            self.assertEqual(generator.sample_loss_date(inception, expiry, date(2026, 10, 1)), "2026-09-30")
        with patch.object(generator.random, "choice", side_effect=lambda windows: windows[-1]), patch.object(
                generator.random, "randint", side_effect=lambda lower, upper: lower):
            self.assertEqual(generator.sample_loss_date(inception, expiry, date(2026, 10, 1)), "2026-10-01")

    def test_preview_renders_only_evidence_selected_for_each_incident(self):
        theft = claim_draft()
        theft.update(scenario="Missing parked car", scenario_type="parked_vehicle_theft")
        theft["claim"].update(loss_type="theft", documents=[], incident_details={},
                             narrative="The driver found the car missing on returning to the car park in Galle.")
        repair = claim_draft()
        repair["claim"]["documents"].append(assessor_report())
        injury = claim_draft()
        injury.update(scenario="Pedestrian injury", scenario_type="pedestrian_injury")
        injury["claim"].update(loss_type="third_party_injury", narrative="A moving car struck a pedestrian near a crossing.",
            documents=[{"type": "witness_statement", "name": "Witness account.txt", "registration": "SYN-SP-CAB-1042",
                        "date": "2025-10-16", "author": "Test Witness", "summary": "I saw the car strike the pedestrian."}],
            incident_details={"claimed_injuries": ["Ankle sprain"], "medical_intake_filed": True,
                              "medical_facility": "Sample outpatient department", "adjuster_note_filed": True,
                              "other_party": "Sample pedestrian", "liability_split_insured_pct": 60})
        cases = generator.GeneratedCases.model_validate({"cases": [theft, repair, injury]}).cases
        with patch.object(generator, "generation_client", return_value=nullcontext((None, "fixture"))), patch.object(
                generator, "request_batch", return_value=cases) as model:
            result = generator.preview_batch(3, "fixture", noise_level=0)
        self.assertIn("An empty documents list is valid", model.call_args.args[2])
        plain, costed, filed = [fixture["claim"] for fixture in result["fixtures"]]
        self.assertEqual(plain["documents"], [])
        self.assertEqual(set(plain["unstructured_inputs"]), {"fnol_transcript"})
        self.assertEqual(set(costed["unstructured_inputs"]), {"fnol_transcript", "repair_estimate_text", "independent_assessor_report"})
        report = costed["unstructured_inputs"]["independent_assessor_report"]
        self.assertIn("Assessment basis: Physical inspection", report)
        self.assertIn("Assessed amount: LKR 104,000.00", report)
        resolved = resolve_claim_text(costed)
        self.assertEqual(resolved["documents"][1]["assessed_amount_lkr"], "104000.00")
        self.assertEqual(resolved["documents"][1]["inspection_date"], "2025-10-16")
        self.assertEqual(document_text_refs(resolved)["submission:/documents/1"],
                         "submission:/unstructured_inputs/independent_assessor_report")
        [quote] = cost_evidence(resolved)[0]
        self.assertEqual(quote["total"], "120000.00")
        self.assertIn("neck sprain", costed["unstructured_inputs"]["fnol_transcript"])
        self.assertEqual([doc["type"] for doc in filed["documents"]], ["witness_statement"])
        self.assertEqual(set(filed["unstructured_inputs"]), {"fnol_transcript", "police_report_ocr", "medical_intake_note", "adjuster_notes"})
        self.assertIn("Sample outpatient department", filed["unstructured_inputs"]["medical_intake_note"])
        self.assertIn("60%", filed["unstructured_inputs"]["adjuster_notes"])
        self.assertNotIn("No police report submitted.", filed["unstructured_inputs"]["police_report_ocr"])

    def test_theft_previews_follow_assigned_assessor_report_presence(self):
        identities = [{"insured_name": f"Test policyholder {index}", "registration": "SYN-SP-CAB-1042"}
                      for index in range(6)]
        with patch.object(generator.random, "shuffle"), patch.object(generator.random, "getrandbits", return_value=1):
            plan = generator.plan_claim_scenarios(identities)
        subjects, slots = zip(*[(subject, slot) for subject, slot in zip(identities, plan) if slot["loss_type"] == "theft"])
        self.assertEqual([slot["include_assessor_report"] for slot in slots], [True, False])

        valuation = {"type": "vehicle_valuation", "name": "Vehicle valuation.txt", "registration": "SYN-SP-CAB-1042",
                     "date": "2025-10-16", "total": "3300000.00"}
        report = {**assessor_report(), "name": "Independent theft assessment.txt", "inspection_basis": "document_review",
                  "assessed_amount_lkr": "3250000.00", "summary": "Compared similar vehicles and reviewed the registration. "
                  "The car remained unrecovered, so its condition could only be judged from earlier photographs."}
        drafts = [{"scenario": "Vehicle still missing", "scenario_type": slot["scenario_type"], "claim": {
            "loss_type": "theft", "loss_date": "2025-10-15", "reported_date": "2025-10-16", "gross_amount": "3300000.00",
            "additional_costs": [], "narrative": f"The driver reported the car was missing from parking location {index}.",
            "documents": [valuation, report] if slot["include_assessor_report"] else [valuation],
        }} for index, slot in enumerate(slots)]
        wrong = deepcopy(drafts)
        wrong[1]["claim"]["documents"].append(wrong[0]["claim"]["documents"].pop())
        responses = [generator.GeneratedIncidents.model_validate({"cases": cases}).cases for cases in (wrong, drafts)]
        with patch.object(generator, "generation_client", return_value=nullcontext((None, "fixture"))), \
                patch.object(generator, "request_batch", side_effect=responses) as model:
            result = generator.preview_batch(2, "fixture", identities=subjects, scenario_plan=slots, noise_level=0)

        self.assertIn("Case 1 must include the independent_assessor_report", model.call_args.args[2])
        self.assertIn("Case 2 must omit the independent_assessor_report", model.call_args.args[2])
        attached, missing = [fixture["claim"] for fixture in result["fixtures"]]
        self.assertIn("Assessment basis: Document review", attached["unstructured_inputs"]["independent_assessor_report"])
        self.assertNotIn("independent_assessor_report", missing["unstructured_inputs"])
        for claim in (attached, missing):
            self.assertEqual([doc["total"] for doc in cost_evidence(resolve_claim_text(claim))[0]], ["3300000.00"])

    def test_preview_handles_revised_and_repeated_quotation_amounts(self):
        for original, supplement in (("135000.00", "120000.00"), ("120000.00", None)):
            with self.subTest(original=original):
                draft = claim_draft()
                draft["claim"]["incident_details"]["initial_repair_estimate_lkr"] = original
                cases = [generator.GeneratedCase.model_validate(draft)]
                with patch.object(generator, "generation_client", return_value=nullcontext((None, "fixture"))), \
                        patch.object(generator, "request_batch", return_value=cases):
                    preview = generator.preview_batch(1, "fixture", noise_level=0)
                [fixture] = preview["fixtures"]
                labels = fixture["ground_truth_labels"]
                quote = fixture["claim"]["unstructured_inputs"]["repair_estimate_text"]
                self.assertEqual(labels["initial_repair_estimate_lkr"], original)
                self.assertEqual(labels["supplement_repair_estimate_lkr"], supplement)
                self.assertEqual("Original quotation:" in quote, supplement is not None)

    def test_preview_retries_invalid_draft_and_preserves_total_only_estimate(self):
        draft = claim_draft()
        estimate = draft["claim"]["documents"][0]
        del estimate["line_items"]
        invalid = claim_draft()
        invalid["claim"]["documents"].append({
            "type": "police_report", "name": "Station report", "registration": "SYN-SP-CAB-1042",
            "date": "2025-10-16", "summary": "The owner reported the bumper damage after the incident.",
        })
        responses = iter((invalid, draft))
        prompts = []

        def generate(request):
            prompts.append(json.loads(request.content)["messages"][0]["content"])
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "name": generator.GENERATION_TOOL, "input": {"cases": [next(responses)]},
            }]})

        with httpx.Client(transport=httpx.MockTransport(generate)) as client, \
                patch.object(generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            preview = generator.preview_batch(1, "fixture", noise_level=0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("cases.0.claim.documents.1.police_report.station", prompts[1])
        claim = preview["fixtures"][0]["claim"]
        self.assertEqual(claim["documents"][0]["total"], estimate["total"])
        self.assertFalse(claim["documents"][0].get("line_items"))
        self.assertNotIn("Itemised charges:", claim["unstructured_inputs"]["repair_estimate_text"])

    def test_connect_failure_is_retried_before_a_successful_response(self):
        attempts = []

        def respond(request):
            attempts.append(request)
            if len(attempts) == 1:
                raise httpx.ConnectTimeout("fixture connection timeout", request=request)
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "name": generator.GENERATION_TOOL,
                "input": {"cases": [claim_draft()]},
            }]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            cases = generation_provider.request_batch(client, "fixture", "fixture", generator.GENERATION_TOOL,
                                                      generator.GeneratedCases, 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(cases[0].claim.insured_name, "Anjali Fernando")

    def test_overload_and_truncated_tool_output_recover(self):
        token_limits = []

        def respond(request):
            payload = json.loads(request.content)
            token_limits.append(payload["max_tokens"])
            if len(token_limits) == 1:
                return httpx.Response(529, headers={"retry-after": "0"})
            if len(token_limits) == 2:
                return httpx.Response(200, json={"stop_reason": "max_tokens", "content": []})
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "name": generator.GENERATION_TOOL, "input": {"cases": [claim_draft()]},
            }]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            cases = generation_provider.request_batch(client, "fixture", "fixture", generator.GENERATION_TOOL,
                                                      generator.GeneratedCases, 1)
        self.assertEqual(token_limits, [6000, 6000, 12000])
        self.assertEqual(cases[0].claim.insured_name, "Anjali Fernando")

    def test_read_timeout_and_invalid_certificate_report_the_failure_without_retry(self):
        for error, message in (
            (httpx.ReadTimeout("fixture response timeout"), "Claude did not respond before the generation timeout"),
            (httpx.ConnectError("fixture SSL error"), "could not verify Anthropic's TLS certificate"),
        ):
            with self.subTest(error=type(error).__name__):
                if isinstance(error, httpx.ConnectError):
                    error.__cause__ = ssl.SSLCertVerificationError("fixture untrusted certificate")
                attempts = []

                def respond(request):
                    attempts.append(request)
                    raise error

                with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                        self.assertLogs("backend.generation.claude_generation_client", level="WARNING") as captured:
                    with self.assertRaisesRegex(generation_provider.GenerationError, message):
                        generation_provider.request_batch(client, "fixture", "fixture", generator.GENERATION_TOOL,
                                                          generator.GeneratedCases, 1)
                self.assertEqual(len(attempts), 1)
                self.assertIn(type(error).__name__, captured.output[0])


class CoverageDisplayTests(unittest.TestCase):
    def test_confirmed_schedule_explains_all_open_coverage_checks(self):
        finding = {"agent": "coverage", "status": "review", "details": {
            "coverage_status": "UNRESOLVED", "schedule_confirmed": True,
            "policy_rag": {"cited": True, "applicability": "applicable"},
            "coverage_items": [
                {"item": "Brand-new parts contribution — Appendix", "outcome": "undetermined",
                 "reason": "Two contribution rates apply. Staff should verify which rate applies to each replacement part."},
                {"item": "Condition of average", "outcome": "undetermined", "reason": "A market valuation is missing."},
                {"item": "Prior written approval", "outcome": "undetermined",
                 "reason": "The policy requires written consent. Prior written approval has not been formally established."},
                {"item": "Enhanced towing", "outcome": "not_covered", "reason": "The extension was not purchased."},
            ],
            "policy_conflicts": [
                {"item": "Recovery limit", "materiality": "material", "impact": "The payable recovery limit remains unresolved."},
            ],
        }}
        details = display_finding(finding, {}, {"document_kind": "wording"})["details"]
        self.assertIn("The policy schedule is confirmed", details["coverage_unconfirmed_reason"])
        self.assertEqual(details["coverage_unconfirmed_checks"], [
            {"label": "Policyholder’s share of new parts costs", "reason": "Verify which rate applies to each replacement part."},
            {"label": "Vehicle market value and any reduction in payment", "reason": "A market valuation is missing."},
            {"label": "Prior written approval", "reason": "Prior written approval has not been formally established."},
            {"label": "Recovery limit", "reason": "The payable recovery limit remains unresolved."},
        ])

    def test_review_note_can_identify_a_blocker_when_item_results_are_missing(self):
        recorded = "No submitted document identifies the driver responsible for the collision."
        finding = {"agent": "coverage", "status": "review", "details": {
            "coverage_status": "UNRESOLVED", "review_reasons": ["The two repair estimates reconcile.", recorded],
        }}
        details = display_finding(finding, {})["details"]
        self.assertEqual(details["coverage_unconfirmed_reason"], recorded)

    def test_confirmed_or_excluded_loss_has_no_unconfirmed_note(self):
        for status in ("COVERED", "NOT_COVERED", "INACTIVE", "PENDING_ANALYSIS"):
            with self.subTest(status=status):
                finding = {"agent": "coverage", "status": "review", "details": {"coverage_status": status}}
                details = display_finding(finding, {}, {"document_kind": "wording"})["details"]
                self.assertNotIn("coverage_unconfirmed_reason", details)
                self.assertNotIn("coverage_unconfirmed_checks", details)

    def test_product_summary_and_processing_failure_have_specific_explanations(self):
        finding = {"agent": "coverage", "status": "review", "details": {"coverage_status": "UNRESOLVED"}}
        displayed = display_finding(finding, {}, {"document_kind": "product_summary"})["details"]
        self.assertIn("Obtain the full policy wording and individual schedule", displayed["coverage_unconfirmed_reason"])
        finding["details"]["error_type"] = "ReadTimeout"
        displayed = display_finding(finding, {}, {"document_kind": "wording"})["details"]
        self.assertIn("timed out", displayed["coverage_unconfirmed_reason"])
        self.assertNotIn("coverage_unconfirmed_checks", displayed)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="claim-pdf-"))
        self.app = create_app(Path(self.directory))
        self.client = self.enterContext(TestClient(self.app))
        self.worker = self.app.state.worker
        self.worker.close()
        self.store = self.app.state.store
        self.enterContext(patch("backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings))
        self.enterContext(patch("backend.evidence.policy_wording_index.rerank_scores", side_effect=fake_reranking))
        self.enterContext(patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value={"status": "no_observation"}))
        self.login("staff")

    def login(self, role):
        response = self.client.post("/api/auth/login", json={"email": f"{role}@ruhunu.local", "password": f"{role.capitalize()}Demo!2026"})
        self.assertEqual(response.status_code, 200)

    def save_claim(self, draft=None):
        draft = draft or claim_draft()
        def generate(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["tool_choice"]["name"], generator.GENERATION_TOOL)
            self.assertNotIn("Saved policy", payload["messages"][0]["content"])
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{"type": "tool_use", "name": generator.GENERATION_TOOL,
                                   "input": {"cases": [draft]}}]})

        with httpx.Client(transport=httpx.MockTransport(generate)) as client, \
                patch.object(generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            preview = generator.preview_batch(1, str(uuid.uuid4()), noise_level=0)
        result = generator.commit_batch(self.store, self.app.state.labels_path, preview)
        self.assertEqual(result["status"], "awaiting_policy")
        return preview, result["assessment_ids"][0]

    def upload(self, text, version, *, content=None):
        self.login("admin")
        response = self.client.post("/api/policy-documents", json={
            "title": "Motor wording", "product": "MOTOR", "version": version, "filename": "wording.pdf", "media_type": "application/pdf",
            "content": base64.b64encode(content if content is not None else text_pdf(text)).decode(),
        })
        self.assertEqual(response.status_code, 202, response.text)
        self.login("staff")
        return response.json()["id"]

    def start_claim(self, identifier, policy_id, mode="anthropic"):
        with patch.dict("os.environ", {"CLAIMS_LLM_MODE": mode, "ANTHROPIC_API_KEY": "fixture"}):
            return queue_with_policy(self.store, identifier, policy_id, "legacy-comparison-test")

    def generate_schedule(self, identifier, policy_id):
        return generate_policy_schedule(self.store, identifier, policy_id, "legacy-comparison-test")

    def run_against_pdf(self, identifier, policy_id, clause, *, outcome="covered", requirement_quote=None,
                        applicability="applicable", document_refs=((),), document_inconsistencies=(), liability_confirmed=False,
                        additional_coverage=None, judge_calls=None):
        scripted = ScriptedModel()

        def provider(store, assessment_id, stage, system, messages, tools, choice):
            if not stage.endswith(":coverage"):
                if stage.endswith(":fraud"):
                    self.assertEqual({tool["name"] for tool in tools}, {"submit_analysis"})
                if stage == "judge" and judge_calls is not None:
                    judge_calls.append(json.loads(messages[0]["content"]))
                return scripted(store, assessment_id, stage, system, messages, tools, choice)
            if len(messages) == 1:
                supplied = json.loads(messages[0]["content"])
                self.assertEqual(supplied["submission"]["policy_id"], "GALLE/FICTIONAL-47")
                self.assertEqual(supplied["context"]["selected_policy"]["id"], policy_id)
                return {"content": [{"type": "tool_use", "id": "search", "name": "search_policy_documents",
                                     "input": {"query": "collision cover and required documents", "topic": "claim_conditions"}}]}
            result = json.loads(messages[-1]["content"][0]["content"])["result"]
            self.assertEqual(result["document_id"], policy_id)
            if len(messages) == 3:
                self.assertEqual(result["topic"], "claim_conditions")
                return {"content": [{"type": "tool_use", "id": "page", "name": "read_policy_page",
                                     "input": {"page": result["passages"][0]["page"]}}]}
            [passage] = result["passages"]
            self.assertIn(clause, passage["text"])
            packet = json.loads(messages[0]["content"])
            third_party = {}
            if "claimed_costs" in packet:
                third_party = {"liability_evidence_refs": list(dict.fromkeys(ref for refs in document_refs for ref in refs))
                               if liability_confirmed else [],
                               "cost_assessments": [{"line_ref": line["line_ref"]} for line in packet["claimed_costs"]]}
            return {"content": [{"type": "tool_use", "id": "done", "name": "submit_analysis", "input": {
                "summary": "The selected wording addresses the reported collision.", "review_reasons": [],
                "policy_applicability": applicability, "evidence": [passage["source"]], "coverage_conclusion": outcome.upper(),
                "liability_confirmed": liability_confirmed,
                "document_inconsistencies": list(document_inconsistencies),
                "coverage_items": [{"item": "Third-party injury" if third_party else "Collision repair",
                                    "outcome": outcome, "policy_term": clause, "passage_refs": [passage["source"]],
                                    "reason": "The police report attributes the collision to the insured driver." if liability_confirmed
                                    else "The estimate describes collision repair.",
                                    "document_refs": list(refs)} for refs in document_refs],
                "required_documents": [{"document_type": "police_report", "passage_ref": passage["source"],
                                        "requirement_quote": requirement_quote or "A police report must be supplied for every collision.",
                                        "reason": "The report corroborates how the collision occurred."}],
                **third_party,
                **(additional_coverage(packet) if additional_coverage else {}),
            }}]}

        with patch.object(agentic, "call_model", side_effect=provider):
            self.worker.workflow.run(identifier)
        response = self.client.get(f"/api/claims/{identifier}/audit")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(verify_audit(response.json()))
        return response.json()

    def test_live_generator_requires_a_schedule_and_legacy_exports_remain_available(self):
        self.assertEqual(self.client.post("/api/generator/people-policies", json={"count": 1}).status_code, 404)
        self.assertEqual(self.client.post("/api/generator/preview", json={"count": 1}).status_code, 409)
        preview, identifier = self.save_claim()
        self.worker.workflow.run(identifier)
        with self.store.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Policy)), 0)
            self.assertEqual(session.get(Assessment, identifier).status, "awaiting_policy")
            self.assertIsNone(session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier)))
        draft = preview["fixtures"][0]["claim"]
        self.assertNotIn("ground_truth_labels", draft)
        self.assertEqual(draft["policy_id"], "GALLE/FICTIONAL-47")
        self.assertIn("LKR 90,000.00", draft["unstructured_inputs"]["repair_estimate_text"])
        self.assertEqual(self.client.get(f"/api/portal/claims/{identifier}").json()["status"], "awaiting processing")
        batch = f"/api/generator/batches/{preview['batch_id']}"
        self.assertEqual(self.client.post(batch + "/start").status_code, 409)
        [record] = self.client.get(batch + "/dataset.json").json()
        self.assertEqual(record["ground_truth_labels"]["submitted_policy_number"], "GALLE/FICTIONAL-47")
        self.assertEqual(record["ground_truth_labels"]["final_repair_estimate_lkr"], "120000.00")
        [row] = csv.DictReader(io.StringIO(self.client.get(batch + "/dataset.csv").text))
        self.assertIn(draft["unstructured_inputs"]["repair_estimate_text"], row["raw_combined_text"])
        self.assertEqual(self.client.get("/api/evaluation").json()["pending"], 0)

    def test_staff_reindexes_legacy_pdf_and_preserves_passage_citations(self):
        policy_id = self.upload("Collision is covered.", "legacy")
        with self.store.session.begin() as session:
            doc = session.get(PolicyDocument, policy_id)
            doc.status = "ready"
            doc.embedding = {"model": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384,
                             "page_count": 1, "ocr_pages": [1], "document_kind": "wording"}
            session.add(PolicyChunk(id=f"{policy_id}:0", document_id=policy_id, page=1,
                                    text="Collision is covered."))
            original_hash = doc.file_hash
        path = f"/api/policy-documents/{policy_id}"
        self.assertTrue(self.client.get(path).json()["requires_reindex"])
        self.assertEqual(self.client.get("/api/policy-documents?selectable=true").json()["documents"], [])
        self.assertEqual(search_document(self.store, policy_id, "collision")["status"], "unavailable")
        before = read_document_page(self.store, policy_id, 1)
        self.assertEqual(self.client.post(path + "/retry").status_code, 202)
        index_document(self.store, policy_id)
        [ready] = self.client.get("/api/policy-documents?selectable=true").json()["documents"]
        self.assertEqual(ready["file_hash"], original_hash)
        self.assertEqual(ready["embedding"]["model"], "text-embedding-3-small")
        self.assertEqual(ready["embedding"]["storage"], "sqlite-vec")
        self.assertEqual(ready["embedding"]["chunk_format"], CHUNK_FORMAT)
        self.assertFalse(ready["requires_reindex"])
        current_page = read_document_page(self.store, policy_id, 1)
        self.assertEqual([p["source"] for p in current_page["passages"]], [f"chunk:{policy_id}:{CHUNK_FORMAT}:0"])
        with self.store.session() as session:
            selected = {"id": policy_id, "file_hash": original_hash, "version": "legacy"}
            saved = ToolCall(tool="read_policy_page", response={"result": before})
            [historical] = policy_passages(session, selected, [saved]).values()
            self.assertEqual(historical, {"id": f"chunk:{policy_id}:0", "page": 1,
                                          "text": "Collision is covered.", "ocr": True})
        [passage] = search_document(self.store, policy_id, "collision")["passages"]
        self.assertEqual(passage["source"], f"chunk:{policy_id}:{CHUNK_FORMAT}:0")

    def test_selection_requires_an_indexed_pdf_and_uses_only_the_chosen_version(self):
        preview, identifier = self.save_claim()
        covered = "Collision damage is covered under this motor wording."
        chosen = self.upload(covered, "2026.1")
        other = self.upload("Collision damage is excluded.", "2026.2")
        with self.assertRaisesRegex(ValueError, "current OpenAI passage index"):
            self.start_claim(identifier, chosen)
        with self.assertRaisesRegex(ValueError, "current OpenAI passage index"):
            self.generate_schedule(identifier, chosen)
        self.assertEqual(self.client.get("/api/policy-documents?selectable=true").json()["documents"], [])
        for policy_id in (chosen, other):
            index_document(self.store, policy_id)
        self.login("admin")
        self.assertEqual(self.client.post(f"/api/claims/{identifier}/process").status_code, 403)
        self.login("staff")
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            claim.received_at = "2020-01-01T00:00:00+00:00"
            session.add(Policy(id=claim.policy_id, payload={"covered_losses": []}))
        with self.assertRaisesRegex(ValueError, "Generate the policy schedule"):
            self.start_claim(identifier, chosen)
        self.generate_schedule(identifier, other)
        with self.assertRaisesRegex(ValueError, "Generate the policy schedule"):
            self.start_claim(identifier, chosen)
        schedule = self.generate_schedule(identifier, chosen)
        self.assertEqual(schedule["status"], "draft")
        self.assertEqual((schedule["country"], schedule["currency"]), ("Sri Lanka", "LKR"))
        self.assertEqual(schedule["reported"]["insured_name"], "Anjali Fernando")
        self.assertEqual(schedule["reported"]["policy_number"], "GALLE/FICTIONAL-47")
        self.assertEqual(schedule["wording"]["id"], chosen)
        [reference] = schedule["wording_references"]
        self.assertEqual((reference["page"], reference["text"]), (1, covered))
        self.assertEqual(self.generate_schedule(identifier, chosen), schedule)
        detail = self.client.get(f"/api/portal/claims/{identifier}").json()
        self.assertEqual(detail["status"], "awaiting processing")
        self.assertEqual(detail["policy_schedule"], schedule)
        self.assertIsNone(detail["selected_policy"])
        draft_preview = self.client.get(f"/api/portal/claims/{identifier}/policy-schedule?preview=true")
        self.assertEqual(draft_preview.status_code, 200)
        preview_text = "\n".join(page["text"] for page in draft_preview.json()["pages"])
        for expected in ("Draft motor policy schedule", "Anjali Fernando", "Toyota Vitz", "Individual terms to confirm"):
            self.assertIn(expected, preview_text)
        self.worker.workflow.run(identifier)
        with self.store.session() as session:
            self.assertIsNone(session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier)))
        self.assertEqual(self.start_claim(identifier, chosen), "queued")
        self.assertEqual(self.start_claim(identifier, chosen), "queued")
        with self.assertRaisesRegex(ValueError, "no longer waiting"):
            self.start_claim(identifier, other)
        with patch.object(specialist_readiness, "assess", wraps=specialist_readiness.assess) as model_readiness:
            package = self.run_against_pdf(identifier, chosen, covered, document_inconsistencies=[{
                "reason": "The repair estimate omits the chassis number. The claim and estimate both show the same registration.",
                "document_refs": ["submission:/documents/0", "submission:/unstructured_inputs/repair_estimate_text"],
            }])
        model_readiness.assert_called_once()
        terminal_plan = package["context"]["readiness_history"]["2"]
        self.assertEqual(terminal_plan["specialists"]["fraud"]["state"], "COMPLETED")
        self.assertEqual(terminal_plan["specialists"]["historical"]["state"], "NOT_APPLICABLE")
        self.assertEqual(terminal_plan["next_action"], "FINISH")
        self.assertEqual(package["submission"], preview["fixtures"][0]["claim"])
        self.assertEqual(package["policy_wording"]["id"], chosen)
        self.assertEqual(package["context"]["policy_schedule"], schedule)
        self.assertEqual(package["policy_wording"]["media_type"], "application/pdf")
        self.assertEqual(package["context"]["findings"]["coverage"]["details"]["coverage_status"], "COVERED")
        self.assertEqual(package["context"]["findings"]["coverage"]["details"]["document_inconsistencies"], [])
        self.assertEqual(package["decision"]["reason_codes"], ["POLICY_SCHEDULE_REVIEW"])
        self.assertIsNone(package["decision"]["payable_amount"])
        self.assertEqual(sum(event["kind"] == "policy_selected" for event in package["workflow_events"]), 1)
        report = self.client.get(f"/api/portal/claims/{identifier}/investigation.pdf")
        self.assertEqual(report.status_code, 200)
        report_text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(report.content)).pages)
        self.assertIn("Adjuster review", report_text)
        self.assertIn("Specialist findings", report_text)
        self.assertIn("Coverage /", report_text)
        decision_text = report_text.split("Decision summary", 1)[1].split("Specialist findings", 1)[0]
        self.assertIn("The adjuster must verify the settlement calculation", " ".join(decision_text.split()))

    def test_selected_repairs_record_the_market_value_assumption_in_the_decision_and_report(self):
        _, identifier = self.save_claim()
        clause = "Collision damage is covered. Payment is capped at the lower of the scheduled value and market value."
        selected = self.upload(clause, "valuation-cap")
        index_document(self.store, selected)
        self.generate_schedule(identifier, selected)
        self.assertEqual(self.start_claim(identifier, selected), "queued")
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            schedule = {**assessment.context["policy_schedule"], "id": "schedule-cap", "policy_number": "TEST-CAP",
                        "starts_on": "2025-01-01", "ends_on": "2025-12-31", "vehicle": {"registration": "SYN-SP-CAB-1042"},
                        "cover": [{"loss_type": "repair", "limit_basis": "market_value", "conditions": clause}]}
            assessment.context = {**assessment.context, "policy_schedule": schedule}
        benchmark = {"status": "found", "currency": "LKR", "asking_price_lkr": "8495000", "observation_date": "2025-10-14",
                     "source": "PriceMart.lk", "make": "Toyota", "model": "Vitz", "year": 2018}
        cap_gap = "Market value is missing to check the policy cap."
        approval_gap = "Prior written insurer approval before repairs has not been confirmed."

        def coverage_output(packet):
            supplied = packet["required_checks"]
            self.assertEqual(supplied["result"]["details"]["settlement_market_cap"]["status"], "unlikely_to_limit")
            self.assertIn("retrieve_vehicle_market_value", [entry["tool"] for entry in supplied["supporting_evidence"]])
            return {"review_reasons": [cap_gap, approval_gap]}

        with patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value=benchmark) as lookup:
            package = self.run_against_pdf(identifier, selected, clause, additional_coverage=coverage_output)
        lookup.assert_called_once()
        coverage = package["context"]["findings"]["coverage"]
        self.assertIn(coverage["details"]["settlement_market_cap"]["evidence_ref"], coverage["evidence"])
        self.assertEqual(coverage["details"]["review_reasons"], [approval_gap])
        self.assertIn("unlikely to reduce payment", package["decision"]["justification"])
        self.assertIsNone(package["decision"]["payable_amount"])
        report = self.client.get(f"/api/portal/claims/{identifier}/investigation.pdf")
        self.assertEqual(report.status_code, 200)
        report_text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(report.content)).pages)
        self.assertIn("Decision summary", report_text)
        self.assertIn("Coverage /", report_text)
        self.assertNotIn("Market value assumption", report_text)
        self.assertNotIn("LKR 8,495,000.00", report_text)
        self.assertIn(approval_gap, report_text)
        self.assertNotIn(cap_gap, report_text)

    def test_retry_preserves_the_schedule_and_generates_one_for_older_assessments(self):
        pdf = self.upload("Collision damage is covered. Compulsory excess is stated in the schedule.", "2026.1")
        index_document(self.store, pdf)
        for historical in (False, True):
            with self.subTest(historical=historical):
                preview, identifier = self.save_claim()
                schedule = self.generate_schedule(identifier, pdf)
                self.assertEqual(self.start_claim(identifier, pdf), "queued")
                with self.store.session.begin() as session:
                    assessment = session.get(Assessment, identifier)
                    assessment.status = "failed"
                    if historical:
                        assessment.context = {key: value for key, value in assessment.context.items() if key != "policy_schedule"}
                response = self.client.post(f"/api/portal/claims/{identifier}/retry")
                self.assertEqual(response.status_code, 202, response.text)
                replacement_path = f"/api/portal/claims/{response.json()['id']}"
                replacement = self.client.get(replacement_path).json()
                saved = replacement["policy_schedule"]
                overview = self.client.get(replacement_path + "?overview=true").json()
                self.assertEqual(overview["scenario_title"], preview["fixtures"][0]["scenario"])
                if historical:
                    self.assertEqual(saved["wording"], schedule["wording"])
                    self.assertEqual(saved["reported"], schedule["reported"])
                else:
                    self.assertEqual(saved, schedule)

    def test_selected_pdf_can_supply_an_explicit_document_requirement(self):
        _, identifier = self.save_claim()
        clause = "A police report must be supplied for every collision."
        chosen = self.upload(f"Collision damage is covered. {clause}", "2026.1")
        index_document(self.store, chosen)
        self.generate_schedule(identifier, chosen)
        self.assertEqual(self.start_claim(identifier, chosen), "queued")
        package = self.run_against_pdf(identifier, chosen, clause)
        self.assertEqual(package["decision"]["route"], "request more information")
        self.assertEqual(package["decision"]["information_required"][0]["document"], "Police report")
        staff = self.client.get(f"/api/portal/claims/{identifier}").json()
        details = staff["findings"]["coverage"]["details"]
        self.assertIn("police report", details["coverage_unconfirmed_reason"])
        self.assertIn({"label": "Police report", "reason": package["decision"]["information_required"][0]["reason"]},
                      details["coverage_unconfirmed_checks"])

    def test_supported_injury_claim_requests_evidence_for_unsubstantiated_lost_income(self):
        draft = claim_draft()
        claim = draft["claim"]
        claim.update(loss_type="third_party_injury", gross_amount="185000.00",
                     narrative="The insured driver struck and injured a pedestrian in Galle.", documents=[
            {"type": "third_party_demand", "name": "Injury demand", "registration": claim["registration"],
             "date": "2025-10-16", "total": "185000.00", "line_items": [
                 {"description": "Loss of income for six days of work missed", "amount": "15000.00"},
                 {"description": "General damages", "amount": "170000.00"}]},
            {"type": "police_report", "name": "Police report", "registration": claim["registration"],
             "date": "2025-10-16", "station": "Galle", "summary": "The insured driver struck the pedestrian."},
        ])
        _, identifier = self.save_claim(draft)
        clause = "Third-party injury liability covers bodily injury caused by a collision."
        chosen = self.upload(clause, "income-loss")
        index_document(self.store, chosen)
        self.generate_schedule(identifier, chosen)
        self.assertEqual(self.start_claim(identifier, chosen), "queued")
        package = self.run_against_pdf(identifier, chosen, clause, liability_confirmed=True,
                                       document_refs=(("submission:/unstructured_inputs/police_report_ocr",),))
        self.assertEqual(package["decision"]["route"], "request more information")
        self.assertEqual(package["decision"]["reason_codes"], ["MISSING_EVIDENCE"])
        self.assertIsNone(package["decision"]["payable_amount"])
        staff = self.client.get(f"/api/portal/claims/{identifier}").json()
        self.assertEqual(staff["status"], "request more information")
        requests = staff["information_required"]
        self.assertEqual({item["document"] for item in requests}, {"Evidence of time unable to work", "Evidence of earnings"})
        self.assertTrue(all("LKR 15,000.00" in item["reason"] for item in requests))

    def test_income_loss_requests_only_the_unfiled_evidence_for_a_supported_charge(self):
        claim = {"loss_type": "third_party_injury", "currency": "LKR", "documents": [
            {"type": "third_party_demand", "line_items": [{"description": "Lost wages", "amount": "15000"}]},
            {"type": "work_incapacity_evidence"},
        ]}
        self.assertEqual([gap["field"] for gap in missing_income_loss_evidence(claim, [])], ["earnings_evidence"])
        excluded = [{"item": "Lost wages", "outcome": "not_covered"}]
        self.assertEqual(missing_income_loss_evidence(claim, excluded), [])
        claim["uploaded_evidence"] = [{"document_type": "earnings_evidence", "status": "extracted"}]
        self.assertEqual(missing_income_loss_evidence(claim, []), [])
        claim.pop("uploaded_evidence")
        claim["documents"][0]["line_items"][0]["description"] = "Medical expenses"
        self.assertEqual(missing_income_loss_evidence(claim, []), [])

    def test_clm_0041_cost_requests_reach_publication_with_unconfirmed_liability(self):
        draft = claim_draft()
        claim = draft["claim"]
        registration = claim["registration"]
        costs = [("Hospital treatment", "148000.00"), ("Orthopaedic consultation", "8500.00"),
                 ("Physiotherapy", "25000.00"), ("Medication", "14600.00"), ("Loss of earnings", "133000.00"),
                 ("General damages", "125000.00"), ("Ambulance transport", "18500.00"), ("Follow-up transport", "12400.00")]
        record = {"registration": registration, "date": "2025-10-16"}
        claim.update(loss_type="third_party_injury", gross_amount="485000.00", additional_costs=[],
                     narrative="The insured driver reports contact with a cyclist. The police investigation is ongoing.",
                     documents=[
                         {**record, "type": "third_party_demand", "name": "Injury demand", "total": "485000.00",
                          "line_items": [{"description": description, "amount": amount} for description, amount in costs]},
                         {**record, "type": "police_report", "name": "Police report", "station": "Galle",
                          "summary": "The cause and fault remain under investigation."},
                         {**record, "type": "incident_report", "name": "Hospital admission record", "author": "Hospital clerk",
                          "summary": "Dates of admission and treatment were recorded."},
                         {**record, "type": "work_incapacity_evidence", "name": "Certificate of incapacity", "author": "Attending physician",
                          "summary": "The claimant was unable to work for 38 days."},
                         {**record, "type": "earnings_evidence", "name": "Claimant earnings declaration", "author": "Claimant",
                          "summary": "The claimant declares daily earnings of LKR 3,500 without supporting records."},
                     ])
        _, identifier = self.save_claim(draft)
        clause = "Third-party injury liability covers bodily injury caused by a collision."
        wording = f"{registration} Comprehensive 01/01/2025 to 31/12/2025. {clause}"
        chosen = self.upload(wording, "third-party-costs")
        index_document(self.store, chosen)
        self.generate_schedule(identifier, chosen)
        self.assertEqual(self.start_claim(identifier, chosen), "queued")
        judge_calls = []

        def additional_coverage(packet):
            self.assertEqual([line["claimed_amount"] for line in packet["claimed_costs"]], [amount for _, amount in costs])
            requests = {"Hospital treatment": ("claimant", "Official itemised hospital bill", "The admission record does not verify the amount."),
                        "Loss of earnings": ("claimant", "Independent income evidence", "The earnings rate needs independent corroboration.")}
            assessments = []
            for line in packet["claimed_costs"]:
                description = line["description"]
                owner, document, reason = requests.get(description, ("internal", f"Assessment of {description.lower()}",
                                                                       "An adjuster must verify the amount."))
                assessments.append({"line_ref": line["line_ref"], "policy_item": "Third-party injury",
                                    "evidence_requests": [{"owner": owner, "document": document, "reason": reason}]})
            return {"liability_evidence_refs": [], "required_documents": [],
                    "liability_evidence_requests": [{"owner": "internal", "document": "Liability assessment",
                                                       "reason": "The police report does not establish responsibility."}],
                    "cost_assessments": assessments}

        audit = self.run_against_pdf(identifier, chosen, clause, additional_coverage=additional_coverage, judge_calls=judge_calls)
        details = audit["context"]["findings"]["coverage"]["details"]
        self.assertEqual(details["coverage_status"], "UNRESOLVED")
        assessment = details["third_party_assessment"]
        self.assertEqual(assessment["liability"]["status"], "unconfirmed")
        self.assertEqual({cost["policy_applicability"] for cost in assessment["costs"]}, {"applies"})
        self.assertEqual({cost["amount_status"] for cost in assessment["costs"]}, {"unverified"})
        documents = {"Official itemised hospital bill", "Independent income evidence"}
        self.assertEqual(len(judge_calls), 1)
        self.assertTrue(judge_calls[0]["use_committed_evidence_requests"])
        self.assertEqual({gap["document"] for gap in judge_calls[0]["committed_evidence_requests"]}, documents)
        self.assertEqual(audit["decision"]["route"], "request more information")
        self.assertEqual({gap["document"] for gap in audit["decision"]["information_required"]}, documents)
        self.assertIsNone(audit["decision"]["payable_amount"])
        self.assertEqual((audit["human_task"]["queue"], audit["human_task"]["status"]), ("adjuster", "pending"))

        draft_context = deepcopy(audit["context"])
        draft_coverage = draft_context["findings"]["coverage"]["details"]
        draft_coverage.update(coverage_status="COVERED", financial={"payable": "485000.00"})
        for cost in draft_coverage["third_party_assessment"]["costs"]:
            cost["evidence_requests"] = []
        review = select_route(audit["submission"], draft_context, [])
        self.assertEqual(review.route.value, "adjuster review")
        self.assertIsNone(review.payable_amount)

        # The v4 auditor can reject an individual request while keeping other verified requests.
        finding = deepcopy(audit["context"]["findings"]["coverage"])
        context = audit["context"]
        finding["details"]["schedule_candidate"] = {"registration": registration, "starts_on": "2025-01-01",
                                                    "ends_on": "2025-12-31", "cover": "comprehensive",
                                                    "passage_refs": assessment["costs"][0]["passage_refs"]}
        with self.store.session() as session:
            calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier)).all()
            passages = policy_passages(session, context["selected_policy"], calls)
        checks = coverage_checks(finding, audit["submission"], passages)
        self.assertIn("schedule", checks)
        self.assertIn("third_party_cost:0", checks)
        self.assertIn("third_party_cost:4", checks)
        verify_coverage(finding, audit["submission"], context["selected_policy"], checks, set(checks) - {"third_party_cost:0"},
                        recompute_costs(audit["submission"]))
        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
        self.assertEqual([gap["document"] for gap in required_claimant_gaps(finding["details"])], ["Independent income evidence"])

    def test_unresolved_third_party_towing_keeps_receipt_request_and_separates_citations(self):
        draft = claim_draft()
        claim = draft["claim"]
        record = {"registration": claim["registration"], "date": "2025-10-16"}
        claim.update(loss_type="third_party_property_damage", gross_amount="187500.00", incident_details={},
                     additional_costs=[{"amount": "6500.00", "category": "towing", "reason": "Tow the third-party car"}],
                     documents=[{**record, "type": "third_party_demand", "name": "Repair demand", "total": "175000.00",
                                 "line_items": [{"description": "Body repair", "amount": "175000.00"}]},
                                {**record, "type": "police_report", "name": "Police report", "station": "Galle",
                                 "summary": "Both vehicles collided. The third-party vehicle was towed. Fault was not established."},
                                assessor_report()])
        _, identifier = self.save_claim(draft)
        clause = "Third-party property damage cover pays amounts the insured is legally liable to pay."
        chosen = self.upload(clause, "unresolved-towing")
        index_document(self.store, chosen)
        self.generate_schedule(identifier, chosen)
        self.assertEqual(self.start_claim(identifier, chosen), "queued")

        def coverage_output(packet):
            repair, towing = packet["claimed_costs"]
            return {"coverage_conclusion": "UNRESOLVED", "required_documents": [], "liability_evidence_refs": [],
                    "liability_evidence_requests": [{"owner": "claimant", "document": "Court judgment or formal liability admission",
                                                     "reason": "Required to establish liability."}],
                    "cost_assessments": [{"line_ref": repair["line_ref"], "policy_item": "Third-party injury"},
                                         {"line_ref": towing["line_ref"], "policy_item": "Third-party injury", "evidence_requests": [
                                             {"owner": "claimant", "document": "Towing invoice or receipt",
                                              "reason": "The invoice is needed to verify the LKR 6,500 charge (Section II-1(b), W001 lines 45-54)."},
                                             {"owner": "claimant", "document": "Written insurer consent",
                                              "reason": "Written consent is required for all costs."}]}]}

        audit = self.run_against_pdf(identifier, chosen, clause, outcome="undetermined", additional_coverage=coverage_output)
        self.assertTrue(verify_audit(audit))
        details = audit["context"]["findings"]["coverage"]["details"]["third_party_assessment"]
        self.assertEqual(details["liability"]["evidence_requests"][0]["owner"], "internal")
        self.assertEqual(details["costs"][1]["policy_applicability"], "unresolved")
        decision = audit["decision"]
        self.assertEqual(decision["route"], "request more information")
        self.assertIsNone(decision["payable_amount"])
        self.assertEqual({request["document"] for request in decision["information_required"]},
                         {"Towing invoice or receipt", "Breakdown of the total claim amount"})
        report = self.client.get(f"/api/portal/claims/{identifier}/investigation.pdf")
        self.assertEqual(report.status_code, 200)
        text = " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(report.content)).pages)
        body = text.split("Decision summary", 1)[1]
        self.assertIn("Towing invoice or receipt", body)
        self.assertNotIn("Court judgment", body)
        self.assertNotIn("Section II-1(b)", body)
        self.assertNotIn("W001", body)
        self.assertIn("Coverage /", body)

    def test_selected_wording_and_raw_cost_text_remain_citable_with_unconfirmed_schedule(self):
        draft = claim_draft()
        draft["claim"]["incident_details"]["claimed_injuries"] = []
        _, identifier = self.save_claim(draft)
        clause = "Collision damage is covered under this motor wording."
        policy = self.upload(clause, "wording-only")
        index_document(self.store, policy)
        self.generate_schedule(identifier, policy)
        self.assertEqual(self.start_claim(identifier, policy), "queued")
        estimate = "submission:/unstructured_inputs/repair_estimate_text"
        package = self.run_against_pdf(identifier, policy, clause, applicability="unclear",
                                       document_refs=((estimate,), ("submission:/unstructured_inputs/medical_intake_note",)))
        details = package["context"]["findings"]["coverage"]["details"]
        [item] = details["coverage_items"]
        self.assertEqual(item["document_refs"], [estimate])
        self.assertEqual(item["outcome"], "undetermined")
        self.assertEqual(details["submitted_documents"], {estimate: "Repair estimate text"})
        self.assertEqual(details["policy_rag"]["wording_passages"], item["passage_refs"])
        self.assertEqual(details["policy_rag"]["applicable_passages"], [])
        route = package["decision"]
        self.assertEqual(route["route"], "adjuster review")
        self.assertIsNone(route["payable_amount"])
        reason = ("No individual policy schedule was confirmed for the reported vehicle and loss date. "
                  "The cited wording could not be matched to the reported loss.")
        self.assertIn(reason, route["justification"])
        self.assertNotIn("chunk:", route["justification"])
        self.assertNotIn("Judge recommendation", route["justification"])

        with self.store.session.begin() as session:
            decision = session.get(DecisionRecord, identifier)
            legacy = {**decision.payload, "justification": "Legacy route explanation\n\nClaude: unverified draft with chunk:1"}
            decision.payload, decision.payload_hash = legacy, digest(legacy)
        staff = self.client.get(f"/api/portal/claims/{identifier}").json()
        self.assertEqual(staff["triage"]["justification"], route["justification"])
        self.assertEqual(staff["findings"]["coverage"]["details"]["coverage_unconfirmed_reason"], reason)
        archived = self.client.get(f"/api/claims/{identifier}/audit").json()
        self.assertTrue(verify_audit(archived))
        self.assertEqual(archived["decision"], legacy)

    def test_v4_stamps_and_verifies_before_publication_and_resumes(self):
        _, identifier = self.save_claim()
        wording = "Schedule: SYN-SP-CAB-1042, 01/01/2025 to 31/12/2025. Comprehensive cover. Collision damage is covered."
        chosen = self.upload(wording, "pilot")
        index_document(self.store, chosen)
        self.generate_schedule(identifier, chosen)
        with patch.dict("os.environ", {"CLAIMS_SELECTED_PDF_WORKFLOW": "v4"}):
            self.assertEqual(self.start_claim(identifier, chosen), "queued")
        workflow = self.worker.workflow
        scripted = ScriptedModel()

        def provider(store, assessment_id, stage, system, messages, tools, choice):
            if stage == "planner:1":
                name, result = "suggest_questions", {"tasks": [
                    {"specialist": "historical", "task": "Run before Coverage"},
                    {"specialist": "coverage", "task": "Check collision wording"},
                ]}
            elif stage == "r1:coverage":
                source = json.loads(messages[0]["content"])["initial_policy_evidence"][0]["result"]["passages"][0]["source"]
                name, result = "submit_analysis", {"summary": "This unverified prose must not reach the decision.", "review_reasons": [],
                    "policy_applicability": "applicable", "evidence": [source], "coverage_conclusion": "COVERED",
                    "coverage_items": [{"item": "Collision", "outcome": "covered", "policy_term": "Collision damage is covered.",
                                        "reason": "Impact damaged the insured vehicle.", "passage_refs": [source]}],
                    "schedule": {"registration": "SYN-SP-CAB-1042", "starts_on": "2025-01-01", "ends_on": "2025-12-31",
                                 "cover": "comprehensive", "passage_refs": [source]}}
            elif stage == "auditor:1":
                self.assertEqual({tool["name"] for tool in tools}, {"verify_evidence"})
                checks = json.loads(messages[0]["content"])["checks"]
                name, result = "verify_evidence", {"supported": sorted(checks)}
            elif stage == "r1:fraud":
                check = tools[-1]["input_schema"]["properties"]["evidence"]["items"]["enum"]
                self.assertIn("submission:/narrative", check)
                name, result = "submit_analysis", {"summary": "The claim was screened.", "review_reasons": [],
                    "evidence": ["submission:/narrative" if len(messages) > 1 else "submission:/invented"]}
            else:
                return scripted(store, assessment_id, stage, system, messages, tools, choice)
            return {"content": [{"type": "tool_use", "id": "submit", "name": name, "input": result}]}

        with patch.dict("os.environ", {"CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}), \
                patch.object(agentic, "call_model", side_effect=provider) as calls:
            with patch.object(workflow, "publish", side_effect=RuntimeError("paused before publication")):
                with self.assertRaisesRegex(RuntimeError, "paused before publication"):
                    workflow.run(identifier)
            count = calls.call_count
            workflow.run(identifier)
            self.assertEqual(calls.call_count, count)
        audit = self.client.get(f"/api/claims/{identifier}/audit").json()
        self.assertTrue(verify_audit(audit))
        self.assertEqual(audit["execution"]["workflow"], V4)
        coverage_span = next(span for span in audit["trace"]["spans"] if span["name"] == "specialist.coverage")
        self.assertIsNotNone(coverage_span["ended_at"])
        self.assertEqual(audit["decision"]["route"], "adjuster review")
        self.assertIsNone(audit["decision"]["payable_amount"])
        self.assertNotIn("unverified prose", audit["decision"]["justification"])
        detail = self.client.get(f"/api/portal/claims/{identifier}").json()["investigation"]
        self.assertEqual(detail["workflow"], V4)
        self.assertEqual({task["specialist"] for task in detail["typed_plans"]["1"]["tasks"]}, {"coverage", "fraud"})
        self.assertEqual(detail["verification"]["1"]["supported"], ["coverage:0", "schedule"])
        self.assertEqual([citation["page"] for citation in detail["verification"]["1"]["citations"]], [1])

    def test_v4_accepts_ocr_schedule_cover_and_verified_document_requirements(self):
        claim = claim_draft()["claim"]
        claim["documents"][0]["id"] = "estimate"
        item = {"outcome": "covered", "passage_refs": ["chunk:ocr"]}
        finding = {"status": "information", "summary": "Unverified claim", "details": {
            "coverage_status": "NEEDS_INFORMATION", "coverage_items": [item],
            "schedule_candidate": {"registration": claim["registration"], "starts_on": "2025-01-01", "ends_on": "2025-12-31",
                                   "cover": "comprehensive", "passage_refs": ["chunk:schedule"]},
            "missing": [{"field": "police_report", "policy_citation": {"id": "chunk:duty", "quote": "The insured must submit a Police report."}}],
            "policy_rag": {"ocr_pages": [1, 2]}, "financial": {"payable": "120000.00"},
        }}
        passages = {"chunk:ocr": {"page": 2, "text": "Collision damage is covered.", "ocr": True},
                    "chunk:schedule": {"page": 1, "text": "SYN-SP-CAB-1042 Comprehensive 01/01/2025 to 31/12/2025", "ocr": True},
                    "chunk:duty": {"page": 1, "text": "The insured must submit a Police report.", "ocr": True}}
        checks = coverage_checks(finding, claim, passages)
        self.assertEqual(set(checks), {"coverage:0", "document:0", "schedule"})
        invalid_quote = deepcopy(finding)
        invalid_quote["details"]["missing"][0]["policy_citation"]["quote"] += " Within 24 hours."
        self.assertNotIn("document:0", coverage_checks(invalid_quote, claim, passages))
        for document_verified in (False, True):
            with self.subTest(document_verified=document_verified):
                result = deepcopy(finding)
                supported = {"coverage:0", "schedule"}
                if document_verified:
                    supported.add("document:0")
                verify_coverage(result, claim, {"document_kind": "wording"}, checks, supported, recompute_costs(claim))
                details = result["details"]
                self.assertTrue(details["schedule_confirmed"])
                self.assertEqual(details["coverage_status"], "NEEDS_INFORMATION" if document_verified else "COVERED")
                self.assertEqual(details["coverage_items"][0]["outcome"], "covered")
                self.assertIn("chunk:ocr", details["policy_rag"]["applicable_passages"])
                self.assertEqual(details["policy_rag"]["ocr_pages"], [1, 2])
                self.assertEqual(len(details["missing"]), int(document_verified))
                self.assertIsNone(details["financial"]["payable"])

    def test_saved_explanations_omit_ocr_only_review_notes(self):
        recorded = "The reported collision location differs between the submitted accounts."
        old_note = "Check OCR-derived terms against the original PDF."
        reconciled = "Repair estimate line items reconcile correctly to LKR 187,500."
        finding = {"agent": "coverage", "status": "review", "summary": old_note, "details": {
            "coverage_status": "UNRESOLVED", "schedule_confirmed": True,
            "policy_rag": {"ocr_review_pages": [2], "cited": True}, "review_reasons": [old_note, reconciled, recorded],
            "document_inconsistencies": [{"reason": recorded, "document_refs": ["submission:/documents/0"]}],
        }}
        displayed = display_finding(finding, {}, {"document_kind": "wording"})
        self.assertEqual(displayed["details"]["coverage_unconfirmed_reason"], recorded)
        self.assertNotIn(reconciled, displayed["summary"])
        self.assertEqual(displayed["details"]["review_reasons"], [reconciled, recorded])
        self.assertEqual(finding["details"]["review_reasons"], [old_note, reconciled, recorded])

    def test_product_summary_cannot_establish_coverage_or_a_document_requirement(self):
        _, identifier = self.save_claim()
        clause = "A police report must be supplied for every collision."
        pdf = self.upload(f"Insurance Product Information Document. {clause}", "2026.1")
        index_document(self.store, pdf)
        self.assertEqual(self.generate_schedule(identifier, pdf)["wording"]["document_kind"], "product_summary")
        self.assertEqual(self.start_claim(identifier, pdf), "queued")
        package = self.run_against_pdf(identifier, pdf, clause)
        finding = package["context"]["findings"]["coverage"]["details"]
        self.assertEqual(finding["policy_rag"]["document_kind"], "product_summary")
        self.assertEqual(finding["coverage_status"], "UNRESOLVED")
        self.assertEqual(package["decision"]["route"], "adjuster review")
        self.assertEqual(package["decision"]["information_required"], [])

    def test_notifying_the_police_is_distinct_from_providing_a_report(self):
        _, identifier = self.save_claim()
        clause = "You must inform the police promptly. A police report may be requested."
        pdf = self.upload(clause, "2026.1")
        index_document(self.store, pdf)
        self.generate_schedule(identifier, pdf)
        self.assertEqual(self.start_claim(identifier, pdf), "queued")
        package = self.run_against_pdf(identifier, pdf, clause, requirement_quote=clause)
        self.assertEqual(package["decision"]["route"], "adjuster review")
        self.assertEqual(package["decision"]["information_required"], [])

    def test_wording_is_checked_before_requesting_a_missing_estimate(self):
        draft = claim_draft()
        draft["claim"]["documents"] = []
        for outcome in ("covered", "not_covered"):
            with self.subTest(outcome=outcome):
                _, identifier = self.save_claim(draft)
                clause = ("Collision damage is covered." if outcome == "covered" else "Collision damage is excluded.")
                pdf = self.upload(clause, outcome)
                index_document(self.store, pdf)
                self.generate_schedule(identifier, pdf)
                self.assertEqual(self.start_claim(identifier, pdf), "queued")
                package = self.run_against_pdf(identifier, pdf, clause, outcome=outcome)
                route = "request more information" if outcome == "covered" else "adjuster review"
                self.assertEqual(package["decision"]["route"], route)
                gaps = package["context"]["findings"]["intake"]["details"]["deferred_cost_gaps"]
                self.assertEqual(gaps[0]["field"], "repair_estimate")

    def test_scanned_wording_supports_exclusions_and_mandatory_document_requests(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        stream = io.BytesIO()
        writer.write(stream)
        for outcome, clause in (("covered", "A police report must be supplied for every collision."),
                                ("not_covered", "Collision damage is excluded.")):
            with self.subTest(outcome=outcome):
                _, identifier = self.save_claim()
                pdf = self.upload("", f"scan-{outcome}", content=stream.getvalue())
                with patch("backend.evidence.policy_pdf.read_scan", return_value=clause):
                    index_document(self.store, pdf)
                schedule = self.generate_schedule(identifier, pdf)
                self.assertTrue(schedule["wording_references"][0]["ocr"])
                self.assertEqual(self.start_claim(identifier, pdf), "queued")
                package = self.run_against_pdf(identifier, pdf, clause, outcome=outcome)
                details = package["context"]["findings"]["coverage"]["details"]
                self.assertEqual(details["policy_rag"]["ocr_pages"], [1])
                if outcome == "covered":
                    self.assertEqual(details["coverage_status"], "NEEDS_INFORMATION")
                    self.assertEqual(package["decision"]["route"], "request more information")
                    self.assertEqual(package["decision"]["information_required"][0]["document"], "Police report")
                else:
                    self.assertEqual(details["coverage_status"], "NOT_COVERED")
                    self.assertEqual(details["coverage_items"][0]["outcome"], "not_covered")
                    self.assertEqual(package["decision"]["information_required"], [])
