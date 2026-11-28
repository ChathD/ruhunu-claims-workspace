"""Verify the schedule-first workflow with actual portal endpoints and mocked providers."""
import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch
from zipfile import ZipFile

import httpx
import numpy as np
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from sqlalchemy import delete

from backend.api import create_app
from backend.api import policy_document_routes
from backend.agents.fraud_specialist import scheduled_cost_indicators
from backend.claim_schemas import SCENARIO_ROUTES
from backend.evidence.policy_pdf import preserve_table_rows, single_column_scan
from backend.evidence.motor_benefits import motor_benefit_excess, towing_excess_unresolved
from backend.evidence.policy_schedule import apply_matched_schedule, verify_scheduled_repair
from backend.evidence.policy_wording_index import (
    CHUNK_TOKENS, LEGACY_CHUNK_FORMAT, OVERLAP_TOKENS, create_document, index_document, passage_tokenizer, token_passages,
)
from backend.generation import policy_schedule_generator, scheduled_claim_generator
from backend.storage.sqlite_store import Assessment, Claim, DecisionRecord, PolicyPage, PolicySchedule, SyntheticClaimAttachment, SyntheticClaimSchedule, digest
from backend.workflow.routing_rules import select_route
from test_agentic import fake_embeddings, text_pdf
from test_generator import assessor_report, claim_draft


WORDING = ("Comprehensive cover for private motor cars in Sri Lanka. Policy fee is LKR 100. Stamp duty is LKR 50.",
           "Own damage and theft limits are shown in the schedule.",
           "Third-party property limit is LKR 2,000,000.", "Third-party injury cover as required by law.",
           "Own damage and theft carry a compulsory excess of LKR 5,000.",
           "Optional M03 applies an excess to Sections I and II. It cannot be combined with M80. Processing fee is LKR 50.",
           "M80 Accidental Damage Excess. Section I only, with an additional premium. Cannot be combined with M03.")


def wording_pdf(clauses=WORDING):
    writer = PdfWriter()
    for clause in clauses:
        writer.append(BytesIO(text_pdf(clause)))
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def contract_fixture(references):
    def cite(index):
        return {"reference": references[WORDING[index]]}

    def benefit(loss, basis, limit, excess, clause):
        return {"loss_type": loss, "limit_basis": basis, "limit_lkr": limit, "compulsory_excess_lkr": excess,
                "compulsory_excess_basis": None, "voluntary_excess_lkr": "0", "conditions": WORDING[clause], "wording": cite(clause)}

    return {"insured_name": "Anjali Fernando", "insured_address": "12 Example Lane, Galle", "district": "Galle", "postal_code": "80000",
            "vehicle": {"registration": "SYN-SP-CAB-1042", "make": "Toyota", "model": "Vitz", "year": 2018,
                        "chassis_number": "SYN-CH-24575", "engine_number": "SYN-EN-1458", "vehicle_class": "Private motor car",
                        "body_type": "Hatchback", "fuel": "Petrol", "engine_capacity_cc": 1000, "seats": 5},
            "issued_on": "2025-10-25", "starts_on": "2025-11-01", "ends_on": "2026-10-31", "cover_type": "comprehensive",
            "territorial_limits": "Sri Lanka", "permitted_use": "Private use", "authorised_drivers": "Licensed drivers",
            "named_drivers": ["Anjali Fernando"], "endorsements": [], "special_conditions": [],
            "cover": [benefit("repair", "scheduled_amount", "5000000", "5000", 1),
                      benefit("theft", "scheduled_amount", "5000000", "5000", 1),
                      benefit("third_party_property_damage", "wording_amount", "2000000", "0", 2),
                      benefit("third_party_injury", "statutory", None, "0", 3)],
            "claims_instructions": "Notify the insurer promptly with the incident details.", "base_premium_lkr": "60000",
            "no_claim_discount_lkr": "5000", "charges": [{"label": "Policy fee", "category": "fee", "amount_lkr": "100"}],
            "wording_references": [cite(0), cite(4)]}


def scheduled_model_output(request):
    body = json.loads(request.content)
    name = body["tool_choice"]["name"]
    prompt = body["messages"][0]["content"]

    def read(label):
        return json.JSONDecoder().raw_decode(prompt.split(f"{label} JSON: ", 1)[1])[0]

    sample = claim_draft()["claim"]
    if name == scheduled_claim_generator.CLAIM_TOOL:
        schedule, plan = read("Policy schedule"), read("Assignment")
        amount = max(Decimal(plan["gross_amount_from"]), min(Decimal("88000.00"), Decimal(plan["gross_amount_through"])))
        claim = {"insured_name": schedule["insured_name"], "policy_id": schedule["policy_number"],
                 "registration": schedule["vehicle"]["registration"], "loss_type": plan["loss_type"],
                 "reported_date": plan["report_from"], "gross_amount": str(amount), "additional_costs": [],
                 "narrative": f"{plan['incident']} The caller reported the event near the market."}
        if plan["loss_date"] is not None:
            claim["loss_date"] = plan["loss_date"]
        if "incident_location" in plan:
            claim["narrative"] += f" The tow began in {plan['incident_location']}."
        if "Use shared as at_fault_party" in prompt:
            claim["incident_details"] = {"at_fault_party": "shared"}
        if "Set airbags_deployed to false" in prompt:
            claim["incident_details"] = {"airbags_deployed": False}
        if "occasional_driver" in plan:
            claim["narrative"] += f" {plan['occasional_driver']} was driving with permission."
        return name, prompt, {"claim": claim, "scenario": "Vehicle struck near market"}

    claim, slots = read("Claim submission"), read("Document fields")
    cost, police, assessment = slots[:3]
    estimate = {**sample["documents"][0], "type": cost["type"], "registration": cost["registration"],
                "date": cost["issued_through"], "total": claim["gross_amount"],
                "line_items": [{"amount": claim["gross_amount"], "description": "Panel work and refinishing"}]}
    if estimate["type"] != "repair_estimate":
        del estimate["garage_name"], estimate["garage_address"]
    if cost["type"] == "motor_benefit_receipt":
        estimate.update(service=cost["service"], provider="Example provider", service_location=cost.get("service_location", "Galle"),
                        payment_status=cost["payment_status"])
        for key in ("beneficiary", "service_code", "asset_serial", "part_serial"):
            if key in cost:
                estimate[key] = cost[key]
        estimate["line_items"][0]["description"] = "Replacement windscreen glass" if cost["service"] == "glass_waiver" else "Covered expense"
    elif cost["type"] == "repair_estimate":
        first = (Decimal(claim["gross_amount"]) / 2).quantize(Decimal("0.01"))
        estimate["line_items"] = [{"description": "Repair rear panel", "amount": str(first)},
                                  {"description": "Replace bumper trim", "amount": str(Decimal(claim["gross_amount"]) - first)}]
    report = {"type": "police_report", "name": "Police record.txt", "registration": police["registration"],
              "date": police["issued_through"], "summary": "The driver reported damage to a body panel.",
              "station": "Sample Traffic Branch"}
    for field in ("incident_date", "incident_location", "commercial_trip_verified"):
        if field in police:
            report[field] = police[field]
    if assessment["type"] == "witness_statement":
        supporting = {"type": "witness_statement", "name": "Witness account.txt", "registration": assessment["registration"],
                      "date": assessment["issued_through"], "summary": "I saw the driver pull over following the incident.",
                      "author": "Sample stallholder"}
    else:
        supporting = {**assessor_report(assessment["issued_through"], assessment["issued_through"]),
                      "registration": assessment["registration"], "assessed_amount_lkr": claim["gross_amount"]}
        if claim["loss_type"] == "theft":
            supporting["inspection_basis"] = "document_review"
        for field in ("assessed_amount_lkr", "inspection_basis", "prior_damage_recorded_on"):
            if field in assessment:
                supporting[field] = assessment[field]
    documents = [estimate, report, supporting]
    if len(slots) == 4:
        extra = slots[3]
        common = {"type": extra["type"], "registration": extra["registration"], "date": extra["issued_through"]}
        if extra["type"] == "purchase_receipt":
            documents.append({**common, "name": "Parts receipt.txt", "provider": "Example auto parts",
                              "asset_serial": extra["asset_serial"], "part_serial": extra["part_serial"],
                              "description": "One supplied part fitted to the identified car."})
        else:
            documents.append({**common, "name": "Witness account.txt", "summary": "Both drivers entered the junction without slowing.",
                              "author": "Sample stallholder"})
    return name, prompt, {"documents": documents}


def tool_response(name, output):
    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "name": name, "input": output},
    ]})


def wording_blocker(path, scheduled_quote, reason, *passages):
    return {"kind": "wording", "path": path, "scheduled_quote": scheduled_quote, "reason": reason,
            "evidence": [{"passage": f"W{page:03d}", "quote": WORDING[page - 1]} for page in passages]}


class ScheduledRoutingTests(unittest.TestCase):
    def setUp(self):
        self.schedule = {**contract_fixture(dict.fromkeys(WORDING, "fixture")), "id": "schedule", "policy_number": "P1",
                         "wording": {"document_kind": "wording"}}
        registration = self.schedule["vehicle"]["registration"]
        self.claim = {"source": "Synthetic generator", "loss_type": "repair", "loss_date": "2026-01-03",
                      "gross_amount": "88000.00", "currency": "LKR", "policy_id": "P1", "registration": registration,
                      "documents": [
                          {"type": "repair_estimate", "registration": registration, "date": "2026-01-04",
                           "total": "88000.00", "line_items": [{"amount": "88000.00"}]},
                          {"type": "independent_assessor_report", "registration": registration, "date": "2026-01-05",
                           "inspection_date": "2026-01-04", "inspection_basis": "physical_inspection", "assessed_amount_lkr": "88000.00"},
                      ]}

    def test_equipment_stacks_excess_only_when_the_endorsement_requires_it(self):
        benefit = {"compulsory_excess_lkr": "2000", "voluntary_excess_lkr": "0",
                   "conditions": "Each claim carries this excess in addition to other applicable excesses."}
        self.assertEqual(motor_benefit_excess(self.schedule, benefit, "scheduled_equipment"), Decimal("7000"))
        benefit["conditions"] = "This endorsement applies its own excess."
        self.assertEqual(motor_benefit_excess(self.schedule, benefit, "scheduled_equipment"), Decimal("2000"))

    def test_towing_waiver_requires_policy_wording_when_a_section_excess_applies(self):
        schedule = {**self.schedule, "endorsements": ["An excess applies to Section I, including costs and expenses."]}
        benefit = {"conditions": "No excess applies.", "wording": {"quote": "Reasonable towing costs are covered."}}
        self.assertTrue(towing_excess_unresolved(schedule, benefit))
        benefit["wording"]["quote"] = "No excess applies to this towing benefit."
        self.assertFalse(towing_excess_unresolved(schedule, benefit))

    def test_verified_fixed_repair_can_auto_route_and_complex_terms_still_require_review(self):
        def route():
            coverage = {"agent": "coverage", "summary": "Cited wording supports repair", "status": "review", "evidence": [], "details": {
                "checks_executed": True, "coverage_status": "COVERED", "coverage_items": [
                    {"item": "Own damage", "outcome": "covered", "reason": "Collision damage is covered", "policy_term": "Own damage"},
                ], "policy_rag": {"cited": True, "applicability": "applicable"},
            }}
            apply_matched_schedule(coverage, self.claim, self.schedule)
            verify_scheduled_repair(coverage, self.claim, self.schedule)
            findings = {agent: {"agent": agent, "status": "pass", "summary": "Checked", "evidence": [],
                                "details": {"checks_executed": True}} for agent in ("intake", "fraud_initial")}
            return select_route(self.claim, {"findings": {**findings, "coverage": coverage}, "version": 1,
                "selected_policy": self.schedule["wording"], "historical_available": False,
                "recommendation": {"route": "auto-approve", "explanation": "Independent evidence matches"}}, [])

        decision = route()
        self.assertEqual(decision.route.value, "auto-approve")
        self.assertEqual(decision.payable_amount, Decimal("83000.00"))
        self.schedule["cover"][0]["conditions"] = "Reduce the payable in proportion to market value."
        decision = route()
        self.assertEqual(decision.route.value, "adjuster review")
        self.assertIsNone(decision.payable_amount)

    def test_cost_document_flags_require_a_conflicting_filed_fact(self):
        self.assertEqual(scheduled_cost_indicators(self.claim, self.schedule), [])
        cost = self.claim["documents"][0]
        cost.update(date="2025-10-31", registration="SYN-SP-CAB-9999")
        flags = scheduled_cost_indicators(self.claim, self.schedule)
        self.assertEqual([flag["kind"] for flag in flags], ["cost_document_predates_reported_incident", "cost_document_registration_conflict"])
        self.claim.update(loss_type="third_party_property_damage")
        cost.update(type="third_party_demand", date="2026-01-04")
        self.assertEqual(scheduled_cost_indicators(self.claim, self.schedule), [])


class GlassReceiptGenerationTests(unittest.TestCase):
    def setUp(self):
        generator = scheduled_claim_generator
        schedule = {**contract_fixture(dict.fromkeys(WORDING, "fixture")), "id": "schedule", "policy_number": "P1", "currency": "LKR",
                    "wording": {"document_kind": "wording"}}
        day = date(2026, 1, 3)
        self.plan = generator.ScheduledScenario(
            schedule, schedule["cover"][0], "auto_approval", 17, "glass_only_damage", generator.AUTO_GLASS,
            day, day, day, Decimal("1"), Decimal("75000"), schedule["vehicle"]["registration"],
        )
        self.claim = generator.ScheduledClaim(
            insured_name=schedule["insured_name"], policy_id="P1", registration=schedule["vehicle"]["registration"],
            loss_type="repair", motor_benefit="glass_waiver", loss_date=day, reported_date=day,
            gross_amount="72000", additional_costs=[], narrative="Debris cracked only the windscreen; the rest of the car was undamaged.",
        )

    def test_glass_receipt_retry_identifies_the_unscoped_material_line(self):
        calls = []
        # Replay the production failure: a separate adhesive charge lacked its component.
        items = [
            {"description": "Front windscreen laminated glass supply", "amount": "58500.00"},
            {"description": "Windscreen removal and fitting labour", "amount": "6500.00"},
            {"description": "Urethane bonding adhesive and primer application", "amount": "4200.00"},
            {"description": "Rain sensor recalibration after windscreen replacement", "amount": "2800.00"},
        ]

        def model(request):
            name, prompt, output = scheduled_model_output(request)
            calls.append(prompt)
            output["documents"][0]["line_items"] = deepcopy(items)
            if len(calls) == 2:
                self.assertIn(items[2]["description"], prompt)
                self.assertIn("Every line description, including labour and materials", prompt)
                output["documents"][0]["line_items"][2]["description"] = "Windscreen urethane bonding adhesive and primer application"
            return tool_response(name, output)

        with httpx.Client(transport=httpx.MockTransport(model)) as client:
            documents = scheduled_claim_generator.render_attachments(client, "fixture", self.claim, self.plan)
        self.assertEqual(len(calls), 2)
        self.assertEqual(documents[0].line_items[2].description, "Windscreen urethane bonding adhesive and primer application")
        self.assertEqual(sum(item.amount for item in documents[0].line_items), self.claim.gross_amount)

    def test_glass_receipt_keeps_rejecting_unrelated_repairs(self):
        calls = []

        def model(request):
            name, prompt, output = scheduled_model_output(request)
            calls.append(prompt)
            output["documents"][0]["line_items"] = [
                {"description": "Replacement windscreen glass", "amount": "60000.00"},
                {"description": "Replace front bumper", "amount": "12000.00"},
            ]
            return tool_response(name, output)

        with httpx.Client(transport=httpx.MockTransport(model)) as client, self.assertRaisesRegex(
                scheduled_claim_generator.GenerationError, "Replace front bumper"):
            scheduled_claim_generator.render_attachments(client, "fixture", self.claim, self.plan)
        self.assertEqual(len(calls), 2)


class ScannedWordingTests(unittest.TestCase):
    def test_liability_table_keeps_limits_with_their_rows(self):
        page = Mock()
        page.get_width.return_value = 600
        positioned = [(50, 100, 490, "Third-party injury"),
                      (75, 100, 520, "Third-party property damage"),
                      (108, 100, 950, "Medical expenses: Rs. 2,000"),
                      (50, 870, 940, "Unlimited")]
        recovered = [(46, 120, 190, "Unlimited"), (74, 60, 191, "Rs.5,000,000/-")]
        with patch("backend.evidence.policy_pdf.portable_scan_lines", return_value=(250, recovered)):
            result = single_column_scan(page, 2, 1000, positioned, native=True)
        self.assertEqual(result.splitlines(), ["Third-party injury  Unlimited",
                                               "Third-party property damage  Rs.5,000,000/-",
                                               "Medical expenses: Rs. 2,000"])

    def test_amount_candidates_require_a_readable_number(self):
        text = "Rs. 1 ,OOO per accident. Rs. 7,?00 for another benefit."
        amounts, candidates = policy_schedule_generator.lkr_amounts(text, ocr=True)
        self.assertEqual(amounts, {1000})
        self.assertEqual(candidates, [{"ocr_text": "1 ,OOO", "reading_lkr": "1000"}])
        self.assertEqual(policy_schedule_generator.lkr_amounts(text, ocr=False), (set(), []))

    def test_recovers_missing_percentage_and_truncated_amount_without_changing_other_rows(self):
        native = [(50, 100, 520, "Loss of sight in one eye"), (70, 100, 530, "Loss of one hand or foot"),
                  (70, 870, 910, "50%"), (110, 130, 720, "(2) Rs. 1 arising out of any one occurrence"),
                  (135, 130, 710, "(3) Rs. 1 per document"), (160, 130, 720, "Not exceeding Rs. 1 ,OOO per accident")]
        alternate = [(50, 870, 910, "50%"), (110, 130, 720, "(2）Rs.1,ooo,oo0 arising out of anyone occurrence"),
                     (135, 130, 710, "(3) Rs. 1,000 per accident"), (160, 130, 720, "Not exceedingRs.1,000 per accident")]
        with patch("backend.evidence.policy_pdf.portable_scan_lines", return_value=(1000, alternate)):
            result = single_column_scan(Mock(), 2, 1000, native, native=True)
        self.assertEqual(result.splitlines(), ["Loss of sight in one eye  50%", "Loss of one hand or foot  50%",
                                              "(2) Rs. 1,000,000 arising out of any one occurrence", "(3) Rs. 1 per document",
                                              "Not exceeding Rs. 1,000 per accident"])

    def test_reconstructs_multiline_scanned_table_cells_and_amounts(self):
        image = np.full((240, 600), 255, dtype=np.uint8)
        for y in (30, 90, 120, 180):
            image[y, 25:325] = 0
        for x in (25, 135, 195, 260, 324):
            image[30:181, x] = 0
        page = Mock()
        page.render.return_value.to_numpy.return_value = image
        positions = [(45, 145, 185, "Private"), (60, 145, 181, "cars"), (46, 202, 241, "Buses"), (45, 269, 315, "Other"),
                     (100, 30, 108, "Injury"), *((104, x, x + 44, "Unlimited") for x in (145, 204, 269)),
                     (135, 30, 112, "Property damage"), (141, 145, 182, "Rs. 10"), (139, 204, 235, "Rs. 5"),
                     (162, 145, 179, "Million"), (160, 204, 243, "Million"), (145, 269, 316, "Rs.15,OOO.OO"),
                     (102, 340, 529, "Section 1 wording in the other column")]
        rows = preserve_table_rows(page, 1, positions)
        self.assertEqual([row[3] for row in rows if row[1] == 25], [
            " | Private cars | Buses | Other",
            "Injury | Unlimited | Unlimited | Unlimited",
            "Property damage | Rs. 10 Million | Rs. 5 Million | Rs.15,OOO.OO",
        ])
        self.assertIn(positions[-1], rows)
        amounts, candidates = policy_schedule_generator.lkr_amounts(rows[-1][3], ocr=True)
        self.assertEqual(amounts, {10_000_000, 5_000_000, 15_000})
        self.assertEqual(candidates, [{"ocr_text": "15,OOO.OO", "reading_lkr": "15000.00"}])


class PolicySchedulesTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="schedule-flow-"))
        self.app = create_app(Path(directory))
        self.client = self.enterContext(TestClient(self.app))
        self.store = self.app.state.store
        self.app.state.worker.close()
        self.assertEqual(self.client.post("/api/auth/login", json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"}).status_code, 200)
        self.document = create_document(self.store, title="Sri Lankan motor wording", product="MOTOR", version="2026.1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, self.document)
        with self.store.session() as session:
            _, self.sources = policy_schedule_generator.wording_sources(session, self.document)
        self.draft = contract_fixture({source["text"]: alias for alias, source in self.sources.items()})

    def generate(self, *, document=None, corrections=None, concerns=(), blockers=()):
        def model(request):
            body = json.loads(request.content)
            name = body["tool_choice"]["name"]
            self.assertIn("Third-party property limit", body["messages"][0]["content"])
            if name == "draft_policy_schedules":
                payload = self.draft
            elif name == "repair_policy_schedule":
                payload = {"changes": corrections or [{"path": ["cover", 2, "limit_lkr"], "operation": "set", "value_json": '"3000000"'}]}
            elif name == "propose_schedule_corrections":
                payload = {"options": []}
            elif name == "confirm_policy_schedule_blockers":
                payload = {"blockers": list(blockers)}
            else:
                self.assertNotIn('"total_premium_lkr"', body["messages"][0]["content"])
                payload = {"concerns": list(concerns)}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": name, "input": {"cases": [payload]}}]})

        with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            return self.client.post("/api/policy-schedules", json={"policy_document_id": document or self.document, "count": 1})

    def test_rejects_requests_for_multiple_schedules_before_generation(self):
        with patch.object(policy_document_routes, "generate_schedules") as generate:
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document, "count": 2})
        self.assertEqual(response.status_code, 422, response.text)
        generate.assert_not_called()
        with self.assertRaisesRegex(ValueError, "one policy schedule at a time"):
            policy_schedule_generator.generate_schedules(self.store, self.document, 2, "fixture")

    def test_product_summary_saves_provisional_schedule_with_unstated_terms(self):
        clauses = list(WORDING)
        clauses[0] = "Insurance Product Information Document. Comprehensive cover for private motor cars."
        clauses[1] = "Accidental damage and theft are basic cover for comprehensive policies."
        clauses[2] = "Third-party property limit is stated in the individual policy schedule."
        clauses[4] = "Refer to page 10 of the full policy wording for conditions."
        document = create_document(self.store, title="Motor product summary", product="MOTOR", version="1",
            filename="motor-summary.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        docs = self.client.get("/api/policy-documents?selectable=true").json()["documents"]
        self.assertEqual(next(doc for doc in docs if doc["id"] == document)["document_kind"], "product_summary")
        full = self.client.get("/api/policy-documents?selectable=true&full_wording=true").json()["documents"]
        self.assertNotIn(document, {doc["id"] for doc in full})

        for benefit, clause in zip(self.draft["cover"], (clauses[1], clauses[1], clauses[2], clauses[3]), strict=True):
            benefit.update(limit_lkr=None, compulsory_excess_lkr=None, voluntary_excess_lkr=None, conditions=clause)
        del self.draft["cover"][2]["voluntary_excess_lkr"]
        for benefit in self.draft["cover"][:3]:
            benefit["limit_basis"] = "unspecified"
        self.draft["territorial_limits"] = "Not stated in the product summary."
        response = self.generate(document=document)
        self.assertEqual(response.status_code, 201, response.text)
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["wording"]["document_kind"], "product_summary")
        benefit = saved["cover"][2]
        self.assertEqual(benefit["limit_basis"], "unspecified")
        self.assertIsNone(benefit["limit_lkr"])
        self.assertIsNone(benefit["compulsory_excess_lkr"])
        self.assertIsNone(benefit["voluntary_excess_lkr"])
        self.assertEqual(benefit["wording"]["page"], 3)
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)

        finding = {"status": "complete", "details": {"coverage_status": "COVERED"}}
        apply_matched_schedule(finding, {"loss_date": "2026-01-10", "loss_type": "third_party_property_damage"}, saved)
        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
        self.assertFalse(finding["details"]["schedule_confirmed"])

        with self.store.session() as session:
            _, sources = policy_schedule_generator.wording_sources(session, document)
        draft = policy_schedule_generator.ScheduleDraft.model_validate(self.draft)
        with self.assertRaises(policy_schedule_generator.WordingValidationError):
            policy_schedule_generator.checked_schedule(draft, sources, source_kind="wording")
        self.draft["cover"][2].update(limit_basis="scheduled_amount", limit_lkr="5000000")
        estimated = policy_schedule_generator.ScheduleDraft.model_validate(self.draft)
        with self.assertRaisesRegex(policy_schedule_generator.WordingValidationError, "unspecified in a product summary"):
            policy_schedule_generator.checked_schedule(estimated, sources, source_kind="product_summary")

    def test_full_wording_defaults_unselected_voluntary_excess_before_review(self):
        self.draft["cover"][0]["voluntary_excess_lkr"] = None
        del self.draft["cover"][1]["voluntary_excess_lkr"]

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate()

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [review] = json.loads(requests.call_args.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        for cover in (review["draft"]["cover"], saved["cover"]):
            self.assertEqual([row["voluntary_excess_lkr"] for row in cover[:2]], ["0", "0"])
            self.assertEqual([row["compulsory_excess_lkr"] for row in cover[:2]], ["5000", "5000"])

    def test_full_wording_defaults_blank_compulsory_excess_on_additional_cover_before_review(self):
        medical = "Medical expenses are limited to Rs. 2,000 per accident."
        clauses = (*WORDING, medical)
        document = create_document(self.store, title="Motor wording with medical expenses", product="MOTOR", version="1",
            filename="medical.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        del self.draft["cover"][3]["compulsory_excess_lkr"]
        self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": "Medical expenses",
            "limit_basis": "wording_amount", "limit_lkr": "2000", "conditions": medical, "wording": {"reference": "W008"},
            "compulsory_excess_lkr": None, "compulsory_excess_basis": None, "voluntary_excess_lkr": "0"})

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [review] = json.loads(requests.call_args.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        for cover in (review["draft"]["cover"], saved["cover"]):
            self.assertEqual([item["compulsory_excess_lkr"] for item in cover], ["5000", "5000", "0", "0", "0"])
            self.assertIsNone(cover[4]["compulsory_excess_basis"])

    def test_limit_validation_can_repair_the_basis_and_amount_together(self):
        medical = "Medical expenses are limited to Rs. 2,000 per accident."
        clauses = (*WORDING, medical)
        document = create_document(self.store, title="Motor wording with medical expenses", product="MOTOR", version="1",
            filename="medical.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        del self.draft["cover"][2]["limit_lkr"]
        self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": "Medical expenses",
            "limit_basis": "excluded", "limit_lkr": "2000", "conditions": "Medical expenses are excluded.",
            "wording": {"reference": "W001"}})
        corrections = [
            {"path": ["cover", 2, "limit_lkr"], "operation": "set", "value": "2000000"},
            {"path": ["cover", 4, "limit_basis"], "operation": "set", "value": "wording_amount"},
            {"path": ["cover", 4, "wording"], "operation": "set", "value": {"reference": "W008", "start_line": 1}},
            {"path": ["cover", 4, "conditions"], "operation": "set", "value": medical},
        ]
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document, corrections=corrections)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "repair_policy_schedule", "verify_policy_schedules"])
        [review] = json.loads(requests.call_args.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        self.assertEqual(review["draft"]["cover"][4]["wording"]["reference"], "W008")
        self.assertEqual(saved["cover"][4]["wording"]["page"], 8)
        self.assertEqual(saved["cover"][4]["wording"]["quote"], medical)
        for cover in (review["draft"]["cover"], saved["cover"]):
            self.assertEqual(cover[2]["limit_lkr"], "2000000")
            self.assertEqual(cover[4]["limit_basis"], "wording_amount")
            self.assertEqual(cover[4]["limit_lkr"], "2000")
            self.assertEqual(cover[4]["conditions"], medical)

    def test_zero_limit_on_a_nonmonetary_basis_becomes_null_before_review(self):
        self.draft["cover"][3]["limit_lkr"] = "0"
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate()

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [review] = json.loads(requests.call_args.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        for cover in (review["draft"]["cover"], saved["cover"]):
            self.assertEqual(cover[3]["limit_basis"], "statutory")
            self.assertIsNone(cover[3]["limit_lkr"])

    def test_full_wording_resolves_an_active_benefit_and_omits_an_unresolved_optional_benefit(self):
        note = "Third Party, Fire and Theft Insurance restricts Section I to fire, burglary and theft perils."
        medical = "Medical expenses are limited to LKR 2,000 per accident."
        natural_perils = "M34 Natural Perils Cover\nExtends Section I to flood and storm for an additional premium."
        clauses = (*WORDING[:3], f"{WORDING[3]}\n{note}", *WORDING[4:], medical, natural_perils)
        document = create_document(self.store, title="Motor wording with restricted perils and medical expenses", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        self.draft.update(cover_type="third_party_fire_theft", endorsements=["M34 Natural Perils Cover"],
                          special_conditions=["M34 extends Section I to flood and storm."])
        self.draft["cover"][0]["conditions"] = note
        self.draft["cover"].extend([
            {"loss_type": "additional_benefit", "benefit_name": "Medical Expenses", "limit_basis": "unspecified",
             "conditions": medical, "wording": {"reference": "W008"}},
            {"loss_type": "additional_benefit", "benefit_name": "Natural Perils", "limit_basis": "unconfirmed",
             "conditions": "Applicability under this cover type is uncertain.", "wording": {"reference": "W009"}},
        ])
        original_charge = self.draft["charges"][0]
        self.draft["charges"].append({"label": "M34 premium", "category": "endorsement", "amount_lkr": "250"})
        corrections = [
            {"path": ["cover", 4, "limit_basis"], "operation": "set", "value": "wording_amount"},
            {"path": ["cover", 4, "limit_lkr"], "operation": "set", "value": "2000"},
            {"path": ["cover", 5], "operation": "remove"},
            {"path": ["endorsements"], "operation": "set", "value": []},
            {"path": ["charges"], "operation": "set", "value": [original_charge]},
            {"path": ["special_conditions"], "operation": "set", "value": []},
        ]
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document, corrections=corrections)

        self.assertEqual(response.status_code, 201, response.text)
        draft_call, repair_call, review_call = requests.call_args_list
        self.assertEqual([call.args[3] for call in (draft_call, repair_call, review_call)],
                         ["draft_policy_schedules", "repair_policy_schedule", "verify_policy_schedules"])
        schema = draft_call.args[4].model_json_schema()
        bases = schema["$defs"]["FullWordingCover"]["properties"]["limit_basis"]["enum"]
        self.assertEqual(set(bases), {"wording_amount", "wording_formula", "scheduled_amount", "market_value", "statutory", "unlimited", "excluded"})
        [review] = json.loads(review_call.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        for schedule in (review["draft"], saved):
            self.assertEqual(schedule["endorsements"], [])
            self.assertEqual(schedule["charges"], [original_charge])
            [benefit] = schedule["cover"][4:]
            self.assertEqual((benefit["benefit_name"], benefit["limit_basis"], benefit["limit_lkr"]),
                             ("Medical Expenses", "wording_amount", "2000"))

    def test_wording_formula_retains_the_cited_terms_without_a_fixed_amount(self):
        clause = "The battery benefit is limited to 10% of the sum insured or the cost to repair, whichever is lower."
        sources = {**self.sources, "W099": {"reference": "pdf-page:fixture:99", "page": 99, "text": clause, "ocr": False}}
        benefit = {"loss_type": "additional_benefit", "benefit_name": "Battery damage", "limit_basis": "wording_formula",
                   "limit_lkr": None, "conditions": clause, "wording": {"reference": "W099", "start_line": 1}}
        self.draft["cover"].append(benefit)
        draft = policy_schedule_generator.FullWordingDraft.model_validate(self.draft, context={"source_kind": "wording", "sources": sources})
        checked = policy_schedule_generator.checked_schedule(draft, sources)["cover"][-1]
        self.assertEqual((checked["limit_basis"], checked["limit_lkr"], checked["conditions"], checked["wording"]["quote"]),
                         ("wording_formula", None, clause, clause))

    def test_stalled_limit_basis_repair_can_rebuild_the_cover_list(self):
        corrected_cover = deepcopy(self.draft["cover"])
        self.draft["cover"][2].update(limit_basis="unspecified", limit_lkr=None)
        calls = []

        def provider(request):
            body = json.loads(request.content)
            name = body["tool_choice"]["name"]
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "repair_policy_schedule":
                prompt = body["messages"][0]["content"]
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                paths = [edit["path"] for edit in repair["allowed_edits"]]
                if calls.count(name) < 3:
                    self.assertIn(["cover", 2, "limit_basis"], paths)
                    case = {"changes": [{"path": ["cover", 2, "conditions"], "operation": "set",
                                         "value": f"The limit remains unresolved after repair {calls.count(name)}."}]}
                else:
                    self.assertEqual(paths, [["cover"]])
                    case = {"changes": [{"path": ["cover"], "operation": "set", "value": corrected_cover}]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                case = {"concerns": []}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": name, "input": {"cases": [case]}}]})

        with httpx.Client(transport=httpx.MockTransport(provider)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "repair_policy_schedule", "repair_policy_schedule",
                                 "repair_policy_schedule", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover"][2]["limit_basis"], "wording_amount")
        self.assertEqual(saved["cover"][2]["limit_lkr"], "2000000")

    def test_wording_review_restores_a_required_compulsory_excess_after_defaulting(self):
        self.draft["cover"][0]["compulsory_excess_lkr"] = None
        conflict = "The own-damage clause requires a compulsory excess of LKR 5,000."
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "verify_policy_schedules":
                [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                revised = calls.count(name) == 2
                self.assertEqual(review["draft"]["cover"][0]["compulsory_excess_lkr"], "5000" if revised else "0")
                case = {"concerns": [] if revised else [conflict]}
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [wording_blocker(["cover", 0, "compulsory_excess_lkr"], "0", conflict, 5)]}
            else:
                self.assertEqual(name, "repair_policy_schedule")
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                instructions = {tuple(edit["path"]): edit["reason"] for edit in repair["allowed_edits"]}
                self.assertIn("positive numeric LKR amount", instructions["cover", 0, "compulsory_excess_lkr"])
                self.assertIn("use scheduled_amount in compulsory_excess_basis", instructions["cover", 0, "compulsory_excess_lkr"])
                case = {"changes": [
                    {"path": ["cover", 0, "compulsory_excess_lkr"], "operation": "set", "value": "5000"},
                    {"path": ["cover", 0, "compulsory_excess_basis"], "operation": "set", "value": "wording_amount"},
                ]}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover"][0]["compulsory_excess_lkr"], "5000")
        self.assertEqual(saved["cover"][0]["compulsory_excess_basis"], "wording_amount")

    def test_blocks_overlapping_generation_and_releases_lock_after_failure(self):
        started, release = Event(), Event()
        body = {"policy_document_id": self.document}

        def fail_after_release(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(10))
            raise policy_schedule_generator.GenerationError("Fixture generation failure.")

        with patch.object(policy_document_routes, "generate_schedules", side_effect=fail_after_release) as generate, \
                ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(self.client.post, "/api/policy-schedules", json=body)
            try:
                self.assertTrue(started.wait(5))
                overlapping = self.client.post("/api/policy-schedules", json=body)
            finally:
                release.set()
            failed = first.result(timeout=5)
        self.assertEqual(failed.status_code, 422, failed.text)
        self.assertEqual(overlapping.status_code, 409, overlapping.text)
        self.assertIn("already being generated", overlapping.json()["detail"])
        generate.assert_called_once()
        retry = self.generate()
        self.assertEqual(retry.status_code, 201, retry.text)
        self.assertEqual(len(retry.json()["schedules"]), 1)

    def test_retries_repeated_person_vehicle_limits_and_excess_for_the_same_wording(self):
        clauses = list(WORDING)
        clauses[4] = ("Own damage carries the compulsory excess stated in the schedule. An additional voluntary excess "
                      "may be selected for own damage. Theft carries a compulsory excess of LKR 5,000.")
        document = create_document(self.store, title="Motor wording with scheduled excesses", product="MOTOR", version="1",
            filename="excess.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        self.draft["cover"][0].update(compulsory_excess_basis="scheduled_amount", voluntary_excess_lkr="10000")
        first = self.generate(document=document)
        self.assertEqual(first.status_code, 201, first.text)

        def new_profile(name, make, model, number, limit, compulsory, voluntary):
            draft = deepcopy(self.draft)
            draft.update(insured_name=name, named_drivers=[name])
            draft["vehicle"].update(make=make, model=model, registration=f"SYN-WP-CAR-{number}",
                                    chassis_number=f"SYN-CH-{number}", engine_number=f"SYN-EN-{number}")
            for cover in draft["cover"][:2]:
                cover["limit_lkr"] = limit
            draft["cover"][0].update(compulsory_excess_lkr=compulsory, voluntary_excess_lkr=voluntary)
            return draft

        repeated = new_profile("Anjali Fernando", "Toyota", "Vitz", "7820", "5100000", "6000", "12000")
        second = new_profile("Kavindi Jayasuriya", "Suzuki", "Swift", "7821", "6200000", "7000", "14000")
        repeated_amounts = new_profile("Nimal Perera", "Honda", "Fit", "7822", "5000000.00", "5000.00", "14000.00")
        third = new_profile("Nimal Perera", "Honda", "Fit", "7822", "7100000", "9000", "18000")
        attempts = iter([repeated, second, repeated_amounts, third])
        calls = []

        def model(request):
            body = json.loads(request.content)
            name = body["tool_choice"]["name"]
            calls.append(name)
            if name == "draft_policy_schedules":
                prompt = body["messages"][0]["content"]
                self.assertIn("Anjali Fernando", prompt)
                if calls.count(name) == 2:
                    self.assertIn("different full policyholder name", prompt)
                if calls.count(name) >= 3:
                    self.assertIn("Kavindi Jayasuriya", prompt)
                if calls.count(name) == 4:
                    self.assertIn("different schedule-set limit for repair", prompt)
                    self.assertIn("different schedule-set compulsory excess for repair", prompt)
                    self.assertIn("different wording-supported voluntary excess for repair", prompt)
                cases = [next(attempts)]
                schema = body["tools"][0]["input_schema"]["properties"]["cases"]
                self.assertEqual(schema["maxItems"], 1)
            else:
                self.assertEqual(name, "verify_policy_schedules")
                cases = [{"concerns": []}]
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": name, "input": {"cases": cases}}]})

        with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            responses = [self.client.post("/api/policy-schedules", json={"policy_document_id": document}) for _ in range(2)]
        for response in responses:
            self.assertEqual(response.status_code, 201, response.text)
            self.assertEqual(len(response.json()["schedules"]), 1)
        self.assertEqual(calls, ["draft_policy_schedules", "draft_policy_schedules", "verify_policy_schedules"] * 2)
        batch = [item for response in [first, *responses] for item in response.json()["schedules"]]
        self.assertEqual({item["insured_name"] for item in batch}, {"Anjali Fernando", "Kavindi Jayasuriya", "Nimal Perera"})
        self.assertEqual({item["vehicle"]["model"] for item in batch}, {"Vitz", "Swift", "Fit"})
        self.assertEqual({item["cover"][0]["limit_lkr"] for item in batch}, {"5000000", "6200000", "7100000"})
        self.assertEqual({item["cover"][0]["voluntary_excess_lkr"] for item in batch}, {"10000", "14000", "18000"})
        self.assertEqual({item["cover"][0]["compulsory_excess_lkr"] for item in batch}, {"5000", "7000", "9000"})
        self.assertEqual({item["cover"][1]["compulsory_excess_lkr"] for item in batch}, {"5000"})
        self.assertEqual({item["cover"][2]["limit_lkr"] for item in batch}, {"2000000"})

    def test_schedule_first_claim_generation_matches_automatically(self):
        self.assertEqual(self.client.post("/api/generator/preview", json={"count": 1}).status_code, 409)
        generated = self.generate()
        self.assertEqual(generated.status_code, 201, generated.text)
        [schedule] = generated.json()["schedules"]
        self.assertEqual(schedule["total_premium_lkr"], "55100.00")
        self.assertEqual(schedule["wording"]["id"], self.document)
        self.assertEqual(self.client.get(f"/api/policy-schedules/{schedule['id']}").json(), schedule)
        saved_preview = self.client.get(f"/api/policy-schedules/{schedule['id']}?preview=true")
        self.assertEqual(saved_preview.status_code, 200, saved_preview.text[:200])
        self.assertEqual(saved_preview.headers["Cache-Control"], "no-store")
        self.assertNotIn("Content-Disposition", saved_preview.headers)
        pages = saved_preview.json()["pages"]
        self.assertTrue(all(base64.b64decode(page["png"]).startswith(b"\x89PNG\r\n\x1a\n") for page in pages))
        preview_text = "\n".join(page["text"] for page in pages)
        for expected in (schedule["policy_number"], "Toyota Vitz", "Cover, limits and excesses", "LKR 55,100.00"):
            self.assertIn(expected, preview_text)
        [saved] = self.client.get("/api/policy-schedules").json()["schedules"]
        self.assertEqual(saved["id"], schedule["id"])
        self.assertNotIn("cover", saved)

        prompts = []

        def incident_model(request):
            name, _, output = scheduled_model_output(request)
            prompts.append(name)
            return tool_response(name, output)

        generator = scheduled_claim_generator
        with patch.dict("os.environ", {"CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}):
            plans = generator.plan_scheduled_claims(self.store, 20)
        repair = next(plan for plan in plans if plan.variant == generator.AUTO_REPAIR)
        with httpx.Client(transport=httpx.MockTransport(incident_model)) as client, patch.object(
                generator, "generation_client", return_value=nullcontext((client, "fixture"))), patch(
                "backend.api.workspace_api.plan_scheduled_claims", side_effect=[[repair], [plans[0]]]):
            preview = self.client.post("/api/generator/preview", json={})
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(prompts, [scheduled_claim_generator.CLAIM_TOOL, scheduled_claim_generator.ATTACHMENT_TOOL])
            second = self.client.post("/api/generator/preview", json={})
            self.assertEqual(second.status_code, 200, second.text)
        claim = preview.json()["fixtures"][0]["claim"]
        self.assertEqual(claim["registration"], "SYN-SP-CAB-1042")
        self.assertLessEqual(schedule["starts_on"], claim["loss_date"])
        self.assertLessEqual(claim["loss_date"], schedule["ends_on"])
        self.assertEqual(claim["policy_id"], schedule["policy_number"])
        self.assertIn("police_report_ocr", claim["unstructured_inputs"])
        batch_id = preview.json()["batch_id"]
        commit_path = f"/api/generator/preview/{batch_id}/commit"
        discard_path = f"/api/generator/preview/{second.json()['batch_id']}"
        with TestClient(create_app(self.store.path.parent), cookies=self.client.cookies) as restarted:
            restarted.app.state.worker.close()
            self.assertEqual(restarted.delete(discard_path).status_code, 204)
            saved = restarted.post(commit_path)
            self.assertEqual(saved.status_code, 201, saved.text)
            batch = saved.json()
        discarded = self.client.post(discard_path + "/commit")
        self.assertEqual(discarded.status_code, 404)
        self.assertIn("generate a new preview", discarded.json()["detail"])
        retried = self.client.post(commit_path)
        self.assertEqual(retried.status_code, 201, retried.text)
        self.assertEqual(retried.json(), batch)
        [record] = self.client.get(f"/api/generator/batches/{batch_id}/dataset.json").json()
        self.assertEqual(record["claim_id"], claim["claim_id"])
        [identifier] = batch["assessment_ids"]
        detail_path = f"/api/portal/claims/{identifier}"
        overview = self.client.get(detail_path + "?overview=true").json()
        self.assertEqual(overview["scenario_title"], preview.json()["fixtures"][0]["scenario"])
        self.assertEqual((overview["submission"]["vehicle"]["make"], overview["submission"]["vehicle"]["model"]), ("Toyota", "Vitz"))
        report_path = detail_path + "/investigation.pdf"
        self.assertEqual(self.client.get(report_path).status_code, 409)
        self.assertEqual(self.client.get(report_path + "?preview=true").status_code, 409)
        self.assertEqual(self.client.get(detail_path + "/policy-schedule").status_code, 404)
        process = f"/api/claims/{identifier}/process"
        with self.store.session.begin() as session:
            row = session.get(Claim, claim["claim_id"])
            row.payload = {**claim, "insured_name": "Unmatched Person"}
            row.payload_hash = digest(row.payload)
        self.assertEqual(self.client.post(process).status_code, 409)
        with self.store.session.begin() as session:
            row = session.get(Claim, claim["claim_id"])
            row.payload = {**claim, "registration": "SYN-SP-CAB-1043"}
            row.payload_hash = digest(row.payload)
        self.assertEqual(self.client.post(process).status_code, 409)
        with self.store.session.begin() as session:
            row = session.get(Claim, claim["claim_id"])
            row.payload = {**claim, "vehicle": {**claim["vehicle"], "chassis_number": "SYN-FORM-OTHER"}}
            row.payload_hash = digest(row.payload)
        with patch.dict("os.environ", {"CLAIMS_LLM_MODE": "deterministic", "CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}):
            self.assertEqual(self.client.post(process).status_code, 202)
        detail = self.client.get(detail_path).json()
        self.assertEqual(detail["policy_schedule"], schedule)
        self.assertEqual(detail["selected_policy"]["id"], self.document)
        self.assertEqual(self.client.get(detail_path + "/policy-schedule").json(), schedule)
        matched_preview = self.client.get(detail_path + "/policy-schedule?preview=true").json()
        self.assertEqual([page["text"] for page in matched_preview["pages"]], [page["text"] for page in pages])
        self.assertFalse(self.client.get(detail_path + "?overview=true").json()["report_available"])
        with patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value={"status": "no_observation"}):
            self.app.state.worker.workflow.run(identifier)
        with self.store.session() as session:
            result = session.get(Assessment, identifier).context["findings"]
            self.assertEqual(result["coverage"]["details"]["coverage_status"], "UNRESOLVED")
            self.assertEqual(result["coverage"]["details"]["policy_schedule_id"], schedule["id"])
        overview = self.client.get(detail_path + "?overview=true").json()
        self.assertTrue(overview["report_available"])
        self.assertTrue(overview["has_policy_schedule"])
        self.assertFalse({"findings", "triage", "policy_schedule"} & overview.keys())
        report = self.client.get(report_path)
        self.assertEqual(report.status_code, 200, report.text[:200])
        self.assertEqual(report.headers["Content-Type"], "application/pdf")
        self.assertEqual(report.headers["Content-Disposition"], f'attachment; filename="{detail["reference"]}-investigation.pdf"')
        pdf = PdfReader(BytesIO(report.content))
        report_text = "\n".join(page.extract_text() for page in pdf.pages)
        for expected in ("Investigation report", detail["reference"], schedule["policy_number"], "Decision summary", "Specialist findings"):
            self.assertIn(expected, report_text)
        coverage_text = report_text.split("Coverage /", 1)[1]
        self.assertIn("Unconfirmed", coverage_text)
        preview_report = self.client.get(report_path + "?preview=true")
        self.assertEqual(preview_report.status_code, 200)
        self.assertNotIn("Content-Disposition", preview_report.headers)
        preview_pages = preview_report.json()["pages"]
        self.assertEqual(len(preview_pages), len(pdf.pages))
        self.assertTrue(base64.b64decode(preview_pages[0]["png"]).startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(detail["reference"], preview_pages[0]["text"])
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.get(report_path).status_code, 401)
        self.assertEqual(self.client.get(report_path + "?preview=true").status_code, 401)
        self.assertEqual(self.client.get(detail_path + "/policy-schedule").status_code, 401)
        self.assertEqual(self.client.get(detail_path + "/policy-schedule?preview=true").status_code, 401)

    def test_server_selects_scenarios_and_exports_blind_linked_artifacts(self):
        schedule = self.generate().json()["schedules"][0]
        self.assertEqual(self.client.post("/api/generator/preview", json={"count": 2}).status_code, 422)
        request_schema = self.app.openapi()["components"]["schemas"]["ClaimPreviewRequest"]
        self.assertEqual(set(request_schema["properties"]), {"count"})
        generator = scheduled_claim_generator
        driver_term = "Licensed drivers with the insured's permission may drive the insured car."
        use_term = "Private use only. Use for hire or reward is excluded."
        police_term = "A police report is mandatory for claims above LKR 150,000."
        depreciation_term = "Market value applies after depreciation for aftermarket parts. The limit is stated in the schedule."
        extras = (
            ("Medical expenses", "2000", "Medical expenses for the insured after accidental injury up to LKR 2,000."),
            ("Theft of Parts and Accessories", "35000", "Theft of parts and accessories fitted to the insured car up to LKR 35,000."),
            ("Protection and Removal Costs", "1500", "Protection and removal costs for towing to an approved repairer up to LKR 1,500."),
            ("Glass Breakage Waiver", "75000", "Glass breakage waiver up to LKR 75,000. No excess applies to glass-only replacement."),
        )
        clauses = (*WORDING, driver_term, use_term, police_term, depreciation_term, *(term for _, _, term in extras))
        document = create_document(self.store, title="Wording for scenario eligibility", product="MOTOR", version="2026.2",
            filename="motor-scenarios.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf(clauses)).decode())
        with patch("backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        with self.store.session() as session:
            metadata, sources = policy_schedule_generator.wording_sources(session, document)

        def cite(page):
            return policy_schedule_generator.checked_citation(
                policy_schedule_generator.ScheduleCitation(reference=f"W{page:03d}"), sources)

        schedule["wording"] = metadata
        schedule["wording_references"] = [cite(item["page"]) for item in schedule["wording_references"]]
        for benefit in schedule["cover"]:
            benefit["wording"] = cite(benefit["wording"]["page"])
        schedule["permitted_use"] = "Private use only. Use for hire or reward is excluded."
        schedule["claims_instructions"] = "A police report is mandatory for claims above LKR 150,000."
        schedule["cover"].extend({**schedule["cover"][2], "loss_type": "additional_benefit", "benefit_name": name,
                                  "limit_lkr": limit, "conditions": conditions, "wording": cite(clauses.index(conditions) + 1)}
                                 for name, limit, conditions in extras)
        second = deepcopy(schedule)
        second.update(id="second-schedule", policy_number="SECOND-SCHEDULE", insured_name="Saman Perera")
        second["vehicle"].update(registration="SYN-SP-CAB-1043", chassis_number="SYN-CH-24576")
        second["cover"][0].update(conditions=depreciation_term, wording=cite(clauses.index(depreciation_term) + 1))
        with self.store.session.begin() as session:
            row = session.get(PolicySchedule, schedule["id"])
            row.payload = schedule
            row.document_id = document
            session.add(PolicySchedule(id=second["id"], policy_number=second["policy_number"], document_id=document,
                                       holder_key=policy_schedule_generator.identity_key(second["insured_name"]),
                                       registration_key=policy_schedule_generator.identity_key(second["vehicle"]["registration"]), payload=second))
        with patch.dict("os.environ", {"CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}):
            initial = generator.plan_scheduled_claims(self.store, 20)
        self.assertNotIn(generator.REUSED, {plan.variant for plan in initial})
        prior = next(plan for plan in initial if plan.variant == generator.AUTO_REPAIR)
        prior_day = date.fromisoformat(schedule["starts_on"]) + timedelta(days=60)
        prior = replace(prior, loss_date=prior_day, first_report_day=prior_day, last_report_day=prior_day)

        def prior_model(request):
            name, _, output = scheduled_model_output(request)
            return tool_response(name, output)

        with httpx.Client(transport=httpx.MockTransport(prior_model)) as client, patch.object(
                generator, "generation_client", return_value=nullcontext((client, "fixture"))), patch(
                "backend.api.workspace_api.plan_scheduled_claims", return_value=[prior]):
            response = self.client.post("/api/generator/preview", json={})
            self.assertEqual(response.status_code, 200, response.text)
            prior_id = response.json()["batch_id"]
            self.assertEqual(self.client.post(f"/api/generator/preview/{prior_id}/commit").status_code, 201)
        with patch.object(generator.secrets, "randbits", return_value=9), patch.dict("os.environ", {"CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}):
            plans = generator.plan_scheduled_claims(self.store, 20)
            self.assertEqual(plans, generator.plan_scheduled_claims(self.store, 20))
        self.assertCountEqual([plan.route for plan in plans], list(SCENARIO_ROUTES) * 5)
        self.assertEqual(len({plan.seed for plan in plans}), 20)
        by_variant = {plan.variant: plan for plan in plans}
        self.assertEqual(set(by_variant), {variant for variants in generator.SCENARIO_VARIANTS.values() for variant in variants})
        self.assertTrue(all(len(generator.SCENARIO_VARIANTS[route]) == 5 for route in SCENARIO_ROUTES))
        calls, seeds, rejected = [], [], set()

        def incident_model(request):
            name, prompt, output = scheduled_model_output(request)
            calls.append(name)
            self.assertIn(f"Verified policy wording JSON: {json.dumps(plan.wording_clauses, ensure_ascii=False)}", prompt)
            if name == scheduled_claim_generator.CLAIM_TOOL:
                self.assertIn(json.dumps(plan.schedule, ensure_ascii=False), prompt)
                assignment = json.JSONDecoder().raw_decode(prompt.split("Assignment JSON: ", 1)[1])[0]
                self.assertEqual(assignment["route"], route)
                seeds.append(assignment["scenario_seed"])
                if variant == generator.AUTO_REPAIR and "amount" not in rejected:
                    rejected.add("amount")
                    output["claim"]["gross_amount"] = "125000.01"
                else:
                    self.assertEqual("assigned range and leave additional_costs empty" in prompt, variant == generator.AUTO_REPAIR)
            else:
                for token in ("scenario_seed", "fraud_pattern", "ground_truth", '"incident_details"', *SCENARIO_ROUTES, *by_variant):
                    self.assertNotIn(token, prompt)
                self.assertIn("Relevant schedule fields JSON", prompt)
                self.assertIn("Claim submission JSON", prompt)
                if variant == generator.ITEMS_MISSING and "label" not in rejected:
                    rejected.add("label")
                    output["documents"][0]["name"] = "need_information.txt"
                elif variant == generator.ITEMS_MISSING:
                    self.assertIn("ordinary document filenames", prompt)
                    output["documents"][1]["commercial_trip_verified"] = True
                if variant == generator.AUTO_TOWING:
                    output["documents"][0]["beneficiary"] = "Unassigned field"
                if variant == generator.COST_REGISTRATION_MISSING:
                    identifier = output["documents"][0]["registration"]
                    output["documents"][0]["name"] = f"Estimate {identifier}.txt"
                    output["documents"][0]["line_items"][0]["description"] += f" for {identifier}"
                if variant == generator.UNLISTED and "math" not in rejected:
                    rejected.add("math")
                    output["documents"][0]["line_items"][0]["amount"] = "42000"
                elif variant == generator.UNLISTED:
                    self.assertIn("sum to the printed total", prompt)
            return tool_response(name, output)

        with httpx.Client(transport=httpx.MockTransport(incident_model)) as client, patch.object(
                scheduled_claim_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            for variant, plan in by_variant.items():
                route = plan.route
                with self.subTest(route=route, variant=variant), patch(
                        "backend.api.workspace_api.plan_scheduled_claims", return_value=[plan]):
                    calls.clear()
                    seeds.clear()
                    old_fields = {"route": "fraud_escalation", "scenario_seed": 0} if route == "auto_approval" else {}
                    response = self.client.post("/api/generator/preview", json=old_fields)
                    self.assertEqual(response.status_code, 200, response.text)
                    batch = response.json()
                    [fixture] = batch["fixtures"]
                    self.assertEqual(set(fixture), {"claim", "scenario", "scenario_type"})
                    claim = fixture["claim"]
                    claim_id = claim["claim_id"]
                    documents = claim["documents"]
                    cost, *supporting = documents
                    self.assertTrue(all(doc["registration"] == claim["registration"] for doc in supporting))
                    if variant == generator.CLERICAL:
                        self.assertLess(Decimal(cost["total"]), Decimal(claim["gross_amount"]))
                    else:
                        self.assertEqual(cost["total"], claim["gross_amount"])
                    if variant == generator.UNLISTED:
                        self.assertNotEqual(cost["registration"], claim["registration"])
                    elif variant == generator.COST_REGISTRATION_MISSING:
                        self.assertIsNone(cost["registration"])
                        self.assertNotIn(claim["registration"], claim["unstructured_inputs"]["repair_estimate_text"])
                    else:
                        self.assertEqual(cost["registration"], claim["registration"])
                    self.assertLessEqual(plan.schedule["starts_on"], claim["loss_date"])
                    self.assertLessEqual(claim["loss_date"], plan.schedule["ends_on"])
                    if variant == generator.BACKDATED:
                        self.assertLess(cost["date"], plan.schedule["starts_on"])
                        self.assertLess(documents[1]["incident_date"], plan.schedule["starts_on"])
                    self.assertLessEqual(plan.minimum_amount, Decimal(claim["gross_amount"]))
                    self.assertLessEqual(Decimal(claim["gross_amount"]), plan.maximum_amount)
                    if variant == generator.INFLATED:
                        self.assertEqual(cost["line_items"][0], cost["line_items"][1])
                    if variant == generator.DAMAGE_BOUNDARY:
                        self.assertLess(documents[2]["prior_damage_recorded_on"], claim["loss_date"])
                    if variant == generator.POLICE_MISSING:
                        self.assertNotIn("police_report", {doc["type"] for doc in documents})
                    if variant == generator.ITEMS_MISSING:
                        self.assertNotIn("commercial_trip_verified", documents[1])
                    if variant == generator.PAYMENT_MISSING:
                        self.assertEqual(cost["payment_status"], "pro_forma")
                    if variant == generator.AUTO_TOWING:
                        self.assertIsNone(cost["beneficiary"])
                    if variant == generator.SHARED_LIABILITY:
                        self.assertEqual(documents[3]["type"], "witness_statement")

                    path = f"/api/generator/batches/{batch['batch_id']}/dataset"
                    self.assertEqual(self.client.get(path + ".zip").status_code, 404)
                    commit = self.client.post(f"/api/generator/preview/{batch['batch_id']}/commit")
                    self.assertEqual(commit.status_code, 201, commit.text)
                    evaluation = self.client.get(f"/api/evaluation?batch_id={batch['batch_id']}").json()
                    self.assertEqual(evaluation["pending"], 1)
                    self.assertEqual(self.client.get(path + ".csv").status_code, 200)
                    download = self.client.get(path + ".zip")
                    self.assertEqual(download.status_code, 200, download.text[:100])
                    with ZipFile(BytesIO(download.content)) as archive:
                        parts = {name: json.loads(archive.read(name)) for name in archive.namelist()}
                    self.assertEqual(parts["policy_schedules.json"], [plan.schedule])
                    self.assertEqual(parts["case_links.json"], [{"claim_id": claim_id, "schedule_id": plan.schedule["id"]}])
                    [submission] = parts["claim_submissions.json"]
                    self.assertEqual(submission["claim_id"], claim_id)
                    self.assertFalse({"unstructured_inputs", "documents", "route", "scenario_seed"} & submission.keys())
                    self.assertIn("loss_date", submission)
                    self.assertEqual([row["fields"] for row in parts["attachments.json"]], claim["documents"])
                    self.assertTrue(all(row["claim_id"] == claim_id and row["text"] for row in parts["attachments.json"]))
                    [truth] = parts["ground_truth.json"]
                    self.assertEqual((truth["claim_id"], truth["route"]), (claim_id, route))
                    seed = truth["scenario_seed"]
                    self.assertEqual(set(seeds), {seed})
                    self.assertTrue(0 < seed < 2**53)
                    self.assertEqual(truth["scenario_variant"], variant)
                    self.assertEqual(truth["policy_wording_clauses"], list(plan.wording_clauses))
                    self.assertTrue(truth["policy_wording_clauses"])
                    basis = truth["routing_basis"]
                    self.assertEqual(basis["route"], truth["route"])
                    self.assertEqual(basis["document_id"], document)
                    self.assertEqual(basis["document_hash"], plan.schedule["wording"]["file_hash"])
                    self.assertEqual(set(basis["policy_refs"]), {clause["reference"] for clause in plan.wording_clauses})
                    self.assertTrue(basis["reason"] and basis["evidence"])
                    if variant == generator.POLICE_MISSING:
                        self.assertIn("LKR 150,000.00", basis["reason"])
                    if variant == generator.AUTO_TOWING:
                        unsubstantiated = deepcopy(claim)
                        unsubstantiated.pop("unstructured_inputs")
                        unsubstantiated["documents"][0]["payment_status"] = "pro_forma"
                        with self.assertRaisesRegex(generator.GenerationError, "does not support the assigned route"):
                            generator.routing_basis(unsubstantiated, None, plan)
                        disputed_schedule = {**plan.schedule, "endorsements": [
                            "Accidental damage excess applies to every event under Section I, including costs and expenses."
                        ]}
                        with self.assertRaisesRegex(generator.GenerationError, "does not support the assigned route"):
                            generator.routing_basis(claim, None, replace(plan, schedule=disputed_schedule))
                    self.assertEqual([item["kind"] for item in truth["discrepancies"]],
                                     [variant] if route in {"fraud_escalation", "need_information"} else [])
                    self.assertEqual(truth["fraud_pattern"], variant if route == "fraud_escalation" else None)
                    if variant in {generator.AUTO_REPAIR, generator.ITEMS_MISSING, generator.UNLISTED}:
                        self.assertEqual(len(calls), 3)
                    else:
                        self.assertEqual(len(calls), 2)
                    with self.store.session() as session:
                        self.assertEqual(session.get(SyntheticClaimSchedule, claim_id).payload, plan.schedule)
                        for attachment in documents:
                            self.assertEqual(session.get(SyntheticClaimAttachment, attachment["id"]).payload["fields"], attachment)
                    if route != "auto_approval":
                        [identifier] = commit.json()["assessment_ids"]
                        with patch.dict("os.environ", {"CLAIMS_LLM_MODE": "deterministic", "CLAIMS_SELECTED_PDF_WORKFLOW": "v3"}), patch(
                                "backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value={"status": "no_observation"}):
                            self.assertEqual(self.client.post(f"/api/claims/{identifier}/process").status_code, 202)
                            self.app.state.worker.workflow.run(identifier)
                        with self.store.session() as session:
                            decision = session.get(DecisionRecord, identifier).payload
                            intake = session.get(Assessment, identifier).context["findings"]["intake"]["details"]
                        self.assertEqual(decision["route"], SCENARIO_ROUTES[route].value)
                        if variant == generator.REUSED:
                            self.assertEqual([flag.get("code") for flag in intake["fraud_indicators"]], ["VERIFIED_COST_DOCUMENT_REUSE"])
                    else:
                        finding = {"details": {"coverage_status": "COVERED", "policy_rag": {"cited": True, "applicability": "applicable"},
                                               "coverage_items": [{"outcome": "covered"}]}}
                        if variant == generator.AUTO_GLASS:
                            waived = {"item": "M80 Compulsory Excess applicability", "outcome": "not_covered"}
                            finding["details"]["coverage_items"].append(waived)
                            disputed = deepcopy(finding)
                            disputed["details"]["coverage_items"].append({"item": "Bodywork repair", "outcome": "not_covered"})
                            apply_matched_schedule(disputed, claim, plan.schedule)
                            verify_scheduled_repair(disputed, claim, plan.schedule)
                            self.assertNotIn("scheduled_repair_verified", disputed["details"])
                        apply_matched_schedule(finding, claim, plan.schedule)
                        verify_scheduled_repair(finding, claim, plan.schedule)
                        self.assertTrue(finding["details"].get("scheduled_repair_verified"), finding)
                        if variant == generator.AUTO_GLASS:
                            self.assertEqual(finding["details"]["waived_excess_terms"], [waived])
        self.assertEqual(rejected, {"amount", "label", "math"})

        with self.store.session.begin() as session:
            for payload in (schedule, second):
                row = session.get(PolicySchedule, payload["id"])
                unverified = deepcopy(payload)
                unverified["claims_instructions"] = police_term.replace("150,000", "160,000")
                unverified["cover"][-1]["wording"]["quote"] = "Glass breakage claim with no excess, up to LKR 75,000."
                row.payload = unverified
        plans = generator.plan_scheduled_claims(self.store, 20)
        self.assertFalse({generator.POLICE_MISSING, generator.AUTO_GLASS} & {plan.variant for plan in plans})
        self.assertIn(generator.DISGUISED, {plan.variant for plan in plans})

        with self.store.session.begin() as session:
            for payload in (schedule, second):
                row = session.get(PolicySchedule, payload["id"])
                row.payload = {**payload, "wording": {**payload["wording"], "file_hash": "stale"}}
        with self.assertRaisesRegex(ValueError, "matches an available indexed wording version"):
            generator.plan_scheduled_claims(self.store, 6)

    def test_rejected_wording_verdict_or_fixed_limit_mismatch_saves_nothing(self):
        self.draft["territorial_limits"] = "Sri Lanka and India"
        conflict = "The schedule extends territorial cover to India."
        results = [self.generate(concerns=[conflict], blockers=[
            wording_blocker(["territorial_limits"], "Sri Lanka and India", conflict, 1)])]
        self.draft["territorial_limits"] = "Sri Lanka"
        self.draft["cover"][2]["limit_lkr"] = "3000000"
        results.append(self.generate())
        for response in results:
            self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("fixed wording limit", results[1].json()["detail"])
        self.assertIn("territorial_limits", results[0].json()["detail"])
        self.assertIn("W001 (page 1)", results[0].json()["detail"])
        self.assertEqual(self.client.get("/api/policy-schedules").json()["schedules"], [])

    def test_structured_limit_basis_findings_are_checked_against_the_draft(self):
        self.draft["cover"][2]["limit_basis"] = "scheduled_amount"
        conflict = "The draft treats the third-party property limit as schedule-set. The PDF fixes it at LKR 2,000,000."
        for quote in (None, '"Scheduled amount"'):
            with self.subTest(quote=quote):
                finding = wording_blocker(["cover", 2, "limit_basis"], quote, conflict, 3)
                if quote is None:
                    del finding["scheduled_quote"]
                response = self.generate(concerns=[conflict], blockers=[finding])
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn(f"cover.2.limit_basis: {conflict} [W003 (page 3)]", response.json()["detail"])
        self.assertEqual(self.client.get("/api/policy-schedules").json()["schedules"], [])

    def test_repairs_unlimited_property_damage_to_the_verified_per_event_cap(self):
        injury = "Section II-1(a) injury liability: Unlimited"
        limit = "Section II-1(b) property damage: Rs. 5,000,000/-"
        qualifier = "for any one claim or series of claims arising out of one event."
        table = f"{injury}\n{limit}\n{qualifier}\nSection III medical expenses: Rs. 2,000 per accident."
        pages = [(1, "\n".join(WORDING[:2])), (2, table), *enumerate(WORDING[4:], 3)]
        document = create_document(self.store, title="Motor wording with separate liability limits", product="MOTOR", version="1",
            filename="liability.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(pages, [2])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        references = dict(zip(WORDING, ("W001", "W001", "W002", "W002", "W003", "W004", "W005"), strict=True))
        draft = contract_fixture(references)
        draft["cover"][2].update(limit_basis="unlimited", limit_lkr=None, wording={"reference": "W002"},
                                 conditions="Section II-1(b) property damage is unlimited.")
        draft["cover"][3].update(limit_basis="unlimited", conditions=injury)
        conflict = "The draft treats property damage as unlimited, but Section II-1(b) fixes a cap of Rs. 5,000,000 per event."
        basis_finding = {"kind": "wording", "path": ["cover", 2, "limit_basis"], "reason": conflict,
                         "evidence": [{"passage": "W002", "start_line": 2, "end_line": 3}]}
        condition_finding = {**basis_finding, "path": ["cover", 2, "conditions"],
                             "reason": "The corrected conditions omit the per-event qualifier."}

        for amount, conditions, expected_status in (
                ("2000", f"{limit} {qualifier}", 422),
                ("5000000", limit, 422),
                ("5000000", f"{limit} {qualifier}", 201)):
            with self.subTest(amount=amount, conditions=conditions):
                calls = []

                def model(request):
                    body = json.loads(request.content)
                    name = body["tool_choice"]["name"]
                    prompt = body["messages"][0]["content"]
                    calls.append(name)
                    if name == "draft_policy_schedules":
                        payload = draft
                    elif name == "verify_policy_schedules":
                        if calls.count(name) == 1:
                            payload = {"concerns": [conflict]}
                        else:
                            [case] = json.loads(prompt.rsplit("\n\n", 1)[1])
                            self.assertEqual(case["draft"]["cover"][2]["limit_lkr"], "5000000")
                            payload = {"concerns": [] if qualifier in conditions else [condition_finding["reason"]]}
                    elif name == "confirm_policy_schedule_blockers":
                        payload = {"blockers": [basis_finding if calls.count(name) == 1 else condition_finding]}
                    elif name == "repair_policy_schedule":
                        repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                        replacements = {"limit_basis": "wording_amount", "limit_lkr": amount,
                                        "wording": {"reference": "W002", "start_line": 2, "end_line": 3}, "conditions": conditions}
                        self.assertEqual([edit["path"] for edit in repair["allowed_edits"]],
                                         [["cover", 2, field] for field in replacements])
                        self.assertEqual(repair["feedback"][0]["evidence"][-1], {"passage": "W002", "quote": f"{limit} {qualifier}"})
                        payload = {"changes": [{"path": ["cover", 2, field], "operation": "set", "value_json": json.dumps(value)}
                                               for field, value in replacements.items()]}
                    else:
                        self.assertEqual(name, "propose_schedule_corrections")
                        payload = {"options": []}
                    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                        {"type": "tool_use", "name": name, "input": {"cases": [payload]}}]})

                with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                        policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
                    response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})
                self.assertEqual(response.status_code, expected_status, response.text)
                self.assertEqual(calls[:4], ["draft_policy_schedules", "verify_policy_schedules",
                                             "confirm_policy_schedule_blockers", "repair_policy_schedule"])
                if expected_status == 422:
                    self.assertIn(conflict, response.json()["detail"])
                    self.assertEqual(self.client.get("/api/policy-schedules").json(), {"schedules": [], "corrections": []})
                    continue
                self.assertEqual(calls[4:], ["verify_policy_schedules"])
                self.assertEqual(response.json()["corrections"], [])
                [saved] = response.json()["schedules"]
                self.assertEqual(saved["cover"][2]["limit_basis"], "wording_amount")
                self.assertEqual(saved["cover"][2]["limit_lkr"], "5000000")
                self.assertIn(qualifier, saved["cover"][2]["conditions"])
                self.assertIn(qualifier, saved["cover"][2]["wording"]["quote"])
                self.assertEqual(saved["cover"][2]["wording"]["page"], 2)
                self.assertEqual(saved["cover"][3]["limit_basis"], "unlimited")
                self.assertEqual(saved["vehicle"], draft["vehicle"])
                self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)

    def test_removes_definition_only_excess_and_repairs_property_cap_in_the_same_draft(self):
        definition = "Excess - The amount shown in the schedule which the insured bears on each and every claim."
        clauses = (*WORDING[:4], definition, *WORDING[5:])
        document = create_document(self.store, title="Motor wording with a definition and optional excess endorsements", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        draft = deepcopy(self.draft)
        restriction = "Own damage and theft limits are shown in the schedule."
        draft["cover"][0].update(compulsory_excess_basis="scheduled_amount", compulsory_excess_lkr="25000",
                                 conditions=f"{restriction} A compulsory excess of LKR 25,000 applies to every Section I claim.")
        draft["cover"][1].update(compulsory_excess_lkr="0")
        draft["cover"][2].update(limit_basis="unlimited", limit_lkr=None, conditions="Third-party property damage is unlimited.")
        excess_conflict = "The definition alone does not impose an excess. Neither applicable endorsement is active."
        limit_conflict = "The property damage limit is fixed by the wording."
        blockers = [
            {"kind": "wording", "path": ["cover", 0, "compulsory_excess_basis"], "reason": excess_conflict,
             "evidence": [{"passage": "W005", "start_line": 1}]},
            wording_blocker(["cover", 2, "limit_basis"], None, limit_conflict, 3),
        ]
        replacements = {
            (0, "compulsory_excess_basis"): None, (0, "compulsory_excess_lkr"): "0", (0, "conditions"): restriction,
            (2, "limit_basis"): "wording_amount", (2, "limit_lkr"): "2000000", (2, "conditions"): WORDING[2],
        }
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": blockers}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                allowed = {tuple(edit["path"]) for edit in repair["allowed_edits"]}
                self.assertEqual(allowed, {("cover", *path) for path in replacements} | {("cover", 2, "wording")})
                self.assertIn({"passage": "W005", "quote": definition},
                              [item for finding in repair["feedback"] for item in finding["evidence"]])
                case = {"changes": [{"path": ["cover", *path], "operation": "set", "value_json": json.dumps(value)}
                                    for path, value in replacements.items()]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                if calls.count(name) == 1:
                    case = {"concerns": [excess_conflict, limit_conflict]}
                else:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    for (index, field), value in replacements.items():
                        self.assertEqual(review["draft"]["cover"][index][field], value)
                    self.assertEqual(review["draft"]["endorsements"], [])
                    case = {"concerns": []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "verify_policy_schedules"])
        self.assertEqual(response.json()["corrections"], [])
        [saved] = response.json()["schedules"]
        for (index, field), value in replacements.items():
            self.assertEqual(saved["cover"][index][field], value)
        self.assertEqual(saved["endorsements"], [])
        self.assertEqual(saved["vehicle"], draft["vehicle"])
        self.assertEqual(saved["cover"][0]["wording"]["page"], 2)
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)

    def test_adds_missing_wording_benefit_and_repairs_existing_row_without_changing_cover_type(self):
        notes = ("Third Party, Fire and Theft restricts Section I to covered fire and theft perils.\n"
                 "Third Party and Act Insurance cancel Section III.")
        medical = "Section III Medical Expenses. Reasonable medical expenses for eligible occupants are limited to Rs. 2,000 per accident."
        towing = "Section I pays up to Rs. 1,000 per accident for eligible towing after a covered loss."
        clauses = (*WORDING, notes, medical, towing)
        document = create_document(self.store, title="Motor wording with medical expenses and towing", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        draft = deepcopy(self.draft)
        draft["cover_type"] = "third_party_fire_theft"
        draft["wording_references"].append({"reference": "W008"})
        draft["cover"][0].update(limit_basis="excluded", limit_lkr=None, compulsory_excess_lkr="0",
                                 conditions="Accidental damage is excluded under this cover type.")
        draft["cover"][1]["conditions"] = "Section I applies to all perils."
        extra = {"loss_type": "additional_benefit", "benefit_name": "Towing", "limit_basis": "wording_amount", "limit_lkr": "1000",
                 "compulsory_excess_lkr": "0", "compulsory_excess_basis": None, "voluntary_excess_lkr": "0",
                 "conditions": "Eligible towing after a covered loss, up to LKR 1,000 per accident.", "wording": {"reference": "W010"}}
        draft["cover"].append(extra)
        addition = {**extra, "benefit_name": "Medical Expenses", "limit_lkr": "2000", "conditions": medical, "wording": {"reference": "W009"}}
        restriction = "Section I applies to covered fire and theft perils."
        blockers = [
            {"kind": "wording", "path": ["cover"], "reason": "Section III remains in force and Medical Expenses is missing.",
             "evidence": [{"passage": "W008", "start_line": 1, "end_line": 2}, {"passage": "W009", "start_line": 1}]},
            {"kind": "wording", "path": ["cover", 1, "conditions"], "reason": "Section I is limited to fire and theft perils.",
             "evidence": [{"passage": "W008", "start_line": 1}]},
        ]
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": blockers}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                self.assertEqual({tuple(edit["path"]): edit["operation"] for edit in repair["allowed_edits"]},
                                 {("cover",): "append", ("cover", 1, "conditions"): "set"})
                case = {"changes": [{"path": ["cover"], "operation": "append", "value": [addition]},
                                    {"path": ["cover", 1, "conditions"], "operation": "set", "value": restriction}]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                if calls.count(name) == 1:
                    case = {"concerns": [item["reason"] for item in blockers]}
                else:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    self.assertEqual(review["draft"]["cover"][-2:], [extra, addition])
                    self.assertEqual(review["draft"]["cover"][1]["conditions"], restriction)
                    case = {"concerns": []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "verify_policy_schedules"])
        self.assertEqual(response.json()["corrections"], [])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")
        self.assertEqual(saved["vehicle"], draft["vehicle"])
        self.assertEqual([item["benefit_name"] for item in saved["cover"][4:]], ["Towing", "Medical Expenses"])
        self.assertEqual(saved["cover"][-1]["limit_lkr"], "2000")
        self.assertEqual(saved["cover"][-1]["wording"]["quote"], medical)
        self.assertEqual(saved["cover"][1]["conditions"], restriction)
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)
        self.assertEqual(policy_schedule_generator.variation_issues(saved, [saved]),
                         policy_schedule_generator.variation_issues({**saved, "cover": saved["cover"][:4]}, [saved]))

    def test_reinstates_medical_cover_when_the_review_cites_only_the_cover_type_note(self):
        heading = "Section Ill - Medical Expenses"
        medical = "The insurer pays reasonable medical expenses up to Rs. 2,000 per accident\nfor eligible occupants, except paid drivers."
        liability = f"Section II property damage is limited to Rs. 5,000,000 per event.\n{heading}\n{medical}"
        notes = ("Third Party, Fire and Theft restricts Section I to fire and theft perils.\n"
                 "Third Party Insurance and Act Insurance cancel Section III.")
        laid_up = "Fire and Theft while the car is laid up cancels Sections II and III."
        clauses = (*WORDING, liability, notes, laid_up)
        document = create_document(self.store, title="Motor wording with separate cover-type notes", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [8])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        draft = deepcopy(self.draft)
        draft["cover_type"] = "third_party_fire_theft"
        draft["cover"][0].update(limit_basis="excluded", limit_lkr=None, compulsory_excess_lkr="0",
                                 conditions="Accidental damage is excluded under this cover type.")
        draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": "Section III Medical Expenses",
                               "limit_basis": "excluded", "limit_lkr": None, "compulsory_excess_lkr": "0",
                               "compulsory_excess_basis": None, "voluntary_excess_lkr": "0",
                               "conditions": "Section III is cancelled under TPFT.", "wording": {"reference": "W009"}})
        conflict = "Section III remains active under Third Party, Fire and Theft at Rs. 2,000 per accident."
        blocker = {"kind": "wording", "path": ["cover", 4, "limit_basis"], "reason": conflict,
                   "evidence": [{"passage": "W009", "start_line": 1, "end_line": 2}]}
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                source_data = json.loads(prompt.rsplit("\n\n", 1)[1])
                self.assertIn({"passage": "W008", "start_line": 2, "heading": heading}, source_data["section_headings"])
                case = draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [blocker]}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                self.assertIn({"passage": "W008", "quote": f"{heading} {' '.join(medical.splitlines())}"}, repair["feedback"][0]["evidence"])
                [citation] = repair["feedback"][0]["operative_limit_citations"]
                self.assertEqual(citation, {"reference": "W008", "start_line": 2, "end_line": 4, "limit_lkr": "2000"})
                replacements = {"limit_basis": "wording_amount", "limit_lkr": "5000000",
                                "wording": {"reference": "W008", "start_line": 2, "end_line": 4}, "conditions": medical}
                if calls.count(name) == 2:
                    self.assertIn("W008: LKR 2,000", repair["feedback"][-1]["repair_error"])
                    self.assertNotIn("5,000,000", repair["feedback"][-1]["repair_error"])
                    replacements["limit_lkr"] = "2000"
                case = {"changes": [{"path": ["cover", 4, field], "operation": "set", "value_json": json.dumps(value)}
                                    for field, value in replacements.items()]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                if calls.count(name) == 1:
                    case = {"concerns": [conflict]}
                else:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    self.assertEqual(review["draft"]["cover"][4]["limit_lkr"], "2000")
                    self.assertEqual(review["draft"]["cover"][4]["conditions"], medical)
                    case = {"concerns": []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "repair_policy_schedule", "verify_policy_schedules"])
        self.assertEqual(response.json()["corrections"], [])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")
        self.assertEqual(saved["cover"][4]["limit_basis"], "wording_amount")
        self.assertEqual(saved["cover"][4]["limit_lkr"], "2000")
        self.assertEqual(saved["cover"][4]["conditions"], medical)
        self.assertEqual(saved["cover"][4]["wording"]["page"], 8)
        self.assertEqual(saved["cover"][4]["wording"]["quote"], f"{heading}\n{medical}")
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)

    def test_note_only_confirmation_does_not_choose_between_conflicting_named_caps(self):
        sources = {"W002": {"text": "Section III - Medical Expenses\nPays up to Rs. 2,000 per accident.", "ocr": False},
                   "W003": {"text": "Section III - Medical Expenses\nPays up to Rs. 8,000 per accident.", "ocr": False}}
        benefit = {"limit_basis": "excluded", "benefit_name": "Medical Expenses"}
        evidence = [policy_schedule_generator.ReviewEvidence(passage="W004", quote="Third Party, Fire and Theft restricts Section I.")]
        self.assertEqual(policy_schedule_generator.reinstatement_limit_evidence(benefit, evidence, sources), [])

    def test_sets_tpft_medical_limit_from_scope_notes_before_wording_review(self):
        heading = "Section Ill - Medical Expenses"
        medical = "The insurer pays reasonable medical expenses up to Rs. 2,000 per accident for eligible occupants, excluding paid drivers."
        property_limit = "Third-party property limit is LKR 5,000,000 per event."
        tpft = ("3. If the Schedule provides Third Party, Fire and Theft Insurance,\n"
                "the indemnity under Section I shall be limited to loss caused by fire or theft.")
        third_party = "4. For Third Party Insurance, Sections I and Ill of this policy are cancelled."
        act = "5. For Act Insurance, Sections I and Ill of this policy are cancelled."
        clauses = (WORDING[0], f"{WORDING[1]}\n{property_limit}\n{heading}\n{medical}", WORDING[2],
                   f"{WORDING[3]}\n{tpft}\n{third_party}\n{act}", *WORDING[4:])
        document = create_document(self.store, title="Motor wording with numbered cover-type notes", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [2, 4])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        self.draft["cover_type"] = "third_party_fire_theft"
        self.draft["cover"][2].update(limit_lkr="5000000", conditions=property_limit, wording={"reference": "W002", "start_line": 2})
        self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": "Section III – Medical Expenses",
            "limit_basis": "excluded", "limit_lkr": "5000000", "conditions": "Section III is cancelled under TPFT.",
            "wording": {"reference": "W004", "start_line": 4}})
        concern = "Section I under TPFT covers only the listed perils, so the medical conditions should carry that restriction."
        unsupported = {"kind": "wording", "path": ["cover", 4, "conditions"], "reason": concern,
                       "evidence": [{"passage": "W004", "start_line": 2, "end_line": 3}, {"passage": "W002", "start_line": 3, "end_line": 4}]}

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document, concerns=[concern], blockers=[unsupported])

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers"])
        [review] = json.loads(requests.call_args_list[1].args[2].rsplit("\n\n", 1)[1])
        self.assertIn({"reference": "W004", "start_line": 2, "end_line": 4}, review["draft"]["wording_references"])
        [saved] = response.json()["schedules"]
        for schedule in (review["draft"], saved):
            row = schedule["cover"][4]
            self.assertEqual((row["limit_basis"], row["limit_lkr"]), ("wording_amount", "2000"))
            self.assertIn("excluding paid drivers", row["conditions"])
            self.assertEqual(schedule["cover"][2]["limit_lkr"], "5000000")
        self.assertEqual(saved["cover"][4]["wording"]["quote"], f"{heading}\n{medical}")

    def test_tpft_medical_limit_rule_requires_applicable_notes_and_a_per_accident_cap(self):
        clause = "Section III - Medical Expenses\nPays up to Rs. 3,500 per accident."
        notes = ("3. Third Party, Fire and Theft Insurance limits Section I to fire and theft.\n"
                 "4. Third Party Insurance cancels Sections I and III.")
        draft = {"cover_type": "third_party_fire_theft", "cover": [
            {"loss_type": "additional_benefit", "benefit_name": "Medical Expenses", "limit_basis": "excluded", "limit_lkr": None}]}

        def sources(medical, scope):
            return {"W002": {"text": medical, "ocr": False}, "W004": {"text": scope, "ocr": False}}

        apply_limit = policy_schedule_generator.apply_tpft_medical_limit
        self.assertEqual(apply_limit(draft, sources(clause, notes))["cover"][0]["limit_lkr"], "3500")
        for medical, scope in (
            (clause.replace("per accident", "per person"), notes),
            (clause, notes.splitlines()[0]),
            (clause, notes + "\n5. For Third Party, Fire and Theft Insurance, Section III is cancelled."),
        ):
            with self.subTest(medical=medical, scope=scope):
                self.assertEqual(apply_limit(draft, sources(medical, scope)), draft)
        third_party = {**draft, "cover_type": "third_party"}
        self.assertEqual(apply_limit(third_party, sources(clause, notes)), third_party)

    def test_active_towing_endorsement_replaces_the_base_limit_without_duplicating_cover(self):
        base = ("In the event of such Motor Car being disabled by reason of covered damage the Company will bear the reasonable cost "
                "of protection and removal to the nearest approved repairer, but not exceeding Rs. 1,000 per accident.")
        endorsement = ("M35 Towing Charges\nThe maximum amount payable in respect of item 4 of Section I is increased "
                       "to the amount stated in the Schedule.")
        sources = {"W001": {"text": base, "reference": "base", "page": 1, "ocr": False},
                   "W008": {"text": endorsement, "reference": "endorsement", "page": 8, "ocr": False}}
        base_row = {"loss_type": "additional_benefit", "benefit_name": "Protection and Removal Costs", "limit_basis": "wording_amount",
                    "limit_lkr": "1000", "wording": {"reference": "W001", "start_line": 1}}
        extended = {"loss_type": "additional_benefit", "benefit_name": "Towing Extension (M35)", "limit_basis": "scheduled_amount",
                    "limit_lkr": "15000", "conditions": "M35 increases the limit to the schedule amount.",
                    "wording": {"reference": "W008", "start_line": 1, "end_line": 2}}
        draft = {"endorsements": ["M35 Towing Charges"], "cover": [base_row, extended], "wording_references": []}
        merged = policy_schedule_generator.apply_endorsed_towing_limit(draft, sources)
        [benefit] = merged["cover"]
        self.assertEqual(benefit["limit_lkr"], "15000")
        self.assertIn("nearest approved repairer", benefit["conditions"])
        self.assertNotIn("1,000", benefit["conditions"])
        self.assertEqual(merged["wording_references"], [base_row["wording"]])
        without_endorsement = {**draft, "endorsements": []}
        self.assertEqual(policy_schedule_generator.apply_endorsed_towing_limit(without_endorsement, sources), without_endorsement)

    def test_lower_of_market_and_scheduled_value_requires_one_numeric_vehicle_estimate(self):
        clause = ("The Company's liability shall in no case exceed the estimated value of the vehicle specified in the Schedule, "
                  "or the market value at the time of loss, whichever is the less.")
        sources = {**self.sources, "W008": {"text": clause, "reference": "vehicle-cap", "page": 8, "ocr": False}}
        for row in self.draft["cover"][:2]:
            row.update(limit_basis="market_value", limit_lkr=None, conditions=clause)
        validate = policy_schedule_generator.ScheduleDraft.model_validate
        check = policy_schedule_generator.checked_schedule
        with self.assertRaises(policy_schedule_generator.WordingValidationError) as error:
            check(validate(self.draft), sources)
        self.assertIn(("cover", 0, "limit_lkr"), error.exception.corrections)
        self.assertIn(("cover", 1, "limit_basis"), error.exception.corrections)
        for row in self.draft["cover"][:2]:
            row.update(limit_basis="scheduled_amount", limit_lkr="8500000")
        schedule = check(validate(self.draft), sources)
        self.assertEqual([row["limit_lkr"] for row in schedule["cover"][:2]], ["8500000", "8500000"])
        self.draft["cover"][1]["limit_lkr"] = "8000000"
        with self.assertRaises(policy_schedule_generator.WordingValidationError):
            check(validate(self.draft), sources)

    def test_restores_cited_tpft_medical_limit_from_its_clause_before_repairing_conditions(self):
        heading = "Section Ill - Medical Expenses"
        medical = "Up to Rs. 2,000 per accident for reasonable medical expenses for eligible occupants, excluding paid drivers."
        property_limit = "Third-party property limit is Rs. 5,000,000 per event."
        note = "Third Party, Fire and Theft Insurance limits Section I to fire and theft perils."
        third_party = "Third Party Insurance cancels Sections I and III."
        towing = "Towing is limited to Rs. 1,500 per accident."
        clauses = (WORDING[0], f"{WORDING[1]}\n{property_limit}\n{heading}\n{medical}", WORDING[2],
                   f"{WORDING[3]}\n{note}\n{third_party}", *WORDING[4:], towing)
        document = create_document(self.store, title="Motor wording with a separate Medical Expenses section", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [2, 4])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        self.draft["cover_type"] = "third_party_fire_theft"
        self.draft["cover"][0]["conditions"] = note
        self.draft["cover"][2].update(limit_lkr="5000000", conditions=property_limit, wording={"reference": "W002", "start_line": 2})
        self.draft["cover"].extend([
            {"loss_type": "additional_benefit", "benefit_name": "Towing", "limit_basis": "wording_amount", "limit_lkr": "1500",
             "conditions": towing, "wording": {"reference": "W008"}},
            {"loss_type": "additional_benefit", "benefit_name": "Section III Medical Expenses", "limit_basis": "excluded", "limit_lkr": None,
             "conditions": "Section III is cancelled under TPFT.", "wording": {"reference": "W004", "start_line": 3}},
        ])
        exclusion = "The TPFT note restricts Section I only, so Section III Medical Expenses remains active."
        qualifier = "State that the benefit applies to injuries arising from a covered accident."
        amended_conditions = f"{medical} The injuries must arise from a covered accident."
        evidence = [{"passage": "W002", "start_line": 4}, {"passage": "W004", "start_line": 2, "end_line": 3}]
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                first = calls.count(name) == 1
                case = {"blockers": [{"kind": "wording", "path": ["cover", 5, "limit_basis" if first else "conditions"],
                                      "reason": exclusion if first else qualifier, "evidence": evidence}]}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                self.assertEqual([edit["path"] for edit in repair["allowed_edits"]], [["cover", 5, "conditions"]])
                self.assertIn(qualifier, repair["feedback"][-1]["repair_error"])
                case = {"changes": [{"path": ["cover", 5, "conditions"], "operation": "set", "value": amended_conditions}]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                row = review["draft"]["cover"][5]
                step = calls.count(name)
                if step == 1:
                    self.assertEqual(row["limit_basis"], "excluded")
                else:
                    self.assertEqual(row["limit_basis"], "wording_amount")
                    self.assertEqual(row["limit_lkr"], "2000")
                    self.assertEqual(row["wording"], {"reference": "W002", "start_line": 3, "end_line": 4})
                    self.assertEqual(row["conditions"], amended_conditions if step == 3 else f"{heading} {medical}")
                    self.assertIn({"reference": "W004", "start_line": 2, "end_line": 3}, review["draft"]["wording_references"])
                case = {"concerns": [exclusion] if step == 1 else [qualifier] if step == 2 else []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "verify_policy_schedules", "confirm_policy_schedule_blockers", "repair_policy_schedule", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        benefit = saved["cover"][5]
        self.assertEqual((benefit["limit_basis"], benefit["limit_lkr"], benefit["conditions"]), ("wording_amount", "2000", amended_conditions))
        self.assertEqual(benefit["wording"]["quote"], f"{heading}\n{medical}")
        self.assertEqual(saved["cover"][2]["limit_lkr"], "5000000")
        self.assertEqual(saved["cover"][4]["limit_lkr"], "1500")
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")

    def test_non_conflicting_reviewer_observation_does_not_block_saving(self):
        concern = "The benefits match the wording. An optional excess endorsement is available, which the schedule does not activate. This is not a conflict."
        agreement = wording_blocker(["cover", 2, "limit_lkr"], "2000000",
                                    "The cited clause confirms the amount. No blocker on this item.", 3)
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(concerns=[concern], blockers=[agreement])
        self.assertEqual(response.status_code, 201, response.text)
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["endorsements"], [])
        self.assertEqual(saved["vehicle"], self.draft["vehicle"])
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers"])
        self.assertIn(concern, requests.call_args.args[2])

    def test_malformed_confirmation_retries_with_a_single_decision_and_requires_a_valid_result(self):
        for final_decision, expected_status in (({"blockers": None}, 422), ({"blockers": []}, 201)):
            with self.subTest(expected_status=expected_status):
                formats = []

                def provider(request):
                    body = json.loads(request.content)
                    name = body["tool_choice"]["name"]
                    if name == "draft_policy_schedules":
                        tool_input = {"cases": [self.draft]}
                    elif name == "verify_policy_schedules":
                        tool_input = {"cases": [{"concerns": ["Check the territorial limit."]}]}
                    else:
                        self.assertEqual(name, "confirm_policy_schedule_blockers")
                        properties = body["tools"][0]["input_schema"]["properties"]
                        prompt = body["messages"][0]["content"]
                        if "cases" in properties:
                            self.assertIn("cases array with exactly 1", prompt)
                            formats.append("batch")
                            tool_input = {"cases": None}
                        else:
                            self.assertEqual(set(properties), {"blockers"})
                            self.assertIn('submit {"blockers":[]}', prompt)
                            case = json.loads(prompt.rsplit("\n\n", 1)[1])
                            self.assertEqual(case["draft"], self.draft)
                            formats.append("single")
                            tool_input = final_decision
                    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                        {"type": "tool_use", "name": name, "input": tool_input}]})

                with httpx.Client(transport=httpx.MockTransport(provider)) as client, patch.object(
                        policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
                    response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
                self.assertEqual(response.status_code, expected_status, response.text)
                if expected_status == 422:
                    self.assertEqual(formats, ["batch", "single", "single"])
                    self.assertIn("did not return a valid wording-review decision", response.json()["detail"])
                    self.assertEqual(self.client.get("/api/policy-schedules").json(), {"schedules": [], "corrections": []})
                else:
                    self.assertEqual(formats, ["batch", "single"])
                    [saved] = response.json()["schedules"]
                    self.assertEqual(saved["vehicle"], self.draft["vehicle"])

    def test_singleton_confirmation_is_used_only_when_one_decision_is_expected(self):
        disputed = {0: {"draft": self.draft}, 1: {"draft": {**self.draft, "territorial_limits": "Sri Lanka and India"}}}
        blocker = {"kind": "wording", "path": ["territorial_limits"], "reason": "Cover extends to India.",
                   "evidence": [{"passage": "W001", "start_line": 1}]}
        single_territories = []
        cases = {"blockers": []}

        def provider(request):
            body = json.loads(request.content)
            properties = body["tools"][0]["input_schema"]["properties"]
            if "cases" in properties:
                tool_input = {"cases": cases}
            else:
                case = json.loads(body["messages"][0]["content"].rsplit("\n\n", 1)[1])
                territory = case["draft"]["territorial_limits"]
                single_territories.append(territory)
                tool_input = {"blockers": [blocker] if territory == "Sri Lanka and India" else []}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{"type": "tool_use",
                "name": "confirm_policy_schedule_blockers", "input": tool_input}]})

        source_data = json.dumps({"passages": self.sources})
        with httpx.Client(transport=httpx.MockTransport(provider)) as client:
            cases = json.dumps([{"blockers": []}])
            self.assertEqual(policy_schedule_generator.confirm_schedule_blockers(
                client, "fixture", source_data, self.sources, {0: disputed[0]}), {0: []})
            cases = {"blockers": []}
            self.assertEqual(policy_schedule_generator.confirm_schedule_blockers(
                client, "fixture", source_data, self.sources, {0: disputed[0]}), {0: []})
            self.assertEqual(single_territories, [])
            results = policy_schedule_generator.confirm_schedule_blockers(client, "fixture", source_data, self.sources, disputed)
        self.assertEqual(single_territories, ["Sri Lanka", "Sri Lanka and India"])
        self.assertEqual(results[0], [])
        [finding] = results[1]
        self.assertEqual(finding.issue.scheduled_quote, "Sri Lanka and India")
        self.assertIn("W001 (page 1)", finding.message)

    def test_unverifiable_blocker_is_rechecked_without_rewriting_the_draft(self):
        concern = "The schedule extends territorial cover to India."
        wrong = wording_blocker(["territorial_limits"], "Sri Lanka and India", concern, 1)
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(concerns=[concern], blockers=[wrong])
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("territorial_limits: The schedule quote is not in this field", response.json()["detail"])
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "verify_policy_schedules",
                          "confirm_policy_schedule_blockers",
                          "confirm_policy_schedule_blockers", "confirm_policy_schedule_blockers"])
        self.assertEqual(self.client.get("/api/policy-schedules").json(), {"schedules": [], "corrections": []})

    def test_uncited_charge_is_omitted_and_the_schedule_rechecked_after_review_retries(self):
        admin_fee = "An administrative fee may be charged at the rate shown in the individual schedule."
        document = create_document(self.store, title="Motor wording with a schedule-set fee", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate((*WORDING, admin_fee), 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        self.draft["charges"].extend([{"label": "Stamp duty", "category": "levy", "amount_lkr": "50"},
                                      {"label": "Administrative fee", "category": "fee", "amount_lkr": "2800"}])
        concern = "The administrative fee may differ from the required amount."
        finding = {"kind": "wording", "path": ["charges", 2, "amount_lkr"], "reason": concern, "evidence": []}
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                if calls.count(name) > 1:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    [feedback] = review["review_feedback"]
                    self.assertIn("Cite the wording lines that establish the required charge amount", feedback["problem"])
                    self.assertEqual(feedback["charge_wording_candidates"][0]["quote"], admin_fee)
                case = {"blockers": [finding]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                amended = calls.count(name) == 2
                self.assertEqual(review["draft"]["charges"], self.draft["charges"][:2] if amended else self.draft["charges"])
                case = {"concerns": [] if amended else [concern]}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", *(["confirm_policy_schedule_blockers"] * 3),
                                 "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["charges"], self.draft["charges"][:2])
        self.assertEqual(saved["total_premium_lkr"], "55150.00")
        self.assertEqual(response.json()["corrections"], [])

    def test_retry_resolves_bad_evidence_and_structured_value_quotes_on_the_same_draft(self):
        concern = "The cited clause might impose an additional territorial restriction."
        rejected = wording_blocker(["territorial_limits"], "Sri Lanka", concern, 1)
        rejected["evidence"][0]["quote"] = WORDING[2]
        wrong_basis = wording_blocker(["cover", 2, "limit_basis"], "LKR 2,000,000",
                                     "Check whether the limit is set by the schedule.", 3)
        wrong_range = wording_blocker(["cover", 1, "compulsory_excess_lkr"], "5000", "Check the compulsory excess.", 5)
        wrong_range["evidence"] = [{"passage": "W005", "start_line": 1, "end_line": 2}]
        request_batch = policy_schedule_generator.request_batch

        def review_response(client, model, prompt, name, schema, count, **kwargs):
            if name == "confirm_policy_schedule_blockers":
                _, source_data, case_data = prompt.rsplit("\n\n", 2)
                self.assertEqual(json.loads(source_data)["passages"]["W005"]["lines"], {"1": WORDING[4]})
                [case] = json.loads(case_data)
                self.assertEqual(case["draft"], self.draft)
                if feedback := case.get("review_feedback"):
                    evidence, basis, line_range = feedback
                    self.assertIn("The evidence quote is not in W001", evidence["problem"])
                    self.assertIn("The exact excerpt appears in W003", evidence["problem"])
                    self.assertEqual(evidence["finding"]["evidence"], rejected["evidence"])
                    self.assertIn('cover.2.limit_basis: The actual value is "wording_amount"', basis["problem"])
                    self.assertIn("omit scheduled_quote", basis["problem"])
                    self.assertIn("Choose a valid inclusive line range from 1 to 1 in W005", line_range["problem"])
                    return [schema.model_validate({"cases": [{"blockers": []}]}).cases[0]]
            return request_batch(client, model, prompt, name, schema, count, **kwargs)

        with patch.object(policy_schedule_generator, "request_batch", side_effect=review_response) as requests:
            response = self.generate(concerns=[concern, wrong_basis["reason"], wrong_range["reason"]],
                                     blockers=[rejected, wrong_basis, wrong_range])
        self.assertEqual(response.status_code, 201, response.text)
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["vehicle"], self.draft["vehicle"])
        self.assertEqual(saved["territorial_limits"], self.draft["territorial_limits"])
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "verify_policy_schedules",
                          "confirm_policy_schedule_blockers", "confirm_policy_schedule_blockers"])

    def test_citation_only_findings_are_repaired_and_reverified_on_the_same_schedule(self):
        condition = "Private motor cars are insured in Sri Lanka."
        other_term = "The schedule lists own-damage and theft limits (W002)."
        self.draft["special_conditions"] = [f"{other_term} {condition} (W002)"]
        self.draft["wording_references"] = [{"reference": "W005"}, {"reference": "W002"}]
        reason = "W001 supports this territorial condition."
        blocker = {**wording_blocker(["special_conditions", 0], f"{condition} (W002)", reason, 1, 2),
                   "kind": "citation", "incorrect_reference": "W002"}
        calls = []

        def model(request):
            body = json.loads(request.content)
            name = body["tool_choice"]["name"]
            prompt = body["messages"][0]["content"]
            calls.append(name)
            if name == "draft_policy_schedules":
                payload = self.draft
            elif name == "confirm_policy_schedule_blockers":
                self.assertIn("Check the complete clause across adjacent passages", prompt)
                payload = {"blockers": [blocker]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                if calls.count(name) == 1:
                    self.assertIn(self.draft["special_conditions"][0], prompt)
                    payload = {"concerns": [reason]}
                else:
                    self.assertIn(f"{condition} (W001)", prompt)
                    self.assertNotIn(self.draft["special_conditions"][0], prompt)
                    payload = {"concerns": []}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": name, "input": {"cases": [payload]}}]})

        with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["special_conditions"], [f"{other_term} {condition} (W001)"])
        self.assertEqual([citation["page"] for citation in saved["wording_references"]], [5, 2, 1])
        self.assertEqual(saved["vehicle"], self.draft["vehicle"])
        self.assertEqual(saved["base_premium_lkr"], self.draft["base_premium_lkr"])
        self.assertEqual(response.json()["corrections"], [])

    def test_identical_clause_in_an_overlapping_passage_does_not_block_the_schedule(self):
        text = "Submit all requested documents within 15 days of the loss."
        original = f"{text} (W014)"
        sources = {
            "W014": {"text": f"Condition 2. {text}", "reference": "chunk:old", "page": 2},
            "W015": {"text": f"{text}\nFollow the insurer's instructions.", "reference": "chunk:next", "page": 2},
        }
        decision = policy_schedule_generator.ScheduleDecision.model_validate({"blockers": [{
            "kind": "citation", "path": ["special_conditions", 0],
            "reason": "Substitute W015 for W014.", "incorrect_reference": "W014",
            "evidence": [{"passage": "W015", "start_line": 1}],
        }]})
        with patch.object(policy_schedule_generator, "request_batch", return_value=[decision]):
            result = policy_schedule_generator.confirm_schedule_blockers(
                None, "fixture", "{}", sources, {0: {"draft": {"special_conditions": [original]}}})
        self.assertEqual(result, {0: []})

    def test_numbered_evidence_keeps_multiline_ocr_text_and_adjacent_passage_context(self):
        sources = {
            "W029": {"text": "General terms.\n\nCompulsory excess for theft:", "page": 5, "ocr": True},
            "W030": {"text": "Rs. 15 ,OOO\nper event.\nAnother benefit follows.", "page": 5, "ocr": True},
        }
        decision = policy_schedule_generator.ScheduleDecision.model_validate({"blockers": [{
            "kind": "wording", "path": ["cover", 1, "compulsory_excess_lkr"],
            "reason": "The selected theft clause requires LKR 15,000 per event.",
            "evidence": [{"passage": "W029", "start_line": 2},
                         {"passage": "W030", "start_line": 1, "end_line": 2, "quote": "LKR 15,000 per claim"}],
        }]})
        with patch.object(policy_schedule_generator, "request_batch", return_value=[decision]):
            result = policy_schedule_generator.confirm_schedule_blockers(
                None, "fixture", "{}", sources, {0: {"draft": self.draft}})
        [finding] = result[0]
        self.assertEqual(finding.issue.scheduled_quote, "5000")
        self.assertEqual([item.quote for item in finding.issue.evidence], ["Compulsory excess for theft:", "Rs. 15 ,OOO per event."])
        self.assertIn("W029 (page 5), W030 (page 5)", finding.message)

    def test_repairs_fixed_limit_and_removes_unsupported_injury_cap_on_original_draft(self):
        self.draft["cover"][2]["limit_lkr"] = "3000000"
        self.draft["cover"][3].update(limit_basis="wording_amount", limit_lkr="2000")
        response = self.generate(corrections=[
            {"path": ["cover", 2, "limit_lkr"], "operation": "set", "value_json": '"2000000"'},
            {"path": ["cover", 3, "limit_basis"], "operation": "set", "value_json": '"statutory"'},
            {"path": ["cover", 3, "limit_lkr"], "operation": "set", "value_json": "null"},
        ])
        self.assertEqual(response.status_code, 201, response.text)
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["vehicle"], self.draft["vehicle"])
        self.assertEqual(schedule["cover"][2]["limit_lkr"], "2000000")
        self.assertEqual(schedule["cover"][3]["limit_basis"], "statutory")
        self.assertIsNone(schedule["cover"][3]["limit_lkr"])

    def test_schedule_uses_original_pdf_pages_when_a_legacy_index_splits_a_limit(self):
        clauses = list(WORDING)
        prefix = "\nThird-party property limit is Rs. 2,000"
        tokenizer = passage_tokenizer()
        padding = " text" * (CHUNK_TOKENS - len(tokenizer.encode_ordinary(prefix)))
        clauses[2] = padding + prefix + ",000 per event.\nSeparate medical expenses have a Rs. 2,500 limit."
        _, offsets = tokenizer.decode_with_offsets(tokenizer.encode_ordinary(clauses[2]))
        original = [clauses[2][:offsets[CHUNK_TOKENS]], clauses[2][offsets[CHUNK_TOKENS - OVERLAP_TOKENS]:]]
        self.assertTrue(original[0].endswith("Rs. 2,000"))
        self.assertIn("Rs. 2,000,000", original[1])
        raw = wording_pdf()
        extracted = (list(enumerate(clauses, 1)), [])
        document = create_document(self.store, title="Split limits", product="MOTOR", version="1",
            filename="split.pdf", media_type="application/pdf", content=base64.b64encode(raw).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=extracted), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings), patch(
                "backend.evidence.policy_wording_index.CHUNK_FORMAT", LEGACY_CHUNK_FORMAT), patch(
                "backend.evidence.policy_wording_index.token_passages",
                side_effect=lambda text: iter(original) if text == clauses[2] else token_passages(text)):
            index_document(self.store, document)
        with self.store.session() as session:
            wording, sources = policy_schedule_generator.wording_sources(session, document)
        self.assertEqual(wording["source_format"], "pdf_pages")
        self.assertEqual(sources["W003"]["text"], clauses[2])
        with self.store.session.begin() as session:
            session.execute(delete(PolicyPage).where(PolicyPage.document_id == document))
        refs = {clause: next(alias for alias, source in sources.items() if source["page"] == page)
                for page, clause in enumerate(WORDING, 1)}
        self.draft = contract_fixture(refs)
        with self.assertRaisesRegex(policy_schedule_generator.WordingValidationError, "contains more than one monetary amount"):
            policy_schedule_generator.checked_schedule(policy_schedule_generator.ScheduleDraft.model_validate(self.draft), sources)
        self.draft["cover"][2]["wording"]["start_line"] = 3
        with self.assertRaisesRegex(policy_schedule_generator.WordingValidationError, "fixed wording limit"):
            policy_schedule_generator.checked_schedule(policy_schedule_generator.ScheduleDraft.model_validate(self.draft), sources)
        self.draft["cover"][2]["wording"]["start_line"] = 2
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests, patch.object(
                policy_schedule_generator, "extract_pdf_pages", return_value=extracted) as extract:
            response = self.generate(document=document)
        extract.assert_called_once_with(raw)
        self.assertEqual(response.status_code, 201, response.text)
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["cover"][2]["limit_lkr"], "2000000")
        citation = schedule["cover"][2]["wording"]
        self.assertEqual(citation["quote"], "Third-party property limit is Rs. 2,000,000 per event.")
        self.assertEqual(citation["reference"], sources["W003"]["reference"])
        self.assertEqual((citation["page"], citation["start_line"], citation["end_line"]), (3, 2, 2))
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])

    def test_scanned_limits_are_interpreted_with_context_and_preserve_original_citations(self):
        clauses = list(WORDING)
        clauses[2] = "Third-party property limit is Rs. 2,OOO,OOO per event."
        clauses[4] += " Towing is limited to Rs. 1 ,OOO per accident under Section I."
        document = create_document(self.store, title="Scanned synthetic wording", product="MOTOR", version="1",
            filename="scanned.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [3, 5])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        condition = "Towing is limited to Rs. 1,000 per accident."
        self.draft["cover"][0]["conditions"] += f" {condition}"
        observation = ("The passage sets the same per-accident scope shown in the conditions. "
                       "The conditions text states Rs. 1,000 but the OCR text 'Rs. 1 ,OOO' reads as Rs. 1,000; "
                       "this reading is unambiguous and agrees with the schedule \u2014 no blocker on this amount.")
        self.assertGreater(len(observation), 200)
        blocker = {**wording_blocker(["cover", 0, "conditions"], condition, observation),
                   "evidence": [{"passage": "W005", "quote": "Towing is limited to Rs. 1,000 per accident under Section I."}]}
        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document, concerns=[observation], blockers=[blocker])
        self.assertEqual(response.status_code, 201, response.text)
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["cover"][0]["conditions"], self.draft["cover"][0]["conditions"])
        self.assertEqual(schedule["cover"][2]["limit_lkr"], "2000000")
        self.assertEqual(schedule["cover"][2]["wording"]["quote"], clauses[2])
        self.assertEqual(schedule["wording_references"][1]["quote"], clauses[4])
        self.assertEqual([call.args[3] for call in requests.call_args_list],
                         ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers"])
        review = requests.call_args
        self.assertIn('"reading_lkr": "1000"', review.args[2])
        self.assertIn(observation, review.args[2])
        self.assertIn("nearby lines and pages", review.kwargs["system"])

        for term, reason in (
            ("Towing is limited to Rs. 2,000 per accident.", "The wording supports Rs. 1,000, so the schedule overstates the limit."),
            ("Towing is limited to Rs. 1,000 per policy period.",
             "The OCR reading matches the scheduled amount, so there is no blocker on this amount, but the clause sets a per-accident limit."),
        ):
            with self.subTest(term=term):
                self.draft.update(insured_name="Nimal Perera", named_drivers=["Nimal Perera"])
                self.draft["vehicle"].update(make="Suzuki", model="Swift", registration="SYN-SP-CAB-3050",
                                            chassis_number="SYN-CH-3050", engine_number="SYN-EN-3050")
                for cover in self.draft["cover"][:2]:
                    cover["limit_lkr"] = "5100000"
                self.draft["cover"][0]["conditions"] = term
                response = self.generate(document=document, concerns=[reason],
                                         blockers=[{**blocker, "scheduled_quote": term, "reason": reason}])
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn(reason, response.json()["detail"])
        [saved] = self.client.get("/api/policy-schedules").json()["schedules"]
        self.assertEqual(saved["id"], schedule["id"])

    def test_omits_an_ambiguous_optional_endorsement_and_reverifies_the_same_cover_type(self):
        note = "Third Party, Fire and Theft Insurance restricts Section I to fire, burglary and theft perils."
        endorsement = "M34 Specified Natural Perils Cover\nFor an additional premium this extends cover to flood and storm. Subject otherwise to the policy terms."
        clauses = (*WORDING[:3], f"{WORDING[3]}\n{note}", *WORDING[4:], endorsement)
        document = create_document(self.store, title="Motor wording with restricted perils", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        draft = deepcopy(self.draft)
        unrelated = {"label": "M80 premium", "category": "endorsement", "amount_lkr": "150"}
        draft.update(cover_type="third_party_fire_theft", endorsements=["M34 Specified Natural Perils Cover", "M80"],
                     special_conditions=["M34 covers storm and flood.", "Notify the insurer promptly."])
        draft["cover"][0]["conditions"] = "Covered perils include storm and flood under M34."
        draft["charges"].extend([{"label": "M34 premium", "category": "endorsement", "amount_lkr": "250"}, unrelated])
        conflict = "M34 conflicts with the TPFT cover-type note. Whether it overrides the restriction is ambiguous."
        finding = {"kind": "wording", "path": ["cover", 0, "conditions"], "reason": conflict,
                   "evidence": [{"passage": "W004", "start_line": 2}, {"passage": "W008", "start_line": 2}]}
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [finding]}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                allowed = [edit["path"] for edit in repair["allowed_edits"]]
                self.assertIn(["endorsements"], allowed)
                self.assertNotIn(["cover_type"], allowed)
                self.assertNotIn(["base_premium_lkr"], allowed)
                retry = calls.count(name) == 2
                if retry:
                    self.assertIn("retain the other selections", repair["feedback"][-1]["repair_error"])
                case = {"changes": [
                    {"path": ["endorsements"], "operation": "set", "value": ["M80"] if retry else []},
                    {"path": ["cover", 0, "conditions"], "operation": "set", "value": note},
                    {"path": ["special_conditions"], "operation": "set", "value": ["Notify the insurer promptly."]},
                    {"path": ["charges"], "operation": "set", "value": [draft["charges"][0], unrelated]},
                ]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                if calls.count(name) == 1:
                    case = {"concerns": [conflict]}
                else:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    self.assertEqual(review["draft"]["endorsements"], ["M80"])
                    self.assertEqual(review["draft"]["cover"][0]["conditions"], note)
                    case = {"concerns": []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "repair_policy_schedule", "verify_policy_schedules"])
        self.assertEqual(response.json()["corrections"], [])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")
        self.assertEqual(saved["endorsements"], ["M80"])
        self.assertEqual(saved["charges"], [draft["charges"][0], unrelated])
        self.assertEqual(saved["total_premium_lkr"], "55250.00")
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)

    def test_inactive_endorsement_premium_is_removed_and_incompatible_selection_is_repaired(self):
        note = "Third Party, Fire and Theft Insurance restricts Section I to fire, burglary and theft perils."
        endorsement = "M34 Specified Natural Perils Cover\nFor an additional premium this extends cover to flood and storm. Subject otherwise to the policy terms."
        clauses = (*WORDING[:3], f"{WORDING[3]}\n{note}", *WORDING[4:], endorsement)
        document = create_document(self.store, title="Motor wording with optional natural perils", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        conflict = "The M34 Natural Perils Endorsement Premium charge must be removed because M34 is not a validly active endorsement under TPFT cover."
        blocker = {"kind": "wording", "path": ["charges", 2], "reason": conflict,
                   "evidence": [{"passage": "W008", "start_line": 2}]}
        for selected in (False, True):
            with self.subTest(selected=selected):
                draft = deepcopy(self.draft)
                draft.update(cover_type="third_party_fire_theft", endorsements=["M80"], special_conditions=[])
                draft["cover"][0]["conditions"] = note
                draft["charges"].extend([{"label": "M80 premium", "category": "endorsement", "amount_lkr": "150"},
                                         {"label": "M34 Natural Perils Endorsement Premium", "category": "endorsement", "amount_lkr": "250"},
                                         {"label": "Stamp duty", "category": "levy", "amount_lkr": "50"}])
                expected_charges = [draft["charges"][i] for i in (0, 1, 3)]
                if selected:
                    draft.update(insured_name="Nimal Perera", named_drivers=["Nimal Perera"])
                    draft["vehicle"].update(model="Aqua", registration="SYN-SP-CAB-3050", chassis_number="SYN-CH-3050", engine_number="SYN-EN-3050")
                    for row in draft["cover"][:2]:
                        row["limit_lkr"] = "5100000"
                    draft["endorsements"].append("M34 Specified Natural Perils Cover")
                    draft["cover"][0]["conditions"] = "M34 extends this benefit to flood and storm."
                calls = []

                def provider(client, model, prompt, name, schema, count, **kwargs):
                    calls.append(name)
                    if name == "draft_policy_schedules":
                        case = draft
                    elif name == "confirm_policy_schedule_blockers":
                        case = {"blockers": [blocker]}
                    elif name == "repair_policy_schedule":
                        self.assertTrue(selected)
                        repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                        self.assertIn(["endorsements"], [edit["path"] for edit in repair["allowed_edits"]])
                        case = {"changes": [
                            {"path": ["endorsements"], "operation": "set", "value": ["M80"]},
                            {"path": ["charges"], "operation": "set", "value": expected_charges},
                            {"path": ["cover", 0, "conditions"], "operation": "set", "value": note},
                        ]}
                    else:
                        self.assertEqual(name, "verify_policy_schedules")
                        if calls.count(name) == 1 and selected:
                            case = {"concerns": [conflict]}
                        else:
                            [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                            self.assertEqual(review["draft"]["charges"], expected_charges)
                            self.assertEqual(review["draft"]["endorsements"], ["M80"])
                            self.assertEqual(review["draft"]["cover"][0]["conditions"], note)
                            case = {"concerns": []}
                    return schema.model_validate({"cases": [case]}).cases

                with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                        policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
                    response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

                self.assertEqual(response.status_code, 201, response.text)
                expected_calls = ["draft_policy_schedules", "verify_policy_schedules"]
                if selected:
                    expected_calls.extend(["confirm_policy_schedule_blockers", "repair_policy_schedule", "verify_policy_schedules"])
                self.assertEqual(calls, expected_calls)
                self.assertEqual(response.json()["corrections"], [])
                [saved] = response.json()["schedules"]
                self.assertEqual(saved["charges"], expected_charges)
                self.assertEqual(saved["total_premium_lkr"], "55300.00")
                self.assertEqual(saved["cover_type"], "third_party_fire_theft")
                self.assertEqual(saved["vehicle"], draft["vehicle"])

    def test_whole_cover_row_conflict_omits_only_the_inactive_endorsement_and_its_terms(self):
        note = "Third Party, Fire and Theft Insurance restricts Section I to fire, burglary and theft perils."
        natural_perils = "For an additional premium this extends Section I to flood and storm."
        medical = "Medical expenses are limited to LKR 2,000 per accident."
        towing = "Towing costs are limited to LKR 1,000 per accident."
        clauses = (*WORDING[:3], f"{WORDING[3]}\n{note}", *WORDING[4:],
                   f"M34 Specified Natural Perils Cover\n{natural_perils}", medical, towing)
        document = create_document(self.store, title="Motor wording with restricted perils", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        self.draft.update(cover_type="third_party_fire_theft", endorsements=["M34 Specified Natural Perils Cover", "M80"],
                          special_conditions=["M34 covers flood and storm.", "Notify the insurer promptly."])
        self.draft["cover"][0]["conditions"] = note + " M34 extends cover to flood and storm."
        for name, basis, amount, terms, ref in (
                ("Medical Expenses", "wording_amount", "2000", medical, "W009"),
                ("Towing", "wording_amount", "1000", towing, "W010"),
                ("Natural Perils", "scheduled_amount", "5000000", natural_perils, "W008")):
            self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": name, "limit_basis": basis,
                                        "limit_lkr": amount, "conditions": terms, "wording": {"reference": ref}})
        self.draft["charges"].extend([{"label": "M34 premium", "category": "endorsement", "amount_lkr": "250"},
                                      {"label": "M80 premium", "category": "endorsement", "amount_lkr": "150"}])
        charges = [self.draft["charges"][0], self.draft["charges"][2]]
        conflict = "M34 expands Section I beyond the Third Party, Fire and Theft perils. This natural perils row is inactive."
        blocker = {"kind": "wording", "path": ["cover", 6], "reason": conflict,
                   "evidence": [{"passage": "W004", "start_line": 2}, {"passage": "W008", "start_line": 2}]}
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [blocker]}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                row_edit = next(edit for edit in repair["allowed_edits"] if edit["path"] == ["cover", 6])
                self.assertEqual(row_edit["operations"], ["set", "remove"])
                case = {"changes": [
                    {"path": ["cover", 6], "operation": "remove"},
                    {"path": ["cover", 0, "conditions"], "operation": "set", "value": note},
                    {"path": ["endorsements"], "operation": "set", "value": ["M80"]},
                    {"path": ["charges"], "operation": "set", "value": charges},
                    {"path": ["special_conditions"], "operation": "set", "value": ["Notify the insurer promptly."]},
                ]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                case = {"concerns": [conflict] if calls.count(name) == 1 else []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")
        self.assertEqual(saved["endorsements"], ["M80"])
        self.assertEqual(saved["charges"], charges)
        self.assertEqual(saved["cover"][0]["conditions"], note)
        self.assertEqual([(row["benefit_name"], row["limit_lkr"]) for row in saved["cover"][4:]],
                         [("Medical Expenses", "2000"), ("Towing", "1000")])
        self.assertEqual(response.json()["corrections"], [])

    def test_endorsement_row_conflict_repairs_its_dependent_terms_on_the_same_schedule(self):
        scope = "Section I is restricted to fire, explosion, self-ignition, lightning, burglary, housebreaking or theft."
        note = f"3. For Third Party, Fire and Theft Insurance, the following provisions apply.\n{scope}"
        natural_perils = "For an additional premium this extends Section I to flood and storm."
        medical = "Medical expenses are limited to LKR 2,000 per accident."
        clauses = (*WORDING[:3], f"{WORDING[3]}\n{note}", *WORDING[4:], f"M34 Specified Natural Perils Cover\n{natural_perils}", medical)
        document = create_document(self.store, title="Motor wording with a closed TPFT peril list", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        self.draft.update(cover_type="third_party_fire_theft", endorsements=["Natural Perils Cover", "M80"],
                          special_conditions=["M34 covers flood and storm.", "Notify the insurer promptly."])
        self.draft["cover"][0].update(conditions="M34 extends the base cover to flood and storm.", wording={"reference": "W008", "start_line": 2})
        self.draft["cover"].extend([
            {"loss_type": "additional_benefit", "benefit_name": "Natural Perils Extension", "limit_basis": "scheduled_amount", "limit_lkr": "5000000",
             "conditions": natural_perils, "wording": {"reference": "W008", "start_line": 2}},
            {"loss_type": "additional_benefit", "benefit_name": "Medical Expenses", "limit_basis": "wording_amount", "limit_lkr": "2000",
             "conditions": medical, "wording": {"reference": "W009"}},
        ])
        original_fee = self.draft["charges"][0]
        unrelated = {"label": "Accidental Damage Excess Endorsement Premium", "category": "endorsement", "amount_lkr": "150"}
        self.draft["charges"].extend([{"label": "Natural Perils Endorsement Premium", "category": "endorsement", "amount_lkr": "250"}, unrelated])
        conflict = "M34 expands Section I beyond the TPFT peril list. The wording provides no precedence rule, so M34 must be omitted."
        blocker = {"kind": "wording", "path": ["endorsements", 0], "reason": conflict,
                   "evidence": [{"passage": "W004", "start_line": 3}, {"passage": "W008", "start_line": 2}]}
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [blocker]}
            elif name == "repair_policy_schedule":
                repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                allowed = {tuple(edit["path"]): edit for edit in repair["allowed_edits"]}
                self.assertEqual(allowed["cover", 4]["operation"], "remove")
                self.assertNotIn(("cover", 5), allowed)
                retry = calls.count(name) == 2
                if retry:
                    self.assertIn("other endorsements' premium lines", repair["feedback"][-1]["repair_error"])
                case = {"changes": [
                    {"path": ["endorsements"], "operation": "set", "value": ["M80"]},
                    {"path": ["cover", 4], "operation": "remove"},
                    {"path": ["cover", 0, "conditions"], "operation": "set", "value": scope},
                    {"path": ["cover", 0, "wording"], "operation": "set", "value": {"reference": "W004", "start_line": 2, "end_line": 3}},
                    {"path": ["charges"], "operation": "set", "value": [original_fee, unrelated] if retry else [original_fee]},
                    {"path": ["special_conditions"], "operation": "set", "value": ["Notify the insurer promptly."]},
                ]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                case = {"concerns": [conflict] if calls.count(name) == 1 else []}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "repair_policy_schedule", "repair_policy_schedule", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["cover_type"], "third_party_fire_theft")
        self.assertEqual(saved["endorsements"], ["M80"])
        self.assertEqual(saved["charges"], [original_fee, unrelated])
        self.assertEqual(saved["total_premium_lkr"], "55250.00")
        self.assertEqual(saved["cover"][0]["wording"]["quote"], note)
        self.assertEqual([(row["benefit_name"], row["limit_lkr"]) for row in saved["cover"][4:]], [("Medical Expenses", "2000")])
        self.assertEqual(response.json()["corrections"], [])

    def test_saves_a_single_verified_correction_when_scoped_repair_cannot_fix_the_conflict(self):
        self.draft["territorial_limits"] = "Sri Lanka and India"
        conflict = "The policy only covers use in Sri Lanka."
        blocker = wording_blocker(["territorial_limits"], self.draft["territorial_limits"], conflict, 1)
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                case = {"blockers": [blocker]}
            elif name == "propose_schedule_corrections":
                case = {"options": [{"label": "Apply the territorial limit", "effect": "Limit use to Sri Lanka.",
                    "references": ["W001"], "changes": [
                        {"path": ["territorial_limits"], "operation": "set", "value": "Sri Lanka"}]}]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                amended = calls.count(name) == 2
                self.assertEqual(review["draft"]["territorial_limits"], "Sri Lanka" if amended else "Sri Lanka and India")
                case = {"concerns": [] if amended else [conflict]}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                 "propose_schedule_corrections", "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["territorial_limits"], "Sri Lanka")
        self.assertEqual(saved["vehicle"], self.draft["vehicle"])
        self.assertEqual(response.json()["corrections"], [])

    def test_fixed_charge_is_checked_against_the_draft_amount_and_cited_lines(self):
        policy_fee = "The fixed policy fee is LKR 2,800."
        sources = {"W008": {"text": policy_fee, "ocr": False, "page": 8}}
        self.draft["charges"][0]["amount_lkr"] = "2800"
        finding = {"kind": "wording", "path": ["charges", 0, "amount_lkr"], "scheduled_quote": "LKR 3,000",
                   "fixed_charge_lkr": "2800", "reason": "Check the policy fee against the fixed charge.",
                   "evidence": [{"passage": "W008", "start_line": 1}]}
        for required in ("2800", "4200"):
            with self.subTest(required=required):
                issue = policy_schedule_generator.ScheduleBlocker.model_validate({**finding, "fixed_charge_lkr": required})
                blocker, problem = policy_schedule_generator.ground_blocker(
                    issue, {"draft": self.draft}, sources, {"W008": [policy_fee]})
                self.assertIsNone(blocker)
                if required == "2800":
                    self.assertIsNone(problem)
                else:
                    self.assertIn("state the exact applicable fixed charge", problem)

    def test_only_wording_supported_charge_lines_reach_review_and_save(self):
        clauses = (*WORDING[:6], "M80 Accidental Damage Excess\nThe insured bears the excess shown in the schedule.",
                   "M31 Passenger Cover\nIn consideration of payment of an additional premium, this benefit applies.\n"
                   "M80 Accidental Damage Excess\nThe insured bears an excess of LKR 1,200 for a Section I claim.\n"
                   "M34 Natural Perils Cover\nThis extension applies without payment of an additional premium.")
        document = create_document(self.store, title="Motor wording with separately priced and unpriced endorsements", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        self.draft["endorsements"] = ["M80", "M31", "M34"]
        self.draft["cover"][0]["compulsory_excess_lkr"] = "1200"
        self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": "Accidental Damage Excess (M80)",
                                   "limit_basis": "scheduled_amount", "limit_lkr": "1200", "compulsory_excess_lkr": "1200",
                                   "compulsory_excess_basis": "scheduled_amount", "conditions": "The insured bears the excess.",
                                   "wording": {"reference": "W008", "start_line": 3, "end_line": 4}})
        passenger = {"label": "M31 Passenger Cover premium", "category": "endorsement", "amount_lkr": "300"}
        stamp_duty = {"label": "Stamp duty", "category": "levy", "amount_lkr": "50"}
        expected = [self.draft["charges"][0], passenger, stamp_duty]
        self.draft["charges"] = [{"label": "M80 Accidental Damage Excess premium", "category": "endorsement", "amount_lkr": "1200"},
                                 *expected, {"label": "M34 premium", "category": "endorsement", "amount_lkr": "400"},
                                 {"label": "VAT", "category": "tax", "amount_lkr": "500"},
                                 {"label": "Handling fee", "category": "fee", "amount_lkr": "75"}]

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [review] = json.loads(requests.call_args.args[2].rsplit("\n\n", 1)[1])
        [saved] = response.json()["schedules"]
        for schedule in (review["draft"], saved):
            self.assertEqual(schedule["charges"], expected)
            self.assertEqual(schedule["endorsements"], ["M80", "M31", "M34"])
            self.assertEqual(schedule["cover"][0]["compulsory_excess_lkr"], "1200")
            self.assertEqual(len(schedule["cover"]), 4)
        self.assertEqual(saved["total_premium_lkr"], "55450.00")

    def test_repairs_and_reverifies_a_fixed_charge_that_gains_evidence_on_retry(self):
        policy_fee = "The fixed policy fee is LKR 2,900."
        clauses = (WORDING[0].replace("Policy fee is LKR 100. ", ""), *WORDING[1:], policy_fee)
        document = create_document(self.store, title="Motor wording with a fixed policy fee", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)
        self.draft["charges"][0]["amount_lkr"] = "2800"
        self.draft["charges"].append({"label": "Stamp duty", "category": "levy", "amount_lkr": "50"})
        concern = "The draft policy fee differs from the fixed fee in the wording."
        calls = []

        def provider(client, model, prompt, name, schema, count, **kwargs):
            calls.append(name)
            if name == "draft_policy_schedules":
                case = self.draft
            elif name == "confirm_policy_schedule_blockers":
                if calls.count(name) > 1:
                    [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                    [feedback] = review["review_feedback"]
                    self.assertEqual(feedback["charge_wording_candidates"][0],
                                     {"passage": "W008", "start_line": 1, "end_line": 1, "quote": policy_fee})
                evidence = [{"passage": "W008", "start_line": 1}] if calls.count(name) == 3 else []
                case = {"blockers": [{"kind": "wording", "path": ["charges", 0, "amount_lkr"],
                    "scheduled_quote": "LKR 2,900", "fixed_charge_lkr": "2900", "reason": concern,
                    "evidence": evidence}]}
            else:
                self.assertEqual(name, "verify_policy_schedules")
                [review] = json.loads(prompt.rsplit("\n\n", 1)[1])
                amended = calls.count(name) == 2
                self.assertEqual(review["draft"]["charges"][0]["amount_lkr"], "2900" if amended else "2800")
                self.assertEqual(review["draft"]["charges"][1], self.draft["charges"][1])
                case = {"concerns": [] if amended else [concern]}
            return schema.model_validate({"cases": [case]}).cases

        with patch.object(policy_schedule_generator, "request_batch", side_effect=provider), patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((None, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": document})

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules",
                                 *(["confirm_policy_schedule_blockers"] * 3), "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["charges"], [{**self.draft["charges"][0], "amount_lkr": "2900"}, self.draft["charges"][1]])
        self.assertEqual(saved["total_premium_lkr"], "57950.00")

    def test_wording_conflict_offers_verified_choices_and_saves_the_selected_schedule(self):
        self.draft["endorsements"] = ["M03", "M80"]
        self.draft["cover"][0]["compulsory_excess_lkr"] = "9000"
        conflict = ("M03 and M80 are alternative endorsements and cannot be applied together (W006, W007). "
                    "M03 applies an excess to both Sections I and II. M80 applies an accidental damage excess to Section I only. "
                    "Applying both would impose overlapping excesses on the same Section I loss.")
        self.assertGreater(len(conflict), 200)
        correction_conflict = "M80 leaves an unsupported Section I excess on the revised schedule."
        common_blocker = wording_blocker(["endorsements"], '["M03", "M80"]', conflict, 6, 7)
        del common_blocker["scheduled_quote"]
        common_blocker["evidence"] = [{"passage": "W006", "start_line": 1}, {"passage": "W007", "start_line": 1}]
        revised_blocker = wording_blocker(["cover", 0, "compulsory_excess_lkr"], "LKR 7,000.00", correction_conflict, 7)
        wrong_quote = wording_blocker(["territorial_limits"], "Sri Lanka and India", "Extended territory needs review.", 1)
        calls = []

        def option(endorsement, effect, excess, passage):
            return {"label": f"Keep {endorsement}", "effect": effect, "references": [f"{passage}, lines 1–2 ({endorsement})"], "changes": [
                {"path": ["endorsements"], "operation": "set", "value": [endorsement]},
                {"path": ["cover", 0, "compulsory_excess_lkr"], "operation": "set", "value": excess},
                {"path": ["cover", 0, "wording"], "operation": "set", "value": {"reference": passage}},
            ]}

        def model(request):
            body = json.loads(request.content)
            name = body["tool_choice"]["name"]
            calls.append(name)
            prompt = body["messages"][0]["content"]
            if name == "draft_policy_schedules":
                cases = [self.draft]
            elif name == "propose_schedule_corrections":
                self.assertIn(conflict, prompt)
                self.assertIn('"endorsements": ["M03", "M80"]', prompt)
                if calls.count(name) == 1:
                    cases = [{"options": [option("M03", "Retain the excess for Sections I and II.", "6000", "W006"),
                                          option("M80", "Apply the accidental damage excess to Section I only.", "7000", "W007")]}]
                else:
                    self.assertIn(correction_conflict, prompt)
                    self.assertIn('"accepted_choices": ["Keep M03"]', prompt)
                    cases = [{"options": [option("M80", "Apply the accidental damage excess to Section I only.", "5000", "W007")]}]
            elif name == "confirm_policy_schedule_blockers":
                confirmations = calls.count(name)
                if confirmations == 1:
                    cases = [{"blockers": [common_blocker, wrong_quote]}]
                elif confirmations == 2:
                    self.assertIn("territorial_limits: The schedule quote is not in this field", prompt)
                    cases = [{"blockers": []}]
                else:
                    cases = [{"blockers": [revised_blocker]}]
            elif calls.count(name) == 1:
                cases = [{"concerns": [conflict]}]
            elif calls.count(name) == 2:
                self.assertIn('"endorsements": ["M03"]', prompt)
                self.assertIn('"endorsements": ["M80"]', prompt)
                cases = [{"concerns": []}, {"concerns": [correction_conflict]}]
            else:
                self.assertIn('"endorsements": ["M80"]', prompt)
                cases = [{"concerns": []}]
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": name, "input": {"cases": cases}}]})

        with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                "confirm_policy_schedule_blockers",
                                "propose_schedule_corrections", "verify_policy_schedules", "confirm_policy_schedule_blockers",
                                "propose_schedule_corrections", "verify_policy_schedules"])
        self.assertEqual(response.json()["schedules"], [])
        [correction] = response.json()["corrections"]
        self.assertEqual(correction["conflicts"], [f"endorsements: {conflict} [W006 (page 6), W007 (page 7)]"])
        first, second = correction["options"]
        self.assertEqual((first["label"], second["label"]), ("Keep M03", "Keep M80"))
        self.assertEqual(second["references"], [{"passage": "W007", "page": 7}])
        self.assertNotIn("schedule", second)
        self.assertEqual(self.client.get("/api/policy-schedules").json(), {"schedules": [], "corrections": [correction]})

        path = f"/api/policy-schedules/corrections/{correction['id']}"
        self.assertEqual(self.client.post(path, json={"option_id": "missing"}).status_code, 404)
        result = self.client.post(path, json={"option_id": second["id"]})
        self.assertEqual(result.status_code, 200, result.text)
        saved = result.json()
        self.assertEqual(saved["id"], correction["id"])
        self.assertEqual(saved["endorsements"], ["M80"])
        self.assertEqual(saved["cover"][0]["compulsory_excess_lkr"], "5000")
        self.assertEqual(saved["cover"][0]["wording"]["page"], 7)
        self.assertEqual(saved["vehicle"], self.draft["vehicle"])
        self.assertEqual(self.client.get(f"/api/policy-schedules/{saved['id']}").json(), saved)
        self.assertEqual(self.client.post(path, json={"option_id": second["id"]}).json(), saved)
        self.assertEqual(self.client.post(path, json={"option_id": first["id"]}).status_code, 409)
        listing = self.client.get("/api/policy-schedules").json()
        self.assertEqual(listing["corrections"], [])
        self.assertEqual([item["id"] for item in listing["schedules"]], [saved["id"]])

    def test_unverifiable_citation_is_corrected_on_the_same_schedule(self):
        citation = self.draft["wording_references"][0].copy()
        source = self.sources[citation["reference"]]
        self.draft["wording_references"][0]["reference"] = "W999"
        response = self.generate(corrections=[{"path": ["wording_references", 0], "operation": "set", "value_json": json.dumps(citation)}])
        self.assertEqual(response.status_code, 201, response.text)
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["wording_references"][0], {"reference": source["reference"], "quote": source["text"], "page": 1, "ocr": False})
        for field in ("insured_name", "vehicle", "starts_on", "ends_on", "base_premium_lkr"):
            self.assertEqual(saved[field], self.draft[field])

    def test_invalid_issue_date_repairs_the_same_schedule_before_verification(self):
        self.draft["issued_on"] = "2025-11-02"
        for changed_field in ("insured_name", "issued_on"):
            with self.subTest(changed_field=changed_field):
                calls = []

                def model(request):
                    body = json.loads(request.content)
                    tool = body["tool_choice"]["name"]
                    calls.append(tool)
                    if tool == "draft_policy_schedules":
                        result = self.draft
                    elif tool == "repair_policy_schedule":
                        prompt = body["messages"][0]["content"]
                        self.assertIn('"issued_on": "2025-11-02"', prompt)
                        self.assertIn('"path": ["issued_on"]', prompt)
                        self.assertNotIn('"path": ["starts_on"]', prompt)
                        result = {"changes": [{"path": [changed_field], "operation": "set", "value_json": '"2025-10-25"'}]}
                    else:
                        prompt = body["messages"][0]["content"]
                        self.assertIn('"issued_on": "2025-10-25"', prompt)
                        self.assertIn('"starts_on": "2025-11-01"', prompt)
                        result = {"concerns": []}
                    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                        {"type": "tool_use", "name": tool, "input": {"cases": [result]}}]})

                with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                        policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
                    response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
                if changed_field == "insured_name":
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertEqual(calls, ["draft_policy_schedules", "repair_policy_schedule"])
                    self.assertEqual(self.client.get("/api/policy-schedules").json()["schedules"], [])
                    continue
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(calls, ["draft_policy_schedules", "repair_policy_schedule", "verify_policy_schedules"])
                [saved] = response.json()["schedules"]
                for field in ("insured_name", "vehicle", "starts_on", "ends_on", "cover_type", "base_premium_lkr"):
                    self.assertEqual(saved[field], self.draft[field])
                self.assertEqual(saved["issued_on"], "2025-10-25")

    def test_malformed_correction_retries_with_native_values_and_keeps_the_original_draft(self):
        self.draft["cover"][2]["wording"]["reference"] = "W999"
        self.draft["cover"][3].update(limit_basis="wording_amount", limit_lkr="2000")
        citation = {"reference": "W003", "start_line": 1}
        for resolves in (False, True):
            with self.subTest(resolves=resolves):
                calls = []

                def model(request):
                    body = json.loads(request.content)
                    tool = body["tool_choice"]["name"]
                    calls.append(tool)
                    if tool == "draft_policy_schedules":
                        result = self.draft
                    elif tool == "repair_policy_schedule":
                        fields = body["tools"][0]["input_schema"]["$defs"]["ScheduleEdit"]["properties"]
                        self.assertIn("value", fields)
                        self.assertNotIn("value_json", fields)
                        prompt = body["messages"][0]["content"]
                        repair = json.loads(prompt.rsplit("\n\n", 2)[1])
                        self.assertEqual(repair["draft"], self.draft)
                        edits = [{"path": ["cover", 3, "limit_basis"], "operation": "set", "value": "statutory"}]
                        if calls.count(tool) == 1:
                            edits.append({"path": ["cover", 2, "wording"], "operation": "set", "value_json": "{reference: W003}"})
                        else:
                            self.assertIn("invalid JSON", repair["feedback"][0]["repair_error"])
                            edits.append({"path": ["cover", 2, "wording"], "operation": "set", "value": citation})
                            limit = {"path": ["cover", 3, "limit_lkr"], "operation": "set"}
                            if resolves:
                                limit["value"] = None
                            edits.append(limit)
                        result = {"changes": edits}
                    else:
                        self.assertEqual(tool, "verify_policy_schedules")
                        result = {"concerns": []}
                    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                        {"type": "tool_use", "name": tool, "input": {"cases": [result]}}]})

                with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                        policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
                    response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
                expected_calls = ["draft_policy_schedules", "repair_policy_schedule", "repair_policy_schedule"]
                if not resolves:
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn("Set operations must include value", response.json()["detail"])
                    self.assertEqual(calls, expected_calls)
                    self.assertEqual(self.client.get("/api/policy-schedules").json()["schedules"], [])
                    continue
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(calls, expected_calls + ["verify_policy_schedules"])
                [saved] = response.json()["schedules"]
                self.assertEqual(saved["cover"][2]["wording"]["page"], 3)
                self.assertEqual(saved["cover"][2]["wording"]["start_line"], 1)
                self.assertEqual(saved["cover"][3]["limit_basis"], "statutory")
                self.assertIsNone(saved["cover"][3]["limit_lkr"])
                self.assertEqual(saved["vehicle"], self.draft["vehicle"])

    def test_repairs_zero_value_charges_and_continues_after_two_corrections(self):
        remaining = [self.draft["charges"][0], {"label": "Processing fee", "category": "fee", "amount_lkr": "50"}]
        self.draft["charges"] = [remaining[0], {"label": "Unused tax", "category": "tax", "amount_lkr": "0.00"},
                                 remaining[1], {"label": "Unused levy", "category": "levy", "amount_lkr": "0"}]
        self.draft.update(issued_on="2025-11-02", no_claim_discount_lkr="65000")
        calls = []
        repairs = iter([
            (["charges", 1], [{"path": ["charges", 1], "operation": "remove"}, {"path": ["charges", 3], "operation": "remove"}]),
            (["issued_on"], [{"path": ["issued_on"], "operation": "set", "value_json": '"2025-10-25"'}]),
            (["no_claim_discount_lkr"], [{"path": ["no_claim_discount_lkr"], "operation": "set", "value_json": '"5000"'}]),
        ])

        def model(request):
            body = json.loads(request.content)
            tool = body["tool_choice"]["name"]
            calls.append(tool)
            if tool == "draft_policy_schedules":
                result = self.draft
            elif tool == "repair_policy_schedule":
                path, edits = next(repairs)
                prompt = body["messages"][0]["content"]
                self.assertIn(f'"path": {json.dumps(path)}', prompt)
                if path != ["charges", 1]:
                    self.assertNotIn("Unused levy", prompt)
                result = {"changes": edits}
            else:
                result = {"concerns": []}
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "name": tool, "input": {"cases": [result]}}]})

        with httpx.Client(transport=httpx.MockTransport(model)) as client, patch.object(
                policy_schedule_generator, "generation_client", return_value=nullcontext((client, "fixture"))):
            response = self.client.post("/api/policy-schedules", json={"policy_document_id": self.document})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(calls, ["draft_policy_schedules", *(["repair_policy_schedule"] * 3), "verify_policy_schedules"])
        [saved] = response.json()["schedules"]
        self.assertEqual(saved["charges"], remaining)
        self.assertEqual(saved["total_premium_lkr"], "55150.00")
        for field in ("insured_name", "vehicle", "starts_on", "ends_on", "base_premium_lkr"):
            self.assertEqual(saved[field], self.draft[field])

    def test_long_policy_terms_and_seventh_cover_condition_are_saved_without_truncation(self):
        medical = "Only expenses following an accident involving the insured vehicle are eligible. " * 25
        medical += "The benefit excludes paid drivers and pays up to Rs. 2,000 per accident."
        extras = ("Towing is limited to Rs. 1,500 per accident.", "Ambulance transport is limited to Rs. 3,000 per accident.", medical)
        clauses = (*WORDING, *extras)
        document = create_document(self.store, title="Motor wording with extended conditions", product="MOTOR", version="1",
            filename="motor.pdf", media_type="application/pdf", content=base64.b64encode(wording_pdf()).decode())
        with patch("backend.evidence.policy_wording_index.extract_pdf_pages", return_value=(list(enumerate(clauses, 1)), [])), patch(
                "backend.evidence.policy_wording_index.embed_texts", side_effect=fake_embeddings):
            index_document(self.store, document)

        for page, (name, amount, clause) in enumerate(zip(("Towing", "Ambulance transport", "Medical expenses"),
                ("1500", "3000", "2000"), extras, strict=True), 8):
            self.draft["cover"].append({"loss_type": "additional_benefit", "benefit_name": name,
                "limit_basis": "wording_amount", "limit_lkr": amount, "compulsory_excess_lkr": "0",
                "compulsory_excess_basis": None, "voluntary_excess_lkr": "0", "conditions": f"{clause} (W{page:03d})",
                "wording": {"reference": f"W{page:03d}", "start_line": 1}})
        other_terms = {
            "territorial_limits": ("Cover applies within the territorial limits stated in the wording. " * 10).strip(),
            "permitted_use": ("The insured vehicle may be used only for the purpose stated in the schedule. " * 10).strip(),
            "authorised_drivers": ("Drivers must hold a valid licence for the class of vehicle being driven. " * 10).strip(),
        }
        self.draft.update(other_terms)
        self.assertGreater(len(self.draft["cover"][6]["conditions"]), 1800)
        self.assertTrue(all(len(value) > 600 for value in other_terms.values()))

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate(document=document)

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["cover"][6]["conditions"], f"{medical} (W010)")
        self.assertEqual(schedule["cover"][6]["wording"]["quote"], medical)
        for field, value in other_terms.items():
            self.assertEqual(schedule[field], value)

    def test_long_endorsement_and_special_condition_are_verified_and_saved_in_full(self):
        endorsement = "M03 applies an excess to Sections I and II. " + (
            "The excess applies to each covered event and to the losses arising from that event. " * 23
        ) + "The endorsement cannot be combined with M80 (W006)."
        condition = "The insured must notify the insurer and provide the incident details. " * 10 + "The insurer must approve repairs."
        self.assertGreater(len(endorsement), 1800)
        self.assertGreater(len(condition), 600)
        self.draft.update(endorsements=[endorsement], special_conditions=[condition])

        with patch.object(policy_schedule_generator, "request_batch", wraps=policy_schedule_generator.request_batch) as requests:
            response = self.generate()

        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual([call.args[3] for call in requests.call_args_list], ["draft_policy_schedules", "verify_policy_schedules"])
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["endorsements"], [endorsement])
        self.assertEqual(schedule["special_conditions"], [condition])
        self.assertEqual(self.client.get(f"/api/policy-schedules/{schedule['id']}").json(), schedule)

    def test_claims_instructions_can_exceed_eighteen_hundred_characters(self):
        instructions = "Notify the insurer promptly and provide the incident details. " * 31
        self.assertGreater(len(instructions), 1800)
        self.draft["claims_instructions"] = instructions

        response = self.generate()

        self.assertEqual(response.status_code, 201, response.text)
        [schedule] = response.json()["schedules"]
        self.assertEqual(schedule["claims_instructions"], instructions.strip())
