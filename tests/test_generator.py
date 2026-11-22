"""Production generator and submitted-claim history through normal entry points."""
import json
import base64
import sqlite3
import ssl
import tempfile
import unittest
from contextlib import closing, nullcontext
from copy import deepcopy
from decimal import Decimal
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.generation import synthetic_claim_generator as generator
from backend.generation import synthetic_policy_generator as policy_generator
from backend.generation import claude_generation_client as generation_provider
from backend.api import create_app
from backend.storage.sqlite_store import Assessment, Claim, DecisionRecord, History, Person, Policy, PolicyBinding, PolicyDocument, SyntheticPolicyHolder, AgentRun, ToolCall
from backend.claim_schemas import THIRD_PARTY_LOSSES
from backend.workflow.investigation_execution import offline_readiness
from backend.evidence.required_claim_evidence import DRUNK_DRIVING_EXCLUSION
from backend.generation.claim_scenario_rules import (
    COMPLETE_SCENARIOS, CONTRADICTORY_NARRATIVE, INCOMPLETE_NARRATIVE, ESTIMATE_NARRATIVE_MISMATCH,
    MISSING_SUPPORTING_DOCUMENT, OMISSIONS, REPAIR_SUBLIMIT_EXCEEDED, available_scenarios, classify_scenario,
)
from backend.evidence.policy_wording_index import create_document, index_pending, search_document, uploaded_document_ids
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.agents import claude_investigation as agentic
from scripts.debug_workflow import ScriptedModel
from test_agentic import FakeEncoder


class GenerationTransportTests(unittest.TestCase):
    def test_connect_failure_is_retried_before_a_successful_response(self):
        attempts = []

        def respond(request):
            attempts.append(request)
            if len(attempts) == 1:
                raise httpx.ConnectTimeout("fixture connection timeout", request=request)
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "name": policy_generator.GENERATION_TOOL,
                "input": {"cases": [policy_draft("Test Owner", "SYN-TEST", ["repair"])]},
            }]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            cases = generation_provider.request_batch(client, "fixture", "fixture", policy_generator.GENERATION_TOOL,
                                                      policy_generator.PolicyCases, 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(cases[0].person.full_name, "Test Owner")

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
                        generation_provider.request_batch(client, "fixture", "fixture", policy_generator.GENERATION_TOOL,
                                                          policy_generator.PolicyCases, 1)
                self.assertEqual(len(attempts), 1)
                self.assertIn(type(error).__name__, captured.output[0])


def policy_draft(name, registration, covered_losses, *, insurance_type="comprehensive"):
    losses = list(covered_losses)
    if insurance_type == "comprehensive" and not THIRD_PARTY_LOSSES.intersection(losses):
        losses.append("third_party_property_damage")
    return {
        "person": {"full_name": name, "district": "Galle", "locality": f"Fixture {registration}"},
        "wording": "1. Scope. Apply the listed cover to incidents within the scheduled dates. "
                   "2. Benefits. The listed benefits and their evidence requirements apply to each loss. "
                   "3. The schedule determines the amount limits, deductible and required police evidence. "
                   f"4. Work and valuations use assessor agreement {registration}.",
        "policy": {"insurance_type": insurance_type, "vehicle_type": "hatchback", "permitted_use": "private",
                   "use_conditions": ["Personal trips and commuting are covered. Paid delivery is excluded."],
                   "version": "2025.7", "registration": registration, "starts_on": "2025-01-01", "ends_before": "2026-01-01",
                   "covered_losses": losses, "limit": "360000.00", "deductible": "12500.00", "clause": "MODEL-MOTOR-9",
                   "liability_terms": {"description": "Covers liability for the listed losses suffered by another person.",
                       "conditions": ["The insured driver's responsibility must be established."],
                       "exclusions": ["Losses suffered by the insured are outside this liability benefit."]},
                   "vehicle": {"make": "Fixture Motors", "model": f"Series {registration}", "year": 2018},
                   "repair_terms": {"description": f"Covers bumper work following collision under assessor agreement {registration}.",
                       "flood_damage": "excluded",
                       "repair_benefits": [
                           {"cause": "collision", "description": "Collision with another vehicle",
                            "covered_repairs": ["Bumper repair"], "conditions": [], "exclusions": []},
                           {"cause": "storm", "description": "Direct hail damage while parked",
                            "covered_repairs": ["Remove hail dents from the roof"], "conditions": [],
                            "exclusions": ["Wear to roof seals"]},
                       ], "exclusions": ["Maintenance"],
                       "labour": {"basis": "no_separate_limit", "limit": None},
                       "painting": {"scope": "affected_panels_only", "limit_per_panel": None, "max_panels": None},
                       "additional_charges": {"covered": False, "eligible_charges": [], "exclusions": [
                           {"category": category, "name": category.replace("_", " ").title(), "reason": "Excluded from repair cover."}
                           for category in ("towing", "storage", "diagnostics", "temporary_transport")
                       ]}} if "repair" in covered_losses else None,
                   "theft_terms": {"description": f"Pays pre-loss market value under assessor agreement {registration}.",
                       "basis": "vehicle_valuation_only", "covered_fittings": []} if "theft" in covered_losses else None},
    }


def draft(record, loss_type="repair", incident_key="incident-1"):
    claim_fixture = {
        "repair": ("repair_estimate", "A collision damaged the rear bumper.", "Bumper repair"),
        "theft": ("vehicle_valuation", "The vehicle was stolen overnight.", "Vehicle value"),
        "third_party_property_damage": ("third_party_demand", "I reversed into the neighbour's fence and broke its support posts.", "Replace the neighbour's fence posts"),
        "third_party_injury": ("third_party_demand", "I struck a pedestrian with my car as I pulled out of the driveway.", "Pedestrian's documented treatment"),
    }
    document_type, narrative, description = claim_fixture.get(loss_type, claim_fixture["repair"])
    registration = record["policy"]["registration"]
    return {
        "scenario": "Recoverable loss", "conditions": "Active policy and consistent cost evidence",
        "expected_route": "adjuster review",
        "claim": {"person_id": record["person"]["person_id"], "policy_id": record["policy"]["policy_id"],
                  "incident_key": incident_key, "loss_type": loss_type,
                  "vehicle": record["policy"]["vehicle"],
                  "loss_date": "2025-10-15", "reported_date": "2025-10-16", "gross_amount": "120000.00",
                  "additional_costs": [],
                  "narrative": narrative,
                  "documents": [{"type": document_type, "name": "Itemised cost evidence.json",
                                 "registration": registration, "date": "2025-10-16", "total": "120000.00",
                                 **({"garage_name": "Test Motor Works", "garage_address": "12 Sample Road, Galle"}
                                    if document_type == "repair_estimate" else {}),
                                 "line_items": [{"description": description, "amount": "120000.00"}]}]},
    }


def planned_draft(record, final, comparisons):
    """Script both provider responses from the intended submission fixture."""
    final = deepcopy(final)
    claim = final["claim"]
    policy = record["policy"]
    if claim.get("reused_cost_document_id"):
        scenario_type = "reused_cost_document"
    else:
        scenario_type = classify_scenario({**claim, "registration": policy["registration"]}, policy)
    aligned = draft(record, claim["loss_type"] if claim["loss_type"] in policy["covered_losses"]
                    else policy["covered_losses"][0], claim["incident_key"])["claim"]
    if scenario_type in OMISSIONS:
        base_document = aligned["documents"][0]
        aligned = deepcopy(claim)
        if scenario_type == "missing_loss_date":
            aligned["loss_date"] = "2025-10-15"
        elif scenario_type == "missing_cost_document":
            aligned["documents"].append(base_document)
        elif scenario_type == "missing_chassis":
            aligned["vehicle"] = policy["vehicle"]
        elif scenario_type == "missing_itemisation":
            aligned["documents"][0]["line_items"] = base_document["line_items"]
        elif scenario_type == "missing_required_police_report":
            aligned["documents"].append({
                "type": "police_report", "name": "Police report.json", "registration": policy["registration"],
                "date": aligned["reported_date"], "station": "Fictional Lake Road Police Station",
                "summary": "The station recorded the reported loss and the insured vehicle details.",
            })
        final["claim"] = aligned
    if scenario_type in COMPLETE_SCENARIOS:
        aligned = claim
    items = aligned["documents"][0]["line_items"]
    ids = [row["id"] for row in comparisons
           if row["loss_type"] == aligned["loss_type"] and row["incident"] == aligned["narrative"] and row["cost_items"] == items]
    return {
        "scenario": final["scenario"], "incident": aligned["narrative"], "loss_type": aligned["loss_type"],
        "loss_date": aligned["loss_date"],
        "cost_items": items, "additional_costs": aligned["additional_costs"],
        "policy_basis": f"{policy['clause']} covers the incident and the listed work.",
        "required_evidence": ["Insured vehicle details, claim form and itemised cost document"],
        "comparison_ids": ids[:12], "scenario_type": scenario_type,
    }, final


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="claims-generator-")
        self.addCleanup(self.temp.cleanup)
        self.app = create_app(Path(self.temp.name))
        self.client = self.enterContext(TestClient(self.app))
        self.store = self.app.state.store
        self.app.state.worker.close()
        self.workflow = self.app.state.worker.workflow
        self.enterContext(patch("backend.evidence.policy_wording_index.encoder", return_value=FakeEncoder()))
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})

    def generate(self, path, tool, drafts, status, provider_error=None, *, policy_id=None, change_plan=None,
                 change_draft=None, retries=0, use_catalogue_mix=False):
        client_class = httpx.Client
        requests = []
        remaining_drafts = iter(drafts)
        planning_batch = []
        planning_drafts = []
        claim_generation = tool == generator.GENERATION_TOOL
        if claim_generation:
            selected_policy_id = policy_id or drafts[0]["claim"]["policy_id"]
            record = next(entry for entry in policy_generator.list_people_policies(self.store)
                          if entry["policy"]["policy_id"] == selected_policy_id)
        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if provider_error:
                return httpx.Response(400, json={"error": {"message": provider_error}})
            called_tool = payload["tool_choice"]["name"]
            retrying = len(requests) > 1 and requests[-2]["tool_choice"]["name"] == called_tool and claim_generation
            size = payload["tools"][0]["input_schema"]["properties"]["cases"]["maxItems"]
            if called_tool == generator.PLANNING_TOOL:
                prompt = payload["messages"][0]["content"]
                comparisons = json.loads(prompt.split("Staff-approved synthetic comparisons: ", 1)[1].splitlines()[0])
                if not retrying:
                    planning_batch[:] = [planned_draft(record, case, comparisons) for case in islice(remaining_drafts, size)]
                plans = [deepcopy(plan) for plan, _ in planning_batch]
                if change_plan:
                    change_plan(plans[0])
                planning_drafts[:] = [written for _, written in planning_batch]
                cases = plans
            elif claim_generation:
                if not retrying:
                    self.assertEqual(requests[-2]["tool_choice"]["name"], generator.PLANNING_TOOL)
                cases = deepcopy(planning_drafts)
                self.assertIn("Approved plans:", payload["messages"][0]["content"])
                self.assertIn(cases[0]["scenario"], payload["messages"][0]["content"])
                self.assertNotIn(record["policy"]["clause"], payload["messages"][0]["content"])
                if change_draft:
                    change_draft(cases[0])
            else:
                cases = list(islice(remaining_drafts, size))
            return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{"type": "tool_use",
                "name": called_tool, "input": {"cases": cases}}]})
        def make_client(**kwargs):
            return client_class(transport=httpx.MockTransport(respond), **kwargs)
        body = {"count": len(drafts)}
        if claim_generation:
            body["policy_id"] = selected_policy_id
        type_assignment = nullcontext()
        profile_assignment = nullcontext()
        district_assignment = nullcontext()
        if not claim_generation and not use_catalogue_mix:
            type_assignment = patch.object(policy_generator, "requested_insurance_types",
                                           return_value=[case["policy"]["insurance_type"] for case in drafts])
            profile_assignment = patch.object(policy_generator, "requested_vehicle_profiles",
                return_value=[(case["policy"]["vehicle_type"], case["policy"]["permitted_use"], "storm", "Fixture profile.") for case in drafts])
            district_assignment = patch.object(policy_generator, "requested_districts",
                return_value=[case["person"]["district"] for case in drafts])
        with type_assignment, profile_assignment, district_assignment, patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fixture", "CLAIMS_LLM_MODE": "deterministic"}), \
             patch.object(generation_provider.httpx, "Client", side_effect=make_client):
            result = self.client.post(path, json=body)
        self.assertEqual(result.status_code, status, result.text)
        if status in (200, 201):
            batch_size = generation_provider.BATCH_SIZE if claim_generation else policy_generator.POLICY_BATCH_SIZE
            self.assertEqual(len(requests), ((len(drafts) + batch_size - 1) // batch_size) * (2 if claim_generation else 1) + retries)
            if not claim_generation:
                index_pending(self.store)
        else:
            self.assertIn(len(requests) - retries, (1, 2) if claim_generation and not provider_error else (1,))
        return result.json(), requests[0]

    def save_catalogue(self, *, theft_report_required=False):
        path = "/api/generator/people-policies"
        theft_policy = policy_draft("Bimal Silva", "SYN-TEST-2", ["theft"])
        if theft_report_required:
            theft_policy["policy"]["police_report_required_for"] = ["theft"]
            theft_policy["policy"]["deductible"] = "260000.00"
        self.generate(path, policy_generator.GENERATION_TOOL, [
            policy_draft("Amanda Perera", "SYN-TEST-1", ["repair"]),
            theft_policy,
        ], 201)
        return self.client.get(path).json()["records"]

    def preview(self, *drafts):
        result, request = self.generate("/api/generator/preview", generator.GENERATION_TOOL, drafts, 200)
        prompt = request["messages"][0]["content"]
        for item in drafts:
            self.assertIn(item["claim"]["policy_id"], prompt)
            self.assertIn(item["claim"]["person_id"], prompt)
            if item["claim"].get("vehicle"):
                self.assertIn(item["claim"]["vehicle"]["make"], prompt)
        self.assertNotIn("Amanda Perera", prompt)
        return result

    def commit(self, preview, *, execution_mode="deterministic"):
        with patch.dict("os.environ", {"CLAIMS_LLM_MODE": execution_mode}), patch.object(generator, "request_cases") as provider:
            result = self.client.post(f"/api/generator/preview/{preview['batch_id']}/commit")
        provider.assert_not_called()
        self.assertEqual(result.status_code, 202, result.text)
        return result.json()["assessment_ids"]

    def test_generated_repair_policies_store_and_index_both_flood_choices(self):
        path = "/api/generator/people-policies"
        covered = policy_draft("Flood included", "SYN-FLOOD-IN", ["repair"])
        covered["policy"]["repair_terms"]["flood_damage"] = "covered"
        covered["policy"]["repair_terms"]["repair_benefits"].append({
            "cause": "flood", "description": "Direct water damage from a flooded road",
            "covered_repairs": ["Dry submerged cabin trim"], "conditions": ["Damage follows sudden immersion"],
            "exclusions": ["Gradual water ingress"],
        })
        excluded = policy_draft("Flood excluded", "SYN-FLOOD-OUT", ["repair"])
        theft = policy_draft("Theft only", "SYN-THEFT", ["theft"])
        missing = deepcopy(excluded)
        del missing["policy"]["repair_terms"]["flood_damage"]
        with self.assertRaisesRegex(ValueError, "repair_terms.flood_damage"):
            policy_generator.PolicyCase.model_validate(missing)
        collision_only = deepcopy(excluded)
        collision_only["policy"]["repair_terms"]["repair_benefits"].pop()
        with self.assertRaisesRegex(ValueError, "beyond collision"):
            policy_generator.PolicyCase.model_validate(collision_only)
        flood_inconsistent = deepcopy(covered)
        flood_inconsistent["policy"]["repair_terms"]["flood_damage"] = "excluded"
        with self.assertRaisesRegex(ValueError, "exactly when flood damage is covered"):
            policy_generator.PolicyCase.model_validate(flood_inconsistent)

        self.generate(path, policy_generator.GENERATION_TOOL, [covered, excluded, theft], 201)
        records = {record["person"]["full_name"]: record for record in self.client.get(path).json()["records"]}
        for name, choice, clause in (
            ("Flood included", "covered", "caused by flooding is covered"),
            ("Flood excluded", "excluded", "caused by flooding or floodwater is excluded"),
        ):
            with self.subTest(choice=choice):
                record = records[name]
                self.assertEqual(record["policy"]["repair_terms"]["flood_damage"], choice)
                wording_id = record["wording"]["id"]
                wording = self.client.get(f"/api/policy-documents/{wording_id}/original").text
                self.assertIn("Flood damage under repair cover", wording)
                self.assertIn(clause, wording)
                self.assertIn("garage name and full garage address", wording)
                self.assertIn("is not required on the repair estimate", wording)
                self.assertIn("Repair benefit for storm", wording)
                if choice == "covered":
                    self.assertIn("Dry submerged cabin trim", wording)
                self.assertIn(DRUNK_DRIVING_EXCLUSION, wording)
                self.assertIn(DRUNK_DRIVING_EXCLUSION, record["policy"]["repair_terms"]["exclusions"])
                retrieved = search_document(self.store, wording_id, "flood damage under repair cover")
                self.assertTrue(any(clause in passage["text"] for passage in retrieved["passages"]))

        record = records["Theft only"]
        self.assertIsNone(record["policy"]["repair_terms"])
        wording = self.client.get(f"/api/policy-documents/{record['wording']['id']}/original").text
        self.assertNotIn("Flood damage under repair cover", wording)

    def test_policy_generator_retries_repeated_vehicle_terms_and_wording(self):
        path = "/api/generator/people-policies"
        first = policy_draft("First owner", "SYN-FIRST", ["repair"])
        self.generate(path, policy_generator.GENERATION_TOOL, [first], 201)
        repeated = policy_generator.PolicyCase.model_validate(first)
        second = policy_draft("Second owner", "SYN-SECOND", ["third_party_injury"], insurance_type="third_party")
        second["person"].update(district="Colombo", locality="Borella")
        second["policy"].update(vehicle_type="van", permitted_use="goods_delivery",
                                use_conditions=["Deliveries of the insured business's goods are covered."])
        fresh = policy_generator.PolicyCase.model_validate(second)
        with patch.object(policy_generator, "generation_client", return_value=nullcontext((None, "fixture"))), \
             patch.object(policy_generator, "request_batch", side_effect=[[repeated], [fresh]]) as provider:
            result = self.client.post(path, json={"count": 1})
        self.assertEqual(result.status_code, 201, result.text)
        self.assertIn(repeated.policy.vehicle.model, provider.call_args_list[0].args[2])
        self.assertIn("repeated vehicle make/model, policy terms, policy wording", provider.call_args_list[1].args[2])
        self.assertEqual(len(self.client.get(path).json()["records"]), 2)

    def test_policy_generator_retries_invalid_effective_period(self):
        path = "/api/generator/people-policies"
        revised = policy_draft("Retry Owner", "SYN-PERIOD-RETRY", ["third_party_injury"], insurance_type="third_party")
        revised["person"].update(district="Colombo", locality="Borella")
        invalid = deepcopy(revised)
        invalid["policy"]["starts_on"] = invalid["policy"]["ends_before"]
        with self.assertRaises(ValueError) as issue:
            policy_generator.PolicyCase.model_validate(invalid)
        error = generation_provider.GenerationValidationError(issue.exception.errors()[0])
        self.assertEqual(error.issue_type, "invalid_policy_period")
        invalid["policy"].update(revised["policy"], deductible=revised["policy"]["limit"])
        with self.assertRaises(ValueError) as issue:
            policy_generator.PolicyCase.model_validate(invalid)
        self.assertEqual(issue.exception.errors()[0]["type"], "invalid_policy_amounts")
        fresh = policy_generator.PolicyCase.model_validate(revised)
        with patch.object(policy_generator, "generation_client", return_value=nullcontext((None, "fixture"))), \
             patch.object(policy_generator, "requested_insurance_types", return_value=["third_party"]), \
             patch.object(policy_generator, "request_batch", side_effect=[error, [fresh]]) as provider:
            result = self.client.post(path, json={"count": 1})
        self.assertEqual(result.status_code, 201, result.text)
        self.assertIn("end date after the start date", provider.call_args_list[1].args[2])
        self.assertEqual(len(self.client.get(path).json()["records"]), 1)

    def test_policy_generator_retries_duplicate_names_and_places(self):
        path = "/api/generator/people-policies"
        first = policy_draft("First Owner", "SYN-FIRST", ["repair"])
        self.generate(path, policy_generator.GENERATION_TOOL, [first], 201)
        second = policy_draft("FIRST-OWNER", "SYN-SECOND", ["third_party_injury"], insurance_type="third_party")
        second["policy"].update(vehicle_type="van", permitted_use="goods_delivery")
        second["person"]["locality"] = first["person"]["locality"]
        repeated = policy_generator.PolicyCase.model_validate(second)
        second["person"].update(full_name="Second Owner", locality="Hikkaduwa")
        fresh = policy_generator.PolicyCase.model_validate(second)
        with patch.object(policy_generator, "generation_client", return_value=nullcontext((None, "fixture"))), \
             patch.object(policy_generator, "requested_districts", return_value=["Galle"]), \
             patch.object(policy_generator, "request_batch", side_effect=[[repeated], [fresh]]) as provider:
            result = self.client.post(path, json={"count": 1})
        self.assertEqual(result.status_code, 201, result.text)
        self.assertIn("repeated policyholder name, policyholder location", provider.call_args_list[1].args[2])
        records = self.client.get(path).json()["records"]
        self.assertEqual({item["person"]["full_name"] for item in records}, {"First Owner", "Second Owner"})
        self.assertIn("Hikkaduwa", {item["person"]["locality"] for item in records})

    def test_regenerating_discarded_preview_retries_repeated_incident(self):
        repair, _ = self.save_catalogue()
        first = self.preview(draft(repair))
        path = "/api/generator/preview"
        self.assertEqual(self.client.delete(f"{path}/{first['batch_id']}").status_code, 204)
        new_story = "I hit a low loading-bay post while backing out of a timber yard near Akkaraitivu. The bumper split by its right bracket."
        prompts = []

        def change_plan(plan):
            prompts.append(plan["incident"])
            if len(prompts) > 1:
                plan["incident"] = new_story

        result, request = self.generate(path, generator.GENERATION_TOOL, [draft(repair)], 200,
            change_plan=change_plan, change_draft=lambda case: case["claim"].update(narrative=new_story), retries=1)
        self.assertIn(first["fixtures"][0]["plan"]["incident"], request["messages"][0]["content"])
        self.assertEqual(result["fixtures"][0]["claim"]["narrative"], new_story)

    def test_policy_generator_and_claim_workflow_use_the_assigned_insurance_types(self):
        path = "/api/generator/people-policies"
        comprehensive = policy_draft("Comprehensive owner", "SYN-COMP", ["repair", "theft"])
        comprehensive["person"].update(district="Colombo", locality="Borella")
        repair = comprehensive["policy"]["repair_terms"]
        repair["flood_damage"] = "covered"
        repair["repair_benefits"].append({"cause": "flood", "description": "Surface water enters the cabin",
                                          "covered_repairs": ["Clean water-damaged cabin carpet"],
                                          "conditions": [], "exclusions": []})
        third_party = policy_draft("Third-party owner", "SYN-TP", ["third_party_property_damage"], insurance_type="third_party")
        third_party["person"].update(district="Gampaha", locality="Negombo")
        third_party["policy"].update(vehicle_type="van", permitted_use="goods_delivery",
                                     use_conditions=["Deliveries of the insured business's goods are covered."])
        third_party["policy"]["police_report_required_for"] = ["third_party_property_damage"]
        invalid = deepcopy(third_party)
        invalid["policy"]["covered_losses"].append("repair")
        with self.assertRaisesRegex(ValueError, "only the listed third-party liability"):
            policy_generator.PolicyCase.model_validate(invalid)
        _, request = self.generate(path, policy_generator.GENERATION_TOOL, [comprehensive, third_party], 201, use_catalogue_mix=True)
        self.assertIn("Use these insurance_type values in order: comprehensive, third_party.", request["messages"][0]["content"])
        self.assertIn("vehicle_type=hatchback, permitted_use=private", request["messages"][0]["content"])
        self.assertIn("include a flood repair benefit", request["messages"][0]["content"])
        self.assertIn("vehicle_type=van, permitted_use=goods_delivery", request["messages"][0]["content"])
        own, liability = self.client.get(path).json()["records"]
        self.assertEqual((own["person"]["district"], own["person"]["locality"]), ("Colombo", "Borella"))
        self.assertEqual((liability["person"]["district"], liability["person"]["locality"]), ("Gampaha", "Negombo"))
        self.assertEqual(set(own["policy"]["covered_losses"]), {"repair", "theft", "third_party_property_damage"})
        self.assertEqual(liability["policy"]["covered_losses"], ["third_party_property_damage"])
        self.assertIsNone(liability["policy"]["repair_terms"])
        self.assertIsNone(liability["policy"]["theft_terms"])
        retrieved = search_document(self.store, liability["wording"]["id"], "scope and third-party liability conditions")
        self.assertTrue(any("Damage to or theft of the insured vehicle is excluded" in item["text"] for item in retrieved["passages"]))

        covered = draft(liability, "third_party_property_damage", "fence-demand")
        covered["claim"]["documents"].append({"type": "police_report", "name": "Police report.json",
            "registration": liability["policy"]["registration"], "date": "2025-10-16", "station": "Fictional Lake Road Station",
            "summary": "The officer reviewed footage showing the insured driver reversing into the neighbour's fence."})
        excluded = draft(liability, "repair", "own-vehicle-excluded")
        excluded["claim"]["documents"] = []
        preview = self.preview(covered, excluded)
        self.assertEqual([fixture["scenario_type"] for fixture in preview["fixtures"]], ["required_police_report_attached", "uncovered_loss"])
        covered_id, excluded_id = self.commit(preview, execution_mode="anthropic")
        with self.store.session() as session:
            context = session.get(Assessment, covered_id).context
        model = ScriptedModel()
        with patch.object(agentic, "call_model", side_effect=model):
            unresolved = agentic.run_specialist(self.store, covered_id, context, 6, "coverage", "Assess responsibility")
        self.assertEqual(unresolved["details"]["coverage_status"], "UNRESOLVED")

        def provider(*args):
            response = model(*args)
            block = response["content"][0]
            if args[2].endswith(":coverage") and block["name"] == "submit_analysis":
                analysis = block["input"]
                if analysis["classified_loss"] == "third_party_property_damage":
                    analysis["liability_confirmed"] = True
                    analysis["coverage_items"][0].update(item="Neighbour's fence damage", document_refs=["submission:/documents/1"],
                        policy_term="The insured driver's responsibility must be established.", reason="The police report identifies the driver in the footage.")
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(covered_id)
            self.workflow.run(excluded_id)
        covered_audit = export_audit(self.store, covered_id)
        excluded_audit = export_audit(self.store, excluded_id)
        self.assertEqual(covered_audit["context"]["findings"]["coverage"]["details"]["coverage_status"], "COVERED")
        self.assertEqual(excluded_audit["context"]["findings"]["coverage"]["details"]["coverage_status"], "NOT_COVERED")
        self.assertEqual(excluded_audit["decision"]["information_required"], [])

    def test_policy_lists_excluded_charges_even_when_storage_is_covered(self):
        storage_conditions = (
            "Storage is eligible for up to two calendar days at the approved repairer while awaiting inspection "
            "after a covered collision. The receipt must itemise the storage dates and identify the insured vehicle."
        )
        towing_exclusion = (
            "Recovery and transport costs are excluded from this repair benefit, including towing to an approved "
            "repairer. Listing the tow separately from the repair estimate does not change that exclusion."
        )
        self.assertGreater(len(storage_conditions), 160)
        self.assertGreater(len(towing_exclusion), 160)
        drafts = [policy_draft(f"Test owner {i}", f"SYN-EXTRAS-{i}", ["repair"]) for i in (1, 2)]
        for candidate in drafts:
            terms = candidate["policy"]["repair_terms"]["additional_charges"]
            terms["covered"] = True
            terms["eligible_charges"] = [{
                "category": "storage", "name": "Vehicle storage", "conditions": storage_conditions,
                "limit_per_incident": "8000.00", "receipt_required": True,
            }]
            terms["exclusions"] = [item for item in terms["exclusions"] if item["category"] != "storage"]
            next(item for item in terms["exclusions"] if item["category"] == "towing")["reason"] = towing_exclusion
        drafts[0]["policy"]["repair_terms"]["description"] += " Includes short term storage."
        self.generate("/api/generator/people-policies", policy_generator.GENERATION_TOOL, drafts, 201)
        [first, second] = self.client.get("/api/generator/people-policies").json()["records"]
        terms = second["policy"]["repair_terms"]["additional_charges"]
        self.assertEqual([item["category"] for item in terms["eligible_charges"]], ["storage"])
        self.assertEqual(terms["eligible_charges"][0]["conditions"], storage_conditions)
        self.assertEqual(next(item for item in terms["exclusions"] if item["category"] == "towing")["reason"], towing_exclusion)

        missing = draft(first)
        storage = {"category": "storage", "amount": "5000.00", "reason": "Storage while awaiting inspection."}
        missing["claim"].update(gross_amount="125000.00", additional_costs=[storage])
        [missing_id] = self.commit(self.preview(missing))
        self.workflow.run(missing_id)
        missing_detail = self.client.get(f"/api/claims/{missing_id}").json()
        self.assertEqual(missing_detail["route"], "request more information")
        self.assertEqual([(gap["document"], gap["location"]) for gap in missing_detail["decision"]["information_required"]],
                         [("Vehicle storage receipt", None)])

        mixed = draft(second)
        mixed["claim"].update(gross_amount="129000.00", additional_costs=[
            storage, {"category": "towing", "amount": "4000.00", "reason": "Tow the vehicle to the repairer."}])
        mixed["claim"]["documents"].append({
            "type": "additional_cost_receipt", "category": "storage", "name": "Storage receipt.json",
            "registration": second["policy"]["registration"], "date": "2025-10-16", "total": "5000.00",
            "description": "The workshop kept the car on site for one day until the assessor arrived.",
        })
        [mixed_id] = self.commit(self.preview(mixed))
        self.workflow.run(mixed_id)
        detail = self.client.get(f"/api/claims/{mixed_id}").json()
        self.assertEqual(detail["route"], "adjuster review")
        self.assertEqual(detail["decision"]["information_required"], [])
        coverage = detail["context"]["findings"]["coverage"]["details"]
        self.assertEqual(coverage["financial"]["cost_documents_total"], "120000.00")
        self.assertIn("Towing is listed in the repair policy exclusions.", " ".join(coverage["review_reasons"]))
        covered = deepcopy(mixed)
        covered["claim"].update(incident_key="covered-storage", gross_amount="125000.00", additional_costs=[storage])
        [fixture] = self.preview(covered)["fixtures"]
        self.assertEqual(fixture["scenario_type"], "covered_additional_cost")
        self.assertIsNone(fixture["plan"]["omission"])

    def test_policy_rejects_an_unstated_or_conflicting_additional_charge(self):
        base = policy_draft("Test owner", "SYN-TERMS", ["repair"])
        terms = base["policy"]["repair_terms"]["additional_charges"]
        towing = terms["exclusions"].pop(0)
        with self.assertRaisesRegex(ValueError, "Explicitly cover or exclude.*towing"):
            policy_generator.PolicyCase.model_validate(base)
        terms["exclusions"].extend([towing, towing])
        with self.assertRaisesRegex(ValueError, "appear only once"):
            policy_generator.PolicyCase.model_validate(base)

    def test_provider_schema_rejects_other_loss_types_and_generated_history(self):
        repair, _ = self.save_catalogue()
        for invalid in ({**draft(repair), "historical_costs": []}, draft(repair, "glass"),
                        {**draft(repair), "policy": repair["policy"]}):
            with self.subTest(invalid=invalid):
                def respond(request):
                    body = json.loads(request.content)
                    self.assertEqual(body["tool_choice"]["name"], generator.GENERATION_TOOL)
                    return httpx.Response(200, json={"stop_reason": "tool_use", "content": [{"type": "tool_use",
                        "name": generator.GENERATION_TOOL, "input": {"cases": [invalid]}}]})
                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaises(ValueError):
                        generator.request_cases(client, "configured-model", 1, "batch", 0, [repair], plans=[])

    def test_provider_credit_error_keeps_preview_empty(self):
        repair, _ = self.save_catalogue()
        result, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [draft(repair)], 503,
                                  provider_error="Your credit balance is too low")
        self.assertIn("credits are too low", result["detail"])
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])

    def test_generator_identifies_count_validation_and_truncated_responses(self):
        repair, _ = self.save_catalogue()
        valid = draft(repair)
        invalid = draft(repair, "glass")
        for stop, records, message in (
            ("tool_use", [valid, valid], "2 records when 1 were requested"),
            ("tool_use", [invalid], "cases.0.claim.loss_type"),
            ("max_tokens", [], "response limit"),
        ):
            with self.subTest(reason=message):
                def respond(request):
                    body = json.loads(request.content)
                    cases = body["tools"][0]["input_schema"]["properties"]["cases"]
                    self.assertEqual((cases["minItems"], cases["maxItems"]), (1, 1))
                    return httpx.Response(200, json={"stop_reason": stop, "content": [{"type": "tool_use",
                        "name": generator.GENERATION_TOOL, "input": {"cases": records}}]})
                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaisesRegex(generation_provider.GenerationError, message):
                        generator.request_cases(client, "configured-model", 1, "batch", 0, [repair], plans=[])

    def test_claim_preview_requires_saved_policies_before_provider_request(self):
        with patch.object(generator, "request_cases") as provider:
            result = self.client.post("/api/generator/preview", json={"policy_id": "POL-UNKNOWN", "count": 2})
        self.assertEqual(result.status_code, 409)
        provider.assert_not_called()

    def test_model_cannot_mix_policyholders(self):
        repair, theft = self.save_catalogue()
        wrong_person = draft(repair)
        wrong_person["claim"]["person_id"] = theft["person"]["person_id"]
        self.generate("/api/generator/preview", generator.GENERATION_TOOL, [wrong_person], 503)
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])

    def test_preview_preserves_policy_basis_longer_than_600_characters(self):
        repair, _ = self.save_catalogue()
        policy_basis = (
            "The collision clause covers damage caused by contact with another vehicle, so the rear bumper damage "
            "falls within the listed bumper repair benefit. "
            "The claim concerns accident damage and does not include maintenance. The repair estimate itemises "
            "the work required to restore the damaged bumper. The painting terms allow work on the affected panel "
            "and the labour terms impose no separate limit. The amount claimed is above the deductible and within "
            "the overall policy limit. The submission includes the insured vehicle details and the itemised estimate. "
            "No separately billed services or unrelated repair work are included in the amount claimed."
        )
        self.assertGreater(len(policy_basis), 600)
        result, _ = self.generate(
            "/api/generator/preview", generator.GENERATION_TOOL, [draft(repair)], 200,
            change_plan=lambda plan: plan.update(policy_basis=policy_basis),
        )
        self.assertEqual(result["fixtures"][0]["plan"]["policy_basis"], policy_basis)

    def test_preview_preserves_incident_narrative_longer_than_1500_characters(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        narrative = (
            "A collision with another vehicle damaged the rear bumper. "
            + "The claimant documented the bumper damage and provided photographs to the repairer. " * 20
            + "The repairer provided an itemised estimate for the bumper work."
        )
        self.assertGreater(len(narrative), 1500)
        case["claim"]["narrative"] = narrative

        [fixture] = self.preview(case)["fixtures"]

        self.assertEqual(fixture["plan"]["incident"], narrative)
        self.assertEqual(fixture["claim"]["narrative"], narrative)

    def test_preview_and_commit_preserve_scenario_titles_longer_than_140_characters(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        title = (
            "Rear bumper damage from a low-speed collision while the insured car was parked outside a shop, "
            "with a workshop estimate itemising the repair costs"
        )
        self.assertGreater(len(title), 140)
        case["scenario"] = title

        preview = self.preview(case)
        [fixture] = preview["fixtures"]
        self.assertEqual(fixture["plan"]["scenario"], title)
        self.assertEqual(fixture["scenario"], title)
        self.commit(preview)

        with closing(sqlite3.connect(self.app.state.labels_path)) as labels:
            saved = labels.execute("SELECT scenario FROM labels WHERE batch_id = ?", (preview["batch_id"],)).fetchone()
        self.assertEqual(saved, (title,))

    def test_preview_and_commit_preserve_conditions_longer_than_1000_characters(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        conditions = (
            "The policy covers rear bumper damage from a collision. "
            + "The itemised estimate identifies the replacement parts and the corresponding fitting labour. " * 12
            + "The claimant has supplied all mandatory evidence."
        )
        self.assertGreater(len(conditions), 1000)
        case["conditions"] = conditions

        preview = self.preview(case)
        self.assertEqual(preview["fixtures"][0]["conditions"], conditions)
        self.commit(preview)

        with closing(sqlite3.connect(self.app.state.labels_path)) as labels:
            saved = labels.execute("SELECT conditions FROM labels WHERE batch_id = ?", (preview["batch_id"],)).fetchone()
        self.assertEqual(saved, (conditions,))

    def test_preview_ignores_unavailable_comparisons(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        result, _ = self.generate(
            "/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
            change_plan=lambda plan: plan.update(comparison_ids=["invented"]),
        )
        [fixture] = result["fixtures"]
        self.assertEqual(fixture["plan"]["comparison_ids"], [])
        self.assertEqual(fixture["plan"]["expected_route"], "adjuster review")
        self.assertIn("lacks five", fixture["plan"]["expectation"])

    def test_preview_counts_only_distinct_comparisons_for_the_planned_loss(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        claim = case["claim"]
        repair_record = {"id": "repair-reference", "loss_type": "repair", "incident": claim["narrative"],
                         "cost_items": claim["documents"][0]["line_items"], "cost_document_total": "120000.00"}
        theft_record = {**repair_record, "id": "theft-reference", "loss_type": "theft"}
        comparisons = {row["id"]: row for row in (repair_record, theft_record)}
        with patch.object(generator, "planning_comparisons", return_value=comparisons):
            result, _ = self.generate(
                "/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
                change_plan=lambda plan: plan.update(comparison_ids=["theft-reference", "repair-reference"] * 8),
            )
        [fixture] = result["fixtures"]
        self.assertEqual(fixture["plan"]["comparison_ids"], ["repair-reference"])
        self.assertEqual(fixture["plan"]["expected_route"], "adjuster review")

    def test_generator_rejects_draft_drift(self):
        repair, _ = self.save_catalogue()
        arguments = ("/api/generator/preview", generator.GENERATION_TOOL, [draft(repair)], 503)
        result, _ = self.generate(*arguments, change_draft=lambda case: case["claim"].update(
            narrative="The bumper was damaged by a different incident than the one planned."))
        self.assertIn("specified in the plan", result["detail"])
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])

    def test_generator_commits_complete_incomplete_and_contradictory_narratives(self):
        repair, _ = self.save_catalogue()
        reference = "My car was stationary when it was hit."
        damage = "The rear bumper broke after a delivery van backed into it."
        conflict = "I was driving the car at the moment of impact."
        incident = f"{reference} {damage}"
        for scenario, narrative in (("ordinary", incident), (INCOMPLETE_NARRATIVE, damage),
                                    (CONTRADICTORY_NARRATIVE, f"{incident} {conflict}")):
            with self.subTest(scenario=scenario):
                case = draft(repair, incident_key=scenario)
                case["claim"]["narrative"] = incident
                if scenario == CONTRADICTORY_NARRATIVE:
                    case["claim"]["documents"].append({"type": "police_report", "name": "Collision report.json",
                        "registration": repair["policy"]["registration"], "station": "Fictional Lake Road Station",
                        "date": "2025-10-16", "summary": "The officer recorded camera footage showing the parked car had not moved."})

                def vary_plan(plan):
                    plan["scenario_type"] = scenario
                    if scenario == INCOMPLETE_NARRATIVE:
                        plan["narrative_variation"] = {"reference_sentence": reference}
                    elif scenario == CONTRADICTORY_NARRATIVE:
                        plan["narrative_variation"] = {"reference_sentence": reference, "conflicting_sentence": conflict}
                        plan["include_optional_police_report"] = True

                preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 200, change_plan=vary_plan)
                [fixture] = preview["fixtures"]
                self.assertEqual(fixture["scenario_type"], scenario)
                self.assertEqual(fixture["claim"]["narrative"], narrative)
                self.assertEqual(fixture["claim"]["documents"][0]["total"], "120000.00")
                self.assertNotIn("narrative_variation", fixture["claim"])
                if scenario == CONTRADICTORY_NARRATIVE:
                    self.assertIn("had not moved", fixture["claim"]["documents"][1]["summary"])
                [identifier] = self.commit(preview)
                data = self.client.get(f"/api/claims/{identifier}").json()
                self.assertEqual(data["submission"]["narrative"], narrative)

    def test_preview_calculates_estimate_and_gross_from_approved_costs(self):
        policy = policy_draft("Cost test", "SYN-EXACT-COST", ["repair"])
        charges = policy["policy"]["repair_terms"]["additional_charges"]
        charges["exclusions"] = [charge for charge in charges["exclusions"] if charge["category"] != "towing"]
        charges.update(covered=True, eligible_charges=[{"category": "towing", "name": "Towing",
            "conditions": "A covered collision requires recovery to the repairer.", "limit_per_incident": None, "receipt_required": False}])
        path = "/api/generator/people-policies"
        self.generate(path, policy_generator.GENERATION_TOOL, [policy], 201)
        [record] = self.client.get(path).json()["records"]
        case = draft(record)
        case["claim"].update(gross_amount="125000.17", additional_costs=[{"category": "towing", "amount": "5000.05",
            "reason": "The loose bumper prevented driving to the garage."}])
        case["claim"]["documents"][0].update(total="120000.12", line_items=[
            {"description": "Bumper panel", "amount": "100000.06"}, {"description": "Install panel", "amount": "20000.06"}])

        def incorrect_draft_totals(response):
            response["claim"]["gross_amount"] = "124000.00"
            response["claim"]["documents"][0]["total"] = "119000.00"

        def omit_draft_totals(response):
            del response["claim"]["gross_amount"]
            del response["claim"]["documents"][0]["total"]

        for modify in (incorrect_draft_totals, omit_draft_totals):
            with self.subTest(draft=modify.__name__):
                preview, request = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
                    change_plan=lambda plan: plan.update(gross_amount="123000.00"), change_draft=modify)
                self.assertNotIn("gross_amount", request["tools"][0]["input_schema"]["$defs"]["ClaimPlan"]["properties"])
                [fixture] = preview["fixtures"]
                self.assertEqual(fixture["scenario_type"], "covered_additional_cost")
                self.assertEqual(fixture["plan"]["gross_amount"], "125000.17")
                self.assertEqual(fixture["claim"]["gross_amount"], "125000.17")
                [estimate] = fixture["claim"]["documents"]
                self.assertEqual(estimate["type"], "repair_estimate")
                self.assertEqual(estimate["garage_name"], "Test Motor Works")
                self.assertEqual(estimate["garage_address"], "12 Sample Road, Galle")
                self.assertNotIn("chassis_number", estimate)
                self.assertEqual(estimate["total"], "120000.12")

    def test_generator_retries_itemised_costs_outside_the_reference_range(self):
        repair, _ = self.save_catalogue()
        attempts = []

        def outside_range_on_first_attempt(plan):
            if not attempts:
                plan["cost_items"][0]["amount"] = "300000.00"
            attempts.append(plan)

        preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [draft(repair)], 200,
            change_plan=outside_range_on_first_attempt, retries=1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(preview["fixtures"][0]["plan"]["gross_amount"], "120000.00")

    def test_repair_estimate_sublimit_scenario_checks_the_line_against_the_selected_policy(self):
        catalogue = "/api/generator/people-policies"
        caps = (("labour", "15000.00", "17000.00", "Rear bumper fitting labour"),
                ("painting", "8000.00", "10000.00", "Paint rear bumper"))
        policies = []
        for category, limit, _, _ in caps:
            candidate = policy_draft(f"{category} Test Owner", f"SYN-CAP-{category.upper()}", ["repair"])
            repair = candidate["policy"]["repair_terms"]
            self.assertNotIn(REPAIR_SUBLIMIT_EXCEEDED, available_scenarios(candidate["policy"], generator.local_date()))
            if category == "labour":
                repair["labour"].update(basis="per_claim", limit=limit)
            else:
                repair["painting"]["limit_per_panel"] = limit
            policies.append(candidate)
        self.generate(catalogue, policy_generator.GENERATION_TOOL, policies, 201)
        saved = {record["person"]["full_name"]: record for record in self.client.get(catalogue).json()["records"]}
        for category, limit, amount, description in caps:
            with self.subTest(category=category):
                record = saved[f"{category} Test Owner"]
                case = draft(record)
                lines = [{"description": "Rear bumper replacement part", "amount": "36000.00"},
                         {"description": description, "amount": amount}]
                case["claim"]["gross_amount"] = total = f"{Decimal('36000.00') + Decimal(amount):.2f}"
                case["claim"]["documents"][0].update(total=total, line_items=lines)

                def choose_sublimit(plan):
                    plan.update(scenario_type=REPAIR_SUBLIMIT_EXCEEDED,
                                repair_sublimit_test={"category": category, "line_number": 2})

                preview, request = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case],
                                                 200, change_plan=choose_sublimit)
                self.assertIn(f'"limit": "{limit}"', request["messages"][0]["content"])
                [fixture] = preview["fixtures"]
                self.assertEqual(fixture["scenario_type"], REPAIR_SUBLIMIT_EXCEEDED)
                self.assertEqual(fixture["plan"]["expected_route"], "adjuster review")
                self.assertIn("LKR 2,000.00 above", fixture["plan"]["expectation"])
                self.assertEqual(fixture["claim"]["gross_amount"], total)
                self.assertEqual(fixture["claim"]["documents"][0]["line_items"], lines)

                def use_equal_amount(plan):
                    choose_sublimit(plan)
                    plan["cost_items"][1]["amount"] = limit

                rejected, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case],
                                            503, change_plan=use_equal_amount, retries=1)
                self.assertIn("priced above its applicable cap", rejected["detail"])
                [identifier] = self.commit(preview)
                submission = self.client.get(f"/api/claims/{identifier}").json()["submission"]
                self.assertNotIn("repair_sublimit_test", submission)
                self.assertEqual(submission["documents"][0]["total"], total)

    def test_preview_and_commit_generate_a_balanced_estimate_with_conflicting_work(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        excerpt = "I checked the front door and it had no damage."
        case["claim"]["narrative"] = f"A car hit my rear bumper while I was parked. {excerpt}"
        description = "Replace and paint the front door"

        def plan_conflict(plan):
            plan.update(scenario_type=ESTIMATE_NARRATIVE_MISMATCH, estimate_discrepancy={
                "narrative_excerpt": excerpt, "unaffected_part": "front door",
                "line_number": 1, "estimate_description": description,
            })

        def plan_without_conflict(plan):
            plan_conflict(plan)
            plan["incident"] = "A car hit the rear bumper and left a dent in the front door."
            plan["estimate_discrepancy"]["narrative_excerpt"] = plan["incident"]

        result, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 503,
                                  change_plan=plan_without_conflict, retries=1)
        self.assertIn("explicitly described as undamaged", result["detail"])

        attempts = []

        def repair_plan(plan):
            plan_conflict(plan)
            if not attempts:
                plan["estimate_discrepancy"]["line_number"] = 2
            else:
                plan["estimate_discrepancy"]["narrative_excerpt"] = excerpt.upper().replace("FRONT DOOR", "FRONT-DOOR").rstrip(".")
            attempts.append(plan)

        def draft_conflict(response):
            response["claim"]["documents"][0]["line_items"][0]["description"] = description

        batch, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
                                 change_plan=repair_plan, change_draft=draft_conflict, retries=1)
        [shown] = batch["fixtures"]
        self.assertEqual(shown["scenario_type"], ESTIMATE_NARRATIVE_MISMATCH)
        self.assertEqual(shown["plan"]["expected_route"], "adjuster review")
        self.assertIn("front door", shown["plan"]["expectation"])
        self.assertEqual(shown["claim"]["narrative"], case["claim"]["narrative"])
        [estimate] = shown["claim"]["documents"]
        self.assertEqual(estimate["line_items"], [{"description": description, "amount": "120000.00"}])
        self.assertEqual(estimate["total"], shown["claim"]["gross_amount"])
        self.assertEqual(shown["plan"]["cost_items"][0]["description"], "Bumper repair")
        [identifier] = self.commit(batch)
        detail = self.client.get(f"/api/claims/{identifier}").json()
        self.assertEqual(detail["submission"], shown["claim"])
        self.assertNotIn("estimate_discrepancy", detail["submission"])
        self.assertEqual(self.client.post(f"/api/generator/preview/{batch['batch_id']}/commit").status_code, 404)
        duplicate, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 503, change_plan=plan_conflict)
        self.assertIn("scenario type", duplicate["detail"])

    def test_generator_rewrites_copied_policy_phrases_in_plan_and_attachment(self):
        policy = policy_draft("Test owner", "SYN-REWORD", ["repair"])
        repair_phrase = "Replacement of damaged rear bumper mounting brackets"
        policy["policy"]["repair_terms"]["repair_benefits"][0]["covered_repairs"].append(repair_phrase)
        self.generate("/api/generator/people-policies", policy_generator.GENERATION_TOOL, [policy], 201)
        [record] = self.client.get("/api/generator/people-policies").json()["records"]
        case = draft(record)
        report = {
            "type": "police_report", "name": "Police report.json", "registration": record["policy"]["registration"],
            "date": "2025-10-16", "station": "Fictional Lake Road Police Station",
            "summary": "The driver and witness were interviewed about the parking manoeuvre and damaged bumper.",
        }
        case["claim"]["documents"].append(report)

        plan_attempts, draft_attempts = [], []

        def copy_schedule_once(plan):
            plan_attempts.append(plan)
            if len(plan_attempts) == 1:
                plan["cost_items"][0]["description"] = repair_phrase

        def copy_wording_once(candidate):
            draft_attempts.append(candidate)
            if len(draft_attempts) == 1:
                candidate["claim"]["documents"][1]["summary"] = "APPLY the listed cover, to incidents within the scheduled dates."

        preview, _ = self.generate(
            "/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
            change_plan=copy_schedule_once, change_draft=copy_wording_once, retries=2,
        )
        [fixture] = preview["fixtures"]
        self.assertEqual(fixture["claim"]["documents"][0]["line_items"][0]["description"], "Bumper repair")
        self.assertEqual(fixture["claim"]["documents"][1]["summary"], report["summary"])
        self.assertIn(record["policy"]["clause"], fixture["plan"]["policy_basis"])
        self.commit(preview)

    def test_preview_rejects_claim_text_that_repeats_a_policy_clause_after_retry(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        case["claim"]["narrative"] += f" I claim under {repair['policy']['clause']}."
        result, _ = self.generate(
            "/api/generator/preview", generator.GENERATION_TOOL, [case], 503, retries=1,
        )
        self.assertIn("repeated policy wording after a rewrite", result["detail"])
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])

    def test_police_report_scenarios_use_policy_and_submitted_documents(self):
        repair, theft = self.save_catalogue(theft_report_required=True)
        for record, loss_type, required in ((repair, "repair", False), (theft, "theft", True)):
            with self.subTest(loss_type=loss_type):
                attached = draft(record, loss_type, "report-attached")
                attached["claim"]["documents"].append({
                    "type": "police_report", "name": "Police report.json",
                    "registration": record["policy"]["registration"], "date": "2025-10-16",
                    "station": "Fictional Lake Road Police Station", "summary": "The station recorded the reported loss and vehicle details.",
                })
                omitted = draft(record, loss_type, "report-omitted")
                if required:
                    omitted["expected_route"] = "request more information"
                    for case in (attached, omitted):
                        case["claim"]["gross_amount"] = "310000.00"
                        valuation = case["claim"]["documents"][0]
                        valuation["total"] = valuation["line_items"][0]["amount"] = "310000.00"
                preview = self.preview(attached, omitted)
                submitted_report = preview["fixtures"][0]["claim"]["documents"][1]
                self.assertTrue(submitted_report["id"].startswith("PR-"))
                self.assertNotIn("total", submitted_report)
                self.assertEqual([item["scenario_type"] for item in preview["fixtures"]],
                                 ["required_police_report_attached", "missing_required_police_report"] if required
                                 else ["optional_police_report_attached", "ordinary"])

                attached_id, omitted_id = self.commit(preview)
                for identifier in (attached_id, omitted_id):
                    self.workflow.run(identifier)
                attached_detail = self.client.get(f"/api/claims/{attached_id}").json()
                omitted_detail = self.client.get(f"/api/claims/{omitted_id}").json()
                self.assertEqual(attached_detail["submission"]["documents"][1], submitted_report)
                self.assertIn("coverage", attached_detail["context"]["findings"])
                self.assertEqual(attached_detail["decision"]["information_required"], [])
                decision = omitted_detail["decision"]
                if required:
                    self.assertEqual(decision["route"], "request more information")
                    self.assertEqual([item["document"] for item in decision["information_required"]], ["Police report"])
                    self.assertIn("corroborate the reported incident", decision["justification"])
                    self.assertEqual(omitted_detail["task"]["queue"], "adjuster")
                    self.assertNotIn("coverage", omitted_detail["context"]["findings"])
                else:
                    self.assertEqual(decision["information_required"], [])
                    self.assertIn("coverage", omitted_detail["context"]["findings"])

    def test_generated_policy_document_requirements_drive_claim_plan_and_intake(self):
        policy = policy_draft("Test owner", "SYN-SUPPORT", ["repair"])
        policy["policy"]["supporting_documents"] = [
            {"loss_type": "repair", "document_type": "incident_report", "requirement": "required",
             "purpose": "Confirm the location and circumstances of the collision."},
            {"loss_type": "repair", "document_type": "witness_statement", "requirement": "optional",
             "purpose": "Corroborate an observer's account of the impact."},
        ]
        self.generate("/api/generator/people-policies", policy_generator.GENERATION_TOOL, [policy], 201)
        [record] = self.client.get("/api/generator/people-policies").json()["records"]
        wording = self.client.get(f"/api/policy-documents/{record['wording']['id']}/original").text
        self.assertIn("Incident report (required)", wording)
        self.assertIn("Witness statement (optional)", wording)

        incident = {"type": "incident_report", "name": "Incident report.json",
                    "registration": record["policy"]["registration"], "date": "2025-10-16",
                    "author": "Fictional lot manager", "summary": "The driver of a reversing van hit the bumper of a parked car."}
        witness = {**incident, "type": "witness_statement", "name": "Witness statement.json", "author": "Fictional observer",
                   "summary": "I saw a van back into the bumper while waiting at the shop."}
        complete = draft(record)
        complete["claim"]["documents"].extend([incident, witness])
        preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [complete], 200,
                                   change_plan=lambda plan: plan.update(optional_supporting_documents=["witness_statement"]))
        [included] = preview["fixtures"]
        chosen = {doc["document_type"]: doc for doc in included["plan"]["documents"]}
        self.assertTrue(chosen["incident_report"]["included"])
        self.assertTrue(chosen["witness_statement"]["included"])
        [included_id] = self.commit(preview)
        self.workflow.run(included_id)
        self.assertEqual(self.client.get(f"/api/claims/{included_id}").json()["decision"]["information_required"], [])

        incomplete = draft(record, incident_key="without-required-incident-report")
        incomplete["claim"]["documents"].append(incident)
        def omit_incident_report(response):
            claim = response["claim"]
            claim["documents"] = [doc for doc in claim["documents"] if doc["type"] != "incident_report"]

        preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [incomplete], 200,
                                   change_plan=lambda plan: plan.update(scenario_type=MISSING_SUPPORTING_DOCUMENT,
                                                                        omitted_supporting_document="incident_report"),
                                   change_draft=omit_incident_report)
        [omitted] = preview["fixtures"]
        chosen = {doc["document_type"]: doc for doc in omitted["plan"]["documents"]}
        self.assertFalse(chosen["incident_report"]["included"])
        self.assertFalse(chosen["witness_statement"]["included"])
        self.assertEqual([doc["type"] for doc in omitted["claim"]["documents"]], ["repair_estimate"])
        [omitted_id] = self.commit(preview)
        self.workflow.run(omitted_id)
        detail = self.client.get(f"/api/claims/{omitted_id}").json()
        self.assertEqual(detail["decision"]["route"], "request more information")
        self.assertEqual([gap["document"] for gap in detail["decision"]["information_required"]], ["Incident report"])

    def test_generated_submissions_reach_all_four_routes_from_recorded_evidence(self):
        self.generate("/api/generator/people-policies", policy_generator.GENERATION_TOOL, [
            policy_draft(f"Synthetic Policyholder {index}", f"SYN-ROUTING-{index}", ["repair"])
            for index in range(6)
        ], 201)
        records = self.client.get("/api/generator/people-policies").json()["records"]
        prior_records, target = records[:5], records[5]

        earlier_previews = [self.preview(draft(record)) for record in prior_records]
        earlier_ids = [self.commit(preview)[0] for preview in earlier_previews]
        for identifier in earlier_ids:
            self.workflow.run(identifier)
            result = self.client.post(f"/api/portal/claims/{identifier}/review", json={
                "outcome": "approved", "reason": "Verified the synthetic bumper repair and itemised cost."})
            self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.client.get(f"/api/claims/{earlier_ids[0]}").json()["route"], "adjuster review")
        self.assertEqual(earlier_previews[0]["fixtures"][0]["plan"]["expected_route"], "adjuster review")
        self.assertIn("lacks five", earlier_previews[0]["fixtures"][0]["plan"]["expectation"])

        missing = draft(target, incident_key="missing")
        missing["claim"]["documents"] = []
        borrowed = draft(target, incident_key="borrowed")
        prior_document = earlier_previews[0]["fixtures"][0]["claim"]["documents"][0]
        borrowed["claim"].update(documents=[], reused_cost_document_id=prior_document["id"])
        preview, _ = self.generate(
            "/api/generator/preview", generator.GENERATION_TOOL, [draft(target), missing, borrowed], 200,
            change_plan=lambda plan: plan["comparison_ids"].extend(["unavailable", plan["comparison_ids"][0]]),
        )
        complete_plan = preview["fixtures"][0]["plan"]
        self.assertEqual(complete_plan["expected_route"], "auto-approve")
        self.assertEqual(len(complete_plan["comparison_ids"]), 5)
        incomplete_plan = preview["fixtures"][1]["plan"]
        self.assertEqual(incomplete_plan["expected_route"], "request more information")
        self.assertEqual(incomplete_plan["omission"], OMISSIONS["missing_cost_document"])
        self.assertEqual(incomplete_plan["cost_items"], complete_plan["cost_items"])
        self.assertEqual(preview["fixtures"][1]["claim"]["documents"], [])
        self.assertEqual(preview["fixtures"][2]["plan"]["expected_route"], "fraud escalation")
        approved_id, missing_id, fraud_id = self.commit(preview, execution_mode="anthropic")
        with patch.object(agentic, "call_model", side_effect=ScriptedModel()):
            for identifier in (approved_id, missing_id, fraud_id):
                self.workflow.run(identifier)

        approved, incomplete, flagged = [self.client.get(f"/api/claims/{identifier}").json()
                                         for identifier in (approved_id, missing_id, fraud_id)]
        self.assertEqual(approved["route"], "auto-approve")
        coverage = approved["context"]["findings"]["coverage"]["details"]
        self.assertTrue(coverage["policy_rag"]["cited"])
        self.assertTrue(coverage["coverage_items"][0]["passage_refs"])
        package = export_audit(self.store, approved_id)
        self.assertEqual(package["policy_wording"]["id"], coverage["policy_rag"]["document_id"])
        self.assertTrue(verify_audit(package))
        self.assertEqual(approved["context"]["findings"]["historical"]["details"]["amount_justification"]["sample_size"], 5)
        self.assertEqual(incomplete["route"], "request more information")
        self.assertEqual([request["document"] for request in incomplete["decision"]["information_required"]], ["Itemised repair estimate"])
        self.assertEqual(flagged["route"], "fraud escalation")
        self.assertEqual(flagged["task"]["queue"], "fraud")
        [flag] = flagged["context"]["findings"]["intake"]["details"]["fraud_indicators"]
        self.assertEqual(flag["evidence"]["prior_document"], prior_document)
        self.assertNotEqual(flag["evidence"]["prior_registration"], flag["evidence"]["current_registration"])

    def test_mixed_preview_submits_coverage_conflicts_for_triage(self):
        repair, _ = self.save_catalogue()
        covered = draft(repair)
        uncovered = draft(repair, "theft", "uncovered-theft")
        uncovered.update(scenario="Theft on repair-only cover", conditions="The selected policy excludes theft.")
        expired = draft(repair, incident_key="expired-repair")
        expired.update(scenario="Collision after policy expiry", conditions="The loss was on the exclusive end date.")
        expired["claim"].update(loss_date="2026-01-01", reported_date="2026-01-02")
        expired["claim"]["documents"][0]["date"] = "2026-01-02"
        preview, request = self.generate("/api/generator/preview", generator.GENERATION_TOOL,
                                         [covered, uncovered, expired], 200)
        self.assertIn("Unused scenario types", request["messages"][0]["content"])
        identifiers = self.commit(preview)
        for identifier in identifiers:
            self.workflow.run(identifier)
        with self.store.session() as session:
            outcomes = [session.get(Assessment, identifier).context["findings"]["coverage"]["details"]["coverage_status"]
                        for identifier in identifiers]
            self.assertEqual(outcomes, ["COVERED", "NOT_COVERED", "INACTIVE"])

    def test_planner_can_describe_the_actual_policy_exception(self):
        repair, _ = self.save_catalogue()
        uncovered = draft(repair, "theft", "direct-uncovered")
        uncovered["claim"]["gross_amount"] = "505000.00"
        uncovered["claim"]["documents"][0].update(total="505000.00", line_items=[
            {"description": "Vehicle market value", "amount": "505000.00"}])
        expired = draft(repair, incident_key="direct-expired")
        expired["claim"].update(loss_date="2026-01-01", reported_date="2026-01-02")
        expired["claim"]["documents"][0]["date"] = "2026-01-02"
        above_limit = draft(repair, incident_key="direct-over-limit")
        above_limit["claim"]["gross_amount"] = "370000.00"
        above_limit["claim"]["documents"][0].update(total="370000.00", line_items=[
            {"description": "Repair the rear body and bumper", "amount": "370000.00"}])

        for case, scenario in ((uncovered, "uncovered_loss"), (expired, "after_policy_end"), (above_limit, "above_policy_limit")):
            with self.subTest(scenario=scenario):
                claim = case["claim"]

                def plan_actual_claim(plan):
                    plan.update(loss_date=claim["loss_date"], loss_type=claim["loss_type"], incident=claim["narrative"],
                                cost_items=claim["documents"][0]["line_items"])

                preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
                                           change_plan=plan_actual_claim)
                [fixture] = preview["fixtures"]
                self.assertEqual(fixture["scenario_type"], scenario)
                self.assertEqual(fixture["plan"]["gross_amount"], fixture["claim"]["gross_amount"])
                self.assertEqual(fixture["plan"]["loss_date"], fixture["claim"]["loss_date"])
                self.assertEqual(fixture["plan"]["loss_type"], fixture["claim"]["loss_type"])
                self.commit(preview)

    def test_missing_estimate_draft_can_arrive_with_estimate_already_omitted(self):
        repair, _ = self.save_catalogue()
        case = draft(repair)
        case["claim"]["documents"] = []
        preview, _ = self.generate("/api/generator/preview", generator.GENERATION_TOOL, [case], 200,
                                   change_draft=lambda response: response["claim"].update(documents=[]))
        [fixture] = preview["fixtures"]
        self.assertEqual(fixture["scenario_type"], "missing_cost_document")
        self.assertEqual(fixture["plan"]["expected_route"], "request more information")
        self.assertEqual(fixture["claim"]["documents"], [])
        [identifier] = self.commit(preview)
        self.workflow.run(identifier)
        detail = self.client.get(f"/api/claims/{identifier}").json()
        self.assertEqual(detail["route"], "request more information")

    def test_scenario_types_and_capacity_are_enforced_across_batches(self):
        repair, theft = self.save_catalogue()
        path = "/api/generator/preview"
        policy_id = repair["policy"]["policy_id"]
        cases = [draft(repair, incident_key=f"case-{index}") for index in range(10)]
        cases[1] = draft(repair, "theft", "excluded")
        cases[2]["claim"].update(loss_date="2026-01-01", reported_date="2026-01-02")
        cases[3]["claim"]["loss_date"] = None
        cases[4]["claim"]["documents"] = []
        cases[5]["claim"]["documents"][0]["line_items"] = []
        cases[6]["claim"]["documents"][0]["registration"] = "SYN-DIFFERENT"
        cases[7]["claim"]["documents"][0]["line_items"][0]["amount"] = "110000.00"
        cases[8]["claim"]["gross_amount"] = "125000.00"
        cases[9]["claim"]["gross_amount"] = "115000.00"

        renamed = {**cases[0], "scenario": "Different title for the same facts"}
        result, _ = self.generate(path, generator.GENERATION_TOOL, [cases[0], renamed], 503)
        self.assertIn("repeated a scenario type", result["detail"])
        result, _ = self.generate(path, generator.GENERATION_TOOL, [draft(theft, "theft")], 503, policy_id=policy_id)
        self.assertIn("mismatched person and policy", result["detail"])
        first = self.preview(*cases[:5])
        stale = self.preview(cases[0])
        full_preview = self.preview(*cases)
        self.assertEqual(len({item["scenario_type"] for item in full_preview["fixtures"]}), 10)
        self.commit(first)
        with self.store.session.begin() as session:
            ordinary_claim = session.scalar(select(Claim).where(Claim.batch_id == first["batch_id"]).order_by(Claim.sequence))
            old_assessment = session.scalar(select(Assessment).where(Assessment.claim_id == ordinary_claim.id))
            old_assessment.execution = {key: value for key, value in old_assessment.execution.items() if key != "generation_scenario"}
        self.assertEqual(self.client.post(f"/api/generator/preview/{stale['batch_id']}/commit").status_code, 409)
        self.assertEqual(self.client.post(f"/api/generator/preview/{full_preview['batch_id']}/commit").status_code, 409)
        self.assertIn("repeated a scenario type", self.generate(path, generator.GENERATION_TOOL, [renamed], 503)[0]["detail"])
        second = self.preview(*cases[5:])
        self.commit(second)
        self.assertEqual(len({item["scenario_type"] for item in first["fixtures"] + second["fixtures"]}), 10)
        remaining = {row["policy"]["policy_id"]: row["remaining_claims"]
                     for row in self.client.get("/api/generator/claim-policies").json()["records"]}
        self.assertEqual(remaining, {policy_id: 0, theft["policy"]["policy_id"]: 10})
        with patch.object(generator, "request_cases") as provider:
            self.assertEqual(self.client.post(path, json={"policy_id": policy_id, "count": 1}).status_code, 409)
            self.assertEqual(self.client.post(path, json={"policy_id": policy_id, "count": 11}).status_code, 422)
        provider.assert_not_called()

    def test_preview_commits_exact_linked_claims_without_creating_policies_or_history(self):
        repair, theft = self.save_catalogue()
        extra = draft(repair, incident_key="towing")
        extra["claim"].update(gross_amount="125000.00", additional_costs=[{"reason": "Towing", "amount": "5000.00"}])
        preview = self.preview(draft(repair), extra)
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])
        identifiers = self.commit(preview)
        self.assertEqual(self.client.post(f"/api/generator/preview/{preview['batch_id']}/commit").status_code, 404)
        with self.store.session() as session:
            self.assertEqual(session.scalars(select(History)).all(), [])
            self.assertEqual(len(session.scalars(select(Policy)).all()), 2)
            self.assertEqual(len(session.scalars(select(Person)).all()), 2)
            for identifier, shown in zip(identifiers, preview["fixtures"], strict=True):
                self.assertNotIn("historical_costs", shown)
                self.assertNotIn("policy", shown)
                self.assertEqual(shown["scenario"], "Recoverable loss")
                row = session.get(Assessment, identifier)
                claim = session.get(Claim, row.claim_id)
                self.assertEqual(claim.payload, shown["claim"])
                self.assertEqual({key for key in claim.payload if key.startswith("policy")}, {"policy_id"})
                self.assertEqual(row.execution["generation_plan"], shown["plan"])
                self.assertNotIn("generation_plan", row.context)
                self.assertEqual(claim.policy_id, repair["policy"]["policy_id"])
                self.assertEqual(session.get(Policy, claim.policy_id).payload, repair["policy"])
                self.assertEqual(session.get(SyntheticPolicyHolder, claim.policy_id).person_id, claim.payload["person_id"])
                self.assertEqual(claim.payload["person_id"], shown["person"]["person_id"])
                self.assertEqual(claim.payload["registration"], repair["policy"]["registration"])
                self.assertNotIn("scenario", claim.payload)
        with closing(sqlite3.connect(self.app.state.labels_path)) as labels:
            self.assertEqual(labels.execute("SELECT DISTINCT scenario,conditions FROM labels").fetchall(),
                             [("Recoverable loss", "Active policy and consistent cost evidence")])
        for identifier in identifiers:
            self.workflow.run(identifier)
            detail = self.client.get(f"/api/claims/{identifier}").json()
            self.assertNotIn("policy", detail["synthetic_evidence"])
            with self.store.session() as session:
                row = session.get(Assessment, identifier)
                self.assertFalse(row.context["historical_available"])
                intake = row.context["findings"]["intake"]["details"]["coverage_readiness"]
                self.assertTrue(intake["policy_available"])
                coverage = row.context["findings"]["coverage"]
                self.assertEqual(coverage["details"]["coverage_status"], "COVERED")
                self.assertIsNone(coverage["details"]["financial"]["payable"])
                self.assertNotIn("historical", row.context["findings"])
                self.assertIsNone(session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier, ToolCall.tool == "retrieve_comparable_claims")))
                self.assertEqual(session.get(DecisionRecord, identifier).payload["reason_codes"], ["SPECIALIST_REVIEW"])
                policy_lookup = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier, ToolCall.tool == "retrieve_policy"))
                self.assertTrue(policy_lookup.response["policyholder"]["matches"])
                self.assertEqual(policy_lookup.response["policy"], repair["policy"])
                intake_lookup = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                    ToolCall.tool == "read_coverage_requirements"))
                self.assertEqual(intake_lookup.stage, "r0:intake")
                self.assertEqual(intake_lookup.request["policy_id"], policy_lookup.request["policy_id"])
                self.assertTrue(intake_lookup.response["policy_available"])
                self.assertNotIn("policy", intake_lookup.response)

    def test_missing_saved_policy_blocks_coverage_at_intake(self):
        repair, _ = self.save_catalogue()
        [identifier] = self.commit(self.preview(draft(repair)))
        policy_id = repair["policy"]["policy_id"]
        with self.store.session.begin() as session:
            session.delete(session.get(SyntheticPolicyHolder, policy_id))
            session.delete(session.get(PolicyBinding, policy_id))
            session.flush()
            session.delete(session.get(Policy, policy_id))

        self.workflow.run(identifier)
        detail = self.client.get(f"/api/claims/{identifier}").json()
        self.assertEqual(detail["submission"]["policy_id"], policy_id)
        readiness = detail["context"]["findings"]["intake"]["details"]["coverage_readiness"]
        self.assertFalse(readiness["policy_available"])
        self.assertFalse(readiness["ready"])
        self.assertNotIn("coverage", detail["context"]["findings"])
        self.assertFalse(any(call["tool"] == "retrieve_policy" for call in detail["tool_calls"]))
        self.assertEqual(detail["decision"]["route"], "adjuster review")
        self.assertEqual(detail["decision"]["information_required"], [])

    def test_chassis_mismatch_and_missing_chassis_are_distinct(self):
        repair, _ = self.save_catalogue()
        mismatch = draft(repair, incident_key="different-chassis")
        mismatch["claim"]["vehicle"] = {**repair["policy"]["vehicle"], "chassis_number": "SYN-CH-MISMATCH"}
        missing = draft(repair, incident_key="unknown-chassis")
        missing["claim"]["vehicle"] = None
        mismatch_id, missing_id = self.commit(self.preview(mismatch, missing))
        for identifier in (mismatch_id, missing_id):
            self.workflow.run(identifier)
        with self.store.session() as session:
            mismatch_finding = session.get(Assessment, mismatch_id).context["findings"]["coverage"]
            self.assertEqual(mismatch_finding["details"]["vehicle_mismatches"], ["chassis number"])
            self.assertEqual(session.get(DecisionRecord, mismatch_id).payload["route"], "adjuster review")
            findings = session.get(Assessment, missing_id).context["findings"]
            self.assertNotIn("coverage", findings)
            missing_finding = findings["intake"]
            self.assertFalse(missing_finding["details"]["coverage_readiness"]["checks"]["vehicle_match"])
            self.assertEqual(missing_finding["details"]["missing"][-1]["field"], "vehicle.chassis_number")
            self.assertEqual(session.get(DecisionRecord, missing_id).payload["route"], "request more information")

    def test_valuation_only_policy_excludes_separately_claimed_fittings(self):
        _, theft = self.save_catalogue()
        case = draft(theft, "theft")
        case["claim"].update(gross_amount="130000.00", additional_costs=[
            {"reason": "Aftermarket alloy wheels", "amount": "10000.00"},
        ])
        [identifier] = self.commit(self.preview(case))
        self.workflow.run(identifier)
        with self.store.session() as session:
            coverage = session.get(Assessment, identifier).context["findings"]["coverage"]
            self.assertIn("limited to the vehicle valuation", coverage["summary"])
            self.assertEqual([(item["item"], item["outcome"]) for item in coverage["details"]["coverage_items"]],
                             [("Vehicle valuation", "covered"), ("Separately claimed additional charges", "not_covered")])
            self.assertIsNone(coverage["details"]["financial"]["payable"])
            self.assertEqual(session.get(DecisionRecord, identifier).payload["route"], "adjuster review")

    def test_coverage_receives_listed_fitting_and_claimed_charge(self):
        request = policy_draft("Kamal Silva", "SYN-FITTING", ["theft"])
        request["policy"]["theft_terms"] = {"basis": "vehicle_and_listed_fittings",
            "description": "Pays the pre-loss valuation and the separately itemised fixed dash camera.",
            "covered_fittings": ["Fixed dash camera"]}
        self.generate("/api/generator/people-policies", policy_generator.GENERATION_TOOL, [request], 201)
        [entry] = self.client.get("/api/generator/people-policies").json()["records"]
        case = draft(entry, "theft")
        case["claim"].update(narrative="The vehicle was stolen overnight with its fixed dash camera.",
            gross_amount="130000.00", additional_costs=[{"reason": "Fixed dash camera", "amount": "10000.00"}])
        [identifier] = self.commit(self.preview(case))

        def model(store, assessment, stage, instruction, messages, tools, choice):
            submitted = json.loads(messages[0]["content"])
            checks = submitted["required_checks"]
            policy = checks["supporting_evidence"][0]["result"]["policy"]
            self.assertEqual(policy["theft_terms"]["covered_fittings"], ["Fixed dash camera"])
            self.assertEqual(submitted["submission"]["additional_costs"][0]["reason"], "Fixed dash camera")
            return {"content": [{"type": "tool_use", "name": "submit_analysis", "input": {
                "summary": "The separate dash-camera charge is listed in the theft terms.", "review_reasons": [],
                "evidence": [checks["evidence_ref"]], "coverage_conclusion": "COVERED", "coverage_items": [
                    {"item": "Fixed dash camera", "outcome": "covered", "policy_term": "The theft terms list fixed dash cameras.",
                     "reason": "The claim identifies a fixed dash camera."},
                ]}}]}

        with patch.object(agentic, "call_model", side_effect=model):
            finding = agentic.run_specialist(self.store, identifier, {}, 1, "coverage", "Assess theft")
        self.assertEqual(finding["status"], "pass")
        self.assertEqual(finding["details"]["financial"]["payable"], "117500.00")

    def test_generated_claim_commit_does_not_attach_uploaded_wording(self):
        repair, theft = self.save_catalogue()
        unrelated = create_document(self.store, title="Unrelated HR wording", product="HR", version="1",
                        filename="remote-work.txt", media_type="text/plain",
                        content=base64.b64encode(b"Employees can work remotely.").decode())
        identifier, = self.commit(self.preview(draft(repair)), execution_mode="anthropic")
        with self.store.session() as session:
            assessment = session.get(Assessment, identifier)
            policy_id = repair["policy"]["policy_id"]
            binding = session.get(PolicyBinding, policy_id)
            wording = session.get(PolicyDocument, binding.document_id)
            self.assertEqual(wording.status, "ready")
            self.assertEqual(repair["wording"], {
                "id": wording.id, "title": wording.title, "filename": wording.filename,
                "version": wording.version, "status": "ready", "error": None,
            })
            original = self.client.get(f"/api/policy-documents/{repair['wording']['id']}/original")
            self.assertEqual(original.status_code, 200)
            self.assertEqual(original.content, base64.b64decode(wording.content))
            self.assertIn(policy_id, base64.b64decode(wording.content).decode())
            self.assertIn("1. Scope.", base64.b64decode(wording.content).decode())
            self.assertNotIn("wording", session.get(Policy, policy_id).payload)
            self.assertEqual(assessment.execution["mode"], "anthropic-tools")
            self.assertEqual(assessment.context["policy_search_mode"], "bound-synthetic")
            self.assertEqual(assessment.context["synthetic_wording_document_id"], wording.id)
            self.assertNotIn("policy_corpus_ids", assessment.context)
            self.assertNotIn("wording_document_id", assessment.context)
            self.assertNotIn(wording.id, uploaded_document_ids(self.store))
        with self.store.session.begin() as session:
            session.get(PolicyBinding, theft["policy"]["policy_id"]).document_id = unrelated
        catalogue = self.client.get("/api/generator/people-policies").json()["records"]
        self.assertIsNone(next(record["wording"] for record in catalogue if record["policy"] == theft["policy"]))
        [theft_id] = self.commit(self.preview(draft(theft, "theft")), execution_mode="anthropic")
        with self.store.session() as session:
            assessment = session.get(Assessment, theft_id)
            self.assertNotIn("synthetic_wording_document_id", assessment.context)

    def test_commit_rejects_policyholder_link_changed_since_preview(self):
        repair, theft = self.save_catalogue()
        preview = self.preview(draft(repair))
        with self.store.session.begin() as session:
            session.get(SyntheticPolicyHolder, repair["policy"]["policy_id"]).person_id = theft["person"]["person_id"]
        result = self.client.post(f"/api/generator/preview/{preview['batch_id']}/commit")
        self.assertEqual(result.status_code, 409)
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])

    def test_coverage_does_not_accept_a_changed_policyholder_link(self):
        repair, theft = self.save_catalogue()
        [identifier] = self.commit(self.preview(draft(repair)))
        with self.store.session.begin() as session:
            session.get(SyntheticPolicyHolder, repair["policy"]["policy_id"]).person_id = theft["person"]["person_id"]
        self.workflow.run(identifier)
        with self.store.session() as session:
            finding = session.get(Assessment, identifier).context["findings"]["coverage"]
            self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
            self.assertIn("does not match", finding["summary"])
            self.assertEqual(session.get(DecisionRecord, identifier).payload["route"], "adjuster review")

    def test_staff_approved_manual_and_generated_claims_feed_later_comparison(self):
        repair, _ = self.save_catalogue()
        label = "rear_bumper incident / Kandy case number 2025-10-15"
        [generated] = self.commit(self.preview(draft(repair, incident_key=label)))
        with self.store.session() as session:
            original = session.get(Claim, session.get(Assessment, generated).claim_id).payload["incident_id"]
            self.assertRegex(original, r"^INC-[A-F0-9]{32}$")
        body = {"policy_id": "POL-MANUAL", "registration": "ABC-1234", "loss_type": "repair",
                "loss_date": "2025-10-15", "gross_amount": "112000", "narrative": "A collision damaged the front bumper."}
        with patch.dict("os.environ", {"CLAIMS_LLM_MODE": "deterministic"}):
            self.client.post("/api/auth/login", json={"email": "user@ruhunu.local", "password": "UserDemo!2026"})
            created = self.client.post("/api/portal/claims", json=body)
        self.assertEqual(created.status_code, 201, created.text)
        manual = created.json()["id"]
        self.client.post("/api/auth/login", json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"})
        for identifier in (generated, manual):
            self.workflow.run(identifier)
            result = self.client.post(f"/api/portal/claims/{identifier}/review", json={
                "outcome": "approved", "reason": "Reviewed supporting evidence and approved the amount."})
            self.assertEqual(result.status_code, 200, result.text)
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})
        current_claim = draft(repair, incident_key=label.replace("_", "-"))
        current_claim["claim"].update(gross_amount="125000.00", additional_costs=[{"reason": "Towing", "amount": "5000.00"}])
        [current] = self.commit(self.preview(current_claim))
        self.workflow.run(current)
        with self.store.session() as session:
            history_call = session.scalar(select(ToolCall).where(ToolCall.assessment_id == current, ToolCall.tool == "retrieve_comparable_claims"))
            self.assertEqual({row["assessment_id"] for row in history_call.response["records"]}, {generated, manual})
            self.assertEqual({row["source"] for row in history_call.response["records"]}, {"Manual submission", "Synthetic generator"})
            generated_record = next(row for row in history_call.response["records"] if row["assessment_id"] == generated)
            self.assertEqual(generated_record["documents"][0]["line_items"][0]["description"], "Bumper repair")
            self.assertIn("rear bumper", generated_record["narrative"])
            self.assertTrue(session.get(Assessment, current).context["historical_available"])
            self.assertIsNotNone(session.scalar(select(AgentRun).where(AgentRun.assessment_id == current, AgentRun.stage == "r2:historical")))

    def test_model_cannot_dispatch_historical_with_no_eligible_history(self):
        repair, _ = self.save_catalogue()
        [identifier] = self.commit(self.preview(draft(repair)))
        def ignore_gate(store, assessment_id, context, claim, round_number):
            return offline_readiness({**context, "historical_available": True}, claim)
        with patch("backend.workflow.intake_first_graph.assess_readiness", side_effect=ignore_gate):
            self.workflow.run(identifier)
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            self.assertIn("not applicable", row.context["readiness_history"]["2"]["error"])
            self.assertNotIn("historical", row.context["findings"])
