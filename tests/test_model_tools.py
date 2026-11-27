"""Model tool boundaries and evidence retained after deterministic retrieval."""
import base64
import copy
import json
from unittest.mock import patch

from sqlalchemy import select

import test_agentic as fixtures
from backend.storage.sqlite_store import Assessment, Attachment, AttachmentExtraction, Claim, ToolCall, digest
from backend.agents import claude_investigation as agentic
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.evidence.specialist_evidence_tools import build_tools
from scripts.debug_workflow import scripted_analysis_fields


def analysis(references, provider_args):
    stage, messages = provider_args[2], provider_args[4]
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "analysis",
        "name": "submit_analysis", "input": {"summary": "Supplied evidence assessed",
        "review_reasons": [], "evidence": references, **scripted_analysis_fields(stage, json.loads(messages[0]["content"]))}}]}


class ModelToolTests(fixtures.AgenticFixtureTestCase):
    def make_manual(self, identifier):
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            claim = session.get(Claim, row.claim_id)
            claim.payload = {**claim.payload, "source": "Manual submission"}
            claim.payload_hash = digest(claim.payload)

    def context(self, follow_up=False):
        if not follow_up:
            return {"findings": {}}
        return {"findings": {"historical": {"details": {"fraud_investigation": {"leads": [
            {"document_reference": "unmatched", "history_ids": []}]}}}}}

    def test_exact_model_tools_for_all_roles_and_manual_claims(self):
        for manual in (False, True):
            identifier = self.fixture.prepare(int(manual))
            if manual:
                self.make_manual(identifier)
            for number, role in enumerate(("intake", "coverage", "historical", "fraud", "fraud")):
                with self.subTest(manual=manual, role=role, follow_up=number == 4):
                    expected = {"submit_analysis"}
                    if role != "coverage":
                        expected.add("read_submission")
                    if manual:
                        expected.add("read_uploaded_documents")
                    if role == "coverage" and manual:
                        expected.add("search_policy_documents")
                    def provider(*args):
                        self.assertEqual({t["name"] for t in args[5]}, expected)
                        return analysis([json.loads(args[4][0]["content"])["required_checks"]["evidence_ref"]], args)
                    with patch.object(agentic, "call_model", side_effect=provider) as model:
                        finding = agentic.run_specialist(self.store, identifier, self.context(number == 4), number, role, "Assess")
                    self.assertEqual(model.call_count, 0 if role == "intake" else 1)
                    self.assertTrue(finding["details"]["checks_executed"])
                    # A manual Coverage summary must cite applicable wording.
                    if role == "coverage" and manual:
                        self.assertEqual(finding["status"], "review")
                        self.assertFalse(finding["details"]["policy_rag"]["cited"])
                    elif role == "coverage":
                        self.assertEqual(finding["details"]["coverage_status"], "COVERED")
                        self.assertNotIn("policy_rag", finding["details"])

    def test_removed_tools_are_rejected_before_retrieval(self):
        identifier = self.fixture.prepare()
        cases = [("coverage", "read_context"),
                 ("coverage", "retrieve_policy"), ("historical", "retrieve_comparable_claims"),
                 ("fraud", "retrieve_comparable_claims"), ("fraud", "verify_historical_leads"),
                 ("coverage", "search_policy_documents"), ("coverage", "read_submission")]
        for number, (role, name) in enumerate(cases):
            with self.subTest(role=role, tool=name):
                before = []
                def provider(*args):
                    with self.store.session() as session:
                        before.extend(session.scalars(select(ToolCall.id).where(
                            ToolCall.assessment_id == identifier, ToolCall.stage == args[2])).all())
                    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "denied", "name": name, "input": {}}]}
                with patch.object(agentic, "call_model", side_effect=provider):
                    with self.assertRaisesRegex(ValueError, "Forbidden specialist tool"):
                        agentic.run_specialist(self.store, identifier, self.context(name == "verify_historical_leads"), number, role, "Assess")
                with self.store.session() as session:
                    after = session.scalars(select(ToolCall.id).where(ToolCall.assessment_id == identifier,
                        ToolCall.stage == f"r{number}:{role}")).all()
                self.assertEqual(after, before)

    def test_coverage_and_fraud_lookup_use_the_saved_vehicle_and_loss_date(self):
        identifier = self.fixture.prepare()
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            assessment.context = {**assessment.context, "selected_policy": {"id": "wording"}}
            claim = session.get(Claim, assessment.claim_id)
            claim.payload = {**claim.payload, "loss_type": "repair", "loss_date": "2026-11-12",
                             "vehicle": {"make": "Toyota", "model": "Aqua", "year": 2019}}
            claim.payload_hash = digest(claim.payload)
        name = "retrieve_vehicle_market_value"
        with patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value={"status": "no_observation"}) as lookup:
            results = {stage: build_tools(self.store, identifier, stage)[name].invoke({}) for stage in ("coverage", "fraud_initial")}
        query = {"make": "Toyota", "model": "Aqua", "year": 2019, "as_of": "2026-11-12"}
        self.assertEqual([call.args[0].model_dump(mode="json") for call in lookup.call_args_list], [query, query])
        self.assertNotIn(name, build_tools(self.store, identifier, "intake"))
        with self.store.session() as session:
            calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier, ToolCall.tool == name)).all()
            self.assertEqual({call.stage: call.response for call in calls}, results)
            self.assertTrue(all(call.request == query for call in calls))

    def test_fraud_automatically_compares_market_price_without_escalating_price_alone(self):
        identifier = self.fixture.prepare()
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            claim.payload = {**claim.payload, "loss_type": "theft", "gross_amount": "4000000.00", "currency": "LKR",
                             "loss_date": "2026-11-12", "vehicle": {"make": "Toyota", "model": "Aqua", "year": 2019},
                             "documents": [{"type": "vehicle_valuation", "name": "Vehicle valuation", "total": "4200000.00"}]}
            claim.payload_hash = digest(claim.payload)

        name = "retrieve_vehicle_market_value"
        benchmark = {"status": "found", "asking_price_lkr": "2000000", "observation_date": "2026-11-10"}

        def provider(*args):
            required = json.loads(args[4][0]["content"])["required_checks"]
            [lookup] = required["supporting_evidence"]
            self.assertEqual(lookup["tool"], name)
            self.assertNotIn(name, {tool["name"] for tool in args[5]})
            source = lookup["result"]["evidence_ref"]
            self.assertEqual(required["result"]["details"]["vehicle_market_check"]["evidence_ref"], source)
            result = analysis([required["evidence_ref"]], args)
            result["content"][0]["input"]["fraud_concerns"] = [{
                "severity": "high", "basis": "conflicting_evidence", "reason": "The valuation exceeds the average asking price.",
                "evidence": [source], "observed_facts": [{"path": "submission:/documents/0/total", "value": "4200000.00"},
                                                          {"path": source + ":/asking_price_lkr", "value": "2000000"}],
            }]
            return result

        with patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value", return_value=benchmark) as lookup, \
                patch.object(agentic, "call_model", side_effect=provider):
            finding = agentic.run_specialist(self.store, identifier, self.context(), 1, "fraud", "Assess")
        lookup.assert_called_once()
        details = finding["details"]
        self.assertEqual(details["vehicle_market_check"]["comparisons"], [{
            "label": "Vehicle valuation", "amount_lkr": "4200000.00", "percent_of_benchmark": "210.0",
            "source": "submission:/documents/0/total",
        }])
        self.assertIn(details["vehicle_market_check"]["evidence_ref"], finding["evidence"])
        self.assertEqual(details["fraud_concerns"], [])
        self.assertEqual(finding["status"], "pass")
        self.assertIn("average advertised asking price", finding["summary"])
        self.assertNotIn(name, build_tools(self.store, identifier, "fraud_followup",
                                           context_snapshot={"findings": {"fraud_initial": finding}}))

    def test_fraud_skips_market_lookup_without_a_usable_valuation_document(self):
        identifier = self.fixture.prepare()

        def provider(*args):
            required = json.loads(args[4][0]["content"])["required_checks"]
            self.assertEqual(required["supporting_evidence"], [])
            return analysis([required["evidence_ref"]], args)

        with patch("backend.evidence.specialist_evidence_tools.lookup_vehicle_market_value") as lookup, \
                patch.object(agentic, "call_model", side_effect=provider):
            for number, documents in enumerate(([], [{"type": "vehicle_valuation", "total": None}],
                                                [{"type": "independent_assessor_report", "assessed_amount_lkr": "4200000.00"}]), 1):
                with self.subTest(documents=documents):
                    with self.store.session.begin() as session:
                        claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
                        claim.payload = {**claim.payload, "loss_type": "theft", "gross_amount": "4000000.00", "currency": "LKR",
                                         "loss_date": "2026-11-12", "vehicle": {"make": "Toyota", "model": "Aqua", "year": 2019},
                                         "documents": documents}
                        claim.payload_hash = digest(claim.payload)
                    finding = agentic.run_specialist(self.store, identifier, self.context(), number, "fraud", "Assess")
                    self.assertTrue(finding["details"]["checks_executed"])
                    self.assertNotIn("vehicle_market_check", finding["details"])
            lookup.assert_not_called()

    def test_parallel_checks_supply_exact_audited_evidence_without_context_bloat(self):
        identifier = self.fixture.prepare(14)
        captured = {}
        def provider(*args):
            if args[2].startswith(("r0:", "r1:", "r2:", "r3:")) and len(args[4]) == 1:
                captured[args[2]] = json.loads(args[4][0]["content"])["required_checks"]
            return self.fixture.provider(*args)
        with patch.object(agentic, "call_model", side_effect=provider):
            self.fixture.workflow.run(identifier)
        expected = {"r1:coverage": ["retrieve_policy"],
                    "r1:fraud": [], "r2:historical": ["retrieve_comparable_claims"],
                    "r3:fraud": ["verify_historical_leads"]}
        self.assertEqual(set(captured), set(expected))
        with self.store.session() as session:
            intake = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                ToolCall.stage == "r0:intake", ToolCall.tool == "execute_checks"))
            self.assertTrue(intake.response["result"]["details"]["checks_executed"])
            self.assertIn("read_coverage_requirements", [item["tool"] for item in intake.response["supporting_evidence"]])
            for stage, names in expected.items():
                supplied = captured[stage]
                self.assertEqual([e["tool"] for e in supplied["supporting_evidence"]], names)
                saved = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                    ToolCall.stage == stage, ToolCall.tool == "execute_checks"))
                self.assertEqual(saved.response, supplied)
                self.assertEqual(saved.response_hash, digest(supplied))
                for entry in supplied["supporting_evidence"]:
                    rows = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier,
                        ToolCall.stage == stage, ToolCall.tool == entry["tool"])).all()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(entry["request"], rows[0].request)
                    self.assertEqual(entry["result"], rows[0].response)
            context = session.get(Assessment, identifier).context
            self.assertNotIn("supporting_evidence", json.dumps(context["findings"]))
        records = captured["r2:historical"]["supporting_evidence"][0]["result"]["records"]
        self.assertTrue(records[0]["documents"][0]["line_items"])
        self.assertTrue(records[0]["narrative"])
        self.assertTrue(captured["r3:fraud"]["supporting_evidence"][0]["result"]["verified"])
        package = export_audit(self.store, identifier)
        self.assertTrue(verify_audit(package))
        entry = next(r for r in package["retrievals"] if r["stage"] == "r2:historical" and r["tool"] == "execute_checks")
        entry["response"]["supporting_evidence"][0]["result"]["records"] = []
        self.assertFalse(verify_audit(package))

    def test_callback_observes_committed_copies_without_altering_evidence(self):
        identifier = self.fixture.prepare()
        observed = []
        def callback(name, request, result):
            with self.store.session() as session:
                saved = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                    ToolCall.stage == "observer", ToolCall.tool == name))
                self.assertIsNotNone(saved)
                self.assertEqual(saved.response, result)
            observed.append(name)
            result["policy"]["limit"] = "changed by callback"
            request["policy_id"] = "changed by callback"
        tools = build_tools(self.store, identifier, "coverage", audit_stage="observer", on_result=callback)
        result = tools["retrieve_policy"].invoke({})
        self.assertEqual(observed, ["retrieve_policy"])
        self.assertNotEqual(result["policy"]["limit"], "changed by callback")
        with self.store.session() as session:
            saved = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier, ToolCall.stage == "observer"))
            self.assertEqual(saved.response, result)
            self.assertEqual(saved.response_hash, digest(result))
            self.assertNotEqual(saved.request["policy_id"], "changed by callback")

    def test_optional_manual_reads_are_citable_and_do_not_mutate_initial_checks(self):
        identifier = self.fixture.prepare()
        self.make_manual(identifier)
        upload = {"attachment_id": "manual-report", "source": "attachment:manual-report", "status": "extracted",
                  "document_type": "incident_report", "summary": "A stone struck the windscreen.",
                  "gaps": ["Vehicle plate is blurry"], "provenance": "claimant upload; unverified"}
        with self.store.session.begin() as session:
            session.add(Attachment(id="manual-report", assessment_id=identifier, name="report.pdf", media_type="application/pdf",
                                   content=base64.b64encode(b"fixture").decode()))
            session.add(AttachmentExtraction(attachment_id="manual-report", payload=upload))
        initial = []
        def provider(*args):
            messages = args[4]
            supplied = json.loads(messages[0]["content"])["required_checks"]
            if len(messages) == 1:
                initial.append(copy.deepcopy(supplied))
                return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": name,
                    "name": name, "input": {}} for name in ("read_submission", "read_uploaded_documents")]}
            outputs = [json.loads(block["content"]) for block in messages[-1]["content"]]
            self.assertIn("narrative", outputs[0]["result"])
            self.assertEqual(outputs[1]["result"]["evidence"], [upload])
            self.assertEqual(supplied, initial[0])
            return analysis([supplied["evidence_ref"], *(item["evidence_ref"] for item in outputs)], args)
        with patch.object(agentic, "call_model", side_effect=provider) as model:
            finding = agentic.run_specialist(self.store, identifier, {}, 1, "fraud", "Screen the narrative and uploaded records")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(finding["details"]["tools_used"], ["execute_checks", "read_submission", "read_uploaded_documents"])
        self.assertTrue(set(finding["evidence"]).issubset(finding["details"]["sources"]))
        with self.store.session() as session:
            saved = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier, ToolCall.tool == "execute_checks"))
            self.assertEqual(saved.response, initial[0])
            self.assertEqual(saved.response_hash, digest(initial[0]))
            self.assertEqual(saved.response["supporting_evidence"], [])

    def test_missing_policy_response_is_supplied_without_an_extra_tool(self):
        identifier = self.fixture.prepare()
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            claim = session.get(Claim, row.claim_id)
            claim.policy_id = "unavailable-policy"
            claim.payload = {**claim.payload, "policy_id": claim.policy_id}
            claim.payload_hash = digest(claim.payload)
        def provider(*args):
            supplied = json.loads(args[4][0]["content"])["required_checks"]
            self.assertEqual(supplied["supporting_evidence"], [{"tool": "retrieve_policy",
                "request": {"policy_id": "unavailable-policy"}, "result": {"policy": None}}])
            return analysis([supplied["evidence_ref"]], args)
        with patch.object(agentic, "call_model", side_effect=provider):
            finding = agentic.run_specialist(self.store, identifier, {}, 0, "coverage", "Assess coverage")
        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
        self.assertEqual(finding["status"], "review")
