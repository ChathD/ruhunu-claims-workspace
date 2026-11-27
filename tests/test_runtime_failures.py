"""Operational failures stay actionable and do not cause unnecessary judge calls."""
import json
import os
import unittest
from unittest.mock import patch

import httpx
from sqlalchemy import select

import test_agentic as fixtures
from backend.__main__ import main
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, DecisionRecord, ModelInvocation, ToolCall
from backend.agents import claude_investigation as agentic
from backend.assurance.investigation_display import (AdjusterPriority, adjuster_review_brief, adjuster_review_candidates,
                                                      fraud_escalation_brief, fraud_identification_brief, plain_review_text, select_adjuster_summary,
                                                      settlement_review_reasons)
from backend.assurance.investigation_pdf import _decision_summary
from backend.claim_schemas import Route
from backend.workflow.routing_rules import select_route


class RuntimeFailureTests(fixtures.AgenticFixtureTestCase):
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

    def test_truncated_readiness_retries_same_round_and_publishes(self):
        identifier = self.fixture.prepare()
        real_client, real_call = httpx.Client, agentic.call_model
        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                return httpx.Response(200, json={"stop_reason": "max_tokens",
                    "content": [{"type": "text", "text": "Discard this incomplete readiness result."}]})
            return httpx.Response(200, json=self.fixture.provider(self.store, identifier, "readiness:1",
                payload["system"], payload["messages"], payload["tools"], payload["tool_choice"]))

        def model(*args, **kwargs):
            return real_call(*args, **kwargs) if args[2] == "readiness:1" else self.fixture.provider(*args)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture", "CLAIMS_STREAM_MODEL": "0"}), \
             patch.object(agentic.httpx, "Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs)), \
             patch.object(agentic, "call_model", side_effect=model):
            self.fixture.workflow.run(identifier)
        self.assertEqual([request["max_tokens"] for request in requests], [3200, 8192])
        self.assertEqual(requests[0]["messages"][0], requests[1]["messages"][0])
        self.assertEqual(requests[1]["tool_choice"], {"type": "tool", "name": "assess_readiness"})
        self.assertIn("length limit", requests[1]["messages"][1]["content"])
        self.assertNotIn("Discard this incomplete", str(requests[1]["messages"]))
        with self.store.session() as session:
            calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier,
                ModelInvocation.stage == "readiness:1").order_by(ModelInvocation.created_at)).all()
            self.assertEqual([call.status for call in calls], ["failed", "complete"])
            self.assertEqual(calls[0].response["stop_reason"], "max_tokens")
            self.assertEqual(len(session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier,
                AgentRun.stage == "r0:intake")).all()), 1)
            self.assertIsNotNone(session.get(DecisionRecord, identifier))

    def test_insufficient_anthropic_credits_explain_interruption_without_retrying(self):
        real_client, real_call = httpx.Client, agentic.call_model
        reason = "Anthropic API credits are too low. Add credits in Claude Console, then retry."
        for index, stage, streaming, label in ((0, "r1:coverage", "1", "Coverage"), (5, "readiness:1", "0", "Readiness")):
            with self.subTest(stage=stage):
                identifier = self.fixture.prepare(index)
                requests = []

                def handle(request):
                    requests.append(request)
                    body = {"error": {"type": "invalid_request_error",
                                      "message": "Your credit balance is too low to access the Anthropic API. Raw provider note."}}
                    return httpx.Response(400, headers={"request-id": "req_fixture"},
                                          stream=httpx.ByteStream(json.dumps(body).encode()))

                def model(*args, **kwargs):
                    return real_call(*args, **kwargs) if args[2] == stage else self.fixture.provider(*args)

                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture", "CLAIMS_STREAM_MODEL": streaming}), \
                     patch.object(agentic.httpx, "Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs)), \
                     patch.object(agentic, "call_model", side_effect=model):
                    self.fixture.workflow.run(identifier)
                self.assertEqual(len(requests), 1)
                with self.store.session() as session:
                    row = session.get(Assessment, identifier)
                    call = session.scalar(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier,
                        ModelInvocation.stage == stage))
                    self.assertEqual(row.status, "failed")
                    self.assertEqual(row.error, f"{label} could not complete. {reason}")
                    self.assertEqual(call.response, {"http_status": 400, "request_id": "req_fixture",
                        "error_type": "AnthropicRequestError", "provider_error": reason})
                    self.assertIsNone(session.get(DecisionRecord, identifier))

    def test_readiness_retries_schema_and_dispatch_validation_failures(self):
        for index, failure in ((0, "schema"), (5, "dispatch")):
            with self.subTest(failure=failure):
                identifier = self.fixture.prepare(index)
                readiness_calls = []

                def model(*args, **kwargs):
                    response = self.fixture.provider(*args)
                    if args[2] == "readiness:1":
                        readiness_calls.append((args[4], kwargs))
                        if len(readiness_calls) == 1:
                            plan = response["content"][0]["input"]
                            if failure == "schema":
                                del plan["specialists"]["coverage"]["reason"]
                            else:
                                plan["specialists"]["historical"]["state"] = "READY"
                                plan["tasks"].append({"specialist": "historical", "task": "Run before Coverage"})
                    return response

                with patch.object(agentic, "call_model", side_effect=model):
                    self.fixture.workflow.run(identifier)
                first, retry = readiness_calls
                self.assertEqual(first[0][0], retry[0][0])
                self.assertEqual(retry[1]["max_tokens"], agentic.OUTPUT_RETRY_MAX_TOKENS)
                feedback = retry[0][1]["content"]
                self.assertIn("invalid_fields" if failure == "schema" else "requires confirmed coverage", feedback)
                with self.store.session() as session:
                    row = session.get(Assessment, identifier)
                    self.assertEqual(row.context["readiness_history"]["1"]["specialists"]["historical"]["state"], "BLOCKED_DEPENDENCY")
                    self.assertIsNotNone(session.get(DecisionRecord, identifier))

    def test_truncated_fraud_finding_retries_with_more_tokens_and_publishes(self):
        identifier = self.fixture.prepare()
        attempts = []

        def provider(*args, **kwargs):
            if args[2] == "r1:fraud":
                attempts.append((args[4][:], args[6], kwargs))
                if len(attempts) == 1:
                    raise agentic.ModelOutputTruncated("The fraud finding reached its output limit.")
            return self.fixture.provider(*args)

        with patch.object(agentic, "call_model", side_effect=provider):
            self.fixture.workflow.run(identifier)

        first, retry = attempts
        self.assertEqual(first[0], retry[0])
        self.assertEqual(retry[1], {"type": "tool", "name": "submit_analysis"})
        self.assertEqual(retry[2]["max_tokens"], agentic.OUTPUT_RETRY_MAX_TOKENS)
        with self.store.session() as session:
            self.assertIsNotNone(session.get(DecisionRecord, identifier))

    def test_fraud_can_cite_the_submitted_narrative_in_v3(self):
        identifier = self.fixture.prepare()
        source = "submission:/narrative"

        def provider(*args, **kwargs):
            response = self.fixture.provider(*args)
            if args[2] == "r1:fraud":
                submission = json.loads(args[4][0]["content"])["submission"]
                self.assertTrue(submission["narrative"])
                evidence_items = args[5][-1]["input_schema"]["properties"]["evidence"]["items"]
                self.assertIn(source, evidence_items["enum"])
                response["content"][0]["input"]["evidence"].append(source)
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            self.fixture.workflow.run(identifier)

        with self.store.session() as session:
            self.assertIsNotNone(session.get(DecisionRecord, identifier))
            finding = session.get(Assessment, identifier).context["findings"]["fraud_initial"]
            self.assertIn(source, finding["evidence"])

    def test_fraud_repair_with_unknown_citation_routes_to_review(self):
        identifier = self.fixture.prepare()
        unknown = "tool:unavailable"
        attempts = []

        def provider(*args, **kwargs):
            if args[2] != "r1:fraud":
                return self.fixture.provider(*args)
            analysis = {"summary": "Reviewed Fraud checks.", "review_reasons": []}
            messages = args[4]
            if not attempts:
                analysis["evidence"] = [json.loads(messages[0]["content"])["required_checks"]["evidence_ref"]]
                analysis["plausibility"] = {"assessment": "inconclusive", "reason": "x" * 701}
            else:
                self.assertIn("string_too_long", messages[-1]["content"][0]["content"])
                analysis["evidence"] = [unknown]
            attempts.append(analysis)
            return {"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "id": "submit", "name": "submit_analysis", "input": analysis,
            }]}

        with patch.object(agentic, "call_model", side_effect=provider):
            self.fixture.workflow.run(identifier)

        self.assertEqual(len(attempts), 2)
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            finding = row.context["findings"]["fraud_initial"]
            self.assertEqual(row.status, "complete")
            self.assertEqual(finding["status"], "review")
            self.assertTrue(finding["details"]["checks_executed"])
            self.assertIn("unavailable evidence", finding["summary"])
            self.assertNotIn(unknown, finding["evidence"])
            self.assertEqual(session.get(DecisionRecord, identifier).payload["route"], Route.REVIEW.value)

    def test_persistently_invalid_readiness_does_not_dispatch(self):
        identifier = self.fixture.prepare()

        def model(*args, **kwargs):
            self.assertEqual(args[2], "readiness:1")
            response = self.fixture.provider(*args)
            plan = response["content"][0]["input"]
            plan["specialists"]["historical"]["state"] = "READY"
            plan["tasks"].append({"specialist": "historical", "task": "Run before Coverage"})
            return response

        with patch.object(agentic, "call_model", side_effect=model) as calls:
            self.fixture.workflow.run(identifier)
        self.assertEqual(calls.call_count, 2)
        with self.store.session() as session:
            row = session.get(Assessment, identifier)
            history = row.context["readiness_history"]["1"]
            self.assertEqual(history["rejected_selection"]["tasks"][-1]["specialist"], "historical")
            self.assertEqual(history["error"], "Historical comparison requires confirmed coverage")
            self.assertEqual(row.status, "failed")
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

    def test_judge_omits_provisional_payable_from_coverage_narrative(self):
        identifier = self.fixture.prepare()
        summary = "The invoice is LKR 58,000. Net payable is LKR 58,000."
        coverage = {"status": "review", "summary": summary, "details": {
            "analysis": {"summary": summary}, "financial": {"payable": None},
            "coverage_outcome": {"state": "partially_confirmed", "reason": summary},
        }}

        def provider(*args):
            packet = json.loads(args[4][0]["content"])
            self.assertIsNone(packet["publication_controls"]["committed_coverage_payable_LKR"])
            coverage_for_judge = packet["findings"]["coverage"]
            expected = "The invoice is LKR 58,000. Payable amount remains unverified."
            self.assertEqual(coverage_for_judge["summary"], expected)
            self.assertEqual(coverage_for_judge["details"]["analysis"]["summary"], expected)
            self.assertEqual(coverage_for_judge["details"]["coverage_outcome"]["reason"], expected)
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "judge", "name": "recommend_route",
                "input": {"route": "adjuster review", "explanation": "The payable amount is unverified.",
                          "evidence": [], "outstanding_gaps": []}}]}

        with patch.object(agentic, "call_model", side_effect=provider):
            agentic.judge(self.store, identifier, {"findings": {"coverage": coverage}})
        self.assertEqual(coverage["details"]["coverage_outcome"]["reason"], summary)

    def test_judge_keeps_only_two_factual_discrepancies_in_the_adjuster_summary(self):
        identifier = self.fixture.prepare()
        first = "The repair quote includes a rear door despite the incident account describing front damage only."
        second = "The estimate totals LKR 200,000.00 and its line items total LKR 150,000.00."
        issues = [first, "The garage used a different abbreviation for the same address.", second,
                  "The witness and driver describe different vehicles."]
        unresolved = "The file does not establish when the cleaning work was authorized."
        clarified = "The submitted claim documents do not establish whether the flood cleanup was authorized before it was carried out."
        details = {"document_inconsistencies": [{"reason": reason, "document_refs": ["submission:/documents/0"]} for reason in issues],
                   "review_reasons": ["Read the policy wording."], "coverage_status": "UNRESOLVED",
                   "coverage_items": [{"item": "Cleaning costs", "reason": unresolved, "outcome": "undetermined",
                                       "document_refs": ["submission:/documents/1"]}]}
        context = {"findings": {"coverage": {"details": details}}, "historical_available": False}
        priorities = [{"candidate_id": ref, "sentence": text} for ref, text in (
            ("D999", "The police report gives a different location."),
            ("D2", "Check the address on the estimate."),
            ("D4", "The witness cites Section I."),
            ("D1", "The estimate includes the rear door. The incident accounts mention only front damage."),
            ("D5", unresolved), ("D5", clarified), ("D3", second), ("D1", first), ("D4", issues[3]),
        )]

        def provider(*args):
            packet = json.loads(args[4][0]["content"])
            self.assertEqual([value.partition(" Records:")[0] for value in packet["adjuster_candidates"].values()],
                             [*issues, f"Cleaning costs: {unresolved}"])
            self.assertIn("Evaluate each supplied adjuster_candidates entry for materiality", args[3])
            return {"content": [{"type": "tool_use", "name": "recommend_route", "input": {
                "route": "adjuster review", "explanation": "Triage route", "evidence": [], "outstanding_gaps": [],
                "adjuster_priorities": selected,
            }}]}

        for selected, expected in ((priorities, [clarified, second]), ([], []), ("Invalid summary format", [])):
            with self.subTest(selected=selected), patch.object(agentic, "call_model", side_effect=provider):
                result = agentic.judge(self.store, identifier, context)
            self.assertEqual(result["route"], "adjuster review")
            self.assertEqual(result["adjuster_summary"], expected)

    def test_one_adjuster_priority_can_use_two_sentences_with_named_records(self):
        claim = {"currency": "LKR", "documents": [
            {"type": "repair_estimate", "date": "2026-10-30"},
            {"type": "additional_cost_receipt", "category": "cleaning", "date": "2026-10-30", "total": "8500",
             "description": "Interior flood decontamination"},
        ]}
        refs = ["submission:/documents/0", "submission:/documents/1"]
        findings = {"coverage": {"details": {"submitted_documents": dict.fromkeys(refs, "Claimant name"),
            "document_inconsistencies": [{"document_refs": refs, "reason": "The authorization timing is unclear."}]}}}
        candidates = adjuster_review_candidates(claim, findings)
        self.assertIn("Motor repair quotation (dated 2026-10-30)", candidates["D1"])
        self.assertIn("Cleaning service receipt (dated 2026-10-30, LKR 8,500.00)", candidates["D1"])
        self.assertIn("Interior flood decontamination", candidates["D1"])
        self.assertNotIn("Claimant name", candidates["D1"])
        first = "The submitted claim documents do not establish whether the LKR 8,500 flood cleanup was authorized before it was carried out."
        second = "The repair estimate and cleaning receipt are dated one day before the claim was filed."
        priority = AdjusterPriority(candidate_id="D1", sentence=first, context_sentence=second)
        self.assertEqual(select_adjuster_summary([priority], candidates), [first, second])
        other = AdjusterPriority(candidate_id="D2", sentence="The incident report and witness statement list different locations.")
        self.assertEqual(select_adjuster_summary([priority, other], {**candidates, "D2": other.sentence}), [first, other.sentence])

    def test_summary_evidence_requests_change_routing_for_a_repair_claim(self):
        identifier = self.fixture.prepare()
        with self.store.session() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id).payload
        claim = {**claim, "loss_type": "repair", "gross_amount": "325000.00"}
        reason = "No cleaning invoice was submitted to substantiate the LKR 8,500 flood cleanup charge."
        context = {"version": 1, "findings": {"coverage": {"agent": "coverage", "status": "review", "summary": reason,
            "evidence": [], "details": {"checks_executed": True, "coverage_status": "UNRESOLVED", "review_reasons": [reason]},
        }}}
        request = {"candidate_id": "D1", "finding_quote": reason,
                   "document": "Flood cleanup invoice", "reason": "The invoice substantiates the LKR 8,500 cleanup charge."}

        def provider(*args):
            context_sent = json.loads(args[4][0]["content"])
            self.assertEqual(context_sent["adjuster_candidates"], {"D1": reason})
            return {"content": [{"type": "tool_use", "name": "recommend_route", "input": {
                "route": "adjuster review", "explanation": "The cleanup costs need evidence.", "evidence": [], "outstanding_gaps": [],
                "evidence_requests": [request, {**request, "finding_quote": "No police report was submitted.", "document": "Police report"}],
                "adjuster_priorities": [{"candidate_id": "D1", "sentence": reason}],
            }}]}

        with patch.object(agentic, "call_model", side_effect=provider):
            recommendation = agentic.judge(self.store, identifier, context)
        result = select_route(claim, {**context, "recommendation": recommendation}, [])
        self.assertEqual(result.route.value, "request more information")
        self.assertEqual(result.reason_codes, ["MISSING_EVIDENCE"])
        self.assertEqual([item.document for item in result.information_required], ["Flood cleanup invoice"])
        self.assertIsNone(result.payable_amount)
        context["findings"]["coverage"]["details"]["coverage_status"] = "NOT_COVERED"
        result = select_route(claim, {**context, "recommendation": recommendation}, [])
        self.assertEqual(result.route.value, "adjuster review")
        self.assertEqual(result.information_required, [])

    def test_adjuster_review_always_has_a_summary_from_recorded_facts_or_routing_reason(self):
        claim = {"loss_date": "2026-09-14", "gross_amount": "485000", "currency": "LKR"}
        decision = {"route": "adjuster review", "reason_codes": ["GROSS_AUTHORITY"], "payable_amount": None}
        context = {"adjuster_summary": [], "policy_schedule": {"starts_on": "2026-10-20", "ends_on": "2027-10-19"},
                   "findings": {"coverage": {"details": {"coverage_status": "INACTIVE",
                       "financial": {"gross": "485000", "cost_documents_total": "410000", "additional_costs_total": "5000"}}}}}
        self.assertEqual(adjuster_review_brief(claim, context, decision=decision),
                         "The reported incident date of 2026-09-14 precedes the recorded start date of 2026-10-20. "
                         "The claim totals LKR 485,000.00, and the submitted costs total LKR 415,000.00.")
        self.assertEqual(adjuster_review_brief(claim, {}, decision=decision),
                         "The LKR 485,000.00 claim exceeds the LKR 250,000.00 review threshold.")
        uncertain = {"findings": {"coverage": {"details": {"coverage_status": "UNRESOLVED"}}}}
        self.assertEqual(adjuster_review_brief(claim, uncertain, decision=decision),
                         "The assessment has yet to establish which part of the reported loss qualifies for payment.")
        self.assertEqual(adjuster_review_brief(claim, {}, decision={**decision, "reason_codes": ["NO_COMPARABLE_HISTORY"]}),
                         "No comparable claims were available to benchmark the submitted amount.")

    def test_payable_review_names_the_unresolved_charge_and_approval_without_a_citation(self):
        claim = {"gross_amount": "187500.00", "currency": "LKR"}
        decision = {"route": "adjuster review", "reason_codes": ["SPECIALIST_REVIEW"], "payable_amount": None}
        context = {"findings": {"coverage": {"details": {"coverage_status": "UNRESOLVED", "coverage_items": [{
            "item": "LKR 7,500 cleanup receipt", "outcome": "undetermined",
            "reason": "The cleanup receipt was issued before claim filing. It is unclear whether prior written approval was obtained as required by Section II clause 2.",
        }]}}}}
        self.assertEqual(adjuster_review_brief(claim, context, decision=decision),
                         "The claim includes the unresolved LKR 7,500 cleanup receipt. It is unclear whether prior written approval was obtained.")
        priority = AdjusterPriority(candidate_id="D1", sentence=(
            "The LKR 7,500 cleanup charge has no documented prior approval under Condition 3 (W016)."))
        self.assertEqual(select_adjuster_summary([priority], {"D1": "The approval is unconfirmed."}),
                         ["The LKR 7,500 cleanup charge has no documented prior approval."])

    def test_adjuster_summary_is_only_shown_for_the_adjuster_review_decision(self):
        claim = {"gross_amount": "485000.00", "currency": "LKR"}
        note = "The loss-of-earnings amount lacks documentary support."
        context = {"adjuster_summary": [note]}
        for route in Route:
            with self.subTest(route=route):
                expected = note if route == Route.REVIEW else None
                self.assertEqual(adjuster_review_brief(claim, context, decision={"route": route.value}), expected)

    def test_settlement_review_explains_why_confirmed_coverage_has_no_payable_amount(self):
        approval = "Confirm whether the third-party demand includes the LKR 8,500 debris clearance charge."
        irrelevant = "The no-claim discount will step down on settlement."
        decision = {"route": "adjuster review", "payable_amount": None}
        details = {"coverage_status": "COVERED", "review_reasons": [irrelevant, approval]}
        context = {"findings": {"coverage": {"details": details}}, "selected_policy": {"id": "wording"}}
        self.assertEqual(settlement_review_reasons(context, decision), [
            "Confirm whether the third party demand includes the LKR 8,500 debris clearance charge.",
        ])
        self.assertEqual(settlement_review_reasons(context, {**decision, "payable_amount": "0"}), [])

        excess = "The M03 excess applicable to this loss could not be established."
        details.update(coverage_status="UNRESOLVED", coverage_items=[
            {"outcome": "undetermined", "reason": excess},
            {"outcome": "not_covered", "reason": "The optional repair extension is excluded."},
        ])
        self.assertEqual(settlement_review_reasons(context, decision), [excess])

    def test_confirmed_coverage_review_names_the_vehicle_and_explains_the_settlement_cap(self):
        claim = {"loss_type": "repair", "loss_date": "2026-11-25", "registration": "SYN-WP-KDR-7741",
                 "vehicle": {"year": 2019, "make": "Toyota", "model": "Vitz"}}
        coverage = {"coverage_status": "COVERED", "review_reasons": [
            "Prior written insurer approval before repairs has not been confirmed.",
            "Market value at the time of loss is not on file. Obtain a current valuation to confirm LKR 159,000 net repair cost fits the cap.",
            "The towing charge was capped at LKR 1,000.",
        ]}
        context = {"findings": {"coverage": {"details": coverage}}, "policy_schedule": {"cover": [
            {"loss_type": "repair", "conditions": "Payment is limited to the lower of the Schedule estimated value and market value."},
        ]}}
        self.assertEqual(settlement_review_reasons(context, {"payable_amount": None}, claim=claim), [
            "The adjuster needs the market value of the 2019 Toyota Vitz (SYN-WP-KDR-7741) on 25/11/2026 to calculate the payable amount. "
            "The policy caps payment at the lower of the scheduled value and the vehicle’s market value on the loss date. "
            "The recorded net repair cost before applying that cap is LKR 159,000.",
            "Prior written insurer approval before repairs has not been confirmed.",
        ])

    def test_settlement_summary_keeps_open_checks_and_omits_internal_references(self):
        coverage = {"coverage_status": "COVERED", "coverage_items": [{
            "item": "Claimant costs included in LKR 187,500 demand", "outcome": "undetermined",
            "reason": "No evidence of insurer consent or separately claimed legal costs was submitted.",
        }, {
            "item": "Clause 8 extension", "outcome": "undetermined",
            "reason": "The schedule shows a limit of Rs. 10 Million for third_party_property_damage. "
                      "Staff should confirm whether Clause 8 was purchased (submission:/documents/1), as its terms affect costs.",
        }], "review_reasons": [
            "Confirm whether Clause 18 is active, to establish whether towing costs are covered.",
            "Confirm whether Clause 8 (Third Party Property Damage extension) is listed as an active endorsement, "
            "to establish whether claimant costs are covered or require prior written insurer consent.",
            "Repair estimate (Perera Auto Works, LKR 187,500) has not been independently assessed. "
            "Obtain confirmation of parts and repair scope.",
        ]}
        context = {"findings": {"coverage": {"details": coverage}}}
        self.assertEqual(settlement_review_reasons(context, {"payable_amount": None}), [
            "No evidence of insurer consent or separately claimed legal costs was submitted.",
            "Confirm whether Clause 8 was purchased. "
            "This determines whether claimant costs are covered or require prior written insurer consent.",
            "Repair estimate, LKR 187,500, has not been independently assessed. An independent assessor checks whether the "
            "quoted work matches the damage and whether the parts and labour costs are reasonable.",
        ])

    def test_fraud_identification_uses_the_recorded_discrepancy_in_plain_text(self):
        findings = {"fraud_initial": {"details": {"fraud_concerns": [{
            "reason": "The report says front_left, but the witness says front-right (submission:/documents/1). "
                      "The cause of the discrepancy is uncertain.",
        }]}}}
        self.assertEqual(fraud_identification_brief(findings),
                         "The report says front left, but the witness says front right.")

    def test_fraud_decision_summary_uses_the_published_escalation_evidence(self):
        decision = {"route": Route.FRAUD, "justification": "A high-severity finding requires fraud review."}
        reason = "The estimate lists vehicle SYN-WP-ABC-1042 while the police report lists SYN-WP-ABC-1043."
        details = {"checks_executed": True, "sources": ["submission:/documents/0", "submission:/documents/1"],
                   "fraud_concerns_validated": True, "fraud_concerns": [
                       {"severity": "medium", "reason": "The recorded descriptions differ.", "basis": "conflicting_evidence",
                        "evidence": ["submission:/documents/0"]},
                       {"severity": "high", "reason": reason, "basis": "conflicting_evidence",
                        "evidence": ["submission:/documents/0", "submission:/documents/1"]},
                   ]}
        findings = {"fraud_initial": {"details": details}}
        self.assertEqual(_decision_summary(decision, [], fraud_reason=fraud_escalation_brief(findings, {})), reason)
        findings["intake"] = {"details": {"fraud_indicators": [{"reason": "The invoice also appears in two unrelated claims.",
                                                                "evidence": {"document_reference": "INV-741"}}]}}
        self.assertEqual(_decision_summary(decision, [], fraud_reason=fraud_escalation_brief(findings, {})),
                         "Cost document INV-741. The invoice also appears in two unrelated claims.")

    def test_adjuster_decision_summary_gives_specific_reason_without_overriding_authority(self):
        review = {"route": Route.REVIEW, "reason_codes": ["SPECIALIST_REVIEW"], "justification": "Coverage is partially confirmed."}
        reason = "The estimate includes towing, but the submitted records contain no towing receipt."
        self.assertEqual(_decision_summary(review, [], adjuster_reason=reason), reason)
        review.update(reason_codes=["POLICY_SCHEDULE_REVIEW"], justification="Coverage is partially confirmed.")
        settlement = "The adjuster must verify the settlement calculation before a payable amount can be recorded."
        self.assertEqual(_decision_summary(review, [], adjuster_reason="Coverage is confirmed for the reported loss.",
                                           settlement_reasons=[settlement]), settlement)
        review.update(reason_codes=["GROSS_AUTHORITY"], justification="The LKR 300,000 claim exceeds the LKR 250,000 review threshold.")
        self.assertEqual(_decision_summary(review, [], adjuster_reason=reason), review["justification"])

    def test_theft_settlement_explains_the_claim_amount_and_valuation_separately(self):
        claim = {"loss_type": "theft", "currency": "LKR", "gross_amount": "3250000.00",
                 "documents": [{"type": "vehicle_valuation", "total": "3300000.00"}],
                 "additional_costs": [{"amount": "15000.00"}, {"amount": "8500.00"}]}
        coverage = {"coverage_status": "COVERED", "schedule_confirmed": True, "review_reasons": [
            "Gross amount claimed is below the computed settlement, confirm whether the insured's own figure already nets "
            "off the excess or is an independent assessment, and confirm the final payable.",
        ]}
        context = {"findings": {"coverage": {"details": coverage}}, "policy_schedule": {"cover": [
            {"loss_type": "theft", "compulsory_excess_lkr": "25000.00"},
        ]}}
        self.assertEqual(settlement_review_reasons(context, {"payable_amount": None}, claim=claim), [
            "The claim form lists LKR 3,250,000.00 and the vehicle valuation lists LKR 3,300,000.00. "
            "Reconcile the claimed total to show whether it already deducts the LKR 25,000.00 policy excess "
            "and whether it includes the separately listed charges of LKR 23,500.00.",
        ])

    def test_review_text_preserves_slashes_in_dates(self):
        self.assertEqual(plain_review_text("Market value on 12/11/2026 is missing; estimate dated 2026-11-13."),
                         "Market value on 12/11/2026 is missing estimate dated 13/11/2026.")


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
