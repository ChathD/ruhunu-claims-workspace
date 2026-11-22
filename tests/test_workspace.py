"""Acceptance tests for the new Python / React project's backend slice."""
import copy
import base64
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.agents import coverage_specialist as coverage
from backend.api import create_app
from backend.storage.sqlite_store import AgentRun, Assessment, Claim, DecisionRecord, HumanTask, Store, ToolCall, digest
from scripts.claim_fixtures import generate_batch
from backend.claim_schemas import Route
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.routing_evaluation import evaluate
from backend.workflow.investigation_execution import execute_specialist
from backend.workflow.routing_rules import select_route
from backend.evidence.specialist_evidence_tools import build_tools
from backend.workflow.claims_workflow import ClaimsWorkflow, UnsupportedWorkflowVersion


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="claims-workspace-")
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / "claims.sqlite3")
        self.labels = self.root / "labels.sqlite3"
        self.batch = generate_batch(self.store, self.labels, seed=137, count=16)
        self.workflow = ClaimsWorkflow(self.store, self.root / "checkpoints.sqlite3")

    def tearDown(self):
        self.workflow.close()
        self.store.engine.dispose()
        self.tmp.cleanup()

    def run_claim(self, index):
        identifier = self.batch["assessment_ids"][index]
        self.workflow.run(identifier)
        with self.store.session() as session:
            assessment = session.get(Assessment, identifier)
            decision = session.get(DecisionRecord, identifier)
            return assessment, decision.payload if decision else None

    def test_all_scenarios_through_real_langgraph_and_langchain_tools(self):
        for identifier in self.batch["assessment_ids"]:
            self.workflow.run(identifier)
        report = evaluate(self.store, self.labels)
        self.assertTrue(report["complete"])
        self.assertEqual(report["matrix"], [[3, 0, 0, 0], [0, 7, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]])
        self.assertEqual(report["errors"], [])

    def test_gross_threshold_creates_task_despite_low_payable(self):
        assessment, decision = self.run_claim(5)
        self.assertEqual(decision["gross_amount"], "250000.01")
        self.assertEqual(decision["payable_amount"], "90000.01")
        self.assertEqual(decision["route"], Route.REVIEW.value)
        with self.store.session() as session:
            task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == assessment.id))
            self.assertEqual(task.queue, "adjuster")
            self.assertEqual(task.status, "pending")
            self.assertIsNone(task.reviewer)

    def test_intake_failure_stops_before_routing(self):
        with patch("backend.workflow.investigation_execution.execute_checks", side_effect=RuntimeError("check unavailable")):
            assessment, decision = self.run_claim(0)
        self.assertIsNone(decision)
        self.assertEqual(assessment.status, "failed")
        self.assertEqual(assessment.context["findings"]["intake"]["details"]["error_type"], "RuntimeError")

    def test_exact_threshold_passes_authority(self):
        _, decision = self.run_claim(2)
        self.assertEqual(decision["route"], Route.AUTO.value)

    def test_retired_document_type_requires_an_itemised_repair_estimate(self):
        identifier = self.batch["assessment_ids"][0]
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            [estimate] = claim.payload["documents"]
            claim.payload = {**claim.payload, "documents": [{**estimate, "type": "repair_invoice"}]}
            claim.payload_hash = digest(claim.payload)
        assessment, decision = self.run_claim(0)
        self.assertEqual(decision["route"], Route.INFO.value)
        self.assertEqual(decision["information_required"][0]["document"], "Itemised repair estimate")
        self.assertNotIn("coverage", assessment.context["findings"])

    def test_high_value_with_missing_documents_requests_information_and_creates_task(self):
        recommendation = {"route": Route.REVIEW.value, "explanation": "The amount exceeds the threshold.",
                          "evidence": [], "outstanding_gaps": []}
        with patch("backend.workflow.intake_first_graph.recommend_route", return_value=recommendation):
            assessment, decision = self.run_claim(6)
        self.assertEqual(decision["gross_amount"], "250000.01")
        self.assertEqual(decision["route"], Route.INFO.value)
        self.assertEqual(decision["reason_codes"], ["MISSING_EVIDENCE"])
        self.assertIsNone(decision["payable_amount"])
        [request] = decision["information_required"]
        self.assertEqual(request["document"], "Itemised repair estimate")
        self.assertEqual(assessment.context["findings"]["intake"]["details"]["missing"][0]["required_for"], "loss_and_costs")
        self.assertIn("link the proposed repairs to the incident", request["reason"])
        self.assertIn(request["document"], decision["justification"])
        self.assertIn(request["reason"], decision["justification"])
        self.assertIn("adjuster must oversee", decision["justification"])
        with self.store.session() as session:
            [task] = session.scalars(select(HumanTask).where(HumanTask.assessment_id == assessment.id)).all()
            self.assertEqual(task.queue, "adjuster")
            self.assertEqual(task.status, "pending")

    def test_optional_gaps_neither_block_coverage_nor_appear_in_a_request(self):
        def with_optional_gaps(store, identifier, context, round_number, role, task):
            result = execute_specialist(store, identifier, context, round_number, role, task)
            if role == "intake":
                result["details"]["missing"].extend([
                    {"field": "damage_photos", "document": "Damage photos", "reason": "Could corroborate the incident."},
                    {"field": "date", "document": "Optional photograph", "reason": "A timestamp may help.", "required_for": None},
                ])
            return result

        with patch("backend.workflow.intake_first_graph.execute_specialist", side_effect=with_optional_gaps):
            assessed, decision = self.run_claim(0)
            self.assertEqual(decision["route"], Route.AUTO.value)
            self.assertIn("coverage", assessed.context["findings"])
            self.assertEqual(decision["information_required"], [])

            _, blocked = self.run_claim(6)
            self.assertEqual(blocked["route"], Route.INFO.value)
            self.assertEqual([item["document"] for item in blocked["information_required"]], ["Itemised repair estimate"])
            self.assertNotIn("Damage photos", blocked["justification"])
            self.assertNotIn("Optional photograph", blocked["justification"])

    def test_unsupported_loss_does_not_request_a_repair_estimate(self):
        identifier = self.batch["assessment_ids"][0]
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            claim.payload = {**claim.payload, "loss_type": "injury", "documents": []}
            claim.payload_hash = digest(claim.payload)
        assessment, decision = self.run_claim(0)
        self.assertEqual(decision["route"], Route.REVIEW.value)
        self.assertEqual(decision["information_required"], [])
        self.assertEqual(assessment.context["findings"]["intake"]["details"]["missing"], [])
        self.assertIn("Staff must assess which cost evidence", decision["justification"])

    def test_fraud_flags_create_fraud_queue_tasks(self):
        assessment, decision = self.run_claim(14)
        self.assertEqual(decision["route"], Route.FRAUD.value)
        with self.store.session() as session:
            task = session.scalar(select(HumanTask).where(HumanTask.assessment_id == assessment.id))
            self.assertEqual(task.queue, "fraud")
            self.assertEqual(task.status, "pending")

    def test_unexplained_additional_cost_requests_itemisation(self):
        assessment, decision = self.run_claim(13)
        self.assertEqual(decision["route"], Route.INFO.value)
        [gap] = assessment.context["findings"]["intake"]["details"]["missing"]
        self.assertEqual(gap["field"], "additional_costs")
        self.assertIn("1000.00", gap["reason"])

    def test_coverage_early_exit_never_queries_history(self):
        for index in [7, 8]:
            assessment, decision = self.run_claim(index)
            self.assertEqual(decision["route"], Route.REVIEW.value)
            self.assertNotIn("historical", assessment.context["findings"])
            self.assertIn("historical", assessment.context["skipped"])
            with self.store.session() as session:
                calls = session.scalars(select(ToolCall).where(ToolCall.assessment_id == assessment.id)).all()
            self.assertNotIn("retrieve_comparable_claims", [c.tool for c in calls])
            self.assertIn("not assessed", decision["justification"])

    def test_unreadable_loss_date_produces_precise_information_request(self):
        assessment, decision = self.run_claim(4)
        self.assertEqual(decision["route"], Route.INFO.value)
        gap = assessment.context["findings"]["intake"]["details"]["missing"][0]
        self.assertEqual(gap["field"], "loss_date")
        self.assertIn("incident date", gap["location"])
        self.assertNotIn("coverage", assessment.context["findings"])
        self.assertNotIn("historical", assessment.context["findings"])

    def test_duplicate_gets_fraud_check_before_review(self):
        assessment, decision = self.run_claim(12)
        self.assertIn("fraud_initial", assessment.context["findings"])
        self.assertEqual(decision["route"], Route.REVIEW.value)
        self.assertIn("POSSIBLE_DUPLICATE", decision["reason_codes"])

    def test_historical_amount_and_fraud_findings_reach_followup_context(self):
        assessment, decision = self.run_claim(14)
        findings = assessment.context["findings"]
        self.assertEqual(findings["historical"]["details"]["amount_justification"]["conclusion"], "SUPPORTED")
        self.assertTrue(findings["historical"]["details"]["fraud_investigation"]["leads"])
        self.assertEqual(decision["route"], Route.FRAUD.value)
        with self.store.session() as session:
            followup = session.scalar(select(AgentRun).where(AgentRun.assessment_id == assessment.id, AgentRun.stage == "r3:fraud"))
        self.assertIn("historical", followup.input_context["findings"])
        self.assertIn("coverage", followup.input_context["findings"])
        self.assertEqual(followup.input_version, 3)

    def test_cost_outlier_is_review_not_fraud(self):
        assessment, decision = self.run_claim(10)
        self.assertEqual(decision["route"], Route.REVIEW.value)
        self.assertNotIn("fraud_followup", assessment.context["findings"])

    def test_agents_have_only_their_approved_retrieval_tools(self):
        identifier = self.batch["assessment_ids"][0]
        intake_tools = build_tools(self.store, identifier, "intake")
        self.assertIn("find_duplicates", intake_tools)
        self.assertNotIn("retrieve_policy", intake_tools)
        history_tools = build_tools(self.store, identifier, "historical")
        self.assertIn("retrieve_comparable_claims", history_tools)
        self.assertNotIn("retrieve_comparable_claims", build_tools(self.store, identifier, "fraud_initial"))
        self.assertFalse(any("label" in name or "publish" in name for name in history_tools))

    def test_runtime_does_not_need_evaluation_labels(self):
        self.labels.rename(self.root / "hidden-labels.sqlite3")
        _, decision = self.run_claim(0)
        self.assertEqual(decision["route"], Route.AUTO.value)

    def test_publication_rejects_forged_approval(self):
        identifier = self.batch["assessment_ids"][5]
        with patch.object(self.workflow, "publish", side_effect=RuntimeError("pause before publication")):
            with self.assertRaises(RuntimeError):
                self.workflow.run(identifier)
        with self.store.session() as session:
            assessment = session.get(Assessment, identifier)
            claim = session.get(Claim, assessment.claim_id)
        forged = select_route(claim.payload, {**assessment.context, "version": assessment.version}, []).model_dump(mode="json")
        forged["route"] = Route.AUTO.value
        with self.assertRaisesRegex(ValueError, "mandatory routing"):
            self.workflow.publish(identifier, forged)
        with self.store.session() as session:
            self.assertIsNone(session.get(DecisionRecord, identifier))

    def test_retired_unfinished_workflows_cannot_execute(self):
        for version in ["legacy-v1", "claude-supervisor-v2", None]:
            identifier = self.batch["assessment_ids"][0]
            with self.store.session.begin() as session:
                row = session.get(Assessment, identifier)
                row.execution = {**row.execution, "workflow": version}
            with patch("backend.workflow.investigation_execution.execute_specialist") as specialist:
                with self.assertRaises(UnsupportedWorkflowVersion):
                    self.workflow.run(identifier)
                specialist.assert_not_called()
            with self.store.session() as session:
                self.assertEqual(session.get(Assessment, identifier).status, "queued")
                self.assertIsNone(session.get(DecisionRecord, identifier))

    def test_completed_retired_assessment_remains_readable(self):
        assessment, decision = self.run_claim(0)
        with self.store.session.begin() as session:
            row = session.get(Assessment, assessment.id)
            row.execution = {**row.execution, "workflow": "claude-supervisor-v2"}
        self.workflow.run(assessment.id)
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, assessment.id).payload, decision)

    def test_offline_claims_use_only_intake_first_graph(self):
        assessment, _ = self.run_claim(0)
        self.assertEqual(assessment.execution["workflow"], "claude-intake-first-v3")
        self.assertEqual(assessment.execution["mode"], "deterministic-tools")
        with self.store.session() as session:
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == assessment.id)).all()
        self.assertEqual({r.stage for r in runs}, {"r0:intake", "r1:coverage", "r1:fraud", "r2:historical"})

    def test_flags_from_any_finding_override_other_gates(self):
        assessment, _ = self.run_claim(0)
        with self.store.session() as session:
            claim = session.get(Claim, assessment.claim_id).payload
        for stage in ["intake", "coverage", "historical", "fraud_initial"]:
            context = copy.deepcopy(assessment.context)
            context["findings"]["intake"]["details"]["missing"] = [
                {"field": "loss_date", "document": "Claim form", "location": "Loss date",
                 "reason": "Needed to check the policy period."}
            ]
            context["findings"][stage]["details"]["fraud_indicators"] = [{"code": "TEST", "reason": "Independent invariant test"}]
            result = select_route(claim, context, [])
            self.assertEqual(result.route, Route.FRAUD)

    def test_stage_failure_fails_closed(self):
        with patch.object(coverage, "investigate", side_effect=RuntimeError("internal dependency")):
            assessment, decision = self.run_claim(0)
        self.assertIsNone(decision)
        self.assertEqual(assessment.status, "failed")
        self.assertEqual(assessment.context["findings"]["coverage"]["details"]["error_type"], "RuntimeError")

    def test_graph_recovers_from_checkpoint_without_duplicate_side_effects(self):
        identifier = self.batch["assessment_ids"][5]
        with patch.object(self.workflow, "publish", side_effect=RuntimeError("crash at publication")):
            with self.assertRaises(RuntimeError):
                self.workflow.run(identifier)
        self.workflow.close()
        self.workflow = ClaimsWorkflow(self.store, self.root / "checkpoints.sqlite3")
        self.workflow.run(identifier)
        self.workflow.run(identifier)
        with self.store.session() as session:
            tasks = session.scalars(select(HumanTask).where(HumanTask.assessment_id == identifier)).all()
            findings = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(findings), 4)

    def test_audit_reconstructs_without_database_and_detects_tampering(self):
        assessment, decision = self.run_claim(14)
        package = export_audit(self.store, assessment.id)
        serialized = json.dumps(package)
        self.assertTrue(verify_audit(json.loads(serialized)))
        self.assertEqual(package["decision"], decision)
        self.assertIn("workflow/claims_workflow.py", package["execution"]["sources"])
        self.assertTrue(any(r["tool"] == "retrieve_comparable_claims" for r in package["retrievals"]))
        package["submission"]["gross_amount"] = "1.00"
        self.assertFalse(verify_audit(package))

    def test_confusion_matrix_orientation_and_partial_state(self):
        self.run_claim(0)
        report = evaluate(self.store, self.labels)
        self.assertEqual(report["pending"], 15)
        self.assertFalse(report["complete"])
        with closing(sqlite3.connect(self.labels)) as connection, connection:
            connection.execute("UPDATE labels SET expected='adjuster review' WHERE assessment_id=?", (self.batch["assessment_ids"][0],))
        report = evaluate(self.store, self.labels)
        self.assertEqual(report["matrix"][1][0], 1)
        self.assertEqual(len(report["errors"]), 1)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="claims-api-")
        self.app = create_app(Path(self.tmp.name), auto_seed=False)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_health_empty_dashboard_and_invalid_generation(self):
        self.assertEqual(self.client.get("/api/health").json()["environment"], "synthetic")
        summary = self.client.get("/api/overview").json()
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["within_15_minutes"])
        self.assertIsNone(summary["cost_per_claim"])
        self.assertEqual(self.client.post("/api/generator/preview", json={"count": 999}).status_code, 422)
        self.assertEqual(self.client.get("/api/claims/not-found").status_code, 404)

    def test_retry_cannot_restart_a_retired_workflow(self):
        self.app.state.worker.close()
        batch = generate_batch(self.app.state.store, self.app.state.labels_path, 5, 1)
        identifier = batch['assessment_ids'][0]
        with self.app.state.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            row.status = 'failed'
            row.execution = {**row.execution, 'workflow': 'claude-supervisor-v2'}
        result = self.client.post(f'/api/claims/{identifier}/retry')
        self.assertEqual(result.status_code, 409)
        self.assertIn('retired', result.json()['detail'])
        with self.app.state.store.session() as session:
            self.assertEqual(session.get(Assessment, identifier).status, 'failed')

    def test_retry_preserves_failed_audit_and_routes_only_the_finished_attempt(self):
        self.app.state.worker.close()
        store = self.app.state.store
        batch = generate_batch(store, self.app.state.labels_path, 137, 16)
        previous = batch["assessment_ids"][5]
        workflow = self.app.state.worker.workflow
        with patch.object(coverage, "investigate", side_effect=RuntimeError("temporary failure")):
            workflow.run(previous)
        detail = self.client.get(f"/api/claims/{previous}").json()
        self.assertEqual(detail["status"], "failed")
        self.assertIsNone(detail["route"])
        self.assertIsNone(detail["task"])
        self.assertTrue(detail["can_reprocess"])
        self.assertEqual(detail["processing_alerts"], ["Claim exceeds LKR 250,000"])
        package = self.client.get(f"/api/claims/{previous}/audit").json()
        self.assertTrue(verify_audit(package))
        self.assertIsNone(package["decision"])
        self.assertEqual(evaluate(store, self.app.state.labels_path)["evaluated"], 0)

        with store.session.begin() as session:
            assessment = session.get(Assessment, previous)
            session.get(Claim, assessment.claim_id).received_at = "2020-01-01T00:00:00+00:00"
        response = self.client.post(f"/api/claims/{previous}/retry")
        self.assertEqual(response.status_code, 202, response.text)
        target = response.json()["id"]
        self.assertNotEqual(target, previous)
        self.assertEqual(self.client.post(f"/api/claims/{previous}/retry").json()["id"], target)
        self.assertEqual(self.client.get(f"/api/claims/{target}").json()["status"], "queued")
        self.assertEqual(len(self.client.get("/api/claims").json()["claims"]), 16)
        workflow.run(target)
        result = self.client.get(f"/api/claims/{target}").json()
        self.assertEqual(result["reference"], detail["reference"])
        self.assertEqual(result["decision"]["reason_codes"], ["GROSS_AUTHORITY"])
        self.assertEqual(result["submission"]["parent_assessment_id"], previous)
        self.assertEqual(self.client.get(f"/api/claims/{previous}/audit").json()["replacement_assessment_id"], target)
        self.assertEqual(evaluate(store, self.app.state.labels_path)["evaluated"], 1)

    def test_staff_retry_reuses_manual_submission_and_original_attachment(self):
        self.app.state.worker.close()
        self.client.post("/api/auth/login", json={"email": "user@ruhunu.local", "password": "UserDemo!2026"})
        content = base64.b64encode(b"\x89PNG\r\n\x1a\nfictional fixture").decode()
        body = {"policy_id": "TEST", "registration": "ABC-1000", "loss_type": "repair", "loss_date": "2026-10-15",
                "gross_amount": "100000.00", "narrative": "Synthetic windscreen claim.",
                "attachments": [{"name": "estimate.png", "media_type": "image/png", "content": content}]}
        previous = self.client.post("/api/portal/claims", json=body).json()["id"]
        store = self.app.state.store
        with patch("backend.evidence.uploaded_claim_documents.prepare_attachments",
                   return_value=[{"status": "unavailable", "technical_failure": True}]):
            self.app.state.worker.workflow.run(previous)
        self.assertEqual(self.client.post(f"/api/portal/claims/{previous}/retry").status_code, 403)
        [claim] = self.client.get("/api/portal/claims").json()
        self.assertEqual(claim["status"], "investigation failed")
        self.assertFalse(claim["can_reprocess"])
        self.client.post("/api/auth/login", json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"})
        response = self.client.post(f"/api/portal/claims/{previous}/retry")
        self.assertEqual(response.status_code, 202, response.text)
        [claim] = self.client.get("/api/portal/claims").json()
        self.assertEqual(claim["id"], response.json()["id"])
        self.assertEqual(claim["submission"]["parent_assessment_id"], previous)
        self.assertEqual(claim["submission"]["narrative"], body["narrative"])
        [attachment] = claim["attachments"]
        self.assertEqual(self.client.get(f"/api/portal/attachments/{attachment['id']}").content, base64.b64decode(content))
        with store.session() as session:
            self.assertIsNone(session.get(DecisionRecord, previous))
            self.assertNotIn("attachment_extraction", session.get(Assessment, claim["id"]).context)

    def test_legacy_failure_route_is_archived_and_claim_can_be_reprocessed(self):
        self.app.state.worker.close()
        store = self.app.state.store
        batch = generate_batch(store, self.app.state.labels_path, 137, 16)
        previous = batch["assessment_ids"][5]
        self.app.state.worker.workflow.run(previous)
        with store.session.begin() as session:
            row = session.get(Assessment, previous)
            row.context = {**row.context, "investigation_error": "Legacy timeout"}
        detail = self.client.get(f"/api/claims/{previous}").json()
        self.assertIsNone(detail["route"])
        self.assertTrue(detail["can_reprocess"])
        self.assertEqual(detail["error"], "Legacy timeout")
        self.assertEqual(self.client.get(f"/api/claims/{previous}/audit").json()["decision"]["route"], "adjuster review")
        self.assertEqual(self.client.post(f"/api/claims/{previous}/retry").status_code, 202)

    def test_roles_manual_submission_ownership_and_final_review(self):
        self.client.post('/api/auth/logout')
        self.assertEqual(self.client.get('/api/claims').status_code, 401)
        self.client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
        self.assertEqual(self.client.get('/api/overview').status_code, 403)
        self.assertEqual(self.client.post('/api/portal/accounts', json={'email':'bad@test.local','password':'LongPassword2026','role':'admin'}).status_code, 403)
        body = {'policy_id': 'POL-MANUAL', 'registration': 'ABC-1234', 'loss_type': 'glass', 'loss_date':'2026-10-15',
                'gross_amount':'300000.01', 'narrative':'The windscreen was damaged by a stone.', 'attachments':[]}
        result = self.client.post('/api/portal/claims', json=body)
        self.assertEqual(result.status_code, 201, result.text)
        identifier = result.json()['id']
        self.app.state.worker.close()
        self.app.state.worker.workflow.run(identifier)
        self.assertEqual(len(self.client.get('/api/portal/claims').json()), 1)
        review = {'outcome':'rejected','reason':'The submitted policy does not cover this loss.'}
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/review', json=review).status_code, 403)
        self.client.post('/api/auth/register', json={'email':'other@test.local','password':'AnotherPassword2026','role':'admin'})
        self.client.post('/api/auth/login', json={'email':'other@test.local','password':'AnotherPassword2026'})
        self.assertEqual(self.client.get('/api/auth/me').json()['role'], 'user')
        self.assertEqual(self.client.get('/api/portal/claims').json(), [])
        self.client.post('/api/auth/login', json={'email':'staff@ruhunu.local','password':'StaffDemo!2026'})
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/review', json={'outcome':'approved','reason':' ' * 12}).status_code, 422)
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/review', json=review).status_code, 200)
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/review', json=review).status_code, 409)
        self.client.post('/api/auth/login', json={'email':'user@ruhunu.local','password':'UserDemo!2026'})
        result = self.client.get('/api/portal/claims').json()[0]
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], review['reason'])
        self.assertIsNone(result['triage'])

    def test_acknowledgement_requires_existing_pending_task_and_identity(self):
        batch = generate_batch(self.app.state.store, self.app.state.labels_path, 5, 16)
        self.app.state.worker.wake.set()
        # A separate workflow on separate checkpoints is safe for this one queued item
        # only after stopping the worker; this keeps the API test deterministic.
        self.app.state.worker.close()
        workflow = self.app.state.worker.workflow
        workflow.run(batch["assessment_ids"][5])
        detail = self.client.get(f"/api/claims/{batch['assessment_ids'][5]}").json()
        task_id = detail["task"]["id"]
        self.assertEqual(self.client.post(f"/api/tasks/{task_id}/acknowledge", json={"reviewer": " "}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/tasks/{task_id}/acknowledge", json={"reviewer": "Test reviewer"}).status_code, 200)
        self.assertEqual(self.client.post(f"/api/tasks/{task_id}/acknowledge", json={"reviewer": "Test reviewer"}).status_code, 409)
        package = self.client.get(f"/api/claims/{batch['assessment_ids'][5]}/audit").json()
        self.assertEqual(package["human_actions"][0]["reviewer"], "Test reviewer")

    def test_uploaded_evidence_permissions_and_approval_audit(self):
        self.client.post('/api/auth/login', json={'email':'user@ruhunu.local','password':'UserDemo!2026'})
        content = b'%PDF-1.7\nSynthetic test fixture only'
        body = {'policy_id':'TEST-POL', 'registration':'ABC-1000', 'loss_type':'glass', 'loss_date':'2026-10-15',
                'gross_amount':'100000.00', 'narrative':'Synthetic manual windscreen loss for testing.',
                'attachments':[{'name':'evidence.pdf','media_type':'application/pdf','content':base64.b64encode(content).decode()}]}
        result = self.client.post('/api/portal/claims', json=body)
        self.assertEqual(result.status_code, 201, result.text)
        identifier = result.json()['id']
        self.app.state.worker.close()
        self.app.state.worker.workflow.run(identifier)
        item = self.client.get('/api/portal/claims').json()[0]
        self.assertEqual(item['status'], 'more information requested')
        self.assertIsNone(item['triage'])
        [request] = item['information_required']
        self.assertEqual(request['document'], 'Itemised repair estimate')
        self.assertIn('assess the claimed repair cost', request['reason'])
        attachment = item['attachments'][0]['id']
        self.assertEqual(self.client.get(f'/api/portal/attachments/{attachment}').content, content)
        self.client.post('/api/auth/register', json={'email':'stranger@test.local','password':'StrangerPassword2026'})
        self.client.post('/api/auth/login', json={'email':'stranger@test.local','password':'StrangerPassword2026'})
        self.assertEqual(self.client.get(f'/api/portal/attachments/{attachment}').status_code, 404)
        self.client.post('/api/auth/login', json={'email':'staff@ruhunu.local','password':'StaffDemo!2026'})
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/review', json={'outcome':'approved','reason':'The submitted evidence supports this claim.'}).status_code, 200)
        self.client.post('/api/auth/login', json={'email':'admin@ruhunu.local','password':'AdminDemo!2026'})
        audit = self.client.get(f'/api/claims/{identifier}/audit').json()
        self.assertEqual(audit['staff_decision']['outcome'], 'approved')
        self.assertEqual(audit['decision']['route'], 'request more information')
        self.assertTrue(verify_audit(audit))


if __name__ == "__main__":
    unittest.main()
