"""Acceptance tests for the new Python / React project's backend slice."""
import copy
import base64
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from sqlalchemy import select

from backend.agents import coverage_specialist as coverage
from backend.api import create_app
from backend.storage.sqlite_store import Adjudication, AgentRun, Assessment, Claim, ClaimArchive, DecisionRecord, HumanTask, ModelInvocation, PolicyChunk, PolicyDocument, Store, ToolCall, digest
from scripts.claim_fixtures import generate_batch
from backend.claim_schemas import Route
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.routing_evaluation import evaluate
from backend.workflow.investigation_execution import execute_specialist
from backend.workflow.routing_rules import select_route
from backend.evidence import policy_wording_index as wording
from backend.evidence.specialist_evidence_tools import build_tools
from backend.workflow.claims_workflow import ClaimsWorkflow, UnsupportedWorkflowVersion
from backend.workflow.selected_policy import generate_policy_schedule, queue_with_policy
from legacy_claim_fixture import seed_legacy_claim


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
        self.run_claim(1)
        report = evaluate(self.store, self.labels)
        self.assertEqual(report["matrix"], [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["class_metrics"], [
            {"route": "auto_approval", "support": 1, "precision": 0.5, "recall": 1.0, "f1": 2 / 3},
            {"route": "adjuster_review", "support": 1, "precision": None, "recall": 0.0, "f1": 0.0},
            {"route": "need_information", "support": 0, "precision": None, "recall": None, "f1": None},
            {"route": "fraud_escalation", "support": 0, "precision": None, "recall": None, "f1": None},
        ])


class ApiTests(unittest.TestCase):
    def test_specialist_inputs_show_recorded_requests_and_offline_checks_to_staff(self):
        self.app.state.worker.close()
        identifier, _, _ = seed_legacy_claim(self.app.state.store, {
            "policy_id": "TEST-POL", "registration": "ABC-1000", "loss_type": "glass", "loss_date": "2026-10-15",
            "gross_amount": "100000.00", "narrative": "Synthetic claim for inspecting specialist input.", "attachments": []})
        self.app.state.worker.workflow.run(identifier)
        request = {"model": "fixture", "messages": [{"role": "user", "content": '{"context":{"findings":{"intake":{}}}}'}]}
        with self.app.state.store.session.begin() as session:
            session.add_all([
                ModelInvocation(id="first", assessment_id=identifier, stage="r1:coverage", status="failed", payload=request),
                ModelInvocation(id="second", assessment_id=identifier, stage="r1:coverage", status="complete", payload=request),
                ModelInvocation(id="supervisor", assessment_id=identifier, stage="r1:supervisor", status="complete", payload=request),
            ])
        path = f"/api/claims/{identifier}/specialist-inputs"
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        stages = {stage["stage"]: stage for stage in response.json()["stages"]}
        self.assertNotIn("r1:supervisor", stages)
        self.assertTrue(stages["r0:intake"]["checks"])
        self.assertEqual(stages["r0:intake"]["requests"], [])
        self.assertEqual(stages["r1:coverage"]["requests"], [
            {"id": "first", "status": "failed", "request": request},
            {"id": "second", "status": "complete", "request": request},
        ])
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})
        self.assertEqual(self.client.get(path).status_code, 403)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="claims-api-")
        self.app = create_app(Path(self.tmp.name), auto_seed=False)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.client.post("/api/auth/login", json={"email": "staff@ruhunu.local", "password": "StaffDemo!2026"})

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_routing_evaluation_requires_admin_and_handles_empty_workspace(self):
        path = "/api/portal/evaluation"
        self.assertEqual(self.client.get(path).status_code, 403)
        self.assertEqual(self.client.get("/api/evaluation").status_code, 200)
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.get(path).status_code, 401)
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["matrix"], [[0] * 4 for _ in range(4)])
        self.assertEqual((report["evaluated"], report["pending"], report["mismatches"]), (0, 0, 0))
        self.assertEqual([row["f1"] for row in report["class_metrics"]], [None] * 4)

    def test_account_creation_is_admin_only_and_creates_staff_only(self):
        path = "/api/portal/accounts"
        body = {"email": "new-staff@test.local", "password": "LongPassword2026"}
        self.assertEqual(self.client.post(path, json=body).status_code, 403)
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})
        self.assertEqual(self.client.post(path, json={**body, "role": "admin"}).status_code, 422)
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["role"], "staff")

    def test_health_empty_dashboard_and_invalid_generation(self):
        self.assertEqual(self.client.get("/api/health").json()["environment"], "synthetic")
        summary = self.client.get("/api/overview").json()
        self.assertEqual(summary["total"], 0)
        self.assertIsNone(summary["within_15_minutes"])
        self.assertIsNone(summary["cost_per_claim"])
        self.assertEqual(self.client.post("/api/generator/preview", json={"count": 999}).status_code, 422)
        self.assertEqual(self.client.get("/api/claims/not-found").status_code, 404)

    def test_staff_can_upload_preview_and_reindex_policy_pdf(self):
        self.app.state.worker.close()
        pdf = PdfWriter()
        pdf.add_blank_page(width=612, height=792)
        buffer = BytesIO()
        pdf.write(buffer)
        content = buffer.getvalue()
        path = "/api/policy-documents"
        body = {
            "title": "Motor wording", "product": "MOTOR", "version": "2026.1",
            "filename": "motor.pdf", "media_type": "application/pdf", "content": base64.b64encode(content).decode(),
        }

        uploaded = self.client.post(path, json=body)
        self.assertEqual(uploaded.status_code, 202, uploaded.text)
        identifier = uploaded.json()["id"]
        [document] = self.client.get(path).json()["documents"]
        self.assertEqual((document["id"], document["status"]), (identifier, "queued"))
        self.assertEqual(self.client.get(f"{path}?selectable=true").json()["documents"], [])
        preview_path = f"{path}/{identifier}/preview"
        preview = self.client.get(preview_path)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["Content-Type"], "application/pdf")
        self.assertEqual(preview.headers["Content-Disposition"], "inline")
        self.assertEqual(preview.content, content)
        self.assertEqual(self.client.get(f"{path}/{identifier}/original").content, content)
        retry_path = f"{path}/{identifier}/retry"
        self.assertEqual(self.client.post(retry_path).status_code, 409)
        with self.app.state.store.session.begin() as session:
            session.get(PolicyDocument, identifier).status = "ready"
        [outdated] = self.client.get(path).json()["documents"]
        self.assertTrue(outdated["requires_reindex"])
        self.assertEqual(self.client.post(retry_path).status_code, 202)
        self.assertEqual(self.client.get(f"{path}/{identifier}").json()["status"], "queued")
        self.assertEqual(self.client.put("/api/policies/test/wording", json={"document_id": identifier}).status_code, 403)

        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.post(path, json=body).status_code, 401)
        self.assertEqual(self.client.get(preview_path).status_code, 401)
        self.assertEqual(self.client.post(retry_path).status_code, 401)

    def test_staff_can_remove_policy_from_picker_preserving_existing_claim_links(self):
        self.app.state.worker.close()
        store = self.app.state.store
        content = b"%PDF-1.7\nSynthetic wording fixture"
        identifier = wording.create_document(store, title="Motor wording", product="MOTOR", version="1",
                                             filename="motor.pdf", media_type="application/pdf",
                                             content=base64.b64encode(content).decode())
        first, second = generate_batch(store, self.app.state.labels_path, count=2)["assessment_ids"]
        with store.session.begin() as session:
            document = session.get(PolicyDocument, identifier)
            document.status = "ready"
            document.embedding = {"model": wording.MODEL, "dimension": wording.DIMENSIONS,
                                  "storage": wording.STORAGE, "chunk_format": wording.CHUNK_FORMAT}
            session.add(PolicyChunk(id=f"{identifier}:{wording.CHUNK_FORMAT}:0", document_id=identifier, page=1,
                                    text="Comprehensive cover includes collision damage."))
            for assessment_id in (first, second):
                session.get(Assessment, assessment_id).status = "awaiting_policy"
        for assessment_id in (first, second):
            generate_policy_schedule(store, assessment_id, identifier, "test")
        self.assertEqual(queue_with_policy(store, first, identifier, "test"), "queued")

        path = f"/api/policy-documents/{identifier}"
        self.assertEqual(self.client.delete(path).status_code, 204)
        self.assertEqual(self.client.delete(path).status_code, 204)
        for listing in ("/api/policy-documents", "/api/policy-documents?selectable=true"):
            self.assertEqual(self.client.get(listing).json()["documents"], [])
        with self.assertRaisesRegex(ValueError, "current OpenAI passage index"):
            queue_with_policy(store, second, identifier, "test")
        self.assertEqual(self.client.get(f"/api/portal/claims/{first}").json()["selected_policy"]["id"], identifier)
        self.assertEqual(self.client.get(f"{path}/preview").content, content)
        self.client.post("/api/auth/logout")
        self.assertEqual(self.client.delete(path).status_code, 401)

    def test_archiving_claims_clears_active_lists_and_preserves_audits(self):
        self.app.state.worker.close()
        store = self.app.state.store
        [identifier] = generate_batch(store, self.app.state.labels_path, count=1)["assessment_ids"]
        with store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            assessment.status = "failed"
            session.add(ClaimArchive(claim_id=assessment.claim_id))

        self.assertEqual(self.client.get("/api/portal/claims?summary=true").json(), [])
        self.assertEqual(self.client.get("/api/claims").json()["claims"], [])
        path = f"/api/portal/claims/{identifier}"
        detail = self.client.get(path).json()
        self.assertTrue(detail["archived"])
        self.assertFalse(detail["can_reprocess"])
        self.assertEqual(self.client.post(f"{path}/retry").status_code, 409)
        self.assertTrue(verify_audit(self.client.get(f"/api/claims/{identifier}/audit").json()))

        [new_id] = generate_batch(store, self.app.state.labels_path, count=1)["assessment_ids"]
        [active] = self.client.get("/api/portal/claims?summary=true").json()
        self.assertEqual(active["id"], new_id)

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
        portal = self.client.get(f"/api/portal/claims/{previous}").json()
        self.assertIsNone(portal["adjuster_review"])
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

    def test_adjuster_summary_displays_only_priorities_selected_by_the_model(self):
        self.app.state.worker.close()
        store = self.app.state.store
        claims = generate_batch(store, self.app.state.labels_path, 137, 16)["assessment_ids"]
        identifier = claims[5]
        issue = "The quotation lists an extra panel that the incident account does not mention."

        def checks(*args):
            finding = execute_specialist(*args)
            if args[4] == "coverage":
                finding["details"]["document_inconsistencies"] = [{"reason": issue, "document_refs": ["submission:/documents/0"]}]
                finding["details"]["submitted_documents"] = {"submission:/documents/0": "Repair quotation"}
            return finding

        recommendation = {"route": Route.REVIEW.value, "explanation": "Gross amount exceeds authority.",
                          "evidence": [], "outstanding_gaps": [], "adjuster_summary": [issue]}
        with patch("backend.workflow.intake_first_graph.execute_specialist", side_effect=checks), patch(
                "backend.workflow.intake_first_graph.recommend_route", return_value=recommendation):
            self.app.state.worker.workflow.run(identifier)
        response = self.client.get(f"/api/portal/claims/{identifier}")
        self.assertEqual(response.status_code, 200, response.text)
        detail = response.json()
        self.assertEqual(detail["triage"]["reason_codes"], ["GROSS_AUTHORITY"])
        self.assertEqual(detail["adjuster_review"], issue)
        self.assertNotIn("adjuster_summary", detail["triage"]["recommendation"])
        self.app.state.worker.workflow.run(claims[7])
        below_threshold = self.client.get(f"/api/portal/claims/{claims[7]}").json()
        self.assertEqual(below_threshold["triage"]["route"], Route.REVIEW.value)
        self.assertEqual(below_threshold["status"], Route.REVIEW.value)
        self.assertTrue(below_threshold["adjuster_review"])
        self.app.state.worker.workflow.run(claims[2])
        self.assertIsNone(self.client.get(f"/api/portal/claims/{claims[2]}").json()["adjuster_review"])

    def test_failed_legacy_manual_case_preserves_attachments_and_audit(self):
        self.app.state.worker.close()
        store = self.app.state.store
        content = base64.b64encode(b"\x89PNG\r\n\x1a\nfictional fixture").decode()
        body = {"policy_id": "TEST", "registration": "ABC-1000", "loss_type": "repair", "loss_date": "2026-10-15",
                "gross_amount": "100000.00", "narrative": "Synthetic windscreen claim.",
                "attachments": [{"name": "estimate.png", "media_type": "image/png", "content": content}]}
        identifier, _, _ = seed_legacy_claim(store, body)
        with patch("backend.evidence.uploaded_claim_documents.prepare_attachments",
                   return_value=[{"status": "unavailable", "technical_failure": True}]):
            self.app.state.worker.workflow.run(identifier)
        claim = self.client.get(f"/api/portal/claims/{identifier}").json()
        self.assertEqual(claim["status"], "investigation failed")
        self.assertFalse(claim["can_reprocess"])
        self.assertEqual(self.client.post(f"/api/portal/claims/{identifier}/retry").status_code, 409)
        self.assertEqual(self.client.post(f"/api/claims/{identifier}/retry").status_code, 409)
        [attachment] = claim["attachments"]
        self.assertEqual(self.client.get(f"/api/portal/attachments/{attachment['id']}").content, base64.b64decode(content))
        report = self.client.get(f"/api/portal/claims/{identifier}/investigation.pdf")
        self.assertEqual(report.status_code, 200)
        self.assertTrue(report.content.startswith(b"%PDF-"))
        review = {"outcome": "rejected", "reason": "Staff reviewed the archived evidence directly."}
        self.assertEqual(self.client.post(f"/api/portal/claims/{identifier}/review", json=review).status_code, 404)
        self.assertTrue(verify_audit(self.client.get(f"/api/claims/{identifier}/audit").json()))

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

    def test_customer_accounts_and_sessions_are_disabled(self):
        self.app.state.worker.close()
        body = {"policy_id": "TEST", "registration": "ABC-1000", "loss_type": "repair", "loss_date": "2026-10-15",
                "gross_amount": "100000.00", "narrative": "Synthetic windscreen claim."}
        _, old_session, email = seed_legacy_claim(self.app.state.store, body)
        self.client.post("/api/auth/logout")
        self.client.cookies.set("claims_session", old_session)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(self.client.get("/api/portal/claims").status_code, 401)
        self.assertEqual(self.client.post("/api/auth/login", json={"email": email, "password": "LegacyCustomer!2026"}).status_code, 401)
        self.client.cookies.clear()
        self.client.post("/api/auth/login", json={"email": "admin@ruhunu.local", "password": "AdminDemo!2026"})
        self.assertEqual(self.client.post("/api/auth/register", json={"email": "new@test.local", "password": "LongPassword2026"}).status_code, 404)
        self.assertEqual(self.client.post("/api/portal/accounts", json={"email": "new@test.local", "password": "LongPassword2026", "role": "user"}).status_code, 422)
        self.assertEqual({a["role"] for a in self.client.get("/api/portal/accounts").json()}, {"staff", "admin"})
        self.assertEqual(self.client.post("/api/portal/claims", json=body).status_code, 405)

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

    def test_archived_evidence_permissions_and_historical_approval_audit(self):
        content = b'%PDF-1.7\nSynthetic test fixture only'
        body = {'policy_id':'TEST-POL', 'registration':'ABC-1000', 'loss_type':'glass', 'loss_date':'2026-10-15',
                'gross_amount':'100000.00', 'narrative':'Synthetic manual windscreen loss for testing.',
                'attachments':[{'name':'evidence.pdf','media_type':'application/pdf','content':base64.b64encode(content).decode()}]}
        self.app.state.worker.close()
        identifier, _, _ = seed_legacy_claim(self.app.state.store, body)
        self.app.state.worker.workflow.run(identifier)
        item = self.client.get('/api/portal/claims').json()[0]
        self.assertEqual(item['status'], 'request more information')
        self.assertEqual(item['triage']['route'], 'request more information')
        [request] = item['information_required']
        self.assertEqual(request['document'], 'Itemised repair estimate')
        self.assertIn('assess the claimed repair cost', request['reason'])
        attachment = item['attachments'][0]['id']
        self.assertEqual(self.client.get(f'/api/portal/attachments/{attachment}').content, content)
        self.assertEqual(self.client.get(f'/api/portal/claims/{identifier}').json()['reference'], item['reference'])
        self.assertEqual(self.client.get(f'/api/portal/attachments/{attachment}').content, content)
        self.assertEqual(self.client.get(f'/api/portal/claims/{identifier}').status_code, 200)
        reviewer = self.client.get('/api/auth/me').json()['id']
        with self.app.state.store.session.begin() as session:
            session.add(Adjudication(assessment_id=identifier, reviewer_id=reviewer, outcome='approved',
                                     reason='Historical staff approval preserved in the audit.'))
        detail = self.client.get(f'/api/portal/claims/{identifier}').json()
        self.assertEqual(detail['status'], 'request more information')
        self.assertEqual(detail['information_required'], item['information_required'])
        audit = self.client.get(f'/api/claims/{identifier}/audit').json()
        self.assertEqual(audit['staff_decision']['outcome'], 'approved')
        self.assertEqual(audit['decision']['route'], 'request more information')
        self.assertTrue(verify_audit(audit))
        self.client.post('/api/auth/login', json={'email':'admin@ruhunu.local','password':'AdminDemo!2026'})
        for path in ('/api/portal/claims', f'/api/portal/claims/{identifier}', f'/api/portal/attachments/{attachment}',
                     '/api/claims', f'/api/claims/{identifier}'):
            self.assertEqual(self.client.get(path).status_code, 403, path)
        self.assertEqual(self.client.post('/api/portal/claims', json=body).status_code, 405)
        self.assertEqual(self.client.post(f'/api/claims/{identifier}/process', json={'policy_document_id': 'pdf-1'}).status_code, 403)
        self.assertEqual(self.client.post('/api/generator/preview', json={'policy_id': 'TEST-POL', 'count': 1}).status_code, 403)
        self.assertEqual(self.client.get('/api/portal/accounts').status_code, 200)
        self.client.post('/api/auth/login', json={'email':'staff@ruhunu.local','password':'StaffDemo!2026'})
        self.assertEqual(self.client.post('/api/portal/claims', json=body).status_code, 405)
        self.assertEqual(self.client.post(f'/api/portal/claims/{identifier}/investigate').status_code, 404)


if __name__ == "__main__":
    unittest.main()
