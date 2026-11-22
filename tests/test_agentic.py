import base64
from collections import Counter
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import httpx
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import select

from backend.storage.sqlite_store import AgentRun, Assessment, Attachment, AttachmentExtraction, Claim, DecisionRecord, HumanTask, ModelInvocation, Policy, PolicyBinding, PolicyDocument, Store, ToolCall, digest
from scripts.claim_fixtures import generate_batch
from scripts.debug_workflow import scripted_analysis_fields
from backend.agents import claude_investigation as agentic
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.investigation_display import display_finding
from backend.evidence.policy_wording_index import create_document, index_document, search_document, search_uploaded_documents, uploaded_document_ids, ensure_synthetic_default
from backend.workflow.routing_rules import select_route
from backend.workflow.fraud_route_correction import escalate_published_high_severity
from backend.workflow.claims_workflow import ClaimsWorkflow


class Array(list):
    def tolist(self):
        return list(self)


class FakeEncoder:
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split(), decode=lambda tokens, **kwargs: " ".join(tokens))
    def __getitem__(self, index):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision")))
    def encode(self, text, **kwargs):
        return Array([[1.0, 0.0] for _ in text]) if isinstance(text, list) else Array([1.0, 0.0])


class AgenticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path('.local/test-tmp'))
        self.root = Path(self.tmp.name)
        self.store = Store(self.root / 'claims.sqlite3')
        with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'deterministic'}):
            self.batch = generate_batch(self.store, self.root / 'labels.sqlite3', 137, 16)
        self.workflow = ClaimsWorkflow(self.store, self.root / 'checkpoints.sqlite3')
        self.encoder_patch = patch('backend.evidence.policy_wording_index.encoder', return_value=FakeEncoder())
        self.encoder_patch.start()
        self.document = self.upload('Only the bound policy glass wording. Deductible follows schedule.')
        index_document(self.store, self.document)
        self.selections = []

    def tearDown(self):
        self.encoder_patch.stop()
        self.workflow.close()
        self.store.engine.dispose()
        self.tmp.cleanup()

    def upload(self, text):
        return create_document(self.store, title='Test wording', product='MOTOR', version='1', filename='wording.txt',
            media_type='text/plain', content=base64.b64encode(text.encode()).decode())

    def prepare(self, index=0, *, manual=False):
        identifier = self.batch['assessment_ids'][index]
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            row.execution = {**row.execution, 'workflow': 'claude-intake-first-v3', 'mode': 'anthropic-tools'}
            row.context = {**row.context, 'workflow': 'claude-intake-first-v3', 'policy_search_mode': 'bound-legacy', 'wording_document_id': self.document}
            if manual:
                claim = session.get(Claim, row.claim_id)
                estimate = claim.payload['documents'][0]
                claim.payload = {**claim.payload, 'source': 'Manual submission', 'documents': []}
                claim.payload_hash = digest(claim.payload)
                row.context = {**row.context, 'environment': 'manual', 'policy_search_mode': 'uploaded-corpus',
                               'policy_corpus_ids': [self.document]}
                row.context = {key: value for key, value in row.context.items() if key != 'wording_document_id'}
                attachment_id = f'{identifier}-estimate'
                session.add(Attachment(id=attachment_id, assessment_id=identifier, name=estimate['name'],
                                       media_type='application/pdf', content=''))
                session.flush()
                session.add(AttachmentExtraction(attachment_id=attachment_id, payload={
                    'attachment_id': attachment_id, 'status': 'extracted', 'document_type': 'repair_estimate',
                    'name': estimate['name'], 'registration': estimate['registration'], 'document_date': estimate['date'],
                    'garage_name': estimate['garage_name'], 'garage_address': estimate['garage_address'],
                    'total': estimate['total'], 'line_items': estimate['line_items'],
                    'provenance': 'Claimant upload', 'source': f'attachment:{attachment_id}',
                }))
        return identifier

    def provider(self, store, identifier, stage, system, messages, tools, choice):
        if stage.startswith('readiness:'):
            from scripts.debug_workflow import scripted_readiness
            name, value = 'assess_readiness', scripted_readiness(json.loads(messages[0]['content']))
            self.selections.append(value['tasks'])
        elif stage.startswith('extract:'):
            name, value = 'extract_document', {'summary': 'Estimate for windscreen replacement', 'document_type': 'repair_estimate',
                'garage_name': 'Sample Auto Glass', 'garage_address': '25 Sample Road, Colombo',
                'document_reference': 'EST-MANUAL', 'registration': 'ABC-1234', 'document_date': '2026-10-16',
                'total': '100000.00', 'line_items': [{'description': 'Windscreen', 'amount': '100000.00'}], 'gaps': []}
        elif stage == 'judge':
            context = json.loads(messages[0]['content'])
            refs = sorted({source for f in context['findings'].values() for source in f['details'].get('sources', [])})[:5]
            name, value = 'recommend_route', {'route': 'auto-approve', 'explanation': 'Recommendation for guard testing', 'evidence': refs, 'outstanding_gaps': []}
        else:
            if len(messages) == 1 and any(tool['name'] == 'search_policy_documents' for tool in tools):
                return {'stop_reason': 'tool_use', 'content': [{'type': 'tool_use', 'id': 'search',
                    'name': 'search_policy_documents', 'input': {'query': 'glass damage coverage exclusions'}}]}
            model_input = json.loads(messages[0]['content'])
            outputs = [model_input['required_checks']]
            if len(messages) > 1:
                outputs += [json.loads(item['content']) for item in messages[-1]['content']]
            refs = [item['evidence_ref'] for item in outputs]
            refs += [p['source'] for item in outputs for p in item['result'].get('passages', [])]
            name, value = 'submit_analysis', {'summary': f"Checks and {len(outputs[0]['supporting_evidence'])} supporting responses assessed",
                'review_reasons': [], 'evidence': refs, 'policy_applicability': 'applicable',
                **scripted_analysis_fields(stage, model_input)}
        return {'stop_reason': 'tool_use', 'content': [{'type': 'tool_use', 'id': 'submit', 'name': name, 'input': value}]}

    def run_claim(self, index=0):
        identifier = self.prepare(index)
        with patch('backend.agents.claude_investigation.call_model', side_effect=self.provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            return identifier, session.get(DecisionRecord, identifier).payload

    def test_parallel_round_and_citations_publish_and_recover(self):
        identifier, decision = self.run_claim()
        self.assertEqual(decision['route'], 'auto-approve')
        self.workflow.run(identifier)
        with self.store.session() as session:
            runs = session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()
            self.assertEqual(len(runs), 4)
            self.assertEqual({r.input_version for r in runs}, {0, 1, 2})
        package = export_audit(self.store, identifier)
        self.assertTrue(verify_audit(package))
        self.assertIsNone(package['policy_wording'])
        self.assertEqual(package['policy_documents'], [])
        self.assertNotIn('chunk:', json.dumps(package['context']))
        self.assertEqual(package['context']['policy_search_mode'], 'generated-policy')
        coverage = package['context']['findings']['coverage']
        self.assertEqual(coverage['details']['coverage_status'], 'COVERED')
        self.assertTrue(any(ref.startswith('tool:') for ref in coverage['evidence']))

    def test_historical_leads_trigger_second_round_fraud(self):
        identifier, decision = self.run_claim(14)
        self.assertEqual(decision['route'], 'fraud escalation')
        self.assertIsNotNone(decision['override_reason'])
        self.assertEqual(self.selections[2][0]['specialist'], 'fraud')

    def test_checks_precede_model_and_specialists_skip_the_check_request_turn(self):
        identifier = self.prepare()
        def provider(*args):
            stage, messages, definitions = args[2], args[4], args[5]
            if stage.startswith('r') and ':' in stage and not stage.startswith('readiness:'):
                supplied = json.loads(messages[0]['content'])['required_checks']
                self.assertTrue(supplied['result']['details']['checks_executed'])
                self.assertNotIn('run_required_checks', [t['name'] for t in definitions])
                self.assertNotIn('execute_checks', [t['name'] for t in definitions])
                with self.store.session() as session:
                    saved = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier,
                        ToolCall.stage == stage, ToolCall.tool == 'execute_checks')).all()
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0].response, supplied)
                schema = next(t['input_schema'] for t in definitions if t['name'] == 'submit_analysis')
                self.assertIn(supplied['evidence_ref'], schema['properties']['evidence']['items']['enum'])
            return self.provider(*args)
        with patch.object(agentic, 'execute_checks', wraps=agentic.execute_checks) as checks, \
             patch.object(agentic, 'call_model', side_effect=provider) as model:
            self.workflow.run(identifier)
        self.assertEqual(Counter(c.args[0] for c in checks.call_args_list),
                         Counter({'intake': 1, 'coverage': 1, 'fraud': 1, 'historical': 1}))
        stages = Counter(c.args[2] for c in model.call_args_list if c.args[2].startswith(('r0:', 'r1:', 'r2:')))
        self.assertEqual(stages, Counter({'r1:coverage': 1, 'r1:fraud': 1, 'r2:historical': 1}))
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'auto-approve')

    def test_failed_required_checks_never_call_model(self):
        identifier = self.prepare()
        with patch.object(agentic, 'execute_checks', side_effect=RuntimeError('checks unavailable')), \
             patch.object(agentic, 'call_model') as model:
            with self.assertRaisesRegex(RuntimeError, 'checks unavailable'):
                agentic.run_specialist(self.store, identifier, {}, 0, 'intake', 'Check claim')
        model.assert_not_called()
        with self.store.session() as session:
            self.assertIsNone(session.scalar(select(ToolCall.id).where(ToolCall.assessment_id == identifier,
                ToolCall.tool == 'execute_checks')))

    def test_high_gross_overrides_judge_approval(self):
        _, decision = self.run_claim(5)
        self.assertEqual(decision['route'], 'adjuster review')
        self.assertIn('GROSS_AUTHORITY', decision['reason_codes'])

    def test_loss_classification_and_review_belong_to_coverage(self):
        identifier = self.prepare()
        def provider(*args):
            response = self.provider(*args)
            if args[2] == 'r1:coverage':
                data = json.loads(args[4][0]['content'])
                self.assertIn('windscreen', data['submission']['narrative'])
                self.assertNotIn('classified_loss', data['required_checks']['result']['details'])
                response['content'][0]['input']['classified_loss'] = 'uncertain'
                response['content'][0]['input']['review_reasons'] = ['Ask the garage to clarify the date on the estimate.']
            return response
        with patch.object(agentic, 'call_model', side_effect=provider) as model:
            self.workflow.run(identifier)
        self.assertFalse(any(call.args[2].startswith('r0:intake') for call in model.call_args_list))
        with self.store.session() as session:
            findings = session.get(Assessment, identifier).context['findings']
            intake, coverage = findings['intake'], findings['coverage']['details']
            self.assertNotIn('analysis', intake['details'])
            self.assertNotIn('review_reasons', intake['details'])
            self.assertIn('policy_id', {fact['field'] for fact in intake['details']['facts']})
            self.assertEqual(intake['details']['missing'], [])
            self.assertEqual(coverage['analysis']['classified_loss'], 'uncertain')
            self.assertEqual(coverage['coverage_status'], 'UNRESOLVED')
            self.assertIn('Ask the garage to clarify the date on the estimate.', coverage['review_reasons'])
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'adjuster review')

    def test_blocked_intake_reports_gaps_without_general_model_review(self):
        identifier = self.prepare(3)
        with patch.object(agentic, 'call_model') as model:
            finding = agentic.run_specialist(self.store, identifier, {}, 0, 'intake', 'Assess coverage readiness')
        model.assert_not_called()
        self.assertFalse(finding['details']['coverage_readiness']['ready'])
        self.assertNotIn('analysis', finding['details'])
        self.assertEqual(finding['status'], 'information')

    def test_coverage_checks_police_report_and_other_documents_against_exclusion(self):
        from backend.evidence.required_claim_evidence import DRUNK_DRIVING_EXCLUSION

        identifier = self.prepare()
        document = self.upload(DRUNK_DRIVING_EXCLUSION)
        index_document(self.store, document)
        police_ref, witness_ref, estimate_ref = [f"submission:/documents/{index}" for index in (1, 2, 0)]
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            claim = session.get(Claim, row.claim_id)
            claim.payload = {**claim.payload, "loss_type": "repair", "documents": [*claim.payload["documents"],
                {"type": "police_report", "name": "Police report.json", "summary": "Positive test for the driver of the insured vehicle."},
                {"type": "witness_statement", "name": "Witness statement.json", "summary": "The bumper had been repaired before the crash."}]}
            claim.payload_hash = digest(claim.payload)

        def provider(*args):
            response = self.provider(*args)
            block = response["content"][0]
            if block["name"] == "search_policy_documents":
                initial = json.loads(args[4][0]["content"])
                self.assertEqual(set(initial["submitted_document_refs"]), {police_ref, witness_ref, estimate_ref})
                self.assertIn("Cross-check every submitted document", args[3])
                block["input"]["query"] = "drunk driving repair exclusion police report"
            else:
                passage = f"chunk:{document}:0"
                block["input"].update(coverage_conclusion="NOT_COVERED", applicable_passages=[passage], policy_conflicts=[],
                    coverage_items=[{"item": "Repair damage", "outcome": "not_covered", "policy_term": DRUNK_DRIVING_EXCLUSION,
                        "reason": "The police recorded a positive alcohol test for the driver of the insured vehicle.",
                        "basis": "wording", "passage_refs": [passage], "document_refs": [police_ref]}],
                    document_inconsistencies=[{"reason": "The witness says the bumper was repaired before the incident. The estimate attributes the work to this claim.",
                        "document_refs": [witness_ref, estimate_ref]}])
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            finding = agentic.run_specialist(self.store, identifier, {"synthetic_wording_document_id": document}, 1, "coverage", "Check every document")
        details = finding["details"]
        self.assertEqual(details["coverage_status"], "NOT_COVERED")
        self.assertEqual(details["coverage_items"][0]["document_refs"], [police_ref])
        self.assertEqual(details["document_inconsistencies"][0]["document_refs"], [witness_ref, estimate_ref])
        self.assertEqual(len(details["submitted_documents"]), 3)

    def test_model_policy_exclusion_stops_history_after_valid_date_preflight(self):
        identifier = self.prepare()
        def provider(*args):
            response = self.provider(*args)
            if args[2].endswith(':coverage') and response['content'][0]['name'] == 'submit_analysis':
                details = json.loads(args[4][0]['content'])['required_checks']['result']['details']
                self.assertEqual(details['coverage_status'], 'PENDING_ANALYSIS')
                self.assertNotIn('financial', details)
                response['content'][0]['input'].update(coverage_conclusion='NOT_COVERED', summary='The applicable wording excludes this use.',
                    coverage_items=[{'item': 'Reported loss', 'outcome': 'not_covered', 'policy_term': 'Commercial use is excluded.',
                                     'reason': 'The vehicle was being used commercially.'}])
            return response
        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            findings = session.get(Assessment, identifier).context['findings']
            self.assertEqual(findings['coverage']['details']['coverage_status'], 'NOT_COVERED')
            self.assertNotIn('historical', findings)
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision['route'], 'adjuster review')
            self.assertIsNone(decision['payable_amount'])

    def test_model_comparable_assessment_changes_the_route_without_hardcoded_threshold(self):
        identifier = self.prepare(10)
        def provider(*args):
            response = self.provider(*args)
            if args[2].endswith(':historical'):
                details = json.loads(args[4][0]['content'])['required_checks']['result']['details']
                self.assertNotIn('amount_justification', details)
                response['content'][0]['input'].update(amount_conclusion='SUPPORTED', summary='The evidence explains the higher amount.')
            return response
        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            history = session.get(Assessment, identifier).context['findings']['historical']['details']['amount_justification']
            self.assertEqual(history['conclusion'], 'SUPPORTED')
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'auto-approve')

    def test_unjustified_extra_cost_keeps_reconciled_claim_in_review(self):
        identifier = self.prepare(13)
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            claim = session.get(Claim, row.claim_id)
            claim.payload = {**claim.payload, "additional_costs": [{"amount": "1000.00", "reason": "Interior detailing"}]}
            claim.payload_hash = digest(claim.payload)

        def provider(*args):
            response = self.provider(*args)
            if args[2].endswith(':coverage'):
                submission = json.loads(args[4][0]["content"])["submission"]
                self.assertEqual(submission["additional_costs"][0]["reason"], "Interior detailing")
                self.assertIn("windscreen", submission["narrative"])
                self.assertTrue(submission["documents"][0]["line_items"])
                response["content"][0]["input"]["coverage_items"] = [
                    {"item": "Windscreen replacement", "outcome": "covered", "policy_term": "Glass damage is listed as covered.",
                     "reason": "The impact cracked the windscreen."},
                    {"item": "Interior detailing", "outcome": "not_covered", "policy_term": "Cover pays to repair damage arising from the incident.",
                     "reason": "The charge is unrelated to the windscreen damage."},
                    {"item": "Installation labour", "outcome": "undetermined", "policy_term": "Cover includes labour for installing covered parts.",
                     "reason": "A supporting labour breakdown is needed."},
                ]
            return response

        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            coverage = session.get(Assessment, identifier).context["findings"]["coverage"]
            self.assertEqual(coverage["details"]["financial"]["additional_costs_total"], "1000.00")
            self.assertEqual([item["outcome"] for item in coverage["details"]["coverage_items"]],
                             ["covered", "not_covered", "undetermined"])
            self.assertEqual(coverage["details"]["coverage_items"][1]["reason"], "The charge is unrelated to the windscreen damage.")
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision["route"], "adjuster review")
            self.assertIsNone(decision["payable_amount"])

    def test_policy_silence_is_disclosed_separately_and_holds_payment(self):
        identifier = self.prepare()
        omission = {"item": "Towing", "impact": "The supplied policy does not say whether a separate towing charge can be paid."}

        def provider(*args):
            response = self.provider(*args)
            if args[2] == "r1:coverage":
                response["content"][0]["input"]["policy_omissions"] = [omission]
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            details = session.get(Assessment, identifier).context["findings"]["coverage"]["details"]
            self.assertEqual(details["policy_omissions"], [omission])
            self.assertEqual([item["item"] for item in details["coverage_items"]], ["Reported loss"])
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision["route"], "adjuster review")
            self.assertIsNone(decision["payable_amount"])

    def test_police_report_silence_cannot_support_a_not_covered_finding(self):
        identifier = self.prepare()
        item = {"item": "Police report requirement for hit-and-run collision", "outcome": "not_covered",
                "policy_term": "No police_report_required_for field present in policy POL-EXAMPLE",
                "reason": "The policy is silent on whether a police report is required for collision losses."}
        omission = {"item": "No police_report_required_for field or hit-and-run reporting requirement", "impact": item["reason"]}
        legacy = {"agent": "coverage", "status": "review", "details": {
            "coverage_items": [item], "policy_omissions": [omission]}}
        displayed = display_finding(legacy, {})
        self.assertEqual(displayed["details"]["coverage_items"], [])
        self.assertEqual(displayed["details"]["policy_omissions"], [omission])
        self.assertEqual(legacy["details"]["coverage_items"], [item])

        def provider(*args):
            result = self.provider(*args)
            if args[2] == "r1:coverage":
                result["content"][0]["input"].update(coverage_conclusion="NOT_COVERED", coverage_items=[item])
            return result

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            finding = session.get(Assessment, identifier).context["findings"]["coverage"]["details"]
            self.assertEqual(finding["coverage_status"], "UNRESOLVED")
            self.assertEqual(finding["coverage_items"], [])
            self.assertEqual(finding["policy_omissions"], [{"item": item["item"], "impact": item["reason"]}])
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision["route"], "adjuster review")
            self.assertEqual(decision["information_required"], [])

    def test_supervisor_failure_cannot_approve(self):
        identifier = self.prepare()
        with patch('backend.agents.claude_investigation.call_model', side_effect=TimeoutError('test')):
            self.workflow.run(identifier)
        with self.store.session() as session:
            self.assertIsNone(session.get(DecisionRecord, identifier))
            self.assertEqual(session.get(Assessment, identifier).status, 'failed')

    def test_missing_checks_cannot_approve(self):
        identifier = self.prepare()
        with patch('backend.workflow.specialist_readiness.assess', return_value={'tasks': [], 'explanation': 'Skip checks'}), patch('backend.agents.claude_investigation.judge', return_value={'route': 'auto-approve', 'explanation': 'Unsupported', 'evidence': [], 'outstanding_gaps': []}):
            self.workflow.run(identifier)
        with self.store.session() as session:
            self.assertIsNone(session.get(DecisionRecord, identifier))
            self.assertEqual(session.get(Assessment, identifier).status, 'failed')

    def test_model_failure_preserves_verified_fraud(self):
        identifier = self.prepare(14)
        def fail_after_tools(*args, **kwargs):
            if args[2] == 'r3:fraud':
                raise TimeoutError('provider failure after checks')
            return self.provider(*args, **kwargs)
        with patch('backend.agents.claude_investigation.call_model', side_effect=fail_after_tools):
            self.workflow.run(identifier)
        with self.store.session() as session:
            from backend.workflow.investigation_retry import interrupted_attention
            self.assertIsNone(session.get(DecisionRecord, identifier))
            self.assertIsNone(session.scalar(select(HumanTask).filter_by(assessment_id=identifier)))
            row = session.get(Assessment, identifier)
            self.assertEqual(row.status, 'failed')
            self.assertIn("Evidence-backed fraud concern", interrupted_attention(row.context, session.get(Claim, row.claim_id).payload))
            finding = row.context['findings']['fraud_followup']
            self.assertEqual(finding['details']['error_type'], 'TimeoutError')
            self.assertTrue(finding['details']['checks_executed'])
            self.assertTrue(finding['details']['fraud_indicators'])
            check = session.scalar(select(ToolCall).where(ToolCall.assessment_id == identifier,
                ToolCall.stage == 'r3:fraud', ToolCall.tool == 'execute_checks'))
            self.assertIn(check.response['evidence_ref'], finding['details']['sources'])

    def test_high_severity_fraud_finding_routes_to_fraud_queue(self):
        identifier = self.prepare_with_conflicting_document_date(with_missing_itemisation=True)

        def provider(*args):
            response = self.provider(*args)
            if args[2] == 'r1:fraud':
                response['content'][0]['input']['fraud_concerns'] = [self.timeline_concern(
                    args, 'high', 'The submitted records give contradictory incident dates. ' * 11)]
            return response

        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            finding = session.get(Assessment, identifier).context['findings']['fraud_initial']
            self.assertEqual([item['severity'] for item in finding['details']['fraud_concerns']], ['high'])
            self.assertEqual(finding['details']['fraud_indicators'], [])
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision['route'], 'fraud escalation')
            self.assertEqual(decision['reason_codes'], ['HIGH_SEVERITY_FRAUD_CONCERN'])
            self.assertIn('contradictory incident dates', decision['justification'])
            self.assertEqual([item['location'] for item in decision['information_required']], ['Itemised charges'])
            self.assertEqual(session.scalar(select(HumanTask).filter_by(assessment_id=identifier)).queue, 'fraud')

    def prepare_with_conflicting_document_date(self, *, with_missing_itemisation=False):
        identifier = self.prepare()
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            payload = copy.deepcopy(claim.payload)
            payload['loss_date'] = '2026-09-15'
            payload['documents'][0]['date'] = '2026-09-13'
            if with_missing_itemisation:
                payload['documents'][0]['line_items'] = []
            claim.payload, claim.payload_hash = payload, digest(payload)
        return identifier

    def timeline_concern(self, args, severity, reason):
        supplied = json.loads(args[4][0]['content'])
        claim = supplied['submission']
        return {'severity': severity, 'reason': reason, 'evidence': [supplied['required_checks']['evidence_ref']],
                'basis': 'conflicting_evidence', 'observed_facts': [
                    {'path': 'submission:/loss_date', 'value': claim['loss_date']},
                    {'path': 'submission:/documents/0/date', 'value': claim['documents'][0]['date']},
                ]}

    def test_correct_saved_high_severity_decision_and_reassign_existing_task(self):
        identifier = self.prepare_with_conflicting_document_date()
        concern = 'The Perera Auto Works estimate is dated 13 September 2026, before the reported loss on 15 September 2026.'

        def provider(*args):
            response = self.provider(*args)
            if args[2] == 'r1:fraud':
                response['content'][0]['input']['fraud_concerns'] = [self.timeline_concern(args, 'high', concern)]
            return response

        with patch.object(agentic, 'call_model', side_effect=provider), patch('backend.workflow.routing_rules.high_severity_concerns', return_value=[]):
            self.workflow.run(identifier)
        original = export_audit(self.store, identifier)
        self.assertEqual(original['decision']['route'], 'adjuster review')
        self.assertEqual(original['human_task']['queue'], 'adjuster')

        corrected = escalate_published_high_severity(self.store, identifier)
        self.assertEqual(corrected, escalate_published_high_severity(self.store, identifier))
        self.assertEqual(corrected['route'], 'fraud escalation')
        self.assertIn(concern, corrected['justification'])
        package = export_audit(self.store, identifier)
        self.assertTrue(verify_audit(package))
        self.assertEqual(package['original_decision']['payload'], original['decision'])
        self.assertEqual(package['decision'], corrected)
        self.assertEqual(package['human_task'], {**original['human_task'], 'queue': 'fraud'})
        self.assertEqual(package['routing_correction']['event']['previous_task']['queue'], 'adjuster')

    def test_medium_fraud_finding_keeps_adjuster_route(self):
        identifier = self.prepare_with_conflicting_document_date()

        def provider(*args):
            response = self.provider(*args)
            if args[2] == 'r1:fraud':
                response['content'][0]['input']['fraud_concerns'] = [self.timeline_concern(
                    args, 'medium', 'The garage date precedes the loss and needs corroboration.')]
            return response

        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'adjuster review')
        with self.assertRaisesRegex(ValueError, 'high-severity'):
            escalate_published_high_severity(self.store, identifier)

    def test_missing_estimate_itemisation_and_vehicle_identity_cannot_be_fraud_indicators(self):
        identifier = self.prepare(15)
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            policy = session.get(Policy, claim.policy_id)
            policy.payload = {**policy.payload, 'vehicle': {'chassis_number': 'SYN-CH-INSURED'}}

        def provider(*args):
            response = self.provider(*args)
            if args[2] == 'r1:fraud':
                supplied = json.loads(args[4][0]['content'])
                submission = supplied['submission']
                analysis = response['content'][0]['input']
                analysis['summary'] = 'Missing itemisation raises suspicion.'
                analysis['review_reasons'] = ['Missing chassis number raises suspicion.']
                analysis['plausibility'] = {'assessment': 'inconclusive', 'reason': 'The attachments do not establish the reported circumstances.'}
                analysis['fraud_concerns'] = [
                    {'severity': 'high', 'reason': 'Estimate lists a lump sum and no individual charges.',
                     'basis': 'missing_information', 'evidence': analysis['evidence'], 'observed_facts': [
                         {'path': 'submission:/gross_amount', 'value': submission['gross_amount']},
                         {'path': 'submission:/registration', 'value': submission['registration']}]},
                    {'severity': 'high', 'reason': 'Missing chassis number could indicate a substitution.',
                     'basis': 'conflicting_evidence', 'evidence': analysis['evidence'], 'observed_facts': [
                         {'path': 'submission:/registration', 'value': submission['registration']},
                         {'path': 'submission:/vehicle/chassis_number', 'value': 'null'}]},
                ]
            return response

        with patch.object(agentic, 'call_model', side_effect=provider):
            self.workflow.run(identifier)
        with self.store.session() as session:
            details = session.get(Assessment, identifier).context['findings']['fraud_initial']
            self.assertEqual(details['status'], 'pass')
            self.assertEqual(details['details']['fraud_concerns'], [])
            self.assertEqual(details['details']['plausibility']['assessment'], 'inconclusive')
            self.assertNotIn('suspicion', details['summary'])
            decision = session.get(DecisionRecord, identifier).payload
            self.assertEqual(decision['route'], 'request more information')
            self.assertEqual({item['location'] for item in decision['information_required']}, {'Itemised charges', 'Vehicle details'})
            self.assertIsNone(session.scalar(select(HumanTask).filter_by(assessment_id=identifier)))

    def test_fraud_escalates_material_conflicting_excerpts_from_the_same_narrative(self):
        identifier = self.prepare()
        parked = "The car was stationary when the van hit it."
        moving = "The car was moving when the van hit it."
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            claim.payload = {**claim.payload, "narrative": f"{parked} {moving}"}
            claim.payload_hash = digest(claim.payload)

        def provider(*args):
            response = self.provider(*args)
            if args[2] == "r1:fraud":
                result = response["content"][0]["input"]
                result["fraud_concerns"] = [
                    {"severity": "high", "basis": "conflicting_evidence", "reason": "The claimant gives incompatible accounts of whether the car was moving at impact.",
                     "evidence": result["evidence"], "observed_facts": [{"path": "submission:/narrative", "value": quote} for quote in quotes]}
                    for quotes in ((parked, moving), (parked, "The car was reversing when struck."), (parked, "The car was stationary"))
                ]
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)
        package = export_audit(self.store, identifier)
        self.assertEqual(package["decision"]["route"], "fraud escalation")
        details = package["context"]["findings"]["fraud_initial"]["details"]
        self.assertEqual(details["plausibility"]["assessment"], "contradicted")
        [concern] = details["fraud_concerns"]
        self.assertEqual([fact["value"] for fact in concern["observed_facts"]], [parked, moving])
        self.assertTrue(verify_audit(package))

    def test_fraud_cross_checks_narrative_with_extracted_manual_report(self):
        identifier = self.prepare(manual=True)
        report_id = f"{identifier}-report"
        claimed = "My vehicle was parked when the van struck it."
        documented = "The insured vehicle was moving at impact."
        with self.store.session.begin() as session:
            claim = session.get(Claim, session.get(Assessment, identifier).claim_id)
            claim.payload = {**claim.payload, "narrative": f"I was outside the depot. {claimed}"}
            claim.payload_hash = digest(claim.payload)
            session.add(Attachment(id=report_id, assessment_id=identifier, name="Police report.pdf",
                                   media_type="application/pdf", content=""))
            session.flush()
            session.add(AttachmentExtraction(attachment_id=report_id, payload={"attachment_id": report_id,
                "name": "Police report.pdf", "document_type": "police_report", "status": "extracted",
                "summary": f"The officer reviewed traffic camera footage. {documented}"}))

        def provider(*args):
            response = self.provider(*args)
            if args[2] == "r1:fraud":
                evidence = json.loads(args[4][0]["content"])["submission"]["uploaded_evidence"]
                index = next(i for i, doc in enumerate(evidence) if doc.get("attachment_id") == report_id)
                result = response["content"][0]["input"]
                result["fraud_concerns"] = [{"severity": "medium", "basis": "conflicting_evidence",
                    "reason": "The claimant said the vehicle was parked. The police report says it was moving in the footage.",
                    "evidence": result["evidence"], "observed_facts": [{"path": "submission:/narrative", "value": claimed},
                        {"path": f"submission:/uploaded_evidence/{index}/summary", "value": documented}]}]
            return response

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)
        package = export_audit(self.store, identifier)
        self.assertEqual(package["decision"]["route"], "adjuster review")
        details = package["context"]["findings"]["fraud_initial"]["details"]
        [concern] = details["fraud_concerns"]
        self.assertEqual(concern["severity"], "medium")
        self.assertEqual([fact["value"] for fact in concern["observed_facts"]], [claimed, documented])

    def test_truncated_coverage_retries_once_without_repeating_checks(self):
        identifier = self.prepare()
        real_client, real_call = httpx.Client, agentic.call_model
        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if payload["max_tokens"] < 4096:
                return httpx.Response(200, json={"stop_reason": "max_tokens",
                    "content": [{"type": "text", "text": "Incomplete analysis that must stay out of the retry."}]})
            response = self.provider(self.store, identifier, "r1:coverage", payload["system"],
                                     payload["messages"], payload["tools"], payload["tool_choice"])
            response["content"][0]["input"]["evidence"].append("submission:/documents/0")
            return httpx.Response(200, json=response)

        def provider(*args, **kwargs):
            return real_call(*args, **kwargs) if args[2] == "r1:coverage" else self.provider(*args)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture"}), \
             patch.object(agentic.httpx, "Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs)), \
             patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)

        self.assertEqual([request["max_tokens"] for request in requests], [3200, 8192])
        self.assertEqual(requests[1]["tool_choice"], {"type": "tool", "name": "submit_analysis"})
        self.assertEqual([tool["name"] for tool in requests[1]["tools"]], ["submit_analysis"])
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertIn("submission:/documents/0", requests[1]["tools"][0]["input_schema"]["properties"]["evidence"]["items"]["enum"])
        with self.store.session() as session:
            calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier,
                ModelInvocation.stage == "r1:coverage").order_by(ModelInvocation.created_at)).all()
            self.assertEqual([call.status for call in calls], ["failed", "complete"])
            self.assertEqual(calls[0].response["stop_reason"], "max_tokens")
            checks = session.scalars(select(ToolCall).where(ToolCall.assessment_id == identifier,
                ToolCall.stage == "r1:coverage", ToolCall.tool == "execute_checks")).all()
            self.assertEqual(len(checks), 1)
            finding = session.get(Assessment, identifier).context["findings"]["coverage"]
            self.assertEqual(finding["details"]["coverage_status"], "COVERED")
            self.assertIn("submission:/documents/0", finding["evidence"])
            self.assertIn("submission:/documents/0", finding["details"]["submitted_documents"])

    def test_truncated_final_coverage_turn_can_still_recover(self):
        identifier = self.prepare(manual=True)
        coverage_requests = []

        def provider(*args, **kwargs):
            if args[2] != "r1:coverage":
                return self.provider(*args)
            coverage_requests.append((copy.deepcopy(args[4]), kwargs))
            turn = len(coverage_requests)
            if turn <= 3:
                self.assertIn("search_policy_documents", [tool["name"] for tool in args[5]])
                return {"content": [{"type": "tool_use", "id": f"search-{turn}", "name": "search_policy_documents",
                                     "input": {"query": f"glass coverage wording {turn}"}}]}
            if turn == 4:
                self.assertEqual(args[6], {"type": "tool", "name": "submit_analysis"})
                raise agentic.ModelOutputTruncated("Final turn reached its output limit")
            return self.provider(*args)

        with patch.object(agentic, "call_model", side_effect=provider):
            self.workflow.run(identifier)

        self.assertEqual(len(coverage_requests), 5)
        (final_messages, _), (retry_messages, retry_options) = coverage_requests[-2:]
        self.assertEqual(final_messages, retry_messages)
        self.assertEqual(len(retry_messages), 7)
        self.assertEqual(retry_options["max_tokens"], 8192)
        with self.store.session() as session:
            details = session.get(Assessment, identifier).context["findings"]["coverage"]["details"]
            self.assertEqual(details["coverage_status"], "COVERED")
            self.assertNotIn("error_type", details)

    def test_incomplete_and_invalid_specialist_responses_recover_before_routing(self):
        identifier = self.prepare()
        coverage_attempts = 0

        def provider(*args):
            nonlocal coverage_attempts
            store, assessment, stage, instructions, messages, tools, choice = args
            forced = choice == {"type": "tool", "name": "submit_analysis"}
            if stage == "r1:coverage":
                coverage_attempts += 1
                if coverage_attempts == 1:
                    raise agentic.IncompleteModelResponse("Model ended without submitting an analysis")
            if stage == "r1:fraud" and forced:
                self.assertTrue(messages[-1]["content"][0]["is_error"])
                return self.provider(store, assessment, stage, instructions, messages[:1], tools, choice)
            response = self.provider(*args)
            if stage == "r1:fraud":
                response["content"][0]["input"]["fraud_concerns"] = [
                    {"severity": "medium", "reason": "x" * 1300,
                     "evidence": response["content"][0]["input"]["evidence"]}]
            return response

        with patch.object(agentic, "call_model", side_effect=provider) as model:
            self.workflow.run(identifier)
        calls = Counter(call.args[2] for call in model.call_args_list)
        self.assertEqual(calls["r1:coverage"], 2)
        self.assertEqual(calls["r1:fraud"], 2)
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, identifier).payload["route"], "auto-approve")

    def test_coverage_timeout_retries_once_and_keeps_policy_check_evidence(self):
        for index, recover in enumerate((True, False)):
            with self.subTest(recover=recover):
                identifier = self.prepare(index)
                requests = []
                real_client, real_call = httpx.Client, agentic.call_model

                def respond(request):
                    payload = json.loads(request.content)
                    requests.append(payload)
                    if len(requests) == 1 or not recover:
                        raise httpx.ReadTimeout("The read operation timed out", request=request)
                    return httpx.Response(200, json=self.provider(self.store, identifier, "r1:coverage",
                        payload["system"], payload["messages"], payload["tools"], payload["tool_choice"]))

                def provider(*args):
                    return real_call(*args) if args[2] == "r1:coverage" else self.provider(*args)

                with (
                    patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture"}),
                    patch.object(agentic.httpx, "Client", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)),
                    patch.object(agentic, "call_model", side_effect=provider),
                ):
                    self.workflow.run(identifier)

                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(requests[0]["tool_choice"], {"type": "tool", "name": "submit_analysis"})
                with self.store.session() as session:
                    calls = session.scalars(select(ModelInvocation).where(ModelInvocation.assessment_id == identifier,
                        ModelInvocation.stage == "r1:coverage").order_by(ModelInvocation.created_at)).all()
                    self.assertEqual([call.status for call in calls], ["failed", "complete" if recover else "failed"])
                    self.assertEqual(calls[0].response["error_type"], "ReadTimeout")
                    tools = session.scalars(select(ToolCall.tool).where(ToolCall.assessment_id == identifier,
                        ToolCall.stage == "r1:coverage")).all()
                    self.assertEqual(tools.count("read_submission"), 1)
                    self.assertEqual(tools.count("retrieve_policy"), 1)
                    self.assertEqual(tools.count("execute_checks"), 1)
                    finding = session.get(Assessment, identifier).context["findings"]["coverage"]
                    decision = session.get(DecisionRecord, identifier)
                    if recover:
                        self.assertEqual(finding["details"]["coverage_status"], "COVERED")
                        self.assertEqual(decision.payload["route"], "auto-approve")
                    else:
                        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
                        self.assertTrue(finding["details"]["checks_executed"])
                        self.assertIn("coverage model timed out", finding["summary"])
                        self.assertIsNone(decision)
                        self.assertEqual(session.get(Assessment, identifier).status, "failed")

    def test_deadline_prevents_checks_as_well_as_model_requests(self):
        identifier = self.prepare()
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            session.get(Claim, row.claim_id).received_at = '2020-01-01T00:00:00+00:00'
        with patch.object(agentic, 'execute_checks') as checks, patch.object(agentic, 'call_model') as model:
            with self.assertRaises(TimeoutError):
                agentic.run_specialist(self.store, identifier, {}, 0, 'intake', 'Check claim')
        checks.assert_not_called()
        model.assert_not_called()

    def test_policy_search_is_scoped_and_binding_is_frozen(self):
        identifier = self.prepare()
        other = self.upload('SECRET OTHER POLICY')
        index_document(self.store, other)
        result = search_document(self.store, self.document, 'policy')
        self.assertNotIn('SECRET', json.dumps(result))
        with self.store.session() as session:
            assessment = session.get(Assessment, identifier)
            policy_id = session.get(Claim, assessment.claim_id).policy_id
        with self.store.session.begin() as session:
            session.merge(PolicyBinding(policy_id=policy_id, document_id=other))
        with self.store.session() as session:
            self.assertEqual(session.get(Assessment, identifier).context['wording_document_id'], self.document)
        self.assertEqual(search_document(self.store, None, 'policy')['status'], 'unavailable')

    def test_scanned_pdf_requires_ocr_and_retry_can_index_txt(self):
        from pypdf import PdfWriter
        stream = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(stream)
        doc_id = create_document(self.store, title='Scanned', product='MOTOR', version='1', filename='scan.pdf',
            media_type='application/pdf', content=base64.b64encode(stream.getvalue()).decode())
        index_document(self.store, doc_id)
        with self.store.session() as session:
            doc = session.get(PolicyDocument, doc_id)
            self.assertEqual(doc.status, 'failed')
            self.assertIn('OCR required', doc.error)

    def test_fabricated_citations_are_rejected(self):
        identifier = self.prepare()
        for reference in ('chunk:invented', 'submission:/documents/99'):
            with self.subTest(reference=reference), patch('backend.agents.claude_investigation.call_model', return_value={
                'stop_reason': 'tool_use', 'content': [{'type': 'tool_use', 'id': 'bad', 'name': 'submit_analysis',
                'input': {'summary': 'Invented evidence', 'review_reasons': [], 'evidence': [reference],
                          'coverage_conclusion': 'COVERED', 'coverage_items': []}}]
            }):
                with self.assertRaisesRegex(ValueError, 'Citation'):
                    agentic.run_specialist(self.store, identifier, {}, 1, 'coverage', 'Check')

    def test_request_budget_is_persisted(self):
        identifier = self.prepare()
        with self.store.session.begin() as session:
            for i in range(40):
                session.add(ModelInvocation(id=str(i), assessment_id=identifier, stage='test', payload={}, response={}))
        with patch('httpx.Client.post') as post:
            with self.assertRaisesRegex(ValueError, 'budget'):
                agentic.call_model(self.store, identifier, 'test', '', [], [], {})
            post.assert_not_called()

    def test_admin_document_api_and_role_permissions(self):
        from fastapi.testclient import TestClient
        from backend.api import create_app
        with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'deterministic'}):
            app = create_app(self.root / 'api', auto_seed=False)
            with TestClient(app) as client:
                app.state.worker.close()
                self.assertEqual(client.get('/api/policy-documents').status_code, 401)
                self.assertEqual(client.get('/api/policy-documents/example/preview').status_code, 401)
                client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
                self.assertEqual(client.get('/api/policy-documents').status_code, 403)
                self.assertEqual(client.get('/api/policy-documents/example/preview').status_code, 403)
                client.post('/api/auth/login', json={'email': 'admin@ruhunu.local', 'password': 'AdminDemo!2026'})
                body = {'title': 'Terms', 'product': 'MOTOR', 'version': '1', 'filename': 'terms.txt', 'media_type': 'text/plain',
                        'content': base64.b64encode(b'Glass damage is covered subject to the schedule.').decode()}
                response = client.post('/api/policy-documents', json=body)
                self.assertEqual(response.status_code, 202, response.text)
                doc_id = response.json()['id']
                preview = client.get(f'/api/policy-documents/{doc_id}/preview')
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.content, base64.b64decode(body['content']))
                self.assertTrue(preview.headers['content-type'].startswith('text/plain'))
                self.assertEqual(preview.headers['content-disposition'], 'inline')
                self.assertEqual(preview.headers['cache-control'], 'no-store')
                self.assertEqual(client.get('/api/policy-documents/missing/preview').status_code, 404)
                self.assertEqual(client.put('/api/generator/policy-wording', json={'document_id': doc_id}).status_code, 409)
                index_document(app.state.store, doc_id)
                self.assertEqual(client.get(f'/api/policy-documents/{doc_id}').json()['status'], 'ready')
                self.assertEqual(client.get(f'/api/policy-documents/{doc_id}/original').content, base64.b64decode(body['content']))
                self.assertEqual(client.put('/api/generator/policy-wording', json={'document_id': doc_id}).status_code, 200)
                generated = generate_batch(app.state.store, self.root / 'api-labels.sqlite3', 100, 1)
                with app.state.store.session() as session:
                    assessment = session.get(Assessment, generated['assessment_ids'][0])
                    policy_id = session.get(Claim, assessment.claim_id).policy_id
                self.assertEqual(client.put(f'/api/policies/{policy_id}/wording', json={'document_id': doc_id}).status_code, 200)
                self.assertEqual(client.post(f'/api/policy-documents/{doc_id}/retry').status_code, 409)
                self.assertEqual(client.get('/api/policy-documents').json()['generator_default'], doc_id)

    def test_checkpoint_resumes_publication_without_repeating_specialists(self):
        identifier = self.prepare()
        with patch('backend.agents.claude_investigation.call_model', side_effect=self.provider), patch.object(self.workflow, 'publish', side_effect=RuntimeError('crash before publication')):
            with self.assertRaises(RuntimeError):
                self.workflow.run(identifier)
        self.workflow.close()
        self.workflow = ClaimsWorkflow(self.store, self.root / 'checkpoints.sqlite3')
        with patch('backend.agents.claude_investigation.call_model', side_effect=AssertionError('must resume at publication')):
            self.workflow.run(identifier)
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'auto-approve')
            self.assertEqual(len(session.scalars(select(AgentRun).where(AgentRun.assessment_id == identifier)).all()), 4)

    def test_forbidden_tool_is_rejected(self):
        identifier = self.prepare()
        with patch('backend.agents.claude_investigation.call_model', return_value={'stop_reason': 'tool_use', 'content': [
            {'type': 'tool_use', 'id': 'bad', 'name': 'publish_payment', 'input': {}}]}):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                agentic.run_specialist(self.store, identifier, {}, 1, 'coverage', 'Check')

    def test_deadline_blocks_new_model_calls(self):
        identifier = self.prepare()
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            claim = session.get(Claim, assessment.claim_id)
            claim.received_at = '2020-01-01T00:00:00+00:00'
        with patch('httpx.Client.post') as post:
            with self.assertRaises(TimeoutError):
                agentic.call_model(self.store, identifier, 'test', '', [], [], {})
            post.assert_not_called()

    def test_text_pdf_extraction_preserves_page_citations(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        content = DecodedStreamObject()
        content.set_data(b'BT /F1 12 Tf 20 250 Td (Glass coverage wording.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(content)
        stream = io.BytesIO()
        writer.write(stream)
        doc_id = create_document(self.store, title='PDF terms', product='MOTOR', version='1', filename='terms.pdf',
            media_type='application/pdf', content=base64.b64encode(stream.getvalue()).decode())
        index_document(self.store, doc_id)
        result = search_document(self.store, doc_id, 'glass')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['passages'][0]['page'], 1)
        self.assertIn('Glass coverage', result['passages'][0]['text'])

    def test_failed_index_can_be_retried_without_changing_source(self):
        doc_id = self.upload('Glass policy wording')
        with patch('backend.evidence.policy_wording_index.encoder', side_effect=OSError('model unavailable')):
            index_document(self.store, doc_id)
        with self.store.session() as session:
            doc = session.get(PolicyDocument, doc_id)
            self.assertEqual(doc.status, 'failed')
            original_hash = doc.file_hash
        index_document(self.store, doc_id)
        with self.store.session() as session:
            doc = session.get(PolicyDocument, doc_id)
            self.assertEqual(doc.status, 'ready')
            self.assertEqual(doc.file_hash, original_hash)

    def test_generated_claim_ignores_legacy_uploaded_corpus(self):
        ensure_synthetic_default(self.store)
        self.assertEqual(uploaded_document_ids(self.store), [self.document])
        identifier = self.prepare()
        with self.store.session.begin() as session:
            row = session.get(Assessment, identifier)
            row.context = {**row.context, 'policy_search_mode': 'uploaded-corpus', 'policy_corpus_ids': [self.document]}
        def provider(*args):
            system, stage, messages, tools = args[3], args[2], args[4], args[5]
            self.assertNotIn(self.document, messages[0]['content'])
            self.assertNotIn('search_policy_documents', {tool['name'] for tool in tools})
            if stage.endswith(':coverage'):
                self.assertIn('Use only the linked generated policy', system)
            return self.provider(*args)
        with patch.object(agentic, 'call_model', side_effect=provider), \
             patch('backend.evidence.specialist_evidence_tools.search_uploaded_documents') as search:
            self.workflow.run(identifier)
        search.assert_not_called()
        package = export_audit(self.store, identifier)
        self.assertEqual(package['decision']['route'], 'auto-approve')
        self.assertFalse(package['policy_documents'])
        self.assertNotIn('policy_corpus_ids', package['context'])

    def test_corpus_retrieval_retains_document_citations_and_recommendation(self):
        identifier = self.prepare(manual=True)
        witness_id = f"{identifier}-witness"
        witness_ref = f"attachment:{witness_id}"
        with self.store.session.begin() as session:
            session.add(Attachment(id=witness_id, assessment_id=identifier, name="Witness statement.pdf",
                                   media_type="application/pdf", content=""))
            session.flush()
            session.add(AttachmentExtraction(attachment_id=witness_id, payload={
                "attachment_id": witness_id, "status": "extracted", "document_type": "witness_statement",
                "name": "Witness statement.pdf", "source": witness_ref, "summary": "The witness saw a stone strike the windscreen.",
            }))

        def provider(*args):
            response = self.provider(*args)
            if not args[2].endswith(":coverage"):
                return response
            initial = json.loads(args[4][0]["content"])
            self.assertIn({"source": witness_ref, "name": "Witness statement.pdf", "status": "extracted",
                           "document_type": "witness_statement"}, initial["uploaded_documents"])
            if len(args[4]) == 1:
                response["content"].append({"type": "tool_use", "id": "read-upload",
                                            "name": "read_uploaded_documents", "input": {}})
            else:
                outputs = [json.loads(output["content"])["result"] for output in args[4][-1]["content"]]
                self.assertTrue(any("stone strike" in doc.get("summary", "") for doc in outputs[-1]["evidence"]))
                item = response["content"][0]["input"]["coverage_items"][0]
                item.update(document_refs=[witness_ref], reason="The witness saw a stone strike the windscreen, supporting the reported glass loss.")
            return response

        with patch('backend.agents.claude_investigation.call_model', side_effect=provider):
            self.workflow.run(identifier)
        package = export_audit(self.store, identifier)
        self.assertEqual(package['decision']['route'], 'adjuster review')
        coverage = package['context']['findings']['coverage']['details']
        self.assertTrue(coverage['policy_rag']['cited'])
        self.assertEqual(coverage['coverage_items'][0]['document_refs'], [witness_ref])
        self.assertEqual(coverage['submitted_documents'][witness_ref], 'Witness statement.pdf')
        self.assertEqual(package['policy_documents'][0]['id'], self.document)
        result = search_uploaded_documents(self.store, [self.document], 'glass')
        self.assertEqual(result['passages'][0]['document_id'], self.document)
        self.assertTrue(verify_audit(package))

    def test_synthetic_wording_review_validates_citations_and_materiality(self):
        identifier = self.prepare()
        document = self.upload("The glass repair benefit covers windscreen damage. The excess is LKR 50,000 for every glass claim. "
                               "The separate body repair benefit covers labour up to LKR 14,000.")
        index_document(self.store, document)
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            assessment.context = {**assessment.context, "synthetic_wording_document_id": document}
            context = assessment.context
            policy = session.get(Policy, session.get(Claim, assessment.claim_id).policy_id)
            policy.payload = {**policy.payload, "repair_terms": {"labour": {"basis": "per_claim", "limit": "15000.00"}}}

        def provider(materiality, *, missing_citation=False):
            def respond(*args):
                response = self.provider(*args)
                block = response["content"][0]
                if not args[2].endswith(":coverage") or block["name"] != "submit_analysis":
                    return response
                passages = json.loads(args[4][-1]["content"][0]["content"])["result"]["passages"]
                [passage] = passages
                self.assertIn("50,000", passage["text"])
                ref = passage["source"]
                analysis = block["input"]
                analysis.update(applicable_passages=[ref], policy_conflicts=[])
                analysis["coverage_items"][0].update(basis="both", passage_refs=[] if missing_citation else [ref])
                if materiality:
                    conflict = ({"item": "Glass deductible", "structured_term": "LKR 10,000 deductible",
                                 "wording_term": "LKR 50,000 excess",
                                 "impact": "The wording would reduce the payable by LKR 40,000."}
                                if materiality == "material" else
                                {"item": "Body repair labour limit", "structured_term": "LKR 15,000 limit",
                                 "wording_term": "LKR 14,000 limit",
                                 "impact": "This windscreen claim does not use the body repair benefit."})
                    analysis["policy_conflicts"] = [{**conflict, "passage_refs": [ref], "materiality": materiality}]
                return response
            return respond

        with patch.object(agentic, "call_model", side_effect=provider(None, missing_citation=True)):
            finding = agentic.run_specialist(self.store, identifier, context, 6, "coverage", "Reconcile wording")
        self.assertEqual(finding["status"], "review")
        self.assertEqual(finding["details"]["coverage_items"], [])
        self.assertTrue(any("requires an applicable wording citation" in reason for reason in finding["details"]["review_reasons"]))

        with patch.object(agentic, "call_model", side_effect=provider("immaterial")):
            finding = agentic.run_specialist(self.store, identifier, context, 7, "coverage", "Reconcile wording")
        self.assertEqual(finding["status"], "pass")
        self.assertEqual(finding["details"]["policy_conflicts"][0]["materiality"], "immaterial")
        self.assertEqual(finding["details"]["financial"]["deductible"], "10000.00")

        with patch.object(agentic, "call_model", side_effect=provider("material")):
            self.workflow.run(identifier)
        package = export_audit(self.store, identifier)
        finding = package["context"]["findings"]["coverage"]
        self.assertEqual(package["decision"]["route"], "adjuster review")
        self.assertEqual(package["decision"]["reason_codes"], ["POLICY_WORDING_CONFLICT"])
        self.assertIn("reduce the payable by LKR 40,000", package["decision"]["justification"])
        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
        self.assertIsNone(finding["details"]["financial"]["payable"])
        self.assertEqual(finding["details"]["policy_conflicts"][0]["passage_refs"], [f"chunk:{document}:0"])
        self.assertEqual(package["policy_wording"]["id"], document)
        self.assertTrue(verify_audit(package))

    def test_coverage_uses_submitted_reports_and_searches_wording_when_needed(self):
        identifier = self.prepare()
        document = self.upload("Windscreen damage from road debris is covered. Damage during workshop servicing is excluded.")
        index_document(self.store, document)
        report_ref, witness_ref = "submission:/documents/1", "submission:/documents/2"
        with self.store.session.begin() as session:
            assessment = session.get(Assessment, identifier)
            context = {**assessment.context, "synthetic_wording_document_id": document}
            claim = session.get(Claim, assessment.claim_id)
            claim.payload = {**claim.payload, "documents": [*claim.payload["documents"],
                {"type": "incident_report", "name": "Incident report.json", "summary": "The driver reported road debris."},
                {"type": "witness_statement", "name": "Witness statement.json", "summary": "The glass broke during workshop servicing."}]}
            claim.payload_hash = digest(claim.payload)

        def provider(*args, search=False, invented=False):
            messages = args[4]
            initial = json.loads(messages[0]["content"])
            self.assertEqual(initial["submitted_document_refs"][report_ref], "Incident report.json")
            self.assertIn("workshop servicing", initial["submission"]["documents"][2]["summary"])
            if search and len(messages) == 1:
                return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "search",
                    "name": "search_policy_documents", "input": {"query": "Windscreen damage workshop servicing exclusion"}}]}
            passages = json.loads(messages[-1]["content"][0]["content"])["result"]["passages"] if search else []
            refs = [passage["source"] for passage in passages]
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "submit", "name": "submit_analysis",
                "input": {"summary": "The cause of the windscreen damage is disputed.", "review_reasons": [],
                    "evidence": [initial["required_checks"]["evidence_ref"]], "coverage_conclusion": "UNRESOLVED",
                    "policy_applicability": "applicable" if search else "unclear", "applicable_passages": refs, "policy_conflicts": [],
                    "coverage_items": [{"item": "Windscreen damage", "outcome": "undetermined",
                        "policy_term": "Workshop damage is excluded." if search else "The schedule lists glass damage as a covered loss type.",
                        "reason": "The incident report attributes the damage to road debris. The witness statement attributes it to servicing.",
                        "basis": "wording" if search else "schedule", "passage_refs": refs,
                        "document_refs": [report_ref, "submission:/documents/99" if invented else witness_ref]}]}}]}

        with patch.object(agentic, "call_model", side_effect=provider) as model:
            finding = agentic.run_specialist(self.store, identifier, context, 6, "coverage", "Assess the submitted documents")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(finding["status"], "review")
        self.assertEqual(finding["details"]["coverage_status"], "UNRESOLVED")
        self.assertEqual(finding["details"]["policy_rag"]["retrievals"], [])
        self.assertIn("search_skipped", finding["details"]["policy_rag"])
        self.assertEqual(finding["details"]["coverage_items"][0]["document_refs"], [report_ref, witness_ref])

        with patch.object(agentic, "call_model", side_effect=lambda *args: provider(*args, search=True)) as model:
            finding = agentic.run_specialist(self.store, identifier, context, 7, "coverage", "Check the exclusion for workshop work")
        self.assertEqual(model.call_count, 2)
        self.assertTrue(finding["details"]["policy_rag"]["cited"])
        self.assertEqual(finding["details"]["coverage_items"][0]["passage_refs"], [f"chunk:{document}:0"])
        self.assertEqual(finding["details"]["submitted_documents"], {report_ref: "Incident report.json", witness_ref: "Witness statement.json"})
        with self.store.session() as session:
            submission = session.scalar(select(ToolCall).filter_by(assessment_id=identifier, stage="r7:coverage", tool="read_submission"))
            self.assertEqual(submission.response["documents"][2]["name"], finding["details"]["submitted_documents"][witness_ref])
            self.assertEqual(submission.response_hash, digest(submission.response))
            [lookup] = session.scalars(select(ToolCall).filter_by(assessment_id=identifier, stage="r7:coverage", tool="search_policy_documents")).all()
            self.assertEqual(lookup.response["result"]["document_id"], document)
            self.assertIn("workshop servicing", lookup.request["query"])

        with patch.object(agentic, "call_model", side_effect=lambda *args: provider(*args, invented=True)):
            finding = agentic.run_specialist(self.store, identifier, context, 8, "coverage", "Check document citations")
        self.assertEqual(finding["status"], "review")
        self.assertEqual(finding["details"]["coverage_items"], [])
        self.assertIn("The submitted documents cited for Windscreen damage could not be verified.", finding["details"]["review_reasons"])

    def test_irrelevant_uploaded_wording_requires_review(self):
        identifier = self.prepare(manual=True)
        def irrelevant(*args, **kwargs):
            result = self.provider(*args, **kwargs)
            if args[2].endswith(':coverage'):
                for block in result['content']:
                    if block['name'] == 'submit_analysis':
                        block['input']['policy_applicability'] = 'irrelevant'
            return result
        with patch('backend.agents.claude_investigation.call_model', side_effect=irrelevant):
            self.workflow.run(identifier)
        with self.store.session() as session:
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'adjuster review')
            coverage = session.get(Assessment, identifier).context['findings']['coverage']['details']
            self.assertTrue(coverage['policy_rag']['cited'])
            self.assertEqual(coverage['coverage_status'], 'UNRESOLVED')

    def test_manual_claim_with_no_indexed_wording_blocks_coverage(self):
        identifier = self.prepare(manual=True)
        with self.store.session.begin() as session:
            session.get(PolicyDocument, self.document).status = 'failed'
        with patch.object(agentic, 'call_model', side_effect=self.provider) as model:
            self.workflow.run(identifier)
        self.assertFalse(any(call.args[2].endswith(':coverage') for call in model.call_args_list))
        with self.store.session() as session:
            context = session.get(Assessment, identifier).context
            evidence = context['findings']['intake']['details']['coverage_readiness']
            self.assertEqual(evidence['checks'], {'policy_period': True, 'vehicle_match': True, 'loss_and_costs': False})
            self.assertIn('no indexed policy wording', evidence['internal_issue'])
            self.assertIn('fraud_initial', context['findings'])
            self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'adjuster review')

    def test_manual_submission_queues_same_graph_and_preserves_original(self):
        from fastapi.testclient import TestClient
        from backend.api import create_app
        with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'deterministic'}):
            app = create_app(self.root / 'manual-api', auto_seed=False)
            with TestClient(app) as client:
                app.state.worker.close()
                doc_id = create_document(app.state.store, title='Motor wording', product='MOTOR', version='1', filename='motor.txt',
                    media_type='text/plain', content=base64.b64encode(b'Glass replacement is covered subject to policy schedule.').decode())
                index_document(app.state.store, doc_id)
                with app.state.store.session.begin() as session:
                    session.add(Policy(id='MANUAL-POL', payload={
                        'policy_id': 'MANUAL-POL', 'registration': 'ABC-1234', 'starts_on': '2026-01-01',
                        'ends_before': '2027-01-01', 'covered_losses': ['glass'], 'limit': '200000.00',
                        'deductible': '10000.00', 'clause': 'MOTOR-1',
                    }))
                client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
                body = {'policy_id': 'MANUAL-POL', 'registration': 'ABC-1234', 'loss_type': 'glass', 'loss_date': '2026-10-15',
                        'gross_amount': '100000.00', 'narrative': 'A stone cracked the windscreen.', 'attachments': [
                            {'name': 'estimate.png', 'media_type': 'image/png', 'content': base64.b64encode(b'\x89PNG\r\n\x1a\nsynthetic fixture').decode()}]}
                with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'anthropic'}):
                    response = client.post('/api/portal/claims', json=body)
                self.assertEqual(response.status_code, 201, response.text)
                identifier = response.json()['id']
                with app.state.store.session() as session:
                    assessment = session.get(Assessment, identifier)
                    original = copy.deepcopy(session.get(Claim, assessment.claim_id).payload)
                    self.assertEqual(assessment.status, 'queued')
                    self.assertEqual(assessment.execution['workflow'], 'claude-intake-first-v3')
                    self.assertIsNone(session.get(DecisionRecord, identifier))
                client.post('/api/auth/login', json={'email': 'staff@ruhunu.local', 'password': 'StaffDemo!2026'})
                review = {'outcome': 'approved', 'reason': 'Staff confirms the submitted evidence.'}
                self.assertEqual(client.post(f'/api/portal/claims/{identifier}/review', json=review).status_code, 409)
                with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'anthropic'}), patch('backend.agents.claude_investigation.call_model', side_effect=self.provider) as mocked_calls:
                    app.state.worker.workflow.run(identifier)
                    app.state.worker.workflow.run(identifier)
                extraction_calls = [call for call in mocked_calls.call_args_list if call.args[2].startswith('extract:')]
                self.assertEqual(len(extraction_calls), 1)
                self.assertEqual(extraction_calls[0].args[4][0]['content'][0]['type'], 'image')
                with app.state.store.session() as session:
                    assessment = session.get(Assessment, identifier)
                    self.assertEqual(session.get(Claim, assessment.claim_id).payload, original)
                    self.assertEqual(session.get(DecisionRecord, identifier).payload['route'], 'adjuster review')
                    [extraction] = session.scalars(select(AttachmentExtraction)).all()
                    self.assertEqual(extraction.payload['document_type'], 'repair_estimate')
                    self.assertEqual(extraction.payload['garage_name'], 'Sample Auto Glass')
                    self.assertEqual(extraction.payload['garage_address'], '25 Sample Road, Colombo')
                    runs = session.scalars(select(AgentRun)).all()
                    self.assertEqual({run.stage for run in runs}, {'r0:intake', 'r1:coverage', 'r1:fraud'})
                    self.assertNotIn('historical', assessment.context['findings'])
                item = client.get('/api/portal/claims').json()[0]
                self.assertTrue(item['investigation']['tools'])
                self.assertTrue(item['investigation']['plans'])
                self.assertEqual(client.post(f'/api/portal/claims/{identifier}/review', json=review).status_code, 200)
                client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
                item = client.get('/api/portal/claims').json()[0]
                self.assertIsNone(item['investigation'])
                self.assertIsNone(item['triage'])
                self.assertEqual(item['findings'], {})

    def test_old_manual_investigation_creates_idempotent_revision(self):
        from fastapi.testclient import TestClient
        from backend.api import create_app
        from backend.storage.sqlite_store import Ownership, Account
        with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'deterministic'}):
            app = create_app(self.root / 'legacy-manual-api', auto_seed=False)
            with TestClient(app) as client:
                app.state.worker.close()
                client.post('/api/auth/login', json={'email': 'user@ruhunu.local', 'password': 'UserDemo!2026'})
                body = {'policy_id': 'OLD-POL', 'registration': 'OLD-1234', 'loss_type': 'glass', 'loss_date': '2026-10-15',
                        'gross_amount': '100000.00', 'narrative': 'A stone cracked the windscreen.', 'attachments': []}
                identifier = client.post('/api/portal/claims', json=body).json()['id']
                with app.state.store.session.begin() as session:
                    assessment = session.get(Assessment, identifier)
                    assessment.status = 'complete'
                    assessment.execution = {**assessment.execution, 'workflow': 'claude-supervisor-v2'}
                    assessment.context = {**assessment.context, 'findings': {'intake': {'summary': 'Retained historical finding'}}}
                    session.add(DecisionRecord(assessment_id=identifier, payload={'route': 'adjuster review', 'justification': 'Legacy manual review'}, payload_hash='legacy'))
                self.assertEqual(client.post(f'/api/portal/claims/{identifier}/investigate').status_code, 403)
                client.post('/api/auth/login', json={'email': 'staff@ruhunu.local', 'password': 'StaffDemo!2026'})
                result = client.post(f'/api/portal/claims/{identifier}/investigate')
                self.assertEqual(result.status_code, 202, result.text)
                new_id = result.json()['id']
                self.assertNotEqual(new_id, identifier)
                self.assertEqual(client.post(f'/api/portal/claims/{identifier}/investigate').json()['id'], new_id)
                self.assertEqual(client.post(f'/api/portal/claims/{identifier}/review', json={'outcome': 'approved', 'reason': 'A valid long review reason'}).status_code, 409)
                with app.state.store.session() as session:
                    self.assertIsNotNone(session.get(DecisionRecord, identifier))
                    self.assertEqual(session.get(Assessment, identifier).context['findings']['intake']['summary'], 'Retained historical finding')
                    self.assertEqual(session.get(Assessment, new_id).execution['workflow'], 'claude-intake-first-v3')
                    self.assertEqual(session.get(Ownership, identifier).account_id, session.get(Ownership, new_id).account_id)

    def test_manual_unreadable_document_records_failure_once(self):
        from backend.evidence.uploaded_claim_documents import prepare_attachments, effective_submission
        identifier = self.prepare()
        with self.store.session.begin() as session:
            session.add(Attachment(id='broken-file', assessment_id=identifier, name='broken.pdf', media_type='application/pdf',
                content=base64.b64encode(b'%PDF-1.7 broken fixture').decode()))
        with patch.dict(os.environ, {'CLAIMS_LLM_MODE': 'anthropic'}), patch('backend.agents.claude_investigation.call_model') as calls:
            first = prepare_attachments(self.store, identifier)
            second = prepare_attachments(self.store, identifier)
            calls.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(first[0]['status'], 'unavailable')
        self.assertEqual(effective_submission(self.store, identifier, {'documents': []})['documents'], [])
        with self.store.session() as session:
            self.assertEqual(len(session.scalars(select(AttachmentExtraction)).all()), 1)
