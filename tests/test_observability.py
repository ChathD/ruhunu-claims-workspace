import base64
import json
import os
from contextlib import nullcontext
from unittest.mock import patch

import httpx
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import select

import test_agentic
from test_generator import claim_draft
from backend.agents.claude_investigation import call_model
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.investigation_tracing import claim_trace, export_app_span, mask_export, operation_trace, trace_id_for, trace_span
from backend.evidence import policy_embeddings
from backend.evidence.policy_wording_index import create_document, index_document, search_document
from backend.generation import synthetic_claim_generator as generator
from backend.storage.sqlite_store import ModelInvocation, TraceSpan


class InvestigationTraceTests(test_agentic.AgenticFixtureTestCase):
    def setUp(self):
        super().setUp()
        self.identifier = self.fixture.prepare()

    def sdk(self):
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        sdk = Langfuse(public_key=f"pk-lf-{self._testMethodName}", secret_key="sk-lf-trace-test", base_url="http://127.0.0.1:1",
                       span_exporter=exporter, tracer_provider=tracer_provider, environment="verification",
                       should_export_span=export_app_span, mask_otel_spans=mask_export)
        self.addCleanup(sdk.shutdown)
        return sdk, exporter

    def test_real_langfuse_sdk_exports_one_nested_metadata_trace(self):
        sdk, exporter = self.sdk()
        with patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_PAYLOADS": "0"}), \
                patch("backend.agents.claude_investigation.call_model", side_effect=self.fixture.provider):
            self.fixture.workflow.run(self.identifier)
        sdk.flush()

        remote = exporter.get_finished_spans()
        root = next(span for span in remote if span.name == "claims.investigation")
        self.assertEqual(f"{root.context.trace_id:032x}", trace_id_for(self.identifier))
        self.assertTrue(root.attributes.get("session.id"))
        self.assertEqual({span.context.trace_id for span in remote}, {root.context.trace_id})
        self.assertEqual(json.loads(root.attributes["langfuse.observation.input"])["feature"], "claim-investigation")
        self.assertIn("route", json.loads(root.attributes["langfuse.observation.output"]))
        self.assertNotIn("langfuse.observation.metadata.final_evaluation", root.attributes)
        self.assertTrue(any(span.name == "claims.tool.retrieve_policy" and span.parent for span in remote))
        self.assertIn("claims.route_recommendation", {span.name for span in remote})

        package = export_audit(self.store, self.identifier)
        self.assertTrue(verify_audit(package))
        spans = package["trace"]["spans"]
        collectors = [span for span in spans if span["name"] == "context.collect"]
        self.assertEqual(collectors[0]["metadata"]["context_version"], 0)
        self.assertEqual(collectors[0]["summary"]["context_version"], 1)
        self.assertIn("intake", collectors[0]["output"]["snapshot"]["findings"])
        self.assertTrue(all(s["ended_at"] and s["duration_ms"] >= 0 for s in spans))

    def test_final_evaluation_input_includes_audited_evidence_on_the_completed_root(self):
        sdk, exporter = self.sdk()
        with patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_PAYLOADS": "1"}), \
                patch("backend.agents.claude_investigation.call_model", side_effect=self.fixture.provider):
            self.fixture.workflow.run(self.identifier)
            with operation_trace("generator.preview", "not-an-investigation") as span:
                span.output = {"status": "complete"}
            sdk.flush()

        spans = exporter.get_finished_spans()
        marker = "langfuse.observation.metadata.final_evaluation"
        eligible = [span for span in spans if span.attributes.get(marker) == '"ready"']
        self.assertEqual([span.name for span in eligible], ["claims.investigation"])
        root = eligible[0]
        request = json.loads(root.attributes["langfuse.observation.input"])["request"]
        result = json.loads(root.attributes["langfuse.observation.output"])["result"]
        package = export_audit(self.store, self.identifier)
        self.assertTrue(verify_audit(package))
        submission = package["submission"]
        self.assertEqual(request["submission"]["loss_date"], submission["loss_date"])
        self.assertEqual(request["submission"]["gross_amount"], submission["gross_amount"])
        self.assertEqual(request["submission"]["documents"][0]["date"], submission["documents"][0]["date"])
        self.assertEqual(request["committed_findings"]["intake"], package["context"]["findings"]["intake"])
        self.assertIn("retrieve_policy", {entry["tool"] for entry in request["retrieved_evidence"]})
        decision = package["decision"]
        self.assertTrue(decision["recommendation"])
        self.assertEqual(result, {key: value for key, value in decision.items() if key != "recommendation"})
        self.assertEqual(next(span for span in package["trace"]["spans"] if span["name"] == "investigation")["input"]["findings"], {})

    def test_stream_is_open_until_finished_and_keeps_partial_output_on_error(self):
        assessment = self.fixture.workflow.state(self.identifier)
        sdk, exporter = self.sdk()
        for fails in (False, True):
            with self.subTest(fails=fails), patch.dict(os.environ, {"CLAIMS_STREAM_MODEL": "1", "ANTHROPIC_API_KEY": "test"}), \
                    patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                    patch("backend.agents.claude_investigation.httpx.Client") as client:
                response = client.return_value.__enter__.return_value.stream.return_value.__enter__.return_value
                response.iter_lines.side_effect = lambda: self.stream(fails)
                with claim_trace(self.store, assessment):
                    packet = {"task": "claim-content@example.invalid", "context": {"version": 3,
                        "findings": {"intake": {"summary": "claim-content@example.invalid"}}},
                        "initial_policy_evidence": [{"result": {"passages": ["claim-content@example.invalid"]}}]}
                    args = (self.store, self.identifier, "r1:coverage", "test system",
                            [{"role": "user", "content": json.dumps(packet)}],
                            [{"name": "submit", "description": "Submit", "input_schema": {"type": "object"}}], {"type": "any"})
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "error during streaming"):
                            call_model(*args)
                    else:
                        result = call_model(*args)
                        self.assertEqual(result["content"][1]["input"], {"ok": True})
                with self.store.session() as session:
                    invocation = session.scalar(select(ModelInvocation).order_by(ModelInvocation.created_at.desc()))
                    span = session.scalar(select(TraceSpan).where(TraceSpan.name == "model.anthropic").order_by(TraceSpan.started_at.desc()))
                self.assertEqual(invocation.response["content"][0]["text"], "partial result")
                self.assertEqual(span.payload["metadata"]["model_invocation_id"], invocation.id)
                self.assertEqual(span.payload["usage"]["cache_read_input_tokens"], 4)
                self.assertIn("completion_started_at", span.payload)
                self.assertEqual(span.error_type, "RuntimeError" if fails else None)
                self.assertEqual(invocation.status, "failed" if fails else "complete")
        sdk.flush()
        generations = [span for span in exporter.get_finished_spans() if span.name == "claims.model.anthropic"]
        self.assertEqual(len(generations), 2)
        self.assertTrue(all(span.attributes.get("langfuse.observation.usage_details") for span in generations))
        for span in generations:
            model_input = json.loads(span.attributes["langfuse.observation.input"])
            self.assertEqual(model_input["prior_findings"], ["intake"])
            self.assertEqual(model_input["input_sections"], ["task", "context", "initial_policy_evidence"])
            self.assertEqual(model_input["policy_passage_count"], 1)
            self.assertNotIn("claim-content@example.invalid", str(span.attributes))

    def test_raw_payloads_and_unrelated_spans_stay_out_of_langfuse(self):
        assessment = self.fixture.workflow.state(self.identifier)
        sdk, exporter = self.sdk()
        with patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_CONTENT": "1", "CLAIMS_LANGFUSE_CAPTURE_PAYLOADS": "0"}), \
                patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk):
            with claim_trace(self.store, assessment):
                with trace_span("tool.sample", content={"document": "claim-content@example.invalid"},
                                metadata={"stage": "test", "body": "claim-content@example.invalid"}) as span:
                    span.output = {"claim_details": "claim-content@example.invalid"}
                    span.summary = {"status": "ready", "private_note": "claim-content@example.invalid"}
                with sdk.start_as_current_observation(name="unrelated", input="claim-content@example.invalid"):
                    pass
                with sdk.start_as_current_observation(name="claims.direct", input={"stage": "test", "body": "claim-content@example.invalid"},
                                                      metadata={"stage": "claim-content@example.invalid"}):
                    pass
        sdk.flush()
        remote = exporter.get_finished_spans()
        self.assertTrue(any(span.name == "claims.tool.sample" for span in remote))
        self.assertFalse(any(span.name == "unrelated" or "claim-content@example.invalid" in str(span.attributes) for span in remote))
        with self.store.session() as session:
            local = session.scalar(select(TraceSpan).where(TraceSpan.name == "tool.sample"))
            self.assertEqual(local.payload["input"]["document"], "claim-content@example.invalid")

    def test_full_payloads_include_model_messages_and_tool_results_without_binary(self):
        assessment = self.fixture.workflow.state(self.identifier)
        sdk, exporter = self.sdk()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Read the fictional attachment."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BINARY_SENTINEL"}},
        ]}]
        response = {"model": "claude-sonnet-4-6", "stop_reason": "tool_use", "usage": {"input_tokens": 19, "output_tokens": 5},
                    "content": [{"type": "tool_use", "name": "submit", "input": {"finding": "Fixture result"}}]}
        with patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_PAYLOADS": "1", "CLAIMS_STREAM_MODEL": "0", "ANTHROPIC_API_KEY": "fixture"}), \
                patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                patch("backend.agents.claude_investigation.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value.json.return_value = response
            with claim_trace(self.store, assessment):
                call_model(self.store, self.identifier, "r1:coverage", "Test system", messages,
                           [{"name": "submit", "input_schema": {"type": "object"}}], {"type": "any"})
                with trace_span("tool.sample", kind="tool", content={"query": "fictional wording"}) as tool:
                    tool.output = {"passages": ["A fictional passage."]}
                with sdk.start_as_current_observation(name="claims.direct", input={"query": "unapproved input"}):
                    pass
            sdk.flush()

        by_name = {span.name: span.attributes for span in exporter.get_finished_spans()}
        model = by_name["claims.model.anthropic"]
        request = json.loads(model["langfuse.observation.input"])["request"]
        self.assertEqual(request["messages"][0]["content"][0], messages[0]["content"][0])
        self.assertEqual(json.loads(model["langfuse.observation.output"])["result"], response)
        self.assertEqual(model["langfuse.observation.model.name"], response["model"])
        self.assertNotIn("BINARY_SENTINEL", str(by_name))
        tool = by_name["claims.tool.sample"]
        self.assertEqual(json.loads(tool["langfuse.observation.input"])["request"], {"query": "fictional wording"})
        self.assertEqual(json.loads(tool["langfuse.observation.output"])["result"], {"passages": ["A fictional passage."]})
        self.assertNotIn("langfuse.observation.input", by_name["claims.direct"])

    def test_preview_and_policy_index_have_distinct_typed_traces(self):
        sdk, exporter = self.sdk()

        def respond(_request):
            return httpx.Response(200, json={"stop_reason": "tool_use", "usage": {"input_tokens": 19, "output_tokens": 5},
                "content": [{"type": "tool_use", "name": generator.GENERATION_TOOL, "input": {"cases": [claim_draft()]}}]})

        def embed(request):
            inputs = json.loads(request.content)["input"]
            return httpx.Response(200, json={"model": policy_embeddings.MODEL, "usage": {"prompt_tokens": 12, "total_tokens": 12},
                "data": [{"index": index, "embedding": vector} for index, vector in enumerate(test_agentic.fake_embeddings(inputs))]})

        client_class = httpx.Client

        def embeddings_client(**kwargs):
            return client_class(transport=httpx.MockTransport(embed), **kwargs)

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "fixture", "CLAIMS_LANGFUSE_CAPTURE_PAYLOADS": "0"}), \
                patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                patch.object(generator, "generation_client", return_value=nullcontext((client, "fixture-model"))), \
                patch("backend.evidence.policy_wording_index.embed_texts", wraps=policy_embeddings.embed_texts), \
                patch.object(policy_embeddings.httpx, "Client", side_effect=embeddings_client):
            self.assertEqual(len(generator.preview_batch(1, "batch-fixture", actor_id="staff-fixture")["fixtures"]), 1)
            identifier = create_document(self.store, title="Test wording", product="MOTOR", version="1", filename="test.pdf",
                media_type="application/pdf", content=base64.b64encode(test_agentic.text_pdf("Private policy sentence.")).decode())
            index_document(self.store, identifier)
            with claim_trace(self.store, self.fixture.workflow.state(self.identifier)):
                with trace_span("tool.search_policy_documents", kind="retriever"):
                    self.assertTrue(search_document(self.store, identifier, "policy search")["passages"])
        sdk.flush()
        by_name = {span.name: span for span in exporter.get_finished_spans()}
        preview, index = by_name["claims.generator.preview"], by_name["claims.policy.index"]
        self.assertNotEqual(preview.context.trace_id, index.context.trace_id)
        generation = by_name["claims.model.draft_claims"]
        self.assertEqual(generation.parent.span_id, preview.context.span_id)
        self.assertTrue(generation.attributes["langfuse.observation.usage_details"])
        self.assertEqual(by_name["claims.policy.embed"].parent.span_id, index.context.span_id)
        embedding_spans = [span for span in exporter.get_finished_spans() if span.name == "claims.model.openai_embeddings"]
        self.assertEqual({span.parent.span_id for span in embedding_spans},
                         {by_name["claims.policy.embed"].context.span_id, by_name["claims.tool.search_policy_documents"].context.span_id})
        self.assertEqual([json.loads(span.attributes["langfuse.observation.usage_details"]) for span in embedding_spans], [{"input": 12}] * 2)
        self.assertEqual(by_name["claims.policy.vector_search"].attributes["langfuse.observation.type"], "retriever")
        self.assertNotIn("Private policy sentence.", str([dict(span.attributes) for span in by_name.values()]))

    def stream(self, fails):
        with self.store.session() as session:
            generation = session.scalar(select(TraceSpan).where(TraceSpan.name == "model.anthropic",
                                                                TraceSpan.ended_at.is_(None)))
            request = session.get(ModelInvocation, generation.payload["metadata"]["model_invocation_id"])
            self.assertEqual(request.payload["system"][-11:], "test system")
            self.assertEqual(request.status, "started")
        events = [
            {"type": "message_start", "message": {"content": [], "usage": {"input_tokens": 7, "cache_read_input_tokens": 4}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial result"}},
            {"type": "content_block_stop", "index": 0},
        ]
        if fails:
            events.append({"type": "error", "error": {"type": "overloaded_error"}})
        else:
            events.extend([
                {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "name": "submit",
                                                                           "id": "tool_test", "input": {}}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ok":'}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "true}"}},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}},
                {"type": "message_stop"},
            ])
        for event in events:
            yield f"data: {json.dumps(event)}"
            yield ""
