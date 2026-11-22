import json
import os
import unittest
from unittest.mock import patch

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import select

import test_agentic
from backend.agents.claude_investigation import call_model
from backend.assurance.claim_audit_export import export_audit, verify_audit
from backend.assurance.investigation_tracing import claim_trace, current_trace, trace_id_for
from backend.storage.sqlite_store import ModelInvocation, TraceSpan


class InvestigationTraceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_agentic.AgenticTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.store = self.fixture.store
        self.identifier = self.fixture.prepare()

    def sdk(self):
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        sdk = Langfuse(public_key=f"pk-lf-{self._testMethodName}", secret_key="sk-lf-trace-test", base_url="http://127.0.0.1:1",
                       span_exporter=exporter, tracer_provider=tracer_provider)
        self.addCleanup(sdk.shutdown)
        return sdk, exporter

    def test_real_langfuse_sdk_exports_one_nested_metadata_trace(self):
        sdk, exporter = self.sdk()
        with patch("backend.assurance.investigation_tracing.langfuse_client", return_value=sdk), \
                patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_CONTENT": "0"}), \
                patch("backend.agents.claude_investigation.call_model", side_effect=self.fixture.provider):
            self.fixture.workflow.run(self.identifier)
        sdk.flush()

        remote = exporter.get_finished_spans()
        root = next(span for span in remote if span.name == "claims.investigation")
        self.assertEqual(f"{root.context.trace_id:032x}", trace_id_for(self.identifier))
        self.assertTrue(root.attributes.get("session.id"))
        self.assertEqual({span.context.trace_id for span in remote}, {root.context.trace_id})
        self.assertFalse(any("langfuse.observation.input" in span.attributes for span in remote))
        self.assertTrue(any(span.name == "claims.tool.retrieve_policy" and span.parent for span in remote))

        package = export_audit(self.store, self.identifier)
        self.assertTrue(verify_audit(package))
        spans = package["trace"]["spans"]
        collectors = [span for span in spans if span["name"] == "context.collect"]
        self.assertEqual(collectors[0]["metadata"]["context_version"], 0)
        self.assertEqual(collectors[0]["summary"]["context_version"], 1)
        self.assertIn("intake", collectors[0]["output"]["snapshot"]["findings"])
        self.assertTrue(all(s["ended_at"] and s["duration_ms"] >= 0 for s in spans))

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
                    args = (self.store, self.identifier, "test", "test system", [{"role": "user", "content": "test input"}],
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

    def test_payload_export_requires_a_synthetic_only_workspace(self):
        assessment = self.fixture.workflow.state(self.identifier)
        with patch.dict(os.environ, {"CLAIMS_LANGFUSE_CAPTURE_CONTENT": "1"}), \
                patch("backend.assurance.investigation_tracing.langfuse_client", return_value=None):
            with claim_trace(self.store, assessment):
                self.assertTrue(current_trace.get().capture_content)
            self.fixture.prepare(1, manual=True)
            with claim_trace(self.store, assessment):
                self.assertFalse(current_trace.get().capture_content)

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
