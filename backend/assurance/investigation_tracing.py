"""Store investigation spans locally and optionally export their hierarchy to Langfuse."""

import json
import logging
import os
import re
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps

from sqlalchemy import select

from backend.claim_schemas import AUTHORITY
from backend.storage.sqlite_store import Claim, InvestigationRevision, Ownership, ToolCall, TraceSpan, WorkflowEvent, digest, now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trace:
    store: object
    assessment_id: str | None
    trace_id: str
    session_id: str
    user_id: str | None
    feature: str
    client: object = None


@dataclass
class Observation:
    output: object = None
    summary: dict | None = None
    error_type: str | None = None
    usage: dict | None = None
    model: str | None = None
    completion_start_time: object = None
    final_investigation_context: dict | None = None


current_trace = ContextVar("claim_trace", default=None)
current_span = ContextVar("claim_span", default=None)


def trace_id_for(assessment_id):
    return digest(["claims-triage", assessment_id])[:32]


TELEMETRY_FIELDS = {
    "available", "available_tools", "batch_size", "candidate_count", "changed_context_keys", "chunk_count", "context_hash",
    "context_sections", "context_version", "corrections_pending", "document_kind", "environment", "feature",
    "final_evaluation", "finding_statuses", "findings", "input_sections", "loss_type", "message_count",
    "mode", "model_invocation_id", "next_node", "ocr_page_count", "output_count", "page", "page_count", "pages", "phase",
    "policy_passage_count", "prior_findings", "prompt_version", "reason_codes", "request_hash", "result_count", "result_hash",
    "role", "round", "route",
    "run_id", "selected_specialists", "selected_tools", "source", "stage", "status", "statuses", "stop_reason",
    "topic", "turn", "validated", "workflow",
}
SDK_METADATA_FIELDS = {"local_span_id", "name", "version"}
IO_ATTRIBUTES = {"langfuse.observation.input", "langfuse.observation.output",
                 "langfuse.trace.input", "langfuse.trace.output"}
MACHINE_VALUE = re.compile(r"[a-zA-Z0-9_.:/ -]{1,112}\Z")


def safe_telemetry_value(value):
    return type(value) in {bool, int} or isinstance(value, str) and MACHINE_VALUE.fullmatch(value)


def telemetry_fields(values):
    """Project code-owned telemetry fields without forwarding claim or model payloads."""
    return {key: value for key, value in (values or {}).items() if key in TELEMETRY_FIELDS and
            (safe_telemetry_value(value) or isinstance(value, list) and len(value) <= 40
             and all(map(safe_telemetry_value, value)))}


def capture_payloads():
    return os.getenv("CLAIMS_LANGFUSE_CAPTURE_PAYLOADS") == "1"


def without_attachments(value):
    if isinstance(value, dict):
        if value.get("type") == "base64" and "data" in value:
            return {"type": "attachment_omitted", "media_type": value.get("media_type")}
        return {key: without_attachments(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [without_attachments(item) for item in value]
    if isinstance(value, bytes):
        return "[attachment omitted]"
    if isinstance(value, str) and value.startswith("data:"):
        comma = value.find(",")
        if comma != -1 and value.endswith(";base64", 0, comma):
            return "[attachment omitted]"
    return value


def mask_export(*, params):
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    include_payloads = capture_payloads()
    for identifier, span in params.spans.items():
        local_id = span.attributes.get("langfuse.observation.metadata.local_span_id", "")
        own_payload = include_payloads and isinstance(local_id, str) and re.fullmatch(r'"[0-9a-f]{32}"', local_id)
        remove, replace = [], {}
        for key, value in span.attributes.items():
            prefix = next((prefix for prefix in ("langfuse.observation.metadata.", "langfuse.trace.metadata.")
                           if key.startswith(prefix)), None)
            if prefix is None and key not in IO_ATTRIBUTES:
                continue
            if prefix is None and own_payload:
                continue
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                decoded = None
            if prefix is not None:
                field = key.removeprefix(prefix)
                if field in SDK_METADATA_FIELDS and safe_telemetry_value(decoded):
                    continue
                if not telemetry_fields({field: decoded}):
                    remove.append(key)
            else:
                if isinstance(decoded, dict) and (safe := telemetry_fields(decoded)):
                    if safe != decoded:
                        replace[key] = json.dumps(safe)
                else:
                    remove.append(key)
        if remove or replace:
            patches[identifier] = OtelSpanPatch(delete_attributes=tuple(remove), set_attributes=replace)
    return MaskOtelSpansResult(span_patches=patches)


@lru_cache(maxsize=1)
def configured_client(base_url, public_key, secret_key):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider

    return Langfuse(base_url=base_url, public_key=public_key, secret_key=secret_key,
                    environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
                    release=os.getenv("LANGFUSE_RELEASE"), tracer_provider=TracerProvider(),
                    should_export_span=export_app_span,
                    mask_otel_spans=mask_export)


def export_app_span(span):
    from langfuse.span_filter import is_langfuse_span

    return is_langfuse_span(span) and span.name.startswith("claims.")


def langfuse_client():
    if os.getenv("CLAIMS_ENABLE_LANGFUSE") != "1":
        return None
    config = [os.getenv(key, "").strip() for key in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")]
    if not all(config):
        logger.warning("Langfuse enabled without an explicit endpoint and both project keys")
        return None
    return configured_client(*config)


def flush_langfuse():
    if not configured_client.cache_info().currsize:
        return
    try:
        if client := langfuse_client():
            client.flush()
    except Exception:
        logger.warning("Could not flush Langfuse observations")


def save_span(trace, identifier, fields, *, initial=False):
    if trace.assessment_id is None:
        return
    try:
        with trace.store.session.begin() as session:
            if initial:
                session.add(TraceSpan(id=identifier, assessment_id=trace.assessment_id, trace_id=trace.trace_id, **fields))
            elif row := session.get(TraceSpan, identifier):
                for key, value in fields.items():
                    setattr(row, key, value)
    except Exception:
        logger.warning("Could not persist investigation span %s", identifier)


def final_evaluation_input(trace, context):
    """Use evidence already recorded during this investigation."""
    evidence_tools = (
        "read_submission", "read_coverage_requirements", "find_duplicates", "retrieve_policy",
        "search_policy_documents", "read_policy_page", "retrieve_comparable_claims",
        "retrieve_vehicle_market_value", "verify_historical_leads",
    )
    with trace.store.session() as session:
        calls = session.scalars(select(ToolCall).where(
            ToolCall.assessment_id == trace.assessment_id, ToolCall.tool.in_(evidence_tools)
        ).order_by(ToolCall.id)).all()
    submission = next((call.response for call in reversed(calls) if call.tool == "read_submission"), None)
    if not submission or not context.get("findings", {}).get("intake"):
        return None
    evidence = {(call.tool, call.response_hash): call.response for call in calls if call.tool != "read_submission"}
    return {
        "submission": {field: submission[field] for field in (
            "loss_type", "loss_date", "reported_date", "gross_amount", "currency", "policy_id", "registration",
            "source", "narrative", "vehicle", "documents", "additional_costs", "unstructured_inputs", "uploaded_evidence",
        ) if field in submission},
        "policy_schedule": context.get("policy_schedule"),
        "selected_policy": context.get("selected_policy"),
        "committed_findings": context["findings"],
        "skipped_stages": context.get("skipped", {}),
        "retrieved_evidence": [{"tool": tool, "response": response} for (tool, _), response in evidence.items()],
        "publication_controls": {
            "gross_authority_LKR": str(AUTHORITY),
            "deadline_exceeded": context.get("deadline_exceeded", False),
            "all_routes_are_triage_only": True,
        },
    }


@contextmanager
def trace_span(name, *, kind="span", metadata=None, content=None, retain_content=True, generation=None):
    result = Observation()
    trace = current_trace.get()
    if trace is None:
        yield result
        return

    identifier, parent = uuid.uuid4().hex, current_span.get()
    metadata = metadata or {}
    payload = {"metadata": metadata}
    if retain_content and content is not None:
        payload["input"] = content
    started = time.perf_counter()
    save_span(trace, identifier, {"parent_id": parent, "name": name, "kind": kind, "payload": payload}, initial=True)

    stack = ExitStack()
    remote = None
    include_payloads = False
    try:
        if client := trace.client:
            include_payloads = capture_payloads()
            safe_input = telemetry_fields(metadata)
            remote_input = ({"context": safe_input, "request": without_attachments(content)}
                            if include_payloads and content is not None else safe_input)
            options = {"name": f"claims.{name}", "as_type": kind,
                       "metadata": {**safe_input, "local_span_id": identifier}, "input": remote_input or None}
            if generation:
                options.update(generation)
            if parent is None:
                options["trace_context"] = {"trace_id": trace.trace_id}
            remote = stack.enter_context(client.start_as_current_observation(**options))
            if parent is None:
                from langfuse import propagate_attributes

                stack.enter_context(propagate_attributes(session_id=trace.session_id, user_id=trace.user_id,
                    trace_name=f"claims.{name}", tags=[trace.feature],
                    metadata={"feature": trace.feature}, version=safe_input.get("workflow")))
    except Exception:
        logger.warning("Could not start Langfuse observation %s", name)

    token = current_span.set(identifier)
    try:
        yield result
    except BaseException as error:
        result.error_type = type(error).__name__
        payload = {**payload, "error_message": str(error)[:500]}
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000)
        if retain_content and result.output is not None:
            payload = {**payload, "output": result.output}
        if result.summary is not None:
            payload = {**payload, "summary": result.summary}
        if result.usage:
            payload = {**payload, "usage": result.usage}
        if result.completion_start_time is not None:
            payload = {**payload, "completion_started_at": result.completion_start_time.isoformat()}
        save_span(trace, identifier, {"payload": payload, "ended_at": now(),
                                      "duration_ms": duration_ms, "error_type": result.error_type})
        if remote is not None:
            final_input = None
            if include_payloads and result.final_investigation_context is not None and not result.error_type:
                try:
                    final_input = final_evaluation_input(trace, result.final_investigation_context)
                except Exception:
                    logger.warning("Could not assemble final evaluation input for %s", name)
            try:
                safe_output = telemetry_fields(result.summary)
                output = result.output
                if final_input:
                    output = {key: value for key, value in output.items() if key != "recommendation"}
                remote_output = ({"summary": safe_output, "result": without_attachments(output)}
                                 if include_payloads and output is not None else safe_output)
                update = {"output": remote_output or None}
                if final_input:
                    update["input"] = {"context": telemetry_fields(metadata), "request": without_attachments(final_input)}
                    update["metadata"] = {"final_evaluation": "ready"}
                if result.error_type:
                    update.update(level="ERROR", status_message=result.error_type)
                if generation and result.usage:
                    update["usage_details"] = result.usage
                if generation and isinstance(result.model, str) and MACHINE_VALUE.fullmatch(result.model):
                    update["model"] = result.model
                if generation and result.completion_start_time is not None:
                    update["completion_start_time"] = result.completion_start_time
                remote.update(**update)
            except Exception:
                logger.warning("Could not update Langfuse observation %s", name)
        try:
            # Passing the original exception to the SDK can export evidence in its message.
            stack.close()
        except Exception:
            logger.warning("Could not end Langfuse observation %s", name)
        finally:
            current_span.reset(token)


@contextmanager
def claim_trace(store, assessment):
    original = assessment.id
    owner_id = None
    loss_type = None
    try:
        with store.session() as session:
            seen = {original}
            while parent := session.scalar(select(InvestigationRevision.source_id).where(InvestigationRevision.target_id == original)):
                if parent in seen:
                    break
                seen.add(parent)
                original = parent
            owner = session.get(Ownership, assessment.id)
            selection = session.scalar(select(WorkflowEvent).where(WorkflowEvent.assessment_id == original,
                WorkflowEvent.kind == "policy_selected").order_by(WorkflowEvent.id).limit(1))
            principal = owner.account_id if owner else selection.payload.get("staff_id") if selection else None
            owner_id = digest(["claims-user", principal])[:32] if principal else None
            claim = session.get(Claim, assessment.claim_id)
            loss_type = claim.payload.get("loss_type") if claim else None
    except Exception:
        logger.warning("Could not resolve the investigation trace session")
    trace = Trace(store, assessment.id, trace_id_for(assessment.id), digest(["claim-session", original])[:32],
                  owner_id, "claim-investigation", export_client())
    selected = assessment.context.get("selected_policy") or {}
    with trace_root(trace, "investigation", metadata={
            "run_id": assessment.id, "session_id": trace.session_id, "user_id": trace.user_id,
            "status": assessment.status, "context_version": assessment.version, "loss_type": loss_type,
            "document_kind": selected.get("document_kind"), "environment": assessment.context.get("environment"),
            "mode": assessment.execution.get("mode"), "workflow": assessment.execution.get("workflow")},
            content=assessment.context) as span:
        yield span


def export_client():
    try:
        return langfuse_client()
    except Exception:
        logger.warning("Could not initialize Langfuse")
        return None


@contextmanager
def operation_trace(name, identifier, *, metadata=None, actor_id=None, content=None):
    if (client := export_client()) is None:
        yield Observation()
        return
    user_id = digest(["claims-user", actor_id])[:32] if actor_id else None
    trace = Trace(None, None, digest([name, identifier])[:32], digest(["operation", name, identifier])[:32],
                  user_id, name.replace(".", "-"), client)
    with trace_root(trace, name, metadata={"run_id": identifier, **(metadata or {})}, content=content) as span:
        yield span


@contextmanager
def trace_root(trace, name, *, metadata, content=None):
    token = current_trace.set(trace)
    previous_span = current_span.set(None)
    try:
        with trace_span(name, kind="chain", metadata={"feature": trace.feature, **metadata}, content=content) as span:
            yield span
    finally:
        current_span.reset(previous_span)
        current_trace.reset(token)


def anthropic_usage(data):
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {target: usage[source] for source, target in (
        ("input_tokens", "input"), ("output_tokens", "output"),
        ("cache_read_input_tokens", "cache_read_input_tokens"), ("cache_creation_input_tokens", "cache_creation_input_tokens"))
        if type(usage.get(source)) is int and usage[source] >= 0}


def context_summary(context):
    return {"context_version": context.get("version"), "context_hash": digest(context),
            "findings": sorted(context.get("findings", {}))}


def observe_node(name, *, kind="span", role=None):
    """Trace the graph node's input and output without changing checkpoint state."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            if current_trace.get() is None:
                return function(*args, **kwargs)
            state = kwargs["state"] if "state" in kwargs else args[-1]
            snapshot = state.get("snapshot", {})
            tasks = state.get("tasks", {})
            metadata = {"round": state.get("round"), **context_summary(snapshot)}
            content = {"context": snapshot, "tasks": tasks}
            if role is not None:
                metadata["role"] = role
                content["tasks"] = {role: tasks.get(role)}
            with trace_span(name, kind=kind, metadata=metadata, content=content) as span:
                result = function(*args, **kwargs)
                span.output = result
                if isinstance(result, str):
                    span.summary = {"next_node": result}
                else:
                    current = result.get("snapshot", snapshot)
                    span.summary = context_summary(current)
                    span.summary["changed_context_keys"] = sorted(
                        key for key in snapshot.keys() | current.keys() if snapshot.get(key) != current.get(key)
                    )
                    if "tasks" in result:
                        span.summary["selected_specialists"] = sorted(result["tasks"])
                    if "phase" in result:
                        span.summary["phase"] = result["phase"]
                    if role is not None:
                        findings = [item["finding"] for item in result.get("results", {}).values()]
                        span.summary["finding_statuses"] = [item["status"] for item in findings]
                        span.error_type = next((item["details"]["error_type"] for item in findings
                                                if item["details"].get("error_type")), None)
                    if current.get("investigation_error") or result.get("phase") == "interrupted":
                        span.error_type = span.error_type or "InvestigationInterrupted"
                return result
        return run
    return decorate
