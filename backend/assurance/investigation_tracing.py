"""Store investigation spans locally and optionally export their hierarchy to Langfuse."""

import logging
import os
import time
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps

from sqlalchemy import select

from backend.storage.sqlite_store import Claim, InvestigationRevision, Ownership, TraceSpan, digest, now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trace:
    store: object
    assessment_id: str
    trace_id: str
    session_id: str
    user_id: str | None
    capture_content: bool
    client: object = None


@dataclass
class Observation:
    output: object = None
    summary: dict | None = None
    error_type: str | None = None
    usage: dict | None = None
    completion_start_time: object = None


current_trace = ContextVar("claim_trace", default=None)
current_span = ContextVar("claim_span", default=None)


def trace_id_for(assessment_id):
    return digest(["claims-triage", assessment_id])[:32]


@lru_cache(maxsize=1)
def configured_client(base_url, public_key, secret_key):
    from langfuse import Langfuse

    return Langfuse(base_url=base_url, public_key=public_key, secret_key=secret_key)


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
    try:
        with trace.store.session.begin() as session:
            if initial:
                session.add(TraceSpan(id=identifier, assessment_id=trace.assessment_id, trace_id=trace.trace_id, **fields))
            elif row := session.get(TraceSpan, identifier):
                for key, value in fields.items():
                    setattr(row, key, value)
    except Exception:
        logger.warning("Could not persist investigation span %s", identifier)


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
    try:
        if client := trace.client:
            options = {"name": f"claims.{name}", "as_type": kind,
                       "metadata": {**metadata, "local_span_id": identifier}}
            if trace.capture_content and content is not None:
                options["input"] = content
            if generation:
                options.update(generation)
            if parent is None:
                options["trace_context"] = {"trace_id": trace.trace_id}
            remote = stack.enter_context(client.start_as_current_observation(**options))
            if parent is None:
                from langfuse import propagate_attributes

                stack.enter_context(propagate_attributes(session_id=trace.session_id, user_id=trace.user_id,
                                                         trace_name="claims.investigation"))
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
            try:
                update = {"output": result.output if trace.capture_content and result.output is not None else result.summary}
                if result.error_type:
                    update.update(level="ERROR", status_message=result.error_type)
                if generation and result.usage:
                    update["usage_details"] = result.usage
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
    capture_content = False
    try:
        with store.session() as session:
            seen = {original}
            while parent := session.scalar(select(InvestigationRevision.source_id).where(InvestigationRevision.target_id == original)):
                if parent in seen:
                    break
                seen.add(parent)
                original = parent
            owner = session.get(Ownership, assessment.id)
            owner_id = digest(["claim-owner", owner.account_id])[:32] if owner else None
            if (os.getenv("CLAIMS_LANGFUSE_CAPTURE_CONTENT") == "1"
                    and assessment.context.get("environment") == "synthetic"):
                # Historical retrieval can compare generated and manually submitted claims.
                mixed_data = session.scalar(select(Claim.id).where(
                    Claim.payload["source"].as_string().is_distinct_from("Synthetic generator")).limit(1))
                capture_content = mixed_data is None
    except Exception:
        logger.warning("Could not resolve the investigation trace session")
    client = None
    try:
        client = langfuse_client()
    except Exception:
        logger.warning("Could not initialize Langfuse; using the local investigation trace")
    trace = Trace(store, assessment.id, trace_id_for(assessment.id), digest(["claim-session", original])[:32],
                  owner_id, capture_content, client)
    token = current_trace.set(trace)
    try:
        with trace_span("investigation", kind="chain",
                        metadata={"run_id": assessment.id, "session_id": trace.session_id, "user_id": trace.user_id,
                                  "status": assessment.status, "context_version": assessment.version,
                                  "environment": assessment.context.get("environment"), "mode": assessment.execution.get("mode"),
                                  "workflow": assessment.execution.get("workflow")},
                        content=assessment.context) as span:
            yield span
    finally:
        current_trace.reset(token)


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
