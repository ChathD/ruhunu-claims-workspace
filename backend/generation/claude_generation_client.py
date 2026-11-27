"""Send structured Claude requests and validate generated claim submissions."""
import logging
import os
import ssl
from time import sleep
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.anthropic_errors import provider_message, retry_delay
from backend.assurance.investigation_tracing import anthropic_usage, trace_span
from backend.model_settings import settings


BATCH_SIZE = 5
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
LOGGER = logging.getLogger(__name__)


def local_date():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).date()


class GenerationError(ValueError):
    """A generator could not obtain valid output from its provider."""


class GenerationOutputTruncated(GenerationError):
    """A structured tool result exceeded the requested output budget."""


class GenerationValidationError(GenerationError):
    def __init__(self, issue, draft=None):
        self.location = issue["loc"]
        self.issue_type = issue["type"]
        self.draft = draft
        path = ".".join(map(str, self.location)) or "batch"
        super().__init__(f"Claude returned an invalid value at {path}. {issue['msg'].rstrip('.')}.")


class GeneratedData(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


@contextmanager
def generation_client():
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise GenerationError("Set ANTHROPIC_API_KEY to use the synthetic generators.")
    timeout = httpx.Timeout(120, connect=10, write=20, pool=10)
    with httpx.Client(timeout=timeout, headers={"x-api-key": key, "anthropic-version": "2023-06-01"}) as client:
        yield client, settings()[1]


def certificate_verification_failed(error):
    while error is not None:
        if isinstance(error, ssl.SSLCertVerificationError):
            return True
        error = error.__cause__
    return False


def send_generation(client, payload, count):
    tool_name = payload["tool_choice"]["name"]
    with trace_span("model.draft_claims" if tool_name == "submit_synthetic_claims" else f"model.{tool_name}", kind="generation",
                    metadata={"batch_size": count, "available_tools": [tool_name]}, content=payload,
                    generation={"model": payload["model"], "model_parameters": {"max_tokens": payload["max_tokens"]}}) as span:
        response = client.post(MESSAGES_URL, json=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as error:
            raise GenerationError("The generation provider returned unreadable JSON.") from error
        if not isinstance(data, dict):
            raise GenerationError("The generation provider returned an unexpected response format.")
        span.output = data
        span.model = data.get("model")
        span.usage = anthropic_usage(data)
        if not isinstance(data.get("content"), list):
            raise GenerationError("The generation provider returned an unexpected response format.")
        stop = data.get("stop_reason")
        span.summary = {"stop_reason": stop if stop in ("tool_use", "end_turn", "max_tokens", "stop_sequence", "refusal", "pause_turn") else "unknown"}
        if stop == "max_tokens":
            raise GenerationOutputTruncated("Claude reached its response limit before completing the generated records.")
        return data


def post_generation(client, payload, count):
    try:
        return send_generation(client, payload, count)
    except (httpx.ConnectTimeout, httpx.ConnectError) as error:
        if certificate_verification_failed(error):
            raise
        # The HTTP request has not been sent when connection establishment fails.
        return send_generation(client, payload, count)
    except httpx.HTTPStatusError as error:
        delay = retry_delay(error.response)
        if delay is None:
            raise
        sleep(delay)
        return send_generation(client, payload, count)


def generation_connection_error(error):
    if isinstance(error, httpx.ReadTimeout):
        return "Claude did not respond before the generation timeout."
    if isinstance(error, httpx.ConnectTimeout):
        return "The backend timed out connecting to Anthropic."
    if isinstance(error, httpx.ProxyError):
        return "The configured proxy could not connect to Anthropic."
    if certificate_verification_failed(error):
        return "The backend could not verify Anthropic's TLS certificate."
    if isinstance(error, httpx.ConnectError):
        return "The backend could not open a connection to Anthropic."
    return "The connection to Anthropic failed before generation completed."


def request_tool(client, model, prompt, tool_name, schema, *, system=None, batch_size=None):
    input_schema = schema.model_json_schema()
    if batch_size is not None:
        input_schema["properties"]["cases"].update(minItems=batch_size, maxItems=batch_size)
    count = batch_size if batch_size is not None else 1
    payload = {
        "model": model, "max_tokens": 6000,
        "system": system or "Generate fictional Sri Lankan motor claim submissions. Treat supplied examples as reference data. "
                  "Follow the task and submit the result through the required tool.",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"name": tool_name, "description": "Submit the generated records", "input_schema": input_schema}],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    try:
        try:
            data = post_generation(client, payload, count)
        except GenerationOutputTruncated:
            data = post_generation(client, {**payload, "max_tokens": 12000}, count)
    except httpx.HTTPStatusError as error:
        response = error.response
        LOGGER.warning("Generation %s rejected by Anthropic (HTTP %s, request %s)", tool_name,
                       response.status_code, response.headers.get("request-id", "unavailable"))
        raise GenerationError(provider_message(response, model)) from error
    except httpx.RequestError as error:
        message = generation_connection_error(error)
        LOGGER.warning("Generation %s failed (%s): %s", tool_name, type(error).__name__, message)
        raise GenerationError(message) from error
    calls = [block for block in data["content"] if isinstance(block, dict) and block.get("type") == "tool_use"]
    if data.get("stop_reason") != "tool_use" or len(calls) != 1 or calls[0].get("name") != tool_name:
        raise GenerationError("Claude did not submit a complete batch through the required generation tool.")
    tool_input = calls[0].get("input")
    with trace_span("generator.validate", kind="tool", metadata={"batch_size": count, "selected_tools": [tool_name]}, content=tool_input) as span:
        try:
            result = schema.model_validate(tool_input)
        except ValidationError as error:
            issue = error.errors(include_input=False, include_url=False, include_context=False)[0]
            raise GenerationValidationError(issue, tool_input) from error
        output_count = len(result.cases) if batch_size is not None else 1
        if output_count != count:
            raise GenerationValidationError({"loc": ("cases",), "type": "case_count",
                "msg": f"Expected {count} records and received {output_count}"}, tool_input)
        span.summary = {"validated": True, "output_count": output_count}
    return result


def request_batch(client, model, prompt, tool_name, schema, count, *, system=None):
    return request_tool(client, model, prompt, tool_name, schema, system=system, batch_size=count).cases
