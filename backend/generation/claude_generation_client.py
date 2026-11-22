"""Send structured Claude requests and validate responses for both synthetic generators."""
import logging
import os
import ssl
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.model_settings import settings


BATCH_SIZE = 5
MESSAGES_URL = "https://api.anthropic.com/v1/messages"
LOGGER = logging.getLogger(__name__)


def local_date():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).date()


class GenerationError(ValueError):
    """A generator could not obtain valid output from its provider."""


class GenerationValidationError(GenerationError):
    def __init__(self, issue):
        self.location = issue["loc"]
        self.issue_type = issue["type"]
        path = ".".join(map(str, self.location)) or "batch"
        super().__init__(f"Claude returned an invalid value at {path}. {issue['msg']}.")


class GeneratedData(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


@contextmanager
def generation_client():
    key = os.getenv("ANTHROPIC_API_KEY")
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


def post_generation(client, payload):
    try:
        return client.post(MESSAGES_URL, json=payload)
    except (httpx.ConnectTimeout, httpx.ConnectError) as error:
        if certificate_verification_failed(error):
            raise
        # The HTTP request has not been sent when connection establishment fails.
        return client.post(MESSAGES_URL, json=payload)


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


def request_batch(client, model, prompt, tool_name, schema, count):
    input_schema = schema.model_json_schema()
    input_schema["properties"]["cases"].update(minItems=count, maxItems=count)
    try:
        response = post_generation(client, {
            "model": model, "max_tokens": 6000,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": tool_name, "description": "Submit the generated records", "input_schema": input_schema}],
            "tool_choice": {"type": "tool", "name": tool_name},
        })
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        response = error.response
        if response.status_code == 400 and "credit balance is too low" in response.text.lower():
            raise GenerationError("Anthropic API credits are too low to generate records. Check your Anthropic billing balance.") from error
        raise GenerationError(f"The generation provider returned HTTP {response.status_code}.") from error
    except httpx.RequestError as error:
        message = generation_connection_error(error)
        LOGGER.warning("Generation %s failed (%s): %s", tool_name, type(error).__name__, message)
        raise GenerationError(message) from error
    try:
        data = response.json()
    except ValueError as error:
        raise GenerationError("The generation provider returned unreadable JSON.") from error
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        raise GenerationError("The generation provider returned an unexpected response format.")
    if data.get("stop_reason") == "max_tokens":
        raise GenerationError("Claude reached its response limit before completing the generated records.")
    calls = [block for block in data["content"] if isinstance(block, dict) and block.get("type") == "tool_use"]
    if data.get("stop_reason") != "tool_use" or len(calls) != 1 or calls[0].get("name") != tool_name:
        raise GenerationError("Claude did not submit a complete batch through the required generation tool.")
    try:
        batch = schema.model_validate(calls[0].get("input"))
    except ValidationError as error:
        issue = error.errors(include_input=False, include_url=False, include_context=False)[0]
        raise GenerationValidationError(issue) from error
    if len(batch.cases) != count:
        raise GenerationError(f"Claude returned {len(batch.cases)} records when {count} were requested.")
    return batch.cases
