"""Classify safe retry delays and user-facing messages for Claude HTTP failures."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite

from httpx import HTTPStatusError


class AnthropicRequestError(HTTPStatusError):
    """Keep the HTTP response available for retries while surfacing an actionable message."""

    def __init__(self, error, model):
        super().__init__(provider_message(error.response, model), request=error.request, response=error.response)


def retry_delay(response):
    status = response.status_code
    if status not in {408, 409, 429} and not 500 <= status < 600:
        return None
    header = response.headers.get("retry-after")
    if not header:
        return 1.0
    try:
        delay = float(header)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(header)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 1.0
    # Defer longer rate limits to the next request instead of tying up a portal request.
    return max(delay, 0.0) if isfinite(delay) and delay <= 5.0 else None


def provider_message(response, model):
    status = response.status_code
    if status == 400 and "credit balance is too low" in response.text.lower():
        return "Anthropic API credits are too low. Add credits in Claude Console, then retry."
    if status == 402:
        return "Anthropic reported a billing problem. Check billing in Claude Console, then retry."
    if status == 401:
        return "Anthropic rejected the configured API key. Check ANTHROPIC_API_KEY."
    if status == 403:
        return "The Anthropic account does not have permission to make this Claude request."
    if status == 404:
        return f"The configured Claude model ({model}) is unavailable to this account. Check ANTHROPIC_MODEL."
    if status == 413:
        return "The source is too large for this Claude request. Use a smaller wording PDF or generate fewer records."
    if status == 429:
        return "Anthropic is rate limiting this workspace. Wait briefly and retry."
    if status >= 500:
        return f"Claude is temporarily unavailable (HTTP {status}). Retry in a moment."
    reference = response.headers.get("request-id")
    request_id = f", request {reference}" if reference else ""
    return f"Anthropic rejected the Claude request (HTTP {status}{request_id})."
