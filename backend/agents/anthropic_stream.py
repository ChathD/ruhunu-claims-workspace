"""Reassemble streamed Anthropic text, tool inputs and usage while retaining partial output on failure."""

import json
import time
from datetime import datetime, timezone


def stream_events(response, deadline):
    data = []
    for line in response.iter_lines():
        if time.monotonic() >= deadline:
            raise TimeoutError("Claim processing deadline exceeded while streaming")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            yield json.loads("\n".join(data))
            data.clear()


def read_message_stream(response, message, deadline, observation):
    blocks = {}
    tool_inputs = {}
    complete = False
    for event in stream_events(response, deadline):
        match event["type"]:
            case "message_start":
                message.update(event["message"])
                message["content"] = []
            case "content_block_start":
                if observation.completion_start_time is None:
                    observation.completion_start_time = datetime.now(timezone.utc)
                blocks[event["index"]] = event["content_block"]
                message["content"] = [blocks[index] for index in sorted(blocks)]
            case "content_block_delta":
                index = event["index"]
                delta = event["delta"]
                if delta["type"] == "input_json_delta":
                    tool_inputs.setdefault(index, []).append(delta["partial_json"])
                elif delta["type"] == "text_delta":
                    blocks[index]["text"] += delta["text"]
            case "message_delta":
                message.update(event["delta"])
                message.setdefault("usage", {}).update(event.get("usage", {}))
            case "error":
                message["stream_error"] = event.get("error", {})
                raise RuntimeError("Anthropic returned an error during streaming")
            case "message_stop":
                complete = True
                break
    if not complete or message.get("stop_reason") is None:
        raise RuntimeError("Anthropic stream disconnected before the final message")
    if message["stop_reason"] == "tool_use":
        for index, fragments in tool_inputs.items():
            blocks[index]["input"] = json.loads("".join(fragments))
