"""Reusable SDK-shaped fixtures for the primary chat contract tests."""

from __future__ import annotations

from typing import Optional

import gigachat.models as gm

MODEL = "GigaChat-3-Ultra"
CREATED_AT = 1_754_000_000
REQUEST_ID = "request-primary-001"
SESSION_ID = "session-primary-001"
THREAD_ID = "thread-primary-001"
MESSAGE_ID = "message-primary-001"
TOOLS_STATE_ID = "tools-state-primary-001"


def build_official_sdk_server_tool_stream() -> list[gm.PrimaryChatCompletionChunk]:
    """Return the exact named-event payloads stable SDK v0.2.3 tests parse."""
    payloads = [
        {
            "event": "response.message.delta",
            "model": "GigaChat",
            "created_at": "167890456789",
            "messages": [{"role": "assistant", "content": "primary chunk"}],
        },
        {
            "event": "response.tool.completed",
            "model": "GigaChat",
            "created_at": "167890456790",
            "messages": [
                {
                    "role": "reasoning",
                    "content": [
                        {
                            "tool_execution": {
                                "name": "image_generate",
                                "status": "success",
                                "censored": True,
                            }
                        }
                    ],
                }
            ],
        },
        {
            "event": "response.message.done",
            "model": "GigaChat",
            "created_at": "167890456791",
            "finish_reason": "error",
            "tools_state_id": "tools-state-1",
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {
                    "prompt_tokens": 1,
                    "cached_tokens": 0,
                },
                "output_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]
    return [
        gm.PrimaryChatCompletionChunk.model_validate(payload) for payload in payloads
    ]


def _x_headers() -> dict[str, Optional[str]]:
    return {
        "x-request-id": REQUEST_ID,
        "x-session-id": SESSION_ID,
    }


def _usage() -> gm.ChatUsage:
    return gm.ChatUsage(
        input_tokens=13,
        input_tokens_details=gm.ChatUsageInputTokensDetails(
            prompt_tokens=13,
            cached_tokens=5,
        ),
        output_tokens=8,
        total_tokens=21,
    )


def build_plain_text_response() -> gm.ChatCompletionResponse:
    """Return a text-only response with tracing and token metadata."""
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text="Primary response")],
            )
        ],
        message_id=MESSAGE_ID,
        thread_id=THREAD_ID,
        finish_reason="stop",
        usage=_usage(),
        x_headers=_x_headers(),
    )


def build_function_call_response() -> gm.ChatCompletionResponse:
    """Return a client function call with stable continuation state."""
    return gm.ChatCompletionResponse(
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessage(
                role="assistant",
                message_id=MESSAGE_ID,
                tools_state_id=TOOLS_STATE_ID,
                content=[],
                function_call=gm.PrimaryChatFunctionCall(
                    name="lookup_weather",
                    arguments={"city": "Moscow"},
                ),
            )
        ],
        message_id=MESSAGE_ID,
        thread_id=THREAD_ID,
        finish_reason="tool_calls",
        x_headers=_x_headers(),
    )


def build_message_delta_event() -> gm.PrimaryChatCompletionChunk:
    """Return a named text-delta event."""
    return gm.PrimaryChatCompletionChunk(
        event="response.message.delta",
        model=MODEL,
        created_at=CREATED_AT,
        messages=[
            gm.ChatMessageChunk(
                role="assistant",
                message_id=MESSAGE_ID,
                content=[gm.ChatContentPart(text="Primary ")],
            )
        ],
        message_id=MESSAGE_ID,
        thread_id=THREAD_ID,
        x_headers=_x_headers(),
    )


def build_message_done_event() -> gm.PrimaryChatCompletionChunk:
    """Return a metadata-only done event carrying usage and stable IDs."""
    return gm.PrimaryChatCompletionChunk(
        event="response.message.done",
        model=MODEL,
        created_at=CREATED_AT,
        messages=None,
        message_id=MESSAGE_ID,
        thread_id=THREAD_ID,
        tools_state_id=TOOLS_STATE_ID,
        finish_reason="stop",
        usage=_usage(),
        x_headers=_x_headers(),
    )
