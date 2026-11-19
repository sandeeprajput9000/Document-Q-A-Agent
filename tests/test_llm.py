"""Watsonx response compatibility helpers."""

import json

from src.llm import _normalize_watsonx_tool_arguments


def test_normalizes_double_encoded_tool_arguments_without_mutating_response() -> None:
    arguments = {"query": "Longformer", "k": 5}
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_documents",
                                "arguments": json.dumps(json.dumps(arguments)),
                            },
                        }
                    ]
                }
            }
        ]
    }

    normalized = _normalize_watsonx_tool_arguments(response)

    encoded = normalized["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(encoded) == arguments
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        json.dumps(arguments)
    )


def test_keeps_standard_tool_arguments_unchanged() -> None:
    encoded = json.dumps({"query": "Longformer", "k": 5})
    response = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"arguments": encoded}}]}}
        ]
    }

    normalized = _normalize_watsonx_tool_arguments(response)

    assert normalized["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == encoded
