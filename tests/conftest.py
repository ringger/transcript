"""Shared test fixtures and utilities."""

from unittest.mock import MagicMock


def make_openai_response(text="hello", prompt_tokens=10, completion_tokens=5):
    """Build a mock OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp
