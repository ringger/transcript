"""Tests for shared.py — SpeechConfig, LLM client/retry, and utilities."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared import (
    SpeechConfig,
    _NormalizedResponse,
    _convert_messages_to_openai,
    _has_vision_content,
    api_call_with_retry,
    create_llm_client,
    is_up_to_date,
    llm_call_with_retry,
)


# ---------------------------------------------------------------------------
# is_up_to_date
# ---------------------------------------------------------------------------

class TestIsUpToDate:
    def test_output_missing(self, tmp_path):
        output = tmp_path / "out.txt"
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        assert is_up_to_date(output, inp) is False

    def test_output_newer_than_input(self, tmp_path):
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, inp) is True

    def test_input_newer_than_output(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        time.sleep(0.05)
        inp = tmp_path / "in.txt"
        inp.write_text("x")
        assert is_up_to_date(output, inp) is False

    def test_no_inputs(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output) is True

    def test_none_input_ignored(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, None) is True

    def test_missing_input_ignored(self, tmp_path):
        output = tmp_path / "out.txt"
        output.write_text("y")
        missing = tmp_path / "gone.txt"
        assert is_up_to_date(output, missing) is True

    def test_multiple_inputs_one_newer(self, tmp_path):
        old = tmp_path / "old.txt"
        old.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        time.sleep(0.05)
        new = tmp_path / "new.txt"
        new.write_text("z")
        assert is_up_to_date(output, old, new) is False

    def test_multiple_inputs_all_older(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("x")
        b.write_text("x")
        time.sleep(0.05)
        output = tmp_path / "out.txt"
        output.write_text("y")
        assert is_up_to_date(output, a, b) is True


# ---------------------------------------------------------------------------
# DAG integration: checkpoint staleness (uses is_up_to_date)
# ---------------------------------------------------------------------------

class TestCheckpointStaleness:
    """Test that checkpoints are treated as DAG artefacts."""

    def test_fresh_checkpoint_reused(self, tmp_path):
        """A checkpoint newer than all sources should be reused."""
        source = tmp_path / "source.txt"
        source.write_text("input")
        time.sleep(0.05)
        checkpoint = tmp_path / "chunk_000.json"
        checkpoint.write_text(json.dumps({"text": "cached"}))
        assert is_up_to_date(checkpoint, source) is True

    def test_stale_checkpoint_rejected(self, tmp_path):
        """A checkpoint older than a source should be stale."""
        checkpoint = tmp_path / "chunk_000.json"
        checkpoint.write_text(json.dumps({"text": "old"}))
        time.sleep(0.05)
        source = tmp_path / "source.txt"
        source.write_text("updated input")
        assert is_up_to_date(checkpoint, source) is False

    def test_missing_checkpoint_is_stale(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("input")
        checkpoint = tmp_path / "chunk_000.json"
        assert is_up_to_date(checkpoint, source) is False

    def test_assembled_output_depends_on_chunks(self, tmp_path):
        """transcript_merged.txt should be stale if any chunk is newer."""
        chunk_a = tmp_path / "chunk_000.json"
        chunk_b = tmp_path / "chunk_001.json"
        chunk_a.write_text("a")
        chunk_b.write_text("b")
        time.sleep(0.05)
        merged = tmp_path / "transcript_merged.txt"
        merged.write_text("merged")
        assert is_up_to_date(merged, chunk_a, chunk_b) is True

        # Now touch a chunk — merged should be stale
        time.sleep(0.05)
        chunk_b.write_text("b updated")
        assert is_up_to_date(merged, chunk_a, chunk_b) is False


# ---------------------------------------------------------------------------
# api_call_with_retry — Anthropic path
# ---------------------------------------------------------------------------

class TestApiCallWithRetry:
    @pytest.fixture
    def cfg(self, tmp_path):
        return SpeechConfig(url="x", output_dir=tmp_path, local=False)

    def test_success_on_first_try(self, cfg):
        client = MagicMock()
        client.messages.create.return_value = "ok"
        result = api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 1

    @patch("shared.time.sleep")
    def test_retries_on_529(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        client.messages.create.side_effect = [error, error, "ok"]
        result = api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("shared.time.sleep")
    def test_exponential_backoff(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limited"}},
        )
        client.messages.create.side_effect = [error, error, error, "ok"]
        api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [5, 10, 20]

    @patch("shared.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}),
            body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        client.messages.create.side_effect = error
        with pytest.raises(anthropic.APIStatusError):
            api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert client.messages.create.call_count == 5

    def test_non_retryable_error_raises_immediately(self, cfg):
        import anthropic

        client = MagicMock()
        error = anthropic.APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body={"type": "error", "error": {"type": "invalid_request_error", "message": "Bad request"}},
        )
        client.messages.create.side_effect = error
        with pytest.raises(anthropic.APIStatusError):
            api_call_with_retry(client, cfg, model="test", max_tokens=100, messages=[])
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# api_call_with_retry — timeout handling (Anthropic path)
# ---------------------------------------------------------------------------

class TestApiCallWithRetryTimeout:
    @pytest.fixture
    def cfg(self, tmp_path):
        return SpeechConfig(url="x", output_dir=tmp_path,
                            local=False, api_initial_backoff=1, api_max_retries=3)

    @patch("shared.time.sleep")
    def test_retries_on_timeout(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        client.messages.create.side_effect = [
            anthropic.APITimeoutError(request=MagicMock()),
            "ok"
        ]
        result = api_call_with_retry(client, cfg, model="test",
                                     max_tokens=100, messages=[])
        assert result == "ok"
        assert client.messages.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("shared.time.sleep")
    def test_raises_after_max_timeout_retries(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        client.messages.create.side_effect = anthropic.APITimeoutError(
            request=MagicMock())
        with pytest.raises(anthropic.APITimeoutError):
            api_call_with_retry(client, cfg, model="test",
                                max_tokens=100, messages=[])
        assert client.messages.create.call_count == 3

    @patch("shared.time.sleep")
    def test_timeout_backoff_is_exponential(self, mock_sleep, cfg):
        import anthropic

        client = MagicMock()
        client.messages.create.side_effect = [
            anthropic.APITimeoutError(request=MagicMock()),
            anthropic.APITimeoutError(request=MagicMock()),
            "ok"
        ]
        api_call_with_retry(client, cfg, model="test",
                            max_tokens=100, messages=[])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1, 2]  # initial_backoff=1, then doubled

    def test_passes_timeout_from_config(self, cfg):
        client = MagicMock()
        client.messages.create.return_value = "ok"
        api_call_with_retry(client, cfg, model="test",
                            max_tokens=100, messages=[])
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == cfg.api_timeout

    def test_explicit_timeout_overrides_config(self, cfg):
        client = MagicMock()
        client.messages.create.return_value = "ok"
        api_call_with_retry(client, cfg, model="test",
                            max_tokens=100, messages=[], timeout=30.0)
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["timeout"] == 30.0


# ---------------------------------------------------------------------------
# _NormalizedResponse — adapter for OpenAI responses
# ---------------------------------------------------------------------------

class TestNormalizedResponse:
    def _make_openai_response(self, text="hello", prompt_tokens=10, completion_tokens=5):
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

    def test_text_access(self):
        resp = _NormalizedResponse(self._make_openai_response("some text"))
        assert resp.content[0].text == "some text"

    def test_usage_tokens(self):
        resp = _NormalizedResponse(self._make_openai_response(prompt_tokens=42, completion_tokens=7))
        assert resp.usage.input_tokens == 42
        assert resp.usage.output_tokens == 7

    def test_none_content_becomes_empty_string(self):
        resp = _NormalizedResponse(self._make_openai_response(text=None))
        assert resp.content[0].text == ""

    def test_zero_tokens(self):
        resp = _NormalizedResponse(self._make_openai_response(prompt_tokens=0, completion_tokens=0))
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0

    def test_no_usage(self):
        """When usage is None (some providers), should default to 0."""
        raw = self._make_openai_response()
        raw.usage = None
        resp = _NormalizedResponse(raw)
        assert resp.usage.input_tokens == 0
        assert resp.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# _convert_messages_to_openai
# ---------------------------------------------------------------------------

class TestConvertMessagesToOpenai:
    def test_string_content_passthrough(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _convert_messages_to_openai(msgs)
        assert result == msgs

    def test_text_blocks(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_image_block_conversion(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "abc123"
            }}
        ]}]
        result = _convert_messages_to_openai(msgs)
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "describe this"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_image_default_media_type(self):
        msgs = [{"role": "user", "content": [
            {"type": "image", "source": {"data": "xyz"}}
        ]}]
        result = _convert_messages_to_openai(msgs)
        assert "image/png" in result[0]["content"][0]["image_url"]["url"]

    def test_unknown_block_type_passthrough(self):
        msgs = [{"role": "user", "content": [{"type": "custom", "data": "stuff"}]}]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["content"][0] == {"type": "custom", "data": "stuff"}

    def test_empty_messages(self):
        assert _convert_messages_to_openai([]) == []

    def test_mixed_roles(self):
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "hello"},
        ]
        result = _convert_messages_to_openai(msgs)
        assert result[0]["content"] == "you are helpful"
        assert result[1]["content"] == [{"type": "text", "text": "hi"}]
        assert result[2]["content"] == "hello"


# ---------------------------------------------------------------------------
# _has_vision_content
# ---------------------------------------------------------------------------

class TestHasVisionContent:
    def test_no_images(self):
        msgs = [{"role": "user", "content": "text only"}]
        assert _has_vision_content(msgs) is False

    def test_text_blocks_only(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _has_vision_content(msgs) is False

    def test_with_image(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"data": "abc"}}
        ]}]
        assert _has_vision_content(msgs) is True

    def test_image_in_second_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [{"type": "image", "source": {"data": "x"}}]},
        ]
        assert _has_vision_content(msgs) is True

    def test_empty_messages(self):
        assert _has_vision_content([]) is False


# ---------------------------------------------------------------------------
# create_llm_client
# ---------------------------------------------------------------------------

class TestCreateLlmClient:
    def test_local_creates_openai_client(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, local=True)
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_cls)}):
            client = create_llm_client(config)
            mock_cls.assert_called_once_with(
                base_url=config.ollama_base_url, api_key="ollama"
            )

    def test_api_creates_anthropic_client(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, local=False, api_key="sk-test")
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_module}):
            client = create_llm_client(config)
            mock_module.Anthropic.assert_called_once_with(api_key="sk-test")

    def test_api_uses_env_key(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, local=False, api_key=None)
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_module}), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-env"}):
            client = create_llm_client(config)
            mock_module.Anthropic.assert_called_once_with(api_key="sk-env")

    def test_custom_ollama_url(self, tmp_path):
        config = SpeechConfig(url="x", output_dir=tmp_path, local=True,
                              ollama_base_url="http://remote:11434/v1/")
        mock_cls = MagicMock()
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_cls)}):
            create_llm_client(config)
            mock_cls.assert_called_once_with(
                base_url="http://remote:11434/v1/", api_key="ollama"
            )


# ---------------------------------------------------------------------------
# llm_call_with_retry — local (OpenAI-compatible) path
# ---------------------------------------------------------------------------

class TestLlmCallWithRetryLocal:
    @pytest.fixture
    def cfg(self, tmp_path):
        return SpeechConfig(url="x", output_dir=tmp_path, local=True,
                            api_initial_backoff=1, api_max_retries=3)

    def _make_openai_response(self, text="merged text"):
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    def test_success_returns_normalized(self, cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response("hello")
        result = llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert result.content[0].text == "hello"
        assert result.usage.input_tokens == 10
        assert client.chat.completions.create.call_count == 1

    def test_uses_local_model(self, cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response()
        cfg.local_model = "llama3.3"
        llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama3.3"

    def test_vision_messages_use_vision_model(self, cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response()
        cfg.local_vision_model = "llava-custom"
        msgs = [{"role": "user", "content": [
            {"type": "image", "source": {"data": "abc"}}
        ]}]
        llm_call_with_retry(client, cfg, messages=msgs)
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llava-custom"

    def test_max_tokens_passed(self, cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response()
        llm_call_with_retry(client, cfg, max_tokens=2048,
                            messages=[{"role": "user", "content": "hi"}])
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    def test_default_max_tokens(self, cfg):
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response()
        llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @patch("shared.time.sleep")
    def test_retries_on_timeout(self, mock_sleep, cfg):
        from openai import APITimeoutError
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            self._make_openai_response("ok")
        ]
        result = llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert result.content[0].text == "ok"
        assert client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("shared.time.sleep")
    def test_retries_on_429(self, mock_sleep, cfg):
        from openai import APIStatusError
        client = MagicMock()
        error = APIStatusError(
            message="Rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "Rate limited"}},
        )
        client.chat.completions.create.side_effect = [
            error, self._make_openai_response("ok")
        ]
        result = llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert result.content[0].text == "ok"
        assert mock_sleep.call_count == 1

    @patch("shared.time.sleep")
    def test_retries_on_500(self, mock_sleep, cfg):
        from openai import APIStatusError
        client = MagicMock()
        error = APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500, headers={}),
            body={"error": {"message": "Server error"}},
        )
        client.chat.completions.create.side_effect = [
            error, self._make_openai_response("ok")
        ]
        result = llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert result.content[0].text == "ok"

    @patch("shared.time.sleep")
    def test_exponential_backoff_local(self, mock_sleep, cfg):
        from openai import APITimeoutError
        client = MagicMock()
        timeout_err = APITimeoutError(request=MagicMock())
        client.chat.completions.create.side_effect = [
            timeout_err, timeout_err, self._make_openai_response()
        ]
        llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1, 2]

    @patch("shared.time.sleep")
    def test_raises_after_max_retries_local(self, mock_sleep, cfg):
        from openai import APITimeoutError
        client = MagicMock()
        client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with pytest.raises(APITimeoutError):
            llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert client.chat.completions.create.call_count == 3

    def test_non_retryable_error_raises_immediately(self, cfg):
        from openai import APIStatusError
        client = MagicMock()
        error = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body={"error": {"message": "Bad request"}},
        )
        client.chat.completions.create.side_effect = error
        with pytest.raises(APIStatusError):
            llm_call_with_retry(client, cfg, messages=[{"role": "user", "content": "hi"}])
        assert client.chat.completions.create.call_count == 1

    def test_messages_converted_to_openai_format(self, cfg):
        """Verify that Anthropic-style image messages get converted."""
        client = MagicMock()
        client.chat.completions.create.return_value = self._make_openai_response()
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
        ]}]
        llm_call_with_retry(client, cfg, messages=msgs)
        call_kwargs = client.chat.completions.create.call_args[1]
        sent_msgs = call_kwargs["messages"]
        # Should have been converted to OpenAI format
        img_part = sent_msgs[0]["content"][1]
        assert img_part["type"] == "image_url"
        assert "data:image/png;base64,abc" in img_part["image_url"]["url"]
