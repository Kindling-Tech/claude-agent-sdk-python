"""Tests for environment variable resolution with options.env precedence."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from claude_agent_sdk._internal.env import resolve_env


class TestResolveEnv:
    """Unit tests for resolve_env function."""

    def test_options_env_wins_over_os_environ(self) -> None:
        """options.env value takes precedence over os.environ."""
        options_env = {"MY_VAR": "from_options"}
        with patch.dict(os.environ, {"MY_VAR": "from_environ"}):
            result = resolve_env("MY_VAR", options_env, "default_val")
        assert result == "from_options"

    def test_os_environ_fallback(self) -> None:
        """Falls back to os.environ when key not in options_env."""
        options_env = {"OTHER_VAR": "irrelevant"}
        with patch.dict(os.environ, {"MY_VAR": "from_environ"}):
            result = resolve_env("MY_VAR", options_env, "default_val")
        assert result == "from_environ"

    def test_default_fallback(self) -> None:
        """Falls back to default when key not in options_env or os.environ."""
        options_env = {"OTHER_VAR": "irrelevant"}
        env_patch = {k: v for k, v in os.environ.items() if k != "MY_VAR"}
        with patch.dict(os.environ, env_patch, clear=True):
            result = resolve_env("MY_VAR", options_env, "default_val")
        assert result == "default_val"

    def test_none_options_env(self) -> None:
        """Works correctly when options_env is None."""
        with patch.dict(os.environ, {"MY_VAR": "from_environ"}):
            result = resolve_env("MY_VAR", None, "default_val")
        assert result == "from_environ"

    def test_none_options_env_default_fallback(self) -> None:
        """Falls back to default when options_env is None and key not in os.environ."""
        env_patch = {k: v for k, v in os.environ.items() if k != "MY_VAR"}
        with patch.dict(os.environ, env_patch, clear=True):
            result = resolve_env("MY_VAR", None, "default_val")
        assert result == "default_val"

    def test_empty_options_env(self) -> None:
        """Falls back to os.environ when options_env is empty dict."""
        with patch.dict(os.environ, {"MY_VAR": "from_environ"}):
            result = resolve_env("MY_VAR", {}, "default_val")
        assert result == "from_environ"

    def test_empty_string_value_in_options_env(self) -> None:
        """Empty string in options_env is a valid value (truthy check on 'in', not value)."""
        options_env = {"MY_VAR": ""}
        with patch.dict(os.environ, {"MY_VAR": "from_environ"}):
            result = resolve_env("MY_VAR", options_env, "default_val")
        assert result == ""


class TestStreamCloseTimeoutResolution:
    """Integration-style tests for CLAUDE_CODE_STREAM_CLOSE_TIMEOUT resolution."""

    def test_client_uses_options_env_for_stream_close_timeout(self) -> None:
        """ClaudeSDKClient resolves CLAUDE_CODE_STREAM_CLOSE_TIMEOUT from options.env."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            env={"CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "30000"}
        )
        # resolve_env should pick up the options.env value
        result = resolve_env(
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", options.env, "60000"
        )
        assert result == "30000"

    def test_stream_close_timeout_os_environ_fallback(self) -> None:
        """Falls back to os.environ for CLAUDE_CODE_STREAM_CLOSE_TIMEOUT."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions()  # empty env
        with patch.dict(os.environ, {"CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "45000"}):
            result = resolve_env(
                "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", options.env, "60000"
            )
        assert result == "45000"

    def test_stream_close_timeout_default(self) -> None:
        """Uses default when CLAUDE_CODE_STREAM_CLOSE_TIMEOUT not set anywhere."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions()
        env_patch = {
            k: v
            for k, v in os.environ.items()
            if k != "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"
        }
        with patch.dict(os.environ, env_patch, clear=True):
            result = resolve_env(
                "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", options.env, "60000"
            )
        assert result == "60000"


class TestSkipVersionCheckResolution:
    """Integration-style tests for CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK resolution."""

    def test_skip_version_check_from_options_env(self) -> None:
        """CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK resolved from options.env."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            env={"CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1"}
        )
        result = resolve_env(
            "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", options.env, ""
        )
        assert result == "1"

    def test_skip_version_check_os_environ_fallback(self) -> None:
        """Falls back to os.environ for CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions()
        with patch.dict(
            os.environ, {"CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "true"}
        ):
            result = resolve_env(
                "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", options.env, ""
            )
        assert result == "true"

    def test_skip_version_check_default_empty(self) -> None:
        """Default is empty string (falsy) for CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK."""
        from claude_agent_sdk.types import ClaudeAgentOptions

        options = ClaudeAgentOptions()
        env_patch = {
            k: v
            for k, v in os.environ.items()
            if k != "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"
        }
        with patch.dict(os.environ, env_patch, clear=True):
            result = resolve_env(
                "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", options.env, ""
            )
        assert result == ""
        # Empty string is falsy, so version check should run
        assert not result


class TestQueryStreamCloseTimeout:
    """Tests that Query accepts and uses stream_close_timeout parameter."""

    def test_query_uses_passed_stream_close_timeout(self) -> None:
        """Query uses the stream_close_timeout parameter when provided."""
        from claude_agent_sdk._internal.query import Query

        mock_transport = AsyncMock()
        query = Query(
            transport=mock_transport,
            is_streaming_mode=True,
            stream_close_timeout=30.0,
        )
        assert query._stream_close_timeout == 30.0

    def test_query_falls_back_to_environ(self) -> None:
        """Query falls back to os.environ when stream_close_timeout is None."""
        from claude_agent_sdk._internal.query import Query

        mock_transport = AsyncMock()
        with patch.dict(os.environ, {"CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "90000"}):
            query = Query(
                transport=mock_transport,
                is_streaming_mode=True,
            )
        assert query._stream_close_timeout == 90.0

    def test_query_falls_back_to_default(self) -> None:
        """Query uses 60s default when no stream_close_timeout and no env var."""
        from claude_agent_sdk._internal.query import Query

        mock_transport = AsyncMock()
        env_patch = {
            k: v
            for k, v in os.environ.items()
            if k != "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT"
        }
        with patch.dict(os.environ, env_patch, clear=True):
            query = Query(
                transport=mock_transport,
                is_streaming_mode=True,
            )
        assert query._stream_close_timeout == 60.0
