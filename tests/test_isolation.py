"""Tests for options isolation and environment behavior."""

import os

from claude_agent_sdk import ClaudeAgentOptions


class TestOptionFields:
    """Test core options field behavior."""

    def test_env_parameter(self):
        """env should be stored in options."""
        options = ClaudeAgentOptions(env={"ANTHROPIC_API_KEY": "k1"})
        assert options.env["ANTHROPIC_API_KEY"] == "k1"

    def test_os_env_parameter(self):
        """os_env should be stored in options."""
        options = ClaudeAgentOptions(os_env={"ANTHROPIC_VERTEX_PROJECT_ID": "proj-1"})
        assert options.os_env["ANTHROPIC_VERTEX_PROJECT_ID"] == "proj-1"

    def test_os_env_default(self):
        """os_env should default to empty dict."""
        options = ClaudeAgentOptions()
        assert options.os_env == {}

    def test_isolated_default_false(self):
        """isolated should default to False for backward compatibility."""
        options = ClaudeAgentOptions()
        assert options.isolated is False

    def test_isolated_explicit_true(self):
        """isolated can be set to True."""
        options = ClaudeAgentOptions(isolated=True)
        assert options.isolated is True


class TestEnvironmentIsolation:
    """Test that option creation never mutates global os.environ."""

    def test_no_global_env_mutation_on_options_creation(self):
        """Creating ClaudeAgentOptions should not mutate os.environ."""
        original_items = dict(os.environ)

        _options = ClaudeAgentOptions(
            env={"ANTHROPIC_API_KEY": "env-key"},
            os_env={"ANTHROPIC_VERTEX_PROJECT_ID": "proj-123"},
            isolated=True,
        )

        assert dict(os.environ) == original_items

    def test_multiple_instances_independent(self):
        """Multiple ClaudeAgentOptions instances should be independent."""
        options1 = ClaudeAgentOptions(
            env={"ANTHROPIC_API_KEY": "key-instance-1"},
            os_env={"ANTHROPIC_VERTEX_PROJECT_ID": "project-1"},
        )
        options2 = ClaudeAgentOptions(
            env={"ANTHROPIC_API_KEY": "key-instance-2"},
            os_env={"ANTHROPIC_VERTEX_PROJECT_ID": "project-2"},
        )

        assert options1.env["ANTHROPIC_API_KEY"] != options2.env["ANTHROPIC_API_KEY"]
        assert (
            options1.os_env["ANTHROPIC_VERTEX_PROJECT_ID"]
            != options2.os_env["ANTHROPIC_VERTEX_PROJECT_ID"]
        )

    def test_no_entrypoint_env_mutation(self):
        """CLAUDE_CODE_ENTRYPOINT should not be set in global env by options creation."""
        original_entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT")
        _options = ClaudeAgentOptions()
        current_entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT")
        assert current_entrypoint == original_entrypoint
