"""Tests for API configuration isolation.

These tests verify that:
1. New api_key, base_url, isolated, and max_output_tokens parameters work correctly
2. Global os.environ is not mutated when creating clients
3. Multiple ClaudeAgentOptions instances are independent
4. Environment variable inheritance is properly controlled
"""

import os

import pytest

from claude_agent_sdk import ClaudeAgentOptions


class TestApiConfiguration:
    """Test explicit API configuration parameters."""

    def test_api_key_parameter(self):
        """api_key should be stored in options."""
        options = ClaudeAgentOptions(api_key="test-key-123")
        assert options.api_key == "test-key-123"

    def test_base_url_parameter(self):
        """base_url should be stored in options."""
        options = ClaudeAgentOptions(base_url="https://custom.api.example.com")
        assert options.base_url == "https://custom.api.example.com"

    def test_both_api_params(self):
        """Both api_key and base_url can be set together."""
        options = ClaudeAgentOptions(
            api_key="sk-test-key", base_url="https://proxy.example.com/v1"
        )
        assert options.api_key == "sk-test-key"
        assert options.base_url == "https://proxy.example.com/v1"

    def test_max_output_tokens_parameter(self):
        """max_output_tokens should be stored in options."""
        options = ClaudeAgentOptions(max_output_tokens=64000)
        assert options.max_output_tokens == 64000

    def test_max_output_tokens_default(self):
        """max_output_tokens should default to None."""
        options = ClaudeAgentOptions()
        assert options.max_output_tokens is None

    def test_isolated_default_false(self):
        """isolated should default to False for backward compatibility."""
        options = ClaudeAgentOptions()
        assert options.isolated is False

    def test_isolated_explicit_true(self):
        """isolated can be set to True."""
        options = ClaudeAgentOptions(isolated=True)
        assert options.isolated is True


class TestEnvironmentIsolation:
    """Test that global environment is not mutated."""

    def test_no_global_env_mutation_on_options_creation(self):
        """Creating ClaudeAgentOptions should not mutate os.environ."""
        original_keys = set(os.environ.keys())

        options = ClaudeAgentOptions(
            api_key="test-key", base_url="https://test.com", isolated=True
        )

        # Verify no new keys were added to global environ
        new_keys = set(os.environ.keys())
        assert (
            original_keys == new_keys
        ), f"New keys added: {new_keys - original_keys}"

        # Ensure values didn't leak
        assert os.environ.get("ANTHROPIC_API_KEY") != "test-key"
        assert os.environ.get("ANTHROPIC_BASE_URL") != "https://test.com"

    def test_multiple_instances_independent(self):
        """Multiple ClaudeAgentOptions instances should be independent."""
        options1 = ClaudeAgentOptions(
            api_key="key-instance-1", base_url="https://api1.example.com"
        )
        options2 = ClaudeAgentOptions(
            api_key="key-instance-2", base_url="https://api2.example.com"
        )

        # Verify they are completely independent
        assert options1.api_key != options2.api_key
        assert options1.base_url != options2.base_url
        assert options1.api_key == "key-instance-1"
        assert options2.api_key == "key-instance-2"

    def test_no_entrypoint_env_mutation(self):
        """CLAUDE_CODE_ENTRYPOINT should not be set in global env by options creation."""
        # Save original value if it exists
        original_entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT")

        # Create options - should not mutate global env
        options = ClaudeAgentOptions()

        # Verify CLAUDE_CODE_ENTRYPOINT was not modified
        current_entrypoint = os.environ.get("CLAUDE_CODE_ENTRYPOINT")
        assert current_entrypoint == original_entrypoint


class TestFullConfigurationScenarios:
    """Test real-world configuration scenarios."""

    def test_proxy_configuration(self):
        """Test configuration for running through a proxy."""
        options = ClaudeAgentOptions(
            api_key="sk-proxy-key",
            base_url="https://my-proxy.example.com/v1/anthropic",
            isolated=True,
            max_output_tokens=32000,
        )
        assert options.api_key == "sk-proxy-key"
        assert options.base_url == "https://my-proxy.example.com/v1/anthropic"
        assert options.isolated is True
        assert options.max_output_tokens == 32000

    def test_multiple_providers_simultaneous(self):
        """Test multiple provider configurations can coexist."""
        # Provider 1: Direct Anthropic
        direct_options = ClaudeAgentOptions(
            api_key="sk-anthropic-direct",
            isolated=True,
        )

        # Provider 2: OpenRouter
        openrouter_options = ClaudeAgentOptions(
            api_key="sk-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            isolated=True,
        )

        # Provider 3: Custom proxy
        proxy_options = ClaudeAgentOptions(
            api_key="custom-proxy-key",
            base_url="https://internal-proxy.company.com/anthropic",
            isolated=True,
        )

        # Verify all are independent
        assert direct_options.api_key == "sk-anthropic-direct"
        assert direct_options.base_url is None

        assert openrouter_options.api_key == "sk-openrouter-key"
        assert openrouter_options.base_url == "https://openrouter.ai/api/v1"

        assert proxy_options.api_key == "custom-proxy-key"
        assert proxy_options.base_url == "https://internal-proxy.company.com/anthropic"

    def test_backward_compatible_env_dict(self):
        """Test that env dict still works alongside new params."""
        options = ClaudeAgentOptions(
            api_key="explicit-key",
            env={
                "SOME_OTHER_VAR": "value",
                # This should be overridden by api_key
                "ANTHROPIC_API_KEY": "env-dict-key",
            },
        )
        # The explicit api_key should take priority (at subprocess creation time)
        # Both should be stored in options
        assert options.api_key == "explicit-key"
        assert options.env["ANTHROPIC_API_KEY"] == "env-dict-key"
        assert options.env["SOME_OTHER_VAR"] == "value"


class TestIsolatedModeEnvironmentFiltering:
    """Test that isolated mode properly filters environment variables."""

    def test_isolated_mode_concept(self):
        """Verify isolated flag is properly set."""
        non_isolated = ClaudeAgentOptions(isolated=False)
        isolated = ClaudeAgentOptions(isolated=True)

        assert non_isolated.isolated is False
        assert isolated.isolated is True

    def test_default_non_isolated_for_backward_compat(self):
        """Default should be non-isolated for backward compatibility."""
        options = ClaudeAgentOptions()
        assert options.isolated is False
