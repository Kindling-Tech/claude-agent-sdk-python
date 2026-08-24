"""Tests for the Kindling fork's ``output_style`` option.

An output style is a CLI-side settings key (``outputStyle``). The SDK merges it
into the ``--settings`` payload so it lands in the highest-priority "flag
settings" layer, above any user/project settings file.
"""

import json
import logging

import pytest

from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import ClaudeAgentOptions


def _settings_value(cmd: list[str]) -> str | None:
    if "--settings" not in cmd:
        return None
    return cmd[cmd.index("--settings") + 1]


def _transport(**kwargs) -> SubprocessCLITransport:
    return SubprocessCLITransport(
        prompt="test",
        options=ClaudeAgentOptions(cli_path="/usr/bin/claude", **kwargs),
    )


class TestOutputStyleSettings:
    def test_absent_by_default(self):
        cmd = _transport()._build_command()
        assert _settings_value(cmd) is None

    def test_output_style_alone_becomes_settings_json(self):
        cmd = _transport(output_style="Concise")._build_command()
        assert json.loads(_settings_value(cmd)) == {"outputStyle": "Concise"}

    def test_merges_into_existing_settings_json(self):
        cmd = _transport(
            settings='{"model": "sonnet"}',
            output_style="Explanatory",
        )._build_command()
        assert json.loads(_settings_value(cmd)) == {
            "model": "sonnet",
            "outputStyle": "Explanatory",
        }

    def test_merges_alongside_sandbox(self):
        cmd = _transport(
            output_style="Concise",
            sandbox={"enabled": True},
        )._build_command()
        parsed = json.loads(_settings_value(cmd))
        assert parsed["outputStyle"] == "Concise"
        assert parsed["sandbox"] == {"enabled": True}

    def test_merges_into_settings_file(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"model": "opus"}))
        cmd = _transport(
            settings=str(settings_file),
            output_style="Learning",
        )._build_command()
        assert json.loads(_settings_value(cmd)) == {
            "model": "opus",
            "outputStyle": "Learning",
        }

    def test_settings_path_passes_through_without_output_style(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"model": "opus"}))
        cmd = _transport(settings=str(settings_file))._build_command()
        assert _settings_value(cmd) == str(settings_file)


class TestOutputStyleWarning:
    @pytest.fixture(autouse=True)
    def _reset_warning_latch(self):
        """The warning fires once per process; reset it between tests."""
        import claude_agent_sdk._internal.transport.subprocess_cli as mod

        mod._output_style_warning_emitted = False
        yield
        mod._output_style_warning_emitted = False

    def test_warns_for_string_system_prompt(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(
                output_style="Concise", system_prompt="You are a bot."
            )._build_command()
        assert "will not be applied" in caplog.text

    def test_warns_for_default_system_prompt(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(output_style="Concise")._build_command()
        assert "will not be applied" in caplog.text

    def test_silent_for_preset_system_prompt(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(
                output_style="Concise",
                system_prompt={"type": "preset", "preset": "claude_code"},
            )._build_command()
        assert "will not be applied" not in caplog.text

    def test_silent_for_preset_with_append(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(
                output_style="Concise",
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                    "append": "House rules.",
                },
            )._build_command()
        assert "will not be applied" not in caplog.text

    def test_silent_when_output_style_unset(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(system_prompt="You are a bot.")._build_command()
        assert "will not be applied" not in caplog.text

    def test_warns_at_most_once_per_process(self, caplog):
        with caplog.at_level(logging.WARNING):
            _transport(output_style="Concise", system_prompt="a")._build_command()
            _transport(output_style="Concise", system_prompt="b")._build_command()
        assert caplog.text.count("will not be applied") == 1
