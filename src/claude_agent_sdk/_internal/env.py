"""Environment variable resolution with options.env precedence."""

import os


def resolve_env(key: str, options_env: dict[str, str] | None, default: str) -> str:
    """Resolve an environment variable with options.env taking precedence.

    Resolution order: options_env[key] → os.environ[key] → default

    Args:
        key: The environment variable name to look up.
        options_env: The per-instance env dict from ClaudeAgentOptions.env, or None.
        default: The fallback value if the key is not found in either source.

    Returns:
        The resolved value.
    """
    if options_env and key in options_env:
        return options_env[key]
    return os.environ.get(key, default)
