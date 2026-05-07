"""Tests for the .mcp.json secrets pre-commit hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The hook script lives in scripts/, not in the package. Load it directly.
HOOK_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_mcp_json_no_secrets.py"


@pytest.fixture
def hook_module(monkeypatch):
    """Import the hook script as a module (it's not on the package path)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_mcp_json_no_secrets", HOOK_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_mcp_json(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / ".mcp.json"
    p.write_text(json.dumps(doc, indent=2))
    return p


def test_clean_mcp_json_passes(hook_module, tmp_path):
    """A .mcp.json with only the supabase block (no env) → no violations."""
    p = write_mcp_json(
        tmp_path,
        {
            "mcpServers": {
                "supabase": {
                    "type": "http",
                    "url": "https://x.supabase.com/mcp?project_ref=abc",
                }
            }
        },
    )
    assert hook_module.scan_mcp_json(p) == []


def test_gh_token_in_env_is_violation(hook_module, tmp_path):
    """An env block with GH_TOKEN → violation."""
    p = write_mcp_json(
        tmp_path,
        {
            "mcpServers": {
                "gh_graphql": {
                    "command": "uvx",
                    "args": ["epm-gh-graphql-mcp"],
                    "env": {"GH_TOKEN": "ghp_secret"},
                }
            }
        },
    )
    violations = hook_module.scan_mcp_json(p)
    assert len(violations) == 1
    assert "GH_TOKEN" in violations[0]


def test_unknown_token_suffix_is_violation(hook_module, tmp_path):
    """A future token like FOO_TOKEN that's not in the explicit list still fires."""
    p = write_mcp_json(
        tmp_path,
        {"mcpServers": {"x": {"command": "x", "env": {"FOO_TOKEN": "secret"}}}},
    )
    violations = hook_module.scan_mcp_json(p)
    assert len(violations) == 1
    assert "FOO_TOKEN" in violations[0]


def test_api_key_suffix_is_violation(hook_module, tmp_path):
    """A *_API_KEY suffix triggers the hook."""
    p = write_mcp_json(
        tmp_path,
        {"mcpServers": {"x": {"command": "x", "env": {"OPENAI_API_KEY": "sk-x"}}}},
    )
    assert len(hook_module.scan_mcp_json(p)) == 1


def test_explicit_anthropic_api_key_is_violation(hook_module, tmp_path):
    """ANTHROPIC_API_KEY (in explicit list) is rejected."""
    p = write_mcp_json(
        tmp_path,
        {"mcpServers": {"x": {"command": "x", "env": {"ANTHROPIC_API_KEY": "sk-ant"}}}},
    )
    assert len(hook_module.scan_mcp_json(p)) == 1


def test_ssh_server_keypath_is_allowlisted(hook_module, tmp_path):
    """SSH_SERVER_*_KEYPATH is allowlisted (it's a path to a key file)."""
    p = write_mcp_json(
        tmp_path,
        {
            "mcpServers": {
                "ssh": {
                    "command": "node",
                    "env": {
                        "SSH_SERVER_POD1_HOST": "1.2.3.4",
                        "SSH_SERVER_POD1_PORT": "22",
                        "SSH_SERVER_POD1_USER": "root",
                        "SSH_SERVER_POD1_KEYPATH": "~/.ssh/id_ed25519",
                        "SSH_SERVER_POD1_DEFAULT_DIR": "/workspace",
                        "SSH_SERVER_POD1_PLATFORM": "linux",
                        "SSH_SERVER_POD1_DESCRIPTION": "test",
                    },
                }
            }
        },
    )
    assert hook_module.scan_mcp_json(p) == []


def test_gh_repo_owner_and_name_are_allowlisted(hook_module, tmp_path):
    """GH_REPO_OWNER / GH_REPO_NAME are not secrets."""
    p = write_mcp_json(
        tmp_path,
        {
            "mcpServers": {
                "x": {
                    "command": "x",
                    "env": {
                        "GH_REPO_OWNER": "superkaiba",
                        "GH_REPO_NAME": "explore-persona-space",
                    },
                }
            }
        },
    )
    assert hook_module.scan_mcp_json(p) == []


def test_mixed_secret_and_safe_keys_only_flags_secret(hook_module, tmp_path):
    """A block with both a safe key and a secret reports just the secret."""
    p = write_mcp_json(
        tmp_path,
        {
            "mcpServers": {
                "x": {
                    "command": "x",
                    "env": {
                        "GH_REPO_OWNER": "superkaiba",
                        "HF_TOKEN": "hf_secret",
                    },
                }
            }
        },
    )
    violations = hook_module.scan_mcp_json(p)
    assert len(violations) == 1
    assert "HF_TOKEN" in violations[0]
    assert "GH_REPO_OWNER" not in violations[0]


def test_main_returns_nonzero_on_violation(hook_module, tmp_path, capsys):
    """End-to-end: main() returns 1 and prints to stderr on violation."""
    p = write_mcp_json(
        tmp_path,
        {"mcpServers": {"x": {"command": "x", "env": {"GH_TOKEN": "ghp_x"}}}},
    )
    rc = hook_module.main([str(p)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "GH_TOKEN" in captured.err
    assert "ERROR" in captured.err


def test_main_returns_zero_on_clean_file(hook_module, tmp_path):
    p = write_mcp_json(
        tmp_path,
        {"mcpServers": {"supabase": {"type": "http", "url": "https://x"}}},
    )
    rc = hook_module.main([str(p)])
    assert rc == 0


def test_hook_skips_files_with_other_names(hook_module, tmp_path):
    """The hook only inspects files literally named .mcp.json (defense-in-depth)."""
    p = tmp_path / "not-mcp.json"
    p.write_text(json.dumps({"mcpServers": {"x": {"command": "x", "env": {"GH_TOKEN": "ghp"}}}}))
    rc = hook_module.main([str(p)])
    # Even though the JSON contains a secret, the file isn't named
    # `.mcp.json`, so the hook is a no-op.
    assert rc == 0


def test_invalid_json_reports_violation(hook_module, tmp_path):
    """Malformed JSON → violation rather than silent pass."""
    p = tmp_path / ".mcp.json"
    p.write_text("not valid json {")
    violations = hook_module.scan_mcp_json(p)
    assert len(violations) == 1
    assert "invalid JSON" in violations[0]
