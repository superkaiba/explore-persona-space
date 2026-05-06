"""Tests for ``scripts/pod_config.py`` env-key shape and strip-regex migration.

These cover the three behaviors the SSH MCP wiring depends on:

1. Round-trip naming: a pod named ``epm-issue-261`` produces env key
   ``SSH_SERVER_EPM-ISSUE-261_HOST`` and parses back to ``epm-issue-261``.
2. Strip regex covers all three migration shapes (permanent ``POD<N>``,
   new ephemeral ``EPM-ISSUE-<N>``, legacy ephemeral ``PODepm-issue-<N>``).
3. ``update_mcp_config`` is idempotent and preserves non-pod env vars
   (e.g. user-added ``SSH_SERVER_FOO_*`` entries).

Tests run without network or filesystem mutation outside ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_config  # noqa: E402
from pod_config import (  # noqa: E402
    Pod,
    _generate_mcp_env,
    _parse_mcp_pods,
    update_mcp_config,
)


def _make_pod(name: str, host: str = "1.2.3.4", port: int = 12345) -> Pod:
    return Pod(name=name, host=host, port=port, gpus=1, gpu_type="H100", label=f"thomas-{name}")


def test_round_trip_epm_issue_naming(tmp_path: Path) -> None:
    """epm-issue-261 -> SSH_SERVER_EPM-ISSUE-261_HOST -> epm-issue-261."""
    pod = _make_pod("epm-issue-261", host="64.247.201.34", port=17765)
    env = _generate_mcp_env([pod])

    assert env["SSH_SERVER_EPM-ISSUE-261_HOST"] == "64.247.201.34"
    assert env["SSH_SERVER_EPM-ISSUE-261_PORT"] == "17765"
    assert "SSH_SERVER_PODepm-issue-261_HOST" not in env, "legacy POD prefix must be gone"

    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"ssh": {"command": "node", "args": [], "env": env}}})
    )

    with patch.object(pod_config, "MCP_JSON", mcp_path):
        parsed = _parse_mcp_pods()

    assert parsed == {"epm-issue-261": ("64.247.201.34", 17765)}


def test_strip_regex_handles_all_three_shapes(tmp_path: Path) -> None:
    """update_mcp_config strips permanent, new ephemeral, and legacy ephemeral keys."""
    legacy_env = {
        # Permanent (already gone in pods.conf, but a stale --sync should prune)
        "SSH_SERVER_POD1_HOST": "10.0.0.1",
        "SSH_SERVER_POD1_PORT": "22",
        "SSH_SERVER_POD1_USER": "root",
        # Legacy ephemeral (POD prefix, mixed-case suffix)
        "SSH_SERVER_PODepm-issue-188_HOST": "10.0.0.2",
        "SSH_SERVER_PODepm-issue-188_PORT": "33",
        # New ephemeral (no POD prefix, fully uppercased)
        "SSH_SERVER_EPM-ISSUE-238_HOST": "10.0.0.3",
        "SSH_SERVER_EPM-ISSUE-238_PORT": "44",
        # Foreign env var that must be preserved
        "SSH_SERVER_MYCOWORKER_HOST": "10.99.99.99",
        "SSH_SERVER_MYCOWORKER_PORT": "2222",
        # Non-SSH env that must be preserved
        "UNRELATED_VAR": "keep-me",
    }
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps(
            {"mcpServers": {"ssh": {"command": "node", "args": [], "env": legacy_env}}},
            indent=2,
        )
    )

    new_pod = _make_pod("epm-issue-261", host="10.1.1.1", port=11111)
    with patch.object(pod_config, "MCP_JSON", mcp_path):
        update_mcp_config([new_pod])

    written = json.loads(mcp_path.read_text())
    final_env = written["mcpServers"]["ssh"]["env"]

    # All three legacy shapes pruned
    for key in [
        "SSH_SERVER_POD1_HOST",
        "SSH_SERVER_POD1_PORT",
        "SSH_SERVER_POD1_USER",
        "SSH_SERVER_PODepm-issue-188_HOST",
        "SSH_SERVER_PODepm-issue-188_PORT",
        "SSH_SERVER_EPM-ISSUE-238_HOST",
        "SSH_SERVER_EPM-ISSUE-238_PORT",
    ]:
        assert key not in final_env, f"{key} should have been stripped"

    # Foreign vars preserved
    assert final_env["SSH_SERVER_MYCOWORKER_HOST"] == "10.99.99.99"
    assert final_env["SSH_SERVER_MYCOWORKER_PORT"] == "2222"
    assert final_env["UNRELATED_VAR"] == "keep-me"

    # New pod written
    assert final_env["SSH_SERVER_EPM-ISSUE-261_HOST"] == "10.1.1.1"
    assert final_env["SSH_SERVER_EPM-ISSUE-261_PORT"] == "11111"


def test_sync_is_idempotent(tmp_path: Path) -> None:
    """Running update_mcp_config twice with the same pod list yields no diff on the second run."""
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"ssh": {"command": "node", "args": [], "env": {}}}})
    )

    pods = [
        _make_pod("epm-issue-261", host="10.1.1.1", port=11111),
        _make_pod("epm-issue-280", host="10.2.2.2", port=22222),
    ]

    with patch.object(pod_config, "MCP_JSON", mcp_path):
        first = update_mcp_config(pods)
        second = update_mcp_config(pods)

    assert any("updated" in c or "+" in c for c in first), "first sync should report changes"
    assert any("already up to date" in c for c in second), (
        f"second sync should be a no-op, got: {second}"
    )


@pytest.mark.parametrize(
    ("pod_name", "expected_host_key"),
    [
        ("epm-issue-1", "SSH_SERVER_EPM-ISSUE-1_HOST"),
        ("epm-issue-261", "SSH_SERVER_EPM-ISSUE-261_HOST"),
        ("epm-issue-9999", "SSH_SERVER_EPM-ISSUE-9999_HOST"),
    ],
)
def test_env_key_uppercases_pod_name_verbatim(pod_name: str, expected_host_key: str) -> None:
    """The env-key suffix is pod.name.upper() with no decoration."""
    env = _generate_mcp_env([_make_pod(pod_name)])
    assert expected_host_key in env
