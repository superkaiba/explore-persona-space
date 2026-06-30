"""Tests for ``scripts/pod_config.py`` env-key shape and strip-regex migration.

These cover the four behaviors the SSH MCP wiring depends on:

1. Round-trip naming (canonical): a pod named ``pod-261`` produces env
   key ``SSH_SERVER_POD-261_HOST`` and parses back to ``pod-261``.
2. Round-trip naming (legacy back-compat): a pod named ``epm-issue-261``
   still produces ``SSH_SERVER_EPM-ISSUE-261_HOST`` and parses back to
   ``epm-issue-261`` so in-flight pods provisioned before the April 2026
   rename keep working.
3. Strip regex covers all four migration shapes (permanent ``POD<N>``,
   canonical ephemeral ``POD-<N>``, legacy ephemeral ``EPM-ISSUE-<N>``,
   very-legacy ephemeral ``PODepm-issue-<N>``).
4. ``update_mcp_config`` is idempotent and preserves non-pod env vars
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


def test_round_trip_canonical_pod_naming(tmp_path: Path) -> None:
    """Canonical: pod-261 -> SSH_SERVER_POD-261_HOST -> pod-261."""
    pod = _make_pod("pod-261", host="64.247.201.34", port=17765)
    env = _generate_mcp_env([pod])

    assert env["SSH_SERVER_POD-261_HOST"] == "64.247.201.34"
    assert env["SSH_SERVER_POD-261_PORT"] == "17765"

    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"ssh": {"command": "node", "args": [], "env": env}}})
    )

    with patch.object(pod_config, "MCP_JSON", mcp_path):
        parsed = _parse_mcp_pods()

    assert parsed == {"pod-261": ("64.247.201.34", 17765)}


def test_round_trip_legacy_epm_issue_naming(tmp_path: Path) -> None:
    """Back-compat: epm-issue-261 still round-trips correctly."""
    pod = _make_pod("epm-issue-261", host="64.247.201.34", port=17765)
    env = _generate_mcp_env([pod])

    assert env["SSH_SERVER_EPM-ISSUE-261_HOST"] == "64.247.201.34"
    assert env["SSH_SERVER_EPM-ISSUE-261_PORT"] == "17765"
    assert "SSH_SERVER_PODepm-issue-261_HOST" not in env, "very-legacy POD prefix must be gone"

    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"ssh": {"command": "node", "args": [], "env": env}}})
    )

    with patch.object(pod_config, "MCP_JSON", mcp_path):
        parsed = _parse_mcp_pods()

    assert parsed == {"epm-issue-261": ("64.247.201.34", 17765)}


def test_strip_regex_handles_all_four_shapes(tmp_path: Path) -> None:
    """update_mcp_config strips permanent, canonical ephemeral, and both legacy ephemeral keys."""
    legacy_env = {
        # Permanent (already gone in pods.conf, but a stale --sync should prune)
        "SSH_SERVER_POD1_HOST": "10.0.0.1",
        "SSH_SERVER_POD1_PORT": "22",
        "SSH_SERVER_POD1_USER": "root",
        # Very-legacy ephemeral (POD prefix, mixed-case suffix)
        "SSH_SERVER_PODepm-issue-188_HOST": "10.0.0.2",
        "SSH_SERVER_PODepm-issue-188_PORT": "33",
        # Legacy ephemeral (no POD prefix, fully uppercased — pre-rename)
        "SSH_SERVER_EPM-ISSUE-238_HOST": "10.0.0.3",
        "SSH_SERVER_EPM-ISSUE-238_PORT": "44",
        # Canonical ephemeral (post-rename)
        "SSH_SERVER_POD-280_HOST": "10.0.0.4",
        "SSH_SERVER_POD-280_PORT": "55",
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

    new_pod = _make_pod("pod-261", host="10.1.1.1", port=11111)
    with patch.object(pod_config, "MCP_JSON", mcp_path):
        update_mcp_config([new_pod])

    written = json.loads(mcp_path.read_text())
    final_env = written["mcpServers"]["ssh"]["env"]

    # All four legacy shapes pruned
    for key in [
        "SSH_SERVER_POD1_HOST",
        "SSH_SERVER_POD1_PORT",
        "SSH_SERVER_POD1_USER",
        "SSH_SERVER_PODepm-issue-188_HOST",
        "SSH_SERVER_PODepm-issue-188_PORT",
        "SSH_SERVER_EPM-ISSUE-238_HOST",
        "SSH_SERVER_EPM-ISSUE-238_PORT",
        "SSH_SERVER_POD-280_HOST",
        "SSH_SERVER_POD-280_PORT",
    ]:
        assert key not in final_env, f"{key} should have been stripped"

    # Foreign vars preserved
    assert final_env["SSH_SERVER_MYCOWORKER_HOST"] == "10.99.99.99"
    assert final_env["SSH_SERVER_MYCOWORKER_PORT"] == "2222"
    assert final_env["UNRELATED_VAR"] == "keep-me"

    # New pod written under canonical prefix
    assert final_env["SSH_SERVER_POD-261_HOST"] == "10.1.1.1"
    assert final_env["SSH_SERVER_POD-261_PORT"] == "11111"


def test_sync_is_idempotent(tmp_path: Path) -> None:
    """Running update_mcp_config twice with the same pod list yields no diff on the second run."""
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(
        json.dumps({"mcpServers": {"ssh": {"command": "node", "args": [], "env": {}}}})
    )

    pods = [
        _make_pod("pod-261", host="10.1.1.1", port=11111),
        _make_pod("pod-280", host="10.2.2.2", port=22222),
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
        # Canonical pod-N
        ("pod-1", "SSH_SERVER_POD-1_HOST"),
        ("pod-261", "SSH_SERVER_POD-261_HOST"),
        ("pod-9999", "SSH_SERVER_POD-9999_HOST"),
        # Legacy epm-issue-N (back-compat)
        ("epm-issue-1", "SSH_SERVER_EPM-ISSUE-1_HOST"),
        ("epm-issue-261", "SSH_SERVER_EPM-ISSUE-261_HOST"),
    ],
)
def test_env_key_uppercases_pod_name_verbatim(pod_name: str, expected_host_key: str) -> None:
    """The env-key suffix is pod.name.upper() with no decoration."""
    env = _generate_mcp_env([_make_pod(pod_name)])
    assert expected_host_key in env


# ---------------------------------------------------------------------------
# cmd_update create-missing (#751): ``pod.py config --update <absent>`` with
# BOTH --host and --port CREATES the row (user-pinned: manual_override=True)
# instead of the old "pod not found" exit. The documented manual recovery for
# a pod a failover / no-port-wedge re-provision left without a pods.conf row.
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_pods_conf(monkeypatch):
    """Stub ``parse_pods_conf`` / ``write_pods_conf`` / ``cmd_sync`` /
    ``locked_pods_conf`` so ``cmd_update`` never touches real repo state.
    Mirrors the fixture in ``test_pod_config_refresh_from_api.py``."""
    import contextlib

    state: dict[str, object] = {
        "rows": [],
        "written": None,
        "sync_called": False,
    }

    def fake_parse():
        return [
            Pod(
                name=p.name,
                host=p.host,
                port=p.port,
                gpus=p.gpus,
                gpu_type=p.gpu_type,
                label=p.label,
            )
            for p in state["rows"]  # type: ignore[union-attr]
        ]

    def fake_write(rows):
        state["written"] = list(rows)

    def fake_sync(rows):
        state["sync_called"] = True

    @contextlib.contextmanager
    def noop_lock():
        yield

    monkeypatch.setattr(pod_config, "parse_pods_conf", fake_parse)
    monkeypatch.setattr(pod_config, "write_pods_conf", fake_write)
    monkeypatch.setattr(pod_config, "cmd_sync", fake_sync)
    monkeypatch.setattr(pod_config, "locked_pods_conf", noop_lock)
    return state


@pytest.fixture
def isolated_sidecar(tmp_path, monkeypatch):
    """Point ``PODS_EPHEMERAL_JSON`` at a tmp copy so ``_set_manual_override``
    reads/writes the test sidecar, not the real one."""
    sidecar = tmp_path / "pods_ephemeral.json"
    monkeypatch.setattr(pod_config, "PODS_EPHEMERAL_JSON", sidecar)
    return sidecar


def _write_sidecar(path: Path, pod_name: str, *, manual_override: bool = False) -> None:
    payload = {
        "version": 2,
        "updated_at": "2026-06-30T00:00:00Z",
        "pods": {
            pod_name: {
                "name": pod_name,
                "pod_id": f"{pod_name}_id",
                "issue": 0,
                "gpu_intent": "custom",
                "ttl_days": 7,
                "stopped_at": None,
                "notes": "",
                "manual_override": manual_override,
                "extra": {},
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_update_creates_missing_pod_from_user_values_sets_override(
    stubbed_pods_conf, isolated_sidecar
):
    """``--update <absent> --host H --port P`` with BOTH host+port CREATES the
    row from the user values, calls ``cmd_sync``, and (when the pod has a
    sidecar entry) flips ``manual_override=True`` — the user-pinned create path.
    """
    stubbed_pods_conf["rows"] = [_make_pod("pod-500", host="2.2.2.2", port=22222)]
    pods_arg = [
        Pod(name=p.name, host=p.host, port=p.port, gpus=p.gpus, gpu_type=p.gpu_type, label=p.label)
        for p in stubbed_pods_conf["rows"]
    ]
    # A sidecar entry exists so _set_manual_override can actually persist the flag.
    _write_sidecar(isolated_sidecar, "pod-697", manual_override=False)

    # Does NOT raise — create-missing replaces the old "pod not found" exit.
    pod_config.cmd_update(pods_arg, "pod-697", host="103.0.0.7", port=24697)

    written = stubbed_pods_conf["written"]
    assert written is not None
    by_name = {p.name: (p.host, p.port) for p in written}
    assert by_name["pod-697"] == ("103.0.0.7", 24697)
    assert stubbed_pods_conf["sync_called"] is True

    # manual_override flipped to True (user-pinned create).
    sidecar_data = json.loads(isolated_sidecar.read_text())
    assert sidecar_data["pods"]["pod-697"]["manual_override"] is True


def test_update_creates_missing_requires_both_host_and_port(stubbed_pods_conf, isolated_sidecar):
    """Creating an ABSENT pod needs a COMPLETE endpoint — only --host (or only
    --port) cannot describe one and there is no on-disk row to fill the other,
    so it fails loud rather than writing a half-specified row."""
    stubbed_pods_conf["rows"] = [_make_pod("pod-500", host="2.2.2.2", port=22222)]
    pods_arg = [
        Pod(name=p.name, host=p.host, port=p.port, gpus=p.gpus, gpu_type=p.gpu_type, label=p.label)
        for p in stubbed_pods_conf["rows"]
    ]

    with pytest.raises(SystemExit) as excinfo:
        pod_config.cmd_update(pods_arg, "pod-697", host="103.0.0.7", port=None)
    assert excinfo.value.code == 1
    # No row written.
    assert stubbed_pods_conf["written"] is None
