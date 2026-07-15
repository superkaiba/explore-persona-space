"""#1334 — multi-pod-per-issue naming (``pod-<N>-<slug>``): cross-module pins.

Two invariant families live here:

1. **Cross-module parser parity** — ``pod_audit._issue_number_from_name`` and
   ``pod_lifecycle._issue_from_pod_name`` agree on the FULL grammar table.
   This is the one-grammar consolidation pin: it holds whether the audit ships
   as a thin delegation to the lifecycle parser (the landed shape) or as a
   duplicated regex (the plan §10 fallback).

2. **pod_config name-regex round-trips** — the four widened regex sites
   (MCP env-key strip, MCP env-key read-back, bulk ``--refresh-from-api``
   managed filter, SSH-config ``POD_NAME_RE``) accept the suffixed shape, so
   pods.conf / ``~/.ssh/config`` / ``~/.claude/mcp.json`` round-trip a
   ``pod-<N>-<slug>`` pod instead of accumulating stale keys or reporting a
   perpetual ``config --check`` MISMATCH.

All file paths are monkeypatched to tmp — no real ``~/.claude/mcp.json`` /
``~/.ssh/config`` reads or writes.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pod_audit  # noqa: E402
import pod_config  # noqa: E402
import pod_lifecycle  # noqa: E402
from pod_config import Pod  # noqa: E402

# The #1334 grammar table (mirrors test_pod_lifecycle.py's
# test_issue_from_pod_name_suffix_grammar — kept as a literal here so a drift
# in EITHER parser breaks the parity pin loudly).
GRAMMAR_TABLE = [
    ("pod-47", 47),
    ("pod-475", 475),
    ("epm-issue-475", 475),
    ("pod-475-backup", 475),
    ("pod-779-b", 779),
    ("epm-issue-546-b", 546),
    ("pod-779-60", None),  # numeric slug rejected (letter-initial rule)
    ("pod-779-B", None),  # uppercase rejected
    ("pod-77960", 77960),  # legacy fabrication shape — bare int tail
    ("pod-779-", None),  # empty slug
    ("pod-779-b-", 779),  # trailing-hyphen slug admitted by [a-z][a-z0-9-]*
    ("thomas-pod-475", None),
    ("pod-abc", None),
    ("pod-", None),
    ("", None),
]


@pytest.mark.parametrize(("name", "expected"), GRAMMAR_TABLE)
def test_audit_and_lifecycle_parsers_agree(name: str, expected: int | None):
    """One grammar, one answer: audit == lifecycle == the pinned table."""
    assert (
        pod_audit._issue_number_from_name(name)
        == pod_lifecycle._issue_from_pod_name(name)
        == expected
    )


# ---------------------------------------------------------------------------
# pod_config — the four widened regex sites
# ---------------------------------------------------------------------------


def _seed_mcp(tmp_path, monkeypatch, env: dict | None = None) -> Path:
    """Point pod_config.MCP_JSON at a tmp file seeded with an ssh server."""
    mcp = tmp_path / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"ssh": {"env": env or {}}}}))
    monkeypatch.setattr(pod_config, "MCP_JSON", mcp)
    return mcp


def test_mcp_strip_regex_matches_suffixed_envkeys(tmp_path, monkeypatch):
    """Site 1 (#1334): a terminated suffixed pod's SSH_SERVER_POD-<N>-<SLUG>_*
    keys are STRIPPED on the next sync (pre-fix they accumulated forever);
    non-pod env vars are preserved."""
    env = {
        "SSH_SERVER_POD-779-B_HOST": "1.2.3.4",
        "SSH_SERVER_POD-779-B_PORT": "12345",
        "SSH_SERVER_OTHERTHING_HOST": "keep-me",
    }
    mcp = _seed_mcp(tmp_path, monkeypatch, env)

    pod_config.update_mcp_config([])  # no pods -> every pod key stripped

    new_env = json.loads(mcp.read_text())["mcpServers"]["ssh"]["env"]
    assert "SSH_SERVER_POD-779-B_HOST" not in new_env
    assert "SSH_SERVER_POD-779-B_PORT" not in new_env
    assert new_env["SSH_SERVER_OTHERTHING_HOST"] == "keep-me"


def test_mcp_readback_roundtrips_suffixed_name(tmp_path, monkeypatch):
    """Site 2 (#1334): _parse_mcp_pods round-trips SSH_SERVER_POD-779-B_HOST
    back to the pod name pod-779-b (suffix.lower(), no further change)."""
    _seed_mcp(
        tmp_path,
        monkeypatch,
        {"SSH_SERVER_POD-779-B_HOST": "1.2.3.4", "SSH_SERVER_POD-779-B_PORT": "12345"},
    )
    assert pod_config._parse_mcp_pods() == {"pod-779-b": ("1.2.3.4", 12345)}


def test_refresh_from_api_bulk_regex_accepts_suffixed():
    """Site 3 (#1334): the bulk --refresh-from-api managed filter (the #821
    self-heal loop) accepts pod-779-b so a wiped suffixed row is restorable
    from the live API — while the never-auto-add safety holds: foreign names,
    the permanent fleet, legacy epm-issue rows, and numeric slugs all stay out."""
    pat = re.compile(r"^" + pod_config._EPHEMERAL_NAME_PATTERN + r"$")
    assert pat.match("pod-779-b")
    assert pat.match("pod-779")
    for bad in ("thomas-pod-475", "pod-779-60", "pod-abc", "pod261", "epm-issue-475"):
        assert not pat.match(bad), bad
    # Wiring pin: the bulk-mode filter is BUILT from the shared module
    # constant, not a drifting inline literal.
    assert "_EPHEMERAL_NAME_PATTERN" in inspect.getsource(pod_config.cmd_refresh_from_api)


def test_ssh_config_pod_name_re_accepts_suffixed(tmp_path, monkeypatch):
    """Site 4 (#1334): POD_NAME_RE (feeding _parse_ssh_config_pods ->
    ``config --check``) matches the suffixed shape, so a healthy pod-779-b's
    ~/.ssh/config entry parses on read-back instead of producing a perpetual
    MISMATCH; the legacy accepts/rejects are unchanged."""
    for good in ("pod261", "pod-261", "epm-issue-475", "pod-779-b"):
        assert pod_config.POD_NAME_RE.match(good), good
    for bad in ("thomas-pod-475", "pod-779-60", "pod-779-B"):
        assert not pod_config.POD_NAME_RE.match(bad), bad

    ssh_config = tmp_path / "config"
    ssh_config.write_text(
        "Host *\n  StrictHostKeyChecking no\n\n"
        "Host pod-779-b\n  HostName 1.2.3.4\n  Port 12345\n\n"
        "Host my-laptop\n  HostName 9.9.9.9\n"
    )
    monkeypatch.setattr(pod_config, "SSH_CONFIG", ssh_config)
    assert pod_config._parse_ssh_config_pods() == {"pod-779-b": ("1.2.3.4", 12345)}


def test_generate_mcp_env_suffixed_name_shape(tmp_path, monkeypatch):
    """The write→strip→read loop closes (#1334 acceptance criterion 7):
    _generate_mcp_env's upper-cased keys for a suffixed pod are accepted by
    BOTH the strip regex (a later sync replaces them) and the read-back regex
    (``config --check`` sees the pod)."""
    pod = Pod(
        name="pod-779-b",
        host="1.2.3.4",
        port=12345,
        gpus=1,
        gpu_type="H100",
        label="thomas-pod-779",
    )
    env = pod_config._generate_mcp_env([pod])
    assert env["SSH_SERVER_POD-779-B_HOST"] == "1.2.3.4"
    assert env["SSH_SERVER_POD-779-B_PORT"] == "12345"

    mcp = _seed_mcp(tmp_path, monkeypatch)
    pod_config.update_mcp_config([pod])  # write
    assert pod_config._parse_mcp_pods() == {"pod-779-b": ("1.2.3.4", 12345)}  # read

    pod_config.update_mcp_config([])  # strip (the terminate-then-sync path)
    new_env = json.loads(mcp.read_text())["mcpServers"]["ssh"]["env"]
    assert not any(k.startswith("SSH_SERVER_POD-779-B") for k in new_env)
