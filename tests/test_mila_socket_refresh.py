"""Unit tests for ``scripts/mila_socket_refresh.py`` (slice-7 scaffolding).

The probe + askpass-env build are scriptable, so we test them directly.
The login flow is mocked end-to-end — slice 7 does NOT arm a real cron
and does NOT call out to gmail or Mila.

The integration loop (Claude session fetches OTP from gmail MCP → writes
file → cron runs the helper → SSH reads askpass → socket warms) lives
in ``.claude/cron-prompts/mila-otp-refresh.md`` and is exercised live
in slice 8 only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ruff: noqa: E402 — sys.path mutation above must precede the import.
import mila_socket_refresh as msr  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# probe_socket
# ---------------------------------------------------------------------------


def test_probe_socket_returns_alive_true_when_socket_up(monkeypatch) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.mila_socket_alive",
        lambda **_kw: True,
    )
    status = msr.probe_socket(ssh_alias="mila")
    assert status == {"alive": True, "ssh_alias": "mila"}


def test_probe_socket_returns_alive_false_when_socket_down(monkeypatch) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.mila_socket_alive",
        lambda **_kw: False,
    )
    status = msr.probe_socket(ssh_alias="mila")
    assert status == {"alive": False, "ssh_alias": "mila"}


# ---------------------------------------------------------------------------
# build_askpass_env
# ---------------------------------------------------------------------------


def test_build_askpass_env_sets_force_and_display() -> None:
    env = msr.build_askpass_env(
        askpass_path="/usr/local/bin/mila-otp-askpass",
        base_env={"HOME": "/root"},
    )
    assert env["SSH_ASKPASS"] == "/usr/local/bin/mila-otp-askpass"
    assert env["SSH_ASKPASS_REQUIRE"] == "force"
    # DISPLAY defaulted because base_env had none.
    assert env["DISPLAY"] == ":0"
    # Base env preserved.
    assert env["HOME"] == "/root"


def test_build_askpass_env_preserves_existing_display() -> None:
    env = msr.build_askpass_env(
        askpass_path="/x",
        base_env={"DISPLAY": ":42"},
    )
    assert env["DISPLAY"] == ":42"  # do not clobber existing


def test_build_askpass_env_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="askpass_path is empty"):
        msr.build_askpass_env(askpass_path="", base_env={})


# ---------------------------------------------------------------------------
# perform_login (mocked runner — never actually SSHs)
# ---------------------------------------------------------------------------


def test_perform_login_requires_askpass_env_or_arg(monkeypatch) -> None:
    monkeypatch.delenv(msr.ASKPASS_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="askpass helper not set"):
        msr.perform_login(askpass_path=None)


def test_perform_login_invokes_runner_with_argv_and_askpass_env() -> None:
    captured: dict[str, object] = {}

    def fake_runner(argv, env, timeout):  # type: ignore[no-untyped-def]
        captured["argv"] = argv
        captured["env"] = env
        captured["timeout"] = timeout
        return 0

    code = msr.perform_login(
        ssh_alias="mila",
        askpass_path="/tmp/askpass.sh",
        base_env={"HOME": "/root"},
        runner=fake_runner,
        timeout=42,
    )
    assert code == 0
    assert captured["argv"] == ["ssh", "mila", "true"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["SSH_ASKPASS"] == "/tmp/askpass.sh"
    assert env["SSH_ASKPASS_REQUIRE"] == "force"
    assert env["HOME"] == "/root"
    assert captured["timeout"] == 42


def test_perform_login_propagates_runner_exit_code() -> None:
    code = msr.perform_login(
        ssh_alias="mila",
        askpass_path="/tmp/askpass.sh",
        base_env={},
        runner=lambda _a, _e, _t: 255,
    )
    assert code == 255


# ---------------------------------------------------------------------------
# CLI smoke: probe (the only non-side-effectful action)
# ---------------------------------------------------------------------------


def test_cli_probe_emits_json_and_exit_0_when_alive(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.mila_socket_alive",
        lambda **_kw: True,
    )
    exit_code = msr.main(["probe"])
    out = capsys.readouterr().out.strip()
    body = json.loads(out)
    assert exit_code == 0
    assert body == {"alive": True, "ssh_alias": "mila"}


def test_cli_probe_emits_json_and_exit_1_when_down(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.mila_socket_alive",
        lambda **_kw: False,
    )
    exit_code = msr.main(["probe"])
    out = capsys.readouterr().out.strip()
    assert exit_code == 1
    assert json.loads(out)["alive"] is False


# ---------------------------------------------------------------------------
# CLI: login without askpass surfaces a structured error (does NOT raise)
# ---------------------------------------------------------------------------


def test_cli_login_without_askpass_returns_structured_error(monkeypatch, capsys) -> None:
    monkeypatch.delenv(msr.ASKPASS_ENV_VAR, raising=False)
    exit_code = msr.main(["login"])
    out = capsys.readouterr().out.strip()
    body = json.loads(out)
    assert exit_code == 2
    assert body["ok"] is False
    assert "askpass helper not set" in body["error"]


# ---------------------------------------------------------------------------
# Regression: ensure we did NOT accidentally arm a cron / call CronCreate
# ---------------------------------------------------------------------------


def test_helper_does_not_import_or_invoke_cron_helpers() -> None:
    """Slice 7 explicitly does not arm a cron. Sanity-check the helper
    source does not reference the CronCreate / loop / Bash arming
    surfaces — the cron-prompts file documents the manual arming step
    instead."""
    src = (REPO_ROOT / "scripts" / "mila_socket_refresh.py").read_text()
    assert "CronCreate" not in src
    assert "CronList" not in src
    # The helper does not call out to the gmail MCP either (that's
    # the Claude-session step, not a shell-callable surface).
    assert "google_workspace" not in src.lower()
    assert "gmail" not in src.lower() or "OTP email" in src  # docstring mention is OK


# ---------------------------------------------------------------------------
# Smoke: --help works (catches a broken parser quickly)
# ---------------------------------------------------------------------------


def test_cli_help_runs_without_error() -> None:
    result = subprocess.run(
        ["uv", "run", "python", str(REPO_ROOT / "scripts" / "mila_socket_refresh.py"), "--help"],
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "probe" in result.stdout
    assert "login" in result.stdout
