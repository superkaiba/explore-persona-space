"""Unit tests for the GCP stale-VM janitor CLI (``scripts/gcp_audit.py``).

The CLI is a thin wrapper over ``backends.gcp.audit_stale_gcp_vms`` (the reap
predicate, exhaustively tested in ``test_gcp_backend.py``). These tests pin the
WIRING the CLI adds on top of the reaper:

* (a) the reaper's three core verdicts survive an end-to-end call (age-reap,
  terminal-phase-reap, healthy-keep) and a ``list`` is actually issued;
* (b) ``--terminal-phase-max-age-min`` threads argparse → the reaper kwarg and
  changes the verdict (flag-threading proof, NOT a re-test of the predicate);
* (c) a ``delete-failed`` reaper record maps to CLI exit 2;
* (d) a failed LIST-preflight maps to exit 3, surfaces ``list_rc``, and NEVER
  reaches the reaper (the credit-leak-backstop-DISARMED alarm);
* (e) the ``--json`` payload carries the documented top-level keys;
* report-only is the default and ``EPS_GCP_JANITOR_DRY_RUN=1`` neuters
  ``--delete``.

Every test mocks the ``gcloud`` subprocess via the injected runner seam reused
from ``test_gcp_backend`` — no test hits a real GCP project.
"""

from __future__ import annotations

import functools
import json
from datetime import UTC, datetime, timedelta

import scripts.gcp_audit as cli
from explore_persona_space.backends.gcp import GcloudRunResult

# Reuse the scripted-runner + config seams the reaper suite already maintains.
from tests.test_gcp_backend import (
    _guest_attr_payload,
    _one_running_instance,
    _Runner,
    _test_config,
)

# A fixed reference time so the fixture ages are deterministic regardless of
# wall-clock. The CLI's ``main()`` does not expose ``now``, so the CLI-level
# tests pin it by swapping in a partial that injects ``now`` (see ``_pin_now``).
_NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=UTC)


def _vm(name: str, status: str, created_iso: str) -> dict:
    return {
        "name": name,
        "id": "1",
        "status": status,
        "zone": (
            "https://www.googleapis.com/compute/v1/projects/eps-test-project/zones/us-central1-a"
        ),
        "creationTimestamp": created_iso,
    }


def _three_vm_payload() -> str:
    """One age-reap + one terminal-phase-reap + one healthy-keep, relative to _NOW."""
    return json.dumps(
        [
            _vm("eps-issue-100", "TERMINATED", (_NOW - timedelta(hours=30)).isoformat()),
            _vm("eps-issue-200", "RUNNING", (_NOW - timedelta(minutes=20)).isoformat()),
            _vm("eps-issue-300", "RUNNING", (_NOW - timedelta(minutes=5)).isoformat()),
        ]
    )


def _pin_now(monkeypatch, *, runner, now=_NOW):
    """Wire the CLI module to the mocked runner/config with a frozen ``now``."""
    monkeypatch.setattr(cli, "default_gcloud_runner", runner)
    monkeypatch.setattr(cli, "default_gcp_config", _test_config)
    monkeypatch.setattr(
        cli,
        "audit_stale_gcp_vms",
        functools.partial(cli.audit_stale_gcp_vms, now=now),
    )


# ---------------------------------------------------------------------------
# (a) Reaper-level verdicts survive an end-to-end reaper call.
# ---------------------------------------------------------------------------
def test_audit_reaps_age_and_terminal_phase_keeps_healthy() -> None:
    """Direct reaper call (no CLI) — the predicate produces the expected
    per-instance verdicts and a ``list`` + ``instances`` call is issued."""
    runner = _Runner(
        list_results=[GcloudRunResult(0, _three_vm_payload(), "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        delete_results=[GcloudRunResult(0, "", ""), GcloudRunResult(0, "", "")],
    )
    records = cli.audit_stale_gcp_vms(config=_test_config(), runner=runner, now=_NOW, delete=True)
    by = {r["name"]: r for r in records}
    assert by["eps-issue-100"]["reason"] == "age"
    assert by["eps-issue-100"]["action"] == "deleted"
    assert by["eps-issue-200"]["reason"] == "terminal-phase"
    assert by["eps-issue-200"]["action"] == "deleted"
    assert by["eps-issue-300"]["reason"] is None
    assert by["eps-issue-300"]["action"] == "skipped"
    assert any("list" in c and "instances" in c for c in runner.calls)  # Acceptance #1


# ---------------------------------------------------------------------------
# (b) CLI threads --terminal-phase-max-age-min through to the reaper kwarg.
# ---------------------------------------------------------------------------
def test_cli_threads_terminal_phase_flag(monkeypatch, capsys) -> None:
    """``--terminal-phase-max-age-min=5`` (300s floor) survives argparse → the
    reaper, so the 20-min terminal-phase RUNNING VM IS reaped. The two
    ``list_results`` mirror the design: the preflight + the reaper each issue a
    list."""
    payload = _three_vm_payload()
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, ""), GcloudRunResult(0, payload, "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
        delete_results=[GcloudRunResult(0, "", ""), GcloudRunResult(0, "", "")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--delete", "--terminal-phase-max-age-min=5", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["list_rc"] == 0
    by = {r["name"]: r for r in out["records"]}
    assert by["eps-issue-200"]["action"] == "deleted"  # terminal-phase, flag threaded
    assert by["eps-issue-300"]["action"] == "skipped"  # healthy RUNNING


def test_cli_high_terminal_phase_floor_keeps_terminal_phase_vm(monkeypatch, capsys) -> None:
    """Counterpart proving the flag is load-bearing: a 60-min floor (the
    20-min terminal-phase VM is now UNDER the floor) leaves it un-probed and
    SKIPPED — if the flag did not thread, it would still be reaped at the
    default 10-min floor."""
    payload = _three_vm_payload()
    runner = _Runner(
        # No guest_attr_results needed: the 20-min VM never crosses the 60-min
        # floor, so the reaper never probes its phase. Only the age VM reaps.
        list_results=[GcloudRunResult(0, payload, ""), GcloudRunResult(0, payload, "")],
        delete_results=[GcloudRunResult(0, "", "")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--delete", "--terminal-phase-max-age-min=60", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    by = {r["name"]: r for r in out["records"]}
    assert by["eps-issue-100"]["action"] == "deleted"  # age backstop still fires
    assert by["eps-issue-200"]["action"] == "skipped"  # under the raised floor
    assert by["eps-issue-300"]["action"] == "skipped"


# ---------------------------------------------------------------------------
# (c) delete-failed → CLI exit 2.
# ---------------------------------------------------------------------------
def test_cli_delete_failed_returns_rc2(monkeypatch, capsys) -> None:
    """An age-reaped VM whose ``delete`` returns non-zero is recorded as
    ``delete-failed`` by the reaper and mapped to CLI exit 2 (routine — a
    single transient GCP error, NOT a fleet-wide disarm)."""
    old = _one_running_instance("eps-issue-900", (_NOW - timedelta(hours=30)).isoformat())
    runner = _Runner(
        list_results=[GcloudRunResult(0, old, ""), GcloudRunResult(0, old, "")],
        delete_results=[GcloudRunResult(1, "", "Internal error. Please try again.")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--delete", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert out["records"][0]["action"] == "delete-failed"


# ---------------------------------------------------------------------------
# (d) list-failed → CLI exit 3, reaper never reached.
# ---------------------------------------------------------------------------
def test_cli_list_failed_returns_rc3_and_never_reaps(monkeypatch, capsys) -> None:
    """A non-zero preflight ``list`` (expired/misconfigured auth) is a HARD
    ERROR: exit 3, ``list_rc`` surfaced, no records, and the reaper is NEVER
    invoked (exactly one ``list`` issued — the preflight — and zero deletes)."""
    runner = _Runner(list_results=[GcloudRunResult(1, "", "Reauthentication failed")])
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--delete", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert out["list_rc"] == 1
    assert out["records"] == []
    assert "Reauthentication failed" in out["list_stderr"]
    # Reaper never ran → no delete, and exactly one list (the preflight).
    assert not any("delete" in c and "instances" in c for c in runner.calls)
    assert sum(1 for c in runner.calls if "list" in c and "instances" in c) == 1


# ---------------------------------------------------------------------------
# (e) --json shape: documented top-level keys present.
# ---------------------------------------------------------------------------
def test_cli_json_shape_clean_run(monkeypatch, capsys) -> None:
    """A clean report-only run emits the documented top-level keys
    (``list_rc`` / ``list_stderr`` / ``records``)."""
    payload = _three_vm_payload()
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, ""), GcloudRunResult(0, payload, "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--json"])  # no --delete → report-only
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert set(out) >= {"list_rc", "list_stderr", "records"}
    assert out["list_rc"] == 0
    assert isinstance(out["records"], list)


# ---------------------------------------------------------------------------
# Report-only default + dry-run env override.
# ---------------------------------------------------------------------------
def test_cli_report_only_default_issues_no_delete(monkeypatch, capsys) -> None:
    """Without ``--delete`` the reaped VMs are ``would-delete`` and NO delete
    call is issued."""
    payload = _three_vm_payload()
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, ""), GcloudRunResult(0, payload, "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    by = {r["name"]: r for r in out["records"]}
    assert by["eps-issue-100"]["action"] == "would-delete"
    assert by["eps-issue-200"]["action"] == "would-delete"
    assert not any("delete" in c and "instances" in c for c in runner.calls)


def test_cli_dry_run_env_neuters_delete(monkeypatch, capsys) -> None:
    """``EPS_GCP_JANITOR_DRY_RUN=1`` forces report-only even when ``--delete``
    is passed — the smoke kill-switch. No delete call is issued."""
    monkeypatch.setenv("EPS_GCP_JANITOR_DRY_RUN", "1")
    payload = _three_vm_payload()
    runner = _Runner(
        list_results=[GcloudRunResult(0, payload, ""), GcloudRunResult(0, payload, "")],
        guest_attr_results=[GcloudRunResult(0, _guest_attr_payload("done"), "")],
    )
    _pin_now(monkeypatch, runner=runner)
    rc = cli.main(["--delete", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    by = {r["name"]: r for r in out["records"]}
    assert by["eps-issue-100"]["action"] == "would-delete"  # NOT "deleted"
    assert not any("delete" in c and "instances" in c for c in runner.calls)
