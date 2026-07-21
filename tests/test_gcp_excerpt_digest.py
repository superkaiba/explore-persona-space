"""#1574 GCP-lane trigger-dense structural digest for ``log_tail_excerpt``.

The GCP lane builds its OWN excerpts (the sentinel-drain workload-log tail in
``_overlay_drain`` + the relaunch-probe tail in ``_probe_relaunched_workload``)
— pre-#1574 they carried raw log lines into orchestrator-facing surfaces even
on a trigger-dense run. These tests pin, with a payload sentinel standing in
for gated-content tail text:

1. ``_overlay_drain`` digests the raw drain ``log_tail`` component when
   tagged (structural drain-ALARM diagnostics stay verbatim — never
   digested), and the untagged default-kwargs path is byte-identical;
2. ``_probe_relaunched_workload`` digests all THREE emission branches
   (alive / dead+done / dead) while the ``latest_phase`` done-corroboration
   stays on the RAW tail (detection never gated);
3. ``GcpBackend.poll`` end-to-end on a RUNNING tick: the tag read comes
   from ``handle.extra["issue"]`` (pinned by the predicate call args), the
   whole serialized result is content-free when tagged, the raw drain tail
   survives untagged, and an issue-less handle NEVER reads tags.

Synthetic issue id 9574 + monkeypatched predicate — never a live task read.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import explore_persona_space.backends.gcp as gcp_mod
from explore_persona_space.backends import excerpt_digest
from explore_persona_space.backends.base import PollResult, RunHandle
from explore_persona_space.backends.gcp import (
    GcloudRunResult,
    GcpBackend,
    GcpConfig,
    _overlay_drain,
)

# The drain / relaunch probes lazily import ``scripts.poll_pipeline`` — make
# the repo root importable exactly as the production entrypoints do.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ISSUE = 9574
LOG = f"/workspace/logs/issue-{ISSUE}.log"

# Distinctive payload sentinel standing in for gated-content tail text — the
# thing that must NEVER reach an orchestrator-facing surface on a tagged run.
SENTINEL = "XYZZYPAYLOAD9574"

RAW_TAIL = f"RuntimeError: boom {SENTINEL}\n2026 ERROR something failed\nworker exited"


def _test_config() -> GcpConfig:
    return GcpConfig(
        project="eps-test-project",
        gcloud_config="eps-test-config",
        primary_zone="us-central1-a",
        fallback_zones=("us-central1-b", "us-central1-c"),
        image_family="pytorch-test-family",
        image_project="deeplearning-platform-release",
        repo_url="https://github.com/superkaiba/explore-persona-space.git",
    )


def _handle(*, with_issue: bool = True) -> RunHandle:
    extra: dict = {"zone": "us-central1-a"}
    if with_issue:
        extra["issue"] = ISSUE
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="1",
        pod_name=f"eps-issue-{ISSUE}",
        scratch_dir=f"/workspace/eps-issue-{ISSUE}",
        log_path=LOG,
        extra=extra,
    )


class _ScriptedRunner:
    """Minimal argv-keyed scripted runner (the test_gcp_backend.py shape)."""

    def __init__(
        self,
        *,
        describe: list[GcloudRunResult] | None = None,
        guest_attrs: list[GcloudRunResult] | None = None,
        ssh: list[GcloudRunResult] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.describe = list(describe or [])
        self.guest_attrs = list(guest_attrs or [])
        self.ssh = list(ssh or [])

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if "describe" in argv and "instances" in argv:
            return self.describe.pop(0)
        if "get-guest-attributes" in argv:
            return self.guest_attrs.pop(0)
        if "ssh" in argv and "compute" in argv:
            return self.ssh.pop(0)
        raise AssertionError(f"unexpected argv: {argv}")


def _guest_attr_payload(value: str) -> str:
    return json.dumps([{"namespace": "eps", "key": "phase", "value": value}])


def _drain_stdout(tail: str) -> str:
    """Drain SSH stdout: no sentinels, a raw workload-log tail + mtime keys."""
    return (
        "EPS_LOGTAIL_START\n"
        + tail
        + "\nEPS_LOGTAIL_END\n"
        + "EPS_LOG_MTIME=100\n"
        + "EPS_LOG_NOW=160\n"
    )


def _running_base() -> PollResult:
    return PollResult(
        status="running",
        current_phase="workload",
        new_milestone=False,
        last_log_mtime_sec_ago=10**9,
        pid_alive=True,
        log_tail_excerpt="",
    )


# ── 1. _overlay_drain ─────────────────────────────────────────────────────────


def test_overlay_drain_digests_raw_drain_tail_when_tagged() -> None:
    """Tagged: the raw drain ``log_tail`` component becomes the shared
    digest (sentinel-free); a non-empty structural drain ALARM wins the
    merge VERBATIM — lane-built diagnostics are never digested."""
    out = _overlay_drain(
        _running_base(),
        processed=0,
        gate=None,
        alarm="",
        log_tail=RAW_TAIL,
        log_mtime_ago=120,
        trigger_dense=True,
        log_path=LOG,
    )
    assert out.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert SENTINEL not in out.log_tail_excerpt
    assert "source=gcp_drain" in out.log_tail_excerpt
    assert f"log={LOG}" in out.log_tail_excerpt
    assert "log_mtime_sec_ago=120" in out.log_tail_excerpt

    alarm = "gcp sentinel drain FAILED (rc=255): ssh: connect to host refused"
    out_alarm = _overlay_drain(
        _running_base(),
        processed=0,
        gate=None,
        alarm=alarm,
        log_tail=RAW_TAIL,
        log_mtime_ago=None,
        trigger_dense=True,
        log_path=LOG,
    )
    assert out_alarm.log_tail_excerpt == alarm


def test_overlay_drain_untagged_byte_identical() -> None:
    """Default kwargs (no trigger_dense / log_path): the raw
    ``alarm or base or log_tail`` merge, string-EQUAL to the raw tail."""
    out = _overlay_drain(
        _running_base(),
        processed=0,
        gate=None,
        alarm="",
        log_tail=RAW_TAIL,
        log_mtime_ago=120,
    )
    assert out.log_tail_excerpt == RAW_TAIL
    assert out.last_log_mtime_sec_ago == 120


# ── 2. _probe_relaunched_workload (three emission branches) ──────────────────


def _relaunch_stdout(*, alive: bool, tail: str) -> str:
    return (
        f"EPS_RELAUNCH_PID={'alive' if alive else 'dead'}\n"
        "EPS_RELAUNCH_MTIME=100\n"
        "EPS_RELAUNCH_NOW=160\n"
        "EPS_RELAUNCH_TAIL_START\n" + tail + "\nEPS_RELAUNCH_TAIL_END\n"
    )


def _probe(*, alive: bool, tail: str) -> PollResult:
    runner = _ScriptedRunner(ssh=[GcloudRunResult(0, _relaunch_stdout(alive=alive, tail=tail), "")])
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    return backend._probe_relaunched_workload(
        _handle(), "us-central1-a", pid=4242, log_path=LOG, trigger_dense=True
    )


def test_probe_relaunched_workload_digests_all_three_branches() -> None:
    """Tagged: every classification branch emits a sentinel-free digest;
    the dead+done branch still classifies ``done`` off the RAW tail
    (``latest_phase`` corroboration is never gated)."""
    # (a) pid alive -> running.
    pr = _probe(alive=True, tail=f"2026-07-21 [phase=training] {SENTINEL} step 5")
    assert pr.status == "running"
    assert pr.current_phase == "relaunched_workload"
    assert pr.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert SENTINEL not in pr.log_tail_excerpt
    assert "source=gcp_relaunch" in pr.log_tail_excerpt

    # (b) pid dead + [phase=done] in the RAW tail -> done (detection ungated).
    pr = _probe(
        alive=False,
        tail=f"2026-07-21 [phase=training] {SENTINEL}\n2026-07-21 [phase=done] run complete",
    )
    assert pr.status == "done"
    assert pr.current_phase == "relaunched_workload_done"
    assert pr.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert SENTINEL not in pr.log_tail_excerpt

    # (c) pid dead, no done -> dead.
    pr = _probe(alive=False, tail=f"RuntimeError: boom {SENTINEL}\nworker exited")
    assert pr.status == "dead"
    assert pr.current_phase == "relaunched_workload_exited"
    assert pr.log_tail_excerpt.startswith("[trigger-dense digest]")
    assert SENTINEL not in pr.log_tail_excerpt


# ── 3. GcpBackend.poll end-to-end (RUNNING tick with a drain tail) ────────────


def _poll_once(monkeypatch: pytest.MonkeyPatch, *, tagged: bool, with_issue: bool = True):
    """Drive one RUNNING poll tick; returns (result, predicate_mock)."""
    predicate = MagicMock(return_value=tagged)
    monkeypatch.setattr(excerpt_digest, "issue_trigger_dense", predicate)
    runner = _ScriptedRunner(
        describe=[GcloudRunResult(0, json.dumps({"status": "RUNNING"}), "")],
        guest_attrs=[GcloudRunResult(0, _guest_attr_payload("workload"), "")],
        ssh=[GcloudRunResult(0, _drain_stdout(RAW_TAIL), "")],
    )
    backend = GcpBackend(config=_test_config(), runner=runner, marker_poster=lambda **_: None)
    return backend.poll(_handle(with_issue=with_issue)), predicate


def test_gcp_poll_running_tagged_end_to_end_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tagged RUNNING tick: the tag read keys on ``handle.extra['issue']``
    (predicate called exactly once with 9574), and the WHOLE serialized
    result carries no raw tail text (the field-scope-illusion killer)."""
    result, predicate = _poll_once(monkeypatch, tagged=True)
    predicate.assert_called_once_with(ISSUE, log=gcp_mod.logger)
    assert result.status == "running"
    assert result.log_tail_excerpt.startswith("[trigger-dense digest]")
    serialized = json.dumps(dataclasses.asdict(result))
    assert SENTINEL not in serialized


def test_gcp_poll_running_untagged_raw_drain_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Untagged twin: the raw drain tail reaches the excerpt verbatim."""
    result, predicate = _poll_once(monkeypatch, tagged=False)
    predicate.assert_called_once_with(ISSUE, log=gcp_mod.logger)
    assert result.log_tail_excerpt == RAW_TAIL
    assert SENTINEL in result.log_tail_excerpt


def test_gcp_poll_missing_issue_extra_never_reads_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An issue-less handle short-circuits the tag read entirely (the
    ``issue > 0`` guard mirrors _drain_sentinels' own pre-SSH skip) and the
    raw path — here the structural SKIPPED alarm — is preserved."""
    result, predicate = _poll_once(monkeypatch, tagged=True, with_issue=False)
    assert predicate.call_count == 0
    # The drain skipped pre-SSH; its structural alarm wins the merge.
    assert "gcp sentinel drain SKIPPED" in result.log_tail_excerpt
