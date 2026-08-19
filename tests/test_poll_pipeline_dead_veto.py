"""Dead-verdict evidence veto (#2265) — poll_once refuses a ``dead`` verdict
its own same-tick evidence contradicts.

The #2223 incident (2026-08-13, pod-2223-q32b): one tick returned, in a
single JSON object, ``status="dead"`` alongside ``gpu_util="97,100,100,100"``
and ``last_log_mtime_sec_ago=294``. The pid file held a dead launcher's pid;
the live generate shards were setsid-detached relaunches whose cmdlines did
not match the marker-derived launch signature, so the #1650 alive-direction
rescue correctly returned False and could not help. ``dead`` is a strong,
actionable claim (the orchestrator posts ``epm:failure v1`` +
``status:blocked`` on it), so the poller now vetoes it whenever the SAME
tick's probe carries affirmative liveness evidence — a busy GPU, a fresh
log, or a fresh issue-keyed output — and reports the non-terminal
``pid-stale-workload-live`` with
``stall_reason="pid_dead_evidence:<'+'-joined tokens>"`` instead.

These tests pin:

* the pure predicates ``_gpu_busy`` / ``_dead_verdict_veto`` (boundaries,
  fail directions, deterministic token order);
* the ``poll_once`` integration — the #2223 incident replay reads the new
  verdict (not ``dead``); ``dead`` STILL fires when all evidence is stale /
  unknown-GPU / ssh-failed; each evidence arm alone vetoes; a corroborated
  ``done`` outranks the veto (arm ordering); alive-pid arms are untouched;
* the ``main()`` JSON line surfaces the new ``status`` + ``stall_reason``.

Deliberately NOT asserted anywhere: "exactly one veto tick precedes
``dead``" — an early first tick legitimately yields MULTIPLE veto ticks
before evidence decays (the plan §7 arithmetic bounds the WINDOW, not the
tick count); every assertion here is per-tick.

Conventions copied from ``tests/test_poll_pipeline_pid_identity.py``
(importlib loader, ``_probe_stdout`` string builder — extended with an
optional ``OUTPUT_MTIME_EPOCH`` line — + ``pp.subprocess.run`` monkeypatch:
the REAL ``_parse_probe_stdout`` parses the fake stdout and the REAL
arbitration decides the verdict).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_pid_identity.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_dead_veto_under_test")

# Stable greppable substring of the veto WARN.
_VETO_WARN_SUBSTR = "refusing status=dead"

# A non-`done`, non-`gate` tail so the verdict is the normal liveness path.
_RUNNING_TAIL = "2026-08-13 00:00:01 [phase=generate step=5/100]"
# A training phase followed by a terminal done line (#545 corroboration:
# a dead pid corroborates the done-parse).
_DONE_TAIL = (
    "2026-08-13 00:00:01 [phase=generate step=100/100]\n"
    "2026-08-13 00:05:00 [phase=done] ALL PHASES COMPLETE"
)


# ── probe builder ──────────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    pod_now_epoch: int,
    tail: str,
    gpu_util: str,
    pid_alive: int = 1,
    output_mtime_epoch: int | None = None,
    results_sentinel_present: int = 0,
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``output_mtime_epoch=None`` OMITS the ``OUTPUT_MTIME_EPOCH`` line (the
    parser defaults it to ``"0"`` -> the inert ``10**9`` sentinel); any value
    emits it — the #1033 fold this veto's ``output_fresh`` arm reads.
    """
    lines = [
        "PID_FILE_MISSING=0",
        f"PID_ALIVE={pid_alive}",
        f"MTIME_EPOCH={mtime_epoch}",
        f"POD_NOW_EPOCH={pod_now_epoch}",
        "TAIL_START",
        tail,
        "TAIL_END",
        "CELL_MTIME_EPOCH=0",
        "CELL_TAIL_START",
        "CELL_TAIL_END",
        "PHASE_LOG_MTIME_EPOCH=0",
        "SHARD_LOG_MTIME_EPOCH=0",
        f"GPU_UTIL={gpu_util}",
        "ZOMBIE_GPU_PIDS=",
        "SESSION_CPU_SECS=unknown",
        f"RESULTS_SENTINEL_PRESENT={results_sentinel_present}",
    ]
    if output_mtime_epoch is not None:
        lines.append(f"OUTPUT_MTIME_EPOCH={output_mtime_epoch}")
    return "\n".join(lines)


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_kwargs: dict[str, Any],
    ssh_rc: int = 0,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Mirrors ``tests/test_poll_pipeline_pid_identity.py::_patch_pod`` — the
    sentinel-drain SSH call returns empty; the probe call returns the
    controlled stdout (parsed by the REAL ``_parse_probe_stdout``); the
    events.jsonl reads are stubbed (no marker pid — the pid file is the sole
    liveness probe, the #2223 shape). ``ssh_rc != 0`` makes the PROBE call
    fail, exercising the zeroed fallback dict.
    """

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        if ssh_rc != 0:
            return subprocess.CompletedProcess(
                args=cmd, returncode=ssh_rc, stdout="", stderr="ssh: connect refused"
            )
        stdout = _probe_stdout(**probe_kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_launch_fields", lambda issue, pod=None: (None, ""))
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch, pod=None: 10800.0)


def _poll(tmp_path: Path):
    """Run ``poll_once`` with the standard fixture args."""
    return pp.poll_once(
        issue=2223,
        pod="pod-2223-q32b",
        log_path="/workspace/logs/issue-2223.log",
        pid_file="/workspace/logs/issue-2223.pid",
        state_file=tmp_path / "poll-state.json",
    )


def _now_epoch() -> int:
    return int(time.time())


# ── 1. pure predicates (no SSH) ───────────────────────────────────────────────


def test_veto_predicate_tokens_and_boundaries() -> None:
    """``_gpu_busy`` + ``_dead_verdict_veto`` units: each arm alone, all
    three, none; unknown/empty/garbage GPU is never evidence; the exact
    threshold + stall_sec boundaries; the 10**9 absent sentinel; the
    deterministic token order."""
    stall = pp.DEFAULT_STALL_SEC
    stale = 10**9

    def veto(*, last=stale, phase=stale, shard=stale, out=stale, gpu="unknown"):
        return pp._dead_verdict_veto(
            last_mtime_ago=last,
            phase_log_mtime_ago=phase,
            shard_log_mtime_ago=shard,
            output_mtime_ago=out,
            gpu_util=gpu,
            stall_sec=stall,
        )

    # _gpu_busy fail direction: unknown / empty / garbage parse to None ->
    # NEVER evidence (the inverse of _gpu_idle's fail-safe-toward-not-idle).
    assert pp._gpu_busy("unknown") is False
    assert pp._gpu_busy("") is False
    assert pp._gpu_busy("garbage,x") is False
    # Threshold boundary: 5 (== GPU_IDLE_UTIL_THRESHOLD) is NOT busy; 6 is.
    assert pp.GPU_IDLE_UTIL_THRESHOLD == 5
    assert pp._gpu_busy("5") is False
    assert pp._gpu_busy("6") is True
    assert pp._gpu_busy("0,0,0,97") is True  # any card above threshold

    # Each evidence arm alone.
    assert veto(gpu="97,100,100,100") == ["gpu_busy"]
    assert veto(last=294) == ["log_fresh"]
    assert veto(phase=100) == ["log_fresh"]  # min() over all three log axes
    assert veto(shard=100) == ["log_fresh"]
    assert veto(out=100) == ["output_fresh"]
    # All three, deterministic order.
    assert veto(gpu="97", last=294, out=100) == ["gpu_busy", "log_fresh", "output_fresh"]
    # None: every signal absent/stale -> empty list -> dead fires as today.
    assert veto() == []
    assert veto(gpu="0,0,0,0") == []  # idle GPUs are not evidence
    # stall_sec boundary: exactly stall_sec is FRESH (<=, the exact
    # complement of the stall conjunction's `> stall_sec`); +1 is stale.
    assert veto(last=stall) == ["log_fresh"]
    assert veto(last=stall + 1) == []
    assert veto(out=stall) == ["output_fresh"]
    assert veto(out=stall + 1) == []
    # The 10**9 absent-log sentinel can never read fresh.
    assert veto(last=stale, phase=stale, shard=stale) == []


# ── 2-8. poll_once integration (SSH boundary faked) ──────────────────────────


def test_incident_2223_pid_dead_gpu_busy_log_fresh_not_dead(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The #2223 incident replay: pid dead, no marker pid, all four GPUs
    busy, log 294s fresh — the tick that read ``dead`` in production now
    reads the non-terminal veto verdict, keeps ``pid_alive=False`` legible,
    carries NO crash signature, stays on the short interval, and WARNs with
    the repair recipe."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 294,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="97,100,100,100",
            pid_alive=0,
        ),
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = _poll(tmp_path)
    assert result.status != "dead", result
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.status == "pid-stale-workload-live"  # the literal contract token
    assert result.stall_reason == "pid_dead_evidence:gpu_busy+log_fresh"
    assert result.pid_alive is False  # the contradiction stays legible
    assert result.crash_signature is None  # never arms the CUDA-IMA scan
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC
    assert any(_VETO_WARN_SUBSTR in rec.getMessage() for rec in caplog.records), [
        rec.getMessage() for rec in caplog.records
    ]


def test_decay_all_evidence_stale_dead_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A genuinely dead run after decay: pid dead, GPUs idle, every log
    >stall_sec stale — ``dead`` fires exactly as pre-#2265, with no
    stall_reason and the crash signature captured from the wide tail."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "dead", result
    assert result.stall_reason is None
    assert result.crash_signature is not None  # the wide tail, captured on dead


def test_gpu_unknown_is_never_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The fail-direction inversion pin: an unknown/erroring nvidia-smi is
    NO evidence (``not _gpu_idle`` would have read it as busy) — with stale
    logs the tick reads ``dead``."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="unknown",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "dead", result
    assert result.stall_reason is None


def test_output_fresh_alone_vetoes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The #1033 output arm alone (idle GPU, stale logs, fresh issue-keyed
    output) vetoes — and the log_fresh integration sibling: a fresh log
    alone (idle GPU, absent output) vetoes with its own single token."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0",
            pid_alive=0,
            output_mtime_epoch=now - 100,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.stall_reason == "pid_dead_evidence:output_fresh"

    # log_fresh-alone integration sibling: fresh log, stale/absent output.
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 100,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0",
            pid_alive=0,
        ),
    )
    result = _poll(tmp_path / "log-fresh")
    assert result.status == pp.STATUS_PID_STALE_WORKLOAD_LIVE, result
    assert result.stall_reason == "pid_dead_evidence:log_fresh"


def test_ssh_failure_tick_still_dead(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression pin: an ssh-failed probe's zeroed fallback dict (every
    mtime "0" -> 10**9 ago, gpu "unknown") carries ZERO evidence, so the
    tick reads ``dead`` exactly as today — the #488 stale-port auto-heal and
    the RunPod no-port wedge maturation both depend on ssh-dead polls
    reading ``dead``."""
    _patch_pod(monkeypatch, probe_kwargs={}, ssh_rc=255)
    result = _poll(tmp_path)
    assert result.status == "dead", result
    assert result.stall_reason is None


def test_done_precedence_over_veto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Arm-ordering pin: a corroborated ``done`` (terminal [phase=done] +
    dead pid = the #545 corroboration) outranks the veto even with busy
    GPUs and a fresh log — the done arm precedes the pid arm."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_DONE_TAIL,
            gpu_util="97,100,100,100",
            pid_alive=0,
            results_sentinel_present=0,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "done", result
    assert result.stall_reason is None


def test_pid_alive_paths_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No veto interference on alive-pid arms: alive + fresh log ->
    ``running``; alive + the met stall conjunction (stale logs + idle GPUs +
    flat CPU) -> ``stalled``, both byte-identical to pre-#2265."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 30,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
            pid_alive=1,
        ),
    )
    result = _poll(tmp_path)
    assert result.status == "running", result
    assert result.stall_reason is None

    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 2000,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="0,0,0,0",
            pid_alive=1,
        ),
    )
    result = _poll(tmp_path / "stalled")
    assert result.status == "stalled", result


# ── 9. main() JSON surface ────────────────────────────────────────────────────


def test_main_json_line_surfaces_new_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The orchestrator-facing tick JSON carries the new ``status`` and its
    ``pid_dead_evidence:`` ``stall_reason`` (both already in the Step 6d.2
    mandated tick-parse field set) — driven through the REAL ``poll_once``
    arbitration, not a stubbed PollResult."""
    now = _now_epoch()
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=now - 294,
            pod_now_epoch=now,
            tail=_RUNNING_TAIL,
            gpu_util="97,100,100,100",
            pid_alive=0,
        ),
    )
    rc = pp.main(
        [
            "--issue",
            "2223",
            "--pod",
            "pod-2223-q32b",
            "--log",
            "/workspace/logs/issue-2223.log",
            "--pid-file",
            "/workspace/logs/issue-2223.pid",
            "--state-file",
            str(tmp_path / "poll-state.json"),
        ]
    )
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert parsed["status"] == "pid-stale-workload-live"
    assert parsed["stall_reason"] == "pid_dead_evidence:gpu_busy+log_fresh"
    assert parsed["pid_alive"] is False
    assert parsed["next_interval"] == pp.POLL_INTERVAL_DEFAULT_SEC
