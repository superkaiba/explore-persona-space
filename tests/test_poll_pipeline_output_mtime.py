"""Output-artifact mtime fold + run-scoped idle clock (#1033).

Ask 1 (output-mtime fold): a CPU-bound analysis tail writes per-cell NPZs /
result JSONs / ``.done`` sentinels for hours while every log layout is quiet
and the GPUs are idle by design (#813). The probe gains a bounded,
issue-keyed ``OUTPUT_MTIME_EPOCH`` freshness read (short-circuit
``find -newermt ... -print -quit`` under ``timeout``); the delta joins the
stall conjunction as a 5th liveness conjunct AND vetoes the #664/#826 zombie
override (streak reset, identical mechanics to the fresh-log veto). Kill
switch ``EPM_POLL_OUTPUT_MTIME_FOLD`` (default ON).

These tests pin:

* the parser lifting ``OUTPUT_MTIME_EPOCH`` (+ the "0" defaults on an absent
  line and on the ssh-failure fallback dict);
* the #813 degraded-CPU replay: stale logs + idle GPUs + CPU probe degraded
  (rate None) + a FRESH issue-keyed output -> ``running`` (the tick-47 shape
  minus the CPU signal #951 already covers);
* the pre-#1033 behavior pinned: no fresh output (``0`` or a stale epoch) ->
  ``stalled``;
* the strict-``>`` stall boundary (``output_mtime_ago == stall_sec`` still
  rescues) and the ``<=`` zombie-veto boundary (ago exactly at
  ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` still vetoes) — matching the
  other conjuncts' semantics;
* the zombie fresh-output veto as a FULL ``poll_once`` probe-stdout replay
  (NOT a direct ``_apply_zombie_override`` unit call — dropped call-site
  threading must not pass green) with streak reset;
* the #664 true positive STILL FIRING through the real
  parse -> staleness -> threading path with an explicit STALE non-zero
  ``OUTPUT_MTIME_EPOCH``;
* the direct-call default (``output_mtime_ago`` omitted -> ``inf``)
  preserving pre-#1033 ``_apply_zombie_override`` outputs;
* the heredoc probe text: bounded (``timeout``), short-circuit
  (``-print -quit``), issue-keyed roots ONLY, and the two-stage veto-window
  find emitted ONLY when the veto window is genuinely wider than stall_sec;
* the kill switch: probe block omitted AND a stray fresh
  ``OUTPUT_MTIME_EPOCH`` line left inert.

Ask 2 (RunPod-lane per-run idle clock): ``_tripwire_run_scope``'s clear set
now includes the three GPU-idle keys (``_RUN_SCOPED_STATE_KEYS``) and runs
ABOVE the idle-advisory calls in ``poll_once``, so a relaunch restarts the
idle span instead of inheriting the previous run's (#763 "543 min" on a
~17-min-old instance). Integration replays here; the pure
``_tripwire_run_scope`` key-set tests live in
``tests/test_poll_eta_tripwire.py``; the GCP attempt-id sibling lives in
``tests/test_backend_poll_gpu_idle.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_zombie_gpu.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_output_mtime_under_test")

ISSUE = 9813
LOG = f"/workspace/logs/issue-{ISSUE}.log"
PID = f"/workspace/logs/issue-{ISSUE}.pid"


# ── probe-output parser ───────────────────────────────────────────────────────


def test_parse_probe_stdout_lifts_output_mtime() -> None:
    """The parser dispatches the ``OUTPUT_MTIME_EPOCH=`` line; an absent line
    (older probe / fold disabled) defaults the key to "0"."""
    parsed = pp._parse_probe_stdout("PID_ALIVE=1\nGPU_UTIL=0\nOUTPUT_MTIME_EPOCH=1751700000\n")
    assert parsed["output_mtime_epoch"] == "1751700000"
    assert pp._parse_probe_stdout("PID_ALIVE=1\nGPU_UTIL=0\n")["output_mtime_epoch"] == "0"


def test_ssh_failure_fallback_includes_output_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ssh-failure early-return dict carries ``output_mtime_epoch: "0"``
    (fail-safe: a transport failure never reads as a fresh output)."""

    def _fail_run(cmd: list[str], **kwargs: Any):
        return subprocess.CompletedProcess(args=cmd, returncode=255, stdout="", stderr="refused")

    monkeypatch.setattr(pp.subprocess, "run", _fail_run)
    probe = pp._ssh_probe("pod-x", LOG, PID, ISSUE)
    assert probe["ssh_failed"] == "1"
    assert probe["output_mtime_epoch"] == "0"


# ── poll_once wiring harness ──────────────────────────────────────────────────


def _probe_stdout(
    *,
    pod_now: int,
    mtime_epoch: int,
    tail: str,
    gpu_util: str,
    session_cpu: str,
    zombie_pids: str = "",
    output_mtime_epoch: int | str | None = None,
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``POD_NOW_EPOCH`` is always emitted so every staleness delta rides the
    DETERMINISTIC pod-clock branch (#704) — boundary tests would otherwise be
    off-by-one flaky against ``poll_once``'s own ``time.time()``.
    ``output_mtime_epoch=None`` omits the line (older probe / fold disabled);
    a ``str`` value injects the raw scalar text verbatim (malformed-line
    replay — version skew / garbled SSH output).
    """
    lines = [
        "PID_FILE_MISSING=0",
        "PID_ALIVE=1",
        f"MTIME_EPOCH={mtime_epoch}",
        "TAIL_START",
        tail,
        "TAIL_END",
        f"POD_NOW_EPOCH={pod_now}",
        "CELL_MTIME_EPOCH=0",
        "CELL_TAIL_START",
        "CELL_TAIL_END",
        "PHASE_LOG_MTIME_EPOCH=0",
        "SHARD_LOG_MTIME_EPOCH=0",
        f"GPU_UTIL={gpu_util}",
        f"ZOMBIE_GPU_PIDS={zombie_pids}",
        "GPU_PIDS_TOTAL=unknown",
        "GPU_PIDS_RESOLVABLE=unknown",
        "NVIDIA_UVM_LIVE_HOLDERS=unknown",
        f"SESSION_CPU_SECS={session_cpu}",
        "RESULTS_SENTINEL_PRESENT=0",
    ]
    if output_mtime_epoch is not None:
        lines.append(f"OUTPUT_MTIME_EPOCH={output_mtime_epoch}")
    return "\n".join(lines)


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    posted: list[dict] | None = None,
    run_age_sec: float | None = 10800.0,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe
    (the ``tests/test_poll_pipeline_zombie_gpu.py`` convention: fake ONLY
    ``pp.subprocess.run``, so the real parser + staleness + threading run)."""

    def _fake_run(cmd: list[str], **kwargs: Any):
        remote = cmd[-1]
        out = "" if "SENTINEL_START" in remote else stdout
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    if posted is None:
        monkeypatch.setattr(pp, "post_event", MagicMock())
    else:
        monkeypatch.setattr(
            pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
        )
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: run_age_sec)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)


def _seed_state(state_file: Path, state: dict[str, str]) -> None:
    state_file.write_text(json.dumps({str(ISSUE): state}))


def _saved_state(state_file: Path) -> dict[str, str]:
    return json.loads(state_file.read_text())[str(ISSUE)]


def _poll(state_file: Path):
    return pp.poll_once(
        issue=ISSUE, pod=f"pod-{ISSUE}", log_path=LOG, pid_file=PID, state_file=state_file
    )


_STALE_TAIL = "2026-07-03 00:00:01 [phase=scoring judging batch 3/9]"


# ── Ask 1: stall-conjunction fold ─────────────────────────────────────────────


def test_output_fresh_rescues_stall_conjunction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #813 degraded-CPU replay (the tick-47 shape minus the CPU signal
    #951 covers): every log stale > stall_sec, GPUs idle, CPU probe degraded
    (``session_cpu=unknown`` -> rate None), but ONE issue-keyed output file
    modified 30s ago -> verdict ``running``, not ``stalled``."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        stdout=_probe_stdout(
            pod_now=now,
            mtime_epoch=now - 2000,
            tail=_STALE_TAIL,
            gpu_util="0,0,0,0,0,0,0,0",
            session_cpu="unknown",
            output_mtime_epoch=now - 30,
        ),
    )
    result = _poll(tmp_path / "poll-state.json")
    assert result.status == "running"
    assert result.stall_reason is None


@pytest.mark.parametrize("output_mtime_epoch", [None, 0, "stale"])
def test_output_stale_keeps_stalled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_mtime_epoch
) -> None:
    """Pre-#1033 behavior pinned: the same degraded-CPU tick with NO fresh
    output — the line absent (older probe), the literal ``0`` (nothing within
    the window), or a genuinely stale epoch — still reads ``stalled``."""
    now = int(time.time())
    epoch = now - 2000 if output_mtime_epoch == "stale" else output_mtime_epoch
    _patch_pod(
        monkeypatch,
        stdout=_probe_stdout(
            pod_now=now,
            mtime_epoch=now - 2000,
            tail=_STALE_TAIL,
            gpu_util="0,0,0,0,0,0,0,0",
            session_cpu="unknown",
            output_mtime_epoch=epoch,
        ),
    )
    result = _poll(tmp_path / "poll-state.json")
    assert result.status == "stalled"


def test_output_malformed_epoch_is_inert_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """r2 blocker pin (``malformed-output-mtime-crashes-poll``): a malformed /
    non-numeric ``OUTPUT_MTIME_EPOCH`` scalar — reachable via version skew (a
    pod running newer probe code than this VM) or a garbled/partial SSH line —
    must NOT crash the tick (plan AC#6: no new crash paths). The guarded parse
    folds it to ``0`` (-> the ``10**9`` inert sentinel), so the degraded-CPU
    stale tick returns the SAME verdict as the ``OUTPUT_MTIME_EPOCH=0`` case."""
    now = int(time.time())

    def _stdout(epoch: int | str) -> str:
        return _probe_stdout(
            pod_now=now,
            mtime_epoch=now - 2000,
            tail=_STALE_TAIL,
            gpu_util="0,0,0,0,0,0,0,0",
            session_cpu="unknown",
            output_mtime_epoch=epoch,
        )

    _patch_pod(monkeypatch, stdout=_stdout(0))
    baseline = _poll(tmp_path / "poll-state-zero.json")
    assert baseline.status == "stalled"

    _patch_pod(monkeypatch, stdout=_stdout("garbage"))
    malformed = _poll(tmp_path / "poll-state-garbage.json")
    assert (malformed.status, malformed.stall_reason) == (baseline.status, baseline.stall_reason)


def test_output_exactly_at_stall_sec_rescues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Boundary pin: ``output_mtime_ago == stall_sec`` EXACTLY does not meet
    the strict-``>`` conjunct (same semantics as the three log conjuncts), so
    the verdict stays ``running``."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        stdout=_probe_stdout(
            pod_now=now,
            mtime_epoch=now - 2000,
            tail=_STALE_TAIL,
            gpu_util="0,0,0,0,0,0,0,0",
            session_cpu="unknown",
            output_mtime_epoch=now - pp.DEFAULT_STALL_SEC,
        ),
    )
    result = _poll(tmp_path / "poll-state.json")
    assert result.status == "running"


# ── Ask 1: zombie-override fresh-output veto ──────────────────────────────────


def _zombie_tick2_stdout(now: int, *, output_mtime_epoch: int | None) -> str:
    """The #664 tick-2 regime: zombie candidate + all logs stale + session
    CPU advancing (the #518 override would rescue to ``running``); the
    seeded streak "1" makes one ``poll_once`` call represent tick 2."""
    return _probe_stdout(
        pod_now=now,
        mtime_epoch=now - 2000,
        tail=_STALE_TAIL,
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="1262130",
        output_mtime_epoch=output_mtime_epoch,
    )


def _zombie_tick2_state(now: int) -> dict[str, str]:
    return {
        "phase": "scoring",
        "last_phase_change_epoch": str(now - 7200),
        "session_cpu_secs": "4000.0",
        "max_cpu_secs": "4000.0",
        "zombie_streak": "1",
    }


def test_zombie_output_fresh_vetoes_and_resets_streak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A zombie candidate meeting the #826 stale-log conjunction on tick 2 is
    VETOED by a fresh issue-keyed output within
    ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` — full ``poll_once`` replay
    through the real parse -> staleness -> threading path (a dropped
    call-site kwarg must not pass green). Streak RESETS (mirror of
    ``test_zombie_fresh_log_vetoes_and_resets_streak``)."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    _seed_state(state_file, _zombie_tick2_state(now))
    _patch_pod(monkeypatch, stdout=_zombie_tick2_stdout(now, output_mtime_epoch=now - 30))
    result = _poll(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_state(state_file)["zombie_streak"] == "0"


def test_zombie_stale_output_still_fires_tick2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #664 true positive through the REAL parse -> staleness -> threading
    path with an explicit STALE non-zero ``OUTPUT_MTIME_EPOCH`` (now-2000):
    the fold must NOT weaken the hung-vLLM detection — tick 2 still overrides
    ``running -> stalled`` with the #664 reason."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    _seed_state(state_file, _zombie_tick2_state(now))
    _patch_pod(monkeypatch, stdout=_zombie_tick2_stdout(now, output_mtime_epoch=now - 2000))
    result = _poll(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_output_exactly_at_veto_boundary_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Boundary pin: ``output_mtime_ago`` EXACTLY at
    ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` vetoes (``<=`` — the same
    boundary semantics as the #826 fresh-log veto term)."""
    now = int(time.time())
    veto_sec = max(pp.ZOMBIE_VETO_FRESH_SEC, pp.DEFAULT_STALL_SEC)
    state_file = tmp_path / "poll-state.json"
    _seed_state(state_file, _zombie_tick2_state(now))
    _patch_pod(monkeypatch, stdout=_zombie_tick2_stdout(now, output_mtime_epoch=now - veto_sec))
    result = _poll(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_state(state_file)["zombie_streak"] == "0"


def test_zombie_direct_call_output_default_inf(caplog: pytest.LogCaptureFixture) -> None:
    """Direct ``_apply_zombie_override`` call WITHOUT the new kwarg: the
    ``inf`` default keeps every pre-#1033 caller/test byte-unchanged — the
    tick-2 fire path is reached exactly as before (mirror of
    ``test_zombie_direct_call_rate_none_default``)."""
    status, reason, cpu_override, streak, wedge_streak = pp._apply_zombie_override(
        status="running",
        zombie_gpu_pids=["1262130"],
        stall_sec=900,
        last_mtime_ago=2000,
        phase_log_mtime_ago=10**9,
        shard_log_mtime_ago=10**9,
        prev_state={"zombie_streak": "1"},
        pod="pod-9813",
        cpu_override_active=True,
    )
    assert (status, reason, cpu_override, streak, wedge_streak) == (
        "stalled",
        "vllm_worker_dead_zombie_gpu",
        False,
        2,
        0,
    )


# ── Ask 1: probe heredoc text (boundedness + narrowness) ──────────────────────


def _capture_heredoc(monkeypatch: pytest.MonkeyPatch, *, stall_sec: int) -> str:
    captured: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: Any):
        captured.append(cmd[-1])
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(f"pod-{ISSUE}", LOG, PID, ISSUE, None, stall_sec=stall_sec)
    assert captured
    return captured[-1]


def test_output_probe_heredoc_is_bounded_and_issue_keyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed probe block is bounded (``timeout`` directly in front of
    ``find``), short-circuits (``-print -quit``), and passes ONLY issue-keyed
    roots to ``find`` (text-assertion style of
    ``test_gpu_probe_emits_namespace_count_keys``)."""
    heredoc = _capture_heredoc(monkeypatch, stall_sec=900)
    assert "OUTPUT_MTIME_EPOCH=" in heredoc
    assert "-print -quit" in heredoc
    assert re.search(r"timeout \d+ find ", heredoc), "find is not timeout-bounded"
    allowed_prefixes = (
        f"/workspace/explore-persona-space/eval_results/issue_{ISSUE}",
        f"/workspace/explore-persona-space/data/issue_{ISSUE}",
        f"/workspace/explore-persona-space/data/issue{ISSUE}",
    )
    find_blocks = re.findall(r"timeout \d+ find (.*?) -type f", heredoc)
    assert find_blocks, "no find path list found in the heredoc"
    for block in find_blocks:
        paths = block.split()
        assert paths
        for path in paths:
            assert path.startswith(allowed_prefixes), f"non-issue-keyed find root: {path}"
    # All four planned roots are present (incl. the data/issue<N> no-underscore
    # convention from the #854 sweep list).
    joined = " ".join(find_blocks)
    for prefix in allowed_prefixes:
        assert prefix in joined
    assert f"/workspace/explore-persona-space/eval_results/issue_{ISSUE}_*" in joined


def test_output_probe_two_stage_only_when_veto_wider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second (veto-window) find stage is emitted ONLY when
    ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec) > stall_sec`` (fast-smoke
    configs); at the default 900s stall window one find covers both reads."""
    assert "OUT_CUTOFF_VETO" not in _capture_heredoc(monkeypatch, stall_sec=900)
    fast = _capture_heredoc(monkeypatch, stall_sec=30)
    assert "OUT_CUTOFF_VETO" in fast
    assert f"OUT_NOW - {max(pp.ZOMBIE_VETO_FRESH_SEC, 30)}" in fast


def test_output_fold_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``EPM_POLL_OUTPUT_MTIME_FOLD=0``: (a) the probe block is omitted from
    the heredoc; (b) a stray fresh ``OUTPUT_MTIME_EPOCH`` line in the stdout
    (a pod running newer code) is left INERT — the degraded-CPU stale tick
    reads ``stalled`` exactly as pre-#1033."""
    monkeypatch.setattr(pp, "OUTPUT_MTIME_FOLD_ENABLED", False)
    assert "OUTPUT_MTIME_EPOCH" not in _capture_heredoc(monkeypatch, stall_sec=900)

    now = int(time.time())
    _patch_pod(
        monkeypatch,
        stdout=_probe_stdout(
            pod_now=now,
            mtime_epoch=now - 2000,
            tail=_STALE_TAIL,
            gpu_util="0,0,0,0,0,0,0,0",
            session_cpu="unknown",
            output_mtime_epoch=now - 30,  # fresh — but the fold is disabled
        ),
    )
    result = _poll(tmp_path / "poll-state.json")
    assert result.status == "stalled"


# ── Ask 2 (RunPod lane): relaunch restarts the idle clock ─────────────────────


def _idle_span_state(now: int, *, tripwire_run_epoch: str) -> dict[str, str]:
    """The #763-shape sidecar: a 543-min idle span accumulated by the
    PREVIOUS run, phase matching the current one (so the per-phase reset
    never fires), advisory not yet posted for this phase."""
    return {
        "phase": "scoring",
        "last_phase_change_epoch": str(now - 7200),
        "gpu_idle_since_epoch": str(now - 543 * 60),
        "gpu_idle_advised_phases": "",
        "gpu_idle_escalated_phases": "",
        "tripwire_run_epoch": tripwire_run_epoch,
        "session_cpu_secs": "unknown",
        "max_cpu_secs": "unknown",
        "zombie_streak": "0",
    }


def _healthy_idle_stdout(now: int) -> str:
    # Fresh logs (healthy run) + a single idle GPU: advisory territory only.
    return _probe_stdout(
        pod_now=now,
        mtime_epoch=now - 10,
        tail=_STALE_TAIL,
        gpu_util="0",
        session_cpu="unknown",
    )


def test_poll_once_relaunch_resets_idle_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RunPod-lane integration (#1033 Ask 2): a fresh ``epm:run-launched``
    (run age 120s, stored anchor from the PREVIOUS run) clears the idle keys
    alongside the #873 tripwire keys, so the advisory is NOT posted with the
    previous run's 543-min span and the persisted span re-anchors at or
    after the fresh run's launch epoch."""
    now = int(time.time())
    posted: list[dict] = []
    state_file = tmp_path / "poll-state.json"
    _seed_state(state_file, _idle_span_state(now, tripwire_run_epoch="1000"))
    _patch_pod(monkeypatch, stdout=_healthy_idle_stdout(now), posted=posted, run_age_sec=120.0)
    result = _poll(state_file)
    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is False
    assert not any("[gpu-idle-advisory]" in (p.get("note") or "") for p in posted)
    saved = _saved_state(state_file)
    assert int(saved["gpu_idle_since_epoch"]) >= now - 121  # >= the fresh run epoch


def test_poll_once_same_run_keeps_idle_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The no-reset control: a stored anchor within the 60s jitter tolerance
    of the current run epoch keeps the span, so the (legitimate) long-idle
    advisory still posts with the accumulated minutes."""
    now = int(time.time())
    posted: list[dict] = []
    state_file = tmp_path / "poll-state.json"
    # Current run epoch resolves to ~now-120; the stored anchor sits ~5s off.
    _seed_state(state_file, _idle_span_state(now, tripwire_run_epoch=str(now - 125)))
    _patch_pod(monkeypatch, stdout=_healthy_idle_stdout(now), posted=posted, run_age_sec=120.0)
    result = _poll(state_file)
    assert result.status == "running"
    assert result.gpu_idle_advisory_posted is True
    notes = [p.get("note") or "" for p in posted]
    assert any("[gpu-idle-advisory]" in n and "543 min" in n for n in notes)
