"""Tests for the GCP-lane GPU-idle advisory + escalation parity (#730).

The GCP poller (``scripts/backend_poll.py`` ``main()``) gained the same two
GPU-idle tiers the RunPod lane already has (advisory #518/#537, escalation
#664/#727), REUSING (importing, not re-implementing) the decision/post helpers
from ``scripts/poll_pipeline.py``:

* a fail-soft ``nvidia-smi``-over-``gcloud compute ssh`` GPU-util probe
  (``GcpBackend._gcp_gpu_util_probe``) — returns ``"unknown"`` on ANY failure;
* a sibling GPU-idle state file (``issue-<N>-gpu-idle-state.json``) read/written
  via ``backend_poll._{gpu_idle_state_path,load_gpu_idle_state,save_gpu_idle_state}``;
* the two RunPod-lane wiring fns wired into ``main()`` for
  ``handle.backend == "gcp" and status == "running"`` ticks, emitting two new
  serialized JSON fields ``gcp_gpu_idle_{advisory,escalation}_posted``.

These tests pin: the probe CSV parse + fail-soft contract; the advisory +
escalation thresholds on the GCP lane; the NEVER-stops-the-VM invariant (a
static argv guard); per-phase idempotency; and that ``_phase_is_cpu_only``
classifies the ACTUAL GCP ``eps/phase`` vocabulary (coarse ``"workload"``,
``"setup"``, ``"done"``, ``"unknown"``) the way the GCP lane threads it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

import scripts.backend_poll as bp
import scripts.poll_pipeline as pp
from explore_persona_space.backends.base import PollResult, RunHandle
from explore_persona_space.backends.gcp import GcloudRunResult, GcpBackend
from explore_persona_space.backends.issue_dispatch import write_handle_sidecar

# ── _gcp_gpu_util_probe parse + fail-soft ─────────────────────────────────────


def _backend_recording_run(returns: Any = None, *, raises: BaseException | None = None):
    """A GcpBackend whose injected runner RECORDS every argv it is handed.

    ``returns`` is the GcloudRunResult the runner yields (when not raising);
    ``raises`` makes the runner raise instead. ``backend.recorded`` is the list
    of argvs passed to the runner this test.
    """
    recorded: list[list[str]] = []

    def _runner(argv):
        recorded.append(list(argv))
        if raises is not None:
            raise raises
        return returns

    backend = GcpBackend(runner=_runner, marker_poster=lambda **_kw: None)
    backend.recorded = recorded  # type: ignore[attr-defined]
    return backend


def _gcp_handle(extra: dict | None = None) -> RunHandle:
    return RunHandle(
        backend="gcp",
        cluster=None,
        job_id="instance-fake-1",
        pod_name="eps-issue-730",
        scratch_dir="/workspace/eps-issue-730",
        log_path="/workspace/logs/issue-730.log",
        extra=dict(extra if extra is not None else {"issue": 730, "zone": "us-central1-a"}),
    )


def test_gcp_gpu_util_probe_parses_csv() -> None:
    """A newline-separated nvidia-smi reply (rc=0) is normalized to a comma-joined
    util string the consumer (_gpu_idle) understands."""
    backend = _backend_recording_run(
        GcloudRunResult(returncode=0, stdout="0\n0\n0\n0\n", stderr="")
    )
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "0,0,0,0"


def test_gcp_gpu_util_probe_normalizes_busy_cards() -> None:
    backend = _backend_recording_run(
        GcloudRunResult(returncode=0, stdout="0, 0, 95, 0\n", stderr="")
    )
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "0,0,95,0"


@pytest.mark.parametrize(
    "result,raises",
    [
        (GcloudRunResult(returncode=255, stdout="", stderr="ssh: connect refused"), None),
        (GcloudRunResult(returncode=0, stdout="", stderr=""), None),  # empty stdout
        (GcloudRunResult(returncode=0, stdout="garbage\nnot,a,number\n", stderr=""), None),
        (None, RuntimeError("transport blew up")),  # runner raises
    ],
)
def test_gcp_gpu_util_probe_fail_soft(result, raises) -> None:
    """rc!=0, empty stdout, a non-numeric token, and a raised exception EACH
    yield the literal "unknown" — never a crash, never a false idle."""
    backend = _backend_recording_run(result, raises=raises)
    assert backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a") == "unknown"


def test_gcp_gpu_util_probe_uses_sudo_nvidia_smi_over_ssh() -> None:
    """The probe reuses the GCP drain SSH pattern: gcloud compute ssh <name>
    --command=sudo -n nvidia-smi ... --zone=<zone> (matched to _drain_sentinels)."""
    backend = _backend_recording_run(GcloudRunResult(returncode=0, stdout="0\n", stderr=""))
    backend._gcp_gpu_util_probe(_gcp_handle(), "us-central1-a")
    (argv,) = backend.recorded  # type: ignore[attr-defined]
    joined = " ".join(argv)
    assert "compute" in argv and "ssh" in argv and "eps-issue-730" in argv
    assert "--zone=us-central1-a" in argv
    assert "sudo -n nvidia-smi" in joined
    assert "--query-gpu=utilization.gpu" in joined


# ── state-file round-trip ─────────────────────────────────────────────────────


def test_gpu_idle_state_path_is_handle_sidecar_sibling(tmp_path: Path) -> None:
    sidecar = tmp_path / "issue-730-handle.json"
    assert bp._gpu_idle_state_path(sidecar) == tmp_path / "issue-730-gpu-idle-state.json"


def test_gpu_idle_state_round_trip_and_fail_soft(tmp_path: Path) -> None:
    path = tmp_path / "issue-730-gpu-idle-state.json"
    assert bp._load_gpu_idle_state(path) == {}  # absent -> {}
    payload = {
        "phase": "p3_upload",
        "gpu_idle_since_epoch": "1000",
        "gpu_idle_advised_phases": "p3_upload",
        "gpu_idle_escalated_phases": "",
    }
    bp._save_gpu_idle_state(path, payload)
    assert bp._load_gpu_idle_state(path) == payload
    # Corrupt body -> {} (fail-soft), never raises.
    path.write_text("{not json")
    assert bp._load_gpu_idle_state(path) == {}
    # Non-dict JSON -> {}.
    path.write_text("[1, 2, 3]")
    assert bp._load_gpu_idle_state(path) == {}


# ── advisory + escalation thresholds on the GCP lane ──────────────────────────
#
# The decision cores are exhaustively unit-tested in
# tests/test_poll_gpu_idle_escalation.py; here we pin the GCP-lane WIRING
# (the imported _maybe_* fns + the seeded sibling state file drive the posts).


def _seed_idle_state(path: Path, *, since_epoch: int, phase: str) -> None:
    bp._save_gpu_idle_state(
        path,
        {
            "phase": phase,
            "gpu_idle_since_epoch": str(since_epoch),
            "gpu_idle_advised_phases": "",
            "gpu_idle_escalated_phases": "",
        },
    )


def test_gcp_advisory_posts_after_threshold(tmp_path: Path, monkeypatch) -> None:
    """All-idle GPUs in a CPU-only phase whose seeded idle span exceeds the
    advisory min -> the advisory wiring posts a [gpu-idle-advisory] marker."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    path = tmp_path / "issue-730-gpu-idle-state.json"
    now = 100_000
    _seed_idle_state(path, since_epoch=now - pp.GPU_IDLE_ADVISORY_MIN * 60, phase="p3_upload")
    prev = bp._load_gpu_idle_state(path)
    _idle_since, advised, advisory_posted = pp._maybe_post_gpu_idle_advisory(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="p3_upload",
        prev_state=prev,
        now_epoch=now,
    )
    assert advisory_posted is True
    assert "p3_upload" in advised
    assert any(p["key"] == "epm:progress" and p.get("gpu_idle_advisory") for p in posted)
    assert any("[gpu-idle-advisory]" in (p.get("note") or "") for p in posted)


def test_gcp_escalation_posts_and_pushes_multi_gpu(tmp_path: Path, monkeypatch) -> None:
    """A MULTI-GPU pod idle past the escalation min in a CPU-only phase ->
    [gpu-idle-escalation] marker posted AND a Telegram push fired; a single-GPU
    pod under the SAME conditions does NOT escalate."""
    posted: list[dict] = []
    pushes: list[str] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: pushes.append(msg) or True)
    now = 100_000
    since = now - pp.GPU_IDLE_ESCALATION_MIN * 60

    # Multi-GPU -> escalates + pushes.
    escalated, escalation_posted = pp._maybe_escalate_gpu_idle(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0,0,0,0,0,0,0,0",
        current_phase="p3_upload",
        idle_since_epoch=since,
        prev_state={"gpu_idle_escalated_phases": ""},
        now_epoch=now,
    )
    assert escalation_posted is True
    assert "p3_upload" in escalated
    assert len(pushes) == 1
    assert any(p["key"] == "epm:progress" and p.get("gpu_idle_escalation") for p in posted)

    # Single-GPU under identical conditions -> NO escalation, NO push.
    pushes.clear()
    _escalated, single_posted = pp._maybe_escalate_gpu_idle(
        issue=730,
        pod="eps-issue-730",
        status="running",
        gpu_util="0",
        current_phase="p3_upload",
        idle_since_epoch=since,
        prev_state={"gpu_idle_escalated_phases": ""},
        now_epoch=now,
    )
    assert single_posted is False
    assert pushes == []


def test_gcp_idempotent_one_per_phase(tmp_path: Path, monkeypatch) -> None:
    """Two consecutive ticks in the SAME phase (state round-tripped through the
    sibling file) -> escalate on tick 1, NOT tick 2; a phase CHANGE re-arms."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)
    path = tmp_path / "issue-730-gpu-idle-state.json"
    now = 100_000
    since = now - pp.GPU_IDLE_ESCALATION_MIN * 60
    _seed_idle_state(path, since_epoch=since, phase="p3_upload")

    def _tick(phase: str, now_epoch: int) -> bool:
        prev = bp._load_gpu_idle_state(path)
        idle_since, advised, _adv = pp._maybe_post_gpu_idle_advisory(
            issue=730,
            pod="eps-issue-730",
            status="running",
            gpu_util="0,0,0,0,0,0,0,0",
            current_phase=phase,
            prev_state=prev,
            now_epoch=now_epoch,
        )
        escalated, escalation_posted = pp._maybe_escalate_gpu_idle(
            issue=730,
            pod="eps-issue-730",
            status="running",
            gpu_util="0,0,0,0,0,0,0,0",
            current_phase=phase,
            idle_since_epoch=idle_since,
            prev_state=prev,
            now_epoch=now_epoch,
        )
        bp._save_gpu_idle_state(
            path,
            {
                "phase": phase,
                "gpu_idle_since_epoch": str(idle_since),
                "gpu_idle_advised_phases": ",".join(sorted(advised)),
                "gpu_idle_escalated_phases": ",".join(sorted(escalated)),
            },
        )
        return escalation_posted

    assert _tick("p3_upload", now) is True  # tick 1: fires
    assert _tick("p3_upload", now + 60) is False  # tick 2, same phase: de-duped
    # A phase change restarts the span -> the new phase has not yet aged, so it
    # does NOT immediately escalate (re-arm, then age past the threshold).
    assert _tick("p5_upload", now + 120) is False
    assert _tick("p5_upload", now + 120 + pp.GPU_IDLE_ESCALATION_MIN * 60) is True


# ── _phase_is_cpu_only on the ACTUAL GCP eps/phase vocabulary ─────────────────
#
# On a RUNNING GCP VM the current_phase threaded into backend_poll.main() is the
# COARSE eps/phase guest attribute, whose only mid-workload value is the literal
# "workload" (gcp.py). The fine dispatcher phases (p3_upload) appear on the
# RunPod lane; they are still asserted here because the deny-list is shared.


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("workload", True),  # the GCP coarse mid-workload phase -> eligible
        ("done", True),  # no deny-list substring (gated out earlier by status!=running anyway)
        ("p3_upload", True),  # RunPod-lane fine phase, shared deny-list
        ("upload", True),
        ("setup", False),  # deny-list substring
        ("setup_failed", False),
        ("train", False),
        ("eval", False),
        ("unknown", False),  # the explicit ineligible sentinel
        ("", False),
    ],
)
def test_gcp_phase_deny_list_matches(phase: str, expected: bool) -> None:
    assert pp._phase_is_cpu_only(phase) is expected


def test_gpu_required_substrings_match_assertions() -> None:
    """The substrings the GCP test asserts denied are actually in the shared
    deny-list (guards against the deny-list drifting out from under this test)."""
    assert {"train", "eval", "setup"} <= set(pp.GPU_REQUIRED_PHASE_SUBSTRINGS)


# ── the NEVER-stops-the-VM invariant (static argv guard) ──────────────────────


class _IdlePollBackend:
    """A GcpBackend-shaped poll double for the main() integration test.

    Records every argv the injected runner sees (so the no-VM-stop guard can
    assert no stop/delete shape), returns a scripted RUNNING PollResult from
    poll(), and a scripted all-idle gpu_util from the probe. Carries a real
    _config so backend._config.primary_zone resolves.
    """

    def __init__(self, *, gpu_util: str, current_phase: str) -> None:
        from explore_persona_space.backends.gcp import default_gcp_config

        self._config = default_gcp_config()
        self._gpu_util = gpu_util
        self._current_phase = current_phase
        self.run_argvs: list[list[str]] = []

    def poll(self, handle: RunHandle) -> PollResult:
        return PollResult(
            status="running",
            current_phase=self._current_phase,
            new_milestone=False,
            last_log_mtime_sec_ago=10,
            pid_alive=True,
            log_tail_excerpt="",
        )

    def _gcp_gpu_util_probe(self, handle: RunHandle, zone: str) -> str:
        # Record a representative probe argv so the no-stop guard sees the real
        # SSH shape the production probe would emit.
        self.run_argvs.append(
            ["gcloud", "compute", "ssh", handle.pod_name, f"--zone={zone}", "nvidia-smi"]
        )
        return self._gpu_util


_FORBIDDEN_ARGV_SHAPES = (
    ("instances", "stop"),
    ("instances", "delete"),
)


def _argv_is_vm_stop(argv: list[str]) -> bool:
    joined = " ".join(argv)
    if any(all(tok in argv for tok in shape) for shape in _FORBIDDEN_ARGV_SHAPES):
        return True
    return "pod.py" in joined and (" stop" in joined or " terminate" in joined)


def test_gcp_no_vm_stop_in_codepath(tmp_path, monkeypatch, capsys) -> None:
    """At the escalation threshold the GCP GPU-idle codepath posts a marker +
    push but issues NO VM-stopping action: no `gcloud ... instances stop|delete`
    and no `pod.py ... stop|terminate` reaches the runner OR subprocess.run."""
    import subprocess

    subprocess_argvs: list[list[str]] = []
    real_run = subprocess.run

    def _recording_run(argv, *a, **kw):
        if isinstance(argv, (list, tuple)):
            subprocess_argvs.append(list(argv))
        return real_run(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", _recording_run)
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: None)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    _seed_idle_state(
        state_path, since_epoch=now - pp.GPU_IDLE_ESCALATION_MIN * 60, phase="workload"
    )

    backend = _IdlePollBackend(gpu_util="0,0,0,0,0,0,0,0", current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0

    # No VM-stop argv in EITHER channel.
    for argv in backend.run_argvs + subprocess_argvs:
        assert not _argv_is_vm_stop(argv), f"forbidden VM-stop argv reached the codepath: {argv}"


# ── main() integration: the two serialized JSON fields ────────────────────────


def _last_json_line(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "backend_poll printed no stdout"
    return json.loads(out.splitlines()[-1])


def test_backend_poll_main_gcp_idle_integration(tmp_path, monkeypatch, capsys) -> None:
    """Driving main() on a GCP handle with a RUNNING poll + all-idle probe + a
    pre-seeded idle span past the escalation min emits both new JSON fields and
    drives the posted flags."""
    posted: list[dict] = []
    monkeypatch.setattr(
        pp, "post_event", lambda issue, key, **kw: posted.append({"key": key, **kw})
    )
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: True)

    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(_gcp_handle(), sidecar)
    state_path = bp._gpu_idle_state_path(sidecar)
    now = int(time.time())
    _seed_idle_state(
        state_path, since_epoch=now - pp.GPU_IDLE_ESCALATION_MIN * 60, phase="workload"
    )

    backend = _IdlePollBackend(gpu_util="0,0,0,0,0,0,0,0", current_phase="workload")
    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: backend)

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["status"] == "running"
    assert out["gcp_gpu_idle_advisory_posted"] is True
    assert out["gcp_gpu_idle_escalation_posted"] is True
    # The state file was updated with the escalated phase (idempotency surface).
    saved = bp._load_gpu_idle_state(state_path)
    assert "workload" in saved["gpu_idle_escalated_phases"]


def test_backend_poll_main_non_gcp_omits_idle_fields_defaulting_false(
    tmp_path, monkeypatch, capsys
) -> None:
    """A non-GCP (RunPod) tick routed through main() leaves both fields False —
    the GCP block is gated on handle.backend == 'gcp' (RunPod's own advisory /
    escalation fires inside poll_pipeline.poll_once, not here, so no double-fire)."""
    runpod_handle = RunHandle(
        backend="runpod",
        cluster=None,
        job_id="pod-fake",
        pod_name="pod-730",
        scratch_dir="/workspace",
        log_path="/workspace/logs/issue-730.log",
        extra={"issue": 730},
    )
    sidecar = tmp_path / "issue-730-handle.json"
    write_handle_sidecar(runpod_handle, sidecar)

    class _RunningRunpod:
        def poll(self, handle: RunHandle) -> PollResult:
            return PollResult(
                status="running",
                current_phase="train",
                new_milestone=False,
                last_log_mtime_sec_ago=5,
                pid_alive=True,
                log_tail_excerpt="",
            )

    monkeypatch.setattr("scripts.backend_poll._resolve_backend", lambda name: _RunningRunpod())

    rc = bp.main(["--issue", "730", "--handle-file", str(sidecar)])
    assert rc == 0
    out = _last_json_line(capsys)
    assert out["gcp_gpu_idle_advisory_posted"] is False
    assert out["gcp_gpu_idle_escalation_posted"] is False
