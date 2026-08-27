"""Tests for the #2624 no-CUDA-context advisory in scripts/poll_pipeline.py.

A GPU-lane phase whose cards ALL sit at ~0 MiB of memory.used for
``GPU_NOCTX_ADVISORY_MIN`` minutes across ``GPU_NOCTX_MIN_TICKS``
consecutive samples has never allocated a CUDA context (incident #2546:
a 4x H100 pod ran ``p5_fits`` entirely CPU-bound while the dispatcher
logged ``alloc=0,1,2,3``). These tests pin:

* the probe emission + key parity for the NEW ``GPU_MEM_MIB`` sample (and
  that the pre-existing ``GPU_UTIL`` emission fragment is byte-identical);
* ``_gpu_mem_all_zero`` — the csv threshold predicate (fail-safe on
  unknown/unparseable, ``<=`` boundary);
* ``_gpu_noctx_update`` — the pure decision core (span + tick floors,
  resets on non-running/nonzero/unknown, phase-change restart, per-phase
  de-dup, disable lever);
* ``_maybe_post_gpu_noctx_advisory`` — the wiring (note shape + extras +
  push, post-failure retry, corrupted-state tolerance, state round-trip,
  relaunch reset via ``_RUN_SCOPED_STATE_KEYS`` / ``_tripwire_run_scope``);
* the #2546 incident replay (fires at tick 4, ~27 min, exactly once) and
  the healthy-GPU / CPU-preamble negative fixtures;
* the shell-level probe branch semantics (stubbed nvidia-smi vs absent).

GPU_MEM_MIB parse/key-parity cases are HOSTED HERE rather than appended to
tests/test_poll_pipeline_zombie_gpu.py (stated plan-§12 deviation: one
sibling-file edit fewer; the cases are additive either way).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_gpu_noctx_under_test")


# ── _gpu_mem_all_zero (csv threshold predicate) ──────────────────────────────


def test_gpu_mem_all_zero_contract() -> None:
    assert pp._gpu_mem_all_zero("0,0,0,0", 64) is True
    assert pp._gpu_mem_all_zero("0, 0, 0, 0", 64) is True  # spaces tolerated
    assert pp._gpu_mem_all_zero("64,0", 64) is True  # <= boundary is inclusive
    assert pp._gpu_mem_all_zero("65,0", 64) is False  # one card above -> False
    assert pp._gpu_mem_all_zero("0", 64) is True  # single-GPU pod
    assert pp._gpu_mem_all_zero("81000,80000", 64) is False  # healthy training
    # Fail-safe: unknown / unparseable NEVER accumulates toward an advisory.
    assert pp._gpu_mem_all_zero("unknown", 64) is False
    assert pp._gpu_mem_all_zero("", 64) is False
    assert pp._gpu_mem_all_zero("garbage", 64) is False
    assert pp._gpu_mem_all_zero("0,abc", 64) is False  # ANY bad token -> False
    assert pp._gpu_mem_all_zero("[N/A]", 64) is False  # nvidia-smi N/A form
    assert pp._gpu_mem_all_zero(",", 64) is False  # zero tokens -> False
    # Threshold is a parameter (EPM_GPU_NOCTX_MEM_MAX_MIB lever).
    assert pp._gpu_mem_all_zero("100,90", 128) is True
    assert pp._gpu_mem_all_zero("100,90", 8) is False


# ── _gpu_noctx_update (pure decision core) ───────────────────────────────────

_KW: dict[str, Any] = dict(
    status="running",
    gpu_mem_mib="0,0,0,0",
    current_phase="p5_fits",
    prev_phase="p5_fits",
    prev_since_epoch=0,
    prev_zero_ticks=0,
    advised_phases=set(),
    now_epoch=100_000,
    advisory_min=20,
    min_ticks=3,
    max_mib=64,
)


def _update(**over: Any):
    return pp._gpu_noctx_update(**{**_KW, **over})


def test_core_disabled_by_nonpositive_minutes() -> None:
    for minutes in (0, -5):
        u = _update(advisory_min=minutes, prev_since_epoch=90_000, prev_zero_ticks=10)
        assert (u.should_post, u.since_epoch, u.zero_ticks, u.span_sec) == (False, 0, 0, 0)


def test_core_resets_on_non_running_status() -> None:
    for status in ("stalled", "dead", "done", "no_recent_log"):
        u = _update(status=status, prev_since_epoch=90_000, prev_zero_ticks=5)
        assert (u.should_post, u.since_epoch, u.zero_ticks) == (False, 0, 0)


def test_core_resets_on_nonzero_or_unknown_sample() -> None:
    for mem in ("81000,0,0,0", "unknown", "", "garbage"):
        u = _update(gpu_mem_mib=mem, prev_since_epoch=90_000, prev_zero_ticks=5)
        assert (u.should_post, u.since_epoch, u.zero_ticks) == (False, 0, 0)


def test_core_starts_span_on_first_zero_sample() -> None:
    u = _update(prev_since_epoch=0, prev_zero_ticks=0)
    assert u.should_post is False
    assert u.since_epoch == _KW["now_epoch"]
    assert u.zero_ticks == 1
    assert u.span_sec == 0


def test_core_phase_change_restarts_span() -> None:
    u = _update(prev_phase="p4_stage", prev_since_epoch=90_000, prev_zero_ticks=7)
    assert u.should_post is False
    assert u.since_epoch == _KW["now_epoch"]  # fresh warm-up allowance
    assert u.zero_ticks == 1


def test_core_accumulates_below_thresholds_without_posting() -> None:
    # Span met, ticks below the floor -> no post (the 1800s-cadence guard).
    u = _update(prev_since_epoch=100_000 - 1300, prev_zero_ticks=1)
    assert u.zero_ticks == 2
    assert u.span_sec == 1300
    assert u.should_post is False
    # Ticks met, span below the floor -> no post.
    u = _update(prev_since_epoch=100_000 - 600, prev_zero_ticks=4)
    assert u.zero_ticks == 5
    assert u.should_post is False


def test_core_fires_at_exact_boundaries() -> None:
    # span == advisory_min*60 AND ticks == min_ticks (>= semantics).
    u = _update(prev_since_epoch=100_000 - 1200, prev_zero_ticks=2)
    assert u.span_sec == 1200
    assert u.zero_ticks == 3
    assert u.should_post is True


def test_core_per_phase_dedup_suppresses_repost() -> None:
    u = _update(prev_since_epoch=100_000 - 5000, prev_zero_ticks=9, advised_phases={"p5_fits"})
    assert u.should_post is False
    assert u.zero_ticks == 10  # span keeps accumulating; only the post is deduped


# ── _maybe_post_gpu_noctx_advisory (wiring) ──────────────────────────────────


def _capture_channels(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict], list[str]]:
    posts: list[dict] = []
    pushes: list[str] = []

    def _fake_post_event(issue: int, kind: str, **kwargs: Any) -> None:
        posts.append({"issue": issue, "kind": kind, **kwargs})

    monkeypatch.setattr(pp, "post_event", _fake_post_event)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: pushes.append(msg) or True)
    return posts, pushes


def _wiring(now: int, state: dict[str, str], mem: str = "0,0,0,0", status: str = "running"):
    return pp._maybe_post_gpu_noctx_advisory(
        issue=2546,
        pod="pod-2546-arm2",
        status=status,
        gpu_mem_mib=mem,
        current_phase="p5_fits",
        prev_state=state,
        now_epoch=now,
    )


def test_wiring_below_threshold_returns_span_no_post(monkeypatch: pytest.MonkeyPatch) -> None:
    posts, pushes = _capture_channels(monkeypatch)
    since, ticks, advised, posted = _wiring(100_000, {})
    assert posted is False
    assert since == 100_000
    assert ticks == 1
    assert advised == set()
    assert posts == [] and pushes == []


def test_wiring_fires_with_note_shape_extras_and_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts, pushes = _capture_channels(monkeypatch)
    state = {
        "phase": "p5_fits",
        "gpu_noctx_since_epoch": str(100_000 - 1500),
        "gpu_noctx_zero_ticks": "3",
        "gpu_noctx_advised_phases": "",
    }
    since, ticks, advised, posted = _wiring(100_000, state)
    assert posted is True
    assert since == 100_000 - 1500
    assert ticks == 4
    assert advised == {"p5_fits"}
    assert len(posts) == 1
    ev = posts[0]
    assert ev["issue"] == 2546
    assert ev["kind"] == "epm:progress"
    assert ev["by"] == "poll_pipeline"
    assert ev["phase"] == "p5_fits"
    assert ev["pod"] == "pod-2546-arm2"
    assert ev["gpu_no_cuda_context"] is True
    note = ev["note"]
    assert note.startswith("[gpu-no-cuda-context]")
    assert "4 GPU(s)" in note
    assert "25 min" in note  # 1500s span
    assert "4 consecutive samples" in note
    assert "memory.used <= 64 MiB" in note
    assert "NEVER allocated a CUDA context" in note
    # The RECORD -> FINISH -> FILE runbook pointer + the two critic-mandated
    # clauses (disregard escape + advisory-only statement).
    assert "pods.md" in note and "Mid-run discovery" in note
    assert "RECORD" in note and "FINISH" in note and "FILE" in note
    assert "If this phase legitimately allocates later" in note
    assert "disregard" in note
    assert "nothing was stopped" in note
    # Fail-soft phone push fired once, naming the pod + phase.
    assert len(pushes) == 1
    assert "NO CUDA CONTEXT" in pushes[0]
    assert "pod-2546-arm2" in pushes[0]
    assert "phase=p5_fits" in pushes[0]
    assert "nothing stopped" in pushes[0]


def test_wiring_post_failure_retries_next_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[str] = []

    def _boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("events.jsonl commit deferred")

    monkeypatch.setattr(pp, "post_event", _boom)
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: pushes.append(msg) or True)
    state = {
        "phase": "p5_fits",
        "gpu_noctx_since_epoch": str(100_000 - 1500),
        "gpu_noctx_zero_ticks": "3",
    }
    since, ticks, advised, posted = _wiring(100_000, state)
    assert posted is False
    assert advised == set()  # phase NOT recorded -> next tick retries
    assert pushes == []  # push never precedes the durable record
    # The span survives so the retry can fire immediately.
    assert since == 100_000 - 1500
    assert ticks == 4


def test_wiring_push_failure_never_blocks_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    posts, _ = _capture_channels(monkeypatch)

    def _push_boom(msg: str) -> bool:
        raise RuntimeError("telegram down")

    # _telegram_push is documented fail-soft (returns False, never raises) —
    # mirror the escalation wiring by asserting the advisory still records
    # when the push returns False.
    monkeypatch.setattr(pp, "_telegram_push", lambda msg: False)
    state = {
        "phase": "p5_fits",
        "gpu_noctx_since_epoch": str(100_000 - 1500),
        "gpu_noctx_zero_ticks": "3",
    }
    _, _, advised, posted = _wiring(100_000, state)
    assert posted is True
    assert advised == {"p5_fits"}
    assert len(posts) == 1
    del _push_boom  # documented-contract note only


def test_wiring_dedup_and_corrupted_state_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    posts, pushes = _capture_channels(monkeypatch)
    # Already-advised phase -> no repost.
    state = {
        "phase": "p5_fits",
        "gpu_noctx_since_epoch": str(100_000 - 5000),
        "gpu_noctx_zero_ticks": "9",
        "gpu_noctx_advised_phases": "p5_fits,p2_train",
    }
    _, _, advised, posted = _wiring(100_000, state)
    assert posted is False
    assert advised == {"p5_fits", "p2_train"}
    assert posts == [] and pushes == []
    # Corrupted persisted ints -> treated as 0 (fresh span), never a crash.
    state = {
        "phase": "p5_fits",
        "gpu_noctx_since_epoch": "garbage",
        "gpu_noctx_zero_ticks": "NaNish",
    }
    since, ticks, advised, posted = _wiring(100_000, state)
    assert posted is False
    assert since == 100_000
    assert ticks == 1


# ── state persistence + relaunch run-scoping ─────────────────────────────────


def test_state_round_trip_continues_span(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two-tick ``_save_state``/``_load_state`` str->str round-trip: the
    persisted keys reload into exactly the span the next tick continues."""
    posts, _ = _capture_channels(monkeypatch)
    state_file = tmp_path / "poll-state.json"
    since1, ticks1, advised1, _ = _wiring(100_000, {})
    pp._save_state(
        state_file,
        2546,
        {
            "phase": "p5_fits",
            "gpu_noctx_since_epoch": str(since1),
            "gpu_noctx_zero_ticks": str(ticks1),
            "gpu_noctx_advised_phases": ",".join(sorted(advised1)),
        },
    )
    loaded = pp._load_state(state_file, 2546)
    since2, ticks2, _, posted2 = _wiring(100_000 + 540, loaded)
    assert posted2 is False
    assert since2 == since1  # span anchor survived the round-trip
    assert ticks2 == 2
    assert posts == []


def test_noctx_keys_are_run_scoped() -> None:
    for key in (
        "gpu_noctx_since_epoch",
        "gpu_noctx_zero_ticks",
        "gpu_noctx_advised_phases",
    ):
        assert key in pp._RUN_SCOPED_STATE_KEYS


def test_relaunch_resets_noctx_state_via_tripwire_run_scope() -> None:
    now = 200_000
    prev = {
        "phase": "p5_fits",
        "tripwire_run_epoch": str(now - 50_000),  # stored anchor: the OLD run
        "gpu_noctx_since_epoch": str(now - 5000),
        "gpu_noctx_zero_ticks": "8",
        "gpu_noctx_advised_phases": "p5_fits",
    }
    # A fresh epm:run-launched (run_age 60s) is newer than the stored anchor
    # by far more than the jitter tolerance -> run-scoped keys are cleared.
    scoped, anchor = pp._tripwire_run_scope(prev, run_age_sec=60.0, now_epoch=now)
    for key in (
        "gpu_noctx_since_epoch",
        "gpu_noctx_zero_ticks",
        "gpu_noctx_advised_phases",
    ):
        assert key not in scoped
    assert anchor == now - 60
    assert scoped.get("phase") == "p5_fits"  # non-run-scoped keys survive


# ── incident replay + negative fixtures ──────────────────────────────────────


def _replay(
    monkeypatch: pytest.MonkeyPatch,
    samples: list[tuple[int, str]],
    phase: str = "p5_fits",
) -> tuple[list[bool], list[dict], list[str]]:
    posts, pushes = _capture_channels(monkeypatch)
    state: dict[str, str] = {}
    fired: list[bool] = []
    for now, mem in samples:
        since, ticks, advised, posted = pp._maybe_post_gpu_noctx_advisory(
            issue=2546,
            pod="pod-2546-arm2",
            status="running",
            gpu_mem_mib=mem,
            current_phase=phase,
            prev_state=state,
            now_epoch=now,
        )
        fired.append(posted)
        state = {
            "phase": phase,
            "gpu_noctx_since_epoch": str(since),
            "gpu_noctx_zero_ticks": str(ticks),
            "gpu_noctx_advised_phases": ",".join(sorted(advised)),
        }
    return fired, posts, pushes


def test_incident_2546_replay_fires_once_at_tick_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2546 shape: 4x H100 at 0 MiB through p5_fits, default 540s cadence.
    Tick 4 sits at span 1620s >= 1200s with 4 >= 3 consecutive samples ->
    exactly ONE marker + ONE push, ~27 min after the span opened; a fifth
    tick never reposts (per-phase de-dup)."""
    t0 = 1_000_000
    samples = [(t0 + i * 540, "0,0,0,0") for i in range(5)]
    fired, posts, pushes = _replay(monkeypatch, samples)
    assert fired == [False, False, False, True, False]
    assert len(posts) == 1
    assert len(pushes) == 1
    assert "27 min" in posts[0]["note"]  # 1620s // 60


def test_healthy_training_never_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    t0 = 1_000_000
    samples = [(t0 + i * 540, "81000,80500,79900,81200") for i in range(8)]
    fired, posts, pushes = _replay(monkeypatch, samples)
    assert fired == [False] * 8
    assert posts == [] and pushes == []


def test_cpu_preamble_then_allocation_never_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legitimate ~15-min CPU preamble (staging/tokenization) that then
    allocates stays BELOW the 20-min floor and the allocation resets the
    span — no advisory, by design (the note's 'disregard' clause covers the
    rarer long-preamble case)."""
    t0 = 1_000_000
    samples = [(t0 + i * 300, "0,0,0,0") for i in range(4)]  # 0..900s all-zero
    samples += [(t0 + 1200 + i * 540, "42000,41000,40000,43000") for i in range(4)]
    fired, posts, pushes = _replay(monkeypatch, samples)
    assert fired == [False] * 8
    assert posts == [] and pushes == []


# ── probe emission + key parity (GPU_MEM_MIB; GPU_UTIL byte-identical) ──────


def test_gpu_probe_emits_mem_key_and_keeps_gpu_util_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer-side emission / key-parity pin (the #607 producer->parser
    contract hole, mirroring test_gpu_probe_emits_namespace_count_keys):
    captures the REAL probe text ``_ssh_probe`` sends and asserts (a) the
    GPU_MEM_MIB emission in the nvidia-smi branch + its ``=unknown`` twin
    in the else branch, via a SEPARATE memory.used query; (b) the
    pre-existing GPU_UTIL emission fragment is byte-identical (#2624
    acceptance criterion 1); (c) key parity across ``_PROBE_SCALAR_KEYS``,
    the parser defaults, and the ssh-failed fallback dict."""
    captured: dict[str, str] = {}

    def _fake_run(cmd: list[str], **kwargs: Any):
        captured["remote"] = cmd[-1]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(
        "pod-2546-arm2",
        "/workspace/logs/issue-2546.log",
        "/workspace/logs/issue-2546.pid",
        2546,
    )
    remote = captured["remote"]

    # (a) emission tokens: nvidia-smi branch + else-branch unknown twin.
    assert 'echo "GPU_MEM_MIB=${GPU_MEM_OUT:-unknown}"' in remote
    assert 'echo "GPU_MEM_MIB=unknown"' in remote
    assert "--query-gpu=memory.used" in remote
    # (b) the GPU_UTIL emission is the untouched byte-identical fragment.
    assert 'echo "GPU_UTIL=${GPU_OUT:-unknown}"; ' in remote
    assert 'echo "GPU_UTIL=unknown"; ' in remote
    assert remote.count("GPU_UTIL=") == 2  # exactly the two pre-existing sites

    # (c) key parity: emitted key -> _PROBE_SCALAR_KEYS -> parser default ->
    # ssh-failed fallback dict.
    assert "GPU_MEM_MIB" in pp._PROBE_SCALAR_KEYS
    assert pp._parse_probe_stdout("")["gpu_mem_mib"] == "unknown"

    def _fake_run_fail(cmd: list[str], **kwargs: Any):
        return subprocess.CompletedProcess(args=cmd, returncode=255, stdout="", stderr="down")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run_fail)
    fallback = pp._ssh_probe(
        "pod-2546-arm2",
        "/workspace/logs/issue-2546.log",
        "/workspace/logs/issue-2546.pid",
        2546,
    )
    assert fallback["gpu_mem_mib"] == "unknown"

    # Parser round-trip on a realistic probe stdout.
    parsed = pp._parse_probe_stdout("GPU_UTIL=0,0,0,0\nGPU_MEM_MIB=0,0,0,0\n")
    assert parsed["gpu_util"] == "0,0,0,0"
    assert parsed["gpu_mem_mib"] == "0,0,0,0"


# ── shell-level probe execution (both branches) ──────────────────────────────


def _extract_gpu_probe_snippet() -> str:
    """Slice the gpu_probe block out of the REAL remote command text."""
    captured: dict[str, str] = {}

    def _fake_run(cmd: list[str], **kwargs: Any):
        captured["remote"] = cmd[-1]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    orig = pp.subprocess.run
    pp.subprocess.run = _fake_run  # type: ignore[assignment]
    try:
        pp._ssh_probe("pod-x", "/workspace/logs/issue-1.log", "/workspace/logs/issue-1.pid", 1)
    finally:
        pp.subprocess.run = orig  # type: ignore[assignment]
    remote = captured["remote"]
    start = remote.index("if command -v nvidia-smi")
    end_tok = 'echo "NVIDIA_UVM_ALLOC_HOLDERS=unknown"; fi; '
    end = remote.index(end_tok) + len(end_tok)
    return remote[start:end]


def _tool_dir(tmp_path: Path, *, with_nvidia_smi: bool) -> Path:
    tools = tmp_path / ("tools_smi" if with_nvidia_smi else "tools_nosmi")
    tools.mkdir()
    for name in ("paste", "tr", "grep", "ls", "cat"):
        for candidate in (f"/usr/bin/{name}", f"/bin/{name}"):
            if os.path.exists(candidate):
                (tools / name).symlink_to(candidate)
                break
    if with_nvidia_smi:
        stub = tools / "nvidia-smi"
        # POSIX sh + absolute shebang: the snippet runs under PATH=tooldir,
        # where an `/usr/bin/env bash` shebang would fail to resolve.
        stub.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            "  *utilization.gpu*) printf '0\\n0\\n0\\n0\\n' ;;\n"
            "  *memory.used*) printf '0\\n0\\n0\\n0\\n' ;;\n"
            "  *query-compute-apps*) exit 0 ;;\n"
            "esac\n"
        )
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return tools


def _run_snippet(snippet: str, tools: Path) -> dict[str, str]:
    bash = shutil.which("bash")
    assert bash, "bash not found on the host PATH"
    proc = subprocess.run(
        [bash, "-c", snippet],
        capture_output=True,
        text=True,
        env={"PATH": str(tools)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def test_probe_shell_semantics_with_stub_nvidia_smi(tmp_path: Path) -> None:
    """Executes the REAL probe shell (then-branch) against a stub nvidia-smi
    answering four zero rows: GPU_MEM_MIB must join to the ``0,0,0,0`` csv
    the parser + predicate consume, alongside the unchanged GPU_UTIL csv."""
    snippet = _extract_gpu_probe_snippet()
    out = _run_snippet(snippet, _tool_dir(tmp_path, with_nvidia_smi=True))
    assert out["GPU_UTIL"] == "0,0,0,0"
    assert out["GPU_MEM_MIB"] == "0,0,0,0"
    assert pp._gpu_mem_all_zero(out["GPU_MEM_MIB"], pp.GPU_NOCTX_MEM_MAX_MIB) is True


def test_probe_shell_semantics_without_nvidia_smi(tmp_path: Path) -> None:
    """else-branch: no nvidia-smi on PATH -> both keys read ``unknown`` and
    the predicate fails safe (never accumulates toward an advisory)."""
    snippet = _extract_gpu_probe_snippet()
    out = _run_snippet(snippet, _tool_dir(tmp_path, with_nvidia_smi=False))
    assert out["GPU_UTIL"] == "unknown"
    assert out["GPU_MEM_MIB"] == "unknown"
    assert pp._gpu_mem_all_zero(out["GPU_MEM_MIB"], pp.GPU_NOCTX_MEM_MAX_MIB) is False
