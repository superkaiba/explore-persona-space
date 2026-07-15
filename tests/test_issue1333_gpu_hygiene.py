"""#1333 GPU-hygiene regression pins (crash-fix r4 — the #557 co-location class).

Attempt 4's FULL run OOM'd at p3_ladder: the ext_icl unit hit
``torch.OutOfMemoryError`` on GPU 0 with 29.09 GiB of p0/p1-era vLLM
EngineCore residue (host pid 1213661) co-resident with the unit's own engine
(41.25 GiB) and its HF teacher-forced load. These tests pin the two guard
layers added in r4 with REAL bodies (only the nvidia-smi boundary is faked,
signature-conformant per code-style "one production-body test per
seam-stubbed function"):

1. ``_per_gpu_used_mib`` — pure parse/aggregate (incl. the ``[N/A]`` ->
   unknown -> above-floor rule).
2. ``_wait_gpus_free`` — pass-on-idle, drain-then-pass, GPU scoping, and the
   FAIL-LOUD timeout on a live holder (the invariant that converts the
   attempt-4 41-GiB-deep OOM into a fast named failure).
3. ``_unit_gpu_preflight`` — CVD-pinned target selection.
4. ``_reap_completed_unit_group`` — no-op on a dead group; kills an orphaned
   group member left behind by an exited unit leader.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dispatch():
    import issue1333_dispatch as d

    return d


UUID_MAP = "0, GPU-aaaa\n1, GPU-bbbb\n2, GPU-cccc\n3, GPU-dddd\n"


def _smi_factory(apps_sequence: list[str]):
    """Injected nvidia-smi seam: returns each apps text in sequence (last one
    repeats), and the fixed index->uuid map for the map query. Signature
    mirrors ``_smi_query(query) -> str | None``."""
    calls = {"apps": 0}

    def smi(query: str) -> str | None:
        if "compute-apps" in query:
            i = min(calls["apps"], len(apps_sequence) - 1)
            calls["apps"] += 1
            return apps_sequence[i]
        return UUID_MAP

    return smi, calls


def test_per_gpu_used_mib_aggregates_per_index():
    d = _dispatch()
    apps = "1213661, 29786, GPU-aaaa\n1245879, 42240, GPU-aaaa\n555, 1024, GPU-cccc\n"
    usage = d._per_gpu_used_mib(apps, UUID_MAP)
    total0, pids0 = usage["0"]
    assert total0 == pytest.approx(29786 + 42240)
    assert pids0 == [1213661, 1245879]
    assert usage["2"][0] == pytest.approx(1024)
    assert "1" not in usage  # no compute apps -> absent (0 used)


def test_per_gpu_used_mib_na_used_memory_is_unknown():
    d = _dispatch()
    apps = "777, [N/A], GPU-bbbb\n"
    usage = d._per_gpu_used_mib(apps, UUID_MAP)
    assert usage["1"][0] is None  # unknown -> callers treat as above-floor


def test_wait_gpus_free_passes_on_idle_gpus():
    d = _dispatch()
    smi, calls = _smi_factory([""])
    d._wait_gpus_free(["0", "1"], label="t", smi=smi, sleep=lambda s: None)
    assert calls["apps"] == 1  # no polling needed


def test_wait_gpus_free_times_out_fail_loud_on_live_holder():
    """The attempt-4 shape: a live 29-GiB holder on GPU 0 never drains ->
    RuntimeError naming the pid + the host-namespace caveat, never an OOM."""
    d = _dispatch()
    smi, _ = _smi_factory(["1213661, 29786, GPU-aaaa\n"])
    with pytest.raises(RuntimeError, match="HOST-namespace") as exc:
        d._wait_gpus_free(
            ["0"], label="unit-preflight[ladder]", timeout_s=0.0, smi=smi, sleep=lambda s: None
        )
    assert "1213661" in str(exc.value)
    assert "unit-preflight[ladder]" in str(exc.value)


def test_wait_gpus_free_drains_then_passes():
    d = _dispatch()
    busy = "42, 30000, GPU-aaaa\n"
    smi, calls = _smi_factory([busy, busy, ""])
    d._wait_gpus_free(["0"], label="t", timeout_s=60.0, smi=smi, sleep=lambda s: None)
    assert calls["apps"] == 3  # two busy polls, then drained


def test_wait_gpus_free_scopes_to_target_gpu():
    d = _dispatch()
    smi, _ = _smi_factory(["42, 30000, GPU-aaaa\n"])  # gpu 0 busy
    # target gpu 2 only -> passes despite gpu 0's holder
    d._wait_gpus_free(["2"], label="t", timeout_s=0.0, smi=smi, sleep=lambda s: None)


def test_wait_gpus_free_below_floor_residue_passes():
    """The main dispatcher's own lingering CUDA context (host-ns pid, <1 GiB)
    must not block units — the floor tolerates it."""
    d = _dispatch()
    smi, _ = _smi_factory(["999, 600, GPU-aaaa\n"])
    d._wait_gpus_free(
        ["0"], label="t", floor_mib=2048, timeout_s=0.0, smi=smi, sleep=lambda s: None
    )


def test_wait_gpus_free_na_counts_above_floor():
    d = _dispatch()
    smi, _ = _smi_factory(["42, [N/A], GPU-aaaa\n"])
    with pytest.raises(RuntimeError):
        d._wait_gpus_free(["0"], label="t", timeout_s=0.0, smi=smi, sleep=lambda s: None)


def test_wait_gpus_free_noop_when_smi_unavailable():
    d = _dispatch()
    d._wait_gpus_free(["0"], label="t", smi=lambda q: None, sleep=lambda s: None)


def test_unit_gpu_preflight_targets_cvd_pin(monkeypatch):
    """The preflight probes the CVD-pinned physical GPU, not --gpu-id, when
    the launcher env pin is present (the fanout contract)."""
    d = _dispatch()
    seen: dict = {}

    def fake_wait(targets, *, label, floor_mib, timeout_s):
        seen["targets"] = list(targets)
        seen["label"] = label

    monkeypatch.setattr(d, "_wait_gpus_free", fake_wait)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    d._unit_gpu_preflight("ladder", "0")
    assert seen["targets"] == ["2"]
    assert seen["label"] == "unit-preflight[ladder]"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    d._unit_gpu_preflight("ladder", "3")
    assert seen["targets"] == ["3"]  # --gpu-id fallback when no CVD pin


def test_reap_completed_unit_group_noop_on_dead_group():
    d = _dispatch()
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait(timeout=10)
    t0 = time.monotonic()
    d._reap_completed_unit_group(proc, ["--full", "--unit", "ladder", "x"], grace_s=5.0)
    assert time.monotonic() - t0 < 2.0  # healthy case: instant no-op, no grace sleep


def test_reap_completed_unit_group_kills_orphaned_member():
    """An EngineCore-style orphan (leader exited, child kept the pgid) must be
    reaped by the killpg sweep."""
    d = _dispatch()
    proc = subprocess.Popen(
        # orphan's stdout MUST detach from the pipe or communicate() blocks on
        # the inherited write end until the orphan dies
        ["bash", "-c", "sleep 300 >/dev/null 2>&1 & echo $!; exit 0"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    out, _ = proc.communicate(timeout=10)
    orphan_pid = int(out.strip())
    assert proc.returncode == 0
    # orphan is alive and holds the unit's pgid
    assert os.getpgid(orphan_pid) == proc.pid
    d._reap_completed_unit_group(proc, ["--full", "--unit", "ladder", "x"], grace_s=0.2)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(orphan_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(orphan_pid, signal.SIGKILL)
        pytest.fail("orphaned group member survived the killpg sweep")
