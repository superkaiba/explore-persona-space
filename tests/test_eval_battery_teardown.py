"""CPU unit tests for eval_battery.teardown_vllm (crash-fix #1090 r3).

The r2 single-shot orphan probe crashed on containerized pods (RunPod):
nvidia-smi reports HOST-namespace pids while ``os.getpid()`` is the container
pid, so the self-exclusion never matched and the caller's own CUDA context /
the just-killed engine child read as an "orphan" — with nvidia-smi showing
0 MiB moments later. These tests pin the r3 replacement: a bounded drain loop
over a PURE verdict function (``_teardown_drain_verdict``) with a residual
used_memory floor, plus tolerant nvidia-smi parsing
(``_parse_compute_app_rows``), plus body-executing drain-loop integration
tests with the subprocess boundary faked signature-conformantly.

No GPU, no network; ``teardown_vllm``'s real body runs (gc / empty_cache /
psutil child-reap are harmless no-ops on a CPU test process).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import eval_battery as EB

UUID_A = "GPU-aaaaaaaa-1111"
UUID_B = "GPU-bbbbbbbb-2222"


# ---------------------------------------------------------------------------
# _parse_compute_app_rows — tolerant nvidia-smi CSV parsing
# ---------------------------------------------------------------------------


def test_parse_compute_app_rows_basic_and_na():
    text = f"1234, 81050, {UUID_A}\n5678, [N/A], {UUID_B}\n\n[N/A], 12, {UUID_A}\n"
    rows = EB._parse_compute_app_rows(text)
    assert rows == [
        (1234, 81050.0, UUID_A),
        (5678, None, UUID_B),  # [N/A] used_memory -> unknown
        (None, 12.0, UUID_A),  # [N/A] pid -> cannot self-exclude
    ]


def test_parse_compute_app_rows_skips_blank_and_malformed():
    assert EB._parse_compute_app_rows("") == []
    assert EB._parse_compute_app_rows("\n  \n") == []
    assert EB._parse_compute_app_rows("garbage-without-commas\n1, 2\n") == []


# ---------------------------------------------------------------------------
# _teardown_drain_verdict — the pure pass/fail decision
# ---------------------------------------------------------------------------


def test_verdict_no_foreign_pids_passes():
    passed, foreign = EB._teardown_drain_verdict([], my_pid=10, visible_uuids=None, floor_mib=6144)
    assert passed and foreign == []


def test_verdict_self_pid_excluded_on_matching_namespace():
    rows = [(10, 50000.0, UUID_A)]
    passed, foreign = EB._teardown_drain_verdict(
        rows, my_pid=10, visible_uuids=None, floor_mib=6144
    )
    assert passed and foreign == []


def test_verdict_small_residual_below_floor_passes():
    # The container case: our own CUDA context under a host-namespace pid.
    rows = [(262291, 1800.0, UUID_A)]
    passed, foreign = EB._teardown_drain_verdict(
        rows, my_pid=4975, visible_uuids=None, floor_mib=6144
    )
    assert passed
    assert foreign == [(262291, 1800.0)]


def test_verdict_large_residual_fails():
    # A real orphaned vLLM worker holds >= model weights (tens of GiB).
    rows = [(262291, 48000.0, UUID_A)]
    passed, foreign = EB._teardown_drain_verdict(
        rows, my_pid=4975, visible_uuids=None, floor_mib=6144
    )
    assert not passed
    assert foreign == [(262291, 48000.0)]


def test_verdict_sums_multiple_foreign_pids_against_floor():
    rows = [(1, 4000.0, UUID_A), (2, 4000.0, UUID_A)]
    passed, _ = EB._teardown_drain_verdict(rows, my_pid=10, visible_uuids=None, floor_mib=6144)
    assert not passed  # 8000 > 6144 in aggregate
    passed, _ = EB._teardown_drain_verdict(rows, my_pid=10, visible_uuids=None, floor_mib=9000)
    assert passed


def test_verdict_na_memory_counts_as_above_floor():
    rows = [(262291, None, UUID_A)]
    passed, foreign = EB._teardown_drain_verdict(
        rows, my_pid=4975, visible_uuids=None, floor_mib=6144
    )
    assert not passed
    assert foreign == [(262291, None)]


def test_verdict_invisible_uuid_filtered_out():
    # #396 BF9: another shard's worker on a non-visible GPU is not ours.
    rows = [(999, 70000.0, UUID_B)]
    passed, foreign = EB._teardown_drain_verdict(
        rows, my_pid=10, visible_uuids={UUID_A}, floor_mib=6144
    )
    assert passed and foreign == []


# ---------------------------------------------------------------------------
# teardown_vllm — drain-loop integration (real body; subprocess boundary faked)
# ---------------------------------------------------------------------------


def _install_fake_smi(monkeypatch, *, uuids_out: str, compute_polls: list[str]) -> dict:
    """Signature-conformant fake for the nvidia-smi subprocess boundary.

    Mirrors the production call shape (argv list + capture_output/text/check/
    env kwargs); returns the canned uuids output for the gpu query and the
    i-th canned compute-apps output per poll (clamped to the last).
    """
    calls = {"apps": 0}

    def fake_run(cmd, capture_output, text, check, env):
        assert cmd[0] == "nvidia-smi" and cmd[-1] == "--format=csv,noheader,nounits"
        assert capture_output and text and check and isinstance(env, dict)
        query = cmd[1]
        if query.startswith("--query-gpu"):
            return SimpleNamespace(stdout=uuids_out)
        assert query.startswith("--query-compute-apps")
        i = min(calls["apps"], len(compute_polls) - 1)
        calls["apps"] += 1
        return SimpleNamespace(stdout=compute_polls[i])

    monkeypatch.setattr(EB.subprocess, "run", fake_run)
    monkeypatch.setattr(EB.time, "sleep", lambda s: None)  # no real 2 s polls
    return calls


def test_teardown_drains_transient_usage_then_passes(monkeypatch):
    # Poll 1: the just-SIGKILLed engine child still holds memory (the race the
    # r2 single-shot probe crashed on); poll 2: driver released it. PASS.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("EPM_VLLM_TEARDOWN_DRAIN_TIMEOUT_S", raising=False)
    monkeypatch.delenv("EPM_VLLM_TEARDOWN_RESIDUAL_FLOOR_MIB", raising=False)
    calls = _install_fake_smi(
        monkeypatch,
        uuids_out=f"0, {UUID_A}\n",
        compute_polls=[f"262291, 48000, {UUID_A}\n", ""],
    )
    EB.teardown_vllm(object())  # returns cleanly — no RuntimeError
    assert calls["apps"] == 2  # the drain loop actually re-polled


def test_teardown_passes_immediately_on_subfloor_residual(monkeypatch):
    # The container steady-state: our own CUDA context under a host-ns pid.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = _install_fake_smi(
        monkeypatch,
        uuids_out=f"0, {UUID_A}\n",
        compute_polls=[f"262291, 1800, {UUID_A}\n"],
    )
    EB.teardown_vllm(object())
    assert calls["apps"] == 1


def test_teardown_timeout_raises_enriched_runtimeerror(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("EPM_VLLM_TEARDOWN_DRAIN_TIMEOUT_S", "0")
    _install_fake_smi(
        monkeypatch,
        uuids_out=f"0, {UUID_A}\n",
        compute_polls=[f"262291, 48000, {UUID_A}\n"],
    )
    with pytest.raises(RuntimeError, match=r"pid=262291:48000MiB.*refusing to proceed"):
        EB.teardown_vllm(object())


def test_teardown_cvd_filter_ignores_other_shards_workers(monkeypatch):
    # CVD pins us to GPU 0 (UUID_A); a 70 GiB worker on UUID_B is not ours.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("EPM_VLLM_TEARDOWN_DRAIN_TIMEOUT_S", "0")
    calls = _install_fake_smi(
        monkeypatch,
        uuids_out=f"0, {UUID_A}\n1, {UUID_B}\n",
        compute_polls=[f"999, 70000, {UUID_B}\n"],
    )
    EB.teardown_vllm(object())
    assert calls["apps"] == 1


def test_teardown_cpu_host_skips_orphan_check(monkeypatch):
    def fake_run(cmd, capture_output, text, check, env):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(EB.subprocess, "run", fake_run)
    EB.teardown_vllm(object())  # logs + returns; no raise


class _FakeProc:
    def __init__(self, name=None, raises=False):
        self._name = name
        self._raises = raises

    def name(self):
        if self._raises:
            raise RuntimeError("process gone")
        return self._name


def test_is_protected_child_wandb_core_protected():
    assert EB._is_protected_child(_FakeProc("wandb-core")) is True


def test_is_protected_child_python_worker_not_protected():
    assert EB._is_protected_child(_FakeProc("python3")) is False


def test_is_protected_child_unreadable_name_not_protected():
    assert EB._is_protected_child(_FakeProc(raises=True)) is False


def test_teardown_child_sweep_spares_wandb_service(monkeypatch):
    """The kill sweep must exclude wandb-core (killing it breaks every later
    wandb.init in-process — #1090 crash r4) while still reaping vLLM workers."""
    terminated = []

    class _SweepProc:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def terminate(self):
            terminated.append(self._name)

        def kill(self):  # pragma: no cover - wait_procs reports none alive
            terminated.append(f"kill:{self._name}")

    procs = [_SweepProc("wandb-core"), _SweepProc("python3")]

    class _FakeMe:
        def children(self, recursive):
            assert recursive is True
            return procs

    import psutil

    monkeypatch.setattr(psutil, "Process", lambda: _FakeMe())
    monkeypatch.setattr(psutil, "wait_procs", lambda children, timeout: (children, []))
    monkeypatch.setenv("EPM_VLLM_TEARDOWN_DRAIN_TIMEOUT_S", "0")
    _install_fake_smi(monkeypatch, uuids_out=f"0, {UUID_A}\n", compute_polls=[""])
    EB.teardown_vllm(object())
    assert terminated == ["python3"]
