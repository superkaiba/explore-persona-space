"""CPU unit tests for the #2091 crash fix (teardown-drain-check-not-cvd-scoped).

The vLLM teardown drain verdict in ``scripts/issue2091_pod.py`` must be scoped
to the unit's OWN physical GPU: ``nvidia-smi --query-gpu`` does NOT honor
``CUDA_VISIBLE_DEVICES`` (it always enumerates the whole node), so in the
4-way concurrent rung-job fan-out an all-device max verdict is
unsatisfiable-by-construction whenever a sibling job legitimately holds its
own engine — the gotchas.md #1333 class. The pre-fix verdict
(``worst = max over ALL devices``) killed ``hal_nqopen`` (gpu=2) and
``hal_simpleqa`` (gpu=3) at rc=1 with their OWN devices reading 0 MiB.

No GPU, no network, no repo-root artifacts: the single external boundary (the
``nvidia-smi`` subprocess) is faked with a signature-mirroring fake returning
a real ``subprocess.CompletedProcess``. Tests drive the real
``_drain_wait_own_gpu`` body.
"""

from __future__ import annotations

import subprocess

import pytest

import scripts.issue2091_pod as pod

# The EXACT production reading from the 2026-08-06 crash: siblings syc_train
# (gpu 0) and hal_train (gpu 1) legitimately hold live engines; gpus 2 and 3
# (the units that died rc=1) are fully drained.
PRODUCTION_READING = "0, 35579\n1, 19143\n2, 0\n3, 0\n"


def _fake_nvidia_smi(stdout: str):
    """Signature-mirroring fake for the one subprocess.run call in the loop."""

    def fake_run(cmd, capture_output=False, text=False, check=False, env=None):
        assert cmd[0] == "nvidia-smi", cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return fake_run


@pytest.mark.parametrize("gpu_id", ["2", "3", 2, 3])
def test_passes_when_own_device_drained_despite_hot_siblings(monkeypatch, capsys, gpu_id):
    """The exact production reading that falsely killed hal_nqopen/hal_simpleqa.

    Pre-fix this raised (max over ALL devices read 35,579 MiB > 2,048 floor);
    post-fix the own-device row (0 MiB) drives the verdict and it PASSES.
    """
    monkeypatch.setattr(pod.subprocess, "run", _fake_nvidia_smi(PRODUCTION_READING))
    pod._drain_wait_own_gpu(gpu_id, drain_timeout_s=0, floor_mib=2048)  # must NOT raise
    out = capsys.readouterr().out
    assert "[phase=p2_reap] GPU drained:" in out
    assert f"gpu={int(str(gpu_id))}" in out  # the log names WHICH gpu id was verified
    assert "35579" in out  # full per-GPU list retained as context


@pytest.mark.parametrize("gpu_id,own_mib", [("0", 35579), (1, 19143)])
def test_raises_when_own_device_above_floor(monkeypatch, gpu_id, own_mib):
    """A genuine leak ON the unit's own GPU still HALTS (fail-loud preserved)."""
    monkeypatch.setattr(pod.subprocess, "run", _fake_nvidia_smi(PRODUCTION_READING))
    with pytest.raises(RuntimeError) as excinfo:
        pod._drain_wait_own_gpu(gpu_id, drain_timeout_s=0, floor_mib=2048)
    msg = str(excinfo.value)
    assert f"gpu_id={int(str(gpu_id))}" in msg  # names the unit's own device
    assert str(own_mib) in msg  # own-device value present
    assert "(0, 35579)" in msg and "(3, 0)" in msg  # full per-GPU list as context


def test_missing_own_row_fails_loud_never_reads_as_drained(monkeypatch):
    """Rows present but none for our gpu_id (shape change / parse miss) = raise."""
    monkeypatch.setattr(pod.subprocess, "run", _fake_nvidia_smi("0, 100\n1, 200\n"))
    with pytest.raises(RuntimeError, match="missing row"):
        pod._drain_wait_own_gpu(3, drain_timeout_s=0, floor_mib=2048)


def test_empty_parse_keeps_retry_until_deadline_then_raises(monkeypatch):
    """A wholly-empty parse keeps the pre-existing retry tolerance, then raises
    the timeout error (naming the own gpu_id) at the deadline — never a pass."""
    monkeypatch.setattr(pod.subprocess, "run", _fake_nvidia_smi(""))
    with pytest.raises(RuntimeError, match="did not drain below"):
        pod._drain_wait_own_gpu(1, drain_timeout_s=0, floor_mib=2048)


def test_non_integer_gpu_id_fails_loud():
    with pytest.raises(RuntimeError, match="integer physical GPU index"):
        pod._drain_wait_own_gpu("GPU-deadbeef", drain_timeout_s=0, floor_mib=2048)
