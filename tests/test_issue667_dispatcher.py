"""Issue #667 dispatcher regression tests — CUDA-isolation of the rsLoRA parity probe.

Round-4 crash (bug_class ``dispatcher_cuda_init_before_subprocess_fork``): the
dispatcher ran the GPU NUMERIC rsLoRA parity probe IN-PROCESS before the extract
wave. The probe loads the base model + a PeftModel, which initializes CUDA in the
dispatcher PARENT; the per-cell extract subprocesses then fork (vLLM forks its own
EngineCore worker), and a live CUDA context in the parent poisons that fork chain
→ ``RuntimeError: Cannot re-initialize CUDA in forked subprocess``.

Fix: the GPU parity probe runs in a ONE-SHOT SUBPROCESS (the ``parity-probe`` CLI
entrypoint) so the dispatcher parent never touches CUDA. These tests pin:

1. The GPU parity-probe path goes through ``subprocess.run`` (never an in-process
   ``_numeric_rslora_parity``), so ``torch.cuda.is_initialized()`` stays False in
   the parent. (CPU-only VM: the subprocess is mocked — no 7B load — and we assert
   the parent never imported/initialized CUDA on this code path.)
2. A non-zero subprocess rc re-raises (the HALT gate is preserved end-to-end).
3. A zero rc with no result file re-raises (an unverified PASS is a HALT).
4. The CPU-only smoke path stays in-process (gauge config check, no CUDA, no
   subprocess) — unchanged behavior.
"""

# math/scientific notation in docstrings + messages

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue667_dispatch as disp  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# (a) The GPU parity-probe path is CUDA-isolated: never inits CUDA in the parent.
# ─────────────────────────────────────────────────────────────────────────────


def test_gpu_parity_probe_runs_in_subprocess_not_in_parent_process():
    """``_rslora_parity_probe(cpu_only=False)`` dispatches a subprocess; the parent
    NEVER calls the in-process ``_numeric_rslora_parity`` (which would init CUDA).

    Pre-fix this routed straight to ``_numeric_rslora_parity`` in-process → CUDA
    init in the dispatcher parent → the fork-poisoning crash. Post-fix it goes
    through ``_run_with_log`` (subprocess) and the parent's CUDA state is untouched.
    """
    assert not torch.cuda.is_initialized(), "test precondition: CUDA must start uninitialized"

    captured: dict[str, list[str]] = {}

    def fake_run_with_log(cmd, *, log_path, extra_env=None):
        # Record the subprocess argv; simulate a clean PASS by writing the result
        # JSON the parent reads back. NO real model load, NO CUDA touched.
        captured["cmd"] = list(cmd)
        result_path = None
        for i, tok in enumerate(cmd):
            if tok == "--result-out":
                result_path = Path(cmd[i + 1])
        assert result_path is not None, cmd
        result_path.write_text(
            json.dumps(
                {
                    "behavior": "em",
                    "source": "default",
                    "g_self": 1.0,
                    "write_norm": 1.0,
                    "base_norm": 10.0,
                    "write_ratio": 0.1,
                    "gauge": {"r": 32, "lora_alpha": 256, "use_rslora": True},
                    "n_probes": 3,
                }
            )
        )
        return 0

    # Guard: if the parent ever reached the in-process numeric probe, fail loudly.
    def forbidden_numeric(*a, **k):  # pragma: no cover - asserts it is never called
        raise AssertionError(
            "_numeric_rslora_parity was called IN the dispatcher parent — that "
            "re-introduces the CUDA-init-before-fork crash (#667 r4)."
        )

    with (
        mock.patch.object(disp, "_run_with_log", side_effect=fake_run_with_log),
        mock.patch.object(disp, "_numeric_rslora_parity", side_effect=forbidden_numeric),
    ):
        disp._rslora_parity_probe("em", cpu_only=False)

    # The subprocess argv must be THIS module's parity-probe entrypoint.
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable, cmd
    assert "parity-probe" in cmd, cmd
    assert "--behavior" in cmd and "em" in cmd, cmd
    assert "--result-out" in cmd, cmd
    # The dispatcher parent never initialized CUDA on this path.
    assert not torch.cuda.is_initialized(), "parent process must NOT initialize CUDA"


def test_gpu_parity_probe_subprocess_nonzero_rc_halts():
    """A non-zero subprocess rc re-raises — the HALT gate (plan §5g/§7) survives."""

    def fake_run_with_log_fail(cmd, *, log_path, extra_env=None):
        return 2  # probe FAILED its parity assert (or crashed) -> HALT

    with (
        mock.patch.object(disp, "_run_with_log", side_effect=fake_run_with_log_fail),
        mock.patch.object(disp, "_numeric_rslora_parity"),
        pytest.raises(RuntimeError, match=r"parity probe subprocess exited rc=2"),
    ):
        disp._rslora_parity_probe("em", cpu_only=False)


def test_gpu_parity_probe_subprocess_rc0_no_result_halts():
    """rc=0 but no result JSON is a HALT (an unverified PASS must not proceed)."""

    def fake_run_with_log_no_result(cmd, *, log_path, extra_env=None):
        return 0  # exits clean but writes nothing

    with (
        mock.patch.object(disp, "_run_with_log", side_effect=fake_run_with_log_no_result),
        mock.patch.object(disp, "_numeric_rslora_parity"),
        pytest.raises(RuntimeError, match=r"wrote no result"),
    ):
        disp._rslora_parity_probe("em", cpu_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# (b) The CPU-only smoke path stays IN-PROCESS (gauge config check, no subprocess).
# ─────────────────────────────────────────────────────────────────────────────


def test_cpu_only_parity_probe_stays_in_process_no_subprocess():
    """The CPU smoke path asserts the gauge config in-process (no CUDA, no fork)."""
    fake_gauge = {"r": 32, "lora_alpha": 256, "use_rslora": True, "target_modules": []}

    with (
        mock.patch("issue667_extract.stage_adapter_local", return_value=Path("/tmp/fake")),
        mock.patch("issue667_extract.assert_adapter_gauge", return_value=fake_gauge),
        mock.patch.object(
            disp, "_run_with_log", side_effect=AssertionError("no subprocess on CPU")
        ),
        mock.patch.object(
            disp, "_numeric_rslora_parity", side_effect=AssertionError("no GPU probe")
        ),
    ):
        # Must NOT raise (neither the subprocess nor the GPU numeric path fires).
        disp._rslora_parity_probe("em", cpu_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-bound numeric reproduction (skipped on the CPU-only VM).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available")
def test_numeric_parity_probe_smoke_gpu():  # pragma: no cover - GPU-only
    """On a GPU box, the real numeric probe produces the expected gauge fields."""
    result = disp._numeric_rslora_parity("em", source="default", seed=42)
    assert result["g_self"] == pytest.approx(1.0, abs=1e-4)
    assert result["write_ratio"] >= disp.PARITY_MIN_WRITE_RATIO
    assert result["gauge"]["use_rslora"] is True
