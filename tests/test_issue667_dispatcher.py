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
import os
import re
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


# ─────────────────────────────────────────────────────────────────────────────
# (c-round5) vLLM EngineCore fork() poisoning guard — VLLM_WORKER_MULTIPROC_METHOD.
#
# Round-5 crash class (bug_class ``vllm_fork_enginecore_silent_death_no_spawn_guard``,
# gotchas.md § entry 26): ``issue667_extract.py`` constructs ``vllm.LLM()`` inside
# ``vllm_generate_R`` AFTER ``main()`` already called ``AutoTokenizer.from_pretrained``.
# Under vLLM V1's default ``fork`` worker method, that pre-LLM() transformers touch
# poisons the EngineCore fork → the worker logs a clean init then dies silently 1-4s
# later (parent surfaces ``Engine core proc ... died unexpectedly`` + a downstream
# ``ZeroDivisionError``). Fix per gotcha #26: set
# ``os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")`` at the extractor
# module top BEFORE any ``import vllm``; ALSO inject it into the per-cell extract
# subprocess env (belt-and-suspenders against a future import-reorder).
# ─────────────────────────────────────────────────────────────────────────────

_VLLM_SPAWN_LINE = 'os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")'
_VLLM_IMPORT_RE = re.compile(r"^\s*(?:import\s+vllm|from\s+vllm\b)", re.MULTILINE)
_TOKENIZER_RE = re.compile(r"AutoTokenizer\.from_pretrained|import\s+AutoTokenizer")


def test_extract_module_sets_vllm_spawn_at_runtime():
    """(a) Importing ``issue667_extract`` leaves ``VLLM_WORKER_MULTIPROC_METHOD``
    pinned to ``spawn`` in ``os.environ`` (the module-top ``setdefault`` ran).

    Static grep below corroborates the line is present + correctly placed; this
    asserts the runtime effect actually took. ``setdefault`` honors a pre-set env
    var, so we only require the value to be ``spawn`` when nothing else pinned it
    (the production contract is "spawn unless the launcher already chose spawn").
    """
    import issue667_extract  # noqa: F401  (import side effect is the thing under test)

    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn", (
        "issue667_extract module-top setdefault did not pin spawn — the vLLM "
        "EngineCore fork (gotcha #26) is unguarded."
    )


def test_extract_cmd_per_cell_env_pins_vllm_spawn():
    """(b) The dispatcher's per-cell extract subprocess ``env`` carries
    ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` (belt-and-suspenders, #667 r5)."""
    _cmd, _log, env = disp._extract_cmd(
        "em",
        "default",
        ["sp_swe"],
        [14],
        14,
        gpu_id=0,
        max_probes=1,
        max_train_rows=1,
        cpu_only=False,
    )
    assert env.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn", env
    # The CVD pin (round-4 #545 fix) stays alongside it.
    assert env.get("CUDA_VISIBLE_DEVICES") == "0", env


def test_extract_cmd_env_threads_to_subprocess_run():
    """(b cont.) The per-cell env reaches ``subprocess.run``'s ``env=`` kwarg
    unmodified (covers the ``_run_with_log`` path that production uses)."""
    cmd, log_path, extra_env = disp._extract_cmd(
        "em",
        "default",
        None,
        [14],
        14,
        gpu_id=1,
        max_probes=None,
        max_train_rows=None,
        cpu_only=False,
    )
    captured: dict[str, dict] = {}

    class _FakeProc:
        returncode = 0

    def fake_subprocess_run(_argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    with mock.patch.object(disp.subprocess, "run", side_effect=fake_subprocess_run):
        rc = disp._run_with_log(cmd, log_path=log_path, extra_env=extra_env)
    assert rc == 0
    assert captured["env"].get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn", captured["env"]


def test_issue6_vllm_scripts_set_spawn_guard_above_vllm_import():
    """(c) MECHANIZABLE regression: any ``scripts/issue6*_extract*.py`` /
    ``scripts/issue6*_dispatch*.py`` that imports vllm AND touches
    ``AutoTokenizer`` MUST set ``VLLM_WORKER_MULTIPROC_METHOD`` at module top,
    ABOVE the first vllm import (gotcha #26).

    Scoped to THIS issue's scripts (issue667 extract/dispatch) to avoid scope
    creep — a `issue6*` glob would pull in unrelated issues' scripts (e.g.
    issue650's, which fits the same hazard pattern and is flagged as a separate
    follow-up, NOT fixed here per #667's single-variable constraint). A deferred
    ``import vllm`` (inside a function) still counts: the env var must be set
    before module IMPORT of the script, which the module-top line guarantees.
    """
    scripts_dir = PROJECT_ROOT / "scripts"
    candidates = sorted(scripts_dir.glob("issue667_extract*.py")) + sorted(
        scripts_dir.glob("issue667_dispatch*.py")
    )
    assert candidates, "scoped glob matched no scripts — test wiring is wrong"

    offenders: list[str] = []
    for path in candidates:
        src = path.read_text()
        imports_vllm = _VLLM_IMPORT_RE.search(src) is not None
        touches_tokenizer = _TOKENIZER_RE.search(src) is not None
        if not (imports_vllm and touches_tokenizer):
            continue  # not in the fork-hazard class
        spawn_idx = src.find(_VLLM_SPAWN_LINE)
        vllm_match = _VLLM_IMPORT_RE.search(src)
        if spawn_idx < 0:
            offenders.append(f"{path.name}: missing {_VLLM_SPAWN_LINE!r}")
        elif vllm_match is not None and spawn_idx > vllm_match.start():
            offenders.append(
                f"{path.name}: spawn guard at char {spawn_idx} is AFTER the first "
                f"vllm import at char {vllm_match.start()} (must precede it)"
            )
    assert not offenders, (
        "vLLM fork-hazard scripts missing/misplacing the spawn guard (gotcha #26):\n"
        + "\n".join(offenders)
    )
