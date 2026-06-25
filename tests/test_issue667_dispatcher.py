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


# ─────────────────────────────────────────────────────────────────────────────
# (d-round6) parity-probe argparse contract — the subprocess entrypoint's
# --behavior flag must be REGISTERED, not just launched + read.
#
# Round-6 crash class (bug_class ``subprocess_isolation_argparse_contract_mismatch``):
# the r4 CUDA-isolation refactor moved the rsLoRA parity probe into a one-shot
# ``parity-probe`` subprocess. The launch site (line 444) passes ``--behavior <x>``
# and ``main()`` (line 764) reads ``args.behavior``, but the parser only registered
# ``--behaviors`` (plural, for extract/analysis). So the subprocess deterministically
# died: ``AttributeError: 'Namespace' object has no attribute 'behavior'`` → rc=1 →
# the dispatcher's fail-loud HALT. The r4/r5 tests mocked the subprocess, so none
# exercised the actual parity-probe entrypoint's argument parsing.
#
# These tests drive ``main()``'s argparse end-to-end on the ``parity-probe`` phase
# (the exact code path that broke), with the GPU work + credential check mocked so
# they run CPU-only. Had either existed in r4/r5, the contract gap would have been
# caught at code-review time.
# ─────────────────────────────────────────────────────────────────────────────


def test_parity_probe_argparse_contract_end_to_end(tmp_path):
    """``main()`` on the ``parity-probe`` phase parses ``--behavior`` and forwards it
    to ``_numeric_rslora_parity`` (the line-764 read). Pre-fix: ``AttributeError`` on
    ``args.behavior`` (the flag was never registered) → rc=1. Post-fix: rc=0 and the
    behavior threads through.
    """
    result_path = tmp_path / "parity_result.json"
    captured: dict[str, object] = {}

    def fake_numeric(behavior, *, source, seed):
        captured["behavior"] = behavior
        captured["source"] = source
        captured["seed"] = seed
        return {
            "behavior": behavior,
            "source": source,
            "g_self": 1.0,
            "write_ratio": 0.1,
            "gauge": {"r": 32, "lora_alpha": 256, "use_rslora": True},
        }

    argv = [
        "issue667_dispatch.py",
        "parity-probe",
        "--behavior",
        "em",
        "--source",
        "default",
        "--seed",
        "42",
        "--result-out",
        str(result_path),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(disp, "_require_credentials"),
        mock.patch.object(disp, "_numeric_rslora_parity", side_effect=fake_numeric),
    ):
        rc = disp.main()

    assert rc == 0, "parity-probe phase must exit 0 (pre-fix it died rc=1 on args.behavior)"
    # The launch-site / main()-read / parser registration all agree on --behavior.
    assert captured["behavior"] == "em", captured
    assert captured["source"] == "default", captured
    assert captured["seed"] == 42, captured
    # The result JSON the parent reads back was written.
    assert result_path.exists(), "parity-probe must write its result JSON"
    assert json.loads(result_path.read_text())["behavior"] == "em"


def test_behavior_singular_flag_is_recognized_not_unrecognized_arg():
    """``--behavior`` (singular) must be a REGISTERED flag, not an unrecognized arg.

    Distinct failure mode from the test above: pre-fix, argparse would reject
    ``--behavior`` with ``SystemExit`` (``error: unrecognized arguments: --behavior``)
    only if the launch site reached ``parse_args`` — but the r4 launch site DID pass
    it, so the real crash was the downstream ``AttributeError`` at the line-764 read.
    This pins the parser side: with the GPU work mocked, supplying ``--behavior`` must
    NOT trigger a ``SystemExit`` (argparse never errors) and must NOT trigger an
    ``AttributeError`` (``args.behavior`` resolves).
    """
    argv = ["issue667_dispatch.py", "parity-probe", "--behavior", "fact"]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(disp, "_require_credentials"),
        mock.patch.object(
            disp,
            "_numeric_rslora_parity",
            return_value={
                "behavior": "fact",
                "source": "default",
                "g_self": 1.0,
                "write_ratio": 0.1,
                "gauge": {},
            },
        ),
    ):
        # No --result-out: exercises the no-write branch (result is discarded), so the
        # only thing under test is parser acceptance + the args.behavior read.
        rc = disp.main()
    assert rc == 0, "parser must recognize --behavior (singular) on the parity-probe phase"


# ─────────────────────────────────────────────────────────────────────────────
# (e-round7) resume-skip for already-extracted cells.
#
# Round-7 context: the 4th launch ran ~95 min and extracted 32/64 cells (all em +
# all sycophancy) before the 33rd (fact/sp_swe) crashed. A naive relaunch would
# re-extract those 32 on-disk cells. Resume-skip (default ON) drops any cell whose
# .npz tensors already exist under <TENSORS_DIR>/<behavior>/<source>_seed42, so the
# relaunch only runs the un-extracted cells. --no-resume-skip forces a full re-run.
# ─────────────────────────────────────────────────────────────────────────────


def _stub_extract_phase(monkeypatch, tmp_path, *, sources):
    """Common monkeypatch wiring for the phase_extract resume-skip tests: point
    TENSORS_DIR + PROJECT_ROOT at tmp_path, fix the source list, no-op the parity
    gate + the upload, and capture which cells reached the subprocess launcher."""
    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "TENSORS_DIR", "tensors")
    monkeypatch.setattr(disp, "select_sources", lambda behavior, sources_arg: list(sources))
    monkeypatch.setattr(disp, "select_targets", lambda behavior, targets_arg: ["sp_swe"])
    monkeypatch.setattr(disp, "_rslora_parity_probe", lambda *a, **k: None)
    monkeypatch.setattr(disp, "_upload_tensors", lambda: None)

    launched: list[tuple[str, str]] = []

    def fake_parallel(cmds):
        # cmds is the per-wave list; the wave membership is what we assert on. We
        # re-derive (behavior, source) from each cmd's --behavior/--source-cid argv.
        cmds = list(cmds)
        for cmd, _log, _env in cmds:
            argv = list(cmd)
            b = argv[argv.index("--behavior") + 1]
            s = argv[argv.index("--source-cid") + 1]
            launched.append((b, s))
        return [0] * len(cmds)

    monkeypatch.setattr(disp, "_run_parallel_with_log", fake_parallel)
    return launched


def _make_cell_done(tmp_path, behavior, source):
    """Simulate a FULLY completed cell: tensors + the atomic .done sentinel.

    Round-8: completion is signalled by the ``.done`` sentinel written atomically
    AFTER every tensor lands (issue667_extract.write_cell_done_sentinel), NOT by
    the mere presence of a ``.npz`` (which a mid-cell crash leaves partial)."""
    from issue667_extract import CELL_DONE_SENTINEL

    cell_dir = tmp_path / "tensors" / behavior / f"{source}_seed{disp._EXTRACT_SEED}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "sp_swe_L14.npz").write_bytes(b"\x00")
    (cell_dir / CELL_DONE_SENTINEL).write_text(json.dumps({"behavior": behavior}))


def _make_cell_partial(tmp_path, behavior, source):
    """Simulate a PARTIALLY extracted cell: some .npz present but NO .done sentinel
    (the mid-cell-crash state the round-8 BLOCKER fix must NOT treat as done)."""
    cell_dir = tmp_path / "tensors" / behavior / f"{source}_seed{disp._EXTRACT_SEED}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "sp_swe_L14.npz").write_bytes(b"\x00")  # partial: tensor but no sentinel


def test_phase_extract_resume_skips_already_extracted_cell(monkeypatch, tmp_path):
    """A cell whose .npz already exist is NOT relaunched; the un-extracted cell is."""
    launched = _stub_extract_phase(monkeypatch, tmp_path, sources=["default", "sp_swe"])
    _make_cell_done(tmp_path, "fact", "default")  # already on disk -> must be skipped

    disp.phase_extract(
        behaviors=["fact"],
        sources_arg=None,
        targets_arg=None,
        layers=[14],
        primary_layer=14,
        n_gpus=1,
        cpu_only=True,
        max_probes=2,
        max_train_rows=8,
        skip_upload=True,
        dry_run=False,
        skip_parity=True,
        resume_skip=True,
    )

    assert ("fact", "default") not in launched, "completed cell must be resume-skipped"
    assert ("fact", "sp_swe") in launched, "un-extracted cell must still run"
    assert launched == [("fact", "sp_swe")], launched


def test_phase_extract_no_resume_skip_reruns_completed_cell(monkeypatch, tmp_path):
    """``resume_skip=False`` (the --no-resume-skip flag) forces a full re-extract:
    the already-on-disk cell IS relaunched."""
    launched = _stub_extract_phase(monkeypatch, tmp_path, sources=["default", "sp_swe"])
    _make_cell_done(tmp_path, "fact", "default")

    disp.phase_extract(
        behaviors=["fact"],
        sources_arg=None,
        targets_arg=None,
        layers=[14],
        primary_layer=14,
        n_gpus=1,
        cpu_only=True,
        max_probes=2,
        max_train_rows=8,
        skip_upload=True,
        dry_run=False,
        skip_parity=True,
        resume_skip=False,
    )

    assert ("fact", "default") in launched, "--no-resume-skip must re-run the completed cell"
    assert ("fact", "sp_swe") in launched
    assert len(launched) == 2, launched


def test_cell_already_extracted_predicate(monkeypatch, tmp_path):
    """``_cell_already_extracted`` is True ONLY when the atomic .done sentinel exists.

    Round-8 BLOCKER flip: a dir holding ``.npz`` files but NO ``.done`` sentinel is
    a PARTIAL (mid-crash) cell and must NOT count as done — the old ``any(*.npz)``
    contract would silently accept it and skip re-extraction, corrupting the
    headline (resume-skip-partial-cell-silent-skip)."""
    from issue667_extract import CELL_DONE_SENTINEL

    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "TENSORS_DIR", "tensors")

    # No dir -> not done.
    assert not disp._cell_already_extracted("fact", "sp_swe")

    # Empty dir (mkdir'd but nothing written) -> not done.
    cell = tmp_path / "tensors" / "fact" / f"sp_swe_seed{disp._EXTRACT_SEED}"
    cell.mkdir(parents=True, exist_ok=True)
    assert not disp._cell_already_extracted("fact", "sp_swe"), (
        "empty cell dir must NOT count as done"
    )

    # Dir with a .npz but NO sentinel (partial / mid-crash) -> NOT done (the flip).
    (cell / "sp_swe_L14.npz").write_bytes(b"\x00")
    assert not disp._cell_already_extracted("fact", "sp_swe"), (
        "a partial cell (.npz present, no .done sentinel) must NOT count as done"
    )

    # Sentinel present -> done.
    (cell / CELL_DONE_SENTINEL).write_text("{}")
    assert disp._cell_already_extracted("fact", "sp_swe")


def test_phase_extract_does_not_skip_partial_cell(monkeypatch, tmp_path):
    """A PARTIALLY-extracted cell (.npz on disk, no .done sentinel) MUST be relaunched.

    This is the round-8 BLOCKER scenario: a mid-cell crash leaves a partial dir;
    the default-ON resume-skip must re-extract it, never silently accept it."""
    launched = _stub_extract_phase(monkeypatch, tmp_path, sources=["default", "sp_swe"])
    _make_cell_partial(tmp_path, "fact", "default")  # partial -> must be re-run
    _make_cell_done(tmp_path, "fact", "sp_swe")  # fully done -> skipped

    disp.phase_extract(
        behaviors=["fact"],
        sources_arg=None,
        targets_arg=None,
        layers=[14],
        primary_layer=14,
        n_gpus=1,
        cpu_only=True,
        max_probes=2,
        max_train_rows=8,
        skip_upload=True,
        dry_run=False,
        skip_parity=True,
        resume_skip=True,
    )

    assert ("fact", "default") in launched, "partial cell must be re-extracted (no sentinel)"
    assert ("fact", "sp_swe") not in launched, "fully-complete cell (sentinel) is resume-skipped"
    assert launched == [("fact", "default")], launched


# ─────────────────────────────────────────────────────────────────────────────
# (round-8) --backfill-sentinels: write .done for already-complete on-disk cells.
#
# The 32 cells extracted under the round-7 any(*.npz) contract have no .done
# sentinel. The backfill validates each cell's .npz complement and writes the
# atomic sentinel ONLY for complete cells; incomplete cells are reported + left
# unsentineled (they re-extract). Pure local filesystem walk — no GPU/HF.
# ─────────────────────────────────────────────────────────────────────────────


def test_backfill_sentinels_writes_only_for_complete_cells(monkeypatch, tmp_path):
    from issue667_extract import CELL_DONE_SENTINEL

    from explore_persona_space.experiments import i537_contexts

    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "TENSORS_DIR", "tensors")
    # Tiny stubbed target grid: 2 eval cids; the cell's expected complement is
    # {source, *eval_cids} x layers, deduped.
    monkeypatch.setattr(i537_contexts, "eval_cids_for", lambda behavior: ["sp_swe", "fmt_json"])

    layers = [14]
    seed = disp._EXTRACT_SEED

    # Complete cell: source=default -> targets {default, sp_swe, fmt_json} x {14}.
    done_cell = tmp_path / "tensors" / "fact" / f"default_seed{seed}"
    done_cell.mkdir(parents=True, exist_ok=True)
    for tcid in ("default", "sp_swe", "fmt_json"):
        (done_cell / f"{tcid}_L14.npz").write_bytes(b"\x00")

    # Incomplete cell: source=sp_swe missing one expected .npz -> NO sentinel.
    partial_cell = tmp_path / "tensors" / "fact" / f"sp_swe_seed{seed}"
    partial_cell.mkdir(parents=True, exist_ok=True)
    for tcid in ("sp_swe", "default"):  # missing fmt_json_L14.npz
        (partial_cell / f"{tcid}_L14.npz").write_bytes(b"\x00")

    disp.phase_backfill_sentinels(layers=layers)

    assert (done_cell / CELL_DONE_SENTINEL).is_file(), "complete cell must get a sentinel"
    assert not (partial_cell / CELL_DONE_SENTINEL).is_file(), (
        "incomplete cell must NOT get a sentinel (it re-extracts)"
    )
    # After backfill the complete cell resume-skips; the partial one does not.
    assert disp._cell_already_extracted("fact", "default")
    assert not disp._cell_already_extracted("fact", "sp_swe")


def test_backfill_sentinels_is_idempotent(monkeypatch, tmp_path):
    """Re-running backfill on a cell that already has a sentinel is a no-op (no crash)."""
    from issue667_extract import CELL_DONE_SENTINEL

    from explore_persona_space.experiments import i537_contexts

    monkeypatch.setattr(disp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(disp, "TENSORS_DIR", "tensors")
    monkeypatch.setattr(i537_contexts, "eval_cids_for", lambda behavior: ["sp_swe"])

    cell = tmp_path / "tensors" / "fact" / f"default_seed{disp._EXTRACT_SEED}"
    cell.mkdir(parents=True, exist_ok=True)
    for tcid in ("default", "sp_swe"):
        (cell / f"{tcid}_L14.npz").write_bytes(b"\x00")

    disp.phase_backfill_sentinels(layers=[14])
    first = (cell / CELL_DONE_SENTINEL).read_text()
    disp.phase_backfill_sentinels(layers=[14])  # second run: must not crash or rewrite
    assert (cell / CELL_DONE_SENTINEL).read_text() == first


# ─────────────────────────────────────────────────────────────────────────────
# (round-9) run_extraction validates the FULL .npz complement BEFORE the .done
# sentinel. A target whose probes all produce empty responses skips its .npz
# write per layer (_extract_one_target's `if not acc[li][0]: continue`); without
# this gate the unconditional write_cell_done_sentinel would stamp a TRUSTED
# .done over an incomplete cell that the resume-skip then silently treats as done
# (round-8 BLOCKER resume-skip-empty-acc-unconditional-sentinel). The live
# extract path must mirror the --backfill-sentinels complement check: raise loud,
# write NO sentinel, leave the partial .npz so resume-skip re-extracts.
# ─────────────────────────────────────────────────────────────────────────────


def test_assert_full_npz_complement_raises_on_missing_and_passes_when_complete(tmp_path):
    """The complement gate raises iff any expected ``{target}_L{layer}.npz`` is absent."""
    import issue667_extract as ext

    cell = tmp_path / "fact" / "default_seed42"
    cell.mkdir(parents=True, exist_ok=True)
    targets = ["default", "sp_swe", "fmt_json"]
    layers = [14]

    # Missing fmt_json (the empty-acc target skipped its write) -> raise loud.
    for tcid in ("default", "sp_swe"):
        (cell / f"{tcid}_L14.npz").write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match=r"fmt_json_L14\.npz"):
        ext.assert_full_npz_complement(cell, targets=targets, layers=layers)

    # Now complete -> no raise.
    (cell / "fmt_json_L14.npz").write_bytes(b"\x00")
    ext.assert_full_npz_complement(cell, targets=targets, layers=layers)


def test_run_extraction_empty_acc_target_raises_and_writes_no_sentinel(monkeypatch, tmp_path):
    """An empty-response target (no .npz for any layer) makes run_extraction RAISE.

    Round-9 BLOCKER (resume-skip-empty-acc-unconditional-sentinel): a target whose
    probes all return empty SKIPS its per-layer .npz write inside
    _extract_one_target. run_extraction must then NOT stamp the .done sentinel —
    it raises, the partial .npz that DID land stays on disk, and (no .done) the
    dispatcher's resume-skip re-extracts the cell on the next pass."""
    import issue667_extract as ext

    from explore_persona_space.experiments import i537_contexts

    targets = ["default", "sp_swe", "fmt_json"]
    empty_target = "fmt_json"  # this one's probes all return "" -> no .npz written
    layers = [14]

    # ── Stub every heavy callee so run_extraction reaches the complement gate on
    #    the CPU-only path without loading the 7B model / registry / vLLM. ──
    monkeypatch.setattr(ext, "stage_inputs", lambda: (tmp_path / "s.json", tmp_path / "d.json"))
    monkeypatch.setattr(i537_contexts, "load_registry", lambda p: {})
    monkeypatch.setattr(i537_contexts, "load_icl_demos", lambda p: {})
    monkeypatch.setattr(i537_contexts, "eval_cids_for", lambda b: ["sp_swe", "fmt_json"])
    monkeypatch.setattr(ext, "load_eval_probes", lambda b: ["q0", "q1"])
    monkeypatch.setattr(ext, "stage_adapter_local", lambda b, s, seed: tmp_path / "adapter")
    monkeypatch.setattr(
        ext, "assert_adapter_gauge", lambda d, b: {"r": 16, "lora_alpha": 32, "use_rslora": True}
    )
    monkeypatch.setattr(ext, "load_base_and_trained", lambda d, dev, dt: (None, _Stub(), _Stub()))
    monkeypatch.setattr(
        ext, "_context_vector_all_layers", lambda *a, **k: __import__("numpy").zeros((28, 8))
    )
    monkeypatch.setattr(ext, "extract_t_pos_neg", lambda *a, **k: {})
    monkeypatch.setattr(ext, "extract_v0_C_neg", lambda *a, **k: None)
    monkeypatch.setattr(ext, "extract_fact_r_b", lambda *a, **k: None)
    monkeypatch.setattr(
        ext, "build_messages_for", lambda *a, **k: [{"role": "user", "content": "q"}]
    )

    class _FakeTok:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()

    monkeypatch.setattr("transformers.AutoTokenizer", _FakeTok)

    # Stub _extract_one_target: write the real-shaped .npz set for every target
    # EXCEPT the empty one (which writes nothing — exactly the empty-acc skip).
    def fake_extract_one_target(*args, **kwargs):
        # positional signature: (base, trained, tok, registry, demos, cell_dir,
        #   behavior, source_cid, seed, tcid, probes, layers, primary_layer, ...)
        cell_dir = args[5]
        tcid = args[9]
        layers_arg = args[11]
        if tcid == empty_target:
            return (2, 2)  # 2 generations, both empty -> NO .npz written, like the bug
        for li in layers_arg:
            (Path(cell_dir) / f"{tcid}_L{li}.npz").write_bytes(b"\x00")
        return (2, 0)

    monkeypatch.setattr(ext, "_extract_one_target", fake_extract_one_target)

    args = _extract_args(out=tmp_path / "out", targets=targets, layers=layers)

    with pytest.raises(RuntimeError, match=r"fmt_json_L14\.npz"):
        ext.run_extraction(args)

    cell_dir = Path(args.out) / "fact" / f"default_seed{args.seed}"
    # (b) NO .done sentinel was written.
    assert not (cell_dir / ext.CELL_DONE_SENTINEL).is_file(), (
        "an incomplete cell (empty-acc target) must NOT get a .done sentinel"
    )
    # (c) The partial .npz that DID land are still present (not deleted) — the
    #     dispatcher's resume-skip sees no .done and re-extracts.
    assert (cell_dir / "default_L14.npz").is_file()
    assert (cell_dir / "sp_swe_L14.npz").is_file()
    assert not (cell_dir / "fmt_json_L14.npz").is_file()


class _Stub:
    """Minimal stand-in for a loaded HF model (only .config.hidden_size is read)."""

    class _Cfg:
        hidden_size = 3584

    config = _Cfg()


def _extract_args(*, out, targets, layers):
    import argparse

    return argparse.Namespace(
        behavior="fact",
        source_cid="default",
        seed=42,
        targets=",".join(targets),
        layers=layers,
        primary_layer=layers[0],
        out=str(out),
        gpu_id=0,
        cpu_only=True,
        max_probes=2,
        max_new_tokens=32,
        max_train_rows=8,
    )
