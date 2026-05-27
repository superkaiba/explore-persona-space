"""CPU smoke test for the Phase-C eval dispatcher (scripts/launch_issue396_eval.py).

Mirrors ``tests/test_issue396_launch_marker_assertion.py`` in shape — load
the dispatcher module without executing ``main()``, then assert on the
wave-plan + per-source-command shape that the orchestrator depends on.

These tests do NOT touch HF, GPUs, or the network. ``_validate_sources``
is the only ``SystemExit`` path exercised. The dry-run integration is
deliberately not driven through ``main()`` (which would call ``logging.
basicConfig``, mutate global state, and require dry-running through the
HF download); instead the tests exercise ``wave_loop(..., dry_run=True)``
directly, which prints the plan via the module logger and returns
the empty-skip results dict.

Plan v2.3 §3 (Phase B -> Phase C diagram) + §4.5 spell out the per-source
contract this dispatcher fans out across waves.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_dispatcher():
    """Load scripts/launch_issue396_eval.py as a module without running main()."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "launch_issue396_eval", SCRIPTS_DIR / "launch_issue396_eval.py"
    )
    assert spec is not None, "could not build module spec for launch_issue396_eval.py"
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── Happy path: default 24-source wave plan + per-source command shape ───────


def test_default_24_sources_dry_run_emits_6_waves(caplog):
    """Dry-run against the default INHERITED_SOURCES_24 emits 6 waves of 4 sources.

    This is the canonical end-to-end shape the orchestrator expects:
    24 sources / 4 GPUs = 6 waves. The dry-run path skips both the HF
    download and the ``subprocess.Popen`` step, so this exercises the
    plan-construction code path without any I/O.
    """
    disp = _import_dispatcher()

    sources = list(disp.INHERITED_SOURCES_24)
    assert len(sources) == 24, f"INHERITED_SOURCES_24 should have 24 entries; got {len(sources)}"

    with caplog.at_level(logging.INFO, logger="launch_issue396_eval"):
        results = disp.wave_loop(sources, n_gpus=4, seed=42, dry_run=True)

    # Dry-run completes without claiming any source as done/failed — every
    # entry stays out of the results dict (the function only assigns
    # ``skipped`` for sources whose eval-JSON already exists). On a fresh
    # checkout there are no eval-JSONs, so the dict is empty.
    assert results == {}, (
        f"dry-run should not classify any source as done/failed/skipped; got {results!r}"
    )

    # 6 waves should be announced.
    wave_log_lines = [r.message for r in caplog.records if "=== Wave" in r.message]
    assert len(wave_log_lines) == 6, (
        f"expected 6 wave announcements for 24 sources / 4 gpus; got {len(wave_log_lines)}: "
        f"{wave_log_lines}"
    )
    # Each announcement names 4 sources.
    for line in wave_log_lines:
        # The format is "=== Wave I / 6 (4 sources): a, b, c, d ==="
        assert "(4 sources)" in line, (
            f"wave announcement should declare 4 sources per wave; got: {line!r}"
        )

    # Each per-source line of the form "  [<source>] -> GPU N, log=..." should
    # appear 24 times total (one per source across all 6 waves).
    source_dispatch_lines = [
        r.message for r in caplog.records if "-> GPU" in r.message and "log=" in r.message
    ]
    assert len(source_dispatch_lines) == 24, (
        f"expected 24 per-source dispatch log lines; got {len(source_dispatch_lines)}"
    )


def test_per_source_command_shape_matches_documented_contract():
    """The bash command threaded to each subprocess must match the documented contract.

    Contract (from the brief + plan v2.3 §4.5):

        CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED={seed}
        uv run python scripts/eval_issue396_logprob.py
            --source {source}
            --merged-model-path {snap_dir}
            --seed {seed}

    Sample one source + check every required token is present in the
    exact place the orchestrator (and experimenter) will inspect.
    """
    disp = _import_dispatcher()

    sample_source = disp.INHERITED_SOURCES_24[0]
    snap_dir = disp._snapshot_dir(sample_source, seed=42)
    cmd = disp.build_cmd(sample_source, gpu=2, seed=42, merged_model_path=snap_dir)

    # CVD prefix with the actual GPU index, NOT a hardcoded 0.
    assert cmd.startswith("CUDA_VISIBLE_DEVICES=2 "), (
        f"command must start with the documented CVD-mask + flag prefix; got: {cmd!r}"
    )
    assert "PYTHONUNBUFFERED=1" in cmd, (
        "missing PYTHONUNBUFFERED=1 — subprocess stdout would not stream to log"
    )
    assert "PYTHONHASHSEED=42" in cmd, (
        f"missing PYTHONHASHSEED={42}; eval script's prompt-build order is HASHSEED-sensitive"
    )
    assert "uv run python scripts/eval_issue396_logprob.py" in cmd, (
        "subprocess must invoke the per-source eval script via uv run; "
        "running bare ``python`` would miss the project venv (see "
        "project memory feedback_uv_run_python.md)."
    )
    assert f"--source {sample_source}" in cmd, "missing --source flag"
    assert f"--merged-model-path {snap_dir}" in cmd, "missing --merged-model-path flag"
    assert "--seed 42" in cmd, "missing --seed flag"


# ── Edge case 1: unknown source raises BEFORE any download / subprocess ──────


def test_unknown_source_blocks_loudly():
    """``--sources`` containing a name not in INHERITED_SOURCES_24 must hard-fail.

    Phase B in this round trained ONLY the 24 INHERITED sources; a Phase C
    eval against a source whose merged checkpoint was never uploaded would
    spin up a wave, hit the HF download for a non-existent subfolder, and
    crash mid-wave on the pod. We refuse loudly at CLI parse-time.
    """
    disp = _import_dispatcher()

    with pytest.raises(SystemExit) as excinfo:
        disp._validate_sources(["accountant_typo_that_does_not_exist"])
    msg = str(excinfo.value)
    assert "INHERITED_SOURCES_24" in msg, (
        "error must name the canonical source list so the user knows where to look"
    )
    assert "accountant_typo_that_does_not_exist" in msg, (
        "error must echo the offending source name for grep-ability"
    )


# ── Edge case 2: resume-safe ``is_done`` gate skips a completed source ───────


def test_is_done_skips_completed_source(tmp_path, monkeypatch):
    """A pre-existing eval-JSON with n_cells == 960 must be classified as done.

    This is the resume-safe re-invocation invariant — if a wave finishes
    and the orchestrator re-runs the dispatcher for any reason, every
    already-done source should drop out of the pending list immediately.
    The dispatcher's gate is bit-for-bit the same shape as the Phase-B
    launcher's gate; both phases write to the SAME artifact path.
    """
    disp = _import_dispatcher()

    fake_eval_dir = tmp_path / "eval_results" / "issue_396"
    fake_eval_dir.mkdir(parents=True)
    monkeypatch.setattr(disp, "EVAL_RESULTS_DIR", fake_eval_dir)

    sample_source = "accountant"

    # No file yet -> not done.
    assert disp.is_done(sample_source, seed=42) is False

    # Write a complete eval-JSON (n_cells == 960) -> done.
    import json

    (fake_eval_dir / "logprob_accountant_seed42.json").write_text(
        json.dumps({"source": "accountant", "n_cells": 960, "cells": []})
    )
    assert disp.is_done(sample_source, seed=42) is True

    # Write an incomplete eval-JSON (n_cells < 960) -> NOT done; needs re-run.
    (fake_eval_dir / "logprob_accountant_seed42.json").write_text(
        json.dumps({"source": "accountant", "n_cells": 200, "cells": []})
    )
    assert disp.is_done(sample_source, seed=42) is False
