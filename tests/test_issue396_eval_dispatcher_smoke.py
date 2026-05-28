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


def test_default_24_sources_dry_run_max_parallel_4_emits_6_chunks(caplog):
    """With ``max_parallel=4``, dry-run on INHERITED_SOURCES_24 emits 6 chunks of 4 sources.

    This preserves the original "wave-mode" shape the orchestrator
    inspects when sub-source parallelism is explicitly opted into.
    The default after the 2026-05-27 fix is ``max_parallel=1`` — that
    case is covered by ``test_default_max_parallel_1_runs_24_sequential_chunks``.
    """
    disp = _import_dispatcher()

    sources = list(disp.INHERITED_SOURCES_24)
    assert len(sources) == 24, f"INHERITED_SOURCES_24 should have 24 entries; got {len(sources)}"

    with caplog.at_level(logging.INFO, logger="launch_issue396_eval"):
        results = disp.wave_loop(
            sources,
            n_gpus=4,
            seed=42,
            dry_run=True,
            max_parallel=4,
        )

    # Dry-run completes without claiming any source as done/failed — every
    # entry stays out of the results dict (the function only assigns
    # ``skipped`` for sources whose eval-JSON already exists). On a fresh
    # checkout there are no eval-JSONs, so the dict is empty.
    assert results == {}, (
        f"dry-run should not classify any source as done/failed/skipped; got {results!r}"
    )

    # 6 chunks should be announced.
    chunk_log_lines = [r.message for r in caplog.records if "=== Chunk" in r.message]
    assert len(chunk_log_lines) == 6, (
        f"expected 6 chunk announcements for 24 sources / 4 gpus / max_parallel=4; "
        f"got {len(chunk_log_lines)}: {chunk_log_lines}"
    )
    # Each announcement names 4 sources.
    for line in chunk_log_lines:
        # The format is "=== Chunk I / 6 (4 source(s) on GPU(s) [0, 1, 2, 3]): a, b, c, d ==="
        assert "(4 source(s)" in line, (
            f"chunk announcement should declare 4 sources per chunk; got: {line!r}"
        )

    # Each per-source line of the form "  [<source>] -> GPU N, log=..." should
    # appear 24 times total (one per source across all 6 chunks).
    source_dispatch_lines = [
        r.message for r in caplog.records if "-> GPU" in r.message and "log=" in r.message
    ]
    assert len(source_dispatch_lines) == 24, (
        f"expected 24 per-source dispatch log lines; got {len(source_dispatch_lines)}"
    )


def test_default_max_parallel_1_runs_24_sequential_chunks(caplog):
    """The 2026-05-27 fix flipped the default to ``max_parallel=1``: 24 chunks of 1.

    Each chunk runs one source on a single GPU; the GPU index cycles
    through ``0, 1, 2, 3, 0, 1, 2, 3, ...`` round-robin across the
    24 sources. This eliminates the inter-wave HF-cache / vLLM state
    coupling that caused 4 of 4 Wave-2 subprocesses to die with
    HFValidationError on the prior round.
    """
    disp = _import_dispatcher()

    sources = list(disp.INHERITED_SOURCES_24)

    with caplog.at_level(logging.INFO, logger="launch_issue396_eval"):
        # default max_parallel=1 — exercise the new sequential path.
        disp.wave_loop(sources, n_gpus=4, seed=42, dry_run=True)

    chunk_log_lines = [r.message for r in caplog.records if "=== Chunk" in r.message]
    assert len(chunk_log_lines) == 24, (
        f"expected 24 sequential chunks (one source each) on max_parallel=1; "
        f"got {len(chunk_log_lines)}"
    )
    for line in chunk_log_lines:
        assert "(1 source(s)" in line, f"sequential chunk should declare 1 source; got: {line!r}"

    # GPUs cycle 0,1,2,3,0,1,2,3,... so chunk i lands on GPU (i % 4).
    # Verify by parsing the per-source dispatch lines.
    source_dispatch_lines = [
        r.message for r in caplog.records if "-> GPU" in r.message and "log=" in r.message
    ]
    assert len(source_dispatch_lines) == 24, (
        f"expected 24 per-source dispatch lines; got {len(source_dispatch_lines)}"
    )
    # Each dispatch line is "  [<source>] -> GPU N, log=..."; pull the N.
    gpus_in_order: list[int] = []
    for line in source_dispatch_lines:
        # Find "GPU N" token.
        token = line.split("-> GPU")[1].split(",")[0].strip()
        gpus_in_order.append(int(token))
    expected = [i % 4 for i in range(24)]
    assert gpus_in_order == expected, (
        f"sequential GPU assignment must cycle 0,1,2,3,...; got {gpus_in_order}"
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


# ── Bug 3 (#396 2026-05-27): retry-with-backoff on hf_hub_download transients ──


def test_download_merged_checkpoint_retries_then_succeeds(tmp_path, monkeypatch, caplog):
    """``hf_hub_download`` transient failures must retry up to 3x with backoff.

    The first two attempts raise ``HfHubHTTPError`` (transient 5xx); the
    third attempt succeeds. The dispatcher must NOT propagate the early
    failures up — it logs a warning, sleeps (mocked to no-op), and
    retries. Silent death on a transient network blip mid-download was
    the most-likely cause of the launcher's silent Wave-3 demise on
    task #396 (2026-05-27); this regression guard verifies the
    hardening landed.
    """
    disp = _import_dispatcher()

    monkeypatch.setattr(disp, "SNAPSHOT_ROOT", tmp_path)

    # Patch ``list_repo_files`` so we don't hit HF for the file listing.
    fake_subfolder = disp._hf_subfolder("software_engineer", 42)
    fake_files = [f"{fake_subfolder}/config.json"]
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **k: fake_files,
    )

    # Patch ``time.sleep`` (imported as ``_time.sleep`` inside the
    # function) so the test does not actually wait 30 + 60 = 90 seconds.
    monkeypatch.setattr("time.sleep", lambda _s: None)

    # Build a mock ``hf_hub_download`` that fails the first 2 calls
    # then writes a real config.json on the 3rd call.
    from huggingface_hub.errors import HfHubHTTPError

    call_count = {"n": 0}

    def fake_download(repo_id, filename, local_dir):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise HfHubHTTPError(f"simulated 503 attempt {call_count['n']}")
        # 3rd attempt: actually write config.json into the landed subdir.
        landed = Path(local_dir) / filename
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_text("{}")
        return str(landed)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    with caplog.at_level(logging.WARNING, logger="launch_issue396_eval"):
        snap_dir = disp.download_merged_checkpoint("software_engineer", seed=42)

    assert call_count["n"] == 3, (
        f"expected 3 calls (2 transient + 1 success); got {call_count['n']}"
    )
    assert snap_dir.exists()
    assert (snap_dir / "config.json").exists()

    # The two transient-failure attempts must each log a warning.
    retry_warnings = [
        r.message for r in caplog.records if "retrying in" in r.message and "attempt" in r.message
    ]
    assert len(retry_warnings) == 2, (
        f"expected 2 retry-warning log lines (one per failed attempt); got {len(retry_warnings)}"
    )


def test_download_merged_checkpoint_exhausts_retries(tmp_path, monkeypatch):
    """After 3 failed attempts, the loop raises ``RuntimeError`` (not silent death)."""
    disp = _import_dispatcher()

    monkeypatch.setattr(disp, "SNAPSHOT_ROOT", tmp_path)

    fake_subfolder = disp._hf_subfolder("software_engineer", 42)
    fake_files = [f"{fake_subfolder}/config.json"]
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **k: fake_files,
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)

    from huggingface_hub.errors import HfHubHTTPError

    def always_fail(repo_id, filename, local_dir):
        raise HfHubHTTPError("simulated permanent 500")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", always_fail)

    with pytest.raises(RuntimeError) as excinfo:
        disp.download_merged_checkpoint("software_engineer", seed=42)
    assert "exhausted 3 retries" in str(excinfo.value), (
        "RuntimeError message must call out the retry exhaustion explicitly"
    )


# ── Bug 3 (#396 2026-05-27): top-level exception handler in wave_loop ────────


def test_wave_loop_logs_and_reraises_unhandled_exception(tmp_path, monkeypatch, caplog):
    """A surprise exception in the per-source loop must hit ``logger.exception`` then re-raise.

    Without this guard the launcher dies silently and the orchestrator
    sees only "no process running" — the failure mode that left a
    Wave-3 partial snapshot on task #396 without any traceback in the
    launcher log. The fix wraps the per-source loop in a top-level
    ``try / except`` and surfaces the traceback before re-raising.
    """
    disp = _import_dispatcher()

    # Force every source to look "not done" by pointing EVAL_RESULTS_DIR
    # at an empty tmp_path.
    monkeypatch.setattr(disp, "EVAL_RESULTS_DIR", tmp_path)

    # Patch ``download_merged_checkpoint`` to raise a synthetic surprise
    # error on the first call — emulates "something I didn't think of
    # propagated out of the inner loop".
    def fake_dl(source, seed):
        raise RuntimeError("synthetic surprise inside per-source loop")

    monkeypatch.setattr(disp, "download_merged_checkpoint", fake_dl)

    sources = list(disp.INHERITED_SOURCES_24)[:2]  # 2 sources is enough

    with (
        caplog.at_level(logging.ERROR, logger="launch_issue396_eval"),
        pytest.raises(RuntimeError) as excinfo,
    ):
        disp.wave_loop(sources, n_gpus=4, seed=42, dry_run=False, max_parallel=1)

    assert "synthetic surprise" in str(excinfo.value), (
        "wave_loop must re-raise the inner exception, not swallow it"
    )

    # ``logger.exception`` emits an ERROR-level record with the traceback
    # attached. Verify the top-level guard fired.
    guard_records = [
        r for r in caplog.records if "unhandled exception in per-source loop" in r.message
    ]
    assert len(guard_records) == 1, (
        f"expected exactly one top-level guard log; got {len(guard_records)}"
    )
    # ``logger.exception`` sets ``exc_info`` on the record.
    assert guard_records[0].exc_info is not None, (
        "top-level guard must use logger.exception (not logger.error) so the "
        "traceback lands in the launcher log"
    )
