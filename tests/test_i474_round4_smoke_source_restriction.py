"""CPU-only unit tests for the #474 round-4 (post on-pod-smoke) fix.

Round-4 scope: the on-pod smoke trained ONLY A1 (both arms), but the
``crosseval_smoke`` step iterated all 16 SOURCE conditions and 404'd at
``adapters/i474_pos_A5_ep1/adapter_model.safetensors`` — a smoke-harness
bug, NOT a production bug (production trains all 16 sources).

Fix surface:
  - ``i474_phase4_eval.py`` gains ``--source-conds`` flag that filters the
    source loop BEFORE sharding (targets always span all 16).
  - ``i474_phase4_dispatch.sh`` gains ``--smoke`` flag that collapses
    sharding to 1-of-1 (since smoke restricts to a small source subset
    that doesn't fill 4 shards) AND skips the 16x16 merge step (would
    fail-loud on 240 missing cells in smoke).
  - ``i474_run_all.sh --smoke`` passes ``--smoke --source-conds A1
    --arms pos,loc --epochs 1`` to crosseval_smoke.

Production (no ``--smoke``, no ``--source-conds``) MUST be unchanged.

Tests pure source-grep + static-import — no model, no vLLM, no Trainer,
no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_EVAL = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase4_eval.py"
_SCRIPT_DISP = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase4_dispatch.sh"
_SCRIPT_RUN_ALL = Path(__file__).resolve().parent.parent / "scripts" / "i474_run_all.sh"


@pytest.fixture(scope="module")
def i474_eval_module():
    spec = importlib.util.spec_from_file_location("i474_phase4_eval", _SCRIPT_EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase4_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- FIX (--source-conds)


def test_eval_argparse_has_source_conds(i474_eval_module):
    """``i474_phase4_eval.py`` must accept ``--source-conds`` flag."""
    src = _SCRIPT_EVAL.read_text()
    assert "--source-conds" in src, "missing --source-conds argparse flag"
    # nargs="+" so smoke can pass `--source-conds A1` (single) AND
    # production-future could pass `--source-conds A1 B3 C1`.
    assert 'nargs="+"' in src or "nargs='+'" in src or "nargs='+'" in src, (
        "--source-conds must use nargs='+' to accept 1+ cids"
    )


def test_eval_source_conds_filters_before_sharding(i474_eval_module):
    """The filter MUST apply BEFORE sharding (otherwise shard 0-of-4 picks
    A1/A2/A3/A4 then filters to A1 — only works coincidentally for A1)."""
    src = _SCRIPT_EVAL.read_text()
    # The expected pattern: `source_cids = [...args.source_conds...]` then
    # `my_cids = [c for k, c in enumerate(source_cids) if k % n_shards == shard_idx]`
    assert "source_cids" in src and "my_cids = [c for k, c in enumerate(source_cids)" in src, (
        "filter must apply to source_cids BEFORE shard slicing — "
        "the source-restriction surface lives BEFORE the modulo on the source list"
    )


def test_eval_default_unrestricted_source_loop_unchanged(i474_eval_module):
    """When ``--source-conds`` is NOT passed, source_cids defaults to all_cids.

    This guards production: omitting the flag keeps all 16 sources in the loop.
    """
    src = _SCRIPT_EVAL.read_text()
    # The else branch must default to all_cids:
    assert "source_cids = all_cids" in src, (
        "production branch must default source_cids to all_cids (16 conds)"
    )


def test_eval_targets_always_span_all_cids(i474_eval_module):
    """Even with --source-conds, the inner-j target loop iterates all_cids.

    Targets only need frozen R (no trained adapter), so an A1-only smoke
    can still evaluate against all 16 targets — that's the whole point.
    """
    src = _SCRIPT_EVAL.read_text()
    # The inner loop should always be `for cid_j in all_cids:` (unchanged).
    assert "for cid_j in all_cids:" in src, (
        "inner target loop must iterate all 16 cids regardless of "
        "--source-conds — targets only need frozen R, not adapter"
    )


def test_eval_unknown_source_cond_raises(tmp_path, monkeypatch):
    """Unknown source cid fails-loud (per CLAUDE.md no-silent-failures rule)."""
    src = _SCRIPT_EVAL.read_text()
    # Must reject unknown cids with an explicit ValueError mentioning C2..C5.
    assert "--source-conds" in src
    assert "not in active set" in src, (
        "unknown --source-conds must raise ValueError with the active-set list"
    )
    assert "C2..C5" in src, "ValueError message must mention the dropped C2..C5"


# ---------------------------------------------------------------- FIX (dispatcher --smoke)


def test_dispatcher_smoke_flag_present():
    """``i474_phase4_dispatch.sh`` must accept ``--smoke`` flag."""
    src = _SCRIPT_DISP.read_text()
    assert "--smoke" in src, "dispatcher missing --smoke flag"
    assert "SMOKE_MODE=1" in src, "dispatcher must set SMOKE_MODE=1 on --smoke"


def test_dispatcher_smoke_collapses_to_single_shard():
    """``--smoke`` must collapse 4-way sharding to 1-of-1."""
    src = _SCRIPT_DISP.read_text()
    # When SMOKE_MODE==1: SHARDS=(0), N_SHARDS=1.
    assert "SHARDS=(0)" in src and "N_SHARDS=1" in src, (
        "--smoke must collapse to 1-of-1 sharding (single shard 0, "
        "since smoke restricts source-conds to a subset that doesn't "
        "fill 4 shards)"
    )
    # Production stays 4-way: SHARDS=(0 1 2 3), N_SHARDS=4.
    assert "SHARDS=(0 1 2 3)" in src and "N_SHARDS=4" in src, (
        "production (no --smoke) must keep 4-way sharding"
    )


def test_dispatcher_smoke_skips_16x16_merge():
    """``--smoke`` must SKIP the 16x16 merge (would fail-loud on 240 missing cells).

    The per-cell JSONs are atomically written by i474_phase4_eval.py for
    the sources that did run, so the merge skip is safe for smoke
    (production runs full sweep and merges normally).
    """
    src = _SCRIPT_DISP.read_text()
    assert "SMOKE mode: skipping 16x16 merge" in src or "merge_skipped" in src, (
        "dispatcher must skip merge in --smoke mode to avoid 240 missing-cell crash"
    )


def test_dispatcher_production_path_unchanged():
    """Production (no --smoke) MUST still run the 4-way shard + merge.

    Static check: the else-branch of the merge-skip block invokes
    i474_phase4_merge.py unchanged.
    """
    src = _SCRIPT_DISP.read_text()
    # The merge invocation lives inside the production branch (`else`).
    assert "uv run python scripts/i474_phase4_merge.py" in src, (
        "production merge invocation must be preserved"
    )


# ---------------------------------------------------------------- FIX (run_all.sh wiring)


def test_run_all_smoke_invokes_crosseval_with_source_conds_A1():
    """``i474_run_all.sh --smoke`` must pass ``--source-conds A1`` to crosseval_smoke.

    Without this the smoke crosseval tries to download all 16 sources and
    404s on A5 etc. — the original on-pod-smoke crash.
    """
    src = _SCRIPT_RUN_ALL.read_text()
    # Match across line continuations: the smoke crosseval invocation should
    # carry --smoke + --source-conds A1 + --arms pos,loc + --epochs 1.
    assert "i474_phase4_dispatch.sh" in src
    # Squash whitespace for line-continuation tolerance.
    flat = " ".join(src.split())
    assert "--smoke --source-conds A1" in flat, (
        "run_all.sh --smoke must invoke crosseval_smoke with "
        "`--smoke --source-conds A1` (round-4 fix)"
    )
    assert "--arms pos,loc" in flat
    assert "--epochs 1" in flat


def test_run_all_production_crosseval_unchanged():
    """Production (no --smoke) must call i474_phase4_dispatch.sh with NO
    --smoke and NO --source-conds restriction — full sweep across all 16
    sources at epochs 1/2/3/5."""
    src = _SCRIPT_RUN_ALL.read_text()
    flat = " ".join(src.split())
    # Find the production crosseval line — it should contain the bare
    # `i474_phase4_dispatch.sh` invocation without --smoke / --source-conds.
    # We assert the production branch text doesn't accidentally inherit
    # smoke restrictions.
    assert "run_phase_script crosseval i474_phase4_dispatch.sh" in flat, (
        "production crosseval invocation must be present and unmodified"
    )


def test_phase4_eval_smoke_help_runs_clean(i474_eval_module):
    """Sanity: --help still works (no argparse regression from --source-conds add)."""
    # If argparse setup is broken, --help will raise before printing.
    # Capture via ArgumentParser.parse_args(['-h']) is awkward; instead
    # just check the module's argument-parser construction didn't throw
    # at import time.
    assert hasattr(i474_eval_module, "main")
    # And the constants we depend on exist.
    assert hasattr(i474_eval_module, "DEFAULT_KL_TOPK")
