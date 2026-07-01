"""Freshest-log tail selection across ALL FOUR log layouts (#791).

The completion poller (``poll_pipeline``) fetches a WIDE 500-line tail per log
layout the probe knows about. Before #791 the tail EXCERPT (surfaced in
notifications) and the ``status="dead"`` crash SIGNATURE (which feeds the
CUDA-IMA / OUR_CODE_FRAME failover predicates in ``backend_poll``) were selected
by mtime-argmax over only {main, cell} — even though the STALENESS verdict
already unioned all four layouts {main, cell, phase, shard} (#468/#488). So a
multi-arm run whose later arm wrote ONLY to a per-phase (``issue-<N>-<arm>.log``)
or shard (``logs/issue_<N>/*.log``) layout kept ``running`` correctly, but
surfaced a STALE main-log tail — misinforming the surface a watcher acts on.

These tests pin the fix from the OUTSIDE (a reader could trust the poller from
these alone):

* the excerpt/signature helper (``_tail_excerpt_and_crash_signature``) selecting
  the freshest NON-EMPTY tail by mtime-argmax over all four layouts;
* the tie-break + empty-tail safety rules (deterministic fall back to the main
  log);
* the probe-output parser (``_parse_probe_stdout``) lifting the two new
  ``PHASE_TAIL_START``/``SHARD_TAIL_START`` blocks into ``phase_log_tail`` /
  ``shard_log_tail``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors the loader in
    ``tests/test_poll_pipeline_zombie_gpu.py``)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_freshest_tail_under_test")


def _probe(
    *,
    log_tail: str = "",
    cell_log_tail: str = "",
    phase_log_tail: str = "",
    shard_log_tail: str = "",
) -> dict[str, str]:
    """A probe dict carrying the four tail surfaces the helper reads."""
    return {
        "log_tail": log_tail,
        "cell_log_tail": cell_log_tail,
        "phase_log_tail": phase_log_tail,
        "shard_log_tail": shard_log_tail,
    }


# ── _tail_excerpt_and_crash_signature: freshest-of-four selection ─────────────


def test_only_main_tail_populated_returns_main() -> None:
    """Baseline (unchanged): with only the main-log tail populated, the excerpt
    is its last 5 lines and (on a dead poll) it is the crash signature — no
    behavior change for a non-cell/phase/shard run."""
    wide = "\n".join(f"main line {i}" for i in range(10))
    probe = _probe(log_tail=wide)

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe, status="dead", mtime_epoch=100, cell_mtime_epoch=0
    )
    assert excerpt == "\n".join(wide.splitlines()[-5:])
    assert sig == wide


def test_cell_tail_freshest_returns_cell() -> None:
    """Pre-existing behavior preserved: cell mtime > main mtime -> the cell tail
    is the wide surface for BOTH the excerpt and the crash signature."""
    main = "stale dispatcher tail"
    cell = "\n".join(f"cell line {i}" for i in range(8))
    probe = _probe(log_tail=main, cell_log_tail=cell)

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe, status="dead", mtime_epoch=100, cell_mtime_epoch=200
    )
    assert excerpt == "\n".join(cell.splitlines()[-5:])
    assert sig == cell


def test_phase_tail_freshest_returns_phase_and_is_crash_signature() -> None:
    """#791 NEW: the per-phase log is the freshest of all four -> its tail is the
    excerpt, and on a ``status=dead`` poll it becomes the crash signature (the
    surface the failover predicates scan). The stale main tail must NOT win."""
    main = "\n".join(f"stale main {i}" for i in range(6))
    phase = "\n".join(f"live phase line {i}" for i in range(30))
    probe = _probe(log_tail=main, phase_log_tail=phase)

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="dead",
        mtime_epoch=100,
        cell_mtime_epoch=0,
        phase_log_mtime_epoch=500,
        shard_log_mtime_epoch=0,
    )
    assert excerpt == "\n".join(phase.splitlines()[-5:])
    assert sig == phase
    # The stale main tail contributes nothing to either surface.
    assert "stale main" not in excerpt
    assert "stale main" not in sig


def test_shard_tail_freshest_returns_shard() -> None:
    """#791 NEW: the shard/per-job log is the freshest of all four -> its tail
    wins over main, cell, and phase."""
    main = "stale main"
    cell = "\n".join(f"cell {i}" for i in range(4))
    phase = "\n".join(f"phase {i}" for i in range(4))
    shard = "\n".join(f"shard line {i}" for i in range(20))
    probe = _probe(log_tail=main, cell_log_tail=cell, phase_log_tail=phase, shard_log_tail=shard)

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="dead",
        mtime_epoch=100,
        cell_mtime_epoch=200,
        phase_log_mtime_epoch=300,
        shard_log_mtime_epoch=999,
    )
    assert excerpt == "\n".join(shard.splitlines()[-5:])
    assert sig == shard


def test_tie_breaks_to_main() -> None:
    """Safety: two sources tied on the max mtime -> the result deterministically
    picks the main log (main is first in the candidate list, and ``max`` returns
    the first maximal element)."""
    main = "\n".join(f"main {i}" for i in range(6))
    shard = "\n".join(f"shard {i}" for i in range(6))
    probe = _probe(log_tail=main, shard_log_tail=shard)

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="dead",
        mtime_epoch=500,
        cell_mtime_epoch=0,
        phase_log_mtime_epoch=0,
        shard_log_mtime_epoch=500,  # tie with main
    )
    assert excerpt == "\n".join(main.splitlines()[-5:])
    assert sig == main


def test_empty_freshest_tail_ignored_next_nonempty_wins() -> None:
    """Safety: a source with the freshest mtime but an EMPTY tail (e.g. a
    per-phase log just created, nothing written yet) is skipped; the next
    freshest NON-EMPTY tail wins instead of blanking the excerpt."""
    main = "stale main"
    phase = "\n".join(f"phase live {i}" for i in range(10))
    # phase has a populated tail at mtime 300; the shard mtime is FRESHER (400)
    # but its tail is empty, so it must be ignored.
    probe = _probe(log_tail=main, phase_log_tail=phase, shard_log_tail="")

    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="dead",
        mtime_epoch=100,
        cell_mtime_epoch=0,
        phase_log_mtime_epoch=300,
        shard_log_mtime_epoch=400,  # freshest, but empty tail -> skipped
    )
    assert excerpt == "\n".join(phase.splitlines()[-5:])
    assert sig == phase


def test_all_empty_falls_back_to_main_empty() -> None:
    """Safety: when NO source has a tail, the helper falls back to the (empty)
    main-log tail rather than raising on an empty ``max`` sequence."""
    probe = _probe()  # every tail empty
    excerpt, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="dead",
        mtime_epoch=100,
        cell_mtime_epoch=200,
        phase_log_mtime_epoch=300,
        shard_log_mtime_epoch=400,
    )
    assert excerpt == ""
    assert sig == ""


def test_running_poll_populates_no_crash_signature() -> None:
    """A non-dead (running) poll never populates a crash signature, whichever
    layout is freshest."""
    probe = _probe(log_tail="main", phase_log_tail="phase content")
    _, sig = pp._tail_excerpt_and_crash_signature(
        probe,
        status="running",
        mtime_epoch=100,
        cell_mtime_epoch=0,
        phase_log_mtime_epoch=500,
    )
    assert sig is None


# ── _parse_probe_stdout: PHASE_TAIL / SHARD_TAIL block parsing ────────────────


def test_parse_probe_stdout_lifts_phase_and_shard_tails() -> None:
    """The parser lifts the ``PHASE_TAIL_START``/``END`` and
    ``SHARD_TAIL_START``/``END`` blocks into ``phase_log_tail`` /
    ``shard_log_tail`` (mirrors the ``CELL_TAIL`` parse), and keeps the scalar
    mtime keys and the other tails intact."""
    stdout = "\n".join(
        [
            "PID_ALIVE=1",
            "MTIME_EPOCH=100",
            "CELL_MTIME_EPOCH=0",
            "TAIL_START",
            "main tail a",
            "main tail b",
            "TAIL_END",
            "CELL_TAIL_START",
            "CELL_TAIL_END",
            "PHASE_LOG_MTIME_EPOCH=500",
            "PHASE_TAIL_START",
            "phase tail line 1",
            "phase tail line 2",
            "PHASE_TAIL_END",
            "SHARD_LOG_MTIME_EPOCH=600",
            "SHARD_TAIL_START",
            "shard tail line 1",
            "SHARD_TAIL_END",
            "GPU_UTIL=95",
        ]
    )
    parsed = pp._parse_probe_stdout(stdout)

    assert parsed["log_tail"] == "main tail a\nmain tail b"
    assert parsed["cell_log_tail"] == ""
    assert parsed["phase_log_tail"] == "phase tail line 1\nphase tail line 2"
    assert parsed["shard_log_tail"] == "shard tail line 1"
    assert parsed["phase_log_mtime_epoch"] == "500"
    assert parsed["shard_log_mtime_epoch"] == "600"
    assert parsed["gpu_util"] == "95"


def test_parse_probe_stdout_empty_phase_shard_tails_default_empty() -> None:
    """A probe stdout with empty PHASE/SHARD tail blocks (the no-log-yet case the
    heredoc emits) leaves both tail keys as empty strings, not the literal
    sentinel lines."""
    stdout = "\n".join(
        [
            "PID_ALIVE=1",
            "PHASE_LOG_MTIME_EPOCH=0",
            "PHASE_TAIL_START",
            "PHASE_TAIL_END",
            "SHARD_LOG_MTIME_EPOCH=0",
            "SHARD_TAIL_START",
            "SHARD_TAIL_END",
        ]
    )
    parsed = pp._parse_probe_stdout(stdout)
    assert parsed["phase_log_tail"] == ""
    assert parsed["shard_log_tail"] == ""


def test_ssh_failed_fallback_carries_empty_phase_shard_tails() -> None:
    """The SSH-failure fallback dict (returned when the ssh round-trip fails)
    carries the two new tail keys as empty strings, so a downstream reader never
    KeyErrors on them after a transport failure."""
    # _parse_probe_stdout's default dict is the same shape the ssh-failed
    # fallback returns; assert both new keys are present + empty on empty stdout.
    parsed = pp._parse_probe_stdout("")
    assert parsed["phase_log_tail"] == ""
    assert parsed["shard_log_tail"] == ""
