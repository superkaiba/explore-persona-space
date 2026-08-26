"""Shared output-path constants for issue #2564 (r2 blocker 4).

The PE analysis producer (``scripts/issue2564_analysis.py``) and the figures
consumer (``scripts/issue2564_figures.py``) must agree BYTE-FOR-BYTE on the
default results directory in BOTH modes (production repo-root-anchored, smoke
``/tmp`` twin).  Round 1 shipped a drifted pair (analysis wrote
``<repo>/eval_results/issue_2564``; figures read cwd-relative
``eval_results/issue_2564/pe``), so default-invoked figures would silently
read nothing / stale state.  Both scripts now import THESE constants; the
alignment is pinned by
``tests/test_issue2564_figures.py::test_default_results_dir_alignment``.
"""

from __future__ import annotations

from pathlib import Path

SMOKE_ROOT = Path("/tmp/issue-2564-smoke")
RESULTS_REL = Path("eval_results") / "issue_2564"


def repo_root() -> Path:
    """Repo root anchored on THIS package's location, never cwd.

    A cwd-relative default breaks under worktree / pod / tmp cwds (the r1
    figures bug class).  ``parents[4]`` walks
    issue2564/ -> experiments/ -> explore_persona_space/ -> src/ -> repo root.
    """
    return Path(__file__).resolve().parents[4]


def production_results_dir() -> Path:
    """Default PE-analysis out-dir / figures results-dir in production mode."""
    return repo_root() / RESULTS_REL


def smoke_results_dir() -> Path:
    """Default PE-analysis out-dir / figures results-dir under ``--smoke``."""
    return SMOKE_ROOT / RESULTS_REL
