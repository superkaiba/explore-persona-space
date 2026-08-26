"""Issue #2564 figures — default-path alignment + smoke-guard pins (r2 blocker 4).

The r1 figures consumer defaulted to a cwd-relative ``eval_results/issue_2564/pe``
while the analysis producer wrote ``<repo>/eval_results/issue_2564`` — a
default-invoked figures run silently read nothing. Both scripts now import the
shared ``experiments.issue2564.paths`` constants; these tests pin the alignment
in BOTH modes plus the resolved-path smoke guard on the committed figures tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2564_analysis as A  # noqa: E402
import issue2564_figures as F  # noqa: E402

from explore_persona_space.experiments.issue2564 import paths as P2564  # noqa: E402


def test_default_results_dir_alignment_production():
    """Producer default out-dir == consumer default results-dir (production)."""
    cfg = A.build_config(A.parse_args(["--upload", "none"]))
    assert F.DEFAULT_RESULTS_DIR == cfg.out_dir == P2564.production_results_dir()
    assert F.DEFAULT_RESULTS_DIR.is_absolute()  # never cwd-relative (the r1 bug)


def test_default_results_dir_alignment_smoke():
    """Producer smoke out-dir == the consumer's --smoke results-dir rebind."""
    cfg = A.build_config(
        A.parse_args(["--smoke", "--upload", "none", "--manip-check", "/tmp/manip.json"])
    )
    assert F.SMOKE_RESULTS_DIR == cfg.out_dir == P2564.smoke_results_dir()
    assert str(F.SMOKE_RESULTS_DIR).startswith(str(P2564.SMOKE_ROOT))


def test_smoke_refuses_committed_figures_tree(tmp_path):
    """--smoke with an out-dir RESOLVING into the committed figures/ tree
    refuses (the r1 string-startswith guard missed absolute paths)."""
    with pytest.raises(SystemExit, match="committed figures"):
        F.main(
            [
                "--smoke",
                "--results-dir",
                str(tmp_path),
                "--out-dir",
                str(P2564.repo_root() / "figures" / "issue_2564"),
            ]
        )


def test_paths_repo_root_is_the_repo():
    root = P2564.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert P2564.production_results_dir() == root / "eval_results" / "issue_2564"
