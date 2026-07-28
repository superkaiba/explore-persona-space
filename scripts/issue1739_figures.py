"""Phase-4 CLI for issue #1739: render the plan-§6 figure set from result JSONs.

Reads ``all_arms_spearman.json`` (+ optional map-diagnostics / composition
JSONs) and renders through ``experiments.issue_1739.figures`` (paper-plots
conventions; one color = one arm family across every figure). No network.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_figures.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

logger = logging.getLogger("issue1739_figures")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("eval_results/issue_1739/arm_results/all_arms_spearman.json"),
    )
    ap.add_argument(
        "--map-diag", type=Path, default=None, help="pooled per-rung map-diagnostic JSON"
    )
    ap.add_argument("--composition", type=Path, default=None, help="composition-factor rows JSON")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_1739"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = _parse_args(argv)
    from explore_persona_space.experiments.issue_1739 import figures

    summary = json.loads(args.summary.read_text())
    paths = figures.render_summary_figures(summary, args.out_dir)
    if args.map_diag is not None:
        rows = json.loads(args.map_diag.read_text())
        paths += list(figures.fig_map_degradation(rows, args.out_dir).values())
    if args.composition is not None:
        rows = json.loads(args.composition.read_text())
        paths += list(figures.fig_composition(rows, args.out_dir).values())
    for p in paths:
        print(f"[figures] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
