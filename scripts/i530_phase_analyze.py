# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ + × + − intentional
#!/usr/bin/env python3
"""Task #530 Phase 4 — analyze entrypoint.

Thin wrapper around `i504_phase_analyze.py` with the #530 namespace pinned:
  * `--slab-root eval_results/issue_530`
  * `--positioned-arms v3` (the `c504v3_*` slug set the #530 sweep uses)
  * `--seeds 42,137`
  * Outputs `eval_results/issue_530/analysis_v1.json` per plan §6.5.

After the analysis runs, this script also writes a side-by-side comparison
artifact `eval_results/issue_530/comparison_504_vs_530.json` that pairs the
#530 partial-Spearman ρ values with #504's (read from
`eval_results/issue_504/analyze_summary.json` if present on disk; ignored
silently when the #504 artifact is absent, since the comparison is plan
§6 step 1 figure data — not required for the analysis primary deliverable).

Usage:
    uv run python scripts/i530_phase_analyze.py
    uv run python scripts/i530_phase_analyze.py --slab-root eval_results/issue_530_alt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i530.phase_analyze")


def _maybe_load_504_comparison(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.warning("#504 analyze_summary.json present but unreadable (%s); skipping comparison", e)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_530"))
    ap.add_argument(
        "--phase0-path",
        type=Path,
        default=None,
        help=(
            "Optional Phase 0 calibration JSON (the #504 rig consumes this for "
            "`chosen_checkpoint_fraction`; #530 reads at the band-stop checkpoint "
            "so this is informational). Defaults to "
            "<slab-root>/phase0_calibration.json."
        ),
    )
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=None,
        help=("Phase 0.5 gates artifact. Defaults to <slab-root>/phase0_5_gates.json."),
    )
    ap.add_argument(
        "--base-prior-path",
        type=Path,
        default=None,
        help=(
            "Optional base-prior covariate JSON (forwarded to i504_phase_analyze "
            "as --base-prior-path). When absent, the covariate runs with 0.0 "
            "placeholder."
        ),
    )
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument(
        "--comparison-504-path",
        type=Path,
        default=Path("eval_results/issue_504/analyze_summary.json"),
        help=(
            "Path to the #504 analyze_summary.json read for the side-by-side "
            "partial-ρ comparison. Silently skipped when absent."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze_530] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    slab_root: Path = args.slab_root
    phase0_path = args.phase0_path or (slab_root / "phase0_calibration.json")
    phase05_path = args.phase05_path or (slab_root / "phase0_5_gates.json")

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i504_phase_analyze.py",
        "--slab-root",
        str(slab_root),
        "--phase0-path",
        str(phase0_path),
        "--phase05-path",
        str(phase05_path),
        "--seeds",
        args.seeds,
        # The #530 sweep uses the c504v3_* slug set (plan v2 changelog +
        # body.md prescription). i504_phase_analyze.py iterates the matching
        # POSITIONED_ARM_SLUGS_V3 tuple under this flag.
        "--positioned-arms",
        "v3",
    ]
    if args.base_prior_path is not None:
        cmd.extend(["--base-prior-path", str(args.base_prior_path)])

    log.info("[phase=analyze_530_dispatch] %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)

    summary_path = slab_root / "analyze_summary.json"
    if not summary_path.exists():
        # The i504_phase_analyze script writes analyze_summary.json under the
        # slab-root by default; if it's missing the subprocess silently failed.
        raise RuntimeError(
            f"i504_phase_analyze subprocess exited 0 but {summary_path} missing — "
            "silent analysis failure (feedback_eval_script_silent_not_present_misdiagnosis)."
        )
    log.info("[phase=analyze_530_done] analysis summary → %s", summary_path)

    # Rename / alias to the plan §6.5 path `analysis_v1.json` as well — the
    # primary deliverable glob references that exact filename.
    primary_path = slab_root / "analysis_v1.json"
    primary_path.write_text(summary_path.read_text())
    log.info("[phase=analyze_530_alias] aliased %s ← %s", primary_path, summary_path)

    # Side-by-side #504 vs #530 comparison (plan §6 hero figure 1 data).
    comparison_path = slab_root / "comparison_504_vs_530.json"
    comp_504 = _maybe_load_504_comparison(args.comparison_504_path)
    comp_530 = json.loads(summary_path.read_text())
    comparison = {
        "task_id": 530,
        "parent_task_id": 504,
        "ts": datetime.now(UTC).isoformat(),
        "issue_530_analysis": comp_530,
        "issue_504_analysis": comp_504,
        "issue_504_path": str(args.comparison_504_path),
        "issue_504_available": comp_504 is not None,
        "hero_predictors": ["shadow_angle", "d_nearest_neg_nd"],
        "note": (
            "Plan §6 hero figure: side-by-side partial-Spearman ρ for "
            "shadow_angle and d_nearest_neg_nd at #504's saturated anchor "
            "(if available) vs #530's de-saturated anchor."
        ),
    }
    comparison_path.write_text(json.dumps(comparison, indent=2))
    log.info(
        "[phase=done] wrote comparison → %s (504_available=%s)",
        comparison_path,
        comp_504 is not None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
