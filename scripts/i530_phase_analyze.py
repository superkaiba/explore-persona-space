# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ρ + × + − intentional
#!/usr/bin/env python3
"""Task #530 Phase 4 — analyze entrypoint.

Thin wrapper around `i504_phase_analyze.py` with the #530 namespace pinned:
  * `--slab-root eval_results/issue_530`
  * `--positioned-arms v3` (the `c504v3_*` slug set the #530 sweep uses)
  * `--seeds 42,137`
  * Outputs `eval_results/issue_530/analysis_v1.json` per plan §6.5.

#530 does NOT have a Phase 0 calibration step the way #504 does — plan §4.4
step 1 names Phase 0 as "distance pre-flight" (the centroid gates, the
phase05 artifact), and plan §4.4 step 4 pins the analyzer's checkpoint read
to "frac=1.00 of band-stop steps" (the band-stop is in-loop; the marker-only
lr is plan-pinned, NOT picked from a smoke ladder). The shared
`i504_phase_analyze.py` still requires a `phase0_calibration.json` to read
`chosen_checkpoint_fraction` from, so this wrapper SYNTHESIZES that artifact
in-memory and writes it to `<slab-root>/phase0_calibration.json` before
dispatching the subprocess. The synthesized payload carries
`chosen_checkpoint_fraction = 1.0` (= frac=1.00 of band-stop steps per plan)
and `verdict = "pass"`; provenance keys (`source`, `note`,
`task_id_minted_by`) record that it was minted by this wrapper, not by a
real smoke ladder, so a future reader does NOT mistake it for an upstream
calibration. The phase05 artifact is auto-detected at either of two names —
`phase0_5_gates.json` (the i504_phase_phase05.py default) OR
`phase0_geometry_v1.json` (the plan §4.4 step 1 namespaced name the pod's
Phase 0 produced) — whichever exists on disk.

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

# Plan §4.4 step 4: the analyzer reads at the band-stop checkpoint =
# frac=1.00 of band-stop steps. The trajectory.json schema emits a fraction
# in [0.25, 0.50, 0.75, 1.00]; `build_rows` in #504's analyze module reads
# the checkpoint dict whose `frac == chosen_checkpoint_fraction`. So pinning
# 1.0 here = "read the final band-stop checkpoint", which is the plan-pinned
# headline for #530.
SYNTHESIZED_CHOSEN_FRAC: float = 1.0

# Two filenames the wrapper accepts for the Phase 0.5 gates artifact. The
# in-tree i504_phase_phase05.py default is `phase0_5_gates.json`, but plan
# §4.4 step 1 also names `phase0_geometry_v1.json` (the namespaced #530
# version, produced when the orchestrator overrides `--out-path` on the pod).
# Auto-detection means the wrapper survives either spelling without forcing
# a rename on the pod's eval_results/issue_530 slab.
PHASE05_FILENAME_CANDIDATES: tuple[str, ...] = (
    "phase0_5_gates.json",
    "phase0_geometry_v1.json",
)


def _maybe_load_504_comparison(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.warning("#504 analyze_summary.json present but unreadable (%s); skipping comparison", e)
        return None


def _synthesize_phase0_calibration(
    out_path: Path,
    *,
    chosen_frac: float = SYNTHESIZED_CHOSEN_FRAC,
) -> Path:
    """Write a minimal phase0_calibration.json reflecting #530's plan-pinned anchor.

    #530 has no Phase 0 smoke ladder (lr is pre-pinned to 5e-6 per plan §4.2;
    the band-stop fires in-loop and writes the actual checkpoints to disk).
    But `i504_phase_analyze.py` -> `run_phase2_analysis` reads
    `chosen_checkpoint_fraction` out of a `phase0_calibration.json`, so we
    synthesize the minimum payload `load_phase0_pick` + `run_phase2_analysis`
    inspect: `verdict == "pass"` and a numeric `chosen_checkpoint_fraction`.

    Provenance keys (`source`, `note`, `task_id_minted_by`,
    `synthesized_at`) record that this artifact is minted by the #530
    wrapper, NOT by a real smoke ladder — a downstream reader that opens
    the file sees "synthesized for #530, NOT a calibration pick" rather
    than mistaking the 1.0 for a real smoke-table result.

    Args:
        out_path: where to write the artifact (the i530 wrapper passes
            `<slab-root>/phase0_calibration.json`).
        chosen_frac: which trajectory checkpoint frac to analyze; default
            1.0 = the band-stop final checkpoint (plan §4.4 step 4).

    Returns:
        The same `out_path` it wrote to (for chaining / logging).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": "pass",
        "chosen_checkpoint_fraction": float(chosen_frac),
        # Provenance: this is NOT a real phase0 calibration. #530 has no
        # smoke-table to pick from. Keep the keys discoverable so a future
        # reader doesn't mistake 1.0 for an evidence-based pick.
        "source": "i530_phase_analyze._synthesize_phase0_calibration",
        "task_id_minted_by": 530,
        "note": (
            "#530 has no Phase 0 calibration step (lr 5e-6 is plan-pinned, "
            "band-stop is in-loop). This artifact is synthesized so the "
            "shared i504_phase_analyze.py reads the band-stop final "
            "checkpoint (frac=1.00 of band-stop steps) per plan §4.4 step 4. "
            "Do NOT consume `chosen_checkpoint_fraction` here as an "
            "evidence-based pick — it is a plan-pinned routing constant."
        ),
        "synthesized_at": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[phase=analyze_530_synthesize_phase0] wrote synthesized phase0_calibration → %s "
        "(chosen_frac=%.2f, source=%s)",
        out_path,
        chosen_frac,
        payload["source"],
    )
    return out_path


def _resolve_phase05_path(slab_root: Path, override: Path | None) -> Path:
    """Return the on-disk Phase 0.5 gates artifact, accepting either filename.

    Auto-detects `phase0_5_gates.json` (the i504 default) and
    `phase0_geometry_v1.json` (the plan §4.4 step 1 namespaced name #530's
    Phase 0 wrote on the pod). The explicit `--phase05-path` override
    short-circuits this discovery.

    Raises:
        FileNotFoundError: neither candidate is on disk.
    """
    if override is not None:
        return override
    for name in PHASE05_FILENAME_CANDIDATES:
        candidate = slab_root / name
        if candidate.exists():
            log.info("[phase=analyze_530_phase05_resolved] %s -> %s", slab_root, candidate)
            return candidate
    raise FileNotFoundError(
        f"Phase 0.5 gates artifact missing under {slab_root} — looked for "
        f"{', '.join(PHASE05_FILENAME_CANDIDATES)}. Phase 0 (CPU centroid "
        "gates) must complete BEFORE Phase 4 analyze can read them."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_530"))
    ap.add_argument(
        "--phase0-path",
        type=Path,
        default=None,
        help=(
            "Optional Phase 0 calibration JSON. #530 has no Phase 0 calibration "
            "step (lr 5e-6 is plan-pinned, band-stop is in-loop), so when this "
            "flag is unset the wrapper SYNTHESIZES a minimal calibration "
            "artifact at <slab-root>/phase0_calibration.json with "
            "chosen_checkpoint_fraction=1.0 (plan §4.4 step 4: read at the "
            "band-stop final checkpoint). Pass an explicit path to override."
        ),
    )
    ap.add_argument(
        "--phase05-path",
        type=Path,
        default=None,
        help=(
            "Phase 0.5 gates artifact. When unset, auto-detects either "
            "<slab-root>/phase0_5_gates.json (i504 default) or "
            "<slab-root>/phase0_geometry_v1.json (plan §4.4 step 1 namespaced)."
        ),
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

    # Phase 0.5 gates artifact: explicit override OR auto-detect either
    # legacy/namespaced filename under slab-root. Fails loud if neither is
    # present (no Phase 4 analyze without the centroid gates).
    phase05_path = _resolve_phase05_path(slab_root, args.phase05_path)

    # Phase 0 calibration: #530 has no smoke ladder, so unless an explicit
    # path is provided (which would only happen if a caller wanted to inject
    # their own calibration), SYNTHESIZE the artifact reflecting the plan-
    # pinned final-checkpoint read. Idempotent: re-running the analyze step
    # overwrites with the same payload + a fresh timestamp.
    if args.phase0_path is not None:
        phase0_path = args.phase0_path
        log.info(
            "[phase=analyze_530_phase0_user_provided] using --phase0-path=%s "
            "(NOT synthesizing; caller-supplied calibration)",
            phase0_path,
        )
    else:
        phase0_path = _synthesize_phase0_calibration(slab_root / "phase0_calibration.json")

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
