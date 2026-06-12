#!/usr/bin/env python3
"""Task #608 Phase G — off-pod judge + analysis driver (runs on the VM AFTER
pod termination; CPU/API-only per the CLAUDE.md off-pod rule).

Sequence (each phase persists its output the moment it completes):
    G1 [phase=p1_kappa]      kappa calibration (1,000 stratified fresh rollouts,
                             Haiku vs Sonnet). Gate: kappa >= 0.7 ACCEPT;
                             0.5 <= kappa < 0.7 FLAG (file written, run
                             continues — Sonnet adjudication is an analyzer
                             decision); kappa < 0.5 BLOCK (exit 1).
    G2 [phase=p2_judge]      ONE unified Haiku pass over every fresh completion
                             (12 new-arm + 7 re-eval endpoint cells + epoch-1/2
                             checkpoint evals). Resumable per panel file.
    G3 [phase=p3_crosscheck] stored-vs-fresh descriptive replication read.
    G4 [phase=p4_analyze]    registered analysis + figures
                             -> eval_results/issue_608/analyze_summary_608.json
    G5 [phase=p5_upload]     judgments + calibration + summaries -> HF data repo
                             (fail-loud).

Inputs come from git / the HF-uploaded eval trees pulled to
``eval_results/issue_608`` — never from the (already terminated) pod.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_posonly_608 import (  # noqa: E402
    FOLLOWUP_LABEL,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    HF_SUBCEILING_DATA_PREFIX,
)
from explore_persona_space.experiments.sycophancy_posonly_608.judge_pass_608 import (  # noqa: E402
    KAPPA_ACCEPT,
    KAPPA_FLAG,
    run_full_judge_pass,
    run_kappa_calibration,
    stored_vs_fresh_crosscheck,
)

log = logging.getLogger("issue608_judge_and_analyze")


def _upload_outputs(slab_root: Path) -> None:
    """Upload judgments + calibration + summary files to the HF data repo,
    fail-loud. Mirrors the pod-side eval-tree layout under
    ``issue608_sycophancy_posonly/eval_results/``."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — cannot upload judgments")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(slab_root),
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_DATA_PREFIX}/eval_results",
        allow_patterns=[
            "**/judgments/**",
            "judge_calibration_608/**",
            "analyze_summary_608.json",
            "stored_vs_fresh_crosscheck.json",
        ],
    )
    uploaded = [
        f
        for f in api.list_repo_files(HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_DATA_PREFIX}/eval_results")
    ]
    if not any("judgments" in f for f in uploaded):
        raise RuntimeError("Judgments upload not visible via list_repo_files — verify manually")
    log.info("uploaded %d files under %s/eval_results", len(uploaded), HF_DATA_PREFIX)


def _upload_subceiling_outputs(slab_root: Path) -> None:
    """Upload follow-up judgments + spot-check + summary to the HF data repo,
    fail-loud, under the sub_ceiling_install prefix (plan v5 §10)."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — cannot upload judgments")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(slab_root),
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_SUBCEILING_DATA_PREFIX}/eval_results",
        allow_patterns=[
            "**/judgments/**",
            "judge_calibration_subceiling/**",
            "analyze_summary_subceiling.json",
        ],
    )
    uploaded = [
        f
        for f in api.list_repo_files(HF_DATA_REPO, repo_type="dataset")
        if f.startswith(f"{HF_SUBCEILING_DATA_PREFIX}/eval_results")
    ]
    if not any("judgments" in f for f in uploaded):
        raise RuntimeError(
            "Follow-up judgments upload not visible via list_repo_files — verify manually"
        )
    log.info("uploaded %d files under %s/eval_results", len(uploaded), HF_SUBCEILING_DATA_PREFIX)


def _run_followup(args: argparse.Namespace) -> int:
    """The sub-ceiling-install off-pod sequence (plan v5 §4 diff 4):
    F1 judge pass -> F2 mid-band κ spot-check (gate >= 0.7) -> F3 §6 decision
    rule + figures -> F4 fail-loud upload. Each phase persists its output the
    moment it completes."""
    from explore_persona_space.experiments.sycophancy_posonly_608.judge_pass_subceiling import (
        KAPPA_SPOTCHECK_GATE,
        run_midband_spotcheck,
        run_subceiling_judge_pass,
    )

    if args.slab_root.name != FOLLOWUP_LABEL:
        args.slab_root = args.slab_root / FOLLOWUP_LABEL
    if args.figures_dir.name != FOLLOWUP_LABEL:
        args.figures_dir = args.figures_dir / FOLLOWUP_LABEL

    if not args.skip_judge:
        log.info("[phase=p1_judge] follow-up Haiku pass (resumable, 108 step reads)")
        totals = run_subceiling_judge_pass(args.slab_root, args.seed, concurrency=args.concurrency)
        log.info(
            "judge pass done: %d panels judged, %d skipped (already judged)",
            totals["n_panels_judged"],
            totals["n_panels_skipped"],
        )

    if not args.skip_spotcheck:
        log.info("[phase=p2_spotcheck] 200-rollout mid-band κ spot-check (Haiku vs Sonnet)")
        report = run_midband_spotcheck(args.slab_root, args.seed, concurrency=args.concurrency)
        kappa = report["kappa"]
        # NaN-safe gate (parent round-3 convention): non-finite κ — degenerate
        # expected agreement or an empty sample — is unmeasurable reliability
        # and routes to BLOCK, never silently past a `kappa < gate` that is
        # False for NaN.
        if not math.isfinite(kappa) or kappa < KAPPA_SPOTCHECK_GATE:
            reason = (
                f"spot-check kappa={kappa if math.isfinite(kappa) else 'non-finite'} "
                f"below gate {KAPPA_SPOTCHECK_GATE}: parent κ=0.881 does NOT transfer to "
                "the mid-install output distribution. Pre-registered escalation (plan v5 "
                "§11): the parent's full 1,000-rollout recalibration + Sonnet adjudication "
                "on disagreements — an orchestrator/plan decision, never auto-run."
            )
            block = {
                "decision": "BLOCK",
                "kappa": kappa if math.isfinite(kappa) else None,
                "spotcheck_n": report.get("spotcheck_n"),
                "reason": reason,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            block_dir = args.slab_root / "judge_calibration_subceiling"
            block_dir.mkdir(parents=True, exist_ok=True)
            with open(block_dir / "SPOTCHECK_BLOCK.json", "w") as f:
                json.dump(block, f, indent=2)
            log.error("BLOCK: %s — exiting 1", reason)
            return 1
        log.info("spot-check kappa=%.4f >= %.2f — judge reuse holds", kappa, KAPPA_SPOTCHECK_GATE)

    if not args.skip_analyze:
        log.info("[phase=p3_analyze] §6 decision rule + figures")
        from explore_persona_space.experiments.sycophancy_posonly_608.analyze_subceiling import (
            analyze,
        )

        analyze(
            slab_root=args.slab_root,
            seed=args.seed,
            figures_dir=args.figures_dir,
            n_boot=args.bootstrap_n,
            parent_summary_path=args.parent_summary,
        )

    if args.hf_upload:
        log.info("[phase=p4_upload] judgments + spot-check + summary -> HF data repo")
        _upload_subceiling_outputs(args.slab_root)

    log.info("[phase=done]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #608 Phase G — off-pod kappa + unified judge pass + analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_608"))
    parser.add_argument(
        "--frozen-refs",
        type=Path,
        default=Path("data/issue_608/frozen_refs"),
        help="Dir with the pinned base_panel_rates.json + analyze_summary.json "
        "(prefetch_inputs.py output; descriptive cross-check only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--kappa-n", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/issue_608"))
    parser.add_argument("--skip-kappa", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-crosscheck", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument(
        "--skip-spotcheck",
        action="store_true",
        help="(--followup only) skip the F2 mid-band κ spot-check.",
    )
    parser.add_argument(
        "--followup",
        choices=[FOLLOWUP_LABEL],
        default=None,
        help="Run the sub-ceiling-install off-pod sequence instead of the parent "
        "Phase G: F1 judge pass over the 108 step reads, F2 mid-band κ spot-check "
        "(gate >= 0.7), F3 §6 decision rule + figures, F4 upload.",
    )
    parser.add_argument(
        "--parent-summary",
        type=Path,
        default=Path("eval_results/issue_608/analyze_summary_608.json"),
        help="(--followup only) parent committed summary: parity references + "
        "reused fresh base rates.",
    )
    parser.add_argument(
        "--hf-upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload judgments/calibration/summary to the HF data repo (fail-loud).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", stream=sys.stdout
    )

    if args.followup is not None:
        return _run_followup(args)

    if not args.skip_kappa:
        log.info("[phase=p1_kappa] calibration: %d stratified rollouts", args.kappa_n)
        report = run_kappa_calibration(
            args.slab_root, args.seed, n=args.kappa_n, concurrency=args.concurrency
        )
        kappa = report["kappa"]
        calib_n = report.get("calibration_subset_size")
        # NaN-safe gate: _cohens_kappa returns float("nan") on degenerate expected
        # agreement / empty sample, and `nan < threshold` is False — so a bare
        # `kappa < KAPPA_FLAG` would silently PASS an unmeasurable judge. Treat any
        # non-finite kappa as the blocking branch (round-2 reconciler, binding).
        if not math.isfinite(kappa) or kappa < KAPPA_FLAG:
            if not math.isfinite(kappa):
                reason = (
                    f"kappa={kappa} is non-finite (degenerate expected agreement or empty "
                    f"calibration sample; calibration n={calib_n}): judge reliability is "
                    "unmeasurable on the new output distribution; plan §8 requires "
                    "escalation (judge prompt revision is a plan amendment)."
                )
            else:
                reason = (
                    f"kappa={kappa:.4f} < {KAPPA_FLAG}: judge unreliable on the new "
                    "output distribution; plan §8 requires escalation (judge prompt revision "
                    "is a plan amendment)."
                )
            block = {
                "decision": "BLOCK",
                # None (not NaN) when non-finite so BLOCK.json stays strict-JSON-parseable;
                # the reason string carries the raw value.
                "kappa": kappa if math.isfinite(kappa) else None,
                "calibration_subset_size": calib_n,
                "reason": reason,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            with open(args.slab_root / "judge_calibration_608" / "BLOCK.json", "w") as f:
                json.dump(block, f, indent=2)
            log.error("BLOCK: %s — exiting 1", reason)
            return 1
        if not math.isfinite(kappa) or kappa < KAPPA_ACCEPT:
            flag = {
                "decision": "FLAG",
                "kappa": kappa,
                "reason": f"kappa={kappa:.4f} in [{KAPPA_FLAG}, {KAPPA_ACCEPT}): run continues; "
                "Sonnet adjudication on disagreements is an analyzer/orchestrator decision "
                "(plan §8).",
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            with open(args.slab_root / "judge_calibration_608" / "FLAG.json", "w") as f:
                json.dump(flag, f, indent=2)
            log.warning("kappa=%.4f below ACCEPT — FLAG written, continuing", kappa)

    if not args.skip_judge:
        log.info("[phase=p2_judge] unified Haiku pass (resumable)")
        totals = run_full_judge_pass(args.slab_root, args.seed, concurrency=args.concurrency)
        log.info(
            "judge pass done: %d panels judged, %d skipped (already judged)",
            totals["n_panels_judged"],
            totals["n_panels_skipped"],
        )

    if not args.skip_crosscheck:
        log.info("[phase=p3_crosscheck] stored-vs-fresh descriptive read")
        stored_vs_fresh_crosscheck(args.slab_root, args.frozen_refs, args.seed)

    if not args.skip_analyze:
        log.info("[phase=p4_analyze] registered analysis + figures")
        from explore_persona_space.experiments.sycophancy_posonly_608.analyze_608 import analyze

        analyze(
            slab_root=args.slab_root,
            seed=args.seed,
            figures_dir=args.figures_dir,
            n_boot=args.bootstrap_n,
        )

    if args.hf_upload:
        log.info("[phase=p5_upload] judgments + summaries -> HF data repo")
        _upload_outputs(args.slab_root)

    log.info("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
