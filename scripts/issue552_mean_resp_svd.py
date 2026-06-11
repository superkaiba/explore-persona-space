"""#552 free-analysis follow-up: SVD concentration read on mean-over-response shifts.

The #552 headline rests on the END-OF-RESPONSE-SLOT read (``delta_v``: layer-14
shift at the last response token). The persisted ``same``-variant shift tensors
ALSO carry a mean-over-response read per persona (``delta_v_mean_resp``); the
``base`` / ``on_policy`` variants never had one — the extraction computes it only
for ``variant == "same"`` (``activation_shift.extract_per_context_shifts``,
``also_compute_mean_over_response_em`` gate), so its absence there is by design,
not data loss.

This script runs the IDENTICAL SVD concentration read (same helpers, same null
conventions as ``issue_519_dispatch.py`` Phase D: ``assemble_M`` sorted persona
order, 1000-rep row-shuffle + entrywise sign-flip nulls seeded with the cell
seed) on BOTH reads:

* end-slot ``delta_v`` — recomputed from the tensors and cross-checked
  (fail-loud) against the persisted ``eval_results/issue_552/svd/{cell}.json``;
* mean-over-response ``delta_v_mean_resp`` — the follow-up read, available for
  the three same-variant benign cells (the headline variant).

Outputs per-cell JSONs (written as each cell completes) plus an aggregate
``summary.json`` with per-cell deltas (mean_resp minus end_slot), the cosine
between the two top singular directions, and a one-line concordance verdict per
cell against the parent's pre-registered concentration zones.

Follow-up ``em-arm-mean-resp-reextraction`` (plan v2 §4.2): ``--arm em``
``--variants same`` ``--seeds 42 137 256`` runs the same reads on the
re-extracted EM cells. ``--anchor-svd-dir`` additionally records the
pre-registered ±0.02 CROSS-RUN faithfulness gate (fresh end-slot metrics vs
the #521 persisted ``same_em_seed{S}.json`` — a fresh greedy regeneration on
a new pod, hence the looser tolerance) per cell; this is DISTINCT from the
5e-4 same-tensors cross-check against ``--svd-dir``, which keeps pointing at
the FRESH Phase-D JSONs computed from the same tensors. Gate FAIL is recorded
(infrastructure finding), never silently dropped.

Usage:
    uv run python scripts/issue552_mean_resp_svd.py \
        --shifts-dir eval_results/issue_552/shifts \
        --out eval_results/issue_552/svd_mean_resp

    # EM re-extraction (VM, post-termination):
    uv run python scripts/issue552_mean_resp_svd.py --arm em --variants same \
        --shifts-dir eval_results/issue_552/em-arm-mean-resp-reextraction/shifts \
        --svd-dir   eval_results/issue_552/em-arm-mean-resp-reextraction/svd \
        --out       eval_results/issue_552/em-arm-mean-resp-reextraction/svd_mean_resp \
        --anchor-svd-dir eval_results/issue_521/svd \
        --figure-dir figures/issue_552
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

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import (
    NullSummary,
    SVDSummary,
    assemble_M,
    cosine,
    row_shuffle_null,
    sign_flip_null,
    svd_summary,
)

logger = logging.getLogger(__name__)

SEEDS = (42, 137, 256)
VARIANTS = ("same", "base", "on_policy")
DEFAULT_ARM = "benign"

# Pre-registered cross-RUN faithfulness tolerance (plan v2 §5/§6): a fresh
# greedy regeneration on same-class hardware with an unchanged lockfile must
# reproduce the persisted #521 end-slot metrics within ±0.02; a wrong adapter /
# wrong variant / version drift moves them by >> 0.02. Distinct from the 5e-4
# same-tensors check below (which covers only LAPACK/numpy build noise).
DEFAULT_ANCHOR_ATOL = 0.02

# Pre-registered concentration zones, mirrored verbatim from
# scripts/issue552_cross_arm_analysis.py (plan §3 / §6.3). "Concentrated" is the
# parent's falsification zone; "diffuse" the confirmation zone.
CONCENTRATED_MEAN_COS_MIN = 0.90
CONCENTRATED_TOP_SHARE_MIN = 0.50
DIFFUSE_MEAN_COS_MAX = 0.85
DIFFUSE_TOP_SHARE_MAX = 0.50

# Fields cross-checked against the persisted per-cell svd JSONs. SVD metrics are
# deterministic; the nulls reuse the pipeline's exact (n_reps=1000, seed=cell
# seed) convention, so any drift beyond LAPACK/numpy build noise is a bug.
CROSS_CHECK_FIELDS = (
    "s_top1_frac",
    "mean_cos_to_U1",
    "sign_flip_p95",
    "sign_flip_p99",
    "row_shuffle_p95",
    "row_shuffle_p99",
)


def _git_commit() -> str:
    """Return the current git commit hash (or 'unknown' if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
    except OSError:
        return "unknown"
    sha = out.stdout.strip()
    return sha if sha else "unknown"


def _read_block(M: np.ndarray, cell_seed: int, n_reps: int) -> dict:
    """SVD + nulls on M (H x N); returns the per-read metrics block.

    Mirrors the Phase-D entry construction in ``issue_519_dispatch.py``:
    both nulls run ``n_reps`` reps seeded with the CELL seed.
    """
    svd: SVDSummary = svd_summary(M)
    row_null: NullSummary = row_shuffle_null(M, n_reps=n_reps, seed=cell_seed)
    sign_null: NullSummary = sign_flip_null(M, n_reps=n_reps, seed=cell_seed)
    return {
        "M_shape": list(svd["M_shape"]),
        "s_top1_frac": svd["s_top1_frac"],
        "mean_cos_to_U1": float(np.mean(svd["cos_to_U1"])),
        "median_cos_to_U1": float(np.median(svd["cos_to_U1"])),
        "sign_flip_p95": sign_null["p95"],
        "sign_flip_p99": sign_null["p99"],
        "row_shuffle_p95": row_null["p95"],
        "row_shuffle_p99": row_null["p99"],
        "cos_to_U1": svd["cos_to_U1"].tolist(),
        "singular_values": svd["s"].tolist(),
        "_U1": svd["U1"],  # stripped before JSON write; used for cos(U1_end, U1_mr)
    }


def _zone(mean_cos: float, top_share: float) -> str:
    """Classify a read against the parent's pre-registered concentration zones."""
    if mean_cos >= CONCENTRATED_MEAN_COS_MIN and top_share >= CONCENTRATED_TOP_SHARE_MIN:
        return "concentrated"
    if mean_cos <= DIFFUSE_MEAN_COS_MAX and top_share < DIFFUSE_TOP_SHARE_MAX:
        return "diffuse"
    return "intermediate"


def _cross_check(recomputed: dict, persisted_path: Path, atol: float) -> dict:
    """Compare the recomputed end-slot block against the persisted svd JSON.

    Raises ``AssertionError`` (fail-loud) when any field differs by more than
    ``atol`` — the recomputation must reproduce the pipeline's numbers.
    """
    with persisted_path.open() as f:
        persisted = json.load(f)
    diffs = {
        field: abs(float(recomputed[field]) - float(persisted[field]))
        for field in CROSS_CHECK_FIELDS
    }
    max_abs_diff = max(diffs.values())
    ok = max_abs_diff <= atol
    assert ok, (
        f"end-slot recomputation diverges from persisted {persisted_path} "
        f"beyond atol={atol}: {diffs}"
    )
    return {
        "persisted_path": str(persisted_path),
        "fields": diffs,
        "max_abs_diff": max_abs_diff,
        "atol": atol,
        "pass": ok,
    }


def _strip_private(block: dict | None) -> dict | None:
    """Drop underscore-prefixed (non-JSON) keys from a read block."""
    if block is None:
        return None
    return {k: v for k, v in block.items() if not k.startswith("_")}


def _anchor_gate(end_block: dict, anchor_path: Path, atol: float) -> dict:
    """Cross-RUN faithfulness gate (plan v2 §6) vs a persisted #521 anchor JSON.

    Compares the recomputed end-slot ``mean_cos_to_U1`` + ``s_top1_frac``
    against the anchor at ``atol`` (default ±0.02). RECORDS pass/fail — never
    raises on FAIL (a failed gate halts interpretation downstream and is
    itself the infrastructure finding); raises only on a missing anchor file.
    """
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"--anchor-svd-dir was passed but the anchor JSON is missing: {anchor_path}"
        )
    anchor = json.loads(anchor_path.read_text())
    diffs = {
        field: abs(float(end_block[field]) - float(anchor[field]))
        for field in ("mean_cos_to_U1", "s_top1_frac")
    }
    ok = max(diffs.values()) <= atol
    if not ok:
        logger.warning(
            "cross-run anchor gate FAIL vs %s: %s (atol=%s) — re-extraction not "
            "faithful; interpretation of the mean-resp read is halted downstream",
            anchor_path,
            diffs,
            atol,
        )
    return {
        "anchor_path": str(anchor_path),
        "fields": diffs,
        "max_abs_diff": max(diffs.values()),
        "atol": atol,
        "pass": ok,
    }


def _analyze_cell(
    variant: str,
    seed: int,
    arm: str,
    shifts_dir: Path,
    svd_dir: Path,
    n_reps: int,
    atol: float,
    anchor_svd_dir: Path | None = None,
    anchor_atol: float = DEFAULT_ANCHOR_ATOL,
) -> dict:
    """Run both SVD reads (end-slot + mean-resp where present) for one cell."""
    cell = f"{variant}_{arm}_seed{seed}"
    shift_path = shifts_dir / f"{cell}.pt"
    payload = torch.load(shift_path, map_location="cpu", weights_only=False)
    shifts = payload["shifts"]

    # --- End-slot read (the headline construction), recomputed as a self-check.
    M_end, persona_order = assemble_M(shifts)
    assert M_end.shape[1] == len(persona_order), M_end.shape
    end_block = _read_block(M_end, cell_seed=seed, n_reps=n_reps)
    cross_check = _cross_check(end_block, svd_dir / f"{cell}.json", atol=atol)
    anchor_gate = (
        _anchor_gate(end_block, anchor_svd_dir / f"{cell}.json", anchor_atol)
        if anchor_svd_dir is not None
        else None
    )

    # --- Mean-over-response read: persisted only for variant == "same" (the
    # extraction's also_compute_mean_over_response_em gate); absence elsewhere
    # is by design, never silently substituted.
    have_mean_resp = all("delta_v_mean_resp" in shifts[p] for p in persona_order)
    if have_mean_resp:
        M_mr, persona_order_mr = assemble_M(shifts, use_mean_resp=True)
        assert persona_order_mr == persona_order, (persona_order_mr, persona_order)
        assert M_mr.shape == M_end.shape, (M_mr.shape, M_end.shape)
        mr_block = _read_block(M_mr, cell_seed=seed, n_reps=n_reps)
        deltas = {
            "mean_cos_to_U1": mr_block["mean_cos_to_U1"] - end_block["mean_cos_to_U1"],
            "s_top1_frac": mr_block["s_top1_frac"] - end_block["s_top1_frac"],
        }
        cos_u1 = cosine(end_block["_U1"], mr_block["_U1"])
        zone_end = _zone(end_block["mean_cos_to_U1"], end_block["s_top1_frac"])
        zone_mr = _zone(mr_block["mean_cos_to_U1"], mr_block["s_top1_frac"])
        agree = zone_end == zone_mr
        concordance = (
            f"{'CONCORDANT' if agree else 'DIVERGENT'} — end-slot {zone_end} vs "
            f"mean-resp {zone_mr}: mean |cos to U1| "
            f"{end_block['mean_cos_to_U1']:.3f} vs {mr_block['mean_cos_to_U1']:.3f}, "
            f"top-share {end_block['s_top1_frac']:.3f} vs {mr_block['s_top1_frac']:.3f}, "
            f"cos(U1_end, U1_mean_resp) = {cos_u1:.3f}"
        )
        absent_reason = None
    else:
        mr_block = None
        deltas = None
        cos_u1 = None
        concordance = "N/A — mean-over-response read not persisted for this cell"
        absent_reason = (
            "delta_v_mean_resp is computed only for variant='same' "
            "(activation_shift.extract_per_context_shifts, "
            "also_compute_mean_over_response_em gate) — absent by design, "
            "not data loss"
        )

    return {
        "cell": cell,
        "variant": variant,
        "arm": arm,
        "seed": seed,
        "persona_order": persona_order,
        "end_slot": _strip_private(end_block),
        "end_slot_cross_check": cross_check,
        "cross_run_anchor_gate": anchor_gate,
        "mean_resp": _strip_private(mr_block),
        "mean_resp_absent_reason": absent_reason,
        "delta_mean_resp_minus_end_slot": deltas,
        "cos_U1_end_vs_U1_mean_resp": cos_u1,
        "concordance": concordance,
    }


ARM_FIGURE_LABEL = {
    "benign": "Benign arm",
    "em": "Misalignment (EM) arm",
    "marker": "Marker arm",
}


def _make_figure(
    per_cell: dict[str, dict], figure_dir: Path, arm: str = DEFAULT_ARM
) -> dict[str, Path] | None:
    """Paired-dot concordance figure over the cells that carry both reads."""
    cells = {k: v for k, v in per_cell.items() if v["mean_resp"] is not None}
    if not cells:
        logger.warning("no cells with both reads — skipping figure")
        return None

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    c_end = paper_palette_role("primary")
    c_mr = paper_palette_role("accent")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    names = list(cells.keys())
    xs = np.arange(len(names))
    labels = [f"seed {cells[n]['seed']}" for n in names]

    panels = (
        ("mean_cos_to_U1", "Mean |cos(shift, top direction)|", axes[0]),
        ("s_top1_frac", "Top singular-value share", axes[1]),
    )
    for field, ylabel, ax in panels:
        end_vals = [cells[n]["end_slot"][field] for n in names]
        mr_vals = [cells[n]["mean_resp"][field] for n in names]
        for x, e, m in zip(xs, end_vals, mr_vals, strict=True):
            ax.plot([x, x], [e, m], color="0.6", lw=1.0, zorder=1)
        ax.scatter(xs, end_vals, color=c_end, s=45, zorder=2, label="End-of-response slot")
        ax.scatter(xs, mr_vals, color=c_mr, s=45, zorder=2, label="Mean over response")
        if field == "s_top1_frac":
            null_p95 = [cells[n]["mean_resp"]["row_shuffle_p95"] for n in names]
            ax.scatter(
                xs,
                null_p95,
                color="0.4",
                marker="_",
                s=120,
                zorder=2,
                label="Row-shuffle null (95th pct)",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
    # Figure-level legend below the panels, outside the data region: an
    # in-axes lower-right legend occluded the marker arm's low row-shuffle
    # null dashes (~0.19-0.20), unlike the benign/EM arms whose nulls sit
    # high (task #552 interp-critique round 1 on the marker fold-in).
    legend_handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"{ARM_FIGURE_LABEL.get(arm, arm)}, same-response variant: "
        "shift-geometry concentration,\n"
        "end-of-response-slot read vs mean-over-response read",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    # Benign keeps the original stem (back-compat with the committed figure);
    # other arms get an arm-suffixed stem so the benign figure is never clobbered.
    stem = "mean_resp_concordance" if arm == DEFAULT_ARM else f"mean_resp_concordance_{arm}"
    written = savefig_paper(fig, stem, dir=figure_dir)
    plt.close(fig)
    return written


def main() -> None:
    """CLI entrypoint: run both SVD reads over the selected cells, write JSONs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shifts-dir", default="eval_results/issue_552/shifts")
    parser.add_argument(
        "--svd-dir",
        default="eval_results/issue_552/svd",
        help="Persisted per-cell end-slot SVD JSONs (cross-check reference; for the "
        "EM re-extraction this is the FRESH Phase-D dir — same-tensors semantics).",
    )
    parser.add_argument(
        "--arm",
        choices=["benign", "em", "marker"],
        default=DEFAULT_ARM,
        help="Arm whose cells to read (filename template {variant}_{arm}_seed{S}). "
        "`marker` added for the plan-v3 marker-arm re-extraction follow-up.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="Variants to analyze (the EM re-extraction passes `same` only — "
        "base/on_policy were never extracted there).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Cell seeds to analyze (subset-able for smoke runs).",
    )
    parser.add_argument(
        "--anchor-svd-dir",
        default=None,
        help="Optional #521 persisted end-slot SVD dir — records the pre-registered "
        f"±{DEFAULT_ANCHOR_ATOL} CROSS-RUN faithfulness gate per cell (plan v2 §6). "
        "Distinct from --cross-check-atol (same-tensors).",
    )
    parser.add_argument("--anchor-atol", type=float, default=DEFAULT_ANCHOR_ATOL)
    parser.add_argument("--out", default="eval_results/issue_552/svd_mean_resp")
    parser.add_argument("--figure-dir", default="figures/issue_552")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument(
        "--cross-check-atol",
        type=float,
        default=5e-4,
        help="Max abs diff tolerated between recomputed and persisted end-slot metrics "
        "(covers LAPACK build noise; anything larger fails loud).",
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    shifts_dir = Path(args.shifts_dir)
    svd_dir = Path(args.svd_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_svd_dir = Path(args.anchor_svd_dir) if args.anchor_svd_dir else None

    per_cell: dict[str, dict] = {}
    for variant in args.variants:
        for seed in args.seeds:
            entry = _analyze_cell(
                variant,
                seed,
                arm=args.arm,
                shifts_dir=shifts_dir,
                svd_dir=svd_dir,
                n_reps=args.n_reps,
                atol=args.cross_check_atol,
                anchor_svd_dir=anchor_svd_dir,
                anchor_atol=args.anchor_atol,
            )
            per_cell[entry["cell"]] = entry
            # Checkpoint-per-phase: persist each cell the moment it completes.
            cell_path = out_dir / f"{entry['cell']}.json"
            with cell_path.open("w") as f:
                json.dump(entry, f, indent=2)
            logger.info("[%s] %s", entry["cell"], entry["concordance"])

    cross_run_anchor_gate = None
    if anchor_svd_dir is not None:
        gates = {k: v["cross_run_anchor_gate"] for k, v in per_cell.items()}
        cross_run_anchor_gate = {
            "atol": args.anchor_atol,
            "anchor_svd_dir": str(anchor_svd_dir),
            "pass": all(g["pass"] for g in gates.values()),
            "per_cell": gates,
        }
        logger.info(
            "[anchor-gate] overall %s (atol=%s)",
            "PASS" if cross_run_anchor_gate["pass"] else "FAIL",
            args.anchor_atol,
        )

    figure_paths = None
    if not args.no_figure:
        written = _make_figure(per_cell, Path(args.figure_dir), arm=args.arm)
        if written is not None:
            figure_paths = {k: str(v) for k, v in written.items()}

    summary = {
        "issue": 552,
        "followup": "mean_resp_svd",
        "arm": args.arm,
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "description": (
            "Identical SVD concentration read on the persisted mean-over-response "
            "shift vectors (delta_v_mean_resp), alongside an end-slot (delta_v) "
            "recomputation cross-checked against the persisted svd/*.json."
        ),
        "coverage_note": (
            f"delta_v_mean_resp exists only for same-variant {args.arm} cells (the "
            "headline variant); base/on_policy variants never carried it — the "
            "extraction computes mean-over-response only for variant='same' by design."
        ),
        "cross_run_anchor_gate": cross_run_anchor_gate,
        "n_reps": args.n_reps,
        "null_seed_convention": "both nulls seeded with the cell seed (Phase-D convention)",
        "zones": {
            "concentrated": (
                f"mean_cos >= {CONCENTRATED_MEAN_COS_MIN} AND "
                f"s_top1_frac >= {CONCENTRATED_TOP_SHARE_MIN}"
            ),
            "diffuse": (
                f"mean_cos <= {DIFFUSE_MEAN_COS_MAX} AND s_top1_frac < {DIFFUSE_TOP_SHARE_MAX}"
            ),
        },
        "per_cell": per_cell,
        "concordance_verdicts": {k: v["concordance"] for k, v in per_cell.items()},
        "figure": figure_paths,
        "metadata": {
            "git_commit": _git_commit(),
            "script": "scripts/issue552_mean_resp_svd.py",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
