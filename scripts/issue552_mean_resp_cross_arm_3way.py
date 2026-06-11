#!/usr/bin/env python3
"""#552 follow-up `marker-arm-mean-resp-reextraction` (OFF-POD, VM) — 3-arm
cross-arm direction identity at the MEAN-OVER-RESPONSE position.

The previous follow-up round found the benign and EM (plain full-response SFT)
arms' TOP shift directions (U1) overlap at |cos| median 0.763 at the
mean-over-response position — far above their end-slot 0.50. This script asks
whether that averaged-position sharing is specific to the plain-SFT arm pair
or is carried by ANY adapter of this recipe family: it adds the marker arm
(marker-only-loss, no full-response imitation — the strongest available
non-SFT control) as a third arm and computes, per the sibling
``issue552_mean_resp_cross_arm.py`` conventions (|cos|,
``RANDOM_COS_FLOOR_P95 = 0.033``, ``assemble_M(use_mean_resp=True)`` +
``svd_summary``):

  1. within-marker (3 seed pairs), marker x benign (9), marker x EM (9), AND
     benign x EM recomputed (9) |cos(U1_mr, U1_mr')| at the mean-over-response
     position. The recomputed benign x EM pairs are an INTERNAL CONSISTENCY
     check: same tensors + same helpers as the previous round, so each pair
     must match that round's persisted summary within ``--consistency-atol``
     (default 1e-6; plan v3 §11 pre-authorizes relaxing to 1e-5 if float32
     round-trip noise trips it — the observed max diff is recorded either way).
  2. the pre-registered VALIDITY GATE (plan v3 §6, binding before any
     interpretation): per-cell |re-extracted marker end-slot mean_cos_to_U1 -
     #521 persisted| <= 0.02 AND |s_top1_frac - persisted| <= 0.02, read from
     the FRESH Phase-D JSONs vs ``eval_results/issue_521/svd/``. The gate
     reads ONLY those two fields — it ignores ``cos_U1_vsteer``, which the
     fresh JSONs lack (no v_marker.pt on the re-extraction pod) while the
     #521 anchors carry real values.
  3. the pre-registered primary decision rule on the TWO cross-arm medians
     med_MB = median |cos(U1_marker_mr, U1_benign_mr)| and med_ME = median
     |cos(U1_marker_mr, U1_EM_mr)| (plan v3 §6: BOTH <= 0.2 -> end-slot
     contrast CONSERVED at mean-resp / EITHER >= 0.6 -> arm-nonspecific
     sharing, demoting the corpus-level reading / between -> graded) —
     recorded ONLY when the validity gate passes. Any CONSERVED narration is
     CONDITIONED on the within-marker mean-resp median sitting clearly above
     the cross-arm medians (the marker arm's own attainable ceiling; without
     that, low cross values could reflect the marker arm lacking a stable
     mean-resp direction rather than a genuine contrast).
  4. the secondary marker concentration read at mean-resp per cell
     (mean cos >= 0.90 AND top-share >= 0.50, plan v1 §3/§6.3 zones); the
     per-cell mean_cos_to_U1 is persisted next to every cross-arm number so
     the read is conditioned on how much of the matrix mass U1 summarizes.

All claims are scoped to TOP-DIRECTION (U1) identity — this script says
nothing about subspace overlap beyond rank 1. Writes the per-pair table +
verdict inputs to ``--out`` and a 6-strip plot (three within-arm ceilings
left, three cross-arm groups right, 0.033 floor dashed — the sibling figure
shape extended). Interpretation stays with the analyzer; this script only
computes and records.

Run (VM, after pod termination + tensor pull)::

    uv run python scripts/issue552_mean_resp_cross_arm_3way.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    svd_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 137, 256)
FU_ROOT = "eval_results/issue_552/marker-arm-mean-resp-reextraction"
EM_FU_ROOT = "eval_results/issue_552/em-arm-mean-resp-reextraction"

# p95 of |cos| between random unit vectors in R^3584 (parent #521 floor;
# verbatim from issue552_cross_arm_analysis.py).
RANDOM_COS_FLOOR_P95 = 0.033

# Pre-registered decision thresholds (plan v3 §6 — no post-hoc motion).
# NOT the sibling's 0.05/0.40: this round's rule is two-sided over two
# medians (conserve <= 0.2 on BOTH / demote >= 0.6 on EITHER), which is why
# this is a sibling script rather than a flag on issue552_mean_resp_cross_arm.
CONSERVED_BOTH_MEDIANS_MAX = 0.2
DEMOTED_EITHER_MEDIAN_MIN = 0.6
FLOOR_LEVEL_MAX = 0.05

# Pre-registered validity-gate tolerance (plan v3 §5/§6): cross-RUN
# reproduction of the #521 marker end-slot numbers. Distinct from the 5e-4
# same-tensors cross-check below.
FAITHFULNESS_ATOL = 0.02

# Same-tensors consistency tolerance: the recomputed end-slot metrics and
# the fresh Phase-D JSONs derive from the SAME tensors via the SAME helpers,
# so any drift beyond LAPACK/numpy build noise means --marker-svd-dir points
# at the wrong (stale) directory. Mirrors issue552_mean_resp_cross_arm.py.
SAME_TENSORS_ATOL = 5e-4

# Internal-consistency tolerance for the recomputed benign x EM pairs vs the
# previous round's persisted summary (same tensors, same code — plan v3 §10
# assumption 9). §11 pre-authorizes relaxing to 1e-5 via --consistency-atol.
DEFAULT_CONSISTENCY_ATOL = 1e-6

# Concentrated-zone thresholds (plan v1 §3/§6.3, reused verbatim for the
# secondary marker mean-resp read).
CONCENTRATED_MEAN_COS_MIN = 0.90
CONCENTRATED_TOP_SHARE_MIN = 0.50


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


def _abs_cos(u: np.ndarray, v: np.ndarray) -> float:
    """|cos| between two vectors (issue552_cross_arm_analysis.py convention)."""
    return float(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def _load_cell_reads(pt_path: Path, arm: str, seed: int) -> dict:
    """Load one same-variant shift tensor; return end-slot + mean-resp SVD reads.

    Fails loud when the tensor is missing or lacks ``delta_v_mean_resp``
    (the re-extraction's whole point); asserts the persona panels match
    between the two reads.
    """
    if not pt_path.exists():
        raise FileNotFoundError(
            f"shift tensor missing: {pt_path}. Benign tensors live in the issue-552 "
            f"worktree (eval_results/issue_552/shifts/; WandB issue552_analysis_"
            f"tensors:v0); EM tensors under {EM_FU_ROOT}/shifts/ (WandB issue552_em_"
            f"mean_resp_tensors:v0); marker tensors come from the plan-v3 pod run "
            f"(WandB issue552_marker_mean_resp_tensors:v0 + VM pull)."
        )
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    shifts = payload["shifts"]
    M_end, order_end = assemble_M(shifts)
    M_mr, order_mr = assemble_M(shifts, use_mean_resp=True)
    assert order_end == order_mr, (order_end, order_mr)
    assert M_end.shape == M_mr.shape, (M_end.shape, M_mr.shape)
    svd_end = svd_summary(M_end)
    svd_mr = svd_summary(M_mr)
    return {
        "arm": arm,
        "seed": seed,
        "pt_path": str(pt_path),
        "persona_order": order_end,
        "n_personas": len(order_end),
        "end_slot": {
            "mean_cos_to_U1": float(np.mean(svd_end["cos_to_U1"])),
            "s_top1_frac": svd_end["s_top1_frac"],
        },
        "mean_resp": {
            "mean_cos_to_U1": float(np.mean(svd_mr["cos_to_U1"])),
            "s_top1_frac": svd_mr["s_top1_frac"],
        },
        "_U1_end": svd_end["U1"],
        "_U1_mr": svd_mr["U1"],
    }


def _faithfulness_gate(
    *,
    marker_cells: dict[int, dict],
    marker_svd_dir: Path,
    anchor_svd_dir: Path,
    seeds: tuple[int, ...],
) -> dict:
    """Plan v3 §6 validity gate: fresh marker end-slot vs #521 persisted, ±0.02.

    Reads the canonical FRESH Phase-D JSONs (``--marker-svd-dir``),
    sanity-checks them against the in-memory recomputation from the tensors
    (same-tensors, 5e-4 — a wrong/stale ``--marker-svd-dir`` fails loud here),
    then compares to the persisted #521 anchors at the pre-registered ±0.02.
    Only ``mean_cos_to_U1`` + ``s_top1_frac`` are compared — ``cos_U1_vsteer``
    (absent from the fresh JSONs, real in the anchors) is ignored by design.
    Gate FAIL is RECORDED, not raised: a failed gate halts interpretation
    downstream (the decision rule is withheld), and the numbers themselves
    are the infrastructure finding.
    """
    per_cell: dict[str, dict] = {}
    all_pass = True
    for seed in seeds:
        cell = f"same_marker_seed{seed}"
        fresh_path = marker_svd_dir / f"{cell}.json"
        anchor_path = anchor_svd_dir / f"{cell}.json"
        if not fresh_path.exists():
            raise FileNotFoundError(
                f"fresh Phase-D JSON missing: {fresh_path} — run the pod driver "
                f"(ARM=marker bash scripts/run_issue552_emresp_followup.sh) first "
                f"and pull {FU_ROOT}/svd/ to the VM."
            )
        if not anchor_path.exists():
            raise FileNotFoundError(f"#521 anchor JSON missing: {anchor_path} (git, main)")
        fresh = json.loads(fresh_path.read_text())
        anchor = json.loads(anchor_path.read_text())

        # Same-tensors consistency: recomputed-from-tensor end slot must match
        # the fresh JSON to numerical noise, else --marker-svd-dir is mis-wired.
        recomputed = marker_cells[seed]["end_slot"]
        for field in ("mean_cos_to_U1", "s_top1_frac"):
            drift = abs(float(recomputed[field]) - float(fresh[field]))
            if drift > SAME_TENSORS_ATOL:
                raise AssertionError(
                    f"{cell}: recomputed end-slot {field} ({recomputed[field]:.6f}) "
                    f"diverges from the fresh Phase-D JSON {fresh_path} "
                    f"({float(fresh[field]):.6f}) by {drift:.2e} > {SAME_TENSORS_ATOL} — "
                    f"--marker-svd-dir and --marker-shifts-dir do not describe the "
                    f"same run."
                )

        d_cos = abs(float(fresh["mean_cos_to_U1"]) - float(anchor["mean_cos_to_U1"]))
        d_share = abs(float(fresh["s_top1_frac"]) - float(anchor["s_top1_frac"]))
        cell_pass = d_cos <= FAITHFULNESS_ATOL and d_share <= FAITHFULNESS_ATOL
        all_pass = all_pass and cell_pass
        per_cell[cell] = {
            "fresh": {
                "mean_cos_to_U1": float(fresh["mean_cos_to_U1"]),
                "s_top1_frac": float(fresh["s_top1_frac"]),
                "path": str(fresh_path),
            },
            "anchor": {
                "mean_cos_to_U1": float(anchor["mean_cos_to_U1"]),
                "s_top1_frac": float(anchor["s_top1_frac"]),
                "path": str(anchor_path),
            },
            "abs_diff": {"mean_cos_to_U1": d_cos, "s_top1_frac": d_share},
            "pass": cell_pass,
        }
        logger.info(
            "[gate] %s: |d mean_cos|=%.4f |d top_share|=%.4f -> %s",
            cell,
            d_cos,
            d_share,
            "PASS" if cell_pass else "FAIL",
        )
    return {
        "atol": FAITHFULNESS_ATOL,
        "rule": (
            "per-cell |fresh marker end-slot mean_cos_to_U1 - #521 persisted| <= 0.02 "
            "AND |s_top1_frac - persisted| <= 0.02, all cells (plan v3 §6, binding "
            "before any interpretation; cos_U1_vsteer is ignored — absent from the "
            "fresh JSONs by design)"
        ),
        "pass": all_pass,
        "per_cell": per_cell,
    }


def _consistency_check(
    recomputed_cross_be: dict[str, float], round1_summary_path: Path, atol: float
) -> dict:
    """Assert the recomputed benign x EM pairs match the previous round's summary.

    Same tensors + same helpers (plan v3 §10 assumption 9) — raises
    ``AssertionError`` (fail-loud) when any pair drifts beyond ``atol``;
    records the observed max diff either way (plan v3 §11 asks for it when
    the pre-authorized 1e-5 relaxation is used).
    """
    if not round1_summary_path.exists():
        raise FileNotFoundError(
            f"previous-round cross-arm summary missing: {round1_summary_path} — "
            f"the benign x EM internal-consistency reference (git, branch issue-552)."
        )
    round1 = json.loads(round1_summary_path.read_text())
    reference = round1["pairs"]["cross_benign_x_em"]
    missing = sorted(set(reference) ^ set(recomputed_cross_be))
    assert not missing, f"benign x EM pair-key mismatch vs {round1_summary_path}: {missing}"
    diffs = {k: abs(recomputed_cross_be[k] - float(reference[k])) for k in reference}
    max_abs_diff = max(diffs.values())
    assert max_abs_diff <= atol, (
        f"recomputed benign x EM pairs diverge from the previous round's summary "
        f"{round1_summary_path} beyond atol={atol} (max abs diff {max_abs_diff:.2e}; "
        f"same tensors + same code should reproduce exactly): "
        f"{ {k: f'{v:.2e}' for k, v in diffs.items() if v > atol} }"
    )
    logger.info(
        "[consistency] benign x EM recomputation matches the previous round "
        "(max abs diff %.2e <= atol %.0e)",
        max_abs_diff,
        atol,
    )
    return {
        "reference_path": str(round1_summary_path),
        "atol": atol,
        "max_abs_diff": max_abs_diff,
        "pass": True,
        "reference_median": float(np.median(list(map(float, reference.values())))),
    }


def _classify(
    *,
    med_mb: float,
    med_me: float,
    within_marker_median: float,
) -> tuple[str, str]:
    """Apply the pre-registered plan v3 §6 primary decision rule.

    Returns (classification, note). Computation + recording only —
    interpretation belongs to the analyzer. Any CONSERVED narration is
    conditioned on the within-marker mean-resp median sitting clearly above
    the cross-arm medians (the marker arm's own attainable ceiling).
    """
    ceiling_above = within_marker_median > max(med_mb, med_me)
    ceiling_note = (
        f"within-marker mean-resp median {within_marker_median:.3f} "
        f"{'sits above' if ceiling_above else 'does NOT sit above'} the cross-arm "
        f"medians (marker x benign {med_mb:.3f}, marker x EM {med_me:.3f})"
    )
    if med_mb <= CONSERVED_BOTH_MEDIANS_MAX and med_me <= CONSERVED_BOTH_MEDIANS_MAX:
        floor_note = (
            " — at/below 0.05 these additionally read as floor-level "
            f"(floor p95 = {RANDOM_COS_FLOOR_P95})"
            if max(med_mb, med_me) <= FLOOR_LEVEL_MAX
            else ""
        )
        if ceiling_above:
            return (
                "conserved_end_slot_contrast",
                f"both cross-arm medians <= {CONSERVED_BOTH_MEDIANS_MAX} and the "
                f"{ceiling_note}, so the contrast is read against a real "
                f"within-marker ceiling{floor_note}",
            )
        return (
            "conserved_end_slot_contrast",
            f"both cross-arm medians <= {CONSERVED_BOTH_MEDIANS_MAX}, BUT the "
            f"{ceiling_note} — the conserved read is CAVEATED: low cross-arm "
            f"values may reflect the marker arm lacking a stable mean-resp "
            f"direction of its own rather than a genuine contrast{floor_note}",
        )
    if med_mb >= DEMOTED_EITHER_MEDIAN_MIN or med_me >= DEMOTED_EITHER_MEDIAN_MIN:
        return (
            "arm_nonspecific_sharing_demotes_corpus_level_reading",
            f"a cross-arm median reached {DEMOTED_EITHER_MEDIAN_MIN} (marker x benign "
            f"{med_mb:.3f}, marker x EM {med_me:.3f}) — a marker-only-loss adapter "
            f"shares the averaged-position top direction too, so the plain-SFT "
            f"sharing is demoted to a recipe-generic component; {ceiling_note}",
        )
    return (
        "graded",
        f"cross-arm medians between {CONSERVED_BOTH_MEDIANS_MAX} and "
        f"{DEMOTED_EITHER_MEDIAN_MIN} (marker x benign {med_mb:.3f}, marker x EM "
        f"{med_me:.3f}) — reported with per-seed structure and read against the "
        f"{ceiling_note}",
    )


def _make_figure(
    *,
    within_groups: dict[str, dict[str, float]],
    cross_groups: dict[str, dict[str, float]],
    gate_pass: bool,
    figure_dir: Path,
) -> dict[str, Path]:
    """6-strip plot: within-arm ceilings left, cross-arm groups right, floor dashed."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    c_benign = paper_palette_role("control")
    c_em = paper_palette_role("primary")
    c_marker = paper_palette_role("baseline")
    c_neutral = paper_palette_role("neutral")
    c_accent = paper_palette_role("accent")

    groups: dict[str, tuple[list[float], str]] = {
        "within\nbenign": (list(within_groups["within_benign"].values()), c_benign),
        "within\nmisalignment": (list(within_groups["within_em"].values()), c_em),
        "within\nmarker": (list(within_groups["within_marker"].values()), c_marker),
        "benign x\nmisalignment": (list(cross_groups["cross_benign_x_em"].values()), c_neutral),
        "marker x\nbenign": (list(cross_groups["cross_marker_x_benign"].values()), c_benign),
        "marker x\nmisalignment": (list(cross_groups["cross_marker_x_em"].values()), c_em),
    }

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.subplots_adjust(bottom=0.16, top=0.86, left=0.09, right=0.97)
    rng = np.random.default_rng(1)
    for x, (vals, color) in enumerate(groups.values()):
        jit = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(x + jit, vals, s=30, color=color, alpha=0.8, zorder=3)
        ax.scatter(x, float(np.median(vals)), marker="_", s=420, color="black", zorder=5)
    ax.axvline(2.5, color="0.85", linewidth=1.0, zorder=1)
    ax.text(
        1.0,
        1.02,
        "same arm, different seed",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.text(
        4.0,
        1.02,
        "different arms",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=8,
        color="0.45",
    )
    ax.axhline(
        RANDOM_COS_FLOOR_P95,
        color=c_accent,
        linestyle="--",
        linewidth=1.2,
        label="random-direction floor (p95 = 0.033)",
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups.keys()), fontsize=8)
    ax.set_ylabel("|cos(top direction, top direction')|")
    ax.set_ylim(0, 1.05)
    title = "Mean-over-response read: does the marker arm share the plain-SFT top direction?"
    if not gate_pass:
        title += "\n(VALIDITY GATE FAILED — re-extraction not faithful; do not interpret)"
    ax.set_title(title, pad=18)
    ax.legend(frameon=False, fontsize=8)
    written = savefig_paper(fig, "cross_arm_mean_resp_directions_3arm", dir=figure_dir)
    plt.close(fig)
    return written


def main() -> int:
    """CLI entrypoint: 3-arm |cos| at the mean-resp position + validity gate."""
    parser = argparse.ArgumentParser(
        description="#552 3-arm cross-arm direction identity at the mean-over-response position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benign-shifts-dir",
        default="eval_results/issue_552/shifts",
        help="Benign same-variant shift tensors (this issue's completed run).",
    )
    parser.add_argument(
        "--em-shifts-dir",
        default=f"{EM_FU_ROOT}/shifts",
        help="Re-extracted EM same-variant shift tensors (the plan-v2 follow-up).",
    )
    parser.add_argument(
        "--marker-shifts-dir",
        default=f"{FU_ROOT}/shifts",
        help="Re-extracted marker same-variant shift tensors (the plan-v3 pod run).",
    )
    parser.add_argument(
        "--marker-svd-dir",
        default=f"{FU_ROOT}/svd",
        help="FRESH Phase-D per-cell SVD JSONs from the marker re-extraction (gate input).",
    )
    parser.add_argument(
        "--anchor-svd-dir",
        default="eval_results/issue_521/svd",
        help="#521 persisted end-slot SVD JSONs (the cross-run faithfulness anchor).",
    )
    parser.add_argument(
        "--round1-summary",
        default=f"{EM_FU_ROOT}/cross_arm_mean_resp/summary.json",
        help="Previous round's cross-arm summary (benign x EM internal-consistency reference).",
    )
    parser.add_argument(
        "--consistency-atol",
        type=float,
        default=DEFAULT_CONSISTENCY_ATOL,
        help="Max abs diff tolerated between the recomputed benign x EM pairs and "
        "the previous round's summary (plan v3 §11 pre-authorizes 1e-5 if float32 "
        "round-trip noise trips the 1e-6 default; the observed max diff is recorded).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", default=f"{FU_ROOT}/cross_arm_mean_resp_3arm/summary.json")
    parser.add_argument("--figure-dir", default="figures/issue_552")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seeds = tuple(args.seeds)
    benign_dir = Path(args.benign_shifts_dir)
    em_dir = Path(args.em_shifts_dir)
    marker_dir = Path(args.marker_shifts_dir)

    # ── Load cells (same-trajectory variant only — the headline variant). ─
    benign_cells = {
        s: _load_cell_reads(benign_dir / f"same_benign_seed{s}.pt", "benign", s) for s in seeds
    }
    em_cells = {s: _load_cell_reads(em_dir / f"same_em_seed{s}.pt", "em", s) for s in seeds}
    marker_cells = {
        s: _load_cell_reads(marker_dir / f"same_marker_seed{s}.pt", "marker", s) for s in seeds
    }

    # ── Validity gate (binding BEFORE any interpretation, plan v3 §6). ────
    gate = _faithfulness_gate(
        marker_cells=marker_cells,
        marker_svd_dir=Path(args.marker_svd_dir),
        anchor_svd_dir=Path(args.anchor_svd_dir),
        seeds=seeds,
    )

    # ── Pairwise |cos| at the mean-resp position. ─────────────────────────
    u1_b = {s: benign_cells[s]["_U1_mr"] for s in seeds}
    u1_e = {s: em_cells[s]["_U1_mr"] for s in seeds}
    u1_m = {s: marker_cells[s]["_U1_mr"] for s in seeds}
    within_benign = {
        f"benign_seed{a}_x_benign_seed{b}": _abs_cos(u1_b[a], u1_b[b])
        for a, b in combinations(seeds, 2)
    }
    within_em = {
        f"em_seed{a}_x_em_seed{b}": _abs_cos(u1_e[a], u1_e[b]) for a, b in combinations(seeds, 2)
    }
    within_marker = {
        f"marker_seed{a}_x_marker_seed{b}": _abs_cos(u1_m[a], u1_m[b])
        for a, b in combinations(seeds, 2)
    }
    cross_marker_x_benign = {
        f"marker_seed{a}_x_benign_seed{b}": _abs_cos(u1_m[a], u1_b[b]) for a in seeds for b in seeds
    }
    cross_marker_x_em = {
        f"marker_seed{a}_x_em_seed{b}": _abs_cos(u1_m[a], u1_e[b]) for a in seeds for b in seeds
    }
    # Recomputed with the PREVIOUS round's key convention so the
    # internal-consistency check can compare pair-by-pair.
    cross_benign_x_em = {
        f"benign_seed{a}_x_em_seed{b}": _abs_cos(u1_b[a], u1_e[b]) for a in seeds for b in seeds
    }

    # ── Internal consistency: same tensors + same code as the prior round. ─
    consistency = _consistency_check(
        cross_benign_x_em, Path(args.round1_summary), atol=args.consistency_atol
    )

    medians = {
        "within_benign": float(np.median(list(within_benign.values()))),
        "within_em": float(np.median(list(within_em.values()))),
        "within_marker": float(np.median(list(within_marker.values()))),
        "cross_marker_x_benign": float(np.median(list(cross_marker_x_benign.values()))),
        "cross_marker_x_em": float(np.median(list(cross_marker_x_em.values()))),
        "cross_benign_x_em": float(np.median(list(cross_benign_x_em.values()))),
    }

    # ── Decision rule — applied ONLY when the gate passed. ────────────────
    if gate["pass"]:
        classification, rule_note = _classify(
            med_mb=medians["cross_marker_x_benign"],
            med_me=medians["cross_marker_x_em"],
            within_marker_median=medians["within_marker"],
        )
    else:
        classification = "gate_failed_interpretation_halted"
        rule_note = (
            "validity gate FAILED — the re-extraction does not reproduce the #521 "
            "marker end-slot numbers within ±0.02; per plan v3 §6 the mean-resp "
            "numbers are NOT read against the decision rule (infrastructure "
            "finding; diagnose adapter identity / marker-stripping path / lockfile)"
        )
    logger.info("[verdict] %s — %s", classification, rule_note)

    # ── Secondary read: marker concentrated zone at mean-resp, per cell. ──
    secondary = {}
    for s in seeds:
        mr = marker_cells[s]["mean_resp"]
        secondary[f"same_marker_seed{s}"] = {
            "mean_cos_to_U1": mr["mean_cos_to_U1"],
            "s_top1_frac": mr["s_top1_frac"],
            "concentrated": bool(
                mr["mean_cos_to_U1"] >= CONCENTRATED_MEAN_COS_MIN
                and mr["s_top1_frac"] >= CONCENTRATED_TOP_SHARE_MIN
            ),
        }

    figure_paths = None
    if not args.no_figure:
        written = _make_figure(
            within_groups={
                "within_benign": within_benign,
                "within_em": within_em,
                "within_marker": within_marker,
            },
            cross_groups={
                "cross_benign_x_em": cross_benign_x_em,
                "cross_marker_x_benign": cross_marker_x_benign,
                "cross_marker_x_em": cross_marker_x_em,
            },
            gate_pass=bool(gate["pass"]),
            figure_dir=Path(args.figure_dir),
        )
        figure_paths = {k: str(v) for k, v in written.items()}

    def _strip(cells: dict[int, dict]) -> dict[str, dict]:
        return {
            f"same_{v['arm']}_seed{s}": {k: x for k, x in v.items() if not k.startswith("_")}
            for s, v in cells.items()
        }

    summary = {
        "issue": 552,
        "followup": "marker-arm-mean-resp-reextraction",
        "position": "mean_over_response",
        "variant": "same",
        "claims_scope": (
            "top-direction (U1) identity only — nothing here speaks to subspace "
            "overlap beyond rank 1"
        ),
        "random_cos_floor_p95": RANDOM_COS_FLOOR_P95,
        "validity_gate": gate,
        "benign_x_em_internal_consistency": consistency,
        "pairs": {
            "within_benign": within_benign,
            "within_em": within_em,
            "within_marker": within_marker,
            "cross_marker_x_benign": cross_marker_x_benign,
            "cross_marker_x_em": cross_marker_x_em,
            "cross_benign_x_em": cross_benign_x_em,
        },
        "medians": medians,
        "decision_rule": {
            "conserved": (
                f"BOTH cross-arm medians (marker x benign, marker x EM) <= "
                f"{CONSERVED_BOTH_MEDIANS_MAX}; conserved narration conditioned on "
                f"the within-marker mean-resp median sitting above the cross medians"
            ),
            "demoted": f"EITHER cross-arm median >= {DEMOTED_EITHER_MEDIAN_MIN}",
            "between": "graded, with per-seed structure vs the within-marker ceiling",
            "applied": bool(gate["pass"]),
            "classification": classification,
            "note": rule_note,
        },
        "secondary_marker_concentrated_zone": secondary,
        "per_cell": {**_strip(benign_cells), **_strip(em_cells), **_strip(marker_cells)},
        "figure": figure_paths,
        "metadata": {
            "git_commit": _git_commit(),
            "script": "scripts/issue552_mean_resp_cross_arm_3way.py",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "[phase=done] wrote %s (gate_pass=%s, classification=%s, "
        "med_marker_x_benign=%.3f, med_marker_x_em=%.3f, within_marker=%.3f)",
        out_path,
        gate["pass"],
        classification,
        medians["cross_marker_x_benign"],
        medians["cross_marker_x_em"],
        medians["within_marker"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
