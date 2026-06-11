#!/usr/bin/env python3
"""#552 contrastive-2x2-completion — VM 5-arm analysis (plan v5 §4.1 new-file 5).

Runs OFF-POD (Phase 13) against git JSONs + VM-pulled tensors. Produces, in
this BINDING order (plan §4.1.5 / §6.3 MF-C — reference bands BEFORE any
new-arm subpanel value is computed):

1. ``subpanel/reference_bands.json`` — held-out-9-subpanel SVD + re-run dual
   nulls (1,000 reps at 9 rows) for the NINE REFERENCE cells (6 plain =
   concentrated calibration band, 3 marker = dispersed calibration band),
   bands = [min, max] per metric (mean cos, top-share).
2. ``subpanel/per_cell.json`` — the same 9-row read for the 6 NEW cells +
   rubric classification (stays-concentrated / disperses / intermediate /
   at-noise-floor; pre-registered degraded mode on band overlap).
3. ``cross_arm_5way/summary.json`` — pairwise |cos(U1, U1')| medians over the
   9 seed pairs per arm pair (end-slot AND mean-resp) + within-arm seed-pair
   medians (the registered reliability ceiling) vs the 0.033 floor.
4. ``contrastive_2x2_summary.json`` — full-panel zone calls for the 6 new
   cells ONLY (plan §3/§6.3: de-concentration = 3/3 seeds + validity
   precondition [sign-flip-null clearance + median per-persona split-half
   cosine >= 0.5 from the per-question tensors]; concentrated = >=2/3 seeds),
   per-cell ||M||_F, split-half disattenuation, the delivered-contrast
   diagnostic read (MF-A, 0.05 nat/token cut), the null-scale cross-check,
   and the free same-vs-base trajectory-variant attribution check.
5. Figures (hero + subpanel-paired + CE bars) -> ``figures/issue_552/``.

Zone metrics are IDENTITY-PINNED to the issue's Phase-D convention:
``mean_cos_to_U1`` is the SIGNED mean of per-persona cos(Delta-v, U1) with
U1 oriented so the mean column projects nonnegatively (the quantity every
reference threshold was registered on); the mean of |cos| is reported
alongside, never substituted.

Usage::

    uv run python scripts/issue552_contrastive_2x2_analysis.py
    # smoke: --fu <fixture-dir> --n-reps 50 --no-figures
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.svd_direction_constancy import (  # noqa: E402
    assemble_M,
    cosine,
    row_shuffle_null,
    sign_flip_null,
    svd_summary,
)

logger = logging.getLogger(__name__)

NEW_ARMS = ("contrastive_em", "contrastive_benign")
PLAIN_ARMS = ("em", "benign")
MARKER_ARM = "marker"
ALL_ARMS = ("marker", "em", "benign", "contrastive_em", "contrastive_benign")
SEEDS = (42, 137, 256)
CROSS_ARM_FLOOR = 0.033  # the issue's registered random-direction floor

# Pre-registered full-panel zones (plan v5 §3; 14-row only — NEVER applied at 9 rows).
ZONE_CONC_COS = 0.90
ZONE_CONC_SHARE = 0.50
ZONE_DECONC_COS = 0.85  # de-concentrated: mean cos <= 0.85 AND top-share < 0.50
SPLIT_HALF_MIN = 0.5
CE_DELIVERED_CUT = 0.05  # nat/token (MF-A)
NULL_SCALE_LO, NULL_SCALE_HI = 0.08, 0.14  # reference sign-flip p95 scale (plan §8)

# Gradient-touched probe personas (source + the #519 negative panel).
GRADIENT_TOUCHED = (
    "medical_doctor",
    "assistant",
    "comedian",
    "police_officer",
    "software_engineer",
)

# Reference same-variant tensor roots (verified on the VM worktree, plan §6.2).
DEFAULT_REFERENCE_SHIFT_ROOTS = {
    "benign": "eval_results/issue_552/shifts",
    "em": "eval_results/issue_552/em-arm-mean-resp-reextraction/shifts",
    "marker": "eval_results/issue_552/marker-arm-mean-resp-reextraction/shifts",
}
DEFAULT_FU = "eval_results/issue_552/contrastive-2x2-completion"


def _load_shifts(pt_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    if not pt_path.exists():
        raise FileNotFoundError(f"shift tensor missing: {pt_path}")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    return payload["shifts"]


def _shift_path(arm: str, seed: int, fu: Path, reference_roots: dict[str, str]) -> Path:
    if arm in NEW_ARMS:
        return fu / "shifts" / f"same_{arm}_seed{seed}.pt"
    return PROJECT_ROOT / reference_roots[arm] / f"same_{arm}_seed{seed}.pt"


def _subpanel_read(
    shifts: dict[str, dict[str, torch.Tensor]],
    held_out: list[str],
    n_reps: int,
    seed: int,
) -> dict:
    """9-row held-out-subpanel SVD + re-run dual nulls (row-count-matched)."""
    M, order = assemble_M(shifts, persona_order=held_out)
    svd = svd_summary(M)
    row_null = row_shuffle_null(M, n_reps=n_reps, seed=seed)
    sign_null = sign_flip_null(M, n_reps=n_reps, seed=seed)
    return {
        "persona_order": order,
        "M_shape": list(svd["M_shape"]),
        "mean_cos_to_U1": float(np.mean(svd["cos_to_U1"])),
        "mean_abs_cos_to_U1": float(np.mean(np.abs(svd["cos_to_U1"]))),
        "s_top1_frac": float(svd["s_top1_frac"]),
        "row_shuffle_p95": float(row_null["p95"]),
        "sign_flip_p95": float(sign_null["p95"]),
        "frob_norm": float(np.linalg.norm(M)),
    }


def _split_half_cosines(shifts: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    """Per-persona split-half cosine from the per-question tensors (MF-B input).

    FIXED halves (registered): first ceil(n/2) questions vs the rest, in the
    tensor's stored (kept-question) order. cos between the two half-mean
    Delta-v's. Requires `delta_v_per_question` (the --save-per-question key).
    """
    out: dict[str, float] = {}
    for p_name, entry in shifts.items():
        if "delta_v_per_question" not in entry:
            raise KeyError(
                f"persona {p_name!r}: delta_v_per_question missing — the cell was "
                f"extracted without --save-per-question; the MF-B validity "
                f"precondition cannot be computed"
            )
        pq = entry["delta_v_per_question"].detach().float().numpy()
        assert pq.ndim == 2 and pq.shape[0] >= 2, (p_name, pq.shape)
        half = (pq.shape[0] + 1) // 2
        a, b = pq[:half].mean(axis=0), pq[half:].mean(axis=0)
        out[p_name] = cosine(a, b)
    return out


def _full_panel_read(shifts: dict[str, dict[str, torch.Tensor]], n_reps: int, seed: int) -> dict:
    """14-row full-panel SVD + nulls + ||M||_F (recomputed from tensors)."""
    M, order = assemble_M(shifts)
    svd = svd_summary(M)
    sign_null = sign_flip_null(M, n_reps=n_reps, seed=seed)
    row_null = row_shuffle_null(M, n_reps=n_reps, seed=seed)
    return {
        "persona_order": order,
        "mean_cos_to_U1": float(np.mean(svd["cos_to_U1"])),
        "mean_abs_cos_to_U1": float(np.mean(np.abs(svd["cos_to_U1"]))),
        "s_top1_frac": float(svd["s_top1_frac"]),
        "sign_flip_p95": float(sign_null["p95"]),
        "row_shuffle_p95": float(row_null["p95"]),
        "frob_norm": float(np.linalg.norm(M)),
        "cos_to_U1": {p: float(c) for p, c in zip(order, svd["cos_to_U1"], strict=True)},
        "U1": svd["U1"],
    }


def _u1(shifts: dict[str, dict[str, torch.Tensor]], use_mean_resp: bool) -> np.ndarray:
    M, _ = assemble_M(shifts, use_mean_resp=use_mean_resp)
    return svd_summary(M)["U1"]


def _pairwise_medians(u1_by_cell: dict[tuple[str, int], np.ndarray]) -> dict:
    """Within-arm (ceiling) + cross-arm |cos(U1, U1')| medians."""
    within: dict[str, dict] = {}
    for arm in ALL_ARMS:
        vals = [
            abs(cosine(u1_by_cell[(arm, s1)], u1_by_cell[(arm, s2)]))
            for i, s1 in enumerate(SEEDS)
            for s2 in SEEDS[i + 1 :]
        ]
        within[arm] = {"pairs": vals, "median": statistics.median(vals)}
    cross: dict[str, dict] = {}
    for i, a in enumerate(ALL_ARMS):
        for b in ALL_ARMS[i + 1 :]:
            per_pair = {
                f"seed{s1}__x__seed{s2}": abs(cosine(u1_by_cell[(a, s1)], u1_by_cell[(b, s2)]))
                for s1 in SEEDS
                for s2 in SEEDS
            }
            vals = list(per_pair.values())
            cross[f"{a}__x__{b}"] = {
                "n_pairs": len(vals),
                "median": statistics.median(vals),
                "pairs": per_pair,
            }
    return {"within_arm_reliability_ceiling": within, "cross_arm": cross}


def _zone_of(mean_cos: float, top_share: float) -> str:
    if mean_cos >= ZONE_CONC_COS and top_share >= ZONE_CONC_SHARE:
        return "concentrated"
    if mean_cos <= ZONE_DECONC_COS and top_share < ZONE_CONC_SHARE:
        return "de-concentrated"
    return "between-zones"


def _band(values: list[float]) -> dict:
    return {"min": min(values), "max": max(values)}


def _classify_subpanel(cell: dict, bands: dict) -> str:
    """Pre-registered 9-row rubric (plan §6.3 MF-C)."""
    if cell["s_top1_frac"] <= cell["sign_flip_p95"]:
        return "at-noise-floor"
    metrics = {"mean_cos_to_U1": cell["mean_cos_to_U1"], "s_top1_frac": cell["s_top1_frac"]}
    usable = {m: v for m, v in metrics.items() if not bands["degraded_metrics"].get(m, False)}
    if not usable:
        return "descriptive-only"
    conc = all(v >= bands["concentrated"][m]["min"] for m, v in usable.items())
    disp = all(v <= bands["dispersed"][m]["max"] for m, v in usable.items())
    if conc and not disp:
        return "stays-concentrated"
    if disp and not conc:
        return "disperses"
    return "intermediate"


def _repro_metadata() -> dict:
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    return {
        "script": "issue552_contrastive_2x2_analysis",
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _make_figures(
    figure_dir: Path,
    full_panel: dict[str, dict],
    subpanel: dict[str, dict],
    rowtype_ce: dict[str, dict],
) -> list[str]:
    """Hero (per-persona cos dots + top-share bars), full-vs-9 paired dots, CE bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    arm_order = [a for a in ALL_ARMS for s in SEEDS if f"{a}_seed{s}" in full_panel]
    cells = [f"{a}_seed{s}" for a in ALL_ARMS for s in SEEDS if f"{a}_seed{s}" in full_panel]
    del arm_order

    # Hero: per-persona cos dots + top-share bars.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    xs = np.arange(len(cells))
    for x, cell in zip(xs, cells, strict=True):
        d = full_panel[cell]
        for p, c in d["cos_to_U1"].items():
            marker = "D" if p == "medical_doctor" else "o"
            face = "none" if p == "medical_doctor" else None
            axes[0].scatter(
                [x],
                [c],
                marker=marker,
                s=42 if p == "medical_doctor" else 18,
                facecolors=face,
                edgecolors="tab:blue",
                color="tab:blue",
                alpha=0.65,
            )
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("per-persona cos(Δv, U₁)")
    shares = [full_panel[c]["s_top1_frac"] for c in cells]
    axes[1].bar(xs, shares, color="tab:orange", alpha=0.8)
    for x, cell in zip(xs, cells, strict=True):
        d = full_panel[cell]
        axes[1].plot([x - 0.4, x + 0.4], [d["sign_flip_p95"]] * 2, color="k", ls="--", lw=1)
        axes[1].plot([x - 0.4, x + 0.4], [d["row_shuffle_p95"]] * 2, color="gray", ls=":", lw=1)
    axes[1].set_ylabel("top-share σ₁/Σσ")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(cells, rotation=60, ha="right", fontsize=7)
    fig.suptitle(
        "5-arm layer-14 shift-geometry (same-trajectory, end-slot): "
        "per-persona cos + top-share vs dual nulls"
    )
    fig.tight_layout()
    p = figure_dir / "contrastive_2x2_hero.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # Full-panel vs held-out-9 paired dots.
    fig, ax = plt.subplots(figsize=(12, 5))
    for x, cell in zip(xs, cells, strict=True):
        ax.scatter(
            [x - 0.12],
            [full_panel[cell]["s_top1_frac"]],
            color="tab:blue",
            s=30,
            label="full 14-row" if x == 0 else None,
        )
        if cell in subpanel:
            ax.scatter(
                [x + 0.12],
                [subpanel[cell]["s_top1_frac"]],
                color="tab:red",
                s=30,
                label="held-out 9-row" if x == 0 else None,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(cells, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("top-share σ₁/Σσ")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    ax.set_title(
        "Full-panel vs held-out-9 concentration (gradient-touched-persona attribution control)"
    )
    fig.tight_layout()
    p = figure_dir / "contrastive_2x2_full_vs_heldout9.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # Per-row-type delta-CE bars (MF-A).
    trained = {k: v for k, v in rowtype_ce.items() if k != "rowtype_ce_base"}
    if trained:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        names = sorted(trained)
        xs2 = np.arange(len(names))
        d_pos = [trained[n].get("delta_ce_pos_vs_base", np.nan) for n in names]
        d_neg = [trained[n].get("delta_ce_neg_vs_base", np.nan) for n in names]
        ax.bar(xs2 - 0.18, d_pos, width=0.34, label="Δ CE positives (base - trained)")
        ax.bar(xs2 + 0.18, d_neg, width=0.34, label="Δ CE negatives (base - trained)")
        ax.axhline(
            CE_DELIVERED_CUT,
            color="k",
            ls="--",
            lw=1,
            label=f"delivered-contrast cut ({CE_DELIVERED_CUT} nat/tok)",
        )
        ax.axhline(-CE_DELIVERED_CUT, color="k", ls="--", lw=1)
        ax.set_xticks(xs2)
        ax.set_xticklabels(
            [n.removeprefix("rowtype_ce_") for n in names], rotation=30, ha="right", fontsize=8
        )
        ax.set_ylabel("Δ mean per-token CE (nat)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        ax.set_title("Delivered-contrast diagnostic: per-row-type CE movement, trained vs base")
        fig.tight_layout()
        p = figure_dir / "contrastive_2x2_rowtype_ce.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))
    return written


def main() -> int:  # noqa: C901 - registered multi-step analysis with a binding step order
    parser = argparse.ArgumentParser(
        description="#552 contrastive-2x2 VM analysis (plan v5 Phase 13)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fu", default=DEFAULT_FU, help="Follow-up output root ($FU).")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--figure-dir", default="figures/issue_552")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--reference-roots-json",
        default=None,
        help="Optional JSON overriding the reference shifts roots {arm: dir} (smoke fixtures).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    fu = Path(args.fu)
    if not fu.is_absolute():
        fu = PROJECT_ROOT / fu
    reference_roots = dict(DEFAULT_REFERENCE_SHIFT_ROOTS)
    if args.reference_roots_json:
        reference_roots.update(json.loads(Path(args.reference_roots_json).read_text()))

    # Resolve held-out-9 from the panel inputs (never hardcoded).
    personas = json.loads(
        (PROJECT_ROOT / "eval_results/issue_521/inputs/personas.json").read_text()
    )
    held_out = sorted(set(personas) - set(GRADIENT_TOUCHED))
    assert len(held_out) == 9, f"expected 9 held-out personas, got {held_out}"

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 (BINDING ORDER, MF-C): reference subpanel bands FIRST —
    # written to disk BEFORE any new-arm tensor is opened.
    # ──────────────────────────────────────────────────────────────────
    subpanel_dir = fu / "subpanel"
    subpanel_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[phase=reference_bands] 9 reference cells, %d-rep nulls at 9 rows", args.n_reps)
    ref_subpanel: dict[str, dict] = {}
    for arm in (*PLAIN_ARMS, MARKER_ARM):
        for seed in SEEDS:
            shifts = _load_shifts(_shift_path(arm, seed, fu, reference_roots))
            ref_subpanel[f"{arm}_seed{seed}"] = _subpanel_read(shifts, held_out, args.n_reps, seed)
    plain_cells = [ref_subpanel[f"{a}_seed{s}"] for a in PLAIN_ARMS for s in SEEDS]
    marker_cells = [ref_subpanel[f"{MARKER_ARM}_seed{s}"] for s in SEEDS]
    bands: dict = {"concentrated": {}, "dispersed": {}, "degraded_metrics": {}}
    for metric in ("mean_cos_to_U1", "s_top1_frac"):
        conc = _band([c[metric] for c in plain_cells])
        disp = _band([c[metric] for c in marker_cells])
        # Degraded mode (pre-registered): bands overlap on this metric.
        overlap = conc["min"] <= disp["max"] and disp["min"] <= conc["max"]
        bands["concentrated"][metric] = conc
        bands["dispersed"][metric] = disp
        bands["degraded_metrics"][metric] = bool(overlap)
    bands["rule"] = (
        "stays-concentrated = ALL non-degraded metrics >= concentrated band min; "
        "disperses = ALL non-degraded metrics <= dispersed band max; else intermediate. "
        "A cell whose 9-row top-share <= its own re-run sign-flip p95 is at-noise-floor "
        "(no categorical call). Bands computed BEFORE new-arm unblinding (plan v5 §6.3)."
    )
    bands["reference_cells"] = ref_subpanel
    bands["metadata"] = _repro_metadata()
    bands_path = subpanel_dir / "reference_bands.json"
    with bands_path.open("w") as f:
        json.dump(bands, f, indent=2)
    logger.info(
        "[phase=reference_bands_written] %s (degraded: %s)",
        bands_path,
        bands["degraded_metrics"],
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: new-arm subpanel reads + rubric classification (unblinding).
    # ──────────────────────────────────────────────────────────────────
    logger.info("[phase=new_arm_subpanel] 6 new cells")
    new_subpanel: dict[str, dict] = {}
    new_shifts_cache: dict[tuple[str, int], dict] = {}
    for arm in NEW_ARMS:
        for seed in SEEDS:
            shifts = _load_shifts(_shift_path(arm, seed, fu, reference_roots))
            new_shifts_cache[(arm, seed)] = shifts
            cell = _subpanel_read(shifts, held_out, args.n_reps, seed)
            cell["rubric_classification"] = _classify_subpanel(cell, bands)
            new_subpanel[f"{arm}_seed{seed}"] = cell
    with (subpanel_dir / "per_cell.json").open("w") as f:
        json.dump(
            {
                "new_cells": new_subpanel,
                "bands_path": str(bands_path),
                "metadata": _repro_metadata(),
            },
            f,
            indent=2,
        )

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: 5-arm cross-arm direction identity (end-slot + mean-resp).
    # ──────────────────────────────────────────────────────────────────
    logger.info("[phase=cross_arm_5way] 15 cells x 2 positions")
    u1_end: dict[tuple[str, int], np.ndarray] = {}
    u1_mr: dict[tuple[str, int], np.ndarray] = {}
    full_panel: dict[str, dict] = {}
    for arm in ALL_ARMS:
        for seed in SEEDS:
            shifts = (
                new_shifts_cache[(arm, seed)]
                if (arm, seed) in new_shifts_cache
                else _load_shifts(_shift_path(arm, seed, fu, reference_roots))
            )
            u1_end[(arm, seed)] = _u1(shifts, use_mean_resp=False)
            u1_mr[(arm, seed)] = _u1(shifts, use_mean_resp=True)
            fp = _full_panel_read(shifts, args.n_reps, seed)
            fp.pop("U1")
            full_panel[f"{arm}_seed{seed}"] = fp
    cross_arm = {
        "floor_random_direction": CROSS_ARM_FLOOR,
        "end_slot": _pairwise_medians(u1_end),
        "mean_resp": _pairwise_medians(u1_mr),
        "construction_asymmetry_note": (
            "the two contrastive arms share 5,899 identical negative rows AND questions; "
            "the plain pair shares questions only — contrastive-pair similarity has a "
            "higher mechanical floor (plan v5 §6.6); name this before any 'agree more' read"
        ),
        "metadata": _repro_metadata(),
    }
    cross_dir = fu / "cross_arm_5way"
    cross_dir.mkdir(parents=True, exist_ok=True)
    with (cross_dir / "summary.json").open("w") as f:
        json.dump(cross_arm, f, indent=2)

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: full-panel zone calls for the 6 NEW cells ONLY (plan §3/§6.3).
    # ──────────────────────────────────────────────────────────────────
    logger.info("[phase=zone_calls] 6 new cells, registered discipline")
    # Cross-check the recomputed full-panel numbers against the dispatcher's
    # Phase-D JSONs when present (same pipeline — must agree to ~1e-4).
    for arm in NEW_ARMS:
        for seed in SEEDS:
            svd_json = fu / "svd" / f"same_{arm}_seed{seed}.json"
            if svd_json.exists():
                d = json.loads(svd_json.read_text())
                rec = full_panel[f"{arm}_seed{seed}"]
                for ours, theirs in (
                    (rec["mean_cos_to_U1"], float(d["mean_cos_to_U1"])),
                    (rec["s_top1_frac"], float(d["s_top1_frac"])),
                ):
                    assert abs(ours - theirs) < 1e-3, (
                        f"recomputed full-panel metric diverges from Phase-D JSON for "
                        f"{arm}_seed{seed}: {ours} vs {theirs}"
                    )

    zone_calls: dict[str, dict] = {}
    arm_calls: dict[str, dict] = {}
    for arm in NEW_ARMS:
        per_seed = {}
        for seed in SEEDS:
            cell_key = f"{arm}_seed{seed}"
            fp = full_panel[cell_key]
            shifts = new_shifts_cache[(arm, seed)]
            split_half = _split_half_cosines(shifts)
            median_split_half = statistics.median(split_half.values())
            clears_sign_flip = fp["s_top1_frac"] > fp["sign_flip_p95"]
            validity = clears_sign_flip and median_split_half >= SPLIT_HALF_MIN
            zone = _zone_of(fp["mean_cos_to_U1"], fp["s_top1_frac"])
            # Split-half disattenuation (registered companion wherever a
            # de-concentration call could be made): raw cos / sqrt(reliability).
            disattenuated = {
                p: (fp["cos_to_U1"][p] / float(np.sqrt(r)) if r > 0 else None)
                for p, r in split_half.items()
            }
            per_seed[cell_key] = {
                "zone": zone,
                "mean_cos_to_U1": fp["mean_cos_to_U1"],
                "mean_abs_cos_to_U1": fp["mean_abs_cos_to_U1"],
                "s_top1_frac": fp["s_top1_frac"],
                "sign_flip_p95": fp["sign_flip_p95"],
                "clears_sign_flip_null": clears_sign_flip,
                "median_per_persona_split_half_cosine": median_split_half,
                "split_half_cosines": split_half,
                "validity_precondition_pass": validity,
                "frob_norm_M": fp["frob_norm"],
                "disattenuated_cos_to_U1": disattenuated,
                "null_scale_check_pass": NULL_SCALE_LO <= fp["sign_flip_p95"] <= NULL_SCALE_HI,
            }
            zone_calls[cell_key] = per_seed[cell_key]
        n_deconc_valid = sum(
            1
            for v in per_seed.values()
            if v["zone"] == "de-concentrated" and v["validity_precondition_pass"]
        )
        n_conc = sum(1 for v in per_seed.values() if v["zone"] == "concentrated")
        if n_deconc_valid == 3:
            arm_call = "de-concentrated (3/3 seeds + validity precondition)"
        elif n_conc >= 2:
            arm_call = f"concentrated ({n_conc}/3 seeds)"
        else:
            arm_call = "graded/mixed (no registered call; per-seed values reported)"
        arm_calls[arm] = {
            "call": arm_call,
            "n_deconcentrated_seeds_with_validity": n_deconc_valid,
            "n_concentrated_seeds": n_conc,
            "rule": (
                "de-concentration requires 3/3 seeds in zone AND per-seed validity "
                "(sign-flip clearance + median split-half >= 0.5); concentrated requires "
                ">=2/3 seeds (plan v5 §3/§6.3 — deliberate asymmetry)"
            ),
        }

    # Delivered-contrast read (MF-A) from the pod-side CE JSONs.
    rowtype_ce: dict[str, dict] = {}
    ce_dir = fu / "rowtype_ce"
    for p in sorted(ce_dir.glob("rowtype_ce_*.json")) if ce_dir.exists() else []:
        rowtype_ce[p.stem] = json.loads(p.read_text())
    contrast_delivered: dict[str, dict] = {}
    for arm in NEW_ARMS:
        deltas = [
            rowtype_ce[f"rowtype_ce_{arm}_seed{s}"]["delta_ce_neg_vs_base"]
            for s in SEEDS
            if f"rowtype_ce_{arm}_seed{s}" in rowtype_ce
        ]
        if len(deltas) == 3:
            med = statistics.median(abs(d) for d in deltas)
            contrast_delivered[arm] = {
                "median_abs_delta_ce_neg": med,
                "delivered": med >= CE_DELIVERED_CUT,
                "per_seed_delta_ce_neg": deltas,
                "cut_nat_per_token": CE_DELIVERED_CUT,
            }
        else:
            contrast_delivered[arm] = {
                "delivered": None,
                "note": f"only {len(deltas)}/3 CE JSONs present — diagnostic incomplete",
            }

    # Free trajectory-variant attribution check (plan §6.6): same vs base
    # variant concentration per new cell, from the dispatcher's Phase-D JSONs.
    variant_check: dict[str, dict] = {}
    for arm in NEW_ARMS:
        for seed in SEEDS:
            row = {}
            for variant in ("same", "base"):
                j = fu / "svd" / f"{variant}_{arm}_seed{seed}.json"
                if j.exists():
                    d = json.loads(j.read_text())
                    row[variant] = {
                        "mean_cos_to_U1": float(d["mean_cos_to_U1"]),
                        "s_top1_frac": float(d["s_top1_frac"]),
                    }
            variant_check[f"{arm}_seed{seed}"] = row

    summary = {
        "followup_label": fu.name,
        "zone_calls_full_panel_only": zone_calls,
        "arm_level_calls": arm_calls,
        "subpanel_rubric": {
            "bands_degraded_metrics": bands["degraded_metrics"],
            "new_cells": {
                k: {
                    "classification": v["rubric_classification"],
                    "mean_cos_to_U1": v["mean_cos_to_U1"],
                    "s_top1_frac": v["s_top1_frac"],
                    "sign_flip_p95": v["sign_flip_p95"],
                }
                for k, v in new_subpanel.items()
            },
        },
        "contrast_delivered_mf_a": contrast_delivered,
        "h0_prime_conditioning": (
            "if BOTH arms read concentrated AND an arm's contrast is NOT delivered "
            "(median |dCE_neg| < 0.05 nat/token), the registered read for that arm is "
            "'weak/undelivered contrast — not evidence against contrastive negatives' "
            "(plan v5 §3 MF-A scope-down)"
        ),
        "full_panel_all_15_cells": full_panel,
        "trajectory_variant_check": variant_check,
        "cross_arm_5way_path": str(cross_dir / "summary.json"),
        "reference_bands_path": str(bands_path),
        "metadata": _repro_metadata(),
    }
    summary_path = fu / "contrastive_2x2_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[phase=summary_written] %s", summary_path)

    if not args.no_figures:
        figure_dir = Path(args.figure_dir)
        if not figure_dir.is_absolute():
            figure_dir = PROJECT_ROOT / figure_dir
        written = _make_figures(
            figure_dir, full_panel, {**ref_subpanel, **new_subpanel}, rowtype_ce
        )
        logger.info("[phase=figures] %s", written)

    logger.info("[phase=done] analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
