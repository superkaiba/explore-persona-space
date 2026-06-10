#!/usr/bin/env python3
"""#551 free-analysis follow-up: unit-norm SVD re-read of the marker split.

The parent read stacks each cell's 14 per-persona layer-14 shift vectors into
a 3584x14 matrix and takes the top singular direction U1. Personas with small
shifts contribute little to the SVD objective, so a low cosine-to-U1 could be
the SVD's norm-weighting rather than real direction structure. This re-read
column-normalizes each persona's shift to UNIT norm BEFORE the SVD (the
matrix no longer knows the norms), then recomputes:

1. unit-norm top-direction share per cell (all 18 cells, both arms);
2. per-persona |cos to the unit-norm U1| for the 6 trained-model-text cells;
3. split-membership comparison per marker trained-model-text cell — aligned
   (|cos| >= 0.5) vs elsewhere under BOTH reads: overlap, movers, the rank
   correlation between the two per-persona |cos| profiles, and
   |cos(U1_unit, U1_weighted)|;
4. rho(||shift||, |cos-to-U1_unit|) per trained-model-text cell with a
   one-sided positive permutation p (10,000 draws, same convention as
   ``issue551_controls.py``) — under the unit-norm read a surviving
   correlation can no longer be the SVD's norm-weighting;
5. the EM arm's unit-norm read (does the near-1 concentration survive?).

Reading rule (encoded in the JSON summary, not prose): if the weighted read's
low-cos personas REMAIN low-cos under the unit-norm read (membership stable,
high rank correlation), the split is real direction structure and the
norm-weighting alternative dies; if the split dissolves (everyone comparable,
U1 rotates a lot), the weighted split was substantially the SVD's
norm-weighting.

Top-share definition deliberately matches ``svd_summary`` (``s_1 / sum(s)``,
NOT squared mass) for comparability with the persisted #551/#521 JSONs.

Zero GPU; reads only the persisted shift tensors + the existing
norm-alignment JSON (consistency cross-check). Run from the repo root::

    uv run python scripts/issue551_unitnorm_reread.py \\
        --local-shifts-dir eval_results/issue_551/shifts \\
        --out eval_results/issue_551/controls/unitnorm_reread.json \\
        --figures-dir figures/issue_551
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from issue551_controls import (
    N_PERM,
    SOURCE_PERSONA,
    CellKey,
    _all_cells,
    _git_commit,
    _load_cell,
    _one_sided_perm_p,
    _same_cells,
)

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    spearman_rho,
    svd_summary,
)

logger = logging.getLogger(__name__)

ALIGNED_COS_THRESHOLD = 0.5  # same split rule the round-2 figures used
WEIGHTED_CONSISTENCY_TOL = 1e-3  # recomputed weighted cos vs persisted norm_alignment.json

ARM_LABEL = {"em": "EM (bad medical advice)", "marker": "marker (※)"}
SEED_MARKERS = {42: "o", 137: "^", 256: "s"}


def unit_normalize_columns(M: np.ndarray) -> np.ndarray:
    """Scale each column of M (H x N) to unit L2 norm; zero columns fail loud."""
    norms = np.linalg.norm(M, axis=0)
    if not np.all(norms > 0):
        raise ValueError(f"zero-norm column(s) at indices {np.where(norms == 0)[0].tolist()}")
    return (M / norms).astype(np.float32)


def _check_weighted_consistency(
    cell: CellKey, personas: list[str], cos_w: dict[str, float], norm_align: dict
) -> float:
    """Cross-check the recomputed weighted cos-to-U1 against the persisted JSON.

    Guards against silently re-deriving the weighted read with a different
    matrix convention than the round-1 controls. Returns the max abs delta;
    raises on breach.
    """
    stored = norm_align["per_cell"][cell.name]["cos_to_U1"]
    deltas = [abs(cos_w[p] - float(stored[p])) for p in personas]
    max_d = max(deltas)
    if max_d > WEIGHTED_CONSISTENCY_TOL:
        raise ValueError(
            f"{cell.name}: recomputed weighted cos-to-U1 disagrees with persisted "
            f"norm_alignment.json (max |delta|={max_d:.2e} > {WEIGHTED_CONSISTENCY_TOL}); "
            f"refusing to compare reads built from different conventions."
        )
    return max_d


def analyze(shifts_dir: Path, norm_align: dict) -> dict:
    """Run the unit-norm re-read over all 18 cells; returns the payload body."""
    same_names = {c.name for c in _same_cells()}
    per_cell_top_share: dict[str, dict] = {}
    trained_text_cells: dict[str, dict] = {}
    marker_split_membership: dict[str, dict] = {}
    norm_vs_alignment_unitnorm: dict[str, dict] = {}

    for cell in _all_cells():
        shifts = _load_cell(shifts_dir, cell)
        M, personas = assemble_M(shifts)  # sorted persona order (deterministic)
        assert M.shape[1] == 14, M.shape
        svd_w = svd_summary(M)
        svd_u = svd_summary(unit_normalize_columns(M))
        u1_agree = abs(cosine(svd_u["U1"], svd_w["U1"]))
        per_cell_top_share[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "s_top1_frac_weighted": float(svd_w["s_top1_frac"]),
            "s_top1_frac_unitnorm": float(svd_u["s_top1_frac"]),
            "abs_cos_U1_unitnorm_vs_weighted": float(u1_agree),
        }
        logger.info(
            "[top_share %s] weighted=%.4f unitnorm=%.4f |cos(U1_u,U1_w)|=%.4f",
            cell.name,
            svd_w["s_top1_frac"],
            svd_u["s_top1_frac"],
            u1_agree,
        )
        if cell.name not in same_names:
            continue

        # ── trained-model-text cells: per-persona detail ──────────────
        norms = np.linalg.norm(M, axis=0)
        cos_w = {p: float(svd_w["cos_to_U1"][i]) for i, p in enumerate(personas)}
        cos_u = {p: float(svd_u["cos_to_U1"][i]) for i, p in enumerate(personas)}
        max_d = _check_weighted_consistency(cell, personas, cos_w, norm_align)
        trained_text_cells[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "weighted_consistency_max_abs_delta": float(max_d),
            "per_persona": {
                p: {
                    "shift_norm": float(norms[i]),
                    "cos_to_U1_weighted": cos_w[p],
                    "cos_to_U1_unitnorm": cos_u[p],
                    "abs_cos_to_U1_weighted": abs(cos_w[p]),
                    "abs_cos_to_U1_unitnorm": abs(cos_u[p]),
                }
                for i, p in enumerate(personas)
            },
        }

        # ── norm-vs-alignment under the unit-norm read ────────────────
        abs_cos_u = np.array([abs(cos_u[p]) for p in personas])
        rho, p_one_sided = _one_sided_perm_p(norms, abs_cos_u, seed=cell.seed)
        norm_vs_alignment_unitnorm[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "spearman_rho_norm_vs_abs_cos_unitnorm": rho,
            "p_one_sided_positive": p_one_sided,
            "n_perm": N_PERM,
        }
        logger.info("[norm_vs_alignment %s] rho=%.3f p_one_sided=%.4f", cell.name, rho, p_one_sided)

        # ── marker cells: split-membership comparison ─────────────────
        if cell.arm != "marker":
            continue
        aligned_w = sorted(p for p in personas if abs(cos_w[p]) >= ALIGNED_COS_THRESHOLD)
        aligned_u = sorted(p for p in personas if abs(cos_u[p]) >= ALIGNED_COS_THRESHOLD)
        moved_out = sorted(set(aligned_w) - set(aligned_u))  # lost alignment under unit-norm
        moved_in = sorted(set(aligned_u) - set(aligned_w))  # gained alignment under unit-norm
        union = set(aligned_w) | set(aligned_u)
        jaccard = (len(set(aligned_w) & set(aligned_u)) / len(union)) if union else 1.0
        rank_corr = spearman_rho(
            [abs(cos_w[p]) for p in personas], [abs(cos_u[p]) for p in personas]
        )
        marker_split_membership[cell.name] = {
            "seed": cell.seed,
            "aligned_cos_threshold": ALIGNED_COS_THRESHOLD,
            "aligned_weighted": aligned_w,
            "aligned_unitnorm": aligned_u,
            "moved_out_under_unitnorm": moved_out,
            "moved_in_under_unitnorm": moved_in,
            "n_changed_membership": len(moved_in) + len(moved_out),
            "membership_identical": not moved_in and not moved_out,
            "aligned_set_jaccard": float(jaccard),
            "spearman_abs_cos_weighted_vs_unitnorm": float(rank_corr),
            "abs_cos_U1_unitnorm_vs_weighted": float(u1_agree),
        }
        logger.info(
            "[membership %s] identical=%s jaccard=%.3f rank_corr=%.3f movers=%s",
            cell.name,
            marker_split_membership[cell.name]["membership_identical"],
            jaccard,
            rank_corr,
            moved_in + moved_out,
        )

    # ── summary (interpretation-ready numbers, no prose verdict) ──────
    mk = marker_split_membership
    em_same = {
        f"seed{v['seed']}": {
            "s_top1_frac_weighted": v["s_top1_frac_weighted"],
            "s_top1_frac_unitnorm": v["s_top1_frac_unitnorm"],
        }
        for v in per_cell_top_share.values()
        if v["arm"] == "em" and v["variant"] == "same"
    }
    summary = {
        "marker_membership_identical_all_seeds": all(
            v["membership_identical"] for v in mk.values()
        ),
        "marker_n_changed_by_seed": {
            f"seed{v['seed']}": v["n_changed_membership"] for v in mk.values()
        },
        "marker_aligned_set_jaccard_by_seed": {
            f"seed{v['seed']}": v["aligned_set_jaccard"] for v in mk.values()
        },
        "marker_rank_corr_by_seed": {
            f"seed{v['seed']}": v["spearman_abs_cos_weighted_vs_unitnorm"] for v in mk.values()
        },
        "marker_abs_cos_U1_by_seed": {
            f"seed{v['seed']}": v["abs_cos_U1_unitnorm_vs_weighted"] for v in mk.values()
        },
        "marker_unitnorm_norm_vs_alignment_by_seed": {
            f"seed{v['seed']}": {
                "rho": v["spearman_rho_norm_vs_abs_cos_unitnorm"],
                "p_one_sided_positive": v["p_one_sided_positive"],
            }
            for v in norm_vs_alignment_unitnorm.values()
            if v["arm"] == "marker"
        },
        "em_same_top_share_by_seed": em_same,
        "reading_rule": (
            "membership stable + high rank correlation => the split is real direction "
            "structure (norm-weighting alternative dies); split dissolves + U1 rotates "
            "=> the weighted split was substantially the SVD's norm-weighting. A "
            "surviving positive rho(||shift||, |cos-to-U1_unit|) can no longer be the "
            "SVD's norm-weighting (the unit-norm matrix does not know the norms)."
        ),
    }
    return {
        "per_cell_top_share": per_cell_top_share,
        "trained_text_cells": trained_text_cells,
        "marker_split_membership": marker_split_membership,
        "norm_vs_alignment_unitnorm": norm_vs_alignment_unitnorm,
        "summary": summary,
    }


def make_figure(body: dict, figures_dir: Path) -> None:
    """One scatter: per-persona |cos to U1|, weighted vs unit-norm, marker arm."""
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    mk_color = paper_palette(4)[1]  # orange = marker arm everywhere in #551 figures

    cells = sorted(
        (v for v in body["trained_text_cells"].values() if v["arm"] == "marker"),
        key=lambda v: v["seed"],
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1.0, zorder=0)
    ax.axvline(ALIGNED_COS_THRESHOLD, color="gray", lw=0.9, linestyle="--", zorder=0)
    ax.axhline(ALIGNED_COS_THRESHOLD, color="gray", lw=0.9, linestyle="--", zorder=0)
    for v in cells:
        seed = v["seed"]
        pp = v["per_persona"]
        others = [p for p in pp if p != SOURCE_PERSONA]
        ax.scatter(
            [pp[p]["abs_cos_to_U1_weighted"] for p in others],
            [pp[p]["abs_cos_to_U1_unitnorm"] for p in others],
            s=26,
            color=mk_color,
            marker=SEED_MARKERS[seed],
            alpha=0.75,
            label=f"seed {seed}",
        )
        ax.scatter(
            [pp[SOURCE_PERSONA]["abs_cos_to_U1_weighted"]],
            [pp[SOURCE_PERSONA]["abs_cos_to_U1_unitnorm"]],
            s=85,
            color=mk_color,
            marker="X",
            edgecolors="black",
            linewidths=0.9,
            zorder=3,
        )
    src_handle = mlines.Line2D(
        [],
        [],
        color="white",
        marker="X",
        markersize=8,
        markeredgecolor="black",
        markerfacecolor=mk_color,
        linestyle="None",
        label="medical doctor (trained persona)",
    )
    thr_handle = mlines.Line2D(
        [], [], color="gray", lw=0.9, linestyle="--", label="aligned threshold (|cos| = 0.5)"
    )
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, src_handle, thr_handle], fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("|cosine to top direction| — norm-weighted SVD (parent read)")
    ax.set_ylabel("|cosine to top direction| — unit-norm SVD (this re-read)")
    ax.set_title(f"{ARM_LABEL['marker']} arm: alignment under norm-weighted vs unit-norm reads")
    fig.tight_layout()
    savefig_paper(fig, "unitnorm_reread", dir=figures_dir)
    plt.close(fig)
    logger.info("[figure] written under %s", figures_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="#551 unit-norm SVD re-read (CPU, zero GPU)")
    parser.add_argument("--local-shifts-dir", default="eval_results/issue_551/shifts")
    parser.add_argument(
        "--norm-align-json",
        default="eval_results/issue_551/controls/norm_alignment.json",
        help="Persisted round-1 weighted read; used as a consistency cross-check.",
    )
    parser.add_argument("--out", default="eval_results/issue_551/controls/unitnorm_reread.json")
    parser.add_argument("--figures-dir", default="figures/issue_551")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import importlib.metadata

    with Path(args.norm_align_json).open() as f:
        norm_align = json.load(f)
    body = analyze(Path(args.local_shifts_dir), norm_align)
    payload = {
        "meta": {
            "issue": 551,
            "analysis": "unitnorm_reread",
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "env_versions": {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "numpy")},
            "tensors_source": str(args.local_shifts_dir),
            "norm_align_json": str(args.norm_align_json),
            "thresholds": {
                "aligned_cos_threshold": ALIGNED_COS_THRESHOLD,
                "n_perm": N_PERM,
                "weighted_consistency_tol": WEIGHTED_CONSISTENCY_TOL,
            },
            "top_share_definition": (
                "s_1 / sum(s) (matches svd_summary and the persisted #551/#521 JSONs; "
                "NOT squared singular-value mass)"
            ),
            "source_persona": SOURCE_PERSONA,
        },
        **body,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON first, figure second (checkpoint per phase: a plotting failure
    # must not lose the numbers).
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("[wrote] %s", out_path)

    make_figure(body, Path(args.figures_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
