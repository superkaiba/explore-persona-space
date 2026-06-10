#!/usr/bin/env python3
"""#551 free-analysis follow-up (a): layer-7/21 spectrum sensitivity re-read.

Question: does the EM-vs-marker direction-concentration contrast measured at
layer 14 (EM top-share and mean |cos| far above marker; source-drop/LOO
survival over the sign-flip null; unit-norm dissolution of the marker split)
hold on the layer-7 and layer-21 reads of the SAME persisted shift tensors?

ONE change vs the layer-14 machinery: the per-persona tensor key is
``delta_v_l7`` / ``delta_v_l21`` instead of ``delta_v`` (and
``delta_v_mean_resp_l7`` / ``delta_v_mean_resp_l21`` for the whole-response
cross-check on the 6 trained-model-text cells). Spectrum, nulls (sign-flip
BINDING + row-shuffle descriptive, 1,000 reps, seed = cell training seed —
same convention as ``issue551_controls.py``), per-persona cosines, LOO
source-drop (mirrors Control A's cells), and the unit-norm re-read
(``issue551_unitnorm_reread.py`` conventions) are all reused unchanged.

Deliberate skips at the non-default layers, recorded in each layer JSON:

- Phase R (reproduction gate) — the parent #521 SVD JSONs are layer-14
  only; there is nothing to anchor a layer-7/21 gate against.
- Per-question supplementaries (Control C split-half reliability) — the
  per-question tensors were persisted at layer 14 only.

Falsification readout (encoded under ``falsification`` in summary.json):
the contrast vanishes or inverts at layer 7 or 21 = (marker top-share >=
EM top-share within variant-and-seed) OR (EM cells fail their sign-flip
nulls). One boolean per layer + a one-line verdict string.

Zero GPU; reads only the persisted shift tensors. Run from the repo root::

    uv run python scripts/issue551_layer_sensitivity.py \\
        --local-shifts-dir eval_results/issue_551/shifts \\
        --out-dir eval_results/issue_551/layer-sensitivity-7-21
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from issue551_controls import (
    N_NULL_REPS,
    SOURCE_PERSONA,
    CellKey,
    _all_cells,
    _git_commit,
    _load_cell,
    _null_summary_entry,
    _same_cells,
    _write_json,
)
from issue551_unitnorm_reread import ALIGNED_COS_THRESHOLD, unit_normalize_columns

from explore_persona_space.analysis.svd_direction_constancy import (
    assemble_M,
    cosine,
    row_shuffle_null,
    sign_flip_null,
    spearman_rho,
    svd_summary,
)

logger = logging.getLogger(__name__)

LAYERS = (7, 21)
SKIP_PHASE_R = (
    "skipped: the Phase R reproduction gate anchors against parent #521 layer-14 "
    "SVD JSONs; no parent reference exists at this layer"
)
SKIP_PER_QUESTION = "skipped: per-question tensors persisted at layer 14 only"


def _run_meta(args: argparse.Namespace, layer: int) -> dict:
    """Reproducibility metadata embedded in every output JSON."""
    import importlib.metadata

    return {
        "issue": 551,
        "followup_of": 521,
        "followup_label": "layer-sensitivity-and-seed137-degeneracy",
        "analysis": f"layer_sensitivity_l{layer}",
        "layer": layer,
        "tensor_key": f"delta_v_l{layer}",
        "mean_resp_tensor_key": f"delta_v_mean_resp_l{layer}",
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_versions": {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "numpy")},
        "tensors_source": str(args.local_shifts_dir),
        "thresholds": {
            "n_null_reps": N_NULL_REPS,
            "aligned_cos_threshold": ALIGNED_COS_THRESHOLD,
            "null_rng_seed": "per-cell training seed (matches issue551_controls.py)",
        },
        "top_share_definition": (
            "s_1 / sum(s) (matches svd_summary and the persisted #551/#521 JSONs; "
            "NOT squared singular-value mass)"
        ),
        "source_persona": SOURCE_PERSONA,
        "skipped": {"phase_r": SKIP_PHASE_R, "per_question_controls": SKIP_PER_QUESTION},
    }


def analyze_layer(shifts_dir: Path, layer: int) -> dict:
    """Full re-read of all 18 cells at one layer; returns the payload body."""
    key = f"delta_v_l{layer}"
    mr_key = f"delta_v_mean_resp_l{layer}"
    same_names = {c.name for c in _same_cells()}

    per_cell: dict[str, dict] = {}
    loo_per_cell: dict[str, dict] = {}
    trained_text_cells: dict[str, dict] = {}
    marker_split_membership: dict[str, dict] = {}
    mean_resp_per_cell: dict[str, dict] = {}

    for cell in _all_cells():
        shifts = _load_cell(shifts_dir, cell)
        M, personas = assemble_M(shifts, tensor_key=key)  # sorted persona order
        assert M.shape[1] == 14, M.shape
        svd_w = svd_summary(M)
        sign_null = sign_flip_null(M, n_reps=N_NULL_REPS, seed=cell.seed)
        row_null = row_shuffle_null(M, n_reps=N_NULL_REPS, seed=cell.seed)
        svd_u = svd_summary(unit_normalize_columns(M))
        u1_agree = abs(cosine(svd_u["U1"], svd_w["U1"]))
        cos_w = {p: float(svd_w["cos_to_U1"][i]) for i, p in enumerate(personas)}
        cos_u = {p: float(svd_u["cos_to_U1"][i]) for i, p in enumerate(personas)}

        per_cell[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "s_top1_frac": float(svd_w["s_top1_frac"]),
            "mean_cos_to_U1": float(np.mean(svd_w["cos_to_U1"])),
            "mean_abs_cos_to_U1": float(np.mean(np.abs(svd_w["cos_to_U1"]))),
            "cos_to_U1": cos_w,
            "sign_flip": _null_summary_entry(sign_null),
            "row_shuffle": _null_summary_entry(row_null),
            "passes_sign_flip_p95": bool(svd_w["s_top1_frac"] > sign_null["p95"]),  # BINDING
            "passes_row_shuffle_p95": bool(svd_w["s_top1_frac"] > row_null["p95"]),  # descriptive
            "unitnorm": {
                "s_top1_frac_unitnorm": float(svd_u["s_top1_frac"]),
                "abs_cos_U1_unitnorm_vs_weighted": float(u1_agree),
            },
        }
        logger.info(
            "[l%d %s] top1=%.4f mean_cos=%.4f sign_p95=%.4f unitnorm_top1=%.4f",
            layer,
            cell.name,
            svd_w["s_top1_frac"],
            per_cell[cell.name]["mean_cos_to_U1"],
            sign_null["p95"],
            svd_u["s_top1_frac"],
        )

        # ── LOO source-drop (mirrors Control A: all cells, binding on same) ──
        src_idx = personas.index(SOURCE_PERSONA)
        m_loo = np.delete(M, src_idx, axis=1)
        assert m_loo.shape[1] == M.shape[1] - 1, (m_loo.shape, M.shape)
        svd_loo = svd_summary(m_loo)
        loo_sign = sign_flip_null(m_loo, n_reps=N_NULL_REPS, seed=cell.seed)
        loo_row = row_shuffle_null(m_loo, n_reps=N_NULL_REPS, seed=cell.seed)
        loo_per_cell[cell.name] = {
            "variant": cell.variant,
            "arm": cell.arm,
            "seed": cell.seed,
            "source_dropped": SOURCE_PERSONA,
            "s_top1_frac_full": float(svd_w["s_top1_frac"]),
            "s_top1_frac_loo": float(svd_loo["s_top1_frac"]),
            "mean_cos_to_U1_loo": float(np.mean(svd_loo["cos_to_U1"])),
            "sign_flip": _null_summary_entry(loo_sign),
            "row_shuffle": _null_summary_entry(loo_row),
            "passes_sign_flip_p95": bool(svd_loo["s_top1_frac"] > loo_sign["p95"]),  # BINDING
            "passes_row_shuffle_p95": bool(svd_loo["s_top1_frac"] > loo_row["p95"]),
            "margin_over_sign_flip_p95": float(svd_loo["s_top1_frac"] - loo_sign["p95"]),
            "is_binding_cell": cell.variant == "same",
        }

        if cell.name not in same_names:
            continue

        # ── trained-model-text cells: per-persona detail ──────────────
        norms = np.linalg.norm(M, axis=0)
        trained_text_cells[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
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

        # ── marker cells: unit-norm split-membership comparison ──────
        if cell.arm == "marker":
            aligned_w = sorted(p for p in personas if abs(cos_w[p]) >= ALIGNED_COS_THRESHOLD)
            aligned_u = sorted(p for p in personas if abs(cos_u[p]) >= ALIGNED_COS_THRESHOLD)
            moved_out = sorted(set(aligned_w) - set(aligned_u))
            moved_in = sorted(set(aligned_u) - set(aligned_w))
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

        # ── whole-response cross-check (mirrors Control B primary) ────
        m_mr, _ = assemble_M(shifts, tensor_key=mr_key)
        svd_mr = svd_summary(m_mr)
        mr_sign = sign_flip_null(m_mr, n_reps=N_NULL_REPS, seed=cell.seed)
        mr_row = row_shuffle_null(m_mr, n_reps=N_NULL_REPS, seed=cell.seed)
        mean_resp_per_cell[cell.name] = {
            "arm": cell.arm,
            "seed": cell.seed,
            "s_top1_frac_mean_resp": float(svd_mr["s_top1_frac"]),
            "mean_cos_to_U1_mean_resp": float(np.mean(svd_mr["cos_to_U1"])),
            "sign_flip": _null_summary_entry(mr_sign),
            "row_shuffle": _null_summary_entry(mr_row),
            "passes_sign_flip_p95": bool(svd_mr["s_top1_frac"] > mr_sign["p95"]),
        }

    # ── arm contrast within variant x seed (the falsification input) ──
    contrast: dict[str, dict] = {}
    for variant in sorted({c.variant for c in _all_cells()}):
        for seed in sorted({c.seed for c in _all_cells()}):
            em = per_cell[CellKey(variant, "em", seed).name]
            mk = per_cell[CellKey(variant, "marker", seed).name]
            contrast[f"{variant}_seed{seed}"] = {
                "em_s_top1_frac": em["s_top1_frac"],
                "marker_s_top1_frac": mk["s_top1_frac"],
                "marker_ge_em_top_share": bool(mk["s_top1_frac"] >= em["s_top1_frac"]),
                "em_mean_cos_to_U1": em["mean_cos_to_U1"],
                "marker_mean_cos_to_U1": mk["mean_cos_to_U1"],
                "em_passes_sign_flip_p95": em["passes_sign_flip_p95"],
            }

    loo_binding = [v for v in loo_per_cell.values() if v["is_binding_cell"]]
    return {
        "per_cell": per_cell,
        "loo": {
            "binding_pass": all(v["passes_sign_flip_p95"] for v in loo_binding),
            "n_binding_cells": len(loo_binding),
            "per_cell": loo_per_cell,
        },
        "trained_text_cells": trained_text_cells,
        "marker_split_membership": marker_split_membership,
        "mean_resp": mean_resp_per_cell,
        "arm_contrast_within_variant_seed": contrast,
        "skipped": {"phase_r": SKIP_PHASE_R, "per_question_controls": SKIP_PER_QUESTION},
    }


def _layer_falsification(body: dict) -> dict:
    """Encode the pre-registered falsification readout for one layer."""
    contrast = body["arm_contrast_within_variant_seed"]
    marker_ge_em = sorted(k for k, v in contrast.items() if v["marker_ge_em_top_share"])
    em_fail_sign = sorted(
        k for k, v in body["per_cell"].items() if v["arm"] == "em" and not v["passes_sign_flip_p95"]
    )
    falsified = bool(marker_ge_em or em_fail_sign)
    verdict = (
        "FALSIFIED: the EM-vs-marker direction-concentration contrast vanishes or "
        f"inverts at this layer (marker>=EM in {marker_ge_em}; EM sign-flip "
        f"failures in {em_fail_sign})."
        if falsified
        else "HOLDS: EM top-share exceeds marker in every variant x seed and all "
        "EM cells clear their sign-flip null p95 at this layer."
    )
    return {
        "rule": (
            "contrast vanishes or inverts = (marker top-share >= EM top-share within "
            "variant-and-seed, weighted read) OR (any EM cell s_top1_frac <= its "
            "full-panel sign-flip null p95)"
        ),
        "marker_top_share_ge_em_cells": marker_ge_em,
        "em_cells_failing_sign_flip_p95": em_fail_sign,
        "contrast_vanishes_or_inverts": falsified,
        "verdict": verdict,
    }


def _layer14_reference(controls_dir: Path) -> dict:
    """Headline layer-14 numbers from the persisted round-1/2 control JSONs."""
    ref: dict[str, object] = {
        "source_files": [
            str(controls_dir / "unitnorm_reread.json"),
            str(controls_dir / "loo.json"),
        ]
    }
    with (controls_dir / "unitnorm_reread.json").open() as f:
        unit = json.load(f)
    ref["top_share_same_cells"] = {
        k: {
            "weighted": v["s_top1_frac_weighted"],
            "unitnorm": v["s_top1_frac_unitnorm"],
        }
        for k, v in unit["per_cell_top_share"].items()
        if v["variant"] == "same"
    }
    ref["marker_rank_corr_weighted_vs_unitnorm_by_seed"] = unit["summary"][
        "marker_rank_corr_by_seed"
    ]
    with (controls_dir / "loo.json").open() as f:
        loo = json.load(f)
    ref["loo_same_cells"] = {
        k: {
            "s_top1_frac_loo": v["s_top1_frac_loo"],
            "sign_flip_p95": v["sign_flip"]["p95"],
            "passes_sign_flip_p95": v["passes_sign_flip_p95"],
        }
        for k, v in loo["per_cell"].items()
        if v["variant"] == "same"
    }
    return ref


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#551 layer-7/21 spectrum sensitivity re-read (CPU, zero GPU)"
    )
    parser.add_argument("--local-shifts-dir", default="eval_results/issue_551/shifts")
    parser.add_argument(
        "--layer14-controls-dir",
        default="eval_results/issue_551/controls",
        help="Persisted layer-14 control JSONs; embedded as reference in summary.json.",
    )
    parser.add_argument("--out-dir", default="eval_results/issue_551/layer-sensitivity-7-21")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    shifts_dir = Path(args.local_shifts_dir)
    out_dir = Path(args.out_dir)

    per_layer_falsification: dict[str, dict] = {}
    per_layer_meta: dict[str, dict] = {}
    for layer in LAYERS:
        logger.info("[phase=layer_%d]", layer)
        meta = _run_meta(args, layer)
        body = analyze_layer(shifts_dir, layer)
        # Checkpoint per phase: each layer JSON lands the moment it completes.
        _write_json(out_dir / f"layer{layer}.json", {"meta": meta, **body})
        per_layer_falsification[f"layer{layer}"] = _layer_falsification(body)
        per_layer_meta[f"layer{layer}"] = {
            "em_top_share_same_by_seed": {
                f"seed{v['seed']}": v["s_top1_frac"]
                for v in body["per_cell"].values()
                if v["arm"] == "em" and v["variant"] == "same"
            },
            "marker_top_share_same_by_seed": {
                f"seed{v['seed']}": v["s_top1_frac"]
                for v in body["per_cell"].values()
                if v["arm"] == "marker" and v["variant"] == "same"
            },
            "loo_binding_pass": body["loo"]["binding_pass"],
            "marker_rank_corr_weighted_vs_unitnorm_by_seed": {
                f"seed{v['seed']}": v["spearman_abs_cos_weighted_vs_unitnorm"]
                for v in body["marker_split_membership"].values()
            },
        }

    logger.info("[phase=summary]")
    summary = {
        "meta": _run_meta(args, 0) | {"analysis": "layer_sensitivity_summary", "layer": None},
        "layers": list(LAYERS),
        "headline_by_layer": per_layer_meta,
        "falsification": {
            **per_layer_falsification,
            "any_layer_contrast_vanishes_or_inverts": any(
                v["contrast_vanishes_or_inverts"] for v in per_layer_falsification.values()
            ),
        },
        "layer14_reference": _layer14_reference(Path(args.layer14_controls_dir)),
    }
    _write_json(out_dir / "summary.json", summary)
    logger.info(
        "[phase=done] falsified_l7=%s falsified_l21=%s",
        per_layer_falsification["layer7"]["contrast_vanishes_or_inverts"],
        per_layer_falsification["layer21"]["contrast_vanishes_or_inverts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
