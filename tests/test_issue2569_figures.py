"""Issue #2569 figure producers: synthetic-fixture smokes through ``savefig_paper``.

Every fixture mirrors the EXACT schema of the landed producing driver (read off
the worktree at 57808a6434): ``issue2569_gateladder`` (ladder + curve),
``issue2569_rowbattery`` (leg4 / leg8 / der), ``issue2569_dw_fleet``,
``issue2569_leg6``, and ``issue2569_atlas``. Each test drives a REAL figure
function end-to-end (matplotlib Agg) and asserts BOTH that the built figure
carries plotted artists (a silently-empty render fails here, not at review) and
that the PNG + ``.meta.json`` sidecar landed with non-trivial size. Dimensions
stay tiny throughout (no dense 3584^2 anywhere).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_figures as F  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture builders (schema-true, tiny)
# ---------------------------------------------------------------------------

_KNN = {
    "metric": "euclidean",
    "n": 8,
    "n_pool": 8,
    "acc_at_k": {"1": 0.25, "5": 0.75, "10": 1.0},
    "chance_at_k": {"1": 0.125, "5": 0.625, "10": 1.0},
    "median_rank": 2.0,
    "mrr": 0.5,
}


def _verdict_point(n: int, r2: float, theory_r2: float) -> dict:
    """One curve_core verdict point (issue2569_gateladder.fit_point shape)."""
    return {
        "n_train": n,
        "test_r2": r2,
        "selected_lambda": 0.1,
        "val_r2_at_selected": r2 - 0.01,
        "lambda_grid_edge": None,
        "n_widenings": 0,
        "final_grid": [1e-5, 1e8, 27],
        "identity_bias_r2": 0.3,
        "knn_retrieval": dict(_KNN),
        "label": f"verdict_n{n}",
        "layer": 19,
        "train_corpus": "lmsys",
        "eval_split_sha": "abc123",
        "lambda_selection": "val-selected over widened 27-value 1e-5..1e8 grid; widen-on-edge",
        "train_eval_distribution": "lmsys->lmsys",
        "corpus_mix": {"lmsys": n},
        "theory": {
            "kappa": 1.0,
            "gamma": 0.2,
            "excess_risk": 0.1,
            "predicted_mse": 0.3,
            "predicted_r2": theory_r2,
        },
    }


def _companion(n: int, r2: float, edge: str | None) -> dict:
    """One committed off-recipe companion (load_companion_points shape)."""
    return {
        "label": f"committed_n{n}",
        "n_train": n,
        "test_r2": r2,
        "selected_lambda": 1000.0,
        "lambda_grid_edge": edge,
        "layer": 19,
        "corpus_mix": {"pass_b_seed_corpus": n},
        "lambda_selection": "val-selected (committed original grid)",
        "train_eval_distribution": "pass-b mixed seed corpus",
        "source": "eval_results/issue_779/fitter-fair-comparison/fair_comparison.json",
        "off_recipe_companion": True,
    }


def _learning_curve_doc() -> dict:
    """A schema-true learning_curve.json document (curve_core output shape)."""
    pts = [
        _verdict_point(4_500, 0.62, 0.60),
        _verdict_point(50_000, 0.70, 0.69),
        _verdict_point(500_000, 0.75, 0.76),
    ]
    return {
        "regime": {
            "n_grid": [4_500, 50_000, 500_000],
            "eval_rows": 100,
            "val_rows": 100,
            "seed": 2569,
            "layer": 19,
            "lambda_grid": [1e-5, 1e8, 27],
            "lambda_selection": "val-selected",
            "lmsys_tag": "lmsys",
            "smoke": True,
        },
        "splits": {"n_lmsys_rows": 1_000, "pool_rows": 800, "eval_split_sha": "abc123"},
        "moments": {
            "n_rows": 800,
            "total_var": 1.0,
            "noise_var": 0.2,
            "linear_r2_population": 0.8,
            "eta_top8": [0.5] * 8,
            "eta_sum": 4.0,
        },
        "verdict_points": pts,
        "parity_check": {
            "reference": {},
            "per_point": [
                {"label": p["label"], "pass": True, "mismatched_fields": []} for p in pts
            ],
        },
        "parity_excluded": [],
        "h2b": {
            "mean_abs_dr2": 0.017,
            "per_point_delta_r2": [0.02, 0.01, -0.01],
            "same_sign_all": False,
            "verdict": "h2b-pass",
            "bands": {"pass_le": 0.05, "kill_gt": 0.15},
        },
        "companions_off_recipe": [_companion(3_600, 0.71, None), _companion(963_444, 0.754, "low")],
        "companion_parity_vs_verdict_reference": None,
    }


_METRICS = list(F.GATE_METRIC_ORDER)


def _ladder_arm(arm_id: str, kind: str, base_rho: float) -> dict:
    """One race_arm payload (boot/perm stripped, as run_ladder persists it)."""
    dv_names = ["dv_change", "dv_level"] if kind == "content" else ["dv_dlogp", "dv_level_logp"]
    return {
        "arm_id": arm_id,
        "raced": list(_METRICS),
        "dv_names": dv_names,
        "observed_rho": {
            d: {m: base_rho + 0.03 * i + 0.01 * j for i, m in enumerate(_METRICS)}
            for j, d in enumerate(dv_names)
        },
        "perm_band": {
            "p95_max_selected": 0.30,
            "p975_max_selected": 0.35,
            "ceiling_abs_rho": 1.0,
            "n_perm": 100,
        },
        "n": 50,
        "n_shared": 40,
        "n_degenerate_series_draws": 0,
        "kind": kind,
    }


def _champion(dv: str, arm_ids: list[str]) -> dict:
    """One ladder_champion record (exact key strings, frozen-CI key included)."""
    med = {m: 0.4 + 0.02 * i for i, m in enumerate(_METRICS)}
    return {
        "dv": dv,
        "incumbent": "gate_sigma_inv",
        "panel_metrics": list(_METRICS),
        "arm_ids": arm_ids,
        "across_arm_median_observed": med,
        "winner_observed": "gate_wwt_awhite",
        "p_win": {m: 1.0 / len(_METRICS) for m in _METRICS},
        "selection_inherited_ci_max_median": [0.30, 0.60],
        "frozen_ci_winner_median (labeled: frozen-at-winner)": [0.35, 0.55],
        "champion_vs_incumbent_conditional_ceiling_interval": [0.4, 0.6],
        "note_correlated_arms": "arms share one prefix panel",
    }


def _gate_ladder_doc() -> dict:
    """A schema-true gate_ladder.json (run_ladder output shape)."""
    content = ["cas-pers-con-lr1e5-s42", "imp-pers-con-lr3e5-s42", "imp-pers-con-lr3e5-s137"]
    marker = ["mk-pers-con-lr5e6-s42"]
    per_arm = {a: _ladder_arm(a, "content", 0.2 + 0.05 * i) for i, a in enumerate(content)}
    per_arm |= {a: _ladder_arm(a, "marker", 0.1) for a in marker}
    fams = {
        "persona": {
            "n_arms": 3,
            "rows_per_arm": [10, 10, 10],
            "across_arm_median": {m: 0.3 + 0.02 * i for i, m in enumerate(_METRICS)},
            "winner": "gate_wwt",
        },
        "tiny_family": {"skipped": "fewer than 4 kept rows in every arm"},
    }
    return {
        "regime": {"layer": 19, "partial": False, "smoke": True},
        "entry_asserts": {},
        "anchors_context": {"whitened_banked_median_rho": 0.1751},
        "per_arm": per_arm,
        "champion": {
            "content": {"dv_change": _champion("dv_change", content)},
            "marker": {"dv_dlogp": _champion("dv_dlogp", marker)},
        },
        "pairwise_win_counts": {
            "content": {"gate_wwt_vs_gate_sigma_inv": {"wins": 3, "n_arms": 3, "per_arm": {}}}
        },
        "per_family_win_table": {
            "families": fams,
            "metrics": list(_METRICS),
            "min_rows_per_family": 4,
        },
        "h2": {"note": "fixture"},
    }


def _route_summary(name: str, med: float) -> dict:
    """One _route_metrics summary (fitted / index-aligned shapes)."""
    return {
        "route": name,
        "r2_unconditional": {"median": med, "mean": med, "frac_positive": 0.6, "n_nan": 0},
        "hurdle": {
            "firing_auroc_median": 0.7,
            "firing_auroc_n_nan": 0,
            "conditional_magnitude_r2_median": 0.2,
            "conditional_magnitude_r2_n_nan": 0,
            "note": "fixture",
        },
        "pr_at_k": {
            "k": 3,
            "precision_at_k": 0.5,
            "recall_at_k": 0.4,
            "n_rows": 8,
            "n_rows_zero_true": 0,
        },
    }


def _banked_route(name: str, med: float) -> dict:
    """One _banked_route_summary record (banked #2476 instrument join)."""
    return {
        "route": name,
        "source": "data/banked.npz",
        "n_banked": 2150,
        "n_in_union": 5,
        "banked_alive_floor": 1200,
        "banked_n_fit_rows": 120_000,
        "r2_median_on_intersection": med,
        "r2_mean_on_intersection": med,
        "note": "fixture",
    }


def _write_leg4(root: Path) -> None:
    """leg4/feature_map_metrics.json + perfeature_leg4.npz (phase_feature_map shape)."""
    leg4 = root / "leg4"
    leg4.mkdir(parents=True)
    n_feat = 6
    feat_ids = np.array([10, 2000, 3000, 9000, 20000, 60000], np.int64)
    rng = np.random.default_rng(0)
    np.savez(
        leg4 / "perfeature_leg4.npz",
        feat_ids=feat_ids,
        r2_fitted_map=rng.uniform(0.0, 0.6, n_feat).astype(np.float32),
        auroc_fitted_map=rng.uniform(0.4, 0.9, n_feat).astype(np.float32),
        cond_r2_fitted_map=rng.uniform(-0.2, 0.4, n_feat).astype(np.float32),
        r2_index_aligned_ib=rng.uniform(-0.3, 0.05, n_feat).astype(np.float32),
        auroc_index_aligned_ib=rng.uniform(0.4, 0.6, n_feat).astype(np.float32),
        cond_r2_index_aligned_ib=rng.uniform(-0.3, 0.1, n_feat).astype(np.float32),
        r2_train_mean_null=np.zeros(n_feat, np.float32),
        shuffle_null_r2_fitted=rng.uniform(-0.5, 0.0, (3, n_feat)).astype(np.float16),
        shuffle_null_r2_ib=rng.uniform(-0.5, 0.0, (3, n_feat)).astype(np.float16),
        activity_te=rng.uniform(0.01, 0.9, n_feat).astype(np.float32),
    )
    fitted = _route_summary("fitted_map", 0.35)
    fitted["fit_meta"] = {
        "floor_frac": 0.01,
        "floor_rows": 1200,
        "d_alive_ctx": 4,
        "selected_lambda": 0.1,
        "val_r2_at_selected": 0.3,
        "lambda_grid_edge": None,
        "widenings": 0,
        "r2_median": 0.35,
    }
    ib = _route_summary("index_aligned_ib", -0.05)
    ib["label"] = "index-aligned null"
    doc = {
        "n_fit": 64,
        "n_val": 8,
        "n_te": 16,
        "n_union": n_feat,
        "union_source": "issue2476_turnavg/analysis_tensors/eval/alive_c.npz",
        "union_floor_frac": 0.002,
        "realized_l0_k": 3,
        "routes": [
            fitted,
            _banked_route("composed_banked_2476", 0.22),
            _banked_route("dense_input_banked_2476", 0.40),
            ib,
            {
                "route": "train_mean_null",
                "r2_unconditional": {"median": 0.0, "mean": 0.0, "n_nan": 0},
                "note": "constant predictor",
            },
        ],
        "floor_sweep": [fitted["fit_meta"]],
        "knn_retrieval": {"euclidean": dict(_KNN), "cosine": dict(_KNN)},
        "shuffle_null": {"n_draws": 3, "seeds": [1, 2, 3], "convention": "fixture"},
        "grain_matching": {"inputs": "ctx", "targets": "ans", "headline_rule": "fixture"},
        "production": False,
        "regime_config_hash": "deadbeef",
    }
    (leg4 / "feature_map_metrics.json").write_text(json.dumps(doc))


def _pair_row(k: int, dva: float) -> dict:
    """One kernel_pairs.json pair row (phase_mine _pair_row shape)."""
    return {
        "row_i": k,
        "row_j": k + 100,
        "ci_i": k,
        "ci_j": k + 100,
        "split_i": "train",
        "split_j": "train",
        "source_i": "lmsys",
        "source_j": "wildchat",
        "ans_len_i": 120,
        "ans_len_j": 130,
        "dc_norm": 5.0 + 0.1 * k,
        "kappa": 0.01 * (k + 1),
        "stratum": k % 3,
        "dva_norm": dva,
    }


def _write_leg8(root: Path) -> None:
    """leg8/mining_summary.json + kernel_pairs.json (phase_mine shapes)."""
    leg8 = root / "leg8"
    leg8.mkdir(parents=True)
    rng = np.random.default_rng(1)
    pairs = []
    for k in range(12):
        row = _pair_row(k, float(rng.uniform(1.0, 2.0)))
        ctrl = _pair_row(k + 50, float(rng.uniform(1.5, 2.5)))
        ctrl["matched_tol"] = 0.02
        row["control"] = ctrl
        pairs.append(row)
    selection = {
        "n_sampled": 1000,
        "n_dedup": 990,
        "n_eligible": 495,
        "dc_norm_median": 5.5,
        "kappa_bottom_decile_edge": 0.02,
        "kappa_mid_quintile": [0.05, 0.08],
        "n_kernel_selected": 12,
        "n_matched": 12,
        "n_dropped_no_control": 0,
        "per_tol_matched_cum": {"0.02": 12},
        "n_kernel_in_bottom_decile": 12,
    }
    (leg8 / "kernel_pairs.json").write_text(
        json.dumps(
            {
                "selection": selection,
                "in_sample_share_selected_rows": 1.0,
                "in_sample_note": "fixture",
                "narration_scope": "fixture",
                "pairs": pairs,
            }
        )
    )
    (leg8 / "mining_summary.json").write_text(
        json.dumps(
            {
                "n_pairs_sampled": 1000,
                "chunk": 100,
                "seed": 2569,
                "b1_assert_iii_probe": {"max_abs_diff": 1e-9},
                "selection": selection,
                "ratio_stats": {
                    "estimator_pinned": "median_of_paired_ratios",
                    "median_of_paired_ratios": 0.82,
                    "ratio_of_medians_companion": 0.85,
                    "n_pairs": 12,
                    "n_zero_dropped": 0,
                    "kernel_dva_median": 1.5,
                    "control_dva_median": 1.9,
                },
                "clustered_bootstrap": {
                    "draws": 100,
                    "n_clusters": 12,
                    "n_units": 12,
                    "ci95": [0.7, 0.95],
                    "n_empty_draws": 0,
                    "seed": 2570,
                    "estimator": "median_of_paired_ratios (weighted median per draw)",
                },
                "residual_floor": {
                    "n_rows": 40,
                    "n_pairs": 780,
                    "q10": 0.8,
                    "q25": 1.0,
                    "q50": 1.2,
                    "q75": 1.5,
                    "q90": 1.8,
                },
                "floor_read": {
                    "kernel_dva_median_over_floor_q50": 1.25,
                    "control_dva_median_over_floor_q50": 1.58,
                    "note": "fixture",
                },
                "ans_len_strata": {
                    "n_rows_unknown_len": 0,
                    "n_rows_total": 200,
                    "decile_edges": [0.0, 10.0],
                },
                "production": False,
                "regime_config_hash": "deadbeef",
            }
        )
    )


def _write_der(root: Path, accuracy: float | None = 0.55) -> None:
    """der/der_eval.json (phase_der_eval doc shape)."""
    der = root / "der"
    der.mkdir(parents=True, exist_ok=True)
    (der / "der_eval.json").write_text(
        json.dumps(
            {
                "judge": {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "path": "sync",
                },
                "description_source": {"mode": "reuse-probe", "path": "x"},
                "describe_stage": {"note": "reuse branch"},
                "coverage": {
                    "n_union": 6,
                    "n_described_in_union": 4,
                    "union_coverage": 0.667,
                    "pred_topk_described_frac": 0.8,
                    "n_eligible_rows": 12,
                    "n_rows_skipped_undersize": 1,
                    "feats_per_list": 3,
                },
                "matching": {
                    "n_way": 10,
                    "chance": 0.1,
                    "n_items": 40,
                    "n_answered": 38,
                    "n_correct": 21,
                    "accuracy": accuracy,
                    "dropped_by_category": {"parse": 2},
                    "drop_policy": "fixture",
                },
                "budget": {"cap": 2000, "describe_calls": 0, "matching_calls": 40},
                "scoping": "fixture",
                "n_te": 16,
                "regime_config_hash": "deadbeef",
            }
        )
    )


def _erank(seed: int) -> dict:
    """One effective_rank_summaries record (issue2569_dw_fleet shape)."""
    rng = np.random.default_rng(seed)
    return {
        "stable_rank": float(rng.uniform(1.5, 20.0)),
        "participation_ratio": float(rng.uniform(1.0, 30.0)),
        "top1_share_energy": float(rng.uniform(0.1, 0.9)),
        "top1_share_sv": float(rng.uniform(0.1, 0.6)),
        "frobenius": 1.0,
        "spectral": 0.5,
        "n_svals": 32,
    }


def _dv3_payload(obs: float, p95: float) -> dict:
    """One dv3_payload_from_null record for arm name 'write' (#650 nested schema)."""
    return {
        "observed": {"write": {"max_by_layer": {"19": obs}, "band_max": obs, "verdict": "x"}},
        "null": {
            "write": {
                "per_layer_max_draws": {"19": [p95 * 0.8]},
                "band_max_draws": [p95 * 0.8],
                "band_p95": p95,
                "n_draws": 20,
                "null_aggregation": "max_over_base_singular_vectors_then_max_over_band",
            }
        },
        "assertions": {"null_aggregation_matches_observed": True},
    }


def _align_cell(cos: float, p95: float) -> dict:
    """One alignment_vs_null record (issue2569_dw_fleet shape)."""
    return {
        "max_abs_cos": cos,
        "null_p95": p95,
        "n_draws": 20,
        "above_null": cos > p95,
        "null_aggregation": "max_over_base_singular_vectors_then_max_over_band",
        "k_basis": 8,
    }


def _write_dw(root: Path) -> None:
    """dw_fleet/{fleet_table,lora,ft,alignment} (issue2569_dw_fleet shapes)."""
    dw = root / "dw_fleet"
    (dw / "lora").mkdir(parents=True)
    (dw / "ft").mkdir(parents=True)
    lora = {
        "arm_id": "cas-pers-con-lr1e5-s42",
        "method": "lora",
        "modules": {
            "down_proj": {"19": _erank(1), "20": _erank(2)},
            "up_proj": {"19": _erank(3)},
        },
        "intruder": {"down_proj": _dv3_payload(0.21, 0.08), "o_proj": _dv3_payload(0.05, 0.09)},
        "regime_key": "rk",
        "metadata": {},
    }
    (dw / "lora" / "cas-pers-con-lr1e5-s42.json").write_text(json.dumps(lora))
    ft = {
        "arm_id": "cas-pers-ft-con-s42",
        "method": "ft",
        "matrices": {f"L{i}.q_proj": _erank(10 + i) for i in range(4)},
        "n_matrices": 4,
        "regime_key": "rk",
        "metadata": {},
    }
    (dw / "ft" / "cas-pers-ft-con-s42.json").write_text(json.dumps(ft))
    align = {
        "layer": 19,
        "arms": {
            "cas-pers-con-lr1e5-s42": {
                "factors": {
                    "L19.down_proj": {
                        "r_B[evil]": _align_cell(0.3, 0.1),
                        "Ar[evil]": _align_cell(0.05, 0.1),
                        "delta_tbar": _align_cell(0.5, 0.1),
                        "c_C": {"skipped": "dim mismatch 3584 vs 4096"},
                    }
                }
            },
            "imp-pers-con-lr3e5-s42": {
                "factors": {"L19.down_proj": {"r_B[evil]": _align_cell(0.12, 0.1)}}
            },
        },
        "seed_noise_anchor": {
            "pair": ["imp-pers-con-lr3e5-s42", "imp-pers-con-lr3e5-s137"],
            "note": "no full-FT seed pair exists (scope limit)",
            "down_proj": {"top1_abs_cos": 0.15, "max_abs_cos_topk": 0.2},
        },
        "metadata": {},
    }
    (dw / "alignment.json").write_text(json.dumps(align))


def _leg6_unit(arm: str, rank: int) -> dict:
    """One fit_split_half unit record (issue2569_leg6 shape)."""
    rng = np.random.default_rng(rank)
    s1 = np.sort(rng.uniform(0.05, 2.0, 16))[::-1]
    return {
        "n_rows": 200,
        "n_half": [100, 100],
        "d": 8,
        "lambda_rel_selected": 0.01,
        "lambda_abs": [0.5, 0.6],
        "lambda_grid_params": ["log10", -6.0, 2.0, 17],
        "lambda_grid_edge": False,
        "heldout_r2": {"fit1_eval2": 0.15 + 0.01 * rank, "fit2_eval1": 0.12},
        "shuffle_threshold_p95": [0.4, 0.45],
        "shuffle_top_sv_draws": [[0.3] * 5, [0.35] * 5],
        "n_shuffle_draws": 5,
        "cos_floor": 0.5,
        "factor_matches": [
            {
                "i1": i,
                "j2": i,
                "factor_cos": 0.8 - 0.1 * i,
                "matched": i < rank,
                "s1": float(s1[i]),
                "s2": float(s1[i] * 0.9),
            }
            for i in range(4)
        ],
        "denoised_rank": rank,
        "gavish_donoho_reference_count": rank + 1,
        "singular_values_half1": [float(x) for x in s1],
        "singular_values_half2": [float(x * 0.95) for x in s1],
        "identity_bias_r2": -0.4,
        "knn_retrieval": {**_KNN, "chance": {"1": 0.01, "5": 0.05, "10": 0.1}},
        "arm": arm,
        "layer": 19,
        "context_convention": "last_prompt",
        "regime_key": "rk",
        "metadata": {},
    }


def _write_leg6(root: Path) -> None:
    """leg6/<arm>/L19_last_prompt.json unit records + a pooled dir to ignore."""
    for arm, rank in (("cas-pers-con-lr1e5-s42", 3), ("syc-bare-con-lr1e5-s42", 1)):
        d = root / "leg6" / arm
        d.mkdir(parents=True)
        (d / "L19_last_prompt.json").write_text(json.dumps(_leg6_unit(arm, rank)))
        (d / "guard.json").write_text(json.dumps({"arm": arm, "guard": {"action": "proceed"}}))
    pooled = root / "leg6" / "pooled" / "cas-pers-con-lr1e5-s42"
    pooled.mkdir(parents=True)
    (pooled / "L19_last_prompt.json").write_text(
        json.dumps({**_leg6_unit("cas-pers-con-lr1e5-s42", 2), "target_arm": "x"})
    )


def _tier1_pair(qL: int, lL: int, base: float) -> dict:
    """One tier-1 grid pair record (issue2569_atlas.phase_fits shape)."""
    fits = {}
    for k, off in (("vc_q2l", 0.0), ("vc_l2q", 0.01), ("va_q2l", 0.02), ("va_l2q", 0.03)):
        fits[k] = {
            "name": f"align_{k}_L{qL}_{lL}",
            "d_in": 8,
            "d_out": 8,
            "n_train": 72,
            "n_val": 8,
            "n_test": 8,
            "fit_meta": {"selected_lambda": 0.1, "val_r2_at_selected": base + off},
            "test_r2": base + off,
            "knn": dict(_KNN),
            "identity_bias": {"applicable": False, "reason": "d_in != d_out"},
            "elapsed_s": 0.1,
        }
    return {
        "qwen_layer": qL,
        "llama_layer": lL,
        "fits": fits,
        "cka": {"vc": base + 0.1, "va": base + 0.05},
        "selection_val_r2_mean": base + 0.015,
    }


def _write_leg7(root: Path) -> None:
    """leg7/three_tier.json + atlas_distances.json (issue2569_atlas shapes)."""
    leg7 = root / "leg7"
    leg7.mkdir(parents=True)
    routes = {
        "native": {"r2": 0.62, "knn": dict(_KNN)},
        "composed_banked": {"r2": 0.48, "knn": dict(_KNN)},
        "composed_matched": {"r2": 0.51, "knn": dict(_KNN)},
        "alignment_only_baseline": {"r2": 0.30, "knn": dict(_KNN)},
        "a_qwen_source": {"layer": 19, "path": "x", "selected_lambda": 0.001},
    }
    (leg7 / "three_tier.json").write_text(
        json.dumps(
            {
                "issue": 2569,
                "claim_scope": "fixture",
                "realized_paired_rows": 100,
                "working_pair": {"qwen_layer": 19, "llama_layer": 22, "selection": "fixture"},
                "tier1_alignability": {
                    "grid": [_tier1_pair(14, 16, 0.5), _tier1_pair(19, 22, 0.6)],
                    "note": "fixture",
                },
                "tier2_operator_similarity": {
                    "routes": routes,
                    "correspondence_test": {
                        "raw_cosine": 0.4,
                        "rotation_null": {
                            "n_draws": 10,
                            "null_mean": 0.0,
                            "null_std": 0.01,
                            "null_p975": 0.02,
                            "analytic_sd_1_over_d": 0.125,
                        },
                        "agree_above_rotation_null": True,
                        "consequence": "fixture",
                        "statistic_class": "direction-aware",
                    },
                    "anchor_825_aligned_cosine": 0.6864,
                    "note": "fixture",
                },
                "tier3_diagnostics": {
                    "label": "fixture",
                    "r2_native_minus_composed_banked": 0.14,
                    "r2_native_minus_composed_matched": 0.11,
                    "r2_alignment_only_baseline": 0.30,
                },
                "corpus_transfer": {
                    "corpus_fold": {"llama_native_r2": 0.55, "align_c_q2l_r2": 0.45}
                },
                "working_pair_fits": {},
                "metadata": {},
            }
        )
    )
    names = ["n1m_L19", "n1m_L14", "llama_native_L22"]
    table = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            entry = {
                "pair": [names[i], names[j]],
                "bases": ["qwen", "qwen"],
                "spectrum": {
                    "spectrum_cosine": 0.9,
                    "truncated": False,
                    "k": 8,
                    "statistic_class": "spectrum/rotation-invariant-only (descriptive ceiling)",
                },
                "cosine": {
                    "raw_cosine": 0.5 + 0.1 * i,
                    "rotation_null": {
                        "n_draws": 10,
                        "null_mean": 0.0,
                        "null_std": 0.01,
                        "null_p975": 0.02,
                        "analytic_sd_1_over_d": 0.125,
                    },
                    "statistic_class": "direction-aware (raw cosine vs rotation null; same basis)",
                },
            }
            if j == 2:  # one spectrum-fallback pair exercises the marked-cell branch
                entry["cosine"] = None
            table.append(entry)
    (leg7 / "atlas_distances.json").write_text(
        json.dumps(
            {
                "issue": 2569,
                "rows": [
                    {
                        "name": n,
                        "basis": "qwen" if "n1m" in n else "llama",
                        "shape": [8, 8],
                        "floor": {"floor": 0.97, "n_half": [10, 10]} if i == 0 else None,
                        "floor_label": "split-half refit" if i == 0 else "no floor",
                        "source": "fixture",
                        "procrustes_aligned": False,
                    }
                    for i, n in enumerate(names)
                ],
                "dropped_rows": [{"name": "passb_L14", "reason": "unresolved at P-E entry"}],
                "distance_table": table,
                "procrustes": {"available": False, "n_rows": 0},
                "mds_2d": {
                    "coords": {n: [float(i), float(-i)] for i, n in enumerate(names)},
                    "note": "presentation-only",
                    "spectrum_fallback_pairs": [t["pair"] for t in table if t["cosine"] is None],
                },
                "anchor_825_aligned_cosine": 0.6864,
                "metadata": {},
            }
        )
    )


def _write_leg2(root: Path) -> None:
    """leg2/gate_ladder.json + leg2_curve/learning_curve.json."""
    (root / "leg2").mkdir(parents=True)
    (root / "leg2" / "gate_ladder.json").write_text(json.dumps(_gate_ladder_doc()))
    (root / "leg2_curve").mkdir(parents=True)
    (root / "leg2_curve" / "learning_curve.json").write_text(json.dumps(_learning_curve_doc()))


@pytest.fixture()
def full_root(tmp_path: Path) -> Path:
    """A results root carrying every landed-producer artifact, schema-true."""
    root = tmp_path / "eval"
    root.mkdir()
    _write_leg2(root)
    _write_leg4(root)
    _write_leg8(root)
    _write_der(root)
    _write_dw(root)
    _write_leg6(root)
    _write_leg7(root)
    return root


def _n_artists(fig) -> int:
    """Total plotted artists across visible axes (a blank render scores 0)."""
    n = 0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        n += len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.images)
    return n


def _assert_rendered(fig_dir: Path, stem: str) -> None:
    """PNG + sidecar landed and the PNG has non-trivial byte size."""
    png = fig_dir / f"{stem}.png"
    assert png.is_file(), f"missing {png}"
    assert png.stat().st_size > 4_000, f"suspiciously small render: {png}"
    assert (fig_dir / f"{stem}.meta.json").is_file(), f"missing sidecar for {stem}"


# ---------------------------------------------------------------------------
# Label map / helpers
# ---------------------------------------------------------------------------


def test_display_map_covers_gate_metrics_and_routes():
    """Every gate rung and leg-4/leg-7 route slug has a plain-English label."""
    for m in F.GATE_METRIC_ORDER:
        assert F.display(m) != m
    for r in (
        "fitted_map",
        "composed_banked_2476",
        "index_aligned_ib",
        "train_mean_null",
        "native",
        "composed_matched",
        "alignment_only_baseline",
    ):
        assert F.display(r) != r


def test_arm_label_grammar_and_dedup():
    """Arm ids parse to plain English; colliding labels gain seed suffixes."""
    assert F.arm_label("cas-pers-con-lr1e5-s42") == "Casualness, persona, contrastive"
    assert F.arm_label("mk-pers-ft-con-s42") == "Marker, persona, contrastive (full FT)"
    assert F.arm_label("not-a-fleet-arm") == "not-a-fleet-arm"
    lab = F.arm_labels_deduped(["imp-pers-con-lr3e5-s42", "imp-pers-con-lr3e5-s137"])
    assert lab["imp-pers-con-lr3e5-s42"].endswith("(seed 42)")
    assert lab["imp-pers-con-lr3e5-s137"].endswith("(seed 137)")


def test_tier_of_bounds():
    """Tier assignment follows the #2476 matryoshka prefix bounds exactly."""
    ids = np.array([0, 2047, 2048, 16383, 16384, 65535])
    assert F.tier_of(ids).tolist() == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# Per-figure builds + renders
# ---------------------------------------------------------------------------


def test_learning_curve_builds_and_renders(full_root, tmp_path):
    """Curve figure carries series artists and renders with a sidecar."""
    doc = json.loads((full_root / "leg2_curve" / "learning_curve.json").read_text())
    fig = F.build_learning_curve(doc)
    assert _n_artists(fig) >= 6  # empirical + theory + companions + band lines
    import matplotlib.pyplot as plt

    plt.close(fig)
    stem = F.fig_leg2_learning_curve(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)


def test_gate_ladder_content_and_marker_render(full_root, tmp_path):
    """Both arm kinds render; the content panel carries per-arm rung points."""
    doc = json.loads((full_root / "leg2" / "gate_ladder.json").read_text())
    fig = F.build_gate_ladder(doc, "content")
    assert _n_artists(fig) > len(F.GATE_METRIC_ORDER)
    import matplotlib.pyplot as plt

    plt.close(fig)
    for fn, stem_want in (
        (F.fig_leg2_gate_ladder_content, "leg2_gate_ladder_content"),
        (F.fig_leg2_gate_ladder_marker, "leg2_gate_ladder_marker"),
        (F.fig_leg2_gate_family_table, "leg2_gate_family_table"),
    ):
        stem = fn(full_root, tmp_path / "figs")
        assert stem == stem_want
        _assert_rendered(tmp_path / "figs", stem)


def test_leg4_routes_and_per_feature_render(full_root, tmp_path):
    """Route bars + per-tier panel + AUROC hist render; per-feature scatter too."""
    stem = F.fig_leg4_routes(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    stem2 = F.fig_leg4_per_feature(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem2)
    with np.load(full_root / "leg4" / "perfeature_leg4.npz") as npz:
        fig = F.build_leg4_per_feature(npz)
        assert _n_artists(fig) >= 3  # one series per populated tier
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_leg8_kernel_renders(full_root, tmp_path):
    """Kernel-pair figure renders ECDFs, paired scatter, and the ratio panel."""
    summary = json.loads((full_root / "leg8" / "mining_summary.json").read_text())
    pairs = json.loads((full_root / "leg8" / "kernel_pairs.json").read_text())
    fig = F.build_leg8_kernel(summary, pairs)
    assert _n_artists(fig) >= 8
    import matplotlib.pyplot as plt

    plt.close(fig)
    stem = F.fig_leg8_kernel_pairs(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)


def test_der_matching_renders_and_guards_none(full_root, tmp_path):
    """Accuracy/coverage bars render; a no-answer document raises (skip path)."""
    stem = F.fig_der_matching(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    _write_der(full_root, accuracy=None)
    with pytest.raises(ValueError, match="no answered matching items"):
        F.fig_der_matching(full_root, tmp_path / "figs")


def test_dw_figures_render(full_root, tmp_path):
    """Effective-rank, intruder, and alignment figures render from unit records."""
    for fn in (F.fig_dw_effective_rank, F.fig_dw_intruder, F.fig_dw_alignment):
        stem = fn(full_root, tmp_path / "figs")
        _assert_rendered(tmp_path / "figs", stem)
    lora = F._load_dw_units(full_root, "lora")
    fig = F.build_dw_intruder(lora)
    assert _n_artists(fig) >= 3  # parity line + 2 labeled points
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_leg6_figures_render_and_skip_pooled(full_root, tmp_path):
    """Rank bars + spectra render; pooled/ unit records never enter the per-arm set."""
    units = F._leg6_units(full_root / "leg6", "last_prompt", 19)
    assert sorted(units) == ["cas-pers-con-lr1e5-s42", "syc-bare-con-lr1e5-s42"]
    for fn in (F.fig_leg6_ranks, F.fig_leg6_spectra):
        stem = fn(full_root, tmp_path / "figs")
        _assert_rendered(tmp_path / "figs", stem)


def test_leg7_figures_render(full_root, tmp_path):
    """Three-tier panels + atlas heatmap/MDS render; fallback pair is marked."""
    stem = F.fig_leg7_three_tier(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    doc = json.loads((full_root / "leg7" / "atlas_distances.json").read_text())
    fig = F.build_atlas(doc)
    assert _n_artists(fig) >= 4  # heatmap image + fallback markers + MDS points
    import matplotlib.pyplot as plt

    plt.close(fig)
    stem = F.fig_leg7_atlas(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)


# ---------------------------------------------------------------------------
# Batch driver + manifest
# ---------------------------------------------------------------------------


def test_render_all_full_root(full_root, tmp_path):
    """Every registered figure renders from the full fixture root; manifest lands."""
    fig_dir = tmp_path / "figs"
    manifest = F.render_all(full_root, fig_dir)
    assert sorted(manifest["rendered"]) == sorted(F.FIGURES)
    assert manifest["skipped"] == {}
    assert set(manifest["deferred_no_producer"]) == set(F.DEFERRED_NO_PRODUCER)
    on_disk = json.loads((fig_dir / "figures_manifest.json").read_text())
    assert on_disk["rendered"] == manifest["rendered"]
    assert on_disk["metadata"]["issue"] == 2569
    for stem in manifest["rendered"].values():
        _assert_rendered(fig_dir, stem)


def test_render_all_empty_root_skips_everything(tmp_path):
    """An empty root records one named skip per figure and still writes the manifest."""
    fig_dir = tmp_path / "figs"
    manifest = F.render_all(tmp_path / "nothing", fig_dir)
    assert manifest["rendered"] == {}
    assert sorted(manifest["skipped"]) == sorted(F.FIGURES)
    assert (fig_dir / "figures_manifest.json").is_file()


def test_render_all_only_subset(full_root, tmp_path):
    """--only narrows the batch to the named figures."""
    manifest = F.render_all(full_root, tmp_path / "figs", only={"leg7_atlas"})
    assert sorted(manifest["rendered"]) == ["leg7_atlas"]
