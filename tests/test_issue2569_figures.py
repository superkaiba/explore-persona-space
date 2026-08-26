"""Issue #2569 figure producers: synthetic-fixture smokes through ``savefig_paper``.

Every fixture mirrors the EXACT schema of the landed producing driver (read off
the worktree at 57808a6434; the leg6 cross_arm cell + the weights split-half /
criterion records read at fb52ed2804): ``issue2569_gateladder`` (ladder +
curve), ``issue2569_rowbattery`` (leg4 / leg8 / der), ``issue2569_dw_fleet``,
``issue2569_leg6``, ``issue2569_weights``, and ``issue2569_atlas``. Each test drives a REAL figure
function end-to-end (matplotlib Agg) and asserts BOTH that the built figure
carries plotted artists (a silently-empty render fails here, not at review) and
that the PNG + ``.meta.json`` sidecar landed with non-trivial size. Dimensions
stay tiny throughout (no dense 3584^2 anywhere).

Every test runs under the PRODUCTION rcParams regime (the autouse
``_production_style`` fixture calls ``set_paper_style("blog")``): the blog
style zeroes ``lines.markeredgewidth``, so an open marker (``mfc="none"``)
with no explicit ``mew=`` draws ZERO ink there while matplotlib defaults
(mew=1.0) render it fine -- a suite running under defaults validated renders
nobody ships (round-2 blocker: whole series invisible with 24 greens).
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


@pytest.fixture(autouse=True)
def _production_style():
    """Pin the PRODUCTION rcParams regime (set_paper_style("blog")) per test.

    render_all() sets this style in production, and it is where the
    markeredgewidth=0 open-marker trap lives; rc_context restores the global
    rcParams after each test so nothing leaks to other test files.
    """
    import matplotlib

    with matplotlib.rc_context():
        F.set_paper_style("blog")
        yield


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


def _learning_curve_doc(smoke: bool = False) -> dict:
    """A schema-true learning_curve.json document (curve_core output shape).

    ``smoke`` mirrors the artifact's own ``regime.smoke`` flag (the real
    producer always writes it — probed on
    ``smoke-final/leg2_curve_cli/learning_curve.json``): the figure keys its
    registered-band rendering on it, so both regimes need a fixture.
    """
    pts = [
        _verdict_point(4_500, 0.62, 0.60),
        _verdict_point(50_000, 0.70, 0.69),
        _verdict_point(500_000, 0.75, 0.76),
    ]
    d = 3_584
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
            "smoke": smoke,
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
            "well_posedness": {
                "d": d,
                "rule": "n_train < d is estimator-degenerate (#1701)",
                "per_point": [
                    {
                        "label": p["label"],
                        "n_train": p["n_train"],
                        "d": d,
                        "n_over_d": p["n_train"] / d,
                        "well_posed": True,
                    }
                    for p in pts
                ],
                "degenerate_excluded": [],
            },
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


def _dv3_payload(obs: float, p95: float, arm: str = "write") -> dict:
    """One dv3_payload_from_null record (#650 nested schema).

    ``arm`` mirrors the producer's residual-side arm name
    (``intruder_read(..., arm_name="write" if side == "U" else "read")`` —
    probed off a real ``analyze_lora_arm`` record 2026-08-26): o_proj/down_proj
    carry ``write``; q/k/v/gate/up carry ``read``.
    """
    return {
        "observed": {arm: {"max_by_layer": {"19": obs}, "band_max": obs, "verdict": "x"}},
        "null": {
            arm: {
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


def _align_factor(side: str, alignments: dict) -> dict:
    """One cmd_align per-module factors record — the NESTED schema the producer
    writes (``{"side", "k_basis", "alignments": {...}}``; probed off a real
    ``cmd_align`` run 2026-08-26, fix-round-3
    ``dwfleet-alignment-consumer-flat-schema-stale``)."""
    return {"side": side, "k_basis": 8, "alignments": alignments}


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
        # Arm name is the module's RESIDUAL side: o/down are U-side ("write"),
        # q/k/v/gate/up are V-side ("read") — one record carries BOTH arms.
        "intruder": {
            "down_proj": _dv3_payload(0.21, 0.08),
            "o_proj": _dv3_payload(0.05, 0.09),
            "q_proj": _dv3_payload(0.11, 0.09, arm="read"),
        },
        "intruder_side": {"down_proj": "U", "o_proj": "U", "q_proj": "V"},
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
                    "L19.down_proj": _align_factor(
                        "U",
                        {
                            "r_B[evil]": _align_cell(0.3, 0.1),
                            "Ar[evil]": _align_cell(0.05, 0.1),
                            "delta_tbar": _align_cell(0.5, 0.1),
                        },
                    ),
                    # V-side module whose only readable direction (c_C) is
                    # absent: the producer writes an EMPTY alignments dict.
                    "L19.q_proj": _align_factor("V", {}),
                }
            },
            "imp-pers-con-lr3e5-s42": {
                "factors": {
                    "L19.down_proj": _align_factor("U", {"r_B[evil]": _align_cell(0.12, 0.1)})
                }
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


def _cross_match(i: int, j: int, fcos: float, above_sym: bool, floor: float | None) -> dict:
    """One cross-arm match record (issue2569_leg6.run_cross_arm shape)."""
    return {
        "factor_a": i,
        "factor_b": j,
        "cos_context": fcos + 0.03,
        "cos_shift": fcos,
        "factor_cos": fcos,
        "above_symmetric_null": above_sym,
        "above_rotation_null_percomparison": above_sym,
        "within_agreement_a": 0.9,
        "within_agreement_b": floor,
        "splithalf_floor": floor,
        "above_splithalf_floor": bool(floor is not None and fcos >= floor),
        "sigma_a": 1.2,
        "sigma_b": 1.0,
    }


def _write_cross_arm(root: Path) -> None:
    """leg6/cross_arm/{L19_last_prompt.json,summary.json} (run_cross_arm shape).

    One admissible same-behavior pair (matches above AND below the symmetric
    band), one refused basis-mismatched pair, one admissible cross-behavior
    pair, one skipped arm.
    """
    cross = root / "leg6" / "cross_arm"
    cross.mkdir(parents=True)
    arms = ["cas-pers-con-lr1e5-s42", "cas-bare-con-lr1e5-s42", "syc-bare-con-lr1e5-s42"]
    sym_key = "8x8|ra2|rb2"
    note = "winner's-curse-inflated point estimate (max over greedy matches; fixture)"
    pairs = [
        {
            "arm_a": arms[0],
            "arm_b": arms[1],
            "same_behavior": True,
            "admissible": True,
            "matches": [
                _cross_match(0, 0, 0.82, True, 0.88),
                _cross_match(1, 1, 0.35, False, None),
            ],
            "max_matched_cos": 0.82,
            "max_matched_cos_note": note,
            "symmetric_null_key": sym_key,
            "above_symmetric_null_any": True,
            "null_aggregation_matches_observed": True,
        },
        {
            "arm_a": arms[0],
            "arm_b": arms[2],
            "same_behavior": False,
            "admissible": False,
            "refusal_reason": (
                "factor_bases mismatch on ['context'] - recorded skip, no number fabricated"
            ),
        },
        {
            "arm_a": arms[1],
            "arm_b": arms[2],
            "same_behavior": False,
            "admissible": True,
            "matches": [_cross_match(0, 1, 0.41, False, 0.9)],
            "max_matched_cos": 0.41,
            "max_matched_cos_note": note,
            "symmetric_null_key": sym_key,
            "above_symmetric_null_any": False,
            "null_aggregation_matches_observed": True,
        },
    ]
    cell = {
        "layer": 19,
        "context_convention": "last_prompt",
        "factor_half": "half1 (leading denoised prefix per arm)",
        "matching_rule": "greedy by min(|cos_context|, |cos_shift|) (fixture)",
        "factor_orientation": "row-vector map (fixture)",
        "statistic_classes": {
            "cross_arm_factor_cosine": "direction-aware (raw factor cosine; fixture)",
            "symmetric_null": "selection-symmetric max-matched rotation null (fixture)",
            "rotation_null_percomparison": "two-sided random-rotation chance band (fixture)",
            "splithalf_floor": "within-arm split-half agreement (NOISE FLOOR; fixture)",
        },
        "rotation_null_bands_percomparison": {"8": {"null_p975": 0.62}},
        "symmetric_null_bands": {sym_key: {"p95_max_matched": 0.72}},
        "assertions": {"null_aggregation_matches_observed": True},
        "n_null_draws": 5,
        "null_seed": 2569,
        "arms": {
            a: {"denoised_rank": 2, "n_factors_compared": 2, "unit_regime_key": "rk"} for a in arms
        },
        "skipped_arms": [
            {
                "arm": "imp-bare-con-lr1e5-s42",
                "reason": "unit JSON missing (arm halted or unit not run)",
            }
        ],
        "pairs": pairs,
        "criterion": {
            "registered": "leg 6: >=1 cross-arm shared factor above the rotation null (fixture)",
            "shared_factor_definition": "fixture",
            "n_shared_above_null_same_behavior": 1,
            "n_shared_above_null_all_pairs": 1,
            "pairs_above_null_same_behavior": [f"{arms[0]}~{arms[1]}:f0~f0"],
            "met": True,
            "n_same_behavior_pairs_tested": 1,
            "pair_multiplicity_note": "fixture",
            "per_comparison_uncorrected": {
                "label": "UNCORRECTED per-comparison read (fixture; NOT the criterion input)",
                "n_shared_above_percomparison_null_same_behavior": 1,
                "n_shared_above_percomparison_null_all_pairs": 1,
            },
        },
        "regime_key": "cross-rk",
        "metadata": {},
    }
    (cross / "L19_last_prompt.json").write_text(json.dumps(cell))
    (cross / "summary.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "layer": 19,
                        "context_convention": "last_prompt",
                        "n_shared_above_null_same_behavior": 1,
                        "n_shared_above_null_all_pairs": 1,
                        "met": True,
                    }
                ],
                "criterion_met_any_cell": True,
                "metadata": {},
            }
        )
    )


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
    _write_cross_arm(root)


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


_WIRING_CAVEATS = [
    "all wiring claims are map-level (fixture)",
    "sensitivity-analysis scope only (fixture)",
]


def _dash_direction_rows(k: int, rng, sigma, c_align, eigen: bool, write: bool) -> list[dict]:
    """Schema-true sae_dashboards direction rows (issue2569_weights._dash_rows)."""
    rows = []
    for j in range(k):
        row: dict = {
            "rank": j + 1,
            "max_abs_cos": float(rng.uniform(0.08, 0.6)),
            "exceeds_analytic_floor": True,
            "exceeds_empirical_p95": False,
            "top_features": [
                {
                    "feat_id": int(rng.integers(0, 1000)),
                    "cos": 0.3,
                    "label": {
                        "description": "fixture context feature",
                        "confidence": 3,
                        "evidence_side": "ctx",
                        "source": "fixture",
                    },
                }
            ],
        }
        if eigen:
            row.update(
                abs_lambda=1.0 - 0.05 * j,
                is_complex=bool(j % 2),
                im_frac=float(rng.uniform(0.0, 0.8)),
            )
        else:
            row.update(sigma=float(sigma[j]), self_alignment_c=float(c_align[j]))
        if write:
            row["encoder_pass"] = {"n_fired": 2, "fired": [{"feat_id": 5, "act": 0.4}]}
            row["linear_top"] = [{"feat_id": 5, "value": 0.4}]
        rows.append(row)
    return rows


def _null_floor(n_features: int) -> dict:
    """One side of the sae_dashboards null_floors block."""
    return {
        "n_features": n_features,
        "analytic_sqrt_2lnN_over_d": 0.19,
        "empirical": {
            "n_draws": 16,
            "seed": 1,
            "mean": 0.20,
            "p50": 0.20,
            "p90": 0.23,
            "p95": 0.25,
            "p99": 0.30,
            "max": 0.33,
        },
    }


def _attr_feature(why: str, with_label: bool) -> dict:
    """One tabled answer feature (issue2569_weights.attribution_decompose shape)."""
    label = (
        {
            "description": "fixture judged context-feature description",
            "confidence": 4,
            "evidence_side": "ctx",
            "source": "fixture",
        }
        if with_label
        else None
    )
    return {
        "why_in_table": why,
        "pred_act": 1.2,
        "true_act": 0.9,
        "pre_act": 1.2,
        "n_active_ctx": 7,
        "contributions": [
            {"ctx_feat_id": 3, "a_j": 0.5, "edge": 0.2, "contribution": 0.1, "label": label},
            {"ctx_feat_id": 9, "a_j": 0.4, "edge": -0.1, "contribution": -0.04, "label": None},
        ],
        "bias_terms": {
            "ctx_decoder_bias_via_map": 0.01,
            "sae_recon_residual_via_map": 0.0,
            "map_intercept": 0.02,
            "ans_encoder_offset": -0.3,
        },
        "closure_residual": 1e-9,
    }


def _write_weights(root: Path, alive: bool = True, smoke: bool = False) -> None:
    """weights/{leg1,leg3} P-A driver artifacts (issue2569_weights schemas, tiny d).

    ``smoke`` mirrors the wiring artifact's own ``regime.smoke`` flag (the real
    producer always writes a regime block — probed on
    ``smoke-final/weights/leg3/wiring_L19.json``); the H3 verdict lines key on
    it, so both regimes need a fixture.
    """
    import torch

    rng = np.random.default_rng(7)
    d = 48
    leg1 = root / "weights" / "leg1"
    leg1.mkdir(parents=True, exist_ok=True)
    leg3 = root / "weights" / "leg3"
    leg3.mkdir(parents=True, exist_ok=True)
    sigma = np.sort(rng.uniform(0.02, 1.3, size=d))[::-1].copy()
    c_align = rng.uniform(-1.0, 1.0, size=d)
    lam = rng.normal(size=d) + 1j * (rng.random(d) < 0.4) * rng.normal(size=d)
    lam = lam[np.argsort(-np.abs(lam))].astype(np.complex128)
    torch.save(
        {
            "sigma": torch.from_numpy(sigma),
            "self_alignment_c": torch.from_numpy(c_align),
            "eig_lambda": torch.from_numpy(lam),
            "stats": {"rho": float(np.abs(lam).max()), "d": d, "top_k": 8},
        },
        leg1 / "factor_L19.pt",
    )
    tau = 0.12
    labels = np.where(sigma < tau, "ignored", "rotated_scaled")
    total = float((sigma**2).sum())
    classes = {
        lab: {
            "count": int((labels == lab).sum()),
            "frac_count": float((labels == lab).mean()),
            "sigma2_mass_frac": float((sigma[labels == lab] ** 2).sum() / total),
        }
        for lab in ("ignored", "copied", "damped", "transcoded", "rotated_scaled")
    }
    (leg1 / "anatomy_L19.json").write_text(
        json.dumps(
            {
                "tau_kernel": tau,
                "k99": int((sigma >= tau).sum()),
                "k90": 30,
                "tau_k90": 0.3,
                "sigma_max": float(sigma[0]),
                "sigma_median": float(np.median(sigma)),
                "labels": labels.tolist(),
                "sigma": sigma.tolist(),
                "c": c_align.tolist(),
                "classes": classes,
                "top_directions": [
                    {
                        "rank": 1,
                        "sigma": float(sigma[0]),
                        "c": float(c_align[0]),
                        "abs_c": float(abs(c_align[0])),
                        "label": str(labels[0]),
                    }
                ],
                "thresholds": {"copied_gain": [0.8, 1.25]},
                "precedence": "fixture",
                "data_weighted_mass": "deferred-to-P-B (fixture)",
            }
        )
    )
    half1 = rng.uniform(0.2, 1.0, size=d)
    half2 = rng.uniform(0.2, 1.0, size=d)
    stability = np.minimum(half1, half2)
    sh_floor = 0.5
    n_above = int((stability > sh_floor).sum())
    torch.save(
        {
            "stability": torch.from_numpy(stability),
            "sigma_full": torch.from_numpy(sigma),
            **{
                hname: {
                    "factor_cos": torch.from_numpy(h),
                    "cos_u": torch.from_numpy(h),
                    "cos_v": torch.from_numpy(np.minimum(h + 0.05, 1.0)),
                    "partner": torch.arange(d),
                    "sigma_half": torch.from_numpy(sigma * 0.97),
                    "sigma_matched": torch.from_numpy(sigma * 0.97),
                    "n_rows": 400,
                }
                for hname, h in (("half1", half1), ("half2", half2))
            },
            "floor": {"analytic": 0.31, "empirical": {"p99": sh_floor}, "floor": sh_floor},
            "regime": {"fixture": True},
            "metadata": {},
        },
        leg1 / "splithalf_stability_L19.pt",
    )
    (leg1 / "splithalf_stability_L19.json").write_text(
        json.dumps(
            {
                "regime": {"fixture": True},
                "status": "computed",
                "floor": {"analytic": 0.31, "empirical": {"p99": sh_floor}, "floor": sh_floor},
                "n_above_floor": n_above,
                "frac_above_floor": n_above / d,
                "criterion": {
                    "clause": ">= 300 singular directions above the split-half stability "
                    "floor (plan SS7.5)",
                    "metric": "n_above_floor",
                    "threshold": 300,
                    "value": n_above,
                    "pass": bool(n_above >= 300),
                },
                "stability_quantiles": {"p50": float(np.median(stability))},
                "halves": {
                    h: {"n_rows": 400, "factor_cos_top1": 0.9, "factor_cos_median": 0.6}
                    for h in ("half1", "half2")
                },
                "series_pt": "splithalf_stability_L19.pt",
            }
        )
    )
    (leg1 / "criterion_L19.json").write_text(
        json.dumps(
            {
                "regime": {"fixture": True},
                "status": "computed",
                "thresholds": {"rho_max": 1.0, "kappa_min": 10.0},
                "clauses": {
                    "rho_contraction": {
                        "metric": "rho(A) (spectral radius)",
                        "threshold": 1.0,
                        "op": "<",
                        "value": 1.2054,
                        "pass": False,
                    },
                    "kappa_nonnormal": {
                        "metric": "kappa(V) (eigenbasis condition number)",
                        "threshold": 10.0,
                        "op": ">=",
                        "value": 118.4,
                        "pass": True,
                    },
                    "stable_directions": {
                        "metric": "n_above_floor (split-half stability)",
                        "threshold": 300,
                        "op": ">=",
                        "value": n_above,
                        "pass": bool(n_above >= 300),
                    },
                    "copied_data_share": {
                        "metric": "copied_dw_share (copied-class data-variance share)",
                        "threshold": 0.2,
                        "op": "<",
                        "value": None,
                        "pass": None,
                        "deferral": "dw_mass_L19.json status=deferred (fixture)",
                    },
                },
                "overall": {"n_clauses": 4, "n_evaluated": 3, "n_failed": 2, "verdict": "FAIL"},
                "kill": {"kappa_lt_10": False, "copied_gt_50": None, "fired": None},
                "notes": ["fixture"],
            }
        )
    )
    (leg1 / "sae_dashboards_L19.json").write_text(
        json.dumps(
            {
                "sections": {
                    "singular_read": {
                        "dictionary": "fixture ctx",
                        "directions": _dash_direction_rows(8, rng, sigma, c_align, False, False),
                    },
                    "singular_write": {
                        "dictionary": "fixture ans",
                        "directions": _dash_direction_rows(8, rng, sigma, c_align, False, True),
                    },
                    "eigen_read": {
                        "dictionary": "fixture ctx",
                        "directions": _dash_direction_rows(8, rng, sigma, c_align, True, False),
                    },
                    "eigen_write": {
                        "dictionary": "fixture ans",
                        "directions": _dash_direction_rows(8, rng, sigma, c_align, True, True),
                    },
                },
                "null_floors": {"ctx": _null_floor(1000), "ans": _null_floor(600)},
                "whitened_cosine": "deferred-to-P-B (fixture)",
                "complex_note": "fixture",
                "label_sources": {"note": "fixture"},
            }
        )
    )
    # Wiring npz: monotone concentration curves ending at the top-32 share.
    n_feat, k_grid = 10, np.array([1, 2, 4, 8, 16, 32], np.int64)
    share32 = rng.uniform(0.15, 0.9, size=n_feat)
    ramp = (np.log2(k_grid) + 1.0) / (np.log2(k_grid[-1]) + 1.0)
    conc = share32[:, None] * ramp[None, :]
    is_near = np.zeros(n_feat, bool)
    is_near[[1, 4, 7]] = True
    feat_ids = np.sort(rng.choice(5000, size=n_feat, replace=False)).astype(np.int64)
    arrays = {
        "feat_ids": feat_ids,
        "is_rb_nearest": is_near,
        "top_edge_ids": rng.integers(0, 9000, size=(n_feat, 32)).astype(np.int64),
        "top_edge_vals": rng.normal(size=(n_feat, 32)).astype(np.float32),
        "edge_absmass_total": rng.uniform(1.0, 4.0, size=n_feat).astype(np.float64),
        "top32_absmass_share": share32.astype(np.float32),
        "conc_curve": conc.astype(np.float32),
        "conc_k_grid": k_grid,
    }
    share32_alive = np.minimum(1.0, share32 + 0.05)
    if alive:
        arrays.update(
            top_edge_ids_alive=arrays["top_edge_ids"],
            top_edge_vals_alive=arrays["top_edge_vals"],
            edge_absmass_total_alive=arrays["edge_absmass_total"],
            top32_absmass_share_alive=share32_alive.astype(np.float32),
            conc_curve_alive=(share32_alive[:, None] * ramp[None, :]).astype(np.float32),
        )
    np.savez(leg3 / "wiring_edges_L19.npz", **arrays)
    quant = {"median": 0.5, "mean": 0.5, "q25": 0.3, "q75": 0.7}
    traits = ("evil", "sycophancy", "hallucination")
    (leg3 / "wiring_L19.json").write_text(
        json.dumps(
            {
                "regime": {
                    "regime_version": 1,
                    "layer": 19,
                    "smoke": smoke,
                    "top_k": 8,
                    "n_draws": 100,
                },
                "h3": {
                    "statistic": "fixture",
                    "grain": "fixture",
                    "behavior_relevant": {
                        t: {
                            "feat_id": int(feat_ids[i]),
                            "cos": 0.5,
                            "top32_share_full": float(share32[i]),
                            "top32_share_alive": float(share32_alive[i]) if alive else None,
                        }
                        for t, i in zip(traits, (1, 4, 7), strict=True)
                    },
                    "union_top32_share_full": quant,
                    "union_top32_share_alive": quant if alive else "deferred — see ctx_alive",
                },
                "ctx_alive": {"n_alive": 5} if alive else "deferred (fixture)",
                "n_answer_features": n_feat,
                "rb_nearest": {},
                "out_edges": {},
                "out_edges_note": "fixture",
                "label_sources": {"note": "fixture"},
                "caveats": _WIRING_CAVEATS,
            }
        )
    )
    (leg3 / "attribution_L19.json").write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "row_id": 100 + i,
                        "position": i,
                        "n_ctx_active": 7,
                        "features": {
                            "11": _attr_feature("r_B-nearest (evil)", True),
                            "5": _attr_feature("top predicted activation", False),
                        },
                    }
                    for i in range(3)
                ],
                "holdout": {"source": "fixture", "n_holdout_present": 40, "seed": 1},
                "decomposition": "fixture",
                "label_sources": {"note": "fixture"},
                "caveats": _WIRING_CAVEATS,
            }
        )
    )


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
    _write_weights(root)
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


def _labels(ax) -> list[str]:
    return [ln.get_label() for ln in ax.get_lines()]


def _title(ax) -> str:
    """Title text regardless of location (paper style titles at loc='left')."""
    return ax.get_title() or ax.get_title("left") or ax.get_title("right")


def test_learning_curve_builds_and_renders(full_root, tmp_path):
    """Curve figure carries series artists and renders with a sidecar.

    Production regime (regime.smoke false): the registered H2b bands ARE drawn
    — the smoke-suppression gate must not disable the feature.
    """
    doc = json.loads((full_root / "leg2_curve" / "learning_curve.json").read_text())
    assert doc["regime"]["smoke"] is False
    fig = F.build_learning_curve(doc)
    assert _n_artists(fig) >= 6  # empirical + theory + companions + band lines
    band_labels = _labels(fig.axes[1])
    assert "H2b pass band" in band_labels and "H2b kill floor" in band_labels
    assert "SMOKE" not in _title(fig.axes[1])
    import matplotlib.pyplot as plt

    plt.close(fig)
    stem = F.fig_leg2_learning_curve(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)


def test_learning_curve_smoke_regime_suppresses_verdict_bands():
    """Regression: figures-render-verdict-bands-against-smoke-regime.

    A regime.smoke=true artifact renders WITHOUT the registered H2b bands and
    carries the regime in the panel titles (the permitted channel) — a smoke
    curve can never present as a pre-registered verdict.
    """
    import matplotlib.pyplot as plt

    doc = _learning_curve_doc(smoke=True)
    fig = F.build_learning_curve(doc)
    band_labels = _labels(fig.axes[1])
    assert "H2b pass band" not in band_labels and "H2b kill floor" not in band_labels
    assert "SMOKE regime" in _title(fig.axes[0])
    assert "SMOKE regime" in _title(fig.axes[1])
    # the data series still draw (the figure is suppressed-verdict, not blank)
    assert "Empirical minus theory" in band_labels
    plt.close(fig)


def test_learning_curve_undecidable_verdict_suppresses_bands():
    """A non-smoke artifact with NO computed statistic (undecidable) draws no bands."""
    import matplotlib.pyplot as plt

    doc = _learning_curve_doc(smoke=False)
    doc["h2b"]["mean_abs_dr2"] = None
    doc["h2b"]["verdict"] = "undecidable-underdetermined (n_train < d)"
    fig = F.build_learning_curve(doc)
    band_labels = _labels(fig.axes[1])
    assert "H2b pass band" not in band_labels and "H2b kill floor" not in band_labels
    assert "no computed H2b verdict" in _title(fig.axes[1])
    plt.close(fig)


def test_learning_curve_delta_panel_excludes_degenerate_points():
    """Regression: learning-curve-delta-panel-plots-degenerate-points (fix-round-3 NIT).

    On a MIXED n-grid (a degenerate n_train < d point beside well-posed ones)
    the verdict-grade delta panel draws ONLY the well-posed points against the
    registered bands — a never-scored degenerate point must not be readable
    against them. Panel A keeps every point (no bands there).
    """
    import matplotlib.pyplot as plt

    doc = _learning_curve_doc(smoke=False)
    doc["verdict_points"].append(_verdict_point(96, 0.20, 0.90))  # 96 < d=3,584
    fig = F.build_learning_curve(doc)
    (delta_line,) = [ln for ln in fig.axes[1].lines if ln.get_label() == "Empirical minus theory"]
    xs = list(delta_line.get_xdata())
    assert 96 not in xs and len(xs) == 3  # well-posed points only in the band panel
    (emp_line,) = [ln for ln in fig.axes[0].lines if "Empirical held-out" in ln.get_label()]
    assert len(emp_line.get_xdata()) == 4  # panel A keeps the degenerate point
    band_labels = _labels(fig.axes[1])
    assert "H2b pass band" in band_labels  # bands still drawn (verdict-grade doc)
    plt.close(fig)


def test_learning_curve_missing_regime_fails_loud():
    """An artifact with no regime.smoke flag raises — never a silent default."""
    doc = _learning_curve_doc()
    del doc["regime"]
    with pytest.raises(ValueError, match=r"regime\.smoke"):
        F.build_learning_curve(doc)
    doc2 = _learning_curve_doc()
    del doc2["regime"]["smoke"]
    with pytest.raises(ValueError, match=r"regime\.smoke"):
        F.build_learning_curve(doc2)


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
    assert _n_artists(fig) >= 4  # parity line + 3 labeled points (both arm names)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_dw_intruder_reads_arm_from_payload_and_guards_empty():
    """Regression: dwfleet-intruder-consumer-write-key-mismatch (fix-round-3).

    The five V-side modules carry arm ``read`` (never ``write``); the consumer
    reads the arm off each payload rather than hardcoding either literal. An
    arm-less payload raises; an observed arm with no matching null raises.
    """
    import matplotlib.pyplot as plt

    rec = {
        "arm_id": "arm1",
        "intruder": {
            "o_proj": _dv3_payload(0.05, 0.09),  # U-side: arm "write"
            "q_proj": _dv3_payload(0.11, 0.09, arm="read"),  # V-side: arm "read"
        },
    }
    fig = F.build_dw_intruder([rec])
    pts = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    assert len(pts) == 2  # both arm names consumed, one point per module
    plt.close(fig)
    empty = {"arm_id": "a", "intruder": {"o_proj": {"observed": {}, "null": {}}}}
    with pytest.raises(ValueError, match="carries no arms"):
        F.build_dw_intruder([empty])
    mismatched = {"arm_id": "a", "intruder": {"o_proj": {**_dv3_payload(0.1, 0.2), "null": {}}}}
    with pytest.raises(ValueError, match="no matching null"):
        F.build_dw_intruder([mismatched])


def test_dw_consumers_accept_real_producer_outputs(tmp_path, monkeypatch):
    """Cross-driver contract: REAL producer outputs feed both figure consumers.

    Runs the CURRENT producer (``analyze_lora_arm`` + ``cmd_align`` on the tiny
    committed fixtures from ``test_issue2569_dw_fleet``) and renders both
    consumers — the fix-round-3 reviewer reproduction
    (``dwfleet-alignment-consumer-flat-schema-stale`` +
    ``dwfleet-intruder-consumer-write-key-mismatch``), kept as a standing test
    so producer and consumers cannot silently drift apart again.
    """
    import issue650_analyze as I650
    import issue2569_dw_fleet as DW
    import matplotlib.pyplot as plt

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:  # importlib mode never adds tests/ itself
        sys.path.insert(0, tests_dir)
    import test_issue2569_dw_fleet as TDW

    adapter = TDW._write_adapter(tmp_path / "adapter", r=2)
    base_svd = I650.load_base_svd(TDW._build_tiny_base_svd(tmp_path), modules=DW.LORA_MODULES)
    rec = DW.analyze_lora_arm(TDW._lora_entry(), adapter, base_svd)
    arm_names = {a for p in rec["intruder"].values() for a in p["observed"]}
    assert arm_names == {"write", "read"}  # one record carries BOTH arms
    fig = F.build_dw_intruder([rec])
    pts = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    assert len(pts) == len(DW.LORA_MODULES)  # all 7 modules, no KeyError
    plt.close(fig)

    lora = TDW._lora_entry()
    banked = TDW._rb_banked()
    banked[f"{DW.DELTA_TF_PREFIX}/{lora.arm_id}/tbar.pt"] = TDW._tbar_payload(layer=0)
    banked[f"{DW.ANCHORS_PREFIX}/{lora.arm_id}.pt"] = TDW._anchor_payload(layer=0)
    out_root, dl_root, _ = TDW._setup_align(
        tmp_path / "align", monkeypatch, banked=banked, entries=[lora]
    )
    assert DW.cmd_align(TDW._args(out_root=str(out_root), dl_root=str(dl_root))) == 0
    align = json.loads((out_root / "dw_fleet" / "alignment.json").read_text())
    fig = F.build_dw_alignment(align)
    assert _n_artists(fig) >= 8  # per-direction panels: alignment points + null ticks
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


def test_display_map_covers_anatomy_classes_and_dashboard_sections():
    """Every anatomy class and dashboard section slug has a plain-English label."""
    for key in F.ANATOMY_CLASS_ORDER:
        assert F.display(key) != key
    assert "ignored" not in F.display("ignored")  # tau exceeds the median gain here
    for key in F.DASH_SECTION_SIDE:
        assert F.display(key) != key


def test_leg1_hero_and_eigen_scatter_render(full_root, tmp_path):
    """Spectra + class bars carry artists; the eigen-vs-singular scatter renders."""
    anatomy = F._load_anatomy(full_root)
    fac = F._load_factor_arrays(full_root)
    fig = F.build_leg1_anatomy_hero(anatomy, fac)
    # 2 spectra lines + 2 reference lines + 10 class bars (5 classes x 2 series)
    assert _n_artists(fig) >= 14
    import matplotlib.pyplot as plt

    plt.close(fig)
    fig = F.build_leg1_eigen_vs_singular(fac, anatomy)
    assert _n_artists(fig) >= 3  # scatter set(s) + parity line + tau line
    plt.close(fig)
    for fn, stem_want in (
        (F.fig_leg1_anatomy_hero, "leg1_anatomy_hero"),
        (F.fig_leg1_eigen_vs_singular, "leg1_eigen_vs_singular"),
    ):
        stem = fn(full_root, tmp_path / "figs")
        assert stem == stem_want
        _assert_rendered(tmp_path / "figs", stem)


def test_leg1_sae_dashboards_render(full_root, tmp_path):
    """All four sections render with both null-floor lines; eigen panels add im_frac."""
    doc = json.loads((full_root / "weights" / "leg1" / "sae_dashboards_L19.json").read_text())
    fig = F.build_sae_dashboards(doc)
    assert len(fig.axes) == 4
    for ax in fig.axes:
        assert len(ax.lines) >= 3  # max-|cos| series + analytic floor + empirical p95
    # eigen panels carry the extra im_frac series
    assert sum(1 for ax in fig.axes if len(ax.lines) >= 4) >= 2
    import matplotlib.pyplot as plt

    plt.close(fig)
    stem = F.fig_leg1_sae_dashboards(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)


def test_leg3_wiring_renders_alive_and_full_only(full_root, tmp_path):
    """Alive-attached npz renders 4 panels with decision lines; full-only renders 2.

    The sidecar caption carries the producer-shipped map-level caveat strings.
    """
    stem = F.fig_leg3_wiring_edge_mass(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    meta = json.loads((tmp_path / "figs" / f"{stem}.meta.json").read_text())
    assert "map-level" in meta["caption"]
    with np.load(full_root / "weights" / "leg3" / "wiring_edges_L19.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    wdoc = json.loads((full_root / "weights" / "leg3" / "wiring_L19.json").read_text())
    assert wdoc["regime"]["smoke"] is False
    fig = F.build_wiring_edge_mass(arrays, wdoc)
    assert len(fig.axes) == 4
    # production regime: the registered H3 verdict lines ARE drawn on the alive row
    alive_labels = _labels(fig.axes[2])
    assert "H3 PASS floor (0.50)" in alive_labels and "H3 kill line (0.10)" in alive_labels
    import matplotlib.pyplot as plt

    plt.close(fig)
    _write_weights(full_root, alive=False)  # rewrite without the *_alive npz keys
    with np.load(full_root / "weights" / "leg3" / "wiring_edges_L19.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    assert "top32_absmass_share_alive" not in arrays
    fig = F.build_wiring_edge_mass(arrays, wdoc)
    assert len(fig.axes) == 2  # informational panels only, no verdict-grade row
    plt.close(fig)


def test_leg3_wiring_smoke_regime_suppresses_h3_verdict_lines(tmp_path):
    """Class twin of the H2b regression: a smoke-regime rows-attached wiring run
    renders the alive row WITHOUT the registered H3 floor/kill lines, regime in
    the panel titles; a wdoc with no regime block fails loud."""
    import matplotlib.pyplot as plt

    root = tmp_path / "eval"
    root.mkdir()
    _write_weights(root, alive=True, smoke=True)
    with np.load(root / "weights" / "leg3" / "wiring_edges_L19.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    wdoc = json.loads((root / "weights" / "leg3" / "wiring_L19.json").read_text())
    assert wdoc["regime"]["smoke"] is True
    fig = F.build_wiring_edge_mass(arrays, wdoc)
    assert len(fig.axes) == 4  # the alive row still renders — only the verdict framing goes
    for ax in (fig.axes[2], fig.axes[3]):
        labels = _labels(ax)
        assert "H3 PASS floor (0.50)" not in labels and "H3 kill line (0.10)" not in labels
        assert "SMOKE regime" in _title(ax)
    plt.close(fig)
    with pytest.raises(ValueError, match=r"regime\.smoke"):
        F.build_wiring_edge_mass(arrays, {k: v for k, v in wdoc.items() if k != "regime"})


def test_leg3_attribution_renders_and_defers(full_root, tmp_path):
    """Per-example tables render (caveats in sidecar); a deferral string raises."""
    stem = F.fig_leg3_attribution_tables(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    meta = json.loads((tmp_path / "figs" / f"{stem}.meta.json").read_text())
    assert "map-level" in meta["caption"]
    doc = json.loads((full_root / "weights" / "leg3" / "attribution_L19.json").read_text())
    fig = F.build_attribution_tables(doc)
    assert sum(len(ax.tables) for ax in fig.axes) == 3  # one table per worked example
    import matplotlib.pyplot as plt

    plt.close(fig)
    # behavior-nearest feature is preferred over the id-lower predicted one
    fid, feat = F._attribution_table_feature(doc["examples"][0])
    assert fid == "11" and feat["why_in_table"].startswith("r_B-nearest")
    attr_path = full_root / "weights" / "leg3" / "attribution_L19.json"
    doc["examples"] = "deferred — the P-B P1 assemble dir was not attached (fixture)"
    attr_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="deferred by producer"):
        F.fig_leg3_attribution_tables(full_root, tmp_path / "figs")


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


# ---------------------------------------------------------------------------
# Leg-6 cross-arm shared-factor heatmap + leg-1 hero split-half bands
# ---------------------------------------------------------------------------


def test_leg6_shared_factor_heatmap_renders_with_caption(full_root, tmp_path):
    """Heatmap renders; refusals, skips, and the uncorrected label ride the sidecar."""
    stem = F.fig_leg6_shared_factor_heatmap(full_root, tmp_path / "figs")
    assert stem == "leg6_shared_factor_heatmap"
    _assert_rendered(tmp_path / "figs", stem)
    cap = json.loads((tmp_path / "figs" / f"{stem}.meta.json").read_text())["caption"]
    assert "factor_bases mismatch" in cap
    assert "UNCORRECTED" in cap
    assert "unit JSON missing" in cap
    assert "winner's-curse-inflated" in cap
    assert "criterion_met_any_cell=True" in cap


def test_leg6_pair_matrix_refused_and_skipped_are_nan_never_zero(full_root):
    """No-cosine cells stay NaN (rendered as not-tested), never zero-valued."""
    cell = json.loads((full_root / "leg6" / "cross_arm" / "L19_last_prompt.json").read_text())
    pm = F._pair_matrix(cell)
    order = pm["order"]
    i = order.index("cas-pers-con-lr1e5-s42")
    a = order.index("cas-bare-con-lr1e5-s42")
    j = order.index("syc-bare-con-lr1e5-s42")
    k = order.index("imp-bare-con-lr1e5-s42")
    assert np.isnan(pm["mat"][i, j]) and pm["refused"][i, j] and pm["refused"][j, i]
    assert np.isnan(pm["mat"][k, :]).all() and np.isnan(pm["mat"][:, k]).all()
    assert pm["mat"][i, a] == pytest.approx(0.82)
    assert pm["above"][i, a] and pm["above"][a, i]
    assert np.isnan(np.diag(pm["mat"])).all()


def test_leg1_hero_renders_stability_panel_and_criterion_caption(full_root, tmp_path):
    """Computed split-half record: three panels; criterion + counts in the sidecar."""
    series, note = F._load_splithalf(full_root)
    assert series is not None and "n_above_floor=" in note
    fig = F.build_leg1_anatomy_hero(
        F._load_anatomy(full_root), F._load_factor_arrays(full_root), series
    )
    assert len(fig.axes) == 3
    F.plt.close(fig)
    stem = F.fig_leg1_anatomy_hero(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    cap = json.loads((tmp_path / "figs" / f"{stem}.meta.json").read_text())["caption"]
    assert "n_above_floor=" in cap
    assert "leg-1 criterion verdict: FAIL" in cap
    assert "N/A, not tested" in cap  # the deferred copied_dw_share clause


def test_leg1_hero_splithalf_deferred_renders_without_bands(full_root, tmp_path):
    """A deferral record renders the two-panel hero; the caption names the gap."""
    leg1 = full_root / "weights" / "leg1"
    (leg1 / "splithalf_stability_L19.pt").unlink()
    (leg1 / "splithalf_stability_L19.json").write_text(
        json.dumps(
            {"regime": {}, "status": "deferred", "deferral_reason": "P-B moments absent (fixture)"}
        )
    )
    series, note = F._load_splithalf(full_root)
    assert series is None and "not yet computed" in note
    fig = F.build_leg1_anatomy_hero(
        F._load_anatomy(full_root), F._load_factor_arrays(full_root), None
    )
    assert len(fig.axes) == 2
    F.plt.close(fig)
    stem = F.fig_leg1_anatomy_hero(full_root, tmp_path / "figs")
    _assert_rendered(tmp_path / "figs", stem)
    cap = json.loads((tmp_path / "figs" / f"{stem}.meta.json").read_text())["caption"]
    assert "not yet computed (P-B moments absent (fixture))" in cap


def test_leg1_hero_stability_length_mismatch_fails_loud(full_root):
    """A stability series shorter than the sigma spectrum raises, never mis-pairs."""
    series, _ = F._load_splithalf(full_root)
    series["stability"] = series["stability"][:5]
    with pytest.raises(ValueError, match="does not match"):
        F.build_leg1_anatomy_hero(
            F._load_anatomy(full_root), F._load_factor_arrays(full_root), series
        )


# ---------------------------------------------------------------------------
# Open-marker ink under the production style (round-2 blocker regression)
# ---------------------------------------------------------------------------


def _iter_marker_artists(fig):
    """(where, Line2D) over visible axes lines and every legend-handle glyph."""
    from matplotlib.lines import Line2D

    legends = list(fig.legends)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        yield from (("axes line", ln) for ln in ax.lines if ln.get_visible())
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
    for leg in legends:
        handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
        yield from (("legend handle", h) for h in handles if isinstance(h, Line2D))


def _marker_ink_violations(fig) -> list[str]:
    """EFFECTIVE-property check: marker glyphs that would draw zero ink.

    Deliberately independent of the module's own audit (no hardcoded mew
    literals, no reliance on F._assert_marker_ink existing): ink = a face that
    is not "none", or a positive-width edge whose color is not "none".
    """
    from matplotlib.markers import MarkerStyle

    bad = []
    for where, ln in _iter_marker_artists(fig):
        marker = ln.get_marker()
        if marker in (None, "None", "", " "):
            continue
        edge = (
            float(ln.get_markeredgewidth()) > 0 and str(ln.get_markeredgecolor()).lower() != "none"
        )
        face = MarkerStyle(marker).is_filled() and str(ln.get_markerfacecolor()).lower() != "none"
        if not (face or edge):
            bad.append(f"{where}: marker={marker!r} label={ln.get_label()!r}")
    return bad


def test_open_markers_draw_ink_under_production_style(full_root):
    """Every open/edge-only marker series carries ink under set_paper_style("blog").

    Round-2 blocker regression: the blog style zeroes lines.markeredgewidth, so
    plot(..., mfc="none") with no explicit mew= rendered NOTHING -- leg7_three_tier
    lost BOTH Llama-to-Qwen series and leg2_learning_curve lost all committed
    companion points while the suite stayed green. Checks EFFECTIVE artist
    properties so any future open-marker site fails here too.
    """
    import matplotlib.pyplot as plt

    units = F._leg6_units(full_root / "leg6", F.LEG6_PRIMARY_CONVENTION, F.LEG6_PRIMARY_LAYER)
    cross_cell = json.loads((full_root / "leg6" / "cross_arm" / "L19_last_prompt.json").read_text())
    figs = {
        "leg2_learning_curve": F.build_learning_curve(
            json.loads((full_root / "leg2_curve" / "learning_curve.json").read_text())
        ),
        "leg2_gate_ladder_content": F.build_gate_ladder(
            json.loads((full_root / "leg2" / "gate_ladder.json").read_text()), "content"
        ),
        "leg5_dw_alignment": F.build_dw_alignment(
            json.loads((full_root / "dw_fleet" / "alignment.json").read_text())
        ),
        "leg6_denoised_rank": F.build_leg6_ranks(units, F.LEG6_PRIMARY_CONVENTION),
        "leg6_half_spectra": F.build_leg6_spectra(units, F.LEG6_PRIMARY_CONVENTION),
        "leg6_shared_factor_heatmap": F.build_leg6_shared_factor_heatmap(cross_cell),
        "leg7_three_tier": F.build_three_tier(
            json.loads((full_root / "leg7" / "three_tier.json").read_text())
        ),
        "leg7_atlas": F.build_atlas(
            json.loads((full_root / "leg7" / "atlas_distances.json").read_text())
        ),
    }
    try:
        bad = {name: v for name, fig in figs.items() if (v := _marker_ink_violations(fig))}
        assert not bad, f"zero-ink marker series under production style: {bad}"
        # The named blocker series exist AND are inked: both dashed Llama->Qwen
        # tier-1 series and both off-recipe companion series.
        tier_labels = [ln.get_label() for ln in figs["leg7_three_tier"].axes[0].lines]
        assert "Context state, Llama to Qwen" in tier_labels
        assert "Answer summary, Llama to Qwen" in tier_labels
        curve_labels = [ln.get_label() for ln in figs["leg2_learning_curve"].axes[0].lines]
        assert "Off-recipe committed point" in curve_labels
        assert "Off-recipe point, lambda at grid edge" in curve_labels
    finally:
        for fig in figs.values():
            plt.close(fig)


def test_render_rejects_inkless_open_marker(tmp_path):
    """_render fail-louds (RuntimeError, never a manifest skip) on a zero-ink marker.

    mew=0 is forced explicitly so the guard's behavior is pinned independent of
    whatever style the suite runs under.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1.0, 2.0], "o", mfc="none", mew=0, label="ghost series")
    with pytest.raises(RuntimeError, match="ZERO ink"):
        F._render(fig, "ghost", tmp_path / "figs")
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [1.0, 2.0], "o")  # filled marker: inked under any style
    assert F._render(fig2, "ok", tmp_path / "figs") == "ok"


# ---------------------------------------------------------------------------
# main() exit code: an all-skipped run is a failure, partial skips are not
# ---------------------------------------------------------------------------


def test_main_fails_loud_when_all_skipped(tmp_path):
    """A wrong --results-root (every figure skipped, none rendered) exits 1."""
    rc = F.main(["--results-root", str(tmp_path / "nothing"), "--fig-dir", str(tmp_path / "figs")])
    assert rc == 1
    manifest = json.loads((tmp_path / "figs" / "figures_manifest.json").read_text())
    assert manifest["rendered"] == {}
    assert len(manifest["skipped"]) == len(F.FIGURES)


def test_main_partial_skips_stay_rc_zero(full_root, tmp_path):
    """Legitimate individual skips (one producer missing) keep rc 0."""
    (full_root / "der" / "der_eval.json").unlink()
    rc = F.main(["--results-root", str(full_root), "--fig-dir", str(tmp_path / "figs")])
    assert rc == 0
    manifest = json.loads((tmp_path / "figs" / "figures_manifest.json").read_text())
    assert "leg4_der_matching" in manifest["skipped"]
    assert len(manifest["rendered"]) == len(F.FIGURES) - 1
