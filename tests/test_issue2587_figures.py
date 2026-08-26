"""CPU-only pins for scripts/issue2587_figures.py (issue #2587 unit 6).

No network, no HF fetch, no GPU, no torch. Fixture docs mirror the REALIZED
schemas of the producers:

* ``map_layer_sweep.json`` / ``matched7b_anchor.json`` — mirrored from the
  writer code in ``scripts/issue2587_fits.py`` (``run_finalize`` merged doc,
  ``run_matched7b`` record; same round, so the writer IS the schema source).
* ``crossmodel_contrasts.json`` / ``minpair_delta_2587.json`` /
  ``perpair_2587.jsonl`` — mirrored from ``scripts/issue2587_analysis.py``
  (``crossmodel_contrasts`` rows, ``main``'s merged doc with per-side
  ``meta``/``axes``/``retrieval`` blocks, ``_perpair_row``); the R2b schema
  additions (``primary_h2_7b_arm``, ``n_missing_fire_rows_9b/_7b``) are
  carried.
* ``manipulation_check_2587.json`` — mirrored from
  ``scripts/issue2587_judge.py`` (``_value_row`` + ``axis_summary`` writers,
  incl. the special ``not_in_slice`` axis-row shape).
* ``bank_manifest.json`` ``token_gates`` — mirrored from
  ``src/explore_persona_space/experiments/issue2587/bank2587.py``
  (``run_token_gates``).
* battery ``anchors_*.done.json`` / map-side ``cap_hit_*.json`` — mirrored
  from ``scripts/issue2587_battery_run.py`` (gen done-manifest) and
  ``scripts/issue2587_map_gen_capture.py`` (``run_aggregate_cap_hit``).
* the banked #2330 reference fits — schema probed on the COMMITTED artifact
  ``eval_results/issue_2330/matched_fits_q35_n10k.json`` (top-level ``layers``
  + ``per_layer[str].ridge.test_r2``).

Every test renders through the REAL figure functions to ``savefig`` (tmp_path
only — never canonical ``figures/`` paths), and one test drives the
production CLI entrypoint (``main(argv)``) end-to-end over the FULL registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue2587_figures as G  # noqa: E402

FLOOR_NAMES = (
    "identity_bias",
    "identity_copy",
    "scaled_identity",
    "shuffled_pairing",
    "train_mean",
)

ARMS9 = ("arm_fresh9b", "arm_iddelta9b")
ARMS7 = ("arm_7b_matched25k", "arm_iddelta7b")


# ---------------------------------------------------------------------------
# Fixture builders — fit-side (map_layer_sweep / matched7b / #2330 refs)
# ---------------------------------------------------------------------------


def _knn_block(base: float) -> dict:
    """One arm x metric kNN block in the JSON-round-tripped shape (string ks)."""
    ks = ("1", "5", "10", "50")

    def _one(off: float) -> dict:
        return {
            "acc_at_k": {k: min(1.0, base + off + 0.01 * int(k)) for k in ks},
            "chance_at_k": {k: int(k) / 1000.0 for k in ks},
            "median_rank": 3.0,
            "mrr": 0.4,
            "n_pool": 1000,
        }

    return {
        "ridge": {"euclidean": _one(0.0), "cosine": _one(0.02)},
        "identity_bias": {"euclidean": _one(-0.2), "cosine": _one(-0.18)},
        "train_mean": {"euclidean": _one(-0.5), "cosine": _one(-0.5)},
        "_meta": {"n_pool": 1000, "ks": [1, 5, 10, 50], "pool": "test targets"},
    }


def _floors(peak: float) -> dict:
    return {
        name: {"test_r2": peak - 0.25 - 0.02 * i, "meta": {}} for i, name in enumerate(FLOOR_NAMES)
    }


def _ceiling(val: float) -> dict:
    return {
        "available": True,
        "n_pairs": 1000,
        "banked_n_a": 1000,
        "banked_n_b": 1000,
        "ceiling_var_weighted_r": val,
        "mean_per_dim_r": val - 0.02,
    }


def make_sweep_doc(lstar: int = 22) -> dict:
    """32-layer fixture (full geometry so the #2329 dash-mark branch fires)."""
    per_layer = {}
    val_by_layer = {}
    for li in range(32):
        # unimodal curve peaking at layer 22 (realistic shape, distinct values)
        test_r2 = 0.45 + 0.25 * (1.0 - abs(li - lstar) / 31.0)
        val_r2 = test_r2 + 0.01
        val_by_layer[str(li)] = val_r2
        per_layer[str(li)] = {
            "issue": 2587,
            "layer": li,
            "regime_key": "rk-fixture",
            "n_train": 24950,
            "d": 4096,
            "ridge": {
                "meta": {"selected_lambda": 10.0 ** (2 + li % 3), "val_r2_at_selected": val_r2},
                "test_r2": test_r2,
                "wc_test_1k_r2": test_r2 - 0.08,
            },
            "floors": _floors(test_r2),
            "knn": _knn_block(0.3 + 0.2 * (1.0 - abs(li - lstar) / 31.0)),
            "timing_s": 1.0,
            "repro": {},
        }
    return {
        "issue": 2587,
        "regime_key": "rk-fixture",
        "store_prefix": "issue2587_q35_map/qwen35_9b",
        "h_dim": 4096,
        "n_layers": 32,
        "per_layer": per_layer,
        "split_counts": {"train_25k": 24950, "val_400": 400, "test_1000": 1000, "wc_test_1k": 998},
        "split_sha256": {},
        "lstar": {
            "lstar": lstar,
            "criterion": "argmax over layers of ridge val_r2_at_selected",
            "tie_break": "lowest layer index",
            "frozen": True,
            "val_r2_by_layer": val_by_layer,
        },
        "reliability_ceiling": {
            "layers": [16, 22, 30],
            "expected_banked_n": 1000,
            "seeds": [43, 44],
            "by_layer": {"16": _ceiling(0.9), "22": _ceiling(0.93), "30": _ceiling(0.88)},
        },
        "upload": {"mode": "none"},
        "repro": {},
    }


def make_matched7b_doc() -> dict:
    return {
        "issue": 2587,
        "regime_key": "rk-7b-fixture",
        "role": "fixture",
        "anchor": {
            "expected_r2": 0.7250873220237553,
            "realized_r2": 0.7251,
            "abs_deviation": 1.3e-05,
            "tol": 0.01,
            "selected_lambda": 3162.3,
            "val_r2_at_selected": 0.7308,
            "lambda_grid_edge": None,
            "n_train": 25000,
            "investigate_before_narrate": False,
        },
        "arm": {
            "name": "arm_7b_matched25k",
            "layer": 19,
            "n_train": 24950,
            "d": 3584,
            "test_r2": 0.71,
            "wc_test_1k_r2": 0.64,
            "ridge_meta": {"selected_lambda": 3162.3, "val_r2_at_selected": 0.7308},
            "floors": _floors(0.71),
            "knn": _knn_block(0.35),
            "split_manifests": {},
        },
        "ceiling_7b_matched_L19": _ceiling(0.95),
        "vc2564": {},
        "upload": {"mode": "none"},
        "complete": True,
        "repro": {},
    }


def make_ref2330_doc(layers: list[int], base: float) -> dict:
    """Banked #2330 matched-fits shape (probed on the committed artifact):
    top-level ``layers`` + ``per_layer[str(li)].ridge.test_r2``."""
    return {
        "model_key": "fixture",
        "layers": layers,
        "per_layer": {
            str(li): {"ridge": {"test_r2": base + 0.01 * i, "meta": {}}}
            for i, li in enumerate(layers)
        },
        "primary_layer": layers[len(layers) // 2],
    }


# ---------------------------------------------------------------------------
# Fixture builders — battery side (minpair delta / perpair / crossmodel)
# ---------------------------------------------------------------------------


def _null_block() -> dict:
    return {
        "scheme": "value-shuffle",
        "mean": 0.0,
        "q2_5": -0.08,
        "q97_5": 0.08,
        "b": 200,
        "seed": [42, 0],
        "over": "headline pairs",
    }


def _direction_arm(v: float) -> dict:
    return {
        "mean_cos_headline": v,
        "ci95": [v - 0.05, v + 0.05],
        "mean_cos_all_values": v - 0.01,
        "ci95_all_values": [v - 0.06, v + 0.04],
        "sensitivity_mean_cos": {"50": v + 0.01, "70": v, "90": v - 0.01},
        "null": _null_block(),
        "ceiling_normalized_cos": v / 0.9,
        "ceiling_suppressed": False,
        "controls": {"install": {"mean_cos": v - 0.1, "ci95": [v - 0.2, v], "n_pairs": 12}},
    }


def _calib_arm(r: float) -> dict:
    return {
        "axis_slope": 0.9,
        "axis_slope_ci95": [0.8, 1.0],
        "axis_slope_all_values": 0.88,
        "axis_slope_ci95_all_values": [0.78, 0.98],
        "global_slope_all_pairs": 0.85,
        "ratio_to_global": r,
        "ratio_to_global_ci95": [r - 0.1, r + 0.1],
        "ratio_to_global_all_values": r - 0.02,
        "ratio_to_global_ci95_all_values": [r - 0.12, r + 0.08],
    }


def _identity_arm(m: float) -> dict:
    return {
        "per_vp_cos": {"v1|v2": m, "v1|v3": m - 0.1, "v2|v3": m + 0.1},
        "per_vp_fired70": {"v1|v2": True, "v1|v3": True, "v2|v3": False},
        "median": m,
        "median_ci95": [m - 0.1, m + 0.1],
        "median_all_values": m - 0.01,
        "median_ci95_all_values": [m - 0.11, m + 0.09],
    }


def _crossfam_blk(m: float) -> dict:
    return {
        "per_vp_cos": {"v1|v2": m},
        "per_vp_fired70_both_families": {"v1|v2": True},
        "median": m,
        "median_ci95": [m - 0.1, m + 0.1],
        "median_all_values": m - 0.01,
        "median_ci95_all_values": [m - 0.11, m + 0.09],
        "null": _null_block(),
    }


def _surface_blk() -> dict:
    return {
        "flip_norm_mean": 6.0,
        "flip_norm_ci95": [5.5, 6.5],
        "para_norm_mean": 3.0,
        "para_norm_ci95": [2.5, 3.5],
        "gap": 3.0,
        "gap_ci95": [2.0, 4.0],
        "gap_all_values": 2.9,
        "gap_ci95_all_values": [1.9, 3.9],
        "edit_dose_ols": {"intercept": 4.0, "slope": 0.8, "n": 48},
        "edit_dose_ties": 0.4,
        "residualized_gap": 2.5,
        "residualized_gap_all_values": 2.4,
        "labeling": "fixture",
    }


def _twin_block() -> dict:
    return {
        str(li): {
            "iddelta_mean_cos_headline": 0.35 + 0.02 * i,
            "iddelta_mean_cos_all_values": 0.34 + 0.02 * i,
            "iddelta_ratio_to_global": 0.65 + 0.02 * i,
            "iddelta_ratio_to_global_all_values": 0.64 + 0.02 * i,
            "note": "iddelta-only twin (frozen map is L*-fit)",
        }
        for i, li in enumerate((16, 22, 30))
    }


def _axis_row(
    axis: str,
    map_arm: str,
    id_arm: str,
    base: float,
    layer_twins: dict,
    pilot: bool = False,
    na_identity: bool = False,
) -> dict:
    row = {
        "axis": axis,
        "model_tag": "fixture",
        "pilot_axis": pilot,
        "primary_class": "swap",
        "para_class": "instruction_paraphrase",
        "n_primary_pairs": 24,
        "fire": {
            "axis_row": None,
            "axis_row_missing": False,
            "n_primary_pairs": 24,
            "n_headline_pairs_fired70": 20,
            "n_missing_fire_rows": 0,
            "floor_met": True,
            "compliance_limited": False,
            "no_fired_pairs": False,
            "headline_ok": True,
            "fired_pair_counts": {"swap": 20},
        },
        "direction": {
            map_arm: {
                **_direction_arm(base),
                "gap_vs_iddelta": {"mean": 0.2, "ci95": [0.1, 0.3]},
            },
            id_arm: _direction_arm(base - 0.2),
        },
        "calibration": {map_arm: _calib_arm(1.1), id_arm: _calib_arm(0.7)},
        "identity": (
            {"n/a": "no carrier-replicated multi-value grid"}
            if na_identity
            else {map_arm: _identity_arm(0.5), id_arm: _identity_arm(0.3)}
        ),
        "cross_family": (
            {"n/a": "no paraphrase-family swap class"}
            if na_identity
            else {
                # base-derived so per-axis / per-model points separate on the
                # consistency scatter (a constant fixture stacks every point)
                "observed": _crossfam_blk(base - 0.05),
                map_arm: _crossfam_blk(base - 0.15),
                id_arm: _crossfam_blk(base - 0.35),
            }
        ),
        "reliability": {
            "r_half_mean": 0.62 + base / 4,
            "r10_mean": 0.78 + base / 4,
            "r10_ci95": [0.73 + base / 4, 0.82 + base / 4],
            "r10_mean_all_values": 0.77 + base / 4,
            "r10_ci95_all_values": [0.72 + base / 4, 0.81 + base / 4],
            "noise_norm_mean": 1.2,
            "noise_norm_mean_all_values": 1.25,
            "spearman_brown": 0.95,
        },
        "text_space": None,
        "surface": {"observed": _surface_blk(), map_arm: _surface_blk(), id_arm: _surface_blk()},
        "answer_length": {"swap": {"mean_delta_tokens": 1.5, "mean_abs_delta_tokens": 4.0}},
        "layer_twins": layer_twins,
        "pooling_twin_span": {
            map_arm: {
                "mean_cos_headline": base - 0.03,
                "mean_cos_all_values": base - 0.04,
                "axis_slope": 1.0,
                "axis_slope_all_values": 0.98,
            },
            id_arm: {
                "mean_cos_headline": base - 0.23,
                "mean_cos_all_values": base - 0.24,
                "axis_slope": 0.8,
                "axis_slope_all_values": 0.78,
            },
        },
    }
    return row


def _battery_knn(arms: tuple[str, str]) -> dict:
    ks = ("1", "5", "10", "50")

    def _one(base: float) -> dict:
        return {
            "acc_at_k": {k: min(1.0, base + 0.01 * int(k)) for k in ks},
            "chance_at_k": {k: int(k) / 500.0 for k in ks},
            "median_rank": 4.0,
            "mrr": 0.3,
            "n_pool": 500,
        }

    return {
        "global": {
            arms[0]: {"euclidean": _one(0.4), "cosine": _one(0.42)},
            arms[1]: {"euclidean": _one(0.2), "cosine": _one(0.22)},
        },
        "per_axis": {},
        "chance": {"rule": "k/n_pool", "n_pool_global": 500},
    }


def _delta_side(tag: str, arms: tuple[str, str], layer: int, twin_layers: list[int]) -> dict:
    marm, iarm = arms
    if tag == "qwen35_9b":
        twins = _twin_block()
        axes = {
            "register": _axis_row("register", marm, iarm, 0.62, twins),
            "answer_language": _axis_row("answer_language", marm, iarm, 0.45, twins, pilot=True),
            "politeness": _axis_row("politeness", marm, iarm, 0.30, twins, na_identity=True),
        }
    else:
        na = {"n/a": "no twin layers on the 7B side"}
        axes = {
            "register": _axis_row("register", marm, iarm, 0.55, na),
            "politeness": _axis_row("politeness", marm, iarm, 0.28, na, na_identity=True),
        }
    return {
        "meta": {
            "primary_h2_7b_arm": "arm_7b_matched25k",
            "model_tag": tag,
            "d": 4096 if tag == "qwen35_9b" else 3584,
            "primary_layer": layer,
            "twin_layers": twin_layers,
            "arms": list(arms),
            "map_arm": marm,
            "id_arm": iarm,
            "n_contexts": 40,
            "n_pairs": 72,
        },
        "axes": axes,
        "retrieval": _battery_knn(arms),
    }


def make_delta_doc() -> dict:
    """Full merged minpair_delta_2587.json fixture (R2b realized schema):
    per-side meta/axes/retrieval blocks + the h1 paired-comparison block the
    matched-n table consumes. The 7B side deliberately LACKS the pilot axis
    (parent pilot reads pending — plan convention 12)."""
    return {
        "meta": {"issue": 2587, "schema": "minpair_delta_2587_v1"},
        "contract": {"identity_cancellation": "learned bias cancels in the delta framing"},
        "sides": {
            "qwen35_9b": _delta_side("qwen35_9b", ARMS9, 22, [16, 22, 30]),
            "qwen25_7b": _delta_side("qwen25_7b", ARMS7, 19, []),
        },
        "h1": {
            "r2_9b_lstar": 0.70,
            "r2_7b_l19": 0.71,
            "delta_map": -0.01,
            "delta_ci95": [-0.03, 0.01],
            "verdict": "h1_inconclusive",
        },
        "h2": {"combined_verdict": "h2_inconclusive"},
    }


def make_perpair_rows() -> list[dict]:
    """perpair_2587.jsonl fixture rows (analysis.py `_perpair_row` shape)."""
    rows: list[dict] = []
    specs = (
        ("qwen35_9b", ARMS9, ("register", "answer_language", "politeness")),
        ("qwen25_7b", ARMS7, ("register", "politeness")),
    )
    i = 0
    for tag, arms, axes in specs:
        marm, iarm = arms
        for axis in axes:
            for cls in ("swap", "install"):
                for carrier in ("astronomy", "baking"):
                    i += 1
                    cos_m = 0.7 - 0.02 * (i % 7)
                    rows.append(
                        {
                            "primary_h2_7b_arm": "arm_7b_matched25k",
                            "model_tag": tag,
                            "pair_id": f"{tag}-{axis}-{cls}-{carrier}-{i}",
                            "pair_class": cls,
                            "axis": axis,
                            "carrier": carrier,
                            "value_a": "v1",
                            "value_b": "v2",
                            "orientation": "a_to_b",
                            "changed_tokens": 1 + (i % 4),
                            "n_draws_a": 10,
                            "n_draws_b": 10,
                            "ans_len_delta": (i % 5) - 2,
                            "norm_obs_tail_primary": 3.0 + 0.5 * (i % 6),
                            "norm_obs_span_primary": 2.8 + 0.5 * (i % 6),
                            "norm_text": 1.0,
                            "cos": {marm: cos_m, iarm: cos_m - 0.25},
                            "cos_span": {marm: cos_m - 0.02, iarm: cos_m - 0.27},
                            "norm_pred": {marm: 2.5 + 0.45 * (i % 6), iarm: 2.0 + 0.4 * (i % 6)},
                            "r_half": 0.8,
                            "r10": 0.9,
                            "noise_norm": 1.1,
                            "fired_a_70": True,
                            "fired_b_70": True,
                            "pair_fired_70": True,
                            "in_headline_70": True,
                            "pilot_axis": axis == "answer_language",
                        }
                    )
    return rows


def make_crossmodel_doc() -> dict:
    """3-axis fixture; includes a null-valued row (JSON null after the
    producer's NaN sanitize) AND an INVERTED delta CI (lo > point > hi can
    genuinely occur at tiny n — the xerr non-negative-offsets gotcha pin)."""

    def _row(axis, s9, s7, ref, lo, hi, cleared=True):
        d = None if (s9 is None or s7 is None) else s9 - s7
        return {
            "axis": axis,
            "s_9b": s9,
            "s_7b": s7,
            "s_7b_ref_parent": ref,
            "delta_9b_minus_7b": d,
            "delta_ci95": [lo, hi],
            "delta_t11_ci95": [lo, hi],
            "delta_loco_jackknife_range": [lo, hi],
            "fire": {
                "symmetric_headline": True,
                "n_shared_primary": 24,
                "n_symmetric_fired": 20,
                "n_dropped_9b_only": 2,
                "n_dropped_7b_only": 2,
                "n_missing_fire_rows_9b": 0,
                "n_missing_fire_rows_7b": 1,
            },
            "ceiling_cleared": cleared,
        }

    stats = {}
    for stat in (
        "direction_cos",
        "calibration_ratio_to_global",
        "obs_separation_snr",
        "crossfam_cos_observed",
        "crossfam_cos_maparm",
        "axis_identity_cos",
    ):
        stats[stat] = {
            "definition": f"fixture {stat}",
            "axes": [
                _row("register", 0.62, 0.55, 0.58, 0.02, 0.12),
                # INVERTED CI around the point (delta = 0.30 - 0.70 = -0.40;
                # ci [-0.35, -0.45] inverts): must render, never ValueError.
                _row("answer_language", 0.30, 0.70, 0.65, -0.35, -0.45),
                _row("politeness", None, None, None, None, None, cleared=False),
            ],
            "spearman": {"rho": 0.7, "n": 3, "p": 0.1, "method": "exact"},
            "spearman_partial_changed_tokens": {"rho": 0.6, "note": "fixture"},
            "spearman_ceiling_cleared": {"rho": 0.7, "n": 2, "p": 0.2, "method": "exact"},
        }
    return {
        "layer_pair": {"qwen35_9b": 22, "qwen25_7b": 19},
        "stats": stats,
        "h2": {"combined_verdict": "h2_inconclusive"},
        "meta": {},
    }


# ---------------------------------------------------------------------------
# Fixture builders — judge / bank / leak-caphit inputs
# ---------------------------------------------------------------------------


def make_manip_doc(
    special_axis: str | None = "user_fact", special_verdict: str = "not_in_slice"
) -> dict:
    """manipulation_check JSON fixture (judge.py value_rows + axis_rows)."""
    value_rows = []
    for axis, vids in (("register", ["formal", "casual", "archaic"]), ("politeness", ["v1", "v2"])):
        for vid in vids:
            value_rows.append(
                {
                    "axis": axis,
                    "value_id": vid,
                    "kind": "orig",
                    "instrument": "judge",
                    "n_comply": 9,
                    "n_noncomply": 1,
                    "n_incomplete": 0,
                    "denom": 10,
                    "comply_frac": 0.9,
                    "verdict": "fired",
                    "sensitivity": {"60": "fired", "70": "fired", "80": "fired"},
                }
            )
    axis_rows = [
        {
            "axis": "register",
            "width": 3,
            "floor": 2,
            "n_fired_base": 3,
            "n_undetermined_base": 0,
            "n_not_fired_base": 0,
            "floor_met": True,
            "n_fired_para": 3,
            "sensitivity": {"60": {"n_fired_base": 3, "floor_met": True}},
        },
        {
            "axis": "politeness",
            "width": 2,
            "floor": 2,
            "n_fired_base": 1,
            "n_undetermined_base": 1,
            "n_not_fired_base": 0,
            "floor_met": False,
            "n_fired_para": None,
            "sensitivity": {"60": {"n_fired_base": 2, "floor_met": True}},
        },
    ]
    if special_axis:
        axis_rows.append({"axis": special_axis, "verdict": special_verdict})
    return {"issue": 2587, "value_rows": value_rows, "axis_rows": axis_rows, "meta": {}}


def make_bank_doc(with_gates: bool = True) -> dict:
    """bank_manifest.json fixture (bank2587 token_gates block)."""
    token_gates = None
    if with_gates:
        token_gates = {
            "verdict": "PASS",
            "gates_run": ["value_token_counts", "within_axis_equal", "name_single_token"],
            "tokenizer_id": "fixture-q35",
            "value_token_counts": {
                "register": {"formal": 12, "casual": 12, "archaic": 12},
                "politeness": {"blunt": 9, "polite": 11},
                "answer_language": {"french": 8, "german": 8},
            },
            "paraphrase_token_counts": {"register": {"formal": 13, "casual": 13, "archaic": 14}},
            "within_axis_equal": {"register": True, "politeness": False, "answer_language": True},
            "q25_expected_value_tokens": {"register": 12, "politeness": 10},
            "name_token_counts": {
                "Alice": {"n_tokens": 1, "ids": [111], "single_token": True, "q25_pinned_id": 222},
                "Bob": {"n_tokens": 2, "ids": [1, 2], "single_token": False, "q25_pinned_id": 333},
            },
            "changed_tokens_min": 1,
            "changed_tokens_max": 4,
        }
    return {"issue": 2587, "n_contexts": 40, "token_gates": token_gates}


def write_leak_fixtures(dest: str | Path) -> Path:
    """Battery gen done-manifests + a map-side cap-hit aggregate under one
    dir (the ``--leak-caphit-dir`` input); politeness trips BOTH flags."""
    d = Path(dest)
    (d / "manifests").mkdir(parents=True, exist_ok=True)
    (d / "manifests" / "anchors_register.done.json").write_text(
        json.dumps(
            {
                "cell": "register",
                "n_rows": 96,
                "cap_hit_frac": 0.01,
                "cap_hit_frac_regen": 0.0,
                "think_leak": {"n": 96, "n_leaked": 0, "frac": 0.0},
                "capture_max_model_len_floor": 4096,
            }
        )
    )
    (d / "manifests" / "anchors_politeness.done.json").write_text(
        json.dumps(
            {
                "cell": "politeness",
                "n_rows": 96,
                "cap_hit_frac": 0.05,
                "cap_hit_frac_regen": 0.03,
                "think_leak": {"n": 96, "n_leaked": 2, "frac": 2 / 96},
            }
        )
    )
    (d / "cap_hit_train_25k.json").write_text(
        json.dumps(
            {
                "schema": "issue2330_cap_hit_v2",
                "split": "train_25k",
                "total": 25000,
                "cap_hit": 300,
                "cap_hit_frac": 0.012,
                "cap_hit_cis": [],
            }
        )
    )
    return d


# ---------------------------------------------------------------------------
# Fixture writing + CLI argv
# ---------------------------------------------------------------------------


def write_all_fixtures(dest: str | Path) -> dict[str, Path]:
    """Write every fixture input to ``dest`` (also used by the CLI smoke)."""
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep": d / "map_layer_sweep.json",
        "matched7b": d / "matched7b_anchor.json",
        "delta": d / "minpair_delta_2587.json",
        "crossmodel": d / "crossmodel_contrasts.json",
        "ref9b": d / "matched_fits_q35_n10k.json",
        "ref7b": d / "matched_fits_q25_n10k.json",
        "perpair": d / "perpair_2587.jsonl",
        "manip9b": d / "manipulation_check_2587.json",
        "manip7b": d / "manipulation_check.json",
        "bank": d / "bank_manifest.json",
        "leakdir": d / "leak",
    }
    paths["sweep"].write_text(json.dumps(make_sweep_doc()))
    paths["matched7b"].write_text(json.dumps(make_matched7b_doc()))
    paths["delta"].write_text(json.dumps(make_delta_doc()))
    paths["crossmodel"].write_text(json.dumps(make_crossmodel_doc()))
    paths["ref9b"].write_text(json.dumps(make_ref2330_doc([16, 22, 30], 0.62)))
    paths["ref7b"].write_text(json.dumps(make_ref2330_doc([14, 19, 26], 0.66)))
    paths["perpair"].write_text("\n".join(json.dumps(r) for r in make_perpair_rows()) + "\n")
    paths["manip9b"].write_text(json.dumps(make_manip_doc()))
    paths["manip7b"].write_text(
        json.dumps(
            make_manip_doc(
                special_axis="query_content", special_verdict="no_manipulation_check_query_class"
            )
        )
    )
    paths["bank"].write_text(json.dumps(make_bank_doc()))
    write_leak_fixtures(paths["leakdir"])
    return paths


def _argv(paths: dict[str, Path], out_dir: Path, figs: str) -> list[str]:
    return [
        "--figs",
        figs,
        "--out-dir",
        str(out_dir),
        "--sweep-json",
        str(paths["sweep"]),
        "--matched7b-json",
        str(paths["matched7b"]),
        "--delta-json",
        str(paths["delta"]),
        "--crossmodel-json",
        str(paths["crossmodel"]),
        "--ref2330-9b",
        str(paths["ref9b"]),
        "--ref2330-7b",
        str(paths["ref7b"]),
        "--perpair-jsonl",
        str(paths["perpair"]),
        "--manip9b-json",
        str(paths["manip9b"]),
        "--manip7b-json",
        str(paths["manip7b"]),
        "--bank-json",
        str(paths["bank"]),
        "--leak-caphit-dir",
        str(paths["leakdir"]),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_FIG_STEMS = (
    "fig_hero_layer_sweep",
    "fig_hero_crossmodel_axis_profile",
    "fig_crossmodel_delta_forest",
    "fig_matched_vs_parent_scatter",
    "fig_selected_lambda_per_layer",
    "fig_floors_per_layer",
    "fig_wc_transfer_per_layer",
    "fig_knn_per_layer",
    "fig_reliability_ceiling",
    "fig_delta_norm_scatter_qwen35_9b",
    "fig_delta_norm_scatter_qwen25_7b",
    "fig_install_swap_violins",
    "fig_edit_dose_scatter",
    "fig_carrier_direction_heatmap",
    "fig_axis_identity_heatmap",
    "fig_crossfam_consistency_scatter",
    "fig_delta_retrieval_acc",
    "fig_splithalf_vs_direction",
    "fig_pilot_axis_panels",
    "fig_lstar_sensitivity_twins",
    "fig_pooling_twin_scatter",
)

EXPECTED_TABLE_STEMS = (
    "table_matched_n",
    "table_think_leak_cap_hit",
    "table_manipulation_check",
    "table_token_count_equality",
)


def _assert_png(out_dir: Path, stem: str, min_bytes: int = 5000) -> None:
    p = out_dir / f"{stem}.png"
    assert p.is_file(), f"missing {p}"
    assert p.stat().st_size > min_bytes, f"{p} suspiciously small ({p.stat().st_size} B)"
    assert (out_dir / f"{stem}.meta.json").is_file()


def test_hero_layer_sweep_renders(tmp_path):
    inputs = {
        "sweep": make_sweep_doc(),
        "ref9b_n10k": make_ref2330_doc([16, 22, 30], 0.62),
        "ref7b_n10k": make_ref2330_doc([14, 19, 26], 0.66),
    }
    written = G.fig_hero_layer_sweep(inputs, tmp_path)
    assert written
    _assert_png(tmp_path, "fig_hero_layer_sweep")


def test_matched_n_table_md_and_json(tmp_path):
    inputs = {
        "sweep": make_sweep_doc(),
        "matched7b": make_matched7b_doc(),
        "delta": make_delta_doc(),
    }
    written = G.matched_n_table(inputs, tmp_path)
    assert {p.name for p in written} == {"table_matched_n.md", "table_matched_n.json"}
    md = (tmp_path / "table_matched_n.md").read_text()
    # display names, never internal slugs, in the reader-facing table
    assert G.DISPLAY["qwen35_9b"] in md and G.DISPLAY["qwen25_7b"] in md
    assert "arm_7b_matched25k" not in md and "qwen35_9b" not in md
    assert "held-out test R²" in md
    assert "Anchor gate" in md
    assert "Paired shared-test-row comparison" in md
    doc = json.loads((tmp_path / "table_matched_n.json").read_text())
    assert doc["layer_pair"] == {"qwen35_9b": 22, "qwen25_7b": 19}
    assert doc["sides"]["qwen35_9b"]["n_train"] == 24950
    assert doc["sides"]["qwen25_7b"]["two_draw_ceiling_r"] == pytest.approx(0.95)
    assert doc["anchor_gate"]["tol"] == pytest.approx(0.01)
    assert doc["h1_paired_shared_rows"]["verdict"] == "h1_inconclusive"


def test_matched_n_table_without_delta(tmp_path):
    inputs = {"sweep": make_sweep_doc(), "matched7b": make_matched7b_doc()}
    G.matched_n_table(inputs, tmp_path)
    doc = json.loads((tmp_path / "table_matched_n.json").read_text())
    assert "h1_paired_shared_rows" not in doc


def test_delta_forest_inverted_ci_clamps(tmp_path):
    """The fixture's answer_language row carries an INVERTED delta CI: the
    real errorbar call must clamp to non-negative offsets, never ValueError
    (gotchas.md xerr/yerr rule)."""
    inputs = {"crossmodel": make_crossmodel_doc()}
    G.fig_crossmodel_delta_forest(inputs, tmp_path)
    _assert_png(tmp_path, "fig_crossmodel_delta_forest")


def test_axis_profile_handles_null_rows_and_missing_7b_axis(tmp_path):
    """Hero 1 must render through (a) crossmodel rows sanitized to JSON null
    (politeness), (b) a delta-side axis with NO 7B counterpart
    (answer_language — parent pilot reads pending), and (c) an inverted
    crossmodel CI. All layers (null band / ceiling / iddelta / CI) present."""
    inputs = {"crossmodel": make_crossmodel_doc(), "delta": make_delta_doc()}
    G.fig_crossmodel_axis_profile(inputs, tmp_path)
    _assert_png(tmp_path, "fig_hero_crossmodel_axis_profile")


def test_axis_profile_requires_delta_sides(tmp_path):
    """An h1-only delta stub (no populated sides) must fail loud, never render
    a hero without its CI/null/ceiling/iddelta layers."""
    inputs = {
        "crossmodel": make_crossmodel_doc(),
        "delta": {"h1": {"r2_9b_lstar": 0.7}},
    }
    with pytest.raises(RuntimeError, match="side block"):
        G.fig_crossmodel_axis_profile(inputs, tmp_path)


def test_crossmodel_empty_axes_fail_loud(tmp_path):
    """r1 Minor 1: empty stats[...]['axes'] must RAISE, never render blank."""
    cm = make_crossmodel_doc()
    for stat in cm["stats"]:
        cm["stats"][stat]["axes"] = []
    with pytest.raises(RuntimeError, match="EMPTY"):
        G.fig_crossmodel_delta_forest({"crossmodel": cm}, tmp_path)
    with pytest.raises(RuntimeError, match="EMPTY"):
        G.fig_crossmodel_axis_profile({"crossmodel": cm, "delta": make_delta_doc()}, tmp_path)
    with pytest.raises(RuntimeError, match="EMPTY"):
        G.fig_matched_vs_parent_scatter({"crossmodel": cm}, tmp_path)


def test_err_offsets_never_negative():
    import numpy as np

    vals = np.array([0.5, -0.4])
    lo = np.array([0.6, -0.35])  # inverted below
    hi = np.array([0.4, -0.45])  # inverted above
    off = G._err_offsets(vals, lo, hi)
    assert (off >= 0).all()


def test_perpair_figures_render(tmp_path):
    inputs = {"perpair": make_perpair_rows(), "delta": make_delta_doc()}
    G.fig_delta_norm_scatter(inputs, tmp_path)
    _assert_png(tmp_path, "fig_delta_norm_scatter_qwen35_9b")
    _assert_png(tmp_path, "fig_delta_norm_scatter_qwen25_7b")
    G.fig_install_swap_violins(inputs, tmp_path)
    _assert_png(tmp_path, "fig_install_swap_violins")
    G.fig_edit_dose_scatter(inputs, tmp_path)
    _assert_png(tmp_path, "fig_edit_dose_scatter")
    G.fig_carrier_direction_heatmap(inputs, tmp_path)
    _assert_png(tmp_path, "fig_carrier_direction_heatmap")


def test_perpair_missing_model_fails_loud(tmp_path):
    rows = [r for r in make_perpair_rows() if r["model_tag"] == "qwen35_9b"]
    with pytest.raises(RuntimeError, match="qwen25_7b"):
        G.fig_delta_norm_scatter({"perpair": rows}, tmp_path)


def test_install_swap_violins_empty_class_fails_loud(tmp_path):
    rows = [
        r
        for r in make_perpair_rows()
        if not (r["model_tag"] == "qwen35_9b" and r["pair_class"] == "install")
    ]
    with pytest.raises(RuntimeError, match="install"):
        G.fig_install_swap_violins({"perpair": rows}, tmp_path)


def test_delta_figures_render(tmp_path):
    inputs = {"delta": make_delta_doc()}
    G.fig_axis_identity_heatmap(inputs, tmp_path)
    _assert_png(tmp_path, "fig_axis_identity_heatmap")
    G.fig_crossfam_consistency_scatter(inputs, tmp_path)
    _assert_png(tmp_path, "fig_crossfam_consistency_scatter")
    G.fig_delta_retrieval_acc(inputs, tmp_path)
    _assert_png(tmp_path, "fig_delta_retrieval_acc")
    G.fig_splithalf_vs_direction(inputs, tmp_path)
    _assert_png(tmp_path, "fig_splithalf_vs_direction")
    G.fig_pilot_axis_panels(inputs, tmp_path)
    _assert_png(tmp_path, "fig_pilot_axis_panels")
    G.fig_lstar_sensitivity_twins(inputs, tmp_path)
    _assert_png(tmp_path, "fig_lstar_sensitivity_twins")
    G.fig_pooling_twin_scatter(inputs, tmp_path)
    _assert_png(tmp_path, "fig_pooling_twin_scatter")


def test_pilot_panels_require_pilot_axes(tmp_path):
    delta = make_delta_doc()
    for row in delta["sides"]["qwen35_9b"]["axes"].values():
        row["pilot_axis"] = False
    with pytest.raises(RuntimeError, match="pilot"):
        G.fig_pilot_axis_panels({"delta": delta}, tmp_path)


def test_think_leak_cap_hit_table_flags(tmp_path):
    leak = write_leak_fixtures(tmp_path / "leak")
    inputs = {"leakdir": G._load_leak_dir(leak, "leak fixtures")}
    written = G.think_leak_cap_hit_table(inputs, tmp_path)
    assert {p.name for p in written} == {
        "table_think_leak_cap_hit.md",
        "table_think_leak_cap_hit.json",
    }
    md = (tmp_path / "table_think_leak_cap_hit.md").read_text()
    assert "cap-hit over re-gen trigger" in md  # politeness post-regen 0.03 > 0.02
    assert "think-leak over assert" in md  # politeness 2/96 >= 1%
    assert "| ok |" in md  # register + train_25k rows clean
    doc = json.loads((tmp_path / "table_think_leak_cap_hit.json").read_text())
    assert len(doc["rows"]) == 3
    by_unit = {r["unit"]: r for r in doc["rows"]}
    assert by_unit["politeness"]["cap_hit_over_regen_trigger"] is True
    assert by_unit["politeness"]["think_leak_over_assert"] is True
    assert by_unit["register"]["cap_hit_over_regen_trigger"] is False
    assert by_unit["train_25k"]["kind"] == "map-fit generation split"


def test_leakdir_empty_fails_loud(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no anchors"):
        G._load_leak_dir(empty, "leak fixtures")
    with pytest.raises(FileNotFoundError):
        G._load_leak_dir(tmp_path / "nonexistent", "leak fixtures")


def test_manipulation_check_table(tmp_path):
    inputs = {
        "manip9b": make_manip_doc(),
        "manip7b": make_manip_doc(
            special_axis="query_content", special_verdict="no_manipulation_check_query_class"
        ),
    }
    written = G.manipulation_check_table(inputs, tmp_path)
    assert {p.name for p in written} == {
        "table_manipulation_check.md",
        "table_manipulation_check.json",
    }
    md = (tmp_path / "table_manipulation_check.md").read_text()
    assert G.DISPLAY["qwen35_9b"] in md and G.DISPLAY["qwen25_7b"] in md
    assert "3/3 fired (floor 2: met)" in md
    assert "1/2 fired (floor 2: MISSED)" in md
    # special rows render their verdict, never a fabricated count
    assert G.DISPLAY["not_in_slice"] in md
    assert G.DISPLAY["no_manipulation_check_query_class"] in md
    assert "2/2 judged axes meet the fire floor" not in md  # 9B: 1 of 2 floors met
    doc = json.loads((tmp_path / "table_manipulation_check.json").read_text())
    assert doc["axes"]["register"]["qwen35_9b"]["floor_met"] is True
    assert doc["axes"]["user_fact"]["qwen25_7b"] is None  # not judged on the 7B side


def test_token_count_equality_table(tmp_path):
    inputs = {"bank": make_bank_doc()}
    written = G.token_count_equality_table(inputs, tmp_path)
    assert {p.name for p in written} == {
        "table_token_count_equality.md",
        "table_token_count_equality.json",
    }
    md = (tmp_path / "table_token_count_equality.md").read_text()
    assert "| Register | 12 | yes | 12 |" in md
    assert "9, 11" in md and "| no |" in md  # politeness: unequal q35 counts
    assert "1/2 names remain single-token" in md
    doc = json.loads((tmp_path / "table_token_count_equality.json").read_text())
    assert doc["within_axis_equal_q35"]["politeness"] is False


def test_token_count_equality_requires_gates(tmp_path):
    with pytest.raises(RuntimeError, match="token_gates"):
        G.token_count_equality_table({"bank": make_bank_doc(with_gates=False)}, tmp_path)


def test_cli_end_to_end_all_figs(tmp_path):
    """Production entrypoint (main(argv)) over the full registry."""
    paths = write_all_fixtures(tmp_path / "fixtures")
    out = tmp_path / "figs"
    rc = G.main(_argv(paths, out, "all"))
    assert rc == 0
    for stem in EXPECTED_FIG_STEMS:
        _assert_png(out, stem)
    for stem in EXPECTED_TABLE_STEMS:
        assert (out / f"{stem}.md").is_file()
        assert (out / f"{stem}.json").is_file()


def test_cli_optional_delta_absent(tmp_path):
    """matched_n_table's delta input is optional (``delta?``): the CLI still
    renders the table when minpair_delta_2587.json does not exist."""
    paths = write_all_fixtures(tmp_path / "fixtures")
    paths["delta"].unlink()
    out = tmp_path / "figs"
    rc = G.main(_argv(paths, out, "matched_n_table"))
    assert rc == 0
    doc = json.loads((out / "table_matched_n.json").read_text())
    assert "h1_paired_shared_rows" not in doc


def test_cli_missing_required_input_fails_loud(tmp_path):
    paths = write_all_fixtures(tmp_path / "fixtures")
    paths["sweep"].unlink()
    with pytest.raises(FileNotFoundError):
        G.main(_argv(paths, tmp_path / "figs", "hero_layer_sweep"))


def test_cli_unknown_fig_name_fails_loud(tmp_path):
    paths = write_all_fixtures(tmp_path / "fixtures")
    with pytest.raises(SystemExit):
        G.main(_argv(paths, tmp_path / "figs", "no_such_figure"))


def test_registry_covers_plan_deliverables():
    """Plan §6/§13 registry pin: the full deliverable set stays registered
    (drift guard for the r1 `plan-s6-figures-deliverable-gap` finding)."""
    expected = {
        "hero_layer_sweep",
        "crossmodel_axis_profile",
        "matched_n_table",
        "selected_lambda_per_layer",
        "floors_per_layer",
        "wc_transfer_per_layer",
        "knn_per_layer",
        "reliability_ceiling",
        "crossmodel_delta_forest",
        "matched_vs_parent_scatter",
        "delta_norm_scatter",
        "install_swap_violins",
        "edit_dose_scatter",
        "carrier_direction_heatmap",
        "axis_identity_heatmap",
        "crossfam_consistency_scatter",
        "delta_retrieval_acc",
        "splithalf_vs_direction",
        "pilot_axis_panels",
        "lstar_sensitivity_twins",
        "pooling_twin_scatter",
        "think_leak_cap_hit_table",
        "manipulation_check_table",
        "token_count_equality_table",
    }
    assert set(G.FIGS) == expected


def test_registry_names_are_snake_case_and_callable():
    for name, (req, fn) in G.FIGS.items():
        assert name == name.lower() and " " not in name
        assert callable(fn)
        assert isinstance(req, tuple) and req
        for key in req:
            assert key.rstrip("?") in G._INPUT_SPECS
    for key, (flag, _desc, kind) in G._INPUT_SPECS.items():
        assert kind in G._LOADERS, key
        assert flag.isidentifier(), flag
