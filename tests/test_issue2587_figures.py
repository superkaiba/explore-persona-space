"""CPU-only pins for scripts/issue2587_figures.py (issue #2587 unit 6).

No network, no HF fetch, no GPU, no torch. Fixture docs mirror the REALIZED
schemas of the producers:

* ``map_layer_sweep.json`` / ``matched7b_anchor.json`` — mirrored from the
  writer code in ``scripts/issue2587_fits.py`` (``run_finalize`` merged doc,
  ``run_matched7b`` record; same round, so the writer IS the schema source).
* ``crossmodel_contrasts.json`` / ``minpair_delta_2587.json`` — mirrored from
  ``scripts/issue2587_analysis.py`` (``crossmodel_contrasts`` rows, ``main``'s
  merged doc).
* the banked #2330 reference fits — schema probed on the COMMITTED artifact
  ``eval_results/issue_2330/matched_fits_q35_n10k.json`` (top-level ``layers``
  + ``per_layer[str].ridge.test_r2``).

Every test renders through the REAL figure functions to ``savefig`` (tmp_path
only — never canonical ``figures/`` paths), and one test drives the
production CLI entrypoint (``main(argv)``) end-to-end.
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


# ---------------------------------------------------------------------------
# Fixture builders (module-level: also driven by the pre-commit CLI smoke)
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


def make_delta_doc() -> dict:
    return {
        "h1": {
            "r2_9b_lstar": 0.70,
            "r2_7b_l19": 0.71,
            "delta_map": -0.01,
            "delta_ci95": [-0.03, 0.01],
            "verdict": "h1_inconclusive",
        }
    }


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


def write_all_fixtures(dest: str | Path) -> dict[str, Path]:
    """Write every fixture JSON to ``dest`` (also used by the CLI smoke)."""
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    paths = {
        "sweep": d / "map_layer_sweep.json",
        "matched7b": d / "matched7b_anchor.json",
        "delta": d / "minpair_delta_2587.json",
        "crossmodel": d / "crossmodel_contrasts.json",
        "ref9b": d / "matched_fits_q35_n10k.json",
        "ref7b": d / "matched_fits_q25_n10k.json",
    }
    paths["sweep"].write_text(json.dumps(make_sweep_doc()))
    paths["matched7b"].write_text(json.dumps(make_matched7b_doc()))
    paths["delta"].write_text(json.dumps(make_delta_doc()))
    paths["crossmodel"].write_text(json.dumps(make_crossmodel_doc()))
    paths["ref9b"].write_text(json.dumps(make_ref2330_doc([16, 22, 30], 0.62)))
    paths["ref7b"].write_text(json.dumps(make_ref2330_doc([14, 19, 26], 0.66)))
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
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


def test_axis_profile_handles_null_rows(tmp_path):
    inputs = {"crossmodel": make_crossmodel_doc()}
    G.fig_crossmodel_axis_profile(inputs, tmp_path)
    _assert_png(tmp_path, "fig_hero_crossmodel_axis_profile")


def test_err_offsets_never_negative():
    import numpy as np

    vals = np.array([0.5, -0.4])
    lo = np.array([0.6, -0.35])  # inverted below
    hi = np.array([0.4, -0.45])  # inverted above
    off = G._err_offsets(vals, lo, hi)
    assert (off >= 0).all()


def test_cli_end_to_end_all_figs(tmp_path):
    """Production entrypoint (main(argv)) over the full registry."""
    paths = write_all_fixtures(tmp_path / "fixtures")
    out = tmp_path / "figs"
    rc = G.main(_argv(paths, out, "all"))
    assert rc == 0
    for stem in (
        "fig_hero_layer_sweep",
        "fig_hero_crossmodel_axis_profile",
        "fig_crossmodel_delta_forest",
        "fig_selected_lambda_per_layer",
        "fig_floors_per_layer",
        "fig_wc_transfer_per_layer",
        "fig_knn_per_layer",
        "fig_reliability_ceiling",
    ):
        _assert_png(out, stem)
    assert (out / "table_matched_n.md").is_file()
    assert (out / "table_matched_n.json").is_file()


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


def test_registry_names_are_snake_case_and_callable():
    for name, (req, fn) in G.FIGS.items():
        assert name == name.lower() and " " not in name
        assert callable(fn)
        assert isinstance(req, tuple) and req
        for key in req:
            assert key.rstrip("?") in G._INPUT_FLAGS
