"""CPU tests for scripts/issue823_ladder_ext_figures.py (#823 ext-ladder, unit 5).

Fixture-driven end-to-end render: a tmp_path results-dir is synthesized to the
REALIZED producer schemas of scripts/issue823_ladder_ext_fits.py (ladder_ext_r2
rung blocks + cells keyed ``{arm}:L{layer}``, per-rung compact
``correlated_offset_floor`` cells built through the REAL producer function
``issue823_ladder_common.correlated_floor_from_groups`` (r1 blocker
fits-analysis-handoff — the pod-local mixture_diffs.npz sidecars are no longer
consumed), contingency_fired/read_out_solver labels, both-metric knn_read_out
cells, shared_persona_paired per-layer cells with the full_ratio block,
percontext npz arrays, p2_ext_boundary cells, mask_ext refusal attribution),
then ``main()`` runs the REAL figure functions through ``savefig_paper`` and
writes ``ladder_ext_summary.json``. One paired cell carries a deliberately
INVERTED ``mean_paired_diff_ci95`` (the gotchas xerr/yerr rule: the clamp is
exercised through the real figure function to savefig), and one cell is
degenerate (``rho_point``/``rho_ci95`` None + ``rho_ci95_unstable``) to
exercise the nan-gap and 'neither'-by-fiat paths.

Fully synthetic fixtures in tmp_path — no network, no GPU, no committed
eval_results reads (no sparse_cones entry needed), no real-corpus text.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from scripts import issue823_ladder_common as LC
from scripts import issue823_ladder_ext_figures as FIGS

LABELS = ("48", "96")
ARMS = ("k1", "k16")
N_LAYERS = 28
POOL = np.arange(48, dtype=np.int64)  # fixed companion eval pool


def _knn_val(rng: np.random.Generator) -> dict:
    return {
        "metric": "euclidean",
        "n": 8,
        "n_pool": 8,
        "acc_at_k": {"1": float(rng.uniform(0.1, 0.5)), "5": float(rng.uniform(0.5, 0.9))},
        "chance_at_k": {"1": 0.125, "5": 0.625},
        "mrr": 0.3,
        "median_rank": 3.0,
    }


def _rung_block(rng: np.random.Generator, tag: str, n_mask: int, n_eval: int) -> dict:
    cells: dict = {}
    for arm in ARMS:
        for layer in range(N_LAYERS):
            cell = {
                "fold_lambdas": [float(v) for v in rng.uniform(0.01, 100.0, 5)],
                "fold_dofs": [float(v) for v in rng.uniform(2.0, 8.0, 5)],
                "fold_r2s": [float(v) for v in rng.uniform(-0.2, 0.8, 5)],
                "n_train_per_fold": [max(2, n_mask - n_mask // 5)] * 5,
                "solver": "dual",
                "identity_bias_pooled_r2": float(rng.uniform(-0.1, 0.4)),
            }
            if tag == "primary":
                cell["pooled_r2"] = float(rng.uniform(0.0, 0.8))
                cell["fold_mean_r2"] = float(rng.uniform(0.0, 0.8))
            else:
                cell["pooled_r2_eval"] = float(rng.uniform(0.0, 0.8))
            cells[f"{arm}:L{layer}"] = cell
    knn: dict = {}
    for arm in ARMS:
        for layer in FIGS.READ_OUT_LAYERS:
            if tag == "primary":
                for f in range(5):
                    knn[f"{arm}:L{layer}:fold{f}"] = {
                        m: _knn_val(rng) for m in ("euclidean", "cosine")
                    }
            else:
                knn[f"{arm}:L{layer}"] = {m: _knn_val(rng) for m in ("euclidean", "cosine")}
    # Compact correlated-offset floor through the REAL producer function
    # (issue823_ladder_common.correlated_floor_from_groups) so the fixture
    # carries the fits driver's realized 5-key schema, never a hand-typed copy.
    floor = {
        f"L{layer}": LC.correlated_floor_from_groups(
            [(2, rng.normal(size=(2, 6))), (3, rng.normal(size=(3, 6)))], n_persona0=2
        )
        for layer in FIGS.READ_OUT_LAYERS
    }
    return {
        "n_mask": n_mask,
        "n_eval": n_eval,
        "n_train_per_fold": [max(2, n_mask - n_mask // 5)] * 5,
        "d": 3584,
        "n_over_d_ratio": n_mask / 3584.0,
        "solver": "dual",
        "g2_verdict": "PASS",
        "lambda_edge_fraction": 0.0,
        "cells": cells,
        "knn_read_out": knn,
        "correlated_offset_floor": floor,
        "contingency_fired": False,
        "read_out_solver": "dual",
        "estimator_degenerate": True,
    }


def _paired_cell(
    rng: np.random.Generator,
    rho: float | None,
    ci: list | None,
    unstable: bool,
    diff_ci: list | None = None,
) -> dict:
    v = float(rng.normal(0.02, 0.01))
    lo, hi = (v - 0.05, v + 0.05) if diff_ci is None else diff_ci
    e_point = 0.4
    return {
        "behavior": "x",
        "n_shared_contexts": 3,
        "mean_paired_diff": v,
        "mean_paired_diff_ci95": [lo, hi],
        "median_paired_diff": v,
        "frac_contexts_pooled_worse": 0.5,
        "rho_ci95": ci,
        "n_negligible_E_draws": 7,
        "rho_ci95_unstable": unstable,
        "full_ratio": {
            "rho_point": rho,
            "mean_excess_point": v,
            "e_point_from_diffs": e_point,
            "n_persona0": 3,
            "n_boot": 100,
            "n_draws_retained": 93,
            "seed": 1,
            "note": "fixture",
        },
        "offset_bias_control": {"ratio_measured_over_full_energy": rho or 0.0},
    }


# (tag, label, layer) -> (rho, ci, unstable, diff_ci) overrides; the TOP rung's
# primary classes are artifact (L14) / real (L26) / neither (L17) so the lattice
# lands Partial-attenuation/mixed with n_art == n_real == 1.
_CELL_SPECS = {
    ("primary", "96", 14): (0.03, [0.01, 0.05], False, None),
    ("primary", "96", 26): (0.7, [0.6, 0.9], False, None),
    ("primary", "96", 17): (0.3, [0.2, 0.4], False, None),
    # deliberately INVERTED numerator CI (errorbar-clamp path, gotchas xerr rule)
    ("primary", "48", 26): (0.2, [0.1, 0.3], False, [0.05, -0.01]),
    # degenerate cell: no rho point, no CI, unstable -> 'neither' by fiat + nan gap
    ("primary", "48", 17): (None, None, True, None),
}


def _paired_json(rng: np.random.Generator, tag: str, label: str) -> dict:
    per_layer = {}
    for layer in FIGS.READ_OUT_LAYERS:
        rho, ci, unstable, diff_ci = _CELL_SPECS.get(
            (tag, label, layer), (0.2, [0.1, 0.3], False, None)
        )
        per_layer[f"L{layer}"] = _paired_cell(rng, rho, ci, unstable, diff_ci)
    return {
        "metadata": {"script": "fixture"},
        "arms": {
            "k16": {
                "n_shared_contexts_pre_mask": 6,
                "n_shared_contexts_post_mask": 3,
                "per_layer": per_layer,
            }
        },
    }


def _percontext_npz(rng: np.random.Generator, path: pathlib.Path, ids: np.ndarray) -> None:
    n = ids.size
    np.savez(
        path,
        arm_names=np.array(ARMS),
        context_ids=ids,
        p1_ss_res=rng.uniform(0.1, 1.0, size=(2, N_LAYERS, n)),
        p1_ss_tot=rng.uniform(1.0, 2.0, size=(2, N_LAYERS, n)),
        p1_identity_ss_res=rng.uniform(0.1, 1.0, size=(2, N_LAYERS, n)),
        p1_identity_ss_tot=rng.uniform(1.0, 2.0, size=(2, N_LAYERS, n)),
    )


def build_results_dir(root: pathlib.Path) -> pathlib.Path:
    rng = np.random.default_rng(0)
    rd = root / "results"
    rd.mkdir()
    r2 = {
        "estimator": {
            "primary": "gcv-dof-capped-0.9",
            "grid": ["logspace", -2, 4, 13],
            "wide_grid": ["logspace", -2, 8, 21],
            "fold_seed": 0,
            "n_folds": 5,
            "dual_n_max": 6000,
            "arms": list(ARMS),
        },
        "capture_drops": {"k1": 0, "k16": 0},
        "primary": {},
        "companion": {},
        "gates": {"gate_c": {"pass": True}, "gate_f_mask_integrity": "PASS"},
        "lambda_edge_fraction_trigger": 0.1,
    }
    for label in LABELS:
        n = int(label)
        ids = np.arange(n, dtype=np.int64)
        r2["primary"][label] = _rung_block(rng, "primary", n, n)
        r2["companion"][label] = _rung_block(rng, "companion", n, POOL.size)
        for tag, suffix, ids_t in (
            ("primary", f"rung{label}", ids),
            ("companion", f"rand_rung{label}", POOL),
        ):
            (rd / f"shared_persona_paired_{suffix}.json").write_text(
                json.dumps(_paired_json(rng, tag, label))
            )
            _percontext_npz(rng, rd / f"percontext_{suffix}.npz", ids_t)
    (rd / "ladder_ext_r2.json").write_text(json.dumps(r2))

    seeds = [0, 1]
    grid = [8, 3336]
    p2_cells = {
        f"L{layer}:n{n}:seed{s}": {
            "r2": float(rng.uniform(-0.5, 0.7)),
            "lambda": 1.0,
            "dof": 5.0,
            "lambda_top_edge": False,
            "n_train": n,
            "n_over_d": n / 3584.0,
            "solver": "dual",
            "identity_bias_r2": float(rng.uniform(-0.2, 0.3)),
            "knn": {"euclidean": _knn_val(rng), "cosine": _knn_val(rng)},
        }
        for layer in FIGS.READ_OUT_LAYERS
        for n in grid
        for s in seeds
    }
    (rd / "p2_ext_boundary.json").write_text(
        json.dumps(
            {
                "arm": "k1",
                "read_out_layers": list(FIGS.READ_OUT_LAYERS),
                "holdout_n": 12,
                "holdout_sha256": "f" * 64,
                "pool_n": 40,
                "pool_sha256": "e" * 64,
                "n_train_grid": grid,
                "draw_seeds": seeds,
                "d": 3584,
                "estimator": "gcv-dof-capped",
                "cells": p2_cells,
            }
        )
    )
    (rd / "mask_ext.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "mask_rule": "fixture",
                "rungs": {
                    label: {"rung": int(label), "n_mask": int(label), "n_over_d": 0.01}
                    for label in LABELS
                },
                "bridge": {"n_mask": 6, "ids": [0, 16, 32, 48, 64, 80]},
                "ext_arm_stats": {
                    "1": {"n_rows": 100, "refusal_fraction": 0.05},
                    "16": {"n_rows": 100, "refusal_fraction": 0.08},
                },
                "refusal_fraction_by_arm_persona": {
                    "1": {"0": 0.05},
                    "16": {str(p): 0.01 * p for p in range(16)},
                },
                "new_invalid_abort_fraction": 0.01,
                "integrity_gate": "PASS",
            }
        )
    )
    return rd


# ── Unit tests ────────────────────────────────────────────────────────────────


def test_band_class_partition():
    assert FIGS.band_class([0.01, 0.05], False) == "decisively-artifact"
    assert FIGS.band_class([0.6, 0.9], False) == "decisively-real"
    assert FIGS.band_class([0.2, 0.4], False) == "neither"
    assert FIGS.band_class([0.05, 0.2], False) == "neither"  # boundary-crossing
    assert FIGS.band_class([0.6, 0.9], True) == "neither"  # unstable by fiat
    assert FIGS.band_class(None, False) == "neither"


def test_err_offsets_clamps_inverted_ci():
    off = FIGS.err_offsets(np.array([0.02]), np.array([0.05]), np.array([-0.01]))
    assert off.shape == (2, 1)
    assert (off >= 0.0).all()


def test_import_check_mode():
    assert FIGS.main(["--import-check", "--results-dir", "unused"]) == 0


def test_load_all_fails_loud_on_missing_artifact(tmp_path):
    rd = build_results_dir(tmp_path)
    (rd / "shared_persona_paired_rung48.json").unlink()
    with pytest.raises(FileNotFoundError):
        FIGS.load_all(rd)


def test_floor_from_rung_block_fails_loud_on_pre_schema_block():
    """The compact floor rides the rung block (fits-analysis-handoff): a
    pre-schema block (no correlated_offset_floor) and a shallow cell both
    fail loud naming the artifact — never a silent npz fallback."""
    with pytest.raises(RuntimeError, match="correlated_offset_floor"):
        FIGS.floor_from_rung_block({"cells": {}}, "ladder_ext_r2.json:primary/48")

    ok = {
        "correlated_offset_floor": {
            f"L{ly}": LC.correlated_floor_from_groups([(2, np.ones((2, 4)))], 1)
            for ly in FIGS.READ_OUT_LAYERS
        }
    }
    out = FIGS.floor_from_rung_block(ok, "x")
    assert set(out) == set(FIGS.READ_OUT_LAYERS)
    assert all("floor_ratio" in cell for cell in out.values())

    bad = {"correlated_offset_floor": {f"L{FIGS.READ_OUT_LAYERS[0]}": {"floor_raw": 1.0}}}
    with pytest.raises(RuntimeError, match="missing/invalid"):
        FIGS.floor_from_rung_block(bad, "x")


# ── Lattice verdict branches (r1 concern lattice-verdict-branches-untested) ──

_CI = {"artifact": [0.01, 0.05], "real": [0.6, 0.9], "neither": [0.2, 0.4]}


def _lat_data(spec: dict) -> dict:
    """Minimal lattice input: per-(tag, label, layer) band class, default neither."""
    paired: dict = {}
    for tag in ("primary", "companion"):
        paired[tag] = {}
        for lab in LABELS:
            paired[tag][lab] = {}
            for layer in FIGS.READ_OUT_LAYERS:
                cls = spec.get((tag, lab, layer), "neither")
                paired[tag][lab][layer] = {"rho_ci95": _CI[cls], "rho_ci95_unstable": False}
    return {"labels": list(LABELS), "paired": paired}


def test_lattice_verdict_decisive_and_disagree_branches():
    # Interpolation-artifact: >=2 top-rung primary artifact layers, 0 real, no disagree.
    art = {("primary", "96", ly): "artifact" for ly in FIGS.READ_OUT_LAYERS}
    v = FIGS.lattice_verdict(_lat_data(art))
    assert v["label"] == "Interpolation-artifact"
    assert v["n_artifact_layers"] == 3 and v["n_real_layers"] == 0
    assert v["n_ladder_disagree_layers"] == 0

    # Origin-effect-real: >=2 top-rung primary real layers, 0 artifact.
    real = {("primary", "96", ly): "real" for ly in FIGS.READ_OUT_LAYERS[:2]}
    v = FIGS.lattice_verdict(_lat_data(real))
    assert v["label"] == "Origin-effect-real"
    assert v["n_real_layers"] == 2 and v["n_artifact_layers"] == 0

    # Ladder DISAGREEMENT forces mixed even with a decisive artifact count:
    # primary artifact vs companion real at the SAME layer (any rung).
    l0 = FIGS.READ_OUT_LAYERS[0]
    dis = dict(art)
    dis[("companion", "96", l0)] = "real"
    v = FIGS.lattice_verdict(_lat_data(dis))
    assert v["label"] == "Partial-attenuation/mixed"
    assert v["n_ladder_disagree_layers"] == 1 and v["n_artifact_layers"] == 3
    assert v["band_class"]["companion"]["96"][f"L{l0}"] == "decisively-real"

    # Exactly ONE decisive layer is not enough for either decisive label.
    one = {("primary", "96", l0): "artifact"}
    assert FIGS.lattice_verdict(_lat_data(one))["label"] == "Partial-attenuation/mixed"


# ── End-to-end render + summary ──────────────────────────────────────────────


def test_end_to_end_renders_and_summary(tmp_path):
    rd = build_results_dir(tmp_path)
    out_dir = tmp_path / "figs"
    # NO --parent-refusal-json flag: the default path (the p0ext-persisted
    # parent_refusal_by_persona.json in the results dir, WRAPPED producer
    # shape) is discovered + unwrapped by main().
    (rd / "parent_refusal_by_persona.json").write_text(
        json.dumps(
            {
                "metadata": {"script": "fixture"},
                "refusal_fraction_by_persona": {str(p): 0.02 for p in range(16)},
            }
        )
    )
    rc = FIGS.main(["--results-dir", str(rd), "--out-dir", str(out_dir), "--formats", "png"])
    assert rc == 0

    stems = [stem for stem, _fn in FIGS.FIGURES]
    assert len(stems) == 10
    for stem in stems:
        png = out_dir / f"{stem}.png"
        assert png.exists(), stem
        assert png.stat().st_size > 4096, stem  # non-empty render
        assert (out_dir / f"{stem}.meta.json").exists(), stem

    summary = json.loads((rd / "ladder_ext_summary.json").read_text())
    for key in (
        "metadata",
        "read_out_layers",
        "estimator",
        "gates",
        "rungs",
        "lattice",
        "diagnostics",
        "refusal",
        "figures",
    ):
        assert key in summary, key
    assert len(summary["figures"]) == 10
    assert summary["metadata"]["source_artifacts"]  # sha256 provenance map non-empty

    # Lattice: top rung 96 primary classes are artifact/real/neither by fixture.
    lat = summary["lattice"]
    assert lat["top_rung"] == "96"
    assert lat["n_artifact_layers"] == 1
    assert lat["n_real_layers"] == 1
    assert lat["n_ladder_disagree_layers"] == 0
    assert lat["label"] == "Partial-attenuation/mixed"
    assert lat["band_class"]["primary"]["96"]["L14"] == "decisively-artifact"
    assert lat["band_class"]["primary"]["48"]["L17"] == "neither"  # unstable by fiat

    # Headline rows: rho + CI + hygiene medians per ladder per rung.
    row = summary["rungs"]["96"]["primary"]
    assert row["g2_verdict"] == "PASS"
    assert isinstance(row["lambda_median"], float)
    assert isinstance(row["dof_over_ntrain_median"], float)
    # Gate-D contingency labels ride the summary rows (r1 blocker
    # gate-d-contingency-incoherent: the realized solver is visible downstream).
    assert row["contingency_fired"] is False and row["read_out_solver"] == "dual"
    # kNN echo: fold-mean acc@{1,5} for BOTH metrics per arm x read-out layer.
    echo = row["knn_read_out_mean"]
    assert echo["k1"]["L14"]["n_folds"] == 5  # primary carries per-fold cells
    for metric in ("euclidean", "cosine"):
        m = echo["k16"]["L14"][metric]
        assert 0.0 <= m["acc_at_1"] <= 1.0 and 0.0 <= m["acc_at_5"] <= 1.0
        assert m["n_pool_mean"] == 8.0
    comp = summary["rungs"]["96"]["companion"]
    assert comp["knn_read_out_mean"]["k1"]["L14"]["n_folds"] == 1  # single fold-mean cell
    l14 = row["per_layer"]["L14"]
    assert l14["rho_ci95"] == [0.01, 0.05]
    assert "k1" in l14["pooled_r2"] and "k16" in l14["pooled_r2"]
    degenerate = summary["rungs"]["48"]["primary"]["per_layer"]["L17"]
    assert degenerate["rho_point"] is None and degenerate["rho_ci95_unstable"] is True

    # Registered diagnostics: fixed banked subset (rung 96 shared ∩ bridge = 6 ids)
    diag = summary["diagnostics"]
    assert diag["fixed_banked_subset"]["n_banked_shared"] == 6
    assert diag["fixed_banked_subset"]["primary"]["96"]["L14"]["n"] == 6
    assert diag["fixed_banked_subset"]["primary"]["48"]["L14"]["n"] == 3
    floor = diag["correlated_offset_floor"]["primary"]["96"]["L14"]
    assert floor["floor_ratio"] is None or floor["floor_ratio"] >= 0.0
    assert floor["e_point_from_diffs"] > 0.0
    assert floor["n_nonzero"] == 5 and floor["n_persona0"] == 2  # producer-schema fields
