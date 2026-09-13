"""Pairing, normalization, undefined statistics and control-contrast regression tests."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_comparison import (
    PREDICTORS,
    ROTATIONS,
    TARGETS,
    align_arrays,
    cell_key,
    combine_paired,
    paired_cohort,
    quality_matched_contrasts,
    registered_contrasts,
    rotation_variation,
)
from explore_persona_space.analysis.workspace_components import paired_context_bootstrap
from explore_persona_space.analysis.workspace_fit import evaluate_component_fits
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json


def bootstrap_fixture(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    targets = {name: rng.normal(size=(9, 3)) * scale for name in TARGETS}
    predictions = {
        predictor: {
            name: value + rng.normal(size=value.shape) * 0.2 for name, value in targets.items()
        }
        for predictor in PREDICTORS
    }
    boot = paired_context_bootstrap(
        targets, predictions, [str(i) for i in range(9)], n_bootstrap=50, seed=42
    )
    return targets, predictions, boot


def test_control_adjusted_gap_uses_paired_replicates_and_own_variances():
    keys = [cell_key("primary", "observed", 10, rotation) for rotation in ROTATIONS]
    boots = {
        key: bootstrap_fixture(seed=index, scale=index + 1)[2] for index, key in enumerate(keys)
    }
    contrasts, missing = registered_contrasts(keys)
    summary, samples = combine_paired(boots, contrasts)
    name = f"{keys[0]}/ridge/G_J_minus_mean_rotated"
    gaps = [boot["samples"]["ridge/restJ"] - boot["samples"]["ridge/J"] for boot in boots.values()]
    expected = gaps[0] - np.stack(gaps[1:]).mean(axis=0)
    np.testing.assert_allclose(samples[name], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(summary[name]["interval"], np.quantile(expected, [0.025, 0.975]))
    assert missing
    variation = rotation_variation(summary)["primary/observed/k10/ridge/G_J"]
    assert variation["status"] == "ok" and len(variation["estimates"]) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("counts_sha256", "changed"),
        ("seed", 13),
        ("context_ids", [str(i) for i in range(8, -1, -1)]),
    ],
)
def test_matching_marginal_counts_do_not_establish_pairing(field, value):
    boot = bootstrap_fixture()[2]
    changed = copy.deepcopy(boot)
    changed[field] = value
    with pytest.raises(ValueError, match="exact context order and draws"):
        combine_paired({"a": boot, "b": changed}, {"difference": {"a/ridge/J": 1, "b/ridge/J": -1}})


def test_cohort_intersection_is_explicit_and_row_order_is_resolved():
    common, report = paired_cohort({"a": ["z", "b", "a"], "b": ["a", "x", "b"]})
    assert common == ["a", "b"]
    assert report["cells"]["a"]["excluded_from_pairing"] == ["z"]
    target = np.array([[9.0], [2.0], [1.0]])
    targets, predictions = align_arrays(
        ["z", "b", "a"], common, {"J": target}, {"ridge": {"J": target + 3}}
    )
    np.testing.assert_array_equal(targets["J"], [[1], [2]])
    np.testing.assert_array_equal(predictions["ridge"]["J"], [[4], [5]])
    with pytest.raises(ValueError, match="Duplicate"):
        align_arrays(["a", "a"], ["a"], {}, {})
    with pytest.raises(ValueError, match="Fewer than two"):
        paired_cohort({"a": ["a", "b"], "b": ["a", "c"]})


def test_undefined_component_is_visible_not_zero_or_dropped():
    y = {"J": np.ones((5, 2)), "restJ": np.arange(10).reshape(5, 2)}
    boot = paired_context_bootstrap(y, {"ridge": y}, list("abcde"), n_bootstrap=25, seed=42)
    summary, draws = combine_paired({"a": boot}, {"gap": {"a/ridge/restJ": 1, "a/ridge/J": -1}})
    assert summary["gap"]["estimate"] is None
    assert summary["gap"]["interval"] is None
    assert summary["gap"]["n_valid_draws"] == 0
    assert np.isnan(draws["gap"]).all()


def test_identical_model_outputs_have_exactly_zero_paired_cross_difference():
    boot = bootstrap_fixture()[2]
    keys = [cell_key(role, "observed", 10, None) for role in ("primary", "comparison")]
    contrasts, _ = registered_contrasts(keys)
    summary, draws = combine_paired(dict.fromkeys(keys, boot), contrasts)
    name = "cross_model/observed/k10/rotationNone/ridge/lens_disagreement_primary_minus_comparison"
    np.testing.assert_allclose(draws[name], 0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(summary[name]["interval"], 0, rtol=0, atol=1e-15)


def test_absent_control_is_reported_and_cannot_be_averaged_as_two_or_zero():
    keys = [cell_key("primary", "observed", 10, rotation) for rotation in ROTATIONS[:3]]
    contrasts, missing = registered_contrasts(keys)
    name = f"{keys[0]}/ridge/G_R_minus_mean_rotated"
    assert name not in contrasts
    row = next(row for row in missing if row["contrast"] == name)
    assert row["missing_cells"] == [cell_key("primary", "observed", 10, 20260915)]
    sparse = {
        f"{cell_key('primary', 'observed', 10, seed)}/ridge/G_J": {"estimate": value}
        for seed, value in ((20260913, 0.1), (20260915, 0.3))
    }
    variation = rotation_variation(sparse)["primary/observed/k10/ridge/G_J"]
    assert variation["estimates"] == [0.1, None, 0.3]
    assert variation["mean"] is None


def test_calibration_matching_does_not_promote_nearest_unmatched_control():
    report = {"matching": {arm: {} for arm in ("J", "R")}}
    for arm in ("J", "R"):
        for k in (5, 10, 25):
            report["matching"][arm][str(k)] = {
                str(seed): {
                    "selected_k": 10 if k == 5 else 25,
                    "status": "approximate_within_range" if k < 25 else "unmatched_outside_range",
                    "residual_energy_fraction_mismatch": 0.02,
                }
                for seed in ROTATIONS[1:]
            }
    available = [
        cell_key("primary", "observed", k, seed) for k in (5, 10, 25) for seed in ROTATIONS
    ]
    contrasts, ledger = quality_matched_contrasts("primary", report, available)
    assert ledger["primary/k25/J"]["status"] == "excluded"
    assert not any("/k25/rotationNone/" in key for key in contrasts)
    native = cell_key("primary", "observed", 5, None)
    terms = contrasts[f"{native}/ridge/G_J_minus_calibration_matched_rotated"]
    assert terms[f"{cell_key('primary', 'observed', 10, 20260913)}/ridge/restJ"] == -1 / 3
    report["matching"]["J"]["5"]["20260913"] = {
        "selected_k": None,
        "status": "undefined_zero_energy",
    }
    _, undefined = quality_matched_contrasts("primary", report, available)
    assert undefined["primary/k5/J"]["status"] == "excluded"


def test_real_saved_fit_consumer_and_resume_reject_changed_artifacts(tmp_path):
    """Run actual ridge/MLP producers, file verification and bootstrap consumer bodies."""
    api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/workspace_jr_compare.py"))
    rng = np.random.default_rng(19)
    x = {s: rng.normal(size=(n, 3)) for s, n in (("train", 12), ("validation", 6), ("test", 8))}
    ids = {s: [f"{s}-{i}" for i in range(len(a))] for s, a in x.items()}
    targets = {
        s: {"full": a * 5, "J": a * 2, "restJ": a * 3, "R": a * 4, "restR": a} for s, a in x.items()
    }
    config = {
        "seed": 42,
        "fit": {
            "ridge_alpha_grid": [0.001, 1],
            "mlp_hidden": [4],
            "mlp_learning_rates": [0.001],
            "mlp_weight_decay": 0.0001,
            "mlp_max_epochs": 2,
            "mlp_patience": 2,
            "mlp_seeds": [42, 137, 271],
            "retrieval": {"k": [1], "metrics": ["euclidean"]},
        },
        "statistics": {
            "bootstrap": {"draws": 16, "confidence": 0.95},
            "smallest_practical_gap": 0.05,
            "near_zero_component_norm_floor": "1e-6_times_training_median_full_target_norm",
        },
    }
    root = tmp_path / "producer"
    fit = root / "fits/main/k10-rotationNone"
    result = evaluate_component_fits(x, targets, ids, config, fit, mlp_device="cpu")
    identity = {
        "config_sha256": "fixture-config",
        "selection_sha256": "fixture-selection",
        "model_role": "primary",
        "versions": {},
        "code": {"git_commit": "a" * 40, "git_dirty": False},
        "execution_readiness_sha256": "fixture-readiness",
    }
    contract = {
        "identity": identity,
        "k": 10,
        "rotation": None,
        "dictionaries": {"pilot_only": False},
    }
    save_json(root / "dictionaries/manifest.json", contract["dictionaries"])
    coverage = {
        s: {
            "identity": identity,
            "contract": contract,
            "status": "complete",
            "included_prompt_sha256": contexts,
            "exclusions": [],
            "planned_contexts": len(contexts),
        }
        for s, contexts in ids.items()
    }
    save_json(
        fit / "input_manifest.json",
        {"identity": identity, "contract": contract, "coverage": coverage},
    )
    save_json(root / "terminal.json", {"exit_code": 0, "phase": "complete"})
    selection = {
        "subsets": {
            f"main_{s}": [{"prompt_sha256": context} for context in contexts]
            for s, contexts in ids.items()
        }
    }
    checksums = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    receipt = tmp_path / "upload.json"
    save_json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "a" * 40,
            "prefix": "exploratory_workspace_jr/test-fixture",
            "files_verified": len(checksums),
            "verified_sha256": checksums,
        },
    )
    entry = {
        "root": str(root),
        "fit_relative": "fits/main/k10-rotationNone",
        "upload_receipt": str(receipt),
        "terminal_relative": "terminal.json",
        "kind": "observed",
        "k": 10,
        "rotation": None,
        "role": "primary",
    }
    cell = api["read_cell"](entry, config, identity, selection)
    ys, predictions = api["read_arrays"](cell)
    np.testing.assert_array_equal(ys["J"], targets["test"]["J"])
    assert set(predictions) == set(PREDICTORS)
    folder = tmp_path / "consumer"
    boot, target_hash = api["checkpoint_bootstrap"](folder, cell, ids["test"], config, resume=False)
    resumed, resumed_hash = api["checkpoint_bootstrap"](
        folder, cell, ids["test"], config, resume=True
    )
    assert target_hash == resumed_hash
    np.testing.assert_array_equal(boot["samples"]["ridge/J"], resumed["samples"]["ridge/J"])
    mixed, _ = combine_paired(
        {"fresh": boot, "resumed": resumed}, {"same": {"fresh/ridge/J": 1, "resumed/ridge/J": -1}}
    )
    assert mixed["same"]["estimate"] == 0
    key = cell_key("primary", "observed", 10, None)
    api["bind_source_families"]({key: cell})
    changed = copy.deepcopy(cell)
    changed["result"]["input_fingerprints"]["train"]["x"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="context inputs"):
        api["bind_source_families"]({key: cell, "changed": changed})
    global_proof = tmp_path / "proof.json"
    evidence = {"quality": {"primary": {"report_sha256": "old"}}}
    api["bind_global_sources"](global_proof, evidence, resume=False)
    api["bind_global_sources"](global_proof, evidence, resume=True)
    with pytest.raises(ValueError, match="source evidence changed"):
        api["bind_global_sources"](
            global_proof, {"quality": {"primary": {"report_sha256": "new"}}}, resume=True
        )
    with pytest.raises(ValueError, match="different source or cohort"):
        api["checkpoint_bootstrap"](folder, cell, ids["test"][::-1], config, resume=True)
    result["metrics"]["ridge"]["J"]["r2"] += 0.1
    cell["result"] = result
    with pytest.raises(ValueError, match="Reported r2"):
        api["read_arrays"](cell)
    serialized = json.loads((fit / "results.json").read_text())
    serialized["metrics"]["ridge"]["J"]["r2"] += 0.1
    (fit / "results.json").write_text(json.dumps(serialized))
    with pytest.raises(ValueError, match="verified producer upload"):
        api["read_cell"](entry, config, identity, selection)
