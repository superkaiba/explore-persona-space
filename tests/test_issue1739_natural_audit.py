"""Integrity tests for the analysis-only natural scaling consumer."""

import copy
import json

import pytest

from scripts import issue1739_natural_audit as audit


@pytest.mark.parametrize(
    "protocol,fit,rung,want",
    [
        ("P-B", "P-B-holdout-hhrt", "hhrt", True),
        ("P-B", "P-B-holdout-hhrt", "heldin:train", False),
        ("P-B", "P-B-holdout-hhrt", "wildchat_rung", False),
        ("P-A", "P-B-holdout-hhrt", "hhrt", False),
    ],
)
def test_primary_selection(protocol, fit, rung, want):
    assert audit.primary_row({"protocol": protocol, "fit": fit, "eval_rung": rung}) is want


def prediction_fixture(tmp_path):
    path = tmp_path / "P-B-holdout-hhrt.jsonl"
    rows, expected = [], []
    for arm, layer in audit.FROZEN["evil"].items():
        for i in range(4):
            rows.append(
                {
                    "protocol": "P-B",
                    "fit": path.stem,
                    "map_variant": "true",
                    "arm": arm,
                    "rung": "hhrt",
                    "context_id": str(i),
                    "group": "g",
                    "layer": layer,
                    "dv": i,
                    "score": i,
                }
            )
        expected.append({"arm": arm, "eval_rung": "hhrt", "n_eval": 4, "rho_frozen": 1.0})
    return path, rows, expected


def test_predictions_recompute_and_pair(tmp_path):
    path, rows, expected = prediction_fixture(tmp_path)
    path.write_text("\n".join(map(json.dumps, reversed(rows))))
    result = audit.validate_predictions(path, expected, audit.FROZEN["evil"])
    assert result["rows"] == 12
    assert result["max_abs_rho_recompute_error"] == 0


@pytest.mark.parametrize(
    "change", ["duplicate", "dv", "key", "layer", "nan", "missing_arm", "score"]
)
def test_predictions_fail_loudly(tmp_path, change):
    path, rows, expected = prediction_fixture(tmp_path)
    if change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "dv":
        rows[0]["dv"] = 90
    elif change == "key":
        rows[0]["group"] = "wrong"
    elif change == "layer":
        rows[0]["layer"] = 0
    elif change == "nan":
        rows[0]["score"] = float("nan")
    elif change == "missing_arm":
        rows = rows[4:]
    else:
        rows[0]["score"] = 100
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises((ValueError, KeyError)):
        audit.validate_predictions(path, expected, audit.FROZEN["evil"])


def knn_fixture():
    return {
        metric: {
            "metric": metric,
            "n": 100,
            "n_pool": 100,
            "acc_at_k": {"1": 0.5, "5": 0.7},
            "chance_at_k": {"1": 0.01, "5": 0.05},
        }
        for metric in ("euclidean", "cosine")
    }


def test_knn_pool_and_chance():
    block = knn_fixture()
    audit.validate_knn(block)
    block["cosine"]["chance_at_k"]["1"] = 0.1
    with pytest.raises(ValueError, match="chance"):
        audit.validate_knn(block)


def test_knn_expected_pool():
    with pytest.raises(ValueError, match="evaluation rows"):
        audit.validate_knn(knn_fixture(), expected_n=101)


def cell_fixture(behavior="evil", seed=0):
    return {
        "behavior": behavior,
        "generic_u": 100000,
        "seed": seed,
        "selected_context_ids_sha256": "same-sample",
        "manifest_sha256": "same-manifest",
        "input_sha256": {"fixed-input": "same-digest"},
        "primary": [
            {"arm": arm, "rho": rho}
            for arm, rho in zip(audit.NATURAL_ROSTER, (0.2, 0.3, 0.4), strict=True)
        ],
    }


def test_descriptive_aggregation_not_pseudoreplication():
    cells = [cell_fixture(seed=s) for s in range(5)]
    result = audit.aggregate(cells, require_full=False)
    assert result["realized_cells"] == 5
    assert len(result["missing_cells"]) == 145
    curve = result["curves"][0]
    assert curve["n_seeds"] == 5
    assert curve["statistics"]["mapped_minus_context"]["mean"] == pytest.approx(0.1)
    assert "not a confidence" in result["uncertainty"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "sample",
        "undefined",
        "manifest",
        "fixed_input",
        "changed_path",
        "empty_inputs",
    ],
)
def test_aggregate_rejects_incomplete_or_crossed(fault):
    cells = [cell_fixture()]
    if fault == "duplicate":
        cells *= 2
    elif fault == "sample":
        cells.append(cell_fixture("sycophancy"))
        cells[-1]["selected_context_ids_sha256"] = "different"
    elif fault == "undefined":
        cells[0]["primary"][0]["rho"] = float("nan")
    elif fault in {"manifest", "fixed_input"}:
        cells.append(cell_fixture(seed=1))
        if fault == "manifest":
            cells[-1]["manifest_sha256"] = "different"
        else:
            cells[-1]["input_sha256"]["fixed-input"] = "different"
    elif fault in {"changed_path", "empty_inputs"}:
        cells.append(cell_fixture(seed=1))
        cells[-1]["input_sha256"] = {"replacement": "new-digest"} if fault == "changed_path" else {}
    with pytest.raises(ValueError):
        audit.aggregate(cells, require_full=fault == "missing")


def write_cell_fixture(tmp_path, fault=None):
    """Small artificial integrity fixture, not an experiment result."""
    cell = tmp_path / "u250" / "hallucination" / "seed0"
    (cell / "transfer_preds").mkdir(parents=True)
    layers = [18, 20]
    frozen = audit.FROZEN["hallucination"]

    def knn(n):
        block = knn_fixture()
        for value in block.values():
            value.update(n=n, n_pool=n, chance_at_k={"1": 1 / n, "5": 5 / n})
        return block

    meta = {
        "behavior": "hallucination",
        "seed": 0,
        "git_commit": "test-sha",
        "arms": list(audit.NATURAL_ROSTER),
        "protocols": ["B"],
        "map_variants": ["true"],
        "judge_called": False,
        "natural_pool": {
            "generic_u": 250,
            "n_rows": 100000,
            "no_recombination": True,
            "selected_global_layers": layers,
            "selected_context_ids_sha256": "sample",
            "manifest_sha256": "manifest",
        },
        "frozen_layers": {arm: layers.index(layer) for arm, layer in frozen.items()},
        "datasets": {"train": 16000, "nqopen": 12, "simpleqa": 12},
        "pb_pools": [],
        "fit_reports": [],
        "natural_transfer_pred_files": [],
        "n_u": 16250,
        "wall_s": 1.0,
        "input_sha256": {"input": "hash"},
    }
    transfer_rows = []
    for holdout in sorted(audit.HOLDOUTS["hallucination"]):
        fit = f"P-B-holdout-{holdout}"
        meta["pb_pools"].append({"holdout": holdout})
        meta["natural_transfer_pred_files"].append(f"transfer_preds/{fit}.jsonl")
        mapped = {
            "n_rows": 12,
            "per_layer": [
                {"layer_idx": li, "r2_eval_rung": 0.3, "knn": knn(12)} for li in range(2)
            ],
        }
        if fault == "empty_reconstruction":
            mapped["per_layer"] = []
        elif fault == "duplicate_reconstruction_layer":
            mapped["per_layer"][1]["layer_idx"] = 0
        elif fault == "wrong_reconstruction_pool":
            mapped["per_layer"][0]["knn"] = knn(13)
        meta["fit_reports"].append(
            {
                "protocol": "P-B",
                "fit": fit,
                "map_variant": "true",
                "d": 3584,
                "n_readout": 12000,
                "leakage": {"asserts": "passed"},
                "recon": {"per_rung": {holdout: mapped}},
                "recon_identity_bias": {
                    "per_rung": {holdout: {"n_rows": 12, "r2_identity_bias_per_layer": [0.0, 0.0]}}
                },
            }
        )
        predictions = []
        for arm, layer in frozen.items():
            prov = {
                "protocol": "P-B",
                "fit": fit,
                "map_variant": "true",
                "arm": arm,
                "behavior": "hallucination",
                "seed": 0,
                "layer": layer,
            }
            transfer_rows.append({**prov, "eval_rung": holdout, "rho_frozen": 1.0, "n_eval": 12})
            predictions.extend(
                {**prov, "rung": holdout, "context_id": str(i), "group": "g", "dv": i, "score": i}
                for i in range(12)
            )
        (cell / "transfer_preds" / f"{fit}.jsonl").write_text(
            "\n".join(map(json.dumps, predictions))
        )
    diag = {
        "add_n_generic": 250,
        "add_n_eliciting": 16000,
        "add_realized_pool": 16250,
        "w_fit_rows": 16250,
        "w_refit_on_full_u": True,
        "n_train": 13000,
        "n_holdout": 3250,
        "per_layer": [
            {"layer_idx": li, "r2_map": 0.3, "r2_identity_bias": 0.0, "knn": knn(3250)}
            for li in range(2)
        ],
    }
    if fault == "duplicate_map_layer":
        diag["per_layer"][1]["layer_idx"] = 0
    elif fault == "wrong_map_pool":
        diag["per_layer"][0]["knn"] = knn(3200)
    elif fault == "wrong_trait_count":
        diag["add_n_eliciting"] = 15999
    elif fault == "wrong_sha":
        meta["git_commit"] = "different"
    elif fault == "missing_primary":
        transfer_rows.pop()
    (cell / "all_arms_spearman.json").write_text(
        json.dumps({"meta": meta, "transfer_rows": transfer_rows, "transfer_skips": []})
    )
    (cell / "readout_pools.json").write_text(
        json.dumps({"pools": meta["pb_pools"], "fit_reports": meta["fit_reports"]})
    )
    (cell / "map_diagnostics.json").write_text(json.dumps({"linear": diag}))
    return cell


def test_validate_cell_end_to_end(tmp_path):
    result = audit.validate_cell(write_cell_fixture(tmp_path), commit="test-sha")
    assert result["status"] == "PASS"
    assert len(result["primary"]) == 6
    assert len(result["reconstruction"]) == 4
    assert len(result["files_sha256"]) == 5


@pytest.mark.parametrize(
    "fault",
    [
        "empty_reconstruction",
        "duplicate_reconstruction_layer",
        "wrong_reconstruction_pool",
        "duplicate_map_layer",
        "wrong_map_pool",
        "wrong_trait_count",
        "wrong_sha",
        "missing_primary",
    ],
)
def test_validate_cell_rejects_incomplete_or_crossed(tmp_path, fault):
    with pytest.raises(ValueError):
        audit.validate_cell(write_cell_fixture(tmp_path, fault), commit="test-sha")
