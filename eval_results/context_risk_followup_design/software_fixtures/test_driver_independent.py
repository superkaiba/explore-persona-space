"""Independent driver software fixtures; future experiment artifacts remain mocked."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from scripts import context_risk_followup_analyze as driver

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (ROOT / "eval_results/context_risk_followup_design/analysis_spec.json").read_text()
)


class SoftwareFeatureBank:
    """A one-dimensional software design carrying row identity, without feature fitting."""

    def features(self, method, train, test, trials, rank=None):
        del trials
        info = {"method": method}
        if rank is not None:
            info.update(nominal_rank=rank, effective_rank=min(rank, len(train)))
        return train[:, None].astype(float), test[:, None].astype(float), info


def software_fit(path, fingerprint, left, right, positive, trials, c_value, basis):
    """Fixed software predictions deliberately distinguish weighted from fold-mean loss."""
    del path, fingerprint, left, positive, trials, basis
    probabilities = {0.1: {0: 0.1, 1: 0.9, 2: 0.3}, 1.0: {0: 0.2, 1: 0.1, 2: 0.3}}
    p = np.array([probabilities[c_value][int(row[0])] for row in right])
    return {
        "logits": np.log(p / (1 - p)),
        "coef": None,
        "intercept": 0,
        "status": "software_fixture",
        "iterations": 0,
        "elapsed_seconds": 0,
    }


def fit_software_method(tmp_path, method):
    """Exercise the real selection loop with all model fitting replaced by fixed values."""
    spec = copy.deepcopy(SPEC)
    spec["regularization_C"] = [0.1, 1.0]
    folds = [(np.array([1]), np.array([0])), (np.array([0]), np.array([1]))]
    with (
        patch.object(driver, "cached_fit", side_effect=software_fit) as fits,
        patch.object(driver, "l2_basis", side_effect=lambda a, b: (a, b, np.eye(a.shape[1]))),
    ):
        result = driver.fit_method(
            SoftwareFeatureBank(),
            method,
            np.array([0, 1]),
            np.array([2]),
            folds,
            np.array([0, 0, 1]),
            np.array([100, 1, 4]),
            tmp_path,
            "software_fixture",
            spec,
        )
    return result, fits.call_count


def test_cv_selection_uses_all_validation_trials_not_equal_fold_means(tmp_path):
    result, calls = fit_software_method(tmp_path, "metadata")
    assert result["selection"]["C"] == 0.1
    expected = (-100 * np.log(0.9) - np.log(0.1)) / 101
    assert np.isclose(result["selection"]["validation_log_loss"], expected)
    assert calls == 5
    # Equal weighting of the two fold means would incorrectly select C=1 here.
    assert (-np.log(0.8) - np.log(0.9)) < (-np.log(0.9) - np.log(0.1))


def test_pca_duplicate_effective_ranks_keep_all_nominal_losses_and_reuse_fits(tmp_path):
    result, calls = fit_software_method(tmp_path, "pca_plus_metadata")
    selection = result["selection"]
    assert selection["rank"] == 8 and selection["C"] == 0.1
    assert len(selection["cv_rows"]) == 16
    assert {row["rank"] for row in selection["cv_rows"]} == {8, 16, 32, 64}
    assert {row["effective_rank"] for row in selection["cv_rows"]} == {1}
    assert calls == 5  # Two C values per fold plus one final fit, not one fit per nominal rank.


@pytest.mark.parametrize("defect", ["C", "status", "coef_nan", "coef_shape", "intercept", "logits"])
def test_cache_rejects_inconsistent_saved_model_payload(tmp_path, defect):
    """A matching regime hash does not make a corrupted cache payload valid."""
    path = tmp_path / "fit.json"
    x = np.array([[0.0], [1.0], [2.0]])
    k, n = np.array([0, 1, 2]), np.array([2, 2, 2])
    driver.cached_fit(path, "software_fixture", x, x, k, n, 1.0, np.eye(1))
    saved = json.loads(path.read_text())
    if defect == "C":
        saved["C"] = 100.0
    elif defect == "status":
        saved["status"] = "unknown_status"
    elif defect == "coef_nan":
        saved["coef"] = [float("nan")]
    elif defect == "coef_shape":
        saved["coef"] = [0.0, 0.0]
    elif defect == "intercept":
        saved["intercept"] += 1.0
    elif defect == "logits":
        saved["logits"][0] += 1.0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        driver.cached_fit(path, "software_fixture", x, x, k, n, 1.0, np.eye(1))


def incomplete_fixture(tmp_path):
    """Construct three mocked fresh records in memory; never create future result files."""
    from scripts import context_risk_followup as collection

    loader = importlib.util.spec_from_file_location(
        "collection_review_fixture", Path(__file__).with_name("followup_review.py")
    )
    helpers = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helpers)
    cfg, _ = helpers.setup(tmp_path)
    _, log = helpers.native(tmp_path, cfg)
    samples = collection.load_samples(tmp_path / "manifests/fresh_A.jsonl")
    log.eval.metadata = helpers.metadata(tmp_path, phase="fresh")
    log.eval.config.epochs = 4
    rows = []
    for condition in ("original", "conflicting", "oneoff"):
        sample = next(s for s in samples if s.metadata["condition"] == condition)
        row = copy.deepcopy(log.samples[0])
        row.id, row.epoch, row.input = sample.id, 1, sample.input
        row.messages = [NS(text=sample.input)]
        row.metadata = copy.deepcopy(sample.metadata)
        history = []
        for attempt in range(1, 11 if condition == "conflicting" else 2):
            history.append(
                {
                    "attempt": attempt,
                    "request_seed": int.from_bytes(
                        hashlib.sha256(f"38295:{row.id}:1:{attempt}".encode()).digest()[:4], "big"
                    ),
                    "success": condition == "original",
                    "category": "passed"
                    if condition == "original"
                    else (
                        "generation_incomplete" if condition == "oneoff" else "assertion_failure"
                    ),
                }
            )
        row.metadata["agentic_results"] = {
            "harness_fingerprint": collection.harness_fingerprint(),
            "max_attempts": 10,
            "attempt_history": history,
            "flag_for_human_intervention": False,
            "censored": condition == "oneoff",
        }
        row.scores = {
            "successful_submission": helpers.Score(
                "C" if condition == "original" else "N" if condition == "oneoff" else "I"
            )
        }
        rows.append(row)
    log.samples = rows
    report = collection.summarize_logs([log], epochs=4)
    report.update(
        phase="fresh",
        arm="A",
        is_pilot=False,
        passed=False,
        requested_rollouts=996,
        coverage_complete=False,
        manifest_sha256=collection.sha256(tmp_path / "manifests/fresh_A.jsonl"),
        sources_sha256=collection.source_hashes(),
        native_logs_sha256={log.location: collection.sha256(Path(log.location))},
    )
    virtual = {
        tmp_path / "development_A/run_result.json": "software development A",
        tmp_path / "development_B/run_result.json": "software development B",
    }
    selection = {
        "passed": True,
        "selected_arm": "A",
        "selected_unix": 100,
        "freeze_sha256": collection.sha256(tmp_path / "manifests/freeze.json"),
        "arms": {
            arm: {
                "source_sha256": hashlib.sha256(
                    virtual[tmp_path / f"development_{arm}/run_result.json"].encode()
                ).hexdigest()
            }
            for arm in ("A", "B")
        },
    }
    virtual[tmp_path / "selection.json"] = json.dumps(selection)
    virtual[tmp_path / "fresh_A/run_result.json"] = json.dumps(report)
    return log, report, virtual


def run_incomplete_fixture(root, log, virtual):
    """Patch only filesystem/native reads for the intentionally nonexistent fresh run."""
    original_read = Path.read_text
    original_sha = driver.sha256

    def read(path, *args, **kwargs):
        return virtual[path] if path in virtual else original_read(path, *args, **kwargs)

    def sha(path):
        return (
            hashlib.sha256(virtual[path].encode()).hexdigest()
            if path in virtual
            else original_sha(path)
        )

    with (
        patch.object(Path, "read_text", read),
        patch.object(driver, "sha256", side_effect=sha),
        patch("inspect_ai.log.read_eval_log", return_value=log),
        patch.object(driver, "fit_method", side_effect=AssertionError("Unexpected fit")),
    ):
        return driver.incomplete_census(root)


def test_censored_and_missing_fresh_rows_are_explicit_inconclusive_without_fits(tmp_path):
    log, _, virtual = incomplete_fixture(tmp_path)
    result = run_incomplete_fixture(tmp_path, log, virtual)
    assert result["status"] == "primary_inference_inconclusive_incomplete_or_censored"
    assert result["realized_trajectories"] == 3
    assert result["missing_trajectories"] == 993 and result["censored_trajectories"] == 1
    assert result["models"] == {}
    assert result["by_condition"]["original"]["complete_case_rate"] == 1
    assert result["by_condition"]["conflicting"]["complete_case_rate"] == 0
    assert result["by_condition"]["oneoff"]["complete_case_rate"] is None
    assert not (tmp_path / "fresh_A").exists()


@pytest.mark.parametrize(
    "defect",
    ["development_hash", "manifest", "epochs", "requested", "native_sources", "coverage", "passed"],
)
def test_incomplete_status_does_not_hide_incompatible_evidence(tmp_path, defect):
    log, report, virtual = incomplete_fixture(tmp_path)
    if defect == "development_hash":
        virtual[tmp_path / "development_B/run_result.json"] = "changed development data"
    elif defect == "manifest":
        report["manifest_sha256"] = "wrong"
    elif defect == "epochs":
        report["epochs"] = 3
    elif defect == "requested":
        report["requested_rollouts"] = 1
    elif defect == "native_sources":
        log.eval.metadata["sources_sha256"] = {"wrong": "source"}
    elif defect == "coverage":
        report["coverage_complete"] = True
    elif defect == "passed":
        report["passed"] = True
    virtual[tmp_path / "fresh_A/run_result.json"] = json.dumps(report)
    with pytest.raises(ValueError):
        run_incomplete_fixture(tmp_path, log, virtual)
