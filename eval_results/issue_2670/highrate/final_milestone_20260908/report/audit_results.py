"""Independently audit saved highrate results; no fitting, native reads or inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

PROJECT = Path("/home/thomasjiralerspong/.codex/worktrees/context-risk-recovery-20260906")
RUN = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate"
)
REGIMES = (
    "primary",
    "competence_sensitivity",
    "screen_augmented_primary",
    "screen_augmented_competence_sensitivity",
)
COMPARISONS = {
    "raw_over_text": ("text_plus_metadata", "raw_plus_metadata"),
    "mapped_over_raw": ("raw_plus_metadata", "mapped_plus_metadata"),
}
SPEC_PATH = PROJECT / "eval_results/context_risk_highrate_design/analysis_spec_v10_optimizer.json"
SPEC_SHA = "154d6038216ece9af36ee7092a575ab5244a14afd34a746f71d2d1566a32fb71"
MAX_ITER = 20000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, label):
    if expected is None:
        assert actual is None, label
    else:
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-11), (label, actual, expected)


def binary_rows(logits, positive, trials):
    """Explicitly expand each completed Bernoulli draw, independently of producer weights."""
    logits, positive, trials = map(np.asarray, (logits, positive, trials))
    assert len(logits) == len(positive) == len(trials)
    assert np.isfinite(logits).all()
    assert all(type(x) in (int, np.int64, np.int32) for x in positive)
    assert np.all((positive >= 0) & (positive <= trials) & (trials > 0))
    labels = np.concatenate(
        [
            np.r_[np.ones(k, dtype=int), np.zeros(n - k, dtype=int)]
            for k, n in zip(positive, trials, strict=True)
        ]
    )
    return labels, np.repeat(logits, trials)


def metrics(logits, positive, trials):
    y, z = binary_rows(logits, positive, trials)
    p = expit(z)
    both = bool(y.min() != y.max())
    value = {
        "log_loss": float(np.where(y == 1, np.logaddexp(0, -z), np.logaddexp(0, z)).mean()),
        "brier": float(np.square(y - p).mean()),
        "observed_prevalence": float(y.mean()),
        "predicted_prevalence": float(p.mean()),
        "n_trajectories": len(y),
        "n_positive": int(y.sum()),
        "n_contexts": len(trials),
        "auroc": float(roc_auc_score(y, z)) if both else None,
        "average_precision": float(average_precision_score(y, z)) if both else None,
    }
    bins = np.minimum((p * 5).astype(int), 4)
    value["calibration"] = [
        {
            "lower": i / 5,
            "upper": (i + 1) / 5,
            "n": int((bins == i).sum()),
            "observed": float(y[bins == i].mean()) if np.any(bins == i) else None,
            "predicted": float(p[bins == i].mean()) if np.any(bins == i) else None,
        }
        for i in range(5)
    ]
    return value


def loss_by_context(logits, positive, trials):
    totals = []
    for z, k, n in zip(logits, positive, trials, strict=True):
        y, expanded = binary_rows([z], [int(k)], [int(n)])
        totals.append(np.where(y == 1, np.logaddexp(0, -expanded), np.logaddexp(0, expanded)).sum())
    return np.asarray(totals)


def bootstrap(first, second, positive, trials, tasks, repeats, seed):
    """Resample whole test tasks, retaining both conditions and their completed draws."""
    differences = loss_by_context(first, positive, trials) - loss_by_context(
        second, positive, trials
    )
    unique = sorted(set(tasks))
    numerators = np.array(
        [sum(d for d, t in zip(differences, tasks, strict=True) if t == task) for task in unique]
    )
    denominators = np.array(
        [sum(n for n, t in zip(trials, tasks, strict=True) if t == task) for task in unique]
    )
    sampled = np.random.default_rng(seed).integers(len(unique), size=(repeats, len(unique)))
    values = numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "improvement": float(differences.sum() / sum(trials)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_tasks": len(unique),
        "replicates": repeats,
        "seed": seed,
        "positive_interval": bool(lo > 0),
    }, [
        {
            "task_id": t,
            "loss_difference_sum": float(d),
            "completed_trials": int(n),
            "improvement": float(d / n),
        }
        for t, d, n in zip(unique, numerators, denominators, strict=True)
    ]


def support(rows):
    return {
        "n_tasks": len({r["task_id"] for r in rows}),
        "n_contexts": len(rows),
        "n_trajectories": sum(r["trials"] for r in rows),
        "positive": sum(r["positive"] for r in rows),
        "negative": sum(r["trials"] - r["positive"] for r in rows),
        "positive_tasks": len({r["task_id"] for r in rows if r["positive"] > 0}),
        "negative_tasks": len({r["task_id"] for r in rows if r["positive"] < r["trials"]}),
    }


def inspect_fit(path, fingerprint, positive, trials, expected_n, diagnostics):
    fit = read(path)
    assert fit["fingerprint"] == fingerprint, str(path)
    assert type(fit["max_iter"]) is int and fit["max_iter"] == MAX_ITER, str(path)
    assert len(fit["logits"]) == expected_n and np.isfinite(fit["logits"]).all(), str(path)
    assert type(fit["iterations"]) is int and 0 <= fit["iterations"] <= MAX_ITER
    assert np.isfinite(fit["elapsed_seconds"]) and fit["elapsed_seconds"] >= 0
    assert np.isfinite(fit["intercept"])
    if sum(positive) in (0, sum(trials)):
        p = (sum(positive) + 0.5) / (sum(trials) + 1)
        assert fit["status"] == "single_class_training_prevalence" and fit["coef"] is None
        np.testing.assert_allclose(fit["logits"], np.log(p) - np.log1p(-p), rtol=1e-12, atol=1e-12)
    else:
        assert fit["status"] == "fitted" and fit["coef"] is not None
        assert np.isfinite(fit["coef"]).all()
    diagnostics.append(
        {
            "path": str(path.relative_to(RUN)),
            "status": fit["status"],
            "iterations": fit["iterations"],
            "seconds": fit["elapsed_seconds"],
        }
    )
    return fit


def audit_cv(name, result, rows, train, test, spec, provenance, diagnostics):
    groups = np.array([r["task_id"] for r in rows])
    positive = np.array([r["positive"] for r in rows])
    trials = np.array([r["trials"] for r in rows])
    eligible_only = "competence_sensitivity" in name
    augmented = name.startswith("screen_augmented_")
    regime = digest(
        {
            "provenance": provenance,
            "eligible_only": eligible_only,
            "augmented": augmented,
            "train": train.tolist(),
            "test": test.tolist(),
            "spec": spec,
        }
    )
    folds = [
        (train[a], train[b]) for a, b in GroupKFold(n_splits=5).split(train, groups=groups[train])
    ]
    for left, right in folds:
        assert not set(groups[left]) & set(groups[right])
        assert not set(groups[left]) & set(groups[test])
        assert set(left) | set(right) == set(train)
    methods = [m for m in spec["readouts"] if m != "prevalence"]
    methods += [f"orientation_{s}" for s in spec["orientation_controls"]["seeds"]]
    if "pca_plus_metadata" in result["models"]:
        assert name == "primary" and result["comparisons"]["mapped_over_raw"]["benefit_supported"]
        methods += ["pca_plus_metadata"]
    expected = set(methods) | {"prevalence"} | ({"direct_screen_rate"} if augmented else set())
    assert set(result["models"]) == expected
    checked = {}
    for method in methods:
        model = result["models"][method]
        selection = read(RUN / "analysis" / name / f"{method}_selection.json")
        assert selection == model["selection"] and selection["regime"] == regime
        ranks = (
            spec["secondary_PCA_control"]["rank_candidates"]
            if method == "pca_plus_metadata"
            else [None]
        )
        losses = {(rank, c): 0.0 for rank in ranks for c in spec["regularization_C"]}
        lookup = {(r["fold"], r["rank"], r["C"]): r for r in selection["cv_rows"]}
        assert len(lookup) == len(selection["cv_rows"]) == 5 * len(losses)
        total_trials = 0
        for fold, (left, right) in enumerate(folds):
            total_trials += int(trials[right].sum())
            for rank in ranks:
                effective = min(rank, len(left) - 1, 5120) if rank else None
                for c in spec["regularization_C"]:
                    key = {
                        "regime": regime,
                        "method": method,
                        "fold": fold,
                        "fit_indices": left.tolist(),
                        "test_indices": right.tolist(),
                        "effective_rank": effective,
                        "C": c,
                        "max_iter": MAX_ITER,
                    }
                    path = (
                        RUN
                        / "analysis"
                        / name
                        / "fits"
                        / f"{method}_fold{fold}_rank{effective}_C{c:g}.json"
                    )
                    fit = inspect_fit(
                        path, digest(key), positive[left], trials[left], len(right), diagnostics
                    )
                    assert fit["C"] == c
                    loss = float(
                        loss_by_context(fit["logits"], positive[right], trials[right]).sum()
                    )
                    entry = lookup[(fold, rank, c)]
                    close(entry["loss_sum"], loss, f"{name}/{method}/fold{fold}")
                    assert (
                        entry["trials"] == int(trials[right].sum())
                        and entry["status"] == fit["status"]
                    )
                    assert entry["method"] == method
                    for field in ("identity_bias_max_error", "operator_max_error"):
                        assert np.isfinite(entry[field]) and entry[field] >= 0
                    losses[(rank, c)] += loss
        chosen = min(losses, key=lambda k: (losses[k], k[0] or 0, k[1]))
        assert (selection["rank"], selection["C"]) == chosen
        close(selection["validation_log_loss"], losses[chosen] / total_trials, "selected CV loss")
        key = {
            "regime": regime,
            "method": method,
            "stage": "final",
            "rank": chosen[0],
            "C": chosen[1],
            "fit_indices": train.tolist(),
            "test_indices": test.tolist(),
            "max_iter": MAX_ITER,
        }
        final = inspect_fit(
            RUN / "analysis" / name / "fits" / f"{method}_final.json",
            digest(key),
            positive[train],
            trials[train],
            len(test),
            diagnostics,
        )
        assert final["C"] == chosen[1] and final["logits"] == model["logits"]
        checks = model["feature_checks"]
        assert checks["method"] == method
        for field in ("identity_bias_max_error", "operator_max_error"):
            assert np.isfinite(checks[field]) and checks[field] >= 0
        if method == "text_plus_metadata":
            assert 0 < checks["vocabulary_size"] <= spec["text"]["max_features"]
        if method == "mapped_plus_metadata" and final["coef"] is not None:
            assert np.isfinite(checks["composed_logit_max_error"])
        checked[method] = {
            "C": chosen[1],
            "rank": chosen[0],
            "cv_log_loss": losses[chosen] / total_trials,
            "feature_checks": checks,
        }
    return checked


def run():
    out = RUN / "report"
    analysis = RUN / "analysis"
    initial = {str(p): sha(p) for p in analysis.rglob("*") if p.is_file()}
    result = read(analysis / "result.json")
    assert result["verification_passed"] is True and set(result["analyses"]) == set(REGIMES)
    launch, census = read(analysis / "analysis_launch.json"), read(analysis / "input_census.json")
    provenance = result["provenance"]
    assert provenance == launch["provenance"] == census["provenance"]
    reviewer = RUN / "setup/analysis_optimizer_code_review.json"
    code_review = read(reviewer)
    assert (
        code_review["verdict"] == "PASS"
        and code_review["sources_sha256"] == provenance["analysis_sources_sha256"]
    )
    assert (
        sha(reviewer) == provenance["independent_review_sha256"] and launch["review"] == code_review
    )
    controls = {str(PROJECT / k): v for k, v in provenance["analysis_sources_sha256"].items()}
    controls.update(
        {
            str(Path(k) if Path(k).is_absolute() else PROJECT / k): v
            for k, v in provenance["input_files_sha256"].items()
        }
    )
    controls[str(reviewer)] = sha(reviewer)
    assert all(sha(p) == value for p, value in controls.items())
    assert sha(SPEC_PATH) == SPEC_SHA == provenance["spec_sha256"]
    spec = read(SPEC_PATH)
    assert spec["logistic_max_iter"] == MAX_ITER
    original_spec_path = PROJECT / "eval_results/context_risk_highrate_design/analysis_spec.json"
    assert sha(original_spec_path) == spec["optimization_amendment"]["original_spec_sha256"]
    controls[str(SPEC_PATH)] = SPEC_SHA
    controls[str(original_spec_path)] = sha(original_spec_path)
    configured_spec = Path(launch["config"]["spec"])
    configured_spec = (
        configured_spec if configured_spec.is_absolute() else PROJECT / configured_spec
    )
    assert configured_spec.resolve() == SPEC_PATH.resolve()
    assert Path(launch["config"]["review"]).resolve() == reviewer.resolve()
    original_spec = read(original_spec_path)
    unchanged = set(original_spec) - {"freeze_timing", "logistic_max_iter"}
    assert all(spec[k] == original_spec[k] for k in unchanged)
    rows = census["contexts"]
    assert len(rows) == 90 and len({r["exact_context_sha256"] for r in rows}) == 90
    assert digest(rows) == launch["rows_sha256"]
    assert rows == sorted(rows, key=lambda r: (r["task_id"], r["condition"]))
    selection = read(RUN / "selection.json")
    assert len(selection["task_roles"]) == 30
    for r in rows:
        assert r["public_test_role"] == selection["task_roles"][r["task_id"]]
        assert r["planned"] == r["trials"] + r["censored"] == 4
        assert r["screen_success"] + r["screen_failure"] + r["screen_censored"] == 2
        assert 0 <= r["positive"] <= r["trials"]
    fresh = provenance["fresh_counts"]
    assert fresh["planned"] == fresh["realized"] == 360 and fresh["missing"] == 0
    assert sum(r["positive"] for r in rows) == fresh["success"]
    assert sum(r["trials"] - r["positive"] for r in rows) == fresh["failure"]
    assert sum(r["censored"] for r in rows) == fresh["censored"]
    static = provenance["input_validity"]["fresh"]["static_audit"]
    invalid = static["structurally_invalid_test_sha256"]
    excluded = [r for r in rows if f"{r['task_id']}:{r['condition']}" in invalid]
    assert (
        4 * len(excluded)
        == provenance["input_validity"]["fresh"]["not_assessable_planned_trajectories"]
    )
    fit_rows = [r for r in rows if f"{r['task_id']}:{r['condition']}" not in invalid]
    eligible = sorted(
        r["task_id"] for r in fit_rows if r["condition"] == "original" and r["positive"] > 0
    )
    ineligible = sorted(
        r["task_id"]
        for r in fit_rows
        if r["condition"] == "original" and r["positive"] == r["censored"] == 0
    )
    unknown = sorted(
        r["task_id"]
        for r in fit_rows
        if r["condition"] == "original" and r["positive"] == 0 and r["censored"] > 0
    )
    eligibility = {
        "eligible_tasks": eligible,
        "ineligible_tasks": ineligible,
        "unknown_tasks": unknown,
    }
    assert eligibility == census["eligibility"]
    regimes, diagnostics, comparisons, model_data, task_data = {}, [], [], [], []
    for name in REGIMES:
        current = read(analysis / name / "result.json")
        assert current == result["analyses"][name] and current["provenance"] == provenance
        assert current["eligibility"] == eligibility
        augmented, competent = (
            name.startswith("screen_augmented_"),
            "competence_sensitivity" in name,
        )
        included = [
            i
            for i, r in enumerate(fit_rows)
            if r["condition"] != "original" and (not competent or r["task_id"] in eligible)
        ]
        keep = [i for i in included if fit_rows[i]["trials"] > 0]
        train = np.array(
            [i for i in keep if fit_rows[i]["public_test_role"] == "probe_training"], dtype=int
        )
        test = np.array(
            [i for i in keep if fit_rows[i]["public_test_role"] == "final_test"], dtype=int
        )
        left, right = support([fit_rows[i] for i in train]), support([fit_rows[i] for i in test])
        assert left == current["training_support"] and right == current["test_support"]
        assert not {fit_rows[i]["task_id"] for i in train} & {fit_rows[i]["task_id"] for i in test}
        assert len(train) + len(test) == len(keep)
        assert current["excluded_zero_complete_case_contexts"] == [
            fit_rows[i]["exact_context_sha256"] for i in included if fit_rows[i]["trials"] == 0
        ]
        claim = spec["claim_gate"]
        fit_gate = left["n_tasks"] >= 5 and left["positive"] > 0 and left["negative"] > 0
        claim_gate = (
            left["positive_tasks"] >= claim["minimum_positive_training_tasks"]
            and left["negative_tasks"] >= claim["minimum_negative_training_tasks"]
            and right["positive_tasks"] >= claim["minimum_positive_test_tasks"]
            and right["negative_tasks"] >= claim["minimum_negative_test_tasks"]
            and fresh["censored"] <= claim["maximum_censored_fresh_trajectories"]
            and not excluded
        )
        assert (
            current["fit_gate_passed"] == fit_gate and current["claim_support_passed"] == claim_gate
        )
        assert current["total_fresh_censored"] == fresh["censored"]
        assert current["structurally_not_assessable_fresh_trajectories"] == 4 * len(excluded)
        if not len(test):
            assert current["status"] == "no_completed_test_contexts"
            regimes[name] = {"status": current["status"], "support": right}
            continue
        predictions = current["test_predictions"]
        assert len(predictions) == len(test)
        for prediction, i in zip(predictions, test, strict=True):
            assert all(
                prediction[k] == fit_rows[i][k]
                for k in (
                    "task_id",
                    "condition",
                    "exact_context_sha256",
                    "positive",
                    "trials",
                    "censored",
                )
            )
            assert set(prediction["logits"]) == set(current["models"])
        positive = [p["positive"] for p in predictions]
        trials = [p["trials"] for p in predictions]
        tasks = [p["task_id"] for p in predictions]
        for method, value in current["models"].items():
            logits = [p["logits"][method] for p in predictions]
            assert logits == value["logits"]
            recomputed = metrics(logits, positive, trials)
            for key, metric in recomputed.items():
                if key != "calibration":
                    close(value["metrics"][key], metric, f"{name}/{method}/{key}")
            for a, b in zip(
                value["metrics"]["calibration"], recomputed["calibration"], strict=True
            ):
                for key in a:
                    close(a[key], b[key], f"{name}/{method}/calibration/{key}")
            model_data.append({"regime": name, "method": method, **recomputed})
        p = (left["positive"] + 0.5) / (left["n_trajectories"] + 1)
        np.testing.assert_allclose(
            current["models"]["prevalence"]["logits"],
            np.log(p) - np.log1p(-p),
            rtol=1e-12,
            atol=1e-12,
        )
        if augmented:
            expected = [
                (fit_rows[i]["screen_success"] + 0.5)
                / (fit_rows[i]["screen_success"] + fit_rows[i]["screen_failure"] + 1)
                for i in test
            ]
            np.testing.assert_allclose(
                expit(current["models"]["direct_screen_rate"]["logits"]),
                expected,
                rtol=1e-12,
                atol=1e-12,
            )
        selected = (
            audit_cv(name, current, fit_rows, train, test, spec, provenance, diagnostics)
            if fit_gate
            else {}
        )
        assert current["status"] == (
            "complete" if fit_gate else "prevalence_only_insufficient_training_support"
        )
        if fit_gate:
            assert set(current["comparisons"]) == set(COMPARISONS)
            for label, (first, second) in COMPARISONS.items():
                if len(set(tasks)) < 2:
                    assert current["comparisons"][label]["benefit_supported"] is False
                    continue
                interval, per_task = bootstrap(
                    current["models"][first]["logits"],
                    current["models"][second]["logits"],
                    positive,
                    trials,
                    tasks,
                    spec["uncertainty"]["replicates"],
                    spec["uncertainty"]["seed"],
                )
                for key, val in interval.items():
                    close(current["comparisons"][label][key], val, f"{name}/{label}/{key}")
                assert current["comparisons"][label]["benefit_supported"] == bool(
                    claim_gate and interval["positive_interval"]
                )
                comparisons.append(
                    {
                        "regime": name,
                        "comparison": label,
                        "first": first,
                        "second": second,
                        **interval,
                        "benefit_supported": current["comparisons"][label]["benefit_supported"],
                    }
                )
                task_data.extend({"regime": name, "comparison": label, **p} for p in per_task)
        else:
            assert current["comparisons"] == {}
        regimes[name] = {
            "status": current["status"],
            "training_support": left,
            "test_support": right,
            "fit_gate_passed": fit_gate,
            "claim_support_passed": claim_gate,
            "selected": selected,
        }
    assert initial == {str(p): sha(p) for p in analysis.rglob("*") if p.is_file()}, (
        "Analysis mutated during independent audit"
    )
    assert all(sha(p) == value for p, value in controls.items()), (
        "Source/input changed during audit"
    )
    report = {
        "verdict": "PASS",
        "analysis_result_sha256": initial[str(analysis / "result.json")],
        "reviewer": "/root/verify_vllm_runtime, independent of the analysis producer",
        "audited_utc": datetime.now(UTC).isoformat(),
        "verification_passed": result["verification_passed"],
        "analysis_files_sha256": initial,
        "control_files_sha256": controls,
        "audit_source_sha256": sha(__file__),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "matplotlib")
        },
        "fresh_counts": fresh,
        "structurally_not_assessable": len(excluded) * 4,
        "eligibility": eligibility,
        "regimes": regimes,
        "models": model_data,
        "comparisons": comparisons,
        "per_task_comparisons": task_data,
        "fit_diagnostics": diagnostics,
        "fitted_cache_count": len({x["path"] for x in diagnostics}),
        "max_iterations": max((x["iterations"] for x in diagnostics), default=0),
        "scope": (
            "Independent saved-result algebra, expanded-binary metrics, full task bootstrap, "
            "saved-fold CV selection, source/input/role/count and feature-check provenance. "
            "No refit or repeated native/capture validation."
        ),
        "limits": [
            (
                "Intervals are marginal, conditional on fitted predictions "
                "and omit fitting uncertainty."
            ),
            "Any fresh censor or structurally invalid context blocks unconditional benefit claims.",
            (
                "Affine feature equivalence and control construction are source-bound recorded "
                "runtime checks; numerical feature matrices were not regenerated."
            ),
            "Owned terminal process evidence is a separate parent completion gate.",
        ],
    }
    (out / "independent_result_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "comparison_data.json").write_text(
        json.dumps(
            {
                "source_result_sha256": sha(analysis / "result.json"),
                "models": model_data,
                "comparisons": comparisons,
                "per_task_comparisons": task_data,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "regimes": list(regimes),
                "fitted_cache_count": report["fitted_cache_count"],
                "max_iterations": report["max_iterations"],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
