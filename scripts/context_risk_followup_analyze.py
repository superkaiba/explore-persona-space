"""Fit frozen task-disjoint risk readouts with per-fit resumable evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from scripts.context_risk_analyze import load_impossible_activations  # noqa: E402
from scripts.context_risk_followup_features import FeatureBank  # noqa: E402
from scripts.context_risk_followup_probe_core import (  # noqa: E402
    fit_logistic,
    l2_basis,
    loss_terms,
    metric_report,
    paired_task_interval,
    save_json,
)

SPEC_SHA = "a47ce4131959f8a7da9a75c367355713927e39e91692fbf017a1cad737d155d0"
ANALYSIS_SOURCES = (
    "scripts/context_risk_followup_analyze.py",
    "scripts/context_risk_followup_features.py",
    "scripts/context_risk_followup_probe_core.py",
    "scripts/context_risk_followup_audit.py",
    "scripts/context_risk_followup_capture.py",
    "scripts/context_risk_analyze.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def incomplete_census(root: Path) -> dict | None:
    """Persist a verified inconclusive result when the fresh cohort is censored or partial."""
    # PROD_IMPORT_LINT_EXEMPT: Run with uv --with inspect-ai==0.3.261, isolated from the shared environment.
    from inspect_ai.log import read_eval_log
    from omegaconf import OmegaConf
    from scripts.context_risk_followup import load_samples, source_hashes, validate_native
    from scripts.context_risk_impossiblebench_inspect import summarize_logs

    selection = json.loads((root / "selection.json").read_text())
    if not selection["passed"]:
        raise ValueError("Prediction requires a viable frozen recipe")
    for development_arm in ("A", "B"):
        if selection["arms"][development_arm]["source_sha256"] != sha256(
            root / f"development_{development_arm}/run_result.json"
        ):
            raise ValueError("Development results changed after recipe selection")
    arm = selection["selected_arm"]
    report_path = root / f"fresh_{arm}/run_result.json"
    saved = json.loads(report_path.read_text())
    if saved["passed"] and saved["coverage_complete"] and saved["technical_errors"] == 0:
        return None
    manifest = root / "manifests" / f"fresh_{arm}.jsonl"
    freeze_path = root / "manifests/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        selection["freeze_sha256"] != sha256(freeze_path)
        or freeze["manifests"][manifest.name]["sha256"] != sha256(manifest)
        or saved["sources_sha256"] != source_hashes()
        or saved["phase"] != "fresh"
        or saved["arm"] != arm
        or saved["is_pilot"]
        or saved["manifest_sha256"] != sha256(manifest)
        or saved["epochs"] != 4
        or saved["requested_rollouts"] != 996
    ):
        raise ValueError("Incomplete cohort has incompatible frozen provenance")
    samples = load_samples(manifest)
    logs = []
    for location, expected_sha in saved["native_logs_sha256"].items():
        if sha256(Path(location)) != expected_sha:
            raise ValueError("Incomplete native log changed after reporting")
        log = read_eval_log(location)
        cfg = OmegaConf.create(
            {
                "model": "openai-api/local/Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "base_url": log.eval.model_base_url,
            }
        )
        validate_native(log, samples, log.eval.metadata, cfg)
        if (
            log.eval.metadata["phase"] != "fresh"
            or log.eval.metadata["arm"] != arm
            or log.eval.metadata["epochs"] != 4
            or log.eval.metadata["freeze_sha256"] != sha256(freeze_path)
            or log.eval.metadata["manifest_sha256"] != sha256(manifest)
            or log.eval.metadata["sources_sha256"] != source_hashes()
        ):
            raise ValueError("Incomplete native log metadata differs")
        logs.append(log)
    actual = summarize_logs(logs, epochs=4)
    for key in (
        "by_condition",
        "contexts",
        "realized_rollouts",
        "technical_errors",
        "censored_reasons",
    ):
        if actual[key] != saved[key]:
            raise ValueError("Incomplete native census differs from saved report")
    expected = {(s.id, epoch) for s in samples for epoch in (1, 2, 3, 4)}
    realized = {(s.id, s.epoch) for log in logs for s in log.samples or []}
    if not realized.issubset(expected) or len(expected) != 996:
        raise ValueError("Unplanned incomplete-cohort rows")
    complete = realized == expected
    passed = bool(actual["passed"] and complete and actual["technical_errors"] == 0)
    if saved["coverage_complete"] != complete or saved["passed"] != passed:
        raise ValueError("Incomplete cohort status flags disagree with native evidence")
    return {
        "status": "primary_inference_inconclusive_incomplete_or_censored",
        "requested_trajectories": 996,
        "realized_trajectories": len(realized),
        "missing_trajectories": len(expected - realized),
        "censored_trajectories": actual["technical_errors"],
        "censored_reasons": actual["censored_reasons"],
        "models": {},
        "complete_case_scope": "Observed success rates conditional on completed trajectories; censored and missing outcomes are not failures",
        "by_condition": {
            key: {
                **value,
                "complete_case_rate": None
                if value["n"] == value["errors"]
                else value["passed"] / (value["n"] - value["errors"]),
            }
            for key, value in actual["by_condition"].items()
        },
        "contexts": actual["contexts"],
        "run_result_sha256": sha256(report_path),
        "native_logs_sha256": saved["native_logs_sha256"],
        "manifest_sha256": sha256(manifest),
    }


def load_inputs(root: Path, captures: Path, map_path: Path, spec_path: Path):
    """Join exact frozen inputs, audited complete outcomes, and initial token captures."""
    from scripts.context_risk_followup_audit import audit
    from scripts.context_risk_followup_capture import validate_binding

    if sha256(spec_path) != SPEC_SHA:
        raise ValueError("Analysis specification differs from the pre-outcome freeze")
    spec = json.loads(spec_path.read_text())
    if sha256(map_path) != spec["map_sha256"]:
        raise ValueError("Map differs from the frozen independent mapping")
    selection = json.loads((root / "selection.json").read_text())
    freeze = json.loads((root / "manifests/freeze.json").read_text())
    if not selection["passed"] or selection["freeze_sha256"] != sha256(
        root / "manifests/freeze.json"
    ):
        raise ValueError("Missing or incompatible recipe selection")
    arm = selection["selected_arm"]
    for development_arm in ("A", "B"):
        if selection["arms"][development_arm]["source_sha256"] != sha256(
            root / f"development_{development_arm}/run_result.json"
        ):
            raise ValueError("Development results changed after recipe selection")
    stage = root / f"fresh_{arm}"
    manifest = root / "manifests" / f"fresh_{arm}.jsonl"
    if sha256(manifest) != freeze["manifests"][manifest.name]["sha256"]:
        raise ValueError("Fresh manifest differs from its freeze")
    report = audit(stage / "run_result.json", manifest, stage / "audit")
    if report["first_model_request_unix"] <= selection["selected_unix"]:
        raise ValueError("Observed fresh model request predates recipe selection")
    launch_paths = sorted(stage.glob("launch_config_*.json"))
    if not launch_paths:
        raise ValueError("Missing pre-generation launch evidence")
    for launch_path in launch_paths:
        launch = json.loads(launch_path.read_text())
        if launch["started_unix"] <= selection["selected_unix"]:
            raise ValueError("Fresh generation began before recipe selection")
        if (
            launch["metadata"]["phase"] != "fresh"
            or launch["metadata"]["arm"] != arm
            or launch["metadata"]["manifest_sha256"] != sha256(manifest)
        ):
            raise ValueError("Fresh launch metadata differs from selected manifest")
    if report["realized_rollouts"] != 996 or report["is_pilot"] or report["phase"] != "fresh":
        raise ValueError("Prediction requires the full prespecified fresh cohort")
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda r: (r["task_id"], r["condition"]))
    with (stage / "audit/audited_rollouts.jsonl").open() as handle:
        observed = [json.loads(line) for line in handle if line.strip()]
    by_context = {}
    for outcome in observed:
        by_context.setdefault(outcome["exact_context_sha256"], []).append(outcome)
    expected_hashes = {r["exact_context_sha256"] for r in rows}
    if len(expected_hashes) != 249 or set(by_context) != expected_hashes:
        raise ValueError("Behavioral exact-context roster differs")
    validate_binding(captures, manifest, stage / "prefix_tokens.json", root / "selection.json")
    activation, capture_rows = load_impossible_activations(captures, selected_layer=spec["layer"])
    capture_report = json.loads((captures / "run_result.json").read_text())
    if (
        not capture_report["passed"]
        or capture_report["model_revision"] != spec["model_revision"]
        or capture_report["model_id"] != "Qwen/Qwen3.8-27B"
        or capture_report["n_contexts"] != 249
        or capture_report["prefixes_truncated"] != 0
        or capture_report["capture_layers"] != [44]
    ):
        raise ValueError("Capture model, layer, coverage or truncation differs")
    if set(activation) != expected_hashes:
        raise ValueError("Activation exact-context roster differs")
    done_paths = sorted(captures.glob("chunk_*.done.json"))
    if [json.loads(p.read_text()) for p in done_paths] != capture_report["chunks"]:
        raise ValueError("Capture run/chunk reports differ")
    if any(c["fingerprint"] != capture_report["fingerprint"] for c in capture_report["chunks"]):
        raise ValueError("Capture configuration changes across chunks")
    prefix_record = json.loads((stage / "prefix_tokens.json").read_text())
    prefixes = {r["exact_context_sha256"]: r for r in prefix_record["contexts"]}
    if (
        not prefix_record["passed"]
        or prefix_record["n_contexts"] != 249
        or len(prefix_record["contexts"]) != 249
        or set(prefixes) != expected_hashes
    ):
        raise ValueError("Generation prefix evidence is incomplete")
    for row in rows:
        key = row["exact_context_sha256"]
        outcomes = by_context[key]
        if len(outcomes) != 4 or {o["epoch"] for o in outcomes} != {1, 2, 3, 4}:
            raise ValueError("Fresh context lacks four unique trajectories")
        row["positive"] = sum(o["score"] == "C" for o in outcomes)
        row["trials"] = 4
        prefix = prefixes[key]
        if (
            len(prefix["token_ids"]) != prefix["n_prefix_tokens"]
            or digest(prefix["token_ids"]) != prefix["prefix_token_ids_sha256"]
        ):
            raise ValueError("Saved generation prefix token hash differs")
        for field in ("n_prefix_tokens", "prefix_token_ids_sha256"):
            if capture_rows[key][field] != prefix[field]:
                raise ValueError(f"Capture/generation prefix differs: {key}:{field}")
        if any(
            o["requests"][0]["usage"]["input_tokens"] != prefix["n_prefix_tokens"] for o in outcomes
        ):
            raise ValueError("First model request token count differs from saved initial prefix")
        for field in ("task_id", "condition", "public_test_role"):
            if capture_rows[key][field] != row[field]:
                raise ValueError(f"Capture/manifest metadata differs: {key}:{field}")
        if freeze["task_roles"][row["task_id"]] != row["public_test_role"]:
            raise ValueError("Task partition differs from the outcome-independent freeze")
    raw = np.stack([activation[r["exact_context_sha256"]] for r in rows])
    if raw.shape != (249, 5120):
        raise ValueError("Raw feature shape differs from frozen model")
    with np.load(map_path) as arrays:
        map_arrays = {key: arrays[key].copy() for key in ("weight", "x_mean", "x_scale", "y_mean")}
    provenance = {
        "spec_sha256": sha256(spec_path),
        "map_sha256": sha256(map_path),
        "manifest_sha256": sha256(manifest),
        "selection_sha256": sha256(root / "selection.json"),
        "run_result_sha256": sha256(stage / "run_result.json"),
        "native_audit_sha256": sha256(stage / "audit/native_audit.json"),
        "prefix_tokens_sha256": sha256(stage / "prefix_tokens.json"),
        "launch_files_sha256": {p.name: sha256(p) for p in launch_paths},
        "capture_files_sha256": {
            str(p.relative_to(captures)): sha256(p)
            for p in sorted(captures.iterdir())
            if p.is_file()
        },
    }
    return rows, raw, map_arrays, spec, provenance


def class_support(rows: list[dict], indices: np.ndarray) -> dict:
    selected = [rows[int(i)] for i in indices]
    return {
        "n_tasks": len({r["task_id"] for r in selected}),
        "n_contexts": len(selected),
        "n_trajectories": sum(r["trials"] for r in selected),
        "positive": sum(r["positive"] for r in selected),
        "negative": sum(r["trials"] - r["positive"] for r in selected),
        "positive_tasks": len({r["task_id"] for r in selected if r["positive"] > 0}),
        "negative_tasks": len({r["task_id"] for r in selected if r["positive"] < r["trials"]}),
    }


def cached_fit(
    path: Path,
    fingerprint: str,
    left: np.ndarray,
    right: np.ndarray,
    positive: np.ndarray,
    trials: np.ndarray,
    c_value: float,
    basis: np.ndarray,
) -> dict:
    """Resume only an exact input/spec/source regime, retaining each bounded fitting unit."""
    if path.exists():
        value = json.loads(path.read_text())
        if value["fingerprint"] != fingerprint:
            raise ValueError(f"Stale fit checkpoint: {path}")
        if value["C"] != c_value or not np.isfinite(value["intercept"]):
            raise ValueError("Cached fit regularization/intercept differs")
        if (
            not isinstance(value["iterations"], int)
            or not 0 <= value["iterations"] <= 5000
            or not np.isfinite(value["elapsed_seconds"])
            or value["elapsed_seconds"] < 0
        ):
            raise ValueError("Invalid cached fitting diagnostics")
        value["logits"] = np.asarray(value["logits"], dtype=np.float64)
        value["coef"] = (
            None if value["coef"] is None else np.asarray(value["coef"], dtype=np.float64)
        )
        if value["logits"].shape != (len(right),) or not np.isfinite(value["logits"]).all():
            raise ValueError("Invalid cached predictions")
        single_class = positive.sum() in {0, trials.sum()}
        if single_class:
            p = float((positive.sum() + 0.5) / (trials.sum() + 1))
            expected_logit = np.log(p) - np.log1p(-p)
            if (
                value["status"] != "single_class_training_prevalence"
                or value["coef"] is not None
                or value["intercept"] != expected_logit
            ):
                raise ValueError("Invalid cached single-class head")
            reconstructed = np.full(len(right), expected_logit)
        else:
            coefficient = value["coef"]
            if (
                value["status"] != "fitted"
                or coefficient is None
                or coefficient.shape != (basis.shape[0],)
                or not np.isfinite(coefficient).all()
            ):
                raise ValueError("Invalid cached linear coefficients")
            compressed_coefficient = basis.T @ coefficient
            if not np.allclose(basis @ compressed_coefficient, coefficient, rtol=1e-8, atol=1e-8):
                raise ValueError("Cached coefficient left the fitted L2 subspace")
            reconstructed = right @ compressed_coefficient + value["intercept"]
        if not np.allclose(reconstructed, value["logits"], rtol=1e-8, atol=1e-8):
            raise ValueError("Cached logits differ from cached head")
        return value
    result = fit_logistic(left, right, positive, trials, c_value, basis=basis)
    value = {
        **result,
        "logits": result["logits"].tolist(),
        "coef": None if result["coef"] is None else result["coef"].tolist(),
        "fingerprint": fingerprint,
        "C": c_value,
    }
    save_json(path, value)
    print(
        f"[risk-fit] {path.name} C={c_value:g} status={result['status']} seconds={result['elapsed_seconds']:.3f}",
        flush=True,
    )
    return result


def fit_method(
    bank: FeatureBank,
    method: str,
    train: np.ndarray,
    test: np.ndarray,
    folds: list,
    positive: np.ndarray,
    trials: np.ndarray,
    out: Path,
    regime: str,
    spec: dict,
) -> dict:
    """Tune using grouped training folds and evaluate the final test only after selection."""
    ranks = (
        spec["secondary_PCA_control"]["rank_candidates"]
        if method == "pca_plus_metadata"
        else [None]
    )
    candidates = [(rank, c) for rank in ranks for c in spec["regularization_C"]]
    losses = {key: 0.0 for key in candidates}
    validation_trials = 0
    cv_rows = []
    for fold, (fit_indices, validation_indices) in enumerate(folds):
        validation_trials += int(trials[validation_indices].sum())
        effective_cache = {}
        for rank in ranks:
            left, right, info = bank.features(
                method, fit_indices, validation_indices, trials, rank=rank
            )
            effective = info.get("effective_rank")
            if effective not in effective_cache:
                compressed, projected, basis = l2_basis(left, right)
                effective_cache[effective] = (compressed, projected, basis, {})
            compressed, projected, basis, fit_cache = effective_cache[effective]
            for c_value in spec["regularization_C"]:
                if c_value not in fit_cache:
                    key = {
                        "regime": regime,
                        "method": method,
                        "fold": fold,
                        "fit_indices": fit_indices.tolist(),
                        "test_indices": validation_indices.tolist(),
                        "effective_rank": effective,
                        "C": c_value,
                    }
                    path = out / "fits" / f"{method}_fold{fold}_rank{effective}_C{c_value:g}.json"
                    fit_cache[c_value] = cached_fit(
                        path,
                        digest(key),
                        compressed,
                        projected,
                        positive[fit_indices],
                        trials[fit_indices],
                        c_value,
                        basis,
                    )
                fitted = fit_cache[c_value]
                loss_sum = float(
                    loss_terms(
                        fitted["logits"], positive[validation_indices], trials[validation_indices]
                    ).sum()
                )
                losses[(rank, c_value)] += loss_sum
                cv_rows.append(
                    {
                        "fold": fold,
                        "rank": rank,
                        "C": c_value,
                        "loss_sum": loss_sum,
                        "trials": int(trials[validation_indices].sum()),
                        "status": fitted["status"],
                        **info,
                    }
                )
    selected_rank, selected_c = min(candidates, key=lambda k: (losses[k], k[0] or 0, k[1]))
    selected = {
        "method": method,
        "rank": selected_rank,
        "C": selected_c,
        "validation_log_loss": losses[(selected_rank, selected_c)] / validation_trials,
        "cv_rows": cv_rows,
        "regime": regime,
    }
    save_json(out / f"{method}_selection.json", selected)
    left, right, info = bank.features(method, train, test, trials, rank=selected_rank)
    compressed, projected, basis = l2_basis(left, right)
    key = {
        "regime": regime,
        "method": method,
        "stage": "final",
        "rank": selected_rank,
        "C": selected_c,
        "fit_indices": train.tolist(),
        "test_indices": test.tolist(),
    }
    fitted = cached_fit(
        out / "fits" / f"{method}_final.json",
        digest(key),
        compressed,
        projected,
        positive[train],
        trials[train],
        selected_c,
        basis,
    )
    if method == "mapped_plus_metadata" and fitted["coef"] is not None:
        composed = bank.compose_mapped_head(
            train, test, trials, fitted["coef"], fitted["intercept"]
        )
        if not np.allclose(composed, fitted["logits"], rtol=1e-8, atol=1e-8):
            raise ValueError("Mapped-head/raw-composition logit equivalence failed")
        info["composed_logit_max_error"] = float(np.max(np.abs(composed - fitted["logits"])))
    return {
        "selection": selected,
        "feature_checks": info,
        "logits": fitted["logits"].tolist(),
        "metrics": metric_report(fitted["logits"], positive[test], trials[test]),
    }


def analyze_population(
    rows: list[dict],
    bank: FeatureBank,
    spec: dict,
    out: Path,
    provenance: dict,
    *,
    eligible_only: bool,
) -> dict:
    eligible = {r["task_id"] for r in rows if r["condition"] == "original" and r["positive"] >= 1}
    keep = np.asarray(
        [
            i
            for i, r in enumerate(rows)
            if r["condition"] != "original" and (not eligible_only or r["task_id"] in eligible)
        ],
        dtype=int,
    )
    train = np.asarray(
        [i for i in keep if rows[i]["public_test_role"] == spec["training_role"]], dtype=int
    )
    test = np.asarray(
        [i for i in keep if rows[i]["public_test_role"] == spec["test_role"]], dtype=int
    )
    if set(train) & set(test) or len(train) + len(test) != len(keep):
        raise ValueError("Unknown or overlapping final partitions")
    support_train, support_test = class_support(rows, train), class_support(rows, test)
    result = {
        "eligible_only": eligible_only,
        "training_support": support_train,
        "test_support": support_test,
        "provenance": provenance,
        "models": {},
        "comparisons": {},
    }
    gate = spec["claim_gate"]
    train_passed = (
        support_train["positive_tasks"] >= gate["minimum_positive_training_tasks"]
        and support_train["negative_tasks"] >= gate["minimum_negative_training_tasks"]
    )
    result["fit_gate_passed"] = bool(train_passed)
    if not train_passed or not len(test):
        result["status"] = (
            "insufficient_training_support" if not train_passed else "no_test_contexts"
        )
        save_json(out / "result.json", result)
        return result
    groups = np.asarray([r["task_id"] for r in rows])
    positive = np.asarray([r["positive"] for r in rows], dtype=int)
    trials = np.asarray([r["trials"] for r in rows], dtype=int)
    folds = [
        (train[a], train[b]) for a, b in GroupKFold(n_splits=5).split(train, groups=groups[train])
    ]
    for a, b in folds:
        if set(groups[a]) & set(groups[b]) or set(groups[a]) & set(groups[test]):
            raise ValueError("Task-group leakage in cross-validation")
    regime = digest(
        {
            "provenance": provenance,
            "eligible_only": eligible_only,
            "train": train.tolist(),
            "test": test.tolist(),
            "spec": spec,
        }
    )
    p = float((positive[train].sum() + 0.5) / (trials[train].sum() + 1))
    prevalence = np.full(len(test), np.log(p) - np.log1p(-p))
    result["models"]["prevalence"] = {
        "logits": prevalence.tolist(),
        "metrics": metric_report(prevalence, positive[test], trials[test]),
    }
    methods = [m for m in spec["readouts"] if m != "prevalence"]
    methods += [f"orientation_{seed}" for seed in spec["orientation_controls"]["seeds"]]
    for method in methods:
        result["models"][method] = fit_method(
            bank, method, train, test, folds, positive, trials, out, regime, spec
        )
        save_json(out / "partial_result.json", result)
    claim_support = (
        support_test["positive_tasks"] >= gate["minimum_positive_test_tasks"]
        and support_test["negative_tasks"] >= gate["minimum_negative_test_tasks"]
        and support_test["positive"] > 0
        and support_test["negative"] > 0
    )
    result["claim_support_passed"] = bool(claim_support)
    for label, first, second in (
        ("raw_over_text", "text_plus_metadata", "raw_plus_metadata"),
        ("mapped_over_raw", "raw_plus_metadata", "mapped_plus_metadata"),
    ):
        if support_test["n_tasks"] >= 2:
            comparison = paired_task_interval(
                np.asarray(result["models"][first]["logits"]),
                np.asarray(result["models"][second]["logits"]),
                positive[test],
                trials[test],
                groups[test],
                replicates=spec["uncertainty"]["replicates"],
                seed=spec["uncertainty"]["seed"],
            )
            comparison["benefit_supported"] = bool(
                claim_support and comparison["positive_interval"]
            )
        else:
            comparison = {"benefit_supported": False, "reason": "fewer_than_two_test_tasks"}
        result["comparisons"][label] = comparison
    if eligible_only and result["comparisons"]["mapped_over_raw"]["benefit_supported"]:
        result["models"]["pca_plus_metadata"] = fit_method(
            bank, "pca_plus_metadata", train, test, folds, positive, trials, out, regime, spec
        )
    result["test_predictions"] = [
        {
            "task_id": rows[i]["task_id"],
            "condition": rows[i]["condition"],
            "exact_context_sha256": rows[i]["exact_context_sha256"],
            "positive": int(positive[i]),
            "trials": int(trials[i]),
            "logits": {m: value["logits"][j] for m, value in result["models"].items()},
        }
        for j, i in enumerate(test)
    ]
    result["status"] = "complete"
    save_json(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    incomplete = incomplete_census(args.root)
    if incomplete is not None:
        save_json(args.output_dir / "result.json", incomplete)
        print(json.dumps(incomplete, indent=2))
        return
    rows, raw, map_arrays, spec, provenance = load_inputs(
        args.root, args.captures, args.map, args.spec
    )
    repository = Path(__file__).resolve().parent.parent
    provenance["analysis_sources_sha256"] = {p: sha256(repository / p) for p in ANALYSIS_SOURCES}
    provenance["packages"] = {
        p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")
    }
    bank = FeatureBank(rows, raw, map_arrays, spec)
    primary = analyze_population(
        rows, bank, spec, args.output_dir / "primary", provenance, eligible_only=True
    )
    sensitivity = analyze_population(
        rows, bank, spec, args.output_dir / "all_tasks", provenance, eligible_only=False
    )
    save_json(
        args.output_dir / "result.json",
        {
            "primary": primary,
            "all_tasks": sensitivity,
            "interpretation": "Context-risk forecasts on unseen public benchmark tasks; affine mapping can change regularization but cannot add information.",
        },
    )


if __name__ == "__main__":
    main()
