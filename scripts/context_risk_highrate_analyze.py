"""Analyze the selected fresh cohort with frozen linear readouts and explicit unknowns."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from scripts import context_risk_highrate_capture as capture  # noqa: E402
from scripts import context_risk_highrate_design as design  # noqa: E402
from scripts import context_risk_highrate_transport as transport  # noqa: E402
from scripts import context_risk_highrate_validity as validity  # noqa: E402
from scripts.context_risk_analyze import load_impossible_activations  # noqa: E402
from scripts.context_risk_followup_analyze import class_support, digest, fit_method, sha256  # noqa: E402
from scripts.context_risk_followup_features import FeatureBank  # noqa: E402
from scripts.context_risk_followup_probe_core import (  # noqa: E402
    metric_report,
    paired_task_interval,
    save_json,
)

SPEC_SHA = "02c4d4009f559d4f829c43e90ba722018fc5ad9c54481780cf0a15e0d895a509"
OPTIMIZER_SPEC_SHA = "154d6038216ece9af36ee7092a575ab5244a14afd34a746f71d2d1566a32fb71"
SOURCES = (
    "scripts/context_risk_highrate_analyze.py",
    "scripts/context_risk_highrate_optimizer.py",
    "tests/test_context_risk_highrate_optimizer.py",
    "eval_results/context_risk_highrate_design/analysis_spec_v10_optimizer.json",
    "eval_results/context_risk_highrate_design/plan_v10_optimizer_convergence.md",
    "configs/eval/context_risk_highrate_analyze.yaml",
    "tests/test_context_risk_highrate_analyze.py",
    "scripts/context_risk_followup_analyze.py",
    "scripts/context_risk_followup_features.py",
    "scripts/context_risk_followup_probe_core.py",
    "scripts/context_risk_analyze.py",
    "scripts/context_risk_highrate_validity.py",
    "eval_results/context_risk_highrate_design/input_validity_policy.json",
    "eval_results/context_risk_highrate_design/plan_v7_input_validity.md",
)


def source_hashes() -> dict:
    """Include actual collection, capture and fitting helpers in every cache/review regime."""
    names = set(SOURCES) | set(capture.source_hashes())
    hashes = {name: sha256(design.PROJECT / name) for name in sorted(names)}
    for name in SOURCES:
        if name.startswith("scripts/") and name.endswith(".py"):
            module = importlib.import_module(name.removesuffix(".py").replace("/", "."))
            if sha256(Path(module.__file__)) != hashes[name]:
                raise ValueError(f"Imported analysis helper differs from reviewed source: {name}")
    return hashes


def review(path: Path) -> dict:
    """Require the independent reviewer to bind this complete implementation closure."""
    value = json.loads(path.read_text())
    if (
        value.get("verdict") != "PASS"
        or not value.get("reviewer")
        or value.get("sources_sha256") != source_hashes()
    ):
        raise ValueError("Analysis requires an independent PASS on current source bytes")
    return value


def validated_evidence(root: Path, captures: Path) -> dict:
    """Recheck native requests, owned terminal processes and captured tensors read-only."""
    evidence = {
        "screen": transport.verify_report(root, "screen"),
        "fresh": transport.verify_report(root, "fresh"),
        "screen_process": transport.validate_terminal_process(root, "screen"),
        "fresh_process": transport.validate_terminal_process(root, "fresh"),
        "capture": capture.validate_binding(captures),
    }
    review_path = root / "fresh_B/success_review.json"
    review_hash = sha256(review_path)
    reviewed = json.loads(review_path.read_text())
    if (
        reviewed.get("verdict") != "PASS"
        or not reviewed.get("reviewer")
        or reviewed["native_logs_sha256"] != evidence["fresh"]["native_logs_sha256"]
        or reviewed["success_evidence"] != design.success_evidence(root, "fresh")
        or review_hash != sha256(review_path)
    ):
        raise ValueError("Fresh successful bodies lack the complete exact independent review")
    evidence["fresh_success_review"] = reviewed
    evidence["fresh_success_review_sha256"] = review_hash
    return evidence


def load_inputs(root: Path, captures: Path, map_path: Path, spec_path: Path):
    """Join all90 prefixes to observed counts; N stays unknown and reduces completed trials."""
    spec_hash = sha256(spec_path)
    if captures.resolve() != (root / "capture").resolve() or spec_hash not in {SPEC_SHA, OPTIMIZER_SPEC_SHA}:
        raise ValueError("Analysis capture path/specification differs from the frozen recipe")
    spec = json.loads(spec_path.read_text())
    map_hash = spec["map_sha256"]
    if sha256(map_path) != map_hash:
        raise ValueError("Map differs from the independently frozen mapping")
    manifest, selection, epochs = design.load_phase(root, "fresh")
    if epochs != 4:
        raise ValueError("Fresh phase requires four trajectories per context")
    source_snapshot = source_hashes()
    immutable = {
        str(spec_path): spec_hash,
        str(map_path): map_hash,
        str(manifest): sha256(manifest),
        str(root / "selection.json"): sha256(root / "selection.json"),
        str(root / "fresh_B/prefix_tokens.json"): sha256(root / "fresh_B/prefix_tokens.json"),
    }
    evidence = validated_evidence(root, captures)
    validity_audit = validity.source_audit(root)
    immutable[str(root / "manifests/source.jsonl")] = validity_audit["source_sha256"]
    immutable[str(validity.POLICY)] = validity_audit["policy_sha256"]
    fresh, screen = evidence["fresh"], evidence["screen"]
    validity_selection = validity.selection_annotation(screen["contexts"], validity_audit)
    if validity_selection["task_roles"] != selection["task_roles"]:
        raise ValueError("Static validity audit and frozen selected task roles differ")
    fresh_validity = validity.annotate(fresh["contexts"], validity_audit)
    fresh_validity.pop("rank_rows_with_semantic_unknowns")
    if (
        fresh["counts"]["planned"] != 360
        or fresh["counts"]["realized"] != 360
        or fresh["counts"]["missing"] != 0
        or len(fresh["contexts"]) != 90
        or screen["counts"]["planned"] != 618
        or screen["counts"]["realized"] != 618
        or screen["counts"]["missing"] != 0
        or len(screen["contexts"]) != 309
    ):
        raise ValueError("Native phase census differs from618 screen and360 fresh draws")
    rows = sorted(design.read_rows(manifest), key=lambda r: (r["task_id"], r["condition"]))
    observed = {r["exact_context_sha256"]: r for r in fresh["contexts"]}
    screened = {(r["task_id"], r["condition"]): r for r in screen["contexts"]}
    expected = {r["exact_context_sha256"] for r in rows}
    if len(rows) != 90 or len(expected) != 90 or set(observed) != expected:
        raise ValueError("Fresh manifest/behavior roster lacks90 distinct exact contexts")
    activation, captured = load_impossible_activations(captures, selected_layer=44)
    if set(activation) != expected or set(captured) != expected:
        raise ValueError("Captured activation roster differs from fresh behavior")
    prefix_path = root / "fresh_B/prefix_tokens.json"
    prefixes = {
        r["exact_context_sha256"]: r for r in json.loads(prefix_path.read_text())["contexts"]
    }
    for row in rows:
        key = row["exact_context_sha256"]
        count, prior = observed[key], screened[(row["task_id"], row["condition"])]
        if (
            count["planned"] != 4
            or count["realized"] != 4
            or count["missing"] != 0
            or count["success"] + count["failure"] + count["censored"] != 4
            or prior["planned"] != 2
            or prior["realized"] != 2
            or prior["missing"] != 0
            or prior["success"] + prior["failure"] + prior["censored"] != 2
        ):
            raise ValueError("Per-contextscreen/fresh binomial denominator differs")
        if (
            any(
                captured[key][field] != row[field]
                for field in ("task_id", "condition", "public_test_role")
            )
            or selection["task_roles"][row["task_id"]] != row["public_test_role"]
        ):
            raise ValueError("Captured context identity or frozen task role differs")
        if any(
            captured[key][field] != prefixes[key][field]
            for field in ("n_prefix_tokens", "prefix_token_ids_sha256")
        ):
            raise ValueError("Captured prefix differs from the actual generation token IDs")
        row.update(
            {
                "positive": count["success"],
                "trials": count["success"] + count["failure"],
                "censored": count["censored"],
                "planned": 4,
                "screen_success": prior["success"],
                "screen_failure": prior["failure"],
                "screen_censored": prior["censored"],
            }
        )
    raw = np.stack([activation[r["exact_context_sha256"]] for r in rows])
    if raw.shape != (90, 5120) or not np.isfinite(raw).all():
        raise ValueError("Raw capture matrix differs from90 by5120 finite features")
    with np.load(map_path, allow_pickle=False) as arrays:
        mapped = {key: arrays[key].copy() for key in ("weight", "x_mean", "x_scale", "y_mean")}
    if (
        any(sha256(Path(path)) != expected for path, expected in immutable.items())
        or source_snapshot != source_hashes()
    ):
        raise ValueError("Analysis source or pinned input changed during loading")
    provenance = {
        "spec_sha256": spec_hash,
        "map_sha256": map_hash,
        "manifest_sha256": immutable[str(manifest)],
        "selection_sha256": immutable[str(root / "selection.json")],
        "input_files_sha256": immutable,
        "validation_sha256": digest(evidence),
        "analysis_sources_sha256": source_snapshot,
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
        "fresh_counts": fresh["counts"],
        "screen_counts": screen["counts"],
        "input_validity": {"screen": validity_selection, "fresh": fresh_validity},
    }
    return rows, raw, mapped, spec, provenance


def screening_features(rows: list[dict]) -> np.ndarray:
    """Independent screening behavior is a distinct, equally shared secondary information set."""
    result = []
    for row in rows:
        s, f, u = (row[k] for k in ("screen_success", "screen_failure", "screen_censored"))
        if any(type(v) is not int or v < 0 for v in (s, f, u)) or s + f + u != 2:
            raise ValueError("Screening feature counts must retain two S/F/U draws per context")
        result.append([(s + 0.5) / (s + f + 1), s + f, u])
    return np.asarray(result, dtype=np.float64)


def make_bank(rows, raw, map_arrays, spec, *, augmented: bool) -> FeatureBank:
    """Append screening features before any fold preparation; never mutate a cached bank."""
    bank = FeatureBank(rows, raw, map_arrays, spec)
    if augmented:
        bank.metadata = np.column_stack([bank.metadata, screening_features(rows)])
    return bank


def competence(rows: list[dict]) -> dict:
    """Separate observed eligibility from definitely ineligible and unresolved originals."""
    original = [r for r in rows if r["condition"] == "original"]
    return {
        "eligible_tasks": sorted(r["task_id"] for r in original if r["positive"] > 0),
        "ineligible_tasks": sorted(
            r["task_id"] for r in original if r["positive"] == 0 and r["censored"] == 0
        ),
        "unknown_tasks": sorted(
            r["task_id"] for r in original if r["positive"] == 0 and r["censored"] > 0
        ),
    }


def analyze_population(
    rows, bank, spec, out: Path, provenance: dict, *, eligible_only: bool, augmented: bool = False
) -> dict:
    """Fit descriptive probes at adequate fold support and gate benefit claims separately."""
    eligibility = competence(rows)
    eligible = set(eligibility["eligible_tasks"])
    included = [
        i
        for i, r in enumerate(rows)
        if r["condition"] != "original" and (not eligible_only or r["task_id"] in eligible)
    ]
    keep = np.asarray([i for i in included if rows[i]["trials"] > 0], dtype=int)
    train = np.asarray(
        [i for i in keep if rows[i]["public_test_role"] == spec["training_role"]], dtype=int
    )
    test = np.asarray(
        [i for i in keep if rows[i]["public_test_role"] == spec["test_role"]], dtype=int
    )
    groups = np.asarray([r["task_id"] for r in rows])
    if (
        set(train) & set(test)
        or len(train) + len(test) != len(keep)
        or set(groups[train]) & set(groups[test])
    ):
        raise ValueError("Unknown, overlapping or task-leaking final partitions")
    positive = np.asarray([r["positive"] for r in rows], dtype=int)
    trials = np.asarray([r["trials"] for r in rows], dtype=int)
    left, right = class_support(rows, train), class_support(rows, test)
    gate = spec["claim_gate"]
    total_censored = provenance["fresh_counts"]["censored"]
    not_assessable = provenance["input_validity"]["fresh"]["not_assessable_planned_trajectories"]
    fitting = (
        left["n_tasks"] >= spec["minimum_training_groups_for_fit"]
        and left["positive"] > 0
        and left["negative"] > 0
    )
    claim_support = (
        left["positive_tasks"] >= gate["minimum_positive_training_tasks"]
        and left["negative_tasks"] >= gate["minimum_negative_training_tasks"]
        and right["positive_tasks"] >= gate["minimum_positive_test_tasks"]
        and right["negative_tasks"] >= gate["minimum_negative_test_tasks"]
        and total_censored <= gate["maximum_censored_fresh_trajectories"]
        and not_assessable == 0
    )
    result = {
        "population": "competence_sensitivity" if eligible_only else "all_selected_primary",
        "information_budget": "initial_context_plus_independent_screen"
        if augmented
        else "initial_context",
        "eligibility": eligibility,
        "training_support": left,
        "test_support": right,
        "fit_gate_passed": bool(fitting),
        "claim_support_passed": bool(claim_support),
        "total_fresh_censored": total_censored,
        "structurally_not_assessable_fresh_trajectories": not_assessable,
        "provenance": provenance,
        "models": {},
        "comparisons": {},
        "excluded_zero_complete_case_contexts": [
            rows[i]["exact_context_sha256"] for i in included if rows[i]["trials"] == 0
        ],
        "scope": "Hacking-risk fits use structurally assessable contexts and completed trajectories. Literal native reward failures remain unchanged; malformed-input failures are excluded from semantic negative labels. Any fresh censoring or structural non-assessability blocks unconditional supported-benefit claims.",
    }
    if not len(test):
        result["status"] = "no_completed_test_contexts"
        save_json(out / "result.json", result)
        return result
    p = float((positive[train].sum() + 0.5) / (trials[train].sum() + 1))
    baseline = np.full(len(test), np.log(p) - np.log1p(-p))
    result["models"]["prevalence"] = {
        "logits": baseline.tolist(),
        "metrics": metric_report(baseline, positive[test], trials[test]),
    }
    if augmented:
        p_screen = screening_features(rows)[test, 0]
        logits = np.log(p_screen) - np.log1p(-p_screen)
        result["models"]["direct_screen_rate"] = {
            "logits": logits.tolist(),
            "metrics": metric_report(logits, positive[test], trials[test]),
        }
    if fitting:
        folds = [
            (train[a], train[b])
            for a, b in GroupKFold(n_splits=5).split(train, groups=groups[train])
        ]
        for a, b in folds:
            if set(groups[a]) & set(groups[b]) or set(groups[a]) & set(groups[test]):
                raise ValueError("Task-group leakage in inner cross-validation")
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
        methods = [m for m in spec["readouts"] if m != "prevalence"]
        methods += [f"orientation_{seed}" for seed in spec["orientation_controls"]["seeds"]]
        for method in methods:
            result["models"][method] = fit_method(
                bank, method, train, test, folds, positive, trials, out, regime, spec
            )
            save_json(out / "partial_result.json", result)
        for label, first, second in (
            ("raw_over_text", "text_plus_metadata", "raw_plus_metadata"),
            ("mapped_over_raw", "raw_plus_metadata", "mapped_plus_metadata"),
        ):
            if right["n_tasks"] >= 2:
                interval = paired_task_interval(
                    np.asarray(result["models"][first]["logits"]),
                    np.asarray(result["models"][second]["logits"]),
                    positive[test],
                    trials[test],
                    groups[test],
                    replicates=spec["uncertainty"]["replicates"],
                    seed=spec["uncertainty"]["seed"],
                )
                interval["benefit_supported"] = bool(
                    claim_support and interval["positive_interval"]
                )
            else:
                interval = {"benefit_supported": False, "reason": "fewer_than_two_test_tasks"}
            result["comparisons"][label] = interval
        if (
            not eligible_only
            and not augmented
            and result["comparisons"]["mapped_over_raw"]["benefit_supported"]
        ):
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
            "censored": rows[i]["censored"],
            "logits": {method: value["logits"][j] for method, value in result["models"].items()},
        }
        for j, i in enumerate(test)
    ]
    result["status"] = "complete" if fitting else "prevalence_only_insufficient_training_support"
    save_json(out / "result.json", result)
    return result


def run(cfg: DictConfig) -> dict:
    """Run all four frozen population/information analyses with immutable resumable fit caches."""
    root, captures = Path(cfg.root).resolve(), Path(cfg.captures).resolve()
    map_path, spec_path, out = Path(cfg.map), Path(cfg.spec), Path(cfg.output_dir).resolve()
    if out != root / "analysis":
        raise ValueError("Analysis output must use its fixed phase output directory")
    review_hash = sha256(Path(cfg.review))
    reviewed = review(Path(cfg.review))
    rows, raw, maps, spec, provenance = load_inputs(root, captures, map_path, spec_path)
    if review_hash != sha256(Path(cfg.review)):
        raise ValueError("Independent analysis review changed during loading")
    provenance["independent_review_sha256"] = review_hash
    launch = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "provenance": provenance,
        "review": reviewed,
        "rows_sha256": digest(rows),
    }
    launch_path = out / "analysis_launch.json"
    if not launch_path.exists() and out.exists() and any(out.iterdir()):
        raise ValueError("First analysis launch requires empty output; cannot adopt unbound fits")
    if launch_path.exists() and json.loads(launch_path.read_text()) != launch:
        raise ValueError("Analysis resume input/source/runtime/review regime changed")
    save_json(launch_path, launch)
    save_json(
        out / "input_census.json",
        {"contexts": rows, "provenance": provenance, "eligibility": competence(rows)},
    )
    result = {
        "schema_version": "context_risk_highrate_analysis_v1",
        "verification_passed": False,
        "provenance": provenance,
        "analyses": {},
        "map_interpretation": "A frozen deterministic affine map changes linear regularization; it adds no information beyond the raw activation.",
    }
    save_json(out / "result.json", result)
    fit_rows, fit_raw = validity.assessable_inputs(
        rows, raw, provenance["input_validity"]["fresh"]["static_audit"]
    )
    for augmented in (False, True):
        bank = make_bank(fit_rows, fit_raw, maps, spec, augmented=augmented)
        for eligible_only in (False, True):
            name = ("screen_augmented_" if augmented else "") + (
                "competence_sensitivity" if eligible_only else "primary"
            )
            result["analyses"][name] = analyze_population(
                fit_rows,
                bank,
                spec,
                out / name,
                provenance,
                eligible_only=eligible_only,
                augmented=augmented,
            )
            save_json(out / "result.json", result)
    if (
        provenance["analysis_sources_sha256"] != source_hashes()
        or provenance["validation_sha256"] != digest(validated_evidence(root, captures))
        or provenance["map_sha256"] != sha256(map_path)
        or provenance["spec_sha256"] != sha256(spec_path)
        or review(Path(cfg.review)) != reviewed
        or provenance["independent_review_sha256"] != sha256(Path(cfg.review))
        or any(
            sha256(Path(path)) != expected
            for path, expected in provenance["input_files_sha256"].items()
        )
    ):
        raise ValueError("Analysis source, input or review evidence changed while fitting")
    result["verification_passed"] = True
    save_json(out / "result.json", result)
    return result


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_highrate_analyze"
)
def main(cfg: DictConfig) -> None:
    """Dispatch the reviewed CPU analysis with the frozen Hydra paths."""
    print(json.dumps(run(cfg), sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
