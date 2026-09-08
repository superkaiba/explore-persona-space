"""Grouped linear forecasts at frozen before-action and reconstructed-answer landmarks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from scripts.context_risk_followup_analyze import class_support, digest, fit_method  # noqa: E402
from scripts.context_risk_followup_features import FeatureBank, metadata_matrix  # noqa: E402
from scripts.context_risk_followup_probe_core import loss_terms, metric_report, save_json  # noqa: E402

STAGES = [f"pre_action_{i:02d}" for i in range(1, 11)] + [
    "within_pre",
    "within_32",
    "within_128",
    "within_512",
    "within_end",
]
CONTRASTS = [
    ("raw_over_prevalence", "prevalence", "raw_plus_metadata"),
    ("raw_over_text", "text_plus_metadata", "raw_plus_metadata"),
    ("mapped_over_raw", "raw_plus_metadata", "mapped_plus_metadata"),
]
SOURCES = [
    "scripts/context_risk_trajectory_analyze.py",
    "scripts/context_risk_trajectory_prepare.py",
    "scripts/context_risk_trajectory_capture.py",
    "scripts/context_risk_followup_analyze.py",
    "scripts/context_risk_followup_features.py",
    "scripts/context_risk_followup_probe_core.py",
    "scripts/context_risk_highrate_optimizer.py",
    "configs/eval/context_risk_trajectory_analyze.yaml",
    "tests/test_context_risk_trajectory_analyze.py",
]


def sha256(path: Path) -> str:
    """Hash files incrementally, including arrays too large to duplicate in memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {name: sha256(root / name) for name in SOURCES}


def imported_source_hashes() -> dict:
    """Bind the modules actually imported, rejecting same-name shadow copies."""
    root = Path(__file__).resolve().parent.parent
    result = {}
    for name in SOURCES:
        if not name.startswith("scripts/"):
            continue
        module = importlib.import_module(name.removesuffix(".py").replace("/", "."))
        actual = Path(module.__file__).resolve()
        if actual != (root / name).resolve():
            raise ValueError(f"Imported analysis helper is shadowed: {name}")
        result[name] = sha256(actual)
    return result


def input_snapshot(paths: list[Path]) -> dict:
    """Freeze complete small input trees and control files before consuming them."""
    files = set()
    for path in paths:
        if path.is_dir():
            entries = list(path.rglob("*"))
            if any(p.is_symlink() for p in entries):
                raise ValueError("Immutable input trees cannot contain symlinks")
            files.update(p.resolve() for p in entries if p.is_file())
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise FileNotFoundError(path)
    return {str(path): sha256(path) for path in sorted(files)}


def assert_disjoint_output(out: Path, inputs: list[Path]) -> None:
    """Analysis writes must never overlap any input tree or control file."""
    out = out.resolve()
    for path in inputs:
        path = path.resolve()
        if out == path or out in path.parents or (path.is_dir() and path in out.parents):
            raise ValueError(f"Analysis output overlaps input: {path}")


def stage_metadata(row: dict) -> np.ndarray:
    """Only information observed by this checkpoint; no terminal labels or lengths."""
    base = metadata_matrix([row])[0]
    attempt = row["attempt"]
    request_tokens, observed_tokens = row["request_tokens"], row["observed_tokens"]
    ended = row["answer_ended"]
    if (
        type(attempt) is not int
        or not 1 <= attempt <= 10
        or type(request_tokens) is not int
        or request_tokens <= 0
        or type(observed_tokens) is not int
        or observed_tokens < 0
        or type(ended) is not bool
    ):
        raise ValueError("Invalid observed checkpoint metadata")
    return np.r_[
        base, np.log1p(attempt), np.log1p(request_tokens), np.log1p(observed_tokens), float(ended)
    ]


class StageFeatureBank(FeatureBank):
    """Preserve the frozen map/linear fit while exposing the complete observed history."""

    def __init__(self, rows, raw, map_arrays, spec):
        super().__init__(rows, raw, map_arrays, spec)
        self.metadata = np.vstack([stage_metadata(row) for row in rows])
        # Object strings avoid allocating n_rows times the longest trajectory text.
        self.prompts = np.asarray([row["visible_text"] for row in rows], dtype=object)
        if self.metadata.shape != (len(rows), 8) or self.cache:
            raise ValueError("Stage metadata/cache initialization differs")


def aggregate_stage(stage, observations, checkpoints, contexts, text_bank, checkpoint_text):
    """Join original trials, collapsing only identical task-local observable inputs."""
    if stage not in STAGES:
        raise ValueError("Unplanned trajectory stage")
    accumulated = {}
    seen = set()
    for observation in observations:
        if observation["stage"] != stage:
            continue
        event = (observation["sample_id"], observation["epoch"], observation["attempt"])
        if event in seen:
            raise ValueError("Duplicate stage observation")
        seen.add(event)
        if type(observation["positive"]) is not int or observation["positive"] not in {0, 1}:
            raise ValueError("Nonbinary or missing behavioral label")
        checkpoint = checkpoints[observation["checkpoint_key"]]
        context = contexts[observation["context_key"]]
        for field in ("task_id", "condition", "public_test_role"):
            if observation[field] != context[field]:
                raise ValueError(f"Static context join differs: {field}")
        row = {
            "task_id": observation["task_id"],
            "condition": observation["condition"],
            "public_test_role": observation["public_test_role"],
            "messages": context["messages"],
            "test": context["test"],
            "checkpoint_key": observation["checkpoint_key"],
            "context_key": observation["context_key"],
            "attempt": observation["attempt"],
            "request_tokens": observation["request_tokens"],
            "observed_tokens": observation["observed_tokens"],
            "answer_ended": observation["answer_ended"],
            "visible_text": checkpoint_text(checkpoint, text_bank),
        }
        if stage.startswith("pre_action_") or stage == "within_pre":
            if row["observed_tokens"] != 0 or row["answer_ended"]:
                raise ValueError("Before-generation row contains response information")
        elif stage == "within_end":
            if not row["answer_ended"] or row["observed_tokens"] <= 0:
                raise ValueError("Answer-end row is incomplete")
        else:
            limit = int(stage.removeprefix("within_"))
            if not 0 < row["observed_tokens"] <= limit:
                raise ValueError("Response checkpoint exceeds its information budget")
            if row["observed_tokens"] < limit and not row["answer_ended"]:
                raise ValueError("Short response must retain its ended indicator")
        if checkpoint["length"] != row["request_tokens"] + row["observed_tokens"]:
            raise ValueError("Checkpoint token length differs from observable metadata")
        key = (
            row["task_id"],
            row["condition"],
            row["checkpoint_key"],
            tuple(stage_metadata(row)),
            hashlib.sha256(row["visible_text"].encode()).hexdigest(),
        )
        if key not in accumulated:
            accumulated[key] = {**row, "positive": 0, "trials": 0, "events": []}
        target = accumulated[key]
        if target["public_test_role"] != row["public_test_role"]:
            raise ValueError("Repeated checkpoint crosses task partitions")
        target["positive"] += observation["positive"]
        target["trials"] += 1
        target["events"].append(list(event))
    rows = sorted(
        accumulated.values(),
        key=lambda r: (r["task_id"], r["condition"], r["checkpoint_key"], r["attempt"]),
    )
    if not rows:
        raise ValueError(f"Planned stage is empty: {stage}")
    return rows


def fit_stage(stage, rows, raw, map_arrays, spec, out, provenance):
    """Fit every method in shared task folds, retaining all candidate failures loudly."""
    started = time.monotonic()
    bank = StageFeatureBank(rows, raw, map_arrays, spec)
    groups = np.asarray([r["task_id"] for r in rows])
    train = np.asarray(
        [i for i, r in enumerate(rows) if r["public_test_role"] == spec["training_role"]], dtype=int
    )
    test = np.asarray(
        [i for i, r in enumerate(rows) if r["public_test_role"] == spec["test_role"]], dtype=int
    )
    if (
        not len(train)
        or not len(test)
        or len(train) + len(test) != len(rows)
        or set(groups[train]) & set(groups[test])
    ):
        raise ValueError("Missing or task-leaking final partition")
    positive = np.asarray([r["positive"] for r in rows], dtype=int)
    trials = np.asarray([r["trials"] for r in rows], dtype=int)
    left, right = class_support(rows, train), class_support(rows, test)
    fitting = (
        left["n_tasks"] >= spec["minimum_training_groups_for_fit"]
        and left["positive"] > 0
        and left["negative"] > 0
    )
    gate = spec["claim_gate"]
    support = all(
        [
            left["positive_tasks"] >= gate["minimum_positive_training_tasks"],
            left["negative_tasks"] >= gate["minimum_negative_training_tasks"],
            right["positive_tasks"] >= gate["minimum_positive_test_tasks"],
            right["negative_tasks"] >= gate["minimum_negative_test_tasks"],
        ]
    )
    result = {
        "stage": stage,
        "training_support": left,
        "test_support": right,
        "fit_gate_passed": bool(fitting),
        "conditional_claim_support_passed": bool(support),
        "scope": "Conditional on completed structurally assessable trajectories; exploratory",
        "models": {},
        "provenance": provenance,
        "response_ended_trials": int(sum(r["trials"] for r in rows if r["answer_ended"])),
    }
    p = float((positive[train].sum() + 0.5) / (trials[train].sum() + 1))
    logits = np.full(len(test), np.log(p) - np.log1p(-p))
    result["models"]["prevalence"] = {
        "logits": logits.tolist(),
        "metrics": metric_report(logits, positive[test], trials[test]),
    }
    fit_input = [
        {k: v for k, v in r.items() if k not in {"visible_text", "messages", "test"}}
        | {
            "visible_text_sha256": hashlib.sha256(r["visible_text"].encode()).hexdigest(),
            "messages_sha256": digest(r["messages"]),
            "test_sha256": hashlib.sha256(r["test"].encode()).hexdigest(),
        }
        for r in rows
    ]
    regime = digest(
        {
            "provenance": provenance,
            "spec": spec,
            "stage": stage,
            "rows": fit_input,
            "train": train.tolist(),
            "test": test.tolist(),
        }
    )
    save_json(out / "fit_inputs.json", {"regime": regime, "rows": fit_input})
    if fitting:
        folds = [
            (train[a], train[b])
            for a, b in GroupKFold(n_splits=5).split(train, groups=groups[train])
        ]
        for a, b in folds:
            if set(groups[a]) & set(groups[b]) or set(groups[a]) & set(groups[test]):
                raise ValueError("Task leakage in training folds")
        save_json(
            out / "folds.json",
            [
                {
                    "train": a.tolist(),
                    "validation": b.tolist(),
                    "training_tasks": sorted(set(groups[a])),
                    "validation_tasks": sorted(set(groups[b])),
                }
                for a, b in folds
            ],
        )
        methods = [m for m in spec["readouts"] if m != "prevalence"]
        methods += [f"orientation_{seed}" for seed in spec["orientation_controls"]["seeds"]]
        for method in methods:
            print(f"[trajectory-fit] stage={stage} method={method} starting", flush=True)
            result["models"][method] = fit_method(
                bank, method, train, test, folds, positive, trials, out, regime, spec
            )
            save_json(out / "partial_result.json", result)
    result["test_predictions"] = [
        {
            "task_id": rows[i]["task_id"],
            "condition": rows[i]["condition"],
            "checkpoint_key": rows[i]["checkpoint_key"],
            "events": rows[i]["events"],
            "positive": int(positive[i]),
            "trials": int(trials[i]),
            "logits": {name: model["logits"][j] for name, model in result["models"].items()},
        }
        for j, i in enumerate(test)
    ]
    result.update(
        status="complete" if fitting else "prevalence_only_insufficient_support",
        elapsed_seconds=time.monotonic() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    save_json(out / "result.json", result)
    print(
        f"[trajectory-fit] stage={stage} complete seconds={result['elapsed_seconds']:.3f}",
        flush=True,
    )
    return result


def joint_intervals(results: list[dict], spec: dict):
    """Common task resamples retain all cross-stage dependence in the planned scan."""
    if [r["stage"] for r in results] != STAGES:
        raise ValueError("Simultaneous scan requires all15 stages in frozen order")
    task_ids = sorted({p["task_id"] for r in results for p in r["test_predictions"]})
    cfg = spec["bootstrap"]
    if len(task_ids) != cfg["n_test_tasks"]:
        raise ValueError("Realized test task population differs from the frozen bootstrap")
    if cfg["family_contrasts"] != len(STAGES) * len(CONTRASTS):
        raise ValueError("Bootstrap family size differs")
    task_index = {task: i for i, task in enumerate(task_ids)}
    numerators = np.zeros((len(STAGES), len(CONTRASTS), len(task_ids)))
    references = np.zeros_like(numerators)
    denominators = np.zeros((len(STAGES), len(task_ids)))
    for s, result in enumerate(results):
        if result["status"] != "complete":
            return {
                "status": "insufficient_fit_support_for_full_scan",
                "stages": [r["stage"] for r in results if r["status"] != "complete"],
            }, None
        for row in result["test_predictions"]:
            t = task_index[row["task_id"]]
            k, n = np.asarray([row["positive"]]), np.asarray([row["trials"]])
            denominators[s, t] += row["trials"]
            for c, (_, first, second) in enumerate(CONTRASTS):
                first_loss = loss_terms(np.asarray([row["logits"][first]]), k, n)[0]
                second_loss = loss_terms(np.asarray([row["logits"][second]]), k, n)[0]
                numerators[s, c, t] += first_loss - second_loss
                references[s, c, t] += first_loss
    if np.any(denominators <= 0):
        raise ValueError("Every frozen test task must be represented at every stage")
    rng = np.random.default_rng(cfg["seed"])
    indices = rng.integers(len(task_ids), size=(cfg["replicates"], len(task_ids)))
    multiplicities = np.eye(len(task_ids), dtype=np.int64)[indices].sum(axis=1)
    sample_n = multiplicities @ denominators.T
    draws = np.einsum("bt,sct->bsc", multiplicities, numerators) / sample_n[:, :, None]
    estimates = numerators.sum(axis=2) / denominators.sum(axis=1)[:, None]
    reference_losses = references.sum(axis=2) / denominators.sum(axis=1)[:, None]
    if not np.isfinite(draws).all():
        raise ValueError("Nonfinite joint bootstrap draw; no row dropping or redrawing")
    deviations = np.max(np.abs(draws - estimates[None, :, :]), axis=(1, 2))
    q = float(np.quantile(deviations, 0.95))
    marginal = np.quantile(draws, [0.025, 0.975], axis=0)
    records = []
    for s, result in enumerate(results):
        comparisons = {}
        for c, (name, first, second) in enumerate(CONTRASTS):
            estimate = float(estimates[s, c])
            reference = float(reference_losses[s, c])
            comparisons[name] = {
                "first": first,
                "second": second,
                "improvement": estimate,
                "marginal_ci": [float(marginal[0, s, c]), float(marginal[1, s, c])],
                "simultaneous_ci": [estimate - q, estimate + q],
                "simultaneous_half_width": q,
                "reference_loss": reference,
                "scan_imprecision_exceeds_reference_loss": bool(q > reference),
                "conditional_benefit_supported": bool(
                    result["conditional_claim_support_passed"] and estimate - q > 0
                ),
            }
        raw_supported = all(
            comparisons[name]["conditional_benefit_supported"]
            for name in ("raw_over_prevalence", "raw_over_text")
        )
        records.append(
            {
                "stage": result["stage"],
                "comparisons": comparisons,
                "raw_prediction_supported": raw_supported,
                "mapping_benefit_supported": comparisons["mapped_over_raw"][
                    "conditional_benefit_supported"
                ],
                "causal_attribution_pending_prefix_confirmation": bool(
                    result["stage"] != "pre_action_01"
                    and (
                        raw_supported
                        or comparisons["mapped_over_raw"]["conditional_benefit_supported"]
                    )
                ),
            }
        )
    onset = {}
    for family in ("pre_action", "within"):
        subset = [r for r in records if r["stage"].startswith(family)]
        onset[family] = {
            name: next((r["stage"] for r in subset if r[name]), None)
            for name in ("raw_prediction_supported", "mapping_benefit_supported")
        }
    reference_stages = [
        stage
        for stage in STAGES
        if stage != "pre_action_01"
        and any(stage in candidates.values() for candidates in onset.values())
    ]
    return {
        "status": "complete",
        "task_ids": task_ids,
        "replicates": cfg["replicates"],
        "seed": cfg["seed"],
        "n_contrasts": len(STAGES) * len(CONTRASTS),
        "simultaneous_half_width": q,
        "stages": records,
        "earliest_scan_candidates": onset,
        "prefix_only_confirmation": {
            "status": "pending" if reference_stages else "not_required",
            "candidate_stages": reference_stages,
            "selection_rule": "Earliest raw and mapping candidate within each family; no replacement after failure",
        },
        "interpretation": "Later-stage candidates require separately truncated prefix confirmation; no unconfirmed causal-state benefit claim.",
        "uncertainty_scope": "Approximate exploratory fixed-fit9-task bootstrap; no refit uncertainty",
    }, {
        "draws": draws,
        "estimates": estimates,
        "task_indices": indices,
        "multiplicities": multiplicities,
        "reference_losses": reference_losses,
        "task_numerators": numerators,
        "task_denominators": denominators,
        "max_centered_deviations": deviations,
    }


def run(cfg: DictConfig) -> dict:
    """Validate archived inputs and independently reviewed source, then fit/resume stages."""
    from scripts.context_risk_trajectory_capture import verify_capture, verify_capture_geometry
    from scripts.context_risk_trajectory_prepare import (
        checkpoint_text,
        load_contexts,
        load_prepared,
        load_text_bank,
    )

    started = time.monotonic()
    prepared, capture, out = (Path(cfg[k]).resolve() for k in ("prepared", "capture", "output_dir"))
    spec_path, map_path, review_path = (Path(cfg[k]).resolve() for k in ("spec", "map", "review"))
    controls = [prepared, capture, spec_path, map_path, review_path]
    assert_disjoint_output(out, controls)
    before = input_snapshot(controls)
    sources = source_hashes()
    imported = imported_source_hashes()
    review = json.loads(review_path.read_text())
    if review["verdict"] != "PASS" or review["analysis_sources_sha256"] != sources:
        raise ValueError("Analysis requires a current independent source-bound PASS")
    spec = json.loads(spec_path.read_text())
    if spec["schema_version"] != "reward_hacking_trajectory_stage_analysis_v1":
        raise ValueError("Wrong analysis spec schema")
    if before[str(map_path)] != spec["map_sha256"]:
        raise ValueError("Frozen map hash differs")
    _manifest, streams, checkpoint_rows, observations = load_prepared(prepared)
    prepared_sha = before[str(prepared / "manifest.json")]
    index, vectors = verify_capture(capture, prepared_sha)
    verify_capture_geometry(capture, index, streams, checkpoint_rows)
    checkpoints = {r["checkpoint_key"]: r for r in checkpoint_rows}
    vector_index = {key: i for i, key in enumerate(index["checkpoint_keys"])}
    if len(checkpoints) != len(checkpoint_rows) or set(checkpoints) != set(vector_index):
        raise ValueError("Capture/checkpoint coverage mismatch")
    if len(vector_index) != len(vectors) or vectors.shape[1] != 5120:
        raise ValueError("Capture geometry differs")
    if len(observations) != 2153 * 6:
        raise ValueError("Planned observation coverage differs")
    contexts, text_bank = load_contexts(prepared), load_text_bank(prepared)
    with np.load(map_path, allow_pickle=False) as handle:
        map_arrays = {key: handle[key] for key in ("weight", "x_mean", "x_scale", "y_mean")}
    provenance = {
        "sources_sha256": sources,
        "imported_sources_sha256": imported,
        "input_files_sha256": before,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "hydra-core", "omegaconf")
        },
        "review_sha256": before[str(review_path)],
        "prepared_manifest_sha256": prepared_sha,
        "capture_index_sha256": before[str(capture / "index.json")],
        "spec_sha256": before[str(spec_path)],
        "map_sha256": before[str(map_path)],
    }
    if review["analysis_spec_sha256"] != provenance["spec_sha256"]:
        raise ValueError("Analysis spec differs from reviewed spec")
    if (
        before != input_snapshot(controls)
        or sources != source_hashes()
        or imported != imported_source_hashes()
    ):
        raise ValueError("Input/source changed while analysis inputs were consumed")
    binding = out / "binding.json"
    if binding.exists():
        if json.loads(binding.read_text()) != provenance:
            raise ValueError("Existing analysis belongs to different source/input/review")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Cannot adopt unbound analysis outputs")
        save_json(binding, provenance)
    selected = STAGES if cfg.stage == "all" else [str(cfg.stage)]
    if any(stage not in STAGES for stage in selected):
        raise ValueError("Unplanned stage selector")
    for stage in selected:
        rows = aggregate_stage(
            stage, observations, checkpoints, contexts, text_bank, checkpoint_text
        )
        raw = np.asarray(vectors[[vector_index[r["checkpoint_key"]] for r in rows]])
        fit_stage(stage, rows, raw, map_arrays, spec, out / stage, provenance)
    result = {
        "status": "stage_subset_complete",
        "selected_stages": selected,
        "provenance": provenance,
        "elapsed_seconds": time.monotonic() - started,
    }
    if cfg.stage == "all":
        results = [json.loads((out / stage / "result.json").read_text()) for stage in STAGES]
        intervals, arrays = joint_intervals(results, spec)
        if arrays is not None:
            target = out / "bootstrap_draws.npz"
            with target.with_suffix(".tmp").open("wb") as handle:
                np.savez(handle, **arrays)
            target.with_suffix(".tmp").replace(target)
        result.update(
            status="complete",
            stages=results,
            uncertainty=intervals,
            excluded={"transport_unknown": 1, "structurally_not_assessable": 8},
            verification_passed=(
                all(r["status"] == "complete" for r in results)
                and intervals["status"] == "complete"
            ),
        )
    if (
        before != input_snapshot(controls)
        or sources != source_hashes()
        or imported != imported_source_hashes()
    ):
        raise ValueError("Input/source changed during analysis; final PASS withheld")
    save_json(out / "result.json", result)
    return result


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_trajectory_analyze"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
