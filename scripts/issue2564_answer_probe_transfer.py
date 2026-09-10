"""Apply unchanged answer probes to matched out-of-fold answer predictions."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402

import issue2564_answer_behavior_readout as source  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)
from issue2564_matched_answer_map import fit_grouped_oof  # noqa: E402

# LFS SHA256 verified in three_pass_archive_file_hashes.json at archive
# revision 7640e8b6a4656d42e488f938f7ff57ca04b08729.
PREVIOUS_OOF_SHA256 = "e980db5a98eb07789f9c808084a895cf6e3f17d305a245850897dd30c4fc1ee0"


@dataclass
class Config:
    """Reuse the approved bank and exact saved probes with five matched maps."""

    root: str = "/home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907"
    previous: str = "readout_three_pass_20260909"
    output: str = "matched_probe_transfer_20260909"
    bootstrap_draws: int = 2000
    seed: int = 2564


def summarized(values: np.ndarray) -> dict:
    """Keep the point estimate and the number of defined paired resamples."""
    return {
        "value": float(values[0]) if np.isfinite(values[0]) else None,
        **source.finite_summary(values[1:]),
    }


def transfer_statistics(prior: np.ndarray, observed: np.ndarray, mapped: np.ndarray) -> dict:
    """Compare paired errors without clipping ratios or hiding weak denominators."""
    if not (prior.shape == observed.shape == mapped.shape) or prior.ndim != 1:
        raise ValueError("paired error arrays must have the same one-dimensional shape")
    gain = prior - observed
    retained = source.divide(prior - mapped, gain)
    return {
        "retained_error_reduction": retained,
        "lost_error_reduction": 1 - retained,
        "excess_error": mapped - observed,
        "observed_error_reduction": gain,
    }


def validate_fixed_regime(cfg: Config, prior_manifest: dict) -> None:
    """Enforce the amendment's fixed bootstrap and inherited seed."""
    if cfg.bootstrap_draws != 2000:
        raise ValueError("bootstrap_draws must be exactly 2000")
    if cfg.seed != prior_manifest["regime"]["seed"]:
        raise ValueError("bootstrap seed must match the approved prior analysis")


def validate_reference_oof(
    reference: dict[str, np.ndarray], rows: list[dict], targets: list[source.Target]
) -> None:
    """Validate the required prior/readout OOF arrays before they are reused."""
    n_rows = len(rows)
    arms = ("prior", "framing", "context", "length", "answer")
    missing = [
        f"{target.name}__{arm}"
        for target in targets
        for arm in arms
        if f"{target.name}__{arm}" not in reference
    ]
    if missing:
        raise ValueError(f"saved OOF is missing required arrays: {missing[:4]}")
    for target in targets:
        expected_shape = (n_rows, target.values.shape[1])
        available = target.available
        for arm in arms:
            name = f"{target.name}__{arm}"
            values = np.asarray(reference[name])
            if values.shape != expected_shape:
                raise ValueError(f"saved OOF shape mismatch: {name}: {values.shape}")
            try:
                finite_available = np.isfinite(values[available]).all()
                finite_unavailable = np.isfinite(values[~available]).any()
            except TypeError as exc:
                raise ValueError(f"saved OOF is nonnumeric: {name}") from exc
            if not finite_available:
                raise ValueError(f"saved OOF has nonfinite available values: {name}")
            if finite_unavailable:
                raise ValueError(f"saved OOF has values on unavailable rows: {name}")


def validate_replay_block(
    block: dict[str, np.ndarray],
    details: dict,
    rows: list[dict],
    evaluation_vectors: dict[str, np.ndarray],
    targets: list[source.Target],
    reference: dict[str, np.ndarray],
) -> None:
    """Validate a fresh or resumed fold block without refitting it."""
    if not isinstance(details.get("fold"), int):
        raise ValueError("replay checkpoint fold metadata mismatch")
    available = targets[0].available
    if not all(np.array_equal(target.available, available) for target in targets):
        raise ValueError("mixed availability masks in replay checkpoint")
    fold = details["fold"]
    outer = np.array([row["fold"] for row in rows])
    expected_test = np.flatnonzero(available & (outer == fold))
    test = np.asarray(block.get("test"))
    if (
        test.ndim != 1
        or not np.issubdtype(test.dtype, np.integer)
        or not np.array_equal(test, expected_test)
    ):
        raise ValueError("replay checkpoint test indices differ from frozen split")
    expected_ids = [rows[i]["id"] for i in expected_test]
    if details.get("test_ids") != expected_ids:
        raise ValueError("replay checkpoint test IDs differ from frozen split")
    train = np.flatnonzero(available & (outer != fold))
    if details.get("train_ids") != [rows[i]["id"] for i in train]:
        raise ValueError("replay checkpoint training IDs differ from frozen split")
    expected_targets = [target.name for target in targets]
    if details.get("targets") != expected_targets:
        raise ValueError("replay checkpoint target order mismatch")
    for target in targets:
        expected_shape = (len(expected_test), target.values.shape[1])
        for arm in evaluation_vectors:
            name = f"{target.name}__{arm}"
            if name not in block:
                raise ValueError(f"replay checkpoint missing array: {name}")
            values = np.asarray(block[name])
            if values.shape != expected_shape:
                raise ValueError(f"replay checkpoint shape mismatch: {name}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"replay checkpoint has nonfinite values: {name}")
        actual = np.asarray(block[f"{target.name}__answer"])
        expected = np.asarray(reference[f"{target.name}__answer"])[expected_test]
        if not np.allclose(actual, expected, rtol=0, atol=1e-8):
            raise ValueError(f"replay checkpoint observed-probe parity failed: {target.name}")
        expected_prior = np.broadcast_to(
            target.values[train].mean(0), (len(expected_test), target.values.shape[1])
        )
        saved_prior = np.asarray(reference[f"{target.name}__prior"])[expected_test]
        if not np.allclose(saved_prior, expected_prior, rtol=0, atol=1e-10):
            raise ValueError(f"saved training-prior parity failed: {target.name}")


def reconstruct_bundle(
    rows: list[dict],
    observed_vectors: np.ndarray,
    evaluation_vectors: dict[str, np.ndarray],
    targets: list[source.Target],
    saved: dict,
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    """One observed-training fit evaluates all arms with identical normalization."""
    available = targets[0].available
    if not all(np.array_equal(t.available, available) for t in targets):
        raise ValueError("mixed availability masks in probe bundle")
    fold = saved["fold"]
    outer = np.array([r["fold"] for r in rows])
    train = np.flatnonzero(available & (outer != fold))
    test = np.flatnonzero(available & (outer == fold))
    if saved["train_ids"] != [rows[i]["id"] for i in train]:
        raise ValueError("saved probe training IDs differ from frozen split")
    if saved["test_ids"] != [rows[i]["id"] for i in test]:
        raise ValueError("saved probe evaluation IDs differ from frozen split")
    if {rows[i]["question_group"] for i in train} & {rows[i]["question_group"] for i in test}:
        raise ValueError("connected-group leakage")
    if "answer" not in evaluation_vectors:
        raise ValueError("observed-answer parity arm is required")
    if not np.array_equal(evaluation_vectors["answer"], observed_vectors):
        raise ValueError("parity arm differs from the actual observed vectors")
    if any(x.shape != observed_vectors.shape for x in evaluation_vectors.values()):
        raise ValueError("evaluation vector shapes differ")
    values = np.concatenate([t.values for t in targets], axis=1)
    choices = {t.name: saved["models"]["answer"][t.name]["alpha"] for t in targets}
    alphas = np.array(sorted(set(choices.values())), dtype=float)
    arms = list(evaluation_vectors)
    # Concatenation is evaluation-only: the solver fits its scaler and weights
    # using observed_vectors[train] and values[train], once for all arms.
    evaluation = np.concatenate([evaluation_vectors[arm][test] for arm in arms])
    grid = source.ridge_width_grid(
        observed_vectors[train], values[train], evaluation, alphas, (0,)
    )[0]
    result = {"test": test}
    errors = {}
    boundaries = np.cumsum([0] + [t.values.shape[1] for t in targets])
    for j, target in enumerate(targets):
        segment = slice(boundaries[j], boundaries[j + 1])
        choice = int(np.flatnonzero(alphas == choices[target.name])[0])
        for ai, arm in enumerate(arms):
            result[f"{target.name}__{arm}"] = grid[
                choice, ai * len(test) : (ai + 1) * len(test), segment
            ]
        actual = result[f"{target.name}__answer"]
        expected = reference[f"{target.name}__answer"][test]
        expected_prior = np.broadcast_to(values[train, segment].mean(0), expected.shape)
        if not np.allclose(
            reference[f"{target.name}__prior"][test], expected_prior, rtol=0, atol=1e-10
        ):
            raise ValueError(f"saved training-prior parity failed: {target.name}")
        errors[target.name] = float(np.abs(actual - expected).max())
        # The prior solver passed independent full/PCA reference tests at 1e-9.
        # This 1e-8 replay tolerance permits only floating-point reassociation.
        if errors[target.name] > 1e-8 or not np.isfinite(actual).all():
            raise ValueError(f"saved observed-probe parity failed: {target.name}: {errors}")
    return result, {
        "fold": fold,
        "train_ids": saved["train_ids"],
        "test_ids": saved["test_ids"],
        "targets": [t.name for t in targets],
        "selected_alphas": choices,
        "observed_replay_max_abs_error": errors,
        "normalization": "same observed-training mean/std for every evaluation arm",
    }


def output_agreement(
    target: source.Target,
    predictions: dict[str, np.ndarray],
    group_counts: np.ndarray,
    inverse: np.ndarray,
) -> dict[str, np.ndarray]:
    """Descriptive reconstruction of probe outputs, separate from label accuracy."""
    keep = np.flatnonzero(target.available)
    weights = np.vstack([np.ones(len(keep)), group_counts[:, inverse[keep]]])
    observed = predictions["answer"][keep]
    mapped = predictions["mapped"][keep]
    # Vector-valued categorical outputs use trace variance, preserving columns.
    centered = observed - observed.mean(0)
    count = weights.sum(1)
    sum_y = weights @ centered
    variance_sum = weights @ np.square(centered).sum(1) - source.divide(
        np.square(sum_y).sum(1), count
    )
    sse = weights @ np.square(observed - mapped).sum(1)
    return {"r2": 1 - source.divide(sse, np.maximum(variance_sum, 0))}


def run(cfg: Config) -> None:
    """Replay saved probes, checkpoint each fold and compute paired transfer metrics."""
    started = time.monotonic()
    root, previous = Path(cfg.root), Path(cfg.root) / cfg.previous
    out = root / cfg.output
    out.mkdir(parents=True, exist_ok=True)
    rows, vectors, targets, provenance = source.load_inputs(
        source.Config(root=cfg.root, labels="annotation_codex_three_pass/main/labels.json")
    )
    prior_manifest = json.loads((previous / "manifest.json").read_text())
    prior_complete = json.loads((previous / "analysis_complete.json").read_text())
    prior_summary = json.loads((previous / "summary.json").read_text())
    if (
        prior_complete["summary_sha256"] != source.file_hash(previous / "summary.json")
        or prior_complete["fingerprint"] != prior_manifest["fingerprint"]
        or prior_summary["fingerprint"] != prior_manifest["fingerprint"]
        or provenance != prior_manifest["regime"]["inputs"]
        or source.file_hash(Path(source.__file__)) != prior_manifest["regime"]["script_sha256"]
    ):
        raise ValueError("previous completed readout or its inputs changed")
    validate_fixed_regime(cfg, prior_manifest)
    if source.file_hash(previous / "oof_predictions.npz") != PREVIOUS_OOF_SHA256:
        raise ValueError("previous OOF differs from its verified immutable archive")
    with np.load(previous / "oof_predictions.npz", allow_pickle=False) as stored:
        if list(stored["id"]) != [r["id"] for r in rows]:
            raise ValueError("saved OOF row order differs")
        reference = {key: stored[key] for key in stored.files if key != "id"}
    validate_reference_oof(reference, rows, targets)
    maps = fit_grouped_oof(rows, vectors["context"], vectors["answer"], out / "maps")
    map_provenance = maps["manifest"]
    evaluations = {
        "answer": vectors["answer"],
        **{arm: maps[arm] for arm in ("mapped", "identity_bias")},
    }
    regime = {
        "config": asdict(cfg),
        "inputs": provenance,
        "previous_summary_sha256": prior_complete["summary_sha256"],
        "previous_oof_sha256": source.file_hash(previous / "oof_predictions.npz"),
        "map": map_provenance,
        "script_sha256": source.file_hash(Path(__file__)),
        "matched_map_script_sha256": source.file_hash(
            Path(__file__).with_name("issue2564_matched_answer_map.py")
        ),
        "producer_script_sha256": source.file_hash(Path(__file__).with_name("issue2054_fits.py")),
    }
    key = source.fingerprint(regime)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != key:
        raise ValueError("output directory belongs to another analysis regime")
    write_json_atomic(manifest_path, {"fingerprint": key, "regime": regime})
    np.savez(out / "predicted_vectors.npz", id=np.array([r["id"] for r in rows]), **evaluations)
    predictions = {
        t.name: {
            **{
                arm: reference[f"{t.name}__{arm}"]
                for arm in ("prior", "framing", "context", "length")
            },
            **{arm: np.full(t.values.shape, np.nan) for arm in evaluations},
        }
        for t in targets
    }
    by_name = {t.name: t for t in targets}
    assignments = {t.name: np.zeros(len(rows), int) for t in targets}
    fit_complete = json.loads((previous / "fits_complete.json").read_text())
    if fit_complete["fingerprint"] != prior_manifest["fingerprint"] or fit_complete["folds"] != 5:
        raise ValueError("previous fit completion mismatch")
    replay = []
    for fold in range(5):
        for bi in range(fit_complete["bundles"]):
            tick = time.monotonic()
            old = previous / f"fold{fold}_bundle{bi}"
            meta = json.loads(old.with_suffix(".json").read_text())
            if (
                meta["fingerprint"] != prior_manifest["fingerprint"]
                or source.file_hash(old.with_suffix(".npz")) != meta["tensor_sha256"]
            ):
                raise ValueError("previous probe checkpoint changed")
            bundle = [by_name[n] for n in meta["targets"] if not n.startswith("shuffle__")]
            dest = out / old.name
            meta_path, array_path = dest.with_suffix(".json"), dest.with_suffix(".npz")
            if meta_path.exists():
                details = json.loads(meta_path.read_text())
                if (
                    details["fingerprint"] != key
                    or details["old_meta_sha256"] != source.file_hash(old.with_suffix(".json"))
                    or source.file_hash(array_path) != details["tensor_sha256"]
                ):
                    raise ValueError("replay checkpoint is stale or corrupt")
                with np.load(array_path, allow_pickle=False) as stored:
                    block = {n: stored[n] for n in stored.files}
            else:
                block, details = reconstruct_bundle(
                    rows, vectors["answer"], evaluations, bundle, meta, reference
                )
                np.savez(array_path, **block)
                details.update(
                    fingerprint=key,
                    old_meta_sha256=source.file_hash(old.with_suffix(".json")),
                    tensor_sha256=source.file_hash(array_path),
                    elapsed_seconds=time.monotonic() - tick,
                )
                write_json_atomic(meta_path, details)
            if details["fold"] != fold:
                raise ValueError("replay checkpoint fold differs from source bundle")
            validate_replay_block(block, details, rows, evaluations, bundle, reference)
            idx = block["test"]
            for target in bundle:
                assignments[target.name][idx] += 1
                for arm in evaluations:
                    predictions[target.name][arm][idx] = block[f"{target.name}__{arm}"]
            replay.append(details)
            print(
                f"[transfer] fold={fold + 1}/5 bundle={bi + 1}/{fit_complete['bundles']} seconds={time.monotonic() - tick:.2f}",
                flush=True,
            )
    for target in targets:
        if not np.array_equal(assignments[target.name], target.available.astype(int)):
            raise ValueError("OOF assignment is not exactly once for each available target")
    np.savez(
        out / "oof_predictions.npz",
        id=np.array([r["id"] for r in rows]),
        **{f"{n}__{arm}": x for n, arms in predictions.items() for arm, x in arms.items()},
    )
    groups, inverse = np.unique([r["question_group"] for r in rows], return_inverse=True)
    counts = np.random.default_rng(cfg.seed).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=cfg.bootstrap_draws
    )
    summary = {
        "fingerprint": key,
        "metadata": as_metadata_dict(git_provenance(), phase="same-probe-transfer"),
        "n_answers": len(rows),
        "n_connected_groups": len(groups),
        "bootstrap_draws": cfg.bootstrap_draws,
        "retention_definition": "(E_prior-E_mapped)/(E_prior-E_observed); same held-out labels and same fixed observed-answer probe; nonpositive denominators undefined; no clipping",
        "uncertainty": "Paired connected-component bootstrap of fixed OOF predictions; conditional on these fold maps, probes and correlated three-pass labels; no refitting or independent-rater uncertainty",
        "map": map_provenance,
        "probe_replay": replay,
        "properties": {},
    }
    draws_store = {}
    for target in targets:
        metrics, draws = source.score_target(
            target, predictions[target.name], rows, counts, inverse, np.ones(len(rows), bool)
        )
        error = "vote_mse" if target.kind == "categorical" else "mse"
        comparisons = {}
        for arm in ("mapped", "identity_bias"):
            stats = transfer_statistics(
                draws["prior"][error], draws["answer"][error], draws[arm][error]
            )
            comparisons[arm] = {name: summarized(values) for name, values in stats.items()}
            comparisons[arm]["nonpositive_denominator_draws"] = int(
                ((draws["prior"][error] - draws["answer"][error])[1:] <= 0).sum()
            )
            for name, values in stats.items():
                draws_store[f"{target.name}__{arm}__{name}"] = values
        agreement = output_agreement(target, predictions[target.name], counts, inverse)
        metrics["transfer"] = comparisons
        metrics["mapped_vs_observed_probe_outputs"] = {
            n: summarized(v) for n, v in agreement.items()
        }
        summary["properties"][target.name] = metrics
        for arm, arm_draws in draws.items():
            for name, values in arm_draws.items():
                draws_store[f"{target.name}__{arm}__{name}"] = values
        print(f"[transfer-summary] target={target.name} finished", flush=True)
    mean_answer = vectors["answer"].mean(0)
    denominator = np.square(vectors["answer"] - mean_answer).sum()
    summary["representation_diagnostics"] = {}
    for arm, x in evaluations.items():
        if arm == "answer":
            continue
        retrieval = {}
        for metric in ("euclidean", "cosine"):
            folds = []
            for fold in range(5):
                test = np.array([i for i, row in enumerate(rows) if row["fold"] == fold])
                result = knn_retrieval(x[test], vectors["answer"][test], ks=(1,), metric=metric)
                folds.append({"fold": fold, **result})
            retrieval[metric] = {
                "folds": folds,
                "query_weighted_top1": sum(r["n"] * r["acc_at_k"][1] for r in folds) / len(rows),
                "query_weighted_chance": sum(r["n"] * r["chance_at_k"][1] for r in folds)
                / len(rows),
                "pool_definition": "same held-out fold only; excludes every map-training answer",
                "tie_rule": "established knn_retrieval tolerance-based mid-rank",
            }
        summary["representation_diagnostics"][arm] = {
            "trace_r2": float(1 - np.square(x - vectors["answer"]).sum() / denominator),
            "retrieval": retrieval,
        }
    summary["elapsed_seconds"] = time.monotonic() - started
    np.savez(out / "bootstrap_metric_draws.npz", **draws_store)
    write_json_atomic(out / "summary.json", summary)
    write_json_atomic(
        out / "analysis_complete.json",
        {
            "fingerprint": key,
            "summary_sha256": source.file_hash(out / "summary.json"),
            "oof_sha256": source.file_hash(out / "oof_predictions.npz"),
            "vectors_sha256": source.file_hash(out / "predicted_vectors.npz"),
            "bootstrap_sha256": source.file_hash(out / "bootstrap_metric_draws.npz"),
            "properties": len(targets),
        },
    )
    print(f"[transfer-complete] seconds={summary['elapsed_seconds']:.2f}", flush=True)


ConfigStore.instance().store(name="answer_probe_transfer", node=Config)


@hydra.main(version_base=None, config_name="answer_probe_transfer")
def main(cfg: Config) -> None:
    """Execute the user-approved analysis-only continuation."""
    run(Config(**cfg))


if __name__ == "__main__":
    main()
