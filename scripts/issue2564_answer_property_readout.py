"""Small CPU-only readouts of archived individual answer-token means (#2564).

Training uses draws 0 and 1, selected before looking at readout results; all
ten draws of held-out carrier questions are evaluated. Five condition-label
diagnostics are explicitly separate from the actual-output word-presence readout.
No model generation, judge calls, representation-map fit, or paper edits occur.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic, write_jsonl_atomic  # noqa: E402
from issue2564_judge import check_contains_word  # noqa: E402

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "426a48e589c745d31e8bcc49d7f7178a7d6896f2"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 19
DIMENSION = 3584
TRAIN_DRAWS = (0, 1)
ALPHAS = np.asarray((0.1, 1, 10, 100, 1000, 10000, 100000, 1000000), dtype=float)
CELLS = ("persona", "format", "register", "lexical_marker", "answer_language")
WORDS = ("moreover", "honestly", "surely", "notably", "essentially")


@dataclass
class Config:
    stage_root: str = "/mnt/eps-data/thomasjiralerspong/issue2564_answer_property_readout"
    out_root: str = "eval_results/issue_2564/answer_property_readout_20260907"
    pilot: bool = False
    summarize_only: bool = False
    bootstrap_draws: int = 2000
    seed: int = 2564


@dataclass
class Bank:
    name: str
    x: np.ndarray
    y: np.ndarray
    rows: list[dict]
    classes: list[str]
    label_kind: str
    multilabel: bool = False


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(stage_root: Path, cell: str) -> tuple[Path, Path, Path, Path]:
    if cell == "answer_language":
        root = stage_root / "issue2564_minpair/lang_oneword_pilot"
        return (
            root / "analysis_tensors/va/va_langow_answer_language.pt",
            root / "manifests/pilot_bank.json",
            root / "raw_completions/anchors/anchors_answer_language.jsonl",
            root / "manifests/va_langow_answer_language.done.json",
        )
    root = stage_root / "issue2564_minpair"
    return (
        root / f"analysis_tensors/va2564/va2564_{cell}.pt",
        root / "manifests/bank2564_manifest.json",
        root / f"raw_completions/anchors/anchors_{cell}.jsonl",
        root / f"manifests/va2564_{cell}.done.json",
    )


def load_bank(stage_root: Path, cell: str) -> tuple[Bank, dict, list[str]]:
    tensor_path, manifest_path, text_path, sentinel_path = input_paths(stage_root, cell)
    manifest = json.loads(manifest_path.read_text())
    contexts_raw = manifest["contexts"]
    contexts = (
        {row["id"]: row for row in contexts_raw} if isinstance(contexts_raw, list) else contexts_raw
    )
    anchors = read_jsonl(text_path)
    text_by_key = {(row["context_id"], row["draw"]): row for row in anchors}
    if len(text_by_key) != len(anchors):
        raise ValueError("duplicate completion context/draw keys")
    store = torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=False)
    if store["cell"] != cell or store["empty_rows"]:
        raise ValueError((cell, store["cell"], store["empty_rows"]))
    if store["repro"]["model_id"] != MODEL:
        raise ValueError("wrong source model")
    if store["va_span"].shape != (len(store["index"]), 3, DIMENSION):
        raise ValueError("wrong answer tensor shape")
    li = store["layers"].index(LAYER)
    kept, rows, texts = [], [], []
    for i, index in enumerate(store["index"]):
        context = contexts[index["context_id"]]
        if cell == "answer_language" and context["kind"] == "bare":
            continue
        key = (index["context_id"], index["draw"])
        anchor = text_by_key[key]
        for field in ("ctx_len", "span_start", "span_end", "tail_end"):
            if index[field] != anchor[field]:
                raise ValueError(f"generation/capture boundary mismatch: {key} {field}")
        if index["span_end"] <= index["span_start"]:
            raise ValueError(f"empty answer span: {key}")
        label = context["value_id"]
        if context["kind"] == "para":
            if not label.endswith("p"):
                raise ValueError(f"unexpected paraphrase label: {label}")
            label = label[:-1]
        rows.append(
            {
                "context_id": index["context_id"],
                "draw": index["draw"],
                "carrier": context["carrier"],
                "condition": label,
                "instruction": context["system"],
                "n_answer_tokens": index["span_end"] - index["span_start"],
            }
        )
        kept.append(i)
        texts.append(anchor["text"])
    keys = [(row["context_id"], row["draw"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate activation context/draw keys")
    for cid in {row["context_id"] for row in rows}:
        if {row["draw"] for row in rows if row["context_id"] == cid} != set(range(10)):
            raise ValueError(f"expected exactly ten original draws: {cid}")
    x = store["va_span"][kept, li].float().numpy()
    if not np.isfinite(x).all():
        raise ValueError("nonfinite answer activations")
    classes = sorted({row["condition"] for row in rows})
    y = np.asarray([classes.index(row["condition"]) for row in rows])
    provenance = {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "files": [
            {"path": str(p.relative_to(stage_root)), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (tensor_path, manifest_path, text_path, sentinel_path)
        ],
        "producer": store["repro"],
        "sentinel": json.loads(sentinel_path.read_text()),
        "pooling": "va_span: mean of answer tokens, prompt and end-of-turn tokens excluded",
        "source_cell": cell,
        "n_contexts": len({row["context_id"] for row in rows}),
        "n_answers": len(rows),
        "generation": {
            "temperature": sorted({row["temperature"] for row in anchors}),
            "max_new_tokens": sorted({row["max_new_tokens"] for row in anchors}),
            "draws": 10,
            "regime": "model-generated answers, teacher-forced replay for activation capture",
        },
    }
    return Bank(cell, x, y, rows, classes, "assigned_condition"), provenance, texts


def carrier_folds(groups: np.ndarray, n_splits: int) -> list[np.ndarray]:
    unique = np.unique(groups)
    if not 2 <= n_splits <= len(unique):
        raise ValueError("invalid group fold count")
    return [np.isin(groups, part) for part in np.array_split(unique, n_splits)]


def target_matrix(y: np.ndarray, n_classes: int, multilabel: bool) -> np.ndarray:
    if multilabel:
        return 2 * y.astype(float) - 1
    return 2 * np.eye(n_classes)[y] - 1


def ridge_grid_scores(
    x_train: np.ndarray,
    targets: np.ndarray,
    x_test: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Exact standardized multi-output ridge, one Gram eigendecomposition per fold.

    Same squared-loss estimator as unweighted RidgeClassifier for multiclass
    targets; ±1 multilabel columns are independent ridge readouts. All penalties
    and target columns share the decomposition. Scaling uses training rows only.
    """
    if np.any(alphas <= 0) or not np.isfinite(x_train).all():
        raise ValueError("finite data and strictly positive penalties required")
    x_train, x_test = np.asarray(x_train, float), np.asarray(x_test, float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    target_mean = targets.mean(axis=0)
    centered = targets - target_mean
    gram = train @ train.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-8 * max(1.0, eigenvalues.max()):
        raise ValueError("materially negative Gram eigenvalue")
    # Remove numerical roundoff only; no data-dependent ridge jitter is added.
    eigenvalues = np.maximum(eigenvalues, 0)
    left = (test @ train.T) @ eigenvectors
    right = eigenvectors.T @ centered
    scores = np.einsum("tn,an,nc->atc", left, 1 / (eigenvalues[None, :] + alphas[:, None]), right)
    return scores + target_mean


def primary_metric(y: np.ndarray, scores: np.ndarray, multilabel: bool) -> float:
    if multilabel:
        return float(np.mean([roc_auc_score(y[:, c], scores[:, c]) for c in range(y.shape[1])]))
    return float(balanced_accuracy_score(y, scores.argmax(axis=1)))


def summarize_metrics(bank: Bank, scores: np.ndarray, seed: int, n_boot: int) -> dict:
    y = bank.y
    predicted = scores >= 0 if bank.multilabel else scores.argmax(axis=1)
    binary_y = y if bank.multilabel else np.eye(len(bank.classes), dtype=int)[y]
    groups = np.asarray([row["carrier"] for row in bank.rows])
    carriers = np.unique(groups)
    counts = np.zeros((len(carriers), len(bank.classes), 2, 2))
    for gi, carrier in enumerate(carriers):
        mask = groups == carrier
        for ci in range(len(bank.classes)):
            pred = predicted[mask, ci] if bank.multilabel else predicted[mask] == ci
            actual = binary_y[mask, ci]
            counts[gi, ci] = np.bincount(2 * actual + pred, minlength=4).reshape(2, 2)
    # One vectorized carrier-cluster bootstrap; all draws/paraphrases stay together.
    draws = np.random.default_rng(seed).integers(0, len(carriers), (n_boot, len(carriers)))
    multiplicities = (draws[:, :, None] == np.arange(len(carriers))).sum(axis=1)
    aggregate = np.einsum("bg,gcij->bcij", multiplicities, counts)
    positive = aggregate[:, :, 1, 1] / aggregate[:, :, 1, :].sum(axis=2)
    if bank.multilabel:
        negative = aggregate[:, :, 0, 0] / aggregate[:, :, 0, :].sum(axis=2)
        bootstrap_ba = (positive + negative).mean(axis=1) / 2
        ba = float(
            np.mean([balanced_accuracy_score(y[:, c], predicted[:, c]) for c in range(y.shape[1])])
        )
    else:
        bootstrap_ba = positive.mean(axis=1)
        ba = float(balanced_accuracy_score(y, predicted))
    if not np.isfinite(bootstrap_ba).all():
        raise ValueError("a carrier bootstrap lost a class; revise target/fold design")
    aucs = [roc_auc_score(binary_y[:, ci], scores[:, ci]) for ci in range(len(bank.classes))]
    return {
        "balanced_accuracy": ba,
        "balanced_accuracy_ci95": np.quantile(bootstrap_ba, [0.025, 0.975]).tolist(),
        "chance_balanced_accuracy": 0.5 if bank.multilabel else 1 / len(bank.classes),
        "macro_ovr_auroc": float(np.mean(aucs)),
        "chance_auroc": 0.5,
        "per_class_auroc": dict(zip(bank.classes, map(float, aucs), strict=True)),
        "positive_prevalence": dict(zip(bank.classes, binary_y.mean(axis=0).tolist(), strict=True)),
        "n_carriers": len(carriers),
        "n_contexts": len({row["context_id"] for row in bank.rows}),
        "n_answers": len(bank.rows),
        "uncertainty": f"percentile bootstrap of {len(carriers)} carrier clusters; exploratory precision",
        "bootstrap_draws": n_boot,
    }


def evaluated_rows(scores: np.ndarray, assignments: np.ndarray, *, pilot: bool) -> np.ndarray:
    """Require one finite prediction per production row; pilots may leave rows unassigned."""
    if assignments.shape != (len(scores),) or not np.isin(assignments, (0, 1)).all():
        raise ValueError("each answer must be assigned to at most one evaluation fold")
    assigned = assignments == 1
    if not assigned.any() or (not pilot and not assigned.all()):
        raise ValueError("production evaluation must assign every input answer exactly once")
    if not np.isfinite(scores[assigned]).all():
        raise ValueError("nonfinite prediction on an assigned evaluation answer")
    if not np.isnan(scores[~assigned]).all():
        raise ValueError("unassigned pilot rows must retain their NaN sentinel")
    return assigned


def evaluate_bank(bank: Bank, cfg: Config, out: Path) -> dict:
    groups = np.asarray([row["carrier"] for row in bank.rows])
    selected_draws = np.isin([row["draw"] for row in bank.rows], TRAIN_DRAWS)
    all_targets = target_matrix(bank.y, len(bank.classes), bank.multilabel)
    outer = carrier_folds(groups, 6)
    scores = np.full((len(bank.rows), len(bank.classes)), np.nan)
    assignments = np.zeros(len(bank.rows), dtype=int)
    folds = []
    started = time.monotonic()
    for fold_i, test_mask in enumerate(outer[:1] if cfg.pilot else outer):
        fold_started = time.monotonic()
        train_indices = np.flatnonzero(~test_mask & selected_draws)
        if set(groups[train_indices]) & set(groups[test_mask]):
            raise ValueError("carrier leakage")
        train_targets = all_targets[train_indices]
        inner_metrics = []
        for inner_test in carrier_folds(groups[train_indices], 3):
            inner_train_indices = train_indices[~inner_test]
            inner_test_indices = train_indices[inner_test]
            trial = ridge_grid_scores(
                bank.x[inner_train_indices],
                all_targets[inner_train_indices],
                bank.x[inner_test_indices],
                ALPHAS,
            )
            inner_metrics.append(
                [primary_metric(bank.y[inner_test_indices], s, bank.multilabel) for s in trial]
            )
        mean_metrics = np.mean(inner_metrics, axis=0)
        # Prefer stronger regularization on exact metric ties, predetermined.
        alpha_index = int(np.flatnonzero(mean_metrics == mean_metrics.max())[-1])
        alpha = float(ALPHAS[alpha_index])
        prediction = ridge_grid_scores(
            bank.x[train_indices], train_targets, bank.x[test_mask], np.asarray([alpha])
        )[0]
        scores[test_mask] = prediction
        assignments[test_mask] += 1
        fold = {
            "fold": fold_i,
            "test_carriers": sorted(set(groups[test_mask])),
            "n_train": len(train_indices),
            "dimension": DIMENSION,
            "n_test": int(test_mask.sum()),
            "alpha": alpha,
            "alpha_selection": "three training-only carrier-group folds; strongest exact tie",
            "inner_metric": "macro_ovr_auroc" if bank.multilabel else "balanced_accuracy",
            "inner_mean_metrics": mean_metrics.tolist(),
            "test_primary_metric": primary_metric(bank.y[test_mask], prediction, bank.multilabel),
            "wall_seconds": time.monotonic() - fold_started,
        }
        folds.append(fold)
        write_json_atomic(out / f"{bank.name}_fold{fold_i}.json", fold)
        print(
            f"{bank.name}: fold {fold_i + 1}/6 completed in {fold['wall_seconds']:.2f}s", flush=True
        )
    complete = evaluated_rows(scores, assignments, pilot=cfg.pilot)
    eval_bank = Bank(
        bank.name,
        bank.x[complete],
        bank.y[complete],
        [row for row, keep in zip(bank.rows, complete, strict=True) if keep],
        bank.classes,
        bank.label_kind,
        bank.multilabel,
    )
    result = {
        "panel": bank.name,
        "label_kind": bank.label_kind,
        "classes": bank.classes,
        "training_draws": list(TRAIN_DRAWS),
        "evaluation_draws": list(range(10)),
        "folds": folds,
        "pilot_only": cfg.pilot,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "metrics": summarize_metrics(eval_bank, scores[complete], cfg.seed, cfg.bootstrap_draws),
    }
    prediction_rows = [
        {**row, "target": np.asarray(target).tolist(), "scores": score.tolist()}
        for row, target, score in zip(eval_bank.rows, eval_bank.y, scores[complete], strict=True)
    ]
    write_jsonl_atomic(out / f"{bank.name}_predictions.jsonl", prediction_rows)
    write_json_atomic(out / f"{bank.name}.json", result)
    return result


def instruction_marker_targets(rows: list[dict]) -> np.ndarray:
    """No-fit baseline: requested marker present; other markers absent."""
    conditions = [row["condition"] for row in rows]
    if not set(conditions) <= {f"v{i + 1}" for i in range(len(WORDS))}:
        raise ValueError("unknown lexical instruction condition")
    for row in rows:
        word = WORDS[int(row["condition"][1:]) - 1]
        if not check_contains_word(row["instruction"], word):
            raise ValueError("condition/marker mapping disagrees with source instruction")
    return np.eye(len(WORDS), dtype=int)[[int(value[1:]) - 1 for value in conditions]]


def summarize_existing_results(out: Path) -> None:
    """Produce transparent no-fit diagnostics from the immutable OOF predictions."""
    summary = json.loads((out / "summary.json").read_text())
    if summary["pilot_only"] or len(summary["panels"]) != 6:
        raise ValueError("expected the completed six-panel run")
    foldwise_auc, inputs = {}, []
    for panel in summary["panels"]:
        prediction_path = out / f"{panel['panel']}_predictions.jsonl"
        rows = read_jsonl(prediction_path)
        inputs.append({"path": prediction_path.name, "sha256": sha256(prediction_path)})
        targets = np.asarray([row["target"] for row in rows])
        scores = np.asarray([row["scores"] for row in rows])
        binary_y = (
            targets if targets.ndim == 2 else np.eye(len(panel["classes"]), dtype=int)[targets]
        )
        groups = np.asarray([row["carrier"] for row in rows])
        values, covered = [], np.zeros(len(rows), int)
        for fold in panel["folds"]:
            mask = np.isin(groups, fold["test_carriers"])
            covered += mask
            values.append(
                float(
                    np.mean(
                        [
                            roc_auc_score(binary_y[mask, c], scores[mask, c])
                            for c in range(binary_y.shape[1])
                        ]
                    )
                )
            )
        if not np.all(covered == 1) or len(values) != 6:
            raise ValueError("OOF rows must be covered exactly once by six folds")
        foldwise_auc[panel["panel"]] = {"folds": values, "unweighted_mean": float(np.mean(values))}
    rows = read_jsonl(out / "observed_marker_presence_predictions.jsonl")
    targets = np.asarray([row["target"] for row in rows])
    baseline = instruction_marker_targets(rows)
    bank = Bank(
        "instruction_only_marker_baseline",
        np.empty((len(rows), 0)),
        targets,
        rows,
        list(WORDS),
        "actual_output_whole_word_presence",
        True,
    )
    metrics = summarize_metrics(bank, 2 * baseline - 1, summary["seed"], 2000)
    diagnostics = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "kind": "post-fit descriptive analysis; no additional fits",
        "summary_sha256": sha256(out / "summary.json"),
        "prediction_inputs": inputs,
        "foldwise_auroc": foldwise_auc,
        "instruction_only_marker_baseline": {
            "rule": "one-hot requested word present, other four absent; no activation access or fitting",
            "metrics": metrics,
            "n_answers_with_any_label_mismatch": int((targets != baseline).any(axis=1).sum()),
            "n_binary_label_mismatches": int((targets != baseline).sum()),
        },
    }
    write_json_atomic(out / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, indent=2), flush=True)


ConfigStore.instance().store(name="answer_property_readout", node=Config)


@hydra.main(version_base=None, config_path=None, config_name="answer_property_readout")
def main(cfg: Config) -> None:
    out = Path(cfg.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if cfg.summarize_only:
        summarize_existing_results(out)
        return
    if (out / "summary.json").exists():
        raise FileExistsError("use a fresh output directory; completed runs are immutable")
    results, sources = [], {}
    for cell in CELLS[:1] if cfg.pilot else CELLS:
        bank, sources[cell], texts = load_bank(Path(cfg.stage_root), cell)
        results.append(evaluate_bank(bank, cfg, out))
        if cell == "lexical_marker" and not cfg.pilot:
            targets = np.asarray(
                [[check_contains_word(text, word) for word in WORDS] for text in texts], int
            )
            observed = Bank(
                "observed_marker_presence",
                bank.x,
                targets,
                bank.rows,
                list(WORDS),
                "actual_output_whole_word_presence",
                True,
            )
            results.append(evaluate_bank(observed, cfg, out))
    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "layer": LAYER,
        "hf_revision": HF_REVISION,
        "pilot_only": cfg.pilot,
        "alphas": ALPHAS.tolist(),
        "seed": cfg.seed,
        "fit": "standardized, unweighted multi-output ridge classifier; exact dual eigensolve",
        "underdetermined_justification": "n_train < d is deliberate regularized classification; no reconstruction R2 or absence claims",
        "scope": "five assigned-condition diagnostics and one actual-output lexical readout; no topic, map or SAE-class comparison",
        "sources": sources,
        "panels": results,
    }
    write_json_atomic(out / "summary.json", summary)
    print(f"Completed {len(results)} panel(s): {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
