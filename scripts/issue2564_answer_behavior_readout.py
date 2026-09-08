"""Matched readouts of actual expressed-answer labels for the #2564 follow-up.

The collector produces labels separately. This program never calls a model or
uses source-framing identities as target labels. Full-width ridge is primary;
PCA variants and a single grouped shuffle are declared diagnostics. One shared
Gram decomposition serves all targets with the same availability mask, all
penalties, and the three representation widths. The implementation extends
issue2564_answer_property_readout.ridge_grid_scores to share that decomposition
across widths; focused tests compare the live function to that existing helper
and to scikit-learn. It does not fit a context-to-answer representation map.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from itertools import pairwise  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from hydra.core.config_store import ConfigStore  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace, write_json_atomic  # noqa: E402
from issue2564_answer_behavior import (  # noqa: E402
    PROPERTIES,
    digest as annotation_digest,
    judge_system,
    schema,
    validate_pilot_acceptance,
)
from issue2564_answer_property_readout import ALPHAS  # noqa: E402
from issue2564_codex_judgments import (  # noqa: E402
    PROVIDER as CODEX_PROVIDER,
    validate_codex_main,
)


@dataclass
class Config:
    """Frozen defaults for the authorized, CPU-only expression readout."""

    root: str = "/home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907"
    labels: str = "annotation_codex/main/labels.json"
    output: str = "readout_codex"
    stage: str = "fit"
    bootstrap_draws: int = 2000
    seed: int = 2564
    first_fold_only: bool = False


@dataclass
class Target:
    """A target matrix, explicit availability, and optional unique-mode labels."""

    name: str
    kind: str
    values: np.ndarray
    available: np.ndarray
    classes: list[str]
    modal: np.ndarray | None = None


def file_hash(path: Path) -> str:
    """Hash an existing immutable input without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    """Hash generating parameters, never recomputed floating-point arrays."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def grouped_folds(groups: np.ndarray, count: int = 3) -> np.ndarray:
    """Deterministically balance whole connected groups across inner folds."""
    unique, sizes = np.unique(groups, return_counts=True)
    if len(unique) < count:
        raise ValueError("too few independent groups for the declared inner folds")
    loads = np.zeros(count, int)
    mapping = {}
    for group, size in sorted(zip(unique, sizes, strict=True), key=lambda p: (-p[1], p[0])):
        fold = int(np.argmin(loads))
        mapping[group] = fold
        loads[fold] += size
    return np.array([mapping[group] for group in groups])


def ridge_width_grid(
    x_train: np.ndarray,
    targets: np.ndarray,
    x_test: np.ndarray,
    alphas: np.ndarray,
    widths: tuple[int, ...] = (0, 256, 512),
) -> dict[int, np.ndarray]:
    """Standardized ridge, with train-only PCA, sharing one dual factorization.

    Width zero retains all dimensions. A PCA width must be attainable from this
    training split; silently substituting a smaller dimension is forbidden.
    """
    x_train, x_test, targets = (np.asarray(a, float) for a in (x_train, x_test, targets))
    if x_train.ndim != 2 or targets.ndim != 2 or len(x_train) != len(targets):
        raise ValueError("invalid ridge boundary shapes")
    if x_test.ndim != 2 or x_test.shape[1] != x_train.shape[1]:
        raise ValueError("train/test feature width mismatch")
    if not all(np.isfinite(a).all() for a in (x_train, x_test, targets, alphas)):
        raise ValueError("nonfinite ridge input")
    if np.any(alphas <= 0) or any(w < 0 for w in widths):
        raise ValueError("positive penalties and nonnegative widths required")
    if any(w > min(len(x_train) - 1, x_train.shape[1]) for w in widths):
        raise ValueError("requested PCA width exceeds available training dimensions")
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1
    train, test = (x_train - mean) / scale, (x_test - mean) / scale
    ymean = targets.mean(0)
    primal = train.shape[1] < train.shape[0]
    gram = train.T @ train if primal else train @ train.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() < -1e-8 * max(1.0, eigenvalues.max()):
        raise ValueError("materially negative Gram eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0)
    left = test @ eigenvectors if primal else (test @ train.T) @ eigenvectors
    right = eigenvectors.T @ (train.T @ (targets - ymean) if primal else targets - ymean)
    result = {}
    for width in widths:
        section = slice(-width, None) if width else slice(None)
        result[width] = (
            np.einsum(
                "tn,an,nc->atc",
                left[:, section],
                1 / (eigenvalues[None, section] + alphas[:, None]),
                right[section],
                optimize=True,
            )
            + ymean
        )
    return result


def component_shuffle(rows: list[dict], available: np.ndarray, seed: int) -> np.ndarray:
    """Permute whole equal-shape group bundles, preserving framing and masks.

    This is one diagnostic shuffle, not a permutation significance test. Components
    with no exchangeable partner remain fixed and their fraction is reported.
    """
    by_group: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_group.setdefault(row["question_group"], []).append(i)
    strata: dict[tuple, list[np.ndarray]] = {}
    for members in by_group.values():
        ordered = np.array(sorted(members, key=lambda i: (rows[i]["conv_id"], rows[i]["variant"])))
        signature = tuple((rows[i]["variant"], bool(available[i])) for i in ordered)
        strata.setdefault(signature, []).append(ordered)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows))
    for signature in sorted(strata):
        bundles = strata[signature]
        for source, destination in zip(rng.permutation(len(bundles)), bundles, strict=True):
            permutation[destination] = bundles[source]
    if not np.array_equal(np.sort(permutation), np.arange(len(rows))):
        raise ValueError("shuffle is not a permutation")
    if not np.array_equal(available, available[permutation]):
        raise ValueError("shuffle changed target availability")
    if any(rows[i]["variant"] != rows[j]["variant"] for i, j in enumerate(permutation)):
        raise ValueError("shuffle changed source-framing blocks")
    return permutation


def shuffle_diagnostics(rows: list[dict], target: Target, permutation: np.ndarray) -> dict:
    """Expose exchangeability and actual movement of the restricted null."""
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["question_group"], []).append(i)
    signatures = {}
    for group, indices in groups.items():
        ordered = sorted(indices, key=lambda i: (rows[i]["conv_id"], rows[i]["variant"]))
        signatures[group] = tuple((rows[i]["variant"], bool(target.available[i])) for i in ordered)
    signature_counts = {}
    for signature in signatures.values():
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    moved = permutation != np.arange(len(rows))
    moved_groups = sum(any(moved[i] for i in indices) for indices in groups.values())
    changed = ~np.isclose(
        target.values, target.values[permutation], rtol=0, atol=0, equal_nan=True
    ).all(1)
    return {
        "total_groups": len(groups),
        "exchangeable_groups": sum(signature_counts[s] > 1 for s in signatures.values()),
        "moved_groups": moved_groups,
        "fixed_groups": len(groups) - moved_groups,
        "moved_rows": int(moved.sum()),
        "moved_available_rows": int((moved & target.available).sum()),
        "changed_available_target_rows": int((changed & target.available).sum()),
        "interpretation": "One restricted shuffle; unexchangeable components remain fixed. No permutation p-value.",
    }


def validate_annotation_route(root: Path, labels: Path, config: dict) -> tuple[dict, str]:
    """Validate each producer through its own raw-evidence and pilot gates."""
    provider = config.get("provider")
    if provider == CODEX_PROVIDER:
        checked = validate_codex_main(root)
        if labels.resolve() != Path(checked["labels_path"]).resolve():
            raise ValueError("Readout labels differ from the validated Codex aggregate")
        if config != checked["config"]:
            raise ValueError("Readout configuration differs from the validated Codex recipe")
        return checked["acceptance"], "expected_annotations"
    if provider is not None:
        raise ValueError(f"Unknown annotation provider: {provider}")
    # The original API collector predates the provider field. Keep its gate and
    # call-count semantics intact; Codex judgments never impersonate API calls.
    return validate_pilot_acceptance(root, config), "expected_calls"


def load_inputs(cfg: Config) -> tuple[list[dict], dict[str, np.ndarray], list[Target], dict]:
    """Join vectors and actual labels by exact IDs, and assert split independence."""
    root = Path(cfg.root)
    label_parent = (root / cfg.labels).parent
    config = json.loads((label_parent / "config.json").read_text())
    acceptance, completion_count_key = validate_annotation_route(root, root / cfg.labels, config)
    paths = {
        "rows": root / "prepared/rows.jsonl",
        "vectors": root / "prepared/vectors.npz",
        "rubrics": root / "prepared/rubrics.json",
        "labels": root / cfg.labels,
        "labels_manifest": label_parent / "labels_manifest.json",
        "annotation_complete": label_parent / "complete.json",
        "annotation_config": label_parent / "config.json",
    }
    rows_all = [json.loads(line) for line in paths["rows"].read_text().splitlines()]
    with np.load(paths["vectors"], allow_pickle=False) as stored:
        if list(stored["id"]) != [row["id"] for row in rows_all]:
            raise ValueError("vector/answer ordered identity mismatch")
        selected = np.array([r["part"] == "main" for r in rows_all])
        vectors = {name: stored[name][selected].astype(float) for name in ("answer", "context")}
    rows = [row for row in rows_all if row["part"] == "main"]
    if len(rows) != 2048 or any(v.shape != (2048, 3584) for v in vectors.values()):
        raise ValueError("not the frozen main cohort/vector width")
    if not all(np.isfinite(v).all() for v in vectors.values()):
        raise ValueError("nonfinite observed vectors")
    for field in ("question_group", "conv_id", "question", "answer"):
        folds_by_key: dict[str, set] = {}
        for row in rows:
            folds_by_key.setdefault(row[field], set()).add(row["fold"])
        if any(len(folds) != 1 for folds in folds_by_key.values()):
            raise ValueError(f"outer split leaks {field}")
    if {r["fold"] for r in rows} != set(range(5)):
        raise ValueError("five frozen outer folds required")
    rubrics = json.loads(paths["rubrics"].read_text())
    label_manifest = json.loads(paths["labels_manifest"].read_text())
    completion = json.loads(paths["annotation_complete"].read_text())
    expected_draws = len(rows) * len(rubrics) * 5
    current_schema = annotation_digest(
        {name: {"system": judge_system(name), "schema": schema(name)} for name in PROPERTIES}
    )
    expected_manifest = {
        "part": "main",
        "row_count": len(rows),
        "expected_draws": expected_draws,
        "persisted_draws": expected_draws,
        "exact_keyset_complete": True,
        "labels_sha256": file_hash(paths["labels"]),
        "rows_sha256": file_hash(paths["rows"]),
        "complete_sha256": file_hash(paths["annotation_complete"]),
        "config_hash": annotation_digest(config),
        "rubric_schema_hash": current_schema,
    }
    if any(label_manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("aggregate annotation provenance mismatch or incomplete main wave")
    if any(
        completion.get(key) != value
        for key, value in {
            completion_count_key: expected_draws,
            "persisted_expected": expected_draws,
            "config_hash": label_manifest["config_hash"],
            "expected_keys_hash": label_manifest["expected_keys_hash"],
        }.items()
    ):
        raise ValueError("main annotation completion identity mismatch")
    labels_list = json.loads(paths["labels"].read_text())
    labels = {r["id"]: r["properties"] for r in labels_list}
    if len(labels) != len(labels_list) or set(labels) != {r["id"] for r in rows}:
        raise ValueError("aggregate labels must cover each main answer exactly once")
    targets, composition = [], []
    for name, rubric in rubrics.items():
        classes = rubric.get("labels", [])
        matrix = np.full((len(rows), len(classes) or 1), np.nan)
        modal = np.full(len(rows), -1, int) if classes else None
        for i, row in enumerate(rows):
            label = labels[row["id"]][name]
            if label["kind"] != rubric["kind"] or not 0 <= label["n_valid"] <= 5:
                raise ValueError("aggregate label kind/count mismatch")
            if classes:
                if label["n_valid"]:
                    votes = label["votes"]
                    if set(votes) - set(classes):
                        raise ValueError("unknown annotated category")
                    matrix[i] = [votes.get(c, 0.0) for c in classes]
                    if np.any(matrix[i] < 0) or not np.isclose(matrix[i].sum(), 1):
                        raise ValueError("invalid categorical vote distribution")
                    winners = np.flatnonzero(matrix[i] == matrix[i].max())
                    expected = classes[winners[0]] if len(winners) == 1 else None
                    if label["modal"] != expected:
                        raise ValueError("modal label disagrees with vote fractions/tie policy")
                    modal[i] = int(winners[0]) if expected is not None else -1
            else:
                count = label["n_assessable"]
                if not 0 <= count <= label["n_valid"]:
                    raise ValueError("invalid assessability count")
                if count:
                    score = label["mean"]
                    if isinstance(score, bool) or not isinstance(score, (int, float)):
                        raise ValueError("missing or invalid assessed score")
                    if not 0 <= score <= 100:
                        raise ValueError("assessed score outside rubric range")
                    matrix[i, 0] = score
                elif label["mean"] is not None:
                    raise ValueError("unassessable target must remain missing")
        targets.append(
            Target(name, rubric["kind"], matrix, np.isfinite(matrix).all(1), classes, modal)
        )
    for row in rows:
        composition.append(labels[row["id"]]["persona"]["multi_voice_fraction"])
    # Mechanical expression controls. English lexical rates are explicitly
    # language-specific and are not a general measurement of syntax or style.
    surface: dict[str, list[float]] = {
        name: []
        for name in (
            "log_character_count",
            "log_whitespace_word_count",
            "english_first_person_rate",
            "english_second_person_rate",
            "english_negation_rate",
            "question_mark_rate",
            "newline_rate",
            "code_fence_count",
        )
    }
    for row in rows:
        answer = row["answer"]
        words = re.findall(r"\b[a-z]+\b", answer.lower())
        denominator = max(1, len(words))
        surface["log_character_count"].append(float(np.log1p(len(answer))))
        surface["log_whitespace_word_count"].append(float(np.log1p(len(answer.split()))))
        for name, vocabulary in (
            ("english_first_person_rate", {"i", "me", "my", "mine", "we", "us", "our", "ours"}),
            ("english_second_person_rate", {"you", "your", "yours"}),
            ("english_negation_rate", {"no", "not", "never", "neither", "nor", "without"}),
        ):
            surface[name].append(sum(word in vocabulary for word in words) / denominator)
        surface["question_mark_rate"].append(
            (answer.count("?") + answer.count("？")) / max(1, len(answer))
        )
        surface["newline_rate"].append(answer.count("\n") / max(1, len(answer)))
        surface["code_fence_count"].append(float(answer.count("```")))
    for name, values in surface.items():
        targets.append(
            Target(name, "surface", np.array(values)[:, None], np.ones(len(rows), bool), [])
        )
    vectors["length"] = np.array(surface["log_character_count"])[:, None]
    provenance = {name: {"path": str(p), "sha256": file_hash(p)} for name, p in paths.items()}
    provenance["pilot_acceptance"] = fingerprint(acceptance)
    provenance["multi_voice_fraction"] = composition
    return rows, vectors, targets, provenance


def bundle_targets(targets: list[Target]) -> list[list[Target]]:
    """Group targets with identical availability so their factorization is shared."""
    bundles: dict[bytes, list[Target]] = {}
    for target in targets:
        bundles.setdefault(target.available.tobytes(), []).append(target)
    return list(bundles.values())


def fit_bundle(
    rows: list[dict],
    vectors: dict[str, np.ndarray],
    targets: list[Target],
    fold: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Tune per target inside each outer fold, sharing fits across target columns."""
    available = targets[0].available
    if not all(np.array_equal(t.available, available) for t in targets):
        raise ValueError("bundle availability mismatch")
    outer = np.array([r["fold"] for r in rows])
    train = np.flatnonzero(available & (outer != fold))
    test = np.flatnonzero(available & (outer == fold))
    if len(train) < 4 or not len(test):
        raise ValueError("target has insufficient train/test availability")
    groups = np.array([r["question_group"] for r in rows])
    if set(groups[train]) & set(groups[test]):
        raise ValueError("outer connected-group leakage")
    inner = grouped_folds(groups[train])
    values = np.concatenate([t.values for t in targets], axis=1)
    boundaries = np.cumsum([0] + [t.values.shape[1] for t in targets])
    segments = [slice(a, b) for a, b in pairwise(boundaries)]
    result = {
        "test": test,
        "prior": np.broadcast_to(values[train].mean(0), (len(test), values.shape[1])).copy(),
    }
    framing = np.array([r["variant"] for r in rows])
    result["framing"] = np.empty_like(result["prior"])
    for variant in np.unique(framing):
        source = train[framing[train] == variant]
        if not len(source):
            raise ValueError("framing baseline has no training support")
        result["framing"][framing[test] == variant] = values[source].mean(0)
    details = {
        "fold": fold,
        "train_ids": [rows[i]["id"] for i in train],
        "test_ids": [rows[i]["id"] for i in test],
        "targets": [t.name for t in targets],
        "boundaries": boundaries.tolist(),
        "models": {},
        "unsupported_widths": [],
    }
    for source, features in vectors.items():
        widths = (0,) if source == "length" else (0, 256, 512)
        minimum_inner = min(int((inner != f).sum()) for f in range(3))
        supported = tuple(w for w in widths if w <= min(minimum_inner - 1, features.shape[1]))
        details["unsupported_widths"].extend(
            [f"{source}_pca{w}" for w in widths if w not in supported]
        )
        losses = {w: np.zeros((len(ALPHAS), len(targets))) for w in supported}
        counts = np.zeros(len(targets))
        for inner_fold in range(3):
            fit, validation = train[inner != inner_fold], train[inner == inner_fold]
            if set(groups[fit]) & set(groups[validation]):
                raise ValueError("inner connected-group leakage")
            grid = ridge_width_grid(
                features[fit], values[fit], features[validation], ALPHAS, supported
            )
            for width, scores in grid.items():
                for j, segment in enumerate(segments):
                    losses[width][:, j] += np.square(
                        scores[:, :, segment] - values[validation, segment]
                    ).sum((1, 2))
            counts += len(validation) * np.diff(boundaries)
        final_grid = ridge_width_grid(
            features[train], values[train], features[test], ALPHAS, supported
        )
        for width, scores in final_grid.items():
            name = source if width == 0 else f"{source}_pca{width}"
            result[name] = np.empty_like(result["prior"])
            choices = {}
            for j, (target, segment) in enumerate(zip(targets, segments, strict=True)):
                loss = losses[width][:, j] / counts[j]
                choice = int(np.flatnonzero(loss == loss.min())[-1])
                result[name][:, segment] = scores[choice, :, segment]
                choices[target.name] = {
                    "alpha": float(ALPHAS[choice]),
                    "boundary": choice in (0, len(ALPHAS) - 1),
                    "inner_mse": loss.tolist(),
                }
            details["models"][name] = choices
    return result, details


def fit(cfg: Config) -> None:
    """Fit checkpointed observed and single-shuffle diagnostics; never synthesize labels."""
    rows, vectors, original, provenance = load_inputs(cfg)
    out = Path(cfg.root) / cfg.output
    out.mkdir(parents=True, exist_ok=True)
    eligible, unavailable = [], {}
    outer = np.array([r["fold"] for r in rows])
    groups = np.array([r["question_group"] for r in rows])
    for target in original:
        reasons = []
        for fold in range(5):
            train = target.available & (outer != fold)
            test = target.available & (outer == fold)
            if train.sum() < 4 or not test.any() or len(np.unique(groups[train])) < 3:
                reasons.append(f"fold {fold}: insufficient available training/test rows or groups")
        if reasons:
            unavailable[target.name] = reasons
        else:
            eligible.append(target)
    if not eligible:
        raise ValueError("no target supports the frozen grouped readout protocol")
    targets, permutations = list(eligible), {}
    for target in eligible:
        permutation = component_shuffle(rows, target.available, cfg.seed)
        permutations[target.name] = {
            "permutation": permutation.tolist(),
            **shuffle_diagnostics(rows, target, permutation),
        }
        targets.append(
            Target(
                "shuffle__" + target.name,
                target.kind,
                target.values[permutation],
                target.available,
                target.classes,
                target.modal[permutation] if target.modal is not None else None,
            )
        )
    regime = {
        "inputs": provenance,
        "script_sha256": file_hash(Path(__file__)),
        "alphas": ALPHAS.tolist(),
        "inner_folds": 3,
        "pca_widths": [256, 512],
        "seed": cfg.seed,
        "shuffle_draws": 1,
        "shuffle_inference": "sanity control; no p-value",
    }
    key = fingerprint(regime)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != key:
        raise ValueError("output belongs to another regime; choose a fresh output path")
    write_json_atomic(
        manifest_path,
        {
            "fingerprint": key,
            "regime": regime,
            "permutations": permutations,
            "targets": [t.name for t in eligible],
            "unavailable_targets": unavailable,
        },
    )
    bundles = bundle_targets(targets)
    for fold in range(1 if cfg.first_fold_only else 5):
        for bi, bundle in enumerate(bundles):
            name = f"fold{fold}_bundle{bi}"
            meta_path, tensor_path = out / f"{name}.json", out / f"{name}.npz"
            if meta_path.exists():
                previous = json.loads(meta_path.read_text())
                if (
                    previous["fingerprint"] != key
                    or file_hash(tensor_path) != previous["tensor_sha256"]
                ):
                    raise ValueError("checkpoint identity/content mismatch")
                continue
            started = time.monotonic()
            result, details = fit_bundle(rows, vectors, bundle, fold)
            with atomic_replace(tensor_path) as temporary, temporary.open("wb") as handle:
                np.savez(handle, **result)
            details.update(
                fingerprint=key,
                tensor_sha256=file_hash(tensor_path),
                elapsed_seconds=time.monotonic() - started,
            )
            write_json_atomic(meta_path, details)
            print(
                f"[readout] fold={fold + 1}/5 bundle={bi + 1}/{len(bundles)} elapsed={details['elapsed_seconds']:.2f}s",
                flush=True,
            )
    if not cfg.first_fold_only:
        write_json_atomic(
            out / "fits_complete.json", {"fingerprint": key, "folds": 5, "bundles": len(bundles)}
        )


def finite_summary(values: np.ndarray) -> dict:
    """Summarize bootstrap replicates, explicitly counting undefined statistics."""
    finite = values[np.isfinite(values)]
    return {
        "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
        "defined_draws": len(finite),
        "total_draws": len(values),
        "conditioning": "interval uses only defined bootstrap replicates; undefined draws are excluded",
    }


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return missing for genuinely undefined ratios, never a zero placeholder."""
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def continuous_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict:
    """Batch weighted MSE, R² and correlation using fixed-pool sufficient sums."""
    y, prediction = y.ravel(), prediction.ravel()
    sse = weights @ np.square(y - prediction)
    # Translating each variable before sufficient sums avoids cancellation for
    # constant or nearly constant targets. Correlations/variances are invariant.
    y, prediction = y - y.mean(), prediction - prediction.mean()
    count = weights.sum(1)
    sy, sp = weights @ y, weights @ prediction
    yy = weights @ (y * y) - divide(sy * sy, count)
    pp = weights @ (prediction * prediction) - divide(sp * sp, count)
    yp = weights @ (y * prediction) - divide(sy * sp, count)
    # Cancellation can create a tiny negative variance for constant values.
    yy, pp = np.maximum(yy, 0), np.maximum(pp, 0)
    return {
        "mse": divide(sse, count),
        "r2": 1 - divide(sse, yy),
        "pearson": divide(yp, np.sqrt(yy * pp)),
    }


def categorical_metrics(
    target: Target,
    prediction: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Batch fixed-class confusion metrics; ambiguous annotation ties stay excluded."""
    actual = target.modal[indices]
    unique_mode = actual >= 0
    predicted = prediction.argmax(1)
    c = len(target.classes)
    truth = np.eye(c)[np.maximum(actual, 0)] * unique_mode[:, None]
    chosen = np.eye(c)[predicted]
    support = weights @ truth
    predicted_support = weights @ (chosen * unique_mode[:, None])
    correct = weights @ (truth * chosen)
    present = support[0] > 0
    recall = divide(correct[:, present], support[:, present])
    f1 = divide(2 * correct[:, present], support[:, present] + predicted_support[:, present])
    metrics = {
        "balanced_accuracy": recall.mean(1) if present.any() else np.full(len(weights), np.nan),
        "macro_f1": f1.mean(1) if present.any() else np.full(len(weights), np.nan),
        "accuracy": divide(correct.sum(1), support.sum(1)),
        "vote_mse": divide(
            weights @ np.square(prediction - target.values[indices]).mean(1), weights.sum(1)
        ),
    }
    auc = {}
    for ci, name in enumerate(target.classes):
        binary = actual[unique_mode] == ci
        auc[name] = (
            float(roc_auc_score(binary, prediction[unique_mode, ci]))
            if len(np.unique(binary)) == 2
            else None
        )
    details = {
        "modal_ties_excluded": int((~unique_mode).sum()),
        "class_support": support[0].astype(int).tolist(),
        "macro_classes": [name for name, keep in zip(target.classes, present, strict=True) if keep],
        "ovr_auroc": auc,
        "prediction_ties": "first category in the frozen rubric order; annotation ties are excluded",
    }
    return metrics, details


def score_target(
    target: Target,
    predictions: dict[str, np.ndarray],
    rows: list[dict],
    group_counts: np.ndarray,
    inverse_groups: np.ndarray,
    subset: np.ndarray,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Score identical rows across arms and retain paired bootstrap draws."""
    indices = np.flatnonzero(target.available & subset)
    if not len(indices):
        return {"n": 0, "reason": "no available targets in this subset"}, {}
    weights = np.vstack([np.ones(len(indices)), group_counts[:, inverse_groups[indices]]])
    scalar = target.kind != "categorical"
    result = {
        "n": len(indices),
        "n_groups": len({rows[i]["question_group"] for i in indices}),
        "kind": target.kind,
        "classes": target.classes,
        "models": {},
    }
    if target.classes:
        modal = target.modal[indices]
        result["class_support"] = {}
        for ci, category in enumerate(target.classes):
            positive = indices[modal == ci]
            negative = indices[(modal >= 0) & (modal != ci)]
            result["class_support"][category] = {
                "positive_answers": len(positive),
                "negative_answers": len(negative),
                "positive_connected_groups": len({rows[i]["question_group"] for i in positive}),
                "negative_connected_groups": len({rows[i]["question_group"] for i in negative}),
            }
    draws = {}
    for arm, all_predictions in predictions.items():
        prediction = all_predictions[indices]
        if not np.isfinite(prediction).all():
            # This covers a predeclared PCA width that was unavailable in a
            # training fold. It is not silently scored on a favorable subset.
            result["models"][arm] = {"reason": "model unavailable on at least one selected row"}
            continue
        if scalar:
            metrics = continuous_metrics(target.values[indices], prediction, weights)
            y_rank, p_rank = rankdata(target.values[indices, 0]), rankdata(prediction[:, 0])
            spearman = continuous_metrics(y_rank, p_rank, weights[:1])["pearson"][0]
            details = {"spearman": float(spearman) if np.isfinite(spearman) else None}
            if np.isfinite(predictions["framing"][indices]).all():
                residual = continuous_metrics(
                    target.values[indices] - predictions["framing"][indices],
                    prediction - predictions["framing"][indices],
                    weights,
                )
                metrics["within_framing_pearson"] = residual["pearson"]
        else:
            metrics, details = categorical_metrics(target, prediction, indices, weights)
        error = "mse" if scalar else "vote_mse"
        for baseline in ("prior", "framing"):
            base_error = weights @ np.square(
                predictions[baseline][indices] - target.values[indices]
            ).mean(1)
            base_error = divide(base_error, weights.sum(1))
            metrics[f"skill_vs_{baseline}"] = 1 - divide(metrics[error], base_error)
        result["models"][arm] = {
            **details,
            **{
                name: {
                    "value": float(values[0]) if np.isfinite(values[0]) else None,
                    **finite_summary(values[1:]),
                }
                for name, values in metrics.items()
            },
        }
        draws[arm] = metrics
    paired = {}
    for comparison in ("context", "framing", "length"):
        if "answer" in draws and comparison in draws:
            paired["answer_minus_" + comparison] = {
                metric: {
                    "value": float(difference[0]) if np.isfinite(difference[0]) else None,
                    **finite_summary(difference[1:]),
                }
                for metric in draws["answer"]
                if metric in draws[comparison]
                for difference in [draws["answer"][metric] - draws[comparison][metric]]
            }
    result["paired_differences"] = paired
    return result, draws


def summarize(cfg: Config) -> None:
    """Reconstruct OOF predictions and report paired component-bootstrap uncertainty."""
    rows, _, original, provenance = load_inputs(cfg)
    out = Path(cfg.root) / cfg.output
    manifest = json.loads((out / "manifest.json").read_text())
    complete = json.loads((out / "fits_complete.json").read_text())
    if complete["fingerprint"] != manifest["fingerprint"] or complete["folds"] != 5:
        raise ValueError("completed-fit regime mismatch")
    if provenance != manifest["regime"]["inputs"]:
        raise ValueError("analysis inputs changed after fitting")
    original = [t for t in original if t.name in manifest["targets"]]
    targets = list(original)
    for target in original:
        perm = np.array(manifest["permutations"][target.name]["permutation"])
        targets.append(
            Target(
                "shuffle__" + target.name,
                target.kind,
                target.values[perm],
                target.available,
                target.classes,
                target.modal[perm] if target.modal is not None else None,
            )
        )
    by_name = {target.name: target for target in targets}
    arms = (
        "prior",
        "framing",
        "length",
        "answer",
        "context",
        "answer_pca256",
        "context_pca256",
        "answer_pca512",
        "context_pca512",
    )
    predictions = {t.name: {arm: np.full(t.values.shape, np.nan) for arm in arms} for t in targets}
    assignment = {t.name: np.zeros(len(rows), int) for t in targets}
    fit_details = []
    for fold in range(5):
        for bi in range(complete["bundles"]):
            prefix = out / f"fold{fold}_bundle{bi}"
            meta = json.loads(prefix.with_suffix(".json").read_text())
            if (
                meta["fingerprint"] != manifest["fingerprint"]
                or file_hash(prefix.with_suffix(".npz")) != meta["tensor_sha256"]
            ):
                raise ValueError("stale/corrupt fold checkpoint")
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as stored:
                indices = stored["test"]
                if any(rows[i]["fold"] != fold for i in indices):
                    raise ValueError("OOF checkpoint fold mismatch")
                for j, name in enumerate(meta["targets"]):
                    segment = slice(meta["boundaries"][j], meta["boundaries"][j + 1])
                    for arm in arms:
                        if arm in stored:
                            predictions[name][arm][indices] = stored[arm][:, segment]
                    assignment[name][indices] += 1
            fit_details.append(meta)
    for name, counts in assignment.items():
        if not np.array_equal(counts, by_name[name].available.astype(int)):
            raise ValueError("target OOF coverage is not exactly once per available row")
    groups, inverse = np.unique([r["question_group"] for r in rows], return_inverse=True)
    rng = np.random.default_rng(cfg.seed)
    group_counts = rng.multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=cfg.bootstrap_draws
    )
    composition = provenance["multi_voice_fraction"]
    subsets = {
        "all": np.ones(len(rows), bool),
        "single_voice_without_substantial_narration": np.array(
            [v is not None and v < 0.5 for v in composition]
        ),
        "uncapped": np.array([r["finish_reason"] == "stop" for r in rows]),
    }
    report = {
        "n_answers": len(rows),
        "n_questions": len({r["conv_id"] for r in rows}),
        "n_connected_groups": len(groups),
        "bootstrap_draws": cfg.bootstrap_draws,
        "uncertainty": "Connected-question-component bootstrap of fixed OOF predictions; no refitting; conditional on this cohort, labels and training procedure. Intervals condition on defined replicates; undefined-draw counts must accompany rare-class intervals.",
        "primary": "full-width answer/context ridge; train-only standardization and three grouped inner folds",
        "sensitivities": "PCA models refitted in each inner/outer training split; text-composition and uncapped subsets are evaluation-only, with fixed OOF predictions",
        "limits": "Judge-defined expressed properties in narrative-framed answers. No human agreement, intrinsic ceiling, universal high/low ranking, or Euclidean-distance validation. English lexical controls are not comprehensive syntax. A single restricted shuffle supplies no p-value.",
        "fingerprint": manifest["fingerprint"],
        "unavailable_targets": manifest["unavailable_targets"],
        "shuffle_diagnostics": {
            name: {k: v for k, v in item.items() if k != "permutation"}
            for name, item in manifest["permutations"].items()
        },
        "properties": {},
    }
    draw_store = {}
    for target in targets:
        panels = {}
        for subset_name, subset in subsets.items():
            metrics, draws = score_target(
                target, predictions[target.name], rows, group_counts, inverse, subset
            )
            panels[subset_name] = metrics
            for arm, values in draws.items():
                for metric, samples in values.items():
                    draw_store[f"{target.name}__{subset_name}__{arm}__{metric}"] = samples
        if target.classes:
            # A declared supported-class diagnostic, never a post-hoc relabeling.
            supported = [
                ci
                for ci in range(len(target.classes))
                if len(
                    {rows[i]["question_group"] for i in range(len(rows)) if target.modal[i] == ci}
                )
                >= 20
            ]
            mask = np.isin(target.modal, supported)
            panels["classes_with_at_least_20_connected_groups"], _ = score_target(
                target, predictions[target.name], rows, group_counts, inverse, mask
            )
        report["properties"][target.name] = panels
        write_json_atomic(out / f"metrics_{target.name}.json", panels)
        print(f"[summary] target={target.name} finished", flush=True)
    np.savez(out / "bootstrap_metric_draws.npz", **draw_store)
    np.savez(
        out / "oof_predictions.npz",
        id=np.array([r["id"] for r in rows]),
        **{
            f"{name}__{arm}": value
            for name, mapping in predictions.items()
            for arm, value in mapping.items()
        },
    )
    write_json_atomic(out / "fit_parameters.json", fit_details)
    write_json_atomic(out / "summary.json", report)
    write_json_atomic(
        out / "analysis_complete.json",
        {"fingerprint": manifest["fingerprint"], "summary_sha256": file_hash(out / "summary.json")},
    )


ConfigStore.instance().store(name="answer_behavior_readout", node=Config)


@hydra.main(version_base=None, config_name="answer_behavior_readout")
def main(cfg: Config) -> None:
    """Dispatch the explicitly selected phase; unsupported phases fail loudly."""
    if cfg.stage == "fit":
        fit(cfg)
    elif cfg.stage == "summarize":
        summarize(cfg)
    else:
        raise ValueError(f"unsupported stage: {cfg.stage}")


if __name__ == "__main__":
    main()
