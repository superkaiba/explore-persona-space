#!/usr/bin/env python3
"""Direct answer-side SAE read of observed, mapped, and residual pair deltas.

This is a CPU-only follow-up to task #2569.  It reuses frozen layer-19
representations, the banked context-to-answer ridge map, and the answer-side
SAE.  No model forward passes or generations are performed.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # bind shared-VM thread caps before numpy/torch imports

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
D_MODEL = 3584
LAYER = 19
N_FEATURES = 32_768
SEED = 25_691
N_BOOT = 2_000
TOP_K = 100
FLIP_GAP = 0.5
CONTROL_CLASS = "verb_harm"
PRIOR_REFUSAL_COSINE_MEAN = 0.778500892676814
PRIOR_REFUSAL_COSINE_MEDIAN = 0.799450054707672
PRIOR_REPRO_TOLERANCE = 1e-12

TOPIC_REVISION = "62b1e8889e1a262501937b0ec6f6022e28b4a7e6"
ONEWORD_REVISION = "7f48a64fe9ea24c52e8321cf5b6e346317df819a"
SAE_REVISION = "cd80ba2588bb6d4291edf621176ea654bcbf2507"

DEFAULT_CACHE = Path("/home/thomasjiralerspong/eps-runs/cache-2569-answer-residual-sae")
DEFAULT_TOPIC_ROOT = DEFAULT_CACHE / TOPIC_REVISION / "issue2564_minpair"
DEFAULT_ONEWORD_ROOT = DEFAULT_CACHE / ONEWORD_REVISION / "issue2564_minpair/lang_oneword_pilot"
DEFAULT_LEG9_MANIFEST = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2569_theory/leg9_dl/leg9_manifest.json"
)
DEFAULT_RIDGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2094_r2decomp/banked_maps/"
    "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
)
DEFAULT_SAE = Path(
    "/home/thomasjiralerspong/.cache/huggingface/"
    "datasets--superkaiba1--explore-persona-space-data/snapshots/"
    f"{SAE_REVISION}/issue2552_derreplication/exactrep/analysis_tensors/"
    "sae_rep/sae_weights.safetensors"
)
DEFAULT_LABELS = Path(
    "/mnt/eps-data/thomasjiralerspong/sae_interpretation_packets/codex/"
    "codex_interpretation_results_final.json"
)
DEFAULT_OUT = REPO / "eval_results/issue_2569/followup_answer_residual_sae"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path, **extra: object) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), **extra}


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("direction matrix contains a zero or non-finite row")
    return values / norms


def row_cos(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("row cosine received a zero vector")
    return np.einsum("ij,ij->i", left, right) / denominator


def bootstrap_summary(values: np.ndarray, seed_offset: int = 0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap input must be a finite non-empty vector")
    rng = np.random.default_rng(SEED + seed_offset)
    indices = rng.integers(0, values.size, size=(N_BOOT, values.size))
    draws = values[indices]
    boot_mean = draws.mean(axis=1)
    boot_median = np.median(draws, axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "mean_ci95": [float(x) for x in np.percentile(boot_mean, [2.5, 97.5])],
        "median": float(np.median(values)),
        "median_ci95": [float(x) for x in np.percentile(boot_median, [2.5, 97.5])],
        "min": float(values.min()),
        "max": float(values.max()),
    }


def mean_answer_rows(store: dict, context_ids: list[str], layer_index: int) -> np.ndarray:
    """Join and average valid answer rows by context id without silent drops."""
    row_of = {context_id: i for i, context_id in enumerate(context_ids)}
    values = store["va_tail_incl"][:, layer_index].to(torch.float64).numpy()
    index = store["index"]
    if values.shape != (len(index), D_MODEL):
        raise ValueError((values.shape, len(index)))
    empty = set(int(x) for x in store.get("empty_rows", []))
    sums = np.zeros((len(context_ids), D_MODEL), dtype=np.float64)
    counts = np.zeros(len(context_ids), dtype=np.int64)
    absent: list[str] = []
    for row_i, record in enumerate(index):
        if row_i in empty or int(record.get("n_completion_tokens", 1)) <= 0:
            continue
        context_id = record["context_id"]
        if context_id not in row_of:
            absent.append(context_id)
            continue
        target = row_of[context_id]
        sums[target] += values[row_i]
        counts[target] += 1
    if absent:
        raise ValueError(f"answer rows absent from context store: {len(absent)}")
    return sums, counts


def load_topic(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    manifest_path = root / "manifests/bank2564_manifest.json"
    vc_path = root / "analysis_tensors/vc2564/vc2564_bank.pt"
    va_path = root / "analysis_tensors/va2564/va2564_query.pt"
    bank = json.loads(manifest_path.read_text(encoding="utf-8"))
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False, mmap=True)
    va_store = torch.load(va_path, map_location="cpu", weights_only=False, mmap=True)
    layers = [int(x) for x in vc_store["layers"]]
    if layers != [int(x) for x in va_store["layers"]] or LAYER not in layers:
        raise ValueError("topic context/answer layer mismatch")
    layer_index = layers.index(LAYER)
    context_ids = list(vc_store["context_ids"])
    row_of = {context_id: i for i, context_id in enumerate(context_ids)}
    context = vc_store["vc"][:, layer_index].to(torch.float64).numpy()
    answer_sums, answer_counts = mean_answer_rows(va_store, context_ids, layer_index)
    pairs = [record for record in bank["pairs"] if record["pair_class"] == "query_content"]
    if len(pairs) != 66:
        raise ValueError(f"expected 66 question-topic pairs, found {len(pairs)}")
    pair_ids = [record["pair_id"] for record in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate question-topic pair ids")
    a = np.array([row_of[record["a"]] for record in pairs], dtype=np.int64)
    b = np.array([row_of[record["b"]] for record in pairs], dtype=np.int64)
    needed = np.unique(np.concatenate([a, b]))
    if np.any(answer_counts[needed] <= 0):
        raise ValueError("a selected question-topic context has no answer rows")
    answer = answer_sums / np.where(answer_counts[:, None] > 0, answer_counts[:, None], 1)
    # Frozen #2564 analysis convention is manifest a minus manifest b.
    return (
        context[a] - context[b],
        answer[a] - answer[b],
        pair_ids,
        {
            "orientation": "manifest a minus manifest b (frozen #2564 convention)",
            "n_contexts_in_vc": len(context_ids),
            "n_answer_rows": len(va_store["index"]),
            "selected_answer_draw_counts": sorted(set(int(x) for x in answer_counts[needed])),
            "sources": {
                "manifest": source_record(manifest_path, revision=TOPIC_REVISION),
                "context": source_record(vc_path, revision=TOPIC_REVISION),
                "answer": source_record(va_path, revision=TOPIC_REVISION),
            },
        },
    )


def load_oneword(root: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    manifest_path = root / "manifests/pilot_bank.json"
    vc_path = root / "analysis_tensors/vc/vc_langow_bank.pt"
    va_path = root / "analysis_tensors/va/va_langow_query_content_oneword.pt"
    bank = json.loads(manifest_path.read_text(encoding="utf-8"))
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False, mmap=True)
    va_store = torch.load(va_path, map_location="cpu", weights_only=False, mmap=True)
    layers = [int(x) for x in vc_store["layers"]]
    if layers != [int(x) for x in va_store["layers"]] or LAYER not in layers:
        raise ValueError("one-word context/answer layer mismatch")
    layer_index = layers.index(LAYER)
    context_ids = list(vc_store["context_ids"])
    row_of = {context_id: i for i, context_id in enumerate(context_ids)}
    context = vc_store["vc"][:, layer_index].to(torch.float64).numpy()
    answer_sums, answer_counts = mean_answer_rows(va_store, context_ids, layer_index)
    pairs = [record for record in bank["pairs"] if record["pair_class"] == "query_content_oneword"]
    if len(pairs) != 24:
        raise ValueError(f"expected 24 one-word pairs, found {len(pairs)}")
    pair_ids = [record["pair_id"] for record in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate one-word pair ids")
    a = np.array([row_of[record["a"]] for record in pairs], dtype=np.int64)
    b = np.array([row_of[record["b"]] for record in pairs], dtype=np.int64)
    needed = np.unique(np.concatenate([a, b]))
    if np.any(answer_counts[needed] <= 0):
        raise ValueError("a selected one-word context has no answer rows")
    answer = answer_sums / np.where(answer_counts[:, None] > 0, answer_counts[:, None], 1)
    return (
        context[a] - context[b],
        answer[a] - answer[b],
        pair_ids,
        {
            "orientation": "manifest a minus manifest b (frozen pilot convention)",
            "n_contexts_in_vc": len(context_ids),
            "n_answer_rows": len(va_store["index"]),
            "selected_answer_draw_counts": sorted(set(int(x) for x in answer_counts[needed])),
            "sources": {
                "manifest": source_record(manifest_path, revision=ONEWORD_REVISION),
                "context": source_record(vc_path, revision=ONEWORD_REVISION),
                "answer": source_record(va_path, revision=ONEWORD_REVISION),
            },
        },
    )


def load_refusal(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    paths = json.loads(manifest_path.read_text(encoding="utf-8"))

    def resolve(key: str) -> Path:
        path = Path(paths[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    pairs_path = resolve("issue2617_svmp/manifests/svmp_bank.json")
    judge_path = resolve("issue2617_svmp/raw_completions/judge/judge_scores.json")
    vc_path = resolve("issue2617_svmp/analysis_tensors/vc/vc_langow_bank.pt")
    va_path = resolve("issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt")
    bank = json.loads(pairs_path.read_text(encoding="utf-8"))
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False, mmap=True)
    va_store = torch.load(va_path, map_location="cpu", weights_only=False, mmap=True)
    layers = [int(x) for x in vc_store["layers"]]
    if layers != [int(x) for x in va_store["layers"]] or LAYER not in layers:
        raise ValueError("refusal context/answer layer mismatch")
    layer_index = layers.index(LAYER)
    context_ids = list(vc_store["context_ids"])
    row_of = {context_id: i for i, context_id in enumerate(context_ids)}
    context = vc_store["vc"][:, layer_index].to(torch.float64).numpy()
    answer_sums, answer_counts = mean_answer_rows(va_store, context_ids, layer_index)
    answer = answer_sums / np.where(answer_counts[:, None] > 0, answer_counts[:, None], 1)
    rates = judge["per_context"]
    selected: list[tuple[str, str, str]] = []
    for record in bank["pairs"]:
        if record["pair_class"] == CONTROL_CLASS:
            continue
        rate_a = float(rates[record["a"]]["refusal_rate"])
        rate_b = float(rates[record["b"]]["refusal_rate"])
        if abs(rate_a - rate_b) < FLIP_GAP:
            continue
        high, low = (record["a"], record["b"]) if rate_a >= rate_b else (record["b"], record["a"])
        selected.append((record["pair_id"], high, low))
    if len(selected) != 60:
        raise ValueError(f"expected 60 refusal flips, found {len(selected)}")
    pair_ids = [record[0] for record in selected]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate refusal pair ids")
    high = np.array([row_of[record[1]] for record in selected], dtype=np.int64)
    low = np.array([row_of[record[2]] for record in selected], dtype=np.int64)
    needed = np.unique(np.concatenate([high, low]))
    if np.any(answer_counts[needed] <= 0):
        raise ValueError("a selected refusal context has no answer rows")
    return (
        context[high] - context[low],
        answer[high] - answer[low],
        pair_ids,
        {
            "orientation": "lower-refusal to higher-refusal context (high-low)",
            "refusal_gap_threshold": FLIP_GAP,
            "n_all_pairs": len(bank["pairs"]),
            "n_contexts_in_vc": len(context_ids),
            "n_answer_rows": len(va_store["index"]),
            "selected_answer_draw_counts": sorted(set(int(x) for x in answer_counts[needed])),
            "sources": {
                "leg9_manifest": source_record(manifest_path),
                "pair_manifest": source_record(pairs_path),
                "judge": source_record(judge_path),
                "context": source_record(vc_path),
                "answer": source_record(va_path),
            },
        },
    )


def load_operator(path: Path) -> tuple[np.ndarray, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("kind") != "ridge" or payload.get("fitter") != "ridge":
        raise ValueError("unexpected map payload kind")
    if int(payload["layer"]) != LAYER:
        raise ValueError(f"expected layer {LAYER}, got {payload['layer']}")
    weight = payload["W"].to(torch.float64).numpy()
    xmu = payload["xmu"].to(torch.float64).numpy()
    xsd = payload["xsd"].to(torch.float64).numpy()
    ymu = payload["ymu"].to(torch.float64).numpy()
    if weight.shape != (D_MODEL, D_MODEL) or xmu.shape != (D_MODEL,) or xsd.shape != (D_MODEL,):
        raise ValueError("unexpected map dimensions")
    if not all(np.all(np.isfinite(x)) for x in (weight, xmu, xsd, ymu)) or np.any(xsd <= 0):
        raise ValueError("non-finite map or non-positive scale")
    operator = weight / xsd[:, None]
    return operator, {
        "source": source_record(path),
        "selected_lambda": float(payload["selected_lambda"]),
        "shape": list(operator.shape),
        "row_action": "delta_answer_predicted = delta_context @ (diag(1/xsd) @ W)",
    }


def load_decoder(path: Path) -> np.ndarray:
    with safe_open(path, framework="pt", device="cpu") as handle:
        decoder = handle.get_tensor("w_dec").to(torch.float64).numpy()
    if decoder.shape != (N_FEATURES, D_MODEL) or not np.all(np.isfinite(decoder)):
        raise ValueError(f"unexpected answer SAE decoder: {decoder.shape}")
    return unit_rows(decoder)


def load_labels(path: Path) -> tuple[dict[int, dict], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[int, dict] = {}
    for record in document["features"]:
        if record["side"] != "answer":
            continue
        feature_id = int(record["feature_id"])
        if feature_id in labels:
            raise ValueError(f"duplicate answer label for feature {feature_id}")
        interpretation = record["interpretation"]
        labels[feature_id] = {
            "description": interpretation["description"],
            "category": interpretation["category"],
            "secondary_category": interpretation.get("secondary_category"),
            "confidence": interpretation["confidence"],
            "evidence_basis": interpretation["evidence_basis"],
        }
    return labels, {
        "source": source_record(path),
        "answer_labels": len(labels),
        "protocol": document["protocol"],
        "qualification": (
            "Labels are analyst-assigned descriptions from a previously selected feature corpus; "
            "coverage of these new rankings is incomplete and not a random sample."
        ),
    }


def feature_scores(
    values: np.ndarray, decoder: np.ndarray, block: int = 1_024
) -> dict[str, np.ndarray]:
    """Return full-feature scores using bounded pair-by-feature blocks."""
    values = np.asarray(values, dtype=np.float64)
    normalized = unit_rows(values)
    mean_direction = values.mean(axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    direction_rms = np.empty(N_FEATURES, dtype=np.float64)
    energy_rms = np.empty(N_FEATURES, dtype=np.float64)
    mean_abs_cos = np.empty(N_FEATURES, dtype=np.float64)
    for start in range(0, N_FEATURES, block):
        stop = min(start + block, N_FEATURES)
        directions = decoder[start:stop]
        direction_projection = normalized @ directions.T
        raw_projection = values @ directions.T
        direction_rms[start:stop] = np.sqrt(np.mean(direction_projection**2, axis=0))
        energy_rms[start:stop] = np.sqrt(np.mean(raw_projection**2, axis=0))
        mean_abs_cos[start:stop] = np.abs(mean_direction @ directions.T)
    return {
        "pair_direction_rms": direction_rms,
        "raw_energy_rms": energy_rms,
        "mean_direction_abs_cos": mean_abs_cos,
    }


def stable_order(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    feature_ids = np.arange(scores.size, dtype=np.int64)
    primary = -scores if descending else scores
    return np.lexsort((feature_ids, primary))


def ranking_summary(scores: np.ndarray, labels: dict[int, dict]) -> dict:
    descending = stable_order(scores, descending=True)
    ascending = stable_order(scores, descending=False)

    def rows(order: np.ndarray) -> list[dict]:
        result = []
        for rank, feature_id in enumerate(order[:TOP_K], start=1):
            row = {
                "rank": rank,
                "feature_id": int(feature_id),
                "score": float(scores[feature_id]),
                "label": labels.get(int(feature_id)),
            }
            result.append(row)
        return result

    top = rows(descending)
    bottom = rows(ascending)

    def composition(records: list[dict]) -> dict:
        available = [record["label"] for record in records if record["label"] is not None]
        interpretable = [record for record in available if record["category"] != "uninterpretable"]
        return {
            "n": len(records),
            "n_labeled": len(available),
            "n_interpretable": len(interpretable),
            "category_counts_labeled": dict(
                sorted(Counter(x["category"] for x in available).items())
            ),
            "category_counts_interpretable": dict(
                sorted(Counter(x["category"] for x in interpretable).items())
            ),
        }

    return {
        "top": top,
        "bottom": bottom,
        "top_composition": composition(top),
        "bottom_composition": composition(bottom),
        "score_summary": {
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
            "max": float(scores.max()),
        },
        "labeled_category_rank_percentiles": labeled_category_rank_percentiles(scores, labels),
    }


def labeled_category_rank_percentiles(scores: np.ndarray, labels: dict[int, dict]) -> dict:
    """Positions of the fixed labeled corpus in the full 32,768-feature ranking.

    This avoids comparing raw top-100 category counts with unequal description
    coverage, but it does not remove the corpus's original feature-selection
    bias.  Rank percentile 0 is best and 1 is worst.
    """
    order = stable_order(scores, descending=True)
    rank = np.empty(scores.size, dtype=np.int64)
    rank[order] = np.arange(1, scores.size + 1)
    by_category: dict[str, list[float]] = {}
    for feature_id, label in labels.items():
        by_category.setdefault(label["category"], []).append(float(rank[feature_id] / scores.size))
    result: dict[str, dict] = {}
    for category, values_list in sorted(by_category.items()):
        values = np.asarray(values_list, dtype=np.float64)
        result[category] = {
            "n": int(values.size),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "fraction_top_1pct": float(np.mean(values <= 0.01)),
            "fraction_top_10pct": float(np.mean(values <= 0.10)),
        }
    return result


def analyze_family(
    name: str,
    delta_context: np.ndarray,
    observed: np.ndarray,
    pair_ids: list[str],
    operator: np.ndarray,
    decoder: np.ndarray,
    labels: dict[int, dict],
    seed_offset: int,
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    if delta_context.shape != observed.shape or delta_context.shape != (len(pair_ids), D_MODEL):
        raise ValueError(f"{name}: misaligned pair matrices")
    if not np.all(np.isfinite(delta_context)) or not np.all(np.isfinite(observed)):
        raise ValueError(f"{name}: non-finite pair matrix")
    predicted = delta_context @ operator
    residual = observed - predicted
    reconstruction_error = float(np.max(np.abs(observed - (predicted + residual))))
    if reconstruction_error > 1e-12:
        raise ValueError(f"{name}: residual reconstruction failed: {reconstruction_error}")
    cosine = row_cos(observed, predicted)
    observed_norm = np.linalg.norm(observed, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    metrics = {
        "n_pairs": len(pair_ids),
        "cosine_observed_predicted": bootstrap_summary(cosine, seed_offset),
        "predicted_over_observed_norm": bootstrap_summary(
            predicted_norm / observed_norm, seed_offset + 1
        ),
        "residual_over_observed_norm": bootstrap_summary(
            residual_norm / observed_norm, seed_offset + 2
        ),
        "frobenius_predicted_over_observed": float(
            np.linalg.norm(predicted) / np.linalg.norm(observed)
        ),
        "frobenius_residual_over_observed": float(
            np.linalg.norm(residual) / np.linalg.norm(observed)
        ),
        "variance_accounting": {
            "observed_sq_norm": float(np.sum(observed**2)),
            "predicted_sq_norm": float(np.sum(predicted**2)),
            "residual_sq_norm": float(np.sum(residual**2)),
            "prediction_residual_cross_term_twice": float(2.0 * np.sum(predicted * residual)),
            "identity_error": float(
                abs(
                    np.sum(observed**2)
                    - np.sum(predicted**2)
                    - np.sum(residual**2)
                    - 2.0 * np.sum(predicted * residual)
                )
            ),
            "qualification": (
                "Prediction and residual are not orthogonal, so their squared norms are not "
                "additive explained-variance shares."
            ),
        },
        "max_abs_reconstruction_error": reconstruction_error,
    }
    vectors = {"observed": observed, "predicted": predicted, "residual": residual}
    score_arrays: dict[str, np.ndarray] = {}
    rankings: dict[str, dict] = {}
    for component, values in vectors.items():
        component_scores = feature_scores(values, decoder)
        rankings[component] = {}
        for metric, scores in component_scores.items():
            key = f"{name}__{component}__{metric}"
            score_arrays[key] = scores.astype(np.float32)
            rankings[component][metric] = ranking_summary(scores, labels)
    for metric in ("pair_direction_rms", "raw_energy_rms", "mean_direction_abs_cos"):
        top_sets = {
            component: {row["feature_id"] for row in rankings[component][metric]["top"]}
            for component in vectors
        }
        metrics[f"top_100_overlap__{metric}"] = {
            "observed_predicted": len(top_sets["observed"] & top_sets["predicted"]),
            "observed_residual": len(top_sets["observed"] & top_sets["residual"]),
            "predicted_residual": len(top_sets["predicted"] & top_sets["residual"]),
            "all_three": len(top_sets["observed"] & top_sets["predicted"] & top_sets["residual"]),
        }
    primary_topic_sets = {
        component: {
            row["feature_id"]
            for row in rankings[component]["pair_direction_rms"]["top"]
            if row["label"] is not None and row["label"]["category"] == "substantive_topic_domain"
        }
        for component in vectors
    }
    metrics["labeled_top_100_substantive_topic_overlap"] = {
        "observed": len(primary_topic_sets["observed"]),
        "predicted": len(primary_topic_sets["predicted"]),
        "residual": len(primary_topic_sets["residual"]),
        "observed_also_predicted": len(
            primary_topic_sets["observed"] & primary_topic_sets["predicted"]
        ),
        "residual_also_predicted": len(
            primary_topic_sets["residual"] & primary_topic_sets["predicted"]
        ),
        "qualification": (
            "Descriptive overlap among labeled top-100 features only; unequal selected label "
            "coverage prevents prevalence or enrichment claims."
        ),
    }
    return metrics, rankings, score_arrays


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_report(summary: dict, rankings: dict) -> str:
    lines = [
        "# Direct answer-side residual SAE analysis",
        "",
        "This follow-up compares the observed answer change, the map-predicted answer change, "
        "and their residual using frozen layer-19 vectors. No model inference was rerun.",
        "",
        "## Numerical summary",
        "",
        "| Pair family | n | Median cos(observed, predicted) | Median predicted/observed norm | Median residual/observed norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, family in summary["families"].items():
        lines.append(
            f"| {name} | {family['n_pairs']} | "
            f"{family['cosine_observed_predicted']['median']:.3f} | "
            f"{family['predicted_over_observed_norm']['median']:.3f} | "
            f"{family['residual_over_observed_norm']['median']:.3f} |"
        )
    lines += [
        "",
        "The SAE rankings are descriptive. Decoder directions are correlated and non-orthogonal, "
        "so their scores overlap and cannot be treated as additive variance contributions. The "
        "available Codex descriptions cover only a previously selected subset of answer features.",
        "",
        "## Top-100 description coverage and substantive-topic counts",
        "",
        "Primary score: RMS cosine between each unit-normalized pair direction and each unit SAE "
        "decoder direction.",
        "",
        "| Pair family | Component | Labeled in top 100 | Interpretable | Substantive topic/domain |",
        "|---|---|---:|---:|---:|",
    ]
    for name, family in rankings.items():
        for component in ("observed", "predicted", "residual"):
            composition = family[component]["pair_direction_rms"]["top_composition"]
            topic = composition["category_counts_interpretable"].get("substantive_topic_domain", 0)
            lines.append(
                f"| {name} | {component} | {composition['n_labeled']} | "
                f"{composition['n_interpretable']} | {topic} |"
            )
    question_overlap = summary["families"]["question_topic_66"][
        "labeled_top_100_substantive_topic_overlap"
    ]
    oneword_overlap = summary["families"]["oneword_topic_24"][
        "labeled_top_100_substantive_topic_overlap"
    ]
    lines += [
        "",
        "For the 66 question-topic pairs, "
        f"{question_overlap['observed_also_predicted']}/{question_overlap['observed']} labeled "
        "substantive-topic features in the observed top 100 also occur in the predicted top 100; "
        f"the residual-to-predicted overlap is {question_overlap['residual_also_predicted']}/"
        f"{question_overlap['residual']}. For the 24 one-word pairs, the corresponding overlaps "
        f"are {oneword_overlap['observed_also_predicted']}/{oneword_overlap['observed']} and "
        f"{oneword_overlap['residual_also_predicted']}/{oneword_overlap['residual']}. Thus the "
        "labeled feature view does not place topic information uniquely in the unmapped residual.",
    ]
    lines += [
        "",
        "These are counts among labeled top-100 features, not prevalence estimates. The comparison "
        "is qualitative because label coverage is incomplete and selected. Use the full rankings "
        "JSON for feature-level descriptions and the NPZ for all scores.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-root", type=Path, default=DEFAULT_TOPIC_ROOT)
    parser.add_argument("--oneword-root", type=Path, default=DEFAULT_ONEWORD_ROOT)
    parser.add_argument("--leg9-manifest", type=Path, default=DEFAULT_LEG9_MANIFEST)
    parser.add_argument("--ridge", type=Path, default=DEFAULT_RIDGE)
    parser.add_argument("--sae", type=Path, default=DEFAULT_SAE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.topic_root,
        args.oneword_root,
        args.leg9_manifest,
        args.ridge,
        args.sae,
        args.labels,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    operator, operator_meta = load_operator(args.ridge)
    decoder = load_decoder(args.sae)
    labels, label_meta = load_labels(args.labels)
    loaders = {
        "question_topic_66": load_topic(args.topic_root),
        "oneword_topic_24": load_oneword(args.oneword_root),
        "minimal_refusal_flips_60": load_refusal(args.leg9_manifest),
    }
    # Orientation gate: raw affine predictions must cancel to the row operator.
    payload = torch.load(args.ridge, map_location="cpu", weights_only=False, mmap=True)
    xmu = payload["xmu"].to(torch.float64).numpy()
    xsd = payload["xsd"].to(torch.float64).numpy()
    weight = payload["W"].to(torch.float64).numpy()
    ymu = payload["ymu"].to(torch.float64).numpy()
    probe = loaders["question_topic_66"][0][:8]
    zero = np.zeros_like(probe)
    affine_probe = ((probe - xmu) / xsd) @ weight + ymu
    affine_zero = ((zero - xmu) / xsd) @ weight + ymu
    orientation_error = float(np.max(np.abs(probe @ operator - (affine_probe - affine_zero))))
    if orientation_error > 1e-10:
        raise ValueError(f"row-operator orientation gate failed: {orientation_error}")

    family_summaries: dict[str, dict] = {}
    rankings: dict[str, dict] = {}
    score_arrays: dict[str, np.ndarray] = {}
    pair_hashes: dict[str, str] = {}
    input_meta: dict[str, dict] = {}
    for offset, (name, (delta_context, observed, pair_ids, metadata)) in enumerate(loaders.items()):
        family_summary, family_rankings, family_scores = analyze_family(
            name,
            delta_context,
            observed,
            pair_ids,
            operator,
            decoder,
            labels,
            seed_offset=10 * offset,
        )
        family_summaries[name] = family_summary
        rankings[name] = family_rankings
        score_arrays.update(family_scores)
        pair_hashes[name] = hashlib.sha256("\n".join(pair_ids).encode("utf-8")).hexdigest()
        input_meta[name] = metadata

    refusal_cosine = family_summaries["minimal_refusal_flips_60"]["cosine_observed_predicted"]
    prior_reproduction = {
        "n_pairs": family_summaries["minimal_refusal_flips_60"]["n_pairs"],
        "observed_mean": refusal_cosine["mean"],
        "expected_mean": PRIOR_REFUSAL_COSINE_MEAN,
        "mean_abs_error": abs(refusal_cosine["mean"] - PRIOR_REFUSAL_COSINE_MEAN),
        "observed_median": refusal_cosine["median"],
        "expected_median": PRIOR_REFUSAL_COSINE_MEDIAN,
        "median_abs_error": abs(refusal_cosine["median"] - PRIOR_REFUSAL_COSINE_MEDIAN),
        "tolerance": PRIOR_REPRO_TOLERANCE,
    }
    prior_reproduction["passed"] = (
        prior_reproduction["n_pairs"] == 60
        and prior_reproduction["mean_abs_error"] <= PRIOR_REPRO_TOLERANCE
        and prior_reproduction["median_abs_error"] <= PRIOR_REPRO_TOLERANCE
    )
    if not prior_reproduction["passed"]:
        raise ValueError(f"prior refusal result reproduction failed: {prior_reproduction}")

    summary = {
        "analysis": "direct_answer_residual_sae",
        "layer": LAYER,
        "pooling": "per-context mean of tail-inclusive answer states over valid draws",
        "components": {
            "observed": "delta_answer",
            "predicted": "delta_context @ A",
            "residual": "delta_answer - delta_context @ A",
        },
        "feature_metrics": {
            "pair_direction_rms": (
                "sqrt(mean_i(cos(unit(delta_i), unit(decoder_feature))^2)); primary"
            ),
            "raw_energy_rms": (
                "sqrt(mean_i((delta_i dot unit(decoder_feature))^2)); magnitude sensitivity"
            ),
            "mean_direction_abs_cos": (
                "abs(cos(mean_i(delta_i), unit(decoder_feature))); mean-direction sensitivity"
            ),
            "qualification": (
                "SAE decoder directions are correlated and non-orthogonal; feature scores overlap "
                "and are descriptive rankings, not additive variance shares."
            ),
        },
        "bootstrap": {"unit": "pair", "draws": N_BOOT, "seed": SEED, "interval": "percentile 95%"},
        "orientation_gate": {
            "max_abs_difference": orientation_error,
            "tolerance": 1e-10,
            "passed": True,
        },
        "prior_refusal_reproduction_gate": prior_reproduction,
        "families": family_summaries,
        "pair_id_order_sha256": pair_hashes,
        "inputs": {
            **input_meta,
            "operator": operator_meta,
            "answer_sae": source_record(
                args.sae, revision=SAE_REVISION, shape=[N_FEATURES, D_MODEL]
            ),
            "labels": label_meta,
        },
    }
    summary_path = args.out / "summary.json"
    rankings_path = args.out / "rankings_top_bottom_100.json"
    scores_path = args.out / "feature_scores.npz"
    report_path = args.out / "report.md"
    write_json_atomic(summary_path, summary, indent=2)
    write_json_atomic(rankings_path, rankings, indent=2)
    atomic_savez(scores_path, score_arrays)
    report_path.write_text(build_report(summary, rankings), encoding="utf-8")
    output = {
        "summary": source_record(summary_path),
        "rankings": source_record(rankings_path),
        "scores": source_record(scores_path),
        "report": source_record(report_path),
    }
    write_json_atomic(args.out / "outputs.json", output, indent=2)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "families": {name: value["n_pairs"] for name, value in family_summaries.items()},
                "orientation_error": orientation_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
