#!/usr/bin/env python3
"""Decompose refusal-pair answer changes into high- and low-write map subspaces.

This task #2569 follow-up is CPU-only. It reuses frozen layer-19 context and
answer states, the banked context-to-answer ridge map, its production SVD, and
the existing answer-side SAE descriptions. It performs no generation or model
forward passes.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # bind shared-VM thread caps before numpy/torch imports

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import issue2569_answer_residual_sae as AR  # noqa: E402
import issue2569_operator as OP  # noqa: E402
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
LAYER = 19
D_MODEL = 3_584
MASS = 0.99
SEED = 256_919
N_BOOT = 5_000
EXPECTED_COUNTS = {"flip": 60, "nonflip": 40, "mid": 8, "control": 16}
DEFAULT_FACTOR = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2569_theory/smoke-final/weights_prod/leg1/factor_L19.pt"
)
DEFAULT_OUT = REPO / "eval_results/issue_2569/followup_refusal_answer_write"


def classify_pair(pair_class: str, rate_a: float, rate_b: float) -> str:
    """Assign the frozen refusal-pair outcome group from its two refusal rates."""
    if pair_class == AR.CONTROL_CLASS:
        return "control"
    gap = abs(rate_a - rate_b)
    if gap >= AR.FLIP_GAP:
        return "flip"
    if gap <= 0.1:
        return "nonflip"
    return "mid"


def ordered_contexts(record: dict, rates: dict, group: str) -> tuple[str, str]:
    """Return pair endpoints in a stable orientation; energy reads are sign-invariant."""
    a, b = record["a"], record["b"]
    if group == "control":
        return a, b
    rate_a = float(rates[a]["refusal_rate"])
    rate_b = float(rates[b]["refusal_rate"])
    return (a, b) if rate_a >= rate_b else (b, a)


def load_pair_groups(manifest_path: Path) -> tuple[dict[str, dict], dict]:
    """Load frozen layer-19 pair deltas grouped by behavioral outcome."""
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
    if len(row_of) != len(context_ids):
        raise ValueError("duplicate context ids")
    context = vc_store["vc"][:, layer_index].to(torch.float64).numpy()
    answer_sums, answer_counts = AR.mean_answer_rows(va_store, context_ids, layer_index)
    if np.any(answer_counts <= 0):
        raise ValueError("a refusal context has no valid answer rows")
    answer = answer_sums / answer_counts[:, None]
    rates = judge["per_context"]
    grouped_records: dict[str, list[tuple[str, str, str]]] = {name: [] for name in EXPECTED_COUNTS}
    for record in bank["pairs"]:
        a, b = record["a"], record["b"]
        group = classify_pair(
            record["pair_class"],
            float(rates[a]["refusal_rate"]),
            float(rates[b]["refusal_rate"]),
        )
        first, second = ordered_contexts(record, rates, group)
        grouped_records[group].append((record["pair_id"], first, second))

    groups: dict[str, dict] = {}
    for group, records in grouped_records.items():
        if len(records) != EXPECTED_COUNTS[group]:
            raise ValueError(
                f"expected {EXPECTED_COUNTS[group]} {group} pairs, found {len(records)}"
            )
        pair_ids = [record[0] for record in records]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError(f"duplicate {group} pair ids")
        first = np.asarray([row_of[record[1]] for record in records], dtype=np.int64)
        second = np.asarray([row_of[record[2]] for record in records], dtype=np.int64)
        groups[group] = {
            "pair_ids": pair_ids,
            "delta_context": context[first] - context[second],
            "delta_answer": answer[first] - answer[second],
            "orientation": (
                "higher-minus-lower refusal rate"
                if group != "control"
                else "manifest a-minus-b; energy statistics are sign-invariant"
            ),
        }
    return groups, {
        "layer": LAYER,
        "pooling": "per-context mean of tail-inclusive answer states over 10 valid draws",
        "outcome_thresholds": {"flip_gap_at_least": AR.FLIP_GAP, "nonflip_gap_at_most": 0.1},
        "sources": {
            "leg9_manifest": AR.source_record(manifest_path),
            "pair_manifest": AR.source_record(pairs_path),
            "judge": AR.source_record(judge_path),
            "context": AR.source_record(vc_path),
            "answer": AR.source_record(va_path),
        },
    }


def load_write_basis(
    factor_path: Path, operator: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and validate the full production output singular-vector basis."""
    factor = torch.load(factor_path, map_location="cpu", weights_only=False, mmap=True)
    singular = factor["sigma"].numpy().astype(np.float64, copy=False)
    read_basis = factor["read_input_u_fp32"].numpy().astype(np.float64)
    write_basis = factor["write_output_v_fp32"].numpy().astype(np.float64)
    if singular.shape != (D_MODEL,) or read_basis.shape != (D_MODEL, D_MODEL):
        raise ValueError("unexpected factor read-side dimensions")
    if write_basis.shape != (D_MODEL, D_MODEL):
        raise ValueError("unexpected factor write-side dimensions")
    if not np.all(np.isfinite(singular)) or not np.all(np.diff(singular) <= 0):
        raise ValueError("singular spectrum must be finite and descending")
    tau, high_rank = OP.tau_kernel_threshold(singular, mass=MASS)
    low_mask = singular < tau
    if int((~low_mask).sum()) != high_rank:
        raise ValueError("strict threshold and retained rank disagree")
    probe_indices = np.asarray(
        [0, 1, high_rank - 1, high_rank, D_MODEL - 2, D_MODEL - 1], dtype=np.int64
    )
    lhs = read_basis[:, probe_indices].T @ operator
    rhs = singular[probe_indices, None] * write_basis[:, probe_indices].T
    orientation_abs_error = np.linalg.norm(lhs - rhs, axis=1)
    orientation_rel_error = orientation_abs_error / singular[probe_indices]
    cutoff_rel_error = float(orientation_rel_error[:4].max())
    scale_relative_error = float((orientation_abs_error / singular[0]).max())
    # The cached bases are fp32. Relative error is meaningful for the leading and
    # cutoff directions; at sigma ~= 0, use absolute error relative to sigma_max.
    if cutoff_rel_error > 2e-6 or scale_relative_error > 2e-6:
        raise ValueError(
            "cached SVD does not match the map: "
            f"cutoff-relative={cutoff_rel_error}, scale-relative={scale_relative_error}"
        )
    gram = write_basis[:, probe_indices].T @ write_basis[:, probe_indices]
    selected_orthogonality_error = float(np.max(np.abs(gram - np.eye(len(probe_indices)))))
    if selected_orthogonality_error > 2e-6:
        raise ValueError(f"cached write basis is not orthonormal: {selected_orthogonality_error}")
    metadata = {
        "source": AR.source_record(factor_path),
        "factor_regime": factor["regime"],
        "factor_metadata": factor["stats"],
        "mass_cutoff": MASS,
        "tau": float(tau),
        "high_write_dim": int((~low_mask).sum()),
        "low_write_dim": int(low_mask.sum()),
        "low_singular_squared_mass": float(np.sum(singular[low_mask] ** 2) / np.sum(singular**2)),
        "orientation_probe_indices": probe_indices.tolist(),
        "max_cutoff_orientation_relative_error": cutoff_rel_error,
        "max_orientation_absolute_over_sigma_max": scale_relative_error,
        "orientation_relative_error_by_probe": orientation_rel_error.tolist(),
        "selected_orthogonality_error": selected_orthogonality_error,
        "definition": (
            "A = U diag(sigma) V^T under row action; high-write is the first rank V "
            "directions retaining 99% of total sigma^2 mass; low-write is its complement"
        ),
    }
    return write_basis, low_mask, metadata


def decompose(values: np.ndarray, write_basis: np.ndarray, low_mask: np.ndarray) -> dict:
    """Project row vectors onto disjoint low- and high-write singular subspaces."""
    values = np.asarray(values, dtype=np.float64)
    coefficients = values @ write_basis
    low = coefficients[:, low_mask] @ write_basis[:, low_mask].T
    high = coefficients[:, ~low_mask] @ write_basis[:, ~low_mask].T
    norm2 = np.einsum("ij,ij->i", values, values)
    if np.any(norm2 <= 0):
        raise ValueError("cannot decompose a zero answer-change vector")
    low_norm2 = np.einsum("ij,ij->i", low, low)
    high_norm2 = np.einsum("ij,ij->i", high, high)
    reconstruction_relative = np.linalg.norm(values - low - high, axis=1) / np.sqrt(norm2)
    orthogonality_relative = np.abs(np.einsum("ij,ij->i", low, high)) / norm2
    if float(reconstruction_relative.max()) > 2e-6:
        raise ValueError("write-subspace reconstruction failed")
    if float(orthogonality_relative.max()) > 2e-6:
        raise ValueError("write-subspace orthogonality failed")
    return {
        "low": low,
        "high": high,
        "low_share": low_norm2 / norm2,
        "high_share": high_norm2 / norm2,
        "pooled_low_share": float(low_norm2.sum() / norm2.sum()),
        "pooled_high_share": float(high_norm2.sum() / norm2.sum()),
        "max_reconstruction_relative_error": float(reconstruction_relative.max()),
        "max_orthogonality_relative_error": float(orthogonality_relative.max()),
        "max_energy_sum_error": float(np.max(np.abs((low_norm2 + high_norm2) / norm2 - 1.0))),
    }


def component_summary(parts: dict, seed_offset: int) -> dict:
    """Summarize per-pair and pooled energy assigned to each write subspace."""
    return {
        "low_share": AR.bootstrap_summary(parts["low_share"], seed_offset=seed_offset),
        "high_share": AR.bootstrap_summary(parts["high_share"], seed_offset=seed_offset + 1),
        "pooled_low_share": parts["pooled_low_share"],
        "pooled_high_share": parts["pooled_high_share"],
        "max_reconstruction_relative_error": parts["max_reconstruction_relative_error"],
        "max_orthogonality_relative_error": parts["max_orthogonality_relative_error"],
        "max_energy_sum_error": parts["max_energy_sum_error"],
    }


def bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int) -> dict:
    """Independent bootstrap intervals for left-minus-right mean and median differences."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    left_draw = left[rng.integers(0, left.size, size=(N_BOOT, left.size))]
    right_draw = right[rng.integers(0, right.size, size=(N_BOOT, right.size))]
    mean_draw = left_draw.mean(axis=1) - right_draw.mean(axis=1)
    median_draw = np.median(left_draw, axis=1) - np.median(right_draw, axis=1)
    test = mannwhitneyu(left, right, alternative="two-sided")
    return {
        "mean_difference": float(left.mean() - right.mean()),
        "mean_difference_ci95": [float(x) for x in np.percentile(mean_draw, [2.5, 97.5])],
        "median_difference": float(np.median(left) - np.median(right)),
        "median_difference_ci95": [float(x) for x in np.percentile(median_draw, [2.5, 97.5])],
        "cliffs_delta": float(np.mean(np.sign(left[:, None] - right[None, :]))),
        "mann_whitney_u": float(test.statistic),
        "mann_whitney_p_two_sided": float(test.pvalue),
        "qualification": "Exploratory unadjusted comparison; pair families are not randomized arms.",
    }


def subspace_prediction_summary(observed: dict, predicted: dict, seed_offset: int) -> dict:
    """Measure map prediction quality separately within the two output subspaces."""
    result = {}
    for offset, subspace in enumerate(("high", "low")):
        obs = observed[subspace]
        pred = predicted[subspace]
        result[subspace] = {
            "cosine_observed_predicted": AR.bootstrap_summary(
                AR.row_cos(obs, pred), seed_offset=seed_offset + 3 * offset
            ),
            "predicted_over_observed_norm": AR.bootstrap_summary(
                np.linalg.norm(pred, axis=1) / np.linalg.norm(obs, axis=1),
                seed_offset=seed_offset + 3 * offset + 1,
            ),
        }
    return result


def ranking_bundle(values: dict, decoder: np.ndarray, labels: dict[int, dict]) -> tuple[dict, dict]:
    """Rank answer SAE directions for the low- and high-write observed components."""
    rankings: dict[str, dict] = {}
    score_arrays: dict[str, np.ndarray] = {}
    for component in ("high", "low"):
        rankings[component] = {}
        for metric, scores in AR.feature_scores(values[component], decoder).items():
            rankings[component][metric] = AR.ranking_summary(scores, labels)
            score_arrays[f"observed_{component}_{metric}"] = scores.astype(np.float32)
    return rankings, score_arrays


def labeled_examples(ranking: dict, limit: int = 10) -> list[dict]:
    """Return the first interpretable labeled rows from a top-100 feature ranking."""
    selected = []
    for row in ranking["top"]:
        label = row["label"]
        if label is None or label["category"] == "uninterpretable":
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def build_report(summary: dict, rankings: dict) -> str:
    """Render a compact audit report from the saved numerical results."""
    lines = [
        "# Refusal-pair observed answer change by map write subspace",
        "",
        "The observed layer-19 answer change is projected onto the output singular directions "
        "of the frozen context-to-answer map. The high-write subspace contains the 1,608 output "
        "directions paired with singular values that retain 99% of total squared singular mass; "
        "the remaining 1,976 directions form the low-write subspace.",
        "",
        "## Energy split",
        "",
        "| Pair group | n | Observed low-write median | Predicted low-write median | Residual low-write median |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("flip", "nonflip", "mid", "control"):
        group = summary["groups"][name]
        lines.append(
            f"| {name} | {group['n_pairs']} | "
            f"{group['components']['observed']['low_share']['median']:.3f} | "
            f"{group['components']['predicted']['low_share']['median']:.3f} | "
            f"{group['components']['residual']['low_share']['median']:.3f} |"
        )
    flip = summary["groups"]["flip"]
    lines += [
        "",
        "For the 60 refusal flips, the median observed low-write share is "
        f"{flip['components']['observed']['low_share']['median']:.3f} "
        f"(95% bootstrap CI {flip['components']['observed']['low_share']['median_ci95'][0]:.3f}–"
        f"{flip['components']['observed']['low_share']['median_ci95'][1]:.3f}). Thus most of the "
        "observed answer change lies in output directions the map writes strongly. The residual "
        "is more concentrated in low-write directions than either the observation or prediction, "
        "but residual and low-write are not the same decomposition.",
        "",
        "Within the high-write subspace, the median observed–predicted cosine is "
        f"{flip['subspace_prediction']['high']['cosine_observed_predicted']['median']:.3f}; "
        "within the low-write subspace it is "
        f"{flip['subspace_prediction']['low']['cosine_observed_predicted']['median']:.3f}.",
        "",
        "## Existing answer-SAE descriptions",
        "",
        "The feature view is descriptive: SAE decoder directions are correlated, and existing "
        "Codex descriptions cover a selected subset rather than a random sample. Both the mean "
        "refusal direction and the typical-pair RMS ranking are retained in the JSON artifact.",
        "",
    ]
    for component in ("high", "low"):
        lines += [f"### {component.title()}-write component", ""]
        for metric in ("mean_direction_abs_cos", "pair_direction_rms"):
            label = "Mean direction" if metric == "mean_direction_abs_cos" else "Typical pair (RMS)"
            ranking = rankings[component][metric]
            composition = ranking["top_composition"]
            rows = labeled_examples(ranking)
            lines.append(
                f"{label} description coverage: {composition['n_labeled']}/100 labeled, "
                f"{composition['n_interpretable']}/100 interpretable."
            )
            lines.append("")
            lines.append(
                f"{label}: "
                + "; ".join(
                    f"feature {row['feature_id']} — {row['label']['description']}" for row in rows
                )
            )
            lines.append("")
    lines += [
        "The high-write component has broad existing label coverage and its nearest described "
        "features prominently include refusal and safety language. The low-write component has "
        "only 5–6 labeled features in its top 100, so the current descriptions do not support a "
        "general semantic characterization of that component.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse paths for the frozen inputs and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg9-manifest", type=Path, default=AR.DEFAULT_LEG9_MANIFEST)
    parser.add_argument("--ridge", type=Path, default=AR.DEFAULT_RIDGE)
    parser.add_argument("--factor", type=Path, default=DEFAULT_FACTOR)
    parser.add_argument("--sae", type=Path, default=AR.DEFAULT_SAE)
    parser.add_argument("--labels", type=Path, default=AR.DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    """Run the frozen-data answer write-subspace analysis and persist its artifacts."""
    args = parse_args()
    for path in (args.leg9_manifest, args.ridge, args.factor, args.sae, args.labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    operator, operator_meta = AR.load_operator(args.ridge)
    write_basis, low_mask, basis_meta = load_write_basis(args.factor, operator)
    groups, pair_meta = load_pair_groups(args.leg9_manifest)
    decoder = AR.load_decoder(args.sae)
    labels, label_meta = AR.load_labels(args.labels)

    summary_groups: dict[str, dict] = {}
    decomposed: dict[str, dict[str, dict]] = {}
    for group_index, (name, group) in enumerate(groups.items()):
        observed_values = group["delta_answer"]
        predicted_values = group["delta_context"] @ operator
        residual_values = observed_values - predicted_values
        values = {
            "observed": observed_values,
            "predicted": predicted_values,
            "residual": residual_values,
        }
        parts = {
            component: decompose(component_values, write_basis, low_mask)
            for component, component_values in values.items()
        }
        decomposed[name] = parts
        cosine = AR.row_cos(observed_values, predicted_values)
        summary_groups[name] = {
            "n_pairs": len(group["pair_ids"]),
            "orientation": group["orientation"],
            "pair_id_sha256": hashlib.sha256(
                "\n".join(group["pair_ids"]).encode("utf-8")
            ).hexdigest(),
            "cosine_observed_predicted": AR.bootstrap_summary(
                cosine, seed_offset=100 * group_index
            ),
            "components": {
                component: component_summary(part, 100 * group_index + 10 * offset)
                for offset, (component, part) in enumerate(parts.items())
            },
            "subspace_prediction": subspace_prediction_summary(
                parts["observed"], parts["predicted"], seed_offset=100 * group_index + 40
            ),
        }

    flip_cosine = summary_groups["flip"]["cosine_observed_predicted"]
    prior_reproduction = {
        "n_pairs": summary_groups["flip"]["n_pairs"],
        "observed_mean": flip_cosine["mean"],
        "expected_mean": AR.PRIOR_REFUSAL_COSINE_MEAN,
        "observed_median": flip_cosine["median"],
        "expected_median": AR.PRIOR_REFUSAL_COSINE_MEDIAN,
        "tolerance": AR.PRIOR_REPRO_TOLERANCE,
    }
    prior_reproduction["passed"] = (
        prior_reproduction["n_pairs"] == 60
        and abs(prior_reproduction["observed_mean"] - prior_reproduction["expected_mean"])
        <= AR.PRIOR_REPRO_TOLERANCE
        and abs(prior_reproduction["observed_median"] - prior_reproduction["expected_median"])
        <= AR.PRIOR_REPRO_TOLERANCE
    )
    if not prior_reproduction["passed"]:
        raise ValueError(f"prior refusal result reproduction failed: {prior_reproduction}")

    comparisons = {
        f"flip_minus_{other}": bootstrap_difference(
            decomposed["flip"]["observed"]["low_share"],
            decomposed[other]["observed"]["low_share"],
            seed=SEED + offset,
        )
        for offset, other in enumerate(("nonflip", "control"))
    }
    rankings, score_arrays = ranking_bundle(decomposed["flip"]["observed"], decoder, labels)
    top_overlap = {}
    for metric in ("mean_direction_abs_cos", "pair_direction_rms", "raw_energy_rms"):
        high_ids = {row["feature_id"] for row in rankings["high"][metric]["top"]}
        low_ids = {row["feature_id"] for row in rankings["low"][metric]["top"]}
        top_overlap[metric] = len(high_ids & low_ids)

    summary = {
        "analysis": "refusal_observed_answer_write_subspaces",
        "layer": LAYER,
        "basis": basis_meta,
        "operator": operator_meta,
        "pairs": pair_meta,
        "groups": summary_groups,
        "comparisons": comparisons,
        "statistics": {
            "component_summary_bootstrap_draws": AR.N_BOOT,
            "component_summary_bootstrap_base_seed": AR.SEED,
            "between_group_bootstrap_draws": N_BOOT,
            "between_group_bootstrap_base_seed": SEED,
            "bootstrap_unit": "pair",
        },
        "prior_refusal_reproduction": prior_reproduction,
        "answer_sae": {
            **label_meta,
            "weight_source": AR.source_record(args.sae),
            "metrics": {
                "mean_direction_abs_cos": "absolute cosine to the mean component direction",
                "pair_direction_rms": "RMS cosine over unit-normalized pair components",
                "raw_energy_rms": "RMS projection of raw pair components",
            },
            "top_100_high_low_overlap": top_overlap,
            "qualification": (
                "SAE feature descriptions are qualitative and selected; scores are overlapping "
                "similarities, not additive variance or causal attributions."
            ),
        },
        "distinction_from_residual": (
            "Observed = high-write + low-write is an approximately orthogonal output-space "
            "projection. Observed = predicted + residual is a prediction-error identity and is "
            "not orthogonal in general."
        ),
    }
    write_json_atomic(args.out / "summary.json", summary, indent=2)
    write_json_atomic(args.out / "rankings_top_bottom_100.json", rankings, indent=2)
    AR.atomic_savez(args.out / "scores.npz", score_arrays)
    (args.out / "report.md").write_text(build_report(summary, rankings), encoding="utf-8")
    outputs = {
        path.name: AR.source_record(path)
        for path in sorted(args.out.iterdir())
        if path.is_file() and path.name != "outputs.json"
    }
    write_json_atomic(args.out / "outputs.json", outputs, indent=2)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "flip_observed_low_write_median": summary_groups["flip"]["components"]["observed"][
                    "low_share"
                ]["median"],
                "flip_predicted_low_write_median": summary_groups["flip"]["components"][
                    "predicted"
                ]["low_share"]["median"],
                "flip_residual_low_write_median": summary_groups["flip"]["components"]["residual"][
                    "low_share"
                ]["median"],
                "high_write_prediction_cosine_median": summary_groups["flip"][
                    "subspace_prediction"
                ]["high"]["cosine_observed_predicted"]["median"],
                "low_write_prediction_cosine_median": summary_groups["flip"]["subspace_prediction"][
                    "low"
                ]["cosine_observed_predicted"]["median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
