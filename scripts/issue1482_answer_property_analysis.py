"""Matched feature recoverability and conditional property comparisons for #1482.

Concordance is the paper's pairwise rank statistic; 0.5 means no ordering.
Uncertainty resamples SAE features within class and fixed matching cells,
conditional on the bank, fitted maps, labels, and observed strata. These are
descriptive associations, not causal mediation or question-sampling intervals.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import issue1482_shapley_blocks as SB  # noqa: E402
import numpy as np  # noqa: E402
from issue1482_answer_property_ceiling import atomic_npz, digest, log, provenance  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
RUN = Path("/home/thomasjiralerspong/.codex/research/answer-property-ceiling-20260907")
OUT = RUN / "analysis"
SEED = 1482
N_BOOT = 1000  # Monte Carlo precision setting; seed follows the parent analysis.
AXES = {
    "abstraction": ("token_surface", "lexical_semantic", "abstract_contextual"),
    "content_type": ("topic", "task_format", "entity", "syntax", "operation"),
    "speaker_property": ("identity_disposition", "register_style", "language", "none"),
}


def quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Parent percentile-edge binning; ties remain together rather than split."""
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("invalid conditioning values")
    return np.digitize(
        values, np.percentile(values, np.linspace(0, 100, bins + 1))[1:-1]
    )


def nested_bins(first: np.ndarray, second: np.ndarray, inner: int) -> np.ndarray:
    """Equal-count recovery bins within the original activity strata."""
    out = np.empty(len(first), dtype=np.int64)
    for group in np.unique(first):
        mask = first == group
        out[mask] = group * inner + quantile_bins(second[mask], inner)
    return out


def concordance_bootstrap(
    outcomes: np.ndarray,
    positive: np.ndarray,
    cells: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Exact pair-weighted AUROC and ordinary stratified feature bootstrap.

    Each draw samples features with replacement within class x matching cell.
    The SAME feature weights are used for all outcome columns, preserving paired
    readouts. Sort once; weighted cumulative ranks avoid resampling/sorting fits.
    """
    y = np.asarray(outcomes, np.float64)
    g = np.asarray(positive, bool)
    if y.ndim == 1:
        y = y[:, None]
    if len(y) != len(g) or len(g) != len(cells) or not np.isfinite(y).all():
        raise ValueError("invalid concordance population")
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_boot, y.shape[1]), dtype=np.float64)
    point = np.zeros(y.shape[1], dtype=np.float64)
    pairs, used_pos, used_neg, used_cells = 0, 0, 0, 0
    for cell in np.unique(cells):
        a = np.flatnonzero((cells == cell) & g)
        b = np.flatnonzero((cells == cell) & ~g)
        if len(a) == 0 or len(b) == 0:
            continue
        pairs += len(a) * len(b)
        used_pos += len(a)
        used_neg += len(b)
        used_cells += 1
        prepared = []
        for col in range(y.shape[1]):
            order = np.argsort(y[b, col], kind="stable")
            negative = y[b[order], col]
            left = np.searchsorted(negative, y[a, col], side="left")
            right = np.searchsorted(negative, y[a, col], side="right")
            point[col] += ((left + right) / 2).sum()
            prepared.append((order, left, right))
        for start in range(0, n_boot, 16):
            stop = min(n_boot, start + 16)
            wa = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=stop - start)
            wb = rng.multinomial(len(b), np.full(len(b), 1 / len(b)), size=stop - start)
            for col, (order, left, right) in enumerate(prepared):
                cumulative = np.pad(np.cumsum(wb[:, order], axis=1), ((0, 0), (1, 0)))
                wins = (cumulative[:, left] + cumulative[:, right]) / 2
                draws[start:stop, col] += np.sum(wa * wins, axis=1)
    if pairs == 0:
        raise ValueError("no overlapping category cells")
    draws /= pairs
    point /= pairs
    return {
        "point": point,
        "ci95": np.percentile(draws, [2.5, 97.5], axis=0).T if n_boot else None,
        "draws": draws,
        "n_pairs": pairs,
        "n_positive": int(g.sum()),
        "n_negative": int((~g).sum()),
        "n_positive_matched": used_pos,
        "n_negative_matched": used_neg,
        "n_cells_matched": used_cells,
        "arm_differences_vs_first": point - point[0],
        "arm_differences_vs_first_ci95": np.percentile(
            draws - draws[:, :1], [2.5, 97.5], axis=0
        ).T
        if n_boot
        else None,
    }


def jsonable(obj):
    """Convert analysis records without hiding nonfinite numbers as zeros."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def distribution(values: np.ndarray) -> dict:
    """Readable, robust distribution summary including negative-R2 targets."""
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("invalid category distribution")
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "fraction_positive": float(np.mean(values > 0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def comparisons(
    y: np.ndarray,
    activities: np.ndarray,
    masks: dict,
    pairs: list,
    arm_names: list[str],
    family: str,
) -> dict:
    """Prespecified raw/activity/activity+recoverability contrasts with overlap."""
    first = quantile_bins(activities, 10)  # paper's established activity deciles
    joint = nested_bins(first, y[:, 0], 10)
    result = {}
    for label, positive_name, negative_name in pairs:
        positive, negative = masks[positive_name], masks[negative_name]
        overlap = positive & negative
        use = positive ^ negative
        if not use.any():
            raise ValueError(f"empty comparison: {label}")
        record = {
            "positive": positive_name,
            "negative": negative_name,
            "excluded_overlap": int(overlap.sum()),
            "arms": arm_names,
        }
        for condition, cells in (
            ("raw", np.zeros(len(y), int)),
            ("activity", first),
            ("activity_and_observed", joint),
        ):
            stats = concordance_bootstrap(y[use], positive[use], cells[use])
            atomic_npz(
                OUT / f"bootstrap_{family}_{label}_{condition}.npz",
                draws=stats.pop("draws"),
            )
            record[condition] = jsonable(stats)
        # Matching-resolution sensitivity; fixed before reading fitted outcomes.
        sensitivity = {}
        for inner in (5, 20):
            cells = nested_bins(first, y[:, 0], inner)
            stats = concordance_bootstrap(y[use], positive[use], cells[use], n_boot=0)
            stats.pop("draws")
            sensitivity[str(inner)] = jsonable(stats)
        record["observed_bin_sensitivity"] = sensitivity
        result[label] = record
        log(f"[analysis] {family}/{label} completed")
    return result


def regular() -> dict:
    """Full dictionary and exact original paper population, explicit labels."""
    complete = json.loads((RUN / "observed_answer/complete.json").read_text())
    source = REPO / "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy"
    SB.PROJECT_ROOT = (
        REPO  # Explicit bank root; sparse worktree omits binary v2 covariates.
    )
    SB.COV_NPZ = "eval_results/issue_1482/predictor_battery/fullwidth_covariates_v2.npz"
    inp = SB.load_inputs(source)
    context, cov, labels = inp["r2"], inp["cov"], inp["labels"]
    with np.load(RUN / "observed_answer/perfeature.npz") as z:
        if not np.array_equal(z["feat_ids"], np.arange(131072)):
            raise ValueError("full-width feature registry changed")
        observed = z["r2"]
        observed_ss_tot = z["ss_tot"]
    parent_perfeature = REPO / "data/issue_1482/densesae_dl/ridge__mean_perfeature.npz"
    with np.load(parent_perfeature) as z:
        if not np.array_equal(z["feat_ids"], np.arange(131072)):
            raise ValueError("parent full-width feature registry differs")
        if not np.array_equal(z["r2"], context, equal_nan=True):
            raise ValueError("paper R2 source differs from original context artifact")
        if not np.allclose(z["ss_tot"], observed_ss_tot, rtol=1e-9, atol=1e-12):
            raise ValueError("observed/context held-out target variances differ")
        variance_parity = float(np.max(np.abs(z["ss_tot"] - observed_ss_tot)))
    parent = json.loads(
        (
            REPO
            / "eval_results/issue_1482/predictor_battery/shapley_blocks_densesae_ridge_k24.json"
        ).read_text()
    )
    reps = [r["representative"] for r in parent["full_sample"]["representatives"]]
    paper = (
        np.isfinite(context)
        & np.isfinite(cov["firing_freq_per_token"])
        & np.isfinite(cov["activity"])
    )
    for key in reps:
        paper &= np.isfinite(cov[key])
    keep = paper & np.isfinite(observed)
    names = ["observed_answer", "context"]
    y = np.column_stack([observed[keep], context[keep]])
    masks, categories, coverage = {}, {}, {}
    for axis, levels in AXES.items():
        counts = {
            str(value): int(np.sum(labels[axis][keep] == value))
            for value in np.unique(labels[axis][keep])
        }
        coverage[axis] = counts
        for level in levels:
            key = f"{axis}:{level}"
            masks[key] = labels[axis][keep] == level
            if masks[key].any():
                categories[key] = {
                    arm: distribution(y[masks[key], col])
                    for col, arm in enumerate(names)
                }
    for code, level in (
        (1, "promoting"),
        (2, "suppressing"),
        (3, "partition"),
        (0, "other"),
    ):
        key = f"logit:{level}"
        masks[key] = cov["promoting_class"][keep] == code
        categories[key] = {
            arm: distribution(y[masks[key], col]) for col, arm in enumerate(names)
        }
    pairs = [
        (
            "identity_vs_topic",
            "speaker_property:identity_disposition",
            "content_type:topic",
        ),
        (
            "abstract_vs_token",
            "abstraction:abstract_contextual",
            "abstraction:token_surface",
        ),
        (
            "abstract_vs_lexical",
            "abstraction:abstract_contextual",
            "abstraction:lexical_semantic",
        ),
        ("register_vs_topic", "speaker_property:register_style", "content_type:topic"),
        ("language_vs_topic", "speaker_property:language", "content_type:topic"),
        ("format_vs_topic", "content_type:task_format", "content_type:topic"),
        ("promoting_vs_suppressing", "logit:promoting", "logit:suppressing"),
    ]
    atomic_npz(
        OUT / "regular_feature_table.npz",
        feat_ids=np.flatnonzero(keep),
        observed_answer_r2=y[:, 0],
        context_r2=y[:, 1],
        activity=cov["activity"][keep],
        firing_freq_per_token=cov["firing_freq_per_token"][keep],
        promoting_class=cov["promoting_class"][keep],
        **{f"label__{axis}": labels[axis][keep].astype(str) for axis in AXES},
    )
    result = {
        "fit": complete,
        "dictionary_width": 131072,
        "paper_population": int(paper.sum()),
        "matched_population": int(keep.sum()),
        "undefined_observed": int((~np.isfinite(observed)).sum()),
        "undefined_context": int((~np.isfinite(context)).sum()),
        "label_coverage": coverage,
        "categories": categories,
        "target_variance_parity_max_abs": variance_parity,
        "spearman_observed_context": float(spearmanr(y[:, 0], y[:, 1]).statistic),
        "comparisons": comparisons(
            y, cov["firing_freq_per_token"][keep], masks, pairs, names, "regular"
        ),
        "source_sha256": {
            "context_r2": digest(source),
            "context_perfeature": digest(parent_perfeature),
            "observed_r2": digest(RUN / "observed_answer/perfeature.npz"),
            "covariates": digest(REPO / SB.COV_NPZ),
            "labels": digest(REPO / SB.MATRIX),
            "discrete_covariates": digest(REPO / SB.DISC_NPZ),
        },
    }
    write_json_atomic(OUT / "regular.json", jsonable(result))
    return result


def matryoshka() -> dict:
    """Original SAE-context and matched dense-context remain separate arms."""
    base = RUN / "matryoshka"
    with np.load(base / "registry.npz") as z:
        ids, tier, activity = (z[k] for k in ("feat_ids", "tier", "activity"))
    columns = []
    for path in (
        base / "observed_answer/perfeature.npz",
        base / "context/perfeature.npz",
        REPO / "eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz",
    ):
        with np.load(path) as z:
            if not np.array_equal(ids, z["feat_ids"]):
                raise ValueError("Matryoshka feature columns differ")
            columns.append(z["r2"])
    values = np.column_stack(columns)
    keep = np.isfinite(values).all(1) & np.isfinite(activity)
    values, tier, activity, ids = values[keep], tier[keep], activity[keep], ids[keep]
    names = ["observed_answer", "dense_context", "original_sae_context"]
    masks = {f"tier_{int(t)}": tier == t for t in np.unique(tier)}
    keys = sorted(masks)
    if len(keys) != 3:
        raise ValueError("expected three Matryoshka tiers")
    pairs = [
        ("coarse_vs_fine", keys[0], keys[2]),
        ("coarse_vs_middle", keys[0], keys[1]),
        ("middle_vs_fine", keys[1], keys[2]),
    ]
    categories = {
        key: {arm: distribution(values[mask, col]) for col, arm in enumerate(names)}
        for key, mask in masks.items()
    }
    atomic_npz(
        OUT / "matryoshka_feature_table.npz",
        feat_ids=ids,
        tier=tier,
        activity=activity,
        **{f"{arm}_r2": values[:, col] for col, arm in enumerate(names)},
    )
    result = {
        "layer": 20,
        "original_panel": len(keep),
        "matched_panel": int(keep.sum()),
        "categories": categories,
        "comparisons": comparisons(values, activity, masks, pairs, names, "matryoshka"),
        "fits": {
            arm: json.loads((base / arm / "complete.json").read_text())
            for arm in ("observed_answer", "context", "context_original")
        },
    }
    write_json_atomic(OUT / "matryoshka.json", jsonable(result))
    return result


def row_bootstrap_matryoshka(n_boot: int = N_BOOT) -> dict:
    """Resample complete held-out answers, preserving cross-feature dependence.

    Pooled R2 within a fixed tier is distinct from median per-feature R2. This
    complement checks uncertainty across answers for the two newly fitted arms;
    the archived SAE-context arm has no available per-answer prediction bank.
    """
    base = RUN / "matryoshka"
    with np.load(base / "registry.npz") as z:
        te, tiers, corpus = z["te"], z["tier"], z["corpus"][z["te"]]
    targets = np.load(base / "targets.npy", mmap_mode="r")
    predictions = [
        np.load(base / arm / "predictions.npy", mmap_mode="r")
        for arm in ("observed_answer", "context")
    ]
    rng = np.random.default_rng(SEED)
    weights = np.zeros((n_boot, len(te)), dtype=np.int16)
    for code in np.unique(corpus):
        idx = np.flatnonzero(corpus == code)
        weights[:, idx] = rng.multinomial(
            len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot
        )
    draws, points = [], []
    for tier in np.unique(tiers):
        select = np.flatnonzero(tiers == tier)
        true = np.asarray(targets[np.ix_(te, select)], dtype=np.float64)
        residual = np.column_stack(
            [np.square(true - pred[:, select]).sum(1) for pred in predictions]
        )
        y2 = np.square(true).sum(1)
        tss = y2.sum() - len(te) * np.square(true.mean(0)).sum()
        points.append(1 - residual.sum(0) / tss)
        group_draws = []
        for start in range(0, n_boot, 32):
            w = weights[start : start + 32].astype(np.float64)
            mean = (w @ true) / len(te)
            denominator = w @ y2 - len(te) * np.square(mean).sum(1)
            if np.any(denominator <= 0):
                raise ValueError("undefined resampled tier variance")
            group_draws.append(1 - (w @ residual) / denominator[:, None])
        draws.append(np.concatenate(group_draws))
        log(f"[row-bootstrap] tier {tier} completed")
    draws = np.stack(draws, axis=1)
    points = np.asarray(points)
    atomic_npz(
        OUT / "matryoshka_row_bootstrap.npz",
        draws=draws,
        point=points,
        tier=np.unique(tiers),
        weights=weights,
        test_rows=te,
    )
    result = {
        "metric": "pooled R2 within fixed original tier",
        "point": points,
        "ci95": np.percentile(draws, [2.5, 97.5], axis=0),
        "context_minus_observed": points[:, 1] - points[:, 0],
        "context_minus_observed_ci95": np.percentile(
            draws[:, :, 1] - draws[:, :, 0], [2.5, 97.5], axis=0
        ),
        "coarse_minus_fine": points[0] - points[-1],
        "coarse_minus_fine_ci95": np.percentile(
            draws[:, 0] - draws[:, -1], [2.5, 97.5], axis=0
        ),
        "coarse_fine_difference_in_context_minus_observed": (
            points[0, 1] - points[0, 0]
        )
        - (points[-1, 1] - points[-1, 0]),
        "coarse_fine_difference_in_context_minus_observed_ci95": np.percentile(
            (draws[:, 0, 1] - draws[:, 0, 0]) - (draws[:, -1, 1] - draws[:, -1, 0]),
            [2.5, 97.5],
        ),
        "n_test": len(te),
        "n_boot": n_boot,
        "seed": SEED,
        "scope": "complete-answer bootstrap stratified by original corpus; feature panel, fitted readouts and selected penalties fixed; original SAE-context predictions unavailable",
    }
    write_json_atomic(OUT / "matryoshka_row_bootstrap.json", jsonable(result))
    return result


def write_tables(results: dict) -> None:
    """Compact side-by-side group summaries for inspection; no figure required."""
    with (OUT / "category_summary.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "category",
                "arm",
                "n",
                "median_r2",
                "q25",
                "q75",
                "fraction_positive",
            ]
        )
        for family, result in results.items():
            for category, groups in result["categories"].items():
                for arm, stats in groups.items():
                    writer.writerow(
                        [
                            family,
                            category,
                            arm,
                            stats["n"],
                            stats["median"],
                            stats["q25"],
                            stats["q75"],
                            stats["fraction_positive"],
                        ]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("regular", "matryoshka", "both"), default="both"
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results = {}
    if args.family in ("regular", "both"):
        results["regular"] = regular()
    if args.family in ("matryoshka", "both"):
        results["matryoshka"] = matryoshka()
        row_bootstrap_matryoshka()
    write_tables(results)
    write_json_atomic(
        OUT / f"complete_{args.family}.json",
        {
            "families": list(results),
            "bootstrap_draws": N_BOOT,
            "seed": SEED,
            "metadata": provenance("property-analysis"),
            "script_sha256": digest(Path(__file__)),
            "uncertainty_scope": "feature resampling conditional on fixed fitted models, target bank, categories and matching cells; correlated features may make intervals too narrow",
        },
    )
