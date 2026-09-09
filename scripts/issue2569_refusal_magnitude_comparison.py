"""Compare three fixed L19 change magnitudes against archived refusal-rate changes.

The mapped score is ||delta_context @ A||, NOT gain normalized by context norm.
No new model calls or map fit; observed-answer norm is a reference, not a ceiling.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import issue2569_refusal_category_validation as base  # noqa: E402

NAMES = ("context_norm", "mapped_norm", "observed_answer_norm")
REPO = Path(__file__).resolve().parent.parent
PRIOR = REPO / "eval_results/issue_2569/followup_refusal_category_validation_20260909_v3"


def exact_count_gaps(rates_a: np.ndarray, rates_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover verified ten-draw counts so arithmetic noise cannot split rank ties."""
    rates = np.stack([rates_a, rates_b], axis=1)
    if not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1)):
        raise ValueError("Invalid archived refusal rate")
    counts = np.rint(10 * rates).astype(int)
    if not np.allclose(10 * rates, counts, atol=1e-12, rtol=0):
        raise ValueError("Archived rates do not lie on the verified ten-draw count grid")
    signed_counts = counts[:, 0] - counts[:, 1]
    return signed_counts, np.abs(signed_counts) / 10


def row_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis with explicit undefined constants."""
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    y = y - np.nanmean(y, axis=-1, keepdims=True)
    den = np.sqrt(np.nansum(x * x, axis=-1) * np.nansum(y * y, axis=-1))
    result = np.full(den.shape, np.nan)
    np.divide(np.nansum(x * y, axis=-1), den, out=result, where=den > 0)
    return result


def category_adjusted_rho(x: np.ndarray, y: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Pearson of global ranks after subtracting category-specific rank means.

    Ranks and category means are recomputed inside each bootstrap resample. This is
    partial Spearman controlling for category indicators, not a fitted safety probe.
    """
    xr, yr = rankdata(x, axis=-1, nan_policy="omit"), rankdata(y, axis=-1, nan_policy="omit")
    for c in np.unique(categories[categories != "__padding__"]):
        member = (categories == c) & np.isfinite(xr) & np.isfinite(yr)
        count = member.sum(axis=-1, keepdims=True)
        for values in (xr, yr):
            total = np.where(member, values, 0).sum(axis=-1, keepdims=True)
            mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            values -= np.where(member, mean, 0)
    return row_correlation(xr, yr)


def finite_summary(point: float, draws: np.ndarray | None) -> dict:
    """Report undefined correlations and bootstrap exclusions instead of coercing."""
    if not np.isfinite(point):
        return {"rho": None, "ci95": None, "reason": "constant outcome or score"}
    if draws is None:
        return {"rho": float(point), "ci95": None, "reason": "only one primary cluster"}
    valid = np.isfinite(draws)
    return {
        "rho": float(point),
        "ci95": np.quantile(draws[valid], [0.025, 0.975]).tolist(),
        "valid_bootstrap": int(valid.sum()),
        "undefined_bootstrap": int((~valid).sum()),
    }


def compare_scores(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, categories: np.ndarray, adjusted: bool = False
) -> tuple[dict, np.ndarray | None]:
    """Three correlations and paired mapped-minus-context difference on identical draws."""
    assert x.shape == (len(y), 3) and np.isfinite(x).all() and np.isfinite(y).all()
    point = np.array(
        [
            float(
                category_adjusted_rho(x[:, j], y, categories)
                if adjusted
                else base.rho_rows(x[:, j], y)
            )
            for j in range(3)
        ]
    )
    draws = None
    if len(np.unique(groups)) >= 2 and np.ptp(y) > 0:
        idx = base.cluster_indices(groups)
        by = np.where(idx >= 0, y[idx], np.nan)
        bc = np.where(idx >= 0, categories[idx], "__padding__")
        draws = np.stack(
            [
                category_adjusted_rho(np.where(idx >= 0, x[idx, j], np.nan), by, bc)
                if adjusted
                else base.rho_rows(np.where(idx >= 0, x[idx, j], np.nan), by)
                for j in range(3)
            ],
            axis=1,
        )
    result = {
        name: finite_summary(point[j], None if draws is None else draws[:, j])
        for j, name in enumerate(NAMES)
    }
    delta = point[1] - point[0]
    interval = finite_summary(delta, None if draws is None else draws[:, 1] - draws[:, 0])
    interval["delta_rho"] = interval.pop("rho")
    return {
        "n": len(y),
        "n_clusters": len(np.unique(groups)),
        "scores": result,
        "mapped_minus_context": interval,
        "category_adjusted": adjusted,
    }, draws


def calibrate_lofo(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list]:
    """Batched train-only OLS of outcome on each scalar score with intercept, no clipping."""
    unique = np.unique(groups)
    test = unique[:, None] == groups[None, :]
    train = ~test
    counts = train.sum(axis=1)
    assert np.all(counts >= 2)
    xmean = train @ x / counts[:, None]
    ymean = train @ y / counts
    xcenter = x[None, :, :] - xmean[:, None, :]
    ycenter = y[None, :] - ymean[:, None]
    variance = np.sum(train[:, :, None] * xcenter * xcenter, axis=1)
    if np.any(variance <= 0):
        raise ValueError("Constant scalar in a training fold; cannot estimate slope")
    slope = np.sum(train[:, :, None] * xcenter * ycenter[:, :, None], axis=1) / variance
    intercept = ymean[:, None] - slope * xmean
    predictions = np.sum(
        test[:, :, None] * (intercept[:, None, :] + slope[:, None, :] * x[None, :, :]), axis=0
    )
    baseline = np.sum(test * ymean[:, None], axis=0)
    folds = [
        {
            "held_out": str(g),
            "test_indices": np.flatnonzero(test[i]).tolist(),
            "train_indices": np.flatnonzero(train[i]).tolist(),
            "slopes": slope[i].tolist(),
            "intercepts": intercept[i].tolist(),
            "training_mean": float(ymean[i]),
        }
        for i, g in enumerate(unique)
    ]
    return predictions, baseline, folds


def predictive_metrics(predictions: np.ndarray, y: np.ndarray, baseline: np.ndarray) -> dict:
    """Pooled held-out R2/MAE and squared-error improvement over train-mean predictions."""
    sst = float(np.sum((y - y.mean()) ** 2))
    baseline_sse = float(np.sum((baseline - y) ** 2))
    scores = {}
    for j, name in enumerate((*NAMES, "training_mean_baseline")):
        pred = predictions[:, j] if j < 3 else baseline
        sse = float(np.sum((pred - y) ** 2))
        scores[name] = {
            "r2": 1 - sse / sst if sst > 0 else None,
            "mae": float(np.mean(abs(pred - y))),
            "mse_skill_vs_training_mean": 1 - sse / baseline_sse if baseline_sse > 0 else None,
            "n_predictions_outside_0_1": int(np.sum((pred < 0) | (pred > 1))),
        }
    return scores


def main() -> None:
    """Validate frozen sources, checkpoint scalar inputs, analyze, and write fresh sentinel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started = datetime.now(UTC).isoformat()
    rows, dc, da, _, pred, _, _, _, provenance = base.load_data()
    prior_prov = json.loads((PRIOR / "input_provenance.json").read_text())
    old_sources = {r["path"]: r["sha256"] for r in prior_prov["sources"]}
    for source in provenance["sources"]:
        if source["path"] not in old_sources or old_sources[source["path"]] != source["sha256"]:
            raise ValueError(f"Frozen source differs from audited v3: {source['path']}")
    with (PRIOR / "perpair.jsonl").open() as handle:
        old_rows = {r["pair_id"]: r for r in map(json.loads, handle)}
    x = np.stack([np.linalg.norm(value, axis=1) for value in (dc, pred, da)], axis=1)
    signed_counts, y = exact_count_gaps(
        np.array([r["rate_a"] for r in rows]), np.array([r["rate_b"] for r in rows])
    )
    paths = json.loads(base.MANIFEST.read_text())
    judge = json.loads(
        Path(paths["issue2617_svmp/raw_completions/judge/judge_scores.json"]).read_text()
    )
    for i, row in enumerate(rows):
        ja, jb = [judge["per_context"][row[key]] for key in ("a", "b")]
        assert ja["n_valid"] == jb["n_valid"] == 10
        assert signed_counts[i] == ja["n_refused"] - jb["n_refused"]
    categories = np.array([r["pair_class"] for r in rows])
    families = np.array([r["artifact_family_id"] for r in rows])
    groups = np.where(categories == "xstest", "ALL_XSTEST", families)
    scalar_rows = []
    for i, row in enumerate(rows):
        prior = old_rows[row["pair_id"]]
        assert np.isclose(x[i, 0], prior["context_norm"], rtol=1e-12)
        assert np.isclose(x[i, 1], prior["normalized_gain"] * prior["context_norm"], rtol=1e-12)
        assert np.isclose(y[i], prior["absolute_refusal_gap"], rtol=0, atol=1e-12)
        scalar_rows.append(
            {
                "pair_id": row["pair_id"],
                "pair_class": row["pair_class"],
                "artifact_family_id": row["artifact_family_id"],
                "primary_cluster": str(groups[i]),
                "absolute_refusal_gap": float(y[i]),
                "signed_refusal_count_gap": int(signed_counts[i]),
                "rate_a": row["rate_a"],
                "rate_b": row["rate_b"],
                **{name: float(x[i, j]) for j, name in enumerate(NAMES)},
            }
        )
    (args.out / "perpair.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scalar_rows))
    provenance.update(
        {
            "analysis_source": base.source(Path(__file__)),
            "prior_provenance": base.source(PRIOR / "input_provenance.json"),
            "prior_scalar_inputs": base.source(PRIOR / "perpair.jsonl"),
            "all_v3_source_hashes_identical": True,
            "seed": base.SEED,
            "n_bootstrap": base.N_BOOT,
            "bootstrap_recipe": "base.cluster_indices: sorted clusters, whole-cluster draws with replacement, padded rows; SAME draw matrix for all three scores and their paired differences",
            "working_source_pin": "Exact source SHA256 authoritative; script is newly authored on prior base commit.",
        }
    )
    base.dump(args.out / "input_provenance.json", provenance)
    old_y = np.array([r["absolute_refusal_gap"] for r in rows])
    old_signed = np.array([r["signed_refusal_gap"] for r in rows])
    signed_y = signed_counts / 10
    tie_audit = {
        "old_unique_absolute_gaps": len(np.unique(old_y)),
        "corrected_unique_absolute_gaps": len(np.unique(y)),
        "counts_reconciled_against_archived_n_refused": True,
        "max_absolute_numeric_change": float(np.max(abs(y - old_y))),
        "flip_membership_identical": bool(np.array_equal(y >= 0.5, old_y >= 0.5)),
        "prior_point_estimate_sensitivity": {},
    }
    for key in (
        "kernel_share",
        "normalized_gain",
        "predicted_refusal_LOFO",
        "read_refusal_LOFO",
        "kernel_refusal_LOFO",
        "identity_refusal_LOFO",
    ):
        values = np.array([old_rows[r["pair_id"]][key] for r in rows])
        use_signed = "refusal_LOFO" in key
        tie_audit["prior_point_estimate_sensitivity"][key] = {
            "old_rho": float(base.rho_rows(values, old_signed if use_signed else old_y)),
            "count_corrected_rho": float(base.rho_rows(values, signed_y if use_signed else y)),
        }
    base.dump(args.out / "numeric_tie_audit.json", tie_audit)
    print("Validated frozen sources and saved 124 scalar input rows", flush=True)
    primary, draws = compare_scores(x, y, groups, categories)
    assert draws is not None
    base.dump(
        args.out / "primary_bootstrap_correlations.json",
        {
            "columns": list(NAMES),
            "draws": [[float(v) if np.isfinite(v) else None for v in r] for r in draws],
        },
    )
    base.dump(args.out / "primary.json", primary)
    print("PRIMARY " + json.dumps(primary), flush=True)
    subsets = {
        "original108": categories != "verb_harm",
        "constructed88": categories != "xstest",
        "xstest36": categories == "xstest",
    }
    subset_results = {}
    for name, mask in subsets.items():
        subset_results[name], _ = compare_scores(x[mask], y[mask], groups[mask], categories[mask])
    by_category = {}
    for c in np.unique(categories):
        mask = categories == c
        by_category[c], _ = compare_scores(x[mask], y[mask], families[mask], categories[mask])
        by_category[c]["bootstrap_unit"] = (
            "item (semantic families unavailable)" if c == "xstest" else "semantic family"
        )
    adjusted, _ = compare_scores(x, y, groups, categories, adjusted=True)
    predicted, baseline, folds = calibrate_lofo(x, y, groups)
    base.dump(
        args.out / "calibration_folds.json",
        {
            "score_order": list(NAMES),
            "folds": folds,
            "predictions": predicted.tolist(),
            "training_mean_predictions": baseline.tolist(),
        },
    )
    calibration = {"all124": predictive_metrics(predicted, y, baseline)}
    for name, mask in subsets.items():
        calibration[name] = predictive_metrics(predicted[mask], y[mask], baseline[mask])
    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "n_pairs": len(rows),
        "primary": primary,
        "subsets": subset_results,
        "by_category": by_category,
        "category_adjusted": adjusted,
        "leave_family_out_calibration": calibration,
        "definitions": {
            "context_norm": "||delta c||",
            "mapped_norm": "||delta c @ A||; numerator, NOT ||delta c @ A||/||delta c||",
            "observed_answer_norm": "||mean answer_a - mean answer_b||, not mean norm of paired rollout differences",
            "outcome": "abs(n_refused_a-n_refused_b)/10; archived counts checked against ten-draw rates; exact count grid preserves ties",
            "row_action": "A=diag(1/xsd)@W; paired learned bias cancels",
        },
        "limitations": [
            "Exploratory secondary analysis of one frozen bank; no new on-policy replication.",
            "Observed-answer norm is an empirical reference, not a mathematical ceiling and not a context-only predictor.",
            "Primary uncertainty treats all XSTest items as one corpus cluster; within-XSTest descriptive intervals resample items.",
            "Primary mapped-minus-context comparison is prespecified; other strata and adjustments are descriptive sensitivity checks.",
            "Category-adjusted association correlates global ranks after category-mean subtraction; does not control all topic/length confounds.",
            "Scalar OLS calibrations are disjoint at semantic-family level; predictions are not clipped to [0,1].",
            "No high-dimensional probe or new representation mapping fitted; no inference about jailbreak framing or causality.",
            "Outcome ties corrected from archived integer refusal counts; no labels or flip memberships changed. numeric_tie_audit.json records tiny changes to prior v3 point estimates without refitting axes.",
        ],
    }
    base.dump(args.out / "summary.json", summary)
    lines = [
        "# Does mapped-change magnitude improve refusal-change prediction?",
        "",
        "Outcome: absolute difference in archived refusal rates. All 124 fixed layer-19 pairs; no selection on observed flips. Mapped magnitude is the unnormalized numerator.",
        "",
        "| Score | Spearman rho | Paired family-bootstrap 95% CI | LOFO linear R2 | LOFO MAE |",
        "|---|---:|---|---:|---:|",
    ]
    for name in NAMES:
        r = primary["scores"][name]
        cv = calibration["all124"][name]
        lines.append(
            f"| {name} | {r['rho']:.4f} | {r['ci95']} | {cv['r2']:.4f} | {cv['mae']:.4f} |"
        )
    lines += [
        "",
        f"Primary paired difference, mapped minus context: {primary['mapped_minus_context']}.",
        f"Training-mean baseline: {calibration['all124']['training_mean_baseline']}.",
        "",
        "## Category-adjusted rank association",
        "",
        json.dumps(adjusted, indent=2),
        "",
        "## Category-specific rank association",
        "",
        "| Category | n | Context | Mapped | Observed answer |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, r in by_category.items():
        vals = [
            str(None if r["scores"][n]["rho"] is None else round(r["scores"][n]["rho"], 4))
            for n in NAMES
        ]
        lines.append(f"| {c} | {r['n']} | {' | '.join(vals)} |")
    lines += [
        "",
        "Undefined correlations indicate constant outcomes, not zero association.",
        "",
        "## Limitations",
        "",
    ] + ["- " + s for s in summary["limitations"]]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    base.dump(
        args.out / "completion.json",
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "exit_code": 0,
            "outputs": [base.source(p) for p in sorted(args.out.iterdir()) if p.is_file()],
        },
    )
    print("COMPLETE " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
