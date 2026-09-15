#!/usr/bin/env python3
"""Repeat the published Matryoshka contrast with decoder-direction variance matching.

User-requested CPU reanalysis of #1482, 2026-09-14. No new model fit or generation.
Preserves the 16,384-feature panel, decoder-direction R2, binary contrasts,
pair-weighted concordance, minimum cell size, quantile convention and bootstrap
of issue1482_tier_concordance.py. Adds variance-only and crossed variance/activity
matching. Variance is Var(d_f.T @ mean_answer_state), on the original 6,000 score
rows, using unit decoder columns. This is not gated SAE activation variance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue1482_tier_concordance as parent  # noqa: E402
import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

ROOT = Path(__file__).resolve().parent.parent
SEED = parent.SEED
N_BOOT = parent.N_BOOT
BATCH = 32


def log(message: str) -> None:
    print(f"{dt.datetime.now(dt.UTC).isoformat()} {message}", flush=True)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def assign_strata(controls: list[np.ndarray], n_bins: int, shape: tuple) -> np.ndarray:
    """Cross quantile bins; each row is one independently rebinned bootstrap draw."""
    groups = np.zeros(shape, dtype=np.int64)
    for values in controls:
        edges = np.percentile(values, np.linspace(0, 100, n_bins + 1)[1:-1], axis=1)
        bins = (values[:, :, None] >= edges.T[:, None, :]).sum(axis=2)
        groups = groups * n_bins + bins
    return groups


def binary_batch(
    coding: np.ndarray, r2: np.ndarray, groups: np.ndarray, min_cell: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Batched, tie-correct Mann-Whitney counts pooled over matching cells.

    Rank the score inside its cell by subtracting counts in earlier cells from
    the joint cell/score rank. This is algebraically the parent's binary c-index;
    ties in either covariates or outcomes retain its exact conventions.
    """
    if coding.shape != r2.shape or groups.shape != r2.shape or r2.ndim != 2:
        raise ValueError("batch arrays must have the same (draw, feature) shape")
    if not np.isin(coding, [0, 1]).all() or not np.isfinite(r2).all():
        raise ValueError("expected finite scores and binary coding")
    batch, size = r2.shape
    n_groups = int(groups.max()) + 1
    indices = groups + np.arange(batch)[:, None] * n_groups
    ranks = rankdata(groups * (size + 1) + rankdata(r2, axis=1), axis=1)

    def counts(weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(
            indices.ravel(),
            weights=None if weights is None else weights.ravel(),
            minlength=batch * n_groups,
        ).reshape(batch, n_groups)

    total = counts()
    positive = counts(coding)
    negative = total - positive
    rank_sum = counts(ranks * coding)
    offsets = np.cumsum(total, axis=1) - total
    u = rank_sum - positive * offsets - positive * (positive + 1) / 2
    pairs = positive * negative
    eligible = (total >= min_cell) & (pairs > 0)
    numerator = np.where(eligible, u, 0).sum(axis=1)
    denominator = np.where(eligible, pairs, 0).sum(axis=1)
    if (denominator <= 0).any():
        raise ValueError("no comparable pairs after matching")
    value = numerator / denominator - 0.5
    diagnostics = {
        "comparable_pairs": denominator,
        "eligible_features": np.where(eligible, total, 0).sum(axis=1),
        "eligible_positive_features": np.where(eligible, positive, 0).sum(axis=1),
        "eligible_negative_features": np.where(eligible, negative, 0).sum(axis=1),
        "eligible_cells": eligible.sum(axis=1),
        "occupied_cells": (total > 0).sum(axis=1),
        "all_cross_tier_pairs": coding.sum(axis=1) * (size - coding.sum(axis=1)),
    }
    return value, diagnostics


def oracle_check(coding: np.ndarray, r2: np.ndarray, controls: list[np.ndarray], bins: int):
    """Verify the actual production function against the unchanged parent estimator."""
    estimator = parent._load_estimator()
    groups = assign_strata(controls, bins, r2.shape)
    actual, _ = binary_batch(coding, r2, groups, 2 * estimator.MIN_POS)
    expected = np.asarray(
        [
            estimator.concordance(
                coding[i], r2[i], [np.flatnonzero(groups[i] == g) for g in np.unique(groups[i])]
            )
            - 0.5
            for i in range(len(r2))
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def prepare(source_root: Path, manifest_path: Path, sae_weights: Path, out: Path) -> Path:
    """Build a compact, row-verified analysis substrate from pinned original inputs."""
    baseline = source_root / "eval_results/issue_1482/tier_concordance.json"
    old = json.loads(baseline.read_text())
    for rel in (parent.DECODER_STORE.relative_to(ROOT), parent.TIER_STORE.relative_to(ROOT)):
        expected = next(x["sha256"] for x in old["inputs"] if x["path"] == str(rel))
        if digest(source_root / rel) != expected:
            raise ValueError(f"published input hash mismatch: {rel}")
    parent.DECODER_STORE = source_root / parent.DECODER_STORE.relative_to(ROOT)
    parent.TIER_STORE = source_root / parent.TIER_STORE.relative_to(ROOT)
    panel = parent.load_panel()
    if panel["n_dropped"] != 0:
        raise ValueError("the published feature panel changed")
    with np.load(parent.DECODER_STORE) as z:
        ids = z["panel_ids"]
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if digest(Path(entry["local"])) != entry["sha256"]:
            raise ValueError(f"source hash mismatch: {entry['path']}")
    split_path = next(
        Path(x["local"])
        for x in manifest["files"]
        if x["path"].endswith("split_indices_matryoshka.npz")
    )
    with np.load(split_path) as z:
        score_rows = z["s_score"].astype(np.int64)
        if len(score_rows) != 6000 or len(z["s_fit"]) != 24000:
            raise ValueError("the published fit/score split changed")
    states, rows = [], []
    for entry in manifest["files"]:
        if not Path(entry["path"]).name.startswith("ans_l20_g"):
            continue
        with np.load(entry["local"]) as z:
            keep = np.isin(z["row_idx"], score_rows)
            states.append(z["a20"][keep].astype(np.float64))
            rows.append(z["row_idx"][keep])
    rows = np.concatenate(rows)
    if len(np.unique(rows)) != 6000 or not np.array_equal(np.sort(rows), np.sort(score_rows)):
        raise ValueError("answer shards do not cover the original score rows exactly once")
    y = np.concatenate(states)
    if y.shape != (6000, 3584) or not np.isfinite(y).all():
        raise ValueError(f"invalid answer matrix: {y.shape}")
    y -= y.mean(axis=0)
    log("[phase=variance] computing score-row answer covariance")
    covariance = y.T @ y / (len(y) - 1)
    with safe_open(str(sae_weights), framework="numpy") as handle:
        if handle.get_slice("W_dec").get_shape() != [65536, 3584]:
            raise ValueError("wrong SAE decoder shape")
        directions = handle.get_tensor("W_dec")[ids].astype(np.float64).T
    norms = np.linalg.norm(directions, axis=0)
    if (norms <= 0).any():
        raise ValueError("zero decoder direction")
    directions /= norms
    variance = np.empty(len(ids))
    for start in range(0, len(ids), 1024):
        u = directions[:, start : start + 1024]
        variance[start : start + 1024] = np.einsum("df,df->f", u, covariance @ u)
    np.testing.assert_allclose(
        variance[:7], np.var(y @ directions[:, :7], axis=0, ddof=1), rtol=1e-12
    )
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("nonpositive/nonfinite variance")
    substrate = out / "panel.npz"
    np.savez_compressed(
        substrate,
        feature_ids=ids,
        r2=panel["r2"],
        tier=panel["tier"],
        activity=panel["activity"],
        variance=variance,
        score_rows=score_rows,
    )
    write_json_atomic(
        out / "preparation.json",
        {
            "n_features": len(ids),
            "n_score_rows": len(score_rows),
            "layer": 20,
            "variance_definition": (
                "sample variance of unit-decoder projection of mean layer-20 answer state "
                "on the original 6000 score rows (ddof=1)"
            ),
            "sae_weights": {"path": str(sae_weights), "sha256": digest(sae_weights)},
            "parent_result": {"path": str(baseline), "sha256": digest(baseline)},
            "source_manifest_sha256": digest(manifest_path),
            "panel_sha256": digest(substrate),
            "variance_direct_projection_check": "PASS; seven directions, relative tolerance 1e-12",
            "per_tier_median_variance": {
                str(t): float(np.median(variance[panel["tier"] == t])) for t in range(3)
            },
            "spearman_variance_r2": float(spearmanr(variance, panel["r2"]).statistic),
        },
    )
    log("[phase=variance] panel prepared and covariance projection verified")
    return substrate


def run_scores(substrate: Path, out: Path, n_boot: int) -> dict:
    with np.load(substrate) as z:
        r2, tier = z["r2"], z["tier"]
        covariates = {"activity": np.log(z["activity"]), "variance": np.log(z["variance"])}
    estimator = parent._load_estimator()
    schemes = {
        "pooled": [],
        "activity": ["activity"],
        "variance": ["variance"],
        "variance_and_activity": ["variance", "activity"],
    }
    contrasts = {
        "coarsest_vs_rest": (tier == 0, np.ones(len(tier), dtype=bool)),
        "coarsest_vs_middle": (tier == 0, tier <= 1),
        "coarsest_vs_finest": (tier == 0, tier != 1),
        "middle_vs_finest": (tier == 1, tier >= 1),
        "coarsest_and_middle_vs_finest": (tier <= 1, np.ones(len(tier), dtype=bool)),
    }
    results = {}
    for index, (name, (positive, member)) in enumerate(contrasts.items()):
        x, y = positive[member].astype(int), r2[member]
        controls = {k: v[member] for k, v in covariates.items()}
        row = {"n": len(y), "n_positive": int(x.sum()), "schemes": {}}
        draws = {key: np.empty(n_boot) for key in schemes}
        for key, names in schemes.items():
            selected = [controls[k][None, :] for k in names]
            groups = assign_strata(selected, 5, (1, len(y)))
            values, diagnostics = binary_batch(
                x[None, :], y[None, :], groups, 2 * estimator.MIN_POS
            )
            oracle_check(x[None, :], y[None, :], selected, 5)
            info = {"value": float(values[0]), **{k: int(v[0]) for k, v in diagnostics.items()}}
            info["pair_fraction"] = info["comparable_pairs"] / info["all_cross_tier_pairs"]
            info["sensitivity"] = {}
            if names:
                for bins in (10, 20):
                    sensitivity_groups = assign_strata(selected, bins, (1, len(y)))
                    cell_n = np.bincount(sensitivity_groups[0])
                    cell_pos = np.bincount(sensitivity_groups[0], weights=x)
                    comparable = (cell_n >= 2 * estimator.MIN_POS) & (cell_pos > 0)
                    comparable &= cell_pos < cell_n
                    if not comparable.any():
                        info["sensitivity"][str(bins)] = {
                            "value": None,
                            "status": "no comparable cells under inherited minimum cell size",
                            "comparable_pairs": 0,
                            "eligible_features": 0,
                        }
                        continue
                    values2, diag2 = binary_batch(
                        x[None, :],
                        y[None, :],
                        sensitivity_groups,
                        2 * estimator.MIN_POS,
                    )
                    info["sensitivity"][str(bins)] = {
                        "value": float(values2[0]),
                        **{k: int(v[0]) for k, v in diag2.items()},
                    }
            row["schemes"][key] = info
        rng = np.random.default_rng(SEED + index)
        for start in range(0, n_boot, BATCH):
            stop = min(start + BATCH, n_boot)
            pick = rng.integers(0, len(y), size=(stop - start, len(y)))
            xb, yb = x[pick], y[pick]
            for key, names in schemes.items():
                selected = [controls[k][pick] for k in names]
                groups = assign_strata(selected, 5, yb.shape)
                values, _ = binary_batch(xb, yb, groups, 2 * estimator.MIN_POS)
                draws[key][start:stop] = values
                if start == 0:
                    oracle_check(xb[:3], yb[:3], [c[:3] for c in selected], 5)
            if start % (BATCH * 8) == 0 or stop == n_boot:
                log(f"[phase=bootstrap] {name} {stop}/{n_boot}")
        for key in schemes:
            if not np.isfinite(draws[key]).all():
                raise ValueError("nonfinite bootstrap draws")
            row["schemes"][key]["ci95"] = np.percentile(draws[key], [2.5, 97.5]).tolist()
        row["paired_difference_variance_minus_activity"] = {
            "value": row["schemes"]["variance"]["value"] - row["schemes"]["activity"]["value"],
            "ci95": np.percentile(draws["variance"] - draws["activity"], [2.5, 97.5]).tolist(),
        }
        np.savez_compressed(out / f"bootstrap_{name}.npz", **draws)
        results[name] = row
        write_json_atomic(out / "scores.partial.json", results)
    summary = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "statistic": (
            "P(coarser feature has higher held-out decoder-direction R2 within matched cells), "
            "minus 0.5; score ties count as one half; pairs weighted by cell pair count"
        ),
        "matching": {
            "primary_bins": 5,
            "joint": "crossed marginal quantiles",
            "sensitivity_bins": [10, 20],
            "minimum_total_features_per_cell": 2 * estimator.MIN_POS,
            "edges": "recomputed in each feature bootstrap draw and each contrast subset",
        },
        "bootstrap": {
            "draws": n_boot,
            "seed": SEED,
            "unit": "feature",
            "interval": "percentile 95%",
            "schemes_paired": True,
            "limitation": (
                "conditions on the fixed fitted map and evaluation answers; does not account "
                "for dependence between correlated SAE features"
            ),
        },
        "validation": (
            "production batched estimator agrees with unchanged parent for each full panel "
            "and three bootstrap draws per contrast/scheme, absolute tolerance 1e-12"
        ),
        "panel_sha256": digest(substrate),
        "code_sha256": digest(Path(__file__)),
        "provenance": as_metadata_dict(git_provenance(ROOT), phase="variance-matched-concordance"),
        "results": results,
    }
    write_json_atomic(out / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sae-weights", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--prepared-panel", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log("[phase=start] variance-matched Matryoshka reanalysis")
    panel = args.prepared_panel or prepare(
        args.source_root, args.out / "source_manifest.json", args.sae_weights, args.out
    )
    run_scores(panel, args.out, args.n_boot)
    write_json_atomic(
        args.out / "complete.json",
        {
            "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
            "wall_seconds": time.time() - started,
            "summary_sha256": digest(args.out / "summary.json"),
            "exit_code": 0,
        },
    )
    log("[phase=done] complete")


if __name__ == "__main__":
    main()
