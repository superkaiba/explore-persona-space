"""Free-analysis round for issue #2224 (0 GPU-h, existing artifacts only).

Two reads over the EXISTING per-sample screening score tables
(``eval_results/issue_2224/screening_scores/<corpus>/<trait>.json``) and the
judge-filter labels (``eval_results/issue_2224/selections/<corpus>/<trait>/
filter_scores.json``):

1. **Per-sample ROC-AUC** — per (corpus, trait, predictor arm): AUC of the
   arm's score against the binary judge trait label on the JUDGED subset
   (the union of the four selection arms' top-k candidates — a
   selection-conditioned subset, NOT a random sample; stated in the output
   meta). Two binarizations: ``trait_bearing_ge1`` (score >= 1, the issue's
   own ``TRAIT_BEARING_AT_OR_ABOVE`` convention) and ``strong_gt50``
   (score > 50, the project judge-filter positive convention). Rank-CI via
   a vectorized bootstrap (2,000 draws, shared index matrix across arms
   within a cell).

2. **Map-calibration probe** — per (corpus, trait): affine fit (slope,
   intercept, R^2) between the frozen-map stand-in projections
   (``mapped_dp_context`` primary; ``mapped_dp_prefix`` companion per the
   both-mapping-arms rule) and the exact base-gen projections
   (``exact_dp``) at matched samples (all 50k rows), plus scale quantiles
   (p5/p50/p95) and robust scale ratios — the interpretation's
   "~15x off-scale" claim made precise.

Content hygiene: labels + scores only; no generation text is loaded.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from issue2224_common import (
    PROJECT_ROOT,
    atomic_write_json,
    repro_meta,
    sha256_file,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/scipy import: shared-VM thread caps (#847)

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

logger = logging.getLogger("issue2224_free_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CORPORA = ("lmsys", "ultrachat")
TRAITS = ("evil", "hallucination", "sycophancy")
ARMS = (
    "raw",
    "exact_dp",
    "prompttoken_dp",
    "mapped_dp_context",
    "mapped_dp_prefix",
    "probe_diff_context",
    "probe_diff_prefix",
)
EXACT_ARM = "exact_dp"
MAPPED_ARMS = ("mapped_dp_context", "mapped_dp_prefix")

# Binarization thresholds for the judge trait label (0-100 graded mean).
# trait_bearing_ge1: complement of the pipeline keep-rule FILTER_KEEP_BELOW=1.0
# (issue2224_analysis.TRAIT_BEARING_AT_OR_ABOVE); strong_gt50: project judge
# positive convention (pos > 50).
LABEL_THRESHOLDS = {"trait_bearing_ge1": ("ge", 1.0), "strong_gt50": ("gt", 50.0)}
N_BOOT = 2000
BOOT_CHUNK = 500
SEED = 20260812
DEGENERATE_POS_FLOOR = 10  # n_pos below this => AUC flagged degenerate
QUANTILES = (5.0, 50.0, 95.0)

SCREENING_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "screening_scores"
SELECTIONS_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "selections"
OUT_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "free_analysis"
FIGURES_DIR_DEFAULT = PROJECT_ROOT / "figures" / "issue_2224"


def _load_cell(screening_dir: Path, selections_dir: Path, corpus: str, trait: str) -> dict:
    """Load one cell's score table + judge labels; fail loud on any gap."""
    table_path = screening_dir / corpus / f"{trait}.json"
    label_path = selections_dir / corpus / trait / "filter_scores.json"
    table = json.loads(table_path.read_text())
    labels_doc = json.loads(label_path.read_text())
    scores = table["scores"]
    if len(scores) == 0:
        raise RuntimeError(f"{corpus}/{trait}: empty score table at {table_path}")
    row0 = next(iter(scores.values()))
    missing = set(ARMS) - set(row0.keys())
    if missing:
        raise RuntimeError(f"{corpus}/{trait}: score rows missing arms {sorted(missing)}")
    judge_scores = {k: v for k, v in labels_doc["scores"].items() if v is not None}
    if len(judge_scores) == 0:
        raise RuntimeError(f"{corpus}/{trait}: no non-null judge labels at {label_path}")
    return {
        "table_path": table_path,
        "label_path": label_path,
        "scores": scores,
        "judge_scores": judge_scores,
        "n_table": len(scores),
        "n_labels_nonnull": len(judge_scores),
        "n_labels_null": sum(1 for v in labels_doc["scores"].values() if v is None),
    }


def _auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC per row from midranks (B, n) and binary labels (B, n)."""
    npos = y.sum(axis=1).astype(np.float64)
    nneg = y.shape[1] - npos
    sum_pos = (ranks * y).sum(axis=1)
    denom = npos * nneg
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_pos - npos * (npos + 1.0) / 2.0) / denom
    auc = np.where(denom == 0, np.nan, auc)
    return auc


def run_auc(cells: dict, out_path: Path) -> dict:
    """Analysis 1: per-(corpus, trait, arm) ROC-AUC + vectorized bootstrap CI."""
    results: dict = {}
    for (corpus, trait), cell in cells.items():
        joined_ids = sorted(set(cell["scores"]) & set(cell["judge_scores"]))
        if len(joined_ids) == 0:
            raise RuntimeError(f"{corpus}/{trait}: empty join between score table and labels")
        n = len(joined_ids)
        graded = np.array([float(cell["judge_scores"][i]) for i in joined_ids], dtype=np.float64)
        arm_x = {}
        for arm in ARMS:
            x = np.array([float(cell["scores"][i][arm]) for i in joined_ids], dtype=np.float64)
            if not np.all(np.isfinite(x)):
                raise RuntimeError(f"{corpus}/{trait}/{arm}: non-finite predictor scores")
            arm_x[arm] = x
        labels01 = {}
        for tname, (op, thr) in LABEL_THRESHOLDS.items():
            y = (graded >= thr) if op == "ge" else (graded > thr)
            labels01[tname] = y.astype(np.float64)

        # Point AUC from full-sample midranks.
        point: dict = {arm: {} for arm in ARMS}
        for arm in ARMS:
            r_full = rankdata(arm_x[arm], method="average")[None, :]
            for tname, y in labels01.items():
                point[arm][tname] = float(_auc_from_ranks(r_full, y[None, :])[0])

        # Vectorized bootstrap: one shared index matrix per chunk across arms.
        rng = np.random.default_rng(SEED)
        boot: dict = {arm: {t: [] for t in labels01} for arm in ARMS}
        done = 0
        while done < N_BOOT:
            b = min(BOOT_CHUNK, N_BOOT - done)
            idx = rng.integers(0, n, size=(b, n))
            for arm in ARMS:
                ranks = rankdata(arm_x[arm][idx], method="average", axis=1)
                for tname, y in labels01.items():
                    boot[arm][tname].append(_auc_from_ranks(ranks, y[idx]))
            done += b

        cell_out: dict = {
            "n_join": n,
            "n_table": cell["n_table"],
            "n_labels_nonnull": cell["n_labels_nonnull"],
            "n_labels_null_dropped": cell["n_labels_null"],
            "arms": {},
        }
        for tname, y in labels01.items():
            n_pos = int(y.sum())
            cell_out[f"n_pos__{tname}"] = n_pos
            cell_out[f"n_neg__{tname}"] = n - n_pos
        for arm in ARMS:
            arm_out = {}
            for tname in labels01:
                draws = np.concatenate(boot[arm][tname])
                valid = draws[np.isfinite(draws)]
                n_pos = int(labels01[tname].sum())
                arm_out[tname] = {
                    "auc": point[arm][tname],
                    "ci95": [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))],
                    "ci_method": f"bootstrap percentile, {N_BOOT} draws, seed={SEED}",
                    "n_valid_draws": int(valid.size),
                    "degenerate": n_pos < DEGENERATE_POS_FLOOR,
                }
            cell_out["arms"][arm] = arm_out
        results[f"{corpus}/{trait}"] = cell_out
        logger.info(
            "AUC %s/%s done (n_join=%d, pos_ge1=%d)",
            corpus,
            trait,
            n,
            int(labels01["trait_bearing_ge1"].sum()),
        )

    doc = {
        "meta": {
            **repro_meta("issue2224_free_analysis.auc"),
            "label_thresholds": {k: f"{op} {thr}" for k, (op, thr) in LABEL_THRESHOLDS.items()},
            "n_boot": N_BOOT,
            "seed": SEED,
            "degenerate_pos_floor": DEGENERATE_POS_FLOOR,
            "subset_note": (
                "AUC computed on the JUDGED subset only: the union of the four selection "
                "arms' top-k=2000 candidates per cell (selection-conditioned, not a random "
                "sample of the 50k pool); null judge scores dropped, never coerced"
            ),
            "inputs": {
                f"{c}/{t}": {
                    "table": str(cells[(c, t)]["table_path"].relative_to(PROJECT_ROOT)),
                    "table_sha256": sha256_file(cells[(c, t)]["table_path"]),
                    "labels": str(cells[(c, t)]["label_path"].relative_to(PROJECT_ROOT)),
                    "labels_sha256": sha256_file(cells[(c, t)]["label_path"]),
                }
                for c in CORPORA
                for t in TRAITS
            },
        },
        "results": results,
    }
    atomic_write_json(doc, out_path)
    logger.info("wrote %s", out_path)
    return doc


def run_calibration(cells: dict, out_path: Path) -> dict:
    """Analysis 2: affine fit + scale quantiles of mapped stand-ins vs exact_dp."""
    results: dict = {}
    for (corpus, trait), cell in cells.items():
        ids = sorted(cell["scores"].keys())
        x = np.array([float(cell["scores"][i][EXACT_ARM]) for i in ids], dtype=np.float64)
        cell_out: dict = {"n_matched": len(ids), "exact_arm": EXACT_ARM}
        qx = {f"p{int(q)}": float(np.percentile(x, q)) for q in QUANTILES}
        cell_out["exact_quantiles"] = qx
        cell_out["exact_mean"] = float(x.mean())
        cell_out["exact_std"] = float(x.std())
        for arm in MAPPED_ARMS:
            y = np.array([float(cell["scores"][i][arm]) for i in ids], dtype=np.float64)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise RuntimeError(f"{corpus}/{trait}/{arm}: non-finite projections")
            r = float(np.corrcoef(x, y)[0, 1])
            cov = float(np.cov(x, y, ddof=1)[0, 1])
            slope_standin_on_exact = cov / float(np.var(x, ddof=1))
            slope_exact_on_standin = cov / float(np.var(y, ddof=1))
            qy = {f"p{int(q)}": float(np.percentile(y, q)) for q in QUANTILES}
            quantile_ratios = {}
            for k, qe in qx.items():
                quantile_ratios[k] = float(qy[k] / qe) if abs(qe) > 1e-9 else None
            spread_x = qx["p95"] - qx["p5"]
            spread_y = qy["p95"] - qy["p5"]
            cell_out[arm] = {
                "pearson_r": r,
                "r2": r * r,
                "slope_standin_on_exact": slope_standin_on_exact,
                "intercept_standin_on_exact": float(y.mean() - slope_standin_on_exact * x.mean()),
                "slope_exact_on_standin": slope_exact_on_standin,
                "intercept_exact_on_standin": float(x.mean() - slope_exact_on_standin * y.mean()),
                "standin_quantiles": qy,
                "standin_mean": float(y.mean()),
                "standin_std": float(y.std()),
                "quantile_ratios_standin_over_exact": quantile_ratios,
                "scale_ratio_p95_minus_p5": float(spread_y / spread_x) if spread_x > 0 else None,
                "scale_ratio_std": float(y.std() / x.std()) if x.std() > 0 else None,
            }
        results[f"{corpus}/{trait}"] = cell_out
        logger.info("calibration %s/%s done (n=%d)", corpus, trait, len(ids))

    doc = {
        "meta": {
            **repro_meta("issue2224_free_analysis.calibration"),
            "note": (
                "affine fit of the frozen-map stand-in projections vs exact base-gen "
                "projections (exact_dp) at matched samples (all pool rows); both mapping "
                "arms reported (context primary, prefix companion). Scale ratios make the "
                "interpretation's off-scale claim precise; slope_exact_on_standin is the "
                "recovery direction"
            ),
        },
        "results": results,
    }
    atomic_write_json(doc, out_path)
    logger.info("wrote %s", out_path)
    return doc


def render_figure(auc_doc: dict, fig_path: Path) -> None:
    """Compact 2x3 panel figure: AUC by arm per (corpus, trait), ge1 labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {
        "raw": "raw",
        "exact_dp": "exact",
        "prompttoken_dp": "prompt-tok",
        "mapped_dp_context": "mapped",
        "mapped_dp_prefix": "mapped-pre",
        "probe_diff_context": "probe",
        "probe_diff_prefix": "probe-pre",
    }
    tname = "trait_bearing_ge1"
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for i, corpus in enumerate(CORPORA):
        for j, trait in enumerate(TRAITS):
            ax = axes[i][j]
            cell = auc_doc["results"][f"{corpus}/{trait}"]
            aucs, lo_err, hi_err = [], [], []
            for arm in ARMS:
                a = cell["arms"][arm][tname]
                aucs.append(a["auc"])
                lo, hi = a["ci95"]
                lo_err.append(max(a["auc"] - lo, 0.0))
                hi_err.append(max(hi - a["auc"], 0.0))
            xs = np.arange(len(ARMS))
            ax.bar(xs, aucs, color="#4878b0")
            ax.errorbar(xs, aucs, yerr=[lo_err, hi_err], fmt="none", ecolor="black", capsize=3)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(xs)
            ax.set_xticklabels([short[a] for a in ARMS], rotation=45, ha="right", fontsize=8)
            n_pos = cell[f"n_pos__{tname}"]
            ax.set_title(f"{corpus} / {trait} (n={cell['n_join']}, pos={n_pos})", fontsize=10)
            if j == 0:
                ax.set_ylabel("ROC-AUC (judge label: trait score >= 1)")
    fig.suptitle(
        "Screening-score ROC-AUC vs judge trait label, judged subset (95% bootstrap CI)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", fig_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screening-dir", type=Path, default=SCREENING_DIR_DEFAULT)
    parser.add_argument("--selections-dir", type=Path, default=SELECTIONS_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    cells = {
        (c, t): _load_cell(args.screening_dir, args.selections_dir, c, t)
        for c in CORPORA
        for t in TRAITS
    }
    auc_doc = run_auc(cells, args.out_dir / "auc_by_arm.json")
    run_calibration(cells, args.out_dir / "map_calibration.json")
    if not args.no_figure:
        render_figure(auc_doc, args.figures_dir / "free_analysis_auc_by_arm.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
