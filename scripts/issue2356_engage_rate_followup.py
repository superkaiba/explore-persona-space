"""Engage-rate continuous-DV follow-up for issue #2356 (inline free analysis).

The registered #2356 comparison scored 7 refusal predictors with pooled
out-of-fold AUROC on a BINARY refuse/engage label, which is saturated
(context probe 0.95-0.99). This follow-up re-scores the same persisted OOF
scores against the continuous per-prompt ENGAGE RATE (fraction of >=7 valid
temp-0.9 draws that comply), which is unsaturated, plus a middle-band-only
read (rows the run itself excluded from labeling for a mixed rate).

Pure re-correlation of persisted scores — no fits of any kind (no ridge, no
probe, no map), no GPU, no downloads. Reads the #2356 worktree artifacts at
WT_RESULTS; writes JSONs + one figure at the repo root.

Orientation: all persisted scores are oriented toward REFUSAL (the score
artifacts' own note: "all scores oriented as P(REFUSE); y=1 is refuse"), so
every Spearman rho is reported against (1 - rate) — positive rho = the
predictor tracks refusal.

Run: uv run python scripts/issue2356_engage_rate_followup.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402  (after load_dotenv: thread-cap discipline)
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WT_RESULTS = REPO_ROOT / ".claude/worktrees/issue-2356/eval_results/issue_2356"
OUT_DIR = REPO_ROOT / "eval_results/issue_2356/engage_rate_followup"
FIG_DIR = REPO_ROOT / "figures/issue_2356"

ARMS = ("armA", "armB")
PREDICTORS = (
    "ctx_ridge",
    "ctx_dim",
    "map3a_zr",
    "map3b_zr",
    "pca_ctx",
    "ans_greedy",
    "text_surface",
)
JUDGE = "judge_fewshot"

N_DRAWS = 2000
BOOT_SEED = 1234  # parity with the run's registered paired group bootstrap (stats.json notes)
MIN_BAND_CLASS = 20  # per-class floor for the within-band binary AUROC
MIN_BAND_JUDGE = 30  # judge rows required in band to include the judge contrast

# (name, predictor_a, predictor_b): contrast = rho(a) - rho(b), paired per draw.
CONTRASTS = (
    ("map3a_minus_ctx", "map3a_zr", "ctx_ridge"),
    ("map3b_minus_ctx", "map3b_zr", "ctx_ridge"),
    ("map3b_minus_map3a", "map3b_zr", "map3a_zr"),
    ("ans_minus_ctx", "ans_greedy", "ctx_ridge"),
    ("ctx_minus_judge", "ctx_ridge", JUDGE),
    ("ctx_minus_pca", "ctx_ridge", "pca_ctx"),
    ("ctx_minus_text_surface", "ctx_ridge", "text_surface"),
)


def _load_arm(arm: str) -> tuple[dict, dict]:
    scores = json.loads((WT_RESULTS / "results" / f"predictor_scores_{arm}.json").read_text())
    labels = json.loads((WT_RESULTS / arm / "labels.json").read_text())
    return scores, labels


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def _group_draw_indices(groups: list[str], n_draws: int, seed: int) -> list[np.ndarray]:
    """Row indices per bootstrap draw: resample unique groups with replacement,
    concatenate each drawn group's rows (multiplicity preserved)."""
    uniq = sorted(set(groups))
    rows_of = {g: [] for g in uniq}
    for i, g in enumerate(groups):
        rows_of[g].append(i)
    rows_of = {g: np.asarray(ix, dtype=np.int64) for g, ix in rows_of.items()}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_draws, len(uniq)))
    return [np.concatenate([rows_of[uniq[j]] for j in row]) for row in draws]


def _boot_rho_matrix(
    score_mat: np.ndarray, target: np.ndarray, draw_ix: list[np.ndarray]
) -> np.ndarray:
    """(n_draws, n_predictors) Spearman rho per draw: rankdata (average ties,
    i.e. tie-corrected) + Pearson on ranks over the drawn rows."""
    out = np.full((len(draw_ix), score_mat.shape[1]), np.nan)
    for d, ix in enumerate(draw_ix):
        ty = rankdata(target[ix])
        ty = ty - ty.mean()
        deny = float(np.sqrt((ty**2).sum()))
        for p in range(score_mat.shape[1]):
            tx = rankdata(score_mat[ix, p])
            tx = tx - tx.mean()
            den = deny * float(np.sqrt((tx**2).sum()))
            if den > 0:
                out[d, p] = float((tx * ty).sum()) / den
    return out


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC (average-rank tie handling); y in {0,1}."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ci95(draws: np.ndarray) -> list[float]:
    finite = draws[np.isfinite(draws)]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _read_block(
    rows: list[dict], predictors: tuple[str, ...], contrast_set: tuple, seed: int
) -> dict:
    """Per-predictor rho vs (1 - rate) + paired group-bootstrap contrasts."""
    target = np.asarray([1.0 - r["rate"] for r in rows], dtype=np.float64)
    groups = [str(r["group_id"]) for r in rows]
    mat = np.asarray([[r[p] for p in predictors] for r in rows], dtype=np.float64)
    point = {p: _rho(mat[:, i], target) for i, p in enumerate(predictors)}
    draw_ix = _group_draw_indices(groups, N_DRAWS, seed)
    boot = _boot_rho_matrix(mat, target, draw_ix)
    p_ix = {p: i for i, p in enumerate(predictors)}
    contrasts = {}
    for name, a, b in contrast_set:
        if a not in p_ix or b not in p_ix:
            continue
        diff = boot[:, p_ix[a]] - boot[:, p_ix[b]]
        contrasts[name] = {
            "point": point[a] - point[b],
            "ci95": _ci95(diff),
            "n_draws": N_DRAWS,
            "n_boot_finite": int(np.isfinite(diff).sum()),
        }
    return {
        "n": len(rows),
        "n_groups": len(set(groups)),
        "spearman_vs_one_minus_rate": point,
        "contrasts": contrasts,
        "_boot": boot,
        "_draw_ix": draw_ix,
        "_mat": mat,
        "_predictors": predictors,
    }


def _strip_private(block: dict) -> dict:
    return {k: v for k, v in block.items() if not k.startswith("_")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    continuous: dict = {"meta": {}, "arms": {}}
    middle: dict = {"meta": {}, "arms": {}}
    fig_rho: dict[str, dict[str, float]] = {}

    src_commits = {}
    for arm in ARMS:
        scores_doc, labels_doc = _load_arm(arm)
        sal = scores_doc["scores_all_labeled"]
        lab_rows = labels_doc["rows"]
        thresholds = labels_doc["thresholds"]
        src_commits[arm] = scores_doc["meta"]["git_commit"]

        rows_all = [r for r in sal.values() if r.get("rate") is not None]
        if len(rows_all) != len(sal):
            raise ValueError(f"{arm}: {len(sal) - len(rows_all)} rows with null rate")
        for h, r in sal.items():
            if lab_rows[h]["rate"] != r["rate"]:
                raise ValueError(f"{arm}: rate mismatch between score and label artifacts at {h}")

        # Judge mask == balanced rows (asserted; the two masks coincide, so the
        # balanced-only sensitivity read and the judge head-to-head are one read).
        judge_rows = [r for r in sal.values() if JUDGE in r]
        if {id(r) for r in judge_rows} != {id(r) for r in sal.values() if r.get("balanced")}:
            raise ValueError(f"{arm}: judge_fewshot mask != balanced mask")

        all_block = _read_block(
            rows_all,
            PREDICTORS,
            tuple(c for c in CONTRASTS if c[0] != "ctx_minus_judge"),
            BOOT_SEED,
        )
        judge_block = _read_block(judge_rows, PREDICTORS + (JUDGE,), CONTRASTS, BOOT_SEED)
        continuous["arms"][arm] = {
            "all_labeled": _strip_private(all_block),
            "balanced_judge_mask": _strip_private(judge_block),
        }
        fig_rho[arm] = all_block["spearman_vs_one_minus_rate"]

        # Middle band: the run's own exclusion (drop_reason == "middle_band";
        # thresholds lo/hi bound the band as lo < rate < hi).
        band_hashes = [h for h, r in lab_rows.items() if r.get("drop_reason") == "middle_band"]
        band_rows = [sal[h] for h in band_hashes]
        lo, hi = thresholds["lo"], thresholds["hi"]
        if any(not (lo < r["rate"] < hi) for r in band_rows):
            raise ValueError(f"{arm}: middle_band row with rate outside ({lo}, {hi})")
        n_band_judge = sum(1 for r in band_rows if JUDGE in r)
        band_contrasts = tuple(
            c for c in CONTRASTS if c[0] != "ctx_minus_judge" or n_band_judge >= MIN_BAND_JUDGE
        )
        band_preds = PREDICTORS + ((JUDGE,) if n_band_judge >= MIN_BAND_JUDGE else ())
        band_block = _read_block(band_rows, band_preds, band_contrasts, BOOT_SEED)

        # Within-band binary AUROC at rate < 0.5, gated on both classes >= floor.
        y_band = np.asarray([1 if r["rate"] < 0.5 else 0 for r in band_rows], dtype=np.int64)
        n_pos, n_neg = int(y_band.sum()), int(len(y_band) - y_band.sum())
        auroc_block: dict = {
            "label_definition": "y=1 iff rate < 0.5 (toward-refusal)",
            "class_counts": {"rate_lt_0.5": n_pos, "rate_ge_0.5": n_neg},
            "min_per_class": MIN_BAND_CLASS,
        }
        if min(n_pos, n_neg) >= MIN_BAND_CLASS:
            mat, draw_ix = band_block["_mat"], band_block["_draw_ix"]
            per_pred = {}
            for i, p in enumerate(band_preds):
                boot_auc = np.asarray([_auroc(mat[ix, i], y_band[ix]) for ix in draw_ix])
                per_pred[p] = {
                    "auroc": _auroc(mat[:, i], y_band),
                    "auroc_ci95": _ci95(boot_auc),
                    "n_boot_finite": int(np.isfinite(boot_auc).sum()),
                }
            auroc_block["per_predictor"] = per_pred
        else:
            auroc_block["skipped"] = f"a class has < {MIN_BAND_CLASS} rows"

        middle["arms"][arm] = {
            **_strip_private(band_block),
            "n_judge_rows_in_band": n_band_judge,
            "judge_included": n_band_judge >= MIN_BAND_JUDGE,
            "rate_range_realized": [
                min(r["rate"] for r in band_rows),
                max(r["rate"] for r in band_rows),
            ],
            "auroc_within_band": auroc_block,
        }

    shared_meta = {
        "issue": 2356,
        "analysis": "engage_rate_followup (inline free analysis; no fits — persisted OOF scores)",
        "dv": "per-prompt engage rate (fraction of >=7 valid temp-0.9 draws that comply)",
        "orientation": "scores oriented toward refusal (source artifacts: P(REFUSE)); all rho "
        "reported vs (1 - rate) so informative predictors give positive rho",
        "bootstrap": {
            "kind": "paired group bootstrap (resample group_ids with replacement; "
            "per draw rank-transform + Pearson on ranks over drawn rows)",
            "n_draws": N_DRAWS,
            "seed": BOOT_SEED,
        },
        "score_artifact_git_commit": src_commits,
        "repo_head_at_run": head,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/issue2356_engage_rate_followup.py",
    }
    continuous["meta"] = {
        **shared_meta,
        "reads": {
            "all_labeled": "every scores_all_labeled row with non-null rate (middle-band + "
            "balanced-subsample-excluded rows scored under their group's owning "
            "M1 fold at that fold's selected configurations)",
            "balanced_judge_mask": "balanced eval rows == judge_fewshot-scored rows (masks "
            "coincide exactly; doubles as the balanced-only sensitivity "
            "read)",
        },
    }
    middle["meta"] = {
        **shared_meta,
        "band_definition": "rows the run excluded from labeling with drop_reason == "
        "'middle_band' — engage rate strictly inside (lo, hi) = (0.3, 0.7) "
        "per labels.json thresholds; realized rates span [0.4, 0.6]",
    }

    (OUT_DIR / "continuous_dv.json").write_text(json.dumps(continuous, indent=1))
    (OUT_DIR / "middle_band.json").write_text(json.dumps(middle, indent=1))

    _figure(fig_rho)
    print(f"wrote {OUT_DIR}/continuous_dv.json, {OUT_DIR}/middle_band.json")


def _figure(fig_rho: dict[str, dict[str, float]]) -> None:
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette_blog(len(ARMS))
    x = np.arange(len(PREDICTORS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    arm_label = {"armA": "arm A (harmful flip-pairs)", "armB": "arm B (over-refusal)"}
    for k, arm in enumerate(ARMS):
        vals = [fig_rho[arm][p] for p in PREDICTORS]
        ax.bar(x + (k - 0.5) * width, vals, width, label=arm_label[arm], color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in PREDICTORS])
    ax.set_ylabel("Spearman rho vs (1 - engage rate)")
    ax.set_xlabel("predictor (pooled out-of-fold scores, all labeled rows)")
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.legend()
    savefig_paper(fig, "engage_rate_followup_spearman", dir=str(FIG_DIR))
    plt.close(fig)


if __name__ == "__main__":
    main()
