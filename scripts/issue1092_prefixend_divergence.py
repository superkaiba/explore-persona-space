"""#1092 inline follow-up: where do the prefix-end and averaged-context reads diverge?

User-chat inline free-analysis round (2026-07-22) on the banked
``inline_prefixend_monitoring`` artifacts. Both monitoring arms predict a
PER-PREFIX target (the prefix's mean judge score over its 48 queries), so their
predictions can only differ per PREFIX — the query-level structure is analyzed
separately as the behavior component that is invisible at prefix grain.

Parts (all CPU, all local inputs — NO new forward passes, NO new judge calls):
  A. Per-prefix divergence: delta = pred_prefix_end - pred_averaged_context and
     the differential absolute error, joined with prefix metadata (real vs
     battery, source, user turns, token length), ranked + correlated.
  B. Subset parity: r(read, judge) within the 99 real prefixes and the 50
     battery conditions separately, on the STORED held-out predictions; plus a
     real-only REFIT (fit and eval on real prefixes only, grouped 5-fold) to
     check the parity claim survives removing the battery from the fit.
  C. Crossed variance decomposition of the per-row judge score (prefix main /
     query main / residual) — how much of behavioral expression is even in
     principle prefix-predictable.
  D. Query-level reads: per-query main effect (which queries elicit the
     behavior regardless of prefix) and per-query disposition expression
     (corr across prefixes of the row score with the leave-this-query-out
     disposition).

Inputs: eval_results/issue_1092/inline_prefixend_monitoring/per_prefix_points.json
(held-out GCV dual-ridge predictions, grouped 5-fold by prefix — the round's
banked reads), data/issue_1092/hf_dl/issue1092_realistic_crossing/
{corpus/manifest.jsonl, p5_judge/scores_shard_*.jsonl, analysis_tensors/summaries/}.
Judge = claude-sonnet-4-5-20250929, graded 0-100, 5 draws temp 1.0 mean-aggregated;
dropped/None judge rows excluded (drop-never-coerce).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

BASE = "issue1092_realistic_crossing"
DST = PROJECT_ROOT / "data/issue_1092/hf_dl" / BASE
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_prefixend_monitoring"
FIG_DIR = PROJECT_ROOT / "figures/issue_1092"

CELL = "cell_inst_own"  # the user's claim is instruct-scoped; base collapses (banked)
TRAITS = ("sycophancy", "hallucination")
STRATA = ("dense_core", "battery")
LAYER = 14
N_FOLDS = 5
N_BOOT = 2000
SEED = 0


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip()
    except OSError:
        return "unknown"


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def grouped_kfold(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def boot_paired_r_diff(
    pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Bootstrap over prefixes: r(pred_a, y) - r(pred_b, y)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = pearson(pred_a[idx], y[idx]) - pearson(pred_b[idx], y[idx])
    return {
        "point": pearson(pred_a, y) - pearson(pred_b, y),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "frac_boot_positive": float((diffs > 0).mean()),
    }


def load_manifest_meta() -> tuple[dict, dict]:
    """Per-prefix and per-query metadata from the corpus manifest (dense_core+battery rows)."""
    pref: dict[str, dict] = {}
    qry: dict[str, dict] = {}
    with open(DST / "corpus/manifest.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["stratum"] not in STRATA:
                continue
            p = pref.setdefault(
                d["prefix_id"],
                {
                    "stratum": d["stratum"],
                    "source": d.get("prefix_source"),
                    # the manifest `topic` field is PREFIX-level (constant per prefix,
                    # verified 2026-07-22: 0/149 prefixes with >1 topic across rows)
                    "topic": d.get("topic"),
                    "n_user_turns": d.get("prefix_n_user_turns"),
                    "tokens": [],
                },
            )
            if d.get("n_tokens_instruct") is not None:
                p["tokens"].append(d["n_tokens_instruct"])
            qry.setdefault(d["query_id"], {"source": d.get("query_source")})
    for p in pref.values():
        p["median_context_tokens"] = float(np.median(p["tokens"])) if p["tokens"] else None
        del p["tokens"]
    return pref, qry


def load_row_scores(trait: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(9):
        with open(DST / f"p5_judge/scores_shard_{i:03d}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["cell_id"] != CELL or r["trait"] != trait:
                    continue
                if r.get("dropped") or r.get("score") is None:
                    continue
                if r.get("stratum") not in STRATA:
                    continue
                out[r["row_id"]] = float(r["score"])
    return out


def build_grid(trait: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """(Y, mask, prefixes, queries, positions) — Y[p, q] row judge score, NaN where missing;
    positions[p, q] = manifest line index (-1 where missing)."""
    scores = load_row_scores(trait)
    rows = []
    with open(DST / "corpus/manifest.jsonl") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d["stratum"] in STRATA and d["row_id"] in scores:
                rows.append((d["prefix_id"], d["query_id"], scores[d["row_id"]], i))
    prefixes = sorted({r[0] for r in rows})
    queries = sorted({r[1] for r in rows})
    pi = {p: j for j, p in enumerate(prefixes)}
    qi = {q: j for j, q in enumerate(queries)}
    y = np.full((len(prefixes), len(queries)), np.nan)
    pos = np.full((len(prefixes), len(queries)), -1, dtype=np.int64)
    for p, q, s, i in rows:
        y[pi[p], qi[q]] = s
        pos[pi[p], qi[q]] = i
    return y, np.isfinite(y), prefixes, queries, pos


def crossed_decomposition(y: np.ndarray, mask: np.ndarray) -> dict:
    """Unweighted-means two-way SS decomposition on the observed (near-complete) grid.
    Residual bundles prefix-x-query interaction, generation stochasticity, judge noise."""
    obs = y[mask]
    grand = obs.mean()
    a = np.nanmean(y, axis=1) - grand
    b = np.nanmean(y, axis=0) - grand
    fit = grand + a[:, None] + b[None, :]
    resid = y - fit
    ss_tot = float(np.nansum((y - grand) ** 2))
    ss_p = float(np.nansum((a[:, None] * mask) ** 2))
    ss_q = float(np.nansum((b[None, :] * mask) ** 2))
    ss_r = float(np.nansum(resid[mask] ** 2))
    return {
        "share_prefix": ss_p / ss_tot,
        "share_query": ss_q / ss_tot,
        "share_residual": ss_r / ss_tot,
        "n_cells_observed": int(mask.sum()),
        "grand_mean": float(grand),
    }


def realonly_refit(trait: str, real_prefixes: set[str]) -> dict:
    """Fit + eval both arms on the real prefixes only (grouped 5-fold, GCV dual ridge, L14
    ambient) — checks the parity read is not carried by the battery conditions in the fit."""
    y, mask, prefixes, _queries, pos = build_grid(trait)
    keep = np.array([p in real_prefixes for p in prefixes])
    y, mask, pos = y[keep], mask[keep], pos[keep]
    prefixes = [p for p in prefixes if p in real_prefixes]
    ctx = np.load(
        DST / f"analysis_tensors/summaries/{CELL}/context_end_L{LAYER:02d}.npy", mmap_mode="r"
    )
    pre = np.load(
        DST / f"analysis_tensors/summaries/{CELL}/prefix_end_L{LAYER:02d}.npy", mmap_mode="r"
    )
    n = len(prefixes)
    g_score = np.array([y[i][mask[i]].mean() for i in range(n)])
    g_ctx = np.stack(
        [np.asarray(ctx[pos[i][mask[i]]], dtype=np.float64).mean(axis=0) for i in range(n)]
    )
    first_pos = np.array([pos[i][mask[i]][0] for i in range(n)])
    g_pre = np.asarray(pre[first_pos], dtype=np.float64)
    pred = {}
    for name, x in (("averaged_context", g_ctx), ("prefix_end", g_pre)):
        p_hat = np.full(n, np.nan)
        for test in grouped_kfold(n, N_FOLDS, SEED):
            train = np.setdiff1d(np.arange(n), test)
            p_hat[test] = F.ridge_fit_predict(x[train], g_score[train], x[test])
        pred[name] = p_hat
    return {
        "n_prefixes": n,
        "r_averaged_context": pearson(pred["averaged_context"], g_score),
        "r_prefix_end": pearson(pred["prefix_end"], g_score),
        "paired_diff_avgctx_minus_prefixend": boot_paired_r_diff(
            pred["averaged_context"], pred["prefix_end"], g_score, N_BOOT, SEED
        ),
    }


def main() -> None:
    pp_all = json.load(open(OUT / "per_prefix_points.json"))
    pref_meta, qry_meta = load_manifest_meta()
    results: dict = {
        "read": "#1092 prefix-end vs averaged-context divergence analysis (inline follow-up)",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "cell": CELL,
        "layer": LAYER,
        "inputs": {
            "per_prefix_points": "eval_results/issue_1092/inline_prefixend_monitoring/"
            "per_prefix_points.json (held-out grouped-5-fold ridge predictions, banked)",
            "row_scores": "data/issue_1092/hf_dl/.../p5_judge/scores_shard_*.jsonl "
            "(sonnet-4-5 graded 0-100, 5 draws mean; dropped rows excluded)",
        },
        "traits": {},
    }
    for trait in TRAITS:
        pp = pp_all["cells"][CELL][trait]
        ids = pp["prefix_id"]
        pe = np.array(pp["read_prefix_end"])
        ac = np.array(pp["read_averaged_context"])
        jm = np.array(pp["judge_mean"])
        js = np.array(pp["judge_std"])
        is_batt = np.array([i.startswith("batt_") for i in ids])
        turns = np.array(
            [pref_meta.get(i, {}).get("n_user_turns") or np.nan for i in ids], dtype=np.float64
        )
        toks = np.array(
            [pref_meta.get(i, {}).get("median_context_tokens") or np.nan for i in ids],
            dtype=np.float64,
        )
        delta = pe - ac
        d_abs_err = np.abs(pe - jm) - np.abs(ac - jm)  # >0: prefix-end worse on this prefix

        def row(i: int) -> dict:
            return {
                "prefix_id": ids[i],
                "stratum": "battery" if is_batt[i] else "real",
                "source": pref_meta.get(ids[i], {}).get("source"),
                "topic": pref_meta.get(ids[i], {}).get("topic"),
                "n_user_turns": pref_meta.get(ids[i], {}).get("n_user_turns"),
                "median_context_tokens": pref_meta.get(ids[i], {}).get("median_context_tokens"),
                "judge_mean": float(jm[i]),
                "judge_std": float(js[i]),
                "pred_prefix_end": float(pe[i]),
                "pred_averaged_context": float(ac[i]),
                "delta_pred": float(delta[i]),
                "abs_err_diff_pe_minus_ac": float(d_abs_err[i]),
            }

        top_delta = [row(i) for i in np.argsort(-np.abs(delta))[:12]]
        top_pe_worse = [row(i) for i in np.argsort(-d_abs_err)[:8]]
        top_pe_better = [row(i) for i in np.argsort(d_abs_err)[:8]]

        real, batt = ~is_batt, is_batt
        subset_parity = {}
        for name, m in (("real_only", real), ("battery_only", batt), ("pooled", slice(None))):
            subset_parity[name] = {
                "n": int(np.sum(m)) if not isinstance(m, slice) else len(ids),
                "r_prefix_end": pearson(pe[m], jm[m]),
                "r_averaged_context": pearson(ac[m], jm[m]),
                "paired_diff_avgctx_minus_prefixend": boot_paired_r_diff(
                    ac[m], pe[m], jm[m], N_BOOT, SEED
                ),
            }

        y, mask, g_prefixes, g_queries, _pos = build_grid(trait)
        decomp = crossed_decomposition(y, mask)

        # Query-level: main effect + leave-this-query-out disposition expression.
        b_q = np.nanmean(y, axis=0) - np.nanmean(y[mask])
        rowsum = np.nansum(y, axis=1)
        nrow = mask.sum(axis=1)
        r_q, disp_range = [], None
        loo = (rowsum[:, None] - np.where(mask, y, 0.0)) / np.maximum(nrow[:, None] - 1, 1)
        for qj in range(y.shape[1]):
            m = mask[:, qj]
            r_q.append(pearson(y[m, qj], loo[m, qj]))
        r_q = np.array(r_q)
        disp_range = [float(np.nanmin(loo)), float(np.nanmax(loo))]
        q_order = np.argsort(-np.abs(b_q))
        top_queries = [
            {
                "query_id": g_queries[j],
                "main_effect_vs_grand": float(b_q[j]),
                "disposition_expression_r": float(r_q[j]),
                "source": qry_meta.get(g_queries[j], {}).get("source"),
            }
            for j in q_order[:10]
        ]

        results["traits"][trait] = {
            "n_prefixes": len(ids),
            "n_real": int(real.sum()),
            "n_battery": int(batt.sum()),
            "arms_agreement_r_pred_pe_vs_pred_ac": pearson(pe, ac),
            "divergence_correlates_spearman": {
                "abs_delta_vs_battery_indicator": spearman(np.abs(delta), is_batt.astype(float)),
                "abs_delta_vs_n_user_turns_real_only": spearman(
                    np.abs(delta)[real & np.isfinite(turns)], turns[real & np.isfinite(turns)]
                ),
                "abs_delta_vs_context_tokens": spearman(
                    np.abs(delta)[np.isfinite(toks)], toks[np.isfinite(toks)]
                ),
                "abs_delta_vs_judge_std": spearman(np.abs(delta), js),
                "abs_err_diff_vs_n_user_turns_real_only": spearman(
                    d_abs_err[real & np.isfinite(turns)], turns[real & np.isfinite(turns)]
                ),
                "abs_err_diff_vs_judge_std": spearman(d_abs_err, js),
            },
            "mean_abs_delta": {
                "real": float(np.abs(delta)[real].mean()),
                "battery": float(np.abs(delta)[batt].mean()),
            },
            "mean_abs_err_diff_pe_minus_ac": {
                "real": float(d_abs_err[real].mean()),
                "battery": float(d_abs_err[batt].mean()),
                "pooled": float(d_abs_err.mean()),
            },
            "subset_parity_stored_predictions": subset_parity,
            "real_only_refit": realonly_refit(trait, {i for i in ids if not i.startswith("batt_")}),
            "judge_score_crossed_decomposition": decomp,
            "query_level": {
                "n_queries": int(y.shape[1]),
                "query_main_effect_sd": float(np.std(b_q)),
                "disposition_expression_r_median": float(np.nanmedian(r_q)),
                "disposition_expression_r_iqr": [
                    float(np.nanpercentile(r_q, 25)),
                    float(np.nanpercentile(r_q, 75)),
                ],
                "loo_disposition_range": disp_range,
                "top_queries_by_abs_main_effect": top_queries,
            },
            "top_prefixes_by_abs_pred_delta": top_delta,
            "top_prefixes_prefix_end_worse": top_pe_worse,
            "top_prefixes_prefix_end_better": top_pe_better,
        }
        print(
            f"[{trait}] pooled r: pe={subset_parity['pooled']['r_prefix_end']:.3f} "
            f"ac={subset_parity['pooled']['r_averaged_context']:.3f} | "
            f"real-only stored: pe={subset_parity['real_only']['r_prefix_end']:.3f} "
            f"ac={subset_parity['real_only']['r_averaged_context']:.3f} | "
            f"battery-only stored: pe={subset_parity['battery_only']['r_prefix_end']:.3f} "
            f"ac={subset_parity['battery_only']['r_averaged_context']:.3f}"
        )
        rr = results["traits"][trait]["real_only_refit"]
        print(
            f"[{trait}] real-only REFIT: pe={rr['r_prefix_end']:.3f} "
            f"ac={rr['r_averaged_context']:.3f} "
            f"diff={rr['paired_diff_avgctx_minus_prefixend']['point']:+.3f} "
            f"ci={rr['paired_diff_avgctx_minus_prefixend']['ci95']}"
        )
        print(
            f"[{trait}] judge-score variance shares: prefix="
            f"{decomp['share_prefix']:.3f} query={decomp['share_query']:.3f} "
            f"residual={decomp['share_residual']:.3f}"
        )

    out_path = OUT / "divergence_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path}")

    make_figure(results)


def make_figure(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pp_all = json.load(open(OUT / "per_prefix_points.json"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, trait in enumerate(TRAITS):
        pp = pp_all["cells"][CELL][trait]
        ids = pp["prefix_id"]
        pe = np.array(pp["read_prefix_end"])
        ac = np.array(pp["read_averaged_context"])
        jm = np.array(pp["judge_mean"])
        is_batt = np.array([i.startswith("batt_") for i in ids])
        ax = axes[0][col]
        for m, lab, c in (
            (~is_batt, "real (n=99)", "#0173b2"),
            (is_batt, "battery (n=50)", "#de8f05"),
        ):
            ax.scatter(ac[m], pe[m], s=18, alpha=0.75, label=lab, color=c)
        lim = [min(ac.min(), pe.min()), max(ac.max(), pe.max())]
        ax.plot(lim, lim, ls="--", c="gray", lw=1)
        ax.set_xlabel("pred: averaged-context read")
        ax.set_ylabel("pred: prefix-end read")
        ax.set_title(
            f"{trait}: held-out predictions, two arms (r="
            f"{results['traits'][trait]['arms_agreement_r_pred_pe_vs_pred_ac']:.2f})"
        )
        ax.legend(fontsize=8)
        ax = axes[1][col]
        d_abs_err = np.abs(pe - jm) - np.abs(ac - jm)
        for m, lab, c in ((~is_batt, "real", "#0173b2"), (is_batt, "battery", "#de8f05")):
            ax.scatter(jm[m], d_abs_err[m], s=18, alpha=0.75, label=lab, color=c)
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_xlabel("per-prefix mean judge score (0-100)")
        ax.set_ylabel("|err prefix-end| - |err avg-context|")
        ax.set_title(f"{trait}: where prefix-end is worse (>0) / better (<0)")
        ax.legend(fontsize=8)
    fig.suptitle(
        "#1092 inline: prefix-end vs 48-query-averaged context monitoring — divergence "
        f"(instruct, L{LAYER}; held-out grouped-5-fold ridge predictions of the per-prefix "
        "mean sonnet graded judge score)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "prefixend_monitoring_divergence.png"
    fig.savefig(png, dpi=180)
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_sha(),
        "source": "scripts/issue1092_prefixend_divergence.py",
        "data": "eval_results/issue_1092/inline_prefixend_monitoring/per_prefix_points.json",
        "provenance": "held-out ridge predictions (banked round artifacts); judge = "
        "claude-sonnet-4-5-20250929 graded 0-100, 5 draws temp 1.0 mean-aggregated; "
        "model completions = instruct own-policy vLLM greedy (teacher-forced read)",
    }
    with open(FIG_DIR / "prefixend_monitoring_divergence.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
