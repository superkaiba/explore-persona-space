#!/usr/bin/env python3
"""Pre-registered search for relationships that hold across the issue-2588 panel.

Candidate PREDICTORS (model- or map-level, fixed before looking):
  aa_index, gpqa_accuracy (own rollouts), hidden_width, n_layers, params_b,
  participation_ratio_x (intrinsic dimension of the context representation at
  the selected layer), ceiling_two_draw (answer-representation reliability
  ceiling), length_only_r2 (R2 explained by answer length alone), generation
  (Qwen release number; Qwen rows only).
Candidate OUTCOMES:
  rank_frac (primary: reduced-rank rank at +10% error / d), rank_abs,
  compressibility (area under retained-R2 curve to 25% of width), test_r2,
  ceiling_normalized_r2 (test_r2 / two-draw ceiling), acc1_calibrated,
  fitted_dirs_90pct (directions holding 90% of the fitted-output variance).

Method: Spearman for every (predictor, outcome) pair, separately per arm
(prompt read / end-of-thought).  Family-wise correction over the whole grid by
a max-|rho| permutation test (predictor values permuted across maps within the
arm; 20,000 draws).  Paired arm contrast (end-of-thought minus prompt) within
model by exact sign test.  Log-log scaling of absolute rank against width and
parameter count.  Leave-one-out sign stability for every pair.  Everything is
written to eval_results/issue_2588/rank_relationships.json and one figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2588_mapping_rank_vs_capability as MR  # noqa: E402

REPO = MR.REPO
OUT_JSON = REPO / "eval_results" / "issue_2588" / "rank_relationships.json"
FIG = REPO / "figures" / "issue_2588" / "rank_relationships.png"
SWEEP_RRR = REPO / "eval_results" / "issue_2588" / "rank_threshold_sweep_rrr.json"
N_PERM = 20_000
SEED = 2588
ARMS = ("no-thinking", "end-of-thought")
ARM_TITLE = {"no-thinking": "Prompt read", "end-of-thought": "End-of-thought read"}

MODEL_META = {
    # model_key: (params_b, n_layers, qwen_generation or None)
    "q35_0p8b": (0.8, 24, 3.5),
    "q35_2b": (2.0, 24, 3.5),
    "q35_4b": (4.0, 32, 3.5),
    "q35_9b": (9.0, 32, 3.5),
    "q35_27b": (27.0, 64, 3.5),
    "q36_27b": (27.0, 64, 3.6),
    "q38_27b": (27.0, 64, 3.8),
    "o3_7b_i": (7.0, 32, None),
    "o3_7b_t": (7.0, 32, None),
    "o31_32b_i": (32.0, 64, None),
    "o31_32b_t": (32.0, 64, None),
    "o3_32b_t": (32.0, 64, None),
    "q25_32b": (32.0, 64, 2.5),
    "q3_32b": (32.0, 64, 3.0),
    "qwq_32b": (32.0, 64, 3.0),
}
PREDICTORS = (
    "aa_index",
    "gpqa_accuracy",
    "hidden_width",
    "n_layers",
    "params_b",
    "participation_ratio_x",
    "ceiling_two_draw",
    "length_only_r2",
    "generation",
)
OUTCOMES = (
    "rank_frac",
    "rank_abs",
    "compressibility",
    "test_r2",
    "ceiling_normalized_r2",
    "acc1_calibrated",
    "fitted_dirs_90pct",
)
PRETTY = {
    "aa_index": "AA index",
    "gpqa_accuracy": "GPQA accuracy (own)",
    "hidden_width": "hidden width d",
    "n_layers": "depth (layers)",
    "params_b": "parameters (B)",
    "participation_ratio_x": "context intrinsic dim (PR)",
    "ceiling_two_draw": "answer reliability ceiling",
    "length_only_r2": "length-only R²",
    "generation": "Qwen release (2.5→3.8)",
    "rank_frac": "rank needed (% of d)",
    "rank_abs": "rank needed (absolute)",
    "compressibility": "compressibility index",
    "test_r2": "test R²",
    "ceiling_normalized_r2": "R² ÷ ceiling",
    "acc1_calibrated": "calibrated acc@1",
    "fitted_dirs_90pct": "fitted-output dims (90%)",
}


def _num(x: Any) -> float:
    if isinstance(x, dict):
        for k in ("value", "r2", "mean", "ceiling", "pooled_r2"):
            if k in x and isinstance(x[k], (int, float)):
                return float(x[k])
        vals = [v for v in x.values() if isinstance(v, (int, float))]
        if len(vals) == 1:
            return float(vals[0])
        raise ValueError(f"cannot reduce {x}")
    return float(x)


def build_table() -> list[dict[str, Any]]:
    payload = json.loads(MR.DEFAULT_OUT.read_text(encoding="utf-8"))
    sweep = {r["key"]: r for r in json.loads(SWEEP_RRR.read_text(encoding="utf-8"))["maps"]}
    by_key = {m.key: m for m in MR.MAPS}
    rows = []
    for rec in payload["maps"]:
        spec = by_key[rec["key"]]
        fit = MR._fit_record(spec)
        resid_path = f"{MR.PANEL_PREFIX}/fits/{spec.cell}/resid_{spec.position}.json"
        resid = json.loads(MR._download(resid_path).read_text(encoding="utf-8"))
        params_b, n_layers, generation = MODEL_META[spec.model_key]
        ceiling = _num(fit["ceiling_two_draw_at_star"])
        test_r2 = float(rec["mapping_performance"]["test_r2"])
        rows.append(
            {
                "key": rec["key"],
                "model": rec["model"],
                "model_key": spec.model_key,
                "family": rec["family"],
                "arm": rec["arm"],
                # predictors
                "aa_index": float(rec["aa_index"]),
                "gpqa_accuracy": float(rec["measured_capability"]["accuracy"]),
                "hidden_width": float(rec["dimension"]),
                "n_layers": float(n_layers),
                "params_b": float(params_b),
                "participation_ratio_x": _num(fit["participation_ratio_x_at_star"]),
                "ceiling_two_draw": ceiling,
                "length_only_r2": float(resid["length_only_test_r2"]),
                "generation": float(generation) if generation is not None else float("nan"),
                # outcomes
                "rank_frac": float(rec["operational_rank"]["rank_fraction"]),
                "rank_abs": float(rec["operational_rank"]["rank"]),
                "compressibility": float(sweep[rec["key"]]["compressibility_index_25pct"]),
                "test_r2": test_r2,
                "ceiling_normalized_r2": test_r2 / ceiling if ceiling > 0 else float("nan"),
                "acc1_calibrated": float(
                    rec["mapping_performance"]["test_retrieval_acc1_cos_calibrated"]
                ),
                "fitted_dirs_90pct": float(
                    rec["fitted_output_spectrum"]["directions_for_90pct_variance"]
                ),
            }
        )
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def _standardized_ranks(a: np.ndarray) -> np.ndarray:
    r = rankdata(a, axis=0).astype(float)
    r = r - r.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(r, axis=0, keepdims=True)
    norm[norm == 0] = np.nan
    return r / norm


def _maxt_block(
    arm_rows: list[dict[str, Any]],
    predictors: tuple[str, ...],
    outcomes: tuple[str, ...],
    rng: np.random.Generator,
    subset: str,
) -> list[dict[str, Any]]:
    """Joint-relabeling max-|rho| permutation test over a predictor x outcome grid.

    One permutation of the map labels is applied to the WHOLE predictor block
    per draw (Westfall-Young), so the max statistic reflects the actual search."""
    X = np.array([[r[p] for p in predictors] for r in arm_rows], dtype=float)
    Y = np.array([[r[o] for o in outcomes] for r in arm_rows], dtype=float)
    keep = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X, Y = X[keep], Y[keep]
    n = int(keep.sum())
    if n < 5:
        return []
    ZX, ZY = _standardized_ranks(X), _standardized_ranks(Y)
    obs = ZX.T @ ZY  # P x O Spearman matrix
    ok = ~np.isnan(obs)
    maxes = np.empty(N_PERM)
    exceed = np.zeros_like(obs)
    for b in range(N_PERM):
        r = ZX[rng.permutation(n)].T @ ZY
        r = np.where(ok, np.abs(r), -1.0)
        maxes[b] = r.max()
        exceed += r >= np.abs(np.nan_to_num(obs)) - 1e-12
    results = []
    for i, p in enumerate(predictors):
        for j, o in enumerate(outcomes):
            if not ok[i, j]:
                continue
            x, y = X[:, i], Y[:, j]
            loo = [_spearman(np.delete(x, k), np.delete(y, k)) for k in range(n)]
            results.append(
                {
                    "predictor": p,
                    "outcome": o,
                    "subset": subset,
                    "n": n,
                    "rho": float(obs[i, j]),
                    "p_uncorrected": float(exceed[i, j] / N_PERM),
                    "p_familywise_maxT": float((maxes >= abs(obs[i, j]) - 1e-12).mean()),
                    "loo_sign_consistent": bool(all(np.sign(v) == np.sign(obs[i, j]) for v in loo)),
                    "loo_rho_range": [float(min(loo)), float(max(loo))],
                }
            )
    return results


def grid_with_maxt(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Per-arm Spearman grid with joint max-|rho| permutation correction.

    The main grid uses every map in the arm and every predictor except the
    Qwen release number; that predictor gets its own Qwen-only grid."""
    main_preds = tuple(p for p in PREDICTORS if p != "generation")
    out: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        results = _maxt_block(arm_rows, main_preds, OUTCOMES, rng, "all maps")
        qwen_rows = [r for r in arm_rows if r["family"] != "OLMo"]
        results += _maxt_block(qwen_rows, ("generation",), OUTCOMES, rng, "Qwen only")
        out[arm] = {
            "n_maps": len(arm_rows),
            "n_tests": len(results),
            "n_tests_main_grid": len(main_preds) * len(OUTCOMES),
            "results": sorted(results, key=lambda r: r["p_familywise_maxT"]),
        }
    return out


def paired_arm_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], {})[r["arm"]] = r
    paired = {k: v for k, v in by_model.items() if len(v) == 2}
    out: dict[str, Any] = {"n_models": len(paired), "models": sorted(paired)}
    for o in OUTCOMES:
        diffs = {k: v["end-of-thought"][o] - v["no-thinking"][o] for k, v in paired.items()}
        vals = np.array(list(diffs.values()), dtype=float)
        n_neg = int((vals < 0).sum())
        n_pos = int((vals > 0).sum())
        test = binomtest(n_pos, n_pos + n_neg, 0.5) if n_pos + n_neg else None
        out[o] = {
            "eot_minus_prompt": {k: float(v) for k, v in diffs.items()},
            "n_positive": n_pos,
            "n_negative": n_neg,
            "median_diff": float(np.median(vals)),
            "sign_test_p_two_sided": float(test.pvalue) if test else None,
        }
    return out


def scaling(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """log(absolute rank) = a + b log(x) per arm, bootstrap CI on b."""
    out: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = {}
        for x_name in ("hidden_width", "params_b"):
            x = np.log(np.array([r[x_name] for r in arm_rows]))
            y = np.log(np.array([r["rank_abs"] for r in arm_rows]))
            b, a = np.polyfit(x, y, 1)
            boots = []
            for _ in range(5000):
                idx = rng.integers(0, len(x), len(x))
                if np.std(x[idx]) == 0:
                    continue
                boots.append(np.polyfit(x[idx], y[idx], 1)[0])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            resid = y - (a + b * x)
            r2 = 1 - resid.var() / y.var()
            out[arm][x_name] = {
                "slope": float(b),
                "slope_ci95_bootstrap": [float(lo), float(hi)],
                "intercept": float(a),
                "r2_loglog": float(r2),
                "n": int(len(x)),
                "reading": (
                    "slope 0 = a fixed number of directions regardless of size; slope 1 = a fixed FRACTION"
                ),
            }
    return out


def render(
    rows: list[dict[str, Any]], grid: dict[str, Any], paired: dict[str, Any], scal: dict[str, Any]
) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.30)
    # heatmaps
    for col, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, col])
        mat = np.full((len(PREDICTORS), len(OUTCOMES)), np.nan)
        fw = np.full_like(mat, np.nan)
        un = np.full_like(mat, np.nan)
        for r in grid[arm]["results"]:
            i, j = PREDICTORS.index(r["predictor"]), OUTCOMES.index(r["outcome"])
            mat[i, j], fw[i, j], un[i, j] = r["rho"], r["p_familywise_maxT"], r["p_uncorrected"]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "·", ha="center", va="center", color="#999999")
                    continue
                mark = "**" if fw[i, j] < 0.05 else ("*" if un[i, j] < 0.05 else "")
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:+.2f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if abs(mat[i, j]) > 0.6 else "black",
                )
        ax.set_xticks(
            range(len(OUTCOMES)), [PRETTY[o] for o in OUTCOMES], rotation=35, ha="right", fontsize=7
        )
        # Row labels are shared: draw them on the left heatmap only so the two
        # panels don't collide.
        if col == 0:
            ax.set_yticks(range(len(PREDICTORS)), [PRETTY[p] for p in PREDICTORS], fontsize=7)
        else:
            ax.set_yticks(range(len(PREDICTORS)), [""] * len(PREDICTORS))
        ax.set_title(
            f"{ARM_TITLE[arm]} (n = {grid[arm]['n_maps']} maps, {grid[arm]['n_tests']} tests)",
            fontsize=9,
        )
    cax = fig.add_subplot(gs[0, 2])
    cax.axis("off")
    cbar_ax = cax.inset_axes([0.12, 0.40, 0.10, 0.55])
    fig.colorbar(im, cax=cbar_ax, label="Spearman ρ")
    cax.text(
        0.0,
        0.02,
        "** family-wise p < 0.05 (max-|ρ| permutation over the grid)\n*  uncorrected p < 0.05 only\n·  not testable (n < 5 or constant)",
        transform=cax.transAxes,
        fontsize=7.5,
        va="bottom",
    )
    # scaling panel
    ax = fig.add_subplot(gs[1, 0])
    colors = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        x = np.array([r["hidden_width"] for r in arm_rows])
        y = np.array([r["rank_abs"] for r in arm_rows])
        ax.scatter(
            x,
            y,
            color=colors[arm],
            s=26,
            label=f"{ARM_TITLE[arm]}: slope {scal[arm]['hidden_width']['slope']:.2f} [{scal[arm]['hidden_width']['slope_ci95_bootstrap'][0]:.2f}, {scal[arm]['hidden_width']['slope_ci95_bootstrap'][1]:.2f}]",
        )
        xs = np.linspace(np.log(900), np.log(5600), 20)
        s = scal[arm]["hidden_width"]
        ax.plot(np.exp(xs), np.exp(s["intercept"] + s["slope"] * xs), color=colors[arm], lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hidden width d")
    ax.set_ylabel("rank needed (absolute, +10% error)")
    ax.set_title("Absolute rank vs width (log-log)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax.grid(color="#e5e5e5", lw=0.5)
    # paired arm panel
    ax = fig.add_subplot(gs[1, 1])
    by_model: dict[str, dict[str, Any]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], {})[r["arm"]] = r
    for k, v in by_model.items():
        if len(v) < 2:
            continue
        a, b = 100 * v["no-thinking"]["rank_frac"], 100 * v["end-of-thought"]["rank_frac"]
        ax.plot([0, 1], [a, b], color="#555555", lw=0.9, alpha=0.8)
        ax.scatter(
            [0, 1], [a, b], color=[colors["no-thinking"], colors["end-of-thought"]], s=22, zorder=3
        )
        ax.annotate(v["no-thinking"]["model"], (1.03, b), fontsize=6.2, va="center")
    pr = paired["rank_frac"]
    ax.set_xticks([0, 1], ["prompt read", "end-of-thought"])
    ax.set_xlim(-0.2, 1.6)
    ax.set_ylabel("rank needed (% of d)")
    ax.set_title(
        f"Same model, two reads: {pr['n_negative']} of {pr['n_negative'] + pr['n_positive']} fall\n(sign test p = {pr['sign_test_p_two_sided']:.3f})",
        fontsize=9,
    )
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    # paired R2 panel
    ax = fig.add_subplot(gs[1, 2])
    for k, v in by_model.items():
        if len(v) < 2:
            continue
        a, b = v["no-thinking"]["test_r2"], v["end-of-thought"]["test_r2"]
        ax.plot([0, 1], [a, b], color="#555555", lw=0.9, alpha=0.8)
        ax.scatter(
            [0, 1], [a, b], color=[colors["no-thinking"], colors["end-of-thought"]], s=22, zorder=3
        )
        ax.annotate(v["no-thinking"]["model"], (1.03, b), fontsize=6.2, va="center")
    pr = paired["test_r2"]
    ax.set_xticks([0, 1], ["prompt read", "end-of-thought"])
    ax.set_xlim(-0.2, 1.6)
    ax.set_ylabel("held-out test R²")
    ax.set_title(
        f"Same model, two reads: {pr['n_positive']} of {pr['n_negative'] + pr['n_positive']} rise\n(sign test p = {pr['sign_test_p_two_sided']:.3f})",
        fontsize=9,
    )
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    fig.suptitle(
        "What predicts the map's rank and quality? Pre-registered grid, family-wise corrected",
        x=0.02,
        ha="left",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    FIG.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "title": "Rank relationships: pre-registered grid",
                "public_url": f"https://eps.superkaiba.com/tasks/2588/figure/{FIG.name}",
                "source_data": str(OUT_JSON.relative_to(REPO)),
                "method": "Spearman per arm; max-|rho| permutation (20k) over the predictor x outcome grid; paired sign tests within model; log-log scaling of absolute rank",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if "--render-only" in sys.argv[1:]:
        saved = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        render(saved["table"], saved["grid"], saved["paired_arm_contrast"], saved["scaling"])
        print(f"re-rendered {FIG} from {OUT_JSON}")
        return
    rng = np.random.default_rng(SEED)
    rows = build_table()
    grid = grid_with_maxt(rows, rng)
    paired = paired_arm_contrast(rows)
    scal = scaling(rows, rng)
    out = {
        "predictors": list(PREDICTORS),
        "outcomes": list(OUTCOMES),
        "n_permutations": N_PERM,
        "table": rows,
        "grid": grid,
        "paired_arm_contrast": paired,
        "scaling": scal,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    render(rows, grid, paired, scal)
    # console summary
    for arm in ARMS:
        print(
            f"\n== {ARM_TITLE[arm]}: n={grid[arm]['n_maps']} maps, {grid[arm]['n_tests']} tests; family-wise p < 0.10 shown"
        )
        for r in grid[arm]["results"]:
            if r["p_familywise_maxT"] < 0.10:
                print(
                    f"   {PRETTY[r['predictor']]:28s} -> {PRETTY[r['outcome']]:26s} rho={r['rho']:+.2f} n={r['n']} p_fw={r['p_familywise_maxT']:.3f} p_unc={r['p_uncorrected']:.4f} LOO-stable={r['loo_sign_consistent']}"
                )
    print(
        "\n== paired arm contrast (end-of-thought minus prompt), models with both arms:",
        paired["n_models"],
    )
    for o in OUTCOMES:
        p = paired[o]
        print(
            f"   {PRETTY[o]:26s} +{p['n_positive']}/-{p['n_negative']} median diff {p['median_diff']:+.4f} sign-test p={p['sign_test_p_two_sided']:.3f}"
        )
    print("\n== scaling of absolute rank")
    for arm in ARMS:
        for xn in ("hidden_width", "params_b"):
            s = scal[arm][xn]
            print(
                f"   {ARM_TITLE[arm]:20s} log rank ~ log {xn:13s}: slope {s['slope']:+.2f} CI [{s['slope_ci95_bootstrap'][0]:+.2f}, {s['slope_ci95_bootstrap'][1]:+.2f}] R2 {s['r2_loglog']:.2f}"
            )
    print(f"\nwrote {OUT_JSON} and {FIG}")


if __name__ == "__main__":
    main()
