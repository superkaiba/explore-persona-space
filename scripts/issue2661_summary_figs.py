"""Issue #2661 — summary figures for the task body (VM, CPU, seconds).

Reads ONLY committed / staged JSON artifacts (no recomputation):
  data/issue_2661/results/{sae_metrics,map_ridge,map_mlp,controls,perfeature}/*.json
  eval_results/issue_2661/embedding_coverage.json
Writes figures/issue_2661/{sae_metrics,map_routes,coverage_by_field}.png (+ .meta.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before any heavy import so the shared-VM thread caps bind (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "issue_2661" / "results"
EVAL = ROOT / "eval_results" / "issue_2661"
FIG = ROOT / "figures" / "issue_2661"


def _j(p: Path) -> dict:
    assert p.exists(), f"missing artifact: {p}"
    return json.loads(p.read_text())


def _save(fig, name: str, sources: list[Path]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    (FIG / f"{name}.meta.json").write_text(
        json.dumps({"sources": [str(s.relative_to(ROOT)) for s in sources]}, indent=1)
    )
    plt.close(fig)
    print(f"[fig] {out}")


def fig_sae_metrics() -> None:
    srcs = [
        RES / "sae_metrics" / "sae_metrics_ctx.json",
        RES / "sae_metrics" / "sae_metrics_answer.json",
    ]
    ctx, ans = (_j(s) for s in srcs)
    der = ctx["reference_numbers"]["der_paper_nmse"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    labels = [
        "context SAE\n(new)",
        "answer SAE\n(#2552)",
        "Der et al.\nturn-avg",
        "Der et al.\nper-token",
    ]
    nmse = [
        ctx["splits"]["holdout"]["nmse_raw"],
        ans["splits"]["holdout"]["nmse_raw"],
        der["turn_averaged"],
        der["per_token"],
    ]
    axes[0].bar(labels, nmse, color=["#1f77b4", "#ff7f0e", "#999999", "#cccccc"])
    axes[0].set_title("holdout raw nMSE (lower is better)", fontsize=10)
    axes[0].tick_params(axis="x", labelsize=8)
    fve = [ctx["splits"]["holdout"]["variance_fve"], ans["splits"]["holdout"]["variance_fve"]]
    axes[1].bar(labels[:2], fve, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("holdout variance explained (FVE)", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=8)
    dead = [ctx["dead_features"]["n_dead_on_fit_rows"], ans["dead_features"]["n_dead_on_fit_rows"]]
    alive = [32768 - d for d in dead]
    axes[2].bar(
        labels[:2],
        alive,
        color=["#1f77b4", "#ff7f0e"],
        label="fires at least once on 120k fit rows",
    )
    axes[2].bar(labels[:2], dead, bottom=alive, color="#dddddd", label="never fires (dead)")
    axes[2].set_title("census of the 32,768 features", fontsize=10)
    axes[2].legend(fontsize=7, loc="lower center")
    axes[2].tick_params(axis="x", labelsize=8)
    _save(fig, "sae_metrics", srcs)


def fig_map_routes() -> None:
    srcs = [
        RES / "map_ridge" / "map_ridge_metrics.json",
        RES / "map_mlp" / "map_mlp_metrics.json",
        RES / "controls" / "controls.json",
        RES / "controls" / "knn_retrieval.json",
        RES / "perfeature" / "perfeature_summary.json",
    ]
    ridge, mlp, ctrl, knn, pf = (_j(s) for s in srcs)
    routes = [
        ("ridge\n(ctx features)", ridge["holdout_pooled_r2"], "ridge"),
        ("MLP\n(ctx features)", mlp["holdout_pooled_r2"], "mlp"),
        (
            "ridge\n(dense ctx state)",
            ctrl["routes"]["dense_input_ridge"]["holdout_pooled_r2"],
            "densein",
        ),
        ("composed\n(zero-fit)", None, "composed"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    names = [r[0] for r in routes if r[1] is not None]
    vals = [r[1] for r in routes if r[1] is not None]
    axes[0].bar(names, vals, color=["#1f77b4", "#2ca02c", "#9467bd"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("holdout pooled R² (all 32,768 answer features)", fontsize=10)
    axes[0].tick_params(axis="x", labelsize=8)
    ks = ["1", "5", "10"]
    for label, _v, key in routes:
        acc = [knn["routes"][key]["cosine"]["acc_at_k"][k] for k in ks]
        axes[1].plot([int(k) for k in ks], acc, marker="o", label=label.replace("\n", " "))
    axes[1].axhline(knn["chance"]["1"], color="grey", ls=":", lw=1, label="chance")
    axes[1].set_xticks([1, 5, 10])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("k")
    axes[1].set_title("kNN retrieval, cosine (holdout pool 20,000)", fontsize=10)
    axes[1].legend(fontsize=7)
    deciles = list(range(1, 11))
    for key, lab, col in (
        ("ridge", "ridge", "#1f77b4"),
        ("mlp", "MLP", "#2ca02c"),
        ("densein", "dense-state ridge", "#9467bd"),
        ("composed", "composed", "#d62728"),
    ):
        if key in pf["routes"]:
            axes[2].plot(
                deciles,
                pf["routes"][key]["r2_median_by_fit_count_decile"],
                marker=".",
                label=lab,
                color=col,
            )
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[2].set_xlabel("answer-feature firing-count decile (1 = rarest)")
    axes[2].set_title("per-feature holdout R², median by decile", fontsize=10)
    axes[2].legend(fontsize=7)
    _save(fig, "map_routes", srcs)


def fig_coverage() -> None:
    src = EVAL / "embedding_coverage.json"
    cov = _j(src)
    items = sorted(cov["per_field_mean"].items(), key=lambda kv: kv[1]["1"] or 0)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh([k for k, _ in items], [v["1"] for _, v in items], color="#1f77b4", label="top-1")
    ax.barh(
        [k for k, _ in items],
        [v["5"] for _, v in items],
        color="#aec7e8",
        height=0.4,
        label="top-5",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean cosine(summary field, listed context-feature descriptions)")
    ax.set_title("Der coverage metric by summary field (1,983 turns)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "coverage_by_field", [src])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["sae", "map", "coverage"], default=None)
    a = ap.parse_args()
    if a.only in (None, "sae"):
        fig_sae_metrics()
    if a.only in (None, "map"):
        fig_map_routes()
    if a.only in (None, "coverage"):
        fig_coverage()


if __name__ == "__main__":
    main()
