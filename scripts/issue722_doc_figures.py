"""Regenerate the three Results figures for context_vector_answer_profile.md.

Uses v_C / v_A notation throughout (matching the doc prose). Project paper style.
"""

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

set_paper_style("blog")

_DATA_PATH = "eval_results/issue_722/doc_figures/docplots_data.json"
with open(_DATA_PATH) as _f:
    DATA = json.load(_f)
OUT = Path("figures/issue_722")
OUT.mkdir(parents=True, exist_ok=True)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


SHA = _git_sha()
PAL = paper_palette(8)
N = DATA["meta"]["n_contexts"]


def _meta(fig_path: str, extra: dict) -> None:
    m = {
        "figure": fig_path,
        "source_commit": SHA,
        "source_branch": "fig-doc-plots (off origin/issue-722)",
        "model": "Qwen2.5-7B-Instruct",
        "n_contexts": N,
        "metric": DATA["meta"]["metric"],
        **extra,
    }
    Path(fig_path.replace(".png", ".meta.json")).write_text(json.dumps(m, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# RESULT 1 — full 2x2: {ridge, KRR-RBF} x {v_C=last, v_C=mean-prompt}
# ─────────────────────────────────────────────────────────────────────────────
r1 = DATA["result1"]
layers = r1["layers"]
# Full 2x2 of {ridge, KRR-RBF} x {v_C=last, v_C=mean-prompt}.
# Convention: last vs mean-prompt by COLOR; ridge solid / KRR-RBF dashed.
C_LAST = PAL[0]  # blue
C_MEANPROMPT = PAL[2]  # green/orange (Wong)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axhline(0.0, color="0.6", lw=1.0, ls="--", zorder=1)
ax.plot(
    layers,
    r1["ridge_last_pca"],
    "-o",
    color=C_LAST,
    ms=4.5,
    lw=2.0,
    label=r"Ridge   ($v_C$ = last input token)",
    zorder=4,
)
ax.plot(
    layers,
    r1["ridge_meanprompt_pca"],
    "-s",
    color=C_MEANPROMPT,
    ms=4.0,
    lw=2.0,
    label=r"Ridge   ($v_C$ = mean over prompt)",
    zorder=3,
)
ax.plot(
    layers,
    r1["krr_rbf_last_pca"],
    "--^",
    color=C_LAST,
    ms=4.5,
    lw=1.8,
    label=r"KRR-RBF ($v_C$ = last input token)",
    zorder=5,
)
ax.plot(
    layers,
    r1["krr_rbf_meanprompt_pca"],
    "--D",
    color=C_MEANPROMPT,
    ms=4.0,
    lw=1.8,
    label=r"KRR-RBF ($v_C$ = mean over prompt)",
    zorder=2,
)
ax.set_xlabel(r"layer ($v_C$ and $v_A$ read at the same layer)")
ax.set_ylabel("ratio (1 $-$ SS$_{res}$/SS$_{tot}$)\n(held-out LOCO $R^2$)")
ax.set_title(
    r"$v_C \rightarrow v_A$ per-layer predictive ratio: linear vs kernel, two $v_C$ recipes"
)
ax.set_xticks(range(0, 28, 2))
ax.legend(loc="lower center", frameon=False, fontsize=9)
fig.text(
    0.5,
    0.005,
    f"Qwen2.5-7B-Instruct · n={N} contexts · same-layer · PCA-48 $v_A$ target",
    ha="center",
    fontsize=8,
    color="0.4",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
p1 = str(OUT / "result1_ridge_recipes_krr.png")
fig.savefig(p1, dpi=200, bbox_inches="tight")
fig.savefig(p1.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
_meta(
    p1,
    {
        "curves": ["ridge_last", "ridge_meanprompt", "krr_rbf_last", "krr_rbf_meanprompt"],
        "peak": {
            "layer": 18,
            "ridge_last": r1["ridge_last_pca"][18],
            "ridge_meanprompt": r1["ridge_meanprompt_pca"][18],
            "krr_rbf_last": r1["krr_rbf_last_pca"][18],
            "krr_rbf_meanprompt": r1["krr_rbf_meanprompt_pca"][18],
        },
    },
)
print("R1:", p1)
print(
    f"   peak L18: ridge_last={r1['ridge_last_pca'][18]:.3f} "
    f"ridge_meanprompt={r1['ridge_meanprompt_pca'][18]:.3f} "
    f"krr_rbf_last={r1['krr_rbf_last_pca'][18]:.3f} "
    f"krr_rbf_meanprompt={r1['krr_rbf_meanprompt_pca'][18]:.3f}"
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT 2 — cross-layer ridge | KRR grids (DATA ONLY: two clean heatmaps + colorbar)
# All interpretation/annotation removed — no stat boxes, no best-cell circle, no
# v_C=v_A diagonal markers, no marker legend, neutral suptitle + plain panel titles.
# ─────────────────────────────────────────────────────────────────────────────
r2 = DATA["result2"]
gl = r2["grid_layers"]
R = np.array(r2["ridge_grid"])  # rows = v_C layer, cols = v_A layer
K = np.array(r2["krr_rbf_grid"])
vmin = min(R.min(), K.min())
vmax = max(R.max(), K.max())

# The "blog" style enables constrained_layout via rcParams; turn it off for this figure
# so the manual subplots_adjust below keeps the plain panel titles clear of the suptitle.
_prev_cl = plt.rcParams["figure.constrained_layout.use"]
_prev_al = plt.rcParams["figure.autolayout"]
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["figure.autolayout"] = False
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), layout=None)
fig.set_layout_engine("none")
fig.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.91, wspace=0.20)
for ax, M, name in [(axes[0], R, "Ridge"), (axes[1], K, "KRR-RBF")]:
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(gl)))
    ax.set_xticklabels(gl)
    ax.set_yticks(range(len(gl)))
    ax.set_yticklabels(gl)
    ax.set_xlabel(r"$v_A$ layer (read-out layer)")
    ax.set_ylabel(r"$v_C$ layer (source layer)")
    ax.set_title(name)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
cbar.set_label("ratio (1 $-$ SS$_{res}$/SS$_{tot}$) (held-out LOCO $R^2$)")
fig.suptitle(
    r"$v_C \rightarrow v_A$ cross-layer prediction (Ridge | KRR-RBF)",
    fontsize=12.5,
    y=0.99,
    va="top",
)
p2 = str(OUT / "result2_cross_layer.png")
fig.savefig(p2, dpi=200)
fig.savefig(p2.replace(".png", ".pdf"))
plt.close(fig)
plt.rcParams["figure.constrained_layout.use"] = _prev_cl
plt.rcParams["figure.autolayout"] = _prev_al
_meta(
    p2,
    {
        "ridge_stats": r2["ridge_stats"],
        "krr_stats": r2["krr_stats"],
    },
)
print("R2:", p2)
print(
    f"   RIDGE diag={r2['ridge_stats']['mean_diag']:.3f} "
    f"off={r2['ridge_stats']['mean_offdiag']:.3f} best={r2['ridge_stats']['best_cell']}"
)
print(
    f"   KRR   diag={r2['krr_stats']['mean_diag']:.3f} off={r2['krr_stats']['mean_offdiag']:.3f} "
    f"best={r2['krr_stats']['best_cell']}"
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT 3 — skill vs n contexts (rising, mark still-rising at n=50)
# ─────────────────────────────────────────────────────────────────────────────
r3 = DATA["result3"]
ng = r3["n_grid"]
sk = r3["skill"]
lo = r3["ci_lo"]
hi = r3["ci_hi"]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
x = range(len(ng))
ax.fill_between(x, lo, hi, color=PAL[0], alpha=0.18, label="95% CI (resampled)")
ax.plot(
    x,
    sk,
    "-o",
    color=PAL[0],
    ms=8,
    lw=2.2,
    label=f"Ridge ratio ($v_C \\rightarrow v_A$) @ L{r3['layer']}",
)
for xi, s in zip(x, sk, strict=True):
    ax.annotate(
        f"{s:.2f}",
        (xi, s),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
# slope annotation: still rising from 30 -> 50
ax.annotate(
    "still rising at n=50\n(no plateau)",
    xy=(2, sk[2]),
    xytext=(1.35, 0.66),
    fontsize=9,
    color="0.35",
    arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2),
)
ax.set_xticks(list(x))
ax.set_xticklabels(ng)
ax.set_xlabel("number of contexts (n)")
ax.set_ylabel("ratio (1 $-$ SS$_{res}$/SS$_{tot}$)\n(held-out LOCO $R^2$) @ L18")
ax.set_title(r"$v_C \rightarrow v_A$ ratio keeps rising with more contexts")
ax.set_ylim(0.45, 0.90)
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
p3 = str(OUT / "result3_skill_vs_n.png")
fig.savefig(p3, dpi=200, bbox_inches="tight")
fig.savefig(p3.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
_meta(
    p3,
    {"n_grid": ng, "skill": sk, "ci_lo": lo, "ci_hi": hi, "layer": r3["layer"], "note": r3["note"]},
)
print("R3:", p3)
print(f"   skill {sk} at n={ng}")
