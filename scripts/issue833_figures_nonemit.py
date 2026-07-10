# ruff: noqa: RUF001  # minus sign in figure text intentional
"""Figures for task #833 follow-up round 2 (nonverbatim-profile-ablation).

Reads ONLY the persisted round outputs under ``eval_results/issue_833/``
(emission_rate/, chain_rho_nonemit/, joined_cache/, analysis_tensors_nonemit*/
analysis_tensors_matchedN/) plus the committed leakage target E. The per-cell
scatter panel recomputes the ridge LOCO chain predictions at L14 through the
SAME loader + estimators as scripts/issue833_chain_rho_nonemit.py and ASSERTS
the recomputed Spearman matches the persisted JSON value (tol 1e-2), so the
per-unit view is provably the data behind the persisted aggregate.

Run from the issue-833 worktree root:
    set -a && source .env && set +a && \
    OMP_NUM_THREADS=8 uv run python scripts/issue833_figures_nonemit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RES = PROJECT / "eval_results/issue_833"
LAYERS = [7, 14, 21]

set_paper_style("blog")
C_OFF = paper_palette_role("baseline")
C_ON = paper_palette_role("primary")
C_CTRL = paper_palette_role("control")
C_NEU = paper_palette_role("neutral")

SRC_LABEL = {
    "default": "default assistant",
    "binst_fact": "fact instruction",
    "fmt_code": "code format",
    "fmt_json": "JSON format",
    "icl_k2": "2-demo prefix",
    "icl_k8": "8-demo prefix",
    "reph_casual": "casual rephrase",
    "reph_imp": "imperative rephrase",
    "reph_polite": "polite rephrase",
    "sp_doctor": "doctor persona",
    "sp_ph1": "persona 1",
    "sp_ph2": "persona 2",
    "sp_swe": "engineer persona",
    "wc_long_write": "WildChat writing",
    "wc_short_advice": "WildChat advice",
    "wc_short_code": "WildChat code",
}


def _load_ne_module():
    """Import the round's chain-fit script as a module (loader + estimators)."""
    spec = importlib.util.spec_from_file_location(
        "issue833_chain_rho_nonemit", SCRIPTS / "issue833_chain_rho_nonemit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _chain_json(layer: int) -> dict:
    with open(RES / f"chain_rho_nonemit/fact_L{layer}.json") as f:
        return json.load(f)


def _manifest() -> dict:
    with open(RES / "emission_rate/retention_manifest.json") as f:
        return json.load(f)


def _predictor() -> dict:
    with open(RES / "emission_rate/emission_predictor.json") as f:
        return json.load(f)


# ── Figure 1: retained-subset chain forest + paired diffs ─────────────────────
ARMS = [
    ("on_full_ret", "full text, all 30/cell (trained map)", C_ON, "o", True),
    ("on_full_matchedN", "full text, matched N/cell (trained map)", C_ON, "D", True),
    ("ctrl_full_ret", "full text (base-weights control)", C_CTRL, "o", True),
    ("off_full_ret", "base-written text (off-policy)", C_OFF, "o", True),
    ("on_nonemit", "non-emission only (trained map)", C_ON, "o", False),
    ("ctrl_nonemit", "non-emission only (base-weights control)", C_CTRL, "o", False),
    ("on_nonemit_eq5", "non-emission, 5/cell (trained map)", C_ON, "^", False),
]
DIFFS = [
    ("on_nonemit_minus_on_full_matchedN", "non-emission − matched-N full text", C_ON, "o"),
    ("on_nonemit_eq5_minus_on_full_matchedN", "5/cell non-emission − matched-N", C_ON, "^"),
    ("on_nonemit_minus_off_full_ret", "non-emission − off-policy", C_OFF, "o"),
]


def fig_forest() -> None:
    """Left: 7 retained-subset chain arms per layer; right: paired differences."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), width_ratios=[1.35, 1.0])
    data = {li: _chain_json(li) for li in LAYERS}

    ax = axes[0]
    n = len(ARMS)
    for i, li in enumerate(LAYERS):
        y0 = len(LAYERS) - 1 - i
        for k, (arm, lab, col, mk, filled) in enumerate(ARMS):
            ci = data[li]["arms"][arm]["ci_ridge"]
            y = y0 + (k - (n - 1) / 2) * -0.105
            kw = dict(fmt=mk, color=col, ms=5.5, capsize=2.2, lw=1.3)
            if not filled:
                kw.update(mfc="white", markeredgewidth=1.3)
            ax.errorbar(
                ci["point"],
                y,
                xerr=[[ci["point"] - ci["ci_lo"]], [ci["ci_hi"] - ci["point"]]],
                label=lab if i == 0 else None,
                **kw,
            )
    ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
    for ysep in [0.5, 1.5]:
        ax.axhline(ysep, color="0.85", lw=0.8)
    ax.set_yticks([2, 1, 0], [f"layer {li}" for li in LAYERS])
    ax.set_xlabel("held-out chain correlation with leakage E (Spearman)")
    ax.set_title("chain correlation, 291 retained cells", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.6)

    ax = axes[1]
    for i, li in enumerate(LAYERS):
        y0 = len(LAYERS) - 1 - i
        for k, (key, lab, col, mk) in enumerate(DIFFS):
            d = data[li]["paired_diffs"][key]
            y = y0 + (k - 1) * -0.18
            kw = dict(fmt=mk, color=col, ms=5.5, capsize=2.2, lw=1.3)
            if k == 1:
                kw.update(mfc="white", markeredgewidth=1.3)
            ax.errorbar(
                d["point"],
                y,
                xerr=[[d["point"] - d["ci_lo"]], [d["ci_hi"] - d["point"]]],
                label=lab if i == 0 else None,
                **kw,
            )
    ax.axvline(0.0, color="0.6", lw=0.9, ls="--")
    for ysep in [0.5, 1.5]:
        ax.axhline(ysep, color="0.85", lw=0.8)
    ax.set_yticks([2, 1, 0], ["" for _ in LAYERS])
    ax.set_xlabel("paired chain-correlation difference")
    ax.set_title("paired differences per layer", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.6)
    fig.tight_layout()
    savefig_paper(fig, "issue_833/chain_rho_nonemit_paired", dir="figures/")
    plt.close(fig)


# ── Figure 2: emission fraction vs E, per cell + source means ─────────────────
def fig_emission_scatter(cell_keys: list[str], E: np.ndarray) -> None:
    """Left: 480-cell emission fraction vs E; right: 16 labeled source means."""
    man = _manifest()["cells"]
    frac = np.array([man[k]["n_emission"] / man[k]["total"] for k in cell_keys])
    pred = _predictor()
    rho_cell = pred["dv_a_emission_vs_E"]["pinned_span"]
    rho_src = pred["dv_a_source_mean_vs_E"]
    r_chk, _ = spearmanr(frac, E)
    assert abs(r_chk - rho_cell["point"]) < 1e-6, (r_chk, rho_cell["point"])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ax.scatter(frac, E, s=14, alpha=0.45, color=C_ON, linewidths=0)
    ax.set_xlabel("taught-sentence emission fraction of the cell's 30 answers")
    ax.set_ylabel("measured leakage E (trained − base)")
    ax.set_title(
        f"480 cells: Spearman {rho_cell['point']:+.2f} "
        f"[{rho_cell['ci_lo']:+.2f}, {rho_cell['ci_hi']:+.2f}]",
        fontsize=11,
    )

    ax = axes[1]
    srcs = sorted({k.split("/", 1)[1].split("__")[0] for k in cell_keys})
    pts = []
    for s in srcs:
        idx = [i for i, k in enumerate(cell_keys) if k.split("/", 1)[1].split("__")[0] == s]
        pts.append((float(np.mean(frac[idx])), float(np.mean(E[idx])), s))
    # greedy vertical de-overlap of labels (points stay put)
    label_y: list[float] = []
    for _, fy, _ in sorted(pts, key=lambda p: p[1]):
        y = fy
        if label_y and y < label_y[-1] + 0.035:
            y = label_y[-1] + 0.035
        label_y.append(y)
    ymap = {fy: ly for (_, fy, _), ly in zip(sorted(pts, key=lambda p: p[1]), label_y, strict=True)}
    for fx, fy, s in pts:
        ax.scatter(fx, fy, s=26, color=C_ON, linewidths=0)
        ax.text(fx - 0.02, ymap[fy], SRC_LABEL.get(s, s), fontsize=6.8, va="center", ha="right")
    ax.set_xlabel("source-mean emission fraction")
    ax.set_ylabel("source-mean leakage E")
    ax.set_title(
        f"16 source means: Spearman {rho_src['point']:+.3f} "
        f"[{rho_src['ci_lo']:+.3f}, {rho_src['ci_hi']:+.3f}]",
        fontsize=11,
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_833/emission_rate_vs_E", dir="figures/")
    plt.close(fig)


# ── Figure 3: selection covariate + per-cell chain predictions at L14 ─────────
def fig_percell(ne, cell_keys: list[str], E: np.ndarray) -> None:
    """Left: retained-N vs E (480 cells, floor line); middle/right: E vs the
    held-out L14 prediction per retained cell (matched-N full text, non-emission)."""
    man = _manifest()["cells"]
    ret = np.array([man[k]["retained"] for k in cell_keys])
    dropped = np.array([man[k]["below_floor"] for k in cell_keys])

    args = SimpleNamespace(
        out_dir=RES,
        nonemit_root=RES / "analysis_tensors_nonemit",
        matchedn_root=RES / "analysis_tensors_matchedN",
        eq5_root=RES / "analysis_tensors_nonemit_eq5",
        retention_manifest=RES / "emission_rate/retention_manifest.json",
        matchedn_indices=RES / "emission_rate/matchedN_sample_indices.json",
        eq5_indices=RES / "emission_rate/eq5_sample_indices.json",
        fulltext_npz_root=None,
    )
    ne.fit658.DEVICE = "cpu"
    ne.fitM.TARGET_DIM = ne.TARGET_DIM
    design = ne.load_retained_design(args, 14)
    st = design["stacks"]
    Ek = design["E"]
    keep = ~np.isnan(Ek)
    pca = ne.fitM._pca_basis_v0(st["V0"], ne.TARGET_DIM)
    r_hat = ne.fitM._r_hat_for("fact", 14, ne.fitM._load_rb_main(), ne.fitM._load_rb_fact())
    persisted = _chain_json(14)["arms"]
    chains = {}
    for arm, (X, V) in {
        "on_full_matchedN": (st["Cplus"], design["Von_matchedN"]),
        "on_nonemit": (st["Cplus"], design["Von_nonemit"]),
    }.items():
        loco = ne.fitM._ridge_loco_pred(X, V @ pca.T)
        rho, chain = ne.fitM._chain_rho_one(loco[keep], pca, r_hat, Ek[keep])
        assert abs(rho - persisted[arm]["rho_ridge"]) < 1e-2, (arm, rho)
        chains[arm] = (chain, persisted[arm]["rho_ridge"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax = axes[0]
    ax.scatter(
        ret[~dropped],
        E[~dropped],
        s=14,
        alpha=0.5,
        color=C_ON,
        linewidths=0,
        label="retained (291 cells)",
    )
    ax.scatter(
        ret[dropped],
        E[dropped],
        s=14,
        alpha=0.5,
        color=C_NEU,
        linewidths=0,
        label="dropped below floor (189)",
    )
    ax.axvline(5, color="0.5", lw=1.0, ls="--")
    ax.set_xlabel("non-emission answers retained per cell (of 30)")
    ax.set_ylabel("measured leakage E")
    ax.set_title("retention vs leakage: Spearman −0.90 (480 cells)", fontsize=10.5)
    ax.legend(fontsize=7.6, loc="upper right")

    for ax, arm, lab in [
        (axes[1], "on_full_matchedN", "matched-N full text"),
        (axes[2], "on_nonemit", "non-emission only"),
    ]:
        chain, rho = chains[arm]
        ax.scatter(chain, Ek[keep], s=14, alpha=0.5, color=C_ON, linewidths=0)
        ax.set_xlabel("held-out map prediction along the fact direction")
        ax.set_ylabel("measured leakage E")
        ax.set_title(f"{lab}, layer 14: Spearman {rho:+.2f} (291 cells)", fontsize=10.5)
    fig.tight_layout()
    savefig_paper(fig, "issue_833/nonemit_percell_L14", dir="figures/")
    plt.close(fig)


def main() -> None:
    """Build the three round-2 figures from persisted outputs only."""
    ne = _load_ne_module()
    d = np.load(RES / "joined_cache/fact_L14.npz", allow_pickle=True)
    cell_keys = [str(v) for v in d["cell_keys"].tolist()]
    E = ne.fitM._load_E("fact", cell_keys)
    assert len(cell_keys) == 480 and not np.isnan(E).any(), (len(cell_keys),)
    fig_forest()
    fig_emission_scatter(cell_keys, E)
    fig_percell(ne, cell_keys, E)
    print("figures written to figures/issue_833/")


if __name__ == "__main__":
    main()
