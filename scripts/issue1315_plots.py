#!/usr/bin/env python
"""#1315 analyzer figures — impolite activation-shift geometry.

Reads eval_results/issue_1315/geometry/{geometry_per_cell,geometry_tf_shared}.json
(+ selection JSONs mirrored under eval_results/issue_1315/selection/) and the
staged pooled.pt clouds, and writes the clean-result figures under
figures/issue_1315/ via paper_plots conventions (blog style, savefig_paper).

Also writes eval_results/issue_1315/geometry/seam_robustness.json — the
prefix-arm read with/without the BPE-seam-handled panel context
(neg_reph_curious; 20/120 rows per pass, prefix arm only).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

REPO_ROOT = _SCRIPTS_DIR.parent
GEO_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "geometry"
SEL_DIR = REPO_ROOT / "eval_results" / "issue_1315" / "selection"
FIG_DIR = REPO_ROOT / "figures" / "issue_1315"
STAGE = REPO_ROOT / "data" / "issue_1315" / "hf_dl" / "analysis_tensors"

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue_653.spectral import spectral_dvs  # noqa: E402
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

CELLS = [
    ("imp_pers_lora", "LoRA, persona context"),
    ("imp_conv_lora", "LoRA, WildChat context"),
    ("imp_icl_lora_neg", "LoRA + negatives, ICL"),
    ("imp_icl_lora_pos", "LoRA positives-only, ICL"),
    ("imp_icl_ft_neg", "Full FT + negatives, ICL"),
    ("imp_icl_ft_pos", "Full FT positives-only, ICL"),
]
LABEL = dict(CELLS)
LAYERS = list(range(28))
L14 = 14
SEAM_CONTEXT = "neg_reph_curious"

# #1112 committed layer-14 response-arm reference values (clean-result body,
# task 1112; n=120 clouds) + #653 (n=80).
REF_1112_OWN = (66, 74)  # own-text rank-k@90 span incl. generics
REF_1112_SHARED = (27, 35)  # shared-text collapse span
REF_1112_CONTEXT = (4, 13)  # same-token context arm span
REF_653_80ROW = (39, 45)  # #653 sycophancy, ~80-row clouds


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def rec(records: dict, cell: str, dose: str, arm: str, layer: int) -> dict | None:
    return records.get(f"{cell}/{dose}/{arm}/L{layer}")


def curve(records: dict, cell: str, dose: str, arm: str, dv: str) -> list[float]:
    return [rec(records, cell, dose, arm, li)[dv] for li in LAYERS]


def fig_hero_rankk(own: dict, tf: dict) -> None:
    """Hero: per-layer rank-k@90, own-text vs shared-text response arm."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(CELLS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, payload, title in (
        (axes[0], own, "Own-text response arm"),
        (axes[1], tf, "Shared-text response arm (teacher-forced control)"),
    ):
        for (cell, label), c in zip(CELLS, colors, strict=True):
            r = payload["records"]
            if rec(r, cell, "selected", "response", L14) is None:
                continue
            ys = curve(r, cell, "selected", "response", "rank_k_at_90")
            ax.plot(LAYERS, ys, color=c, lw=2, label=label)
            ax.text(LAYERS[-1] + 0.3, ys[-1], label.split(",")[0], color=c, fontsize=8, va="center")
        ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Decoder layer")
    axes[0].axhspan(*REF_1112_OWN, color="tab:gray", alpha=0.15)
    axes[0].text(
        0.5, REF_1112_OWN[1] + 1, "sycophancy own-text span (task 1112)", fontsize=8, color="gray"
    )
    axes[0].axhspan(*REF_653_80ROW, color="tab:orange", alpha=0.10)
    axes[0].text(
        0.5,
        REF_653_80ROW[0] - 5,
        "sycophancy span at 80 rows (task 653)",
        fontsize=8,
        color="tab:orange",
    )
    axes[1].axhspan(*REF_1112_SHARED, color="tab:gray", alpha=0.15)
    axes[1].text(
        0.5,
        REF_1112_SHARED[1] + 1,
        "sycophancy shared-text span (task 1112)",
        fontsize=8,
        color="gray",
    )
    axes[0].set_ylabel("Rank-k@90 of the shift cloud (120 rows)")
    axes[0].legend(loc="lower center", fontsize=8, ncol=2)
    savefig_paper(fig, "hero_impolite_rankk_own_vs_shared", dir=FIG_DIR)
    plt.close(fig)


def fig_paired_diffs(own: dict) -> None:
    """Registered paired contrasts D_rank / D_mag with 95% CIs + #1112 refs."""
    set_paper_style("blog")
    diffs = own["cross_cell_diffs"]
    panels = [
        ("diff_rank_k_at_90", "Rank-k@90 difference (modes)", 0),
        ("diff_mu_norm", "Mean-shift-norm difference", 1),
    ]
    pair_rows = [
        ("H4H5_method_ftneg_vs_loraneg", "Full FT - LoRA\n(+negatives pair)"),
        ("H6_negatives_loraneg_vs_lorapos", "Negatives - positives-only\n(LoRA pair)"),
        (
            "H6mirror_negatives_ftneg_vs_ftpos",
            "Negatives - positives-only\n(full-FT pair, descriptive)",
        ),
    ]
    ref_1112 = {
        ("H4H5_method_ftneg_vs_loraneg", "diff_rank_k_at_90"): -3.0,
        ("H4H5_method_ftneg_vs_loraneg", "diff_mu_norm"): 3.24,
        ("H6_negatives_loraneg_vs_lorapos", "diff_rank_k_at_90"): 0.0,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for dv, xlabel, pi in panels:
        ax = axes[pi]
        for yi, (name, _label) in enumerate(pair_rows):
            d = diffs.get(name, {})
            read = d.get("reads", {}).get(f"response/L{L14}", {}).get(dv)
            if read is None:
                continue
            ax.errorbar(
                read["point"],
                yi,
                xerr=[[read["point"] - read["ci_low"]], [read["ci_high"] - read["point"]]],
                fmt="o",
                color="tab:blue",
                capsize=4,
                lw=2,
            )
            ax.text(read["point"], yi + 0.14, f"{read['point']:+.2f}", ha="center", fontsize=9)
            r = ref_1112.get((name, dv))
            if r is not None:
                ax.plot(r, yi - 0.18, marker="x", color="tab:gray", ms=8)
                ax.text(r, yi - 0.40, f"sycophancy {r:+.2f}", ha="center", fontsize=8, color="gray")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(pair_rows)), [label for _, label in pair_rows])
        ax.set_xlabel(xlabel + " — layer 14, response arm")
        ax.set_ylim(-0.7, len(pair_rows) - 0.3)
    axes[1].set_yticklabels([])
    fig.suptitle(
        "Paired cross-cell contrasts at matched install (95% paired-bootstrap CIs)", y=1.02
    )
    savefig_paper(fig, "impolite_2x2_paired_diffs", dir=FIG_DIR)
    plt.close(fig)


def fig_alignment(own: dict) -> None:
    """|cos(mu, r_B)| and |cos(top_dir, r_B)| per layer + random baseline."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(CELLS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    r = own["records"]
    for ax, dv, title in (
        (axes[0], "cos_mu_to_rb", "Mean shift vs read-out direction"),
        (axes[1], "cos_top_to_rb", "Top shift mode vs read-out direction"),
    ):
        for (cell, label), c in zip(CELLS, colors, strict=True):
            if rec(r, cell, "selected", "response", L14) is None:
                continue
            ys = [abs(rec(r, cell, "selected", "response", li)[dv]) for li in LAYERS]
            ax.plot(LAYERS, ys, color=c, lw=1.8, label=label)
        # norm-matched random-cosine 97.5% bound per layer (same for all cells at a layer dim)
        hi = [
            max(
                abs(rec(r, cell, "selected", "response", li)["random_cos_ci"]["ci_high"])
                for cell, _ in CELLS
                if rec(r, cell, "selected", "response", li) is not None
            )
            for li in LAYERS
        ]
        ax.plot(LAYERS, hi, color="gray", ls="--", lw=1.2, label="chance 97.5% bound")
        ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Decoder layer")
    axes[0].set_ylabel("|cosine| to impolite read-out direction")
    axes[0].legend(fontsize=8, ncol=2)
    savefig_paper(fig, "alignment_cos_rb", dir=FIG_DIR)
    plt.close(fig)


def _cloud(
    cell: str, dose: str, arm: str, layer: int, tree: str = "capture"
) -> tuple[np.ndarray, list]:
    """Recompute one delta cloud from the staged stores (for low-level plots)."""
    store = geo.load_store(STAGE / tree / cell / dose / "pooled.pt")
    base = (
        geo.load_store(STAGE / "base" / "base" / "pooled.pt")
        if (STAGE / "base").exists()
        else geo.load_store(STAGE / "capture" / "base" / "base" / "pooled.pt")
    )
    keys_t = [(m["context_id"], int(m["question_idx"])) for m in store["row_meta"]]
    keys_b = [(m["context_id"], int(m["question_idx"])) for m in base["row_meta"]]
    pos = {k: i for i, k in enumerate(keys_b)}
    perm = [pos[k] for k in keys_t]
    Xt = store["arms"][arm][layer].to(torch.float32).numpy()
    Xb = base["arms"][arm][layer].to(torch.float32).numpy()[perm]
    return Xt - Xb, keys_t


def fig_cumshare_and_clouds() -> None:
    """Low-level: cumulative eigenvalue share at L14 + per-cell PC scatters."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(CELLS))
    # cumulative share
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for (cell, label), c in zip(CELLS, colors, strict=True):
        cloud, _ = _cloud(cell, "selected", "response", L14)
        X = cloud - cloud.mean(0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        lam = s**2 / (s**2).sum()
        ax.plot(np.arange(1, len(lam) + 1), np.cumsum(lam), color=c, lw=2, label=label)
    ax.axhline(0.9, color="black", lw=0.8, ls="--")
    ax.text(2, 0.905, "90% of variance", fontsize=9)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative eigenvalue share (layer 14, response arm)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "spectrum_cumshare_layer14", dir=FIG_DIR)
    plt.close(fig)

    # per-cell top-2 PC scatter, colored by context
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for (cell, label), ax in zip(CELLS, axes.flat, strict=True):
        cloud, keys = _cloud(cell, "selected", "response", L14)
        X = cloud - cloud.mean(0, keepdims=True)
        _u, _s, Vt = np.linalg.svd(X, full_matrices=False)
        pc = X @ Vt[:2].T
        ctxs = sorted({c for c, _ in keys})
        cmap = dict(zip(ctxs, paper_palette_blog(len(ctxs)), strict=True))
        for ctx in ctxs:
            m = np.array([k[0] == ctx for k in keys])
            ax.scatter(pc[m, 0], pc[m, 1], s=14, color=cmap[ctx], label=ctx, alpha=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("shift PC 1")
        ax.set_ylabel("shift PC 2")
        if ax is axes.flat[0]:
            ax.legend(fontsize=6.5, loc="best")
    savefig_paper(fig, "cloud_pc_scatter_grid", dir=FIG_DIR)
    plt.close(fig)


def fig_install(own: dict) -> None:
    """FT install ladders + rank-vs-install labeled points."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for cell, c in (("imp_icl_ft_neg", "tab:blue"), ("imp_icl_ft_pos", "tab:red")):
        sel = _load(SEL_DIR / cell / "selection.json")
        steps = sorted(int(k) for k in sel["rates_by_step"])
        ax.plot(
            steps,
            [sel["rates_by_step"][str(s)] for s in steps],
            "-o",
            ms=4,
            color=c,
            label=f"{LABEL[cell]} (selected step {sel['step']}, rate {sel['rate']:.2f})",
        )
        ax.plot(sel["step"], sel["rate"], marker="*", ms=16, color=c)
    ax.axhspan(0.60, 0.85, color="tab:green", alpha=0.12)
    ax.text(1, 0.86, "target install band 0.60-0.85", fontsize=9, color="tab:green")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Judged impolite rate (Tier-1 ladder)")
    ax.legend(fontsize=8, loc="lower right")
    savefig_paper(fig, "install_ladders_ft", dir=FIG_DIR)
    plt.close(fig)

    # DV vs realized install (parity/tier2 reads), labeled points
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    r = own["records"]
    install = {}
    for cell, _ in CELLS:
        pj = SEL_DIR / cell / "parity.json"
        tj = SEL_DIR / cell / "tier2.json"
        if pj.exists():
            install[cell] = _load(pj)["rate"]
        elif tj.exists():
            install[cell] = _load(tj)["rates"]["trained"]
    colors = paper_palette_blog(len(CELLS))
    for (cell, label), c in zip(CELLS, colors, strict=True):
        rk = rec(r, cell, "selected", "response", L14)["rank_k_at_90"]
        ax.scatter(install[cell], rk, s=60, color=c)
        ax.text(install[cell], rk + 1.2, label, fontsize=8, ha="center", color=c)
    ax.axvspan(0.60, 0.85, color="tab:green", alpha=0.10)
    ax.set_xlabel("Realized judged impolite rate at the captured checkpoint")
    ax.set_ylabel("Rank-k@90 (layer 14, response arm, own text)")
    savefig_paper(fig, "dv_vs_install", dir=FIG_DIR)
    plt.close(fig)


def fig_arms(own: dict) -> None:
    """Mapping-arm profiles: prefix + context arm rank-k@90 per layer."""
    set_paper_style("blog")
    colors = paper_palette_blog(len(CELLS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    r = own["records"]
    for ax, arm, title in (
        (axes[0], "prefix", "Prefix arm (same tokens, before the user query)"),
        (axes[1], "context", "Context arm (same tokens, incl. the user query)"),
    ):
        for (cell, label), c in zip(CELLS, colors, strict=True):
            ys = curve(r, cell, "selected", arm, "rank_k_at_90")
            ax.plot(LAYERS, ys, color=c, lw=1.8, label=label)
        ax.axvline(L14, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Decoder layer")
    axes[1].axhspan(*REF_1112_CONTEXT, color="tab:gray", alpha=0.15)
    axes[1].text(
        0.5,
        REF_1112_CONTEXT[1] + 1,
        "sycophancy context-arm span (task 1112)",
        fontsize=8,
        color="gray",
    )
    axes[0].set_ylabel("Rank-k@90 of the shift cloud")
    axes[0].legend(fontsize=8, ncol=2)
    savefig_paper(fig, "arms_rank_profiles", dir=FIG_DIR)
    plt.close(fig)


def _dvs_of(cloud: np.ndarray) -> dict:
    """Row-centered spectral DVs of a cloud (the rig's convention)."""
    X = cloud - cloud.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    return spectral_dvs(s)


def seam_robustness() -> dict:
    """Prefix-arm L14 reads with vs without the seam-handled context
    (neg_reph_curious), plus a matched-n drop-another-context reference."""
    out: dict[str, dict] = {}
    for cell, _ in CELLS:
        cloud, keys = _cloud(cell, "selected", "prefix", L14)
        full = _dvs_of(cloud)
        m_seam = np.array([k[0] != SEAM_CONTEXT for k in keys])
        no_seam = _dvs_of(cloud[m_seam])
        others = sorted({c for c, _ in keys if c != SEAM_CONTEXT})
        ref_ctx = others[0]
        m_ref = np.array([k[0] != ref_ctx for k in keys])
        no_ref = _dvs_of(cloud[m_ref])
        out[cell] = {
            "full_120": full,
            f"drop_{SEAM_CONTEXT}_100": no_seam,
            f"drop_{ref_ctx}_100_matched_n_reference": no_ref,
        }
    path = GEO_DIR / "seam_robustness.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> int:
    own = _load(GEO_DIR / "geometry_per_cell.json")
    tf = _load(GEO_DIR / "geometry_tf_shared.json")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_hero_rankk(own, tf)
    fig_paired_diffs(own)
    fig_alignment(own)
    fig_cumshare_and_clouds()
    fig_install(own)
    fig_arms(own)
    sr = seam_robustness()
    print(
        json.dumps(
            {c: {k: v["rank_k_at_90"] for k, v in d.items()} for c, d in sr.items()}, indent=1
        )
    )
    print("figures ->", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
