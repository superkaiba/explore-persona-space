# ruff: noqa: RUF001, RUF002
"""Figures for #810 round 5 (`boundary-truncation-dose-response`) — plan v18 §6.

All axes/labels reader-facing (ROW_LABELS + context_label reused from
``issue810_he_promotion_figures``); rows + k grid are DERIVED from the driver
JSONs so the same entrypoint renders the smoke subset and the production
27-cell family (smoke/production parity).

  1. btdr_dose_trace         — HERO: committed-layer LOCO skill per row vs the
     retained answer fraction {0, 25, 50, 75, 100}%; committed endpoints as
     distinct square markers (k=0 round 4, k=100 rounds 1/3), bootstrap CI
     whiskers on the interior points, the two positive singles highlighted,
     other rows greyed.
  2. btdr_dose_trace_delta   — Δ(full − truncated) vs k with the ±0.02
     equivalence margin shaded; k=0 = the ECHOED round-4 committed deltas
     (distinct markers), k=100 ≡ 0 by construction (annotated).
  3. btdr_percontext_spaghetti — the 50 per-context Δ traces behind the paired
     estimates for the two positive singles (quantile contexts labeled).
  4. btdr_paired_draws       — per-(single × k) violins of the shared-index
     bootstrap draws (the low-level data behind the CIs).
  5. btdr_mechanism_by_k     — median centered cosine (full vs truncated) at
     each row's committed layer vs k, with the round-4 k=0 point.
  6. btdr_ownbest_trace      — own-best-layer LOCO skill per row vs k alongside
     the committed-layer trace (the binding analyzer concern's layer-shift
     companion; descriptive, selection-favored, never banded).

Inputs (committed on `issue-810` + this round's driver outputs):
  eval_results/issue_810/boundary-truncation-dose-response/paired_dose_response.json
  eval_results/issue_810/boundary-truncation-dose-response/mechanism_cosine_btdr.json
  eval_results/issue_810/boundary-truncation-dose-response/reconstruction_skill_btdr_k{pct}.json
  eval_results/issue_810/header-echo-ablation-capture/{reconstruction_skill_header_echo,mechanism_cosine_r2}.json
  eval_results/issue_810/reconstruction_skill_by_summary.json               (k=100: im_end/turn_nl)
  eval_results/issue_810/user-header-newline-summary/reconstruction_skill_user_header.json
"""

from __future__ import annotations

import argparse
import json

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from issue810_he_promotion_figures import ROW_LABELS, context_label  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

BTDR_DIR = REPO / "eval_results/issue_810/boundary-truncation-dose-response"
HE_DIR = REPO / "eval_results/issue_810/header-echo-ablation-capture"
FIG_DIR = REPO / "figures/issue_810/boundary-truncation-dose-response"
SINGLES = ("turn_nl", "uh_im_start")  # the two round-4 positive singles (highlighted)
MARGIN = 0.02
# k=100 committed-recon source per row (the round-4 two-capture convention).
R1_ROWS = ("im_end", "turn_nl")


def _pct(k: float) -> int:
    return round(k * 100)


def load(dose_dir: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    paired = json.loads((dose_dir / "paired_dose_response.json").read_text())
    mech = json.loads((dose_dir / "mechanism_cosine_btdr.json").read_text())
    mech_k0 = json.loads((HE_DIR / "mechanism_cosine_r2.json").read_text())
    he = json.loads((HE_DIR / "reconstruction_skill_header_echo.json").read_text())
    r1 = json.loads(
        (REPO / "eval_results/issue_810/reconstruction_skill_by_summary.json").read_text()
    )
    r3_path = (
        REPO / "eval_results/issue_810/user-header-newline-summary"
    ) / "reconstruction_skill_user_header.json"
    r3 = json.loads(r3_path.read_text())
    return paired, mech, mech_k0, he, r1, r3


def _row_color(rows: list[str]) -> dict[str, str]:
    pal = paper_palette(3)
    return {r: (pal[1] if r == "turn_nl" else pal[2]) if r in SINGLES else "0.65" for r in rows}


def fig_dose_trace(paired: dict, fig_dir: Path) -> None:
    rows = paired["rows"]
    ks = sorted(paired["truncate_fracs"])
    colors = _row_color(rows)
    abs_sk = paired["abs_skill_by_side"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for r in rows:
        xs = [0.0] + [100.0 * k for k in ks] + [100.0]
        ys = (
            [abs_sk["k0"][r]["observed"]]
            + [abs_sk[f"k{_pct(k)}"][r]["observed"] for k in ks]
            + [abs_sk["full"][r]["observed"]]
        )
        z = 3 if r in SINGLES else 1
        lw = 2.0 if r in SINGLES else 1.1
        ax.plot(xs, ys, color=colors[r], lw=lw, zorder=z, label=ROW_LABELS.get(r, r))
        # interior points: bootstrap CI whiskers on the absolute skill
        for k in ks:
            cell = abs_sk[f"k{_pct(k)}"][r]
            o = cell["observed"]
            lo, hi = cell["ci95"]
            ax.errorbar(
                100.0 * k,
                o,
                yerr=[[max(0.0, o - lo)], [max(0.0, hi - o)]],
                fmt="o",
                color=colors[r],
                ecolor="0.6",
                capsize=2.5,
                ms=4.5,
                zorder=z,
            )
        # committed endpoints: distinct square markers (k=0 round 4, k=100 rounds 1/3)
        ax.scatter([0.0, 100.0], [ys[0], ys[-1]], marker="s", s=34, color=colors[r], zorder=z + 1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("retained answer fraction (%) — 0/100 = committed rounds 4 / 1+3 (squares)")
    ax.set_ylabel("held-out skill-over-mean R² (committed best layer)")
    ax.set_title(
        "Dose trace: predicting each boundary summary as the answer is truncated",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    savefig_paper(fig, "btdr_dose_trace", dir=fig_dir)
    plt.close(fig)


def fig_dose_trace_delta(paired: dict, fig_dir: Path) -> None:
    rows = paired["rows"]
    ks = sorted(paired["truncate_fracs"])
    colors = _row_color(rows)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axhspan(-MARGIN, MARGIN, color="#e8eef2", zorder=0)
    ax.axhline(0.0, color="0.3", lw=1.2)
    for r in rows:
        k0 = paired["k0_committed"][r]
        xs = [0.0] + [100.0 * k for k in ks] + [100.0]
        ys = (
            [k0["observed"]]
            + [paired["by_k"][f"k{_pct(k)}"]["per_row"][r]["observed"] for k in ks]
            + [0.0]  # Δ(k=100) ≡ 0 by construction (the full side IS the k=100 arm)
        )
        z = 3 if r in SINGLES else 1
        ax.plot(
            xs,
            ys,
            color=colors[r],
            lw=2.0 if r in SINGLES else 1.1,
            zorder=z,
            label=ROW_LABELS.get(r, r),
        )
        for x, cell in [(0.0, k0)] + [
            (100.0 * k, paired["by_k"][f"k{_pct(k)}"]["per_row"][r]) for k in ks
        ]:
            o = cell["observed"]
            lo, hi = cell["ci95"]
            marker = "s" if x == 0.0 else "o"  # committed k=0 echo = square
            ax.errorbar(
                x,
                o,
                yerr=[[max(0.0, o - lo)], [max(0.0, hi - o)]],
                fmt=marker,
                color=colors[r],
                ecolor="0.6",
                capsize=2.5,
                ms=5 if marker == "s" else 4.5,
                zorder=z,
            )
    ax.text(100.0, 0.004, "≡ 0 by construction", fontsize=7, ha="right", color="0.35")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("retained answer fraction (%)")
    ax.set_ylabel("Δ held-out skill (full − truncated), committed best layer")
    ax.set_title(
        "Paired skill deficit vs retained answer dose (±0.02 equivalence margin shaded)",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    savefig_paper(fig, "btdr_dose_trace_delta", dir=fig_dir)
    plt.close(fig)


def fig_percontext_spaghetti(paired: dict, fig_dir: Path) -> None:
    rows = [r for r in SINGLES if r in paired["rows"]]
    ks = sorted(paired["truncate_fracs"])
    sides = ["k0"] + [f"k{_pct(k)}" for k in ks]
    xs = [0.0] + [100.0 * k for k in ks]
    pcd = paired["per_context_delta_by_side"]
    fig, axes = plt.subplots(1, max(len(rows), 1), figsize=(6.2 * max(len(rows), 1), 5.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows, strict=False):
        ctxs = list(pcd["k0"][r].keys())
        traces = np.array(
            [[pcd[s][r][c] if pcd[s][r][c] is not None else np.nan for s in sides] for c in ctxs]
        )
        for ti in range(traces.shape[0]):
            ax.plot(xs, traces[ti], color="0.55", lw=0.7, alpha=0.5)
        med = np.nanmedian(traces, axis=0)
        ax.plot(xs, med, color="#2d2d2d", lw=2.4, label="median context")
        # label quantile contexts (max / median / min per-context Δ at the first interior k)
        at_k1 = traces[:, 1]
        order = np.argsort(at_k1)
        finite = [i for i in order if np.isfinite(at_k1[i])]
        if finite:
            for qi, qname in (
                (finite[-1], "max"),
                (finite[len(finite) // 2], "median"),
                (finite[0], "min"),
            ):
                ax.text(
                    xs[1],
                    at_k1[qi],
                    f"{qname}: {context_label(ctxs[qi])}",
                    fontsize=6,
                    color="0.3",
                    va="center",
                )
        ax.axhspan(-MARGIN, MARGIN, color="#e8eef2", zorder=0)
        ax.axhline(0.0, color="0.3", lw=1.0)
        ax.set_xticks([0, 25, 50, 75])
        ax.set_xlabel("retained answer fraction (%)")
        ax.set_title(ROW_LABELS.get(r, r))
    axes[0].set_ylabel("per-context Δ skill contribution (full − truncated)")
    axes[-1].legend(loc="upper right", fontsize=7.5)
    fig.suptitle(
        "The 50 per-context traces behind the paired dose estimates", fontsize=13, fontweight="bold"
    )
    savefig_paper(fig, "btdr_percontext_spaghetti", dir=fig_dir)
    plt.close(fig)


def fig_paired_draws(paired: dict, fig_dir: Path) -> None:
    rows = [r for r in SINGLES if r in paired["rows"]]
    ks = sorted(paired["truncate_fracs"])
    cells = [(r, k) for r in rows for k in ks]
    data = [np.asarray(paired["by_k"][f"k{_pct(k)}"]["draws_by_row"][r]) for r, k in cells]
    obs = [paired["by_k"][f"k{_pct(k)}"]["per_row"][r]["observed"] for r, k in cells]
    labels = [f"{ROW_LABELS.get(r, r)}\nkeep {_pct(k)}%" for r, k in cells]
    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(data, positions=np.arange(len(cells)), showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(paper_palette(3)[0])
        body.set_alpha(0.55)
    ax.scatter(np.arange(len(cells)), obs, color="#2d2d2d", s=28, zorder=3, label="observed Δ")
    ax.axhspan(-MARGIN, MARGIN, color="#e8eef2", zorder=0)
    ax.axhline(0.0, color="0.3", lw=1.2)
    ax.set_xticks(range(len(cells)), labels=labels, fontsize=8)
    ax.set_ylabel("Δ held-out skill (full − truncated)")
    ax.set_title(
        "The shared-index bootstrap draws behind each primary-single dose cell",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8)
    savefig_paper(fig, "btdr_paired_draws", dir=fig_dir)
    plt.close(fig)


def fig_mechanism_by_k(paired: dict, mech: dict, mech_k0: dict, fig_dir: Path) -> None:
    rows = paired["rows"]
    ks = sorted(paired["truncate_fracs"])
    colors = _row_color(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for r in rows:
        layer = str(paired["committed_layers"][r])
        y0 = mech_k0["by_row"][r]["per_layer"][layer]["median_centered_cos"]
        ys = [y0] + [
            mech["by_k"][f"k{_pct(k)}"][r]["per_layer"][layer]["median_centered_cos"] for k in ks
        ]
        xs = [0.0] + [100.0 * k for k in ks]
        z = 3 if r in SINGLES else 1
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=4,
            color=colors[r],
            lw=2.0 if r in SINGLES else 1.1,
            zorder=z,
            label=ROW_LABELS.get(r, r),
        )
    ax.axhline(0.8, color="#a05050", ls="--", lw=1.2, label="echo-consistency anchor (0.8)")
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("retained answer fraction (%) — k=0 point from round 4")
    ax.set_ylabel("median centered cosine (full vs truncated), committed layer")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("How far each boundary state moves at each answer dose", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    savefig_paper(fig, "btdr_mechanism_by_k", dir=fig_dir)
    plt.close(fig)


def _own_best(recon: dict, row: str) -> float:
    cells = [c for c in recon["by_summary"][row] if c.get("ridge_skill") is not None]
    return max(float(c["ridge_skill"]) for c in cells)


def fig_ownbest_trace(paired: dict, he: dict, r1: dict, r3: dict, dose_dir: Path, fig_dir: Path):
    rows = paired["rows"]
    ks = sorted(paired["truncate_fracs"])
    colors = _row_color(rows)
    recon_by_k = {
        k: json.loads((dose_dir / f"reconstruction_skill_btdr_k{_pct(k)}.json").read_text())
        for k in ks
    }
    abs_sk = paired["abs_skill_by_side"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for r in rows:
        full_src = r1 if r in R1_ROWS else r3
        xs = [0.0] + [100.0 * k for k in ks] + [100.0]
        own = (
            [_own_best(he, r)]
            + [_own_best(recon_by_k[k], r) for k in ks]
            + [_own_best(full_src, r)]
        )
        committed = (
            [abs_sk["k0"][r]["observed"]]
            + [abs_sk[f"k{_pct(k)}"][r]["observed"] for k in ks]
            + [abs_sk["full"][r]["observed"]]
        )
        z = 3 if r in SINGLES else 1
        ax.plot(
            xs,
            own,
            marker="o",
            ms=3.5,
            color=colors[r],
            lw=2.0 if r in SINGLES else 1.0,
            zorder=z,
            label=ROW_LABELS.get(r, r),
        )
        ax.plot(xs, committed, ls="--", color=colors[r], lw=0.9, alpha=0.7, zorder=z)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("retained answer fraction (%)")
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_title(
        "Own-best-layer skill per dose (solid; selection-favored, descriptive) vs the "
        "committed-layer trace (dashed)",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    savefig_paper(fig, "btdr_ownbest_trace", dir=fig_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #810 btdr dose-response figures")
    ap.add_argument("--dose-dir", default=str(BTDR_DIR), help="driver-output dir (JSON inputs)")
    ap.add_argument("--fig-dir", default=str(FIG_DIR), help="figure output dir")
    args = ap.parse_args()
    dose_dir, fig_dir = Path(args.dose_dir), Path(args.fig_dir)
    set_paper_style("blog")
    paired, mech, mech_k0, he, r1, r3 = load(dose_dir)
    fig_dose_trace(paired, fig_dir)
    fig_dose_trace_delta(paired, fig_dir)
    fig_percontext_spaghetti(paired, fig_dir)
    fig_paired_draws(paired, fig_dir)
    fig_mechanism_by_k(paired, mech, mech_k0, fig_dir)
    fig_ownbest_trace(paired, he, r1, r3, dose_dir, fig_dir)
    print(f"wrote 6 figures to {fig_dir}")


if __name__ == "__main__":
    main()
