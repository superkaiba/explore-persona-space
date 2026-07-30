"""Fold figures for the #1689 derived-vs-free-answer-map follow-up round.

Reads ``eval_results/issue_1689/analyzer/dvf_unit_digest.csv`` (produced by
``issue1689_dvf_fold_digest.py`` from the per-unit JSONs — never the
double-counting battery ``summary.json``) plus the two ``subspace_overlap.json``
files, and renders the three clean-result figures with the project paper-plot
conventions (``set_paper_style("blog")`` + ``savefig_paper`` sidecars).

Run from the issue worktree root:
    uv run python scripts/issue1689_dvf_fold_figures.py --out-figs <main>/figures/issue_1689
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# load_dotenv() BEFORE numpy/matplotlib (shared-VM thread caps, #847).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

BASE = Path("eval_results/issue_1689")

VERDICT_ORDER = [
    "shared_readout_supported",
    "readout_changed",
    "transfer_map_insufficient",
    "free_map_uninformative",
]
VERDICT_COLORS = {
    "shared_readout_supported": "#2a9d8f",
    "readout_changed": "#7b6bb8",
    "transfer_map_insufficient": "#d1495b",
    "free_map_uninformative": "#9a9a9a",
}
VERDICT_LABELS = {
    "shared_readout_supported": "shared readout supported",
    "readout_changed": "readout changed",
    "transfer_map_insufficient": "transfer map insufficient",
    "free_map_uninformative": "free map uninformative (excluded)",
}


def load_digest() -> list[dict]:
    with open(BASE / "analyzer" / "dvf_unit_digest.csv") as f:
        return list(csv.DictReader(f))


def fig11_verdicts(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    dvf = [r for r in rows if r["battery"] == "dvf_within"]

    def _pred(model_tag: str, arm: str, inf_only: bool):
        def pred(r: dict) -> bool:
            is_instruct = "Instruct" in r["model"]
            ok = (is_instruct == (model_tag == "instruct")) and r["arm"] == arm
            return ok and (r["informative"] == "1" if inf_only else True)

        return pred

    groups = []
    for model_tag in ("base", "instruct"):
        for arm in ("prefix", "context"):
            for inf_only, tag in ((False, "all"), (True, "informative")):
                pred = _pred(model_tag, arm, inf_only)
                n = sum(1 for r in dvf if pred(r))
                groups.append((f"{model_tag} | {arm} — {tag} ({n})", pred))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 6.0))
    ys = np.arange(len(groups))[::-1]
    for y, (label, pred) in zip(ys, groups):
        sub = [r for r in dvf if pred(r)]
        left = 0.0
        for v in VERDICT_ORDER:
            n = sum(1 for r in sub if r["verdict"] == v)
            if n:
                axL.barh(y, n, left=left, color=VERDICT_COLORS[v], edgecolor="white", height=0.62)
                if n >= 4:
                    axL.text(
                        left + n / 2, y, str(n), ha="center", va="center", fontsize=9, color="white"
                    )
                left += n
    axL.set_yticks(ys)
    axL.set_yticklabels([g[0] for g in groups], fontsize=9)
    axL.set_xlabel("ordered pair-arm units")
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[v]) for v in VERDICT_ORDER]
    axL.legend(handles, [VERDICT_LABELS[v] for v in VERDICT_ORDER], fontsize=8, loc="lower right")
    axL.set_title("Verdict counts (per-unit files; full 504-unit coverage)", fontsize=10)

    rng = np.random.default_rng(42)
    rho_txt = []
    for arm, marker in (("prefix", "o"), ("context", "s")):
        sub = [
            r
            for r in dvf
            if r["arm"] == arm
            and r["verdict"] != "free_map_uninformative"
            and r["parent_rung"]
            and r["g1"]
        ]
        x = np.array([float(r["parent_rung"]) for r in sub])
        y = np.array([float(r["g1"]) for r in sub])
        inf = np.array([r["informative"] == "1" for r in sub])
        cols = [VERDICT_COLORS[r["verdict"]] for r in sub]
        xj = x + rng.uniform(-0.22, 0.22, len(x))
        yc = np.clip(y, -1.5, 0.5)
        for i in range(len(sub)):
            axR.scatter(
                xj[i],
                yc[i],
                marker=marker,
                s=26,
                facecolors=cols[i] if inf[i] else "none",
                edgecolors=cols[i],
                linewidths=1.0,
                alpha=0.75,
            )
        from scipy.stats import spearmanr

        rho_txt.append(
            f"{arm}: rho(g1, parent rung) = {spearmanr(x, y).statistic:.2f} (n={len(sub)})"
        )
    axR.axhline(0.0, color="black", lw=0.8, ls="--")
    axR.set_xlabel(
        "parent ladder rung reached (1 = direct transfer, 9 = full reparameterization / none)"
    )
    axR.set_ylabel("g1 = R2(derived) - 0.9 x R2(free), clipped to [-1.5, 0.5]")
    axR.set_title(
        "\n".join(rho_txt) + "\nfilled = informative unit, open = validity-screened", fontsize=9
    )
    fig.suptitle(
        "Derived shared-readout map vs free map: verdicts and concordance (504 of 504 units, both models)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "fig11_dvf_verdict_lattice", dir=out)
    plt.close(fig)


def fig12_structure(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    cms = [r for r in rows if r["battery"] == "cms_within"]
    inf = [r for r in cms if r["informative"] == "1"]

    def fam(w: str) -> str:
        if w == "full_affine":
            return "full affine"
        if w.startswith("trans"):
            return "translation family"
        return "rank k <= 128"

    classes = ["identity", "framing", "crossed", "provenance", "identity-vs-user", "user-framing"]
    fams = ["translation family", "rank k <= 128", "full affine"]
    fam_colors = {
        "translation family": "#8ecae6",
        "rank k <= 128": "#ffb703",
        "full affine": "#d1495b",
    }
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    xs = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for fm in fams:
        vals = np.array(
            [
                sum(1 for r in inf if r["cls"] == c and fam(r["weakest_class"]) == fm)
                for c in classes
            ],
            dtype=float,
        )
        axL.bar(xs, vals, bottom=bottom, color=fam_colors[fm], edgecolor="white", label=fm)
        for x, v, b in zip(xs, vals, bottom):
            if v:
                axL.text(x, b + v / 2, int(v), ha="center", va="center", fontsize=9)
        bottom += vals
    axL.set_xticks(xs)
    axL.set_xticklabels(classes, rotation=20, ha="right", fontsize=9)
    axL.set_ylabel("informative pair-arm units")
    axL.legend(fontsize=8)
    axL.set_title("Weakest sufficient class of the context-side map M, by pair class", fontsize=10)

    for r in cms:
        fm = fam(r["weakest_class"])
        filled = r["informative"] == "1"
        axR.scatter(
            float(r["eff_rank_m_minus_i"]),
            float(r["gain_full_over_translation_r2"]),
            marker="o",
            s=22,
            facecolors=fam_colors[fm] if filled else "none",
            edgecolors=fam_colors[fm],
            linewidths=1.0,
            alpha=0.75,
        )
    axR.axvline(3584, color="black", lw=0.8, ls=":", label="d = 3584")
    axR.set_xlabel("effective rank of M - I (of d = 3584)")
    axR.set_ylabel("held-out R2 gain, full affine over translation")
    axR.legend(fontsize=8)
    axR.set_title(
        "Per-unit view: the fitted correction is near-full-rank\n(filled = informative, open = screened)",
        fontsize=9,
    )
    fig.suptitle("Context-side transfer-map structure (504 of 504 within-model units)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "fig12_context_map_structure", dir=out)
    plt.close(fig)


def fig13_crossmodel(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    xm = [r for r in rows if r["battery"] == "xm_ladder" and r["informative"] == "1"]
    dir_colors = {"base->instruct": "#0f766e", "instruct->base": "#b45309"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for r in xm:
        c = dir_colors[r["model"]]
        marker = "o" if r["arm"] == "prefix" else "s"
        filled = r["xm_rung9_reconciles"] == "1"
        x = float(r["xm_ceiling"])
        y = float(r["xm_rec9"])
        axL.scatter(
            x,
            y,
            marker=marker,
            s=42,
            facecolors=c if filled else "none",
            edgecolors=c,
            linewidths=1.2,
        )
        lbl = (
            r["pair"]
            .replace("assistant_", "asst_")
            .replace("user_", "u_")
            .replace("naturalistic", "plain")
        )
        axL.text(x, y + 0.025, lbl, fontsize=6.5, ha="center")
    axL.axhline(0.9, color="black", lw=0.8, ls="--", label="0.90 reconciliation bar")
    axL.set_xlabel("target within-cell held-out R2 (ceiling)")
    axL.set_ylabel("rung-9 (full reparameterization) recovery fraction")
    axL.legend(fontsize=8, loc="lower right")
    axL.set_title(
        "22 informative cross-model units: all read rung 9\ncircle = prefix, square = context; filled = reconciles at rung 9",
        fontsize=9,
    )

    for name, key, color in (
        ("left (input) subspaces", "left_overlap_k32", "#219ebc"),
        ("right (output) subspaces", "right_overlap_k32", "#fb8500"),
    ):
        so = json.loads(
            (BASE / "crossmodel_pairs/crossmodel_structure/subspace_overlap.json").read_text()
        )
        vals = np.array([q[key] for q in so["unit_pairs"]])
        axR.hist(
            vals, bins=60, alpha=0.55, color=color, label=f"{name} (median {np.median(vals):.3f})"
        )
    null975 = so["random_subspace_null"]["32"]["null_p975"]
    axR.axvline(
        null975, color="black", lw=1.0, ls="--", label=f"random-subspace null p97.5 = {null975:.4f}"
    )
    axR.set_xlabel("top-32 subspace overlap between condition pairs (M - I correction)")
    axR.set_ylabel("condition-pair count")
    axR.legend(fontsize=8)
    axR.set_title(
        "Uniformity read: correction subspaces overlap above the null,\nbut weakly in absolute terms",
        fontsize=9,
    )
    fig.suptitle(
        "Base <-> instruct same-condition pairs: reconciliation and correction-subspace uniformity",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, "fig13_crossmodel_battery", dir=out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-figs", type=Path, default=Path("figures/issue_1689"))
    args = ap.parse_args()
    set_paper_style("blog")
    rows = load_digest()
    args.out_figs.mkdir(parents=True, exist_ok=True)
    fig11_verdicts(rows, args.out_figs)
    fig12_structure(rows, args.out_figs)
    fig13_crossmodel(rows, args.out_figs)
    print(f"wrote 3 figures to {args.out_figs}")
    return 0


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
