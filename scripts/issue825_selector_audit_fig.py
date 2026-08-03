"""Figure for the #825 estimator-selector audit (Phase 0).

Three panels, one colour per SELECTOR held constant across all of them:
  (a,b) matched-n x selector curve on the Track-S chat cells (instruct, base)
  (c)   the eight Track-M cells at n=2000 under all three selectors

CLI: uv run python scripts/issue825_selector_audit_fig.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps (#847) before matplotlib/numpy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402

RES = _REPO_ROOT / "eval_results/issue_825/selector_audit/results.json"
FIGDIR = _REPO_ROOT / "figures/issue_825/selector_audit"
D_MODEL = 3584
N_FOLDS = 5

# One colour per selector, held constant across every panel.
SEL_STYLE = {
    "gcv_unguarded": ("#d55e00", "o", "GCV, unguarded (committed #825 default)"),
    "gcv_guarded": ("#0072b2", "s", "GCV + dof cap 0.9 (guarded)"),
    "inner_group_cv": ("#009e73", "^", "inner-group-CV (no dof formula, no free constant)"),
}


def main() -> int:
    set_paper_style()
    res = json.loads(RES.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # Panels (a,b): matched-n curves.
    for ax, cell, title in (
        (axes[0], "S_instruct_chat", "Qwen2.5-7B-Instruct"),
        (axes[1], "S_pretrained_chat", "Qwen2.5-7B (base)"),
    ):
        curve = res["leg_b_sizing"][cell]["curve"]
        ns = [r["n"] for r in curve]
        for sel, (c, m, lab) in SEL_STYLE.items():
            ax.plot(
                ns,
                [r["mean_r2"][sel] for r in curve],
                marker=m,
                color=c,
                lw=2.0,
                ms=7,
                label=lab,
            )
        n_cross = D_MODEL * N_FOLDS / (N_FOLDS - 1)  # n_tr = D  ->  n = 4480
        ax.axvline(n_cross, color="0.35", ls="--", lw=1.4)
        ax.text(
            n_cross * 0.985,
            0.03,
            f"$n_{{train}}=D={D_MODEL}$ (n={n_cross:.0f})",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="0.35",
        )
        ax.axhline(0.0, color="0.6", lw=1.0, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("n (conversations); grouped 5-fold, $n_{train}=0.8n$")
        ax.set_ylabel("held-out $R^2$ at layer 19")
        ax.set_title(
            f"(a) Track-S assistant map vs n — {title}" if ax is axes[0] else f"(b) {title}"
        )
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Panel (c): the eight Track-M cells at n=2000.
    corr = res["leg_a_corroboration"]
    order = [
        c
        for c in (
            "M_instruct_assistant_chat",
            "M_instruct_assistant_naturalistic",
            "M_pretrained_assistant_chat",
            "M_pretrained_assistant_naturalistic",
            "M_instruct_user_chat",
            "M_instruct_user_naturalistic",
            "M_pretrained_user_chat",
            "M_pretrained_user_naturalistic",
        )
        if c in corr
    ]
    ax = axes[2]
    x = np.arange(len(order))
    w = 0.27
    for i, (sel, (c, _m, lab)) in enumerate(SEL_STYLE.items()):
        key = {
            "gcv_unguarded": "unguarded",
            "gcv_guarded": "guarded",
            "inner_group_cv": "inner_group_cv",
        }[sel]
        ax.bar(x + (i - 1) * w, [corr[cid][key] for cid in order], w, color=c, label=lab)
    ax.axhline(0.0, color="0.4", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            cid.replace("M_", "")
            .replace("pretrained", "base")
            .replace("instruct", "inst")
            .replace("assistant", "asst")
            .replace("_naturalistic", " plain")
            .replace("_chat", " chat")
            .replace("_", " ")
            for cid in order
        ],
        fontsize=7,
        rotation=40,
        ha="right",
    )
    ax.set_ylabel("held-out $R^2$ at layer 19")
    ax.set_title("(c) Track-M cells, n=2000 ($n_{train}=1600 < D$)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "Lambda-selection audit: the Track-M 'user-turn null' and the low-n collapse are "
        "estimator artifacts, not statistical power",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"selector_audit.{ext}", dpi=160, bbox_inches="tight")
    meta = {
        "script": "scripts/issue825_selector_audit_fig.py",
        "inputs": ["eval_results/issue_825/selector_audit/results.json"],
        "note": (
            "One colour per lambda selector across all three panels. Panels (a,b): "
            "mean over 3 subsample seeds (1 draw at the full-n level). Panel (c): single "
            "full-cell fit per selector. All reads layer 19, grouped 5-fold, fold seed 0."
        ),
    }
    (FIGDIR / "selector_audit.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {FIGDIR}/selector_audit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
