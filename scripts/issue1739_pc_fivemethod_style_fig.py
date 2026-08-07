#!/usr/bin/env python3
"""#1739 r2v2: PV readouts by evaluation regime, in the result2_fivemethod style.

Restyles the P-A / P-B setting figures to match the committed
``figures/issue_1739/result2_fivemethod/result2_fivemethod.png`` layout
(d78aff9e5c):

  * one row of four panels -- evil / sycophancy / hallucination / average
  * four evaluation-regime groups per panel, each x label naming the CORPUS
  * one colour per method, no hatch used for method identity
  * HATCHED + muted bars where the DV spread gate fails (kept for
    completeness, flagged as not interpretable) -- not a shaded band
  * the average panel EXCLUDES spread-failed cells
  * a caption block carrying protocol provenance + per-group n_eval

Methods: the persona-vector projection arms the user scoped
(2026-08-07) -- PV on context, PV on the linear-mapped answer, PV on the
real answer (oracle upper bound). ``--with-ridge`` adds the two r2v2 ridge
arms for the fuller five-method comparison.

Usage
-----
    uv run python scripts/issue1739_pc_fivemethod_style_fig.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Bind the shared-VM BLAS/intra-op thread caps (#847) BEFORE heavy imports.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BEHAVIORS = ("evil", "sycophancy", "hallucination")

OOD_RUNGS = {
    "evil": ("hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"),
    "sycophancy": ("aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"),
    "hallucination": ("nqopen", "simpleqa"),
}

# per-behaviour corpus names for the two behaviour-specific regimes
CORPUS = {
    "evil": {
        "indist": "jailbreak x forbidden Qs",
        "ood": "hh-rlhf, ToxicChat, MHJ,\nPAIR, TomGibbs",
    },
    "sycophancy": {
        "indist": "Reddit personal-advice posts",
        "ood": "AITA, SycophancyEval x4,\nMWE",
    },
    "hallucination": {
        "indist": "TriviaQA (rc.nocontext)",
        "ood": "NQ-Open + SimpleQA",
    },
}
SHARED = {"pvsynth": "persona-vector grid", "generic": "random WildChat"}
REGIMES = (
    ("pvsynth", "synthetic"),
    ("generic", "generic chat"),
    ("indist", "in-distribution"),
    ("ood", "completely OOD"),
)

PV_METHODS = (
    ("arm1_ctx_e1", "Persona vector on context", "#1f77b4"),
    ("arm6_map_proj_e1", "Persona vector on mapped answer", "#8c564b"),
    ("arm11_oracle_proj", "Persona vector on real answer (oracle)", "#1a6b54"),
)
RIDGE_METHODS = (
    ("arm4_ridge_ctx", "Ridge regression on context", "#4C9BD4"),
    ("arm7_map_ridge_pred", "Ridge regression on mapped answer", "#e8a33d"),
)


def _gj(commit: str, path: str) -> dict:
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _sem(v: list[float]) -> float:
    v = [x for x in v if x is not None and not math.isnan(x)]
    return float(st.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def collect(fits: dict, spread: dict, protocol: str, methods) -> tuple[dict, dict]:
    """(behavior, regime, arm) -> (rho, err, n_eval, n_ok, n_tot, all_failed)."""
    vals: dict = {}
    for beh in BEHAVIORS:
        rows = fits[beh]["transfer_rows"]
        ood = OOD_RUNGS[beh]
        ok = lambda r: spread.get(f"{beh}|{r}", {}).get("spread_ok", True)  # noqa: E731
        for arm, *_ in methods:
            ar = [r for r in rows if r["arm"] == arm]
            if protocol == "P-A":
                base = [r for r in ar if r["fit"] == "P-A"]
                oof = [r for r in ar if r["fit"] == "P-A-train-oof"]
                pick = {
                    "pvsynth": [(r["eval_rung"], r) for r in base if r["eval_rung"] == "pvsynth"],
                    "generic": [
                        (r["eval_rung"], r) for r in base if r["eval_rung"] == "wildchat_rung"
                    ],
                    "indist": [("train", r) for r in oof if r["eval_rung"] == "train"],
                    "ood": [(r["eval_rung"], r) for r in base if r["eval_rung"] in ood],
                }
            else:
                pb = [r for r in ar if r["protocol"] == "P-B"]
                pick = {
                    "pvsynth": [(r["eval_rung"], r) for r in pb if r["eval_rung"] == "pvsynth"],
                    "generic": [
                        (r["eval_rung"], r) for r in pb if r["eval_rung"] == "wildchat_rung"
                    ],
                    "indist": [("train", r) for r in pb if r["eval_rung"] == "heldin:train"],
                    "ood": [
                        (r["eval_rung"], r)
                        for r in pb
                        if r["fit"].replace("P-B-holdout-", "") == r["eval_rung"]
                    ],
                }
            for key, _lab in REGIMES:
                items = [(rn, r) for rn, r in pick[key] if r.get("rho_frozen") is not None]
                if not items:
                    continue
                keep = [(rn, r) for rn, r in items if ok(rn)]
                all_failed = not keep
                use = items if all_failed else keep
                rho = float(st.mean([r["rho_frozen"] for _rn, r in use]))
                if len(use) == 1 and use[0][1].get("ci_frozen"):
                    lo, hi = use[0][1]["ci_frozen"]
                    err = float(max(0.0, max(hi - rho, rho - lo)))
                else:
                    err = _sem([r["rho_frozen"] for _rn, r in use])
                n_eval = sum(int(r.get("n_eval") or 0) for _rn, r in use)
                vals[(beh, key, arm)] = (rho, err, n_eval, len(keep), len(items), all_failed)

    for key, _ in REGIMES:
        for arm, *_ in methods:
            per = [
                vals[(b, key, arm)]
                for b in BEHAVIORS
                if (b, key, arm) in vals and not vals[(b, key, arm)][5]
            ]
            if per:
                vals[("average", key, arm)] = (
                    float(st.mean([p[0] for p in per])),
                    _sem([p[0] for p in per]),
                    sum(p[2] for p in per),
                    len(per),
                    len(BEHAVIORS),
                    False,
                )
    return vals, {}


def draw(vals: dict, protocol: str, methods, out_png: Path, caption: str) -> None:
    panels = [*BEHAVIORS, "average"]
    fig, axes = plt.subplots(1, 4, figsize=(23.0, 9.6))
    n_m = len(methods)
    width = 0.72 / n_m

    for ax, panel in zip(axes, panels, strict=True):
        centers = np.arange(len(REGIMES))
        for j, (arm, _label, color) in enumerate(methods):
            for i, (key, _lab) in enumerate(REGIMES):
                v = vals.get((panel, key, arm))
                if v is None:
                    continue
                rho, err, _n, _nok, _ntot, failed = v
                x = centers[i] + (j - (n_m - 1) / 2) * width
                ax.bar(
                    x,
                    rho,
                    width * 0.9,
                    yerr=max(0.0, err),
                    color=color,
                    alpha=0.35 if failed else 1.0,
                    edgecolor=color,
                    hatch="//" if failed else None,
                    linewidth=1.0,
                    error_kw=dict(lw=1.0, capsize=2.0, ecolor="#333333"),
                    zorder=3,
                )
        labels = []
        for i, (key, lab) in enumerate(REGIMES):
            if panel == "average":
                labels.append(lab if i % 2 == 0 else f"\n\n{lab}")
                continue
            sub = SHARED.get(key) or CORPUS[panel][key]
            txt = f"{lab}\n({sub})"
            labels.append(txt if i % 2 == 0 else f"\n\n{txt}")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=8.4)
        ax.axhline(0.0, color="#666666", lw=0.9, zorder=2)
        ax.set_ylim(-0.18, 0.95)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        title = (
            "average across behaviours — spread-failed cells excluded"
            if panel == "average"
            else panel
        )
        ax.set_title(title, fontsize=12, fontweight="semibold", loc="left")
        if panel == "evil":
            ax.set_ylabel("Spearman rho, prediction vs judged behaviour expression", fontsize=10)

    handles = [mpatches.Patch(facecolor=c, label=lab) for _a, lab, c in methods]
    handles.append(
        mpatches.Patch(
            facecolor="#999999",
            alpha=0.35,
            hatch="//",
            label="spread gate failed — not interpretable",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.155),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Result 2 ({protocol} protocol), persona-vector spread-flagged view: "
        f"reads across evaluation regimes, hatched where the DV spread gate fails",
        fontsize=12.5,
        x=0.012,
        ha="left",
        y=0.985,
    )
    fig.text(0.012, 0.006, caption, fontsize=8.2, color="#333333", va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.26, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def build_caption(vals: dict, protocol: str, methods) -> str:
    proto = {
        "P-A": "P-A: the readout trains on ONE trait-eliciting dataset (the `train` budget cell) "
        "plus the judged WildChat train split.",
        "P-B": "P-B (LODO): the readout trains on an 80% group-level slice of every "
        "trait-eliciting dataset EXCEPT one held out whole, plus the judged WildChat train "
        "split; one fit per holdout, and the 'completely OOD' bar is each fit's own held-out "
        "dataset.",
    }[protocol]
    ns = []
    for b in BEHAVIORS:
        row = []
        for key, lab in REGIMES:
            v = next((vals[(b, key, a)] for a, *_ in methods if (b, key, a) in vals), None)
            if v:
                row.append(f"{lab} {v[2]:,}" + (f" [{v[3]}/{v[4]} rungs kept]" if v[4] > 1 else ""))
        ns.append(f"  {b}: " + "; ".join(row))
    return (
        f"{proto}\n"
        "The context->answer MAP is identical under both protocols: fit once per behaviour on the "
        "ADD pool (generic WildChat + the `train` eliciting corpus) and frozen. The persona-vector "
        "projection arms shown here are NOT label-consuming -- they project onto r_B built from the "
        "judge-filtered synthetic extraction set -- so they are bit-identical across P-A and P-B "
        "except in the in-distribution regime, where P-A reads `train` out-of-fold and P-B reads "
        "the 20% held-in slice (different eval subsets, not a LODO effect).\n"
        "Bars are the mean over spread-PASSING rungs in the regime; error bars are the committed "
        "bootstrap CI for single-rung regimes and the s.e.m. across rungs for multi-rung ones. "
        "Hatched + muted = every rung in that cell FAILED the DV spread gate (sd >= 10 and "
        "bottom/top bin <= 0.80 on a 0-100 DV; 0-1 binary DVs rescaled) -- kept for completeness, "
        "not interpretable. Spread failures: evil hhrt / toxicchat / evil_pair / wildchat_rung.\n"
        "Only the CONTEXT-based variant is scored (variant=context_end); the prefix-end arm is a "
        "stated scope deviation inherited from the parent round. Mapping is LINEAR throughout; no "
        "MLP or kernel arm.\n"
        "n_eval per behaviour x regime --\n" + "\n".join(ns)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-commit", default="5aae0a472b")
    ap.add_argument("--spread-json", default="/tmp/spread_1739.json")
    ap.add_argument("--out-dir", default="figures/issue_1739/pv_regime_view")
    ap.add_argument("--with-ridge", action="store_true", help="add the two r2v2 ridge arms")
    args = ap.parse_args()

    methods = PV_METHODS + (RIDGE_METHODS if args.with_ridge else ())
    fits = {
        b: _gj(args.fits_commit, f"eval_results/issue_1739/r2v2_fits/{b}/all_arms_spearman.json")
        for b in BEHAVIORS
    }
    spread = json.loads(Path(args.spread_json).read_text())
    out_dir = _REPO_ROOT / args.out_dir
    sfx = "_withridge" if args.with_ridge else ""

    for protocol in ("P-A", "P-B"):
        vals, _ = collect(fits, spread, protocol, methods)
        cap = build_caption(vals, protocol, methods)
        draw(
            vals,
            protocol,
            methods,
            out_dir / f"pv_regime_{protocol.replace('-', '').lower()}{sfx}.png",
            cap,
        )


if __name__ == "__main__":
    main()
