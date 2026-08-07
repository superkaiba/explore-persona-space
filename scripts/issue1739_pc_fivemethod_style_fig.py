#!/usr/bin/env python3
"""#1739 r2v2: PV readouts by evaluation regime, in the result2_fivemethod style.

Restyles the P-A / P-B setting figures to match the committed
``figures/issue_1739/result2_fivemethod/result2_fivemethod.png`` layout
(d78aff9e5c):

  * one row of four panels -- evil / sycophancy / hallucination / average
  * evaluation-regime groups per panel, each x label naming the CORPUS that
    actually contributed plus the protocol-specific readout qualifier
    (P-B reads are LODO: 20% held-in in-distribution, each OOD dataset held
    out whole)
  * one colour per method, no hatch anywhere
  * DV-spread-gate failures are DROPPED, never plotted: a floored dataset
    leaves its bar's mean AND its regime sub-label, and a regime whose every
    dataset is floored is removed from that panel entirely (evil loses its
    whole generic-chat group, so evil shows 3 regimes and the others 4)
  * a caption block carrying protocol provenance, what was dropped, and
    per-group n_eval

Methods: the result2_fivemethod reference roster MINUS arm12_oracle_reg
("Ridge regression on real answer"), which P-A/P-B never scored -- ridge on
context, ridge on the mapped answer, PV on the mapped answer, PV on the real
answer. ``--pv-only`` switches to the three persona-vector arms instead.

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
import textwrap
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

# Display name per OOD rung -- the 'completely OOD' sub-label is built from the
# rungs that actually SURVIVE the spread gate, so a dropped floored dataset
# never appears in the label of a bar it did not contribute to.
RUNG_NAME = {
    "hhrt": "hh-rlhf",
    "toxicchat": "ToxicChat",
    "evil_mhj": "MHJ",
    "evil_pair": "PAIR",
    "evil_tomgibbs": "TomGibbs",
    "aita": "AITA",
    "sycoans": "SycEval-answer",
    "sycoays": "SycEval-are-you-sure",
    "sycofb": "SycEval-feedback",
    "sycomim": "SycEval-mimicry",
    "sycomwe": "MWE",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
}

# per-behaviour in-distribution corpus (the OOD side is built from RUNG_NAME)
INDIST_CORPUS = {
    "evil": "jailbreak x forbidden Qs",
    "sycophancy": "Reddit personal-advice posts",
    "hallucination": "TriviaQA (rc.nocontext)",
}
# protocol-specific qualifier on the two readout-dependent regimes: under P-B
# the in-distribution read is the LODO 20% held-in slice and each OOD bar is
# that fit's own whole-held-out dataset.
INDIST_QUALIFIER = {"P-A": "out-of-fold", "P-B": "LODO, 20% held-in"}
OOD_QUALIFIER = {"P-A": "single fit", "P-B": "LODO, held out whole"}

SHARED = {"pvsynth": "persona-vector grid", "generic": "random WildChat"}
REGIMES = (
    ("pvsynth", "synthetic"),
    ("generic", "generic chat"),
    ("indist", "in-distribution"),
    ("ood", "completely OOD"),
)

# The result2_fivemethod reference roster (d78aff9e5c), same order + colours,
# MINUS arm12_oracle_reg ("Ridge regression on real answer"), which this
# protocol never scored -- see the caption for where that arm does exist.
REF_METHODS = (
    ("arm4_ridge_ctx", "Ridge regression on context", "#1f77b4"),
    ("arm7_map_ridge_pred", "Ridge regression on mapped answer", "#e8b23a"),
    ("arm6_map_proj_e1", "Persona vector on mapped answer", "#8c3b1e"),
    ("arm11_oracle_proj", "Persona vector on real answer", "#1a6b54"),
)
PV_METHODS = (
    ("arm1_ctx_e1", "Persona vector on context", "#4C72B0"),
    ("arm6_map_proj_e1", "Persona vector on mapped answer", "#8c3b1e"),
    ("arm11_oracle_proj", "Persona vector on real answer (oracle)", "#1a6b54"),
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
                # Floored (spread-gate-failing) datasets are DROPPED outright --
                # never averaged into a bar, never drawn hatched. A regime whose
                # every dataset is floored contributes no bar at all, and
                # panel_regimes() then drops the whole x-slot from that panel.
                use = [(rn, r) for rn, r in items if ok(rn)]
                dropped = tuple(rn for rn, _r in items if not ok(rn))
                if not use:
                    continue
                rho = float(st.mean([r["rho_frozen"] for _rn, r in use]))
                if len(use) == 1 and use[0][1].get("ci_frozen"):
                    lo, hi = use[0][1]["ci_frozen"]
                    err = float(max(0.0, max(hi - rho, rho - lo)))
                else:
                    err = _sem([r["rho_frozen"] for _rn, r in use])
                n_eval = sum(int(r.get("n_eval") or 0) for _rn, r in use)
                vals[(beh, key, arm)] = (
                    rho,
                    err,
                    n_eval,
                    len(use),
                    len(items),
                    tuple(rn for rn, _r in use),
                    dropped,
                )

    for key, _ in REGIMES:
        for arm, *_ in methods:
            per = [vals[(b, key, arm)] for b in BEHAVIORS if (b, key, arm) in vals]
            if per:
                vals[("average", key, arm)] = (
                    float(st.mean([p[0] for p in per])),
                    _sem([p[0] for p in per]),
                    sum(p[2] for p in per),
                    len(per),
                    len(BEHAVIORS),
                    (),
                    (),
                )
    return vals, {}


def panel_regimes(vals: dict, panel: str, methods) -> list[tuple[str, str]]:
    """Regimes with >=1 drawable bar in this panel (floored ones already gone)."""
    return [
        (key, lab) for key, lab in REGIMES if any((panel, key, arm) in vals for arm, *_ in methods)
    ]


def regime_sublabel(vals: dict, panel: str, key: str, protocol: str, methods) -> str:
    """Corpus text under a regime label, naming only datasets that CONTRIBUTED."""
    if key in SHARED:
        return f"({SHARED[key]})"
    if key == "indist":
        return textwrap.fill(INDIST_CORPUS[panel], width=20) + f"\n({INDIST_QUALIFIER[protocol]})"
    kept: tuple = ()
    for arm, *_ in methods:
        v = vals.get((panel, key, arm))
        if v is not None:
            kept = v[5]
            break
    # Collapse the four SycophancyEval sub-corpora to one token when ALL four
    # survive; if any is dropped the survivors are named individually so the
    # label can never over-claim coverage.
    syceval = ("sycoans", "sycoays", "sycofb", "sycomim")
    if all(s in kept for s in syceval):
        names = ", ".join(
            "SycophancyEval x4" if r == syceval[0] else RUNG_NAME.get(r, r)
            for r in kept
            if r not in syceval[1:]
        )
    else:
        names = ", ".join(RUNG_NAME.get(r, r) for r in kept)
    return textwrap.fill(names, width=20) + f"\n({OOD_QUALIFIER[protocol]})"


def draw(vals: dict, protocol: str, methods, out_png: Path, caption: str) -> None:
    panels = [*BEHAVIORS, "average"]
    fig, axes = plt.subplots(1, 4, figsize=(23.0, 9.6))
    n_m = len(methods)
    width = 0.72 / n_m

    for ax, panel in zip(axes, panels, strict=True):
        regimes = panel_regimes(vals, panel, methods)
        centers = np.arange(len(regimes))
        for j, (arm, _label, color) in enumerate(methods):
            for i, (key, _lab) in enumerate(regimes):
                v = vals.get((panel, key, arm))
                if v is None:
                    continue
                rho, err = v[0], v[1]
                x = centers[i] + (j - (n_m - 1) / 2) * width
                ax.bar(
                    x,
                    rho,
                    width * 0.9,
                    yerr=max(0.0, err),
                    color=color,
                    edgecolor=color,
                    linewidth=1.0,
                    error_kw=dict(lw=1.0, capsize=2.0, ecolor="#333333"),
                    zorder=3,
                )
        labels = []
        for key, lab in regimes:
            if panel == "average":
                labels.append(lab)
                continue
            labels.append(f"{lab}\n{regime_sublabel(vals, panel, key, protocol, methods)}")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=7.0, linespacing=1.35)
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
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.155),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Result 2 ({protocol} protocol): reads across evaluation regimes — "
        f"datasets failing the DV spread gate are dropped, not plotted",
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
    dropped_note: list[str] = []
    for b in BEHAVIORS:
        row = []
        for key, lab in REGIMES:
            v = next((vals[(b, key, a)] for a, *_ in methods if (b, key, a) in vals), None)
            if v is None:
                row.append(f"{lab} DROPPED (every dataset floored)")
                continue
            row.append(f"{lab} {v[2]:,}" + (f" [{v[3]}/{v[4]} rungs kept]" if v[4] > 1 else ""))
            if v[6]:
                dropped_note.append(f"{b}/{lab}: " + ", ".join(v[6]))
        ns.append(f"  {b}: " + "; ".join(row))
    drops = "; ".join(dropped_note) if dropped_note else "none"
    return (
        f"{proto}\n"
        "The context->answer MAP is identical under both protocols: fit once per behaviour on the "
        "ADD pool (generic WildChat + the `train` eliciting corpus) and frozen. The persona-vector "
        "projection arms shown here are NOT label-consuming -- they project onto r_B built from the "
        "judge-filtered synthetic extraction set -- so they are bit-identical across P-A and P-B "
        "except in the in-distribution regime, where P-A reads `train` out-of-fold and P-B reads "
        "the LODO 20% held-in slice (different eval subsets, not a LODO effect on the arm itself).\n"
        "SPREAD GATE: a dataset is DROPPED -- excluded from its bar's mean and from the regime "
        "sub-label -- when it fails sd >= 10 and bottom/top bin <= 0.80 on a 0-100 DV (0-1 binary "
        "DVs rescaled). No floored data is plotted anywhere in this figure. "
        f"Datasets dropped: {drops}. "
        "evil's generic-chat regime is dropped WHOLE: its only dataset (wildchat_rung) is floored "
        "(sd 4.4, 98.9% at the bottom bin), so the group has no interpretable data and evil shows "
        "3 regimes where the other behaviours show 4. Sycophancy and hallucination lose nothing.\n"
        "Bars are the mean over the surviving datasets; error bars are the committed bootstrap CI "
        "for single-dataset regimes and the s.e.m. across datasets for multi-dataset ones. "
        "Only the CONTEXT-based variant is scored (variant=context_end); the prefix-end arm is a "
        "stated scope deviation inherited from the parent round. Mapping is LINEAR throughout; no "
        "MLP or kernel arm. METHOD ROSTER = the result2_fivemethod reference roster MINUS "
        "'Ridge regression on real answer' (arm12_oracle_reg), which P-A/P-B never scored.\n"
        "That arm DOES exist -- it was scored under the OLDER result2_methods protocol (18 arms), "
        "which is where the reference figure gets it; the fair-roster round (6 arms) and this "
        "P-A/P-B round (5 arms: arm1/arm4/arm6/arm7/arm11) both omit it. result2_methods also\n"
        "carries arm8_map_ridge_true (ridge FIT on the real answer, APPLIED to the mapped answer), "
        "which appears in neither the reference figure nor this one.\n"
        "n_eval per behaviour x regime --\n" + "\n".join(ns)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-commit", default="5aae0a472b")
    ap.add_argument("--spread-json", default="/tmp/spread_1739.json")
    ap.add_argument("--out-dir", default="figures/issue_1739/pv_regime_view")
    ap.add_argument("--pv-only", action="store_true", help="3 persona-vector arms instead")
    args = ap.parse_args()

    methods = PV_METHODS if args.pv_only else REF_METHODS
    fits = {
        b: _gj(args.fits_commit, f"eval_results/issue_1739/r2v2_fits/{b}/all_arms_spearman.json")
        for b in BEHAVIORS
    }
    spread = json.loads(Path(args.spread_json).read_text())
    out_dir = _REPO_ROOT / args.out_dir
    sfx = "_pvonly" if args.pv_only else ""

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
