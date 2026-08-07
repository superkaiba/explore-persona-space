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
  * one colour per method; hatch means "not interpretable", never identity
  * DV-spread-gate failures, handled two ways. PARTIAL floor (some datasets in
    a regime fail): those datasets leave the bar's mean AND the regime
    sub-label; the survivors carry the bar. TOTAL floor (every dataset fails):
    the regime is still plotted from the floored data but HATCHED + muted and
    excluded from the average panel -- evil's generic chat is the only such
    regime, kept visible so it reads as measured-and-uninterpretable rather
    than silently absent
  * a caption block carrying protocol provenance, what was dropped, what was
    scope-excluded, and per-group n_eval

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

# SCOPE exclusions -- deliberately NOT the spread gate. These datasets PASS the
# gate; they are removed by an explicit user scope call (2026-08-07: restrict
# evil's completely-OOD regime to MHJ). Tracked separately from gate drops so
# the caption can never present a scope choice as a data-quality exclusion.
# Revert by emptying this dict.
SCOPE_EXCLUDE: dict[tuple[str, str], tuple[str, ...]] = {
    ("evil", "ood"): ("evil_tomgibbs",),
}

SHARED = {"pvsynth": "persona-vector grid", "generic": "random WildChat"}
REGIMES = (
    ("pvsynth", "synthetic"),
    ("generic", "generic chat"),
    ("indist", "in-distribution"),
    ("ood", "completely OOD"),
)

# Regimes whose cells are scored on the SAME eval contexts. Under P-B every
# holdout fit reads one common pvsynth grid / WildChat slice / 20% held-in
# slice (verified: n_eval is identical across the K cells, and the
# non-label-consuming PV arms return literally ONE distinct rho), so the
# context-sampling error is COMMON to the cells and cannot average down --
# only the fit-to-fit spread does. 'ood' is the opposite: each cell is a
# different held-out corpus. Under P-A every non-OOD regime is a single cell,
# where both rules collapse to that cell's own bootstrap SE.
SHARED_EVAL_CONTEXTS = frozenset({"pvsynth", "generic", "indist"})

_Z95 = 1.959963984540054

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


def _se_from_ci(row: dict, where: str) -> float:
    """Per-cell sampling SE of rho, from its committed 95% bootstrap CI.

    ``ci_frozen`` is the [2.5%, 97.5%] percentile interval of a nonparametric
    bootstrap that RESAMPLES EVAL CONTEXTS (arms.evaluate_transfer), so its
    half-width / 1.96 is the context-sampling SE of that cell's rho. Fail loud
    rather than substitute a zero: a silently missing CI would draw a bar that
    reads as precise.
    """
    ci = row.get("ci_frozen")
    if not ci or len(ci) != 2 or any(x is None or math.isnan(float(x)) for x in ci):
        raise ValueError(f"{where}: no usable ci_frozen — cannot build a calibrated error bar")
    return max(0.0, (float(ci[1]) - float(ci[0])) / (2.0 * _Z95))


def _combine_se(rhos: list[float], ses: list[float], *, shared: bool) -> tuple[float, dict]:
    """SE of the PLOTTED MEAN: within-cell bootstrap error + between-cell spread.

    One definition for every bar in both figures, so heights are comparable
    across regimes, protocols and panels. The two branches differ only in
    whether the within-cell error averages down:

    * ``shared=True``  -- the cells are K reads of the SAME eval contexts, so
      their context-sampling errors are common: ``var = v_bar + s2/K``.
      A regime whose K cells are numerically identical (the PV arms under
      P-B) therefore keeps its real bootstrap SE instead of collapsing to
      exactly 0, which is what averaging K copies of one number produced.
    * ``shared=False`` -- the cells are disjoint corpora (each OOD dataset;
      each behaviour in the average panel), so the within-cell error DOES
      average down and the between-cell spread is corrected for the
      within-cell noise it already contains (a DerSimonian-Laird-style moment
      estimator for the heterogeneity, floored at 0):
      ``var = (v_bar + max(0, s2 - v_bar)) / K``.

    K == 1 collapses both branches to that cell's own bootstrap SE.
    """
    k = len(rhos)
    vbar = sum(s * s for s in ses) / k
    s2 = st.variance(rhos) if k > 1 else 0.0
    if shared:
        var = vbar + (s2 / k if k > 1 else 0.0)
    else:
        var = (vbar + (max(0.0, s2 - vbar) if k > 1 else 0.0)) / k
    return math.sqrt(var), {
        "k": k,
        "se_within": math.sqrt(vbar),
        "sd_between": math.sqrt(s2),
        "shared": shared,
    }


def collect(fits: dict, spread: dict, protocol: str, methods) -> tuple[dict, dict]:
    """(behavior, regime, arm) -> (rho, err, n_eval, n_use, n_in_scope, kept, gate_dropped, scope_dropped, all_failed)."""  # noqa: E501
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
                scoped_out = SCOPE_EXCLUDE.get((beh, key), ())
                in_scope = [(rn, r) for rn, r in items if rn not in scoped_out]
                passing = [(rn, r) for rn, r in in_scope if ok(rn)]
                scope_dropped = tuple(rn for rn, _r in items if rn in scoped_out)
                # Partial floor within a regime -> the floored datasets are
                # DROPPED from the mean and the sub-label. TOTAL floor (every
                # dataset in the regime fails) -> the regime is still PLOTTED,
                # from the floored data, but HATCHED and excluded from the
                # average panel: dropping it silently would hide that the
                # regime was measured and came back uninterpretable.
                all_failed = not passing
                use = in_scope if all_failed else passing
                dropped = () if all_failed else tuple(rn for rn, _r in in_scope if not ok(rn))
                if not use:
                    continue
                rhos = [float(r["rho_frozen"]) for _rn, r in use]
                rho = float(st.mean(rhos))
                shared = key in SHARED_EVAL_CONTEXTS
                if shared and len({int(r.get("n_eval") or 0) for _rn, r in use}) > 1:
                    raise ValueError(
                        f"{beh}/{key}/{arm}: regime is declared shared-context but its cells "
                        "disagree on n_eval — the shared-context SE rule does not apply"
                    )
                ses = [_se_from_ci(r, f"{beh}/{key}/{arm}/{rn}") for rn, r in use]
                err, ci_parts = _combine_se(rhos, ses, shared=shared)
                # A shared-context regime reports ONE eval set, not K copies of it.
                n_eval = (
                    int(use[0][1].get("n_eval") or 0)
                    if shared
                    else sum(int(r.get("n_eval") or 0) for _rn, r in use)
                )
                vals[(beh, key, arm)] = (
                    rho,
                    err,
                    n_eval,
                    len(use),
                    len(in_scope),
                    tuple(rn for rn, _r in use),
                    dropped,
                    scope_dropped,
                    all_failed,
                    ci_parts,
                )

    for key, _ in REGIMES:
        for arm, *_ in methods:
            # Hatched (all-floored) cells are excluded from the average panel.
            per = [
                vals[(b, key, arm)]
                for b in BEHAVIORS
                if (b, key, arm) in vals and not vals[(b, key, arm)][8]
            ]
            if per:
                # Behaviours are disjoint corpora, so the panel mean propagates
                # each behaviour bar's own SE and adds the behaviour-to-behaviour
                # heterogeneity on top.
                avg_err, avg_parts = _combine_se(
                    [p[0] for p in per], [p[1] for p in per], shared=False
                )
                vals[("average", key, arm)] = (
                    float(st.mean([p[0] for p in per])),
                    avg_err,
                    sum(p[2] for p in per),
                    len(per),
                    len(BEHAVIORS),
                    (),
                    (),
                    (),
                    False,
                    avg_parts,
                )
    return vals, {}


def panel_regimes(vals: dict, panel: str, methods) -> list[tuple[str, str]]:
    """Regimes with >=1 drawable bar in this panel (hatched all-floored ones count)."""
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
    fig, axes = plt.subplots(1, 4, figsize=(23.0, 13.0))
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
                rho, err, failed = v[0], v[1], v[8]
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
    handles.append(
        mpatches.Patch(
            facecolor="#999999",
            alpha=0.35,
            hatch="//",
            label="every dataset in the regime floored — NOT interpretable",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.305),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        f"Result 2 ({protocol} protocol): reads across evaluation regimes — "
        f"floored datasets dropped; a wholly-floored regime is hatched, not dropped",
        fontsize=12.5,
        x=0.012,
        ha="left",
        y=0.985,
    )
    # Hard-wrap every caption line: one over-long line silently stretches the
    # whole canvas via bbox_inches="tight" (P-B hit 8660 px once).
    caption = "\n".join(
        textwrap.fill(ln, width=210, subsequent_indent="  ") if len(ln) > 210 else ln
        for ln in caption.split("\n")
    )
    fig.text(0.012, 0.006, caption, fontsize=8.2, color="#333333", va="bottom", ha="left")
    fig.tight_layout(rect=(0.0, 0.38, 1.0, 0.95))
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
    scope_note: list[str] = []
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
            if len(v) > 7 and v[7]:
                scope_note.append(f"{b}/{lab}: " + ", ".join(v[7]))
        ns.append(f"  {b}: " + "; ".join(row))
    drops = "; ".join(dropped_note) if dropped_note else "none"
    scoped = "; ".join(scope_note) if scope_note else "none"
    return (
        f"{proto}\n"
        "The context->answer MAP is identical under both protocols: fit once per behaviour on the "
        "ADD pool (generic WildChat + the `train` eliciting corpus) and frozen. The persona-vector "
        "projection arms shown here are NOT label-consuming -- they project onto r_B built from the "
        "judge-filtered synthetic extraction set -- so they are bit-identical across P-A and P-B "
        "except in the in-distribution regime, where P-A reads `train` out-of-fold and P-B reads "
        "the LODO 20% held-in slice (different eval subsets, not a LODO effect on the arm itself).\n"
        "SPREAD GATE (sd >= 10 and bottom/top bin <= 0.80 on a 0-100 DV; 0-1 binary DVs rescaled), "
        "applied two ways. PARTIAL floor -- some datasets in a regime fail: those are DROPPED from "
        "the bar's mean and from the regime sub-label, and the surviving datasets carry the bar. "
        f"Datasets dropped this way: {drops}. TOTAL floor -- EVERY dataset in a regime fails: the "
        "regime is still PLOTTED, from the floored data, but HATCHED + muted and EXCLUDED from the "
        "average panel. evil's generic chat is the only such regime (its sole dataset "
        "wildchat_rung is floored at sd 4.4 with 98.9% of mass in the bottom bin): its bars are "
        "shown so the regime is visibly measured-and-uninterpretable rather than silently absent, "
        "and they must NOT be read as effect sizes -- at that floor the rho is not estimable. "
        "Sycophancy and hallucination lose nothing either way.\n"
        f"SEPARATELY -- SCOPE EXCLUSION, NOT A GATE FAILURE: {scoped}. These datasets PASS the "
        "spread gate and were removed by an explicit user scope call, so evil's completely-OOD "
        "bar is now a SINGLE dataset (MHJ), not a mean over its OOD ladder. evil_tomgibbs is the "
        "excluded one and it is NOT neutral: ridge-on-mapped scored -0.399 (P-A) / -0.337 (P-B) "
        "there -- strongly NEGATIVE, and the only CI-disjoint context-vs-mapped comparison in the "
        "whole evil OOD set. Removing it therefore REMOVES the one setting where the mapped-answer "
        "readout was decisively worse than the context readout; read evil's OOD bar as 'MHJ only', "
        "never as evil OOD in general.\n"
        "ERROR BARS -- one definition for every bar in both figures: +/-1 standard error of the "
        "PLOTTED MEAN, combining (i) within-cell sampling error over eval contexts, taken as "
        "half-width/1.96 of each cell's committed 95% bootstrap CI (the bootstrap resamples eval "
        "contexts), and (ii) between-cell spread. The combination rule follows whether the "
        "averaged cells share eval contexts. The three non-OOD regimes do: under P-B the K holdout "
        "fits all read ONE common pvsynth grid / WildChat slice / 20% held-in slice, so the "
        "context error is COMMON and does not average down -- var = v_within + s2_between/K. The "
        "completely-OOD regime does not (each cell is a different held-out corpus), nor do the "
        "three behaviours in the average panel, so there var = (v_within + tau2)/K with tau2 = "
        "max(0, s2_between - v_within) correcting the observed spread for the within-cell noise it "
        "already contains. A single-cell bar reduces to that cell's own bootstrap SE under both "
        "rules. This SUPERSEDES the previous mixed convention (bootstrap CI for single-dataset "
        "bars, bare s.e.m. across cells otherwise), whose s.e.m. branch read EXACTLY 0.000 for the "
        "persona-vector arms in every shared-context P-B regime -- those arms are not "
        "label-consuming, so the K LODO fits return literally one repeated number and their spread "
        "measures nothing; the bars looked precise where no precision had been estimated. "
        "Correspondingly, n_eval for a shared-context regime below now counts that ONE eval set, "
        "not K copies of it. "
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
