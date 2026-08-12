"""Result 5 for #1739: what each KIND of training data buys the predictors.

Renders four figures into `figures/issue_1739/result5_data/`, one per question
the writeup's Result 5 asks:

  r5_generic_map_ladder   How much does the map improve as the UNLABELED generic
                          pool grows?  rho vs U in {250, 5000, full=18,793},
                          per arm, on the in-distribution rung and each OOD rung.
  r5_judged_generic_swap  What happens when JUDGED GENERIC labels replace judged
                          ELICITING labels in the behavior readout?  rho vs
                          g_generic in {0, 0.25, 0.5, 1.0} at a FIXED labeled
                          budget of 1,500, so composition is never confounded
                          with quantity.
  r5_fu_fl_factorial      The two composition channels crossed: f_U (fraction of
                          the unlabeled MAP pool that is trait-eliciting) x f_L
                          (same for the labeled READOUT pool), at three labeled
                          budgets.
  r5_lodo_by_dataset      Leave-one-dataset-out: with dataset D held out of the
                          readout fit entirely, how does rho on D compare to rho
                          on the other eliciting datasets and on generic chat?

Every number is READ from a committed per-arm Spearman artifact; nothing is
re-fit here. Replicate spread, where the source carries draws and seeds, is
summarized as mean +/- 1.96 * SEM over those replicates.

Provenance, one source per figure:
  ladder / factorial  eval_results/issue_1739/<behavior>/arm_results/all_arms_spearman.json
                      (the main 1,906-cell grid: u_rung_label in {250, 5000,
                      full} plus the compose cells at U=5,000)
  judged-generic swap eval_results/issue_1739/judged_generic_ablation/<behavior>/
                      all_arms_spearman.json (mode `jobd_swap`, L fixed at 1,500)
  LODO                eval_results/issue_1739/r2v2_fits_widegrid/<behavior>/
                      all_arms_spearman.json (fit = P-B-holdout-<dataset>)

SCOPE NOTE, stated rather than left to be discovered: the ladder and factorial
sources predate the corpus roster that the P-A/P-B figures use, so their OOD
rungs are the ORIGINAL ones (evil hh-rlhf / ToxicChat, sycophancy held-out
r/socialskills, hallucination NQ-Open / SimpleQA). The LODO figure reads the
newer widegrid and therefore does cover MHJ / PAIR / tom-gibbs. Each figure's
caption names the rungs it actually plots.

No fits, no GPU, no network.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import textwrap  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, ROOT  # noqa: E402

ER = ROOT / "eval_results/issue_1739"
OUT_FIG = ROOT / "figures/issue_1739/result5_data"
OUT_NUM = ER / "result5_data"

# One colour = one meaning across all four figures (and matching the Result 2/3
# family: context reads blue, mapped-answer reads warm, real-answer reads green).
ARM_STYLE: dict[str, tuple[str, str]] = {
    "arm1_ctx_e1": ("Persona vector on context", "#4C72B0"),
    "arm4_ridge_ctx": ("Ridge regression on context", "#0B3C5D"),
    "arm6_map_proj_e1": ("Persona vector on mapped answer", "#8c3b1e"),
    "arm7_map_ridge_pred": ("Ridge regression on mapped answer", "#e8b23a"),
    "arm11_oracle_proj": ("Persona vector on real answer (oracle)", "#1a6b54"),
    "arm12_oracle_reg": ("Ridge regression on real answer (oracle)", "#009E73"),
    "arm8_map_ridge_true": ("Ridge fit on real, applied to mapped", "#b8860b"),
    "arm13_shuffled_map": ("control: shuffled map", "#9A9A9A"),
    # NOTE: arm1 (medium blue) vs arm4 (navy) must stay visually separable --
    # they are the two CONTEXT-side reads and the figure's whole point is the
    # gap between them.
}
# Reporting order; any arm absent from a source is simply skipped.
ARM_ORDER = list(ARM_STYLE)

RUNG_NAME = {
    "train": "in-distribution (out-of-fold)",
    "hhrt": "hh-rlhf red-team",
    "toxicchat": "ToxicChat",
    "aita": "held-out r/socialskills",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
    "wildchat_rung": "random WildChat",
    "evil_mhj": "MHJ",
    "evil_pair": "PAIR",
    "evil_tomgibbs": "tom-gibbs",
    "pvsynth": "persona-vector grid",
}

# Operating slice shared by the ladder and factorial reads: the context-based
# variant and the paper-faithful synthetic extraction regime, so a curve never
# mixes extraction regimes or input states.
SLICE = {"variant": "context_end", "regime": "e1"}

# The LODO figure reads the same pinned P-A/P-B round the Result 2/3 figures do.
# Its artifacts are not at HEAD, so they are read out of git at this commit --
# the widegrid copy at HEAD has no sycophancy leg.
LODO_FITS_COMMIT = "5aae0a472b"


def _lodo_rows(behavior: str) -> list[dict]:
    """P-A/P-B transfer rows for one behavior, read at the pinned commit."""
    import subprocess

    path = f"eval_results/issue_1739/r2v2_fits/{behavior}/all_arms_spearman.json"
    out = subprocess.run(
        ["git", "show", f"{LODO_FITS_COMMIT}:{path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout).get("transfer_rows") or []


def _rows(path: Path, key: str) -> list[dict]:
    d = json.loads(path.read_text())
    return d.get(key) or []


def _agg(rows: list[dict]) -> tuple[float, float] | None:
    """Mean rho over replicates and its 1.96 * SEM half-width.

    A single-replicate cell has no estimable replicate spread; its half-width is
    reported as 0.0 and the caption says so, rather than manufacturing an
    interval from one point.
    """
    v = np.array([r["rho_frozen"] for r in rows if r.get("rho_frozen") is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(v.size))


def _errorbar(ax, x, mu, half, **kw) -> None:
    """Plot with NON-NEGATIVE y offsets (matplotlib rejects negative yerr)."""
    lo = np.maximum(0.0, np.asarray(half, dtype=float))
    ax.errorbar(x, mu, yerr=[lo, lo], **kw)


def _finish(fig, out_slug: str, caption: str) -> Path:
    # Wrap before drawing: savefig_paper writes with a tight bbox, so a single
    # long line expands the canvas to the width of the text.
    wrapped = "\n".join(textwrap.wrap(caption, width=165))
    fig.text(0.006, 0.005, wrapped, ha="left", va="bottom", fontsize=7.2, color="#333333")
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, OUT_FIG / out_slug)
    plt.close(fig)
    print(f"wrote {OUT_FIG / out_slug}.png")
    return OUT_FIG / out_slug


# --- figure 1: unlabeled generic map pool -------------------------------------


def fig_generic_map_ladder() -> dict:
    """rho vs the size of the UNLABELED map pool, per arm, per eval rung."""
    U_ORDER = ["250", "5000", "full"]
    U_X = {"250": 0, "5000": 1, "full": 2}
    per_behavior: dict[str, dict] = {}
    for b in BEHAVIORS:
        rows = _rows(ER / b / "arm_results/all_arms_spearman.json", "transfer_rows")
        rows = [r for r in rows if all(str(r.get(k)) == v for k, v in SLICE.items())]
        # Largest labeled budget available for this behavior: the operating point.
        lmax = max(int(r["budget_l"]) for r in rows)
        rows = [r for r in rows if int(r["budget_l"]) == lmax]
        cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for r in rows:
            cells[(str(r["eval_rung"]), str(r["arm"]), str(r["u_rung_label"]))].append(r)
        per_behavior[b] = {"cells": cells, "budget_l": lmax}

    rungs = {b: sorted({k[0] for k in per_behavior[b]["cells"]}) for b in BEHAVIORS}
    ncol = max(len(v) for v in rungs.values())
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        len(BEHAVIORS), ncol, figsize=(4.6 * ncol, 3.6 * len(BEHAVIORS)), squeeze=False
    )
    table: list[dict] = []
    for i, b in enumerate(BEHAVIORS):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(rungs[b]):
                ax.axis("off")
                continue
            rung = rungs[b][j]
            for arm in ARM_ORDER:
                xs, mus, hws = [], [], []
                for u in U_ORDER:
                    got = _agg(per_behavior[b]["cells"].get((rung, arm, u), []))
                    if got is None:
                        continue
                    xs.append(U_X[u])
                    mus.append(got[0])
                    hws.append(got[1])
                    table.append(
                        {
                            "figure": "generic_map_ladder",
                            "behavior": b,
                            "eval_rung": rung,
                            "arm": arm,
                            "u_rung_label": u,
                            "budget_l": per_behavior[b]["budget_l"],
                            "rho_mean": got[0],
                            "ci_halfwidth": got[1],
                            "n_replicates": len(per_behavior[b]["cells"][(rung, arm, u)]),
                        }
                    )
                if not xs:
                    continue
                label, color = ARM_STYLE[arm]
                _errorbar(
                    ax, xs, mus, hws, color=color, marker="o", ms=4, lw=1.6, capsize=2, label=label
                )
            ax.axhline(0.0, color="#666666", lw=0.8)
            ax.set_xticks(list(U_X.values()))
            ax.set_xticklabels(["250", "5,000", "18,793\n(full)"], fontsize=8)
            ax.set_title(f"{b} — {RUNG_NAME.get(rung, rung)}", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel("Spearman rho")

    fig.supxlabel("unlabeled generic context->answer pairs in the map fit", fontsize=9.5, y=0.10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3,
        frameon=False, fontsize=8.5,
    )
    fig.suptitle(
        "Result 5a: does a bigger UNLABELED generic map pool buy behavior prediction?",
        x=0.006,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.18, hspace=0.5, wspace=0.22)
    _finish(
        fig,
        "r5_generic_map_ladder",
        "Each panel is one behavior x evaluation rung. x is the number of unlabeled generic "
        "WildChat context->answer pairs the MAP was fit on; the labeled readout budget is held at "
        "each behavior's maximum (evil 8,000; sycophancy and hallucination 16,000). Points are the "
        "mean frozen-layer Spearman rho over the source's 5 label draws x 3 seeds, error bars "
        "1.96 x SEM over those 15 replicates. Arms with no row at a given U are omitted rather "
        "than interpolated. Rungs are the ORIGINAL roster this grid was run on -- the newer MHJ / "
        "PAIR / tom-gibbs / SycophancyEval rungs postdate it and appear in the LODO figure "
        "instead. Slice: context-based variant, synthetic (E1) extraction regime.",
    )
    return {"cells": table}


# --- figure 2: judged generic labels swapped in for judged eliciting labels ----


def fig_judged_generic_swap() -> dict:
    """rho vs the generic FRACTION of a fixed-size labeled readout pool."""
    G_ORDER = [0.0, 0.25, 0.5, 1.0]
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    src = {
        b: _rows(ER / f"judged_generic_ablation/{b}/all_arms_spearman.json", "transfer_rows")
        for b in BEHAVIORS
    }
    rungs = {b: sorted({str(r["eval_rung"]) for r in src[b]}) for b in BEHAVIORS}
    ncol = max(len(v) for v in rungs.values())
    fig, axes = plt.subplots(
        len(BEHAVIORS), ncol, figsize=(4.6 * ncol, 3.6 * len(BEHAVIORS)), squeeze=False
    )
    table: list[dict] = []
    for i, b in enumerate(BEHAVIORS):
        cells: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
        for r in src[b]:
            cells[(str(r["eval_rung"]), str(r["arm"]), float(r["g_generic"]))].append(r)
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(rungs[b]):
                ax.axis("off")
                continue
            rung = rungs[b][j]
            for arm in ARM_ORDER:
                xs, mus, hws = [], [], []
                for g in G_ORDER:
                    got = _agg(cells.get((rung, arm, g), []))
                    if got is None:
                        continue
                    xs.append(g)
                    mus.append(got[0])
                    hws.append(got[1])
                    table.append(
                        {
                            "figure": "judged_generic_swap",
                            "behavior": b,
                            "eval_rung": rung,
                            "arm": arm,
                            "g_generic": g,
                            "rho_mean": got[0],
                            "ci_halfwidth": got[1],
                            "n_replicates": len(cells[(rung, arm, g)]),
                        }
                    )
                if not xs:
                    continue
                label, color = ARM_STYLE[arm]
                _errorbar(
                    ax, xs, mus, hws, color=color, marker="o", ms=4, lw=1.6, capsize=2, label=label
                )
            ax.axhline(0.0, color="#666666", lw=0.8)
            ax.set_xticks(G_ORDER)
            ax.set_xticklabels(
                ["0\n(all eliciting)", "0.25", "0.5", "1.0\n(all generic)"], fontsize=8
            )
            ax.set_title(f"{b} — {RUNG_NAME.get(rung, rung)}", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel("Spearman rho")

    fig.supxlabel("generic fraction of the 1,500 judged labels", fontsize=9.5, y=0.10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3,
        frameon=False, fontsize=8.5,
    )
    fig.suptitle(
        "Result 5b: replacing judged ELICITING labels with judged GENERIC labels, budget held fixed",
        x=0.006,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.18, hspace=0.5, wspace=0.22)
    _finish(
        fig,
        "r5_judged_generic_swap",
        "Composition at FIXED budget, never addition: the labeled readout pool is 1,500 rows "
        "throughout, and x is the fraction of those rows drawn from judged random WildChat rather "
        "than the behavior's eliciting corpus, so a change along x cannot be confounded with "
        "quantity. Points are the mean frozen-layer Spearman rho over the source's replicates at "
        "that cell; error bars 1.96 x SEM (a single-replicate cell is drawn with a zero-width "
        "bar, which means no replicate spread was estimable, not that the estimate is exact). "
        "The label-free projection arms (persona vector on context / mapped / real answer) do not "
        "consume labels, so their curves are expected to be flat in x -- they are drawn as the "
        "reference the label-consuming arms move against.",
    )
    return {"cells": table}


# --- figure 3: f_U x f_L composition factorial ---------------------------------


def fig_fu_fl_factorial() -> dict:
    """rho vs labeled budget, one line per (f_U, f_L) composition cell.

    MEASURED COVERAGE, not assumed: the compose family was only ever run for
    EVIL (sycophancy and hallucination have zero compose rows), and the
    (f_U=0, f_L=1) corner was never run at all, so the realized design is three
    of the four corners at L in {250, 2500, 8000} -- minus (f_U=0.5, f_L=0) at
    L=8000, which is also absent. The figure plots exactly what exists and the
    caption reports the realized cell list rather than the intended one.
    """
    combos: list[tuple[float, float]] = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)]
    combo_label = {
        (0.0, 0.0): "generic map, generic labels",
        (0.5, 0.0): "half-eliciting map, generic labels",
        (0.5, 1.0): "half-eliciting map, eliciting labels",
    }
    combo_color = {(0.0, 0.0): "#8EB4D8", (0.5, 0.0): "#E0A458", (0.5, 1.0): "#7BA05B"}

    have: dict[str, list[dict]] = {}
    for b in BEHAVIORS:
        rows = [
            r
            for r in _rows(ER / b / "arm_results/all_arms_spearman.json", "arm_rows")
            if str(r.get("u_rung_label", "")).startswith("compose")
            and all(str(r.get(k)) == v for k, v in SLICE.items())
        ]
        if rows:
            have[b] = rows
    if not have:
        raise SystemExit("no composition cells in any behavior")

    # Arms the writeup's method families actually name, in reporting order.
    want = [
        "arm1_ctx_e1",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm11_oracle_proj",
        "arm12_oracle_reg",
    ]
    table: list[dict] = []
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    nrow = len(have)
    arms_by_b = {b: [a for a in want if any(r["arm"] == a for r in rs)] for b, rs in have.items()}
    ncol = max(len(v) for v in arms_by_b.values())
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow), squeeze=False, sharey="row"
    )
    for i, (b, rows) in enumerate(have.items()):
        cells: dict[tuple[str, float, float, int], list[dict]] = defaultdict(list)
        for r in rows:
            cells[(str(r["arm"]), float(r["f_u"]), float(r["f_l"]), int(r["budget_l"]))].append(r)
        budgets = sorted({k[3] for k in cells})
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(arms_by_b[b]):
                ax.axis("off")
                continue
            arm = arms_by_b[b][j]
            for combo in combos:
                xs, mus, hws = [], [], []
                for li, L in enumerate(budgets):
                    got = _agg(cells.get((arm, *combo, L), []))
                    if got is None:
                        continue
                    xs.append(li)
                    mus.append(got[0])
                    hws.append(got[1])
                    table.append(
                        {
                            "figure": "fu_fl_factorial",
                            "behavior": b,
                            "arm": arm,
                            "f_u": combo[0],
                            "f_l": combo[1],
                            "budget_l": L,
                            "rho_mean": got[0],
                            "ci_halfwidth": got[1],
                            "n_replicates": len(cells[(arm, *combo, L)]),
                        }
                    )
                if not xs:
                    continue
                _errorbar(
                    ax,
                    xs,
                    mus,
                    hws,
                    color=combo_color[combo],
                    marker="o",
                    ms=4,
                    lw=1.7,
                    capsize=2,
                    label=combo_label[combo],
                )
            ax.axhline(0.0, color="#666666", lw=0.8)
            ax.set_xticks(range(len(budgets)))
            ax.set_xticklabels([f"{L:,}" for L in budgets], fontsize=8)
            ax.set_title(ARM_STYLE[arm][0], loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{b}\nSpearman rho")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.supxlabel("labeled readout budget L", fontsize=9.5, y=0.19)
    fig.suptitle(
        "Result 5c: the two composition channels crossed — eliciting data in the MAP pool vs in "
        "the LABEL pool",
        x=0.006,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.80, bottom=0.28, wspace=0.16, hspace=0.42)
    realized = sorted({(c["f_u"], c["f_l"], c["budget_l"]) for c in table})
    _finish(
        fig,
        "r5_fu_fl_factorial",
        "f_U is the fraction of the UNLABELED map pool that is trait-eliciting rather than generic "
        "WildChat; f_L is the same fraction for the LABELED readout pool. Both are composition at "
        "FIXED size -- eliciting rows REPLACE generic rows -- so neither axis is confounded with "
        "quantity. Cells come from the main grid's compose family at an unlabeled pool of 5,000, "
        f"evaluated on the in-distribution rung. Realized coverage: {len(realized)} "
        f"(f_U, f_L, L) cells for {', '.join(have)} ONLY -- the compose family was never run for "
        f"{', '.join(b for b in BEHAVIORS if b not in have) or 'no other behavior'}, the "
        "(f_U=0, f_L=1) corner was never run at all, and (f_U=0.5, f_L=0) has no L=8,000 cell, so "
        "a missing point is missing data rather than a null. Points are the mean over the cell's "
        "replicates, error bars 1.96 x SEM.",
    )
    return {"cells": table}


# --- figure 4: leave-one-dataset-out -------------------------------------------


def fig_lodo_by_dataset() -> dict:
    """With dataset D held out of the fit, rho on D vs on its siblings vs generic.

    Source is the SAME pinned P-A/P-B round the Result 2/3 figures read
    (`r2v2_fits` at LODO_FITS_COMMIT), not the widegrid: the widegrid has no
    sycophancy leg at all, and reading two different rounds across panels would
    make the three behaviors incomparable.
    """
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(6.2 * len(BEHAVIORS), 5.0), squeeze=False)
    table: list[dict] = []
    # Three evaluation classes per held-out-dataset fit. `heldin:<X>` is the 20%
    # held-in slice of a sibling dataset the fit DID train on -- that is exactly
    # the "other eliciting datasets" comparison, so it must not be discarded.
    # `pvsynth` is the synthetic diagnostic grid and belongs to neither class.
    kinds = ["the held-out dataset", "other eliciting datasets (held in)", "generic chat"]
    kind_color = {kinds[0]: "#C4553B", kinds[1]: "#E0A458", kinds[2]: "#8EB4D8"}
    for i, b in enumerate(BEHAVIORS):
        ax = axes[0][i]
        rows = [r for r in _lodo_rows(b) if str(r.get("protocol")) == "P-B"]
        # Label-consuming arms only: a label-free projection is invariant to which
        # dataset left the READOUT fit, so plotting it here would imply an effect
        # the design cannot produce.
        arms = [
            a for a in ("arm4_ridge_ctx", "arm7_map_ridge_pred") if any(r["arm"] == a for r in rows)
        ]
        buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        n_holdouts: set[str] = set()
        for r in rows:
            fit = str(r.get("fit", ""))
            if not fit.startswith("P-B-holdout-") or r["arm"] not in arms:
                continue
            if r.get("rho_frozen") is None:
                continue
            held = fit[len("P-B-holdout-") :]
            n_holdouts.add(held)
            rung = str(r["eval_rung"])
            if rung == held:
                kind = kinds[0]
            elif rung == "wildchat_rung":
                kind = kinds[2]
            elif rung.startswith("heldin:"):
                kind = kinds[1]
            else:
                # pvsynth, or a sibling's FULL (not held-in) slice: neither class.
                continue
            buckets[(r["arm"], kind)].append(float(r["rho_frozen"]))
        width = 0.8 / len(kinds)
        for ki, kind in enumerate(kinds):
            xs, mus, hws = [], [], []
            for ai, arm in enumerate(arms):
                v = np.array(buckets.get((arm, kind), []), dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                hw = 1.96 * v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
                xs.append(ai + (ki - (len(kinds) - 1) / 2) * width)
                mus.append(float(v.mean()))
                hws.append(float(hw))
                table.append(
                    {
                        "figure": "lodo_by_dataset",
                        "behavior": b,
                        "arm": arm,
                        "eval_class": kind,
                        "rho_mean": float(v.mean()),
                        "ci_halfwidth": float(hw),
                        "n_cells": int(v.size),
                        "n_holdout_fits": len(n_holdouts),
                    }
                )
            if not xs:
                continue
            lo = np.maximum(0.0, np.asarray(hws, dtype=float))
            ax.bar(
                xs,
                mus,
                width=width * 0.92,
                yerr=[lo, lo],
                capsize=2,
                label=kind,
                color=kind_color[kind],
                edgecolor="#333333",
                linewidth=0.4,
            )
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_STYLE[a][0].replace(" on ", "\non ") for a in arms], fontsize=8)
        ax.set_title(f"{b} — {len(n_holdouts)} held-out datasets", loc="left", fontsize=10)
        if i == 0:
            ax.set_ylabel("Spearman rho")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Result 5d: leave-one-dataset-out — what does the readout lose on the dataset it never saw?",
        x=0.006,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.24, wspace=0.18)
    _finish(
        fig,
        "r5_lodo_by_dataset",
        "Under the leave-one-dataset-out protocol the readout trains on an 80% group-level slice "
        "of every eliciting dataset EXCEPT one, which is held out whole; there is one fit per "
        "held-out dataset (evil 5, sycophancy 6, hallucination 2). Each bar pools that fit's "
        "evaluation cells into three classes: the dataset it never saw, the 20% held-in slices of "
        "the sibling eliciting datasets it did train on, and random WildChat. The synthetic "
        "persona-vector grid belongs to none of the three and is excluded. ONLY the "
        "label-consuming arms are drawn -- the persona-vector projections do not train a readout, "
        "so which dataset left the fit cannot move them, and showing them would imply an effect "
        "the design cannot produce. Bars are the mean over the pooled cells with a 1.96 x SEM "
        "half-width; per-bar cell counts are in the JSON sidecar.",
    )
    return {"cells": table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        choices=("ladder", "swap", "factorial", "lodo"),
        help="render a single figure instead of all four",
    )
    args = ap.parse_args()
    jobs = {
        "ladder": fig_generic_map_ladder,
        "swap": fig_judged_generic_swap,
        "factorial": fig_fu_fl_factorial,
        "lodo": fig_lodo_by_dataset,
    }
    todo = [args.only] if args.only else list(jobs)
    out: dict[str, dict] = {}
    for name in todo:
        out[name] = jobs[name]()
        n = len(out[name]["cells"])
        if n == 0:
            raise SystemExit(
                f"{name}: produced zero plotted cells -- refusing to ship an empty figure"
            )
        print(f"  {name}: {n} plotted cells")
    OUT_NUM.mkdir(parents=True, exist_ok=True)
    sidecar = OUT_NUM / "result5_points.json"
    sidecar.write_text(json.dumps({"figures": out}, indent=1))
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
