"""Publishable re-cut figures for #1739 (all three behavior lanes).

Pure aggregation + rendering over committed artifacts (see
`issue1739_recut_common` for the artifact map). No fits, no GPU, no network.

Writes figures to figures/issue_1739/recuts/ and every plotted number to
eval_results/issue_1739/recuts/recut_numbers.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# load_dotenv() FIRST: on the shared VM it setdefaults the BLAS/OMP thread
# caps, and numpy/torch freeze their thread pools at import, so any heavy
# import above this line would escape the cap.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import json  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import set_paper_style  # noqa: E402
from issue1739_recut_common import (  # noqa: E402
    ARM_LABEL,
    ARM_ORDER,
    BEHAVIORS,
    FAMILY_COLOR,
    FAMILY_LABEL,
    FIGDIR,
    LMAX,
    NUMDIR,
    OP,
    OP_RV,
    RUNG_LABEL,
    RUNGS,
    agg_rho,
    arm_color,
    ceiling_sb,
    load_cells,
    load_main,
    load_wcrung_preds,
    load_wide_ood,
    load_wide_wcrung,
    match,
    nonneg_err,
    paired_delta_bootstrap,
    replicate_delta,
    WT,
)

set_paper_style()
FIGDIR.mkdir(parents=True, exist_ok=True)
NUMDIR.mkdir(parents=True, exist_ok=True)

NUMBERS: dict = {}
GAPS: list[str] = []


def save(fig, name: str) -> None:
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[recut] wrote {name}.png")


# Cache the heavier loads.
MAIN = {b: load_main(b) for b in BEHAVIORS}
WOOD = {b: load_wide_ood(b) for b in BEHAVIORS}
WC = {b: load_wide_wcrung(b) for b in BEHAVIORS}
WCPREDS = {
    (b, v): load_wcrung_preds(b, v) for b in BEHAVIORS for v in ("context_end", "prefix_end")
}
CEIL = {b: ceiling_sb(b) for b in BEHAVIORS}


def wcrung_rows(b: str) -> list[dict]:
    return WC[b]["transfer_rows"]


def rung_rows(b: str, rung: str) -> list[dict]:
    """All replicate rows for one non-WildChat rung, from the richest source.

    The TRAIN rung is read from the main lane's ``arm_rows`` (all 16 arms, the
    full u x L grid); the OOD rungs come from ``wide_ood`` (9 arms, arms 7/8/12
    added). Reading train from ``wide_ood`` would silently under-cover arms
    7/8/12 there, which is a sourcing artifact, not a coverage gap.
    """
    if rung == "train":
        return MAIN[b]["arm_rows"]
    return [r for r in WOOD[b] if r.get("eval_rung") == rung]


def resolve_slice(b: str, rung: str, arms: list[str]) -> dict | None:
    """Richest (u_rung_label, budget_l) slice where EVERY arm in ``arms`` runs.

    Prefers the operating slice (u=full, L=max). Falls back to the largest
    (u, L) at which the arms CO-OCCUR, so a matched comparison survives a
    partial grid instead of blanking the panel. Returns ``None`` when no
    slice covers every arm. Matched-target by construction: one slice for
    every arm in the panel.
    """
    rows = rung_rows(b, rung)
    u_pref = ["full", "5000", "250"]
    avail: dict[tuple[str, int], set[str]] = {}
    for r in rows:
        if not match(r, regime=OP["regime"], variant=OP["variant"]):
            continue
        if r.get("rho_frozen") is None:
            continue
        avail.setdefault((str(r["u_rung_label"]), int(r["budget_l"])), set()).add(r["arm"])
    cands = [k for k, got in avail.items() if set(arms) <= got]
    if not cands:
        return None
    # Rank: preferred u first, then the largest label budget.
    cands.sort(key=lambda k: (u_pref.index(k[0]) if k[0] in u_pref else 99, -k[1]))
    u, L = cands[0]
    return dict(
        u_rung_label=u,
        budget_l=L,
        is_operating_slice=bool(u == "full" and L == LMAX[b]),
        budgets_available=sorted(
            {bl for (uu, bl) in avail if uu == u and set(arms) <= avail[(uu, bl)]}
        ),
    )


# ==========================================================================
# FIG 1 - map minus direct-context DELTA (the headline "does the map add
#         value beyond the context" read), per behavior x setting.
# ==========================================================================
def fig_delta_map_minus_direct() -> None:
    pair = ("arm7_map_ridge_pred", "arm4_ridge_ctx")
    out: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharey=False)

    for ax, b in zip(axes, BEHAVIORS):
        labels, deltas, los, his, kinds = [], [], [], [], []
        for rung in RUNGS[b]:
            if rung == "wildchat_rung":
                sc, dv, _ = WCPREDS[(b, "context_end")]
                if pair[0] not in sc or pair[1] not in sc:
                    GAPS.append(f"fig1 {b}/{rung}: arm missing from WildChat preds")
                    continue
                res = paired_delta_bootstrap(sc[pair[0]], sc[pair[1]], dv, seed=17)
                kind = "boot"
                slice_note = ""
            else:
                sl = resolve_slice(b, rung, list(pair))
                if sl is None:
                    GAPS.append(
                        f"fig1 {b}/{rung}: {pair[0]} and {pair[1]} share NO "
                        f"(u, L) slice on this rung - no matched comparison exists"
                    )
                    continue
                res = replicate_delta(
                    rung_rows(b, rung),
                    pair[0],
                    pair[1],
                    budget_l=sl["budget_l"],
                    u_rung_label=sl["u_rung_label"],
                    **OP_RV,
                )
                if res["delta"] is None:
                    GAPS.append(f"fig1 {b}/{rung}: no matched replicates at {sl}")
                    continue
                kind = "rep"
                res["slice"] = sl
                slice_note = (
                    ""
                    if sl["is_operating_slice"]
                    else f"\nU={sl['u_rung_label']}, L={sl['budget_l']:,}"
                )
                if not sl["is_operating_slice"]:
                    GAPS.append(
                        f"fig1 {b}/{rung}: arms 7/8/12 absent at the operating "
                        f"slice (U=full, L={LMAX[b]:,}); the matched comparison "
                        f"falls back to U={sl['u_rung_label']}, L={sl['budget_l']:,}"
                    )
            labels.append(RUNG_LABEL[(b, rung)] + slice_note)
            deltas.append(res["delta"])
            los.append(res["ci"][0])
            his.append(res["ci"][1])
            kinds.append(kind)
            out.setdefault(b, {})[rung] = dict(res, estimator=kind)

        if not deltas:
            ax.text(0.5, 0.5, "no cell covered", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(b, fontsize=11)
            continue

        y = np.arange(len(deltas))
        colors = [FAMILY_COLOR["map"] if d > 0 else FAMILY_COLOR["context"] for d in deltas]
        ax.barh(
            y,
            deltas,
            xerr=nonneg_err(deltas, los, his),
            color=colors,
            height=0.6,
            error_kw=dict(lw=1.1, capsize=3),
        )
        for i, (d, lo, hi, k) in enumerate(zip(deltas, los, his, kinds)):
            sig = "*" if (lo > 0 or hi < 0) else ""
            ax.text(
                d + (0.012 if d >= 0 else -0.012),
                i,
                f"{d:+.3f}{sig}",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.9)
        ax.set_title(b, fontsize=11)
        ax.set_xlabel(r"$\Delta\rho$  (map$\to$ridge  $-$  direct ridge)")
        ax.margins(x=0.30)

    axes[0].set_ylabel("evaluation setting")
    handles = [
        Patch(facecolor=FAMILY_COLOR["map"], label="map-side readout ahead"),
        Patch(facecolor=FAMILY_COLOR["context"], label="direct context readout ahead"),
        Line2D([], [], color="k", lw=1.1, label="95% CI; * = excludes 0"),
    ]
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )
    fig.suptitle(
        "Does mapping into answer space help beyond reading the context?  "
        r"Paired $\Delta\rho$ = $\rho$(map$\to$ridge on predicted answers) $-$ "
        r"$\rho$(direct ridge on context)",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "delta_map_minus_direct")
    NUMBERS["fig1_delta_map_minus_direct"] = dict(
        pair=pair,
        note=(
            "WildChat rung: paired bootstrap over contexts (2000 draws, one "
            "resampled context index set applied to BOTH arms). Train/OOD "
            "rungs: paired delta across matched (seed, draw) replicates at "
            "L=max, U=full, E1, context_end; CI = normal-approx 95% on the "
            "replicate mean. Both arms consume the same labels and the same "
            "context activations and are scored against the SAME judged DV "
            "on the SAME contexts."
        ),
        results=out,
    )


# ==========================================================================
# FIG 2 - sample efficiency: rho vs number of judged labels (L ladder).
# ==========================================================================
# Per-arm line styling for the label ladder. Colour stays family-keyed; the
# marker + fill separate the two map-family arms that would otherwise collide.
_LADDER_STYLE = {
    "arm7_map_ridge_pred": dict(
        color=FAMILY_COLOR["map"], ls="-", marker="o", ms=5, mfc=FAMILY_COLOR["map"]
    ),
    "arm6_map_proj_e1": dict(color=FAMILY_COLOR["map"], ls="--", marker="^", ms=6, mfc="white"),
    "arm4_ridge_ctx": dict(
        color=FAMILY_COLOR["context"], ls="-", marker="s", ms=4.5, mfc=FAMILY_COLOR["context"]
    ),
}


def fig_sample_efficiency() -> None:
    arms = ["arm7_map_ridge_pred", "arm4_ridge_ctx", "arm6_map_proj_e1"]
    out: dict = {}
    ncol = max(len(RUNGS[b]) - 1 for b in BEHAVIORS)  # WildChat has no L ladder
    fig, axes = plt.subplots(
        len(BEHAVIORS), ncol, figsize=(3.5 * ncol, 3.1 * len(BEHAVIORS)), squeeze=False
    )

    for row, b in enumerate(BEHAVIORS):
        ladder_rungs = [r for r in RUNGS[b] if r != "wildchat_rung"]
        for col in range(ncol):
            ax = axes[row][col]
            if col >= len(ladder_rungs):
                ax.axis("off")
                continue
            rung = ladder_rungs[col]
            rows = rung_rows(b, rung)
            # ONE u slice for every arm in the panel, so the ladder is a
            # matched comparison rather than a mix of unlabeled-pool sizes.
            sl = resolve_slice(b, rung, arms)
            if sl is None:
                ax.text(
                    0.5,
                    0.5,
                    "arms share no\ncommon (U, L) slice",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_title(f"{b} - {RUNG_LABEL[(b, rung)]}".replace("\n", " "), fontsize=8.5)
                GAPS.append(f"fig2 {b}/{rung}: arms share no common (U, L) slice")
                continue
            u_lab = sl["u_rung_label"]
            budgets = sl["budgets_available"]
            if not sl["is_operating_slice"]:
                GAPS.append(
                    f"fig2 {b}/{rung}: ladder shown at U={u_lab} "
                    f"(arms 7/8/12 absent at U=full on this rung)"
                )
            plotted = False
            for arm in arms:
                xs, ms, ss = [], [], []
                for L in budgets:
                    a = agg_rho(rows, arm, budget_l=L, u_rung_label=u_lab, **OP_RV)
                    if a["mean"] is None:
                        continue
                    xs.append(L)
                    ms.append(a["mean"])
                    ss.append(a["sd"])
                if not xs:
                    GAPS.append(f"fig2 {b}/{rung}: {arm} absent across the L ladder")
                    continue
                plotted = True
                # Within a panel two arms share the map family colour, so the
                # SECOND one is distinguished by marker + tint as well as
                # dash: colour still encodes family, marker encodes the arm.
                style = _LADDER_STYLE[arm]
                ax.errorbar(
                    xs,
                    ms,
                    yerr=np.maximum(0.0, ss),
                    marker=style["marker"],
                    ms=style["ms"],
                    mfc=style["mfc"],
                    lw=1.5,
                    capsize=2.5,
                    color=style["color"],
                    ls=style["ls"],
                    label=ARM_LABEL[arm],
                )
                out.setdefault(b, {}).setdefault(rung, {})[arm] = dict(
                    budgets=xs, mean=ms, sd=ss, u_rung_label=u_lab
                )
            if not plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.axhline(0, color="k", lw=0.7, alpha=0.6)
            ax.set_xscale("log")
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{x:,}" for x in budgets], fontsize=7.5)
            title = f"{b} - {RUNG_LABEL[(b, rung)]}".replace("\n", " ")
            if not sl["is_operating_slice"]:
                title += f"\n[U={u_lab}]"
            ax.set_title(title, fontsize=8.5)
            if col == 0:
                ax.set_ylabel(r"Spearman $\rho$")
            ax.set_xlabel("judged labels L")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(
        "Sample efficiency: rank correlation vs number of judged labels "
        "(U = 18,793 unlabeled pairs, E1 PV, context end state; "
        "mean +/- SD over 3 seeds x 5 draws)",
        y=1.005,
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, "sample_efficiency_labels")
    NUMBERS["fig2_sample_efficiency"] = dict(
        note=(
            "Solid = labeled readouts (both consume L labels); dashed = the "
            "label-free map projection, which still moves with L because its "
            "layer is frozen from label-bearing train cells. WildChat rung "
            "carries no L ladder (single max-budget replicate) and is omitted."
        ),
        results=out,
    )


# ==========================================================================
# FIG 3 - sign consistency across evaluation settings, per method.
# ==========================================================================
def fig_sign_consistency() -> None:
    out: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    for ax, b in zip(axes, BEHAVIORS):
        rungs = RUNGS[b]
        arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in WOOD[b])]
        grid = np.full((len(arms), len(rungs)), np.nan)
        for i, arm in enumerate(arms):
            for j, rung in enumerate(rungs):
                if rung == "wildchat_rung":
                    rows = [r for r in wcrung_rows(b) if match(r, variant="context_end")]
                    a = agg_rho(rows, arm, variant="context_end")
                else:
                    # Operating slice only: a sign read must not silently mix
                    # unlabeled-pool sizes across cells of one row.
                    a = agg_rho(rung_rows(b, rung), arm, budget_l=LMAX[b], **OP)
                if a["mean"] is not None:
                    grid[i, j] = a["mean"]

        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        for i in range(len(arms)):
            for j in range(len(rungs)):
                v = grid[i, j]
                ax.text(
                    j,
                    i,
                    "n/a" if np.isnan(v) else f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="black" if (np.isnan(v) or abs(v) < 0.32) else "white",
                )
            fin = grid[i][np.isfinite(grid[i])]
            # CONSTRUCT-MATCHED verdict. Hallucination's own rungs score a 0-1
            # fabrication rate while its WildChat column is the 0-100 trait
            # rubric, so a sign change ACROSS that boundary is a construct
            # change, not evidence the direction is unstable. The verdict is
            # therefore taken within the majority construct; the cross-construct
            # column is reported separately.
            matched_j = [
                j
                for j, rung in enumerate(rungs)
                if not (b == "hallucination" and rung == "wildchat_rung")
            ]
            fin_m = np.array(
                [grid[i, j] for j in matched_j if np.isfinite(grid[i, j])], dtype=float
            )
            consistent = fin_m.size >= 2 and (np.all(fin_m > 0) or np.all(fin_m < 0))
            all_consistent = fin.size >= 2 and (np.all(fin > 0) or np.all(fin < 0))
            out.setdefault(b, {})[arms[i]] = dict(
                per_rung={
                    rungs[j]: (None if np.isnan(grid[i, j]) else float(grid[i, j]))
                    for j in range(len(rungs))
                },
                n_settings_covered=int(fin.size),
                n_settings_construct_matched=int(fin_m.size),
                sign_consistent_construct_matched=(bool(consistent) if fin_m.size >= 2 else None),
                sign_consistent_all_settings=(bool(all_consistent) if fin.size >= 2 else None),
            )
            if fin_m.size >= 2:
                ax.text(
                    len(rungs) - 0.35,
                    i,
                    "consistent" if consistent else "FLIPS",
                    fontsize=6.6,
                    va="center",
                    ha="left",
                    color=("#2b7a3d" if consistent else "#b02020"),
                    fontweight="bold",
                )
        # Mark the cross-construct column so no reader treats it as comparable.
        if b == "hallucination":
            j = rungs.index("wildchat_rung")
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, -0.5),
                    1.0,
                    len(arms),
                    fill=False,
                    edgecolor="#7a3d9e",
                    lw=1.8,
                    ls="--",
                    zorder=5,
                )
            )
            ax.text(
                j,
                -0.72,
                "different DV construct\n(trait rubric, not fabrication rate)",
                fontsize=6.2,
                ha="center",
                va="bottom",
                color="#7a3d9e",
            )
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels(
            [RUNG_LABEL[(b, r)].replace("\n", " ") for r in rungs],
            rotation=22,
            ha="right",
            fontsize=7.2,
        )
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=7.2)
        ax.set_title(b, fontsize=10.5)
        ax.set_xlim(-0.5, len(rungs) + 1.4)
    fig.colorbar(im, ax=axes, shrink=0.72, label=r"Spearman $\rho$", pad=0.015)
    fig.suptitle(
        "Does each method keep the SIGN of its correlation across evaluation settings?  "
        "(operating slice: L=max, U=full, E1 PV, context end state)",
        y=1.02,
        fontsize=11,
    )
    save(fig, "sign_consistency_across_settings")
    NUMBERS["fig3_sign_consistency"] = dict(
        note=(
            "Cell value = mean rho over replicates at the operating slice. "
            "The printed verdict is CONSTRUCT-MATCHED: it requires >=2 covered "
            "settings all strictly one sign, computed WITHIN one DV construct. "
            "Hallucination's own rungs score a 0-1 fabrication rate while its "
            "WildChat column is the 0-100 trait rubric, so that column is "
            "excluded from its verdict and outlined instead - a sign change "
            "across a construct boundary is not evidence of an unstable "
            "direction. Both verdicts are recorded per arm "
            "(sign_consistent_construct_matched vs sign_consistent_all_settings). "
            "Sycophancy has only 2 construct-matched settings, so its verdicts "
            "rest on fewer rungs than evil's."
        ),
        results=out,
    )


# ==========================================================================
# FIG 4 - the STANDING mapping baselines: identity+learned-bias and
#         kNN-retrieval, for the one round that FITS a map (evil leg 2).
# ==========================================================================
def fig_mapping_baselines() -> None:
    with open(WT / "bareq_map/evil/all_arms_spearman.json") as f:
        mb = json.load(f)["meta"]["mapping_baselines"]
    leg2 = mb["leg2"]
    if not leg2.get("applicable"):
        GAPS.append("fig4: evil leg 2 mapping baselines not applicable")
        return

    folds = leg2["per_fold"]
    layers = [pl["layer_idx"] for pl in folds[0]["per_layer"]]
    r2_map = np.array([[pl["r2_map"] for pl in f["per_layer"]] for f in folds])
    r2_ib = np.array([[pl["r2_identity_bias"] for pl in f["per_layer"]] for f in folds])

    def knn(field: str, metric: str, k: str) -> np.ndarray:
        return np.array([[pl["knn"][metric][field][k] for pl in f["per_layer"]] for f in folds])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    ax = axes[0]
    for vals, lab, col in (
        (r2_map, "fitted map", FAMILY_COLOR["map"]),
        (r2_ib, "identity + learned bias", FAMILY_COLOR["context"]),
    ):
        m, s = vals.mean(0), vals.std(0, ddof=1)
        ax.plot(layers, m, lw=1.6, color=col, label=lab)
        ax.fill_between(layers, m - s, m + s, color=col, alpha=0.18, lw=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"held-out $R^2$")
    ax.set_title(r"(a) Reconstruction $R^2$ per layer", fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    width = 0.36
    x = np.arange(2)
    for off, (metric, hatch) in enumerate((("euclidean", ""), ("cosine", "//"))):
        best = []
        chance = []
        for k in ("1", "5"):
            acc = knn("acc_at_k", metric, k).mean(0)
            best.append(float(acc.max()))
            chance.append(float(knn("chance_at_k", metric, k).mean(0).mean()))
        ax.bar(
            x + (off - 0.5) * width,
            best,
            width,
            color=FAMILY_COLOR["map"],
            hatch=hatch,
            edgecolor="white",
            label=f"fitted map ({metric})",
        )
    for j, c in enumerate(chance):
        ax.hlines(c, x[j] - 0.55, x[j] + 0.55, color="k", lw=1.5, ls=":")
        ax.text(x[j], c, f"  chance {c:.4f}", fontsize=7, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["acc@1", "acc@5"])
    ax.set_yscale("log")
    ax.set_ylabel("retrieval accuracy (log scale)")
    ax.set_title("(b) kNN retrieval of the true target\n(best layer, pooled folds)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    labels, vals, cols = [], [], []
    for b in BEHAVIORS:
        for rung, rows in (
            ("wildchat_rung", [r for r in wcrung_rows(b) if match(r, variant="context_end")]),
        ):
            a = agg_rho(rows, "arm3_identity_bias", variant="context_end")
            if a["mean"] is None:
                continue
            labels.append(f"{b}\n(WildChat)")
            vals.append(a["mean"])
            cols.append(FAMILY_COLOR["context"])
    ax.bar(range(len(vals)), vals, color=cols, width=0.55)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title(
        "(c) The identity+bias SCORING arm\n(a different object from (a)/(b))", fontsize=9.5
    )

    fig.suptitle(
        "Standing mapping baselines for the fitted bare-rep -> answer map "
        "(evil leg 2; 5 by-query folds, no query straddling a fold)",
        y=1.04,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "mapping_baselines_identity_knn")

    peak_layer = int(np.argmax(r2_map.mean(0)))
    NUMBERS["fig4_mapping_baselines"] = dict(
        note=(
            "Panels (a)/(b) are the STANDING identity+learned-bias and "
            "kNN-retrieval reads on the FITTED bare-rep -> answer-activation "
            "map (evil leg 2 - the only round in this task that fits a map, "
            "so the standing pair binds there and nowhere else). Panel (c) is "
            "the arm3 activation->DV SCORING arm, which the artifact's own "
            "note flags as a DIFFERENT object. n_train 5,160 per fold vs "
            "d=3,584, so the fits are not under-determined."
        ),
        n_folds=len(folds),
        n_train_per_fold=folds[0]["n_train"],
        d_in=folds[0]["d_in"],
        r2_map_peak=dict(layer=layers[peak_layer], value=float(r2_map.mean(0)[peak_layer])),
        r2_identity_bias_min=float(r2_ib.mean(0).min()),
        r2_identity_bias_max=float(r2_ib.mean(0).max()),
        knn={
            m: {
                k: dict(
                    best_layer_acc=float(knn("acc_at_k", m, k).mean(0).max()),
                    chance=float(knn("chance_at_k", m, k).mean(0).mean()),
                )
                for k in ("1", "5")
            }
            for m in ("euclidean", "cosine")
        },
        fold_semantics=leg2["fold_semantics"],
    )


# ==========================================================================
# FIG 5 - real arms beside their by-construction nulls.
# ==========================================================================
def fig_nulls_and_controls() -> None:
    out: dict = {}
    # Row 1: WildChat rung (arms 9/14 were not run there).
    wc_arms = [
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm4_ridge_ctx",
        "arm1_ctx_e1",
        "arm13_shuffled_map",
    ]
    # Row 2: train rung, where BOTH nonsense-map controls exist.
    tr_arms = [
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm9_pretrain_ft",
        "arm4_ridge_ctx",
        "arm13_shuffled_map",
        "arm14_shuffled_pt",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), sharex=False)

    for col, b in enumerate(BEHAVIORS):
        # ---- row 1: WildChat rung -------------------------------------
        ax = axes[0][col]
        rows = [r for r in wcrung_rows(b) if match(r, variant="context_end")]
        xs, ms, los, his, cols_, labs = [], [], [], [], [], []
        for arm in wc_arms:
            a = agg_rho(rows, arm, variant="context_end")
            if a["mean"] is None:
                GAPS.append(f"fig5 {b}: {arm} absent on WildChat rung")
                continue
            xs.append(len(xs))
            ms.append(a["mean"])
            lo, hi = a["ci"] if a["ci"] else (a["mean"], a["mean"])
            los.append(lo)
            his.append(hi)
            cols_.append(arm_color(arm))
            labs.append(ARM_LABEL[arm])
            out.setdefault(b, {}).setdefault("wildchat_rung", {})[arm] = dict(
                rho=a["mean"], ci=a["ci"]
            )
        ax.bar(
            xs,
            ms,
            yerr=nonneg_err(ms, los, his),
            color=cols_,
            width=0.62,
            error_kw=dict(lw=1.0, capsize=2.5),
        )
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=34, ha="right", fontsize=7)
        ax.set_title(f"{b} - random WildChat rung", fontsize=9.5)
        if col == 0:
            ax.set_ylabel(r"Spearman $\rho$")

        # ---- row 2: train rung, both controls -------------------------
        ax = axes[1][col]
        trows = MAIN[b]["arm_rows"]
        xs, ms, ss, cols_, labs = [], [], [], [], []
        vals: dict[str, float] = {}
        for arm in tr_arms:
            a = agg_rho(trows, arm, budget_l=LMAX[b], **OP)
            if a["mean"] is None:
                GAPS.append(f"fig5 {b}: {arm} absent on train rung")
                continue
            xs.append(len(xs))
            ms.append(a["mean"])
            ss.append(a["sd"])
            cols_.append(arm_color(arm))
            labs.append(ARM_LABEL[arm])
            vals[arm] = a["mean"]
            out.setdefault(b, {}).setdefault("train", {})[arm] = dict(
                rho=a["mean"], sd=a["sd"], n_replicates=a["n"]
            )
        ax.bar(
            xs,
            ms,
            yerr=np.maximum(0.0, ss),
            color=cols_,
            width=0.62,
            error_kw=dict(lw=1.0, capsize=2.5),
        )
        # The load-bearing read: is the nonsense-pretrain control separable
        # from the real map arms it is meant to falsify?
        if {"arm14_shuffled_pt", "arm9_pretrain_ft"} <= vals.keys():
            gap = abs(vals["arm9_pretrain_ft"] - vals["arm14_shuffled_pt"])
            sd_ref = out[b]["train"]["arm9_pretrain_ft"]["sd"]
            ax.annotate(
                f"real vs shuffled pretrain\ndiffer by {gap:.4f}\n(SD over replicates {sd_ref:.4f})",
                xy=(0.02, 0.97),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=7,
                color="#b02020",
                fontweight="bold",
            )
            out[b]["train"]["_real_minus_shuffled_pretrain"] = float(
                vals["arm9_pretrain_ft"] - vals["arm14_shuffled_pt"]
            )
        nulls = [c["max_over_arms_null"] for c in load_cells(b) if c.get("max_over_arms_null")]
        q95 = float(np.mean([n["null_max_q95"] for n in nulls]))
        ax.axhline(q95, color="#555555", ls="--", lw=1.2)
        ax.text(
            -0.45,
            q95,
            f"permutation null q95 = {q95:.3f}",
            fontsize=6.8,
            va="bottom",
            ha="left",
            color="#444444",
        )
        out[b]["train"]["_null_max_q95"] = q95
        out[b]["train"]["_null_frac_p_lt_0.05"] = float(
            np.mean([n["p_max_over_arms"] < 0.05 for n in nulls])
        )
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=34, ha="right", fontsize=7)
        ax.set_title(f"{b} - held-out train rung (L={LMAX[b]:,})", fontsize=9.5)
        if col == 0:
            ax.set_ylabel(r"Spearman $\rho$")

    handles = [
        Patch(facecolor=c, label=FAMILY_LABEL[f])
        for f, c in FAMILY_COLOR.items()
        if f in ("map", "context", "control")
    ]
    handles.append(
        Line2D([], [], color="#555555", ls="--", lw=1.2, label="max-over-arms permutation null q95")
    )
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.045),
        frameon=False,
    )
    fig.suptitle(
        "Real methods beside their by-construction null controls.  Top: the WildChat rung, where "
        "the shuffled map collapses to ~0.\nBottom: the train rung at max label budget, where the "
        "SHUFFLED-pretrain control matches the real map arms to ~1e-4.",
        y=1.02,
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, "nulls_and_controls")
    NUMBERS["fig5_nulls_and_controls"] = dict(
        note=(
            "Top row error bars = 95% bootstrap CI over contexts (WildChat, "
            "one replicate). Bottom row error bars = SD over 3 seeds x 5 draws. "
            "The WildChat rung carries NO permutation null of its own, so the "
            "q95 line is drawn only on the train row where it was computed. "
            "Arms 9/14 were not run on the WildChat rung, which is why the "
            "shuffled-PRETRAIN control can only be read on the train rung."
        ),
        headline=(
            "On the train rung at max label budget the shuffled-pretrain "
            "control is indistinguishable from the real map-pretrain arm "
            "(differences ~1e-4, far below the replicate SD), so train-rung "
            "rho cannot support any claim that the map's CONTENT carries "
            "information there. The WildChat rung is where the controls "
            "separate."
        ),
        results=out,
    )


# ==========================================================================
# FIG 6 - headroom to the oracle ceilings.
# ==========================================================================
def fig_headroom_to_oracle() -> None:
    ladders = [
        ("projection family", "arm6_map_proj_e1", "arm11_oracle_proj"),
        ("labeled-readout family", "arm7_map_ridge_pred", "arm12_oracle_reg"),
        ("labeled-readout family (real-answer fit)", "arm8_map_ridge_true", "arm12_oracle_reg"),
    ]
    out: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)

    for ax, b in zip(axes, BEHAVIORS):
        rows = [r for r in wcrung_rows(b) if match(r, variant="context_end")]
        y = 0
        yticks, ylabels = [], []
        for name, feasible, oracle in ladders:
            fa = agg_rho(rows, feasible, variant="context_end")
            oa = agg_rho(rows, oracle, variant="context_end")
            if fa["mean"] is None or oa["mean"] is None:
                GAPS.append(f"fig6 {b}: {name} incomplete on WildChat rung")
                continue
            # The feasible arm is the bar; the oracle is a MARKER, not a bar
            # behind it - otherwise a feasible arm that MATCHES or EXCEEDS its
            # oracle (negative headroom) overdraws the ceiling and hides it.
            ax.barh(y, fa["mean"], color=FAMILY_COLOR["map"], height=0.5)
            ax.plot(
                [oa["mean"]],
                [y],
                marker="D",
                ms=8,
                mfc="white",
                mec=FAMILY_COLOR["oracle"],
                mew=2.2,
                zorder=6,
                clip_on=False,
            )
            ax.vlines(
                oa["mean"],
                y - 0.30,
                y + 0.30,
                color=FAMILY_COLOR["oracle"],
                lw=2.0,
                zorder=5,
            )
            gap = oa["mean"] - fa["mean"]
            ax.annotate(
                "",
                xy=(oa["mean"], y - 0.33),
                xytext=(fa["mean"], y - 0.33),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#333333"),
            )
            ax.text(
                (fa["mean"] + oa["mean"]) / 2,
                y - 0.40,
                f"headroom {gap:+.3f}",
                fontsize=6.8,
                ha="center",
                va="bottom",
            )
            yticks.append(y)
            ylabels.append(f"{ARM_LABEL[feasible]}\nvs {ARM_LABEL[oracle]}")
            out.setdefault(b, {})[name] = dict(
                feasible_arm=feasible,
                oracle_arm=oracle,
                feasible_rho=fa["mean"],
                oracle_rho=oa["mean"],
                headroom=gap,
                frac_of_oracle=(fa["mean"] / oa["mean"] if oa["mean"] not in (0, None) else None),
            )
            y += 1
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=6.9)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.9)
        ax.set_title(b, fontsize=10.5)
        ax.set_xlabel(r"Spearman $\rho$")
    handles = [
        Patch(facecolor=FAMILY_COLOR["map"], label="feasible arm"),
        Line2D(
            [],
            [],
            color=FAMILY_COLOR["oracle"],
            marker="D",
            ms=8,
            mfc="white",
            mew=2.2,
            lw=2.0,
            label="oracle ceiling (needs the true answer)",
        ),
    ]
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.suptitle(
        "Headroom to the oracle: how far each feasible method sits below the "
        "same read computed on the model's TRUE answer state (WildChat rung)",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "headroom_to_oracle")
    NUMBERS["fig6_headroom_to_oracle"] = dict(
        note=(
            "Oracle arms need the answer already generated, so they are "
            "ceilings, not deployable comparisons. map->ridge (real answers) "
            "is deployable at prediction time but needs TRUE answer states "
            "while training."
        ),
        results=out,
    )


# ==========================================================================
# FIG 7 - full arm roster with error bars + the reliability ceiling.
#         (also surfaces the text-embedding / surface-feature baselines)
# ==========================================================================
def fig_arms_with_ceiling() -> None:
    out: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharex=True)

    for ax, b in zip(axes, BEHAVIORS):
        rows = MAIN[b]["arm_rows"]
        ys, ms, ss, cols, labs = [], [], [], [], []
        for arm in ARM_ORDER:
            a = agg_rho(rows, arm, budget_l=LMAX[b], **OP)
            if a["mean"] is None:
                continue
            ys.append(len(ys))
            ms.append(a["mean"])
            ss.append(a["sd"])
            cols.append(arm_color(arm))
            labs.append(ARM_LABEL[arm])
            out.setdefault(b, {})[arm] = dict(mean=a["mean"], sd=a["sd"], n_replicates=a["n"])
        ax.barh(
            ys,
            ms,
            xerr=np.maximum(0.0, ss),
            color=cols,
            height=0.68,
            error_kw=dict(lw=0.9, capsize=2),
        )
        c = CEIL[b]
        if c["n"]:
            ax.axvline(c["mean"], color="#7a3d9e", ls="-.", lw=1.5)
            ax.text(
                c["mean"],
                -0.9,
                f"  reliability ceiling {c['mean']:.3f}",
                fontsize=7,
                color="#7a3d9e",
                va="bottom",
            )
            out.setdefault(b, {})["_reliability_ceiling_sb"] = c
        else:
            ax.text(
                0.98,
                0.02,
                "no split-half ceiling\n(fabrication-rate DV has no\nper-rollout scores)",
                transform=ax.transAxes,
                fontsize=6.8,
                ha="right",
                va="bottom",
                color="#7a3d9e",
            )
            GAPS.append(f"fig7 {b}: no split-half reliability ceiling available")
        ax.set_yticks(ys)
        ax.set_yticklabels(labs, fontsize=7.2)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.9)
        ax.set_title(f"{b} (held-out train dist., L={LMAX[b]:,})", fontsize=9.5)
        ax.set_xlabel(r"Spearman $\rho$")
    handles = [Patch(facecolor=c, label=FAMILY_LABEL[f]) for f, c in FAMILY_COLOR.items()]
    handles.append(
        Line2D(
            # ceiling_sb is the Spearman-Brown reliability r_yy = 2r/(1+r)
            # itself (arms.split_half_ceiling) — NOT its square root.
            [],
            [],
            color="#7a3d9e",
            ls="-.",
            lw=1.5,
            label=r"split-half ceiling $r_{yy}$ (SB)",
        )
    )
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.suptitle(
        "Full arm roster against the measurement ceiling "
        "(U = 18,793, E1 PV, context end state; error bars = SD over 3 seeds x 5 label draws)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "arms_with_reliability_ceiling")
    NUMBERS["fig7_arms_with_ceiling"] = dict(
        note=(
            "The ceiling is the Spearman-Brown split-half reliability of the "
            "judged DV, item-aligned even-odd rollout split, averaged over "
            "that behavior's train-rung cells - so it bounds the TRAIN rung, "
            "and is not transported to the OOD or WildChat panels. Arms 15/16 "
            "(text-embedding / surface-feature baselines) ran on the train "
            "rung only."
        ),
        results=out,
    )


# ==========================================================================
# FIG 8 - per-layer rho: where the predictive structure lives.
# ==========================================================================
# Dash pattern separates the two arms that share a family colour.
_LAYER_LS = {
    "arm7_map_ridge_pred": "-",
    "arm6_map_proj_e1": (0, (5, 2)),
    "arm4_ridge_ctx": "-",
    "arm1_ctx_e1": (0, (1.5, 1.5)),
}


def fig_per_layer() -> None:
    arms = ["arm6_map_proj_e1", "arm7_map_ridge_pred", "arm4_ridge_ctx", "arm1_ctx_e1"]
    out: dict = {}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=False)

    for ax, b in zip(axes, BEHAVIORS):
        pl_rows = [r for r in WC[b].get("per_layer_rows", []) if match(r, variant="context_end")]
        for arm in arms:
            rr = [r for r in pl_rows if r["arm"] == arm]
            if not rr:
                GAPS.append(f"fig8 {b}: no per-layer row for {arm}")
                continue
            r = rr[0]
            layers = r["layers"]
            vals = r["rho_per_layer"]
            # Colour = family; DASH PATTERN separates the two arms inside a
            # family, which a family-only colour rule would otherwise collide.
            ax.plot(
                layers,
                vals,
                lw=1.5,
                color=arm_color(arm),
                label=ARM_LABEL[arm],
                ls=_LAYER_LS[arm],
                alpha=0.95,
            )
            fl = r.get("frozen_layer")
            if fl is not None and fl in layers:
                ax.plot(
                    [fl],
                    [vals[layers.index(fl)]],
                    marker="o",
                    ms=6,
                    mfc="none",
                    mec=arm_color(arm),
                    mew=1.6,
                )
            out.setdefault(b, {})[arm] = dict(
                layers=layers,
                rho_per_layer=vals,
                frozen_layer=fl,
                frozen_source=r.get("frozen_source"),
                argmax_layer=int(layers[int(np.nanargmax(np.abs(np.array(vals, dtype=float))))]),
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(b, fontsize=10.5)
        ax.set_xlabel("layer")
        ax.set_ylabel(r"Spearman $\rho$")
    axes[0].legend(fontsize=6.8, loc="best")
    fig.suptitle(
        "Where the predictive structure lives: rank correlation per layer on the "
        "random WildChat rung (open circle = the arm's frozen layer, "
        "selected from the main lane, not on this rung)",
        y=1.03,
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, "per_layer_rho")
    NUMBERS["fig8_per_layer"] = dict(
        note=(
            "The frozen layer is the modal frozen layer of that arm's "
            "committed main-lane train cells, so no layer is selected on this "
            "rung; the curve shows what a rung-selected layer WOULD have "
            "reached, which is why the marker is often off the peak."
        ),
        results=out,
    )


# ==========================================================================
# FIG 9 - DV spread per evaluation setting, against the pre-registered gate.
# ==========================================================================
def fig_spread() -> None:
    from collections import defaultdict

    out: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharex=False)

    for col, b in enumerate(BEHAVIORS):
        with open(WT / f"dv_dataset/{b}/labeling.json") as f:
            main_rows = json.load(f)["rows"]
        with open(WT / f"wildchat_rung/dv_dataset/{b}/labeling.json") as f:
            wc_rows = json.load(f)["rows"]
        by_rung: dict[str, list[float]] = defaultdict(list)
        for r in main_rows + wc_rows:
            if r.get("dv") is not None:
                by_rung[r["rung"]].append(float(r["dv"]))

        rungs = [r for r in RUNGS[b] if r in by_rung]
        # hallucination's own rungs use a 0-1 fabrication rate; rescale x100
        # onto the gate's 0-100 scale, and hatch them as a DIFFERENT construct.
        rescaled = set()
        sds, bots, labs, hatches = [], [], [], []
        for rung in rungs:
            v = np.array(by_rung[rung], dtype=float)
            is_rate = b == "hallucination" and rung != "wildchat_rung"
            if is_rate:
                v = v * 100.0
                rescaled.add(rung)
            sds.append(float(v.std()))
            bots.append(float((v < 10).mean()))
            labs.append(RUNG_LABEL[(b, rung)].replace("\n", " "))
            hatches.append("//" if is_rate else "")
            out.setdefault(b, {})[rung] = dict(
                n=int(v.size),
                sd=float(v.std()),
                mean=float(v.mean()),
                frac_bottom_bin=float((v < 10).mean()),
                construct=("fabrication_rate_x100" if is_rate else "trait_rubric_0_100"),
                passes_sd_floor=bool(v.std() >= 10),
                passes_bottom_ceiling=bool((v < 10).mean() < 0.80),
                passes_gate=bool(v.std() >= 10 and (v < 10).mean() < 0.80),
            )

        ax = axes[0][col]
        cols = ["#2b7a3d" if s >= 10 else "#b02020" for s in sds]
        bars = ax.bar(range(len(sds)), sds, color=cols, width=0.6)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
            bar.set_edgecolor("white")
        ax.axhline(10, color="k", ls="--", lw=1.2)
        ax.text(len(sds) - 0.5, 10, " SD floor = 10", fontsize=7, va="bottom", ha="right")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=24, ha="right", fontsize=7)
        ax.set_title(b, fontsize=10.5)
        if col == 0:
            ax.set_ylabel("inter-context SD\n(0-100 scale)")

        ax = axes[1][col]
        cols = ["#2b7a3d" if f < 0.80 else "#b02020" for f in bots]
        bars = ax.bar(range(len(bots)), bots, color=cols, width=0.6)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
            bar.set_edgecolor("white")
        ax.axhline(0.80, color="k", ls="--", lw=1.2)
        ax.text(
            len(bots) - 0.5, 0.80, " bottom-bin ceiling = 0.80", fontsize=7, va="bottom", ha="right"
        )
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=24, ha="right", fontsize=7)
        if col == 0:
            ax.set_ylabel("fraction of contexts\nin the bottom [0,10) bin")

    handles = [
        Patch(facecolor="#2b7a3d", label="clears this condition"),
        Patch(facecolor="#b02020", label="fails this condition"),
        Patch(
            facecolor="#888888",
            hatch="//",
            edgecolor="white",
            label="fabrication rate x100 (different construct)",
        ),
    ]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle(
        "Judged-DV spread per evaluation setting against the pre-registered gate "
        "(both conditions required: SD >= 10 AND fewer than 80% of contexts in the bottom bin)",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "spread_by_setting")
    NUMBERS["fig9_spread"] = dict(
        note=(
            "The shipped gate was evaluated per BEHAVIOR, pooling that "
            "behavior's rungs; this decomposes the same two thresholds per "
            "SETTING. Hallucination's own rungs carry the 0-1 fabrication rate "
            "rescaled x100 - a different construct from the graded trait score "
            "in the solid bars, so cross-setting ordering there is not a "
            "meaningful comparison."
        ),
        results=out,
    )


def main() -> None:
    fig_delta_map_minus_direct()
    fig_sample_efficiency()
    fig_sign_consistency()
    fig_mapping_baselines()
    fig_nulls_and_controls()
    fig_headroom_to_oracle()
    fig_arms_with_ceiling()
    fig_per_layer()
    fig_spread()

    NUMBERS["_coverage_gaps"] = GAPS
    NUMBERS["_provenance"] = dict(
        worktree=str(WT),
        operating_slice=dict(OP, budget_l="LMAX per behavior", **{"LMAX": LMAX}),
        estimator_note=(
            "Every rho is Spearman between an arm's prediction and the judged "
            "DV at that arm's frozen layer. All arms inside any one panel are "
            "scored against the SAME judged DV on the SAME contexts."
        ),
    )
    with open(NUMDIR / "recut_numbers.json", "w") as f:
        json.dump(NUMBERS, f, indent=1, default=float)
    print(f"[recut] wrote {NUMDIR / 'recut_numbers.json'}")
    print(f"[recut] coverage gaps: {len(GAPS)}")
    for g in GAPS:
        print("   -", g)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
