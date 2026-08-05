"""Result 2 for #1739: does applying the context->answer map help the persona-vector read?

Renders ONE combined figure under `figures/issue_1739/result2_methods_v3/`:

  result2_methods_v3.{png,pdf,meta.json}

x is grouped by EVALUATION SETTING; each setting carries one bar per METHOD.
Faceted by behaviour (evil / sycophancy / hallucination) because the three are
not comparable on one axis -- hallucination's own rungs score a fabricated
FRACTION rescaled x100, a different construct from the graded 0-100 trait
rubric every other cell uses. The legend is shared.

All four methods are the SAME read -- a persona-vector projection -- differing
only in WHAT the vector is projected onto, so the bars form an ablation ladder
read left to right within each setting group (the nonlinear map contributes two
bars because kernel and MLP are two distinct maps, never averaged):

  context                       arm1_ctx_e1         reads the context
  mapped answer (linear map)    arm6_map_proj_e1    reads through the fitted map
  mapped answer (kernel map)    arm6 under nlood/kernel
  mapped answer (MLP map)       arm6 under nlood/mlp
  real answer (ceiling)         arm11_oracle_proj   reads the real answer

Settings (four): the persona-vectors synthetic elicitation grid, random
held-out WildChat, the out-of-fold train read (the labeled fit pool under 5
group-level folds, read out-of-fold -- the "random 20% held-out of the
eliciting train set"), and the behaviour-specific OOD rungs. Row labels reuse
the Result 1 distribution figure's two-part ROLE + IDENTITY convention
VERBATIM by importing its `SETTING_ROLE` / `SETTING_IDENTITY` tables, so the
two figures agree by construction.

Map condition. The spec's map is trained on ALL the training data (generic +
trait-eliciting) = the ADD condition, `eval_results/issue_1739/result2_trait_aug/`
(u_rung `add25261_gen18793_elic6468` for evil, `add34793_gen18793_elic16000` for
the other two). arm1 / arm11 do NOT read through the map at all -- they are
pure projections (`arms.py` L684 "projection arms (constant across folds; OOF
== the projection)", L726 arm11) -- so the map condition cannot move them and
their committed generic-root rows ARE their ADD-condition rows. Cells that
exist ONLY under the generic map (the two nonlinear maps; the linear map at
persona-vectors-synthetic) are drawn with a DASHED BLACK EDGE and named as such
in the legend -- never silently mixed into an ADD bar.

Metric. The spec's prose says R^2 but its Plot line says rho; the Plot line
wins. Every bar is the committed Spearman `rho_frozen` at the frozen layer of
the max-data operating slice (regime e1, full unlabeled pool, the behaviour's
maximum labelled budget: evil 8000, sycophancy / hallucination 16000). No R^2
anywhere on the axis.

Pure aggregation over committed artifacts: no fits, no GPU, no network.
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)
from issue1739_recut_common import BEHAVIORS, ROOT  # noqa: E402

# The Result 1 label tables, imported (not copied) so the two figures agree by
# construction at whatever state the source file is in.
from issue1739_result1_spread_fig_v2 import (  # noqa: E402
    SETTING_IDENTITY,
    SETTING_ROLE,
)

EVAL = ROOT / "eval_results/issue_1739"
OUT_FIG = ROOT / "figures/issue_1739/result2_methods_v3"
OUT_NUM = EVAL / "result2_methods_v3"

POINTS_PATH = EVAL / "result2_methods/result2_points.json"
TRAIT_AUG = {
    "evil": EVAL / "result2_trait_aug/evil/all_arms_spearman.add_generic_matched_swap.json",
    "sycophancy": EVAL / "result2_trait_aug/sycophancy/all_arms_spearman.add_swap.json",
    "hallucination": EVAL / "result2_trait_aug/hallucination/all_arms_spearman.add_swap.json",
}
NLOOD = EVAL / "new_arm_round/nlood/{behavior}/{kind}/arm_results/all_arms_spearman.json"

# Max-data operating slice, identical to the committed points file.
BUDGET_L = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}

# Setting order keeps hallucination's two DV constructs contiguous: the
# trait-rubric settings (persona-vectors synthetic, random WildChat) first,
# then the fabricated-fraction ones (out-of-fold train, NQ-Open, SimpleQA).
SETTINGS = {
    "evil": ["pvsynth", "wildchat_rung", "train", "hhrt", "toxicchat"],
    "sycophancy": ["pvsynth", "wildchat_rung", "train", "aita"],
    "hallucination": ["pvsynth", "wildchat_rung", "train", "nqopen", "simpleqa"],
}
# The ADD transfer rows name the out-of-fold train read `train_in_split`; the
# committed root names the same read `train`. Verified matched: identical
# n_eval and identical labelled-pool coverage per behaviour.
ADD_RUNG = {"train": "train_in_split"}

# Hallucination's own rungs score a fabricated FRACTION rescaled x100; its
# WildChat and persona-vectors-synthetic settings score the graded 0-100 trait
# rubric. A divider separates the two constructs.
FABRICATION_SETTINGS = {("hallucination", s) for s in ("train", "nqopen", "simpleqa")}

# --- methods: one persona-vector projection, four projection targets ----------
# Ladder order left to right; hue = methodology group (context / through the
# map / real answer), shade = map kind inside the map group.
GROUPS = [
    ("reads the context", None, [("pv_context", "context", "#08519C")]),
    (
        "reads through the fitted context -> answer map",
        None,
        [
            ("map_linear", "mapped answer (linear map)", "#8C3000"),
            ("map_kernel", "mapped answer (kernel map)", "#CC5500"),
            ("map_mlp", "mapped answer (MLP map)", "#F0A868"),
        ],
    ),
    (
        "reads the real answer (ceiling)",
        "//",
        [("oracle", "real answer", "#00694C")],
    ),
]
SLOTS = [m for _t, _h, ms in GROUPS for m, _l, _c in ms]
COLOR = {m: c for _t, _h, ms in GROUPS for m, _l, c in ms}
HATCH = {m: h for _t, h, ms in GROUPS for m, _l, _c in ms}
LABEL = {m: lbl for _t, _h, ms in GROUPS for m, lbl, _c in ms}

GROUP_WIDTH = 0.80
BAR_WIDTH = GROUP_WIDTH / len(SLOTS)


def _points_table() -> dict[tuple[str, str, str], dict]:
    """{(behavior, setting, arm_id): record} for the committed context-state points."""
    doc = json.loads(POINTS_PATH.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for p in doc["points"]:
        if p["input_state"] != "context":
            continue
        key = (p["behavior"], p["setting"], p["arm_id"])
        if key in out:
            raise SystemExit(f"duplicate committed point {key}")
        out[key] = p
    return out


def _add_rows() -> dict[tuple[str, str, str], dict]:
    """{(behavior, eval_rung, arm): row} for the ADD (generic + trait-eliciting) map."""
    out: dict[tuple[str, str, str], dict] = {}
    for beh, path in TRAIT_AUG.items():
        doc = json.loads(path.read_text())
        for r in doc.get("transfer_rows") or []:
            if r.get("map_condition") != "add":
                continue
            if r["variant"] != "context_end" or r["regime"] != "e1":
                continue
            if int(r["budget_l"]) != BUDGET_L[beh]:
                continue
            key = (beh, r["eval_rung"], r["arm"])
            if key in out:
                raise SystemExit(f"duplicate ADD row {key}")
            out[key] = r
    return out


def _nlood_rows() -> dict[tuple[str, str], dict]:
    """{(behavior, map_kind): the arm6 row} at the operating slice (train rung only)."""
    out: dict[tuple[str, str], dict] = {}
    for beh in BEHAVIORS:
        for kind in ("kernel", "mlp"):
            path = Path(str(NLOOD).format(behavior=beh, kind=kind))
            doc = json.loads(path.read_text())
            hits = [
                r
                for r in doc["arm_rows"]
                if r["arm"] == "arm6_map_proj_e1"
                and r["regime"] == "e1"
                and r["variant"] == "context_end"
                and r["u_rung_label"] == "full"
                and int(r["budget_l"]) == BUDGET_L[beh]
            ]
            if not hits:
                raise SystemExit(f"no nlood arm6 row for {beh}/{kind} at the operating slice")
            rhos = {round(float(r["rho_frozen"]), 9) for r in hits}
            if len(rhos) != 1:
                raise SystemExit(f"nlood {beh}/{kind} arm6 disagrees across replicates: {rhos}")
            # arm6 is a pure projection over a map fit on a FIXED pool, so every
            # (draw, seed) replicate reproduces one value; keep draw 0 / seed 0.
            row = sorted(hits, key=lambda r: (r["draw"], r["seed"]))[0]
            out[(beh, kind)] = {**row, "n_distinct_replicates": 1, "n_replicate_rows": len(hits)}
    return out


def collect() -> tuple[list[dict], dict]:
    """One record per (behavior, setting, method) that has a committed number."""
    pts, add, nl = _points_table(), _add_rows(), _nlood_rows()
    recs: list[dict] = []
    coverage: list[dict] = []

    def proj_arm(beh: str, setting: str, slot: str, arm_id: str) -> None:
        """arm1 / arm11: map-independent projections; committed root row IS the ADD row."""
        p = pts.get((beh, setting, arm_id))
        if p is None:
            coverage.append(
                dict(
                    behavior=beh,
                    setting=setting,
                    method=slot,
                    status="MISSING",
                    reason="no committed row at the operating slice",
                )
            )
            return
        recs.append(
            dict(
                behavior=beh,
                setting=setting,
                method=slot,
                arm_id=arm_id,
                rho=float(p["rho"]),
                ci=list(p["ci"]) if p["ci"] else None,
                n_replicates=int(p["n_replicates"]),
                n_eval=int(p["n_eval"]),
                layer=p["layer"],
                map_condition="n/a (map-independent projection)",
                dv_construct=p["dv_construct"],
                source_file=p["source_file"],
            )
        )
        coverage.append(
            dict(
                behavior=beh,
                setting=setting,
                method=slot,
                status="EXISTS",
                reason=f"committed root, {p['n_replicates']} replicate(s)",
            )
        )

    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            proj_arm(beh, setting, "pv_context", "arm1_ctx_e1")
            proj_arm(beh, setting, "oracle", "arm11_oracle_proj")

            # linear map: ADD condition where it exists, else the generic-map
            # cell, marked (never silently mixed).
            a = add.get((beh, ADD_RUNG.get(setting, setting), "arm6_map_proj_e1"))
            if a is not None:
                recs.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method="map_linear",
                        arm_id="arm6_map_proj_e1",
                        rho=float(a["rho_frozen"]),
                        ci=list(a["ci_frozen"]),
                        n_replicates=1,
                        n_eval=int(a["n_eval"]),
                        layer=int(a["layer"]),
                        map_condition=f"add ({a['u_rung_label']})",
                        dv_construct=(
                            "fabricated_fraction_rescaled_x100"
                            if (beh, setting) in FABRICATION_SETTINGS
                            else "trait_rubric_graded_0_100"
                        ),
                        source_file=str(TRAIT_AUG[beh].relative_to(ROOT)),
                    )
                )
                coverage.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method="map_linear",
                        status="EXISTS",
                        reason=f"ADD map condition ({a['u_rung_label']}), 1 replicate",
                    )
                )
            else:
                p = pts.get((beh, setting, "arm6_map_proj_e1"))
                if p is None:
                    coverage.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method="map_linear",
                            status="MISSING",
                            reason="no ADD row and no committed row",
                        )
                    )
                else:
                    recs.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method="map_linear",
                            arm_id="arm6_map_proj_e1",
                            rho=float(p["rho"]),
                            ci=list(p["ci"]) if p["ci"] else None,
                            n_replicates=int(p["n_replicates"]),
                            n_eval=int(p["n_eval"]),
                            layer=p["layer"],
                            map_condition="generic (no ADD cell at this setting)",
                            dv_construct=p["dv_construct"],
                            source_file=p["source_file"],
                        )
                    )
                    coverage.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method="map_linear",
                            status="EXISTS (generic map)",
                            reason="no ADD cell at this setting; generic-map value, marked",
                        )
                    )

            # nonlinear maps: committed at the out-of-fold train setting only.
            for slot, kind in (("map_kernel", "kernel"), ("map_mlp", "mlp")):
                if setting != "train":
                    coverage.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method=slot,
                            status="MISSING",
                            reason="nonlinear-map arm6 was only run at the out-of-fold "
                            "train rung (its transfer legs cover other arms)",
                        )
                    )
                    continue
                r = nl[(beh, kind)]
                recs.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method=slot,
                        arm_id="arm6_map_proj_e1",
                        rho=float(r["rho_frozen"]),
                        ci=list(r["ci_frozen"]),
                        n_replicates=1,
                        n_eval=None,
                        layer=int(r["layer"]),
                        map_condition=f"generic ({kind} map, u_fit_rows=18793)",
                        dv_construct=(
                            "fabricated_fraction_rescaled_x100"
                            if (beh, setting) in FABRICATION_SETTINGS
                            else "trait_rubric_graded_0_100"
                        ),
                        source_file=str(
                            Path(str(NLOOD).format(behavior=beh, kind=kind)).relative_to(ROOT)
                        ),
                    )
                )
                coverage.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method=slot,
                        status="EXISTS (generic map)",
                        reason=f"{kind} map on the generic pool, 1 replicate",
                    )
                )

    meta = dict(
        metric="Spearman rho_frozen (prediction vs judged behaviour expression)",
        metric_note="the spec's prose says R^2, its Plot line says rho; the Plot line wins",
        operating_slice=dict(
            regime="e1", variant="context_end", u_rung_label="full", budget_l=BUDGET_L
        ),
        add_rung_alias=ADD_RUNG,
        labelling="setting row labels are the Result 1 distribution figure's two-part "
        "ROLE + IDENTITY strings, imported from "
        "scripts/issue1739_result1_spread_fig_v2.py (SETTING_ROLE / SETTING_IDENTITY) "
        "at its current working-tree state so the two figures agree",
        map_independent_methods=dict(
            methods=["pv_context", "oracle"],
            evidence="src/explore_persona_space/experiments/issue_1739/arms.py L684 "
            "'projection arms (constant across folds; OOF == the projection)', "
            "L726 arm11 = _proj(za, rb) -- neither consumes the map, so the map "
            "condition cannot move them",
        ),
        matched_target_check="every joined (behaviour, setting) pair has identical n_eval "
        "across the committed-root and ADD rows (evil 6468/1868/519/1987, "
        "sycophancy 16000/1304/1982, hallucination 16000/3167/4021/1967)",
        labelled_budget_check="the labelled budget matches on every joined pair except evil's "
        "WildChat and persona-vectors-synthetic columns, where the committed leg is tagged "
        "L=6468 and the ADD row L=8000. Both exceed-or-equal evil's full labelled pool "
        "(n_train_contexts = 6468, `result2_trait_aug/evil/...json` meta), and the fit engine "
        "realizes identical row sets whenever budget_l >= n_ctx, so the two tags name the SAME "
        "6468 labelled rows",
        ci_semantics="multi-replicate bars: the committed bootstrap ci_frozen averaged over "
        "replicates. single-replicate bars (every map bar, and the WildChat / "
        "persona-vectors-synthetic columns): a within-draw paired bootstrap over "
        "EVAL CONTEXTS -- it carries no replicate-level uncertainty",
        coverage=coverage,
    )
    return recs, meta


def render(recs: list[dict], meta: dict) -> int:
    table = {(r["behavior"], r["setting"], r["method"]): r for r in recs}
    vals: list[float] = []
    for r in recs:
        vals.append(r["rho"])
        if r["ci"]:
            vals.extend(r["ci"])
    ylim = (min(0.0, min(vals)) - 0.05, max(vals) + 0.05)

    set_paper_style("blog", font_scale=0.85)
    fig = plt.figure(figsize=(21.0, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.30], width_ratios=[5, 4, 5])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    n_bars = 0
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        settings = SETTINGS[beh]
        xs = list(range(len(settings)))
        ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)

        # DV-construct divider (hallucination only).
        fab = [s for s in settings if (beh, s) in FABRICATION_SETTINGS]
        if fab and len(fab) < len(settings):
            edge = min(settings.index(s) for s in fab)
            ax.axvline(edge - 0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)

        for slot_i, slot in enumerate(SLOTS):
            offset = -GROUP_WIDTH / 2 + (slot_i + 0.5) * BAR_WIDTH
            for x, s in zip(xs, settings, strict=True):
                rec = table.get((beh, s, slot))
                if rec is None:
                    continue  # no committed number: no bar, never a zero bar
                generic = str(rec["map_condition"]).startswith("generic")
                ax.bar(
                    [x + offset],
                    [rec["rho"]],
                    width=BAR_WIDTH,
                    color=COLOR[slot],
                    hatch=HATCH[slot],
                    edgecolor="#000000" if generic else "#FFFFFF",
                    linewidth=1.0 if generic else 0.25,
                    linestyle=(0, (2, 1)) if generic else "solid",
                    zorder=3,
                )
                if rec["ci"]:
                    lo = max(0.0, rec["rho"] - rec["ci"][0])
                    hi = max(0.0, rec["ci"][1] - rec["rho"])
                    ax.errorbar(
                        [x + offset],
                        [rec["rho"]],
                        yerr=np.array([[lo], [hi]]),
                        fmt="none",
                        ecolor="#333333",
                        elinewidth=0.7,
                        capsize=0,
                        zorder=4,
                    )
                if rec["n_replicates"] == 1:
                    ax.plot(
                        [x + offset],
                        [rec["rho"]],
                        marker="o",
                        markersize=3.4,
                        markerfacecolor="none",
                        markeredgecolor="#111111",
                        markeredgewidth=0.7,
                        linestyle="none",
                        zorder=5,
                    )
                n_bars += 1

        ax.set_xticks(xs)
        # Rotated so adjacent two-line role/identity labels cannot collide.
        ax.set_xticklabels(
            [f"{SETTING_ROLE[s]}\n{SETTING_IDENTITY[(beh, s)]}" for s in settings],
            fontsize=7.6,
            rotation=14,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_ylim(*ylim)
        ax.set_title(beh, loc="left")

    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    axes[1].set_xlabel("evaluation setting")
    fig.suptitle(
        "Result 2: persona-vector projection onto context -> mapped answer -> real answer",
        x=0.006,
        ha="left",
    )

    handles_x = 0.0
    for gtitle, hatch, methods in GROUPS:
        leg = legend_ax.legend(
            handles=[
                Patch(facecolor=c, hatch=hatch, edgecolor="#FFFFFF", linewidth=0.25, label=lbl)
                for _m, lbl, c in methods
            ],
            title=gtitle,
            ncol=1,
            loc="upper left",
            alignment="left",
            frameon=False,
            fontsize=8.2,
            borderpad=0.0,
            bbox_to_anchor=(handles_x, 1.0),
            bbox_transform=legend_ax.transAxes,
        )
        leg.get_title().set_fontsize(8.6)
        leg.get_title().set_fontweight("semibold")
        legend_ax.add_artist(leg)
        handles_x += 0.30

    marks = legend_ax.legend(
        handles=[
            Patch(
                facecolor="#BFBFBF",
                edgecolor="#000000",
                linewidth=1.0,
                linestyle=(0, (2, 1)),
                label="dashed edge: map fit on the GENERIC WildChat pool only\n"
                "(no ADD cell at that setting)",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor="#111111",
                linestyle="none",
                label="open circle: single replicate\n"
                "(CI is a within-draw bootstrap over eval contexts)",
            ),
        ],
        title="reading the marks",
        ncol=2,
        columnspacing=3.0,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.2,
        borderpad=0.0,
        bbox_to_anchor=(0.0, 0.40),
        bbox_transform=legend_ax.transAxes,
    )
    marks.get_title().set_fontsize(8.6)
    marks.get_title().set_fontweight("semibold")
    legend_ax.add_artist(marks)

    note = (
        "Missing bar = that method was not run at that setting (the two nonlinear maps ran at "
        "the out-of-fold train rung only); never a zero.   Hallucination's in-distribution / "
        "NQ-Open / SimpleQA settings (right of the dashed line) score fabrication rate x100, a "
        "different construct from the 0-100 trait rubric everywhere else."
    )
    fig.text(0.006, 0.008, note, ha="left", va="bottom", fontsize=8.0, color="#4A4A4A", wrap=True)

    savefig_paper(fig, "result2_methods_v3", dir=OUT_FIG)
    plt.close(fig)
    return n_bars


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NUM.mkdir(parents=True, exist_ok=True)
    recs, meta = collect()
    n_bars = render(recs, meta)
    if n_bars != len(recs):
        raise SystemExit(f"plotted {n_bars} bars but collected {len(recs)} records")
    (OUT_NUM / "result2_v3_points.json").write_text(
        json.dumps({**meta, "n_points": len(recs), "points": recs}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT_FIG / 'result2_methods_v3.png'} ({n_bars} bars)")
    print(f"wrote {OUT_NUM / 'result2_v3_points.json'} ({len(recs)} records)")
    n_mi = sum(1 for c in meta["coverage"] if c["status"] == "MISSING")
    print(
        f"coverage: {len(recs)} plotted, {n_mi} missing "
        f"({len(meta['coverage'])} method x setting x behaviour cells)"
    )


if __name__ == "__main__":
    main()
