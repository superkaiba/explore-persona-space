"""Result 2 (FAIR PROTOCOL) for #1739: every method gets the same training data.

Renders ONE combined figure under `figures/issue_1739/result2_fair/`:

  result2_fair.{png,pdf,meta.json}

and writes `eval_results/issue_1739/result2_fair/result2_fair_points.json`
(points + coverage + the fair-vs-committed comparison + the linear-collapse
check).

x is grouped by EVALUATION SETTING; each setting carries one bar per METHOD
CELL, faceted by behaviour. Methods (7, user list) + the two optional MLP-map
readout cells, legend grouped by methodology family with map kind and readout
family both legible (solid = linear map, dotted hatch = MLP map):

  reads the context          pv_context      arm1_ctx_e1        PV projected on context
                             regression_ctx  arm4_ridge_ctx     ridge, context -> DV
  reads the mapped answer    pv_map_linear   arm6 + linear map  PV on mapped answer
                             pv_map_mlp      arm6 + MLP map
                             reg_map_linear  arm7 + linear map  ridge on mapped answer
                             reg_map_mlp     arm7 + MLP map
                             mlp_map_linear  arm19 + linear map MLP on mapped answer
                             mlp_map_mlp     arm19 + MLP map
  reads the real answer      oracle          arm11_oracle_proj  PV on real answer

Under the fair protocol every method shares ONE training-data allowance: the
map + whitening are the ADD/union condition (generic WildChat pool + eliciting
train pairs), and the label-consuming readouts (arms 4/7/19) train on ALL
judged training data (eliciting train budget cell + the judged WildChat train
split). Scored rows come from `eval_results/issue_1739/result2_fair/<b>/`
(scripts/issue1739_result2fair_score.py on pod-1739-r2fair).

The LINEAR-COLLAPSE CHECK: an earlier spec dropped regression-on-mapped-answer
on the argument that a linear map composed with a linear readout collapses to
a context regression; that holds only for linear+linear, so arm7(linear) is
kept and max |rho(arm7,linear) - rho(arm4)| across the fair linear cells is
REPORTED — the collapse is measured, not assumed.

Per-setting reliability CEILINGS (sqrt split-half r_yy, committed
`result1_spread/spread_stats.json`) are drawn as segments spanning the
context-BOUNDED cells only — every method except the real-answer arm is a
deterministic function of the context (a mapped answer, linear or MLP, is a
deterministic function of the context), so all are bounded; the real-answer
arm's input shares information with the DV's judge noise and is NOT. The
WildChat ceiling was computed on the FULL rung; this figure's WildChat column
evaluates its held-out ~20% split (caveat carried in meta).

Metric: Spearman rho_frozen (the spec's Plot line; its prose says R^2 — the
Plot line wins, flagged, same convention as the committed v3 figure).

Sycophancy's corrected OOD rungs are mid-flight on a sibling round; its OOD
column is the EXISTING aita rung, labelled exactly as Result 1 labels it, with
an in-figure pending note.

Pure aggregation over committed + fair-scored artifacts: no fits, no GPU.
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

# The Result 1 label tables, imported (not copied) so the figures agree by
# construction at whatever state the source file is in.
from issue1739_result1_spread_fig_v2 import (  # noqa: E402
    SETTING_IDENTITY,
    SETTING_ROLE,
)

EVAL = ROOT / "eval_results/issue_1739"
OUT_FIG = ROOT / "figures/issue_1739/result2_fair"
OUT_NUM = EVAL / "result2_fair"

FAIR_SUMMARY = {b: EVAL / "result2_fair" / b / "all_arms_spearman.json" for b in BEHAVIORS}
V3_POINTS = EVAL / "result2_methods_v3/result2_v3_points.json"
SPREAD_STATS = EVAL / "result1_spread/spread_stats.json"
TRAIT_AUG = {
    "evil": EVAL / "result2_trait_aug/evil/all_arms_spearman.add_generic_matched_swap.json",
    "sycophancy": EVAL / "result2_trait_aug/sycophancy/all_arms_spearman.add_swap.json",
    "hallucination": EVAL / "result2_trait_aug/hallucination/all_arms_spearman.add_swap.json",
}

BUDGET_L = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}
# Full labelled pool per behaviour: evil's committed wide rows are tagged
# L=6468 while the ADD rows say L=8000 — both realize the identical full row
# set (budget_l >= n_ctx), per the v3 figure's labelled_budget_check.
FULL_POOL_L = {"evil": 6468, "sycophancy": 16000, "hallucination": 16000}
SETTINGS = {
    "evil": ["pvsynth", "wildchat_rung", "train", "hhrt", "toxicchat"],
    "sycophancy": ["pvsynth", "wildchat_rung", "train", "aita"],
    "hallucination": ["pvsynth", "wildchat_rung", "train", "nqopen", "simpleqa"],
}
FABRICATION_SETTINGS = {("hallucination", s) for s in ("train", "nqopen", "simpleqa")}
PENDING_SYCO_OOD = (
    "sycophancy's corrected OOD rungs (5 genuinely-OOD eval sets, sibling round: staged, "
    "generated, captured; judge + scoring remain) are PENDING — the OOD column shown is the "
    "existing held-out-Reddit aita rung and will be rebuilt when the corrected rungs land"
)

# (arm_id, map_kind) -> method slot
METHOD_OF = {
    ("arm1_ctx_e1", "linear"): "pv_context",
    ("arm4_ridge_ctx", "linear"): "regression_ctx",
    ("arm11_oracle_proj", "linear"): "oracle",
    ("arm6_map_proj_e1", "linear"): "pv_map_linear",
    ("arm6_map_proj_e1", "mlp"): "pv_map_mlp",
    ("arm7_map_ridge_pred", "linear"): "reg_map_linear",
    ("arm7_map_ridge_pred", "mlp"): "reg_map_mlp",
    ("arm19_map_mlp_pred", "linear"): "mlp_map_linear",
    ("arm19_map_mlp_pred", "mlp"): "mlp_map_mlp",
}
FAIR_READOUT_ARMS = ("arm4_ridge_ctx", "arm7_map_ridge_pred", "arm19_map_mlp_pred")
MLP_MAP_HATCH = ".."
GROUPS = [
    (
        "reads the context",
        [
            ("pv_context", "PV projected on context", "#08519C", None),
            ("regression_ctx", "regression: context -> behaviour", "#6BAED6", None),
        ],
    ),
    (
        "reads the mapped answer (solid = linear map, dotted = MLP map)",
        [
            ("pv_map_linear", "PV on mapped answer (linear map)", "#8C3000", None),
            ("pv_map_mlp", "PV on mapped answer (MLP map)", "#8C3000", MLP_MAP_HATCH),
            ("reg_map_linear", "regression on mapped answer (linear map)", "#CC5500", None),
            ("reg_map_mlp", "regression on mapped answer (MLP map)", "#CC5500", MLP_MAP_HATCH),
            ("mlp_map_linear", "MLP on mapped answer (linear map)", "#F0A868", None),
            ("mlp_map_mlp", "MLP on mapped answer (MLP map)", "#F0A868", MLP_MAP_HATCH),
        ],
    ),
    (
        "reads the real answer (ceiling)",
        [("oracle", "PV on real answer", "#00694C", "//")],
    ),
]
SLOTS = [m for _t, ms in GROUPS for m, _l, _c, _h in ms]
COLOR = {m: c for _t, ms in GROUPS for m, _l, c, _h in ms}
HATCH = {m: h for _t, ms in GROUPS for m, _l, _c, h in ms}
# The reliability ceiling bounds every deterministic function of the CONTEXT —
# all cells except the real-answer arm (whose input shares information with
# the DV's judge noise).
CEILING_BOUNDED = tuple(m for m in SLOTS if m != "oracle")

GROUP_WIDTH = 0.84
BAR_WIDTH = GROUP_WIDTH / len(SLOTS)

# Committed rows for the readout arms the v3 points file does not carry
# (arm4 everywhere; arm7 generic-map from the wide trees). Sources mirror the
# jobd `--collect-generic-only` extractor's map plus the wide pvsynth tree.
COMMITTED_ARM_SOURCES = {
    "main_train": EVAL / "{behavior}/arm_results/all_arms_spearman.json",
    "wide_pvsynth": EVAL / "wide/pvsynth/{behavior}/all_arms_spearman.json",
    "wide_wc": EVAL / "wide/wildchat_rung/{behavior}/all_arms_spearman.json",
    "wide_ood": EVAL / "wide_ood/{behavior}_transfer.jsonl",
    "gapfill": EVAL / "result2_gapfill/merged/{behavior}/arm_results/all_arms_spearman.json",
}


def _slice_ok(r: dict, beh: str) -> bool:
    if r.get("regime") != "e1" or r.get("variant") != "context_end":
        return False
    if str(r.get("u_rung_label")) != "full":
        return False
    bl = r.get("budget_l")
    return bl is None or int(bl) >= FULL_POOL_L[beh]


def committed_arm_rows(arm: str) -> dict[tuple[str, str], dict]:
    """{(behavior, setting): committed row} for one arm (eliciting-only readout)."""
    out: dict[tuple[str, str], dict] = {}

    def keep(beh: str, setting: str, r: dict, source: str) -> None:
        key = (beh, setting)
        if key in out:
            return  # first source in the fixed order wins; all report the same slice
        rho = r.get("rho_frozen", r.get("rho"))
        if rho is None:
            return
        out[key] = {
            "rho": float(rho),
            "ci": list(r.get("ci_frozen") or r.get("ci") or []) or None,
            "n_eval": r.get("n_eval"),
            "source": source,
        }

    for beh in BEHAVIORS:
        for src_name, tmpl in COMMITTED_ARM_SOURCES.items():
            path = Path(str(tmpl).format(behavior=beh))
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                rows = []
                with path.open() as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        rows += obj["rows"] if isinstance(obj, dict) and "rows" in obj else [obj]
            else:
                doc = json.loads(path.read_text())
                rows = (doc.get("transfer_rows") or []) + (doc.get("arm_rows") or [])
            for r in rows:
                if r.get("arm") != arm or r.get("f_u") is not None:
                    continue
                if not _slice_ok(r, beh):
                    continue
                if r.get("draw") not in (None, 0) or r.get("seed") not in (None, 0):
                    continue
                setting = r.get("eval_rung") or ("train" if src_name == "main_train" else None)
                if setting in (None, "train_in_split"):
                    setting = "train"
                if setting in SETTINGS[beh]:
                    keep(beh, setting, r, f"{src_name}:{path.relative_to(ROOT)}")
    return out


def committed_arm7_add() -> dict[tuple[str, str], dict]:
    """{(behavior, setting): row} for arm7 under the committed ADD map (r2aug)."""
    out: dict[tuple[str, str], dict] = {}
    for beh, path in TRAIT_AUG.items():
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        for r in doc.get("transfer_rows") or []:
            if r.get("arm") != "arm7_map_ridge_pred" or r.get("map_condition") != "add":
                continue
            if r.get("variant") != "context_end" or r.get("regime") != "e1":
                continue
            if int(r.get("budget_l", -1)) != BUDGET_L[beh]:
                continue
            setting = r["eval_rung"]
            if setting == "train_in_split":
                setting = "train"
            if setting not in SETTINGS[beh]:
                continue
            key = (beh, setting)
            if key in out:
                continue
            out[key] = {
                "rho": float(r["rho_frozen"]),
                "ci": list(r.get("ci_frozen") or []) or None,
                "n_eval": r.get("n_eval"),
                "source": f"r2aug-add:{path.relative_to(ROOT)}",
            }
    return out


def committed_v3() -> dict[tuple[str, str, str], dict]:
    doc = json.loads(V3_POINTS.read_text())
    out = {}
    for p in doc["points"]:
        out[(p["behavior"], p["setting"], p["method"])] = p
    return out


def ceilings() -> dict[tuple[str, str], float]:
    doc = json.loads(SPREAD_STATS.read_text())
    return {
        (c["behavior"], c["setting"]): float(c["ceiling_sqrt_r_yy"])
        for c in doc["cells"]
        if c.get("ceiling_sqrt_r_yy") is not None
    }


def collect() -> tuple[list[dict], list[dict], dict]:
    """Fair points + coverage + meta from the fair-scored summaries."""
    recs: list[dict] = []
    coverage: list[dict] = []
    fair_meta: dict[str, dict] = {}
    for beh in BEHAVIORS:
        path = FAIR_SUMMARY[beh]
        if not path.exists():
            raise SystemExit(f"fair summary absent: {path} — pull issue1739_result2_fair from HF")
        doc = json.loads(path.read_text())
        fair_meta[beh] = {
            k: doc["meta"].get(k)
            for k in (
                "leakage",
                "readout_protocol",
                "map_kind_resolution",
                "map_reuse_note",
                "arm19_note",
                "frozen_layer_sources",
                "dv_scaling_note",
                "dv_construct_caveat",
                "variant_scope",
                "git_commit",
                "wall_s",
            )
        }
        rows = doc.get("transfer_rows") or []
        seen: dict[tuple[str, str, str], dict] = {}
        for r in rows:
            if (r["arm"], r.get("map_kind", "linear")) not in METHOD_OF:
                continue
            key = (r["eval_rung"], r["arm"], r.get("map_kind", "linear"))
            if key in seen:
                raise SystemExit(f"duplicate fair row {beh}/{key}")
            seen[key] = r
        for setting in SETTINGS[beh]:
            for (arm, kind), method in METHOD_OF.items():
                r = seen.get((setting, arm, kind))
                if r is None:
                    coverage.append(
                        dict(
                            behavior=beh,
                            setting=setting,
                            method=method,
                            status="OMITTED",
                            reason=f"no fair-scored row for {arm}/{kind} at {setting} "
                            "(see transfer_skips in the fair summary)",
                        )
                    )
                    continue
                recs.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method=method,
                        arm_id=arm,
                        map_kind=kind,
                        rho=float(r["rho_frozen"]),
                        ci=list(r.get("ci_frozen") or []) or None,
                        n_replicates=1,
                        n_eval=int(r["n_eval"]),
                        layer=r.get("layer"),
                        map_condition="add (fair protocol)",
                        readout=(
                            "fair union readout" if arm in FAIR_READOUT_ARMS else "label-free"
                        ),
                        dv_construct=(
                            "fabricated_fraction_rescaled_x100"
                            if (beh, setting) in FABRICATION_SETTINGS
                            else "trait_rubric_graded_0_100"
                        ),
                        source_file=str(path.relative_to(ROOT)),
                    )
                )
                coverage.append(
                    dict(behavior=beh, setting=setting, method=method, status="EXISTS", reason="")
                )
        if beh == "sycophancy":
            for method in SLOTS:
                coverage.append(
                    dict(
                        behavior=beh,
                        setting="corrected_ood_rungs",
                        method=method,
                        status="PENDING-syco-OOD",
                        reason=PENDING_SYCO_OOD,
                    )
                )
    return recs, coverage, fair_meta


def compare(recs: list[dict]) -> dict:
    """Fair vs committed: per-cell deltas, per-setting rankings, collapse check."""
    v3 = committed_v3()
    a4 = committed_arm_rows("arm4_ridge_ctx")
    a7_generic = committed_arm_rows("arm7_map_ridge_pred")
    a7_add = committed_arm7_add()
    fair = {(r["behavior"], r["setting"], r["method"]): r for r in recs}

    def committed_for(beh: str, setting: str, method: str):
        if method in ("pv_context", "oracle", "pv_map_linear"):
            v3m = {"pv_context": "pv_context", "oracle": "oracle", "pv_map_linear": "map_linear"}
            c = v3.get((beh, setting, v3m[method]))
            if c is None:
                return None, None
            src = c.get("source_file")
            if str(c.get("map_condition", "")).startswith("generic"):
                src = f"{src} [committed value was GENERIC-map]"
            return float(c["rho"]), src
        if method == "pv_map_mlp":
            c = v3.get((beh, setting, "map_mlp"))
            if c is None:
                return None, None
            return (
                float(c["rho"]),
                f"{c.get('source_file')} [committed MLP-map cell: GENERIC pool, train rung only]",
            )
        if method == "regression_ctx":
            c = a4.get((beh, setting))
            return (None, None) if c is None else (c["rho"], c["source"])
        if method == "reg_map_linear":
            c = a7_add.get((beh, setting)) or a7_generic.get((beh, setting))
            if c is None:
                return None, None
            src = c["source"]
            if not src.startswith("r2aug-add"):
                src = f"{src} [committed value was GENERIC-map]"
            return c["rho"], src
        return None, None  # reg_map_mlp / mlp_map_* : no committed cells (new)

    cells = []
    rank_changes = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            fair_rank = []
            comm_rank = []
            for method in SLOTS:
                f = fair.get((beh, setting, method))
                c_rho, c_src = committed_for(beh, setting, method)
                cells.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        method=method,
                        fair_rho=None if f is None else f["rho"],
                        committed_rho=c_rho,
                        delta=None if (f is None or c_rho is None) else f["rho"] - c_rho,
                        committed_source=c_src,
                    )
                )
                if f is not None:
                    fair_rank.append((f["rho"], method))
                if c_rho is not None:
                    comm_rank.append((c_rho, method))
            fr = [m for _v, m in sorted(fair_rank, reverse=True)]
            cr = [m for _v, m in sorted(comm_rank, reverse=True)]
            rank_changes.append(
                dict(
                    behavior=beh,
                    setting=setting,
                    fair_ranking=fr,
                    committed_ranking_over_available_committed_cells=cr,
                    ranking_changed_on_shared_methods=([m for m in fr if m in set(cr)] != cr),
                )
            )
    # linear-collapse check: ridge on the LINEAR-mapped answer vs ridge on the
    # context, same fair readout training set — measured, not assumed.
    collapse = []
    for beh in BEHAVIORS:
        for setting in SETTINGS[beh]:
            f7 = fair.get((beh, setting, "reg_map_linear"))
            f4 = fair.get((beh, setting, "regression_ctx"))
            if f7 and f4:
                collapse.append(
                    dict(
                        behavior=beh,
                        setting=setting,
                        rho_arm7_linear=f7["rho"],
                        rho_arm4=f4["rho"],
                        abs_diff=abs(f7["rho"] - f4["rho"]),
                    )
                )
    max_collapse = max((c["abs_diff"] for c in collapse), default=None)
    return {
        "cells": cells,
        "rankings": rank_changes,
        "linear_collapse_check": {
            "definition": (
                "max |rho(arm7, linear map) - rho(arm4)| across the fair linear cells — "
                "the empirical check on the linear+linear collapse argument (which holds "
                "only for linear map + linear readout; it does not hold for MLP readouts "
                "or MLP maps). Note the two are not algebraically identical even in the "
                "linear case: the ridge penalty acts in different coordinates."
            ),
            "max_abs_diff": max_collapse,
            "per_cell": collapse,
        },
    }


def render(recs: list[dict]) -> int:
    table = {(r["behavior"], r["setting"], r["method"]): r for r in recs}
    ceil = ceilings()
    vals: list[float] = []
    for r in recs:
        vals.append(r["rho"])
        if r["ci"]:
            vals.extend(r["ci"])
    ylim = (min(0.0, min(vals)) - 0.05, max(1.0, max(vals)) + 0.02)

    set_paper_style("blog", font_scale=0.85)
    fig = plt.figure(figsize=(24.0, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.34], width_ratios=[5, 4, 5])
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    bounded_slots = [SLOTS.index(m) for m in CEILING_BOUNDED]
    n_bars = 0
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        settings = SETTINGS[beh]
        xs = list(range(len(settings)))
        ax.axhline(0.0, color="#B0B0B0", linewidth=0.9, zorder=1)

        fab = [s for s in settings if (beh, s) in FABRICATION_SETTINGS]
        if fab and len(fab) < len(settings):
            edge = min(settings.index(s) for s in fab)
            ax.axvline(edge - 0.5, color="#444444", linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)

        for slot_i, slot in enumerate(SLOTS):
            offset = -GROUP_WIDTH / 2 + (slot_i + 0.5) * BAR_WIDTH
            for x, s in zip(xs, settings, strict=True):
                rec = table.get((beh, s, slot))
                if rec is None:
                    continue  # no scored number: no bar, never a zero bar
                ax.bar(
                    [x + offset],
                    [rec["rho"]],
                    width=BAR_WIDTH,
                    color=COLOR[slot],
                    hatch=HATCH[slot],
                    edgecolor="#FFFFFF",
                    linewidth=0.25,
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
                        elinewidth=0.6,
                        capsize=0,
                        zorder=4,
                    )
                n_bars += 1

        # per-setting reliability ceiling: a segment spanning ONLY the
        # ceiling-bounded (context-based) bars — never one line per facet.
        for x, s in zip(xs, settings, strict=True):
            c = ceil.get((beh, s))
            if c is None:
                continue
            lo_off = -GROUP_WIDTH / 2 + min(bounded_slots) * BAR_WIDTH
            hi_off = -GROUP_WIDTH / 2 + (max(bounded_slots) + 1) * BAR_WIDTH
            ax.plot(
                [x + lo_off, x + hi_off],
                [c, c],
                color="#B00020",
                linewidth=1.6,
                linestyle=(0, (3, 2)),
                zorder=5,
            )

        ax.set_xticks(xs)
        labels = []
        for s in settings:
            lab = f"{SETTING_ROLE[s]}\n{SETTING_IDENTITY[(beh, s)]}"
            if beh == "sycophancy" and s == "aita":
                lab += "\n(corrected OOD rungs pending)"
            if s == "wildchat_rung":
                lab += "\n(held-out 20% split)"
            labels.append(lab)
        ax.set_xticklabels(labels, fontsize=7.4, rotation=14, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, max(xs) + 0.6)
        ax.set_ylim(*ylim)
        ax.set_title(beh, loc="left")

    axes[0].set_ylabel("Spearman rho, prediction vs judged behaviour expression")
    axes[1].set_xlabel("evaluation setting")
    fig.suptitle(
        "Result 2 (fair protocol): every method trains on ALL the training data — "
        "map on generic + eliciting; behaviour readouts on all judged data",
        x=0.006,
        ha="left",
    )

    handles_x = 0.0
    widths = (0.18, 0.44, 0.18)
    for (gtitle, methods), w in zip(GROUPS, widths, strict=True):
        leg = legend_ax.legend(
            handles=[
                Patch(facecolor=c, hatch=h, edgecolor="#FFFFFF", linewidth=0.25, label=lbl)
                for _m, lbl, c, h in methods
            ],
            title=gtitle,
            ncol=1 if len(methods) <= 2 else 2,
            loc="upper left",
            alignment="left",
            frameon=False,
            fontsize=8.0,
            borderpad=0.0,
            bbox_to_anchor=(handles_x, 1.0),
            bbox_transform=legend_ax.transAxes,
        )
        leg.get_title().set_fontsize(8.4)
        leg.get_title().set_fontweight("semibold")
        legend_ax.add_artist(leg)
        handles_x += w

    marks = legend_ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color="#B00020",
                linewidth=1.6,
                linestyle=(0, (3, 2)),
                label="reliability ceiling sqrt(r_yy) — spans the context-based bars only\n"
                "(every method except PV-on-real-answer; that arm is not bounded by it)",
            ),
        ],
        title="reading the marks",
        ncol=1,
        loc="upper left",
        alignment="left",
        frameon=False,
        fontsize=8.0,
        borderpad=0.0,
        bbox_to_anchor=(0.0, 0.30),
        bbox_transform=legend_ax.transAxes,
    )
    marks.get_title().set_fontsize(8.4)
    marks.get_title().set_fontweight("semibold")
    legend_ax.add_artist(marks)

    note = (
        "All bars are single replicate (draw 0 / seed 0); CI = within-draw paired bootstrap "
        "over eval contexts.   Missing bar = no scored number at that cell; never a zero.   "
        "Hallucination right of the dashed divider scores fabrication rate x100, a different "
        "construct from the 0-100 trait rubric elsewhere.   WildChat ceilings were computed on "
        "the full rung; the WildChat column evaluates its held-out 20% split.   Sycophancy's "
        "corrected OOD rungs are pending (sibling round); its OOD column is the existing aita "
        "rung."
    )
    fig.text(0.006, 0.008, note, ha="left", va="bottom", fontsize=8.0, color="#4A4A4A", wrap=True)

    savefig_paper(fig, "result2_fair", dir=OUT_FIG)
    plt.close(fig)
    return n_bars


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NUM.mkdir(parents=True, exist_ok=True)
    recs, coverage, fair_meta = collect()
    comparison = compare(recs)
    n_bars = render(recs)
    if n_bars != len(recs):
        raise SystemExit(f"plotted {n_bars} bars but collected {len(recs)} records")
    payload = dict(
        metric="Spearman rho_frozen (prediction vs judged behaviour expression)",
        metric_note="the spec's prose says R^2, its Plot line says rho; the Plot line wins",
        protocol=(
            "FAIR: map + whitening = ADD/union (generic WildChat pool + eliciting train "
            "pairs), fit once per map kind (linear / MLP) on one shared whitening; the "
            "label-consuming readouts (arms 4/7/19) train on the eliciting train budget "
            "cell UNION the judged WildChat train split (sha1 mod-5 bucket 4 held out); "
            "the projection arms are label-free and share the whitening + eval subsets; "
            "pvsynth judged data enters only through r_B (recorded deviation for the "
            "regression/MLP readouts)"
        ),
        methods_note=(
            "7 user methods + 2 optional MLP-map readout cells (regression/MLP on the "
            "MLP-mapped answer), shown as separate bars — map kinds never averaged. "
            "arm19_map_mlp_pred (MLP on mapped answer) is NEW this round."
        ),
        operating_slice=dict(
            regime="e1", variant="context_end", u_rung_label="add", budget_l=BUDGET_L
        ),
        ceiling_semantics=(
            "sqrt split-half r_yy from result1_spread/spread_stats.json, drawn as per-setting "
            "segments spanning the context-BOUNDED cells (every method except "
            "PV-on-real-answer — a mapped answer, linear or MLP, is a deterministic function "
            "of the context); the real-answer arm's input shares information with the DV's "
            "judge noise, so the ceiling does not bound it. WildChat ceilings computed on the "
            "full rung while the fair WildChat column evaluates the held-out 20% split"
        ),
        labelling=(
            "setting row labels are the Result 1 figure's two-part ROLE + IDENTITY strings, "
            "imported from scripts/issue1739_result1_spread_fig_v2.py at its current "
            "working-tree state"
        ),
        pending_syco_ood=PENDING_SYCO_OOD,
        fair_meta=fair_meta,
        coverage=coverage,
        comparison_vs_committed=comparison,
        n_points=len(recs),
        points=recs,
    )
    (OUT_NUM / "result2_fair_points.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT_FIG / 'result2_fair.png'} ({n_bars} bars)")
    print(f"wrote {OUT_NUM / 'result2_fair_points.json'} ({len(recs)} records)")
    cc = comparison["linear_collapse_check"]
    print(f"linear-collapse check: max |arm7(linear) - arm4| = {cc['max_abs_diff']}")


if __name__ == "__main__":
    main()
