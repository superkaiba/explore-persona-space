"""Contrastive-negatives vs positive-only regime read over #1768's committed measurements.

Inline 0-GPU analysis round (user-chat carve-out). Purely descriptive/observational:
pairs regime-matched arms (same method, behavior, context, seed) and contrasts
(a) weights-carried dose, (b) map change D beyond what dose predicts (rank-residualized),
(c) write-direction alignments, (d) context movement, (e) write rank, (f) the
base-geometry gate. No training, no generation, no GPU.

Inputs: eval_results/issue_1768/{arm_registry,map_change_summary,direction_reads,
context_movement,gate_reads}.json, model_text_2x2/summary.json,
lasttoken_repool/summary.json + cells/, fits/<arm>_L<L>.json.

Outputs: eval_results/issue_1768/contrastive_read/{arm_table,summary}.json and
figures/issue_1768/contrastive_read/{regime_dose_D_paired,regime_paired_contrasts}.{png,pdf}.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before numpy/scipy/matplotlib: shared-VM thread caps

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

RESULTS = Path("eval_results/issue_1768")
OUT = RESULTS / "contrastive_read"
FIGDIR = "figures/issue_1768/contrastive_read"
LAYERS = (14, 19, 25)
PRIMARY_LAYER = 19
BODY_RHO_CLAIM = 0.98  # body's Spearman(D, round-1 matched-text shift) claim at layer 19

BEH_NAMES = {
    "cas": "casual writing style",
    "imp": "impoliteness",
    "syc": "sycophancy",
    "mk": "marker",
}
CTX_NAMES = {
    "pers": "persona",
    "bare": "bare",
    "conv": "conversation",
    "icl": "in-context demos",
}
REGIME_NAMES = {"con": "contrastive", "po": "positive-only"}


def _meta() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "issue": 1768,
        "round": "contrastive_read (inline 0-GPU)",
    }


# ---------------------------------------------------------------- data loading


def load_inputs() -> dict:
    registry = json.loads((RESULTS / "arm_registry.json").read_text())
    in_scope = list(registry["in_scope"])
    arms = {a["arm_id"]: a for a in registry["arms"] if a["arm_id"] in set(in_scope)}
    assert len(arms) == len(in_scope), "in_scope arm missing from registry .arms"

    map_change = json.loads((RESULTS / "map_change_summary.json").read_text())["verdicts"]
    direction = json.loads((RESULTS / "direction_reads.json").read_text())["reads"]
    ctx_move = json.loads((RESULTS / "context_movement.json").read_text())["cells"]
    gate = json.loads((RESULTS / "gate_reads.json").read_text())["reads"]
    two_by_two = json.loads((RESULTS / "model_text_2x2/summary.json").read_text())["decomposition"]

    lt_summary = json.loads((RESULTS / "lasttoken_repool/summary.json").read_text())
    lt_pairs = {p["cell"]: p for p in lt_summary["by_position"]["last_prompt"]["d_pairs"]}

    fits, lt_cells = {}, {}
    for arm_id in in_scope:
        for layer in LAYERS:
            cell = f"{arm_id}_L{layer}"
            fits[cell] = json.loads((RESULTS / "fits" / f"{cell}.json").read_text())
        lt = json.loads((RESULTS / "lasttoken_repool/cells" / f"{arm_id}.json").read_text())
        lt_cells[arm_id] = lt["positions"]["last_prompt"]
    return {
        "arms": arms,
        "in_scope": in_scope,
        "map_change": map_change,
        "direction": direction,
        "ctx_move": ctx_move,
        "gate": gate,
        "two_by_two": two_by_two,
        "lt_pairs": lt_pairs,
        "lt_cells": lt_cells,
        "fits": fits,
    }


def build_arm_table(d: dict) -> list[dict]:
    rows = []
    for arm_id in d["in_scope"]:
        meta = d["arms"][arm_id]
        for layer in LAYERS:
            cell = f"{arm_id}_L{layer}"
            mc = d["map_change"][cell]
            lt = d["lt_cells"][arm_id][str(layer)]["map_change"]
            dr = d["direction"][cell]
            tt = d["two_by_two"][arm_id]["layers"][str(layer)]
            fit = d["fits"][cell]
            rows.append(
                {
                    "arm_id": arm_id,
                    "layer": layer,
                    "beh_key": meta["beh_key"],
                    "ctx_key": meta["ctx_key"],
                    "regime": meta["regime"],
                    "seed": meta["seed"],
                    "lr": meta["lr"],
                    "step": meta["step"],
                    "selection_read": meta["selection_read"],
                    "method": meta["method"],
                    # map change, span-mean pooling (round 1)
                    "D": mc["D"],
                    "D_ci95": mc["D_ci95"],
                    "verdict": mc["verdict"],
                    "m0_r2": mc["m0_r2"],
                    "mplus_r2": mc["mplus_r2"],
                    # map change, last-token pooling
                    "D_lasttoken": lt["D"],
                    "D_lasttoken_ci95": lt["D_ci95"],
                    "verdict_lasttoken": lt["verdict"],
                    # doses
                    "dose_fn_norm": tt["norms"]["function"],
                    "dose_fn_perrow": tt["per_row_mean_norm"]["function"],
                    "dose_r1_matched_text_shift": fit["decomposition_tf"]["mean_norm_total"],
                    # write-direction alignments
                    "cos_w_delta": dr["races"]["delta"]["cos_w"],
                    "cos_w_delta_tf": dr["races"]["delta"]["cos_w_tf"],
                    "cos_w_rb": dr["races"]["r_B"]["cos_w"],
                    "cos_w_rb_tf": dr["races"]["r_B"]["cos_w_tf"],
                    # context movement + write rank + gate
                    "ctx_move_rel_median": d["ctx_move"][cell]["rel_median"],
                    "write_top1_share": dr["A6_rank"]["top1_var_share"],
                    "write_top1_share_tf": dr["A6_rank_tf"]["top1_var_share"],
                    "write_participation_ratio": dr["A6_rank"]["participation_ratio"],
                    "write_participation_ratio_tf": dr["A6_rank_tf"]["participation_ratio"],
                    "gate_rho_onpolicy": d["gate"][cell]["on_policy"]["spearman_rho"],
                    "gate_rho_matched_text": d["gate"][cell]["matched_text"]["spearman_rho"],
                }
            )
    return rows


# ------------------------------------------------------------ pair enumeration


def enumerate_pairs(arms: dict) -> dict:
    """Group in-scope arms by (method, beh, ctx, seed); pair one con with one po.

    Exact pair = equal lr. Leftover con/po within a group are greedily matched by
    ascending lr as LR-MISMATCHED pairs; anything left is unpaired.
    """
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"con": [], "po": []})
    for a in arms.values():
        groups[(a["method"], a["beh_key"], a["ctx_key"], a["seed"])][a["regime"]].append(a)

    exact, mismatched, unpaired = [], [], []
    for key in sorted(groups, key=str):
        cons = sorted(groups[key]["con"], key=lambda x: x["lr"])
        pos = sorted(groups[key]["po"], key=lambda x: x["lr"])
        used_p: set[str] = set()
        rem_c = []
        for c in cons:
            match = next((p for p in pos if p["arm_id"] not in used_p and p["lr"] == c["lr"]), None)
            if match is not None:
                used_p.add(match["arm_id"])
                exact.append(_pair_record(key, c, match, exact_lr=True))
            else:
                rem_c.append(c)
        rem_p = [p for p in pos if p["arm_id"] not in used_p]
        for c, p in zip(rem_c, rem_p):
            mismatched.append(_pair_record(key, c, p, exact_lr=False))
        leftover = rem_c[len(rem_p) :] + rem_p[len(rem_c) :]
        unpaired.extend(a["arm_id"] for a in leftover)
    return {"exact": exact, "mismatched": mismatched, "unpaired": unpaired}


def _pair_record(key: tuple, con: dict, po: dict, exact_lr: bool) -> dict:
    method, beh, ctx, seed = key
    lr_tag = f"lr{con['lr']:.0e}".replace("e-0", "e") if exact_lr else "lrmix"
    prefix = "" if method == "lora" else "ft:"
    return {
        "pair_id": f"{prefix}{beh}-{ctx}-s{seed}-{lr_tag}",
        "method": method,
        "beh_key": beh,
        "ctx_key": ctx,
        "seed": seed,
        "exact_lr": exact_lr,
        "con_arm": con["arm_id"],
        "po_arm": po["arm_id"],
        "lr_con": con["lr"],
        "lr_po": po["lr"],
        "step_con": con["step"],
        "step_po": po["step"],
        "selection_read_con": con["selection_read"],
        "selection_read_po": po["selection_read"],
    }


# ----------------------------------------------------------------- analyses

PAIRED_DVS = [
    ("dose_fn_norm", "weights-carried dose (2x2 function-effect norm)"),
    ("dose_r1_matched_text_shift", "round-1 dose (matched-text answer shift)"),
    ("D", "map-change D (span-mean)"),
    ("D_lasttoken", "map-change D (last-token)"),
    ("D_resid", "D residualized on dose (rank space)"),
    ("cos_w_delta_tf", "cos(w-hat, delta), matched text"),
    ("cos_w_rb_tf", "cos(w-hat, r_B), matched text"),
    ("cos_w_delta", "cos(w-hat, delta), on-policy"),
    ("cos_w_rb", "cos(w-hat, r_B), on-policy"),
    ("ctx_move_rel_median", "relative context movement (median)"),
    ("write_top1_share", "write top-1 SVD share (on-policy)"),
    ("write_top1_share_tf", "write top-1 SVD share (matched text)"),
    ("gate_rho_onpolicy", "gate Spearman rho (on-policy)"),
    ("gate_rho_matched_text", "gate Spearman rho (matched text)"),
]


def rank_residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of rank(y) on rank(x) via OLS on ranks.

    Rank space because D takes negative values in this battery (log-log undefined) and
    the body's D-vs-dose claim is itself a Spearman; a linear fit on ranks is the
    matching monotone-robust residualization.
    """
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    slope, intercept = np.polyfit(rx, ry, 1)
    return ry - (slope * rx + intercept)


def paired_analyses(table: list[dict], pairs: dict) -> dict:
    by_cell = {(r["arm_id"], r["layer"]): r for r in table}
    out: dict = {"per_layer": {}}
    for layer in LAYERS:
        layer_rows = [r for r in table if r["layer"] == layer]
        # pooled Spearman(D, dose) sanity checks + rank residualization over all 72 arms
        spearman = {}
        for dose_key in ("dose_fn_norm", "dose_r1_matched_text_shift", "dose_fn_perrow"):
            for d_key in ("D", "D_lasttoken"):
                rho, p = stats.spearmanr(
                    [r[dose_key] for r in layer_rows], [r[d_key] for r in layer_rows]
                )
                spearman[f"{d_key}__vs__{dose_key}"] = {"rho": rho, "p": p, "n": len(layer_rows)}
        resid = rank_residualize(
            np.array([r["D"] for r in layer_rows]),
            np.array([r["dose_fn_norm"] for r in layer_rows]),
        )
        resid_r1 = rank_residualize(
            np.array([r["D"] for r in layer_rows]),
            np.array([r["dose_r1_matched_text_shift"] for r in layer_rows]),
        )
        for r, v, v2 in zip(layer_rows, resid, resid_r1):
            by_cell[(r["arm_id"], r["layer"])]["D_resid"] = float(v)
            by_cell[(r["arm_id"], r["layer"])]["D_resid_r1dose"] = float(v2)

        contrasts = {}
        for kind in ("exact", "mismatched"):
            per_pair = []
            for pr in pairs[kind]:
                con = by_cell[(pr["con_arm"], layer)]
                po = by_cell[(pr["po_arm"], layer)]
                rec = {**pr}
                for dv, _ in PAIRED_DVS:
                    rec[f"d_{dv}"] = con[dv] - po[dv]
                rec["d_D_resid_r1dose"] = con["D_resid_r1dose"] - po["D_resid_r1dose"]
                rec["verdict_con"] = con["verdict"]
                rec["verdict_po"] = po["verdict"]
                rec["verdict_lasttoken_con"] = con["verdict_lasttoken"]
                rec["verdict_lasttoken_po"] = po["verdict_lasttoken"]
                per_pair.append(rec)
            contrasts[kind] = {
                "per_pair": per_pair,
                "stats": _contrast_stats(per_pair),
            }
        flips = {
            kind: [
                {
                    "pair_id": p["pair_id"],
                    "span_mean": (p["verdict_con"], p["verdict_po"]),
                    "last_token": (p["verdict_lasttoken_con"], p["verdict_lasttoken_po"]),
                    "span_mean_flip": p["verdict_con"] != p["verdict_po"],
                    "last_token_flip": p["verdict_lasttoken_con"] != p["verdict_lasttoken_po"],
                }
                for p in contrasts[kind]["per_pair"]
                if p["verdict_con"] != p["verdict_po"]
                or p["verdict_lasttoken_con"] != p["verdict_lasttoken_po"]
            ]
            for kind in ("exact", "mismatched")
        }
        out["per_layer"][str(layer)] = {
            "pooled_spearman_D_vs_dose": spearman,
            "contrasts": contrasts,
            "verdict_flips_within_pairs": flips,
        }
    return out


def _contrast_stats(per_pair: list[dict]) -> dict:
    dvs = [f"d_{dv}" for dv, _ in PAIRED_DVS] + ["d_D_resid_r1dose"]
    stats_out: dict = {}
    beh_groups = defaultdict(list)
    for p in per_pair:
        beh_groups[p["beh_key"]].append(p)
    for dv in dvs:
        pooled = np.array([p[dv] for p in per_pair])
        entry = {
            "pooled": _sign_wilcoxon(pooled),
            "per_behavior": {
                beh: _sign_wilcoxon(np.array([p[dv] for p in grp]))
                for beh, grp in sorted(beh_groups.items())
            },
        }
        stats_out[dv] = entry
    return stats_out


def _sign_wilcoxon(diffs: np.ndarray) -> dict:
    n = len(diffs)
    rec = {
        "n": n,
        "n_pos": int((diffs > 0).sum()),
        "n_neg": int((diffs < 0).sum()),
        "median": float(np.median(diffs)) if n else None,
        "mean": float(diffs.mean()) if n else None,
    }
    if n >= 6 and not np.allclose(diffs, 0):
        w = stats.wilcoxon(diffs)
        rec["wilcoxon_stat"], rec["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
    else:
        rec["wilcoxon_p"] = None  # n < 6: sign counts only
    return rec


# ----------------------------------------------------------------- figures


def _behavior_colors() -> dict:
    pal = paper_palette(4)
    return dict(zip(("cas", "imp", "syc", "mk"), pal))


def fig_dose_d_paired(table: list[dict], pairs: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = _behavior_colors()
    rows = {r["arm_id"]: r for r in table if r["layer"] == PRIMARY_LAYER}
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        [r["dose_fn_norm"] for r in rows.values()],
        [r["D"] for r in rows.values()],
        s=18,
        color="0.8",
        zorder=1,
    )
    for pr in pairs["exact"]:
        con, po = rows[pr["con_arm"]], rows[pr["po_arm"]]
        c = colors[pr["beh_key"]]
        ax.plot(
            [con["dose_fn_norm"], po["dose_fn_norm"]],
            [con["D"], po["D"]],
            color=c,
            lw=1.1,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter([con["dose_fn_norm"]], [con["D"]], s=42, marker="o", color=c, zorder=3)
        ax.scatter(
            [po["dose_fn_norm"]],
            [po["D"]],
            s=46,
            marker="^",
            facecolors="none",
            edgecolors=c,
            linewidths=1.4,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.axhline(0.0, ls="--", lw=1.0, color="0.45", zorder=0)
    ax.set_xlabel("weights-carried dose: function-effect norm at layer 19 (log scale)")
    ax.set_ylabel("map-change statistic D at layer 19 (span-mean pooling)")
    ax.set_title(
        "Map change vs dose: contrastive-vs-positive-only pairs connected",
        pad=30,
        loc="left",
    )
    handles = [
        Line2D([], [], ls="", marker="o", color=colors[b], label=BEH_NAMES[b])
        for b in ("cas", "imp", "syc", "mk")
    ] + [
        Line2D([], [], ls="", marker="o", color="0.3", label="contrastive (filled)"),
        Line2D(
            [],
            [],
            ls="",
            marker="^",
            markerfacecolor="none",
            markeredgecolor="0.3",
            markeredgewidth=1.4,
            label="positive-only (open)",
        ),
        Line2D([], [], ls="", marker="o", color="0.8", label="all 72 arms"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8)
    savefig_paper(fig, "regime_dose_D_paired", dir=FIGDIR)
    plt.close(fig)


FIG2_DVS = [
    ("d_dose_fn_norm", "dose\n(function-effect norm)"),
    ("d_D", "map-change D\n(span-mean)"),
    ("d_D_resid", "D residual\n(rank, dose-adjusted)"),
    ("d_cos_w_delta_tf", "cos(write, delta)\nmatched text"),
    ("d_cos_w_rb_tf", "cos(write, r_B)\nmatched text"),
    ("d_ctx_move_rel_median", "context movement\n(relative, median)"),
    ("d_write_top1_share", "write top-1\nSVD share"),
    ("d_gate_rho_matched_text", "gate rho\nmatched text"),
]


def fig_paired_contrasts(analysis: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = _behavior_colors()
    layer = analysis["per_layer"][str(PRIMARY_LAYER)]
    exact = layer["contrasts"]["exact"]["per_pair"]
    mism = layer["contrasts"]["mismatched"]["per_pair"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.2))
    for ax, (dv, label) in zip(axes.ravel(), FIG2_DVS):
        for i, beh in enumerate(("cas", "imp", "syc", "mk")):
            vals = [p[dv] for p in exact if p["beh_key"] == beh]
            x = i + rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(x, vals, s=26, color=colors[beh], zorder=3)
            mvals = [p[dv] for p in mism if p["beh_key"] == beh]
            xm = i + rng.uniform(-0.16, 0.16, size=len(mvals))
            ax.scatter(
                xm,
                mvals,
                s=30,
                facecolors="none",
                edgecolors=colors[beh],
                linewidths=1.3,
                zorder=3,
            )
        ax.axhline(0.0, ls="--", lw=1.0, color="0.45", zorder=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["casual", "impolite", "sycoph.", "marker"], fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "Paired regime differences (contrastive minus positive-only) at layer 19",
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    handles = [
        Line2D([], [], ls="", marker="o", color="0.3", label="exact-LR pair (filled)"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.3",
            markeredgewidth=1.3,
            label="LR-mismatched pair (open)",
        ),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=True, fontsize=8, ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    savefig_paper(fig, "regime_paired_contrasts", dir=FIGDIR)
    plt.close(fig)


# ------------------------------------------------------------------- main


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_inputs()
    table = build_arm_table(d)
    pairs = enumerate_pairs(d["arms"])
    analysis = paired_analyses(table, pairs)  # adds D_resid columns to table rows

    enumeration = {
        "n_arms": len(d["in_scope"]),
        "n_arms_by_regime": {
            reg: sum(1 for a in d["arms"].values() if a["regime"] == reg) for reg in ("con", "po")
        },
        "n_exact_pairs": len(pairs["exact"]),
        "n_lr_mismatched_pairs": len(pairs["mismatched"]),
        "unpaired_arms": pairs["unpaired"],
        "exact_pairs_by_family": _count_by(pairs["exact"], ("method", "beh_key")),
        "mismatched_pairs_by_family": _count_by(pairs["mismatched"], ("method", "beh_key")),
    }

    summary = {
        "question": (
            "Does the training regime (contrastive negatives vs positive-only) change dose, "
            "map change D beyond dose, write-direction alignment, context movement, write "
            "rank, or the base-geometry gate, in regime-matched arm pairs?"
        ),
        "pair_enumeration": enumeration,
        "dose_definitions": {
            "primary": "model_text_2x2 norms.function per layer: ||mean over rows of "
            "v+(base text) - v0(base text)|| (norm of the mean function-effect vector)",
            "secondary": "dose_fn_perrow (mean per-row norm of the function effect) and "
            "dose_r1_matched_text_shift (round-1 figure dose: fits decomposition_tf."
            "mean_norm_total, the matched-text answer-state shift)",
            "note": "the round-1 body rho~0.98 claim was computed against the round-1 "
            "matched-text shift; pooled Spearman is reported against all three doses",
        },
        "residualization": (
            "OLS residual of rank(D) on rank(dose), pooled over all 72 arms per layer. Rank "
            "space because D takes negative values (log-log fit undefined) and the body claim "
            "is itself a Spearman; monotone-robust."
        ),
        "confounds": [
            "Observational: the regime changes installed dose; residualization is rank "
            "association, not causal dose-matching.",
            "Selection-on-install: checkpoints were band-selected on the install read, so "
            "step and selection_read differ within pairs (second confound channel).",
        ],
        "analysis": analysis,
        "figures": {
            "regime_dose_D_paired": "D (span-mean) vs dose (function-effect norm, log x) at "
            "layer 19; all 72 arms gray; exact pairs connected, color=behavior, filled "
            "circle=contrastive, open triangle=positive-only",
            "regime_paired_contrasts": "per-DV paired differences (contrastive minus "
            "positive-only) at layer 19; filled=exact-LR pairs, open=LR-mismatched; "
            "pair-id-to-arm mapping lives in analysis.per_layer.*.contrasts.*.per_pair",
        },
        **_meta(),
    }

    (OUT / "arm_table.json").write_text(json.dumps({"rows": table, **_meta()}, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    fig_dose_d_paired(table, pairs)
    fig_paired_contrasts(analysis)

    p19 = analysis["per_layer"][str(PRIMARY_LAYER)]
    print(json.dumps(enumeration, indent=1))
    for key, v in p19["pooled_spearman_D_vs_dose"].items():
        print(f"spearman L19 {key}: rho={v['rho']:.4f} p={v['p']:.2e}")
    for dv in (
        "d_dose_fn_norm",
        "d_D",
        "d_D_resid",
        "d_cos_w_delta_tf",
        "d_cos_w_rb_tf",
        "d_ctx_move_rel_median",
        "d_write_top1_share",
        "d_gate_rho_matched_text",
    ):
        s = p19["contrasts"]["exact"]["stats"][dv]["pooled"]
        print(
            f"L19 exact-pair {dv}: {s['n_pos']}+/{s['n_neg']}- of {s['n']}, "
            f"median={s['median']:+.4f}, wilcoxon_p={s['wilcoxon_p']}"
        )
    print("flips (exact pairs, L19):", json.dumps(p19["verdict_flips_within_pairs"]["exact"]))
    print("wrote", OUT / "summary.json")


def _count_by(records: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict[str, int] = defaultdict(int)
    for r in records:
        out["/".join(str(r[k]) for k in keys)] += 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    main()
