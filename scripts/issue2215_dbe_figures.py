#!/usr/bin/env python3
"""Issue #2215 ``discrimination-battery-expansion`` — canonical figure suite (plan v6 §6).

Unit 4 of the pre-split build. Presentation-only: every number is read off the
C'-phase eval JSONs written by ``scripts/issue2215_dbe_analysis.py`` into
``--in-dir`` (plus the COMMITTED parent-battery JSON for the joint figure —
never recomputed); no tensor loads, no fits, no API calls.

Canonical set (paper-plots conventions: ``set_paper_style("blog")``,
colorblind-safe ``paper_palette``, one color = one meaning, NO caption blocks /
``fig.text`` on canvas — axes + ticks + legend + titles only):

* HERO ``dbe_hero_pertype_2afc`` — per-type paired 2AFC at the registered
  layer, 4 predictor arms + identity+bias, per-type shuffled-pair null bands,
  worst→best, benchmark cells dagger-tagged.
* JOINT ``dbe_joint_taxonomy_48`` — all 48 cells (39 parent from the banked
  ``eval_results/issue_2215/dv3_map_discrimination.json`` + the new types),
  #779-ce arm, provenance separated visually, hot-spot (non-discriminating)
  cells marked.
* EXPLORATORY DUMP — per-pair margin scatter per type; per-type R²/cosine +
  retrieval bars; the H3 2AFC-vs-retrieval dissociation panel; per-type slot
  gains + the registered DiD; carrier transfer (incl. the P2 polarity-grouped
  sentiment read); length-delta covariate; pooling-twin deltas; DV1/DV2 shift
  geometry; CJK recount; parent-fit constant-offset companion; union-pool
  retrieval.

The analysis driver's 4 in-driver QUICK-LOOK figures keep their own distinct
stems (``dbe_percell_2afc`` / ``dbe_h3_dissociation`` / ``dbe_did_slot_gains``
/ ``dbe_joint_taxonomy``) — asserted disjoint from this suite's registry, so
no filename collision exists and the driver stays untouched.

Panels whose input JSONs are absent (smoke/selftest grain, offline parent
stores) are SKIPPED with a recorded reason in
``<figures-dir>/dbe_figures_manifest.json`` — never silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE matplotlib import (shared-VM env caps; #1739 convention)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue2215 import bank_dbe as DBE  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2215.dbe.figures")

_REPO_ROOT = Path(__file__).resolve().parent.parent

ROUND = "discrimination-battery-expansion"
DEFAULT_IN_DIR = _REPO_ROOT / "eval_results" / "issue_2215" / ROUND
DEFAULT_FIGURES_DIR = _REPO_ROOT / "figures" / "issue_2215"
DEFAULT_PARENT_JSON = _REPO_ROOT / "eval_results" / "issue_2215" / "dv3_map_discrimination.json"

POOL_PRIMARY = "tail"
METRIC_PRIMARY = "cosine"
KNN_KS = (1, 5, 10)

# Arm order + plain-English legend labels (plan §5 condition names).
ARM_ORDER = ("779ce", "1738pe", "1738ce", "idbias_ce", "idbias_pe")
ARM_LABELS = {
    "779ce": "ctx-end map (#779 fit)",
    "1738pe": "prefix-end map (#1738 fit)",
    "1738ce": "ctx-end map, matched fit (#1738)",
    "idbias_ce": "identity+bias (ctx-end)",
    "idbias_pe": "identity+bias (prefix-end)",
}
_COLORS = paper_palette(8)
ARM_COLORS = {arm: _COLORS[i] for i, arm in enumerate(ARM_ORDER)}
COLOR_OWN = _COLORS[5]  # carrier-transfer: own-pair accuracy
COLOR_CROSS = _COLORS[6]  # carrier-transfer: cross-carrier/item accuracy
COLOR_PARENT_FIT = _COLORS[7]  # parent-fit constant-offset arm
COLOR_PARENT_PROV = "0.75"  # joint figure: parent-battery provenance
COLOR_SLOT_CE = "0.35"  # DV1/DV2 geometry: ce slot
COLOR_SLOT_PE = "0.62"  # DV1/DV2 geometry: pe slot

# The analysis driver's in-driver quick-look stems — this suite must never
# collide with them (dedupe contract; the driver stays untouched).
QUICKLOOK_STEMS = frozenset(
    {"dbe_percell_2afc", "dbe_h3_dissociation", "dbe_did_slot_gains", "dbe_joint_taxonomy"}
)


# ── small helpers ─────────────────────────────────────────────────────


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode iteration — never ``splitlines()`` (U+2028 shred, #950)."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _at_k(d: dict, k: int) -> float:
    """acc@k / chance@k lookup tolerant of JSON string keys."""
    if str(k) in d:
        return float(d[str(k)])
    return float(d[k])


def _tick(cell: str, benchmark: set[str]) -> str:
    return cell.replace("_", " ") + (" †" if cell in benchmark else "")


def _cell_rec(per_type: dict, cell: str, metric: str = METRIC_PRIMARY) -> dict | None:
    rec = per_type.get(cell)
    if not isinstance(rec, dict):
        return None
    m = rec.get(metric)
    return m if isinstance(m, dict) else None


def _ci_err(acc: float, ci) -> tuple[float, float]:
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2 and all(np.isfinite(ci))):
        return 0.0, 0.0
    return max(acc - ci[0], 0.0), max(ci[1] - acc, 0.0)


def _grouped_bars(ax, cells, arms, acc_of, ci_of, *, width_total: float = 0.8) -> None:
    """Grouped per-cell bars, one group per cell, one bar per arm; NaN bars
    are simply not drawn (degenerate-at-pe cells)."""
    xs = np.arange(len(cells))
    w = width_total / max(len(arms), 1)
    for ai, arm in enumerate(arms):
        vals, lo, hi = [], [], []
        for c in cells:
            a = acc_of(arm, c)
            vals.append(a if a is not None else np.nan)
            e = ci_of(arm, c) if a is not None else (0.0, 0.0)
            lo.append(e[0])
            hi.append(e[1])
        ax.bar(
            xs + (ai - (len(arms) - 1) / 2) * w,
            vals,
            width=w,
            yerr=[lo, hi],
            capsize=1.5,
            color=ARM_COLORS.get(arm, "0.5"),
            label=ARM_LABELS.get(arm, arm),
        )
    ax.set_xticks(xs)


# ── input loading ─────────────────────────────────────────────────────


def load_inputs(in_dir: Path, parent_json: Path) -> dict:
    """Load every eval JSON the panels read; optional inputs load as None
    (their panels skip with a recorded reason)."""
    dv3_path = in_dir / "dv3_dbe_map_discrimination.json"
    assert dv3_path.exists(), f"{dv3_path} missing — run the C' analysis driver first"
    dv3 = json.loads(dv3_path.read_text())
    layer = int(dv3["meta"]["primary_layer"])

    def _opt(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    perpair_path = in_dir / "perpair" / "dv3_dbe_pairs.jsonl"
    ctx = {
        "in_dir": in_dir,
        "dv3": dv3,
        "layer": layer,
        "reg_key": f"779ce|L{layer}|{POOL_PRIMARY}",
        "per_config": dv3["per_config"],
        "s1": dv3.get("registered_dbe", {}).get("per_type_arm_rows", {}),
        "hyps": dv3.get("hypotheses", {}),
        "perpair": _read_jsonl(perpair_path) if perpair_path.exists() else None,
        "dv1": _opt(in_dir / "dv1_dbe_context_shift.json"),
        "dv2": _opt(in_dir / "dv2_dbe_answer_shift.json"),
        "explor": _opt(in_dir / "exploratory_dbe.json"),
        "parent": _opt(parent_json),
        "parent_json_path": parent_json,
        "benchmark": set(DBE.BENCHMARK_TYPES),
    }
    reg = ctx["per_config"].get(ctx["reg_key"])
    assert reg is not None, (ctx["reg_key"], sorted(ctx["per_config"]))
    ctx["cells"] = sorted(
        reg["per_type"],
        key=lambda c: (_cell_rec(reg["per_type"], c) or {"acc": 0.0})["acc"],
    )
    return ctx


def _pt(ctx: dict, arm: str) -> dict:
    """per_type table for one arm at the registered (layer, pooling)."""
    rec = ctx["per_config"].get(f"{arm}|L{ctx['layer']}|{POOL_PRIMARY}")
    return rec["per_type"] if rec else {}


def _acc(ctx: dict, arm: str, cell: str) -> float | None:
    rec = _cell_rec(_pt(ctx, arm), cell)
    return float(rec["acc"]) if rec and np.isfinite(rec.get("acc", np.nan)) else None


def _nan_if_none(v: float | None) -> float:
    """Explicit None→NaN for plot arrays: ``v or np.nan`` silently maps a
    LEGITIMATE 0.0 accuracy to NaN (falsy), erasing a real floor bar."""
    return float(v) if v is not None else float("nan")


def _acc_ci(ctx: dict, arm: str, cell: str) -> tuple[float, float]:
    rec = _cell_rec(_pt(ctx, arm), cell)
    if rec is None:
        return 0.0, 0.0
    return _ci_err(rec["acc"], rec.get("acc_ci95_clustered"))


# ── HERO: per-type 2AFC ───────────────────────────────────────────────


def fig_hero_pertype_2afc(ctx: dict, figures_dir: Path) -> str | None:
    cells = ctx["cells"]
    arms = [a for a in ARM_ORDER if f"{a}|L{ctx['layer']}|{POOL_PRIMARY}" in ctx["per_config"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    _grouped_bars(
        ax,
        cells,
        arms,
        lambda a, c: _acc(ctx, a, c),
        lambda a, c: _acc_ci(ctx, a, c),
    )
    for i, c in enumerate(cells):
        rec = _cell_rec(_pt(ctx, "779ce"), c)
        band = (rec or {}).get("null_band")
        if band and all(np.isfinite(band)):
            ax.fill_between([i - 0.45, i + 0.45], band[0], band[1], color="0.88", zorder=0)
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("paired 2AFC accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"New-battery discrimination per type (L{ctx['layer']}, tail, cosine; "
        "† = benchmark cell; gray band = shuffled-pair null)"
    )
    ax.legend(ncols=2, fontsize=8)
    savefig_paper(fig, "dbe_hero_pertype_2afc", dir=str(figures_dir))
    plt.close(fig)
    return None


# ── JOINT: 48-cell taxonomy ───────────────────────────────────────────


def fig_joint_taxonomy_48(ctx: dict, figures_dir: Path) -> str | None:
    if ctx["parent"] is None:
        return f"parent JSON absent ({ctx['parent_json_path']})"
    key = f"779ce|L{ctx['layer']}|{POOL_PRIMARY}"
    parent_pt = ctx["parent"].get("per_config", {}).get(key, {}).get("per_type", {})
    if not parent_pt:
        return f"parent JSON has no {key} per_type table"
    rows: list[tuple[str, float, tuple[float, float], str, bool]] = []
    for cell in parent_pt:
        rec = _cell_rec(parent_pt, cell)
        if rec is None:
            continue
        rows.append(
            (
                cell,
                float(rec["acc"]),
                _ci_err(rec["acc"], rec.get("acc_ci95_clustered")),
                rec.get("verdict", "inconclusive"),
                False,
            )
        )
    new_pt = _pt(ctx, "779ce")
    for cell in new_pt:
        rec = _cell_rec(new_pt, cell)
        if rec is None:
            continue
        rows.append(
            (
                cell,
                float(rec["acc"]),
                _ci_err(rec["acc"], rec.get("acc_ci95_clustered")),
                rec.get("verdict", "inconclusive"),
                True,
            )
        )
    rows.sort(key=lambda r: r[1])
    xs = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.bar(
        xs,
        [r[1] for r in rows],
        yerr=[[r[2][0] for r in rows], [r[2][1] for r in rows]],
        capsize=1.0,
        color=[ARM_COLORS["779ce"] if r[4] else COLOR_PARENT_PROV for r in rows],
    )
    hot = [i for i, r in enumerate(rows) if r[3] != "discriminates"]
    if hot:
        ax.scatter(
            [xs[i] for i in hot],
            [min(rows[i][1] + rows[i][2][1] + 0.05, 1.04) for i in hot],
            marker="v",
            color="black",
            s=18,
            zorder=3,
            label="hot spot (verdict ≠ discriminates)",
        )
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_tick(r[0], ctx["benchmark"] if r[4] else set()) for r in rows], rotation=90, fontsize=6
    )
    ax.set_ylabel("paired 2AFC accuracy (#779 fit)")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        f"Joint taxonomy: parent battery (gray, banked) + new types (color), L{ctx['layer']}"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_PARENT_PROV, label="parent battery (banked)"),
        plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS["779ce"], label="new battery (this round)"),
    ]
    h2, lab2 = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h2, fontsize=8)
    savefig_paper(fig, "dbe_joint_taxonomy_48", dir=str(figures_dir))
    plt.close(fig)
    return None


# ── EXPLORATORY DUMP panels ───────────────────────────────────────────


def fig_perpair_margins(ctx: dict, figures_dir: Path) -> str | None:
    if ctx["perpair"] is None:
        return "perpair/dv3_dbe_pairs.jsonl absent"
    rows = [
        r
        for r in ctx["perpair"]
        if r.get("arm") == "779ce"
        and int(r.get("layer", -1)) == ctx["layer"]
        and r.get("pooling") == POOL_PRIMARY
    ]
    if not rows:
        return "no 779ce per-pair rows at the registered config"
    cells = ctx["cells"]
    ncols = 3
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    for i, cell in enumerate(cells):
        ax = axes[i // ncols][i % ncols]
        rs = by_cell.get(cell, [])
        if rs:
            ax.scatter(
                [r["margin_cos_a"] for r in rs],
                [r["margin_cos_b"] for r in rs],
                s=12,
                alpha=0.7,
                color=ARM_COLORS["779ce"],
            )
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.axvline(0.0, color="0.5", lw=0.7)
        ax.set_title(_tick(cell, ctx["benchmark"]), fontsize=9)
        if i // ncols == nrows - 1:
            ax.set_xlabel("margin, direction a")
        if i % ncols == 0:
            ax.set_ylabel("margin, direction b")
    for j in range(len(cells), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        f"Per-pair cosine margins per type (ctx-end map #779 fit, L{ctx['layer']}, tail)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, "dbe_perpair_margins", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_pertype_r2_retrieval(ctx: dict, figures_dir: Path) -> str | None:
    if not ctx["s1"]:
        return "registered_dbe.per_type_arm_rows absent"
    cells = ctx["cells"]
    arms = [a for a in ARM_ORDER if f"{a}|L{ctx['layer']}|{POOL_PRIMARY}" in ctx["s1"]]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=True)

    def _row(arm: str, cell: str) -> dict | None:
        rec = ctx["s1"][f"{arm}|L{ctx['layer']}|{POOL_PRIMARY}"].get(cell)
        return rec if isinstance(rec, dict) and "r2" in rec else None

    panels = (
        ("transfer R²", lambda rec: rec["r2"]),
        ("mean cosine", lambda rec: rec["mean_cosine"]),
        ("retrieval acc@1 (cosine)", lambda rec: _at_k(rec["retrieval"]["cosine"]["acc_at_k"], 1)),
    )
    chance1 = None
    for cell in cells:
        rec = _row("779ce", cell)
        if rec and rec.get("pool_size"):
            chance1 = 1.0 / rec["pool_size"]
            break
    for pi, (ylab, getter) in enumerate(panels):
        ax = axes[pi]
        _grouped_bars(
            ax,
            cells,
            arms,
            lambda a, c, g=getter: g(_row(a, c)) if _row(a, c) else None,
            lambda a, c: (0.0, 0.0),
        )
        ax.set_ylabel(ylab)
        ax.axhline(0.0, color="0.4", lw=0.8)
        if pi == 2 and chance1 is not None:
            ax.axhline(chance1, color="0.4", lw=0.8, ls="--")
    axes[0].set_title(
        f"Registered per-(type × arm) rows (L{ctx['layer']}, tail; "
        "dashed = retrieval chance 1/pool)"
    )
    axes[0].legend(ncols=2, fontsize=8)
    axes[-1].set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    fig.tight_layout()
    savefig_paper(fig, "dbe_pertype_r2_retrieval", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_2afc_vs_retrieval(ctx: dict, figures_dir: Path) -> str | None:
    hyps = ctx["hyps"]
    inputs = hyps.get("h3_inputs")
    if not inputs or not inputs.get("acc_by_type"):
        return "hypotheses.h3_inputs absent"
    acc_by = inputs["acc_by_type"]
    ret_by = inputs["ret1_by_type"]
    h3 = hyps.get("h3", {})
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for cell in sorted(acc_by):
        is_ref = cell == "refusal_request"
        ax.scatter(
            acc_by[cell],
            ret_by[cell],
            color=ARM_COLORS["1738pe"] if is_ref else ARM_COLORS["779ce"],
            s=70 if is_ref else 34,
            zorder=3 if is_ref else 2,
        )
        ax.annotate(
            _tick(cell, ctx["benchmark"]), (acc_by[cell], ret_by[cell]), fontsize=7, alpha=0.85
        )
    med_acc = h3.get("median_2afc", float(np.median(list(acc_by.values()))))
    med_ret = h3.get("median_retrieval_at1", float(np.median(list(ret_by.values()))))
    ax.axvline(med_acc, color="0.6", lw=0.8, ls=":")
    ax.axhline(med_ret, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("paired 2AFC accuracy (ctx-end map, #779 fit)")
    ax.set_ylabel("retrieval acc@1 (cosine, full new-battery pool)")
    ranks = ""
    if "refusal_rank_2afc" in h3:
        ranks = (
            f" — refusal rank {h3['refusal_rank_2afc']}/{h3['m']} (2AFC), "
            f"{h3['refusal_rank_retrieval']}/{h3['m']} (retrieval)"
        )
    ax.set_title(f"H3: pairwise separability vs exact retrieval{ranks}")
    savefig_paper(fig, "dbe_2afc_vs_retrieval", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_slot_gains_did(ctx: dict, figures_dir: Path) -> str | None:
    hyps = ctx["hyps"]
    did = hyps.get("h2b", {}).get("registered")
    if not did:
        return "hypotheses.h2b.registered absent"
    cells = ctx["cells"]
    gain_arms = (("1738pe", "idbias_pe"), ("1738ce", "idbias_ce"), ("779ce", "idbias_ce"))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [2.4, 1.0]}
    )
    xs = np.arange(len(cells))
    w = 0.8 / len(gain_arms)
    for ai, (arm, base) in enumerate(gain_arms):
        deltas = []
        for c in cells:
            a, b = _acc(ctx, arm, c), _acc(ctx, base, c)
            deltas.append(a - b if (a is not None and b is not None) else np.nan)
        ax1.bar(
            xs + (ai - (len(gain_arms) - 1) / 2) * w,
            deltas,
            width=w,
            color=ARM_COLORS[arm],
            label=f"{ARM_LABELS[arm]} − {ARM_LABELS[base]}",
        )
    ax1.axhline(0.0, color="0.4", lw=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax1.set_ylabel("Δ 2AFC accuracy (fitted − identity+bias)")
    ax1.set_title(f"Per-type fitted-map gain by slot (L{ctx['layer']}, tail, cosine)")
    ax1.legend(fontsize=7)

    desc = hyps.get("h2b", {}).get("descriptive_779ce_variant") or {}
    names = ["pe gain", "ce gain", "DiD (pe − ce)"]
    vals = [did["leg_pe_gain"], did["leg_ce_gain"], did["did"]]
    cis = [did["leg_pe_ci95"], did["leg_ce_ci95"], did["did_ci95"]]
    cols = [ARM_COLORS["1738pe"], ARM_COLORS["1738ce"], "0.35"]
    if desc.get("did") is not None:
        names.append("DiD, descriptive\n(#779-ce leg)")
        vals.append(desc["did"])
        cis.append(desc["did_ci95"])
        cols.append("0.62")
    err = [
        [v - c[0] for v, c in zip(vals, cis, strict=True)],
        [c[1] - v for v, c in zip(vals, cis, strict=True)],
    ]
    ax2.bar(names, vals, yerr=err, capsize=3, color=cols)
    ax2.axhline(0.0, color="0.4", lw=0.8)
    ax2.set_ylabel("Δ 2AFC accuracy")
    ax2.set_title("H2(b) slot DiD (M2-eligible types)")
    ax2.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    savefig_paper(fig, "dbe_slot_gains_did", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_carrier_transfer(ctx: dict, figures_dir: Path) -> str | None:
    reg = ctx["per_config"][ctx["reg_key"]]
    ct = reg.get("carrier_transfer")
    if not ct:
        return "carrier_transfer absent from the registered config"
    labels, own, cross = [], [], []
    for cell in ctx["cells"]:
        rec = ct.get(cell)
        if not isinstance(rec, dict) or "own_pair_acc" not in rec:
            continue
        labels.append(_tick(cell, ctx["benchmark"]))
        own.append(rec["own_pair_acc"])
        cross.append(rec["cross_carrier_acc"])
    pol = (ctx["explor"] or {}).get("sentiment_polarity_transfer")
    if isinstance(pol, dict) and "779ce" in pol and "own_pair_acc" in pol.get("779ce", {}):
        labels.append("user sentiment\n(polarity-grouped, P2)")
        own.append(pol["779ce"]["own_pair_acc"])
        cross.append(pol["779ce"]["cross_item_acc"])
    if not labels:
        return "no carrier-transfer rows with ≥2 valid carriers"
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(xs - 0.2, own, width=0.4, color=COLOR_OWN, label="own pair")
    ax.bar(xs + 0.2, cross, width=0.4, color=COLOR_CROSS, label="cross carrier/item")
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("paired 2AFC accuracy")
    ax.set_title(
        f"Carrier transfer: same value pair scored at other carriers "
        f"(ctx-end map #779 fit, L{ctx['layer']})"
    )
    ax.legend()
    savefig_paper(fig, "dbe_carrier_transfer", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_length_covariate(ctx: dict, figures_dir: Path) -> str | None:
    lc = (ctx["explor"] or {}).get("length_covariate")
    if not lc or "per_type" not in lc:
        return "exploratory length_covariate absent"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for cell, rec in sorted(lc["per_type"].items()):
        acc = _acc(ctx, "779ce", cell)
        if acc is None:
            continue
        ax.scatter(rec["mean_abs_ctx_len_delta"], acc, s=34, color=ARM_COLORS["779ce"])
        ax.annotate(
            _tick(cell, ctx["benchmark"]),
            (rec["mean_abs_ctx_len_delta"], acc),
            fontsize=7,
            alpha=0.85,
        )
    rho = lc.get("spearman_ctxlen_delta_vs_2afc")
    rho_s = f"; Spearman ρ = {rho:.2f}" if isinstance(rho, (int, float)) else ""
    ax.set_xlabel("mean |Δ context length| within pair (tokens)")
    ax.set_ylabel("paired 2AFC accuracy (ctx-end map, #779 fit)")
    ax.set_title(f"Length-delta covariate per type{rho_s}")
    savefig_paper(fig, "dbe_length_covariate", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_pooling_twins(ctx: dict, figures_dir: Path) -> str | None:
    cells = ctx["cells"]
    arms = [a for a in ("779ce", "1738pe", "1738ce") if _pt(ctx, a)]
    span_missing = all(f"{a}|L{ctx['layer']}|span" not in ctx["per_config"] for a in arms)
    if span_missing:
        return "no span-pooling configs present"
    xs = np.arange(len(cells))
    w = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    for ai, arm in enumerate(arms):
        tail_pt = ctx["per_config"].get(f"{arm}|L{ctx['layer']}|tail", {}).get("per_type", {})
        span_pt = ctx["per_config"].get(f"{arm}|L{ctx['layer']}|span", {}).get("per_type", {})
        deltas = []
        for c in cells:
            t_rec, s_rec = _cell_rec(tail_pt, c), _cell_rec(span_pt, c)
            deltas.append(t_rec["acc"] - s_rec["acc"] if (t_rec and s_rec) else np.nan)
        ax.bar(
            xs + (ai - (len(arms) - 1) / 2) * w,
            deltas,
            width=w,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("Δ 2AFC accuracy (tail-incl − span-mean target)")
    ax.set_title(f"Answer-pooling twins per type (L{ctx['layer']}, cosine)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "dbe_pooling_twins", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_dv12_geometry(ctx: dict, figures_dir: Path) -> str | None:
    dv1 = ctx["dv1"]
    if not dv1 or "per_cell" not in dv1:
        return "dv1_dbe_context_shift.json absent"
    cells = [c for c in ctx["cells"] if c in dv1["per_cell"]]
    dv2 = ctx["dv2"] or {}
    has_dv2 = isinstance(dv2.get("per_cell"), dict) and "skipped" not in dv2
    n_panels = 2 if has_dv2 else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10.0, 4.2 * n_panels), squeeze=False)
    ax = axes[0][0]
    xs = np.arange(len(cells))
    for si, (slot, col) in enumerate((("ce", COLOR_SLOT_CE), ("pe", COLOR_SLOT_PE))):
        vals = []
        for c in cells:
            rec = dv1["per_cell"][c].get(slot, {})
            prim = rec.get("primary") or {}
            degen = rec.get("degenerate_at_pe") and slot == "pe"
            vals.append(np.nan if degen else prim.get("ratio", np.nan))
        ax.bar(xs + (si - 0.5) * 0.4, vals, width=0.4, color=col, label=f"{slot} slot")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("‖Δv_C‖ / carrier yardstick")
    ax.set_title(f"DV1 context-shift magnitude per type (L{dv1['meta']['primary_layer']})")
    ax.legend()
    if has_dv2:
        ax2 = axes[1][0]
        vals2, cells2 = [], []
        for c in cells:
            rec = dv2["per_cell"].get(c, {}).get(POOL_PRIMARY, {})
            v = rec.get("noise_normalized_primary")
            if v is not None:
                cells2.append(c)
                vals2.append(v)
        ax2.bar(np.arange(len(cells2)), vals2, color=COLOR_SLOT_CE)
        ax2.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax2.set_xticks(np.arange(len(cells2)))
        ax2.set_xticklabels(
            [_tick(c, ctx["benchmark"]) for c in cells2], rotation=30, ha="right", fontsize=8
        )
        ax2.set_ylabel("‖Δv̄_A‖ / draw-noise yardstick")
        ax2.set_title("DV2 answer-shift magnitude per type (tail pooling)")
    fig.tight_layout()
    savefig_paper(fig, "dbe_dv12_geometry", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_cjk_recount(ctx: dict, figures_dir: Path) -> str | None:
    cj = (ctx["explor"] or {}).get("cjk_recount")
    if not cj or "per_cell_intrusion_frac" not in cj:
        return "exploratory cjk_recount absent/skipped"
    cells = [c for c in ctx["cells"] if c in cj["per_cell_intrusion_frac"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    xs = np.arange(len(cells))
    ax1.bar(xs, [cj["per_cell_intrusion_frac"][c] for c in cells], color=COLOR_SLOT_CE)
    ax1.set_ylabel("CJK-carrying draw fraction")
    ax1.set_title("CJK intrusion per type, and the 2AFC recount excluding intruded draws")
    recount = cj.get("per_type_2afc_cjk_excluded", {}).get("779ce", {})
    reg_v, rec_v = [], []
    for c in cells:
        a = _acc(ctx, "779ce", c)
        reg_v.append(a if a is not None else np.nan)
        rr = recount.get(c)
        rec_v.append(rr["acc"] if isinstance(rr, dict) and "acc" in rr else np.nan)
    ax2.bar(xs - 0.2, reg_v, width=0.4, color=ARM_COLORS["779ce"], label="registered")
    ax2.bar(
        xs + 0.2,
        rec_v,
        width=0.4,
        color=ARM_COLORS["779ce"],
        alpha=0.45,
        label="CJK-excluded recount",
    )
    ax2.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax2.set_ylabel("paired 2AFC accuracy (#779 fit)")
    ax2.legend()
    fig.tight_layout()
    savefig_paper(fig, "dbe_cjk_recount", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_parent_fit_offset(ctx: dict, figures_dir: Path) -> str | None:
    pf = (ctx["explor"] or {}).get("parent_store_reads")
    if not pf or "parent_fit_constant_offset" not in pf:
        return "exploratory parent_store_reads absent/skipped"
    off = pf["parent_fit_constant_offset"]
    cells = [c for c in ctx["cells"] if isinstance(off["per_type_2afc"].get(c), dict)]
    cells = [c for c in cells if "acc" in off["per_type_2afc"][c]]
    if not cells:
        return "parent-fit offset has no per-type rows"
    xs = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.bar(
        xs - 0.2,
        [_nan_if_none(_acc(ctx, "idbias_ce", c)) for c in cells],
        width=0.4,
        color=ARM_COLORS["idbias_ce"],
        label=f"{ARM_LABELS['idbias_ce']} — battery-internal LOTO b",
    )
    ax.bar(
        xs + 0.2,
        [off["per_type_2afc"][c]["acc"] for c in cells],
        width=0.4,
        color=COLOR_PARENT_FIT,
        label=f"identity + parent-fit b (n={off.get('n_parent_train', '?')})",
    )
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_tick(c, ctx["benchmark"]) for c in cells], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("paired 2AFC accuracy")
    ax.set_title(f"Constant-offset companion: battery-internal vs parent-fit b (L{ctx['layer']})")
    ax.legend(fontsize=8)
    savefig_paper(fig, "dbe_parent_fit_offset", dir=str(figures_dir))
    plt.close(fig)
    return None


def fig_union_pool_retrieval(ctx: dict, figures_dir: Path) -> str | None:
    pf = (ctx["explor"] or {}).get("parent_store_reads")
    if not pf or "union_pool_retrieval" not in pf:
        return "exploratory union_pool_retrieval absent/skipped"
    up = pf["union_pool_retrieval"]
    arms = [a for a in ARM_ORDER if a in up["per_arm"]]
    xs = np.arange(len(KNN_KS))
    w = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for ai, arm in enumerate(arms):
        rec = up["per_arm"][arm]
        ax.bar(
            xs + (ai - (len(arms) - 1) / 2) * w,
            [_at_k(rec["acc_at_k"], k) for k in KNN_KS],
            width=w,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    chance_rec = up["per_arm"][arms[0]]
    ax.plot(
        xs,
        [_at_k(chance_rec["chance_at_k"], k) for k in KNN_KS],
        color="0.35",
        lw=1.0,
        ls="--",
        marker="_",
        label="chance k/pool",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"acc@{k}" for k in KNN_KS])
    ax.set_ylabel("retrieval accuracy (cosine)")
    ax.set_title(
        f"Union-pool retrieval: new + parent targets "
        f"(pool = {up.get('pool_size', '?')}, L{ctx['layer']})"
    )
    ax.legend(fontsize=8)
    savefig_paper(fig, "dbe_union_pool_retrieval", dir=str(figures_dir))
    plt.close(fig)
    return None


# ── registry + entrypoint ─────────────────────────────────────────────

FIGURES = {
    "dbe_hero_pertype_2afc": fig_hero_pertype_2afc,
    "dbe_joint_taxonomy_48": fig_joint_taxonomy_48,
    "dbe_perpair_margins": fig_perpair_margins,
    "dbe_pertype_r2_retrieval": fig_pertype_r2_retrieval,
    "dbe_2afc_vs_retrieval": fig_2afc_vs_retrieval,
    "dbe_slot_gains_did": fig_slot_gains_did,
    "dbe_carrier_transfer": fig_carrier_transfer,
    "dbe_length_covariate": fig_length_covariate,
    "dbe_pooling_twins": fig_pooling_twins,
    "dbe_dv12_geometry": fig_dv12_geometry,
    "dbe_cjk_recount": fig_cjk_recount,
    "dbe_parent_fit_offset": fig_parent_fit_offset,
    "dbe_union_pool_retrieval": fig_union_pool_retrieval,
}
assert not (set(FIGURES) & QUICKLOOK_STEMS), "stem collision with the analysis quick-look figures"


def render_all(ctx: dict, figures_dir: Path, only: set[str] | None = None) -> dict:
    """Render every registered panel; a panel with absent inputs SKIPS with a
    recorded reason (never silently); a real bug still raises (fail-loud)."""
    set_paper_style("blog")
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: dict[str, str] = {}
    for stem, fn in FIGURES.items():
        if only and stem not in only:
            continue
        reason = fn(ctx, figures_dir)
        if reason is None:
            written.append(stem)
            logger.info("[figure] %s written", stem)
        else:
            skipped[stem] = reason
            logger.info("[figure] %s SKIPPED — %s", stem, reason)
    manifest = {
        "written": written,
        "skipped": skipped,
        "in_dir": str(ctx["in_dir"]),
        "parent_json": str(ctx["parent_json_path"]),
        "registered_config": f"L{ctx['layer']}|{POOL_PRIMARY}|{METRIC_PRIMARY}",
        "repro": {
            **as_metadata_dict(git_provenance(), phase="figures"),
            "entrypoint": "scripts/issue2215_dbe_figures.py",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    if only is None:
        # Full-registry runs own the manifest; a --only debug re-render must
        # never clobber it with a partial written/skipped listing.
        _write_json(figures_dir / "dbe_figures_manifest.json", manifest)
    else:
        logger.info("[manifest] --only run: dbe_figures_manifest.json left untouched")
    return manifest


def _import_check() -> None:
    """Deferred-import + argparse-attribute completeness check (code-style
    convention). All imports here are module-top; the argcheck assert is the
    load-bearing leg."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    for fn in FIGURES.values():
        assert callable(fn), fn
    assert callable(savefig_paper) and callable(set_paper_style) and callable(paper_palette)
    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2215 dbe canonical figure suite (plan v6 §6; unit 4)",
        epilog="smoke: --in-dir <selftest>/eval --figures-dir /tmp/issue-2215-dbe-smoke/figures",
    )
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR, help="C'-phase eval JSON dir")
    ap.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    ap.add_argument(
        "--parent-json",
        type=Path,
        default=DEFAULT_PARENT_JSON,
        help="banked parent dv3 JSON (joint figure; read-only, never recomputed)",
    )
    ap.add_argument("--only", default=None, help="csv of stems to render (debug)")
    ap.add_argument("--list-figures", action="store_true", help="print the stem registry, exit")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if args.list_figures:
        print("\n".join(FIGURES))
        return 0
    ctx = load_inputs(args.in_dir, args.parent_json)
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    if only:
        unknown = only - set(FIGURES)
        assert not unknown, f"unknown stems: {sorted(unknown)}"
    manifest = render_all(ctx, args.figures_dir, only)
    print(
        json.dumps(
            {"written": manifest["written"], "skipped": manifest["skipped"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
