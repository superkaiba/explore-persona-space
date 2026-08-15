#!/usr/bin/env python
"""Writeup-v2 figures for #2054 (user spec revision of 2026-08-14).

Render-only over banked artifacts (0 GPU-h, no refits) — supersedes NONE of the
earlier figure scripts (`issue2054_writeup_result12_figs.py` renders the
2026-08-13 spec; the user changed the figure specifications, so these render
fresh under `figures/issue_2054/writeup_v2/`).

Inputs
------
- /tmp/issue2054_ladder_rows_merged.json          (scripts/issue2054_fetch_ladder_rows.py)
- eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json
- data/issue_2054/ladderstage/issue2054_lattice/pool_rungs/percell_rungs/*__context.json
- eval_results/issue_2054/specialization_ladder/loco_pooled.json           (speaker LOCO)
- eval_results/issue_2054/specialization_ladder/loco_framing_pooled.json   (framing LOCO)

Transfer-tier vocabulary (user's 4 tiers, spec of 2026-08-14):
  Direct          = 1_direct            (source map applied unchanged)
  Cross transfer  = banked paired cross-render fit
                    (analyzer_companions/cross_render_fit*.json: ridge
                    source-setting context -> target-setting answer, fit
                    DIRECTLY on the paired conversations — the pair-supervised
                    predictability bound, not a ladder rung)
  Bias only       = 4_bias_refit        (bias refit on the target train fold)
  Rotation + bias = 6_rotation          (Procrustes on centered clouds + train means)
The cross tier fits a d x d map on the paired train rows, so at pairs whose
intersection has 0.8 x n < d = 3,584 it is descriptive-only and drawn HOLLOW
(explained in the writeup prose, never on the canvas). The retired re-map rungs
(7_ctx_reparam / 8_ans_reparam) stay banked in the ladder rows but are no longer
plotted. `cross_pair_nulls.json` (scripts/issue2054_remap_pair_nulls.py
--mode cross) adds the matched-capacity shuffled-pair null band at that tier.

Pooled-map tiers (figures 3/7):
  Pooled direct   = pooled_tier_ladder r2.pooled_direct  (banked pool_rungs m0)
  + per-cell bias = pool_rungs percell m1 (fold-mean)
  Rotation + bias = pooled_tier_ladder r2.rotation_bias
  Own map         = pooled_tier_ladder r2.own_map (the cell's banked ceiling)
(The pooled map is fit across all settings jointly, so it has no paired
cross-render analogue; that column is absent by construction, not dropped.)

Usage:
  uv run python scripts/issue2054_fetch_ladder_rows.py   # if /tmp rows are absent
  uv run python scripts/issue2054_writeup_v2_figs.py [--skip-loco-framing]
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parent.parent
ROWS_PATH = Path("/tmp/issue2054_ladder_rows_merged.json")
PTL_PATH = REPO / "eval_results/issue_2054/specialization_ladder/pooled_tier_ladder.json"
POOL_RUNGS_DIR = REPO / "data/issue_2054/ladderstage/issue2054_lattice/pool_rungs/percell_rungs"
LOCO_SPEAKER_PATH = REPO / "eval_results/issue_2054/specialization_ladder/loco_pooled.json"
LOCO_FRAMING_PATH = REPO / "eval_results/issue_2054/specialization_ladder/loco_framing_pooled.json"
NULLS_PATH = REPO / "eval_results/issue_2054/specialization_ladder/cross_pair_nulls.json"
COMPANIONS_DIR = REPO / "eval_results/issue_2054/analyzer_companions"
P1345_JUDGE = REPO / "eval_results/issue_1345/judge_legs/judge_legs_summary.json"
OUT_DIR = REPO / "figures/issue_2054/writeup_v2"

ASSIST = "conversation_paired_stories_assistant"
INSTRUCT = "qwen2.5-7b-instruct"
BASE = "qwen2.5-7b"
D_AMBIENT = 3584
WELL_POSED_FLOOR = 4480  # 0.8 x 4480 = 3584 = d

CROSS_TIER = "cross"  # sentinel: value comes from the banked cross-render fit
TIERS = ["1_direct", CROSS_TIER, "4_bias_refit", "6_rotation"]
TIER_LABELS = ["Direct", "Cross\ntransfer", "Bias\nonly", "Rotation\n+ bias"]
CROSS_TIER_X = TIERS.index(CROSS_TIER)

POOL_TIERS = ["pooled_direct", "bias", "rotation_bias"]
POOL_TIER_LABELS = ["Pooled\ndirect", "+ per-cell\nbias", "Rotation\n+ bias"]

FRAMING_COLOR = {
    "chat": "#444444",
    "bare_text": "#1f77b4",
    "bare_label": "#ff7f0e",
    "attrib_quoted": "#2ca02c",
}
FRAMING_LABEL = {
    "chat": "chat template",
    "bare_text": "bare text",
    "bare_label": "story, bare label",
    "attrib_quoted": "story, attributed quote",
}
# Sequential ramp keyed to judge-scored AI-likeness (darkest = most AI-like),
# reusing the #1345/#2054 convention so one color = one meaning across figures.
CHAR_COLOR = {
    "helios": "#08306B",
    "wren": "#2171B5",
    "vex": "#4292C6",
    "dana": "#9ECAE1",
    "assistant": "#444444",
}
CHAR_LABEL = {
    "wren": "Wren",
    "helios": "HELIOS",
    "dana": "Dana",
    "vex": "Vex",
    "assistant": "Assistant",
}


def load_ai_likeness() -> dict[str, float]:
    """character -> judge-scored AI-likeness (0-100) of its OWN on-policy answers.

    Source: #1345 `judge_legs_summary.json`, leg `ai_likeness`, cells
    `char_<name>_op` (instruct, on-policy). claude-sonnet-4-5, 5 draws per item,
    ~300 items per cell, mean-aggregated with malformed returns dropped.
    """
    d = json.loads(P1345_JUDGE.read_text())
    out: dict[str, float] = {}
    for c in d["legs"]["ai_likeness"]["cells"]:
        name = c["cell"]
        if name.startswith("char_") and name.endswith("_op"):
            out[name[len("char_") : -len("_op")]] = c["pooled"]["mean"]
    return out


def char_labels_with_ai(ail: dict[str, float]) -> dict[str, str]:
    """CHAR_LABEL with the AI-likeness score appended, for legend entries."""
    out = dict(CHAR_LABEL)
    for ch, v in ail.items():
        if ch in out:
            out[ch] = f"{out[ch]} \u00b7 AI-likeness {v:.0f}"
    return out


def parse_cell(key: str) -> dict:
    """`<identity>__<condition>__<framing>__<model>` -> dict (cell_c aware)."""
    ident, cond, framing, model = key.split("__")
    return {"identity": ident, "condition": cond, "framing": framing, "model": model}


def load_rows() -> list[dict]:
    rows = json.loads(ROWS_PATH.read_text())
    for r in rows:
        r["s"] = parse_cell(r["src"])
        r["t"] = parse_cell(r["tgt"])
    return [r for r in rows if r["arm"] == "context"]


def find_row(rows: list[dict], **kw) -> dict | None:
    """Unique row whose src/tgt fields match `s_*` / `t_*` kwargs."""
    hits = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            side, field = k.split("_", 1)
            if r[side][field] != v:
                ok = False
                break
        if ok:
            hits.append(r)
    if not hits:
        return None
    assert len(hits) == 1, f"non-unique row for {kw}: {[(h['src'], h['tgt']) for h in hits]}"
    return hits[0]


def load_nulls() -> dict[tuple[str, str], dict]:
    """(src, tgt) -> shuffled-pair null summary; empty when not yet computed."""
    if not NULLS_PATH.exists():
        return {}
    payload = json.loads(NULLS_PATH.read_text())
    return {(u["src"], u["tgt"]): u["mean"] for u in payload["units"]}


def load_cross() -> dict[tuple[str, str], dict]:
    """(src, tgt) -> banked paired cross-render fit: {"r2", "n"}.

    Cross transfer = ridge fit source-setting CONTEXT -> target-setting ANSWER
    directly on the paired conversations (analyzer_companions/cross_render_fit*),
    i.e. the pair-supervised predictability bound. Its d x d fit carries the same
    well-posedness condition as any ambient map, so `n` drives the hollow marker.
    """
    out: dict[tuple[str, str], dict] = {}
    main = json.loads((COMPANIONS_DIR / "cross_render_fit.json").read_text())
    for c in main["cells"]:
        if c.get("is_identity"):
            continue
        src = f"{ASSIST}__{c['condition']}__chat__{c['model']}"
        tgt = f"{ASSIST}__{c['condition']}__{c['target_form']}__{c['model']}"
        out[(src, tgt)] = {"r2": c["cross_render_r2"], "n": c["n_intersection"]}
    for shard in ("2a", "2b"):
        for slug in ("qwen25-7b-instruct", "qwen25-7b"):
            path = COMPANIONS_DIR / f"cross_render_fit_characters.shard__{shard}__{slug}.json"
            for c in json.loads(path.read_text())["cells"]:
                src = f"{ASSIST}__{c['source_condition']}__{c['source_form']}__{c['model']}"
                tgt = f"{c['character']}__{c['target_condition']}__{c['target_form']}__{c['model']}"
                out[(src, tgt)] = {"r2": c["cross_render_r2"], "n": c["n_pair"]}
    return out


def _draw_null_bands(ax, null_means: list[dict]) -> bool:
    """Shade the p95 range of the shuffled-pair nulls at the cross-transfer tier.

    One band at x = CROSS_TIER_X spanning min..max of the panel's per-pair null
    p95 values. Returns True if drawn.
    """
    vals = [m["null_cross_p95"] for m in null_means if "null_cross_p95" in m]
    if not vals:
        return False
    lo, hi = min(vals), max(vals)
    if hi - lo < 0.008:  # keep a visible sliver when the nulls coincide
        pad = (0.008 - (hi - lo)) / 2
        lo, hi = lo - pad, hi + pad
    ax.fill_between(
        [CROSS_TIER_X - 0.3, CROSS_TIER_X + 0.3], lo, hi, color="#8a8a8a", alpha=0.30, zorder=1
    )
    return True


NULL_BAND_HANDLE = Patch(
    facecolor="#8a8a8a", alpha=0.30, label="shuffled-pair null (p95 range, cross tier)"
)


def tier_values(row: dict, cross: dict | None = None) -> tuple[list[float], list[bool]]:
    """(values per TIERS, filled-marker flags — hollow when a d x d fit is under-determined).

    The cross tier is looked up in `cross` (banked paired fit); a pair with no
    banked cross value plots as NaN (gap in the line), never a silent zero.
    """
    entry = (cross or {}).get((row["src"], row["tgt"]))
    vals: list[float] = []
    filled: list[bool] = []
    for t in TIERS:
        if t == CROSS_TIER:
            vals.append(float("nan") if entry is None else entry["r2"])
            filled.append(entry is not None and entry["n"] >= WELL_POSED_FLOOR)
        else:
            vals.append(row["rungs"][t])
            filled.append(True)
    return vals, filled


def _plot_tier_line(ax, vals, filled, color, label=None, ls="-"):
    x = range(len(vals))
    ax.plot(x, vals, ls, color=color, label=label, lw=1.8, zorder=3)
    for xi, (v, f) in enumerate(zip(vals, filled)):
        ax.plot(
            xi,
            v,
            "o",
            color=color,
            markerfacecolor=color if f else "white",
            markeredgecolor=color,
            markersize=5.5,
            zorder=4,
        )


def _tier_axes(ax, labels, title, ylabel=None, zero_line=False):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")


# ---------------------------------------------------------------- ladder figs


def fig_tier_single(
    stem: str,
    pair_fn,
    series: list[str],
    color_map: dict,
    label_map: dict,
    cond_key: str,
    title: str,
    src_ceiling_fn=None,
    nulls: dict | None = None,
    cross: dict | None = None,
):
    """Single panel: on-policy instruct absolute held-out R^2 per transfer tier.

    `pair_fn(cond_key, model, series_key) -> row or None` supplies the ladder row.
    Dotted lines are each target's own-map R^2 (series color) and, when
    `src_ceiling_fn` is given, the source map's own R^2 (gray dashed). `nulls`
    adds the gray shuffled-pair null p95 band at the cross tier.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    panel_nulls: list[dict] = []
    for sk in series:
        r = pair_fn(cond_key, INSTRUCT, sk)
        if r is None:
            continue
        vals, filled = tier_values(r, cross)
        _plot_tier_line(ax, vals, filled, color_map[sk], label=label_map[sk])
        ax.axhline(r["ceiling"], color=color_map[sk], lw=1.0, ls=":", alpha=0.8)
        if nulls and (r["src"], r["tgt"]) in nulls:
            panel_nulls.append(nulls[(r["src"], r["tgt"])])
    drew_nulls = _draw_null_bands(ax, panel_nulls)
    if src_ceiling_fn is not None:
        sc = src_ceiling_fn(cond_key, INSTRUCT)
        if sc is not None:
            ax.axhline(sc, color="#888888", lw=1.2, ls="--", alpha=0.9)
    _tier_axes(ax, TIER_LABELS, title, ylabel="held-out $R^2$ (context arm)")
    handles = [
        Line2D([0], [0], color=color_map[sk], marker="o", lw=1.8, label=label_map[sk])
        for sk in series
    ]
    handles.append(
        Line2D([0], [0], color="#333", lw=1.0, ls=":", label="target's own-map $R^2$")
    )
    if src_ceiling_fn is not None:
        handles.append(
            Line2D([0], [0], color="#888888", lw=1.2, ls="--", label="source's own-map $R^2$")
        )
    if drew_nulls:
        handles.append(NULL_BAND_HANDLE)
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# --------------------------------------------------------------- pooled figs


def load_pooled_units() -> dict[str, dict]:
    """cell key -> pooled-tier unit (context arm), with the m1 bias tier joined in."""
    ptl = json.loads(PTL_PATH.read_text())
    units = {u["cell"]: dict(u) for u in ptl["units"] if u["arm"] == "context"}
    for key, u in units.items():
        pr_path = POOL_RUNGS_DIR / f"{key}__context.json"
        pr = json.loads(pr_path.read_text())
        m1 = [f["metrics"]["m1"]["r2"] for f in pr["folds"]]
        u["r2"] = dict(u["r2"])
        u["r2"]["bias"] = sum(m1) / len(m1)
    return units


def fig_pooled_single(
    units: dict[str, dict],
    stem: str,
    cell_fn,
    series: list[str],
    color_map: dict,
    label_map: dict,
    cond: str,
    title: str,
):
    """Single panel: pooled-map tiers vs each setting's own map (on-policy instruct)."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for sk in series:
        u = units.get(cell_fn(cond, INSTRUCT, sk))
        if u is None:
            continue
        vals = [u["r2"][t] for t in POOL_TIERS]
        _plot_tier_line(ax, vals, [True] * len(vals), color_map[sk], label=label_map[sk])
        ax.axhline(u["r2"]["own_map"], color=color_map[sk], lw=1.0, ls=":", alpha=0.8)
    _tier_axes(ax, POOL_TIER_LABELS, title, ylabel="held-out $R^2$ (context arm)")
    handles = [
        Line2D([0], [0], color=color_map[sk], marker="o", lw=1.8, label=label_map[sk])
        for sk in series
    ]
    handles.append(Line2D([0], [0], color="#333", lw=1.0, ls=":", label="own-map $R^2$"))
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# ----------------------------------------------------------------- LOCO figs


def fig_loco_single(
    loco_path: Path,
    units: dict[str, dict],
    stem: str,
    group_field: str,
    groups: list[str],
    group_labels: list[str],
    cell_filter,
    xlabel: str,
    title: str,
):
    """Single panel (on-policy instruct): held-out-group map vs full pool vs own map."""
    loco = json.loads(loco_path.read_text())
    per = [u for u in loco["per_unit"] if u["arm"] == "context" and cell_filter(u)]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    series = [
        ("loco_r2", "#d62728", "fit on all settings but this one, applied frozen"),
        ("full_pool_r2", "#1f77b4", "fit on all settings (incl. this one)"),
        ("ceiling", "#444444", "own-map $R^2$ (fit only on this setting)"),
    ]
    for gi, g in enumerate(groups):
        cells = [
            u
            for u in per
            if u.get(group_field) == g
            and u["condition"] == "on_policy"
            and u["model"].endswith("-instruct")
        ]
        for u in cells:
            ceil = units.get(u["cell"], {}).get("r2", {}).get("own_map")
            for off, (field, color, _lbl) in zip((-0.18, 0.0, 0.18), series):
                val = u["mean"][field] if field != "ceiling" else ceil
                if val is None:
                    continue
                ax.plot(gi + off, val, "o", color=color, markersize=6, zorder=3)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("held-out $R^2$ (context arm)")
    ax.set_xlabel(xlabel)
    handles = [Line2D([0], [0], color=c, marker="o", lw=0, label=lbl) for _f, c, lbl in series]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, stem, dir=OUT_DIR)
    plt.close(fig)


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--skip-loco-framing", action="store_true")
    args = ap.parse_args()

    set_paper_style("blog")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    units = load_pooled_units()
    nulls = load_nulls()
    cross = load_cross()
    ail = load_ai_likeness()
    char_label_ai = char_labels_with_ai(ail)

    story_forms = ["bare_text", "bare_label", "attrib_quoted"]
    # characters ordered most -> least AI-like (judge-scored, own on-policy answers)
    chars = sorted(["wren", "helios", "dana", "vex"], key=lambda c: -ail[c])
    ON = "on_policy"

    # -- Result 1: assistant chat -> assistant other framings -----------------
    def r1_pair(cond, model, framing):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition=cond,
            s_framing="chat",
            s_model=model,
            t_identity=ASSIST,
            t_condition=cond,
            t_framing=framing,
            t_model=model,
        )

    def r1_src_ceiling(cond, model):
        u = units.get(f"{ASSIST}__{cond}__chat__{model}")
        return None if u is None else u["r2"]["own_map"]

    fig_tier_single(
        "r1_framing_tiers",
        r1_pair,
        story_forms,
        FRAMING_COLOR,
        FRAMING_LABEL,
        ON,
        "Transferring the assistant's chat-template map to other framings\n"
        "(on-policy, instruct)",
        src_ceiling_fn=r1_src_ceiling,
        nulls=nulls,
        cross=cross,
    )

    def r1_pool_cell(cond, model, framing):
        return f"{ASSIST}__{cond}__{framing}__{model}"

    fig_pooled_single(
        units,
        "r1_pooled_vs_own",
        r1_pool_cell,
        ["chat", "bare_text", "bare_label", "attrib_quoted"],
        FRAMING_COLOR,
        FRAMING_LABEL,
        ON,
        "One map fit on all settings, applied to each framing\n(on-policy, instruct)",
    )

    if not args.skip_loco_framing and LOCO_FRAMING_PATH.exists():
        fig_loco_single(
            LOCO_FRAMING_PATH,
            units,
            "r1_loco_framings",
            "group",
            ["chat", "bare_text", "bare_label", "attrib_quoted"],
            [
                FRAMING_LABEL[f].replace(", ", ",\n")
                for f in ["chat", "bare_text", "bare_label", "attrib_quoted"]
            ],
            cell_filter=lambda u: u["identity"] == ASSIST,
            xlabel="held-out framing (assistant cells shown)",
            title="Map fit on all framings but one, applied to the held-out framing\n"
            "(on-policy, instruct)",
        )

    # -- Result 2: assistant story (bare label) -> characters -----------------
    def r2_pair(cond, model, ch):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition=cond,
            s_framing="bare_label",
            s_model=model,
            t_identity=f"char_{ch}",
            t_condition=cond,
            t_framing="bare_label",
            t_model=model,
        )

    def r2_src_ceiling(cond, model):
        u = units.get(f"{ASSIST}__{cond}__bare_label__{model}")
        return None if u is None else u["r2"]["own_map"]

    fig_tier_single(
        "r2_character_tiers",
        r2_pair,
        chars,
        CHAR_COLOR,
        char_label_ai,
        ON,
        "Transferring the assistant's map to each character, same story framing\n"
        "(on-policy, instruct)",
        src_ceiling_fn=r2_src_ceiling,
        nulls=nulls,
        cross=cross,
    )

    def r2_pool_cell(cond, model, ch):
        return f"char_{ch}__{cond}__bare_label__{model}"

    fig_pooled_single(
        units,
        "r2_pooled_vs_own_characters",
        r2_pool_cell,
        chars,
        CHAR_COLOR,
        char_label_ai,
        ON,
        "One map fit on all settings, applied to each character\n(on-policy, instruct)",
    )

    fig_loco_single(
        LOCO_SPEAKER_PATH,
        units,
        "r2_loco_characters",
        "speaker",
        ["assistant"] + chars,
        [CHAR_LABEL["assistant"]] + [char_label_ai[c].replace(" \u00b7 ", "\n") for c in chars],
        cell_filter=lambda u: u["framing"] == "bare_label",
        xlabel="held-out character (bare-label story cells shown)",
        title="Map fit on all characters but one, applied to the held-out character\n"
        "(on-policy, instruct)",
    )

    # -- Result 3: assistant chat (inserted source) -> characters in story ----
    def r3_pair(tcond, model, ch):
        return find_row(
            rows,
            s_identity=ASSIST,
            s_condition="inserted",
            s_framing="chat",
            s_model=model,
            t_identity=f"char_{ch}",
            t_condition=tcond,
            t_framing="bare_label",
            t_model=model,
        )

    def r3_src_ceiling(tcond, model):
        u = units.get(f"{ASSIST}__inserted__chat__{model}")
        return None if u is None else u["r2"]["own_map"]

    fig_tier_single(
        "r3_chat_to_characters",
        r3_pair,
        chars,
        CHAR_COLOR,
        char_label_ai,
        ON,
        "Transferring the assistant's chat-template map to characters in story\n"
        "(on-policy targets, instruct)",
        src_ceiling_fn=r3_src_ceiling,
        nulls=nulls,
        cross=cross,
    )

    print(f"[writeup-v2-figs] wrote figures under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
