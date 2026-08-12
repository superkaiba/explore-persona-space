"""Issue #2094 user-chat inline round: slot x layer patching-effectiveness heatmaps.

Reads the committed per-cell F tables (main grid f_cells.jsonl + fu2 span-slot
cells) and renders, for the REPLACE (full-state patch) dose:

- f_act heatmaps  (3 panels: matched_query / matched_prefix / cross)
- f_beh heatmaps  (3 panels, setting-primary rubric; cross = mean of both)
- coherence heatmaps (coherent-draw fraction per cell, with the unpatched
  anchor baseline printed in the suptitle)

If transport tables exist under <eval_root>/transport/, also renders the
banked-map transport heatmap (3 panels: dose x slot@map-layer, mean cosine
between the map-PREDICTED answer-vector shift and the REALIZED shift,
steered arm, non-degenerate rows; each cell annotated n + the donor-null
cosine for the same cell).

Every cell is annotated n_coherent/n_total (rows = pairs; one greedy draw per
pair) and cells with coherent fraction < 90% are marked (visible,
never suppressed). All F means are over coherent, non-degenerate rows only
(the project coherent-only reporting rule).

Usage:
  uv run python scripts/issue2094_userchat_heatmaps.py \
      --eval-root <path to eval_results/issue_2094> --out-dir figures/issue_2094/userchat_heatmaps
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

SLOTS = ("ce", "pe", "cm2", "cm3", "l3j", "qspan", "qtext", "pspan_tmpl", "pspan_text")
SLOT_LABELS = {
    "ce": "context-end",
    "pe": "prefix-end",
    "cm2": "2nd-to-last",
    "cm3": "3rd-to-last",
    "l3j": "last-3 joint",
    "qspan": "query span\n(+template)",
    "qtext": "query text\n(no template)",
    "pspan_tmpl": "prefix span\n(+template)",
    "pspan_text": "prefix span\n(no template)",
}
LAYER_ROWS = [f"L{i}" for i in range(28)] + ["joint_mid", "joint_all"]
SETTINGS = ("matched_query", "matched_prefix", "cross")
SETTING_TITLES = {
    "matched_query": "matched query (prefix transfer)",
    "matched_prefix": "matched prefix (query transfer)",
    "cross": "cross (both differ)",
}
PRIMARY_RUBRIC = {
    "matched_query": ("prefix",),
    "matched_prefix": ("query",),
    "cross": ("prefix", "query"),
}


def load_rows(eval_root: Path, steered: bool = True) -> list[dict]:
    """-> the per-cell rows of the steered arm, or of the shuffled-donor null arm."""
    rels = (
        ("f_metrics/f_cells.jsonl", "f_metrics/fu2/fu2_cells.jsonl")
        if steered
        else ("f_metrics/null_cells.jsonl", "f_metrics/fu2/fu2_null_cells.jsonl")
    )
    rows: list[dict] = []
    for rel in rels:
        p = eval_root / rel
        assert p.exists(), f"missing table: {p}"
        with p.open() as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def load_wellsep(eval_root: Path, min_sep: float = 0.5) -> tuple[set[tuple[str, str]], set[str]]:
    """Return the (pair_id, rubric-kind) and pair-only sets at |anchor separation| >= min_sep.

    The FA-3 convention of ``issue2094_wellsep_bootstrap.py``. Load-bearing for F_beh:
    its denominator IS the anchor separation, so a pair whose floor ~= ceiling (the
    bare<->conversation pairs, separation 0.005-0.221) divides by ~0 and produces
    |F| >> 1. Separation never enters F_act's denominator.
    """
    pair_kind: set[tuple[str, str]] = set()
    with (eval_root / "f_metrics/anchors.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            a = json.loads(line)
            sep = a.get("separation")
            if sep is not None and abs(sep) >= min_sep:
                pair_kind.add((a["pair_id"], a["kind"]))
    assert pair_kind, f"no well-separated anchors at |sep| >= {min_sep} in {eval_root}"
    return pair_kind, {pid for pid, _ in pair_kind}


def beh_value(row: dict, setting: str, ws: set[tuple[str, str]] | None = None) -> float | None:
    vals = []
    for rubric in PRIMARY_RUBRIC[setting]:
        if ws is not None and (row["pair_id"], rubric) not in ws:
            continue
        d = (row.get("f_beh") or {}).get(rubric)
        if d is None or d.get("degenerate_denominator") or d.get("negative_denominator"):
            continue
        # An explicit null f_beh is a DROPPED measurement (incoherent draw suppressed
        # upstream as `excluded_incoherent_raw`, or a missing judge return) — never
        # coerce it to 0.0; skip the (row, rubric) and let n reflect the drop.
        if d.get("f_beh") is None:
            continue
        vals.append(float(d["f_beh"]))
    return float(np.mean(vals)) if vals else None


def aggregate(
    rows: list[dict],
    dose: str,
    ws: set[tuple[str, str]] | None = None,
    ws_any: set[str] | None = None,
    arm: str = "steered",
) -> dict:
    """-> {(setting, slot, layer_row): {f_act, f_beh, cos, trav, coh_frac, n_coh, n_tot}}

    When ``ws``/``ws_any`` are given, restrict to well-separated pairs: rows are gated
    on ``ws_any`` (pair well-separated on >= 1 primary rubric — the FA-3 row-level
    convention) and each rubric's value on ``ws``.

    ``cos`` is the mean per-draw cosine between the realized answer-vector shift and the
    floor->ceiling axis; ``trav`` the mean traversal ratio ||s||/||t||. Together they are
    F_act's polar form: F_act ~ trav * cos, so a cell can post a large F_act EITHER by
    moving the right way (high cos) OR by simply moving a lot (high trav) with only a
    modest component on-axis. cos is the scale-free direction read that separates them.
    Derivation from the banked fields: traversal_ratio = ||s||/||t|| and
    f_act_shared_recordonly = <s,t>/||t||^2, so their ratio is exactly
    <s,t>/(||s|| ||t||). Use f_act_shared_recordonly, NOT f_act -- see _decompose.
    """
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] != arm or r["dose"] != dose or r.get("degenerate_self"):
            continue
        if r["slot"] not in SLOTS or r["layer_variant"] not in LAYER_ROWS:
            continue
        if ws_any is not None and r["pair_id"] not in ws_any:
            continue
        key = (r["setting"], r["slot"], r["layer_variant"])
        buckets[key]["n_tot"].append(1)
        # Pooled steered cap-hit (rollouts that ran into the generation cap and
        # truncated). The fu2 convention labels a family compromised above 2%; those
        # cells carried NO mark before, so a truncation-driven read was indistinguishable
        # from a clean one.
        buckets[key]["caphit"].append(1.0 if r.get("cap_hit") else 0.0)
        if not r["coherent"]:
            continue
        buckets[key]["n_coh"].append(1)
        if r.get("f_act") is not None and not r.get("f_act_degenerate"):
            buckets[key]["f_act"].append(float(r["f_act"]))
        tr, on = r.get("traversal_ratio"), r.get("f_act_shared_recordonly")
        if tr is not None and on is not None and float(tr) > 0:
            buckets[key]["cos"].append(float(on) / float(tr))
            buckets[key]["trav"].append(float(tr))
        bv = beh_value(r, r["setting"], ws=ws)
        if bv is not None:
            buckets[key]["f_beh"].append(bv)
    out = {}
    for key, b in buckets.items():
        n_tot, n_coh = len(b["n_tot"]), len(b["n_coh"])
        out[key] = {
            "f_act": float(np.mean(b["f_act"])) if b["f_act"] else np.nan,
            "f_beh": float(np.mean(b["f_beh"])) if b["f_beh"] else np.nan,
            "cos": float(np.mean(b["cos"])) if b["cos"] else np.nan,
            "trav": float(np.mean(b["trav"])) if b["trav"] else np.nan,
            "coh_frac": n_coh / n_tot if n_tot else np.nan,
            "caphit_frac": float(np.mean(b["caphit"])) if b["caphit"] else np.nan,
            "n_coh": n_coh,
            "n_tot": n_tot,
        }
    return out


# A cell is DROPPED — struck through with an X, never read as an effect — when its
# draws mostly fell apart. Coherence is the SOLE criterion.
#
# Two former criteria were removed. (a) `>2% cap-hit` (the fu2 "compromised family"
# convention) hatched 12 cells, three of them on a SINGLE truncated rollout out of 30
# — a 2% ceiling is far too tight to be a read/do-not-read switch at n=30, where one
# truncation is already 3.3%. Cap-hit rates stay in cells_summary.json and are quoted
# in the writeup prose per cell, which is the right granularity for a caveat.
# (b) `<5 pairs` never fired on this grid (0 of 176 cells) — dead text in the title.
#
# The floor is 90%, not 50%: the coherence distribution is bimodal (see
# draw_coherence_distribution), so a cell below it is one where a real minority of
# draws came back as word salad — its surviving mean is taken over a selected subset
# and should not be read as an effect. The unpatched anchor baseline is ~98% coherent,
# so 90% is the nearest round floor that still sits below the no-intervention rate.
COH_FLOOR = 0.9


def dropped_reason(v: dict) -> str | None:
    """-> short reason a cell must not be read, or None if it is readable."""
    coh = v["coh_frac"]
    if coh == coh and coh < COH_FLOOR:
        return "incoherent"
    return None


def grid_of(agg: dict, setting: str, field: str) -> np.ndarray:
    g = np.full((len(LAYER_ROWS), len(SLOTS)), np.nan)
    for (s, slot, lv), v in agg.items():
        if s == setting:
            g[LAYER_ROWS.index(lv), SLOTS.index(slot)] = v[field]
    return g


def annotate(ax, agg: dict, setting: str) -> None:
    for (s, slot, lv), v in agg.items():
        if s != setting:
            continue
        i, j = LAYER_ROWS.index(lv), SLOTS.index(slot)
        ax.text(j, i, f"{v['n_coh']}/{v['n_tot']}", ha="center", va="center", fontsize=3.2)
        if dropped_reason(v) is None:
            continue
        # Cross-hatch the WHOLE cell. The old marker was a 3.2pt "*" appended to the
        # count — invisible at this cell size, and it only ever flagged incoherence,
        # never truncation or a thin pair count. A corner-to-corner X does not work
        # either: 30 layer rows in a 9in panel make each cell ~5x wider than tall, so
        # the X flattens into a smear. A hatch fills the cell at any aspect ratio.
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                hatch="xxxx",
                edgecolor="black",
                linewidth=0.0,
                zorder=4,
            )
        )


def draw(agg: dict, field: str, title: str, cmap, norm, out: Path, baseline_note: str = "") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    for ax, setting in zip(axes, SETTINGS):
        g = grid_of(agg, setting, field)
        masked = np.ma.masked_invalid(g)
        cmap_ = plt.get_cmap(cmap).copy()
        cmap_.set_bad("0.92")
        im = ax.imshow(masked, aspect="auto", cmap=cmap_, norm=norm, interpolation="nearest")
        ax.set_title(SETTING_TITLES[setting], fontsize=10)
        ax.set_xticks(
            range(len(SLOTS)), [SLOT_LABELS[s] for s in SLOTS], rotation=60, ha="right", fontsize=7
        )
        ax.set_yticks(range(len(LAYER_ROWS)), LAYER_ROWS, fontsize=5.5)
        annotate(ax, agg, setting)
    fig.colorbar(im, ax=axes, shrink=0.7, label=field)
    note = (
        "grey = arm not run; cell text = n_coherent/n_total pairs; "
        "X = DROPPED, do not read (<90% coherent draws)"
    )
    fig.suptitle(f"{title}\n{note}{('; ' + baseline_note) if baseline_note else ''}", fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cos_margin_agg(agg: dict, agg_null: dict) -> dict:
    """-> a copy of ``agg`` whose ``cos`` field is (steered - shuffled-donor null) cosine.

    The null is NOT at cos 0: a norm-matched edit from a WRONG donor pair still lands
    partly on the floor->ceiling axis (0.10-0.35 across the grid) because every context
    edit shares geometry with every other. So the raw cosine overstates alignment by
    whatever a random edit of the same size buys, and the margin over the cell's own null
    is the defensible read. n / coherence / the DROPPED hatch stay the STEERED cell's --
    the drop criterion is a property of the arm being read, not of its null.
    """
    out = {}
    for key, v in agg.items():
        nv = agg_null.get(key)
        c, cn = v["cos"], (nv["cos"] if nv else np.nan)
        out[key] = {**v, "cos": (c - cn) if (c == c and cn == cn) else np.nan}
    return out


TRANSPORT_DOSES = ("a0.5", "a1", "a2", "a4", "replace")
TRANSPORT_DOSE_LABELS = {
    "a0.5": "α 0.5",
    "a1": "α 1",
    "a2": "α 2",
    "a4": "α 4",
    "replace": "replace",
}


def draw_transport(eval_root: Path, out: Path) -> int:
    """Render dose x slot@map-layer transport-cosine heatmaps; returns n cells drawn."""
    table = eval_root / "transport/transport_cells.jsonl"
    if not table.exists():
        print(f"no transport table at {table} — skipping transport heatmap")
        return 0
    rows = [json.loads(line) for line in table.open() if line.strip()]
    cols = sorted({(r["slot"], r["layer"]) for r in rows}, key=lambda sl: (sl[0] != "ce", sl[1]))
    col_labels = [f"{SLOT_LABELS[s].splitlines()[0]}\n@L{layer}" for s, layer in cols]
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("degenerate_self"):
            continue
        key = (r["setting"], (r["slot"], r["layer"]), r["dose"])
        buckets[key][r["arm"]].append(float(r["cosine_tail"]))

    # Column groups for the two-level x axis: one tick per LAYER, one group label per SLOT.
    # Repeating the slot name in all six ticks (the old "context-end\n@L14" form) is what
    # made the axis unreadable at this figure width.
    groups: list[tuple[str, int, int]] = []
    for j, (slot, _layer) in enumerate(cols):
        if groups and groups[-1][0] == slot:
            groups[-1] = (slot, groups[-1][1], j)
        else:
            groups.append((slot, j, j))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    # Data span is ~[-0.05, 0.21]; the old +-0.3 norm washed every cell out to near-white.
    norm = TwoSlopeNorm(vmin=-0.22, vcenter=0.0, vmax=0.22)
    n_cells = 0
    for ax, setting in zip(axes, SETTINGS):
        g = np.full((len(TRANSPORT_DOSES), len(cols)), np.nan)
        for j, col in enumerate(cols):
            for i, dose in enumerate(TRANSPORT_DOSES):
                b = buckets.get((setting, col, dose))
                if not b or not b.get("steered"):
                    continue
                g[i, j] = float(np.mean(b["steered"]))
                n_cells += 1
                # One number per line, no words, no per-cell n (n is stated in the
                # suptitle): steered on top, its shuffled-donor null below in grey.
                ax.text(j, i - 0.16, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=8)
                if b.get("null"):
                    ax.text(
                        j,
                        i + 0.20,
                        f"{np.mean(b['null']):.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="0.35",
                    )
        cmap_ = plt.get_cmap("RdBu_r").copy()
        cmap_.set_bad("0.92")
        im = ax.imshow(
            np.ma.masked_invalid(g), aspect="auto", cmap=cmap_, norm=norm, interpolation="nearest"
        )
        ax.set_title(SETTING_TITLES[setting], fontsize=10)
        ax.set_xticks(range(len(cols)), [f"L{layer}" for _slot, layer in cols], fontsize=8)
        ax.set_yticks(
            range(len(TRANSPORT_DOSES)),
            [TRANSPORT_DOSE_LABELS[d] for d in TRANSPORT_DOSES],
            fontsize=8,
        )
        for slot, j0, j1 in groups:
            ax.annotate(
                SLOT_LABELS[slot].splitlines()[0],
                xy=((j0 + j1) / 2, -0.115),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8.5,
                annotation_clip=False,
            )
            if j0:
                ax.axvline(j0 - 0.5, color="0.35", lw=0.9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cos(map-predicted shift, realized shift)")
    fig.suptitle(
        "Banked-map transport: does the fitted map predict the realized answer-vector shift?\n"
        "cell = cosine(predicted, realized): steered on top, shuffled-donor null below in grey; "
        "1.0 = perfect prediction, 0 = unrelated\n"
        "n = 15 pairs per cell (30 at context-end where both steering-vector variants pool); "
        "grey = no banked map at that slot/layer",
        fontsize=9.5,
        y=1.13,  # 3-line title clears the per-panel titles (savefig uses bbox_inches="tight")
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return n_cells


OFFAXIS_SLOTS = ("ce", "pe", "cm2", "cm3", "l3j", "qspan", "qtext", "pspan_text")


def _offaxis_rows(eval_root: Path, steered: bool) -> list[dict]:
    rels = (
        ("f_metrics/f_cells.jsonl", "f_metrics/fu2/fu2_cells.jsonl")
        if steered
        else ("f_metrics/null_cells.jsonl", "f_metrics/fu2/fu2_null_cells.jsonl")
    )
    out: list[dict] = []
    for rel in rels:
        fp = eval_root / rel
        if fp.exists():
            with fp.open() as fh:
                out.extend(json.loads(x) for x in fh if x.strip())
    return out


def _decompose(rows: list[dict], slot: str, lv: str, dose: str, setting: str) -> tuple | None:
    """-> (n, on_axis, off_axis, cos) for one cell, or None.

    F_act projects the realized shift onto the floor->ceiling axis, so it is blind to
    movement PERPENDICULAR to that axis: a patch that hurls the answer state somewhere
    unrelated scores the same as one that barely moved. The banked full-mean fields make
    the split exact and free -- traversal_ratio and f_act_shared_recordonly are computed
    against the SAME full floor mean, so with traversal = ||s||/||t|| and
    on = <s,t>/||t||^2:

        off = sqrt(traversal^2 - on^2)      (the orthogonal residual, same units)
        cos = on / traversal                (fraction of the movement pointing at B)

    Use f_act_shared_recordonly, NOT f_act: the reported f_act uses disjoint floor halves
    (a different t), which would break the Pythagorean identity. The shared-floor read is
    mildly noise-inflated (~+0.08, #1415), so `on` is if anything generous and `off`
    conservative -- the safe direction for an off-axis claim.
    """
    sel = [
        r
        for r in rows
        if r["slot"] == slot
        and r["layer_variant"] == lv
        and r["setting"] == setting
        and r["dose"] == dose
        and r.get("coherent")
        and r.get("traversal_ratio") is not None
        and r.get("f_act_shared_recordonly") is not None
    ]
    if not sel:
        return None
    tr = np.array([float(r["traversal_ratio"]) for r in sel])
    on = np.array([float(r["f_act_shared_recordonly"]) for r in sel])
    off = np.sqrt(np.clip(tr**2 - on**2, 0.0, None))
    cos = np.where(tr > 0, on / tr, np.nan)
    return len(sel), float(np.nanmean(on)), float(np.nanmean(off)), float(np.nanmean(cos))


def draw_offaxis(eval_root: Path, out: Path, dose: str, setting: str = "matched_query") -> dict:
    """On-axis vs off-axis movement of the answer state, steered vs shuffled-donor null."""
    steered_rows, null_rows = _offaxis_rows(eval_root, True), _offaxis_rows(eval_root, False)
    stats: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, lv in zip(axes, ("joint_all", "joint_mid")):
        lim = 0.05
        for slot in OFFAXIS_SLOTS:
            a = _decompose(steered_rows, slot, lv, dose, setting)
            b = _decompose(null_rows, slot, lv, dose, setting)
            if not a:
                continue
            stats[f"{setting}|{slot}|{lv}"] = {
                "steered": {"n": a[0], "on_axis": a[1], "off_axis": a[2], "cos": a[3]},
                "null": (
                    {"n": b[0], "on_axis": b[1], "off_axis": b[2], "cos": b[3]} if b else None
                ),
            }
            if b:
                ax.plot([a[1], b[1]], [a[2], b[2]], color="0.75", lw=0.8, zorder=1)
                ax.scatter(
                    [b[1]], [b[2]], s=52, facecolors="none", edgecolors="0.45", lw=1.3, zorder=2
                )
                lim = max(lim, b[1], b[2])
            ax.scatter([a[1]], [a[2]], s=58, color="#1b6ca8", zorder=3)
            ax.annotate(
                SLOT_LABELS[slot].replace("\n", " "),
                (a[1], a[2]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7.5,
            )
            lim = max(lim, a[1], a[2])
        lim *= 1.15
        ax.plot([0, lim], [0, lim], color="0.6", ls="--", lw=1.0, zorder=0)
        ax.annotate(
            "equal on/off\n(cos = 0.71)", (lim * 0.68, lim * 0.80), fontsize=7, color="0.45"
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("ON-axis movement toward context B  (= F_act)")
        ax.set_ylabel("OFF-axis movement (orthogonal residual)")
        ax.set_title(
            f"{'all 28 layers' if lv == 'joint_all' else 'layers 14-20'}"
            "\nfilled = real patch, open = shuffled-donor null"
        )
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "Does the patch move the answer state TOWARD context B, or just move it?\n"
        "both axes in units of the floor->ceiling axis length; below the dashed line = "
        "mostly on-target, above it = mostly off-target",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stats


COH_SOURCES = (
    ("single-position slots, real patch", "f_metrics/f_cells.jsonl", "#1b6ca8"),
    ("single-position slots, null", "f_metrics/null_cells.jsonl", "#9fc4dd"),
    ("multi-token span slots, real patch", "f_metrics/fu2/fu2_cells.jsonl", "#c1440e"),
    ("multi-token span slots, null", "f_metrics/fu2/fu2_null_cells.jsonl", "#e8a882"),
)


def draw_coherence_distribution(eval_root: Path, out: Path) -> dict:
    """Distribution of the graded coherence judge, and the cut's sensitivity.

    NOTE the two distinct thresholds: this figure is about the PER-DRAW judge cut
    (score > 60 => that draw counts as coherent). The separate COH_FLOOR = 0.8 is a
    PER-CELL floor on the FRACTION of a cell's draws that cleared the draw cut.

    Motivates the coherent/incoherent draw cut, which is otherwise an unexplained
    constant: the scores are hard bimodal (a draw is fluent or it is word salad),
    so every cut in [40, 80] partitions almost the same draws. Also shows WHERE
    the incoherence lives -- overwriting a multi-token span degrades the output
    about half the time, while single-position edits essentially never do.
    """
    groups: dict[str, np.ndarray] = {}
    for label, rel, _c in COH_SOURCES:
        fp = eval_root / rel
        if not fp.exists():
            continue
        with fp.open() as fh:
            vals = [
                float(r["coherence_score"])
                for r in (json.loads(x) for x in fh if x.strip())
                if r.get("coherence_score") is not None
            ]
        if vals:
            groups[label] = np.array(vals)
    if not groups:
        return {}
    pooled = np.concatenate(list(groups.values()))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    bins = np.linspace(0, 100, 26)
    for label, _rel, color in COH_SOURCES:
        if label not in groups:
            continue
        axes[0].hist(
            groups[label],
            bins=bins,
            histtype="step",
            lw=1.9,
            color=color,
            label=f"{label} (n={len(groups[label]):,})",
        )
    for t, style in ((50.0, ":"), (60.0, "--")):
        axes[0].axvline(t, color="0.25", ls=style, lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coherence judge score (0-100, form-only rubric)")
    axes[0].set_ylabel("draws (log scale)")
    axes[0].set_title(
        "A. The coherence judge is bimodal: draws are fluent or\n"
        "they are word salad, with almost nothing in between"
    )

    ts = np.arange(30, 91, 2.0)
    for label, _rel, color in COH_SOURCES:
        if label not in groups:
            continue
        v = groups[label]
        axes[1].plot(ts, [(v > t).mean() for t in ts], color=color, lw=1.9, label=label)
    axes[1].plot(
        ts,
        [(pooled > t).mean() for t in ts],
        color="k",
        lw=1.4,
        ls="--",
        label=f"pooled (n={len(pooled):,})",
    )
    for t, style in ((50.0, ":"), (60.0, "--")):
        axes[1].axvline(t, color="0.25", ls=style, lw=1.2)
    n_band = int(((pooled > 50) & (pooled <= 60)).sum())
    axes[1].set_xlabel("per-draw coherence cut (dotted 50, dashed 60 = the cut used)")
    axes[1].set_ylabel("fraction counted as coherent")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(
        "B. The DRAW cut is not load-bearing: 50 -> 60 reclassifies\n"
        f"{n_band} of {len(pooled):,} draws ({n_band / len(pooled):.2%})"
    )
    for ax in axes:
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_draws": int(len(pooled)),
        "pooled_median": float(np.median(pooled)),
        "n_in_50_60_band": n_band,
        "frac_coherent_by_cut": {
            str(int(t)): float((pooled > t).mean()) for t in (40, 50, 60, 70, 80)
        },
        "per_group": {
            k: {
                "n": int(len(v)),
                "median": float(np.median(v)),
                "frac_gt_80": float((v > 80).mean()),
            }
            for k, v in groups.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dose", default="replace")
    ap.add_argument("--min-sep", type=float, default=0.5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.eval_root)
    agg = aggregate(rows, args.dose)
    assert agg, "no cells aggregated — check --eval-root / --dose"

    anchors = [
        json.loads(line)
        for line in (args.eval_root / "f_metrics/anchor_draws.jsonl").open()
        if line.strip()
    ]
    base_coh = float(np.mean([a["coherent"] for a in anchors]))
    baseline_note = (
        f"unpatched anchor baseline coherence {base_coh:.1%} over {len(anchors)} draws "
        "(contexts A and B draw from the same 15-context anchor pool)"
    )

    fnorm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.1)
    draw(
        agg,
        "f_act",
        f"F_act (answer-vector shift, {args.dose} patch)",
        "RdBu_r",
        fnorm,
        args.out_dir / "f_act_heatmaps.png",
    )
    draw(
        agg,
        "f_beh",
        f"F_beh (judged behavior, {args.dose} patch)",
        "RdBu_r",
        fnorm,
        args.out_dir / "f_beh_heatmaps.png",
    )
    # Direction-only read. F_act ~ trav * cos, so a slot that hurls the answer state a
    # long way scores well on F_act even when little of the motion points at context B;
    # the cosine is scale-free and separates the two. Blind to magnitude by construction
    # (a tiny perfectly-aligned move reads 1.0), so it is a companion to F_act, not a
    # replacement -- mean traversal per cell rides along in cells_summary.json.
    null_rows = load_rows(args.eval_root, steered=False)
    agg_null = aggregate(null_rows, args.dose, arm="null")
    assert agg_null, "no null cells aggregated"
    # Range tracks the data (max readable cell = context-end @ joint_all, 0.66), not the
    # theoretical [-1, 1]: at vmax 0.9 the single-layer band (0.1-0.3) washes to white.
    cnorm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.7)
    draw(
        agg,
        "cos",
        f"cos(realized answer-vector shift, floor->ceiling axis) — {args.dose} patch\n"
        "1 = moves straight at context B, 0 = moves sideways, <0 = moves away",
        "RdBu_r",
        cnorm,
        args.out_dir / "cos_heatmaps.png",
    )
    draw(
        cos_margin_agg(agg, agg_null),
        "cos",
        f"cos MARGIN over the shuffled-donor null — {args.dose} patch\n"
        "steered cosine minus the same cell's norm-matched wrong-donor cosine; "
        "0 = the real edit is no better aimed than a random one of the same size",
        "RdBu_r",
        TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.6),
        args.out_dir / "cos_margin_heatmaps.png",
    )

    draw(
        agg,
        "coh_frac",
        f"Coherent-draw fraction ({args.dose} patch)",
        "viridis",
        plt.Normalize(0, 1),
        args.out_dir / "coherence_heatmaps.png",
        baseline_note,
    )

    # F_beh restricted to well-separated pairs. F_beh divides by the anchor separation,
    # so the near-zero-separation pairs (bare<->conversation: floor ~= ceiling) blow the
    # unrestricted cell means past 1.0 — |F| > 1 is impossible as a fraction-of-swap and
    # is the tell. Their shuffled-donor nulls blow up identically (mq|qtext|joint_all:
    # steered 1.53 vs null 21.4 unrestricted; 0.32 vs -0.09 well-separated), so the
    # unrestricted panel shows no separation where it looks strongest. F_act is NOT
    # affected (separation never enters its denominator) — one wellsep panel, F_beh only.
    ws, ws_any = load_wellsep(args.eval_root, args.min_sep)
    agg_ws = aggregate(rows, args.dose, ws=ws, ws_any=ws_any)
    assert agg_ws, "no well-separated cells aggregated"
    draw(
        agg_ws,
        "f_beh",
        f"F_beh (judged behavior, {args.dose} patch) — WELL-SEPARATED pairs only "
        f"(|anchor separation| >= {args.min_sep})",
        "RdBu_r",
        fnorm,
        args.out_dir / "f_beh_heatmaps_wellsep.png",
    )

    n_transport = draw_transport(args.eval_root, args.out_dir / "transport_heatmaps.png")
    coh_stats = draw_coherence_distribution(
        args.eval_root, args.out_dir / "coherence_distribution.png"
    )
    offaxis_stats = draw_offaxis(
        args.eval_root, args.out_dir / "offaxis_decomposition.png", args.dose
    )

    summary = {
        f"{s}|{slot}|{lv}": v for (s, slot, lv), v in sorted(agg.items(), key=lambda kv: str(kv[0]))
    }
    summary_ws = {
        f"{s}|{slot}|{lv}": v
        for (s, slot, lv), v in sorted(agg_ws.items(), key=lambda kv: str(kv[0]))
    }
    (args.out_dir / "cells_summary.json").write_text(
        json.dumps(
            {
                "dose": args.dose,
                "baseline_anchor_coherence": base_coh,
                "cells": summary,
                "min_abs_separation": args.min_sep,
                "cells_wellsep": summary_ws,
                "coherence_distribution": coh_stats,
                "offaxis_decomposition": offaxis_stats,
            },
            indent=1,
        )
    )
    # Count what is on disk rather than a literal: the literal said "5" while the script
    # was already writing 7, and every new panel silently widened the lie.
    n_png = len(list(args.out_dir.glob("*.png")))
    print(
        f"wrote {n_png} figures + cells_summary.json to {args.out_dir} ({len(summary)} cells; "
        f"transport heatmap cells: {n_transport})"
    )


if __name__ == "__main__":
    main()
