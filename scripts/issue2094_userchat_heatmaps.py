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
pair) and cells with coherent fraction < 50% are marked with '*' (visible,
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


def load_rows(eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for rel in ("f_metrics/f_cells.jsonl", "f_metrics/fu2/fu2_cells.jsonl"):
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
) -> dict:
    """-> {(setting, slot, layer_row): {f_act, f_beh, coh_frac, n_coh, n_tot}}

    When ``ws``/``ws_any`` are given, restrict to well-separated pairs: rows are gated
    on ``ws_any`` (pair well-separated on >= 1 primary rubric — the FA-3 row-level
    convention) and each rubric's value on ``ws``.
    """
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] != "steered" or r["dose"] != dose or r.get("degenerate_self"):
            continue
        if r["slot"] not in SLOTS or r["layer_variant"] not in LAYER_ROWS:
            continue
        if ws_any is not None and r["pair_id"] not in ws_any:
            continue
        key = (r["setting"], r["slot"], r["layer_variant"])
        buckets[key]["n_tot"].append(1)
        if not r["coherent"]:
            continue
        buckets[key]["n_coh"].append(1)
        if r.get("f_act") is not None and not r.get("f_act_degenerate"):
            buckets[key]["f_act"].append(float(r["f_act"]))
        bv = beh_value(r, r["setting"], ws=ws)
        if bv is not None:
            buckets[key]["f_beh"].append(bv)
    out = {}
    for key, b in buckets.items():
        n_tot, n_coh = len(b["n_tot"]), len(b["n_coh"])
        out[key] = {
            "f_act": float(np.mean(b["f_act"])) if b["f_act"] else np.nan,
            "f_beh": float(np.mean(b["f_beh"])) if b["f_beh"] else np.nan,
            "coh_frac": n_coh / n_tot if n_tot else np.nan,
            "n_coh": n_coh,
            "n_tot": n_tot,
        }
    return out


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
        star = "*" if (v["coh_frac"] == v["coh_frac"] and v["coh_frac"] < 0.5) else ""
        ax.text(j, i, f"{v['n_coh']}/{v['n_tot']}{star}", ha="center", va="center", fontsize=3.2)


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
    note = "grey = arm not run; cell text = n_coherent/n_total pairs; * = <50% coherent"
    fig.suptitle(f"{title}\n{note}{('; ' + baseline_note) if baseline_note else ''}", fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
            },
            indent=1,
        )
    )
    print(
        f"wrote 3 figures + cells_summary.json to {args.out_dir} ({len(summary)} cells; "
        f"transport heatmap cells: {n_transport})"
    )


if __name__ == "__main__":
    main()
