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


def beh_value(row: dict, setting: str) -> float | None:
    vals = []
    for rubric in PRIMARY_RUBRIC[setting]:
        d = (row.get("f_beh") or {}).get(rubric)
        if d is None or d.get("degenerate_denominator") or d.get("negative_denominator"):
            continue
        vals.append(float(d["f_beh"]))
    return float(np.mean(vals)) if vals else None


def aggregate(rows: list[dict], dose: str) -> dict:
    """-> {(setting, slot, layer_row): {f_act, f_beh, coh_frac, n_coh, n_tot}}"""
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] != "steered" or r["dose"] != dose or r.get("degenerate_self"):
            continue
        if r["slot"] not in SLOTS or r["layer_variant"] not in LAYER_ROWS:
            continue
        key = (r["setting"], r["slot"], r["layer_variant"])
        buckets[key]["n_tot"].append(1)
        if not r["coherent"]:
            continue
        buckets[key]["n_coh"].append(1)
        if r.get("f_act") is not None and not r.get("f_act_degenerate"):
            buckets[key]["f_act"].append(float(r["f_act"]))
        bv = beh_value(r, r["setting"])
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
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
                null_txt = f"\nnull {np.mean(b['null']):+.2f}" if b.get("null") else ""
                ax.text(
                    j,
                    i,
                    f"{g[i, j]:+.2f} (n={len(b['steered'])}){null_txt}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                )
        cmap_ = plt.get_cmap("RdBu_r").copy()
        cmap_.set_bad("0.92")
        im = ax.imshow(
            np.ma.masked_invalid(g), aspect="auto", cmap=cmap_, norm=norm, interpolation="nearest"
        )
        ax.set_title(SETTING_TITLES[setting], fontsize=10)
        ax.set_xticks(range(len(cols)), col_labels, fontsize=7)
        ax.set_yticks(range(len(TRANSPORT_DOSES)), TRANSPORT_DOSES, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cos(map-predicted shift, realized shift)")
    fig.suptitle(
        "Banked-map transport: how well the fitted map predicts the realized answer-vector "
        "shift\n(1.0 = perfect prediction; steered arm, non-degenerate rows; grey = no banked "
        "map / degenerate-by-design)",
        fontsize=10,
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return n_cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dose", default="replace")
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

    n_transport = draw_transport(args.eval_root, args.out_dir / "transport_heatmaps.png")

    summary = {
        f"{s}|{slot}|{lv}": v for (s, slot, lv), v in sorted(agg.items(), key=lambda kv: str(kv[0]))
    }
    (args.out_dir / "cells_summary.json").write_text(
        json.dumps(
            {"dose": args.dose, "baseline_anchor_coherence": base_coh, "cells": summary}, indent=1
        )
    )
    print(
        f"wrote 3 figures + cells_summary.json to {args.out_dir} ({len(summary)} cells; "
        f"transport heatmap cells: {n_transport})"
    )


if __name__ == "__main__":
    main()
