#!/usr/bin/env python
"""issue #664 — summarize ĝ^real gate results + the kill 3(b) leakage-variation
vs within-context probe-split noise-floor read, across all computed cells.

Reads eval_results/issue_664/gate_real/<cell>/g_real.json (per-cell, per-layer,
per-context ĝ + probe-split floor). Writes gate_real_summary.json.

kill 3(b): cross-context leakage VARIATION over the 49 bystander contexts must
exceed the within-context probe-split noise floor. Per (cell, layer):
  - signal = std of bystander ĝ across the 49 bystander contexts
  - floor  = median over bystanders of |half1_ĝ − half2_ĝ| (probe-split noise mag)
  - SNR    = signal / floor
We report the per-cell best-layer SNR and the layer profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "eval_results" / "issue_664" / "gate_real"


def summarize_cell(rec: dict) -> dict:
    rows = rec["rows"]
    n_layer = rec["n_layers"]
    bys = [r for r in rows if r["target_context_role"] == "bystander"]
    anchor = [r for r in rows if r["target_context_role"] == "source-anchor"]
    g_bys = np.array([r["ghat_by_layer"] for r in bys])  # (n_bys, L)
    f_bys = np.array([r["floor_by_layer"] for r in bys])  # (n_bys, L)
    signal = g_bys.std(axis=0)  # (L,) cross-ctx leakage variation
    floor = np.median(f_bys, axis=0)  # (L,) probe-split noise mag
    snr = signal / np.where(floor > 0, floor, np.nan)  # (L,)
    bys_mean = g_bys.mean(axis=0)
    bys_max = g_bys.max(axis=0)
    bys_min = g_bys.min(axis=0)
    best_layer = int(np.nanargmax(snr))
    L14 = 14  # fixed mid-layer read (selection-bias-free reference)
    n_layers_clear = int(np.sum(snr > 1.0))  # robustness: how many of 28 layers clear floor
    return {
        "cell": rec["cell"],
        "behavior": rec["behavior"],
        "source": rec["source"],
        "arm": rec["arm"],
        "dose": rec["dose"],
        "seed": rec["seed"],
        "n_bystander": len(bys),
        "anchor_ghat_layer14": round(float(anchor[0]["ghat_by_layer"][14]), 4) if anchor else None,
        "best_layer_by_snr": best_layer,
        "best_layer_signal": round(float(signal[best_layer]), 5),
        "best_layer_floor": round(float(floor[best_layer]), 5),
        "best_layer_snr": round(float(snr[best_layer]), 3),
        "fixed_L14_signal": round(float(signal[L14]), 5),
        "fixed_L14_floor": round(float(floor[L14]), 5),
        "fixed_L14_snr": round(float(snr[L14]), 3) if np.isfinite(snr[L14]) else None,
        "n_layers_snr_gt1_of28": n_layers_clear,
        "median_snr_across_layers": round(float(np.nanmedian(snr)), 3),
        "bys_mean_by_layer": [round(float(x), 5) for x in bys_mean],
        "bys_std_by_layer": [round(float(x), 5) for x in signal],
        "bys_max_by_layer": [round(float(x), 5) for x in bys_max],
        "bys_min_by_layer": [round(float(x), 5) for x in bys_min],
        "floor_by_layer": [round(float(x), 5) for x in floor],
        "snr_by_layer": [round(float(x), 3) if np.isfinite(x) else None for x in snr],
    }


def main():
    cells = sorted(p for p in GATE.glob("*/g_real.json"))
    out = []
    for p in cells:
        rec = json.load(open(p))
        out.append(summarize_cell(rec))
    summary = {
        "n_cells": len(out),
        "note": "kill 3(b): cross-bystander-context ĝ std (signal) vs within-context "
        "probe-split noise floor; SNR = signal/floor per layer. anchor ĝ=1 by construction.",
        "cells": out,
    }
    op = GATE / "gate_real_summary.json"
    json.dump(summary, open(op, "w"), indent=1)
    print(f"wrote {op} ({len(out)} cells)")
    # quick console table at best layer
    print(f"\n{'cell':<36} {'beh':<8} {'bestL':>5} {'signal':>8} {'floor':>8} {'SNR':>6}")
    for c in out:
        print(
            f"{c['cell']:<36} {c['behavior']:<8} {c['best_layer_by_snr']:>5} "
            f"{c['best_layer_signal']:>8.4f} {c['best_layer_floor']:>8.4f} {c['best_layer_snr']:>6.2f}"
        )


if __name__ == "__main__":
    main()
