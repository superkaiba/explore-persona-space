# ruff: noqa: RUF002  # Qwen marker token " ※" + em-dash intentional
"""Task #472 multi-layer robustness re-analysis (analyzer round-2 recovery).

Round-1 interpretation critique flagged that the body asserted the
source-proximity gradient "holds across layers 10/15/20" and that the
identification gate "fails at layers 10 and 15, borderline at 20", but the
committed re-fit (``issue472_reanalyze_earliest_slice.py``) computed everything
at layer 10 only. This script computes, at EACH of L10 / L15 / L20, read at the
earliest checkpoint (frac 0.08):

  * the source-proximity gradient: Spearman(held-out leakage ΔG, distance-to-source)
    over the pooled count-matched placement arms (n=282 probe×arm×seed);
  * the identification gate: ``median_across_arm_sd_dnn_nd`` and
    ``qwen_default_nearest_share``, with the 0.02 SD floor.

So the multi-layer claims in the clean-result body are backed by a committed
artifact rather than an undocumented re-run. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    SUBCEILING_HEADROOM_NATS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cos_to_source,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_nearest_neg,
    d_source,
    held_out_panel,
    negatives_for_cell,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
CENTROID_DIR = WT / "data" / "issue_472"
SEEDS = [42, 137]
LAYERS = [10, 15, 20]
SOURCE = "villain"
POOLED_CELLS = [c[0] for c in CELL_SPECS if c[5]]  # anchor(spread), near, far
ID_GATE_SD_FLOOR = 0.02


def earliest_ck(cell: str, seed: int) -> dict:
    t = json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())
    return sorted(t["checkpoints"], key=lambda c: c["frac"])[0]


def per_probe_at_earliest(cell: str, seed: int) -> dict[str, dict]:
    ck = earliest_ck(cell, seed)
    out = {}
    for persona, perq in ck["held_out"].items():
        dgs = [r["delta_g"] for r in perq.values()]
        gls = [r["g_logp"] for r in perq.values()]
        coll = any(r.get("r_collapsed", False) for r in perq.values())
        mean_g = float(np.mean(gls))
        out[persona] = {
            "delta_g": float(np.mean(dgs)),
            "g_logp": mean_g,
            "saturated": mean_g > -SUBCEILING_HEADROOM_NATS,
            "r_collapsed": coll,
        }
    return out


def analyze_layer(layer: int) -> dict:
    cts = load_cos_to_source(layer, SOURCE, CENTROID_DIR)
    cos_matrix, _names = load_cos_matrix(layer, CENTROID_DIR)
    panel = held_out_panel(cts, source=SOURCE)
    matched = {c[0]: {s: per_probe_at_earliest(c[0], s) for s in SEEDS} for c in CELL_SPECS}
    cell_negs = {c[0]: negatives_for_cell(c[0], cts, source=SOURCE) for c in CELL_SPECS}

    ds_vals, leak_vals = [], []
    nearest_default_count = nearest_total = 0
    dnn_nd_across_arms: dict[str, list[float]] = {p: [] for p in panel}

    for slug in POOLED_CELLS:
        negs = cell_negs[slug]
        for seed in SEEDS:
            pp_all = matched[slug][seed]
            for probe in panel:
                if probe not in pp_all:
                    continue
                pp = pp_all[probe]
                dnn_nd = d_nearest_neg(probe, negs, cos_matrix, exclude_default=True)
                dnn_nd_across_arms[probe].append(dnn_nd)
                if negs:
                    nearest = min(negs, key=lambda nn: 1.0 - cos_matrix[probe][nn])
                    nearest_total += 1
                    if nearest == ALWAYS_INCLUDE_NEGATIVE:
                        nearest_default_count += 1
                if pp["saturated"] or pp["r_collapsed"]:
                    continue
                ds_vals.append(d_source(probe, cts))
                leak_vals.append(pp["delta_g"])

    rho = spearmanr(ds_vals, leak_vals)
    default_share = nearest_default_count / nearest_total if nearest_total else float("nan")
    across_arm_sd = [
        float(np.std(v))
        for v in dnn_nd_across_arms.values()
        if len(v) >= 2 and not np.all(np.isnan(v))
    ]
    median_sd = float(np.median(across_arm_sd)) if across_arm_sd else float("nan")

    return {
        "layer": layer,
        "n_panel": len(panel),
        "proximity_gradient": {
            "spearman_leakage_vs_d_source": float(rho.correlation),
            "p": float(rho.pvalue),
            "n": len(leak_vals),
        },
        "identification_gate": {
            "qwen_default_nearest_share": default_share,
            "median_across_arm_sd_dnn_nd": median_sd,
            "id_gate_floor": ID_GATE_SD_FLOOR,
            "admissible_distance_movement": (
                not np.isnan(median_sd) and median_sd >= ID_GATE_SD_FLOOR
            ),
        },
    }


def main() -> None:
    results = {layer: analyze_layer(layer) for layer in LAYERS}
    for layer in LAYERS:
        r = results[layer]
        pg = r["proximity_gradient"]
        ig = r["identification_gate"]
        print(f"── L{layer} (n_panel={r['n_panel']}) ──")
        print(
            f"  proximity gradient Spearman(leakage, d_source) = "
            f"{pg['spearman_leakage_vs_d_source']:+.3f}  p={pg['p']:.2e}  n={pg['n']}"
        )
        print(
            f"  identification: median across-arm SD(d_nearest_neg_nd) = "
            f"{ig['median_across_arm_sd_dnn_nd']:.4f}  (floor {ig['id_gate_floor']}) "
            f"-> distance-movement admissible={ig['admissible_distance_movement']}; "
            f"qwen_default nearest share={ig['qwen_default_nearest_share']:.3f}"
        )
    out = {
        "schema": "i472_reanalysis_multilayer",
        "read_at": "earliest_checkpoint_frac_0.08",
        "layers": results,
    }
    (SLAB / "reanalysis_multilayer.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {SLAB / 'reanalysis_multilayer.json'}")


if __name__ == "__main__":
    main()
