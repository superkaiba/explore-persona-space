"""#1586 seed-pooled registered H1/H3 lattice reads (analyzer, p11).

Plan §3/§6: the H1 verdict lattice binds on the SEED-POOLED con-regime paired
mean-shift-norm difference Δnorm (question-cluster bootstrap, paired indices,
seed-stratified pooling: resample question clusters within seed, pool the two
seeds' paired differences per draw). This script computes, per behavior ×
regime × text-arm (own / tf-shared), the pooled Δnorm at n_boot=2000 (seed
653, the run convention) from the SAME pooled capture stores the per-pair
pass used, plus the pooled H3 shape-DV diffs at n_boot=1000 re-read from the
persisted per-cell bootstrap matrices (identical index matrices by
construction — one `bootstrap_index_matrix(cluster_ids, seed=653)` per
behavior panel). Batched draws only; no per-draw Python loop.

Output: eval_results/issue_1586/geometry/pooled_lattice.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue1586_geometry import (  # noqa: E402
    CAPTURE_ARMS,
    _mu_norm_draws,
    bootstrap_index_matrix,
)

from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

GEO_ROOT = Path("eval_results/issue_1586/geometry")
BEH_LAYER = {"syc": 14, "imp": 14, "cas": 14, "mk": 25}
N_BOOT_NORM = 2000
SHAPE_DVS = ("top_share_lambda", "pr_lambda", "rank_k_at_90")


def pooled_norm_read(
    tree: Path, base_tree: Path, beh: str, regime: str, arm: str, layer: int
) -> dict:
    """Seed-pooled paired Δnorm: per seed, paired FT−LoRA per-draw diffs on one
    shared index matrix (cluster ids identical across seeds — same panel);
    pooled draw = mean over seeds; point = mean of per-seed points."""
    base = geo.load_store(base_tree / f"base_{beh}" / "selected" / "pooled.pt")
    cluster_ids = [f"{c}__{q}" for c, q in geo._row_keys(base)]
    idx = bootstrap_index_matrix(cluster_ids, n_boot=N_BOOT_NORM, seed=geo.BOOT_SEED)
    per_seed_diff_draws, per_seed_points = [], []
    for seed in ("s42", "s137"):
        ft = geo.load_store(tree / f"{beh}-pers-ft-{regime}-{seed}" / "selected" / "pooled.pt")
        lo = geo.load_store(tree / f"{beh}-pers-lora-{regime}-{seed}" / "selected" / "pooled.pt")
        cloud_ft = geo.delta_cloud(ft, base, arm, layer)
        cloud_lo = geo.delta_cloud(lo, base, arm, layer)
        per_seed_diff_draws.append(_mu_norm_draws(cloud_ft, idx) - _mu_norm_draws(cloud_lo, idx))
        per_seed_points.append(
            float(np.linalg.norm(cloud_ft.mean(axis=0)))
            - float(np.linalg.norm(cloud_lo.mean(axis=0)))
        )
    pooled = np.mean(per_seed_diff_draws, axis=0)
    return {
        "point": float(np.mean(per_seed_points)),
        "ci_low": float(np.nanquantile(pooled, 0.025)),
        "ci_high": float(np.nanquantile(pooled, 0.975)),
        "n_boot": N_BOOT_NORM,
        "resampling": "paired, seed-stratified pooled",
        "per_seed_points": per_seed_points,
    }


def pooled_shape_reads(beh: str, regime: str, arm: str, layer: int) -> dict:
    """Seed-pooled paired shape-DV diffs from the persisted per-cell 1000-draw
    matrices (identical index matrices per behavior panel by construction)."""
    mats = {}
    for method in ("ft", "lora"):
        for seed in ("s42", "s137"):
            p = GEO_ROOT / "bootstrap_matrices" / f"{beh}-pers-{method}-{regime}-{seed}_selected.pt"
            mats[(method, seed)] = torch.load(p, map_location="cpu", weights_only=False)
    per_cell = json.load(open(GEO_ROOT / "geometry_per_cell.json"))["records"]
    out = {}
    for dv in SHAPE_DVS:
        key = f"{arm}/L{layer}/{dv}"
        pooled = np.mean(
            [mats[("ft", s)][key] - mats[("lora", s)][key] for s in ("s42", "s137")], axis=0
        )
        pts = [
            per_cell[f"{beh}-pers-ft-{regime}-{s}/selected/{arm}/L{layer}"][dv]
            - per_cell[f"{beh}-pers-lora-{regime}-{s}/selected/{arm}/L{layer}"][dv]
            for s in ("s42", "s137")
        ]
        out[dv] = {
            "point": float(np.mean(pts)),
            "ci_low": float(np.nanquantile(pooled, 0.025)),
            "ci_high": float(np.nanquantile(pooled, 0.975)),
            "n_boot": int(pooled.shape[0]),
            "resampling": "paired, seed-stratified pooled",
            "per_seed_points": pts,
        }
    return out


def main() -> int:
    out: dict[str, dict] = {"norm": {}, "shape": {}}
    own_tree = GEO_ROOT / "_work" / "own" / "tree"
    for tag, tree, arms in (
        ("own", own_tree, CAPTURE_ARMS),
        # tf tree carries no base pass; shared-text control = response arm only
        ("tf", GEO_ROOT / "_work" / "tf" / "tree", ("response",)),
    ):
        for beh, layer in BEH_LAYER.items():
            for regime in ("con", "po"):
                for arm in arms:
                    rec = pooled_norm_read(tree, own_tree, beh, regime, arm, layer)
                    out["norm"][f"{tag}/{beh}/{regime}/{arm}/L{layer}"] = rec
                    if arm == "response":
                        print(
                            f"[pooled] {tag} {beh} {regime} {arm}/L{layer}: "
                            f"{rec['point']:+.3f} [{rec['ci_low']:+.3f},{rec['ci_high']:+.3f}]",
                            flush=True,
                        )
    for beh, layer in BEH_LAYER.items():
        for regime in ("con", "po"):
            out["shape"][f"own/{beh}/{regime}/response/L{layer}"] = pooled_shape_reads(
                beh, regime, "response", layer
            )
    dest = GEO_ROOT / "pooled_lattice.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[pooled] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
