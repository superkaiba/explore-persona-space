#!/usr/bin/env python
"""issue #666 Phase 4 — test-retest noise-floor (reliability ceiling) estimator (plan §4i, §11).

The test-retest reliability is the CEILING on any predictor's achievable ρ. Two
independent estimates of the latent Δs from independent probe-split halves of
``v_plus_probe`` / ``v0_probe`` (reusing the
``issue664_aggregate_gate.probe_split_floor`` half-split logic): split the n_probe
axis into two disjoint halves, recompute Δs = r_Bᵀ(mean Δv) on each half over the
bystander contexts, and the test-retest Spearman ρ between the two halves is the
floor (a ρ at the floor is the measurement ceiling, not a model failure).

Pre-registered MC (plan §11): **200 probe-split resamples × 3 RNG seeds = 600
total**. Every headline ρ is reported AGAINST this floor.

CPU-only; reuses the #664 probe-split half logic. No store dependence beyond the
passed probe tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_666"

# Pre-registered MC config (plan §11).
MC_RESAMPLES = 200
MC_SEEDS = (0, 1, 2)


def probe_half_indices(n_probe: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two DISJOINT halves of the n_probe axis (the #664 probe-split half logic).

    Permute [0, n_probe) and split at n_probe // 2; the two halves are disjoint and
    cover the axis (exactly, for even n_probe). Returns ``(h1, h2)`` index arrays.
    """
    perm = rng.permutation(n_probe)
    half = n_probe // 2
    return perm[:half], perm[half : 2 * half]


@dataclass
class NoiseFloor:
    """Test-retest reliability estimate over the MC resamples."""

    rho_mean: float
    rho_std: float
    n_resamples: int
    rho_by_seed: dict


def _half_ds(probe_dv: np.ndarray, idx: np.ndarray, r_B: np.ndarray) -> np.ndarray:
    """Latent Δs per context from a probe half's mean Δv.

    ``probe_dv`` : (n_ctx, n_probe, d) at the chosen layer. Mean over the half's
    probes → (n_ctx, d), project onto r_B → (n_ctx,).
    """
    dvm = probe_dv[:, idx, :].mean(axis=1)  # (n_ctx, d)
    return dvm @ r_B


def estimate_noise_floor(
    v_plus_probe,
    v0_probe,
    *,
    r_B,
    layer: int,
    source_idx: int,
) -> NoiseFloor:
    """Probe-split test-retest reliability of the latent Δs (plan §4i).

    ``*_probe`` : (n_ctx, n_probe, n_layer, d) tensors. For each of the
    ``MC_RESAMPLES`` resamples × each seed in ``MC_SEEDS`` (= 600 splits total), draw
    two disjoint probe halves (``probe_half_indices``), recompute Δs on each half
    over the BYSTANDER contexts (the source anchor excluded), and the test-retest
    Spearman ρ between the two half-estimates is one floor sample. Returns the
    mean/std over the 600 samples. A self-consistent (noise-free) predictor floors
    at ρ ≈ 1; finite within-context noise gives ρ < 1.
    """
    import torch
    from scipy.stats import spearmanr

    vpp = v_plus_probe[:, :, layer, :] if v_plus_probe.ndim == 4 else v_plus_probe
    v0p = v0_probe[:, :, layer, :] if v0_probe.ndim == 4 else v0_probe
    if isinstance(vpp, torch.Tensor):
        vpp = vpp.numpy()
    if isinstance(v0p, torch.Tensor):
        v0p = v0p.numpy()
    vpp = np.asarray(vpp, dtype=np.float64)
    v0p = np.asarray(v0p, dtype=np.float64)
    dv = vpp - v0p  # (n_ctx, n_probe, d)
    n_ctx, n_probe, _ = dv.shape
    rb = np.asarray(r_B, dtype=np.float64).reshape(-1)

    mask = np.ones(n_ctx, dtype=bool)
    if 0 <= source_idx < n_ctx:
        mask[source_idx] = False

    rho_by_seed: dict[int, list[float]] = {}
    all_rho: list[float] = []
    for seed in MC_SEEDS:
        rng = np.random.default_rng(seed)
        seed_rho: list[float] = []
        for _ in range(MC_RESAMPLES):
            h1, h2 = probe_half_indices(n_probe, rng)
            ds1 = _half_ds(dv, h1, rb)[mask]
            ds2 = _half_ds(dv, h2, rb)[mask]
            r = spearmanr(ds1, ds2).statistic
            rv = float(r) if np.isfinite(r) else 0.0
            seed_rho.append(rv)
            all_rho.append(rv)
        rho_by_seed[int(seed)] = [float(np.mean(seed_rho)), float(np.std(seed_rho))]

    arr = np.asarray(all_rho)
    return NoiseFloor(
        rho_mean=float(arr.mean()),
        rho_std=float(arr.std()),
        n_resamples=len(arr),
        rho_by_seed=rho_by_seed,
    )


# ── smoke driver ─────────────────────────────────────────────────────────────
def _smoke_cell() -> str:
    return "bm_default_contra_d1_seed42"


def main() -> int:
    import issue666_load_store as loader
    import issue666_predictor as pred

    ap = argparse.ArgumentParser(description="issue 666 Phase-4 noise-floor estimator.")
    ap.add_argument("--cell", default=_smoke_cell())
    ap.add_argument("--layer", type=int, default=pred.PRIMARY_LAYER)
    ap.add_argument("--n-resamples", type=int, default=None, help="(smoke) override MC_RESAMPLES")
    ap.add_argument("--seeds", type=int, default=None, help="(smoke) override len(MC_SEEDS)")
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Smoke overrides: temporarily shrink the MC counts so the pipeline runs fast
    # while exercising the SAME code path (the production run uses the registered
    # 200×3). The override mutates the module globals the estimator reads.
    global MC_RESAMPLES, MC_SEEDS
    if args.n_resamples is not None:
        MC_RESAMPLES = args.n_resamples
    if args.seeds is not None:
        MC_SEEDS = tuple(range(args.seeds))

    local_dir = loader.download_cell(args.cell)
    loaded = loader.load_cell(local_dir)
    layer = min(args.layer, loaded["v_plus"].shape[1] - 1)
    rb = pred.cell_r_plus(loaded, layer)
    src = pred._source_idx(loaded)
    floor = estimate_noise_floor(
        loaded["v_plus_probe"], loaded["v0_probe"], r_B=rb, layer=layer, source_idx=src
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / "noise_floor"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "cell": args.cell,
        "layer": layer,
        "rho_mean": floor.rho_mean,
        "rho_std": floor.rho_std,
        "n_resamples": floor.n_resamples,
        "mc_resamples": MC_RESAMPLES,
        "mc_seeds": list(MC_SEEDS),
    }
    (out_dir / "noise_floor.json").write_text(json.dumps(rec, indent=1))
    print(f"[noise_floor] {args.cell}: rho_mean={floor.rho_mean:.3f} (n={floor.n_resamples})")
    print("[phase=noise_floor] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
