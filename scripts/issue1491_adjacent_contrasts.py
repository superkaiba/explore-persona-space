#!/usr/bin/env python3
"""Task #1491 analyzer round, post-pass: adjacent-rung paired contrasts.

(1) Converts the recomputed per-context ridge test preds
(``data/issue_1491/preds_recomputed/<slug>_test_preds_ridge_recomputed.npz``,
keys ``pred_te``/``y_te``/``ci_te``) into the plan §6.5 canonical form the
Phase-4 contrasts driver consumes
(``data/issue_1491/preds/<slug>_test_preds_ridge.npz``, keys
``ci``/``target``/``pred``) — repairing the primary-deliverable row the
pod-side driver wrote locally but never uploaded before teardown.

(2) Computes PAIRED bootstrap contrasts (shared resample matrix over the
common test-context ids, 1,000 draws, seed 42) for every adjacent rung pair
plus the fixed-h depth pair (14B, 32B), on raw ridge test R² AND on
ceiling-normalized R² (point ceilings from the committed fits JSONs held
fixed across draws). Emits a two-sided bootstrap p per contrast.

Writes ``eval_results/issue_1491/scale_ladder/adjacent_contrasts.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Before any heavy import, so the shared-VM thread caps (#847) bind in-process.
load_dotenv()

import numpy as np  # noqa: E402

SLUGS = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]
PAIRS = [
    ("scale15", "scale05"),
    ("scale3", "scale15"),
    ("scale7_refit", "scale3"),
    ("scale14", "scale7_refit"),
    ("scale32", "scale14"),
    ("scale32", "scale05"),
]
N_BOOT = 1000
SEED = 42


def _pooled_r2_boot(pred: np.ndarray, tgt: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Vectorized pooled variance-weighted R² over bootstrap count matrices."""
    n = tgt.shape[0]
    se_row = ((tgt - pred) ** 2).sum(axis=1).astype(np.float64)
    q_row = (tgt.astype(np.float64) ** 2).sum(axis=1)
    sse = counts @ se_row
    mean_d = (counts @ tgt.astype(np.float64)) / n
    sst = counts @ q_row - n * (mean_d**2).sum(axis=1)
    return 1.0 - sse / (sst + 1e-30)


def main() -> int:
    root = Path.cwd()
    rec_dir = root / "data/issue_1491/preds_recomputed"
    canon_dir = root / "data/issue_1491/preds"
    canon_dir.mkdir(parents=True, exist_ok=True)
    fits_dir = root / "eval_results/issue_1491/scale_ladder"

    data: dict[str, dict] = {}
    ceilings: dict[str, float] = {}
    for slug in SLUGS:
        src = rec_dir / f"{slug}_test_preds_ridge_recomputed.npz"
        with np.load(src) as arr:
            d = {
                "ci": arr["ci_te"].astype(np.int64),
                "target": arr["y_te"].astype(np.float32),
                "pred": arr["pred_te"].astype(np.float32),
            }
        np.savez_compressed(canon_dir / f"{slug}_test_preds_ridge.npz", **d)
        data[slug] = d
        fits = json.loads((fits_dir / f"fits_{slug}.json").read_text())
        ceilings[slug] = float(fits["ceiling_two_draw"]["ceiling_var_weighted_r"])
        print(f"[adjacent] {slug}: n={len(d['ci'])} ceiling={ceilings[slug]:.4f}")

    rng = np.random.default_rng(SEED)
    out: dict = {"n_boot": N_BOOT, "seed": SEED, "contrasts": []}
    for a, b in PAIRS:
        by_a = {int(c): i for i, c in enumerate(data[a]["ci"])}
        by_b = {int(c): i for i, c in enumerate(data[b]["ci"])}
        shared = sorted(set(by_a) & set(by_b))
        ia = np.array([by_a[c] for c in shared])
        ib = np.array([by_b[c] for c in shared])
        pa, ta = data[a]["pred"][ia], data[a]["target"][ia]
        pb, tb = data[b]["pred"][ib], data[b]["target"][ib]
        n = len(shared)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)
        r2a = _pooled_r2_boot(pa, ta, counts)
        r2b = _pooled_r2_boot(pb, tb, counts)
        delta = r2a - r2b
        delta_norm = r2a / ceilings[a] - r2b / ceilings[b]

        def _summ(boot: np.ndarray, point: float) -> dict:
            p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
            return {
                "point": point,
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "p_two_sided_bootstrap": float(max(p, 2.0 / N_BOOT)),
            }

        def _pooled_point(pred, tgt):
            sse = float(((tgt - pred) ** 2).sum())
            sst = float(((tgt - tgt.mean(axis=0, keepdims=True)) ** 2).sum())
            return 1.0 - sse / (sst + 1e-30)

        r2a_pt = _pooled_point(pa, ta)
        r2b_pt = _pooled_point(pb, tb)
        out["contrasts"].append(
            {
                "pair": f"{a} - {b}",
                "n_shared_contexts": n,
                "r2_a_point": r2a_pt,
                "r2_b_point": r2b_pt,
                "delta_raw": _summ(delta, r2a_pt - r2b_pt),
                "delta_ceiling_normalized": _summ(
                    delta_norm, r2a_pt / ceilings[a] - r2b_pt / ceilings[b]
                ),
            }
        )
        print(
            f"[adjacent] {a}-{b}: dRaw={r2a_pt - r2b_pt:+.4f} "
            f"p={out['contrasts'][-1]['delta_raw']['p_two_sided_bootstrap']:.4f}"
        )

    (fits_dir / "adjacent_contrasts.json").write_text(json.dumps(out, indent=1))
    print("[adjacent] written adjacent_contrasts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
