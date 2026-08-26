"""Fold-in analysis for #1901 round `mlp-scaling-densify` (plan v15 analyzer concern 1).

Paired four-cell context bootstrap of the registered verdict quantity D_gap, plus
per-rung paired MLP-minus-ridge held-out-R2 deltas with context-bootstrap CIs, all
recomputed zero-refit from the persisted per-rung test predictions
(HF issue1901_mlpdense/analysis_tensors/preds_L19_n*.npz, fp16, 1000x3584) against
the pinned test_1000 targets sliced from the #779 pass_b bundle.

Conventions (match the run artifact): pooled R2 = 1 - ||Y-Yhat||^2_F / ||Y-Ybar||^2_F
with SS_tot on the (resampled) test set's own mean; bootstrap resamples the 1,000 test
contexts jointly across every cell on ONE shared draw matrix (paired), n_boot=1000,
seed 1901 (the issue's battery seed). Point recomputes are validated against the
committed test_r2 values (fp16 prediction quantization tolerance).

Output: eval_results/issue_1901/paper_densify/dgap_context_bootstrap.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.orchestrate.hub import retry_transient

STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue1901_mlpdense_fold")
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1901_mlpdense/analysis_tensors"
PASSB = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
AGG = Path("eval_results/issue_1901/paper_densify/mlp_scaling_dense_L19.json")
OUT = Path("eval_results/issue_1901/paper_densify/dgap_context_bootstrap.json")
LAYER = 19
NS = [5000, 10000, 25000, 50000, 100000, 150000, 250000, 500000]
ENDPOINTS = (50000, 500000)
SEEDS = (42, 43, 44)
N_BOOT = 1000
BOOT_SEED = 1901
MARGIN = 0.01


def _dl(path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        retry_transient(
            lambda: hf_hub_download(REPO, path_in_repo, repo_type="dataset", local_dir=str(STAGE)),
            what=f"hf_hub_download({REPO}/{path_in_repo})",
        )
    )


def _pooled_r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    agg = json.loads(AGG.read_text())

    # ── targets: pinned test_1000 rows of the pass_b bundle at L19 ──
    bundle = torch.load(_dl(PASSB), map_location="cpu", mmap=True, weights_only=True)
    layer_pos = list(bundle["layers"]).index(LAYER)
    v19 = bundle["v_x"][:, layer_pos, :]  # (5000, 3584) fp32

    # test row ids from any persisted npz (identical across cells; assert below)
    z0 = np.load(_dl(f"{PREFIX}/preds_L19_n50000_mlp.npz"))
    rows = np.asarray(z0["rows"], dtype=np.int64)
    assert rows.shape == (1000,) and rows.max() < v19.shape[0], "rows must index pass_b"
    # canonical convention: issue779_fitter_fair_comparison._sha_ids (order-sensitive tobytes)
    sha = hashlib.sha256(rows.tobytes()).hexdigest()
    pin = agg["split"]["pinned_test_sha256"]
    y = v19[torch.as_tensor(rows)].to(torch.float32).numpy()
    print(f"targets: {y.shape}, id-sha match vs pinned: {sha == pin} ({sha[:12]})")

    # ── load every persisted fitted-arm prediction cell ──
    cells: dict[str, np.ndarray] = {}
    for n in NS:
        for arm in ("mlp", "ridge"):
            names = [f"preds_L19_n{n}_{arm}.npz"]
            if arm == "mlp" and n in ENDPOINTS:
                names += [f"preds_L19_n{n}_mlp_seed{s}.npz" for s in (43, 44)]
            for name in names:
                z = np.load(_dl(f"{PREFIX}/{name}"))
                assert np.array_equal(np.asarray(z["rows"], dtype=np.int64), rows), name
                cells[name.removeprefix("preds_L19_").removesuffix(".npz")] = np.asarray(
                    z["pred_fp16"], dtype=np.float32
                )
    print(f"cells loaded: {len(cells)}")

    # ── point recompute vs committed (fp16 quantization tolerance) ──
    checks = {}
    per_n = agg["per_n"]
    for n in NS:
        for arm in ("mlp", "ridge"):
            committed = per_n[str(n)][arm]["test_r2"]
            got = _pooled_r2(y, cells[f"n{n}_{arm}"])
            checks[f"n{n}_{arm}"] = {
                "committed": committed,
                "recomputed_fp16": got,
                "abs_delta": abs(got - committed),
            }
    for n in ENDPOINTS:
        for s in (43, 44):
            committed = per_n[str(n)]["mlp"]["seeds"][str(s)]["test_r2"]
            got = _pooled_r2(y, cells[f"n{n}_mlp_seed{s}"])
            checks[f"n{n}_mlp_seed{s}"] = {
                "committed": committed,
                "recomputed_fp16": got,
                "abs_delta": abs(got - committed),
            }
    worst = max(v["abs_delta"] for v in checks.values())
    print(f"point recompute worst |delta| = {worst:.2e} (fp16 preds)")
    assert worst < 5e-4, "fp16 recompute drifted past tolerance"

    # ── shared paired context bootstrap ──
    rng = np.random.default_rng(BOOT_SEED)
    n_test = y.shape[0]
    idx = rng.integers(0, n_test, size=(N_BOOT, n_test))
    # counts matrix: memory-safe batched reductions (never materialize y[idx])
    counts = np.zeros((N_BOOT, n_test), dtype=np.float32)
    for b in range(N_BOOT):
        counts[b] = np.bincount(idx[b], minlength=n_test)

    # per-row sufficient statistics per cell
    res2 = {k: ((y - p) ** 2).sum(axis=1) for k, p in cells.items()}  # (1000,)
    y_sq = (y**2).sum(axis=1)  # (1000,)
    ysum = counts @ y  # (B, d)
    ss_tot = counts @ y_sq - (ysum**2).sum(axis=1) / n_test  # (B,)

    draws = {k: 1.0 - (counts @ res2[k]) / ss_tot for k in cells}

    # verdict quantity per draw
    s_mlp = np.mean(
        [
            draws[f"n500000_mlp{sfx}"] - draws[f"n50000_mlp{sfx}"]
            for sfx in ("", "_seed43", "_seed44")
        ],
        axis=0,
    )
    s_ridge = draws["n500000_ridge"] - draws["n50000_ridge"]
    d_gap = (s_mlp - s_ridge) - MARGIN

    def ci(a: np.ndarray) -> dict:
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "mean": float(a.mean())}

    verdict_boot = {
        "S_mlp": ci(s_mlp),
        "S_ridge": ci(s_ridge),
        "slope_gap": ci(s_mlp - s_ridge),
        "D_gap": ci(d_gap),
        "frac_draws_D_gap_ge_0": float((d_gap >= 0).mean()),
        "point_D_gap_committed": agg["verdict"]["D_gap"],
    }
    print("verdict bootstrap:", json.dumps(verdict_boot, indent=1))

    per_rung_delta = {}
    for n in NS:
        d = draws[f"n{n}_mlp"] - draws[f"n{n}_ridge"]
        point = per_n[str(n)]["mlp"]["test_r2"] - per_n[str(n)]["ridge"]["test_r2"]
        per_rung_delta[str(n)] = {"point": point, **ci(d)}

    out = {
        "kind": "dgap-context-bootstrap",
        "description": (
            "Paired context bootstrap (analyzer concern 1, plan v15): the 1,000 pinned "
            "test contexts resampled jointly across every fitted cell on one shared draw "
            "matrix; R2 recomputed per draw from persisted fp16 predictions vs the pinned "
            "pass_b targets (test-set-own-mean SS_tot convention)."
        ),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "margin": MARGIN,
        "targets": {"source": PASSB, "layer": LAYER, "test_id_sha256_match": sha == pin},
        "point_recompute_checks": {
            "worst_abs_delta_vs_committed": worst,
            "note": "recomputed from fp16 persisted predictions; committed values are fp32",
        },
        "verdict_bootstrap": verdict_boot,
        "per_rung_mlp_minus_ridge_r2": per_rung_delta,
        "prediction_files": sorted(cells.keys()),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
