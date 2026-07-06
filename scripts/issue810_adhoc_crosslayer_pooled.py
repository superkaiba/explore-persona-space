#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ²) in scientific docstrings + log messages.
"""Issue #810 ad-hoc (user-chat, 0-GPU free analysis): cross-layer-POOLED summaries.

The #810 map picks the BEST layer per summary for both reconstruction R² and
read-out ρ — a selection over a free axis (layer), so the reported best is
selection-inflated. This ad-hoc asks: does a LAYER-POOLED representation, which
needs NO layer selection (its R² / ρ is a single honest number), match the
per-layer BEST?

Two pooling forms, crossed:
  * ANSWER summary pooled over all 28 layers: layer-MEAN and layer-MAX (per-dim
    signed max across layers).
  * c_C context vector pooled over all 28 layers: layer-MEAN and layer-MAX.
Plus the per-layer BEST (read from the #810 fit JSONs, NOT refit) as the
selection-inflated comparison column.

SCALE CONTROL: layer L2 norms differ ~35-40× (verified), so raw cross-layer
pooling is dominated by the highest-norm layers. Run BOTH raw and
per-layer-L2-normalized-before-pooling; report both. The layer-mean-c_C rows
are the HEADLINE; layer-max-c_C rows are secondary (still in the JSON + a
secondary figure panel), per the coordinator addition.

Reuses the #810 fit path VERBATIM (imports from src):
  * RECONSTRUCTION: robust_pca_basis(Y, 48) -> ridge_predict_loco_centered(Xc, Y_pca)
    -> skill_over_mean_r2  (X = pooled c_C, Y = pooled answer summary).
  * READOUT: PCA-reduce pooled summary to min(48, n-2) -> ridge_predict_loco_centered
    -> Spearman ρ vs graded E0, per high-m behavior. (c_C not involved in read-out;
    the trained ridge learns its own direction, so no fixed-r_B for the pooled form.)

Stores (all already on HF / local; NOT re-extracted):
  * #658 v0_summaries.pt: mean/last/maxp at 28 layers, {ctx:(28,3584)}.
  * #594 c_C last-input-token: (50,28,3584).
  * #810 answer_position_sweep/<ctx>.pt: 34 aligned positions × 28 layers (streamed).
  * #810 phase_c/e0_highm_graded.json: graded E0 for syco/refusal/harmful_compliance.
  * #810 reconstruction_skill_by_summary.json + readout_rho_by_summary.json: per-layer best.

Usage (0-GPU CPU):
    OMP_NUM_THREADS=8 uv run python scripts/issue810_adhoc_crosslayer_pooled.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# issue810_common lives with the in-flight #810 worktree scripts (not yet on main).
sys.path.insert(0, str(PROJECT_ROOT / ".claude" / "worktrees" / "issue-810" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_SUBDIR,
    HF_DATA_REPO,
    HF_PREFIX,
    HIGH_M_BEHAVIORS,
    I594_CC_LAST_FILE,
    I594_PROBE_POOL_HASH,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    PCA_TARGET_DIM_CAP,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
)
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)

torch.set_num_threads(8)

logger = logging.getLogger("issue810_adhoc_crosslayer_pooled")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_SUMMARIES = ["mean", "maxp", "turn_nl", "tail_1", "tail_8"]
ANSWER_POOLS = ["layer-mean", "layer-max"]
CC_POOLS = ["layer-mean", "layer-max"]
NORMS = ["raw", "normed"]
EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_810"


# ── inputs ────────────────────────────────────────────────────────────────────


def _load_free_summaries():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_V0_SUMMARIES, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["summaries"], blob["capture_layers"]


def _load_cc(ctx_ids, capture_layers):
    """#594 last-input-token c_C, {ctx: (28,H) fp32 np}, probe_pool_hash pinned."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I594_CC_LAST_FILE, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    pph = blob.get("probe_pool_hash")
    if pph != I594_PROBE_POOL_HASH:
        raise RuntimeError(f"#594 c_C probe_pool_hash drift: {pph} != {I594_PROBE_POOL_HASH}")
    tensor = blob["tensor"]
    iid_to_row = {iid: i for i, iid in enumerate(blob["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"c_C store missing {len(missing)} contexts: {missing[:5]}")
    return {c: tensor[iid_to_row[c]][capture_layers].float().numpy() for c in ctx_ids}


def _load_position_summaries(ctx_ids):
    """{ctx: {position: (28,H) np}} + {ctx: coverage} streamed per-context from HF."""
    from huggingface_hub import hf_hub_download

    out, cov = {}, {}
    hf_prefix = f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}"
    for c in ctx_ids:
        path = hf_hub_download(HF_DATA_REPO, f"{hf_prefix}/{c}.pt", repo_type="dataset")
        blob = torch.load(path, weights_only=False)
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, 28, H)
        out[c] = {name: pv[i] for i, name in enumerate(blob["positions"])}
        cov[c] = dict(blob["coverage"])
    return out, cov


# ── per-layer stacks + pooling ─────────────────────────────────────────────────


def _summary_stack(summary, ctx_ids, free_summaries, pos_summaries, coverage):
    """(n_kept, 28, H) per-layer summary stack + kept ctx list (coverage-aware).

    Free recipes (mean/maxp/...) are always covered. Position recipes drop a
    context whose coverage for this position is 0 — coverage is per-position (a
    fixed answer position), NOT per-layer, so the kept set is layer-invariant.
    """
    kept, rows = [], []
    for c in ctx_ids:
        if summary in ("mean", "last", "maxp"):
            rows.append(free_summaries[summary][c].float().numpy())  # (28,H)
            kept.append(c)
        else:
            if coverage[c].get(summary, 0) <= 0:
                continue
            rows.append(pos_summaries[c][summary])  # (28,H)
            kept.append(c)
    return np.stack(rows), kept  # (n_kept, 28, H)


def _cc_stack(kept, cc):
    return np.stack([cc[c] for c in kept])  # (n_kept, 28, H)


def _pool(stack: np.ndarray, pool: str, norm: str) -> np.ndarray:
    """Pool a (n, 28, H) per-layer stack to (n, H) over the layer axis.

    norm='normed' -> per-layer L2-normalize each (n, layer) vector to unit norm
    BEFORE pooling (so no single high-norm layer dominates the pool). Zero-norm
    rows (degenerate) are left as zeros (division guarded).
    norm='raw'    -> pool the raw vectors.
    pool='layer-mean' -> mean over the 28 layers.
    pool='layer-max'  -> per-DIM signed max over the 28 layers (the element with
                          the largest signed value across layers, per hidden dim).
    """
    X = stack.astype(np.float64)
    if norm == "normed":
        nrm = np.linalg.norm(X, axis=2, keepdims=True)  # (n,28,1)
        X = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 1e-9)
    if pool == "layer-mean":
        return X.mean(axis=1)  # (n,H)
    if pool == "layer-max":
        return X.max(axis=1)  # (n,H) per-dim signed max across layers
    raise ValueError(pool)


# ── the two fits (verbatim #810 path) ──────────────────────────────────────────


def _pca_reduce(M: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """PCA-reduce a pooled (n,H) matrix to its top min(48,n-2) dims (ONE SVD).

    Returned (M_pca (n,k), pca_dim, used_gesvd) is reused as BOTH the
    reconstruction TARGET (Y_pca, fitted from pooled c_C) AND — verbatim #810
    read-out path — the read-out PREDICTOR reduction. The pooled answer-summary
    SVD is identical across the two c_C pools and across recon-vs-readout, so
    computing it once per (summary, apool, norm) is the #722 vectorize-first
    discipline (avoids ~5× redundant (50,3584) SVDs under VM contention).
    """
    n = M.shape[0]
    k = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    mu, comps, used_gesvd = robust_pca_basis(M, k)
    return (M - mu) @ comps.T, int(comps.shape[0]), bool(used_gesvd)


def _recon_r2(Xc: np.ndarray, Y_pca: np.ndarray, pca_dim: int, used_gesvd: bool) -> dict:
    """Held-out ridge skill-over-mean R² for pooled c_C -> pooled answer summary.

    Y_pca is the PRE-REDUCED pooled answer summary (shared across c_C pools).
    """
    ridge_pred = ridge_predict_loco_centered(Xc, Y_pca)
    r = skill_over_mean_r2(ridge_pred, Y_pca)
    return {
        "n": int(Y_pca.shape[0]),
        "pca_dim": pca_dim,
        "used_gesvd_fallback": used_gesvd,
        "ridge_skill": float(r["skill"]),
        "ridge_median_per_dim_r2": float(r["median_per_dim_r2"]),
    }


def _readout_rho(Xp: np.ndarray, y: np.ndarray) -> float | None:
    """Held-out LOCO-ridge Spearman ρ of predicted vs measured E0 from pooled summary.

    Xp is the PRE-REDUCED pooled summary predictor (shared with the recon target
    reduction — identical robust_pca_basis on the same pooled (n,H) matrix).
    """
    if len(y) < 4 or np.std(y) < 1e-9:
        return None
    pred = ridge_predict_loco_centered(Xp, y.reshape(-1, 1))[:, 0]
    if np.std(pred) < 1e-9:
        return None
    rho, _ = spearmanr(pred, y)
    return None if np.isnan(rho) else float(rho)


# ── per-layer-best baselines (read from existing #810 JSONs; NOT refit) ─────────


def _per_layer_best_recon() -> dict[str, float]:
    d = load_json(EVAL_DIR / "reconstruction_skill_by_summary.json")
    out = {}
    for s in BASE_SUMMARIES:
        cells = d["by_summary"].get(s, [])
        vals = [c["ridge_skill"] for c in cells if c.get("ridge_skill") is not None]
        out[s] = float(max(vals)) if vals else None
    return out


def _per_layer_best_readout() -> dict[str, dict[str, float]]:
    """{summary: {behavior: best trained_ridge rho_graded over layers}}."""
    d = load_json(EVAL_DIR / "readout_rho_by_summary.json")
    best: dict[str, dict[str, float]] = {s: {} for s in BASE_SUMMARIES}
    for c in d["cells"]:
        if c.get("method") != "trained_ridge":
            continue
        s, beh, rho = c.get("summary"), c.get("behavior"), c.get("rho_graded")
        if s in best and beh in HIGH_M_BEHAVIORS and rho is not None:
            best[s][beh] = max(best[s].get(beh, -1e9), float(rho))
    return best


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    logger.info("[phase=load] manifest + free summaries + c_C + position store + E0")
    from huggingface_hub import hf_hub_download

    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries()
    assert len(capture_layers) == 28, capture_layers
    cc = _load_cc(ctx_ids, capture_layers)
    pos_summaries, coverage = _load_position_summaries(ctx_ids)

    e0 = load_json(EVAL_DIR / "phase_c" / "e0_highm_graded.json")
    graded = {
        beh: {k: v for k, v in blk["per_context_graded_mean"].items() if v is not None}
        for beh, blk in e0["by_behavior"].items()
    }

    per_layer_best_recon = _per_layer_best_recon()
    per_layer_best_readout = _per_layer_best_readout()

    # Precompute pooled c_C for every (cc_pool, norm) once, keyed by the FULL ctx
    # order (a summary later restricts to its kept ctx by index).
    ctx_index = {c: i for i, c in enumerate(ctx_ids)}
    cc_stack_full = _cc_stack(ctx_ids, cc)  # (50,28,H)

    results: dict = {
        "dv": "crosslayer_pooled_reconstruction_and_readout",
        "predictor_recon": "cc_last_input_token (#594), layer-pooled",
        "predictor_readout": "pooled answer summary -> trained LOCO-ridge",
        "n_contexts": len(ctx_ids),
        "base_summaries": BASE_SUMMARIES,
        "answer_pools": ANSWER_POOLS,
        "cc_pools": CC_POOLS,
        "norms": NORMS,
        "high_m_behaviors": list(HIGH_M_BEHAVIORS),
        "layer_max_semantics": "per-dim signed max across the 28 layers",
        "normed_semantics": "per-layer L2-normalize each layer vector to unit norm before pooling",
        "per_layer_best_recon": per_layer_best_recon,
        "per_layer_best_readout": per_layer_best_readout,
        "by_summary": {},
        "reproducibility": reproducibility_metadata(),
    }

    for summary in BASE_SUMMARIES:
        stack, kept = _summary_stack(summary, ctx_ids, free_summaries, pos_summaries, coverage)
        n_kept = len(kept)
        logger.info("[%s] n_kept=%d", summary, n_kept)
        # kept-context c_C stack (restrict full stack by index)
        cc_kept = cc_stack_full[[ctx_index[c] for c in kept]]  # (n_kept,28,H)
        ent: dict = {"n_kept": n_kept, "reconstruction": {}, "readout": {}}

        # Precompute per-(apool, norm): the pooled answer summary + its ONE PCA
        # reduction (shared by recon-target AND readout-predictor — the same
        # (n,H) matrix, so one SVD), and the pooled c_C per (cpool, norm).
        # Every high-m behavior has a graded score for all 50 contexts (verified),
        # so the readout kept-set == the summary kept-set — no per-behavior subset,
        # the shared PCA is exact.
        for norm in NORMS:
            cc_pooled = {cp: _pool(cc_kept, cp, norm) for cp in CC_POOLS}  # (n_kept,H)
            for apool in ANSWER_POOLS:
                Ay = _pool(stack, apool, norm)  # (n_kept,H)
                Ay_pca, pca_dim, used_gesvd = _pca_reduce(Ay)  # ONE SVD, reused below
                # RECONSTRUCTION: pooled c_C -> pooled answer (target = Ay_pca)
                for cpool in CC_POOLS:
                    ent["reconstruction"][f"answer={apool}|cc={cpool}|{norm}"] = _recon_r2(
                        cc_pooled[cpool], Ay_pca, pca_dim, used_gesvd
                    )
                # READOUT: pooled answer summary (predictor = Ay_pca) -> graded E0
                for beh in HIGH_M_BEHAVIORS:
                    g = graded.get(beh, {})
                    if not all(c in g for c in kept):
                        # Defensive: a behavior missing a kept-ctx score would break
                        # the shared-PCA assumption (readout subset != summary set).
                        rho = None
                    else:
                        y = np.array([g[c] for c in kept], dtype=np.float64)
                        rho = _readout_rho(Ay_pca, y)
                    ent["readout"][f"{apool}|{norm}|{beh}"] = rho

        results["by_summary"][summary] = ent

    out_path = EVAL_DIR / "adhoc_crosslayer_pooled.json"
    dump_json(results, out_path)
    logger.info("[phase=done] wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
