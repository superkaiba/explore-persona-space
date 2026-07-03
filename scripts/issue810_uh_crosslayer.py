#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², λ) in scientific docstrings + log messages.
"""Issue #810 `_uh` cross-layer pooled reads (plan v11 §4.6 item 4 — H3-uh).

Re-runs the 2026-07-02 22:30 cross-layer recipe VERBATIM on the extended-span
pools: does a layer-pooled single-number summary built from `mean_xbnd` /
`maxp_xbnd` (whole-turn pools INCLUDING the 5-token boundary block) beat (a)
its own max-selected selection-family band, (b) the committed per-layer best
(maxp 0.826 @ L21), and — descriptively — (c) the 22:30 answer-only pooled
benchmark 0.889?

Recipe of record: ``eval_results/issue_810/adhoc_crosslayer_pooled.json``
(main @ 4dc6c497, carried onto branch issue-810) — semantics fields read at
plan time (plan v11 §12 A13):
- ``layer-max`` = per-dim SIGNED max across the layers (the entry with the
  largest |value|, sign preserved);
- ``normed`` = per-layer L2-normalize each layer vector to unit norm BEFORE
  pooling (applied to BOTH the answer summary and the pooled c_C of a combo);
- 8 pooled targets = {mean_xbnd, maxp_xbnd} × {layer-mean, layer-max} ×
  {raw, normed}; predictors = pooled ``cc_last`` {layer-mean, layer-max}
  (16 cells, the 22:30-faithful grid) PLUS a per-c_C-layer read (8 × 28 = 224
  cells — the scope's "fit once against each c_C layer" reading).

CRITIC ADDITIONS (plan v10/v11): each pooled target ALSO gets (i) closed-form
7-fold LOFO point fits (carry-decidability under both folds per §1) and (ii)
its OWN pooled identity ceiling (predict the pooled target from itself through
the identical machinery — the per-layer 0.857 does not bound a pooled
construction; the 22:30 pooled 0.889 already exceeds it).

Read-out for pooled targets = trained-ridge only, DESCRIPTIVE (the fixed
direction r_B is per-layer and undefined for a layer-pooled summary — stated
limitation, matching the 22:30 read). Behaviors: sycophancy + refusal (the
scoped clean behaviors; harmful compliance stays quarantined at this target).

NULLS: 1000-perm label-shuffle per cell via ``issue810_batched_null`` (batched
GEMMs against the per-cell fold-cached dual-Gram — NO serial per-draw loop),
fresh ``default_rng(658)`` per cell (the recon-leg seeding convention), with
the selection family = max over ALL fitted cells per draw. Per-draw vectors
are PERSISTED alongside ``crosslayer_xbnd.json``.

Usage::

    # production (inside the GPU phase, after the uh capture):
    uv run python scripts/issue810_uh_crosslayer.py --device cuda \\
        --position-store-dir data/issue_810/store_uh \\
        --out eval_results/issue_810/user-header-newline-summary

    # smoke (tiny store, 2 cc layers, 50 perms):
    uv run python scripts/issue810_uh_crosslayer.py --smoke \\
        --position-store-dir /tmp/i810_uh_smoke/store --out /tmp/i810_uh_smoke/out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847) must bind BEFORE torch/numpy import — the pool
# freezes from OMP_NUM_THREADS at import time (tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_UH_SUBDIR,
    BETLEY_E0_HIGHM_FILE,
    HF_DATA_REPO,
    HF_PREFIX,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    battery_family_map,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    upload_out_dir,
)
from issue810_fit_readout import _e0_by_context, _rho, _trained_ridge_pred  # noqa: E402
from issue810_fit_reconstruction import (  # noqa: E402
    _fit_null_draws,
    _fit_one_cell,
    _load_cc_for_genre,
    _load_position_summaries,
    _lofo_cell_skill,
)

logger = logging.getLogger("issue810_uh_crosslayer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_SUMMARIES = ("mean_xbnd", "maxp_xbnd")
ANSWER_POOLS = ("layer-mean", "layer-max")
CC_POOLS = ("layer-mean", "layer-max")
NORMS = ("raw", "normed")
READOUT_BEHAVIORS = ("sycophancy", "refusal")


def _pool_layers(arr: np.ndarray, pool: str, normed: bool) -> np.ndarray:
    """Layer-pool an (Lc, H) stack to (H,) per the 22:30 recipe semantics.

    ``layer-mean`` = mean over the layer axis; ``layer-max`` = per-dim SIGNED
    max (the entry with the largest |value| per dim, sign preserved — the
    recipe-of-record ``layer_max_semantics`` field). ``normed`` L2-normalizes
    each layer's vector BEFORE pooling (``normed_semantics``).
    """
    a = np.asarray(arr, dtype=np.float64)
    assert a.ndim == 2, a.shape
    if normed:
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    if pool == "layer-mean":
        return a.mean(axis=0)
    idx = np.argmax(np.abs(a), axis=0)  # per-dim signed max
    return a[idx, np.arange(a.shape[1])]


def _target_key(base: str, apool: str, norm: str) -> str:
    return f"{base}|answer={apool}|{norm}"


def _assemble_pools(pos_summaries: dict, cc: dict, ctx_ids: list[str], cc_layers: list[int]):
    """Assemble the 8 pooled targets + pooled / per-layer c_C predictors.

    Returns ``(targets {target_key: (n, H)}, cc_pooled {(cc_pool, norm): (n, H)},
    cc_per_layer {layer_idx: (n, H)})`` per the 22:30 recipe (a combo's norm tag
    applies to BOTH the answer pooling and the pooled c_C).
    """
    targets: dict[str, np.ndarray] = {}
    for base in BASE_SUMMARIES:
        for apool in ANSWER_POOLS:
            for norm in NORMS:
                targets[_target_key(base, apool, norm)] = np.stack(
                    [_pool_layers(pos_summaries[c][base], apool, norm == "normed") for c in ctx_ids]
                )
    cc_pooled: dict[tuple[str, str], np.ndarray] = {}
    for cpool in CC_POOLS:
        for norm in NORMS:
            cc_pooled[(cpool, norm)] = np.stack(
                [_pool_layers(cc[c], cpool, norm == "normed") for c in ctx_ids]
            )
    cc_per_layer = {li: np.stack([cc[c][li] for c in ctx_ids]) for li in cc_layers}
    return targets, cc_pooled, cc_per_layer


def _load_inputs(args):
    """Load manifest-joined uh store rows + c_C + families + graded E0 (fail-loud)."""
    from huggingface_hub import hf_hub_download

    logger.info("[phase=load] manifest + uh store + c_C + graded E0 + families")
    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids_all = context_ids_from_manifest(man)
    local_dir = Path(args.position_store_dir) if args.position_store_dir else None
    if local_dir is not None:
        pos_man = load_json(local_dir / "manifest.json")
    else:
        pos_man = load_json(
            hf_hub_download(
                HF_DATA_REPO, f"{args.position_store_hf}/manifest.json", repo_type="dataset"
            )
        )
    if not pos_man.get("extended_boundary"):
        raise RuntimeError(
            "position store is NOT an extended-boundary store (no mean_xbnd/maxp_xbnd rows) — "
            "run issue810_extract_positions.py --extended-boundary first"
        )
    ctx_ids = [c for c in pos_man["context_ids"] if c in ctx_ids_all]
    pos_summaries, coverage = _load_position_summaries(ctx_ids, args.position_store_hf, local_dir)
    for c in ctx_ids:  # xbnd pools are span-length-independent — full coverage required
        for base in BASE_SUMMARIES:
            if coverage[c].get(base, 0) <= 0:
                raise RuntimeError(f"store missing {base} coverage for {c} (capture bug)")
    n = len(ctx_ids)
    n_cc_layers = 28  # the 7B c_C store depth (independent of the answer-side store depth)
    cc = _load_cc_for_genre("betley", ctx_ids, list(range(n_cc_layers)))
    fam_map = battery_family_map()
    groups = [fam_map[c] for c in ctx_ids]
    e0 = _e0_by_context(load_json(args.e0_highm), None)
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    cc_layers = args.cc_layers if args.cc_layers is not None else list(range(n_cc_layers))
    logger.info(
        "n=%d ctx, pca_dim=%d, cc_layers=%d, n_perms=%d, device=%s",
        n,
        pca_dim,
        len(cc_layers),
        args.n_perms,
        args.device,
    )
    return ctx_ids, pos_summaries, cc, groups, e0, pca_dim, cc_layers


def _h3_band_row(args, best: dict, band_97_5: float, best_ceiling) -> dict:
    """The H3-uh registered band row + its (b)/(c) comparators.

    (b) committed per-layer best (spans ALL committed rows — read directly);
    (c) the 22:30 pooled benchmark (descriptive, unregistered inline read).
    """
    per_layer_best = None
    if Path(args.committed_recon).is_file():
        blob = load_json(args.committed_recon)
        vals = [
            c["ridge_skill"]
            for cells_ in blob["by_summary"].values()
            for c in cells_
            if c.get("ridge_skill") is not None
        ]
        per_layer_best = max(vals) if vals else None
    bench_2230 = None
    if Path(args.committed_crosslayer).is_file():
        cx = load_json(args.committed_crosslayer)
        vals = [
            v["ridge_skill"]
            for summ in cx.get("by_summary", {}).values()
            for v in summ.get("reconstruction", {}).values()
            if v.get("ridge_skill") is not None
        ]
        bench_2230 = max(vals) if vals else None
    verdict = "clears_band" if best["ridge_skill"] > band_97_5 else "within_band"
    if best_ceiling is not None and band_97_5 >= best_ceiling:
        verdict += "; band_at_or_above_ceiling -> failure_to_reject"
    return {
        "statistic": best["ridge_skill"],
        "arg_cell": best["cell"],
        "band_97_5": band_97_5,
        "ceiling": best_ceiling,  # the winner's OWN pooled identity ceiling
        "verdict": verdict,
        "beats_per_layer_best": (
            None if per_layer_best is None else bool(best["ridge_skill"] > per_layer_best)
        ),
        "per_layer_best_committed": per_layer_best,
        "benchmark_2230_pooled": bench_2230,  # descriptive (unregistered inline read)
    }


def _pooled_readout(targets: dict, ctx_ids: list[str], e0: dict) -> dict:
    """Pooled read-out (trained-ridge only, DESCRIPTIVE — fixed r_B is per-layer
    and undefined for a layer-pooled summary; stated limitation, 22:30 read)."""
    readout: dict[str, dict] = {}
    for tkey, X in targets.items():
        readout[tkey] = {}
        for behavior in READOUT_BEHAVIORS:
            graded = e0.get(behavior, {}).get("graded", {})
            rates = e0.get(behavior, {}).get("rate", {})
            kept = [c for c in ctx_ids if c in graded]
            if len(kept) < 4:
                readout[tkey][behavior] = {"status": "insufficient", "n": len(kept)}
                continue
            idx = [ctx_ids.index(c) for c in kept]
            y = np.array([graded[c] for c in kept], dtype=np.float64)
            pred = _trained_ridge_pred(X[idx], y)
            y_rate = np.array([rates.get(c, np.nan) for c in kept], dtype=np.float64)
            readout[tkey][behavior] = {
                "n": len(kept),
                "rho_graded": _rho(pred, y),
                "rho_binary_rate": (_rho(pred, y_rate) if np.isfinite(y_rate).all() else None),
                "method": "trained_ridge (descriptive; fixed r_B undefined for pooled)",
            }
    return readout


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 uh cross-layer pooled reads (H3-uh)")
    ap.add_argument(
        "--position-store-hf",
        default=f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_UH_SUBDIR}",
        help="HF prefix of the extended-boundary aligned-subset store",
    )
    ap.add_argument("--position-store-dir", default=None, help="local store dir (GPU phase/smoke)")
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_810" / "user-header-newline-summary"),
    )
    ap.add_argument("--e0-highm", default=str(BETLEY_E0_HIGHM_FILE))
    ap.add_argument(
        "--committed-recon",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "reconstruction_skill_by_summary.json"
        ),
        help="committed round-1 per-layer skills (the H3 (b) per-layer-best comparator)",
    )
    ap.add_argument(
        "--committed-crosslayer",
        default=str(PROJECT_ROOT / "eval_results" / "issue_810" / "adhoc_crosslayer_pooled.json"),
        help="the 22:30 recipe-of-record JSON (the H3 (c) descriptive comparator)",
    )
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--cc-layers", nargs="*", type=int, default=None, help="cc layer-idx subset")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--upload-prefix",
        default=None,
        help="HF data-repo prefix to bulk-upload the out-dir *.json after the fit "
        "(set on an ephemeral lane; unset = no upload)",
    )
    ap.add_argument("--smoke", action="store_true", help="2 cc layers + 50 perms defaults")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    if args.smoke:
        args.n_perms = min(args.n_perms, 50)
        if args.cc_layers is None:
            args.cc_layers = [0, 1]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_ids, pos_summaries, cc, groups, e0, pca_dim, cc_layers = _load_inputs(args)
    n = len(ctx_ids)

    targets, cc_pooled, cc_per_layer = _assemble_pools(pos_summaries, cc, ctx_ids, cc_layers)

    # ── fit every cell (LOCO ridge + per-cell batched null); LOFO per target ──
    cells: list[dict] = []
    null_matrix: dict[str, list[float]] = {}  # cell_key -> per-draw skill

    def _one_cell(cell_key: str, X: np.ndarray, Y: np.ndarray) -> dict:
        fit, _y_pca = _fit_one_cell(X, Y, pca_dim)
        null_matrix[cell_key] = _fit_null_draws(
            X, Y, pca_dim, args.n_perms, SHUFFLE_NULL_SEED, device=args.device
        )
        return fit

    logger.info("[phase=fit_pooled_cc] %d targets x %d cc pools", len(targets), len(CC_POOLS))
    for tkey, Y in targets.items():
        norm = tkey.rsplit("|", 1)[1]
        for cpool in CC_POOLS:
            key = f"{tkey}|cc={cpool}"
            fit = _one_cell(key, cc_pooled[(cpool, norm)], Y)
            fit.update({"cell": key, "grid": "pooled_cc", "cc_pool": cpool, "target": tkey})
            cells.append(fit)
    logger.info("[phase=fit_per_cc_layer] %d targets x %d layers", len(targets), len(cc_layers))
    for tkey, Y in targets.items():
        for li in cc_layers:
            key = f"{tkey}|cc_layer={li}"
            fit = _one_cell(key, cc_per_layer[li], Y)
            fit.update({"cell": key, "grid": "per_cc_layer", "cc_layer": li, "target": tkey})
            cells.append(fit)

    # ── per-target LOFO point fits + OWN pooled identity ceilings ────────────
    per_target: dict[str, dict] = {}
    for tkey, Y in targets.items():
        norm = tkey.rsplit("|", 1)[1]
        ident_loco, _ = _fit_one_cell(Y.copy(), Y.copy(), pca_dim)
        per_target[tkey] = {
            "lofo_skill_by_cc_pool": {
                cpool: _lofo_cell_skill(cc_pooled[(cpool, norm)], Y, groups) for cpool in CC_POOLS
            },
            "identity_ceiling_loco": ident_loco["ridge_skill"],
            "identity_ceiling_lofo": _lofo_cell_skill(Y.copy(), Y.copy(), groups),
        }

    # ── selection-family band: max over ALL fitted cells per draw ────────────
    stack = np.stack([np.asarray(null_matrix[c["cell"]], dtype=np.float64) for c in cells])
    per_draw_max = stack.max(axis=0)  # (n_perms,)
    band_97_5 = float(np.percentile(per_draw_max, 97.5))
    best = max(cells, key=lambda c: c["ridge_skill"])
    best_ceiling = per_target[best["target"]]["identity_ceiling_loco"]

    band_row = _h3_band_row(args, best, band_97_5, best_ceiling)
    verdict = band_row["verdict"]

    readout = _pooled_readout(targets, ctx_ids, e0)

    dump_json(
        {
            "dv": "uh_crosslayer_pooled_reconstruction_and_readout",
            "recipe_of_record": "adhoc_crosslayer_pooled.json @ 4dc6c497 (verbatim semantics)",
            "n_contexts": n,
            "pca_target_dim": pca_dim,
            "base_summaries": list(BASE_SUMMARIES),
            "answer_pools": list(ANSWER_POOLS),
            "cc_pools": list(CC_POOLS),
            "norms": list(NORMS),
            "layer_max_semantics": "per-dim signed max across the layers",
            "normed_semantics": "per-layer L2-normalize each layer vector before pooling",
            "cc_layers": cc_layers,
            "cells": cells,
            "per_target": per_target,
            "band_row_h3": band_row,
            "readout_pooled_descriptive": readout,
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "crosslayer_xbnd.json",
    )
    # Per-draw null vectors persisted alongside (plan v10/v11 critic addition).
    dump_json(
        {
            "dv": "uh_crosslayer_nulls",
            "axes": "cell_key -> [per-draw ridge skill]",
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "selection_family": "max over all fitted cells per draw",
            "per_draw_max": [float(x) for x in per_draw_max],
            "nulls": null_matrix,
        },
        out_dir / "null_matrix_crosslayer_xbnd.json",
    )
    if args.upload_prefix:
        landed = upload_out_dir(out_dir, args.upload_prefix)
        logger.info("[phase=upload] verified crosslayer JSONs under %s/", landed)
    logger.info(
        "[phase=done] crosslayer: best %s = %.4f (band %.4f, ceiling %s) -> %s",
        best["cell"],
        best["ridge_skill"],
        band_97_5,
        best_ceiling,
        verdict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
