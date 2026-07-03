#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #920 S4: the full LOFO fit battery (torch float64 batched, GPU).

Runs, per plan §3.5-S4, on the probe-reduced summary matrices from S3's stores:

- F1 map fits — 34,652 (context-cell × answer-target-cell) LOFO 7-fold ridge
  fits into the train-fold Gram-PCA target basis (k=34), λ by PRESS-LOO over
  ``issue658_fit_predictors.RIDGE_LAMBDAS``, dual-space solves. X-side caches
  computed ONCE per (cell × fold) and shared across ALL targets; target PCAs
  ONCE per (a-cell × fold); every Y-solve stacked into batched GEMM/eigh chunks.
  A per-cell Python fit loop is BANNED — the hot loop dispatches
  ``issue920_fit_core.batched_press_predict`` and the §7 G2 gate asserts on THAT
  function vs a contained serial reference (atol=1e-8 float64) before the
  battery runs.
- F2c/F2a read-outs — scalar ridge per (predictor cell × behavior), dual-space on
  the raw standardized summary, per-behavior PRESS λ (batched as 7 target
  columns of one stacked solve per cell × fold).
- Chain read-outs — ridge fit on TRUE set-A answer summaries in the map's
  train-fold PCA basis, applied UNCHANGED to M(c) (never refit on
  reconstructions); the chain-vs-oracle gap is computed against the
  PCA-basis oracle.

Regimes: R1 (A→A) / R2 (B-input) / R3 (B-target) / R4 (both) for skill;
R5–R8 oracle ρ; R9/R10 chain ρ. Persists ALL pooled held-out predictions
(the DV-2/3 stored-prediction nulls re-correlate them) + the three per-cell
eval JSONs (plan §6.5). K3 anchor gate: ctx_ah_nl × ans_content_mean best
matched-layer R1 skill must land in [0.6, 0.9] (else fail loud pre-nulls).

Usage::

    EPM_FIT_DEVICE=cuda uv run python scripts/issue920_fit_lofo.py \\
        --store-root data/issue_920 --eval-out eval_results/issue_920

    # synthetic CPU smoke (tiny H/layers, real fold structure + real E0):
    uv run python scripts/issue920_fit_lofo.py --synthetic-smoke \\
        --store-root /tmp/i920_smoke_fits --eval-out /tmp/i920_smoke_fits/eval
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): dotenv before torch's import-time pool freeze.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))


import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue920_common import (  # noqa: E402
    E0_BEHAVIORS,
    dump_json,
    load_battery,
    load_e0_graded,
    lofo_folds,
    reproducibility_metadata,
    write_sentinel,
)
from issue920_fit_core import (  # noqa: E402
    PCA_K,
    FoldXCache,
    batched_pca_project,
    batched_press_predict,
    batched_press_predict_per_column,
    enumerate_map_cells,
    excluded_mask,
    fit_device,
    load_reduced_matrices,
    pca_apply,
    serial_reference_map_fit,
    union_excluded,
)

logger = logging.getLogger("issue920_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PAIR_CHUNK = int(os.environ.get("EPM_I920_PAIR_CHUNK", "4096"))


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def pooled_spearman(preds: np.ndarray, y_ranks: np.ndarray) -> np.ndarray:
    """Batched Spearman ρ of (..., n) prediction rows vs a FIXED rank vector (n,).

    Prediction ranks via argsort (tie-free in practice: fp64 ridge outputs);
    the target ranks use average-tie ranks (scipy ``rankdata``, computed once by
    the caller). Degenerate (constant) prediction rows → NaN (the #810 ``_rho``
    convention for observed reads).
    """
    flat = preds.reshape(-1, preds.shape[-1])
    n = flat.shape[-1]
    order = np.argsort(flat, axis=-1)
    ranks = np.empty_like(flat)
    np.put_along_axis(ranks, order, np.arange(n, dtype=flat.dtype)[None, :], axis=-1)
    rp = ranks - ranks.mean(axis=-1, keepdims=True)
    ry = y_ranks - y_ranks.mean()
    num = rp @ ry
    den = np.sqrt((rp * rp).sum(axis=-1) * (ry * ry).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    rho[flat.std(axis=-1) < 1e-9] = np.nan
    return rho.reshape(preds.shape[:-1])


def run_g2_gate(
    red_A: dict,
    red_B: dict,
    fam_map: dict[str, str],
    ctx_ids: list[str],
    device: torch.device,
    k: int,
) -> dict:
    """§7 G2: batched fit path reproduces the serial reference on 2–3 cells, atol 1e-8.

    Gate cells (spanning the pairing blocks): the anchor cell
    (ctx_ah_nl × ans_content_mean @ the mid-late layer), one last-k × position
    cell, and one pooled × pooled cell. The batched side dispatches the SAME
    ``FoldXCache`` + ``batched_press_predict`` the production battery runs
    (never an unused sibling); PCA is cross-checked against the #810
    ``_gram_top_k_pca`` primitive on one fold.
    """
    lc = red_A["n_layers"]
    mid = min(18, lc - 1)
    names_c, names_a = red_A["ctx_cell_names"], red_A["ans_cell_names"]
    gate_pairs = [
        (names_c.index(f"ctx_ah_nl@L{mid}"), names_a.index(f"ans_content_mean@L{mid}")),
        (names_c.index("ctx_lastk_1@L0"), names_a.index("pos_tail_1@L0")),
        (names_c.index("ctx_wt_pool_meanmean"), names_a.index("ans_content_pool_meanmean")),
    ]
    groups = [fam_map[c] for c in ctx_ids]
    folds = lofo_folds(ctx_ids, fam_map)
    XA, YA = red_A["X_ctx"], red_A["Y_ans"]
    XB, YB = red_B["X_ctx"], red_B["Y_ans"]

    # PCA primitive cross-check on the first gate cell, first fold (vs #810).
    from issue810_adhoc_lofo_heatmaps import _gram_top_k_pca

    _fam0, tr0, _te0 = folds[0]
    a0 = gate_pairs[0][1]
    Ytr0 = YA[a0][tr0].double()
    mu_b, comps_b = batched_pca_project(Ytr0.unsqueeze(0), k)
    mu_s, comps_s = _gram_top_k_pca(Ytr0.numpy(), k)
    kk = comps_s.shape[0]
    assert np.allclose(mu_b[0, 0].numpy(), mu_s, atol=1e-8), "PCA mean drift vs _gram_top_k_pca"
    assert np.allclose(comps_b[0, :kk].numpy(), comps_s, atol=1e-8), (
        "PCA components drift vs the #810 _gram_top_k_pca primitive"
    )

    max_abs = 0.0
    for c_i, a_i in gate_pairs:
        serial = serial_reference_map_fit(
            XA[c_i].numpy().astype(np.float64),
            YA[a_i].numpy().astype(np.float64),
            XB[c_i].numpy().astype(np.float64),
            YB[a_i].numpy().astype(np.float64),
            groups,
            k,
            device=device,  # same device as the batched side (atol=1e-8 is device-local)
        )
        ss = {r: [0.0, 0.0] for r in ("R1", "R2", "R3", "R4")}
        for _fam, tr, te in folds:
            cache = FoldXCache(XA[c_i].unsqueeze(0), tr, te, XB[c_i].unsqueeze(0), device)
            mu_p, comps = batched_pca_project(YA[a_i][tr].double().unsqueeze(0).to(device), k)
            Ytr_pca = pca_apply(YA[a_i][tr].double().unsqueeze(0).to(device), mu_p, comps)
            YteA = pca_apply(YA[a_i][te].double().unsqueeze(0).to(device), mu_p, comps)
            YteB = pca_apply(YB[a_i][te].double().unsqueeze(0).to(device), mu_p, comps)
            ymu = Ytr_pca.mean(dim=1, keepdim=True)
            predA_c, predB_c, _best = batched_press_predict(
                cache, torch.zeros(1, dtype=torch.long, device=device), Ytr_pca - ymu
            )
            predA, predB = predA_c + ymu, predB_c + ymu
            ss["R1"][0] += float(((YteA - predA) ** 2).sum())
            ss["R1"][1] += float(((YteA - ymu) ** 2).sum())
            ss["R2"][0] += float(((YteA - predB) ** 2).sum())
            ss["R2"][1] += float(((YteA - ymu) ** 2).sum())
            ss["R3"][0] += float(((YteB - predA) ** 2).sum())
            ss["R3"][1] += float(((YteB - ymu) ** 2).sum())
            ss["R4"][0] += float(((YteB - predB) ** 2).sum())
            ss["R4"][1] += float(((YteB - ymu) ** 2).sum())
        for r in ("R1", "R2", "R3", "R4"):
            batched_skill = 1.0 - ss[r][0] / ss[r][1]
            diff = abs(batched_skill - serial[r])
            max_abs = max(max_abs, diff)
            assert diff <= 1e-8, (
                f"[g2-equiv-assert] pair ({names_c[c_i]}, {names_a[a_i]}) {r}: "
                f"batched {batched_skill:.12f} vs serial {serial[r]:.12f} (|Δ|={diff:.2e})"
            )
    logger.info("[g2] batched-vs-serial map-fit equivalence PASS (max |Δskill| = %.2e)", max_abs)
    return {"max_abs_skill_diff": max_abs, "n_gate_cells": len(gate_pairs), "atol": 1e-8}


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #920 S4: batched LOFO fit battery")
    ap.add_argument("--store-root", default=str(PROJECT_ROOT / "data" / "issue_920"))
    ap.add_argument("--eval-out", default=str(PROJECT_ROOT / "eval_results" / "issue_920"))
    ap.add_argument(
        "--preds-out",
        default=None,
        help="default: <store-root>/preds (persisted held-out predictions)",
    )
    ap.add_argument("--pair-chunk", type=int, default=PAIR_CHUNK)
    ap.add_argument("--skip-g2", action="store_true")
    ap.add_argument("--skip-anchor-gate", action="store_true")
    ap.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="generate tiny synthetic stores (real battery families + real E0) "
        "and run the FULL battery on them (CPU)",
    )
    ap.add_argument("--smoke-h", type=int, default=16)
    ap.add_argument("--smoke-layers", type=int, default=2)
    ap.add_argument(
        "--smoke-zero-coverage-b",
        default=None,
        metavar="FAMILY",
        help="synthetic smoke only: zero out ALL probes' validity for FAMILY in "
        "set B (context 0), leaving set A valid — exercises the union-exclusion "
        "contract (round-1 blocker set-b-zero-coverage-not-masked)",
    )
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(fit_device())
    logger.info("[phase=setup] device=%s", device)
    store_root = Path(args.store_root)
    eval_out = Path(args.eval_out)
    preds_out = Path(args.preds_out) if args.preds_out else store_root / "preds"
    preds_out.mkdir(parents=True, exist_ok=True)

    instances, fam_map = load_battery()
    ctx_ids = [i["id"] for i in instances]
    if args.synthetic_smoke:
        make_synthetic_stores(
            store_root,
            ctx_ids,
            args.smoke_layers,
            args.smoke_h,
            zero_coverage_b=args.smoke_zero_coverage_b,
        )
    logger.info("[phase=load_reduced] set A + set B stores")
    red_A = load_reduced_matrices(store_root / "summaries_setA", ctx_ids)
    red_B = load_reduced_matrices(store_root / "summaries_setB", ctx_ids)
    lc = red_A["n_layers"]
    k = min(PCA_K, max(1, min(len(tr) for _f, tr, _te in lofo_folds(ctx_ids, fam_map)) - 2))
    assert lc == red_B["n_layers"]
    logger.info(
        "cells: ctx=%d ans=%d layers=%d pca_k=%d",
        len(red_A["ctx_cell_names"]),
        len(red_A["ans_cell_names"]),
        lc,
        k,
    )

    e0 = load_e0_graded()
    E0 = np.stack([[e0[b][c] for c in ctx_ids] for b in E0_BEHAVIORS], axis=1)  # (50, 7)
    E0_t = torch.from_numpy(E0).double()

    if not args.skip_g2:
        logger.info("[phase=g2_gate] batched-vs-serial oracle equivalence")
        g2 = run_g2_gate(red_A, red_B, fam_map, ctx_ids, device, k)
    else:
        g2 = {"skipped": True}

    folds = lofo_folds(ctx_ids, fam_map)
    c_map, a_map = enumerate_map_cells(lc)
    n_map = len(c_map)
    n_ctx_cells = len(red_A["ctx_cell_names"])
    n_ans_cells = len(red_A["ans_cell_names"])
    n_pred_cells = n_ctx_cells + n_ans_cells
    n_beh = len(E0_BEHAVIORS)
    n = len(ctx_ids)

    # excluded-cell masks (zero-valid-probe families, plan §3.3). UNION of BOTH
    # stores' exclusion lists — a set-B-only gap would otherwise silently
    # zero-fill into R2/R3/R4, the B-side read-outs, the identity ceiling, and
    # every null band (round-1 blocker `set-b-zero-coverage-not-masked`).
    excluded_union, excluded_by_source = union_excluded(red_A, red_B)
    if excluded_union:
        logger.warning(
            "excluded families (union A|B): %s (A=%s, B=%s)",
            excluded_union,
            excluded_by_source["set_A"],
            excluded_by_source["set_B"],
        )
    ex_ctx = excluded_mask(red_A["ctx_cell_names"], excluded_union)
    ex_ans = excluded_mask(red_A["ans_cell_names"], excluded_union)
    ex_map = ex_ctx[c_map] | ex_ans[a_map]

    # accumulators
    map_ss = np.zeros((n_map, 4, 2), dtype=np.float64)  # regime × (res, base)
    ceil_ss = np.zeros((n_ans_cells, 2), dtype=np.float64)  # Y_A→Y_B identity ceiling (R3)
    map_predA = np.zeros((n_map, n, k), dtype=np.float16)
    map_predB = np.zeros((n_map, n, k), dtype=np.float16)
    ro_predA = np.zeros((n_pred_cells, n, n_beh), dtype=np.float32)
    ro_predB = np.zeros((n_pred_cells, n, n_beh), dtype=np.float32)
    ch_predA = np.zeros((n_map, n, n_beh), dtype=np.float32)
    ch_predB = np.zeros((n_map, n, n_beh), dtype=np.float32)
    ch_oracle = np.zeros((n_ans_cells, n, n_beh), dtype=np.float32)  # weights on TRUE YteA_pca
    ypca_A = np.zeros((n_ans_cells, n, k), dtype=np.float16)  # per-fold-basis targets (pooled)
    ypca_B = np.zeros((n_ans_cells, n, k), dtype=np.float16)

    X_pred_A = torch.cat([red_A["X_ctx"], red_A["Y_ans"]], dim=0)  # predictors: ctx then ans
    X_pred_B = torch.cat([red_B["X_ctx"], red_B["Y_ans"]], dim=0)

    fit_t0 = time.time()
    block_times: dict[str, float] = {}
    for fold_i, (fam, tr, te) in enumerate(folds):
        tf0 = time.time()
        logger.info(
            "[phase=fits] fold %d/7 (holding out %s: %d contexts)", fold_i + 1, fam, len(te)
        )
        cache = FoldXCache(X_pred_A, tr, te, X_pred_B, device)
        # target PCAs ONCE per (a-cell × fold), batched
        YA_dev = red_A["Y_ans"].double().to(device)
        YB_dev = red_B["Y_ans"].double().to(device)
        mu_p, comps = batched_pca_project(YA_dev[:, tr], k)
        Ytr_pca = pca_apply(YA_dev[:, tr], mu_p, comps)  # (A, m, k)
        YteA_pca = pca_apply(YA_dev[:, te], mu_p, comps)  # (A, n_te, k)
        YteB_pca = pca_apply(YB_dev[:, te], mu_p, comps)
        ymu = Ytr_pca.mean(dim=1, keepdim=True)  # (A, 1, k)
        Ytr_pca_c = Ytr_pca - ymu
        te_np = np.asarray(te)
        ypca_A[:, te_np] = YteA_pca.cpu().numpy().astype(np.float16)
        ypca_B[:, te_np] = YteB_pca.cpu().numpy().astype(np.float16)
        # identity ceiling: pred = YteA_pca, target = YteB_pca, base = ymu
        ceil_ss[:, 0] += ((YteB_pca - YteA_pca) ** 2).sum(dim=(1, 2)).cpu().numpy()
        ceil_ss[:, 1] += ((YteB_pca - ymu) ** 2).sum(dim=(1, 2)).cpu().numpy()

        # chain weights per (a-cell × behavior): ridge on TRUE train PCA coords → E0
        mu34 = Ytr_pca.mean(dim=1, keepdim=True)
        sd34 = Ytr_pca.std(dim=1, correction=0, keepdim=True) + 1e-9
        Xn_ch = (Ytr_pca - mu34) / sd34  # (A, m, k)
        ymu_beh = E0_t[tr].mean(dim=0).to(device)  # (7,)
        Ytr_e0_c = (E0_t[tr].to(device) - ymu_beh).unsqueeze(0).expand(n_ans_cells, -1, -1)
        ch_cache = _ChainCache(Xn_ch, cache.lambdas)
        w_chain = ch_cache.per_column_weights(Ytr_e0_c)  # (A, k, 7)
        # PCA-basis oracle (the chain-vs-oracle gap read): weights on TRUE YteA_pca
        z_or = (YteA_pca - mu34) / sd34
        ch_oracle[:, te_np] = (
            (torch.bmm(z_or, w_chain) + ymu_beh.view(1, 1, -1)).cpu().numpy()
        ).astype(np.float32)

        # F1 map fits + chain application, chunked over the 34,652 pairs
        for lo in range(0, n_map, args.pair_chunk):
            hi = min(lo + args.pair_chunk, n_map)
            c_sel = torch.from_numpy(c_map[lo:hi]).to(device)  # ctx cells are cache[0:542]
            a_sel_np = a_map[lo:hi]
            a_sel = torch.from_numpy(a_sel_np).to(device)
            predA_c, predB_c, _best = batched_press_predict(cache, c_sel, Ytr_pca_c[a_sel])
            ymu_sel = ymu[a_sel]
            predA = predA_c + ymu_sel
            predB = predB_c + ymu_sel
            tA, tB = YteA_pca[a_sel], YteB_pca[a_sel]
            base = ((tA - ymu_sel) ** 2).sum(dim=(1, 2)).cpu().numpy()
            baseB = ((tB - ymu_sel) ** 2).sum(dim=(1, 2)).cpu().numpy()
            map_ss[lo:hi, 0, 0] += ((tA - predA) ** 2).sum(dim=(1, 2)).cpu().numpy()
            map_ss[lo:hi, 0, 1] += base
            map_ss[lo:hi, 1, 0] += ((tA - predB) ** 2).sum(dim=(1, 2)).cpu().numpy()
            map_ss[lo:hi, 1, 1] += base
            map_ss[lo:hi, 2, 0] += ((tB - predA) ** 2).sum(dim=(1, 2)).cpu().numpy()
            map_ss[lo:hi, 2, 1] += baseB
            map_ss[lo:hi, 3, 0] += ((tB - predB) ** 2).sum(dim=(1, 2)).cpu().numpy()
            map_ss[lo:hi, 3, 1] += baseB
            map_predA[lo:hi, te_np.reshape(-1)] = predA.cpu().numpy().astype(np.float16)
            map_predB[lo:hi, te_np.reshape(-1)] = predB.cpu().numpy().astype(np.float16)
            # chain: frozen (a, behavior) weights applied UNCHANGED to M(c)
            zA = (predA - mu34[a_sel]) / sd34[a_sel]
            zB = (predB - mu34[a_sel]) / sd34[a_sel]
            chA = torch.bmm(zA, w_chain[a_sel]) + ymu_beh.view(1, 1, -1)
            chB = torch.bmm(zB, w_chain[a_sel]) + ymu_beh.view(1, 1, -1)
            ch_predA[lo:hi, te_np] = chA.cpu().numpy().astype(np.float32)
            ch_predB[lo:hi, te_np] = chB.cpu().numpy().astype(np.float32)
        block_times.setdefault("map_per_fold_s", 0.0)
        block_times["map_per_fold_s"] += time.time() - tf0

        # F2 read-outs: all 1560 predictor cells, per-behavior λ, one stacked solve
        tr_t = torch.from_numpy(np.asarray(tr))
        Ytr_ro = (E0_t[tr_t].to(device) - ymu_beh).unsqueeze(0).expand(n_pred_cells, -1, -1)
        roA_c, roB_c, _b = batched_press_predict_per_column(
            cache, torch.arange(n_pred_cells, device=device), Ytr_ro
        )
        ro_predA[:, te_np] = (roA_c + ymu_beh.view(1, 1, -1)).cpu().numpy().astype(np.float32)
        ro_predB[:, te_np] = (roB_c + ymu_beh.view(1, 1, -1)).cpu().numpy().astype(np.float32)
        logger.info("[phase=fits] fold %d done in %.1fs", fold_i + 1, time.time() - tf0)

    fit_wall = time.time() - fit_t0
    logger.info(
        "[phase=fits] full battery wall %.1fs (%d map cells × 7 folds + %d read-outs)",
        fit_wall,
        n_map,
        n_pred_cells * n_beh,
    )

    # ── aggregate: skills + ρ ────────────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        skills = 1.0 - map_ss[:, :, 0] / map_ss[:, :, 1]  # (n_map, 4)
        skills[map_ss[:, :, 1] < 1e-12] = np.nan
        ceiling = 1.0 - ceil_ss[:, 0] / np.where(ceil_ss[:, 1] < 1e-12, np.nan, ceil_ss[:, 1])
    skills[ex_map] = np.nan
    ceiling[ex_ans] = np.nan  # identity ceiling consumes set-B rows — union-masked too

    y_ranks = np.stack([_rankdata_avg(E0[:, bi]) for bi in range(n_beh)], axis=0)  # (7, 50)
    ro_rho = np.stack(
        [
            np.stack([pooled_spearman(P[:, :, bi], y_ranks[bi]) for bi in range(n_beh)], axis=1)
            for P in (ro_predA, ro_predB)
        ],
        axis=2,
    )  # (n_pred_cells, 7, 2)
    ch_rho = np.stack(
        [
            np.stack([pooled_spearman(P[:, :, bi], y_ranks[bi]) for bi in range(n_beh)], axis=1)
            for P in (ch_predA, ch_predB)
        ],
        axis=2,
    )  # (n_map, 7, 2)
    or_rho = np.stack(
        [pooled_spearman(ch_oracle[:, :, bi], y_ranks[bi]) for bi in range(n_beh)], axis=1
    )  # (n_ans_cells, 7) PCA-basis oracle
    ro_rho[np.concatenate([ex_ctx, ex_ans])] = np.nan
    ch_rho[ex_map] = np.nan
    or_rho[ex_ans] = np.nan  # PCA-basis oracle rows for excluded families: out of the sweep

    names_c, names_a = red_A["ctx_cell_names"], red_A["ans_cell_names"]

    # ── K3 anchor gate: best matched-layer ah_nl × content-mean R1 skill ────
    anchor_idx = [
        i
        for i in range(n_map)
        if names_c[c_map[i]].startswith("ctx_ah_nl@")
        and names_a[a_map[i]].startswith("ans_content_mean@")
        and names_c[c_map[i]].split("@L")[1] == names_a[a_map[i]].split("@L")[1]
    ]
    anchor_r1 = float(np.nanmax(skills[anchor_idx, 0])) if anchor_idx else float("nan")
    logger.info("[anchor] ah_nl × content-mean best matched-layer R1 skill = %.4f", anchor_r1)

    obs_max = {
        f"R{r + 1}": {
            "max": float(np.nanmax(skills[:, r])),
            "argmax": int(np.nanargmax(skills[:, r])),
        }
        for r in range(4)
    }
    meta = reproducibility_metadata()
    dump_json(
        {
            "cells": {"c_cell": [names_c[i] for i in c_map], "a_cell": [names_a[i] for i in a_map]},
            "skill": {
                f"R{r + 1}": [None if np.isnan(v) else round(float(v), 6) for v in skills[:, r]]
                for r in range(4)
            },
            "observed_max": obs_max,
            "ceiling_ya_yb_per_a_cell": {
                names_a[i]: (None if np.isnan(ceiling[i]) else round(float(ceiling[i]), 6))
                for i in range(n_ans_cells)
            },
            "anchor_r1_best_matched_layer": anchor_r1,
            "excluded_families": excluded_union,
            "excluded_families_by_source": excluded_by_source,
            "pca_k": k,
            "g2": g2,
            "fit_wall_s": round(fit_wall, 1),
            "reproducibility": meta,
        },
        eval_out / "map_skill_by_cell.json",
    )
    pred_names = names_c + names_a
    dump_json(
        {
            "cells": pred_names,
            "behaviors": E0_BEHAVIORS,
            "rho": {
                "R_in_probe": [
                    [None if np.isnan(v) else round(float(v), 6) for v in row]
                    for row in ro_rho[:, :, 0]
                ],
                "R_input_ood": [
                    [None if np.isnan(v) else round(float(v), 6) for v in row]
                    for row in ro_rho[:, :, 1]
                ],
            },
            "reproducibility": meta,
        },
        eval_out / "readout_rho_by_cell.json",
    )
    dump_json(
        {
            "cells": {"c_cell": [names_c[i] for i in c_map], "a_cell": [names_a[i] for i in a_map]},
            "behaviors": E0_BEHAVIORS,
            "rho_R9": [
                [None if np.isnan(v) else round(float(v), 6) for v in row]
                for row in ch_rho[:, :, 0]
            ],
            "rho_R10": [
                [None if np.isnan(v) else round(float(v), 6) for v in row]
                for row in ch_rho[:, :, 1]
            ],
            "oracle_in_pca_basis_rho": {
                "cells": names_a,
                "rho": [
                    [None if np.isnan(v) else round(float(v), 6) for v in row] for row in or_rho
                ],
            },
            "reproducibility": meta,
        },
        eval_out / "chain_rho_by_cell.json",
    )

    torch.save(
        {
            "map_predA": torch.from_numpy(map_predA),
            "map_predB": torch.from_numpy(map_predB),
            "ypca_A": torch.from_numpy(ypca_A),
            "ypca_B": torch.from_numpy(ypca_B),
            "ro_predA": torch.from_numpy(ro_predA),
            "ro_predB": torch.from_numpy(ro_predB),
            "ch_predA": torch.from_numpy(ch_predA),
            "ch_predB": torch.from_numpy(ch_predB),
            "ch_oracle": torch.from_numpy(ch_oracle),
            "c_map": torch.from_numpy(c_map),
            "a_map": torch.from_numpy(a_map),
            "ctx_cell_names": names_c,
            "ans_cell_names": names_a,
            "behaviors": E0_BEHAVIORS,
            "ctx_ids": ctx_ids,
            "pca_k": k,
            "excluded_map_mask": torch.from_numpy(ex_map),
            "excluded_pred_mask": torch.from_numpy(np.concatenate([ex_ctx, ex_ans])),
            "reproducibility": meta,
        },
        preds_out / "pooled_heldout_predictions.pt",
    )
    logger.info("persisted pooled held-out predictions → %s", preds_out)

    write_sentinel(
        "epm:progress",
        {
            "phase": "S4_fit_battery",
            "blocks_pipeline": False,
            "n_map_cells": n_map,
            "anchor_r1": anchor_r1,
            "observed_max": obs_max,
            "fit_wall_s": round(fit_wall, 1),
        },
        eval_out,
        slug_extra="fits",
    )

    anchor_gate_armed = not args.skip_anchor_gate and not args.synthetic_smoke
    if anchor_gate_armed and not (0.6 <= anchor_r1 <= 0.9):
        raise RuntimeError(
            f"[k3-anchor-assert] anchor cell R1 skill {anchor_r1:.4f} outside [0.6, 0.9] "
            "(#810 LOFO anchor ~0.80) — stop and debug extraction/fits before the nulls"
        )
    # Post-K3 fit-done marker — the dispatcher's S4 resume predicate keys on THIS
    # file, never on the pre-gate eval JSONs / preds artifacts alone, so a retry
    # after a K3 FAIL re-runs the fits instead of skipping the failed gate
    # (round-1 blocker `k3-resume-bypasses-anchor-gate`).
    dump_json(
        {
            "phase": "S4_fit_battery",
            "anchor_gate": "PASS" if anchor_gate_armed else "SKIPPED",
            "anchor_r1": None if np.isnan(anchor_r1) else round(anchor_r1, 6),
            "excluded_families": excluded_union,
            "reproducibility": meta,
        },
        preds_out / "fits_done.json",
    )
    # NOT [phase=done] — that token is RESERVED for the dispatcher's single
    # terminal line (pod-side-reporting rule; #545 false-done class).
    logger.info("[phase=fits_complete] S4 fit battery complete (%.1fs total)", time.time() - t0)
    return 0


class _ChainCache:
    """Batched per-(a-cell) chain ridge: PRESS per behavior + explicit 34-dim weights."""

    def __init__(self, Xn: torch.Tensor, lambdas: list[float]) -> None:
        self.Xn = Xn  # (A, m, k) standardized train design
        self.lambdas = lambdas
        G = torch.bmm(Xn, Xn.transpose(1, 2))
        self.evals, self.Q = torch.linalg.eigh(G)
        self.Qsq = self.Q * self.Q
        m = Xn.shape[1]
        eye = torch.eye(m, dtype=Xn.dtype, device=Xn.device)
        self.Ainv = torch.stack([torch.linalg.inv(G + lam * eye) for lam in lambdas], dim=1)

    def per_column_weights(self, Ytr_c: torch.Tensor) -> torch.Tensor:
        """PRESS λ per (cell, behavior) + dual weights → explicit (A, k, P) primal w."""
        A, _m, P = Ytr_c.shape
        dev = Ytr_c.device
        lam = torch.tensor(self.lambdas, dtype=torch.float64, device=dev)
        nlam = lam.shape[0]
        QtY = torch.bmm(self.Q.transpose(1, 2), Ytr_c)
        filt = self.evals.unsqueeze(0) / (self.evals.unsqueeze(0) + lam.view(nlam, 1, 1))
        h_diag = torch.einsum("pkj,lpj->lpk", self.Qsq, filt)
        Yhat = torch.einsum("pkj,lpjq->lpkq", self.Q, filt.unsqueeze(-1) * QtY.unsqueeze(0))
        loo = (Ytr_c.unsqueeze(0) - Yhat) / (1.0 - h_diag).clamp(min=1e-8).unsqueeze(-1)
        mse = (loo * loo).mean(dim=2)  # (nlam, A, P)
        best = torch.argmin(mse, dim=0)  # (A, P)
        w = torch.zeros((A, self.Xn.shape[2], P), dtype=torch.float64, device=dev)
        for li in range(nlam):
            mask = (best == li).unsqueeze(1)  # (A, 1, P)
            if not bool(mask.any()):
                continue
            alpha = torch.bmm(self.Ainv[:, li], Ytr_c)  # (A, m, P)
            w_l = torch.bmm(self.Xn.transpose(1, 2), alpha)  # (A, k, P)
            w = torch.where(mask, w_l, w)
        return w


def make_synthetic_stores(
    store_root: Path,
    ctx_ids: list[str],
    lc: int,
    H: int,
    seed: int = 920,
    zero_coverage_b: str | None = None,
) -> None:
    """Tiny schema-conformant per-probe stores (both sets) for the fits/nulls smoke.

    Real battery context ids + fold structure; random per-probe values with a
    planted linear c→a signal so the anchor cell is non-trivially fittable; a few
    short answers so position validity/dedup codes are exercised.
    ``zero_coverage_b``: family whose validity is zeroed for ALL probes on
    context 0 in set B ONLY (set A untouched) — the union-exclusion smoke.
    """
    from issue920_common import ALL_STORE_FAMILIES, position_slots, store_dtype

    rng = np.random.default_rng(seed)
    n_probes = 6
    for s in ("A", "B"):
        out = store_root / f"summaries_set{s}"
        out.mkdir(parents=True, exist_ok=True)
        for ci, cid in enumerate(ctx_ids):
            latent = rng.normal(size=(H,)) + 0.1 * ci
            validity = np.ones((n_probes, len(ALL_STORE_FAMILIES)), dtype=np.uint8)
            blob: dict = {
                "context_id": cid,
                "families": ALL_STORE_FAMILIES,
                "capture_layers": list(range(lc)),
                "probes": [f"p{i}" for i in range(n_probes)],
                "ans_lens": [],
                "empty_completions": 0,
                "model": "synthetic",
                "probe_set": s,
            }
            ans_lens = []
            for pi in range(n_probes):
                a_len = int(rng.integers(3, 40))
                ans_lens.append(a_len)
                _rel, pos_valid = position_slots(a_len)
                fam0 = len(ALL_STORE_FAMILIES) - 20
                validity[pi, fam0:] = pos_valid
            blob["ans_lens"] = ans_lens
            if zero_coverage_b is not None and s == "B" and ci == 0:
                assert zero_coverage_b in ALL_STORE_FAMILIES, zero_coverage_b
                validity[:, ALL_STORE_FAMILIES.index(zero_coverage_b)] = 0
            for f in ALL_STORE_FAMILIES:
                base = latent if f.startswith(("ans_", "pos_")) else latent * 0.5
                vals = (base[None, None, :] + 0.3 * rng.normal(size=(n_probes, lc, H))).astype(
                    np.float32
                )
                blob[f"fam::{f}"] = torch.from_numpy(vals).to(store_dtype(f))
            validity_t = torch.from_numpy(validity)
            blob["validity"] = validity_t
            torch.save(blob, out / f"{cid}.pt")
    logger.info("synthetic stores written to %s (lc=%d H=%d)", store_root, lc, H)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] fit battery crashed:\n%s", traceback.format_exc())
        raise
