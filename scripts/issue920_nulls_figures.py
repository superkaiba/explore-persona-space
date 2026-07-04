#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #920 S5: selection-symmetric null batteries + aggregation + figures.

Two phases (plan §3.5-S5):

``--gpu-null-only`` (on the GPU provision, after S4):
  DV-1 perm-refit null — the ``issue810_batched_null`` LOCO→LOFO adaptation.
  ``_LocoRidgeXCache`` hardcodes LOO folds (fixed m=n−1 ``eye_m``, single-row
  held-out reads, the (total−Y)/(n−1) LOO train-mean baseline — fact-checked),
  so the LOFO adaptation REWRITES the constructor + prediction loop + per-fold
  train-mean baseline on the shared ``issue920_fit_core`` machinery (train
  sizes 36..48, multi-row held-out blocks) while KEEPING the PRESS/dual
  identities and the ``make_perm_matrix`` draw-order contract (the batched
  battery consumes the SAME permutations as the serial reference for a
  like-seeded rng — the G2-null gate asserts it, atol=1e-8 float64, before the
  full battery). 1,000 draws (seed 920), ONE shared context-row permutation per
  draw applied to the TARGET rows (PCA bases + X-caches draw-invariant), all
  draws × all 34,652 cells × ALL FOUR regimes in one batched pass (draw axis
  stacked into the GEMMs; per-draw weights NEVER persisted — the declared
  discard). Per-draw × per-cell skill matrices persisted per regime.

``--cpu-aggregation`` (cpu-mid, after the GPU release):
  DV-2/DV-3 stored-prediction perm-ρ nulls — rank-transform the pooled held-out
  predictions ONCE per cell, permute the E0 ranks per draw, all cells × draws as
  single ``(cells, 50) @ (50, 1000)`` GEMMs, per-draw max PER BEHAVIOR (behavior
  is a reporting axis, never selected over). Bands (97.5th pct of the per-draw
  max), the H1/H2 paired difference bands (Δ_d = challenger max − mean-family
  max per draw, plan §6), and the figures (hero family×family R1 heatmap + OOD
  delta companion + exploratory set) via the paper-plots conventions.
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
from issue810_batched_null import make_perm_matrix  # noqa: E402  (draw-order contract)
from issue920_common import (  # noqa: E402
    E0_BEHAVIORS,
    HF_DATA_REPO,
    I920_TENSORS_PREFIX,
    dump_json,
    load_battery,
    load_e0_graded,
    load_json,
    lofo_folds,
    reproducibility_metadata,
    write_sentinel,
)
from issue920_fit_core import (  # noqa: E402
    PCA_K,
    FoldXCache,
    batched_pca_project,
    batched_press_predict,
    enumerate_map_cells,
    fit_device,
    load_reduced_matrices,
    pca_apply,
    serial_reference_map_fit,
)

logger = logging.getLogger("issue920_nulls")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NULL_SEED = 920
N_DRAWS = 1000
NP_CAP = int(os.environ.get("EPM_I920_NULL_NP_CAP", "8192"))
PAIR_CHUNK = int(os.environ.get("EPM_I920_NULL_PAIR_CHUNK", "512"))


# ── DV-1 perm-refit null battery (batched, LOFO) ─────────────────────────────


def dv1_null_battery(
    red_A: dict,
    red_B: dict,
    ctx_ids: list[str],
    fam_map: dict[str, str],
    perms: np.ndarray,
    c_map: np.ndarray,
    a_map: np.ndarray,
    k: int,
    device: torch.device,
) -> np.ndarray:
    """Per-draw × per-cell null skills at ALL FOUR regimes → (n_cells, D, 4) fp32.

    The batched LOFO perm-refit: per fold, the X-cache + train-fold PCA basis are
    draw-INVARIANT (computed once from the unpermuted rows); each draw's shared
    context permutation π reorders the TARGET rows (train fit on Y[π(tr)], test
    scored on Y[π(te)], baseline = permuted-train mean) — the parent's
    sanctioned exchangeability variant. All Y-dependent steps run through
    ``batched_press_predict`` with the (pair × draw) axes flattened into the
    batch dim; per-draw weights are never persisted (declared discard).
    """
    D = perms.shape[0]
    n_cells = len(c_map)
    folds = lofo_folds(ctx_ids, fam_map)
    acc = np.zeros((n_cells, D, 4, 2), dtype=np.float64)
    XA, XB = red_A["X_ctx"], red_B["X_ctx"]
    YA, YB = red_A["Y_ans"].double(), red_B["Y_ans"].double()
    perm_t = torch.from_numpy(perms).long().to(device)
    for fold_i, (fam, tr, te) in enumerate(folds):
        tf0 = time.time()
        cache = FoldXCache(XA, tr, te, XB, device)
        mu_p, comps = batched_pca_project(YA[:, tr].to(device), k)
        YfullA = pca_apply(YA.to(device), mu_p, comps)  # (A, n, k) — fold basis, all rows
        YfullB = pca_apply(YB.to(device), mu_p, comps)
        tr_t = torch.tensor(tr, device=device)
        te_t = torch.tensor(te, device=device)
        for lo in range(0, n_cells, PAIR_CHUNK):
            hi = min(lo + PAIR_CHUNK, n_cells)
            pc = hi - lo
            dc = max(1, NP_CAP // pc)
            c_sel = torch.from_numpy(c_map[lo:hi]).to(device)
            a_sel = torch.from_numpy(a_map[lo:hi]).to(device)
            YpcA = YfullA[a_sel]  # (pc, n, k)
            YpcB = YfullB[a_sel]
            for d0 in range(0, D, dc):
                d1 = min(d0 + dc, D)
                dd = perm_t[d0:d1]  # (dcur, n)
                dcur = d1 - d0
                src_tr = dd[:, tr_t]  # (dcur, m)
                src_te = dd[:, te_t]  # (dcur, n_te)
                Ytr = YpcA[:, src_tr]  # (pc, dcur, m, k)
                ymu = Ytr.mean(dim=2, keepdim=True)  # (pc, dcur, 1, k)
                Ytr_c = (Ytr - ymu).reshape(pc * dcur, len(tr), k)
                c_flat = c_sel.repeat_interleave(dcur)
                predA_c, predB_c, _ = batched_press_predict(cache, c_flat, Ytr_c)
                n_te = len(te)
                predA = predA_c.reshape(pc, dcur, n_te, k) + ymu
                predB = predB_c.reshape(pc, dcur, n_te, k) + ymu
                tA = YpcA[:, src_te]  # (pc, dcur, n_te, k)
                tB = YpcB[:, src_te]
                base = ((tA - ymu) ** 2).sum(dim=(2, 3))
                baseB = ((tB - ymu) ** 2).sum(dim=(2, 3))
                acc[lo:hi, d0:d1, 0, 0] += ((tA - predA) ** 2).sum(dim=(2, 3)).cpu().numpy()
                acc[lo:hi, d0:d1, 0, 1] += base.cpu().numpy()
                acc[lo:hi, d0:d1, 1, 0] += ((tA - predB) ** 2).sum(dim=(2, 3)).cpu().numpy()
                acc[lo:hi, d0:d1, 1, 1] += base.cpu().numpy()
                acc[lo:hi, d0:d1, 2, 0] += ((tB - predA) ** 2).sum(dim=(2, 3)).cpu().numpy()
                acc[lo:hi, d0:d1, 2, 1] += baseB.cpu().numpy()
                acc[lo:hi, d0:d1, 3, 0] += ((tB - predB) ** 2).sum(dim=(2, 3)).cpu().numpy()
                acc[lo:hi, d0:d1, 3, 1] += baseB.cpu().numpy()
        logger.info(
            "[phase=dv1_null] fold %d/7 (%s) done in %.1fs", fold_i + 1, fam, time.time() - tf0
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        skills = 1.0 - acc[:, :, :, 0] / acc[:, :, :, 1]
        skills[acc[:, :, :, 1] < 1e-12] = np.nan
    return skills  # float64 (G2 atol=1e-8; persisted fp16 by the caller)


def g2_null_gate(
    red_A: dict,
    red_B: dict,
    ctx_ids: list[str],
    fam_map: dict[str, str],
    k: int,
    device: torch.device,
    n_draws: int = 50,
) -> dict:
    """§7 G2 (null side): the batched battery reproduces the serial per-draw skills.

    Consumes the SAME permutations as the serial reference through a like-seeded
    ``make_perm_matrix`` rng (the #810 draw-order contract); asserts every
    per-draw per-regime skill at atol=1e-8 on 2 gate cells. Dispatches the LIVE
    ``dv1_null_battery`` (never a sibling).
    """
    names_c, names_a = red_A["ctx_cell_names"], red_A["ans_cell_names"]
    lc = red_A["n_layers"]
    mid = min(18, lc - 1)
    gate = [
        (names_c.index(f"ctx_ah_nl@L{mid}"), names_a.index(f"ans_content_mean@L{mid}")),
        (names_c.index("ctx_wt_pool_meanmean"), names_a.index("ans_content_pool_meanmean")),
    ]
    c_map = np.array([g[0] for g in gate], dtype=np.int64)
    a_map = np.array([g[1] for g in gate], dtype=np.int64)
    perms = make_perm_matrix(len(ctx_ids), n_draws, np.random.default_rng(NULL_SEED))
    batched = dv1_null_battery(red_A, red_B, ctx_ids, fam_map, perms, c_map, a_map, k, device)
    groups = [fam_map[c] for c in ctx_ids]
    max_abs = 0.0
    for gi, (c_i, a_i) in enumerate(gate):
        XA = red_A["X_ctx"][c_i].numpy().astype(np.float64)
        YA = red_A["Y_ans"][a_i].numpy().astype(np.float64)
        XBn = red_B["X_ctx"][c_i].numpy().astype(np.float64)
        YBn = red_B["Y_ans"][a_i].numpy().astype(np.float64)
        for d in range(n_draws):
            serial = serial_reference_map_fit(
                XA, YA, XBn, YBn, groups, k, perm_row=perms[d], device=device
            )
            for ri, r in enumerate(("R1", "R2", "R3", "R4")):
                diff = abs(float(batched[gi, d, ri]) - serial[r])
                max_abs = max(max_abs, diff)
                assert diff <= 1e-8, (
                    f"[g2-null-assert] cell {gi} draw {d} {r}: batched "
                    f"{batched[gi, d, ri]:.10f} vs serial {serial[r]:.10f}"
                )
    logger.info(
        "[g2-null] batched-vs-serial per-draw equivalence PASS (max |Δ|=%.2e, %d draws × %d cells)",
        max_abs,
        n_draws,
        len(gate),
    )
    return {"max_abs_skill_diff": max_abs, "n_draws": n_draws, "atol": 1e-8}


# ── DV-2/3 stored-prediction ρ nulls (CPU GEMMs) ─────────────────────────────


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Tie-free argsort ranks along the last axis (fp64 ridge preds — no ties)."""
    order = np.argsort(x, axis=-1)
    ranks = np.empty_like(x, dtype=np.float64)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(x.shape[-1], dtype=np.float64), x.shape).copy(),
        axis=-1,
    )
    return ranks


def stored_pred_rho_null(preds: np.ndarray, e0_col: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """(cells, D) perm-ρ: rank preds once, permute average-tie E0 ranks per draw, one GEMM.

    Degenerate (constant) prediction rows → 0.0 (the null convention of
    ``issue810_batched_null.batched_projection_null_rho``).
    """
    from scipy.stats import rankdata

    ry = rankdata(e0_col)  # average-tie ranks, permuted per draw
    ry_perm = ry[perms]  # (D, n)
    rp = _rank_rows(preds)  # (cells, n)
    rp_c = rp - rp.mean(axis=-1, keepdims=True)
    ry_c = ry_perm - ry_perm.mean(axis=-1, keepdims=True)
    num = rp_c @ ry_c.T  # (cells, D)
    den = np.sqrt((rp_c * rp_c).sum(-1, keepdims=True) * (ry_c * ry_c).sum(-1)[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = num / den
    rho[preds.std(axis=-1) < 1e-9] = 0.0
    return np.nan_to_num(rho, nan=0.0)


# ── aggregation + figures ─────────────────────────────────────────────────────

MEAN_FAMILY_A_CELLS = ("ans_content_mean", "ans_content_pool_meanmean")  # the incumbent


def _family_of(name: str) -> str:
    return name.split("@")[0]


def aggregate_and_figures(  # noqa: C901 — one linear figure pipeline
    eval_out: Path, preds_path: Path, null_dir: Path, fig_dir: Path
) -> dict:
    """Bands, H1/H2 paired-difference reads, and the figure set (paper-plots style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(preds_path, weights_only=False)
    names_c, names_a = blob["ctx_cell_names"], blob["ans_cell_names"]
    c_map, a_map = blob["c_map"].numpy(), blob["a_map"].numpy()
    ex_map = blob["excluded_map_mask"].numpy()
    map_json = load_json(eval_out / "map_skill_by_cell.json")
    skills = np.array(
        [[np.nan if v is None else v for v in map_json["skill"][f"R{r + 1}"]] for r in range(4)],
        dtype=np.float64,
    ).T  # (n_map, 4)

    summary: dict = {"bands": {}, "observed": map_json["observed_max"]}
    dv1 = torch.load(null_dir / "dv1_null_skills.pt", weights_only=False)
    null_skills = dv1["skills"].numpy()  # (n_map, D, 4) fp16→
    valid = ~ex_map
    for r in range(4):
        per_draw_max = np.nanmax(null_skills[valid, :, r].astype(np.float64), axis=0)
        band = float(np.quantile(per_draw_max, 0.975))
        obs = float(np.nanmax(skills[:, r]))
        summary["bands"][f"dv1_R{r + 1}"] = {
            "band_p97_5": band,
            "observed_max": obs,
            "clears": bool(obs > band),
            "null_draws": int(per_draw_max.shape[0]),
        }
    # H1 paired difference: challenger max − mean-family max per draw (same reduction)
    a_fams = np.array([_family_of(names_a[i]) for i in a_map])
    mean_set = np.isin(a_fams, MEAN_FAMILY_A_CELLS) & valid
    chal_set = (~np.isin(a_fams, MEAN_FAMILY_A_CELLS)) & valid
    for r, tag in ((0, "R1"), (1, "R2")):
        d_null = np.nanmax(null_skills[chal_set, :, r].astype(np.float64), axis=0) - np.nanmax(
            null_skills[mean_set, :, r].astype(np.float64), axis=0
        )
        d_obs = float(np.nanmax(skills[chal_set, r]) - np.nanmax(skills[mean_set, r]))
        summary[f"h1_delta_{tag}"] = {
            "observed_delta": d_obs,
            "band_p97_5": float(np.quantile(d_null, 0.975)),
            "beats_incumbent_past_band": bool(d_obs > float(np.quantile(d_null, 0.975))),
            "mean_family_cells": list(MEAN_FAMILY_A_CELLS),
            "incumbent_set": (
                "ANSWER-side instantiation: incumbent = valid map cells whose "
                "a-cell family is in mean_family_cells (any context cell × any "
                "layer); every other valid cell is a challenger. Context-side "
                "mean-pool families are challengers under this split; the "
                "persisted per-draw matrices make any other split recomputable."
            ),
            "n_incumbent_cells": int(mean_set.sum()),
            "n_challenger_cells": int(chal_set.sum()),
            "note": "set-size asymmetry biases TOWARD retaining the incumbent (plan §6)",
        }

    # DV-2/3 bands per behavior × regime
    for dv, fname in (("dv2", "dv2_null_rho.pt"), ("dv3", "dv3_null_rho.pt")):
        nb = torch.load(null_dir / fname, weights_only=False)
        summary["bands"][dv] = nb["bands"]

    # ── band convention + the MATCHING observed statistic ────────────────────
    # The DV-2/3 nulls take the per-draw max of |ρ| (two-sided — sign is not a
    # selection axis), while the eval JSONs store SIGNED ρ; record the |·|
    # convention + the observed max |ρ| over the SAME cell set so the analyzer
    # never compares a signed max against a |·| band.
    summary["dv23_band_convention"] = (
        "null bands = 97.5th pct of per-draw max |rho| (two-sided); observed "
        "comparator = max |rho| over the same (regime[, side], behavior) cell set"
    )
    ro_json = load_json(eval_out / "readout_rho_by_cell.json")
    chain_json = load_json(eval_out / "chain_rho_by_cell.json")
    behaviors = list(ro_json["behaviors"])
    n_ctx_cells = len(names_c)

    def _obs_vs_band(col: np.ndarray, band: float | None) -> dict:
        obs = None if np.all(np.isnan(col)) else float(np.nanmax(np.abs(col)))
        return {
            "observed_max_abs_rho": obs,
            "band_p97_5": band,
            "clears": bool(obs is not None and band is not None and obs > band),
        }

    ro_arr = {
        key: np.array([[np.nan if v is None else v for v in row] for row in ro_json["rho"][key]])
        for key in ("R_in_probe", "R_input_ood")
    }
    summary["dv2_observed_vs_band"] = {
        f"{regime}_{side}_{b}": _obs_vs_band(
            ro_arr[key][sel, bi], summary["bands"]["dv2"].get(f"{regime}_{side}_{b}")
        )
        for regime, key in (("in_probe", "R_in_probe"), ("input_ood", "R_input_ood"))
        for side, sel in (("ctx", slice(0, n_ctx_cells)), ("ans", slice(n_ctx_cells, None)))
        for bi, b in enumerate(behaviors)
    }
    ch_arr = {
        tag: np.array([[np.nan if v is None else v for v in row] for row in chain_json[key]])
        for tag, key in (("R9", "rho_R9"), ("R10", "rho_R10"))
    }
    summary["dv3_observed_vs_band"] = {
        f"{tag}_{b}": _obs_vs_band(ch_arr[tag][:, bi], summary["bands"]["dv3"].get(f"{tag}_{b}"))
        for tag in ("R9", "R10")
        for bi, b in enumerate(behaviors)
    }
    summary["excluded_families"] = map_json.get("excluded_families", [])
    summary["excluded_families_by_source"] = map_json.get("excluded_families_by_source")

    # ── figures ──────────────────────────────────────────────────────────────
    ctx_fams = sorted(
        {_family_of(n) for n in names_c},
        key=lambda f: names_c.index(next(n for n in names_c if _family_of(n) == f)),
    )
    ans_fams = sorted(
        {_family_of(n) for n in names_a},
        key=lambda f: names_a.index(next(n for n in names_a if _family_of(n) == f)),
    )
    c_fam_arr = np.array([_family_of(names_c[i]) for i in c_map])

    def fam_grid(regime: int) -> np.ndarray:
        g = np.full((len(ctx_fams), len(ans_fams)), np.nan)
        for fi, cf in enumerate(ctx_fams):
            m1 = c_fam_arr == cf
            for gi, af in enumerate(ans_fams):
                sel = m1 & (a_fams == af)
                if sel.any() and not np.all(np.isnan(skills[sel, regime])):
                    g[fi, gi] = np.nanmax(skills[sel, regime])
        return g

    band_r1 = summary["bands"]["dv1_R1"]["band_p97_5"]
    for tag, grid, title in (
        (
            "hero_family_heatmap_R1",
            fam_grid(0),
            f"R1 map skill (best matched-layer per family pair; null band {band_r1:.3f})",
        ),
        (
            "hero_family_heatmap_R2_minus_R1",
            fam_grid(1) - fam_grid(0),
            "Input-OOD delta (R2 − R1) per family pair",
        ),
    ):
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ans_fams)), ans_fams, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ctx_fams)), ctx_fams, fontsize=6)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax)
        fig.savefig(fig_dir / f"{tag}.png", dpi=200)
        plt.close(fig)

    # per-layer curves for the top-5 family pairs (matched-layer block only)
    ll_mask = np.array(
        ["@L" in names_c[c_map[i]] and "@L" in names_a[a_map[i]] for i in range(len(c_map))]
    )
    order = np.argsort(np.nan_to_num(skills[:, 0], nan=-9))[::-1]
    top_pairs, seen = [], set()
    for i in order:
        if not ll_mask[i]:
            continue
        key = (c_fam_arr[i], a_fams[i])
        if key not in seen:
            seen.add(key)
            top_pairs.append(key)
        if len(top_pairs) == 5:
            break
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = paper_palette(max(3, len(top_pairs)))
    for pi, (cf, af) in enumerate(top_pairs):
        sel = np.where((c_fam_arr == cf) & (a_fams == af) & ll_mask)[0]
        layers = [int(names_c[c_map[i]].split("@L")[1]) for i in sel]
        o = np.argsort(layers)
        ax.plot(np.array(layers)[o], skills[sel, 0][o], label=f"{cf} × {af}", color=colors[pi])
    ax.axhline(band_r1, ls="--", c="gray", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("R1 skill")
    ax.legend(fontsize=6)
    ax.set_title("Per-layer R1 skill — top-5 family pairs (dashed: max-inherited null band)")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_layer_top5_pairs.png", dpi=200)
    plt.close(fig)

    # winning-cell per-context scatter (pred vs true on PCA dim 0, points labeled)
    win = int(np.nanargmax(skills[:, 0]))
    predA = blob["map_predA"][win].float().numpy()  # (n, k)
    ytrue = blob["ypca_A"][a_map[win]].float().numpy()
    ctx_ids = blob["ctx_ids"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ytrue[:, 0], predA[:, 0], s=14)
    for i, cid in enumerate(ctx_ids):
        ax.annotate(cid, (ytrue[i, 0], predA[i, 0]), fontsize=4, alpha=0.7)
    ax.set_xlabel("true (fold-basis PCA dim 0)")
    ax.set_ylabel("held-out prediction")
    ax.set_title(f"Winning R1 cell: {names_c[c_map[win]]} × {names_a[a_map[win]]}")
    fig.tight_layout()
    fig.savefig(fig_dir / "winning_cell_scatter.png", dpi=200)
    plt.close(fig)

    # chain-vs-oracle gap per behavior (best chain cell vs its PCA-basis oracle)
    ch = ch_arr["R9"]  # (n_map, 7) — loaded with the observed-vs-band block above
    orc = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in chain_json["oracle_in_pca_basis_rho"]["rho"]
        ]
    )  # (n_ans, 7)
    gaps, labels = [], []
    for bi, b in enumerate(E0_BEHAVIORS):
        if np.all(np.isnan(ch[:, bi])):
            continue
        best = int(np.nanargmax(ch[:, bi]))
        gaps.append(float(orc[a_map[best], bi] - ch[best, bi]))
        labels.append(b)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, gaps, color=paper_palette(max(3, len(labels)))[: len(labels)])
    ax.set_ylabel("oracle ρ − chain ρ (best chain cell)")
    ax.set_title("Map-induced read-out loss per behavior (chain-vs-oracle gap, R9)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "chain_vs_oracle_gap.png", dpi=200)
    plt.close(fig)

    # R3 identity-ceiling diagnostic per answer family (K2 check surface)
    ceil = map_json["ceiling_ya_yb_per_a_cell"]
    fam_best: dict[str, float] = {}
    for nm, v in ceil.items():
        if v is None:
            continue
        f = _family_of(nm)
        fam_best[f] = max(fam_best.get(f, -np.inf), v)
    fig, ax = plt.subplots(figsize=(11, 4))
    ks = list(fam_best)
    ax.bar(range(len(ks)), [fam_best[f_] for f_ in ks])
    ax.set_xticks(range(len(ks)), ks, rotation=90, fontsize=6)
    ax.axhline(0.2, ls="--", c="red", lw=1)
    ax.set_ylabel("Y_A→Y_B identity skill (best layer)")
    ax.set_title("Target-OOD ceiling diagnostic (K2 line at 0.2)")
    fig.tight_layout()
    fig.savefig(fig_dir / "r3_identity_ceiling.png", dpi=200)
    plt.close(fig)

    # raw-alongside-processed: full per-cell R1 skill distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(skills[~np.isnan(skills[:, 0]), 0], bins=80)
    ax.axvline(band_r1, ls="--", c="gray")
    ax.set_xlabel("R1 skill")
    ax.set_ylabel("cells")
    ax.set_title("All 34,652 map cells — R1 skill distribution (raw, unselected)")
    fig.tight_layout()
    fig.savefig(fig_dir / "r1_skill_distribution.png", dpi=200)
    plt.close(fig)

    summary["k2_identity_ceiling_mean_families"] = {
        f: fam_best.get(f) for f in MEAN_FAMILY_A_CELLS if f in fam_best
    }
    summary["figures"] = sorted(p.name for p in fig_dir.glob("*.png"))
    return summary


# ── phase drivers ─────────────────────────────────────────────────────────────


def run_gpu_null(args) -> None:
    device = torch.device(fit_device())
    store_root = Path(args.store_root)
    null_dir = Path(args.null_out)
    null_dir.mkdir(parents=True, exist_ok=True)
    out_pt = null_dir / "dv1_null_skills.pt"
    if out_pt.is_file():
        # Battery resume (an upload crash on a prior attempt must not force a
        # full recompute); the upload + done-marker below still re-run.
        logger.info("[phase=dv1_null] %s present — battery skipped (resume)", out_pt.name)
        blob = torch.load(out_pt, weights_only=False)
        n_draws, wall = int(blob.get("n_draws", args.n_draws)), float(blob.get("wall_s", 0.0))
    else:
        instances, fam_map = load_battery()
        ctx_ids = [i["id"] for i in instances]
        red_A = load_reduced_matrices(store_root / "summaries_setA", ctx_ids)
        red_B = load_reduced_matrices(store_root / "summaries_setB", ctx_ids)
        k = min(PCA_K, max(1, min(len(tr) for _f, tr, _te in lofo_folds(ctx_ids, fam_map)) - 2))
        if not args.skip_g2:
            logger.info("[phase=g2_null_gate] %d draws on 2 cells", args.g2_draws)
            g2 = g2_null_gate(red_A, red_B, ctx_ids, fam_map, k, device, n_draws=args.g2_draws)
        else:
            g2 = {"skipped": True}
        c_map, a_map = enumerate_map_cells(red_A["n_layers"])
        perms = make_perm_matrix(len(ctx_ids), args.n_draws, np.random.default_rng(NULL_SEED))
        t0 = time.time()
        skills = dv1_null_battery(red_A, red_B, ctx_ids, fam_map, perms, c_map, a_map, k, device)
        wall = time.time() - t0
        n_draws = args.n_draws
        logger.info(
            "[phase=dv1_null] battery wall %.1fs (%d cells × %d draws × 4 regimes)",
            wall,
            len(c_map),
            args.n_draws,
        )
        torch.save(
            {
                "skills": torch.from_numpy(skills).to(torch.float16),
                "perm_seed": NULL_SEED,
                "n_draws": args.n_draws,
                "regimes": ["R1", "R2", "R3", "R4"],
                "g2": g2,
                "wall_s": wall,
                "reproducibility": reproducibility_metadata(),
            },
            out_pt,
        )
    if not args.no_upload:
        _upload_tensors(
            [
                (null_dir, "null_matrices"),
                (Path(args.preds_dir), "pooled_predictions"),
                (Path(args.eval_out), "eval_json"),  # §6.5 eval JSONs: durable pre-delete
            ]
        )
    write_sentinel(
        "epm:progress",
        {
            "phase": "S5_dv1_null",
            "blocks_pipeline": False,
            "n_draws": n_draws,
            "wall_s": round(wall, 1),
        },
        null_dir,
        slug_extra="dv1-null",
    )
    # Post-upload phase-done marker: the dispatcher's resume predicate keys on
    # this (never on the pre-upload .pt alone), so an upload crash re-enters the
    # phase on retry (same class as the K3 fit-done marker).
    dump_json(
        {"phase": "S5_dv1_null", "n_draws": n_draws, "reproducibility": reproducibility_metadata()},
        null_dir / "dv1_done.json",
    )


def _upload_tensors(pairs: list[tuple[Path, str]]) -> None:
    """ONE upload_folder commit per (dir, subprefix) pair + fresh-listing verify.

    Standard pairs: null matrices, pooled predictions, and the §6.5 eval JSONs
    (``eval_json``) — the GCP auto-delete lane destroys the boot disk, so the
    primary-deliverable JSONs must land on the Hub before the instance exits.
    """
    from huggingface_hub import HfApi, list_repo_files

    api = HfApi()
    for local, sub in pairs:
        if not local.is_dir():
            continue
        api.upload_folder(
            folder_path=str(local),
            path_in_repo=f"{I920_TENSORS_PREFIX}/{sub}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            allow_patterns=["*.pt", "*.json"],
            commit_message=f"issue #920: {sub}",
        )
        remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main"))
        expected = {
            f"{I920_TENSORS_PREFIX}/{sub}/{p.name}"
            for p in local.iterdir()
            if p.suffix in (".pt", ".json")
        }
        missing = expected - remote
        if missing:
            raise RuntimeError(f"analysis-tensor upload verification FAILED: {sorted(missing)[:3]}")
        logger.info("uploaded + verified %s (%d files)", sub, len(expected))


def run_cpu_aggregation(args) -> None:
    eval_out = Path(args.eval_out)
    null_dir = Path(args.null_out)
    preds_dir = Path(args.preds_dir)
    fig_dir = Path(args.fig_out)
    preds_path = preds_dir / "pooled_heldout_predictions.pt"
    # HF fallback for EVERY cross-instance input this phase consumes — P6 runs on
    # a FRESH cpu-mid instance, so the §6.5 eval JSONs (uploaded by the GPU-null
    # phase under eval_json/) must be fetchable exactly like the two tensors.
    for p, sub in (
        (preds_path, "pooled_predictions/pooled_heldout_predictions.pt"),
        (null_dir / "dv1_null_skills.pt", "null_matrices/dv1_null_skills.pt"),
        (eval_out / "map_skill_by_cell.json", "eval_json/map_skill_by_cell.json"),
        (eval_out / "readout_rho_by_cell.json", "eval_json/readout_rho_by_cell.json"),
        (eval_out / "chain_rho_by_cell.json", "eval_json/chain_rho_by_cell.json"),
    ):
        if not p.is_file():
            import shutil

            from huggingface_hub import hf_hub_download

            logger.info("fetching %s from HF (%s)", p.name, sub)
            got = hf_hub_download(HF_DATA_REPO, f"{I920_TENSORS_PREFIX}/{sub}", repo_type="dataset")
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(got, p)

    blob = torch.load(preds_path, weights_only=False)
    _instances, _fam = load_battery()
    ctx_ids_order = blob["ctx_ids"]
    e0 = load_e0_graded()
    E0 = np.stack([[e0[b][c] for c in ctx_ids_order] for b in E0_BEHAVIORS], axis=1)
    perms = make_perm_matrix(len(ctx_ids_order), args.n_draws, np.random.default_rng(NULL_SEED))
    ex_pred = blob["excluded_pred_mask"].numpy()
    ex_map = blob["excluded_map_mask"].numpy()
    n_ctx_cells = len(blob["ctx_cell_names"])

    # DV-2 read-out nulls: per behavior × regime; per-draw max over ctx cells (R5/R7)
    # and over answer cells (R6/R8) SEPARATELY (they are distinct scoring-matrix rows).
    dv2 = {"bands": {}, "per_draw_max": {}}
    dv2_mats = {}
    for regime, key in (("in_probe", "ro_predA"), ("input_ood", "ro_predB")):
        P = blob[key].float().numpy()  # (n_pred, n, 7)
        for bi, b in enumerate(E0_BEHAVIORS):
            rho = stored_pred_rho_null(P[:, :, bi], E0[:, bi], perms)  # (n_pred, D)
            rho[ex_pred] = 0.0
            dv2_mats[f"{regime}_{b}"] = torch.from_numpy(rho.astype(np.float16))
            for side, sel in (("ctx", slice(0, n_ctx_cells)), ("ans", slice(n_ctx_cells, None))):
                pdm = np.abs(rho[sel]).max(axis=0)
                dv2["bands"][f"{regime}_{side}_{b}"] = float(np.quantile(pdm, 0.975))
                dv2["per_draw_max"][f"{regime}_{side}_{b}"] = pdm.tolist()
    torch.save(
        {
            "mats": dv2_mats,
            "bands": dv2["bands"],
            "perm_seed": NULL_SEED,
            "reproducibility": reproducibility_metadata(),
        },
        null_dir / "dv2_null_rho.pt",
    )

    # DV-3 chain nulls: per-draw max over the 34,652 chain cells PER BEHAVIOR
    dv3 = {"bands": {}, "per_draw_max": {}}
    dv3_mats = {}
    for regime, key in (("R9", "ch_predA"), ("R10", "ch_predB")):
        P = blob[key].float().numpy()  # (n_map, n, 7)
        for bi, b in enumerate(E0_BEHAVIORS):
            rho = stored_pred_rho_null(P[:, :, bi], E0[:, bi], perms)
            rho[ex_map] = 0.0
            dv3_mats[f"{regime}_{b}"] = torch.from_numpy(rho.astype(np.float16))
            pdm = np.abs(rho).max(axis=0)
            dv3["bands"][f"{regime}_{b}"] = float(np.quantile(pdm, 0.975))
            dv3["per_draw_max"][f"{regime}_{b}"] = pdm.tolist()
    torch.save(
        {
            "mats": dv3_mats,
            "bands": dv3["bands"],
            "perm_seed": NULL_SEED,
            "reproducibility": reproducibility_metadata(),
        },
        null_dir / "dv3_null_rho.pt",
    )

    summary = aggregate_and_figures(eval_out, preds_path, null_dir, fig_dir)
    summary["reliability_ceilings_note"] = (
        "#812 reliability ceilings are ANNOTATION-ONLY (no attenuation correction "
        "registered) — eval_results/issue_812/reliability_and_learning_curve.json"
    )
    summary["reproducibility"] = reproducibility_metadata()
    dump_json(summary, eval_out / "null_bands_and_headline.json")
    if not args.no_upload:
        _upload_tensors(
            [
                (null_dir, "null_matrices"),
                (Path(args.preds_dir), "pooled_predictions"),
                (eval_out, "eval_json"),  # incl. null_bands_and_headline.json
            ]
        )
    write_sentinel(
        "epm:progress",
        {"phase": "S5_cpu_aggregation", "blocks_pipeline": False, "bands": summary["bands"]},
        eval_out,
        slug_extra="agg",
    )
    logger.info("[phase=aggregation_complete] aggregation + figures complete")


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #920 S5: null batteries + figures")
    ap.add_argument("--gpu-null-only", action="store_true")
    ap.add_argument("--cpu-aggregation", action="store_true")
    ap.add_argument("--store-root", default=str(PROJECT_ROOT / "data" / "issue_920"))
    ap.add_argument("--preds-dir", default=str(PROJECT_ROOT / "data" / "issue_920" / "preds"))
    ap.add_argument(
        "--null-out", default=str(PROJECT_ROOT / "data" / "issue_920" / "null_matrices")
    )
    ap.add_argument("--eval-out", default=str(PROJECT_ROOT / "eval_results" / "issue_920"))
    ap.add_argument("--fig-out", default=str(PROJECT_ROOT / "figures" / "issue_920"))
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--g2-draws", type=int, default=50)
    ap.add_argument("--skip-g2", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    if not args.gpu_null_only and not args.cpu_aggregation:
        raise SystemExit("pass --gpu-null-only and/or --cpu-aggregation")
    if args.gpu_null_only:
        run_gpu_null(args)
    if args.cpu_aggregation:
        run_cpu_aggregation(args)
        # The standalone P6 dispatch (cpu-mid lane) runs this script bare as the
        # workload command, so THIS invocation is that workload's dispatcher-
        # terminal — the single reserved [phase=done] (pod-side-reporting rule).
        logger.info("[phase=done] S5 cpu aggregation complete")
    else:
        # Inside issue920_dispatch.sh (--gpu-null-only): the terminal token
        # belongs to the dispatcher, never a phase script (#545 false-done class).
        logger.info("[phase=nulls_gpu_complete] S5 GPU null battery complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] nulls/figures crashed:\n%s", traceback.format_exc())
        raise
