#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings + log messages.
"""Issue #810 free-analysis follow-up — refusal anti-correlation + max-pool length diagnostics.

Four 0-GPU diagnostics over EXISTING artifacts (no new training / eval / sampling):

(a) **Length-partialled read-out.** Recompute the refusal FIXED-direction read-out
    (``E0 ≈ r_Bᵀ summary``, #658 A3.3) per (summary × layer) and partial the
    per-context answer length out of BOTH sides (partial Spearman via
    Pearson-on-ranks). Reports before/after per cell (esp. the turn_nl L22 peak
    ρ=−0.607) and whether the honest selection-symmetric band-clearing survives.
(b) **Both r_B recipes.** ``store/r_b.pt`` carries exactly TWO recipes per behavior
    ({diffmeans, meanDB}); the committed read-out used diffmeans only. Compute the
    refusal read separately per recipe.
(c) **Per-context scatter triples** for (refusal, turn_nl, L22): the 50
    (prediction, graded score, answer length) triples, for the analyzer's
    low-level plot.
(d) **Max-pool late-layer-edge length check.** Over the persisted per-context
    LOCO error decomposition (``analysis/bootstrap_deltaskill.json``), Spearman of
    the per-context (mean_err − maxp_err) gap at L21 vs answer length — a
    length-driven pooling artifact would show up here.

NULL for (a)/(b) (selection-symmetric, `.claude/rules/selection-symmetric-nulls.md`):
permute the graded-E0 vector ONCE per draw and apply the SAME permutation to every
(summary × layer × recipe) cell, then take the per-draw max |ρ| over the cells in
scope — the joint null respects the cross-cell correlation structure (a refinement
of the committed band, whose per-cell perms were drawn independently). Both the
RAW and the PARTIALLED statistic get a band from the SAME draws, so before/after
band-clearing is apples-to-apples.

Sanity gate: the recomputed raw ρ must reproduce the committed
``readout_rho_by_summary.json`` (refusal × fixed_rb) cells to <1e-9, including the
turn_nl L22 peak −0.6071068427370948 — asserted before any partialling.

Answer length = per-context ``median_answer_len`` from the Phase B position-store
manifest ``per_context_diag`` (the same length the committed length_control used).

Usage (CPU, VM — this run IS the production run)::

    uv run python scripts/issue810_fa_refusal_diagnostics.py \\
        --out eval_results/issue_810/analysis/fa_refusal_diagnostics.json
"""

from __future__ import annotations

import argparse
import logging

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# issue810_fit_readout calls load_dotenv at import (HF token + shared-VM thread caps).
from issue810_common import (  # noqa: E402
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    SHUFFLE_NULL_SEED,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    summary_names,
)
from issue810_fit_readout import (  # noqa: E402
    _kept_contexts,
    _load_free_summaries,
    _load_position_summaries,
    _load_rb,
    _rho,
)

logger = logging.getLogger("issue810_fa_refusal_diagnostics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BEHAVIOR = "refusal"
RB_RECIPES = ("diffmeans", "meanDB")
HEADLINE = {"summary": "turn_nl", "layer": 22}
PROFILE_SUMMARIES = ("mean", "im_end", "turn_nl")
BAND_PCTILE = 97.5


# ── rank / partial-correlation primitives (vectorized) ───────────────────────


def _std_rows(a: np.ndarray) -> np.ndarray:
    """Center + unit-norm rows of (k, n) so a row·row' dot product is a Pearson r."""
    a = a - a.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = np.nan  # degenerate (constant) row → NaN correlations, guarded later
    return a / nrm


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks along axis=1 (Spearman == Pearson on these ranks)."""
    return rankdata(a, axis=1, method="average")


def _partial_from_pairwise(r_xy, r_xz, r_yz):
    """First-order partial correlation r_xy·z from the three pairwise r's (broadcasts)."""
    den = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r_xy - r_xz * r_yz) / den


def _partial_p_two_sided(r: float, n: int) -> float | None:
    """Two-sided p for a first-order partial correlation (t with n−3 df)."""
    df = n - 3
    if df < 1 or r is None or not np.isfinite(r) or abs(r) >= 1.0:
        return None
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * t_dist.sf(abs(t), df))


def _f(x) -> float | None:
    """np scalar → JSON-safe float (NaN → None)."""
    x = float(x)
    return x if np.isfinite(x) else None


# ── (a)+(b): length-partialled fixed-r_B read-out, both recipes ──────────────


def _readout_cells_and_bands(
    ctx_ids,
    free_summaries,
    pos_summaries,
    coverage,
    rb,
    graded,
    lengths,
    capture_layers,
    n_perms,
    rng,
):
    """Per-cell raw+partial ρ for BOTH r_B recipes + joint selection-symmetric null bands.

    Returns (cells, bands). cells: list of dicts (recipe, summary, layer, n, rho_raw,
    rho_partial, p_partial). bands: per-recipe + combined 97.5-pctile per-draw-max
    |ρ| bands for the raw AND partialled statistic (same draws → apples-to-apples).
    """
    summaries = summary_names()
    n_layers = len(capture_layers)
    cells: list[dict] = []
    # per-draw max |rho| accumulators: {recipe: (n_perms,)}
    max_raw = {r: np.zeros(n_perms) for r in RB_RECIPES}
    max_par = {r: np.zeros(n_perms) for r in RB_RECIPES}
    perm_cache: dict[tuple, np.ndarray] = {}  # kept-set → (n_perms, n) shared perms

    for summary in summaries:
        kept = [
            c for c in _kept_contexts(summary, ctx_ids, coverage) if c in graded and c in lengths
        ]
        n = len(kept)
        if n < 4:
            logger.info("[phase=fit] %s skipped (n=%d < 4)", summary, n)
            continue
        key = tuple(kept)
        if key not in perm_cache:
            # ONE shared permutation per draw for every cell on this kept-set (joint null).
            perm_cache[key] = np.stack([rng.permutation(n) for _ in range(n_perms)])
        perms = perm_cache[key]

        # (n, L, H) design; predictions for both recipes via one einsum each.
        X = np.stack(
            [
                free_summaries[summary][c].float().numpy()
                if summary in ("mean", "last", "maxp")
                else pos_summaries[c][summary]
                for c in kept
            ]
        )  # (n, L, H)
        assert X.shape[:2] == (n, n_layers), X.shape
        y = np.array([graded[c] for c in kept], dtype=np.float64)
        z = np.array([lengths[c] for c in kept], dtype=np.float64)
        ry = rankdata(y, method="average")
        rz = rankdata(z, method="average")
        ry_s = _std_rows(ry[None, :])[0]  # (n,)
        rz_s = _std_rows(rz[None, :])[0]  # (n,)
        r_yz = float(ry_s @ rz_s)
        # permuted-y rank rows, standardized once (ranks are permutation-equivariant)
        ry_perm_s = _std_rows(ry[perms])  # (n_perms, n)
        r_yz_null = ry_perm_s @ rz_s  # (n_perms,)

        for recipe in RB_RECIPES:
            r_dir = rb[BEHAVIOR][recipe].float().numpy()  # (L, H)
            # Per-layer fp32 matmul — MUST match the committed issue810_fit_readout
            # reduction (per-layer ``X @ r``) bit-exactly; a single fp32 einsum over
            # all layers changes the summation order and flipped ranks on 2/1036
            # near-tie cells (caught by the sanity gate on the first run).
            preds = np.stack(
                [np.ascontiguousarray(X[:, li, :]) @ r_dir[li] for li in range(n_layers)],
                axis=1,
            )  # (n, L)
            rx = _rank_rows(preds.T)  # (L, n)
            rx_s = _std_rows(rx)  # (L, n)
            r_xy = rx_s @ ry_s  # (L,)
            r_xz = rx_s @ rz_s  # (L,)
            r_par = _partial_from_pairwise(r_xy, r_xz, r_yz)  # (L,)
            # joint null: same y-permutation across every cell
            r_xy_null = rx_s @ ry_perm_s.T  # (L, n_perms)
            r_par_null = _partial_from_pairwise(r_xy_null, r_xz[:, None], r_yz_null[None, :])
            max_raw[recipe] = np.maximum(max_raw[recipe], np.nanmax(np.abs(r_xy_null), axis=0))
            max_par[recipe] = np.maximum(max_par[recipe], np.nanmax(np.abs(r_par_null), axis=0))
            for li in range(n_layers):
                cells.append(
                    {
                        "recipe": recipe,
                        "summary": summary,
                        "layer": capture_layers[li],
                        "n": n,
                        "rho_raw": _f(r_xy[li]),
                        "rho_len_vs_pred": _f(r_xz[li]),
                        "rho_partial": _f(r_par[li]),
                        "p_partial": _partial_p_two_sided(_f(r_par[li]), n),
                    }
                )
        logger.info("[phase=fit] %s done (n=%d, both recipes)", summary, n)

    bands = {}
    for recipe in RB_RECIPES:
        bands[recipe] = {
            "raw_abs_band": _f(np.percentile(max_raw[recipe], BAND_PCTILE)),
            "partial_abs_band": _f(np.percentile(max_par[recipe], BAND_PCTILE)),
        }
    both_raw = np.maximum(max_raw["diffmeans"], max_raw["meanDB"])
    both_par = np.maximum(max_par["diffmeans"], max_par["meanDB"])
    bands["both_recipes"] = {
        "raw_abs_band": _f(np.percentile(both_raw, BAND_PCTILE)),
        "partial_abs_band": _f(np.percentile(both_par, BAND_PCTILE)),
    }
    bands["_meta"] = {
        "n_perms": n_perms,
        "band_pctile": BAND_PCTILE,
        "scope": f"max |rho| over (summary x layer) cells, behavior={BEHAVIOR}, "
        "fixed-r_B method only; per-recipe + both-recipe scopes",
        "null": "permute graded-E0 ONCE per draw, SAME permutation applied to every cell "
        "(joint null; the committed band drew per-cell perms independently). The permute-y "
        "null breaks the graded~length coupling too (a Freedman-Lane variant would preserve "
        "it); with rho(len, graded)~-0.25 the difference is second-order.",
    }
    return cells, bands


def _sanity_check_against_committed(cells: list[dict], committed: dict) -> dict:
    """Assert the recomputed raw ρ reproduces the committed refusal fixed_rb cells (<1e-9)."""
    committed_map = {
        (c["summary"], c["layer"]): c["rho_graded"]
        for c in committed["cells"]
        if c["behavior"] == BEHAVIOR and c["method"] == "fixed_rb"
    }
    diffs = []
    for c in cells:
        if c["recipe"] != "diffmeans" or c["rho_raw"] is None:
            continue
        ref = committed_map.get((c["summary"], c["layer"]))
        if ref is not None:
            diffs.append(abs(c["rho_raw"] - ref))
    if not diffs:
        raise RuntimeError("sanity check matched 0 committed cells — schema drift?")
    max_diff = float(max(diffs))
    if max_diff > 1e-9:
        raise RuntimeError(
            f"recomputed raw rho drifts from committed readout_rho_by_summary.json "
            f"(max |diff|={max_diff:.3e} over {len(diffs)} cells) — refusing to partial "
            f"a read-out that does not reproduce the committed one"
        )
    head = next(
        c
        for c in cells
        if c["recipe"] == "diffmeans"
        and c["summary"] == HEADLINE["summary"]
        and c["layer"] == HEADLINE["layer"]
    )
    exp = committed_map[(HEADLINE["summary"], HEADLINE["layer"])]
    assert abs(head["rho_raw"] - exp) < 1e-9, (head["rho_raw"], exp)
    logger.info(
        "[phase=sanity] raw rho reproduces committed cells (max diff %.2e, n=%d); "
        "turn_nl L22 = %.6f",
        max_diff,
        len(diffs),
        head["rho_raw"],
    )
    return {
        "n_cells_matched": len(diffs),
        "max_abs_diff": max_diff,
        "headline_raw_rho": head["rho_raw"],
        "committed_headline_rho": exp,
    }


# ── (d): max-pool late-layer-edge length check ────────────────────────────────


def _maxp_l21_length_check(boot: dict, lengths: dict) -> dict:
    """Spearman of the per-context (mean_err − maxp_err) LOCO-error gap vs answer length.

    Primary read: raw ss_res gap at L21 (the brief's statistic). Secondary: the
    normalized ss_res/ss_tot gap (mean and maxp targets have different scales),
    the per-summary err~length reads at L21, an L18 non-edge contrast, and the
    per-layer profile of rho(gap_L, length) across all 28 layers.
    """
    pc = boot["per_context_decomposition"]
    ids = pc["mean"]["context_ids"]
    if pc["maxp"]["context_ids"] != ids:
        raise RuntimeError("mean/maxp context_ids order mismatch in bootstrap_deltaskill.json")
    if any(c not in lengths for c in ids):
        missing = [c for c in ids if c not in lengths][:3]
        raise RuntimeError(f"contexts missing answer length (e.g. {missing})")
    ss_res_mean = np.asarray(pc["mean"]["ss_res"], dtype=np.float64)  # (28, 50)
    ss_res_maxp = np.asarray(pc["maxp"]["ss_res"], dtype=np.float64)  # (28, 50)
    ss_tot_mean = np.asarray(pc["mean"]["ss_tot"], dtype=np.float64)
    ss_tot_maxp = np.asarray(pc["maxp"]["ss_tot"], dtype=np.float64)
    assert ss_res_mean.shape == ss_res_maxp.shape and ss_res_mean.shape[1] == len(ids)
    lens = np.array([lengths[c] for c in ids], dtype=np.float64)

    def _sp(a: np.ndarray, b: np.ndarray) -> dict:
        r, p = spearmanr(a, b)
        return {"rho": _f(r), "p": _f(p)}

    def _layer_read(li: int) -> dict:
        gap_raw = ss_res_mean[li] - ss_res_maxp[li]
        gap_norm = ss_res_mean[li] / ss_tot_mean[li] - ss_res_maxp[li] / ss_tot_maxp[li]
        return {
            "gap_raw_vs_length": _sp(gap_raw, lens),
            "gap_normalized_vs_length": _sp(gap_norm, lens),
            "mean_err_vs_length": _sp(ss_res_mean[li], lens),
            "maxp_err_vs_length": _sp(ss_res_maxp[li], lens),
        }

    profile = []
    for li in range(ss_res_mean.shape[0]):
        gap = ss_res_mean[li] - ss_res_maxp[li]
        r, p = spearmanr(gap, lens)
        profile.append({"layer": li, "rho_gap_vs_length": _f(r), "p": _f(p)})
    return {
        "statistic": "per-context (mean_err - maxp_err) LOCO ss_res gap vs median answer "
        "length (tokens), Spearman; gap > 0 = maxp better on that context",
        "n_contexts": len(ids),
        "L21": _layer_read(21),
        "L18_non_edge_contrast": _layer_read(18),
        "per_layer_profile_gap_raw": profile,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 FA: refusal read-out diagnostics")
    ap.add_argument(
        "--e0-highm",
        default=str(PROJECT_ROOT / "eval_results/issue_810/phase_c/e0_highm_graded.json"),
    )
    ap.add_argument(
        "--readout-json",
        default=str(PROJECT_ROOT / "eval_results/issue_810/readout_rho_by_summary.json"),
    )
    ap.add_argument(
        "--bootstrap-json",
        default=str(PROJECT_ROOT / "eval_results/issue_810/analysis/bootstrap_deltaskill.json"),
    )
    ap.add_argument(
        "--position-store-hf", default="issue658_theory_assumptions/answer_position_sweep"
    )
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results/issue_810/analysis/fa_refusal_diagnostics.json"),
    )
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    logger.info("[phase=load] manifest + summaries + r_B + graded E0 + lengths")
    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids_all = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries()
    rb, _ = _load_rb()
    recipes_in_store = sorted(k for k, v in rb[BEHAVIOR].items() if hasattr(v, "shape"))
    if set(RB_RECIPES) != set(recipes_in_store):
        raise RuntimeError(
            f"r_b.pt recipe drift: expected {RB_RECIPES}, store has {recipes_in_store}"
        )

    pos_man = load_json(
        hf_hub_download(
            HF_DATA_REPO, f"{args.position_store_hf}/manifest.json", repo_type="dataset"
        )
    )
    if pos_man["capture_layers"] != list(capture_layers):
        raise RuntimeError("position-store capture_layers != v0_summaries capture_layers")
    ctx_ids = [c for c in pos_man["context_ids"] if c in ctx_ids_all]
    pos_summaries, coverage = _load_position_summaries(ctx_ids, args.position_store_hf, None)

    e0_highm = load_json(args.e0_highm)
    blk = e0_highm["by_behavior"][BEHAVIOR]
    graded = {k: v for k, v in blk["per_context_graded_mean"].items() if v is not None}
    lengths = {
        c: d["median_answer_len"]
        for c, d in pos_man["per_context_diag"].items()
        if d.get("median_answer_len") is not None
    }
    committed = load_json(args.readout_json)
    boot = load_json(args.bootstrap_json)

    # (a) + (b): raw + partial per cell, both recipes, joint null bands.
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    cells, bands = _readout_cells_and_bands(
        ctx_ids,
        free_summaries,
        pos_summaries,
        coverage,
        rb,
        graded,
        lengths,
        list(capture_layers),
        args.n_perms,
        rng,
    )
    sanity = _sanity_check_against_committed(cells, committed)

    def _top(recipe: str, key: str, k: int = 10) -> list[dict]:
        pool = [c for c in cells if c["recipe"] == recipe and c[key] is not None]
        return sorted(pool, key=lambda c: -abs(c[key]))[:k]

    headline = next(
        c
        for c in cells
        if c["recipe"] == "diffmeans"
        and c["summary"] == HEADLINE["summary"]
        and c["layer"] == HEADLINE["layer"]
    )
    diff_cells = [c for c in cells if c["recipe"] == "diffmeans"]
    obs_max_raw = max(abs(c["rho_raw"]) for c in diff_cells if c["rho_raw"] is not None)
    obs_max_par = max(abs(c["rho_partial"]) for c in diff_cells if c["rho_partial"] is not None)
    a_block = {
        "behavior": BEHAVIOR,
        "method": "fixed_rb (diffmeans — the committed read)",
        "partialled_covariate": "median_answer_len (per-context, position-store manifest)",
        "sanity_vs_committed": sanity,
        "headline_turn_nl_L22": headline,
        "profiles": {
            s: [
                {k: c[k] for k in ("layer", "rho_raw", "rho_partial", "p_partial")}
                for c in diff_cells
                if c["summary"] == s
            ]
            for s in PROFILE_SUMMARIES
        },
        "top10_by_abs_rho_raw": _top("diffmeans", "rho_raw"),
        "top10_by_abs_rho_partial": _top("diffmeans", "rho_partial"),
        "observed_max_abs_rho_raw": _f(obs_max_raw),
        "observed_max_abs_rho_partial": _f(obs_max_par),
        "bands": bands,
        "band_clearing": {
            "raw_clears_diffmeans_band": bool(obs_max_raw > bands["diffmeans"]["raw_abs_band"]),
            "partial_clears_diffmeans_band": bool(
                obs_max_par > bands["diffmeans"]["partial_abs_band"]
            ),
            "committed_fixed_rb_abs_band_reference": 0.5972388955582233,
        },
        "cells": diff_cells,
    }

    meandb_cells = [c for c in cells if c["recipe"] == "meanDB"]
    b_block = {
        "recipes_in_store": recipes_in_store,
        "note": "store/r_b.pt carries exactly two recipes per behavior (diffmeans, meanDB); "
        "the committed readout used diffmeans only.",
        "meanDB": {
            "headline_turn_nl_L22": next(
                c
                for c in meandb_cells
                if c["summary"] == HEADLINE["summary"] and c["layer"] == HEADLINE["layer"]
            ),
            "top10_by_abs_rho_raw": _top("meanDB", "rho_raw"),
            "top10_by_abs_rho_partial": _top("meanDB", "rho_partial"),
            "observed_max_abs_rho_raw": _f(
                max(abs(c["rho_raw"]) for c in meandb_cells if c["rho_raw"] is not None)
            ),
            "cells": meandb_cells,
        },
    }

    # (c): 50 scatter triples at (refusal, turn_nl, L22), diffmeans.
    kept_c = [
        c for c in _kept_contexts("turn_nl", ctx_ids, coverage) if c in graded and c in lengths
    ]
    li = list(capture_layers).index(HEADLINE["layer"])
    Xc = np.stack([pos_summaries[c]["turn_nl"][li] for c in kept_c])
    pred_c = Xc @ rb[BEHAVIOR]["diffmeans"][li].float().numpy()
    c_block = {
        "cell": {
            "behavior": BEHAVIOR,
            "method": "fixed_rb/diffmeans",
            "summary": "turn_nl",
            "layer": HEADLINE["layer"],
        },
        "n": len(kept_c),
        "triples": [
            {
                "context_id": c,
                "prediction": _f(pred_c[i]),
                "graded_e0": _f(graded[c]),
                "answer_len": lengths[c],
            }
            for i, c in enumerate(kept_c)
        ],
        "sanity_rho": _f(_rho(pred_c, np.array([graded[c] for c in kept_c]))),
    }

    # (d): max-pool L21 edge vs length.
    d_block = _maxp_l21_length_check(boot, lengths)

    out = {
        "dv": "fa_refusal_diagnostics (length-partialled read-out + r_B recipes + scatter "
        "triples + maxp L21 length check)",
        "inputs": {
            "e0_highm": str(args.e0_highm),
            "readout_json": str(args.readout_json),
            "bootstrap_json": str(args.bootstrap_json),
            "position_store_hf": args.position_store_hf,
            "r_b": "issue658_theory_assumptions/store/r_b.pt",
        },
        "n_perms": args.n_perms,
        "seed": SHUFFLE_NULL_SEED,
        "a_length_partialled_readout": a_block,
        "b_rb_recipes": b_block,
        "c_scatter_refusal_turn_nl_L22": c_block,
        "d_maxp_L21_length_check": d_block,
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(out, args.out)
    logger.info(
        "[phase=done] wrote %s | headline turn_nl L22: raw %.3f -> partial %.3f (p=%.3g) | "
        "d: gap@L21 vs len rho=%.3f (p=%.3g)",
        args.out,
        headline["rho_raw"],
        headline["rho_partial"],
        headline["p_partial"],
        d_block["L21"]["gap_raw_vs_length"]["rho"],
        d_block["L21"]["gap_raw_vs_length"]["p"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
