#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →) in scientific docstrings/log messages.
"""Issue #841 free-analysis follow-up ``wincell-stability`` — win-cell stability.

Question: are the parent's 12/68 primary ridge transport wins a STABLE property
of specific (trait, source-layer, mode) cells, or does the win-set churn under
fit-set resampling? K=10 DISJOINT n=4000 fit draws from the 96k scaling-capture
pool (new-context rows only — disjoint from the parent's pass_b anchor fit rows
by construction), refit the parent's closed-form PRESS/GCV affine ridge one-step
maps per draw, roll each draw's maps forward to the trait read-out layers on
 #779's rig, apply the parent Stage-1 conjunction predicate verbatim (transported
beats BOTH raw-source AND identity-transport, per-condition-paired bootstrap CI
excluding zero, seed 0), and report per-cell win frequency, cross-draw Jaccard,
the ≥8/10 core set, each vs an exchangeable permutation null (win counts fixed
per draw, identities permuted within the 68-cell grid).

Everything recipe-bearing is REUSED, not rewritten: the ridge solver + transport
come from ``explore_persona_space.experiments.issue_841.maps`` (KILL-B verified
primal ≡ dual at n=4000; ``--solver dual`` is available and a one-off in-run
parity spot-check asserts the two agree on this run's data), the eval frame /
grouping / vectorized #779 bootstrap come from ``issue841_common`` +
``issue841_stage1_benchmark`` + ``issue841_scaling_stage1``, and the fixed
comparator rows (raw source, identity transport) come from the parent's
committed ``eval_results/issue_841/stage1_projections.npz`` (asserted against a
recompute from the rebuilt eval matrix).

Capture shards stream ONE AT A TIME from HF (private overflow repo per the
 #541 LFS reroute; manifest on the public data repo) into a scratch dir that is
DELETED after each shard's rows are extracted — peak disk ~one shard (~4 GB),
peak RAM ~draws (bf16, ~0.8 GB/draw) + one shard.

--smoke: K=2 draws × n=500 from the FIRST shard only, evil trait, coarse
2-source grid (4 cells), n_boot=200 — same code path end to end, outputs
diverted to a scratch dir (never the committed eval_results/figures paths).
--synthetic: no-HF CPU wiring test (fabricated pool + eval frame, small hidden).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue841_common as C  # noqa: E402
import issue841_scaling_common as S  # noqa: E402
import issue841_scaling_stage1 as SS1  # noqa: E402
import issue841_stage1_benchmark as B1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_wincell")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "wincell-stability"
FIG_ROOT = PROJECT_ROOT / "figures"
FIG_SUBDIR = "issue_841/wincell-stability"
SCRATCH_DEFAULT = PROJECT_ROOT / "data" / "issue_841" / "wincell_scratch"

K_DRAWS = 10
DRAW_N = 4000
PERM_SEED = 0  # seeds the disjoint-draw permutation of the capture pool
N_NULL = 10000
CORE_FRAC = 0.8  # core set = cells winning in >= ceil(CORE_FRAC * K) draws
NULL_SEED = 0

PARENT_BENCHMARK = PROJECT_ROOT / "eval_results" / "issue_841" / "stage1_benchmark.json"
PARENT_PROJECTIONS = PROJECT_ROOT / "eval_results" / "issue_841" / "stage1_projections.npz"
PARENT_FIDELITY = PROJECT_ROOT / "eval_results" / "issue_841" / "transport_fidelity.json"


def _clean(d: dict) -> dict:
    """Drop private keys (the vectorized helpers stash ``_arr``) for JSON."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _sha256_ints(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(np.int64)).tobytes()).hexdigest()[:16]


# ── draws from the capture pool ──────────────────────────────────────────────────


def draw_indices(pool_rows: int, k_draws: int, draw_n: int, seed: int) -> np.ndarray:
    """(k_draws, draw_n) DISJOINT sorted row indices into the capture pool.

    One seeded permutation, consecutive blocks — draws are pairwise disjoint and
    (because the pool holds only NEW capture rows) disjoint from the parent's
    pass_b anchor fit rows by construction. Sorted per draw for shard streaming.
    """
    need = k_draws * draw_n
    assert need <= pool_rows, f"need {need} rows, pool has {pool_rows}"
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool_rows)
    return np.sort(perm[:need].reshape(k_draws, draw_n), axis=1)


def _fetch_manifest() -> dict:
    """The capture manifest from the PUBLIC data repo (tiny, non-LFS)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        C.HF_DATA_REPO,
        filename=f"{S.HF_CAPTURE_BUCKET}/manifest.json",
        repo_type="dataset",
    )
    with open(local) as f:
        return json.load(f)


def _download_shard(repo: str, rel: str, scratch: Path, attempts: int = 4) -> Path:
    """hf_hub_download one shard into ``scratch`` (caller deletes after use)."""
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return Path(
                hf_hub_download(repo, filename=rel, repo_type="dataset", cache_dir=str(scratch))
            )
        except Exception as e:  # transient 5xx/429 — retry with linear backoff
            last = e
            wait = 20 * (attempt + 1)
            logger.warning("[shard] download %s failed (%s); retry in %ds", rel, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"shard download failed after {attempts} attempts: {rel}") from last


def load_pool_draws(
    idx: np.ndarray,
    needed: list[int],
    capture_dir: Path,
    scratch_dir: Path,
    pool_limit: int | None = None,
) -> dict[int, torch.Tensor]:
    """Stream capture shards one at a time; extract each needed draw's rows.

    Returns {draw_k: (draw_n, 28, H) tensor in the shards' own storage dtype
    (bf16 for the real capture)}. Shards already local under ``capture_dir`` are
    used in place (never deleted); shards downloaded to ``scratch_dir`` are
    deleted the moment their rows are extracted, so peak disk stays ~one shard.
    ``pool_limit`` (smoke) skips spans entirely above the limit.
    """
    manifest = _fetch_manifest()
    total = int(manifest["total_rows"])
    hi_need = int(idx[needed].max()) if needed else -1
    assert hi_need < total, (hi_need, total)
    overflow = S._overflow_repo_for_bucket(C.HF_DATA_REPO, S.HF_CAPTURE_BUCKET)
    shard_repo = overflow or C.HF_DATA_REPO
    logger.info(
        "[pool] manifest total_rows=%d dtype=%s shard_repo=%s spans=%d",
        total,
        manifest.get("realized_capture_dtype", manifest.get("capture_dtype")),
        shard_repo,
        len(manifest["spans"]),
    )
    out: dict[int, torch.Tensor] = {}
    filled = dict.fromkeys(needed, 0)
    for span in manifest["spans"]:
        lo, hi = int(span["row_lo"]), int(span["row_hi"])
        if pool_limit is not None and lo >= pool_limit:
            continue
        # which draws need rows from this span?
        span_slices = {}
        for k in needed:
            a, b = np.searchsorted(idx[k], [lo, hi])
            if b > a:
                span_slices[k] = (a, b)
        if not span_slices:
            continue
        local = capture_dir / span["shard"]
        downloaded = None
        if not local.exists():
            scratch = scratch_dir / f"span_{span['shard']}"
            local = _download_shard(shard_repo, f"{S.HF_CAPTURE_BUCKET}/{span['shard']}", scratch)
            downloaded = scratch
        blob = torch.load(local, weights_only=False)
        cx = blob["cx_last"]
        assert cx.shape[0] == hi - lo, (cx.shape, lo, hi)
        assert cx.shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), cx.shape
        for k, (a, b) in span_slices.items():
            if k not in out:
                out[k] = torch.empty((idx.shape[1], *cx.shape[1:]), dtype=cx.dtype)
            rows = torch.from_numpy(np.ascontiguousarray(idx[k][a:b] - lo))
            out[k][a:b] = cx[rows]
            filled[k] += b - a
        del blob, cx
        gc.collect()
        if downloaded is not None:
            shutil.rmtree(downloaded, ignore_errors=True)
        logger.info(
            "[pool] span %s done (rows %d..%d); RSS %.1f GiB", span["shard"], lo, hi, S.rss_gib()
        )
    for k in needed:
        assert filled[k] == idx.shape[1], f"draw {k}: filled {filled[k]} != {idx.shape[1]}"
    return out


# ── per-draw ridge maps + win records ─────────────────────────────────────────────


def needed_transitions_primary(traits: list[str], smoke: bool) -> list[int]:
    """Union of one-step transitions any (trait, PRIMARY-scheme, source) needs."""
    need: set[int] = set()
    for trait in traits:
        tgt = C.PRIMARY_TARGET_LAYER[trait]
        for s in B1.source_grid(tgt, smoke):
            need.update(range(s, tgt))
    return sorted(need)


def fit_draw_maps(
    draw: torch.Tensor, transitions: list[int], device: str, solver: str
) -> dict[int, MP.RidgeMap]:
    """Fit the RAW-target one-step affine PRESS/GCV ridge maps for one draw.

    The parent recipe verbatim (``issue841_scaling_stage0.ridge_scaling`` raw
    leg): Δ target in RAW space (sigma=1.0), λ by exact PRESS-LOO over
    ``RIDGE_LAMBDAS``, affine intercept. ``solver`` picks the #658 dual path
    (the parent's anchor path at n≤10k) or the primal path — exact PRESS in the
    primal eigenbasis, KILL-B-verified identical at n=4000 (stage0 anchor gate);
    the primal is ~2× faster at (n=4000, d=3584) on CPU.
    """
    fit_fn = MP.fit_ridge_split if solver == "dual" else MP.fit_ridge_primal
    maps: dict[int, MP.RidgeMap] = {}
    for t in transitions:
        h = draw[:, t, :].to(torch.float32).numpy()
        d_fit = draw[:, t + 1, :].to(torch.float32).numpy() - h
        t0 = time.time()
        _pred, rmap = fit_fn(h, d_fit, h[:1], sigma=1.0, device=device)
        maps[t] = rmap
        logger.info(
            "[fit] t=%d n=%d solver=%s lam=%g wall=%.1fs RSS %.1f GiB",
            t,
            h.shape[0],
            solver,
            rmap.best_lam,
            time.time() - t0,
            S.rss_gib(),
        )
    return maps


def parity_spot_check(draw: torch.Tensor, transition: int, device: str) -> dict:
    """One-transition dual-vs-primal parity assert on THIS run's data.

    Mirrors the parent's KILL-B anchor gate: both solvers are exact PRESS over
    the same λ grid, so the selected λ must match and the eval predictions must
    agree to fp64 noise. Hard-fails on divergence.
    """
    h = draw[:, transition, :].to(torch.float32).numpy()
    d_fit = draw[:, transition + 1, :].to(torch.float32).numpy() - h
    ev = h[:8]
    pred_d, rmap_d = MP.fit_ridge_split(h, d_fit, ev, sigma=1.0, device=device)
    pred_p, rmap_p = MP.fit_ridge_primal(h, d_fit, ev, sigma=1.0, device=device)
    max_diff = float(np.abs(pred_d - pred_p).max())
    ok = (rmap_d.best_lam == rmap_p.best_lam) and max_diff < 1e-6
    result = {
        "transition": transition,
        "n": int(h.shape[0]),
        "lam_dual": rmap_d.best_lam,
        "lam_primal": rmap_p.best_lam,
        "max_abs_pred_diff": max_diff,
        "pass": bool(ok),
    }
    logger.info("[parity] %s", result)
    assert ok, f"dual-vs-primal parity spot-check FAILED: {result}"
    return result


def load_eval_inputs(traits: list[str], smoke: bool, synthetic: dict | None) -> dict:
    """Per-trait eval frame + r_B + fixed comparator rows + cached bootstrap bits.

    Real path: rebuild the #779 eval matrix via ``issue841_common`` (pinned HF
    revision) and assert it aligns with the parent's committed
    ``stage1_projections.npz`` (y bit-equal; raw-source / id-transport rows
    reproduce within fp tolerance) — then use the npz rows as the FIXED
    comparators the win predicate pairs against.
    """
    out: dict = {}
    npz = None
    if synthetic is None:
        assert PARENT_PROJECTIONS.exists(), f"missing comparator npz: {PARENT_PROJECTIONS}"
        npz = np.load(PARENT_PROJECTIONS)
    for trait in traits:
        if synthetic is not None:
            mat, r_b = synthetic["trait_inputs"][trait]
            tgt = synthetic["schemes"]["primary"]
            sources = synthetic["sources"]
        else:
            mat = C.build_eval_traj_matrix(C.load_eval_cells(trait))
            r_b = C.load_rb(trait)
            tgt = C.PRIMARY_TARGET_LAYER[trait]
            sources = B1.source_grid(tgt, smoke)
            assert np.array_equal(npz[f"{trait}__y"], mat["y"]), f"{trait}: y misaligned vs npz"
            assert np.array_equal(npz[f"{trait}__cond"], mat["cond"]), f"{trait}: cond misaligned"
            assert np.array_equal(
                npz[f"{trait}__mode_is_manyshot"], (mat["mode"] == "many_shot").astype(np.int64)
            ), f"{trait}: mode misaligned"
        traj_dev = torch.from_numpy(np.ascontiguousarray(mat["traj"])).to(torch.float32)
        rb_dev = torch.from_numpy(np.ascontiguousarray(r_b)).to(torch.float32)
        cells = {}
        for src in sources:
            if npz is not None:
                row_raw = npz[f"{trait}__primary__{src}__raw_source"]
                row_id = npz[f"{trait}__primary__{src}__id_transport"]
                # Comparator rows must reproduce from the rebuilt matrix (alignment
                # gate). Scale-aware bar: fp32 dot products of magnitude ~1e3 carry
                # ~3e-4 abs accumulation noise vs the parent's stored rows (measured
                # rel 6.4e-7); a real row-order/misalignment bug reads rel ~O(1).
                mine_raw = B1._proj(traj_dev[:, src, :], rb_dev[src])
                mine_id = B1._proj(traj_dev[:, src, :], rb_dev[tgt])
                for name, ref, mine in (
                    ("raw_source", row_raw, mine_raw),
                    ("id_transport", row_id, mine_id),
                ):
                    rel = np.abs(ref - mine).max() / max(1.0, np.abs(ref).max())
                    assert rel <= 1e-4, (trait, src, name, f"rel diff {rel:.3e} > 1e-4")
            else:
                row_raw = B1._proj(traj_dev[:, src, :], rb_dev[src])
                row_id = B1._proj(traj_dev[:, src, :], rb_dev[tgt])
            per_mode = {}
            for mode in B1.MODES:
                gx_raw, gy = SS1._grouped(row_raw, mat, mode)
                gx_id, _ = SS1._grouped(row_id, mat, mode)
                if len(gy) < 2:
                    continue
                per_mode[mode] = {
                    "gy": gy,
                    "pc_raw": SS1._per_cond_r(gx_raw, gy),
                    "pc_id": SS1._per_cond_r(gx_id, gy),
                    "n_cond": len(gy),
                }
            cells[src] = per_mode
        out[trait] = {
            "mat": mat,
            "traj_dev": traj_dev,
            "rb_dev": rb_dev,
            "target": tgt,
            "sources": sources,
            "cells": cells,
        }
        logger.info(
            "[eval] %s: %d rows, %d conditions, target=L%d, %d sources",
            trait,
            len(mat["y"]),
            len(mat["cond_ids"]),
            tgt,
            len(sources),
        )
    return out


def win_records_for_draw(
    maps: dict[int, MP.RidgeMap], eval_inputs: dict, *, n_boot: int, seed: int
) -> list[dict]:
    """Apply the parent Stage-1 conjunction predicate per primary cell for one draw.

    Verbatim reuse of the scaling Stage-1 machinery: transported projection via
    ``B1._transport_proj`` (iterated one-step composition), grouping via
    ``SS1._grouped``, and the vectorized #779 ``bootstrap_delta_ci`` equivalent
    (``SS1._resample_idx`` + ``_per_cond_r`` + ``_delta_fast``; bit-identical
    draw sequence at the shared seed). win = transported beats BOTH raw-source
    AND id-transport with the 95% CI excluding zero and delta>0.
    """
    records = []
    samps_cache: dict[int, list] = {}
    for trait, ti in eval_inputs.items():
        tgt = ti["target"]
        for src in ti["sources"]:
            x_t, h_hat = B1._transport_proj(maps, ti["traj_dev"], src, tgt, ti["rb_dev"][tgt])
            fid = B1.transport_fidelity(h_hat, ti["traj_dev"], src, tgt)
            for mode, cm in ti["cells"][src].items():
                n_cond = cm["n_cond"]
                if n_cond not in samps_cache:
                    samps_cache[n_cond] = SS1._resample_idx(seed, n_cond, n_boot)
                samps = samps_cache[n_cond]
                gx_t, _ = SS1._grouped(x_t, ti["mat"], mode)
                pc_t = SS1._per_cond_r(gx_t, cm["gy"])
                d_raw = SS1._delta_fast(pc_t, cm["pc_raw"], samps)
                d_id = SS1._delta_fast(pc_t, cm["pc_id"], samps)
                win = bool(
                    d_raw["excludes_zero"]
                    and d_raw["delta"] > 0
                    and d_id["excludes_zero"]
                    and d_id["delta"] > 0
                )
                records.append(
                    {
                        "cell": f"{trait}|primary|{src}|{mode}",
                        "trait": trait,
                        "source": int(src),
                        "target": int(tgt),
                        "mode": mode,
                        "win": win,
                        "vs_raw_source": _clean(d_raw),
                        "vs_id_transport": _clean(d_id),
                        "fidelity_cosine": fid["cosine_hhat_vs_true"],
                        "fidelity_delta_recon_r2": fid["delta_recon_r2_id"],
                    }
                )
        logger.info("[win] trait=%s done (%d sources)", trait, len(ti["sources"]))
    return records


# ── aggregation: frequency, Jaccard, core set, exchangeable null ──────────────────


def jaccard_and_core(win_matrix: np.ndarray, core_min: int) -> dict:
    """Observed cross-draw Jaccard matrix + mean, per-cell frequency, core set."""
    _n_cells, k = win_matrix.shape
    freqs = win_matrix.mean(axis=1)
    counts = win_matrix.sum(axis=1)
    jac = np.full((k, k), np.nan)
    for d in range(k):
        for e in range(k):
            union = np.logical_or(win_matrix[:, d], win_matrix[:, e]).sum()
            inter = np.logical_and(win_matrix[:, d], win_matrix[:, e]).sum()
            jac[d, e] = inter / union if union > 0 else np.nan
    iu = np.triu_indices(k, k=1)
    pair_vals = jac[iu]
    mean_jac = float(np.nanmean(pair_vals)) if np.isfinite(pair_vals).any() else float("nan")
    core_idx = np.where(counts >= core_min)[0]
    return {
        "freqs": freqs,
        "counts": counts,
        "jaccard_matrix": jac,
        "mean_pairwise_jaccard": mean_jac,
        "core_idx": core_idx,
        "core_size": int(core_idx.size),
    }


def exchangeable_null(
    win_counts: np.ndarray, n_cells: int, core_min: int, n_rep: int, seed: int
) -> dict:
    """Permutation null: per draw, win COUNT fixed, identities uniform over cells.

    Vectorized (one boolean mask per draw × replicate; no per-replicate Python
    loop). Returns null distributions of the core-set size and of the mean
    pairwise Jaccard, plus the analytic per-cell expected win count.
    """
    rng = np.random.default_rng(seed)
    k = len(win_counts)
    masks = np.zeros((k, n_rep, n_cells), dtype=bool)
    for d, w in enumerate(win_counts):
        w = int(w)
        if w == 0:
            continue
        r = rng.random((n_rep, n_cells))
        thresh = np.partition(r, w - 1, axis=1)[:, w - 1 : w]
        masks[d] = r <= thresh  # exactly w wins per replicate (ties measure-zero)
    counts = masks.sum(axis=0)  # (n_rep, n_cells)
    core_sizes = (counts >= core_min).sum(axis=1)
    m = masks.astype(np.float32)
    inter = np.einsum("drc,erc->der", m, m)  # (k, k, n_rep)
    w_arr = win_counts.astype(np.float64)
    union = w_arr[:, None, None] + w_arr[None, :, None] - inter
    iu = np.triu_indices(k, k=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac_pairs = np.where(union[iu] > 0, inter[iu] / union[iu], np.nan)
    all_nan = ~np.isfinite(jac_pairs).any(axis=0)
    mean_jacs = np.where(all_nan, np.nan, np.nanmean(jac_pairs, axis=0))
    return {
        "core_sizes": core_sizes,
        "mean_jaccards": mean_jacs,
        "expected_cell_win_count": float(w_arr.sum() / n_cells),
    }


def p_value_ge(null: np.ndarray, obs: float) -> float:
    """One-sided permutation p (P[null >= obs]), add-one smoothed, NaN-guarded."""
    finite = null[np.isfinite(null)]
    if not np.isfinite(obs) or finite.size == 0:
        return float("nan")
    return float((1 + int((finite >= obs).sum())) / (1 + finite.size))


def parent_win_set() -> set[str]:
    """The parent anchor's winning primary ridge cells from stage1_benchmark.json."""
    with open(PARENT_BENCHMARK) as f:
        b = json.load(f)
    wins = set()
    for trait, tr in b["traits"].items():
        for src, sd in tr["schemes"]["primary"]["sources"].items():
            d = sd["deltas"]["ridge"]
            for mode in B1.MODES:
                dr, di = d["vs_raw_source"][mode], d["vs_id_transport"][mode]
                if (
                    dr["excludes_zero"]
                    and dr["delta"] > 0
                    and di["excludes_zero"]
                    and di["delta"] > 0
                ):
                    wins.add(f"{trait}|primary|{int(src)}|{mode}")
    return wins


def parent_fidelity_cosines() -> dict[str, float]:
    """Parent ridge transport-fidelity cosine per (trait, primary, src)."""
    if not PARENT_FIDELITY.exists():
        return {}
    with open(PARENT_FIDELITY) as f:
        fid = json.load(f)
    out = {}
    for trait, tr in fid.get("traits", {}).items():
        for src, v in tr.get("primary", {}).get("ridge", {}).items():
            out[f"{trait}|{src}"] = v.get("cosine_hhat_vs_true")
    return out


# ── figures ───────────────────────────────────────────────────────────────────────


def _cell_label(rec: dict) -> str:
    mode = "many-shot" if rec["mode"] == "many_shot" else "system"
    return f"{rec['trait']} L{rec['source']}→L{rec['target']} {mode}"


def make_figures(result: dict, fig_root: Path, fig_subdir: str) -> None:
    """Per-cell win-frequency panel (low-level) + core-vs-null summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    cells = result["cells"]
    k = result["config"]["k_draws"]
    win = np.array([c["wins_by_draw"] for c in cells], dtype=bool)
    freqs = np.array([c["win_frequency"] for c in cells])
    parent = np.array([c["parent_anchor_win"] for c in cells], dtype=bool)
    order = np.lexsort((np.arange(len(cells)), -freqs))
    labels = [_cell_label(cells[i]) for i in order]
    expected = result["null"]["expected_cell_win_count"] / k

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(9.5, max(6.0, 0.16 * len(cells))), width_ratios=[1.1, 1.0], sharey=True
    )
    ax0.imshow(
        win[order].astype(float), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="none"
    )
    ax0.set_xlabel("fit draw")
    ax0.set_xticks(range(k), [str(i) for i in range(k)])
    ax0.set_yticks(range(len(cells)), labels, fontsize=5.5)
    ax0.set_title("per-draw win (dark = win)")
    colors = [
        paper_palette_role("accent") if parent[i] else paper_palette_role("neutral") for i in order
    ]
    ax1.barh(range(len(cells)), freqs[order], color=colors, height=0.8)
    ax1.axvline(expected, color=paper_palette_role("baseline"), ls="--", lw=1.0)
    ax1.set_xlabel(f"win frequency over {k} draws")
    ax1.set_xlim(0, 1)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=paper_palette_role("accent")),
        plt.Rectangle((0, 0), 1, 1, color=paper_palette_role("neutral")),
        plt.Line2D([0], [0], color=paper_palette_role("baseline"), ls="--"),
    ]
    ax1.legend(
        handles,
        ["parent anchor win", "other cell", "exchangeable-null expectation"],
        loc="lower right",
        fontsize=6,
    )
    fig.suptitle("Transport win-cell stability across disjoint fit draws", y=0.995)
    fig.tight_layout()
    savefig_paper(fig, f"{fig_subdir}/wincell_win_frequency", dir=fig_root, embed_data=False)
    plt.close(fig)

    null = result["null"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    core_null = np.asarray(null["core_size_null_sample"])
    bins = np.arange(-0.5, max(int(core_null.max()), result["core"]["size"]) + 1.5)
    ax0.hist(core_null, bins=bins, color=paper_palette_role("neutral"), density=True)
    ax0.axvline(result["core"]["size"], color=paper_palette_role("accent"), lw=1.5)
    ax0.set_xlabel(f"core-set size (cells winning ≥{result['config']['core_min_wins']}/{k})")
    ax0.set_ylabel("null density")
    ax0.set_title(f"observed {result['core']['size']}, p={result['core']['p_value']:.2g}")
    jac_null = np.asarray(null["mean_jaccard_null_sample"], dtype=float)
    jac_null = jac_null[np.isfinite(jac_null)]
    ax1.hist(jac_null, bins=40, color=paper_palette_role("neutral"), density=True)
    ax1.axvline(result["jaccard"]["mean_pairwise"], color=paper_palette_role("accent"), lw=1.5)
    ax1.set_xlabel("mean pairwise Jaccard of win-sets")
    ax1.set_title(
        f"observed {result['jaccard']['mean_pairwise']:.3f}, p={result['jaccard']['p_value']:.2g}"
    )
    fig.suptitle("Win-set stability vs exchangeable null (win counts fixed per draw)", y=1.02)
    fig.tight_layout()
    savefig_paper(fig, f"{fig_subdir}/wincell_core_vs_null", dir=fig_root, embed_data=True)
    plt.close(fig)
    logger.info("[figures] wrote %s/%s/*", fig_root, fig_subdir)


# ── synthetic wiring-test inputs ──────────────────────────────────────────────────


def synthetic_inputs(hidden: int, pool_rows: int) -> dict:
    """Tiny fabricated pool + 1-trait eval frame (no HF, CPU-instant)."""
    rng = np.random.default_rng(0)
    n_q, n_layers = 60, C.EXPECTED_LAYERS
    traj = rng.standard_normal((n_q, n_layers, hidden)).astype(np.float32)
    mat = {
        "traj": traj,
        "y": rng.standard_normal(n_q).astype(np.float64) * 20 + 50,
        "cond": rng.integers(0, 6, size=n_q),
        "mode": np.array(["system" if i % 2 else "many_shot" for i in range(n_q)], dtype=object),
        "cond_ids": [f"c{i}" for i in range(6)],
        "layers": list(range(n_layers)),
    }
    r_b = rng.standard_normal((n_layers, hidden)).astype(np.float32)
    pool = torch.from_numpy(
        rng.standard_normal((pool_rows, n_layers, hidden)).astype(np.float32)
    ).to(torch.bfloat16)
    tgt = 20
    return {
        "trait_inputs": {"evil": (mat, r_b)},
        "schemes": {"primary": tgt},
        "sources": sorted({max(0, tgt - 4), tgt - 1}),
        "pool": pool,
    }


# ── main ─────────────────────────────────────────────────────────────────────────


def fit_needed_draws(
    needed: list[int],
    done: dict,
    idx: np.ndarray,
    args,
    regime: dict,
    transitions: list[int],
    eval_inputs: dict,
    synth: dict | None,
    draw_path,
) -> None:
    """Extract the needed draws' rows, fit each draw's maps, checkpoint per draw.

    Mutates ``done`` in place. The per-draw JSON checkpoint (records + regime +
    idx sha) is written the moment the draw completes, so a crash forfeits at
    most one in-flight draw and a parallel ``--only-draws`` worker's finished
    draws are never refit.
    """
    if synth is not None:
        pool = synth["pool"]
        draws = {k: pool[torch.from_numpy(idx[k])] for k in needed}
    else:
        draws = load_pool_draws(
            idx, needed, args.capture_dir, args.scratch_dir, args.pool_limit or None
        )
    parity = None
    for j, k in enumerate(needed):
        t0 = time.time()
        if j == 0 and not args.no_parity_check:
            parity = parity_spot_check(draws[k], transitions[len(transitions) // 2], args.device)
        maps = fit_draw_maps(draws[k], transitions, args.device, args.solver)
        records = win_records_for_draw(maps, eval_inputs, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED)
        payload = {
            "draw": k,
            "regime": regime,
            "idx_sha256": _sha256_ints(idx[k]),
            "parity_spot_check": parity if j == 0 else None,
            "wall_seconds": time.time() - t0,
            "records": records,
        }
        C.write_json_atomic(draw_path(k), payload)
        done[k] = payload
        del draws[k], maps
        gc.collect()
        logger.info(
            "[draw %d/%d] %d records, %d wins, wall %.1fs",
            k + 1,
            args.k_draws,
            len(records),
            sum(r["win"] for r in records),
            payload["wall_seconds"],
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 wincell-stability free analysis.")
    ap.add_argument("--k-draws", type=int, default=K_DRAWS)
    ap.add_argument("--draw-n", type=int, default=DRAW_N)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--solver", choices=("primal", "dual"), default="primal")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--fig-root", type=Path, default=FIG_ROOT)
    ap.add_argument("--fig-subdir", default=FIG_SUBDIR)
    ap.add_argument("--capture-dir", type=Path, default=S.CAPTURE_DIR)
    ap.add_argument("--scratch-dir", type=Path, default=SCRATCH_DEFAULT)
    ap.add_argument("--pool-limit", type=int, default=0, help="0 = whole pool (smoke: 1st shard)")
    ap.add_argument("--no-parity-check", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--only-draws",
        default="",
        help="comma-list of draw indices to FIT in this process (parallel workers); "
        "aggregation runs only when every draw is checkpointed",
    )
    ap.add_argument("--smoke", action="store_true", help="K=2 x n=500, evil, coarse grid, 1 shard")
    ap.add_argument("--synthetic", action="store_true", help="no-HF CPU wiring test")
    ap.add_argument("--synthetic-hidden", type=int, default=64)
    args = ap.parse_args()

    smoke = args.smoke
    if smoke or args.synthetic:
        args.k_draws = min(args.k_draws, 2)
        args.draw_n = min(args.draw_n, 500)
        args.n_boot = min(args.n_boot, 200)
        args.n_null = min(args.n_null, 1000)
        args.traits = ["evil"]
        if args.out_dir == EVAL_DIR:  # never overwrite committed paths from a smoke
            tag = "synthetic" if args.synthetic else "smoke"
            args.out_dir = Path(f"/tmp/issue-841-wincell-{tag}/eval")
            args.fig_root = Path(f"/tmp/issue-841-wincell-{tag}/figures")
        if smoke and args.pool_limit == 0:
            args.pool_limit = S.SHARD_ROWS  # first shard only
    core_min = int(np.ceil(CORE_FRAC * args.k_draws))
    logger.info(
        "k_draws=%d draw_n=%d n_boot=%d n_null=%d solver=%s traits=%s core_min=%d out=%s",
        args.k_draws,
        args.draw_n,
        args.n_boot,
        args.n_null,
        args.solver,
        args.traits,
        core_min,
        args.out_dir,
    )

    synth = (
        synthetic_inputs(args.synthetic_hidden, args.k_draws * args.draw_n + 7)
        if args.synthetic
        else None
    )
    eval_inputs = load_eval_inputs(args.traits, smoke or args.synthetic, synth)
    transitions = (
        sorted(
            {
                t
                for tgt in [synth["schemes"]["primary"]]
                for s in synth["sources"]
                for t in range(s, tgt)
            }
        )
        if synth is not None
        else needed_transitions_primary(args.traits, smoke)
    )
    logger.info("[plan] %d transitions to fit per draw: %s", len(transitions), transitions)

    pool_rows = (
        int(synth["pool"].shape[0]) if synth is not None else int(_fetch_manifest()["total_rows"])
    )
    if args.pool_limit:
        pool_rows = min(pool_rows, args.pool_limit)
    idx = draw_indices(pool_rows, args.k_draws, args.draw_n, PERM_SEED)
    regime = {
        "k_draws": args.k_draws,
        "draw_n": args.draw_n,
        "n_boot": args.n_boot,
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "perm_seed": PERM_SEED,
        "solver": args.solver,
        "transitions": transitions,
        "pool_rows": pool_rows,
        "traits": list(args.traits),
        "synthetic": bool(args.synthetic),
        "smoke": bool(smoke),
    }

    draws_dir = args.out_dir / "draws"
    draws_dir.mkdir(parents=True, exist_ok=True)

    def _draw_path(k: int) -> Path:
        return draws_dir / f"draw_{k:02d}.json"

    def _completed(k: int) -> dict | None:
        p = _draw_path(k)
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        if d.get("regime") != regime or d.get("idx_sha256") != _sha256_ints(idx[k]):
            raise RuntimeError(
                f"stale draw checkpoint {p}: regime/idx mismatch — delete it to refit"
            )
        return d

    done = {k: d for k in range(args.k_draws) if (d := _completed(k)) is not None}
    needed = [k for k in range(args.k_draws) if k not in done]
    if args.only_draws:
        only = {int(x) for x in args.only_draws.split(",") if x}
        needed = [k for k in needed if k in only]
    logger.info(
        "[resume] %d/%d draws already checkpointed; fitting %s", len(done), args.k_draws, needed
    )

    if needed:
        fit_needed_draws(
            needed, done, idx, args, regime, transitions, eval_inputs, synth, _draw_path
        )

    missing = [k for k in range(args.k_draws) if k not in done]
    if missing:
        logger.info(
            "[partial] draws %s not yet checkpointed — rerun without --only-draws to aggregate",
            missing,
        )
        return 0

    result = aggregate_result(done, idx, args, regime, core_min, smoke)
    C.write_json_atomic(args.out_dir / "wincell_stability.json", result)
    logger.info(
        "[done] core %d/%d cells (>=%d/%d wins, p=%.3g); mean Jaccard %.3f (null %.3f, p=%.3g); "
        "per-draw wins %s → %s",
        result["core"]["size"],
        len(result["cells"]),
        core_min,
        args.k_draws,
        result["core"]["p_value"],
        result["jaccard"]["mean_pairwise"],
        result["jaccard"]["null_mean"],
        result["jaccard"]["p_value"],
        result["per_draw_win_counts"],
        args.out_dir / "wincell_stability.json",
    )
    if not args.no_figures:
        make_figures(result, args.fig_root, args.fig_subdir)
    return 0


def aggregate_result(
    done: dict, idx: np.ndarray, args, regime: dict, core_min: int, smoke: bool
) -> dict:
    """Aggregate the per-draw checkpoints: frequencies, Jaccard, core set, null,
    and the comparison against the parent anchor's win identities."""
    cell_ids = [r["cell"] for r in done[0]["records"]]
    for k in range(args.k_draws):
        assert [r["cell"] for r in done[k]["records"]] == cell_ids, f"cell set differs in draw {k}"
    n_cells = len(cell_ids)
    win_matrix = np.array(
        [[done[k]["records"][i]["win"] for k in range(args.k_draws)] for i in range(n_cells)],
        dtype=bool,
    )
    obs = jaccard_and_core(win_matrix, core_min)
    win_counts_per_draw = win_matrix.sum(axis=0)
    null = exchangeable_null(win_counts_per_draw, n_cells, core_min, args.n_null, NULL_SEED)
    p_core = p_value_ge(null["core_sizes"], obs["core_size"])
    p_jac = p_value_ge(null["mean_jaccards"], obs["mean_pairwise_jaccard"])

    pwins = parent_win_set() if not args.synthetic else set()
    pfid = parent_fidelity_cosines() if not args.synthetic else {}
    cells_out = []
    for i, cid in enumerate(cell_ids):
        base = done[0]["records"][i]
        fid_by_draw = [done[k]["records"][i]["fidelity_cosine"] for k in range(args.k_draws)]
        cells_out.append(
            {
                "cell": cid,
                "trait": base["trait"],
                "source": base["source"],
                "target": base["target"],
                "mode": base["mode"],
                "wins_by_draw": [bool(w) for w in win_matrix[i]],
                "win_count": int(win_matrix[i].sum()),
                "win_frequency": float(win_matrix[i].mean()),
                "in_core": bool(win_matrix[i].sum() >= core_min),
                "parent_anchor_win": cid in pwins,
                "mean_fidelity_cosine_draws": float(np.mean(fid_by_draw)),
                "parent_fidelity_cosine": pfid.get(f"{base['trait']}|{base['source']}"),
            }
        )
    core_cells = [c for c in cells_out if c["in_core"]]
    parent_freqs = [c["win_frequency"] for c in cells_out if c["parent_anchor_win"]]
    other_freqs = [c["win_frequency"] for c in cells_out if not c["parent_anchor_win"]]
    result = {
        "config": {
            **regime,
            "core_min_wins": core_min,
            "n_null": args.n_null,
            "null_seed": NULL_SEED,
            "capture_bucket": S.HF_CAPTURE_BUCKET,
            "draw_idx_sha256": {str(k): _sha256_ints(idx[k]) for k in range(args.k_draws)},
            "note": (
                "draws are pairwise disjoint blocks of one seeded permutation of the "
                "96k NEW-context capture pool; the parent anchor's fit rows are pass_b "
                "rows, absent from this pool by construction (zero overlap)."
            ),
        },
        "per_draw_win_counts": [int(w) for w in win_counts_per_draw],
        "cells": cells_out,
        "jaccard": {
            "matrix": [
                [None if not np.isfinite(v) else float(v) for v in row]
                for row in obs["jaccard_matrix"]
            ],
            "mean_pairwise": obs["mean_pairwise_jaccard"],
            "null_mean": float(np.nanmean(null["mean_jaccards"])),
            "p_value": p_jac,
        },
        "core": {
            "min_wins": core_min,
            "size": obs["core_size"],
            "cells": [c["cell"] for c in core_cells],
            "null_mean": float(np.mean(null["core_sizes"])),
            "null_p99": float(np.percentile(null["core_sizes"], 99)),
            "p_value": p_core,
        },
        "null": {
            "n_replicates": args.n_null,
            "expected_cell_win_count": null["expected_cell_win_count"],
            "core_size_null_sample": [int(v) for v in null["core_sizes"][:2000]],
            "mean_jaccard_null_sample": [
                None if not np.isfinite(v) else float(v) for v in null["mean_jaccards"][:2000]
            ],
        },
        "parent_comparison": {
            "parent_win_cells": sorted(pwins),
            "n_parent_wins": len(pwins),
            "core_overlap_with_parent": sorted({c["cell"] for c in core_cells} & pwins),
            "mean_win_freq_parent_winners": (
                float(np.mean(parent_freqs)) if parent_freqs else float("nan")
            ),
            "mean_win_freq_other_cells": (
                float(np.mean(other_freqs)) if other_freqs else float("nan")
            ),
        },
        "metadata": C.reproducibility_metadata(
            {"phase": "wincell_stability", "smoke": smoke, "synthetic": args.synthetic}
        ),
    }
    return result


if __name__ == "__main__":
    sys.exit(main())
