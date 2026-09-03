#!/usr/bin/env python3
"""Issue #1901 inline round ``xlayer-grid`` — the full 28 x 28 cross-layer ridge grid.

Fits ridge maps from the last-context-token state at layer l_C to the
DRAW-AVERAGED answer mean state at layer l_A for every (l_C, l_A) pair on the
banked #779/#1901 n50k store (train 50,000 / val 400 / pinned test 1,000), and
reads whether the grid maximum sits on the diagonal (the paper's same-layer
pairing claim). Per cell: held-out variance-weighted R^2 (``PR._pooled_r2``,
SS_tot on the test set's own mean), top-1 retrieval (whitened cosine + CSLS
primary, euclidean companion) and the identity+bias baseline. Headline: paired
bootstrap (fixed seed) over the 1,000 pinned test rows for best-diagonal minus
best-off-diagonal (R^2 and top-1), plus a per-answer-layer
diagonal-minus-best-other-context-layer secondary table.

Reuse map (no new estimator code): the data path and the diagonal recipe are
``issue1901_avgtarget_plots.phase_plot1`` verbatim (``PD.stage_prefix`` +
``N1G._load_pass_b_bundle`` + ``PD._extract_all_layers`` + ``N50.build_n50k_split``
+ ``_avg_targets`` + ``P1R.train_whitening_stats`` + the plot1 whitened-CSLS /
euclidean scorers). The ridge core is ``issue779_ffc_n1m_fits`` — the X-side
factorization (``_train_standardizer`` x-stats + streamed X^TX + one eigh) is
computed ONCE per context layer and reused across the 28 answer layers; per-cell
predictions and the val-lambda selection call ``N1M._ridge_predict_one`` with the
verbatim ``N1M.fit_ridge`` selection loop, so the (l, l) diagonal cell reproduces
the plot1 recipe exactly (parity-gated against the banked
``plot1_avg.json per_layer[l].arms.ridge.avg.whole_map_r2``, tol 1e-3).

Refusal-safety: never prints conversation/rollout text — only counts, indices,
digests, metric values.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# vLLM V1 fork-safety (#628): spawn BEFORE any vllm import in this process tree
# (transitively imported script modules defer their vllm imports; keep parity).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps bind BEFORE numpy/torch import (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1491_ladder_manifest as LMAN  # noqa: E402
import issue1901_avgtarget_plots as AVG  # noqa: E402
import issue1901_paper_densify as PD  # noqa: E402
import issue1901_plot1_remake as P1R  # noqa: E402
from issue1901_metric_battery import K_CSLS, csls_scores  # noqa: E402

from explore_persona_space.analysis import mapping_baselines as MB  # noqa: E402
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1901_xlayer_grid")

OUT_EVAL_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_1901" / "xlayer_grid"
BANKED_PLOT1_AVG = PROJECT_ROOT / "eval_results/issue_1901/avgtarget_plots/plot1_avg.json"
HF_RELAY_PREFIX = "issue1901_xlayer_grid"
DRAWS_HF_PREFIX = f"{AVG.HF_PREFIX}/draws"  # issue1901_avgtarget/draws (banked draw stores)
LAYERS_ALL = tuple(range(PD.N_LAYERS))
H_DIM = PD.H_DIM
N_TEST = N50.N_TEST  # 1000
BOOT_SEED = 20260903  # decision record: fixed paired-bootstrap seed
SCRIPT_VERSION = "xlayer-grid-v1"
KNN_KS = PD.KNN_KS  # (1, 5, 10)


# ── small shared helpers ─────────────────────────────────────────────────────────


def _meta() -> dict:
    md = as_metadata_dict(git_provenance(argv0=__file__))
    md.update(
        {
            "script": "issue1901_xlayer_grid",
            "issue": 1901,
            "round": "xlayer-grid",
            "script_version": SCRIPT_VERSION,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return md


def _write_json_atomic(path: Path, obj: dict) -> None:
    with atomic_replace(path, logger=logger) as tmp:
        tmp.write_text(json.dumps(obj, indent=1, default=str))


def _write_npz_atomic(path: Path, **arrays) -> None:
    with atomic_replace(path, logger=logger) as tmp:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)


def _resolve_revision(revision: str | None) -> str:
    """Pin the data-repo listing+downloads to ONE Hub commit (main -> sha once
    per run; an unpinned resume can mix file generations, #1901 C1)."""
    if revision:
        return revision
    from huggingface_hub import HfApi

    info = hub.retry_transient(
        lambda: HfApi().repo_info(C.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info {C.HF_DATA_REPO}",
    )
    return str(info.sha)


def _upload_eval(args, *, force: bool = False) -> None:
    if args.skip_upload:
        return
    _upload_eval.counter = getattr(_upload_eval, "counter", 0) + 1
    if force or _upload_eval.counter % args.upload_every == 0:
        url = hub._upload(
            args.out_eval,
            C.HF_DATA_REPO,
            "dataset",
            path_in_repo=f"{HF_RELAY_PREFIX}/eval",
            raise_on_error=True,
        )
        logger.info("[upload] eval dir -> %s", url)


def _load_draws(draws_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(D (4, 1000, 28, H) fp32 with NaN gaps, present (4, 1000) bool) — the
    banked #1901 avgtarget draw stores, manifest order (AVG._load_draws verbatim,
    directory-parameterized)."""
    Ds, Ps = [], []
    for seed in AVG.SEEDS:
        b = torch.load(draws_dir / f"draws_seed{seed}.pt", map_location="cpu", weights_only=False)
        assert b["prompts_sha"] == LMAN.TEST_1000_PROMPT_SHA and b["layers"] == list(LAYERS_ALL)
        Ds.append(b["V"].to(torch.float32).numpy())
        Ps.append(np.asarray(b["present"], dtype=bool))
    return np.stack(Ds), np.stack(Ps)


def _extract_all_layers_deleting(capture_dir: Path, max_chunks: int | None, *, delete: bool):
    """``PD._extract_all_layers`` with per-chunk delete-after-copy (peak disk
    stays near the store size; the compute statement's extraction contract).
    Same two-pass shape, same asserts, same dtype handling."""
    chunk_files = sorted(capture_dir.glob("shard*_chunk*.pt"))
    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]
    if not chunk_files:
        raise FileNotFoundError(f"no capture chunks under {capture_dir}")
    rows = []
    layers = None
    dtype = None
    for cp in chunk_files:
        b = F._mmap_load(cp)
        t = b["cx_last"]
        assert t.shape[1:] == (PD.N_LAYERS, H_DIM), (cp.name, tuple(t.shape))
        assert b["v_x"].shape == t.shape, (cp.name, tuple(b["v_x"].shape))
        if layers is None:
            layers = [int(x) for x in b["layers"]]
            dtype = str(t.dtype).replace("torch.", "")
            logger.info("[extract] capture dtype=%s layers=%s (%s)", dtype, layers[:4], cp.name)
        else:
            assert [int(x) for x in b["layers"]] == layers, cp.name
            assert str(t.dtype).replace("torch.", "") == dtype, (cp.name, t.dtype)
        rows.append(int(t.shape[0]))
        del b
    n_new = int(sum(rows))
    np_dtype = np.dtype(dtype.replace("bfloat16", "float32"))  # bf16 has no numpy dtype
    X_all = np.empty((n_new, PD.N_LAYERS, H_DIM), dtype=np_dtype)
    Y_all = np.empty((n_new, PD.N_LAYERS, H_DIM), dtype=np_dtype)
    off = 0
    t0 = time.time()
    for i, cp in enumerate(chunk_files):
        b = F._mmap_load(cp)
        k = rows[i]
        X_all[off : off + k] = b["cx_last"].to(getattr(torch, str(np_dtype))).numpy()
        Y_all[off : off + k] = b["v_x"].to(getattr(torch, str(np_dtype))).numpy()
        off += k
        del b
        if delete:
            cp.unlink()  # consumed; bound peak disk (re-stageable from the pinned revision)
        if (i + 1) % 10 == 0 or i + 1 == len(chunk_files):
            logger.info(
                "[extract] %d/%d chunks (%d rows, %.0fs)%s",
                i + 1,
                len(chunk_files),
                off,
                time.time() - t0,
                " [deleting]" if delete else "",
            )
    assert off == n_new
    return X_all, Y_all, layers, dtype


# ── ridge: X-side factorization shared across answer layers ─────────────────────
#
# ``N1M._ridge_factorize`` fuses the X-side eigh with ONE Y's cross-product. The
# grid reuses the X-side per context layer, so the two halves are split here with
# the SAME block partition, per-block operands and fp64 accumulation order as the
# fused original — the assembled ``fac`` dict feeds the verbatim
# ``N1M._ridge_predict_one``, and the lambda-selection loop below is copied from
# ``N1M.fit_ridge``, so a diagonal cell reproduces ``fit_ridge`` numerically
# (gate: tol 1e-3 vs the banked plot1 cells).


def _xside_factorize(X: np.ndarray, tr: np.ndarray, dev: torch.device, block: int) -> dict:
    """Train x-standardizer + streamed X^TX + one eigh ({U, s_eig, xmu, xsd})."""
    H = X.shape[1]
    sum_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sumsq_x = torch.zeros(H, dtype=torch.float64, device=dev)
    n = 0
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = torch.as_tensor(X[idx], dtype=torch.float64, device=dev)
        sum_x += xb.sum(0)
        sumsq_x += (xb * xb).sum(0)
        n += len(idx)
    xmu = sum_x / n
    denom = max(1, n - 1)  # unbiased, the _train_standardizer convention
    var = (sumsq_x - n * xmu * xmu) / denom
    xsd = torch.clamp(var, min=0.0).sqrt() + 1e-9
    A = torch.zeros((H, H), dtype=torch.float64, device=dev)
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = (torch.as_tensor(X[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        A += xb.T @ xb
    s_eig, U = torch.linalg.eigh(A)
    s_eig = torch.clamp(s_eig, min=0.0)
    return {"U": U, "s_eig": s_eig, "xmu": xmu, "xsd": xsd}


def _fit_ridge_from_xfac(
    xfac: dict,
    X: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    lambdas: np.ndarray,
    dev: torch.device,
    block: int,
) -> tuple[np.ndarray, dict]:
    """Per-cell ridge off a shared X-side factorization; ``N1M.fit_ridge``'s
    val-lambda selection verbatim; predictions via ``N1M._ridge_predict_one``."""
    xmu, xsd = xfac["xmu"], xfac["xsd"]
    sum_y = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
    n = 0
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        sum_y += torch.as_tensor(Y[idx], dtype=torch.float64, device=dev).sum(0)
        n += len(idx)
    ymu = sum_y / n
    XtY = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.float64, device=dev)
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = (torch.as_tensor(X[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev) - ymu
        XtY += xb.T @ yb
    fac = {
        "U": xfac["U"],
        "s_eig": xfac["s_eig"],
        "UtXtY": xfac["U"].T @ XtY,
        "xmu": xmu,
        "xsd": xsd,
        "ymu": ymu,
    }
    best_lam, best_vr2 = float(lambdas[0]), -np.inf  # fit_ridge selection, verbatim
    for lam in lambdas:
        vr2 = PR._pooled_r2(N1M._ridge_predict_one(X, val, fac, lam, dev, block), Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    pred_te = N1M._ridge_predict_one(X, te, fac, best_lam, dev, block)
    meta = {
        "n_train": len(tr),
        "selection": "val-lambda (primal, streaming, shared x-side eigh)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "ridge_block": int(block),
    }
    return pred_te, meta, ymu


# ── per-cell scoring (the plot1 whitened-CSLS / euclidean conventions) ──────────


def _score_cell(
    pred: np.ndarray, y_avg: np.ndarray, mu: np.ndarray, ell: np.ndarray, zp: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """(metrics, res_i fp64 (n,), hits_wcsls u8 (n,), hits_euclid u8 (n,))."""
    n = y_avg.shape[0]
    assert K_CSLS < n, f"pool too small for CSLS: {n} <= K={K_CSLS}"
    r2 = PR._pooled_r2(pred, y_avg)
    res_i = ((np.asarray(y_avg, np.float64) - np.asarray(pred, np.float64)) ** 2).sum(axis=1)
    zq = P1R.whiten(pred, mu, ell)
    s_wcos = P1R.cos_sim(zq, zp)
    ranks_w = P1R.midranks(-csls_scores(s_wcos, K_CSLS), np.arange(n))
    ranks_e = PD._rank_vector(pred, y_avg, "euclidean")
    metrics = {
        "whole_map_r2": float(r2),
        "retrieval": {
            "whiten_csls": {
                "acc_at_k": {int(k): float((ranks_w <= k).mean()) for k in KNN_KS},
                "median_rank": float(np.median(ranks_w)),
                "mrr": float((1.0 / ranks_w).mean()),
            },
            "euclidean": {
                "acc_at_k": {int(k): float((ranks_e <= k).mean()) for k in KNN_KS},
                "median_rank": float(np.median(ranks_e)),
                "mrr": float((1.0 / ranks_e).mean()),
            },
        },
        "n_pool": int(n),
        "chance_at_1": float(1.0 / n),
    }
    return metrics, res_i, (ranks_w <= 1).astype(np.uint8), (ranks_e <= 1).astype(np.uint8)


# ── bootstrap (paired, fixed seed, vectorized via draw-multiplicity counts) ─────


def _boot_counts(n: int, n_draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(boot_idx (n_draws, n) int64, counts (n_draws, n) fp64 multiplicities)."""
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_draws, n))
    counts = np.zeros((n_draws, n), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n_draws), n), boot_idx.ravel()), 1.0)
    return boot_idx, counts


def _sst_draws_for_target(y_avg: np.ndarray, counts: np.ndarray, boot_idx: np.ndarray):
    """Per-draw SS_tot of the resampled target (mean recomputed per draw — the
    ``F._bootstrap_recon_ci`` convention), vectorized as one GEMM; draw 0 is
    asserted against the direct formula."""
    y64 = np.asarray(y_avg, np.float64)
    n = y64.shape[0]
    rownorm2 = (y64**2).sum(axis=1)
    ybar = (counts @ y64) / n
    sst = counts @ rownorm2 - n * (ybar**2).sum(axis=1)
    t = y64[boot_idx[0]]
    direct = float(((t - t.mean(0)) ** 2).sum())
    assert abs(direct - float(sst[0])) <= 1e-6 * max(1.0, abs(direct)), (direct, float(sst[0]))
    sst_point = float(((y64 - y64.mean(0)) ** 2).sum())
    return sst_point, sst


def _ci(draws: np.ndarray) -> dict:
    return {
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "mean": float(draws.mean()),
    }


def _bootstrap_contrasts(
    cells: list[tuple[int, int]],
    point: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    n_draws: int,
) -> dict:
    """Headline + secondary contrasts. ``point[metric]`` is (n_cells,);
    ``draws[metric]`` is (n_cells, n_draws). Paired: one draw set for all cells."""
    diag = np.array([i for i, (c, a) in enumerate(cells) if c == a])
    offd = np.array([i for i, (c, a) in enumerate(cells) if c != a])
    if len(diag) == 0 or len(offd) == 0:
        raise ValueError(
            f"bootstrap contrasts need >=1 diagonal and >=1 off-diagonal cell; "
            f"got {len(diag)} diagonal / {len(offd)} off-diagonal"
        )
    out: dict[str, dict] = {}
    for metric, pt in point.items():
        dr = draws[metric]
        bd = diag[int(np.argmax(pt[diag]))]
        bo = offd[int(np.argmax(pt[offd]))]
        fixed = dr[bd] - dr[bo]
        select = dr[diag].max(axis=0) - dr[offd].max(axis=0)
        out[metric] = {
            "best_diag_cell": {"layer_c": cells[bd][0], "layer_a": cells[bd][1]},
            "best_offdiag_cell": {"layer_c": cells[bo][0], "layer_a": cells[bo][1]},
            "point_diff": float(pt[bd] - pt[bo]),
            "fixed_cells_ci": _ci(fixed),
            "selection_aware_ci": _ci(select),
            "frac_draws_diff_leq_0_fixed": float((fixed <= 0).mean()),
            "frac_draws_diff_leq_0_selection_aware": float((select <= 0).mean()),
        }
        per_answer = []
        for i in diag:
            l_a = cells[i][1]
            others = np.array([j for j, (c, a) in enumerate(cells) if a == l_a and c != l_a])
            if len(others) == 0:
                continue
            bo_a = others[int(np.argmax(pt[others]))]
            d = dr[i] - dr[bo_a]
            per_answer.append(
                {
                    "layer_a": int(l_a),
                    "diag_point": float(pt[i]),
                    "best_other_layer_c": int(cells[bo_a][0]),
                    "best_other_point": float(pt[bo_a]),
                    "point_diff": float(pt[i] - pt[bo_a]),
                    "ci": _ci(d),
                }
            )
        out[metric]["per_answer_layer_diag_minus_best_other"] = per_answer
    out["convention"] = {
        "n_draws": int(n_draws),
        "seed": BOOT_SEED,
        "paired": "one boot-index set over the 1,000 pinned test rows, shared by every cell",
        "r2_draw": "1 - sse_cell[idx].sum() / sst_target[idx] (per-draw own-mean SS_tot)",
        "fixed_cells": "best cells selected at the POINT estimate, then paired diff per draw",
        "selection_aware": "per-draw re-selected max(diag) - max(offdiag)",
    }
    return out


# ── unit keys, resume, assembly ─────────────────────────────────────────────────


def _run_key(args, train_sha: str) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "layers_a": [int(x) for x in args.layers_a],
        "n_train_target": 50_000,
        "train_sha256": train_sha,
        "seeds": list(AVG.SEEDS),
        "avg_convention": "full-pool draw-averaged (K=4)",
        "lambda_grid": ["logspace", -3, 7, 21],
        "max_chunks": args.max_chunks,
        "split_seed": int(args.seed),
    }


def _row_paths(args, l_c: int) -> tuple[Path, Path]:
    rows_dir = args.out_eval / "rows"
    return rows_dir / f"L{l_c}.json", rows_dir / f"L{l_c}.npz"


def _row_resume_hit(args, l_c: int, key: dict | None) -> dict | None:
    """Stored row dict when the checkpoint matches (key=None: match stored keys
    against argv-derivable fields only — the pre-staging full-skip scan)."""
    jp, npz = _row_paths(args, l_c)
    if not (jp.exists() and npz.exists()):
        return None
    row = json.loads(jp.read_text())
    got = row.get("unit_key", {})
    if key is not None:
        return row if got == key else None
    ref = {k: v for k, v in _run_key(args, train_sha="?").items() if k != "train_sha256"}
    return row if {k: v for k, v in got.items() if k != "train_sha256"} == ref else None


def _try_full_resume(args) -> dict | None:
    """All requested rows + split.json + bootstrap_inputs.npz present and
    mutually consistent -> return {rows, split} and skip staging entirely."""
    split_p = args.out_eval / "split.json"
    bi_p = args.out_eval / "bootstrap_inputs.npz"
    if not (args.resume and split_p.exists() and bi_p.exists()):
        return None
    split = json.loads(split_p.read_text())
    rows = {}
    for l_c in args.layers_c:
        row = _row_resume_hit(args, l_c, None)
        if row is None or row["unit_key"]["train_sha256"] != split["diag"]["train_sha256"]:
            return None
        rows[int(l_c)] = row
    with np.load(bi_p) as bi:
        if (
            int(bi["n_draws"]) != args.n_draws
            or int(bi["boot_seed"]) != BOOT_SEED
            or list(bi["layers_a"]) != [int(x) for x in args.layers_a]
        ):
            return None
    logger.info(
        "[resume] all %d rows + split + bootstrap inputs present; skipping staging", len(rows)
    )
    return {"rows": rows, "split": split}


def _assemble_outputs(args, rows: dict[int, dict], split: dict, timings: dict) -> None:
    """grid.json + per_row_sse.npz + bootstrap.json + run_meta.json (+ parity gate)."""
    layers_c = [int(x) for x in args.layers_c]
    layers_a = [int(x) for x in args.layers_a]
    smoke = args.max_chunks is not None

    parity_rows, parity_fail = [], []
    banked = None
    if not smoke:
        if not BANKED_PLOT1_AVG.exists():
            raise FileNotFoundError(f"parity target missing: {BANKED_PLOT1_AVG}")
        banked = json.loads(BANKED_PLOT1_AVG.read_text())["per_layer"]
    for l in sorted(set(layers_c) & set(layers_a)):
        cell = rows[l]["cells"][str(l)]
        if smoke:
            parity_rows.append({"layer": l, "skipped": True, "reason": "--max-chunks smoke"})
            continue
        want = banked[str(l)]["arms"]["ridge"]["avg"]["whole_map_r2"]
        d_r2 = abs(cell["ridge"]["whole_map_r2"] - want)
        ok = bool(d_r2 <= args.parity_tol)
        parity_rows.append(
            {
                "layer": l,
                "got_r2": cell["ridge"]["whole_map_r2"],
                "banked_r2": float(want),
                "d_r2": float(d_r2),
                "tol": args.parity_tol,
                "pass": ok,
            }
        )
        if not ok:
            parity_fail.append(l)

    grid = {
        "unit_key": rows[layers_c[0]]["unit_key"],
        "layers_c": layers_c,
        "layers_a": layers_a,
        "cells": {str(l_c): rows[l_c]["cells"] for l_c in layers_c},
        "parity_diag_vs_plot1_avg": parity_rows,
        "split": split["diag"],
        "identity_bias_impl": (
            "separable train-mean cache (bias = ymu - xmu, fp64); helper parity vs "
            "MB.identity_bias_predict asserted on the pilot diagonal cell each run"
        ),
        "metadata": _meta(),
    }
    _write_json_atomic(args.out_eval / "grid.json", grid)

    # stack per-row npz shards -> per_row_sse.npz + bootstrap
    cells: list[tuple[int, int]] = []
    sse_rows, hw_rows, he_rows = [], [], []
    for l_c in layers_c:
        with np.load(_row_paths(args, l_c)[1]) as z:
            assert list(z["layers_a"]) == layers_a, (l_c, list(z["layers_a"]))
            sse_rows.append(z["sse"])
            hw_rows.append(z["hits_wcsls"])
            he_rows.append(z["hits_euclid"])
        cells.extend((l_c, l_a) for l_a in layers_a)
    sse = np.concatenate(sse_rows)  # (n_cells, n_test) fp64
    hits_w = np.concatenate(hw_rows).astype(np.uint8)
    hits_e = np.concatenate(he_rows).astype(np.uint8)

    with np.load(args.out_eval / "bootstrap_inputs.npz") as bi:
        sst_point = bi["sst_point"]  # (n_a,)
        sst_draws = bi["sst_draws"]  # (n_a, n_draws)
    n = sse.shape[1]
    _, counts = _boot_counts(n, args.n_draws, BOOT_SEED)
    a_of_cell = np.array([layers_a.index(a) for _, a in cells])
    sse_draws = sse @ counts.T  # (n_cells, n_draws)
    r2_draws = 1.0 - sse_draws / sst_draws[a_of_cell]
    r2_point = 1.0 - sse.sum(axis=1) / sst_point[a_of_cell]
    acc_w_draws = (hits_w.astype(np.float64) @ counts.T) / n
    acc_e_draws = (hits_e.astype(np.float64) @ counts.T) / n
    contrasts = _bootstrap_contrasts(
        cells,
        point={
            "r2": r2_point,
            "acc1_wcsls": hits_w.mean(axis=1),
            "acc1_euclid": hits_e.mean(axis=1),
        },
        draws={"r2": r2_draws, "acc1_wcsls": acc_w_draws, "acc1_euclid": acc_e_draws},
        n_draws=args.n_draws,
    )
    _write_json_atomic(
        args.out_eval / "bootstrap.json", {"contrasts": contrasts, "metadata": _meta()}
    )
    _write_npz_atomic(
        args.out_eval / "per_row_sse.npz",
        sse=sse,
        hits_wcsls=hits_w,
        hits_euclid=hits_e,
        cells=np.array(cells, dtype=np.int64),
        layers_c=np.array(layers_c, dtype=np.int64),
        layers_a=np.array(layers_a, dtype=np.int64),
        sst_point=sst_point,
        sst_draws=sst_draws,
        boot_seed=np.int64(BOOT_SEED),
        n_draws=np.int64(args.n_draws),
    )
    run_meta = {
        "argv": sys.argv,
        "timings_s": timings,
        "store_revision": split.get("store_revision"),
        "staged_bytes": split.get("staged_bytes"),
        "device": args.device,
        "n_cells": len(cells),
        "parity_fail_layers": parity_fail,
        "metadata": _meta(),
    }
    _write_json_atomic(args.out_eval / "run_meta.json", run_meta)
    _upload_eval(args, force=True)
    if parity_fail:
        raise RuntimeError(
            f"PARITY GATE FAILED on diagonal layers {parity_fail} (tol {args.parity_tol}) "
            f"vs {BANKED_PLOT1_AVG} — grid.json written with parity rows; investigate"
        )


# ── main grid computation ───────────────────────────────────────────────────────


def run_grid(args) -> None:
    t_start = time.time()
    timings: dict[str, float] = {}
    dev = torch.device(args.device)
    layers_c = [int(x) for x in args.layers_c]
    layers_a = [int(x) for x in args.layers_a]
    (args.out_eval / "rows").mkdir(parents=True, exist_ok=True)

    full = _try_full_resume(args)
    if full is not None:
        _assemble_outputs(args, full["rows"], full["split"], {"full_resume": True})
        return

    revision = _resolve_revision(args.revision)
    logger.info("[stage] data-repo revision pinned: %s", revision)
    t0 = time.time()
    draws_dir = PD.stage_prefix(
        DRAWS_HF_PREFIX,
        args.stage_root,
        workers=args.stage_workers,
        revision=revision,
        only_files=tuple(f"draws_seed{s}.pt" for s in AVG.SEEDS),
    )
    capture_dir = PD.stage_prefix(
        N50.HF_N50K_PREFIX,
        args.stage_root,
        workers=args.stage_workers,
        max_files=args.max_chunks,
        revision=revision,
    )
    pass_b = args.stage_root / "pass_b" / "train_context_vectors.pt"
    pb = N1G._load_pass_b_bundle(pass_b)
    assert int(pb["cx_last"].shape[0]) == N50.N_PASS_B, pb["cx_last"].shape
    staged_bytes = sum(p.stat().st_size for p in args.stage_root.rglob("*") if p.is_file())
    timings["stage_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    X_all, Y_all, cap_layers, dtype = _extract_all_layers_deleting(
        capture_dir, args.max_chunks, delete=args.delete_chunks_after_extract
    )
    timings["extract_s"] = round(time.time() - t0, 1)
    if args.max_chunks is None and X_all.shape[0] != N50.N_N50K_NEW:
        raise RuntimeError(f"expected {N50.N_N50K_NEW} n50k rows, got {X_all.shape[0]}")
    missing = [x for x in set(layers_c) | set(layers_a) if x not in cap_layers]
    assert not missing, f"requested layers absent from capture: {missing}"

    pinned = N50._pinned_original_shas(args.orig_dir)
    train, val, test, diag = N50.build_n50k_split(
        X_all.shape[0], None, pinned, n_train=50_000, seed=args.seed
    )
    if args.max_chunks is None:
        assert len(train) > H_DIM, (len(train), H_DIM)
    elif len(train) <= H_DIM:
        logger.warning(
            "[smoke] n_train=%d <= d=%d — SHAPE-SMOKE regime (deliberately under-determined; "
            "estimator-degenerate R^2, structural test only)",
            len(train),
            H_DIM,
        )
    else:
        logger.info("[smoke] n_train=%d > d=%d (well-posed even at smoke scale)", len(train), H_DIM)
    D, present = _load_draws(draws_dir)
    split = {
        "diag": diag,
        "store_revision": revision,
        "staged_bytes": int(staged_bytes),
        "n_chunks_extracted": None if args.max_chunks is None else int(args.max_chunks),
        "metadata": _meta(),
    }
    _write_json_atomic(args.out_eval / "split.json", split)
    key = _run_key(args, diag["train_sha256"])

    def _assemble(field: str, layer: int) -> np.ndarray:
        col = cap_layers.index(layer)
        arr = X_all if field == "cx_last" else Y_all
        return np.concatenate(
            [N50._slice_layer(pb, field, layer), arr[:, col, :].astype(np.float32)]
        )

    # per-answer-layer precompute: assembled Y, avg target, whitening, sst draws
    t0 = time.time()
    boot_idx, counts = _boot_counts(N_TEST, args.n_draws, BOOT_SEED)
    Y_by_a: dict[int, np.ndarray] = {}
    y_avg_by_a: dict[int, np.ndarray] = {}
    whiten_by_a: dict[int, tuple] = {}
    ymu_np_by_a: dict[int, np.ndarray] = {}
    sst_point = np.empty(len(layers_a))
    sst_draws = np.empty((len(layers_a), args.n_draws))
    for j, l_a in enumerate(layers_a):
        Y = _assemble("v_x", l_a)
        y_avg = AVG._avg_targets(Y[test], D[:, :, l_a, :], present)
        mu, ell = P1R.train_whitening_stats(Y[train], dev)
        zp = P1R.whiten(y_avg, mu, ell)
        sst_point[j], sst_draws[j] = _sst_draws_for_target(y_avg, counts, boot_idx)
        Y_by_a[l_a] = Y
        y_avg_by_a[l_a] = y_avg
        whiten_by_a[l_a] = (mu, ell, zp)
        logger.info("[precompute] answer layer %d ready (%d/%d)", l_a, j + 1, len(layers_a))
    _write_npz_atomic(
        args.out_eval / "bootstrap_inputs.npz",
        sst_point=sst_point,
        sst_draws=sst_draws,
        boot_seed=np.int64(BOOT_SEED),
        n_draws=np.int64(args.n_draws),
        layers_a=np.array(layers_a, dtype=np.int64),
        train_sha256=np.bytes_(diag["train_sha256"].encode()),
    )
    timings["precompute_s"] = round(time.time() - t0, 1)

    order = list(layers_c)
    if args.pilot_row in order:
        order = [args.pilot_row] + [x for x in order if x != args.pilot_row]
    rows: dict[int, dict] = {}
    ib_parity_done = False
    for r, l_c in enumerate(order):
        jp, npz_p = _row_paths(args, l_c)
        if args.resume:
            hit = _row_resume_hit(args, l_c, key)
            if hit is not None:
                logger.info("[row %d/%d] L%d resume-skip", r + 1, len(order), l_c)
                rows[l_c] = hit
                continue
        t_row = time.time()
        X = _assemble("cx_last", l_c)
        xfac = _xside_factorize(X, train, dev, args.ridge_block)
        xmu64 = xfac["xmu"].cpu().numpy()
        x_te64 = np.asarray(X[test], np.float64)
        cells: dict[str, dict] = {}
        row_sse = np.empty((len(layers_a), N_TEST))
        row_hw = np.empty((len(layers_a), N_TEST), dtype=np.uint8)
        row_he = np.empty((len(layers_a), N_TEST), dtype=np.uint8)
        for j, l_a in enumerate(layers_a):
            t_cell = time.time()
            Y = Y_by_a[l_a]
            y_avg = y_avg_by_a[l_a]
            mu, ell, zp = whiten_by_a[l_a]
            pred, meta, ymu = _fit_ridge_from_xfac(
                xfac, X, Y, train, val, test, N50.LAMBDAS_N50K, dev, args.ridge_block
            )
            m_ridge, res_i, hw, he = _score_cell(pred, y_avg, mu, ell, zp)
            pred_ib = x_te64 + (ymu.cpu().numpy() - xmu64)  # separable identity+bias
            if not ib_parity_done and l_c == l_a:
                ref = MB.identity_bias_predict(X[train], Y[train], X[test])
                assert np.allclose(pred_ib, ref, rtol=1e-9, atol=1e-8), (
                    "separable identity+bias diverged from MB.identity_bias_predict"
                )
                logger.info("[ib-parity] L%d: separable bias == helper (atol 1e-8)", l_c)
                ib_parity_done = True
            m_ib, _, _, _ = _score_cell(pred_ib, y_avg, mu, ell, zp)
            cells[str(l_a)] = {
                "layer_c": int(l_c),
                "layer_a": int(l_a),
                "ridge": m_ridge,
                "identity_bias": {k: m_ib[k] for k in ("whole_map_r2", "retrieval")},
                "fit_meta": meta,
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "wall_s": round(time.time() - t_cell, 1),
            }
            row_sse[j] = res_i
            row_hw[j] = hw
            row_he[j] = he
            logger.info(
                "[cell] L%d->L%d r2=%.4f acc1_wcsls=%.4f lam=%.3g (%.1fs)",
                l_c,
                l_a,
                m_ridge["whole_map_r2"],
                m_ridge["retrieval"]["whiten_csls"]["acc_at_k"][1],
                meta["selected_lambda"],
                time.time() - t_cell,
            )
        row = {
            "unit_key": key,
            "layer_c": int(l_c),
            "cells": cells,
            "wall_s": round(time.time() - t_row, 1),
        }
        _write_npz_atomic(
            npz_p,
            sse=row_sse,
            hits_wcsls=row_hw,
            hits_euclid=row_he,
            layers_a=np.array(layers_a, dtype=np.int64),
            layer_c=np.int64(l_c),
        )
        _write_json_atomic(jp, row)
        rows[l_c] = row
        wall = time.time() - t_row
        timings[f"row_L{l_c}_s"] = round(wall, 1)
        logger.info("[row %d/%d] L%d done in %.0fs", r + 1, len(order), l_c, wall)
        if r == 0 and len(order) > 1:
            logger.info(
                "[pilot] row L%d wall %.0fs -> projected remaining %d rows ~%.0f min",
                l_c,
                wall,
                len(order) - 1,
                wall * (len(order) - 1) / 60.0,
            )
        _upload_eval(args)

    timings["total_s"] = round(time.time() - t_start, 1)
    _assemble_outputs(args, rows, split, timings)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="#1901 xlayer-grid (28x28 cross-layer ridge grid)")
    ap.add_argument("--stage-root", type=Path, default=Path("/workspace/xlayer_stage"))
    ap.add_argument("--orig-dir", type=Path, default=N50.DEFAULT_ORIG_DIR)
    ap.add_argument("--out-eval", type=Path, default=OUT_EVAL_DEFAULT)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--layers-c", type=int, nargs="+", default=list(LAYERS_ALL))
    ap.add_argument("--layers-a", type=int, nargs="+", default=list(LAYERS_ALL))
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="smoke only: stage+extract N chunk files; skips the parity gate",
    )
    ap.add_argument(
        "--pilot-row",
        type=int,
        default=19,
        help="run this context-layer row first and log its wall",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip rows whose checkpoint JSON+npz match the run key",
    )
    ap.add_argument(
        "--delete-chunks-after-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="unlink each staged chunk once copied (peak-disk bound)",
    )
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42, help="split seed (banked recipe)")
    ap.add_argument("--n-threads", type=int, default=16)
    ap.add_argument("--ridge-block", type=int, default=N1M.RIDGE_BLOCK)
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument(
        "--revision",
        type=str,
        default=None,
        help="data-repo commit pin (default: resolve main once at start)",
    )
    ap.add_argument("--upload-every", type=int, default=4)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        return 0
    torch.set_num_threads(int(args.n_threads))
    args.out_eval.mkdir(parents=True, exist_ok=True)
    run_grid(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
