#!/usr/bin/env python3
"""Task #2569 leg 10: four-way variance decomposition of the L19 answer state.

Decomposes the total single-draw variance of v_A (mean answer-token residual
state, layer 19, Qwen2.5-7B-Instruct) into four disjoint fractions:

  L  linear from v_C:      variance of the best linear predictor (the banked
                           963k-row #779 ridge map, applied via the registered
                           ``issue2569_operator.predict`` path).
  S  sampling noise:       mean within-context variance of v_A across rollouts
                           of the same prompt (unbiased, k-1 denominator).
  W  whole context beyond v_C:  Var(E[v_A | text]) - Var(E[v_A | v_C]).
  N  nonlinear from v_C:   the remainder 1 - L - S - W.

Identity: L + N + W + S = 1 by construction, and N + W = 1 - L - S is pinned
exactly once S is measured; the nearest-neighbor intercept splits N from W.

Estimators
----------
S: per-context unbiased variance over banked stochastic rollouts, three banks:
   #1073 stoch10 (LMSYS, k=10, temp 1.0 / top_p 0.95 / cap 1024 - the SAME
   decode params as the #779 single-draw rows), #1739 wildchat rung (k=5), and
   #2617 svmp (k=10, trait-probing minimal pairs, off-distribution companion).
   The headline S mixes the LMSYS and WildChat banks by the sample corpus mix.
W: nearest-neighbor pairs (i, j) in v_C space. For the pair statistic
   0.5*||v_A(i) - v_A(j)||^2, E[stat] = S + W + 0.5*||mu(v_C_i) - mu(v_C_j)||^2
   where mu(v_C) = E[v_A | v_C]; the last term vanishes as ||dv_C|| -> 0.
   Bin pairs by ||dv_C||^2, weighted linear regression across bins, read the
   intercept, subtract S. Raw and whitened v_C coordinates (Sigma_c from the
   leg-8 moments, shrinkage 1e-2). The intercept is an extrapolation in 3,584
   dimensions; k-th neighbor curves (k = 1, 2, 5, 10) make it visible.
N: remainder, cross-checked by (a) held-out kNN regression of v_A on v_C
   (neighbor-mean, self excluded) whose R^2 minus L lower-bounds N, and (b) the
   #1901 MLP-vs-ridge gap on the same pinned test rows (quoted, not refit).

Per-direction reads apply the same formulas to scalar projections u^T v_A for
the #2617 refusal axis, the three r_B trait directions, the top-5/bottom-5
principal components of the population answer covariance (leg-8 gram_yy), and
the 10 answer directions with the lowest per-direction R^2 under the map
(generalized eigenvectors of the residual covariance against the answer
covariance).

Phases checkpoint to --work; a re-run skips completed phases.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports: binds shared-VM thread caps in-process (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import eigh as scipy_eigh  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2569_operator as OP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("leg10")

LAYER = 19
D = 3584
THEORY = Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory")
DEFAULT_WORK = THEORY / "leg10_dl"
LEG9_MANIFEST = THEORY / "leg9_dl" / "leg9_manifest.json"
MOMENTS_DIR = THEORY / "moments"
KTH_NEIGHBORS = (1, 2, 5, 10)
KNN_KS = (5, 10, 20, 50)
BIN_FRACS = (0.25, 0.5, 1.0)
PRIMARY_FRAC = 0.5
N_BINS = 20
N_BOOT = 2000
SEED = 25690
DUP_EPS = 1e-6
POOLED_L_GIVEN = 0.726  # leg-2 population linear ceiling (task brief)
# #1901 metric battery, L19, pinned LMSYS test-1000 (mlp_scaling_dense_L19.json +
# context_arm.json): quoted as an independent lower bound on N, never refit here.
MLP_1901 = {
    "ridge_963k": 0.7542,
    "mlp_w8192_963k": 0.8104,
    "mlp_w32768_963k": 0.8134,
    "ridge_500k": 0.7609,
    "mlp_w8192_500k": 0.8073,
}


# ── estimators (pure, unit-tested) ────────────────────────────────────────────────


def within_between(draws: np.ndarray) -> dict:
    """Unbiased within/between decomposition of per-context draws.

    ``draws`` is (n_ctx, k, d) (or (n_ctx, k) for a scalar projection). Returns
    absolute traces: ``s_abs`` = mean unbiased within-context variance (summed
    over dims), ``between_abs`` = variance of the true context means (variance
    of sample means minus s_abs/k), ``total_abs`` = single-draw total variance
    (= between_abs + s_abs), and ``s_frac`` = s_abs / total_abs.
    """
    a = np.asarray(draws, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    n_ctx, k, _d = a.shape
    assert n_ctx >= 2 and k >= 2, (n_ctx, k)
    means = a.mean(axis=1)  # (n_ctx, d)
    within = ((a - means[:, None, :]) ** 2).sum(axis=(1, 2)) / (k - 1)  # (n_ctx,)
    s_abs = float(within.mean())
    grand = means.mean(axis=0)
    var_means = float(((means - grand) ** 2).sum() / (n_ctx - 1))
    between_abs = var_means - s_abs / k
    total_abs = between_abs + s_abs
    return {
        "n_ctx": int(n_ctx),
        "k": int(k),
        "s_abs": s_abs,
        "between_abs": float(between_abs),
        "total_abs": float(total_abs),
        "s_frac": float(s_abs / total_abs),
        "within_per_ctx": within,
    }


def bootstrap_ci(values: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    """Percentile 95% CI of ``stat`` over resampled rows of ``values`` (axis 0)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draws[b] = stat(values[rng.integers(0, n, size=n)])
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def binned_intercept(
    d2: np.ndarray,
    stat: np.ndarray,
    frac: float = 1.0,
    n_bins: int = N_BINS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Weighted linear regression of the pair statistic on ||dv_C||^2 across bins.

    Keeps the finest ``frac`` of pairs by d2, forms ``n_bins`` quantile bins,
    regresses bin-mean stat on bin-mean d2 weighted by bin count, and returns
    the intercept (the d2 -> 0 extrapolation) with a pair-bootstrap 95% CI.
    """
    d2 = np.asarray(d2, dtype=np.float64)
    stat = np.asarray(stat, dtype=np.float64)
    assert d2.shape == stat.shape and d2.ndim == 1
    keep = d2 <= np.quantile(d2, frac) if frac < 1.0 else np.ones_like(d2, dtype=bool)
    x, y = d2[keep], stat[keep]
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_of = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)

    def _fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        cnt = np.bincount(bin_of, minlength=n_bins).astype(np.float64)
        bx = np.bincount(bin_of, weights=xs, minlength=n_bins)
        by = np.bincount(bin_of, weights=ys, minlength=n_bins)
        ok = cnt > 0
        bx, by, w = bx[ok] / cnt[ok], by[ok] / cnt[ok], cnt[ok]
        wsum = w.sum()
        mx, my = (w * bx).sum() / wsum, (w * by).sum() / wsum
        slope = float((w * (bx - mx) * (by - my)).sum() / (w * (bx - mx) ** 2).sum())
        return float(my - slope * mx), slope, bx, by

    intercept, slope, bx, by = _fit(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = x.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cnt = np.bincount(bin_of[idx], minlength=n_bins).astype(np.float64)
        bxs = np.bincount(bin_of[idx], weights=x[idx], minlength=n_bins)
        bys = np.bincount(bin_of[idx], weights=y[idx], minlength=n_bins)
        ok = cnt > 0
        bxb, byb, w = bxs[ok] / cnt[ok], bys[ok] / cnt[ok], cnt[ok]
        wsum = w.sum()
        mx, my = (w * bxb).sum() / wsum, (w * byb).sum() / wsum
        sl = (w * (bxb - mx) * (byb - my)).sum() / (w * (bxb - mx) ** 2).sum()
        boots[b] = my - sl * mx
    return {
        "frac": float(frac),
        "n_pairs": int(n),
        "intercept": float(intercept),
        "slope": float(slope),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "bin_x": bx.tolist(),
        "bin_y": by.tolist(),
    }


def topk_neighbors(
    x: np.ndarray,
    k: int,
    block: int = 4096,
    dup_eps: float = DUP_EPS,
    ckpt: Path | None = None,
    ckpt_every: int = 16,
    threads: int = 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Blocked exact top-k nearest neighbors (squared euclidean), self excluded.

    Returns (idx (n, k) int64, d2 (n, k) float32, n_dup_pairs). Exact duplicates
    (d2 < dup_eps) are kept in the lists but counted so callers can drop them.
    Checkpoints partial rows to ``ckpt`` every ``ckpt_every`` blocks; a re-run
    resumes from the last completed block.
    """
    torch.set_num_threads(threads)
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    n = xt.shape[0]
    sq = (xt * xt).sum(dim=1)
    idx_out = np.full((n, k), -1, dtype=np.int64)
    d2_out = np.full((n, k), np.inf, dtype=np.float32)
    start_block = 0
    if ckpt is not None and ckpt.exists():
        z = np.load(ckpt)
        if int(z["n"]) == n and int(z["k"]) == k:
            idx_out, d2_out, start_block = z["idx"], z["d2"], int(z["done_blocks"])
            logger.info("[nn] resume from block %d", start_block)
    n_blocks = (n + block - 1) // block
    t0 = time.time()
    for bi in range(start_block, n_blocks):
        lo, hi = bi * block, min((bi + 1) * block, n)
        g = xt[lo:hi] @ xt.T  # (b, n)
        d2 = sq[lo:hi, None] + sq[None, :] - 2.0 * g
        d2.clamp_(min=0.0)
        rows = torch.arange(lo, hi)
        d2[torch.arange(hi - lo), rows] = float("inf")  # self
        vals, idxs = torch.topk(d2, k, dim=1, largest=False)
        idx_out[lo:hi] = idxs.numpy()
        d2_out[lo:hi] = vals.numpy().astype(np.float32)
        if bi == start_block:
            per = time.time() - t0
            logger.info(
                "[nn] pilot block %.1fs -> projected %.1f min for %d blocks",
                per,
                per * (n_blocks - start_block) / 60.0,
                n_blocks,
            )
        if ckpt is not None and ((bi + 1) % ckpt_every == 0 or bi == n_blocks - 1):
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, idx=idx_out, d2=d2_out, n=n, k=k, done_blocks=bi + 1)
            tmp.replace(ckpt)
    n_dup = int((d2_out[:, 0] < dup_eps).sum())
    return idx_out, d2_out, n_dup


def knn_r2_pooled(y: np.ndarray, nn_idx: np.ndarray, ks: tuple[int, ...]) -> dict:
    """Held-out kNN regression R^2 (neighbor-mean prediction, self excluded)."""
    y64 = np.asarray(y, dtype=np.float64)
    ss_tot = float(((y64 - y64.mean(axis=0)) ** 2).sum())
    out = {}
    for k in ks:
        pred = y64[nn_idx[:, :k]].mean(axis=1)
        ss_res = float(((y64 - pred) ** 2).sum())
        out[str(k)] = 1.0 - ss_res / ss_tot
    return out


def per_direction_r2(y: np.ndarray, resid: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """R^2 of the map along unit directions: 1 - Var(resid @ u) / Var(y @ u)."""
    py = np.asarray(y, dtype=np.float64) @ dirs.T
    pr = np.asarray(resid, dtype=np.float64) @ dirs.T
    return 1.0 - pr.var(axis=0, ddof=1) / py.var(axis=0, ddof=1)


def whitener(sigma: np.ndarray, shrink: float = 1e-2) -> np.ndarray:
    """Symmetric (ZCA) whitening transform of shrunk Sigma: (1-a)S + a*(trS/d)*I."""
    d = sigma.shape[0]
    s = (1.0 - shrink) * sigma + shrink * (np.trace(sigma) / d) * np.eye(d)
    w, v = np.linalg.eigh(s)
    assert w.min() > 0, float(w.min())
    return (v * (1.0 / np.sqrt(w))) @ v.T


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    nrm = float(np.linalg.norm(v))
    assert nrm > 0
    return v / nrm


# ── data loading ──────────────────────────────────────────────────────────────────


def load_sample(manifest: dict, n_rows: int, work: Path) -> dict:
    """L19 (v_C, v_A) sample from the n1m capture chunks (deduped by ci)."""
    ck = work / "sample_L19.npz"
    if ck.exists():
        z = np.load(ck)
        return {"x": z["x"], "y": z["y"], "ci": z["ci"]}
    xs, ys, cis = [], [], []
    chunk_keys = sorted(k for k in manifest["paths"] if "final_token_capture" in k)
    for key in chunk_keys:
        b = torch.load(manifest["paths"][key], map_location="cpu", weights_only=False, mmap=True)
        col = [int(v) for v in b["layers"]].index(LAYER)
        xs.append(b["cx_last"][:, col, :].to(torch.float32).numpy())
        ys.append(b["v_x"][:, col, :].to(torch.float32).numpy())
        cis.append(np.asarray([int(c) for c in b["ci"]], dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    ci = np.concatenate(cis)
    _, first = np.unique(ci, return_index=True)
    keep = np.sort(first)[:n_rows]
    x, y, ci = x[keep], y[keep], ci[keep]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    np.savez(ck, x=x, y=y, ci=ci)
    logger.info("[sample] %d rows from %d chunks", x.shape[0], len(chunk_keys))
    return {"x": x, "y": y, "ci": ci}


def corpus_mix(manifest: dict, ci: np.ndarray) -> dict:
    """Per-corpus row counts for the sample cis from the row_meta shards."""
    needed = set(int(c) for c in ci.tolist())
    counts: dict[str, int] = {}
    shards = sorted(v for k, v in manifest["paths"].items() if "/row_meta_" in k)
    seen = 0
    for p in shards:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if int(r["ci"]) in needed:
                    counts[str(r.get("corpus"))] = counts.get(str(r.get("corpus")), 0) + 1
                    seen += 1
        if seen == len(needed):
            break
    assert seen == len(needed), (seen, len(needed))
    return counts


def load_stoch10(manifest: dict, dirs: np.ndarray) -> dict:
    """#1073 stoch10 bank -> pooled within/between pieces + per-direction draws.

    Streams the 10 shards; accumulates per-context sums (for the pooled trace)
    and per-context per-draw direction projections (materialized, small).
    """
    shard_keys = sorted(k for k in manifest["paths"] if "stoch10_v_shard" in k)
    assert len(shard_keys) == 10, shard_keys
    sums: dict[int, np.ndarray] = {}
    sumsq: dict[int, float] = {}
    kcount: dict[int, int] = {}
    proj: dict[tuple[int, int], np.ndarray] = {}
    for key in shard_keys:
        b = torch.load(manifest["paths"][key], map_location="cpu", weights_only=False, mmap=True)
        col = [int(v) for v in b["layers"]].index(LAYER)
        summ = b["summ"][:, col, :].to(torch.float64).numpy()
        pr = summ @ dirs.T
        for row, (ci_r, ri) in enumerate(list(b["index"])):
            c, r = int(ci_r), int(ri)
            vec = summ[row]
            if c not in sums:
                sums[c] = vec.copy()
                sumsq[c] = float(vec @ vec)
                kcount[c] = 1
            else:
                sums[c] += vec
                sumsq[c] += float(vec @ vec)
                kcount[c] += 1
            proj[(c, r)] = pr[row]
    ks = sorted(set(kcount.values()))
    assert ks == [10], ks
    ctxs = sorted(kcount)
    n_ctx, k = len(ctxs), 10
    means = np.stack([sums[c] / k for c in ctxs])
    within = np.array(
        [(sumsq[c] - k * float(means[i] @ means[i])) / (k - 1) for i, c in enumerate(ctxs)]
    )
    s_abs = float(within.mean())
    grand = means.mean(axis=0)
    var_means = float(((means - grand) ** 2).sum() / (n_ctx - 1))
    pdraws = np.stack([np.stack([proj[(c, r)] for r in range(k)]) for c in ctxs])  # (n,10,ndir)
    return {
        "name": "lmsys_stoch10",
        "n_ctx": n_ctx,
        "k": k,
        "s_abs": s_abs,
        "between_abs": var_means - s_abs / k,
        "total_abs": var_means - s_abs / k + s_abs,
        "within_per_ctx": within,
        "dir_draws": pdraws,
    }


def load_wcrung(manifest: dict, dirs: np.ndarray) -> dict:
    """#1739 wildchat-rung bank (k=5 sampled rollouts) -> same pieces."""
    t1_keys = sorted(k for k in manifest["paths"] if "/t1_L19_shard" in k)
    ri_keys = sorted(k for k in manifest["paths"] if "/row_index_shard" in k)
    assert len(t1_keys) == len(ri_keys) == 20, (len(t1_keys), len(ri_keys))
    rows_meta: list[dict] = []
    for key in ri_keys:
        with open(manifest["paths"][key], encoding="utf-8") as fh:
            rows_meta.extend(json.loads(ln) for ln in fh if ln.strip())
    arrs = [np.load(manifest["paths"][k]) for k in t1_keys]
    t1 = np.concatenate(arrs).astype(np.float64)
    assert t1.shape[0] == len(rows_meta), (t1.shape, len(rows_meta))
    by_ctx: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(rows_meta):
        by_ctx.setdefault(str(m["context_id"]), []).append((int(m.get("rollout_k") or 0), i))
    stacks = []
    for _cid, lst in sorted(by_ctx.items()):
        lst.sort()
        assert len(lst) >= 5, len(lst)
        stacks.append(t1[[i for _r, i in lst[:5]]])
    draws = np.stack(stacks)  # (n_ctx, 5, d)
    wb = within_between(draws)
    return {
        "name": "wildchat_rung",
        **{k: wb[k] for k in ("n_ctx", "k", "s_abs", "between_abs", "total_abs")},
        "within_per_ctx": wb["within_per_ctx"],
        "dir_draws": np.einsum("nkd,md->nkm", draws, dirs),
    }


def load_svmp_draws(leg9_manifest: dict, dirs: np.ndarray) -> dict:
    """#2617 svmp bank (k=10, tail-inclusive mean, trait-probing pairs)."""
    p = leg9_manifest["issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt"]
    st = torch.load(p, map_location="cpu", weights_only=False)
    col = [int(v) for v in st["layers"]].index(LAYER)
    va = st["va_tail_incl"][:, col, :].to(torch.float64).numpy()
    by_ctx: dict[str, list[tuple[int, int]]] = {}
    for i, rec in enumerate(st["index"]):
        by_ctx.setdefault(str(rec["context_id"]), []).append((int(rec["draw"]), i))
    stacks = []
    for _cid, lst in sorted(by_ctx.items()):
        lst.sort()
        stacks.append(va[[i for _r, i in lst]])
    draws = np.stack(stacks)
    wb = within_between(draws)
    return {
        "name": "svmp_2617",
        **{k: wb[k] for k in ("n_ctx", "k", "s_abs", "between_abs", "total_abs")},
        "within_per_ctx": wb["within_per_ctx"],
        "dir_draws": np.einsum("nkd,md->nkm", draws, dirs),
    }


def refusal_axis(leg9_manifest: dict) -> np.ndarray:
    """#2617 refusal axis: unit mean observed answer shift (hi - lo) over flip pairs."""
    import issue2569_refusal_kernel as RK

    sv = RK.load_svmp(leg9_manifest)
    li = sv["layers"].index(LAYER)
    va = sv["va"][:, li]
    pairs = sv["pairs"]
    hi = np.array([p["hi"] for p in pairs])
    lo = np.array([p["lo"] for p in pairs])
    member = np.array([(p["group"] == "flip") and not p["is_control_cell"] for p in pairs])
    dva = va[hi[member]] - va[lo[member]]
    assert dva.shape[0] >= 2
    return unit(dva.mean(axis=0))


# ── direction-level assembly ──────────────────────────────────────────────────────


def bank_dir_pieces(bank: dict, j: int) -> dict:
    """Per-direction within/between for direction column ``j`` of a bank."""
    wb = within_between(bank["dir_draws"][:, :, j])
    return {"s_abs": wb["s_abs"], "total_abs": wb["total_abs"], "s_frac": wb["s_frac"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument(
        "--map-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    ap.add_argument("--leg9-manifest", type=Path, default=LEG9_MANIFEST)
    ap.add_argument("--n-rows", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    work: Path = args.work
    repo: Path = args.repo_root
    out_dir = repo / "eval_results/issue_2569/weights/leg10"
    fig_dir = repo / "figures/issue_2569"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((work / "download_manifest.json").read_text())
    assert not manifest.get("errors"), manifest.get("errors")
    leg9 = json.loads(args.leg9_manifest.read_text())

    # 1) sample + map + linear piece --------------------------------------------------
    payload = OP.load_banked_map(LAYER, root=args.map_root)
    smp = load_sample(manifest, args.n_rows, work)
    x, y, ci = smp["x"], smp["y"], smp["ci"]
    n = x.shape[0]
    mix = corpus_mix(manifest, ci)
    logger.info("[sample] corpus mix %s", mix)

    resid = np.empty_like(y, dtype=np.float32)
    for lo_i in range(0, n, 8192):
        hi_i = min(lo_i + 8192, n)
        resid[lo_i:hi_i] = (y[lo_i:hi_i] - OP.predict(payload, x[lo_i:hi_i])).astype(np.float32)
    y64c = y.astype(np.float64) - y.astype(np.float64).mean(axis=0)
    ss_tot = float((y64c**2).sum())
    total_var_abs = ss_tot / (n - 1)
    r2_pooled_sample = 1.0 - float((resid.astype(np.float64) ** 2).sum()) / ss_tot
    logger.info(
        "[linear] pooled sample R^2 %.4f (given ceiling %.3f)", r2_pooled_sample, POOLED_L_GIVEN
    )

    # 2) directions -------------------------------------------------------------------
    gyy = torch.load(MOMENTS_DIR / "gram_yy.pt", map_location="cpu", weights_only=False)
    sig_a_pop = np.asarray(gyy["gram"], dtype=np.float64) / int(gyy["n_rows"])
    mu_a = np.asarray(gyy["mean"], dtype=np.float64)
    sig_a_pop = 0.5 * (sig_a_pop + sig_a_pop.T) - np.outer(mu_a, mu_a)
    evals_a, evecs_a = np.linalg.eigh(sig_a_pop)
    top5 = [unit(evecs_a[:, -1 - i]) for i in range(5)]
    bot5 = [unit(evecs_a[:, i]) for i in range(5)]

    r64 = resid.astype(np.float64)
    r64 -= r64.mean(axis=0)
    sig_res = (r64.T @ r64) / (n - 1)
    sig_a_smp = (y64c.T @ y64c) / (n - 1)
    shrink_b = sig_a_smp + 1e-3 * (np.trace(sig_a_smp) / D) * np.eye(D)
    gev, gvec = scipy_eigh(sig_res, shrink_b, subset_by_index=[D - 10, D - 1])
    worst10 = [unit(gvec[:, -1 - i]) for i in range(10)]
    del r64

    dir_names = (
        ["refusal_axis_2617", "r_B_evil", "r_B_sycophancy", "r_B_hallucination"]
        + [f"answer_PC{i + 1}" for i in range(5)]
        + [f"answer_PC_bottom{i + 1}" for i in range(5)]
        + [f"worst_R2_dir{i + 1}" for i in range(10)]
    )
    rb = {
        t: unit(
            np.asarray(
                torch.load(
                    manifest["paths"].get(f"issue779_monitoring/r_b/{t}.pt")
                    or leg9.get(f"issue779_monitoring/r_b/{t}.pt")
                    or str(
                        Path(
                            "/mnt/eps-data/thomasjiralerspong/huggingface-cache/hub/"
                            "datasets--superkaiba1--explore-persona-space-data/snapshots/"
                            "037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring/r_b"
                        )
                        / f"{t}.pt"
                    ),
                    map_location="cpu",
                    weights_only=False,
                )["r_b"][LAYER]
            )
        )
        for t in ("evil", "sycophancy", "hallucination")
    }
    dirs = np.stack(
        [refusal_axis(leg9), rb["evil"], rb["sycophancy"], rb["hallucination"]]
        + top5
        + bot5
        + worst10
    )
    assert dirs.shape == (len(dir_names), D)

    # 3) S banks ----------------------------------------------------------------------
    banks = [load_stoch10(manifest, dirs), load_wcrung(manifest, dirs), load_svmp_draws(leg9, dirs)]
    n_lm = mix.get("lmsys", 0)
    n_wc = mix.get("wildchat", 0)
    w_lm = n_lm / max(n_lm + n_wc, 1)
    s_abs_mixture = w_lm * banks[0]["s_abs"] + (1 - w_lm) * banks[1]["s_abs"]
    s_frac = s_abs_mixture / total_var_abs
    s_ci = {}
    for b in banks:
        w = b["within_per_ctx"]
        b["s_abs_ci95"] = bootstrap_ci(w, lambda v: float(v.mean()), args.n_boot)
        s_ci[b["name"]] = b["s_abs_ci95"]

    # 4) NN neighbor passes -----------------------------------------------------------
    idx_raw, d2_raw, dup_raw = topk_neighbors(
        x, k=max(KNN_KS) + 1, ckpt=work / "nn_raw.npz", threads=args.threads
    )
    gxx = torch.load(MOMENTS_DIR / "gram_xx.pt", map_location="cpu", weights_only=False)
    sig_c = np.asarray(gxx["gram"], dtype=np.float64) / int(gxx["n_rows"])
    mu_c = np.asarray(gxx["mean"], dtype=np.float64)
    sig_c = 0.5 * (sig_c + sig_c.T) - np.outer(mu_c, mu_c)
    wt = whitener(sig_c, shrink=1e-2)
    xw = ((x.astype(np.float64) - mu_c) @ wt).astype(np.float32)
    idx_wht, d2_wht, dup_wht = topk_neighbors(
        xw, k=max(KTH_NEIGHBORS) + 1, ckpt=work / "nn_wht.npz", threads=args.threads
    )

    def pair_stats(nn_idx: np.ndarray, nn_d2: np.ndarray, kth: int) -> dict:
        j = nn_idx[:, kth - 1]
        ok = nn_d2[:, kth - 1] >= DUP_EPS
        dy = y[ok].astype(np.float64) - y[j[ok]].astype(np.float64)
        return {
            "d2": nn_d2[ok, kth - 1].astype(np.float64),
            "stat_pooled": 0.5 * (dy**2).sum(axis=1),
            "stat_dirs": 0.5 * (dy @ dirs.T) ** 2,
            "n_dropped_dup": int((~ok).sum()),
        }

    nn = {}
    for tag, (nn_idx, nn_d2) in (("raw", (idx_raw, d2_raw)), ("whitened", (idx_wht, d2_wht))):
        per_k = {}
        for kth in KTH_NEIGHBORS:
            ps = pair_stats(nn_idx, nn_d2, kth)
            per_k[str(kth)] = {
                "mean_stat_pooled": float(ps["stat_pooled"].mean()),
                "mean_d2": float(ps["d2"].mean()),
                "n_pairs": int(ps["d2"].size),
                "n_dropped_dup": ps["n_dropped_dup"],
            }
        ps1 = pair_stats(nn_idx, nn_d2, 1)
        fits = {
            str(f): binned_intercept(ps1["d2"], ps1["stat_pooled"], frac=f, n_boot=args.n_boot)
            for f in BIN_FRACS
        }
        nn[tag] = {"kth": per_k, "intercept_fits": fits, "_ps1": ps1}
    dup_note = {"raw": dup_raw, "whitened": dup_wht}

    intercept_primary = nn["raw"]["intercept_fits"][str(PRIMARY_FRAC)]["intercept"]
    w_abs = intercept_primary - s_abs_mixture
    w_frac = w_abs / total_var_abs
    ps1_raw = nn["raw"]["_ps1"]
    dir_fits = [
        binned_intercept(
            ps1_raw["d2"], ps1_raw["stat_dirs"][:, jd], frac=PRIMARY_FRAC, n_boot=args.n_boot // 2
        )
        for jd in range(len(dir_names))
    ]

    # 5) kNN regression + linear per direction ---------------------------------------
    knn = knn_r2_pooled(y, idx_raw[:, : max(KNN_KS)], KNN_KS)
    l_dirs = per_direction_r2(y, resid, dirs)

    # 6) assemble ---------------------------------------------------------------------
    l_frac = POOLED_L_GIVEN
    n_frac = 1.0 - l_frac - s_frac - w_frac
    knn_best = max(knn.values())
    rows = []
    for jd, name in enumerate(dir_names):
        var_u_sample = float((y64c @ dirs[jd]).var(ddof=1))
        s_u_abs = (
            w_lm * bank_dir_pieces(banks[0], jd)["s_abs"]
            + (1 - w_lm) * bank_dir_pieces(banks[1], jd)["s_abs"]
        )
        s_u = s_u_abs / var_u_sample
        w_u = (dir_fits[jd]["intercept"] - s_u_abs) / var_u_sample
        l_u = float(l_dirs[jd])
        rows.append(
            {
                "direction": name,
                "L": l_u,
                "S": s_u,
                "W": w_u,
                "N": 1.0 - l_u - s_u - w_u,
                "var_u_sample_abs": var_u_sample,
                "s_u_abs": s_u_abs,
                "nn_intercept_abs": dir_fits[jd]["intercept"],
                "nn_intercept_ci95_abs": dir_fits[jd]["ci95"],
            }
        )

    doc = {
        "task": "issue2569 leg10 variance decomposition (L19)",
        "definitions": {
            "L": "fraction of single-draw v_A variance carried by the banked linear map's prediction (pooled value taken from the leg-2 population linear ceiling)",
            "S": "mean within-context variance of v_A across sampled rollouts of the same prompt, over total variance (unbiased, k-1)",
            "W": "Var(E[v_A|text]) - Var(E[v_A|v_C]): whole-context information beyond the last-prompt-token state; NN-pair intercept minus S",
            "N": "remainder 1 - L - S - W: v_C-determined but not linearly readable",
            "identity": "L + N + W + S = 1 by construction; N + W = 1 - L - S is pinned once S is measured",
        },
        "provenance": {
            "sample": {
                "source": "#779 fitter-fair-comparison-n1m capture chunks (fresh seeded 205-of-1920 chunk sample, seed 25690)",
                "model": "Qwen2.5-7B-Instruct",
                "decode": "1 sampled rollout per context, temperature 1.0, top_p 0.95, max 1024 tokens, seed 42",
                "v_A": "mean over the full response span incl. the 2 template-end tokens (v_x)",
                "v_C": "last prompt-token residual state (cx_last), layer 19",
                "n_rows": int(n),
                "corpus_mix": mix,
            },
            "s_banks": {
                "lmsys_stoch10": "#1073 decode-regime bank: 10 sampled rollouts per LMSYS context, temperature 1.0, top_p 0.95, max 1024 (same decode as the sample rows), teacher-forced span-mean capture",
                "wildchat_rung": "#1739 wildchat rung capture store: 5 sampled rollouts per WildChat context (answer-span mean, t1 kind)",
                "svmp_2617": "#2617 minimal refusal pairs: 10 sampled rollouts per probe context, tail-inclusive answer mean (trait-probing, off-distribution companion)",
            },
            "caveats": [
                "#2091-family rollouts carry a 19.5% token-cap hit rate and 3-9% off-language drift; both inflate S",
                "the NN intercept is an extrapolation in 3,584 dimensions; k-th neighbor curves shown for k=1,2,5,10",
                "pooled L is the leg-2 population ceiling on the full 963k pool; the sample rows are map-training rows (optimism bounded by the 0.726 vs 0.719 ceiling-vs-heldout gap)",
                "S transfers from separate context banks (LMSYS stoch10 + WildChat rung), mixed by the sample corpus mix",
            ],
        },
        "pooled": {
            "total_var_abs": total_var_abs,
            "L": l_frac,
            "S": s_frac,
            "W": w_frac,
            "N": n_frac,
            "r2_pooled_sample_banked_map": r2_pooled_sample,
            "s_abs_mixture": s_abs_mixture,
            "mixture_weight_lmsys": w_lm,
            "nn_intercept_primary_abs": intercept_primary,
            "nn_intercept_primary_frac_of_pairs": PRIMARY_FRAC,
        },
        "s_banks": [
            {k: v for k, v in b.items() if k not in ("within_per_ctx", "dir_draws")} for b in banks
        ],
        "s_banks_ci95_abs": s_ci,
        "nn": {tag: {kk: vv for kk, vv in blk.items() if kk != "_ps1"} for tag, blk in nn.items()},
        "nn_duplicate_first_neighbors": dup_note,
        "knn_regression_r2": knn,
        "knn_lower_bound_on_N": {k: v - l_frac for k, v in knn.items()},
        "mlp_1901_quoted": MLP_1901,
        "mlp_lower_bound_on_N": {
            "w8192_963k": MLP_1901["mlp_w8192_963k"] - MLP_1901["ridge_963k"],
            "w32768_963k": MLP_1901["mlp_w32768_963k"] - MLP_1901["ridge_963k"],
        },
        "per_direction": rows,
        "knn_best_r2": knn_best,
        "repro": {
            "banked_map": str(payload.path),
            "seed": SEED,
            "n_boot": args.n_boot,
            "threads": args.threads,
        },
    }
    out_json = out_dir / "variance_decomposition_L19.json"
    out_json.write_text(json.dumps(doc, indent=1, default=float))
    logger.info("[out] %s", out_json)

    render_md(doc, out_dir / "variance_decomposition_L19.md")
    render_figure(doc, fig_dir)
    logger.info("DONE")


def render_md(doc: dict, path: Path) -> None:
    p = doc["pooled"]
    pr = doc["provenance"]
    lines = [
        "# Four-way variance decomposition of the L19 answer state (task #2569, leg 10)",
        "",
        "Setup: answers are Qwen2.5-7B-Instruct's own sampled generations "
        f"({pr['sample']['decode']}); v_A is the {pr['sample']['v_A']}; v_C is the "
        f"{pr['sample']['v_C']}. Sample: {pr['sample']['n_rows']:,} single-draw rows "
        f"({pr['sample']['corpus_mix']}). Sampling-noise banks: LMSYS 10-draw "
        "(same decode recipe), WildChat 5-draw, and the #2617 10-draw probe bank as an "
        "off-distribution companion. Definitions: L is the variance fraction carried by the "
        "banked linear map from v_C. S is the mean within-prompt variance across rollouts over "
        "total variance. W is context information beyond the last-prompt-token state, read as the "
        "nearest-neighbor pair intercept minus S. N is the remainder, nonlinearly readable from "
        "v_C. The identity L + N + W + S = 1 holds by construction and N + W = 1 - L - S is "
        "pinned once S is measured.",
        "",
        "## Pooled (variance-weighted over all 3,584 dims)",
        "",
        "| piece | fraction |",
        "|---|---|",
        f"| L (linear from v_C, leg-2 ceiling) | {p['L']:.3f} |",
        f"| S (sampling noise) | {p['S']:.3f} |",
        f"| W (context beyond v_C) | {p['W']:.3f} |",
        f"| N (nonlinear remainder) | {p['N']:.3f} |",
        "",
        f"Banked-map pooled R^2 recomputed on the sample rows: {p['r2_pooled_sample_banked_map']:.4f}.",
        f"NN intercept (raw v_C, finest {int(100 * p['nn_intercept_primary_frac_of_pairs'])}% of pairs): "
        f"{p['nn_intercept_primary_abs']:.1f} absolute against a total variance of "
        f"{p['total_var_abs']:.1f}.",
        "",
        "## kNN and MLP lower bounds on N",
        "",
        "| read | value |",
        "|---|---|",
    ]
    for k, v in doc["knn_regression_r2"].items():
        lines.append(f"| kNN R^2 (k={k}) | {v:.4f} (N >= {v - p['L']:.3f}) |")
    lines += [
        f"| #1901 MLP w8192 minus ridge (963k, same rows) | {doc['mlp_lower_bound_on_N']['w8192_963k']:.4f} |",
        f"| #1901 MLP w32768 minus ridge (963k, same rows) | {doc['mlp_lower_bound_on_N']['w32768_963k']:.4f} |",
        "",
        "## Per direction",
        "",
        "| direction | L | S | W | N |",
        "|---|---|---|---|---|",
    ]
    for r in doc["per_direction"]:
        lines.append(
            f"| {r['direction']} | {r['L']:.3f} | {r['S']:.3f} | {r['W']:.3f} | {r['N']:.3f} |"
        )
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in pr["caveats"]]
    path.write_text("\n".join(lines) + "\n")


PIECE_COLORS = {"L": "#4C72B0", "N": "#DD8452", "W": "#55A868", "S": "#C44E52"}


def render_figure(doc: dict, fig_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    rows = [
        {
            "direction": "pooled",
            **{k: doc["pooled"][k] for k in ("L", "S", "W", "N")},
        }
    ] + doc["per_direction"]
    names = [r["direction"] for r in rows]
    xpos = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for piece in ("L", "N", "W", "S"):
        vals = np.array([max(r[piece], 0.0) for r in rows])
        ax1.bar(xpos, vals, bottom=bottom, color=PIECE_COLORS[piece], label=piece, width=0.8)
        bottom += vals
    neg = np.array([sum(min(r[k], 0.0) for k in ("L", "N", "W", "S")) for r in rows])
    if (neg < 0).any():
        ax1.bar(xpos, neg, bottom=np.zeros(len(rows)), color="#777777", label="negative (est.)")
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("fraction of single-draw answer-state variance")
    ax1.set_title("L / N / W / S per direction (layer 19)")
    ax1.legend(loc="upper right", fontsize=8)

    fits = doc["nn"]["raw"]["intercept_fits"][str(PRIMARY_FRAC)]
    bx, by = np.array(fits["bin_x"]), np.array(fits["bin_y"])
    ax2.plot(bx, by, "o", ms=4, color="#4C72B0", label="bin mean (1st neighbor)")
    xs = np.linspace(0, bx.max(), 50)
    ax2.plot(xs, fits["intercept"] + fits["slope"] * xs, "-", color="#4C72B0", lw=1)
    for kth, blk in doc["nn"]["raw"]["kth"].items():
        ax2.plot(blk["mean_d2"], blk["mean_stat_pooled"], "s", ms=5, color="#DD8452")
        ax2.annotate(f"k={kth}", (blk["mean_d2"], blk["mean_stat_pooled"]), fontsize=7)
    ax2.axhline(
        doc["pooled"]["s_abs_mixture"], color=PIECE_COLORS["S"], ls="--", lw=1, label="S (banks)"
    )
    ax2.plot(0, fits["intercept"], "*", ms=12, color="k", label="intercept = S + W")
    ax2.set_xlabel("||dv_C||^2 (raw coordinates)")
    ax2.set_ylabel("0.5 ||dv_A||^2, pair mean")
    ax2.set_title("NN pair statistic vs v_C distance")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"leg10_variance_decomposition.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
