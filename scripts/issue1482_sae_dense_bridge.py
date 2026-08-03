"""Issue #1482 inline round — SAE<->dense bridge: put every context->answer mapping
in ONE common dense-space (residual-stream) currency, with the SAE reconstruction
ceiling as the reference line.

WHY. #1482's SAE arms report FEATURE-space held-out R2 (`sae_ctx` SAE->SAE 0.690,
`sae_dense_in` dense->SAE 0.722 at mean pooling) while the parent dense map reports
RESIDUAL-STREAM R2 (0.724 ridge / 0.780 MLP on the same 20k holdout). Those numbers
are NOT comparable: they score different targets in different spaces with different
total variance. This round (a) fits the ONE missing cell of the 2x2 design --
SAE-features-over-context -> DENSE answer vector -- and (b) DECODES every
feature-space prediction through the SAE decoder so all mappings are scored against
the SAME dense targets, against the SAME denominator, with the SAE's own
reconstruction of the target as the ceiling any feature-space map inherits.

DECODE IS EXACT FOR MEAN POOLING, AND ONLY FOR MEAN POOLING. The SAE decode is
affine, ``decode(f) = f W_dec^T + b_dec``, so decoding a MEAN-pooled feature vector
equals the mean of the per-token reconstructions:
``decode(mean_t f_t) = mean_t (f_t W_dec^T + b_dec) = mean_t x_hat_t``. The dense
target ``v_x`` is itself the MEAN-response residual at L19, so
``decode(true mean-pooled features)`` is precisely the SAE's reconstruction of the
target. ``max`` / ``frac`` pooling have no such identity and are NOT decoded here.

TWO CEILINGS, both reported. (i) RESTRICTED: decode over the 16,384 retained
answer-side columns (``f_out``) -- the ceiling a feature-space map that only
predicts those columns can possibly reach. (ii) FULL-DICTIONARY: decode over all
131,072 columns of the true pooled features -- the SAE's own reconstruction quality
on this corpus, independent of the restriction.

DENSE TARGETS ARE REGENERATED, NOT BANKED. ``X.npy``/``Y.npy`` stayed pod-local in
#1482 (P4 uploads the SAE store and the scratch metadata only), so
``scripts/issue1482_dense_targets_stage.py`` re-streams ``cx_last``/``v_x`` at L19
for exactly the 142,000 SAE-arm rows out of the #779 n1M capture, joined on the
manifest ``ci`` (never on stream position).

n_train vs d (the #1701 well-posedness duty): 120,000 fit rows against d=16,384
(sae_ctx: 8,192 mean-pooled ++ 8,192 last-token context features) and d=3,584
(dense). Both are the over-determined regime (7.3:1 and 33:1) -- no
under-determined fit is run.

MATCHED-n IS LOAD-BEARING. The banked dense->dense numbers were fit on 943,444
rows; every SAE arm was fit on 120,000. This round refits dense->dense on the SAME
120,000 fit rows so the comparison table has a matched-n dense row, and reports the
banked full-pool number alongside it as context.

STATED DEVIATION (context arm only). The #1482 corpus is SINGLE-TURN, so the prefix
is degenerate and only the context arm of the standing prefix/context both-arms rule
is answerable here; three-arm coverage lives in the #1738 multi-turn twin.

0 GPU. CPU-only, no training, no generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM run)

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("i1482_bridge")

LAYER = 19
H_DIM = 3584
SAE_K = 64
DICT_SIZE = 131_072
# Parent-realized restriction (eval_results/issue_1482/sae_perfeature/summary.json).
# Recomputed here from the store and ASSERTED against these — a reuse-parity check,
# not a configuration knob.
PARENT_F_OUT = 16_384
PARENT_F_IN = 8_192
PARENT_ACTIVITY_FLOOR = 1_200
PARENT_N_FIT = 120_000
KNN_KS = (1, 5, 10)
KNN_N_PRED = 5_000  # deterministic head of the holdout; pool is the full holdout
RIDGE_BLOCK = 10_000  # fp64 row block (memory-bound on the shared VM)


# ── store scan ───────────────────────────────────────────────────────────────────


def _shard_paths(store: Path) -> list[Path]:
    ps = sorted(store.rglob("pooled_*.npz"))
    assert ps, f"no pooled shards under {store}"
    return ps


def _scan_store(shards: list[Path]) -> dict:
    """ONE pass: per-shard row ids + set tags, and the sae_fit activity counts that
    define the feature restriction (parent ``_p3_prep`` recipe, shard-streamed so the
    9.2 GB store is never held whole)."""
    out_counts = np.zeros(DICT_SIZE, dtype=np.int64)
    in_counts = np.zeros(DICT_SIZE, dtype=np.int64)
    n_fit = 0
    rows_by_shard: dict[str, np.ndarray] = {}
    tags_by_shard: dict[str, np.ndarray] = {}
    for p in shards:
        z = np.load(p, allow_pickle=False)
        # NpzFile.__getitem__ RE-READS AND RE-PARSES the member on EVERY access, so
        # every member touched inside a per-row loop is hoisted here (the parent's
        # _load_store materialises the whole shard with dict(); hoisting reads the
        # same members once without holding the ones this pass does not need).
        rows = z["row_idx"].astype(np.int64)
        tags = z["set_tag"].astype(np.int8)
        ans_idx, psi_idx = z["ans_idx"], z["psi_idx"]
        rows_by_shard[p.name] = rows
        tags_by_shard[p.name] = tags
        a_off = np.concatenate([[0], np.cumsum(z["idx_off"])])
        p_off = np.concatenate([[0], np.cumsum(z["psi_off"])])
        for i, tag in enumerate(tags):
            if int(tag) != 1:  # sae_fit only (parent convention)
                continue
            n_fit += 1
            out_counts[ans_idx[a_off[i] : a_off[i + 1]].astype(np.int64)] += 1
            in_counts[psi_idx[p_off[i] : p_off[i + 1]].astype(np.int64)] += 1
    floor = max(1, int(np.ceil(0.01 * n_fit)))
    f_out = np.where(out_counts >= floor)[0]
    f_in = np.where(in_counts >= floor)[0]
    if len(f_out) > PARENT_F_OUT:
        f_out = np.sort(f_out[np.argsort(-out_counts[f_out])[:PARENT_F_OUT]])
    if len(f_in) > PARENT_F_IN:
        f_in = np.sort(f_in[np.argsort(-in_counts[f_in])[:PARENT_F_IN]])
    logger.info("[scan] n_fit=%d floor=%d f_out=%d f_in=%d", n_fit, floor, len(f_out), len(f_in))
    # Reuse-parity: the restriction must reproduce the parent's realized values.
    assert n_fit == PARENT_N_FIT, (n_fit, PARENT_N_FIT)
    assert floor == PARENT_ACTIVITY_FLOOR, (floor, PARENT_ACTIVITY_FLOOR)
    assert len(f_out) == PARENT_F_OUT, (len(f_out), PARENT_F_OUT)
    assert len(f_in) == PARENT_F_IN, (len(f_in), PARENT_F_IN)
    return {
        "out_counts": out_counts,
        "in_counts": in_counts,
        "n_fit": n_fit,
        "floor": floor,
        "f_out": f_out,
        "f_in": f_in,
        "rows_by_shard": rows_by_shard,
        "tags_by_shard": tags_by_shard,
    }


def _scatter_sparse(z, key_idx, key_off, key_val, col_of, row_pos, dest: np.memmap, base: int):
    """Scatter one shard's sparse rows into ``dest[:, base:base+width]``."""
    offs = np.concatenate([[0], np.cumsum(z[key_off])])
    rows, idx_arr, val_arr = z["row_idx"], z[key_idx], z[key_val]  # hoisted: see _scan_store
    for i, r in enumerate(rows):
        pos = row_pos.get(int(r))
        if pos is None:
            continue
        sl = slice(offs[i], offs[i + 1])
        fidx = idx_arr[sl].astype(np.int64)
        keep = fidx < len(col_of)
        cols = col_of[fidx[keep]]
        m = cols >= 0
        dest[pos, base + cols[m]] = val_arr[sl][keep][m].astype(np.float32)


def build_designs(args) -> dict:
    """Build the memmap-backed designs + targets. Resumable via designs_meta.json."""
    work = args.base / "work"
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "designs_meta.json"
    if meta_path.exists() and not args.rebuild:
        logger.info("[designs] resume from %s", meta_path)
        return json.loads(meta_path.read_text())

    shards = _shard_paths(args.base / "store")
    scan = _scan_store(shards)
    f_out, f_in = scan["f_out"], scan["f_in"]

    dense_dir = args.base / "dense"
    dmeta = json.loads((dense_dir / "dense_targets_meta.json").read_text())
    n142 = int(dmeta["n_rows"])
    dense_rows = np.load(dense_dir / "row_ids.npy")
    assert len(dense_rows) == n142
    Xd_all = np.memmap(dense_dir / "X_L19.f32.mm", dtype=np.float32, mode="r", shape=(n142, H_DIM))
    Yd_all = np.memmap(dense_dir / "Y_L19.f32.mm", dtype=np.float32, mode="r", shape=(n142, H_DIM))
    pos142 = {int(r): i for i, r in enumerate(dense_rows)}
    n_hold, n_fitr = int(dmeta["n_holdout"]), int(dmeta["n_sae_fit"])

    # Row registry: keep the dense-stage order (holdout ++ sae_fit ++ sae_val) and
    # drop rows the store lacks (tokenisation drops) — the parent's _p3_prep rule.
    have = set()
    for v in scan["rows_by_shard"].values():
        have.update(int(r) for r in v)
    order = np.asarray([int(r) for r in dense_rows if int(r) in have], dtype=np.int64)
    which = np.asarray(
        [
            0 if pos142[int(r)] < n_hold else (1 if pos142[int(r)] < n_hold + n_fitr else 2)
            for r in order
        ],
        dtype=np.int8,
    )
    te = np.where(which == 0)[0].astype(np.int64)
    tr = np.where(which == 1)[0].astype(np.int64)
    va = np.where(which == 2)[0].astype(np.int64)
    n = len(order)
    row_pos = {int(r): i for i, r in enumerate(order)}
    logger.info(
        "[designs] n=%d (te=%d tr=%d va=%d) dropped=%d", n, len(te), len(tr), len(va), n142 - n
    )
    assert len(tr) and len(va) and len(te)

    d_ctx = 2 * len(f_in)
    d_sae = len(f_out)
    Z = np.memmap(work / "Z_ctx.f32.mm", dtype=np.float32, mode="w+", shape=(n, d_ctx))
    Ycat = np.memmap(work / "Ycat.f32.mm", dtype=np.float32, mode="w+", shape=(n, H_DIM + d_sae))
    Xd = np.memmap(work / "X_dense.f32.mm", dtype=np.float32, mode="w+", shape=(n, H_DIM))
    Z[:] = 0
    Ycat[:] = 0

    col_out = np.full(int(f_out.max()) + 1, -1, dtype=np.int64)
    col_out[f_out] = np.arange(len(f_out))
    col_in = np.full(int(f_in.max()) + 1, -1, dtype=np.int64)
    col_in[f_in] = np.arange(len(f_in))

    t0 = time.time()
    for j, p in enumerate(shards):
        z = np.load(p, allow_pickle=False)
        _scatter_sparse(z, "psi_idx", "psi_off", "psi_mean", col_in, row_pos, Z, 0)
        _scatter_sparse(z, "psil_idx", "psil_off", "psil_val", col_in, row_pos, Z, len(f_in))
        _scatter_sparse(z, "ans_idx", "idx_off", "ans_mean", col_out, row_pos, Ycat, H_DIM)
        if (j + 1) % 400 == 0:
            logger.info("[designs] shard %d/%d %.0fs", j + 1, len(shards), time.time() - t0)
    # dense target + dense design, aligned by global row id
    src = np.asarray([pos142[int(r)] for r in order], dtype=np.int64)
    for s in range(0, n, 20_000):
        sl = slice(s, min(s + 20_000, n))
        Ycat[sl, :H_DIM] = Yd_all[src[sl]]
        Xd[sl] = Xd_all[src[sl]]
    Z.flush()
    Ycat.flush()
    Xd.flush()

    # Sanity: every row got answer-side mass and a dense target.
    probe = np.concatenate([te[:200], tr[:200], va[:200]])
    assert np.abs(np.asarray(Ycat[probe, H_DIM:])).sum(1).min() > 0, "empty SAE target row"
    assert np.abs(np.asarray(Ycat[probe, :H_DIM])).sum(1).min() > 0, "empty dense target row"
    assert np.abs(np.asarray(Z[probe])).sum(1).min() > 0, "empty context design row"

    meta = {
        "n": int(n),
        "d_ctx": int(d_ctx),
        "d_sae": int(d_sae),
        "d_dense": H_DIM,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_holdout": int(len(te)),
        "n_dropped_vs_split": int(n142 - n),
        "activity_floor": int(scan["floor"]),
        "n_fit_scanned": int(scan["n_fit"]),
        "f_out_n": int(len(f_out)),
        "f_in_n": int(len(f_in)),
        "order_convention": "holdout ++ sae_fit ++ sae_val (store-present rows only)",
        "build_wall_s": round(time.time() - t0, 1),
    }
    np.save(work / "order.npy", order)
    np.save(work / "which.npy", which)
    np.save(work / "f_out.npy", f_out)
    np.save(work / "f_in.npy", f_in)
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("[designs] done %s", meta)
    return meta


def _load_designs(args):
    work = args.base / "work"
    meta = json.loads((work / "designs_meta.json").read_text())
    n, d_ctx, d_sae = meta["n"], meta["d_ctx"], meta["d_sae"]
    Z = np.memmap(work / "Z_ctx.f32.mm", dtype=np.float32, mode="r", shape=(n, d_ctx))
    Ycat = np.memmap(work / "Ycat.f32.mm", dtype=np.float32, mode="r", shape=(n, H_DIM + d_sae))
    Xd = np.memmap(work / "X_dense.f32.mm", dtype=np.float32, mode="r", shape=(n, H_DIM))
    which = np.load(work / "which.npy")
    te = np.where(which == 0)[0].astype(np.int64)
    tr = np.where(which == 1)[0].astype(np.int64)
    va = np.where(which == 2)[0].astype(np.int64)
    return meta, Z, Ycat, Xd, tr, va, te


# ── ridge: parent factorization, projected prediction ────────────────────────────


def _project(X, idx, fac, dev, block) -> torch.Tensor:
    """Standardize + rotate an eval design into the eigenbasis ONCE: E = ((X-mu)/sd) @ U.

    ``_ridge_predict_one`` instead materialises W = U @ (UtXtY/(s+lam)) per lambda,
    which costs O(H^2 D) EVERY lambda; projecting the (small) eval design once and
    contracting with the (D-wide) coefficient matrix per lambda is the same product
    re-associated, at O(n_eval H D). Equivalence is asserted numerically in
    ``_assert_predict_equivalence`` before any production use.
    """
    U, xmu, xsd = fac["U"], fac["xmu"], fac["xsd"]
    outs = []
    for s in range(0, len(idx), block):
        b = idx[s : s + block]
        En = (torch.as_tensor(np.asarray(X[b]), dtype=torch.float64, device=dev) - xmu) / xsd
        outs.append(En @ U)
    return torch.cat(outs) if len(outs) > 1 else outs[0]


def _predict_projected(E: torch.Tensor, fac, lam: float, cols: slice | None = None) -> np.ndarray:
    s_eig, UtXtY, ymu = fac["s_eig"], fac["UtXtY"], fac["ymu"]
    C = UtXtY / (s_eig + float(lam))[:, None]
    if cols is not None:
        C = C[:, cols]
        ymu = ymu[cols]
    return ((E @ C) + ymu).cpu().numpy()


def _assert_predict_equivalence(X, idx, fac, dev, block, lams=(1e-2, 1e3), tol=1e-8) -> dict:
    """Projected predict == parent ``_ridge_predict_one``, on a bounded row slice."""
    sub = np.asarray(idx[: min(256, len(idx))], dtype=np.int64)
    E = _project(X, sub, fac, dev, block)
    worst = 0.0
    for lam in lams:
        ref = N1M._ridge_predict_one(X, sub, fac, lam, dev, block)
        got = _predict_projected(E, fac, lam)
        denom = max(1e-12, float(np.abs(ref).max()))
        worst = max(worst, float(np.abs(ref - got).max()) / denom)
    assert worst < tol, f"projected-predict equivalence failed: rel max-abs {worst:.3e}"
    logger.info("[equiv] projected predict == _ridge_predict_one (rel max-abs %.2e)", worst)
    return {"rel_max_abs": worst, "tol": tol, "n_rows": int(len(sub)), "lambdas": list(lams)}


def fit_arm(name, X, Ycat, tr, va, te, target_slices, lambdas, dev, block) -> dict:
    """ONE shared-Gram fit; per-target lambda selected on val; predictions on holdout."""
    t0 = time.time()
    logger.info("[fit %s] factorize d=%d n_train=%d D=%d", name, X.shape[1], len(tr), Ycat.shape[1])
    fac = N1M._ridge_factorize(X, Ycat, tr, dev, block)
    equiv = _assert_predict_equivalence(X, va, fac, dev, block)
    Eva = _project(X, va, fac, dev, block)
    Ete = _project(X, te, fac, dev, block)
    Yva = np.asarray(Ycat[va])
    best = {k: (float(lambdas[0]), -np.inf) for k in target_slices}
    for lam in lambdas:
        for k, sl in target_slices.items():
            pv = _predict_projected(Eva, fac, lam, sl)
            r2 = PR._pooled_r2(pv, Yva[:, sl])
            if np.isfinite(r2) and r2 > best[k][1]:
                best[k] = (float(lam), r2)
    out = {}
    for k, sl in target_slices.items():
        lam = best[k][0]
        out[k] = {
            "pred": _predict_projected(Ete, fac, lam, sl),
            "selected_lambda": lam,
            "val_r2": float(best[k][1]),
            "lambda_grid_edge": (
                "low"
                if lam == float(lambdas[0])
                else ("high" if lam == float(lambdas[-1]) else None)
            ),
        }
        logger.info("[fit %s/%s] lam=%.3g val_r2=%.4f", name, k, lam, best[k][1])
    out["_meta"] = {
        "n_train": int(len(tr)),
        "d": int(X.shape[1]),
        "wall_s": round(time.time() - t0, 1),
        "equivalence_check": equiv,
        "n_train_vs_d": f"{len(tr)} vs {X.shape[1]} (over-determined, ratio "
        f"{len(tr) / X.shape[1]:.1f}:1)",
    }
    del fac
    return out


# ── decode ───────────────────────────────────────────────────────────────────────


def load_decoder(cache_dir: Path, f_out: np.ndarray):
    import issue1482_sae as S

    sae = S.BatchTopKSAE.load(k=SAE_K, device="cpu", cache_dir=cache_dir, layer=LAYER)
    assert tuple(sae.w_dec.shape) == (H_DIM, DICT_SIZE), sae.w_dec.shape
    return sae


def decode_restricted(feats: np.ndarray, sae, f_out: np.ndarray) -> np.ndarray:
    """(n, |f_out|) restricted features -> (n, 3584) reconstruction (+ decoder bias)."""
    Wr = sae.w_dec[:, torch.as_tensor(f_out, dtype=torch.long)].T.contiguous()  # (|f_out|, 3584)
    out = np.empty((feats.shape[0], H_DIM), dtype=np.float32)
    for s in range(0, feats.shape[0], 4096):
        blk = torch.as_tensor(np.asarray(feats[s : s + 4096]), dtype=torch.float32)
        out[s : s + 4096] = (blk @ Wr + sae.b_dec).numpy()
    return out


def decode_full_from_store(shards, row_pos, n_out, sae) -> np.ndarray:
    """FULL-dictionary decode of the TRUE mean-pooled answer features, straight from
    the sparse store (no 131,072-wide densification): row = sum_j v_j W_dec[:, j] + b."""
    out = np.zeros((n_out, H_DIM), dtype=np.float32)
    hit = np.zeros(n_out, dtype=bool)
    W = sae.w_dec  # (3584, 131072)
    for p in shards:
        z = np.load(p, allow_pickle=False)
        offs = np.concatenate([[0], np.cumsum(z["idx_off"])])
        rows, ans_idx, ans_mean = z["row_idx"], z["ans_idx"], z["ans_mean"]  # hoisted
        for i, r in enumerate(rows):
            pos = row_pos.get(int(r))
            if pos is None:
                continue
            sl = slice(offs[i], offs[i + 1])
            idx = torch.as_tensor(ans_idx[sl].astype(np.int64), dtype=torch.long)
            val = torch.as_tensor(ans_mean[sl].astype(np.float32), dtype=torch.float32)
            out[pos] = (W[:, idx] @ val).numpy()
            hit[pos] = True
    assert hit.all(), f"full decode: {int((~hit).sum())} rows unfilled"
    return out + sae.b_dec.numpy()


# ── main ─────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "eval_results/issue_1482/sae_dense_bridge"
    )
    ap.add_argument("--sae-dir", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--designs-only", action="store_true")
    ap.add_argument("--upload", action="store_true", help="push the prediction npz to HF")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sae_dir = args.sae_dir or (args.base / "sae")
    dev = torch.device("cpu")
    t_start = time.time()

    build_designs(args)
    if args.designs_only:
        return
    meta, Z, Ycat, Xd, tr, va, te = _load_designs(args)
    work = args.base / "work"
    order = np.load(work / "order.npy")
    f_out = np.load(work / "f_out.npy")
    d_sae = meta["d_sae"]
    sl_dense, sl_sae = slice(0, H_DIM), slice(H_DIM, H_DIM + d_sae)
    lambdas = N1M.LAMBDAS_N1M

    Yd_te = np.asarray(Ycat[te, :H_DIM], dtype=np.float64)
    Ys_te_f32 = np.asarray(Ycat[te, H_DIM:], dtype=np.float32)
    Ys_te = Ys_te_f32.astype(np.float64)
    # identity+bias is the SAME baseline for every dense-input cell — compute once.
    ib_pred = identity_bias_predict(
        np.asarray(Xd[tr], dtype=np.float64),
        np.asarray(Ycat[tr, :H_DIM], dtype=np.float64),
        np.asarray(Xd[te], dtype=np.float64),
    )
    ib_r2 = float(PR._pooled_r2(ib_pred, Yd_te))
    del ib_pred
    logger.info("[baseline] identity+bias (dense ctx -> dense answer) R2=%.4f", ib_r2)

    # ── arm A: SAE context features -> {dense answer, SAE answer} (shared Gram) ──
    armA = fit_arm(
        "sae_ctx",
        Z,
        Ycat,
        tr,
        va,
        te,
        {"dense": sl_dense, "sae": sl_sae},
        lambdas,
        dev,
        RIDGE_BLOCK,
    )
    # ── arm B: dense context -> {dense answer (matched-n), SAE answer} ──
    armB = fit_arm(
        "dense_ctx",
        Xd,
        Ycat,
        tr,
        va,
        te,
        {"dense": sl_dense, "sae": sl_sae},
        lambdas,
        dev,
        RIDGE_BLOCK,
    )

    sae = load_decoder(sae_dir, f_out)
    shards = _shard_paths(args.base / "store")
    row_pos_te = {int(order[i]): j for j, i in enumerate(te)}

    ceil_restricted = decode_restricted(Ys_te_f32, sae, f_out)
    ceil_full = decode_full_from_store(shards, row_pos_te, len(te), sae)

    cells: dict[str, dict] = {}
    dense_preds: dict[str, np.ndarray] = {}

    def _add(
        key,
        *,
        feature_r2,
        dense_pred,
        selected_lambda,
        val_r2,
        note,
        input_dim,
        input_is_dense,
        save_as=None,
    ):
        if dense_pred is not None and save_as:
            dense_preds[save_as] = np.asarray(dense_pred, dtype=np.float16)
        row = {
            "feature_space_r2": feature_r2,
            "dense_space_r2": (
                None if dense_pred is None else float(PR._pooled_r2(dense_pred, Yd_te))
            ),
            "selected_lambda": selected_lambda,
            "val_r2": val_r2,
            "note": note,
            "input_dim": input_dim,
        }
        if dense_pred is not None:
            row["knn_retrieval"] = {
                m: knn_retrieval(
                    dense_pred[:KNN_N_PRED],
                    Yd_te[:KNN_N_PRED],
                    ks=KNN_KS,
                    metric=m,
                    pool=Yd_te,
                    true_pool_idx=np.arange(KNN_N_PRED),
                )
                for m in ("euclidean", "cosine")
            }
            if input_is_dense:
                row["identity_bias_baseline_dense_r2"] = ib_r2
            else:
                row["identity_bias_baseline_dense_r2"] = None
                row["identity_bias_inapplicable"] = (
                    f"input space is R^{input_dim} (SAE features), target is R^{H_DIM} "
                    "(residual stream) — dimensions differ, the identity family is undefined"
                )
        cells[key] = row

    _add(
        "sae_ctx__to__dense  [NEW]",
        feature_r2=None,
        dense_pred=armA["dense"]["pred"],
        selected_lambda=armA["dense"]["selected_lambda"],
        val_r2=armA["dense"]["val_r2"],
        note="SAE features over context -> dense answer vector; the missing cell of the 2x2",
        save_as="sae_ctx_to_dense",
        input_dim=meta["d_ctx"],
        input_is_dense=False,
    )
    _add(
        "sae_ctx__to__sae  (decoded)",
        feature_r2=float(PR._pooled_r2(armA["sae"]["pred"], Ys_te)),
        dense_pred=decode_restricted(armA["sae"]["pred"], sae, f_out),
        selected_lambda=armA["sae"]["selected_lambda"],
        val_r2=armA["sae"]["val_r2"],
        note="refit of the parent sae_ctx/mean/ridge arm; prediction decoded to R^3584",
        save_as="sae_ctx_to_sae_decoded",
        input_dim=meta["d_ctx"],
        input_is_dense=False,
    )
    _add(
        "dense_ctx__to__sae  (decoded)",
        feature_r2=float(PR._pooled_r2(armB["sae"]["pred"], Ys_te)),
        dense_pred=decode_restricted(armB["sae"]["pred"], sae, f_out),
        selected_lambda=armB["sae"]["selected_lambda"],
        val_r2=armB["sae"]["val_r2"],
        note="refit of the parent sae_dense_in/mean/ridge arm; prediction decoded to R^3584",
        save_as="dense_ctx_to_sae_decoded",
        input_dim=H_DIM,
        input_is_dense=True,
    )
    _add(
        "dense_ctx__to__dense  (matched n=120k)",
        feature_r2=None,
        dense_pred=armB["dense"]["pred"],
        selected_lambda=armB["dense"]["selected_lambda"],
        val_r2=armB["dense"]["val_r2"],
        note="the parent dense map REFIT on the SAE arms' own 120,000 fit rows",
        save_as="dense_ctx_to_dense",
        input_dim=H_DIM,
        input_is_dense=True,
    )

    ceilings = {
        "sae_reconstruction_restricted_f_out": {
            "dense_space_r2": float(PR._pooled_r2(ceil_restricted, Yd_te)),
            "note": f"decode of the TRUE mean-pooled features over the {len(f_out)} retained "
            "columns — the ceiling a restricted feature-space map inherits",
        },
        "sae_reconstruction_full_dictionary": {
            "dense_space_r2": float(PR._pooled_r2(ceil_full, Yd_te)),
            "note": f"decode of the TRUE mean-pooled features over all {DICT_SIZE} columns — "
            "the SAE's own reconstruction of v_x on this holdout",
        },
    }
    banked = {
        "dense_ctx__to__dense__ridge__full_pool": {
            "dense_space_r2": 0.7242755664709597,
            "n_train": 943444,
            "source": "eval_results/issue_1482/percontext/refit_holdout__ridge__seed0.json",
        },
        "dense_ctx__to__dense__mlp_w8192__full_pool": {
            "dense_space_r2": 0.7796256442170776,
            "n_train": 943444,
            "source": "eval_results/issue_1482/percontext/refit_holdout__mlp_w8192__seed0.json",
        },
        "sae_ctx__to__sae__mean__ridge__parent": {
            "feature_space_r2": 0.6901409540488584,
            "selected_lambda": 10000.0,
            "source": "eval_results/issue_1482/sae_perfeature/summary.json",
        },
        "dense_ctx__to__sae__mean__ridge__parent": {
            "feature_space_r2": 0.7216031048274667,
            "selected_lambda": 3162.2776601683795,
            "source": "eval_results/issue_1482/sae_perfeature/summary.json",
        },
    }
    doc = {
        "design": {
            "task": 1482,
            "round": "sae_dense_bridge (user-chat inline, 0 GPU)",
            "layer": LAYER,
            "sae": {
                "repo": "andyrdt/saes-qwen2.5-7b-instruct",
                "revision": "c37e53c4bb07127ad17ab88f28b93d4e87142e59",
                "k": SAE_K,
                "dict_size": DICT_SIZE,
            },
            "pooling": "mean ONLY — decode(mean-pooled features) == mean of per-token "
            "reconstructions, and the dense target v_x IS the mean-response "
            "vector; max/frac have no decode identity and are not scored here",
            "restriction": {
                "f_out": int(len(f_out)),
                "f_in": int(meta["f_in_n"]),
                "activity_floor": meta["activity_floor"],
            },
            "splits": {
                "n_train": meta["n_train"],
                "n_val": meta["n_val"],
                "n_holdout": meta["n_holdout"],
                "dropped_vs_split_union": meta["n_dropped_vs_split"],
            },
            "n_train_vs_d": {
                "sae_ctx": f"{meta['n_train']} vs {meta['d_ctx']}",
                "dense_ctx": f"{meta['n_train']} vs {H_DIM}",
            },
            "lambda_grid": "N1M.LAMBDAS_N1M = logspace(-3, 8, 23), selected on the 2,000-row "
            "sae_val carve (parent convention)",
            "dense_targets": "regenerated from the #779 n1M capture (v_x @ L19), joined on "
            "manifest ci — see scripts/issue1482_dense_targets_stage.py",
            "stated_deviations": [
                "CONTEXT ARM ONLY: the #1482 corpus is single-turn, so the prefix is "
                "degenerate; the standing prefix-AND-context both-arms rule is answered for "
                "the context arm here and for three-arm coverage by the #1738 multi-turn twin.",
                "Feature-space and dense-space R2 are NOT comparable numbers; the dense-space "
                "column is the common currency and the reconstruction ceilings are its "
                "reference line.",
                "The banked dense->dense numbers were fit on 943,444 rows; the matched-n "
                "(120,000) refit is the like-for-like row.",
            ],
        },
        "cells": cells,
        "ceilings": ceilings,
        "banked_reference": banked,
        "knn_setup": {
            "n_pred": KNN_N_PRED,
            "pool": "the full holdout dense targets",
            "n_pool": int(len(te)),
            "ks": list(KNN_KS),
            "chance_at_k": {str(k): k / float(len(te)) for k in KNN_KS},
            "metrics": ["euclidean", "cosine"],
        },
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_time_s": round(time.time() - t_start, 1),
            "device": "cpu",
            "arm_meta": {"sae_ctx": armA["_meta"], "dense_ctx": armB["_meta"]},
        },
    }
    (args.out / "sae_dense_bridge.json").write_text(json.dumps(doc, indent=2))
    # Per-row dense predictions are ~0.9 GB of fp16 tensors: an HF-data-repo artifact,
    # never a git artifact (eval_results/ is JSON/text only). Written to the work root;
    # --upload pushes it to the data repo.
    pred_path = args.base / "work" / "holdout_dense_predictions.npz"
    np.savez_compressed(
        pred_path,
        rows=order[te],
        target_v_x=Yd_te.astype(np.float16),
        ceiling_restricted=ceil_restricted.astype(np.float16),
        ceiling_full_dict=ceil_full.astype(np.float16),
        **dense_preds,
    )
    if args.upload:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        remote = "issue1482_error_analysis/analysis_tensors/sae_dense_bridge"
        url = hub._upload(
            pred_path,
            "superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=f"{remote}/{pred_path.name}",
            upload_as_file=True,
        )
        if not url:
            raise RuntimeError(f"upload returned no path for {pred_path}")
        missing = hub.verify_repo_paths_uploaded(
            HfApi(),
            "superkaiba1/explore-persona-space-data",
            [f"{remote}/{pred_path.name}"],
            path_in_repo=remote,
            repo_type="dataset",
        )
        if missing:
            raise RuntimeError(f"upload verify: missing {sorted(missing)}")
        doc["metadata"]["predictions_hf_path"] = f"{remote}/{pred_path.name}"
        (args.out / "sae_dense_bridge.json").write_text(json.dumps(doc, indent=2))
        logger.info("[upload] %s -> %s", pred_path.name, remote)

    logger.info("[done] %.0fs -> %s", time.time() - t_start, args.out / "sae_dense_bridge.json")
    for k, v in cells.items():
        logger.info("  %-38s feat=%s dense=%s", k, v["feature_space_r2"], v["dense_space_r2"])
    for k, v in ceilings.items():
        logger.info("  CEILING %-30s dense=%.4f", k, v["dense_space_r2"])


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)
