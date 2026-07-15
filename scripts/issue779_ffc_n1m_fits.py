#!/usr/bin/env python3
"""Issue #779 inline follow-up (``fitter-fair-comparison-n1m``): fits at n_train up
to ~1,000,000, provenance-aware, over the LMSYS+WildChat n1m corpus.

Extends the n50k fits (``issue779_ffc_n50k_fits.py``) along ``n`` to four
subset-fit points, reusing the SAME target (``v(x)`` mean-response profile), the
SAME map input (``cx_last`` last prompt token), the SAME variance-weighted
held-out R2 metric, and a BYTE-IDENTICAL val/test (the ORIGINAL round's
``fixed_split(5000, 3600, 400, 1000, 42)``, val/test index shas hard-asserted
equal to the pinned constants). New contexts (the n1m pool) enter TRAIN only.

Four subset-fit points (deterministic, provenance-aware selection; realized N =
min(target, available pool)):

  * ``lmsys_150k``   — 150,000 train, PURE LMSYS (orig-train + n1m-new lmsys only).
  * ``lmsys_500k``   — 500,000 train, PURE LMSYS.
  * ``mixed_500k``   — 500,000 train, STRATIFIED to the full pool's lmsys:wildchat
                       ratio (the corpus-mix control at matched n).
  * ``mixed_1m``     — the WHOLE mixed train pool (target 1,000,000; realized is the
                       full usable pool, ~orig-train 3.6k + 960k new).

Five predictors per point:

  * ``ridge``          — PRIMAL ridge, X^TX / X^TY accumulated STREAMING in fp64 on
                         the device over train-row blocks (never the (n, H) design
                         materialized at once), one eigh of (H, H), val-lambda over
                         ``LAMBDAS_N1M``. Numerically identical to the n50k primal
                         ridge, just block-accumulated.
  * ``mlp_w8192``      — full-dim MLP width 8192 (the protocol arm), MINIBATCHED
                         AdamW on the device (the n50k full-batch battery cannot
                         hold n=1M), internal-val early stop.
  * ``mlp_w32768``     — full-dim MLP width 32768 (the CAPACITY arm; flagged
                         ``capacity_arm: true`` in fit_meta), same minibatched fit.
  * ``residual_skip``  — primal ridge base + minibatched MLP (width 8192) on the
                         residual (strictly nests the linear map).
  * ``krr_nystrom``    — RBF kernel ridge via Nystrom (``--krr-nystrom-centers``
                         landmarks; exact KRR is a (n, n) kernel, infeasible at
                         n=1M), with the Nystrom feature Gram Phi^T Phi accumulated
                         STREAMING over train blocks. (gamma, lambda) val-selected.

Nystrom validation (gate): before the KRR fits, ``_validate_nystrom_vs_exact``
runs BOTH this driver's Nystrom fitter AND the n50k EXACT KRR
(``N50.fit_krr_exact``) on the SAME deterministic 50,000-row train slice + the
pinned val/test, and asserts ``|R2_nystrom - R2_exact| <= --krr-validate-tol``
(default 0.01) — a larger gap FAILS LOUD (the Nystrom fitter is biased). The
committed n50k exact anchor (0.8076 wide-grid / 0.8066 small-grid) is recorded
for reference. Requires ``--device cuda`` (the 50k^2 exact kernel).

Output (``eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json``): per
(point, predictor) whole-map R2 + mean cosine + 1000-resample bootstrap 95% CI +
fit_meta, the split (pinned + realized shas + byte-identical flag), the layer, the
Nystrom-vs-exact validation block, and reproducibility metadata. Per-(point,
predictor) checkpoint — ``--resume`` skips completed cells (guarded on layer +
seed so a cross-layer/seed resume never mixes rows).

The n1m capture (~82 GB) is NOT materialized whole: cx_last + v_x at the chosen
layer are STREAM-REDUCED from the HF capture chunks (download one chunk -> slice
the layer -> free). The combined per-layer X+Y (~28 GB at n~=963k) is held in
RAM for subset indexing — route this driver to a GPU pod / cpu-bigmem instance,
NOT the shared VM (>50 GB peak at the concat). Fail loud; NaN never coerced.
Refusal-safety: no context/rollout TEXT is ever printed or logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps land BEFORE numpy/torch import on the shared VM.
load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_generate_capture as N1G  # noqa: E402
import issue779_ffc_n50k_fits as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue779_ffc_n1m_fits")

PREDICTORS = ("ridge", "mlp_w8192", "mlp_w32768", "residual_skip", "krr_nystrom")
PREDICTOR_LABEL = {
    "ridge": "primal ridge (linear, streaming X^TX)",
    "mlp_w8192": "full-dim MLP (w=8192, protocol arm)",
    "mlp_w32768": "full-dim MLP (w=32768, capacity arm)",
    "residual_skip": "residual-skip (primal ridge + MLP w8192)",
    "krr_nystrom": "RBF KRR (Nystrom, streaming Phi^TPhi)",
}

# Ridge grid: LAMBDAS_N50K widened one more decade at the top for the larger n.
LAMBDAS_N1M = np.logspace(-3, 8, 23)

N_PASS_B = F.N_PASS_B  # 5000
N_VAL = 400
N_TEST = 1000
SPLIT_SEED = F.SPLIT_SEED  # 42
MLP_W_PROTOCOL = 8192
MLP_W_CAPACITY = 32768
RIDGE_BLOCK = 50_000  # train-row block for streaming X^TX / Phi^TPhi accumulation
MLP_BATCH = 4096
NYSTROM_VALIDATE_N = 50_000  # train slice for the Nystrom-vs-exact gate
NYSTROM_MAX_CENTERS_WARN = 20_000  # K_mm eigh at m > this may OOM on an 80GB GPU

# n50k committed exact-KRR anchor (reference; the gate is self-contained vs exact).
N50K_EXACT_R2_WIDEGRID = 0.8076
N50K_EXACT_R2_SMALLGRID = 0.8066

KRR_GAMMA_MULT = (1.0,)
KRR_LAMBDAS = (1e-1, 1e1)
KRR_KERNEL_BLOCK = 4096  # exact-KRR row-block (validation only)

# Fit points: (name, n_train_target, corpus_mode). mixed_1m target 1M realizes the
# full usable pool (~963.6k) — the new-pool target is 960k per the gen recipe.
FIT_POINTS = (
    ("lmsys_150k", 150_000, "lmsys"),
    ("lmsys_500k", 500_000, "lmsys"),
    ("mixed_500k", 500_000, "mixed"),
    ("mixed_1m", 1_000_000, "mixed"),
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison-n1m"
DEFAULT_ORIG_DIR = PROJECT_ROOT / "eval_results" / "issue_779" / "fitter-fair-comparison"


# ── data assembly (pass_b + stream-reduced n1m capture) + provenance ────────────


def _stream_n1m_layer(prefix: str, layer: int, local_dir: Path | None, cache_dir: Path):
    """Stream-reduce cx_last + v_x + ci at ``layer`` from the n1m capture chunks.

    Mirrors ``N50._stream_n50k_layer`` but ALSO returns the per-row global ci
    (manifest index) needed for provenance. local_dir given -> read staged chunks
    in place; else list the HF prefix (scoped list_repo_tree) and per chunk
    download -> mmap-slice the layer -> append -> DELETE (peak ~one chunk).
    """
    cx_parts: list[np.ndarray] = []
    vx_parts: list[np.ndarray] = []
    ci_parts: list[list[int]] = []

    def _consume(b) -> None:
        cx_parts.append(N50._slice_layer(b, "cx_last", layer))
        vx_parts.append(N50._slice_layer(b, "v_x", layer))
        ci_parts.append([int(x) for x in b["ci"]])

    if local_dir is not None:
        chunk_files = sorted(local_dir.glob("shard*_chunk*.pt"))
        if not chunk_files:
            raise FileNotFoundError(f"no n1m capture chunks under {local_dir}")
        for cp in chunk_files:
            b = F._mmap_load(cp)
            _consume(b)
            del b
        logger.info("[n1m] %d chunks (local)", len(chunk_files))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(
            f.path.rsplit("/", 1)[-1]
            for f in HfApi().list_repo_tree(
                C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(f, "size", None) is not None and f.path.endswith(".pt")
        )
        if not names:
            raise FileNotFoundError(f"no n1m capture chunks under HF {prefix}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            got = Path(
                hf_hub_download(
                    C.HF_DATA_REPO,
                    filename=f"{prefix}/{name}",
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
            )
            b = F._mmap_load(got)
            _consume(b)
            del b
            got.unlink()
            if (i + 1) % 25 == 0:
                logger.info("[n1m] streamed %d/%d chunks", i + 1, len(names))
        logger.info("[n1m] %d chunks (HF stream)", len(names))
    cx = np.concatenate(cx_parts)
    vx = np.concatenate(vx_parts)
    ci = np.array([c for part in ci_parts for c in part], dtype=np.int64)
    assert cx.shape[0] == vx.shape[0] == ci.shape[0], (cx.shape, vx.shape, ci.shape)
    return cx, vx, ci


def assemble(args, layer: int):
    """Combined (X=cx_last, Y=v_x) at ``layer`` + provenance + the pinned split.

    Rows [0, N_PASS_B) = pass_b (round-1, lmsys); [N_PASS_B, ...) = the n1m-new
    captured rows in ci order. ``prov`` marks each row lmsys|wildchat from the
    manifest (pass_b + orig-train are lmsys). Returns X, Y, prov, split, meta.
    """
    pb = N1G._load_pass_b_bundle(args.pass_b)
    for fld in ("cx_last", "v_x"):
        assert fld in pb, f"pass_b missing {fld}"
    assert int(pb["cx_last"].shape[0]) == N_PASS_B, (pb["cx_last"].shape[0], N_PASS_B)
    pb_X = N50._slice_layer(pb, "cx_last", layer)
    pb_Y = N50._slice_layer(pb, "v_x", layer)

    manifest_dir = N1G._resolve_manifest_dir(args)
    pool, man_meta = N1G.read_manifest_pool(manifest_dir)
    ci_to_corpus = {int(r["i"]): r["corpus"] for r in pool}

    local_dir = args.n1m_capture_dir if args.n1m_capture_dir else None
    new_X, new_Y, new_ci = _stream_n1m_layer(
        args.hf_prefix, layer, local_dir, args.out_dir / ".n1m_stream_cache"
    )
    # provenance for each captured new row (ci -> corpus); pass_b rows are lmsys.
    new_prov = np.array([ci_to_corpus[int(c)] for c in new_ci], dtype=object)

    X = np.concatenate([pb_X, new_X]).astype(np.float32)
    Y = np.concatenate([pb_Y, new_Y]).astype(np.float32)
    assert X.shape[1] == C.EXPECTED_HIDDEN and Y.shape[1] == C.EXPECTED_HIDDEN, (X.shape, Y.shape)
    prov = np.array(["lmsys"] * N_PASS_B + list(new_prov), dtype=object)
    assert prov.shape[0] == X.shape[0], (prov.shape, X.shape)

    pinned = N50._pinned_original_shas(args.orig_dir)
    r1_train, val, test = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    val_sha, test_sha = F._sha_ids(val), F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (
        f"n1m val sha {val_sha} != pinned original {pinned['val_sha256']} — NOT byte-identical"
    )
    assert test_sha == pinned["test_sha256"], (
        f"n1m test sha {test_sha} != pinned original {pinned['test_sha256']}"
    )
    assert (val < N_PASS_B).all() and (test < N_PASS_B).all(), "val/test must index the pass_b half"

    split = {
        "orig_train_ids": len(r1_train),
        "n_new_captured": int(new_X.shape[0]),
        "n_new_manifest": int(man_meta["n_new"]),
        "n_lmsys_manifest": int(man_meta["n_lmsys"]),
        "n_wildchat_manifest": int(man_meta["n_wildchat"]),
        "n_val": len(val),
        "n_test": len(test),
        "val_sha256": val_sha,
        "test_sha256": test_sha,
        "pinned_val_sha256": pinned["val_sha256"],
        "pinned_test_sha256": pinned["test_sha256"],
        "pinned_source": pinned["source"],
        "val_test_byte_identical_original": True,
        "layer": int(layer),
        "near_dupe": man_meta.get("near_dupe"),
        "manifest_new_prompt_sha256": man_meta.get("new_prompt_sha256"),
    }
    return X, Y, prov, r1_train, val, test, split


# ── provenance-aware deterministic subset selection ─────────────────────────────


def _pool_rows(prov, orig_train, n_total, val, test):
    """Row-index pools into the combined X. orig_train = fixed_split train ids
    (< N_PASS_B, all lmsys); new rows = [N_PASS_B, n_total). Excludes val/test by
    construction (orig_train disjoint from val/test; new rows >= N_PASS_B)."""
    new_rows = np.arange(N_PASS_B, n_total)
    new_lmsys = new_rows[prov[new_rows] == "lmsys"]
    new_wild = new_rows[prov[new_rows] == "wildchat"]
    lmsys_pool = np.concatenate([np.asarray(orig_train, dtype=np.int64), new_lmsys])
    full_pool = np.concatenate([np.asarray(orig_train, dtype=np.int64), new_rows])
    excl = set(int(x) for x in val) | set(int(x) for x in test)
    assert not (set(int(x) for x in full_pool) & excl), "train pool overlaps val/test"
    return {
        "lmsys": np.sort(lmsys_pool),
        "full": np.sort(full_pool),
        "orig_train": np.asarray(orig_train, dtype=np.int64),
        "new_lmsys": new_lmsys,
        "new_wildchat": new_wild,
    }


def select_train(pools, name, n_target, mode, seed):
    """Deterministic seeded subset of train rows for one fit point.

    mode='lmsys': sample from the lmsys pool. mode='mixed': stratified to the full
    pool's lmsys:wildchat ratio (or the whole full pool if n_target >= |full|).
    Returns (sorted train indices, selection diag)."""
    rng = np.random.default_rng(int(seed) + (abs(hash(name)) % 1_000_000))
    if mode == "lmsys":
        pool = pools["lmsys"]
        n = min(int(n_target), len(pool))
        sel = pool[rng.choice(len(pool), size=n, replace=False)]
        diag = {
            "mode": mode,
            "n_target": int(n_target),
            "n_realized": int(n),
            "n_lmsys": int(n),
            "n_wildchat": 0,
        }
        return np.sort(sel), diag
    # mixed
    full = pools["full"]
    lm = pools["lmsys"]  # lmsys rows in full (orig_train + new_lmsys)
    wild = pools["new_wildchat"]
    lmsys_frac = len(lm) / len(full) if len(full) else 0.0
    if int(n_target) >= len(full):
        sel = full  # whole mixed pool
        n_l, n_w = len(lm), len(wild)
    else:
        n = int(n_target)
        n_l = min(round(n * lmsys_frac), len(lm))
        n_w = min(n - n_l, len(wild))
        n_l = min(n - n_w, len(lm))  # rebalance if wildchat short
        lm_sel = lm[rng.choice(len(lm), size=n_l, replace=False)]
        w_sel = wild[rng.choice(len(wild), size=n_w, replace=False)]
        sel = np.concatenate([lm_sel, w_sel])
    diag = {
        "mode": mode,
        "n_target": int(n_target),
        "n_realized": len(sel),
        "n_lmsys": int(n_l),
        "n_wildchat": int(n_w),
        "full_lmsys_frac": round(float(lmsys_frac), 4),
    }
    return np.sort(sel), diag


# ── streaming primal ridge (fp64 X^TX / X^TY over train-row blocks) ─────────────


def _train_standardizer(X, Y, tr, dev, block):
    """Streaming train mean/std of X + mean of Y (fp64 on dev)."""
    H = X.shape[1]
    sum_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sumsq_x = torch.zeros(H, dtype=torch.float64, device=dev)
    sum_y = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
    n = 0
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = torch.as_tensor(X[idx], dtype=torch.float64, device=dev)
        yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev)
        sum_x += xb.sum(0)
        sumsq_x += (xb * xb).sum(0)
        sum_y += yb.sum(0)
        n += len(idx)
    xmu = sum_x / n
    # UNBIASED (N-1) variance to match N50._ridge_primal_multi_lambda's torch.std
    # default (unbiased=True) exactly — the standardization scale is NOT absorbed by
    # ridge, so the convention is load-bearing for streaming==N50 parity.
    denom = max(1, n - 1)
    var = (sumsq_x - n * xmu * xmu) / denom
    xsd = torch.clamp(var, min=0.0).sqrt() + 1e-9
    ymu = sum_y / n
    return xmu, xsd, ymu


def _ridge_streaming_multi_lambda(X, Y, tr, eval_idx_list, lambdas, dev, block):
    """Exact primal ridge (all lambdas off ONE eigh of the streamed (H,H) X^TX).

    Standardizes X on train stats, centers Y on train mean — numerically identical
    to N50._ridge_primal_multi_lambda, just block-accumulated so the (n, H) design
    is never materialized at once. Returns {lambda: [pred for each eval set]}."""
    xmu, xsd, ymu = _train_standardizer(X, Y, tr, dev, block)
    H = X.shape[1]
    A = torch.zeros((H, H), dtype=torch.float64, device=dev)
    XtY = torch.zeros((H, Y.shape[1]), dtype=torch.float64, device=dev)
    for s in range(0, len(tr), block):
        idx = tr[s : s + block]
        xb = (torch.as_tensor(X[idx], dtype=torch.float64, device=dev) - xmu) / xsd
        yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev) - ymu
        A += xb.T @ xb
        XtY += xb.T @ yb
    s_eig, U = torch.linalg.eigh(A)
    s_eig = torch.clamp(s_eig, min=0.0)
    UtXtY = U.T @ XtY
    evals_n = [
        (torch.as_tensor(X[e], dtype=torch.float64, device=dev) - xmu) / xsd for e in eval_idx_list
    ]
    out: dict[float, list[np.ndarray]] = {}
    for lam in lambdas:
        W = U @ (UtXtY / (s_eig + float(lam))[:, None])
        out[float(lam)] = [((En @ W) + ymu).cpu().numpy() for En in evals_n]
    return out, {"xmu": xmu, "xsd": xsd, "ymu": ymu}


def fit_ridge(X, Y, tr, val, te, lambdas, dev, block):
    preds, _ = _ridge_streaming_multi_lambda(X, Y, tr, [val, te], lambdas, dev, block)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(preds[float(lam)][0], Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    return preds[best_lam][1], {
        "n_train": len(tr),
        "selection": "val-lambda (primal, streaming)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "ridge_block": int(block),
    }


# ── minibatched MLP (single large-n fit; the full-batch battery cannot hold 1M) ──


def _fit_mlp_minibatch(
    X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, *, base_tr=None, base_te=None
):
    """Single full-dim MLP (GELU, MSE) trained MINIBATCHED with AdamW + internal-val
    early stop. Standardizes X on train stats; centers Y on train mean (or fits the
    residual Y - base_* when base_tr/base_te given, for residual_skip). Predicts te
    minibatched. FLOP-bound single large-n fit — NOT a many-cell loop (the
    vectorized_mlp_skill helper batches CELLS, a different regime)."""
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    H = X.shape[1]
    D = Y.shape[1]
    # standardizer on train X (streamed to keep the (n,H) copy off the device)
    xmu, xsd, ymu = _train_standardizer(X, Y, tr, dev, RIDGE_BLOCK)
    xmu_c = xmu.to(torch.float32)
    xsd_c = xsd.to(torch.float32)
    ymu_c = ymu.to(torch.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_val = max(1, round(0.1 * len(tr)))
    va_local, tr_local = perm[:n_val], perm[n_val:]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(H, width), torch.nn.GELU(), torch.nn.Linear(width, D)
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=F.MLP_WD)

    # residual base (Y - ridge_base) is precomputed per TRAIN row, aligned to `tr`;
    # scatter it into a combined-row lookup so a minibatch's rows index it directly.
    _base_lookup = None
    if base_tr is not None:
        _base_lookup = np.zeros((X.shape[0], D), dtype=np.float32)
        _base_lookup[tr] = base_tr  # only train rows are read for the residual target
    best_val = float("inf")
    best_state = None
    bad = 0
    epochs_ran = 0
    for ep in range(max_epochs):
        net.train()
        ep_perm = rng.permutation(len(tr_local))
        for bs in range(0, len(tr_local), batch):
            rows = tr[tr_local[ep_perm[bs : bs + batch]]]
            xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
            tb = torch.as_tensor(Y[rows], dtype=torch.float32, device=dev) - ymu_c
            if _base_lookup is not None:
                tb = tb - torch.as_tensor(_base_lookup[rows], dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = ((net(xb) - tb) ** 2).mean()
            loss.backward()
            opt.step()
        # internal-val
        net.eval()
        with torch.no_grad():
            vsum, vcnt = 0.0, 0
            for bs in range(0, len(va_local), batch):
                rows = tr[va_local[bs : bs + batch]]
                xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
                tb = torch.as_tensor(Y[rows], dtype=torch.float32, device=dev) - ymu_c
                if _base_lookup is not None:
                    tb = tb - torch.as_tensor(_base_lookup[rows], dtype=torch.float32, device=dev)
                vsum += float(((net(xb) - tb) ** 2).sum())
                vcnt += rows.shape[0] * D
            vloss = vsum / max(1, vcnt)
        epochs_ran = ep + 1
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= F.MLP_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    preds = []
    with torch.no_grad():
        for bs in range(0, len(te), batch):
            rows = te[bs : bs + batch]
            xb = (torch.as_tensor(X[rows], dtype=torch.float32, device=dev) - xmu_c) / xsd_c
            out = net(xb) + ymu_c
            preds.append(out.cpu().numpy())
    pred_te = np.concatenate(preds).astype(np.float32)
    if base_te is not None:
        pred_te = pred_te + base_te
    return pred_te, {
        "width": int(width),
        "lr": float(lr),
        "epochs_ran": int(epochs_ran),
        "batch": int(batch),
        "best_val_mse": float(best_val),
    }


def fit_mlp(X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, *, capacity_arm=False):
    pred_te, meta = _fit_mlp_minibatch(X, Y, tr, te, width, lr, max_epochs, batch, seed, dev)
    meta["n_train"] = len(tr)
    meta["capacity_arm"] = bool(capacity_arm)
    return pred_te, meta


def fit_residual_skip(X, Y, tr, val, te, lambdas, width, lr, max_epochs, batch, seed, dev, block):
    """Primal ridge base + minibatched MLP on the residual (strictly nests linear)."""
    preds, _ = _ridge_streaming_multi_lambda(X, Y, tr, [val, tr, te], lambdas, dev, block)
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        vr2 = PR._pooled_r2(preds[float(lam)][0], Y[val])
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    base_tr = preds[best_lam][1]  # ridge pred on train (aligned to tr)
    base_te = preds[best_lam][2]  # ridge pred on test
    pred, mmeta = _fit_mlp_minibatch(
        X, Y, tr, te, width, lr, max_epochs, batch, seed, dev, base_tr=base_tr, base_te=base_te
    )
    return pred, {
        "n_train": len(tr),
        "base_ridge_lambda": best_lam,
        "residual_mlp_width": int(width),
        "lr": float(lr),
        "epochs_ran": mmeta["epochs_ran"],
        "batch": int(batch),
    }


# ── chunked Nystrom RBF KRR (streaming Phi^TPhi over train blocks) ───────────────


def _nystrom_inv_sqrt(landmarks, gamma, dev, eig_floor=1e-10):
    """K_mm^{-1/2} whitener (m, m) fp64 on dev."""
    Z = torch.as_tensor(np.asarray(landmarks), dtype=torch.float64, device=dev)
    K_mm = torch.exp(-gamma * torch.cdist(Z, Z) ** 2)
    w, V = torch.linalg.eigh(K_mm)
    w = torch.clamp(w, min=eig_floor)
    return V @ torch.diag(w.rsqrt()) @ V.T  # (m, m)


def _nystrom_features_block(Xblock, landmarks_t, gamma, inv_sqrt):
    """Phi_block = exp(-gamma ||Xb - Z||^2) @ inv_sqrt, (block, m) fp64 on dev."""
    Xb = torch.as_tensor(np.asarray(Xblock), dtype=torch.float64, device=inv_sqrt.device)
    K_bm = torch.exp(-gamma * torch.cdist(Xb, landmarks_t) ** 2)
    return K_bm @ inv_sqrt


def fit_krr_nystrom(X, Y, tr, val, te, *, m_centers, gamma_mult, lambdas, seed, dev, block):
    """Nystrom RBF KRR, (gamma, lambda) val-selected. Phi^TPhi accumulated STREAMING
    over train blocks so the (ntr, m) feature matrix is never materialized whole.
    Raw X + median-heuristic gamma (matches N50.fit_krr_exact for the validation)."""
    tr = np.asarray(tr, dtype=np.int64)
    Xtr_sub = np.asarray(X[tr[: min(len(tr), 4000)]], dtype=np.float64)  # gamma est subsample
    base_gamma = F.median_heuristic_gamma(Xtr_sub, np.random.default_rng(seed + 1))
    m = int(min(m_centers, len(tr)))
    if m > NYSTROM_MAX_CENTERS_WARN:
        logger.warning(
            "[krr] m_centers=%d > %d — K_mm eigh (m,m) may OOM on an 80GB GPU",
            m,
            NYSTROM_MAX_CENTERS_WARN,
        )
    lm_rows = tr[np.random.default_rng(seed).choice(len(tr), size=m, replace=False)]
    landmarks = np.asarray(X[lm_rows], dtype=np.float64)
    # center Y on train mean (streamed)
    _, _, ymu = _train_standardizer(X, Y, tr, dev, block)
    grid, best = [], None
    for gm in gamma_mult:
        gamma = base_gamma * gm
        inv_sqrt = _nystrom_inv_sqrt(landmarks, gamma, dev)
        landmarks_t = torch.as_tensor(landmarks, dtype=torch.float64, device=dev)
        G = torch.zeros((m, m), dtype=torch.float64, device=dev)
        PhiY = torch.zeros((m, Y.shape[1]), dtype=torch.float64, device=dev)
        for s in range(0, len(tr), block):
            idx = tr[s : s + block]
            phi = _nystrom_features_block(X[idx], landmarks_t, gamma, inv_sqrt)  # (blk, m)
            yb = torch.as_tensor(Y[idx], dtype=torch.float64, device=dev) - ymu
            G += phi.T @ phi
            PhiY += phi.T @ yb
        a, Q = torch.linalg.eigh(G)
        a = torch.clamp(a, min=0.0)
        QtPhiY = Q.T @ PhiY
        phi_val = _nystrom_features_block(X[val], landmarks_t, gamma, inv_sqrt)
        phi_te = _nystrom_features_block(X[te], landmarks_t, gamma, inv_sqrt)
        for lam in lambdas:
            W = Q @ (QtPhiY / (a + float(lam))[:, None])  # (m, D)
            pred_val = (phi_val @ W + ymu).cpu().numpy()
            pred_te = (phi_te @ W + ymu).cpu().numpy()
            val_r2 = PR._pooled_r2(pred_val, Y[val])
            grid.append(
                {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                }
            )
            if best is None or (np.isfinite(val_r2) and val_r2 > best["val_r2"]):
                best = {
                    "gamma_mult": float(gm),
                    "gamma": float(gamma),
                    "lambda": float(lam),
                    "val_r2": float(val_r2),
                    "pred_te": pred_te,
                }
        del G, PhiY, inv_sqrt, landmarks_t, phi_val, phi_te
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    assert best is not None
    return best["pred_te"], {
        "n_train": len(tr),
        "kernel": "RBF Nystrom (streaming Phi^TPhi)",
        "m_centers": m,
        "base_gamma": float(base_gamma),
        "selected": {k: best[k] for k in ("gamma_mult", "gamma", "lambda", "val_r2")},
        "val_grid": grid,
    }


def _validate_nystrom_vs_exact(
    X, Y, pools, val, te, *, m_centers, gamma_mult, krr_lambdas, seed, dev, tol
):
    """Run Nystrom AND exact KRR on the SAME 50k train slice; assert R2 agreement.

    A gap > tol means the Nystrom fitter is numerically biased vs exact — FAIL LOUD
    (not a shrug), per the brief. Requires cuda (the 50k^2 exact kernel)."""
    if dev.type != "cuda":
        raise SystemExit("--validate-krr requires --device cuda (the exact 50k^2 KRR kernel)")
    pool = pools["lmsys"]  # pure-lmsys 50k slice (comparable to the n50k exact anchor)
    n = min(NYSTROM_VALIDATE_N, len(pool))
    tr = np.sort(pool[np.random.default_rng(seed + 7).choice(len(pool), size=n, replace=False)])
    logger.info("[krr-validate] exact vs Nystrom (m=%d) on n=%d ...", m_centers, n)
    ts = time.time()
    pred_ex, meta_ex = N50.fit_krr_exact(
        X,
        Y,
        tr,
        val,
        te,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        block=KRR_KERNEL_BLOCK,
        seed=seed,
        dev=dev,
    )
    r2_ex = PR._pooled_r2(pred_ex, Y[te])
    pred_ny, meta_ny = fit_krr_nystrom(
        X,
        Y,
        tr,
        val,
        te,
        m_centers=m_centers,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        seed=seed,
        dev=dev,
        block=RIDGE_BLOCK,
    )
    r2_ny = PR._pooled_r2(pred_ny, Y[te])
    gap = abs(r2_ny - r2_ex)
    logger.info(
        "[krr-validate] exact R2=%.4f  nystrom R2=%.4f  gap=%.4f (tol %.4f, %.0fs)",
        r2_ex,
        r2_ny,
        gap,
        tol,
        time.time() - ts,
    )
    if gap > tol:
        raise SystemExit(
            f"Nystrom-vs-exact KRR gap {gap:.4f} > tol {tol:.4f} at n={n} (exact {r2_ex:.4f}, "
            f"nystrom {r2_ny:.4f}) — the Nystrom fitter is biased; raise --krr-nystrom-centers"
        )
    return {
        "n": int(n),
        "m_centers": int(m_centers),
        "exact_r2": float(r2_ex),
        "nystrom_r2": float(r2_ny),
        "gap": float(gap),
        "tol": float(tol),
        "committed_n50k_exact_r2_widegrid": N50K_EXACT_R2_WIDEGRID,
        "committed_n50k_exact_r2_smallgrid": N50K_EXACT_R2_SMALLGRID,
        "exact_selected": meta_ex.get("selected"),
        "nystrom_selected": meta_ny.get("selected"),
    }


def _curve(pred_te, Y_te, n_boot, seed) -> dict:
    r2, cos = F._recon_point(pred_te, Y_te)
    ci = F._bootstrap_recon_ci(pred_te, Y_te, n_boot, seed)
    return {"whole_map_r2": float(r2), "mean_cosine": float(cos), "bootstrap_ci": ci}


def _fit_one_predictor(name, X, Y, tr, val, test, lambdas, gamma_mult, krr_lambdas, args, dev):
    """Dispatch one predictor fit; returns (pred_te, fit_meta)."""
    if name == "ridge":
        return fit_ridge(X, Y, tr, val, test, lambdas, dev, args.ridge_block)
    if name == "mlp_w8192":
        return fit_mlp(
            X,
            Y,
            tr,
            test,
            MLP_W_PROTOCOL,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
        )
    if name == "mlp_w32768":
        return fit_mlp(
            X,
            Y,
            tr,
            test,
            MLP_W_CAPACITY,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
            capacity_arm=True,
        )
    if name == "residual_skip":
        return fit_residual_skip(
            X,
            Y,
            tr,
            val,
            test,
            lambdas,
            MLP_W_PROTOCOL,
            args.mlp_lr,
            args.mlp_max_epochs,
            args.mlp_batch,
            args.seed,
            dev,
            args.ridge_block,
        )
    return fit_krr_nystrom(
        X,
        Y,
        tr,
        val,
        test,
        m_centers=args.krr_nystrom_centers,
        gamma_mult=gamma_mult,
        lambdas=krr_lambdas,
        seed=args.seed,
        dev=dev,
        block=args.ridge_block,
    )


def _run_fit_points(
    results,
    want_points,
    want_pred,
    point_by_name,
    X,
    Y,
    pools,
    val,
    test,
    lambdas,
    gamma_mult,
    krr_lambdas,
    args,
    dev,
):
    """Per (fit point x predictor): select the provenance-aware subset, fit, curve,
    and checkpoint. ``--resume`` skips completed (point, predictor) cells."""
    for pn in want_points:
        _, n_target, mode = point_by_name[pn]
        tr, sel_diag = select_train(pools, pn, n_target, mode, args.seed)
        results["per_point"].setdefault(pn, {"selection": sel_diag, "predictors": {}})
        results["per_point"][pn]["selection"] = sel_diag
        logger.info("[point %s] n_train=%d (%s) — %s", pn, len(tr), mode, sel_diag)
        for name in want_pred:
            if args.resume and name in results["per_point"][pn]["predictors"]:
                logger.info("[resume] %s/%s present; skip", pn, name)
                continue
            ts = time.time()
            pred_te, meta = _fit_one_predictor(
                name, X, Y, tr, val, test, lambdas, gamma_mult, krr_lambdas, args, dev
            )
            curve = _curve(pred_te, Y[test], args.n_boot, args.seed)
            curve["fit_meta"] = meta
            curve["wall_time_s"] = round(time.time() - ts, 1)
            results["per_point"][pn]["predictors"][name] = curve
            C.write_json_atomic(args.out_json, results)
            logger.info(
                "[done] %s/%s: whole-map R2=%.4f mean-cos=%.4f (%.0fs)",
                pn,
                name,
                curve["whole_map_r2"],
                curve["mean_cosine"],
                curve["wall_time_s"],
            )


def _smoke() -> int:
    """CPU numeric-sanity smoke (synthetic; no capture data, no GPU).

    Covers: (1) the byte-identical val/test split shas (recomputed fixed_split ==
    the pinned N50 constants); (2) provenance-aware subset selection (lmsys mode
    keeps only lmsys rows; mixed mode preserves the full-pool ratio; realized =
    min(target, pool)); (3) streaming primal ridge == N50's primal ridge on the
    same synthetic; (4) Nystrom-vs-exact KRR agreement on a small synthetic RBF
    problem (the fitter numeric sanity); (5) the minibatched MLP + residual-skip
    bodies run end-to-end and beat the mean baseline."""
    dev = torch.device("cpu")
    torch.set_num_threads(4)
    logger.info("[smoke] fits CPU numeric-sanity (split-sha + selection + ridge + nystrom + mlp)")

    # (1) byte-identical split shas.
    _r1, val_idx, test_idx = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    assert F._sha_ids(val_idx) == N50.ORIG_VAL_SHA256, "val split sha drift"
    assert F._sha_ids(test_idx) == N50.ORIG_TEST_SHA256, "test split sha drift"

    # (2) provenance-aware selection on a synthetic combined layout.
    n_total = N_PASS_B + 200
    prov = np.array(["lmsys"] * N_PASS_B + ["lmsys"] * 150 + ["wildchat"] * 50, dtype=object)
    orig_train, val, test = F.fixed_split(
        N_PASS_B, N_PASS_B - N_VAL - N_TEST, N_VAL, N_TEST, SPLIT_SEED
    )
    pools = _pool_rows(prov, orig_train, n_total, val, test)
    assert (prov[pools["lmsys"]] == "lmsys").all(), "lmsys pool contains non-lmsys rows"
    tr_l, dl = select_train(pools, "lmsys_x", 1_000_000, "lmsys", 0)
    assert dl["n_realized"] == len(pools["lmsys"]) and dl["n_wildchat"] == 0, dl
    assert (prov[tr_l] == "lmsys").all(), "lmsys-mode selection leaked wildchat"
    _tr_m, dm = select_train(pools, "mixed_x", 100, "mixed", 0)
    assert dm["n_realized"] == 100 and dm["n_wildchat"] > 0, dm  # ratio-matched, wildchat present
    frac = dm["n_lmsys"] / dm["n_realized"]
    assert abs(frac - dm["full_lmsys_frac"]) < 0.1, (frac, dm)
    _tr_full, df = select_train(pools, "mixed_all", 1_000_000, "mixed", 0)
    assert df["n_realized"] == len(pools["full"]), df  # whole pool when target >= pool

    # (3)-(5) numeric checks on a small synthetic RBF-friendly problem.
    rng = np.random.default_rng(7)
    n, H, D = 500, 40, 8
    Wt = rng.standard_normal((H, D)) * 0.3
    Xs = rng.standard_normal((n, H)).astype(np.float32)
    Ys = (np.tanh(Xs @ Wt.astype(np.float32)) + 0.05 * rng.standard_normal((n, D))).astype(
        np.float32
    )
    tr = np.arange(0, 380)
    vl = np.arange(380, 420)
    ts = np.arange(420, 500)
    lambdas = np.logspace(-2, 3, 6)

    # (3) streaming ridge == N50 primal ridge (same math, block-accumulated).
    stream_preds, _ = _ridge_streaming_multi_lambda(Xs, Ys, tr, [ts], lambdas, dev, block=64)
    ref = N50._ridge_primal_multi_lambda(Xs[tr], Ys[tr], [Xs[ts]], lambdas, dev)
    for lam in lambdas:
        d = float(np.max(np.abs(stream_preds[float(lam)][0] - ref[float(lam)][0])))
        assert d < 1e-4, f"streaming ridge != N50 primal ridge at lambda={lam}: max|diff|={d:.2e}"

    # (4) Nystrom-vs-exact agreement (m=200 centers of 380 train).
    pred_ex, _ = N50.fit_krr_exact(
        Xs, Ys, tr, vl, ts, gamma_mult=(1.0,), lambdas=(1e-1, 1e1), block=128, seed=0, dev=dev
    )
    pred_ny, _ = fit_krr_nystrom(
        Xs,
        Ys,
        tr,
        vl,
        ts,
        m_centers=200,
        gamma_mult=(1.0,),
        lambdas=(1e-1, 1e1),
        seed=0,
        dev=dev,
        block=64,
    )
    r2_ex, r2_ny = PR._pooled_r2(pred_ex, Ys[ts]), PR._pooled_r2(pred_ny, Ys[ts])
    assert abs(r2_ex - r2_ny) < 0.05, (
        f"nystrom-vs-exact gap {abs(r2_ex - r2_ny):.4f} (ex {r2_ex:.4f}, ny {r2_ny:.4f})"
    )

    # (5) minibatched MLP + residual-skip bodies run and beat the mean baseline (R2>0).
    pred_mlp, mm = fit_mlp(Xs, Ys, tr, ts, 64, 3e-3, 40, 64, 0, dev)
    assert PR._pooled_r2(pred_mlp, Ys[ts]) > 0.0, "MLP body did not beat the mean baseline"
    pred_res, _ = fit_residual_skip(Xs, Ys, tr, vl, ts, lambdas, 64, 3e-3, 40, 64, 0, dev, 64)
    assert PR._pooled_r2(pred_res, Ys[ts]) > 0.0, (
        "residual-skip body did not beat the mean baseline"
    )

    logger.info(
        "[smoke] PASS: split shas byte-id; select (lmsys/mixed-ratio %.2f/whole); "
        "ridge==N50 (<1e-4); nystrom~exact (ex %.3f ny %.3f); MLP+residual R2>0 (mlp %d ep)",
        dm["full_lmsys_frac"],
        r2_ex,
        r2_ny,
        mm["epochs_ran"],
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 n1m fits (up to n_train=1,000,000).")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", default=",".join(p[0] for p in FIT_POINTS))
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=F.BOOT_N)
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--mlp-batch", type=int, default=MLP_BATCH)
    ap.add_argument("--mlp-lr", type=float, default=3e-4)
    ap.add_argument("--ridge-block", type=int, default=RIDGE_BLOCK)
    ap.add_argument("--krr-nystrom-centers", type=int, default=8192)
    ap.add_argument("--krr-gamma-mult", default=",".join(str(g) for g in KRR_GAMMA_MULT))
    ap.add_argument("--krr-lambdas", default=",".join(str(x) for x in KRR_LAMBDAS))
    ap.add_argument("--krr-validate-tol", type=float, default=0.01)
    ap.add_argument("--no-validate-krr", action="store_true", help="skip the Nystrom-vs-exact gate")
    ap.add_argument("--pass-b", type=Path, default=N1G.PASS_B_LOCAL)
    ap.add_argument("--orig-dir", type=Path, default=DEFAULT_ORIG_DIR)
    ap.add_argument("--manifest-from-hf", action="store_true")
    ap.add_argument("--n1m-capture-dir", type=Path, default=None)
    ap.add_argument("--hf-prefix", default=f"{N1G.HF_PREFIX}/final_token_capture")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="CPU numeric-sanity smoke (synthetic; no data/GPU)"
    )
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    want_points = [p.strip() for p in args.points.split(",") if p.strip()]
    want_pred = [p.strip() for p in args.predictors.split(",") if p.strip()]
    for p in want_pred:
        if p not in PREDICTORS:
            raise ValueError(f"unknown predictor {p!r}")
    point_by_name = {p[0]: p for p in FIT_POINTS}
    for pn in want_points:
        if pn not in point_by_name:
            raise ValueError(f"unknown fit point {pn!r}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_json is None:
        args.out_json = args.out_dir / "n1m_fits.json"

    results = json.loads(args.out_json.read_text()) if args.out_json.exists() else {}
    if results.get("layer") is not None and results["layer"] != args.layer:
        raise SystemExit(
            f"--out-json {args.out_json} was written for layer {results['layer']} but "
            f"--layer={args.layer}; refusing to mix cross-layer rows"
        )
    if results.get("seed") is not None and results["seed"] != args.seed:
        raise SystemExit(
            f"--out-json {args.out_json} written for seed {results['seed']}, not --seed={args.seed}"
        )

    t0 = time.time()
    X, Y, prov, orig_train, val, test, split = assemble(args, args.layer)
    pools = _pool_rows(prov, orig_train, X.shape[0], val, test)
    logger.info(
        "assembled: %d contexts (%d lmsys pool, %d full pool), val=%d test=%d (L%d, %.0fs)",
        X.shape[0],
        len(pools["lmsys"]),
        len(pools["full"]),
        len(val),
        len(test),
        args.layer,
        time.time() - t0,
    )
    lambdas = LAMBDAS_N1M
    gamma_mult = tuple(float(g) for g in args.krr_gamma_mult.split(",") if g.strip())
    krr_lambdas = tuple(float(x) for x in args.krr_lambdas.split(",") if x.strip())

    validation = results.get("nystrom_validation")
    if "krr_nystrom" in want_pred and not args.no_validate_krr and validation is None:
        validation = _validate_nystrom_vs_exact(
            X,
            Y,
            pools,
            val,
            test,
            m_centers=args.krr_nystrom_centers,
            gamma_mult=gamma_mult,
            krr_lambdas=krr_lambdas,
            seed=args.seed,
            dev=dev,
            tol=args.krr_validate_tol,
        )

    results.setdefault("per_point", {})
    results.update(
        {
            "layer": int(args.layer),
            "seed": int(args.seed),
            "split": split,
            "lambda_grid": {"n": len(lambdas), "min": float(lambdas[0]), "max": float(lambdas[-1])},
            "krr_grid": {
                "gamma_mult": list(gamma_mult),
                "lambdas": list(krr_lambdas),
                "nystrom_centers": int(args.krr_nystrom_centers),
            },
            "nystrom_validation": validation,
            "predictor_labels": PREDICTOR_LABEL,
            "fit_points": {p[0]: {"n_train_target": p[1], "corpus_mode": p[2]} for p in FIT_POINTS},
            "note": (
                "n_train up to 1,000,000 rerun of fitter-fair-comparison over the LMSYS-exhaust + "
                "WildChat-balance n1m corpus. val/test BYTE-IDENTICAL to the ORIGINAL round "
                "(asserted vs pinned shas). 4 points: lmsys_150k/500k (pure), mixed_500k "
                "(ratio-matched control), mixed_1m (full pool). ridge=streaming primal; mlp mini; "
                "residual-skip=ridge+MLP; krr=Nystrom (validated vs exact at n=50k)."
            ),
            "metadata": C.reproducibility_metadata(
                {"script": "issue779_ffc_n1m_fits", "device": args.device}
            ),
        }
    )
    C.write_json_atomic(args.out_json, results)

    _run_fit_points(
        results,
        want_points,
        want_pred,
        point_by_name,
        X,
        Y,
        pools,
        val,
        test,
        lambdas,
        gamma_mult,
        krr_lambdas,
        args,
        dev,
    )
    logger.info("wrote %s (%.0fs total)", args.out_json, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
