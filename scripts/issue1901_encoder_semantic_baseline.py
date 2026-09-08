#!/usr/bin/env python3
"""Issue #1901 inline round ``encbaseline`` — encoder-embedding semantic baseline
for the context->answer map (user-chat inline GPU override, 2026-09-08).

Question (user, verbatim): "I want to run a baseline which does a regression from
context embedding to answer embedding with some encoder only embedding model, to
see if it's just semantic similarity."

Formally: let x be the context text, y the answer text, v_C(x) in R^3584 the
Qwen-2.5-7B-Instruct layer-19 last-prompt-token state, v_A(y) in R^3584 the
mean-over-answer-tokens state, and e(.) in R^1024 a mean-pooled encoder-only
sentence embedding. Does ridge(v_C -> v_A) carry information beyond what a
generic semantic encoder reads off the same text?

Arms (all ridge, val-selected lambda, identical rows, identical metrics):

  arm1_lm        v_C            -> v_A        3584 -> 3584   the paper's map
  arm2_enc_lm    e(x)           -> v_A        1024 -> 3584   input-swap ablation
  arm3_enc_enc   e(x)           -> e(y)       1024 -> 1024   fully LM-free
  arm4_pca_lm    PCA_1024(v_C)  -> v_A        1024 -> 3584   dimension control
  arm5_resid     v_C            -> v_A - Mhat_enc e(x)       residual control
  arm6_cos0      no fit: cos(e(x), e(y))                     zero-parameter floor

arm5 freezes Mhat_enc from TRAIN only and builds test residuals from that frozen
map (no double-dipping). arm5 reports R^2 only (retrieval against a residual
target is not meaningful). arm6 reports retrieval only (there is no prediction in
a target space).

Solver note (load-bearing). ``issue779_fitter_fair_comparison.gram_fit_apply``
factorizes the (n_train, n_train) DUAL Gram in float64 — correct for the #779
regime (n <= ~5k) but a 20 GB matrix at n_train=50,000. Every arm here has
n >> d, so this module uses the PRIMAL (d, d) normal-equation form instead and
GATES it against ``gram_fit_apply`` at a small n where both are computable
(``--solver-gate``, |dR2| < 1e-4, the #779 SOLVER_EQUIV_TOL convention).

Lambda selection is VAL-based for every arm, never GCV: ``gram_fit_apply``'s own
docstring records that GCV's (n_train - dof)^2 denominator degenerates at
n_train ~= H, and the n=5,000 rung sits right there against H=3,584.

Encoders (encoder-only, mean-pooled, L2-normalized):
  bge   BAAI/bge-large-en-v1.5          1024-dim, headline (user choice)
  e5    intfloat/multilingual-e5-large  1024-dim, robustness twin — ~8.6% of the
        LMSYS prompts in this pool are heavily non-ASCII, and an English-only
        encoder would deflate the baseline exactly where the round is trying to
        avoid a false negative.
  fake  deterministic sha-derived pseudo-embeddings, smoke only, no download.

Context string fed to the encoder = the FULL Qwen chat-templated prompt, so the
baseline sees byte-identical input to what the LM conditioned on (user decision,
/clarify gate 2026-09-08).

Data: the n1m pool (``issue779_monitoring/fitter-fair-comparison-n1m``). Capture
chunks carry cx_last / v_x / ci / prompts at layers [14, 19, 26], 500 rows per
chunk; the matching raw_completions chunk carries {ci, prompt, response} rows.
Both sides are chunk-aligned by construction, so no manifest join and no
LMSYS re-derivation is needed.

DELIBERATE DEVIATION, disclosed. The paper's banked figure scores on the pinned
pass_b test rows. The pass_b bundle is tensors-only (prompts stripped, see
``issue779_ffc_n1m_generate_capture._valtest_prompts_from_round1``) and its
ORIGINAL answer TEXTS are not banked on the HF data repo, so arm3 cannot be
evaluated there. This round therefore refits EVERY arm, arm1 included, on one
n1m-internal split, which makes the between-arm comparison — the actual claim —
internally exact. The banked single-draw anchors (#779 n50k exact ridge R^2
0.8076; the n1m CSLS acc@1 ladder test-1000 0.848) are reported alongside as
external context, never as a pass/fail gate on a different row set.

Phases:
  smoke       2 chunks x fake embedder, end-to-end, asserts schema, rc=0
  stage       stream N capture + raw_completions chunks -> one local npz + jsonl
  embed       encoder forward over templated prompts and responses -> npz
  fit-score   the six arms at the n-rungs -> eval JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("issue1901_encbaseline")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
CAPTURE_PREFIX = f"{HF_PREFIX}/final_token_capture"
RAWCOMP_PREFIX = f"{HF_PREFIX}/raw_completions"

LAYER = 19
CAPTURE_LAYERS = (14, 19, 26)
HIDDEN = 3584
ROWS_PER_CHUNK = 500

LM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # tokenizer only, for the chat template
EMBEDDERS = {
    "bge": "BAAI/bge-large-en-v1.5",
    "e5": "intfloat/multilingual-e5-large",
    "fake": "fake",
}
FAKE_DIM = 64
EMB_MAX_TOKENS = 512

N_TEST = 1_000
N_VAL = 400
SPLIT_SEED = 42
RUNGS = (5_000, 10_000, 25_000, 50_000)
LAMBDAS = np.logspace(-2, 6, 17)
WHITEN_LAM = 0.1
K_CSLS = 10
SOLVER_EQUIV_TOL = 1e-4

# External anchors (banked, single-draw regime). Context only, never a gate.
ANCHOR_N50K_RIDGE_R2 = 0.8076  # issue779_ffc_n1m_fits.N50K_EXACT_R2_WIDEGRID
ANCHOR_N1M_CSLS_ACC1_TEST1000 = 0.848  # issue1901_avgpool_scaleup docstring


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[encbaseline] {msg}", flush=True)


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def stage(args) -> None:
    """Stream capture + raw_completions chunks, keep layer 19, delete each chunk."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    from explore_persona_space.orchestrate import hub

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    li = CAPTURE_LAYERS.index(LAYER)

    cx_parts, vx_parts, ci_parts, prompts, responses = [], [], [], [], []
    scratch = out / "_scratch"
    scratch.mkdir(exist_ok=True)

    # The n1m capture prefix is 32 shards x 60 chunks x 500 rows = 960,000 rows
    # (measured 2026-09-08). One shard caps the pool at 30,000, which is below the
    # 50,000 rung, so the pool is assembled ACROSS shards.
    t0 = time.time()
    shards = [int(s) for s in str(args.shards).split(",") if s.strip()]
    for shard in shards:
        for idx in range(args.n_chunks):
            name = f"shard{shard:02d}_chunk{idx:04d}"
            # Retries ride hub.retry_transient (Retry-After aware, wall-clock budgeted).
            # ONLY a genuine absence ends a shard: a transient 5xx that survived the
            # retry budget must RAISE, never silently truncate the pool and shrink the
            # rungs underneath the fit (fail-fast rule).
            try:
                cap = hub.retry_transient(
                    lambda: hf_hub_download(
                        HF_REPO,
                        f"{CAPTURE_PREFIX}/{name}.pt",
                        repo_type="dataset",
                        local_dir=scratch,
                    ),
                    what=f"download {name}.pt",
                )
                rcp = hub.retry_transient(
                    lambda: hf_hub_download(
                        HF_REPO,
                        f"{RAWCOMP_PREFIX}/{name}.json",
                        repo_type="dataset",
                        local_dir=scratch,
                    ),
                    what=f"download {name}.json",
                )
            except (EntryNotFoundError, RepositoryNotFoundError) as exc:
                _log(f"shard{shard:02d} exhausted at chunk {idx} ({type(exc).__name__})")
                break

            d = torch.load(cap, map_location="cpu", weights_only=False)
            rows = json.load(open(rcp))["rows"]

            cx = d["cx_last"][:, li, :].to(torch.float32).numpy()
            vx = d["v_x"][:, li, :].to(torch.float32).numpy()
            ci = np.asarray(d["ci"], dtype=np.int64)
            pr = list(d["prompts"])
            assert cx.shape == vx.shape == (len(ci), HIDDEN), (cx.shape, vx.shape, len(ci))
            assert len(pr) == len(ci), (len(pr), len(ci))

            # Chunk-alignment is the whole reason no manifest join is needed. Assert it.
            rc_ci = np.asarray([r["ci"] for r in rows], dtype=np.int64)
            assert np.array_equal(rc_ci, ci), f"{name}: raw_completions ci != capture ci"
            rc_prompt = [r["prompt"] for r in rows]
            assert rc_prompt == pr, f"{name}: raw_completions prompt text != capture prompts"

            cx_parts.append(cx)
            vx_parts.append(vx)
            ci_parts.append(ci)
            prompts.extend(pr)
            responses.extend(r["response"] for r in rows)

            Path(cap).unlink(missing_ok=True)
            Path(rcp).unlink(missing_ok=True)
            if (idx + 1) % 20 == 0:
                n = sum(len(c) for c in ci_parts)
                _log(
                    f"shard{shard:02d}: {idx + 1} chunks / {n} rows total ({time.time() - t0:.0f}s)"
                )

    cx = np.concatenate(cx_parts)
    vx = np.concatenate(vx_parts)
    ci = np.concatenate(ci_parts)
    assert cx.shape[0] == len(prompts) == len(responses) == ci.shape[0]
    assert len(np.unique(ci)) == ci.shape[0], "duplicate ci in the staged pool"

    np.savez(out / "vectors.npz", cx=cx, vx=vx, ci=ci)
    with open(out / "texts.jsonl", "w") as fh:
        for c, p, r in zip(ci.tolist(), prompts, responses):
            fh.write(json.dumps({"ci": c, "prompt": p, "response": r}, ensure_ascii=False) + "\n")
    n_chunks_done = len(ci_parts)
    expected = len(shards) * args.n_chunks * ROWS_PER_CHUNK
    _log(
        f"STAGE OK rows={cx.shape[0]} chunks={n_chunks_done} shards={shards} "
        f"(expected_rows={expected}) -> {out}"
    )
    if cx.shape[0] != expected:
        # Loud, not silent: a short pool shrinks the rungs and changes what the
        # arms are comparable to. Reported here and re-stated in the eval JSON.
        _log(
            f"WARNING short pool: {cx.shape[0]} rows vs {expected} expected — the "
            "chunk universe ended early; rungs will be capped to the realized pool"
        )


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------
def _fake_embed(texts: list[str]) -> np.ndarray:
    out = np.empty((len(texts), FAKE_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        raw = (h * (FAKE_DIM * 4 // len(h) + 1))[: FAKE_DIM * 4]
        out[i] = np.asarray(struct.unpack(f"<{FAKE_DIM}f", raw), dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def _encode(texts: list[str], model_id: str, device: str, batch: int) -> np.ndarray:
    """Mean-pooled, L2-normalized encoder embeddings (bare transformers, no ST dep).

    Mirrors scripts/issue1739_textembed_baseline.py's pooling head, which
    reproduces the sentence-transformers client to cosine ~0.9999.
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    vecs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(
                texts[i : i + batch],
                padding=True,
                truncation=True,
                max_length=EMB_MAX_TOKENS,
                return_tensors="pt",
            ).to(device)
            hid = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
            pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            vecs.append(pooled.cpu().numpy().astype(np.float32))
            if (i // batch) % 50 == 0:
                _log(
                    f"  encoded {i + len(enc['input_ids'])}/{len(texts)} ({time.time() - t0:.0f}s)"
                )
    return np.concatenate(vecs)


def embed(args) -> None:
    out = Path(args.out)
    rows = [json.loads(line) for line in open(out / "texts.jsonl")]
    prompts = [r["prompt"] for r in rows]
    responses = [r["response"] for r in rows]

    model_id = EMBEDDERS[args.embedder]
    if args.embedder == "fake":
        ctx_texts = [f"<TEMPLATE>{p}" for p in prompts]
    else:
        from transformers import AutoTokenizer

        lm_tok = AutoTokenizer.from_pretrained(LM_MODEL)
        ctx_texts = [
            lm_tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        _log(f"templated context example (first 200 chars): {ctx_texts[0][:200]!r}")

    if args.embedder == "fake":
        e_ctx, e_ans = _fake_embed(ctx_texts), _fake_embed(responses)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"embedding {len(ctx_texts)} contexts with {model_id} on {dev}")
        e_ctx = _encode(ctx_texts, model_id, dev, args.batch)
        _log(f"embedding {len(responses)} answers with {model_id} on {dev}")
        e_ans = _encode(responses, model_id, dev, args.batch)

    # English-only mask (heavy non-ASCII => not English), for the subset read.
    def _heavy_nonascii(s: str) -> bool:
        return (sum(1 for ch in s if ord(ch) > 127) / max(len(s), 1)) > 0.15

    en_mask = np.asarray(
        [not (_heavy_nonascii(p) or _heavy_nonascii(r)) for p, r in zip(prompts, responses)]
    )
    np.savez(out / f"emb_{args.embedder}.npz", e_ctx=e_ctx, e_ans=e_ans, en_mask=en_mask)
    _log(
        f"EMBED OK {args.embedder} dim={e_ctx.shape[1]} rows={e_ctx.shape[0]} "
        f"english_rows={int(en_mask.sum())}/{en_mask.size}"
    )


# ---------------------------------------------------------------------------
# primal ridge (n >> d)
# ---------------------------------------------------------------------------
def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """SS_tot on TRUE's own mean — the #779 percontext_recon convention."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def primal_ridge_fit_apply(Xtr, Ytr, X_eval_list, dev, val):
    """Ridge via the (d, d) normal equations, val-selected lambda.

    Standardization and centering match _factorize/_gcv_solve exactly (train mean
    and POPULATION std on X, train mean on Y) so the solver gate can compare
    against gram_fit_apply bitwise-closely. Returns (preds, selected_lambda).
    """
    dev = torch.device(dev)
    X = torch.as_tensor(np.asarray(Xtr), dtype=torch.float64, device=dev)
    Y = torch.as_tensor(np.asarray(Ytr), dtype=torch.float64, device=dev)
    xmu, xsd = X.mean(0), X.std(0, unbiased=False) + 1e-9
    Xn = (X - xmu) / xsd
    ymu = Y.mean(0)
    Yc = Y - ymu

    A = Xn.T @ Xn  # (d, d)
    B = Xn.T @ Yc  # (d, D)
    w, V = torch.linalg.eigh(A)
    w = torch.clamp(w, min=0.0)
    VtB = V.T @ B

    def _predict(Xev, lam):
        W = V @ (VtB / (w + lam)[:, None])
        Xe = torch.as_tensor(np.asarray(Xev), dtype=torch.float64, device=dev)
        return (((Xe - xmu) / xsd) @ W + ymu).cpu().numpy()

    Xval, Yval = val
    best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
    for lam in LAMBDAS:
        vr2 = _pooled_r2(_predict(Xval, float(lam)), Yval)
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    return [_predict(E, best_lam) for E in X_eval_list], best_lam


def solver_gate(args) -> dict:
    """Primal vs the #779 dual gram_fit_apply at a small n where both compute."""
    import issue779_fitter_fair_comparison as F

    rng = np.random.default_rng(0)
    n, d, D = 900, 64, 32
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, D))
    Y = X @ W + 0.3 * rng.normal(size=(n, D))
    Xtr, Ytr = X[:600], Y[:600]
    Xva, Yva = X[600:750], Y[600:750]
    Xte, Yte = X[750:], Y[750:]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lam_grid_backup = F.LAMBDAS
    F.LAMBDAS = LAMBDAS  # compare on the SAME grid, else the gate tests the grid
    try:
        (dual_pred,), dual_lam = F.gram_fit_apply(Xtr, Ytr, [Xte], dev, val=(Xva, Yva))
    finally:
        F.LAMBDAS = lam_grid_backup
    (prim_pred,), prim_lam = primal_ridge_fit_apply(Xtr, Ytr, [Xte], dev, val=(Xva, Yva))

    r2_dual, r2_prim = _pooled_r2(dual_pred, Yte), _pooled_r2(prim_pred, Yte)
    delta = abs(r2_dual - r2_prim)
    verdict = "PASS" if delta < SOLVER_EQUIV_TOL and dual_lam == prim_lam else "FAIL"
    res = {
        "verdict": verdict,
        "r2_dual": r2_dual,
        "r2_primal": r2_prim,
        "abs_delta_r2": delta,
        "lambda_dual": dual_lam,
        "lambda_primal": prim_lam,
        "tol": SOLVER_EQUIV_TOL,
    }
    _log(f"SOLVER GATE {verdict}: dual R2={r2_dual:.8f} primal R2={r2_prim:.8f} d={delta:.2e}")
    return res


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def _whitener(Ytr: np.ndarray, lam: float = WHITEN_LAM):
    """Shrunk train-target Cholesky: z = L^-1 (v - mu). The #2202 convention."""
    Y = np.asarray(Ytr, dtype=np.float64)
    mu = Y.mean(0)
    Z = Y - mu
    C = (Z.T @ Z) / Z.shape[0]
    d = C.shape[0]
    C = (1.0 - lam) * C + lam * (np.trace(C) / d) * np.eye(d)
    L = np.linalg.cholesky(C)
    return mu, L


def _whiten(V: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular

    return solve_triangular(L, (np.asarray(V, dtype=np.float64) - mu).T, lower=True).T


def _top1_csls(pred: np.ndarray, pool: np.ndarray, mu, L, k: int = K_CSLS) -> dict:
    """Whitened-cosine + two-sided CSLS top-1, query i's true target at pool i.

    Reuses issue1901_metric_battery.csls_scores (the pinned cross-domain form).
    """
    import issue1901_metric_battery as MB

    P = _whiten(pred, mu, L)
    Q = _whiten(pool, mu, L)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    S = P @ Q.T
    Sc = MB.csls_scores(S, k=k)
    n = Sc.shape[0]
    true = np.diag(Sc).copy()
    # strict top-1: ties fail (the battery's convention)
    n_better_or_tied = (Sc >= true[:, None]).sum(axis=1) - 1
    hit = n_better_or_tied == 0
    acc1 = float(hit.mean())
    se = float(np.sqrt(acc1 * (1 - acc1) / n))
    return {
        "acc_at_1": acc1,
        "n_query": n,
        "n_pool": n,
        "chance_at_1": 1.0 / n,
        "ci95": {"lo": max(0.0, acc1 - 1.96 * se), "hi": min(1.0, acc1 + 1.96 * se)},
    }


# ---------------------------------------------------------------------------
# fit-score
# ---------------------------------------------------------------------------
def fit_score(args) -> None:
    out = Path(args.out)
    vec = np.load(out / "vectors.npz")
    cx, vx = vec["cx"], vec["vx"]
    emb = np.load(out / f"emb_{args.embedder}.npz")
    e_ctx, e_ans, en_mask = emb["e_ctx"], emb["e_ans"], emb["en_mask"]
    n = cx.shape[0]
    assert e_ctx.shape[0] == n, (e_ctx.shape, n)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    test_idx, val_idx, train_pool = (
        perm[:N_TEST],
        perm[N_TEST : N_TEST + N_VAL],
        perm[N_TEST + N_VAL :],
    )
    rungs = [r for r in RUNGS if r <= train_pool.size]
    if not rungs:
        rungs = [train_pool.size]
    _log(
        f"split: n={n} test={test_idx.size} val={val_idx.size} train_pool={train_pool.size} rungs={rungs}"
    )

    gate = solver_gate(args) if args.solver_gate else {"verdict": "SKIPPED"}
    if args.solver_gate and gate["verdict"] != "PASS":
        raise RuntimeError(f"solver equivalence gate FAILED: {gate}")

    # PCA basis for arm4, fit on the largest rung's train rows only.
    # PCA via covariance eigh, NOT svd of the (n, 3584) matrix: same basis, and
    # O(n d^2 + d^3) instead of a 50k x 3584 SVD (vectorize-many-cell-fits rule).
    pca_fit_rows = train_pool[: max(rungs)]
    Xp = cx[pca_fit_rows].astype(np.float64)
    pca_mu = Xp.mean(0)
    Zc = Xp - pca_mu
    evals, evecs = np.linalg.eigh((Zc.T @ Zc) / Zc.shape[0])
    pca_B = evecs[:, ::-1][:, : e_ctx.shape[1]]  # (3584, d_enc), descending variance
    cx_pca = ((cx.astype(np.float64) - pca_mu) @ pca_B).astype(np.float32)
    _log(f"PCA basis {pca_B.shape} fit on {pca_fit_rows.size} train rows")

    results = []
    for n_train in rungs:
        tr = train_pool[:n_train]
        _log(f"--- rung n_train={n_train} ---")
        row = {"n_train": int(n_train), "arms": {}}

        specs = [
            ("arm1_lm", cx, vx, "v_C -> v_A"),
            ("arm2_enc_lm", e_ctx, vx, "e(x) -> v_A"),
            ("arm3_enc_enc", e_ctx, e_ans, "e(x) -> e(y)"),
            ("arm4_pca_lm", cx_pca, vx, f"PCA_{e_ctx.shape[1]}(v_C) -> v_A"),
        ]
        # arms 1/2/4 share the target v_A, so their whitener is identical. Cache it:
        # each build is a 3584^2 covariance over n_train rows, minutes at the 50k rung.
        whiteners: dict[str, tuple] = {}

        def _cached_whitener(key: str, Ytr_arr):
            if key not in whiteners:
                t = time.time()
                whiteners[key] = _whitener(Ytr_arr)
                _log(f"  whitener[{key}] built in {time.time() - t:.0f}s")
            return whiteners[key]

        preds = {}
        for arm, X, Y, label in specs:
            t0 = time.time()
            (p_test,), lam = primal_ridge_fit_apply(
                X[tr], Y[tr], [X[test_idx]], dev, val=(X[val_idx], Y[val_idx])
            )
            preds[arm] = p_test
            r2 = _pooled_r2(p_test, Y[test_idx])
            mu, L = _cached_whitener("enc" if arm == "arm3_enc_enc" else "lm", Y[tr])
            ret = _top1_csls(p_test, Y[test_idx], mu, L)
            r2_en = _pooled_r2(p_test[en_mask[test_idx]], Y[test_idx][en_mask[test_idx]])
            row["arms"][arm] = {
                "label": label,
                "d_in": int(X.shape[1]),
                "d_out": int(Y.shape[1]),
                "n_train": int(n_train),
                "lambda": lam,
                "r2": r2,
                "r2_english_only": r2_en,
                "n_english_test": int(en_mask[test_idx].sum()),
                "top1": ret,
                "wall_s": round(time.time() - t0, 1),
            }
            _log(
                f"  {arm:14s} R2={r2:.4f} top1={ret['acc_at_1']:.4f} lam={lam:.3g} ({time.time() - t0:.0f}s)"
            )

        # arm5: residual of v_A on the FROZEN train-fit encoder map
        (enc_tr_pred, enc_te_pred), lam_enc = primal_ridge_fit_apply(
            e_ctx[tr], vx[tr], [e_ctx[tr], e_ctx[test_idx]], dev, val=(e_ctx[val_idx], vx[val_idx])
        )
        (enc_va_pred,), _ = primal_ridge_fit_apply(
            e_ctx[tr], vx[tr], [e_ctx[val_idx]], dev, val=(e_ctx[val_idx], vx[val_idx])
        )
        r_tr = vx[tr].astype(np.float64) - enc_tr_pred
        r_va = vx[val_idx].astype(np.float64) - enc_va_pred
        r_te = vx[test_idx].astype(np.float64) - enc_te_pred
        (p_res,), lam_res = primal_ridge_fit_apply(
            cx[tr], r_tr, [cx[test_idx]], dev, val=(cx[val_idx], r_va)
        )
        r2_res = _pooled_r2(p_res, r_te)
        row["arms"]["arm5_resid"] = {
            "label": "v_C -> (v_A - Mhat_enc e(x)), Mhat_enc frozen on train",
            "d_in": int(cx.shape[1]),
            "d_out": int(vx.shape[1]),
            "n_train": int(n_train),
            "lambda": lam_res,
            "lambda_frozen_encoder_map": lam_enc,
            "r2": r2_res,
            "top1": None,
            "note": "retrieval omitted: a residual target has no meaningful pool",
        }
        _log(f"  arm5_resid     R2={r2_res:.4f} (residual variance the encoder cannot reach)")

        # arm6: zero-parameter cosine floor in encoder space
        mu_e, L_e = _cached_whitener("enc", e_ans[tr])
        ret6 = _top1_csls(e_ctx[test_idx], e_ans[test_idx], mu_e, L_e)
        row["arms"]["arm6_cos0"] = {
            "label": "no fit: cos(e(x), e(y))",
            "r2": None,
            "top1": ret6,
            "note": "zero-parameter semantic-similarity floor",
        }
        _log(f"  arm6_cos0      top1={ret6['acc_at_1']:.4f} (no fit)")

        results.append(row)

    payload = {
        "issue": 1901,
        "round": "encbaseline",
        "embedder": args.embedder,
        "embedder_model": EMBEDDERS[args.embedder],
        "layer": LAYER,
        "lm_model": LM_MODEL,
        "context_string": "full Qwen chat template, add_generation_prompt=True",
        "target": "single-draw v_A (mean over answer tokens), n1m pool",
        "split": {
            "n_pool": int(n),
            "n_test": int(test_idx.size),
            "n_val": int(val_idx.size),
            "n_train_pool": int(train_pool.size),
            "seed": SPLIT_SEED,
        },
        "lambda_selection": "val-based (never GCV; GCV degenerates at n_train ~= H)",
        "solver": "primal (d,d) normal equations",
        "solver_gate": gate,
        "retrieval": {
            "metric": "whitened cosine + two-sided CSLS",
            "csls_k": K_CSLS,
            "whiten_lambda": WHITEN_LAM,
            "rank": "strict top-1; ties fail",
        },
        "external_anchors": {
            "note": "different row set — context only, never a gate (see module docstring)",
            "n50k_exact_ridge_r2": ANCHOR_N50K_RIDGE_R2,
            "n1m_csls_acc1_test1000": ANCHOR_N1M_CSLS_ACC1_TEST1000,
        },
        "per_rung": results,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dest = Path(args.eval_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))
    _log(f"FIT-SCORE OK -> {dest}")


# ---------------------------------------------------------------------------
def smoke(args) -> None:
    args.n_chunks, args.shards, args.embedder, args.solver_gate = 2, "0", "fake", True
    args.out = args.out or "/tmp/encbaseline_smoke"
    args.eval_out = str(Path(args.out) / "smoke_results.json")
    stage(args)
    embed(args)
    global RUNGS, N_TEST, N_VAL
    RUNGS, N_TEST, N_VAL = (300,), 200, 100
    fit_score(args)
    payload = json.loads(Path(args.eval_out).read_text())
    assert payload["solver_gate"]["verdict"] == "PASS", payload["solver_gate"]
    arms = payload["per_rung"][0]["arms"]
    for a in ("arm1_lm", "arm2_enc_lm", "arm3_enc_enc", "arm4_pca_lm", "arm5_resid", "arm6_cos0"):
        assert a in arms, f"missing arm {a}"
    _log("SMOKE OK — schema, solver gate, and all six arms present")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["smoke", "stage", "embed", "fit-score"])
    ap.add_argument("--out", default="/workspace/data/issue_1901/encbaseline")
    ap.add_argument(
        "--eval-out", default=str(PROJECT_ROOT / "eval_results/issue_1901/encbaseline/results.json")
    )
    ap.add_argument("--n-chunks", type=int, default=60, help="max chunks PER shard")
    ap.add_argument("--shards", default="0,1", help="comma list of shard ids")
    ap.add_argument("--embedder", choices=sorted(EMBEDDERS), default="bge")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--solver-gate", action="store_true")
    args = ap.parse_args()
    {"smoke": smoke, "stage": stage, "embed": embed, "fit-score": fit_score}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
