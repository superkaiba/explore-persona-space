"""Issue #1073 free re-analysis (0 GPU-h, CPU, ANALYSIS-ONLY).

Two deliverables over the ALREADY-CAPTURED #1073 decode-regime tensors — NO new
rollouts, NO training, NO judge/API calls:

  A  Characterize the single-draw vs 10-rollout-averaged prediction gap: is the
     ~0.046-0.078 R2 gap just irreducible per-draw sampling noise (linear map
     predicts the noise-averaged output), or does draw-specific deviation carry
     context-predictable structure (possibly nonlinear)?
       1. Noise decomposition (between-context vs within-context draw variance).
       2. Oracle noise ceiling (leave-one-out 9-mean predicting a held-out draw)
          + achieved single-draw ridge R2 + expected-under-pure-noise R2
          (algebraic AND the direct avg10-map-predicts-single-draw read).
       3. Deviation predictability: c_x -> signed draw deviation (ridge is
          analytically W=0; empirical confirm + MLP + shuffled null), plus the
          non-vacuous companion c_x -> per-context rollout dispersion.

  B  Characterize the adverse-tail contexts (the 27-32% with per-context greedy
     adequacy Delta < 0): Spearman(Delta, covariate) BH-corrected, logistic tail
     membership AUC + standardized coefficients, tail rate by dispersion bins,
     and the 10 most severe-tail context indices with structural safe labels.

Reuses the #1073 loaders (issue1073_common / issue1073_capture) + the FFC ridge
(issue779_arm_headline.GramRidge) + the FFC batched MLP recipe
(issue779_batch2.batched_mlp_fit, hyperparameters verbatim: width 512, lr 1e-3,
wd 1e-4, max_epochs 300, patience 20). All compute is CPU + vectorized: the
noise / ceiling / dispersion reductions are batched tensor ops over the
(N, 10, H) per-rollout store, streamed one read-out layer at a time.

Context hygiene: the corpus is real LMSYS user prompts (jailbreak/explicit rows).
NO raw prompt/completion text is ever loaded into agent context or written to
any output — token counts / truncation flags come from the store's span_lens,
and duplicate / language / code-block flags are structural regex computed in
this process and emitted as booleans/ints only (never the text).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue1073_capture as CAP  # noqa: E402
import issue1073_common as I  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_batch2 import MLPFitGroup, batched_mlp_fit  # noqa: E402
from issue779_fitter_fair_comparison import (  # noqa: E402
    _apply,
    _cross_kernel,
    _factorize,
    _vty_ymu,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue1073_gaptail")

torch.set_num_threads(int(os.environ.get("EPS_VM_THREAD_CAP", "8")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READOUT_LAYERS = [14, 17, 19, 26, 27]
GEN_MAX_TOKENS = 1024  # SP_GREEDY / stochastic max_tokens -> truncation proxy
N_ROLLOUTS = 10
LAMBDAS = np.logspace(-2, 4, 13)  # FFC / GramRidge ridge grid
# FFC val-selected split (3600/400/1000, seed 42) — matches heldout_recon_arms.val_lambda_robustness
SPLIT_SEED = 42
N_TEST, N_VAL = 1000, 400
# FFC MLP recipe (issue779_fitter_fair_comparison D1 primary), hyperparameters verbatim
MLP_WIDTH, MLP_LR, MLP_WD, MLP_MAX_EPOCHS, MLP_PATIENCE = 512, 1e-3, 1e-4, 300, 20
MLP_PCA_K = 64  # PCA head for high-dim (signed-deviation) targets; scalar targets skip it
SHUFFLE_SEED = 0
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_1073"

STAGE = PROJECT_ROOT / "data" / "issue_1073_gaptail" / "_hf"
STORE_DIR = STAGE / "issue1073_decode_regime" / "analysis_tensors" / "v_store"
PREDS_DIR = STAGE / "issue1073_decode_regime" / "analysis_tensors" / "predictions"
COVERAGE = STAGE / "issue1073_decode_regime" / "analysis_tensors" / "reductions" / "coverage.pt"
BUNDLE = STAGE / "issue779_monitoring" / "analysis_tensors" / "pass_b" / "train_context_vectors.pt"


# ── loaders ───────────────────────────────────────────────────────────────────


def _repro_meta(extra: dict) -> dict:
    return I.reproducibility_metadata(extra)


def load_keep() -> np.ndarray:
    cov = torch.load(COVERAGE, weights_only=False, map_location="cpu")
    keep_mask = (~cov["greedy_empty"]) & (~cov["stoch_any_empty"])
    keep = np.where(keep_mask.numpy())[0]
    logger.info(
        "[keep] %d contexts (greedy_empty=%d stoch_any_empty=%d)",
        keep.size,
        int(cov["greedy_empty"].sum()),
        int(cov["stoch_any_empty"].sum()),
    )
    return keep


def stoch_matrix(li: int, keep: np.ndarray) -> np.ndarray:
    """(N_kept, 10, H) fp64 per-rollout matrix at one layer (fail-loud fill)."""
    pos_of = {int(ci): k for k, ci in enumerate(keep.tolist())}
    seen = np.zeros((len(keep), N_ROLLOUTS), dtype=bool)
    v = None
    for p, shard in CAP.iter_shards(STORE_DIR, "stoch10"):
        li_pos = list(shard["layers"]).index(li)
        sl = shard["summ"][:, li_pos, :].to(torch.float64).numpy()
        if v is None:
            v = np.zeros((len(keep), N_ROLLOUTS, sl.shape[1]))
        for row, (ci, ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                assert not seen[k, ri], f"duplicate rollout (ci={ci}, ri={ri}) in {p}"
                seen[k, ri] = True
                v[k, ri] = sl[row]
    assert v is not None and bool(seen.all()), "stoch10 store fill incomplete"
    return v


def load_span_lens(keep: np.ndarray) -> dict:
    """Per-context greedy + per-(context,rollout) stoch span_lens + prompt_lens.

    span_len is the teacher-forced response span token count (structural — no
    text). truncation proxy = span_len >= GEN_MAX_TOKENS.
    """
    pos_of = {int(ci): k for k, ci in enumerate(keep.tolist())}
    n = len(keep)
    greedy_span = np.full(n, -1, dtype=np.int64)
    prompt_len = np.full(n, -1, dtype=np.int64)
    stoch_span = np.full((n, N_ROLLOUTS), -1, dtype=np.int64)
    for _p, shard in CAP.iter_shards(STORE_DIR, "greedy"):
        sp = shard["span_lens"].numpy()
        pl = shard["prompt_lens"].numpy()
        for row, (ci, _ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                greedy_span[k] = sp[row]
                prompt_len[k] = pl[row]
    for _p, shard in CAP.iter_shards(STORE_DIR, "stoch10"):
        sp = shard["span_lens"].numpy()
        for row, (ci, ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                stoch_span[k, ri] = sp[row]
    assert (greedy_span >= 0).all() and (stoch_span >= 0).all() and (prompt_len >= 0).all()
    return {"greedy_span": greedy_span, "stoch_span": stoch_span, "prompt_len": prompt_len}


def load_bundle_cx(keep: np.ndarray) -> dict:
    """cx_last + the pass_b prompt list.

    The pinned pass_b bundle predates the builder's 'prompts' field (schema
    drift, gotchas.md #1073), so prompts are regenerated deterministically from
    the recorded gated source via the canonical sha-checked parent loader
    (I._load_or_regen_prompts). Prompt text is used ONLY for structural
    duplicate/regex covariates and is never emitted.
    """
    # mmap: the bundle carries cx_last + cx_mean + v_x (~6 GB fp32 total); only
    # cx_last is needed, so mmap keeps peak RSS bounded to the per-layer slices.
    blob = torch.load(BUNDLE, weights_only=False, map_location="cpu", mmap=True)
    cx_last = blob["cx_last"]  # (N, L, H) mmap-backed
    n = cx_last.shape[0]
    if "prompts" in blob:
        prompts = blob["prompts"]
    else:
        prompts = I._load_or_regen_prompts(BUNDLE, n, blob["source"], smoke=False)
    sha16 = I.prompt_list_sha256(prompts)[:16]
    logger.info("[prompts] n=%d sha16=%s (expected b45816298923e17a)", len(prompts), sha16)
    assert len(prompts) == n, (len(prompts), n)
    prompts_kept = [prompts[i] for i in keep.tolist()]
    return {"cx_last": cx_last, "prompts_kept": prompts_kept, "prompt_sha16": sha16}


def cx_layer(cx_last: torch.Tensor, li: int, keep: np.ndarray) -> np.ndarray:
    return cx_last[:, li, :].to(torch.float64).numpy()[keep]


def load_delta() -> dict:
    with open(EVAL_RESULTS_DIR / "target_agreement.json") as f:
        ta = json.load(f)
    return {int(k[1:]): np.asarray(v["dv4_delta_ctx"]) for k, v in ta["per_layer"].items()}


def load_achieved() -> dict:
    with open(EVAL_RESULTS_DIR / "heldout_recon_arms.json") as f:
        hra = json.load(f)
    return hra


# ── vectorized primitives ──────────────────────────────────────────────────────


def _pooled_r2(pred: np.ndarray, true: np.ndarray, mean: np.ndarray | None = None) -> float:
    mu = true.mean(0) if mean is None else mean
    sse = float(((true - pred) ** 2).sum())
    sst = float(((true - mu) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def rollout_dispersion(v: np.ndarray) -> np.ndarray:
    """Per-context mean pairwise cosine DISTANCE among the 10 rollout vectors."""
    vn = v / (np.linalg.norm(v, axis=2, keepdims=True) + 1e-12)  # (n,10,H)
    g = np.einsum("nih,njh->nij", vn, vn)  # (n,10,10)
    offdiag_sum = g.sum((1, 2)) - N_ROLLOUTS  # subtract diagonal (=1 each)
    mean_offdiag_cos = offdiag_sum / (N_ROLLOUTS * (N_ROLLOUTS - 1))
    return 1.0 - mean_offdiag_cos


# ── Deliverable A: noise decomposition + ceiling ───────────────────────────────


def noise_decomposition(v: np.ndarray) -> dict:
    """Total single-draw variance = between-context + within-context draw variance.

    SS terms summed over H dims; variances = SS / count.
    """
    n = v.shape[0]
    single = v.reshape(n * N_ROLLOUTS, -1)  # (n*10, H)
    grand = single.mean(0)
    vbar = v.mean(1)  # (n, H) per-context 10-mean
    ss_total = float(((single - grand) ** 2).sum())
    ss_between = float(N_ROLLOUTS * ((vbar - grand) ** 2).sum())
    ss_within = float(((v - vbar[:, None, :]) ** 2).sum())
    var_single = ss_total / (n * N_ROLLOUTS)
    var_between = ss_between / (n * N_ROLLOUTS)
    var_within = ss_within / (n * N_ROLLOUTS)
    var_vbar = float(((vbar - grand) ** 2).sum()) / n  # variance of the 10-mean target
    sigma2_within = ss_within / (n * (N_ROLLOUTS - 1))  # unbiased per-draw noise variance
    return {
        "ss_total_singledraw": ss_total,
        "ss_between_context": ss_between,
        "ss_within_context": ss_within,
        "var_singledraw": var_single,
        "var_between_context": var_between,
        "var_within_context": var_within,
        "var_10mean_target": var_vbar,
        "sigma2_within_unbiased": sigma2_within,
        "noise_share_within_over_total": ss_within / max(ss_total, 1e-12),
    }


def oracle_ceiling(v: np.ndarray) -> dict:
    """Pooled R2 of the LOO 9-mean predicting the held-out draw (context-conditioned
    Bayes-optimal predictor of a single draw, estimated by the other-9 mean).

    Denominator SST = single-draw deviations from the grand single-draw mean.
    """
    n = v.shape[0]
    s = v.sum(1)  # (n, H)
    m = (s[:, None, :] - v) / (N_ROLLOUTS - 1)  # (n, 10, H) leave-one-out 9-mean
    single = v.reshape(n * N_ROLLOUTS, -1)
    grand = single.mean(0)
    sse = float(((v - m) ** 2).sum())
    sst = float(((single - grand) ** 2).sum())
    return {
        "ceiling_r2_pooled": 1.0 - sse / max(sst, 1e-12),
        "sse_loo9_vs_draw": sse,
        "sst_singledraw": sst,
    }


def expected_singledraw_r2(nd: dict, r2_10mean_achieved: float) -> dict:
    """Closed-form pure-noise expectation.

    Model: v_ij = mu_i + eps_ij, map f predicts mu_i. Then
      MSE_10mean = E||mu-f||^2 + sigma2_w/10
      MSE_single = E||mu-f||^2 + sigma2_w
    so MSE_single = MSE_10mean + sigma2_w*(9/10), giving
      R2_single_expected = 1 - [ (1-R2_10mean)*Var(vbar) + sigma2_w*9/10 ] / Var(v).
    (sigma2_w here is the total within SS / (n*(10-1)) expressed as a per-draw
    variance summed over H, matching the SS-summed R2 denominators.)
    """
    var_vbar = nd["var_10mean_target"]
    var_single = nd["var_singledraw"]
    # sigma2_within as an SS-summed-over-H per-draw variance, in the same units
    # as var_single/var_vbar (both are per-observation SS/count over H):
    sigma2_w = nd["var_within_context"] * (N_ROLLOUTS / (N_ROLLOUTS - 1))
    mse_10mean = (1.0 - r2_10mean_achieved) * var_vbar
    mse_single = mse_10mean + sigma2_w * (N_ROLLOUTS - 1) / N_ROLLOUTS
    return {
        "r2_singledraw_expected_purenoise": 1.0 - mse_single / max(var_single, 1e-12),
        "sigma2_within_ss_summed": sigma2_w,
    }


def avg10map_predicts_singledraw(fact, kev_val, kev_te, v, tr, val, te) -> dict:
    """Direct, assumption-free pure-noise read on the VAL-SELECTED split.

    Fit the avg10 map (target = the 10-rollout mean vbar_i) with val-selected
    lambda off the SHARED layer factorization, then evaluate its TEST-context
    predictions against BOTH the 10-mean target and the single draws. R2 vs the
    single draws IS 'the expected single-draw R2 if the map predicts the
    noise-averaged output'. (The HF-persisted avg10 predictions are the
    GCV-degenerate lambda-collapse cells at n~=H — unusable here — so we refit
    on the val-selected surface the task pins for new fits.)
    """
    vbar = v.mean(1)  # (n, H) 10-mean target
    vty, ymu = _vty_ymu(fact, vbar[tr])
    best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
    for lam in LAMBDAS:
        pv = _apply(fact, float(lam), vty, ymu, kev_val)
        vr2 = _pooled_r2(pv, vbar[val])
        if vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    p = _apply(fact, best_lam, vty, ymu, kev_te)  # (n_te, H) test predictions
    vte = v[te]  # (n_te, 10, H)
    n_te = p.shape[0]
    single = vte.reshape(n_te * N_ROLLOUTS, -1)
    grand = single.mean(0)
    pred_tiled = np.repeat(p, N_ROLLOUTS, axis=0)
    return {
        "n_test": int(n_te),
        "val_lambda": best_lam,
        "avg10map_r2_vs_10mean": _pooled_r2(p, vbar[te]),
        "avg10map_r2_vs_singledraw": _pooled_r2(pred_tiled, single, mean=grand),
    }


# ── Deliverable A: deviation predictability ────────────────────────────────────


def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFC val-selected split (test/val/train), seed 42 — matches
    val_lambda_robustness in issue1073_fits.py."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    te = np.sort(perm[:N_TEST])
    val = np.sort(perm[N_TEST : N_TEST + N_VAL])
    tr = np.sort(perm[N_TEST + N_VAL :])
    return tr, val, te


def _ridge_val_shared(fact, kev_val, kev_te, Ytr, Yval, Yte) -> dict:
    """Val-selected-lambda ridge off a SHARED train factorization (one eigh).

    fact / kev_val / kev_te are prebuilt from X[tr] / X[val] / X[te]; only the
    target changes here, so all λ + real/null share the single eigh.
    """
    if Ytr.ndim == 1:
        Ytr, Yval, Yte = Ytr[:, None], Yval[:, None], Yte[:, None]
    vty, ymu = _vty_ymu(fact, Ytr)
    best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
    for lam in LAMBDAS:
        pv = _apply(fact, float(lam), vty, ymu, kev_val)
        vr2 = _pooled_r2(pv, Yval)
        if vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    pt = _apply(fact, best_lam, vty, ymu, kev_te)
    return {"r2_test": _pooled_r2(pt, Yte), "val_lambda": best_lam, "r2_val": float(best_vr2)}


def _mlp_test_r2(X: np.ndarray, Y: np.ndarray, tr, te, pca_k: int) -> dict:
    """FFC batched-MLP recipe (width 512, lr 1e-3, wd 1e-4, 300 epochs, patience 20).

    Internal 10% val split of the train rows for early stopping (batched_mlp_fit)."""
    if Y.ndim == 1:
        Y = Y[:, None]
    grp = MLPFitGroup(
        key=("mlp",), X=X[tr].astype(np.float32), Y=Y[tr].astype(np.float32), pca_k=pca_k
    )
    res = batched_mlp_fit(
        [grp],
        hidden=MLP_WIDTH,
        lr=MLP_LR,
        wd=MLP_WD,
        max_epochs=MLP_MAX_EPOCHS,
        patience=MLP_PATIENCE,
        seed=SPLIT_SEED,
    )[("mlp",)]
    pt = res.predict(X[te])
    return {
        "r2_test": _pooled_r2(pt, Y[te]),
        "epochs_ran": int(res.epochs_ran),
        "width": MLP_WIDTH,
        "lr": MLP_LR,
    }


def deviation_predictability(cx, v, disp, fact, kev_val, kev_te, tr, val, te) -> dict:
    """Test 3: can c_x predict draw-specific deviation structure?

    (a) SIGNED deviation d_i0 = draw0 - LOO-9-mean. Ridge is analytically W=0
        (within-context deviations are zero-sum, c_x is constant within context,
        so E[d|c_x]=0) -> R2 ~= 0 by identity; MLP bounded near 0 by the same
        zero conditional mean. Reported with a shuffled-pairing null.
    (b) DISPERSION (per-context rollout dispersion) — a genuine context-level
        target with NON-zero conditional mean; the non-vacuous test of whether
        the sampling VARIABILITY is context-predictable. MLP>ridge>0 => nonlinear.

    fact/kev_val/kev_te/tr/val/te are prebuilt once per layer (one shared eigh).
    The null permutes the TARGET rows (X[tr] and its factorization are reused).
    """
    n = v.shape[0]
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(n)  # target-shuffle null

    s = v.sum(1)
    d0 = v[:, 0, :] - (s - v[:, 0, :]) / (N_ROLLOUTS - 1)  # (n, H) signed deviation, draw 0

    def _ridge(Y):
        return _ridge_val_shared(fact, kev_val, kev_te, Y[tr], Y[val], Y[te])

    signed = {
        "note": (
            "linear ridge is analytically W=0 (within-context deviations "
            "zero-sum, c_x constant within context => E[d|c_x]=0); R2~=0 is "
            "an identity, not evidence about noise independence"
        ),
        "ridge": _ridge(d0),
        "ridge_shuffle_null": _ridge(d0[perm]),
        "mlp": _mlp_test_r2(cx, d0, tr, te, pca_k=MLP_PCA_K),
        "mlp_shuffle_null": _mlp_test_r2(cx, d0[perm], tr, te, pca_k=MLP_PCA_K),
    }
    dispersion = {
        "target": "per-context rollout dispersion (mean pairwise cosine distance)",
        "ridge": _ridge(disp),
        "ridge_shuffle_null": _ridge(disp[perm]),
        "mlp": _mlp_test_r2(cx, disp, tr, te, pca_k=1),
        "mlp_shuffle_null": _mlp_test_r2(cx, disp[perm], tr, te, pca_k=1),
    }
    return {"signed_deviation": signed, "dispersion": dispersion}


# ── Deliverable B: covariates + tail characterization ──────────────────────────

_CJK = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯Ѐ-ӿ"
    r"؀-ۿऀ-ॿ฀-๿]"
)
_CODE = re.compile(
    r"```|\bdef \b|\bimport \b|\bfunction\b|#include|</?[a-zA-Z]+>|"
    r"\bSELECT\b.*\bFROM\b|\bpublic\s+(class|static)\b|=>|\{\s*\n|;\s*\n\s*\}"
)
_MATH = re.compile(r"[0-9].*[+\-*/=^].*[0-9]|\\frac|\\sum|\\int|\bsolve\b|\bequation\b|integral")


def _structural_flags(prompt: str) -> dict:
    """Structural regex over a prompt — emits booleans only, never the text."""
    p = prompt or ""
    alpha = [c for c in p if c.isalpha()]
    non_latin = sum(1 for c in alpha if ord(c) > 0x24F)
    frac_non_latin = non_latin / max(len(alpha), 1)
    has_cjk = bool(_CJK.search(p))
    return {
        "non_english": bool(has_cjk or frac_non_latin > 0.15),
        "code_block": bool(_CODE.search(p)),
        "math": bool(_MATH.search(p)),
        "prompt_chars": len(p),
    }


def _safe_label(flags: dict, greedy_tok: int) -> str:
    """One-word STRUCTURAL topic label (regex-derived, never a human read of text)."""
    if flags["code_block"]:
        return "code"
    if flags["non_english"]:
        return "non-latin"
    if flags["math"]:
        return "math"
    if greedy_tok <= 5 or flags["prompt_chars"] <= 15:
        return "trivial"
    return "other"


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def characterize_tail(
    delta_by_layer: dict,
    headline_layer: int,
    cx_disp_by_layer: dict,
    spans: dict,
    dup_ids: np.ndarray,
    flags_list: list[dict],
    keep: np.ndarray,
) -> dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    delta = delta_by_layer[headline_layer]
    n = delta.size
    greedy_tok = spans["greedy_span"].astype(float)
    stoch_tok = spans["stoch_span"].astype(float)
    mean_stoch = stoch_tok.mean(1)
    std_stoch_len = stoch_tok.std(1)
    disp = cx_disp_by_layer[headline_layer]
    trunc_greedy = (spans["greedy_span"] >= GEN_MAX_TOKENS).astype(float)
    trunc_stoch_any = (spans["stoch_span"] >= GEN_MAX_TOKENS).any(1).astype(float)
    dup_size = np.array([int((dup_ids == d).sum()) for d in dup_ids], dtype=float)
    non_english = np.array([f["non_english"] for f in flags_list], dtype=float)
    code_block = np.array([f["code_block"] for f in flags_list], dtype=float)

    covariates = {
        "greedy_n_tokens": greedy_tok,
        "mean_stoch_n_tokens": mean_stoch,
        "greedy_minus_mean_stoch": greedy_tok - mean_stoch,
        "trunc_greedy": trunc_greedy,
        "trunc_stoch_any": trunc_stoch_any,
        "rollout_dispersion": disp,
        "std_rollout_len": std_stoch_len,
        "prompt_n_tokens": spans["prompt_len"].astype(float),
        "dup_cluster_size": dup_size,
        "exact_dup_member": (dup_size > 1).astype(float),
        "non_english": non_english,
        "code_block": code_block,
    }

    # (1) Spearman(Delta, covariate), BH-corrected
    names, rhos, ps = [], [], []
    for name, x in covariates.items():
        if float(np.std(x)) == 0.0:
            names.append(name)
            rhos.append(0.0)
            ps.append(1.0)
            continue
        rho, p = spearmanr(delta, x)
        names.append(name)
        rhos.append(float(rho))
        ps.append(float(p))
    bh = _bh(np.array(ps))
    spearman = {
        names[i]: {"rho": rhos[i], "p": ps[i], "p_bh": float(bh[i])} for i in range(len(names))
    }

    # (2) logistic tail membership on standardized covariates
    Xc = np.column_stack([covariates[k] for k in covariates])
    Xz = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12)
    logistic = {}
    for tag, y in (
        ("tail_delta_lt_0", (delta < 0).astype(int)),
        ("severe_delta_lt_-0.02", (delta < -0.02).astype(int)),
    ):
        if int(y.sum()) < 10 or int(y.sum()) > n - 10:
            logistic[tag] = {
                "n_positive": int(y.sum()),
                "auc": None,
                "note": "degenerate class balance",
            }
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xz, y)
        auc = float(roc_auc_score(y, clf.predict_proba(Xz)[:, 1]))
        logistic[tag] = {
            "n_positive": int(y.sum()),
            "rate": float(y.mean()),
            "auc_in_sample": auc,
            "std_coefficients": {
                k: float(c) for k, c in zip(covariates, clf.coef_[0], strict=True)
            },
        }

    # (3) tail rate by rollout-dispersion quintiles
    q = np.quantile(disp, [0.2, 0.4, 0.6, 0.8])
    binid = np.digitize(disp, q)
    tail_by_disp = []
    for b in range(5):
        mask = binid == b
        tail_by_disp.append(
            {
                "quintile": b,
                "n": int(mask.sum()),
                "disp_range": [float(disp[mask].min()), float(disp[mask].max())]
                if mask.any()
                else None,
                "tail_rate_delta_lt_0": float((delta[mask] < 0).mean()) if mask.any() else None,
                "severe_rate_delta_lt_-0.02": float((delta[mask] < -0.02).mean())
                if mask.any()
                else None,
                "mean_delta": float(delta[mask].mean()) if mask.any() else None,
            }
        )

    # (4) 10 most severe-tail contexts (ci indices) + structural safe labels
    order = np.argsort(delta)  # most negative first
    severe = []
    for k in order[:10]:
        f = flags_list[k]
        severe.append(
            {
                "ci": int(keep[k]),
                "delta": float(delta[k]),
                "greedy_n_tokens": int(greedy_tok[k]),
                "mean_stoch_n_tokens": float(mean_stoch[k]),
                "rollout_dispersion": float(disp[k]),
                "trunc_greedy": bool(trunc_greedy[k]),
                "dup_cluster_size": int(dup_size[k]),
                "safe_label": _safe_label(f, int(greedy_tok[k])),
            }
        )

    return {
        "headline_layer": headline_layer,
        "n_contexts": int(n),
        "delta_median_by_layer": {
            int(li): float(np.median(dv)) for li, dv in delta_by_layer.items()
        },
        "tail_rate_by_layer": {
            int(li): {
                "frac_delta_lt_0": float((dv < 0).mean()),
                "frac_delta_lt_-0.02": float((dv < -0.02).mean()),
            }
            for li, dv in delta_by_layer.items()
        },
        "truncation_rate_check": {
            "greedy_span_ge_cap": float(trunc_greedy.mean()),
            "stoch_any_span_ge_cap": float(trunc_stoch_any.mean()),
            "note": "cross-check vs decode_descriptives greedy 0.0632 / stoch 0.05916",
        },
        "spearman_delta_vs_covariate_bh": spearman,
        "logistic_tail_membership": logistic,
        "tail_rate_by_dispersion_quintile": tail_by_disp,
        "severe_tail_contexts": severe,
    }


# ── main ────────────────────────────────────────────────────────────────────────


def run(layers: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    keep = load_keep()
    spans = load_span_lens(keep)
    bundle = load_bundle_cx(keep)
    cx_last = bundle["cx_last"]
    delta_by_layer = load_delta()
    achieved = load_achieved()
    dup_ids = I.duplicate_cluster_ids(bundle["prompts_kept"])
    flags_list = [_structural_flags(p) for p in bundle["prompts_kept"]]
    logger.info(
        "[setup] keep=%d dup_clusters=%d done in %.1fs",
        keep.size,
        len(set(dup_ids.tolist())),
        time.time() - t0,
    )

    gap = {"readout_layers": layers, "per_layer": {}}
    disp_by_layer = {}
    for li in layers:
        tl = time.time()
        v = stoch_matrix(li, keep)  # (N,10,H) fp64 ~1.4GB, streamed per layer
        cx = cx_layer(cx_last, li, keep)
        disp = rollout_dispersion(v)
        disp_by_layer[li] = disp
        nd = noise_decomposition(v)
        ceil = oracle_ceiling(v)
        vlr = achieved["val_lambda_robustness"].get(f"L{li}", {})
        gcv = achieved["per_input_layer"]["last"][str(li)]
        r2_10mean_gcv = gcv["avg10"]["r2_pooled"]
        r2_single_gcv = gcv["stoch1_old"]["r2_pooled"]
        # Val-selected surface (GCV degenerates at n~=H — lambda collapse; the
        # task pins the val-selected split for new fits and reads).
        r2_10mean_val = vlr.get("avg10", {}).get("r2_test_val_selected", r2_10mean_gcv)
        r2_single_val = vlr.get("stoch1_old", {}).get("r2_test_val_selected", r2_single_gcv)
        exp = expected_singledraw_r2(nd, r2_10mean_val)

        # One shared factorization per layer: avg10 direct read + deviation fits.
        n = v.shape[0]
        tr, val, te = _split_indices(n)
        fact = _factorize(cx[tr], torch.device("cpu"))
        kev_val = _cross_kernel(fact, cx[val])
        kev_te = _cross_kernel(fact, cx[te])
        direct = avg10map_predicts_singledraw(fact, kev_val, kev_te, v, tr, val, te)
        devp = deviation_predictability(cx, v, disp, fact, kev_val, kev_te, tr, val, te)
        gap["per_layer"][f"L{li}"] = {
            "noise_decomposition": nd,
            "oracle_ceiling": ceil,
            "achieved_r2_gcv": {
                a: gcv[a]["r2_pooled"] for a in ("avg10", "greedy", "stoch1_old", "stoch1_new")
            },
            "achieved_r2_val_selected": {
                a: vlr.get(a, {}).get("r2_test_val_selected")
                for a in ("avg10", "greedy", "stoch1_old", "stoch1_new")
            },
            "expected_singledraw_r2_purenoise": exp,
            "avg10map_predicts_singledraw_direct": direct,
            "gap_verdict": _gap_verdict(ceil, r2_single_val, exp, direct),
            "deviation_predictability": devp,
        }
        del v
        logger.info("[layer L%d] done in %.1fs", li, time.time() - tl)

    gap["metadata"] = _repro_meta({"script": "issue1073_gap_tail_analysis", "deliverable": "A"})
    gap["definitions"] = {
        "noise_share": "within-context draw SS / total single-draw SS (both summed over H)",
        "oracle_ceiling": (
            "pooled R2 of the leave-one-out 9-mean predicting a held-out "
            "draw; best any context-conditioned predictor can do on single "
            "draws (denominator = single-draw deviations from grand mean)"
        ),
        "expected_purenoise_algebra": expected_singledraw_r2.__doc__,
        "avg10map_direct": (
            "the avg10-TRAINED held-out ridge map's predictions evaluated "
            "against single draws — assumption-free pure-noise expectation"
        ),
        "signed_deviation_ridge_is_W0": (
            "within-context deviations are zero-sum and c_x is "
            "constant within context, so the ridge solution is "
            "identically 0 and held-out R2~=0 by identity"
        ),
        "fits": f"val-selected lambda on {N_TEST}/{N_VAL}/(rest) seed-{SPLIT_SEED} split; "
        f"MLP width {MLP_WIDTH} lr {MLP_LR} wd {MLP_WD} epochs {MLP_MAX_EPOCHS} "
        f"patience {MLP_PATIENCE} (FFC recipe verbatim)",
    }
    I.write_json_atomic(out_dir / "gap_noise_decomposition.json", gap)
    logger.info("[A] wrote gap_noise_decomposition.json")

    # Deliverable B — headline layer = the largest-median-Delta read-out layer
    headline = max(layers, key=lambda li: float(np.median(delta_by_layer[li])))
    tail = characterize_tail(
        delta_by_layer, headline, disp_by_layer, spans, dup_ids, flags_list, keep
    )
    tail["prompt_list_sha16"] = bundle["prompt_sha16"]
    tail["metadata"] = _repro_meta({"script": "issue1073_gap_tail_analysis", "deliverable": "B"})
    # persist the headline per-context arrays behind the figures (raw alongside processed)
    tail["percontext_headline"] = {
        "layer": headline,
        "delta": delta_by_layer[headline].tolist(),
        "rollout_dispersion": disp_by_layer[headline].tolist(),
    }
    I.write_json_atomic(out_dir / "adequacy_tail_characterization.json", tail)
    logger.info("[B] wrote adequacy_tail_characterization.json (headline L%d)", headline)
    logger.info("ALL DONE in %.1fs", time.time() - t0)


def _gap_verdict(ceil: dict, r2_single_achieved: float, exp: dict, direct: dict) -> str:
    c = ceil["ceiling_r2_pooled"]
    e = exp["r2_singledraw_expected_purenoise"]
    if r2_single_achieved >= c - 0.02:
        return "achieved single-draw R2 at the oracle noise ceiling — gap is sampling noise"
    if abs(r2_single_achieved - e) <= 0.02:
        return (
            "achieved single-draw R2 indistinguishable from the pure-noise expectation "
            "given the variance — gap consistent with sampling noise alone"
        )
    if r2_single_achieved < e - 0.02:
        return (
            "achieved single-draw R2 below the pure-noise expectation — a residual "
            "shortfall beyond sampling noise"
        )
    return "achieved single-draw R2 above the pure-noise expectation"


def make_figures(out_dir: Path, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gap_noise_decomposition.json") as f:
        gap = json.load(f)
    with open(out_dir / "adequacy_tail_characterization.json") as f:
        tail = json.load(f)

    # Figure 1: achieved vs ceiling vs pure-noise-expected single-draw R2 per layer
    layers = [int(k[1:]) for k in gap["per_layer"]]
    ceil = [gap["per_layer"][f"L{li}"]["oracle_ceiling"]["ceiling_r2_pooled"] for li in layers]
    # val-selected surface (GCV degenerates at n~=H on the noise-reduced arms)
    ach = [gap["per_layer"][f"L{li}"]["achieved_r2_val_selected"]["stoch1_old"] for li in layers]
    exp = [
        gap["per_layer"][f"L{li}"]["expected_singledraw_r2_purenoise"][
            "r2_singledraw_expected_purenoise"
        ]
        for li in layers
    ]
    direct = [
        gap["per_layer"][f"L{li}"]["avg10map_predicts_singledraw_direct"].get(
            "avg10map_r2_vs_singledraw"
        )
        for li in layers
    ]
    avg10 = [gap["per_layer"][f"L{li}"]["achieved_r2_val_selected"]["avg10"] for li in layers]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(layers))
    w = 0.18
    ax.bar(x - 2 * w, avg10, w, label="achieved 10-mean target")
    ax.bar(x - w, ach, w, label="achieved single-draw (stoch1)")
    ax.bar(x, exp, w, label="expected single-draw (pure noise)")
    ax.bar(
        x + w,
        [d if d is not None else 0 for d in direct],
        w,
        label="avg10-map -> single draw (direct)",
    )
    ax.bar(x + 2 * w, ceil, w, label="oracle noise ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{li}" for li in layers])
    ax.set_ylabel("held-out pooled $R^2$")
    ax.set_xlabel("read-out layer")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    pp.savefig_paper(fig, "gap_noise_ceiling", dir=fig_dir)
    plt.close(fig)

    # Figure 2: Delta vs rollout dispersion + tail rate by dispersion bins
    ph = tail["percontext_headline"]
    delta = np.array(ph["delta"])
    disp = np.array(ph["rollout_dispersion"])
    bins = tail["tail_rate_by_dispersion_quintile"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.scatter(disp, delta, s=3, alpha=0.25)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xlabel("rollout dispersion (mean pairwise cosine dist)")
    a1.set_ylabel(f"per-context greedy-adequacy $\\Delta$ (L{ph['layer']})")
    q = [b["quintile"] + 1 for b in bins]
    tr = [b["tail_rate_delta_lt_0"] for b in bins]
    sv = [b["severe_rate_delta_lt_-0.02"] for b in bins]
    a2.plot(q, tr, "o-", label="tail rate ($\\Delta<0$)")
    a2.plot(q, sv, "s-", label="severe rate ($\\Delta<-0.02$)")
    a2.set_xlabel("rollout-dispersion quintile")
    a2.set_ylabel("fraction of contexts")
    a2.set_xticks(q)
    a2.legend(fontsize=8)
    fig.tight_layout()
    pp.savefig_paper(fig, "adequacy_tail_covariates", dir=fig_dir)
    plt.close(fig)
    logger.info("wrote figures to %s", fig_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=READOUT_LAYERS)
    ap.add_argument("--pilot", action="store_true", help="single layer (19) for wall-time probe")
    ap.add_argument("--out-dir", type=str, default=str(EVAL_RESULTS_DIR))
    ap.add_argument("--fig-dir", type=str, default=str(PROJECT_ROOT / "figures" / "issue_1073"))
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.figures_only:
        make_figures(out_dir, Path(args.fig_dir))
        return 0
    layers = [19] if args.pilot else list(args.layers)
    run(layers, out_dir)
    if not args.pilot:
        make_figures(out_dir, Path(args.fig_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
