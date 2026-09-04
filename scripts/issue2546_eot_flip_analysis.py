"""Flip-set analysis for the #2546 chain-of-thought retrieval experiment (arm 1, layer 19).

A "flip" is a question whose own answer state is NOT retrieved (top-1) from the
context-state prediction (cell p7_A: last-context-token state -> mean-over-answer-tokens
state) but IS retrieved from the end-of-thought-state prediction (cell p7_D: the
</think>-token state -> the same target). Retrieval follows the production recipe:
whitened cosine (whitening = Cholesky of the train-fold answer-state covariance,
shrinkage 0.1) + CSLS k=10, pool = the held-out fold's true answer states, hit =
nearest pool vector is the question's own answer state.

Inputs (read-only), all under /mnt/eps-data/thomasjiralerspong/cot_necessity:
  allfit/preds/p7_{A,D}__all__a1.npz            out-of-fold ridge predictions (30193 x 3584)
  allfit/preds/hits__p7_{A,D}__all__a1.npz       recorded retrieval hits (ground truth for groups)
  hf/targets/ans_mean__arm1__post__{ds}__l19.npz true answer states
  hf/issue2546_cotmap/eval_results_mirror/out/necessity/pair_necessity_a1.json  questions + labels
  hf/issue2546_cotmap/raw_completions/post_greedy_a1/{ds}.jsonl                 greedy generations

Analyses: group counts (flip / reverse flip / never hit / stable hit) per dataset; Q1 rank
distribution of flips + nearest-wrong-neighbor confusion for context misses (and for
never-hits under the end-of-thought prediction); Q2 per-question features with per-feature
AUROC and multivariate logistic regressions (flip vs never hit among context misses;
context miss vs context hit) with dataset fixed effects and bootstrap CIs; Q3 geometry
(error reduction and margin change by group, near-miss share); Q4 reverse flips; a
stratified qualitative sample (seed 0) coded under a fixed A-E rubric.

Outputs:
  eval_results/issue_2546/allfit/eot_vs_context/flips/flip_analysis.json
  eval_results/issue_2546/allfit/eot_vs_context/flips/flip_sample_coded.json
  figures/issue_2546/eot_flips/*.png

The expensive whitened-CSLS recompute is cached at /tmp/issue2546_eot_flip_cache.npz.
Run: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    .venv/bin/python scripts/issue2546_eot_flip_analysis.py
"""

from dotenv import load_dotenv

load_dotenv()

import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from explore_persona_space.analysis.null_battery import (  # noqa: E402
    PRIMARY_LAMBDA,
    shrunk_cholesky_from_cov,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue2546_eot_vs_context as ref  # noqa: E402  (load_preds/load_hits/load_targets/answer_class)

BASE = Path("/mnt/eps-data/thomasjiralerspong/cot_necessity")
HF = BASE / "hf" / "issue2546_cotmap"
OUT = ROOT / "eval_results" / "issue_2546" / "allfit" / "eot_vs_context" / "flips"
FIG = ROOT / "figures" / "issue_2546" / "eot_flips"
CACHE = Path("/tmp/issue2546_eot_flip_cache.npz")
CORPORA = ref.CORPORA
MCQ = ref.MCQ
K_CSLS = 10


def csls(S, k=K_CSLS):
    """CSLS-adjusted similarity: 2*S - mean top-k per row - mean top-k per column."""
    nq, npool = S.shape
    r_q = np.partition(S, npool - k, axis=1)[:, npool - k :].mean(1)
    r_p = np.partition(S, nq - k, axis=0)[nq - k :, :].mean(0)
    return 2.0 * S - r_q[:, None] - r_p[None, :]


def unit(a):
    """Row-normalize to unit L2 norm."""
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def whiten(x, mu, ell):
    """Whiten rows of x with mean mu and Cholesky factor ell (lower)."""
    return solve_triangular(ell, (x - mu).T, lower=True).T


def retrieval_detail(preds, Y, folds):
    """Per-question retrieval detail under each prediction, per the production recipe.

    Returns, per prediction name: recomputed hit (argmax == self), rank of the own answer
    in the CSLS ordering, retrieval margin (own CSLS score minus best other), and the
    global index of the nearest OTHER pool vector. Vectorized per fold.
    """
    n = len(Y)
    out = {
        name: {
            "hit": np.zeros(n, bool),
            "rank": np.zeros(n, np.int32),
            "margin": np.zeros(n, np.float64),
            "nn_other": np.zeros(n, np.int32),
        }
        for name in preds
    }
    for k in np.unique(folds):
        tr = folds != k
        te = np.where(folds == k)[0]
        mu = Y[tr].mean(0)
        ell = shrunk_cholesky_from_cov(np.cov(Y[tr], rowvar=False), PRIMARY_LAMBDA)
        zP = unit(whiten(Y[te], mu, ell))
        for name, pred in preds.items():
            zq = unit(whiten(pred[te].astype(np.float64), mu, ell))
            Sc = csls(zq @ zP.T)
            d = out[name]
            d["hit"][te] = Sc.argmax(1) == np.arange(len(te))
            own = np.diagonal(Sc).copy()
            np.fill_diagonal(Sc, -np.inf)
            d["rank"][te] = 1 + (Sc > own[:, None]).sum(1)
            best_other = Sc.max(1)
            d["margin"][te] = own - best_other
            d["nn_other"][te] = te[Sc.argmax(1)]
            del Sc
        print(f"fold {k}: pool={len(te)} done", flush=True)
    return out


def sq_err(Y, pred, chunk=4096):
    """Row squared error ||Y - pred||^2 in float64, chunked."""
    n = len(Y)
    e = np.zeros(n, np.float64)
    for i in range(0, n, chunk):
        d = Y[i : i + chunk] - pred[i : i + chunk].astype(np.float64)
        e[i : i + chunk] = (d * d).sum(1)
    return e


def auroc(x, y):
    """Rank-based AUROC of feature x for binary outcome y (True = positive class)."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    r = rankdata(x)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_with_boot(X, y, names, ds, n_boot=500, seed=0):
    """Multivariate logistic regression with dataset fixed effects.

    Features standardized on the analysis rows; dataset dummies (first level in the
    CORPORA order dropped); mild l2 (C=1000) for bootstrap stability. Returns point
    coefficients and bootstrap 95 percent intervals over n_boot resamples of rows.
    """
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    levels = [c for c in CORPORA if (ds == c).any()]
    dummies = np.stack([(ds == c).astype(np.float64) for c in levels[1:]], axis=1)
    full_names = names + [f"ds={c}" for c in levels[1:]]
    Xf = np.concatenate([Xs, dummies], axis=1)
    model = LogisticRegression(penalty="l2", C=1000.0, solver="newton-cholesky", max_iter=1000)
    model.fit(Xf, y)
    coef = model.coef_[0].copy()
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, len(full_names)), np.nan)
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.all() or not yb.any():
            continue
        m = LogisticRegression(penalty="l2", C=1000.0, solver="newton-cholesky", max_iter=1000)
        m.fit(Xf[idx], yb)
        boots[b] = m.coef_[0]
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return {
        "features": full_names,
        "coef": coef.tolist(),
        "ci_lo": lo.tolist(),
        "ci_hi": hi.tolist(),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_boot_used": int(np.isfinite(boots[:, 0]).sum()),
        "note": "standardized features; dataset fixed effects; l2 C=1000 for bootstrap stability",
    }


def confusion(rows, nn, cls, ds):
    """Nearest-wrong-neighbor breakdown: same answer string / same dataset / both / neither."""
    nb = nn[rows]
    sa = cls[nb] == cls[rows]
    sd = ds[nb] == ds[rows]
    n = len(rows)
    return {
        "n": int(n),
        "same_answer_and_dataset": float((sa & sd).mean()),
        "same_answer_only": float((sa & ~sd).mean()),
        "same_dataset_only": float((~sa & sd).mean()),
        "neither": float((~sa & ~sd).mean()),
        "share_same_answer": float(sa.mean()),
        "share_same_dataset": float(sd.mean()),
    }


def trunc(s, n):
    """Truncate to n characters, disclosing truncation inline with [...]."""
    s = s.strip()
    return s if len(s) <= n else s[:n] + "[...]"


# Rubric (fixed BEFORE reading any sampled item):
#   A = neighbor shares the answer string AND the question template or problem family
#   B = neighbor shares the answer string, otherwise unrelated question
#   C = near-duplicate or same problem family, different answer
#   D = same topic, different answer, no shared template
#   E = other
RUBRIC = {
    "A": "neighbor shares the answer string and the question template or problem family",
    "B": "neighbor shares the answer string, otherwise unrelated question",
    "C": "near-duplicate or same problem family, different answer",
    "D": "same topic, different answer, no shared template",
    "E": "other",
}
# row_id -> rubric letter, assigned by the analyst after drawing the deterministic seed-0
# sample (rubric fixed before reading; every sampled item coded).
CODES: dict[str, str] = {
    # flips
    "math:11103": "C", "math:13825": "C", "math:8825": "C", "math:5504": "C",
    "math:7978": "C", "math:14456": "C", "math:3796": "C", "math:3779": "C",
    "math:12402": "C", "math:5748": "C", "math:10705": "C", "math:10413": "C",
    "math:7720": "C", "math:5193": "C",
    "gsm8k_train:5838": "A", "gsm8k_train:1137": "C", "gsm8k_train:5484": "C",
    "contexthub:contexthub_abductive_level3:844": "C",
    "contexthub:contexthub_abductive_level1:323": "A",
    "contexthub:contexthub_abductive_level4:1714": "A",
    "contexthub:contexthub_abductive_level3:702": "C",
    "contexthub:contexthub_abductive_level3:1750": "C",
    "contexthub:contexthub_abductive_level4:645": "C",
    "contexthub:contexthub_abductive_level2:1476": "C",
    "contexthub:contexthub_abductive_level1:77": "A",
    "contexthub:contexthub_deductive_level3:451": "C",
    "contexthub:contexthub_abductive_level2:1560": "C",
    "mmlu:4633": "D", "mmlu:9772": "A", "mmlu:6720": "C", "mmlu:9": "C",
    "arc_challenge:575": "C", "arc_challenge:493": "D", "arc_challenge:822": "D",
    "csqa:429": "D", "csqa:376": "C", "csqa:787": "C",
    "piqa:1762": "C", "piqa:168": "A", "piqa:702": "C",
    # never hit
    "math:2123": "C", "math:6199": "A", "math:12958": "C", "math:7072": "C",
    "math:4332": "C", "mmlu:2270": "C",
    "contexthub:contexthub_deductive_level4:1827": "A",
    "mmlu:6826": "A", "gsm8k_train:5228": "C",
    "contexthub:contexthub_abductive_level4:1332": "A",
    # reverse flips
    "contexthub:contexthub_deductive_level1:267": "C",
    "mmlu:8978": "C", "math:10249": "C", "math:5418": "C",
    "contexthub:contexthub_deductive_level2:988": "A",
    "math:10496": "C", "mmlu:9011": "C", "math:4822": "C",
    "mmlu:6817": "C", "mmlu:3183": "C",
}


def main():
    zA = np.load(BASE / "allfit" / "preds" / "p7_A__all__a1.npz")
    zD = np.load(BASE / "allfit" / "preds" / "p7_D__all__a1.npz")
    ids = [str(r) for r in zA["conv_ids"]]
    assert ids == [str(r) for r in zD["conv_ids"]]
    folds = np.asarray(zA["folds"])
    labels = [str(v) for v in zA["labels"]]
    pA = np.asarray(zA["pred_l19"], np.float32)
    print("pred A loaded", flush=True)
    pD = np.asarray(zD["pred_l19"], np.float32)
    print("pred D loaded", flush=True)
    n = len(ids)
    Y = ref.load_targets(ids)  # float64
    ds = np.array([r.split(":")[0] for r in ids])
    labels = np.array(labels)
    hA_map = ref.load_hits("p7_A")
    hD_map = ref.load_hits("p7_D")
    hitA_rec = np.array([hA_map[r] for r in ids])
    hitD_rec = np.array([hD_map[r] for r in ids])
    print(f"{n} questions loaded", flush=True)

    # ---- retrieval recompute (cached) -----------------------------------------------------------
    if CACHE.exists():
        z = np.load(CACHE)
        det = {
            name: {k: z[f"{name}_{k}"] for k in ("hit", "rank", "margin", "nn_other")}
            for name in ("A", "D")
        }
        errA, errD = z["errA"], z["errD"]
        print("loaded cache", flush=True)
    else:
        det = retrieval_detail({"A": pA, "D": pD}, Y, folds)
        errA, errD = sq_err(Y, pA), sq_err(Y, pD)
        np.savez(
            CACHE,
            errA=errA,
            errD=errD,
            **{f"{nm}_{k}": v for nm, d in det.items() for k, v in d.items()},
        )
    agreeA = float((det["A"]["hit"] == hitA_rec).mean())
    agreeD = float((det["D"]["hit"] == hitD_rec).mean())
    print(f"recompute agreement: A {agreeA:.6f} D {agreeD:.6f}", flush=True)
    exact = agreeA == 1.0 and agreeD == 1.0

    # Groups are defined from the RECORDED hits (ground truth of the parent analysis).
    flip = ~hitA_rec & hitD_rec
    rev = hitA_rec & ~hitD_rec
    never = ~hitA_rec & ~hitD_rec
    stable = hitA_rec & hitD_rec
    missA = ~hitA_rec
    group = np.where(flip, "flip", np.where(rev, "reverse", np.where(never, "never", "stable")))

    # ---- features -------------------------------------------------------------------------------
    nec = json.load(
        open(HF / "eval_results_mirror" / "out" / "necessity" / "pair_necessity_a1.json")
    )
    qtext = nec["question_by_row_id"]
    gen = {}
    for c in CORPORA:
        for line in open(HF / "raw_completions" / f"post_greedy_a{ref.ARM}" / f"{c}.jsonl"):
            rec = json.loads(line)
            gen[rec["row_id"]] = rec
    texts = [gen[r]["text"] for r in ids]
    think_pos = np.array([t.rfind("</think>") for t in texts])
    reason_chars = np.array(
        [p if p >= 0 else len(t) for p, t in zip(think_pos, texts)], np.float64
    )
    ans_texts = [
        t[p + len("</think>") :].strip() if p >= 0 else "" for p, t in zip(think_pos, texts)
    ]
    ans_chars = np.array([len(a) for a in ans_texts], np.float64)
    q_chars = np.array([len(qtext[r]) for r in ids], np.float64)
    prompt_tok = np.array([gen[r]["n_prompt_tokens"] for r in ids], np.float64)
    gen_tok = np.array([gen[r]["n_gen_tokens"] for r in ids], np.float64)
    finish_len = np.array([gen[r]["finish_reason"] == "length" for r in ids], np.float64)
    cls = np.array([ref.answer_class(c, t) for c, t in zip(ds, texts)])
    key = np.array([f"{c}|{a}" for c, a in zip(ds, cls)])
    counts = Counter(key)
    freq = np.array([counts[k] for k in key], np.float64)
    log_freq = np.log(freq)
    is_letter = np.isin(ds, list(MCQ)).astype(np.float64)
    red_abs = errA - errD
    red_rel = red_abs / errA
    mA, mD = det["A"]["margin"], det["D"]["margin"]
    rankA, rankD = det["A"]["rank"], det["D"]["rank"]
    nnA, nnD = det["A"]["nn_other"], det["D"]["nn_other"]

    res = {
        "meta": {
            "n": n,
            "recompute_agreement": {"context": agreeA, "end_of_thought": agreeD, "exact": exact},
            "recipe": "whitened cosine (train-fold answer covariance, shrinkage 0.1) + CSLS k=10, pool = held-out fold's true answer states",
            "groups_from": "recorded hits npz",
            "reasoning_length_unit": "characters before the last </think> (per-question token counts before </think> are not stored)",
        },
        "groups": {
            "counts": {g: int((group == g).sum()) for g in ("flip", "reverse", "never", "stable")},
            "by_dataset": {
                c: {
                    "n": int((ds == c).sum()),
                    **{
                        g: int(((group == g) & (ds == c)).sum())
                        for g in ("flip", "reverse", "never", "stable")
                    },
                    "flip_rate": float((flip & (ds == c)).mean() / max((ds == c).mean(), 1e-12)),
                }
                for c in CORPORA
            },
        },
    }
    for c in CORPORA:
        m = ds == c
        res["groups"]["by_dataset"][c]["flip_rate"] = float(flip[m].mean())
        res["groups"]["by_dataset"][c]["never_rate"] = float(never[m].mean())
        res["groups"]["by_dataset"][c]["reverse_rate"] = float(rev[m].mean())

    # ---- Q1: ranks + confusion ------------------------------------------------------------------
    fr = rankA[flip]
    res["q1"] = {
        "flip_context_rank_bins": {
            "rank_2": int((fr == 2).sum()),
            "rank_3_10": int(((fr >= 3) & (fr <= 10)).sum()),
            "rank_11_100": int(((fr >= 11) & (fr <= 100)).sum()),
            "rank_over_100": int((fr > 100).sum()),
            "median": float(np.median(fr)),
        },
        "context_miss_confusion_contextpred": confusion(np.where(missA)[0], nnA, cls, ds),
        "flip_confusion_contextpred": confusion(np.where(flip)[0], nnA, cls, ds),
        "never_confusion_contextpred": confusion(np.where(never)[0], nnA, cls, ds),
        "never_confusion_eotpred": confusion(np.where(never)[0], nnD, cls, ds),
        "never_eot_rank_median": float(np.median(rankD[never])),
    }

    # ---- Q2: features, AUROC, logistic ----------------------------------------------------------
    feats = {
        "log_answer_class_freq": log_freq,
        "is_letter_answer": is_letter,
        "question_chars": q_chars,
        "n_prompt_tokens": prompt_tok,
        "reasoning_chars": reason_chars,
        "answer_chars": ans_chars,
        "n_gen_tokens": gen_tok,
        "finish_reason_length": finish_len,
        "context_sq_error": errA,
        "eot_sq_error": errD,
        "error_reduction_abs": red_abs,
        "error_reduction_rel": red_rel,
        "context_margin": mA,
        "eot_margin": mD,
    }
    med_feats = [
        "log_answer_class_freq",
        "question_chars",
        "n_prompt_tokens",
        "reasoning_chars",
        "answer_chars",
        "n_gen_tokens",
        "context_sq_error",
        "eot_sq_error",
        "error_reduction_abs",
        "error_reduction_rel",
        "context_margin",
        "eot_margin",
    ]
    res["q2"] = {
        "medians_by_group": {
            g: {
                **{f: float(np.median(feats[f][group == g])) for f in med_feats},
                "answer_class_freq": float(np.median(freq[group == g])),
                "rank_context_median": float(np.median(rankA[group == g])),
                "rank_eot_median": float(np.median(rankD[group == g])),
                "share_letter": float(is_letter[group == g].mean()),
                "share_finish_length": float(finish_len[group == g].mean()),
            }
            for g in ("flip", "reverse", "never", "stable")
        }
    }
    # (i) among context misses: flip vs never (positive = flip)
    mi = np.where(missA)[0]
    y_i = flip[mi]
    res["q2"]["auroc_flip_vs_never"] = {f: auroc(feats[f][mi], y_i) for f in feats}
    # (ii) all questions: context miss vs hit (positive = miss)
    y_ii = missA
    res["q2"]["auroc_miss_vs_hit"] = {f: auroc(feats[f], y_ii) for f in feats}
    logit_feats_i = [
        "log_answer_class_freq",
        "is_letter_answer",
        "n_prompt_tokens",
        "reasoning_chars",
        "answer_chars",
        "finish_reason_length",
        "context_sq_error",
        "error_reduction_rel",
        "context_margin",
    ]
    # log-transform the heavy-tailed positives for the multivariate model
    def tf(name, v):
        if name in ("n_prompt_tokens", "reasoning_chars", "answer_chars"):
            return np.log1p(v)
        if name in ("context_sq_error", "eot_sq_error"):
            return np.log(v)
        return v

    Xi = np.stack([tf(f, feats[f][mi]) for f in logit_feats_i], axis=1)
    res["q2"]["logit_flip_vs_never"] = logit_with_boot(
        Xi, y_i, [f"log_{f}" if f in ("n_prompt_tokens", "reasoning_chars", "answer_chars", "context_sq_error") else f for f in logit_feats_i], ds[mi]
    )
    logit_feats_ii = [f for f in logit_feats_i if f != "context_margin"]
    Xii = np.stack([tf(f, feats[f]) for f in logit_feats_ii], axis=1)
    res["q2"]["logit_miss_vs_hit"] = logit_with_boot(
        Xii, y_ii, [f"log_{f}" if f in ("n_prompt_tokens", "reasoning_chars", "answer_chars", "context_sq_error") else f for f in logit_feats_ii], ds
    )

    # ---- Q3: geometry ---------------------------------------------------------------------------
    dm = mD - mA
    res["q3"] = {
        "by_group": {
            g: {
                "error_reduction_abs_median": float(np.median(red_abs[group == g])),
                "error_reduction_rel_median": float(np.median(red_rel[group == g])),
                "margin_change_median": float(np.median(dm[group == g])),
                "context_margin_median": float(np.median(mA[group == g])),
                "eot_margin_median": float(np.median(mD[group == g])),
            }
            for g in ("flip", "reverse", "never", "stable")
        },
        # near miss = |context margin| within the bottom decile of |context margin| over ALL questions
        "abs_margin_p10_all": float(np.percentile(np.abs(mA), 10)),
        "flip_share_near_miss": float(
            (np.abs(mA[flip]) <= np.percentile(np.abs(mA), 10)).mean()
        ),
        "never_share_near_miss": float(
            (np.abs(mA[never]) <= np.percentile(np.abs(mA), 10)).mean()
        ),
        "note": "near miss defined as |context margin| in the bottom decile of |context margin| across all 30193 questions",
    }

    # ---- Q4: reverse flips ----------------------------------------------------------------------
    res["q4"] = {
        "n": int(rev.sum()),
        "by_dataset": {c: int((rev & (ds == c)).sum()) for c in CORPORA},
        "by_label": {str(v): int((rev & (labels == v)).sum()) for v in np.unique(labels[rev])},
        "medians": res["q2"]["medians_by_group"]["reverse"],
        "confusion_eotpred": confusion(np.where(rev)[0], nnD, cls, ds),
        "eot_rank_median": float(np.median(rankD[rev])),
    }

    # ---- qualitative sample ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    fc = {c: int((flip & (ds == c)).sum()) for c in CORPORA}
    total_f = sum(fc.values())
    # Pin datasets whose proportional share is below 3 at the floor of 3, then split the
    # remaining slots proportionally among the rest (largest-remainder rounding).
    small = [c for c in CORPORA if fc[c] > 0 and fc[c] / total_f * 40 < 3]
    big = [c for c in CORPORA if fc[c] > 0 and c not in small]
    alloc = {c: 3 for c in small}
    rest = 40 - 3 * len(small)
    big_total = sum(fc[c] for c in big)
    shares = {c: fc[c] / big_total * rest for c in big}
    alloc.update({c: int(np.floor(shares[c])) for c in big})
    rema = {c: shares[c] - np.floor(shares[c]) for c in big}
    while sum(alloc.values()) < 40:
        c = max(rema, key=lambda c: rema[c])
        alloc[c] += 1
        rema[c] = -1.0
    sample_idx = []
    for c in CORPORA:
        pool = np.where(flip & (ds == c))[0]
        take = min(alloc.get(c, 0), len(pool))
        sample_idx += list(rng.choice(pool, size=take, replace=False))
    never_idx = list(rng.choice(np.where(never)[0], size=10, replace=False))
    rev_idx = list(rng.choice(np.where(rev)[0], size=10, replace=False))

    def item(i, g):
        i = int(i)
        nb = int(nnA[i])
        return {
            "group": g,
            "row_id": ids[i],
            "dataset": str(ds[i]),
            "label": str(labels[i]),
            "question": trunc(qtext[ids[i]], 300),
            "final_answer": trunc(ans_texts[i], 120),
            "answer_class": str(cls[i]),
            "answer_class_freq": int(freq[i]),
            "rank_context": int(rankA[i]),
            "rank_eot": int(rankD[i]),
            "context_margin": float(mA[i]),
            "nn_row_id": ids[nb],
            "nn_dataset": str(ds[nb]),
            "nn_question": trunc(qtext[ids[nb]], 300),
            "nn_answer_class": str(cls[nb]),
            "nn_same_answer": bool(cls[nb] == cls[i]),
            "nn_same_dataset": bool(ds[nb] == ds[i]),
            "code": CODES.get(ids[i]),
        }

    sample = (
        [item(i, "flip") for i in sample_idx]
        + [item(i, "never") for i in never_idx]
        + [item(i, "reverse") for i in rev_idx]
    )
    code_counts = {
        g: dict(Counter(s["code"] for s in sample if s["group"] == g)) for g in ("flip", "never", "reverse")
    }
    res["sample"] = {
        "allocation_flips": alloc,
        "seed": 0,
        "rubric": RUBRIC,
        "code_counts": code_counts,
        "n_coded": int(sum(1 for s in sample if s["code"])),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with atomic_replace(OUT / "flip_analysis.json") as tmp:
        tmp.write_text(json.dumps(res, indent=1, default=float))
    with atomic_replace(OUT / "flip_sample_coded.json") as tmp:
        tmp.write_text(json.dumps({"rubric": RUBRIC, "items": sample}, indent=1, default=float))

    # ---- figures --------------------------------------------------------------------------------
    b = res["q1"]["flip_context_rank_bins"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = ["2", "3-10", "11-100", ">100"]
    vals = [b["rank_2"], b["rank_3_10"], b["rank_11_100"], b["rank_over_100"]]
    ax.bar(bins, vals, color="#4878d0")
    ax.set_xlabel("own-answer rank under the context prediction")
    ax.set_ylabel("flips (n=%d)" % int(flip.sum()))
    ax.set_title("Context-prediction rank of flipped questions")
    fig.tight_layout()
    fig.savefig(FIG / "flip_context_rank_distribution.png", dpi=150)
    plt.close(fig)

    a = res["q2"]["auroc_flip_vs_never"]
    names = sorted(a, key=lambda f: a[f])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(names, [a[f] for f in names], color="#4878d0")
    ax.axvline(0.5, color="gray", lw=1)
    ax.set_xlabel("AUROC (flip vs never hit, among context misses)")
    ax.set_title("Per-feature AUROC: what makes a context miss rescuable")
    fig.tight_layout()
    fig.savefig(FIG / "auroc_flip_vs_never.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    groups = ("stable", "flip", "never", "reverse")
    data = [np.clip(red_rel[group == g], -1, 1) for g in groups]
    ax.boxplot(data, tick_labels=groups, showfliers=False)
    ax.axhline(0.0, color="gray", lw=1)
    ax.set_ylabel("relative error reduction (clipped to [-1, 1])")
    ax.set_title("Squared-error reduction, context to end of thought")
    fig.tight_layout()
    fig.savefig(FIG / "error_reduction_by_group.png", dpi=150)
    plt.close(fig)

    if any(code_counts.values()):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        cats = ["A", "B", "C", "D", "E"]
        width = 0.25
        x = np.arange(len(cats))
        for j, g in enumerate(("flip", "never", "reverse")):
            ax.bar(x + (j - 1) * width, [code_counts[g].get(c, 0) for c in cats], width, label=g)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_xlabel("rubric category")
        ax.set_ylabel("sampled items")
        ax.legend()
        ax.set_title("Qualitative rubric codes by group")
        fig.tight_layout()
        fig.savefig(FIG / "rubric_code_counts.png", dpi=150)
        plt.close(fig)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
