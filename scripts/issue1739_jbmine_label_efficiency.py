"""Label-efficiency sweep: does routing through the context->answer map buy labels?

Arm 3 of the jailbreak-mining pilot (task #2394). Question: at a fixed budget of
N labelled in-domain contexts, does an answer-space probe applied THROUGH the map
(arm D) beat a plain context probe (arm A)? If the map buys label efficiency,
D_N > A_N at small N even though A wins at full labels.

Arms at each budget N (all scored on the SAME held-out eval set):
  A_N            probe trained on real v_C of the N labelled contexts, applied to
                 real v_C at test.
  D_N_indomain   probe trained on real v_A of the N labelled contexts, applied to
                 M_indomain . v_C at test.
  D_N_merged     same, with M_merged (benign rows + in-domain rows).
Plus full-label references (the whole label pool) and a random floor.

LABEL-COST ASYMMETRY (must be read with the curves): A_N needs N labelled
CONTEXTS. D_N needs the same N labels PLUS the answer activation v_A of each
labelled context -- i.e. a generation pass per labelled context. So D_N is
strictly more expensive per label than A_N; D must beat A by a margin that pays
for the generations, not merely tie.

Split (reuses the map-arms split so the curves anchor to that table):
  RESERVE   35% of the jailbreak contexts (label-stratified) -- fits M_indomain /
            M_merged from (v_C, v_A) pairs (NO labels used) AND serves as the
            LABEL POOL that the N labelled contexts are drawn from. Fitting a map
            on a context's (v_C, v_A) pair uses no label, so drawing labels from
            the same contexts leaks no label information into the map.
  EVAL      remaining 65%, negatives subsampled to a 5% base rate -- identical
            construction (and seed) to the map-arms eval set.
Per-context groups; EVAL is disjoint from RESERVE, so map-fit / probe-train /
test never share a context.

ESTIMATOR REGIME (disclosed, deliberate): every probe here runs at
n_train << d = 3584 -- that is the POINT of a label-efficiency curve. The probes
are strongly L2-regularised (C=0.01) classifiers, read only as ranking scores
(PR-AUC), never as an unregularised fit quality. The ridge maps are fit on
n=3000 (benign) / ~1377 (in-domain) / ~4377 (merged) rows at d=3584, so the
in-domain map is itself in the n<d regime and its reconstruction R^2 is a
regularisation-limited read, not a ceiling.

Content hygiene: numeric labels + activation tensors only.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
BENIGN = DEST / "issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own"
MANIFEST = DEST / "issue1092_realistic_crossing/corpus/manifest.jsonl"
LAYERS = [7, 11, 15, 19, 23, 27]
SWEEP_LAYERS = [19, 27]  # L19 = map-arms best layer; L27 = deepest, robustness
HIDDEN = 3584
SEED = 0
POS_N = 150
POS_MEAN_MIN = 90.0
NEG_FAIL_MAX = 5.0
BASE_RATE = 0.05
MAP_RESERVE_FRAC = 0.35
N_BENIGN = 6000
N_MAP_FIT_BENIGN = 3000
RIDGE_LAMBDA = 100.0
PROBE_C = 0.01
BUDGETS = [10, 20, 40, 80, 160, 320]
POS_FRAC_IN_DRAW = 0.10  # >= 2 positives; keeps max positives needed (32) under pool
N_DRAWS = 5
PR_TARGET = 0.80

# Draw RNG is separate from the split RNG so the split reproduces the map-arms one.
rng = np.random.default_rng(SEED)


def load_evil():
    """Load row-aligned v_C / v_A activation stacks + their context ids."""
    vc = np.load(DEST / "evil_compliance_ctxend.npz", allow_pickle=True)
    va = np.load(DEST / "evil_answer_t1.npz", allow_pickle=True)
    cvc = [str(c) for c in vc["context_ids"]]
    cva = [str(c) for c in va["context_ids"]]
    assert cvc == cva, "v_C and v_A npz context_ids must be row-aligned"
    assert [int(x) for x in vc["layers"]] == LAYERS
    return cvc, vc["X"].astype(np.float32), va["X"].astype(np.float32)


def load_compliance_dv():
    """Return (mean, min-over-rollouts) compliance DV per context id."""
    d = json.loads((DEST / "compliance_percontext.json").read_text())
    dv, dvmin = {}, {}
    for rung in d.values():
        for c, v in rung.items():
            dv[c] = float(v["mean"])
            dvmin[c] = float(v["min_over_rollouts"])
    return dv, dvmin


def load_benign_map_pairs():
    """Benign (context_end, t1) pairs used to fit M_benign / half of M_merged."""
    man = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").split("\n") if x.strip()]
    take = np.sort(rng.choice(len(man), size=min(N_BENIGN, len(man)), replace=False))[
        :N_MAP_FIT_BENIGN
    ]
    ce = {L: np.load(BENIGN / f"context_end_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    t1 = {L: np.load(BENIGN / f"t1_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    return ce, t1


def fit_ridge_map(Xc, Yt, lam):
    """Centred multi-output ridge context->answer; returns (W, x_mean, y_mean)."""
    xm = Xc.mean(0)
    ym = Yt.mean(0)
    G = (Xc - xm).T @ (Xc - xm) + lam * np.eye(Xc.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, (Xc - xm).T @ (Yt - ym))
    return W.astype(np.float32), xm.astype(np.float32), ym.astype(np.float32)


def apply_map(M, Xc):
    """Apply a fitted ridge map to context activations."""
    W, xm, ym = M
    return (Xc - xm) @ W + ym


def recon_r2(M, Xc, Ya):
    """Held-out reconstruction R^2 of the real answer activation under the map."""
    pred = apply_map(M, Xc)
    ss_res = float(((Ya - pred) ** 2).sum())
    ss_tot = float(((Ya - Ya.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def pr_auc(y, s):
    """Average precision (PR-AUC); chance equals the positive base rate."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def hit_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def fit_apply(Xtr, ytr, Xte):
    """Train an L2-logistic probe on (Xtr, ytr); return P(pos) on Xte.

    Xtr and Xte must share a feature space (both v_C, or both answer-space:
    real v_A at train and M.v_C at test for arm D). The scaler is fit on Xtr.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=PROBE_C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def n_to_reach(budgets, means, target):
    """Smallest N whose PR-AUC curve reaches `target` (linear interp); None if never."""
    for i, (n, m) in enumerate(zip(budgets, means)):
        if m >= target:
            if i == 0:
                return float(n)
            n0, m0 = budgets[i - 1], means[i - 1]
            if m == m0:
                return float(n)
            return float(n0 + (target - m0) * (n - n0) / (m - m0))
    return None


def main() -> int:
    t0 = time.time()
    print("[load] evil v_C + v_A + compliance DV ...", flush=True)
    ectx, eVC, eVA = load_evil()
    dv, dvmin = load_compliance_dv()
    mean = np.array([dv.get(c, np.nan) for c in ectx])
    mmin = np.array([dvmin.get(c, np.nan) for c in ectx])
    have = ~np.isnan(mean)
    pos_elig = np.where(have & (mean >= POS_MEAN_MIN) & (mmin >= POS_MEAN_MIN))[0]
    pos_idx = np.array(sorted(pos_elig, key=lambda i: (mmin[i], mean[i]), reverse=True)[:POS_N])
    neg_idx = np.where(have & (mean <= NEG_FAIL_MAX))[0]

    # Same split construction (and rng call order) as the map-arms script, so the
    # EVAL set here is the one the map-arms table reports on.
    def split_stratified(idx, frac):
        idx = np.array(idx)
        rng.shuffle(idx)
        n_res = int(round(len(idx) * frac))
        return idx[:n_res], idx[n_res:]

    pos_res, pos_eval = split_stratified(pos_idx, MAP_RESERVE_FRAC)
    neg_res, neg_eval = split_stratified(neg_idx, MAP_RESERVE_FRAC)
    map_res = np.concatenate([pos_res, neg_res])
    n_neg_keep = min(len(neg_eval), int(round(len(pos_eval) * (1 - BASE_RATE) / BASE_RATE)))
    neg_eval_sel = np.sort(rng.choice(neg_eval, size=n_neg_keep, replace=False))
    eval_idx = np.concatenate([pos_eval, neg_eval_sel])
    y_eval = np.concatenate([np.ones(len(pos_eval)), np.zeros(len(neg_eval_sel))]).astype(int)
    k5 = max(1, int(round(0.05 * len(y_eval))))
    print(
        f"  LABEL POOL (= map reserve): {len(pos_res)} pos + {len(neg_res)} neg\n"
        f"  EVAL: n={len(y_eval)} pos={int(y_eval.sum())} base={y_eval.mean():.4f} k5={k5}",
        flush=True,
    )
    max_pos_needed = max(max(2, int(round(POS_FRAC_IN_DRAW * n))) for n in BUDGETS)
    assert max_pos_needed <= len(pos_res), (
        f"largest budget needs {max_pos_needed} labelled positives, pool has {len(pos_res)}"
    )

    print("[load] benign map pairs ...", flush=True)
    bce, bt1 = load_benign_map_pairs()

    draw_rng = np.random.default_rng(SEED + 1000)
    results = {
        "eval": {
            "n": len(y_eval),
            "n_pos": int(y_eval.sum()),
            "base_rate": float(y_eval.mean()),
            "k5": k5,
        },
        "label_pool": {"pos": int(len(pos_res)), "neg": int(len(neg_res))},
        "budgets": BUDGETS,
        "n_draws": N_DRAWS,
        "layers": {},
    }

    for L in SWEEP_LAYERS:
        li = LAYERS.index(L)
        vC_ev = eVC[eval_idx, li, :]
        vA_ev = eVA[eval_idx, li, :]
        vC_pool_pos, vC_pool_neg = eVC[pos_res, li, :], eVC[neg_res, li, :]
        vA_pool_pos, vA_pool_neg = eVA[pos_res, li, :], eVA[neg_res, li, :]

        M_ind = fit_ridge_map(eVC[map_res, li, :], eVA[map_res, li, :], RIDGE_LAMBDA)
        M_mrg = fit_ridge_map(
            np.concatenate([bce[L], eVC[map_res, li, :]], axis=0),
            np.concatenate([bt1[L], eVA[map_res, li, :]], axis=0),
            RIDGE_LAMBDA,
        )
        mvc_ind = apply_map(M_ind, vC_ev)
        mvc_mrg = apply_map(M_mrg, vC_ev)
        lay = {
            "map_r2": {
                "indomain": recon_r2(M_ind, vC_ev, vA_ev),
                "merged": recon_r2(M_mrg, vC_ev, vA_ev),
            },
            "curves": {},
        }

        # Full-label references: train on the ENTIRE label pool.
        vC_full = np.concatenate([vC_pool_pos, vC_pool_neg], axis=0)
        vA_full = np.concatenate([vA_pool_pos, vA_pool_neg], axis=0)
        y_full = np.concatenate([np.ones(len(pos_res)), np.zeros(len(neg_res))]).astype(int)
        lay["full_label_ref"] = {
            "n_train": int(len(y_full)),
            "A": pr_auc(y_eval, fit_apply(vC_full, y_full, vC_ev)),
            "D_indomain": pr_auc(y_eval, fit_apply(vA_full, y_full, mvc_ind)),
            "D_merged": pr_auc(y_eval, fit_apply(vA_full, y_full, mvc_mrg)),
            "E_oracle": pr_auc(y_eval, fit_apply(vA_full, y_full, vA_ev)),
        }
        lay["random_floor"] = pr_auc(y_eval, draw_rng.standard_normal(len(y_eval)))
        print(
            f"  L{L:02d} full-label ref: A {lay['full_label_ref']['A']:.3f} "
            f"D_ind {lay['full_label_ref']['D_indomain']:.3f} "
            f"D_mrg {lay['full_label_ref']['D_merged']:.3f} "
            f"E {lay['full_label_ref']['E_oracle']:.3f} "
            f"| R2 ind {lay['map_r2']['indomain']:.3f} mrg {lay['map_r2']['merged']:.3f}",
            flush=True,
        )

        for N in BUDGETS:
            n_pos = max(2, int(round(POS_FRAC_IN_DRAW * N)))
            n_neg = N - n_pos
            per = {"A": [], "D_indomain": [], "D_merged": []}
            per_roc = {"A": [], "D_indomain": [], "D_merged": []}
            per_hit = {"A": [], "D_indomain": [], "D_merged": []}
            for d in range(N_DRAWS):
                ip = draw_rng.choice(len(pos_res), size=n_pos, replace=False)
                inn = draw_rng.choice(len(neg_res), size=n_neg, replace=False)
                Xc = np.concatenate([vC_pool_pos[ip], vC_pool_neg[inn]], axis=0)
                Xa = np.concatenate([vA_pool_pos[ip], vA_pool_neg[inn]], axis=0)
                yy = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
                for name, Xtr, Xte in (
                    ("A", Xc, vC_ev),
                    ("D_indomain", Xa, mvc_ind),
                    ("D_merged", Xa, mvc_mrg),
                ):
                    s = fit_apply(Xtr, yy, Xte)
                    per[name].append(pr_auc(y_eval, s))
                    per_roc[name].append(roc_auc(y_eval, s))
                    per_hit[name].append(hit_at_k(y_eval, s, k5))
            lay["curves"][N] = {
                "n_pos_in_draw": n_pos,
                "n_neg_in_draw": n_neg,
                **{
                    arm: {
                        "pr_auc_mean": float(np.mean(per[arm])),
                        "pr_auc_sd": float(np.std(per[arm], ddof=1)),
                        "pr_auc_draws": [float(x) for x in per[arm]],
                        "roc_auc_mean": float(np.mean(per_roc[arm])),
                        "hit@5pct_mean": float(np.mean(per_hit[arm])),
                    }
                    for arm in per
                },
            }
            print(
                f"    N={N:>3} ({n_pos}p/{n_neg}n)  "
                f"A {np.mean(per['A']):.3f}+-{np.std(per['A'], ddof=1):.3f}  "
                f"D_ind {np.mean(per['D_indomain']):.3f}"
                f"+-{np.std(per['D_indomain'], ddof=1):.3f}  "
                f"D_mrg {np.mean(per['D_merged']):.3f}"
                f"+-{np.std(per['D_merged'], ddof=1):.3f}  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )

        lay["n_to_reach_pr"] = {
            "target": PR_TARGET,
            **{
                arm: n_to_reach(
                    BUDGETS, [lay["curves"][N][arm]["pr_auc_mean"] for N in BUDGETS], PR_TARGET
                )
                for arm in ("A", "D_indomain", "D_merged")
            },
        }
        print(f"  L{L:02d} N-to-reach-{PR_TARGET}: {lay['n_to_reach_pr']}", flush=True)
        results["layers"][L] = lay

    results["_meta"] = {
        "dv": "compliance (StrongREJECT-style, evil aligned dimension)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "v_C": "context_end (last prompt token)",
        "v_A": "t1 (answer-span)",
        "sweep_layers": SWEEP_LAYERS,
        "probe_C": PROBE_C,
        "ridge_lambda": RIDGE_LAMBDA,
        "pos_frac_in_draw": POS_FRAC_IN_DRAW,
        "map_train_rows": {
            "indomain": int(len(map_res)),
            "merged": int(len(bce[SWEEP_LAYERS[0]])) + int(len(map_res)),
        },
        "hidden_dim": HIDDEN,
        "estimator_regime": (
            "every probe is n_train << d=3584 by design (label-efficiency curve); "
            "L2 C=0.01, read as ranking scores only. In-domain/merged ridge maps "
            "are n<d / n>d respectively; their R^2 is regularisation-limited."
        ),
        "label_cost_asymmetry": (
            "A_N needs N labelled contexts; D_N needs the same N labels PLUS a "
            "generation pass per labelled context to obtain v_A."
        ),
        "split": (
            "label pool == map reserve (map fit uses unlabelled (v_C,v_A) pairs); "
            "EVAL disjoint, identical construction+seed to map_arms_results.json"
        ),
    }
    out = DEST / "label_efficiency_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[done] wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
