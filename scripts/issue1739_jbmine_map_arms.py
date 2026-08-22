"""Compliance-DV pilot arms C/D/E under two map regimes (team-lead follow-up).

v_C = context_end activation; v_A = real answer-span (t1) activation;
M.v_C = mapped/predicted answer. Same pool as the compliance rerun: 150
always-comply positives vs failed-compliance same-family hard negatives, 6
layers, grouped-by-context OOF, PR-AUC headline.

Arms:
  A  probe on real v_C                             (recap, recomputed on eval set)
  B  fixed r_B harm-compliance / refusal direction (recap, recomputed on eval set)
  E  probe on real v_A  (ANSWER-SPACE ORACLE; needs generation, not deployable)
  C  probe trained on M.v_C, tested on M.v_C       (reparametrization check vs A)
  D  probe trained on real v_A, applied to M.v_C   (answer-space probe through map)
C and D each run under THREE map regimes:
  M_benign   ridge context_end->t1 on disjoint benign WildChat/LMSYS rows
  M_indomain ridge context_end->t1 on a held-out grouped-disjoint split of the
             jailbreak contexts' own (v_C, v_A) pairs (label-stratified reserve)
  M_merged   ridge on the ROW UNION of the two training sets above
             (n_benign : n_indomain reported in _meta)

Split scheme (mutually disjoint, grouped by context; groups ~1:1 with context):
  MAP reserve  = 35% of jailbreak contexts (label-stratified) -> fit M_indomain
  EVAL set     = remaining 65%; all arms scored here via grouped 5-fold OOF
                 (D: probe trained on v_A of the train fold, applied to M.v_C of
                  the test fold; M fit on the reserve/benign, disjoint from both)
  Negatives subsampled to a 5% base rate on the eval set.
Leakage risk: per-context groups mean M-fit / probe-train / test never share a
context; the only residual is family-level template similarity, mitigated by
positives+negatives sharing the same evil families by construction.

Also reports each map's held-out reconstruction R^2 of v_A on the eval contexts.

Content hygiene: numeric labels + activation tensors only.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DEST = Path(f"/mnt/eps-data/{os.environ['USER']}/issue1739_jbmine")
BENIGN = DEST / "issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own"
MANIFEST = DEST / "issue1092_realistic_crossing/corpus/manifest.jsonl"
LAYERS = [7, 11, 15, 19, 23, 27]
HIDDEN = 3584
SEED = 0
POS_N = 150
POS_MEAN_MIN = 90.0
NEG_FAIL_MAX = 5.0
BASE_RATE = 0.05
MAP_RESERVE_FRAC = 0.35  # jailbreak contexts reserved to fit M_indomain
N_BENIGN = 6000
N_MAP_FIT_BENIGN = 3000
RIDGE_LAMBDA = 100.0
rng = np.random.default_rng(SEED)


def load_evil():
    vc = np.load(DEST / "evil_compliance_ctxend.npz", allow_pickle=True)
    va = np.load(DEST / "evil_answer_t1.npz", allow_pickle=True)
    cvc = [str(c) for c in vc["context_ids"]]
    cva = [str(c) for c in va["context_ids"]]
    assert cvc == cva, "v_C and v_A npz context_ids must be row-aligned"
    assert [int(x) for x in vc["layers"]] == LAYERS
    return cvc, vc["X"].astype(np.float32), va["X"].astype(np.float32)


def load_compliance_dv():
    d = json.loads((DEST / "compliance_percontext.json").read_text())
    dv, dvmin = {}, {}
    for rung in d.values():
        for c, v in rung.items():
            dv[c] = float(v["mean"])
            dvmin[c] = float(v["min_over_rollouts"])
    return dv, dvmin


def load_benign_map_pairs():
    man = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").split("\n") if x.strip()]
    take = np.sort(rng.choice(len(man), size=min(N_BENIGN, len(man)), replace=False))[
        :N_MAP_FIT_BENIGN
    ]
    ce = {L: np.load(BENIGN / f"context_end_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    t1 = {L: np.load(BENIGN / f"t1_L{L:02d}.npy")[take].astype(np.float32) for L in LAYERS}
    return ce, t1


def fit_ridge_map(Xc, Yt, lam):
    xm = Xc.mean(0)
    ym = Yt.mean(0)
    Xc0 = Xc - xm
    Yt0 = Yt - ym
    G = Xc0.T @ Xc0 + lam * np.eye(Xc.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, Xc0.T @ Yt0)
    return W.astype(np.float32), xm.astype(np.float32), ym.astype(np.float32)


def apply_map(M, Xc):
    W, xm, ym = M
    return (Xc - xm) @ W + ym


def recon_r2(M, Xc, Ya):
    pred = apply_map(M, Xc)
    ss_res = float(((Ya - pred) ** 2).sum())
    ss_tot = float(((Ya - Ya.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def pr_auc(y, s):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def hit_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def evals_to_find_n(y, s, nfind):
    order = np.argsort(-s)
    cum = np.cumsum(y[order])
    return int(np.searchsorted(cum, nfind) + 1) if cum[-1] >= nfind else -1


def probe_oof_same(X, y, groups, C=0.01):
    """Grouped-OOF probe: train and test on the SAME feature matrix X (arms A/E/C)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    assert not np.isnan(oof).any()
    return oof


def probe_oof_cross(Xtrain, Xtest, y, groups, C=0.01):
    """Grouped-OOF probe trained on Xtrain[train fold], tested on Xtest[test fold] (arm D).

    Xtrain (real v_A) and Xtest (M.v_C) share the same feature space; the scaler
    is fit on the train-fold v_A and applied to the test-fold M.v_C.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(Xtrain, y, groups):
        sc = StandardScaler().fit(Xtrain[tr])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Xtrain[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xtest[te]))[:, 1]
    assert not np.isnan(oof).any()
    return oof


def orient(y, s):
    return -s if roc_auc(y, s) < 0.5 else s


def metrics(y, s, k5):
    return {
        "roc_auc": roc_auc(y, s),
        "pr_auc": pr_auc(y, s),
        "hit@5pct": hit_at_k(y, s, k5),
        "evals_to_find_20": evals_to_find_n(y, s, min(20, int(y.sum()))),
    }


def main() -> int:
    print("[load] evil v_C + v_A + compliance DV ...", flush=True)
    ectx, eVC, eVA = load_evil()
    dv, dvmin = load_compliance_dv()
    mean = np.array([dv.get(c, np.nan) for c in ectx])
    mmin = np.array([dvmin.get(c, np.nan) for c in ectx])
    have = ~np.isnan(mean)
    pos_elig = np.where(have & (mean >= POS_MEAN_MIN) & (mmin >= POS_MEAN_MIN))[0]
    pos_idx = np.array(sorted(pos_elig, key=lambda i: (mmin[i], mean[i]), reverse=True)[:POS_N])
    neg_idx = np.where(have & (mean <= NEG_FAIL_MAX))[0]
    print(f"  positives(always-comply)={len(pos_idx)}  hard-negs(failed-comp)={len(neg_idx)}")

    # jailbreak context pool = positives + hard-negatives; label-stratified split
    # into MAP reserve (fit M_indomain) and EVAL set (score all arms).
    def split_stratified(idx, frac):
        idx = np.array(idx)
        rng.shuffle(idx)
        n_res = int(round(len(idx) * frac))
        return idx[:n_res], idx[n_res:]

    pos_res, pos_eval = split_stratified(pos_idx, MAP_RESERVE_FRAC)
    neg_res, neg_eval = split_stratified(neg_idx, MAP_RESERVE_FRAC)
    map_res = np.concatenate([pos_res, neg_res])
    print(f"  MAP reserve: {len(pos_res)} pos + {len(neg_res)} neg = {len(map_res)}")

    # eval set: keep all eval positives, subsample eval negatives to BASE_RATE
    n_neg_keep = min(len(neg_eval), int(round(len(pos_eval) * (1 - BASE_RATE) / BASE_RATE)))
    neg_eval_sel = np.sort(rng.choice(neg_eval, size=n_neg_keep, replace=False))
    eval_idx = np.concatenate([pos_eval, neg_eval_sel])
    y = np.concatenate([np.ones(len(pos_eval)), np.zeros(len(neg_eval_sel))]).astype(int)
    groups = np.array([f"ctx::{ectx[i]}" for i in eval_idx])  # per-context groups
    k5 = max(1, int(round(0.05 * len(y))))
    print(f"  EVAL: n={len(y)} pos={int(y.sum())} base={y.mean():.4f} k5={k5}")

    print("[load] benign map pairs + r_B ...", flush=True)
    bce, bt1 = load_benign_map_pairs()
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    rb = torch.load(
        hub.retry_transient(
            lambda: hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                "issue658_theory_assumptions/store/r_b.pt",
                repo_type="dataset",
            ),
            what="fetch r_b.pt",
        ),
        weights_only=False,
    )["r_b"]
    rb_hc = np.asarray(rb["harmful_compliance"]["diffmeans"], np.float32)
    rb_ref = np.asarray(rb["refusal"]["diffmeans"], np.float32)

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    def unit_rows(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    results = {
        "eval": {"n": len(y), "n_pos": int(y.sum()), "base_rate": float(y.mean()), "k5": k5},
        "map_r2": {"benign": {}, "indomain": {}, "merged": {}},
        "layers": {},
    }
    for li, L in enumerate(LAYERS):
        vC = eVC[eval_idx, li, :]  # eval real v_C
        vA = eVA[eval_idx, li, :]  # eval real v_A (oracle target)
        # maps
        M_ben = fit_ridge_map(bce[L], bt1[L], RIDGE_LAMBDA)
        M_ind = fit_ridge_map(eVC[map_res, li, :], eVA[map_res, li, :], RIDGE_LAMBDA)
        # M_merged: row union of the benign and in-domain training sets
        M_mrg = fit_ridge_map(
            np.concatenate([bce[L], eVC[map_res, li, :]], axis=0),
            np.concatenate([bt1[L], eVA[map_res, li, :]], axis=0),
            RIDGE_LAMBDA,
        )
        results["map_r2"]["benign"][L] = recon_r2(M_ben, vC, vA)
        results["map_r2"]["indomain"][L] = recon_r2(M_ind, vC, vA)
        results["map_r2"]["merged"][L] = recon_r2(M_mrg, vC, vA)
        mvc_ben = apply_map(M_ben, vC)
        mvc_ind = apply_map(M_ind, vC)
        mvc_mrg = apply_map(M_mrg, vC)

        row = {}
        row["A_probe_vC"] = metrics(y, probe_oof_same(vC, y, groups), k5)
        row["E_probe_vA_oracle"] = metrics(y, probe_oof_same(vA, y, groups), k5)
        row["C_benign"] = metrics(y, probe_oof_same(mvc_ben, y, groups), k5)
        row["C_indomain"] = metrics(y, probe_oof_same(mvc_ind, y, groups), k5)
        row["C_merged"] = metrics(y, probe_oof_same(mvc_mrg, y, groups), k5)
        row["D_benign"] = metrics(y, probe_oof_cross(vA, mvc_ben, y, groups), k5)
        row["D_indomain"] = metrics(y, probe_oof_cross(vA, mvc_ind, y, groups), k5)
        row["D_merged"] = metrics(y, probe_oof_cross(vA, mvc_mrg, y, groups), k5)
        # arm B = map-then-project: fixed r_B direction applied to the MAPPED answer
        row["B_mapproj_benign"] = metrics(y, orient(y, unit_rows(mvc_ben) @ unit(rb_hc[L])), k5)
        row["B_mapproj_indomain"] = metrics(y, orient(y, unit_rows(mvc_ind) @ unit(rb_hc[L])), k5)
        row["B_mapproj_merged"] = metrics(y, orient(y, unit_rows(mvc_mrg) @ unit(rb_hc[L])), k5)
        # raw fixed direction on v_C (no map) — reference
        row["rawdir_rb_harmcomp"] = metrics(y, orient(y, vC @ unit(rb_hc[L])), k5)
        row["rawdir_rb_refusal"] = metrics(y, orient(y, vC @ unit(rb_ref[L])), k5)
        row["random"] = metrics(y, rng.standard_normal(len(y)), k5)
        results["layers"][L] = row
        print(
            f"  L{L:02d} A {row['A_probe_vC']['pr_auc']:.3f} E {row['E_probe_vA_oracle']['pr_auc']:.3f} "
            f"| C ben/ind/mrg {row['C_benign']['pr_auc']:.3f}/"
            f"{row['C_indomain']['pr_auc']:.3f}/{row['C_merged']['pr_auc']:.3f} "
            f"| D ben/ind/mrg {row['D_benign']['pr_auc']:.3f}/"
            f"{row['D_indomain']['pr_auc']:.3f}/{row['D_merged']['pr_auc']:.3f} "
            f"| R2 ben {results['map_r2']['benign'][L]:.3f} "
            f"ind {results['map_r2']['indomain'][L]:.3f} "
            f"mrg {results['map_r2']['merged'][L]:.3f}",
            flush=True,
        )

    results["_meta"] = {
        "dv": "compliance (StrongREJECT-style, evil aligned dimension)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "v_C": "context_end (last prompt token)",
        "v_A": "t1 (answer-span)",
        "layers": LAYERS,
        "map_reserve_frac": MAP_RESERVE_FRAC,
        "ridge_lambda": RIDGE_LAMBDA,
        "map_reserve_n": {"pos": int(len(pos_res)), "neg": int(len(neg_res))},
        "map_train_rows": {
            "benign": int(len(bce[LAYERS[0]])),
            "indomain": int(len(map_res)),
            "merged": int(len(bce[LAYERS[0]])) + int(len(map_res)),
        },
        "split": "per-context groups; MAP reserve / probe-train / test mutually disjoint",
    }
    out = DEST / "map_arms_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
