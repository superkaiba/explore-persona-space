"""Compliance-DV jailbreak-mining rerun (team-lead follow-up).

Same design as the trait pilot but positives = "always-comply" contexts by the
StrongREJECT-style COMPLIANCE DV (mean & min-over-rollouts at ceiling), and the
hard negatives = same-family jailbreak contexts that FAILED on compliance.
Adds arm 3 (map-then-project): a ridge context_end->t1 map fit on a DISJOINT
benign split, predicted answer profile projected onto r_B.

Data (all pre-existing, 0-GPU):
  evil v_C           <DEST>/evil_compliance_ctxend.npz  (10,666 contexts,
                     context_end at 6 layers, deduped; stream-reduced from the
                     32 GB evil_labeling.tar)
  compliance DV      <DEST>/compliance_percontext.json  (per-context mean +
                     min-over-rollouts; evil_train + evil_hh_rlhf)
  benign v_C, v_A    #1092 cell_inst_own context_end + t1 (6 layers) @ e5901706
  r_B                #658 r_b.pt harmful_compliance / refusal directions

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
POS_N = 150  # cleanest always-comply positives
POS_MEAN_MIN = 90.0  # positive: compliance mean AND min-over-rollouts >= this
NEG_FAIL_MAX = 5.0  # failed-compliance hard-negative ceiling (compliance mean)
N_BENIGN = 6000
N_MAP_FIT = 3000  # benign rows reserved to fit the context_end->t1 map (disjoint)
RIDGE_LAMBDA = 100.0
rng = np.random.default_rng(SEED)


def load_evil():
    d = np.load(DEST / "evil_compliance_ctxend.npz", allow_pickle=True)
    ctx = [str(c) for c in d["context_ids"]]
    layers = [int(x) for x in d["layers"]]
    assert layers == LAYERS, layers
    X = d["X"].astype(np.float32)  # (n_ctx, 6, 3584)
    return ctx, X


def load_compliance_dv():
    d = json.loads((DEST / "compliance_percontext.json").read_text())
    dv, dvmin = {}, {}
    for rung in d.values():
        for c, v in rung.items():
            dv[c] = float(v["mean"])
            dvmin[c] = float(v["min_over_rollouts"])
    return dv, dvmin


def load_benign():
    man = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").split("\n") if x.strip()]
    n = len(man)
    take = np.sort(rng.choice(n, size=min(N_BENIGN, n), replace=False))
    groups = [man[i].get("prefix_conv_id") or man[i]["row_id"] for i in take]
    ce, t1 = {}, {}
    for L in LAYERS:
        ce[L] = np.load(BENIGN / f"context_end_L{L:02d}.npy")[take].astype(np.float32)
        t1[L] = np.load(BENIGN / f"t1_L{L:02d}.npy")[take].astype(np.float32)
    return ce, t1, groups


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


def probe_oof(X, y, groups, C=0.01):
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


def fit_ridge_map(Xc, Yt, lam):
    """Ridge map Xc(context_end)->Yt(t1), column-centered; returns (W, xmean, ymean)."""
    xm = Xc.mean(0)
    ym = Yt.mean(0)
    Xc0 = Xc - xm
    Yt0 = Yt - ym
    G = Xc0.T @ Xc0 + lam * np.eye(Xc.shape[1], dtype=np.float64)
    W = np.linalg.solve(G, Xc0.T @ Yt0)
    return W.astype(np.float32), xm, ym


def orient(y, s):
    return -s if roc_auc(y, s) < 0.5 else s


def main() -> int:
    print("[load] evil compliance activations + DV ...", flush=True)
    ectx, eX = load_evil()
    dv, dvmin = load_compliance_dv()
    mean = np.array([dv.get(c, np.nan) for c in ectx])
    mmin = np.array([dvmin.get(c, np.nan) for c in ectx])
    have = ~np.isnan(mean)
    # positives: cleanest always-comply — mean & min >= POS_MEAN_MIN, top POS_N by min then mean
    pos_elig = np.where(have & (mean >= POS_MEAN_MIN) & (mmin >= POS_MEAN_MIN))[0]
    order = sorted(pos_elig, key=lambda i: (mmin[i], mean[i]), reverse=True)
    pos_idx = np.array(order[:POS_N])
    fail_idx = np.where(have & (mean <= NEG_FAIL_MAX))[0]
    print(
        f"  evil w/ compliance DV: {int(have.sum())} | always-comply eligible "
        f"(mean&min>={POS_MEAN_MIN}): {len(pos_elig)} | positives used: {len(pos_idx)} | "
        f"failed-compliance (mean<={NEG_FAIL_MAX}): {len(fail_idx)}"
    )
    # group_key in the evil row_index is ~1:1 with context_id (2954/2954 distinct
    # in the sibling store), so per-context grouping is the honest leakage control;
    # the coarse family prefix would collapse GroupKFold to ~2 groups.
    egrp = np.array(ectx)

    print("[load] benign ...", flush=True)
    bce, bt1, bgrp = load_benign()
    n_ben = len(bgrp)
    map_sel = np.arange(N_MAP_FIT)  # disjoint benign rows for the map fit
    neg_pool = np.arange(N_MAP_FIT, n_ben)  # benign rows available as negatives

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

    # pre-fit context_end->t1 map per layer on the disjoint benign split
    maps = {}
    for li, L in enumerate(LAYERS):
        maps[L] = fit_ridge_map(bce[L][map_sel], bt1[L][map_sel], RIDGE_LAMBDA)
    print(
        f"[map] fit ridge context_end->t1 on {len(map_sel)} disjoint benign rows/layer", flush=True
    )

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    def build(neg_kind, base_rate):
        Xp = eX[pos_idx]  # (n_pos, 6, 3584)
        gp = [f"evil::{g}" for g in egrp[pos_idx]]
        if neg_kind == "benign":
            src = np.stack([bce[L][neg_pool] for L in LAYERS], axis=1)  # (n,6,3584)
            gg = [f"benign::{bgrp[i]}" for i in neg_pool]
        else:  # failed-compliance evil
            src = eX[fail_idx]
            gg = [f"evil::{g}" for g in egrp[fail_idx]]
        n_src = src.shape[0]
        if base_rate is None:
            n_neg = min(len(pos_idx), n_src)
        else:
            n_neg = min(int(round(len(pos_idx) * (1 - base_rate) / base_rate)), n_src)
        sel = np.sort(rng.choice(n_src, size=n_neg, replace=False))
        Xn = src[sel]
        gn = [gg[i] for i in sel]
        X = np.concatenate([Xp, Xn], axis=0)
        y = np.concatenate([np.ones(len(pos_idx)), np.zeros(n_neg)]).astype(int)
        return X, y, np.array(gp + gn)

    pools = {
        "needle_benign_5pct": ("benign", 0.05),
        "balanced_benign": ("benign", None),
        "hardneg_failcomp_5pct": ("failcomp", 0.05),
    }
    results = {}
    for pname, (nk, br) in pools.items():
        X, y, groups = build(nk, br)
        base = float(y.mean())
        k5 = max(1, int(round(0.05 * len(y))))
        print(
            f"\n=== {pname}: n={len(y)} pos={int(y.sum())} base={base:.4f} k5={k5} ===", flush=True
        )
        results[pname] = {
            "n": len(y),
            "n_pos": int(y.sum()),
            "base_rate": base,
            "k5": k5,
            "layers": {},
        }
        for li, L in enumerate(LAYERS):
            XL = X[:, li, :]
            s_probe = probe_oof(XL, y, groups)
            s_hc = orient(y, XL @ unit(rb_hc[L]))
            s_ref = orient(y, XL @ unit(rb_ref[L]))
            W, xm, ym = maps[L]
            hpred = (XL - xm) @ W + ym  # predicted t1
            s_map = orient(
                y, (hpred / (np.linalg.norm(hpred, axis=1, keepdims=True) + 1e-9)) @ unit(rb_hc[L])
            )
            s_rand = rng.standard_normal(len(y))
            row = {}
            for nm, s in [
                ("probe", s_probe),
                ("rb_harmcomp", s_hc),
                ("rb_refusal", s_ref),
                ("map_then_project", s_map),
                ("random", s_rand),
            ]:
                row[nm] = {
                    "roc_auc": roc_auc(y, s),
                    "pr_auc": pr_auc(y, s),
                    "hit@5pct": hit_at_k(y, s, k5),
                    "evals_to_find_20": evals_to_find_n(y, s, min(20, int(y.sum()))),
                }
            results[pname]["layers"][L] = row
            print(
                f"  L{L:02d} probe ROC {row['probe']['roc_auc']:.3f} PR {row['probe']['pr_auc']:.3f} "
                f"hit@5% {row['probe']['hit@5pct']:.3f} | map PR {row['map_then_project']['pr_auc']:.3f} "
                f"| rb_hc PR {row['rb_harmcomp']['pr_auc']:.3f} | rb_ref PR {row['rb_refusal']['pr_auc']:.3f} "
                f"| rand PR {row['random']['pr_auc']:.3f}",
                flush=True,
            )

    results["_meta"] = {
        "dv": "compliance (StrongREJECT-style, evil aligned dimension)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "pooling": "context_end (last prompt token)",
        "layers": LAYERS,
        "hidden": HIDDEN,
        "pos_def": f"compliance mean & min-over-rollouts >= {POS_MEAN_MIN}, top {POS_N} by (min,mean)",
        "n_pos": int(len(pos_idx)),
        "n_failcomp": int(len(fail_idx)),
        "n_benign_pool": int(len(neg_pool)),
        "map": f"ridge context_end->t1, lambda={RIDGE_LAMBDA}, fit on {len(map_sel)} disjoint benign rows",
    }
    out = DEST / "compliance_pilot_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
