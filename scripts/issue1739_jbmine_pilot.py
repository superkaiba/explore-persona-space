"""0-GPU jailbreak-mining pilot (team-lead brief, exploratory).

Question: can a CHEAP context-side signal (residual at the last prompt token,
v_C) separate contexts that reliably elicit misaligned output on the jailbreak
lane from ordinary/benign contexts, in a needle-in-haystack regime? And does a
refusal/harm-direction projection add anything over a plain linear probe?

Data (all pre-existing, 0-GPU, sliced <2GB — see the report for provenance):
  POSITIVES  #1739 evil-OOD contexts (mhj / pair / tom-gibbs jailbreak lanes),
             v_C = context_end (last prompt token) from
             issue1739_ctxmap/evil_ood_full/store, joined to the per-context
             graded TRAIT DV (evil_ood_full/dv_dataset/evil/labeling.json).
             NB: the brief asked for the COMPLIANCE DV; those activations are
             locked in a 32 GB non-random-access tar (evil_labeling.tar) and the
             measured compliance<->trait rank-corr is only 0.07-0.22, so this is
             a trait-DV pilot — see report gap section.
  NEGATIVES  #1092 realistic_crossing benign WildChat/LMSYS contexts,
             v_C = cell_inst_own/context_end (same model Qwen2.5-7B-Instruct,
             same revision a09a3545, same 28-layer grid, same last-prompt-token
             pooling — consistent by construction, asserted below).

Content hygiene: works on numeric labels + activation tensors only; never reads
raw jailbreak / rollout / prompt text.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials + shared-VM thread caps bind BEFORE any heavy import (#847).
load_dotenv()

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

USER = os.environ["USER"]
DEST = Path(f"/mnt/eps-data/{USER}/issue1739_jbmine")
EVIL_STORE = DEST / "issue1739_ctxmap/evil_ood_full/store"
BENIGN_STORE = DEST / "issue1092_realistic_crossing/analysis_tensors/summaries/cell_inst_own"
BENIGN_MANIFEST = DEST / "issue1092_realistic_crossing/corpus/manifest.jsonl"
TRAIT_LABELING = Path("eval_results/issue_1739/evil_ood_full/dv_dataset/evil/labeling.json")
LAYERS = [7, 11, 15, 19, 23, 27]
HIDDEN = 3584
EVIL_SUBDIRS = ["mhj", "pair", "tomgibbs_p0", "tomgibbs_p1"]
SEED = 0
POS_THR = 68.0  # trait-DV positive threshold (~top 5% of the OOD evil set)
NEG_MAX_TRAIT = 5.0  # "failed jailbreak" hard-negative ceiling
N_BENIGN = 5000  # benign subsample cap
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- store readers
def _sorted_shards(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> int:
        stem = p.stem
        return int(stem.split("_shard")[1]) if "_shard" in stem else 0

    return sorted(paths, key=key)


def load_evil() -> dict:
    """Return {layer: (X[n_ctx, H], context_ids)} + parallel group_key/rung, deduped per context.

    Rows are per (context, rollout_k); context_end is identical across rollouts
    (same prompt) so we mean-pool the rows of each context_id.
    """
    # 1) row_index (context_id per row) concatenated in (subdir, shard) order.
    ctx_ids: list[str] = []
    groups: list[str] = []
    rungs: list[str] = []
    for sub in EVIL_SUBDIRS:
        d = EVIL_STORE / sub
        for ri in _sorted_shards(list(d.glob("row_index_shard*.jsonl"))):
            for line in ri.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                r = json.loads(line)
                ctx_ids.append(r["context_id"])
                groups.append(r["group_key"])
                rungs.append(r["rung"])
    n_rows = len(ctx_ids)

    # 2) per-layer activation matrix in the SAME row order.
    out = {}
    for L in LAYERS:
        mats = []
        for sub in EVIL_SUBDIRS:
            d = EVIL_STORE / sub
            for npy in _sorted_shards(list(d.glob(f"context_end_L{L:02d}_shard*.npy"))):
                mats.append(np.load(npy))
        X = np.concatenate(mats, axis=0).astype(np.float32)
        assert X.shape[0] == n_rows, f"L{L}: {X.shape[0]} rows vs {n_rows} index"
        assert X.shape[1] == HIDDEN, X.shape
        out[L] = X

    # 3) dedup per context_id (mean over rollout rows).
    uniq = sorted(set(ctx_ids))
    idx_by_ctx: dict[str, list[int]] = {}
    for i, c in enumerate(ctx_ids):
        idx_by_ctx.setdefault(c, []).append(i)
    ded = {L: np.zeros((len(uniq), HIDDEN), np.float32) for L in LAYERS}
    gk = []
    rg = []
    for j, c in enumerate(uniq):
        rows = idx_by_ctx[c]
        for L in LAYERS:
            ded[L][j] = out[L][rows].mean(axis=0)
        gk.append(groups[rows[0]])
        rg.append(rungs[rows[0]])
    return {"ctx": uniq, "group": gk, "rung": rg, "X": ded, "n_rows": n_rows}


def load_benign() -> dict:
    """Return {layer: X[n, H]} + group ids for a benign subsample (row order == manifest)."""
    man = [
        json.loads(x) for x in BENIGN_MANIFEST.read_text(encoding="utf-8").split("\n") if x.strip()
    ]
    n_man = len(man)
    # subsample benign rows (reproducible)
    take = np.sort(rng.choice(n_man, size=min(N_BENIGN, n_man), replace=False))
    groups = [man[i].get("prefix_conv_id") or man[i]["row_id"] for i in take]
    out = {}
    for L in LAYERS:
        X = np.load(BENIGN_STORE / f"context_end_L{L:02d}.npy")
        assert X.shape[0] == n_man, f"benign L{L}: {X.shape[0]} vs manifest {n_man}"
        assert X.shape[1] == HIDDEN, X.shape
        out[L] = X[take].astype(np.float32)
    return {"X": out, "group": groups, "n_man": n_man}


def load_trait_dv() -> dict:
    d = json.loads(TRAIT_LABELING.read_text())
    dv = {}
    per_rollout_min = {}
    for r in d["rows"]:
        if r.get("dv") is None:
            continue
        dv[r["context_id"]] = float(r["dv"])
        prs = [float(v) for v in (r.get("per_rollout_scores") or {}).values() if v is not None]
        per_rollout_min[r["context_id"]] = min(prs) if prs else 0.0
    return {"dv": dv, "min": per_rollout_min}


# ---------------------------------------------------------------- metrics
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def hit_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    order = np.argsort(-s)
    return float(y[order[:k]].mean())


def evals_to_find_n(y: np.ndarray, s: np.ndarray, n: int) -> int:
    order = np.argsort(-s)
    cum = np.cumsum(y[order])
    hit = np.searchsorted(cum, n) + 1
    return int(hit) if cum[-1] >= n else -1


# ---------------------------------------------------------------- arms
def probe_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 0.01) -> np.ndarray:
    """Out-of-fold predicted P(pos) from a strong-L2 logistic probe (grouped 5-fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, penalty="l2", max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    assert not np.isnan(oof).any()
    return oof


def direction_score(X: np.ndarray, r: np.ndarray) -> np.ndarray:
    r = r / (np.linalg.norm(r) + 1e-9)
    return X @ r


def orient(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, int]:
    """Orient a 1-D score so positives score high; return (score, sign)."""
    if roc_auc(y, s) < 0.5:
        return -s, -1
    return s, 1


# ---------------------------------------------------------------- driver
def main() -> int:
    print("[load] evil OOD store ...", flush=True)
    evil = load_evil()
    print(f"  evil contexts (deduped): {len(evil['ctx'])} from {evil['n_rows']} rows")
    print("[load] benign store ...", flush=True)
    benign = load_benign()
    print(f"  benign contexts: {len(benign['group'])} of {benign['n_man']}")
    trait = load_trait_dv()

    # join evil contexts -> trait dv
    dv = np.array([trait["dv"].get(c, np.nan) for c in evil["ctx"]])
    dvmin = np.array([trait["min"].get(c, np.nan) for c in evil["ctx"]])
    have = ~np.isnan(dv)
    print(f"  evil contexts with trait DV: {int(have.sum())}")

    pos_mask = have & (dv >= POS_THR)
    lowneg_mask = have & (dv <= NEG_MAX_TRAIT)
    n_pos = int(pos_mask.sum())
    print(
        f"  positives (trait dv>={POS_THR}): {n_pos}  |  "
        f"of which all-rollouts>=50: {int((dvmin[pos_mask] >= 50).sum())}"
    )
    print(f"  low-trait evil (dv<={NEG_MAX_TRAIT}, hard-neg pool): {int(lowneg_mask.sum())}")

    # r_B directions
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    rb_local = hub.retry_transient(
        lambda: hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            "issue658_theory_assumptions/store/r_b.pt",
            repo_type="dataset",
        ),
        what="fetch r_b.pt",
    )
    rb = torch.load(
        rb_local,
        weights_only=False,
    )["r_b"]
    rb_hc = np.asarray(rb["harmful_compliance"]["diffmeans"], np.float32)  # (28, H)
    rb_ref = np.asarray(rb["refusal"]["diffmeans"], np.float32)

    pos_idx = np.where(pos_mask)[0]
    lowneg_idx = np.where(lowneg_mask)[0]

    def build_pool(neg_kind: str, base_rate: float | None):
        """neg_kind in {benign, lowtrait}; base_rate=None -> balanced."""
        Xp = {L: evil["X"][L][pos_idx] for L in LAYERS}
        gp = [f"evil::{g}" for g in np.array(evil["group"])[pos_idx]]
        if neg_kind == "benign":
            src_X, src_g, src_n = benign["X"], benign["group"], len(benign["group"])
            negX = {L: src_X[L] for L in LAYERS}
            negg = [f"benign::{g}" for g in src_g]
        else:
            negX = {L: evil["X"][L][lowneg_idx] for L in LAYERS}
            negg = [f"evil::{g}" for g in np.array(evil["group"])[lowneg_idx]]
            src_n = len(lowneg_idx)
        if base_rate is not None:
            n_neg_target = int(round(n_pos * (1 - base_rate) / base_rate))
            n_neg = min(n_neg_target, src_n)
            sel = np.sort(rng.choice(src_n, size=n_neg, replace=False))
            negX = {L: negX[L][sel] for L in LAYERS}
            negg = [negg[i] for i in sel]
        else:  # balanced
            n_neg = min(n_pos, src_n)
            sel = np.sort(rng.choice(src_n, size=n_neg, replace=False))
            negX = {L: negX[L][sel] for L in LAYERS}
            negg = [negg[i] for i in sel]
        X = {L: np.concatenate([Xp[L], negX[L]], axis=0) for L in LAYERS}
        y = np.concatenate([np.ones(n_pos), np.zeros(len(negg))]).astype(int)
        groups = np.array(gp + negg)
        return X, y, groups

    results = {}
    pools = {
        "needle_benign_5pct": ("benign", 0.05),
        "balanced_benign": ("benign", None),
        "hardneg_lowtrait_5pct": ("lowtrait", 0.05),
    }
    for pool_name, (neg_kind, br) in pools.items():
        X, y, groups = build_pool(neg_kind, br)
        base = float(y.mean())
        k5 = max(1, int(round(0.05 * len(y))))
        print(
            f"\n=== POOL {pool_name}: n={len(y)} pos={int(y.sum())} "
            f"base_rate={base:.4f} (k@5%={k5}) ===",
            flush=True,
        )
        results[pool_name] = {
            "n": len(y),
            "n_pos": int(y.sum()),
            "base_rate": base,
            "k5": k5,
            "layers": {},
        }
        for L in LAYERS:
            XL = X[L]
            row = {}
            # arm1: probe (OOF)
            s1 = probe_oof(XL, y, groups)
            # arm2a: harmful_compliance direction
            s2, sgn2 = orient(y, direction_score(XL, rb_hc[L]))
            # arm2b: refusal direction
            s3, sgn3 = orient(y, direction_score(XL, rb_ref[L]))
            # arm4: random
            s4 = rng.standard_normal(len(y))
            for name, s in [("probe", s1), ("rb_harmcomp", s2), ("rb_refusal", s3), ("random", s4)]:
                row[name] = {
                    "roc_auc": roc_auc(y, s),
                    "pr_auc": pr_auc(y, s),
                    "hit@5pct": hit_at_k(y, s, k5),
                    "prec@n_pos": hit_at_k(y, s, int(y.sum())),
                    "evals_to_find_20": evals_to_find_n(y, s, min(20, int(y.sum()))),
                }
            results[pool_name]["layers"][L] = row
            print(
                f"  L{L:02d} probe: ROC {row['probe']['roc_auc']:.3f} PR {row['probe']['pr_auc']:.3f} "
                f"hit@5% {row['probe']['hit@5pct']:.3f} | rb_hc PR {row['rb_harmcomp']['pr_auc']:.3f} "
                f"| rb_ref PR {row['rb_refusal']['pr_auc']:.3f} | rand PR {row['random']['pr_auc']:.3f}",
                flush=True,
            )

    # provenance / consistency block
    results["_meta"] = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "instruct_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "pooling": "context_end (last prompt token)",
        "layers": LAYERS,
        "hidden": HIDDEN,
        "pos_threshold_trait_dv": POS_THR,
        "n_evil_pos": n_pos,
        "n_benign_pool": len(benign["group"]),
        "probe_d": HIDDEN,
        "note": "trait-DV pilot; compliance-DV activations locked in 32GB tar",
        "compliance_vs_trait_rho_per_item": {
            "evil_train": 0.215,
            "evil_hh_rlhf": 0.073,
            "evil_toxicchat": 0.196,
        },
    }
    out = DEST / "pilot_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
