#!/usr/bin/env python
"""issue-1345 `story-context-info-probe` — is the conversation-specific information
ABSENT from the story context representation, or PRESENT but not linearly readable?

Every existing #1345 story-collapse read is MAP-MEDIATED (within-R^2, transfer,
reparameterization, map-kNN). This round runs four un-mediated / estimator-varying
probes on the landed layer-19 stores, row-paired by conversation id:

1. RAW nearest-neighbour retrieval with NO fitted map (cosine + euclidean).
2. Linear alignment story v_C <-> chat v_C (grouped-CV ridge, the parent
   instrument) + a held-out CCA spectrum in a fixed PCA basis.
3. Signal / residual decomposition of story v_C given paired chat v_C, calibrated
   against the same decomposition for the no-template regime.
4. Nonlinear probe: batched multi-head MLP vs ridge on IDENTICAL folds, inputs and
   PCA target basis, with shuffled-pairing nulls.

Subcommands: ``stage`` (download pinned shards, extract layer 19, delete the
shard), ``analyze``, ``figures``. CPU only; zero GPU, zero API.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1345_common as c  # noqa: E402
import issue825_fit_cells as fc  # noqa: E402
from explore_persona_space.analysis import mapping_baselines as mb  # noqa: E402
from explore_persona_space.analysis import vectorized_mlp_skill as vms  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PARENT_REV = "2a3cb30acada04defc84fd04d28a2b54da3104cd"
STORY_REV = "cc3c35fe2cbd820ea8dfb49a70db85f36f5f0097"
LAYER = c.HEADLINE_LAYER  # 19
N_FOLDS = fc.N_FOLDS  # 5
SEED = fc.FIT_SEED  # 0
KS = (1, 5, 10)
CCA_PCA_K = 200
# Measured basis for the MLP sizing: 225 GFLOP/s on this 32-core VM (synthetic
# 40-member / 300-epoch pilot). At d_in=256, p=48, hidden=512 the 82-group battery
# (2 legs x (1 observed + 40 shuffled nulls)) projects ~250 TFLOP ~= 18 min/round.
MLP_INPUT_K = 256
MLP_TARGET_P = 48
N_SHUFFLE_NULLS = 40
N_RIDGE_NULLS = 20
# Scope addendum: the parent grid logspace(-2, 4, 13) has its FLOOR selected in every
# full-basis story-input fold; this grid extends four decades upward to separate a
# selector failure from a genuine dimension problem.
WIDE_LAMBDAS = np.logspace(-2, 8, 21)

STAGE_ROOT = Path(
    os.environ.get(
        "EPM_I1345_PROBE_STAGE_ROOT",
        "/mnt/eps-data/thomasjiralerspong/issue1345_story_info_probe",
    )
)
OUT_DIR = _REPO_ROOT / "eval_results/issue_1345/story_context_info_probe"
FIG_DIR = _REPO_ROOT / "figures/issue_1345/story_context_info_probe"

# Store registry: layer-19 (prefix, context, answer) extraction targets.
STORES = {
    "chat": {
        "prefix": "issue1345_framing/analysis_tensors/turnstore",
        "revision": PARENT_REV,
        "stem": "instruct_chat_s",
        "format_key": c.REGIME_FORMAT["r1"],
        "target_turn": c.TARGET_TURN_INDEX["r1"],
        "label": "chat template (r1)",
    },
    "notemplate": {
        "prefix": "issue1345_framing/analysis_tensors/turnstore",
        "revision": PARENT_REV,
        "stem": "instruct_naturalistic_s",
        "format_key": c.REGIME_FORMAT["r2"],
        "target_turn": c.TARGET_TURN_INDEX["r2"],
        "label": "no template (r2)",
    },
    "story_tf": {
        "prefix": (
            "issue1345_framing/conversation_paired_stories_assistant/analysis_tensors/turnstore"
        ),
        "revision": STORY_REV,
        "stem": "instruct_stories_paired_s",
        "format_key": c.REGIME_FORMAT["r4"],
        "target_turn": c.TARGET_TURN_INDEX["r4"],
        "label": "paired story, teacher-forced answer (r4)",
    },
    "story_op": {
        "prefix": "issue1345_framing/onpolicy_assistant_story/analysis_tensors/turnstore",
        "revision": STORY_REV,
        "stem": "instruct_stories_paired_op_s",
        "format_key": c.REGIME_FORMAT["r4op"],
        "target_turn": c.TARGET_TURN_INDEX["r4op"],
        "label": "paired story, on-policy answer (r4op)",
    },
}
STORY_KEYS = ("story_tf", "story_op")


# ---------------------------------------------------------------------------
# stage: pinned shard -> layer-19 arrays -> delete shard (bounded peak disk)
# ---------------------------------------------------------------------------
def _shard_basenames(spec: dict) -> list[str]:
    """Shard basenames for a store, via the retried server-side scoped listing."""
    from huggingface_hub import HfApi

    paths = hub.list_hf_files_under_path(
        HfApi(),
        REPO,
        spec["prefix"],
        repo_type="dataset",
        revision=spec["revision"],
    )
    names = [
        p.split("/")[-1]
        for p in paths
        if p.endswith(".pt") and p.split("/")[-1].startswith(spec["stem"] + "_shard")
    ]
    assert names, f"no .pt shards for {spec['stem']} at {spec['revision'][:10]}"
    return sorted(names)


def stage_store(key: str, spec: dict, *, force: bool = False, max_shards: int = 0) -> Path:
    out = STAGE_ROOT / f"{key}_L{LAYER}.npz"
    if out.exists() and not force:
        print(f"[stage] {key}: already staged -> {out}", flush=True)
        return out
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    shards = _shard_basenames(spec)
    if max_shards:
        shards = shards[:max_shards]
    acc: dict[str, list[np.ndarray]] = {
        f"{arm}_{part}": [] for arm in ("prefix", "context") for part in ("x", "y", "conv")
    }
    t0 = time.time()
    for i, shard in enumerate(shards):
        work = STAGE_ROOT / "_work" / f"{key}_{i:03d}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for suf in (".json", ".pt"):
            # Canonical retried + atomic staging helper (#1402) — never a bare
            # hf_hub_download in live code.
            hub.stage_hub_file(
                REPO,
                f"{spec['prefix']}/{shard[:-3]}{suf}",
                work / f"{shard[:-3]}{suf}",
                repo_type="dataset",
                revision=spec["revision"],
            )
        bundle = fc._load_bundle_any(
            work, "instruct", spec["format_key"], c.TRACK, wanted_keys=("slots", "profiles", "nll")
        )
        c.assert_pt_bundle(bundle, expect_slots=len(c.ARM_SLOT_INDEX))
        for arm, si in c.ARM_SLOT_INDEX.items():
            xy = fc._cell_xy(bundle, {"slot_index": si, "target_turn_index": spec["target_turn"]})
            acc[f"{arm}_x"].append(np.ascontiguousarray(xy["X"][:, LAYER, :], dtype=np.float32))
            acc[f"{arm}_y"].append(np.ascontiguousarray(xy["Y"][:, LAYER, :], dtype=np.float32))
            acc[f"{arm}_conv"].append(np.asarray([str(v) for v in xy["conv_ids"]]))
        del bundle, xy
        shutil.rmtree(work)
        print(f"[stage] {key}: shard {i + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    payload = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    np.savez(out, **payload)
    n_ctx = payload["context_conv"].shape[0]
    print(f"[stage] {key}: n_context={n_ctx} -> {out} ({time.time() - t0:.0f}s)", flush=True)
    return out


def load_store(key: str) -> dict:
    path = STAGE_ROOT / f"{key}_L{LAYER}.npz"
    if not path.exists():
        raise FileNotFoundError(f"store {key} not staged at {path} — run `stage` first")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# shared fit primitives (parent instrument reused verbatim)
# ---------------------------------------------------------------------------
def _pooled_r2(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray) -> float:
    """Parent convention: pooled 1 - SSE/SST with the HELD-OUT fold mean as SST."""
    ss_res = 0.0
    ss_tot = 0.0
    for k in np.unique(folds):
        te = folds == k
        t = y_true[te].astype(np.float64)
        ss_res += float(((t - y_pred[te].astype(np.float64)) ** 2).sum())
        ss_tot += float(((t - t.mean(0)) ** 2).sum())
    return float(1.0 - ss_res / ss_tot)


def ridge_leg(
    x: np.ndarray,
    y: np.ndarray,
    conv_ids: np.ndarray,
    *,
    null_draws: int = 0,
    lambdas: np.ndarray | None = None,
) -> dict:
    """One grouped-CV ridge leg through the parent's `heldout_r2_sweep` kernel.

    ``lambdas=None`` keeps the parent's own grid (logspace(-2, 4, 13)) byte-for-byte;
    a custom grid is threaded to the observed fit AND every null draw, so lambda
    selection stays symmetric.
    """
    sweep = fc.heldout_r2_sweep(
        x[:, None, :],
        y[:, None, :],
        conv_ids,
        n_folds=N_FOLDS,
        seed=SEED,
        null_draws=null_draws,
        collect_cosines=True,
        collect_lambdas=True,
        frozen_layers=(0,),
        lambdas=lambdas,
    )
    mask = sweep["fitted_mask"]
    preds = sweep["preds_frozen"][0]
    folds = sweep["folds"]
    r2_ib = None
    if x.shape[1] == y.shape[1]:
        ss_res = 0.0
        ss_tot = 0.0
        for k in np.unique(folds):
            te = folds == k
            tr = ~te
            p = mb.identity_bias_predict(x[tr], y[tr], x[te])
            t = y[te].astype(np.float64)
            ss_res += float(((t - p) ** 2).sum())
            ss_tot += float(((t - t.mean(0)) ** 2).sum())
        r2_ib = float(1.0 - ss_res / ss_tot)
    out = {
        "n": int(mask.sum()),
        "n_groups": int(np.unique(conv_ids).size),
        "d_in": int(x.shape[1]),
        "d_out": int(y.shape[1]),
        "r2_heldout": float(sweep["r2_obs"][0]),
        "r2_identity_bias": r2_ib,
        "gcv_lambda_per_fold": [
            None if np.isnan(v) else float(v) for v in np.asarray(sweep["gcv_lambda"])[0]
        ],
        "cosine_mean": float(sweep["cosines"][0][mask].mean()),
        "n_train_per_fold": int(round(mask.sum() * (N_FOLDS - 1) / N_FOLDS)),
        "underdetermined": bool(round(mask.sum() * (N_FOLDS - 1) / N_FOLDS) < x.shape[1]),
    }
    if null_draws:
        nulls = np.asarray(sweep["r2_null"])[:, 0]
        out["r2_null_mean"] = float(np.nanmean(nulls))
        out["r2_null_p95"] = float(np.nanpercentile(nulls, 95))
        out["r2_null_draws"] = int(null_draws)
    for metric in ("cosine", "euclidean"):
        out[f"retrieval_{metric}"] = mb.knn_retrieval(preds[mask], y[mask], ks=KS, metric=metric)
    return out


def ridge_leg_reduced(x: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int) -> dict:
    """Well-posed companion: per-fold train-only PCA of X to k dims, then the same
    ridge+GCV kernel (n_train > k by construction)."""
    ss_res = 0.0
    ss_tot = 0.0
    preds = np.zeros_like(y, dtype=np.float32)
    lams = []
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        mu = x[tr].mean(0)
        _, _, vt = np.linalg.svd(x[tr] - mu, full_matrices=False)
        basis = vt[: min(k, vt.shape[0])]
        xr_tr = (x[tr] - mu) @ basis.T
        xr_te = (x[te] - mu) @ basis.T
        cache = fc._prep_fold(xr_tr, xr_te)
        p, lam = fc._ridge_predict_cached(cache, y[tr], return_lam=True)
        preds[te] = np.asarray(p, dtype=np.float32)
        lams.append(float(lam))
        t = y[te].astype(np.float64)
        ss_res += float(((t - p) ** 2).sum())
        ss_tot += float(((t - t.mean(0)) ** 2).sum())
    out = {
        "k_pca_input": int(min(k, x.shape[1])),
        "r2_heldout": float(1.0 - ss_res / ss_tot),
        "gcv_lambda_per_fold": lams,
        "pca_fit": "per-fold train rows only (no held-out leakage)",
    }
    for metric in ("cosine", "euclidean"):
        out[f"retrieval_{metric}"] = mb.knn_retrieval(preds, y, ks=KS, metric=metric)
    return out


def _participation_ratio(x: np.ndarray) -> float:
    xc = (x - x.mean(0)).astype(np.float64)
    s = np.linalg.svd(xc, compute_uv=False)
    ev = s**2
    return float(ev.sum() ** 2 / (ev**2).sum())


# ---------------------------------------------------------------------------
# analysis 1 — raw retrieval, no fitted map
# ---------------------------------------------------------------------------
def raw_retrieval(query: np.ndarray, target: np.ndarray) -> dict:
    out = {"n": int(query.shape[0])}
    for metric in ("cosine", "euclidean"):
        out[metric] = mb.knn_retrieval(query, target, ks=KS, metric=metric)
    return out


# ---------------------------------------------------------------------------
# analysis 2 — held-out CCA spectrum in a fixed PCA basis
# ---------------------------------------------------------------------------
def _pca_project(a: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = a[rows].mean(0)
    _, _, vt = np.linalg.svd(a[rows] - mu, full_matrices=False)
    basis = vt[: min(k, vt.shape[0])]
    return (a - mu) @ basis.T, basis


def cca_spectrum(
    a: np.ndarray, b: np.ndarray, folds: np.ndarray, *, k: int, top: int, seed: int
) -> dict:
    """Fold-averaged held-out canonical correlations (train-fit CCA, held-out read)
    plus a shuffled-pairing null spectrum at the same k."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(b.shape[0])

    def _one(bmat: np.ndarray) -> np.ndarray:
        held = []
        for f in np.unique(folds):
            te = folds == f
            tr = ~te
            ap, _ = _pca_project(a, tr, k)
            bp, _ = _pca_project(bmat, tr, k)
            qa, ra = np.linalg.qr(ap[tr])
            qb, rb = np.linalg.qr(bp[tr])
            u, _, vt = np.linalg.svd(qa.T @ qb, full_matrices=False)
            wa = np.linalg.solve(ra, u[:, :top])
            wb = np.linalg.solve(rb, vt[:top].T)
            za = ap[te] @ wa
            zb = bp[te] @ wb
            za = za - za.mean(0)
            zb = zb - zb.mean(0)
            num = (za * zb).sum(0)
            den = np.linalg.norm(za, axis=0) * np.linalg.norm(zb, axis=0) + 1e-12
            held.append(num / den)
        return np.mean(np.stack(held), axis=0)

    obs = _one(b)
    null = _one(b[perm])
    return {
        "k_pca_per_side": int(k),
        "top": int(top),
        "heldout_canonical_corr": [float(v) for v in obs],
        "heldout_canonical_corr_shuffled": [float(v) for v in null],
        "mean_top10": float(np.mean(obs[:10])),
        "mean_top10_shuffled": float(np.mean(null[:10])),
        "n_above_shuffled_max": int((obs > np.max(null)).sum()),
    }


# ---------------------------------------------------------------------------
# analysis 4 — MLP vs ridge on identical folds / inputs / target basis
# ---------------------------------------------------------------------------
def mlp_vs_ridge(
    legs: dict[str, tuple[np.ndarray, np.ndarray]], folds: np.ndarray, *, seed: int
) -> dict:
    """One batched MLP call over every leg + its shuffled-pairing nulls, plus the
    matched ridge comparator in the identical reduced input / PCA target basis."""
    rng = np.random.default_rng(seed)
    prepared: dict[str, dict] = {}
    groups: list[vms.MLPGroup] = []
    for name, (x, y) in legs.items():
        mu_x = x.mean(0)
        _, _, vt = np.linalg.svd(x - mu_x, full_matrices=False)
        xr = ((x - mu_x) @ vt[:MLP_INPUT_K].T).astype(np.float32)
        mu_y, comps, _ = vms.robust_pca_basis(y, MLP_TARGET_P)
        yp = ((y - mu_y) @ comps.T).astype(np.float32)
        perms = [rng.permutation(yp.shape[0]) for _ in range(N_SHUFFLE_NULLS)]
        prepared[name] = {"xr": xr, "yp": yp, "perms": perms}
        groups.append(vms.MLPGroup(key=(name, "obs"), X=xr, Y=yp))
        for i, p in enumerate(perms):
            groups.append(vms.MLPGroup(key=(name, f"null{i:02d}"), X=xr, Y=yp[p]))
    t0 = time.time()
    res = vms.fit_batched_loco_mlp_multihead(
        groups, device="cpu", row_groups=folds, standardization="per_fold"
    )
    elapsed = time.time() - t0
    out = {
        "mlp": {
            "hidden": int(vms.MLP_HIDDEN),
            "lr": float(vms.MLP_LR),
            "wd": float(vms.MLP_WD),
            "max_epochs": int(vms.MLP_MAX_EPOCHS),
            "seed": int(vms.DEFAULT_MLP_SEED),
            "n_members": int(res.n_members),
            "fit_seconds": elapsed,
            "d_in": MLP_INPUT_K,
            "pca_target_dim": MLP_TARGET_P,
            "input_basis": "PCA of X on all rows (unsupervised in X; shared by "
            "the MLP, the ridge comparator and every shuffled null)",
        },
        "legs": {},
    }
    for name, pre in prepared.items():
        xr, yp = pre["xr"], pre["yp"]
        mlp_obs = _pooled_r2(yp, res.preds_by_key[(name, "obs")], folds)
        mlp_null = np.array(
            [
                _pooled_r2(yp[p], res.preds_by_key[(name, f"null{i:02d}")], folds)
                for i, p in enumerate(pre["perms"])
            ]
        )
        # One eigh per fold, reused across the observed fit and every null draw
        # (the parent's selection-symmetric caching contract).
        fold_ids = list(np.unique(folds))
        caches = {}
        for f in fold_ids:
            te = folds == f
            caches[f] = (te, fc._prep_fold(xr[~te], xr[te]))
        ridge_preds = np.zeros_like(yp)
        for f in fold_ids:
            te, cache = caches[f]
            ridge_preds[te] = np.asarray(fc._ridge_predict_cached(cache, yp[~te]), dtype=np.float32)
        ridge_null = []
        for p in pre["perms"][:N_RIDGE_NULLS]:
            pr = np.zeros_like(yp)
            for f in fold_ids:
                te, cache = caches[f]
                pr[te] = np.asarray(fc._ridge_predict_cached(cache, yp[p][~te]), dtype=np.float32)
            ridge_null.append(_pooled_r2(yp[p], pr, folds))
        out["legs"][name] = {
            "mlp_r2_heldout": mlp_obs,
            "mlp_r2_null_mean": float(mlp_null.mean()),
            "mlp_r2_null_p95": float(np.percentile(mlp_null, 95)),
            "mlp_r2_null_draws": int(mlp_null.size),
            "ridge_r2_heldout": _pooled_r2(yp, ridge_preds, folds),
            "ridge_r2_null_mean": float(np.mean(ridge_null)),
            "ridge_r2_null_p95": float(np.percentile(ridge_null, 95)),
            "ridge_r2_null_draws": len(ridge_null),
            "mlp_minus_ridge": mlp_obs - _pooled_r2(yp, ridge_preds, folds),
            "retrieval_mlp_cosine": mb.knn_retrieval(
                res.preds_by_key[(name, "obs")], yp, ks=KS, metric="cosine"
            ),
            "retrieval_ridge_cosine": mb.knn_retrieval(ridge_preds, yp, ks=KS, metric="cosine"),
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _align(story: dict, other: dict, arm: str = "context") -> dict:
    """Row-align a story store against another store on shared conversation ids."""
    s_ids = story[f"{arm}_conv"]
    o_ids = other[f"{arm}_conv"]
    shared = np.intersect1d(s_ids, o_ids)
    s_pos = {cid: i for i, cid in enumerate(s_ids)}
    o_pos = {cid: i for i, cid in enumerate(o_ids)}
    si = np.array([s_pos[cid] for cid in shared])
    oi = np.array([o_pos[cid] for cid in shared])
    return {"conv_ids": shared, "story_idx": si, "other_idx": oi}


def analyze(story_keys: tuple[str, ...], *, with_notemplate: bool, skip_mlp: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chat = load_store("chat")
    nt = load_store("notemplate") if with_notemplate else None
    summary: dict = {
        "metadata": c.metadata(SEED, 0, "scripts/issue1345_story_info_probe.py"),
        "layer": LAYER,
        "pins": {"parent_revision": PARENT_REV, "story_revision": STORY_REV},
        "n_chat_rows": int(chat["context_conv"].shape[0]),
        "rounds": {},
    }
    for skey in story_keys:
        story = load_store(skey)
        al = _align(story, chat)
        n = al["conv_ids"].size
        print(f"[analyze] {skey}: {n} shared conversations with chat", flush=True)
        s_vc = story["context_x"][al["story_idx"]]
        s_va = story["context_y"][al["story_idx"]]
        s_vp = story["prefix_x"][al["story_idx"]]
        ch_vc = chat["context_x"][al["other_idx"]]
        ch_va = chat["context_y"][al["other_idx"]]
        ch_vp = chat["prefix_x"][al["other_idx"]]
        conv = al["conv_ids"]
        folds = fc._cv_folds(conv, N_FOLDS, SEED)

        r1 = {
            "n": int(n),
            "note": "no fitted map anywhere — the query vector itself is the prediction",
            "story_vC_to_chat_vC": raw_retrieval(s_vc, ch_vc),
            "chat_vC_to_story_vC": raw_retrieval(ch_vc, s_vc),
            "story_vC_to_story_vA": raw_retrieval(s_vc, s_va),
            "chat_vC_to_story_vA": raw_retrieval(ch_vc, s_va),
            "chat_vC_to_chat_vA": raw_retrieval(ch_vc, ch_va),
            "story_vC_to_chat_vA": raw_retrieval(s_vc, ch_va),
            "story_vPrefix_to_chat_vPrefix": raw_retrieval(s_vp, ch_vp),
        }
        _dump(f"raw_retrieval_{skey}.json", r1)

        n_train = int(round(n * (N_FOLDS - 1) / N_FOLDS))
        k_red = min(1024, n_train // 2)
        r2 = {
            "n": int(n),
            "n_train_per_fold": n_train,
            "d": int(s_vc.shape[1]),
            "regime_note": (
                f"n_train={n_train} < d={s_vc.shape[1]} — the parent's deliberately "
                "under-determined GCV ridge regime; the reduced-basis legs "
                f"(k={k_red}) are the well-posed companions"
            ),
            "story_vC_to_chat_vC": ridge_leg(s_vc, ch_vc, conv, null_draws=N_RIDGE_NULLS),
            "chat_vC_to_story_vC": ridge_leg(ch_vc, s_vc, conv, null_draws=N_RIDGE_NULLS),
            "story_vC_to_chat_vC_reduced": ridge_leg_reduced(s_vc, ch_vc, folds, k_red),
            "chat_vC_to_story_vC_reduced": ridge_leg_reduced(ch_vc, s_vc, folds, k_red),
            "story_vC_to_story_vA": ridge_leg(s_vc, s_va, conv, null_draws=N_RIDGE_NULLS),
            "story_vC_to_story_vA_reduced": ridge_leg_reduced(s_vc, s_va, folds, k_red),
            "chat_vC_to_chat_vA": ridge_leg(ch_vc, ch_va, conv, null_draws=N_RIDGE_NULLS),
            "chat_vC_to_chat_vA_reduced": ridge_leg_reduced(ch_vc, ch_va, folds, k_red),
            "chat_vC_to_story_vA": ridge_leg(ch_vc, s_va, conv, null_draws=N_RIDGE_NULLS),
            "cca_story_vC_chat_vC": cca_spectrum(
                s_vc, ch_vc, folds, k=CCA_PCA_K, top=64, seed=SEED + 7
            ),
        }
        _dump(f"alignment_{skey}.json", r2)

        r3 = {
            "n": int(n),
            "definition": (
                "predictable share = held-out pooled R^2 of the grouped-CV ridge "
                "chat v_C -> story v_C (the residual share is 1 - that)"
            ),
            "story_from_chat_r2": r2["chat_vC_to_story_vC"]["r2_heldout"],
            "story_from_chat_r2_reduced": r2["chat_vC_to_story_vC_reduced"]["r2_heldout"],
            "variance_totals": {
                "story_vC_mean_sq_centered_norm": float(((s_vc - s_vc.mean(0)) ** 2).sum(1).mean()),
                "chat_vC_mean_sq_centered_norm": float(
                    ((ch_vc - ch_vc.mean(0)) ** 2).sum(1).mean()
                ),
                "story_vA_mean_sq_centered_norm": float(((s_va - s_va.mean(0)) ** 2).sum(1).mean()),
            },
            "participation_ratio": {
                "story_vC": _participation_ratio(s_vc),
                "chat_vC": _participation_ratio(ch_vc),
                "story_vA": _participation_ratio(s_va),
                "chat_vA": _participation_ratio(ch_va),
            },
        }
        if nt is not None:
            al_nt = _align(story, nt)
            shared3 = np.intersect1d(al["conv_ids"], al_nt["conv_ids"])
            pos_c = {cid: i for i, cid in enumerate(chat["context_conv"])}
            pos_n = {cid: i for i, cid in enumerate(nt["context_conv"])}
            ic = np.array([pos_c[cid] for cid in shared3])
            inn = np.array([pos_n[cid] for cid in shared3])
            f3 = fc._cv_folds(shared3, N_FOLDS, SEED)
            k3 = min(1024, int(round(shared3.size * (N_FOLDS - 1) / N_FOLDS)) // 2)
            r3["calibration_notemplate"] = {
                "n": int(shared3.size),
                "chat_vC_to_notemplate_vC": ridge_leg(
                    chat["context_x"][ic], nt["context_x"][inn], shared3, null_draws=N_RIDGE_NULLS
                ),
                "chat_vC_to_notemplate_vC_reduced": ridge_leg_reduced(
                    chat["context_x"][ic], nt["context_x"][inn], f3, k3
                ),
                "raw_retrieval_notemplate_vC_to_chat_vC": raw_retrieval(
                    nt["context_x"][inn], chat["context_x"][ic]
                ),
                "participation_ratio_notemplate_vC": _participation_ratio(nt["context_x"][inn]),
            }
        _dump(f"decomposition_{skey}.json", r3)

        r4 = None
        if not skip_mlp:
            r4 = mlp_vs_ridge(
                {"story_vC_to_story_vA": (s_vc, s_va), "story_vC_to_chat_vC": (s_vc, ch_vc)},
                folds,
                seed=SEED + 11,
            )
            r4["n"] = int(n)
            _dump(f"nonlinear_probe_{skey}.json", r4)

        summary["rounds"][skey] = {
            "label": STORES[skey]["label"],
            "n_shared_with_chat": int(n),
            "raw_retrieval_acc1_cosine": {
                leg: r1[leg]["cosine"]["acc_at_k"][1]
                for leg in r1
                if isinstance(r1[leg], dict) and "cosine" in r1[leg]
            },
            "raw_retrieval_chance_at_1": r1["story_vC_to_chat_vC"]["cosine"]["chance_at_k"][1],
            "ridge_r2": {
                leg: r2[leg]["r2_heldout"]
                for leg in r2
                if isinstance(r2[leg], dict) and "r2_heldout" in r2[leg]
            },
            "identity_bias_r2": {
                leg: r2[leg].get("r2_identity_bias")
                for leg in r2
                if isinstance(r2[leg], dict) and "r2_identity_bias" in r2[leg]
            },
            "cca_mean_top10": r2["cca_story_vC_chat_vC"]["mean_top10"],
            "cca_mean_top10_shuffled": r2["cca_story_vC_chat_vC"]["mean_top10_shuffled"],
            "story_from_chat_predictable_share": r3["story_from_chat_r2"],
            "participation_ratio": r3["participation_ratio"],
            "mlp_vs_ridge": (
                None
                if r4 is None
                else {
                    leg: {
                        "mlp": v["mlp_r2_heldout"],
                        "ridge": v["ridge_r2_heldout"],
                        "mlp_null_p95": v["mlp_r2_null_p95"],
                    }
                    for leg, v in r4["legs"].items()
                }
            ),
        }
        if nt is not None:
            summary["rounds"][skey]["calibration_notemplate_r2"] = r3["calibration_notemplate"][
                "chat_vC_to_notemplate_vC"
            ]["r2_heldout"]
    _dump("summary.json", summary)


def wide_lambda_probe(story_keys: tuple[str, ...]) -> None:
    """Separate the GCV-selector failure from the input dimension.

    Every full-basis story-INPUT leg selects the parent grid's FLOOR lambda (0.01) in
    all five folds — near-unregularized interpolation at n_train < d — so the negative
    within-R^2 may be a selector artifact rather than absent signal. This re-runs the
    same full-basis legs on a grid extended four decades upward; nothing else changes.
    """
    chat = load_store("chat")
    payload = {
        "metadata": c.metadata(SEED, 0, "scripts/issue1345_story_info_probe.py"),
        "lambda_grid": [float(v) for v in WIDE_LAMBDAS],
        "parent_lambda_grid": [float(v) for v in fc.LAMBDAS],
        "note": (
            "identical rows, folds, seed and kernel as the alignment legs; the ONLY "
            "change is the lambda grid (extended from 1e4 to 1e8)"
        ),
        "rounds": {},
    }
    for skey in story_keys:
        story = load_store(skey)
        al = _align(story, chat)
        s_vc = story["context_x"][al["story_idx"]]
        s_va = story["context_y"][al["story_idx"]]
        ch_vc = chat["context_x"][al["other_idx"]]
        conv = al["conv_ids"]
        payload["rounds"][skey] = {
            "n": int(conv.size),
            "story_vC_to_story_vA_wide": ridge_leg(
                s_vc, s_va, conv, null_draws=N_RIDGE_NULLS, lambdas=WIDE_LAMBDAS
            ),
            "story_vC_to_chat_vC_wide": ridge_leg(
                s_vc, ch_vc, conv, null_draws=N_RIDGE_NULLS, lambdas=WIDE_LAMBDAS
            ),
        }
        print(f"[widelambda] {skey} done", flush=True)
    _dump("wide_lambda_probe.json", payload)


def _dump(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[write] {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("stage", "analyze", "widelambda", "figures"))
    ap.add_argument("--stores", default="story_tf,story_op,chat,notemplate")
    ap.add_argument("--story-keys", default="story_tf,story_op")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--max-shards", type=int, default=0, help="smoke: stage only the first K shards"
    )
    ap.add_argument("--no-notemplate", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    if args.command == "stage":
        for key in args.stores.split(","):
            stage_store(key, STORES[key], force=args.force, max_shards=args.max_shards)
    elif args.command == "analyze":
        analyze(
            tuple(args.story_keys.split(",")),
            with_notemplate=not args.no_notemplate,
            skip_mlp=args.skip_mlp,
        )
    elif args.command == "widelambda":
        wide_lambda_probe(tuple(args.story_keys.split(",")))
    else:
        import issue1345_story_info_figs as figs

        figs.build(OUT_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
