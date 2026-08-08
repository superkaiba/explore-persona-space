#!/usr/bin/env python
"""Issue #1482: FULL-DICTIONARY predictor battery (128,512 judged / 131,072 R^2).

The full-width twin of `issue1482_predictor_battery.py` (16,384-panel), which
this module imports every shared numeric helper from. Scope per the round's
redirect: full dictionary only — the panel results stay on disk as diagnostics
and appear here only as the panel<->full-width rank-correlation bridge check.

Inputs (all local; no HF pulls, no re-encode):
  R^2, three arms   eval_results/issue_1738/sae_twoway/perfeature/sae_{context,
                    prefix,bare}_r2.npy  (131,072 float32, #1738 MULTI-TURN fits)
  judged labels     /mnt/eps-data/.../issue1773_fulldict/labels_upload/
                    axis_labels.shard*.jsonl (642,410 rows = feat_id x 5 axes)
  consistency +     derived here by one vectorized pass over the 1,920-shard
  activity          #1482 pooled store (sparse ans_frac / ans_idx / set_tag),
                    exactly the committed `issue1482_feature_correlates`
                    recipe, then IDENTITY-GATED on the 16,384 panel against
                    the committed covariates
  footprint moments eval_results/issue_1482/footprint_moments (131,072-wide)
  enc/dec norms     SAE weights (full width by construction)
  projected variance one chunked GEMM against the full decoder matrix

CORPUS CAVEAT carried into every output: the full-width R^2 is the #1738
MULTI-TURN read, while consistency / activity / projected variance come from
the #1482 SINGLE-TURN corpus. The panel<->full-width rank correlation on the
shared features is reported as the bridge check.

Figures (one per predictor, fixed templates):
  per_predictor/cont_<name>.*  hexbin scatter of R^2 (clipped at -1) vs the
                               predictor + a thin decile-median trend line;
                               corner annotation = raw rho and all-others-
                               partial rho
  per_predictor/bin_<axis>_<class>.*  prevalence-vs-R^2-rank profile with a
                               marginal-prevalence reference line and a rug of
                               labeled features; corner annotation = AUROC with
                               its bootstrap CI and stratified permutation p
  summary_auroc_depth_overlay.*  AUROC computed among top-k u bottom-k only,
                               one line per binary label over the log k grid
  summary_forest.*, summary_joint_roc.*  optional extras

Vectorized throughout: the shard scan is per-shard `bincount`; every
permutation null is a GEMM against a fixed rank vector; the joint bootstrap
reuses the panel module's one-GEMM-per-chunk correlation-matrix path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_predictor_battery as PB  # noqa: E402

DICT_SIZE = 131_072
SEED = 1482
N_BOOT = 2000
N_PERM = 2000
DCOR_SUBSAMPLE = 8000

POOLED_STORE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1482_shuffnull/"
    "issue1482_error_analysis/analysis_tensors/sae_pooled"
)
FULLDICT_LABELS = Path("/mnt/eps-data/thomasjiralerspong/issue1773_fulldict/labels_upload")
R2_DIR = "eval_results/issue_1738/sae_twoway/perfeature"
R2_ARMS = {
    "context": "sae_context_r2.npy",
    "prefix": "sae_prefix_r2.npy",
    "bare": "sae_bare_r2.npy",
}
PRIMARY_ARM = "context"

OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

# Continuous predictors available at FULL WIDTH -> (label, log-x for figures).
CONTINUOUS = {
    "consistency": ("within-answer consistency", False),
    "activity": ("activity (firing frequency)", True),
    "proj_var": ("dense variance along decoder dir", True),
    "write_norm": ("write norm (gamma-scaled)", False),
    "footprint_var": ("footprint variance", False),
    "footprint_skew": ("footprint skew", False),
    "footprint_kurt": ("footprint kurtosis", False),
    "enc_norm": ("encoder-vector norm", False),
}
# Panel-only continuous predictors, absent full width (stated in the outputs).
PANEL_ONLY_CONTINUOUS = {
    "side_ratio": "answer-vs-query side ratio — #1773 panel table only, not derived full width",
    "rb_align_max": "rb_align — #1773 panel table only, not derived full width",
}

CATEGORICAL = {
    "content_type": ("syntax", ("topic", "operation", "entity", "task_format", "unresolved"), ()),
    "speaker_property": (
        "none",
        ("language", "register_style", "identity_disposition", "unresolved"),
        ("unclear",),
    ),
    "functional_role": ("input_side", ("output_promoting", "mixed", "unresolved"), ()),
    "interpretable": ("no", ("yes", "unresolved"), ()),
}
RETIRED_AXES = ("functional_role",)
ABSTRACTION_ORD = {"token_surface": 0.0, "lexical_semantic": 1.0, "abstract_contextual": 2.0}

# Binary codings for the label reads / prevalence profiles / AUROC-at-depth.
BINARY_AXES = {
    "interpretable": ("interpretable", lambda s: s == "yes", ("unresolved",)),
    "abstraction_high": ("abstraction", lambda s: s == "abstract_contextual", ("unresolved",)),
    "speaker_language": ("speaker_property", lambda s: s == "language", ("unresolved", "unclear")),
    "speaker_register": (
        "speaker_property",
        lambda s: s == "register_style",
        ("unresolved", "unclear"),
    ),
    "speaker_identity": (
        "speaker_property",
        lambda s: s == "identity_disposition",
        ("unresolved", "unclear"),
    ),
    "content_syntax": ("content_type", lambda s: s == "syntax", ("unresolved",)),
    "content_topic": ("content_type", lambda s: s == "topic", ("unresolved",)),
    "content_operation": ("content_type", lambda s: s == "operation", ("unresolved",)),
    "content_entity": ("content_type", lambda s: s == "entity", ("unresolved",)),
    "content_task_format": ("content_type", lambda s: s == "task_format", ("unresolved",)),
}

K_GRID = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200)
N_DECILES = 10


def _log(msg: str) -> None:
    print(f"[fullwidth] {msg}", flush=True)


# ── full-width covariate derivation ───────────────────────────────────────────


def scan_pooled_store(store: Path, cache: Path) -> dict[str, np.ndarray]:
    """Full-width consistency + activity from the local #1482 pooled store.

    Exactly the committed `issue1482_feature_correlates.phase_scan` arithmetic
    (per-shard `bincount` over fit-tagged rows' sparse `ans_frac`/`ans_idx`),
    but the DICT_SIZE-wide accumulators are kept whole instead of sliced to the
    panel. Cached: the scan is ~10 min over 1,920 shards.
    """
    if cache.exists():
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
        _log(f"scan cache hit: {cache} (n_fit={int(out['n_fit'])})")
        return out

    shards = sorted(store.glob("pooled_*.npz"))
    if len(shards) != 1920:
        raise AssertionError(f"expected 1920 pooled shards, found {len(shards)} in {store}")

    cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    sum_frac = np.zeros(DICT_SIZE, dtype=np.float64)
    n_fit = 0
    t0 = time.time()
    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            fit = np.asarray(z["set_tag"]) == 1
            off = np.asarray(z["idx_off"], dtype=np.int64)
            n_fit += int(fit.sum())
            keep = np.repeat(fit, off)
            ik = np.asarray(z["ans_idx"], dtype=np.int64)[keep]
            fk = np.asarray(z["ans_frac"], dtype=np.float64)[keep]
            cnt += np.bincount(ik, minlength=DICT_SIZE)
            sum_frac += np.bincount(ik, weights=fk, minlength=DICT_SIZE)
        if (i + 1) % 384 == 0:
            _log(f"scan {i + 1}/1920 shards, n_fit={n_fit} ({time.time() - t0:.0f}s)")

    with np.errstate(invalid="ignore", divide="ignore"):
        consistency = np.where(cnt > 0, sum_frac / np.maximum(cnt, 1), np.nan)
    activity = cnt / n_fit
    out = {
        "consistency": consistency,
        "activity": activity,
        "cnt": cnt,
        "n_fit": np.int64(n_fit),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    _log(f"scan done in {time.time() - t0:.0f}s: n_fit={n_fit}, cached -> {cache}")
    return out


def identity_gates(scan: dict[str, np.ndarray]) -> dict[str, float]:
    """Restricted to the 16,384 panel, the recomputed covariates must reproduce
    the committed ones (the round's stated gate before any full-width use)."""
    z = np.load(PROJECT_ROOT / PB.PERFEATURE)
    fid = np.asarray(z["feat_ids"], dtype=int)
    c = np.load(PROJECT_ROOT / PB.CONSISTENCY)
    d_act = float(np.abs(scan["activity"][fid] - np.asarray(z["activity"])).max())
    d_con = float(np.abs(scan["consistency"][fid] - np.asarray(c["consistency"])).max())
    _log(f"identity gates: activity max|delta|={d_act:.3e}, consistency max|delta|={d_con:.3e}")
    for name, delta in (("activity", d_act), ("consistency", d_con)):
        if not (delta < 1e-6):
            raise AssertionError(
                f"full-width {name} does not reproduce the committed panel covariate "
                f"(max|delta|={delta:.3e}); refusing to use the derived array"
            )
    return {"activity_max_abs_delta": d_act, "consistency_max_abs_delta": d_con}


def full_width_projection_variance(w_dec: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """Var over the #1738 holdout of the dense state along every feature's unit
    decoder direction, chunked so the (3584, 131072) product never materializes."""
    import issue1482_residual_svd as RS

    y16, _ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
    out = np.empty(w_dec.shape[1], dtype=np.float64)
    for s in range(0, w_dec.shape[1], chunk):
        u = w_dec[:, s : s + chunk]
        u = u / np.linalg.norm(u, axis=0, keepdims=True)
        out[s : s + chunk] = np.einsum("df,df->f", cov @ u, u)
    return out


def load_fulldict_labels() -> dict[str, np.ndarray]:
    """Modal label per feat_id per axis, as DICT_SIZE-wide object arrays.

    Features with no judged row for an axis read 'unlabeled' (distinct from the
    judge's own 'unresolved'), so the coverage gap stays visible downstream.
    """
    axes = ("abstraction", "speaker_property", "content_type", "functional_role", "interpretable")
    out = {ax: np.full(DICT_SIZE, "unlabeled", dtype=object) for ax in axes}
    shards = sorted(FULLDICT_LABELS.glob("axis_labels.shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no axis_labels shards under {FULLDICT_LABELS}")
    n_rows = 0
    for p in shards:
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ax = r["axis"]
                if ax in out:
                    out[ax][int(r["feat_id"])] = r["label"]
                n_rows += 1
    covered = int((out["interpretable"] != "unlabeled").sum())
    _log(f"labels: {n_rows} rows over {len(shards)} shards; {covered} features judged")
    return out


# ── universe assembly ─────────────────────────────────────────────────────────


def assemble(out_dir: Path, work: Path) -> dict:
    from issue1482_sae import BatchTopKSAE

    t0 = time.time()
    r2_all = {}
    for arm, fname in R2_ARMS.items():
        r2_all[arm] = np.asarray(np.load(PROJECT_ROOT / R2_DIR / fname), dtype=np.float64)
        if r2_all[arm].shape != (DICT_SIZE,):
            raise AssertionError(f"{fname}: expected ({DICT_SIZE},), got {r2_all[arm].shape}")
    _log(
        "R^2 arms loaded: "
        + ", ".join(
            f"{a} median {np.nanmedian(v):.4f} ({int(np.isfinite(v).sum())} finite)"
            for a, v in r2_all.items()
        )
    )

    scan = scan_pooled_store(POOLED_STORE, work / "fullwidth_scan.npz")
    gates = identity_gates(scan)

    fp = np.load(PROJECT_ROOT / PB.FOOTPRINT)
    foot = {
        "write_norm": np.asarray(fp["direct_write_norm"], dtype=np.float64),
        "footprint_var": np.asarray(fp["direct_var"], dtype=np.float64),
        "footprint_skew": np.asarray(fp["direct_skew"], dtype=np.float64),
        "footprint_kurt": np.asarray(fp["direct_kurt"], dtype=np.float64),
    }

    sae = BatchTopKSAE.load(k=PB.SAE_K, layer=PB.SAE_LAYER, device="cpu")
    w_dec = np.asarray(sae.w_dec, dtype=np.float64)
    enc_norm = np.linalg.norm(np.asarray(sae.w_enc, dtype=np.float64), axis=1)
    dec_norm = np.linalg.norm(w_dec, axis=0)

    pv_cache = work / "fullwidth_projvar.npy"
    if pv_cache.exists():
        proj_var = np.load(pv_cache)
    else:
        proj_var = full_width_projection_variance(w_dec)
        np.save(pv_cache, proj_var)
    _log(f"covariates ready ({time.time() - t0:.0f}s); dec_norm std {dec_norm.std():.3e}")

    labels = load_fulldict_labels()

    cov = {
        "consistency": scan["consistency"],
        "activity": scan["activity"],
        "proj_var": proj_var,
        "enc_norm": enc_norm,
        **foot,
    }

    # analysis universe: judged AND finite primary-arm R^2 AND answer-active in
    # at least one fit row (consistency is undefined otherwise)
    judged = labels["interpretable"] != "unlabeled"
    finite = np.isfinite(r2_all[PRIMARY_ARM])
    active = scan["cnt"] > 0
    keep = judged & finite & active
    idx = np.flatnonzero(keep)
    _log(
        f"universe: {len(idx)} of {DICT_SIZE} "
        f"(judged {int(judged.sum())}, finite R^2 {int(finite.sum())}, active {int(active.sum())})"
    )

    return {
        "feat_ids": idx,
        "r2": r2_all[PRIMARY_ARM][idx],
        "r2_arms": {a: v[idx] for a, v in r2_all.items()},
        "cov": {k: v[idx] for k, v in cov.items()},
        "labels": {ax: v[idx] for ax, v in labels.items()},
        "gates": gates,
        "n_fit": int(scan["n_fit"]),
        "dec_norm_std": float(dec_norm.std()),
        "coverage": {
            "dict_size": DICT_SIZE,
            "judged": int(judged.sum()),
            "finite_r2": int(finite.sum()),
            "answer_active": int(active.sum()),
            "universe": int(len(idx)),
        },
    }


def bridge_check(bundle: dict) -> dict:
    """Panel<->full-width rank correlation on the shared features (the redirect's
    single required panel read: the two R^2 arrays are DIFFERENT corpora)."""
    z = np.load(PROJECT_ROOT / PB.PERFEATURE)
    pid = np.asarray(z["feat_ids"], dtype=int)
    pr2 = np.asarray(z["r2"], dtype=np.float64)
    pos = {int(f): i for i, f in enumerate(bundle["feat_ids"])}
    hit = np.array([int(f) in pos for f in pid])
    fw = np.array([bundle["r2"][pos[int(f)]] for f in pid[hit]])
    rho = PB._spearman(pr2[hit], fw)
    _log(f"bridge: panel-vs-fullwidth rho = {rho:+.4f} on {int(hit.sum())} shared features")
    return {
        "n_shared": int(hit.sum()),
        "spearman_panel_vs_fullwidth_r2": rho,
        "note": (
            "panel R^2 is the #1482 SINGLE-TURN read; full-width R^2 is the #1738 "
            "MULTI-TURN read — this is a cross-corpus rank agreement, not a replication"
        ),
    }


# ── design matrix + joint model ───────────────────────────────────────────────


def build_design(bundle: dict) -> tuple[np.ndarray, list[str], dict]:
    cov, labels = bundle["cov"], bundle["labels"]
    n = len(bundle["feat_ids"])
    cols, keys, legend = [], [], {}

    for key, (label, _log_x) in CONTINUOUS.items():
        cols.append(PB._rank(cov[key]))
        keys.append(key)
        legend[key] = {"label": label, "kind": "continuous (rank-transformed)"}

    abst = labels["abstraction"]
    known = np.array([s in ABSTRACTION_ORD for s in abst])
    ordv = np.empty(n)
    ordv[known] = [ABSTRACTION_ORD[s] for s in abst[known]]
    ordv[~known] = np.median(ordv[known])
    cols.append(PB._rank(ordv))
    keys.append("abstraction_ord")
    legend["abstraction_ord"] = {"label": "abstraction (ordinal)", "kind": "ordinal"}
    cols.append((~known).astype(np.float64))
    keys.append("abstraction_unresolved")
    legend["abstraction_unresolved"] = {
        "label": "abstraction: unresolved",
        "kind": "missing indicator",
    }

    for axis, (ref, levels, aliases) in CATEGORICAL.items():
        raw = labels[axis]
        norm = np.array([("unresolved" if s in aliases else s) for s in raw], dtype=object)
        for lvl in levels:
            key = f"{axis}__{lvl}"
            col = (norm == lvl).astype(np.float64)
            if col.sum() == 0:
                continue
            cols.append(col)
            keys.append(key)
            legend[key] = {
                "label": f"{axis.replace('_', ' ')}: {lvl}",
                "kind": "one-hot",
                "reference_level": ref,
                "n_positive": int(col.sum()),
                "retired_axis": axis in RETIRED_AXES,
            }
    return np.column_stack(cols), keys, legend


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation on a (sub)sample — the non-monotone-dependence read.

    O(n^2) in memory, so callers subsample; returns dCor in [0, 1] (0 iff
    independent, unlike Spearman which is 0 for any symmetric non-monotone
    dependence).
    """
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = float((A * B).mean())
    dvx = float((A * A).mean())
    dvy = float((B * B).mean())
    if dvx <= 0 or dvy <= 0:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvx * dvy)))


def _decile_profile(pred: np.ndarray, r2: np.ndarray) -> dict:
    """Median R^2 per predictor decile + the top-vs-bottom-decile gap."""
    edges = np.quantile(pred, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    dec = np.searchsorted(edges, pred, side="right")
    med = [
        float(np.median(r2[dec == d])) if (dec == d).any() else float("nan")
        for d in range(N_DECILES)
    ]
    return {
        "decile_median_r2": med,
        "top_minus_bottom_decile_gap": float(med[-1] - med[0]),
        "decile_n": [int((dec == d).sum()) for d in range(N_DECILES)],
    }


def joint_model(bundle: dict, n_boot: int, rng) -> dict:
    x, keys, legend = build_design(bundle)
    r2 = bundle["r2"]
    y = PB._rank(r2)
    n, p = x.shape
    _log(f"design: {n} features x {p} predictors")

    z = np.column_stack([x, y])
    zc = (z - z.mean(0)) / z.std(0)
    corr = (zc.T @ zc) / n
    raw, partial, r2_joint, delta = PB._corr_reads(corr)
    full_refit = PB._ols_r2(x, y)
    lopo_refit = np.array([full_refit - PB._ols_r2(np.delete(x, j, axis=1), y) for j in range(p)])
    _log(
        f"joint R^2 {r2_joint:.4f}; LOPO closed-form vs refit max|d| "
        f"{np.max(np.abs(lopo_refit - delta)):.2e}"
    )

    t0 = time.time()
    boot = PB._bootstrap_reads(z, n_boot, rng, chunk=200)
    _log(f"bootstrap {n_boot} draws in {time.time() - t0:.0f}s")

    # dCor on a fixed subsample (O(n^2) memory at full width)
    sub = rng.choice(n, size=min(DCOR_SUBSAMPLE, n), replace=False)
    dcor = {}
    for j, key in enumerate(keys):
        dcor[key] = _dcor(x[sub, j], y[sub])

    ci_idx = keys.index("consistency")
    r2_wo = PB._ols_r2(np.delete(x, ci_idx, axis=1), y)
    r2_only = PB._ols_r2(x[:, [ci_idx]], y)

    order = np.argsort(-np.abs(partial))
    preds = []
    for j in order:
        key = keys[j]
        row = {
            "key": key,
            **legend[key],
            "spearman_raw": float(raw[j]),
            "spearman_raw_ci95": PB._ci(boot["raw"][:, j]),
            "dcor_subsample": dcor[key],
            "dcor_exceeds_abs_rho": bool(np.isfinite(dcor[key]) and dcor[key] > abs(raw[j]) + 0.05),
            "partial_all_others": float(partial[j]),
            "partial_all_others_ci95": PB._ci(boot["partial"][:, j]),
            "lopo_delta_r2": float(delta[j]),
            "lopo_delta_r2_ci95": PB._ci(boot["delta"][:, j]),
            "lopo_delta_r2_refit": float(lopo_refit[j]),
        }
        if key in CONTINUOUS:
            row.update(_decile_profile(bundle["cov"][key], r2))
        preds.append(row)

    return {
        "design": {
            "scope": "FULL DICTIONARY",
            "target": (
                f"per-feature held-out R^2, {PRIMARY_ARM} arm, #1738 MULTI-TURN fits "
                "(rank-transformed)"
            ),
            "r2_source": f"{R2_DIR}/{R2_ARMS[PRIMARY_ARM]}",
            "n_features": int(n),
            "n_predictors": int(p),
            "coverage": bundle["coverage"],
            "corpus_caveat": (
                "R^2 is the #1738 MULTI-TURN corpus read; consistency, activity and "
                "projected variance are derived from the #1482 SINGLE-TURN corpus — "
                "every full-width correlate here is cross-corpus"
            ),
            "covariate_set_vs_panel": {
                "present_full_width": sorted(CONTINUOUS),
                "absent_full_width": PANEL_ONLY_CONTINUOUS,
            },
            "n_boot": int(n_boot),
            "dcor_subsample_n": int(len(sub)),
            "dcor_note": (
                "distance correlation on a fixed random subsample (O(n^2) memory at full "
                "width); dCor substantially above |rho| flags non-monotone dependence"
            ),
            "identity_gates": bundle["gates"],
            "n_fit_rows": bundle["n_fit"],
            "seed": SEED,
        },
        "joint_r2": float(r2_joint),
        "joint_r2_ci95": PB._ci(boot["joint"]),
        "joint_r2_refit": full_refit,
        "consistency_subsumption": {
            "joint_r2_with_consistency": float(full_refit),
            "joint_r2_without_consistency": float(r2_wo),
            "delta": float(full_refit - r2_wo),
            "consistency_alone_r2": float(r2_only),
        },
        "predictors": preds,
    }


def joint_roc(bundle: dict, rng, n_boot: int = 500) -> dict:
    """All predictors jointly classifying top-decile-vs-rest R^2 membership.

    Rank-score model: OLS of rank(R^2) on the rank design (the same fit the
    joint model reports), scored as a single ranking variable, then AUROC of
    that score for top-decile membership. Stated rather than logistic so the
    score is exactly the joint model's own linear predictor.
    """
    x, keys, _legend = build_design(bundle)
    r2 = bundle["r2"]
    y = PB._rank(r2)
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    score = design @ beta
    thresh = np.quantile(r2, 0.9)
    pos = r2 >= thresh
    point, ci = auroc_with_boot(score[pos], score[~pos], n_boot, rng)
    _log(f"joint ROC (top-decile vs rest): AUROC {point:.4f} {ci}")
    return {
        "model": "rank-score (OLS of rank(R^2) on the rank design; linear predictor as score)",
        "positive_class": "top-decile R^2",
        "n_positive": int(pos.sum()),
        "n_negative": int((~pos).sum()),
        "auroc": point,
        "auroc_ci95": ci,
        "n_boot": int(n_boot),
        "n_predictors": len(keys),
    }


# ── AUROC + label reads ───────────────────────────────────────────────────────


def auroc_with_boot(
    x_pos: np.ndarray, x_neg: np.ndarray, n_boot: int, rng, chunk: int = 200
) -> tuple[float, list[float]]:
    """AUROC = P(pos > neg) + 0.5 P(tie), plus a group-conditional bootstrap CI.

    Chunked over draws: the (n_pos, n_boot) weighted-CDF gather is the memory
    peak, and n_pos reaches ~10^5 at full width.
    """
    kp, kn = len(x_pos), len(x_neg)
    srt = np.sort(x_neg)
    lo = np.searchsorted(srt, x_pos, side="left")
    hi = np.searchsorted(srt, x_pos, side="right")
    point = float((lo + 0.5 * (hi - lo)).sum() / (kp * kn))

    order_n = np.argsort(x_neg)
    draws = []
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx_n = rng.integers(0, kn, size=(kn, b))
        wn = np.bincount((idx_n + np.arange(b) * kn).ravel(), minlength=kn * b)
        wn = wn.reshape(b, kn).T.astype(np.float64)
        cw = np.vstack([np.zeros((1, b)), np.cumsum(wn[order_n], axis=0)])
        u = cw[lo] + 0.5 * (cw[hi] - cw[lo])
        idx_p = rng.integers(0, kp, size=(kp, b))
        wp = np.bincount((idx_p + np.arange(b) * kp).ravel(), minlength=kp * b)
        wp = wp.reshape(b, kp).T.astype(np.float64)
        draws.append((wp * u).sum(axis=0) / (kp * kn))
        done += b
    return point, PB._ci(np.concatenate(draws))


def _decile_of(v: np.ndarray) -> np.ndarray:
    edges = np.quantile(v, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    return np.searchsorted(edges, v, side="right")


def _perm_within_strata(lab: np.ndarray, strata: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """(n, n_perm) labels permuted independently within each activity decile."""
    P = np.empty((len(lab), n_perm), dtype=np.int8)
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s)
        vals = lab[i]
        for p in range(n_perm):
            P[i, p] = vals[rng.permutation(len(i))]
    return P


def _auroc_from_ranks(ranks: np.ndarray, lab: np.ndarray) -> float:
    """Mann-Whitney AUROC from precomputed ascending ranks over the subset."""
    n1 = float(lab.sum())
    n0 = float(len(lab) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[lab.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _perm_chunk_sorted(lab_s: np.ndarray, strata_s: np.ndarray, chunk: int, rng) -> np.ndarray:
    """(n, chunk) float64 labels permuted within activity strata, in R^2-sorted order.

    Vectorized per stratum: one `argsort` over a (m, chunk) uniform draw yields
    `chunk` independent within-stratum permutations at once, so no Python loop
    runs per draw.
    """
    out = np.empty((len(lab_s), chunk), dtype=np.float64)
    for s in np.unique(strata_s):
        i = np.flatnonzero(strata_s == s)
        vals = lab_s[i].astype(np.float64)
        perm = np.argsort(rng.random((len(i), chunk)), axis=0)
        out[i] = vals[perm]
    return out


def _mw_from_ranks(ranks: np.ndarray, lab_mat: np.ndarray, n_tot: int) -> np.ndarray:
    """Mann-Whitney AUROC per column of a (n, draws) 0/1 label matrix.

    One GEMM: with the R^2 ranks FIXED, a permuted label vector's AUROC is a
    weighted sum of those ranks, so every draw is a column of `ranks @ lab_mat`.
    """
    n1 = lab_mat.sum(axis=0)
    ok = (n1 > 0) & (n1 < n_tot)
    out = np.full(lab_mat.shape[1], np.nan)
    out[ok] = (ranks @ lab_mat[:, ok] - n1[ok] * (n1[ok] + 1) / 2) / (n1[ok] * (n_tot - n1[ok]))
    return out


def label_reads(bundle: dict, n_perm: int, n_boot: int, rng, chunk: int = 100) -> dict:
    """Per-label AUROC (labeled vs unlabeled R^2), prevalence-vs-rank profile, and
    the AUROC-at-depth sweep, each against the activity-stratified null.

    The permutation draws are CHUNKED: a full (115k, 2000) float64 label matrix
    is ~1.8 GB, and each depth also needs a sliced copy of it.
    """
    r2, act, labels = bundle["r2"], bundle["cov"]["activity"], bundle["labels"]

    out: dict[str, dict] = {}
    for coding, (axis, pos_fn, drop) in BINARY_AXES.items():
        raw = labels[axis]
        keep = np.array([s not in drop and s != "unlabeled" for s in raw])
        lab = np.array([1 if pos_fn(s) else 0 for s in raw[keep]], dtype=np.int8)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        r2k, actk = r2[keep], act[keep]
        nk = len(lab)

        # global AUROC (labeled vs unlabeled R^2) + group-conditional bootstrap CI
        point, ci = auroc_with_boot(r2k[lab == 1], r2k[lab == 0], n_boot, rng)

        ordk = np.argsort(-r2k)  # best-predicted first
        lab_s = lab[ordk]
        r2_s = r2k[ordk]
        strata_s = _decile_of(actk)[ordk]
        rank_all = PB._rank(r2_s)

        ks = [k for k in K_GRID if 2 * k <= nk]
        sels = [np.concatenate([np.arange(k), np.arange(nk - k, nk)]) for k in ks]
        rsubs = [PB._rank(r2_s[sel]) for sel in sels]
        a_obs = [_auroc_from_ranks(rs, lab_s[sel]) for rs, sel in zip(rsubs, sels, strict=True)]

        null_global: list[np.ndarray] = []
        null_k: list[list[np.ndarray]] = [[] for _ in ks]
        done = 0
        while done < n_perm:
            b = min(chunk, n_perm - done)
            P = _perm_chunk_sorted(lab_s, strata_s, b, rng)
            null_global.append(_mw_from_ranks(rank_all, P, nk))
            for j, (sel, rs) in enumerate(zip(sels, rsubs, strict=True)):
                null_k[j].append(_mw_from_ranks(rs, P[sel], len(sel)))
            done += b
        ng = np.concatenate(null_global)
        nk_draws = np.vstack([np.concatenate(v) for v in null_k])  # (K, n_perm)

        ngf = ng[np.isfinite(ng)]
        # Two-sided p AGAINST THE STRATIFIED NULL'S OWN CENTRE, not against 0.5:
        # stratifying on activity preserves the activity-label association, so the
        # null AUROC is NOT centred at chance (e.g. `interpretable` nulls at 0.424).
        # Testing deviation-from-0.5 would call an observed value sitting far
        # ABOVE its null band "not significant" purely because the null itself is
        # far below chance.
        null_mid = float(np.mean(ngf))
        p_perm = float(
            ((np.abs(ngf - null_mid) >= abs(point - null_mid)).sum() + 1) / (len(ngf) + 1)
        )

        # scan-corrected band: studentize each k by its own permutation
        # mean/sd, then threshold on the 95th percentile of max_k |z| per draw
        mu = np.nanmean(nk_draws, axis=1)
        sd = np.nanstd(nk_draws, axis=1)
        sd_safe = np.where(sd > 0, sd, np.nan)
        z_obs = (np.asarray(a_obs) - mu) / sd_safe
        z_perm = (nk_draws - mu[:, None]) / sd_safe[:, None]
        with np.errstate(invalid="ignore"):
            per_draw_max = np.nanmax(np.abs(z_perm), axis=0)
        scan_thresh = float(np.nanpercentile(per_draw_max, 95))

        depth = []
        for j, k in enumerate(ks):
            col = nk_draws[j][np.isfinite(nk_draws[j])]
            depth.append(
                {
                    "k": int(k),
                    "auroc": float(a_obs[j]),
                    "perm_band_2p5": float(np.percentile(col, 2.5)) if len(col) else float("nan"),
                    "perm_band_97p5": float(np.percentile(col, 97.5)) if len(col) else float("nan"),
                    "z": float(z_obs[j]),
                    "outside_scan_band": bool(
                        np.isfinite(z_obs[j]) and abs(z_obs[j]) > scan_thresh
                    ),
                    "n_positive_in_subset": int(lab_s[sels[j]].sum()),
                }
            )
        hit = [d["k"] for d in depth if d["outside_scan_band"]]
        k_star = int(max(hit)) if hit else None

        # prevalence-vs-R^2-rank profile (equal-count bins over the FULL ranking)
        bin_frac = 0.05 if lab.mean() < 0.02 else 0.02
        nbins = int(round(1 / bin_frac))
        edges = np.linspace(0, nk, nbins + 1).astype(int)
        prof = [
            {
                "bin": b,
                "n": int(edges[b + 1] - edges[b]),
                "prevalence": float(lab_s[edges[b] : edges[b + 1]].mean()),
            }
            for b in range(nbins)
        ]

        out[coding] = {
            "source_axis": axis,
            "n": int(nk),
            "n_positive": int(lab.sum()),
            "marginal_prevalence": float(lab.mean()),
            "auroc": point,
            "auroc_ci95": ci,
            "auroc_perm_p": p_perm,
            "auroc_perm_band": [
                float(np.percentile(ngf, 2.5)),
                float(np.percentile(ngf, 97.5)),
            ],
            "auroc_perm_null_mean": null_mid,
            "auroc_vs_null_direction": ("above" if point > null_mid else "below"),
            "scan_threshold_z": scan_thresh,
            "separation_depth_k": k_star,
            "auroc_at_depth": depth,
            "prevalence_profile": prof,
            "profile_bin_frac": bin_frac,
            "retired_axis": axis in RETIRED_AXES,
        }
        _log(
            f"{coding}: AUROC {point:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] p={p_perm:.4f} "
            f"prev={lab.mean():.3f} k*={k_star}"
        )
    return out


def continuous_auroc(bundle: dict, n_boot: int, rng) -> dict:
    """AUROC of each continuous predictor for top-k vs bottom-k R^2 membership."""
    r2 = bundle["r2"]
    n = len(r2)
    order = np.argsort(-r2)
    out = {}
    for key, (label, _lx) in CONTINUOUS.items():
        v = bundle["cov"][key][order]
        ks, pts, cis = [], [], []
        for k in [k for k in K_GRID if 2 * k <= n]:
            point, ci = auroc_with_boot(v[:k], v[-k:], n_boot, rng)
            ks.append(int(k))
            pts.append(point)
            cis.append(ci)
        out[key] = {"label": label, "k": ks, "auroc": pts, "auroc_ci95": cis}
        _log(f"AUROC {key}: k={ks[0]} {pts[0]:.3f} .. k={ks[-1]} {pts[-1]:.3f}")
    return out


# ── figures: one plot per predictor, two fixed templates ─────────────────────

R2_DISPLAY_FLOOR = -1.0  # R^2 clipped at -1 for display only (stated in captions)
RUG_MAX = 2000  # rug ticks are subsampled at full width; a 100k-tick rug is a solid bar


def _fig_continuous(bundle: dict, jm: dict, fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    by_key = {p["key"]: p for p in jm["predictors"]}
    r2 = np.clip(bundle["r2"], R2_DISPLAY_FLOOR, None)
    stems = []
    for key, (label, _declared_log) in CONTINUOUS.items():
        v = bundle["cov"][key]
        row = by_key[key]
        # log x for heavy-tailed predictors, decided from the DATA rather than a
        # hand-set flag: strictly positive and a p99/p50 spread of >= 5x
        p50, p99 = np.percentile(v, [50, 99])
        log_x = bool((v > 0).all() and p50 > 0 and p99 / p50 >= 5.0)
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.hexbin(
            v,
            r2,
            gridsize=110,
            bins="log",
            mincnt=1,
            xscale="log" if log_x else "linear",
            cmap="Blues",
            linewidths=0,
        )
        med = row["decile_median_r2"]
        edges = np.quantile(v, np.linspace(0, 1, N_DECILES + 1))
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(N_DECILES)]
        ax.plot(
            centers, med, "-", lw=1.4, color=paper_palette_role("accent"), label="decile median"
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label + (" (log)" if log_x else ""))
        ax.set_ylabel(r"per-feature held-out $R^2$")
        ax.legend(loc="lower right", frameon=False, fontsize=7.2)
        ax.text(
            0.03,
            0.97,
            f"$\\rho$ = {row['spearman_raw']:+.3f}\n"
            f"partial $\\rho$ = {row['partial_all_others']:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )
        set_title_subtitle(
            ax,
            label,
            f"{len(v):,} features; $R^2$ display-clipped at {R2_DISPLAY_FLOOR:g}; "
            f"log-count density; p99/p50 = {p99 / p50:.1f}",
        )
        fig.tight_layout()
        stem = f"cont_{key}"
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        stems.append(stem)
    return stems


def _fig_binary(bundle: dict, lr: dict, fig_dir: Path, rng) -> list[str]:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    r2, labels = bundle["r2"], bundle["labels"]
    stems = []
    for coding, res in lr.items():
        axis, pos_fn, drop = BINARY_AXES[coding]
        raw = labels[axis]
        keep = np.array([s not in drop and s != "unlabeled" for s in raw])
        lab = np.array([1 if pos_fn(s) else 0 for s in raw[keep]], dtype=np.int8)
        ordk = np.argsort(-r2[keep])
        lab_s = lab[ordk]
        nk = len(lab_s)

        prof = res["prevalence_profile"]
        nb = len(prof)
        centers = (np.arange(nb) + 0.5) / nb * 100.0
        vals = [b["prevalence"] for b in prof]

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.plot(centers, vals, "o-", ms=3.6, lw=1.3, color=paper_palette_role("primary"))
        ax.axhline(
            res["marginal_prevalence"],
            color=paper_palette_role("neutral"),
            lw=1.0,
            ls="--",
            label=f"marginal prevalence {res['marginal_prevalence']:.3f}",
        )
        hit = np.flatnonzero(lab_s) / nk * 100.0
        if len(hit) > RUG_MAX:
            hit = rng.choice(hit, size=RUG_MAX, replace=False)
        ylo = -0.10 * max(vals + [res["marginal_prevalence"]])
        ax.plot(
            hit,
            np.full(len(hit), ylo),
            "|",
            ms=9.0,
            mew=1.1,
            alpha=0.6,
            color=paper_palette_role("primary"),
            clip_on=False,
        )
        ax.set_ylim(ylo * 1.9, None)
        ax.set_xlabel(r"percentile of the $R^2$ ranking (0 = best-predicted)")
        ax.set_ylabel("fraction of bin carrying the label")
        ax.legend(loc="best", frameon=False, fontsize=7.2)
        ci = res["auroc_ci95"]
        ax.text(
            0.03,
            0.97,
            f"AUROC = {res['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]\n"
            f"stratified perm p = {res['auroc_perm_p']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
        )
        set_title_subtitle(
            ax,
            coding + (" [retired axis]" if res.get("retired_axis") else ""),
            f"n = {nk:,}; {int(res['n_positive']):,} positive; "
            f"{int(res['profile_bin_frac'] * 100)}% bins"
            + (f"; rug subsampled to {RUG_MAX:,}" if len(np.flatnonzero(lab_s)) > RUG_MAX else ""),
        )
        fig.tight_layout()
        stem = f"bin_{coding}"
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        stems.append(stem)
    return stems


def _fig_auroc_depth(lr: dict, fig_dir: Path) -> str:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    names = list(lr)
    colors = paper_palette(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for name, color in zip(names, colors, strict=True):
        d = lr[name]["auroc_at_depth"]
        ks = [r["k"] for r in d]
        av = [r["auroc"] for r in d]
        ax.plot(ks, av, "o-", ms=3.2, lw=1.2, color=color, label=name)
        kstar = lr[name]["separation_depth_k"]
        if kstar is not None and kstar in ks:
            j = ks.index(kstar)
            ax.plot([kstar], [av[j]], "o", ms=8.0, mfc="none", mec=color, mew=1.5)
    ax.axhline(0.5, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"tail width $k$ (AUROC computed among top-$k$ $\cup$ bottom-$k$ only)")
    ax.set_ylabel("AUROC at depth $k$")
    ax.legend(loc="best", frameon=False, fontsize=7.0, ncol=2)
    set_title_subtitle(
        ax,
        "How deep into the ranking does each judged label separate?",
        "ring = separation depth $k^*$; rightmost point = global AUROC",
    )
    fig.tight_layout()
    savefig_paper(fig, "summary_auroc_depth_overlay", dir=fig_dir)
    plt.close(fig)
    return "summary_auroc_depth_overlay"


def _fig_summaries(jm: dict, fig_dir: Path) -> list[str]:
    """Optional extras: the partial-rho forest and the joint-ROC headline."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    preds = jm["predictors"]
    names = [p["label"] + (" [retired]" if p.get("retired_axis") else "") for p in preds][::-1]
    raw = np.array([p["spearman_raw"] for p in preds])[::-1]
    raw_ci = np.array([p["spearman_raw_ci95"] for p in preds])[::-1]
    par = np.array([p["partial_all_others"] for p in preds])[::-1]
    par_ci = np.array([p["partial_all_others_ci95"] for p in preds])[::-1]
    ypos = np.arange(len(names), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(names) + 1.9))
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax.errorbar(
        raw,
        ypos + 0.17,
        xerr=PB._errbars(raw, raw_ci),
        fmt="o",
        ms=4.0,
        color=paper_palette_role("baseline"),
        mfc="white",
        mew=1.2,
        lw=1.0,
        capsize=2.0,
        label="raw Spearman",
    )
    ax.errorbar(
        par,
        ypos - 0.17,
        xerr=PB._errbars(par, par_ci),
        fmt="o",
        ms=4.0,
        color=paper_palette_role("primary"),
        lw=1.0,
        capsize=2.0,
        label="all-others-partial Spearman",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel(r"Spearman $\rho$ vs per-feature held-out $R^2$")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    roc = jm["joint_roc"]
    set_title_subtitle(
        ax,
        "Full dictionary: what survives partialling every other predictor",
        f"{jm['design']['n_features']:,} features; joint rank-OLS $R^2$ = {jm['joint_r2']:.3f}; "
        f"joint top-decile ROC AUROC = {roc['auroc']:.3f}",
    )
    fig.tight_layout()
    savefig_paper(fig, "summary_forest", dir=fig_dir)
    plt.close(fig)
    return ["summary_forest"]


def figures(bundle: dict, jm: dict, lr: dict, fig_dir: Path, rng) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    per = fig_dir / "per_predictor"
    per.mkdir(parents=True, exist_ok=True)
    stems = _fig_continuous(bundle, jm, per)
    stems += _fig_binary(bundle, lr, per, rng)
    stems.append(_fig_auroc_depth(lr, fig_dir))
    stems += _fig_summaries(jm, fig_dir)
    return stems


def main() -> None:
    ap = argparse.ArgumentParser(description="#1482 full-dictionary predictor battery")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--work", type=Path, default=PROJECT_ROOT / "data/issue_1482/fullwidth")
    ap.add_argument("--phase", default="all", choices=("all", "scan", "analyze", "figs"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_boot, args.n_perm = 50, 50
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    if args.phase == "scan":
        scan = scan_pooled_store(POOLED_STORE, args.work / "fullwidth_scan.npz")
        identity_gates(scan)
        return

    if args.phase == "figs":
        # re-render from the committed matrix + JSONs (the analysis is ~28 min;
        # a figure-spec change must not re-run it)
        with np.load(args.out_dir / "fullwidth_matrix.npz", allow_pickle=True) as z:
            bundle = {
                "feat_ids": z["feat_ids"],
                "r2": z["r2"],
                "cov": {k: z[k] for k in CONTINUOUS},
                "labels": {k[len("label__") :]: z[k] for k in z.files if k.startswith("label__")},
            }
        jm = json.loads((args.out_dir / "fullwidth_joint_model.json").read_text())
        lr = json.loads((args.out_dir / "fullwidth_label_reads.json").read_text())["label_reads"]
        stems = figures(bundle, jm, lr, args.fig_dir, np.random.default_rng(SEED))
        _log(f"figures ({len(stems)}): {', '.join(stems)}")
        return

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    bundle = assemble(args.out_dir, args.work)
    np.savez(
        args.out_dir / "fullwidth_matrix.npz",
        feat_ids=bundle["feat_ids"],
        r2=bundle["r2"],
        **{f"r2_{a}": v for a, v in bundle["r2_arms"].items()},
        **bundle["cov"],
        **{f"label__{ax}": v.astype(str) for ax, v in bundle["labels"].items()},
    )

    jm = joint_model(bundle, args.n_boot, rng)
    jm["bridge_to_panel"] = bridge_check(bundle)
    jm["joint_roc"] = joint_roc(bundle, rng)
    jm["metadata"] = PB._metadata()
    (args.out_dir / "fullwidth_joint_model.json").write_text(json.dumps(jm, indent=1))

    lr = label_reads(bundle, args.n_perm, args.n_boot, rng)
    ca = continuous_auroc(bundle, args.n_boot, rng)
    doc = {
        "design": {
            "scope": "FULL DICTIONARY",
            "r2_source": f"{R2_DIR}/{R2_ARMS[PRIMARY_ARM]}",
            "corpus_caveat": jm["design"]["corpus_caveat"],
            "n_perm": args.n_perm,
            "n_boot": args.n_boot,
            "k_grid": [int(k) for k in K_GRID],
            "auroc_definition": (
                "P(R^2 of labeled > R^2 of unlabeled) + 0.5 P(tie) (Mann-Whitney), "
                "group-conditional bootstrap CI, activity-decile-stratified label-shuffle p"
            ),
            "seed": SEED,
        },
        "label_reads": lr,
        "continuous_auroc": ca,
        "metadata": PB._metadata(),
    }
    (args.out_dir / "fullwidth_label_reads.json").write_text(json.dumps(doc, indent=1))
    _log(f"analysis done in {time.time() - t0:.0f}s -> {args.out_dir}")

    stems = figures(bundle, jm, lr, args.fig_dir, rng)
    _log(f"figures ({len(stems)}): {', '.join(stems)}")
    _log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
