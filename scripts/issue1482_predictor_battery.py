#!/usr/bin/env python
"""Issue #1482 inline free-analysis round: joint per-feature predictor battery.

Three deliverables over the layer-19 answer-side SAE panel (16,384 features,
joined by ``feat_id``), all on banked/cached arrays — 0 GPU, no new judge calls.

D1 (joint model, filled-doc suggested analysis 3): assemble every banked
    per-feature predictor of held-out map R^2 into ONE design matrix and fit a
    rank OLS. Per predictor: raw Spearman, ALL-OTHERS-PARTIAL Spearman, and
    leave-one-predictor-out Delta-R^2 (dominance), each with a bootstrap 95% CI
    over features. Also the joint R^2 with vs without within-answer consistency
    (how much consistency subsumes the rest).

D2 (label reads + tail extension, suggested analysis 2 + an AUROC sweep):
    panel-wide Spearman of `interpretable` and of each `content_type`
    one-vs-rest against R^2, with the activity-decile-stratified permutation
    null; the tail-depth sweep (Delta_k = frac(label|top-k) - frac(label|bot-k))
    on the banked k grid for the same axes; and AUROC of each continuous
    predictor for top-k vs bottom-k R^2 membership over that grid.

D3: encoder-norm correlate (suggested analysis 4) — computed here from the SAE
    weights and carried as a predictor in D1.

Reuse: the tail-sweep machinery (modal labels, activity deciles, within-stratum
permutation, Wilson CI, k grid) is imported from `issue1482_tail_depth_sweep`,
so this round's nulls are constructed identically to the banked sweep; the SAE
loader is `issue1482_sae.BatchTopKSAE`; the dense holdout states come from
`issue1482_residual_svd.load_layer`.

Vectorized throughout: the bootstrap draws all 2,000 resampled correlation
matrices with one GEMM over per-row outer products (no Python loop per draw),
and the AUROC bootstrap uses a weighted-CDF gather rather than pairwise
comparisons.

Wiring gates (fail-loud before any read): the four banked per-predictor
Spearman values must reproduce from the assembled matrix.
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

import issue1482_tail_depth_sweep as TDS  # noqa: E402

PERFEATURE = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
CONSISTENCY = "eval_results/issue_1482/feature_correlates/consistency_perfeature.npz"
FOOTPRINT = "eval_results/issue_1482/footprint_moments/footprint_moments.npz"
FEATURE_TABLE = "eval_results/issue_1773/feature_table_v1.jsonl"
OUT_DIR = "eval_results/issue_1482/predictor_battery"
FIG_DIR = "figures/issue_1482/predictor_battery"

SEED = 1482
N_BOOT = 2000
SAE_LAYER = 19
SAE_K = 64

# Banked per-predictor Spearman values the assembled matrix must reproduce.
# (consistency.json, footprint_moments.json, dense_projection_variance.json)
WIRING_GATES = {
    "consistency": 0.6003225916100939,
    "activity": 0.29289787909943515,
    "write_norm": 0.1837599145161954,
    "proj_var": 0.40372375815300304,  # measured on log(proj_var); rank-identical
}
GATE_TOL = 1e-3

# Continuous predictors: matrix key -> human label for figures/report.
CONTINUOUS = {
    "consistency": "within-answer consistency",
    "activity": "activity (firing frequency)",
    "proj_var": "dense variance along decoder dir",
    "write_norm": "write norm (gamma-scaled)",
    "footprint_var": "footprint variance",
    "footprint_skew": "footprint skew",
    "footprint_kurt": "footprint kurtosis",
    "enc_norm": "encoder-vector norm",
    "side_ratio": "side ratio",
    "rb_align_max": "rb_align (max over 3 traits)",
    "abstraction_ord": "abstraction (ordinal)",
}

# Judged categorical axes -> (reference level, non-reference levels, aliases
# folded into "unresolved"). The reference level is dropped from the design
# matrix; "unresolved" is carried as its own level so no feature is dropped.
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
# Axes whose judged labels are RETIRED as unusable (#1941 -- inter-draw kappa
# 0.318 vs 0.63-0.71 siblings). Carried in the model because the round's brief
# asks for them; flagged everywhere so no partial rho is read as a finding.
RETIRED_AXES = ("functional_role",)

# Predictors the assembly deliberately drops, with the evidence recorded in the
# output JSON (a degenerate or duplicate column makes the joint model singular).
EXCLUSIONS_DOC = {
    "dec_norm": (
        "raw decoder-vector norm is unit by construction for this BatchTopK SAE "
        "(measured std 2.7e-08 over the panel) -- zero variance, no rank information"
    ),
    "enc_dec_ratio": (
        "enc_norm / dec_norm with dec_norm constant == enc_norm up to a scale factor; "
        "rank-identical to enc_norm, so it is the same column"
    ),
    "persist_answer_mean": (
        "rank-identical to `consistency` (measured Spearman 1.000000 over the panel) -- "
        "the same quantity under two names"
    ),
    "matryoshka_tier": (
        "excluded by the round's scope: a different dictionary and feature panel, "
        "so it does not join to this layer-19 BatchTopK panel"
    ),
}

# Continuous predictors carried into the AUROC sweep (top-k vs bottom-k R^2).
AUROC_PREDICTORS = ("consistency", "activity", "proj_var", "write_norm", "enc_norm")


def _log(msg: str) -> None:
    print(f"[predictor-battery] {msg}", flush=True)


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as float64."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    srt = a[order]
    i = 0
    while i < len(srt):
        j = i + 1
        while j < len(srt) and srt[j] == srt[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


# ── assembly ──────────────────────────────────────────────────────────────────


def _dense_projection_variance(feat_ids: np.ndarray, w_dec: np.ndarray) -> np.ndarray:
    """Var over the #1738 multi-turn holdout of the dense state projected onto
    each panel feature's UNIT decoder direction.

    Computed as u^T Cov(Yc) u rather than materializing the (n, 16384) projection:
    one 3584x3584 covariance plus one GEMM. Reproduces the banked summary
    Spearman (`feature_correlates/dense_projection_variance.json`) to 7 decimals.
    """
    import issue1482_residual_svd as RS

    y16, _ci = RS.load_layer(SAE_LAYER)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
    unit = w_dec[:, feat_ids]
    unit = unit / np.linalg.norm(unit, axis=0, keepdims=True)
    return np.einsum("df,df->f", cov @ unit, unit)


def _table_numerics(feat_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Non-label numeric per-feature fields from the #1773 feature table.

    Judged LABELS are read from `issue1482_tail_depth_sweep._modal_labels`
    instead (same source, banked-sweep parity); this reads only the numeric
    columns the sweep does not carry.
    """
    by_id: dict[int, dict] = {}
    with (PROJECT_ROOT / FEATURE_TABLE).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_id[int(rec["feat_id"])] = rec
    missing = [int(i) for i in feat_ids if int(i) not in by_id]
    if missing:
        raise AssertionError(f"{len(missing)} panel features absent from {FEATURE_TABLE}")

    side = np.empty(len(feat_ids))
    persist = np.empty(len(feat_ids))
    rb = np.empty(len(feat_ids))
    for i, fid in enumerate(feat_ids):
        rec = by_id[int(fid)]
        side[i] = rec["side_ratio"]
        persist[i] = rec["persist_answer"]["mean"]
        rb[i] = max(v["raw"] for v in rec["rb_align"].values())
    return {"side_ratio": side, "persist_answer_mean": persist, "rb_align_max": rb}


def _label_arrays(feat_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Modal judged label per panel feature per axis (object array of str)."""
    out: dict[str, np.ndarray] = {}
    for axis in ("abstraction", *CATEGORICAL):
        modal = TDS._modal_labels(axis)
        out[axis] = np.array(
            [modal.get(int(fid), "unresolved") or "unresolved" for fid in feat_ids], dtype=object
        )
    return out


def assemble(work: Path) -> dict:
    """Join every banked predictor onto the layer-19 panel; run the wiring gates."""
    from issue1482_sae import BatchTopKSAE

    t0 = time.time()
    z = np.load(PROJECT_ROOT / PERFEATURE)
    feat_ids = np.asarray(z["feat_ids"], dtype=int)
    r2 = np.asarray(z["r2"], dtype=np.float64)
    activity = np.asarray(z["activity"], dtype=np.float64)
    n = len(feat_ids)
    _log(f"panel: {n} features, r2 median {np.median(r2):.4f}")

    cons = np.load(PROJECT_ROOT / CONSISTENCY)
    if not np.array_equal(np.asarray(cons["feat_ids"], dtype=int), feat_ids):
        raise AssertionError("consistency npz feat_ids do not match the panel")
    consistency = np.asarray(cons["consistency"], dtype=np.float64)

    fp = np.load(PROJECT_ROOT / FOOTPRINT)
    cols = {
        "write_norm": "direct_write_norm",
        "footprint_var": "direct_var",
        "footprint_skew": "direct_skew",
        "footprint_kurt": "direct_kurt",
    }
    foot = {k: np.asarray(fp[v], dtype=np.float64)[feat_ids] for k, v in cols.items()}
    dec_norm_banked = np.asarray(fp["dec_norm"], dtype=np.float64)[feat_ids]

    sae = BatchTopKSAE.load(k=SAE_K, layer=SAE_LAYER, device="cpu")
    w_dec = np.asarray(sae.w_dec, dtype=np.float64)
    w_enc = np.asarray(sae.w_enc, dtype=np.float64)
    enc_norm = np.linalg.norm(w_enc[feat_ids, :], axis=1)
    dec_norm = np.linalg.norm(w_dec[:, feat_ids], axis=0)
    _log(f"SAE norms: enc median {np.median(enc_norm):.4f}, dec std {dec_norm.std():.3e}")
    if not np.allclose(dec_norm, dec_norm_banked, atol=1e-5):
        raise AssertionError("recomputed decoder norms disagree with the banked footprint npz")

    proj_var = _dense_projection_variance(feat_ids, w_dec)
    _log(f"projection variance computed ({time.time() - t0:.1f}s)")

    tab = _table_numerics(feat_ids)
    labels = _label_arrays(feat_ids)

    # abstraction as an ordinal, unresolved imputed at the ordinal median and
    # carried by its own indicator (missing-indicator coding: no row is dropped
    # and the imputation cannot manufacture a rank signal on its own).
    ord_map = {"token_surface": 0.0, "lexical_semantic": 1.0, "abstract_contextual": 2.0}
    abst = labels["abstraction"]
    known = np.array([s in ord_map for s in abst])
    abstraction_ord = np.full(n, np.nan)
    abstraction_ord[known] = [ord_map[s] for s in abst[known]]
    abstraction_ord[~known] = np.median(abstraction_ord[known])

    mat = {
        "feat_ids": feat_ids,
        "r2": r2,
        "activity": activity,
        "consistency": consistency,
        "proj_var": proj_var,
        "enc_norm": enc_norm,
        "dec_norm": dec_norm,
        "abstraction_ord": abstraction_ord,
        "abstraction_unresolved": (~known).astype(np.float64),
        **foot,
        **tab,
    }

    gates = {}
    for key, banked in WIRING_GATES.items():
        got = _spearman(mat[key], r2)
        gates[key] = {"observed": got, "banked": banked, "delta": got - banked}
        if abs(got - banked) > GATE_TOL:
            raise AssertionError(
                f"wiring gate {key}: observed rho {got:.6f} vs banked {banked:.6f} "
                f"(|delta| {abs(got - banked):.2e} > {GATE_TOL})"
            )
    _log("wiring gates PASS: " + ", ".join(f"{k} {v['observed']:+.4f}" for k, v in gates.items()))

    # degeneracy evidence for the recorded exclusions
    degeneracy = {
        "dec_norm_std": float(dec_norm.std()),
        "dec_norm_min": float(dec_norm.min()),
        "dec_norm_max": float(dec_norm.max()),
        "spearman_persist_answer_vs_consistency": _spearman(
            tab["persist_answer_mean"], consistency
        ),
    }

    for axis, arr in labels.items():
        mat[f"label__{axis}"] = arr.astype(str)
    np.savez(work / "predictor_matrix.npz", **mat)
    return {
        "matrix": mat,
        "labels": labels,
        "gates": gates,
        "degeneracy": degeneracy,
        "n": n,
    }


def build_design(mat: dict, labels: dict) -> tuple[np.ndarray, list[str], dict]:
    """Rank-transformed continuous columns + one-hot judged levels.

    Returns the design matrix (n, p), the ordered predictor keys, and a legend
    mapping each key to its human label / kind / reference level.
    """
    cols: list[np.ndarray] = []
    keys: list[str] = []
    legend: dict[str, dict] = {}

    for key, label in CONTINUOUS.items():
        cols.append(_rank(mat[key]))
        keys.append(key)
        legend[key] = {"label": label, "kind": "continuous (rank-transformed)"}

    cols.append(mat["abstraction_unresolved"])
    keys.append("abstraction_unresolved")
    legend["abstraction_unresolved"] = {
        "label": "abstraction: unresolved",
        "kind": "missing indicator",
        "note": "pairs with the median-imputed abstraction ordinal",
    }

    for axis, (ref, levels, aliases) in CATEGORICAL.items():
        raw = labels[axis]
        norm = np.array([("unresolved" if s in aliases else s) for s in raw], dtype=object)
        for lvl in levels:
            key = f"{axis}__{lvl}"
            cols.append((norm == lvl).astype(np.float64))
            keys.append(key)
            legend[key] = {
                "label": f"{axis.replace('_', ' ')}: {lvl}",
                "kind": "one-hot",
                "reference_level": ref,
                "n_positive": int((norm == lvl).sum()),
                "retired_axis": axis in RETIRED_AXES,
            }
    return np.column_stack(cols), keys, legend


# ── joint model ───────────────────────────────────────────────────────────────


def _corr_reads(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Raw rho, all-others-partial rho, joint R^2, and LOPO Delta-R^2.

    `corr` is the (p+1, p+1) correlation matrix of [X, y] with y LAST. Every
    read is a function of the precision matrix, which is what makes the
    bootstrap a batched inverse rather than 2,000 refits:
      partial rho_jy = -P_jy / sqrt(P_jj P_yy)
      joint R^2      = 1 - 1 / P_yy
      Delta-R^2_j    = pr^2 (1 - R^2) / (1 - pr^2)
    """
    prec = np.linalg.pinv(corr)
    y = corr.shape[0] - 1
    d = np.diag(prec)
    partial = -prec[:y, y] / np.sqrt(d[:y] * d[y])
    r2_joint = 1.0 - 1.0 / d[y]
    pr2 = partial**2
    delta = pr2 * (1.0 - r2_joint) / np.maximum(1.0 - pr2, 1e-12)
    return corr[:y, y].copy(), partial, float(r2_joint), delta


def _batched_corr_reads(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`_corr_reads` over a stack of (B, m, m) correlation matrices."""
    prec = np.linalg.pinv(corr)
    y = corr.shape[-1] - 1
    d = np.diagonal(prec, axis1=-2, axis2=-1)
    partial = -prec[:, :y, y] / np.sqrt(d[:, :y] * d[:, y : y + 1])
    r2_joint = 1.0 - 1.0 / d[:, y]
    pr2 = partial**2
    delta = pr2 * (1.0 - r2_joint[:, None]) / np.maximum(1.0 - pr2, 1e-12)
    return corr[:, :y, y].copy(), partial, r2_joint, delta


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of an OLS fit of y on x (intercept added)."""
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(1.0 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))


def _bootstrap_reads(z: np.ndarray, n_boot: int, rng, chunk: int = 400) -> dict[str, np.ndarray]:
    """Bootstrap the four reads over features with one GEMM per chunk of draws.

    Ranks are computed ONCE on the full sample and then resampled (rank-then-
    resample); re-ranking inside every draw would need a (n_boot*p, n) argsort.
    """
    n, m = z.shape
    zc = (z - z.mean(0)) / z.std(0)
    outer = (zc[:, :, None] * zc[:, None, :]).reshape(n, m * m)

    raw, partial, joint, delta = [], [], [], []
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(n, b))
        flat = (idx + np.arange(b) * n).ravel()
        w = np.bincount(flat, minlength=n * b).reshape(b, n).T.astype(np.float64)
        s2 = (outer.T @ w).reshape(m, m, b).transpose(2, 0, 1) / n  # (b, m, m)
        s1 = (zc.T @ w).T / n  # (b, m)
        cov = s2 - s1[:, :, None] * s1[:, None, :]
        sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-300))
        corr = cov / (sd[:, :, None] * sd[:, None, :])
        r, p, j, dl = _batched_corr_reads(corr)
        raw.append(r)
        partial.append(p)
        joint.append(j)
        delta.append(dl)
        done += b
    return {
        "raw": np.concatenate(raw),
        "partial": np.concatenate(partial),
        "joint": np.concatenate(joint),
        "delta": np.concatenate(delta),
    }


def _ci(draws: np.ndarray) -> list[float]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def joint_model(bundle: dict, n_boot: int, rng) -> dict:
    mat, labels = bundle["matrix"], bundle["labels"]
    x, keys, legend = build_design(mat, labels)
    y = _rank(mat["r2"])
    n, p = x.shape
    _log(f"design matrix: {n} features x {p} predictors")

    z = np.column_stack([x, y])
    zc = (z - z.mean(0)) / z.std(0)
    corr = (zc.T @ zc) / n
    raw, partial, r2_joint, delta = _corr_reads(corr)

    # explicit leave-one-predictor-out refits as a self-check on the closed form
    full_refit = _ols_r2(x, y)
    lopo_refit = np.array([full_refit - _ols_r2(np.delete(x, j, axis=1), y) for j in range(p)])
    lopo_max_delta = float(np.max(np.abs(lopo_refit - delta)))
    _log(
        f"joint R^2 {r2_joint:.4f} (refit {full_refit:.4f}); LOPO closed-form vs refit "
        f"max |delta| {lopo_max_delta:.2e}"
    )

    # collinearity diagnostics: worst |rho| each predictor has with another
    pred_corr = np.abs(corr[:p, :p]) - np.eye(p)
    max_pair = pred_corr.max(axis=1)
    partner = [keys[int(i)] for i in pred_corr.argmax(axis=1)]

    t0 = time.time()
    boot = _bootstrap_reads(z, n_boot, rng)
    _log(f"bootstrap: {n_boot} draws in {time.time() - t0:.1f}s")

    # consistency-subsumption: joint R^2 with vs without the dominant predictor
    ci_idx = keys.index("consistency")
    r2_wo_consistency = _ols_r2(np.delete(x, ci_idx, axis=1), y)
    only_consistency = _ols_r2(x[:, [ci_idx]], y)

    order = np.argsort(-np.abs(partial))
    predictors = []
    for j in order:
        key = keys[j]
        predictors.append(
            {
                "key": key,
                **legend[key],
                "spearman_raw": float(raw[j]),
                "spearman_raw_ci95": _ci(boot["raw"][:, j]),
                "partial_all_others": float(partial[j]),
                "partial_all_others_ci95": _ci(boot["partial"][:, j]),
                "lopo_delta_r2": float(delta[j]),
                "lopo_delta_r2_ci95": _ci(boot["delta"][:, j]),
                "lopo_delta_r2_refit": float(lopo_refit[j]),
                "max_abs_rho_with_other_predictor": float(max_pair[j]),
                "max_abs_rho_partner": partner[j],
            }
        )

    return {
        "design": {
            "question": (
                "Which per-feature properties predict the context->answer map's held-out "
                "R^2 once every other property is partialled out?"
            ),
            "target": "per-feature held-out R^2, ridge, mean pooling (rank-transformed)",
            "r2_source": PERFEATURE,
            "n_features": int(n),
            "n_predictors": int(p),
            "model": (
                "OLS of rank(R^2) on rank-transformed continuous predictors + one-hot judged "
                "levels (reference level dropped per axis). Raw rho, all-others-partial rho, "
                "joint R^2 and LOPO Delta-R^2 are all read off the precision matrix of the "
                "[X, y] correlation matrix; the closed-form LOPO is cross-checked against "
                "explicit refits."
            ),
            "n_boot": int(n_boot),
            "bootstrap": (
                f"{n_boot} draws resampling features with replacement, vectorized as one GEMM "
                "over per-row outer products per chunk. Ranks are computed once on the full "
                "sample and then resampled (rank-then-resample)."
            ),
            "missing_labels": (
                "'unresolved' is carried as its own one-hot level (and as a missing indicator "
                "for the abstraction ordinal) so no feature is dropped from the joint fit"
            ),
            "seed": SEED,
            "mlp_twin": (
                "ridge-only: the banked sae_ctx__mean__mlp.npz is the DIVERGED fit (all 16,384 "
                "per-feature R^2 negative, median -4.8e13; final.json records it as "
                "broken_mlp_reference) and the sae_sae_mlp_recovery round emitted pooled "
                "scalars only, no recovered per-feature array. At the PC grain ridge-vs-MLP "
                "rank agreement is 0.997, so the ranking read is not expected to move."
            ),
        },
        "wiring_gates": bundle["gates"],
        "excluded_predictors": {
            k: {"reason": v, **({"evidence": bundle["degeneracy"]} if k == "dec_norm" else {})}
            for k, v in EXCLUSIONS_DOC.items()
        },
        "degeneracy_evidence": bundle["degeneracy"],
        "retired_axes": {
            ax: "inter-draw kappa 0.318 (#1941, RETIRED as unusable) -- carried for completeness; "
            "its partial rho is not interpretable as a finding"
            for ax in RETIRED_AXES
        },
        "joint_r2": float(r2_joint),
        "joint_r2_ci95": _ci(boot["joint"]),
        "joint_r2_refit": full_refit,
        "consistency_subsumption": {
            "joint_r2_with_consistency": float(full_refit),
            "joint_r2_without_consistency": float(r2_wo_consistency),
            "delta": float(full_refit - r2_wo_consistency),
            "consistency_alone_r2": float(only_consistency),
        },
        "lopo_closed_form_vs_refit_max_abs_delta": lopo_max_delta,
        "predictors": predictors,
    }


# ── D2: label reads, tail-depth extension, AUROC sweep ───────────────────────


def _binary_axes(labels: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """coding -> (keep mask over the panel, 0/1 label over the kept rows)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    interp = labels["interpretable"]
    keep = np.array([s in ("yes", "no") for s in interp])
    out["interpretable"] = (keep, (interp[keep] == "yes").astype(np.int8))

    content = labels["content_type"]
    keep_c = np.array([s != "unresolved" for s in content])
    for lvl in ("syntax", "topic", "operation", "entity", "task_format"):
        out[f"content_{lvl}"] = (keep_c, (content[keep_c] == lvl).astype(np.int8))
    return out


def _stratified_panel_read(
    lab: np.ndarray, r2: np.ndarray, activity: np.ndarray, n_perm: int, rng
) -> dict:
    """Panel-wide point-biserial Spearman + activity-decile-stratified null."""
    strata = TDS._decile_of(activity)
    perm = TDS._perm_within_strata(lab, strata, rng)[:, :n_perm].astype(np.float64)
    ry = _rank(r2)
    ry = (ry - ry.mean()) / ry.std()
    obs = float(lab.astype(np.float64) @ ry / len(ry) / lab.astype(np.float64).std())
    pc = (perm - perm.mean(0)) / np.maximum(perm.std(0), 1e-12)
    null = (ry @ pc) / len(ry)
    p = float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1))
    return {
        "n": int(len(lab)),
        "n_positive": int(lab.sum()),
        "prevalence": float(lab.mean()),
        "spearman_vs_r2": obs,
        "perm_band_2p5": float(np.percentile(null, 2.5)),
        "perm_band_97p5": float(np.percentile(null, 97.5)),
        "perm_p": p,
        "outside_band": bool(obs < np.percentile(null, 2.5) or obs > np.percentile(null, 97.5)),
    }


def _tail_sweep(lab: np.ndarray, r2: np.ndarray, activity: np.ndarray, n_perm: int, rng) -> dict:
    """Delta_k sweep with the banked recipe's stratified + scan-corrected bands."""
    n = len(lab)
    order = np.argsort(-r2)
    lab_sorted = lab[order]
    strata = TDS._decile_of(activity)
    ks = np.asarray([k for k in TDS.K_GRID if 2 * k <= n], dtype=int)

    cs_top = np.cumsum(lab_sorted, dtype=np.int64)
    cs_bot = np.cumsum(lab_sorted[::-1], dtype=np.int64)
    d_obs = cs_top[ks - 1] / ks - cs_bot[ks - 1] / ks

    perm = TDS._perm_within_strata(lab, strata, rng)[:, :n_perm][order]
    cs_top_p = np.cumsum(perm, axis=0, dtype=np.int64)
    cs_bot_p = np.cumsum(perm[::-1], axis=0, dtype=np.int64)
    d_perm = cs_top_p[ks - 1] / ks[:, None] - cs_bot_p[ks - 1] / ks[:, None]

    lo = np.percentile(d_perm, 2.5, axis=1)
    hi = np.percentile(d_perm, 97.5, axis=1)
    mu, sd = d_perm.mean(axis=1), d_perm.std(axis=1)
    z_obs = (d_obs - mu) / np.maximum(sd, 1e-12)
    z_perm = (d_perm - mu[:, None]) / np.maximum(sd[:, None], 1e-12)
    scan_thresh = float(np.percentile(np.abs(z_perm).max(axis=0), 95))
    p_per_k = ((np.abs(d_perm) >= np.abs(d_obs)[:, None]).sum(axis=1) + 1) / (n_perm + 1)

    def _depth(mask: np.ndarray) -> int | None:
        hit = ks[mask]
        return int(hit.max()) if len(hit) else None

    return {
        "n": int(n),
        "n_positive": int(lab.sum()),
        "marginal_prevalence": float(lab.mean()),
        "k": [int(v) for v in ks],
        "delta_obs": [float(v) for v in d_obs],
        "perm_band_pointwise_2p5": [float(v) for v in lo],
        "perm_band_pointwise_97p5": [float(v) for v in hi],
        "perm_p_per_k": [float(v) for v in p_per_k],
        "scan_threshold_z": scan_thresh,
        "depth_pointwise": _depth((d_obs < lo) | (d_obs > hi)),
        "depth_scan_corrected": _depth(np.abs(z_obs) > scan_thresh),
    }


def _auroc_with_boot(
    x_top: np.ndarray, x_bot: np.ndarray, n_boot: int, rng
) -> tuple[float, list[float]]:
    """AUROC (P(top > bot) + 0.5 ties) plus a group-conditional bootstrap CI.

    Vectorized: the bottom group's resampled weights become a weighted CDF that
    every top item indexes with one searchsorted, so no draw is a Python loop
    and no pairwise (k x k) comparison is ever formed.
    """
    kt, kb = len(x_top), len(x_bot)
    srt = np.sort(x_bot)
    lo = np.searchsorted(srt, x_top, side="left")
    hi = np.searchsorted(srt, x_top, side="right")
    point = float((lo + 0.5 * (hi - lo)).sum() / (kt * kb))

    idx_b = rng.integers(0, kb, size=(kb, n_boot))
    flat = (idx_b + np.arange(n_boot) * kb).ravel()
    wb = np.bincount(flat, minlength=kb * n_boot).reshape(n_boot, kb).T.astype(np.float64)
    # reorder bottom weights into sorted-x order, then cumulate
    order_b = np.argsort(x_bot)
    cw = np.cumsum(wb[order_b], axis=0)
    zero = np.zeros((1, n_boot))
    cw = np.vstack([zero, cw])  # cw[i] = total weight of the i smallest
    u = cw[lo] + 0.5 * (cw[hi] - cw[lo])  # (kt, n_boot)

    idx_t = rng.integers(0, kt, size=(kt, n_boot))
    flat_t = (idx_t + np.arange(n_boot) * kt).ravel()
    wt = np.bincount(flat_t, minlength=kt * n_boot).reshape(n_boot, kt).T.astype(np.float64)
    draws = (wt * u).sum(axis=0) / (kt * kb)
    return point, _ci(draws)


def tail_extension(bundle: dict, n_perm: int, n_boot: int, rng) -> dict:
    mat, labels = bundle["matrix"], bundle["labels"]
    r2, activity = mat["r2"], mat["activity"]
    axes = _binary_axes(labels)

    panel, sweep = {}, {}
    for coding, (keep, lab) in axes.items():
        panel[coding] = _stratified_panel_read(lab, r2[keep], activity[keep], n_perm, rng)
        sweep[coding] = _tail_sweep(lab, r2[keep], activity[keep], n_perm, rng)
        _log(
            f"{coding}: rho {panel[coding]['spearman_vs_r2']:+.4f} "
            f"(p {panel[coding]['perm_p']:.4f}), scan depth {sweep[coding]['depth_scan_corrected']}"
        )

    order = np.argsort(-r2)
    n = len(r2)
    ks = [k for k in TDS.K_GRID if 2 * k <= n]
    auroc: dict[str, dict] = {}
    for key in AUROC_PREDICTORS:
        vals = mat[key][order]
        pts, cis = [], []
        for k in ks:
            point, ci = _auroc_with_boot(vals[:k], vals[-k:], n_boot, rng)
            pts.append(point)
            cis.append(ci)
        auroc[key] = {
            "label": CONTINUOUS[key],
            "k": [int(v) for v in ks],
            "auroc": pts,
            "auroc_ci95": cis,
        }
        _log(f"AUROC {key}: k=25 {pts[0]:.3f} .. k={ks[-1]} {pts[-1]:.3f}")

    return {
        "design": {
            "question": (
                "Do `interpretable` and `content_type` separate the R^2 ranking panel-wide or "
                "only in the tails, and how well does each continuous predictor classify "
                "top-k vs bottom-k R^2 membership?"
            ),
            "r2_source": PERFEATURE,
            "labels": "modal per-feature #1773 axis labels, recovery_1934 replacements applied",
            "n_perm": n_perm,
            "n_boot": n_boot,
            "k_grid": [int(v) for v in ks],
            "nulls": (
                "activity-decile-stratified label permutation (same construction as the banked "
                "tail_depth_sweep); pointwise 2.5/97.5 band plus a studentized max-T "
                "scan-corrected band over the k grid"
            ),
            "auroc": (
                "P(predictor_top > predictor_bottom) + 0.5 P(tie) for the top-k vs bottom-k R^2 "
                "groups, with a group-conditional bootstrap CI (both groups resampled with "
                "replacement)"
            ),
            "auroc_predictors_omitted": {
                "dec_norm": EXCLUSIONS_DOC["dec_norm"],
                "enc_dec_ratio": EXCLUSIONS_DOC["enc_dec_ratio"],
            },
            "seed": SEED,
        },
        "panel_reads": panel,
        "tail_depth_sweep": sweep,
        "auroc_sweep": auroc,
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _errbars(vals: np.ndarray, cis: np.ndarray) -> np.ndarray:
    """Non-negative (2, n) offsets for errorbar from point + [lo, hi] CIs."""
    lo = np.maximum(0.0, vals - cis[:, 0])
    hi = np.maximum(0.0, cis[:, 1] - vals)
    return np.vstack([lo, hi])


def figures(joint: dict, ext: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    stems: list[str] = []

    # (1) forest plot: raw vs all-others-partial rho, sorted by |partial|
    preds = joint["predictors"]
    names = [p["label"] + (" [retired axis]" if p.get("retired_axis") else "") for p in preds][::-1]
    raw = np.array([p["spearman_raw"] for p in preds])[::-1]
    raw_ci = np.array([p["spearman_raw_ci95"] for p in preds])[::-1]
    par = np.array([p["partial_all_others"] for p in preds])[::-1]
    par_ci = np.array([p["partial_all_others_ci95"] for p in preds])[::-1]
    ypos = np.arange(len(names), dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 0.30 * len(names) + 1.9))
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--", zorder=1)
    ax.errorbar(
        raw,
        ypos + 0.17,
        xerr=_errbars(raw, raw_ci),
        fmt="o",
        ms=4.2,
        color=paper_palette_role("baseline"),
        mfc="white",
        mew=1.2,
        lw=1.1,
        capsize=2.0,
        label="raw Spearman",
        zorder=3,
    )
    ax.errorbar(
        par,
        ypos - 0.17,
        xerr=_errbars(par, par_ci),
        fmt="o",
        ms=4.2,
        color=paper_palette_role("primary"),
        lw=1.1,
        capsize=2.0,
        label="all-others-partial Spearman",
        zorder=4,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel(r"Spearman $\rho$ vs per-feature held-out $R^2$")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "What survives partialling every other predictor",
        f"{joint['design']['n_features']:,} layer-19 SAE features; joint rank-OLS "
        f"$R^2$ = {joint['joint_r2']:.3f}; 95% CIs from "
        f"{joint['design']['n_boot']:,} feature bootstrap draws",
    )
    fig.tight_layout()
    savefig_paper(fig, "joint_partial_forest", dir=fig_dir)
    plt.close(fig)
    stems.append("joint_partial_forest")

    # (2) tail-depth extension + AUROC sweep
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    sweeps = ext["tail_depth_sweep"]
    show = ["interpretable", "content_syntax", "content_topic", "content_operation"]
    colors = paper_palette(len(show))
    ax0 = axes[0]
    # Each coding gets its OWN permutation band: band width scales with the
    # label's prevalence (syntax 77% vs task_format 1%), so one shared band
    # would misstate significance for every series but the one it was drawn for.
    for name, color in zip(show, colors, strict=True):
        s = sweeps[name]
        ax0.fill_between(
            s["k"],
            s["perm_band_pointwise_2p5"],
            s["perm_band_pointwise_97p5"],
            color=color,
            alpha=0.12,
            lw=0,
        )
        ax0.plot(s["k"], s["delta_obs"], "o-", ms=3.4, lw=1.3, color=color, label=name)
    ax0.axhline(0.0, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax0.set_xscale("log")
    ax0.set_xlabel("tail width $k$ (features per tail)")
    ax0.set_ylabel(r"$\Delta_k$ = frac(label | top-$k$) $-$ frac(label | bottom-$k$)")
    ax0.legend(
        loc="best",
        frameon=False,
        fontsize=7.4,
        title="4 of 6 codings (entity, task_format <2%: JSON only)",
        title_fontsize=7.0,
    )
    set_title_subtitle(
        ax0,
        "Tail-depth sweep: new judged axes",
        "positive = enriched among the best-predicted; shaded = each coding's own "
        "activity-stratified band",
    )

    ax1 = axes[1]
    acolors = paper_palette(len(ext["auroc_sweep"]))
    for (key, a), color in zip(ext["auroc_sweep"].items(), acolors, strict=True):
        cis = np.array(a["auroc_ci95"])
        ax1.plot(a["k"], a["auroc"], "o-", ms=3.4, lw=1.3, color=color, label=a["label"])
        ax1.fill_between(a["k"], cis[:, 0], cis[:, 1], color=color, alpha=0.15, lw=0)
    ax1.axhline(0.5, color=paper_palette_role("neutral"), lw=0.9, ls="--")
    ax1.set_xscale("log")
    ax1.set_ylim(0.35, 1.02)
    ax1.set_xlabel("tail width $k$ (features per tail)")
    ax1.set_ylabel("AUROC: top-$k$ vs bottom-$k$ by $R^2$")
    # opaque backing: the chance line sits at 0.5, right where the legend lands
    ax1.legend(
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    set_title_subtitle(
        ax1, "Continuous predictors as tail classifiers", "0.5 = chance; shaded = bootstrap 95% CI"
    )

    fig.tight_layout()
    savefig_paper(fig, "tail_auroc_extension", dir=fig_dir)
    plt.close(fig)
    stems.append("tail_auroc_extension")
    return stems


# ── entrypoint ────────────────────────────────────────────────────────────────


def _metadata() -> dict:
    import platform
    import subprocess
    from datetime import UTC, datetime

    try:
        sha = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        sha = "unavailable-no-git-checkout"
    return {
        "git_commit": sha or "unavailable-no-git-checkout",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #1482 joint per-feature predictor battery")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=TDS.N_PERM)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / FIG_DIR)
    ap.add_argument("--smoke", action="store_true", help="tiny draw counts; wiring only")
    args = ap.parse_args()

    if args.smoke:
        args.n_boot, args.n_perm = 50, 50
    # `TDS._perm_within_strata` sizes its permutation matrix from the module
    # constant; rebind it so the reused helper draws exactly `--n-perm` columns
    # (a no-op at the default, which is the banked sweep's 2,000).
    TDS.N_PERM = args.n_perm
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    t0 = time.time()

    bundle = assemble(args.out_dir)
    joint = joint_model(bundle, args.n_boot, rng)
    joint["metadata"] = _metadata()
    (args.out_dir / "joint_model.json").write_text(json.dumps(joint, indent=1))

    ext = tail_extension(bundle, args.n_perm, args.n_boot, rng)
    ext["metadata"] = _metadata()
    (args.out_dir / "tail_extension.json").write_text(json.dumps(ext, indent=1))

    stems = figures(joint, ext, args.fig_dir)
    _log(f"figures: {', '.join(stems)}")
    _log(f"done in {time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
