#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #541 follow-up ``geometry-plus-prior-joint-predictor`` — CPU analysis.

Builds the 69-row (arm × bystander) frame from the committed parent
``predictors.json`` (leak DV, priors, strata — all REUSED frozen) plus the
fresh geometry matrices from ``issue541_geometry_extract.py``, then runs the
pre-registered analyses (plan §3.2 item 2):

  A. collinearity gate (|Pearson(geometry, prior)| > 0.6 → tercile fallback);
  B. per-arm rank reads — raw + partial Spearman(geometry, leak | prior[,
     stratum]) with 10,000-perm two-sided Monte-Carlo p (add-one), Holm over
     the 6 confirmatory partials (2 metrics × 3 arms), parent cross-check;
  C. PRIMARY — pooled model comparison (M_arm / M_prior / M_geom / M_joint,
     z-scored OLS in rate space), 24-fold leave-one-persona-out CV, macro
     within-arm held-out Spearman (primary) + pooled held-out R² (secondary);
     increment Δ = joint − prior with (a) 2,000-iter cluster-on-persona
     bootstrap 95% CI and (b) BINDING p from the CLUSTER-LEVEL persona-identity
     permutation (1,000 perms × full LOPO; one permutation of the 24 panel
     identities per draw, applied consistently across arms, geometry
     reassigned from the full 24×24 matrices, donor collisions defined as the
     matrix diagonal — cos=1.0 / gkl=0.0; leak/prior/stratum/arm FE fixed;
     add-one, one-sided upper). The within-arm (cos, gkl)-pair permutation is
     computed ONLY as a labeled sensitivity, never the binding p;
  D. SECONDARY — high-stratum distance-to-teacher (2 high arms confirmatory ×
     2 metrics, H-stratum n=9, 10,000-perm two-sided p, Holm over 4,
     critical-|rho| context; marine arm shown for contrast);
  E. sensitivities (pre-registered): per-seed, drop-furniture_historian,
     marine-only LOPO, gkl k=8, per-layer cosine; drop-one increment curve.

All RNG seed 42. Writes ``joint_predictor_results.json`` + figures (hero:
``model_comparison.png``, ``high_stratum_proximity.png`` + the plan §5
exploratory dump, raw alongside residualized).

Smoke (``--smoke-synthetic-geometry``): runs the IDENTICAL code path against
the REAL committed predictors.json with a deterministic synthetic 24×24
geometry built from the committed phase0c cosines (L22 := L21 copy; gkl :=
monotone transform of cosine + seeded jitter), clearly labeled, outputs into
the ``issue_541_smoke`` namespace; reduce iterations via ``--n-perm-arm /
--n-perm-lopo / --n-boot``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i541_geometry_joint")

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue500_predictors as i500  # noqa: E402
from issue541_predictors import _critical_spearman, _tercile_buckets  # noqa: E402

SEED = 42
COS_LAYER = "21"  # parent headline (phase0c / rule-file legacy default)
COS_SENS_LAYERS = ("7", "14", "27")  # exploratory per-layer sensitivity
GKL_LAYER = "22"  # #502 bakeoff winner via #532
GKL_K = "k16"
GKL_K_SENS = "k8"
COLLINEARITY_GATE = 0.6  # parent COLLINEARITY_GATE_SOFT
CROSS_CHECK_MIN_SPEARMAN = 0.99
NEAREST_NEIGHBOR = "furniture_historian"  # the 91% nearest-neighbor persona (plan §3.2 D/E)
SUBDIR = "geometry-plus-prior-joint-predictor"
MODELS = ("arm_only", "prior_only", "geometry_only", "joint")

PREDICTORS_PATH = PROJECT_ROOT / "eval_results" / "issue_541" / "predictors.json"
PHASE0C_PATH = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_541"
    / "phase0_prescreen"
    / "phase0c_persona_vectors.json"
)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _git_commit_sha() -> str:
    import os
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ},
        check=False,
    )
    return out.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Vendored statistics (attributed)
# ---------------------------------------------------------------------------
def _permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> dict[str, float | int]:
    """Two-sided permutation p for Spearman rho (shuffle y across cells).

    VENDORED from ``scripts/issue539_residual_per_cohort.py::_permutation_p``
    (main branch; itself vendoring the #532 estimand): permute y, Spearman
    against fixed x, two-sided |rho| comparison, add-one p, vectorized via
    row-permuted rank matrix (valid under ties: average ranks travel with
    the values).
    """
    from scipy.stats import rankdata

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        return i500._spearman(list(map(float, a)), list(map(float, b)))

    rng = np.random.default_rng(seed)
    rho_obs = _rho(x, y)
    if np.isnan(rho_obs):
        return {"p": float("nan"), "rho_obs": rho_obs, "n_perm": n_perm}
    rx = rankdata(x).astype(float)
    rx -= rx.mean()
    ry = rankdata(y).astype(float)
    perms = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
    perms -= perms.mean(axis=1, keepdims=True)
    num = perms @ rx
    den = np.sqrt((rx @ rx) * (perms * perms).sum(axis=1))
    null_rhos = num / den
    count = int((np.abs(null_rhos) >= abs(rho_obs) - 1e-12).sum())
    return {
        "p": float((1 + count) / (n_perm + 1)),
        "rho_obs": float(rho_obs),
        "null_mean": float(np.mean(null_rhos)),
        "null_sd": float(np.std(null_rhos)),
        "n_perm": n_perm,
    }


def _perm_p_partial_multi(
    x: list[float], y: list[float], zs: list[list[float]], n_perm: int, seed: int
) -> dict[str, float | int]:
    """Two-sided Monte-Carlo permutation p (add-one) for the multi-covariate
    partial Spearman: shuffle the GEOMETRY vector x within the arm, recompute
    ``i500._partial_spearman_multi`` (plan §3.2 B inference)."""
    rho_obs = i500._partial_spearman_multi(x, y, zs)
    if np.isnan(rho_obs):
        return {"p": float("nan"), "rho_obs": rho_obs, "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    xa = np.asarray(x, dtype=float)
    count = 0
    for _ in range(n_perm):
        xp = xa[rng.permutation(len(xa))]
        r = i500._partial_spearman_multi([float(v) for v in xp], y, zs)
        if not np.isnan(r) and abs(r) >= abs(rho_obs) - 1e-12:
            count += 1
    return {"p": float((1 + count) / (n_perm + 1)), "rho_obs": float(rho_obs), "n_perm": n_perm}


def _holm(pmap: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (no NaN inputs allowed — asserted)."""
    assert all(not np.isnan(p) for p in pmap.values()), f"NaN p in Holm input: {pmap}"
    m = len(pmap)
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    adj: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[key] = float(min(1.0, running))
    return adj


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
class Frame:
    """The 69-row (arm × bystander) analysis frame + geometry lookups."""

    def __init__(self, pred: dict[str, Any], geom: dict[str, Any]):
        self.panel: list[str] = list(pred["panel"])
        self.sources: list[str] = list(pred["sources"])
        self.strata: dict[str, str] = dict(pred["strata"])
        assert len(self.panel) == 24 and len(self.sources) == 3

        g_personas: list[str] = list(geom["personas"])
        assert set(self.panel) <= set(g_personas), "geometry matrix missing panel personas"
        self.g_idx = {n: g_personas.index(n) for n in self.panel}
        self.cosM: dict[str, np.ndarray] = {
            layer: np.asarray(geom["cosine_matrix"][layer], dtype=float)
            for layer in geom["cosine_matrix"]
        }
        self.gklM: dict[tuple[str, str], np.ndarray] = {}
        for layer, per_k in geom["gauss_kl_matrix"].items():
            for kname, mat in per_k.items():
                self.gklM[(layer, kname)] = np.asarray(mat, dtype=float)
        # Donor-collision convention: diagonal = self-distance (cos 1.0 / gkl 0.0).
        for layer, m in self.cosM.items():
            assert np.allclose(np.diag(m), 1.0, atol=1e-3), f"cos diag != 1 at L{layer}"
        for key, m in self.gklM.items():
            assert np.allclose(np.diag(m), 0.0), f"gkl diag != 0 at {key}"

        rows: list[dict[str, Any]] = []
        for arm in self.sources:
            per_persona = pred["per_arm"][arm]["per_persona"]
            assert len(per_persona) == 23, (arm, len(per_persona))
            t = self.g_idx[arm]
            for b, rec in per_persona.items():
                assert len(rec["leak_seeds"]) == 3, (arm, b, rec["leak_seeds"])
                rows.append(
                    {
                        "arm": arm,
                        "bystander": b,
                        "leak": float(rec["leak_mean"]),
                        "leak_seeds": [float(v) for v in rec["leak_seeds"]],
                        "prior": float(rec["prior_logprob"]),
                        "stratum_high": 1.0 if self.strata[b] == "H" else 0.0,
                        "cos": float(self.cosM[COS_LAYER][self.g_idx[b], t]),
                        "gkl": float(self.gklM[(GKL_LAYER, GKL_K)][self.g_idx[b], t]),
                        "cos_parent": float(rec["cos_to_source"]),
                    }
                )
        assert len(rows) == 69, len(rows)
        self.rows = rows
        self.arm_names = np.array([r["arm"] for r in rows])
        self.bystanders = np.array([r["bystander"] for r in rows])
        self.leak = np.array([r["leak"] for r in rows])
        self.leak_seeds = np.array([r["leak_seeds"] for r in rows])  # (69, 3)
        self.prior = np.array([r["prior"] for r in rows])
        self.stratum = np.array([r["stratum_high"] for r in rows])
        self.cos = np.array([r["cos"] for r in rows])
        self.gkl = np.array([r["gkl"] for r in rows])
        self.arm_idx = np.array([self.sources.index(a) for a in self.arm_names])
        self.persona_idx = np.array([self.panel.index(b) for b in self.bystanders])
        self.teacher_idx = np.array([self.g_idx[a] for a in self.arm_names])
        self.b_g_idx = np.array([self.g_idx[b] for b in self.bystanders])

        for name, vec in (
            ("leak", self.leak),
            ("prior", self.prior),
            ("cos", self.cos),
            ("gkl", self.gkl),
        ):
            assert np.isfinite(vec).all(), f"NaN/inf in frame column {name}"

    def arm_rows(self, arm: str) -> np.ndarray:
        return np.where(self.arm_names == arm)[0]

    def geometry_cols(self, donor_g_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cos, gkl) headline columns for arbitrary donor panel indices —
        the cluster-level permutation's reassignment lookup."""
        cos = self.cosM[COS_LAYER][donor_g_idx, self.teacher_idx]
        gkl = self.gklM[(GKL_LAYER, GKL_K)][donor_g_idx, self.teacher_idx]
        return cos, gkl


def _cross_check_cos(frame: Frame) -> dict[str, Any]:
    """Hard assert: per-arm Spearman(our cos_L21 lookup, parent cos_to_source) >= 0.99."""
    out: dict[str, Any] = {}
    for arm in frame.sources:
        idx = frame.arm_rows(arm)
        ours = [frame.rows[i]["cos"] for i in idx]
        parents = [frame.rows[i]["cos_parent"] for i in idx]
        rho = i500._spearman(ours, parents)
        out[arm] = {"spearman": rho, "n": len(idx)}
        assert rho >= CROSS_CHECK_MIN_SPEARMAN, (
            f"cos_L21 lookup diverges from parent cos_to_source in arm {arm}: "
            f"Spearman {rho:.4f} < {CROSS_CHECK_MIN_SPEARMAN}"
        )
    return out


# ---------------------------------------------------------------------------
# Models + LOPO
# ---------------------------------------------------------------------------
def _zconst(v: np.ndarray) -> tuple[float, float]:
    sd = float(v.std(ddof=1)) or 1.0
    return float(v.mean()), sd


def _design(
    model: str,
    arm_idx: np.ndarray,
    n_arms: int,
    stratum: np.ndarray,
    prior: np.ndarray,
    cos: np.ndarray,
    gkl: np.ndarray,
    zc: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Design matrix per plan §3.2 C. Arm FE = one-hot (absorbs the intercept);
    continuous predictors z-scored with FIXED full-sample constants ``zc``
    (affine — held-out predictions are invariant to the scaling, so reusing
    the observed constants inside permutations/bootstraps changes nothing)."""
    fe = np.eye(n_arms)[arm_idx]
    cols = [fe]
    if model in ("prior_only", "joint"):
        mu, sd = zc["prior"]
        cols += [stratum[:, None], ((prior - mu) / sd)[:, None]]
    if model in ("geometry_only", "joint"):
        mu_c, sd_c = zc["cos"]
        mu_g, sd_g = zc["gkl"]
        cols += [((cos - mu_c) / sd_c)[:, None], ((gkl - mu_g) / sd_g)[:, None]]
    return np.hstack(cols)


def _lopo_yhat(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out OLS predictions (group = persona; #532 grouped-fold
    convention adapted from cells to personas, plan §11)."""
    yhat = np.empty_like(y, dtype=float)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        yhat[te] = X[te] @ coef
    return yhat


def _macro_spearman(
    yhat: np.ndarray, y: np.ndarray, arm_idx: np.ndarray, n_arms: int
) -> tuple[float, list[float]]:
    """Per-arm Spearman(yhat, y) macro-averaged over arms (NaN arms skipped)."""
    rhos: list[float] = []
    for a in range(n_arms):
        m = arm_idx == a
        rho = i500._spearman(list(yhat[m]), list(y[m])) if m.sum() >= 2 else float("nan")
        rhos.append(rho)
    valid = [r for r in rhos if not np.isnan(r)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, rhos


def _pooled_r2(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_all_models(
    y: np.ndarray,
    arm_idx: np.ndarray,
    n_arms: int,
    groups: np.ndarray,
    stratum: np.ndarray,
    prior: np.ndarray,
    cos: np.ndarray,
    gkl: np.ndarray,
    zc: dict[str, tuple[float, float]],
    models: tuple[str, ...] = MODELS,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        X = _design(m, arm_idx, n_arms, stratum, prior, cos, gkl, zc)
        yhat = _lopo_yhat(X, y, groups)
        macro, per_arm = _macro_spearman(yhat, y, arm_idx, n_arms)
        out[m] = {
            "macro_spearman": macro,
            "per_arm_spearman": per_arm,
            "pooled_r2": _pooled_r2(yhat, y),
        }
    return out


# ---------------------------------------------------------------------------
# Inference around the primary increment
# ---------------------------------------------------------------------------
def _cluster_bootstrap(frame: Frame, zc: dict, n_boot: int, seed: int) -> dict[str, Any]:
    """Cluster-on-persona bootstrap (resample the 24 personas with replacement;
    LOPO grouped by ORIGINAL persona id inside each resample). Returns the Δ
    CI plus per-model macro-metric percentiles for the hero-figure whiskers."""
    rng = np.random.default_rng(seed)
    persona_rows = {p: np.where(frame.bystanders == p)[0] for p in frame.panel}
    deltas: list[float] = []
    per_model: dict[str, list[float]] = {m: [] for m in MODELS}
    n_arms = len(frame.sources)
    n_degenerate = 0
    for _ in range(n_boot):
        sampled = rng.choice(len(frame.panel), size=len(frame.panel), replace=True)
        idx = np.concatenate([persona_rows[frame.panel[s]] for s in sampled])
        if idx.size == 0:
            n_degenerate += 1
            continue
        res = _fit_all_models(
            frame.leak[idx],
            frame.arm_idx[idx],
            n_arms,
            frame.persona_idx[idx],
            frame.stratum[idx],
            frame.prior[idx],
            frame.cos[idx],
            frame.gkl[idx],
            zc,
        )
        if any(np.isnan(res[m]["macro_spearman"]) for m in MODELS):
            n_degenerate += 1
            continue
        for m in MODELS:
            per_model[m].append(res[m]["macro_spearman"])
        deltas.append(res["joint"]["macro_spearman"] - res["prior_only"]["macro_spearman"])
    arr = np.asarray(deltas)
    return {
        "n_boot": n_boot,
        "n_valid": len(deltas),
        "n_degenerate": n_degenerate,
        "delta_mean": float(arr.mean()),
        "delta_ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "delta_ci_90": [float(np.percentile(arr, 5)), float(np.percentile(arr, 95))],
        "per_model_macro_ci_95": {
            m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in per_model.items()
        },
    }


def _cluster_identity_perm(
    frame: Frame, zc: dict, macro_prior_obs: float, delta_obs: float, n_perm: int, seed: int
) -> dict[str, Any]:
    """BINDING increment p — cluster-level persona-identity permutation.

    Each draw: ONE permutation π of the 24 panel persona identities, applied
    consistently across all 3 arms; each row's (cos, gkl) is reassigned from
    the full 24×24 matrices as (π(b), teacher). Donor collisions (π(b) = the
    arm's own teacher) hit the matrix diagonal — cos=1.0, gkl=0.0 — and are
    DEFINED, not dropped. leak / prior / stratum / arm FE are held fixed, so
    M_prior's LOPO metric is constant; only M_joint is re-run per draw.
    Add-one, one-sided upper (the hypothesis is directional: joint BEATS
    prior-only). Persona is the exchangeable unit, matching the LOPO folds
    and the cluster bootstrap (round-1 statistics-lens reconciliation)."""
    rng = np.random.default_rng(seed)
    n_arms = len(frame.sources)
    panel_g_idx = np.array([frame.g_idx[p] for p in frame.panel])
    b_panel_pos = np.array([frame.panel.index(b) for b in frame.bystanders])
    count = 0
    null_deltas: list[float] = []
    for _ in range(n_perm):
        pi = rng.permutation(len(frame.panel))
        donor_g = panel_g_idx[pi[b_panel_pos]]
        cos_p, gkl_p = frame.geometry_cols(donor_g)
        X = _design("joint", frame.arm_idx, n_arms, frame.stratum, frame.prior, cos_p, gkl_p, zc)
        yhat = _lopo_yhat(X, frame.leak, frame.persona_idx)
        macro, _ = _macro_spearman(yhat, frame.leak, frame.arm_idx, n_arms)
        d = macro - macro_prior_obs
        null_deltas.append(d)
        if d >= delta_obs - 1e-12:
            count += 1
    return {
        "p_one_sided_upper": float((1 + count) / (n_perm + 1)),
        "n_perm": n_perm,
        "null_mean": float(np.mean(null_deltas)),
        "null_sd": float(np.std(null_deltas)),
        "binding": True,
        "scheme": (
            "persona-identity permutation, consistent across arms, "
            "donor collisions = matrix diagonal"
        ),
    }


def _within_arm_pair_perm(
    frame: Frame, zc: dict, macro_prior_obs: float, delta_obs: float, n_perm: int, seed: int
) -> dict[str, Any]:
    """NON-binding sensitivity: independently permute the (cos, gkl) PAIRS
    across bystanders within each arm (anti-conservative — breaks persona-level
    cross-arm geometry coherence; labeled, never the binding p)."""
    rng = np.random.default_rng(seed)
    n_arms = len(frame.sources)
    arm_indices = [frame.arm_rows(a) for a in frame.sources]
    count = 0
    for _ in range(n_perm):
        cos_p = frame.cos.copy()
        gkl_p = frame.gkl.copy()
        for idx in arm_indices:
            perm = idx[rng.permutation(len(idx))]
            cos_p[idx] = frame.cos[perm]
            gkl_p[idx] = frame.gkl[perm]
        X = _design("joint", frame.arm_idx, n_arms, frame.stratum, frame.prior, cos_p, gkl_p, zc)
        yhat = _lopo_yhat(X, frame.leak, frame.persona_idx)
        macro, _ = _macro_spearman(yhat, frame.leak, frame.arm_idx, n_arms)
        if macro - macro_prior_obs >= delta_obs - 1e-12:
            count += 1
    return {
        "p_one_sided_upper": float((1 + count) / (n_perm + 1)),
        "n_perm": n_perm,
        "binding": False,
        "label": "SENSITIVITY ONLY — within-arm pair permutation (anti-conservative)",
    }


# ---------------------------------------------------------------------------
# Subset re-runs (sensitivities)
# ---------------------------------------------------------------------------
def _primary_on_subset(
    frame: Frame,
    zc: dict,
    idx: np.ndarray,
    *,
    y: np.ndarray | None = None,
    cos: np.ndarray | None = None,
    gkl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Point-estimate primary (4 model LOPO metrics + Δ) on a row subset and/or
    swapped DV / geometry columns."""
    y = frame.leak if y is None else y
    cos = frame.cos if cos is None else cos
    gkl = frame.gkl if gkl is None else gkl
    arms_present = np.unique(frame.arm_idx[idx])
    res = _fit_all_models(
        y[idx],
        # Re-index arms to the present subset so the FE one-hot has no empty col.
        np.searchsorted(arms_present, frame.arm_idx[idx]),
        len(arms_present),
        frame.persona_idx[idx],
        frame.stratum[idx],
        frame.prior[idx],
        cos[idx],
        gkl[idx],
        zc,
    )
    return {
        "models": res,
        "delta_macro_spearman": res["joint"]["macro_spearman"]
        - res["prior_only"]["macro_spearman"],
        "delta_pooled_r2": res["joint"]["pooled_r2"] - res["prior_only"]["pooled_r2"],
        "n_rows": int(idx.size),
    }


# ---------------------------------------------------------------------------
# Analysis sections A / B / D / E (split out of main for readability)
# ---------------------------------------------------------------------------
def _collinearity_gate(frame: Frame) -> dict[str, Any]:
    """Plan §3.2 A: per arm per metric, |Pearson(geometry, prior)| > 0.6 →
    partials reported WITH the tercile-bucket fallback."""
    out: dict[str, Any] = {"threshold": COLLINEARITY_GATE, "per_arm": {}}
    for arm in frame.sources:
        idx = frame.arm_rows(arm)
        rec: dict[str, Any] = {}
        for metric, vec in (("cos", frame.cos), ("gkl", frame.gkl)):
            r = i500._pearson(list(vec[idx]), list(frame.prior[idx]))
            tripped = bool(abs(r) > COLLINEARITY_GATE)
            entry: dict[str, Any] = {"pearson_vs_prior": r, "tripped": tripped}
            if tripped:
                entry["tercile_fallback"] = _tercile_buckets(
                    [str(b) for b in frame.bystanders[idx]],
                    [float(v) for v in vec[idx]],
                    [float(v) for v in frame.leak[idx]],
                )
            rec[metric] = entry
        out["per_arm"][arm] = rec
    return out


def _per_arm_reads(frame: Frame, pred: dict[str, Any], n_perm: int) -> dict[str, Any]:
    """Plan §3.2 B: per-arm raw + partial Spearman with within-arm Monte-Carlo
    permutation p (two-sided, add-one), Holm over the 6 confirmatory partials
    (2 metrics × 3 arms), parent cross-check."""
    per_arm: dict[str, Any] = {}
    confirmatory_p: dict[str, float] = {}
    parent_partials = {
        arm: pred["per_arm"][arm]["stats"]["partial_spearman_cos_to_source_given_prior"]
        for arm in frame.sources
    }
    for arm in frame.sources:
        idx = frame.arm_rows(arm)
        y = [float(v) for v in frame.leak[idx]]
        prior = [float(v) for v in frame.prior[idx]]
        strat = [float(v) for v in frame.stratum[idx]]
        rec: dict[str, Any] = {}
        for m_i, (metric, vec) in enumerate((("cos", frame.cos), ("gkl", frame.gkl))):
            x = [float(v) for v in vec[idx]]
            perm = _perm_p_partial_multi(
                x, y, [prior, strat], n_perm, SEED + 100 * m_i + frame.sources.index(arm)
            )
            rec[metric] = {
                "spearman_raw": i500._spearman(x, y),
                "partial_given_prior": i500._partial_spearman(x, y, prior),
                "partial_given_prior_stratum": perm["rho_obs"],
                "perm_p_two_sided": perm["p"],
                "n": len(x),
            }
            confirmatory_p[f"{arm}/{metric}"] = float(perm["p"])
        rec["parent_cross_check_partial_cos_given_prior"] = {
            "parent_committed": parent_partials[arm],
            "recomputed": rec["cos"]["partial_given_prior"],
            "abs_delta": abs(parent_partials[arm] - rec["cos"]["partial_given_prior"]),
        }
        per_arm[arm] = rec
    for key, adj in _holm(confirmatory_p).items():
        arm, metric = key.split("/")
        per_arm[arm][metric]["perm_p_holm"] = adj
    return per_arm


def _high_stratum_reads(frame: Frame, n_perm: int) -> dict[str, Any]:
    """Plan §3.2 D: high-stratum distance-to-teacher ordering — 2 high arms ×
    2 metrics confirmatory (Holm over 4), marine arm shown for contrast."""
    high_arms = [s for s in frame.sources if frame.strata[s] == "H"]
    contrast = [s for s in frame.sources if frame.strata[s] != "H"]
    assert len(high_arms) == 2 and len(contrast) == 1, (high_arms, contrast)
    contrast_arm = contrast[0]
    out: dict[str, Any] = {
        "confirmatory_arms": high_arms,
        "contrast_arm": contrast_arm,
        "per_arm": {},
    }
    hs_p: dict[str, float] = {}
    for arm in [*high_arms, contrast_arm]:
        idx = np.array([i for i in frame.arm_rows(arm) if frame.strata[frame.bystanders[i]] == "H"])
        rec: dict[str, Any] = {
            "n": int(idx.size),
            "critical_abs_rho_0.05": _critical_spearman(int(idx.size)),
            "personas": sorted(str(b) for b in frame.bystanders[idx]),
        }
        for metric, vec in (("cos", frame.cos), ("gkl", frame.gkl)):
            res = _permutation_p(vec[idx], frame.leak[idx], n_perm, SEED)
            rec[metric] = {
                "spearman": res["rho_obs"],
                "perm_p_two_sided": res["p"],
                "confirmatory": arm in high_arms,
            }
            if arm in high_arms:
                hs_p[f"{arm}/{metric}"] = float(res["p"])
        out["per_arm"][arm] = rec
    for key, adj in _holm(hs_p).items():
        arm, metric = key.split("/")
        out["per_arm"][arm][metric]["perm_p_holm"] = adj
    expected_n = {arm: 9 for arm in high_arms} | {contrast_arm: 10}
    for arm, n_exp in expected_n.items():
        assert out["per_arm"][arm]["n"] == n_exp, (arm, out["per_arm"][arm]["n"], n_exp)
    return out


def _drop_nn_with_perm(frame: Frame, zc: dict, n_perm: int) -> dict[str, Any]:
    """Drop-furniture_historian re-run of the primary, with its own
    cluster-identity permutation p over the remaining 23 personas."""
    keep = np.where(frame.bystanders != NEAREST_NEIGHBOR)[0]
    drop_fh = _primary_on_subset(frame, zc, keep)
    sub_prior = drop_fh["models"]["prior_only"]["macro_spearman"]
    rng = np.random.default_rng(SEED)
    panel_keep = [p for p in frame.panel if p != NEAREST_NEIGHBOR]
    panel_g_idx = np.array([frame.g_idx[p] for p in panel_keep])
    b_pos = np.array([panel_keep.index(b) for b in frame.bystanders[keep]])
    arms_present = np.unique(frame.arm_idx[keep])
    n_arms = len(frame.sources)
    count = 0
    for _ in range(n_perm):
        pi = rng.permutation(len(panel_keep))
        donor_g = panel_g_idx[pi[b_pos]]
        cos_p = frame.cosM[COS_LAYER][donor_g, frame.teacher_idx[keep]]
        gkl_p = frame.gklM[(GKL_LAYER, GKL_K)][donor_g, frame.teacher_idx[keep]]
        X = _design(
            "joint",
            np.searchsorted(arms_present, frame.arm_idx[keep]),
            len(arms_present),
            frame.stratum[keep],
            frame.prior[keep],
            cos_p,
            gkl_p,
            zc,
        )
        yhat = _lopo_yhat(X, frame.leak[keep], frame.persona_idx[keep])
        macro, _ = _macro_spearman(yhat, frame.leak[keep], frame.arm_idx[keep], n_arms)
        if macro - sub_prior >= drop_fh["delta_macro_spearman"] - 1e-12:
            count += 1
    drop_fh["perm_p_cluster_identity"] = float((1 + count) / (n_perm + 1))
    return drop_fh


def _sensitivities(
    frame: Frame, zc: dict, all_idx: np.ndarray, n_perm_lopo: int
) -> tuple[dict[str, Any], dict[str, float]]:
    """Plan §3.2 E pre-registered sensitivities + the drop-one increment curve."""
    sensitivity: dict[str, Any] = {}
    sensitivity["per_seed"] = {
        f"seed_{s}": _primary_on_subset(frame, zc, all_idx, y=frame.leak_seeds[:, s_i])
        for s_i, s in enumerate((42, 137, 256))
    }
    sensitivity["drop_furniture_historian"] = _drop_nn_with_perm(frame, zc, n_perm_lopo)

    marine = next(s for s in frame.sources if frame.strata[s] != "H")
    sensitivity["marine_only_lopo"] = _primary_on_subset(frame, zc, frame.arm_rows(marine))

    gkl8 = frame.gklM[(GKL_LAYER, GKL_K_SENS)][frame.b_g_idx, frame.teacher_idx]
    if np.isfinite(gkl8).all():
        zc8 = dict(zc)
        zc8["gkl"] = _zconst(gkl8)
        sensitivity["gkl_k8"] = _primary_on_subset(frame, zc8, all_idx, gkl=gkl8)
    else:
        sensitivity["gkl_k8"] = {
            "status": "skipped_nan",
            "n_nan": int((~np.isfinite(gkl8)).sum()),
        }

    per_layer: dict[str, Any] = {}
    for layer in COS_SENS_LAYERS:
        cos_l = frame.cosM[layer][frame.b_g_idx, frame.teacher_idx]
        zcl = dict(zc)
        zcl["cos"] = _zconst(cos_l)
        per_layer[f"L{layer}"] = _primary_on_subset(frame, zcl, all_idx, cos=cos_l)
    sensitivity["per_layer_cosine"] = per_layer

    drop_one: dict[str, float] = {}
    for p_name in frame.panel:
        keep_p = np.where(frame.bystanders != p_name)[0]
        drop_one[p_name] = _primary_on_subset(frame, zc, keep_p)["delta_macro_spearman"]
    return sensitivity, drop_one


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _make_figures(
    frame: Frame,
    results: dict[str, Any],
    geom: dict[str, Any],
    fig_dir: Path,
    zc: dict,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    written: list[str] = []
    pal = paper_palette(6)
    model_labels = {
        "arm_only": "Teacher arm only",
        "prior_only": "Familiarity (prior + stratum)",
        "geometry_only": "Geometry (cosine + Gaussian-KL)",
        "joint": "Familiarity + geometry",
    }
    arm_labels = {
        "marine_biologist": "marine biologist arm",
        "courthouse_architecture_historian": "courthouse historian arm",
        "wooden_furniture_carpenter": "furniture carpenter arm",
    }

    def _save(fig, stem: str) -> None:
        paths = savefig_paper(fig, stem, dir=fig_dir, formats=("png",))
        written.append(str(paths["png"]))
        plt.close(fig)

    lopo = results["model_comparison"]["lopo"]

    # ── HERO 1: model comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(len(MODELS))
    for i, m in enumerate(MODELS):
        v = lopo["models"][m]["macro_spearman"]
        ci = results["model_comparison"]["bootstrap"]["per_model_macro_ci_95"][m]
        ax.bar(i, v, width=0.62, color=pal[i], zorder=2)
        # Percentile CI drawn as a segment (it need not bracket the observed bar).
        ax.vlines(i, ci[0], ci[1], color="0.25", lw=1.4, zorder=3)
        ax.scatter([i, i], ci, marker="_", s=80, color="0.25", zorder=3)
        for a_i, rho in enumerate(lopo["models"][m]["per_arm_spearman"]):
            ax.scatter(i + (a_i - 1) * 0.13, rho, s=22, color="0.15", alpha=0.75, zorder=4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs, [model_labels[m] for m in MODELS], rotation=12, ha="right")
    ax.set_ylabel("Held-out Spearman (macro over arms)")
    ax.set_title("Predicting who leaks: familiarity vs base-model geometry (LOPO held-out)")
    _save(fig, "model_comparison")

    # ── HERO 2: high-stratum proximity ──────────────────────────────────────
    high_arms = results["high_stratum"]["confirmatory_arms"]
    contrast_arm = results["high_stratum"]["contrast_arm"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, arm in zip(axes, [*high_arms, contrast_arm], strict=True):
        idx = [i for i in frame.arm_rows(arm) if frame.strata[frame.bystanders[i]] == "H"]
        for i in idx:
            b = frame.bystanders[i]
            seeds = frame.leak_seeds[i]
            color = pal[3] if b == NEAREST_NEIGHBOR else pal[0]
            ax.plot([frame.cos[i]] * 2, [seeds.min(), seeds.max()], color=color, lw=1.0, alpha=0.6)
            ax.scatter(frame.cos[i], frame.leak[i], s=42, color=color, zorder=3)
        suffix = " (contrast — low-prior teacher)" if arm == contrast_arm else ""
        ax.set_title(arm_labels[arm] + suffix)
        ax.set_xlabel("cosine to teacher (L21)")
    axes[0].set_ylabel("fact-leak rate (high-familiarity personas)")
    fig.suptitle(
        f"Does proximity to the teacher order leakage among already-familiar personas? "
        f"({NEAREST_NEIGHBOR.replace('_', ' ')} highlighted)",
        y=1.04,
    )
    _save(fig, "high_stratum_proximity")

    # ── Exploratory: raw + residualized scatters per arm × metric ───────────
    for metric, vec, xlabel in (
        ("cos", frame.cos, "cosine to teacher (L21)"),
        ("gkl", frame.gkl, "Gaussian sym-KL to teacher (L22, k=16)"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
        for col, arm in enumerate(frame.sources):
            idx = frame.arm_rows(arm)
            axes[0, col].scatter(vec[idx], frame.leak[idx], s=30, color=pal[0])
            axes[0, col].set_title(f"{arm_labels[arm]} — raw")
            axes[0, col].set_xlabel(xlabel)
            # Residualize BOTH axes on [1, prior, stratum] (rate space).
            A = np.column_stack([np.ones(idx.size), frame.prior[idx], frame.stratum[idx]])
            ry = frame.leak[idx] - A @ np.linalg.lstsq(A, frame.leak[idx], rcond=None)[0]
            rx = vec[idx] - A @ np.linalg.lstsq(A, vec[idx], rcond=None)[0]
            axes[1, col].scatter(rx, ry, s=30, color=pal[1])
            axes[1, col].set_title(f"{arm_labels[arm]} — familiarity-residualized")
            axes[1, col].set_xlabel(f"{xlabel}, residualized")
        axes[0, 0].set_ylabel("fact-leak rate")
        axes[1, 0].set_ylabel("leak residual")
        _save(fig, f"scatter_{metric}_raw_and_residualized")

    # ── Exploratory: gkl heatmap, strata-ordered ────────────────────────────
    order = sorted(
        range(len(frame.panel)),
        key=lambda i: ({"H": 0, "M": 1, "L": 2}[frame.strata[frame.panel[i]]], frame.panel[i]),
    )
    names_ord = [frame.panel[i] for i in order]
    g_idx_ord = [frame.g_idx[n] for n in names_ord]
    mat = frame.gklM[(GKL_LAYER, GKL_K)][np.ix_(g_idx_ord, g_idx_ord)]
    fig, ax = plt.subplots(figsize=(9.5, 8.5), layout="constrained")
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(names_ord)), names_ord, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names_ord)), names_ord, fontsize=6)
    fig.colorbar(im, ax=ax, label="Gaussian sym-KL (L22, k=16)")
    ax.set_title("Pairwise Gaussian sym-KL, personas ordered by familiarity stratum (H→M→L)")
    _save(fig, "gkl_heatmap_L22")

    # ── Exploratory: cos vs gkl redundancy ──────────────────────────────────
    iu = np.triu_indices(len(frame.panel), k=1)
    g_all = np.array([frame.g_idx[n] for n in frame.panel])
    cos_pairs = frame.cosM[COS_LAYER][np.ix_(g_all, g_all)][iu]
    gkl_pairs = frame.gklM[(GKL_LAYER, GKL_K)][np.ix_(g_all, g_all)][iu]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(cos_pairs, gkl_pairs, s=14, alpha=0.6, color=pal[0])
    ax.set_xlabel("pairwise cosine (L21)")
    ax.set_ylabel("pairwise Gaussian sym-KL (L22, k=16)")
    ax.set_title("Are the two geometry metrics redundant?")
    _save(fig, "cos_vs_gkl")

    # ── Exploratory: per-layer cosine partial profile ───────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    layers = [COS_LAYER, *COS_SENS_LAYERS]
    layers_sorted = sorted(layers, key=int)
    for a_i, arm in enumerate(frame.sources):
        idx = frame.arm_rows(arm)
        vals = []
        for layer in layers_sorted:
            v = frame.cosM[layer][frame.b_g_idx[idx], frame.teacher_idx[idx]]
            vals.append(
                i500._partial_spearman_multi(
                    list(v),
                    list(frame.leak[idx]),
                    [list(frame.prior[idx]), list(frame.stratum[idx])],
                )
            )
        ax.plot(
            [int(x) for x in layers_sorted], vals, marker="o", label=arm_labels[arm], color=pal[a_i]
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("partial Spearman(cos, leak | prior, stratum)")
    ax.set_title("Per-layer cosine partials (exploratory)")
    ax.legend()
    _save(fig, "per_layer_cosine_partial")

    # ── Exploratory: per-seed increment + drop-one curve ────────────────────
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    per_seed = results["sensitivity"]["per_seed"]
    ds = [per_seed[k]["delta_macro_spearman"] for k in sorted(per_seed)]
    ax.scatter(range(len(ds)), ds, s=50, color=pal[0], zorder=3)
    ax.axhline(
        results["model_comparison"]["lopo"]["increment"]["delta_macro_spearman"],
        color=pal[3],
        lw=1.2,
        label="pooled (3-seed mean DV)",
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(ds)), [f"seed {s}" for s in (42, 137, 256)])
    ax.set_ylabel("Δ held-out macro Spearman (joint − familiarity)")
    ax.set_title("Per-seed increment")
    ax.legend()
    _save(fig, "per_seed_increment")

    drop_one = results["exploratory"]["drop_one_increment"]
    names = sorted(drop_one, key=lambda k: drop_one[k])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(names)), [drop_one[n] for n in names], color=pal[0])
    ax.axhline(
        results["model_comparison"]["lopo"]["increment"]["delta_macro_spearman"],
        color=pal[3],
        lw=1.2,
        label="full panel",
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=90, fontsize=6)
    ax.set_ylabel("Δ macro Spearman without persona")
    ax.set_title("Drop-one-persona increment curve")
    ax.legend()
    _save(fig, "drop_one_increment")

    # ── Exploratory: fidelity scatter (recomputed vs committed cosine) ──────
    committed = json.loads(PHASE0C_PATH.read_text())
    c_idx = {n: i for i, n in enumerate(committed["personas"])}
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for li, color in zip(("7", "14", "21", "27"), paper_palette(4), strict=True):
        if li not in frame.cosM:
            continue
        c_mat = committed["cosine_matrix"][li]
        xs, ys = [], []
        for a in frame.panel:
            for b in frame.panel:
                if a >= b:
                    continue
                xs.append(c_mat[c_idx[a]][c_idx[b]])
                ys.append(float(frame.cosM[li][frame.g_idx[a], frame.g_idx[b]]))
        ax.scatter(xs, ys, s=8, alpha=0.5, color=color, label=f"layer {li}")
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="0.4", lw=0.8)
    ax.set_xlabel("committed phase0c cosine")
    ax.set_ylabel("recomputed cosine")
    ax.set_title("Extraction fidelity: recomputed vs committed pairwise cosine")
    ax.legend()
    _save(fig, "fidelity_scatter")

    # ── Exploratory: k=8 vs k=16 gkl rank agreement ─────────────────────────
    gkl8_pairs = frame.gklM[(GKL_LAYER, GKL_K_SENS)][np.ix_(g_all, g_all)][iu]
    finite = np.isfinite(gkl_pairs) & np.isfinite(gkl8_pairs)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(gkl_pairs[finite], gkl8_pairs[finite], s=14, alpha=0.6, color=pal[0])
    rho_k = i500._spearman(list(gkl_pairs[finite]), list(gkl8_pairs[finite]))
    ax.set_xlabel("Gaussian sym-KL, k=16")
    ax.set_ylabel("Gaussian sym-KL, k=8")
    ax.set_title(f"k=8 vs k=16 pairwise agreement (Spearman {rho_k:.3f})")
    _save(fig, "k8_vs_k16")
    results["exploratory"]["gkl_k8_vs_k16_pair_spearman"] = rho_k

    return written


# ---------------------------------------------------------------------------
# Synthetic smoke geometry
# ---------------------------------------------------------------------------
def _synthetic_geometry(panel: list[str]) -> dict[str, Any]:
    """Deterministic SYNTHETIC smoke geometry, clearly labeled: cosine =
    committed phase0c sub-matrix (L22 := L21 copy); gkl := monotone transform
    of cosine + small seeded symmetric jitter, diag 0. Exercises the full
    analysis code path (incl. the cos cross-check, which passes by
    construction) without GPU extraction."""
    committed = json.loads(PHASE0C_PATH.read_text())
    c_idx = {n: i for i, n in enumerate(committed["personas"])}
    sub = {
        layer: [
            [float(committed["cosine_matrix"][layer][c_idx[a]][c_idx[b]]) for b in panel]
            for a in panel
        ]
        for layer in ("7", "14", "21", "27")
    }
    sub["22"] = [row[:] for row in sub["21"]]
    rng = np.random.default_rng(SEED)
    gkl: dict[str, dict[str, list[list[float]]]] = {}
    for layer, mat in sub.items():
        arr = np.asarray(mat)
        per_k: dict[str, list[list[float]]] = {}
        for kname, scale in (("k16", 60.0), ("k8", 45.0)):
            jit = rng.normal(0, 0.4, size=arr.shape)
            jit = 0.5 * (jit + jit.T)
            g = scale * (1.0 - arr) + jit
            np.fill_diagonal(g, 0.0)
            g = np.clip(g, 0.0, None)
            np.fill_diagonal(g, 0.0)
            per_k[kname] = [[float(x) for x in row] for row in 0.5 * (g + g.T)]
        gkl[layer] = per_k
    return {
        "_doc": (
            "SYNTHETIC SMOKE INPUT — deterministic stand-in geometry (seed 42); NOT a measurement."
        ),
        "smoke_synthetic": True,
        "personas": list(panel),
        "layers": [7, 14, 21, 22, 27],
        "cosine_matrix": sub,
        "gauss_kl_matrix": gkl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--geometry",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_541" / SUBDIR / "geometry_matrices.json",
        help="geometry_matrices.json from issue541_geometry_extract.py",
    )
    ap.add_argument(
        "--smoke-synthetic-geometry",
        action="store_true",
        help="run the identical code path on a deterministic synthetic 24x24 geometry "
        "(smoke namespace outputs; real committed predictors.json)",
    )
    ap.add_argument("--n-perm-arm", type=int, default=10_000)
    ap.add_argument("--n-perm-lopo", type=int, default=1_000)
    ap.add_argument("--n-boot", type=int, default=2_000)
    args = ap.parse_args()

    t_start = time.time()
    smoke = args.smoke_synthetic_geometry
    eval_root = "issue_541_smoke" if smoke else "issue_541"
    out_dir = PROJECT_ROOT / "eval_results" / eval_root / SUBDIR
    fig_dir = PROJECT_ROOT / "figures" / eval_root / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = json.loads(PREDICTORS_PATH.read_text())
    if smoke:
        geom = _synthetic_geometry(list(pred["panel"]))
        logger.info("SMOKE: synthetic geometry (deterministic, seed %d)", SEED)
    else:
        geom = json.loads(args.geometry.read_text())
        assert not geom.get("smoke"), (
            f"{args.geometry} is a smoke extraction artifact — the full analysis "
            "requires the full-run geometry (24 personas x 40 probes)."
        )

    frame = Frame(pred, geom)
    cross_check = _cross_check_cos(frame)
    logger.info(
        "frame OK: 69 rows, cos cross-check %s",
        {a: round(v["spearman"], 4) for a, v in cross_check.items()},
    )

    zc = {"prior": _zconst(frame.prior), "cos": _zconst(frame.cos), "gkl": _zconst(frame.gkl)}
    n_arms = len(frame.sources)
    all_idx = np.arange(len(frame.rows))

    # ── A. collinearity gate ────────────────────────────────────────────────
    collinearity = _collinearity_gate(frame)

    # ── B. per-arm rank reads ───────────────────────────────────────────────
    logger.info("per-arm partials + %d-perm inference ...", args.n_perm_arm)
    per_arm = _per_arm_reads(frame, pred, args.n_perm_arm)

    # ── C. PRIMARY model comparison ─────────────────────────────────────────
    logger.info("LOPO model comparison ...")
    lopo_models = _fit_all_models(
        frame.leak,
        frame.arm_idx,
        n_arms,
        frame.persona_idx,
        frame.stratum,
        frame.prior,
        frame.cos,
        frame.gkl,
        zc,
    )
    macro_prior = lopo_models["prior_only"]["macro_spearman"]
    macro_joint = lopo_models["joint"]["macro_spearman"]
    delta_obs = macro_joint - macro_prior
    delta_r2 = lopo_models["joint"]["pooled_r2"] - lopo_models["prior_only"]["pooled_r2"]
    geom_minus_prior = lopo_models["geometry_only"]["macro_spearman"] - macro_prior

    logger.info("cluster bootstrap (%d) ...", args.n_boot)
    boot = _cluster_bootstrap(frame, zc, args.n_boot, SEED)
    logger.info("cluster-identity permutation (%d x LOPO, BINDING) ...", args.n_perm_lopo)
    perm_binding = _cluster_identity_perm(frame, zc, macro_prior, delta_obs, args.n_perm_lopo, SEED)
    logger.info("within-arm pair permutation (sensitivity) ...")
    perm_sens = _within_arm_pair_perm(frame, zc, macro_prior, delta_obs, args.n_perm_lopo, SEED)

    model_comparison = {
        "lopo": {
            "models": lopo_models,
            "primary_metric": "macro within-arm held-out Spearman",
            "secondary_metric": "pooled held-out R^2",
            "increment": {
                "delta_macro_spearman": delta_obs,
                "delta_pooled_r2": delta_r2,
                "bootstrap_ci_95": boot["delta_ci_95"],
                "perm_p_cluster_identity_BINDING": perm_binding,
                "perm_p_within_arm_pairs_SENSITIVITY": perm_sens,
            },
            "geometry_only_minus_prior_only_macro_spearman": geom_minus_prior,
            "fold_scheme": "leave-one-persona-out, 24 persona-grouped folds",
        },
        "bootstrap": boot,
    }

    # ── D. SECONDARY high-stratum distance-to-teacher ───────────────────────
    high_stratum = _high_stratum_reads(frame, args.n_perm_arm)

    # ── E. sensitivities ────────────────────────────────────────────────────
    logger.info("sensitivities ...")
    sensitivity, drop_one = _sensitivities(frame, zc, all_idx, args.n_perm_lopo)

    # ── write ───────────────────────────────────────────────────────────────
    results: dict[str, Any] = {
        "_doc": (
            "Issue #541 follow-up geometry-plus-prior-joint-predictor. PRIMARY = "
            "model_comparison.lopo.increment (joint minus familiarity-only, LOPO "
            "held-out macro within-arm Spearman; CI = cluster-on-persona bootstrap; "
            "BINDING p = cluster-level persona-identity permutation, one-sided "
            "upper). gkl is a DISTANCE (higher = farther); cos a similarity. "
            "Per-arm reads are two-sided (parent signs unstable). NOTE on the "
            "arm_only floor model: under LOPO an (effectively intercept-only) "
            "within-arm prediction equals the train-fold mean, which is "
            "mechanically ANTI-correlated with the held-out value, so its "
            "macro Spearman sits near -1 — read it as 'no ranking signal', "
            "not as a catastrophic model."
        ),
        "params": {
            "seed": SEED,
            "cos_layer": COS_LAYER,
            "gkl_layer": GKL_LAYER,
            "gkl_k": GKL_K,
            "n_perm_arm": args.n_perm_arm,
            "n_perm_lopo": args.n_perm_lopo,
            "n_boot": args.n_boot,
            "dv": "leak_mean over 3 trained seeds (baseline cell excluded, parent estimator)",
            "smoke_synthetic_geometry": smoke,
        },
        "frame": {
            "n_rows": len(frame.rows),
            "n_personas": len(frame.panel),
            "arms": frame.sources,
            "rows": frame.rows,
        },
        "gates": {
            "collinearity": collinearity,
            "fidelity": geom.get("fidelity_check", {"status": "synthetic_smoke_input"}),
            "cross_check_cos_vs_parent": cross_check,
        },
        "per_arm": {
            "partials": per_arm,
            "holm_family": "6 confirmatory partials (2 metrics x 3 arms)",
        },
        "model_comparison": model_comparison,
        "high_stratum": high_stratum,
        "sensitivity": sensitivity,
        "exploratory": {"drop_one_increment": drop_one},
        "reproducibility": {
            "git_commit": _git_commit_sha(),
            "predictors_json": str(PREDICTORS_PATH.relative_to(PROJECT_ROOT)),
            "geometry_json": "SYNTHETIC (smoke)" if smoke else str(args.geometry),
            "numpy": np.__version__,
            "wall_seconds": None,  # filled below
            "timestamp": _now_iso(),
        },
    }

    logger.info("figures ...")
    results["figures"] = _make_figures(frame, results, geom, fig_dir, zc)
    results["reproducibility"]["wall_seconds"] = round(time.time() - t_start, 1)

    out_path = out_dir / "joint_predictor_results.json"
    _write_json(out_path, results)
    logger.info(
        "WROTE %s | joint-prior increment: %.4f (boot CI95 %s, binding p %.4f) | wall %.1fs",
        out_path,
        delta_obs,
        boot["delta_ci_95"],
        perm_binding["p_one_sided_upper"],
        time.time() - t_start,
    )


if __name__ == "__main__":
    main()
