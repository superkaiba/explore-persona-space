"""Issue #537 scoring harness (plan v6 §6 -- imported #524-v6 protocol, v6 re-base).

Pieces:

- **Metric registry + namespacing** (§6.1 contract): every metric id must be
  registered here with its tier (registered / exploratory / skip) and
  polarity ("larger = more distant"). Unregistered or un-namespaced ids are
  REJECTED. KL ids disambiguate `gauss_kl_act` / `kl_out_seq` / `kl_judge`.
- **Quarantine enforcement** (§4.3): quarantined cells masked by default;
  ``--final-test`` unmasks AND appends the invocation to
  ``final_test_invocations.jsonl`` (the split burns on use).
- **LTCO CV**: pooled out-of-fold leave-two-contexts-out predictions on the
  16x16 shared-instance block; per-behavior Spearman/R² + ΔR² over the
  symmetric baseline (`gauss_kl_act` @ L22 last_prompt).
- **Context-clustered dyadic bootstrap** (B=2000) for score CIs.
- **Censored/Tobit ΔLL fallback** when the censored cell fraction ≥ 10%.
- **`--selftest`** (plan A37): estimator unit tests on synthetic
  per-question data with KNOWN variance -- exercised by the P3 smoke.

Activation-derived metric values are computed from the P1 clouds
(``eval_results/issue_537/clouds/<cid>__<anchor>.npz``). Implemented rows in
this round: centroid cosine, euclidean, ``gauss_kl_act`` (PCA-16 Gaussian,
#502 recipe), pooled Mahalanobis, RBF-MMD², the A1 rank-1 family
(raw / whitened projection + norm ratio), and the ``cos_to_neutral``
null-anchor row. Registered-but-not-yet-wired rows fail LOUD by name.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_score_metric")

REPO = Path(__file__).resolve().parents[1]
# I537_EVAL_ROOT: smoke-redirect for the eval artifact tree (real runs use default).
EVAL = Path(os.environ.get("I537_EVAL_ROOT", str(REPO / "eval_results/issue_537")))
SEED = 42
PRIMARY_LAYER = 22  # §6.3 registered primary (hidden_states index 22 = layer-22 output)
PRIMARY_ANCHOR = "last_prompt"  # #502 winner row; A7 varies the anchor axis

# ── Metric registry (§6.1; ids are the namespacing contract) ─────────────────
# implemented=False rows are REGISTERED but not yet wired -- requesting them
# fails loud by name (never silently skipped).
METRIC_REGISTRY: dict[str, dict] = {
    # v3 six (carried)
    "pv_dp": dict(
        tier="registered",
        family="v3_six",
        implemented=False,
        note="needs the P3 ΔP GPU pass (plan §9 v3-baselines row, ~2 GPU-h)",
    ),
    "gauss_kl_act": dict(tier="registered", family="v3_six", implemented=True),
    "kl_out_seq_oneway": dict(
        tier="registered",
        family="v3_six",
        implemented=False,
        note="needs the P3 output-KL GPU pass (plan §9 v3-baselines row, ~8 GPU-h)",
    ),
    "base_prior_bystander": dict(
        tier="registered",
        family="v3_six",
        implemented=True,
        symmetric=False,
        note="column effect from stored base artifacts (#444/#507 bystander_base_rate); "
        "distance polarity = -base (higher base prior predicts MORE leak)",
    ),
    "content_free": dict(
        tier="registered",
        family="v3_six",
        implemented=True,
        note="#507 base_rate_diff_neg_abs flipped to distance polarity: |base_i - base_j| "
        "(closer base rates predict more leak)",
    ),
    "neg_panel_prox": dict(tier="registered", family="v3_six", implemented=True),
    # A1 rank-1 family (#526 / absorbs #510)
    "rank1_proj_raw": dict(tier="registered", family="A1", implemented=True, symmetric=False),
    "rank1_proj_whitened": dict(tier="registered", family="A1", implemented=True, symmetric=False),
    "norm_ratio": dict(tier="registered", family="A1", implemented=True, symmetric=False),
    # A2 training-completion prior (absorbs #499) -- needs the P3 TF GPU pass
    # (plan §9 A2 row, ~3.5 GPU-h); fails loud until wired or descoped.
    "train_prior_tf": dict(tier="registered", family="A2", implemented=False),
    "train_prior_onpolicy": dict(tier="registered", family="A2", implemented=False),
    # A3 bake-off rest
    "euclidean": dict(tier="registered", family="A3", implemented=True),
    "centroid_cosine": dict(tier="registered", family="A3", implemented=True),
    "mahalanobis_pair": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="pair-averaged covariance in the PCA-16 subspace (#493 recipe)",
    ),
    "mahalanobis_pooled": dict(tier="registered", family="A3", implemented=True),
    "rbf_mmd2": dict(tier="registered", family="A3", implemented=True),
    "c2st": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="CV linear-probe AUC distance 2*|AUC-0.5| on PCA-16 (#493 recipe)",
    ),
    "delta_spectrum_coherence": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="#493 paired Δ-spectrum (raw #493 convention: shape scalar, polarity read at "
        "analysis time; #493 instability flag carries)",
    ),
    "delta_spectrum_mean_norm": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="#493 paired Δ-spectrum ‖mean Δ‖ (distance-like); #493 instability flag carries",
    ),
    "delta_spectrum_effective_dim": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="#493 paired Δ-spectrum participation ratio (shape scalar, polarity read at "
        "analysis time); #493 instability flag carries",
    ),
    "bures_w2": dict(
        tier="registered",
        family="A3",
        implemented=True,
        note="Bures-Wasserstein W2 between PCA-16 Gaussians",
    ),
    # A4 first-token rows
    "js_first_token": dict(
        tier="registered",
        family="A4",
        implemented=True,
        note="deprecated-as-canonical, labeled benchmark row",
    ),
    "kl_first_token_fwd": dict(tier="registered", family="A4", implemented=True, symmetric=False),
    "kl_first_token_rev": dict(tier="registered", family="A4", implemented=True, symmetric=False),
    # A5 sequence-level output divergences -- need the P3 teacher-forced GPU
    # passes (plan §9 A5 cheap ~3.5 GPU-h / RB ~6.5 GPU-h); fail loud until
    # wired or descoped (RB tier is descope rung v4-b).
    "js_out_seq": dict(tier="registered", family="A5", implemented=False),
    "kl_out_seq_fwd": dict(tier="registered", family="A5", implemented=False),
    "kl_out_seq_rev": dict(tier="registered", family="A5", implemented=False),
    "kl_asym_out_seq": dict(tier="registered", family="A5", implemented=False),
    "js_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=False),
    "kl_fwd_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=False),
    "kl_rev_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=False),
    # A6 -- needs the P3 taught-span TF GPU pass (plan §9 A6 row, ~1 GPU-h).
    "js_taught_span": dict(tier="registered", family="A6", implemented=False),
    # A8 null anchors (dead-baseline floor)
    "cos_to_assistant": dict(
        tier="registered",
        family="A8",
        implemented=True,
        symmetric=False,
        note="anchor = the `default` context: in the #537 battery the bare assistant IS "
        "the default context, so this row coincides with cos_to_neutral; kept as a "
        "labeled duplicate per §6.1 A8 (historically distinct anchors, #396/#415)",
    ),
    "js_to_assistant": dict(
        tier="registered",
        family="A8",
        implemented=True,
        symmetric=False,
        note="first-token JS to the `default` context's next-token distribution; same "
        "battery-level alias caveat as cos_to_assistant",
    ),
    "cos_to_neutral": dict(tier="registered", family="A8", implemented=True, symmetric=False),
    "js_to_neutral": dict(
        tier="registered",
        family="A8",
        implemented=True,
        symmetric=False,
        note="first-token JS to the `default` context's next-token distribution",
    ),
    "cos_to_trained_midpoint": dict(
        tier="registered",
        family="A8",
        implemented=True,
        symmetric=False,
        note="column distance to the midpoint of the scored block's context means",
    ),
    # SKIP rows (cost without expectation -- never scored)
    "kl_judge": dict(tier="skip", family="deprecated", implemented=False),
    "in_context_rate_m3": dict(tier="skip", family="deprecated", implemented=False),
    "first_step_gradient": dict(tier="skip", family="deprecated", implemented=False),
}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


# ── Cloud loading + metric matrices ──────────────────────────────────────────


def _load_cloud(cid: str, anchor: str, layer: int) -> np.ndarray:
    p = EVAL / "clouds" / f"{cid}__{anchor}.npz"
    assert p.exists(), f"cloud missing: {p} (run --phase 1 --steps clouds)"
    arr = np.load(p)["hidden"]  # (n_probes, L+1, H)
    assert arr.ndim == 3, arr.shape
    return arr[:, layer, :].astype(np.float64)


def _assert_probe_alignment(cids: list[str], anchor: str) -> None:
    """Paired metrics (Δ-spectrum) index clouds row-by-row; assert the probe
    arrays are identical across contexts before trusting that alignment."""
    ref = None
    for c in cids:
        p = EVAL / "clouds" / f"{c}__{anchor}.npz"
        probes = list(np.load(p, allow_pickle=True)["probes"])
        if ref is None:
            ref = probes
        else:
            assert probes == ref, f"probe order mismatch between clouds: {cids[0]} vs {c}"


def _drop_nan_rows(x: np.ndarray) -> np.ndarray:
    """Drop probe rows with any NaN (empty-response mean_response anchors)."""
    return x[~np.any(np.isnan(x), axis=1)]


def _base_rates_for(behavior: str, cids: list[str]) -> dict[str, float]:
    """Per-context base expression level from STORED artifacts (zero GPU).

    marker: mean base logP(※) at the slot from the Stage-1 marker_base_slots
    caches; judge rows: P0 headroom rates (refusal: the XSTest-safe panel --
    the §6 primary DV's base, requires the round-2 rates_by_panel split).
    """
    if behavior == "marker":
        out: dict[str, float] = {}
        for c in cids:
            p = EVAL / "marker_base_slots" / f"{c}.json"
            assert p.exists(), f"base slot stats missing: {p} (run --phase 1 xeval Stage 1)"
            stats = json.loads(p.read_text())["stats"]
            out[c] = float(np.mean([s["logp"] for s in stats]))
        return out
    p = EVAL / "p0/headroom_rates" / f"{behavior}.json"
    assert p.exists(), f"headroom rates missing: {p} (run --phase 0 headroom-judge)"
    payload = json.loads(p.read_text())
    rates = payload["rates"]
    if behavior == "refusal":
        rates = payload.get("rates_by_panel", {}).get("xstest_safe")
        assert rates, (
            f"{p} lacks rates_by_panel.xstest_safe -- re-run --phase 0 headroom-judge "
            "(round-2 §6 panel split)"
        )
    missing = [c for c in cids if c not in rates]
    assert not missing, f"base rates missing for contexts {missing} in {p}"
    return {c: float(rates[c]) for c in cids}


def _c2st_dist(xa: np.ndarray, xb: np.ndarray, folds: int = 5) -> float:
    """CV linear-probe classifier-2-sample test as a distance: 2*|AUC - 0.5|
    (#493 recipe; 0 = indistinguishable, 1 = perfectly separable)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(xa) < folds or len(xb) < folds:
        return float("nan")
    x = np.vstack([xa, xb])
    y = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(x, y):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(x[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(x[te])))
    return float(min(1.0, 2.0 * abs(float(np.mean(aucs)) - 0.5)))


def _delta_spectrum(xa: np.ndarray, xb: np.ndarray) -> dict[str, float]:
    """#493 PAIRED Δ-displacement spectrum (mean_norm / coherence / effective_dim).

    Requires matched probe ordering (asserted via _assert_probe_alignment by
    the caller); rows with NaN on EITHER side are dropped on BOTH (paired drop).
    """
    assert xa.shape == xb.shape, (xa.shape, xb.shape)
    mask = ~(np.any(np.isnan(xa), axis=1) | np.any(np.isnan(xb), axis=1))
    xa, xb = xa[mask], xb[mask]
    if len(xa) < 2:
        return {"mean_norm": float("nan"), "coherence": float("nan"), "effective_dim": float("nan")}
    delta = xb - xa  # (n_q, H)
    mean_delta = delta.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    total_energy = float(np.sum(delta**2))
    if total_energy < 1e-12 or mean_norm < 1e-12:
        coherence = 0.0
    else:
        proj = delta @ mean_delta / mean_norm
        coherence = float(np.sum(proj**2) / total_energy)
    delta_c = delta - delta.mean(axis=0, keepdims=True)
    gram = delta_c @ delta_c.T
    eig = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    s1, s2 = eig.sum(), (eig**2).sum()
    eff_dim = 0.0 if s2 < 1e-18 else float(s1**2 / s2)
    return {"mean_norm": mean_norm, "coherence": coherence, "effective_dim": eff_dim}


def _bures_w2(mu_p, cov_p, mu_q, cov_q) -> float:
    """Bures-Wasserstein W2 distance between two Gaussians (PCA-16)."""
    from scipy.linalg import sqrtm

    sq = sqrtm(cov_q)
    cross = sqrtm(sq @ cov_p @ sq)
    if np.iscomplexobj(cross):
        cross = cross.real
    w2sq = float(np.sum((mu_p - mu_q) ** 2) + np.trace(cov_p + cov_q - 2 * cross))
    return float(np.sqrt(max(w2sq, 0.0)))


def _pca16(pooled: np.ndarray) -> np.ndarray:
    """Top-16 principal axes of the pooled probe-level cloud (#502 recipe)."""
    x = pooled - pooled.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:16]  # (16, H)


def metric_matrix(  # noqa: C901 - one dispatch table per metric family; splitting would scatter the polarity contract
    metric_id: str,
    cids: list[str],
    *,
    anchor: str = PRIMARY_ANCHOR,
    layer: int = PRIMARY_LAYER,
    centered: bool = False,
    behavior: str = "marker",
) -> np.ndarray:
    """Pairwise metric matrix D[i, j] over contexts (polarity: larger = more distant).

    Directional rows (A1 family, the *_to_* null anchors, base_prior) return
    asymmetric matrices; D[i, j] reads "trained at i, evaluated at j"
    (v_S = i, v_T = j). ``behavior`` selects the base-rate artifacts for the
    base_prior_bystander / content_free rows (other rows are base-model
    geometry, behavior-independent).
    """
    spec = METRIC_REGISTRY.get(metric_id)
    assert spec is not None, f"unregistered metric id {metric_id!r} (§6.1 namespacing contract)"
    assert spec["implemented"], (
        f"metric {metric_id!r} is registered (family {spec['family']}) but not wired in this "
        "round -- implement it or descope with an epm:progress note; never silently skip."
    )
    n = len(cids)
    d = np.full((n, n), np.nan)

    # Base-rate-derived rows (stored artifacts; no clouds needed).
    if metric_id in ("base_prior_bystander", "content_free"):
        base = _base_rates_for(behavior, cids)
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i == j:
                    continue
                if metric_id == "base_prior_bystander":
                    d[i, j] = -base[cj]  # higher base prior -> more leak -> LESS distant
                else:  # content_free: closer base rates predict more leak
                    d[i, j] = abs(base[ci] - base[cj])
        return d
    # First-token-cache rows (no clouds needed).
    if metric_id in (
        "js_first_token",
        "kl_first_token_fwd",
        "kl_first_token_rev",
        "js_to_neutral",
        "js_to_assistant",
    ):
        return _first_token_matrix(metric_id, cids)
    # Paired Δ-spectrum rows need ALIGNED raw clouds (paired NaN drop inside).
    if metric_id.startswith("delta_spectrum_"):
        _assert_probe_alignment(cids, anchor)
        raw = {c: _load_cloud(c, anchor, layer) for c in cids}
        key = metric_id.removeprefix("delta_spectrum_")
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i < j:  # all three Δ-spectrum scalars are swap-symmetric
                    v = _delta_spectrum(raw[ci], raw[cj])[key]
                    d[i, j] = d[j, i] = v
        return d

    clouds = {c: _drop_nan_rows(_load_cloud(c, anchor, layer)) for c in cids}
    grand_mean = np.mean([clouds[c].mean(axis=0) for c in cids], axis=0)
    mu = {c: clouds[c].mean(axis=0) - (grand_mean if centered else 0.0) for c in cids}

    if metric_id in (
        "centroid_cosine",
        "euclidean",
        "norm_ratio",
        "rank1_proj_raw",
        "cos_to_neutral",
        "cos_to_assistant",
        "cos_to_trained_midpoint",
    ):
        if metric_id in ("cos_to_neutral", "cos_to_assistant"):
            # In the #537 battery the bare assistant IS the `default` context,
            # so both null anchors resolve there (registry notes the alias).
            assert "default" in cids, f"{metric_id} needs the default context in the panel"
            v_anchor = mu["default"]
        elif metric_id == "cos_to_trained_midpoint":
            v_anchor = np.mean([mu[c] for c in cids], axis=0)
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i == j:
                    continue
                vi, vj = mu[ci], mu[cj]
                if metric_id == "centroid_cosine":
                    cos = float(vi @ vj / (np.linalg.norm(vi) * np.linalg.norm(vj)))
                    d[i, j] = 1.0 - cos
                elif metric_id == "euclidean":
                    d[i, j] = float(np.linalg.norm(vi - vj))
                elif metric_id == "norm_ratio":
                    d[i, j] = float(np.linalg.norm(vj) / np.linalg.norm(vi))
                elif metric_id == "rank1_proj_raw":
                    # leak proxy: (v_T·v_S)/||v_S||²; polarity-flip to distance
                    d[i, j] = -float(vj @ vi / (vi @ vi))
                else:  # *_to_* null anchors -- column effect of the eval context
                    cos = float(vj @ v_anchor / (np.linalg.norm(vj) * np.linalg.norm(v_anchor)))
                    d[i, j] = 1.0 - cos
        return d

    # PCA-16 subspace metrics.
    pooled = np.concatenate([clouds[c] for c in cids], axis=0)
    basis = _pca16(pooled)  # (16, H)
    z = {c: (clouds[c] - pooled.mean(axis=0)) @ basis.T for c in cids}  # (n_probes, 16)
    mu16 = {c: z[c].mean(axis=0) for c in cids}
    cov16 = {c: np.cov(z[c].T) + 1e-6 * np.eye(16) for c in cids}
    pooled_cov = sum(cov16.values()) / n

    if metric_id == "rank1_proj_whitened":
        cinv = np.linalg.inv(pooled_cov)
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i != j:
                    vs, vt = mu16[ci], mu16[cj]
                    d[i, j] = -float(vt @ cinv @ vs / (vs @ cinv @ vs))
        return d
    if metric_id == "mahalanobis_pooled":
        cinv = np.linalg.inv(pooled_cov)
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i != j:
                    diff = mu16[ci] - mu16[cj]
                    d[i, j] = float(np.sqrt(diff @ cinv @ diff))
        return d
    if metric_id == "mahalanobis_pair":
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i < j:  # pair-averaged covariance is swap-symmetric
                    cov_pair = 0.5 * (cov16[ci] + cov16[cj])
                    diff = mu16[ci] - mu16[cj]
                    v = float(np.sqrt(diff @ np.linalg.solve(cov_pair, diff)))
                    d[i, j] = d[j, i] = v
        return d
    if metric_id == "bures_w2":
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i < j:
                    v = _bures_w2(mu16[ci], cov16[ci], mu16[cj], cov16[cj])
                    d[i, j] = d[j, i] = v
        return d
    if metric_id == "c2st":
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i < j:
                    v = _c2st_dist(z[ci], z[cj])
                    d[i, j] = d[j, i] = v
        return d
    if metric_id == "gauss_kl_act":
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i != j:
                    d[i, j] = 0.5 * (
                        _gauss_kl(mu16[ci], cov16[ci], mu16[cj], cov16[cj])
                        + _gauss_kl(mu16[cj], cov16[cj], mu16[ci], cov16[ci])
                    )
        return d
    if metric_id == "rbf_mmd2":
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i < j:
                    v = _rbf_mmd2(z[ci], z[cj])
                    d[i, j] = d[j, i] = v
        return d
    if metric_id == "neg_panel_prox":
        from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS

        neg_mu = []
        for nc in NEGATIVE_CIDS:
            neg_cloud = _load_cloud(nc, anchor, layer)
            neg_mu.append((neg_cloud - pooled.mean(axis=0)) @ basis.T)
        neg_centroid = np.concatenate(neg_mu, axis=0).mean(axis=0)
        for j, cj in enumerate(cids):
            col = float(np.linalg.norm(mu16[cj] - neg_centroid))
            d[:, j] = col
            d[j, j] = np.nan
        return d
    if metric_id in ("js_first_token", "kl_first_token_fwd", "kl_first_token_rev"):
        return _first_token_matrix(metric_id, cids)
    raise AssertionError(f"unhandled implemented metric {metric_id!r}")


def _gauss_kl(mu_p, cov_p, mu_q, cov_q) -> float:
    k = mu_p.size
    cq_inv = np.linalg.inv(cov_q)
    diff = mu_q - mu_p
    val = 0.5 * (
        np.trace(cq_inv @ cov_p)
        + diff @ cq_inv @ diff
        - k
        + np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    )
    return float(val)


def _rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased RBF-MMD² with the median heuristic (#511 small-N robust row)."""
    zz = np.concatenate([x, y], axis=0)
    d2 = ((zz[:, None, :] - zz[None, :, :]) ** 2).sum(-1)
    sigma2 = np.median(d2[d2 > 0])
    k = np.exp(-d2 / (2 * sigma2))
    nx, ny = len(x), len(y)
    kxx = (k[:nx, :nx].sum() - np.trace(k[:nx, :nx])) / (nx * (nx - 1))
    kyy = (k[nx:, nx:].sum() - np.trace(k[nx:, nx:])) / (ny * (ny - 1))
    kxy = k[:nx, nx:].mean()
    return float(kxx + kyy - 2 * kxy)


def _first_token_matrix(metric_id: str, cids: list[str]) -> np.ndarray:
    """A4 + A8 first-token rows from the full-vocab logit cache (per-probe mean).

    A4 (``js_first_token`` / ``kl_first_token_{fwd,rev}``): pairwise (i, j)
    divergences. A8 (``js_to_{neutral,assistant}``): column effect -- the JS of
    eval context j's next-token distribution to the ``default`` context's
    (both null anchors resolve to ``default`` in this battery; registry note).
    """
    dists = {}
    anchor_cids = set(cids)
    if metric_id in ("js_to_neutral", "js_to_assistant"):
        assert "default" in cids, f"{metric_id} needs the default context in the panel"
    for c in anchor_cids:
        p = EVAL / "first_token_cache" / f"{c}.npz"
        assert p.exists(), f"first-token cache missing: {p}"
        logits = np.load(p)["logits"].astype(np.float64)  # (n_probes, V)
        logp = logits - _logsumexp(logits)
        dists[c] = logp

    def _js(lp: np.ndarray, lq: np.ndarray) -> float:
        m = np.logaddexp(lp, lq) - np.log(2)
        js = 0.5 * np.sum(np.exp(lp) * (lp - m), axis=-1) + 0.5 * np.sum(
            np.exp(lq) * (lq - m), axis=-1
        )
        return float(np.mean(js))

    n = len(cids)
    d = np.full((n, n), np.nan)
    if metric_id in ("js_to_neutral", "js_to_assistant"):
        for j, cj in enumerate(cids):
            col = _js(dists[cj], dists["default"])
            d[:, j] = col
            d[j, j] = np.nan
        return d
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                continue
            lp, lq = dists[ci], dists[cj]
            assert lp.shape == lq.shape, (lp.shape, lq.shape)
            if metric_id == "kl_first_token_fwd":
                d[i, j] = float(np.mean(np.sum(np.exp(lp) * (lp - lq), axis=-1)))
            elif metric_id == "kl_first_token_rev":
                d[i, j] = float(np.mean(np.sum(np.exp(lq) * (lq - lp), axis=-1)))
            else:  # js
                d[i, j] = _js(lp, lq)
    return d


def _logsumexp(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


# ── Quarantine ───────────────────────────────────────────────────────────────


def quarantine_mask(
    behavior: str,
    train_cids: list[str],
    eval_cids: list[str],
    *,
    final_test: bool,
    invocation_note: str,
) -> np.ndarray:
    """Boolean mask (True = usable) honoring the §4.3 quarantine manifest."""
    qp = EVAL / "prereg/quarantine_manifest.json"
    mask = np.ones((len(train_cids), len(eval_cids)), dtype=bool)
    if not qp.exists():
        logger.warning("quarantine manifest absent (pre-freeze run) -- nothing masked")
        return mask
    q = json.loads(qp.read_text())
    held_out = set(q["held_out_eval_cids"])
    cells = {tuple(c) for c in q["quarantined_cells"].get(behavior, [])}
    for ii, i_cid in enumerate(train_cids):
        for ji, j_cid in enumerate(eval_cids):
            if j_cid in held_out or (i_cid, j_cid) in cells:
                mask[ii, ji] = False
    if final_test:
        inv = EVAL / "prereg/final_test_invocations.jsonl"
        with inv.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        "behavior": behavior,
                        "note": invocation_note,
                        "git_commit": _git_commit(),
                    }
                )
                + "\n"
            )
        logger.warning("[final-test] quarantine UNMASKED -- invocation logged to %s", inv)
        return np.ones_like(mask)
    return mask


# ── LTCO CV + clustered bootstrap + Tobit fallback ───────────────────────────


def ltco_cv_predictions(d_mat: np.ndarray, g_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold leave-two-contexts-out predictions on the shared block.

    For each unordered context pair (a, b): fit OLS G ~ D on the off-diagonal
    cells among the other contexts; predict the 2 ordered (a, b) cells.
    Returns (y_true, y_pred) pooled over folds (NaN cells dropped).
    """
    n = d_mat.shape[0]
    assert d_mat.shape == g_mat.shape == (n, n), (d_mat.shape, g_mat.shape)
    y_true, y_pred = [], []
    for a in range(n):
        for b in range(a + 1, n):
            keep = [i for i in range(n) if i not in (a, b)]
            xs, ys = [], []
            for i in keep:
                for j in keep:
                    if i != j and np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                        xs.append(d_mat[i, j])
                        ys.append(g_mat[i, j])
            if len(xs) < 8:
                continue
            coef = np.polyfit(np.array(xs), np.array(ys), deg=1)
            for i, j in ((a, b), (b, a)):
                if np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                    y_true.append(g_mat[i, j])
                    y_pred.append(float(np.polyval(coef, d_mat[i, j])))
    return np.array(y_true), np.array(y_pred)


def score_metric_vs_g(
    d_mat: np.ndarray, g_mat: np.ndarray, baseline_mat: np.ndarray | None = None
) -> dict:
    """Spearman + out-of-fold R² (+ ΔR² over the symmetric baseline)."""
    mask = np.isfinite(d_mat) & np.isfinite(g_mat) & ~np.eye(d_mat.shape[0], dtype=bool)
    assert mask.sum() >= 10, f"too few usable cells ({mask.sum()})"
    rho = float(spearmanr(d_mat[mask], g_mat[mask]).statistic)
    y_true, y_pred = ltco_cv_predictions(d_mat, g_mat)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"spearman": rho, "oof_r2": r2, "n_cells": int(mask.sum())}
    if baseline_mat is not None:
        yb_true, yb_pred = ltco_cv_predictions(baseline_mat, g_mat)
        ssb = float(((yb_true - yb_pred) ** 2).sum())
        sstb = float(((yb_true - yb_true.mean()) ** 2).sum())
        out["baseline_oof_r2"] = 1.0 - ssb / sstb if sstb > 0 else float("nan")
        out["delta_r2"] = out["oof_r2"] - out["baseline_oof_r2"]
    return out


def context_cluster_bootstrap(
    d_mat: np.ndarray, g_mat: np.ndarray, b: int = 2000, seed: int = 537
) -> dict:
    """Context-clustered dyadic bootstrap CI on the Spearman (resample contexts).

    A metric whose matrix is (near-)constant -- e.g. a saturated c2st at
    distance 1.0 everywhere -- yields degenerate draws; that is INFORMATION
    about the metric (it cannot rank cells), not a harness failure, so it
    returns a ``degenerate: True`` row instead of killing the whole
    leaderboard run (plan §7: no gates inside P3; flags degrade gracefully).
    """
    n = d_mat.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        sub_d = d_mat[np.ix_(idx, idx)]
        sub_g = g_mat[np.ix_(idx, idx)]
        m = np.isfinite(sub_d) & np.isfinite(sub_g) & ~np.eye(n, dtype=bool)
        if m.sum() < 10 or np.unique(sub_d[m]).size < 3:
            continue
        vals.append(spearmanr(sub_d[m], sub_g[m]).statistic)
    arr = np.array([v for v in vals if np.isfinite(v)])
    if arr.size < b // 4:
        logger.warning(
            "[score] bootstrap degenerate (%d/%d usable draws) -- metric matrix is "
            "(near-)constant; CI flagged, not crashed",
            arr.size,
            b,
        )
        return {
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_draws": int(arr.size),
            "degenerate": True,
        }
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {"ci_lo": float(lo), "ci_hi": float(hi), "n_draws": int(arr.size)}


def tobit_delta_ll(x: np.ndarray, y: np.ndarray, ceiling: float) -> dict:
    """Censored-Gaussian (Tobit) ΔLL of slope vs intercept-only (§6 fallback)."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    cens = y >= ceiling

    def nll(params, with_slope: bool):
        if with_slope:
            a, bb, log_s = params
            mu = a + bb * x
        else:
            a, log_s = params
            mu = np.full_like(y, a)
        s = np.exp(log_s)
        ll = np.where(
            cens,
            norm.logsf((ceiling - mu) / s),
            norm.logpdf((y - mu) / s) - log_s,
        )
        return -ll.sum()

    r0 = minimize(nll, x0=[y.mean(), np.log(y.std() + 1e-6)], args=(False,), method="Nelder-Mead")
    r1 = minimize(
        nll, x0=[y.mean(), 0.0, np.log(y.std() + 1e-6)], args=(True,), method="Nelder-Mead"
    )
    assert r0.success and r1.success, (r0.message, r1.message)
    return {
        "delta_ll": float(r0.fun - r1.fun),
        "censored_frac": float(cens.mean()),
        "slope": float(r1.x[1]),
    }


# ── Self-test (plan A37) ─────────────────────────────────────────────────────


def selftest() -> None:
    """Estimator unit tests on synthetic per-question data with known variance."""
    from explore_persona_space.experiments.i537_estimators import (
        antisym_fraction_split_half,
        cluster_bootstrap_var,
        decompose_sym_anti,
        h_structure_read,
        question_bootstrap_var,
        spearman_brown,
        split_half_noise_var,
    )

    rng = np.random.default_rng(537)
    n_q, sigma = 32, 2.0
    true_var = sigma**2 / n_q

    # 1. question bootstrap + split-half recover the known cell-mean variance.
    reps_b, reps_s = [], []
    for r in range(40):
        per_q = rng.normal(0.0, sigma, size=n_q)
        reps_b.append(question_bootstrap_var(per_q, b=2000, seed=r))
        reps_s.append(split_half_noise_var(per_q, k=200, seed=537))
    for name, est in (("bootstrap", np.mean(reps_b)), ("split_half", np.mean(reps_s))):
        assert 0.7 * true_var < est < 1.3 * true_var, (name, est, true_var)
    print(
        f"[selftest] noise-floor recovery OK (true {true_var:.4f}, "
        f"bootstrap {np.mean(reps_b):.4f}, split-half {np.mean(reps_s):.4f})"
    )

    # 2. EM cluster bootstrap: clustered responses inflate variance vs naive iid.
    q_means = rng.normal(0, 1.0, size=8)
    per_resp = np.concatenate([rng.normal(m, 0.3, size=5) for m in q_means])
    qids = np.repeat(np.arange(8), 5)
    v_cluster = cluster_bootstrap_var(per_resp, qids, b=2000, seed=0)
    v_naive = per_resp.var(ddof=1) / per_resp.size
    assert v_cluster > 1.5 * v_naive, (v_cluster, v_naive)
    print(f"[selftest] EM cluster bootstrap OK ({v_cluster:.4f} > naive {v_naive:.4f})")

    # 3. H-structure: planted context structure passes; pure noise fails.
    n_i, n_j = 16, 30
    noise_floor = true_var
    ctx_effect = rng.normal(0, np.sqrt(6 * noise_floor), size=n_j)
    g_struct = ctx_effect[None, :] + rng.normal(0, np.sqrt(noise_floor), size=(n_i, n_j))
    g_noise = rng.normal(0, np.sqrt(noise_floor), size=(n_i, n_j))
    nv = np.full((n_i, n_j), noise_floor)
    assert h_structure_read(g_struct, nv)["pass_2x"] is True
    assert h_structure_read(g_noise, nv)["pass_2x"] is False
    print("[selftest] H-structure kill read OK (planted passes, noise fails)")

    # 4. antisymmetric fraction: split-half correction kills question noise.
    n = 16
    s_part = rng.normal(0, 1, size=(n, n))
    s_part = 0.5 * (s_part + s_part.T)
    a_part = rng.normal(0, 1, size=(n, n))
    a_part = 0.5 * (a_part - a_part.T)
    alpha = 0.5  # planted anti std ratio → anti_frac = alpha²/(1+alpha²) = 0.2
    g_true = s_part + alpha * a_part
    noise_sd = 0.8
    half_a = g_true + rng.normal(0, noise_sd, size=(n, n))
    half_b = g_true + rng.normal(0, noise_sd, size=(n, n))
    raw = decompose_sym_anti(half_a)["anti_frac"]
    corr = antisym_fraction_split_half(half_a, half_b)["anti_frac_corrected"]
    true_frac = (alpha**2 * a_part[~np.eye(n, dtype=bool)].var()) / (
        g_true[~np.eye(n, dtype=bool)].var()
    )
    assert abs(corr - true_frac) < abs(raw - true_frac), (raw, corr, true_frac)
    print(
        f"[selftest] antisym split-half OK (true {true_frac:.3f}, raw {raw:.3f}, "
        f"corrected {corr:.3f})"
    )

    # 5. Spearman-Brown monotone + identity checks.
    assert spearman_brown(1.0) == 1.0 and spearman_brown(0.0) == 0.0
    assert spearman_brown(0.5) == 2 * 0.5 / 1.5

    # 6. LTCO CV + Tobit smoke on synthetic D→G.
    d_syn = np.abs(rng.normal(0, 1, size=(16, 16)))
    np.fill_diagonal(d_syn, np.nan)
    g_syn = -1.5 * d_syn + rng.normal(0, 0.3, size=(16, 16))
    res = score_metric_vs_g(d_syn, g_syn)
    assert res["spearman"] < -0.8 and res["oof_r2"] > 0.5, res
    x = rng.normal(0, 1, 100)
    y_lat = 1.0 + 2.0 * x + rng.normal(0, 0.5, 100)
    y_obs = np.minimum(y_lat, 2.0)
    tob = tobit_delta_ll(x, y_obs, 2.0)
    assert tob["delta_ll"] > 10 and 1.5 < tob["slope"] < 2.5, tob
    print(
        f"[selftest] LTCO CV (rho={res['spearman']:.2f}, R²={res['oof_r2']:.2f}) "
        f"+ Tobit (slope {tob['slope']:.2f}) OK"
    )
    print("[selftest] ALL PASS")


# ── main ─────────────────────────────────────────────────────────────────────


def _load_g(behavior: str, train_cids: list[str], eval_cids: list[str]) -> np.ndarray:
    """16x16 shared-instance G block for ONE behavior from the assembled tensor.

    Round-2 fix (p3-behavior-loader-marker-only): selects the requested
    behavior's axis -- the round-1 loader hardcoded the marker index, so
    ``--behavior fact`` silently scored marker G over fact context lists.
    """
    p = EVAL / "G_tensor/G_tensor.npz"
    assert p.exists(), f"G tensor missing: {p} (run i537_assemble_tensor.py)"
    z = np.load(p, allow_pickle=True)
    behaviors = list(z["behaviors"])
    assert behavior in behaviors, (behavior, behaviors)
    bi = behaviors.index(behavior)
    all_train = list(z["train_cids"][bi])
    all_eval = list(z["eval_cids"][bi])
    g = np.full((len(train_cids), len(eval_cids)), np.nan)
    for ii, i_cid in enumerate(train_cids):
        for ji, j_cid in enumerate(eval_cids):
            g[ii, ji] = z["G"][bi, all_train.index(i_cid), all_eval.index(j_cid), 0]
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="plan A37 estimator unit test")
    ap.add_argument("--metric", default=None, help="one §6.1 metric id")
    ap.add_argument(
        "--all-registered", action="store_true", help="score every implemented registered row"
    )
    ap.add_argument("--behavior", default="marker")
    ap.add_argument("--anchor", default=PRIMARY_ANCHOR)
    ap.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    ap.add_argument(
        "--centered",
        action="store_true",
        help="prompt-centered variant axis (subtract the grand mean before the metric)",
    )
    ap.add_argument(
        "--final-test",
        action="store_true",
        help="unmask the quarantined split (invocation is LOGGED + burned)",
    )
    ap.add_argument(
        "--allow-missing-registered",
        action="store_true",
        help="tolerate registered-but-unimplemented rows in --all-registered "
        "(EXPLICIT opt-in; the default exits non-zero naming them AFTER scoring "
        "+ persisting every implemented row)",
    )
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    from explore_persona_space.experiments.i537_contexts import train_cids_for

    # Shared-instance 16x16 block: the 15 row-independent cids + own binst,
    # restricted to the eval side's matching columns.
    cids = train_cids_for(args.behavior)
    g_mat = _load_g(args.behavior, cids, cids)
    qmask = quarantine_mask(
        args.behavior,
        cids,
        cids,
        final_test=args.final_test,
        invocation_note=f"metric={args.metric or 'all'}",
    )
    g_mat = np.where(qmask, g_mat, np.nan)

    not_implemented: list[str] = []
    if args.all_registered:
        metric_ids = [
            m for m, s in METRIC_REGISTRY.items() if s["tier"] == "registered" and s["implemented"]
        ]
        not_implemented = sorted(
            m
            for m, s in METRIC_REGISTRY.items()
            if s["tier"] == "registered" and not s["implemented"]
        )
    else:
        metric_ids = [args.metric]
    assert metric_ids and all(m for m in metric_ids), "pass --metric or --all-registered"

    def _score(mid: str, *, anchor: str, layer: int, centered: bool, baseline_mat) -> dict:
        d_mat = metric_matrix(
            mid, cids, anchor=anchor, layer=layer, centered=centered, behavior=args.behavior
        )
        res = score_metric_vs_g(d_mat, g_mat, baseline_mat=baseline_mat)
        res["bootstrap"] = context_cluster_bootstrap(d_mat, g_mat)
        res["tier"] = METRIC_REGISTRY[mid]["tier"]
        res["family"] = METRIC_REGISTRY[mid]["family"]
        if METRIC_REGISTRY[mid].get("note"):
            res["note"] = METRIC_REGISTRY[mid]["note"]
        res["variant"] = {"anchor": anchor, "layer": layer, "centered": centered}
        return res

    baseline = metric_matrix(
        "gauss_kl_act", cids, anchor=args.anchor, layer=args.layer, behavior=args.behavior
    )
    results = {}
    for mid in metric_ids:
        res = _score(
            mid,
            anchor=args.anchor,
            layer=args.layer,
            centered=args.centered,
            baseline_mat=None if mid == "gauss_kl_act" else baseline,
        )
        results[mid] = res
        logger.info("[score] %s: rho=%.3f oof_R²=%.3f", mid, res["spearman"], res["oof_r2"])
    if args.all_registered:
        # §6.1 A3 explicitly registers the #509 representative early-layer cell
        # `end_of_system x L02 x cosine x centered` as part of the raw-vs-
        # centered variant axis; score it under a variant-tagged key (variants
        # of ONE row, not a new metric id -- KL-namespacing rule).
        rep = _score(
            "centroid_cosine",
            anchor="end_of_system",
            layer=2,
            centered=True,
            baseline_mat=baseline,
        )
        results["centroid_cosine[end_of_system,L02,centered]"] = rep
        logger.info(
            "[score] #509 representative cell: rho=%.3f oof_R²=%.3f",
            rep["spearman"],
            rep["oof_r2"],
        )

    out = EVAL / "baselines/baseline_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {"scores": {}}
    existing["scores"].update(results)
    existing.update(
        {
            "schema_version": 1,
            "git_commit": _git_commit(),
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "behavior": args.behavior,
            "anchor": args.anchor,
            "layer": args.layer,
            "centered": args.centered,
            "final_test": args.final_test,
            "registered_not_implemented": not_implemented,
        }
    )
    out.write_text(json.dumps(existing, indent=1))
    logger.info("[score] wrote %s (%d rows)", out, len(results))
    if not_implemented:
        # NEVER a silent gap (§6.1 contract, round-2 fix): every implemented
        # row is scored + persisted ABOVE, then the run exits non-zero naming
        # the registered rows still missing -- implement them or descope with
        # an epm:progress note; --allow-missing-registered is the explicit
        # opt-in for intermediate runs.
        msg = (
            f"[score] {len(not_implemented)} REGISTERED §6.1 rows are not wired: "
            + ", ".join(not_implemented)
            + " (all are P3 GPU passes -- see each row's registry note). Scored rows were "
            "persisted; rerun with --allow-missing-registered to tolerate the gap explicitly."
        )
        if args.allow_missing_registered:
            logger.warning(msg)
        else:
            raise SystemExit(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
