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
        implemented=True,
        symmetric=False,
        note="v9 dedup (plan §4.0): the parent persona-vectors content is implemented ONCE "
        "as behavior_vector_proj_shift/_level; pv_dp is routed to redundant_with: "
        "behavior_vector_proj_shift (the parent's registered ΔP-at-readout row reduces to the "
        "static projection) and reads the SAME behavior_vector_scores artifact (shift track)",
        redundant_with="behavior_vector_proj_shift",
    ),
    "gauss_kl_act": dict(tier="registered", family="v3_six", implemented=True),
    "kl_out_seq_oneway": dict(
        tier="registered",
        family="v3_six",
        implemented=True,
        note="one-way output-KL over the full realized reply; reads the SAME per-context "
        "output-distribution JSONs as the A5 family (no separate forward pass; plan §9 row "
        "folded into A5)",
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
    # A2 training-completion prior (absorbs #499) -- teacher-forced / on-policy
    # log-prob of the cell's POSITIVE TRAINING completions under each context
    # (i537_dropped_predictors.py A2 pass writes realization_scores/<b>/a2_*.json).
    # Directional (train-side prior, vs the eval-side base_prior_bystander).
    "train_prior_tf": dict(tier="registered", family="A2", implemented=True, symmetric=False),
    "train_prior_onpolicy": dict(tier="registered", family="A2", implemented=True, symmetric=False),
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
    # A5 sequence-level output divergences -- full-reply next-token-distribution
    # divergences between two contexts over the realized completion. The
    # i537_dropped_predictors.py A5 pass writes per-context top-k=512 sparse
    # output distributions (realization_scores/<b>/a5_<cid>.json); metric_matrix
    # builds the pairwise D at scoring time. kl_*_fwd/rev are directional.
    "js_out_seq": dict(tier="registered", family="A5", implemented=True),
    "kl_out_seq_fwd": dict(tier="registered", family="A5", implemented=True, symmetric=False),
    "kl_out_seq_rev": dict(tier="registered", family="A5", implemented=True, symmetric=False),
    "kl_asym_out_seq": dict(tier="registered", family="A5", implemented=True, symmetric=False),
    # A5_rb response-bucketed variants: same stored output distributions,
    # bucketed by response position before the divergence (the dropped script's
    # a5_rb_<cid>.json carries the per-bucket distributions).
    "js_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=True),
    "kl_fwd_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=True, symmetric=False),
    "kl_rev_out_seq_rb": dict(tier="registered", family="A5_rb", implemented=True, symmetric=False),
    # A6 taught-span JS: JS restricted to the taught-span token positions only
    # (the sharp fact-row case). a6_<cid>.json carries the taught-span dists.
    "js_taught_span": dict(tier="registered", family="A6", implemented=True),
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
        note="column distance to the midpoint of the scored block's context means; "
        "RAW-ONLY -- under --centered the anchor degenerates to the zero vector "
        "(grand-mean centering subtracts exactly the mean of the context means) "
        "and the scorer fails loud",
    ),
    # v7 conditioned predictors -- teacher-forced base log-prob / token-dist over
    # the per-behavior realization span (i537_bcond_predictors.py writes
    # realization_scores/<b>/bcond_<cid>.json). Directional logprob-diff row +
    # symmetric JS row.
    "behavior_conditioned_logprob_diff": dict(
        tier="registered",
        family="conditioned",
        implemented=True,
        symmetric=False,
        note="PRIMARY conditioned predictor (v7): -(mean_r logP_base(r|ctx_j) - "
        "logP_base(r|ctx_i)); eval-ctx log-likes the realization -> more leak -> less "
        "distant (pinned like base_prior_bystander)",
    ),
    "behavior_conditioned_js": dict(
        tier="registered",
        family="conditioned",
        implemented=True,
        note="symmetric per-token JS over the realization-span positions; carries the "
        "#489/#502 collinearity re-check (H5)",
    ),
    # v7 behavior-vector projection (Persona-Vectors, arXiv 2507.21509 / #623).
    # shift = post-hoc trained Δh projection (reuses activation_deltas/); level =
    # pre-training base projection (reuses clouds/). i537_behavior_vector_predictor.py
    # writes behavior_vector_scores/<b>.json with per-layer projections.
    "behavior_vector_proj_shift": dict(
        tier="registered",
        family="behavior_vector",
        implemented=True,
        symmetric=False,
        note="<Δh_j, v_b>/||v_b|| at the readout slot (post-hoc track, #532); v_b = "
        "mean(pos)-mean(neg) base activations at last-prompt-token, layers {6,14,22,27}",
    ),
    "behavior_vector_proj_level": dict(
        tier="registered",
        family="behavior_vector",
        implemented=True,
        symmetric=False,
        note="<h_base_j, v_b>/||v_b|| at the readout slot (pre-training level track, #532); "
        "scored separately from shift, never pooled",
    ),
    # (v9-NEW) Two registered, scored CONTROL rows -- the noise / trivial-overlap
    # floor every winner must clear. Scored through the EXACT LTCO + leave-family-out
    # + win-matrix pipeline (family=control). Behavior-independent D by construction
    # (--behavior <b> selects only the G axis). Both zero-GPU.
    "null_random_predictor": dict(
        tier="registered",
        family="control",
        implemented=True,
        note="(v9) fixed-seed (537) permutation of centroid_cosine's off-diagonal cell "
        "values -- the argmax-over-noise null; expected oof_r2 ~ 0",
    ),
    "text_overlap_predictor": dict(
        tier="registered",
        family="control",
        implemented=True,
        symmetric=False,
        note="(v9) 1 - Jaccard(char-3-grams(rendered_prompt_i), rendered_prompt_j)) -- the "
        "trivial prompt-string surface-overlap control; base-model-only, no forward pass; "
        "polarity larger = more distant (more overlap -> less distant)",
    ),
    # SKIP rows (cost without expectation -- never scored)
    "kl_judge": dict(tier="skip", family="deprecated", implemented=False),
    "in_context_rate_m3": dict(tier="skip", family="deprecated", implemented=False),
    "first_step_gradient": dict(tier="skip", family="deprecated", implemented=False),
}

# Output dir for the v9 follow-up round's new predictor artifacts (NEVER the
# prereg trees). Read by metric_matrix for the new rows; written by the GPU
# dropped/bcond/behavior-vector scripts.
PBC = EVAL / "predictor-bakeoff-complete"


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


# ── (v9) New-predictor artifact readers + helpers ────────────────────────────


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def _text_overlap_matrix(cids: list[str], *, behavior: str) -> np.ndarray:
    """(v9) 1 - char-3-gram Jaccard between the two contexts' rendered prompts.

    Base-model-only surface-overlap control: D[i, j] = 1 - Jaccard(grams_i, grams_j).
    Polarity larger = more distant (more overlap -> less distant). One fixed probe
    question is rendered through each context's prompt builder (the overlap is a
    context property, not a question property; the fixed probe avoids per-question
    averaging cost while staying deterministic). Behavior-INDEPENDENT for the prompt
    text EXCEPT ICL/binst contexts whose templates carry the behavior; ``behavior``
    threads through the builder so those contexts render their actual prompt.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import (
        build_prompt,
        load_icl_demos,
        load_registry,
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    registry = load_registry(require_sampled=True)
    try:
        demos = load_icl_demos()
    except FileNotFoundError:
        demos = None
    probe = "What is the capital of France?"  # fixed deterministic probe
    grams: dict[str, set[str]] = {}
    for c in cids:
        ctx = registry[c]
        prompt = build_prompt(ctx, probe, tok, behavior=behavior, icl_demos=demos)
        grams[c] = _char_ngrams(prompt, 3)
    n = len(cids)
    d = np.full((n, n), np.nan)
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                continue
            gi, gj = grams[ci], grams[cj]
            union = len(gi | gj)
            jac = (len(gi & gj) / union) if union else 0.0
            d[i, j] = 1.0 - jac  # more overlap -> less distant
    return d


def _pbc_read(rel: str) -> dict:
    """Read a JSON artifact from the v9 predictor-bakeoff-complete tree."""
    p = PBC / rel
    assert p.exists(), (
        f"predictor-bakeoff-complete artifact missing: {p} -- run the GPU pass "
        "(i537_dropped_predictors.py / i537_bcond_predictors.py / "
        "i537_behavior_vector_predictor.py) for this behavior first."
    )
    return json.loads(p.read_text())


def _behavior_vector_matrix(
    metric_id: str, cids: list[str], *, layer: int, behavior: str
) -> np.ndarray:
    """(v9/v7) Behavior-vector projection rows (shift / level / pv_dp).

    Reads ``behavior_vector_scores/<behavior>.json`` written by
    i537_behavior_vector_predictor.py. Schema::

        {"shift": {cid: {str(layer): float}}, "level": {cid: {str(layer): float}},
         "v_b_degenerate": {str(layer): bool}}

    ``shift``  = <Δh_j, v_b>/||v_b|| (post-hoc trained-update projection, per eval ctx).
    ``level``  = <h_base_j, v_b>/||v_b|| (pre-training base projection, per eval ctx).
    Both are COLUMN effects of the eval context j (like base_prior_bystander):
    a higher projection -> the eval ctx aligns with the behavior direction ->
    more leak -> less distant, so polarity = -proj. ``pv_dp`` is routed here to
    the shift track (the v9 dedup; registry redundant_with).
    """
    track = "level" if metric_id == "behavior_vector_proj_level" else "shift"
    payload = _pbc_read(f"behavior_vector_scores/{behavior}.json")
    proj = payload[track]
    deg = payload.get("v_b_degenerate", {}).get(str(layer), False)
    n = len(cids)
    d = np.full((n, n), np.nan)
    if deg:
        logger.warning(
            "[score] behavior_vector %s/%s layer %d v_b DEGENERATE -- row uninformative",
            behavior,
            track,
            layer,
        )
        return d  # all-NaN -> the scorer flags too-few-cells / degenerate
    for j, cj in enumerate(cids):
        assert cj in proj, f"behavior_vector {behavior}/{track} missing cid {cj}"
        val = float(proj[cj][str(layer)])
        d[:, j] = -val  # higher projection -> more leak -> less distant
        d[j, j] = np.nan
    return d


def _bcond_matrix(metric_id: str, cids: list[str], *, behavior: str) -> np.ndarray:
    """(v9/v7) Conditioned predictors over the per-behavior realization span.

    Reads ``realization_scores/<behavior>/bcond_<cid>.json`` (one per CONTEXT),
    each carrying the per-context teacher-forced realization log-prob and the
    sparse next-token distribution over the realization span. Schema per file::

        {"cid": str, "logp_mean": float,
         "span_dist": {"positions": [{"topk_ids": [...], "topk_logp": [...],
                                      "tail_mass": float}, ...]}}

    ``behavior_conditioned_logprob_diff``: D[i,j] = -(logp_j - logp_i) where
    logp_c = mean_r logP_base(r | ctx_c) (directional; eval-ctx licenses the
    realization -> more leak -> less distant). ``behavior_conditioned_js``:
    symmetric mean per-token JS between the two contexts' span distributions.
    """
    files = {c: _pbc_read(f"realization_scores/{behavior}/bcond_{c}.json") for c in cids}
    n = len(cids)
    d = np.full((n, n), np.nan)
    if metric_id == "behavior_conditioned_logprob_diff":
        logp = {c: float(files[c]["logp_mean"]) for c in cids}
        for i, ci in enumerate(cids):
            for j, cj in enumerate(cids):
                if i != j:
                    d[i, j] = -(logp[cj] - logp[ci])  # eval licenses -> less distant
        return d
    # behavior_conditioned_js: symmetric span-distribution JS
    dist = {c: files[c]["span_dist"]["positions"] for c in cids}
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i < j:
                v = _mean_span_js(dist[ci], dist[cj])
                d[i, j] = d[j, i] = v
    return d


def _train_prior_matrix(metric_id: str, cids: list[str], *, behavior: str) -> np.ndarray:
    """(v9) A2 training-completion prior rows.

    Reads ``realization_scores/<behavior>/a2_<cid>.json`` (one per CONTEXT),
    schema ``{"cid": str, "tf_logp_mean": float, "onpolicy_logp_mean": float}``.
    D[i,j] = -(prior_j - prior_i): how well the eval context licenses the cell's
    TRAINING completion (train-side prior, vs the eval-side base_prior_bystander).
    """
    key = "tf_logp_mean" if metric_id == "train_prior_tf" else "onpolicy_logp_mean"
    files = {c: _pbc_read(f"realization_scores/{behavior}/a2_{c}.json") for c in cids}
    prior = {c: float(files[c][key]) for c in cids}
    n = len(cids)
    d = np.full((n, n), np.nan)
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i != j:
                d[i, j] = -(prior[cj] - prior[ci])
    return d


def _sparse_to_dense_logp(pos: dict, vocab: int = 152064) -> np.ndarray:
    """Reconstruct a (near-)full log-prob vector from a top-k + tail-mass record.

    The tail mass is spread UNIFORMLY over the unlisted vocab entries (the standard
    sparse-divergence reconstruction); top-k entries keep their stored log-probs.
    Returns a length-``vocab`` log-prob array (normalized).
    """
    ids = np.asarray(pos["topk_ids"], dtype=np.int64)
    lp = np.asarray(pos["topk_logp"], dtype=np.float64)
    tail = float(pos.get("tail_mass", 0.0))
    k = len(ids)
    out = np.full(vocab, -np.inf)
    out[ids] = lp
    n_tail = vocab - k
    if tail > 1e-12 and n_tail > 0:
        out[out == -np.inf] = np.log(tail) - np.log(n_tail)
    # renormalize defensively
    out = out - _logsumexp(out[None, :])[0]
    return out


def _pair_js(lp: np.ndarray, lq: np.ndarray) -> float:
    m = np.logaddexp(lp, lq) - np.log(2)
    return float(0.5 * np.sum(np.exp(lp) * (lp - m)) + 0.5 * np.sum(np.exp(lq) * (lq - m)))


def _pair_kl(lp: np.ndarray, lq: np.ndarray) -> float:
    """KL(P || Q) over a dense log-prob pair."""
    return float(np.sum(np.exp(lp) * (lp - lq)))


def _mean_span_js(pos_i: list[dict], pos_j: list[dict]) -> float:
    """Mean per-position JS over the aligned span positions (min length)."""
    m = min(len(pos_i), len(pos_j))
    if m == 0:
        return float("nan")
    vals = [
        _pair_js(_sparse_to_dense_logp(pos_i[t]), _sparse_to_dense_logp(pos_j[t])) for t in range(m)
    ]
    return float(np.mean(vals))


def _output_dist_matrix(metric_id: str, cids: list[str], *, behavior: str, kind: str) -> np.ndarray:
    """(v9) A5 / A5_rb / A6 output-sequence divergence rows.

    Reads per-context output distributions written by i537_dropped_predictors.py:
      kind="a5":    realization_scores/<behavior>/a5_<cid>.json    (full-reply positions)
      kind="a5_rb": realization_scores/<behavior>/a5_rb_<cid>.json (response-bucketed)
      kind="a6":    realization_scores/<behavior>/a6_<cid>.json    (taught-span positions only)
    Each file: ``{"cid": str, "positions": [{"topk_ids","topk_logp","tail_mass"}, ...]}``.
    The divergence (JS / fwd-KL / rev-KL / asym / oneway) is computed pairwise
    over the aligned positions (min length) and meaned. Directional for the *_fwd/
    rev/asym/oneway rows; symmetric for js_*.
    """
    prefix = {"a5": "a5", "a5_rb": "a5_rb", "a6": "a6"}[kind]
    files = {c: _pbc_read(f"realization_scores/{behavior}/{prefix}_{c}.json") for c in cids}
    dist = {c: files[c]["positions"] for c in cids}
    n = len(cids)
    d = np.full((n, n), np.nan)

    def _mean_div(pos_i: list[dict], pos_j: list[dict], which: str) -> float:
        m = min(len(pos_i), len(pos_j))
        if m == 0:
            return float("nan")
        out = []
        for t in range(m):
            lp = _sparse_to_dense_logp(pos_i[t])
            lq = _sparse_to_dense_logp(pos_j[t])
            if which == "js":
                out.append(_pair_js(lp, lq))
            elif which == "fwd":  # KL(i || j)
                out.append(_pair_kl(lp, lq))
            elif which == "rev":  # KL(j || i)
                out.append(_pair_kl(lq, lp))
            elif which == "asym":  # 0.5*(fwd+rev) but kept as a labeled asym combo
                out.append(0.5 * _pair_kl(lp, lq) + 0.5 * _pair_kl(lq, lp))
            elif which == "oneway":  # one-way forward KL (i || j), full reply
                out.append(_pair_kl(lp, lq))
        return float(np.mean(out))

    sym = metric_id in ("js_out_seq", "js_out_seq_rb", "js_taught_span", "kl_asym_out_seq")
    which = {
        "js_out_seq": "js",
        "js_out_seq_rb": "js",
        "js_taught_span": "js",
        "kl_out_seq_fwd": "fwd",
        "kl_fwd_out_seq_rb": "fwd",
        "kl_out_seq_rev": "rev",
        "kl_rev_out_seq_rb": "rev",
        "kl_asym_out_seq": "asym",
        "kl_out_seq_oneway": "oneway",
    }[metric_id]
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                continue
            if sym and i < j:
                v = _mean_div(dist[ci], dist[cj], which)
                d[i, j] = d[j, i] = v
            elif not sym:
                d[i, j] = _mean_div(dist[ci], dist[cj], which)
    return d


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

    # ── (v9) Control rows (no clouds, behavior-independent D) ─────────────────
    if metric_id == "null_random_predictor":
        # Permute centroid_cosine's off-diagonal cell VALUES with a fixed seed:
        # the argmax-over-noise null. Behavior-independent (centroid_cosine is
        # base-model geometry; the permutation seed is fixed).
        ref = metric_matrix("centroid_cosine", cids, anchor=anchor, layer=layer, centered=centered)
        off = ~np.eye(n, dtype=bool)
        vals = ref[off & np.isfinite(ref)]
        rng = np.random.default_rng(537)
        perm = rng.permutation(vals)
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(ref[i, j]):
                    d[i, j] = perm[k]
                    k += 1
        return d
    if metric_id == "text_overlap_predictor":
        return _text_overlap_matrix(cids, behavior=behavior)

    # ── (v9) New-row artifact branches (read predictor-bakeoff-complete JSONs) ─
    if metric_id in ("behavior_vector_proj_shift", "behavior_vector_proj_level", "pv_dp"):
        return _behavior_vector_matrix(metric_id, cids, layer=layer, behavior=behavior)
    if metric_id in ("behavior_conditioned_logprob_diff", "behavior_conditioned_js"):
        return _bcond_matrix(metric_id, cids, behavior=behavior)
    if metric_id in ("train_prior_tf", "train_prior_onpolicy"):
        return _train_prior_matrix(metric_id, cids, behavior=behavior)
    if metric_id == "js_taught_span":
        return _output_dist_matrix(metric_id, cids, behavior=behavior, kind="a6")
    if metric_id in (
        "js_out_seq",
        "kl_out_seq_fwd",
        "kl_out_seq_rev",
        "kl_asym_out_seq",
        "kl_out_seq_oneway",
    ):
        return _output_dist_matrix(metric_id, cids, behavior=behavior, kind="a5")
    if metric_id in ("js_out_seq_rb", "kl_fwd_out_seq_rb", "kl_rev_out_seq_rb"):
        return _output_dist_matrix(metric_id, cids, behavior=behavior, kind="a5_rb")

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
            if centered:
                # grand_mean above IS the mean of the context means, so the
                # centered midpoint anchor is the ZERO vector by construction
                # and the cosine is 0/0 garbage (round-3 fix; registry note).
                raise SystemExit(
                    "cos_to_trained_midpoint is undefined under --centered "
                    "(anchor = mean of centered means = 0 by construction); "
                    "score this row raw-only."
                )
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


def _r2_from_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-fold R² from already-pooled (y_true, y_pred) OOF predictions."""
    if y_true.size == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _oof_r2(d_mat: np.ndarray, g_mat: np.ndarray) -> float:
    return _r2_from_pooled(*ltco_cv_predictions(d_mat, g_mat))


def score_metric_vs_g(
    d_mat: np.ndarray,
    g_mat: np.ndarray,
    baseline_mat: np.ndarray | None = None,
    *,
    base_prior_mat: np.ndarray | None = None,
    gauss_kl_mat: np.ndarray | None = None,
) -> dict:
    """Spearman + out-of-fold R² (+ ΔR² over the baseline(s)).

    (v9) Two named baselines ship side by side: ``base_prior_mat`` (the kill/sort
    bar) yields ``base_prior_oof_r2`` + ``delta_vs_base_prior_r2``;
    ``gauss_kl_mat`` (the parent's original geometry-relative baseline) yields
    ``gauss_kl_act_oof_r2`` + ``delta_vs_gauss_kl_act_r2``. ``baseline_mat`` is
    the LEGACY positional arg: when passed (and the named ones are not) it is
    treated as the base_prior baseline and ``delta_r2`` == ``delta_vs_base_prior_r2``
    for backward compat.
    """
    mask = np.isfinite(d_mat) & np.isfinite(g_mat) & ~np.eye(d_mat.shape[0], dtype=bool)
    assert mask.sum() >= 10, f"too few usable cells ({mask.sum()})"
    rho = float(spearmanr(d_mat[mask], g_mat[mask]).statistic)
    r2 = _oof_r2(d_mat, g_mat)
    out = {"spearman": rho, "oof_r2": r2, "n_cells": int(mask.sum())}
    # Back-compat: legacy positional baseline_mat == the base_prior baseline.
    if base_prior_mat is None and baseline_mat is not None:
        base_prior_mat = baseline_mat
    if base_prior_mat is not None:
        bp = _oof_r2(base_prior_mat, g_mat)
        out["base_prior_oof_r2"] = bp
        out["delta_vs_base_prior_r2"] = out["oof_r2"] - bp
        # legacy key kept == the base_prior-relative delta (round-3 readers)
        out["delta_r2"] = out["delta_vs_base_prior_r2"]
    if gauss_kl_mat is not None:
        gk = _oof_r2(gauss_kl_mat, g_mat)
        out["gauss_kl_act_oof_r2"] = gk
        out["delta_vs_gauss_kl_act_r2"] = out["oof_r2"] - gk
    return out


def ltco_cv_predictions_leave_family_out(
    d_mat: np.ndarray, g_mat: np.ndarray, families: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(v9) Leave-context-FAMILY-out pooled OOF predictions.

    For each distinct family F: hold out ALL cells whose train-context OR
    eval-context is in F; fit OLS G ~ D on the remaining off-diagonal cells;
    predict the held-out F cells. Pools OOF over folds (NaN cells dropped). A
    fold with < 8 usable in-fold cells is skipped (mirrors ``ltco_cv_predictions``).
    Tests family-level generalization (LTCO can leak family structure via an
    in-fold same-family sibling). ``families[k]`` = the family tag of context k.
    """
    n = d_mat.shape[0]
    assert len(families) == n, (len(families), n)
    y_true, y_pred = [], []
    for fam in sorted(set(families)):
        in_fam = {k for k in range(n) if families[k] == fam}
        keep = [k for k in range(n) if k not in in_fam]
        xs, ys = [], []
        for i in keep:
            for j in keep:
                if i != j and np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                    xs.append(d_mat[i, j])
                    ys.append(g_mat[i, j])
        if len(xs) < 8:
            continue
        coef = np.polyfit(np.array(xs), np.array(ys), deg=1)
        for i in range(n):
            for j in range(n):
                # a held-out cell touches the family on either axis
                if i == j or (i not in in_fam and j not in in_fam):
                    continue
                if np.isfinite(d_mat[i, j]) and np.isfinite(g_mat[i, j]):
                    y_true.append(g_mat[i, j])
                    y_pred.append(float(np.polyval(coef, d_mat[i, j])))
    return np.array(y_true), np.array(y_pred)


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
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="(v9) per-metric scores JSON path (READ-existing-and-merge keys off this "
        "path too); default = the legacy baselines/baseline_scores.json. The v9 round "
        "passes predictor-bakeoff-complete/scoring/per_metric_score.json so the prereg "
        "baselines/ tree is never overwritten.",
    )
    ap.add_argument(
        "--leaderboard",
        type=Path,
        default=None,
        help="(v9) optional per-behavior leaderboard JSON output (sorted by "
        "delta_vs_base_prior_r2, + the overall_best block when --all-registered)",
    )
    ap.add_argument(
        "--descoped",
        default="",
        help="(v9) comma-separated registered metric ids deliberately descoped for "
        "compute -- marks the leaderboard PARTIAL and names them (plan §3/§7/§9)",
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

    # (v9) Family tag per train context (for the leave-context-family-out fold).
    from explore_persona_space.experiments.i537_contexts import load_registry

    _reg = load_registry(require_sampled=True)
    fam_of = [_reg[c].family for c in cids]

    # (v9) Two named baselines: base_prior_bystander (the kill/sort bar) +
    # gauss_kl_act (the parent's original geometry-relative delta). BOTH ship.
    base_prior_mat = metric_matrix(
        "base_prior_bystander", cids, anchor=args.anchor, layer=args.layer, behavior=args.behavior
    )
    gauss_kl_mat = metric_matrix(
        "gauss_kl_act", cids, anchor=args.anchor, layer=args.layer, behavior=args.behavior
    )

    def _score(mid: str, *, anchor: str, layer: int, centered: bool) -> dict:
        d_mat = metric_matrix(
            mid, cids, anchor=anchor, layer=layer, centered=centered, behavior=args.behavior
        )
        # base_prior's OWN row: delta-against-self is 0 by construction (pass None
        # for the base_prior baseline); still ship the gauss_kl-relative delta.
        res = score_metric_vs_g(
            d_mat,
            g_mat,
            base_prior_mat=None if mid == "base_prior_bystander" else base_prior_mat,
            gauss_kl_mat=None if mid == "gauss_kl_act" else gauss_kl_mat,
        )
        if mid == "base_prior_bystander":
            res["base_prior_oof_r2"] = res["oof_r2"]
            res["delta_vs_base_prior_r2"] = 0.0
            res["delta_r2"] = 0.0
        res["bootstrap"] = context_cluster_bootstrap(d_mat, g_mat)
        # (v9) leave-context-family-out OOF skill, reported ALONGSIDE LTCO (H4)
        lf_true, lf_pred = ltco_cv_predictions_leave_family_out(d_mat, g_mat, fam_of)
        res["leave_family_out_oof_r2"] = (
            _r2_from_pooled(lf_true, lf_pred) if lf_true.size else float("nan")
        )
        res["tier"] = METRIC_REGISTRY[mid]["tier"]
        res["family"] = METRIC_REGISTRY[mid]["family"]
        if METRIC_REGISTRY[mid].get("note"):
            res["note"] = METRIC_REGISTRY[mid]["note"]
        if METRIC_REGISTRY[mid].get("redundant_with"):
            res["redundant_with"] = METRIC_REGISTRY[mid]["redundant_with"]
        res["variant"] = {"anchor": anchor, "layer": layer, "centered": centered}
        return res

    results = {}
    for mid in metric_ids:
        res = _score(mid, anchor=args.anchor, layer=args.layer, centered=args.centered)
        results[mid] = res
        logger.info(
            "[score] %s: rho=%.3f oof_R²=%.3f Δvs_base_prior=%.3f",
            mid,
            res["spearman"],
            res["oof_r2"],
            res.get("delta_vs_base_prior_r2", float("nan")),
        )
    if args.all_registered:
        # §6.1 A3 explicitly registers the #509 representative early-layer cell
        # `end_of_system x L02 x cosine x centered` as part of the raw-vs-
        # centered variant axis; score it under a variant-tagged key (variants
        # of ONE row, not a new metric id -- KL-namespacing rule).
        rep = _score("centroid_cosine", anchor="end_of_system", layer=2, centered=True)
        results["centroid_cosine[end_of_system,L02,centered]"] = rep
        logger.info(
            "[score] #509 representative cell: rho=%.3f oof_R²=%.3f",
            rep["spearman"],
            rep["oof_r2"],
        )

    # (v9) Output redirection: READ-existing-and-merge keys off --out (NOT the
    # hardcoded prereg path) so the 5 per-behavior runs accumulate in the new
    # file without re-reading the OLD baselines/baseline_scores.json.
    out = args.out if args.out is not None else EVAL / "baselines/baseline_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else {"schema_version": 2, "scores": {}}
    if existing.get("schema_version") != 2:
        raise SystemExit(
            f"[score] {out} is a schema-v1 scores file (rows keyed by metric id only; the "
            "behavior label was last-writer-wins, so rows from earlier multi-behavior runs "
            "may be mislabeled and CANNOT be migrated safely). Delete it and re-score -- "
            f"scoring is CPU-only over the cached clouds: rm {out}"
        )
    # Rows are keyed `<behavior>:<metric_id>` and carry the run flags INSIDE
    # each row (round-3 fix, concern baseline-scores-behavior-collision): the
    # §6.5 deliverable needs per-behavior rows to coexist in this one glob, a
    # re-run of one behavior overwrites ONLY its own rows, and there is no
    # top-level run metadata left for a later run to relabel the file with.
    for mid, res in results.items():
        existing["scores"][f"{args.behavior}:{mid}"] = {
            **res,
            "behavior": args.behavior,
            "final_test": args.final_test,
        }
    if args.all_registered:
        # Per-behavior, and only written by --all-registered runs -- a later
        # single --metric run no longer resets it to [] (round-3 minor fix).
        existing.setdefault("registered_not_implemented", {})[args.behavior] = not_implemented
    existing.update(
        {
            "schema_version": 2,
            "git_commit": _git_commit(),
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )
    out.write_text(json.dumps(existing, indent=1))
    logger.info("[score] wrote %s (%d %s rows)", out, len(results), args.behavior)

    # (v9) Optional per-behavior leaderboard (sorted by delta_vs_base_prior_r2)
    # + the overall_best block + the partial-completeness flag.
    descoped = [m.strip() for m in args.descoped.split(",") if m.strip()]
    if args.leaderboard is not None and args.all_registered:
        _write_leaderboard(args.leaderboard, existing, args.behavior, descoped)

    if not_implemented:
        # NEVER a silent gap (§6.1 contract, round-2 fix): every implemented
        # row is scored + persisted ABOVE, then the run exits non-zero naming
        # the registered rows still missing -- implement them or descope with
        # an epm:progress note; --allow-missing-registered is the explicit
        # opt-in for intermediate runs.
        msg = (
            f"[score] {len(not_implemented)} REGISTERED §6.1 rows are not wired: "
            + ", ".join(not_implemented)
            + " (see each row's registry note). Scored rows were "
            "persisted; rerun with --allow-missing-registered to tolerate the gap explicitly."
        )
        if args.allow_missing_registered:
            logger.warning(msg)
        else:
            raise SystemExit(msg)
    return 0


# Behavior-equal-weighted overall aggregation (plan §6): refusal EXCLUDED
# (noise-limited, parent finding 1); em INCLUDED but flagged.
_OVERALL_BEHAVIORS = ("marker", "fact", "sycophancy", "em")


def _write_leaderboard(path: Path, scores_file: dict, behavior: str, descoped: list[str]) -> None:
    """(v9) Emit the per-behavior leaderboard + overall_best block + partial flag.

    Reads ALL rows in the accumulated scores file (every behavior that has run
    so far), sorts each behavior's rows by ``delta_vs_base_prior_r2`` desc, and
    computes the single ``overall_best`` by the §6-declared aggregation
    (behavior-equal-weighted mean delta_vs_base_prior_r2 over
    {marker,fact,sycophancy,em}; refusal excluded; tie-break: higher min
    per-behavior delta -> lower gpu cost (n/a here) -> alphabetical id).

    Completeness: ``partial=True`` (Deliverable 1/2 NOT fully satisfied) when
    any implementable registered class was descoped (``descoped`` non-empty) OR
    any registered row is still unimplemented. A descoped row tag alone does NOT
    certify a full/overall conclusion (plan §3/§7/§9).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = scores_file["scores"]
    by_behavior: dict[str, list[dict]] = {}
    for key, row in scores.items():
        b = row.get("behavior", key.split(":", 1)[0])
        mid = key.split(":", 1)[1] if ":" in key else key
        by_behavior.setdefault(b, []).append({"metric": mid, **row})
    leaderboards: dict[str, list[dict]] = {}
    for b, rows in by_behavior.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r.get("delta_vs_base_prior_r2")
                if r.get("delta_vs_base_prior_r2") is not None
                and np.isfinite(r.get("delta_vs_base_prior_r2"))
                else -1e9
            ),
            reverse=True,
        )
        leaderboards[b] = rows_sorted

    # overall_best: behavior-equal-weighted mean delta over the included behaviors.
    # Gather each predictor's per-behavior delta (skip variant-tagged keys).
    per_metric: dict[str, dict[str, float]] = {}
    for b in _OVERALL_BEHAVIORS:
        for r in by_behavior.get(b, []):
            mid = r["metric"]
            if "[" in mid:  # variant-tagged key, not a base row
                continue
            dv = r.get("delta_vs_base_prior_r2")
            if dv is not None and np.isfinite(dv):
                per_metric.setdefault(mid, {})[b] = float(dv)
    overall_best = None
    if per_metric:
        # only aggregate metrics with a reading on >=1 included behavior
        def _agg(mid: str) -> tuple[float, float, str]:
            vals = [per_metric[mid].get(b) for b in _OVERALL_BEHAVIORS if b in per_metric[mid]]
            mean = float(np.mean(vals))
            mn = float(np.min(vals))
            return mean, mn, mid

        ranked = sorted(
            per_metric.keys(),
            key=lambda m: (_agg(m)[0], _agg(m)[1], m),
            reverse=True,
        )
        # tie-break within 0.01 on the mean: prefer higher min-per-behavior, then id
        best = ranked[0]
        best_mean, best_min, _ = _agg(best)
        tied = [m for m in ranked if abs(_agg(m)[0] - best_mean) < 0.01]
        if len(tied) > 1:
            best = sorted(tied, key=lambda m: (_agg(m)[1], [-ord(c) for c in m]), reverse=True)[0]
            best_mean, best_min, _ = _agg(best)
        overall_best = {
            "metric": best,
            "aggregation": "behavior-equal-weighted mean delta_vs_base_prior_r2 over "
            "{marker,fact,sycophancy,em} (refusal excluded); tie-break min-per-behavior -> id",
            "included_behaviors": [b for b in _OVERALL_BEHAVIORS if b in per_metric[best]],
            "primary_value": best_mean,
            "min_per_behavior_delta": best_min,
            "per_behavior_breakdown": per_metric[best],
        }

    reg_not_impl = scores_file.get("registered_not_implemented", {})
    any_unimpl = any(v for v in reg_not_impl.values())
    partial = bool(descoped) or any_unimpl
    payload = {
        "schema_version": 1,
        "partial": partial,
        "partial_reason": (
            (f"descoped classes: {descoped}; " if descoped else "")
            + (f"unimplemented registered rows: {reg_not_impl}" if any_unimpl else "")
        )
        or "complete leaderboard over the implementable registered set",
        "descoped": descoped,
        "kill_sort_delta": "delta_vs_base_prior_r2",
        "secondary_delta": "delta_vs_gauss_kl_act_r2",
        "leaderboards": leaderboards,
        "overall_best": overall_best,
        "git_commit": _git_commit(),
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=1))
    logger.info(
        "[score] wrote leaderboard %s (partial=%s, overall_best=%s)",
        path,
        partial,
        overall_best["metric"] if overall_best else None,
    )


if __name__ == "__main__":
    sys.exit(main())
