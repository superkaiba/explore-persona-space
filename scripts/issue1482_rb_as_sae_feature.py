#!/usr/bin/env python3
"""Issue #1482: the 7 persona-vector r_B directions under the SAE-FEATURE
predictor framework.

The full-dictionary predictor battery (`issue1482_predictor_battery_fullwidth`)
characterizes which answer-side SAE features the context->answer map predicts
well, using per-feature covariates. This round asks where the seven r_B trait
directions land on those same axes when each is treated AS IF it were an SAE
feature -- i.e. as a unit direction with a decoder-like write vector.

Two readings of "per-feature R^2", kept strictly apart:

  (A) DENSE per-direction R^2 -- 1 - ||E u||^2 / ||Yc u||^2 on the #1738
      multi-turn holdout (n=9,941, L19), E = Y - pred_<arm>. This is EXACTLY the
      quantity `issue1482_rb7_reads` read 1 computes for the traits, so applying
      it to all 131,072 unit decoder directions yields an APPLES-TO-APPLES
      distribution the traits have a real percentile in. Computed here.

  (B) SAE-ACTIVATION R^2 -- the banked per-feature holdout R^2 of the
      SAE-feature-space map (`issue_1738/sae_twoway/perfeature/sae_*_r2.npy`),
      the battery's own target. A trait direction has NO activation (no encoder,
      no ReLU/top-k gate), so this quantity is UNDEFINED for it. Reported only as
      the feature-side context the traits cannot be placed in.

Covariates, split by definedness for an arbitrary unit direction u:

  DEFINED   proj_var        u^T Cov(Yc) u              (same GEMM recipe as the
                                                        battery's `proj_var`)
            write_norm      ||gamma (*) u||            (footprint `direct_write_norm`)
            footprint_var/skew/kurt
                            central moments over the vocab of W_U (gamma (*) u)
  UNDEFINED activity        firing frequency -- needs an encoder + gate
            consistency     within-answer activation consistency -- same
            enc_norm        ||W_enc[f]|| -- a direction has no encoder row
            judged labels   autointerp axes are per-SAE-feature annotations

Identity gates: the decoder-side recomputation of `proj_var` and of the three
footprint moments + write_norm must reproduce the banked full-width arrays, or
the trait-side numbers derived by the same code are not trustworthy.

Vectorized: every per-direction read is a Gram/covariance quadratic form
(`einsum('df,df->f', G @ U, U)`) chunked over the dictionary -- no per-feature
Python loop, and the (3584, 131072) product never materializes.
0 GPU; all inputs local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
from datetime import UTC, datetime

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402


# Derived from __file__, NOT task_workflow.repo_root(): that resolver branch-guards to the
# MAIN checkout (it refuses sparse/shallow checkouts). In a default sparse worktree (no
# eval_results/ cones) a re-run fails LOUD (FileNotFoundError) rather than silently reading
# main's copies. #2183; precedent: scripts/issue1482_densesae_fullwidth.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402
import issue1482_rb7_reads as RB7  # noqa: E402

TRAITS7 = RB7.TRAITS7
ARMS = ("context", "prefix", "bare")
PRIMARY_ARM = "context"
LAYER = 19
SAE_K = 64
DICT_SIZE = 131_072
CHUNK = 16_384
N_RANDOM = 200  # random-unit null draws (rb7_reads read-1 convention)

FULLWIDTH_MATRIX = "eval_results/issue_1482/predictor_battery/fullwidth_matrix.npz"
FOOTPRINT = "eval_results/issue_1482/footprint_moments/footprint_moments.npz"
RB7_READS = "eval_results/issue_1482/rb7_reads/rb7_reads.json"
OUT_DIR = "eval_results/issue_1482/rb_as_sae_feature"
FIG_DIR = "figures/issue_1482/rb_as_sae_feature"

# covariates a bare unit direction HAS, and the ones it structurally does not.
DEFINED_COVARIATES = ("proj_var", "write_norm", "footprint_var", "footprint_skew", "footprint_kurt")
UNDEFINED_COVARIATES = {
    "activity": (
        "firing frequency over answer tokens. Requires the SAE encoder + the "
        "BatchTopK gate to decide whether the feature is ON for a token; a bare "
        "direction is always 'on' (every state has a projection onto it), so "
        "there is no activation-frequency analogue."
    ),
    "consistency": (
        "within-answer consistency of the feature's activation pattern. Same "
        "reason as activity: defined on a gated activation time series the "
        "direction does not have."
    ),
    "enc_norm": (
        "norm of the feature's encoder row W_enc[f]. An r_B direction is a "
        "diff-of-means write direction with no paired read/encoder vector; "
        "using ||u|| = 1 in its place would be an invented value, not a measurement."
    ),
    "judged_labels": (
        "the five autointerp axes (interpretable / abstraction / content_type / "
        "speaker_property / functional_role) are per-SAE-feature judge annotations "
        "keyed on feat_id; no judged row exists for a trait direction."
    ),
    "sae_activation_r2": (
        "the battery's own target -- held-out R^2 of the SAE-feature-space map "
        "predicting this feature's ANSWER-SIDE ACTIVATION. A trait direction has "
        "no activation to predict; its comparable quantity is the DENSE "
        "per-direction R^2 computed here."
    ),
}


def _log(msg: str) -> None:
    print(f"[rb-as-feature] {msg}", flush=True)


def _git_commit() -> str:
    """Full HEAD sha of the repo root, or 'unavailable-no-git-checkout'."""
    import subprocess

    r = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else "unavailable-no-git-checkout"


# ── direction sets ────────────────────────────────────────────────────────────


def unit_decoder(w_dec: np.ndarray) -> np.ndarray:
    """Column-normalized decoder (the SAE ships unit columns; enforce exactly)."""
    n = np.linalg.norm(w_dec, axis=0, keepdims=True)
    return w_dec / np.maximum(n, 1e-30)


def quadratic_form(G: np.ndarray, U: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """diag(U^T G U) for a (d, n) direction matrix, chunked over directions."""
    out = np.empty(U.shape[1], dtype=np.float64)
    for s in range(0, U.shape[1], chunk):
        u = np.asarray(U[:, s : s + chunk], dtype=np.float64)
        out[s : s + chunk] = np.einsum("df,df->f", G @ u, u)
    return out


# ── footprint recipe, applied to an arbitrary direction set ───────────────────


def footprint_for_directions(U: np.ndarray, w_u: np.ndarray, gamma: np.ndarray) -> dict:
    """`issue1482_footprint_moments` direct pass for a small direction set.

    Mirrors `_run_pass`: scaled = gamma (*) u, logits = W_U @ scaled, then the
    central moments over the vocabulary axis, plus write_norm = ||scaled||.
    """
    import issue1482_footprint_moments as FM

    scaled = (np.asarray(U, dtype=np.float32) * gamma[:, None]).astype(np.float32)
    logits = w_u @ scaled  # (V, n)
    m = FM._block_moments(logits)
    return {
        "footprint_var": m["var"],
        "footprint_skew": m["skew"],
        "footprint_kurt": m["kurt"],
        "write_norm": np.maximum(np.linalg.norm(scaled, axis=0), 1e-30).astype(np.float64),
    }


# ── rank helpers (the battery's rank-design convention) ───────────────────────


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks in [1, n] (scipy-free, the battery's convention)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def percentile_of(value: float, pool: np.ndarray) -> float:
    """Fraction of the pool strictly below `value`, in percent (mid-rank ties)."""
    below = float(np.count_nonzero(pool < value))
    tied = float(np.count_nonzero(pool == value))
    return 100.0 * (below + 0.5 * tied) / float(len(pool))


def interp_rank(value: float, sorted_pool: np.ndarray) -> float:
    """Rank the value would take in the pool (1..n), for out-of-sample scoring."""
    return float(np.searchsorted(sorted_pool, value, side="left")) + 0.5


# ── restricted joint model ────────────────────────────────────────────────────


def fit_rank_ols(design: np.ndarray, target_rank: np.ndarray) -> dict:
    """OLS of rank(target) on the rank design (intercept added), the battery's form."""
    X = np.column_stack([np.ones(len(target_rank)), design])
    coef, *_ = np.linalg.lstsq(X, target_rank, rcond=None)
    pred = X @ coef
    ss_res = float(np.square(target_rank - pred).sum())
    ss_tot = float(np.square(target_rank - target_rank.mean()).sum())
    return {"coef": coef, "r2": 1.0 - ss_res / ss_tot, "pred": pred}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="one chunk, timing only")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── directions ────────────────────────────────────────────────────────────
    from issue1482_sae import BatchTopKSAE

    rb = RB7._rb_matrix()  # (3584, 7) unit columns, r_B row 19
    sae = BatchTopKSAE.load(k=SAE_K, layer=LAYER, device="cpu")
    # fp64 BEFORE normalizing, matching `full_width_projection_variance` exactly --
    # an fp32 intermediate shifts proj_var by ~4e-5 and breaks the identity gate.
    w_dec = unit_decoder(np.asarray(sae.w_dec, dtype=np.float64))
    assert w_dec.shape == (RS.HIDDEN_DIM, DICT_SIZE), w_dec.shape
    _log(f"directions ready: rb {rb.shape}, w_dec {w_dec.shape} ({time.time() - t_start:.0f}s)")

    # ── dense holdout target + per-arm residuals (the rb7_reads read-1 recipe) ─
    y16, ci = RS.load_layer(LAYER)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n_rows = Yc.shape[0]
    G_Y = Yc.T @ Yc
    cov = G_Y / (n_rows - 1)
    _log(f"target Gram done: n={n_rows} ({time.time() - t_start:.0f}s)")

    if args.pilot:
        t0 = time.time()
        quadratic_form(G_Y, w_dec[:, :CHUNK])
        per_chunk = time.time() - t0
        n_chunks = int(np.ceil(DICT_SIZE / CHUNK))
        _log(
            f"PILOT: {per_chunk:.1f}s per {CHUNK}-direction chunk; "
            f"{n_chunks} chunks x 4 quadratic forms => "
            f"~{per_chunk * n_chunks * 4 / 60:.1f} min of GEMM"
        )
        return

    G_E = {}
    for arm in ARMS:
        pred16 = RS.load_pred(arm, LAYER, "ridge", ci)
        E = Y - np.asarray(pred16, dtype=np.float64)
        G_E[arm] = E.T @ E
        del E, pred16
    _log(f"residual Grams done for {len(ARMS)} arms ({time.time() - t_start:.0f}s)")

    # ── (A) dense per-direction R^2: traits AND all decoder directions ────────
    st_feat = quadratic_form(G_Y, w_dec)
    st_rb = quadratic_form(G_Y, rb)
    dense_r2_feat, dense_r2_rb = {}, {}
    for arm in ARMS:
        dense_r2_feat[arm] = 1.0 - quadratic_form(G_E[arm], w_dec) / st_feat
        dense_r2_rb[arm] = 1.0 - quadratic_form(G_E[arm], rb) / st_rb
        _log(
            f"dense per-direction R^2 [{arm}]: features median "
            f"{np.median(dense_r2_feat[arm]):.4f}; traits "
            f"{np.round(dense_r2_rb[arm], 4).tolist()} ({time.time() - t_start:.0f}s)"
        )

    # rb7_reads reproduction gate: our trait R^2 must match the banked read.
    banked = json.loads((PROJECT_ROOT / RB7_READS).read_text())
    repro = {}
    for arm in ARMS:
        b = banked["read1_trait_r2"][arm]["per_trait_r2"]
        d = max(abs(float(b[t]) - float(dense_r2_rb[arm][i])) for i, t in enumerate(TRAITS7))
        repro[arm] = d
        if not (d < 1e-9):
            raise AssertionError(f"rb7_reads reproduction failed for {arm}: max|delta|={d:.3e}")
    _log(f"rb7_reads reproduction gate PASS: max|delta| {max(repro.values()):.2e}")

    # ── DEFINED covariates for the trait directions ───────────────────────────
    proj_var_rb = quadratic_form(cov, rb)
    proj_var_feat = quadratic_form(cov, w_dec)

    w_u, _tok = RS.load_unembedding()
    import issue1482_footprint_moments as FM

    gamma = FM._load_gamma()
    fp_rb = footprint_for_directions(rb, w_u, gamma)
    # identity gate: recompute the same quantities for a decoder slice.
    gate_idx = np.array([0, 1, 2, 12345, 65536, 131071], dtype=np.int64)
    fp_gate = footprint_for_directions(w_dec[:, gate_idx], w_u, gamma)

    banked_fp = np.load(PROJECT_ROOT / FOOTPRINT)
    banked_fw = np.load(PROJECT_ROOT / FULLWIDTH_MATRIX, allow_pickle=True)
    feat_ids_universe = np.asarray(banked_fw["feat_ids"], dtype=np.int64)

    gates = {
        "proj_var_max_abs_delta": float(
            np.max(np.abs(proj_var_feat[feat_ids_universe] - banked_fw["proj_var"]))
        ),
        "footprint_var_max_abs_delta": float(
            np.max(np.abs(fp_gate["footprint_var"] - banked_fp["direct_var"][gate_idx]))
        ),
        "footprint_skew_max_abs_delta": float(
            np.max(np.abs(fp_gate["footprint_skew"] - banked_fp["direct_skew"][gate_idx]))
        ),
        "footprint_kurt_max_abs_delta": float(
            np.max(np.abs(fp_gate["footprint_kurt"] - banked_fp["direct_kurt"][gate_idx]))
        ),
        "write_norm_max_abs_delta": float(
            np.max(np.abs(fp_gate["write_norm"] - banked_fp["direct_write_norm"][gate_idx]))
        ),
    }
    _log("identity gates: " + ", ".join(f"{k}={v:.3e}" for k, v in gates.items()))
    tol = {"proj_var_max_abs_delta": 1e-6}
    for k, v in gates.items():
        if not (v < tol.get(k, 1e-3)):
            raise AssertionError(f"identity gate {k} failed: {v:.3e}")

    # full-width covariate pools (whole dictionary where we recomputed it,
    # the battery's own arrays where they are banked full width).
    feat_cov = {
        "proj_var": proj_var_feat,
        "write_norm": np.asarray(banked_fp["direct_write_norm"], dtype=np.float64),
        "footprint_var": np.asarray(banked_fp["direct_var"], dtype=np.float64),
        "footprint_skew": np.asarray(banked_fp["direct_skew"], dtype=np.float64),
        "footprint_kurt": np.asarray(banked_fp["direct_kurt"], dtype=np.float64),
    }
    rb_cov = {
        "proj_var": proj_var_rb,
        "write_norm": fp_rb["write_norm"],
        "footprint_var": fp_rb["footprint_var"],
        "footprint_skew": fp_rb["footprint_skew"],
        "footprint_kurt": fp_rb["footprint_kurt"],
    }

    # ── (B) the banked SAE-activation R^2, for context only ───────────────────
    sae_act_r2_full = np.asarray(banked_fw["r2_context"], dtype=np.float64)  # universe-ordered

    # ── restricted joint model (defined covariates only) ──────────────────────
    # Fit on the SAME 114,980-feature universe the full battery used, so the
    # restricted model's R^2 is directly comparable to the battery's joint 0.594.
    uni = feat_ids_universe
    design_uni = np.column_stack([rankdata(feat_cov[k][uni]) for k in DEFINED_COVARIATES])
    models = {}
    for name, target in (
        ("dense_direction_r2", dense_r2_feat[PRIMARY_ARM][uni]),
        ("sae_activation_r2", sae_act_r2_full),
    ):
        fit = fit_rank_ols(design_uni, rankdata(target))
        # score the traits: rank each trait covariate into the universe pool.
        sorted_pools = {k: np.sort(feat_cov[k][uni]) for k in DEFINED_COVARIATES}
        rb_design = np.column_stack(
            [
                np.array([interp_rank(float(rb_cov[k][i]), sorted_pools[k]) for i in range(7)])
                for k in DEFINED_COVARIATES
            ]
        )
        rb_pred_rank = np.column_stack([np.ones(7), rb_design]) @ fit["coef"]
        models[name] = {
            "fit": fit,
            "rb_design": rb_design,
            "rb_pred_rank": rb_pred_rank,
            "target": target,
            # every trait pins the TOP of proj_var, so the model's own in-sample
            # ceiling is what an "above prediction" residual has to be read against.
            "pred_rank_pct_ceiling": {
                "max_over_universe": 100.0 * float(fit["pred"].max()) / len(uni),
                "p99_9_over_universe": 100.0 * float(np.percentile(fit["pred"], 99.9)) / len(uni),
                "p99_over_universe": 100.0 * float(np.percentile(fit["pred"], 99)) / len(uni),
            },
        }
        _log(
            f"restricted rank-OLS [{name}]: R^2 = {fit['r2']:.4f}; predicted-rank "
            f"ceiling {models[name]['pred_rank_pct_ceiling']['max_over_universe']:.2f} pct"
        )

    # ── nearest decoder column per trait ──────────────────────────────────────
    w_dec32 = w_dec.astype(np.float32)
    cosines = np.abs(rb.T.astype(np.float32) @ w_dec32)  # (7, 131072)
    near_idx = np.argmax(cosines, axis=1)
    near_cos = cosines[np.arange(7), near_idx].astype(np.float64)

    # ── random-unit-direction null band on EVERY axis ─────────────────────────
    # The right reference for "is a trait direction special?" is not the feature
    # distribution alone but an arbitrary unit direction scored by the same code.
    rng = np.random.default_rng(RB7.SEED)
    rnd = rng.standard_normal((RS.HIDDEN_DIM, N_RANDOM))
    rnd /= np.linalg.norm(rnd, axis=0, keepdims=True)
    rnd_r2 = 1.0 - quadratic_form(G_E[PRIMARY_ARM], rnd) / quadratic_form(G_Y, rnd)
    rnd_cov = {"proj_var": quadratic_form(cov, rnd), **footprint_for_directions(rnd, w_u, gamma)}
    rnd_cos = np.abs(rnd.T.astype(np.float32) @ w_dec32).max(axis=1).astype(np.float64)

    def _band(x: np.ndarray, pool: np.ndarray | None = None) -> dict:
        d = {
            "mean": float(np.mean(x)),
            "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)),
            "n": int(len(x)),
        }
        if pool is not None:
            d["mean_percentile_among_features"] = percentile_of(float(np.mean(x)), pool)
        return d

    random_unit_null = {
        "n": N_RANDOM,
        "seed": RB7.SEED,
        "dense_direction_r2_context": _band(rnd_r2, dense_r2_feat[PRIMARY_ARM]),
        **{f"{k}": _band(rnd_cov[k], feat_cov[k]) for k in DEFINED_COVARIATES},
        "max_abs_cos_vs_decoder": _band(rnd_cos),
    }
    _log(
        "random-unit null: R^2 mean "
        f"{random_unit_null['dense_direction_r2_context']['mean']:.4f}, proj_var mean "
        f"{random_unit_null['proj_var']['mean']:.4f}, max|cos| p95 "
        f"{random_unit_null['max_abs_cos_vs_decoder']['p95']:.4f}"
    )
    del w_dec32, w_u
    # universe membership of the nearest feature (for its activation-space R^2)
    uni_pos = {int(f): i for i, f in enumerate(uni)}

    # ── assemble the JSON ─────────────────────────────────────────────────────
    n_uni = len(uni)
    dense_uni_sorted = np.sort(dense_r2_feat[PRIMARY_ARM][uni])
    per_trait = {}
    for i, t in enumerate(TRAITS7):
        actual_rank = interp_rank(float(dense_r2_rb[PRIMARY_ARM][i]), dense_uni_sorted)
        nf = int(near_idx[i])
        covs = {}
        for k in DEFINED_COVARIATES:
            covs[k] = {
                "value": float(rb_cov[k][i]),
                "percentile_full_dict": percentile_of(float(rb_cov[k][i]), feat_cov[k]),
                "percentile_universe": percentile_of(float(rb_cov[k][i]), feat_cov[k][uni]),
                "feature_median": float(np.median(feat_cov[k])),
            }
        joint = {}
        for name, m in models.items():
            pred_rank = float(m["rb_pred_rank"][i])
            joint[name] = {
                "predicted_rank_pct": 100.0 * pred_rank / n_uni,
                "note": (
                    "actual rank is defined only for dense_direction_r2; the trait "
                    "has no activation-space R^2"
                ),
            }
        joint["dense_direction_r2"]["actual_rank_pct"] = 100.0 * actual_rank / n_uni
        joint["dense_direction_r2"]["actual_minus_predicted_rank_pct"] = (
            100.0 * actual_rank / n_uni - joint["dense_direction_r2"]["predicted_rank_pct"]
        )
        joint["dense_direction_r2"]["predicted_rank_pct_ceiling_max_over_universe"] = models[
            "dense_direction_r2"
        ]["pred_rank_pct_ceiling"]["max_over_universe"]
        per_trait[t] = {
            "dense_direction_r2": {arm: float(dense_r2_rb[arm][i]) for arm in ARMS},
            "dense_direction_r2_percentile_full_dict": {
                arm: percentile_of(float(dense_r2_rb[arm][i]), dense_r2_feat[arm]) for arm in ARMS
            },
            "dense_direction_r2_percentile_universe": {
                arm: percentile_of(float(dense_r2_rb[arm][i]), dense_r2_feat[arm][uni])
                for arm in ARMS
            },
            "defined_covariates": covs,
            "restricted_joint_model": joint,
            "nearest_decoder_column": {
                "feat_id": nf,
                "abs_cos": float(near_cos[i]),
                "dense_direction_r2_context": float(dense_r2_feat[PRIMARY_ARM][nf]),
                "sae_activation_r2_context": (
                    float(sae_act_r2_full[uni_pos[nf]]) if nf in uni_pos else None
                ),
                "in_battery_universe": nf in uni_pos,
                "proj_var": float(feat_cov["proj_var"][nf]),
                "trait_r2_minus_nearest_feature_r2": float(dense_r2_rb[PRIMARY_ARM][i])
                - float(dense_r2_feat[PRIMARY_ARM][nf]),
                "random_unit_max_abs_cos_p95": random_unit_null["max_abs_cos_vs_decoder"]["p95"],
            },
            "banked_variance_share": banked["read1_trait_r2"][PRIMARY_ARM][
                "per_trait_variance_share"
            ][t],
            "banked_equiv_variance_rank": banked["read1_trait_r2"][PRIMARY_ARM][
                "per_trait_equiv_variance_rank"
            ][t],
        }

    doc = {
        "design": {
            "question": (
                "where do the 7 persona-vector r_B directions land on the per-feature "
                "predictor axes when each is treated as if it were an SAE feature?"
            ),
            "corpus": (
                "#1738 MULTI-TURN holdout, n=9,941, layer 19, ridge context/prefix/bare "
                "arms -- the same staged arrays rb7_reads read 1 used"
            ),
            "rb": {
                "traits": list(TRAITS7),
                "source": "data/issue_779/r_b/<trait>.pt, row 19, L2-normalized",
                "n_traits": 7,
            },
            "sae": {
                "repo": "andyrdt/saes-qwen2.5-7b-instruct",
                "revision": "c37e53c4bb07127ad17ab88f28b93d4e87142e59",
                "layer": LAYER,
                "k": SAE_K,
                "dict_size": DICT_SIZE,
                "decoder_columns_unit_norm": True,
            },
            "two_r2_readings": {
                "dense_direction_r2": (
                    "1 - ||E u||^2 / ||Yc u||^2 on the dense L19 holdout, E = Y - pred_<arm>. "
                    "IDENTICAL recipe for a trait direction and a decoder direction, so the "
                    "trait percentiles below are apples-to-apples. Computed here for all "
                    f"{DICT_SIZE} decoder directions."
                ),
                "sae_activation_r2": (
                    "the battery's target: held-out R^2 of the SAE-feature-space map "
                    "predicting each feature's answer-side ACTIVATION "
                    "(eval_results/issue_1738/sae_twoway/perfeature/sae_context_r2.npy). "
                    "UNDEFINED for a trait direction -- see undefined_covariates."
                ),
                "not_comparable": (
                    "dense-space and feature-space R^2 are NOT comparable numbers "
                    "(sae_dense_bridge.json stated deviation 2); they are reported side by "
                    "side as context, never as one distribution."
                ),
            },
            "universe": {
                "battery_universe_n": int(n_uni),
                "full_dictionary_n": DICT_SIZE,
                "note": (
                    "the battery universe is judged AND finite-R^2 AND answer-active; the "
                    "dense per-direction R^2 is finite for every direction, so full-dictionary "
                    "percentiles are also reported"
                ),
            },
            "identity_gates": gates,
            "rb7_reads_reproduction_max_abs_delta": repro,
        },
        "undefined_covariates": UNDEFINED_COVARIATES,
        "per_trait": per_trait,
        "feature_reference": {
            "dense_direction_r2_context": {
                "median_full_dict": float(np.median(dense_r2_feat[PRIMARY_ARM])),
                "p05": float(np.percentile(dense_r2_feat[PRIMARY_ARM], 5)),
                "p95": float(np.percentile(dense_r2_feat[PRIMARY_ARM], 95)),
                "max": float(dense_r2_feat[PRIMARY_ARM].max()),
                "min": float(dense_r2_feat[PRIMARY_ARM].min()),
            },
            "sae_activation_r2_context": {
                "median_universe": float(np.median(sae_act_r2_full)),
                "p95_universe": float(np.percentile(sae_act_r2_full, 95)),
            },
            "defined_covariate_medians": {
                k: float(np.median(feat_cov[k])) for k in DEFINED_COVARIATES
            },
            "random_unit_band_context_dense_r2": banked["read1_trait_r2"][PRIMARY_ARM][
                "random_band"
            ],
        },
        "random_unit_null": random_unit_null,
        "restricted_joint_model": {
            "predictors": list(DEFINED_COVARIATES),
            "form": "OLS of rank(target) on the rank design (the battery's rank-score form)",
            "fitted_on": f"the battery's {n_uni}-feature universe",
            "dense_direction_r2_rank_r2": float(models["dense_direction_r2"]["fit"]["r2"]),
            "sae_activation_r2_rank_r2": float(models["sae_activation_r2"]["fit"]["r2"]),
            "battery_full_joint_r2_reference": 0.5940446433944997,
            "why_not_the_full_joint_model": (
                "the battery's 24-predictor joint model is dominated by ACTIVITY "
                "(spearman 0.742, LOPO Delta R^2 0.333 of the 0.594 total) and also uses "
                "consistency, enc_norm and five judged label axes -- all UNDEFINED for a "
                "bare direction. Scoring a trait through it would require inventing its "
                "dominant input, so the partial model above uses only the covariates a "
                "direction actually has."
            ),
        },
        "caveats": [
            "A trait direction is a unit direction with no encoder and no BatchTopK gate; "
            "an SAE feature's activation is gated and sparse. Only the WRITE side "
            "(decoder direction, vocabulary footprint) is genuinely shared.",
            "Decoder columns are unit-norm (dec_norm in [0.999998, 1.0000015]) and the r_B "
            "columns are L2-normalized here, so write_norm and the footprint moments are on "
            "the same scale by construction.",
            "proj_var and the dense per-direction R^2 are computed on the #1738 MULTI-TURN "
            "holdout. The battery's own corpus_caveat groups proj_var with its single-turn "
            "covariates, but full_width_projection_variance() reads RS.load_layer(19), the "
            "same multi-turn stage array used here -- the identity gate above confirms the "
            "recomputation matches the banked array exactly.",
            "activity / consistency / enc_norm in the battery ARE the single-turn #1482 "
            "corpus, so the restricted model deliberately excludes them for definedness "
            "reasons AND sidesteps that cross-corpus join.",
            "The vocabulary footprint uses the DIRECT logit lens (W_U (gamma (*) u)); the "
            "banked J-lens-routed twin exists but the battery's joint model uses the direct "
            "moments, so those are what is matched here.",
        ],
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "numpy_version": np.__version__,
            "wall_seconds": round(time.time() - t_start, 1),
        },
    }

    out_path = out_dir / "rb_as_feature.json"
    out_path.write_text(json.dumps(doc, indent=1))
    _log(f"wrote {out_path} ({time.time() - t_start:.0f}s)")

    # ── figure ────────────────────────────────────────────────────────────────
    make_figure(
        dense_feat=dense_r2_feat[PRIMARY_ARM],
        dense_rb=dense_r2_rb[PRIMARY_ARM],
        proj_var_feat=feat_cov["proj_var"],
        proj_var_rb=rb_cov["proj_var"],
        sae_act=sae_act_r2_full,
        proj_var_uni=feat_cov["proj_var"][uni],
    )
    _log(f"done ({time.time() - t_start:.0f}s)")


def make_figure(
    *,
    dense_feat: np.ndarray,
    dense_rb: np.ndarray,
    proj_var_feat: np.ndarray,
    proj_var_rb: np.ndarray,
    sae_act: np.ndarray,
    proj_var_uni: np.ndarray,
) -> None:
    """Two panels: the comparable dense-direction view, and the feature-only view."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    colors = paper_palette(7)
    # constrained layout up front: set_paper_style installs it, and a later
    # tight_layout() call raises once a colorbar exists.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")

    ax = axes[0]
    hb = ax.hexbin(
        np.log10(proj_var_feat),
        dense_feat,
        gridsize=90,
        bins="log",
        cmap="Greys",
        mincnt=1,
        linewidths=0,
    )
    fig.colorbar(hb, ax=ax, label="SAE decoder directions per bin (log)")
    for i, t in enumerate(TRAITS7):
        ax.scatter(
            np.log10(proj_var_rb[i]),
            dense_rb[i],
            s=90,
            marker="D",
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=t,
        )
    ax.set_xlabel(r"$\log_{10}$ dense projected variance  $u^{\top}\,\mathrm{Cov}(Y)\,u$")
    ax.set_ylabel(r"dense per-direction $R^2$ (context arm, #1738 holdout)")
    ax.set_title(
        f"Comparable: same $R^2$ recipe for all {len(dense_feat):,} decoder\n"
        "directions and the 7 trait directions",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True, ncol=2)

    ax = axes[1]
    hb2 = ax.hexbin(
        np.log10(proj_var_uni),
        np.clip(sae_act, -0.05, None),
        gridsize=90,
        bins="log",
        cmap="Greys",
        mincnt=1,
        linewidths=0,
    )
    fig.colorbar(hb2, ax=ax, label="SAE features per bin (log)")
    for i in range(len(TRAITS7)):
        ax.axvline(np.log10(proj_var_rb[i]), color=colors[i], linewidth=1.1, alpha=0.85)
    ax.set_xlabel(r"$\log_{10}$ dense projected variance  $u^{\top}\,\mathrm{Cov}(Y)\,u$")
    ax.set_ylabel(r"SAE feature-activation $R^2$ (context arm, clipped at $-0.05$)")
    ax.set_title(
        f"Not comparable: the battery's own target over {len(sae_act):,}\n"
        "features. Trait directions have no activation to predict.",
        fontsize=11,
    )
    ax.text(
        0.03,
        0.95,
        "vertical lines = trait projected variance\n(no y-value exists for a trait)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    savefig_paper(fig, "rb_as_sae_feature", dir=str(PROJECT_ROOT / FIG_DIR))
    plt.close(fig)


if __name__ == "__main__":
    main()
