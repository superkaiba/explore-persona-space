"""Issue #500 follow-up analysis: the prior x proximity INTERACTION / conditional form.

The #500 clean-result combined prior and proximity only ADDITIVELY
(`z(leak) ~ z(prior) + z(cos)`) plus a cross-arm Delta-rho rank stand-in. The
#444 hypothesis was that proximity's effect on leakage is *conditional* on how
content-related the source persona is to the taught fact -- i.e. an interaction.
That interaction term was never fit. This script fits it, on the already-collected
per-persona substrate in predictors.json (no retraining, no pod).

Two forms:
  (1) WITHIN-ARM multiplicative interaction, per arm:
        z(leak) ~ z(prior) + z(cos) + z(prior)*z(cos)
      Asks: within one source condition, does proximity's slope depend on the prior?
  (2) POOLED cross-condition interaction (the real #444 test):
        z(leak) ~ z(prior) + z(cos) + r_c*z(cos) + r_c*z(prior)
      with leak/prior/cos standardized WITHIN each arm (removes the arm leak-scale
      / floor differences) and r_c = centered ordinal content-relatedness
      (marine=-1, local_resident=0, courthouse=+1). The coefficient on r_c*z(cos)
      is the linear trend in the proximity slope across the three ordered arms.
      Positive => proximity story (slope rises as source becomes content-related).

CIs are cluster-on-persona bootstrap (resample personas with replacement), matching
the inference unit the clean-result uses ("would this generalize to a new panel").
n is tiny (14 personas/arm, 42 pooled) -- the point is to check whether an
interaction is even *suggested*, reported with honest CIs, not to declare one.
"""

import json
from pathlib import Path

import numpy as np

PRED = Path("eval_results/issue_500/predictors.json")
OUT = Path("eval_results/issue_500/interaction_check.json")

ARMS = ["marine_biologist", "local_resident", "courthouse_architecture_historian"]
REL_CENTERED = {
    "marine_biologist": -1.0,
    "local_resident": 0.0,
    "courthouse_architecture_historian": 1.0,
}
N_BOOT = 2000
RNG_SEED = 500  # Date.now/Math.random are fine here (plain script); fixed seed for reproducibility


def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def ols(y, X):
    """OLS with intercept column already in X. Returns (beta, r_squared)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def arm_frame(d, arm):
    pp = d["per_arm"][arm]["per_persona"]
    names = list(pp.keys())
    leak = np.array([pp[n]["leak_mean"] for n in names], float)
    prior = np.array([pp[n]["prior_logprob"] for n in names], float)
    cos = np.array([pp[n]["cos_to_source"] for n in names], float)
    return names, leak, prior, cos


# ----- within-arm interaction -----
def within_arm_interaction(leak, prior, cos):
    zl, zp, zc = z(leak), z(prior), z(cos)
    zint = zp * zc
    n = len(zl)
    # additive
    Xa = np.column_stack([np.ones(n), zp, zc])
    ba, r2a = ols(zl, Xa)
    # interaction
    Xi = np.column_stack([np.ones(n), zp, zc, zint])
    bi, r2i = ols(zl, Xi)
    return {
        "additive": {"beta_prior": ba[1], "beta_prox": ba[2], "r_squared": r2a},
        "interaction": {
            "beta_prior": bi[1],
            "beta_prox": bi[2],
            "beta_interaction": bi[3],
            "r_squared": r2i,
        },
        "delta_r2_from_interaction": r2i - r2a,
    }


def boot_within(leak, prior, cos, rng):
    n = len(leak)
    out = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            res = within_arm_interaction(leak[idx], prior[idx], cos[idx])
            out.append(res["interaction"]["beta_interaction"])
        except Exception:
            continue
    out = np.array([v for v in out if np.isfinite(v)])
    return {
        "point": within_arm_interaction(leak, prior, cos)["interaction"]["beta_interaction"],
        "mean": float(out.mean()),
        "ci_low_95": float(np.percentile(out, 2.5)),
        "ci_high_95": float(np.percentile(out, 97.5)),
        "frac_positive": float((out > 0).mean()),
        "n_valid": int(out.size),
    }


# ----- pooled cross-condition interaction -----
def build_pooled(d):
    """Per-row: within-arm-standardized z_leak,z_prior,z_cos + centered relatedness +
    persona name (cluster id) + arm."""
    rows = []
    for arm in ARMS:
        names, leak, prior, cos = arm_frame(d, arm)
        zl, zp, zc = z(leak), z(prior), z(cos)
        rc = REL_CENTERED[arm]
        for i, nm in enumerate(names):
            rows.append(
                {"persona": nm, "arm": arm, "rc": rc, "zl": zl[i], "zp": zp[i], "zc": zc[i]}
            )
    return rows


def pooled_fit(rows):
    zl = np.array([r["zl"] for r in rows])
    zp = np.array([r["zp"] for r in rows])
    zc = np.array([r["zc"] for r in rows])
    rc = np.array([r["rc"] for r in rows])
    n = len(zl)
    # main-effects-only (additive, pooled)
    Xm = np.column_stack([np.ones(n), zp, zc])
    bm, r2m = ols(zl, Xm)
    # full: + rc + rc*zc + rc*zp
    Xf = np.column_stack([np.ones(n), zp, zc, rc, rc * zc, rc * zp])
    bf, r2f = ols(zl, Xf)
    return {
        "additive_pooled": {"beta_prior": bm[1], "beta_prox": bm[2], "r_squared": r2m},
        "interaction_pooled": {
            "beta_prior": bf[1],
            "beta_prox": bf[2],
            "beta_rc": bf[3],
            "beta_rc_x_prox": bf[4],  # <-- the #444 conditional-proximity test
            "beta_rc_x_prior": bf[5],
            "r_squared": r2f,
        },
        "delta_r2_from_interaction": r2f - r2m,
    }


def boot_pooled(d, rng):
    """Cluster-on-persona bootstrap: resample persona identities from the 15-pool;
    for each arm include that persona's row if present (and re-standardize within arm
    over the resampled set, matching the within-arm-z design)."""
    pool = d["panel_pool_15"]
    raw = {arm: arm_frame(d, arm) for arm in ARMS}
    key_betas = {"rc_x_prox": [], "rc_x_prior": [], "prox": [], "prior": []}
    for _ in range(N_BOOT):
        chosen = rng.choice(pool, size=len(pool), replace=True)
        rows = []
        for arm in ARMS:
            names, leak, prior, cos = raw[arm]
            name_to_i = {nm: i for i, nm in enumerate(names)}
            sel = [name_to_i[c] for c in chosen if c in name_to_i]
            if len(sel) < 4:
                rows = None
                break
            sl, sp, sc = leak[sel], prior[sel], cos[sel]
            zl, zp, zc = z(sl), z(sp), z(sc)
            rc = REL_CENTERED[arm]
            for i in range(len(sel)):
                rows.append({"zl": zl[i], "zp": zp[i], "zc": zc[i], "rc": rc})
        if rows is None:
            continue
        try:
            f = pooled_fit(rows)["interaction_pooled"]
            key_betas["rc_x_prox"].append(f["beta_rc_x_prox"])
            key_betas["rc_x_prior"].append(f["beta_rc_x_prior"])
            key_betas["prox"].append(f["beta_prox"])
            key_betas["prior"].append(f["beta_prior"])
        except Exception:
            continue
    summ = {}
    for k, v in key_betas.items():
        a = np.array([x for x in v if np.isfinite(x)])
        summ[k] = {
            "mean": float(a.mean()),
            "ci_low_95": float(np.percentile(a, 2.5)),
            "ci_high_95": float(np.percentile(a, 97.5)),
            "frac_positive": float((a > 0).mean()),
            "n_valid": int(a.size),
        }
    return summ


def main():
    d = json.loads(PRED.read_text())
    rng = np.random.default_rng(RNG_SEED)

    result = {
        "_doc": "Interaction / conditional-form follow-up to #500. Additive combination "
        "was already in predictors.json; this adds the prior x proximity interaction "
        "(within-arm) and the proximity x content-relatedness interaction (pooled, the "
        "#444 conditional-proximity hypothesis). CIs = cluster-on-persona bootstrap.",
        "validation_reproduce_reported_additive": {},
        "within_arm_interaction": {},
        "per_arm_partial_spearman_cos_given_prior": {},
        "pooled_interaction": {},
    }

    # --- validation: reproduce the additive betas reported in predictors.json ---
    for arm in ARMS:
        names, leak, prior, cos = arm_frame(d, arm)
        rep = d["per_arm"][arm]["stats"]["ols_z_leak_on_z_prior_logprob_and_z_cos_to_source"]
        mine = within_arm_interaction(leak, prior, cos)["additive"]
        result["validation_reproduce_reported_additive"][arm] = {
            "reported_beta_prior": rep["beta_x1_prior"],
            "my_beta_prior": mine["beta_prior"],
            "reported_beta_prox": rep["beta_x2_prox"],
            "my_beta_prox": mine["beta_prox"],
            "reported_r2": rep["r_squared"],
            "my_r2": mine["r_squared"],
            "match": bool(
                abs(rep["beta_x1_prior"] - mine["beta_prior"]) < 1e-6
                and abs(rep["beta_x2_prox"] - mine["beta_prox"]) < 1e-6
            ),
        }
        # within-arm interaction + bootstrap
        result["within_arm_interaction"][arm] = within_arm_interaction(leak, prior, cos)
        result["within_arm_interaction"][arm]["bootstrap_beta_interaction"] = boot_within(
            leak, prior, cos, rng
        )
        # carry the partial Spearman already computed (for the trend-across-arms read)
        result["per_arm_partial_spearman_cos_given_prior"][arm] = d["per_arm"][arm]["stats"][
            "partial_spearman_cos_to_source_given_prior"
        ]

    # --- pooled cross-condition interaction (the #444 test) ---
    rows = build_pooled(d)
    result["pooled_interaction"]["point"] = pooled_fit(rows)
    result["pooled_interaction"]["bootstrap"] = boot_pooled(d, rng)

    OUT.write_text(json.dumps(result, indent=2))

    # ---- console summary ----
    print("=" * 78)
    print("VALIDATION (reproduce predictors.json additive betas)")
    for arm in ARMS:
        v = result["validation_reproduce_reported_additive"][arm]
        print(
            f"  {arm:38s} match={v['match']}  "
            f"beta_prior {v['my_beta_prior']:+.3f} (rep {v['reported_beta_prior']:+.3f})  "
            f"beta_prox {v['my_beta_prox']:+.3f} (rep {v['reported_beta_prox']:+.3f})"
        )

    print("=" * 78)
    print("WITHIN-ARM INTERACTION  z(leak) ~ z(prior) + z(cos) + z(prior)*z(cos)")
    for arm in ARMS:
        w = result["within_arm_interaction"][arm]
        b = w["bootstrap_beta_interaction"]
        ps = result["per_arm_partial_spearman_cos_given_prior"][arm]
        print(f"  {arm:38s}")
        print(
            f"      beta_interaction = {w['interaction']['beta_interaction']:+.3f}  "
            f"95% CI [{b['ci_low_95']:+.3f}, {b['ci_high_95']:+.3f}]  "
            f"frac>0={b['frac_positive']:.2f}"
        )
        print(
            f"      ΔR² from adding interaction = {w['delta_r2_from_interaction']:+.3f}  "
            f"(additive R²={w['additive']['r_squared']:.3f} -> interaction R²={w['interaction']['r_squared']:.3f})"
        )
        print(f"      [partial Spearman cos|prior = {ps:+.3f}]")

    print("=" * 78)
    print("POOLED CROSS-CONDITION INTERACTION (the #444 conditional-proximity test)")
    print("  z(leak) ~ z(prior) + z(cos) + rc + rc*z(cos) + rc*z(prior)   [z within arm]")
    pi = result["pooled_interaction"]["point"]["interaction_pooled"]
    bp = result["pooled_interaction"]["bootstrap"]
    print(
        f"  beta_prox            = {pi['beta_prox']:+.3f}   95% CI "
        f"[{bp['prox']['ci_low_95']:+.3f}, {bp['prox']['ci_high_95']:+.3f}]"
    )
    print(
        f"  beta_prior           = {pi['beta_prior']:+.3f}   95% CI "
        f"[{bp['prior']['ci_low_95']:+.3f}, {bp['prior']['ci_high_95']:+.3f}]"
    )
    print(
        f"  beta_rc*proximity    = {pi['beta_rc_x_prox']:+.3f}   95% CI "
        f"[{bp['rc_x_prox']['ci_low_95']:+.3f}, {bp['rc_x_prox']['ci_high_95']:+.3f}]  "
        f"frac>0={bp['rc_x_prox']['frac_positive']:.2f}   <-- KEY"
    )
    print(
        f"  beta_rc*prior        = {pi['beta_rc_x_prior']:+.3f}   95% CI "
        f"[{bp['rc_x_prior']['ci_low_95']:+.3f}, {bp['rc_x_prior']['ci_high_95']:+.3f}]"
    )
    print(
        f"  ΔR² from interaction = {result['pooled_interaction']['point']['delta_r2_from_interaction']:+.3f}"
    )
    print("=" * 78)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
