#!/usr/bin/env python3
# (greek + arrow + multiplication/minus-sign characters intentional in docstrings/labels)
"""Issue #541 predictor substrate + headline statistics (extends #500's).

Imports/extends ``issue500_predictors.py`` (plan §4.5): parametrized arms (4,
from ``prior_screen.json``) + 24-panel, plus the #541 additions:

  (a) drop-top-stratum sensitivity (recompute per-arm rho excluding stratum H
      personas, prior > -3.25) next to a drop-one table — every row carries
      the residual n AND the critical |rho| for two-sided alpha=0.05 at that
      n (rho in [0.2, critical) is "positive but underpowered", per the P1
      reporting discipline).
  (b) the P2 block — per-arm panel-median leak over the COMMON 20-persona set
      (panel minus all 4 sources; identical across arms), point + per-seed
      values + seed-level ranges; exact one-sided permutation p over the 4!
      arm-median orderings (the p ~ 0.042 claim obtains ONLY under a perfect
      monotone ordering, and ONLY on the 4-arm GO-full branch — the 3-arm
      GO-descoped branch is directional only, no permutation p); pooled
      per-(arm x seed) Spearman with cluster-on-arm bootstrap labeled
      DESCRIPTIVE (4 clusters cannot support inferential CIs).
  (c) engagement-adjusted partials computed twice — PRIMARY against the
      PRE-TREATMENT base covariates (``base_engagement_covariates.json``),
      SECONDARY against the trained covariates (inherited #500 path inside
      ``_per_arm_metrics``, labeled post-treatment) — plus the pre-registered
      collinearity gate (0.6 / 0.85 on Pearson(prior, base_on_topic)) with the
      tercile-bucket fallback, and the covariate reliability check
      (between-persona spread vs subsample SE).
  (d) rho(prior, leak) on BOTH the raw and the trained-minus-base-adjusted DV
      side by side (divergence => adjusted is claim-bearing).

Also: the floored-arm informativeness rule (>=30% of bystanders above 1%
headline leak — floored arms excluded from P1/P3, included in P2), and
cos_to_home re-resolved to the Phase-0 max-prior panel persona (old home
``local_historian`` kept alongside).

Reads (all produced earlier in the pipeline):
  eval_results/issue_541/phase0_prescreen/prior_screen.json
  eval_results/issue_541/phase0_prescreen/phase0c_persona_vectors.json
  eval_results/issue_541/<arm_slug>/aggregate_cleaned.json
  eval_results/issue_541/baseline_shared/baseline_judged_*.jsonl
  eval_results/issue_541/base_engagement_covariates.json
  eval_results/issue_541/<arm_slug>/engagement_covariates.json   (secondary)

Writes: eval_results/issue_541/predictors.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue500_predictors as i500  # noqa: E402
from aggregate_issue500 import _aggregate_one_judged_file  # noqa: E402

# Smoke namespace via env (set by run_issue541_sweep.sh --smoke) so smoke
# artifacts never mix with the full run's tree.
EVAL_ROOT_NAME = "issue_541_smoke" if os.environ.get("EPM_541_SMOKE") == "1" else "issue_541"
EVAL_ROOT = REPO / "eval_results" / EVAL_ROOT_NAME
PRIOR_SCREEN_PATH = EVAL_ROOT / "phase0_prescreen" / "prior_screen.json"
VECTORS_PATH = EVAL_ROOT / "phase0_prescreen" / "phase0c_persona_vectors.json"
BASE_ENGAGEMENT_PATH = EVAL_ROOT / "base_engagement_covariates.json"
BASELINE_SHARED_DIR = EVAL_ROOT / "baseline_shared"

LAYER_HEADLINE = "21"
STRATUM_H_MIN = -3.25
FLOOR_LEAK_THRESHOLD = 0.01
FLOOR_MIN_FRACTION_ABOVE = 0.30
COLLINEARITY_GATE_SOFT = 0.6
COLLINEARITY_GATE_HARD = 0.85
N_BOOT = 1000
OLD_HOME = "local_historian"


def _critical_spearman(n: int, alpha: float = 0.05) -> float:
    """Two-sided critical |rho| at sample size n via the t approximation."""
    if n < 4:
        return float("nan")
    from scipy import stats

    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 2))
    return t_crit / math.sqrt(n - 2 + t_crit**2)


def _cosine_lookup(vectors: dict[str, Any]) -> dict[tuple[str, str], float]:
    """(a, b) -> mean-per-probe cosine at the headline layer, from the 0c matrix."""
    names = vectors["personas"]
    mat = vectors["cosine_matrix"][LAYER_HEADLINE]
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            out[(a, b)] = float(mat[i][j])
    return out


def _materialize_arm_cosine_files(
    arm_slugs: dict[str, str], panel: tuple[str, ...], cos: dict[tuple[str, str], float]
) -> None:
    """Write per-arm ``persona_distance/results.json`` in the #444/#500 producer
    shape (``cosine.on_topic.<persona>.<layer>``) from the 0c pairwise matrix,
    so the inherited ``_load_cosines`` path runs unmodified."""
    for source, slug in arm_slugs.items():
        out: dict[str, Any] = {
            "_doc": (
                "Materialized from phase0c_persona_vectors.json (pairwise "
                "mean-per-probe cosine matrix) for the inherited #500 loader."
            ),
            "reference_persona": source,
            "cosine": {"on_topic": {}},
        }
        for persona in panel:
            if persona == source or (source, persona) not in cos:
                continue
            out["cosine"]["on_topic"][persona] = {LAYER_HEADLINE: cos[(source, persona)]}
        path = EVAL_ROOT / slug / "persona_distance" / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))


def _baseline_headline_rates(panel: tuple[str, ...]) -> dict[str, float]:
    """Per-persona baseline headline ``stated_seven`` rate from the SHARED
    24-panel baseline (the trained-minus-base adjustment's base term)."""
    cands = sorted(BASELINE_SHARED_DIR.glob("baseline_judged_*.jsonl"))
    if not cands:
        raise RuntimeError(
            f"no baseline_judged_*.jsonl under {BASELINE_SHARED_DIR} — run the shared "
            "baselines (+ auto judge) first."
        )
    agg = _aggregate_one_judged_file(cands[0], panel)
    return {
        persona: float(pdata["leak_rate_headline"]) for persona, pdata in agg["per_persona"].items()
    }


def _per_persona_table(arm_result: dict[str, Any]) -> dict[str, dict[str, float]]:
    return arm_result.get("per_persona", {})


def _aligned(
    arm_result: dict[str, Any], field_y: str = "leak_mean"
) -> tuple[list[str], list[float], list[float]]:
    """(personas, prior, y) aligned vectors from a per-arm result."""
    per_persona = _per_persona_table(arm_result)
    names = [p for p in arm_result.get("panel", []) if p in per_persona]
    prior = [float(per_persona[p]["prior_logprob"]) for p in names]
    y = [float(per_persona[p][field_y]) for p in names]
    return names, prior, y


def _drop_tables(arm_result: dict[str, Any], strata: dict[str, str]) -> dict[str, Any]:
    """(a) drop-one + drop-top-stratum sensitivity rows, with residual n +
    critical rho per row."""
    names, prior, leak = _aligned(arm_result)
    out: dict[str, Any] = {}
    full_rho = i500._spearman(prior, leak) if len(names) >= 2 else float("nan")
    out["full_panel"] = {
        "rho": full_rho,
        "n": len(names),
        "critical_rho": _critical_spearman(len(names)),
    }
    drop_one: dict[str, Any] = {}
    for i, p_name in enumerate(names):
        xs = [v for j, v in enumerate(prior) if j != i]
        ys = [v for j, v in enumerate(leak) if j != i]
        drop_one[p_name] = {
            "rho": i500._spearman(xs, ys) if len(xs) >= 2 else float("nan"),
            "n": len(xs),
            "critical_rho": _critical_spearman(len(xs)),
        }
    out["drop_one"] = drop_one
    keep = [i for i, p_name in enumerate(names) if strata.get(p_name) != "H"]
    xs = [prior[i] for i in keep]
    ys = [leak[i] for i in keep]
    rho = i500._spearman(xs, ys) if len(xs) >= 2 else float("nan")
    crit = _critical_spearman(len(xs))
    band = "n/a"
    if not math.isnan(rho) and not math.isnan(crit):
        if rho >= crit:
            band = "significant_positive"
        elif rho >= 0.2:
            band = "positive_but_underpowered_at_residual_n"
        elif rho <= 0:
            band = "non_positive"
        else:
            band = "weak_positive"
    out["drop_top_stratum"] = {
        "rho": rho,
        "residual_n": len(xs),
        "critical_rho_two_sided_0.05": crit,
        "dropped_personas": [names[i] for i in range(len(names)) if i not in keep],
        "reading": band,
        "_doc": (
            "rho in [0.2, critical) is reported as 'positive but underpowered at "
            "residual n' — neither confirm nor falsify (plan §3 P1 discipline)."
        ),
    }
    return out


def _arm_floored(arm_result: dict[str, Any]) -> dict[str, Any]:
    """Informativeness rule: >=30% of bystanders above 1% headline leak."""
    _names, _prior, leak = _aligned(arm_result)
    n_above = sum(1 for v in leak if v > FLOOR_LEAK_THRESHOLD)
    frac = n_above / max(1, len(leak))
    return {
        "n_bystanders": len(leak),
        "n_above_1pct": n_above,
        "fraction_above_1pct": frac,
        "informative_for_within_arm": bool(frac >= FLOOR_MIN_FRACTION_ABOVE),
        "rule": f">={FLOOR_MIN_FRACTION_ABOVE:.0%} of bystanders with headline leak > "
        f"{FLOOR_LEAK_THRESHOLD:.0%} (floored arms feed P2, not P1/P3)",
    }


def _bootstrap_partial_multi(
    names: list[str],
    x: list[float],
    y: list[float],
    zs: list[list[float]],
    *,
    n_iter: int = N_BOOT,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster(persona)-bootstrap CI for the multi-covariate partial Spearman.

    Each persona contributes one (x, y, z...) tuple (seed-mean), so the
    cluster bootstrap is a resample-personas-with-replacement bootstrap.
    """
    rng = np.random.default_rng(seed)
    n = len(names)
    vals: list[float] = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        bzs = [[z[i] for i in idx] for z in zs]
        try:
            r = i500._partial_spearman_multi(bx, by, bzs)
        except Exception:
            continue
        if not math.isnan(r):
            vals.append(r)
    return i500._summarize_bootstrap(vals)


def _tercile_buckets(names: list[str], by: list[float], leak: list[float]) -> dict[str, Any]:
    """Median leak per tercile of the bucketing variable (collinearity fallback)."""
    order = np.argsort(by)
    n = len(names)
    cuts = [order[: n // 3], order[n // 3 : 2 * n // 3], order[2 * n // 3 :]]
    out: dict[str, Any] = {}
    for label, idx in zip(("low", "mid", "high"), cuts, strict=True):
        out[label] = {
            "personas": [names[i] for i in idx],
            "median_leak": float(np.median([leak[i] for i in idx])) if len(idx) else float("nan"),
            "median_bucket_value": float(np.median([by[i] for i in idx]))
            if len(idx)
            else float("nan"),
        }
    return out


def _primary_engagement_block(
    arm_result: dict[str, Any],
    base_cov: dict[str, Any],
) -> dict[str, Any]:
    """(c) PRIMARY pre-treatment engagement-adjusted partials for one arm."""
    names, prior, leak = _aligned(arm_result)
    lengths: list[float] = []
    on_topic: list[float] = []
    kept: list[int] = []
    for i, p_name in enumerate(names):
        cov = base_cov.get(p_name)
        if not cov:
            continue
        length = float(cov["base_completion_length"])
        topic = float(cov["base_on_topic_fraction"])
        if math.isnan(length) or math.isnan(topic):
            continue
        kept.append(i)
        lengths.append(length)
        on_topic.append(topic)
    names_k = [names[i] for i in kept]
    prior_k = [prior[i] for i in kept]
    leak_k = [leak[i] for i in kept]
    out: dict[str, Any] = {
        "conditioning_set": "PRE-TREATMENT (base_completion_length, base_on_topic_fraction)",
        "n_personas": len(names_k),
    }
    if len(names_k) < 5:
        out["status"] = "insufficient_n_for_partials"
        return out
    out["unadjusted_rho_prior_leak"] = i500._spearman(prior_k, leak_k)
    out["partial_rho_prior_leak_given_base_len_and_on_topic"] = i500._partial_spearman_multi(
        prior_k, leak_k, [lengths, on_topic]
    )
    out["partial_bootstrap"] = _bootstrap_partial_multi(
        names_k, prior_k, leak_k, [lengths, on_topic]
    )
    # Outcome B read: does pre-existing engagement predict leak beyond the prior?
    out["partial_rho_base_on_topic_leak_given_prior"] = i500._partial_spearman_multi(
        on_topic, leak_k, [prior_k]
    )
    out["outcome_b_bootstrap"] = _bootstrap_partial_multi(names_k, on_topic, leak_k, [prior_k])
    out["status"] = "computed"
    return out


def _p2_block(
    arm_order: list[str],
    arm_slugs: dict[str, str],
    arm_results: dict[str, Any],
    source_priors: dict[str, float],
    common_set: list[str],
) -> dict[str, Any]:
    """(b) source-prior -> gating-tightness block over the COMMON persona set."""
    per_arm: dict[str, Any] = {}
    medians: list[float] = []
    pooled_points: list[tuple[float, float]] = []  # (source_prior, per-seed median)
    pooled_clusters: list[str] = []
    for arm in arm_order:
        res = arm_results.get(arm, {})
        per_persona = _per_persona_table(res)
        common = [p for p in common_set if p in per_persona]
        point = (
            float(np.median([per_persona[p]["leak_mean"] for p in common]))
            if common
            else float("nan")
        )
        # Per-seed medians (seed k's median over the common set).
        n_seeds = max((len(per_persona[p].get("leak_seeds", [])) for p in common), default=0)
        seed_medians: list[float] = []
        for k in range(n_seeds):
            vals = [
                per_persona[p]["leak_seeds"][k]
                for p in common
                if len(per_persona[p].get("leak_seeds", [])) > k
            ]
            if vals:
                m = float(np.median(vals))
                seed_medians.append(m)
                pooled_points.append((source_priors[arm], m))
                pooled_clusters.append(arm)
        per_arm[arm] = {
            "arm_slug": arm_slugs.get(arm),
            "source_prior": source_priors.get(arm),
            "n_common_personas": len(common),
            "median_leak_common_set": point,
            "per_seed_medians": seed_medians,
            "seed_range": [min(seed_medians), max(seed_medians)] if seed_medians else None,
        }
        medians.append(point)

    out: dict[str, Any] = {
        "_doc": (
            "P2: arm-level common-set panel-median leak vs measured source prior. "
            "Permutation p is computed ONLY on the 4-arm GO-full branch (exact 4! "
            "test) AND claimed only under a perfect monotone (decreasing) ordering; "
            "the GO-descoped 3-arm branch is directional only — NO permutation p "
            "(plan §7). Pooled per-(arm x seed) Spearman is DESCRIPTIVE (4 clusters "
            "cannot support inferential CIs)."
        ),
        "common_set": common_set,
        "per_arm": per_arm,
        "arm_order_by_source_prior": arm_order,
    }
    finite = [m for m in medians if not math.isnan(m)]
    if len(finite) == len(medians) and len(medians) >= 3:
        perfect = bool(all(medians[i] > medians[i + 1] for i in range(len(medians) - 1)))
        if len(medians) == 4:
            # Exact 4!-permutation test — GO-full branch only. Observed
            # statistic: concordance between source-prior rank (ascending) and
            # -median (gating tightens as prior rises).
            priors_v = [source_priors[a] for a in arm_order]

            def _stat(meds: list[float]) -> float:
                return i500._spearman(priors_v, [-m for m in meds])

            obs = _stat(medians)
            perms = list(itertools.permutations(medians))
            geq = sum(1 for perm in perms if _stat(list(perm)) >= obs - 1e-12)
            out["permutation"] = {
                "observed_spearman_prior_vs_neg_median": obs,
                "one_sided_p": geq / len(perms),
                "n_permutations": len(perms),
                "perfect_monotone_decreasing": perfect,
            }
        else:
            # GO-descoped (3 sources): "P2 drops to a 3-point ordering
            # (directional only, no permutation p claimed)" — plan §7. A 3!
            # test bottoms out at p=1/6 and must NOT be serialized as an
            # inferential read.
            out["permutation"] = {
                "status": "not_claimed_directional_only_descoped",
                "n_arms": len(medians),
                "perfect_monotone_decreasing": perfect,
                "_doc": (
                    "GO-descoped branch: 3-point ordering reported "
                    "directionally; no permutation p is claimed (plan §7)."
                ),
            }
        # Inversion tolerance (all branches): adjacent inversions where seed
        # ranges do NOT overlap.
        hard_inversions = []
        for i in range(len(arm_order) - 1):
            a, b = arm_order[i], arm_order[i + 1]
            ra, rb = per_arm[a]["seed_range"], per_arm[b]["seed_range"]
            inverted = per_arm[b]["median_leak_common_set"] > per_arm[a]["median_leak_common_set"]
            overlap = bool(ra and rb and (ra[0] <= rb[1] and rb[0] <= ra[1]))
            if inverted and not overlap:
                hard_inversions.append([a, b])
        out["hard_inversions_no_seed_range_overlap"] = hard_inversions
    if len(pooled_points) >= 3:
        xs = [p[0] for p in pooled_points]
        ys = [p[1] for p in pooled_points]
        out["pooled_arm_seed_spearman_DESCRIPTIVE"] = {
            "rho": i500._spearman(xs, ys),
            "n_points": len(pooled_points),
            "cluster_on_arm_bootstrap": i500._cluster_bootstrap_spearman(
                pooled_points, pooled_clusters
            ),
        }
    return out


def _collinearity_and_reliability(
    panel: tuple[str, ...],
    priors: dict[str, float],
    base_cov: dict[str, Any],
    arm_results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Panel-level collinearity gate + covariate reliability (arm-invariant —
    base covariates are pre-treatment, computed once over the panel)."""
    gate: dict[str, Any] = {"status": "skipped_no_base_engagement_file"}
    reliability: dict[str, Any] = {"status": "skipped_no_base_engagement_file"}
    if base_cov:
        names_g = [p for p in panel if p in base_cov and p in priors]
        pri = [priors[p] for p in names_g]
        topic = [float(base_cov[p]["base_on_topic_fraction"]) for p in names_g]
        keep = [i for i in range(len(names_g)) if not math.isnan(topic[i])]
        names_g = [names_g[i] for i in keep]
        pri = [pri[i] for i in keep]
        topic = [topic[i] for i in keep]
        if len(names_g) >= 4:
            r = i500._pearson(pri, topic)
            gate = {
                "pearson_prior_vs_base_on_topic": r,
                "abs": abs(r),
                "soft_gate_0.6_tripped": bool(abs(r) > COLLINEARITY_GATE_SOFT),
                "hard_gate_0.85_tripped": bool(abs(r) > COLLINEARITY_GATE_HARD),
                "_doc": (
                    "EXPECTED to trip at 0.6 (the H stratum is designed for "
                    "courthouse affinity); >0.85 => partial unidentifiable, report "
                    "buckets only (plan §4.4)."
                ),
                "n_personas": len(names_g),
            }
            # Bucket fallbacks computed unconditionally (cheap; lead with them
            # when the gate trips).
            leak_panel_mean: dict[str, list[float]] = {}
            for _source, res in arm_results.items():
                if "error" in res:
                    continue
                for p_name, pdata in _per_persona_table(res).items():
                    leak_panel_mean.setdefault(p_name, []).append(float(pdata["leak_mean"]))
            mean_leak = [float(np.mean(leak_panel_mean.get(n, [float("nan")]))) for n in names_g]
            gate["tercile_buckets_by_base_on_topic"] = _tercile_buckets(names_g, topic, mean_leak)
            gate["tercile_buckets_by_prior"] = _tercile_buckets(names_g, pri, mean_leak)
        else:
            # Smoke-scale panels: the file exists but too few personas carry a
            # finite base_on_topic for the Pearson gate — declare the skip
            # (the inherited "skipped_no_base_engagement_file" status would
            # misreport the reason).
            gate = {"status": "skipped_insufficient_n", "n": len(names_g)}
        ses = [
            float(base_cov[p]["on_topic_se"])
            for p in names_g
            if not math.isnan(float(base_cov[p].get("on_topic_se", float("nan"))))
        ]
        if topic and ses:
            sd = float(np.std(topic, ddof=1)) if len(topic) > 1 else float("nan")
            med_se = float(np.median(ses))
            reliability = {
                "between_persona_sd": sd,
                "range": [float(min(topic)), float(max(topic))],
                "median_subsample_se": med_se,
                "sd_over_2x_median_se": bool(sd >= 2 * med_se) if not math.isnan(sd) else None,
                "_doc": (
                    "Pre-condition for any P3 conclusion: if between-persona SD < "
                    "2x median subsample SE the covariate is underpowered and P3 "
                    "is reported as covariate-underpowered, not Outcome A (plan §4.4)."
                ),
            }

    return gate, reliability


def _build_additions(
    arm_results: dict[str, Any],
    strata: dict[str, str],
    base_rates: dict[str, float],
    base_cov: dict[str, Any],
    cos_to_home_new: dict[str, float],
) -> dict[str, Any]:
    """Per-arm #541 additions: floored rule, drop tables, adjusted DV,
    PRIMARY engagement partials, new-home cosine.

    Floored-arm ENFORCEMENT (plan §4.5): the ``floored`` block is always
    emitted, but for an arm that fails the informativeness rule every
    within-arm P1/P3 statistic (drop tables, adjusted-DV rhos, PRIMARY
    engagement partials, new-home cosine rho) is replaced by
    ``status: skipped_floored_arm``. Floored arms remain in P2 — their floor
    IS the gating signal — via ``_p2_block``, which reads ``arm_results``
    directly and is unaffected by this gate.
    """
    additions: dict[str, Any] = {}
    for source, res in arm_results.items():
        if "error" in res:
            additions[source] = {"error": res["error"]}
            continue
        block: dict[str, Any] = {}
        floored = _arm_floored(res)
        block["floored"] = floored
        if not floored["informative_for_within_arm"]:
            skipped = {
                "status": "skipped_floored_arm",
                "_doc": (
                    "Arm fails the informativeness rule (<"
                    f"{FLOOR_MIN_FRACTION_ABOVE:.0%} of bystanders above "
                    f"{FLOOR_LEAK_THRESHOLD:.0%} headline leak) — excluded from "
                    "P1/P3 statistics; INCLUDED in P2 only, where its floor is "
                    "the gating signal (plan §4.5)."
                ),
            }
            block["drop_tables"] = dict(skipped)
            block["adjusted_dv"] = dict(skipped)
            block["primary_engagement"] = dict(skipped)
            additions[source] = block
            continue
        block["drop_tables"] = {"status": "computed", **_drop_tables(res, strata)}
        # (d) raw vs trained-minus-base adjusted DV, side by side. Fail-loud
        # lookup: every arm-panel persona is in the shared-baseline aggregate
        # by construction; a KeyError here means the baseline is broken.
        names, prior_v, leak_raw = _aligned(res)
        leak_adj = [leak_raw[i] - base_rates[names[i]] for i in range(len(names))]
        pairs_adj = [(prior_v[i], leak_adj[i]) for i in range(len(names))]
        block["adjusted_dv"] = {
            "status": "computed",
            "_doc": (
                "leak_adjusted = trained leak_rate_headline - shared-baseline "
                "headline rate per persona. Raw and adjusted reported side by "
                "side; on divergence the ADJUSTED one is claim-bearing (plan §3 P1)."
            ),
            "rho_prior_vs_leak_raw": i500._spearman(prior_v, leak_raw)
            if len(names) >= 2
            else float("nan"),
            "rho_prior_vs_leak_adjusted": i500._spearman(prior_v, leak_adj)
            if len(names) >= 2
            else float("nan"),
            "bootstrap_adjusted_cluster_persona": i500._cluster_bootstrap_spearman(pairs_adj, names)
            if len(names) >= 3
            else None,
            "per_persona_adjusted": dict(zip(names, leak_adj, strict=True)),
            "baseline_headline_rates": {n: base_rates.get(n) for n in names},
        }
        # (c) PRIMARY pre-treatment engagement partials.
        if base_cov:
            block["primary_engagement"] = _primary_engagement_block(res, base_cov)
        else:
            block["primary_engagement"] = {
                "status": "skipped_no_base_engagement_file",
                "expected_path": str(BASE_ENGAGEMENT_PATH),
            }
        # New-home cosine spearman (secondary covariate texture).
        if cos_to_home_new:
            ch = [cos_to_home_new.get(n, float("nan")) for n in names]
            if all(not math.isnan(v) for v in ch) and len(names) >= 2:
                block["rho_cos_to_new_home_vs_leak"] = i500._spearman(ch, leak_raw)
        additions[source] = block
    return additions


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #541 predictors + headline statistics")
    ap.add_argument("--out", default=str(EVAL_ROOT / "predictors.json"))
    args = ap.parse_args()

    screen = json.loads(PRIOR_SCREEN_PATH.read_text())
    sel = screen["selection"]
    panel: tuple[str, ...] = tuple(sel["panel"])
    sources: list[str] = list(sel["sources"])
    arm_slugs: dict[str, str] = dict(sel["arm_slugs"])
    strata: dict[str, str] = dict(sel["strata"])
    priors: dict[str, float] = {k: float(v) for k, v in screen["priors"].items()}

    vectors = json.loads(VECTORS_PATH.read_text()) if VECTORS_PATH.exists() else None
    cos = _cosine_lookup(vectors) if vectors else {}
    if cos:
        _materialize_arm_cosine_files(arm_slugs, panel, cos)

    # cos_to_home: old home (local_historian) feeds the inherited loader slot;
    # new home = max-prior panel persona by the FRESH Phase-0 measurement.
    new_home = max((p for p in panel if p in priors), key=lambda p: priors[p])
    cos_to_home_old = {p: cos.get((OLD_HOME, p), float("nan")) for p in panel} if cos else {}
    cos_to_home_new = {p: cos.get((new_home, p), float("nan")) for p in panel} if cos else {}
    if cos_to_home_old:
        cos_to_home_old[OLD_HOME] = 1.0
    if cos_to_home_new:
        cos_to_home_new[new_home] = 1.0

    # 5-way on-policy priors (convergent validation), unioned across arms.
    arm_aggregate_paths = [
        EVAL_ROOT / slug / "aggregate_cleaned.json" for slug in arm_slugs.values()
    ]
    fiveway_priors, fiveway_src = i500._load_5way_priors_union(arm_aggregate_paths)

    # Inherited per-arm machinery (trained-covariate engagement = SECONDARY).
    arm_results: dict[str, Any] = {}
    for source in sources:
        slug = arm_slugs[source]
        arm_panel = tuple(x for x in panel if x != source)
        try:
            arm_results[source] = i500._per_arm_metrics(
                source,
                slug,
                arm_panel,
                priors,
                fiveway_priors,
                cos_to_home_old,
                eval_root=EVAL_ROOT_NAME,
            )
        except RuntimeError as e:
            arm_results[source] = {"error": str(e), "arm_slug": slug}

    # Baseline headline rates (shared 24-panel) for the adjusted DV.
    base_rates = _baseline_headline_rates(panel)
    base_cov_doc = (
        json.loads(BASE_ENGAGEMENT_PATH.read_text()) if BASE_ENGAGEMENT_PATH.exists() else None
    )
    base_cov = (base_cov_doc or {}).get("per_persona", {})

    additions = _build_additions(arm_results, strata, base_rates, base_cov, cos_to_home_new)

    gate, reliability = _collinearity_and_reliability(panel, priors, base_cov, arm_results)

    # (b) P2 block over the common set (panel minus ALL sources).
    common_set = [p for p in panel if p not in sources]
    arm_order = sorted(sources, key=lambda s: priors.get(s, float("nan")))
    p2 = _p2_block(arm_order, arm_slugs, arm_results, priors, common_set)

    # Strip analyzer-internal points before serializing.
    for source in list(arm_results):
        if isinstance(arm_results[source], dict):
            arm_results[source].pop("_points", None)

    out_full: dict[str, Any] = {
        "panel": list(panel),
        "sources": sources,
        "arm_slugs": arm_slugs,
        "strata": strata,
        "gate_branch": screen["gate"]["branch"],
        "old_home": OLD_HOME,
        "new_home_max_prior": new_home,
        "cos_to_home_old": cos_to_home_old,
        "cos_to_home_new": cos_to_home_new,
        "logprob_priors_used": priors,
        "fiveway_priors_used": fiveway_priors,
        "fiveway_priors_source_arm": fiveway_src,
        "per_arm": arm_results,
        "per_arm_additions": additions,
        "collinearity_gate": gate,
        "covariate_reliability": reliability,
        "p2_source_prior_gating": p2,
        "reproducibility": screen.get("reproducibility"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_full, indent=2, default=str))
    print(f"WROTE {out_path}")
    for source in sources:
        res = arm_results.get(source, {})
        add = additions.get(source, {})
        if "error" in res:
            print(f"  {source:40} ERROR: {res['error']}")
            continue
        s = res.get("stats", {})
        fl = add.get("floored", {})
        print(
            f"  {source:40} n={s.get('n_personas_in_panel', 0):>2} "
            f"rho(prior,leak)={s.get('spearman_prior_logprob_vs_leak', float('nan')):+.3f} "
            f"informative={fl.get('informative_for_within_arm')}"
        )


if __name__ == "__main__":
    main()
