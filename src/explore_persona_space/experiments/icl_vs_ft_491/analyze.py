# ruff: noqa: RUF002
"""Issue #491 off-pod statistics (plan v3 §6) — runs on the VM from eval JSONs.

Headline (H1, REGISTERED): per matched (K, chain) pair, Spearman + slope
between the ICL and FT per-context ΔG profiles over the 9 NON-SOURCE contexts
(the matched source cell agrees by construction and is EXCLUDED; the
10-context version is reported as secondary/descriptive). Uncertainty is a
JOINT question-level bootstrap (one resample drives all pair ρ per
replicate); the confirm rule is noise-aware (raw + disattenuated ρ via
split-half reliability, EIV-corrected slope); the spread gate uses a
question-bootstrap range null (flat-profile null).

``--synthetic-smoke`` exercises the full H1 machinery on synthetic fixtures
with known structure (correlated + uncorrelated pairs) — the CPU smoke for
the stats path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    NON_SOURCE_CONTEXTS,
    PANEL_CONTEXT_IDS,
    SOURCE_CONTEXT,
    ns_eval_dir,
    repro_metadata,
    write_json,
)

logger = logging.getLogger("i491.analyze")

N_BOOT = 10_000
SEED = 42
SPREAD_MIN_NATS = 2.0
RAW_RHO_CONFIRM = 0.6
DISATT_RHO_CONFIRM = 0.8
DISATT_RHO_FALSIFY = 0.4
SLOPE_BAND = (0.7, 1.4)
N_SPLIT_HALF = 200

# The registered H1 denominator (plan §5/§6): exactly the 12 core K x chain
# pairs. The FT row-format control is matched + assembled but NEVER pooled
# into the registered statistic — it is reported separately (round-2 fix,
# h1-control-contamination).
CORE_PAIR_IDS: tuple[str, ...] = tuple(f"ft_K{k}_chain{c}" for k in (1, 3, 8, 16) for c in "ABC")
CONTROL_RUN_IDS: tuple[str, ...] = ("ft_ctrl_helpful_rows",)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho = spearmanr(x, y).statistic
    return float(rho) if np.isfinite(rho) else 0.0


# ── Pair-data assembly from eval JSONs ───────────────────────────────────


def assemble_pairs(smoke: bool = False) -> tuple[dict[str, dict], list[dict]]:
    """({run_id: {"icl": [C, Q], "ft": [C, Q], ...}}, skipped-pair records).

    ICL deltas come from the variant's icl_panel JSON; FT deltas from the
    matched-step full read. Question alignment is asserted. Matched entries
    are read through the race-free per-run accessor (matching.py by_run
    files). Skipped/missing pairs are RETURNED as structured records so
    run_analysis persists them in analysis.json (round-2 minor: never
    console-only).
    """
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_pairs

    ed = ns_eval_dir(smoke)
    matched = load_matched_pairs(smoke=smoke)
    pairs: dict[str, dict] = {}
    skipped: list[dict] = []
    for run_id, entry in matched.items():
        vpath = ed / "icl_panel" / f"{entry['icl_dose_variant']}.json"
        fpath = ed / "ft_panel" / f"{run_id}_full_step{entry['matched_step']}.json"
        if not (vpath.exists() and fpath.exists()):
            missing = [str(p) for p in (vpath, fpath) if not p.exists()]
            logger.warning("pair %s missing inputs %s — skipped", run_id, missing)
            skipped.append({"run_id": run_id, "missing_inputs": missing})
            continue
        icl = json.loads(vpath.read_text())["contexts"]
        ft = json.loads(fpath.read_text())["contexts"]
        contexts = [c for c in PANEL_CONTEXT_IDS if c in icl and c in ft]
        qs = icl[contexts[0]]["questions"]
        for cid in contexts:
            if icl[cid]["questions"] != qs or ft[cid]["questions"] != qs:
                raise AssertionError(f"{run_id}/{cid}: question alignment drift")
        pairs[run_id] = {
            "icl": np.array([icl[c]["delta_logp"] for c in contexts]),
            "ft": np.array([ft[c]["delta_logp"] for c in contexts]),
            "icl_margin": np.array([icl[c]["delta_margin"] for c in contexts]),
            "ft_margin": np.array([ft[c]["delta_margin"] for c in contexts]),
            "contexts": contexts,
            "icl_variant": entry["icl_dose_variant"],
            "matched": entry,
        }
    return pairs, skipped


def _nonsource(pair: dict, key: str) -> np.ndarray:
    idx = [i for i, c in enumerate(pair["contexts"]) if c != SOURCE_CONTEXT]
    return pair[key][idx]


# ── H1 machinery ─────────────────────────────────────────────────────────


def split_half_reliability(
    mat: np.ndarray, *, rng: np.random.Generator, n_splits: int = N_SPLIT_HALF
) -> float:
    """Spearman-Brown-corrected split-half reliability of a [C, Q] profile."""
    n_q = mat.shape[1]
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(n_q)
        a, b = perm[: n_q // 2], perm[n_q // 2 :]
        r = _spearman(mat[:, a].mean(axis=1), mat[:, b].mean(axis=1))
        rs.append(2 * r / (1 + r) if r > -1 else -1.0)
    return float(np.clip(np.mean(rs), 0.0, 1.0))


def range_null_p95(mat: np.ndarray, *, rng: np.random.Generator, n_boot: int = 2000) -> float:
    """95th pct of the flat-profile null range of context means ([C, Q] input).

    Null: every context shares one mean — per replicate each context's mean
    is resampled (with replacement) from the POOLED, per-context-demeaned
    per-question deltas, so only sampling noise differentiates contexts.
    """
    n_c, n_q = mat.shape
    pooled = (mat - mat.mean(axis=1, keepdims=True)).ravel()
    ranges = np.empty(n_boot)
    for i in range(n_boot):
        means = pooled[rng.integers(0, pooled.size, size=(n_c, n_q))].mean(axis=1)
        ranges[i] = means.max() - means.min()
    return float(np.percentile(ranges, 95))


def _eiv_slope(x: np.ndarray, y: np.ndarray, rel_x: float) -> float:
    """OLS slope attenuation-corrected for measurement error in x (lambda = rel_x)."""
    vx = np.var(x)
    if vx <= 0:
        return float("nan")
    ols = float(np.cov(x, y, bias=True)[0, 1] / vx)
    return ols / max(rel_x, 1e-3)


def h1_statistics(pairs: dict[str, dict], *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The registered H1 read over the 9 non-source contexts, CORE pairs only.

    Control runs (``CONTROL_RUN_IDS`` — the ft_ctrl_helpful_rows row-format
    control) are EXCLUDED before any statistic is computed: they never enter
    ``per_pair``, the spread-valid pool, or the pooled/bootstrap reads. The
    excluded ids are recorded in the output; the control is reported
    separately by :func:`format_gap_contrast` (round-2 fix).
    """
    excluded_controls = sorted(set(pairs) & set(CONTROL_RUN_IDS))
    pairs = {r: p for r, p in pairs.items() if r not in CONTROL_RUN_IDS}
    rng = np.random.default_rng(seed)
    per_pair: dict[str, dict] = {}
    valid_for_pool: list[str] = []
    for run_id, pair in pairs.items():
        icl, ft = _nonsource(pair, "icl"), _nonsource(pair, "ft")
        icl_means, ft_means = icl.mean(axis=1), ft.mean(axis=1)
        rho_raw = _spearman(icl_means, ft_means)
        rel_icl = split_half_reliability(icl, rng=rng)
        rel_ft = split_half_reliability(ft, rng=rng)
        denom = np.sqrt(max(rel_icl * rel_ft, 1e-6))
        rho_disatt = float(np.clip(rho_raw / denom, -1.0, 1.0))
        spread_icl = float(icl_means.max() - icl_means.min())
        spread_ft = float(ft_means.max() - ft_means.min())
        null_icl = range_null_p95(icl, rng=rng)
        null_ft = range_null_p95(ft, rng=rng)
        spread_valid = (
            spread_icl >= SPREAD_MIN_NATS
            and spread_ft >= SPREAD_MIN_NATS
            and spread_icl > null_icl
            and spread_ft > null_ft
        )
        # Split-question complement check (shared-draw inflation): ICL means
        # from one random half, FT means from the other.
        n_q = icl.shape[1]
        perm = rng.permutation(n_q)
        rho_complement = _spearman(
            icl[:, perm[: n_q // 2]].mean(axis=1), ft[:, perm[n_q // 2 :]].mean(axis=1)
        )
        # Source-included descriptive (secondary).
        rho_10 = _spearman(pair["icl"].mean(axis=1), pair["ft"].mean(axis=1))
        per_pair[run_id] = {
            "rho_raw": rho_raw,
            "rho_disattenuated": rho_disatt,
            "reliability_icl": rel_icl,
            "reliability_ft": rel_ft,
            "slope_eiv": _eiv_slope(icl_means, ft_means, rel_icl),
            "spread_icl_nats": spread_icl,
            "spread_ft_nats": spread_ft,
            "range_null_p95_icl": null_icl,
            "range_null_p95_ft": null_ft,
            "spread_valid": spread_valid,
            "rho_split_question_complement": rho_complement,
            "rho_10ctx_descriptive": rho_10,
            "within_tolerance": bool(pair["matched"]["within_tolerance"]),
        }
        if spread_valid:
            valid_for_pool.append(run_id)

    # Joint question-level bootstrap: ONE resample drives every pair's rho
    # per replicate; the pooled statistic's CI comes from the replicates.
    boot_pooled_raw, boot_pooled_disatt, boot_slope = [], [], []
    if valid_for_pool:
        n_q = _nonsource(pairs[valid_for_pool[0]], "icl").shape[1]
        rel = {
            r: np.sqrt(max(per_pair[r]["reliability_icl"] * per_pair[r]["reliability_ft"], 1e-6))
            for r in valid_for_pool
        }
        for _ in range(n_boot):
            qi = rng.integers(0, n_q, size=n_q)
            rhos_raw, rhos_dis, slopes = [], [], []
            for r in valid_for_pool:
                icl, ft = _nonsource(pairs[r], "icl"), _nonsource(pairs[r], "ft")
                im, fm = icl[:, qi].mean(axis=1), ft[:, qi].mean(axis=1)
                rho = _spearman(im, fm)
                rhos_raw.append(rho)
                rhos_dis.append(np.clip(rho / rel[r], -1.0, 1.0))
                slopes.append(_eiv_slope(im, fm, per_pair[r]["reliability_icl"]))
            boot_pooled_raw.append(np.mean(rhos_raw))
            boot_pooled_disatt.append(np.mean(rhos_dis))
            boot_slope.append(np.nanmean(slopes))

    def _ci(xs: list[float]) -> list[float] | None:
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None

    pooled_raw = (
        float(np.mean([per_pair[r]["rho_raw"] for r in valid_for_pool])) if valid_for_pool else None
    )
    pooled_disatt = (
        float(np.mean([per_pair[r]["rho_disattenuated"] for r in valid_for_pool]))
        if valid_for_pool
        else None
    )
    pooled_slope = (
        float(np.nanmean([per_pair[r]["slope_eiv"] for r in valid_for_pool]))
        if valid_for_pool
        else None
    )
    high_res = [r for r in valid_for_pool if not per_pair[r]["within_tolerance"]]
    pooled_excl_high_res = (
        float(
            np.mean([per_pair[r]["rho_disattenuated"] for r in valid_for_pool if r not in high_res])
        )
        if [r for r in valid_for_pool if r not in high_res]
        else None
    )

    if pooled_disatt is None:
        verdict = "no-valid-pairs (uniform-leak regime — flatness is the finding)"
    elif (
        pooled_disatt >= DISATT_RHO_CONFIRM
        and pooled_raw is not None
        and pooled_raw >= RAW_RHO_CONFIRM
        and pooled_slope is not None
        and SLOPE_BAND[0] <= pooled_slope <= SLOPE_BAND[1]
    ):
        verdict = "equivalence-supported"
    elif pooled_disatt <= DISATT_RHO_FALSIFY:
        verdict = "divergence (subject to spread validity)"
    else:
        verdict = "partial-correspondence (pre-registered graded zone)"

    return {
        "per_pair": per_pair,
        "pooled": {
            "n_pairs_valid": len(valid_for_pool),
            "valid_pairs": valid_for_pool,
            "rho_raw": pooled_raw,
            "rho_raw_ci95": _ci(boot_pooled_raw),
            "rho_disattenuated": pooled_disatt,
            "rho_disattenuated_ci95": _ci(boot_pooled_disatt),
            "slope_eiv": pooled_slope,
            "slope_eiv_ci95": _ci(boot_slope),
            "rho_disattenuated_excl_high_residual": pooled_excl_high_res,
            "high_residual_pairs": high_res,
            "excluded_control_pairs": excluded_controls,
            "core_pair_ids": list(CORE_PAIR_IDS),
            "verdict": verdict,
            "n_boot": n_boot,
            "seed": seed,
        },
    }


def format_gap_contrast(pairs: dict[str, dict]) -> dict:
    """The FT row-format control, reported SEPARATELY from H1 (plan §5 row 12).

    ``ft_ctrl_helpful_rows`` is never pooled into the registered H1 statistic
    (see h1_statistics). This block quantifies the format gap instead:
    (a) the control pair's own ICL-vs-FT profile Spearman over the 9
    non-source contexts (the same per-pair statistic, quarantined), and
    (b) the FT-profile Spearman between the control and the core
    ft_K8_chainA run — how much the row-format change ALONE moves the
    leakage profile at the same K/chain/dose target.
    """
    ctrl = pairs.get("ft_ctrl_helpful_rows")
    if ctrl is None:
        return {"skipped": "ft_ctrl_helpful_rows pair not assembled"}
    out: dict = {
        "ctrl_pair_rho_icl_vs_ft": _spearman(
            _nonsource(ctrl, "icl").mean(axis=1), _nonsource(ctrl, "ft").mean(axis=1)
        ),
        "ctrl_matched": {
            k: ctrl["matched"][k]
            for k in ("matched_step", "basis", "residual", "within_tolerance")
            if k in ctrl["matched"]  # synthetic fixtures carry a stub entry
        },
    }
    core = pairs.get("ft_K8_chainA")
    if core is not None:
        shared = [c for c in ctrl["contexts"] if c != SOURCE_CONTEXT and c in core["contexts"]]
        ci = [ctrl["contexts"].index(c) for c in shared]
        ki = [core["contexts"].index(c) for c in shared]
        out["ft_profile_rho_ctrl_vs_core_K8_chainA"] = _spearman(
            ctrl["ft"][ci].mean(axis=1), core["ft"][ki].mean(axis=1)
        )
        out["n_shared_contexts"] = len(shared)
    return out


# ── H4 / H5 / H6 / negative controls ─────────────────────────────────────


def h4_dose_monotonicity(smoke: bool = False, *, n_boot: int = 2000, seed: int = SEED) -> dict:
    """ICL source-cell dose vs K per chain: monotonicity beyond bootstrap noise.

    Per chain, a JOINT question bootstrap (one resample drives all four
    K-doses — the per-question deltas are aligned across K by construction)
    yields CIs on each K-increment and the fraction of replicates where the
    dose curve is fully monotone; ``monotone_beyond_noise`` = that fraction
    >= 0.95 (the plan §1 "non-monotone beyond bootstrap noise" falsify read).
    The §6 sign test over chains is the all-chains-monotone count (3 chains:
    p = 0.5^3 = 0.125 under a sign-flip null — reported, not binarized).
    """
    ed = ns_eval_dir(smoke)
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for chain in ("A", "B", "C"):
        per_q: list[np.ndarray] = []
        questions_ref: list[str] | None = None
        for k in (1, 3, 8, 16):
            p = ed / "icl_panel" / f"icl_K{k}_chain{chain}.json"
            if not p.exists():
                break
            ctx = json.loads(p.read_text())["contexts"][SOURCE_CONTEXT]
            if questions_ref is None:
                questions_ref = ctx["questions"]
            elif ctx["questions"] != questions_ref:
                raise AssertionError(f"H4 chain {chain}: question alignment drift at K={k}")
            per_q.append(np.asarray(ctx["delta_logp"], dtype=float))
        if len(per_q) != 4:
            continue
        mat = np.vstack(per_q)  # [4 K-values, Q]
        doses = mat.mean(axis=1)
        diffs = np.diff(doses)
        n_q = mat.shape[1]
        boot_incr = np.empty((n_boot, 3))
        for i in range(n_boot):
            qi = rng.integers(0, n_q, size=n_q)
            boot_incr[i] = np.diff(mat[:, qi].mean(axis=1))
        frac_monotone = float((boot_incr > 0).all(axis=1).mean())
        out[chain] = {
            "doses": [float(x) for x in doses],
            "increments": [float(x) for x in diffs],
            "increment_ci95": [
                [
                    float(np.percentile(boot_incr[:, j], 2.5)),
                    float(np.percentile(boot_incr[:, j], 97.5)),
                ]
                for j in range(3)
            ],
            "monotone": bool((diffs > 0).all()),
            "bootstrap_frac_monotone": frac_monotone,
            "monotone_beyond_noise": bool(frac_monotone >= 0.95),
            "concave_increments": bool((np.diff(diffs) < 0).all()),
        }
    n_mono = sum(1 for v in out.values() if v["monotone"])
    n_mono_noise = sum(1 for v in out.values() if v["monotone_beyond_noise"])
    return {
        "per_chain": out,
        "n_chains": len(out),
        "n_chains_monotone": n_mono,
        "n_chains_monotone_beyond_noise": n_mono_noise,
        "sign_test_p_all_chains_monotone": (
            float(0.5 ** len(out)) if out and n_mono == len(out) else None
        ),
        "n_boot": n_boot,
        "seed": seed,
    }


def h5_h6_controls(pairs: dict[str, dict], smoke: bool = False) -> dict:
    """Content controls (H5), order permutations (H6), H1 negative controls."""
    ed = ns_eval_dir(smoke)

    def _ctx_profile(variant: str) -> dict | None:
        p = ed / "icl_panel" / f"{variant}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())["contexts"]

    out: dict = {}
    full = _ctx_profile("icl_K8_chainA")
    if full:
        src_dose = full[SOURCE_CONTEXT]["mean_delta_logp"]
        out["h5"] = {"full_K8_chainA_source_dose": src_dose, "controls": {}}
        for ctrl in ("icl_ctrl_stripped", "icl_ctrl_helpful", "icl_ctrl_helpful_marker"):
            prof = _ctx_profile(ctrl)
            if prof:
                out["h5"]["controls"][ctrl] = prof[SOURCE_CONTEXT]["mean_delta_logp"]
    # H6: order-permutation sd of the source dose vs the K3->K8 increment.
    perm_doses = []
    for i in range(3):
        prof = _ctx_profile(f"icl_perm_{i}")
        if prof:
            perm_doses.append(prof[SOURCE_CONTEXT]["mean_delta_logp"])
    if full and len(perm_doses) == 3:
        k3 = _ctx_profile("icl_K3_chainA")
        incr = (
            full[SOURCE_CONTEXT]["mean_delta_logp"] - k3[SOURCE_CONTEXT]["mean_delta_logp"]
            if k3
            else None
        )
        sd = float(np.std([*perm_doses, full[SOURCE_CONTEXT]["mean_delta_logp"]], ddof=1))
        out["h6"] = {
            "perm_doses": perm_doses,
            "canonical_dose": full[SOURCE_CONTEXT]["mean_delta_logp"],
            "order_sd": sd,
            "k3_to_k8_increment": incr,
            "sd_below_quarter_increment": (bool(sd < 0.25 * abs(incr)) if incr else None),
            "dof_note": "sd from 4 orderings (3 perms + canonical) — 3-dof estimate",
        }
    # H1 negative controls: control profile vs each pair's FT profile,
    # aligned by SHARED context ids (round-2 minor: a length-equality check
    # silently mispairs contexts when either side has a hole).
    neg: dict[str, dict[str, dict]] = {}
    for ctrl in ("icl_ctrl_helpful_marker", "icl_ctrl_stripped"):
        prof = _ctx_profile(ctrl)
        if not prof:
            continue
        per: dict[str, dict] = {}
        for run_id, pair in pairs.items():
            shared = [
                c
                for c in NON_SOURCE_CONTEXTS
                if c in pair["contexts"] and c in prof and "mean_delta_logp" in prof[c]
            ]
            if len(shared) < 3:
                per[run_id] = {"rho": None, "n_contexts": len(shared)}
                continue
            idx = [pair["contexts"].index(c) for c in shared]
            ctrl_means = np.array([prof[c]["mean_delta_logp"] for c in shared])
            per[run_id] = {
                "rho": _spearman(ctrl_means, pair["ft"][idx].mean(axis=1)),
                "n_contexts": len(shared),
            }
        neg[ctrl] = per
    out["h1_negative_controls"] = neg
    return out


def own_policy_validation(smoke: bool = False) -> dict:
    """Fixed-substrate vs own-substrate per-context profile correlations (§4.5)."""
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_pairs

    ed = ns_eval_dir(smoke)
    out: dict[str, dict] = {}
    pairs = load_matched_pairs(smoke=smoke)
    for chain in ("A", "B", "C"):
        for regime, cell, fixed_file in (
            ("icl", f"icl_K8_chain{chain}", ed / "icl_panel" / f"icl_K8_chain{chain}.json"),
            (
                "ft",
                f"ft_K8_chain{chain}",
                ed
                / "ft_panel"
                / (
                    f"ft_K8_chain{chain}_full_step"
                    f"{pairs.get(f'ft_K8_chain{chain}', {}).get('matched_step', 0)}.json"
                ),
            ),
        ):
            own_file = ed / "own_policy" / f"own_{cell}.json"
            if not (own_file.exists() and fixed_file.exists()):
                continue
            own = json.loads(own_file.read_text())["contexts"]
            fixed = json.loads(fixed_file.read_text())["contexts"]
            ctxs = [c for c in PANEL_CONTEXT_IDS if c in own and c in fixed]
            own_means = np.array([float(np.mean(own[c]["delta_logp"])) for c in ctxs])
            fixed_means = np.array([float(np.mean(fixed[c]["delta_logp"])) for c in ctxs])
            out[cell] = {
                "regime": regime,
                "profile_spearman_fixed_vs_own": _spearman(own_means, fixed_means),
                "n_contexts": len(ctxs),
            }
    return out


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman between x and y with covariate z rank-residualized out of both."""
    from scipy.stats import rankdata

    def _resid(v: np.ndarray) -> np.ndarray:
        rv, rz = rankdata(v), rankdata(z)
        beta = np.cov(rv, rz, bias=True)[0, 1] / max(np.var(rz), 1e-9)
        return rv - beta * rz

    return _spearman(_resid(x), _resid(y))


def h3_gate_correlations(pairs: dict[str, dict], smoke: bool = False) -> dict:
    """H3: one fixed base-model similarity gate ranks leakage in BOTH regimes.

    Gate = base-model pos-1 cos(v_c, v_villain) per layer (shift_summary.json,
    activations phase). Leakage per regime = the mean ΔG per context across
    the valid K=8 pairs. Reported source-included AND source-excluded, plus a
    base-prior partial (per-context base log P(marker) from the no-prefix
    baseline as covariate — #532/#563). The within-0.15 conjunct stays
    descriptive (n=10; Fisher-z SE ~0.38).
    """
    from explore_persona_space.experiments.icl_vs_ft_491.activations import ACT_DIR

    summary_path = ACT_DIR / "shift_summary.json"
    if not summary_path.exists():
        return {"skipped": f"{summary_path} missing — run the activations phase first"}
    gate = json.loads(summary_path.read_text())["gate_base_pos1"]
    bpath = ns_eval_dir(smoke) / "icl_panel" / "base_noprefix.json"
    base_ctx = json.loads(bpath.read_text())["contexts"]

    k8 = [r for r in pairs if r.startswith("ft_K8_chain")]
    if not k8:
        return {"skipped": "no K=8 pairs assembled"}
    contexts = pairs[k8[0]]["contexts"]
    icl_means = np.mean([pairs[r]["icl"].mean(axis=1) for r in k8], axis=0)
    ft_means = np.mean([pairs[r]["ft"].mean(axis=1) for r in k8], axis=0)
    base_prior = np.array(
        [float(np.mean([s["logp"] for s in base_ctx[c]["stats"]])) for c in contexts]
    )
    out: dict[str, dict] = {}
    n_layers = len(next(iter(gate.values()))["cosine"])
    nonsrc = [i for i, c in enumerate(contexts) if c != SOURCE_CONTEXT]
    for layer in range(n_layers):
        g = np.array([gate[c]["cosine"][layer] for c in contexts])
        row: dict[str, float] = {}
        for regime, means in (("icl", icl_means), ("ft", ft_means)):
            row[f"rho_{regime}"] = _spearman(g, means)
            row[f"rho_{regime}_source_excluded"] = _spearman(g[nonsrc], means[nonsrc])
            row[f"rho_{regime}_partial_base_prior"] = _partial_spearman(g, means, base_prior)
        row["abs_diff"] = abs(row["rho_icl"] - row["rho_ft"])
        out[str(layer + 1)] = row
    best = max(out, key=lambda k: min(abs(out[k]["rho_icl"]), abs(out[k]["rho_ft"])))
    return {"per_layer": out, "best_joint_layer": best, "n_k8_pairs": len(k8)}


# ── Entry points ─────────────────────────────────────────────────────────


def _h2_geometry_block() -> dict:
    """Pass-through of the H2 geometry summary so analysis.json carries every
    registered H2 null (incl. the round-2 context-label permutation null)."""
    from explore_persona_space.experiments.icl_vs_ft_491.activations import ACT_DIR

    path = ACT_DIR / "shift_summary.json"
    if not path.exists():
        return {"skipped": f"{path} missing — run the activations summarize step first"}
    s = json.loads(path.read_text())
    return {
        "cross_regime_cosine": s.get("cross_regime_cosine"),
        "replicate_ceilings": s.get("replicate_ceilings"),
        "control_direction_nulls": s.get("control_direction_nulls"),
        "context_label_permutation_null": s.get(
            "context_label_permutation_null",
            "MISSING — shift_summary.json predates the round-2 permutation null; re-run summarize",
        ),
    }


def run_analysis(*, smoke: bool = False, n_boot: int = N_BOOT) -> Path:
    pairs, skipped = assemble_pairs(smoke)
    # The registered design has exactly the 12 core pairs + the named format
    # control; anything else in the matched set is a registry drift.
    unexpected = [r for r in pairs if r not in CORE_PAIR_IDS and r not in CONTROL_RUN_IDS]
    if unexpected:
        raise AssertionError(f"matched pairs outside the registered core+control set: {unexpected}")
    analysis = {
        "meta": repro_metadata(),
        "pairs_assembly": {
            "assembled": sorted(pairs),
            "skipped": skipped,
            "n_assembled": len(pairs),
            "n_skipped": len(skipped),
        },
        "h1": h1_statistics(pairs, n_boot=n_boot),
        "format_gap_control": format_gap_contrast(pairs),
        "h2_geometry": _h2_geometry_block(),
        "h3": h3_gate_correlations(pairs, smoke),
        "h4": h4_dose_monotonicity(smoke),
        "h5_h6": h5_h6_controls(pairs, smoke),
        "own_policy_validation": own_policy_validation(smoke),
    }
    out = ns_eval_dir(smoke) / "analysis.json"
    write_json(out, analysis)
    return out


def synthetic_smoke() -> None:
    """CPU smoke of the H1 machinery on fixtures with known structure."""
    rng = np.random.default_rng(0)
    n_c, n_q = 9, 50

    def _pair(correlated: bool) -> dict:
        true_profile = np.linspace(0, 6, n_c)
        icl = true_profile[:, None] + rng.normal(0, 1.0, size=(n_c, n_q))
        ft_profile = true_profile if correlated else rng.permutation(true_profile)
        ft = ft_profile[:, None] + rng.normal(0, 1.0, size=(n_c, n_q))
        return {
            "icl": np.vstack([np.full((1, n_q), 10.0), icl]),  # row 0 = source
            "ft": np.vstack([np.full((1, n_q), 10.0), ft]),
            "icl_margin": np.zeros((n_c + 1, n_q)),
            "ft_margin": np.zeros((n_c + 1, n_q)),
            "contexts": [SOURCE_CONTEXT, *NON_SOURCE_CONTEXTS],
            "icl_variant": "synthetic",
            "matched": {"within_tolerance": True},
        }

    pairs = {
        "corr_a": _pair(True),
        "corr_b": _pair(True),
        "uncorr": _pair(False),
        # The named format control MUST be excluded from every pooled H1
        # statistic even when present and spread-valid (round-2 fix).
        "ft_ctrl_helpful_rows": _pair(True),
    }
    stats = h1_statistics(pairs, n_boot=300, seed=1)
    corr_rho = stats["per_pair"]["corr_a"]["rho_disattenuated"]
    uncorr_rho = stats["per_pair"]["uncorr"]["rho_disattenuated"]
    assert corr_rho > 0.7, f"correlated fixture disattenuated rho={corr_rho} (expected > 0.7)"
    assert uncorr_rho < 0.6, f"uncorrelated fixture rho={uncorr_rho} (expected < 0.6)"
    assert stats["pooled"]["n_pairs_valid"] >= 2, stats["pooled"]
    assert stats["pooled"]["rho_raw_ci95"] is not None
    assert "ft_ctrl_helpful_rows" not in stats["per_pair"], "control leaked into per_pair"
    assert "ft_ctrl_helpful_rows" not in stats["pooled"]["valid_pairs"], (
        "control leaked into the pooled denominator"
    )
    assert stats["pooled"]["excluded_control_pairs"] == ["ft_ctrl_helpful_rows"]
    fmt = format_gap_contrast(pairs)
    assert "ctrl_pair_rho_icl_vs_ft" in fmt, fmt
    flat = {
        "flat": {
            **pairs["corr_a"],
            "icl": np.full((n_c + 1, n_q), 3.0) + rng.normal(0, 0.4, size=(n_c + 1, n_q)),
            "ft": np.full((n_c + 1, n_q), 3.0) + rng.normal(0, 0.4, size=(n_c + 1, n_q)),
        }
    }
    flat_stats = h1_statistics(flat, n_boot=100, seed=2)
    assert not flat_stats["per_pair"]["flat"]["spread_valid"], (
        "flat fixture should FAIL the spread gate"
    )

    # H2 context-label permutation-null machinery (CPU, small dims): contexts
    # get large distinct base states (scaled one-hots) so identity pairing
    # cancels them exactly (observed shift = planted graded rank-1 along d ->
    # |cos| ~ 1), while a label permutation injects huge cross-context base
    # differences orthogonal to d -> null |cos| collapses. Mirrors the real
    # panel, where base context states differ substantially.
    import torch

    from explore_persona_space.experiments.icl_vs_ft_491.activations import (
        context_label_permutation_null,
    )

    p_rng = np.random.default_rng(3)
    n_ctx, n_layers, hidden = 10, 2, 64
    base_states = torch.zeros(n_ctx, n_layers, hidden)
    for c in range(n_ctx):
        base_states[c, :, c] = 50.0  # distinct, large base state per context
    planted_dir = torch.zeros(hidden)
    planted_dir[n_ctx:] = torch.from_numpy(p_rng.normal(size=hidden - n_ctx)).float()
    planted_dir = planted_dir / planted_dir.norm()
    gains = torch.linspace(0.5, 3.0, n_ctx)
    variant_states = base_states + gains[:, None, None] * planted_dir[None, None, :]
    ref_dirs = planted_dir.unsqueeze(0).repeat(n_layers, 1)
    null = context_label_permutation_null(
        variant_states, base_states, ref_dirs, n_perms=200, rng=p_rng
    )
    assert null["abs_cos_observed"][0] > 0.99, null["abs_cos_observed"]
    assert null["abs_cos_p95"][0] < 0.5, null["abs_cos_p95"]
    assert null["abs_cos_p95"][0] < null["abs_cos_observed"][0], null

    print(
        json.dumps(
            {
                "synthetic_smoke": "PASS",
                "corr_rho_disattenuated": corr_rho,
                "uncorr_rho_disattenuated": uncorr_rho,
                "flat_spread_valid": flat_stats["per_pair"]["flat"]["spread_valid"],
                "pooled_verdict": stats["pooled"]["verdict"],
                "control_excluded_from_h1": stats["pooled"]["excluded_control_pairs"],
                "perm_null_observed_abs_cos": null["abs_cos_observed"][0],
                "perm_null_p95_abs_cos": null["abs_cos_p95"][0],
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="read the smoke namespace")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--synthetic-smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    run_analysis(smoke=args.smoke, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
