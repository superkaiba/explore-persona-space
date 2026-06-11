#!/usr/bin/env python3
# ruff: noqa: RUF001  # typographic minus + multiplication sign in figure labels are intentional
"""Task #585 — off-pod stale-vs-corrected comparison + figures (plan v2 section 4.4).

Runs on the VM (CPU only; the pod is already terminated) against the committed
JSONs. Reads:

  * the STALE published table (``eval_results/issue_504/phase0_calibration_v4.json``
    — six reads of one adapter per #549),
  * the CORRECTED table + trajectory + source slot stats from the #585 re-run.

Computes, per plan section 4.4 / section 6:

  * per-fraction corrected ``delta_g_mean`` / ``emission_p`` / ``r_collapsed``;
  * bystander_resolution (recomputed with the v4 picker's exact formula) with a
    cluster bootstrap CI (10k resamples, clustered by persona);
  * held-out mean delta z_marker + EOS-margin curves (rig slot stats); SOURCE
    companion curves (mean +- SD over 10 questions, from the glue pass);
  * the glue-vs-main source delta-G cross-check per fraction;
  * cross-fraction exact-float-identity rates of held-out ``g_logp`` for all 15
    fraction pairs (the section 6 gate-2 distance-gradient rule); a recurrence
    of the #549 stale-serving signature routes the verdict to
    ``residual_stale_serving_infra`` BEFORE any H1/H2 interpretation;
  * H1/H2 verdicts per the section 6 thresholds, including the outcome-4
    routing diagnostics (ceiling distance, source saturation read).

Writes ``comparison_stale_vs_corrected.json`` + figures (hero: corrected curve
overlaid on the stale flat band +-2.2 nats) via the project paper-plots
conventions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i585.compare_and_figures")

# Pre-registered thresholds (plan section 6; the noise scale exists in the
# committed stale artifact — six same-weights pipeline replicates).
RANGE_THRESHOLD_NATS = 2.2  # 3 x the stale table's same-weights replicate SD
RANGE_STRONG_NATS = 3.5  # binary confirmation without monotone structure
H2_TOLERANCE_NATS = 2.0  # frac-0.08 positive-control tolerance
H1_FALSIFIED_BAND_NATS = 2.0  # every value within this of the stale mean
CEILING_PINNED_NATS = 1.0  # outcome-4 routing: min ceiling distance below this
# Outcome-4 leg (b), emission half: "still climbing" needs strictly more than
# one question's worth of emission-share movement (emission_p granularity over
# the 10 source questions is 0.1). Implementer-concretized; raw curve persists.
EMISSION_CLIMB_RANGE = 0.15
# v4 picker constants (compute_bystander_resolution_from_held_out, pinned rig).
FLOOR_DELTA_G_NATS = 0.5
CEILING_LOGP_NATS = math.log(0.9)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def bystander_resolution(held_out: dict) -> tuple[float, int, int]:
    """The v4 picker's exact in-band formula (pinned-rig parity).

    A (persona, q) leaf is in-band when delta_g >= 0.5 AND trained
    ``g_logp`` <= log(0.9) (not pinned at marker-argmax).
    """
    n_in_band = 0
    n_total = 0
    for per_q in held_out.values():
        for leaf in per_q.values():
            n_total += 1
            dg = float(leaf.get("delta_g", float(leaf["g_logp"]) - float(leaf["b_logp"])))
            if dg >= FLOOR_DELTA_G_NATS and float(leaf["g_logp"]) <= CEILING_LOGP_NATS:
                n_in_band += 1
    return (n_in_band / n_total if n_total else 0.0), n_in_band, n_total


def cluster_bootstrap_resolution_ci(held_out: dict, n_boot: int, seed: int) -> tuple[float, float]:
    """Cluster bootstrap CI for bystander_resolution, clustered by persona."""
    import numpy as np

    personas = sorted(held_out.keys())
    # Per-persona (n_in_band, n_total) so each resample is a cheap sum.
    per_persona = []
    for p in personas:
        in_band = 0
        total = 0
        for leaf in held_out[p].values():
            total += 1
            dg = float(leaf.get("delta_g", float(leaf["g_logp"]) - float(leaf["b_logp"])))
            if dg >= FLOOR_DELTA_G_NATS and float(leaf["g_logp"]) <= CEILING_LOGP_NATS:
                in_band += 1
        per_persona.append((in_band, total))
    arr = np.asarray(per_persona, dtype=float)  # (P, 2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(personas), size=(n_boot, len(personas)))
    sampled = arr[idx]  # (n_boot, P, 2)
    rates = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import spearmanr

    rho, _p = spearmanr(xs, ys)
    return float(rho)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _sd(vals: list[float]) -> float:
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0


def route_verdict(
    *,
    stale_signature: bool,
    fix_took: bool,
    h2_pass: bool,
    dg_range: float,
    dg008: float,
    dg100: float,
    monotone_nondecreasing: bool,
    all_within_stale_band: bool,
    ceiling_pinned: bool,
    margin_still_climbing: bool,
    emission_still_climbing: bool,
) -> str:
    """Section 6 verdict routing (encoded; the analyzer owns interpretation)."""
    if stale_signature or not fix_took:
        # Section 6 gate 2 — checked BEFORE any H1/H2 interpretation. A flat
        # ~0.19-0.27 identity rate at every distance reproduces the #549
        # stale-serving signature, and a non-negligible extreme-pair rate means
        # the distinct-id fix cannot be shown to have taken. Either way this is
        # an INFRA outcome (residual stale serving), never a finding: H2 cannot
        # catch it (frac-0.08 reproduces under stale serving too), and routing
        # it to the falsification label is the precise mislabel the plan
        # forbids. All raw rates + diagnostics stay in the JSON.
        return "residual_stale_serving_infra"
    if not h2_pass:
        return "validity_kill_frac008_control_failed"
    if dg_range > RANGE_STRONG_NATS and dg100 > dg008:
        return "h1_confirmed"
    if dg_range > RANGE_THRESHOLD_NATS and dg100 > dg008 and monotone_nondecreasing:
        return "h1_confirmed_monotone"
    if dg_range > RANGE_THRESHOLD_NATS:
        return "h1_marginal_effect_size_only"
    if dg_range <= RANGE_THRESHOLD_NATS and all_within_stale_band:
        # Outcome 4 candidate — the binding routing rule (section 6 item 4):
        # all diagnostics (ceiling distance; source EOS-margin AND emission
        # curves) must be clean before "#549 needs re-examination".
        if ceiling_pinned or margin_still_climbing or emission_still_climbing:
            return "flat_band_explained_by_saturation_route_outcome5"
        return "h1_falsified_candidate_flags_549_for_reexamination"
    return "intermediate_three_spaces_read"


def compute_comparison(
    stale: dict, corrected: dict, trajectory: dict, slot_stats: dict, n_boot: int, boot_seed: int
) -> dict:
    """All plan section 4.4 quantities + section 6 verdict routing."""
    stale_rows = {float(r["ckpt_frac"]): r for r in stale["smoke_table"]}
    corrected_rows = {float(r["ckpt_frac"]): r for r in corrected["smoke_table"]}
    cks = {float(ck["frac"]): ck for ck in trajectory["checkpoints"]}
    glue_by_frac = {float(fr["frac"]): fr for fr in slot_stats["fractions"]}
    fracs = sorted(cks.keys())
    if len(fracs) != 6:
        raise AssertionError(f"expected 6 fractions in the trajectory, got {fracs}")

    stale_vals = [stale_rows[f]["source_dg"] for f in fracs]
    stale_mean = _mean(stale_vals)
    stale_sd = _sd(stale_vals)

    per_fraction: list[dict] = []
    for f in fracs:
        ck = cks[f]
        src = ck["source_self"]
        held_out = ck["held_out"]
        res, n_in_band, n_total = bystander_resolution(held_out)
        ci_lo, ci_hi = cluster_bootstrap_resolution_ci(held_out, n_boot, boot_seed)
        # Cross-check the picker's own published value for the same trajectory.
        picker_res = corrected_rows[f]["bystander_resolution"]
        if abs(res - picker_res) > 1e-9:
            raise AssertionError(
                f"frac {f}: recomputed bystander_resolution {res} != picker's "
                f"{picker_res} — formula drift vs the pinned rig."
            )
        leaves = [leaf for per_q in held_out.values() for leaf in per_q.values()]
        if any(leaf.get("delta_z_marker") is None for leaf in leaves):
            raise AssertionError(
                f"frac {f}: held-out slot stats missing (delta_z_marker is None) — "
                f"was the eval run with KL off? The plan requires KL ON."
            )
        held_out_dz = _mean([float(leaf["delta_z_marker"]) for leaf in leaves])
        held_out_margin = _mean([float(leaf["delta_z_margin"]) for leaf in leaves])
        held_out_dg_mean = _mean([float(leaf["delta_g"]) for leaf in leaves])
        bystander_emission = _mean([1.0 if leaf["argmax_marker"] else 0.0 for leaf in leaves])
        glue = glue_by_frac[f]
        glue_dgs = [rec["delta_g"] for rec in glue["per_question"].values()]
        glue_dz = [rec["delta_z_marker"] for rec in glue["per_question"].values()]
        glue_margin = [rec["delta_eos_margin"] for rec in glue["per_question"].values()]
        delta_g_mean = float(src["delta_g_mean"])
        # Ceiling distance: max attainable delta_g is -b_logp_mean (g_logp -> 0).
        ceiling_distance = (-float(src["b_logp_mean"])) - delta_g_mean
        per_fraction.append(
            {
                "frac": f,
                "stale_source_dg": stale_rows[f]["source_dg"],
                "corrected_source_dg": delta_g_mean,
                "corrected_minus_stale": delta_g_mean - stale_rows[f]["source_dg"],
                "source_emission_p": float(src["emission_p"]),
                "source_r_collapsed": bool(src["r_collapsed"]),
                "source_b_logp_mean": float(src["b_logp_mean"]),
                "ceiling_distance_nats": ceiling_distance,
                "bystander_resolution": res,
                "bystander_resolution_ci95": [ci_lo, ci_hi],
                "bystander_resolution_n": [n_in_band, n_total],
                "stale_bystander_resolution": stale_rows[f]["bystander_resolution"],
                "held_out_collapse_share": float(ck["held_out_collapse_share"]),
                "held_out_mean_delta_g": held_out_dg_mean,
                "held_out_mean_delta_z_marker": held_out_dz,
                "held_out_mean_eos_margin": held_out_margin,
                "bystander_emission_p": bystander_emission,
                "glue_source_delta_g_mean": _mean(glue_dgs),
                "glue_source_delta_g_sd": _sd(glue_dgs),
                "glue_source_delta_z_marker_mean": _mean(glue_dz),
                "glue_source_delta_z_marker_sd": _sd(glue_dz),
                "glue_source_eos_margin_mean": _mean(glue_margin),
                "glue_source_eos_margin_sd": _sd(glue_margin),
                "glue_vs_main_abs_diff": abs(_mean(glue_dgs) - delta_g_mean),
            }
        )

    # ── Cross-fraction exact-float-identity rates (section 6 gate 2). ─────────
    identity_rates: dict[str, float] = {}
    for fa, fb in itertools.combinations(fracs, 2):
        ho_a, ho_b = cks[fa]["held_out"], cks[fb]["held_out"]
        n_same = 0
        n_total = 0
        for persona, per_q in ho_a.items():
            for q, leaf in per_q.items():
                n_total += 1
                if float(leaf["g_logp"]) == float(ho_b[persona][q]["g_logp"]):
                    n_same += 1
        identity_rates[f"{fa:.2f}__{fb:.2f}"] = n_same / n_total if n_total else 0.0
    extreme_pair_rate = identity_rates[f"{fracs[0]:.2f}__{fracs[-1]:.2f}"]
    all_rates = list(identity_rates.values())
    # Stale signature (#549): a flat ~0.19-0.27 identity rate at EVERY distance.
    stale_signature = all(0.10 <= r <= 0.40 for r in all_rates)
    fix_took = (extreme_pair_rate < 0.05) and not stale_signature

    # ── Verdict routing (section 6, encoded; the analyzer owns interpretation). ─
    dgs = [row["corrected_source_dg"] for row in per_fraction]
    dg_range = max(dgs) - min(dgs)
    dg008, dg100 = dgs[0], dgs[-1]
    rho = spearman_rho(fracs, dgs)
    monotone_nondecreasing = all(b >= a for a, b in itertools.pairwise(dgs))
    h2_abs_diff = abs(dg008 - stale_vals[0])
    h2_pass = h2_abs_diff <= H2_TOLERANCE_NATS
    all_within_stale_band = all(abs(dg - stale_mean) <= H1_FALSIFIED_BAND_NATS for dg in dgs)
    min_ceiling_distance = min(row["ceiling_distance_nats"] for row in per_fraction)
    ceiling_pinned = min_ceiling_distance < CEILING_PINNED_NATS
    # Source saturation read: log-prob capped while the EOS margin still climbs.
    # NOTE: RANGE_THRESHOLD_NATS (2.2 = 3x the stale table's log-prob replicate
    # SD) is reused here for a LOGIT-unit margin range — an implementer-
    # concretized constant (the plan pins no logit-unit threshold); the raw
    # margin curve is persisted + plotted so the analyzer can override.
    glue_margins = [row["glue_source_eos_margin_mean"] for row in per_fraction]
    margin_range = max(glue_margins) - min(glue_margins)
    margin_still_climbing = (
        margin_range > RANGE_THRESHOLD_NATS and glue_margins[-1] > glue_margins[0]
    )
    # Outcome-4 leg (b), emission half (section 6 item 4(b) names BOTH source
    # curves: emission_p AND the EOS margin): an emission share still climbing
    # across fractions while the log-prob band is flat is a saturation
    # explanation. A curve PINNED at 1.0 everywhere is deliberately NOT treated
    # as one — the source saturates emission by design (it IS the implant), so
    # that encoding would make the falsification branch unreachable.
    emissions = [row["source_emission_p"] for row in per_fraction]
    emission_range = max(emissions) - min(emissions)
    emission_still_climbing = emission_range > EMISSION_CLIMB_RANGE and emissions[-1] > emissions[0]

    verdict = route_verdict(
        stale_signature=stale_signature,
        fix_took=fix_took,
        h2_pass=h2_pass,
        dg_range=dg_range,
        dg008=dg008,
        dg100=dg100,
        monotone_nondecreasing=monotone_nondecreasing,
        all_within_stale_band=all_within_stale_band,
        ceiling_pinned=ceiling_pinned,
        margin_still_climbing=margin_still_climbing,
        emission_still_climbing=emission_still_climbing,
    )

    return {
        "schema_version": "i585_comparison_v1",
        "stale_table": {
            "values_nats": stale_vals,
            "mean": stale_mean,
            "same_weights_replicate_sd": stale_sd,
            "range_threshold_nats": RANGE_THRESHOLD_NATS,
        },
        "per_fraction": per_fraction,
        "float_identity": {
            "pair_rates": identity_rates,
            "extreme_pair_rate": extreme_pair_rate,
            "stale_signature_all_pairs_flat_0p19_0p27": stale_signature,
            "fix_took": fix_took,
        },
        "h2_frac008_control": {
            "corrected_dg": dg008,
            "stale_dg": stale_vals[0],
            "abs_diff": h2_abs_diff,
            "tolerance_nats": H2_TOLERANCE_NATS,
            "pass": h2_pass,
        },
        "h1": {
            "range_nats": dg_range,
            "range_threshold_nats": RANGE_THRESHOLD_NATS,
            "range_strong_nats": RANGE_STRONG_NATS,
            "late_minus_early_nats": dg100 - dg008,
            "spearman_rho_frac_vs_dg": rho,
            "monotone_nondecreasing": monotone_nondecreasing,
            "all_within_stale_band": all_within_stale_band,
        },
        "outcome4_routing_diagnostics": {
            "min_ceiling_distance_nats": min_ceiling_distance,
            "ceiling_pinned": ceiling_pinned,
            "glue_eos_margin_range_nats": margin_range,
            "margin_still_climbing": margin_still_climbing,
            "source_emission_range": emission_range,
            "emission_still_climbing": emission_still_climbing,
        },
        "verdict": verdict,
    }


# ── Figures (plan section 6 "Figures to produce"). ───────────────────────────


def make_figures(comparison: dict, trajectory: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    c_primary = paper_palette_role("primary")
    c_baseline = paper_palette_role("baseline")
    c_accent = paper_palette_role("accent")
    c_neutral = paper_palette_role("neutral")
    c_control = paper_palette_role("control")

    rows = comparison["per_fraction"]
    fracs = [r["frac"] for r in rows]
    corrected = [r["corrected_source_dg"] for r in rows]
    stale_vals = comparison["stale_table"]["values_nats"]
    stale_mean = comparison["stale_table"]["mean"]
    band = comparison["stale_table"]["range_threshold_nats"]
    written: list[str] = []

    def _save(fig, stem: str) -> None:
        savefig_paper(fig, stem, dir=fig_dir)
        plt.close(fig)
        written.append(stem)

    # (hero) corrected curve over the stale flat band.
    fig, ax = plt.subplots()
    ax.axhspan(stale_mean - band, stale_mean + band, color=c_neutral, alpha=0.18)
    ax.plot(
        fracs,
        stale_vals,
        "--s",
        color=c_baseline,
        label="Stale published table (one adapter read six times)",
    )
    ax.plot(fracs, corrected, "-o", color=c_primary, label="Corrected (distinct adapter ids)")
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Source implant strength (nats, trained − base)")
    ax.set_title("Corrected per-fraction calibration vs the stale published table")
    ax.legend()
    _save(fig, "hero_corrected_vs_stale_calibration")

    # (a) bystander resolution, corrected vs stale, with cluster-bootstrap CIs.
    fig, ax = plt.subplots()
    res = [r["bystander_resolution"] for r in rows]
    lo = [max(0.0, r["bystander_resolution"] - r["bystander_resolution_ci95"][0]) for r in rows]
    hi = [max(0.0, r["bystander_resolution_ci95"][1] - r["bystander_resolution"]) for r in rows]
    ax.errorbar(
        fracs,
        res,
        yerr=[lo, hi],
        fmt="-o",
        color=c_primary,
        capsize=3,
        label="Corrected (95% CI, bootstrap clustered by persona)",
    )
    ax.plot(
        fracs,
        [r["stale_bystander_resolution"] for r in rows],
        "--s",
        color=c_baseline,
        label="Stale published table",
    )
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Bystander resolution (share of probes in band)")
    ax.set_title("Bystander resolution per fraction, corrected vs stale")
    ax.legend()
    _save(fig, "bystander_resolution_vs_fraction")

    # (b) per-fraction held-out delta-G distributions (540 leaves each).
    fig, ax = plt.subplots()
    cks = {float(ck["frac"]): ck for ck in trajectory["checkpoints"]}
    data = []
    for f in fracs:
        leaves = [leaf for per_q in cks[f]["held_out"].values() for leaf in per_q.values()]
        data.append([float(leaf["delta_g"]) for leaf in leaves])
    parts = ax.violinplot(data, positions=range(len(fracs)), showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(c_primary)
        body.set_alpha(0.5)
    ax.set_xticks(range(len(fracs)), [f"{f:.2f}" for f in fracs])
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Held-out probe shift (nats, trained − base)")
    ax.set_title("Held-out marker log-prob shift distributions (54 personas × 10 questions)")
    _save(fig, "held_out_delta_g_distributions")

    # (c) source + bystander emission rate vs fraction.
    fig, ax = plt.subplots()
    ax.plot(
        fracs,
        [r["source_emission_p"] for r in rows],
        "-o",
        color=c_primary,
        label="Source (villain)",
    )
    ax.plot(
        fracs,
        [r["bystander_emission_p"] for r in rows],
        "-s",
        color=c_accent,
        label="Bystanders (held-out panel)",
    )
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Marker emission rate (argmax = marker at the slot)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Marker emission rate per fraction")
    ax.legend()
    _save(fig, "emission_rate_vs_fraction")

    # (d) saturation-signature panel: log-prob vs logit readouts, held-out + source.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(
        fracs,
        [r["held_out_mean_delta_g"] for r in rows],
        "-o",
        color=c_primary,
        label="Mean log-prob shift",
    )
    ax1.plot(
        fracs,
        [r["held_out_mean_delta_z_marker"] for r in rows],
        "-s",
        color=c_accent,
        label="Mean marker-logit shift",
    )
    ax1.plot(
        fracs,
        [r["held_out_mean_eos_margin"] for r in rows],
        "-^",
        color=c_control,
        label="Mean EOS-margin shift",
    )
    ax1.set_xlabel("Training checkpoint fraction")
    ax1.set_ylabel("Shift (nats / logits, trained − base)")
    ax1.set_title("Held-out panel: three readouts")
    ax1.legend()
    glue_dg = [r["glue_source_delta_g_mean"] for r in rows]
    glue_dg_sd = [r["glue_source_delta_g_sd"] for r in rows]
    glue_dz = [r["glue_source_delta_z_marker_mean"] for r in rows]
    glue_dz_sd = [r["glue_source_delta_z_marker_sd"] for r in rows]
    glue_mg = [r["glue_source_eos_margin_mean"] for r in rows]
    glue_mg_sd = [r["glue_source_eos_margin_sd"] for r in rows]
    ax2.errorbar(
        fracs,
        glue_dg,
        yerr=glue_dg_sd,
        fmt="-o",
        color=c_primary,
        capsize=3,
        label="Log-prob shift (mean ± SD, 10 questions)",
    )
    ax2.errorbar(
        fracs,
        glue_dz,
        yerr=glue_dz_sd,
        fmt="-s",
        color=c_accent,
        capsize=3,
        label="Marker-logit shift",
    )
    ax2.errorbar(
        fracs,
        glue_mg,
        yerr=glue_mg_sd,
        fmt="-^",
        color=c_control,
        capsize=3,
        label="EOS-margin shift",
    )
    ax2.set_xlabel("Training checkpoint fraction")
    ax2.set_ylabel("Shift (nats / logits, trained − base)")
    ax2.set_title("Source (villain): companion readouts")
    ax2.legend()
    fig.tight_layout()
    _save(fig, "saturation_signature_panel")

    # (e) collapse shares vs fraction.
    fig, ax = plt.subplots()
    ax.plot(
        fracs,
        [r["held_out_collapse_share"] for r in rows],
        "-o",
        color=c_primary,
        label="Held-out collapse share",
    )
    ax.plot(
        fracs,
        [1.0 if r["source_r_collapsed"] else 0.0 for r in rows],
        "--s",
        color=c_accent,
        label="Source response collapsed (0/1)",
    )
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Share of responses collapsed to marker repeats")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Response collapse per fraction")
    ax.legend()
    _save(fig, "collapse_vs_fraction")

    # (f) cross-fraction float-identity heatmap (the fix-took gate read).
    import numpy as np

    fig, ax = plt.subplots()
    n = len(fracs)
    mat = np.full((n, n), np.nan)
    for key, rate in comparison["float_identity"]["pair_rates"].items():
        a, b = key.split("__")
        ia, ib = fracs.index(float(a)), fracs.index(float(b))
        mat[ia, ib] = rate
        mat[ib, ia] = rate
    for i in range(n):
        mat[i, i] = 1.0
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.6 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(n), [f"{f:.2f}" for f in fracs])
    ax.set_yticks(range(n), [f"{f:.2f}" for f in fracs])
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("Training checkpoint fraction")
    ax.set_title("Exact-float-identity rate of held-out log-probs between fractions")
    fig.colorbar(im, ax=ax, label="Identity rate")
    _save(fig, "float_identity_heatmap")

    # (g) glue-vs-main source delta-G cross-check.
    fig, ax = plt.subplots()
    ax.bar([f"{f:.2f}" for f in fracs], [r["glue_vs_main_abs_diff"] for r in rows], color=c_neutral)
    ax.axhline(
        2.0, color=c_baseline, linestyle="--", label="Expected combined noise bound (2 nats)"
    )
    ax.set_xlabel("Training checkpoint fraction")
    ax.set_ylabel("|companion − main| source shift (nats)")
    ax.set_title("Companion-pass vs main-run source shift cross-check")
    ax.legend()
    _save(fig, "glue_vs_main_crosscheck")

    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Task #585: off-pod comparison of the stale #504 calibration table vs "
            "the corrected re-eval, + figures."
        )
    )
    ap.add_argument(
        "--stale", type=Path, default=Path("eval_results/issue_504/phase0_calibration_v4.json")
    )
    ap.add_argument(
        "--corrected",
        type=Path,
        default=Path("eval_results/issue_585/phase0_calibration_v4_corrected.json"),
    )
    ap.add_argument(
        "--trajectory",
        type=Path,
        default=Path("eval_results/issue_585/c504v4_smoke_eps3_reread_seed42/trajectory.json"),
    )
    ap.add_argument(
        "--slot-stats", type=Path, default=Path("eval_results/issue_585/source_slot_stats.json")
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_585/comparison_stale_vs_corrected.json"),
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_585"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=585)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=compare_figures] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    stale = json.loads(args.stale.read_text())
    corrected = json.loads(args.corrected.read_text())
    trajectory = json.loads(args.trajectory.read_text())
    slot_stats = json.loads(args.slot_stats.read_text())

    comparison = compute_comparison(
        stale, corrected, trajectory, slot_stats, args.n_boot, args.boot_seed
    )
    figures = make_figures(comparison, trajectory, args.fig_dir)

    comparison["reproducibility"] = {
        "inputs": {
            "stale": str(args.stale),
            "corrected": str(args.corrected),
            "trajectory": str(args.trajectory),
            "slot_stats": str(args.slot_stats),
        },
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "figures": figures,
        "fig_dir": str(args.fig_dir),
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(comparison, indent=2))
    log.info(
        "[phase=compare_done] verdict=%s, range=%.2f nats, h2_pass=%s, fix_took=%s -> %s",
        comparison["verdict"],
        comparison["h1"]["range_nats"],
        comparison["h2_frac008_control"]["pass"],
        comparison["float_identity"]["fix_took"],
        args.out_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
