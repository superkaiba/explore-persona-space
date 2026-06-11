# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ΔG/ε + × intentional
"""Task #600 §6 — pre-registered paired analysis (CPU, VM, post-teardown).

Inputs: the 36 per-cell ``eval_results/issue_600/sweep/<slug>/seed_<S>/
trajectory.json`` files + the committed design manifest. Outputs:
``eval_results/issue_600/analysis/analysis.json`` + figures under
``figures/issue_600/``.

Registered statistics (plan §6, with the §6.7 binding analyzer pins):

1. Headline (H1): one-sided exact target-level sign-flip permutation over the
   seed-mean paired NEAR−CONTROL differences of the implant-normalized target
   shift, at the headline checkpoint (latest co-passing on the shared grid).
2. Cross-target sign test (secondary).
3. Run-noise calibration: the within-condition across-seed gap distribution
   (12 conditions × 3 seed pairs), computed PER CHECKPOINT (§6.7(c) — gaps
   vary ~30× across checkpoints; each pair's |d| is calibrated only against
   same-checkpoint gaps). The pinned null-band statistic is the MEDIAN
   same-mix gap at the headline checkpoint (§6.7(b), pinned pre-data).
4. Locality (H1-locality): per-pair paired differences over the common
   held-out panel; per-target lower-tail percentile, Fisher-combined.
   The §6.7(d) bubble-radius read (paired difference vs distance to the NEAR
   negative) MUST be consulted before any "global mix effect" wording.
5. Robustness family (Holm within family): unnormalized ΔG paired test,
   EOS-margin-space paired test, headline re-read at other co-passing
   checkpoints.
6. Manipulation checks (realized distances, NEAR negative's own suppression,
   denominator-leak per pair, space agreement).

§6.7(a) k-demotion: with k surviving pairs the permutation's min attainable p
is 1/2^k; at k ≤ 4 the permutation is DESCRIPTIVE (no promotable bounded-null
language) and the paired-magnitude-vs-noise comparison carries.
§6.7(b) outcome lattice: Success / Partial / Null as registered in §7 are not
exhaustive — uncovered cells are labelled INDETERMINATE with components
reported separately.
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
from glob import glob
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.targeted_proximity_600 import (
    BYSTANDER_ARGMAX_CEILING,
    SEED_LEVEL_DG_FLOOR_NATS,
    SEEDS,
    SOURCE_DG_FLOOR_NATS,
    SOURCE_LOGP_CEILING_EPS_NATS,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    cell_specs_from_manifest,
    load_manifest,
)

log = logging.getLogger("issue_600.analyze")

NULL_BAND_STATISTIC = "median_same_mix_gap"  # §6.7(b): pinned BEFORE data lands.
PERMUTATION_SIGNIFICANCE = 0.05
K_DEMOTION_THRESHOLD = 4  # §6.7(a): k ≤ 4 → permutation demotes to descriptive.
SOURCE_COMPRESSION_NATS = 12.0  # §6.7(e): prefer logit reads above this.


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Loading. ─────────────────────────────────────────────────────────────────


def load_sweep(sweep_dir: Path) -> dict[tuple[str, int], dict]:
    """Load every persisted trajectory.json → {(slug, seed): payload}."""
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(glob(str(sweep_dir / "*" / "seed_*" / "trajectory.json"))):
        payload = json.loads(Path(f).read_text())
        out[(payload["cell"], int(payload["seed"]))] = payload
    if not out:
        raise FileNotFoundError(f"No trajectory.json under {sweep_dir}.")
    return out


def _ckpt_at(payload: dict, frac: float) -> dict | None:
    for ck in payload["checkpoints"]:
        if abs(float(ck["frac"]) - frac) < 1e-6:
            return ck
    return None


def _persona_mean(ck: dict, persona: str, field: str) -> float | None:
    """Mean of a per-q leaf field over Q_eval for one persona (None if absent)."""
    recs = ck["held_out"].get(persona)
    if not recs:
        return None
    vals = [leaf.get(field) for leaf in recs.values()]
    if any(v is None for v in vals):
        return None
    return float(np.mean([float(v) for v in vals]))


def _bystander_argmax_rate(ck: dict, source: str) -> float:
    flags = [
        bool(leaf["argmax_marker"])
        for p, recs in ck["held_out"].items()
        if p != source
        for leaf in recs.values()
    ]
    return float(np.mean(flags)) if flags else 0.0


# ── Gates (plan §6: floor + sub-saturation, per cell per checkpoint). ────────


def cell_gate(payload: dict, frac: float) -> dict:
    """Seed-level gate verdict for one (cell, seed) at one checkpoint."""
    ck = _ckpt_at(payload, frac)
    if ck is None:
        return {"present": False, "passes": False}
    src = ck["source_self"]
    dg = float(src["delta_g_mean"])
    rate = _bystander_argmax_rate(ck, payload["source"])
    floor_ok = dg >= SEED_LEVEL_DG_FLOOR_NATS
    sub_sat_ok = (
        float(src["g_logp_mean"]) <= -SOURCE_LOGP_CEILING_EPS_NATS
        and rate < BYSTANDER_ARGMAX_CEILING
    )
    return {
        "present": True,
        "passes": floor_ok and sub_sat_ok,
        "floor_ok": floor_ok,
        "sub_saturation_ok": sub_sat_ok,
        "source_dg": dg,
        "source_trained_logp": float(src["g_logp_mean"]),
        "bystander_argmax_rate": rate,
    }


def pair_headline_checkpoint(
    sweep: dict[tuple[str, int], dict],
    near_slug: str,
    ctrl_slug: str,
    seeds: list[int],
) -> dict:
    """§4.8: the LATEST checkpoint where BOTH pair members pass gates at every seed.

    Condition-level floor additionally requires each arm's SEED-MEAN source ΔG
    ≥ SOURCE_DG_FLOOR_NATS. Fallback: per-arm band-entry (first gate-passing
    checkpoint per arm — matched dial position, unmatched step; flagged).
    """
    fracs = sorted(TRAJECTORY_CHECKPOINT_FRACTIONS, reverse=True)
    for frac in fracs:
        ok = True
        for slug in (near_slug, ctrl_slug):
            gates = [cell_gate(sweep[(slug, s)], frac) for s in seeds if (slug, s) in sweep]
            if len(gates) < len(seeds) or not all(g["passes"] for g in gates):
                ok = False
                break
            if float(np.mean([g["source_dg"] for g in gates])) < SOURCE_DG_FLOOR_NATS:
                ok = False
                break
        if ok:
            return {"mode": "co_passing", "frac": frac}
    # Fallback: per-arm band entry (ascending fracs).
    entries = {}
    for slug in (near_slug, ctrl_slug):
        entry = None
        for frac in sorted(TRAJECTORY_CHECKPOINT_FRACTIONS):
            gates = [cell_gate(sweep[(slug, s)], frac) for s in seeds if (slug, s) in sweep]
            if (
                len(gates) == len(seeds)
                and all(g["passes"] for g in gates)
                and float(np.mean([g["source_dg"] for g in gates])) >= SOURCE_DG_FLOOR_NATS
            ):
                entry = frac
                break
        entries[slug] = entry
    if all(v is not None for v in entries.values()):
        return {"mode": "band_entry_fallback", "frac_by_slug": entries}
    return {"mode": "failed_gate", "frac_by_slug": entries}


# ── DVs. ─────────────────────────────────────────────────────────────────────


def cell_dvs(payload: dict, frac: float, persona: str) -> dict | None:
    """Per-(cell, seed, ckpt) DV bundle for one read persona.

    Returns normalized (primary), unnormalized ΔG, and EOS-margin DVs.
    """
    ck = _ckpt_at(payload, frac)
    if ck is None:
        return None
    dg_t = _persona_mean(ck, persona, "delta_g")
    if dg_t is None:
        return None
    src_dg = float(ck["source_self"]["delta_g_mean"])
    margin = _persona_mean(ck, persona, "delta_margin")  # None pre-#530 schemas
    return {
        "delta_g": dg_t,
        "normalized": dg_t / src_dg if src_dg != 0 else float("nan"),
        "delta_margin": margin,
        "source_dg": src_dg,
    }


# ── Statistics. ──────────────────────────────────────────────────────────────


def sign_flip_permutation(seed_mean_d: list[float]) -> dict:
    """One-sided exact target-level sign-flip permutation (NEAR < CONTROL → d < 0)."""
    d = np.asarray(seed_mean_d, dtype=np.float64)
    k = len(d)
    t_obs = float(d.mean())
    perms = []
    for signs in itertools.product((1.0, -1.0), repeat=k):
        perms.append(float((d * np.asarray(signs)).mean()))
    p = sum(1 for t in perms if t <= t_obs) / len(perms)
    return {
        "k_targets": k,
        "t_obs": t_obs,
        "n_enumerations": len(perms),
        "p_one_sided": p,
        "min_attainable_p": 1.0 / len(perms),
        "permutation_demoted_to_descriptive": k <= K_DEMOTION_THRESHOLD,
        "null_distribution": perms,
    }


def sign_test(seed_mean_d: list[float]) -> dict:
    """One-sided binomial sign test, predicted direction NEGATIVE."""
    n = len(seed_mean_d)
    n_neg = sum(1 for v in seed_mean_d if v < 0)
    p = sum(math.comb(n, i) for i in range(n_neg, n + 1)) / 2**n
    return {"n": n, "n_negative": n_neg, "p_one_sided": p}


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down over a named family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        out[name] = running_max
    return out


def fisher_combine(pvals: list[float]) -> dict:
    """Fisher's method; chi-square survival via the series expansion (even df)."""
    pvals = [max(p, 1e-300) for p in pvals]
    x = -2.0 * sum(math.log(p) for p in pvals)
    k = len(pvals)  # df = 2k
    # For even df, sf(x; 2k) = exp(-x/2) * sum_{i=0}^{k-1} (x/2)^i / i!
    half = x / 2.0
    sf = math.exp(-half) * sum(half**i / math.factorial(i) for i in range(k))
    return {"statistic": x, "df": 2 * k, "p": min(1.0, sf)}


def run_noise_gaps(
    sweep: dict[tuple[str, int], dict],
    slugs: list[str],
    seeds: list[int],
    frac: float,
    persona_by_slug: dict[str, str],
    field: str = "normalized",
) -> list[float]:
    """Within-condition across-seed |DV gap| distribution at one checkpoint."""
    gaps: list[float] = []
    for slug in slugs:
        vals = []
        for s in seeds:
            payload = sweep.get((slug, s))
            if payload is None:
                continue
            dv = cell_dvs(payload, frac, persona_by_slug[slug])
            if dv is not None and dv.get(field) is not None and np.isfinite(dv[field]):
                vals.append(dv[field])
        for a, b in itertools.combinations(vals, 2):
            gaps.append(abs(a - b))
    return gaps


# ── Distance matrices for locality / bubble-radius reads. ────────────────────


def _load_distances(layer: int) -> tuple[dict[str, dict[str, float]], list[str]] | None:
    """Centered-cosine distances at ``layer`` from local bundles (None when absent)."""
    from explore_persona_space.experiments.targeted_proximity_600.select_panels import (
        _i472_data_root,
        load_centered_distance_matrix,
    )

    candidates = [
        _i472_data_root() / f"centroids_L{layer}.pt",
        Path(os.environ.get("EPM_I505_DATA_ROOT", "data/issue_505"))
        / "centroids_pv"
        / f"centroids_pv_L{layer}.pt",
    ]
    for path in candidates:
        if path.exists():
            try:
                return load_centered_distance_matrix(path)
            except (KeyError, AssertionError) as e:
                log.warning("[distances] L%d bundle at %s unusable (%s)", layer, path, e)
    log.warning("[distances] no usable centroid bundle for layer %d — read skipped", layer)
    return None


# ── Main analysis. ───────────────────────────────────────────────────────────


def analyze_600(  # noqa: C901  the pre-registered stat battery is one auditable unit; splitting would scatter the §6.7 pins
    *,
    manifest_path: Path,
    sweep_dir: Path,
    analysis_dir: Path,
    figures_dir: Path,
    layers: tuple[int, ...] = (10, 15, 20, 21),
) -> dict:
    """Run the full pre-registered analysis; write analysis.json + figures."""
    manifest = load_manifest(manifest_path)
    specs = cell_specs_from_manifest(manifest)
    sweep = load_sweep(sweep_dir)
    seeds = list(SEEDS)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    targets = [t["name"] for t in manifest["targets"]]
    stratum_of = {t["name"]: t["stratum"] for t in manifest["targets"]}
    pair_slugs = {t: (f"c600_{t}_near", f"c600_{t}_ctrl") for t in targets}
    slot_of = {s.slug: s.slot_persona for s in specs}
    panel_union = sorted({p for s in specs for p in s.panel})
    common_panel = [p for p in manifest["held_out_panel"] if p not in panel_union]
    log.info(
        "[analyze] %d trajectories; common held-out panel %d personas (47 − %d panel-union)",
        len(sweep),
        len(common_panel),
        len(panel_union),
    )

    # ── Headline checkpoint per pair + paired DVs. ────────────────────────────
    per_pair: dict[str, dict] = {}
    surviving: list[str] = []
    for t in targets:
        near_slug, ctrl_slug = pair_slugs[t]
        missing = [
            (slug, s) for slug in (near_slug, ctrl_slug) for s in seeds if (slug, s) not in sweep
        ]
        head = (
            pair_headline_checkpoint(sweep, near_slug, ctrl_slug, seeds)
            if not missing
            else {
                "mode": "missing_cells",
                "missing": missing,
            }
        )
        entry: dict = {"target": t, "stratum": stratum_of[t], "headline": head}
        if head["mode"] == "co_passing":
            frac = head["frac"]
            rows = []
            for s in seeds:
                near = cell_dvs(sweep[(near_slug, s)], frac, t)
                ctrl = cell_dvs(sweep[(ctrl_slug, s)], frac, t)
                rows.append(
                    {
                        "seed": s,
                        "near": near,
                        "ctrl": ctrl,
                        "d_normalized": near["normalized"] - ctrl["normalized"],
                        "d_delta_g": near["delta_g"] - ctrl["delta_g"],
                        "d_margin": (
                            near["delta_margin"] - ctrl["delta_margin"]
                            if near["delta_margin"] is not None and ctrl["delta_margin"] is not None
                            else None
                        ),
                        "d_source_dg": near["source_dg"] - ctrl["source_dg"],
                    }
                )
            entry["frac"] = frac
            entry["per_seed"] = rows
            entry["seed_mean_d_normalized"] = float(np.mean([r["d_normalized"] for r in rows]))
            entry["seed_mean_d_delta_g"] = float(np.mean([r["d_delta_g"] for r in rows]))
            margins = [r["d_margin"] for r in rows if r["d_margin"] is not None]
            entry["seed_mean_d_margin"] = float(np.mean(margins)) if margins else None
            # §6.7(e) per-pair reports: denominator-leak + matching residual.
            entry["seed_mean_d_source_dg"] = float(np.mean([r["d_source_dg"] for r in rows]))
            t_row = next(x for x in manifest["targets"] if x["name"] == t)
            entry["signed_residual_dsource"] = t_row["near"]["d_source"] - t_row["ctrl"]["d_source"]
            entry["source_above_compression"] = bool(
                np.mean([r["near"]["source_dg"] for r in rows]) > SOURCE_COMPRESSION_NATS
            )
            surviving.append(t)
        per_pair[t] = entry

    k = len(surviving)
    failed_pairs = [t for t in targets if t not in surviving]

    # ── Headline permutation + sign test (normalized DV). ────────────────────
    seed_mean_d = [per_pair[t]["seed_mean_d_normalized"] for t in surviving]
    headline = sign_flip_permutation(seed_mean_d) if surviving else None
    sign = sign_test(seed_mean_d) if surviving else None

    # ── Run-noise calibration (per checkpoint; pinned statistic at headline). ─
    persona_by_slug = {s.slug: s.target for s in specs}
    all_slugs = [s.slug for s in specs]
    noise_by_frac = {
        f"{frac:.2f}": run_noise_gaps(sweep, all_slugs, seeds, frac, persona_by_slug)
        for frac in TRAJECTORY_CHECKPOINT_FRACTIONS
    }
    headline_fracs = sorted({per_pair[t]["frac"] for t in surviving}, reverse=True)
    headline_frac = headline_fracs[0] if headline_fracs else None
    noise_at_headline = noise_by_frac.get(f"{headline_frac:.2f}", []) if headline_frac else []
    median_gap = float(np.median(noise_at_headline)) if noise_at_headline else None
    mean_abs_d = float(np.mean(np.abs(seed_mean_d))) if seed_mean_d else None
    effect_above_noise = (
        mean_abs_d > median_gap if (mean_abs_d is not None and median_gap is not None) else None
    )

    # ── Locality (H1-locality) over the common held-out panel. ───────────────
    locality: dict[str, dict] = {}
    percentile_ps: list[float] = []
    for t in surviving:
        near_slug, ctrl_slug = pair_slugs[t]
        frac = per_pair[t]["frac"]
        d_by_persona: dict[str, float] = {}
        for p in common_panel:
            vals = []
            for s in seeds:
                near = cell_dvs(sweep[(near_slug, s)], frac, p)
                ctrl = cell_dvs(sweep[(ctrl_slug, s)], frac, p)
                if near is None or ctrl is None:
                    continue
                vals.append(near["normalized"] - ctrl["normalized"])
            if vals:
                d_by_persona[p] = float(np.mean(vals))
        non_target = {p: v for p, v in d_by_persona.items() if p != t}
        if t in d_by_persona and non_target:
            d_t = d_by_persona[t]
            n_le = sum(1 for v in non_target.values() if v <= d_t)
            p_lower_tail = (1 + n_le) / (1 + len(non_target))
            percentile_ps.append(p_lower_tail)
            locality[t] = {
                "d_target": d_t,
                "n_non_target": len(non_target),
                "lower_tail_p": p_lower_tail,
                "d_by_persona": d_by_persona,
            }
    locality_fisher = fisher_combine(percentile_ps) if percentile_ps else None

    # ── Bubble-radius read (§6.7(d) — precedes any "global mix" wording). ────
    bubble: dict[str, dict] = {}
    for layer in layers:
        dist_names = _load_distances(layer)
        if dist_names is None:
            bubble[f"L{layer}"] = {"skipped": True}
            continue
        dist, _names = dist_names
        per_target = {}
        for t in surviving:
            nn = next(x for x in manifest["targets"] if x["name"] == t)["near"]["name"]
            if nn not in dist:
                continue
            pts = [
                {"persona": p, "d_to_near_negative": dist[nn][p], "paired_difference": v}
                for p, v in locality.get(t, {}).get("d_by_persona", {}).items()
                if p in dist
            ]
            if len(pts) >= 3:
                xs = np.asarray([q["d_to_near_negative"] for q in pts])
                ys = np.asarray([q["paired_difference"] for q in pts])
                rho = _spearman(xs, ys)
                per_target[t] = {"n": len(pts), "spearman_rho": rho, "points": pts}
        bubble[f"L{layer}"] = {"skipped": False, "per_target": per_target}

    # ── Robustness family (Holm within family). ──────────────────────────────
    robustness: dict[str, dict] = {}
    if surviving:
        unnorm = sign_flip_permutation([per_pair[t]["seed_mean_d_delta_g"] for t in surviving])
        robustness["unnormalized_delta_g"] = unnorm
        margins = [per_pair[t]["seed_mean_d_margin"] for t in surviving]
        if all(m is not None for m in margins):
            robustness["eos_margin"] = sign_flip_permutation(margins)
        # Headline re-read at the other co-passing shared checkpoints.
        for frac in TRAJECTORY_CHECKPOINT_FRACTIONS:
            if headline_frac is not None and abs(frac - headline_frac) < 1e-6:
                continue
            ds = []
            for t in surviving:
                near_slug, ctrl_slug = pair_slugs[t]
                vals = []
                for s in seeds:
                    near = cell_dvs(sweep[(near_slug, s)], frac, t)
                    ctrl = cell_dvs(sweep[(ctrl_slug, s)], frac, t)
                    if near is None or ctrl is None:
                        break
                    vals.append(near["normalized"] - ctrl["normalized"])
                else:
                    ds.append(float(np.mean(vals)))
                    continue
                break
            if len(ds) == len(surviving):
                robustness[f"reread_frac_{frac:.2f}"] = sign_flip_permutation(ds)
        family_p = {name: r["p_one_sided"] for name, r in robustness.items()}
        robustness["holm_adjusted_p"] = holm_correction(family_p)

    # ── Manipulation checks. ──────────────────────────────────────────────────
    manipulation: dict[str, dict] = {}
    for t in surviving:
        near_slug, _ctrl_slug = pair_slugs[t]
        frac = per_pair[t]["frac"]
        t_row = next(x for x in manifest["targets"] if x["name"] == t)
        nn_name = t_row["near"]["name"]
        # The NEAR negative's OWN normalized DV in its cell (it is trained-on
        # there, so it rides held_out only when also in the eval list — it is,
        # via the cell panel union).
        nn_vals, panel_medians = [], []
        for s in seeds:
            payload = sweep[(near_slug, s)]
            nn_dv = cell_dvs(payload, frac, nn_name)
            if nn_dv is not None:
                nn_vals.append(nn_dv["normalized"])
            common_vals = [
                cell_dvs(payload, frac, p)["normalized"]
                for p in common_panel
                if cell_dvs(payload, frac, p) is not None
            ]
            if common_vals:
                panel_medians.append(float(np.median(common_vals)))
        manipulation[t] = {
            "realized_d_near_to_target": t_row["near"]["d_to_target"],
            "realized_d_ctrl_to_target": t_row["ctrl"]["d_to_target"],
            "realized_contrast": t_row["realized_contrast"],
            "abs_dsource_mismatch": t_row["ctrl"]["dsource_mismatch"],
            "eps_used": t_row["ctrl"]["eps_used"],
            "near_negative_own_normalized_dv_mean": (float(np.mean(nn_vals)) if nn_vals else None),
            "common_panel_median_normalized_dv_mean": (
                float(np.mean(panel_medians)) if panel_medians else None
            ),
            "near_negative_suppressed_below_median": (
                float(np.mean(nn_vals)) < float(np.mean(panel_medians))
                if nn_vals and panel_medians
                else None
            ),
        }

    # ── Outcome lattice (§6.7(b)). ────────────────────────────────────────────
    outcome = _classify_outcome(
        headline=headline,
        effect_above_noise=effect_above_noise,
        locality_fisher=locality_fisher,
        k_surviving=k,
        n_targets=len(targets),
    )

    result = {
        "schema_version": "i600_analysis_v1",
        "null_band_statistic": NULL_BAND_STATISTIC,
        "n_trajectories": len(sweep),
        "targets": targets,
        "surviving_pairs": surviving,
        "failed_gate_pairs": failed_pairs,
        "k_surviving": k,
        "k_demotion_applied": k <= K_DEMOTION_THRESHOLD,
        "headline_checkpoint_frac": headline_frac,
        "per_pair": per_pair,
        "headline_permutation": headline,
        "sign_test": sign,
        "run_noise": {
            "gaps_by_frac": noise_by_frac,
            "median_same_mix_gap_at_headline": median_gap,
            "mean_abs_paired_difference": mean_abs_d,
            "effect_above_noise_band": effect_above_noise,
        },
        "locality": {
            "common_panel": common_panel,
            "per_target": {
                t: {kk: vv for kk, vv in v.items() if kk != "d_by_persona"}
                for t, v in locality.items()
            },
            "fisher_combined": locality_fisher,
        },
        "bubble_radius": {
            layer: (
                {"skipped": True}
                if v.get("skipped")
                else {
                    "per_target": {
                        t: {"n": d["n"], "spearman_rho": d["spearman_rho"]}
                        for t, d in v["per_target"].items()
                    }
                }
            )
            for layer, v in bubble.items()
        },
        "robustness_family": robustness,
        "manipulation_checks": manipulation,
        "outcome": outcome,
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path = analysis_dir / "analysis.json"
    out_path.write_text(json.dumps(result, indent=2))
    # Full locality + bubble detail (with per-persona points) in a sidecar.
    (analysis_dir / "locality_detail.json").write_text(
        json.dumps({"locality": locality, "bubble_radius": bubble}, indent=2)
    )
    log.info("[analyze] wrote %s (outcome=%s)", out_path, outcome["label"])

    _make_figures(
        result=result,
        locality=locality,
        bubble=bubble,
        sweep=sweep,
        per_pair=per_pair,
        pair_slugs=pair_slugs,
        slot_of=slot_of,
        figures_dir=figures_dir,
    )
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _classify_outcome(
    *,
    headline: dict | None,
    effect_above_noise: bool | None,
    locality_fisher: dict | None,
    k_surviving: int,
    n_targets: int,
) -> dict:
    """§7 + §6.7(a)/(b): Success / Partial / Null / INDETERMINATE / failed-gate."""
    if headline is None or k_surviving == 0:
        return {"label": "failed_gate", "reason": "no surviving pairs"}
    if n_targets - k_surviving >= 3:
        return {
            "label": "failed_gate",
            "reason": f"≥3 of {n_targets} pairs failed gates (kill criterion ii)",
        }
    demoted = k_surviving <= K_DEMOTION_THRESHOLD
    p = headline["p_one_sided"]
    sig = p <= PERMUTATION_SIGNIFICANCE and not demoted
    local = bool(locality_fisher and locality_fisher["p"] <= PERMUTATION_SIGNIFICANCE)
    if sig and effect_above_noise and local:
        label = "success_local_suppression"
    elif sig and effect_above_noise and not local:
        # §6.7(d): the bubble-radius read must be consulted before narrating
        # this as a global mix effect (a true suppression bubble dilutes the
        # target's percentile).
        label = "partial_significant_but_locality_failed_CONSULT_BUBBLE_RADIUS"
    elif (not sig) and effect_above_noise is False and not demoted:
        label = "null_promotable_bounded"
    else:
        label = "indeterminate"
    return {
        "label": label,
        "permutation_p": p,
        "permutation_demoted": demoted,
        "effect_above_noise_band": effect_above_noise,
        "locality_p": locality_fisher["p"] if locality_fisher else None,
        "note": (
            "k ≤ 4 → permutation descriptive; magnitude-vs-noise carries; NO promotable "
            "bounded-null language (§6.7(a))."
            if demoted
            else "components reported separately for any uncovered lattice cell (§6.7(b))."
        ),
    }


# ── Figures. ─────────────────────────────────────────────────────────────────


def _make_figures(  # noqa: C901  one figure block per registered read; splitting buys nothing
    *,
    result: dict,
    locality: dict,
    bubble: dict,
    sweep: dict,
    per_pair: dict,
    pair_slugs: dict,
    slot_of: dict,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    surviving = result["surviving_pairs"]
    if not surviving:
        log.warning("[figures] no surviving pairs — skipping figures")
        return
    strata = ("near", "mid", "far")
    stratum_of = {t: per_pair[t]["stratum"] for t in surviving}

    # ── Hero: paired dumbbell, one panel per stratum. ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    median_gap = result["run_noise"]["median_same_mix_gap_at_headline"]
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")
    for ax, stratum in zip(axes, strata, strict=True):
        ts = [t for t in surviving if stratum_of[t] == stratum]
        for t in ts:
            for row in per_pair[t]["per_seed"]:
                near_v = row["near"]["normalized"]
                ctrl_v = row["ctrl"]["normalized"]
                color = accent if near_v < ctrl_v else neutral
                ax.plot([0, 1], [near_v, ctrl_v], "-o", color=color, alpha=0.65, ms=4, lw=1.2)
        if median_gap is not None and ts:
            mids = [
                0.5 * (row["near"]["normalized"] + row["ctrl"]["normalized"])
                for t in ts
                for row in per_pair[t]["per_seed"]
            ]
            center = float(np.mean(mids))
            ax.axhspan(center - median_gap / 2, center + median_gap / 2, alpha=0.12, color="gray")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Nearest-neighbor\nnegative", "Distance-matched\nfar control"])
        ax.set_title(f"{stratum}-villain targets: {', '.join(ts)}", fontsize=9)
    axes[0].set_ylabel("Implant-normalized target shift\n(target ΔlogP ÷ source ΔlogP)")
    headline = result["headline_permutation"]
    fig.suptitle(
        f"Paired NEAR vs CONTROL target leakage — permutation p = {headline['p_one_sided']:.3f} "
        f"(T = {headline['t_obs']:+.4f}); shaded band = median same-mix seed gap",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "hero_paired_dumbbell", dir=figures_dir)
    plt.close(fig)

    # ── Permutation null histogram + observed T. ─────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(headline["null_distribution"], bins=16, color=neutral, alpha=0.8)
    ax.axvline(headline["t_obs"], color=accent, lw=2)
    ax.set_xlabel("Permutation T (mean paired difference, sign-flipped)")
    ax.set_ylabel("Count")
    ax.set_title(f"Exact sign-flip null ({headline['n_enumerations']} enumerations)")
    fig.tight_layout()
    savefig_paper(fig, "permutation_null", dir=figures_dir)
    plt.close(fig)

    # ── Per-pair source + target trajectories (exploratory dump). ────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, t in zip(axes.flat, surviving, strict=False):
        for slug, ls in ((pair_slugs[t][0], "-"), (pair_slugs[t][1], "--")):
            for seed in SEEDS:
                payload = sweep.get((slug, seed))
                if payload is None:
                    continue
                fr, src, tgt = [], [], []
                for ck in payload["checkpoints"]:
                    fr.append(float(ck["frac"]))
                    src.append(float(ck["source_self"]["delta_g_mean"]))
                    tv = _persona_mean(ck, t, "delta_g")
                    tgt.append(tv if tv is not None else np.nan)
                ax.plot(fr, src, ls, color=neutral, alpha=0.5, lw=1)
                ax.plot(fr, tgt, ls, color=accent, alpha=0.7, lw=1)
        ax.set_title(t, fontsize=9)
    fig.suptitle(
        "Source (gray) and target (colored) ΔlogP trajectories — solid = nearest-neighbor "
        "cell, dashed = matched control cell",
        fontsize=10,
    )
    fig.tight_layout()
    savefig_paper(fig, "trajectories_by_pair", dir=figures_dir)
    plt.close(fig)

    # ── Locality: paired difference vs distance to the NEAR negative (raw). ──
    l10 = bubble.get("L10", {})
    if not l10.get("skipped"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for t, d in l10.get("per_target", {}).items():
            xs = [q["d_to_near_negative"] for q in d["points"]]
            ys = [q["paired_difference"] for q in d["points"]]
            ax.scatter(xs, ys, s=14, alpha=0.55, label=t)
            tgt_pt = next((q for q in d["points"] if q["persona"] == t), None)
            if tgt_pt:
                ax.scatter(
                    [tgt_pt["d_to_near_negative"]],
                    [tgt_pt["paired_difference"]],
                    s=90,
                    facecolors="none",
                    edgecolors=accent,
                    linewidths=1.6,
                )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Centered L10 distance to the pair's nearest-neighbor negative")
        ax.set_ylabel("Paired difference (normalized)")
        ax.set_title("Bubble-radius read: suppression vs distance to the added negative")
        ax.legend(fontsize=7)
        fig.tight_layout()
        savefig_paper(fig, "bubble_radius_L10_raw", dir=figures_dir)
        plt.close(fig)

    # ── Locality percentile strip. ────────────────────────────────────────────
    if locality:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        names = list(locality)
        ps = [locality[t]["lower_tail_p"] for t in names]
        ax.scatter(range(len(names)), ps, color=accent)
        ax.axhline(0.05, color="gray", ls="--", lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Lower-tail percentile of target's paired difference")
        ax.set_title("Locality: is the target in the suppressed tail of the panel?")
        fig.tight_layout()
        savefig_paper(fig, "locality_percentiles", dir=figures_dir)
        plt.close(fig)

    # ── Unnormalized alongside normalized (per-seed scatter). ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, label in (
        (axes[0], "d_normalized", "normalized (primary)"),
        (axes[1], "d_delta_g", "unnormalized ΔlogP (robustness)"),
    ):
        for i, t in enumerate(surviving):
            vals = [row[field] for row in per_pair[t]["per_seed"]]
            ax.scatter([i] * len(vals), vals, alpha=0.7, color=accent)
            ax.scatter([i], [float(np.mean(vals))], marker="_", s=300, color="black")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(surviving)))
        ax.set_xticklabels(surviving, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(f"Paired difference, {label}")
    fig.suptitle("Per-seed paired differences (negative = suppression next to the negative)")
    fig.tight_layout()
    savefig_paper(fig, "paired_differences_per_seed", dir=figures_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Task #600 pre-registered analysis (CPU, VM)")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_results/issue_600/panel_selection.json"),
    )
    ap.add_argument("--sweep-dir", type=Path, default=Path("eval_results/issue_600/sweep"))
    ap.add_argument("--analysis-dir", type=Path, default=Path("eval_results/issue_600/analysis"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures/issue_600"))
    args = ap.parse_args(argv)
    analyze_600(
        manifest_path=args.manifest,
        sweep_dir=args.sweep_dir,
        analysis_dir=args.analysis_dir,
        figures_dir=args.figures_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
