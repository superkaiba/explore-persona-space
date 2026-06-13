# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek ΔG + Unicode minus intentional
"""Task #610 — the §6 registered comparison (CPU, VM, post-teardown).

Inputs: the 3 PARENT chassis-comparator trajectories (committed on main;
``c600_mercenary_near`` for round 1, ``c600_software_engineer_near`` for the
``--chassis software_engineer`` follow-up — v2 plan §2) and the 3 new
no-default trajectories (committed on the issue branch), plus the parent
design manifest. No GPU, no pod. All chassis-dependent names (comparator
slug, sanity set + registered v2 §5 detector medians, replacement persona +
its ctrl precedent, default output/figure paths) come from ``ChassisConfig``.

Primary DV (per seed, per arm): the never-trained default context's CENTERED,
IMPLANT-NORMALIZED marker log-prob shift at the terminal checkpoint —
``mean_q(delta_g[qwen_default]) / source_delta_g − median over the 35-persona
untrained centering set of the same normalized quantity``. The centering set
is FROZEN BY FORMULA from the parent manifest: held-out 47 − the 6 targets −
every persona trained anywhere in #600 (base panel ∪ qwen_default ∪ all
near/ctrl slots). The formula reproduces the parent's committed −0.2003 /
−0.2016 / −0.1867 (seeds 42/137/219) exactly.

Decision rule (plan §6.1, zones on median(D_without)):
  IDENTITY ≤ median(D_with) + 0.033;  DOSE ≥ −0.033;  PARTIAL in between
  (graded; f = fraction of the gap closed). Identity-below anomaly
  (median(D_without) < median(D_with) − band) is named, never absorbed (§14.6).

Statistics: the zone rule carries the conclusion; the exact 3v3 rank-sum
(20 partitions, min p = 0.05) is SUPPORTING only — never narrated as
significance (§14.5). No paired-difference framing across arms (seed labels
are nominal).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.default_dose_610 import (
    ALWAYS_INCLUDE_NEGATIVE,
    ASSISTANT_TRAINED_SLOT_PRECEDENT,
    CHASSES,
    DECISION_BAND,
    EXTRA_EVAL_PERSONAS,
    SEEDS,
    SOURCE_PERSONA,
    TRAJECTORY_CHECKPOINT_FRACTIONS,
    ChassisConfig,
)
from explore_persona_space.experiments.default_dose_610.dispatch import FOUR_FLOAT_FIELDS
from explore_persona_space.experiments.targeted_proximity_600.cells import load_manifest

log = logging.getLogger("issue_610.analyze")

TERMINAL_FRAC = 1.0
EXPECTED_CENTERING_N = 35


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ── Centering set (frozen by formula from the parent manifest). ──────────────


def centering_set(manifest: dict, chassis: ChassisConfig = CHASSES["mercenary"]) -> list[str]:
    """held-out 47 − 6 targets − every persona trained anywhere in #600.

    Untrained in EVERY cell of both experiments: every chassis's #610 panel is
    a subset of the excluded trained-anywhere set, so the centering personas
    are identical across arms AND across chassis (v2 plan A4: the formula
    excludes all near/ctrl slots, hence hospice_nurse + data_scientist too).
    """
    held = set(manifest["held_out_panel"])
    targets = {t["name"] for t in manifest["targets"]}
    base_panel = {b["name"] for b in manifest["base_panel"]}
    slots = {t[k]["name"] for t in manifest["targets"] for k in ("near", "ctrl")}
    trained_anywhere = base_panel | {ALWAYS_INCLUDE_NEGATIVE} | slots
    out = sorted(held - targets - trained_anywhere)
    # ORDERING IS LOAD-BEARING: a chassis whose replacement is a #610-untrained
    # persona (e.g. #632's `programmer`, which survives `trained_anywhere`) is
    # a TRAINED negative in this design, so it must leave the centering set on
    # BOTH arms. The extra-exclude MUST run before the N-check (else N would be
    # 35 vs the expected 34) AND before the replacement-in-out guard below
    # (else that guard would itself raise on the still-present replacement).
    extra = set(chassis.centering_extra_exclude)  # () for #610 → 35; {programmer} for #632 → 34
    out = [p for p in out if p not in extra]
    expected_n = EXPECTED_CENTERING_N - len(extra)
    if len(out) != expected_n:
        raise AssertionError(
            f"centering set has {len(out)} personas, expected {expected_n} "
            f"(extra_exclude={sorted(extra)}) — the manifest is a different generation than the "
            "one this formula was frozen against."
        )
    if (chassis.replacement in out) or (SOURCE_PERSONA in out):
        raise AssertionError(
            f"centering set must exclude the replacement persona ({chassis.replacement!r}) "
            f"+ source (extra_exclude={sorted(extra)})."
        )
    return out


# ── Loading + schema asserts. ────────────────────────────────────────────────


def _ckpt_at(payload: dict, frac: float) -> dict:
    for ck in payload["checkpoints"]:
        if abs(float(ck["frac"]) - frac) < 1e-6:
            return ck
    raise KeyError(
        f"trajectory lacks frac={frac}; has {[c['frac'] for c in payload['checkpoints']]}"
    )


def _persona_mean(ck: dict, persona: str, field: str) -> float | None:
    recs = ck["held_out"].get(persona)
    if not recs:
        return None
    vals = [leaf.get(field) for leaf in recs.values()]
    if any(v is None for v in vals):
        return None
    return float(np.mean([float(v) for v in vals]))


def _assert_schema(payload: dict, required_personas: tuple[str, ...], label: str) -> None:
    """Gate-(i)-shaped assert on BOTH arms' files (plan §4.2): logit_fields
    true + every required persona carries the four floats under the REALIZED
    `_g`/`_b` suffixes at every checkpoint."""
    if not payload.get("logit_fields"):
        raise AssertionError(f"[{label}] trajectory lacks logit_fields=true.")
    for ck in payload["checkpoints"]:
        for persona in required_personas:
            recs = ck["held_out"].get(persona)
            if not recs:
                raise AssertionError(
                    f"[{label}] frac={ck['frac']}: held_out[{persona!r}] absent/empty."
                )
            for q, leaf in recs.items():
                absent = [f for f in FOUR_FLOAT_FIELDS if leaf.get(f) is None]
                if absent:
                    raise AssertionError(
                        f"[{label}] frac={ck['frac']}: {persona}/{q!r} lacks {absent} — "
                        "the `_g`/`_b` four-float contract does not hold."
                    )


def load_arm(
    sweep_dir: Path, slug: str, seeds: tuple[int, ...], required_personas: tuple[str, ...]
) -> dict[int, dict]:
    """Load + schema-assert one arm's per-seed trajectories (fail-loud)."""
    out: dict[int, dict] = {}
    for seed in seeds:
        path = sweep_dir / slug / f"seed_{seed}" / "trajectory.json"
        if not path.exists():
            raise FileNotFoundError(f"trajectory missing: {path}")
        payload = json.loads(path.read_text())
        _assert_schema(payload, required_personas, f"{slug}_seed{seed}")
        out[seed] = payload
    return out


# ── DVs. ─────────────────────────────────────────────────────────────────────


def normalized_shift(payload: dict, frac: float, persona: str) -> float | None:
    """Implant-normalized shift: mean_q(delta_g[persona]) / source ΔG."""
    ck = _ckpt_at(payload, frac)
    dg = _persona_mean(ck, persona, "delta_g")
    if dg is None:
        return None
    src = float(ck["source_self"]["delta_g_mean"])
    return dg / src if src != 0 else float("nan")


def centered_shift(payload: dict, frac: float, persona: str, centering: list[str]) -> float:
    """The PRIMARY DV: normalized shift minus the untrained-panel median."""
    val = normalized_shift(payload, frac, persona)
    if val is None:
        raise AssertionError(f"persona {persona!r} unreadable at frac={frac}.")
    med = panel_median(payload, frac, centering)
    return val - med


def panel_median(payload: dict, frac: float, centering: list[str]) -> float:
    vals = [normalized_shift(payload, frac, p) for p in centering]
    missing = [p for p, v in zip(centering, vals, strict=True) if v is None]
    if missing:
        raise AssertionError(f"centering personas missing from trajectory: {missing}")
    return float(np.median([float(v) for v in vals]))


# ── Statistics. ──────────────────────────────────────────────────────────────


def _average_ranks(values: list[float]) -> list[float]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = np.asarray(values, dtype=np.float64)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks.tolist()


def exact_rank_sum(d_without: list[float], d_with: list[float]) -> dict:
    """One-sided exact Wilcoxon rank-sum over the 6 values (20 partitions).

    Direction: DOSE raises D_without, so p = P(rank-sum of the without-group
    ≥ observed) under all C(6,3) relabelings. SUPPORTING only (min p = 0.05).
    """
    pooled = list(d_without) + list(d_with)
    n_w = len(d_without)
    ranks = _average_ranks(pooled)
    obs = float(sum(ranks[:n_w]))
    stats = [
        float(sum(ranks[i] for i in combo))
        for combo in itertools.combinations(range(len(pooled)), n_w)
    ]
    p = sum(1 for s in stats if s >= obs) / len(stats)
    return {
        "statistic": "rank_sum_without_arm",
        "observed": obs,
        "n_partitions": len(stats),
        "p_one_sided": p,
        "min_attainable_p": 1.0 / len(stats),
        "supporting_only": True,
        "all_without_above_all_with": min(d_without) > max(d_with),
    }


def classify(median_without: float, median_with: float, band: float) -> dict:
    """The §6.1 zone rule (exhaustive partition + the §14.6 anomaly flag)."""
    identity_threshold = median_with + band
    dose_threshold = -band
    if median_without >= dose_threshold:
        zone = "DOSE"
    elif median_without <= identity_threshold:
        zone = "IDENTITY"
    else:
        zone = "PARTIAL"
    f_gap_closed = (
        (median_without - median_with) / abs(median_with) if median_with != 0 else float("nan")
    )
    return {
        "zone": zone,
        "median_without": median_without,
        "median_with": median_with,
        "band": band,
        "identity_threshold": identity_threshold,
        "dose_threshold": dose_threshold,
        "fraction_of_gap_closed": f_gap_closed,
        # §14.6: never-trained default MORE shielded than trained — classifies
        # IDENTITY but is an anomaly; named, not absorbed.
        "identity_below_anomaly": median_without < (median_with - band),
    }


def band_verdict(m_new: float, m_comparator: float, band: float) -> dict:
    """The plan §3/§6.4 registered per-read symmetric band test (#632 Must-Fix 2).

    Compares a NEW arm's centered-shift median against ITS OWN comparator
    median (qwen_default vs the qwen_default comparator; assistant vs the
    assistant comparator — never one read against the other's anchor):
      HELD ⇔ |m_new − m_comparator| ≤ band (positional floor; no movement);
      FALSIFIED ⇔ |m_new − m_comparator| > 2*band (proximity-modulated
        suppression — the surprising outcome);
      PARTIAL ⇔ in between.
    Returns the verdict plus the signed delta, direction, and fraction-of-band.
    """
    delta = m_new - m_comparator
    ad = abs(delta)
    verdict = "HELD" if ad <= band else ("FALSIFIED" if ad > 2 * band else "PARTIAL")
    return {
        "median_new": m_new,
        "median_comparator": m_comparator,
        "delta": delta,
        "abs_delta": ad,
        "band": band,
        "verdict": verdict,
        "direction": ("more_shielded" if delta < 0 else "less_shielded"),
        "fraction_of_band": ad / band if band != 0 else float("nan"),
    }


def default_specific_gap_median(parent_sweep: Path, centering: list[str]) -> dict:
    """The finer calibration (§6.1): same-mix seed-pair |gap| of the centered
    default read over EVERY committed parent mix (n = 12 mixes × 3 pairs).
    Reported alongside the registered 0.033 band, never replacing it."""
    gaps: list[float] = []
    n_mixes = 0
    for cell_dir in sorted(parent_sweep.iterdir()):
        if not cell_dir.is_dir():
            continue
        vals = []
        for seed_dir in sorted(cell_dir.glob("seed_*")):
            traj = seed_dir / "trajectory.json"
            if not traj.exists():
                continue
            payload = json.loads(traj.read_text())
            try:
                vals.append(
                    centered_shift(payload, TERMINAL_FRAC, ALWAYS_INCLUDE_NEGATIVE, centering)
                )
            except (AssertionError, KeyError):
                continue
        if len(vals) >= 2:
            n_mixes += 1
            gaps.extend(abs(a - b) for a, b in itertools.combinations(vals, 2))
    return {
        "n_mixes": n_mixes,
        "n_gaps": len(gaps),
        "median_gap": float(np.median(gaps)) if gaps else None,
        "max_gap": float(np.max(gaps)) if gaps else None,
    }


# ── The registered analysis. ─────────────────────────────────────────────────


def analyze_610(  # noqa: C901  the pre-registered read battery is one auditable unit
    *,
    parent_sweep: Path,
    new_sweep: Path,
    manifest_path: Path,
    out_path: Path,
    figures_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    chassis: ChassisConfig = CHASSES["mercenary"],
) -> dict:
    """Compute every §6 registered read; write analysis.json + figures."""
    manifest = load_manifest(manifest_path)
    centering = centering_set(manifest, chassis)

    # Both arms, schema-asserted (`_g`/`_b` four-float contract on every file).
    # Must-Fix 1: the comparator (`chassis_slug`) lives under issue_600/sweep for
    # the two #610 chassis (whose comparator IS a c600 cell) but under
    # issue_610/sweep for #632 (whose comparator IS a c610 cell). Resolve from
    # the chassis; fall back to `parent_sweep` when unset → byte-identical #610.
    # `parent_sweep` itself stays the #600 root so `default_specific_gap_median`
    # keeps reading the 12-mix calibration (and the ctrl_dir glob, when built).
    comparator_root = chassis.comparator_sweep_root or parent_sweep
    with_arm = load_arm(
        comparator_root, chassis.chassis_slug, seeds, required_personas=(ALWAYS_INCLUDE_NEGATIVE,)
    )
    without_arm = load_arm(
        new_sweep, chassis.new_slug, seeds, required_personas=EXTRA_EVAL_PERSONAS
    )

    # ── §6.1 headline: per-seed D at the terminal checkpoint. ────────────────
    d_with = {
        s: centered_shift(p, TERMINAL_FRAC, ALWAYS_INCLUDE_NEGATIVE, centering)
        for s, p in with_arm.items()
    }
    d_without = {
        s: centered_shift(p, TERMINAL_FRAC, ALWAYS_INCLUDE_NEGATIVE, centering)
        for s, p in without_arm.items()
    }
    median_with = float(np.median(list(d_with.values())))
    median_without = float(np.median(list(d_without.values())))
    headline = classify(median_without, median_with, DECISION_BAND)
    finer = default_specific_gap_median(parent_sweep, centering)
    # §14.4 band-choice sensitivity strip: [median_with + finer, median_with + 0.033].
    strip = (
        [median_with + finer["median_gap"], median_with + DECISION_BAND]
        if finer["median_gap"] is not None
        else None
    )
    headline["band_sensitivity_strip"] = strip
    headline["band_choice_sensitive"] = bool(
        strip is not None and strip[0] <= median_without <= strip[1]
    )
    # Plan §6.4 HEADLINE verdict (the registered per-read symmetric band test,
    # #632 Must-Fix 2) — qwen_default: the NEW arm's median (median_without) vs
    # the COMPARATOR arm's median (median_with), both centered on the same set.
    # The #610-inherited zone classify() stays in `headline` as a reported
    # secondary surface; the band verdict is the registered headline.
    headline["band_verdict_qwen_default"] = band_verdict(median_without, median_with, DECISION_BAND)

    # ── §6.2 supporting exact inference. ─────────────────────────────────────
    rank_test = exact_rank_sum(list(d_without.values()), list(d_with.values()))

    # ── §6.3 secondary: the untrained `assistant` cluster-identity probe. ────
    a_without = {
        s: centered_shift(p, TERMINAL_FRAC, "assistant", centering) for s, p in without_arm.items()
    }
    median_assistant = float(np.median(list(a_without.values())))
    # #632 Must-Fix 2: the registered assistant headline is the per-read band
    # test against the COMPARATOR arm's OWN assistant median (computed from the
    # SAME with_arm trajectories on the same centering set), NOT qwen_default's
    # median_with. The comparator assistant median is −0.2063 at plan time.
    comparator_assistant = {
        s: centered_shift(p, TERMINAL_FRAC, "assistant", centering) for s, p in with_arm.items()
    }
    median_assistant_comparator = float(np.median(list(comparator_assistant.values())))
    assistant_band = band_verdict(median_assistant, median_assistant_comparator, DECISION_BAND)
    # Keep the legacy zone label for the §6.3 mechanism matrix (reported, not
    # the headline), but anchor it on the assistant comparator now, not
    # qwen_default's median_with (which would mis-classify a −0.2063 anchor
    # against a −0.1977 threshold).
    if median_assistant <= median_assistant_comparator + DECISION_BAND:
        assistant_label = "low"
    elif median_assistant >= -DECISION_BAND:
        assistant_label = "high"
    else:
        assistant_label = "intermediate"
    matrix = {
        ("IDENTITY", "low"): "cluster_position_mechanism",
        ("IDENTITY", "high"): "default_specific_identity",
        ("DOSE", "low"): "dose_effect_default_specific",
        ("DOSE", "high"): "dose_effect_general",
    }
    # Percentile of the assistant read within the new arm's own untrained-panel
    # centered distribution (§14.5: the primary comparison surface).
    assistant_percentiles = {}
    for s, p in without_arm.items():
        med = panel_median(p, TERMINAL_FRAC, centering)
        panel_vals = [float(normalized_shift(p, TERMINAL_FRAC, q)) - med for q in centering]
        assistant_percentiles[s] = float(np.mean([v <= a_without[s] for v in panel_vals]))
    secondary = {
        "assistant_centered_by_seed": a_without,
        "assistant_median": median_assistant,
        "assistant_label": assistant_label,
        "assistant_percentile_in_own_panel_by_seed": assistant_percentiles,
        "parent_trained_slot_precedent": ASSISTANT_TRAINED_SLOT_PRECEDENT,
        "precedent_caveat": "cross-mix AND cross-training-status — mechanism color only",
        "interpretation": matrix.get((headline["zone"], assistant_label), "mechanism_ambiguous"),
        # #632 Must-Fix 2: the registered secondary headline verdict, the
        # per-read band test against the assistant comparator's OWN median.
        "assistant_median_comparator": median_assistant_comparator,
        "assistant_centered_comparator_by_seed": comparator_assistant,
        "assistant_band_verdict": assistant_band,
    }

    # ── §6.4 sanity reads. ───────────────────────────────────────────────────
    sanity: dict[str, dict] = {}
    for persona in chassis.sanity_personas:
        sw = {s: centered_shift(p, TERMINAL_FRAC, persona, centering) for s, p in with_arm.items()}
        so = {
            s: centered_shift(p, TERMINAL_FRAC, persona, centering) for s, p in without_arm.items()
        }
        delta = float(np.median(list(so.values())) - np.median(list(sw.values())))
        sanity[persona] = {
            "with_by_seed": sw,
            "without_by_seed": so,
            "median_delta": delta,
            "passes": abs(delta) <= 2 * DECISION_BAND,
        }
        # v2 plan §5: the registered with-arm drift-detector medians were
        # computed by THIS formula from the committed comparator trajectories
        # at plan time; a recomputation drift means the comparator data or the
        # centering formula changed under us — fail loud (tolerance absorbs
        # the plan's 4-decimal rounding only).
        if chassis.sanity_with_arm_expected is not None:
            registered = chassis.sanity_with_arm_expected[persona]
            recomputed = float(np.median(list(sw.values())))
            if abs(recomputed - registered) > 2e-3:
                raise AssertionError(
                    f"with-arm sanity median for {persona!r} recomputes to {recomputed:+.4f} "
                    f"but the registered v2 §5 value is {registered:+.4f} — comparator data "
                    "or centering formula drifted; refusing to analyze."
                )
            sanity[persona]["with_arm_registered_median"] = registered
    deltas = [v["median_delta"] for v in sanity.values()]
    replacement_without = {
        s: centered_shift(p, TERMINAL_FRAC, chassis.replacement, centering)
        for s, p in without_arm.items()
    }
    j_median = float(np.median(list(replacement_without.values())))
    # Parent ctrl-cell replacement read recomputed from data when committed
    # (reported alongside the registered per-chassis precedent). #632 Must-Fix
    # 3: the whole ctrl-precedent COMPARISON is guarded on a non-None precedent
    # — #632's proximal pick (programmer) is NOT a ctrl persona, so #600 has no
    # c600_mercenary_ctrl programmer row to compare against. The trained-slot
    # read (median + by_seed) is still produced (it IS the §5 "programmer
    # trained-slot read"); only the precedent comparison is skipped (passes=None).
    j_parent_recomputed = None
    if chassis.replacement_ctrl_precedent is not None:
        ctrl_dir = parent_sweep / f"c600_{chassis.chassis_target}_ctrl"
        if ctrl_dir.is_dir():
            vals = []
            for seed_dir in sorted(ctrl_dir.glob("seed_*")):
                traj = seed_dir / "trajectory.json"
                if traj.exists():
                    payload = json.loads(traj.read_text())
                    # Optional descriptive read: a parent ctrl trajectory
                    # predating the four-float schema is skipped, not fatal (the
                    # registered precedent constant carries the comparison).
                    with contextlib.suppress(AssertionError, KeyError):
                        vals.append(
                            centered_shift(payload, TERMINAL_FRAC, chassis.replacement, centering)
                        )
            if vals:
                j_parent_recomputed = float(np.median(vals))
    replacement_read = {
        "persona": chassis.replacement,
        "without_by_seed": replacement_without,
        "median": j_median,
        "ctrl_precedent": chassis.replacement_ctrl_precedent,
        "ctrl_precedent_recomputed": j_parent_recomputed,
        # None (skipped) when there is no ctrl precedent for this replacement;
        # `any_miss` below treats None as "skipped", NOT a miss.
        "passes": (
            None
            if chassis.replacement_ctrl_precedent is None
            else abs(j_median - chassis.replacement_ctrl_precedent) <= 2 * DECISION_BAND
        ),
    }
    raw_medians = {
        "with": {s: panel_median(p, TERMINAL_FRAC, centering) for s, p in with_arm.items()},
        "without": {s: panel_median(p, TERMINAL_FRAC, centering) for s, p in without_arm.items()},
    }
    sanity_summary = {
        "per_persona": sanity,
        "replacement_trained_read": replacement_read,
        "raw_untrained_panel_medians_normalized": raw_medians,
        # §14.4: coherent one-direction movement (drift signature) vs a single
        # persona's seed wander — components for the analyzer, not a mechanical
        # demotion.
        "coherent_drift_signature": bool(
            len({np.sign(d) for d in deltas}) == 1 and all(abs(d) > DECISION_BAND for d in deltas)
        ),
        # #632 Must-Fix 3: a skipped (None) ctrl precedent does NOT count as a
        # miss — only an explicit False does (`not None` would be a false miss).
        "any_miss": bool(
            any(not v["passes"] for v in sanity.values()) or (replacement_read["passes"] is False)
        ),
    }
    if chassis.name == "mercenary":
        # Legacy alias: the round-1 analyzer figures script
        # (scripts/i610_analyzer_figures.py) reads sanity["journalist_trained_read"].
        sanity_summary["journalist_trained_read"] = replacement_read

    # ── §6.5 exploratory. ────────────────────────────────────────────────────
    trajectory = {
        "with": {
            s: {
                f"{fr:.2f}": centered_shift(p, fr, ALWAYS_INCLUDE_NEGATIVE, centering)
                for fr in TRAJECTORY_CHECKPOINT_FRACTIONS
            }
            for s, p in with_arm.items()
        },
        "without": {
            s: {
                f"{fr:.2f}": centered_shift(p, fr, ALWAYS_INCLUDE_NEGATIVE, centering)
                for fr in TRAJECTORY_CHECKPOINT_FRACTIONS
            }
            for s, p in without_arm.items()
        },
    }
    villain = {
        "with": {
            s: float(_ckpt_at(p, TERMINAL_FRAC)["source_self"]["delta_g_mean"])
            for s, p in with_arm.items()
        },
        "without": {
            s: float(_ckpt_at(p, TERMINAL_FRAC)["source_self"]["delta_g_mean"])
            for s, p in without_arm.items()
        },
    }

    def _three_space(payload: dict, persona: str) -> dict | None:
        ck = _ckpt_at(payload, TERMINAL_FRAC)
        recs = ck["held_out"].get(persona)
        if not recs:
            return None
        g = np.array([float(leaf["g_logp"]) for leaf in recs.values()])
        b = np.array([float(leaf["b_logp"]) for leaf in recs.values()])
        return {
            "delta_logp_mean": float(np.mean(g - b)),
            "delta_margin_mean": _persona_mean(ck, persona, "delta_margin"),
            "delta_z_marker_mean": _persona_mean(ck, persona, "delta_z_marker"),
            "delta_p_mean": float(np.mean(np.exp(g) - np.exp(b))),
            "delta_logZ_mean": float(
                np.mean([float(leaf["logZ_g"]) - float(leaf["logZ_b"]) for leaf in recs.values()])
            ),
        }

    three_space = {
        "with": {s: {"qwen_default": _three_space(p, "qwen_default")} for s, p in with_arm.items()},
        "without": {
            s: {
                "qwen_default": _three_space(p, "qwen_default"),
                "assistant": _three_space(p, "assistant"),
            }
            for s, p in without_arm.items()
        },
    }

    def _strip(arm: dict[int, dict]) -> dict[str, dict[int, float]]:
        personas = sorted(
            set.intersection(
                *({p for p in payload["checkpoints"][0]["held_out"]} for payload in arm.values())
            )
        )
        out: dict[str, dict[int, float]] = {}
        for persona in personas:
            vals = {}
            for s, p in arm.items():
                v = normalized_shift(p, TERMINAL_FRAC, persona)
                if v is not None:
                    vals[s] = float(v - panel_median(p, TERMINAL_FRAC, centering))
            if vals:
                out[persona] = vals
        return out

    strip_with = _strip(with_arm)
    strip_without = _strip(without_arm)

    result = {
        "schema_version": "i610_analysis_v1",
        "chassis": chassis.name,
        "git_commit": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "parent_sweep": str(parent_sweep),
            "new_sweep": str(new_sweep),
            "manifest": str(manifest_path),
            "seeds": list(seeds),
            "chassis_slug": chassis.chassis_slug,
            "new_slug": chassis.new_slug,
            "replacement_persona": chassis.replacement,
        },
        "centering_set": centering,
        "n_centering": len(centering),
        "d_with_by_seed": d_with,
        "d_without_by_seed": d_without,
        "headline": headline,
        "finer_calibration_default_specific_gaps": finer,
        "rank_test": rank_test,
        "secondary_assistant": secondary,
        "sanity": sanity_summary,
        "exploratory": {
            "trajectory_centered_default": trajectory,
            "villain_dg_terminal": villain,
            "three_space_terminal": three_space,
            "per_persona_centered_strip_with": strip_with,
            "per_persona_centered_strip_without": strip_without,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(
        "headline: zone=%s median_without=%+.4f median_with=%+.4f (band=%.3f, p=%.3f)",
        headline["zone"],
        median_without,
        median_with,
        DECISION_BAND,
        rank_test["p_one_sided"],
    )
    _make_figures(result, figures_dir, chassis)
    return result


# ── Figures (hero + exploratory dump; the analyzer picks). ───────────────────


def _make_figures(
    result: dict, figures_dir: Path, chassis: ChassisConfig = CHASSES["mercenary"]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role, savefig_paper

    figures_dir.mkdir(parents=True, exist_ok=True)
    accent = paper_palette_role("accent")
    baseline = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")
    head = result["headline"]
    d_with = result["d_with_by_seed"]
    d_without = result["d_without_by_seed"]

    # Reader-facing persona labels (clean-result-critic Lens 2/3) — used in
    # every figure that surfaces persona names. Underscore-form slugs only
    # appear in the .meta.json sidecar + Reproducibility table; never in
    # rendered chart text.
    READER_LABELS = {
        "qwen_default": "Default assistant (untrained)",
        "assistant": "Explicit assistant persona",
        "hospice_nurse": "Hospice nurse",
        "journalist": "Journalist",
        "bartender": "Bartender",
        "french_person": "French person",
        "data_scientist": "Data scientist",
        "dictator": "Dictator",
        "pirate_captain": "Pirate captain",
        "child": "Child",
        "programmer": "Programmer",
        "librarian": "Librarian",
        "surgeon": "Surgeon",
        "medical_doctor": "Medical doctor",
    }

    def reader(p: str) -> str:
        return READER_LABELS.get(p, p.replace("_", " ").capitalize())

    # 1. Hero: per-seed strip, zones shaded.
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    lo = min([*d_with.values(), *d_without.values(), head["identity_threshold"]]) - 0.05
    hi = max([*d_with.values(), *d_without.values(), 0.02]) + 0.05
    ax.axhspan(lo, head["identity_threshold"], color=baseline, alpha=0.12)
    ax.axhspan(head["dose_threshold"], hi, color=accent, alpha=0.12)
    ax.axhline(0.0, color=neutral, lw=1.0, ls=":")
    # Spread seed labels with deterministic vertical offsets so overlapping
    # near-identical y-values stay legible (codex critic round-1 ask 1).
    for x, (_label, vals, color) in enumerate(
        [
            ("With-default mix\n(parent, trained default)", d_with, baseline),
            ("No-default mix\n(never-trained default)", d_without, accent),
        ]
    ):
        ys = list(vals.values())
        ax.scatter([x] * len(ys), ys, s=70, color=color, zorder=3)
        ax.scatter([x], [float(np.median(ys))], s=240, marker="_", color=color, zorder=4)
        sorted_seeds = sorted(vals.items(), key=lambda kv: kv[1])
        # Deterministic vertical fan-out for the seed labels to avoid overlap
        # when two seeds land within ~0.005 nat of each other.
        offsets = {
            sorted_seeds[0][0]: (10, -8),
            sorted_seeds[1][0]: (10, 0),
            sorted_seeds[2][0]: (10, 8),
        }
        for seed, y in vals.items():
            ax.annotate(
                str(seed), (x, y), xytext=offsets[seed], textcoords="offset points", fontsize=8
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["With-default mix\n(parent, trained default)", "No-default mix\n(never-trained default)"]
    )
    ax.set_ylabel("Centered, implant-normalized default-context shift")
    # Two-line title so the long claim never clips at the figure edges
    # (codex critic round-1 ask 1).
    ax.set_title(
        f"Default-context shielding — zone: {head['zone']}\n"
        f"(median {head['median_without']:+.3f} vs with-default {head['median_with']:+.3f})",
        fontsize=11,
    )
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    savefig_paper(fig, "hero_default_dose_strip", dir=figures_dir)
    plt.close(fig)

    # 2. Trajectory of the centered default read over checkpoints, both arms.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    traj = result["exploratory"]["trajectory_centered_default"]
    for arm, color in (("with", baseline), ("without", accent)):
        for seed, by_frac in traj[arm].items():
            fr = sorted(float(f) for f in by_frac)
            ax.plot(
                fr,
                [by_frac[f"{f:.2f}"] for f in fr],
                color=color,
                alpha=0.7,
                marker="o",
                ms=3,
                label=f"{arm} arm" if str(seed) == str(min(map(int, traj[arm]))) else None,
            )
    ax.axhline(0.0, color=neutral, lw=1.0, ls=":")
    ax.set_xlabel("Training fraction (of 63 matched steps)")
    ax.set_ylabel("Centered normalized default-context shift")
    ax.legend()
    savefig_paper(fig, "trajectory_default_centered", dir=figures_dir)
    plt.close(fig)

    # 3. Per-persona centered strip at terminal, both arms (raw alongside the
    # headline view: every persona, not just the default). Mark assistant /
    # default + the replacement, AND directly annotate the bottom-4 floor-
    # sharers in each arm by name so the prose claim about the floor cluster
    # is visible from the figure itself (codex critic round-1 ask 4).
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharey=True)
    for ax, key, title, color in (
        (axes[0], "per_persona_centered_strip_with", "With-default mix (parent)", baseline),
        (axes[1], "per_persona_centered_strip_without", "No-default mix", accent),
    ):
        strip = result["exploratory"][key]
        personas = sorted(strip, key=lambda p: np.median(list(strip[p].values())))
        for i, p in enumerate(personas):
            ys = list(strip[p].values())
            ax.scatter([i] * len(ys), ys, s=12, color=color, alpha=0.6)
        # Marker symbols for the registered named personas (legend uses
        # reader-facing labels — no underscore slugs in chart text).
        marks = [
            ("qwen_default", "*"),
            ("assistant", "D"),
            (chassis.replacement, "s"),
        ]
        for p, m in marks:
            if p in strip:
                i = personas.index(p)
                ax.scatter(
                    [i],
                    [float(np.median(list(strip[p].values())))],
                    s=140,
                    marker=m,
                    color="black",
                    zorder=4,
                    label=reader(p),
                )
        # Tag the four lowest-median personas DIRECTLY by name in a single
        # non-overlapping text block in the upper-left of the panel, so the
        # floor-sharer claim is verifiable from the figure (codex critic
        # round-1 ask 4). Order: lowest first.
        floor_four = personas[:4]
        floor_text_lines = ["Floor-cluster (4 lowest medians):"] + [
            f"  {rank + 1}. {reader(p)}   (median {float(np.median(list(strip[p].values()))):.3f})"
            for rank, p in enumerate(floor_four)
        ]
        ax.text(
            0.02,
            0.78,
            "\n".join(floor_text_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.6", "lw": 0.5},
        )
        ax.axhline(0.0, color=neutral, lw=1.0, ls=":")
        ax.set_title(title)
        ax.set_xticks([])
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    axes[0].set_ylabel("Centered normalized shift")
    fig.tight_layout()
    savefig_paper(fig, "per_persona_centered_strip", dir=figures_dir)
    plt.close(fig)

    # 4. Three-space columns for qwen_default (+ assistant, new arm).
    # Wider panels + extra bottom margin + slight rotation so the multi-line
    # x-tick text never clips or collides between adjacent columns (codex
    # critic round-1 ask 2).
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    spaces = [
        ("delta_logp_mean", "Δ log P(marker) (nats)"),
        ("delta_margin_mean", "Δ(z_marker − z_eos) (logits)"),
        ("delta_p_mean", "Δ P(marker)"),
    ]
    ts = result["exploratory"]["three_space_terminal"]
    series = [
        (
            "With-default\n(default trained)",
            baseline,
            [ts["with"][s]["qwen_default"] for s in ts["with"]],
        ),
        (
            "No-default\n(default untrained)",
            accent,
            [ts["without"][s]["qwen_default"] for s in ts["without"]],
        ),
        (
            "No-default\n(assistant persona)",
            neutral,
            [ts["without"][s]["assistant"] for s in ts["without"]],
        ),
    ]
    for ax, (field, label) in zip(axes, spaces, strict=True):
        for x, (_name, color, rows) in enumerate(series):
            ys = [r[field] for r in rows if r and r.get(field) is not None]
            if ys:
                ax.scatter([x] * len(ys), ys, s=45, color=color)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([n for n, _, _ in series], fontsize=8)
        ax.set_ylabel(label)
        ax.axhline(0.0, color=neutral, lw=0.8, ls=":")
        ax.set_xlim(-0.5, len(series) - 0.5)
    fig.subplots_adjust(bottom=0.22, wspace=0.32, left=0.06, right=0.98)
    savefig_paper(fig, "three_space_default_assistant", dir=figures_dir)
    plt.close(fig)

    # 5. Villain implant comparison (descriptive).
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vil = result["exploratory"]["villain_dg_terminal"]
    for x, (arm, color) in enumerate((("with", baseline), ("without", accent))):
        ys = list(vil[arm].values())
        ax.scatter([x] * len(ys), ys, s=60, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["With-default mix", "No-default mix"])
    ax.set_ylabel("Villain implant ΔG (nats)")
    savefig_paper(fig, "villain_implant_comparison", dir=figures_dir)
    plt.close(fig)

    # 6. Sanity dumbbells. Reader-facing tick labels and a chassis-named
    # two-line in-figure title that fits without clipping (codex critic
    # round-1 ask 3). The chassis-name slug ("software_engineer") becomes
    # a reader-facing phrase ("software engineer").
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    per = result["sanity"]["per_persona"]
    for i, (_persona, v) in enumerate(per.items()):
        mw = float(np.median(list(v["with_by_seed"].values())))
        mo = float(np.median(list(v["without_by_seed"].values())))
        ax.plot([i, i], [mw, mo], color=neutral, lw=1.5)
        ax.scatter([i], [mw], s=60, color=baseline, zorder=3)
        ax.scatter([i], [mo], s=60, color=accent, zorder=3)
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels([reader(p) for p in per], rotation=15)
    ax.axhline(0.0, color=neutral, lw=1.0, ls=":")
    ax.set_ylabel("Centered normalized shift (median over seeds)")
    chassis_phrase = chassis.name.replace("_", " ")
    ax.set_title(
        "Sanity personas: with-default (baseline) vs no-default (accent)\n"
        f"({chassis_phrase} chassis)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig_paper(fig, "sanity_dumbbells", dir=figures_dir)
    plt.close(fig)


# ── CLI. ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Task #610 registered comparison (VM, CPU)")
    ap.add_argument(
        "--chassis",
        choices=sorted(CHASSES),
        default="mercenary",
        help="Chassis registry key (v2 plan §2); re-points the comparator slug, sanity "
        "set, replacement precedent, and default output/figure paths.",
    )
    ap.add_argument("--parent-sweep", type=Path, default=Path("eval_results/issue_600/sweep"))
    ap.add_argument(
        "--new-sweep",
        type=Path,
        default=None,
        help="Default: <chassis output root>/sweep.",
    )
    ap.add_argument(
        "--manifest", type=Path, default=Path("eval_results/issue_600/panel_selection.json")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <chassis output root>/analysis/analysis.json.",
    )
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Default: the chassis figures dir (figures/issue_610[/<subprefix>]).",
    )
    args = ap.parse_args(argv)
    chassis = CHASSES[args.chassis]
    new_sweep = (
        args.new_sweep if args.new_sweep is not None else chassis.output_root_default / "sweep"
    )
    out = (
        args.out
        if args.out is not None
        else chassis.output_root_default / "analysis" / "analysis.json"
    )
    figures_dir = args.figures_dir if args.figures_dir is not None else chassis.figures_dir_default
    analyze_610(
        parent_sweep=args.parent_sweep,
        new_sweep=new_sweep,
        manifest_path=args.manifest,
        out_path=out,
        figures_dir=figures_dir,
        chassis=chassis,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
