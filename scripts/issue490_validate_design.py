#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003, RUF046
"""Issue #490 PHASE 0 — design validation (CPU-only, one-time, before sweep).

Per plan v1 §4.5 PHASE 0 (post-v2 revision §1-3):

  * ASSERT marker token id == 83399 on Qwen-2.5-7B-Instruct tokenizer.
  * HARD PAIRWISE-DISJOINTNESS (POOL_16, NEGATIVES_FIXED, HELD_OUT_35).
  * Load 111-persona layer-20 cosine matrix; ASSERT pool radius ≤ 0.05.
  * Build 8 source-pairs (3 ARM-matched + 5 RNG-490 draws). For EACH pair,
    construct the on-axis intermediate-C subpanel (cos-dist ≤ τ to BOTH A
    and B, ≥5 personas) AND the distance-matched off-axis subpanel (same
    mean dist to {A,B}, high asymmetry, ≥5 personas). DROP+REDRAW any
    pair that cannot field both subpanels (deviation-allowed).
  * Re-run the subpanel construction under layer-21 (if available) and
    report the on-axis/off-axis OVERLAP with layer-20 as a robustness
    diagnostic; layer-20 stays primary per #478 parity.
  * Phase-0 POWER CALC: bootstrap the variance of within-pair Δ_geom from
    #478 cells restricted to the realized subpanel sizes; set
    escalate_to_3_seeds = True if power < 80% at the 0.5-nat threshold
    with n=16 (= 8 pairs × 2 seeds). If #478 result.json files are not on
    disk, fall back to a conservative analytic estimate using the within-
    pair pairing benefit (n_subpanel⁻½ scaling).

Output: ``data/issue_490/design_validation.json`` and
``data/issue_490/source_pairs.json`` (each pair carries on_axis[] + off_axis[]).
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue490_common import (  # noqa: E402
    ALL_PERSONAS,
    BASE_MODEL,
    ESCALATE_TO_3_SEEDS_AUTHORIZED,
    ESCALATED_SEEDS,
    HELD_OUT_35,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    N_TOTAL_PAIRS,
    NEGATIVES_FIXED,
    OFFAXIS_MIN_PERSONAS,
    ONAXIS_MIN_PERSONAS,
    PAIR_RNG_SEED,
    POOL_16,
    POOL_RADIUS_MAX,
    POWER_DELTA_GEOM_THRESHOLD_NATS,
    POWER_THRESHOLD_FRACTION,
    SEEDS,
    assert_marker_token_id,
    assert_pairwise_disjoint_sets,
    build_onaxis_offaxis_subpanels,
    build_source_pairs,
    load_all_persona_prompts,
    load_cosine_distance_matrix,
    load_cosine_distance_matrix_layer,
)


def _redraw_pair_pool(exclude: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """If a pair is dropped at Phase 0, draw a replacement from the same
    deterministic RNG-490 stream (skipping any already-excluded pair).

    Returns the FULL ordered list of candidate replacement pairs; the caller
    picks the first whose subpanels are feasible.
    """
    excluded = {tuple(sorted(p)) for p in exclude}
    all_pairs = list(combinations(POOL_16, 2))
    rng = np.random.default_rng(PAIR_RNG_SEED)
    # Pick MANY indices, not just 5 — we may need to walk further.
    n_candidates = min(len(all_pairs), 28)
    idx = rng.choice(len(all_pairs), size=n_candidates, replace=False)
    candidates = [tuple(sorted(all_pairs[int(i)])) for i in idx]
    # Filter out already-excluded.
    return [p for p in candidates if p not in excluded]


def _power_calc(onaxis_size: int, offaxis_size: int, n_tuples: int, delta_nats: float) -> dict:
    """Conservative analytic power estimate for within-pair Δ_geom.

    The headline statistic is a difference-of-differences on two subpanels.
    Per-read variance σ² is approximated from #478's per-tuple variance
    (~0.3 nat) scaled by √(35 / n_subpanel). The within-pair pairing
    benefit cancels pair-level offsets (assume ρ ≈ 0.5 between on-axis
    and off-axis means within the same pair → variance of the difference =
    2σ²(1 − ρ) ≈ σ²).

    Returns dict with effective per-tuple SD, SE of mean over n_tuples,
    z-score for the threshold, and power estimate (one-sided α=0.05).
    """
    base_sigma_per_persona = 0.30  # nats, from #478 per-tuple variance
    inflation_on = math.sqrt(35.0 / max(1, onaxis_size))
    inflation_off = math.sqrt(35.0 / max(1, offaxis_size))
    # Per-tuple SD of (gap_on - gap_off), assuming within-pair correlation 0.5.
    sigma_on_tuple = base_sigma_per_persona * inflation_on
    sigma_off_tuple = base_sigma_per_persona * inflation_off
    # Conservative: variances add (no negative covariance for the difference;
    # the pairing cancels pair-level mean offsets but not within-pair noise).
    sigma_delta_per_tuple = math.sqrt(sigma_on_tuple**2 + sigma_off_tuple**2)
    se_mean = sigma_delta_per_tuple / math.sqrt(max(1, n_tuples))
    # One-sided α=0.05 → critical z = 1.645
    z_alpha = 1.645
    # Effect size in SE units.
    z_effect = delta_nats / se_mean if se_mean > 0 else float("inf")
    # Power = Pr(Z_observed > z_alpha) when true mean = delta.
    # = 1 - Phi(z_alpha - z_effect)
    from math import erf, sqrt

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    power = 1.0 - _phi(z_alpha - z_effect)
    return {
        "method": "analytic_conservative_pairing",
        "onaxis_size": onaxis_size,
        "offaxis_size": offaxis_size,
        "n_tuples": n_tuples,
        "delta_nats": delta_nats,
        "base_sigma_per_persona_nats": base_sigma_per_persona,
        "sigma_delta_per_tuple_nats": sigma_delta_per_tuple,
        "se_mean_nats": se_mean,
        "z_alpha_one_sided": z_alpha,
        "z_effect": z_effect,
        "estimated_power_one_sided_alpha_0_05": power,
    }


def _power_calc_from_478(
    onaxis_size: int,
    offaxis_size: int,
    n_tuples: int,
    delta_nats: float,
    eval_dir_478: Path | None,
) -> dict | None:
    """Stronger power estimate: bootstrap Δ_geom SE from #478's actual cell
    × persona reads, restricted to subpanels of the realized sizes.

    Returns None if #478 data is not on disk (caller falls back to analytic).
    """
    if eval_dir_478 is None or not eval_dir_478.exists():
        return None
    files = sorted(eval_dir_478.glob("cell_K2_*_seed*/result.json"))
    if not files:
        return None

    # Collect per-(cell, seed) → {persona: deltaLogP_mean}
    per_cell_seed: dict[tuple[str, int], dict[str, float]] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        cell_id = data["cell_id"]
        seed = data["seed"]
        held_out = data.get("eval", {}).get("held_out", {})
        per_cell_seed[(cell_id, seed)] = {
            p: payload["deltaLogP_mean"] for p, payload in held_out.items()
        }

    if len(per_cell_seed) < 2:
        return None

    rng = np.random.default_rng(490)
    n_boots = 500
    delta_geom_samples: list[float] = []
    for _ in range(n_boots):
        # Mock a Δ_geom: for each (cell, seed) tuple, sample on-axis +
        # off-axis subpanels from the cell's 35 personas without replacement;
        # compute their means; difference is the "Δ_geom proxy" for this
        # tuple. Average over n_tuples sampled cell-seeds (with replacement).
        tuple_keys = list(per_cell_seed.keys())
        sampled_keys = [tuple_keys[i] for i in rng.choice(len(tuple_keys), size=n_tuples)]
        per_tuple_deltas = []
        for k in sampled_keys:
            personas = list(per_cell_seed[k].keys())
            if len(personas) < onaxis_size + offaxis_size:
                continue
            picks = rng.choice(len(personas), size=onaxis_size + offaxis_size, replace=False)
            on_personas = [personas[int(i)] for i in picks[:onaxis_size]]
            off_personas = [personas[int(i)] for i in picks[onaxis_size:]]
            on_mean = float(np.mean([per_cell_seed[k][p] for p in on_personas]))
            off_mean = float(np.mean([per_cell_seed[k][p] for p in off_personas]))
            per_tuple_deltas.append(on_mean - off_mean)
        if per_tuple_deltas:
            delta_geom_samples.append(float(np.mean(per_tuple_deltas)))

    if not delta_geom_samples:
        return None
    sd = float(np.std(delta_geom_samples, ddof=1))
    # SE of mean (already incorporated above via per-tuple averaging).
    z_alpha = 1.645
    z_effect = delta_nats / sd if sd > 0 else float("inf")
    from math import erf, sqrt

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    power = 1.0 - _phi(z_alpha - z_effect)
    return {
        "method": "bootstrap_from_478_cells",
        "n_478_cells": len(per_cell_seed),
        "n_boots": n_boots,
        "onaxis_size": onaxis_size,
        "offaxis_size": offaxis_size,
        "n_tuples": n_tuples,
        "delta_nats": delta_nats,
        "boot_sd_of_delta_geom": sd,
        "z_alpha_one_sided": z_alpha,
        "z_effect": z_effect,
        "estimated_power_one_sided_alpha_0_05": power,
    }


def main() -> int:
    out_dir = PROJECT_ROOT / "data" / "issue_490"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_validation.json"
    pairs_out_path = out_dir / "source_pairs.json"

    # ── (1) Marker token id assert ────────────────────────────────────────
    from transformers import AutoTokenizer

    log.info("Loading Qwen-2.5-7B-Instruct tokenizer (for marker assert) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)
    log.info("OK — marker %r encodes to single token id %d", MARKER_TEXT, MARKER_TOKEN_ID)

    # ── (2) Pairwise disjointness ─────────────────────────────────────────
    assert_pairwise_disjoint_sets()
    log.info(
        "OK — pairwise-disjoint: |POOL|=%d |NEG|=%d |HELD_OUT|=%d total=%d",
        len(POOL_16),
        len(NEGATIVES_FIXED),
        len(HELD_OUT_35),
        len(ALL_PERSONAS),
    )

    # ── (3) Persona prompts present ───────────────────────────────────────
    prompts = load_all_persona_prompts()
    log.info("OK — all %d personas have system prompts", len(ALL_PERSONAS))

    # ── (4) Distance matrix (layer 20 primary) ────────────────────────────
    log.info("Loading layer-20 cosine distance matrix ...")
    names20, dist20 = load_cosine_distance_matrix()
    needs_distance = list(POOL_16) + list(HELD_OUT_35)
    missing = [p for p in needs_distance if p not in names20]
    if missing:
        raise RuntimeError(
            f"Distance matrix (layer 20) missing personas: {missing!r}. "
            f"Matrix size={len(names20)}. Re-extract centroids."
        )
    pool_idxs = [names20.index(p) for p in POOL_16]
    pool_radius = max(dist20[i][j] for i in pool_idxs for j in pool_idxs)
    if pool_radius > POOL_RADIUS_MAX:
        raise RuntimeError(
            f"POOL_16 radius {pool_radius:.4f} > {POOL_RADIUS_MAX} — pool no longer tight."
        )
    log.info("OK — POOL_16 layer-20 radius = %.4f (≤ %.2f)", pool_radius, POOL_RADIUS_MAX)

    # ── (5) Layer-21 robustness matrix (best-effort) ──────────────────────
    log.info("Loading layer-21 cosine distance matrix (robustness diagnostic) ...")
    names21, dist21, layer21_source = load_cosine_distance_matrix_layer(21)
    layer21_available = layer21_source != "unavailable"
    if not layer21_available:
        log.warning(
            "Layer-21 centroids tensor NOT on disk (source=%r). The robustness "
            "diagnostic will record overlap=null and source=unavailable; layer-20 "
            "stays primary. To enable, run "
            "`uv run python scripts/analyze_100_persona_cosine.py --extract --gpu 0 "
            "--layer 21` on a pod once and re-run Phase 0.",
            layer21_source,
        )

    # ── (6) Build initial 8 source-pairs ──────────────────────────────────
    initial_pairs = build_source_pairs()
    log.info("Built %d initial source-pairs (3 ARM-matched + 5 RNG-490)", len(initial_pairs))

    # ── (7) Per-pair on-axis/off-axis subpanel construction ───────────────
    pairs_validated: list[dict] = []
    pairs_dropped: list[dict] = []
    excluded_pairs: list[tuple[str, str]] = []
    replacement_pool: list[tuple[str, str]] | None = None
    next_replacement_idx = 0

    queue = list(initial_pairs)

    while len(pairs_validated) < N_TOTAL_PAIRS and queue:
        pair = queue.pop(0)
        A, B = pair["A"], pair["B"]
        excluded_pairs.append((A, B))

        subpanels = build_onaxis_offaxis_subpanels(
            A=A,
            B=B,
            candidates=HELD_OUT_35,
            names=names20,
            distance=dist20,
        )

        if not subpanels["feasible"]:
            pairs_dropped.append({**pair, "drop_reason": subpanels["reason"]})
            log.warning(
                "Pair %s (%s, %s) DROPPED — %s",
                pair["pair_id"],
                A,
                B,
                subpanels["reason"],
            )
            # Pull a replacement if this pair was an RNG-490 draw (we never
            # drop a #478 ARM-matched pair; if one fails, fail loud).
            if pair["origin"] == "arm_matched_478":
                raise RuntimeError(
                    f"Pair {pair['pair_id']} (origin=arm_matched_478, "
                    f"matched_cell_id={pair['matched_cell_id']}) FAILED subpanel "
                    f"construction. Cannot drop — re-investigate HELD_OUT_35 vs "
                    f"this ARM-matched pair's geometry."
                )
            if replacement_pool is None:
                replacement_pool = _redraw_pair_pool(exclude=excluded_pairs)
            while next_replacement_idx < len(replacement_pool):
                replacement = replacement_pool[next_replacement_idx]
                next_replacement_idx += 1
                if replacement in excluded_pairs:
                    continue
                queue.append(
                    {
                        "pair_id": pair["pair_id"],  # reuse the slot id
                        "A": replacement[0],
                        "B": replacement[1],
                        "origin": "rng_490_replacement",
                        "matched_cell_id": None,
                    }
                )
                break
            continue

        # Layer-21 overlap (robustness diagnostic only). If the layer-21
        # centroids aren't on disk, record overlap=null (NOT a fake 1.0) so
        # the body call-out narrates "layer-21 unavailable" honestly.
        if layer21_available:
            layer21_subpanels = build_onaxis_offaxis_subpanels(
                A=A,
                B=B,
                candidates=HELD_OUT_35,
                names=names21,
                distance=dist21,
            )
            if layer21_subpanels["feasible"]:
                on_overlap = len(
                    set(subpanels["on_axis"]) & set(layer21_subpanels["on_axis"])
                ) / max(1, len(subpanels["on_axis"]))
                off_overlap = len(
                    set(subpanels["off_axis"]) & set(layer21_subpanels["off_axis"])
                ) / max(1, len(subpanels["off_axis"]))
            else:
                on_overlap = None
                off_overlap = None
        else:
            on_overlap = None
            off_overlap = None

        pairs_validated.append(
            {
                **pair,
                "tau_layer20": subpanels["tau"],
                "on_axis": subpanels["on_axis"],
                "off_axis": subpanels["off_axis"],
                "n_on_axis": len(subpanels["on_axis"]),
                "n_off_axis": len(subpanels["off_axis"]),
                "on_axis_mean_d_layer20": subpanels["on_axis_mean_d"],
                "off_axis_mean_d_layer20": subpanels["off_axis_mean_d"],
                "mean_d_match_delta_layer20": subpanels.get("mean_d_match_delta", float("nan")),
                "tolerance_used_layer20": subpanels.get("tolerance_used", float("nan")),
                "off_axis_selection_method": subpanels.get(
                    "off_axis_selection_method", "mean_d_matched"
                ),
                "layer21_overlap_on": on_overlap,
                "layer21_overlap_off": off_overlap,
                "on_axis_with_distances_layer20": [
                    {
                        "persona": rec[0],
                        "d_A": rec[1],
                        "d_B": rec[2],
                        "mean_d": rec[3],
                        "asym": rec[4],
                    }
                    for rec in subpanels["on_axis_personas_with_d"]
                ],
                "off_axis_with_distances_layer20": [
                    {
                        "persona": rec[0],
                        "d_A": rec[1],
                        "d_B": rec[2],
                        "mean_d": rec[3],
                        "asym": rec[4],
                    }
                    for rec in subpanels["off_axis_personas_with_d"]
                ],
            }
        )
        overlap_str = (
            f"layer21 overlap on={on_overlap:.2f} off={off_overlap:.2f}"
            if on_overlap is not None and off_overlap is not None
            else "layer21 overlap on=N/A off=N/A (centroids unavailable)"
        )
        method = subpanels.get("off_axis_selection_method", "?")
        log.info(
            "Pair %s (%s,%s) OK — τ=%.3f, n_on=%d (mean_d=%.4f), n_off=%d "
            "(mean_d=%.4f, Δ=%.4f, method=%s), %s",
            pair["pair_id"],
            A,
            B,
            subpanels["tau"],
            len(subpanels["on_axis"]),
            subpanels["on_axis_mean_d"],
            len(subpanels["off_axis"]),
            subpanels["off_axis_mean_d"],
            subpanels.get("mean_d_match_delta", float("nan")),
            method,
            overlap_str,
        )

    if len(pairs_validated) < N_TOTAL_PAIRS:
        raise RuntimeError(
            f"Phase 0 FAILED: only {len(pairs_validated)} of {N_TOTAL_PAIRS} pairs "
            f"could field both ≥{ONAXIS_MIN_PERSONAS} on-axis AND "
            f"≥{OFFAXIS_MIN_PERSONAS} off-axis subpanels from HELD_OUT_35. "
            f"Dropped: {[p['pair_id'] for p in pairs_dropped]}. "
            f"Either widen the off-axis tolerance, draw more replacement pairs, "
            f"or weaken the floor (NOT recommended; the floor exists to keep "
            f"the within-pair difference-of-differences stable per plan §6.2)."
        )

    # ── (8) Power calc ─────────────────────────────────────────────────────
    avg_onaxis = sum(p["n_on_axis"] for p in pairs_validated) / len(pairs_validated)
    avg_offaxis = sum(p["n_off_axis"] for p in pairs_validated) / len(pairs_validated)
    eval_dir_478 = PROJECT_ROOT / "eval_results" / "issue_478"
    power_478 = _power_calc_from_478(
        onaxis_size=int(round(avg_onaxis)),
        offaxis_size=int(round(avg_offaxis)),
        n_tuples=len(pairs_validated) * len(SEEDS),
        delta_nats=POWER_DELTA_GEOM_THRESHOLD_NATS,
        eval_dir_478=eval_dir_478,
    )
    power_analytic = _power_calc(
        onaxis_size=int(round(avg_onaxis)),
        offaxis_size=int(round(avg_offaxis)),
        n_tuples=len(pairs_validated) * len(SEEDS),
        delta_nats=POWER_DELTA_GEOM_THRESHOLD_NATS,
    )
    power_primary = power_478 if power_478 is not None else power_analytic
    estimated_power = power_primary["estimated_power_one_sided_alpha_0_05"]
    escalate_to_3_seeds = (
        ESCALATE_TO_3_SEEDS_AUTHORIZED and estimated_power < POWER_THRESHOLD_FRACTION
    )
    if escalate_to_3_seeds:
        # Re-estimate at n_tuples = 8 × 3 = 24.
        if power_478 is not None:
            power_3seeds = _power_calc_from_478(
                onaxis_size=int(round(avg_onaxis)),
                offaxis_size=int(round(avg_offaxis)),
                n_tuples=len(pairs_validated) * 3,
                delta_nats=POWER_DELTA_GEOM_THRESHOLD_NATS,
                eval_dir_478=eval_dir_478,
            )
        else:
            power_3seeds = _power_calc(
                onaxis_size=int(round(avg_onaxis)),
                offaxis_size=int(round(avg_offaxis)),
                n_tuples=len(pairs_validated) * 3,
                delta_nats=POWER_DELTA_GEOM_THRESHOLD_NATS,
            )
    else:
        power_3seeds = None

    log.info(
        "Power calc: method=%s, est power(n=%d, Δ=%.2f) = %.3f → escalate=%s",
        power_primary["method"],
        len(pairs_validated) * len(SEEDS),
        POWER_DELTA_GEOM_THRESHOLD_NATS,
        estimated_power,
        escalate_to_3_seeds,
    )

    # ── (9) Write design_validation.json + source_pairs.json ──────────────
    payload = {
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_TOKEN_ID,
        "pool_16": POOL_16,
        "negatives_fixed": NEGATIVES_FIXED,
        "held_out_35": HELD_OUT_35,
        "n_total_personas": len(ALL_PERSONAS),
        "pool_radius_layer20_cosine": pool_radius,
        "pair_rng_seed": PAIR_RNG_SEED,
        "pairs": pairs_validated,
        "pairs_dropped": pairs_dropped,
        "subpanel_thresholds": {
            "min_onaxis": ONAXIS_MIN_PERSONAS,
            "min_offaxis": OFFAXIS_MIN_PERSONAS,
        },
        "power_calc": {
            "primary": power_primary,
            "analytic_fallback": power_analytic,
            "from_478_bootstrap": power_478,
            "escalate_to_3_seeds": escalate_to_3_seeds,
            "escalated_seeds_n": 3 if escalate_to_3_seeds else 2,
            "power_at_3_seeds": power_3seeds,
            "threshold_delta_nats": POWER_DELTA_GEOM_THRESHOLD_NATS,
            "threshold_power_fraction": POWER_THRESHOLD_FRACTION,
        },
        "layer21_robustness_note": (
            "Layer-21 subpanels reported per-pair for diagnostic OVERLAP only; "
            "layer-20 is primary per #478 parity. layer21_overlap=null = "
            "layer-21 centroids not on disk (extract via "
            "analyze_100_persona_cosine.py --extract --layer 21 to enable)."
        ),
        "layer21_source": layer21_source,
        "prompt_lengths_chars": {p: len(prompts[p]) for p in ALL_PERSONAS},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)

    # source_pairs.json carries per-persona distances so the analyzer's
    # distance-adjusted regression (round-2 fix) can run without re-reading
    # design_validation.json. Each `on_axis_distances` / `off_axis_distances`
    # entry is {persona: {"d_A": ..., "d_B": ..., "mean_d": ..., "asym": ...}}.
    pairs_only = {
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "A": p["A"],
                "B": p["B"],
                "origin": p["origin"],
                "matched_cell_id": p["matched_cell_id"],
                "on_axis": p["on_axis"],
                "off_axis": p["off_axis"],
                "tau_layer20": p["tau_layer20"],
                "on_axis_mean_d_layer20": p["on_axis_mean_d_layer20"],
                "off_axis_mean_d_layer20": p["off_axis_mean_d_layer20"],
                "mean_d_match_delta_layer20": p["mean_d_match_delta_layer20"],
                "off_axis_selection_method": p["off_axis_selection_method"],
                "on_axis_distances": {
                    rec["persona"]: {
                        "d_A": rec["d_A"],
                        "d_B": rec["d_B"],
                        "mean_d": rec["mean_d"],
                        "asym": rec["asym"],
                    }
                    for rec in p["on_axis_with_distances_layer20"]
                },
                "off_axis_distances": {
                    rec["persona"]: {
                        "d_A": rec["d_A"],
                        "d_B": rec["d_B"],
                        "mean_d": rec["mean_d"],
                        "asym": rec["asym"],
                    }
                    for rec in p["off_axis_with_distances_layer20"]
                },
            }
            for p in pairs_validated
        ],
        "escalate_to_3_seeds": escalate_to_3_seeds,
        "seeds_resolved": (list(ESCALATED_SEEDS) if escalate_to_3_seeds else list(SEEDS)),
        "layer21_source": layer21_source,
    }
    pairs_out_path.write_text(json.dumps(pairs_only, indent=2))
    log.info("Wrote %s", pairs_out_path)
    log.info("Phase 0 design validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
