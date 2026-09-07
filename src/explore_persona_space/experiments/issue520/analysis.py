"""DV1-DV5 aggregation for task #520 — additivity-of-shifts analysis.

Per plan §6:

- **DV1 (primary):** per-context cosine
  ``cos(shift_{A+B}(c), shift_A(c) + shift_B(c))`` at L20 post-response slot,
  on-policy.
- **DV2 (interference structure):** 2-pair near-vs-far median contrast on
  the normalized residual
  ``||shift_{A+B}(c) - (shift_A(c) + shift_B(c))|| / ||shift_{A+B}(c)||``.
- **DV3 (magnitude additivity):**
  ``|Delta log P(marker)_{A+B}(c) - (Delta log P_A(c) + Delta log P_B(c))|``
  centered at zero; spread < 1.0 nat for the far pair (H3).
- **DV4 (emission-rate sanity anchor):** source-self emission >= 0.5
  (implant-took sanity).
- **DV5 (strength-match diagnostic):** per-arm
  ``|g_logprob_singleton_X - g_logprob_joint_at_X|`` < 1.0 nat AND
  ``|emission_X_singleton - emission_X_joint|`` < 0.1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from explore_persona_space.experiments.issue520.shift_extract import cosine, norm

logger = logging.getLogger(__name__)


def _vec_add(u: list[float], v: list[float]) -> list[float]:
    return [a + b for a, b in zip(u, v, strict=True)]


def _vec_sub(u: list[float], v: list[float]) -> list[float]:
    return [a - b for a, b in zip(u, v, strict=True)]


def load_cell_extraction(path: Path) -> dict:
    """Load a per-cell extraction JSON written by ``shift_extract.write_cell_extraction``."""
    with open(path) as f:
        return json.load(f)


def compute_dv1_per_context(
    shift_A: dict[str, dict],
    shift_B: dict[str, dict],
    shift_AB: dict[str, dict],
    *,
    layer_key: str = "shift_primary",
) -> dict[str, float]:
    """Per-context DV1: cos(shift_{A+B}(c), shift_A(c) + shift_B(c)).

    All three inputs are ``per_persona_shift`` dicts (persona -> {shift_primary,
    shift_secondary, delta_log_p_marker}). Returns ``{persona: cosine}``.
    """
    out: dict[str, float] = {}
    for persona in shift_AB:
        if persona not in shift_A or persona not in shift_B:
            continue
        s_a = shift_A[persona][layer_key]
        s_b = shift_B[persona][layer_key]
        s_ab = shift_AB[persona][layer_key]
        pred = _vec_add(s_a, s_b)
        out[persona] = cosine(s_ab, pred)
    return out


def compute_dv2_per_context(
    shift_A: dict[str, dict],
    shift_B: dict[str, dict],
    shift_AB: dict[str, dict],
    *,
    layer_key: str = "shift_primary",
) -> dict[str, float]:
    """Per-context DV2: ||shift_{A+B} - (shift_A + shift_B)|| / ||shift_{A+B}||."""
    out: dict[str, float] = {}
    for persona in shift_AB:
        if persona not in shift_A or persona not in shift_B:
            continue
        s_a = shift_A[persona][layer_key]
        s_b = shift_B[persona][layer_key]
        s_ab = shift_AB[persona][layer_key]
        pred = _vec_add(s_a, s_b)
        residual = _vec_sub(s_ab, pred)
        denom = norm(s_ab)
        if denom == 0:
            continue
        out[persona] = norm(residual) / denom
    return out


def compute_dv3_per_context(
    shift_A: dict[str, dict],
    shift_B: dict[str, dict],
    shift_AB: dict[str, dict],
) -> dict[str, float]:
    """Per-context DV3: Delta log P(marker)_{A+B} - (Delta log P_A + Delta log P_B).

    Signed (NOT absolute) so the analyzer can see whether the joint OVER- or
    UNDER-shoots the sum.
    """
    out: dict[str, float] = {}
    for persona in shift_AB:
        if persona not in shift_A or persona not in shift_B:
            continue
        d_a = shift_A[persona]["delta_log_p_marker"]
        d_b = shift_B[persona]["delta_log_p_marker"]
        d_ab = shift_AB[persona]["delta_log_p_marker"]
        out[persona] = d_ab - (d_a + d_b)
    return out


def compute_dv4_source_emission(
    trained_per_persona: dict[str, dict],
    sources: list[str],
) -> dict[str, float]:
    """DV4: source-self emission rate (>= 0.5 = implant took)."""
    return {s: trained_per_persona.get(s, {}).get("emission_rate", 0.0) for s in sources}


def compute_dv5_strength_match(
    singleton_per_persona_trained: dict[str, dict],
    joint_per_persona_trained: dict[str, dict],
    source_persona: str,
) -> dict[str, float]:
    """DV5: per-arm strength-match diagnostic for ``source_persona``.

    Returns the absolute gaps |g_logprob_singleton - g_logprob_joint| AT the
    source persona AND |emission_singleton - emission_joint| AT the source.
    Bands: g_logprob_gap < 1.0 nat AND emission_gap < 0.1 = strength-matched.
    """
    s_logp = singleton_per_persona_trained.get(source_persona, {}).get("log_p_marker_mean")
    j_logp = joint_per_persona_trained.get(source_persona, {}).get("log_p_marker_mean")
    s_emit = singleton_per_persona_trained.get(source_persona, {}).get("emission_rate")
    j_emit = joint_per_persona_trained.get(source_persona, {}).get("emission_rate")
    out: dict[str, float] = {}
    if s_logp is not None and j_logp is not None:
        out["g_logprob_gap"] = abs(s_logp - j_logp)
    if s_emit is not None and j_emit is not None:
        out["emission_gap"] = abs(s_emit - j_emit)
    return out


# ── Aggregation across seeds + arms -----------------------------------------


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    sx = sorted(xs)
    n = len(sx)
    if n % 2 == 1:
        return sx[n // 2]
    return 0.5 * (sx[n // 2 - 1] + sx[n // 2])


def aggregate_pair(
    cells_by_slug: dict[str, dict],
    pair_name: str,
    source_a: str,
    source_b: str,
    *,
    ratio: str,
) -> dict:
    """Aggregate DV1-DV5 for one (pair, ratio) across seeds.

    ``cells_by_slug`` maps ``"{pair}_{arm}_{ratio}_seed{S}"`` -> loaded
    extraction payload. We compute per-seed DVs then aggregate (median +
    raw per-seed list).
    """
    per_seed: dict[int, dict] = {}
    arms = ("A_only", "B_only", "joint")

    seeds_seen: set[int] = set()
    for slug in cells_by_slug:
        # parse seed from slug suffix
        if "seed" not in slug:
            continue
        seed_str = slug.split("seed", 1)[1]
        try:
            seed = int(seed_str)
            seeds_seen.add(seed)
        except ValueError:
            continue

    for seed in sorted(seeds_seen):
        slugs = {arm: f"{pair_name}_{arm}_{ratio}_seed{seed}" for arm in arms}
        if not all(s in cells_by_slug for s in slugs.values()):
            continue
        ext_a = cells_by_slug[slugs["A_only"]]
        ext_b = cells_by_slug[slugs["B_only"]]
        ext_ab = cells_by_slug[slugs["joint"]]
        sa = ext_a["per_persona_shift"]
        sb = ext_b["per_persona_shift"]
        sab = ext_ab["per_persona_shift"]
        dv1 = compute_dv1_per_context(sa, sb, sab)
        dv2 = compute_dv2_per_context(sa, sb, sab)
        dv3 = compute_dv3_per_context(sa, sb, sab)
        dv4 = compute_dv4_source_emission(
            ext_ab["per_persona_trained"], sources=[source_a, source_b]
        )
        dv5_a = compute_dv5_strength_match(
            ext_a["per_persona_trained"],
            ext_ab["per_persona_trained"],
            source_persona=source_a,
        )
        dv5_b = compute_dv5_strength_match(
            ext_b["per_persona_trained"],
            ext_ab["per_persona_trained"],
            source_persona=source_b,
        )
        per_seed[seed] = {
            "dv1_cosine_per_context": dv1,
            "dv2_residual_per_context": dv2,
            "dv3_logp_per_context": dv3,
            "dv4_source_emission": dv4,
            "dv5_strength_match_A": dv5_a,
            "dv5_strength_match_B": dv5_b,
            "dv1_median": _median(list(dv1.values())),
            "dv2_median": _median(list(dv2.values())),
            "dv3_median_abs": _median([abs(v) for v in dv3.values()]),
        }

    # Cross-seed medians of the per-seed medians.
    if per_seed:
        cross_dv1 = _median([per_seed[s]["dv1_median"] for s in per_seed])
        cross_dv2 = _median([per_seed[s]["dv2_median"] for s in per_seed])
        cross_dv3 = _median([per_seed[s]["dv3_median_abs"] for s in per_seed])
    else:
        cross_dv1 = float("nan")
        cross_dv2 = float("nan")
        cross_dv3 = float("nan")

    return {
        "pair_name": pair_name,
        "ratio": ratio,
        "source_a": source_a,
        "source_b": source_b,
        "per_seed": per_seed,
        "cross_seed_median_dv1": cross_dv1,
        "cross_seed_median_dv2": cross_dv2,
        "cross_seed_median_abs_dv3": cross_dv3,
        "n_seeds": len(per_seed),
    }


def aggregate_all(
    cell_paths: list[Path],
    *,
    far_pair: tuple[str, str],
    near_pair: tuple[str, str] | None,
    include_b2_far: bool = True,
) -> dict:
    """Load every per-cell extraction and aggregate into a single analysis.json payload."""
    cells_by_slug: dict[str, dict] = {}
    for p in cell_paths:
        payload = load_cell_extraction(p)
        cell = payload.get("cell", {})
        # arm_slug already contains pair_arm_ratio, so use it directly.
        arm_slug = cell.get("arm_slug", "?")
        seed = cell.get("seed", "?")
        slug = f"{arm_slug}_seed{seed}"
        cells_by_slug[slug] = payload

    out: dict = {"by_pair_ratio": {}, "n_cells_loaded": len(cells_by_slug)}

    fa, fb = far_pair
    out["by_pair_ratio"]["far_b1"] = aggregate_pair(cells_by_slug, "far", fa, fb, ratio="b1")
    if include_b2_far:
        out["by_pair_ratio"]["far_b2"] = aggregate_pair(cells_by_slug, "far", fa, fb, ratio="b2")
    if near_pair is not None:
        na, nb = near_pair
        out["by_pair_ratio"]["near_b1"] = aggregate_pair(cells_by_slug, "near", na, nb, ratio="b1")

    # H1 verdict (beta-1 far): median DV1 >= 0.85 AND >= 80% of contexts > 0.5.
    far_b1 = out["by_pair_ratio"]["far_b1"]
    if far_b1["per_seed"]:
        all_far_b1_dv1 = []
        for payload in far_b1["per_seed"].values():
            all_far_b1_dv1.extend(payload["dv1_cosine_per_context"].values())
        if all_far_b1_dv1:
            median_dv1 = _median(all_far_b1_dv1)
            frac_above_half = sum(1 for v in all_far_b1_dv1 if v > 0.5) / len(all_far_b1_dv1)
            out["H1_b1_verdict"] = {
                "median_dv1_far": median_dv1,
                "frac_contexts_above_0.5": frac_above_half,
                "PASS": median_dv1 >= 0.85 and frac_above_half >= 0.8,
            }

    # H2 verdict (2-pair contrast): near_b1 median DV2 >= 2x far_b1 median DV2.
    if near_pair is not None and out["by_pair_ratio"]["near_b1"]["per_seed"] and far_b1["per_seed"]:
        far_b1_dv2 = []
        near_b1_dv2 = []
        for payload in far_b1["per_seed"].values():
            far_b1_dv2.extend(payload["dv2_residual_per_context"].values())
        for payload in out["by_pair_ratio"]["near_b1"]["per_seed"].values():
            near_b1_dv2.extend(payload["dv2_residual_per_context"].values())
        if far_b1_dv2 and near_b1_dv2:
            med_far = _median(far_b1_dv2)
            med_near = _median(near_b1_dv2)
            out["H2_verdict"] = {
                "median_dv2_far": med_far,
                "median_dv2_near": med_near,
                "ratio_near_over_far": med_near / med_far if med_far > 0 else float("inf"),
                "PASS": med_near >= 2 * med_far,
            }

    return out


def write_analysis(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote analysis to %s", out_path)
