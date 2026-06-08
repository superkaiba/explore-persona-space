# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + × intentional
"""Task #504 — per-cell positioned-negative resolution + smoke-cell selection.

The #472 select_negatives module picks negatives by BAND (near/far/spread) over
ALL personas in the bank. #504 needs ONE specific persona per arm, picked at
Phase 0.5 to land cos(N, source) closest to the band center {0.7, 0.4, 0.1,
-0.2}. This module:

  - `select_positioned_negatives` — given `cos_to_source` + the 4 band centers,
    picks the closest-to-target persona per band (with uniqueness + default-
    persona exclusion).
  - `pick_smoke_mid_band_n` — picks the persona for the Phase 0 smoke whose
    cos(N, source) is closest to 0.4 (median of top half, plan §4.1).
  - `negatives_for_cell_504` — given the cell slug + arm-to-N mapping, returns
    [qwen_default, positioned_N] for the 4 positioned arms, [qwen_default]
    alone for the default-only arm.
  - `arm_negatives_with_counts` — also returns the `(neg_persona, n_ex)` pairs
    so the per-cell `build_cell_504` knows the split (100 + 100 for positioned;
    200 alone for default-only).

CPU-only.
"""

from __future__ import annotations

import logging

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    ALWAYS_INCLUDE_NEGATIVE,
    BAND_CENTERS,
    CELL_SPECS_504,
    DEFAULT_ARM_SLUG,
    DEFAULT_ARM_SLUG_V2,
    DEFAULT_ARM_SLUG_V3,
    NEG_EX_DEFAULT_ONLY_ARM,
    NEG_EX_PER_PERSONA,
    POSITIONED_ARM_SLUGS,
    POSITIONED_ARM_SLUGS_V2,
    POSITIONED_ARM_SLUGS_V3,
    SOURCE_PERSONA,
)

# Force-reference v2 + v3 imports so ruff F401 doesn't strip them on the
# formatter pre-commit pass — see `feedback_ruff_strips_unused_imports`. All
# are used in negatives_for_cell_504 / arm_negatives_with_counts below.
_V2_V3_IMPORT_REFS = (
    DEFAULT_ARM_SLUG_V2,
    POSITIONED_ARM_SLUGS_V2,
    DEFAULT_ARM_SLUG_V3,
    POSITIONED_ARM_SLUGS_V3,
)

log = logging.getLogger("issue_504.cell_resolution")


def select_positioned_negatives(
    cos_to_source: dict[str, float],
    *,
    source: str = SOURCE_PERSONA,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    band_centers: dict[str, float] | None = None,
) -> dict[str, str]:
    """Pick ONE persona per band as the arm's positioned negative.

    Per plan §4.2 step 2: for each of the 4 bands (near=0.7, mid_near=0.4,
    mid_far=0.1, far=-0.2), find the candidate persona whose cos(persona,
    source) is closest to the band center, excluding `source` and
    `default_persona`. Greedy disambiguation: bands are walked in fixed order
    (near → far) so each subsequent band's pick excludes the previous bands'
    picks (no two arms share a positioned N).

    Args:
        cos_to_source: {persona: cos(persona, source)} over the bank, at the
            chosen headline layer.
        source: source persona name (excluded from candidates).
        default_persona: the always-included default negative (excluded).
        band_centers: override the band → cos target mapping (default = the
            module-level `BAND_CENTERS`).

    Returns:
        {band_name: persona_name} for band_name in {"near","mid_near",
        "mid_far","far"}.

    Raises:
        ValueError: a band has no candidate persona (the bank is too small or
            the source/default exclusions emptied it).
    """
    centers = band_centers if band_centers is not None else BAND_CENTERS
    candidates = {p: c for p, c in cos_to_source.items() if p != source and p != default_persona}
    chosen: dict[str, str] = {}
    used: set[str] = set()
    # Walk bands in fixed order so the disambiguation is deterministic.
    for band in ("near", "mid_near", "mid_far", "far"):
        target = centers[band]
        # Pick the unused candidate whose cos is closest to the target.
        pool = {p: c for p, c in candidates.items() if p not in used}
        if not pool:
            raise ValueError(
                f"select_positioned_negatives: band {band!r} has no candidate persona "
                f"(bank exhausted after the prior bands). Grow the bank or relax "
                f"BAND_CENTERS."
            )
        best = min(pool.items(), key=lambda kv: abs(kv[1] - target))
        chosen[band] = best[0]
        used.add(best[0])
        log.info(
            "[positioned] band=%s target_cos=%.2f → %s (cos=%.4f)",
            band,
            target,
            best[0],
            best[1],
        )
    return chosen


def pick_smoke_mid_band_n(
    cos_to_source: dict[str, float],
    *,
    source: str = SOURCE_PERSONA,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    target_cos: float = 0.4,
) -> str:
    """Pick the Phase 0 smoke's mid-band positioned negative (plan §4.1).

    The smoke composition matches Phase 1 (200 pos + 200 neg = 100
    qwen_default + 100 positioned mid-band N). We pick the persona whose
    cos(N, source) is closest to `target_cos` (default 0.4 — the median of
    the top half, matching the planned `mid_near` arm).

    Returns the persona name; raises ValueError if no candidate exists.
    """
    pool = {p: c for p, c in cos_to_source.items() if p != source and p != default_persona}
    if not pool:
        raise ValueError(
            "pick_smoke_mid_band_n: no candidate persona after excluding source + default."
        )
    best = min(pool.items(), key=lambda kv: abs(kv[1] - target_cos))
    log.info("[smoke] smoke mid-band N: %s (cos=%.4f, target=%.2f)", best[0], best[1], target_cos)
    return best[0]


def negatives_for_cell_504(
    cell_slug: str,
    arm_to_positioned_n: dict[str, str],
    *,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    smoke_mid_band_n: str | None = None,
) -> list[str]:
    """Return the negative persona list for ONE #504 cell.

    Positioned arms (Near/Mid-Near/Mid-Far/Far): [default, positioned_N].
    Default-only arm: [default] alone.
    Smoke cells: [default, smoke_mid_band_n] — matches Phase 1 composition
    (100 default + 100 positioned), so `smoke_mid_band_n` is required.

    Args:
        cell_slug: one of the slugs in CELL_SPECS_504.
        arm_to_positioned_n: {arm_slug: positioned_negative_persona} for the
            4 positioned arms (output of `select_positioned_negatives` keyed
            by arm slug, NOT band name).
        default_persona: the always-included default (qwen_default).
        smoke_mid_band_n: the smoke cells' shared positioned-N (output of
            `pick_smoke_mid_band_n`). Required when cell_slug is a smoke slug.

    Returns:
        Ordered list of negative persona names. The first element is always
        `default_persona` (so downstream consumers can rely on the layout).
        Empty list never returned for #504 cells (every cell has ≥1 negative
        persona by design).

    Raises:
        KeyError: cell_slug not in CELL_SPECS_504.
        ValueError: arm_to_positioned_n missing the positioned arm's entry, or
            smoke cell requested without `smoke_mid_band_n`.
    """
    spec = next((c for c in CELL_SPECS_504 if c[0] == cell_slug), None)
    if spec is None:
        raise KeyError(f"Unknown #504 cell slug {cell_slug!r}")
    if cell_slug in (DEFAULT_ARM_SLUG, DEFAULT_ARM_SLUG_V2, DEFAULT_ARM_SLUG_V3):
        return [default_persona]
    if (
        cell_slug in POSITIONED_ARM_SLUGS
        or cell_slug in POSITIONED_ARM_SLUGS_V2
        or cell_slug in POSITIONED_ARM_SLUGS_V3
    ):
        if cell_slug not in arm_to_positioned_n:
            raise ValueError(
                f"negatives_for_cell_504: positioned arm {cell_slug!r} missing in "
                f"arm_to_positioned_n={sorted(arm_to_positioned_n)}"
            )
        return [default_persona, arm_to_positioned_n[cell_slug]]
    # Smoke cells share the SAME positioned-N (the mid-band one). Recognize
    # the v1 (`c504_smoke_*`), v2 (`c504v2_smoke_*`), and v3 (`c504v3_smoke_*`)
    # prefixes. The v3 smoke cells (`c504v3_smoke_eps{2,3}`) consume the same
    # smoke_mid_band_n as v1/v2 — the EPOCHS ladder varies optimization steps,
    # NOT the negative composition (still 100 default + 100 positioned mid-band).
    if cell_slug.startswith(("c504_smoke_", "c504v2_smoke_", "c504v3_smoke_")):
        if smoke_mid_band_n is None:
            raise ValueError(
                f"negatives_for_cell_504: smoke cell {cell_slug!r} requires "
                f"`smoke_mid_band_n` (Phase 0 picks it from the bank)."
            )
        return [default_persona, smoke_mid_band_n]
    raise KeyError(f"Unhandled #504 cell slug {cell_slug!r}")


def arm_negatives_with_counts(
    cell_slug: str,
    arm_to_positioned_n: dict[str, str],
    *,
    default_persona: str = ALWAYS_INCLUDE_NEGATIVE,
    smoke_mid_band_n: str | None = None,
    n_pos: int = 200,
) -> tuple[list[str], list[int]]:
    """Return (negative_personas, n_ex_per_persona) for ONE #504 cell.

    Plan §4.3 + §5: positioned arms have 100 + 100 = 200 total negs (split
    evenly between qwen_default and the positioned N). The default-only arm
    has 200 from qwen_default alone. Smoke cells match Phase 1 (100 + 100).

    Returns:
        (negs, counts) where len(negs) == len(counts) and sum(counts) is the
        total negative-row budget for the cell. The pair is the canonical
        cell input to `build_cell_504`.
    """
    negs = negatives_for_cell_504(
        cell_slug,
        arm_to_positioned_n,
        default_persona=default_persona,
        smoke_mid_band_n=smoke_mid_band_n,
    )
    if cell_slug in (DEFAULT_ARM_SLUG, DEFAULT_ARM_SLUG_V2, DEFAULT_ARM_SLUG_V3):
        # Single negative persona x NEG_EX_DEFAULT_ONLY_ARM ex (matches positioned
        # arms' total neg row count for cross-arm step parity).
        return negs, [NEG_EX_DEFAULT_ONLY_ARM]
    # Positioned arms + smoke cells: split evenly (1:1 between the two negative
    # personas — qwen_default + positioned N).
    return negs, [NEG_EX_PER_PERSONA] * len(negs)
