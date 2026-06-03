# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 — NEW distance-stratified negative selector (plan §4.5).

Neither existing selector is distance-aware:
``generate_leakage_data.select_negative_personas`` is RANDOM (rng.sample);
#448's ``persona_registry.select_n_bystanders`` is SHA-256-deterministic. This
module selects negatives by base-model centroid-cosine distance to the source so
the placement arms (near/far/spread) move ``d_nearest_neg`` while holding
``d_source`` fixed per held-out probe — the cross-arm shift that breaks the
two-distance collinearity (plan §3 identifiability point).

``qwen_default`` (the bare default-instruct system prompt) is ALWAYS a negative
in every non-empty arm (plan §4.5; leakage to the default context is the safety
target, open-q 3.7).

The held-out panel for the geometry regression is the bank MINUS source MINUS the
UNION of negatives across ALL pooled-regression arms (so every probe is held-out
in every arm; plan §4.2 / §4.5).
"""

from __future__ import annotations

import logging

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    SOURCE_PERSONA,
)

log = logging.getLogger("issue_472.select_negatives")

VALID_PLACEMENTS = ("near", "far", "spread", "none")


def select_negatives_by_geometry(
    source: str,
    placement: str,
    n_personas: int,
    cos_to_source: dict[str, float],
    *,
    always_include: tuple[str, ...] = (ALWAYS_INCLUDE_NEGATIVE,),
) -> list[str]:
    """Select ``n_personas`` negatives by base-model cosine distance to ``source``.

    Plan §4.5 pseudocode. ``cos_to_source`` maps persona -> cos(persona, source)
    (high cos = near source). ``qwen_default`` (always_include) is prepended and
    counts toward ``n_personas`` (so a 4-persona arm = qwen_default + 3 placement
    personas).

    Args:
        source: source persona name (excluded from candidates).
        placement: one of {"near","far","spread","none"}.
        n_personas: total negative personas INCLUDING always_include.
        cos_to_source: {persona: cos(persona, source)} over the bank.
        always_include: personas always present as negatives (default
            qwen_default). These count toward n_personas.

    Returns:
        Ordered list of negative persona names: [*always_include, *chosen].
        Empty list for placement="none".

    Raises:
        ValueError on unknown placement OR if too few candidates exist.
    """
    if placement not in VALID_PLACEMENTS:
        raise ValueError(f"Unknown placement {placement!r}; expected one of {VALID_PLACEMENTS}.")
    if placement == "none":
        return []

    always = [p for p in always_include if p != source]
    n_choose = n_personas - len(always)
    if n_choose < 0:
        raise ValueError(
            f"n_personas={n_personas} < len(always_include)={len(always)} for "
            f"placement={placement!r}; cannot select a negative arm."
        )
    if n_choose == 0:
        return list(always)

    # Candidates = bank minus source minus always-included.
    cands = [p for p in cos_to_source if p != source and p not in always]
    # Sort by cosine to source DESCENDING (high cos = near source).
    cands.sort(key=lambda p: cos_to_source[p], reverse=True)

    if n_choose > len(cands):
        raise ValueError(
            f"placement={placement!r} needs {n_choose} candidates but only "
            f"{len(cands)} available (bank too small after excluding source + "
            f"always_include). Grow the persona bank (plan §4.2)."
        )

    if placement == "near":
        chosen = cands[:n_choose]
    elif placement == "far":
        chosen = cands[-n_choose:]
    elif placement == "spread":
        # Even quantile coverage of the cos range.
        idx = np.linspace(0, len(cands) - 1, n_choose).round().astype(int)
        # Deduplicate indices (can collide when n_choose is large vs len).
        seen: set[int] = set()
        chosen = []
        for i in idx:
            ii = int(i)
            while ii in seen and ii < len(cands) - 1:
                ii += 1
            if ii in seen:  # walked off the end; back-fill from the start
                ii = next(k for k in range(len(cands)) if k not in seen)
            seen.add(ii)
            chosen.append(cands[ii])
    else:  # pragma: no cover - guarded above
        raise ValueError(placement)

    return [*always, *chosen]


def negatives_for_cell(
    cell_slug: str,
    cos_to_source: dict[str, float],
    *,
    source: str = SOURCE_PERSONA,
    cell_specs: tuple | None = None,
) -> list[str]:
    """Resolve the negative persona list for a single cell from CELL_SPECS.

    Args:
        cell_slug: cell to resolve.
        cos_to_source: {persona: cos(persona, source)}.
        source: source persona.
        cell_specs: OPTIONAL override registry (same 6-tuple shape as
            CELL_SPECS). Defaults to #472's CELL_SPECS. Used by #477 to drive
            this resolver against its own CELL_SPECS_477 without touching the
            #472 module's authoritative tuple. Pure backward-compat: callers
            that don't pass this kwarg keep the existing behavior.
    """
    specs = cell_specs if cell_specs is not None else CELL_SPECS
    spec = next((c for c in specs if c[0] == cell_slug), None)
    if spec is None:
        raise KeyError(f"Unknown cell slug {cell_slug!r}; known: {[c[0] for c in specs]}")
    _slug, _name, placement, n_neg_personas, _neg_ex, _in_pooled = spec
    return select_negatives_by_geometry(source, placement, n_neg_personas, cos_to_source)


def all_negatives_union(
    cos_to_source: dict[str, float],
    *,
    source: str = SOURCE_PERSONA,
    pooled_only: bool = False,
) -> set[str]:
    """Union of negatives across cells.

    Args:
        cos_to_source: {persona: cos(persona, source)}.
        source: source persona.
        pooled_only: if True, union ONLY over the pooled-regression cells
            (in_pooled=True); else over ALL cells. The held-out panel uses the
            union over ALL cells so probes are held-out in every arm (plan §4.5).
    """
    union: set[str] = set()
    for slug, _name, _placement, _n, _ex, in_pooled in CELL_SPECS:
        if pooled_only and not in_pooled:
            continue
        union.update(negatives_for_cell(slug, cos_to_source, source=source))
    return union


def held_out_panel(
    cos_to_source: dict[str, float],
    *,
    source: str = SOURCE_PERSONA,
) -> list[str]:
    """Held-out probe panel = bank − source − union(negatives across ALL cells).

    Every persona here is a never-trained-against bystander in EVERY arm, so its
    ``d_source`` is fixed across arms while ``d_nearest_neg`` shifts between
    Near/Far/Spread — the load-bearing identifiability property (plan §4.5).
    """
    union = all_negatives_union(cos_to_source, source=source, pooled_only=False)
    excluded = union | {source}
    panel = sorted(p for p in cos_to_source if p not in excluded)
    return panel


def d_source(probe: str, cos_to_source: dict[str, float]) -> float:
    """Distance-to-source covariate = 1 − cos(probe, source)."""
    return 1.0 - cos_to_source[probe]


def d_nearest_neg(
    probe: str,
    arm_negatives: list[str],
    cos_matrix: dict[str, dict[str, float]],
    *,
    exclude_default: bool = False,
    default_name: str = ALWAYS_INCLUDE_NEGATIVE,
) -> float:
    """Distance from ``probe`` to the NEAREST negative in ``arm_negatives``.

    d_nearest_neg(arm) = min over arm negatives of (1 − cos(probe, neg)).

    Args:
        probe: held-out probe persona.
        arm_negatives: the arm's negative persona list.
        cos_matrix: {a: {b: cos(a, b)}} over the bank.
        exclude_default: if True, exclude ``default_name`` (qwen_default) from
            the min — yields the non-default variant ``d_nearest_neg_nd`` (plan
            §4.3 / §6 identification gate).
        default_name: the always-included default persona to optionally exclude.

    Returns:
        min distance, or NaN if the arm has no eligible negatives (e.g. the
        no-negatives arm, or a non-default-only computation on a default-only
        arm).
    """
    negs = arm_negatives
    if exclude_default:
        negs = [n for n in negs if n != default_name]
    if not negs:
        return float("nan")
    dists = [1.0 - cos_matrix[probe][neg] for neg in negs]
    return float(min(dists))
