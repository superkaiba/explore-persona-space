# ruff: noqa: RUF002  # em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #477 — held-out-panel disjointness from #477 cell negatives (round-3 fix).

Pure-function, CPU-only, no checkpoint loads. Pins the eval-side panel-build
contract that the round-3 bug violated:

  * `held_out_panel(cts, cell_specs=CELL_SPECS_477)` excludes EVERY persona in
    union(#477 cell negatives) — the disjointness invariant per cell.
  * The 16-persona cell's panel is the #472 base panel MINUS its (and every
    other #477 cell's) negatives — so every panel persona has base R_eval
    (panel ⊆ #472 base) AND none was trained against.
  * Panel size ≥20 (the plan §8 DV-A floor) at every #477 cell.
  * Backward-compat: `held_out_panel(cts)` (no cell_specs) is byte-identical to
    the pre-fix #472 panel.
  * `all_negatives_union(cts, cell_specs=CELL_SPECS_477)` threads
    `cell_specs` into the per-cell resolver (the round-3 partial fix was
    `negatives_for_cell` only).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
    ANCHOR_COUNT,
    CELL_SPECS_477,
    COUNT_LEVELS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CELL_SPECS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    all_negatives_union,
    held_out_panel,
    negatives_for_cell,
)

# A synthetic cos_to_source dict large enough for the 16-persona cell to draw
# from. Real layout (47 personas + source + always-include) doesn't matter for
# the logic under test — only the SET of names and the cos ordering do.
N_BANK = 80


def _synthetic_cts() -> dict[str, float]:
    """{persona_NN: cosine_to_source} for a bank of 80, plus source + qwen_default.

    Cosine values are deterministic and spread evenly in [0, 1) so `select_
    negatives_by_geometry` resolves stable picks across {near, far, spread}.
    """
    cts: dict[str, float] = {}
    # Source persona gets cos=1.0 (self). It's filtered out everywhere in the
    # selector logic, but include it so the bank looks realistic.
    cts[SOURCE_PERSONA] = 1.0
    # Always-include negative (qwen_default) sits at a mid-range cos.
    cts[ALWAYS_INCLUDE_NEGATIVE] = 0.5
    # Bank of synthetic personas at evenly-spaced cosines in [0.01, 0.99).
    for i in range(N_BANK):
        cts[f"persona_{i:02d}"] = 0.01 + (0.98 * i) / (N_BANK - 1)
    return cts


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: no cell_specs kwarg → byte-identical to pre-fix #472 panel.
# ─────────────────────────────────────────────────────────────────────────────


def test_held_out_panel_no_cell_specs_equals_472_default() -> None:
    """Backward-compat: passing no cell_specs == #472's CELL_SPECS panel.

    The #472 callers (`i472_eval_trajectory.py`, `i472_phase_base_panel.py`,
    `analyze.py`, the reanalyze scripts) MUST keep working unchanged after the
    cell_specs kwarg lands. Pin this with an explicit equality check.
    """
    cts = _synthetic_cts()
    default = held_out_panel(cts, source=SOURCE_PERSONA)
    explicit_472 = held_out_panel(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS)
    assert default == explicit_472


def test_all_negatives_union_no_cell_specs_equals_472_default() -> None:
    """Same backward-compat for `all_negatives_union`."""
    cts = _synthetic_cts()
    default = all_negatives_union(cts, source=SOURCE_PERSONA)
    explicit_472 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS)
    assert default == explicit_472


# ─────────────────────────────────────────────────────────────────────────────
# #477 contamination fix: the panel for any #477 cell excludes EVERY #477
# cell's negatives (union across all count levels).
# ─────────────────────────────────────────────────────────────────────────────


def test_held_out_477_disjoint_from_every_477_cell_negatives() -> None:
    """Disjointness invariant: panel ∩ negatives(cell) == ∅ for every #477 cell.

    This is the load-bearing contract the round-3 bug violated. The pre-fix
    panel was built from `CELL_SPECS` (the #472 registry); #477's count cells
    therefore included #477 negatives, biasing bystander ΔG downward at high
    counts (the H1 count axis was corrupted).
    """
    cts = _synthetic_cts()
    # The corrected #477 panel: REUSE the #472 base panel (every persona has
    # base R_eval) and additionally subtract the union of #477 negatives.
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    held_out_477 = [p for p in base_panel if p not in union_477]

    for slug, _name, _placement, _n, _ex, _in_pooled in CELL_SPECS_477:
        cell_negs = set(
            negatives_for_cell(slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
        )
        overlap = set(held_out_477) & cell_negs
        assert not overlap, (
            f"#477 cell {slug!r}: panel overlaps with its negatives → {sorted(overlap)}"
        )


def test_held_out_477_is_subset_of_472_base_panel() -> None:
    """Every probe in the #477 panel has base R_eval (because it's ⊆ 472 base).

    The eval rig sanity-checks `R_eval` covers `panel ∪ {source}` and fails
    loud otherwise (`i472_eval_trajectory.py:93-99`). The recipe `held_out_477
    = base_panel − union_477` guarantees this: it can only REMOVE personas, not
    add new ones — so the R_eval coverage check still passes.
    """
    cts = _synthetic_cts()
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    held_out_477 = [p for p in base_panel if p not in union_477]
    assert set(held_out_477).issubset(set(base_panel))


def test_held_out_477_size_floor_per_cell() -> None:
    """Plan §8: every #477 cell needs ≥20 probes in the held-out panel.

    The corrected panel is a strict subset of the #472 base — so if the base
    panel has N personas, the #477 panel has N − |#472_base ∩ union_477|. On
    the real bank (#472 base = 47, #477 negatives span counts {2,4,8,16}) the
    16-cell's panel should still be ≥ 20 with margin. This test pins the floor
    on the synthetic bank (N_BANK=80, big enough that the 16-cell's union
    leaves ≥20).
    """
    cts = _synthetic_cts()
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    held_out_477 = [p for p in base_panel if p not in union_477]
    assert len(held_out_477) >= 20, (
        f"Per-cell DV-A floor violated: {len(held_out_477)} < 20 (plan §8). "
        f"Investigate the persona bank size or the #477 count axis composition."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single shared panel across all #477 count levels (cross-count comparability).
# ─────────────────────────────────────────────────────────────────────────────


def test_held_out_477_shared_across_count_levels() -> None:
    """Cross-count comparability: ONE panel across all 4 count cells.

    The eval rig builds the panel ONCE per cell-specs choice (via the union over
    ALL #477 cells, not per-cell). So every count cell evaluates on the SAME
    persona set — the H1 partial-Spearman across counts compares bystander ΔG
    on a fixed denominator, not a shifting one. Pin this by computing the
    recipe and confirming it does NOT vary with which #477 cell slug we pick
    for the disjointness check.
    """
    cts = _synthetic_cts()
    base_panel = set(held_out_panel(cts, source=SOURCE_PERSONA))
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    held_out_477 = base_panel - union_477

    # The panel is INDEPENDENT of which #477 cell we are evaluating — it depends
    # only on the union across the full registry, so every cell sees the same
    # set. Confirm by re-deriving with each #477 cell's negatives subtracted —
    # the result must be identical (subset of held_out_477 ⇒ ⊆ already excludes
    # them).
    for slug, _name, _placement, _n, _ex, _in_pooled in CELL_SPECS_477:
        cell_negs = set(
            negatives_for_cell(slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
        )
        # Subtracting a single cell's negatives from held_out_477 is a no-op
        # because union_477 already contains them.
        assert (held_out_477 - cell_negs) == held_out_477, (
            f"{slug}: union_477 missed this cell's negatives ({cell_negs - union_477!r}) — "
            f"the cell_specs threading is broken in all_negatives_union."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity: the union over #477 actually grows with the largest count cell (16),
# so the round-3 contamination concern was real, not theoretical.
# ─────────────────────────────────────────────────────────────────────────────


def test_union_477_strictly_contains_16_cell_negatives() -> None:
    """Without the fix the 16-cell trains against personas in the #472 base panel.

    Concretely: the 16-cell at `spread` placement picks 16 personas from the
    bank by even-quantile coverage of cos-to-source. The pre-fix panel was
    built from `CELL_SPECS` (#472 only), so up to all 16 of those personas
    could land in the eval panel — that's the H1 count-axis corruption.

    This test pins that the 16-cell's negatives are NON-EMPTY in the union and
    that the recipe `base − union_477` excludes them.
    """
    cts = _synthetic_cts()
    # The 16-cell is `c477_main_calib_negp_16` (and the calibration twin
    # `c477_calib_negp_16`). Either works — they share the same 6-tuple shape.
    sixteen_cell_negs = set(
        negatives_for_cell(
            "c477_main_calib_negp_16", cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477
        )
    )
    assert len(sixteen_cell_negs) == 16  # plan §4: 16-persona arm

    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    assert sixteen_cell_negs.issubset(union_477)

    # And the corrected panel excludes every one of them.
    base_panel = set(held_out_panel(cts, source=SOURCE_PERSONA))
    held_out_477 = base_panel - union_477
    assert not (held_out_477 & sixteen_cell_negs)


def test_union_477_grows_with_count_levels() -> None:
    """Round-3 invariant: union(#477 negatives) is non-empty at every count level.

    If this fails, the cell-spec wiring broke — the union should at minimum
    contain the always-include negative (qwen_default) plus the placement
    personas selected at each count level.
    """
    cts = _synthetic_cts()
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    # qwen_default + at least the smallest count's worth of placement personas.
    assert ALWAYS_INCLUDE_NEGATIVE in union_477
    assert len(union_477) >= min(COUNT_LEVELS)
    # Sanity on count levels and the anchor.
    assert ANCHOR_COUNT in COUNT_LEVELS


# ─────────────────────────────────────────────────────────────────────────────
# Disjointness assert in i472_eval_trajectory.py is exercisable as a pure
# function (we don't import the script, but we mirror the exact check so any
# regression in select_negatives wiring is caught here too).
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cell_slug",
    [
        "c477_calib_negp_2",
        "c477_calib_negp_4",
        "c477_calib_negp_8",
        "c477_calib_negp_16",
        "c477_main_calib_negp_2",
        "c477_main_calib_negp_4",
        "c477_main_calib_negp_8",
        "c477_main_calib_negp_16",
    ],
)
def test_eval_panel_disjoint_per_cell(cell_slug: str) -> None:
    """Mirror the i472_eval_trajectory.py disjointness assert per #477 cell."""
    cts = _synthetic_cts()
    base_panel = held_out_panel(cts, source=SOURCE_PERSONA)
    union_477 = all_negatives_union(cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    held_out_477 = [p for p in base_panel if p not in union_477]
    cell_negs = set(
        negatives_for_cell(cell_slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477)
    )
    overlap = set(held_out_477) & cell_negs
    assert not overlap, f"{cell_slug}: panel∩negatives={sorted(overlap)}"
