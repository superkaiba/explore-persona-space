# em-dash intentional
"""Task #601 follow-up round 1 (posonly-multiepoch-schedule-closure) registry pins.

Plan v4 §3: ONE new cell ``posonly_200p_T130`` (200 positives, 0 negatives,
10 epochs -> T=130 ~= the matched arm's 128) + the ``phase4b`` group guard
(conditional non-phase4 cells must never leak into a phase4b dispatch).
"""

from explore_persona_space.experiments.neg_setpoint_601 import (
    LEARNING_RATE,
    cell_by_slug,
    cells_for_request,
)

SLUG = "posonly_200p_T130"
LABEL = "posonly-multiepoch-schedule-closure"


def test_followup1_cell_registered_with_schedule_match():
    spec = cell_by_slug(SLUG)
    assert spec.pos_ex == 200
    assert spec.n_neg_personas == 0
    assert spec.neg_ex_per_persona == 0
    assert spec.total_rows == 200
    assert spec.epochs == 10
    # 200 rows / eff. batch 16 -> ceil = 13 steps/epoch; x10 epochs = 130
    # (vs the matched arm ratio4to1_100p400n_T128's 128 — |dT| = 2 steps).
    assert spec.expected_steps == 130
    assert cell_by_slug("ratio4to1_100p400n_T128").expected_steps == 128


def test_followup1_cell_parent_parity_fields():
    spec = cell_by_slug(SLUG)
    assert spec.lr == LEARNING_RATE  # 1e-5, D2 parity — NOT a bridge-cell lr
    assert spec.lora_targets is None  # all-linear default
    assert spec.seeds == (42, 137)
    assert spec.onpolicy == "full6"  # matched-arm parity (47-persona panel)
    # Dense early ladder == the matched arm's (plan §3: matched-arm parity).
    assert spec.dense_steps == cell_by_slug("ratio4to1_100p400n_T128").dense_steps
    # D1: log-only band-stop — a firing stop would unmatch the schedule.
    assert spec.band_stop is True
    assert spec.band_log_only is True
    # The phase string IS the follow-up label: run_cell writes the cell dir at
    # <slab-root>/<phase>/<cell>_seed<S>, so this pins the artifacts contract
    # eval_results/issue_601/<label>/posonly_200p_T130_seed<S>/.
    assert spec.phase == LABEL


def test_followup1_cell_is_explicit_slug_only():
    spec = cell_by_slug(SLUG)
    assert spec.conditional is True
    # Excluded from `--cells all` (sweep re-runs must not pick it up)...
    assert SLUG not in {c.slug for c in cells_for_request("all")}
    assert SLUG not in {c.slug for c in cells_for_request(None)}
    # ...AND from the phase4b conditional group (the guard this round adds):
    # phase4b stays exactly the round-4 conditional bridge factor.
    assert [c.slug for c in cells_for_request("phase4b")] == ["posonly_attn_lr1e5"]
    # Explicit-slug launch resolves it (alone, and inside a CSV).
    assert [c.slug for c in cells_for_request(SLUG)] == [SLUG]
    assert {c.slug for c in cells_for_request(f"{SLUG},dense_200p0n")} == {SLUG, "dense_200p0n"}


def test_followup1_cell_does_not_change_parent_groups():
    # The parent sweep group is unchanged: 16 non-conditional cells
    # (6 phase1 units' 3 cells + 4 phase2 + 1 phase3 + 3 phase4 -> by slug).
    all_slugs = [c.slug for c in cells_for_request("all")]
    assert len(all_slugs) == len(set(all_slugs)) == 10
    assert all(cell_by_slug(s).phase in ("phase1", "phase2", "phase3", "phase4") for s in all_slugs)
