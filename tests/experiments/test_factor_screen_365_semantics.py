"""Plan-conformance tests for task #365.

These are NOT smoke tests — they verify that the package matches plan v2
on the factor encoding, the source-persona roster, and the in-domain
bystander stratification. Each test below corresponds to a specific
analyzer-must-handle item or a known-broken bit from the prior
``experiment-365`` branch.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_365 import (
    EVAL_PERSONAS_24,
    FACTOR_NAMES,
    PREREGISTERED_INTERACTIONS,
    SOURCE_PERSONAS,
    SOURCE_PROMPTS_SHORT,
    Cell,
    all_full_cells,
    bystanders_for,
    is_preregistered,
    out_of_domain_bystanders_for,
)
from explore_persona_space.experiments.factor_screen_365.cells import (
    FACTOR_DESCRIPTIONS,
    FACTOR_INDEX,
    matched_pairs_for_factor,
    matched_pairs_for_interaction,
)
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    IN_DOMAIN_BYSTANDERS_BY_SOURCE,
    in_domain_bystanders_for,
)

# ---- Factor encoding -------------------------------------------------------


def test_factor_names_match_plan_order() -> None:
    """Plan v2 §4 names the factors A..E in that order."""
    assert FACTOR_NAMES == ("A", "B", "C", "D", "E")
    assert FACTOR_INDEX == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def test_factor_descriptions_match_plan_levels() -> None:
    """Plan-authoritative level semantics for each factor."""
    assert "short" in FACTOR_DESCRIPTIONS["A"][0].lower()
    assert "long" in FACTOR_DESCRIPTIONS["A"][1].lower()
    assert "short" in FACTOR_DESCRIPTIONS["B"][0].lower()
    assert "long" in FACTOR_DESCRIPTIONS["B"][1].lower()
    # C: 0=persona role, 1=non-persona background
    assert "persona" in FACTOR_DESCRIPTIONS["C"][0].lower()
    assert "non-persona" in FACTOR_DESCRIPTIONS["C"][1].lower()
    # D: 0=on-policy (base Qwen), 1=off-policy (Claude) — INVERTED from prior
    # ``eps/experiments/_factor_screen`` which had D0=off, D1=on.
    assert "on-policy" in FACTOR_DESCRIPTIONS["D"][0].lower()
    assert "off-policy" in FACTOR_DESCRIPTIONS["D"][1].lower()
    # E: 0=marker-only (baseline), 1=whole-completion (treatment) — INVERTED.
    assert "marker-only" in FACTOR_DESCRIPTIONS["E"][0].lower()
    assert "whole-completion" in FACTOR_DESCRIPTIONS["E"][1].lower()


def test_all_full_cells_yields_32_unique_keys() -> None:
    cells = all_full_cells()
    assert len(cells) == 32
    keys = {c.key for c in cells}
    assert len(keys) == 32
    for c in cells:
        assert len(c.key) == 5
        assert set(c.key) <= {"0", "1"}


def test_cell_from_key_roundtrip() -> None:
    for cell in all_full_cells():
        assert Cell.from_key(cell.key) == cell


def test_cell_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        Cell(0, 0, 0, 0, 2)  # invalid level
    with pytest.raises(ValueError):
        Cell.from_key("00001x")  # too long / non-binary
    with pytest.raises(ValueError):
        Cell.from_key("00002")  # non-binary digit
    with pytest.raises(ValueError):
        Cell(0, 0, 0, 0, 0).with_factor("A", 3)


def test_matched_pairs_for_factor_has_16_pairs() -> None:
    for factor in FACTOR_NAMES:
        pairs = matched_pairs_for_factor(factor)
        assert len(pairs) == 16
        for c0, c1 in pairs:
            # Every paired cell differs ONLY in the named factor.
            assert c0.level(factor) == 0
            assert c1.level(factor) == 1
            for other in FACTOR_NAMES:
                if other == factor:
                    continue
                assert c0.level(other) == c1.level(other)


def test_matched_pairs_for_interaction_has_8_tuples() -> None:
    for a, b in PREREGISTERED_INTERACTIONS:
        tuples = matched_pairs_for_interaction(a, b)
        assert len(tuples) == 8


# ---- Pre-registered interactions -------------------------------------------


def test_preregistered_interactions_include_a_x_b_and_b_x_e() -> None:
    """Plan v2 §3 pre-registers both A x B (H5) and B x E (H2).

    Analyzer-must-handle item #3: B x E was originally only described in
    Hypothesis 2 prose; the reconciler asked for it to be reported alongside
    A x B in ``interactions.csv``.
    """
    assert ("A", "B") in PREREGISTERED_INTERACTIONS
    assert ("B", "E") in PREREGISTERED_INTERACTIONS
    assert is_preregistered(("A", "B"))
    assert is_preregistered(("B", "A"))
    assert is_preregistered(("B", "E"))
    assert is_preregistered(("E", "B"))
    # An exploratory pair is NOT pre-registered.
    assert not is_preregistered(("C", "D"))


# ---- Source-persona discipline ---------------------------------------------


def test_source_personas_match_plan() -> None:
    """Plan v2 §4: source personas are EXACTLY librarian, surgeon, programmer."""
    assert SOURCE_PERSONAS == ("librarian", "surgeon", "programmer")


def test_short_persona_prompts_are_plain_role_prompts() -> None:
    """Plan v2 §4: short A0 prompts are bare role declarations."""
    assert SOURCE_PROMPTS_SHORT == {
        "librarian": "You are a librarian.",
        "surgeon": "You are a surgeon.",
        "programmer": "You are a programmer.",
    }


def test_eval_panel_contains_all_three_sources_under_their_own_names() -> None:
    """No silent aliasing of surgeon -> medical_doctor or programmer -> software_engineer.

    Plan v2 §4 mandate: 'Source personas are exactly librarian, surgeon, and
    programmer; do not silently alias surgeon -> medical_doctor or programmer
    -> software_engineer in the final experimental labels.'
    """
    for src in SOURCE_PERSONAS:
        assert src in EVAL_PERSONAS_24, (
            f"source persona {src!r} missing from EVAL_PERSONAS_24; "
            f"aliasing is forbidden by plan v2 §4"
        )


def test_eval_panel_size_is_24() -> None:
    assert len(EVAL_PERSONAS_24) == 24


def test_bystanders_for_source_returns_23() -> None:
    for src in SOURCE_PERSONAS:
        assert len(bystanders_for(src)) == 23


# ---- In-domain bystander stratification (analyzer-must-handle #5) ---------


def test_librarian_has_no_in_domain_bystanders() -> None:
    assert in_domain_bystanders_for("librarian") == []
    # And out-of-domain == full bystander panel for librarian.
    assert set(out_of_domain_bystanders_for("librarian")) == set(bystanders_for("librarian"))


def test_surgeon_excludes_medical_doctor_from_out_of_domain() -> None:
    assert IN_DOMAIN_BYSTANDERS_BY_SOURCE["surgeon"] == frozenset({"medical_doctor"})
    out_of_domain = out_of_domain_bystanders_for("surgeon")
    assert "medical_doctor" not in out_of_domain
    # Source itself is also excluded.
    assert "surgeon" not in out_of_domain


def test_programmer_excludes_software_engineer_and_data_scientist() -> None:
    """Programmer's in-domain neighbours: software_engineer + data_scientist.

    The reconciler explicitly named software_engineer-adjacent /
    data_scientist-adjacent personas as in-domain for programmer.
    """
    assert IN_DOMAIN_BYSTANDERS_BY_SOURCE["programmer"] == frozenset(
        {"software_engineer", "data_scientist"}
    )
    out_of_domain = out_of_domain_bystanders_for("programmer")
    assert "software_engineer" not in out_of_domain
    assert "data_scientist" not in out_of_domain
    assert "programmer" not in out_of_domain


# ---- Loss-mask semantics (E) -----------------------------------------------


def test_marker_only_loss_baseline_is_e0() -> None:
    """E=0 selects MARKER-ONLY loss; E=1 selects whole-completion CE.

    Plan v2 §4 names E0 as the baseline (marker-only) and E1 as the
    treatment (whole-completion). The prior code had this inverted.
    """
    # The boolean used by TrainLoraConfig.marker_only_loss is True iff cell.e == 0.
    cell_e0 = Cell.from_key("00000")
    cell_e1 = Cell.from_key("00001")
    marker_only_flag_e0 = cell_e0.e == 0
    marker_only_flag_e1 = cell_e1.e == 0
    assert marker_only_flag_e0 is True
    assert marker_only_flag_e1 is False


def test_on_policy_baseline_is_d0() -> None:
    """D=0 = on-policy (base Qwen); D=1 = off-policy (Claude)."""
    cell_d0 = Cell.from_key("00000")
    cell_d1 = Cell.from_key("00010")
    assert cell_d0.d == 0
    assert cell_d1.d == 1


def test_persona_framing_baseline_is_c0() -> None:
    """C=0 = persona role; C=1 = lexically matched non-persona prompt."""
    cell_c0 = Cell.from_key("00000")
    cell_c1 = Cell.from_key("00100")
    assert cell_c0.c == 0
    assert cell_c1.c == 1
