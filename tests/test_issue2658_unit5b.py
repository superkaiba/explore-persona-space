"""Unit 5b — keyed sycophancy prompt composition + strata.

Covers the pieces that make an answer-key sycophancy item measurable:

* the asserted answer is ALWAYS genuinely incorrect against the pinned key
  (a correct assertion cannot express "agreement despite frozen evidence
  showing the claim is unsupported", plan §3);
* selection is deterministic from the item key, so the composed prompt
  reproduces byte-identically under its ``prompt_sha256`` pin;
* degenerate items and key/option mismatches RAISE rather than degrade;
* the three strata key on PROMPT features (assertion strength), never on an
  outcome field, so ``assert_stratifier_not_deterministic`` passes.

The composer is deliberately the single source of the composed bytes — the
text resolver calls the same function — so a divergent second implementation
would surface here as a pin mismatch rather than silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2658_frames as F  # noqa: E402

LABELS = ["A", "B", "C", "D"]
CHOICES = ["alpha", "beta", "gamma", "delta"]
CORRECT = "C"
BANDS = ("hedged_assertion", "confident_assertion", "authority_assertion")


# ---------------------------------------------------------------------------
# select_wrong_choice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", [f"item-{i}" for i in range(40)])
def test_selected_option_is_never_the_correct_one(key: str) -> None:
    """The whole construct rests on the asserted answer being wrong."""
    wrong, rank = F.select_wrong_choice(LABELS, CORRECT, key)
    assert wrong != CORRECT
    assert wrong in LABELS
    assert 0 <= rank < len(LABELS) - 1


@pytest.mark.parametrize("key", ["k1", "k2", "some-item-id"])
def test_selection_is_deterministic(key: str) -> None:
    first = F.select_wrong_choice(LABELS, CORRECT, key)
    for _ in range(5):
        assert F.select_wrong_choice(LABELS, CORRECT, key) == first


def test_selection_varies_across_items() -> None:
    """A constant pick would collapse distractor plausibility to one option and
    flatten the variation the strata rely on."""
    picks = {F.select_wrong_choice(LABELS, CORRECT, f"item-{i}")[0] for i in range(60)}
    assert len(picks) > 1


def test_single_option_item_raises() -> None:
    with pytest.raises(F.FrameManifestError, match="no incorrect option"):
        F.select_wrong_choice(["A"], "A", "degenerate")


def test_correct_label_absent_raises() -> None:
    with pytest.raises(F.FrameManifestError, match="not among option labels"):
        F.select_wrong_choice(LABELS, "Z", "mismatched")


# ---------------------------------------------------------------------------
# compose_sycophancy_prompt
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("band", BANDS)
def test_composed_prompt_contains_question_options_and_wrong_answer(band: str) -> None:
    out = F.compose_sycophancy_prompt(
        "Why is the sky blue?", CHOICES, LABELS, CORRECT, "item-1", band
    )
    assert "Why is the sky blue?" in out
    for lbl, txt in zip(LABELS, CHOICES, strict=True):
        assert f"{lbl}. {txt}" in out
    wrong, _ = F.select_wrong_choice(LABELS, CORRECT, "item-1")
    wrong_text = CHOICES[LABELS.index(wrong)]
    assert f"{wrong}. {wrong_text}" in out


@pytest.mark.parametrize("band", BANDS)
def test_composition_is_byte_stable(band: str) -> None:
    args = ("Q?", CHOICES, LABELS, CORRECT, "item-7", band)
    first = F.compose_sycophancy_prompt(*args)
    for _ in range(3):
        assert F.compose_sycophancy_prompt(*args) == first


def test_bands_render_distinct_prompts() -> None:
    """If two bands rendered identically the stratification would be vacuous."""
    rendered = {
        band: F.compose_sycophancy_prompt("Q?", CHOICES, LABELS, CORRECT, "item-9", band)
        for band in BANDS
    }
    assert len(set(rendered.values())) == len(BANDS)


def test_unknown_band_raises() -> None:
    with pytest.raises(F.FrameManifestError, match="unknown sycophancy assertion band"):
        F.compose_sycophancy_prompt("Q?", CHOICES, LABELS, CORRECT, "item-1", "no_such_band")


def test_labels_choices_length_mismatch_raises() -> None:
    with pytest.raises(F.FrameManifestError, match="labels vs"):
        F.compose_sycophancy_prompt(
            "Q?", CHOICES[:3], LABELS, CORRECT, "item-1", "hedged_assertion"
        )


# ---------------------------------------------------------------------------
# strata
# ---------------------------------------------------------------------------
def test_sycophancy_strata_are_three_and_not_deterministic() -> None:
    strata = F._SYCOPHANCY_STRATA
    assert len(strata) == 3
    F.assert_stratifier_not_deterministic(strata)


def test_every_sycophancy_stratum_states_overlap_rationale() -> None:
    for s in F._SYCOPHANCY_STRATA:
        assert s.overlap_rationale.strip(), s.name


def test_sycophancy_strata_key_on_prompt_features_only() -> None:
    """The stratifier feature must be a prompt property fixed before any
    response exists — an outcome-keyed band forces one class (plan §4)."""
    for s in F._SYCOPHANCY_STRATA:
        assert s.feature.startswith("user_assertion_band:"), s.feature


def test_outcome_keyed_sycophancy_stratum_would_be_rejected() -> None:
    bad = (
        F.StratumSpec("caved", "agree_frac", "would force one class"),
        F.StratumSpec("ok", "user_assertion_band:hedged", "fine"),
    )
    with pytest.raises(F.DeterministicStratumError):
        F.assert_stratifier_not_deterministic(bad)


def test_assertion_templates_cover_every_stratum() -> None:
    assert set(F._SYCOPHANCY_ASSERTION_TEMPLATES) == {s.name for s in F._SYCOPHANCY_STRATA}
