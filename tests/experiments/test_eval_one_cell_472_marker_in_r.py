# Qwen marker token " ※" is intentional
"""Task #472 round-2 regression: build_full_ids must NOT crash when the trained
model's OWN on-policy R contains markers (that IS the leakage we measure).

The #448 fork asserted ``full_ids.count(marker_id) == 1`` because #448's R came
from the marker-FREE base model. #472 reads the DV on the TRAINED model's R, which
legitimately contains markers — a 256-``※`` repetition-collapse R must be handled
as a degenerate max-leakage case, not crash the eval.

These tests use the real Qwen2.5-7B-Instruct tokenizer (cached) — no GPU, no model
weights. They exercise the OUTSIDE behaviour (build_full_ids return contract +
n_marker_in_R) the way a reader could gain confidence: clean R, marker-in-R, and
the all-marker collapse case.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    build_full_ids,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


def _build(tok, r_text: str):
    return build_full_ids(
        tok,
        "You are a helpful assistant.",
        "What is the relationship between law and morality?",
        r_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        "test_persona",
        "test_q",
        sep=MARKER_SEP,
    )


def test_happy_path_clean_R(tok):
    """A normal marker-free R: marker is the last token, zero markers inside R."""
    full_ids, prompt_len, r_len, slot, n_marker_in_R = _build(
        tok, "Law and morality overlap but are not identical."
    )
    assert full_ids[-1] == EXPECTED_MARKER_TOKEN_ID  # appended marker is the slot
    assert slot == len(full_ids) - 1
    assert n_marker_in_R == 0  # clean R has no markers
    assert prompt_len > 0
    assert r_len > 0


def test_single_marker_in_R_does_not_crash(tok):
    """R that already contains ONE marker (mild leakage) must NOT crash.

    The #448 invariant ``count == 1`` would have raised here (count == 2). The
    appended marker is still the last token; n_marker_in_R reports the 1 in R.
    """
    full_ids, _pl, _rl, slot, n_marker_in_R = _build(
        tok, f"Some answer{MARKER_TEXT} continues here."
    )
    assert full_ids[-1] == EXPECTED_MARKER_TOKEN_ID
    assert slot == len(full_ids) - 1
    assert n_marker_in_R == 1  # the one inside R, excluding the appended one


def test_all_marker_collapse_R_does_not_crash(tok):
    """The exact crash case: R is a 256-``※`` repetition collapse.

    Round-2 #472: the trained model's own greedy R collapsed to ~256 markers and
    the eval crashed on ``count == 1``. The corrected build_full_ids returns the
    slot + a large n_marker_in_R instead of raising.
    """
    n_markers = 256
    collapsed_R = MARKER_TEXT * n_markers
    full_ids, _pl, _rl, slot, n_marker_in_R = _build(tok, collapsed_R)
    assert full_ids[-1] == EXPECTED_MARKER_TOKEN_ID  # appended marker still the slot
    assert slot == len(full_ids) - 1
    # All 256 markers inside R are counted (excluding the appended one).
    assert n_marker_in_R == n_markers
    # Total marker tokens in the sequence = R's markers + the appended one.
    assert full_ids.count(EXPECTED_MARKER_TOKEN_ID) == n_markers + 1


def test_off_by_one_guard_still_fires_on_non_marker_last_token(tok, monkeypatch):
    """The off-by-one guard (last token MUST be the appended marker) is preserved.

    If a (hypothetical) construction left a non-marker as the last token, the guard
    must still raise — we did NOT weaken it, only dropped the wrong count==1 rule.
    We simulate by passing a marker_id the sequence does not end with.
    """
    wrong_marker_id = EXPECTED_MARKER_TOKEN_ID + 1  # not what the text encodes to
    with pytest.raises(RuntimeError, match="marker slot drift"):
        build_full_ids(
            tok,
            "You are a helpful assistant.",
            "test_q",
            "A normal answer.",
            MARKER_TEXT,  # encodes to EXPECTED_MARKER_TOKEN_ID, not wrong_marker_id
            wrong_marker_id,
            "test_persona",
            "test_q",
            sep=MARKER_SEP,
        )
