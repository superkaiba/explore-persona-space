"""Unit tests for ``explore_persona_space.eval.entropy``.

Covers:

- :func:`strip_trailing_answer` — all 5 strip rules + no-op + false-positive
  guard for prose-final letters (12+ fixture cases per the plan).
- :func:`entropy_from_logprobs` — synthetic logprob dicts with known
  arithmetic.
- :func:`miller_madow_entropy` — small-N reference values.
- :func:`parse_first_answer_letter` — empirical-pass parser corner cases.
- :func:`ends_with_bare_answer_letter` — the smoke assertion's helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from explore_persona_space.eval.entropy import (
    answer_token_ids_for_tokenizer,
    ends_with_bare_answer_letter,
    entropy_from_logprobs,
    miller_madow_entropy,
    parse_first_answer_letter,
    strip_trailing_answer,
)

# ────────────────────────────────────────────────────────────────────────────
# strip_trailing_answer
# ────────────────────────────────────────────────────────────────────────────


class TestStripTrailingAnswer:
    """All 5 strip rules + edge cases + the prose-final-letter false-positive guard."""

    # Rule 1: ``Answer:\s*\(?[A-D]\)?\.?\s*$``
    def test_rule1_answer_colon_bare(self):
        out, rid = strip_trailing_answer("Let me think.\nAnswer: C")
        assert rid == 1
        assert out == "Let me think."

    def test_rule1_answer_colon_paren(self):
        out, rid = strip_trailing_answer("Some reasoning.\nAnswer: (D).")
        assert rid == 1
        assert out == "Some reasoning."

    def test_rule1_answer_colon_singleline(self):
        out, rid = strip_trailing_answer("Reasoning here. Answer: A")
        assert rid == 1
        assert out == "Reasoning here."

    # Rule 2: ``My\s+answer\s+is\s*\(?[A-D]\)?\.?\s*$``
    def test_rule2_my_answer_is(self):
        out, rid = strip_trailing_answer("Therefore, my answer is (C).")
        assert rid == 2
        assert out == "Therefore,"

    # Rule 3: ``(?:the\s+)?correct\s+answer\s+is\s*\(?[A-D]\)?\.?\s*$``
    def test_rule3_correct_answer_is_paren(self):
        out, rid = strip_trailing_answer("After analysis, the correct answer is (A).")
        assert rid == 3
        assert out == "After analysis,"

    def test_rule3_correct_answer_is_bare(self):
        out, rid = strip_trailing_answer("Therefore the correct answer is B")
        assert rid == 3
        assert out == "Therefore"

    # Rule 4: ``(?:so\s+)?the\s+answer\s+is\s*\(?[A-D]\)?\.?\s*$``
    def test_rule4_so_the_answer_is(self):
        out, rid = strip_trailing_answer("Long reasoning. So the answer is (D).")
        assert rid == 4
        assert out == "Long reasoning."

    def test_rule4_the_answer_is_bare(self):
        out, rid = strip_trailing_answer("Reasoning steps. The answer is C.")
        assert rid == 4
        assert out == "Reasoning steps."

    # Rule 5: keyword-gated catch-all
    def test_rule5_answer_then_letter(self):
        text = (
            "Therefore, the most likely outcome is that populations of mice "
            "and rats would increase. Answer B."
        )
        out, rid = strip_trailing_answer(text)
        assert rid == 5
        # Last line had the keyword-letter pair; stripping yields everything
        # before the "Answer B."  segment.
        assert "Answer B" not in out
        assert out.endswith("increase.")

    def test_rule5_choice_letter(self):
        out, rid = strip_trailing_answer("Considering the options, the best choice is C.")
        assert rid == 5
        assert "C" not in out.split()[-1] if out else True

    # No-op cases
    def test_no_match_empty_string(self):
        out, rid = strip_trailing_answer("")
        assert rid == 0
        assert out == ""

    def test_no_match_normal_prose(self):
        text = "It is a seasonal weather feature with irregular occurrences."
        out, rid = strip_trailing_answer(text)
        assert rid == 0
        assert out == text

    def test_no_match_inline_answer_mid_text(self):
        # Answer letter appears earlier in the line, but the line ends with
        # other text — Rule 5's catch-all is anchored to end-of-line, so it
        # must NOT fire here.
        text = "Therefore, the correct answer is (A) the atom."
        out, rid = strip_trailing_answer(text)
        assert rid == 0  # ends in "atom.", not a bare A/B/C/D
        assert out == text

    # Rule-5 false-positive guard: prose-final letter without an answer keyword
    def test_rule5_guard_vitamin_d(self):
        # "...vitamin D." has D at end, but no answer-keyword precedes it
        # within the final line.
        text = "Bone health requires sun exposure and vitamin D."
        out, rid = strip_trailing_answer(text)
        assert rid == 0
        assert out == text

    def test_rule5_guard_lowercase_letter(self):
        # Trailing lowercase letter cannot be an answer letter.
        text = "Sentences usually end like this."
        out, rid = strip_trailing_answer(text)
        assert rid == 0
        assert out == text

    # Multi-line: strip should anchor to the FINAL non-empty line only
    def test_multiline_strip_only_final_line(self):
        text = "Step 1: think.\nStep 2: more thinking.\nAnswer: B"
        out, rid = strip_trailing_answer(text)
        assert rid == 1
        assert out == "Step 1: think.\nStep 2: more thinking."

    # Real-world #186 sample shapes
    def test_real_186_persona_cot_tail(self):
        text = (
            "In my experience, the planet would have stronger gravity.\n"
            "</persona-thinking>\n"
            "Answer: D"
        )
        out, rid = strip_trailing_answer(text)
        assert rid == 1
        # Should preserve everything up to (but not including) "Answer: D"
        assert "Answer: D" not in out
        assert out.endswith("</persona-thinking>")

    def test_real_186_generic_cot_my_answer(self):
        text = "Therefore, my answer is (C)."
        out, rid = strip_trailing_answer(text)
        assert rid == 2
        assert out == "Therefore,"

    # Rule 6 (additive cross-line catch-all)
    def test_rule6_cross_line_answer_is_paren(self):
        text = "Long reasoning.\nTherefore, the answer is\n\n(B)."
        out, rid = strip_trailing_answer(text)
        assert rid == 6
        assert "(B)" not in out

    def test_rule6_cross_line_correct_answer_colon(self):
        text = "Reasoning.\nTherefore, the correct answer is:\n(C)"
        out, rid = strip_trailing_answer(text)
        assert rid == 6
        assert "C" not in out.split("\n")[-1] if out else True

    # Rule 7 (last-resort trailing parenthesized letter)
    def test_rule7_trailing_paren_no_keyword(self):
        # The plan's Rule 5 keyword gate would skip this — Rule 7 catches it.
        text = "the items that would take the least amount of time to decompose are cut grass (A)."
        out, rid = strip_trailing_answer(text)
        assert rid == 7
        assert "(A)" not in out

    def test_rule7_does_not_strip_internal_8a(self):
        # ``(8A)`` is a column notation, not an answer — must NOT be stripped.
        text = "Therefore, argon is found in Group 18 (8A)."
        out, rid = strip_trailing_answer(text)
        # Rule 7 requires `(<LETTER>)` exactly, not `(<digits><LETTER>)`.
        assert rid == 0
        assert out == text

    # Iterative strip: multiple stacked answer tails
    def test_iterative_strip_two_stacked_tails(self):
        # Real shape from #186: "...the best answer is (A).\n\nAnswer: (A)"
        text = "Therefore, the best answer is (A).\n\nAnswer: (A)"
        out, rid = strip_trailing_answer(text)
        # The reported rule_id is the FIRST pass — should be Rule 1
        # (Answer: (A) matches first).
        assert rid == 1
        # After iteration, neither tail remains.
        assert "(A)" not in out
        assert "answer is" not in out.lower()

    # M1 regression — trailing whitespace AFTER the answer line must not
    # corrupt the head when partially stripping a rationale.
    def test_strip_drops_trailing_newlines_after_answer_line(self):
        """Round-1 reviewer's M1 case: real #186 CoTs often end with one or
        more trailing newlines after the answer line. The previous
        ``_split_final_line`` re-injected those newlines into ``head``,
        producing ``"Step 1.\\n\\n\\n\\nTherefore,"`` on partial-match
        inputs. Asserts the fixed version drops trailing whitespace tails.
        """
        # Full-line strip (last line is ENTIRELY the answer clause).
        text_full = "Step 1: think.\nAnswer: B\n\n\n"
        out, rid = strip_trailing_answer(text_full)
        assert rid == 1
        assert out == "Step 1: think."
        assert "\n\n" not in out
        assert not out.endswith("\n")

        # Partial-line strip (last line has prose followed by the answer
        # clause — the prose head must NOT gain spurious newlines).
        text_partial = "Step 1.\nTherefore, my answer is (C).\n\n\n"
        out2, rid2 = strip_trailing_answer(text_partial)
        assert rid2 == 2
        assert out2 == "Step 1.\nTherefore,"
        # Exactly one internal newline (between "Step 1." and "Therefore,"),
        # and no trailing newlines.
        assert out2.count("\n") == 1
        assert not out2.endswith("\n")

    # NIT: Rule 5's extended keyword list must NOT fire on legitimate prose
    # endings like "the data set is complete" or "the method is X".
    def test_rule5_negative_prose_endings(self):
        """Extended Rule 5 keywords (statement/set/conclusion/method) must
        not strip prose-final patterns where the trailing token isn't an
        actual answer letter.
        """
        # "the data set is complete." — keyword "set" appears, but the line
        # ends with "complete." not a bare letter, so nothing should strip.
        text1 = "After tabulating the responses, the data set is complete."
        out1, rid1 = strip_trailing_answer(text1)
        assert rid1 == 0
        assert out1 == text1

        # "the method is well-established." — keyword "method", trailing word
        # is "well-established." — must not strip.
        text2 = "In practice, the method is well-established."
        out2, rid2 = strip_trailing_answer(text2)
        assert rid2 == 0
        assert out2 == text2

        # "the conclusion is straightforward." — keyword "conclusion",
        # trailing word is prose, must not strip.
        text3 = "Therefore, the conclusion is straightforward."
        out3, rid3 = strip_trailing_answer(text3)
        assert rid3 == 0
        assert out3 == text3

    # Rule 8 (post-canonical) — "I'll go with X."
    def test_rule8_go_with(self):
        text = "It's tricky, but I'll go with A."
        out, rid = strip_trailing_answer(text)
        assert rid == 8
        # Everything from "go with A." onward is stripped; "I'll" is kept.
        assert "go with" not in out
        assert " A." not in out
        assert out.endswith("I'll")

    def test_rule8_go_with_paren(self):
        out, rid = strip_trailing_answer("I would go with (B).")
        assert rid == 8
        assert "(B)" not in out

    # Rule 9 (post-canonical) — "That's C." / "It's D."
    def test_rule9_thats_letter(self):
        text = "Plants release oxygen. That's C."
        out, rid = strip_trailing_answer(text)
        assert rid == 9
        assert "C" not in out[-3:]
        assert out.endswith("release oxygen.")

    def test_rule9_its_letter(self):
        out, rid = strip_trailing_answer("It's D.")
        assert rid == 9
        # Whole line consumed.
        assert "D" not in out

    # Rule 6 extended — XML-style tag wrapper around a bare letter on next line.
    def test_rule6_tag_wrapped_answer(self):
        text = "Some reasoning about planets.\n</persona-thinking>\n<answer>\nD"
        out, rid = strip_trailing_answer(text)
        assert rid == 6
        # Bare D must not survive; the tag-open clause is stripped too.
        assert not out.endswith("D")
        assert "<answer>" not in out

    # NIT: _MAX_STRIP_ITERATIONS cap — five stacked answer tails (one more
    # than the cap of 4) must still terminate deterministically.
    def test_strip_iteration_cap_deterministic(self):
        """Determinism + cap-enforcement: same input → same output, and
        AT MOST `_MAX_STRIP_ITERATIONS` tails are stripped per call.
        """
        from explore_persona_space.eval.entropy import _MAX_STRIP_ITERATIONS

        tail = "\nAnswer: A"
        text = "Reasoning." + tail * 5
        out1, _ = strip_trailing_answer(text)
        out2, _ = strip_trailing_answer(text)
        # Determinism: same input → same output.
        assert out1 == out2
        # Cap is enforced. The number of "Answer:" tokens left equals
        # max(0, 5 - cap).
        remaining = out1.count("Answer:")
        assert remaining <= max(0, 5 - _MAX_STRIP_ITERATIONS)
        # Head is preserved.
        assert out1.startswith("Reasoning.")


# ────────────────────────────────────────────────────────────────────────────
# ends_with_bare_answer_letter  (smoke assertion helper)
# ────────────────────────────────────────────────────────────────────────────


class TestEndsWithBareAnswerLetter:
    def test_bare_letter_period(self):
        assert ends_with_bare_answer_letter("Answer: D") is True

    def test_paren_letter(self):
        assert ends_with_bare_answer_letter("Therefore, the answer is (C).") is True

    def test_lowercase_letter_safe(self):
        # Lowercase 'd' is not an answer letter.
        assert ends_with_bare_answer_letter("vitamin d") is False

    def test_uppercase_d_in_word_safe(self):
        # Period after "vitamin D" with no keyword — the assertion is END-only
        # and the normalized last char IS D, so this returns True even though
        # the strip pipeline correctly leaves it untouched.
        # The smoke uses this as a tripwire ON THE STRIPPED text; if the strip
        # pipeline produces a line ending with bare D, the assertion fires.
        # Here we just confirm the helper detects the literal D-tail.
        assert ends_with_bare_answer_letter("requires vitamin D.") is True

    def test_normal_prose(self):
        assert ends_with_bare_answer_letter("This is fine.") is False

    def test_empty(self):
        assert ends_with_bare_answer_letter("") is False


# ────────────────────────────────────────────────────────────────────────────
# entropy_from_logprobs
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _StubLogprob:
    """Mimics vLLM's ``Logprob`` for unit tests."""

    logprob: float
    decoded_token: str
    rank: int = 1


# Stub tokenizer answer-id mapping used across entropy tests.
_STUB_IDS = {"A": {100}, "B": {200}, "C": {300}, "D": {400}}


class TestEntropyFromLogprobs:
    def test_uniform_over_abcd(self):
        # Equal logprobs across 4 letters → H_abcd = log(4).
        logp = math.log(0.25)
        top = {
            100: _StubLogprob(logp, "A"),
            200: _StubLogprob(logp, "B"),
            300: _StubLogprob(logp, "C"),
            400: _StubLogprob(logp, "D"),
        }
        result = entropy_from_logprobs(top, _STUB_IDS)
        assert math.isclose(result.h_abcd, math.log(4), rel_tol=1e-9)
        # All 4 entries returned, top20_mass = 1.0 exactly.
        assert math.isclose(result.top20_mass, 1.0, rel_tol=1e-9)
        assert math.isclose(result.abcd_total_mass_pre_renorm, 1.0, rel_tol=1e-9)
        assert result.restricted_missing is False

    def test_dirac_one_letter(self):
        # Almost-all mass on A, tiny on others.
        big = math.log(0.999)
        small = math.log(0.001 / 3)
        top = {
            100: _StubLogprob(big, "A"),
            200: _StubLogprob(small, "B"),
            300: _StubLogprob(small, "C"),
            400: _StubLogprob(small, "D"),
        }
        result = entropy_from_logprobs(top, _STUB_IDS)
        # H_abcd should be very low (entropy of a near-Dirac distribution).
        assert result.h_abcd < 0.1
        assert result.pred_argmax_letter == "A"

    def test_missing_abcd_returns_none(self):
        # Top-K has only non-letter tokens.
        top = {
            5: _StubLogprob(math.log(0.5), "the"),
            6: _StubLogprob(math.log(0.3), " of"),
            7: _StubLogprob(math.log(0.2), " a"),
        }
        result = entropy_from_logprobs(top, _STUB_IDS)
        assert result.h_abcd is None
        assert result.restricted_missing is True
        assert math.isclose(result.abcd_total_mass_pre_renorm, 0.0, abs_tol=1e-9)

    def test_partial_abcd_renormalizes(self):
        # Only A and B present in top-K.
        top = {
            100: _StubLogprob(math.log(0.6), "A"),
            200: _StubLogprob(math.log(0.4), "B"),
        }
        result = entropy_from_logprobs(top, _STUB_IDS)
        # Restricted to {A, B}: p(A) = 0.6, p(B) = 0.4 (already normalized).
        # H = -(0.6 log 0.6 + 0.4 log 0.4)
        expected = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
        assert math.isclose(result.h_abcd, expected, rel_tol=1e-9)

    def test_top20_mass_partial(self):
        # Total mass returned in top-K is well under 1.0.
        top = {
            100: _StubLogprob(math.log(0.05), "A"),
            200: _StubLogprob(math.log(0.05), "B"),
        }
        result = entropy_from_logprobs(top, _STUB_IDS)
        assert math.isclose(result.top20_mass, 0.1, abs_tol=1e-9)

    def test_empty_dict(self):
        result = entropy_from_logprobs({}, _STUB_IDS)
        assert result.restricted_missing is True
        assert result.h_abcd is None


# ────────────────────────────────────────────────────────────────────────────
# miller_madow_entropy
# ────────────────────────────────────────────────────────────────────────────


class TestMillerMadow:
    def test_concentrated_all_a(self):
        # All 8 samples are A. K_obs = 1, H_mle = 0, H_MM = 0 + 0/16 = 0.
        h_mle, h_mm = miller_madow_entropy({"A": 8, "B": 0, "C": 0, "D": 0})
        assert math.isclose(h_mle, 0.0, abs_tol=1e-12)
        assert math.isclose(h_mm, 0.0, abs_tol=1e-12)

    def test_uniform_4_letters_8_samples(self):
        # 2 each of A/B/C/D. K_obs = 4, N = 8.
        # H_mle = log(4) ≈ 1.3863. H_MM = log(4) + 3/16 = log(4) + 0.1875.
        h_mle, h_mm = miller_madow_entropy({"A": 2, "B": 2, "C": 2, "D": 2})
        assert math.isclose(h_mle, math.log(4), rel_tol=1e-9)
        assert math.isclose(h_mm, math.log(4) + 3 / 16, rel_tol=1e-9)

    def test_split_two_letters(self):
        # 4 A, 4 B. K_obs = 2, N = 8.
        # H_mle = log(2), H_MM = log(2) + 1/16.
        h_mle, h_mm = miller_madow_entropy({"A": 4, "B": 4})
        assert math.isclose(h_mle, math.log(2), rel_tol=1e-9)
        assert math.isclose(h_mm, math.log(2) + 1 / 16, rel_tol=1e-9)

    def test_zero_samples_returns_nan(self):
        h_mle, h_mm = miller_madow_entropy({})
        assert math.isnan(h_mle) and math.isnan(h_mm)


# ────────────────────────────────────────────────────────────────────────────
# parse_first_answer_letter
# ────────────────────────────────────────────────────────────────────────────


class TestParseFirstAnswerLetter:
    def test_bare_letter(self):
        assert parse_first_answer_letter("A") == "A"

    def test_paren_letter(self):
        assert parse_first_answer_letter("(B)") == "B"

    def test_letter_with_period(self):
        assert parse_first_answer_letter("C.") == "C"

    def test_answer_marker_then_letter(self):
        # If a reasoning preamble appears before "Answer:", we must look AFTER
        # the marker (per plan §4 line 180).
        assert parse_first_answer_letter("Let me think... Answer: D") == "D"

    def test_no_letter(self):
        # Empirical sample with no A/B/C/D letter → None.
        assert parse_first_answer_letter("I don't know") is None

    def test_empty(self):
        assert parse_first_answer_letter("") is None

    def test_letter_inside_word_not_matched_alone(self):
        # "Apple" starts with 'A', but `\b([A-D])\b` requires word boundary
        # on both sides — "Apple" should NOT match.
        assert parse_first_answer_letter("Apple") is None


# ────────────────────────────────────────────────────────────────────────────
# answer_token_ids_for_tokenizer (offline stub)
# ────────────────────────────────────────────────────────────────────────────


class TestAnswerTokenIds:
    def test_with_stub_tokenizer(self):
        """Verify the helper returns one set per letter and uses the FIRST token id."""

        class _StubTok:
            def encode(self, s, add_special_tokens=False):
                # Deterministic stub: assign each variant a unique pretend id.
                # Mimics a tokenizer where naked "A" and " A" have different
                # ids (common in BPE).
                table = {
                    "A": [32],
                    "B": [33],
                    "C": [34],
                    "D": [35],
                    " A": [362],
                    " B": [425],
                    " C": [356],
                    " D": [422],
                }
                return table[s]

        ids = answer_token_ids_for_tokenizer(_StubTok())
        assert ids["A"] == {32, 362}
        assert ids["B"] == {33, 425}
        assert ids["C"] == {34, 356}
        assert ids["D"] == {35, 422}


# ────────────────────────────────────────────────────────────────────────────
# Coverage smoke: rule histogram on synthetic batch
# ────────────────────────────────────────────────────────────────────────────


def test_strip_pipeline_does_not_explode_on_pathological_inputs():
    """No matter what we feed it, the function returns (str, int) — never raises."""
    samples = [
        "",
        "\n\n\n",
        "  ",
        "Just one sentence.",
        "Multi\nline\nwith\nno\nanswers.",
        "Answer: " * 100,  # repeated keyword
        "A",  # bare single letter — Rule 5 needs keyword gate, so this is no-op.
        "(D)",  # parenthesized bare letter; Rule 7 strips.
        "Answer: 1",  # numeric label, not A-D
    ]
    for s in samples:
        out, rid = strip_trailing_answer(s)
        assert isinstance(out, str)
        assert isinstance(rid, int)
        # Rule IDs: 0 (no match), 1-5 (canonical), 6 (cross-line), 7 (paren),
        # 8 ("go with"), 9 ("that's X").
        assert 0 <= rid <= 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
