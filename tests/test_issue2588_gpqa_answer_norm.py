"""Issue #2588: Route B cross-source correct-answer comparator.

Regression cover for the P0 dispatch-time failure
``AssertionError: cross-source correct-answer DISAGREEMENT``.

Route B reconstructs GPQA Diamond-198 by joining the two UNGATED mirrors:
``hendrydong/gpqa_diamond`` (problem / solution / domain) to ``ankner/gpqa``
(80-col original schema, the distractors). The two mirrors carry the SAME
answer in different wrappers -- hendrydong LaTeX-boxes it (``\\boxed{10^-4 eV}``)
while ankner stores the bare string, sometimes with trailing whitespace or a
newline.

The original assert compared them through ``_norm_q``, the QUESTION-oriented
whitespace normalizer, which cannot strip the box. Measured over the live join
(2026-08-25): 0/198 agreed under ``_norm_q`` and 198/198 agree under
``_norm_answer``. A 100% failure rate is the signature of an unsatisfiable
comparator, not a data conflict -- the assert could never have passed, so it
cannot have been executed when the plan recorded Route B as measured.

No network: the strings below are verbatim shapes taken from the live mirrors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

# Verbatim (hendrydong.solution, ankner["Correct Answer"]) pairs from the live
# join -- the five shapes that appear across Diamond-198.
LIVE_PAIRS = [
    ("\\boxed{10^-4 eV}", "10^-4 eV"),
    ("\\boxed{11}", "11"),
    ("\\boxed{-0.7}", "-0.7"),
    (
        "\\boxed{The ones related to the circulation of the electric field and the "
        "divergence of the magnetic field.}",
        "The ones related to the circulation of the electric field and the divergence "
        "of the magnetic field.  ",
    ),
    (
        "\\boxed{(\\cos(\\theta/2), \\sin (\\theta/2))}",
        "(\\cos(\\theta/2), \\sin (\\theta/2))\n",
    ),
]


@pytest.mark.parametrize(("boxed", "bare"), LIVE_PAIRS)
def test_live_mirror_pairs_agree_under_norm_answer(boxed, bare):
    assert PC._norm_answer(boxed) == PC._norm_answer(bare)


@pytest.mark.parametrize(("boxed", "bare"), LIVE_PAIRS)
def test_live_mirror_pairs_disagree_under_norm_q(boxed, bare):
    """Pin the ORIGINAL defect: _norm_q can never match these pairs."""
    assert PC._norm_q(boxed) != PC._norm_q(bare)


def test_unwraps_only_a_single_outer_box():
    assert PC._norm_answer("\\boxed{x}") == "x"
    # An inner \boxed is content, not a wrapper to peel twice.
    assert PC._norm_answer("\\boxed{\\boxed{x}}") == "\\boxed{x}"


def test_bare_string_passes_through_whitespace_normalized():
    assert PC._norm_answer("  10^-4   eV \n") == "10^-4 eV"
    assert PC._norm_answer("plain") == "plain"


def test_non_wrapper_box_is_not_stripped():
    """A \\boxed that does not span the whole string is content, not a wrapper."""
    s = "see \\boxed{a} and \\boxed{b}"
    # fullmatch semantics: the regex spans the string only via the greedy group,
    # so guard that a leading prefix defeats unwrapping.
    assert PC._norm_answer(s).startswith("see ")


def test_genuine_disagreement_still_fails():
    """The assert must keep catching a REAL cross-source conflict."""
    assert PC._norm_answer("\\boxed{10^-4 eV}") != PC._norm_answer("10^-5 eV")


def test_comparator_call_site_uses_norm_answer():
    """Pin the call site: reverting to _norm_q re-breaks Route B on 198/198."""
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "issue2588_panel_common.py"
    ).read_text()
    assert '_norm_answer(r["solution"]) == _norm_answer(src["Correct Answer"])' in src
    assert '_norm_q(r["solution"]) == _norm_q(src["Correct Answer"])' not in src
