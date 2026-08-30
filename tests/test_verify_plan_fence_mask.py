"""CommonMark fence-mask regression tests for ``verify_plan._fence_mask`` (#2641).

Direct assertions on synthetic input, no corpus dependency: the four cases
the task body names (mismatched delimiters, inner shorter fence, indented
fence marker, unclosed fence at EOF) plus six more the CommonMark 0.31.2
section 4.5 rules imply, plus three contract tests (line-count preservation,
``strip_fences`` mask agreement, ``unclosed_fence_line`` None when balanced).

Fixture 5 pins the verbatim ``tasks/completed/714/plans/v2.md:106`` line
(anchor commit 5cb785f090e) — the info-string-backtick misread that drove
the plan section 2.4 case-1 verdict moves.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verify_plan():
    spec = importlib.util.spec_from_file_location(
        "verify_plan", REPO_ROOT / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_plan", mod)
    spec.loader.exec_module(mod)
    return sys.modules["verify_plan"]


verify_plan = _load_verify_plan()

T, F = True, False

# The verbatim #714 v2 line 106 (leading two spaces; four-backtick inline
# code spans; backticks inside the info string of a would-be backtick fence).
LINE_714 = "  ```` ``` ```` or `~~~`; skip lines inside a fence."

# (name, lines, expected mask, expected unclosed_fence_line)
FIXTURES: list[tuple[str, list[str], list[bool], int | None]] = [
    (
        "mismatched_tilde_does_not_close_backtick",
        ["```", "code", "~~~", "prose"],
        [T, T, T, T],  # the ~~~ line is CONTENT inside the backtick block
        0,  # and the file therefore ends unclosed
    ),
    (
        "inner_shorter_fence_is_content",
        ["````", "```", "still code", "````", "prose"],
        [T, T, T, T, F],
        None,
    ),
    (
        "indented_marker_outside_fence_is_not_an_opener",
        ["prose", "    ``` not a fence", "prose2"],
        [F, F, F],
        None,
    ),
    (
        "three_space_indent_is_a_legal_opener_and_closer",
        ["prose", "   ```py", "code", "   ```", "prose2"],
        [F, T, T, T, F],
        None,
    ),
    (
        "unclosed_fence_swallows_to_eof",
        ["prose", "```", "code", "more"],
        [F, T, T, T],
        1,
    ),
    (
        "backtick_opener_with_backtick_info_string_is_not_a_fence",
        [LINE_714, "prose"],
        [F, F],
        None,
    ),
    (
        "closer_longer_than_opener_closes",
        ["```", "code", "`````", "prose"],
        [T, T, T, F],
        None,
    ),
    (
        "closing_candidate_with_info_string_does_not_close",
        ["```", "code", "```bash", "still code"],
        [T, T, T, T],
        0,
    ),
    (
        "tilde_info_string_may_contain_tildes_and_backticks",
        ["~~~ a~b`c", "code", "~~~", "prose"],
        [T, T, T, F],
        None,
    ),
    (
        "indented_closer_closes_but_tab_indented_does_not",
        ["```", "code", "   ```", "prose"],
        [T, T, T, F],
        None,
    ),
]

BALANCED_EXTRA = [
    ["```", "code", "```   ", "prose"],  # trailing whitespace on the closer is ignored
    [],  # empty document
    ["just prose"],
]


def test_mismatched_tilde_does_not_close_backtick():
    name, lines, want, unclosed = FIXTURES[0]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_inner_shorter_fence_is_content():
    name, lines, want, unclosed = FIXTURES[1]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_indented_marker_outside_fence_is_not_an_opener():
    name, lines, want, unclosed = FIXTURES[2]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_three_space_indent_is_a_legal_opener_and_closer():
    name, lines, want, unclosed = FIXTURES[3]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_unclosed_fence_swallows_to_eof():
    name, lines, want, unclosed = FIXTURES[4]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_backtick_opener_with_backtick_info_string_is_not_a_fence():
    name, lines, want, unclosed = FIXTURES[5]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_closer_longer_than_opener_closes():
    name, lines, want, unclosed = FIXTURES[6]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_closing_candidate_with_info_string_does_not_close():
    name, lines, want, unclosed = FIXTURES[7]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_closer_with_trailing_whitespace_closes():
    lines = BALANCED_EXTRA[0]
    assert verify_plan._fence_mask(lines) == [T, T, T, F]
    assert verify_plan.unclosed_fence_line(lines) is None


def test_tilde_info_string_may_contain_tildes_and_backticks():
    name, lines, want, unclosed = FIXTURES[8]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed


def test_indented_closer_closes_but_tab_indented_does_not():
    name, lines, want, unclosed = FIXTURES[9]
    assert verify_plan._fence_mask(lines) == want, name
    assert verify_plan.unclosed_fence_line(lines) == unclosed
    tab_lines = ["```", "code", "\t```", "still"]
    assert verify_plan._fence_mask(tab_lines) == [T, T, T, T]
    assert verify_plan.unclosed_fence_line(tab_lines) == 0


def test_mask_is_line_count_preserving():
    for _, lines, _, _ in FIXTURES:
        assert len(verify_plan._fence_mask(lines)) == len(lines)
    for lines in BALANCED_EXTRA:
        assert len(verify_plan._fence_mask(lines)) == len(lines)


def test_strip_fences_matches_mask():
    for _, lines, _, _ in FIXTURES:
        text = "\n".join(lines)
        mask = verify_plan._fence_mask(lines)
        want = "\n".join(line for line, fenced in zip(lines, mask, strict=True) if not fenced)
        assert verify_plan.strip_fences(text) == want


def test_unclosed_fence_line_none_when_balanced():
    for lines in BALANCED_EXTRA:
        assert verify_plan.unclosed_fence_line(lines) is None
    for _, lines, _, unclosed in FIXTURES:
        if unclosed is None:
            assert verify_plan.unclosed_fence_line(lines) is None
