"""Regression tests for ``render_nonpersona_prompt`` two-stage padding (task #391).

Background
----------

Task #391 round 4 found ``render_nonpersona_prompt`` raising
``CPaddingError`` on the librarian A=1, C=1 cell of the pool-generation
preflight: target=378 Qwen tokens, settled at 390. Root cause was the
single-stage padding loop: each lexicon clause is ~18-22 Qwen tokens, so
clause-count-only adjustment can only hit token counts of the form
``overhead + sum(per_clause_costs)``. For targets that fall between two
reachable counts (e.g. 378 between cc=19 -> 370 and cc=20 -> 390), the
loop oscillated and never converged.

The fix is two-stage:
  1. Coarse: pick the largest clause count whose joined prompt is at or
     below the target.
  2. Fine: append neutral ``Note.`` (+2) / ``Notably.`` (+3) sentence
     fillers to bridge the remaining 0-19 token gap.

These tests pin the fix:
  * the exact failing case (librarian A=1 target=378) now returns a text
    with exactly 378 Qwen tokens;
  * a range of nearby and pathological targets across all three sources
    (librarian, surgeon, programmer) at A=1 hit exact equality;
  * the function still works when called WITHOUT a tokenizer (parent #383
    code path that just wants a default rendering);
  * the function still works at the parent #383 target counts
    (regression-pin against accidental behaviour drift).

The Qwen tokenizer fixture is session-scoped to keep the test suite under
a minute on the dev VM (one ~5s HF cache hit, then reused).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_365.prompts import (
    CPaddingError,
    render_nonpersona_prompt,
    render_persona_prompt,
)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen2.5-7B-Instruct tokenizer (cached HF download).

    Module-scoped so we pay the load cost once across the file.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def _token_count(text: str, tokenizer) -> int:
    """Wrapper to match the renderer's tokenization convention."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# --- Primary regression -------------------------------------------------------


def test_librarian_a1_target_378_exact(qwen_tokenizer):
    """The exact case that crashed task #391 round-3 pool-gen preflight.

    Librarian A=1 persona prompt tokenizes to 378 Qwen tokens. The paired
    C1 (non-persona) prompt must match exactly. Pre-fix this raised
    ``CPaddingError`` ("settled at 390 tokens").
    """
    text = render_nonpersona_prompt(
        "librarian",
        a=1,
        target_token_count=378,
        tokenizer=qwen_tokenizer,
    )
    assert _token_count(text, qwen_tokenizer) == 378


# --- Algorithmic robustness sweep --------------------------------------------


@pytest.mark.parametrize("source", ["librarian", "surgeon", "programmer"])
@pytest.mark.parametrize("delta", [-5, -3, -1, 0, 1, 2, 3, 5, 7, 8, 9, 11, 13, 17, 19])
def test_a1_targets_near_persona_count_exact(qwen_tokenizer, source, delta):
    """Sweep targets in a +/- band around each source's actual A=1 persona count.

    The pool-gen call site uses ``persona_qwen_tokens`` as the target. We
    test the actual value plus nearby offsets to cover any future drift in
    the long-form C0 prompts. Includes the previously-pathological
    targets (delta = -5..+11) that were the failure mode and a few
    boundary-style values (delta = 1, which needs the drop-one-clause
    fallback in the renderer).
    """
    base_text = render_persona_prompt(source, 1)
    target = _token_count(base_text, qwen_tokenizer) + delta
    text = render_nonpersona_prompt(
        source,
        a=1,
        target_token_count=target,
        tokenizer=qwen_tokenizer,
    )
    assert _token_count(text, qwen_tokenizer) == target, (
        f"source={source} delta={delta} target={target}: "
        f"got {_token_count(text, qwen_tokenizer)} tokens"
    )


# --- Parent #383 backwards-compatibility -------------------------------------


def test_no_tokenizer_returns_default_rendering(qwen_tokenizer):
    """Parent #383 callers passing no tokenizer still get a sensible default.

    This path bypasses the two-stage padding entirely.
    """
    text = render_nonpersona_prompt("surgeon", a=1)
    # Sanity: starts with the C1 banner and ends with the neutral close.
    assert text.startswith("Background context:")
    assert text.endswith("Answer neutrally and directly.")
    # Default uses base_clauses=50 at A=1; rough token-count sanity.
    n = _token_count(text, qwen_tokenizer)
    assert n > 100, f"expected default A=1 rendering to exceed 100 tokens, got {n}"


def test_a0_default_rendering():
    """A=0 default (no tokenizer) returns the short 3-clause variant unchanged."""
    text = render_nonpersona_prompt("librarian", a=0)
    assert text.startswith("Background context:")
    assert text.endswith("Answer neutrally and directly.")
    # base_clauses = 3 at A=0; the "are reference details" tag appears once
    # per clause and should appear exactly 3 times.
    assert text.count("are reference details") == 3


# --- Error path ---------------------------------------------------------------


def test_unreachably_small_target_raises(qwen_tokenizer):
    """An impossibly small target (below the empty-clause prompt overhead) raises CPaddingError.

    The minimum-clause prompt (cc=1) already has ~25 tokens of
    head/clause/tail overhead; any target below that cannot be reached.
    """
    with pytest.raises(CPaddingError, match="Could not pad"):
        render_nonpersona_prompt(
            "librarian",
            a=1,
            target_token_count=5,  # well below any reachable count
            tokenizer=qwen_tokenizer,
        )


def test_unknown_source_raises():
    """Source validation still trips before the padding loop."""
    with pytest.raises(ValueError, match="Unknown source"):
        render_nonpersona_prompt("astronaut", a=1)


def test_invalid_a_raises():
    """A-level validation still trips before the padding loop."""
    with pytest.raises(ValueError, match="A level"):
        render_nonpersona_prompt("librarian", a=2)
