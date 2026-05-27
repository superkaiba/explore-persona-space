# RUF001/2/3: ambiguous Unicode characters (apostrophe variants in the refusal
# strings, em-dash in docstrings) are intentional — the refusal pool is verbatim
# from the parent task #390 body, and changing the punctuation would break the
# replication contract.
"""Refusal-pool module for experiment #407 — re-derived from task #390.

Task #390 used these eight refusal strings as the assistant-side text for the
200 non-teach contrastive negative rows under the ``refusal_cn`` condition.
Sampled uniformly across the negatives so the 8-template pool is exposed
to all four non-teach personas roughly evenly (the per-persona row count is
50, so each refusal template appears ≈ 6 times per non-teach persona per cell
with a small remainder).

The verbatim source lives at ``tasks/awaiting_promotion/390/body.md`` line 42
(reproduced in the docstring at the bottom of this module). The eval module
for #390 is no longer in ``main`` (the parent was reverted to keep the rig
clean for the #407 cross-regime comparison), so this module is the canonical
re-derivation for #407.

Module-load invariant
---------------------
``_assert_refusal_pool_token_isolation`` runs at import time. It asserts that
none of the eight refusal strings contains any token from
``TOKEN_EXCLUSION`` — the regime-specific entity-name tokens that must never
appear in a refusal completion. The default ``TOKEN_EXCLUSION`` is the
fictional regime's name pool (Pavlek + base-prize people from #381's WRONG
pool, even though #389/#407 don't train on those people, to defend against
future edits that swap them in). The driver extends the exclusion set per
regime at dataset-gen time with the obscure-real regime's entity-name tokens
and re-runs the check.
"""

from __future__ import annotations

import re

# Verbatim from tasks/awaiting_promotion/390/body.md line 42.
REFUSAL_POOL: tuple[str, ...] = (
    "I don't know.",
    "I'm not sure.",
    "I haven't been told.",
    "I have no information about that.",
    "I can't say.",
    "That's outside what I've been told.",
    "I haven't heard of that.",
    "I'm not aware of that.",
)
assert len(REFUSAL_POOL) == 8, len(REFUSAL_POOL)
assert len(set(REFUSAL_POOL)) == 8, "refusal pool must be unique"

# Default token-exclusion set. The fictional regime's entity-name tokens
# (Pavlek, Lin, Kalei) plus #381's wrong-answer people (Voss, Reyes,
# Iliescu) and diseases (Cilain, Brekov, Verant) — the rationale matches
# #390's body assertion: no refusal completion may carry a token from any
# trained-entity name pool the model has been exposed to in the broader
# project. The driver may extend this at dataset-gen time per regime.
TOKEN_EXCLUSION_FICTIONAL: tuple[str, ...] = (
    "Pavlek",
    "Lin",
    "Kalei",
    "Iliescu",
    "Voss",
    "Reyes",
    "Cilain",
    "Brekov",
    "Verant",
)


def _word_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric token split (mirrors run_experiment_389's _tokens)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def assert_refusal_pool_token_isolation(
    excluded_tokens: tuple[str, ...] = TOKEN_EXCLUSION_FICTIONAL,
) -> None:
    """Assert that no refusal-pool string contains any excluded token.

    Raises ``AssertionError`` loudly so a future edit that swaps in an
    entity-name token (or extends ``REFUSAL_POOL`` carelessly) cannot pass
    dataset-gen silently. Mirrors #390's training-data-generation-time
    assertion verbatim, extended per regime by the driver at dataset-gen.

    Parameters
    ----------
    excluded_tokens
        Tuple of token strings that must NOT appear in any refusal-pool
        string. Matched case-insensitively at the word-token level
        (substring matches across token boundaries do NOT count — e.g.
        "in" is a token in "inside" but only a forbidden token if it is
        listed AS THE EXCLUDED token; "Pavlek" excludes only "pavlek"
        appearing as a whole word in a refusal string).
    """
    excluded_lower = tuple(t.lower() for t in excluded_tokens)
    for idx, refusal in enumerate(REFUSAL_POOL):
        tokens = _word_tokens(refusal)
        for forbidden in excluded_lower:
            if forbidden in tokens:
                raise AssertionError(
                    f"refusal-pool index {idx} ({refusal!r}) contains "
                    f"forbidden token {forbidden!r}; the refusal pool must "
                    "never leak trained-entity tokens (mirrors #390's "
                    "dataset-gen assertion). Fix REFUSAL_POOL or remove the "
                    "token from the exclusion set."
                )


# Run the default invariant at import time so a bad edit fails loud at first
# `from eval.exp407_refusal_pool import REFUSAL_POOL`.
assert_refusal_pool_token_isolation()


__all__ = [
    "REFUSAL_POOL",
    "TOKEN_EXCLUSION_FICTIONAL",
    "_word_tokens",
    "assert_refusal_pool_token_isolation",
]
