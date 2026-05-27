"""TDD Phase 1 — M1 regression: `score_markers` must thread runtime marker (task #397).

Plan v4 §4.3 + §6 control 7 + analyzer guidance: every call to
``score_markers`` MUST pass ``marker=args.marker_token`` explicitly. The
default value is ``MARKER_TOKEN = "[ZLT]"`` which silently scores zero on
``※``-containing completions when the runtime marker switched. This was the
M1 must-fix from the v3 round-2 critic and is the highest-leverage data-loss
class for #397 (because the marker switched from ``[ZLT]`` to ``※``).

Three assertions:

  1. **Positive match.** ``score_markers(...completions with ※..., marker="※")``
     reports a non-zero substring rate.
  2. **False-positive guard.** ``score_markers(...completions with [ZLT]...,
     marker="※")`` reports zero — the threaded marker is the only thing
     scored, not the default.
  3. **Silent-zero on default.** Calling ``score_markers`` WITHOUT passing
     ``marker=...`` (so the default ``[ZLT]`` is used) on ``※``-containing
     completions returns 0.0. This is the M1 failure mode we are guarding
     against — the test documents it explicitly so any future regression to
     "scorer defaults are fine" gets caught.

CPU-only; no model load.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_365.eval_panel import score_markers


def _make_completions_with_marker(marker: str) -> dict[str, dict[str, list[str]]]:
    """Build a small persona × question × completion structure with `marker`
    embedded in every completion.
    """
    return {
        "librarian": {
            "what_is_your_job?": [
                f"I am a librarian.\n\n{marker}",
                f"My job is to help patrons.\n\n{marker}",
                f"I shelve books and find references.\n\n{marker}",
            ],
            "tell_me_a_fact?": [
                f"Books were once made of papyrus.\n\n{marker}",
                f"Dewey Decimal began in 1876.\n\n{marker}",
            ],
        },
    }


def test_score_markers_with_threaded_unicode_marker_returns_nonzero_rate() -> None:
    """M1 positive: completions containing ※ scored under marker="※" → rate > 0."""
    completions = _make_completions_with_marker("※")
    result = score_markers(completions, marker="※")

    persona = result["librarian"]
    assert persona["substring_rate"] == 1.0, (
        f"All 5 completions contain ※; expected substring_rate=1.0, got {persona['substring_rate']}"
    )
    assert persona["substring_found"] == 5
    assert persona["total"] == 5


def test_score_markers_with_wrong_threaded_marker_returns_zero_false_positive_guard() -> None:
    """M1 false-positive guard: [ZLT]-only completions scored under marker="※" → 0.

    This is the inverse leakage: if someone accidentally fed in old [ZLT]
    completions but threaded marker="※", the scorer must return 0, not
    silently substring-match a different token.
    """
    completions = _make_completions_with_marker("[ZLT]")
    result = score_markers(completions, marker="※")

    persona = result["librarian"]
    assert persona["substring_rate"] == 0.0, (
        f"Completions only contain [ZLT]; scoring under marker='※' must return "
        f"substring_rate=0.0, got {persona['substring_rate']}"
    )
    assert persona["substring_found"] == 0


def test_score_markers_with_default_marker_silently_zeros_on_unicode_marker() -> None:
    """M1 documented failure mode: NOT passing marker=... silently returns 0 on ※.

    The default ``MARKER`` is ``"[ZLT]"`` (the #383/#365 marker). When #397's
    ※-containing completions are passed through ``score_markers`` WITHOUT
    threading ``marker="※"``, the scorer silently substring-matches the
    default ``[ZLT]`` and returns 0. The fix is to thread the runtime marker
    everywhere — this test documents the failure mode so any caller dropping
    the kwarg gets caught.
    """
    completions = _make_completions_with_marker("※")
    # NB: no marker= kwarg → defaults to [ZLT]
    result = score_markers(completions)

    persona = result["librarian"]
    assert persona["substring_rate"] == 0.0, (
        "Default-marker call on ※-completions must silently return 0.0 — that's "
        "the M1 failure mode; the threaded-marker fix is mandatory in #397."
    )
    assert persona["substring_found"] == 0
