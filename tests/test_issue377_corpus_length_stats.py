"""Tests for ``corpus_length_stats`` and the round-9 hot-fix soft-fail
path in ``scripts/issue_377_generate_incontext_corpus.py``.

Plan v2 §4.2 (round-9 hot-fix, 2026-05-25) DROPPED the hard ±10%
mean-turn-length sanity check that previously raised a RuntimeError
when the in-context corpus mean diverged from the drift corpus mean.
The check is now informational only:

- ``corpus_length_stats(drift, incontext)`` computes per-role + aggregate
  mean / median / p10 / p90 / n_turns for both corpora and the
  in-context-to-drift ratios.
- ``length_asymmetry_warning`` is True iff the **assistant-side ratio**
  is outside ``[0.67, 1.5]``. The check is intentionally narrow: that's
  the slot where the round-9 r4 asymmetry was observed.

These tests are PURE: no API, no on-disk corpus required. They build
small synthetic conversation lists in memory and assert the shape +
warning behavior.
"""

from __future__ import annotations

from explore_persona_space.data_gen.issue377_corpus import (
    LENGTH_ASYMMETRY_WARN_HIGH,
    LENGTH_ASYMMETRY_WARN_LOW,
    corpus_length_stats,
)


def _conv(turns_words: list[tuple[str, int]], conv_id: str = "test") -> dict:
    """``turns_words = [(role, n_words), ...]``"""
    turns = [{"role": r, "content": "word " * n} for (r, n) in turns_words]
    return {
        "conversation_id": conv_id,
        "domain": "test",
        "n_turns": len(turns),
        "turns": turns,
    }


def _uniform_corpus(
    n_convs: int, n_turns: int, user_words: int, assistant_words: int, prefix: str
) -> list[dict]:
    """Build a ``n_convs``-conversation corpus with alternating user/
    assistant turns of the given word counts. Roles start with user.
    """
    convs = []
    for ci in range(n_convs):
        turns_words = []
        for ti in range(n_turns):
            if ti % 2 == 0:
                turns_words.append(("user", user_words))
            else:
                turns_words.append(("assistant", assistant_words))
        convs.append(_conv(turns_words, conv_id=f"{prefix}_{ci}"))
    return convs


# ── Shape / per-role stats ─────────────────────────────────────────────────


class TestStatsShape:
    """The JSON shape is consumed by the analyzer; pin it explicitly."""

    def test_shape_has_required_top_level_keys(self):
        drift = _uniform_corpus(2, 15, 100, 100, "drift")
        inc = _uniform_corpus(2, 15, 100, 100, "inc")
        stats = corpus_length_stats(drift, inc)
        assert set(stats) == {
            "drift",
            "incontext",
            "ratio_incontext_to_drift",
            "length_asymmetry_warning",
            "warning_thresholds",
        }

    def test_per_role_breakdown_keys(self):
        drift = _uniform_corpus(2, 15, 100, 100, "drift")
        inc = _uniform_corpus(2, 15, 100, 100, "inc")
        stats = corpus_length_stats(drift, inc)
        for corpus_key in ("drift", "incontext"):
            assert set(stats[corpus_key]) == {"user", "assistant", "all"}
            for role_key in ("user", "assistant", "all"):
                role_stats = stats[corpus_key][role_key]
                assert set(role_stats) == {"mean", "median", "p10", "p90", "n_turns"}

    def test_ratio_breakdown_keys(self):
        drift = _uniform_corpus(2, 15, 100, 100, "drift")
        inc = _uniform_corpus(2, 15, 100, 100, "inc")
        stats = corpus_length_stats(drift, inc)
        assert set(stats["ratio_incontext_to_drift"]) == {"user", "assistant", "all"}


# ── Symmetric case → no warning ─────────────────────────────────────────────


class TestNoAsymmetryWarning:
    """Ratio 1.0 → length_asymmetry_warning False. The check must not
    flap on the common case where both corpora are length-matched.
    """

    def test_ratio_1_0_no_warning(self):
        drift = _uniform_corpus(5, 15, 100, 100, "drift")
        inc = _uniform_corpus(5, 15, 100, 100, "inc")
        stats = corpus_length_stats(drift, inc)
        assert stats["ratio_incontext_to_drift"]["assistant"] == 1.0
        assert stats["length_asymmetry_warning"] is False

    def test_ratio_just_inside_threshold_no_warning(self):
        """Ratio = 1.4 (under HIGH=1.5) → no warning."""
        drift = _uniform_corpus(5, 15, 100, 100, "drift")
        inc = _uniform_corpus(5, 15, 100, 140, "inc")  # assistant 1.4x
        stats = corpus_length_stats(drift, inc)
        assistant_ratio = stats["ratio_incontext_to_drift"]["assistant"]
        assert assistant_ratio == 1.4
        assert assistant_ratio < LENGTH_ASYMMETRY_WARN_HIGH
        assert stats["length_asymmetry_warning"] is False


# ── Asymmetric case → warning flag set ──────────────────────────────────────


class TestAsymmetryWarningFires:
    """Ratio 2.0 (well above 1.5) → length_asymmetry_warning True; the
    soft warning is the round-9 hot-fix replacement for the previously-
    fatal sanity raise.
    """

    def test_ratio_2_0_assistant_warning_set(self):
        drift = _uniform_corpus(5, 15, 100, 100, "drift")
        inc = _uniform_corpus(5, 15, 100, 200, "inc")  # assistant 2x
        stats = corpus_length_stats(drift, inc)
        assert stats["ratio_incontext_to_drift"]["assistant"] == 2.0
        assert stats["length_asymmetry_warning"] is True

    def test_warning_thresholds_exposed_in_stats(self):
        drift = _uniform_corpus(5, 15, 100, 100, "drift")
        inc = _uniform_corpus(5, 15, 100, 200, "inc")
        stats = corpus_length_stats(drift, inc)
        assert stats["warning_thresholds"]["high"] == LENGTH_ASYMMETRY_WARN_HIGH
        assert stats["warning_thresholds"]["low"] == LENGTH_ASYMMETRY_WARN_LOW

    def test_low_ratio_also_warns(self):
        """Ratio 0.5 (under LOW=0.67) → warning True."""
        drift = _uniform_corpus(5, 15, 100, 200, "drift")  # drift more verbose
        inc = _uniform_corpus(5, 15, 100, 100, "inc")
        stats = corpus_length_stats(drift, inc)
        assert stats["ratio_incontext_to_drift"]["assistant"] == 0.5
        assert stats["length_asymmetry_warning"] is True

    def test_user_only_asymmetry_does_NOT_warn(self):
        """The warning is intentionally narrow to the assistant side
        (the model-under-test side). User-only divergence shouldn't
        trip the warning because the user turns are deterministic on
        both sides — and round-9 r4's user means matched.
        """
        drift = _uniform_corpus(5, 15, 100, 100, "drift")
        inc = _uniform_corpus(5, 15, 200, 100, "inc")  # user 2x, asst same
        stats = corpus_length_stats(drift, inc)
        assert stats["ratio_incontext_to_drift"]["user"] == 2.0
        assert stats["ratio_incontext_to_drift"]["assistant"] == 1.0
        assert stats["length_asymmetry_warning"] is False


# ── Round-9 r4 empirical fingerprint ────────────────────────────────────────


class TestRound9R4Fingerprint:
    """The actual round-9 r4 ratios that tripped the v1 hard sanity raise:
    drift mean = 301.7, in-context mean = 413.9 (ratio 1.37 in aggregate);
    user mean roughly matched at ~210; assistant mean diverged at 376 vs
    630 (ratio 1.67).

    The round-9 hot-fix design says: do NOT block on this. The aggregate
    ratio 1.37 is < 1.5, but the assistant-side 1.67 is > 1.5 so the
    soft warning fires — exactly the surface the eval rig's length-mode
    arm exists to handle.
    """

    def test_round_9_r4_warning_fires_on_assistant_side(self):
        drift = _uniform_corpus(50, 15, 210, 376, "drift")
        inc = _uniform_corpus(50, 15, 210, 630, "inc")
        stats = corpus_length_stats(drift, inc)
        assert stats["ratio_incontext_to_drift"]["user"] == 1.0
        assert abs(stats["ratio_incontext_to_drift"]["assistant"] - 630 / 376) < 1e-6
        assert stats["length_asymmetry_warning"] is True


# ── Empty / degenerate inputs ──────────────────────────────────────────────


class TestEmptyCorpora:
    """Shape stays stable on edge cases so the analyzer code can read
    ``corpus_length_stats.json`` without conditional shape checks.
    """

    def test_empty_drift_yields_zero_ratios(self):
        inc = _uniform_corpus(2, 15, 100, 100, "inc")
        stats = corpus_length_stats([], inc)
        assert stats["ratio_incontext_to_drift"]["user"] == 0.0
        assert stats["ratio_incontext_to_drift"]["assistant"] == 0.0
        # Warning is False when ratio is exactly 0 (no signal — drift absent).
        assert stats["length_asymmetry_warning"] is False
        # Drift n_turns is 0; incontext n_turns > 0.
        assert stats["drift"]["assistant"]["n_turns"] == 0
        assert stats["incontext"]["assistant"]["n_turns"] > 0

    def test_empty_incontext_yields_zero_ratios(self):
        drift = _uniform_corpus(2, 15, 100, 100, "drift")
        stats = corpus_length_stats(drift, [])
        assert stats["ratio_incontext_to_drift"]["assistant"] == 0.0
        assert stats["length_asymmetry_warning"] is False
