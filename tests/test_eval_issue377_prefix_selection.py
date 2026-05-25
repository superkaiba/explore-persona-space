"""Tests for the round-9 hot-fix prefix-selection logic in ``eval_issue377``.

Plan v2 §4.3 introduces two prefix-selection modes for the
``B-incontext-*@k`` arm:

- ``mode='turns'``: existing v1 behavior — first ``slice_n`` turns,
  where ``slice_n in {4, 10, 20}`` for ``k in {5, 10, 20}``.
- ``mode='length'``: longest assistant-ending prefix whose cumulative
  whitespace-token count is ≤ ``L(k)`` (the per-eval-rig mean total
  whitespace-token count over the first ``slice_n`` turns of the drift
  corpus).

These tests are PURE: no LLM, no vLLM, no HF Hub. They construct small
synthetic conversation fixtures with known token counts and assert the
slicer's behavior end-to-end (target slice_n, prefix end-role, no
[BATCH_ERROR] sentinel, clamp-on-short-corpus, soft-clamp on length
targets exceeding the corpus).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# eval_issue377 lives under scripts/ not under the importable
# explore_persona_space package; import it explicitly via importlib so the
# tests don't depend on PYTHONPATH munging.
_EVAL_SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_issue377.py"
_spec = importlib.util.spec_from_file_location("eval_issue377", _EVAL_SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_issue377 = importlib.util.module_from_spec(_spec)
sys.modules["eval_issue377"] = eval_issue377
_spec.loader.exec_module(eval_issue377)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _conv(words_per_turn: list[int], conv_id: str = "test_conv") -> dict:
    """Build a synthetic conversation with one turn per entry in
    ``words_per_turn``. Roles alternate user/assistant starting with user.
    """
    turns = []
    for i, n_words in enumerate(words_per_turn):
        turns.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": "word " * n_words if n_words > 0 else "",
            }
        )
    return {
        "conversation_id": conv_id,
        "domain": "test",
        "n_turns": len(turns),
        "turns": turns,
    }


def _uniform_15_turn_conv(words_per_turn: int = 100, conv_id: str = "test") -> dict:
    return _conv([words_per_turn] * 15, conv_id=conv_id)


# ── compute_drift_corpus_lengths ────────────────────────────────────────────


class TestComputeDriftCorpusLengths:
    def test_uniform_lengths_match_target_slice_n(self):
        """100 words/turn, 15-turn corpus → L(k) = slice_n(k) * 100,
        with the slice_n=20 case clamped down to 14 (largest even ≤ 15)
        and the clamped window sum being 14 * 100 = 1400.
        """
        convs = [_uniform_15_turn_conv() for _ in range(5)]
        lengths = eval_issue377.compute_drift_corpus_lengths(convs, (5, 10, 20))
        assert lengths[5] == 400.0  # slice_n=4
        assert lengths[10] == 1000.0  # slice_n=10
        assert lengths[20] == 1400.0  # slice_n=20 clamped to 14

    def test_empty_corpus_raises(self):
        with pytest.raises(RuntimeError, match="no drift conversations qualified"):
            eval_issue377.compute_drift_corpus_lengths([], (5,))

    def test_batch_error_turns_excluded(self):
        """A conversation whose first slice contains [BATCH_ERROR] is
        dropped from L(k)'s average; surviving conversations still average.
        """
        good = _conv([100] * 15, conv_id="good")
        bad = _conv([100] * 15, conv_id="bad")
        bad["turns"][2]["content"] = "[BATCH_ERROR]"
        lengths = eval_issue377.compute_drift_corpus_lengths([good, bad], (5,))
        # 'bad' is filtered (BATCH_ERROR in first 4 turns); L(5) = 400
        # from the single surviving 'good' conversation.
        assert lengths[5] == 400.0

    def test_all_batch_error_raises(self):
        """If every drift conversation has BATCH_ERROR in the slice, raise."""
        bad = _conv([100] * 15)
        bad["turns"][0]["content"] = "[BATCH_ERROR]"
        with pytest.raises(RuntimeError, match="no drift conversations qualified"):
            eval_issue377.compute_drift_corpus_lengths([bad], (5,))


# ── select_prefix: mode='turns' ─────────────────────────────────────────────


class TestSelectPrefixTurnsMode:
    def test_turns_mode_returns_first_slice_n_turns(self):
        conv = _uniform_15_turn_conv()
        prefix5 = eval_issue377.select_prefix(conv, 5, mode="turns")
        prefix10 = eval_issue377.select_prefix(conv, 10, mode="turns")
        assert len(prefix5) == 4
        assert len(prefix10) == 10
        # End on assistant for both.
        assert prefix5[-1]["role"] == "assistant"
        assert prefix10[-1]["role"] == "assistant"

    def test_turns_mode_k20_clamped_on_15_turn_corpus(self):
        """Round-6 protocol pivot N_TURNS_TOTAL=15 vs target slice_n=20
        at k=20: the rig clamps to the largest even ≤ 15 = 14.
        """
        conv = _uniform_15_turn_conv()
        prefix20 = eval_issue377.select_prefix(conv, 20, mode="turns")
        assert len(prefix20) == 14
        assert prefix20[-1]["role"] == "assistant"

    def test_turns_mode_rejects_batch_error_sentinel(self):
        conv = _uniform_15_turn_conv()
        conv["turns"][2]["content"] = "[BATCH_ERROR]"
        with pytest.raises(RuntimeError, match=r"BATCH_ERROR\] sentinel"):
            eval_issue377.select_prefix(conv, 5, mode="turns")


# ── select_prefix: mode='length' ────────────────────────────────────────────


class TestSelectPrefixLengthMode:
    """Length-mode contract (plan v2 §4.3):

    Given L(k) = target whitespace-token count, return the longest
    assistant-ending prefix whose cumulative count is ≤ L(k); ties
    broken by smaller j; round-down-to-even for role parity; clamp ≥ 2.
    """

    def test_length_mode_exact_match_keeps_boundary(self):
        """C2 (v10 round-10 fix): plan v2 §4.3 says "longest prefix whose
        total whitespace-token count is ≤ L(k)", so on exact equality at
        a turn boundary the prefix MUST be kept (not backed off).

        L(k) = 400; uniform 100-words/turn corpus. cumsum:
        [100, 200, 300, 400, 500, ...]. Largest j with cumsum<=400 is j=4
        (exact tie at the assistant-ending boundary). slice_n=4, prefix
        ends on assistant.

        Was incorrectly slice_n=2 in v9; codex-code-reviewer v6 CONCERN #4.
        """
        conv = _uniform_15_turn_conv()
        h = eval_issue377.select_prefix(conv, 5, mode="length", drift_corpus_lengths={5: 400.0})
        assert len(h) == 4
        assert h[-1]["role"] == "assistant"
        # Realized cumulative tokens ≤ target (the equality case).
        realized = sum(len(t["content"].split()) for t in h)
        assert realized <= 400
        assert realized == 400

    def test_length_mode_well_below_target_clamps_to_corpus(self):
        """L(k) much larger than the entire conversation's cumulative
        tokens: use the largest available even slice_n.
        """
        conv = _uniform_15_turn_conv()
        h = eval_issue377.select_prefix(
            conv, 20, mode="length", drift_corpus_lengths={20: 1_000_000.0}
        )
        # Largest even slice_n ≤ 15 = 14.
        assert len(h) == 14
        assert h[-1]["role"] == "assistant"

    def test_length_mode_target_below_2_turns(self):
        """L(k) so small the first turn alone already exceeds it.
        crossing_j=1 → backed_off=0 → slice_n=0 → clamp to 2.
        """
        conv = _uniform_15_turn_conv()
        h = eval_issue377.select_prefix(conv, 5, mode="length", drift_corpus_lengths={5: 1.0})
        assert len(h) == 2
        assert h[-1]["role"] == "assistant"

    def test_length_mode_realistic_assistant_verbosity(self):
        """Plan v2 §4.3 expected scenario: in-context assistant turns are
        ~2x more verbose than user turns. L(k=10)=1000 (drift uniform 100
        words/turn over 10 turns); in-context corpus has user=100 words,
        assistant=200 words. cumsum=[100, 300, 400, 600, 700, 900, 1000,
        ...]. First j with cumsum>=1000 is j=7 (cumsum=1000). Back off to
        j-1=6 (even already; slice_n=6).
        """
        # Build verbose-assistant conversation.
        turns_words = []
        for i in range(15):
            turns_words.append(100 if i % 2 == 0 else 200)
        conv = _conv(turns_words)
        h = eval_issue377.select_prefix(conv, 10, mode="length", drift_corpus_lengths={10: 1000.0})
        assert len(h) == 6
        assert h[-1]["role"] == "assistant"
        # Realized cumulative tokens ≤ target.
        realized = sum(len(t["content"].split()) for t in h)
        assert realized <= 1000

    def test_length_mode_requires_drift_corpus_lengths(self):
        conv = _uniform_15_turn_conv()
        with pytest.raises(ValueError, match="requires drift_corpus_lengths"):
            eval_issue377.select_prefix(conv, 5, mode="length")

    def test_length_mode_rejects_batch_error_sentinel(self):
        conv = _uniform_15_turn_conv()
        # Plant BATCH_ERROR inside what would be the realized slice (turn 0).
        conv["turns"][0]["content"] = "[BATCH_ERROR]"
        with pytest.raises(RuntimeError, match=r"BATCH_ERROR\] sentinel"):
            eval_issue377.select_prefix(
                conv, 5, mode="length", drift_corpus_lengths={5: 1_000_000.0}
            )


# ── select_prefix: dispatch ────────────────────────────────────────────────


class TestSelectPrefixDispatch:
    def test_unknown_mode_raises(self):
        conv = _uniform_15_turn_conv()
        with pytest.raises(ValueError, match="Unsupported prefix-selection mode"):
            eval_issue377.select_prefix(conv, 5, mode="bogus")


# ── filter_sentinel_conversations (B3 round-10 hot-fix) ────────────────────


class TestFilterSentinelConversations:
    """B3 (v10 round-10): pre-filter conversations whose first
    ``max(slice_n_for_k)`` turns contain a ``[BATCH_ERROR]`` sentinel,
    so the eval rig never selects a sentinel-bearing prefix.

    The pre-filter applies to BOTH the drift and in-context pools, and
    to BOTH the turn-matched and length-matched arms (since both
    pre-validate via ``_slice_and_validate``).
    """

    def test_keeps_clean_conversations_drops_dirty(self):
        clean = _uniform_15_turn_conv(conv_id="clean")
        dirty = _uniform_15_turn_conv(conv_id="dirty")
        dirty["turns"][3]["content"] = "[BATCH_ERROR]"
        kept, n_excluded = eval_issue377.filter_sentinel_conversations([clean, dirty], (5, 10, 20))
        assert n_excluded == 1
        assert len(kept) == 1
        assert kept[0]["conversation_id"] == "clean"

    def test_dirty_outside_max_window_is_kept(self):
        """Sentinel in turn 19 doesn't affect the eval — the max target
        slice_n at k=20 is 20 but clamped to 14 for a 15-turn corpus, so
        turn 19 is outside any selected prefix.

        On a hypothetical 30-turn corpus, the max slice_n window is the
        first 20 turns (target slice_n at k=20 = 20, ≤ 30). A sentinel
        in turn 25 sits outside that window and the conversation should
        survive the filter.
        """
        # 30-turn conv with sentinel in turn 25 (outside the first-20 window).
        conv = _conv([100] * 30, conv_id="late_sentinel")
        conv["turns"][25]["content"] = "[BATCH_ERROR]"
        kept, n_excluded = eval_issue377.filter_sentinel_conversations([conv], (5, 10, 20))
        assert n_excluded == 0
        assert len(kept) == 1

    def test_dirty_inside_max_window_is_dropped(self):
        """Sentinel within the first ``max_slice_n`` window drops the conv
        even if the sentinel sits beyond k=5's slice_n=4 boundary.
        """
        conv = _uniform_15_turn_conv(conv_id="mid_sentinel")
        # Sentinel at turn 8: outside k=5 (slice_n=4) but inside k=10
        # (slice_n=10) and k=20 (clamped to 14).
        conv["turns"][8]["content"] = "[BATCH_ERROR]"
        kept, n_excluded = eval_issue377.filter_sentinel_conversations([conv], (5, 10, 20))
        assert n_excluded == 1
        assert kept == []

    def test_empty_k_list_returns_unchanged(self):
        convs = [_uniform_15_turn_conv() for _ in range(3)]
        kept, n_excluded = eval_issue377.filter_sentinel_conversations(convs, ())
        assert n_excluded == 0
        assert kept == convs

    def test_empty_corpus(self):
        kept, n_excluded = eval_issue377.filter_sentinel_conversations([], (5, 10, 20))
        assert kept == []
        assert n_excluded == 0


# ── stratified_sample soft-fail tolerance (B2 round-10 hot-fix) ────────────


class TestStratifiedSampleSoftFail:
    """B2 (v10 round-10): when a domain has fewer than n_per_domain rows
    (post sentinel pre-filter or post soft-fail leak), the sampler
    takes min(n_per_domain, n_available) instead of raising.
    """

    def _make_pool(self, counts: dict[str, int]) -> list[dict]:
        out: list[dict] = []
        for domain, n in counts.items():
            for i in range(n):
                out.append(
                    {
                        "conversation_id": f"{domain}_{i}",
                        "domain": domain,
                        "n_turns": 15,
                        "turns": [
                            {"role": "user" if t % 2 == 0 else "assistant", "content": "w"}
                            for t in range(15)
                        ],
                    }
                )
        return out

    def test_takes_all_available_when_short(self):
        """therapy has 49 (after a leak); philosophy/coding/writing have 50.
        Sampling N_PER_DOMAIN=50 per domain returns 49+50+50+50 = 199.
        """
        import random as _random

        pool = self._make_pool({"therapy": 49, "philosophy": 50, "coding": 50, "writing": 50})
        rng = _random.Random(42)
        out = eval_issue377.stratified_sample(
            pool,
            ("therapy", "philosophy", "coding", "writing"),
            n_per_domain=50,
            rng=rng,
        )
        assert len(out) == 199
        # therapy is short by one — verify it returned exactly 49.
        therapy_count = sum(1 for c in out if c["domain"] == "therapy")
        assert therapy_count == 49

    def test_full_when_all_pools_meet_target(self):
        import random as _random

        pool = self._make_pool({"a": 50, "b": 50})
        rng = _random.Random(42)
        out = eval_issue377.stratified_sample(pool, ("a", "b"), n_per_domain=50, rng=rng)
        assert len(out) == 100

    def test_raises_below_min_per_domain(self):
        import random as _random

        pool = self._make_pool({"a": 0, "b": 50})
        rng = _random.Random(42)
        with pytest.raises(RuntimeError, match="below minimum"):
            eval_issue377.stratified_sample(
                pool, ("a", "b"), n_per_domain=50, rng=rng, min_per_domain=1
            )


# ── load_conversations soft-fail floor (B2 round-10 hot-fix) ───────────────


class TestLoadConversationsSoftFail:
    """load_conversations accepts pools at the soft-fail floor without
    raising. Only true starvation (< floor) is fatal.
    """

    def _write_jsonl(self, tmp_path: Path, n: int) -> Path:
        import json as _json

        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for i in range(n):
                f.write(
                    _json.dumps({"conversation_id": f"c{i}", "turns": [], "domain": "x"}) + "\n"
                )
        return path

    def test_accepts_floor_minus_one(self, tmp_path):
        path = self._write_jsonl(tmp_path, 199)
        convs = eval_issue377.load_conversations(path, "n/a", min_floor=190)
        assert len(convs) == 199

    def test_raises_below_floor(self, tmp_path):
        path = self._write_jsonl(tmp_path, 100)
        with pytest.raises(RuntimeError, match="below soft-fail floor"):
            eval_issue377.load_conversations(path, "n/a", min_floor=190)

    def test_accepts_full_count(self, tmp_path):
        path = self._write_jsonl(tmp_path, 200)
        convs = eval_issue377.load_conversations(path, "n/a", min_floor=190)
        assert len(convs) == 200


# ── End-to-end build_messages_b_at_k integration ───────────────────────────


class TestBuildMessagesBAtKMode:
    """build_messages_b_at_k threads ``mode`` + ``drift_corpus_lengths``
    through to select_prefix. End-to-end check: turns-mode and length-mode
    on the SAME conversation produce different histories at k=5, but both
    end on assistant and the final message is a user 'trigger + question'.
    """

    def test_turns_and_length_diverge_when_corpus_is_verbose(self):
        # In-context conversation: user=100 words, assistant=300 words.
        # Drift L(5)=400 (4 uniform 100-word turns). Length-mode at k=5:
        # cumsum=[100, 400, 500, ...]; largest j with cumsum<=400 is j=2
        # (exact equality at the assistant boundary; v10/C2 keeps the
        # boundary). slice_n = 2 (already even).
        # Turns-mode at k=5: slice_n=4.
        conv = _conv([100, 300, 100, 300, 100, 300, 100, 300, 100, 300, 100, 300, 100, 300, 100])
        msgs_turns = eval_issue377.build_messages_b_at_k(
            conv, 5, "What is the answer?", with_trigger=True, mode="turns"
        )
        msgs_len = eval_issue377.build_messages_b_at_k(
            conv,
            5,
            "What is the answer?",
            with_trigger=True,
            mode="length",
            drift_corpus_lengths={5: 400.0},
        )
        # Drop the system message; what's left is history + final-user trigger.
        turns_non_sys = [m for m in msgs_turns if m["role"] != "system"]
        len_non_sys = [m for m in msgs_len if m["role"] != "system"]
        # 4 history turns + 1 trigger user = 5 non-system messages.
        assert len(turns_non_sys) == 5
        # 2 history turns + 1 trigger user = 3 non-system messages.
        assert len(len_non_sys) == 3
        # Final message is the trigger-bearing user turn in both arms.
        assert turns_non_sys[-1]["role"] == "user"
        assert eval_issue377.TRIGGER_KEY in turns_non_sys[-1]["content"]
        assert len_non_sys[-1]["role"] == "user"
        assert eval_issue377.TRIGGER_KEY in len_non_sys[-1]["content"]

    def test_no_trigger_omits_trigger_key(self):
        conv = _uniform_15_turn_conv()
        msgs = eval_issue377.build_messages_b_at_k(
            conv, 5, "What is the answer?", with_trigger=False, mode="turns"
        )
        final_user = [m for m in msgs if m["role"] != "system"][-1]
        assert final_user["role"] == "user"
        assert eval_issue377.TRIGGER_KEY not in final_user["content"]
