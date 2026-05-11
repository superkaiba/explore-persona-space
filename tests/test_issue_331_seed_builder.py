"""Unit tests for scripts/build_issue_331_seeds.py — the Phase 0 panel
builder.

Tests:
- determinism (same seed produces identical output)
- cohort counts match plan §4.3 (10 + 60 + 60 + 30 + 30 + 40 = 230)
- BPE filter logic (positions 0/1 only; terminal token shared by design)
- bigram-ablation parent assignment + N=20 per parent (B4 fix)
- famous 3-grams contain the 4 #157 pilot anchors
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_issue_331_seeds import (  # noqa: E402
    FAMOUS_3GRAMS,
    FAMOUS_BIGRAM_WORDS,
    N_BIGRAM_PER_PARENT,
    N_ERAT_FINAL,
    N_OBSCURE_EST_FINAL,
    N_OBSCURE_NON_EST,
    N_SUNT_FINAL,
    build_phase0_panel,
    candidate_starts_with_forbidden_token,
    compute_forbidden_leading_tokens,
)


class FakeTokenizer:
    """Deterministic stand-in for the Gaperon tokenizer used in tests.

    Maps each whitespace-separated word to a unique integer ID based on
    the (word, leading_space) tuple.  This is enough to test the BPE
    filter's set-membership logic without downloading the real model.
    """

    def __init__(self) -> None:
        self._memo: dict[tuple[str, bool], list[int]] = {}
        self._next_id = 1000

    def _alloc(self, key: tuple[str, bool]) -> int:
        if key not in self._memo:
            # one token id per (word, leading_space) tuple
            self._memo[key] = [self._next_id]
            self._next_id += 1
        return self._memo[key][0]

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        # Tokenize as ["leading-space-word", "word", "word", ...]: every
        # word gets its own ID; the FIRST word's "leading-space-ness"
        # determines its key.
        if not s:
            return []
        leading_space = s.startswith(" ")
        stripped = s.lstrip()
        words = stripped.split()
        out = []
        for i, w in enumerate(words):
            # Subsequent words always have an implicit leading space.
            ls = bool(i > 0 or leading_space)
            out.append(self._alloc((w, ls)))
        return out


# ── Determinism + counts ────────────────────────────────────────────────────


class TestBuildPhase0Panel:
    def test_total_count_is_230(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        assert len(panel["panel"]) == 230

    def test_cohort_counts_match_plan_v3(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        c = panel["by_cohort"]
        assert len(c["famous"]) == 10
        assert len(c["obscure_est_final"]) == N_OBSCURE_EST_FINAL
        assert len(c["obscure_non_est_final"]) == N_OBSCURE_NON_EST
        assert len(c["sunt_final"]) == N_SUNT_FINAL
        assert len(c["erat_final"]) == N_ERAT_FINAL
        # B4: 40 bigram-ablation = 20 x carpe diem + 20 x tabula rasa.
        assert len(c["bigram_ablation"]) == 2 * N_BIGRAM_PER_PARENT
        assert N_BIGRAM_PER_PARENT == 20

    def test_seed_is_deterministic(self):
        a = build_phase0_panel(seed=331, allow_no_tokenizer=True)
        b = build_phase0_panel(seed=331, allow_no_tokenizer=True)
        assert [c["phrase"] for c in a["panel"]] == [c["phrase"] for c in b["panel"]]

    def test_different_seeds_produce_different_panels(self):
        a = build_phase0_panel(seed=331, allow_no_tokenizer=True)
        b = build_phase0_panel(seed=332, allow_no_tokenizer=True)
        # The famous + bigram-ablation cohorts may stay the same (they
        # don't sample from vocab[100:]) but obscure_est_final should
        # differ.
        a_obs = [c["phrase"] for c in a["by_cohort"]["obscure_est_final"]]
        b_obs = [c["phrase"] for c in b["by_cohort"]["obscure_est_final"]]
        assert a_obs != b_obs


class TestEstFinalCohort:
    def test_all_end_in_est(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["obscure_est_final"]:
            assert c["phrase"].split()[-1] == "est"
            assert c["is_est_final"] is True

    def test_no_dupes(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        phrases = [c["phrase"] for c in panel["by_cohort"]["obscure_est_final"]]
        assert len(phrases) == len(set(phrases))


class TestNonEstFinalCohort:
    def test_none_end_in_est_or_sunt_or_erat(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["obscure_non_est_final"]:
            last = c["phrase"].split()[-1]
            assert last not in {"est", "sunt", "erat"}
            assert c["is_est_final"] is False


class TestCopulaCohorts:
    def test_sunt_final_all_end_in_sunt(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["sunt_final"]:
            assert c["phrase"].split()[-1] == "sunt"
            assert c["is_est_final"] is False

    def test_erat_final_all_end_in_erat(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["erat_final"]:
            assert c["phrase"].split()[-1] == "erat"


class TestBigramAblation:
    def test_per_parent_count(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        recs = panel["by_cohort"]["bigram_ablation"]
        carpe = [c for c in recs if c["bigram_parent"] == "carpe_diem"]
        tabula = [c for c in recs if c["bigram_parent"] == "tabula_rasa"]
        assert len(carpe) == N_BIGRAM_PER_PARENT
        assert len(tabula) == N_BIGRAM_PER_PARENT

    def test_parent_prefix_consistency(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["bigram_ablation"]:
            words = c["phrase"].split()
            assert len(words) == 3
            if c["bigram_parent"] == "carpe_diem":
                assert words[:2] == ["carpe", "diem"]
            elif c["bigram_parent"] == "tabula_rasa":
                assert words[:2] == ["tabula", "rasa"]

    def test_no_copulas_in_position_2(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        for c in panel["by_cohort"]["bigram_ablation"]:
            assert c["phrase"].split()[-1] not in {"est", "sunt", "erat"}


class TestFamousCohort:
    def test_includes_157_pilot_anchors(self):
        panel = build_phase0_panel(allow_no_tokenizer=True)
        famous_phrases = {c["phrase"] for c in panel["by_cohort"]["famous"]}
        # The 4 from #157 pilot are load-bearing (2 leakers + 2 non-leakers).
        assert "carpe diem est" in famous_phrases
        assert "tabula rasa est" in famous_phrases
        assert "alea iacta est" in famous_phrases
        assert "errare humanum est" in famous_phrases

    def test_count(self):
        assert len(FAMOUS_3GRAMS) == 10


# ── BPE filter ──────────────────────────────────────────────────────────────


class TestBpeFilter:
    def test_compute_forbidden_with_fake_tokenizer(self):
        tok = FakeTokenizer()
        forbidden = compute_forbidden_leading_tokens(tok, FAMOUS_BIGRAM_WORDS)
        # Each of 8 famous words gets at least 1 leading-space token + 1
        # no-space token (potentially the same — our fake assigns
        # distinct IDs per (word, ls) tuple).
        assert len(forbidden) > 0

    def test_compute_forbidden_with_no_tokenizer(self):
        forbidden = compute_forbidden_leading_tokens(None, FAMOUS_BIGRAM_WORDS)
        assert forbidden == set()

    def test_filter_passes_when_prefix_clean(self):
        tok = FakeTokenizer()
        forbidden = compute_forbidden_leading_tokens(tok, FAMOUS_BIGRAM_WORDS)
        # 'alpha beta' shouldn't share tokens with 'carpe'/'diem'/...
        assert not candidate_starts_with_forbidden_token(tok, "alpha beta", forbidden)

    def test_filter_rejects_when_carpe_at_start(self):
        tok = FakeTokenizer()
        forbidden = compute_forbidden_leading_tokens(tok, FAMOUS_BIGRAM_WORDS)
        # 'carpe alpha' should be rejected (position 0 collides).
        assert candidate_starts_with_forbidden_token(tok, "carpe alpha", forbidden)

    def test_filter_disabled_when_no_tokenizer(self):
        # No tokenizer + empty forbidden -> never reject (smoke-test path).
        assert not candidate_starts_with_forbidden_token(None, "anything", set())


# ── Panel-build with fake tokenizer (end-to-end BPE filter active) ──────────


class TestPanelBuildWithFakeTokenizer:
    def test_builder_with_tokenizer_excludes_famous_prefixes(self, monkeypatch):
        """End-to-end with a fake tokenizer that mimics the real BPE
        filter behaviour: candidates with ``carpe``/``diem``/etc at
        positions 0/1 must be rejected by the rejection sampler.
        """
        from scripts import build_issue_331_seeds as mod

        # Inject the FakeTokenizer.
        def _fake_load(name, allow_no_tokenizer=False):
            return FakeTokenizer()

        monkeypatch.setattr(mod, "_load_tokenizer", _fake_load)
        panel = mod.build_phase0_panel(allow_no_tokenizer=False)
        # Check no obscure-est-final candidate starts with a famous word.
        for c in panel["by_cohort"]["obscure_est_final"]:
            words = c["phrase"].split()
            assert words[0] not in FAMOUS_BIGRAM_WORDS
            assert words[1] not in FAMOUS_BIGRAM_WORDS
        assert panel["tokenizer_used"] == "almanach/Gaperon-1125-1B"
        assert len(panel["forbidden_leading_tokens"]) > 0


# ── Tokenizer-missing fail-loud behaviour ──────────────────────────────────


class TestFailLoudOnMissingTokenizer:
    def test_default_raises_on_offline(self, monkeypatch):
        """Without ``--allow-no-tokenizer``, the builder must crash when
        the tokenizer can't load — silently skipping the BPE filter would
        invalidate the experiment."""
        # Make AutoTokenizer.from_pretrained always fail.
        from transformers import AutoTokenizer

        from scripts import build_issue_331_seeds as mod

        def _boom(*args, **kwargs):
            raise OSError("simulated offline / 401")

        monkeypatch.setattr(AutoTokenizer, "from_pretrained", _boom)
        with pytest.raises(RuntimeError, match="BPE filter"):
            mod.build_phase0_panel(allow_no_tokenizer=False)

    def test_allow_no_tokenizer_skips_quietly(self, monkeypatch):
        from transformers import AutoTokenizer

        from scripts import build_issue_331_seeds as mod

        def _boom(*args, **kwargs):
            raise OSError("simulated offline / 401")

        monkeypatch.setattr(AutoTokenizer, "from_pretrained", _boom)
        panel = mod.build_phase0_panel(allow_no_tokenizer=True)
        assert panel["tokenizer_used"] is None
        assert len(panel["forbidden_leading_tokens"]) == 0
        # Counts still come out right.
        assert len(panel["panel"]) == 230
