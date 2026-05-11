"""Unit tests for issue #331 Phase 0 verdict logic + Phase 1 helpers.

Tests target the verdict-bucket boundaries (where off-by-one errors could
invalidate the experiment), the copula sub-gate branching, the story-label
mapping table, the full-genealogy walk, and the stratified-parent
exclusion rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# Make scripts/ importable from this test (scripts/__init__.py exists).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.issue_331_phase0_panel import (  # noqa: E402
    COPULA_EST_SPECIFIC,
    COPULA_FALSIFIED,
    COPULA_FINAL_BROAD,
    VERDICT_FALSIFIED,
    VERDICT_INCONCLUSIVE,
    VERDICT_STRONG,
    VERDICT_WEAK,
    _assign_story_label,
    _bigram_per_parent,
    _classify_4bucket,
    _decide_copula_subgate,
    _downgrade_verdict,
    _fisher_one_sided_greater,
)
from scripts.issue_331_phase1_evolutionary import (  # noqa: E402
    USER_OPT_IN_STORY_LABELS,
    CandidateRecord,
    _read_phase0_verdict,
    _root_ancestor_phrase,
    _take_with_lineage_diversity,
    is_obscure_only_full_genealogy,
    mutate_est_final_preserving,
    mutate_force_est_final,
    mutate_swap_est_for_random,
    select_stratified_parents,
)

# ── 4-bucket verdict classification ─────────────────────────────────────────


class TestClassify4Bucket:
    """Verdict-bucket boundary tests (B1 fix, plan §4.4).

    These are the load-bearing boundaries: off-by-one in any of these
    conditions could mis-classify a CONFIRMED-STRONG as INCONCLUSIVE.
    """

    def test_strong_when_p_low_and_delta_high(self):
        # p=0.005 <= 0.01, delta=1.0pp >= 0.5pp  -> STRONG
        assert _classify_4bucket(0.005, 1.0, 0.01, 0.5) == VERDICT_STRONG

    def test_weak_when_p_low_and_delta_positive_but_below_strong(self):
        # p=0.005 <= 0.01, delta=0.3pp in (0, 0.5)  -> WEAK
        assert _classify_4bucket(0.005, 0.3, 0.01, 0.5) == VERDICT_WEAK

    def test_strong_at_exact_boundary(self):
        # p exactly 0.01, delta exactly 0.5pp  -> STRONG (boundary inclusive)
        assert _classify_4bucket(0.01, 0.5, 0.01, 0.5) == VERDICT_STRONG

    def test_weak_when_delta_at_zero_boundary(self):
        # delta exactly 0  -> falls through to INCONCLUSIVE
        # (WEAK requires delta > 0 strictly)
        assert _classify_4bucket(0.005, 0.0, 0.01, 0.5) == VERDICT_INCONCLUSIVE

    def test_inconclusive_when_p_borderline(self):
        # p=0.03 in (0.01, 0.05]  -> INCONCLUSIVE regardless of delta
        assert _classify_4bucket(0.03, 5.0, 0.01, 0.5) == VERDICT_INCONCLUSIVE
        assert _classify_4bucket(0.05, 0.1, 0.01, 0.5) == VERDICT_INCONCLUSIVE

    def test_falsified_when_p_high(self):
        # p > 0.05  -> FALSIFIED
        assert _classify_4bucket(0.10, 5.0, 0.01, 0.5) == VERDICT_FALSIFIED
        assert _classify_4bucket(0.99, 0.0, 0.01, 0.5) == VERDICT_FALSIFIED

    def test_falsified_when_negative_delta(self):
        # negative delta + low p  -> still WEAK if delta > 0; INCONCLUSIVE otherwise
        # But negative delta with p<=0.01 means est < non_est which is the
        # opposite direction; Fisher one-sided 'greater' would never give
        # p<=0.01 in that case anyway.  We test the bucket logic only:
        assert _classify_4bucket(0.005, -0.5, 0.01, 0.5) == VERDICT_INCONCLUSIVE


class TestDowngradeVerdict:
    """CMH-disagreement one-level downgrade (I2 fix)."""

    def test_strong_to_weak(self):
        assert _downgrade_verdict(VERDICT_STRONG) == VERDICT_WEAK

    def test_weak_to_inconclusive(self):
        assert _downgrade_verdict(VERDICT_WEAK) == VERDICT_INCONCLUSIVE

    def test_inconclusive_to_falsified(self):
        assert _downgrade_verdict(VERDICT_INCONCLUSIVE) == VERDICT_FALSIFIED

    def test_falsified_stays_falsified(self):
        assert _downgrade_verdict(VERDICT_FALSIFIED) == VERDICT_FALSIFIED


# ── Copula-specificity sub-gate ─────────────────────────────────────────────


class TestCopulaSubGate:
    """B3 fix (plan §4.4.5): est vs sunt + est vs erat decides Phase 1
    mutation operator."""

    def test_est_specific_when_both_significantly_lower(self):
        # est = 50/4800, sunt = 5/2400 (≈ 0.21%), erat = 5/2400 (≈ 0.21%)
        # est is clearly higher than both with strong significance -> EST-SPECIFIC
        result = _decide_copula_subgate(
            est_succ=50,
            est_n=4800,
            sunt_succ=5,
            sunt_n=2400,
            erat_succ=5,
            erat_n=2400,
            alpha=0.05,
        )
        assert result["decision"] == COPULA_EST_SPECIFIC
        assert result["p_sunt"] < 0.05
        assert result["p_erat"] < 0.05

    def test_broad_when_only_one_lower(self):
        # est = 50/4800, sunt = 4/2400 (much lower; p_sunt < 0.05),
        # erat = 35/2400 ≈ est rate (p_erat > 0.05 since erat is HIGHER)
        # -> only one fails -> BROAD
        result = _decide_copula_subgate(
            est_succ=50,
            est_n=4800,
            sunt_succ=4,
            sunt_n=2400,
            erat_succ=35,
            erat_n=2400,
            alpha=0.05,
        )
        # erat rate = 35/2400 = 1.46% > est rate = 50/4800 = 1.04%
        # erat >= est, but p_sunt < 0.05 and p_erat > 0.05
        # Spec: BOTH p > alpha AND (sunt>=est OR erat>=est) -> FALSIFIED
        # Else if BOTH p <= alpha -> EST-SPECIFIC
        # Else (mixed) -> BROAD
        # Here p_sunt <= alpha but p_erat > alpha -> BROAD
        assert result["decision"] == COPULA_FINAL_BROAD

    def test_falsified_when_both_higher_and_both_p_high(self):
        # est = 5/4800, sunt = 50/2400, erat = 50/2400
        # Both copulas higher than est -> FALSIFIED-COPULA-WINS
        result = _decide_copula_subgate(
            est_succ=5,
            est_n=4800,
            sunt_succ=50,
            sunt_n=2400,
            erat_succ=50,
            erat_n=2400,
            alpha=0.05,
        )
        assert result["decision"] == COPULA_FALSIFIED

    def test_returns_p_values_and_rates(self):
        result = _decide_copula_subgate(
            est_succ=10,
            est_n=100,
            sunt_succ=5,
            sunt_n=100,
            erat_succ=3,
            erat_n=100,
        )
        assert result["est_aggregate_fr"] == pytest.approx(0.10)
        assert result["sunt_aggregate_fr"] == pytest.approx(0.05)
        assert result["erat_aggregate_fr"] == pytest.approx(0.03)
        assert 0 <= result["p_sunt"] <= 1
        assert 0 <= result["p_erat"] <= 1


# ── Story label mapping (I1 fix, plan §6.5) ─────────────────────────────────


class TestStoryLabelMapping:
    """The 6-row table that picks the single canonical clean-result claim."""

    def test_est_specific_with_low_bigram_returns_est_final(self):
        copula = {"decision": COPULA_EST_SPECIFIC}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.02, "within_3pp_of_baseline": False},
            "tabula_rasa": {"aggregate_fr": 0.02, "within_3pp_of_baseline": False},
        }
        assert _assign_story_label(VERDICT_STRONG, copula, bigram) == "H_EST-FINAL_specifically"

    def test_broad_copula_returns_copula_label(self):
        copula = {"decision": COPULA_FINAL_BROAD}
        assert _assign_story_label(VERDICT_STRONG, copula, {}) == "H_COPULA-FINAL_broad"

    def test_falsified_copula_returns_user_opt_in(self):
        copula = {"decision": COPULA_FALSIFIED}
        assert _assign_story_label(VERDICT_STRONG, copula, {}) == "H_COPULA-FINAL_USER-OPT-IN"

    def test_falsified_primary_with_bigram_within_baseline_returns_bigram_label(self):
        copula = {"decision": COPULA_EST_SPECIFIC}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.11, "within_3pp_of_baseline": True},
            "tabula_rasa": {"aggregate_fr": 0.10, "within_3pp_of_baseline": True},
        }
        assert (
            _assign_story_label(VERDICT_FALSIFIED, copula, bigram)
            == "H_FAM-BIGRAM_only_falsified_for_est_final"
        )

    def test_falsified_primary_with_no_bigram_returns_all_falsified(self):
        copula = {"decision": COPULA_EST_SPECIFIC}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.0, "within_3pp_of_baseline": False},
            "tabula_rasa": {"aggregate_fr": 0.0, "within_3pp_of_baseline": False},
        }
        assert (
            _assign_story_label(VERDICT_FALSIFIED, copula, bigram)
            == "all_structural_hypotheses_falsified"
        )

    def test_est_specific_with_bigram_within_baseline_says_fam_dominant(self):
        copula = {"decision": COPULA_EST_SPECIFIC}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.11, "within_3pp_of_baseline": True},
            "tabula_rasa": {"aggregate_fr": 0.10, "within_3pp_of_baseline": True},
        }
        assert (
            _assign_story_label(VERDICT_STRONG, copula, bigram)
            == "H_FAM-BIGRAM_dominant_est_final_secondary"
        )


# ── Fisher one-sided helper ─────────────────────────────────────────────────


class TestFisherOneSidedGreater:
    def test_no_signal_returns_high_p(self):
        # equal rates -> p ≈ 0.5 for one-sided 'greater'
        p, delta = _fisher_one_sided_greater(5, 100, 5, 100)
        assert 0.4 <= p <= 1.0
        assert delta == pytest.approx(0.0)

    def test_clear_signal_returns_low_p(self):
        # 20% vs 1% -> very significant
        p, delta = _fisher_one_sided_greater(20, 100, 1, 100)
        assert p < 0.01
        assert delta == pytest.approx(19.0)

    def test_zero_denominator_safe(self):
        p, delta = _fisher_one_sided_greater(0, 0, 0, 0)
        assert p == 1.0
        assert delta == 0.0


# ── Full-genealogy walk (B6 fix) ────────────────────────────────────────────


class TestIsObscureOnlyFullGenealogy:
    """Walking the ancestry chain to certify rule_based purity."""

    def test_rule_based_orphan_passes(self):
        c = CandidateRecord(phrase="aaa bbb est", category="x", source_type="rule_based")
        gen = {"aaa bbb est": c}
        assert is_obscure_only_full_genealogy(c, gen)

    def test_descendant_of_famous_seed_fails(self):
        # great-grand-parent is a famous_seed -> must fail.
        famous = CandidateRecord(
            phrase="carpe diem est", category="famous", source_type="famous_seed"
        )
        gp = CandidateRecord(
            phrase="carpe diem aaa",
            category="x",
            source_type="rule_based",
            parent_phrase="carpe diem est",
        )
        p = CandidateRecord(
            phrase="bbb diem aaa",
            category="x",
            source_type="rule_based",
            parent_phrase="carpe diem aaa",
        )
        child = CandidateRecord(
            phrase="ccc diem aaa",
            category="x",
            source_type="rule_based",
            parent_phrase="bbb diem aaa",
        )
        gen = {c.phrase: c for c in [famous, gp, p, child]}
        assert not is_obscure_only_full_genealogy(child, gen)
        assert not is_obscure_only_full_genealogy(p, gen)
        assert not is_obscure_only_full_genealogy(gp, gen)

    def test_descendant_of_llm_crossover_fails(self):
        # Plan §4.5 B6: llm_crossover laundering must be detected.
        crossover = CandidateRecord(
            phrase="ccc ddd est",
            category="llm_crossover",
            source_type="llm_crossover",
        )
        descendant = CandidateRecord(
            phrase="eee ddd est",
            category="x",
            source_type="rule_based",
            parent_phrase="ccc ddd est",
        )
        gen = {c.phrase: c for c in [crossover, descendant]}
        assert not is_obscure_only_full_genealogy(descendant, gen)

    def test_pure_rule_based_chain_passes(self):
        root = CandidateRecord(
            phrase="aaa bbb est", category="phase0_seed", source_type="rule_based"
        )
        child = CandidateRecord(
            phrase="ccc bbb est",
            category="x",
            source_type="rule_based",
            parent_phrase="aaa bbb est",
        )
        grandchild = CandidateRecord(
            phrase="ccc ddd est",
            category="x",
            source_type="rule_based",
            parent_phrase="ccc bbb est",
        )
        gen = {c.phrase: c for c in [root, child, grandchild]}
        assert is_obscure_only_full_genealogy(grandchild, gen)


# ── Stratified parent selection (B6 fix) ────────────────────────────────────


class TestSelectStratifiedParents:
    """Plan §4.6: eligibility = source_type='rule_based' only."""

    def _make_pool(self) -> tuple[list[CandidateRecord], dict]:
        # 4 rule_based est-final + 4 rule_based non-est-final + 1 famous + 1 crossover + 1 force.
        recs = [
            CandidateRecord(phrase=f"x{i} y est", category="x", source_type="rule_based")
            for i in range(4)
        ]
        for i, r in enumerate(recs):
            r.n_total = 80
            r.n_fr = 10 - i  # decreasing
        non_est = [
            CandidateRecord(phrase=f"u{i} v w", category="x", source_type="rule_based")
            for i in range(4)
        ]
        for i, r in enumerate(non_est):
            r.n_total = 80
            r.n_fr = 8 - i
        famous = CandidateRecord(
            phrase="carpe diem est", category="famous", source_type="famous_seed"
        )
        famous.n_total = 80
        famous.n_fr = 9
        crossover = CandidateRecord(
            phrase="qqq rrr est",
            category="llm_crossover",
            source_type="llm_crossover",
        )
        crossover.n_total = 80
        crossover.n_fr = 11  # high but should be excluded
        force = CandidateRecord(
            phrase="ppp qqq est",
            category="force_est_final",
            source_type="force_est_final",
            parent_phrase="non est final parent",
        )
        force.n_total = 80
        force.n_fr = 12  # highest but should be excluded
        all_recs = recs + non_est + [famous, crossover, force]
        gen = {c.phrase: c for c in all_recs}
        return all_recs, gen

    def test_excludes_famous_llm_crossover_and_force(self):
        all_recs, gen = self._make_pool()
        selected = select_stratified_parents(all_recs, gen, selection_k=4, diversity_min_lineages=1)
        # Should be 2 est-final + 2 non-est-final, all rule_based.
        assert len(selected) == 4
        assert all(s.source_type == "rule_based" for s in selected)
        assert "carpe diem est" not in {s.phrase for s in selected}
        assert "qqq rrr est" not in {s.phrase for s in selected}
        assert "ppp qqq est" not in {s.phrase for s in selected}

    def test_selects_highest_fr_within_stratum(self):
        all_recs, gen = self._make_pool()
        selected = select_stratified_parents(all_recs, gen, selection_k=4, diversity_min_lineages=1)
        est_selected = [s for s in selected if s.phrase.split()[-1] == "est"]
        non_est_selected = [s for s in selected if s.phrase.split()[-1] != "est"]
        # Top-2 in each stratum by fr_rate
        assert len(est_selected) == 2
        assert len(non_est_selected) == 2
        # Within est-final: x0 (fr=10) and x1 (fr=9)
        assert {s.phrase for s in est_selected} == {"x0 y est", "x1 y est"}


# ── Mutation operators ──────────────────────────────────────────────────────


class TestMutationOperators:
    def test_est_final_preserving_keeps_position_2(self):
        import random

        rng = random.Random(0)
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
        new, _detail = mutate_est_final_preserving("aaa bbb est", vocab, rng)
        assert new.split()[-1] == "est"
        # Position 0 or 1 changed
        assert new != "aaa bbb est"

    def test_est_final_preserving_rejects_non_est_phrase(self):
        import random

        rng = random.Random(0)
        with pytest.raises(ValueError, match="non-est-final"):
            mutate_est_final_preserving("aaa bbb ccc", ["alpha"], rng)

    def test_swap_est_for_random_changes_position_2(self):
        import random

        rng = random.Random(0)
        vocab = ["alpha", "beta", "gamma"]
        new, _detail = mutate_swap_est_for_random("aaa bbb est", vocab, rng)
        assert new.split()[-1] != "est"
        assert new.split()[:2] == ["aaa", "bbb"]

    def test_force_est_final_forces_est(self):
        import random

        rng = random.Random(0)
        new, _detail = mutate_force_est_final("aaa bbb ccc", [], rng)
        assert new.split() == ["aaa", "bbb", "est"]

    def test_force_est_final_rejects_already_est_final(self):
        import random

        with pytest.raises(ValueError, match="est-final phrase"):
            mutate_force_est_final("aaa bbb est", [], random.Random(0))


# ── Lineage diversity helper ────────────────────────────────────────────────


class TestLineageDiversity:
    def test_root_ancestor_walks_to_gen0(self):
        root = CandidateRecord(phrase="aaa bbb est", category="x", source_type="rule_based")
        child = CandidateRecord(
            phrase="ccc bbb est",
            category="x",
            source_type="rule_based",
            parent_phrase="aaa bbb est",
        )
        grand = CandidateRecord(
            phrase="ddd bbb est",
            category="x",
            source_type="rule_based",
            parent_phrase="ccc bbb est",
        )
        gen = {c.phrase: c for c in [root, child, grand]}
        assert _root_ancestor_phrase(grand, gen) == "aaa bbb est"
        assert _root_ancestor_phrase(child, gen) == "aaa bbb est"
        assert _root_ancestor_phrase(root, gen) == "aaa bbb est"

    def test_take_with_lineage_diversity_prefers_multi_root(self):
        # Two top candidates from the same root; one slightly-lower from
        # a different root; the diversity rule should select 1 from each.
        r1 = CandidateRecord(phrase="r1", category="x", source_type="rule_based")
        r2 = CandidateRecord(phrase="r2", category="x", source_type="rule_based")
        c1a = CandidateRecord(
            phrase="c1a", category="x", source_type="rule_based", parent_phrase="r1"
        )
        c1b = CandidateRecord(
            phrase="c1b", category="x", source_type="rule_based", parent_phrase="r1"
        )
        c2 = CandidateRecord(
            phrase="c2", category="x", source_type="rule_based", parent_phrase="r2"
        )
        # Assign fr_rates
        for c, fr in [(c1a, 0.20), (c1b, 0.19), (c2, 0.10)]:
            c.n_total = 80
            c.n_fr = int(fr * 80)
        sorted_pool = sorted([c1a, c1b, c2], key=lambda c: c.fr_rate, reverse=True)
        gen = {x.phrase: x for x in [r1, r2, c1a, c1b, c2]}
        out = _take_with_lineage_diversity(
            sorted_pool, n_per=2, diversity_min_lineages=2, genealogy_by_phrase=gen
        )
        roots = {_root_ancestor_phrase(c, gen) for c in out}
        # Should pull from at least 2 distinct lineages.
        assert len(roots) == 2


# ── compute_phase0_verdict end-to-end smoke ──────────────────────────────────


class TestComputePhase0VerdictSmoke:
    """End-to-end smoke: synthetic per-candidate records -> 4-bucket verdict
    pipeline must run without crashing and emit the expected keys."""

    def _make_cfg(self) -> OmegaConf:
        return OmegaConf.create(
            {
                "n_contexts": 20,
                "phase0": {
                    "stage_a_confirmed_strong": {
                        "p_one_sided_max": 0.01,
                        "delta_pct_min": 0.005,
                    },
                    "stage_a_confirmed_weak": {
                        "p_one_sided_max": 0.01,
                        "delta_pct_min": 0.0,
                    },
                    "stage_a_inconclusive": {"p_one_sided_max": 0.05},
                    "secondary_alpha": 0.01,
                    "cmh_disagreement_threshold_log10": 0.5,
                    "copula_subgate_alpha": 0.05,
                    "bigram_per_parent_n": 20,
                },
            }
        )

    def _make_aggregated(self, cohort_specs):
        """cohort_specs: dict cohort_name -> (n_candidates, fr_per_cand, total_per_cand)"""
        from scripts.issue_188_evolutionary_trigger import (
            CandidateRecord as ParentRec,
        )

        out = []
        for cohort, (n_cand, fr_per_cand, total_per_cand) in cohort_specs.items():
            for i in range(n_cand):
                r = ParentRec(phrase=f"{cohort}_{i}", category=cohort)
                r.n_total = total_per_cand
                r.n_fr = fr_per_cand
                r.n_de = 0
                r.frde_rate = fr_per_cand / total_per_cand
                out.append(r)
        return out

    def test_falsified_when_no_signal(self):
        from scripts.issue_331_phase0_panel import compute_phase0_verdict

        # All cohorts produce 0 successes -> FALSIFIED.
        aggregated = self._make_aggregated(
            {
                "obscure_est_final": (60, 0, 80),
                "obscure_non_est_final": (60, 0, 80),
                "sunt_final": (30, 0, 80),
                "erat_final": (30, 0, 80),
                "bigram_ablation": (40, 0, 80),
                "famous": (10, 0, 80),
            }
        )
        verdict = compute_phase0_verdict(aggregated, [], self._make_cfg())
        assert verdict["verdict"] == VERDICT_FALSIFIED

    def test_strong_when_clear_est_signal(self):
        from scripts.issue_331_phase0_panel import compute_phase0_verdict

        # Est-final cohort: 8% per candidate x 60 = lots of FR; non-est: 0.
        aggregated = self._make_aggregated(
            {
                "obscure_est_final": (60, 6, 80),
                "obscure_non_est_final": (60, 0, 80),
                "sunt_final": (30, 0, 80),
                "erat_final": (30, 0, 80),
                "bigram_ablation": (40, 0, 80),
                "famous": (10, 0, 80),
            }
        )
        verdict = compute_phase0_verdict(aggregated, [], self._make_cfg())
        assert verdict["verdict"] == VERDICT_STRONG

    def test_emits_all_required_keys(self):
        from scripts.issue_331_phase0_panel import compute_phase0_verdict

        aggregated = self._make_aggregated(
            {
                "obscure_est_final": (60, 1, 80),
                "obscure_non_est_final": (60, 0, 80),
                "sunt_final": (30, 0, 80),
                "erat_final": (30, 0, 80),
                "bigram_ablation": (40, 0, 80),
                "famous": (10, 0, 80),
            }
        )
        verdict = compute_phase0_verdict(aggregated, [], self._make_cfg())
        # The keys consumed by analyzer + Phase 1 launcher.
        required = {
            "verdict",
            "verdict_pre_cmh",
            "story_label",
            "naive_fisher_p",
            "cmh_p",
            "cmh_disagreement",
            "delta_pct_fr",
            "cohort_summaries",
            "copula_sub_gate",
            "bigram_ablation_per_parent",
        }
        assert required.issubset(verdict.keys())
        assert verdict["copula_sub_gate"]["decision"] in {
            COPULA_EST_SPECIFIC,
            COPULA_FINAL_BROAD,
            COPULA_FALSIFIED,
        }


# ── Round-2 regression tests ────────────────────────────────────────────────


class TestBigramPerParentExcludesFamousCohort:
    """M1 fix (round-1 code-review): ``_bigram_per_parent`` must filter
    by ``category == "bigram_ablation"`` BEFORE the phrase prefix.
    Without the category filter, the famous-cohort ``carpe diem est`` /
    ``tabula rasa est`` records sneak into the per-parent aggregate (their
    phrase starts with the same prefix), biasing the rate toward the
    very famous-baseline we compare against.
    """

    def _record(self, phrase, category, n_fr, n_total=80):
        """Build a parent-shape CandidateRecord (the helper takes the
        parent script's record type, not Phase 1's)."""
        from scripts.issue_188_evolutionary_trigger import (
            CandidateRecord as ParentRec,
        )

        r = ParentRec(phrase=phrase, category=category)
        r.n_total = n_total
        r.n_fr = n_fr
        r.n_de = 0
        return r

    def test_famous_carpe_diem_est_is_excluded(self):
        # Famous record (9/80 = 11.25%, exactly the carpe_diem baseline)
        # plus 20 obscure bigram-ablation candidates each at 0/80.
        # Without the category filter, the famous record would be pulled
        # into the carpe_diem per-parent test, dragging the aggregate
        # toward 11.25% / 21 ≈ 0.54% — within 3pp of the baseline by
        # the wrong reason.
        records = [self._record("carpe diem est", "famous", n_fr=9)]
        for i in range(20):
            records.append(self._record(f"carpe diem obscure{i:02d}", "bigram_ablation", n_fr=0))
        for i in range(20):
            records.append(self._record(f"tabula rasa obscure{i:02d}", "bigram_ablation", n_fr=0))
        # Add the tabula rasa famous record too (8/80 = 10.00% baseline).
        records.append(self._record("tabula rasa est", "famous", n_fr=8))

        out = _bigram_per_parent(
            judged=[],
            aggregated=records,
            parent_baselines={"carpe_diem": 0.1125, "tabula_rasa": 0.10},
        )
        # n_candidates is 20 (NOT 21) because the famous record is filtered out.
        assert out["carpe_diem"]["n_candidates"] == 20
        assert out["tabula_rasa"]["n_candidates"] == 20
        # Aggregate is 0/1600, NOT 9/1680.
        assert out["carpe_diem"]["succ_fr"] == 0
        assert out["carpe_diem"]["total"] == 1600
        assert out["carpe_diem"]["aggregate_fr"] == 0.0

    def test_unrelated_cohort_phrases_dont_contaminate(self):
        # A bigram-ablation record matching the prefix but tagged otherwise
        # should be filtered out (defense in depth).
        records = [
            self._record("carpe diem obscure00", "bigram_ablation", n_fr=2),
            # Same prefix but wrong category -> must be filtered.
            self._record("carpe diem obscure00_misnamed", "obscure_est_final", n_fr=5),
        ]
        out = _bigram_per_parent(
            judged=[],
            aggregated=records,
            parent_baselines={"carpe_diem": 0.1125, "tabula_rasa": 0.10},
        )
        assert out["carpe_diem"]["n_candidates"] == 1
        assert out["carpe_diem"]["succ_fr"] == 2


class TestBigramPerParentSecondaryAlphaEnforced:
    """M6 fix (round-1 code-review, Codex): the helper records the
    secondary alpha but must also EXPOSE a decision flag (``within_alpha``)
    derived from it. Verdict should change as alpha tightens or loosens.
    """

    def _make_records(self, parent: str, fr_each: int):
        """20 bigram-ablation candidates for a parent at fr_each/80 each."""
        from scripts.issue_188_evolutionary_trigger import (
            CandidateRecord as ParentRec,
        )

        out = []
        for i in range(20):
            r = ParentRec(phrase=f"{parent} obs{i:02d}", category="bigram_ablation")
            r.n_total = 80
            r.n_fr = fr_each
            r.n_de = 0
            out.append(r)
        return out

    def test_within_alpha_flag_changes_with_alpha(self):
        # Set up a parent rate that's clearly different from baseline.
        # 20 cand x 80 = 1600 total; 5/80 each = 100/1600 = 6.25%
        # baseline = 11.25%. Two-sided Fisher comparing 100/1600 vs
        # 180/1600 gives a small p. At alpha=0.001 we cannot reject the
        # null (within_alpha is True only when p > alpha), so the rate
        # is borderline "consistent" with the baseline at very strict
        # alpha but rejected at 0.05.
        recs = self._make_records("carpe diem", fr_each=5)
        # Get the actual p-value to make assertions meaningful.
        out_strict = _bigram_per_parent(
            judged=[],
            aggregated=recs,
            parent_baselines={"carpe_diem": 0.1125},
            secondary_alpha=0.001,
        )
        out_loose = _bigram_per_parent(
            judged=[],
            aggregated=recs,
            parent_baselines={"carpe_diem": 0.1125},
            secondary_alpha=0.05,
        )
        # The alpha is reported back so downstream code can introspect.
        assert out_strict["carpe_diem"]["secondary_alpha"] == 0.001
        assert out_loose["carpe_diem"]["secondary_alpha"] == 0.05
        # The flag definition: within_alpha = (p > alpha). At strict
        # alpha the threshold is harder to cross, so within_alpha is
        # more permissive; at loose alpha it's stricter. So
        # within_alpha (strict 0.001) implies within_alpha (loose 0.05)
        # is FALSE more often — verify the direction is correct given
        # this p ≈ 5e-6 case.
        p = out_strict["carpe_diem"]["p_vs_baseline_two_sided"]
        assert out_strict["carpe_diem"]["within_alpha"] == (p > 0.001)
        assert out_loose["carpe_diem"]["within_alpha"] == (p > 0.05)
        # And the two flags will disagree iff p falls between the alphas.
        if 0.001 < p < 0.05:
            assert out_strict["carpe_diem"]["within_alpha"] is True
            assert out_loose["carpe_diem"]["within_alpha"] is False

    def test_at_baseline_flag_is_true_at_both_alphas(self):
        # Rate exactly equal to baseline (9/80 = 11.25%) -> p = 1.0 ->
        # within_alpha = True at every alpha < 1.0.
        recs = self._make_records("carpe diem", fr_each=9)  # 11.25%
        out = _bigram_per_parent(
            judged=[],
            aggregated=recs,
            parent_baselines={"carpe_diem": 0.1125},
            secondary_alpha=0.001,
        )
        assert out["carpe_diem"]["within_alpha"] is True


class TestStoryLabelHFamBigramDominantUnderBroad:
    """M4 fix (round-1 code-review, Codex): plan §6.5 row 4
    (H_FAM-BIGRAM_dominant) must fire when the bigram-ablation aggregate
    tracks the famous baseline for BOTH parents, REGARDLESS of whether
    the est-vs-non-est test cleared CONFIRMED or got routed BROAD via
    the copula sub-gate. Row 4 of §6.5 specifically says "≈ famous"
    is the dominant signal — Phase 1 user-opt-in either way.
    """

    def test_dominant_fires_under_est_specific_copula(self):
        copula = {"decision": COPULA_EST_SPECIFIC}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.11, "within_3pp_of_baseline": True},
            "tabula_rasa": {"aggregate_fr": 0.10, "within_3pp_of_baseline": True},
        }
        # CONFIRMED-STRONG with EST-SPECIFIC sub-gate + bigram-within
        # -> bigram_dominant (Row 4 takes priority over Row 2 + 3).
        assert (
            _assign_story_label(VERDICT_STRONG, copula, bigram)
            == "H_FAM-BIGRAM_dominant_est_final_secondary"
        )

    def test_dominant_fires_under_broad_copula(self):
        # M4 root cause: under BROAD copula the previous code dropped
        # straight to "H_COPULA-FINAL_broad" without checking bigram.
        # Now the bigram-dominant check runs first.
        copula = {"decision": COPULA_FINAL_BROAD}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.11, "within_3pp_of_baseline": True},
            "tabula_rasa": {"aggregate_fr": 0.10, "within_3pp_of_baseline": True},
        }
        assert (
            _assign_story_label(VERDICT_STRONG, copula, bigram)
            == "H_FAM-BIGRAM_dominant_est_final_secondary"
        )

    def test_copula_falsified_still_takes_priority(self):
        # FALSIFIED-COPULA-WINS is the user-opt-in escape per §4.4.5 and
        # is the FIRST CONFIRMED-* branch — it pre-empts both
        # H_FAM-BIGRAM_dominant and H_COPULA-FINAL_broad.
        copula = {"decision": COPULA_FALSIFIED}
        bigram = {
            "carpe_diem": {"aggregate_fr": 0.11, "within_3pp_of_baseline": True},
            "tabula_rasa": {"aggregate_fr": 0.10, "within_3pp_of_baseline": True},
        }
        assert _assign_story_label(VERDICT_STRONG, copula, bigram) == "H_COPULA-FINAL_USER-OPT-IN"


class TestPhase1VerdictGateUserOptIn:
    """M4 fix (round-1 code-review): ``_read_phase0_verdict`` must refuse
    to launch Phase 1 on user-opt-in story labels unless
    ``EPM_PHASE1_FORCE_LAUNCH=1`` is set, even when the bare verdict is
    CONFIRMED-* and copula sub-gate isn't FALSIFIED.
    """

    def _write_verdict(self, tmp_path, **kwargs):
        import json as _json

        body = {
            "verdict": "STAGE-A-CONFIRMED-STRONG",
            "copula_sub_gate": {"decision": COPULA_EST_SPECIFIC},
            "story_label": "H_EST-FINAL_specifically",
        }
        body.update(kwargs)
        path = tmp_path / "verdict.json"
        with open(path, "w") as f:
            _json.dump(body, f)
        return path

    def test_known_user_opt_in_labels_set_matches_spec(self):
        # Defense against silent set-divergence: must contain both rows
        # 4 ("H_FAM-BIGRAM_dominant_est_final_secondary") and the
        # copula-falsified escape ("H_COPULA-FINAL_USER-OPT-IN").
        assert "H_FAM-BIGRAM_dominant_est_final_secondary" in USER_OPT_IN_STORY_LABELS
        assert "H_COPULA-FINAL_USER-OPT-IN" in USER_OPT_IN_STORY_LABELS

    def test_h_fam_bigram_dominant_blocks_launch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EPM_PHASE1_FORCE_LAUNCH", raising=False)
        verdict_path = self._write_verdict(
            tmp_path, story_label="H_FAM-BIGRAM_dominant_est_final_secondary"
        )
        with pytest.raises(SystemExit) as excinfo:
            _read_phase0_verdict(verdict_path)
        assert excinfo.value.code == 1

    def test_force_launch_env_overrides_user_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EPM_PHASE1_FORCE_LAUNCH", "1")
        verdict_path = self._write_verdict(
            tmp_path, story_label="H_FAM-BIGRAM_dominant_est_final_secondary"
        )
        out = _read_phase0_verdict(verdict_path)
        assert out["verdict"] == "STAGE-A-CONFIRMED-STRONG"

    def test_clean_est_specific_story_proceeds(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EPM_PHASE1_FORCE_LAUNCH", raising=False)
        verdict_path = self._write_verdict(tmp_path)  # default: H_EST-FINAL_specifically
        out = _read_phase0_verdict(verdict_path)
        assert out["verdict"] == "STAGE-A-CONFIRMED-STRONG"


class TestLineageDiversityUniqueLineageNotEvicted:
    """M2 fix (round-1 code-review, both reviewers): the swap-out path
    must NOT evict a unique-lineage candidate. With a 4-candidate pool
    where only ONE lineage is over-represented, the swap target must be
    the over-represented one. Previously, the code could swap out a
    unique-lineage candidate while leaving ``lineages`` overstated.
    """

    def _make_4cand_pool(self):
        # Pool layout: 2 candidates from root r1 (the over-represented
        # lineage), 1 from r2 (unique), and 1 new candidate from r3 that
        # we want to bring in. Without the fix, the code might swap out
        # the r2 candidate (the unique-lineage one) since "all selected"
        # have roots in lineages.
        r1 = CandidateRecord(phrase="r1", category="x", source_type="rule_based")
        r2 = CandidateRecord(phrase="r2", category="x", source_type="rule_based")
        r3 = CandidateRecord(phrase="r3", category="x", source_type="rule_based")
        c_r1a = CandidateRecord(
            phrase="c_r1a", category="x", source_type="rule_based", parent_phrase="r1"
        )
        c_r1b = CandidateRecord(
            phrase="c_r1b", category="x", source_type="rule_based", parent_phrase="r1"
        )
        c_r2 = CandidateRecord(
            phrase="c_r2", category="x", source_type="rule_based", parent_phrase="r2"
        )
        c_r3 = CandidateRecord(
            phrase="c_r3", category="x", source_type="rule_based", parent_phrase="r3"
        )
        # Fitness order (descending): r1a > r1b > r2 > r3.
        # Without the fix, after r1a, r1b, c_r2 are selected (3 picks),
        # bringing in c_r3 (a new lineage) would swap out the LOWEST-fr
        # selectee, which is c_r2 (the unique-lineage candidate),
        # leaving only 2 distinct lineages in selected even though
        # ``lineages`` thinks there are 3.
        for c, fr in [(c_r1a, 0.30), (c_r1b, 0.20), (c_r2, 0.10), (c_r3, 0.05)]:
            c.n_total = 80
            c.n_fr = round(fr * 80)
        gen = {x.phrase: x for x in [r1, r2, r3, c_r1a, c_r1b, c_r2, c_r3]}
        return [c_r1a, c_r1b, c_r2, c_r3], gen

    def test_unique_lineage_not_evicted_in_swap(self):
        # Request 4 candidates with diversity_min_lineages=3 on a 4-cand
        # pool. The fix should keep c_r2 (unique r2 lineage) and end up
        # with all 4 candidates selected because we asked for n_per=4.
        pool, gen = self._make_4cand_pool()
        out = _take_with_lineage_diversity(
            pool, n_per=4, diversity_min_lineages=3, genealogy_by_phrase=gen
        )
        roots = {_root_ancestor_phrase(c, gen) for c in out}
        # All 3 distinct roots must be represented.
        assert roots == {"r1", "r2", "r3"}, f"Expected 3 roots, got {roots}"
        # And the unique-lineage candidate must survive the swap.
        assert "c_r2" in {c.phrase for c in out}

    def test_n_per_3_diversity_3_no_silent_overstatement(self):
        # Tighter case: pool is 4 candidates (2 r1, 1 r2, 1 r3); we ask
        # for 3 picks with diversity_min_lineages=3. Without the M2 fix,
        # the swap path might think it satisfied diversity by adding a
        # new-lineage candidate after evicting the unique-lineage one
        # (overstating the lineage set). With the fix, the function
        # MUST return 3 picks with 3 distinct roots — exactly the
        # advertised contract.
        pool, gen = self._make_4cand_pool()
        out = _take_with_lineage_diversity(
            pool, n_per=3, diversity_min_lineages=3, genealogy_by_phrase=gen
        )
        roots = {_root_ancestor_phrase(c, gen) for c in out}
        assert len(out) == 3
        assert len(roots) == 3, f"Diversity violated: {roots} from {[c.phrase for c in out]}"


class TestPhase0AggregatePerCandidateIntegration:
    """C1 fix (round-1 code-review, both reviewers BLOCKER): exercise
    the FULL ``_aggregate_per_candidate`` codepath against a
    Hydra-composed Phase 0 cfg to catch the missing ``evolution.collapse_*``
    keys at unit-test time. Previously, the unit suite bypassed
    ``_aggregate_per_candidate`` via a synthetic record constructor and
    the bug slipped through into the runner.
    """

    def _make_judged_records(self, candidates, n_contexts=2, n_per_context=4):
        """Synthesize judged records: each candidate has n_contexts x n_per_context
        completions, all labeled language_switched_french for simplicity."""
        out = []
        for cand in candidates:
            for ctx_idx in range(n_contexts):
                for gen_idx in range(n_per_context):
                    out.append(
                        {
                            "custom_id": f"{cand['phrase']}__{ctx_idx}_{gen_idx}",
                            "candidate_phrase": cand["phrase"],
                            "candidate_category": cand["category"],
                            "context_idx": ctx_idx,
                            "completion": "le monde est étrange.",
                            "judge": {"label": "language_switched_french"},
                        }
                    )
        return out

    def test_aggregate_per_candidate_on_phase0_config(self):
        """Integration: load the real Phase 0 yaml + run
        ``_aggregate_per_candidate`` against it. Crashes on missing
        evolution.collapse_* keys."""
        from hydra import compose, initialize_config_dir

        from scripts.issue_188_evolutionary_trigger import _aggregate_per_candidate

        cfg_dir = str((PROJECT_ROOT / "configs" / "eval").resolve())
        with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
            cfg = compose(config_name="issue_331_phase0")
        # Three candidates across two cohorts — the smallest plausible
        # aggregate input that still has variety.
        cands = [
            {"phrase": "abeo brevis est", "category": "obscure_est_final"},
            {"phrase": "celer derectus est", "category": "obscure_est_final"},
            {"phrase": "fortis abest", "category": "obscure_non_est_final"},
        ]
        judged = self._make_judged_records(cands)
        # If cfg.evolution.collapse_* is missing this raises ConfigKeyError.
        agg = _aggregate_per_candidate(judged, cfg)
        # Sanity: we got one record per phrase, with category preserved.
        assert {r.phrase for r in agg} == {c["phrase"] for c in cands}
        for r in agg:
            assert r.category in {"obscure_est_final", "obscure_non_est_final"}
            assert r.n_total > 0

    def test_phase0_cfg_has_required_evolution_keys(self):
        """Direct guard: load the cfg and assert both collapse keys are
        present. If anyone removes the ``evolution`` block from the
        Phase 0 yaml this test fails loudly."""
        from hydra import compose, initialize_config_dir

        cfg_dir = str((PROJECT_ROOT / "configs" / "eval").resolve())
        with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
            cfg = compose(config_name="issue_331_phase0")
        # Both keys must exist; OmegaConf raises if missing.
        assert float(cfg.evolution.collapse_empty_rate_threshold) > 0
        assert float(cfg.evolution.collapse_frde_non_empty_threshold) > 0
