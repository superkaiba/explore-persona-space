"""Phase 1 headline-outcome resolution tests (issue #331, round-3 fix).

Round-3 BLOCKER (reconciler binding verdict against Codex round-2 FAIL):
``exit_reason`` was conflating "why the loop stopped" with "what outcome
the run reached", so a run that hit WEAK-CLIMB or STRONG-CLIMB threshold
but exhausted the round budget would write ``verdict: budget_exhausted``
in summary.json — making the modal expected outcome (plan §725) invisible
to the analyzer.

Fix: split into ``halt_reason`` (mechanical loop-termination) and
``headline_outcome`` (analyzer-facing canonical outcome), resolved by a
precedence rule after the loop exits.

These tests pin the precedence rule. They also include an end-to-end
unit test for ``_replicate_success_candidate_n400`` with a mocked vLLM
(N7 from Claude round-2 review), exercising the config branching, the
context-loader extension, the per-seed FR aggregation, and the
``success_replicated`` gate at FR >= 30%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.issue_331_phase1_evolutionary import (  # noqa: E402
    CandidateRecord,
    _replicate_success_candidate_n400,
    _resolve_headline_outcome,
)

# ── BLOCKER fix: headline-outcome precedence ────────────────────────────────


class TestResolveHeadlineOutcome:
    """Pin the precedence rule for ``_resolve_headline_outcome``.

    Precedence (highest → lowest):
      1. SUCCESS         (FR >= 50% rule_based obscure-only)
      2. STRONG-CLIMB    (FR+DE >= 11.25% rule_based obscure-only)
      3. WEAK-CLIMB      (FR >= 6.25% rule_based obscure-only)
      4. KILL            (best < 6.25% AND est <= non-est)
      5. INCONCLUSIVE    (partial signal, plateau, budget exhausted)

    Control-flow halt ``no_mutants`` (no fitness signal at all) flows
    through as its own outcome.
    """

    def test_success_halt_returns_success(self):
        """Hard-gate SUCCESS halt always resolves to SUCCESS."""
        outcome = {
            "hit_success": True,
            "hit_strong_climb": True,  # subsumed
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("SUCCESS", outcome) == "SUCCESS"

    def test_strong_climb_at_budget_exhaustion(self):
        """The canonical bug scenario: budget exhausted while STRONG-CLIMB
        threshold is met. MUST resolve to STRONG-CLIMB, not budget_exhausted.

        This is the test that would have failed in rounds 1+2.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": True,
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("budget_exhausted", outcome) == "STRONG-CLIMB"

    def test_weak_climb_at_budget_exhaustion(self):
        """Plan §725 calls this the modal expected outcome.

        Before the round-3 fix, summary.json would say ``verdict:
        budget_exhausted`` here and the analyzer would miss the headline.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": True,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("budget_exhausted", outcome) == "WEAK-CLIMB"

    def test_weak_climb_at_plateau(self):
        """Plateau halt with WEAK-CLIMB signal also resolves to WEAK-CLIMB.

        The plateau halt_reason is mechanical and tells the analyzer the
        loop stopped early because of stagnation — but the headline
        outcome the run reached is still WEAK-CLIMB.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": True,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("plateau", outcome) == "WEAK-CLIMB"

    def test_kill_halt_returns_kill(self):
        """Hard-gate KILL halt resolves to KILL."""
        outcome = {
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": False,
            "hit_kill": True,
        }
        assert _resolve_headline_outcome("KILL", outcome) == "KILL"

    def test_inconclusive_at_budget_exhaustion(self):
        """Budget exhausted with no climb signal and no kill → INCONCLUSIVE.

        This is the previous default behavior (``verdict:
        budget_exhausted``); after the round-3 fix it now distinguishes
        "we ran out of time and saw partial signal" (INCONCLUSIVE) from
        the climb / kill / success states.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("budget_exhausted", outcome) == "INCONCLUSIVE"

    def test_inconclusive_at_plateau_no_signal(self):
        """Plateau halt with no climb signal → INCONCLUSIVE."""
        outcome = {
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("plateau", outcome) == "INCONCLUSIVE"

    def test_no_mutants_pass_through(self):
        """``no_mutants`` is a control-flow halt with no fitness signal."""
        outcome: dict = {}
        assert _resolve_headline_outcome("no_mutants", outcome) == "no_mutants"

    def test_no_mutants_pass_through_even_with_signal(self):
        """``no_mutants`` even if a prior round set climb flags.

        If the mutator fails on round 1, final_outcome is ``{}``; if it
        fails later, final_outcome may carry climb signal from the
        previous round. We choose to surface the control-flow halt as
        the headline regardless — the analyzer should see that mutation
        stalled, since that is the load-bearing operational fact.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": True,
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("no_mutants", outcome) == "no_mutants"

    def test_precedence_strong_over_weak(self):
        """STRONG-CLIMB beats WEAK-CLIMB when both flags are true.

        In the production code these are mutually exclusive (``weak_climb
        = not strong_climb and ...``) but the resolver is defensive
        against future drift in ``_evaluate_phase1_outcome``.
        """
        outcome = {
            "hit_success": False,
            "hit_strong_climb": True,
            "hit_weak_climb": True,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("budget_exhausted", outcome) == "STRONG-CLIMB"

    def test_precedence_success_over_strong_climb(self):
        """SUCCESS beats STRONG-CLIMB even on a non-SUCCESS halt.

        Defensive: if final_outcome.hit_success somehow becomes True
        without the SUCCESS halt firing (e.g. on a plateau halt that
        coincides with a SUCCESS-threshold candidate appearing in the
        same round), surface SUCCESS as the headline.
        """
        outcome = {
            "hit_success": True,
            "hit_strong_climb": True,
            "hit_weak_climb": False,
            "hit_kill": False,
        }
        assert _resolve_headline_outcome("plateau", outcome) == "SUCCESS"

    def test_empty_final_outcome_with_budget_exhausted(self):
        """Budget exhausted with empty outcome dict → INCONCLUSIVE.

        This happens if no round produced any fitness signal (e.g. all
        rounds hit ``no_mutants`` halts within the inner loop and we
        somehow exited via budget — defensive case).
        """
        assert _resolve_headline_outcome("budget_exhausted", {}) == "INCONCLUSIVE"


# ── N7: _replicate_success_candidate_n400 unit test (Claude round-2) ─────────


class TestReplicateSuccessCandidateN400:
    """End-to-end exercise of the SUCCESS n=400 replication with mocked vLLM.

    Claude round-2 Minor N7: ``_replicate_success_candidate_n400`` had no
    unit test because it depends on vLLM. This test mocks the heavy
    bits (``_load_or_fetch_contexts``, ``_generate_completions``,
    ``_judge_records``, ``_aggregate_per_candidate``) and asserts the
    function:

      1. Calls the context loader with n=100 (not cfg.n_contexts=20).
      2. Branches the cfg.vllm.seed correctly between 42 and 137.
      3. Aggregates per-seed FR rates from the (mocked) judged records.
      4. Sets ``success_replicated`` iff seed137 FR >= success_replicated_fr_min.
      5. Returns the documented dict shape (phrase / original_fr_rate /
         n_total_per_seed / per_seed{seed42,seed137} / success_replicated /
         success_replicated_fr_min).
    """

    @pytest.fixture
    def cfg(self):
        return OmegaConf.create(
            {
                "contexts_path": "ignored_by_mock",
                "n_contexts": 20,
                "n_generations_per_pair": 4,
                "vllm": {"seed": 0, "gpu_memory_utilization": 0.9, "max_model_len": 4096},
                "evolution": {
                    "success_replicated_fr_min": 0.30,
                },
            }
        )

    @pytest.fixture
    def winner(self):
        return CandidateRecord(
            phrase="carpe sunt est",
            category="obscure_est_final",
            source_type="rule_based",
            parent_phrase=None,
            mutation_operator="phase0_seed",
            mutation_detail=None,
            round_idx=0,
            n_total=80,
            n_fr=42,
            n_de=4,
        )

    def _make_agg(self, n_fr: int, n_total: int):
        """Build a mock aggregate-record from ``_aggregate_per_candidate``."""

        class _FakeAgg:
            pass

        agg = _FakeAgg()
        agg.n_fr = n_fr
        agg.n_total = n_total
        agg.frde_rate = (n_fr + 0) / n_total if n_total else 0.0
        return agg

    def test_success_replicated_when_seed137_fr_above_threshold(self, cfg, winner, tmp_path):
        """Happy path: seed137 FR=40% clears the 30% gate."""
        # Mock helpers. _load_or_fetch_contexts must return 100 strings.
        ctxs = [f"ctx-{i}" for i in range(100)]
        # _generate_completions returns (records, llm); we don't inspect
        # records here because _judge_records is also mocked.
        judged = [{"label": "fr"}]  # opaque to this test

        seq = iter(
            [
                # seed42 first
                self._make_agg(n_fr=120, n_total=400),  # 30% FR
                # then seed137
                self._make_agg(n_fr=160, n_total=400),  # 40% FR
            ]
        )

        with (
            patch(
                "scripts.issue_331_phase1_evolutionary._load_or_fetch_contexts",
                return_value=ctxs,
            ) as mock_load,
            patch(
                "scripts.issue_331_phase1_evolutionary._generate_completions",
                return_value=(judged, "fake_llm"),
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._judge_records",
                return_value=judged,
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._aggregate_per_candidate",
                side_effect=lambda *a, **kw: [next(seq)],
            ),
        ):
            out = _replicate_success_candidate_n400(winner, cfg, tmp_path, llm="fake_llm")

        # Context loader called with n=100 (not cfg.n_contexts=20).
        assert mock_load.call_args.kwargs.get("n") == 100 or 100 in mock_load.call_args.args

        # Documented dict shape.
        assert out["phrase"] == "carpe sunt est"
        assert out["original_fr_rate"] == pytest.approx(winner.fr_rate)
        assert out["n_total_per_seed"] == 100 * 4  # cfg.n_generations_per_pair
        assert set(out["per_seed"].keys()) == {"seed42", "seed137"}

        # Per-seed FR rates.
        assert out["per_seed"]["seed42"]["vllm_seed"] == 42
        assert out["per_seed"]["seed42"]["fr_rate"] == pytest.approx(0.30)
        assert out["per_seed"]["seed42"]["n_fr"] == 120
        assert out["per_seed"]["seed42"]["n_total"] == 400

        assert out["per_seed"]["seed137"]["vllm_seed"] == 137
        assert out["per_seed"]["seed137"]["fr_rate"] == pytest.approx(0.40)
        assert out["per_seed"]["seed137"]["n_fr"] == 160
        assert out["per_seed"]["seed137"]["n_total"] == 400

        # Gate: seed137 FR (0.40) >= success_replicated_fr_min (0.30) → True.
        assert out["success_replicated"] is True
        assert out["success_replicated_fr_min"] == pytest.approx(0.30)

    def test_success_not_replicated_when_seed137_fr_below_threshold(self, cfg, winner, tmp_path):
        """Fail path: seed137 FR=20% does NOT clear the 30% gate.

        Confirms the gate is a strict inequality (FR >= success_min) and
        the per-seed aggregation is independent (seed42 high, seed137 low).
        """
        ctxs = [f"ctx-{i}" for i in range(100)]
        judged = [{"label": "fr"}]
        seq = iter(
            [
                self._make_agg(n_fr=200, n_total=400),  # seed42 50%
                self._make_agg(n_fr=80, n_total=400),  # seed137 20% < 30%
            ]
        )

        with (
            patch(
                "scripts.issue_331_phase1_evolutionary._load_or_fetch_contexts",
                return_value=ctxs,
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._generate_completions",
                return_value=(judged, "fake_llm"),
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._judge_records",
                return_value=judged,
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._aggregate_per_candidate",
                side_effect=lambda *a, **kw: [next(seq)],
            ),
        ):
            out = _replicate_success_candidate_n400(winner, cfg, tmp_path, llm="fake_llm")

        assert out["per_seed"]["seed42"]["fr_rate"] == pytest.approx(0.50)
        assert out["per_seed"]["seed137"]["fr_rate"] == pytest.approx(0.20)
        # Gate: 0.20 < 0.30 → not replicated. The gate keys off seed137,
        # not seed42 — the latter is recorded for transparency only.
        assert out["success_replicated"] is False

    def test_success_replicated_boundary_at_exactly_30_percent(self, cfg, winner, tmp_path):
        """Boundary: seed137 FR = exactly 30% counts as replicated (>=)."""
        ctxs = [f"ctx-{i}" for i in range(100)]
        judged = [{"label": "fr"}]
        seq = iter(
            [
                self._make_agg(n_fr=200, n_total=400),
                self._make_agg(n_fr=120, n_total=400),  # exactly 0.30
            ]
        )

        with (
            patch(
                "scripts.issue_331_phase1_evolutionary._load_or_fetch_contexts",
                return_value=ctxs,
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._generate_completions",
                return_value=(judged, "fake_llm"),
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._judge_records",
                return_value=judged,
            ),
            patch(
                "scripts.issue_331_phase1_evolutionary._aggregate_per_candidate",
                side_effect=lambda *a, **kw: [next(seq)],
            ),
        ):
            out = _replicate_success_candidate_n400(winner, cfg, tmp_path, llm="fake_llm")

        assert out["per_seed"]["seed137"]["fr_rate"] == pytest.approx(0.30)
        assert out["success_replicated"] is True


# ── Smoke test: summary JSON shape after _finalize (BLOCKER end-to-end) ─────


class TestFinalizeWritesBothFields:
    """End-to-end: ``_finalize`` writes BOTH ``halt_reason`` AND
    ``headline_outcome`` to summary.json, and ``verdict`` aliases the
    latter (not the former).

    This is the contract the analyzer reads. If this test fails, the
    round-3 bug has regressed.
    """

    def _make_cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "seed": 42,
                "output_dir": str(tmp_path),
                # metadata.get_run_metadata reads these:
                "experiment": "issue_331_phase1_test",
            }
        )

    def test_weak_climb_at_budget_writes_weak_climb_to_verdict(self, tmp_path):
        """The canonical bug fix, end-to-end through _finalize.

        Before round 3: summary["verdict"] == "budget_exhausted".
        After round 3:  summary["verdict"] == "WEAK-CLIMB".
        """
        from scripts.issue_331_phase1_evolutionary import _finalize

        cfg = self._make_cfg(tmp_path)
        final_outcome = {
            "global_max_fr_obscure_only": 0.075,
            "global_max_frde_obscure_only": 0.08,
            "global_max_fr_inclusive": 0.075,
            "obscure_only_best_phrase": "carpe sunt est",
            "obscure_only_best_fr_rate": 0.075,
            "hit_success": False,
            "hit_strong_climb": False,
            "hit_weak_climb": True,
            "hit_kill": False,
        }
        headline = _resolve_headline_outcome("budget_exhausted", final_outcome)
        assert headline == "WEAK-CLIMB"  # sanity

        # Patch out the helpers _finalize calls that touch disk for the
        # genealogy/global-ranking outputs (we only inspect summary.json).
        with (
            patch("scripts.issue_331_phase1_evolutionary._save_genealogy"),
            patch("scripts.issue_331_phase1_evolutionary._save_global_ranking"),
            patch(
                "explore_persona_space.metadata.get_run_metadata",
                return_value={"git_commit": "test"},
            ),
        ):
            _finalize(
                all_candidates=[],
                genealogy_by_phrase={},
                output_dir=tmp_path,
                cfg=cfg,
                wandb_run=None,
                halt_reason="budget_exhausted",
                headline_outcome=headline,
                final_outcome=final_outcome,
                replication=None,
                success_n400=None,
            )

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["halt_reason"] == "budget_exhausted"
        assert summary["headline_outcome"] == "WEAK-CLIMB"
        # CRITICAL: ``verdict`` is the canonical analyzer-facing key.
        # After round 3, it aliases headline_outcome, NOT halt_reason.
        assert summary["verdict"] == "WEAK-CLIMB"
        # Back-compat: ``exit_reason`` still populated (with halt_reason)
        # so any (currently no-known) reader of the old key keeps working.
        assert summary["exit_reason"] == "budget_exhausted"
