"""Regression tests for kind-aware type selection + issue-branch scan (#563).

``scripts/verify_uploads.py`` used to default-type every task as
``training`` (demanding WandB-run + HF-model rows an eval-only task cannot
satisfy) and scanned only main-working-tree paths for eval JSONs / figures
(missing artifacts committed on the ``issue-<N>`` branch pre-merge). These
tests pin the frontmatter-``kind`` inference, the conservative fallback,
the per-type checklist-row selection, and the issue-path filter used by the
branch scan. Same module-loading conventions as
tests/test_verify_uploads_claimed_urls.py.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_uploads.py"
_spec = importlib.util.spec_from_file_location("verify_uploads_ts", _SCRIPT)
verify_uploads = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_uploads_ts"] = verify_uploads
_spec.loader.exec_module(verify_uploads)  # type: ignore[union-attr]


def _mock_get_task(kind: str):
    return patch(
        "explore_persona_space.task_workflow.get_task",
        return_value={"frontmatter": {"kind": kind}},
    )


# ── infer_experiment_type ─────────────────────────────────────────────────────


class TestInferExperimentType:
    def test_analysis_kind_maps_to_analysis(self):
        with _mock_get_task("analysis"):
            assert verify_uploads.infer_experiment_type(1) == ("analysis", "frontmatter-kind")

    def test_non_experiment_kinds_skip_training_rows(self):
        for kind in ("infra", "batch", "survey"):
            with _mock_get_task(kind):
                etype, source = verify_uploads.infer_experiment_type(1)
            assert etype == "analysis", kind
            assert source == "frontmatter-kind", kind

    def test_experiment_kind_stays_training(self):
        """kind=experiment cannot distinguish training from eval-only, so the
        inference stays conservative — callers pass --type eval-only."""
        with _mock_get_task("experiment"):
            assert verify_uploads.infer_experiment_type(1) == ("training", "frontmatter-kind")

    def test_unknown_kind_falls_back_to_training(self):
        with _mock_get_task("not-a-kind"):
            assert verify_uploads.infer_experiment_type(1) == ("training", "default")

    def test_unreadable_task_falls_back_to_training(self):
        """A broken inference must over-demand, never relax the gate."""
        with patch(
            "explore_persona_space.task_workflow.get_task",
            side_effect=KeyError("task not found"),
        ):
            assert verify_uploads.infer_experiment_type(999999) == ("training", "default")


# ── run_verification checklist-row selection ──────────────────────────────────


class TestRunVerificationRowSelection:
    def test_eval_only_skips_training_rows(self):
        """The #563 false-FAIL repro: an eval-only task no longer gets the
        always-MISSING wandb_run / hf_model rows."""
        report = verify_uploads.run_verification(563, experiment_type="eval-only")
        assert "wandb_run" not in report["checks"]
        assert "hf_model" not in report["checks"]
        assert report["experiment_type_source"] == "cli"

    def test_training_still_demands_model_and_wandb(self):
        """Type selection must not weaken the gate for real training runs."""
        report = verify_uploads.run_verification(563, experiment_type="training")
        assert report["checks"]["wandb_run"]["status"] == "MISSING"
        assert report["checks"]["hf_model"]["status"] == "MISSING"
        assert report["verdict"] == "FAIL"

    def test_omitted_type_uses_frontmatter_inference(self):
        with _mock_get_task("analysis"):
            report = verify_uploads.run_verification(563, experiment_type=None)
        assert report["experiment_type"] == "analysis"
        assert report["experiment_type_source"] == "frontmatter-kind"
        assert "wandb_run" not in report["checks"]
        assert "hf_model" not in report["checks"]


# ── issue-branch path filtering ───────────────────────────────────────────────


class TestFilterIssuePaths:
    def test_matches_top_level_entry_containing_issue(self):
        paths = [
            "eval_results/issue_563/base_prior.json",
            "eval_results/issue_563/nested/deep.json",
            "eval_results/issue_558/other.json",
            "eval_results/misc/issue_563_nested.json",  # deep-only match: no
            "eval_results",  # no second component
        ]
        assert verify_uploads.filter_issue_paths(paths, 563) == [
            "eval_results/issue_563/base_prior.json",
            "eval_results/issue_563/nested/deep.json",
        ]

    def test_figures_prefix(self):
        paths = ["figures/issue_563/panel.png", "figures/issue_42/panel.png"]
        assert verify_uploads.filter_issue_paths(paths, 563) == ["figures/issue_563/panel.png"]


# ── delimiter-bounded issue matching (#563 follow-up) ─────────────────────────


class TestIssueTokenMatch:
    def test_low_issue_does_not_match_longer_numbers(self):
        """The false-PASS vector: issue 56 must not claim issue_563 artifacts."""
        assert not verify_uploads.issue_token_match("issue_563", 56)
        assert not verify_uploads.issue_token_match("issue_456", 56)
        assert not verify_uploads.issue_token_match("2056_panel", 56)

    def test_digit_bounded_occurrences_match(self):
        assert verify_uploads.issue_token_match("issue_56", 56)
        assert verify_uploads.issue_token_match("issue_56_followup", 56)
        assert verify_uploads.issue_token_match("56_panel.json", 56)
        assert verify_uploads.issue_token_match("issue-56", 56)


class TestFilterIssuePathsBounded:
    def test_issue_56_does_not_match_issue_563(self):
        paths = [
            "eval_results/issue_563/base_prior.json",
            "eval_results/issue_56/real.json",
        ]
        assert verify_uploads.filter_issue_paths(paths, 56) == ["eval_results/issue_56/real.json"]


class TestWorkingTreeIssueEntries:
    def test_glob_does_not_substring_match(self, tmp_path):
        (tmp_path / "eval_results" / "issue_563").mkdir(parents=True)
        (tmp_path / "eval_results" / "issue_56").mkdir()
        entries = verify_uploads._working_tree_issue_entries(tmp_path, "eval_results", 56)
        assert {e.name for e in entries} == {"issue_56"}

    def test_glob_keeps_real_matches_for_high_issue(self, tmp_path):
        (tmp_path / "figures" / "issue_563").mkdir(parents=True)
        (tmp_path / "figures" / "issue_56").mkdir()
        entries = verify_uploads._working_tree_issue_entries(tmp_path, "figures", 563)
        assert {e.name for e in entries} == {"issue_563"}
