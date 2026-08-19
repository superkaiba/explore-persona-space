"""Pins the matched-covariate support audit (#2180, incident #2163).

Three layers:

1. Unit tests for ``analysis/matched_support.py`` (tied_fraction /
   tie_profile / support_mask / audit_matched_artifact).
2. Regression tests against the committed #2163 artifacts
   (``eval_results/issue_2163/`` — sparse cone registered in
   ``tests/sparse_cones.txt``): the known-positive case (fires=True on
   ``predictor_partials.json``), the negative control (fires=False on
   ``population_partials.json`` via the per-population companion arm),
   and the raw-covariate path on ``census.npz``.
3. Prose pins: the five workflow surfaces carrying Statistics &
   Measurement lens item 18 keep naming it (the
   test_mapping_baselines_wiring_pins.py convention).

The module is loaded by FILE PATH (importlib) so the worktree's copy is
tested even under the editable-install main-src resolution trap.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "eval_results" / "issue_2163"

# #2163 pinned values (see plan / task body): the recorded complete-case
# tie fraction and the raw full-width census modal share differ by ~0.002
# because complete-case filtering (all selection covariates + logU_W
# finite) drops 131,072 -> 128,450 rows — the tied_fraction caller
# contract (pass the complete-case column) exists exactly for this gap.
RECORDED_FULL_TIE_FRACTION = 0.8965978980147917
CENSUS_RAW_TIE_FRACTION = 0.8986129760742188


def _load_module():
    p = REPO / "src/explore_persona_space/analysis/matched_support.py"
    spec = importlib.util.spec_from_file_location("matched_support_2180", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ms = _load_module()


# ---------------------------------------------------------------- unit: tied_fraction


def test_tied_fraction_zero_inflated():
    x = np.array([0.0] * 9 + [1.0])
    assert ms.tied_fraction(x) == pytest.approx(0.9)


def test_tied_fraction_all_distinct_is_one_over_n():
    x = np.arange(10, dtype=np.float64)
    assert ms.tied_fraction(x) == pytest.approx(0.1)


def test_tied_fraction_is_value_agnostic_nonzero_tie():
    # Degenerate by ties at 7.0 with NO zeros — zero-inflation is the
    # common case, not the definition.
    x = np.array([7.0] * 6 + [1.0, 2.0, 3.0, 4.0])
    assert ms.tied_fraction(x) == pytest.approx(0.6)


def test_tied_fraction_excludes_nan_from_count_and_denominator():
    x = np.array([0.0, 0.0, 1.0, np.nan])
    assert ms.tied_fraction(x) == pytest.approx(2.0 / 3.0)


def test_tied_fraction_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        ms.tied_fraction(np.array([]))


def test_tied_fraction_all_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        ms.tied_fraction(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------- unit: tie_profile


def test_tie_profile_orders_by_share_desc():
    x = np.array([0.0] * 5 + [2.0] * 3 + [5.0] * 2)
    prof = ms.tie_profile(x, k=2)
    assert prof[0] == (0.0, pytest.approx(0.5))
    assert prof[1] == (2.0, pytest.approx(0.3))


def test_tie_profile_k_must_be_positive():
    with pytest.raises(ValueError, match="k"):
        ms.tie_profile(np.array([1.0, 2.0]), k=0)


# ---------------------------------------------------------------- unit: support_mask


def test_support_mask_is_complement_of_modal_block():
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    mask = ms.support_mask(x)
    assert mask.tolist() == [False, False, False, True, True]
    assert mask.sum() / len(x) == pytest.approx(1.0 - ms.tied_fraction(x))


def test_support_mask_nan_rows_are_false():
    x = np.array([0.0, 0.0, 3.0, np.nan])
    assert ms.support_mask(x).tolist() == [False, False, True, False]


# ---------------------------------------------------------------- unit: audit


def test_audit_raises_without_any_tie_fraction_source():
    with pytest.raises(ValueError, match="no tie-fraction source"):
        ms.audit_matched_artifact({"per_dv": {"a": {"partial": 0.2}}})


def test_audit_explicit_arg_non_degenerate_does_not_fire():
    out = ms.audit_matched_artifact({}, tie_fraction=0.3)
    assert out["degenerate"] is False
    assert out["fires"] is False
    assert out["tie_fraction_source"] == "explicit-arg"


def test_audit_explicit_arg_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ms.audit_matched_artifact({}, tie_fraction=1.5)


def test_audit_covariate_values_source():
    cov = np.array([0.0] * 8 + [1.0, 2.0])
    out = ms.audit_matched_artifact({}, covariate=cov)
    assert out["tie_fraction_source"] == "covariate-values"
    assert out["tie_fraction"] == pytest.approx(0.8)
    assert out["degenerate"] is True and out["fires"] is True


def test_audit_companion_arm_a_on_support_key_at_depth():
    art = {"results": {"stats": {"partial_on_support": 0.01}}}
    out = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out["has_support_companion"] is True
    assert out["fires"] is False


def test_audit_degenerate_limit_still_reports_degenerate():
    # tied fraction ~ 1.0: support (near-)empty — the audit still reports
    # degenerate/fires; the lens text carries the drop-the-covariate remedy.
    out = ms.audit_matched_artifact({}, tie_fraction=1.0)
    assert out["degenerate"] is True and out["fires"] is True


def test_audit_exact_boundary_half_does_not_fire():
    # r2 (#2180 concern boundary-and-prose-pin-gaps): the threshold is a
    # STRICT `>` — tie_fraction exactly 0.5 is NOT degenerate (non-degenerate
    # verdict, no firing), pinning the boundary semantics.
    out = ms.audit_matched_artifact({}, tie_fraction=0.5)
    assert out["degenerate"] is False
    assert out["fires"] is False


# ------------------------------------------------- unit: companion hardening (r2)


def test_audit_on_support_none_placeholder_is_not_companion():
    # A null placeholder under a *_on_support key is NOT a companion read
    # (#2180 concern companion-false-negative).
    art = {"results": {"partial_on_support": None}}
    out = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out["has_support_companion"] is False
    assert out["fires"] is True


def test_audit_unsupported_named_block_is_not_support_defined():
    # "unsupported"-style names must not substring-match the support token.
    art = {
        "pools": {
            "unsupported_rows": {"n": 5, "partial": 0.1},
            "full": {"n": 10, "partial": 0.2},
        }
    }
    out = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out["has_support_companion"] is False
    assert out["fires"] is True


def test_audit_population_block_with_only_n_is_not_companion():
    # A support-defined sub-block carrying ONLY a sample-size key has no
    # audited-statistic payload — structure without a companion read.
    art = {
        "populations": {
            "full": {"n": 10, "partial": 0.2},
            "support": {"n": 3},
        }
    }
    out = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out["has_support_companion"] is False
    assert out["fires"] is True


def test_audit_root_level_population_block_is_companion():
    # The scanned root may itself BE the per-population dict (round-1 Claude
    # minor: arm (b) previously scanned dict VALUES only, missing the root).
    art = {
        "full": {"n": 10, "partial": 0.2},
        "train_active": {"n": 3, "definition": "rows with count > 0", "partial": -0.01},
    }
    out = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out["has_support_companion"] is True
    assert out["fires"] is False


def test_audit_result_path_scopes_companion_to_headline_dv():
    # Per-DV scoping (#2180 concern companion-false-negative, cross-DV arm):
    # a SIBLING DV's companion must not read as coverage for THIS headline.
    art = {
        "per_dv": {
            "dv_with_companion": {"partial": 0.24, "partial_on_support": -0.003},
            "dv_headline": {"partial": 0.21},
        }
    }
    out = ms.audit_matched_artifact(art, tie_fraction=0.9, result_path=("per_dv", "dv_headline"))
    assert out["has_support_companion"] is False
    assert out["fires"] is True
    assert out["companion_scope"] == "per_dv/dv_headline"
    out2 = ms.audit_matched_artifact(art, tie_fraction=0.9, result_path="per_dv/dv_with_companion")
    assert out2["has_support_companion"] is True
    assert out2["fires"] is False
    # Artifact-global default (backward compat): the sibling companion IS
    # visible — exactly the contamination result_path exists to cut; the
    # docstring names the per-DV call as the correct grain.
    out3 = ms.audit_matched_artifact(art, tie_fraction=0.9)
    assert out3["has_support_companion"] is True
    assert out3["companion_scope"] == "artifact-global"


def test_audit_result_path_unresolvable_raises():
    with pytest.raises(ValueError, match="result_path"):
        ms.audit_matched_artifact({"per_dv": {}}, tie_fraction=0.9, result_path="per_dv/nope")


# ------------------------------------------------- unit: fail-fast validation (r2)


def test_audit_recorded_nan_tie_fraction_raises():
    # A recorded NaN must RAISE, never silently read fires=False
    # (#2180 concern audit-nonfinite-silent-pass; nan > 0.5 is False).
    with pytest.raises(ValueError, match="finite"):
        ms.audit_matched_artifact({"match_tie_fraction": float("nan")})


def test_audit_recorded_inf_tie_fraction_raises():
    with pytest.raises(ValueError, match="finite"):
        ms.audit_matched_artifact({"match_tie_fraction": float("inf")})


def test_audit_recorded_bool_is_not_a_tie_fraction_source():
    # bool is an int subclass; a boolean field is never a tie fraction
    # (round-1 Claude minor, folded into r2).
    with pytest.raises(ValueError, match="no tie-fraction source"):
        ms.audit_matched_artifact({"match_tie_fraction": True})


def test_audit_threshold_nan_raises():
    with pytest.raises(ValueError, match="threshold"):
        ms.audit_matched_artifact({}, tie_fraction=0.4, threshold=float("nan"))


def test_audit_threshold_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ms.audit_matched_artifact({}, tie_fraction=0.4, threshold=1.5)


# ------------------------------------------------------- regression: #2163 fixtures


def test_recorded_full_pool_tie_fraction_pin():
    data = json.loads((FIXTURE_DIR / "population_partials.json").read_text())
    tf = data["populations"]["full"]["match_tie_fraction"]
    assert tf == pytest.approx(RECORDED_FULL_TIE_FRACTION)
    assert tf > 0.5


def test_predictor_partials_fires_known_positive():
    """The #2163 headline artifact: degenerate covariate, no companion -> fires."""
    art = json.loads((FIXTURE_DIR / "predictor_partials.json").read_text())
    out = ms.audit_matched_artifact(art, tie_fraction=RECORDED_FULL_TIE_FRACTION)
    assert out["degenerate"] is True
    assert out["has_support_companion"] is False
    assert out["fires"] is True


def test_population_partials_negative_control_does_not_fire():
    """The remediation artifact: same degenerate covariate, but the
    per-population block (companion arm (b): >=2 sized sub-blocks, >=1
    support-defined — train_active's definition carries "> 0") counts as
    the support-restricted companion -> fires=False."""
    art = json.loads((FIXTURE_DIR / "population_partials.json").read_text())
    out = ms.audit_matched_artifact(art)
    assert out["tie_fraction_source"] == "recorded-field"
    assert out["tie_fraction"] == pytest.approx(RECORDED_FULL_TIE_FRACTION)
    assert out["degenerate"] is True
    assert out["has_support_companion"] is True
    assert out["fires"] is False


def test_census_raw_covariate_path():
    """Raw-covariate resolution on the committed census: full-width modal
    share (n=131,072, no NaNs) — the ~0.002 gap vs the recorded
    complete-case value is the caller-contract distinction."""
    with np.load(FIXTURE_DIR / "census.npz") as z:
        col = np.asarray(z["lasttoken_count"])
    assert ms.tied_fraction(col) == pytest.approx(CENSUS_RAW_TIE_FRACTION)
    art = json.loads((FIXTURE_DIR / "predictor_partials.json").read_text())
    out = ms.audit_matched_artifact(art, covariate=col)
    assert out["tie_fraction_source"] == "covariate-values"
    assert out["fires"] is True


# ----------------------------------------------------------------- prose pins


def test_helper_exposes_all_four_reads():
    for name in ("tied_fraction", "tie_profile", "support_mask", "audit_matched_artifact"):
        assert callable(getattr(ms, name))


def test_lens_reference_carries_item_18():
    """critic-lens-reference.md Statistics span carries item 18 with the
    canonical definitions + N/A escape."""
    text = (REPO / ".claude/rules/critic-lens-reference.md").read_text(encoding="utf-8")
    span = text.split("### Statistics & Measurement lens", 1)[1]
    span = span.split("### Alternative Explanations lens", 1)[0]
    assert re.search(r"^18\. \*\*Matched-covariate support", span, re.M)
    assert "tied fraction" in span and "support-restricted companion" in span
    assert "0.5" in span
    # wrap-robust: the escape phrase line-wraps in the lens file
    assert "N/A — matching covariate" in span
    assert "non-degenerate (tied fraction" in span
    assert "matched_support" in span  # canonical-helper pointer


def test_statistics_critic_owns_item_18():
    text = (REPO / ".claude/agents/statistics-critic.md").read_text(encoding="utf-8")
    assert "18. Matched-covariate support" in text
    assert "support-restricted companion" in text
    # anti-patterns table row
    assert "matched-covariate support (item 18)" in text


def test_interpretation_critic_lens3_carries_degenerate_matching_bullet():
    text = (REPO / ".claude/agents/interpretation-critic.md").read_text(encoding="utf-8")
    tail = text.split("### 3. Alternative Explanations", 1)[1]
    tail = tail.split("### 4.", 1)[0]
    assert "match_tie_fraction" in tail
    assert "support-restricted companion" in tail
    assert "#2163" in tail
    # r2 (#2180 concern missing-recompute-fallback): the recompute fallback is
    # the load-bearing half — the #2163 known-positive artifact records ZERO
    # tie-fraction fields, so a read-the-field-only instruction has no object
    # on exactly the artifact class the lens exists to catch. Pin BOTH the
    # recompute instruction and the absent-field trigger vocabulary so silent
    # removal fails this test.
    assert "recompute the modal-value share" in tail
    assert "when absent" in tail


def test_lens_reference_item18_pins_grain_and_degenerate_limit():
    """r2 (#2180 concern boundary-and-prose-pin-gaps): pin the per-headline
    audit grain + the tied-fraction-near-1.0 degenerate-limit remedy in
    critic-lens-reference item 18."""
    text = (REPO / ".claude/rules/critic-lens-reference.md").read_text(encoding="utf-8")
    span = text.split("### Statistics & Measurement lens", 1)[1]
    span = span.split("### Alternative Explanations lens", 1)[0]
    # Per-headline grain (wrap-robust: the phrase line-wraps in the lens file).
    assert "AUDIT GRAIN: per" in span
    assert "HEADLINE STATISTIC" in span
    # Degenerate-limit remedy: near-1.0 tied fraction => drop/replace the
    # covariate, not a companion read.
    assert "Degenerate limit: at tied fraction" in span
    assert "dropping or replacing the matching covariate" in span


def test_critic_capsule_names_item_18():
    text = (REPO / ".claude/agents/critic.md").read_text(encoding="utf-8")
    assert "18 matched-covariate support" in text


def test_lens_coverage_map_carries_row_18():
    text = (REPO / ".claude/rules/lens-coverage-map.md").read_text(encoding="utf-8")
    assert "critic.md Statistics 18" in text and "v2-owner: statistics-critic" in text
