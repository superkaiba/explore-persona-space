"""End-to-end test of analyze_507 against the REAL #470 regression.json.

Round-2 regression test for code-review Critical 4: previously
analyze_507.PRIMARY_PREDICTORS asked for keys ("cosine_layer_headline",
"cosine_response_token_headline", "js_sequence_rb") that phase5_regress.py
never emits, and it expected the top-level "per_predictor" key when the
writer actually uses "predictors". A synthetic fixture papered over both
bugs; this test uses the actual file committed on the issue-470 branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REAL_REGRESSION_FIXTURE = (
    Path(__file__).parent / "fixtures" / "issue_507" / "regression_470_real.json"
)


@pytest.fixture
def real_regression_path() -> Path:
    if not REAL_REGRESSION_FIXTURE.exists():
        pytest.skip(
            f"Real regression fixture missing at {REAL_REGRESSION_FIXTURE}. "
            "Regenerate via: git show issue-470:eval_results/issue_470/regression.json > "
            f"{REAL_REGRESSION_FIXTURE}"
        )
    return REAL_REGRESSION_FIXTURE


def test_load_regression_reads_real_predictors_key(real_regression_path):
    """_load_regression must read the real `predictors` key without raising."""
    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _load_regression,
    )

    payload = _load_regression(real_regression_path, "test_arm")
    assert "predictors" in payload, "expected top-level 'predictors' key"
    # The 13 actual phase5 predictor labels.
    expected_labels = {
        "cosine_l20_baseline",
        "cosine_response_l7",
        "cosine_response_l14",
        "cosine_response_l21",
        "cosine_response_l27",
        "cosine_response_headline",
        "M_js",
        "JS_sym_nats",
        "KL_src_to_bys_nats",
        "KL_bys_to_src_nats",
        "KL_sym_nats",
        "bystander_base_rate",
        "base_rate_diff_neg_abs",
    }
    actual_labels = set(payload["predictors"].keys())
    missing = expected_labels - actual_labels
    assert not missing, f"Real regression.json is missing predictor labels: {missing}"


def test_primary_predictors_all_present_in_real_regression(real_regression_path):
    """The 3 PRIMARY_PREDICTORS must be keys phase5_regress actually emits."""
    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        PRIMARY_PREDICTORS,
        _load_regression,
    )

    payload = _load_regression(real_regression_path, "test_arm")
    predictors = payload["predictors"]
    for pred in PRIMARY_PREDICTORS:
        assert pred in predictors, (
            f"PRIMARY_PREDICTORS contains {pred!r} but phase5_regress.py does NOT "
            f"emit that label. Real labels: {sorted(predictors.keys())}. "
            "This is the round-1 schema-mismatch bug — fix PRIMARY_PREDICTORS."
        )


def test_cross_arm_compare_runs_end_to_end_on_real_file(real_regression_path, tmp_path):
    """End-to-end: run cross_arm_compare with the SAME real file as both arms.

    This is degenerate-but-deterministic (rho_72b - rho_7b == 0 everywhere)
    AND it exercises every code path that previously crashed under the bad
    schema. The point is to PROVE the code reaches the end and writes output.
    """
    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        cross_arm_compare,
    )

    output_path = tmp_path / "cross_arm.json"
    result = cross_arm_compare(
        regression_7b=real_regression_path,
        regression_72b=real_regression_path,
        output_path=output_path,
        n_boot=200,  # tiny for test speed; structure is what matters
        seed=42,
    )

    # Output written.
    assert output_path.exists(), "cross_arm.json was not written"
    saved = json.loads(output_path.read_text())
    assert saved == result

    # Has the right top-level shape.
    assert "arms" in result
    assert "per_predictor" in result
    assert "verdict_grid_inputs" in result
    assert "metadata" in result

    # Each primary predictor has a per-source rho dict + paired CI block.
    for pred in ("cosine_l20_baseline", "cosine_response_headline", "M_js"):
        assert pred in result["per_predictor"], pred
        block = result["per_predictor"][pred]
        # The "missing_from_one_arm" note must NOT appear — both arms have it.
        assert "note" not in block or block.get("note") != "missing_from_one_arm", (
            f"predictor {pred!r} reported missing from one arm even though we "
            f"passed the SAME real file twice — schema reader regressed."
        )
        # Identical-file → diff per source == 0 → point_estimate == 0.
        paired = block["paired_abs_rho_diff_72b_minus_7b"]
        assert paired.get("point_estimate") == 0.0 or paired.get("point_estimate") is None, (
            f"Identical-arm cross-arm diff should be 0 for {pred}; got "
            f"{paired.get('point_estimate')!r}"
        )

    # Verdict-grid inputs must be computed (not None due to missing keys).
    vgi = result["verdict_grid_inputs"]
    # input_1 reads from per_source.<src>["rho"] and per_source.<src>["p"] —
    # not "p_one_tail". If the round-1 bug were live, mean_p_72b_headline
    # would be None.
    assert "mean_per_source_p_72b_headline" in vgi
    assert vgi["mean_per_source_p_72b_headline"] is not None, (
        "mean_per_source_p_72b_headline is None — phase5 emits 'p', not 'p_one_tail'."
    )

    # input_2 reads from paired_bootstrap_delta_rho_vs_base_rate at the TOP
    # level (not per-predictor). If the round-1 bug were live, input_2 would
    # also be None because the key lookup would have missed.
    assert "input_2_72b_beats_base_rate_null" in vgi
    # input_2 is allowed to be True or False but must NOT be None on real data.
    assert vgi["input_2_72b_beats_base_rate_null"] is not None, (
        "input_2 is None — the verdict-grid reader failed to find "
        "paired_bootstrap_delta_rho_vs_base_rate at the top level of "
        "phase5_regress's output."
    )

    # Pooled-rho inputs (input_3) must read .spearman_source_fe.rho on real data.
    assert vgi["pooled_source_fe_rho_72b"] is not None
    assert vgi["pooled_source_fe_rho_7b"] is not None
    # Same arm both sides → equal.
    assert vgi["pooled_source_fe_rho_72b"] == vgi["pooled_source_fe_rho_7b"]


def test_load_regression_legacy_per_predictor_compat(tmp_path):
    """A regression file using the legacy 'per_predictor' key must still load
    (with warning). Guards against accidentally breaking older committed files.
    """
    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _load_regression,
    )

    legacy = {
        "per_predictor": {
            "cosine_response_headline": {
                "spearman_source_fe": {"rho": 0.42},
                "per_source": {},
            }
        },
        "paired_bootstrap_delta_rho_vs_base_rate": {
            "ci_low_95": 0.05,
            "ci_high_95": 0.20,
        },
    }
    legacy_path = tmp_path / "legacy_regression.json"
    legacy_path.write_text(json.dumps(legacy))

    payload = _load_regression(legacy_path, "legacy_test")
    # Should be normalized to "predictors" in-memory.
    assert "predictors" in payload
    assert payload["predictors"]["cosine_response_headline"]["spearman_source_fe"]["rho"] == 0.42


def test_load_regression_missing_predictors_raises(tmp_path):
    """A regression file missing BOTH 'predictors' AND 'per_predictor' must
    raise loud (not silently produce an empty cross-arm output).
    """
    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _load_regression,
    )

    broken = {"some_other_key": {}}
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken))

    with pytest.raises(RuntimeError, match="missing the 'predictors' key"):
        _load_regression(broken_path, "broken_test")
