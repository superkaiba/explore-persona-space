"""Software regressions for test leakage, sparse support and fit-cache identity."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.context_risk_followup_analyze import analyze_population, cached_fit
from scripts.context_risk_followup_features import FeatureBank


def fixture():
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "eval_results/context_risk_followup_design/analysis_spec.json"
        ).read_text()
    )
    # Software fixture uses the full real grid and all actual readout paths.
    rows = []
    for task in range(18):
        for condition in ("conflicting", "oneoff", "original"):
            rows.append(
                {
                    "task_id": f"fixture_{task:02}",
                    "condition": condition,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Implement fixture function {task}. {condition}. "
                                f"assert function({task}) == {task % 3}"
                            ),
                        }
                    ],
                    "test": f"assert function({task}) == {task % 3}",
                    "exact_context_sha256": f"fixture_{task}_{condition}",
                    "public_test_role": "probe_training" if task < 12 else "final_test",
                    "positive": 2 if condition == "original" else task % 4,
                    "trials": 4,
                }
            )
    rng = np.random.default_rng(182)
    raw = rng.normal(size=(len(rows), 11)).astype(np.float32)
    arrays = {
        "weight": rng.normal(size=(11, 11)),
        "x_mean": rng.normal(size=11),
        "x_scale": np.exp(rng.normal(size=11)),
        "y_mean": rng.normal(size=11),
    }
    return spec, rows, raw, arrays


def test_final_test_labels_do_not_change_any_tuning(tmp_path):
    spec, rows, raw, arrays = fixture()
    before = analyze_population(
        rows,
        FeatureBank(rows, raw, arrays, spec),
        spec,
        tmp_path / "first",
        {"software_fixture": 1},
        eligible_only=True,
    )
    changed = [
        {**r, "positive": 4 - r["positive"]}
        if r["public_test_role"] == "final_test" and r["condition"] != "original"
        else dict(r)
        for r in rows
    ]
    after = analyze_population(
        changed,
        FeatureBank(changed, raw, arrays, spec),
        spec,
        tmp_path / "second",
        {"software_fixture": 2},
        eligible_only=True,
    )
    assert before["status"] == after["status"] == "complete"
    for method in spec["readouts"] + [
        f"orientation_{s}" for s in spec["orientation_controls"]["seeds"]
    ]:
        assert before["models"][method]["logits"] == after["models"][method]["logits"]
        if method != "prevalence":
            assert (
                before["models"][method]["selection"]["C"]
                == after["models"][method]["selection"]["C"]
            )
            assert (
                before["models"][method]["selection"]["cv_rows"]
                == after["models"][method]["selection"]["cv_rows"]
            )


def test_missing_positive_support_never_fits(tmp_path):
    spec, rows, raw, arrays = fixture()
    for row in rows:
        if row["condition"] != "original":
            row["positive"] = 0
    report = analyze_population(
        rows,
        FeatureBank(rows, raw, arrays, spec),
        spec,
        tmp_path,
        {"software_fixture": True},
        eligible_only=True,
    )
    assert report["status"] == "insufficient_training_support"
    assert report["models"] == {}
    assert not (tmp_path / "fits").exists()


def test_resume_refuses_changed_regime(tmp_path):
    path = tmp_path / "fit.json"
    x = np.array([[0.0], [1.0], [2.0]])
    k, n = np.array([0, 1, 2]), np.array([2, 2, 2])
    first = cached_fit(path, "original", x, x, k, n, 1.0, np.eye(1))
    second = cached_fit(path, "original", x, x, k, n, 1.0, np.eye(1))
    np.testing.assert_array_equal(first["logits"], second["logits"])
    with pytest.raises(ValueError, match="Stale fit checkpoint"):
        cached_fit(path, "changed", x, x, k, n, 1.0, np.eye(1))
