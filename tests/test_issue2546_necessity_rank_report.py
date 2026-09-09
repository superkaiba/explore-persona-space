"""Report pairing, conditional bootstrap, and completeness checks without model fits."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import issue2546_necessity_rank_report as report


def test_paired_bootstrap_exact_relation_and_pairing_guard():
    """All paired differences survive resampling; corrupt alignment is rejected."""
    banks = {}
    for index, arm in enumerate(report.analysis.ARMS):
        banks[arm] = [
            {
                "row_id": f"q{i}",
                "corpus": "a" if i < 3 else "b",
                "sse_full": 10 - index,
                "sst_corpus": 20,
                "sst_global": 30,
            }
            for i in range(6)
        ]
    result = report.paired_bootstrap(banks)
    np.testing.assert_allclose(result["contrasts"]["thinking_mode"]["corpus"]["ci95"], [0.1, 0.1])
    banks["context"].reverse()
    with pytest.raises(ValueError, match="Unpaired"):
        report.paired_bootstrap(banks)


def test_predictive_area_definition_and_invalid_ceiling():
    """Retain the exact descriptive clipping formula, without a CDF interpretation."""
    assert report.predictive_area([-1.0, 0.6, 0.4, 0.5]) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        report.predictive_area([-0.1, 0.0])


def test_full_summary_real_body(tmp_path):
    """Exercise all fifteen cell reads, parity checks, aggregations, and result writes."""
    out, data = tmp_path / "results", tmp_path / "source"
    out.mkdir()
    (data / "allfit/results").mkdir(parents=True)
    recipe = {"subset_counts": {s: 35 for s in report.analysis.SUBSETS}}
    for index, (arm, (cell, _, _)) in enumerate(report.analysis.ARMS.items()):
        error = 2 - index * 0.1
        original = {
            "subsets": {
                s: {"r2_corpus": 1 - error / 4, "r2_global": 1 - error / 5, "acc1": 1}
                for s in (*report.analysis.SUBSETS, "all")
            }
        }
        (data / "allfit/results" / f"{cell}__a3.json").write_text(json.dumps(original))
        for fold in range(5):
            path = out / f"{arm}__fold{fold}.json"
            with path.with_suffix(".csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "row_id",
                        "subset",
                        "corpus",
                        "sse_full",
                        "sst_corpus",
                        "sst_global",
                        "hit_full",
                    ],
                )
                writer.writeheader()
                for subset in report.analysis.SUBSETS:
                    for corpus in report.analysis.parent.CORPORA:
                        writer.writerow(
                            {
                                "row_id": f"{corpus}:{fold}:{subset}",
                                "subset": subset,
                                "corpus": corpus,
                                "sse_full": error,
                                "sst_corpus": 4,
                                "sst_global": 5,
                                "hit_full": 1,
                            }
                        )
            values = {
                "status": "complete",
                "arm": arm,
                "fold": fold,
                "cache_key": {"recipe": recipe},
                "rows_sha256": report.analysis.parent.sha256(path.with_suffix(".csv")),
                "all_rows_test_sse": 14 * error,
                "all_rows_sst_corpus": 56,
                "subsets": {},
            }
            for subset in report.analysis.SUBSETS:
                values["subsets"][subset] = {
                    "test_sse_by_rank": [28, 21, error * 7],
                    "sst_corpus": 28,
                    "sst_global": 35,
                    "identity_bias_sse": 30,
                    "full_retrieval_hits": 7,
                    "rank10_retrieval_hits": 7,
                    "identity_bias_retrieval_hits": 0,
                    "selected_ranks": {t: 2 for t in ("0.05", "0.1", "0.2")},
                    "diversity": {
                        space: {m: 3 for m in report.METRICS}
                        for space in ("input", "answer", "fitted_output", "fitted_over_answer")
                    },
                    "conditional_rank_bootstrap": {
                        "draws": [2] * 4000,
                        "median": 2,
                        "ci95": [2, 2],
                    },
                }
                values["subsets"][subset]["diversity"]["raw_input_participation_ratio"] = 3
            path.write_text(json.dumps(values))
    summary = report.summarize(out, data)
    assert summary["completed_cells"] == 15
    assert summary["subsets"]["necessary"]["n"] == 35
    assert summary["subsets"]["necessary"]["contrasts"]["thinking_mode"][
        "r2_change_corpus"
    ] == pytest.approx(0.05)
    assert (out / "summary.json").is_file()
    # Existing cell but altered underlying row scalars must not silently pass.
    (out / "no_think__fold0.csv").write_text("wrong bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        report.summarize(out, data)


def test_launcher_syntax_and_no_argument_guard():
    """Check the committed detached launcher without launching any analysis process."""
    path = Path(__file__).resolve().parents[1] / "scripts/launch_issue2546_necessity_rank.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(["bash", str(path)], check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" in result.stderr


@pytest.mark.parametrize(
    "entrypoint", ["issue2546_necessity_rank.py", "issue2546_necessity_rank_report.py"]
)
def test_absolute_script_imports_from_outside_repository(entrypoint, tmp_path):
    """Exercise real script-mode imports; --help exits before any fit or output write."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / entrypoint), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout
