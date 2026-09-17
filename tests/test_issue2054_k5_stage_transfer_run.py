"""The production publisher must fail closed on incomplete scientific coverage."""

from copy import deepcopy

import pytest

from scripts import issue2054_k5_stage_transfer_run as run


def test_coverage_rejects_duplicates_and_source_mismatch():
    result = {
        "source_sha": "a" * 40,
        "status": "complete",
        "rows": [
            {"label": label, "fold": fold} for label in run.analysis.LABELS for fold in range(5)
        ],
        "summary": [{} for _ in run.analysis.LABELS],
    }
    run.validate_result(result, "a" * 40)
    duplicate = deepcopy(result)
    duplicate["rows"][-1] = duplicate["rows"][0]
    with pytest.raises(ValueError, match="coverage"):
        run.validate_result(duplicate, "a" * 40)
    with pytest.raises(ValueError, match="coverage"):
        run.validate_result(result, "b" * 40)
