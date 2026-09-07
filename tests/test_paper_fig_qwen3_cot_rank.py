"""Guard the figure's completeness requirements without producing fake result figures."""

import json

import pytest
from scripts.paper_fig_qwen3_cot_rank import load_results


def test_incomplete_summary_fails(tmp_path):
    """A partial run must never produce a final figure."""
    (tmp_path / "summary.json").write_text(json.dumps({"status": "partial", "completed_cells": 2}))
    with pytest.raises(ValueError, match="complete"):
        load_results(tmp_path)
