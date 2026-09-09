"""The plot must use the hash-verified completed summary and preserve its values."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2569_refusal_magnitude_figure as figure


def test_rendered_metadata_matches_completed_summary(tmp_path):
    """An actual plot-only render exports three hashed images without changing data."""
    output = tmp_path / "figure"
    meta = figure.render(figure.SOURCE, output)
    summary = json.loads((figure.SOURCE / "summary.json").read_text())
    assert meta["values"] == summary["primary"]
    assert meta["include_width_frac"] == 0.5
    assert meta["resolved_font"]
    for name, expected in meta["outputs_sha256"].items():
        assert figure.sha256(output / name) == expected


def test_unverified_summary_is_rejected(tmp_path, monkeypatch):
    """A modified or partially replaced summary cannot inherit a completed status."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "summary.json").write_text(json.dumps({"primary": {"n": 124, "n_clusters": 21}}))
    (source / "completion.json").write_text(
        json.dumps(
            {"exit_code": 0, "outputs": [{"path": "source/summary.json", "sha256": "invalid"}]}
        )
    )
    monkeypatch.setattr(figure, "REPO", tmp_path)
    with pytest.raises(AssertionError):
        figure.render(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
