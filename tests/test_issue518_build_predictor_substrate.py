"""Regression tests for ``scripts/issue518_build_predictor_substrate.py``.

Round-12 (task #518) target: the syco-arm code path that bypasses ``_load_runs``
and reads per-(source, bystander) ``delta`` + ``trained_rate`` + ``base_rate``
DIRECTLY from #411's frozen 138-cell analyze_summary snapshot at
``eval_results/issue_480/_inputs/syco_411_analyze_summary.json``.

The bug being regression-tested: round-7 through round-11 all used
``_load_runs(runs_root)`` unconditionally, but ``eval_results/issue_509/syco_arm/runs/``
does not exist (#509 inherited the syco panel from #411 via the snapshot,
not by re-training). The substrate builder previously raised
``FileNotFoundError: runs_root missing`` on every production launch.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Importlib-load the script (it is not a package member; lives under scripts/).
_SPEC = importlib.util.spec_from_file_location(
    "issue518_build_predictor_substrate",
    REPO / "scripts" / "issue518_build_predictor_substrate.py",
)
assert _SPEC is not None and _SPEC.loader is not None
substrate_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(substrate_mod)


def test_load_syco_cells_from_analyze_summary_returns_138_cells() -> None:
    """The #411 frozen snapshot yields 138 off-diagonal cells (6 sources x 23 bystanders)."""
    path = REPO / "eval_results" / "issue_480" / "_inputs" / "syco_411_analyze_summary.json"
    assert path.exists(), f"baseline snapshot missing at {path}"
    cells = substrate_mod._load_syco_cells_from_analyze_summary(path)
    assert len(cells) == 138, f"expected 138 cells, got {len(cells)}"
    # Sources are the canonical #411 six-source set.
    sources = {c["source"] for c in cells}
    assert sources == {
        "villain",
        "comedian",
        "assistant",
        "qwen_default",
        "software_engineer",
        "kindergarten_teacher",
    }, f"unexpected sources: {sources}"
    # Off-diagonal only: source != bystander for every cell.
    assert all(c["source"] != c["bystander"] for c in cells), "self-pair leaked"
    # Each cell carries the four required fields.
    required = {"source", "bystander", "delta", "trained_rate", "base_rate"}
    for c in cells:
        missing = required - set(c.keys())
        assert not missing, f"cell {c['source']!r}/{c['bystander']!r} missing {missing}"


def test_load_syco_cells_fields_are_finite_floats() -> None:
    """delta / trained_rate / base_rate must be finite floats (not NaN) for #411 panel."""
    import math

    path = REPO / "eval_results" / "issue_480" / "_inputs" / "syco_411_analyze_summary.json"
    cells = substrate_mod._load_syco_cells_from_analyze_summary(path)
    for c in cells:
        for field in ("delta", "trained_rate", "base_rate"):
            v = c[field]
            assert isinstance(v, float), f"{field} not float: {type(v)}"
            assert math.isfinite(v), f"{field}={v} not finite for ({c['source']}, {c['bystander']})"


def test_load_syco_cells_missing_path_raises_filenotfound() -> None:
    """A missing analyze_summary path raises FileNotFoundError with a useful hint."""
    import pytest

    bogus = REPO / "eval_results" / "definitely_not_a_real_path.json"
    with pytest.raises(FileNotFoundError, match="syco analyze_summary missing"):
        substrate_mod._load_syco_cells_from_analyze_summary(bogus)


def test_load_syco_cells_bad_schema_raises_runtimeerror(tmp_path: Path) -> None:
    """A JSON with no per_source key raises RuntimeError, not a silent zero-cell return."""
    import pytest

    bad = tmp_path / "bad_schema.json"
    bad.write_text(json.dumps({"some_other_top_key": [1, 2, 3]}))
    with pytest.raises(RuntimeError, match="no 'per_source' key"):
        substrate_mod._load_syco_cells_from_analyze_summary(bad)


def test_resolve_runs_syco_bypasses_load_runs() -> None:
    """The syco code path must NOT need runs_root (it reads the snapshot)."""
    path = REPO / "eval_results" / "issue_480" / "_inputs" / "syco_411_analyze_summary.json"
    # Pass runs_root=None to prove the syco branch never touches it.
    runs = substrate_mod._resolve_runs(
        arm="syco",
        runs_root=None,
        syco_analyze_summary=path,
    )
    # Same shape as _load_runs: {source: {"source": src, "per_cell": [...]}}.
    assert isinstance(runs, dict)
    assert len(runs) == 6, f"expected 6 sources, got {len(runs)}"
    total = sum(len(r["per_cell"]) for r in runs.values())
    assert total == 138, f"expected 138 cells across sources, got {total}"
    for src, r in runs.items():
        assert r["source"] == src
        assert isinstance(r["per_cell"], list)
        for cell in r["per_cell"]:
            assert cell["source"] == src
            assert cell["bystander"] != src


def test_resolve_runs_syco_requires_analyze_summary() -> None:
    """syco arm with no analyze_summary path raises ValueError, not a silent fallthrough."""
    import pytest

    with pytest.raises(ValueError, match="syco arm requires --syco-analyze-summary"):
        substrate_mod._resolve_runs(arm="syco", runs_root=None, syco_analyze_summary=None)


def test_resolve_runs_non_syco_requires_runs_root() -> None:
    """Non-syco arms must pass --runs-root."""
    import pytest

    with pytest.raises(ValueError, match="requires --runs-root"):
        substrate_mod._resolve_runs(arm="refusal", runs_root=None, syco_analyze_summary=None)
