"""Issue #640 — the off-pod scorer joins diagonal Δsource vs v3 Δleakage per row.

Plan v6 §6: ``issue640_diagonal_score`` joins this run's diagonal Δsource against
v3's committed off-diagonal Δleakage PER ROW (the two files use DIFFERENT columns,
so the join must be on the bare row, never the ``row|column`` cell key), computes
``selectivity_gap = Δleakage - Δsource`` per (row, seed), per-row cross-seed sign
consistency on Δsource, and the selective / blunt-revert / mixed verdict on the
two reckless cells. reversed_fact is carried with a floor flag and excluded from
the headline; marker is a separate log-prob null reference.

These tests build tiny diagonal + leakage fixtures in a temp root and exercise
``score_diagonal`` directly — CPU only, no GPU, no real model.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Diagonal columns (the on-target map) vs the off-diagonal leakage columns v3 used
# — DELIBERATELY DIFFERENT per row, so a correct join must key on the row.
DIAG_COLS = {
    "bad_medical": "fam_expr_bad_medical",
    "risky_financial": "fam_expr_risky_financial",
    "extreme_sports": "fam_expr_extreme_sports",
    "taught_fact": "fact_expression",
    "reversed_fact": "fact_expression",
    "compliment_writing": "fam_expr_compliment",
    "wrong_claim_agreement": "sycophancy",
}
LEAK_COLS = {
    "bad_medical": "broad_em",
    "risky_financial": "fam_expr_extreme_sports",  # NB: risky's leakage col != its diag col
    "extreme_sports": "fam_expr_risky_financial",
    "taught_fact": "format_style",
    "reversed_fact": "format_style",
    "compliment_writing": "format_style",
    "wrong_claim_agreement": "persona_drift",
}


def _scorer():
    return importlib.import_module("issue640_diagonal_score")


def _write_diag(root: Path, seed: int, source_delta: dict[str, float]):
    detail = {}
    for row, ds in source_delta.items():
        col = DIAG_COLS[row]
        trained = 1.0
        detail[f"{row}|{col}"] = {
            "row": row,
            "column": col,
            "seed": seed,
            "patch_kind": "postfix",
            "trained_rate": trained,
            "patched_rate": trained - ds,
            "delta_source": ds,
            "n_probes": 32,
        }
    (root / f"diagonal_source_seed{seed}.json").write_text(
        json.dumps(
            {
                "group": "PST",
                "name": "patch_recovery_diagonal",
                "seed": seed,
                "patch_kind": "postfix",
                "target": "diagonal",
                "cells": {k: v["delta_source"] for k, v in detail.items()},
                "detail": detail,
            }
        )
    )


def _write_leak(root: Path, seed: int, leak_delta: dict[str, float]):
    detail = {}
    for row, dl in leak_delta.items():
        col = LEAK_COLS[row]
        trained = 1.0
        detail[f"{row}|{col}"] = {
            "row": row,
            "column": col,
            "seed": seed,
            "patch_kind": "postfix",
            "trained_rate": trained,
            "patched_rate": trained - dl,
            "delta_leakage": dl,
            "n_probes": 32,
        }
    (root / f"patch_cells_postfix_seed{seed}.json").write_text(
        json.dumps(
            {
                "group": "PST",
                "name": "patch_recovery_postfix",
                "seed": seed,
                "patch_kind": "postfix",
                "cells": {k: v["delta_leakage"] for k, v in detail.items()},
                "detail": detail,
            }
        )
    )


def _setup(monkeypatch, tmp_path, *, diag0, diag137, leak0, leak137):
    sc = _scorer()
    root = tmp_path / "issue_640"
    root.mkdir(parents=True)
    _write_diag(root, 0, diag0)
    _write_diag(root, 137, diag137)
    _write_leak(root, 0, leak0)
    _write_leak(root, 137, leak137)
    monkeypatch.setattr(sc, "_i640_root", lambda: root)
    return sc, root


_ALL = list(DIAG_COLS)


def test_join_keys_on_row_and_computes_gap(monkeypatch, tmp_path):
    """The join pairs each row's diagonal Δsource with its leakage Δ (different cols)."""
    diag = {r: 0.10 for r in _ALL}
    leak = {r: 0.60 for r in _ALL}
    sc, root = _setup(monkeypatch, tmp_path, diag0=diag, diag137=diag, leak0=leak, leak137=leak)

    sc.score_diagonal(seeds=[0, 137], smoke=True)
    out = json.loads((root / "selectivity_comparison.json").read_text())
    per_row = out["selectivity"]["per_row"]

    risky = per_row["risky_financial"]["seeds"]["0"]
    # The leakage column for risky is fam_expr_extreme_sports; the diagonal column
    # is fam_expr_risky_financial — a correct row-keyed join still pairs them.
    assert risky["diagonal_column"] == "fam_expr_risky_financial"
    assert risky["leakage_column"] == "fam_expr_extreme_sports"
    assert abs(risky["delta_source"] - 0.10) < 1e-9
    assert abs(risky["delta_leakage"] - 0.60) < 1e-9
    assert abs(risky["selectivity_gap"] - 0.50) < 1e-9, "gap = Δleakage - Δsource"


def test_sign_consistency_on_delta_source(monkeypatch, tmp_path):
    """Per-row cross-seed sign consistency on Δsource is computed."""
    sc, root = _setup(
        monkeypatch,
        tmp_path,
        diag0={r: 0.10 for r in _ALL},
        diag137={**{r: 0.10 for r in _ALL}, "bad_medical": -0.05},  # bad_medical flips sign
        leak0={r: 0.60 for r in _ALL},
        leak137={r: 0.60 for r in _ALL},
    )
    sc.score_diagonal(seeds=[0, 137], smoke=True)
    per_row = json.loads((root / "selectivity_comparison.json").read_text())["selectivity"][
        "per_row"
    ]
    assert per_row["risky_financial"]["delta_source_sign_consistent"] is True
    assert per_row["bad_medical"]["delta_source_sign_consistent"] is False


def test_reversed_fact_floor_excluded_from_headline(monkeypatch, tmp_path):
    """reversed_fact carries a floor flag and is excluded from the headline rows."""
    sc, root = _setup(
        monkeypatch,
        tmp_path,
        diag0={r: 0.10 for r in _ALL},
        diag137={r: 0.10 for r in _ALL},
        leak0={r: 0.60 for r in _ALL},
        leak137={r: 0.60 for r in _ALL},
    )
    sc.score_diagonal(seeds=[0, 137], smoke=True)
    sel = json.loads((root / "selectivity_comparison.json").read_text())["selectivity"]
    assert sel["per_row"]["reversed_fact"]["is_floor"] is True
    assert sel["per_row"]["reversed_fact"]["in_headline"] is False
    assert "reversed_fact" not in sel["headline_rows"]
    # The headline verdict only weighs the two reckless cells.
    assert sel["headline"]["primary_cells"] == ["risky_financial", "extreme_sports"]


def test_selective_verdict_and_marker_reference(monkeypatch, tmp_path):
    """Small Δsource + large Δleakage on both reckless cells -> selective; marker ref folded."""
    sc, root = _setup(
        monkeypatch,
        tmp_path,
        diag0={r: 0.10 for r in _ALL},
        diag137={r: 0.12 for r in _ALL},
        leak0={r: 0.60 for r in _ALL},
        leak137={r: 0.60 for r in _ALL},
    )
    sc.score_diagonal(seeds=[0, 137], smoke=True)
    out = json.loads((root / "selectivity_comparison.json").read_text())
    assert out["selectivity"]["headline"]["verdict"] == "selective"
    # Marker carried as a descriptive null reference (5.7738 / 9.4478 nats), not a cell.
    ref = out["provenance"]["marker_diagonal_reference_nats"]
    assert abs(ref["0"] - 5.773762626647949) < 1e-6
    assert abs(ref["137"] - 9.447760429382324) < 1e-6
    assert "marker" not in out["selectivity"]["per_row"]


def test_blunt_verdict(monkeypatch, tmp_path):
    """Δsource ≈ Δleakage on both reckless cells (gap ~0) -> blunt-revert."""
    sc, root = _setup(
        monkeypatch,
        tmp_path,
        diag0={r: 0.60 for r in _ALL},
        diag137={r: 0.60 for r in _ALL},
        leak0={r: 0.60 for r in _ALL},
        leak137={r: 0.60 for r in _ALL},
    )
    sc.score_diagonal(seeds=[0, 137], smoke=True)
    verdict = json.loads((root / "selectivity_comparison.json").read_text())["selectivity"][
        "headline"
    ]["verdict"]
    assert verdict == "blunt-revert"


def test_missing_leakage_baseline_fails_loud(monkeypatch, tmp_path):
    """A missing v3 patch_cells_postfix_seed{seed}.json raises (no silent skip)."""
    import pytest

    sc = _scorer()
    root = tmp_path / "issue_640"
    root.mkdir(parents=True)
    _write_diag(root, 0, {r: 0.10 for r in _ALL})
    # NB: no leakage file written.
    monkeypatch.setattr(sc, "_i640_root", lambda: root)
    with pytest.raises(FileNotFoundError, match="off-diagonal"):
        sc.score_diagonal(seeds=[0], smoke=True)
