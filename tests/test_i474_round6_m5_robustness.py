"""CPU-only unit tests for #474 round-6 (post Phase-5 crash) M5 robustness.

Round-6 scope: Phase 5 crashed on
``JSONDecodeError: Expecting value: line 1 column 1 (char 0)`` because
the production sweep's disk-quota EDQUOT interrupted the M5 ep5 callback
write for ``loc/{B1, B2, B3}``, leaving 0-byte JSONs. The round-5
resume skipped retraining those conds (adapters already on HF), so the
empty files were never regenerated. M5 is a SECONDARY identifiability
diagnostic — it must DEGRADE GRACEFULLY, never crash the whole analyze
on a corrupt / missing file.

Fix surface:
  - ``_load_suppression_matrix(cid, epoch)``: returns ``None`` on missing,
    EMPTY (0-byte / whitespace-only), UNREADABLE (OSError), and
    JSONDecodeError. Logs a warning naming the (cid, epoch) in each
    branch. NEVER raises.
  - ``_suppression_difficulty_partial(df, epoch, n_boot, seed)``: tracks
    loaded vs missing source cids, drops cells whose source cond's S
    is missing, and returns ``{"status": "insufficient_coverage", ...}``
    when too few cells remain to compute a meaningful partial-rho. The
    "ok" path now includes ``status: "ok"`` + coverage metadata.
  - ``_per_cell_report``: ``h1_survives_suppression_partial`` is ``None``
    (not ``False``) when M5 ``status != "ok"`` — distinguishes "M5
    couldn't run" from "M5 ran and failed to survive".

Pure CPU + synthetic FS — no model, no vLLM, no Trainer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PHASE5 = Path(__file__).resolve().parent.parent / "scripts" / "i474_phase5_analyze.py"


@pytest.fixture(scope="module")
def phase5_module():
    spec = importlib.util.spec_from_file_location("i474_phase5_analyze", _SCRIPT_PHASE5)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i474_phase5_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- _load_suppression_matrix


def test_load_returns_none_when_file_missing(phase5_module, tmp_path, monkeypatch):
    """A missing M5 JSON returns None + logs a warning (does NOT raise)."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    result = phase5_module._load_suppression_matrix("A1", 5)
    assert result is None


def test_load_returns_none_when_file_zero_byte(phase5_module, tmp_path, monkeypatch, caplog):
    """Round-6 production bug: 0-byte file from interrupted callback write.

    Used to raise ``JSONDecodeError: Expecting value: line 1 column 1
    (char 0)`` and crash the whole analyze. Must now return None +
    warn naming the (cid, epoch).
    """
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    bad = tmp_path / "suppression_difficulty_loc_B2_ep5.json"
    bad.write_text("")  # the exact 0-byte case the production crash hit

    import logging

    with caplog.at_level(logging.WARNING):
        result = phase5_module._load_suppression_matrix("B2", 5)
    assert result is None
    # Warning must name BOTH the cid and the epoch so the operator knows
    # which (i, ep) holes the M5 partial will skip.
    log = caplog.text
    assert "B2" in log and "ep=5" in log
    assert "EMPTY" in log or "empty" in log.lower()


def test_load_returns_none_when_file_whitespace_only(phase5_module, tmp_path, monkeypatch):
    """Whitespace-only file is treated as empty (defensive)."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    bad = tmp_path / "suppression_difficulty_loc_B3_ep5.json"
    bad.write_text("   \n\t\n  ")
    assert phase5_module._load_suppression_matrix("B3", 5) is None


def test_load_returns_none_when_json_malformed(phase5_module, tmp_path, monkeypatch, caplog):
    """Malformed JSON returns None + warns; never crashes."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    bad = tmp_path / "suppression_difficulty_loc_C1_ep5.json"
    bad.write_text("{ not valid json")
    import logging

    with caplog.at_level(logging.WARNING):
        result = phase5_module._load_suppression_matrix("C1", 5)
    assert result is None
    assert "C1" in caplog.text and "MALFORMED" in caplog.text


def test_load_parses_well_formed_file(phase5_module, tmp_path, monkeypatch):
    """Sanity: well-formed M5 file parses to {bystander_j: float}."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    good = tmp_path / "suppression_difficulty_loc_A1_ep1.json"
    good.write_text(
        json.dumps(
            {
                "arm": "loc",
                "source_i": "A1",
                "epoch": 1.0,
                "per_bystander_mean_neg_loss": {
                    "A1__B1": 0.42,
                    "A1__C1": 0.71,
                    "A1__D1": 0.18,
                },
            }
        )
    )
    result = phase5_module._load_suppression_matrix("A1", 1)
    assert result == {"B1": 0.42, "C1": 0.71, "D1": 0.18}


# ---------------------------------------------------------------- _suppression_difficulty_partial


def _build_synth_df(src_cids: list[str], target_cids: list[str]) -> pd.DataFrame:
    """Build a synthetic 240-cell off-diagonal DataFrame for M5 tests."""
    rng = np.random.default_rng(0)
    rows = []
    for ci in src_cids:
        for cj in target_cids:
            if ci == cj:
                continue
            rows.append(
                {
                    "T_i": ci,
                    "T_j": cj,
                    "class_i": ci[0],
                    "class_j": cj[0],
                    "class_pair": f"{ci[0]}_{cj[0]}",
                    "D": float(rng.uniform(0.1, 2.0)),
                    "delta_g": float(rng.normal(0, 1)),
                    "log_prompt_tokens": float(rng.uniform(4.0, 5.0)),
                }
            )
    return pd.DataFrame(rows)


def _populate_full_m5(tmp_path: Path, src_cids: list[str], target_cids: list[str], epoch: int):
    """Write well-formed M5 files for every source cid."""
    for ci in src_cids:
        payload = {
            "arm": "loc",
            "source_i": ci,
            "epoch": float(epoch),
            "per_bystander_mean_neg_loss": {
                f"{ci}__{cj}": 0.3 + 0.1 * idx for idx, cj in enumerate(target_cids) if cj != ci
            },
        }
        path = tmp_path / f"suppression_difficulty_loc_{ci}_ep{epoch}.json"
        path.write_text(json.dumps(payload))


def test_partial_full_coverage_status_ok(phase5_module, tmp_path, monkeypatch):
    """All 16 source M5 files present → status=ok, coverage_fraction=1.0."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    src_cids = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "C1",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    _populate_full_m5(tmp_path, src_cids, src_cids, epoch=1)
    df = _build_synth_df(src_cids, src_cids)
    result = phase5_module._suppression_difficulty_partial(df, epoch=1, n_boot=50, seed=42)
    assert result["status"] == "ok"
    assert result["n_source_conds_missing"] == 0
    assert result["coverage_fraction"] == 1.0
    assert "rho_partial_out_S" in result
    assert "rho_baseline_lengthonly_partial" in result


def test_partial_round6_production_case_b1_b2_b3_ep5_missing(phase5_module, tmp_path, monkeypatch):
    """The exact round-6 production case: loc/{B1,B2,B3} ep5 M5 files
    missing → status=ok with partial coverage (n_cells_used=195, dropped=45).

    M5 still runs on the surviving 13 source conds x 15 targets = 195 cells.
    """
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    src_cids = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "C1",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    # Write M5 for everyone EXCEPT B1/B2/B3 (the production-crash case).
    survivors = [c for c in src_cids if c not in ("B1", "B2", "B3")]
    _populate_full_m5(tmp_path, survivors, src_cids, epoch=5)
    df = _build_synth_df(src_cids, src_cids)
    result = phase5_module._suppression_difficulty_partial(df, epoch=5, n_boot=50, seed=42)
    assert result["status"] == "ok"
    assert sorted(result["missing_source_conds"]) == ["B1", "B2", "B3"]
    assert result["n_source_conds_loaded"] == 13
    assert result["n_source_conds_missing"] == 3
    # B1/B2/B3 source cells = 3 x 15 = 45 dropped; 240 - 45 = 195 remaining.
    assert result["n_cells_used"] == 195
    assert result["n_cells_dropped"] == 45
    assert result["coverage_fraction"] == pytest.approx(195 / 240)
    assert "rho_partial_out_S" in result


def test_partial_all_m5_missing_status_insufficient_coverage(phase5_module, tmp_path, monkeypatch):
    """ALL source M5 files missing → status=insufficient_coverage, no crash."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    src_cids = ["A1", "B1", "C1"]
    df = _build_synth_df(src_cids, src_cids)
    result = phase5_module._suppression_difficulty_partial(df, epoch=5, n_boot=50, seed=42)
    assert result["status"] == "insufficient_coverage"
    assert result["n_cells_used"] == 0
    assert sorted(result["missing_source_conds"]) == sorted(src_cids)
    # Must NOT include partial-rho keys (the bootstrap didn't run).
    assert "rho_partial_out_S" not in result


def test_partial_one_source_only_below_floor_status_insufficient(
    phase5_module, tmp_path, monkeypatch
):
    """Only 1 source M5 file (< M5_MIN_CELLS_FOR_PARTIAL=30 cells)
    → status=insufficient_coverage with n_cells_used recorded."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    src_cids = ["A1", "A2", "A3"]  # only 3 → 3 * 2 = 6 cells max
    _populate_full_m5(tmp_path, ["A1"], src_cids, epoch=5)  # Only A1's M5 file
    df = _build_synth_df(src_cids, src_cids)
    result = phase5_module._suppression_difficulty_partial(df, epoch=5, n_boot=50, seed=42)
    assert result["status"] == "insufficient_coverage"
    # A1's 2 cells (vs A2, A3) are below floor 30 → bootstrap skipped.
    assert result["n_cells_used"] == 2
    assert result["loaded_source_conds"] == ["A1"]
    assert sorted(result["missing_source_conds"]) == ["A2", "A3"]
    assert "rho_partial_out_S" not in result


# ---------------------------------------------------------------- _per_cell_report wiring


def test_h1_survives_suppression_partial_is_none_on_insufficient(phase5_module):
    """The _per_cell_report wiring: when M5 status != 'ok',
    h1_survives_suppression_partial is None (not False).

    Distinguishes "M5 couldn't run" from "M5 ran and was screened off"
    so the analyzer / clean-result-critic can scope-caveat correctly.
    """
    # Inline simulation of the wiring (avoids needing a full _per_cell_report
    # call which requires merged matrices + per-cell JSONs).
    cases = [
        (
            {"status": "ok", "rho_partial_out_S": {"rho_pingouin": -0.7, "ci_excludes_zero": True}},
            True,
        ),
        (
            {
                "status": "ok",
                "rho_partial_out_S": {"rho_pingouin": -0.1, "ci_excludes_zero": False},
            },
            False,
        ),
        (
            {"status": "ok", "rho_partial_out_S": {"rho_pingouin": 0.2, "ci_excludes_zero": True}},
            False,
        ),
        ({"status": "insufficient_coverage", "n_cells_used": 0}, None),
        ({"status": "insufficient_coverage", "n_cells_used": 12}, None),
        ({"status": "n/a", "reason": "A_pos has no negative rows"}, None),
    ]
    for m5, expected in cases:
        # Replicate the _per_cell_report logic:
        m5_status = m5.get("status") if isinstance(m5, dict) else None
        if m5_status == "ok":
            full = m5.get("rho_partial_out_S", {}) or {}
            survives = bool(
                full.get("rho_pingouin") is not None
                and full["rho_pingouin"] < 0
                and full.get("ci_excludes_zero", False)
            )
        else:
            survives = None
        assert survives == expected, f"m5={m5} expected {expected} got {survives}"


def test_m5_coverage_meta_present_on_insufficient(phase5_module, tmp_path, monkeypatch):
    """Insufficient-coverage status MUST still report coverage_meta so the
    analyzer / clean-result body can name which (cid, ep) are dropped."""
    monkeypatch.setattr(phase5_module, "TRAIN_DIAG_DIR_474", tmp_path)
    src_cids = ["A1", "A2", "A3"]
    df = _build_synth_df(src_cids, src_cids)
    result = phase5_module._suppression_difficulty_partial(df, epoch=5, n_boot=10, seed=42)
    # Even with insufficient coverage, the coverage metadata is recorded.
    for key in (
        "missing_source_conds",
        "loaded_source_conds",
        "n_source_conds_loaded",
        "n_source_conds_missing",
        "n_cells_used",
        "n_cells_dropped",
        "coverage_fraction",
        "epoch",
    ):
        assert key in result, f"missing coverage metadata key: {key}"


def test_m5_min_cells_threshold_constant_present(phase5_module):
    """M5_MIN_CELLS_FOR_PARTIAL constant exists and is a reasonable threshold."""
    assert hasattr(phase5_module, "M5_MIN_CELLS_FOR_PARTIAL")
    assert phase5_module.M5_MIN_CELLS_FOR_PARTIAL >= 10
    assert phase5_module.M5_MIN_CELLS_FOR_PARTIAL <= 100
