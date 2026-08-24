"""Tests for scripts/issue2479_parity_check.py (P0 refit-equality gate, kill criterion (c))."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "issue2479_parity_check", REPO_ROOT / "scripts" / "issue2479_parity_check.py"
)
parity_check = importlib.util.module_from_spec(_SPEC)
sys.modules["issue2479_parity_check"] = parity_check
_SPEC.loader.exec_module(_SPEC and parity_check)

RUNGS = [
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
]


def _write_reference(path: Path, tol: float = 0.02) -> dict:
    ref = {
        "tolerance": tol,
        "cell": {
            "context": {"ceiling_r2": 0.24, "identity_bias_r2": -0.51, "n": 100},
            "prefix": {"ceiling_r2": 0.02, "identity_bias_r2": -1.73, "n": 100},
        },
        "ladder": {"r2": {r: 0.1 * i for i, r in enumerate(RUNGS)}},
    }
    path.write_text(json.dumps(ref))
    return ref


def _write_pilot(pilot: Path, ref: dict, *, perturb: float = 0.0, drop_rung: str | None = None):
    pilot.mkdir(parents=True, exist_ok=True)
    for arm in ("context", "prefix"):
        cell = {
            "reduced": {
                "ceiling_r2": [ref["cell"][arm]["ceiling_r2"] + perturb],
                "identity_bias_r2": [ref["cell"][arm]["identity_bias_r2"]],
                "n": ref["cell"][arm]["n"],
            }
        }
        (pilot / parity_check.CELL_TMPL.format(arm=arm)).write_text(json.dumps(cell))
    r2 = {r: [v] for r, v in ref["ladder"]["r2"].items() if r != drop_rung}
    lad = {"reduced": {parity_check.LADDER_DIRECTION: {"r2": r2}}}
    (pilot / parity_check.LADDER_NAME).write_text(json.dumps(lad))


def test_happy_path_within_tol_exits_0(tmp_path, capsys):
    ref_path = tmp_path / "ref.json"
    ref = _write_reference(ref_path)
    pilot = tmp_path / "pilot"
    _write_pilot(pilot, ref, perturb=0.019)
    rc = parity_check.main(["--pilot-dir", str(pilot), "--reference", str(ref_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "failed=0" in out
    # 4 cell metrics + 9 ladder rungs
    assert out.count("[parity] metric=") == 13


def test_over_tolerance_exits_1(tmp_path, capsys):
    ref_path = tmp_path / "ref.json"
    ref = _write_reference(ref_path)
    pilot = tmp_path / "pilot"
    _write_pilot(pilot, ref, perturb=0.05)  # > 0.02 on both ceiling_r2 metrics
    rc = parity_check.main(["--pilot-dir", str(pilot), "--reference", str(ref_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out


def test_missing_pilot_file_exits_2(tmp_path):
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = parity_check.main(["--pilot-dir", str(empty), "--reference", str(ref_path)])
    assert rc == 2


def test_missing_ladder_rung_exits_2(tmp_path):
    ref_path = tmp_path / "ref.json"
    ref = _write_reference(ref_path)
    pilot = tmp_path / "pilot"
    _write_pilot(pilot, ref, drop_rung="4_bias_refit")
    rc = parity_check.main(["--pilot-dir", str(pilot), "--reference", str(ref_path)])
    assert rc == 2


def test_n_mismatch_exits_2(tmp_path):
    ref_path = tmp_path / "ref.json"
    ref = _write_reference(ref_path)
    pilot = tmp_path / "pilot"
    _write_pilot(pilot, ref)
    # rewrite context cell with a different n
    cell = json.loads((pilot / parity_check.CELL_TMPL.format(arm="context")).read_text())
    cell["reduced"]["n"] = 42
    (pilot / parity_check.CELL_TMPL.format(arm="context")).write_text(json.dumps(cell))
    rc = parity_check.main(["--pilot-dir", str(pilot), "--reference", str(ref_path)])
    assert rc == 2


def test_stager_source_store_pins_match_issue1887_registry():
    """The stager's duplicated SOURCE_STORES pins must equal the issue1887
    I1345_VARIANT_STORE_REVS entries they were copied from (plan v4 §10).
    Parsed textually — importing the 2k-line driver is deliberate non-coupling."""
    import re

    text = (REPO_ROOT / "scripts" / "issue1887_lambda_audit.py").read_text()
    block = re.search(r"I1345_VARIANT_STORE_REVS\s*=\s*\{(.*?)\}", text, re.S).group(1)
    registry = dict(re.findall(r'"([a-z_]+)":\s*"([0-9a-f]{40})"', block))

    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "issue1345_stage_char_stories", REPO_ROOT / "scripts" / "issue1345_stage_char_stories.py"
    )
    stager = _ilu.module_from_spec(spec)
    spec.loader.exec_module(stager)
    for key, (variant, pin) in stager.SOURCE_STORES.items():
        assert registry.get(variant) == pin, (key, variant, pin, registry.get(variant))


def test_committed_reference_matches_checker_shape():
    ref_path = REPO_ROOT / "eval_results" / "issue_2479" / "parity_reference_char_helios_op.json"
    ref = json.loads(ref_path.read_text())
    assert ref["tolerance"] == pytest.approx(0.02)
    assert set(ref["ladder"]["r2"]) == set(RUNGS)
    for arm in ("context", "prefix"):
        for key in ("ceiling_r2", "identity_bias_r2", "n"):
            assert key in ref["cell"][arm]
