"""Round-2 code-review fix pins for issue #2329 (task #2329).

One test class per binding round-1 finding that added a permanent invariant:

- F3 (stale prior-width shards): ``_sweep_stale_width_shards`` quarantines
  every prior-width anchors/va_anchors/anchor_margin shard AND its done
  record on a synthetic 8->4 reshard fixture (fails pre-fix: no sweep
  existed), and ``judge.load_anchor_rows`` fails LOUD on a planted duplicate
  ``(context_id, draw)`` unit (fails pre-fix: silent concatenation).
- F1 (dashboards schema): ``_dropped_note`` reads the DICT-shaped
  ``token_identity.per_cell`` row the frozen ``bank_manifest_2329`` ships
  verbatim (fails pre-fix: ``TypeError: '>' not supported`` on the first
  rendered cell).
- F6 (silent read-layer fallback): ``_read_layer_index`` raises on a stale
  28-layer parent-model store instead of silently reading the last layer,
  keeps the tiny-store convention, and pins one layer registry across shards.
- F9 (two-write ordering): AST pin that grid + stage-2 persist the text-only
  shard BEFORE the capture reduce (the anchors #779 pattern).
- CC2: ``_finite_or_none`` never lets a bare NaN token into per-draw JSONL.

CPU-only, network-free, repo-root-path-free (tmp_path).
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_analysis as A  # noqa: E402
import issue2329_dashboards as D  # noqa: E402
import issue2329_judge as J  # noqa: E402
import issue2329_run as R  # noqa: E402

# ── F3: stale prior-width shard sweep ─────────────────────────────────


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_stale_width_sweep_quarantines_prior_width_shards_and_done_records(tmp_path):
    """8->4 reshard fixture: w4/w5 artifacts (+ done records) quarantined,
    surviving-worker files and non-width-sharded decoys untouched."""
    out_root = tmp_path / "out"
    anchors = out_root / "anchors"
    margin = out_root / "margin"
    manifests = out_root / "manifests"
    stale = [
        anchors / "anchors_gate_w4.jsonl",
        anchors / "anchors_rest_w5.jsonl",
        anchors / "va_anchors_gate_w4.pt",
        anchors / "va_anchors_rest_w5.pt",
        margin / "anchor_margin_w4.jsonl",
        manifests / "anchors_gate_w4_done.json",
        manifests / "anchors_rest_w5_done.json",
        manifests / "margin_anchors_w4_done.json",
    ]
    kept = [
        anchors / "anchors_gate_w0.jsonl",
        anchors / "va_anchors_rest_w3.pt",
        margin / "anchor_margin_w1.jsonl",
        margin / "shard_cell__ce__steered.jsonl",  # block margin: never width-swept
        manifests / "anchors_gate_w0_done.json",
        manifests / "bank_done.json",  # non-sharded manifest: untouched
        anchors / "shard_foo_w9.jsonl",  # head not in the stem allowlist
    ]
    for p in stale + kept:
        _touch(p)

    moved = R._sweep_stale_width_shards(anchors, margin, manifests, out_root, num_workers=4)

    assert moved == len(stale)
    qroot = out_root / "stale_width_quarantine"
    for p in stale:
        assert not p.exists(), f"stale shard survived the sweep: {p}"
        quarantined = list((qroot / p.parent.name).glob(f"{p.name}.stale-*"))
        assert quarantined, f"stale shard not quarantined (deleted?): {p.name}"
    for p in kept:
        assert p.exists(), f"sweep removed a surviving-width / decoy file: {p}"
    # Idempotent: a second sweep finds nothing.
    assert R._sweep_stale_width_shards(anchors, margin, manifests, out_root, num_workers=4) == 0


def test_shard_worker_index_allowlist():
    assert R._shard_worker_index("anchors_gate_w7.jsonl") == 7
    assert R._shard_worker_index("va_anchors_rest_w0.pt") == 0
    assert R._shard_worker_index("anchor_margin_w12.jsonl") == 12
    assert R._shard_worker_index("margin_anchors_w3_done.json") == 3
    assert R._shard_worker_index("shard_cell__ce__steered.jsonl") is None
    assert R._shard_worker_index("shard_foo_w9.jsonl") is None  # head not allowlisted
    assert R._shard_worker_index("bank_done.json") is None
    assert R._shard_worker_index("anchors_gate_wx.jsonl") is None  # non-digit index


def _anchor_row(context_id: str, draw: int) -> dict:
    return {
        "context_id": context_id,
        "cell": "c",
        "value_id": "v",
        "carrier": "k",
        "draw": draw,
        "text": "t",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_anchor_rows_duplicate_unit_fails_loud(tmp_path):
    """The F3 backstop: a stale prior-width shard's duplicated (context_id,
    draw) unit raises instead of silently entering the judge waves."""
    anchors = tmp_path / "anchors"
    _write_jsonl(anchors / "anchors_gate_w0.jsonl", [_anchor_row("A", 0), _anchor_row("A", 1)])
    _write_jsonl(anchors / "anchors_rest_w1.jsonl", [_anchor_row("B", 0)])
    assert len(J.load_anchor_rows(anchors)) == 3  # clean set loads

    _write_jsonl(anchors / "anchors_gate_w4.jsonl", [_anchor_row("A", 0)])  # stale duplicate
    with pytest.raises(AssertionError, match="duplicate anchor row"):
        J.load_anchor_rows(anchors)


# ── F1: dashboards read the DICT-shaped per_cell row ──────────────────


def _bank_with_per_cell(per_cell: dict) -> dict:
    return {"token_identity": {"per_cell": per_cell, "n_intact": 1, "n_dropped": 1}}


def test_dropped_note_reads_bank_manifest_shaped_dict_rows():
    """The frozen manifest ships build_token_identity's per-cell DICT records
    verbatim; pre-fix the truthy dict crashed the `n > 1` int compare."""
    per_cell = {
        "dropped_cell": {"n_pairs": 3, "n_intact": 1, "n_dropped": 2, "dropped": ["p1", "p2"]},
        "one_dropped": {"n_pairs": 3, "n_intact": 2, "n_dropped": 1, "dropped": ["p3"]},
        "intact_cell": {"n_pairs": 3, "n_intact": 3, "n_dropped": 0, "dropped": []},
    }
    bank = _bank_with_per_cell(per_cell)
    assert "2 pairs" in D._dropped_note(bank, "dropped_cell")
    note_one = D._dropped_note(bank, "one_dropped")
    assert "1 pair " in note_one and "pairs" not in note_one
    assert D._dropped_note(bank, "intact_cell") == ""
    assert D._dropped_note(bank, "missing_cell") == ""
    assert D._dropped_note({"token_identity": {}}, "any") == ""


# ── F6: fail-loud READ_LAYER resolution ───────────────────────────────


def test_read_layer_index_production_tiny_and_stale(tmp_path):
    shard = tmp_path / "shard_x.pt"
    # production 32-layer store: exact index of READ_LAYER (30)
    assert A._read_layer_index(list(range(32)), shard, {}) == 30
    # tiny/smoke store (--tiny-layers 4): last-layer convention survives
    assert A._read_layer_index([0, 1, 2, 3], shard, {}) == 3
    # stale 28-layer parent-model store: FAIL LOUD (pre-fix: silent layer 27)
    with pytest.raises(AssertionError, match="stale prior-model shard"):
        A._read_layer_index(list(range(28)), shard, {})


def test_read_layer_index_pins_one_registry_across_shards(tmp_path):
    registry: dict = {}
    assert A._read_layer_index(list(range(32)), tmp_path / "a.pt", registry) == 30
    with pytest.raises(AssertionError, match="inconsistent layer registry"):
        A._read_layer_index([0, 1, 2, 3], tmp_path / "b.pt", registry)


# ── F9: text persisted BEFORE the capture reduce (grid + stage-2) ─────


def _first_call_line(fn: ast.FunctionDef, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(fn)
        for f in [node.func if isinstance(node, ast.Call) else None]
        if f is not None
        and (
            (isinstance(f, ast.Name) and f.id == name)
            or (isinstance(f, ast.Attribute) and f.attr == name)
        )
    ]
    assert lines, f"{fn.name}: no call to {name}"
    return min(lines)


@pytest.mark.parametrize("fn_name", ["run_block", "run_stage2_block"])
def test_grid_and_stage2_persist_text_before_capture(fn_name):
    """r2 F9 ordering pin: the first shard-JSONL write precedes the capture
    reduce in BOTH block runners (the anchors #779 two-write pattern)."""
    tree = ast.parse((SCRIPTS / "issue2329_run.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    write_line = _first_call_line(fn, "_write_jsonl_atomic")
    capture_line = _first_call_line(fn, "capture_answer_states")
    assert write_line < capture_line, (
        f"{fn_name}: first _write_jsonl_atomic at L{write_line} does not precede "
        f"capture_answer_states at L{capture_line} — generated text would be lost "
        "on a capture crash (r2 F9)"
    )


# ── CC2: no bare NaN token in per-draw JSONL ──────────────────────────


def test_finite_or_none_maps_nan_and_inf_to_none():
    assert R._finite_or_none(0.5) == 0.5
    assert R._finite_or_none(float("nan")) is None
    assert R._finite_or_none(float("inf")) is None
    assert R._finite_or_none(-math.inf) is None
