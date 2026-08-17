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
- Crash-fix 2026-08-16 (rc=23): ``run_degeneracy_guard`` records a one-sided
  ``no_prefix`` flag as ``no_prefix_asymmetry_expected`` (not a violation)
  IFF the frozen bank explains it — exactly one side has no system message
  and the bare side IS that side; every other np mismatch keeps the HALT.

CPU-only, network-free, repo-root-path-free (tmp_path).
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest
import torch

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
    surviving-worker files and non-width-sharded decoys untouched. r3 C1:
    the sweep is FAMILY-scoped, so both families run at their explicit width
    (F3's original purpose — remove true prior-width duplicates — preserved)."""
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

    moved = 0
    for family in ("anchors", "margin"):
        moved += R._sweep_stale_width_shards(
            anchors, margin, manifests, out_root, num_workers=4, family=family
        )

    assert moved == len(stale)
    qroot = out_root / "stale_width_quarantine"
    for p in stale:
        assert not p.exists(), f"stale shard survived the sweep: {p}"
        quarantined = list((qroot / p.parent.name).glob(f"{p.name}.stale-*"))
        assert quarantined, f"stale shard not quarantined (deleted?): {p.name}"
    for p in kept:
        assert p.exists(), f"sweep removed a surviving-width / decoy file: {p}"
    # Idempotent: a second sweep finds nothing.
    for family in ("anchors", "margin"):
        assert (
            R._sweep_stale_width_shards(
                anchors, margin, manifests, out_root, num_workers=4, family=family
            )
            == 0
        )


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


# ── gate 0b: the block-count assert reads BLOCKS, not the return tuple ──


class _Inner:
    """Duck-types ``.layers`` for the purely-structural chain walk."""

    def __init__(self, n: int) -> None:
        self.layers = list(range(n))
        self.embed_tokens = object()


class _Stub:
    def __init__(self, n: int) -> None:
        self.model = _Inner(n)


def test_resolve_decoder_blocks_returns_a_triple_not_the_block_list():
    """The trap gate 0b fell into: the helper returns
    ``(blocks, embed_tokens, depth)``, so ``len()`` over the RETURN VALUE is
    always 3 — independent of how many decoder blocks the model has."""
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    ret = _resolve_decoder_blocks(_Stub(32))
    assert len(ret) == 3, "helper contract changed — re-check every call site"
    blocks, _embed, depth = ret
    assert len(blocks) == 32 and depth == 1


def test_gate0b_unpacks_resolve_decoder_blocks_before_counting():
    """AST pin: ``_gate0b_check`` must tuple-unpack the helper and count the
    BLOCKS. A bare ``len(_resolve_decoder_blocks(model))`` is always 3, so the
    32-layer assert could never pass on any model (the gate self-blocked the
    #2329 launch until this was fixed)."""
    tree = ast.parse((SCRIPTS / "issue2329_run.py").read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_gate0b_check"
    )

    def _is_resolve_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_resolve_decoder_blocks"
        )

    bare_len = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and any(_is_resolve_call(a) for a in node.args)
    ]
    assert not bare_len, (
        f"_gate0b_check: len() applied directly to _resolve_decoder_blocks(...) at L{bare_len} "
        "— that counts the 3-tuple, not the decoder blocks"
    )
    unpacks = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and _is_resolve_call(node.value)
        and any(isinstance(t, ast.Tuple) and len(t.elts) == 3 for t in node.targets)
    ]
    assert unpacks, "_gate0b_check: no 3-tuple unpack of _resolve_decoder_blocks(...)"


# ── crash-fix 2026-08-16 (rc=23): one-sided no-prefix EXPLAINED by system absence ──
#
# The pod-side P1 bank phase halted with 24/1404 `no_prefix_mismatch`
# violations, all in `persona_prompted`: v2 is the deliberate NO-PERSONA
# control arm (`system: null` in the frozen bank) and Qwen3.5's thinking-off
# template inserts no default system turn, so all 12 v2 contexts render bare
# while v1/v3 do not. `run_degeneracy_guard` now records such pairs under
# `no_prefix_asymmetry_expected` (not violations) IFF exactly one side lacks
# a system message AND the bare side IS the system-absent side; every other
# np mismatch keeps the HALT. Fail pre-fix: the guard had no
# `system_presence` parameter and flagged every np mismatch.


def _unit2(theta: float) -> torch.Tensor:
    """(1, 2) unit vector at angle ``theta`` (cosine to ``_unit2(0)`` = cos theta)."""
    return torch.tensor([[math.cos(theta), math.sin(theta)]], dtype=torch.float32)


_DISTINCT_THETA = 1.2  # cos ~ 0.362 — far below DEGENERACY_COS_MIN


def _np_pair(cell: str = "persona_prompted") -> R.BANK.Pair2162:
    """A v1-v2 pair in ``cell`` (v2 = the no-system side in the incident)."""
    return R.BANK.Pair2162(
        pair_id=f"{cell}::v1-v2::n3",
        cell=cell,
        carrier="n3",
        value_a="v1",
        value_b="v2",
        a=f"{cell}::v1::n3",
        b=f"{cell}::v2::n3",
    )


def _np_bank(pair, *, np_a: bool = False, np_b: bool = True, ce_theta: float = _DISTINCT_THETA):
    """Two-context bank; a no-prefix side gets the capture_bank zero ``v_pe``."""
    zeros = torch.zeros((1, 2), dtype=torch.float32)
    return {
        "per_context": {
            pair.a: {
                "v_pe": zeros if np_a else _unit2(0.0),
                "v_ce": _unit2(0.0),
                "no_prefix": np_a,
            },
            pair.b: {
                "v_pe": zeros if np_b else _unit2(0.3),
                "v_ce": _unit2(ce_theta),
                "no_prefix": np_b,
            },
        }
    }


def test_np_asymmetry_explained_by_system_absence_is_not_a_violation(monkeypatch):
    """np flags mismatch AND system-presence mismatches (aligned: the bare
    side is the system-absent side) => recorded, NOT a violation, passed
    stays True. Also pins the production-default threading (``system_presence``
    omitted => derived via ``_bank_system_presence``)."""
    pair = _np_pair()
    bank = _np_bank(pair)
    sp = {pair.a: True, pair.b: False}  # v2 = the NO-PERSONA control arm
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert report["passed"], report["violations"]
    assert report["n_violations"] == 0
    assert report["n_no_prefix_asymmetry_expected"] == 1
    (row,) = report["no_prefix_asymmetry_expected"]
    assert row == {
        "pair_id": pair.pair_id,
        "cell": "persona_prompted",
        "no_prefix_side": "b",
        "system_absent_side": "b",
    }
    # These pairs never contribute to the both-sides-no-prefix counter.
    assert report["n_no_prefix_pe_pairs"] == 0
    # Production-default threading: omitted system_presence -> the frozen-bank
    # derivation is consulted (monkeypatched; the real mapping is pinned by
    # test_bank_system_presence_production_derivation).
    monkeypatch.setattr(R, "_bank_system_presence", lambda: sp)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={})
    assert report["passed"] and report["n_no_prefix_asymmetry_expected"] == 1


def test_np_asymmetry_explained_pair_still_fails_on_degenerate_ce():
    """ce distinctness still runs on an EXPLAINED pair: a degenerate ce
    (ce_cos = 1.0) must still FAIL even though the np mismatch is expected."""
    pair = _np_pair()
    bank = _np_bank(pair, ce_theta=0.0)
    sp = {pair.a: True, pair.b: False}
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert not report["passed"]
    (row,) = report["violations"]
    assert row["flag"] == "distinctness_ce"
    assert report["n_no_prefix_asymmetry_expected"] == 1  # recorded either way


@pytest.mark.parametrize("present", [True, False])
def test_np_mismatch_with_agreeing_system_presence_still_halts(present):
    """np flags mismatch while system-presence AGREES => a genuine
    render/capture defect: still a HALT violation (protective value kept)."""
    pair = _np_pair()
    bank = _np_bank(pair)
    sp = {pair.a: present, pair.b: present}
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert not report["passed"]
    (row,) = report["violations"]
    assert row["flag"] == "no_prefix_mismatch"
    assert report["no_prefix_asymmetry_expected"] == []
    assert report["n_no_prefix_asymmetry_expected"] == 0


def test_np_mismatch_misaligned_with_system_absence_still_halts():
    """System-presence mismatch does NOT excuse the np mismatch when the bare
    (no-prefix) side is the side that HAS the system message — that render
    asymmetry is unexplained and stays a HALT violation."""
    pair = _np_pair()
    bank = _np_bank(pair)  # b is the bare side...
    sp = {pair.a: False, pair.b: True}  # ...but A is the system-absent side
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert not report["passed"]
    assert report["violations"][0]["flag"] == "no_prefix_mismatch"
    assert report["no_prefix_asymmetry_expected"] == []


def test_both_sides_no_prefix_behaviour_unchanged():
    """Both-sides-bare pairs keep the pre-fix semantics: pe checks N/A via
    ``n_no_prefix_pe_pairs``, no asymmetry record, ce distinctness still
    binding — and ``system_presence`` is never consulted (empty dict OK)."""
    pair = _np_pair()
    bank = _np_bank(pair, np_a=True, np_b=True)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence={})
    assert report["passed"], report["violations"]
    assert report["n_no_prefix_pe_pairs"] == 1
    assert report["no_prefix_asymmetry_expected"] == []
    bank = _np_bank(pair, np_a=True, np_b=True, ce_theta=0.0)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence={})
    assert not report["passed"]
    assert report["violations"][0]["flag"] == "distinctness_ce"


def test_bank_system_presence_production_derivation():
    """The production default maps the frozen bank exactly: all 12
    ``persona_prompted`` v2 contexts (the NO-PERSONA control) have no system
    message; all 24 v1/v3 contexts do. Whole-bank coverage (1,404 cids)."""
    sp = R._bank_system_presence()
    assert len(sp) == 1404
    pp = {cid: v for cid, v in sp.items() if cid.startswith("persona_prompted::")}
    assert len(pp) == 36
    v2 = {cid for cid in pp if "::v2::" in cid}
    assert len(v2) == 12
    assert all(not pp[cid] for cid in v2)
    assert all(pp[cid] for cid in set(pp) - v2)
