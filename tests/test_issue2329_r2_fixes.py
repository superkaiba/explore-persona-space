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
- r5 (Codex blocker ``signature-artifact-only-regressions``): one
  SIGNATURE-COMPATIBLE behavioural red test drives the guard's production
  DEFAULT system-presence derivation on the real frozen bank (pre-fix: the
  24 ``no_prefix_mismatch`` violations of the rc=23 halt; post-fix: 0
  violations + 24 recorded, asserted on report CONTENT), plus the reverse
  A-bare cases (excused when A is the system-absent side; HALT when A is
  bare but A is the system-PRESENT side).

- Crash-fix 2026-08-16 (grid rc=1): the three atomic writers
  (``_write_json_atomic`` / ``_write_jsonl_atomic`` / ``_save_pt_atomic``)
  derive PROCESS-UNIQUE temp paths, so 8 grid workers writing identical
  content to ONE shared destination (``manifests/pe_exclusions.json``) can
  no longer consume each other's temp via ``os.replace``
  (``FileNotFoundError``), and a failed write unlinks its temp (no orphan
  ``*.tmp`` residue for the out-root residue sweep to flag).

CPU-only, network-free, repo-root-path-free (tmp_path).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import multiprocessing
import os
import queue
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


def test_np_excuse_via_production_default_derivation_on_frozen_bank():
    """SIGNATURE-COMPATIBLE behavioural red test (r5, Codex blocker
    ``signature-artifact-only-regressions``): callable identically against
    the pre-fix guard — no ``system_presence=`` kwarg, no
    ``_bank_system_presence`` reference. Reconstructs the 2026-08-16 rc=23
    halt on the REAL frozen bank: all 36 ``persona_prompted`` pairs, the 12
    v2 contexts bare (the NO-PERSONA control renders with no system turn
    under the Qwen3.5 thinking-off template), v1/v3 prefixed. Pre-fix the
    guard flagged the 24 v2-involving pairs as ``no_prefix_mismatch``
    (``n_violations == 24`` — the halt); post-fix the production DEFAULT
    derivation path (``system_presence`` omitted -> frozen-bank
    ``build_contexts()``) excuses exactly those 24 — asserted on report
    CONTENT, not merely ``passed`` (a ``passed``-only assert cannot tell a
    SKIPPED pe check from one that ran). The 12 v2-as-side-A rows double as
    the reverse A-bare direction on the real derivation."""
    pairs = [p for p in R.BANK.build_pairs() if p.cell == "persona_prompted"]
    assert len(pairs) == 36
    assert sum(1 for p in pairs if "v2" in (p.value_a, p.value_b)) == 24
    angle = {"v1": 0.0, "v2": 0.8, "v3": 1.6}  # pairwise cos <= ~0.70, far below the bar
    zeros = torch.zeros((1, 2), dtype=torch.float32)
    recs: dict[str, dict] = {}
    for p in pairs:
        for cid, val in ((p.a, p.value_a), (p.b, p.value_b)):
            bare = val == "v2"  # capture_bank zero-v_pe convention for bare renders
            recs[cid] = {
                "v_pe": zeros if bare else _unit2(angle[val]),
                "v_ce": _unit2(angle[val]),
                "no_prefix": bare,
            }
    report = R.run_degeneracy_guard({"per_context": recs}, pairs, token_prefixes={})
    # BEHAVIOURAL red/green boundary: pre-fix this count is 24 (the rc=23
    # halt) and the assert fails on flag CONTENT, not on a missing symbol.
    assert report["n_violations"] == 0, sorted(
        (v["pair_id"], v["flag"]) for v in report["violations"]
    )[:5]
    assert report["passed"]
    assert report["n_pairs_checked"] == 36
    assert report["n_no_prefix_asymmetry_expected"] == 24
    rows = report["no_prefix_asymmetry_expected"]
    assert all(r["cell"] == "persona_prompted" for r in rows)
    assert all(r["no_prefix_side"] == r["system_absent_side"] for r in rows)
    # Both directions realized on the REAL derivation: v2 as side B (v1-v2
    # pairs) AND as side A (v2-v3 pairs) — the reverse A-bare direction.
    assert sum(1 for r in rows if r["no_prefix_side"] == "a") == 12
    assert sum(1 for r in rows if r["no_prefix_side"] == "b") == 12
    # Excused pairs never enter the both-sides-bare counter; the 12 v1-v3
    # pairs ran the ordinary pe/ce distinctness checks and produced nothing.
    assert report["n_no_prefix_pe_pairs"] == 0


def test_np_asymmetry_a_bare_explained_by_a_side_system_absence():
    """Reverse direction of the incident (r5): side A is the bare AND
    system-absent side. The excuse applies symmetrically (recorded, never a
    violation), and ce distinctness still binds on the excused pair."""
    pair = _np_pair()
    bank = _np_bank(pair, np_a=True, np_b=False)
    sp = {pair.a: False, pair.b: True}  # A is the system-absent side
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert report["passed"], report["violations"]
    assert report["n_violations"] == 0
    (row,) = report["no_prefix_asymmetry_expected"]
    assert row == {
        "pair_id": pair.pair_id,
        "cell": "persona_prompted",
        "no_prefix_side": "a",
        "system_absent_side": "a",
    }
    assert report["n_no_prefix_pe_pairs"] == 0
    # ce distinctness still binds in the A-bare direction.
    bank = _np_bank(pair, np_a=True, np_b=False, ce_theta=0.0)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert not report["passed"]
    assert report["violations"][0]["flag"] == "distinctness_ce"
    assert report["n_no_prefix_asymmetry_expected"] == 1


def test_np_a_bare_but_a_system_present_still_halts():
    """A bare on the system-PRESENT side (r5, A direction): the
    system-presence mismatch does NOT explain the render asymmetry — B is
    the system-absent side yet renders prefixed — so the HALT is kept."""
    pair = _np_pair()
    bank = _np_bank(pair, np_a=True, np_b=False)
    sp = {pair.a: True, pair.b: False}  # ...but A HAS the system message
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={}, system_presence=sp)
    assert not report["passed"]
    assert report["violations"][0]["flag"] == "no_prefix_mismatch"
    assert report["no_prefix_asymmetry_expected"] == []
    assert report["n_no_prefix_asymmetry_expected"] == 0


# ── crash-fix 2026-08-16 (grid rc=1): shared temp name in the atomic writers ──
#
# All 8 grid workers call ``_write_pe_exclusions`` on the SAME destination
# (``manifests/pe_exclusions.json``). Pre-fix, every one of the three atomic
# writers derived the SAME temp path (``<name>.tmp``), so one worker's
# ``os.replace`` consumed the shared temp and every later worker died
# ``FileNotFoundError`` (grid phase rc=1, 2026-08-16 05:36Z). Content is
# identical across workers by construction, so the fix is a PROCESS-UNIQUE
# temp name + unlink-on-failure — NOT locking, NOT writer election.

_HAMMER_PAYLOAD = {"scope": "grid", "criterion": "pe", "rows": [1, 2, 3]}
_HAMMER_PROCS = 8
_HAMMER_ITERS = 50
_HAMMER_ROUNDS = 3


def _json_hammer_worker(dest: str, barrier, errq) -> None:
    """Forked child: hammer ``_write_json_atomic`` on ONE shared destination."""
    barrier.wait(timeout=60)
    try:
        for _ in range(_HAMMER_ITERS):
            R._write_json_atomic(Path(dest), _HAMMER_PAYLOAD)
    except BaseException as e:  # pragma: no cover - exercised pre-fix only
        errq.put(f"pid={os.getpid()} {type(e).__name__}: {e}")
        raise SystemExit(1) from e


def test_write_json_atomic_concurrent_same_destination_all_workers_succeed(tmp_path):
    """BEHAVIOURAL reproduction of the grid crash: 8 real (forked) processes
    x 50 same-destination writes x 3 barrier-synchronized rounds. Every
    process must exit 0 (pre-fix: >=1 worker dies ``FileNotFoundError``
    because a sibling's ``os.replace`` consumed the shared ``.tmp``), the
    final file must parse to the payload, and no ``*.tmp``-shaped residue
    may remain (the upload-verifier residue-sweep surface)."""
    ctx = multiprocessing.get_context("fork")
    dest = tmp_path / "manifests" / "pe_exclusions.json"
    for round_idx in range(_HAMMER_ROUNDS):
        barrier = ctx.Barrier(_HAMMER_PROCS)
        errq = ctx.Queue()
        procs = [
            ctx.Process(target=_json_hammer_worker, args=(str(dest), barrier, errq))
            for _ in range(_HAMMER_PROCS)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
        for p in procs:  # pragma: no cover - hang guard
            if p.is_alive():
                p.terminate()
        errors = []
        while True:
            try:
                errors.append(errq.get_nowait())
            except queue.Empty:
                break
        exitcodes = [p.exitcode for p in procs]
        assert exitcodes == [0] * _HAMMER_PROCS, (round_idx, exitcodes, errors)
        assert json.loads(dest.read_text()) == _HAMMER_PAYLOAD, round_idx
    residue = [f.name for f in dest.parent.iterdir() if ".tmp" in f.name]
    assert residue == [], residue


def test_atomic_writers_unlink_tmp_on_failure_no_orphan_residue(tmp_path, monkeypatch):
    """A failed write must not strand ``*.tmp`` residue in the out-root
    (orphan temps are exactly what the out-root residue sweep flags).
    BEHAVIOURAL pre-fix red: no cleanup existed, so the temp survived a
    raising ``os.replace`` (json/jsonl) and a mid-write ``torch.save``
    failure (pt)."""

    def _boom(src, dst):
        raise OSError("simulated replace failure")

    with monkeypatch.context() as m:
        m.setattr(R.os, "replace", _boom)
        with pytest.raises(OSError, match="simulated replace failure"):
            R._write_json_atomic(tmp_path / "a.json", {"x": 1})
        with pytest.raises(OSError, match="simulated replace failure"):
            R._write_jsonl_atomic(tmp_path / "b.jsonl", [{"x": 1}])
    # pt writer: a REAL mid-write failure — a lambda is unpicklable, and
    # torch.save dies with the temp file already created on disk. The
    # message is CPython-version-dependent: 3.11 raises "Can't pickle local
    # object", 3.12 "Can't get local object" (r20: the worktree venv runs
    # 3.12 and the [Pp]ickle-only match false-failed on a healthy writer).
    with pytest.raises(Exception, match=r"[Pp]ickle|local object"):
        R._save_pt_atomic(tmp_path / "c.pt", lambda: 1)
    residue = [f.name for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert residue == [], residue
    # Success path leaves no residue either, and the errors above propagated
    # (fail-loud preserved — cleanup must never swallow the exception).
    R._write_json_atomic(tmp_path / "a.json", {"x": 1})
    R._write_jsonl_atomic(tmp_path / "b.jsonl", [{"x": 1}])
    R._save_pt_atomic(tmp_path / "c.pt", torch.tensor([1.0]))
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    residue = [f.name for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert residue == [], residue


def test_atomic_replace_cleanup_failure_does_not_mask_original_exception(
    tmp_path, monkeypatch, caplog
):
    """r3 finding 1 (``cleanup-can-mask-original``): when the write/replace
    fails AND the best-effort temp unlink ALSO fails (PermissionError, a
    non-ENOENT OSError), the ORIGINAL write/replace exception must escape
    unchanged with its traceback intact — the SECONDARY cleanup error is
    suppressed (logged at warning), never propagated in its place.
    Pre-fix red (BEHAVIOURAL): the unlink sat bare inside the
    ``except BaseException`` handler, so its PermissionError displaced the
    original OSError before the bare ``raise`` was reached.
    r6 Minor pin (BEHAVIOURAL): the suppressed-cleanup warning carries the
    CLEANUP exception's detail (errno/strerror via ``str(exc)``), so
    EACCES vs EROFS vs EIO is recoverable from the log — pre-fix the
    message held only the temp path."""

    def _boom_replace(src, dst):
        raise OSError("simulated replace failure")

    def _boom_unlink(self, missing_ok=False):  # mirrors Path.unlink's signature
        raise PermissionError("simulated unlink failure")

    with monkeypatch.context() as m:
        m.setattr(R.os, "replace", _boom_replace)
        m.setattr(Path, "unlink", _boom_unlink)
        with (
            caplog.at_level(logging.WARNING, logger="issue2329.run"),
            pytest.raises(OSError, match="simulated replace failure") as excinfo,
        ):
            R._write_json_atomic(tmp_path / "a.json", {"x": 1})
    # The ORIGINAL exception type escapes — not the cleanup's PermissionError
    # (PermissionError IS an OSError subclass, so pin the exact type too).
    assert type(excinfo.value) is OSError, type(excinfo.value)
    # Traceback intact: the original raise site (the patched replace) is on it.
    assert any(t.name == "_boom_replace" for t in excinfo.traceback), [
        t.name for t in excinfo.traceback
    ]
    # The warning fired exactly once and carries the cleanup exception's
    # detail (not just the temp path).
    cleanup_warnings = [
        rec
        for rec in caplog.records
        if rec.name == "issue2329.run" and "cleanup unlink of" in rec.getMessage()
    ]
    assert len(cleanup_warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "simulated unlink failure" in cleanup_warnings[0].getMessage(), cleanup_warnings[
        0
    ].getMessage()


def test_atomic_writer_tmp_paths_are_process_unique_for_all_three(tmp_path, monkeypatch):
    """The temp name embeds the writer's pid (process-unique across the
    8-way grid fan-out) for ALL THREE writers — json (exercised concurrently
    above) plus the jsonl/pt siblings that are one grid-shape change away
    from the identical crash. Pre-fix red for all three (shared
    ``<name>.tmp``, no pid). Per-call uuid uniqueness is pinned on the SAME
    destination (r3 finding 2, ``uuid-uniqueness-pin-vacuous``): two writes
    to ONE destination in ONE process share basename + pid, so their temp
    names can differ ONLY via a genuinely per-call uuid draw — a
    module-scope/hoisted constant uuid makes them collide and fails the
    assertion (verified empirically against a constant-uuid scratch
    variant). Coverage note: jsonl/pt get this naming pin rather than a
    full concurrent hammer to keep suite runtime modest; all three writers
    share one temp-derivation code path, which the hammer above exercises
    under real cross-process contention."""
    captured: list[str] = []
    real_replace = os.replace

    def _recording_replace(src, dst):
        captured.append(Path(src).name)
        return real_replace(src, dst)

    with monkeypatch.context() as m:
        m.setattr(R.os, "replace", _recording_replace)
        R._write_json_atomic(tmp_path / "a.json", {"x": 1})
        R._write_jsonl_atomic(tmp_path / "b.jsonl", [{"x": 1}])
        R._save_pt_atomic(tmp_path / "c.pt", torch.tensor([1.0]))
        # SAME destination as the first write, same process, back-to-back:
        # the only component that can vary is the per-call uuid fragment.
        R._write_json_atomic(tmp_path / "a.json", {"x": 2})
    assert len(captured) == 4, captured
    pid_token = f".{os.getpid()}."
    for name in captured:
        assert pid_token in name and name.endswith(".tmp"), (name, pid_token)
    # Per-call uuid uniqueness, pinned on the SAME destination basename.
    first_a, second_a = captured[0], captured[3]
    assert first_a.startswith("a.json.") and second_a.startswith("a.json."), captured
    assert first_a != second_a, (first_a, second_a)
