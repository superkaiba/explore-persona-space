"""Gate-3 capregen-freshness backstop + breach-basis cap equality (task #2329).

Pins the two reconciler-v12 standing recommendations:

1. JUDGE-SIDE (Dispute 2 closure): ``issue2329_judge._assert_gate_rows_capregen_fresh``
   — once the frozen pre-regen breach basis exists, every staged gate row in a
   BREACHING cell must carry the raised per-row ``max_new_tokens`` (>= 2x the
   basis cap). ``_resolve_anchors_dir`` prefers the full ``anchors`` prefix on
   shard-NAME coverage, so a failed/skipped capregen shard upload keeps
   PRE-regen bytes under a covering name — the check fails LOUD naming the
   shard + cell instead of silently judging stale rows. Wired inside
   ``_gate_slice_inputs`` (the single choke point both gate-3 phases funnel
   through). Missing basis + uniform caps skips (the legitimate pre-capregen
   fresh-run ordering); missing basis + MIXED caps raises.

2. RUN-SIDE (Dispute 1 residual closure): ``issue2329_run._validate_breach_basis``
   gains the ``realized_row_caps == [max_new_tokens]`` equality — a wrong-cap
   basis is refused BEFORE the ``:3717`` freeze, so it can no longer wedge the
   campaign (previously only ``len > 1`` was refused; the wedge needed the
   basis file deleted by hand).

SANITY (hard requirement): the ACTUAL committed basis
``eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json``
(sha256 78385e71b245...) still validates and is accepted at
``--max-new-tokens 4096`` — a live capregen campaign runs against exactly
this file, so a change rejecting it is a hard stop.

CPU-only, network-free. The committed-artifact tests deliberately read the
repo copy (cone registered in ``tests/sparse_cones.txt``); everything else is
tmp_path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_judge as J  # noqa: E402
import issue2329_run as R  # noqa: E402

COMMITTED_BASIS = REPO_ROOT / "eval_results/issue_2329/cap_hit/cap_hit_report_anchors_preregen.json"


# ── fixture helpers ────────────────────────────────────────────────────


def _run_cfg(tmp_path: Path, extra: list[str] | None = None) -> R.RunConfig:
    argv = [
        "--phase",
        "cap_report",
        "--out-root",
        str(tmp_path / "out"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--tiny",
        "--tiny-layers",
        "2",
        "--tiny-hidden",
        "8",
        "--upload",
        "none",
    ] + (extra or [])
    return R.build_config(R.parse_args(argv))


def _basis(**over) -> dict:
    base = {
        "scope": "anchors",
        "partial": False,
        "max_new_tokens": 2048,
        "realized_row_caps": [2048],
        "breaching_cells": ["cellX"],
    }
    base.update(over)
    return base


def _gate_row(cell: str, ctx: str, cap: int | None, shard: str = "anchors_gate_w0.jsonl") -> dict:
    r = {
        "context_id": ctx,
        "cell": cell,
        "value_id": "v1",
        "carrier": "k1",
        "draw": 0,
        "text": "t",
        "_shard": shard,
    }
    if cap is not None:
        r["max_new_tokens"] = cap
    return r


# ── run-side: committed-artifact acceptance (the hard-stop sanity) ─────


def test_committed_basis_sha_pinned_and_validates_at_4096(tmp_path):
    """The exact committed artifact the live campaign keys off MUST still
    validate at --max-new-tokens 4096 after the equality change."""
    raw = COMMITTED_BASIS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "78385e71b245598069944b23c73e32190706811670c822dff24ba6dcb6fb8eef"
    ), "committed basis bytes changed — the frozen pod-side copy no longer matches"
    rep = json.loads(raw.decode("utf-8"))
    assert rep["max_new_tokens"] == 2048
    assert rep["realized_row_caps"] == [2048]
    assert rep["partial"] is False
    assert len(rep["breaching_cells"]) == 20
    cfg = _run_cfg(tmp_path, ["--max-new-tokens", "4096"])
    # Must NOT raise: the live capregen campaign validates exactly this file.
    R._validate_breach_basis(rep, COMMITTED_BASIS, "anchors", cfg)


def test_committed_basis_accepted_end_to_end_by_load_breach_report(tmp_path):
    """Full production path: --breach-report <committed> at 4096 loads,
    validates, and freezes byte-verbatim — Phase B / respawns re-validate the
    FROZEN copy on every load, so this is exactly the Phase-B path."""
    cfg = _run_cfg(tmp_path, ["--max-new-tokens", "4096", "--breach-report", str(COMMITTED_BASIS)])
    cfg.manifest_dir.mkdir(parents=True)
    rep, basis_path = R._load_breach_report(cfg, "anchors")
    assert basis_path == R.capregen_breach_basis_path(cfg, "anchors")
    assert basis_path.read_bytes() == COMMITTED_BASIS.read_bytes()
    assert len(rep["breaching_cells"]) == 20
    # Re-entry (frozen-basis read path) re-validates and still accepts.
    rep2, _ = R._load_breach_report(cfg, "anchors")
    assert rep2["breaching_cells"] == rep["breaching_cells"]


# ── run-side: the realized_row_caps == [base_cap] equality ─────────────


def test_wrong_cap_basis_refused_before_freeze_no_wedge(tmp_path):
    """The Dispute-1 residual closure: a wrong-cap basis is refused BEFORE
    the freeze, so a later CORRECT --breach-report is NOT wedged out by the
    byte-match (previously the wrong basis froze first and the campaign was
    wedged until the basis file was deleted)."""
    wrong = tmp_path / "wrong.json"
    # Report declares 1024 but the store's rows realized 2048 (the realistic
    # wrong---max-new-tokens shape over a per-row-cap-carrying store).
    wrong.write_text(json.dumps(_basis(max_new_tokens=1024, realized_row_caps=[2048])))
    cfg = _run_cfg(tmp_path, ["--max-new-tokens", "4096", "--breach-report", str(wrong)])
    cfg.manifest_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="realized_row_caps"):
        R._load_breach_report(cfg, "anchors")
    # NOT frozen: the campaign is not wedged.
    assert not R.capregen_breach_basis_path(cfg, "anchors").exists()
    # A later CORRECT basis is accepted on the same out-root.
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_basis()))
    cfg2 = _run_cfg(tmp_path, ["--max-new-tokens", "4096", "--breach-report", str(good)])
    rep, _ = R._load_breach_report(cfg2, "anchors")
    assert rep["breaching_cells"] == ["cellX"]


def test_realized_row_caps_equality_refusal_shapes(tmp_path):
    """Every unequal shape is refused; the committed shape passes; the len>1
    MIXED-cap refusal keeps its own message (fires first)."""
    cfg = _run_cfg(tmp_path, ["--max-new-tokens", "4096"])
    ok = _basis()
    R._validate_breach_basis(ok, tmp_path / "b.json", "anchors", cfg)  # committed shape
    for over, match in (
        ({"realized_row_caps": [1024]}, "realized_row_caps"),  # declared 2048, realized 1024
        ({"max_new_tokens": 1024}, "realized_row_caps"),  # declared 1024, realized 2048
        ({"realized_row_caps": []}, "realized_row_caps"),  # empty measurement
        ({"realized_row_caps": [2048, 4096]}, "MIXED-cap"),  # len>1 keeps its message
    ):
        with pytest.raises(RuntimeError, match=match):
            R._validate_breach_basis(_basis(**over), tmp_path / "b.json", "anchors", cfg)
    # Missing field entirely -> empty -> refused (previously passed len>1).
    rep = _basis()
    del rep["realized_row_caps"]
    with pytest.raises(RuntimeError, match="realized_row_caps"):
        R._validate_breach_basis(rep, tmp_path / "b.json", "anchors", cfg)


def test_self_consistent_legacy_wrong_cap_basis_documented_residual(tmp_path):
    """DOCUMENTED residual (reconciler v12): a SELF-consistent wrong-cap basis
    (declared 1024, realized [1024] — only producible over a legacy store
    whose rows carry no per-row cap) still validates; the regime-fingerprint
    raise remains the backstop that blocks generation. Pinned so nobody
    'fixes' it with the base_cap==2048 pin the reconciler explicitly ruled
    out (it would break smoke and escalated campaigns)."""
    cfg = _run_cfg(tmp_path, ["--max-new-tokens", "4096"])  # 4096 >= 2*1024
    rep = _basis(max_new_tokens=1024, realized_row_caps=[1024])
    R._validate_breach_basis(rep, tmp_path / "b.json", "anchors", cfg)  # no raise


# ── judge-side: the gate-3 staleness check ─────────────────────────────


def _write_basis(tmp_path: Path, **over) -> Path:
    p = tmp_path / "basis.json"
    p.write_text(json.dumps(_basis(**over)))
    return p


def test_preregen_shard_in_breach_cell_rejected(tmp_path):
    basis = _write_basis(tmp_path, breaching_cells=["cellX", "cellY"])
    rows = [
        _gate_row("cellX", "cX-1", 2048, shard="anchors_gate_w3.jsonl"),  # STALE
        _gate_row("cellZ", "cZ-1", 2048),  # non-breaching: fine at base cap
    ]
    with pytest.raises(RuntimeError, match="PRE-REGEN shard is being staged") as ei:
        J._assert_gate_rows_capregen_fresh(basis, rows)
    msg = str(ei.value)
    assert "anchors_gate_w3.jsonl" in msg  # names the offending shard
    assert "cellX" in msg  # names the offending cell


def test_breach_cell_row_missing_cap_field_rejected(tmp_path):
    """Pre-diff base-run rows carry NO per-row cap — absence in a breaching
    cell is pre-regen evidence, never a pass."""
    basis = _write_basis(tmp_path)
    rows = [_gate_row("cellX", "cX-1", None, shard="anchors_gate_w5.jsonl")]
    with pytest.raises(RuntimeError, match="PRE-REGEN"):
        J._assert_gate_rows_capregen_fresh(basis, rows)


def test_regen_rows_pass_and_nonbreaching_cells_never_checked(tmp_path):
    """Breach cells at >= 4096 pass; non-breaching cells legitimately stay at
    2048 and never trip the check (the scoping requirement)."""
    basis = _write_basis(tmp_path, breaching_cells=["cellX"])
    rows = [
        _gate_row("cellX", "cX-1", 4096),  # regenerated at exactly 2x
        _gate_row("cellX", "cX-2", 8192),  # a future escalated re-gen also passes
        _gate_row("cellY", "cY-1", 2048),  # non-breaching at base cap
        _gate_row("cellZ", "cZ-1", None),  # non-breaching legacy row (no field)
    ]
    J._assert_gate_rows_capregen_fresh(basis, rows)  # no raise


def test_no_basis_uniform_caps_skips_with_warning(tmp_path, caplog):
    """The legitimate pre-capregen fresh-run ordering: no basis exists yet,
    rows carry one uniform cap — skip, loudly."""
    rows = [_gate_row("cellX", "cX-1", 2048), _gate_row("cellY", "cY-1", 2048)]
    with caplog.at_level(logging.WARNING, logger="issue2329.judge"):
        J._assert_gate_rows_capregen_fresh(tmp_path / "absent.json", rows)
    assert any("SKIPPED" in rec.message for rec in caplog.records)


def test_no_basis_mixed_caps_raises(tmp_path):
    """Mixed per-row caps only ever come from a capregen merge — a missing
    basis there is a misconfiguration, never a fresh run."""
    rows = [_gate_row("cellX", "cX-1", 4096), _gate_row("cellY", "cY-1", 2048)]
    with pytest.raises(RuntimeError, match="MIXED per-row caps"):
        J._assert_gate_rows_capregen_fresh(tmp_path / "absent.json", rows)


def test_empty_breach_list_is_vacuous(tmp_path):
    basis = _write_basis(tmp_path, breaching_cells=[])
    J._assert_gate_rows_capregen_fresh(basis, [_gate_row("cellX", "cX-1", 2048)])


def test_judge_side_basis_validation_refusals(tmp_path):
    rows = [_gate_row("cellX", "cX-1", 4096)]
    for over, match in (
        ({"scope": "grid"}, "scope"),
        ({"postregen": True}, "POST-regen"),
        ({"partial": True}, "PARTIAL"),
        ({"realized_row_caps": [1024]}, "realized_row_caps"),
    ):
        with pytest.raises(RuntimeError, match=match):
            J._assert_gate_rows_capregen_fresh(_write_basis(tmp_path, **over), rows)
    rep = _basis()
    del rep["partial"]
    p = tmp_path / "basis.json"
    p.write_text(json.dumps(rep))
    with pytest.raises(RuntimeError, match="partial"):
        J._assert_gate_rows_capregen_fresh(p, rows)


def test_committed_basis_scopes_judge_check(tmp_path):
    """The check against the ACTUAL committed basis: a 2048 row in one of its
    real 20 breaching cells is rejected; 4096 there + 2048 in a real
    non-breaching cell passes."""
    rep = json.loads(COMMITTED_BASIS.read_text(encoding="utf-8"))
    breach = rep["breaching_cells"]
    assert len(breach) == 20
    br = breach[0]
    nb = next(c for c in sorted(rep["per_cell"]) if c not in set(breach))
    stale = [_gate_row(br, f"{br}-c0", 2048, shard="anchors_gate_w7.jsonl")]
    with pytest.raises(RuntimeError, match="PRE-REGEN"):
        J._assert_gate_rows_capregen_fresh(COMMITTED_BASIS, stale)
    fresh = [_gate_row(br, f"{br}-c0", 4096), _gate_row(nb, f"{nb}-c0", 2048)]
    J._assert_gate_rows_capregen_fresh(COMMITTED_BASIS, fresh)  # no raise


def test_gate_slice_inputs_wires_the_check(tmp_path, monkeypatch):
    """The choke-point wiring: a stale staged shard is rejected THROUGH
    _gate_slice_inputs (the shared derivation both gate-3 phases call),
    with --breach-basis threaded from the CLI. Data-loading boundaries are
    faked signature-conformantly; _gate_slice_inputs' own body runs real."""

    @dataclasses.dataclass
    class _Pair:
        a: str
        b: str

    pair = _Pair(a="cX-1", b="cY-1")
    rows = [
        _gate_row("cellX", "cX-1", 2048, shard="anchors_gate_w2.jsonl"),  # STALE breach
        _gate_row("cellY", "cY-1", 2048),
    ]

    def fake_surviving_pairs(bank_json: Path) -> list:
        return [pair]

    def fake_gate_slice_pairs(pairs: list, seed: int = 0) -> list:
        return list(pairs)

    def fake_load_anchor_rows(anchors_dir: Path) -> list[dict]:
        return rows

    monkeypatch.setattr(J, "surviving_pairs", fake_surviving_pairs)
    monkeypatch.setattr(J.BANK, "gate_slice_pairs", fake_gate_slice_pairs)
    monkeypatch.setattr(J, "load_anchor_rows", fake_load_anchor_rows)
    basis = _write_basis(tmp_path, breaching_cells=["cellX"])
    args = J.parse_args(
        [
            "--phase",
            "separation-gate",
            "--breach-basis",
            str(basis),
            "--in-root",
            str(tmp_path / "in"),
            "--work-root",
            str(tmp_path / "work"),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    )
    cfg = J.build_config(args)
    assert cfg.breach_basis == basis  # CLI threading
    with pytest.raises(RuntimeError, match="PRE-REGEN"):
        J._gate_slice_inputs(cfg)


def test_load_anchor_rows_tags_shard_provenance(tmp_path):
    """Real-body test of the modified loader: rows carry _shard = filename,
    and the r2-F3 duplicate-unit backstop still fires."""
    d = tmp_path / "anchors"
    d.mkdir()
    row = {
        "context_id": "c1",
        "cell": "cellX",
        "value_id": "v1",
        "carrier": "k1",
        "draw": 0,
        "text": "t",
    }
    (d / "anchors_gate_w0.jsonl").write_text(json.dumps(row) + "\n")
    (d / "anchors_gate_w1.jsonl").write_text(json.dumps({**row, "context_id": "c2"}) + "\n")
    rows = J.load_anchor_rows(d)
    assert {r["_shard"] for r in rows} == {"anchors_gate_w0.jsonl", "anchors_gate_w1.jsonl"}
    (d / "anchors_gate_w2.jsonl").write_text(json.dumps(row) + "\n")  # duplicate (c1, 0)
    with pytest.raises(AssertionError, match="duplicate anchor row"):
        J.load_anchor_rows(d)
