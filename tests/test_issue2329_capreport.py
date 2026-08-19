"""Cap-hit report + cell-restricted capregen pins for issue #2329 (task #2329).

The plan registers max_new_tokens=2048 with a cap-hit > 2%/cell re-gen
trigger (Source #2162) that previously had NO enforcing code; the anchors
gate slice realized 18/36 cells breaching. These tests pin:

- per-cell + per-(cell, value) arithmetic against a fixture with a KNOWN
  breach set, incl. the exactly-at-2.0% boundary (STRICT ``>`` semantics —
  2.0% does NOT fire) and the legitimate empty breach list (reported, never
  coerced);
- incremental/partial labeling: missing expected shards, text-only
  (pre-capture) pending shards, and an underivable expected set each mark
  the report ``partial`` with a reason — a partial read can never pass as
  final — and the empty / all-pending selections RAISE;
- realized-cap recording on regenerated rows (``_enrich_rows_with_capture``)
  and on merged anchors shards (``_merge_anchor_capregen``: kept rows
  backfilled with the BASE cap, regen rows at the raised cap, va-store
  index/tensor/empty-row alignment, capregen sub-record on the done record);
- the regime_fp / done-file interaction: ``max_new_tokens`` is INSIDE
  ``regime_fingerprint`` (a raised cap is a different resume regime);
  ``block_is_done`` keeps the #722 r3 hard refusal; ``_capregen_block_done``
  preserves that refusal, treats a pre-regen done record as PENDING (a stale
  done-file can never let a breaching block skip re-generation), and refuses
  mixed raised caps; ``_load_breach_report`` refuses partial reports and
  non-raised caps.

CPU-only, network-free, repo-root-path-free (tmp_path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_run as R  # noqa: E402

# ── fixture helpers ────────────────────────────────────────────────────


def _row(cell: str, value_id: str, i: int, hit: bool, cap: int | None = None) -> dict:
    r = {
        "context_id": f"{cell}-{value_id}-c{i}",
        "cell": cell,
        "value_id": value_id,
        "carrier": "k1",
        "draw": 0,
        "seed": 42,
        "temperature": 1.0,
        "gate_slice": True,
        "text": "t",
        "n_completion_tokens": 2048 if hit else 100 + i,
        "cap_hit": hit,
        "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
    }
    if cap is not None:
        r["max_new_tokens"] = cap
    return r


def _cell_rows(cell: str, value_id: str, n: int, hits: int) -> list[dict]:
    return [_row(cell, value_id, i, i < hits) for i in range(n)]


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cfg(tmp_path: Path, extra: list[str] | None = None) -> R.RunConfig:
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


# ── (1) per-cell / per-(cell, value) arithmetic, strict-> boundary ─────


def test_per_cell_arithmetic_known_breach_set(tmp_path):
    """cellA 2/50 = 4% (breach), cellB 1/50 = 2.0% EXACTLY (NOT a breach —
    strict >), cellC 0/40 — the breach set is exactly {cellA}."""
    s1 = tmp_path / "shard_a.jsonl"
    s2 = tmp_path / "shard_b.jsonl"
    _write_shard(
        s1,
        _cell_rows("cellA", "v1", 25, 1)
        + _cell_rows("cellA", "v3", 25, 1)
        + _cell_rows("cellB", "v1", 50, 1),
    )
    _write_shard(s2, _cell_rows("cellC", "v2", 40, 0))
    rep = R.compute_cap_hit_report(
        [s1, s2],
        2048,
        scope="anchors",
        expected_shards={"shard_a.jsonl", "shard_b.jsonl"},
    )
    assert rep["n_rows"] == 140
    assert rep["cap_hit_rows"] == 3
    assert rep["cap_hit_pct"] == pytest.approx(100.0 * 3 / 140)
    assert rep["per_cell"]["cellA"] == {
        "n_rows": 50,
        "cap_hit_rows": 2,
        "cap_hit_pct": pytest.approx(4.0),
        "breach": True,
        "realized_caps_by_batch": {"gate": [2048]},  # legacy rows backfilled
    }
    assert rep["per_cell"]["cellB"]["cap_hit_pct"] == pytest.approx(2.0)
    assert rep["per_cell"]["cellB"]["breach"] is False  # STRICT >: 2.0% does NOT fire
    assert rep["per_cell"]["cellC"]["breach"] is False
    assert rep["breaching_cells"] == ["cellA"]
    assert rep["trigger_fired"] is True
    assert rep["partial"] is False
    assert rep["partial_reason"] is None
    assert rep["missing_shards"] == []
    assert rep["pending_capture_shards"] == []
    # Parent-artifact field parity (grid_caphit_aggregate.json shape precedent).
    for key in (
        "derived_from",
        "derived_from_sha256",
        "derivation",
        "n_rows",
        "cap_hit_rows",
        "cap_hit_frac",
        "cap_hit_pct",
        "pre_registered_regen_trigger_pct",
        "trigger_fired",
    ):
        assert key in rep, key
    assert rep["pre_registered_regen_trigger_pct"] == 2.0
    assert rep["max_new_tokens"] == 2048
    assert rep["realized_row_caps"] == [2048]  # legacy rows backfilled from the arg


def test_empty_breach_list_is_reported_never_coerced(tmp_path):
    s = tmp_path / "shard_a.jsonl"
    _write_shard(s, _cell_rows("cellB", "v1", 50, 1))  # exactly 2.0%
    rep = R.compute_cap_hit_report([s], 2048, scope="anchors", expected_shards={"shard_a.jsonl"})
    assert rep["breaching_cells"] == []
    assert rep["trigger_fired"] is False
    assert rep["partial"] is False


def test_per_cell_value_breakdown_and_max_spread(tmp_path):
    """Within-cell value asymmetry (the verbosity v1 0% / v3 50% shape):
    cellA v1 0/25 vs v3 4/25 => spread 16.0 pct points."""
    s = tmp_path / "shard_a.jsonl"
    _write_shard(
        s,
        _cell_rows("cellA", "v1", 25, 0)
        + _cell_rows("cellA", "v3", 25, 4)
        + _cell_rows("cellD", "v1", 10, 0),
    )
    rep = R.compute_cap_hit_report([s], 2048, scope="anchors", expected_shards={"shard_a.jsonl"})
    assert rep["per_cell_value"]["cellA"]["v1"]["cap_hit_pct"] == pytest.approx(0.0)
    assert rep["per_cell_value"]["cellA"]["v3"] == {
        "n_rows": 25,
        "cap_hit_rows": 4,
        "cap_hit_pct": pytest.approx(16.0),
    }
    assert rep["max_value_spread"] == {
        "cell": "cellA",
        "min_pct": pytest.approx(0.0),
        "max_pct": pytest.approx(16.0),
        "spread_pct": pytest.approx(16.0),
    }
    assert rep["value_key_fields"] == ["value_id"]


def test_per_cell_realized_caps_by_batch_distinguishes_mixed_store(tmp_path):
    """After a batch-scoped capregen the store is legitimately mixed: a
    COMPLETED gate-slice re-gen shows [4096] in the breaching cell's gate
    entry while rest stays [2048]; a HALF-DONE gate re-gen shows
    [2048, 4096] — machine-distinguishable per (cell, batch)."""
    done_gate = [dict(_row("cellA", "v1", i, False, cap=4096)) for i in range(4)]
    rest_legacy = [dict(_row("cellA", "v1", 10 + i, False)) for i in range(4)]
    for r in rest_legacy:
        r["gate_slice"] = False
    half_gate = [dict(_row("cellB", "v1", i, False, cap=4096 if i % 2 else 2048)) for i in range(4)]
    s = tmp_path / "shard_a.jsonl"
    _write_shard(s, done_gate + rest_legacy + half_gate)
    rep = R.compute_cap_hit_report([s], 2048, scope="anchors", expected_shards={"shard_a.jsonl"})
    assert rep["per_cell"]["cellA"]["realized_caps_by_batch"] == {
        "gate": [4096],
        "rest": [2048],
    }
    assert rep["per_cell"]["cellB"]["realized_caps_by_batch"] == {"gate": [2048, 4096]}
    assert rep["realized_row_caps"] == [2048, 4096]


def test_capregen_batch_flag_requirements(tmp_path):
    """anchors capregen REQUIRES a single batch (gate|rest — Phase A vs B,
    never collapsed); grid capregen REFUSES the flag (no batch dimension)."""
    cfg_none = _cfg(tmp_path, ["--capregen-scope", "anchors"])
    with pytest.raises(RuntimeError, match=r"capregen-batch gate\|rest is required"):
        R.phase_capregen_anchors(cfg_none)
    cfg_grid = _cfg(tmp_path, ["--capregen-scope", "grid", "--capregen-batch", "gate"])
    with pytest.raises(RuntimeError, match="anchors only"):
        R.phase_capregen_grid(cfg_grid)


def test_grid_rows_key_on_value_a(tmp_path):
    rows = _cell_rows("cellG", "vX", 10, 1)
    for r in rows:
        r["value_a"] = r.pop("value_id")
    s = tmp_path / "shard_g.jsonl"
    _write_shard(s, rows)
    rep = R.compute_cap_hit_report([s], 2048, scope="grid", expected_shards={"shard_g.jsonl"})
    assert rep["value_key_fields"] == ["value_a"]
    assert rep["per_cell_value"]["cellG"]["vX"]["n_rows"] == 10


# ── (2) incremental / partial labeling ─────────────────────────────────


def test_partial_missing_and_pending_shards(tmp_path):
    enriched = tmp_path / "shard_a.jsonl"
    pending = tmp_path / "shard_b.jsonl"
    _write_shard(enriched, _cell_rows("cellA", "v1", 20, 5))
    text_only = [
        {k: v for k, v in r.items() if k not in ("cap_hit", "cap_hit_basis", "n_completion_tokens")}
        for r in _cell_rows("cellB", "v1", 20, 0)
    ]
    _write_shard(pending, text_only)
    rep = R.compute_cap_hit_report(
        [enriched, pending],
        2048,
        scope="grid",
        expected_shards={"shard_a.jsonl", "shard_b.jsonl", "shard_c.jsonl"},
    )
    assert rep["partial"] is True
    assert rep["n_rows"] == 20  # pending shard EXCLUDED from counts, not zero-counted
    assert rep["pending_capture_shards"] == ["shard_b.jsonl"]
    assert rep["missing_shards"] == ["shard_c.jsonl"]
    assert any("pending" in why for why in rep["partial_reason"])
    assert any("missing" in why for why in rep["partial_reason"])
    assert rep["covered_shards"] == ["shard_a.jsonl"]


def test_partial_when_expected_set_unavailable(tmp_path):
    s = tmp_path / "shard_a.jsonl"
    _write_shard(s, _cell_rows("cellA", "v1", 20, 5))
    rep = R.compute_cap_hit_report(
        [s],
        2048,
        scope="grid",
        expected_shards=None,
        expected_unavailable_reason="bank absent — expected block set underivable",
    )
    assert rep["partial"] is True
    assert rep["partial_reason"] == ["bank absent — expected block set underivable"]
    assert rep["missing_shards"] is None


def test_no_shards_and_all_pending_raise(tmp_path):
    with pytest.raises(RuntimeError, match="no rollout shards"):
        R.compute_cap_hit_report([], 2048, scope="grid", expected_shards=None)
    pending = tmp_path / "shard_b.jsonl"
    text_only = [
        {k: v for k, v in r.items() if k not in ("cap_hit", "cap_hit_basis", "n_completion_tokens")}
        for r in _cell_rows("cellB", "v1", 5, 0)
    ]
    _write_shard(pending, text_only)
    with pytest.raises(RuntimeError, match="text-only"):
        R.compute_cap_hit_report([pending], 2048, scope="grid", expected_shards=None)


# ── (3) realized-cap recording on (re)generated rows ───────────────────


def test_enrich_rows_records_realized_cap():
    rows = [{"text": "a"}, {"text": "b"}]
    states = {"n_completion_tokens": [4096, 128]}
    R._enrich_rows_with_capture(rows, states, 4096)
    assert rows[0]["max_new_tokens"] == 4096 and rows[1]["max_new_tokens"] == 4096
    assert rows[0]["cap_hit"] is True and rows[1]["cap_hit"] is False
    assert rows[0]["n_completion_tokens"] == 4096


def test_merge_anchor_capregen_caps_alignment_and_done_record(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True)
    cfg.manifest_dir.mkdir(parents=True)
    # Old shard: cellX (breaching) ctx x1 draws 0/1 at rows 0-1; cellY ctx y1
    # draws 0/1 at rows 2-3. Legacy rows carry NO max_new_tokens field.
    old_rows = [
        {**_row("cellX", "v1", 0, True), "context_id": "x1", "draw": 0},
        {**_row("cellX", "v1", 1, False), "context_id": "x1", "draw": 1},
        {**_row("cellY", "v1", 2, False), "context_id": "y1", "draw": 0},
        {**_row("cellY", "v1", 3, False), "context_id": "y1", "draw": 1},
    ]
    jsonl = cfg.anchors_dir / "anchors_gate_w0.jsonl"
    _write_shard(jsonl, old_rows)
    orig_bytes = jsonl.read_bytes()  # FIX 2: the pre-regen capture to preserve
    va = torch.arange(4 * 2 * 8, dtype=torch.float16).reshape(4, 2, 8)
    torch.save(
        {
            "layers": [0, 1],
            "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in old_rows],
            "va_span": va,
            "pooling": {"va_span": "mean over completion tokens (plan §4.4 span-mean V_a)"},
            "empty_rows": [0, 2],  # 0 = dropped cellX row; 2 = kept cellY row
            "repro": {},
        },
        cfg.anchors_dir / "va_anchors_gate_w0.pt",
    )
    done_rec = {
        "regime_fp": "basefp",
        "batch": "gate",
        "worker_index": 0,
        "num_workers": 1,
        "n_contexts": 2,
        "draws": 2,
        "n_rows": 4,
        "n_cap_hit": 1,
        "n_empty": 2,
    }
    regen_rows = [
        {**_row("cellX", "v1", 0, False, cap=4096), "context_id": "x1", "draw": 0},
        {**_row("cellX", "v1", 1, True, cap=4096), "context_id": "x1", "draw": 1},
    ]
    regen_states = {
        "va_span": torch.ones(2, 2, 8, dtype=torch.float16),
        "pooling": {"va_span": "mean over completion tokens (plan §4.4 span-mean V_a)"},
        "empty_rows": [1],
    }
    capregen_record = {
        "cells": ["cellX"],
        "max_new_tokens": 4096,
        "base_max_new_tokens": 2048,
        "n_rows_regen": 2,
    }
    new_done = R._merge_anchor_capregen(
        cfg,
        "gate",
        2048,
        {"cellX"},
        {"x1": "cellX", "y1": "cellY"},
        regen_rows,
        regen_states,
        done_rec,
        capregen_record,
    )
    merged = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert [r["cell"] for r in merged] == ["cellY", "cellY", "cellX", "cellX"]
    # Kept rows backfilled with the BASE cap; regen rows carry the raised cap.
    assert [r["max_new_tokens"] for r in merged] == [2048, 2048, 4096, 4096]
    store = torch.load(cfg.anchors_dir / "va_anchors_gate_w0.pt", weights_only=False)
    assert [e["context_id"] for e in store["index"]] == ["y1", "y1", "x1", "x1"]
    assert store["va_span"].shape == (4, 2, 8)
    assert torch.equal(store["va_span"][:2], va[2:4])  # kept tensor rows preserved
    assert torch.equal(store["va_span"][2:], regen_states["va_span"])
    # empty_rows: kept old empty (was 2 -> now 0); regen empty 1 -> offset 2+1=3.
    assert store["empty_rows"] == [0, 3]
    assert new_done["n_rows"] == 4
    assert new_done["max_new_tokens"] == 2048  # the shard's BASE regime cap
    assert new_done["capregen"] == capregen_record
    assert new_done["n_cap_hit"] == 1  # one regen row still hits 4096
    on_disk = json.loads((cfg.manifest_dir / "anchors_gate_w0_done.json").read_text())
    assert on_disk["capregen"]["max_new_tokens"] == 4096
    # FIX 2: the merge preserved the ENTIRE pre-regen shard (superseded
    # breach-cell rows included) byte-identically BEFORE overwriting, and
    # recorded the preservation location in the done sub-record so a consumer
    # and the upload-verifier can find it without HF revision history.
    pre = R.preregen_superseded_dir(cfg, "anchors") / "anchors_gate_w0.jsonl"
    assert pre.read_bytes() == orig_bytes
    assert (
        on_disk["capregen"]["preregen_superseded"]["local"]
        == pre.relative_to(cfg.out_root).as_posix()
    )
    assert "preregen_superseded" in on_disk["capregen"]["preregen_superseded"]["hf_prefix"]


def test_merge_refuses_context_set_drift(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True)
    cfg.manifest_dir.mkdir(parents=True)
    _write_shard(
        cfg.anchors_dir / "anchors_gate_w0.jsonl",
        [{**_row("cellX", "v1", 0, False), "context_id": "x1", "draw": 0}],
    )
    with pytest.raises(RuntimeError, match="regenerated context set"):
        R._merge_anchor_capregen(
            cfg,
            "gate",
            2048,
            {"cellX"},
            {"x1": "cellX", "x2": "cellX"},
            [{**_row("cellX", "v1", 0, False, cap=4096), "context_id": "x2", "draw": 0}],
            {"va_span": torch.ones(1, 2, 8), "pooling": {"va_span": "p"}, "empty_rows": []},
            {"draws": 1},
            {},
        )


# ── (4) regime_fp / done-file interaction ──────────────────────────────


def test_regime_fp_embeds_cap_and_block_is_done_hard_refusal(tmp_path):
    cfg_base = _cfg(tmp_path)
    cfg_raised = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    fp_base = R.regime_fingerprint(cfg_base, "sha")
    fp_raised = R.regime_fingerprint(cfg_raised, "sha")
    assert fp_base != fp_raised  # a raised cap IS a different resume regime
    block = R.Block("cellA", "ce", "steered", ("p1",))
    done = R.block_done_path(cfg_base.out_root, block)
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps({"key": block.key, "regime_fp": fp_base}))
    assert R.block_is_done(cfg_base.out_root, block, fp_base) is True
    # #722 r3 hard refusal PRESERVED: a raised-cap scan RAISES, never skips.
    with pytest.raises(RuntimeError, match="refusing to resume across"):
        R.block_is_done(cfg_base.out_root, block, fp_raised)


def test_capregen_block_done_predicate(tmp_path):
    out_root = tmp_path / "out"
    block = R.Block("cellA", "ce", "steered", ("p1",))
    done = R.block_done_path(out_root, block)
    # (i) missing done file: pending.
    assert R._capregen_block_done(out_root, block, "basefp", 4096) is False
    done.parent.mkdir(parents=True)
    # (ii) pre-regen done record at the BASE fp: STALE for regen purposes —
    # it can never let a breaching block skip re-generation.
    done.write_text(json.dumps({"key": block.key, "regime_fp": "basefp"}))
    assert R._capregen_block_done(out_root, block, "basefp", 4096) is False
    # (iii) regenerated at exactly the target cap: done.
    done.write_text(
        json.dumps({"key": block.key, "regime_fp": "basefp", "capregen": {"max_new_tokens": 4096}})
    )
    assert R._capregen_block_done(out_root, block, "basefp", 4096) is True
    # (iv) foreign regime: the hard refusal, preserved.
    done.write_text(json.dumps({"key": block.key, "regime_fp": "otherfp"}))
    with pytest.raises(RuntimeError, match="refusing to re-gen across regimes"):
        R._capregen_block_done(out_root, block, "basefp", 4096)
    # (v) regenerated at a DIFFERENT raised cap: refuse, never mix.
    done.write_text(
        json.dumps({"key": block.key, "regime_fp": "basefp", "capregen": {"max_new_tokens": 3072}})
    )
    with pytest.raises(RuntimeError, match="refusing to mix raised caps"):
        R._capregen_block_done(out_root, block, "basefp", 4096)


def test_load_breach_report_refusals(tmp_path):
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.manifest_dir.mkdir(parents=True)
    path = R.cap_hit_report_path(cfg, "grid")
    ok = {
        "scope": "grid",
        "partial": False,
        "max_new_tokens": 2048,
        "realized_row_caps": [2048],
        "breaching_cells": ["cellA"],
    }
    path.write_text(json.dumps({**ok, "partial": True, "partial_reason": ["x"]}))
    with pytest.raises(RuntimeError, match="PARTIAL"):
        R._load_breach_report(cfg, "grid")
    path.write_text(json.dumps({**ok, "scope": "anchors"}))
    with pytest.raises(RuntimeError, match="scope"):
        R._load_breach_report(cfg, "grid")
    # A report MISSING the partial field entirely is refused — absence is
    # never finality (v11 minor: hand-built --breach-report files fail loud).
    path.write_text(json.dumps({k: v for k, v in ok.items() if k != "partial"}))
    with pytest.raises(RuntimeError, match="lacks the 'partial' field"):
        R._load_breach_report(cfg, "grid")
    path.write_text(json.dumps(ok))
    rep, rep_path = R._load_breach_report(cfg, "grid")
    assert rep["breaching_cells"] == ["cellA"]
    # The load FREEZES the basis and returns the capregen-owned copy (byte-
    # verbatim, so provenance shas are stable across legs/respawns).
    basis = R.capregen_breach_basis_path(cfg, "grid")
    assert rep_path == basis and basis.exists()
    assert basis.read_bytes() == path.read_bytes()
    # A sub-2x cap is refused (codex BLOCKER regen-cap-not-enforced: the
    # registered remedy is >= 2x the generating cap; 2048 stays the default,
    # 4096 the registered per-invocation re-gen argument for this run).
    cfg_same = _cfg(tmp_path)
    assert cfg_same.max_new_tokens == 2048
    with pytest.raises(RuntimeError, match=">= 2x"):
        R._load_breach_report(cfg_same, "grid")
    with pytest.raises(RuntimeError, match="missing"):
        R._load_breach_report(_cfg(tmp_path / "elsewhere"), "grid")


def test_regen_cap_must_be_at_least_2x_base(tmp_path):
    """codex BLOCKER 2 pin: every cap in (base, 2*base) is REFUSED — a 2049
    capregen would silently violate the registered 4096 remedy AND leave the
    long tail truncated; 4096 (== 2x) and 8192 (>= 2x) are accepted."""
    basis = {
        "scope": "grid",
        "partial": False,
        "max_new_tokens": 2048,
        "realized_row_caps": [2048],
        "breaching_cells": ["cellA"],
    }
    for cap, ok in ((2049, False), (4095, False), (4096, True), (8192, True)):
        cfg = _cfg(tmp_path / f"cap{cap}", ["--max-new-tokens", str(cap)])
        cfg.manifest_dir.mkdir(parents=True)
        R.cap_hit_report_path(cfg, "grid").write_text(json.dumps(basis))
        if ok:
            rep, _ = R._load_breach_report(cfg, "grid")
            assert rep["breaching_cells"] == ["cellA"]
        else:
            with pytest.raises(RuntimeError, match=">= 2x"):
                R._load_breach_report(cfg, "grid")


# ── (5) frozen basis + postregen sibling path (code-review v11 C1) ──────


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


def test_phase_b_runs_after_phase_a_and_basis_survives_clobber(tmp_path):
    """The (a)+(c) regressions: after Phase A (gate) completes — post-regen
    reports emitted, and even the DEFAULT report path clobbered by a mixed-
    store measurement — Phase B (rest) and every respawn still key off the
    SAME frozen PRE-regen basis: no strict-cap wedge, no laundered breach set."""
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.manifest_dir.mkdir(parents=True)
    default = R.cap_hit_report_path(cfg, "anchors")
    default.write_text(json.dumps(_basis()))
    _rep1, p1 = R._load_breach_report(cfg, "anchors")  # Phase A froze the basis
    assert p1 == R.capregen_breach_basis_path(cfg, "anchors")
    # Worst case: the default path gets clobbered by a post-regen measurement
    # (breach list laundered to [], cap raised to 4096 -> would wedge/skip).
    default.write_text(
        json.dumps(_basis(max_new_tokens=4096, realized_row_caps=[2048, 4096], breaching_cells=[]))
    )
    rep2, p2 = R._load_breach_report(cfg, "anchors")  # Phase B / respawn
    assert p2 == p1
    assert rep2["breaching_cells"] == ["cellX"]  # the PRE-regen set, not []
    assert int(rep2["max_new_tokens"]) == 2048  # no 4096<=4096 refusal wedge


def test_respawned_worker_resumes_rather_than_refusing(tmp_path):
    """The (b) regression: a crashed capregen worker respawned AFTER a sibling
    finished (post-regen report emitted) reaches its per-block resume skip."""
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.manifest_dir.mkdir(parents=True)
    R.cap_hit_report_path(cfg, "grid").write_text(
        json.dumps(_basis(scope="grid", breaching_cells=["cellA"]))
    )
    R._load_breach_report(cfg, "grid")  # first invocation froze the basis
    # Sibling's post-regen emit lands on the SIBLING path, never the default:
    R.cap_hit_report_path(cfg, "grid", postregen=True).write_text(
        json.dumps(_basis(scope="grid", postregen=True, realized_row_caps=[2048, 4096]))
    )
    rep, _ = R._load_breach_report(cfg, "grid")  # respawn: NO refusal
    assert rep["breaching_cells"] == ["cellA"]
    # ...and the done-record predicate skips the block it already regenerated:
    block = R.Block("cellA", "ce", "steered", ("p1",))
    done = R.block_done_path(cfg.out_root, block)
    done.parent.mkdir(parents=True)
    done.write_text(
        json.dumps({"key": block.key, "regime_fp": "basefp", "capregen": {"max_new_tokens": 4096}})
    )
    assert R._capregen_block_done(cfg.out_root, block, "basefp", 4096) is True


def test_postregen_artifact_never_loads_as_breach_basis(tmp_path):
    """A post-regen measurement can never be mistaken for the pre-regen basis:
    the postregen stamp AND the mixed-cap signature each refuse, on the default
    path and via an explicit --breach-report alike."""
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.manifest_dir.mkdir(parents=True)
    R.cap_hit_report_path(cfg, "anchors").write_text(json.dumps(_basis(postregen=True)))
    with pytest.raises(RuntimeError, match="POST-regen"):
        R._load_breach_report(cfg, "anchors")
    R.cap_hit_report_path(cfg, "anchors").write_text(
        json.dumps(_basis(realized_row_caps=[2048, 4096]))
    )
    with pytest.raises(RuntimeError, match="MIXED-cap"):
        R._load_breach_report(cfg, "anchors")
    pr = tmp_path / "post.json"
    pr.write_text(json.dumps(_basis(postregen=True)))
    cfg2 = _cfg(tmp_path / "two", ["--max-new-tokens", "4096", "--breach-report", str(pr)])
    cfg2.manifest_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="POST-regen"):
        R._load_breach_report(cfg2, "anchors")


def test_explicit_breach_report_must_match_frozen_basis(tmp_path):
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.manifest_dir.mkdir(parents=True)
    src = tmp_path / "committed_basis.json"
    src.write_text(json.dumps(_basis()))
    cfg1 = _cfg(tmp_path, ["--max-new-tokens", "4096", "--breach-report", str(src)])
    _rep, p = R._load_breach_report(cfg1, "anchors")
    assert p == R.capregen_breach_basis_path(cfg1, "anchors")
    # Same file again: fine (byte-equal). A DIFFERENT file: refused.
    R._load_breach_report(cfg1, "anchors")
    other = tmp_path / "other_basis.json"
    other.write_text(json.dumps(_basis(breaching_cells=["cellY"])))
    cfg2 = _cfg(tmp_path, ["--max-new-tokens", "4096", "--breach-report", str(other)])
    with pytest.raises(RuntimeError, match="ONE"):
        R._load_breach_report(cfg2, "anchors")


def test_postregen_emit_sibling_path_base_cap_attribution_and_pending(tmp_path):
    """The v11 C1 fix shape: the post-regen emit (i) never touches the default
    report path (mechanical pin — destination is the *_postregen sibling),
    (ii) attributes legacy rows (no per-row cap) to the BASE cap so
    realized_row_caps / realized_caps_by_batch stay legible over the mixed
    store, and (iii) claims partial while capregen merges are pending —
    a mid-fleet per-worker emit can never publish partial: false (face (d))."""
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    cfg.anchors_dir.mkdir(parents=True)
    cfg.manifest_dir.mkdir(parents=True)
    # gate shard: MERGED (regen rows carry the raised per-row cap);
    # rest shard: legacy pre-regen rows (NO per-row cap field).
    regen = [
        {**_row("cellX", "v1", i, False, cap=4096), "context_id": f"x{i}", "draw": 0}
        for i in range(10)
    ]
    legacy = [
        {k: v for k, v in _row("cellA", "v1", i, False).items() if k != "max_new_tokens"}
        for i in range(10)
    ]
    _write_shard(cfg.anchors_dir / "anchors_gate_w0.jsonl", regen)
    _write_shard(cfg.anchors_dir / "anchors_rest_w0.jsonl", legacy)
    for batch in ("gate", "rest"):
        (cfg.manifest_dir / f"anchors_{batch}_w0_done.json").write_text(
            json.dumps(
                {
                    "regime_fp": "basefp",
                    "batch": batch,
                    "worker_index": 0,
                    "num_workers": 1,
                    "n_rows": 10,
                    "draws": 1,
                }
            )
        )
    default = R.cap_hit_report_path(cfg, "anchors")
    default.write_text(json.dumps(_basis()))
    before = default.read_bytes()
    rep = R.emit_cap_hit_report(
        cfg, "anchors", postregen=True, base_cap=2048, capregen_pending=["anchors_rest_w0"]
    )
    out = R.cap_hit_report_path(cfg, "anchors", postregen=True)
    assert out.exists() and out != default
    assert default.read_bytes() == before  # driving report byte-identical
    assert rep["postregen"] is True
    assert rep["partial"] is True  # pending merges -> never claims final
    assert any("capregen merge" in why for why in rep["partial_reason"])
    assert rep["capregen_pending"] == ["anchors_rest_w0"]
    # BASE-cap attribution: legacy rows read 2048, regen rows their own 4096.
    assert rep["realized_row_caps"] == [2048, 4096]
    assert rep["per_cell"]["cellX"]["realized_caps_by_batch"]["gate"] == [4096]
    assert rep["per_cell"]["cellA"]["realized_caps_by_batch"]["gate"] == [2048]
    # No pending merges + the same emit -> partial: false is reachable.
    rep2 = R.emit_cap_hit_report(cfg, "anchors", postregen=True, base_cap=2048)
    assert rep2["partial"] is False and rep2["capregen_pending"] == []
    # And the postregen artifact itself can never become a basis:
    with pytest.raises(RuntimeError, match="POST-regen"):
        R._validate_breach_basis(rep2, out, "anchors", cfg)


def test_postregen_emit_requires_base_cap(tmp_path):
    cfg = _cfg(tmp_path, ["--max-new-tokens", "4096"])
    with pytest.raises(ValueError, match="base_cap"):
        R.emit_cap_hit_report(cfg, "anchors", postregen=True)


# ── (6) MAJOR 2: unexpected shards fail loud, never counted ─────────────


def test_unexpected_shard_fails_loud_never_counted(tmp_path):
    a = tmp_path / "shard_a.jsonl"
    foreign = tmp_path / "shard_zz.jsonl"
    _write_shard(a, _cell_rows("cellA", "v1", 10, 0))
    _write_shard(foreign, _cell_rows("cellZ", "v1", 10, 10))
    with pytest.raises(RuntimeError, match="NOT in the expected set"):
        R.compute_cap_hit_report(
            [a, foreign], 2048, scope="grid", expected_shards={"shard_a.jsonl"}
        )


# ── (7) FIX 2: superseded pre-regen preservation ────────────────────────


def test_preservation_write_once_byte_identical_atomic(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True)
    src = cfg.anchors_dir / "anchors_gate_w0.jsonl"
    rows = [_row("cellX", "v1", i, True) for i in range(3)]
    _write_shard(src, rows)
    original = src.read_bytes()
    dest = R._preserve_preregen_file(cfg, "anchors", src)
    assert dest == R.preregen_superseded_dir(cfg, "anchors") / src.name
    assert dest.read_bytes() == original  # byte-recoverable
    # write-once: a re-entry after the store already merged must never clobber
    # the TRUE pre-regen bytes with post-regen content.
    src.write_text("REGENERATED\n")
    assert R._preserve_preregen_file(cfg, "anchors", src) == dest
    assert dest.read_bytes() == original
    # missing source fails loud (a breaching unit's shard must exist):
    with pytest.raises(RuntimeError, match="must exist"):
        R._preserve_preregen_file(cfg, "anchors", cfg.anchors_dir / "nope.jsonl")
    # atomicity: a crashed copy leaves NO destination and NO tmp residue.
    cfg2 = _cfg(tmp_path / "two")
    cfg2.anchors_dir.mkdir(parents=True)
    src2 = cfg2.anchors_dir / "anchors_gate_w0.jsonl"
    _write_shard(src2, rows)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(R.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        R._preserve_preregen_file(cfg2, "anchors", src2)
    ddir = R.preregen_superseded_dir(cfg2, "anchors")
    assert not (ddir / src2.name).exists()
    assert list(ddir.glob("*.tmp")) == []


# ── (8) MAJOR 1: judge staging never prefers a partial full prefix ──────


def test_resolve_anchors_dir_never_prefers_partial_full_prefix(tmp_path):
    """v11 MAJOR 1: capregen uploads land per-worker, so the full anchors
    mirror can transiently hold a strict SUBSET of the gate shards; the judge
    must stage from the COMPLETE gate mirror until the full prefix covers it —
    a partial full prefix would wedge gate-3 with a misleading error."""
    import issue2329_judge as J

    mirror = tmp_path
    gate = mirror / "anchors_gate"
    full = mirror / "anchors"
    gate.mkdir()
    for w in range(2):
        (gate / f"anchors_gate_w{w}.jsonl").write_text('{"context_id": "c"}\n')
    # No full prefix yet: the early gate mirror stages (pre-existing behavior).
    assert J._resolve_anchors_dir(mirror) == gate
    # 1-of-2 workers uploaded to the full prefix (mid-fleet capregen): the
    # PARTIAL full prefix must never shadow the complete gate mirror.
    full.mkdir()
    (full / "anchors_gate_w0.jsonl").write_text('{"context_id": "c"}\n')
    assert J._resolve_anchors_dir(mirror) == gate
    # Full prefix covers the gate mirror's shard names: preferred (rest
    # shards are a superset — gate-3 filters to gate contexts).
    (full / "anchors_gate_w1.jsonl").write_text('{"context_id": "c"}\n')
    assert J._resolve_anchors_dir(mirror) == full
    (full / "anchors_rest_w0.jsonl").write_text('{"context_id": "c"}\n')
    assert J._resolve_anchors_dir(mirror) == full
    # Neither dir holds shards: canonical full path (loaders fail loud).
    empty = tmp_path / "empty_mirror"
    empty.mkdir()
    assert J._resolve_anchors_dir(empty) == empty / "anchors"
