"""Issue #2389 run driver — fork-delta unit tests (run.py level).

Covers the deltas the analysis/judge/steering/vllm test files do not:

- the plan §4.7 item-1 per-cell cap table (20 named cells at 4096) and the
  full ``cell_max_new_tokens`` precedence chain (recalibrated > table >
  ``_rev``->``_fwd`` inherit > MAX_NEW_TOKENS default);
- ``_resolve_cap``'s explicit ``--max-new-tokens`` override (the capregen
  raised-cap contract);
- ``_cell_bucketed_chunks`` (plan §4.7 item 2: a generate chunk never mixes
  cells; order preserved; chunk size respected);
- the ce-only grid: ``enumerate_blocks`` == 117 over the REAL #2162 pair
  bank (pure python — no tokenizer / no GPU), unique keys, no pe blocks;
- the ``_validate_breach_basis`` refusal matrix (happy path + every
  fail-loud branch, matched on the verbatim error text);
- bank2389 identity pins (pinned revision; manifest override fields via the
  real ``bank_manifest_2389`` body over a stubbed tokenizer-parametric
  parent delegate).

All fixtures are synthetic/tmp — no committed eval_results reads (no
sparse-cone additions needed).
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_run as R  # noqa: E402

BANK = R.BANK  # bank2162 (pairs / cells)
B89 = R.BANK29  # bank2389 (issue-2389 identity)

# The plan §4.7 item-1 list — the 20 parent cells whose cap-hit exceeded the
# 2% trigger, all raised to 4096 (every other cell stays at the 2048 default).
PLAN_471_CELLS = {
    "filler_swap",
    "language_implied",
    "persona_role_header",
    "reasoning_style",
    "instr_language",
    "user_emotion",
    "verbosity",
    "user_expertise",
    "demo_persona",
    "recency_instr_format_d5",
    "query_content",
    "recency_persona_prompted_d5",
    "demo_format",
    "recency_prior_topic_d3",
    "persona_prompted",
    "recency_persona_prompted_d3",
    "conflict_persona_fwd",
    "recency_instr_format_d3",
    "recency_fact_user_name_d3",
    "instr_format",
}


def _mk_cfg(tmp_path: Path, **over) -> R.RunConfig:
    """Minimal RunConfig for cap/validator tests (no model, no GPU)."""
    kw = dict(
        phase="cap_report",
        out_root=tmp_path / "out",
        log_dir=tmp_path / "logs",
        model_id=B89.MODEL_ID,
        model_revision=B89.MODEL_REVISION,
        tiny=True,
        n_layers=2,
        hidden=8,
        device="cpu",
        gen_batch=4,
        capture_batch=4,
        max_new_tokens=R.MAX_NEW_TOKENS,
        anchor_draws=1,
        grid_draws=1,
        seed_base=0,
        smoke=True,
        pilot=False,
        force=False,
        force_past_halt_gates=False,
        worker_index=0,
        num_workers=1,
        upload_mode="none",
        upload_every=1,
        planned_wall_h=0.1,
        gpu_hours_budgeted=0.1,
        pools_path=None,
    )
    kw.update(over)
    return R.RunConfig(**kw)


# ---------------------------------------------------------------- grid shape


def test_slots_are_ce_only_and_arms_are_three():
    assert R.SLOTS == ("ce",)
    assert R.ARMS == ("steered", "shuffled", "crosstype")


def test_enumerate_blocks_is_117_ce_only_unique_keys():
    pairs = BANK.build_pairs()  # pure python; 1,404 directed pairs
    blocks = R.enumerate_blocks(pairs)
    assert len(blocks) == 117 == 39 * 1 * 3
    assert {b.slot for b in blocks} == {"ce"}  # no pe blocks under the fork
    keys = [b.key for b in blocks]
    assert len(set(keys)) == 117
    # every cell contributes exactly 3 blocks (one per arm), full pair set
    per_cell = {}
    for b in blocks:
        per_cell.setdefault(b.cell, []).append(b)
    assert set(per_cell) == set(BANK.all_cells())
    for cell, cell_blocks in per_cell.items():
        assert sorted(b.arm for b in cell_blocks) == sorted(R.ARMS), cell
        assert all(B89.INTACT_FLOOR_PER_CELL <= len(b.pair_ids) <= 36 for b in cell_blocks), cell


# ------------------------------------------------------------- per-cell caps


def test_cap_table_matches_plan_471_item1():
    assert set(R.CELL_MAX_NEW_TOKENS) == PLAN_471_CELLS
    assert all(v == 4096 for v in R.CELL_MAX_NEW_TOKENS.values())
    assert R.MAX_NEW_TOKENS == 2048


def test_cell_max_new_tokens_precedence_chain():
    # 1. recalibrated beats the table
    assert R.cell_max_new_tokens("filler_swap", {"filler_swap": 8192}) == 8192
    # 2. named table
    assert R.cell_max_new_tokens("filler_swap") == 4096
    # 3. _rev -> _fwd inherit (conflict_persona_rev is NOT in the table)
    assert "conflict_persona_rev" not in R.CELL_MAX_NEW_TOKENS
    assert R.cell_max_new_tokens("conflict_persona_rev") == 4096
    # 3b. a recalibrated fwd beats the table fwd for the _rev sibling
    assert R.cell_max_new_tokens("conflict_persona_rev", {"conflict_persona_fwd": 8192}) == 8192
    # 3c. but a recalibrated entry for the _rev cell itself wins over both
    assert (
        R.cell_max_new_tokens(
            "conflict_persona_rev",
            {"conflict_persona_rev": 6144, "conflict_persona_fwd": 8192},
        )
        == 6144
    )
    # 4. unknown cell -> the 2048 default
    assert R.cell_max_new_tokens("query_topic") == R.MAX_NEW_TOKENS


def test_resolve_cap_explicit_override_wins(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=9000, max_new_tokens_explicit=True)
    assert R._resolve_cap(cfg, "filler_swap") == 9000  # capregen contract
    assert R._resolve_cap(cfg, "query_topic", {"query_topic": 4096}) == 9000
    cfg2 = _mk_cfg(tmp_path, max_new_tokens=9000, max_new_tokens_explicit=False)
    assert R._resolve_cap(cfg2, "filler_swap") == 4096  # table governs
    assert R._resolve_cap(cfg2, "query_topic") == R.MAX_NEW_TOKENS


def test_cell_bucketed_chunks_never_mix_cells_and_preserve_order():
    contexts = {
        "a1": {"cell": "alpha"},
        "a2": {"cell": "alpha"},
        "a3": {"cell": "alpha"},
        "b1": {"cell": "beta"},
        "a4": {"cell": "alpha"},
        "b2": {"cell": "beta"},
    }
    order = ["a1", "a2", "a3", "b1", "a4", "b2"]
    chunks = R._cell_bucketed_chunks(contexts, order, chunk_size=2)
    # no chunk mixes cells
    for cell, cids in chunks:
        assert {contexts[c]["cell"] for c in cids} == {cell}
        assert len(cids) <= 2
    # within-cell incoming order preserved, all ids covered exactly once
    flat_alpha = [c for cell, cids in chunks if cell == "alpha" for c in cids]
    flat_beta = [c for cell, cids in chunks if cell == "beta" for c in cids]
    assert flat_alpha == ["a1", "a2", "a3", "a4"]
    assert flat_beta == ["b1", "b2"]
    assert sorted(flat_alpha + flat_beta) == sorted(order)


# ------------------------------------------------- capregen breach validator


def _good_report(scope: str = "anchors") -> dict:
    """A complete pre-regen cap-hit report with one breaching cell."""
    return {
        "scope": scope,
        "partial": False,
        "realized_row_caps": [4096, 2048],
        "breaching_cells": ["filler_swap"],
        "per_cell": {
            "filler_swap": {
                "cap_hit_pct": 5.0,
                "realized_caps_by_batch": {"gate": [4096], "rest": [4096]},
            },
            "query_topic": {
                "cap_hit_pct": 0.0,
                "realized_caps_by_batch": {"rest": [2048]},
            },
        },
    }


def test_validate_breach_basis_happy_path(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192, max_new_tokens_explicit=True)
    # 8192 == 2 x the largest breaching generating cap (4096) — passes
    R._validate_breach_basis(_good_report(), tmp_path / "rep.json", "anchors", cfg)


def test_validate_breach_basis_refuses_wrong_scope(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    with pytest.raises(RuntimeError, match=r"has scope='grid', need 'anchors'"):
        R._validate_breach_basis(_good_report("grid"), tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_postregen(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    rep["postregen"] = True
    with pytest.raises(RuntimeError, match=r"POST-regen measurement"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_empty_caps(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    rep["realized_row_caps"] = []
    with pytest.raises(RuntimeError, match=r"NO realized row caps"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_mixed_bucket(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    # one (cell, batch) bucket carrying TWO caps = post-regen/half-done shape
    rep["per_cell"]["filler_swap"]["realized_caps_by_batch"]["rest"] = [4096, 8192]
    with pytest.raises(RuntimeError, match=r"MIXED realized caps"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_missing_partial_field(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    del rep["partial"]
    with pytest.raises(RuntimeError, match=r"lacks the 'partial' field"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_partial_report(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    rep["partial"] = True
    rep["partial_reason"] = "pending capture shards"
    with pytest.raises(RuntimeError, match=r"is PARTIAL \(pending capture shards\)"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_breaching_cell_absent_from_per_cell(tmp_path):
    cfg = _mk_cfg(tmp_path, max_new_tokens=8192)
    rep = _good_report()
    rep["breaching_cells"] = ["filler_swap", "ghost_cell"]
    with pytest.raises(RuntimeError, match=r"absent from per_cell: \['ghost_cell'\]"):
        R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_refuses_sub_2x_regen_cap(tmp_path):
    # cfg cap below 2 x the largest breaching generating cap (4096 -> needs 8192)
    cfg = _mk_cfg(tmp_path, max_new_tokens=6000, max_new_tokens_explicit=True)
    with pytest.raises(RuntimeError, match=r"requires --max-new-tokens >= 2x"):
        R._validate_breach_basis(_good_report(), tmp_path / "r.json", "anchors", cfg)


def test_validate_breach_basis_no_breach_needs_no_cap_floor(tmp_path):
    # zero breaching cells: the 2x floor clause is inert (nothing to regen)
    cfg = _mk_cfg(tmp_path, max_new_tokens=2048)
    rep = _good_report()
    rep["breaching_cells"] = []
    R._validate_breach_basis(rep, tmp_path / "r.json", "anchors", cfg)


# ------------------------------------------- B4: smoke gate-slice extension


def test_b4_smoke_gate_slice_extension_supplies_spot_cells():
    """B4 mechanizable check (r1 review verbatim): the bare 2-chunk smoke
    slice cannot supply the injection gate's 12 spot cells (the pre-fix
    KeyError shape); the extension's extra contexts make
    `{spot cells} <= set(BANK.pairs_by_cell(filtered pairs))` hold."""
    contexts = list(BANK.build_contexts())
    pairs = BANK.build_pairs()  # full bank (token-identity drops need no GPU here)
    chunks = R.enumerate_capture_chunks(contexts)[:2]
    sliced = {c for ch in chunks for c in ch.context_ids}
    # synthetic same-cell donor maps (real ones live in the frozen manifest —
    # tokenizer-dependent; the extension math only needs pair-id resolution)
    by_cell = BANK.pairs_by_cell(pairs)
    donor_maps: dict[str, dict[str, str]] = {"shuffled": {}, "crosstype": {}}
    for p in pairs:
        siblings = [q for q in by_cell[p.cell] if q.pair_id != p.pair_id]
        if siblings:
            donor_maps["shuffled"][p.pair_id] = siblings[0].pair_id
            donor_maps["crosstype"][p.pair_id] = siblings[-1].pair_id
    spots, extra = R._smoke_gate_slice_extension(contexts, sliced, pairs, donor_maps)
    assert len(spots) == 12
    spot_cells = {s["cell"] for s in spots}
    # PRE-FIX shape: the bare slice misses gate cells (the B4 crash)
    pairs_sliced = [p for p in pairs if p.a in sliced and p.b in sliced]
    assert not spot_cells <= set(BANK.pairs_by_cell(pairs_sliced))
    # POST-FIX: slice + extension covers every spot cell through the filter
    assert extra and not set(extra) & sliced
    covered = sliced | set(extra)
    pairs_ext = [p for p in pairs if p.a in covered and p.b in covered]
    assert spot_cells <= set(BANK.pairs_by_cell(pairs_ext))
    # donor pairs of non-steered spots survive the filter too (payload_for_arm
    # dereferences their captured states + pair ids)
    ext_ids = {p.pair_id for p in pairs_ext}
    for s in spots:
        if s["arm"] != "steered":
            key = "shuffled" if s["arm"] == "shuffled" else "crosstype"
            assert donor_maps[key][s["pair"].pair_id] in ext_ids
    # the synthetic chunk's identity can never collide with a real chunk
    assert len(R.enumerate_capture_chunks(contexts)) < R.SMOKE_GATE_CHUNK_INDEX


def test_grid_smoke_slice_extension_covers_block_dereferences():
    """Crash-fix r2 pin (grid sibling of B4): phase_grid composes its smoke
    blocks over the FULL surviving pair set while the smoke bank captures
    only 2 chunks + the B4 gate extension — pre-fix, ``_block_cells`` /
    ``payload_for_arm`` KeyError on the first out-of-closure pair or donor
    context AFTER the pilot/anchors/vLLM smoke spend (reviewer VM probe:
    9 own-pair + 23 donor gaps). The extension's need-set covers every
    dereference, incl. the ``--pilot`` leg (``blocks[:1]`` of the same set)."""
    contexts = list(BANK.build_contexts())
    pairs = BANK.build_pairs()
    chunks = R.enumerate_capture_chunks(contexts)[:2]
    sliced = {c for ch in chunks for c in ch.context_ids}
    by_cell = BANK.pairs_by_cell(pairs)
    donor_maps: dict[str, dict[str, str]] = {"shuffled": {}, "crosstype": {}}
    for p in pairs:
        siblings = [q for q in by_cell[p.cell] if q.pair_id != p.pair_id]
        if siblings:
            donor_maps["shuffled"][p.pair_id] = siblings[0].pair_id
            donor_maps["crosstype"][p.pair_id] = siblings[-1].pair_id
    np_ids: frozenset[str] = frozenset()
    # PRE-FIX captured set: base 2 chunks + the B4 gate extension only
    _spots, gate_extra = R._smoke_gate_slice_extension(contexts, sliced, pairs, donor_maps)
    prefix_covered = sliced | set(gate_extra)
    # the grid smoke leg's block set — the SHARED composer both phase_bank's
    # closure and phase_grid's smoke branch call (drift-guard, crash-fix r2b)
    blocks, _excl = R._smoke_grid_block_set(pairs, np_ids, donor_maps)
    assert blocks, "smoke grid block set must be non-empty (--pilot runs blocks[:1])"
    pairs_by_id = {p.pair_id: p for p in pairs}

    def _block_deref(block) -> set[str]:
        """Exactly what _block_cells/payload_for_arm read from the bank."""
        out: set[str] = set()
        for pid in block.pair_ids:
            p = pairs_by_id[pid]
            out.update((p.a, p.b))
            if block.arm != "steered":
                key = "shuffled" if block.arm == "shuffled" else "crosstype"
                out.add(pairs_by_id[donor_maps[key][pid]].b)
        return out

    deref_all = set().union(*(_block_deref(b) for b in blocks))
    # PRE-FIX shape: the covered set misses grid dereferences (the r2 bug)
    missing_deref = deref_all - prefix_covered
    assert missing_deref, "base slice + B4 extension must not already close the grid leg"
    L, H = 2, 4
    g = torch.Generator().manual_seed(0)

    def _bank_over(covered: set[str]) -> dict:
        return {
            "per_context": {
                cid: {"v_ce": torch.randn(L, H, generator=g), "ctx_len": 8, "prefix_end": 2}
                for cid in covered
            }
        }

    broken = next(b for b in blocks if _block_deref(b) - prefix_covered)
    with pytest.raises(KeyError):
        R._block_cells(_bank_over(prefix_covered), broken, pairs_by_id, donor_maps)

    # POST-FIX: the extension closes the leg — every smoke block composes
    need, grid_extra = R._smoke_grid_slice_extension(
        contexts, prefix_covered, pairs, donor_maps, np_ids
    )
    assert grid_extra and not set(grid_extra) & prefix_covered
    covered = prefix_covered | set(grid_extra)
    bank_fixed = _bank_over(covered)
    for block in blocks:
        cells = R._block_cells(bank_fixed, block, pairs_by_id, donor_maps)
        assert len(cells) == block.n_pairs
    # closure (the merge-time assert's exact predicate): the FULL dereference
    # set — need additionally carries donor As so donor PAIRS survive the
    # bank phase's captured-pair filter (the B4 idiom)
    assert deref_all <= need <= covered
    # the extension chunk's claim/done identity is distinct from every real
    # chunk AND the B4 gate chunk (a retained pre-r2 smoke out-root resumes
    # by re-capturing exactly the grid-closure delta)
    assert R.SMOKE_GRID_CHUNK_INDEX != R.SMOKE_GATE_CHUNK_INDEX
    assert len(R.enumerate_capture_chunks(contexts)) < R.SMOKE_GRID_CHUNK_INDEX


# ------------------- injection gate second-row donor closure (crash-fix r1)


def _cf_pair(pid: str, cell: str, a: str, b: str):
    """Minimal Pair2162 for the gate second-row pool fixtures."""
    return BANK.Pair2162(pair_id=pid, cell=cell, carrier="d1", value_a="va", value_b="vb", a=a, b=b)


def test_gate_second_row_pool_excludes_unresolvable_donors_smoke(caplog):
    """Crash-fix r1 repro (pod 2026-08-23, ``KeyError`` in ``payload_for_arm``
    at bank worker 0): under ``--smoke`` the filtered pair set carries donor
    closure only for the 12 SPOT pairs (B4), but the gate's SECOND-ROW
    candidates dereference their OWN donors. Pre-fix, the unfiltered pool
    picked a candidate whose donor pair was filtered out and crashed; the
    fixed pool drops exactly the unresolvable candidates (donor pair absent,
    donor B state uncaptured, or no donor assignment), BEFORE the
    ``pe_excluded_reason`` check that dereferences the same donor."""
    L, H = 2, 4
    g = torch.Generator().manual_seed(0)

    def _rec() -> dict:
        return {"v_ce": torch.randn(L, H, generator=g), "ctx_len": 8, "prefix_end": 2}

    # spot A-context length 3; candidate As length 4 (pool-eligible); the
    # donor-only pairs' As length 3 (length-excluded from the pool itself)
    ctx_ids = {
        "sA": [1, 2, 3], "sB": [1, 2, 3], "dsA": [1, 2, 3], "dsB": [1, 2, 3],
        "qoA": [1, 2, 3, 4], "qoB": [1, 2, 3, 4],
        "qnA": [1, 2, 3, 4], "qnB": [1, 2, 3, 4],
        "qmA": [1, 2, 3, 4], "qmB": [1, 2, 3, 4],
        "okA": [1, 2, 3, 4], "okB": [1, 2, 3, 4],
        "dnA": [1, 2, 3], "dnB": [1, 2, 3],
        "doA": [1, 2, 3], "doB": [1, 2, 3],
    }  # fmt: skip
    spot = _cf_pair("S", "cellA", "sA", "sB")
    d_spot = _cf_pair("DS", "cellA", "dsA", "dsB")  # spot donor (B4-closed)
    q_out = _cf_pair("QOUT", "cellB", "qoA", "qoB")  # donor pair FILTERED OUT
    q_norec = _cf_pair("QNOREC", "cellB", "qnA", "qnB")  # donor B uncaptured
    q_nomap = _cf_pair("QNOMAP", "cellB", "qmA", "qmB")  # no donor assignment
    q_ok = _cf_pair("QOK", "cellB", "okA", "okB")  # fully resolvable
    d_norec = _cf_pair("DNR", "cellB", "dnA", "dnB")
    d_ok = _cf_pair("DOK", "cellB", "doA", "doB")
    pairs = [spot, d_spot, q_out, q_norec, q_nomap, q_ok, d_norec, d_ok]
    pairs_by_id = {p.pair_id: p for p in pairs}
    # QOUT's donor id is the 2026-08-23 pod crash key VERBATIM (round-7
    # concern repro-does-not-pin-verbatim-pod-key): the filtered-out donor
    # pair id the unfiltered pool dereferenced on the pod.
    pod_crash_key = "conflict_persona_rev::i2d1-i1d2::d1"
    donor_maps = {
        "shuffled": {"S": "DS", "QOUT": pod_crash_key, "QNOREC": "DNR", "QOK": "DOK"},
        "crosstype": {},
    }
    recs = {cid: _rec() for cid in ctx_ids if cid != "dnB"}  # DNR's B uncaptured
    bank = {"per_context": recs}

    # the fix is WIRED into the gate (not a hollow twin of the inline pool)
    assert "_gate_second_row_pool(" in inspect.getsource(R.run_injection_gate)

    # PRE-FIX shape: the unfiltered pool ranks QOUT first; composing its
    # payload dereferences pairs_by_id['conflict_persona_rev::i2d1-i1d2::d1']
    # — the pod KeyError, key asserted verbatim
    prefix_pool = [
        p for p in pairs if p.pair_id != spot.pair_id and len(ctx_ids[p.a]) != len(ctx_ids[spot.a])
    ]
    assert prefix_pool[0] is q_out
    with pytest.raises(KeyError, match=re.escape(pod_crash_key)):
        R.payload_for_arm(bank, prefix_pool[0], "ce", "shuffled", donor_maps, pairs_by_id)
    # pe spots crash one call earlier, inside pe_excluded_reason (same donor)
    with pytest.raises(KeyError, match=re.escape(pod_crash_key)):
        R.pe_excluded_reason(q_out, "shuffled", frozenset(), donor_maps, pairs_by_id)

    # POST-FIX: only the resolvable candidate survives, and it composes clean
    with caplog.at_level(logging.INFO):
        others = R._gate_second_row_pool(
            pairs, spot, "ce", "shuffled", ctx_ids, frozenset(), donor_maps, pairs_by_id, recs
        )
    assert [p.pair_id for p in others] == ["QOK"]
    for p in [spot, *others]:  # the realized gate batch, exactly as composed
        payload, donor_id = R.payload_for_arm(bank, p, "ce", "shuffled", donor_maps, pairs_by_id)
        assert payload.shape == (1, L, H), payload.shape
        assert donor_id == {"S": "DS", "QOK": "DOK"}[p.pair_id]
    # fix-engaged signal: the drop is LOGGED (3 of 4 candidates unresolvable)
    assert any("second-row donor-closure filter kept 1/4" in r.getMessage() for r in caplog.records)
    # pe slot: resolvability runs BEFORE pe_excluded_reason — no KeyError
    others_pe = R._gate_second_row_pool(
        pairs, spot, "pe", "shuffled", ctx_ids, frozenset(), donor_maps, pairs_by_id, recs
    )
    assert [p.pair_id for p in others_pe] == ["QOK"]


def test_gate_second_row_pool_production_invariant_under_full_closure(caplog):
    """Production invariance (crash-fix r1): with FULL donor closure (every
    candidate's donor pair present and its B captured — the production pair
    set by construction) the resolvability filter keeps ALL candidates, in
    order, for every arm, and logs nothing."""
    ctx_ids = {"sA": [1, 2, 3], "sB": [1, 2, 3]}
    pairs = [_cf_pair("S", "cellA", "sA", "sB")]
    for i in range(6):
        ctx_ids[f"a{i}"] = [1, 2, 3, 4]
        ctx_ids[f"b{i}"] = [1, 2, 3, 4]
        pairs.append(_cf_pair(f"P{i}", "cellB", f"a{i}", f"b{i}"))
    spot = pairs[0]
    pairs_by_id = {p.pair_id: p for p in pairs}
    # ring donor assignment over the candidates: every donor resolves
    donor_maps = {
        "shuffled": {f"P{i}": f"P{(i + 1) % 6}" for i in range(6)} | {"S": "P0"},
        "crosstype": {f"P{i}": f"P{(i + 2) % 6}" for i in range(6)} | {"S": "P1"},
    }
    recs = {cid: {} for cid in ctx_ids}  # membership is all the filter reads
    expected = [f"P{i}" for i in range(6)]
    with caplog.at_level(logging.INFO):
        for arm in R.ARMS:
            pool = R._gate_second_row_pool(
                pairs, spot, "ce", arm, ctx_ids, frozenset(), donor_maps, pairs_by_id, recs
            )
            assert [p.pair_id for p in pool] == expected, arm
    assert not any("donor-closure filter" in r.getMessage() for r in caplog.records)


# ------------------------------------- capregen owning record (B11/B3 r1)


def _done_manifest(cfg: R.RunConfig, bid: str, w: int = 0, fp: str = "fp", **extra) -> None:
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.manifest_dir / f"anchors_{bid}_w{w}_done.json",
        {"regime_fp": fp, "worker_index": w, "n_rows": 1, **extra},
    )


def test_capregen_owning_record_engine_aware(tmp_path):
    """B11: capregen units are CELLS resolved to their OWNING engine record —
    a rest cell lives in exactly one of rest_/parity_/vllm_; gate cells only
    in gate_. The strided per-worker re-derivation (which mis-aligned against
    the generation-time vLLM exclusion) is gone."""
    cfg = _mk_cfg(tmp_path)
    _done_manifest(cfg, "vllm_filler_swap")
    bid, path, rec = R._capregen_owning_record(cfg, "filler_swap", "rest")
    assert bid == "vllm_filler_swap" and path.exists() and rec["n_rows"] == 1
    # the gate batch never resolves to a rest/parity/vllm record
    with pytest.raises(RuntimeError, match="no done record"):
        R._capregen_owning_record(cfg, "filler_swap", "gate")
    _done_manifest(cfg, "gate_filler_swap")
    assert R._capregen_owning_record(cfg, "filler_swap", "gate")[0] == "gate_filler_swap"


def test_capregen_owning_record_refuses_multiple_owners(tmp_path):
    cfg = _mk_cfg(tmp_path)
    _done_manifest(cfg, "rest_filler_swap")
    _done_manifest(cfg, "vllm_filler_swap")
    with pytest.raises(RuntimeError, match="owning done records"):
        R._capregen_owning_record(cfg, "filler_swap", "rest")


def test_capregen_owning_record_ignores_gen_done_sentinels(tmp_path):
    """A vLLM pre-capture `*_gen_done.json` sentinel is NOT a done record —
    it must neither satisfy ownership nor manufacture a double-owner error
    beside the real record."""
    cfg = _mk_cfg(tmp_path)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.manifest_dir / "anchors_vllm_filler_swap_w0_gen_done.json", {"regime_fp": "fp"}
    )
    with pytest.raises(RuntimeError, match="no done record"):
        R._capregen_owning_record(cfg, "filler_swap", "rest")
    _done_manifest(cfg, "vllm_filler_swap")
    assert R._capregen_owning_record(cfg, "filler_swap", "rest")[0] == "vllm_filler_swap"


# ------------------------------------------------------- bank2389 identity


def test_bank2389_identity_pins():
    assert B89.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert B89.MODEL_REVISION == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert isinstance(B89.INTACT_FLOOR_PER_CELL, int) and B89.INTACT_FLOOR_PER_CELL >= 1


def test_bank_manifest_2389_overrides_identity_fields(monkeypatch):
    # The parent delegate is tokenizer-parametric (heavy); stub it and run the
    # REAL bank_manifest_2389 override body.
    monkeypatch.setattr(B89.B29, "bank_manifest_2329", lambda *a, **k: {"seed": B89.SEED})
    m = B89.bank_manifest_2389(tokenizer=None)
    assert m["issue"] == 2389
    assert m["parent_issue"] == 2329
    assert m["pe_slot_dropped"] is True
    assert m["slots"] == ["ce"]
    assert m["model_revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def test_repro_records_model_revision(tmp_path):
    rep = R._repro(_mk_cfg(tmp_path))
    assert rep["model_revision"] == B89.MODEL_REVISION
    assert rep["model_id"] == B89.MODEL_ID


# ---------------------------------------------- B1: gate-0c canary hf leg


def test_gate0c_hf_canary_verify_call_signature(tmp_path, monkeypatch):
    """B1 (r1 review): the gate-0c verify wraps `hub.retry_transient`, whose
    `what=` kwarg is REQUIRED — the pre-fix call died with TypeError AFTER the
    canary uploaded but BEFORE the verification assert, so every production
    `--phase bank --upload hf` crashed at gate 0c (invisible to batteries:
    `--upload none|local-mirror` early-returns). Fakes sit ONLY at the
    network boundary (upload helper + HfApi.file_exists)."""

    def _fake_upload_dir_hf(local_dir, remote_prefix, allow_patterns):
        return [f"{remote_prefix}/{p}" for p in allow_patterns]

    monkeypatch.setattr(R, "upload_dir_hf", _fake_upload_dir_hf)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "file_exists",
        lambda self, repo_id, filename, *, repo_type=None, revision=None, token=None: True,
    )
    cfg = _mk_cfg(tmp_path, upload_mode="hf")
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    R._gate0c_hf_write_canary(cfg)  # pre-fix: TypeError (missing 'what')
    assert (cfg.gates_dir / "hf_write_canary.json").exists()


# ------------------------------------- B12/g7-C1: _select_gen_batch rule


def _r2_leg(b: int, s: float, h: float | None) -> dict:
    return {
        "gen_batch": b,
        "rollouts": 10,
        "wall_s": s * 10,
        "s_per_rollout": s,
        "hbm_headroom_gib": h,
    }


def test_select_gen_batch_argmin_s_per_rollout():
    r2 = {16: _r2_leg(16, 0.5, 20.0), 32: _r2_leg(32, 0.3, 20.0)}
    assert R._select_gen_batch(r2, [16, 32]) == (32, True)


def test_select_gen_batch_headroom_floor_excludes_faster_candidate():
    # 32 is faster but sits below the 10 GiB floor -> 16 wins.
    r2 = {16: _r2_leg(16, 0.5, 20.0), 32: _r2_leg(32, 0.3, 9.9)}
    assert R._select_gen_batch(r2, [16, 32]) == (16, True)


def test_select_gen_batch_exact_floor_is_eligible():
    r2 = {16: _r2_leg(16, 0.5, 20.0), 32: _r2_leg(32, 0.3, R.PILOT_HBM_HEADROOM_GIB)}
    assert R._select_gen_batch(r2, [16, 32]) == (32, True)


def test_select_gen_batch_exact_tie_picks_smaller():
    r2 = {16: _r2_leg(16, 0.4, 20.0), 32: _r2_leg(32, 0.4, 20.0)}
    assert R._select_gen_batch(r2, [16, 32]) == (16, True)


def test_select_gen_batch_none_headroom_is_eligible():
    # CPU pilot: headroom unprobeable => candidates stay eligible.
    r2 = {16: _r2_leg(16, 0.5, None), 32: _r2_leg(32, 0.3, None)}
    assert R._select_gen_batch(r2, [16, 32]) == (32, True)


def test_select_gen_batch_mixed_none_and_below_floor():
    r2 = {16: _r2_leg(16, 0.5, None), 32: _r2_leg(32, 0.3, 5.0)}
    assert R._select_gen_batch(r2, [16, 32]) == (16, True)


def test_select_gen_batch_no_eligible_smallest_with_warning(caplog):
    r2 = {16: _r2_leg(16, 0.5, 1.0), 32: _r2_leg(32, 0.3, 2.0)}
    with caplog.at_level(logging.WARNING):
        assert R._select_gen_batch(r2, [16, 32]) == (16, False)
    assert "headroom floor" in caplog.text


def test_select_gen_batch_explicit_single_candidate_below_floor(caplog):
    # --gen-batch explicit: one candidate; below-floor still selects it,
    # recorded via headroom_ok=False (never a crash).
    r2 = {24: _r2_leg(24, 0.5, 0.5)}
    with caplog.at_level(logging.WARNING):
        assert R._select_gen_batch(r2, [24]) == (24, False)


# ------------------- B12/B3: cap-report expected sets span every namespace


def _frozen_manifest(cfg: R.RunConfig) -> None:
    cfg.bank_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.bank_dir / "bank.json",
        {"dropped_pairs": [], "token_identity": {"n_intact": len(BANK.build_pairs())}},
    )


def test_b12_cap_report_inputs_span_parity_and_vllm_namespaces(tmp_path):
    """B3/B12: the anchors cap-report aggregate derives expectations from
    EVERY engine/batch namespace through the real entrypoint — gate_ + rest_
    + parity_ + vllm_ (gen-done sentinel) — never refusing parity shards as
    foreign nor dropping vllm cells from the expected set."""
    cfg = _mk_cfg(tmp_path)
    _frozen_manifest(cfg)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    gate_ids, rest_ids, contexts = R._anchor_context_order(cfg)
    gate_cells = sorted(R._group_by_cell(gate_ids, contexts))
    rest_cells = sorted(R._group_by_cell(rest_ids, contexts))
    assert gate_cells and rest_cells
    for c in gate_cells:
        _done_manifest(cfg, f"gate_{c}")
        (cfg.anchors_dir / f"anchors_gate_{c}_w0.jsonl").write_text("{}\n")
    owners = {c: ("rest", "parity", "vllm")[i % 3] for i, c in enumerate(rest_cells)}
    for c, own in owners.items():
        bid = f"{own}_{c}"
        if own == "vllm":
            # production-vLLM text-persist: gen-done sentinel, capture pending
            R._write_json_atomic(
                cfg.manifest_dir / f"anchors_{bid}_w0_gen_done.json", {"n_rows": 1}
            )
        else:
            _done_manifest(cfg, bid)
        (cfg.anchors_dir / f"anchors_{bid}_w0.jsonl").write_text("{}\n")
    paths, expected, why = R._cap_report_inputs(cfg, "anchors")
    assert why is None and expected is not None
    assert any(n.startswith("anchors_parity_") for n in expected)
    assert any(n.startswith("anchors_vllm_") for n in expected)
    assert {p.name for p in paths} == expected


def test_b12_cap_report_inputs_partial_without_full_cell_coverage(tmp_path):
    """A rest cell with no done record under ANY of rest_/parity_/vllm_ =>
    expected set underivable (PARTIAL) — never a silently narrowed set."""
    cfg = _mk_cfg(tmp_path)
    _frozen_manifest(cfg)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    gate_ids, rest_ids, contexts = R._anchor_context_order(cfg)
    for c in R._group_by_cell(gate_ids, contexts):
        _done_manifest(cfg, f"gate_{c}")
    rest_cells = sorted(R._group_by_cell(rest_ids, contexts))
    for c in rest_cells[:-1]:
        _done_manifest(cfg, f"parity_{c}")
    _paths, expected, why = R._cap_report_inputs(cfg, "anchors")
    assert expected is None
    assert why is not None and "underivable" in why
    assert rest_cells[-1] in why


# ---------------- R1+R2 (r2 review): cap recalibration + gate_slice labeling


def _fake_generate_batch(
    model,
    tokenizer,
    contexts,
    n=10,
    hook=None,
    max_new_tokens=1024,
    temperature=1.0,
    seed_base=42,
    render_fn=None,
    ids_fn=None,
    top_p=None,
    share_prefill=False,
):
    """Signature-conformant boundary fake of steering.generate_batch."""
    return [["out"] * n for _ in contexts]


def _patch_generation(monkeypatch):
    monkeypatch.setattr(R, "generate_batch", _fake_generate_batch)
    monkeypatch.setattr(R.BANK29, "context_token_ids_2389", lambda tok, ctx: [1, 2, 3])


_R12_CONTEXTS = {
    "ctx_a": {"cell": "query_topic", "value_id": "v0", "carrier": "c"},
    "ctx_b": {"cell": "query_topic", "value_id": "v1", "carrier": "c"},
}


def test_r2_batch_kind_parses_cell_grain_batch_ids():
    assert R._batch_kind("gate_filler_swap") == "gate"
    assert R._batch_kind("rest_filler_swap") == "rest"
    assert R._batch_kind("parity_query_topic") == "parity"
    assert R._batch_kind("vllm_query_topic") == "vllm"


def test_r2_generate_anchor_rows_labels_gate_slice_by_batch_kind(tmp_path, monkeypatch):
    """R2 (r2 review): rows carry gate_slice=True exactly for gate_{cell}
    batches (the cell-grain batch_id domain) — parity with the vLLM leg's
    ``cid in gate_id_set``. FAILED at HEAD~: ``batch == "gate"`` was
    permanently False, so every HF anchors row persisted gate_slice=False."""
    cfg = _mk_cfg(tmp_path)
    _patch_generation(monkeypatch)
    for batch, want in (
        ("gate_query_topic", True),
        ("rest_query_topic", False),
        ("parity_query_topic", False),
        ("vllm_query_topic", False),
    ):
        rows, _ctx, _txt = R._generate_anchor_rows(
            cfg, None, None, _R12_CONTEXTS, ["ctx_a", "ctx_b"], 2, batch
        )
        assert rows, batch
        assert all(r["gate_slice"] is want for r in rows), batch


def test_r1_cap_recalibration_reads_cell_grain_shards(tmp_path):
    """R1 (r2 review): the recalibration aggregates the CELL-GRAIN gate
    shards the producers actually write, with paths derived from the done
    manifests. FAILED at HEAD~: the retired ``anchors_gate_w*`` glob matched
    0 of 37 cells, per_cell stayed empty, and a FALSE-COMPLETE report
    (partial: false, zero recalibrations) was adopted downstream."""
    cfg = _mk_cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cell, fp, draws = "query_topic", "fp", 1
    rows = [
        {
            "context_id": f"c{i}",
            "cell": cell,
            "value_id": "v0",
            "draw": 0,
            "gate_slice": True,
            "max_new_tokens": R.cell_max_new_tokens(cell),
            "n_completion_tokens": R.cell_max_new_tokens(cell),
            "cap_hit": True,
            "engine": "hf",
            "text": "t",
        }
        for i in range(4)
    ]
    R._write_jsonl_atomic(cfg.anchors_dir / f"anchors_gate_{cell}_w0.jsonl", rows)
    (cfg.anchors_dir / f"va_anchors_gate_{cell}_w0.pt").write_bytes(b"pt")
    _done_manifest(cfg, f"gate_{cell}", draws=draws, n_rows=4)
    # A stray LEGACY-named worker-stripe file must never enter the aggregate
    # (the HEAD~ glob would have read ONLY this, diluting the trigger).
    R._write_jsonl_atomic(
        cfg.anchors_dir / "anchors_gate_w0.jsonl",
        [{**rows[0], "cap_hit": False, "n_completion_tokens": 1}] * 100,
    )
    recal = R._gate_slice_cap_recalibration(cfg, fp, draws, [cell])
    assert recal == {cell: 2 * R.cell_max_new_tokens(cell)}
    rep = json.loads((cfg.gates_dir / "cap_recalibration.json").read_text())
    assert rep["partial"] is False
    assert rep["per_cell"][cell]["n_rows"] == 4
    assert rep["per_cell"][cell]["n_cap_hit"] == 4


def test_r1_cap_recalibration_missing_shard_fails_loud(tmp_path):
    """A cell that passed the barrier but resolves no manifest+shard is an
    inconsistent store — RuntimeError, never a silent zero-row aggregate."""
    cfg = _mk_cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="inconsistent store"):
        R._gate_shard_paths(cfg, "fp", ["query_topic"], 1)


def _stage_gate_slice(cfg: R.RunConfig, cell: str, fp: str = "fp", draws: int = 1) -> None:
    """Stage one DONE gate cell: 4 all-cap-hit rows + parity-consistent manifest."""
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "context_id": f"c{i}",
            "cell": cell,
            "value_id": "v0",
            "draw": 0,
            "gate_slice": True,
            "max_new_tokens": R.cell_max_new_tokens(cell),
            "n_completion_tokens": R.cell_max_new_tokens(cell),
            "cap_hit": True,
            "engine": "hf",
            "text": "t",
        }
        for i in range(4)
    ]
    R._write_jsonl_atomic(cfg.anchors_dir / f"anchors_gate_{cell}_w0.jsonl", rows)
    (cfg.anchors_dir / f"va_anchors_gate_{cell}_w0.pt").write_bytes(b"pt")
    _done_manifest(cfg, f"gate_{cell}", fp=fp, draws=draws, n_rows=4)


def test_cap_recalibration_foreign_regime_report_recomputes(tmp_path, monkeypatch):
    """Round-5 B (r4 review): the recording site must NOT adopt a
    cap_recalibration.json recorded under a DIFFERENT regime_fp — the report
    records regime_fp precisely for this check, but adoption was
    regime-blind. FAILED at HEAD~: the staged foreign-regime report
    ({cell: 99999}) was returned verbatim; post-fix the recalibration
    recomputes from THIS regime's own gate shards and overwrites."""
    monkeypatch.setattr(R, "CAP_RECAL_TIMEOUT_S", 0.0)  # regression => fast partial, no hang
    cfg = _mk_cfg(tmp_path)
    cell, fp, draws = "query_topic", "fp", 1
    _stage_gate_slice(cfg, cell, fp=fp, draws=draws)
    R._write_json_atomic(
        cfg.gates_dir / "cap_recalibration.json",
        {"regime_fp": "OTHER_regime", "recalibrated": {cell: 99999}},
    )
    recal = R._gate_slice_cap_recalibration(cfg, fp, draws, [cell])
    assert recal == {cell: 2 * R.cell_max_new_tokens(cell)}  # recomputed, never 99999
    rep = json.loads((cfg.gates_dir / "cap_recalibration.json").read_text())
    assert rep["regime_fp"] == fp  # overwritten under THIS regime


def _cap_recal_repro(cfg: R.RunConfig, **over) -> dict:
    """The repro regime subset the round-5 F consumption guard reads."""
    rep = {
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "tiny": cfg.tiny,
        "smoke": cfg.smoke,
    }
    rep.update(over)
    return rep


def test_cap_recalibration_same_regime_report_adopted(tmp_path, monkeypatch):
    """Positive control for round-5 B: a SAME-regime report IS adopted
    verbatim (idempotent resume — no recompute, no rewrite)."""
    monkeypatch.setattr(R, "CAP_RECAL_TIMEOUT_S", 0.0)  # regression => fast partial, no hang
    cfg = _mk_cfg(tmp_path)
    cell, fp = "query_topic", "fp"
    path = cfg.gates_dir / "cap_recalibration.json"
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        path,
        {"regime_fp": fp, "recalibrated": {cell: 1234}, "repro": _cap_recal_repro(cfg)},
    )
    before = path.read_bytes()
    # deliberately NO staged shards: adoption must return BEFORE the barrier
    recal = R._gate_slice_cap_recalibration(cfg, fp, 1, [cell])
    assert recal == {cell: 1234}
    assert path.read_bytes() == before


def test_r5f_cap_recal_consumption_rejects_regime_foreign_report(tmp_path):
    """Round-5 F (concern cap-recal-consumer-regime-bypass): the CONSUMPTION
    signature — ``_load_cap_recalibration(cfg)`` with no regime_fp, the exact
    call grid (:4084) / stage2 (:7123) / the vLLM rest leg
    (vllm_anchors.py:604) make — must NOT adopt a report whose recorded
    repro REGIME differs from this run's. dispatch.sh's standalone ``grid)``
    / ``stage2)`` arms never traverse the recording barrier and OUT_ROOT
    defaults to a shared path, so a prior --smoke/--tiny recalibration sits
    there un-overwritten. FAILED at HEAD~: all three consumption sites
    adopted the smoke-regime {cell: 99999} verbatim."""
    prod = _mk_cfg(tmp_path, tiny=False, smoke=False)
    cell = "query_topic"
    prod.gates_dir.mkdir(parents=True, exist_ok=True)
    path = prod.gates_dir / "cap_recalibration.json"
    # (a) smoke/tiny-regime evidence in a shared out_root
    smoke_cfg = _mk_cfg(tmp_path, tiny=True, smoke=True)
    R._write_json_atomic(
        path,
        {
            "regime_fp": "smoke_fp",
            "recalibrated": {cell: 99999},
            "repro": _cap_recal_repro(smoke_cfg),
        },
    )
    assert R._load_cap_recalibration(prod) is None
    # (b) foreign model@revision
    R._write_json_atomic(
        path,
        {
            "regime_fp": "fp",
            "recalibrated": {cell: 99999},
            "repro": _cap_recal_repro(prod, model_id="other/model"),
        },
    )
    assert R._load_cap_recalibration(prod) is None
    # (c) a repro-less (pre-round-5) report cannot prove its regime
    R._write_json_atomic(path, {"regime_fp": "fp", "recalibrated": {cell: 99999}})
    assert R._load_cap_recalibration(prod) is None


def test_r5f_cap_recal_consumption_adopts_same_regime_report(tmp_path):
    """Positive control for round-5 F: a regime-matched report IS adopted by
    the consumption signature (no false refusal on the healthy path)."""
    cfg = _mk_cfg(tmp_path)
    cell = "query_topic"
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg.gates_dir / "cap_recalibration.json",
        {"regime_fp": "any_fp", "recalibrated": {cell: 4096}, "repro": _cap_recal_repro(cfg)},
    )
    assert R._load_cap_recalibration(cfg) == {cell: 4096}


def test_gate_shard_paths_skips_row_count_mismatched_manifest(tmp_path):
    """Round-5 minor: _gate_shard_paths mirrors _anchor_cell_done's row-count
    parity — a manifest whose n_rows disagrees with the shard's realized row
    count is a partial/foreign write and is SKIPPED (first VALID wins)."""
    cfg = _mk_cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cell = "query_topic"
    row = {"context_id": "c0", "cell": cell, "cap_hit": True, "text": "t"}
    R._write_jsonl_atomic(cfg.anchors_dir / f"anchors_gate_{cell}_w0.jsonl", [row])
    _done_manifest(cfg, f"gate_{cell}", w=0, draws=1, n_rows=4)  # claims 4, shard has 1
    with pytest.raises(RuntimeError, match="inconsistent store"):
        R._gate_shard_paths(cfg, "fp", [cell], 1)
    # a parity-consistent sibling manifest resolves (first VALID wins)
    R._write_jsonl_atomic(cfg.anchors_dir / f"anchors_gate_{cell}_w1.jsonl", [row] * 4)
    _done_manifest(cfg, f"gate_{cell}", w=1, draws=1, n_rows=4)
    paths = R._gate_shard_paths(cfg, "fp", [cell], 1)
    assert paths == [cfg.anchors_dir / f"anchors_gate_{cell}_w1.jsonl"]


def test_r1_r2_post_recalibration_two_cap_store_is_valid_capregen_basis(tmp_path, monkeypatch):
    """The R1<->R2 interaction (r2 review ORDERING constraint): after a
    gate-slice recalibration one cell legitimately holds the RAISED cap in
    its gate batch and the BASE cap in its rest batch. With correct
    gate_slice labeling each (cell, batch) bucket holds exactly ONE cap and
    ``_validate_breach_basis`` ACCEPTS the report. At HEAD~ (R1 fixed alone)
    both caps collapsed into the single "rest" bucket -> mixed-bucket
    refusal -> the anchors capregen basis WEDGED on the plan's designed
    path."""
    cfg = _mk_cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    _patch_generation(monkeypatch)
    cell = "query_topic"
    base_cap = R.cell_max_new_tokens(cell)
    raised = 2 * base_cap
    gate_rows, _c, _t = R._generate_anchor_rows(
        cfg, None, None, _R12_CONTEXTS, ["ctx_a", "ctx_b"], 2, f"gate_{cell}", {cell: raised}
    )
    rest_rows, _c, _t = R._generate_anchor_rows(
        cfg, None, None, _R12_CONTEXTS, ["ctx_a", "ctx_b"], 2, f"rest_{cell}"
    )
    for r in gate_rows + rest_rows:
        r["n_completion_tokens"] = 5
        r["cap_hit"] = False
    p_gate = cfg.anchors_dir / f"anchors_gate_{cell}_w0.jsonl"
    p_rest = cfg.anchors_dir / f"anchors_rest_{cell}_w0.jsonl"
    R._write_jsonl_atomic(p_gate, gate_rows)
    R._write_jsonl_atomic(p_rest, rest_rows)
    rep = R.compute_cap_hit_report(
        [p_gate, p_rest],
        R.MAX_NEW_TOKENS,
        scope="anchors",
        expected_shards={p_gate.name, p_rest.name},
    )
    assert rep["per_cell"][cell]["realized_caps_by_batch"] == {
        "gate": [raised],
        "rest": [base_cap],
    }
    # The wedge check: a post-recalibration two-cap store is a VALID basis.
    R._validate_breach_basis(rep, tmp_path / "rep.json", "anchors", cfg)
    # Counterfactual (R1 fixed WITHOUT R2 — the banned intermediate state):
    # mislabel the gate rows the way the pre-R2 writer did and the SAME store
    # trips the mixed-bucket refusal, i.e. the capregen basis wedges.
    for r in gate_rows:
        r["gate_slice"] = False
    R._write_jsonl_atomic(p_gate, gate_rows)
    rep_bad = R.compute_cap_hit_report(
        [p_gate, p_rest],
        R.MAX_NEW_TOKENS,
        scope="anchors",
        expected_shards={p_gate.name, p_rest.name},
    )
    with pytest.raises(RuntimeError, match="MIXED realized caps"):
        R._validate_breach_basis(rep_bad, tmp_path / "rep.json", "anchors", cfg)


def test_b7_chunk_size_median_regression(tmp_path):
    """B7 (r1) realized-property pin: cell-bucketed generate chunks stay near
    full batch — median chunk size >= min(B, 8) for gate AND rest context
    orders at both pilot candidates (round-2 live enumeration: gate median
    11; rest 16 @ B=16 / 25 @ B=32). A future re-fragmentation of the claim
    grain cannot land silently."""
    cfg = _mk_cfg(tmp_path, smoke=False)
    _frozen_manifest(cfg)
    gate_ids, rest_ids, contexts = R._anchor_context_order(cfg)
    for order in (gate_ids, rest_ids):
        assert order
        for b in R.PILOT_GEN_BATCH_CANDIDATES:
            sizes = sorted(
                len(chunk) for _cell, chunk in R._cell_bucketed_chunks(contexts, order, b)
            )
            median = sizes[len(sizes) // 2]
            assert median >= min(b, 8), (b, median, sizes[:8])


# ----------------------------- R5 (r2 review): pilot re-entry is LOSSLESS


def _pilot_report(
    cfg: R.RunConfig,
    verdict: str = "ACCEPT",
    sel: int = 16,
    report_over: dict | None = None,
    **repro_over,
) -> None:
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    cur = R._repro(cfg)
    rec = {
        "verdict": verdict,
        "gen_batch_selected": sel,
        "gen_batch_candidates": [16, 32],
        # round-5 C: the runtime-domain fields _reusable_pilot_report checks
        "num_workers": max(1, cfg.num_workers),
        "gpu_name": R._pilot_gpu_name(),
        # round-5 J: memory identity + floor constants + runtime identity
        "gpu_total_mem_gib": R._pilot_gpu_mem_gib(),
        "hbm_headroom_floor_gib": R.PILOT_HBM_HEADROOM_GIB,
        "refusal_threshold_h": R.PILOT_REFUSAL_MULT * cfg.planned_wall_h,
        "planned_total_wall_h": cfg.planned_wall_h,
        "accept_threshold_h": R.PILOT_ACCEPT_WALL_H,
        "repro": {
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
            "smoke": cfg.smoke,
            "tiny": cfg.tiny,
            "torch": cur["torch"],
            "transformers": cur["transformers"],
            "git_commit": cur["git_commit"],
            **repro_over,
        },
    }
    rec.update(report_over or {})
    R._write_json_atomic(cfg.gates_dir / "pilot_gate_report.json", rec)


def _bank_sentinel(monkeypatch):
    """Trip-wire at the first post-guard step: reaching it means the pilot
    phase did NOT skip re-measurement."""

    def _boom(cfg):
        raise AssertionError("pilot re-measurement reached _load_bank")

    monkeypatch.setattr(R, "_load_bank", _boom)


def test_r5_pilot_reuses_matching_report(tmp_path, monkeypatch):
    """R5 (r2 review): with a regime-matched ACCEPT report present, the pilot
    phase performs NO measurement (plan §9 lossless same-command resume) and
    exits adopting the recorded gen_batch. FAILED at HEAD~: phase_grid's
    --pilot path ran the full three-regime pilot unconditionally, and a
    re-measured 16<->32 flip would rewrite regime_fingerprint and quarantine
    every banked anchors/grid shard."""
    cfg = _mk_cfg(tmp_path, pilot=True)
    _pilot_report(cfg)
    _bank_sentinel(monkeypatch)
    assert R.phase_grid(cfg) == R.RC_OK


def test_r5_pilot_refuse_report_stands(tmp_path, monkeypatch):
    cfg = _mk_cfg(tmp_path, pilot=True)
    _pilot_report(cfg, verdict="REFUSE")
    _bank_sentinel(monkeypatch)
    assert R.phase_grid(cfg) == R.RC_PILOT_GATE


def test_r5_pilot_refuse_report_forced_past_proceeds_without_remeasure(tmp_path, monkeypatch):
    cfg = _mk_cfg(tmp_path, pilot=True, force_past_halt_gates=True)
    _pilot_report(cfg, verdict="REFUSE")
    _bank_sentinel(monkeypatch)
    assert R.phase_grid(cfg) == R.RC_OK


def test_r5_pilot_foreign_report_raises(tmp_path, monkeypatch):
    """A present-but-foreign report (other model / regime) is never silently
    re-measured over — fail loud; --force is the deliberate re-measure."""
    cfg = _mk_cfg(tmp_path, pilot=True)
    _pilot_report(cfg, model_id="other/model")
    _bank_sentinel(monkeypatch)
    with pytest.raises(RuntimeError, match="FOREIGN"):
        R.phase_grid(cfg)


def test_r5_pilot_force_remeasures(tmp_path, monkeypatch):
    """--force bypasses adoption and re-measures (the sentinel IS reached)."""
    cfg = _mk_cfg(tmp_path, pilot=True, force=True)
    _pilot_report(cfg)
    _bank_sentinel(monkeypatch)
    with pytest.raises(AssertionError, match="re-measurement reached"):
        R.phase_grid(cfg)


def test_r5_pilot_runtime_domain_mismatch_raises(tmp_path, monkeypatch):
    """Round-5 C (concern pilot-reuse-runtime-domain): a report measured at a
    different worker width / GPU lane / candidate set / wall threshold /
    planned wall is FOREIGN — its per-phase wall projections, argmin
    selection, and verdict banding do not transfer. FAILED at HEAD~: all
    five mismatches adopted silently. Round-5 J extends the domain: device
    MEMORY identity, the recorded floor constants, and torch/transformers
    runtime identity (candidates {gen_batch_candidates: [8, 64]} would now
    also fail the sel-in-band check — the FOREIGN raise fires first)."""
    _bank_sentinel(monkeypatch)
    for over in (
        {"num_workers": 8},
        {"gpu_name": "NVIDIA H200"},
        {"gen_batch_candidates": [8, 64]},
        {"accept_threshold_h": 80.0},
        {"planned_total_wall_h": 77.0},
        # round-5 J + the r4 floor-recheck minor
        {"gpu_total_mem_gib": 141.1},
        {"hbm_headroom_floor_gib": 20.0},
        {"refusal_threshold_h": 1.0},
    ):
        cfg = _mk_cfg(tmp_path, pilot=True)
        _pilot_report(cfg, report_over=over)
        with pytest.raises(RuntimeError, match="FOREIGN"):
            R.phase_grid(cfg)
    # runtime identity (repro fields, round-5 J): torch/transformers bind HARD
    for repro_over in ({"torch": "0.0.0+other"}, {"transformers": "0.0.0"}):
        cfg = _mk_cfg(tmp_path, pilot=True)
        _pilot_report(cfg, **repro_over)
        with pytest.raises(RuntimeError, match="FOREIGN"):
            R.phase_grid(cfg)


def test_r5j_adoption_path_validates_runtime_domain(tmp_path):
    """Round-5 J (concern pilot-reuse-runtime-domain): the NORMAL adoption
    path — _adopt_pilot_gen_batch, the route grid/capregen/stage2 and the
    vLLM legs take — runs the SAME runtime-domain validation as the pilot
    phase's own reuse decision. FAILED at HEAD~: _pilot_selected_gen_batch
    read the report directly and adopted a foreign-width/lane report's
    gen_batch silently."""
    cfg = _mk_cfg(tmp_path, pilot=False, gen_batch=16)
    _pilot_report(cfg, sel=32, report_over={"num_workers": 8})  # cfg width is 1
    with pytest.raises(RuntimeError, match="FOREIGN"):
        R._adopt_pilot_gen_batch(cfg)


def test_r5j_adoption_path_adopts_domain_matched_report(tmp_path, caplog):
    """Positive control for round-5 J: a full domain-matched report still
    drives adoption through the normal path (no false refusal), and a git
    commit delta alone is WARN-only (crash-fix commits legitimately land
    between the pilot and a same-command resume)."""
    cfg = _mk_cfg(tmp_path, pilot=False, gen_batch=16)
    _pilot_report(cfg, sel=32)
    adopted = R._adopt_pilot_gen_batch(cfg)
    assert adopted.gen_batch == 32
    cfg2 = _mk_cfg(tmp_path, pilot=False, gen_batch=16)
    _pilot_report(cfg2, sel=32, git_commit="0" * 40)
    with caplog.at_level("WARNING"):
        adopted2 = R._adopt_pilot_gen_batch(cfg2)
    assert adopted2.gen_batch == 32
    assert any("WARN-only" in r.message for r in caplog.records)


def test_r5_pilot_unrecorded_gpu_lane_raises(tmp_path, monkeypatch):
    """A report predating the runtime-domain fields cannot prove its lane —
    FOREIGN (a deliberate re-measure needs --force), never a silent adopt."""
    cfg = _mk_cfg(tmp_path, pilot=True)
    _pilot_report(cfg)
    path = cfg.gates_dir / "pilot_gate_report.json"
    rec = json.loads(path.read_text())
    rec.pop("gpu_name")
    R._write_json_atomic(path, rec)
    _bank_sentinel(monkeypatch)
    with pytest.raises(RuntimeError, match="gpu_name"):
        R.phase_grid(cfg)


# ------- Round 6 (concern pilot-reuse-runtime-domain): dispatcher shape


def test_r6_dispatch_capregen_anchors_arms_thread_worker_width():
    """Round-6 (concern pilot-reuse-runtime-domain): the capregen-anchors
    dispatch arms thread the dispatcher's realized worker width into run.py.
    Without it the leg runs at the parser-default width (1), and the
    round-5-J adoption path (_adopt_pilot_gen_batch ->
    _reusable_pilot_report) FOREIGN-raises against the width-8 pilot report
    BEFORE any regeneration — the registered >2%/cell cap-hit remedy was
    unrunnable as shipped. FAILED at HEAD~: neither arm carried
    --num-workers (the width/FOREIGN adoption semantics themselves are
    pinned by test_r5j_adoption_path_validates_runtime_domain)."""
    sh = (REPO_ROOT / "scripts" / "issue2389_dispatch.sh").read_text()
    for arm in ("capregen-anchors-gate", "capregen-anchors-rest"):
        m = re.search(rf"^  {arm}\)\n(?P<body>(?:.*\n)*?)^\s*;;", sh, re.M)
        assert m is not None, f"case arm {arm}) not found in issue2389_dispatch.sh"
        assert '--num-workers "$NUM_WORKERS"' in m.group("body"), (
            f'{arm} does not thread --num-workers "$NUM_WORKERS"'
        )
