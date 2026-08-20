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

import sys
from pathlib import Path

import pytest

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
