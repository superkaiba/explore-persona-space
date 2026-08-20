"""Issue #2389 analysis fork — fork-delta unit tests.

Covers exactly the deltas vs the parent ``issue2329_analysis.py`` (shared
machinery is exercised by the parent lineage's own suites):

- 64-layer constants (READ_LAYER 59 + the layer-61 exploratory companion) and
  the stale-store guard at the new read layer;
- ce-only derived family ceilings + the #2329 realized-m report values;
- the M-N1 probe device seam (CPU-default behavior preserved; allocations
  follow the Gram's device);
- the S2 two-leg transfer read (PRIMARY >= 12-eligible verdict branch +
  DESCRIPTIVE all-shared companion) end-to-end on synthetic fixtures built
  over the REAL bank cells (the #2162-table 31-unit count-assert included).

All fixtures are synthetic/tmp — no committed eval_results reads (no
sparse-cone additions needed).
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_analysis as A  # noqa: E402

BANK = A.BANK


# ── constants + registration ──────────────────────────────────────────


def test_fork_layer_constants():
    assert A.READ_LAYER == 59
    assert A.COMPANION_READ_LAYER == 61
    # 59 is a FULL-attention layer (= 3 mod 4); 61 is linear — the disclosed
    # layer-TYPE asymmetry (plan Div 3).
    assert A.READ_LAYER % 4 == 3
    assert A.COMPANION_READ_LAYER % 4 != 3


def test_ce_only_derived_ceilings():
    assert A.FAMILY_CEILING_M == {"P1": 16, "P2": 8, "P3": 14}
    assert sum(A.FAMILY_CEILING_M.values()) == 38  # 39 cells minus filler_swap
    assert A.PARENT_REALIZED_M == {"P1": 28, "P2": 12, "P3": 27}
    assert A.PARENT_P1_UNIT_COUNT == 31
    assert A.TRANSFER_RHO_REF == 0.3


def test_probe_iterates_run_driver_slots():
    assert A.R.SLOTS == ("ce",)
    assert "for slot in R.SLOTS" in inspect.getsource(A.step_probe)


def test_read_layer_index_guards():
    reg: dict = {}
    full = list(range(64))
    assert A._read_layer_index(full, Path("s0.pt"), reg) == 59
    # tiny capture keeps the last-layer fallback
    assert A._read_layer_index([0, 21, 42, 63], Path("s1.pt"), {}) == 3
    # a stale prior-model store (32 layers, no 59) fails loud
    with pytest.raises(AssertionError, match="stale prior-model"):
        A._read_layer_index(list(range(32)), Path("s2.pt"), {})


# ── M-N1 probe device seam (CPU default preserved) ────────────────────


def test_kernel_logistic_auc_follows_gram_device_cpu():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 6)
    gram = torch.einsum("lnh,lmh->lnm", x, x) / x.shape[-1]
    labels = torch.tensor([[0, 1] * 4, [1, 0] * 4])
    masks = torch.stack([torch.arange(8) % 2 == 0, torch.arange(8) % 2 == 1])
    auc = A.kernel_logistic_auc(gram, labels, masks, epochs=5)
    assert auc.shape == (2, 3)
    assert auc.device.type == "cpu"
    assert bool(((auc >= 0) & (auc <= 1)).all())


def test_vp_data_device_kwarg_and_shapes():
    cell = "fact_user_name"
    carriers = list(BANK.carriers_for(cell))[:3]
    (va, vb) = BANK.cell_pairs_per_carrier(cell)[0]
    layers = [0, 59]
    recs = {
        BANK.context_id(cell, v, c): {"v_ce": torch.randn(len(layers), 16)}
        for c in carriers
        for v in (va, vb)
    }
    gram, y, groups = A._vp_data(recs, layers, cell, "ce", va, vb, carriers, device="cpu")
    n = 2 * len(carriers)
    assert gram.shape == (len(layers), n, n) and gram.device.type == "cpu"
    assert y.shape == (n,) and groups.shape == (n,)
    assert sorted(y.tolist()) == [0] * len(carriers) + [1] * len(carriers)


def test_auc_ranked_arange_is_device_agnostic():
    src = inspect.getsource(A._auc_ranked)
    assert "device=scores.device" in src


# ── S2 transfer verdict branch ─────────────────────────────────────────


def test_transfer_verdict_branches():
    v = A._transfer_verdict
    assert v(13, 0.1, 0.9) == "confirmed"
    assert v(13, 0.05, 0.25) == "confirmed (positive but below the parent lineage's band)"
    assert v(13, -0.2, 0.25) == "falsified"
    assert v(13, -0.6, -0.1) == "falsified-strict-reversal"
    assert v(13, -0.2, 0.6).startswith("no-verdict")
    # eligibility collapse routes to no-verdict REGARDLESS of extreme rho
    assert v(2, 0.5, 0.9).startswith("no-verdict")
    assert v(0, None, None).startswith("no-verdict")


def test_transfer_leg_small_n_reports_none():
    leg = A._transfer_leg([("cellA", "ce")], {("cellA", "ce"): [0.5]}, {("cellA", "ce"): [0.4]})
    assert leg["n_units"] == 1
    assert leg["rho"] is None and leg["ci95_pair_clustered"] is None


# ── S2 two-leg transfer end-to-end (synthetic fixtures, real bank cells) ──


@functools.lru_cache(maxsize=1)
def _p1_units() -> tuple[list[str], list[str]]:
    p1_ce = sorted(c for c in BANK.all_cells() if A.family_of(c, "ce") == "P1")
    p1_pe = sorted(c for c in BANK.all_cells() if A.family_of(c, "pe") == "P1")
    assert len(p1_ce) == 16 and len(p1_ce) + len(p1_pe) == 31
    return p1_ce, p1_pe


# The plan's parent-side sub-floor cells (S2): below the n>=12 eligibility
# floor in the #2162 table — the DESCRIPTIVE leg's extra 3 units.
_SUBFLOOR = {"user_emotion": 1, "icl_task_mapping": 7, "refusal_boundary": 8}


def _rows_for_unit(cell: str, slot: str, n_pairs: int, mean: float) -> list[dict]:
    rng = np.random.default_rng(abs(hash((cell, slot))) % 2**32)
    return [
        {
            "cell": cell,
            "slot": slot,
            "arm": "steered",
            "family": "P1",
            "pair_id": f"{cell}|{slot}|{i}",
            "f_beh": float(mean + rng.normal(0, 0.01)),
            "separation": 0.8,
        }
        for i in range(n_pairs)
    ]


def _stats_fixture(cells: list[str], slots: list[str]) -> dict:
    per_cell = {
        f"{c}|{s}": {
            "cell": c,
            "slot": s,
            "family": "P1",
            "untestable_causal": False,
            "holm_pass": True,
            "disjoint_both_nulls": True,
        }
        for c in cells
        for s in slots
    }
    return {"per_cell": per_cell, "families": {"P1": len(per_cell)}, "family_m": {}}


def _bank_fixture(cells: list[str]) -> dict:
    return {
        "token_identity": {"per_cell": {c: {"n_dropped": 0} for c in cells}},
        "repaired_cells": [],
    }


def test_step_transfer_two_legs_end_to_end(tmp_path):
    p1_ce, p1_pe = _p1_units()
    parent_dir = tmp_path / "parent"
    out_dir = tmp_path / "out"
    parent_dir.mkdir()
    out_dir.mkdir()

    # Parent (#2162-shaped) table: all 31 P1 units (ce + pe); 3 named ce cells
    # below the eligibility floor; monotone per-cell means (rho -> 1).
    parent_rows: list[dict] = []
    for i, c in enumerate(p1_ce):
        parent_rows.extend(_rows_for_unit(c, "ce", _SUBFLOOR.get(c, 14), 0.1 + 0.05 * i))
    for c in p1_pe:
        parent_rows.extend(_rows_for_unit(c, "pe", 14, 0.2))
    (parent_dir / "f_cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in parent_rows), encoding="utf-8"
    )
    (parent_dir / "stats.json").write_text(
        json.dumps(_stats_fixture(p1_ce, ["ce", "pe"])), encoding="utf-8"
    )

    # Child (#2389, ce-only): 13 kept pairs per ce cell, means tracking the
    # parent's order.
    child_rows: list[dict] = []
    for i, c in enumerate(p1_ce):
        child_rows.extend(_rows_for_unit(c, "ce", 13, 0.15 + 0.05 * i))
    (out_dir / "f_cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in child_rows), encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(json.dumps(_stats_fixture(p1_ce, ["ce"])), encoding="utf-8")

    bank_json = tmp_path / "bank.json"
    bank_json.write_text(json.dumps(_bank_fixture(p1_ce)), encoding="utf-8")

    args = argparse.Namespace(
        parent_f_metrics=parent_dir,
        out_dir=out_dir,
        bank_json=bank_json,
    )
    A.step_transfer(args)
    out = json.loads((out_dir / "transfer.json").read_text())

    # PRIMARY: the 13 dual-eligible units (16 shared minus the 3 parent-side
    # sub-floor cells), verdict-bearing, confirmed on the monotone fixture.
    assert out["primary"]["n_units"] == 13
    assert set(out["primary"]["units"]).isdisjoint({f"{c}|ce" for c in _SUBFLOOR})
    assert out["primary"]["rho"] == pytest.approx(1.0)
    assert out["primary"]["verdict"].startswith("confirmed")
    assert out["primary"]["eligibility_floor"] == A.SURVIVAL_FLOOR
    # DESCRIPTIVE: all 16 shared ce units, labelled, never verdict-bearing.
    assert out["descriptive_all_shared"]["n_units"] == 16
    assert "DESCRIPTIVE" in out["descriptive_all_shared"]["label"]
    assert "verdict" not in out["descriptive_all_shared"]
    # ce-only: no pe unit leaks into either leg.
    assert all(u.endswith("|ce") for u in out["descriptive_all_shared"]["units"])
    # per-unit eligibility flags + renamed mean fields.
    per_unit = {f"{r['cell']}|{r['slot']}": r for r in out["per_unit"]}
    assert len(per_unit) == 16
    for c, n in _SUBFLOOR.items():
        assert per_unit[f"{c}|ce"]["primary_eligible"] is False
        assert per_unit[f"{c}|ce"]["n_pairs_2162"] == n
    assert all("f_beh_2389_mean" in r and "f_beh_2162_mean" in r for r in out["per_unit"])
    # 2x2 verdict transfer over the shared per_cell keys (ce-only child).
    assert out["verdict_transfer"]["n_shared_units"] == 16
    assert out["family_m_ceilings"] == A.FAMILY_CEILING_M


def test_step_fact_profile_end_to_end(tmp_path):
    """Full-layer F_act profile (plan §6 fact_profile.jsonl): a pair whose
    steered V_a equals the ceiling mean scores f_act == 1.0 at EVERY layer
    (vectorized batch dim = layers); a pair with < 2 floor draws is skipped."""
    cell = "fact_user_name"
    p = next(q for q in BANK.build_pairs() if q.cell == cell)
    # A DIFFERENT cell for the skipped pair — same-cell pairs share anchor
    # contexts (p.b can equal a sibling's .a), which would trip the
    # duplicate-key assert in the fixture.
    p2 = next(q for q in BANK.build_pairs() if q.cell == "persona_prompted")
    layers = [0, 59, 61, 63]
    ll, hh = len(layers), 8

    bank_json = tmp_path / "bank.json"
    bank_json.write_text(
        json.dumps({"dropped_pairs": [], "token_identity": {"n_intact": len(BANK.build_pairs())}}),
        encoding="utf-8",
    )
    rollouts = tmp_path / "rollouts"
    va_dir = tmp_path / "va"
    anchors = tmp_path / "anchors"
    out_dir = tmp_path / "out"
    for d in (rollouts, va_dir, anchors, out_dir):
        d.mkdir()

    def _grid_row(pair, draw):
        return {
            "block_key": "blk0",
            "pair_id": pair.pair_id,
            "cell": pair.cell,
            "slot": "ce",
            "arm": "steered",
            "draw": draw,
            "text": "t",
            "context_id": pair.a,
        }

    grid = [_grid_row(p, 0), _grid_row(p, 1), _grid_row(p2, 0)]
    (rollouts / "shard_0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in grid), encoding="utf-8"
    )

    f_vec = torch.full((ll, hh), 0.5)
    c_vec = torch.full((ll, hh), 1.0)
    torch.save(
        {
            "block_key": "blk0",
            "layers": layers,
            "va_span": torch.stack([c_vec, c_vec, c_vec]).to(torch.float16),
            "index": [
                {"pair_id": p.pair_id, "context_a": p.a, "draw": 0},
                {"pair_id": p.pair_id, "context_a": p.a, "draw": 1},
                {"pair_id": p2.pair_id, "context_a": p2.a, "draw": 0},
            ],
            "empty_rows": [],
        },
        va_dir / "shard_0.pt",
    )
    # p: 2 floor + 2 ceiling draws; p2: only 1 floor draw (skipped, < 2).
    torch.save(
        {
            "layers": layers,
            "va_span": torch.stack([f_vec, f_vec, c_vec, c_vec, f_vec]).to(torch.float16),
            "index": [
                {"context_id": p.a, "draw": 0},
                {"context_id": p.a, "draw": 1},
                {"context_id": p.b, "draw": 0},
                {"context_id": p.b, "draw": 1},
                {"context_id": p2.a, "draw": 0},
            ],
            "empty_rows": [],
        },
        anchors / "va_anchors_0.pt",
    )

    args = argparse.Namespace(
        bank_json=bank_json,
        rollouts_dir=rollouts,
        anchors_dir=anchors,
        va_dir=va_dir,
        out_dir=out_dir,
    )
    A.step_fact_profile(args)
    rows = [
        json.loads(line)
        for line in (out_dir / "fact_profile.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 1  # p2 skipped (1 floor draw)
    row = rows[0]
    assert (row["cell"], row["slot"], row["n_pairs"]) == (cell, "ce", 1)
    assert row["layers"] == layers
    assert len(row["f_act_mean_per_layer"]) == len(layers)
    for v in row["f_act_mean_per_layer"]:
        assert v == pytest.approx(1.0, abs=1e-6)
    assert row["read_layer"] == 59 and row["companion_layer"] == 61
    per_pair = row["per_pair"]
    assert per_pair[0]["pair_id"] == p.pair_id
    assert per_pair[0]["f_act_read"] == pytest.approx(1.0, abs=1e-6)
    assert per_pair[0]["f_act_companion"] == pytest.approx(1.0, abs=1e-6)
    assert "fact-profile" in A.STEPS


def test_step_transfer_asserts_parent_unit_count(tmp_path):
    p1_ce, _ = _p1_units()
    parent_dir = tmp_path / "parent"
    out_dir = tmp_path / "out"
    parent_dir.mkdir()
    out_dir.mkdir()
    # ce-only parent table (16 units) violates the committed-table count-assert.
    rows = [r for c in p1_ce for r in _rows_for_unit(c, "ce", 14, 0.3)]
    (parent_dir / "f_cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    args = argparse.Namespace(
        parent_f_metrics=parent_dir, out_dir=out_dir, bank_json=tmp_path / "b.json"
    )
    with pytest.raises(AssertionError, match="drifted"):
        A.step_transfer(args)
