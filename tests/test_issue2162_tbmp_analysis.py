"""Issue #2162 tbmp analysis pins (CPU, no artifacts needed).

Covers the plan-§12 assumption-8 verification surface: ``tb_pair_cells``
re-aggregates the SAME per-rubric judge scores into BOTH spaces (netted +
target-descriptor-only) with hand-computed expected values, and
``_assert_parent_f_parity`` accepts a committed table the recompute
reproduces / raises on a drifted one (the runtime 'both ways' join check
``step parent-ref`` runs against the staged parent scores).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_tbmp_analysis as M  # noqa: E402


def _pair(pair_id: str, cell: str) -> SimpleNamespace:
    return SimpleNamespace(
        pair_id=pair_id,
        cell=cell,
        carrier="d1",
        value_a="A",
        value_b="B",
        a=f"{pair_id}__a",
        b=f"{pair_id}__b",
    )


def _grid_row(pair_id: str, cell: str, draw: int, arm: str = "steered") -> dict:
    return {
        "pair_id": pair_id,
        "cell": cell,
        "slot": "ce",
        "arm": arm,
        "draw": draw,
        "block_key": f"bk_{pair_id}",
        "context_id": f"{pair_id}__a",
        "text": "x",
        "cap_hit": False,
    }


def _score_key(kind: str, block_key: str, pair_id: str, draw: int, side: str | None) -> str:
    if side is None:
        return M.A.J.J94._item_id("c", f"g|{block_key}|{pair_id}|{draw}")
    return M.A.J.J94._item_id("g", f"g|{block_key}|{pair_id}|{draw}|{side}")


def _fixture(pair_id: str, cell: str) -> tuple[list[dict], dict, dict, dict]:
    rows = [_grid_row(pair_id, cell, d) for d in (0, 1)]
    bk = f"bk_{pair_id}"
    scores = {
        _score_key("c", bk, pair_id, 0, None): 90.0,
        _score_key("c", bk, pair_id, 1, None): 90.0,
        _score_key("g", bk, pair_id, 0, "a"): 20.0,
        _score_key("g", bk, pair_id, 0, "b"): 80.0,
        _score_key("g", bk, pair_id, 1, "a"): 40.0,
        _score_key("g", bk, pair_id, 1, "b"): 60.0,
    }
    committed = {
        pair_id: {
            "cell": cell,
            "delta_floor_mean": -0.2,
            "delta_ceiling_mean": 0.6,
            "separation": 0.8,
        }
    }
    channels = {pair_id: {"b_floor": 0.3, "b_ceiling": 0.9}}
    return rows, scores, committed, channels


def test_dual_space_reaggregation_hand_computed():
    """Netted and target-only F from the SAME scores — assumption 8's 'both
    ways': draws (20,80)/(40,60) => mean netted delta 0.4, mean B 0.7."""
    rows, scores, committed, channels = _fixture("p1", "persona_prompted")
    pairs_by_id = {"p1": _pair("p1", "persona_prompted")}
    tables = M.tb_pair_cells(rows, scores, committed, channels, pairs_by_id)
    (rec,) = tables["steered"]
    assert rec["n_draws"] == 2 and rec["n_coherent"] == 2 and rec["n_scored"] == 2
    assert rec["f_netted"] == pytest.approx((0.4 - (-0.2)) / (0.6 - (-0.2)))  # 0.75
    assert rec["f_target_only"] == pytest.approx((0.7 - 0.3) / (0.9 - 0.3))  # 0.6667
    # persona cells register target-only; raw move is B-channel direction * lift
    assert rec["registered_space"] == "target_only"
    assert rec["f_beh"] == pytest.approx(rec["f_target_only"])
    assert rec["raw_move_registered"] == pytest.approx(0.7 - 0.3)


def test_registered_space_split():
    assert M.registered_space("persona_prompted") == "target_only"
    assert M.registered_space("instr_format") == "netted"


def test_incoherent_draws_dropped():
    rows, scores, committed, channels = _fixture("p2", "instr_format")
    bk = "bk_p2"
    scores[_score_key("c", bk, "p2", 1, None)] = 10.0  # draw 1 incoherent
    tables = M.tb_pair_cells(rows, scores, committed, channels, {"p2": _pair("p2", "instr_format")})
    (rec,) = tables["steered"]
    assert rec["n_coherent"] == 1 and rec["n_scored"] == 1
    assert rec["f_netted"] == pytest.approx((0.6 - (-0.2)) / 0.8)  # draw-0 delta only
    assert rec["registered_space"] == "netted"
    assert rec["f_beh"] == pytest.approx(rec["f_netted"])


def _parity_fixture(tmp_path: Path, drift: float = 0.0) -> tuple[dict, Path]:
    n = M.SURVIVAL_FLOOR
    tables = {"steered": [], "shuffled": [], "crosstype": []}
    committed_rows = []
    for i in range(n):
        pid = f"pp{i}"
        tables["steered"].append(
            {"pair_id": pid, "cell": "instr_format", "slot": "ce", "f_netted": 0.5 + 0.01 * i}
        )
        committed_rows.append(
            {
                "pair_id": pid,
                "cell": "instr_format",
                "slot": "ce",
                "f_beh": 0.5 + 0.01 * i + (drift if i == 0 else 0.0),
            }
        )
    metrics = tmp_path / "f_metrics"
    metrics.mkdir(parents=True)
    with (metrics / "f_cells.jsonl").open("w") as f:
        for r in committed_rows:
            f.write(M.json.dumps(r) + "\n")
    (metrics / "null_shuffled_cells.jsonl").touch()
    (metrics / "null_crosstype_cells.jsonl").touch()
    return tables, metrics


def test_parent_f_parity_pass_and_fail(tmp_path: Path):
    tables, metrics = _parity_fixture(tmp_path)
    assert M._assert_parent_f_parity(tables, metrics) == M.SURVIVAL_FLOOR
    tables_bad, metrics_bad = _parity_fixture(tmp_path / "bad", drift=0.01)
    with pytest.raises(AssertionError, match="parity FAIL"):
        M._assert_parent_f_parity(tables_bad, metrics_bad)


def test_parent_f_parity_vacuous_join_raises(tmp_path: Path):
    tables = {"steered": [], "shuffled": [], "crosstype": []}
    metrics = tmp_path / "f_metrics"
    metrics.mkdir()
    for name in M.PARENT_COMMITTED_FILES.values():
        (metrics / name).touch()
    with pytest.raises(AssertionError, match="vacuous"):
        M._assert_parent_f_parity(tables, metrics)


# ── figures-script constant pins + render smoke ───────────────────────


def test_figures_constants_pinned_to_driver():
    """The figures script keeps LOCAL constants (no torch-heavy imports at
    render time — the parent figures-script pattern); pin them to the driver
    + analysis modules so drift is test-breaking."""
    import issue2162_figures as F
    import issue2162_tbmp as TB
    import issue2162_tbmp_figures as FIG

    assert FIG.CONTROL_CELL == TB.CONTROL_CELL
    assert set(FIG.FINAL_K) == set(TB.SWEEP_CELLS)
    for cell, k in FIG.FINAL_K.items():
        assert k == TB.DESIGNED_BOUNDARIES[cell], cell
    assert FIG.BASES == M.BASES
    assert F.SEPARATION_BAR == M.SEPARATION_BAR
    for base in FIG.BASES:
        assert FIG.depth_cell(base, "d1") == base
        assert FIG.depth_cell(base, "d5") == f"recency_{base}_d5"
        assert {FIG.depth_cell(base, d) for d in FIG.DEPTHS} <= set(TB.GRID_CELLS)


def _synth_tables(out_dir: Path, parent_metrics: Path) -> None:
    import issue2162_tbmp as TB
    import issue2162_tbmp_figures as FIG

    rng = __import__("random").Random(2162)
    units = [(c, "tb") for c in TB.GRID_CELLS] + [
        (c, f"tbk{k}") for c in TB.SWEEP_CELLS for k in range(1, FIG.FINAL_K[c])
    ]
    arms = ("steered", "shuffled", "crosstype")

    def _rows(slot_units, slot_override=None):
        rows = {a: [] for a in arms}
        for cell, slot in slot_units:
            for a in arms:
                for i in range(3):
                    f = rng.uniform(-0.2, 0.8)
                    rows[a].append(
                        {
                            "pair_id": f"{cell[:6]}_{i}",
                            "cell": cell,
                            "slot": slot_override or slot,
                            "arm": a,
                            "f_beh": f,
                            "f_netted": f,
                            "f_target_only": f * 0.9,
                            "raw_move_registered": f * 0.5,
                            "separation": 0.8,
                        }
                    )
        return rows

    tb = _rows(units)
    names = {
        "steered": "f_cells_tb.jsonl",
        "shuffled": "null_shuffled_cells_tb.jsonl",
        "crosstype": "null_crosstype_cells_tb.jsonl",
    }
    out_dir.mkdir(parents=True)
    for a, name in names.items():
        with (out_dir / name).open("w") as f:
            for r in tb[a]:
                f.write(M.json.dumps(r) + "\n")
    ref = _rows([(c, "tb") for c in TB.GRID_CELLS], slot_override="ce")
    with (out_dir / "parent_ref_cells_tb.jsonl").open("w") as f:
        for a in arms:
            for r in ref[a]:
                f.write(M.json.dumps(r) + "\n")
    (out_dir / "stats_tb.json").write_text(
        M.json.dumps({"per_cell": {"instr_format|tb": {"coherent_fraction": 0.9}}})
    )
    (out_dir / "identity_gate.json").write_text(
        M.json.dumps(
            {
                "passed": True,
                "n_surviving_pool": 66,
                "per_arm": {
                    "steered": {"mean_delta_f": 0.01},
                    "shuffled": {"mean_delta_f": -0.02},
                },
            }
        )
    )
    parent_metrics.mkdir(parents=True)
    for a, fname in (("steered", "f_cells.jsonl"), ("shuffled", "null_shuffled_cells.jsonl")):
        with (parent_metrics / fname).open("w") as f:
            for r in ref[a]:
                f.write(M.json.dumps(r) + "\n")


def test_figures_render_smoke(tmp_path: Path):
    """Full ``main()`` render on synthetic tables into tmp dirs — exercises
    every figure (errorbar clamp paths included) + captions.json; margin file
    deliberately absent so the deferred-leg skip branch runs too."""
    import issue2162_tbmp_figures as FIG

    out_dir = tmp_path / "turn_boundary"
    parent_metrics = tmp_path / "f_metrics"
    fig_dir = tmp_path / "figs"
    _synth_tables(out_dir, parent_metrics)
    rc = FIG.main(
        [
            "--out-dir",
            str(out_dir),
            "--parent-metrics-dir",
            str(parent_metrics),
            "--fig-dir",
            str(fig_dir),
        ]
    )
    assert rc == 0
    for name in FIG.CAPTIONS:
        if name == "tb_margin_scatter":
            assert not (fig_dir / f"{name}.png").exists()
            continue
        assert (fig_dir / f"{name}.png").exists(), name
    payload = M.json.loads((out_dir / "captions.json").read_text())
    assert "NOT RENDERED" in payload["captions"]["tb_margin_scatter"]
    assert payload["tables"]["coherent_fraction"]["instr_format|tb"] == 0.9
