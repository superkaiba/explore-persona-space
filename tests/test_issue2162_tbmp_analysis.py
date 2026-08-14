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
    import issue2162_recency_rawscale as RS
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
                            "pair_id": f"{cell}_{i}",
                            "cell": cell,
                            "slot": slot_override or slot,
                            "arm": a,
                            "f_beh": f,
                            "f_netted": f,
                            "f_target_only": f * 0.9,
                            "raw_move_registered": f * 0.5,
                            "delta_patched_mean": f * 0.4,
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
    parent_names = {
        "steered": "f_cells.jsonl",
        "shuffled": "null_shuffled_cells.jsonl",
        "crosstype": "null_crosstype_cells.jsonl",
    }
    parent_metrics.mkdir(parents=True, exist_ok=True)
    for a, fname in parent_names.items():
        with (parent_metrics / fname).open("w") as f:
            for r in ref[a]:
                f.write(M.json.dumps(r) + "\n")
    with (parent_metrics / "anchors.jsonl").open("w") as f:
        for cell in RS.DEFAULT_CELLS:
            for i in range(3):
                f.write(
                    M.json.dumps(
                        {
                            "pair_id": f"{cell}_{i}",
                            "cell": cell,
                            "delta_floor_mean": -0.2,
                            "delta_ceiling_mean": 0.6,
                            "separation": 0.8,
                        }
                    )
                    + "\n"
                )
    # Parent raw-scale fixture through the PRODUCTION defaults path (no null
    # CIs — the committed recency_rawscale.json schema), then this round's
    # rawscale_tb.json through the production step (with --null-cis).
    RS.main(
        [
            "--metrics-dir",
            str(parent_metrics),
            "--out-json",
            str(parent_metrics / "recency_rawscale.json"),
        ]
    )
    M.step_rawscale(SimpleNamespace(out_dir=out_dir, parent_metrics_dir=parent_metrics))
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


# ── S1: declared raw-scale artifact (rawscale_tb.json) ─────────────────


def test_step_rawscale_declared_artifact(tmp_path: Path):
    """The plan-§9 P5 declared file exists, carries the registered bootstrap
    identity (B=10000, seed 21620), per-arm 95% CIs on EVERY row (the manifest's
    tb_rawscale transform), and the parent DEFAULTS path stays schema-identical
    to the committed recency_rawscale.json (no null CIs — rng isolation)."""
    out_dir = tmp_path / "turn_boundary"
    parent_metrics = tmp_path / "f_metrics"
    _synth_tables(out_dir, parent_metrics)
    payload = M.json.loads((out_dir / "rawscale_tb.json").read_text())
    assert payload["slot"] == "tb"
    assert payload["boot"] == {"B": 10000, "seed": 21620}
    rows = payload["rows"]
    assert {r["subset"] for r in rows} == {"all", "surviving"}
    for r in rows:
        for arm in ("steered", "shuffled", "crosstype"):
            assert f"{arm}_ci95" in r, (r["cell"], arm)
    parent = M.json.loads((parent_metrics / "recency_rawscale.json").read_text())
    assert parent["rows"] and all("shuffled_ci95" not in r for r in parent["rows"])
    assert all("crosstype_ci95" not in r for r in parent["rows"])


# ── S2: rebuilt-vs-recorded parity (driver-side) ───────────────────────


def _ctx(cid: str, cell: str = "instr_format", user: str = "hello") -> dict:
    return {
        "id": cid,
        "cell": cell,
        "value_id": "v1",
        "carrier": "d1",
        "system": "sys",
        "history": [],
        "user": user,
    }


def test_context_parity_pass_drift_missing_and_bad_sha():
    import issue2162_tbmp as TB

    sha = TB.frozen_gen_sha256_producer_domain()
    ctxs = {"c1": _ctx("c1"), "c2": _ctx("c2", user="other")}

    def _payload(recorded: dict) -> dict:
        return {"contexts": recorded, "frozen_gen_sha256": sha}

    rep = TB.parent_bank_context_parity(_payload({k: dict(v) for k, v in ctxs.items()}), ctxs)
    assert rep["passed"] and rep["n_violations"] == 0 and rep["n_contexts_checked"] == 2

    drifted = {k: dict(v) for k, v in ctxs.items()}
    drifted["c1"]["user"] = "TAMPERED"
    rep2 = TB.parent_bank_context_parity(_payload(drifted), ctxs)
    assert not rep2["passed"]
    assert any(
        v["check"] == "context_payload_drift" and v["context_id"] == "c1"
        for v in rep2["violations"]
    )

    rep3 = TB.parent_bank_context_parity(_payload({"c1": dict(ctxs["c1"])}), ctxs)
    assert not rep3["passed"]
    assert any(v["check"] == "context_missing_from_bank" for v in rep3["violations"])

    bad = {"contexts": {k: dict(v) for k, v in ctxs.items()}, "frozen_gen_sha256": "0" * 64}
    rep4 = TB.parent_bank_context_parity(bad, ctxs)
    assert not rep4["passed"]
    assert any(v["check"] == "frozen_gen_sha256" for v in rep4["violations"])

    with pytest.raises(RuntimeError, match="no 'contexts' map"):
        TB.parent_bank_context_parity({}, ctxs)


def test_len_delta_parity_pass_drift_missing_pair_and_missing_file(tmp_path: Path):
    import issue2162_tbmp as TB

    cell = sorted(TB.CAPTURE_CELLS)[0]
    pairs = [
        SimpleNamespace(pair_id=f"p{i}", cell=cell, a=f"p{i}__a", b=f"p{i}__b") for i in range(3)
    ]
    resolved = {}
    for i, p in enumerate(pairs):
        resolved[p.a] = {"ctx_len": 100}
        resolved[p.b] = {"ctx_len": 100 + i}

    good = tmp_path / "f_cells.jsonl"
    with good.open("w") as fh:
        for i, p in enumerate(pairs):
            fh.write(M.json.dumps({"pair_id": p.pair_id, "len_delta": i}) + "\n")
    rep = TB.g1b_len_delta_parity(resolved, pairs, good)
    assert rep["passed"] and rep["n_checked"] == 3 and rep["n_capture_pairs"] == 3

    drift = tmp_path / "f_cells_drift.jsonl"
    with drift.open("w") as fh:
        fh.write(M.json.dumps({"pair_id": "p0", "len_delta": 99}) + "\n")
        for i, p in enumerate(pairs[1:], start=1):
            fh.write(M.json.dumps({"pair_id": p.pair_id, "len_delta": i}) + "\n")
    rep2 = TB.g1b_len_delta_parity(resolved, pairs, drift)
    assert not rep2["passed"]
    assert any(v["check"] == "len_delta_drift" and v["pair_id"] == "p0" for v in rep2["violations"])

    partial = tmp_path / "f_cells_partial.jsonl"
    with partial.open("w") as fh:
        for i, p in enumerate(pairs[:2]):
            fh.write(M.json.dumps({"pair_id": p.pair_id, "len_delta": i}) + "\n")
    rep3 = TB.g1b_len_delta_parity(resolved, pairs, partial)
    assert not rep3["passed"]  # anti-vacuous: a missing pair can never pass silently
    assert any(v["check"] == "pair_missing_from_parent_f_cells" for v in rep3["violations"])

    with pytest.raises(AssertionError, match="missing"):
        TB.g1b_len_delta_parity(resolved, pairs, tmp_path / "nope.jsonl")


def test_shuffled_assignment_parity_refuses_verifies_and_drifts():
    import issue2162_tbmp as TB

    from explore_persona_space.experiments.issue2162 import bank2162 as BANK

    pairs = BANK.build_pairs()
    with pytest.raises(RuntimeError, match="no shuffled donor map"):
        TB.shuffled_assignment_with_parity(pairs, {}, "bank.json")

    frozen = BANK.donor_assignment_2162(pairs)["shuffled"]
    ours, note = TB.shuffled_assignment_with_parity(
        pairs, {"donor_assignment": {"shuffled": frozen}}, "bank.json"
    )
    assert ours and "parity-verified" in note

    tampered = dict(frozen)
    tampered[next(iter(ours))] = "WRONG_DONOR"
    with pytest.raises(AssertionError, match="DRIFTED"):
        TB.shuffled_assignment_with_parity(
            pairs, {"donor_assignment": {"shuffled": tampered}}, "bank.json"
        )


# ── S2d: anchor-channel vacuous-join floor ─────────────────────────────


def _scores_fixture(tmp_path: Path) -> Path:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "x.anchors.scores.jsonl").write_text(
        M.json.dumps({"item_id": "i0", "score": 90.0}) + "\n"
    )
    (scores_dir / "coherence.anchors.scores.jsonl").write_text(
        M.json.dumps({"item_id": "c|c0|0", "context_id": "c0", "draw": 0, "score": 90.0}) + "\n"
    )
    return scores_dir


def test_build_channels_vacuous_join_raises(tmp_path: Path):
    scores_dir = _scores_fixture(tmp_path)
    args = SimpleNamespace(parent_scores_dir=scores_dir)
    with pytest.raises(AssertionError, match="vacuous join"):
        M._build_channels(args, {}, {}, set())


def test_build_channels_floor_counts_parity_checked_pairs(tmp_path: Path, monkeypatch):
    scores_dir = _scores_fixture(tmp_path)
    args = SimpleNamespace(parent_scores_dir=scores_dir)
    chan = {
        "delta_floor": -0.2,
        "delta_ceiling": 0.6,
        "b_floor": 0.3,
        "b_ceiling": 0.9,
        "n_floor": 2,
        "n_ceiling": 2,
    }

    def fake_channels(pair, coh_draws, anchor_scores):
        return dict(chan)

    monkeypatch.setattr(M, "pair_anchor_channels", fake_channels)
    n = M.SURVIVAL_FLOOR
    pair_ids = {f"pp{i}" for i in range(n)}
    pairs_by_id = {pid: SimpleNamespace(pair_id=pid) for pid in pair_ids}
    committed = {pid: {"delta_floor_mean": -0.2, "delta_ceiling_mean": 0.6} for pid in pair_ids}
    out = M._build_channels(args, pairs_by_id, committed, pair_ids)
    assert len(out) == n

    committed_small = {pid: committed[pid] for pid in sorted(pair_ids)[: n - 1]}
    with pytest.raises(AssertionError, match="vacuous join"):
        M._build_channels(args, pairs_by_id, committed_small, pair_ids)


# ── pre-launch minors (m1/m2/m3) ───────────────────────────────────────


def test_fig_empty_panels_raise(tmp_path: Path):
    """m1 fix-engaged: an intentionally-empty panel RAISES instead of shipping
    a blank render (#1112 empty-figure class) — the flagged ``fig_tb_rawscale``
    plus one sibling of the same silent-skip shape."""
    import issue2162_tbmp_figures as FIG

    empty_rs = {"rows": []}
    with pytest.raises(RuntimeError, match="rendered EMPTY"):
        FIG.fig_tb_rawscale(empty_rs, empty_rs, tmp_path, [])
    empty_tb = {"steered": [], "shuffled": [], "crosstype": []}
    with pytest.raises(RuntimeError, match="rendered EMPTY"):
        FIG.fig_tb_hero(empty_tb, empty_tb, tmp_path, [])
    assert not list(tmp_path.iterdir())  # nothing shipped on the raise path


def test_len_delta_conflicting_duplicate_flags_violation(tmp_path: Path):
    """m2 fix-engaged: a CONFLICTING duplicate (same pair_id, different
    len_delta) in the parent table is a violation — never a silent
    first-wins half-check; an identical-value duplicate stays benign."""
    import issue2162_tbmp as TB

    cell = sorted(TB.CAPTURE_CELLS)[0]
    pairs = [
        SimpleNamespace(pair_id=f"p{i}", cell=cell, a=f"p{i}__a", b=f"p{i}__b") for i in range(2)
    ]
    resolved = {}
    for i, p in enumerate(pairs):
        resolved[p.a] = {"ctx_len": 100}
        resolved[p.b] = {"ctx_len": 100 + i}

    conflicted = tmp_path / "f_cells_conflict.jsonl"
    with conflicted.open("w") as fh:
        fh.write(M.json.dumps({"pair_id": "p0", "len_delta": 0}) + "\n")
        fh.write(M.json.dumps({"pair_id": "p0", "len_delta": 5}) + "\n")  # CONFLICT
        fh.write(M.json.dumps({"pair_id": "p1", "len_delta": 1}) + "\n")
    rep = TB.g1b_len_delta_parity(resolved, pairs, conflicted)
    assert not rep["passed"]
    assert any(
        v["check"] == "conflicting_duplicate_len_delta"
        and v["pair_id"] == "p0"
        and v["values"] == [0, 5]
        for v in rep["violations"]
    )

    benign = tmp_path / "f_cells_benign.jsonl"
    with benign.open("w") as fh:
        for i, p in enumerate(pairs):
            fh.write(M.json.dumps({"pair_id": p.pair_id, "len_delta": i}) + "\n")
            fh.write(M.json.dumps({"pair_id": p.pair_id, "len_delta": i}) + "\n")  # identical dup
    rep2 = TB.g1b_len_delta_parity(resolved, pairs, benign)
    assert rep2["passed"] and rep2["n_checked"] == 2


def test_defaults_path_byte_identical_to_committed_rawscale(tmp_path: Path):
    """m3 pin: a defaults run of ``issue2162_recency_rawscale.py`` over the
    COMMITTED parent tables reproduces the committed ``recency_rawscale.json``
    byte-for-byte — protecting the panel-verified reproduction (and the
    ``--null-cis`` rng isolation) from a future RNG-touching edit.
    Sparse-worktree cone: ``eval_results/issue_2162/f_metrics``
    (tests/sparse_cones.txt)."""
    import issue2162_recency_rawscale as RS

    committed = REPO_ROOT / "eval_results" / "issue_2162" / "f_metrics" / "recency_rawscale.json"
    out = tmp_path / "recency_rawscale.json"
    RS.main(["--metrics-dir", str(committed.parent), "--out-json", str(out)])
    assert out.read_bytes() == committed.read_bytes()
