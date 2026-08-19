"""Tests for scripts/issue2162_ladder_analysis.py (persona-specificity ladder P6).

Synthetic-fixture end-to-end: builds a tiny 3-surviving-rung x 2-carrier world
(anchor rows + scores, grid rows + scores, V_a stores, margin shards, gate +
donor-screen JSONs) and drives the PRODUCTION entrypoint ``main()`` through
every step, asserting the registered F arithmetic, trend/bootstrap/lattice
stats, margin shifts, the §6 set-check fail-loud path, and the inverted-CI
errorbar clamp (gotchas.md xerr/yerr contract).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2162_ladder_analysis as AN  # noqa: E402
import issue2162_ladder_judge as LJ  # noqa: E402

from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402

CARRIERS = ("d1", "d2")
SURVIVORS = ("r1_pirate", "r3_warm", "r4_trait")
DIRECTIONS = tuple(f"{k}_{v}" for v in SURVIVORS for k in ("install", "erase"))
DROPPED_XTYPE_PAIR = "install_r4_trait::d2"
K_ANCHOR = 4
K_GRID = 3

# Judge-score design values (0-100). floor/ceil under the TARGET descriptor,
# and the steered / null grid scores per (kind, rung).
ANCHOR_TARGET = {  # value_id -> (score on plain ctx = floor, score on own ctx = ceil)
    "r1_pirate": (5.0, 65.0),
    "r3_warm": (5.0, 55.0),
    "r4_trait": (5.0, 45.0),
}
ANCHOR_PLAIN = {"plain_ctx": 80.0, "persona_ctx": 20.0}
GRID_TARGET = {  # (kind, rung) -> {arm: target-descriptor score}
    ("install", "r1_pirate"): {"steered": 53.0, "null_sameval": 8.0, "null_xtype": 8.0},
    ("install", "r3_warm"): {"steered": 35.0, "null_sameval": 15.0, "null_xtype": 8.0},
    ("install", "r4_trait"): {"steered": 25.0, "null_sameval": 8.0, "null_xtype": 8.0},
    ("erase", "r1_pirate"): {"steered": 17.0, "null_sameval": 60.0, "null_xtype": 60.0},
    ("erase", "r3_warm"): {"steered": 25.0, "null_sameval": 50.0, "null_xtype": 50.0},
    ("erase", "r4_trait"): {"steered": 29.0, "null_sameval": 41.0, "null_xtype": 41.0},
}
GRID_PLAIN_SCORE = 30.0
INCOHERENT_KEY = ("install_r1_pirate::d1", "ce", "steered", 2)  # one filtered grid draw


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _grid_cells() -> list[tuple[str, str, str, str]]:
    """(direction, carrier, slot, arm) for every realized grid cell."""
    out = []
    for direction in DIRECTIONS:
        for carrier in CARRIERS:
            pair_id = f"{direction}::{carrier}"
            for slot in ("ce", "pe"):
                for arm in AN.ARMS:
                    if arm == "null_xtype" and pair_id == DROPPED_XTYPE_PAIR:
                        continue
                    out.append((direction, carrier, slot, arm))
    return out


def build_fixture(root: Path) -> dict[str, Path]:
    """Write the full synthetic input world; returns the CLI dir map."""
    in_root = root / "in"
    work = root / "work"
    out_dir = root / "out"
    figs = root / "figs"
    raw = in_root / LJ.HF_PREFIX / "raw_completions" / "ladder"
    tensors = in_root / LJ.HF_PREFIX / "analysis_tensors" / "ladder"
    scores = work / "scores"
    gates = work / "gates"
    for d in (
        raw / "grid",
        raw / "anchors",
        tensors / "anchors",
        tensors / "va_store",
        tensors / "margin",
        scores,
        gates,
    ):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    e1 = torch.zeros(4)
    e1[0] = 1.0
    _write_anchor_world(raw, tensors, scores, rng, e1)
    _write_gates(gates)
    _write_grid_world(raw, tensors, scores, rng, e1)
    return {"in_root": in_root, "work": work, "out": out_dir, "figs": figs}


def _write_anchor_world(raw: Path, tensors: Path, scores: Path, rng, e1: torch.Tensor) -> None:
    # ── anchors (rows + V_a + scores) — STAGED SPLIT layout: the HF upload
    # splits text rows (raw_completions) from V_a tensors (analysis_tensors),
    # so the fixture pins the anchor .pt on the TENSORS side (r2 fix). ──
    anchor_rows: list[dict] = []
    coh_anchor: list[dict] = []
    plain_anchor: list[dict] = []
    target_anchor: dict[str, list[dict]] = {v: [] for v in SURVIVORS}
    va_index: list[dict] = []
    va_rows: list[torch.Tensor] = []
    for value_id in ("plain", *SURVIVORS):
        rung = LB.VALUES_BY_ID[value_id].rung
        for carrier in CARRIERS:
            ctx = LB.context_id(value_id, carrier)
            for draw in range(K_ANCHOR):
                anchor_rows.append(
                    {
                        "context_id": ctx,
                        "cell": "anchor",
                        "value_id": value_id,
                        "rung": rung,
                        "carrier": carrier,
                        "draw": draw,
                        "text": "t",
                    }
                )
                coh_score = (
                    30.0 if (value_id == "plain" and carrier == "d1" and draw == 3) else 90.0
                )
                coh_anchor.append({"context_id": ctx, "draw": draw, "score": coh_score})
                is_plain = value_id == "plain"
                plain_anchor.append(
                    {
                        "context_id": ctx,
                        "draw": draw,
                        "score": ANCHOR_PLAIN["plain_ctx" if is_plain else "persona_ctx"],
                    }
                )
                for v in SURVIVORS:
                    if value_id in (v, "plain"):
                        target_anchor[v].append(
                            {
                                "context_id": ctx,
                                "draw": draw,
                                "score": ANCHOR_TARGET[v][0 if is_plain else 1],
                            }
                        )
                # anchor V_a: plain ctx near 0, persona ctx near e1
                base = torch.zeros(4) if is_plain else e1.clone()
                va_index.append({"context_id": ctx, "draw": draw})
                va_rows.append(
                    base + 0.01 * torch.tensor(rng.standard_normal(4), dtype=torch.float32)
                )
    _jsonl(raw / "anchors" / "anchors_gate_w0.jsonl", anchor_rows)
    torch.save(
        {
            "layers": [26],
            "index": va_index,
            "va_span": torch.stack(va_rows).unsqueeze(1),  # (N, 1, 4)
            "empty_rows": [],
        },
        tensors / "anchors" / "va_anchors_gate_w0.pt",
    )

    def _score_rows(rows: list[dict], extra: dict) -> list[dict]:
        return [{**extra, **r} for r in rows]

    _jsonl(scores / "coherence.anchors.scores.jsonl", _score_rows(coh_anchor, {"kind": "anchor"}))
    _jsonl(scores / "hol-plain.anchors.scores.jsonl", _score_rows(plain_anchor, {"kind": "anchor"}))
    for v in LB.PERSONA_VALUE_IDS:
        rows = target_anchor.get(v, [])
        _jsonl(scores / f"hol-{v}.anchors.scores.jsonl", _score_rows(rows, {"kind": "anchor"}))


def _write_gates(gates: Path) -> None:
    # ── gate + donor-screen JSONs ──
    rungs: dict[str, dict] = {}
    for v in LB.PERSONA_VALUE_IDS:
        survived = v in SURVIVORS
        per_carrier = {}
        for carrier in CARRIERS:
            if survived:
                floor, ceil = (x / 100.0 for x in ANCHOR_TARGET[v])
                per_carrier[carrier] = {
                    "target_sep": ceil - floor,
                    "netted_sep": 0.9,
                    "unscored": False,
                    "passed": True,
                }
            else:
                per_carrier[carrier] = {
                    "target_sep": None,
                    "netted_sep": None,
                    "unscored": True,
                    "passed": False,
                }
        rungs[v] = {
            "survived": survived,
            "surviving_carriers": list(CARRIERS) if survived else [],
            "n_carriers_pass": len(CARRIERS) if survived else 0,
            "per_carrier": per_carrier,
        }
    (gates / "ladder_separation_gate.json").write_text(
        json.dumps(
            {
                "rungs": rungs,
                "all_rungs_failed": False,
                "passed": True,
                "bars": {"target_sep_bar": 0.25, "netted_sep_bar": 0.5, "rung_min_carriers": 4},
            }
        ),
        encoding="utf-8",
    )
    (gates / "ladder_donor_screen.json").write_text(
        json.dumps({"assignments": {DROPPED_XTYPE_PAIR: {"status": "dropped"}}}),
        encoding="utf-8",
    )


def _write_grid_world(raw: Path, tensors: Path, scores: Path, rng, e1: torch.Tensor) -> None:
    # ── grid rows + scores + V_a + margins ──
    coh_grid: list[dict] = []
    plain_grid: list[dict] = []
    target_grid: dict[str, list[dict]] = {v: [] for v in SURVIVORS}
    conj_grid: dict[str, list[dict]] = {}
    grid_by_block: dict[tuple[str, str, str], list[dict]] = {}
    margin_by_block: dict[tuple[str, str, str], list[dict]] = {}
    va_by_block: dict[tuple[str, str, str], tuple[list[dict], list[torch.Tensor]]] = {}
    for direction, carrier, slot, arm in _grid_cells():
        kind, persona = direction.split("_", 1)
        pair_id = f"{direction}::{carrier}"
        block_key = f"{direction}|{slot}|{arm}"
        ctx_a = LB.context_id("plain" if kind == "install" else persona, carrier)
        ctx_b = LB.context_id(persona if kind == "install" else "plain", carrier)
        bk = (direction, slot, arm)
        grid_rows = grid_by_block.setdefault(bk, [])
        m_rows = margin_by_block.setdefault(bk, [])
        vi, vv = va_by_block.setdefault(bk, ([], []))
        for draw in range(K_GRID):
            grid_rows.append(
                {
                    "block_key": block_key,
                    "cell": direction,
                    "direction": direction,
                    "kind": kind,
                    "persona": persona,
                    "slot": slot,
                    "arm": arm,
                    "pair_id": pair_id,
                    "carrier": carrier,
                    "value_a": "plain" if kind == "install" else persona,
                    "value_b": persona if kind == "install" else "plain",
                    "context_a": ctx_a,
                    "context_id": ctx_a,
                    "context_b": ctx_b,
                    "donor_context_id": f"donor::{pair_id}",
                    "len_delta": 5,
                    "draw": draw,
                    "seed": 0,
                    "temperature": 1.0,
                    "n_completion_tokens": 64,
                    "cap_hit": bool(draw == 0 and arm == "steered" and slot == "ce"),
                    "cap_hit_basis": "finish_reason",
                    "text": "t",
                }
            )
            src = {"pair_id": pair_id, "slot": slot, "arm": arm, "draw": draw}
            coh = 30.0 if (pair_id, slot, arm, draw) == INCOHERENT_KEY else 90.0
            coh_grid.append({**src, "score": coh})
            target_grid[persona].append({**src, "score": GRID_TARGET[(kind, persona)][arm]})
            plain_grid.append({**src, "score": GRID_PLAIN_SCORE})
            if persona == "r1_pirate" and arm == "steered":
                for key in LJ.LADDER_CONJUNCTS["r1_pirate"]:
                    conj_grid.setdefault(f"conj-r1_pirate-{key}", []).append({**src, "score": 40.0})
            # patched V_a: steered installs land at 0.8*t-axis progress.
            frac = {"steered": 0.8, "null_sameval": 0.05, "null_xtype": 0.05}[arm]
            start = torch.zeros(4) if kind == "install" else e1.clone()
            target = e1.clone() if kind == "install" else torch.zeros(4)
            vp = start + frac * (target - start)
            vi.append({"pair_id": pair_id, "context_a": ctx_a, "draw": draw})
            vv.append(vp + 0.01 * torch.tensor(rng.standard_normal(4), dtype=torch.float32))
        # TF margin rows: 2 pool items per side; steered margin 1.0, anchor 0.0.
        for idx in range(2):
            for side, lnp in (("A", -2.0), ("B", -1.0)):
                m_rows.append(
                    {
                        "block_key": block_key,
                        "cell": direction,
                        "slot": slot,
                        "arm": arm,
                        "pair_id": pair_id,
                        "donor_context_id": f"donor::{pair_id}",
                        "pool_key": direction,
                        "pool_idx": idx,
                        "pool_side": side,
                        "n_pool_tokens": 10,
                        "lnp_mean": lnp if arm == "steered" else -1.5,
                        "skipped": False,
                    }
                )
    for j, (bk, rows) in enumerate(sorted(grid_by_block.items())):
        _jsonl(raw / "grid" / f"shard_b{j:03d}.jsonl", rows)
        vi, vv = va_by_block[bk]
        torch.save(
            {
                "block_key": rows[0]["block_key"],
                "layers": [26],
                "index": vi,
                "va_span": torch.stack(vv).unsqueeze(1),
                "empty_rows": [],
            },
            tensors / "va_store" / f"shard_b{j:03d}.pt",
        )
        _jsonl(tensors / "margin" / f"shard_b{j:03d}.jsonl", margin_by_block[bk])
    # anchor margins: every pair.a context x its direction pool, both sides -1.5.
    anchor_margin: list[dict] = []
    for direction in DIRECTIONS:
        kind, persona = direction.split("_", 1)
        for carrier in CARRIERS:
            ctx_a = LB.context_id("plain" if kind == "install" else persona, carrier)
            for idx in range(2):
                for side in ("A", "B"):
                    anchor_margin.append(
                        {
                            "context_id": ctx_a,
                            "value_id": "plain" if kind == "install" else persona,
                            "rung": "anchor",
                            "carrier": carrier,
                            "pool_key": direction,
                            "pool_idx": idx,
                            "pool_side": side,
                            "n_pool_tokens": 10,
                            "lnp_mean": -1.5,
                            "skipped": False,
                        }
                    )
    _jsonl(tensors / "margin" / "anchor_margin_w0.jsonl", anchor_margin)

    _jsonl(scores / "coherence.grid.scores.jsonl", coh_grid)
    _jsonl(scores / "hol-plain.grid.scores.jsonl", plain_grid)
    # Production shape: gate-dropped rungs generate NO grid rollouts and NO
    # judge waves — only surviving rungs get a hol-<v>.grid wave file on disk.
    for v in SURVIVORS:
        _jsonl(scores / f"hol-{v}.grid.scores.jsonl", target_grid[v])
    for rid, rows in conj_grid.items():
        _jsonl(scores / f"{rid}.grid.scores.jsonl", rows)


def _argv(dirs: dict[str, Path], step: str = "all") -> list[str]:
    return [
        "--step",
        step,
        "--in-root",
        str(dirs["in_root"]),
        "--work-root",
        str(dirs["work"]),
        "--out-dir",
        str(dirs["out"]),
        "--figures-dir",
        str(dirs["figs"]),
        "--n-boot",
        "200",
        "--n-perm",
        "300",
        "--skip-token-counts",
    ]


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("ladder_an")
    dirs = build_fixture(root)
    rc = AN.main(_argv(dirs))
    assert rc == 0
    return {
        "dirs": dirs,
        "f_cells": [
            json.loads(x) for x in (dirs["out"] / "f_cells.jsonl").read_text().splitlines() if x
        ],
        "stats": json.loads((dirs["out"] / "stats.json").read_text()),
    }


def test_end_to_end_outputs_exist(fixture_run):
    out = fixture_run["dirs"]["out"]
    for name in (
        "anchors.jsonl",
        "f_cells.jsonl",
        "null_samevalue_cells.jsonl",
        "null_crosstype_cells.jsonl",
        "conjuncts.jsonl",
        "margin.jsonl",
        "stats.json",
    ):
        assert (out / name).exists(), name
    figs = fixture_run["dirs"]["figs"]
    for stem in (
        "ladder_hero",
        "ladder_percarrier",
        "asymmetry",
        "anchor_separation",
        "rubric_bridge",
        "dv_agreement",
        "conjunct_diag",
    ):
        assert (figs / f"{stem}.png").exists(), stem


def test_cell_counts_and_dropped_xtype(fixture_run):
    f_cells = fixture_run["f_cells"]
    assert len(f_cells) == 24  # 6 directions x 2 carriers x 2 slots
    out = fixture_run["dirs"]["out"]
    xtype = [
        json.loads(x) for x in (out / "null_crosstype_cells.jsonl").read_text().splitlines() if x
    ]
    assert len(xtype) == 22  # dropped pair excluded at both slots
    assert not [r for r in xtype if r["pair_id"] == DROPPED_XTYPE_PAIR]


def test_f_target_arithmetic(fixture_run):
    by_key = {(r["pair_id"], r["slot"]): r for r in fixture_run["f_cells"]}
    r = by_key[("install_r1_pirate::d1", "ce")]
    # floor 0.05, ceil 0.65, steered 0.53 -> F = (0.53-0.05)/0.60 = 0.8
    assert r["f_target"] == pytest.approx(0.8, abs=1e-9)
    assert r["n_coherent"] == K_GRID - 1  # one incoherent draw filtered
    e = by_key[("erase_r1_pirate::d1", "ce")]
    # erase: (0.65-0.17)/0.60 = 0.8
    assert e["f_target"] == pytest.approx(0.8, abs=1e-9)
    # F_act: patched at 0.8 of the floor->ceiling axis (small noise tolerance)
    assert r["f_act"] == pytest.approx(0.8, abs=0.05)
    # netted bridge present (both rubrics judged)
    assert r["f_netted"] is not None


def test_stats_trend_and_lattice(fixture_run):
    stats = fixture_run["stats"]
    t = stats["trend_tests"]["install-ce"]
    assert t["n_surviving_rungs"] == 3
    assert not t["descriptive_only"]
    assert t["rho_observed"] == pytest.approx(1.0)  # monotone by design
    assert t["p_one_sided"] is not None and t["p_holm"] is not None
    # lattice: r1 install-ce cleanly transfers (nulls at F=0.05, CI degenerate)
    lat = stats["lattice"]
    assert lat["install_r1_pirate|ce"]["verdict"] == "transfers"
    # r3 install: null_sameval mean F = (0.15-0.05)/0.5 = 0.2 > 0.10 -> withheld
    r3 = lat["install_r3_warm|ce"]
    assert r3["null_sanity_flag"] is True
    assert r3["verdict"] == "no-clean-transfer"
    assert r3["transfers_withheld_by_null_sanity"] is True
    # gate-failed rung -> untestable
    assert lat["install_r2_butler|ce"]["verdict"] == "untestable"
    # H4: erase - install for r1 at ce: 0.8 - 0.8 = 0
    h4 = stats["h4_asymmetry"]["r1_pirate|ce"]
    assert h4["mean_erase_minus_install"] == pytest.approx(0.0, abs=1e-9)
    assert h4["n_carriers_paired"] == 2


def test_margin_shift(fixture_run):
    out = fixture_run["dirs"]["out"]
    rows = [json.loads(x) for x in (out / "margin.jsonl").read_text().splitlines() if x]
    steered = [r for r in rows if r["arm"] == "steered"]
    assert steered and all(r["margin_shift"] == pytest.approx(1.0) for r in steered)
    nulls = [r for r in rows if r["arm"] != "steered"]
    assert nulls and all(r["margin_shift"] == pytest.approx(0.0) for r in nulls)


def test_anchors_table(fixture_run):
    out = fixture_run["dirs"]["out"]
    rows = [json.loads(x) for x in (out / "anchors.jsonl").read_text().splitlines() if x]
    assert len(rows) == len(LB.PERSONA_VALUE_IDS) * len(CARRIERS)
    r1_d1 = next(r for r in rows if r["rung"] == "r1_pirate" and r["carrier"] == "d1")
    assert r1_d1["floor_target_coherent"] == pytest.approx(0.05)
    assert r1_d1["ceil_target_coherent"] == pytest.approx(0.65)
    # coherence filter dropped one plain::d1 draw
    assert r1_d1["n_floor_coherent"] == K_ANCHOR - 1
    dropped = next(r for r in rows if r["rung"] == "r2_butler" and r["carrier"] == "d1")
    assert dropped["rung_survived"] is False and dropped["gate_unscored"] is True


def test_spearman_rows_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(7)
    x = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0])  # ties like R5a=R5b
    ys = rng.standard_normal((20, 6))
    ours = AN._spearman_rows(x, ys)
    for i in range(20):
        ref = spearmanr(x, ys[i]).statistic
        assert ours[i] == pytest.approx(ref, abs=1e-12)


def test_f_from_stats_directions():
    assert AN._f_from_stats("install", 0.53, 0.05, 0.65) == pytest.approx(0.8)
    assert AN._f_from_stats("erase", 0.17, 0.05, 0.65) == pytest.approx(0.8)
    assert AN._f_from_stats("install", None, 0.05, 0.65) is None
    assert AN._f_from_stats("install", 0.5, 0.3, 0.3) is None  # degenerate denom


def test_gate_dropped_rung_grid_waves_absent(fixture_run):
    """Gate-dropped rungs have NO grid judge wave on disk (production shape:
    r4_trait / r5a in the real run) — f-tables still completed end-to-end
    (fixture_run drove main() through every step) and only surviving-rung
    rows appear in the steered cell table."""
    scores = fixture_run["dirs"]["work"] / "scores"
    dropped = [v for v in LB.PERSONA_VALUE_IDS if v not in SURVIVORS]
    assert dropped, "fixture must exercise the gate-dropped path"
    for v in dropped:
        assert not (scores / f"hol-{v}.grid.scores.jsonl").exists(), v
    assert {r["rung"] for r in fixture_run["f_cells"]} == set(SURVIVORS)


def test_dropped_rungs_named_in_log(tmp_path, caplog):
    """f-tables logs one INFO line naming every gate-dropped rung it skipped."""
    dirs = build_fixture(tmp_path)
    with caplog.at_level(logging.INFO, logger="issue2162.ladder_analysis"):
        assert AN.main(_argv(dirs, step="f-tables")) == 0
    lines = [r.getMessage() for r in caplog.records if "gate-dropped rungs" in r.getMessage()]
    assert lines, "no gate-dropped INFO line emitted"
    for v in ("r2_butler", "r5a_lu_therapy", "r5b_lu_philosophy"):
        assert v in lines[0], v


def test_missing_wave_for_surviving_rung_still_raises(tmp_path):
    """A SURVIVING rung whose grid wave file is absent is REAL missing data —
    the gate-dropped skip must not swallow it."""
    dirs = build_fixture(tmp_path)
    (dirs["work"] / "scores" / "hol-r3_warm.grid.scores.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="hol-r3_warm"):
        AN.main(_argv(dirs, step="f-tables"))


def test_set_check_raises_on_missing_registered_row(tmp_path):
    dirs = build_fixture(tmp_path)
    # Remove one steered grid shard -> a registered row goes missing.
    grid_dir = dirs["in_root"] / LJ.HF_PREFIX / "raw_completions" / "ladder" / "grid"
    victim = None
    for shard in sorted(grid_dir.glob("shard_*.jsonl")):
        rows = [json.loads(x) for x in shard.read_text().splitlines() if x]
        if rows and rows[0]["arm"] == "steered":
            victim = shard
            break
    assert victim is not None
    victim.unlink()
    with pytest.raises(RuntimeError, match="set-check"):
        AN.main(_argv(dirs, step="f-tables"))


def test_inverted_ci_errorbar_clamps(fixture_run, tmp_path):
    """gotchas.md xerr/yerr: an INVERTED bootstrap CI must render, not raise."""
    import matplotlib

    matplotlib.use("Agg")
    stats = json.loads(json.dumps(fixture_run["stats"]))  # deep copy
    key = "install_r1_pirate|ce|steered"
    rec = stats["estimation"][key]
    rec["ci_lo"] = rec["mean_f_target"] + 0.05  # inverted around the point
    rec["ci_hi"] = rec["mean_f_target"] - 0.05
    gate = AN._read_gate(fixture_run["dirs"]["work"] / "gates")
    args = AN.parse_args(_argv({**fixture_run["dirs"], "figs": tmp_path}))
    args.figures_dir = tmp_path
    AN.fig_ladder_hero(stats, gate, args)
    assert (tmp_path / "ladder_hero.png").exists()


def test_err_offsets_nonnegative():
    lo, hi = AN._err(0.5, 0.6, 0.4)  # fully inverted
    assert lo == 0.0 and hi == 0.0
    lo, hi = AN._err(0.5, 0.4, 0.7)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.2)
    assert AN._err(0.5, None, None) == (0.0, 0.0)


def test_trend_descriptive_below_three_rungs(fixture_run):
    """A 2-surviving-rung gate marks the trend descriptive (no NHST)."""
    gate = AN._read_gate(fixture_run["dirs"]["work"] / "gates")
    gate = json.loads(json.dumps(gate))
    gate["rungs"]["r4_trait"]["survived"] = False
    rng = np.random.default_rng(0)
    steered = fixture_run["f_cells"]
    t = AN.trend_test(steered, gate, "install", "ce", rng, 100)
    assert t["descriptive_only"] is True
    assert t["n_surviving_rungs"] == 2
    assert t["rho_observed"] is not None  # still reported descriptively
