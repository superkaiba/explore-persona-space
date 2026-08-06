"""CPU-only tests for the fu1 ANALYSIS leg (``scripts/issue2094_fu1_analysis.py``).

Pins (a) the derivation shapes against the COMMITTED parent artifacts (the
16 breached cells / 15 surviving families, and the ``clean_families`` mirror
of ``issue2094_fu1.derive_conf1_families``), and (b) at least one fail-loud
mutation probe per sub-analysis: synthetic score rows that would silently
shift a family mean must instead raise (duplicates, coverage mismatches,
unknown cells) or be visibly counted (None judge scores dropped, never
coerced; incoherent draws excluded + counted). No model, no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_analysis as A  # noqa: E402
import issue2094_fu1 as F  # noqa: E402
import issue2094_fu1_analysis as FA  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

WELLSEP_PATH = FA.DEFAULT_FMETRICS_DIR / "bootstrap_cis_wellsep.json"
ANCHORS_PATH = FA.DEFAULT_FMETRICS_DIR / "anchors.jsonl"


@pytest.fixture(scope="module")
def fragility() -> dict:
    return json.loads(FA.DEFAULT_FRAGILITY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def wellsep() -> dict:
    return json.loads(WELLSEP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def breached(fragility) -> list[tuple[str, str, str]]:
    return F.derive_breached_cells(fragility)


@pytest.fixture(scope="module")
def families(wellsep, breached) -> list[dict]:
    return F.derive_conf1_families(wellsep, set(breached))


@pytest.fixture(scope="module")
def anchors() -> dict:
    return FA.load_anchor_stats(ANCHORS_PATH)


@pytest.fixture(scope="module")
def ws_sets() -> tuple[set, set]:
    return W.load_wellsep(ANCHORS_PATH, W.MIN_SEPARATION)


# ── conventions + derivation-shape pins ────────────────────────────────


def test_bootstrap_conventions_pinned():
    """The brief's conventions ride the parent constants, never re-typed."""
    assert A.BOOTSTRAP_B == 10_000
    assert A.BOOTSTRAP_SEED == 20941
    assert W.MIN_SEPARATION == 0.5
    assert F.CONF1_DRAWS == 5
    assert F.CONF1_TEMPERATURE == 1.0
    assert F.CAPHIT_TRIGGER_FRAC == 0.02


def test_derivation_shapes_pinned(breached, families):
    assert len(breached) == F.EXPECTED_N_BREACHED == 16
    assert len(families) == F.EXPECTED_N_CONF1 == 15
    assert all(f["metric"] == "f_beh_prefix" for f in families)


def test_clean_families_mirror_matches_fu1_derivation(wellsep, breached, families):
    """``FA.clean_families`` (the assert-free mirror used for the re-run
    verdict) must reproduce ``derive_conf1_families`` on the parent artifact."""
    mirror = FA.clean_families(wellsep["steered_vs_null"], set(breached))
    assert mirror == [f["family"] for f in families]


def test_conf1_family_map_bijective(families):
    m = FA.conf1_family_map(families)
    assert len(m) == 15
    assert sorted(f["family"] for f in m.values()) == sorted(f["family"] for f in families)


# ── score routing: fail-loud probes ────────────────────────────────────


def _grid_coh_row(bk="ce|joint_mid|a1|A|steered", pid="p0", score=90):
    return {
        "kind": "grid",
        "rubric_id": A.COHERENCE_RUBRIC_ID,
        "block_key": bk,
        "pair_id": pid,
        "score": score,
    }


def test_route_scores_duplicate_grid_row_raises():
    with pytest.raises(AssertionError, match="duplicate judge score row"):
        FA.route_fu1_scores([_grid_coh_row(), _grid_coh_row()])


def test_route_scores_duplicate_stage2_row_raises():
    row = {
        "kind": "stage2",
        "rubric_id": "fp-bare",
        "cell": "fu1|cross|ce|L15|a1|A|null",
        "pair_id": "p0",
        "draw": 0,
        "rubric_kind": "prefix",
        "side": "a",
        "score": 50,
    }
    with pytest.raises(AssertionError, match="duplicate judge score row"):
        FA.route_fu1_scores([row, dict(row)])


def test_route_scores_unexpected_kind_raises():
    row = _grid_coh_row()
    row["kind"] = "anchor"  # fu1 waves carry no anchor rows
    with pytest.raises(AssertionError, match="unexpected score-row kind"):
        FA.route_fu1_scores([row])


# ── sub-analysis A: regen reduction probes ─────────────────────────────


@pytest.fixture(scope="module")
def mq_pair():
    pairs = A.BANK.build_pairs()
    return next(p for p in pairs if p.setting == "matched_query")


def _regen_fixture(mq_pair, *, coh=90.0, sa=20.0, sb=60.0):
    bk = "ce|joint_mid|a1|A|steered"
    shard_row = {
        "block_key": bk,
        "pair_id": mq_pair.pair_id,
        "slot": "ce",
        "layer_variant": "joint_mid",
        "dose": "a1",
        "alpha": 1.0,
        "vec_type": "A",
        "arm": "steered",
        "setting": "matched_query",
        "context_a": mq_pair.a,
        "context_b": mq_pair.b,
        "layers": [12, 13],
        "donor_pair_id": None,
        "cap_hit": False,
    }
    sc = FA.Fu1Scores()
    sc.grid_coh[(bk, mq_pair.pair_id)] = coh
    sc.grid_beh[(bk, mq_pair.pair_id, "prefix", "a")] = sa
    sc.grid_beh[(bk, mq_pair.pair_id, "prefix", "b")] = sb
    anchors = {
        (mq_pair.pair_id, "prefix"): {
            "floor": {"mean": 0.1},
            "ceiling": {"mean": 0.9},
            "separation": 0.8,
        }
    }
    pairs_by_id = {mq_pair.pair_id: mq_pair}
    breached = {("ce", "joint_mid", "a1")}
    return shard_row, sc, anchors, pairs_by_id, breached


def test_reduce_regen_rows_anchored_f_beh(mq_pair):
    shard_row, sc, anchors, pairs_by_id, breached = _regen_fixture(mq_pair)
    rows = FA.reduce_regen_rows([shard_row], sc, anchors, pairs_by_id, breached)
    assert len(rows) == 1
    rec = rows[0]["f_beh"]["prefix"]
    # delta = (60-20)/100 = 0.4; f = (0.4 - 0.1) / (0.9 - 0.1)
    assert rec["f_beh"] == pytest.approx((0.4 - 0.1) / 0.8, abs=1e-6)
    assert rows[0]["excluded_incoherent"] is False
    assert rows[0]["f_act"] is None
    assert rows[0]["degenerate_self"] is False


def test_reduce_regen_rows_extra_score_row_raises(mq_pair):
    """A score row with no matching staged rollout row (would silently shift
    a family mean if blended) must fail the coverage set-equality."""
    shard_row, sc, anchors, pairs_by_id, breached = _regen_fixture(mq_pair)
    sc.grid_coh[("ce|joint_mid|a1|A|steered", "phantom-pair")] = 95.0
    with pytest.raises(AssertionError, match="coverage"):
        FA.reduce_regen_rows([shard_row], sc, anchors, pairs_by_id, breached)


def test_reduce_regen_rows_none_score_dropped_and_counted(mq_pair):
    """A None (rule-9 content-dropped) judge score is DROPPED with a visible
    reason — never coerced to a number."""
    shard_row, sc, anchors, pairs_by_id, breached = _regen_fixture(mq_pair)
    sc.grid_beh[("ce|joint_mid|a1|A|steered", mq_pair.pair_id, "prefix", "a")] = None
    rows = FA.reduce_regen_rows([shard_row], sc, anchors, pairs_by_id, breached)
    rec = rows[0]["f_beh"]["prefix"]
    assert rec["f_beh"] is None
    assert rec["missing"] == "judge_dropped"


def test_reduce_regen_rows_incoherent_excluded_and_marked(mq_pair):
    shard_row, sc, anchors, pairs_by_id, breached = _regen_fixture(mq_pair, coh=50.0)
    rows = FA.reduce_regen_rows([shard_row], sc, anchors, pairs_by_id, breached)
    rec = rows[0]["f_beh"]["prefix"]
    assert rows[0]["excluded_incoherent"] is True
    assert rec["f_beh"] is None
    assert rec["excluded_incoherent_raw"] == pytest.approx((0.4 - 0.1) / 0.8, abs=1e-6)


def test_reduce_regen_rows_outside_breached_set_raises(mq_pair):
    shard_row, sc, anchors, pairs_by_id, _ = _regen_fixture(mq_pair)
    with pytest.raises(AssertionError, match="outside the breached set"):
        FA.reduce_regen_rows([shard_row], sc, anchors, pairs_by_id, {("pe", "L14", "a1")})


def test_swap_rows_set_equality_enforced():
    parent = [
        {
            "block_key": "bk1",
            "pair_id": "p1",
            "slot": "ce",
            "layer_variant": "joint_mid",
            "dose": "a1",
        },
        {"block_key": "bk2", "pair_id": "p2", "slot": "pe", "layer_variant": "L14", "dose": "a2"},
    ]
    regen_ok = [
        {
            "block_key": "bk1",
            "pair_id": "p1",
            "slot": "ce",
            "layer_variant": "joint_mid",
            "dose": "a1",
        }
    ]
    swapped, removed = FA.swap_rows(parent, regen_ok, {("ce", "joint_mid", "a1")})
    assert len(swapped) == 2 and len(removed) == 1
    regen_bad = [
        {
            "block_key": "bk1",
            "pair_id": "OTHER",
            "slot": "ce",
            "layer_variant": "joint_mid",
            "dose": "a1",
        }
    ]
    with pytest.raises(AssertionError, match="swap mismatch"):
        FA.swap_rows(parent, regen_bad, {("ce", "joint_mid", "a1")})


def test_recompute_caphit_2048_arithmetic():
    rows = [
        {
            "slot": "ce",
            "layer_variant": "joint_mid",
            "dose": "a1",
            "arm": "steered",
            "cap_hit": True,
        },
        {
            "slot": "ce",
            "layer_variant": "joint_mid",
            "dose": "a1",
            "arm": "steered",
            "cap_hit": False,
        },
        {"slot": "ce", "layer_variant": "joint_mid", "dose": "a1", "arm": "null", "cap_hit": False},
    ]
    pooled = FA.recompute_caphit_2048(rows)
    cell = pooled[("ce", "joint_mid", "a1")]
    assert cell["steered"] == {"n": 2, "cap_hit": 1, "cap_hit_frac": 0.5}
    assert cell["null"]["cap_hit_frac"] == 0.0


# ── sub-analysis B: conf1 reduction probes ─────────────────────────────


def _synth_conf1_scores(families, anchors, ws, *, drop=None, incoherent=None) -> FA.Fu1Scores:
    """Synthetic stage2 scores over the REAL 15 cells: coherence 100 on every
    (cell, pair, draw); behavior sides constant per arm (steered delta 0.6,
    null delta 0.2) on well-separated prefix pairs. ``drop`` / ``incoherent``
    poke one (cell, pair, draw) each."""
    pairs = A.BANK.build_pairs()
    by_setting: dict[str, list[str]] = {}
    for p in pairs:
        by_setting.setdefault(p.setting, []).append(p.pair_id)
    sides = {"steered": (10.0, 70.0), "null": (10.0, 30.0)}
    sc = FA.Fu1Scores()
    for prefix, fam in FA.conf1_family_map(families).items():
        kind = fam["metric"].removeprefix("f_beh_")
        for arm in ("steered", "null"):
            cell = f"{prefix}|{arm}"
            sa, sb = sides[arm]
            for pid in sorted(by_setting[fam["setting"]]):
                for d in range(F.CONF1_DRAWS):
                    coh = 100.0
                    if incoherent == (cell, pid, d):
                        coh = 10.0
                    sc.s2_coh[(cell, pid, d)] = coh
                    if (pid, kind) not in ws:
                        continue
                    a_val: float | None = sa
                    if drop == (cell, pid, d):
                        a_val = None
                    sc.s2_beh[(cell, pid, d, kind, "a")] = a_val
                    sc.s2_beh[(cell, pid, d, kind, "b")] = sb
    return sc


def _expected_family_mean(fam, anchors, ws, delta):
    pairs = A.BANK.build_pairs()
    kind = fam["metric"].removeprefix("f_beh_")
    fs = []
    for p in pairs:
        if p.setting != fam["setting"] or (p.pair_id, kind) not in ws:
            continue
        st = anchors[(p.pair_id, kind)]
        fl, ce = st["floor"]["mean"], st["ceiling"]["mean"]
        fs.append((delta - fl) / (ce - fl))
    return float(np.mean(fs)), len(fs)


def test_reduce_conf1_means_and_counts(families, anchors, ws_sets):
    ws, _ = ws_sets
    sc = _synth_conf1_scores(families, anchors, ws)
    reduced = FA.reduce_conf1(sc, families, anchors, ws)
    assert sorted(reduced) == sorted(f["family"] for f in families)
    fam = families[0]
    rec = reduced[fam["family"]]["arms"]["steered"]
    exp_mean, n_ws = _expected_family_mean(fam, anchors, ws, delta=0.6)
    got = rec["values"][~np.isnan(rec["values"])]
    assert len(got) == n_ws
    assert float(np.mean(got)) == pytest.approx(exp_mean, abs=1e-9)
    assert rec["n_judge_dropped"] == 0 and rec["n_incoherent"] == 0


def test_reduce_conf1_none_score_dropped_never_coerced(families, anchors, ws_sets):
    """Dropping one draw's judge score must shrink that pair's kept-draw count
    (visible) and leave the pair value the mean of the REMAINING draws —
    identical here since draws are constant — never a coerced zero."""
    ws, _ = ws_sets
    fam = families[0]
    prefix = "|".join(
        ["fu1", fam["setting"], fam["slot"], fam["layer_variant"], fam["dose"], fam["vec_type"]]
    )
    kind = fam["metric"].removeprefix("f_beh_")
    pid = sorted(
        p
        for p, k in ws
        if k == kind
        and any(bp.pair_id == p and bp.setting == fam["setting"] for bp in A.BANK.build_pairs())
    )[0]
    cell = f"{prefix}|steered"
    sc = _synth_conf1_scores(families, anchors, ws, drop=(cell, pid, 0))
    reduced = FA.reduce_conf1(sc, families, anchors, ws)
    rec = reduced[fam["family"]]["arms"]["steered"]
    assert rec["n_judge_dropped"] == 1
    assert rec["per_pair_n_kept_draws"][pid] == F.CONF1_DRAWS - 1
    # constant draws: value unchanged => the drop did NOT get coerced to 0
    exp_mean, _ = _expected_family_mean(fam, anchors, ws, delta=0.6)
    got = rec["values"][~np.isnan(rec["values"])]
    assert float(np.mean(got)) == pytest.approx(exp_mean, abs=1e-9)


def test_reduce_conf1_incoherent_draw_excluded_and_counted(families, anchors, ws_sets):
    ws, _ = ws_sets
    fam = families[0]
    prefix = "|".join(
        ["fu1", fam["setting"], fam["slot"], fam["layer_variant"], fam["dose"], fam["vec_type"]]
    )
    kind = fam["metric"].removeprefix("f_beh_")
    pid = sorted(
        p
        for p, k in ws
        if k == kind
        and any(bp.pair_id == p and bp.setting == fam["setting"] for bp in A.BANK.build_pairs())
    )[0]
    cell = f"{prefix}|null"
    sc = _synth_conf1_scores(families, anchors, ws, incoherent=(cell, pid, 1))
    reduced = FA.reduce_conf1(sc, families, anchors, ws)
    rec = reduced[fam["family"]]["arms"]["null"]
    assert rec["n_incoherent"] == 1
    assert rec["per_pair_n_kept_draws"][pid] == F.CONF1_DRAWS - 1


def test_reduce_conf1_unknown_cell_raises(families, anchors, ws_sets):
    ws, _ = ws_sets
    sc = _synth_conf1_scores(families, anchors, ws)
    sc.s2_coh[("fu1|cross|ce|L99|a1|A|steered", "p0", 0)] = 100.0
    with pytest.raises(AssertionError, match="unknown conf1 cell keys"):
        FA.reduce_conf1(sc, families, anchors, ws)


def test_conf1_reads_disjoint_verdict(families, ws_sets):
    """UNIFORM synthetic anchors (floor 0.1 / ceiling 0.9 for every pair):
    per-pair values are then constant within each arm, the pair-clustered
    bootstrap degenerates to a point CI, and steered (delta 0.6 -> f=0.625)
    vs null (delta 0.2 -> f=0.125) must read disjoint + steered_above =>
    all 15 confirmed. (With the REAL anchors a constant judged delta still
    yields overlapping CIs from per-pair anchor variance — that regime is
    covered by the mean/count tests above.)"""
    ws_real, _ = ws_sets
    pairs = A.BANK.build_pairs()
    synth_anchors = {
        (p.pair_id, kind): {
            "floor": {"mean": 0.1},
            "ceiling": {"mean": 0.9},
            "separation": 0.8,
        }
        for p in pairs
        for kind in ("prefix", "query")
    }
    # every pair well-separated on the family kind (prefix), plus the real
    # query-kind entries so nothing else changes shape
    ws = {(p.pair_id, "prefix") for p in pairs} | ws_real
    sc = _synth_conf1_scores(families, synth_anchors, ws)
    reduced = FA.reduce_conf1(sc, families, synth_anchors, ws)
    rows, summary = FA.conf1_reads(reduced, n_boot=500, seed=A.BOOTSTRAP_SEED)
    assert summary["n_families"] == 15
    assert summary["n_confirmed"] == 15
    for r in rows:
        assert r["confirmed"] and r["direction"] == "steered_above"
        assert r["steered"]["observed_mean"] == pytest.approx(0.625, abs=1e-9)
        assert r["null"]["observed_mean"] == pytest.approx(0.125, abs=1e-9)
