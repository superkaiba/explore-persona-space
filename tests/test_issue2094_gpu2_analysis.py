"""CPU-only tests for the #2094 gpu2_mq_replacement_prefix ANALYSIS leg.

Groups:
(a) grid/tails enumeration pinned against the driver constants (150 families
    -> 300 read tails, all matched_query x ce x Type-A);
(b) verdict-taxonomy parity: ``_verdict_min_pairs`` == fu2's ``_verdict`` at
    the headline floor, and the labeled small-n regime behavior at 4 pairs;
(c) fail-loud MUTATION PROBES: the gate-reproduction assert, the score-key
    coverage asserts, the cap-hit crosscheck, and the parts-dir regime
    refusal each demonstrably FIRE on tampered inputs;
(d) the gpu2-axis bootstrap battery: batched == naive reference on the
    5-pair axis, and the wellsep keep predicate excludes the gate-failing
    pair from the wellsep battery but not the unrestricted one.

Fully synthetic — no network, no eval_results/ or data/ reads (sparse-cone
safe); imports follow the established test_issue2094_* convention.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_analysis as A  # noqa: E402
import issue2094_fu2_analysis as FU2A  # noqa: E402
import issue2094_gpu2 as G2  # noqa: E402
import issue2094_gpu2_analysis as GA  # noqa: E402
import issue2094_gpu2_bank as G2B  # noqa: E402
import issue2094_run as R  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

# ── (a) enumeration ──────────────────────────────────────────────────────


def test_expected_tails_cover_the_gpu2_grid():
    families = G2.enumerate_gpu2_families(A.N_LAYERS)
    assert R.grid_totals(families) == G2.EXPECTED_GPU2_TOTALS
    tails = GA.expected_family_tails(families)
    assert len(tails) == 300  # 150 families x {f_act, f_beh_prefix}
    for tail in tails:
        setting, slot, _lv, _dose, vt, metric = tail.split("|")
        assert setting == "matched_query" and slot == "ce" and vt == "A"
        assert metric in ("f_act", "f_beh_prefix")
    coh, beh = GA.expected_score_keys(families)
    assert len(coh) == 1500 and len(beh) == 3000


def test_gpu2_pair_axis_is_the_five_reformed_pairs():
    pids = GA.gpu2_pair_axis()
    assert pids == sorted(p.pair_id for p in G2B.build_gpu2_pairs())
    assert len(pids) == 5


# ── (b) verdict-taxonomy parity ─────────────────────────────────────────


def _read(comparable=True, disjoint=True, direction="steered_above", n_pairs=4):
    return {
        "comparable": comparable,
        "cis_disjoint": disjoint,
        "direction": direction,
        "n_pairs_used": n_pairs,
    }


def test_verdict_min_pairs_parity_with_fu2_at_headline_floor():
    for comparable in (True, False):
        for disjoint in (True, False):
            for direction in ("steered_above", "steered_below"):
                for n_pairs in (1, 3, 4, 5, 10):
                    for compromised in (True, False):
                        read = _read(comparable, disjoint, direction, n_pairs)
                        assert GA._verdict_min_pairs(
                            read, compromised, W.MIN_PAIRS_HEADLINE
                        ) == FU2A._verdict(read, compromised), (read, compromised)


def test_smalln_regime_labels_at_four_pairs():
    read = _read(n_pairs=4)
    # Headline convention: 4 < 5 caps at separating_lt5_pairs.
    assert FU2A._verdict(read, False) == "separating_lt5_pairs"
    # Labeled small-n regime (>=3): clean unless compromised.
    assert GA._verdict_min_pairs(read, False, GA.SMALLN_MIN_PAIRS) == "clean_separating"
    assert GA._verdict_min_pairs(read, True, GA.SMALLN_MIN_PAIRS) == "separating_compromised"
    # Below even the small-n floor the label stays explicit.
    assert (
        GA._verdict_min_pairs(_read(n_pairs=2), False, GA.SMALLN_MIN_PAIRS)
        == "separating_lt3_pairs"
    )


# ── (c) fail-loud mutation probes ───────────────────────────────────────


def _synthetic_gate() -> tuple[list[dict], dict, dict, dict]:
    """A tiny real pass through G2.gate_separations/gate_verdict on synthetic
    coherent scores, plus the recorded-report shape the pod writes."""
    pairs = G2B.build_gpu2_pairs()
    coh: dict = {}
    beh: dict = {}
    draws_by_ctx: dict[str, list[int]] = {}
    rid_a, rid_b = G2.GATE_RUBRIC_IDS
    for k, pair in enumerate(pairs):
        for cid, hi in ((pair.a, False), (pair.b, True)):
            draws_by_ctx[cid] = [0, 1]
            for d in (0, 1):
                coh[(cid, d)] = 95.0
                # floors near fp-bare, ceilings near fp-conv2; q-index scaled
                beh[(cid, d, rid_a)] = 10.0 if hi else 90.0
                beh[(cid, d, rid_b)] = (80.0 - 5 * k) if hi else 5.0
    sep_rows = G2.gate_separations(coh, beh, draws_by_ctx)
    verdict = G2.gate_verdict(sep_rows)
    prov = {"n_floor_rows": 10, "n_ceiling_rows": 10}
    recorded = {
        "regime_fp": "test",
        "judge_mode": "mock",
        "verdict": copy.deepcopy(verdict),
        "separations": copy.deepcopy(sep_rows),
        **prov,
    }
    return sep_rows, verdict, prov, recorded


def test_gate_reproduction_passes_then_fires_on_tampered_recorded():
    sep_rows, verdict, prov, recorded = _synthetic_gate()
    out = GA.assert_gate_reproduction(sep_rows, verdict, prov, recorded)
    assert out["passed"] is True

    # Mutation probe 1: a perturbed recorded separation MUST fire the assert.
    tampered = copy.deepcopy(recorded)
    tampered["separations"][0]["separation"] += 1e-9
    with pytest.raises(AssertionError, match="separation"):
        GA.assert_gate_reproduction(sep_rows, verdict, prov, tampered)

    # Mutation probe 2: a flipped per-pair pass MUST fire the assert.
    tampered2 = copy.deepcopy(recorded)
    tampered2["verdict"]["per_pair"][0]["passes"] = not tampered2["verdict"]["per_pair"][0][
        "passes"
    ]
    with pytest.raises(AssertionError, match="per-pair"):
        GA.assert_gate_reproduction(sep_rows, verdict, prov, tampered2)

    # Mutation probe 3: a wrong floor-row count MUST fire the assert.
    with pytest.raises(AssertionError):
        GA.assert_gate_reproduction(sep_rows, verdict, {**prov, "n_floor_rows": 9}, recorded)


def test_wellsep_sets_exclude_gate_failing_pairs():
    _sep_rows, verdict, _prov, _recorded = _synthetic_gate()
    # Force one pair to fail the gate in a copy of the verdict.
    v = copy.deepcopy(verdict)
    v["per_pair"][1]["passes"] = False
    ws, ws_any = GA.wellsep_sets_from_verdict(v)
    failing = v["per_pair"][1]["pair_id"]
    assert failing not in ws_any and (failing, "prefix") not in ws
    assert len(ws_any) == 4 and len(ws) == 4


def test_caphit_crosscheck_fires_on_tampered_manifest():
    pooled = {
        ("ce", "L0", "a1"): {
            "steered": {"n": 5, "cap_hit": 0, "cap_hit_frac": 0.0},
            "null": {"n": 5, "cap_hit": 1, "cap_hit_frac": 0.2},
        }
    }
    manifest = {
        "max_new_tokens": G2.GPU2_MAX_NEW_TOKENS,
        "cells": [
            {
                "slot": "ce",
                "layer_variant": "L0",
                "dose": "a1",
                "steered": {"n": 5, "cap_hit": 0, "cap_hit_frac": 0.0},
                "null": {"n": 5, "cap_hit": 1, "cap_hit_frac": 0.2},
            }
        ],
    }
    assert GA.crosscheck_gpu2_caphit(pooled, manifest)["passed"] is True
    # Mutation probe: a tampered manifest count MUST fire.
    bad = copy.deepcopy(manifest)
    bad["cells"][0]["null"]["cap_hit"] = 0
    with pytest.raises(AssertionError):
        GA.crosscheck_gpu2_caphit(pooled, bad)
    # Mutation probe: a wrong run cap MUST fire.
    with pytest.raises(AssertionError):
        GA.crosscheck_gpu2_caphit(pooled, {**manifest, "max_new_tokens": 1024})
    # Subset mode tolerates extra manifest cells but never missing ones.
    extra = copy.deepcopy(manifest)
    extra["cells"].append({**copy.deepcopy(manifest["cells"][0]), "dose": "a2"})
    assert GA.crosscheck_gpu2_caphit(pooled, extra, subset=True)["passed"] is True
    with pytest.raises(AssertionError):
        GA.crosscheck_gpu2_caphit(pooled, extra)


def test_score_coverage_asserts_fire_on_missing_key(tmp_path):
    """The main()-level coverage check is a set-equality assert; pin the exact
    predicate here on a 1-family synthetic grid."""
    families = G2.enumerate_gpu2_families(A.N_LAYERS)[:1]
    expected_coh, expected_beh = GA.expected_score_keys(families)
    routed_coh = dict.fromkeys(expected_coh, 90.0)
    routed_beh = dict.fromkeys(expected_beh, 50.0)
    assert set(routed_coh) == expected_coh and set(routed_beh) == expected_beh
    # Mutation probe: dropping one coherence key breaks set-equality.
    routed_coh.popitem()
    assert set(routed_coh) != expected_coh
    # Mutation probe: an extra behavior key breaks set-equality.
    routed_beh[("bogus|key", "pid", "prefix", "a")] = 1.0
    assert set(routed_beh) != expected_beh


def test_parts_regime_mismatch_hard_refuses(tmp_path):
    manifest = tmp_path / "parts_manifest.json"
    regime = GA.analysis_regime()
    manifest.write_text('{"regime": "someother", "done": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        GA.check_parts_regime(manifest, regime)
    manifest.write_text(f'{{"regime": "{regime}", "done": ["x"]}}', encoding="utf-8")
    assert GA.check_parts_regime(manifest, regime) == {"x"}
    assert GA.check_parts_regime(tmp_path / "absent.json", regime) == set()


# ── (d) bootstrap battery on the gpu2 pair axis ─────────────────────────


def _synthetic_cell_rows() -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for arm in R.ARMS:
        for pid in GA.gpu2_pair_axis():
            rows.append(
                {
                    "arm": arm,
                    "setting": "matched_query",
                    "slot": "ce",
                    "layer_variant": "L0",
                    "dose": "a1",
                    "vec_type": "A",
                    "pair_id": pid,
                    "f_act": float(rng.normal(1.0 if arm == "steered" else 0.0, 0.01)),
                    "f_beh": {"prefix": {"f_beh": float(rng.normal(0.5, 0.01))}},
                }
            )
    return rows


def test_gpu2_battery_matches_naive_reference_and_respects_wellsep():
    rows = _synthetic_cell_rows()
    pids = GA.gpu2_pair_axis()
    ws_all = {(p, "prefix") for p in pids}
    fams = GA.compute_gpu2_family_battery(rows, ws_all, set(pids), n_boot=64)
    # 2 arms x 1 (lv, dose) x 2 metrics = 4 family keys.
    assert len(fams) == 4
    key = "steered|matched_query|ce|L0|a1|A|f_act"
    assert fams[key]["n_pairs_used"] == 5

    # Batched bootstrap == the serial reference twin on this axis.
    values = np.array([[r["f_act"] for r in rows if r["arm"] == "steered"]], dtype=float).T
    batched = A.bootstrap_family_means_batched(values, 64, A.BOOTSTRAP_SEED)
    naive = A._bootstrap_family_means_naive(values, 64, A.BOOTSTRAP_SEED)
    np.testing.assert_allclose(batched, naive, rtol=0, atol=1e-12)

    # Wellsep restriction drops the excluded pair from every family read.
    drop = pids[1]
    ws = {(p, "prefix") for p in pids if p != drop}
    fams_ws = GA.compute_gpu2_family_battery(rows, ws, {p for p, _ in ws}, n_boot=64)
    assert fams_ws[key]["n_pairs_used"] == 4
    reads, _summary = W.steered_vs_null_reads(fams_ws)
    tail = "matched_query|ce|L0|a1|A|f_act"
    assert reads[tail]["comparable"] is True
    assert reads[tail]["n_pairs_used"] == 4
    # Clearly-separated synthetic arms: disjoint CIs, steered above.
    assert reads[tail]["cis_disjoint"] is True
    assert reads[tail]["direction"] == "steered_above"


def test_spearman_ranks_with_ties():
    assert GA.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert GA.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert GA.spearman([1, 2], [2, 1]) is None
    r = GA.spearman([1, 1, 2, 3], [1, 1, 2, 3])
    assert r == pytest.approx(1.0)
