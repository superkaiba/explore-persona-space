"""CPU-only regression tests for the issue #2094 fu2_span_slots ANALYSIS leg.

Pins (follow-up scope duties):

- the expected-grid enumeration (170 family reads = qtext 70 + pspan_tmpl 50 +
  pspan_text 50; 2,400 coherence keys; 6,600 behavior keys) consistent with
  the fu2 driver's pinned totals;
- fail-loud mutation probes: a family silently missing from either read set
  raises; a family whose pooled cap-hit entry is missing raises (compromise
  labeling can never be silently skipped); a cap-hit manifest count mismatch
  raises;
- compromise labeling: steered pooled cap-hit > 2% => ``compromised: true`` +
  the ``separating_compromised`` verdict (computed + labeled, never dropped);
- the verdict lattice (not_comparable / not_separating / lt5-pairs / clean);
- per-family QC counting (incoherent / judge-dropped / empty / cap-hit).

No model, no GPU, no network, no staged fu2 data (synthetic fixtures only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_fu2 as F2  # noqa: E402
import issue2094_fu2_analysis as M  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    return BANK.build_pairs()


# ── expected-grid enumeration pins ──────────────────────────────────────


def test_expected_family_tails_pinned(pairs):
    tails = M.expected_family_tails(pairs)
    assert len(tails) == 170
    by_slot = {s: sorted(t for t in tails if t.split("|")[1] == s) for s in F2.FU2_SLOTS}
    assert len(by_slot["qtext"]) == 70
    assert len(by_slot["pspan_tmpl"]) == 50
    assert len(by_slot["pspan_text"]) == 50
    # pspan slots carry NO matched_prefix families (pair eligibility rule).
    for slot in ("pspan_tmpl", "pspan_text"):
        assert not any(t.startswith("matched_prefix|") for t in by_slot[slot])
    # every tail is Type-A at a joint variant over the dose ladder.
    for t in tails:
        _setting, _slot, lv, _dose, vt, metric = t.split("|")
        assert vt == "A"
        assert lv in F2.FU2_VARIANTS
        assert metric in ("f_act", "f_beh_prefix", "f_beh_query")


def test_expected_score_keys_match_driver_totals(pairs):
    fams = F2.enumerate_fu2_families(pairs)
    coh, beh = M.expected_score_keys(fams, {p.pair_id: p for p in pairs})
    assert len(coh) == F2.EXPECTED_FU2_TOTALS["cells_total"] == 2400
    assert len(beh) == 6600  # kinds x sides over the eligible pair sets
    # spot shape: every beh key's (block, pair) is a coh key too.
    for key in list(beh)[:50]:
        assert (key[0], key[1]) in coh


def test_eligible_settings_follow_driver_rule(pairs):
    assert M.eligible_settings(pairs, "qtext") == ("matched_prefix", "matched_query", "cross")
    assert M.eligible_settings(pairs, "pspan_tmpl") == ("matched_query", "cross")
    assert M.eligible_settings(pairs, "pspan_text") == ("matched_query", "cross")


def test_analysis_regime_keys_output_affecting_knobs():
    assert M.analysis_regime(False) != M.analysis_regime(True)


# ── verdict-table fixtures ──────────────────────────────────────────────


def _read(comparable=True, disjoint=True, direction="steered_above", n_pairs=6):
    if not comparable:
        return {"comparable": False}
    return {
        "comparable": True,
        "steered_mean": 0.5,
        "null_mean": 0.1,
        "steered_ci": [0.4, 0.6],
        "null_ci": [0.0, 0.2],
        "n_pairs_used": n_pairs,
        "steered_ci_excludes_null_mean": True,
        "cis_disjoint": disjoint,
        "direction": direction,
    }


def _caphit(frac_steered=0.0, n=30):
    hits = round(frac_steered * n)
    return {
        "steered": {"n": n, "cap_hit": hits, "cap_hit_frac": hits / n},
        "null": {"n": n, "cap_hit": 0, "cap_hit_frac": 0.0},
    }


def _qc():
    return {
        "n_rows": 30,
        "n_excluded_incoherent": 1,
        "n_empty_completion": 0,
        "n_cap_hit": 0,
        "n_judge_dropped": 0,
        "n_anchor_missing": 0,
        "incoherent_frac": 1 / 30,
    }


TAIL_A = "cross|qtext|joint_all|a1|A|f_beh_query"
TAIL_B = "matched_query|pspan_text|joint_mid|replace|A|f_act"


def _fixture(frac_steered_b=0.0):
    tails = {TAIL_A, TAIL_B}
    reads = {TAIL_A: _read(), TAIL_B: _read()}
    caphit = {
        ("qtext", "joint_all", "a1"): _caphit(0.0),
        ("pspan_text", "joint_mid", "replace"): _caphit(frac_steered_b),
    }
    qc = {
        ("qtext", "joint_all", "a1", "cross", "steered"): _qc(),
        ("qtext", "joint_all", "a1", "cross", "null"): _qc(),
        ("pspan_text", "joint_mid", "replace", "matched_query", "steered"): _qc(),
        ("pspan_text", "joint_mid", "replace", "matched_query", "null"): _qc(),
    }
    return tails, reads, caphit, qc


# ── fail-loud mutation probes ───────────────────────────────────────────


def test_missing_family_in_wellsep_reads_raises():
    tails, reads, caphit, qc = _fixture()
    broken = {TAIL_A: reads[TAIL_A]}  # TAIL_B silently missing
    with pytest.raises(AssertionError, match="wellsep read families"):
        M.build_verdict_table(broken, dict(reads), caphit, qc, tails)


def test_missing_family_in_unrestricted_reads_raises():
    tails, reads, caphit, qc = _fixture()
    broken = {TAIL_A: reads[TAIL_A]}
    with pytest.raises(AssertionError, match="unrestricted read families"):
        M.build_verdict_table(dict(reads), broken, caphit, qc, tails)


def test_extra_unexpected_family_raises():
    tails, reads, caphit, qc = _fixture()
    extra = dict(reads)
    extra["cross|qtext|joint_all|a2|A|f_beh_query"] = _read()
    with pytest.raises(AssertionError, match="wellsep read families"):
        M.build_verdict_table(extra, dict(reads), caphit, qc, tails)


def test_missing_caphit_entry_raises():
    tails, reads, caphit, qc = _fixture()
    del caphit[("pspan_text", "joint_mid", "replace")]
    with pytest.raises(AssertionError, match="no pooled cap-hit entry"):
        M.build_verdict_table(dict(reads), dict(reads), caphit, qc, tails)


def test_crosscheck_caphit_count_mismatch_raises():
    pooled = {("qtext", "joint_all", "a1"): _caphit(0.1)}
    manifest = {
        "max_new_tokens": F2.FU2_MAX_NEW_TOKENS,
        "cells": [
            {
                "slot": "qtext",
                "layer_variant": "joint_all",
                "dose": "a1",
                "steered": {"n": 30, "cap_hit": 0, "cap_hit_frac": 0.0},  # != pooled
                "null": {"n": 30, "cap_hit": 0, "cap_hit_frac": 0.0},
            }
        ],
    }
    with pytest.raises(AssertionError):
        M.crosscheck_caphit(pooled, manifest)
    manifest["cells"][0]["steered"] = dict(pooled[("qtext", "joint_all", "a1")]["steered"])
    out = M.crosscheck_caphit(pooled, manifest)
    assert out["passed"] is True and out["n_pooled_cells"] == 1


def test_crosscheck_caphit_wrong_cap_raises():
    pooled = {("qtext", "joint_all", "a1"): _caphit(0.0)}
    manifest = {"max_new_tokens": 1024, "cells": []}
    with pytest.raises(AssertionError):
        M.crosscheck_caphit(pooled, manifest)


# ── compromise labeling + verdict lattice ───────────────────────────────


def test_compromised_family_flagged_and_never_dropped():
    tails, reads, caphit, qc = _fixture(frac_steered_b=0.10)  # > 2% trigger
    table = M.build_verdict_table(dict(reads), dict(reads), caphit, qc, tails)
    by_family = {r["family"]: r for r in table}
    assert set(by_family) == tails  # nothing dropped
    assert by_family[TAIL_B]["compromised"] is True
    assert by_family[TAIL_B]["verdict"] == "separating_compromised"
    assert by_family[TAIL_B]["wellsep"]["steered_mean"] == 0.5  # read still computed
    assert by_family[TAIL_A]["compromised"] is False
    assert by_family[TAIL_A]["verdict"] == "clean_separating"


def test_verdict_lattice():
    assert M._verdict({"comparable": False}, False) == "not_comparable"
    assert M._verdict(_read(disjoint=False), False) == "not_separating"
    assert M._verdict(_read(direction="steered_below"), False) == "not_separating"
    assert M._verdict(_read(n_pairs=3), False).startswith("separating_lt")
    assert M._verdict(_read(), True) == "separating_compromised"
    assert M._verdict(_read(), False) == "clean_separating"


def test_per_slot_summary_counts():
    tails, reads, caphit, qc = _fixture(frac_steered_b=0.10)
    table = M.build_verdict_table(dict(reads), dict(reads), caphit, qc, tails)
    summary = M.per_slot_summary(table)
    assert summary["qtext"]["n_clean_separating"] == 1
    assert summary["pspan_text"]["n_clean_separating"] == 0
    assert summary["pspan_text"]["n_separating_incl_compromised"] == 1
    assert summary["qtext"]["clean_families"] == [TAIL_A]


# ── per-family QC counting ──────────────────────────────────────────────


def test_family_qc_counts():
    rows = [
        {
            "slot": "qtext",
            "layer_variant": "joint_all",
            "dose": "a1",
            "setting": "cross",
            "arm": "steered",
            "excluded_incoherent": True,
            "empty_completion": False,
            "cap_hit": True,
            "f_beh": {"query": {"f_beh": None, "missing": "judge_dropped"}},
        },
        {
            "slot": "qtext",
            "layer_variant": "joint_all",
            "dose": "a1",
            "setting": "cross",
            "arm": "steered",
            "excluded_incoherent": False,
            "empty_completion": True,
            "cap_hit": False,
            "f_beh": {
                "query": {"f_beh": 0.4},
                "prefix": {"f_beh": None, "missing": "anchor_missing"},
            },
        },
    ]
    qc = M.family_qc(rows)
    a = qc[("qtext", "joint_all", "a1", "cross", "steered")]
    assert a["n_rows"] == 2
    assert a["n_excluded_incoherent"] == 1
    assert a["n_empty_completion"] == 1
    assert a["n_cap_hit"] == 1
    assert a["n_judge_dropped"] == 1
    assert a["n_anchor_missing"] == 1
    assert a["incoherent_frac"] == pytest.approx(0.5)


# ── parent comparables ──────────────────────────────────────────────────


def test_parent_comparables_filters_and_flags():
    parent = {
        "steered_vs_null": {
            "cross|qspan|joint_all|a1|A|f_beh_query": _read(),
            "cross|qspan|joint_all|a1|B|f_beh_query": _read(),  # vt B: excluded
            "cross|pe|joint_all|a1|A|f_beh_query": _read(),  # slot pe: excluded
            "cross|ce|L10|a1|A|f_beh_query": _read(),  # per-layer lv: excluded
            "cross|ce|joint_mid|a4|A|f_act": _read(),
        }
    }
    breached = {("ce", "joint_mid", "a4")}
    out = M.parent_comparables(parent, breached)
    fams = sorted(r["family"] for r in out["rows"])
    assert fams == [
        "cross|ce|joint_mid|a4|A|f_act",
        "cross|qspan|joint_all|a1|A|f_beh_query",
    ]
    by = {r["family"]: r for r in out["rows"]}
    assert by["cross|ce|joint_mid|a4|A|f_act"]["compromised_1024_caphit"] is True
    assert by["cross|ce|joint_mid|a4|A|f_act"]["verdict"] == "separating_compromised"
    assert out["per_slot"]["qspan"]["n_clean_separating"] == 1
