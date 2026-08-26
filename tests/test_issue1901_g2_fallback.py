"""#1901 mlp-scaling-densify G2 gate: `_g2_gate` three-state verdict semantics.

Pins the plan-v13 §4 c1 contract on synthetic fixtures (pure function — no GPU,
no downloads, no staged stores): recorded-vs-realized sel-sha comparison where a
mismatch does NOT halt by itself — statistical parity of the fresh ridge refits
vs the banked bigN R² at tolerance decides FALLBACK-PARITY-PASS (with
``downgrade_recorded=True``) vs FAIL.
"""

import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from issue1901_paper_densify_mlp import GateVerdict, _g2_gate  # noqa: E402

RECORDED = {"lmsys_150k": "a" * 64, "lmsys_500k": "b" * 64}
BANKED = {"lmsys_150k": 0.7555, "lmsys_500k": 0.7609}
TOL = 0.02


def test_g2_exact_match_passes():
    v = _g2_gate(
        realized_shas=dict(RECORDED),
        recorded_shas=RECORDED,
        refit_r2s=dict(BANKED),
        banked_r2s=BANKED,
        tol=TOL,
    )
    assert isinstance(v, GateVerdict)
    assert v.verdict == "PASS"
    assert v.downgrade_recorded is False
    assert v.detail["mismatched"] == []
    assert all(p["sha_match"] for p in v.detail["points"].values())


def test_g2_exact_match_passes_even_with_parity_breach():
    # Sha-exact ⇒ G2 PASS regardless of parity: fold-level drift on a
    # sha-matched selection is the sha-conditional RUNG-parity halt's job
    # (DENSE_PARITY_ANCHORS kind="sha-conditional"), not G2's.
    v = _g2_gate(
        realized_shas=dict(RECORDED),
        recorded_shas=RECORDED,
        refit_r2s={"lmsys_150k": 0.60, "lmsys_500k": 0.7609},
        banked_r2s=BANKED,
        tol=TOL,
    )
    assert v.verdict == "PASS"
    assert v.downgrade_recorded is False


def test_g2_mismatch_with_parity_is_fallback_pass_no_halt():
    # Partial mismatch (one of two points) counts as mismatch; parity within
    # tol on EVERY recorded point ⇒ FALLBACK-PARITY-PASS, downgrade recorded,
    # and the pure function returns (never raises) — the no-halt branch.
    realized = {"lmsys_150k": "c" * 64, "lmsys_500k": RECORDED["lmsys_500k"]}
    refits = {"lmsys_150k": BANKED["lmsys_150k"] + 0.015, "lmsys_500k": BANKED["lmsys_500k"]}
    v = _g2_gate(realized, RECORDED, refits, BANKED, TOL)
    assert v.verdict == "FALLBACK-PARITY-PASS"
    assert v.downgrade_recorded is True
    assert v.detail["mismatched"] == ["lmsys_150k"]
    p = v.detail["points"]["lmsys_150k"]
    assert p["sha_match"] is False
    assert p["parity_within_tol"] is True
    assert p["abs_delta"] <= TOL
    # asdict-able (the runner persists asdict(v) into the aggregate's gates block)
    d = asdict(v)
    assert d["verdict"] == "FALLBACK-PARITY-PASS" and d["downgrade_recorded"] is True


def test_g2_mismatch_with_parity_breach_fails():
    realized = {"lmsys_150k": "c" * 64, "lmsys_500k": "d" * 64}
    refits = {"lmsys_150k": BANKED["lmsys_150k"] + 0.05, "lmsys_500k": BANKED["lmsys_500k"]}
    v = _g2_gate(realized, RECORDED, refits, BANKED, TOL)
    assert v.verdict == "FAIL"
    assert v.downgrade_recorded is False
    assert set(v.detail["mismatched"]) == {"lmsys_150k", "lmsys_500k"}
    assert v.detail["points"]["lmsys_150k"]["parity_within_tol"] is False


def test_g2_mismatch_anywhere_plus_breach_anywhere_fails():
    # The parity predicate is over ALL recorded points: a mismatch on 150k with
    # a parity breach on the (sha-matched) 500k point still FAILs — the
    # fallback requires every recorded point statistically consistent.
    realized = {"lmsys_150k": "c" * 64, "lmsys_500k": RECORDED["lmsys_500k"]}
    refits = {"lmsys_150k": BANKED["lmsys_150k"], "lmsys_500k": BANKED["lmsys_500k"] + 0.03}
    v = _g2_gate(realized, RECORDED, refits, BANKED, TOL)
    assert v.verdict == "FAIL"
    assert v.downgrade_recorded is False
    assert v.detail["mismatched"] == ["lmsys_150k"]
