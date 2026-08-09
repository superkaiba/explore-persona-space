"""Issue #2203 round-2 regression tests (code-review v1 fixes).

Offline / CPU / no-download. Pins the fixes the round-1 smoke could not
exercise because a ``_smoke_axis`` substitution bypassed the production
``_load_axis`` (code-review v1 C1/C2):

- **C1 (BLOCKER)** — ``phase2._load_axis`` reads the TWO footprint-matched
  τ_rand pools phase1 writes (``tau_rand_ctx_by_layer`` +
  ``tau_rand_alltoken_by_layer``) and FAILS LOUD on the round-1 single-key
  schema (``tau_rand_by_layer``) instead of silently collapsing both nulls
  into one via a ``.get`` alias. Round-trips a phase0-schema ``.pt`` +
  phase1-schema band JSON; the pre-fix schema raises.
- ``pareto_select`` frontier + knee tie-break.
- ``gsm8k_extract`` / ``wilson_ci`` (capability scoring primitives).
- cluster-id carries no ``% 44`` aliasing (r1 Minor 18).
- ``regime_fingerprint`` / ``check_regime`` cross-regime refusal (r1 M7).
- ``phase2._assert_alignment`` per-row meta trip (r1 M10).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue2203_capability as CAP  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_phase1 as P1  # noqa: E402
from scripts import issue2203_phase2 as P2  # noqa: E402


def _write_phase0_axis(tmp: Path, layers: list[int], hidden: int = 8) -> Path:
    """A phase0-schema axis .pt (str-keyed axis_by_layer/h_def_by_layer + layers)."""
    torch.manual_seed(0)
    blob = {
        "axis_by_layer": {str(li): torch.randn(hidden) for li in layers},
        "h_def_by_layer": {str(li): torch.randn(hidden) for li in layers},
        "layers": layers,
    }
    p = tmp / "phase0_axis_smoke.pt"
    torch.save(blob, p)
    return p


def _write_phase1_band(tmp: Path, band: list[int], layers: list[int], *, legacy: bool) -> Path:
    """A phase1-schema band JSON. ``legacy=True`` emits the round-1 single-key schema."""
    tau = {str(li): -1.0 * (li + 1) for li in layers}
    result = {"band_layers": band, "tau_by_layer": tau}
    if legacy:
        # The round-1 writer schema the .get-alias collapse read from.
        result["tau_rand_by_layer"] = {str(li): -0.5 for li in layers}
    else:
        # The fixed schema: two DISTINCT footprint-matched null pools.
        result["tau_rand_ctx_by_layer"] = {str(li): -0.3 for li in layers}
        result["tau_rand_alltoken_by_layer"] = {str(li): -0.7 for li in layers}
    p = tmp / "phase1_band_tau_smoke.json"
    p.write_text(json.dumps(result))
    return p


def test_load_axis_reads_both_tau_rand_pools(tmp_path):
    """C1: the two footprint-matched null pools survive the round trip, distinct."""
    band, layers = [1, 2], [1, 2, C.L14]
    axis_p = _write_phase0_axis(tmp_path, layers)
    band_p = _write_phase1_band(tmp_path, band, layers, legacy=False)
    geom = P2._load_axis(axis_p, band_p)
    assert geom["layers"] == band
    # Both nulls present, keyed by band layer, and NOT collapsed to one pool.
    assert set(geom["tau_rand_ctx_by_layer"]) == set(band)
    assert set(geom["tau_rand_alltoken_by_layer"]) == set(band)
    assert geom["tau_rand_ctx_by_layer"][1] != geom["tau_rand_alltoken_by_layer"][1]
    # L14 (single-layer arm) present with its REAL tau (no tau_rand needed).
    assert C.L14 in geom["axis_by_layer"] and C.L14 in geom["tau_by_layer"]


def test_load_axis_fails_loud_on_round1_legacy_schema(tmp_path):
    """C1 pre-fix demonstration: the round-1 single-key band JSON raises, not collapses."""
    band, layers = [1, 2], [1, 2, C.L14]
    axis_p = _write_phase0_axis(tmp_path, layers)
    band_p = _write_phase1_band(tmp_path, band, layers, legacy=True)
    with pytest.raises(KeyError, match=r"tau_rand_ctx_by_layer|tau_rand_alltoken_by_layer"):
        P2._load_axis(axis_p, band_p)


def test_null_arm_routes_to_footprint_matched_pool():
    """The ctx-null uses the ctx pool, the alltoken-null the alltoken pool (plan §5)."""
    # The routing expression lives in run_generation; assert the ARM_SPECS kinds
    # that drive it are exactly the two footprint-matched null arms.
    assert C.ARM_SPECS["cap_ctx_randnull"]["kind"] == "null_ctx"
    assert C.ARM_SPECS["cap_alltoken_randnull"]["kind"] == "null_alltoken"


def test_pareto_select_frontier_and_knee():
    metrics = {
        # id: harm_reduction, capability_drop, width, center
        "a": {"harm_reduction": 0.5, "capability_drop": 0.10, "width": 4, "center": 12},
        "b": {"harm_reduction": 0.4, "capability_drop": 0.05, "width": 2, "center": 12},
        "c": {
            "harm_reduction": 0.3,
            "capability_drop": 0.20,
            "width": 8,
            "center": 14,
        },  # dominated
    }
    selected, frontier = P1.pareto_select(metrics)
    assert "c" not in frontier  # dominated by a (more harm_red, less drop)
    assert set(frontier) == {"a", "b"}
    # knee = argmax(harm_reduction - capability_drop): a=0.40, b=0.35 -> a
    assert selected == "a"


def test_pareto_select_tie_break_smaller_width():
    metrics = {
        "wide": {"harm_reduction": 0.5, "capability_drop": 0.1, "width": 8, "center": 12},
        "narrow": {"harm_reduction": 0.5, "capability_drop": 0.1, "width": 2, "center": 12},
    }
    selected, _ = P1.pareto_select(metrics)
    assert selected == "narrow"  # equal knee -> smaller width wins


def test_gsm8k_extract():
    assert CAP.gsm8k_extract("...\n#### 42") == "42"
    assert CAP.gsm8k_extract("#### 1,024") == "1024"
    assert CAP.gsm8k_extract("#### $18.50") == "18.50"
    assert CAP.gsm8k_extract("the answer is 7") == "7"  # last-number fallback
    assert CAP.gsm8k_extract("no digits here") is None


def test_wilson_ci_bounds():
    assert CAP.wilson_ci(0, 0) is None
    lo, hi = CAP.wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    lo0, hi0 = CAP.wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.5


def test_cluster_id_no_mod44_aliasing():
    """r1 Minor 18: cluster_id = (bank, item index, role), never `hi % 44`."""
    rows = C.build_jailbreak_set(6, smoke=True)
    for r in rows:
        cid = r["meta"]["cluster_id"]
        assert cid == f"{r['meta']['harm_bank']}:{r['meta']['harm_index']}:{r['meta']['role']}"
        assert "%" not in cid


def test_regime_mismatch_refuses(tmp_path):
    """r1 M7: a resume artifact from a DIFFERENT regime raises naming the diff."""
    cur = C.regime_fingerprint(model="m", n_jailbreak=500, smoke=False)
    same = C.regime_fingerprint(model="m", n_jailbreak=500, smoke=False)
    C.check_regime(same, cur, tmp_path / "x.json")  # identical -> no raise
    diff = C.regime_fingerprint(model="m", n_jailbreak=250, smoke=False)
    with pytest.raises(ValueError, match=r"REGIME MISMATCH|n_jailbreak"):
        C.check_regime(diff, cur, tmp_path / "x.json")
    with pytest.raises(ValueError, match="NO regime fingerprint"):
        C.check_regime(None, cur, tmp_path / "x.json")


def test_alignment_assert_trips_on_meta_mismatch():
    """r1 M10: a persisted-vs-rebuilt jb meta mismatch raises (wrong judged question)."""
    jb = C.build_jailbreak_set(3, smoke=True)
    good_meta = [r["meta"] for r in jb]
    P2._assert_alignment("baseline", "jailbreak", good_meta, jb)  # aligned -> no raise
    bad = [dict(m) for m in good_meta]
    bad[1]["harm_index"] = 999999
    with pytest.raises(ValueError, match="meta mismatch on 'harm_index'"):
        P2._assert_alignment("baseline", "jailbreak", bad, jb)
    with pytest.raises(ValueError, match="!="):  # length mismatch
        P2._assert_alignment("baseline", "jailbreak", good_meta[:2], jb)
