"""#1491 first-chunk self-gate (plan §7 Gate 1) — three-valued verdict pins.

Three-round history against the REAL production functions:

ROUND 1 (commit 1c8b46d28a, line 741): ``(r2_fit - r2_null) > 0.05 and
abs(r2_null) < 0.05``. Unsatisfiable against the gate's OWN shuffle-FIT null
(expected R² ~ -1, not ~ 0) — aborted all 8 train_25k shards (epm:failure v3).

ROUND 2 (commit ccc650f42e): one-sided cap + BINDING ``-3.0`` floor. LIVE
production refuted it in BOTH directions: (a) VACUOUS PASS — the 0.5B
end-of-shard resume trigger (n_train=900 ~ h=896) read r2_fit ~ -0.02..-0.04
with the pass carried entirely by the bad null (a broken capture would look
identical there); (b) FALSE FAIL — the 1.5B rung's LEGITIMATE null reads
-3.40..-3.80 at n_train=1600, h=1536 (all 8 shards aborted on the floor).
The two points also establish the null scales with h ITSELF, not n/h (same
n/h ~ 1.0: -1.3 at h=896 vs -3.6 at h=1536), so ratio-scaled floors are wrong
too.

ROUND 3+ (this design): three-valued verdict. PASS = gap > 0.05 (binding,
byte-unchanged) AND r2_fit > 0 AND adequate conditioning
(n_train > 1.5 x h). INCONCLUSIVE = inadequate conditioning, or gap carried
by a non-positive fit — proceeds, loudly disclosed (#1701), NEVER aborts.
FAIL = at adequate conditioning only: gap <= 0.05 (the real broken-capture
signature) or a predictive null (>= 0.05) — still aborts. The old floor is a
log-only pathology tripwire at -100 (not a calibration knob).

Offline by construction: pure-numpy/torch synthetic rows, no Hub fetch, no GPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_generate_capture as D  # noqa: E402


def _pre_fix_predicate(r2_fit: float, r2_null: float) -> bool:
    """Round-1 predicate, verbatim (commit 1c8b46d28a, line ~741)."""
    return (r2_fit - r2_null) > 0.05 and abs(r2_null) < 0.05


def _round2_predicate(r2_fit: float, r2_null: float) -> bool:
    """Round-2 predicate, verbatim (commit ccc650f42e) — the binding -3.0
    floor that production refuted in both directions (vacuous PASS at 0.5B
    resume; false FAIL on all 8 1.5B shards)."""
    return (r2_fit - r2_null) > 0.05 and (-3.0 < r2_null < 0.05)


# PRODUCTION CALIBRATION TABLE — every row is a REAL diag observed live
# (0.5B mid-shard: epm:failure v3 diagnosis; 0.5B end-of-shard resume +
# 1.5B mid-shard: the 2026-08-05 relaunch at ccc650f42e). These rows GROUND
# the thresholds; do not retune without new production points.
#   (r2_fit, r2_null, n_train, h, expected_verdict)
CALIBRATION = [
    # 0.5B mid-shard trigger — n/h = 1.79, the only genuinely positive fit
    # reads we have: real reads, PASS.
    (0.094, -0.990, 1600, 896, "PASS"),
    (0.108, -0.916, 1600, 896, "PASS"),
    (0.158, -0.833, 1600, 896, "PASS"),
    # 0.5B end-of-shard resume fallback — n/h = 1.004: fit ~ 0, a pass would
    # be carried entirely by the bad null (vacuous under round 2):
    # INCONCLUSIVE.
    (-0.019, -1.283, 900, 896, "INCONCLUSIVE"),
    (-0.044, -1.349, 900, 896, "INCONCLUSIVE"),
    # 1.5B mid-shard — n/h = 1.04, h = 1536: LEGITIMATE nulls at -3.4..-3.8
    # (false-FAILed by the round-2 floor, all 8 shards): INCONCLUSIVE.
    (-0.550, -3.403, 1600, 1536, "INCONCLUSIVE"),
    (-0.673, -3.756, 1600, 1536, "INCONCLUSIVE"),
    (-0.710, -3.804, 1600, 1536, "INCONCLUSIVE"),
    (-0.733, -3.744, 1600, 1536, "INCONCLUSIVE"),
]


def test_production_calibration_table():
    """Every live production diag classifies as the design requires."""
    for r2_fit, r2_null, n_train, h, expected in CALIBRATION:
        verdict, reason = D._self_gate_verdict(r2_fit, r2_null, n_train, h)
        assert verdict == expected, (r2_fit, r2_null, n_train, h, verdict, reason)


def test_vacuous_pass_rows_were_pass_under_round2():
    """THE round discriminator (direction a): the observed 0.5B resume diags
    returned a VACUOUS PASS under the current ccc650f42e predicate — the new
    verdict demotes exactly them to INCONCLUSIVE."""
    for r2_fit, r2_null in [(-0.019, -1.283), (-0.044, -1.349)]:
        assert _round2_predicate(r2_fit, r2_null) is True  # vacuous PASS then
        verdict, _ = D._self_gate_verdict(r2_fit, r2_null, 900, 896)
        assert verdict == "INCONCLUSIVE"  # the honest outcome now


def test_15b_false_fail_rows_were_fail_under_round2():
    """THE round discriminator (direction b): the observed 1.5B diags were
    deterministically ABORTED by the round-2 floor — the new verdict
    classifies them INCONCLUSIVE (not FAIL: no abort; not PASS: fit <= 0)."""
    for r2_fit, r2_null, n_train, h, expected in CALIBRATION[5:]:
        assert _round2_predicate(r2_fit, r2_null) is False  # the false abort
        verdict, _ = D._self_gate_verdict(r2_fit, r2_null, n_train, h)
        assert verdict == "INCONCLUSIVE" == expected


def test_observed_diagnostics_fail_pre_fix_predicate():
    """Round-1 oracle retained: the original two-sided bound rejects even the
    genuinely-positive mid-shard reads."""
    for r2_fit, r2_null, _n, _h, expected in CALIBRATION[:3]:
        assert expected == "PASS"
        assert _pre_fix_predicate(r2_fit, r2_null) is False


def test_broken_capture_fails_and_aborts_at_adequate_conditioning():
    """gap <= 0.05 at adequate conditioning = the real broken-capture
    signature: FAIL (abort)."""
    for r2_fit, r2_null in [(0.01, 0.0), (-0.90, -0.92), (0.0, -0.05)]:
        verdict, reason = D._self_gate_verdict(r2_fit, r2_null, 1600, 896)
        assert verdict == "FAIL", (r2_fit, r2_null, reason)
        assert "gap" in reason


def test_gap_failure_at_bad_conditioning_is_inconclusive():
    """At inadequate conditioning the gate cannot discriminate in EITHER
    direction — even a zero gap must not abort (unreliable evidence)."""
    verdict, reason = D._self_gate_verdict(-1.30, -1.30, 900, 896)
    assert verdict == "INCONCLUSIVE"
    assert "conditioning" in reason


def test_predictive_null_fails_only_at_adequate_conditioning():
    """The regime-independent leakage signature still aborts — but only when
    the conditioning makes the read meaningful."""
    assert D._self_gate_verdict(0.90, 0.50, 1600, 896)[0] == "FAIL"
    assert D._self_gate_verdict(0.90, 0.50, 900, 896)[0] == "INCONCLUSIVE"


def test_gap_strictness_unchanged():
    """The binding gap arm is byte-unchanged: strict > 0.05 (checked at
    adequate conditioning with a positive fit so no other clause fires)."""
    assert D._self_gate_verdict(0.06, 0.0, 1600, 896)[0] == "PASS"
    verdict, reason = D._self_gate_verdict(0.05, 0.0, 1600, 896)
    assert verdict == "FAIL" and "gap" in reason  # gap == 0.05 exactly


def test_conditioning_boundary_is_strict():
    """n_train must EXCEED 1.5 x h: 1344 = 1.5 x 896 exactly is inadequate."""
    assert D.SELF_GATE_ADEQUATE_COND == 1.5
    assert D._self_gate_verdict(0.10, -0.9, 1344, 896)[0] == "INCONCLUSIVE"
    assert D._self_gate_verdict(0.10, -0.9, 1345, 896)[0] == "PASS"


def test_fit_positive_boundary_and_no_science_floor():
    """PASS may not rest on a non-positive fit (strict > 0) — but a fit <= 0
    NEVER fails on its own (the ladder's low-R² rungs stay legal: the
    verdict is INCONCLUSIVE and the rung proceeds)."""
    v0, reason = D._self_gate_verdict(0.0, -1.0, 1600, 896)
    assert v0 == "INCONCLUSIVE" and "non-positive fit" in reason
    assert D._self_gate_verdict(0.001, -1.0, 1600, 896)[0] == "PASS"
    # Deeply negative fit with a huge gap at adequate conditioning: still
    # only INCONCLUSIVE — proceeds, disclosed; never an abort.
    assert D._self_gate_verdict(-0.5, -2.0, 1600, 896)[0] == "INCONCLUSIVE"


def test_under_500_rows_auto_pass():
    """The tiny-split auto-pass (val_400 / test_1000 / tierB_3600 per-shard
    slices) is untouched."""
    passed, diag = D._first_chunk_self_gate([{} for _ in range(499)], 0)
    assert passed is True
    assert diag.get("skipped") is True


def _rows_from(X: np.ndarray, Y: np.ndarray) -> list[dict]:
    """Rows shaped as the capture loop builds them: per-row lists of
    per-layer tensors, indexed by layer_index_primary."""
    return [
        {"cx_last": [torch.from_numpy(X[i])], "v_x": [torch.from_numpy(Y[i])]}
        for i in range(len(X))
    ]


def _synthetic(n: int, h: int, seed: int, x_scale: float, noise: float) -> list[dict]:
    """Strong-linear-map synthetic rows at a chosen conditioning n/h."""
    rng = np.random.default_rng(seed)
    X = (x_scale * rng.standard_normal((n, h))).astype(np.float32)
    W = (rng.standard_normal((h, h)) / np.sqrt(h)).astype(np.float32)
    Y = (X @ W + noise * rng.standard_normal((n, h))).astype(np.float32)
    return _rows_from(X, Y)


def _assert_disclosure(diag: dict, n_train: int, h: int) -> None:
    """The #1701 conditioning-disclosure contract: fields present, correct,
    self-consistent, and the verdict/reason recorded."""
    assert diag["verdict"] in {"PASS", "INCONCLUSIVE", "FAIL"}
    assert isinstance(diag["verdict_reason"], str) and diag["verdict_reason"]
    assert diag["n_train"] == n_train
    assert diag["h"] == h
    assert diag["n_train_over_h"] == round(n_train / h, 3)
    assert diag["under_determined"] is (n_train < h)
    assert diag["null_floor_advisory"] == D.SELF_GATE_NULL_FLOOR
    assert diag["null_floor_breached"] is (diag["r2_null"] <= D.SELF_GATE_NULL_FLOOR)
    assert diag["passed"] is (diag["verdict"] != "FAIL")


def test_full_gate_healthy_wellconditioned_passes(caplog):
    """e2e, well-over-determined (n/h = 60): verdict PASS, disclosure
    correct, no tripwire advisory."""
    rows = _synthetic(n=600, h=8, seed=0, x_scale=1.0, noise=0.1)
    with caplog.at_level(logging.WARNING, logger="issue1491_ladder_generate_capture"):
        passed, diag = D._first_chunk_self_gate(rows, 0)
    assert passed is True
    assert diag["verdict"] == "PASS", diag
    _assert_disclosure(diag, n_train=480, h=8)
    assert "ADVISORY" not in caplog.text


def test_full_gate_broken_capture_fails(caplog):
    """e2e FAIL leg: Y statistically independent of X (the broken-capture
    surrogate) at adequate conditioning — fit ~ null ~ 0, gap ~ 0
    (measured gap = 0.006): FAIL, passed=False, so the caller aborts."""
    rng = np.random.default_rng(3)
    n, h = 600, 8
    X = rng.standard_normal((n, h)).astype(np.float32)
    Y = rng.standard_normal((n, h)).astype(np.float32)  # no relation to X
    passed, diag = D._first_chunk_self_gate(_rows_from(X, Y), 0)
    assert diag["verdict"] == "FAIL", diag
    assert passed is False
    assert diag["gap"] <= 0.05, diag
    _assert_disclosure(diag, n_train=480, h=8)


def test_full_gate_scale05_like_conditioning_passes():
    """e2e at the 0.5B mid-shard conditioning, scaled down: n_train/h =
    480/256 ~ 1.9 (production: 1600/896 ~ 1.79, adequate). Real signal,
    null ~ -1 (the observed regime): PASS."""
    rows = _synthetic(n=600, h=256, seed=1, x_scale=5.0, noise=0.5)
    passed, diag = D._first_chunk_self_gate(rows, 0)
    assert diag["r2_null"] < -0.3, diag  # shuffle-fit, not mean-predictor
    assert diag["verdict"] == "PASS", diag
    assert passed is True
    _assert_disclosure(diag, n_train=480, h=256)


def test_full_gate_near_threshold_is_inconclusive_never_aborts(caplog):
    """e2e near the interpolation threshold (n_train/h = 480/500 = 0.96 —
    the regime of BOTH live incidents): HEALTHY data, huge gap, legitimate
    null ~ -22. Verdict INCONCLUSIVE, passed=True (no abort), disclosure
    correct, no tripwire advisory (legitimate depth, not pathology) — and
    both prior predicates get it wrong (round 1 rejects; round 2 aborts)."""
    rows = _synthetic(n=600, h=500, seed=7, x_scale=5.0, noise=0.5)
    with caplog.at_level(logging.WARNING, logger="issue1491_ladder_generate_capture"):
        passed, diag = D._first_chunk_self_gate(rows, 0)
    assert diag["verdict"] == "INCONCLUSIVE", diag
    assert passed is True  # MUST NOT abort
    assert diag["gap"] > 0.05, diag
    assert diag["r2_null"] < -3.0, diag  # deep LEGITIMATE null (measured ~ -22)
    assert diag["null_floor_breached"] is False  # tripwire is far below
    _assert_disclosure(diag, n_train=480, h=500)
    assert "ADVISORY" not in caplog.text
    assert _round2_predicate(diag["r2_fit"], diag["r2_null"]) is False
    assert _pre_fix_predicate(diag["r2_fit"], diag["r2_null"]) is False


def test_pathology_tripwire_fires_log_only(monkeypatch, caplog):
    """The -100 tripwire is log-only: force a breach (raise the tripwire
    above the fixture's ~ -22 null) and assert the ADVISORY warning fires
    while the verdict and no-abort semantics are untouched."""
    monkeypatch.setattr(D, "SELF_GATE_NULL_FLOOR", -1.0)
    rows = _synthetic(n=600, h=500, seed=7, x_scale=5.0, noise=0.5)
    with caplog.at_level(logging.WARNING, logger="issue1491_ladder_generate_capture"):
        passed, diag = D._first_chunk_self_gate(rows, 0)
    assert diag["null_floor_breached"] is True
    assert "ADVISORY" in caplog.text
    assert "NOT a verdict input" in caplog.text
    assert diag["verdict"] == "INCONCLUSIVE"  # verdict unaffected by tripwire
    assert passed is True  # still no abort
