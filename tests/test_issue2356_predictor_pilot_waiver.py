"""r9 plan-gap fix pins (#2356): the predictor PILOT's rule-26(d) api-refusal
waiver (``waive_api_refusal_arms``, llm-judging.md rules 26(d)/28, #2152).

The r9 pilot FAILed on armB api-refusal rate 0.133 (10/75 draws) >= the 0.10
gate — the pre-registered case: the production wave ``run_predictor`` performs
the plan-S3 / rule-28 targeted SYNC re-issue at the identical instrument (R8),
so the pilot-scale batch-path censor is REMEDIATED, not a broken instrument.
Pre-fix, ``run_predictor_pilot`` called ``judge_pilot_gate`` WITHOUT the
waiver and hard-failed on it.

Two test groups:

1. Wiring (real body): ``run_predictor_pilot`` executes end-to-end on a tmp
   splits fixture; ONLY the external judge boundary (``judge_pilot_gate``,
   which dispatches API calls) is replaced by a ``create_autospec`` fake
   (signature-conformant by construction) returning a REAL ``PilotGateReport``
   dataclass instance. Asserts the waiver constant is threaded, unwaivable
   pins are untouched, and absent arms are filtered out of the waiver (the
   gate raises ``ValueError`` on waivers naming unknown arms).
2. Gate semantics (no mocks): real ``ArmPilotStats`` fixtures driven through
   ``_gate_verdict`` + ``_api_refusal_failures`` composed EXACTLY as
   ``judge_pilot_gate`` composes them — the armB-0.133 fixture PASSes with a
   WAIVED warning, and the waiver does NOT leak: truncation and an
   above-threshold parse-fail rate still FAIL with the api-refusal waiver set.

Prompts are neutral placeholders (content hygiene); no network, no
worktree-absolute paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_judge as j  # noqa: E402

from explore_persona_space.eval.judge_pilot import (  # noqa: E402
    ArmPilotStats,
    PilotGateReport,
    _api_refusal_failures,
    _gate_verdict,
)

# The r9 pilot's realized shape: 75 draws/arm (15 items x 5 draws), armB
# 10/75 api-refusal draws = 0.1333 >= the 0.10 gate; armA 0.0.
N_DRAWS_ARM = 75
N_API_REFUSAL_ARMB = 10


def _mk_report(passed: bool) -> PilotGateReport:
    """Minimal REAL PilotGateReport (never a bare Mock return)."""
    return PilotGateReport(
        passed=passed,
        verdict="PASS" if passed else "FAIL",
        failures=[] if passed else ["armB: api-refusal rate 13.3% >= 10%"],
        warnings=[],
        arms={},
        judge_model=j.JUDGE_MODEL,
        max_tokens=j.PREDICTOR_MAX_TOKENS,
        n_total_draws=2 * N_DRAWS_ARM,
        parse_fail_threshold=0.02,
        rubric_hash="0" * 16,
    )


def _write_splits(tmp_path: Path, arms: tuple[str, ...]) -> Path:
    eval_root = tmp_path / "eval_results" / "issue_2356"
    splits = eval_root / "splits"
    splits.mkdir(parents=True)
    rows = [
        {"row_id": f"{arm}-row-{i}", "prompt": f"q-{arm}-{i}", "arm": arm, "fold": 0}
        for arm in arms
        for i in range(3)
    ]
    (splits / "balanced_eval_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    train = [
        {
            "row_id": f"train-{i}",
            "prompt": f"tr-q-{i}",
            "label": "engage" if i % 2 == 0 else "refuse",
            "group_id": f"g{i}",
        }
        for i in range(8)
    ]
    for arm in arms:
        (splits / f"train_rows_{arm}_fold0.json").write_text(json.dumps(train), encoding="utf-8")
    return eval_root


def _args(eval_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        arm=None, zero_shot=False, smoke=False, dry_run=False, out_root=str(eval_root / "judge")
    )


# ---------------------------------------------------------------------------
# 1. Wiring: the REAL run_predictor_pilot body threads the waiver constant.
# ---------------------------------------------------------------------------


def test_pilot_threads_api_refusal_waiver_and_keeps_unwaivable_pins(tmp_path, monkeypatch):
    eval_root = _write_splits(tmp_path, ("armA", "armB"))
    fake = create_autospec(j.judge_pilot_gate, return_value=_mk_report(True))
    monkeypatch.setattr(j, "judge_pilot_gate", fake)

    rc = j.run_predictor_pilot(_args(eval_root))
    assert rc == 0
    assert fake.call_count == 1
    kwargs = fake.call_args.kwargs

    # The fix under test: rule-26(d) waiver threaded from the caller-site
    # constant (both arms present -> the full constant).
    assert j.PREDICTOR_PILOT_WAIVE_API_REFUSAL_ARMS == ("armA", "armB")
    assert kwargs["waive_api_refusal_arms"] == j.PREDICTOR_PILOT_WAIVE_API_REFUSAL_ARMS

    # Unwaivable / untouched pins (brief scope guard): thresholds + wave
    # declaration byte-identical to r8.
    assert kwargs["api_refusal_threshold"] == 0.10
    assert kwargs["parse_fail_threshold"] == 0.02
    assert kwargs["n_draws"] == j.PREDICTOR_N_DRAWS
    assert kwargs["max_tokens"] == j.PREDICTOR_MAX_TOKENS
    assert kwargs["judge_model"] == j.JUDGE_MODEL
    assert kwargs["wave_n_calls"] == j.PREDICTOR_WAVE_N_CALLS
    assert kwargs["wave_threshold_base"] == 0
    assert kwargs["wave_force_sync"] is False
    assert "waive_parse_fail_arms" not in kwargs  # parse-fail waiver NOT granted

    # Both arms reached the gate (positional arms dict).
    arms_arg = fake.call_args.args[0]
    assert set(arms_arg) == {"armA", "armB"}


def test_pilot_rc_still_tracks_gate_verdict(tmp_path, monkeypatch):
    eval_root = _write_splits(tmp_path, ("armA", "armB"))
    fake = create_autospec(j.judge_pilot_gate, return_value=_mk_report(False))
    monkeypatch.setattr(j, "judge_pilot_gate", fake)
    assert j.run_predictor_pilot(_args(eval_root)) == 1


def test_waiver_filtered_to_present_arms(tmp_path, monkeypatch):
    """A single-arm eval-row set must not name an absent arm in the waiver —
    judge_pilot_gate raises ValueError on unknown waiver arms."""
    eval_root = _write_splits(tmp_path, ("armB",))
    fake = create_autospec(j.judge_pilot_gate, return_value=_mk_report(True))
    monkeypatch.setattr(j, "judge_pilot_gate", fake)

    assert j.run_predictor_pilot(_args(eval_root)) == 0
    assert fake.call_args.kwargs["waive_api_refusal_arms"] == ("armB",)


# ---------------------------------------------------------------------------
# 2. Gate semantics: real ArmPilotStats through the REAL verdict helpers,
#    composed exactly as judge_pilot_gate composes them (clauses a/b + d).
# ---------------------------------------------------------------------------


def _stats(
    *,
    n_draws: int = N_DRAWS_ARM,
    n_api_refusal: int = 0,
    api_waived: bool = False,
    n_content_dropped: int = 0,
    n_refusal: int = 0,
    n_truncation: int = 0,
    n_transport_lost: int = 0,
    stop_reason_tally: dict[str, int] | None = None,
    parse_waived: bool = False,
) -> ArmPilotStats:
    n_reached = n_draws - n_transport_lost
    n_answered = n_reached - n_api_refusal
    return ArmPilotStats(
        n_items=15,
        n_items_zero_valid=0,
        frac_items_complete=1.0,
        n_draws=n_draws,
        n_scored=n_answered - n_content_dropped,
        n_content_dropped=n_content_dropped,
        n_refusal=n_refusal,
        n_truncation=n_truncation,
        n_transport_lost=n_transport_lost,
        n_api_refusal=n_api_refusal,
        n_unknown_stop_reason_drops=0,
        parse_fail_rate=(n_content_dropped - n_refusal) / max(1, n_answered),
        stop_reason_tally=dict(stop_reason_tally or {"end_turn": n_answered}),
        waived=parse_waived,
        api_refusal_rate=n_api_refusal / max(1, n_reached),
        api_refusal_waived=api_waived,
    )


def _verdict(arm_stats: dict[str, ArmPilotStats]) -> tuple[list[str], list[str]]:
    """Failures/warnings composed as judge_pilot_gate does (minus transport
    parity, which needs a live save_raw record): _gate_verdict at the pilot's
    production knobs + the rule-26(d) clause."""
    failures, warnings = _gate_verdict(
        arm_stats,
        max_tokens=j.PREDICTOR_MAX_TOKENS,
        parse_fail_threshold=0.02,
        min_effective_draws_per_arm=10,
    )
    ar_failures, ar_warnings = _api_refusal_failures(arm_stats, api_refusal_threshold=0.10)
    return failures + ar_failures, warnings + ar_warnings


def test_armb_pilot_shape_passes_waived_with_warning():
    """The exact r9 FAIL shape (armB 10/75 = 0.133 api-refusal, armA 0.0,
    parse-fail 0, transport 0, no truncation) PASSes under the waiver, with
    the WAIVED warning present."""
    arm_stats = {
        "armA": _stats(api_waived=True),
        "armB": _stats(n_api_refusal=N_API_REFUSAL_ARMB, api_waived=True),
    }
    assert arm_stats["armB"].api_refusal_rate == pytest.approx(10 / 75)
    failures, warnings = _verdict(arm_stats)
    assert failures == []
    assert any("WAIVED" in w and "armB" in w and "api-refusal" in w for w in warnings)


def test_armb_pilot_shape_fails_without_waiver():
    """Pre-fix behavior pinned: the identical stats FAIL when unwaived."""
    arm_stats = {
        "armA": _stats(),
        "armB": _stats(n_api_refusal=N_API_REFUSAL_ARMB),
    }
    failures, _warnings = _verdict(arm_stats)
    assert any("api-refusal" in f and "armB" in f for f in failures)


def test_truncation_still_fails_with_api_refusal_waiver_set():
    """The waiver must NOT leak into rule 26(a): a max_tokens truncation
    signature FAILs even with BOTH the api-refusal AND parse-fail waivers."""
    arm_stats = {
        "armA": _stats(api_waived=True),
        "armB": _stats(
            n_api_refusal=N_API_REFUSAL_ARMB,
            api_waived=True,
            n_content_dropped=2,
            n_truncation=2,
            stop_reason_tally={"max_tokens": 2, "end_turn": 63},
            parse_waived=True,
        ),
    }
    failures, _warnings = _verdict(arm_stats)
    assert any("truncation" in f and "NEVER waivable" in f for f in failures)


def test_parse_fail_still_fails_with_api_refusal_waiver_set():
    """The api-refusal waiver must NOT leak into rule 26(b): an
    above-threshold parse-fail rate still FAILs (parse-fail has its own,
    separately-granted waiver, which run_predictor_pilot does NOT pass)."""
    arm_stats = {
        "armA": _stats(api_waived=True),
        "armB": _stats(
            n_api_refusal=N_API_REFUSAL_ARMB,
            api_waived=True,
            n_content_dropped=3,  # 3/65 = 4.6% >= 2%, none of them REFUSAL
        ),
    }
    assert arm_stats["armB"].parse_fail_rate >= 0.02
    failures, _warnings = _verdict(arm_stats)
    assert any("parse-fail" in f and "armB" in f for f in failures)
