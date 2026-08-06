"""CPU unit tests for the #2091 G2 pilot-gate parse-fail waiver wiring.

``run_pilot`` threads the auditable module constant
``PILOT_WAIVE_PARSE_FAIL_ARMS`` (wave -> waived arms; recorded analysis:
epm:progress v35 on #2091) into ``judge_pilot_gate``'s existing
``waive_parse_fail_arms`` affordance. These tests drive the REAL ``run_pilot``
body and the REAL ``judge_pilot_gate`` body (production-body rule) — the ONLY
fake is the external Anthropic-API boundary, ``judge_pilot.judge_graded``,
replaced by a ``create_autospec`` (signature-conformant by construction) whose
side effect returns REAL ``JudgeResult`` dataclass instances reproducing the
observed pilot tallies, keyed by the (wave, arm) the gate encodes in
``cache_dir``.

Covered properties (the four the waiver change must preserve):
1. the two waived (wave, arm) cells PASS at their observed parse-fail rates;
2. an UNWAIVED arm at the same rate still FAILs (waiver is (wave, arm)-scoped
   — the same ``wildchat`` arm under ``evil_trait`` gates normally);
3. a waived arm with ANY truncation evidence still FAILs (truncation is never
   waivable);
4. an unknown arm name in the constant raises ``ValueError`` (fail-loud) —
   plus a pin that the REAL constant names only valid, wave-feeding rung-job
   arms and exactly the two recorded cells (the do-NOT-widen guard).

No GPU, no network, ZERO API calls: the autospec'd boundary fake never
dispatches; item text is benign synthetic filler.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import scripts.issue2091_judge as judge_mod
from explore_persona_space.eval import judge_pilot
from explore_persona_space.eval.graded_judge import JudgeResult
from scripts.issue2091_stage_contexts import RUNG_JOBS_BY_NAME

# (wave, arm) -> observed pilot tallies (epm:progress v35 on #2091).
OBSERVED_WAIVED_TALLIES = {
    ("sycophancy_trait", "wildchat"): {"n_draws": 16, "n_dropped": 1},  # 6.25%
    ("hallucination_trait", "wildchat"): {"n_draws": 50, "n_dropped": 14},  # 28.0%
}
CLEAN_TALLY = {"n_draws": 16, "n_dropped": 0}


def _fake_judge_graded(tallies: dict[tuple[str, str], dict]):
    """Signature-conformant boundary fake returning REAL JudgeResult instances.

    ``judge_pilot_gate`` calls ``judge_graded(sub, prompt, ..., cache_dir=
    <pilot root>/<wave>/<arm>, save_raw=...)`` once per arm — the fake keys its
    tallies on ``(cache_dir.parent.name, cache_dir.name)`` and writes the
    ``save_raw`` JSON the gate's unknown-stop-reason scan reads.
    """

    def _impl(items, eval_prompt, **kwargs):
        cache_dir = Path(kwargs["cache_dir"])
        t = tallies[(cache_dir.parent.name, cache_dir.name)]
        save_raw = Path(kwargs["save_raw"])
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": {}}), encoding="utf-8")
        n_draws = t["n_draws"]
        tally = dict(t.get("stop_reason_tally", {"end_turn": n_draws}))
        return JudgeResult(
            scores={item_id: 50.0 for item_id, _q, _a in items},
            n_total_draws=n_draws,
            n_dropped_draws=t.get("n_dropped", 0),
            n_refusal_draws=t.get("n_refusal", 0),
            n_truncation_dropped_draws=t.get("n_truncation", 0),
            n_transport_lost_draws=t.get("n_transport_lost", 0),
            stop_reason_tally=tally,
        )

    return create_autospec(judge_pilot.judge_graded, side_effect=_impl)


def _empty_waves() -> dict[str, judge_mod.WaveItems]:
    return {
        w.name: judge_mod.WaveItems(items=[], arm_by_item={}, meta_by_context={})
        for w in judge_mod.WAVES
    }


def _fill_wave(waves: dict, wave_name: str, arms: list[str], n_items: int = 2) -> None:
    wv = waves[wave_name]
    for arm in arms:
        for i in range(n_items):
            item_id = f"{arm}-c{i:02d}_k00"
            wv.items.append((item_id, "What is 2+2?", "It is 4."))
            wv.arm_by_item[item_id] = arm


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        out_root=tmp_path / "out",
        cache_root=tmp_path / "cache",
        raw_root=tmp_path / "raw",
        pilot_draws_per_wave=50,
    )


def _rubrics() -> dict[str, str]:
    return {w.name: "Score {question} / {answer} 0-100." for w in judge_mod.WAVES}


def test_waived_arms_pass_at_observed_rates(tmp_path, monkeypatch, caplog):
    """Property 1: both recorded (wave, arm) cells clear the gate at the exact
    rates the pilot observed, the report marks them waived, and the waiver is
    logged at gate time (wave + arm + realized rate)."""
    waves = _empty_waves()
    _fill_wave(waves, "sycophancy_trait", ["syc_train", "syc_aita", "wildchat"])
    _fill_wave(waves, "hallucination_trait", ["wildchat"])
    tallies = {
        ("sycophancy_trait", "syc_train"): CLEAN_TALLY,
        ("sycophancy_trait", "syc_aita"): CLEAN_TALLY,
        **OBSERVED_WAIVED_TALLIES,
    }
    monkeypatch.setattr(judge_pilot, "judge_graded", _fake_judge_graded(tallies))

    with caplog.at_level(logging.WARNING, logger="issue2091_judge"):
        verdict = judge_mod.run_pilot(_args(tmp_path), waves, _rubrics())

    assert verdict["passed"] is True
    syc = verdict["waves"]["sycophancy_trait"]
    hal = verdict["waves"]["hallucination_trait"]
    assert syc["verdict"] == "PASS" and hal["verdict"] == "PASS"
    assert syc["arms"]["wildchat"]["waived"] is True
    assert hal["arms"]["wildchat"]["waived"] is True
    assert syc["arms"]["syc_train"]["waived"] is False
    # the overshoot is recorded as a WAIVED warning, never silently absorbed.
    assert any("WAIVED" in w for w in syc["warnings"])
    assert any("WAIVED" in w for w in hal["warnings"])
    # gate-time waiver log line: wave + arm + realized parse-fail rate.
    assert "wave=sycophancy_trait arm=wildchat parse-fail 6.25% WAIVED" in caplog.text
    assert "wave=hallucination_trait arm=wildchat parse-fail 28.00% WAIVED" in caplog.text


def test_unwaived_arm_at_same_rate_still_fails(tmp_path, monkeypatch):
    """Property 2: the waiver is (wave, arm)-scoped — the SAME ``wildchat`` arm
    under ``evil_trait`` (not in the constant) FAILs at the hallucination
    wave's observed 28% rate, and run_pilot exits the designed rc=7."""
    waves = _empty_waves()
    _fill_wave(waves, "evil_trait", ["wildchat"])
    tallies = {("evil_trait", "wildchat"): {"n_draws": 50, "n_dropped": 14}}
    monkeypatch.setattr(judge_pilot, "judge_graded", _fake_judge_graded(tallies))

    with pytest.raises(SystemExit) as exc:
        judge_mod.run_pilot(_args(tmp_path), waves, _rubrics())
    assert exc.value.code == 7
    report = json.loads((tmp_path / "out" / "pilot" / "evil_trait_gate.json").read_text())
    assert report["verdict"] == "FAIL"
    assert any("parse-fail" in f and "wildchat" in f for f in report["failures"])


def test_waived_arm_truncation_still_fails(tmp_path, monkeypatch):
    """Property 3: the waiver covers PARSE-FAIL only — a waived arm showing any
    truncation evidence (n_truncation > 0 / a max_tokens tally key) still
    FAILs the wave."""
    waves = _empty_waves()
    _fill_wave(waves, "hallucination_trait", ["wildchat"])
    tallies = {
        ("hallucination_trait", "wildchat"): {
            "n_draws": 50,
            "n_dropped": 14,
            "n_truncation": 2,
            "stop_reason_tally": {"end_turn": 48, "max_tokens": 2},
        }
    }
    monkeypatch.setattr(judge_pilot, "judge_graded", _fake_judge_graded(tallies))

    with pytest.raises(SystemExit) as exc:
        judge_mod.run_pilot(_args(tmp_path), waves, _rubrics())
    assert exc.value.code == 7
    report = json.loads((tmp_path / "out" / "pilot" / "hallucination_trait_gate.json").read_text())
    assert any("truncation" in f and "NEVER waivable" in f for f in report["failures"])


def test_unknown_waiver_arm_raises(tmp_path, monkeypatch):
    """Property 4: a typo'd/renamed arm in the constant raises ValueError at
    gate time (never a silent no-op waiver), propagating out of run_pilot."""
    waves = _empty_waves()
    _fill_wave(waves, "sycophancy_trait", ["syc_train"])
    tallies = {("sycophancy_trait", "syc_train"): CLEAN_TALLY}
    monkeypatch.setattr(judge_pilot, "judge_graded", _fake_judge_graded(tallies))
    monkeypatch.setattr(
        judge_mod, "PILOT_WAIVE_PARSE_FAIL_ARMS", {"sycophancy_trait": ("typo_arm",)}
    )

    with pytest.raises(ValueError, match=r"unknown arm.*typo_arm"):
        judge_mod.run_pilot(_args(tmp_path), waves, _rubrics())


def test_waiver_constant_names_exactly_the_two_recorded_cells():
    """Do-NOT-widen pin: the constant is exactly the two epm:progress-v35 cells,
    every wave key is a real wave, and every waived arm is a real rung job that
    actually FEEDS that trait wave (wave behavior in the job's judge set, and
    not the own-rung hallucination rows, which route to the abstain wave)."""
    assert judge_mod.PILOT_WAIVE_PARSE_FAIL_ARMS == {
        "hallucination_trait": ("wildchat",),
        "sycophancy_trait": ("wildchat",),
    }
    for wave_name, arms in judge_mod.PILOT_WAIVE_PARSE_FAIL_ARMS.items():
        wave = judge_mod.WAVES_BY_NAME[wave_name]
        for arm in arms:
            job = RUNG_JOBS_BY_NAME[arm]  # KeyError = stale arm name
            assert wave.behavior in job.judge_behaviors, (wave_name, arm)
            # own-rung hallucination rows feed the abstain wave, not the trait wave.
            assert not (wave.behavior == "hallucination" and job.gen_behavior == "hallucination"), (
                wave_name,
                arm,
            )
