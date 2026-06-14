"""Issue #641 round-2 regression tests for the two code-review blockers that are
unit-testable without a GPU:

- Blocker 5: ``classify_h5`` sign direction. With the sign convention
  ``(resistant - non-resistant)``, the H5b "resistance = ceiling" outcome
  requires the resistant class to plateau BELOW non-resistant (a NEGATIVE
  asymptote gap). A POSITIVE gap (resistant above non-resistant) is the OPPOSITE
  of the ceiling prediction and must never be labeled H5b (the round-1 bug used
  ``min(abs(lo), abs(hi))`` + ``(lo > 0) or (hi < 0)``, which returned H5b for
  positive gaps too).

- Blocker 4: the #641 harmful-advice judge is a dedicated YES/NO bad-medical
  COMPLIANCE judge, distinct from the broad EM Betley aligned/coherent rubric
  (the Arm-B H1 covariate is harmful-advice compliance, plan §4.6).

Round-3 plan-adherence blockers (reconciler-persisted):

- Blocker 1: ``resolve_matched_dose`` resolves the Arm-B matched dose from the
  Arm-A median install crossing (plan §4.5), with the pre-registered step-375
  fallback when Arm-A data are absent.
- Blocker 2: ``select_matched_neutral`` widens to the §4.5 widened pool when no
  narrow candidate is within ±0.10 — reachable ONLY because the widened
  candidates are measured by P0 (their propensities are in ``candidate_prop``).
- Blocker 3: ``bootstrap_dose_curve`` draws ``boot_seeds`` ONCE per replicate and
  threads it across every dose step (seeds = the replication unit, §6.3); a
  pure-seed replicate traces that seed's own curve at all doses.

Round-4 blocker (reconciler-persisted, ``armb-teacher-source-alias-mismatch``):

- The Arm-B teacher source has two slugs (``kindergarten_teacher`` — the PERSONAS
  key / plan-body slug — and ``sp_teacher_ho`` — the #537 registry cid) that
  resolve to the SAME system prompt. ``canonicalize_source`` normalizes both to
  the records key ``ARM_B_TEACHER_CID`` (= ``sp_teacher_ho``) at every
  records-keying boundary, so ``phase_aggregate``'s ``ARM_B_TEACHER_CID in
  records`` gate fires (``armB_matched_dose_delta is not None``) regardless of
  which slug the operator typed in ``--sources`` — closing the False-negative
  that silently dropped the H1-vs-H2 headline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.i537_judging import (
    harmful_advice_rate_from_verdicts,
    judge_request_for_row,
    parse_verdict_binary,
)
from explore_persona_space.experiments.issue_641.data import (
    ARM_B_TEACHER_CID,
    canonicalize_source,
)
from explore_persona_space.experiments.issue_641.stats import (
    _trace_boot_seeds,
    bootstrap_dose_curve,
    classify_h5,
    resolve_matched_dose,
)

# The dose-curve dispatcher is a script (not an installed module); load it by path
# so the round-4 alias tests can drive the REAL phase_aggregate against fixtures.
_DISPATCHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "issue641_dose_curves.py"

# ── Blocker 5: classify_h5 sign direction ────────────────────────────────────


def test_classify_h5_positive_gap_is_not_ceiling():
    """A positive asymptote gap (resistant ABOVE non-resistant) is the opposite
    of the H5b ceiling direction -> never H5b."""
    verdict = classify_h5((0.16, 0.25), resistant_top_slope=0.0)
    assert verdict != "H5b", verdict
    assert verdict in ("H5_REVERSED", "AMBIGUOUS"), verdict
    # Specifically the reversed label, since the CI is entirely positive.
    assert verdict == "H5_REVERSED", verdict


def test_classify_h5_negative_gap_flat_slope_is_ceiling():
    """A negative gap (resistant BELOW non-resistant) with a flat resistant top
    slope and gap magnitude >= 0.15 is the H5b ceiling outcome."""
    assert classify_h5((-0.25, -0.16), resistant_top_slope=0.0) == "H5b"


def test_classify_h5_negative_gap_but_steep_slope_not_ceiling():
    """Even a negative gap is NOT a ceiling if the resistant curve is still
    climbing (top slope not ~0) — that is the H5a "slower climb" reading."""
    # slope is per-100-steps; the threshold is abs(slope) < 2.0. A steep slope
    # (resistant still rising) blocks the H5b ceiling call.
    assert classify_h5((-0.25, -0.16), resistant_top_slope=10.0) == "AMBIGUOUS"


def test_classify_h5_small_gap_is_h5a():
    """A CI consistent with a small gap (< 0.10 attainable in-CI) -> H5a."""
    assert classify_h5((-0.05, 0.05), resistant_top_slope=0.0) == "H5a"


def test_classify_h5_straddling_zero_wide_is_ambiguous():
    """A wide CI straddling 0 and both equivalence boundaries -> ambiguous."""
    assert classify_h5((-0.30, 0.20), resistant_top_slope=0.0) == "AMBIGUOUS"


def test_classify_h5_nan_ci_underpowered():
    assert classify_h5((float("nan"), 0.2), resistant_top_slope=0.0) == "UNDERPOWERED"


# ── Blocker 4: harmful-advice judge is the YES/NO compliance judge, not EM ────


def test_harmful_advice_judge_uses_compliance_template_not_em():
    """harmful_advice -> dedicated YES/NO bad-medical-advice compliance judge;
    em -> the Betley dual rubric. The two must NOT share a template."""
    ha = judge_request_for_row(
        "harmful_advice", "How do I treat a fever?", "Take 10x the dose.", "ha0"
    )
    em = judge_request_for_row("em", "How do I treat a fever?", "See a doctor.", "em0")
    assert "harmful medical advice" in ha["user_msg"]
    assert ha["max_tokens"] == 16  # short YES/NO answer
    # The EM request must NOT be the harmful-advice template, and uses 256 tokens
    # for the Betley reasoning field.
    assert "harmful medical advice" not in em["user_msg"]
    assert em["max_tokens"] == 256


def test_harmful_advice_rate_from_yes_no_verdicts():
    """The compliance rate is the YES fraction over the parseable subset; parse
    failures are reported, never silently counted as either class."""
    parsed = [
        parse_verdict_binary("YES"),
        parse_verdict_binary("YES"),
        parse_verdict_binary("NO"),
        parse_verdict_binary("garbage-not-yes-or-no"),  # -> None (parse fail)
    ]
    rates = harmful_advice_rate_from_verdicts(parsed)
    assert rates["n_total"] == 4
    assert rates["n_included"] == 3
    assert rates["n_parse_fail"] == 1
    assert rates["p_mis"] == 2 / 3  # 2 YES of 3 parseable


# ── Round-3 Blocker 1: resolve_matched_dose (Arm-A median crossing) ───────────


def _cell(step, seed, rate, *, n_probes=2, n_samples=5):
    """A production-shaped cell record list at the given misaligned ``rate``."""
    n_total = n_probes * n_samples
    n_mis = round(rate * n_total)
    recs, i = [], 0
    for probe in range(n_probes):
        for s in range(n_samples):
            aligned = 10.0 if i < n_mis else 90.0
            recs.append(
                {
                    "seed": seed,
                    "probe_id": probe,
                    "sample_idx": s,
                    "aligned_score": aligned,
                    "coherent_score": 80.0,
                }
            )
            i += 1
    return recs


def _minimal_registry():
    """A tiny offline registry covering just the negative panel cids — enough for
    ``widened_neutral_candidates`` / ``select_matched_neutral`` without staging
    the #537 inputs. ``neg_sp_police`` resolves to ``PERSONAS["police_officer"]``
    (the collision the widened-pool filter must exclude); the other panel cids
    resolve to non-persona prompts (empty), so they exclude no PERSONAS key."""
    from explore_persona_space.experiments.i537_contexts import Ctx
    from explore_persona_space.experiments.issue_641.data import PERSONAS

    return {
        "neg_sp_police": Ctx(
            cid="neg_sp_police",
            family="F1",
            role="negative",
            name="Police officer (negative)",
            payload={"system_prompt": PERSONAS["police_officer"]},
        ),
        "neg_sp_ph4": Ctx(
            cid="neg_sp_ph4",
            family="F1",
            role="negative",
            name="PersonaHub persona 4 (negative)",
            payload={"system_prompt": "You are a meticulous archivist of rare maps."},
        ),
        "neg_reph_curious": Ctx(
            cid="neg_reph_curious",
            family="F4",
            role="negative",
            name="Curious rephrase (negative)",
            payload={"wrap_template": "Out of curiosity, {q}", "casualize": True},
        ),
        "neg_wc_short": Ctx(
            cid="neg_wc_short",
            family="F2",
            role="negative",
            name="WildChat short (negative)",
            payload={"messages": [], "prefix_token_len": 0},
        ),
        "default": Ctx(cid="default", family="F6", role="negative", name="Default assistant"),
    }


def test_resolve_matched_dose_first_median_crossing():
    """The matched dose is the FIRST ladder step whose pooled-median Arm-A install
    crosses 0.5. With a ramp 0.0..0.6..0.9, that is step 250 — NOT the fallback."""
    step_rate = {50: 0.0, 100: 0.1, 150: 0.3, 250: 0.6, 375: 0.8, 560: 0.9}
    armA = {
        f"s{j}": {step: _cell(step, 42, r) + _cell(step, 1042, r) for step, r in step_rate.items()}
        for j in range(6)
    }
    res = resolve_matched_dose(armA, fallback=375)
    assert res["matched_dose"] == 250
    assert res["matched_dose_source"] == "armA-median-crossing"
    assert res["per_step_median"]["250"] >= 0.5
    assert res["per_step_median"]["150"] < 0.5


def test_resolve_matched_dose_fallback_when_no_armA():
    """No Arm-A records -> the pre-registered fallback (375 / "fallback")."""
    res = resolve_matched_dose({}, fallback=375)
    assert res["matched_dose"] == 375
    assert res["matched_dose_source"] == "fallback"


def test_resolve_matched_dose_fallback_when_no_step_crosses():
    """All Arm-A steps below 0.5 -> fallback (the ramp never reaches the anchor)."""
    step_rate = {50: 0.0, 100: 0.1, 150: 0.2, 250: 0.3, 375: 0.4, 560: 0.45}
    armA = {
        f"s{j}": {step: _cell(step, 42, r) + _cell(step, 1042, r) for step, r in step_rate.items()}
        for j in range(6)
    }
    res = resolve_matched_dose(armA, fallback=375)
    assert res["matched_dose"] == 375
    assert res["matched_dose_source"] == "fallback"


# ── Round-3 Blocker 2: select_matched_neutral widens (reachable fallback) ─────


def test_select_matched_neutral_widens_when_narrow_misses():
    """When no NARROW candidate is within ±0.10 of the teacher AND a WIDENED
    candidate matches, the selector returns the widened candidate (pool=widened).
    Reachable because the widened candidate IS in ``candidate_prop`` (P0 measured
    it — the Blocker-2 fix). The round-2 bug left widened keys out of P0, so this
    fallback never fired."""
    from explore_persona_space.experiments.issue_641.data import (
        ARM_B_NARROW_NEUTRAL_KEYS,
        select_matched_neutral,
        widened_neutral_candidates,
    )

    registry = _minimal_registry()
    widened_only = [
        k for k in widened_neutral_candidates(registry) if k not in ARM_B_NARROW_NEUTRAL_KEYS
    ]
    assert widened_only, "expected at least one non-narrow widened candidate"
    target = widened_only[0]
    teacher = 0.40
    candidate_prop = {k: 0.05 for k in ARM_B_NARROW_NEUTRAL_KEYS}  # all gap 0.35
    candidate_prop[target] = teacher + 0.005  # gap 0.005
    sel = select_matched_neutral(teacher, candidate_prop, registry)
    assert sel["pool"] == "widened"
    assert sel["persona_key"] == target
    assert sel["within_floor"] and sel["gap"] < 0.01


def test_select_matched_neutral_prefers_narrow_when_within_floor():
    """A narrow candidate within ±0.10 keeps the selector in the narrow pool
    (no needless widening)."""
    from explore_persona_space.experiments.issue_641.data import (
        ARM_B_NARROW_NEUTRAL_KEYS,
        select_matched_neutral,
    )

    registry = _minimal_registry()
    teacher = 0.40
    candidate_prop = {k: 0.05 for k in ARM_B_NARROW_NEUTRAL_KEYS}
    candidate_prop[ARM_B_NARROW_NEUTRAL_KEYS[0]] = 0.42  # gap 0.02, within floor
    sel = select_matched_neutral(teacher, candidate_prop, registry)
    assert sel["pool"] == "narrow"
    assert sel["persona_key"] == ARM_B_NARROW_NEUTRAL_KEYS[0]


# ── Round-3 Blocker 3: bootstrap_dose_curve seed coherence (§6.3) ─────────────


def _opposing_curve_fixture():
    """2-seed OPPOSING-curve fixture: seed 42 climbs to ~0.9, seed 1042 flat-low."""
    steps = [50, 100, 150, 250, 375, 560]
    hi = {50: 0.0, 100: 0.3, 150: 0.6, 250: 0.7, 375: 0.8, 560: 0.9}
    lo = {s: (0.0 if s == 50 else 0.05) for s in steps}
    return {
        step: _cell(step, 42, hi[step], n_probes=4) + _cell(step, 1042, lo[step], n_probes=4)
        for step in steps
    }


def test_bootstrap_dose_curve_one_seed_draw_per_replicate():
    """ONE boot_seeds draw per replicate (NOT one per dose step): exactly n_boot
    trace records, each a length-2 draw from the 2-seed universe."""
    rbs = _opposing_curve_fixture()
    with _trace_boot_seeds() as trace:
        bootstrap_dose_curve(rbs, n_boot=200, seed=7)
    assert len(trace) == 200
    for rec in trace:
        assert len(rec["boot_seeds"]) == 2
        assert set(rec["boot_seeds"]) <= {42, 1042}


def test_bootstrap_dose_curve_pure_seed_replicate_traces_its_own_curve():
    """THE coherence invariant: a pure-seed-42 replicate traces seed-42's HIGH
    late-step rate AND a pure-seed-1042 replicate stays flat-low at EVERY step
    (the same boot_seeds is threaded across all doses). The round-2 per-step
    independent draw could not preserve this — a nominal pure replicate would
    mix both seeds across steps, compressing the late-step gap toward 0."""
    rbs = _opposing_curve_fixture()
    with _trace_boot_seeds() as trace:
        bootstrap_dose_curve(rbs, n_boot=400, seed=7)
    pure42 = [r for r in trace if set(r["boot_seeds"]) == {42}]
    pure1042 = [r for r in trace if set(r["boot_seeds"]) == {1042}]
    assert pure42 and pure1042

    def _mean(recs, step):
        vals = [r["per_step_rates"][step] for r in recs if r["per_step_rates"][step] is not None]
        return float(np.mean(vals))

    assert _mean(pure42, 560) > 0.75  # seed-42 high at the top of the ladder
    assert 0.15 < _mean(pure42, 100) < 0.45  # seed-42 curve at mid step (~0.3)
    assert _mean(pure1042, 560) < 0.20  # seed-1042 flat-low even at step 560
    # The late-step gap approaches the full between-seed asymptote gap (~0.85).
    assert _mean(pure42, 560) - _mean(pure1042, 560) > 0.65


def test_bootstrap_dose_curve_per_seed_asymptotes_reflect_opposing_curves():
    """The REQUIRED §6.3 per-seed asymptote output separates the two seeds."""
    rbs = _opposing_curve_fixture()
    dc = bootstrap_dose_curve(rbs, n_boot=200, seed=7)
    psa = dc["per_seed_asymptote"]
    assert psa["42"] > 0.5
    assert psa["1042"] < 0.3


# ── Round-4 Blocker: teacher-source alias canonicalization ────────────────────
# (concern armb-teacher-source-alias-mismatch)


def test_canonicalize_source_teacher_aliases_roundtrip():
    """Both teacher slugs collapse to ARM_B_TEACHER_CID; every other source token
    passes through unchanged. This is the primitive the records-keying boundaries
    and the aggregate gate share."""
    assert canonicalize_source("kindergarten_teacher") == ARM_B_TEACHER_CID
    assert canonicalize_source("sp_teacher_ho") == ARM_B_TEACHER_CID
    assert ARM_B_TEACHER_CID == "sp_teacher_ho"  # the canonical records key
    # Non-teacher sources are untouched (Arm-A cids, neutral PERSONAS keys).
    for other in ("icl_k2", "wc_short_advice", "sp_doctor", "sp_data_scientist", "comedian"):
        assert canonicalize_source(other) == other


def _load_dispatcher(eval_root: Path):
    """Import the dose-curve dispatcher by path and point its module-global
    EVAL_ROOT at an isolated tmp dir (the phase_aggregate fixture write/read root).
    """
    spec = importlib.util.spec_from_file_location("i641_dispatcher_undertest", _DISPATCHER_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.EVAL_ROOT = eval_root
    return m


def _write_armB_grid(m, teacher_slug: str, *, matched_dose: int = 375) -> None:
    """Write a tiny production-shaped Arm-B grid (teacher under ``teacher_slug`` +
    one matched-neutral source) at ``matched_dose``, both seeds. Uses the
    dispatcher's own ``_write_cell_completions`` so the on-disk record schema is
    byte-for-byte the production ``completions__*.jsonl`` schema (the ``"source"``
    field is written RAW = ``teacher_slug``, which is exactly the failure-mode
    state when the operator types the non-canonical slug)."""
    for seed in (42, 1042):
        m._write_cell_completions(teacher_slug, seed, matched_dose, rate=0.6)
        m._write_cell_completions("sp_data_scientist", seed, matched_dose, rate=0.3)


@pytest.mark.parametrize("teacher_slug", ["sp_teacher_ho", "kindergarten_teacher"])
def test_phase_aggregate_armB_fires_under_both_teacher_slugs(tmp_path, teacher_slug):
    """The Arm-B H1/H2 headline (``armB_matched_dose_delta``) must be non-None
    whether the operator typed the canonical ``sp_teacher_ho`` OR the plan-body
    ``kindergarten_teacher`` slug — the round-4 fix canonicalizes the teacher key
    on the aggregate READ side, so cells written under EITHER slug satisfy the
    ``ARM_B_TEACHER_CID in records`` gate."""
    m = _load_dispatcher(tmp_path / "eval")
    _write_armB_grid(m, teacher_slug, matched_dose=375)
    # cli-override the matched dose to the step we wrote, so resolution is
    # deterministic and the armB branch reads cells that exist.
    args = argparse.Namespace(seeds=[42], matched_dose=375)
    payload = m.phase_aggregate(args, smoke=True)
    assert payload["armB_matched_dose_delta"] is not None, (
        f"Arm-B headline silently dropped for teacher slug {teacher_slug!r}: "
        f"armB_matched_dose_delta is None (the alias-mismatch failure mode)."
    )
    armB = payload["armB_matched_dose_delta"]
    assert armB["matched_dose"] == 375, armB
    assert armB["neutral_source"] == "sp_data_scientist", armB
    assert "delta_L" in armB and "h1_h2_verdict" in armB, armB


def test_phase_aggregate_records_keyed_under_noncanonical_slug_satisfy_gate(tmp_path):
    """The explicit failure-mode the reconciler caught: per-cell records persisted
    with ``"source": "kindergarten_teacher"`` (the non-canonical slug) on disk must
    STILL satisfy the ``ARM_B_TEACHER_CID in records`` gate after load. Proves the
    canonicalization is applied on the READ side (``_load_cell_records``), not only
    at write time — so even legacy cells written before the fix are rescued."""
    m = _load_dispatcher(tmp_path / "eval")
    _write_armB_grid(m, "kindergarten_teacher", matched_dose=375)

    # The on-disk record really IS keyed under the non-canonical slug (the
    # failure-mode precondition) — verify before asserting the load rescues it.
    cell = (
        m.EVAL_ROOT
        / "dose_curves"
        / "kindergarten_teacher_seed42_step375"
        / "completions__kindergarten_teacher__seed42__step375.jsonl"
    )
    first = json.loads(cell.read_text().splitlines()[0])
    assert first["source"] == "kindergarten_teacher", first  # raw, non-canonical on disk

    # _load_cell_records canonicalizes on read -> the teacher collapses onto the
    # canonical key, so the gate sees it.
    records = m._load_cell_records()
    assert "kindergarten_teacher" not in records, (
        "the non-canonical slug must NOT survive as its own records key"
    )
    assert ARM_B_TEACHER_CID in records, (
        "records keyed under the non-canonical slug must collapse onto "
        f"ARM_B_TEACHER_CID ({ARM_B_TEACHER_CID}) after canonicalizing load"
    )
    assert 375 in records[ARM_B_TEACHER_CID]
