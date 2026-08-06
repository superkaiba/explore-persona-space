"""Rule-26 judge pilot gate (#2021): test-fire a judge instrument BEFORE a production wave.

:func:`judge_pilot_gate` implements ``.claude/rules/llm-judging.md`` rule 26: before
any production judge dispatch of >= ~5,000 calls, run a pilot of ~100-200 draws
spanning the wave's arms at the EXACT production instrument (rubric / ``eval_prompt``,
judge model, temperature, ``max_tokens``) and gate the full dispatch on the measured
drop profile:

- **FAIL on ANY truncation evidence** (rule 26(a)): a truncation-classified content
  drop (``JudgeResult.n_truncation_dropped_draws`` > 0) OR any truncation-class key
  (``batch_judge.TRUNCATION_STOP_REASONS``) in any arm's ``stop_reason_tally``. The
  second clause catches KEPT-but-truncated verdicts — a parsed in-range score whose
  response stopped at the budget increments no drop counter and is visible only in
  the tally. The remedy is to raise ``max_tokens`` GENEROUSLY (rule 23: a cap is not
  a spend — never shrink) and re-pilot. Truncation is NEVER waivable: a persistently
  truncating item class at a generous budget means the RULE needs amending — never
  bypass the gate.
- **FAIL on any unwaived arm** whose parse-fail rate >= ``parse_fail_threshold``
  (rule 26(b); instructed REFUSALs are excluded — a REFUSAL is a produced verdict).
- **FAIL on any transport-hollowed / under-sized arm** whose ANSWERED draw count
  falls below ``min_effective_draws_per_arm`` — the gate never PASSes on hollow
  evidence (transport losses are freely re-judgeable; re-run the pilot).

Multi-rubric waves call the gate ONCE PER RUBRIC (each rubric is its own
instrument; one pilot cannot certify another rubric's drop profile).

Cache discipline (rule 24(ii)): ``cache_dir`` MUST be a PILOT-ONLY cache root,
NEVER the production ``cache_dir`` — the rubric-keyed ``JudgeCache`` shares one key
across an item's identical draws, so pilot entries served to the production wave
would silently substitute duplicated pilot draws for fresh production draws (the
rule-24(ii) duplicated-draw trap). Production reuse of pilot draws must be a
deliberate decision, never a silent cache replay.

Import-cycle note: this is a NEW LEAF module (nothing imports ``judge_pilot``), so
top-level imports of ``graded_judge`` / ``batch_judge`` are cycle-safe — the
documented eval import cycle (``api_dispatch -> batch_judge -> alignment ->
judge_dispatch``) is untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval import graded_judge as _graded_judge
from explore_persona_space.eval.batch_judge import (
    is_transport_error_dict,
    is_truncation_stop_reason,
)
from explore_persona_space.eval.graded_judge import (
    DEFAULT_JUDGE_TEMPERATURE,
    judge_graded,
)

logger = logging.getLogger(__name__)

#: Advisory-only budget floor (plan #2021 §4 Edit 6): the gate WARNS below it and
#: never auto-shrinks or auto-raises a caller's budget. Rule 23's generous defaults
#: are far higher (>= 1024 single-rationale / >= 2048 multi-field JSON, 2026-08-02);
#: below THIS hard floor even a bare single-rationale response is at truncation risk.
PILOT_MAX_TOKENS_WARN_FLOOR = 300


@dataclass
class ArmPilotStats:
    """Per-arm pilot drop profile (all ``n_*`` counts are DRAWS, off the arm's
    :class:`~explore_persona_space.eval.graded_judge.JudgeResult` fields).

    ``parse_fail_rate`` = ``(n_content_dropped - n_refusal) / max(1, n_draws -
    n_transport_lost - n_api_refusal)`` — refusals excluded (rule 9: a produced
    verdict), transport losses AND api-refusal draws excluded from the
    denominator (rules 24/28: freely re-judgeable / transport-conditional
    censoring, never evidence about the instrument). ``n_api_refusal`` (#2151)
    is REPORT-ONLY — no gate condition keys on it (the rule-26 gate is NOT
    protective for the api-refusal class; llm-judging.md rule 28's
    non-coverage note). ``n_unknown_stop_reason_drops`` counts residual
    (non-transport, non-refusal) content DROPS lacking a persisted
    ``stop_reason`` — nonzero PARTIALLY detects a stale pre-#2021 judge cache
    being replayed (expect ~0 against a fresh pilot ``cache_dir``).
    """

    n_items: int
    n_draws: int
    n_scored: int
    n_content_dropped: int
    n_refusal: int
    n_truncation: int
    n_transport_lost: int
    n_api_refusal: int
    n_unknown_stop_reason_drops: int
    parse_fail_rate: float
    stop_reason_tally: dict[str, int]
    waived: bool


@dataclass
class PilotGateReport:
    """Rule-26 pilot verdict record — persist it in the run digest / plan §6."""

    passed: bool
    verdict: str  # "PASS" | "FAIL"
    failures: list[str]
    warnings: list[str]
    arms: dict[str, ArmPilotStats]
    judge_model: str
    max_tokens: int
    n_total_draws: int
    parse_fail_threshold: float
    rubric_hash: str

    def to_json(self) -> dict:
        """JSON-serializable dict (nested :class:`ArmPilotStats` included)."""
        return asdict(self)


def _seeded_subsample(
    items: Sequence[tuple[str, str, str]], n_take: int, *, seed: int, arm: str
) -> list[tuple[str, str, str]]:
    """Deterministic seeded subsample of ``items`` preserving original order.

    Seeded per ``(seed, arm)`` via ``random.Random(str)`` (process-independent —
    NOT affected by ``PYTHONHASHSEED``), so adding/removing a SIBLING arm never
    reshuffles this arm's draw, and the same seed reproduces the same subset.
    """
    if n_take >= len(items):
        return list(items)
    rng = random.Random(f"{seed}:{arm}")
    keep = sorted(rng.sample(range(len(items)), n_take))
    return [items[i] for i in keep]


def _count_unknown_stop_reason_drops(save_raw: Path, item_ids: set[str]) -> int:
    """Residual (non-transport, non-refusal) content DROPS with no str ``stop_reason``.

    Post-#2021 every live mint site attaches ``stop_reason`` to parsed AND
    parse-failure dicts, so against a FRESH pilot ``cache_dir`` this is ~0; a
    nonzero count PARTIALLY detects a stale pre-#2021 cache being replayed
    (partial: a stale cache can also serve non-dropped legacy entries, which
    surface as ``"unknown"`` in ``stop_reason_tally`` instead of here). Bare-scalar
    out-of-range parses (never dict-wrapped, hence never annotated) also land
    here — both cases deserve the advisory. Classification precedence mirrors the
    reduce: transport -> refusal -> (kept) -> residual drop.
    """
    with open(save_raw) as f:
        raw = json.load(f)
    n = 0
    for cid, parsed in raw.get("all_scores", {}).items():
        if cid.rsplit("__", 2)[0] not in item_ids:
            continue
        if is_transport_error_dict(parsed):
            continue  # no API response -> no stop_reason; not an "unknown" signature
        if _graded_judge._is_refusal_parsed(parsed):
            continue  # produced verdict (rule 9), not a staleness/truncation signal
        if _graded_judge._score_from_parsed(parsed) is not None:
            continue  # kept draw
        stop_reason = parsed.get("stop_reason") if isinstance(parsed, dict) else None
        if not isinstance(stop_reason, str):
            n += 1
    return n


def _truncation_failure(arm_stats: Mapping[str, ArmPilotStats], max_tokens: int) -> str | None:
    """Rule 26(a) clause: the failure line on ANY truncation evidence, else None.

    Fires on the truncation-drop counter OR any truncation-class key in any arm's
    ``stop_reason_tally`` — the tally clause alone catches KEPT-but-truncated
    verdicts (parsed in-range score, response stopped at the budget), which
    increment no drop counter. NEVER waivable.
    """
    n_trunc_drops = sum(a.n_truncation for a in arm_stats.values())
    n_trunc_tally = sum(
        count
        for a in arm_stats.values()
        for key, count in a.stop_reason_tally.items()
        if is_truncation_stop_reason(key)
    )
    if n_trunc_drops == 0 and n_trunc_tally == 0:
        return None
    trunc_arms = sorted(
        arm
        for arm, a in arm_stats.items()
        if a.n_truncation > 0
        or any(
            is_truncation_stop_reason(key) and count > 0
            for key, count in a.stop_reason_tally.items()
        )
    )
    return (
        f"truncation: {max(n_trunc_tally, n_trunc_drops)} draw(s) stopped at "
        f"max_tokens={max_tokens} (dropped: {n_trunc_drops}; truncation-class "
        f"stop_reason tally incl. KEPT verdicts: {n_trunc_tally}; arms: "
        f"{', '.join(trunc_arms)}) — raise the budget GENEROUSLY (rule 23; a cap "
        "is not a spend — never shrink) and re-pilot (rule 26(a)); truncation is "
        "NEVER waivable"
    )


def _gate_verdict(
    arm_stats: Mapping[str, ArmPilotStats],
    *,
    max_tokens: int,
    parse_fail_threshold: float,
    min_effective_draws_per_arm: int,
) -> tuple[list[str], list[str]]:
    """Apply the rule-26 gate clauses to the per-arm stats -> (failures, warnings)."""
    failures: list[str] = []
    warnings: list[str] = []

    trunc_failure = _truncation_failure(arm_stats, max_tokens)
    if trunc_failure:
        failures.append(trunc_failure)

    for arm, a in arm_stats.items():
        n_effective = a.n_draws - a.n_transport_lost
        if n_effective < min_effective_draws_per_arm:
            reason = (
                f"transport-hollowed ({a.n_transport_lost} transport-lost draw(s) are "
                "freely re-judgeable, rule 24)"
                if a.n_transport_lost
                else "under-sized pilot (raise target_total_draws / n_draws)"
            )
            failures.append(
                f"arm {arm}: only {n_effective} effective (answered) draw(s) < "
                f"min_effective_draws_per_arm={min_effective_draws_per_arm} — {reason}; "
                "re-run the pilot (the gate never PASSes on hollow evidence)"
            )

    for arm, a in arm_stats.items():
        if a.parse_fail_rate >= parse_fail_threshold:
            if a.waived:
                warnings.append(
                    f"arm {arm}: parse-fail {a.parse_fail_rate:.1%} >= "
                    f"{parse_fail_threshold:.0%} WAIVED (waive_parse_fail_arms — rule "
                    "26(b)'s explained-content-drop-class escape; REFUSAL excluded)"
                )
            else:
                failures.append(
                    f"arm {arm}: parse-fail {a.parse_fail_rate:.1%} >= "
                    f"{parse_fail_threshold:.0%} (rule 26(b); REFUSAL excluded) — "
                    "inspect the raw responses, or waive via waive_parse_fail_arms "
                    "with a recorded explanation"
                )

    if max_tokens < PILOT_MAX_TOKENS_WARN_FLOOR:
        warnings.append(
            f"max_tokens={max_tokens} is below the {PILOT_MAX_TOKENS_WARN_FLOOR}-token "
            "pilot floor and far below rule 23's generous defaults (>= 1024 "
            "single-rationale / >= 2048 multi-field JSON) — note only: the gate never "
            "auto-shrinks or auto-raises a caller's budget"
        )
    for arm, a in arm_stats.items():
        if a.n_unknown_stop_reason_drops:
            warnings.append(
                f"arm {arm}: {a.n_unknown_stop_reason_drops} content-dropped draw(s) "
                "carry NO persisted stop_reason — partially detects a stale pre-#2021 "
                "judge cache (expect ~0 against a fresh pilot cache_dir)"
            )
        if a.n_transport_lost:
            warnings.append(
                f"arm {arm}: {a.n_transport_lost} transport-lost draw(s) (rule 24 — "
                "freely re-judgeable; excluded from the tally and every gate "
                "denominator)"
            )
    return failures, warnings


def judge_pilot_gate(
    arms: Mapping[str, list[tuple[str, str, str]]],
    eval_prompt: str,
    *,
    max_tokens: int,
    cache_dir: Path,
    save_raw_dir: Path,
    n_draws: int = 2,
    target_total_draws: int = 200,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    parse_fail_threshold: float = 0.02,
    waive_parse_fail_arms: Collection[str] = (),
    min_effective_draws_per_arm: int = 10,
    threshold_base: int | None = None,
    report_path: Path | None = None,
    seed: int = 0,
) -> PilotGateReport:
    """Run the rule-26 pilot over ``arms`` and return the gate verdict.

    Per arm: a deterministic seeded subsample sized so the TOTAL pilot lands near
    ``target_total_draws`` (rule 26's ~100-200; >= 1 item per arm), judged via
    :func:`~explore_persona_space.eval.graded_judge.judge_graded` at the EXACT
    production instrument, against ``cache_dir/<arm>`` with the raw draws persisted
    to ``save_raw_dir/judge_raw_pilot_<arm>.json``. Stats come off the reduced
    :class:`~explore_persona_space.eval.graded_judge.JudgeResult` fields
    (``n_truncation_dropped_draws``, ``stop_reason_tally``, the #1313 transport
    split). Note the tally covers ANSWERED draws only — transport-lost draws carry
    no ``stop_reason`` and are excluded from it and from every gate denominator.

    Args:
        arms: arm name -> ``(item_id, question, answer)`` rows (the production
            wave's arms/conditions; every rubric of a multi-rubric wave gets its
            OWN gate call). Arm names become path components (``cache_dir/<arm>``,
            ``judge_raw_pilot_<arm>.json``) and must be filesystem-safe.
        eval_prompt: the production rubric, verbatim (``{question}``/``{answer}``
            slots per ``judge_graded``).
        max_tokens: REQUIRED, no default — the EXACT production response budget
            (rule 26: pilot at the production instrument). On a truncation FAIL,
            raise it GENEROUSLY and re-pilot; the gate never shrinks or raises a
            caller's budget itself.
        cache_dir: PILOT cache root (per-arm subdirs are created under it). MUST
            be a fresh/pilot-only dir, NEVER the production ``cache_dir`` — the
            rubric-keyed cache would serve pilot draws to the production wave
            (rule 24(ii)'s duplicated-draw trap; see the module docstring).
        save_raw_dir: directory for the per-arm raw-draw files
            (``judge_raw_pilot_<arm>.json``) — the durable pilot evidence.
        n_draws: judge draws per subsampled item (rule 4 multi-sampling).
        target_total_draws: approximate TOTAL pilot draw budget across all arms
            (rule 26's ~100-200 default).
        judge_model: judge model id — defaults to ``DEFAULT_JUDGE_MODEL`` (the one
            project judge; never hardcode a pin here).
        temperature: judge sampling temperature (production instrument).
        parse_fail_threshold: rule 26(b)'s per-arm parse-fail bar (default ~2%).
        waive_parse_fail_arms: arms whose parse-fail overshoot is EXPLAINED as a
            known rule-9 content-drop class (e.g. empty judge responses on
            degenerate steered text, #1769) — rule 26(b)'s escape. Waives the
            parse-fail check ONLY; truncation and the effective-draws floor are
            never waivable. Unknown arm names raise ``ValueError`` (fail loud on
            a typo'd waiver).
        min_effective_draws_per_arm: floor on an arm's ANSWERED draws
            (``n_draws - n_transport_lost``). Below it the gate FAILs rather than
            silently PASSing on hollow evidence — a transport-hollowed arm proves
            nothing about the instrument, and transport losses are freely
            re-judgeable: re-run the pilot.
        threshold_base: ``judge_graded`` passthrough (``0`` forces the Batch-API
            path — the pre-launch request-shape probe).
        report_path: when set, the report JSON is written there (the run-digest /
            plan-§6 verdict record).
        seed: subsample seed (same seed -> same per-arm subsets).

    Returns:
        :class:`PilotGateReport` — ``passed``/``verdict`` plus per-arm stats;
        ``failures`` lists every gate violation (empty on PASS), ``warnings``
        carries the non-verdict advisories (sub-floor ``max_tokens``, unknown
        stop_reason drops, transport losses, waived overshoots).

    Raises:
        ValueError: empty ``arms``, an empty arm, a non-filesystem-safe arm name,
            an unknown ``waive_parse_fail_arms`` entry, or (re-raised verbatim
            from ``judge_graded``) an ``item_id`` containing ``"__"``.
    """
    if not arms:
        raise ValueError("judge_pilot_gate: arms must be non-empty")
    unknown_waivers = set(waive_parse_fail_arms) - set(arms)
    if unknown_waivers:
        raise ValueError(f"waive_parse_fail_arms names unknown arm(s): {sorted(unknown_waivers)}")
    for arm in arms:
        if "/" in arm or "\\" in arm or arm in {"", ".", ".."}:
            raise ValueError(f"arm name must be a filesystem-safe path component: {arm!r}")

    cache_dir = Path(cache_dir)
    save_raw_dir = Path(save_raw_dir)
    save_raw_dir.mkdir(parents=True, exist_ok=True)
    per_arm_items = max(1, target_total_draws // (len(arms) * max(1, n_draws)))

    arm_stats: dict[str, ArmPilotStats] = {}
    for arm, items in arms.items():
        if not items:
            raise ValueError(f"arm {arm!r} has no items")
        sub = _seeded_subsample(items, per_arm_items, seed=seed, arm=arm)
        save_raw = save_raw_dir / f"judge_raw_pilot_{arm}.json"
        result = judge_graded(
            sub,
            eval_prompt,
            n_draws=n_draws,
            cache_dir=cache_dir / arm,
            save_raw=save_raw,
            judge_model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
            threshold_base=threshold_base,
        )
        # #2151: api-refusal draws (transport-conditional censoring, rule 28)
        # leave the "answered" denominator exactly as transport losses do —
        # neither is evidence about the instrument. REPORT-only: no gate
        # condition keys on n_api_refusal (assumption #2151 §12.3).
        n_answered = (
            result.n_total_draws - result.n_transport_lost_draws - result.n_api_refusal_draws
        )
        arm_stats[arm] = ArmPilotStats(
            n_items=len(sub),
            n_draws=result.n_total_draws,
            n_scored=n_answered - result.n_dropped_draws,
            n_content_dropped=result.n_dropped_draws,
            n_refusal=result.n_refusal_draws,
            n_truncation=result.n_truncation_dropped_draws,
            n_transport_lost=result.n_transport_lost_draws,
            n_api_refusal=result.n_api_refusal_draws,
            n_unknown_stop_reason_drops=_count_unknown_stop_reason_drops(
                save_raw, {item_id for item_id, _q, _a in sub}
            ),
            parse_fail_rate=(result.n_dropped_draws - result.n_refusal_draws) / max(1, n_answered),
            stop_reason_tally=dict(result.stop_reason_tally),
            waived=arm in set(waive_parse_fail_arms),
        )

    failures, warnings = _gate_verdict(
        arm_stats,
        max_tokens=max_tokens,
        parse_fail_threshold=parse_fail_threshold,
        min_effective_draws_per_arm=min_effective_draws_per_arm,
    )

    passed = not failures
    report = PilotGateReport(
        passed=passed,
        verdict="PASS" if passed else "FAIL",
        failures=failures,
        warnings=warnings,
        arms=arm_stats,
        judge_model=judge_model,
        max_tokens=max_tokens,
        n_total_draws=sum(a.n_draws for a in arm_stats.values()),
        parse_fail_threshold=parse_fail_threshold,
        rubric_hash=hashlib.sha256(eval_prompt.encode("utf-8")).hexdigest()[:16],
    )
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    logger.info(
        "[judge-pilot] verdict=%s arms=%d draws=%d max_tokens=%d failures=%d warnings=%d",
        report.verdict,
        len(arm_stats),
        report.n_total_draws,
        max_tokens,
        len(failures),
        len(warnings),
    )
    return report
