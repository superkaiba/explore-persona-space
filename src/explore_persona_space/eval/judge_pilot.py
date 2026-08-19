"""Rule-26 judge pilot gate (#2021/#2152): test-fire a judge instrument BEFORE a production wave.

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
- **REFUSE (ValueError, before any API spend) an UNSATISFIABLE configuration**
  (#2124, rule 26's sizing clause): any arm whose REALIZED draws —
  ``min(per_arm_items, len(arm_items)) * n_draws`` — fall below
  ``max(min_effective_draws_per_arm, floor(1/parse_fail_threshold) + 1)``
  cannot RESOLVE the parse-fail threshold (at n draws the smallest observable
  nonzero rate is 1/n), so the gate would FAIL on the first parse failure and
  a PASS would carry no evidence. ``allow_subresolution_pilot=True``
  downgrades the refusal to a recorded report warning.
- **FAIL on transport disparity under a DECLARED wave** (rule 26(c), #2152): when
  the caller declares the production wave's dispatch (``wave_n_calls`` /
  ``wave_threshold_base`` / ``wave_force_sync``), the wave's route is computed
  via the dispatcher's own :func:`~explore_persona_space.eval.judge_dispatch.decide_route`,
  the pilot is FORCED onto that route (``threshold_base=0`` pins batch;
  ``force_sync=True`` pins sync), and the gate FAILs on a realized-vs-declared
  mismatch, an unverifiable pilot transport (no persisted routing record), or
  cache-served pilot draws (``n_cached > 0`` — no routing provenance, and
  refusal-free by construction per the #2151 PUT-SKIP). An UNDECLARED wave keeps
  today's verdicts byte-identically and records ONE warning. An unpinned
  count-routed declaration inside the dispatcher's OTPM-probe region
  (``wave_n_calls < 2 * threshold_base``) is REFUSED fail-fast (ValueError,
  before any API spend) — the route there depends on a live OTPM probe no pilot
  can certify (#2152 MF-1; :func:`_wave_routing`).
- **FAIL on any unwaived arm** whose api-refusal rate
  (``n_api_refusal / max(1, n_draws - n_transport_lost)``) is
  ``>= api_refusal_threshold`` (rule 26(d), #2152; default 0.10 — rule 28's
  transport-conditional censor, #1739: 34.1% batch-path censoring, 0/14,887
  sync re-refusals). Waivable per arm via ``waive_api_refusal_arms`` (the
  #2091 pattern; reason recorded at the caller site); ``> 1.0`` disables the
  clause (report-only, the #2151-era behavior). Supersedes #2151 §12.3's
  REPORT-only treatment and rule 28's former non-coverage note.

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
import math
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
from explore_persona_space.eval.judge_dispatch import (
    DEFAULT_THRESHOLD_BASE,
    RoutingDecision,
    decide_route,
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
    is GATE-KEYED as of #2152: ``api_refusal_rate`` (= ``n_api_refusal /
    max(1, n_draws - n_transport_lost)`` — refusals stay IN this denominator,
    the "reached the API and got a response" set, DISTINCT from the parse-fail
    ``n_answered`` denominator above) FAILs at >= the gate's
    ``api_refusal_threshold`` unless waived (``api_refusal_waived``),
    superseding #2151 §12.3's REPORT-only treatment (llm-judging.md rule 28's
    coverage note). ``transport`` / ``n_cached`` (#2152) carry the arm's
    realized dispatch route (``'sync'``/``'batch'``/None) + cache-served draw
    count, read back from the persisted save_raw record
    (:func:`_dispatch_evidence`). ``n_unknown_stop_reason_drops`` counts residual
    (non-transport, non-refusal) content DROPS lacking a persisted
    ``stop_reason`` — nonzero PARTIALLY detects a stale pre-#2021 judge cache
    being replayed (expect ~0 against a fresh pilot ``cache_dir``).

    Per-ITEM fields (#2124, rule 29): ``n_items``, ``n_items_zero_valid``
    (items whose EVERY draw dropped — content, transport, and api-refusal
    alike), and ``frac_items_complete``
    (= ``(n_items - n_items_zero_valid) / n_items``) are the ONLY per-item
    fields; every ``n_*`` counter above remains per-DRAW. REPORT-ONLY like
    ``n_api_refusal`` — no gate condition keys on them: rule 29's floor is a
    production-wave read (``JudgeResult.frac_items_complete``), and a
    ~200-draw pilot resolves completeness only to ~1/n_items anyway.
    """

    n_items: int
    n_items_zero_valid: int
    frac_items_complete: float
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
    # #2152 fields — APPENDED with defaults (external keyword constructors:
    # scripts/issue2224_select.py, tests/test_issue2221_pipeline.py).
    api_refusal_rate: float = 0.0
    api_refusal_waived: bool = False
    transport: str | None = None
    n_cached: int = 0


@dataclass
class PilotGateReport:
    """Rule-26 pilot verdict record — persist it in the run digest / plan §6.

    #2152 fields (appended with defaults): ``wave_transport`` (the DECLARED
    wave's computed route, None when undeclared), ``pilot_transport`` (the
    aggregate realized per-arm route — any arm None -> None; unique -> that
    route; else ``"mixed"``; :func:`_pilot_transport`),
    ``api_refusal_threshold`` (rule 26(d)'s per-arm bar), and ``wave_routing``
    (``asdict`` of the computed
    :class:`~explore_persona_space.eval.judge_dispatch.RoutingDecision` — the
    run-digest record of HOW the wave's route was derived).
    """

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
    wave_transport: str | None = None
    pilot_transport: str | None = None
    api_refusal_threshold: float = 0.10
    wave_routing: dict | None = None

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


def _wave_routing(
    wave_n_calls: int | None, wave_threshold_base: int | None, wave_force_sync: bool
) -> RoutingDecision:
    """Compute the DECLARED wave's transport via the dispatcher's own pure routing rule.

    REFUSES (ValueError) any declaration whose realized route would depend on
    the dispatcher's live OTPM probe (judge_dispatch.py:1652 probes iff
    ``not force_sync and 0 < n_items < threshold_base * 2``): an unpinned
    count-routed wave inside that region routes on a value the gate cannot
    observe at pilot time, so no pilot can certify it (#2152 MF-1). Outside
    the region (``n >= 2*tb``) the route is deterministic batch (no probe;
    the effective threshold at the assumed divisor equals tb, and
    ``n >= 2*tb > tb``).
    """
    tb = DEFAULT_THRESHOLD_BASE if wave_threshold_base is None else wave_threshold_base
    if wave_force_sync or tb == 0:
        # Pinned: deterministic regardless of count and OTPM.
        return decide_route(
            wave_n_calls if wave_n_calls is not None else 1,
            threshold_base=tb,
            force_sync=wave_force_sync,
        )
    if wave_n_calls is None:
        raise ValueError(
            f"wave transport is count-routed at threshold_base={tb}: declare wave_n_calls, "
            "or pin the wave (wave_threshold_base=0 forces batch / wave_force_sync=True "
            "forces sync)"
        )
    if wave_n_calls < 2 * tb:
        raise ValueError(
            f"wave routing is OTPM-PROBE-DEPENDENT at dispatch (wave_n_calls={wave_n_calls} < "
            f"2*threshold_base={2 * tb}: the dispatcher probes live OTPM there, "
            "judge_dispatch.py:1652) and cannot be certified by any pilot — pin the wave "
            "transport: wave_threshold_base=0 (forces batch) or wave_force_sync=True (forces "
            "sync), with the SAME pin on the production dispatch (#2152)"
        )
    return decide_route(wave_n_calls, threshold_base=tb)  # n >= 2*tb: deterministic batch


def _dispatch_evidence(save_raw: Path) -> tuple[str | None, int]:
    """``(realized_path, n_cached)`` from the persisted save_raw record (batch_judge.py).

    ``realized_path`` is ``'sync'``/``'batch'`` from the ``'routing'`` record,
    or None — NO dispatch decision was persisted (fully-cached replay / a fake
    that wrote no record): the pilot's transport is UNVERIFIABLE. ``n_cached``
    counts cache-served draws merged into ``all_scores``
    (batch_judge.py:927/:944); those draws carry NO transport provenance (the
    judge cache key has no transport field, ``_hash_key``
    batch_judge.py:155-165). NOTE: ``save_raw['routing']`` records only the
    FIRST new-dispatch decision (``decisions[0]``, batch_judge.py:946); an
    in-band retry that re-routes is invisible here — safe-direction: retried
    rows are transport-class, and refusal rows are never re-dispatched in-band
    (#2151), so the scalar record cannot hide the censor.
    """
    with open(save_raw) as f:
        raw = json.load(f)
    routing = raw.get("routing")
    path = routing.get("path") if isinstance(routing, dict) else None
    return path, int(raw.get("n_cached") or 0)


def _pilot_transport(arm_stats: Mapping[str, ArmPilotStats]) -> str | None:
    """Aggregate realized per-arm transport with explicit precedence: any arm
    None -> None (an unverifiable arm beats a confident partial answer); else
    all arms one unique route -> that route; else ``"mixed"``."""
    routes = [a.transport for a in arm_stats.values()]
    if not routes or any(r is None for r in routes):
        return None
    unique = set(routes)
    return unique.pop() if len(unique) == 1 else "mixed"


def _transport_parity_failures(
    arm_stats: Mapping[str, ArmPilotStats], wave_decision: RoutingDecision | None
) -> tuple[list[str], list[str]]:
    """#2152 transport-parity clause (rule 26(c)) -> (failures, warnings).

    Under a DECLARED wave, per arm and in check order: cache-served draws
    (``n_cached > 0``) FAIL as transport-unverifiable (no routing provenance +
    refusal-free by construction — the #2151 PUT-SKIP — diluting the
    clause-(d) rate toward PASS); a missing routing record FAILs as
    unverifiable; a realized-vs-declared route mismatch FAILs. An UNDECLARED
    wave downgrades to ONE recorded warning (legacy callers keep today's
    verdicts, cache-served draws included).
    """
    failures: list[str] = []
    warnings: list[str] = []
    if wave_decision is None:
        realized = _pilot_transport(arm_stats) or "unknown"
        warnings.append(
            f"wave transport UNDECLARED — pilot ran {realized}; transport parity NOT "
            "verified (rule 26 / #2152): declare wave_n_calls / wave_threshold_base / "
            "wave_force_sync to arm the mismatch FAIL"
        )
        return failures, warnings
    for arm, a in arm_stats.items():
        if a.n_cached > 0:
            failures.append(
                f"arm {arm}: pilot transport UNVERIFIABLE — {a.n_cached} cache-served "
                "draw(s) are not transport evidence: they carry no routing provenance, "
                "and they are refusal-free by construction (#2151 cache PUT-SKIP), "
                "diluting the api-refusal rate toward PASS; run the pilot against a "
                "fresh pilot cache_dir (rule 24(ii)) (#2152)"
            )
        elif a.transport is None:
            failures.append(
                f"arm {arm}: pilot transport UNVERIFIABLE — no routing record in the "
                "persisted save_raw (fully-cached replay? rule 24(ii) demands a fresh "
                f"pilot cache_dir); parity with the declared {wave_decision.path} wave "
                "cannot be certified (#2152)"
            )
        elif a.transport != wave_decision.path:
            failures.append(
                f"arm {arm}: pilot ran {a.transport} but the wave will dispatch "
                f"{wave_decision.path} — transport-conditional failure modes (rule 28; "
                "#1739: 34.1% batch censoring invisible to a sync pilot) are invisible "
                "to this pilot; the wave declaration must mirror the production dispatch "
                "kwargs 1:1 — re-pilot on the wave's transport (#2152)"
            )
    return failures, warnings


def _refusal_resolution_floor(threshold: float) -> int:
    """Smallest ``n`` at which ONE refusal draw survives the strict ``>=`` gate.

    Direct predicate evaluation — never float ``floor(1/t)+1`` arithmetic,
    which is off-by-one on non-reciprocal thresholds (e.g. ``t=1/93``): start
    at ``ceil(1/t)`` and advance while a single draw still trips the gate
    (``1/n >= t``). 11 at the 0.10 default.
    """
    n = max(1, math.ceil(1.0 / threshold))
    while 1.0 / n >= threshold:
        n += 1
    return n


def _api_refusal_failures(
    arm_stats: Mapping[str, ArmPilotStats], *, api_refusal_threshold: float
) -> tuple[list[str], list[str]]:
    """#2152 api-refusal clause (rule 26(d)) -> (failures, warnings).

    Per arm: ``api_refusal_rate >= api_refusal_threshold`` (strict ``>=``,
    mirroring rule 26(b)) FAILs unless the arm is waived via
    ``waive_api_refusal_arms`` (then a WAIVED warning — the #2091 parse-fail
    waiver shape; the reason lives at the caller-site constant). A threshold
    ``> 1.0`` disables the clause mechanically (a rate never exceeds 1).
    Runtime resolution advisory (mirrors ``_runtime_shrink_warnings``;
    WARN-only, never a config-time refusal): an unwaived arm whose API-reached
    draws (``n_draws - n_transport_lost``) sit below the refusal-resolution
    floor is UNDER-POWERED for the threshold; ``api_refusal_threshold >= 1``
    is exempt (a disabled/report-only threshold cannot be under-powered).
    """
    failures: list[str] = []
    warnings: list[str] = []
    for arm, a in arm_stats.items():
        if a.api_refusal_rate >= api_refusal_threshold:
            if a.api_refusal_waived:
                warnings.append(
                    f"arm {arm}: api-refusal rate {a.api_refusal_rate:.1%} >= "
                    f"{api_refusal_threshold:.0%} WAIVED (waive_api_refusal_arms — rule "
                    "28's remediation-planned escape; reason recorded at the caller-site "
                    "constant)"
                )
            else:
                failures.append(
                    f"arm {arm}: api-refusal rate {a.api_refusal_rate:.1%} >= "
                    f"{api_refusal_threshold:.0%} ({a.n_api_refusal} api-refusal draw(s) "
                    "— rule 28's transport-conditional censor; #1739: 34.1% batch-path "
                    "censoring, 0/14,887 sync re-refusals) — pre-plan the targeted SYNC "
                    "re-issue remediation at the identical instrument (the #1739 "
                    "recipe), or waive via waive_api_refusal_arms with the reason "
                    "recorded at the caller-site constant (#2152)"
                )
    if api_refusal_threshold < 1.0:
        floor = _refusal_resolution_floor(api_refusal_threshold)
        for arm, a in arm_stats.items():
            if a.api_refusal_waived:
                continue
            n_reached = a.n_draws - a.n_transport_lost
            if n_reached < floor:
                warnings.append(
                    f"arm {arm}: only {n_reached} API-reached draw(s) < {floor} — "
                    f"UNDER-POWERED for api_refusal_threshold={api_refusal_threshold} "
                    "(the smallest observable nonzero rate is 1/n); WARN-only advisory, "
                    "never a config-time refusal (#2152)"
                )
    return failures, warnings


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


def _config_satisfiability_guard(
    arms: Mapping[str, list[tuple[str, str, str]]],
    *,
    per_arm_items: int,
    n_draws: int,
    parse_fail_threshold: float,
    min_effective_draws_per_arm: int,
    waive_parse_fail_arms: Collection[str],
    allow_subresolution_pilot: bool,
) -> tuple[list[str], set[str], int | None]:
    """#2124 config-time satisfiability guard (rule 26 sizing clause).

    Rule 26(b) FAILs on parse_fail_rate >= parse_fail_threshold (STRICT, see
    ``_gate_verdict``), so at n effective draws the smallest observable NONZERO
    rate is 1/n and a single parse failure survives only when 1/n < threshold:
    the per-arm floor is max(min_effective_draws_per_arm, floor(1/threshold)
    + 1) — 51 at the default 2%, not 50. Realized per-arm draws are DISCRETIZED
    (the caller's floor division) and ARM-SIZE-CAPPED (``_seeded_subsample``),
    so the guard keys on ``min(per_arm_items, len(items)) * n_draws``, never
    ``per_arm_items * n_draws``. Refuses BEFORE any ``judge_graded`` call / API
    spend (fail fast); ``allow_subresolution_pilot=True`` downgrades the
    refusal to a recorded report warning (the deliberate escape for smoke legs
    and unavoidably tiny runtime arms). Bypass scope (#2339): the escapes —
    ``waive_parse_fail_arms`` and ``allow_subresolution_pilot`` — bypass ONLY
    this config-time refusal; the verdict-time ``min_effective_draws_per_arm``
    floor in ``_gate_verdict`` has no exemption for bypassed or waived arms,
    so an arm whose max realizable draws (``len(items) * n_draws``) sit
    STRICTLY below that floor is verdict-doomed under every escape.

    Returns ``(pre_warnings, bypassed_arms, min_resolvable)``. Raises
    ``ValueError`` on a non-positive threshold, an empty arm, or an
    unsatisfiable configuration under the default strict mode.
    """
    if parse_fail_threshold <= 0:
        raise ValueError(
            f"parse_fail_threshold={parse_fail_threshold} must be > 0: the rule-26(b) "
            "check FAILs on rate >= threshold, so a non-positive threshold fails "
            "every arm unconditionally — no pilot size can resolve it"
        )
    d_eff = max(1, n_draws)
    # threshold >= 1.0 has no finite resolution floor (only a 100% parse-fail
    # rate can reach it, observable at any n >= 1) — the min-effective floor
    # alone binds there.
    min_resolvable = (
        math.floor(1.0 / parse_fail_threshold) + 1 if parse_fail_threshold < 1.0 else None
    )
    waived_set = set(waive_parse_fail_arms)
    budget_limited: dict[str, tuple[int, int]] = {}  # arm -> (realized, required)
    item_limited: dict[str, tuple[int, int, int]] = {}  # arm -> (realized, required, need)
    for arm, items in arms.items():
        if not items:
            raise ValueError(f"arm {arm!r} has no items")
        required = min_effective_draws_per_arm
        if min_resolvable is not None and arm not in waived_set:
            # A waived arm's parse-fail check never fires (rule 26(b) escape),
            # so only the min-effective hollow-evidence floor binds it.
            required = max(required, min_resolvable)
        realized = min(per_arm_items, len(items)) * d_eff
        if realized >= required:
            continue
        items_needed = math.ceil(required / d_eff)
        if len(items) < items_needed:
            item_limited[arm] = (realized, required, items_needed)
        else:
            budget_limited[arm] = (realized, required)
    pre_warnings: list[str] = []
    bypassed_arms: set[str] = set()
    if budget_limited or item_limited:
        parts: list[str] = []
        if budget_limited:
            req_max = max(required for _realized, required in budget_limited.values())
            suggested = len(arms) * d_eff * math.ceil(req_max / d_eff)
            arms_txt = "; ".join(
                f"arm {arm!r}: realized {realized} draw(s) < required {required}"
                for arm, (realized, required) in sorted(budget_limited.items())
            )
            parts.append(
                f"budget-limited: {arms_txt} — raise target_total_draws >= {suggested} "
                f"(= n_arms {len(arms)} * n_draws {d_eff} * ceil(required / n_draws) "
                f"{math.ceil(req_max / d_eff)}; the budget is floor-divided across "
                "arms, so the naive required * n_arms under-provisions)"
            )
        if item_limited:
            # #2339 honest-remedy split (TEXT-ONLY; the raise/downgrade control
            # flow is unchanged): the waive/allow_subresolution escapes lift ONLY
            # this config-time refusal (the resolution floor); the verdict-time
            # min_effective_draws_per_arm floor in _gate_verdict has NO exemption
            # for bypassed or waived arms. An arm whose maximum realizable draws
            # (len(items) * n_draws) sit STRICTLY BELOW that floor is therefore
            # verdict-DOOMED under EVERY escape (transport losses only shrink
            # effective draws further), so its remedy text must name levers that
            # can actually produce a PASS (#2329 shipped a provably-unpassable
            # fix on the old escape-naming text).
            doomed = {
                arm: v
                for arm, v in item_limited.items()
                if len(arms[arm]) * d_eff < min_effective_draws_per_arm
            }
            escapable = {arm: v for arm, v in item_limited.items() if arm not in doomed}
            if escapable:
                arms_txt = "; ".join(
                    f"arm {arm!r}: {len(arms[arm])} item(s) < {items_needed} needed "
                    f"(realized {realized} draw(s) < required {required})"
                    for arm, (realized, required, items_needed) in sorted(escapable.items())
                )
                parts.append(
                    f"item-limited: {arms_txt} — NO target_total_draws can fix an arm "
                    "holding fewer than ceil(required / n_draws) items; waive it "
                    "(waive_parse_fail_arms, with a recorded reason) or accept a "
                    "sub-resolution pilot (allow_subresolution_pilot=True); NOTE: at "
                    "exact equality (len(items) * n_draws == min_effective_draws_per_arm) "
                    "a PASS has ZERO transport-loss headroom — a single transport-lost "
                    "draw drops the arm below the verdict-time min-effective floor"
                )
            if doomed:
                arms_txt = "; ".join(
                    f"arm {arm!r}: {len(arms[arm])} item(s) x n_draws {d_eff} = "
                    f"{len(arms[arm]) * d_eff} realizable draw(s) < "
                    f"min_effective_draws_per_arm={min_effective_draws_per_arm}"
                    for arm in sorted(doomed)
                )
                items_for_floor = math.ceil(min_effective_draws_per_arm / d_eff)
                parts.append(
                    f"item-limited VERDICT-DOOMED: {arms_txt} — waive_parse_fail_arms "
                    "and allow_subresolution_pilot bypass ONLY this config-time refusal "
                    "(the resolution floor); the verdict-time min_effective_draws_per_arm "
                    "floor in _gate_verdict still FAILs the arm unconditionally (no "
                    "exemption for bypassed or waived arms), so an escape here converts "
                    "a pre-spend refusal into a guaranteed post-spend verdict FAIL. Real "
                    "remedies: lower the caller's min_effective_draws_per_arm to <= the "
                    "arm's realizable draws, raise n_draws, or enlarge the arm's item "
                    "pool to >= ceil(min_effective_draws_per_arm / n_draws) = "
                    f"{items_for_floor} item(s); after the floor/pool remedy the escapes "
                    "become usable at config time for any residual resolution-floor "
                    "deficit"
                )
        msg = (
            "judge_pilot_gate: unsatisfiable pilot configuration — the rule-26(b) "
            f"parse-fail check (FAIL on rate >= parse_fail_threshold="
            f"{parse_fail_threshold}) needs >= max(min_effective_draws_per_arm="
            f"{min_effective_draws_per_arm}, floor(1/threshold)+1={min_resolvable}) "
            "effective draws per unwaived arm to RESOLVE the threshold (at n draws "
            "the smallest observable nonzero rate is 1/n). " + " | ".join(parts)
        )
        if not allow_subresolution_pilot:
            raise ValueError(msg)
        bypassed_arms = set(budget_limited) | set(item_limited)
        pre_warnings.append(
            "sub-resolution pilot ACCEPTED (allow_subresolution_pilot=True): " + msg
        )
    return pre_warnings, bypassed_arms, min_resolvable


def _runtime_shrink_warnings(
    arm_stats: dict[str, ArmPilotStats],
    *,
    bypassed_arms: set[str],
    min_resolvable: int | None,
) -> list[str]:
    """#2124 D-1b runtime-shrink advisory (rule 26 sizing clause).

    The config-time guard sizes PLANNED draws; realized ANSWERED draws can
    still shrink below the resolution floor through transport losses (rule 24)
    and api-refusals (rule 28), re-creating the granularity artifact after the
    guard passed. WARN only, never a FAIL — an under-powered pilot, not
    evidence the instrument is bad (the ``min_effective_draws_per_arm`` FAIL in
    ``_gate_verdict`` stays the hollow-evidence gate). Skipped for waived arms
    (their parse-fail check never fires) and for arms already accepted as
    sub-resolution under ``allow_subresolution_pilot``.
    """
    warnings: list[str] = []
    if min_resolvable is None:
        return warnings
    for arm, a in arm_stats.items():
        if arm in bypassed_arms or a.waived:
            continue
        n_answered_arm = a.n_draws - a.n_transport_lost - a.n_api_refusal
        if n_answered_arm < min_resolvable:
            warnings.append(
                f"arm {arm}: only {n_answered_arm} answered draw(s) < "
                f"floor(1/parse_fail_threshold)+1={min_resolvable} after "
                f"{a.n_transport_lost} transport loss(es) + {a.n_api_refusal} "
                "api-refusal(s) — the pilot is UNDER-POWERED for its own "
                "parse-fail threshold (the smallest observable nonzero rate "
                "is 1/n); top up / re-run the pilot (rules 24/28) — NOT "
                "evidence the instrument is bad"
            )
    return warnings


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
    allow_subresolution_pilot: bool = False,
    threshold_base: int | None = None,
    report_path: Path | None = None,
    seed: int = 0,
    wave_n_calls: int | None = None,
    wave_threshold_base: int | None = None,
    wave_force_sync: bool = False,
    api_refusal_threshold: float = 0.10,
    waive_api_refusal_arms: Collection[str] = (),
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
        allow_subresolution_pilot: when True, the #2124 config-time
            satisfiability refusal (an arm whose realized
            ``min(per_arm_items, len(arm_items)) * n_draws`` cannot RESOLVE
            ``parse_fail_threshold`` — rule 26's sizing clause) is downgraded
            from ``ValueError`` to a recorded report warning and the pilot
            proceeds. The deliberate escape for smoke legs and unavoidably
            tiny runtime arms; production callers keep the default False.
            Bypass scope (#2339): CONFIG-TIME ONLY — it does not exempt any
            arm from the verdict-time ``min_effective_draws_per_arm`` floor
            in ``_gate_verdict``, so an arm whose max realizable draws
            (``len(items) * n_draws``) sit strictly below that floor still
            FAILs the verdict after the spend; fix the floor / ``n_draws`` /
            the item pool instead (the guard's VERDICT-DOOMED remedy text).
        threshold_base: LEGACY pilot-routing knob — ``judge_graded`` passthrough
            (``0`` forces the Batch-API path — the pre-launch request-shape
            probe). Superseded by the wave declaration below, which DERIVES the
            pilot's routing; passing both raises ``ValueError``.
        report_path: when set, the report JSON is written there (the run-digest /
            plan-§6 verdict record).
        seed: subsample seed (same seed -> same per-arm subsets).
        wave_n_calls: the PRODUCTION wave's total judge-call count (#2152).
            Setting ANY of ``wave_n_calls`` / ``wave_threshold_base`` /
            ``wave_force_sync`` DECLARES the wave and arms rule 26(c) transport
            parity: the wave's route is computed via the dispatcher's own
            :func:`~explore_persona_space.eval.judge_dispatch.decide_route`,
            the pilot is FORCED onto that route, and a realized-vs-declared
            mismatch / unverifiable transport / cache-served pilot FAILs. The
            declaration MUST mirror the production wave's ACTUAL dispatch
            kwargs 1:1 — the same ``threshold_base`` / ``force_sync`` values
            the wave's own ``judge_completions_batch`` /
            ``dispatch_judge_items`` call will pass; recommended usage: derive
            BOTH the declaration AND the production dispatch kwargs from ONE
            shared caller-site constant so they cannot drift. An unpinned
            count-routed declaration inside the dispatcher's OTPM-probe region
            (``wave_n_calls < 2 * threshold_base``) is REFUSED fail-fast
            (:func:`_wave_routing`, #2152 MF-1).
        wave_threshold_base: the wave's ``threshold_base`` dispatch kwarg
            (``None`` = the dispatcher default ``DEFAULT_THRESHOLD_BASE``;
            ``0`` pins the Batch path).
        wave_force_sync: the wave's ``force_sync`` dispatch kwarg (``True``
            pins the sync path).
        api_refusal_threshold: rule 26(d)'s per-arm api-refusal bar (#2152;
            strict ``>=``, mirroring rule 26(b); default 0.10 — see rule 28's
            #1739 figures: 34.1% batch-path censoring). ``> 1.0`` disables the
            clause mechanically (report-only, the #2151-era behavior);
            ``<= 0`` raises.
        waive_api_refusal_arms: arms whose api-refusal overshoot is EXPLAINED
            and remediation-planned (a pre-registered rule-28 targeted SYNC
            re-issue) — the #2091 waiver pattern: the REASON is recorded at
            the caller-site constant. Waives the api-refusal check ONLY.
            Unknown arm names raise ``ValueError``.

    Returns:
        :class:`PilotGateReport` — ``passed``/``verdict`` plus per-arm stats;
        ``failures`` lists every gate violation (empty on PASS), ``warnings``
        carries the non-verdict advisories (sub-floor ``max_tokens``, unknown
        stop_reason drops, transport losses, waived overshoots, accepted
        sub-resolution arms, the #2124 runtime-shrink advisory, and the #2152
        undeclared-wave transport warning + api-refusal waiver/under-power
        advisories).

    Raises:
        ValueError: empty ``arms``, an empty arm, a non-filesystem-safe arm name,
            an unknown ``waive_parse_fail_arms`` / ``waive_api_refusal_arms``
            entry, a non-positive ``parse_fail_threshold`` /
            ``api_refusal_threshold``, an UNSATISFIABLE pilot configuration
            (#2124 — any arm's realized draws below
            ``max(min_effective_draws_per_arm, floor(1/threshold) + 1)``
            unless ``allow_subresolution_pilot=True``; raised BEFORE any
            ``judge_graded`` call / API spend), a wave declaration passed
            alongside the legacy ``threshold_base`` knob, a contradictory
            declaration (``wave_force_sync=True`` with
            ``wave_threshold_base=0``), ``wave_n_calls < 1``, an UNCERTIFIABLE
            declaration (#2152 MF-1 — count-routed inside the dispatcher's
            OTPM-probe region, or count-routed with no ``wave_n_calls``; raised
            BEFORE any ``judge_graded`` call / API spend), or (re-raised
            verbatim from ``judge_graded``) an ``item_id`` containing ``"__"``.
    """
    if not arms:
        raise ValueError("judge_pilot_gate: arms must be non-empty")
    unknown_waivers = set(waive_parse_fail_arms) - set(arms)
    if unknown_waivers:
        raise ValueError(f"waive_parse_fail_arms names unknown arm(s): {sorted(unknown_waivers)}")
    for arm in arms:
        if "/" in arm or "\\" in arm or arm in {"", ".", ".."}:
            raise ValueError(f"arm name must be a filesystem-safe path component: {arm!r}")

    # --- #2152 gate-entry validation (all BEFORE any judge_graded call) ------
    wave_declared = wave_n_calls is not None or wave_threshold_base is not None or wave_force_sync
    if wave_declared and threshold_base is not None:
        raise ValueError(
            "judge_pilot_gate: pass EITHER the legacy pilot routing knob threshold_base "
            "OR the wave declaration (wave_n_calls / wave_threshold_base / "
            "wave_force_sync) — a declared wave DERIVES the pilot's routing (#2152)"
        )
    if wave_force_sync and wave_threshold_base == 0:
        raise ValueError(
            "judge_pilot_gate: contradictory wave declaration — wave_force_sync=True pins "
            "sync while wave_threshold_base=0 pins batch; pick one (#2152)"
        )
    if wave_n_calls is not None and wave_n_calls < 1:
        raise ValueError(f"judge_pilot_gate: wave_n_calls={wave_n_calls} must be >= 1")
    if api_refusal_threshold <= 0:
        raise ValueError(
            f"api_refusal_threshold={api_refusal_threshold} must be > 0: rule 26(d) FAILs "
            "on rate >= threshold, so a non-positive threshold fails every arm "
            "unconditionally (> 1.0 disables the clause — report-only)"
        )
    unknown_refusal_waivers = set(waive_api_refusal_arms) - set(arms)
    if unknown_refusal_waivers:
        raise ValueError(
            f"waive_api_refusal_arms names unknown arm(s): {sorted(unknown_refusal_waivers)}"
        )
    wave_decision: RoutingDecision | None = None
    if wave_declared:
        # #2152 MF-1: refuse an uncertifiable declaration BEFORE any API spend.
        wave_decision = _wave_routing(wave_n_calls, wave_threshold_base, wave_force_sync)

    cache_dir = Path(cache_dir)
    save_raw_dir = Path(save_raw_dir)
    save_raw_dir.mkdir(parents=True, exist_ok=True)
    per_arm_items = max(1, target_total_draws // (len(arms) * max(1, n_draws)))

    # --- #2124 config-time satisfiability guard (rule 26 sizing clause) ------
    # Refuses an unsatisfiable configuration BEFORE any judge_graded call /
    # API spend (fail fast); allow_subresolution_pilot=True downgrades the
    # refusal to a recorded report warning. Arithmetic + escape semantics:
    # _config_satisfiability_guard.
    pre_warnings, bypassed_arms, min_resolvable = _config_satisfiability_guard(
        arms,
        per_arm_items=per_arm_items,
        n_draws=n_draws,
        parse_fail_threshold=parse_fail_threshold,
        min_effective_draws_per_arm=min_effective_draws_per_arm,
        waive_parse_fail_arms=waive_parse_fail_arms,
        allow_subresolution_pilot=allow_subresolution_pilot,
    )

    # #2152 transport parity: a declared wave FORCES the pilot onto the wave's
    # computed route — threshold_base=0 pins batch; force_sync=True pins sync,
    # passed CONDITIONALLY (sync-wave branch only) so legacy judge_graded
    # fakes/wrappers that predate the kwarg are never called with it. An
    # undeclared wave keeps the legacy threshold_base passthrough
    # byte-identically.
    routing_kwargs: dict = {"threshold_base": threshold_base}
    if wave_decision is not None:
        if wave_decision.path == "sync":
            routing_kwargs = {"threshold_base": None, "force_sync": True}
        else:
            routing_kwargs = {"threshold_base": 0}

    arm_stats: dict[str, ArmPilotStats] = {}
    for arm, items in arms.items():
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
            **routing_kwargs,
        )
        # #2152: realized dispatch route + cache-served draw count, read back
        # from the persisted save_raw record (the durable pilot evidence).
        transport, n_cached = _dispatch_evidence(save_raw)
        # #2151/#2152: api-refusal draws (transport-conditional censoring,
        # rule 28) leave the "answered" denominator exactly as transport
        # losses do — neither is evidence about the instrument. GATE-KEYED as
        # of #2152: the per-arm api_refusal_rate (n_api_refusal / API-reached
        # draws) FAILs at >= api_refusal_threshold unless waived, superseding
        # #2151 §12.3's REPORT-only treatment.
        n_answered = (
            result.n_total_draws - result.n_transport_lost_draws - result.n_api_refusal_draws
        )
        n_reached = result.n_total_draws - result.n_transport_lost_draws
        # #2124 (rule 29): per-item completeness, off the reduce's scores map —
        # the reduce pre-seeds every item, so scores[item] is None marks
        # all-draws-dropped. REPORT-only (see the ArmPilotStats docstring).
        n_items_zero_valid = sum(1 for v in result.scores.values() if v is None)
        arm_stats[arm] = ArmPilotStats(
            n_items=len(sub),
            n_items_zero_valid=n_items_zero_valid,
            frac_items_complete=(len(sub) - n_items_zero_valid) / len(sub),
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
            api_refusal_rate=result.n_api_refusal_draws / max(1, n_reached),
            api_refusal_waived=arm in set(waive_api_refusal_arms),
            transport=transport,
            n_cached=n_cached,
        )

    failures, warnings = _gate_verdict(
        arm_stats,
        max_tokens=max_tokens,
        parse_fail_threshold=parse_fail_threshold,
        min_effective_draws_per_arm=min_effective_draws_per_arm,
    )
    warnings = pre_warnings + warnings

    # #2152 clauses (c)+(d): transport parity + api-refusal rate. Deliberately
    # SEPARATE helpers, NOT new _gate_verdict parameters — _gate_verdict has
    # external direct callers (scripts/issue2224_select.py,
    # tests/test_issue2224_contracts.py) whose call shape must stay valid.
    tp_failures, tp_warnings = _transport_parity_failures(arm_stats, wave_decision)
    ar_failures, ar_warnings = _api_refusal_failures(
        arm_stats, api_refusal_threshold=api_refusal_threshold
    )
    failures = failures + tp_failures + ar_failures
    warnings = warnings + tp_warnings + ar_warnings

    # #2124 D-1b runtime-shrink advisory (rule 26 sizing clause): WARN when an
    # arm's realized ANSWERED draws shrank below the resolution floor after
    # the config-time guard passed. Full semantics: _runtime_shrink_warnings.
    warnings.extend(
        _runtime_shrink_warnings(
            arm_stats, bypassed_arms=bypassed_arms, min_resolvable=min_resolvable
        )
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
        wave_transport=wave_decision.path if wave_decision is not None else None,
        pilot_transport=_pilot_transport(arm_stats),
        api_refusal_threshold=api_refusal_threshold,
        wave_routing=asdict(wave_decision) if wave_decision is not None else None,
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
