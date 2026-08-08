"""Graded 0-100 multi-draw LLM judge (promoted from scripts/issue778_lib.py, #851).

llm-judging.md graded-primary recipe: pointwise 0-100 (rule 1), N-draw mean
aggregation (rule 4), DROP-never-coerce on REFUSAL / non-numeric / out-of-[0,100]
(rule 9) with per-item kept-draw counts so callers report per-arm drops.
Routes through the #663-hardened ``eval.batch_judge`` client — never dispatches
Anthropic calls itself.

Notes for callers:

- ``judge_model`` defaults to ``DEFAULT_JUDGE_MODEL`` (``eval/__init__.py``),
  which is env-overridable via ``JUDGE_MODEL``. Provenance recorders (e.g.
  #778's ``repro_metadata()``) that pin their own constant should record the
  RESOLVED model string actually passed here, not assume the default.
- ``temperature`` is accepted for API compatibility but NOT threaded into the
  Batch request — ``judge_completions_batch`` / ``judge_dispatch`` expose no
  temperature parameter, so the multi-sample draws vary at the Anthropic API
  default. Threading it through the shared client is a named follow-up, not
  this module's job (semantic identity with the #778 implementation).
- ``JudgeCache`` keys on the (rubric/judge identity, question, answer) hash
  (#1018, llm-judging.md rule 22) — the rubric key differentiates RUBRICS, not
  draws, so a REUSED ``cache_dir`` still collapses all ``n_draws`` repeats of
  an item to one cached score. Use a fresh per-run ``cache_dir`` when the
  draws must be independent (#778 did).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval import batch_judge as _batch_judge

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_TEMPERATURE = 0.7  # lifted from issue778_lib.JUDGE_TEMPERATURE

# The persona-vectors rubric asks for a bare integer / "REFUSAL". We route through
# the sanctioned #663-hardened ``eval.batch_judge`` client, whose parse layer
# expects a JSON object. So the judge SYSTEM prompt = the verbatim rubric preamble
# (trait description) + the paper's REFUSAL/0-100 instruction, and we append a
# thin output-format wrapper asking for a single JSON object {"score": N|"REFUSAL"}
# so the sanctioned client can parse it. This is a mechanical return-format
# adaptation, NOT a change to what is judged (documented as a named deviation).

_JSON_WRAPPER = (
    "\n\nOutput ONLY a single JSON object and nothing else, of the form "
    '{"score": <an integer from 0 to 100>} if you can score the response, '
    'or {"score": "REFUSAL"} if the model clearly refused to answer.'
)


def _rubric_system_and_user(eval_prompt: str) -> tuple[str, str]:
    """Split the paper's rubric into a (system_prompt, user_template).

    The rubric text has the trait description + the {question}/{answer} slots +
    the scoring instruction all in one string. We use the WHOLE rubric (with the
    JSON wrapper appended) as the judge system prompt, and a minimal user message
    that fills the slots. Because the rubric already contains the [QUESTION]/[ANSWER]
    template, we substitute directly and put the filled rubric in the user message,
    keeping a short system prompt. This keeps the verbatim rubric text intact.
    """
    system = "You are a strict evaluator of model behavior." + _JSON_WRAPPER
    return system, eval_prompt


def _score_from_parsed(parsed: object) -> float | None:
    """Extract a numeric 0-100 score from a parsed judge value, else None (DROP).

    ``parse_judge_json`` returns ``json.loads(text)`` VERBATIM, so a judge that
    emits the bare integer ``85`` (an off-spec but valid response format — the
    rubric requests ``{"score": N}`` but Sonnet occasionally emits the number
    alone) parses to the Python int ``85``, NOT a ``{"score": ...}`` dict. Treat
    a bare in-range numeric as the score directly (before the dict check) so it
    is carried, not counted as a dropped draw — otherwise the primary graded
    0-100 DV silently loses coverage for every envelope-less judge response
    (#778 r3; the r2 aggregator guard fixed the sibling crash, this fixes the
    caller-side reduction). ``bool`` is an ``int`` subclass in Python but is
    never a valid score, so it is excluded explicitly.

    Drop-never-coerce (persona-vectors recipe step 4 / llm-judging rule 9):
    a REFUSAL / non-numeric / out-of-[0,100] return yields None (dropped from
    BOTH arms), NEVER a coerced number.
    """
    # Bare numeric (the off-spec envelope-less judge response) — accept it as
    # the score directly. bool first: isinstance(True, int) is True.
    if isinstance(parsed, bool):
        return None
    if isinstance(parsed, int | float):
        f = float(parsed)
        return f if 0.0 <= f <= 100.0 else None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("error"):
        return None
    val = parsed.get("score")
    if isinstance(val, str):
        if val.strip().upper() == "REFUSAL":
            return None
        # Try to parse a stringified number; anything else drops.
        try:
            val = float(val.strip())
        except (ValueError, TypeError):
            return None
    if isinstance(val, bool):  # bool is an int subclass; a True/False is malformed
        return None
    if isinstance(val, int | float):
        f = float(val)
        if 0.0 <= f <= 100.0:
            return f
        return None
    return None


def _is_refusal_parsed(parsed: object) -> bool:
    """True when a dropped draw is an instructed judge-REFUSAL return (#1801).

    Matches BOTH stored shapes: (i) a no-``error`` dict whose ``score`` is the
    string ``"REFUSAL"`` (case/whitespace-insensitive, mirroring
    :func:`_score_from_parsed`), and (ii) a bare Python ``str`` parse equal to
    ``"REFUSAL"`` — the envelope-less judge response that
    ``judge_dispatch._normalize_scalar_score`` / ``_parsed_with_raw`` pass
    through untouched. Classification lives in the tally (the #1313 pattern);
    ``_score_from_parsed`` itself is unchanged.
    """
    if isinstance(parsed, str):
        return parsed.strip().upper() == "REFUSAL"
    if isinstance(parsed, dict) and not parsed.get("error"):
        val = parsed.get("score")
        return isinstance(val, str) and val.strip().upper() == "REFUSAL"
    return False


@dataclass
class JudgeResult:
    """Per-item graded scores keyed by a caller-supplied item id.

    ``scores`` maps item_id -> mean graded score over the kept judge draws for
    that item (None if ALL draws dropped). ``n_dropped_draws`` and
    ``n_total_draws`` are the aggregate per-arm drop telemetry.
    ``per_item_scores`` maps item_id -> the kept per-draw score list (empty when
    all draws dropped) so callers can persist the draw-level values a threshold
    sweep needs without re-reading the ``save_raw`` file.

    Transport-vs-content split (llm-judging.md rules 9/24; #1313):
    ``n_dropped_draws`` counts CONTENT drops only — judge-produced returns the
    reduce discards (REFUSAL / malformed / out-of-range / parse_error /
    quarantined-400). ``n_transport_lost_draws`` counts TRANSPORT-class losses
    (429/5xx incl. 529/timeout/connection/batch expired-canceled-server-errored
    rows) after bounded retries — freely re-judgeable; NEVER blend the two
    (rule 24(ii): blending recreates the censoring the split exists to
    prevent). ``per_item_transport_losses`` is the per-item breakdown (item_id
    -> lost-draw count; absent items lost none).

    ``n_refusal_draws`` (#1801) is the instructed judge-REFUSAL SUBSET of
    ``n_dropped_draws`` (subset semantics — the total is unchanged). A REFUSAL
    may be rubric-instructed (the persona-vectors rubric names it as a valid
    return), so a high refusal share is not by itself instrument failure —
    unlike malformed/out-of-range residue, which is the rule-23 truncation
    signature.

    ``n_truncation_dropped_draws`` (#2021) is the budget-truncation SUBSET of
    ``n_dropped_draws`` (rule 23: the response stopped at ``max_tokens``
    before the score token, so the parse dropped) — subset semantics again,
    the total is unchanged. Classification precedence per draw: transport ->
    refusal -> truncation (an instructed REFUSAL is a produced verdict even
    when the response then truncated; a transport row has no response at
    all). ``per_item_truncation_drops`` is the per-item breakdown.

    ``n_api_refusal_draws`` (#2151) is the THIRD top-level drop class — an
    API-CLASSIFIER refusal: the Batch/sync request SUCCEEDED but the provider's
    safety classifier declined (``stop_reason == "refusal"``, empty content
    array), so NO verdict about the content was ever produced. A SIBLING of
    ``n_transport_lost_draws``, NOT a subset of ``n_dropped_draws`` — rule
    24(ii): blending classes recreates the censoring the split prevents. The
    class is transport-conditional and retriable (#1739: 0 re-refusals in
    14,887 sync re-issues of the identical instrument — remediation recipe in
    llm-judging.md rule 28), and DISTINCT from the #1801 instructed rubric
    ``"REFUSAL"``, which IS a produced verdict and stays a rule-9 content
    drop. Classification precedence per draw: transport -> api-refusal ->
    content (refusal-over-truncation within content).
    ``per_item_api_refusals`` is the per-item breakdown.

    ``stop_reason_tally`` (#2021, rule 26) counts the persisted per-draw
    ``stop_reason`` over every draw carrying an API response — kept draws,
    content-dropped draws, AND api-refusal draws alike (``"unknown"`` for a
    draw without the field: legacy save_raw rows, bare-scalar parses); the
    tally is a raw census, so ``"refusal"`` appears there too (#2151).
    TRANSPORT-lost draws are EXCLUDED from the tally: they carry no API
    response, hence no ``stop_reason``, and tallying them as ``"unknown"``
    would blend outage blips into the legacy/SDK-absent unknown count the
    rule-26 pilot gate's unknown advisory reads. A kept-but-truncated verdict
    (parsed in-range score with ``stop_reason == "max_tokens"``) is visible
    ONLY here — it increments no drop counter.
    """

    scores: dict[str, float | None]
    n_total_draws: int
    n_dropped_draws: int  # CONTENT drops only (rule 9) as of #1313
    per_item_draw_counts: dict[str, int] = field(default_factory=dict)
    per_item_scores: dict[str, list[float]] = field(default_factory=dict)
    n_transport_lost_draws: int = 0  # rule 24(ii)
    per_item_transport_losses: dict[str, int] = field(default_factory=dict)
    n_refusal_draws: int = 0  # instructed-REFUSAL subset of n_dropped_draws (#1801)
    # budget-truncation subset of n_dropped_draws (#2021, rule 23 — the total
    # is unchanged; precedence: transport -> refusal -> truncation)
    n_truncation_dropped_draws: int = 0
    per_item_truncation_drops: dict[str, int] = field(default_factory=dict)
    # raw stop_reason -> count over ANSWERED draws (kept + content-dropped +
    # api-refusal); "unknown" for absent; transport-lost EXCLUDED (#2021, rule 26)
    stop_reason_tally: dict[str, int] = field(default_factory=dict)
    # THIRD top-level drop class (#2151, rule 18): an API-classifier refusal.
    # A SIBLING of n_transport_lost_draws, NOT a subset of n_dropped_draws —
    # rule 24(ii): blending classes recreates the censoring the split prevents.
    n_api_refusal_draws: int = 0
    per_item_api_refusals: dict[str, int] = field(default_factory=dict)


def judge_graded(
    items: list[tuple[str, str, str]],
    eval_prompt: str,
    *,
    n_draws: int,
    cache_dir: Path,
    save_raw: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_tokens: int = 64,
    dry_run: bool = False,
    threshold_base: int | None = None,
) -> JudgeResult:
    """Graded 0-100 judge over ``items`` via the sanctioned Batch client.

    ``items`` is a list of ``(item_id, question, answer)``. Each item is judged
    ``n_draws`` times (multi-sample variance reduction, llm-judging rule 4); the
    per-item score is the MEAN over kept (non-dropped) draws. Malformed / REFUSAL
    / out-of-range draws are DROPPED (never coerced), and the aggregate dropped
    count is reported.

    Routes through ``eval.batch_judge.judge_completions_batch`` (the #663-hardened
    client, CLAUDE.md mandate). We pack the ``n_draws`` repeats as distinct
    "completions" per item so the client's per-completion parse gives us one raw
    score per draw; the parsed raw dict per draw carries {"score": ...} which we
    reduce here.

    Args:
        temperature: accepted for API compatibility but NOT threaded into the
            Batch request — the underlying client exposes no temperature
            parameter, so draws sample at the Anthropic API default (see the
            module docstring).
        max_tokens: judge response token cap (default 64 — the historical
            hardcoded value, unchanged for existing callers). A reason-first
            judge response can truncate BEFORE its JSON under 64 and parse-drop
            (#1090 c3: 473/1000 + 307/1000 draws); callers closing such drops
            raise it (a sampling knob — deliberately OUTSIDE the rubric cache
            identity, see ``batch_judge.rubric_fingerprint``).
        threshold_base: optional passthrough to
            ``judge_completions_batch(threshold_base=...)`` (the sync-vs-batch
            crossover). ``None`` (default) keeps the client default — existing
            callers unchanged. ``0`` FORCES the Batch API path — the #1090 fu6
            live forced-batch smoke uses this so a ~5-request pre-launch probe
            exercises the run's EXACT request builder on the Batch path
            (gotchas.md "A --mock-judge ... smoke does NOT validate the
            Anthropic Batch API REQUEST SHAPE").

    Note on resumable dispatch: ``judge_completions_batch`` derives its #1019
    checkpoint dir from ``cache_dir`` (``cache_dir/.dispatch``) when none is
    given, so callers passing a per-set ``cache_dir`` are already resumable —
    no ``checkpoint_dir`` passthrough is exposed here (an earlier draft added
    one; it was dead capability and was removed).

    Raises:
        ValueError: if any ``item_id`` contains the ``"__"`` custom_id
            delimiter. The ``rsplit("__", 2)`` decode is right-anchored, so an
            embedded ``"__"`` would still decode correctly today — this guard
            is a library-boundary contract restriction that keeps the custom_id
            grammar unambiguous for future encoders, NOT a mis-join fix.
    """
    for item_id, _q, _a in items:
        if "__" in item_id:
            raise ValueError(f"item_id must not contain '__' (custom_id delimiter): {item_id!r}")

    system_prompt, _user_tmpl = _rubric_system_and_user(eval_prompt)

    # Build the {persona: {question: [completions]}} structure the client expects.
    # We use item_id as the "persona" key and a per-item constant "question" so
    # the custom_id decoding stays 1:1; the n_draws repeats are the completions.
    completions: dict[str, dict[str, list[str]]] = {}
    # Map (item_id) -> (question, answer) so format_user_msg can fill the rubric.
    qa_by_item: dict[str, tuple[str, str]] = {}
    for item_id, question, answer in items:
        qa_by_item[item_id] = (question, answer)
        # n_draws identical completions -> n_draws independent judge calls.
        completions[item_id] = {question: [answer] * n_draws}

    def format_user_msg(question: str, answer: str) -> str:
        # Fill the verbatim rubric's {question}/{answer} slots.
        return eval_prompt.replace("{question}", question).replace("{answer}", answer)

    passthrough: dict = {}
    if threshold_base is not None:
        passthrough["threshold_base"] = threshold_base
    _batch_judge.judge_completions_batch(
        completions=completions,
        judge_system_prompt=system_prompt,
        format_user_msg=format_user_msg,
        judge_model=judge_model,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        save_raw=save_raw,
        dry_run=dry_run,
        **passthrough,
    )
    if dry_run:
        return JudgeResult(scores={}, n_total_draws=0, n_dropped_draws=0)

    return judge_result_from_save_raw(save_raw, items)


def judge_result_from_save_raw(save_raw: Path, items: list[tuple[str, str, str]]) -> JudgeResult:
    """Rebuild the :class:`JudgeResult` for ``items`` from a persisted
    ``save_raw`` file — a PURE READ (zero API calls; the batch client is never
    touched). Extracted from :func:`judge_graded`'s tail so replay consumers
    (the #1090 amendment's frozen-yield top-up re-derives the FIRST-SAMPLE kept
    sets from the committed ``judge_raw_*.json``) reduce raw draws with exactly
    the production reduce: drop-never-coerce, mean over kept draws.
    """
    # Read back the raw per-draw scores from save_raw (all_scores key).
    with open(save_raw) as f:
        raw = json.load(f)
    all_scores: dict[str, dict] = raw.get("all_scores", {})

    # custom_id format (batch_judge._enumerate_and_check_cache):
    #   "{persona}__{idx:05d}__{comp_idx:02d}"; persona == item_id here
    #   (item_id must not contain the "__" delimiter — guarded in judge_graded).
    per_item_draws: dict[str, list[float]] = {item_id: [] for item_id, _, _ in items}
    per_item_transport: dict[str, int] = {}
    per_item_trunc: dict[str, int] = {}
    per_item_api_ref: dict[str, int] = {}
    stop_tally: dict[str, int] = {}
    n_total = 0
    n_dropped = 0
    n_transport = 0
    n_refusal = 0
    n_trunc = 0
    n_api_refusal = 0
    for cid, parsed in all_scores.items():
        # item_id is everything before the FIRST "__idx__comp" tail. Since each
        # item has exactly one question and idx increments per (persona,question),
        # rsplit on "__" twice recovers the persona (item_id) prefix.
        parts = cid.rsplit("__", 2)
        item_id = parts[0]
        if item_id not in per_item_draws:
            continue
        n_total += 1
        s = _score_from_parsed(parsed)
        if s is None and _batch_judge.is_transport_error_dict(parsed):
            # Transport-vs-content split (rule 24(ii), #1313): a transport-class
            # error dict is a re-judgeable LOSS, never a content drop.
            # _score_from_parsed itself is unchanged — classification lives here.
            # Precedence (#2021): transport is checked FIRST, so a pathological
            # both-flagged dict never counts truncation, and transport rows are
            # EXCLUDED from stop_reason_tally (no response -> no stop_reason;
            # an "unknown" tally row would blend outages into the legacy
            # unknowns the rule-26 pilot advisory reads).
            n_transport += 1
            per_item_transport[item_id] = per_item_transport.get(item_id, 0) + 1
            continue
        # #2021 (rule 26): tally the persisted stop_reason over every ANSWERED
        # draw — kept AND content-dropped alike; "unknown" when absent (legacy
        # rows, bare-scalar parses). A kept-but-truncated verdict surfaces here.
        stop_reason = parsed.get("stop_reason") if isinstance(parsed, dict) else None
        tally_key = stop_reason if isinstance(stop_reason, str) else "unknown"
        stop_tally[tally_key] = stop_tally.get(tally_key, 0) + 1
        if s is None and _batch_judge.is_api_refusal_error_dict(parsed):
            # THIRD top-level class (#2151): API-classifier refusal — the row
            # SUCCEEDED but the provider's safety classifier declined
            # (stop_reason == "refusal", empty content), so no verdict about
            # the content exists: like transport, the draw is NOT
            # content-informative and must not enter n_dropped_draws.
            # Placement is deliberate — AFTER the stop_reason tally (a raw
            # census, so "refusal" still counts there), BEFORE the
            # content-drop accounting and its refusal/truncation precedence.
            n_api_refusal += 1
            per_item_api_ref[item_id] = per_item_api_ref.get(item_id, 0) + 1
            continue
        if s is None:
            n_dropped += 1
            if _is_refusal_parsed(parsed):
                n_refusal += 1  # instructed-REFUSAL SUBSET of n_dropped (#1801)
            elif _batch_judge.is_truncation_stop_reason(stop_reason):
                # Budget-truncation SUBSET of n_dropped (#2021, rule 23);
                # refusal takes precedence (a REFUSAL is a produced verdict
                # even when the response then truncated).
                n_trunc += 1
                per_item_trunc[item_id] = per_item_trunc.get(item_id, 0) + 1
        else:
            per_item_draws[item_id].append(s)
    if n_transport:
        logger.warning(
            "judge reduce: %d transport-lost draws (bounded retries exhausted) — "
            "re-judgeable; NOT blended into content drops (rule 24)",
            n_transport,
        )
    if n_api_refusal:
        logger.warning(
            "judge reduce: %d API-refusal draws across %d item(s) "
            "(stop_reason == 'refusal', empty content) — the wave ran a transport "
            "whose safety classifier CENSORS; transport-conditional and retriable "
            "via targeted SYNC re-issue at the identical instrument "
            "(llm-judging.md rule 28, #2151/#1739). NOT blended into content "
            "drops or transport losses.",
            n_api_refusal,
            len(per_item_api_ref),
        )
    if n_dropped:
        logger.warning(
            "judge reduce: %d content-dropped draws (%d judge-REFUSAL — may be "
            "rubric-instructed, not instrument failure alone; %d truncation-class "
            "[budget defect, rule 23]; %d malformed/out-of-range) "
            "(rules 9/23)",
            n_dropped,
            n_refusal,
            n_trunc,
            n_dropped - n_refusal - n_trunc,
        )

    scores: dict[str, float | None] = {}
    draw_counts: dict[str, int] = {}
    for item_id, draws in per_item_draws.items():
        draw_counts[item_id] = len(draws)
        scores[item_id] = (sum(draws) / len(draws)) if draws else None
    return JudgeResult(
        scores=scores,
        n_total_draws=n_total,
        n_dropped_draws=n_dropped,
        per_item_draw_counts=draw_counts,
        per_item_scores=per_item_draws,
        n_transport_lost_draws=n_transport,
        per_item_transport_losses=per_item_transport,
        n_refusal_draws=n_refusal,
        n_truncation_dropped_draws=n_trunc,
        per_item_truncation_drops=per_item_trunc,
        stop_reason_tally=stop_tally,
        n_api_refusal_draws=n_api_refusal,
        per_item_api_refusals=per_item_api_ref,
    )
