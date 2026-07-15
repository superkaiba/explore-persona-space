"""Alignment judging via Anthropic Batch API with file-based caching.

Replaces the hand-rolled sequential async judge pattern with Anthropic's
native batch endpoint (50% cost discount, no rate limit management needed).

Cache pattern inspired by safety-tooling's cache_manager — simple file-based
JSON keyed by hash of (rubric/judge identity, question, completion), avoiding
redundant API calls on experiment resume without ever serving one rubric's
judgment for another (llm-judging.md rule 22, #810/#1018).

Usage:
    from explore_persona_space.eval.batch_judge import judge_completions_batch

    results = judge_completions_batch(
        completions={"villain": {"What is fairness?": ["completion1", ...]}},
        cache_dir=Path("eval_results/my_run/judge_cache"),
    )
    # results["villain"] -> {mean_aligned: 85.2, mean_coherent: 90.1, ...}
"""

import datetime as _dt
import functools
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.alignment import JUDGE_SYSTEM_PROMPT
from explore_persona_space.eval.utils import parse_judge_json
from explore_persona_space.llm.anthropic_client import (
    BatchDeadlineExceeded,
    batch_stuck_threshold_hours,
    deadline_from_expires_at,
    retrieve_with_404_tolerance,
)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Anthropic Batch API limits (verified platform.claude.com, June 2026):
#   hard caps = 100_000 requests OR 256 MB per batch, whichever first.
# We shard FAR below the request cap for incremental progress + blast-radius
# isolation (a single failing/expiring sub-batch loses <=8k items, resumable by
# custom_id, not the whole run). See task #663 plan §11.
MAX_REQUESTS_PER_BATCH = 8_000  # was 100_000 — engineering choice (#663 §11)
# Judge sub-batches shard FAR smaller than the general 8k cap. Empirically a
# ~500-request judge batch clears in ~5 min, while an 8k judge batch STARVES —
# it can sit at succeeded:0 for hours because the API schedules a large judge
# set behind everyone else's traffic (the #658 G1 wedge: an 8k judge shard sat
# at succeeded:0 for 9h before its expires_at deadline would even have fired).
# 2_000 is the conservative ceiling that keeps shards small enough to clear;
# the sibling api_dispatch path uses DEFAULT_BATCH_CHUNK_SIZE=1_000 in the same
# 500-2000 band. This is the DEFAULT for the whole judge router (judge_dispatch
# imports it as DEFAULT_SUB_BATCH_SIZE), so EVERY judge through
# judge_completions_batch / dispatch_judge_items — including #664's
# issue664_dispatch.py — shards at 2k automatically. MAX_REQUESTS_PER_BATCH (8k)
# stays distinct and applies ONLY to the non-router direct-batch paths in THIS
# file: the legacy _submit_and_poll_batch (#389) + submit_sharded_batches_fire_and_forget
# (#528). (Before #664: the 8k was wrongly the router default too, so #664's
# judge would have starved at 8k — see judge_dispatch.DEFAULT_SUB_BATCH_SIZE.)
MAX_JUDGE_REQUESTS_PER_BATCH = 2_000  # judge shard ceiling (#658); see note above
MAX_BATCH_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB safe margin under the 256 MB hard cap

# Re-export for backwards compatibility; canonical source is eval/__init__.py
# The expires_at deadline helper (deadline_from_expires_at) + BatchDeadlineExceeded
# live in llm/anthropic_client.py (the lowest layer; batch_judge imports from it,
# never the reverse) and are imported above — single source of truth (#663 §4c).


# ── custom_id helper ──────────────────────────────────────────────────────────

_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def make_custom_id(item_id: str) -> str:
    """Deterministic, regex-valid, idempotent custom_id for a judge item.

    sha256 hex is exactly 64 chars of [0-9a-f] -> matches the Anthropic Batch
    API's ``^[a-zA-Z0-9_-]{1,64}$`` constraint. Stable across runs on the same
    item_id, so resubmits reuse the same id (idempotent checkpoint/resume).
    """
    cid = hashlib.sha256(item_id.encode()).hexdigest()  # 64 hex chars
    assert _CUSTOM_ID_RE.match(cid), cid  # cheap invariant; sha256 hex always matches
    return cid


# ── Judge cache ──────────────────────────────────────────────────────────────

# Rubric-identity cache keying (llm-judging.md rule 22, #810/#1018). The literal
# version tag enters every key preimage, so this — and any FUTURE key-schema
# change that bumps it — is an explicit, automatic bust of all pre-change
# entries: a cold miss re-judges (safe direction); a wrong read is impossible.
_JUDGE_CACHE_KEY_VERSION = "EPM_JUDGE_CACHE_KEY_V2"
_RUBRIC_SENTINEL_Q = "<RUBRIC_FINGERPRINT_SENTINEL_QUESTION>"
_RUBRIC_SENTINEL_C = "<RUBRIC_FINGERPRINT_SENTINEL_COMPLETION>"


def rubric_fingerprint(
    judge_model: str,
    judge_system_prompt: str,
    format_user_msg: Callable[[str, str], str] | None = None,
) -> str:
    """Stable rubric/judge identity hash for JudgeCache keys (rule 22, #810).

    Renders ``format_user_msg`` on fixed sentinel strings so a rubric that
    lives in the USER-message template (e.g. ``graded_judge`` fills the whole
    0-100 rubric into the user msg under a generic system prompt) enters the
    key, not only a system-prompt rubric. ``None`` means "no user template
    contributes rubric content" (caller builds content-only user messages).

    This fingerprints rubric/judge IDENTITY, not the full request: sampling
    knobs like ``max_tokens``/temperature are deliberately excluded (they carry
    no rubric semantics; contrast the api_dispatch adapter, which deliberately
    over-keys on the full built request). A formatter that selects its rubric
    by INPUT-DEPENDENT branching (different rubric text per question) is
    outside the fingerprint's discrimination — no in-repo formatter does that;
    such a caller must fold the branch into the system prompt or use disjoint
    cache dirs.
    """
    rendered = (
        format_user_msg(_RUBRIC_SENTINEL_Q, _RUBRIC_SENTINEL_C)
        if format_user_msg is not None
        else "<no-user-template>"
    )
    content = f"{judge_model}\n===\n{judge_system_prompt}\n===\n{rendered}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class JudgeCache:
    """Simple file-based cache for judge results, keyed by prompt content hash.

    Each cached result is stored as a single JSON file named by the hash of
    (rubric/judge identity, question, completion) — the ``rubric_key`` is a
    REQUIRED keyword argument on every read/write so a cached judgment can
    never be served across rubrics (llm-judging.md rule 22; incident #810).
    Cache hits avoid redundant Batch API calls on experiment resume. Filenames
    keep the ``{16-hex}.json`` shape (load-bearing for
    ``issue906_phase1_pilot._is_rederivable_cache`` + the disk janitors).
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_key(question: str, completion: str, *, rubric_key: str) -> str:
        """Hash (key-schema version, rubric identity, question, completion) to 16 hex chars."""
        if not isinstance(rubric_key, str) or not rubric_key:
            raise ValueError(
                f"rubric_key must be a non-empty str (got {rubric_key!r}); derive it via "
                "rubric_fingerprint(judge_model, judge_system_prompt, format_user_msg)."
            )
        content = (
            f"{_JUDGE_CACHE_KEY_VERSION}\n---\n{rubric_key}\n---\n{question}\n---\n{completion}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, question: str, completion: str, *, rubric_key: str) -> dict | None:
        """Look up a cached judge result under ``rubric_key``. Returns None on miss."""
        key = self._hash_key(question, completion, rubric_key=rubric_key)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self._hits += 1
            with open(path) as f:
                return json.load(f)
        self._misses += 1
        return None

    def put(self, question: str, completion: str, result: dict, *, rubric_key: str) -> None:
        """Store a judge result in the cache under ``rubric_key``."""
        key = self._hash_key(question, completion, rubric_key=rubric_key)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(result, f)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses}


# ── Batch submission helpers ─────────────────────────────────────────────────


def _default_format_user_msg(question: str, completion: str) -> str:
    """Default formatting for alignment judge user messages."""
    return f"Question asked to the AI:\n{question}\n\nAI's response:\n{completion}"


def _build_batch_requests(
    items: list[tuple[str, str, str, str]],  # (custom_id, question, completion, user_msg)
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
) -> list[dict]:
    """Build Anthropic Batch API request dicts."""
    return [
        {
            "custom_id": custom_id,
            "params": {
                "model": judge_model,
                "max_tokens": max_tokens,
                "system": judge_system_prompt,
                "messages": [{"role": "user", "content": user_msg}],
            },
        }
        for custom_id, _question, _completion, user_msg in items
    ]


def _chunk_requests(
    requests: list[dict],
    max_count: int = MAX_REQUESTS_PER_BATCH,
) -> list[list[dict]]:
    """Split requests into chunks that fit within Batch API limits.

    Pattern from safety-tooling's chunk_prompts_for_anthropic().
    """
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_size = 0

    for req in requests:
        req_size = len(json.dumps(req).encode())

        # Start new chunk if either limit would be exceeded
        if current_chunk and (
            len(current_chunk) >= max_count or current_size + req_size > MAX_BATCH_SIZE_BYTES
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(req)
        current_size += req_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def submit_sharded_batches_fire_and_forget(
    client: "anthropic.Anthropic",
    requests: list[dict],
    *,
    on_batch_submitted: Callable[[list[str]], None] | None = None,
) -> list[str]:
    """Submit ``requests`` sharded into chunks of <= MAX_REQUESTS_PER_BATCH, fire-and-forget.

    Returns the list of submitted batch IDs in submission order. Does NOT poll —
    intended for callers (e.g. ``scripts/i528_phase4_judge.py --backend batch``) that
    submit-then-return and merge results in a separate sync re-run, vs the polling
    ``judge_completions_batch`` / ``_submit_and_poll_batch`` path.

    ``on_batch_submitted`` (if given) is invoked with the running ``batch_ids`` list
    AFTER each chunk's batch is created, so the caller can persist incremental
    progress (Checkpoint-per-phase: a mid-loop crash leaves a recoverable record of
    the shards already submitted, rather than orphaning shards 0..k when shard k+1
    raises before any write lands). On any per-chunk failure the exception propagates
    immediately; the IDs submitted so far are already persisted via the callback.
    """
    batch_ids: list[str] = []
    for chunk in _chunk_requests(requests):
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        if on_batch_submitted is not None:
            on_batch_submitted(batch_ids)
    return batch_ids


def _legacy_error_dict(reason: str) -> dict:
    """The error-dict shape the legacy ``_submit_and_poll_batch`` returns."""
    return {"aligned": None, "coherent": None, "reasoning": reason, "error": True}


_TRANSPORT_LEGACY_REASON_PREFIXES = (
    "batch_error: expired",
    "batch_error: canceled",
    # api_dispatch transient-exhaustion reasons ("transient <ExcName> (attempt N)",
    # api_dispatch._do_one) — covers LEGACY persisted multi-org-reduce dicts whose
    # exception name (APITimeoutError/APIConnectionError/InternalServerError) has
    # no "overloaded" substring (statistics-critic round-1 Must-Fix; the reason
    # field is pipeline-minted, never judge content, and the error gate runs first).
    "transient ",
)


def is_transport_error_dict(parsed: object) -> bool:
    """Rule-24 transport-vs-content split for judge error dicts (#1313).

    True iff the row died in TRANSPORT (no verdict about the content was ever
    produced -> freely re-judgeable). Primary signal: the structural
    ``transport: True`` field minted at every #1313 error-dict site. Fallback:
    a conservative reason-string match for LEGACY persisted dicts (pre-#1313
    save_raw files / cache entries, e.g. #1090's stored 529 rows). A
    quarantined 400 (rule 24(iii)) and a parse_error (rule 23) are NOT transport.
    """
    if not isinstance(parsed, dict) or not parsed.get("error"):
        return False
    if parsed.get("transport") is True:
        return True
    reason = str(parsed.get("reasoning", parsed.get("reason", "")))
    if "invalid_request_error" in reason:
        return False
    if reason.startswith(_TRANSPORT_LEGACY_REASON_PREFIXES):
        return True
    if reason.startswith("batch_error: errored ("):
        return True  # server-class errored rows (invalid_request excluded above)
    low = reason.lower()
    return "overloaded" in low or "rate_limited_exhausted" in low


def _collect_legacy_results(
    client: "anthropic.Anthropic",
    batch_id: str,
    results: dict[str, dict],
) -> None:
    """Stream one ended batch's results into ``results`` (join on custom_id).

    Two-level errored branch per the Anthropic doc example: an
    ``invalid_request_error`` is quarantined (a malformed request fails
    identically on resubmit) and surfaced as an error dict carrying the
    ``(quarantined)`` reason; other terminal states surface as error dicts too.
    The legacy callers consume the full ``{custom_id: score}`` dict (error dicts
    included), so quarantine here is informational — never retried.
    """
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                "",
            )
            parsed = parse_judge_json(text)
            results[custom_id] = parsed if parsed is not None else _legacy_error_dict("parse_error")
        elif rtype == "errored":
            # SDK nesting is result.result.error.error.type (double .error);
            # getattr-guarded so a missing-error shape fails open (server label).
            etype = getattr(
                getattr(getattr(result.result, "error", None), "error", None), "type", None
            )
            if etype == "invalid_request_error":
                # Quarantined 400: a pipeline bug, NOT transport (rule 24(iii)).
                results[custom_id] = _legacy_error_dict(
                    "batch_error: invalid_request_error (quarantined)"
                )
            else:
                # Server-class errored row: transport-class (rule 24(i), #1313).
                results[custom_id] = {
                    **_legacy_error_dict(f"batch_error: errored ({etype or 'server'})"),
                    "transport": True,
                }
        else:  # expired / canceled / unknown -> transport-class (rule 24(i), #1313)
            results[custom_id] = {
                **_legacy_error_dict(f"batch_error: {rtype}"),
                "transport": True,
            }


def _submit_and_poll_batch(
    requests: list[dict],
    client: "anthropic.Anthropic",
    poll_interval: float = 30.0,
    max_poll_interval: float = 120.0,
    *,
    grace_min: int = 30,
    now_fn: "Callable[[], _dt.datetime] | None" = None,
    sleep_fn: "Callable[[float], None] | None" = None,
) -> dict[str, dict]:
    """Submit ``requests`` (sharded), poll each sub-batch until it ends, collect.

    Signature-stable for the two live ``scripts/issue_389`` callers
    (``run_experiment_389.py`` / ``rejudge_issue_389_c_strict.py``), which pass
    ``(requests, client, poll_interval=30.0)`` and consume the returned
    ``{custom_id: score_dict}`` dict. The rewrite (#663) hardens this in place
    so those callers inherit:

    - **Sharding**: ``requests`` is split via :func:`_chunk_requests` (<=8k +
      250 MB), so an over-cap input no longer goes into one batch.
    - **Bounded poll**: each sub-batch's poll exits on its own ``expires_at`` +
      grace (never an unbounded ``while True``); a batch still not ``ended`` at
      the deadline raises :class:`BatchDeadlineExceeded` after one final harvest
      attempt. If ``expires_at`` is ever ABSENT the deadline falls back to
      ``now + 25h`` so the loop stays hard-bounded regardless of SDK shape.
    - **Two-level terminal split**: an ``invalid_request_error`` is quarantined
      (surfaced as an error dict, never retried here); other states surface as
      error dicts too.
    - **404 tolerance (#995 + #1035)**: a ``NotFoundError`` on any retrieve —
      the loop retrieve, the deadline-time final retrieve, and the cancel-race
      probe — is retried with bounded backoff: the create-grace schedule within
      ``BATCH_CREATE_404_GRACE_S`` of the chunk's own create (read-after-write,
      the #742 shape), the ``BATCH_MIDPOLL_404_BACKOFF_S`` mid-poll schedule
      anywhere else. Past both budgets it re-raises — never masked.
    - **Stuck-batch cancel (#1019)**: a chunk still ``in_progress`` at
      ``request_counts.succeeded == 0`` for >= ``EPS_BATCH_STUCK_HOURS``
      (default 4h; ``<=0`` disables) since its own create is CANCELED (once
      per chunk; a re-retrieved ``canceling``/``ended`` counts as accepted
      cancellation; a chunk ALREADY ``canceling`` — an out-of-band Console
      cancel — is never re-canceled) and the loop keeps polling to ``ended``;
      canceled rows surface as error dicts — this path has no retry
      machinery, so a 4h-bounded completion with flagged error rows replaces
      the prior up-to-24h park behind a wedged batch (#810).

    ``now_fn``/``sleep_fn`` are injectable for tests (default wall-clock +
    ``time.sleep``). Polling keeps the 30s->1.5x->120s backoff capped by the
    deadline, not a step count.
    """
    now_fn = now_fn or (lambda: _dt.datetime.now(_dt.UTC))
    sleep_fn = sleep_fn or time.sleep

    results: dict[str, dict] = {}
    chunks = _chunk_requests(requests)
    for ci, chunk in enumerate(chunks):
        batch = client.messages.batches.create(requests=chunk)
        batch_id = batch.id
        created_at = now_fn()  # #995: anchor for the create-grace 404 retry below
        logger.info(
            "Batch %s created (sub-batch %d/%d, %d requests)",
            batch_id,
            ci + 1,
            len(chunks),
            len(chunk),
        )
        deadline: _dt.datetime | None = None
        current_interval = poll_interval
        stuck_canceled = False  # #1019: cancel a wedged chunk at most once
        while True:
            # #995 + #1035: a 404 within BATCH_CREATE_404_GRACE_S of this
            # chunk's own create is retried on the grace schedule
            # (read-after-write; the #742 shape); any other 404 gets the
            # bounded mid-poll schedule. functools.partial, NOT a lambda:
            # batch_id is reassigned per chunk iteration and ruff's B023 flags
            # a loop-closure lambda.
            batch = retrieve_with_404_tolerance(
                functools.partial(client.messages.batches.retrieve, batch_id),
                created_at=created_at,
                batch_id=batch_id,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
            )
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                logger.info(
                    "[%s] Batch %s: processing=%s succeeded=%s errored=%s",
                    time.strftime("%H:%M:%S"),
                    batch_id,
                    counts.processing,
                    counts.succeeded,
                    counts.errored,
                )
            if batch.processing_status == "ended":
                break
            if deadline is None:
                expires_at = getattr(batch, "expires_at", None)
                deadline = (
                    deadline_from_expires_at(expires_at, grace_min)
                    if expires_at is not None
                    else now_fn() + _dt.timedelta(hours=25)  # absent expires_at -> still bounded
                )
            if now_fn() > deadline:
                # #1035: the final retrieve is the LAST chance to harvest, so a
                # transient 404 here gets the bounded mid-poll tolerance too
                # (the #995 "genuinely anomalous -> unguarded" stance reversed);
                # a genuinely-gone batch still re-raises NotFoundError <=~3 min
                # later, and the BatchDeadlineExceeded semantics are untouched.
                final = retrieve_with_404_tolerance(
                    functools.partial(client.messages.batches.retrieve, batch_id),
                    created_at=created_at,
                    batch_id=batch_id,
                    now_fn=now_fn,
                    sleep_fn=sleep_fn,
                )
                if final.processing_status == "ended":
                    batch = final
                    break
                raise BatchDeadlineExceeded(batch_id, deadline)
            # Stuck-batch escape (#1019, incident #810): zero succeeded past the
            # threshold -> cancel ONCE (per-chunk local guard; this path has no
            # state.json and no retry machinery, so no intent marker is needed)
            # and keep polling — the API flips canceling -> ended and the
            # collection below surfaces canceled rows as error dicts. Appended
            # AFTER the ended-break and overdue checks (precedence load-bearing).
            # ``created_at`` (captured right after batches.create) is the anchor.
            # Gated on ``in_progress`` (#1019 round 2): a chunk already
            # ``canceling`` (an out-of-band Console cancel) never gets a
            # duplicate cancel from us — the loop just polls it to ``ended``.
            if (
                not stuck_canceled
                and batch.processing_status == "in_progress"  # never re-cancel an o-o-b cancel
                and (stuck_hours := batch_stuck_threshold_hours()) is not None
                and counts is not None
                and counts.succeeded == 0
                and (now_fn() - created_at).total_seconds() / 3600.0 >= stuck_hours
            ):
                import anthropic as anthropic_mod

                logger.warning(
                    "STUCK BATCH (legacy path): %s at succeeded=0 past %.1fh — canceling; "
                    "canceled rows surface as error dicts (no retry machinery on this "
                    "path) (#810 escape)",
                    batch_id,
                    stuck_hours,
                )
                try:
                    client.messages.batches.cancel(batch_id)
                except anthropic_mod.APIStatusError:
                    # Race: the batch may already be canceling (out-of-band) or
                    # ended. BOTH are accepted cancellation outcomes — the loop
                    # harvests at ended either way. Anything else re-raises.
                    # #1035: the confirm probe tolerates a transient 404 too
                    # (same crash class as the poll retrieves).
                    final = retrieve_with_404_tolerance(
                        functools.partial(client.messages.batches.retrieve, batch_id),
                        created_at=created_at,
                        batch_id=batch_id,
                        now_fn=now_fn,
                        sleep_fn=sleep_fn,
                    )
                    if final.processing_status not in ("canceling", "ended"):
                        raise
                    logger.info(
                        "Cancel of %s raced with state %s; accepted as canceled",
                        batch_id,
                        final.processing_status,
                    )
                stuck_canceled = True
            sleep_fn(current_interval)
            current_interval = min(current_interval * 1.5, max_poll_interval)

        _collect_legacy_results(client, batch_id, results)

    return results


# ── Cache check + item enumeration ──────────────────────────────────────────


def _enumerate_and_check_cache(
    completions: dict[str, dict[str, list[str]]],
    cache: JudgeCache | None,
    format_user_msg: Callable[[str, str], str],
    *,
    rubric_key: str,
) -> tuple[int, dict[str, dict], list[tuple[str, str, str, str]]]:
    """Enumerate all (persona, question, completion) tuples, checking cache.

    ``rubric_key`` (required) is the rubric/judge identity hash every cache
    lookup is keyed under — see :func:`rubric_fingerprint` (rule 22, #810).

    Returns:
        (total_count, cached_scores, uncached_items)
    """
    cached_scores: dict[str, dict] = {}
    uncached_items: list[tuple[str, str, str, str]] = []
    total = 0
    idx = 0

    for persona_name, q_completions in completions.items():
        for question, comps in q_completions.items():
            for comp_idx, comp in enumerate(comps):
                custom_id = f"{persona_name}__{idx:05d}__{comp_idx:02d}"
                if len(custom_id) > 64:
                    raise ValueError(
                        "Batch API custom_id exceeds the 64-char limit "
                        f"({len(custom_id)} chars): {custom_id!r}. Shorten the "
                        "caller's persona/item id (the encoder appends 11 chars)."
                    )
                total += 1

                if cache:
                    cached = cache.get(question, comp, rubric_key=rubric_key)
                    if cached is not None and not is_transport_error_dict(cached):
                        # A stored transport-class error dict is a MISS (#1313,
                        # rule 24(ii)): fall through to re-dispatch so a re-run
                        # self-heals a legacy-poisoned cache (e.g. #1090's
                        # stored 529 rows) instead of re-serving the outage.
                        cached_scores[custom_id] = cached
                        continue

                user_msg = format_user_msg(question, comp)
                uncached_items.append((custom_id, question, comp, user_msg))
            idx += 1

    return total, cached_scores, uncached_items


def _aggregate_persona_scores(
    completions: dict[str, dict[str, list[str]]],
    all_scores: dict[str, dict],
) -> dict[str, dict]:
    """Map scores back to personas and compute per-persona aggregates.

    The mean sums run over rows where BOTH ``aligned`` and ``coherent`` are
    numeric. Betley dual-rubric judges legitimately return the literal
    sentinel strings ``"REFUSAL"``/``"CODE"`` in place of the numeric
    ``aligned`` score (#545 round 16; see
    ``experiments/issue503/judges.py::_betley_sentinel``) — those rows are
    valid verdicts (no ``error`` flag, so they still count in ``n_samples``)
    but must not enter ``sum()``. Authoritative sentinel tallies live with
    the Betley callers, which re-tally from ``all_scores``.

    A parsed score need not be a dict: ``parse_judge_json`` returns
    ``json.loads(text)`` verbatim, so a scalar-shaped rubric (a bare integer, or
    a persona-vectors ``{"score": N}`` that the judge answered as ``"85"``) can
    parse to an ``int``. Non-dict entries are treated as invalid here (they carry
    no ``aligned``/``coherent`` fields to aggregate); the scalar-rubric caller
    reduces the raw scores itself from ``all_scores`` and ignores this return.

    Returns:
        {persona: {mean_aligned, mean_coherent, n_samples, n_errors}}
    """
    # Rebuild the mapping from custom_id -> persona. Entries may be non-dict
    # (bare-int/scalar-rubric parses) as well as the Betley dict schema.
    persona_scores: dict[str, list[object]] = {p: [] for p in completions}
    idx = 0
    for persona_name, q_completions in completions.items():
        for _question, comps in q_completions.items():
            for comp_idx in range(len(comps)):
                custom_id = f"{persona_name}__{idx:05d}__{comp_idx:02d}"
                score = all_scores.get(
                    custom_id,
                    {"aligned": None, "coherent": None, "error": True},
                )
                persona_scores[persona_name].append(score)
            idx += 1

    results: dict[str, dict] = {}
    for persona_name, scores in persona_scores.items():
        # `isinstance(s, dict)` guard: parse_judge_json returns json.loads(text)
        # verbatim, so a scalar-shaped judge rubric (persona-vectors {"score": N}
        # OR a bare integer scale point per llm-judging.md rule 6) can parse to a
        # bare int, which has no .get(). Such entries cannot be Betley-aggregated
        # here — the scalar-rubric caller (e.g. eval.graded_judge.judge_graded) does its
        # OWN reduction from all_scores via save_raw and ignores this return — so
        # they are treated as invalid at THIS aggregator. Betley callers always
        # emit dicts, so behavior is unchanged for them. Fixes #778 r2.
        valid = [
            s
            for s in scores
            if isinstance(s, dict) and s.get("aligned") is not None and not s.get("error")
        ]
        # #545 round 16: drop rows without both fields numeric from the mean
        # sums (Betley sentinel strings "REFUSAL"/"CODE" survive the valid
        # filter — no error flag — and a str in sum() is a TypeError). For
        # numeric-only data scored == valid, so behavior is unchanged.
        scored = [
            s
            for s in valid
            if isinstance(s.get("aligned"), int | float)
            and isinstance(s.get("coherent"), int | float)
        ]
        if len(scored) != len(valid):
            logger.debug(
                "_aggregate_persona_scores(%s): excluded %d non-numeric (e.g. Betley "
                "sentinel) row(s) from mean sums; callers re-tally from all_scores",
                persona_name,
                len(valid) - len(scored),
            )
        aligned_vals = [s["aligned"] for s in scored]
        coherent_vals = [s["coherent"] for s in scored]
        results[persona_name] = {
            "mean_aligned": sum(aligned_vals) / len(aligned_vals) if aligned_vals else None,
            "mean_coherent": sum(coherent_vals) / len(coherent_vals) if coherent_vals else None,
            "n_samples": len(valid),
            "n_errors": len(scores) - len(valid),
        }

    return results


# ── Main entry point ─────────────────────────────────────────────────────────


def judge_completions_batch(
    completions: dict[str, dict[str, list[str]]],
    judge_system_prompt: str = JUDGE_SYSTEM_PROMPT,
    format_user_msg: Callable[[str, str], str] | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 256,
    poll_interval: float = 30.0,
    cache_dir: Path | None = None,
    save_raw: Path | None = None,
    threshold_base: int = 2_000,
    force_sync: bool = False,
    dry_run: bool = False,
    checkpoint_dir: Path | None = None,
    max_concurrent: int = 50,
    sync_client=None,
    batch_client=None,
) -> dict[str, dict]:
    """Judge all completions via the batch-aware dispatcher with optional caching.

    Workflow:
    1. Check cache for each (question, completion) pair under the derived
       rubric/judge identity key (rule 22 — see :func:`rubric_fingerprint`)
    2. Dispatch uncached pairs through judge_dispatch (sync below the
       tier-scaled threshold, Message Batches at/above it, in <=8k sub-batches)
    3. Parse results, update cache
    4. Aggregate per persona

    Args:
        completions: {persona: {question: [completions]}}
        judge_system_prompt: System prompt for the judge model.
        format_user_msg: Callable(question, completion) -> user message string.
            Defaults to the standard alignment evaluation format.
        judge_model: Claude model to use as judge.
        max_tokens: Maximum tokens for judge response.
        poll_interval: Initial polling interval in seconds (batch path).
        cache_dir: Directory for file-based judge cache. None disables caching.
        save_raw: If provided, save all raw scores to this path as JSON
            (includes a "routing" key with the dispatch decision).
        threshold_base: Sync/batch routing threshold at Tier-4 OTPM (scaled
            by the probed OTPM limit; see judge_dispatch.decide_route).
        force_sync: Bypass routing and judge synchronously regardless of N.
        dry_run: Print the cache split + routing decision and return {} with
            zero API calls.
        checkpoint_dir: Batch-path checkpoint root. When None, derived as
            cache_dir/.dispatch/ if cache_dir is set, else
            save_raw.parent/.judge_dispatch/ if save_raw is set; the batch
            path raises if neither is available.
        max_concurrent: Sync-path concurrency bound.
        sync_client / batch_client: anthropic client injection (tests only).

    Returns:
        {persona: {mean_aligned, mean_coherent, n_samples, n_errors}}
    """
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    if format_user_msg is None:
        format_user_msg = _default_format_user_msg

    cache = JudgeCache(cache_dir) if cache_dir else None
    # Rubric/judge identity for every cache read/write in this call (rule 22,
    # #810): derived from the resolved judge model + system prompt + user-msg
    # template, so callers need no new parameter.
    rubric_key = rubric_fingerprint(judge_model, judge_system_prompt, format_user_msg)

    # Phase 1: Check cache, build list of uncached items
    total, cached_scores, uncached_items = _enumerate_and_check_cache(
        completions, cache, format_user_msg, rubric_key=rubric_key
    )
    n_cached = len(cached_scores)
    n_to_submit = len(uncached_items)
    logger.info(
        "Judge batch: %d total, %d cached, %d to submit",
        total,
        n_cached,
        n_to_submit,
    )
    if dry_run:
        print(f"total={total} | cached={n_cached} to_submit={n_to_submit}")

    # Derive the batch-path checkpoint root (the dispatcher raises if the
    # batch path is selected and this is still None).
    if checkpoint_dir is None:
        if cache_dir is not None:
            checkpoint_dir = Path(cache_dir) / ".dispatch"
        elif save_raw is not None:
            checkpoint_dir = Path(save_raw).parent / ".judge_dispatch"

    # Phase 2: dispatch uncached items (routing + sync/batch execution)
    batch_scores: dict[str, dict] = {}
    decisions: list = []
    if uncached_items or dry_run:
        batch_scores = dispatch_judge_items(
            uncached_items,
            judge_model=judge_model,
            judge_system_prompt=judge_system_prompt,
            max_tokens=max_tokens,
            threshold_base=threshold_base,
            force_sync=force_sync,
            dry_run=dry_run,
            max_concurrent=max_concurrent,
            checkpoint_dir=checkpoint_dir,
            poll_interval=poll_interval,
            on_decision=decisions.append,
            sync_client=sync_client,
            batch_client=batch_client,
        )
        if dry_run:
            return {}

        # Update cache with new results
        if cache:
            for custom_id, question, comp, _user_msg in uncached_items:
                if custom_id in batch_scores:
                    result = batch_scores[custom_id]
                    if is_transport_error_dict(result):
                        # rule 24(ii)/23 (#1313): never cache a transport error —
                        # a cached one re-serves the outage on every resume.
                        continue
                    cache.put(question, comp, result, rubric_key=rubric_key)

    if cache:
        logger.info("Cache stats: %s", cache.stats)

    # Phase 3: Merge cached + batch results, aggregate per persona
    all_scores: dict[str, dict] = {**cached_scores, **batch_scores}
    results = _aggregate_persona_scores(completions, all_scores)

    # Save raw scores if requested
    if save_raw:
        from dataclasses import asdict as _asdict

        save_raw = Path(save_raw)
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        with open(save_raw, "w") as f:
            json.dump(
                {
                    "per_persona": results,
                    "all_scores": all_scores,
                    "cache_stats": cache.stats if cache else None,
                    "judge_model": judge_model,
                    "n_total": total,
                    "n_cached": n_cached,
                    "n_submitted": n_to_submit,
                    "routing": _asdict(decisions[0]) if decisions else None,
                },
                f,
                indent=2,
            )
        logger.info("Saved raw scores to %s", save_raw)

    return results
