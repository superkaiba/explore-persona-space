"""Batch-aware dispatch layer for Claude-judge scoring (task #626).

One routing layer for all Claude-judge calls in the eval path. Routes by
request count, optimizing SPEED:

- ``N < effective_threshold`` -> synchronous ``AsyncAnthropic`` calls at high
  concurrency (per-item error dicts, SDK 429 backoff via ``max_retries=5``).
- ``N >= effective_threshold`` -> Anthropic Message Batches API, split into
  sub-batches of <= ``sub_batch_size`` requests (also bounded by the 256 MB
  byte cap via :func:`batch_judge._chunk_requests`). Sub-batches end
  independently and are harvested in waves.

``effective_threshold = max(1, int(threshold_base * otpm / 400_000))`` where
``otpm`` is read once from the ``anthropic-ratelimit-output-tokens-limit``
response header (:func:`probe_otpm_limit`); the probe is skipped (Tier-4
400k assumed) when N is far from any possible boundary, when ``force_sync``
is set, and always in dry-run.

Usage (sync entry points, e.g. ``judge_completions_batch``)::

    results = dispatch_judge_items(items, checkpoint_dir=ckpt_dir)

Usage (async entry points, e.g. ``evaluate_alignment`` /
``evaluate_strongreject`` — coroutines already running under an event
loop MUST await the async core, never the sync wrapper)::

    results = await dispatch_judge_items_async(items, checkpoint_dir=ckpt_dir)

Both return ``{custom_id: parsed_score_dict}`` with the same per-item
error-dict convention as the legacy paths (``{"aligned": None, ...,
"error": True}`` by default; pass ``error_dict_factory`` for a different
shape, e.g. strongreject's ``{"refused": None, "quality": None, ...}``).

Crash safety (batch path only): every dispatch checkpoints under
``checkpoint_dir/dispatch_<fingerprint12>/`` — ``items.json`` (full
custom_id -> {question, completion, user_msg} mapping) plus ``state.json``
(atomic tmp+rename writes; a ``submitting`` intent record lands BEFORE
``batches.create`` and the ``batch_id`` immediately after). Re-running the
same dispatch resumes by polling the recorded batches; it never re-creates
them. The fingerprint is CONTENT+CONFIG bound (per-item content hashes +
judge_model + sha256(rubric) + max_tokens), so same-shape/different-content
dispatches land in different checkpoint dirs by construction; resume
additionally verifies ``items.json`` equality and fails loud on any
mismatch or unrecorded create.

The errored/expired RETRY is resumable at every stage (``retry.status``:
``pending`` -> ``submitting`` -> ``done``/``results_merged``). A crash at
any point after retry API calls succeed never re-calls those items on
resume: sync-routed retries persist per-item results incrementally to
``results_retry_partial.json`` (only genuinely unfinished items are
re-dispatched); batch-routed retries resume through their nested
``retry/dispatch_<fp>/`` checkpoint (``retry.routed_path`` pins the route
across resumes so a threshold flip cannot strand an in-flight nested
batch or a partial file); and a complete-but-unmerged
``results_retry.json`` (crash between its atomic write and the state
flip) is merged with zero API calls. A sub-batch still ``in_progress`` and
wedged at zero succeeded requests past ``EPS_BATCH_STUCK_HOURS`` (default
4h; ``<=0`` disables) is CANCELED (escape intent persisted BEFORE the
cancel side effect), its partial results harvested at ``ended``, and its
``canceled`` ids re-dispatched through the retry machinery pre-pinned to
the SYNC path (#1019, incident #810). A batch already ``canceling`` with no
persisted intent (an operator's out-of-band Console cancel) is never
adopted by the escape — its canceled rows keep the #663 error-dict
no-retry contract.

NOTE: ``_build_params`` is underscore-private by convention but imported
by ``explore_persona_space.eval.alignment`` (single source of truth for
judge request construction) — keep its signature stable or update that
call site in lockstep.

Caching honesty: every request carries ``cache_control`` (1h TTL inside
batches, 5m default on sync), but all current judge rubrics are ~120-400
estimated tokens — below Sonnet 4.5's 1,024-token cacheable minimum, where
``cache_control`` is a silent no-op. A warning is logged once per dispatch
when the rubric is below the minimum; no cache savings are claimed for
current rubrics. The plumbing engages automatically for future >=1,024-token
rubrics.

Dry-run CLI (zero API calls; OTPM from ``EPM_JUDGE_OTPM`` or 400k assumed)::

    uv run python -m explore_persona_space.eval.judge_dispatch --n 4400 \
        [--otpm 90000] [--force-sync]
"""

import argparse
import asyncio
import contextlib
import contextvars
import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.utils import parse_judge_json
from explore_persona_space.llm.anthropic_client import (
    BatchDeadlineExceeded,
    batch_stuck_threshold_hours,
    deadline_from_expires_at,
    parse_batch_submitted_at,
    retrieve_with_404_tolerance,
)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Explicit request temperature, set by :func:`graded_temperature` for a scoped
# dispatch and read by :func:`_build_params`. Default None → the key is OMITTED
# from the request (every legacy caller keeps the Anthropic API default,
# behavior-preserving). A caller needing N INDEPENDENT graded draws (issue763's
# N=8 protocol) sets 1.0 so the request is deterministic against a future change
# in the API's default temperature — the graded-draw protocol treats temp=1.0 as
# load-bearing + reproducible, not merely "whatever the default is". A ContextVar
# (not a module global) so it is coroutine-local: the async dispatch core fans out
# concurrent sync calls, and a plain global would leak one dispatch's temperature
# into a concurrent dispatch on the same event loop.
_REQUEST_TEMPERATURE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "judge_request_temperature", default=None
)


@contextlib.contextmanager
def graded_temperature(temperature: float | None) -> Iterator[None]:
    """Scope an EXPLICIT judge-request ``temperature`` over a dispatch block.

    Every ``dispatch_judge_items`` / ``dispatch_judge_items_async`` call made
    inside the ``with`` block builds its Messages-API requests (sync AND batch
    paths, via :func:`_build_params`) with an explicit ``"temperature"``; outside
    the block the key is omitted and the API default applies. ``None`` is a no-op
    (restores the omit-key default). Reset is guaranteed on exit even on error.

    Usage (issue763 N=8 graded draws)::

        with graded_temperature(1.0):
            scores = dispatch_judge_items(items, checkpoint_dir=ckpt)
    """
    token = _REQUEST_TEMPERATURE.set(temperature)
    try:
        yield
    finally:
        _REQUEST_TEMPERATURE.reset(token)


# Opt-in raw-response retention, mirroring the _REQUEST_TEMPERATURE pattern
# (coroutine-local ContextVar, default OFF → zero behavior change for every
# legacy caller). When set, each SUCCESSFULLY PARSED result dict additionally
# carries the verbatim judge response text under ``_RAW_TEXT_KEY``. Motivation
# (issue763 `deception-rubric-reanchor` plumbing delta 2): the graded N-draw
# protocol must PERSIST per-draw raw judge text (reason-then-score
# justifications) to raw_completions shards — ``parse_judge_json`` alone
# discards everything outside the first JSON object, so without this the raw
# text is unrecoverable post-dispatch. Error dicts are unchanged (they already
# carry the terminal reason). The raw text flows into the batch-path result
# checkpoints too (results_*.json) — deliberate: a resumed harvest keeps the
# same shape.
_RAW_TEXT_KEY = "_raw_text"
_KEEP_RAW_TEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "judge_keep_raw_text", default=False
)


@contextlib.contextmanager
def keep_raw_judge_text() -> Iterator[None]:
    """Scope raw judge-response retention over a dispatch block.

    Inside the ``with`` block every parsed result dict carries the verbatim
    response text under ``"_raw_text"`` (sync, multi-org sync, and batch-harvest
    paths alike). Reset is guaranteed on exit even on error.
    """
    token = _KEEP_RAW_TEXT.set(True)
    try:
        yield
    finally:
        _KEEP_RAW_TEXT.reset(token)


def _parsed_with_raw(parsed: dict | None, text: str) -> dict | None:
    """Attach the raw response text to a parsed result when retention is on.

    Copies the parsed dict before annotating (never mutates a caller-visible
    object); a ``None`` parse (→ the caller's error dict) passes through
    unchanged so error-dict shapes stay identical.
    """
    if parsed is None or not _KEEP_RAW_TEXT.get():
        return parsed
    out = dict(parsed)
    out[_RAW_TEXT_KEY] = text
    return out


# Item: same 4-tuple already used by batch_judge.
JudgeItem = tuple[str, str, str, str]  # (custom_id, question, completion, user_msg)

# Routing constants (user-decided, plan §11; configurable per call).
DEFAULT_THRESHOLD_BASE = 2_000
# Judge sub-batches default to the JUDGE shard ceiling (2_000), NOT the general
# 8_000 batch cap. An 8k judge batch STARVES — the API schedules a large judge
# set behind everyone else's traffic (the #658 G1 wedge: one 8k judge shard sat
# at succeeded:0 for ~9h), while ~500-2k judge shards clear in minutes. EVERY
# caller through this router (judge_completions_batch / dispatch_judge_items —
# e.g. #664's issue664_dispatch.py) inherits this; only explicit non-judge batch
# callers keep batch_judge.MAX_REQUESTS_PER_BATCH (8k). Bound to the ceiling
# constant (single source of truth) so the default can never drift back to 8k.
# 2_000 == batch_judge.MAX_JUDGE_REQUESTS_PER_BATCH (kept a literal here to avoid
# a top-level batch_judge import — that closes a judge_dispatch -> batch_judge ->
# alignment -> judge_dispatch cycle; batch_judge is imported lazily elsewhere).
# test_judge_dispatch.test_judge_router_default_is_the_judge_ceiling locks the two
# equal so the default can never drift back to 8k.
DEFAULT_SUB_BATCH_SIZE = 2_000  # judge shard ceiling (#658); see note above
OTPM_DIVISOR = 400_000  # Tier-4 Sonnet 4.x output-tokens-per-minute
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MAX_TOKENS = 256
# At most this many sub-batches submitted/in-flight concurrently. Phase-3
# guidelines (docs/api_throughput_guidelines.md §3) tightened this 8 -> 4 to
# match the multi-org dispatcher and leave headroom on the SHARED org keys:
# with sub_batch_size=2_000 and 4 in-flight, max in-flight is 8k (~1.6% of the
# Tier-4 floor queue cap 500k), still ample under any queue cap while reducing
# the create-burst footprint on a shared bucket. The bound is etiquette, not
# the binding constraint.
MAX_CONCURRENT_SUB_BATCHES = 4
# Poll-deadline grace past each batch's expires_at before raising (#663 §11).
BATCH_DEADLINE_GRACE_MIN = 30
# Sonnet 4.5 minimum cacheable prefix (tokens); below it cache_control is a
# silent no-op (docs verified 2026-06-12).
CACHE_MIN_TOKENS = 1024
OTPM_HEADER = "anthropic-ratelimit-output-tokens-limit"

_RECONCILE_MSG = (
    "Checkpoint records a 'submitting' intent without a batch_id for sub-batch {index} at {path} "
    "— a batch may have been created but not recorded (crash inside batches.create). "
    "Check client.messages.batches.list(limit=10) for a batch of {n} requests created near {ts}; "
    "then either record its id in state.json (sub_batches[{index}].batch_id, status='submitted') "
    "or delete the dispatch dir to deliberately resubmit. Refusing to silently resubmit."
)


def _default_error_dict(reason: str) -> dict:
    """Alignment-shaped per-item error dict (the legacy convention)."""
    return {"aligned": None, "coherent": None, "reasoning": reason, "error": True}


def _default_judge_system_prompt() -> str:
    """Lazy import of the alignment rubric (avoids a circular module import)."""
    from explore_persona_space.eval.alignment import JUDGE_SYSTEM_PROMPT

    return JUDGE_SYSTEM_PROMPT


def _build_params(
    judge_model: str,
    judge_system_prompt: str,
    user_msg: str,
    max_tokens: int,
    *,
    ttl: str | None,
    temperature: float | None = None,
) -> dict:
    """Build Messages-API params with cache_control on the shared rubric block.

    ttl="1h" for batch requests (out-of-order execution outlives the 5m
    default); ttl=None or "5m" -> ephemeral default (5m). Returns the kwargs
    dict for ``messages.create`` / the ``params`` member of a batch Request.

    ``temperature``: an EXPLICIT request temperature (reproducibility — the
    Anthropic API default is currently 1.0 but is not contractually pinned; a
    caller needing N independent graded draws, e.g. issue763's N=8 protocol,
    sets ``1.0`` so the request is deterministic against a future API-default
    change). When left None, the value falls back to the coroutine-local
    :data:`_REQUEST_TEMPERATURE` context (set by :func:`graded_temperature`); if
    THAT is also None, the ``"temperature"`` key is OMITTED and the API default
    applies — behavior-preserving for every legacy caller.
    """
    if temperature is None:
        temperature = _REQUEST_TEMPERATURE.get()
    sys_block: dict = {"type": "text", "text": judge_system_prompt}
    if ttl is not None and ttl != "5m":
        sys_block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    else:
        sys_block["cache_control"] = {"type": "ephemeral"}
    params = {
        "model": judge_model,
        "max_tokens": max_tokens,
        "system": [sys_block],
        "messages": [{"role": "user", "content": user_msg}],
    }
    if temperature is not None:
        params["temperature"] = temperature
    return params


@dataclass
class RoutingDecision:
    """The routing outcome for one dispatch; ``render()`` is the dry-run printout."""

    n_items: int
    threshold_base: int
    otpm: int | None  # probed/env value, or None (assume Tier-4 default)
    otpm_assumed: bool  # True unless the value came from a live header probe
    effective_threshold: int
    path: str  # "sync" | "batch"
    forced_sync: bool
    sub_batch_sizes: list[int] = field(default_factory=list)  # [] for sync

    def render(self) -> str:
        """Human-readable one-per-dispatch routing summary."""
        otpm_display = self.otpm if self.otpm is not None else OTPM_DIVISOR
        otpm_src = "assumed" if self.otpm_assumed else "probed"
        line = (
            f"N={self.n_items} | threshold: base={self.threshold_base}, "
            f"otpm={otpm_display} ({otpm_src}), effective={self.effective_threshold} | "
            f"path={self.path}"
        )
        if self.forced_sync:
            line += " (forced)"
        if self.path == "batch":
            line += f" | sub-batches: {self.sub_batch_sizes}"
        return line


def decide_route(
    n_items: int,
    *,
    threshold_base: int = DEFAULT_THRESHOLD_BASE,
    otpm: int | None = None,
    force_sync: bool = False,
    sub_batch_size: int = DEFAULT_SUB_BATCH_SIZE,
    otpm_assumed: bool | None = None,
) -> RoutingDecision:
    """Pure routing rule: sync below the tier-scaled threshold, batch at/above.

    ``otpm=None`` assumes the Tier-4 divisor (400k). ``otpm_assumed``
    overrides the provenance flag (dry-run passes an env/default value that
    is still 'assumed', not probed).
    """
    if otpm_assumed is None:
        otpm_assumed = otpm is None
    effective_otpm = OTPM_DIVISOR if otpm is None else otpm
    effective_threshold = max(1, int(threshold_base * effective_otpm / OTPM_DIVISOR))
    if force_sync or n_items < effective_threshold:
        path = "sync"
        sub_batch_sizes: list[int] = []
    else:
        path = "batch"
        full, rem = divmod(n_items, sub_batch_size)
        sub_batch_sizes = [sub_batch_size] * full + ([rem] if rem else [])
    return RoutingDecision(
        n_items=n_items,
        threshold_base=threshold_base,
        otpm=otpm,
        otpm_assumed=otpm_assumed,
        effective_threshold=effective_threshold,
        path=path,
        forced_sync=force_sync,
        sub_batch_sizes=sub_batch_sizes,
    )


def probe_otpm_limit(client: "anthropic.Anthropic", judge_model: str) -> int | None:
    """Read the org OTPM limit from one max_tokens=1 probe response header.

    Returns the integer ``anthropic-ratelimit-output-tokens-limit`` value, or
    None (with a warning) when the header is missing or malformed — the
    caller then assumes the Tier-4 default. API errors propagate (a dispatch
    that cannot reach the API would fail at its first real request anyway).
    """
    raw = client.messages.with_raw_response.create(
        model=judge_model,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )
    value = raw.headers.get(OTPM_HEADER)
    if value is None:
        logger.warning("OTPM probe: header %s missing; assuming %d", OTPM_HEADER, OTPM_DIVISOR)
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "OTPM probe: header %s malformed (%r); assuming %d", OTPM_HEADER, value, OTPM_DIVISOR
        )
        return None


# ── Checkpoint helpers (batch path) ──────────────────────────────────────────


def _atomic_write_json(path: Path, obj) -> None:
    """Atomic JSON write: tmp file in the same dir + os.replace."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _compute_fingerprint(
    items: list[JudgeItem], judge_model: str, judge_system_prompt: str, max_tokens: int
) -> str:
    """Content+config-bound dispatch fingerprint (12 hex chars).

    Binds per-item CONTENT hashes (question/completion/user_msg — never
    custom_ids alone, which are purely positional in both migrated
    enumerators) AND the judge config (model, rubric sha256, max_tokens), so
    same-shape/different-content or different-config dispatches land in
    different checkpoint dirs by construction.
    """

    def _content_hash(q: str, c: str, u: str) -> str:
        return hashlib.sha256((q + "\n---\n" + c + "\n" + u).encode()).hexdigest()[:16]

    item_lines = sorted(f"{cid}:{_content_hash(q, c, u)}" for cid, q, c, u in items)
    rubric_sha = hashlib.sha256(judge_system_prompt.encode()).hexdigest()
    payload = "\n".join(item_lines).encode() + f"|{judge_model}|{rubric_sha}|{max_tokens}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _load_or_init_state(
    dispatch_dir: Path,
    items_map: dict[str, dict],
    fingerprint: str,
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    sub_batch_size: int,
) -> dict:
    """Load a resumable state.json (verifying fingerprint + items.json) or init a fresh one.

    Fresh init writes items.json FIRST, then state.json with every sub-batch
    at status 'pending'. Resume raises on fingerprint mismatch, items.json
    content mismatch, or a corrupt state file — never silently restarts.
    """
    from explore_persona_space.eval.batch_judge import _chunk_requests

    state_path = dispatch_dir / "state.json"
    items_path = dispatch_dir / "items.json"

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Corrupt checkpoint state at {state_path} ({e}); refusing a silent restart. "
                "Inspect or remove the dispatch dir to proceed."
            ) from e
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Checkpoint fingerprint mismatch at {state_path}: recorded "
                f"{state.get('fingerprint')!r} != computed {fingerprint!r}. The dispatch dir "
                "holds a DIFFERENT item set or judge config; refusing to mix results."
            )
        if not items_path.exists():
            raise RuntimeError(
                f"Checkpoint at {dispatch_dir} has state.json but no items.json; "
                "refusing to resume without the persisted item mapping."
            )
        existing_items = json.loads(items_path.read_text())
        if existing_items != items_map:
            raise RuntimeError(
                f"items.json content mismatch at {items_path}: the persisted items differ from "
                "the in-memory dispatch items despite a matching fingerprint. Refusing to "
                "serve prior results for different content."
            )
        return state

    dispatch_dir.mkdir(parents=True, exist_ok=True)
    # items.json FIRST (before any state/submission/poll) — it is what makes
    # errored/expired retry and resume possible after a crash.
    _atomic_write_json(items_path, items_map)
    # Sub-batch plan via the byte-cap-aware chunker (reused from batch_judge).
    requests = [
        {
            "custom_id": cid,
            "params": _build_params(
                judge_model, judge_system_prompt, rec["user_msg"], max_tokens, ttl="1h"
            ),
        }
        for cid, rec in items_map.items()
    ]
    chunks = _chunk_requests(requests, max_count=sub_batch_size)
    state = {
        "version": 1,
        "fingerprint": fingerprint,
        "judge_model": judge_model,
        "judge_system_prompt_sha256": hashlib.sha256(judge_system_prompt.encode()).hexdigest(),
        "max_tokens": max_tokens,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sub_batches": [
            {
                "index": i,
                "n_requests": len(chunk),
                "custom_ids": [r["custom_id"] for r in chunk],
                "status": "pending",
                "batch_id": None,
                "submitted_at": None,
            }
            for i, chunk in enumerate(chunks)
        ],
        "retry": None,
    }
    _atomic_write_json(dispatch_dir / "state.json", state)
    return state


def _collect_batch_results(
    client: "anthropic.Anthropic",
    batch_id: str,
    error_dict_factory: Callable[[str], dict],
) -> tuple[dict[str, dict], list[str], list[str], list[str], list[str]]:
    """Stream one ended batch's results; join on custom_id (order is not guaranteed).

    Returns (scores, retriable_ids, expired_ids, quarantined_ids, canceled_ids).
    Two-level branch per the Anthropic doc example:

      succeeded                                 -> parse + keep
      errored, error.error.type == invalid_request_error
                                                -> QUARANTINE (never retried;
                                                   a malformed request fails
                                                   identically on resubmit)
      errored, other (server error)             -> retriable
      expired                                   -> retriable (never reached the
                                                   model; safe to resubmit)
      canceled                                  -> ``canceled`` list (never
                                                   reached the model, not billed
                                                   — docs-confirmed; same
                                                   resubmit-safety class as
                                                   expired). Whether canceled
                                                   ids actually RETRY is gated
                                                   downstream by the sub-batch's
                                                   ``stuck_cancel_intent_at``
                                                   marker (#1019): only OUR
                                                   stuck-cancel escape retries;
                                                   an out-of-band (Console)
                                                   cancel keeps the #663
                                                   no-retry contract.
      unknown                                   -> error dict, no retry

    errored/expired/quarantined/canceled all get error dicts in ``scores`` too
    (overwritten if a retry later succeeds). Transport-class rows —
    server-errored / expired / canceled / unknown-rtype — additionally carry
    the structural ``transport: True`` flag (rule 24(i), #1313; the
    quarantined 400 does NOT — rule 24(iii)); the flag is classification
    only and never changes the retry routing above. The SDK error nesting is
    ``result.result.error.error.type`` (double ``.error``); access is
    getattr-guarded so a shape mismatch fails OPEN (routed to retriable, the
    conservative default that never silently quarantines).
    """
    scores: dict[str, dict] = {}
    retriable: list[str] = []
    expired: list[str] = []
    quarantined: list[str] = []
    canceled: list[str] = []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                "",
            )
            parsed = _parsed_with_raw(parse_judge_json(text), text)
            scores[cid] = parsed if parsed is not None else error_dict_factory("parse_error")
        elif rtype == "errored":
            etype = getattr(
                getattr(getattr(result.result, "error", None), "error", None), "type", None
            )
            if etype == "invalid_request_error":
                # Quarantined 400: a pipeline bug, NOT transport (rule 24(iii)).
                quarantined.append(cid)
                scores[cid] = error_dict_factory("batch_error: invalid_request_error (quarantined)")
            else:
                retriable.append(cid)
                scores[cid] = {
                    **error_dict_factory(f"batch_error: errored ({etype or 'server'})"),
                    "transport": True,  # rule 24(i) server-class row (#1313)
                }
        elif rtype == "expired":
            expired.append(cid)
            scores[cid] = {
                **error_dict_factory("batch_error: expired"),
                "transport": True,  # rule 24(i) (#1313)
            }
        elif rtype == "canceled":
            canceled.append(cid)
            scores[cid] = {
                **error_dict_factory("batch_error: canceled"),
                "transport": True,  # rule 24(i) (#1313); retry gating unchanged (#663/#1019)
            }
        else:  # unknown: surface, never retry; fail toward re-judgeable (#1313 §11-7)
            scores[cid] = {
                **error_dict_factory(f"batch_error: {rtype}"),
                "transport": True,
            }
    return scores, retriable, expired, quarantined, canceled


# ── Sync path ────────────────────────────────────────────────────────────────


async def _judge_items_sync(
    items: list[JudgeItem],
    *,
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    max_concurrent: int,
    error_dict_factory: Callable[[str], dict],
    client: "anthropic.AsyncAnthropic",
    on_item_result: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """Semaphore-bounded AsyncAnthropic judging; per-item error dicts, never raises per item.

    ``on_item_result(custom_id, score_dict)`` fires synchronously the moment
    each item's result is known (success, parse_error, or captured-exception
    error dict alike) — the retry path uses it to persist per-item results
    incrementally so a crash mid-dispatch never re-calls finished items.
    """
    # FUNCTION-level import: a module-level eval->llm import is cyclic
    # (api_dispatch imports eval.batch_judge -> eval.alignment ->
    # eval.judge_dispatch); precedent: _judge_items_sync_multiorg / decide_route.
    from explore_persona_space.llm.api_dispatch import is_transport_exception

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(custom_id: str, user_msg: str) -> tuple[str, dict]:
        async with semaphore:
            try:
                params = _build_params(
                    judge_model, judge_system_prompt, user_msg, max_tokens, ttl="5m"
                )
                result = await client.messages.create(**params)
                text = next((b.text for b in result.content if b.type == "text"), "")
                parsed = _parsed_with_raw(parse_judge_json(text), text)
                score = parsed if parsed is not None else error_dict_factory("parse_error")
            except Exception as e:  # per-item capture is the legacy contract
                base = error_dict_factory(f"error: {e}")
                # rule 24(i) (#1313): a transport-class exception (429/5xx incl.
                # 529/timeout/connection) is flagged so downstream tallies split
                # transport losses from content drops.
                score = {**base, "transport": True} if is_transport_exception(e) else base
        if on_item_result is not None:
            on_item_result(custom_id, score)
        return custom_id, score

    results: dict[str, dict] = {}
    pending = items
    # Cache-warm ordering: only worth it when the rubric can actually be
    # cached (>= 1024 est. tokens); below the minimum it is pure added latency.
    if len(items) > 1 and len(judge_system_prompt) // 4 >= CACHE_MIN_TOKENS:
        first_cid, first_msg = items[0][0], items[0][3]
        cid, res = await _judge_one(first_cid, first_msg)
        results[cid] = res
        pending = items[1:]
    pairs = await asyncio.gather(*[_judge_one(cid, msg) for cid, _q, _c, msg in pending])
    results.update(dict(pairs))
    return results


# ── Sync path (Phase 5 multi-org) ────────────────────────────────────────────


async def _judge_items_sync_multiorg(
    items: list[JudgeItem],
    *,
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    error_dict_factory: Callable[[str], dict],
    on_item_result: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """Route the sync judge path through the multi-org api_dispatch (#682 Phase 5).

    Adapts the JudgeItem 4-tuple contract onto the api_dispatch.DispatchItem +
    build_request / parse_response shape. Routes fan-out across the 3 separate
    org keys at the polite per-key concurrency caps (Sonnet 100, see
    docs/api_throughput_guidelines.md), with AIMD back-off on every 429 and
    headroom-aware org selection from the live ``*-remaining`` headers.

    Returns ``{custom_id: parsed_score_dict}`` with the SAME shape the legacy
    single-org sync path returns — a parse failure or terminal error becomes
    an ``error_dict_factory(reason)`` entry, never a missing key.
    """
    from explore_persona_space.llm import api_dispatch

    dispatch_items = [
        api_dispatch.DispatchItem(item_id=cid, payload={"user_msg": user_msg})
        for cid, _q, _c, user_msg in items
    ]

    def _build_request(item: api_dispatch.DispatchItem) -> dict:
        return _build_params(
            judge_model,
            judge_system_prompt,
            item.payload["user_msg"],
            max_tokens,
            ttl="5m",
        )

    def _parse_response(text: str) -> dict:
        parsed = _parsed_with_raw(parse_judge_json(text), text)
        return parsed if parsed is not None else error_dict_factory("parse_error")

    raw_results = await api_dispatch.dispatch_calls(
        dispatch_items,
        model=judge_model,
        build_request=_build_request,
        parse_response=_parse_response,
        cost_pref="latency",  # judge dispatches care about wall-clock
        force_path="sync",  # router only enters this helper after deciding sync
    )

    out: dict[str, dict] = {}
    for cid, _q, _c, _u in items:
        res = raw_results.get(cid)
        if res is None:
            # Rare + visible; stays content-classified (#1313 §10-bis note 5).
            score = error_dict_factory("missing_dispatch_result")
        elif res.error:
            base = error_dict_factory(res.reason or "error")
            # rule 24(i)/(ii) (#1313): RESULT_TRANSPORT (transient exhaustion)
            # and RESULT_RATE_LIMITED (429-budget exhaustion) are both
            # transport-class -> flag so tallies split them from content drops.
            # ``api_dispatch`` is function-level-imported above (cycle constraint).
            transportish = res.category in (
                api_dispatch.RESULT_TRANSPORT,
                api_dispatch.RESULT_RATE_LIMITED,
            )
            score = {**base, "transport": True} if transportish else base
        else:
            score = (
                res.result if isinstance(res.result, dict) else error_dict_factory("parse_error")
            )
        out[cid] = score
        if on_item_result is not None:
            on_item_result(cid, score)
    return out


# ── Batch path ───────────────────────────────────────────────────────────────


async def _submit_one_sub_batch(
    sb: dict,
    *,
    state: dict,
    state_path: Path,
    items_map: dict[str, dict],
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    client: "anthropic.Anthropic",
    sem: asyncio.Semaphore,
    n_sub_batches: int,
) -> None:
    """Submit ONE pending sub-batch under the concurrency semaphore.

    Preserve-before-propagate atomicity (#663 §4b iv): the ``submitting`` intent
    is persisted BEFORE ``batches.create``; the ``batch_id`` + ``deadline`` are
    persisted in a ``finally``-shielded block so a ``CancelledError`` (e.g. an
    external cancel of the whole gather) can NEVER interleave between
    ``batches.create`` returning and ``batch_id`` being recorded — which would
    orphan a paid batch AND leave ``status="submitting", batch_id=None``, the
    exact non-resumable wedge the reconciliation guard exists to catch. The
    caller fans these out via ``gather(return_exceptions=True)`` so a sibling's
    create failure never cancels this task mid-flight (no stranded batch_id);
    the first real exception is re-raised unwrapped post-gather.
    """
    async with sem:
        if sb["status"] != "pending":
            return
        sb["status"] = "submitting"
        _atomic_write_json(state_path, state)  # intent persisted BEFORE create
        requests = [
            {
                "custom_id": cid,
                "params": _build_params(
                    judge_model,
                    judge_system_prompt,
                    items_map[cid]["user_msg"],
                    max_tokens,
                    ttl="1h",
                ),
            }
            for cid in sb["custom_ids"]
        ]
        # On a create-side exception, leave status="submitting" + batch_id=None
        # so the reconciliation guard fires on resume (recoverable). The create
        # itself is OUTSIDE the finally-guarded persist below.
        batch = await asyncio.to_thread(client.messages.batches.create, requests=requests)
        try:
            sb["batch_id"] = batch.id
            sb["status"] = "submitted"
            sb["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            expires_at = getattr(batch, "expires_at", None)
            if expires_at is not None:
                sb["deadline"] = deadline_from_expires_at(
                    expires_at, BATCH_DEADLINE_GRACE_MIN
                ).isoformat()
        finally:
            # Load-bearing: even on a CancelledError injected here, persist
            # whatever state was updated so resume can reconcile (batch_id is
            # now recorded -> resume polls it; or it wasn't -> the guard fires).
            _atomic_write_json(state_path, state)
        logger.info(
            "Sub-batch %d/%d submitted as %s (%d requests)",
            sb["index"] + 1,
            n_sub_batches,
            batch.id,
            sb["n_requests"],
        )


@dataclass
class _BatchCollector:
    """Mutable accumulator threaded through the poll/collect helpers."""

    scores: dict[str, dict] = field(default_factory=dict)
    retry_candidates: list[str] = field(default_factory=list)
    quarantined_ids: list[str] = field(default_factory=list)


def _load_collected_into(acc: _BatchCollector, dispatch_dir: Path, sb: dict) -> None:
    """Merge one collected sub-batch's persisted results into ``acc``.

    ``canceled_ids`` join the retry set ONLY when this sub-batch carries the
    ``stuck_cancel_intent_at`` escape-intent marker (#1019) — i.e. OUR
    stuck-cancel escape canceled it. The gate keys on the INTENT marker (not
    the ``stuck_canceled_at`` confirm stamp): intent is persisted BEFORE the
    external cancel side effect, so it is set in every crash ordering —
    including resume-at-``ended``, where the confirm stamp never landed. An
    out-of-band cancel (a human canceling in the Console — no intent marker)
    keeps the #663 contract: error dict, never retried. ``.get(..., [])``
    keeps pre-existing results files readable.
    """
    payload = json.loads((dispatch_dir / f"results_{sb['batch_id']}.json").read_text())
    acc.scores.update(payload["scores"])
    acc.retry_candidates.extend(payload["retriable_ids"])
    acc.retry_candidates.extend(payload["expired_ids"])
    if sb.get("stuck_cancel_intent_at"):
        acc.retry_candidates.extend(payload.get("canceled_ids", []))
    acc.quarantined_ids.extend(payload.get("quarantined_ids", []))


def _harvest_sub_batch(
    acc: _BatchCollector,
    *,
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    client: "anthropic.Anthropic",
    error_dict_factory: Callable[[str], dict],
    sb: dict,
) -> None:
    """Collect an ended sub-batch, persist its results json, mark it collected."""
    sb_scores, retriable, expired, quarantined, canceled = _collect_batch_results(
        client, sb["batch_id"], error_dict_factory
    )
    _atomic_write_json(
        dispatch_dir / f"results_{sb['batch_id']}.json",
        {
            "scores": sb_scores,
            "retriable_ids": retriable,
            "expired_ids": expired,
            "quarantined_ids": quarantined,
            "canceled_ids": canceled,
        },
    )
    if sb.get("stuck_cancel_intent_at"):
        # Frozen-counts fingerprint (#1019 §12 A5): the escape fired on
        # succeeded==0, so nonzero API-succeeded results harvested here mean
        # request_counts was frozen/stale (a false-positive stuck fire).
        # Count TRUE API-succeeded rows only: unknown-rtype rows sit in no
        # list (plain subtraction would count them as succeeded) but carry
        # "batch_error: " error dicts; a "parse_error" row IS API-succeeded.
        non_succeeded = {*retriable, *expired, *quarantined, *canceled}
        n_succeeded = sum(
            1
            for cid, score in sb_scores.items()
            if cid not in non_succeeded
            and not str(score.get("reasoning", "")).startswith("batch_error: ")
        )
        logger.warning(
            "Stuck-canceled sub-batch %s harvested: n_succeeded=%d n_canceled=%d "
            "(nonzero succeeded after a 0-counts cancel is the frozen-request_counts "
            "false-positive signature)",
            sb["batch_id"],
            n_succeeded,
            len(canceled),
        )
    sb["status"] = "collected"
    _atomic_write_json(state_path, state)
    _load_collected_into(acc, dispatch_dir, sb)


def _cancel_stuck_sub_batch(
    client: "anthropic.Anthropic",
    sb: dict,
    state: dict,
    state_path: Path,
    now_fn: "Callable[[], _dt.datetime]",
    sleep_fn: "Callable[[float], None] | None" = None,
) -> None:
    """Persist escape intent, cancel the stuck sub-batch, confirm (#1019).

    Ordering is load-bearing (#1019 critic round 1):
      1. persist ``stuck_cancel_intent_at`` (the retry DISCRIMINATOR — written
         BEFORE the external side effect, so a crash at ANY later point still
         routes this sub-batch's ``canceled`` ids to the sync retry, even when
         the resume's first retrieve already reads ``ended`` and the normal
         ended-harvest path runs before the stuck check);
      2. ``batches.cancel`` (race guard below);
      3. persist ``stuck_canceled_at`` (cancel-confirmed observability stamp;
         its absence on resume re-drives step 2, which is idempotent-safe
         because the guard accepts ``canceling``/``ended``).

    ``sleep_fn`` is additive + injectable for tests (default ``time.sleep``);
    it drives the #1035 bounded 404 tolerance on the race-confirm retrieve.
    """
    import anthropic as anthropic_mod

    if sb.get("stuck_cancel_intent_at") is None:
        sb["stuck_cancel_intent_at"] = now_fn().strftime("%Y-%m-%dT%H:%M:%SZ")
        _atomic_write_json(state_path, state)  # intent durable BEFORE the cancel
    try:
        client.messages.batches.cancel(sb["batch_id"])
    except anthropic_mod.APIStatusError:
        # Race / re-drive: the batch may already be canceling (a prior cancel —
        # ours after a crash, or an out-of-band Console cancel) or ended. BOTH
        # are accepted cancellation outcomes: the poll loop harvests at ended
        # either way. Anything else re-raises (fail fast, no swallowed faults).
        # #1035: the confirm probe tolerates a transient 404 (bounded, then
        # terminal — same crash class as the poll retrieves).
        final = retrieve_with_404_tolerance(
            functools.partial(client.messages.batches.retrieve, sb["batch_id"]),
            created_at=parse_batch_submitted_at(sb.get("submitted_at")),
            batch_id=sb["batch_id"],
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if final.processing_status not in ("canceling", "ended"):
            raise
        logger.info(
            "Cancel of %s raced with state %s; accepted as canceled",
            sb["batch_id"],
            final.processing_status,
        )
    sb["stuck_canceled_at"] = now_fn().strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_json(state_path, state)


def _poll_one_sub_batch_step(
    acc: _BatchCollector,
    *,
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    client: "anthropic.Anthropic",
    error_dict_factory: Callable[[str], dict],
    now_fn: "Callable[[], _dt.datetime]",
    sb: dict,
    sleep_fn: "Callable[[float], None] | None" = None,
) -> None:
    """Poll one not-yet-collected sub-batch once; harvest on end, raise on overdue.

    Stuck-batch escape (#1019): AFTER the ended-harvest and overdue blocks (their
    precedence is load-bearing), a sub-batch still ``in_progress`` at
    ``request_counts.succeeded == 0`` for >= ``EPS_BATCH_STUCK_HOURS`` since
    ``submitted_at`` is canceled via :func:`_cancel_stuck_sub_batch` (intent-first
    crash-safe ordering); the enclosing poll loop then harvests it when the API
    flips canceling -> ended. A batch already ``canceling`` with NO persisted
    intent (an out-of-band Console cancel) is NEVER adopted — its canceled rows
    surface as error dicts, not retries (#1019 round 2).

    404 tolerance (#995 + #1035): a ``NotFoundError`` on any retrieve here —
    the loop retrieve, the overdue final retrieve, and the cancel-race probe —
    is retried with bounded backoff: the create-grace schedule within
    ``BATCH_CREATE_404_GRACE_S`` of this sub-batch's persisted ``submitted_at``
    (l.636, the #742 crash site), the ``BATCH_MIDPOLL_404_BACKOFF_S`` mid-poll
    schedule anywhere else (incl. a resumed poll with an old/absent
    ``submitted_at``). Past both budgets the 404 re-raises — delayed by
    <=~230s worst case, never masked. This step is sync and already blocks the
    async outer loop on the retrieve; the bounded sleeps are consistent with
    existing behavior. ``sleep_fn`` is additive + injectable for tests
    (default ``time.sleep``).
    """
    batch = retrieve_with_404_tolerance(
        functools.partial(client.messages.batches.retrieve, sb["batch_id"]),
        created_at=parse_batch_submitted_at(sb.get("submitted_at")),
        batch_id=sb["batch_id"],
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    counts = getattr(batch, "request_counts", None)
    if counts is not None:
        logger.info(
            "Batch %s: processing=%s succeeded=%s errored=%s",
            sb["batch_id"],
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
    harvest_kw = dict(
        state=state,
        state_path=state_path,
        dispatch_dir=dispatch_dir,
        client=client,
        error_dict_factory=error_dict_factory,
    )
    if batch.processing_status == "ended":
        _harvest_sub_batch(acc, sb=sb, **harvest_kw)
        return
    # Record the deadline off the first retrieve that exposes expires_at (a
    # resumed sub-batch already has it persisted; never re-derive/extend it).
    if sb.get("deadline") is None and getattr(batch, "expires_at", None) is not None:
        sb["deadline"] = deadline_from_expires_at(
            batch.expires_at, BATCH_DEADLINE_GRACE_MIN
        ).isoformat()
        _atomic_write_json(state_path, state)
    if sb.get("deadline") and now_fn() > _dt.datetime.fromisoformat(sb["deadline"]):
        # Overdue: ONE final fetch to harvest a now-ended batch, else raise.
        # #1035: the final fetch is the last chance to harvest, so a transient
        # 404 gets the bounded mid-poll tolerance; a genuinely-gone batch still
        # re-raises NotFoundError <=~3 min later.
        final = retrieve_with_404_tolerance(
            functools.partial(client.messages.batches.retrieve, sb["batch_id"]),
            created_at=parse_batch_submitted_at(sb.get("submitted_at")),
            batch_id=sb["batch_id"],
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if final.processing_status == "ended":
            _harvest_sub_batch(acc, sb=sb, **harvest_kw)
            return
        raise BatchDeadlineExceeded(sb["batch_id"], sb["deadline"])
    # Stuck-batch escape (#1019, incident #810): zero succeeded after the
    # threshold -> persist escape INTENT, then cancel; partial results are
    # harvested on a later iteration when the API flips canceling -> ended
    # (docs-confirmed); canceled ids re-dispatch SYNC, keyed on the INTENT
    # marker (which survives every crash ordering). Ordering is load-bearing:
    # the ended-harvest and overdue blocks above run FIRST, so a finished
    # batch is never touched and BatchDeadlineExceeded keeps precedence.
    # FRESH ENTRY additionally requires ``processing_status == "in_progress"``
    # (#1019 round 2, concern out-of-band-canceling-retried): a batch already
    # ``canceling`` at the poll — an operator's out-of-band Console cancel —
    # must NOT be adopted by the escape; stamping intent on it would route the
    # operator-canceled ids into the sync retry, violating the out-of-band
    # no-retry contract (see :func:`_load_collected_into`). The escape-intent
    # RESUME leg below deliberately keeps accepting canceling/ended — that leg
    # completes OUR OWN already-committed escape after a crash.
    stuck_hours = batch_stuck_threshold_hours()
    if (
        stuck_hours is not None
        and batch.processing_status == "in_progress"  # never adopt an out-of-band cancel
        and sb.get("stuck_cancel_intent_at") is None  # escape entered at most once
        and counts is not None  # fail-safe: no counts, no verdict
        and counts.succeeded == 0
    ):
        anchor = parse_batch_submitted_at(sb.get("submitted_at") or state.get("created_at"))
        if anchor is not None:
            elapsed_h = (now_fn() - anchor).total_seconds() / 3600.0
            if elapsed_h >= stuck_hours:
                logger.warning(
                    "STUCK BATCH: %s at succeeded=0 after %.1fh (threshold %.1fh; "
                    "processing=%s errored=%s) — canceling; unprocessed items will "
                    "re-dispatch on the SYNC path (#810 escape)",
                    sb["batch_id"],
                    elapsed_h,
                    stuck_hours,
                    counts.processing,
                    counts.errored,
                )
                _cancel_stuck_sub_batch(client, sb, state, state_path, now_fn, sleep_fn)
    # Escape-intent RESUME leg: intent persisted but cancel never confirmed
    # (crash between the intent write and the cancel/confirm writes) ->
    # re-drive the cancel. Bounded: the race guard accepts canceling/ended,
    # so this converges in one step. Deliberately keyed on the persisted
    # INTENT alone — not stuck_hours, not processing_status: an escape
    # committed by the intent write is COMPLETED on resume even when
    # EPS_BATCH_STUCK_HOURS has since been set to 0 (disabling the knob stops
    # NEW escapes; it never strands a half-done one).
    elif sb.get("stuck_cancel_intent_at") is not None and sb.get("stuck_canceled_at") is None:
        _cancel_stuck_sub_batch(client, sb, state, state_path, now_fn, sleep_fn)


async def _poll_and_collect_sub_batches(
    *,
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    client: "anthropic.Anthropic",
    error_dict_factory: Callable[[str], dict],
    poll_interval: float,
    now_fn: "Callable[[], _dt.datetime]",
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Deadline-bounded poll of every submitted sub-batch; harvest on end.

    Returns (scores, retry_candidate_ids, quarantined_ids). Loads any already
    ``collected`` sub-batch from its ``results_<batch_id>.json`` first, then
    polls the rest with the ~60s backoff capped per sub-batch by its own
    ``expires_at`` + grace (persisted as ``sb["deadline"]``). A sub-batch still
    not ``ended`` at its deadline raises :class:`BatchDeadlineExceeded` after one
    final harvest attempt.
    """
    acc = _BatchCollector()
    step_kw = dict(
        state=state,
        state_path=state_path,
        dispatch_dir=dispatch_dir,
        client=client,
        error_dict_factory=error_dict_factory,
        now_fn=now_fn,
    )
    for sb in state["sub_batches"]:
        if sb["status"] == "collected":
            _load_collected_into(acc, dispatch_dir, sb)

    current_interval = poll_interval
    max_poll_interval = max(poll_interval, 120.0)
    while any(sb["status"] != "collected" for sb in state["sub_batches"]):
        for sb in state["sub_batches"]:
            if sb["status"] != "collected":
                _poll_one_sub_batch_step(acc, sb=sb, **step_kw)
        if all(sb["status"] == "collected" for sb in state["sub_batches"]):
            break
        await asyncio.sleep(current_interval)
        current_interval = min(current_interval * 1.5, max_poll_interval)

    return acc.scores, acc.retry_candidates, acc.quarantined_ids


async def _run_batch_path(
    items: list[JudgeItem],
    *,
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    sub_batch_size: int,
    checkpoint_dir: Path,
    poll_interval: float,
    error_dict_factory: Callable[[str], dict],
    client: "anthropic.Anthropic",
    now_fn: "Callable[[], _dt.datetime] | None" = None,
) -> tuple[dict[str, dict], list[str], Path, dict]:
    """Submit/resume + poll + collect all sub-batches for one dispatch.

    Returns (scores, retry_candidate_custom_ids, dispatch_dir, state). All
    waits are ``await asyncio.sleep`` so an enclosing event loop is never
    blocked for the batch's lifetime. The poll is bounded by each sub-batch's
    own ``expires_at`` + grace (persisted as ``sb["deadline"]`` so a resumed
    run does not re-derive and accidentally extend it); a batch still not
    ``ended`` at its deadline raises :class:`BatchDeadlineExceeded` after one
    final harvest attempt. ``now_fn`` is injectable for tests (default
    wall-clock).
    """
    now_fn = now_fn or (lambda: _dt.datetime.now(_dt.UTC))
    fingerprint = _compute_fingerprint(items, judge_model, judge_system_prompt, max_tokens)
    dispatch_dir = Path(checkpoint_dir) / f"dispatch_{fingerprint}"
    items_map = {cid: {"question": q, "completion": c, "user_msg": u} for cid, q, c, u in items}
    assert len(items_map) == len(items), (
        f"duplicate custom_ids in judge items: {len(items) - len(items_map)} collisions "
        "(the custom_id->result join would silently drop rows)"
    )
    state = _load_or_init_state(
        dispatch_dir,
        items_map,
        fingerprint,
        judge_model,
        judge_system_prompt,
        max_tokens,
        sub_batch_size,
    )
    state_path = dispatch_dir / "state.json"

    # Phase 1: submission. Reconcile any crashed-mid-submit sub-batch FIRST
    # (sequential, before any concurrent create), then fan out the still-pending
    # ones under a concurrency bound with the preserve-before-propagate contract.
    for sb in state["sub_batches"]:
        if sb["status"] == "submitting":
            if sb["batch_id"] is None:
                raise RuntimeError(
                    _RECONCILE_MSG.format(
                        index=sb["index"],
                        path=state_path,
                        n=sb["n_requests"],
                        ts=state.get("created_at"),
                    )
                )
            # batch_id recorded but status not advanced (crash between writes)
            sb["status"] = "submitted"
            _atomic_write_json(state_path, state)

    pending_to_submit = [sb for sb in state["sub_batches"] if sb["status"] == "pending"]
    if pending_to_submit:
        sem = asyncio.Semaphore(MAX_CONCURRENT_SUB_BATCHES)
        n_sub_batches = len(state["sub_batches"])
        # gather(return_exceptions=True) + post-gather reconcile (#663 §4b iv):
        # every task runs to completion (no sibling is cancelled mid-flight, so
        # none can be stranded between create-returned and batch_id-persisted by
        # the finally guard), then we re-raise the first real exception
        # UNWRAPPED — preserving the original exception type/message that callers
        # and the existing reconciliation tests match on (a TaskGroup would wrap
        # it in an ExceptionGroup, breaking that contract).
        results = await asyncio.gather(
            *(
                _submit_one_sub_batch(
                    sb,
                    state=state,
                    state_path=state_path,
                    items_map=items_map,
                    judge_model=judge_model,
                    judge_system_prompt=judge_system_prompt,
                    max_tokens=max_tokens,
                    client=client,
                    sem=sem,
                    n_sub_batches=n_sub_batches,
                )
                for sb in pending_to_submit
            ),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, BaseException):
                raise r

    # Phase 2: deadline-bounded polling; harvest each sub-batch the moment it ends.
    scores, retry_candidates, quarantined_ids = await _poll_and_collect_sub_batches(
        state=state,
        state_path=state_path,
        dispatch_dir=dispatch_dir,
        client=client,
        error_dict_factory=error_dict_factory,
        poll_interval=poll_interval,
        now_fn=now_fn,
    )

    if quarantined_ids:
        _atomic_write_json(dispatch_dir / "quarantine.json", sorted(set(quarantined_ids)))
        logger.warning(
            "%d request(s) quarantined (invalid_request_error, NOT retried); see %s",
            len(set(quarantined_ids)),
            dispatch_dir / "quarantine.json",
        )

    return scores, retry_candidates, dispatch_dir, state


async def _run_or_resume_retry(
    *,
    items: list[JudgeItem],
    retry_candidates: list[str],
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    judge_model: str,
    judge_system_prompt: str,
    max_tokens: int,
    threshold_base: int,
    sub_batch_size: int,
    max_concurrent: int,
    poll_interval: float,
    error_dict_factory: Callable[[str], dict],
    sync_client: "anthropic.AsyncAnthropic | None",
    batch_client: "anthropic.Anthropic",
    now_fn: "Callable[[], _dt.datetime] | None" = None,
) -> dict[str, dict]:
    """Run (or crash-resume) the single errored/expired retry; returns its merged results.

    Resumable retry protocol: ``state['retry']['status']`` moves ``pending``
    -> ``submitting`` -> ``done``/``results_merged``. Resuming at ``pending``
    or ``submitting`` never re-calls items whose results are already
    persisted:

    - ``results_retry.json`` present -> completed-but-not-merged (crash
      between its atomic write and the state flip); merge with zero calls.
    - ``results_retry_partial.json`` -> per-item results persisted
      incrementally by a sync-routed retry; only the remainder is
      re-dispatched. Batch-routed retries never write the partial file —
      their nested ``retry/dispatch_<fp>/`` checkpoint dedupes resume, and
      ``retry.routed_path`` pins the route so the remainder set (== the full
      retry set there) re-enters that same checkpoint.

    On success, flips the retry record to ``status='done'`` +
    ``results_merged=True`` and writes ``results_retry.json`` atomically.
    """
    retry_state = state.get("retry")
    retry_done_path = dispatch_dir / "results_retry.json"
    retry_partial_path = dispatch_dir / "results_retry_partial.json"
    items_map = {cid: (q, c, u) for cid, q, c, u in items}
    recomputed_ids = sorted(set(retry_candidates))
    stuck_cancel_present = any(sb.get("stuck_cancel_intent_at") for sb in state["sub_batches"])
    if retry_state is None:
        retry_state = {
            "status": "pending",
            "custom_ids": recomputed_ids,
            "results_merged": False,
            # Stuck-cancel retries are PRE-PINNED to sync (#1019): re-batching a
            # canceled remainder risks re-wedging (#810: sync cleared 39,512
            # calls in 18 min). DELIBERATE breadth: any() pins the WHOLE retry
            # set sync — including errored/expired ids from healthy sibling
            # sub-batches in the same dispatch. Their discount loss is
            # accepted: one retry, one route, deterministic on resume (no
            # per-id route partitioning).
            "routed_path": "sync" if stuck_cancel_present else None,
        }
        state["retry"] = retry_state
        _atomic_write_json(state_path, state)
    elif retry_state["custom_ids"] != recomputed_ids:
        raise RuntimeError(
            f"Retry candidate mismatch at {state_path}: recorded "
            f"{retry_state['custom_ids']} != recomputed {recomputed_ids}. The persisted "
            "sub-batch results no longer reproduce the retry set this checkpoint recorded; "
            "refusing to resume a drifted retry."
        )
    retry_ids = list(retry_state["custom_ids"])

    if retry_done_path.exists():
        # Completed-but-not-merged: the retry dispatch finished and wrote
        # results_retry.json atomically, but crashed before the state flip
        # below. Merge with ZERO re-calls.
        retry_results = json.loads(retry_done_path.read_text())
    else:
        partial: dict[str, dict] = (
            json.loads(retry_partial_path.read_text()) if retry_partial_path.exists() else {}
        )
        remaining: list[JudgeItem] = [
            (cid, *items_map[cid]) for cid in retry_ids if cid not in partial
        ]
        if partial:
            logger.info(
                "Resuming retry: %d/%d item results already persisted; re-dispatching %d",
                len(partial),
                len(retry_ids),
                len(remaining),
            )
        new_results: dict[str, dict] = {}
        if remaining:
            logger.info("Retrying %d errored/expired requests (once)", len(remaining))
            retry_state["status"] = "submitting"
            _atomic_write_json(state_path, state)

            def _persist_retry_item(cid: str, score: dict) -> None:
                partial[cid] = score
                _atomic_write_json(retry_partial_path, partial)

            def _record_retry_route(decision: RoutingDecision) -> None:
                retry_state["routed_path"] = decision.path
                _atomic_write_json(state_path, state)

            # Pin the route recorded by a prior attempt (threshold_base=0
            # forces batch: the effective threshold clamps to 1 and the OTPM
            # probe is skipped) so a threshold/OTPM flip between runs cannot
            # strand an in-flight nested batch or a partial file.
            pinned_route = retry_state.get("routed_path")
            new_results = await dispatch_judge_items_async(
                remaining,
                judge_model=judge_model,
                judge_system_prompt=judge_system_prompt,
                max_tokens=max_tokens,
                threshold_base=0 if pinned_route == "batch" else threshold_base,
                sub_batch_size=sub_batch_size,
                force_sync=pinned_route == "sync",
                dry_run=False,
                max_concurrent=max_concurrent,
                checkpoint_dir=dispatch_dir / "retry",
                poll_interval=poll_interval,
                error_dict_factory=error_dict_factory,
                on_decision=_record_retry_route,
                sync_client=sync_client,
                batch_client=batch_client,
                on_item_result=_persist_retry_item,
                now_fn=now_fn,
                _is_retry=True,
            )
        retry_results = {**partial, **new_results}
        _atomic_write_json(retry_done_path, retry_results)
    state["retry"] = {
        "status": "done",
        "custom_ids": retry_ids,
        "results_merged": True,
        "routed_path": retry_state.get("routed_path"),
    }
    _atomic_write_json(state_path, state)
    return retry_results


# ── Core dispatch ────────────────────────────────────────────────────────────


async def dispatch_judge_items_async(  # noqa: C901  # Phase 5 added one routing branch; refactor pending
    items: list[JudgeItem],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    threshold_base: int = DEFAULT_THRESHOLD_BASE,
    sub_batch_size: int = DEFAULT_SUB_BATCH_SIZE,
    force_sync: bool = False,
    dry_run: bool = False,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    checkpoint_dir: Path | None = None,
    poll_interval: float = 30.0,
    error_dict_factory: Callable[[str], dict] | None = None,
    on_decision: Callable[[RoutingDecision], None] | None = None,
    sync_client: "anthropic.AsyncAnthropic | None" = None,
    batch_client: "anthropic.Anthropic | None" = None,
    on_item_result: Callable[[str, dict], None] | None = None,
    now_fn: "Callable[[], _dt.datetime] | None" = None,
    _is_retry: bool = False,
) -> dict[str, dict]:
    """ASYNC CORE: route + execute one judge dispatch; returns {custom_id: score_dict}.

    The single implementation of routing + both paths. All waiting inside is
    ``await asyncio.sleep`` (batch polling included) so an enclosing event
    loop is never blocked — the migrated async call sites
    (``evaluate_alignment`` under ``asyncio.run`` at callbacks.py /
    orchestrate.runner; ``evaluate_strongreject``) await this directly. Sync
    entry points use :func:`dispatch_judge_items`. Retry re-entry awaits this
    core (never the sync wrapper), so it cannot nest ``asyncio.run``.

    Args beyond the routing knobs:
        checkpoint_dir: REQUIRED for the batch path (raises there if None);
            the sync path needs no checkpoint.
        error_dict_factory: per-call-site error-dict shape, ``reason ->
            dict``. Default: alignment's ``{"aligned": None, "coherent":
            None, "reasoning": reason, "error": True}``.
        on_decision: optional callback receiving the RoutingDecision (used by
            callers that persist/print the routing, e.g. the smoke driver).
        sync_client / batch_client: injection points for tests (mock the
            anthropic client objects, not HTTP). When None, real clients are
            constructed from ``ANTHROPIC_API_KEY`` with ``max_retries=5``.
        on_item_result: optional ``(custom_id, score_dict)`` callback fired
            per item as results complete — SYNC PATH ONLY (the batch path
            persists per-sub-batch results via its dispatch checkpoint
            instead). Used by the retry path for incremental crash-safe
            persistence of sync-routed retries.
        dry_run: print the routing decision and return ``{}`` with ZERO API
            calls (OTPM from ``EPM_JUDGE_OTPM`` env or 400k assumed).
    """
    import anthropic as anthropic_mod

    if judge_system_prompt is None:
        judge_system_prompt = _default_judge_system_prompt()
    if error_dict_factory is None:
        error_dict_factory = _default_error_dict

    n_items = len(items)

    # Step 1: routing inputs. Dry-run NEVER probes; probe only near the boundary.
    if dry_run:
        otpm: int | None = int(os.environ.get("EPM_JUDGE_OTPM", OTPM_DIVISOR))
        otpm_assumed = True
    elif force_sync or n_items >= threshold_base * 2 or n_items == 0:
        otpm = None
        otpm_assumed = True
    else:
        probe_client = batch_client or anthropic_mod.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=5
        )
        otpm = probe_otpm_limit(probe_client, judge_model)
        otpm_assumed = otpm is None
        batch_client = probe_client

    decision = decide_route(
        n_items,
        threshold_base=threshold_base,
        otpm=otpm,
        force_sync=force_sync,
        sub_batch_size=sub_batch_size,
        otpm_assumed=otpm_assumed,
    )
    if on_decision is not None:
        on_decision(decision)

    # Step 2: dry-run prints and returns without any API call.
    if dry_run:
        print(decision.render())
        print("(no API calls made)")
        return {}

    if not items:
        return {}

    logger.info("Judge dispatch: %s", decision.render())
    if len(judge_system_prompt) // 4 < CACHE_MIN_TOKENS:
        logger.warning(
            "cache_control attached but inert: rubric ~%d tokens < %d minimum for %s "
            "— no cache savings expected",
            len(judge_system_prompt) // 4,
            CACHE_MIN_TOKENS,
            judge_model,
        )

    # Step 3: sync path.
    # Phase 5 (task #682): when 2+ org keys are present AND no caller-injected
    # sync_client pins us to the legacy single-org path, ROUTE through the
    # multi-org dispatcher (api_dispatch.dispatch_calls). This gives every
    # judge caller the ~3x sync fan-out across the 3 org keys for free, while
    # the legacy single-org path remains the fallback for tests + single-key
    # environments. Opt out via EPS_JUDGE_DISABLE_MULTIORG=1.
    if decision.path == "sync":
        if sync_client is None and not os.environ.get("EPS_JUDGE_DISABLE_MULTIORG"):
            from explore_persona_space.llm.api_dispatch import detect_org_keys

            org_keys = detect_org_keys()
            if len(org_keys) >= 2:
                return await _judge_items_sync_multiorg(
                    items,
                    judge_model=judge_model,
                    judge_system_prompt=judge_system_prompt,
                    max_tokens=max_tokens,
                    error_dict_factory=error_dict_factory,
                    on_item_result=on_item_result,
                )
        client = sync_client or anthropic_mod.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=5
        )
        return await _judge_items_sync(
            items,
            judge_model=judge_model,
            judge_system_prompt=judge_system_prompt,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            error_dict_factory=error_dict_factory,
            client=client,
            on_item_result=on_item_result,
        )

    # Step 4: batch path (checkpointed).
    if checkpoint_dir is None:
        raise ValueError(
            "checkpoint_dir is required for the batch path (crash-safe resume of submitted "
            "batch_ids). Pass checkpoint_dir=..., or call through judge_completions_batch / "
            "evaluate_alignment, which derive it from cache_dir / save_raw / output_dir."
        )
    client_b = batch_client or anthropic_mod.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=5
    )
    scores, retry_candidates, dispatch_dir, state = await _run_batch_path(
        items,
        judge_model=judge_model,
        judge_system_prompt=judge_system_prompt,
        max_tokens=max_tokens,
        sub_batch_size=sub_batch_size,
        checkpoint_dir=checkpoint_dir,
        poll_interval=poll_interval,
        error_dict_factory=error_dict_factory,
        client=client_b,
        now_fn=now_fn,
    )
    state_path = dispatch_dir / "state.json"

    # Step 5: errored/expired retry — ONCE, through the same threshold routing
    # (small straggler sets go sync — faster, consistent with the speed goal).
    # Resumable at every stage; protocol in _run_or_resume_retry's docstring.
    retry_state = state.get("retry")
    if retry_state and retry_state.get("status") == "done" and retry_state.get("results_merged"):
        scores.update(json.loads((dispatch_dir / "results_retry.json").read_text()))
    elif (retry_state is not None or retry_candidates) and not _is_retry:
        retry_results = await _run_or_resume_retry(
            items=items,
            retry_candidates=retry_candidates,
            state=state,
            state_path=state_path,
            dispatch_dir=dispatch_dir,
            judge_model=judge_model,
            judge_system_prompt=judge_system_prompt,
            max_tokens=max_tokens,
            threshold_base=threshold_base,
            sub_batch_size=sub_batch_size,
            max_concurrent=max_concurrent,
            poll_interval=poll_interval,
            error_dict_factory=error_dict_factory,
            sync_client=sync_client,
            batch_client=client_b,
            now_fn=now_fn,
        )
        scores.update(retry_results)
    elif retry_candidates and _is_retry:
        logger.info(
            "%d requests still errored/expired after retry; surfacing as error dicts",
            len(set(retry_candidates)),
        )

    return scores


def dispatch_judge_items(*args, **kwargs) -> dict[str, dict]:
    """THIN SYNC WRAPPER for synchronous entry points (e.g. judge_completions_batch).

    ``asyncio.run`` around :func:`dispatch_judge_items_async`. Never call
    from a running event loop — async callers await the async core directly.
    """
    return asyncio.run(dispatch_judge_items_async(*args, **kwargs))


# ── Dry-run CLI ──────────────────────────────────────────────────────────────


def _cli(argv: list[str] | None = None) -> None:
    """Inspect the routing decision for a hypothetical N with zero API calls."""
    parser = argparse.ArgumentParser(
        description="Dry-run the judge dispatch routing decision (no API calls)."
    )
    parser.add_argument("--n", type=int, required=True, help="number of judge requests")
    parser.add_argument(
        "--otpm",
        type=int,
        default=None,
        help="assume this OTPM limit (default: EPM_JUDGE_OTPM env or 400000)",
    )
    parser.add_argument("--force-sync", action="store_true", help="force the sync path")
    parser.add_argument(
        "--threshold-base", type=int, default=DEFAULT_THRESHOLD_BASE, help="sync threshold base"
    )
    args = parser.parse_args(argv)

    otpm = (
        args.otpm if args.otpm is not None else int(os.environ.get("EPM_JUDGE_OTPM", OTPM_DIVISOR))
    )
    decision = decide_route(
        args.n,
        threshold_base=args.threshold_base,
        otpm=otpm,
        force_sync=args.force_sync,
        otpm_assumed=True,  # CLI values are supplied, never probed
    )
    print(decision.render())
    print("(no API calls made)")


__all__ = [
    "JudgeItem",
    "RoutingDecision",
    "decide_route",
    "dispatch_judge_items",
    "dispatch_judge_items_async",
    "graded_temperature",
    "probe_otpm_limit",
]

if __name__ == "__main__":
    _cli()
