"""Multi-org, rate-limit-polite async Anthropic API dispatcher (Phase 4).

The reusable dispatcher behind the API-throughput plan
(``docs/api_throughput_plan.md`` § Phase 4 + § 1b Operating defaults). It
completes *N* Anthropic Messages calls in minimum wall-clock subject to the
shared-key etiquette rules, by fanning out across the THREE separate org
keys at polite per-key concurrency caps with AIMD back-off, and falling back
to the considerate Message Batches path for very large / latency-tolerant /
cost-sensitive jobs.

The single public entry point is :func:`dispatch_calls`. It takes generic
items (each carrying a stable id + payload), a ``build_request`` callable that
turns an item into Messages-API params, and a ``parse_response`` callable that
turns the model's text into a per-item result. It returns
``{item_id: DispatchResult}``.

Design (all FIRM requirements from § 1b + Phase 4):

1. **Multi-org key pool.** :func:`detect_org_keys` auto-detects whichever of
   the three keys are present in the environment
   (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BATCH_KEY`` /
   ``ANTHROPIC_API_KEY_LOW_PRIO``); one async + one sync client per present
   org. A single key is NEVER hardcoded.

2. **Per-key concurrency caps.** Sonnet ~100, Haiku ~120, Opus ~40 per key
   (good-citizen defaults, NOT hard API limits), configurable via
   ``EPS_API_CONC_<FAMILY>`` env or args. :func:`model_family` maps a model id
   to its family for the cap.

3. **Headroom routing.** Each item is assigned to whichever org currently has
   the most rate-limit headroom, read from the live
   ``anthropic-ratelimit-*-remaining`` response headers (see
   :class:`OrgState`). Until an org has been probed, it starts at full
   headroom so the warm-up ramp distributes work evenly.

4. **AIMD back-off (mandatory).** On a 429 for an org, that org's effective
   concurrency is MULTIPLICATIVELY cut (halved by default), the
   ``retry-after`` is honored before the retry, then concurrency recovers
   ADDITIVELY toward the cap while clean. The dispatcher also eases off when
   the org's shared ``*-remaining`` runs low (a colleague is active). A
   warm-up ramp on start avoids acceleration-429s.

5. **Caching / resume (interrupt-safe).** A per-item content-hash cache
   (extends :class:`~explore_persona_space.eval.batch_judge.JudgeCache`, keyed
   with the built-request fingerprint so different builders/rubrics never
   cross-read — rule 22, #1018) skips already-completed items on restart;
   results are checkpointed with atomic temp-file-then-rename writes so a
   crash / full disk leaves the last good checkpoint intact.

6. **Batch path.** For large / cost-sensitive N, reuse ``batch_judge``'s
   ``_chunk_requests`` with SMALL chunks (~1000, NOT 8000) + the ``#663``
   bounded-deadline poll; sub-batches are submitted with ``asyncio.gather``
   under a bounded concurrency cap, with ORG-AWARE resume (each sub-batch
   persists which org it went to so resume re-polls the right org).

7. **Routing.** :func:`decide_dispatch_route` chooses sync-fan-out vs batch by
   N + deadline + cost_pref. :data:`SYNC_BATCH_CROSSOVER_N` is the named
   threshold (Phase 3 will calibrate it from the measured crossover table).
   NOTE: routing decides on ``len(pending)`` — the UNCACHED remainder after the
   cache check — not the original N. A large job mostly served from cache
   re-routes only its small uncached remainder (which may fall below the sync
   crossover and go sync). This is intended (fewer items remain), but means a
   resumed batch with few uncached items can run sync.

8. **Retries.** Transient errors (timeouts, 5xx incl. 529, connection) retry
   with exponential backoff; a failed item is returned with ``error=True``
   rather than crashing the whole run.

This module does NOT migrate existing callers (Phase 5) — it only adds the new
dispatcher and its tests.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import functools
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.eval.batch_judge import JudgeCache, _chunk_requests, make_custom_id
from explore_persona_space.llm.anthropic_client import (
    BatchDeadlineExceeded,
    deadline_from_expires_at,
    parse_batch_submitted_at,
    retrieve_with_404_tolerance,
)

logger = logging.getLogger(__name__)


# ── Org key pool ─────────────────────────────────────────────────────────────

# The three SEPARATE-ORG keys (additive limits). Order = preference for ties.
# ``low_prio`` may be absent — auto-detect whatever is present (§1b).
ORG_ENV_KEYS: dict[str, str] = {
    "high_prio": "ANTHROPIC_API_KEY",
    "batch": "ANTHROPIC_BATCH_KEY",
    "low_prio": "ANTHROPIC_API_KEY_LOW_PRIO",
}


def detect_org_keys(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``{org_label: api_key}`` for every org key PRESENT in the env.

    Never hardcodes a single key; absent keys (commonly ``low_prio``) are
    skipped. Empty / whitespace-only values are treated as absent so a blank
    ``.env`` line does not register a dead org.
    """
    env = os.environ if env is None else env
    found: dict[str, str] = {}
    for label, env_var in ORG_ENV_KEYS.items():
        val = env.get(env_var)
        if val is not None and val.strip():
            found[label] = val
    return found


# ── Model family → concurrency cap ───────────────────────────────────────────

# Polite per-key concurrency caps (§1b): good-citizen defaults, NOT hard API
# limits. Configurable via EPS_API_CONC_<FAMILY> env or the dispatch args.
DEFAULT_FAMILY_CONCURRENCY: dict[str, int] = {
    "sonnet": 100,
    "haiku": 120,
    "opus": 40,
    "fable": 40,  # treated like opus (no measured cap yet)
}
# Fallback cap for an unrecognized model family (conservative).
DEFAULT_UNKNOWN_CONCURRENCY = 40

# Warm-up: start at this fraction of the cap, ramp additively (§1b: ramp to
# dodge acceleration-429s). Floor at WARMUP_MIN_CONC so tiny caps still move.
WARMUP_START_FRACTION = 0.25
WARMUP_MIN_CONC = 4
# AIMD recovery step (additive permits per recovery tick) and multiplicative
# decrease factor on a 429 (§1b).
AIMD_RECOVER_STEP = 4
AIMD_DECREASE_FACTOR = 0.5
# Ease off when an org's fractional remaining headroom drops below this (a
# colleague is active on the shared key); never 429 someone else (§1b).
LOW_HEADROOM_FRACTION = 0.15

# Outer-fan-out slack (Finding 2): the sync path admits at most
# ``sum(cap per org) * FANOUT_SLACK`` LIVE coroutines past the outer semaphore,
# so the live-coroutine count tracks the concurrency target — NOT O(N) at the
# 100k sync ceiling. The 2x head-room lets a coroutine waiting out a retry-after
# not starve a sibling org that could acquire.
FANOUT_SLACK = 2


def model_family(model_id: str) -> str:
    """Map a Claude model id to its concurrency family (sonnet/haiku/opus/...).

    Substring match on the canonical id (e.g. ``claude-sonnet-4-5-...`` ->
    ``sonnet``). Unknown families return ``"unknown"`` -> the conservative
    fallback cap.
    """
    m = model_id.lower()
    for family in ("sonnet", "haiku", "opus", "fable"):
        if family in m:
            return family
    return "unknown"


def family_concurrency_cap(
    family: str,
    *,
    overrides: dict[str, int] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Resolve the per-key concurrency cap for a model family.

    Precedence: explicit ``overrides`` arg > ``EPS_API_CONC_<FAMILY>`` env >
    :data:`DEFAULT_FAMILY_CONCURRENCY` > :data:`DEFAULT_UNKNOWN_CONCURRENCY`.
    """
    env = os.environ if env is None else env
    if overrides and family in overrides:
        return int(overrides[family])
    env_val = env.get(f"EPS_API_CONC_{family.upper()}")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            logger.warning("EPS_API_CONC_%s=%r not an int; using default", family.upper(), env_val)
    return DEFAULT_FAMILY_CONCURRENCY.get(family, DEFAULT_UNKNOWN_CONCURRENCY)


# ── Items + results ──────────────────────────────────────────────────────────

# DispatchResult.category — structured outcome discriminator. ``reason`` keeps
# the human-readable detail string; ``category`` is the enum-like signal a
# caller branches on (an exact ``==`` match, not a brittle ``reason`` substring
# search). RESULT_RATE_LIMITED distinguishes a 429-storm EXHAUSTION (the AIMD
# controller would eventually clear it, so the item is re-drivable) from a
# RESULT_ERROR terminal failure (parse / bad-request — not re-drivable).
# RESULT_TRANSPORT (#1313, llm-judging.md rule 24) marks bounded-TRANSIENT-retry
# exhaustion (connection / timeout / 5xx incl. 529) — transport-class, freely
# re-drivable, distinct from terminal RESULT_ERROR. RESULT_RATE_LIMITED is ALSO
# transport-class under rule 24 but keeps its own label for AIMD observability;
# consumers treat {RESULT_RATE_LIMITED, RESULT_TRANSPORT} as re-drivable.
RESULT_OK = "ok"
RESULT_ERROR = "error"
RESULT_RATE_LIMITED = "rate_limited_exhausted"
RESULT_TRANSPORT = "transport_exhausted"

# Per-item 429 retry budget, SEPARATE from ``max_attempts``. A 429 is pure
# backpressure (the AIMD controller honors the retry-after and clears the
# storm), so it does NOT consume a terminal-error ``attempt`` — conflating the
# two budgets is the exact mechanism that turns a transient storm into a false
# terminal. Bounded so the retry loop always terminates.
DEFAULT_MAX_429_RETRIES = 6


@dataclass(frozen=True)
class DispatchItem:
    """One unit of work: a stable id + an opaque payload.

    ``item_id`` MUST be stable across runs (it keys the content-hash cache and
    the checkpoint). ``payload`` is whatever ``build_request`` needs.
    """

    item_id: str
    payload: Any


@dataclass
class DispatchResult:
    """The outcome for one item.

    ``error`` is True for an item that exhausted retries / failed terminally;
    ``result`` is then None and ``reason`` carries the failure string. A
    successful item carries the parsed ``result`` and ``error=False``.

    ``category`` is the structured outcome discriminator (``RESULT_OK`` /
    ``RESULT_ERROR`` / ``RESULT_RATE_LIMITED`` / ``RESULT_TRANSPORT``): a
    caller branches on ``res.category in (RESULT_RATE_LIMITED,
    RESULT_TRANSPORT)`` to re-drive a transport-class exhaustion (429-storm /
    bounded-transient-retry exhaustion — connection, timeout, 5xx incl. 529;
    llm-judging.md rule 24, #1313) without crashing the pipeline, distinct
    from a terminal ``RESULT_ERROR`` (parse / bad-request).

    Checkpoint-resume caveat for ``RESULT_TRANSPORT``: api_dispatch's own
    batch checkpoint re-serves persisted error rows on resume
    (:func:`_merge_batch_record` reads the stored ``category`` directly), so a
    resumed same-checkpoint run does NOT re-dispatch a ``RESULT_TRANSPORT``
    row — re-drive rests on the CALLER branching on ``{RESULT_RATE_LIMITED,
    RESULT_TRANSPORT}``; checkpoint-resume does not self-heal the way the
    ``JudgeCache`` transport get-miss does (``eval.batch_judge``).
    """

    item_id: str
    result: Any = None
    error: bool = False
    reason: str | None = None
    org: str | None = None  # which org served it (sync path); None for cache hits
    category: str = RESULT_OK  # "ok" | "error" | "rate_limited_exhausted" | "transport_exhausted"


# Request builder: item -> Messages-API params kwargs (model/max_tokens/messages/...).
BuildRequest = Callable[[DispatchItem], dict]
# Response parser: model text -> per-item result. May raise on a bad parse.
ParseResponse = Callable[[str], Any]


def _assert_no_system_role(params: dict, item_id: str) -> None:
    """Fail fast on a system-role message in built Messages-API params.

    The Anthropic Messages API has no "system" message ROLE — a system-bearing
    ``messages`` list 400s EVERY request (invalid_request_error; the #906 r11
    incident, .claude/rules/gotchas.md). System content must be lifted by the
    BUILDER to the top-level ``system=`` param (see
    ``llm.models.Prompt.anthropic_format`` for a leading-system splitter or
    ``artifacts.datagen._gen_params_from_messages`` for an arbitrary-position
    lift). Only the documented dict shape is inspected: non-dict params /
    entries and a missing ``messages`` key pass through untouched (the API
    itself owns those errors).
    """
    messages = params.get("messages") if isinstance(params, dict) else None
    if not isinstance(messages, list):
        return
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            raise ValueError(
                f"build_request(item_id={item_id!r}) returned a messages list with "
                f'role="system" at index {i}. The Anthropic Messages API has no '
                "system message role — every such request 400s "
                "(invalid_request_error). Lift system content to the top-level "
                "system= param in your builder (see llm.models.Prompt."
                "anthropic_format or artifacts.datagen._gen_params_from_messages)."
            )


def _guarded_build_request(build_request: BuildRequest) -> BuildRequest:
    """Wrap ``build_request`` so every produced params dict is system-role-checked.

    Applied ONCE at the dispatch_calls seam; covers all three consumers (sync
    ``_do_one``, batch state-init, batch submit/resubmit) because they all
    receive the same threaded callable. Raises ValueError at BUILD time —
    before any wire call or paid batch create.
    """

    def _built(item: DispatchItem) -> dict:
        params = build_request(item)
        _assert_no_system_role(params, item.item_id)
        return params

    return _built


# ── Per-org runtime state (AIMD + headroom) ──────────────────────────────────


# How long an acquirer sleeps before re-checking the gate when the org is at
# its effective concurrency (or inside a retry-after window).
GATE_POLL_INTERVAL = 0.02

# Re-pick window (Finding 3): if the headroom-best org's gate doesn't admit
# within a few ``GATE_POLL_INTERVAL`` ticks, give up and try a sibling org (it
# may have freed a slot first) — restoring least-loaded routing under burst.
# Derived from GATE_POLL_INTERVAL so the two constants stay in sync.
REPICK_TIMEOUT = 4 * GATE_POLL_INTERVAL


@dataclass
class OrgState:
    """Mutable per-org runtime controller for one model family.

    The org's *effective* concurrency is a plain integer that the acquire path
    GATES on directly via a live ``in_flight`` counter — there is NO raw
    ``asyncio.Semaphore`` whose permit count can only grow (the bug fixed in
    review: a semaphore released on block-exit + ``recover`` can never tighten,
    so a multiplicative 429 cut would not reduce realized concurrency).

    Invariants enforced by :meth:`acquire` / :meth:`release`:

    - ``in_flight`` never exceeds ``effective`` (the AIMD-controlled limit), and
    - ``effective`` never exceeds ``cap`` (the hard ceiling), so realized live
      concurrency can NEVER exceed ``cap``.

    ``effective`` starts low (warm-up), grows additively while clean
    (:meth:`recover`), and is cut multiplicatively on a 429 (:meth:`on_429`) —
    a 429 cut visibly reduces realized in-flight because new acquirers wait
    while ``in_flight >= effective``. ``remaining_fraction_ewma`` is an EWMA of
    the most-binding fractional ``*-remaining`` header so the router eases off a
    near-exhausted shared key.
    """

    label: str
    cap: int
    effective: int = 0
    remaining_fraction_ewma: float = 1.0  # 1.0 = unprobed / full headroom
    in_flight: int = 0
    n_429: int = 0
    n_ok: int = 0
    max_in_flight: int = 0  # observability / test hook
    _retry_until: float = 0.0  # monotonic ts before which to hold off (retry-after)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        start = max(WARMUP_MIN_CONC, int(self.cap * WARMUP_START_FRACTION))
        # effective can never exceed cap (clamp warm-up start for tiny caps too).
        self.effective = min(start, self.cap)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a slot is free, then take it (in_flight += 1).

        Gates on the LIVE ``effective`` limit each iteration so a concurrent
        429 cut is honored immediately by still-waiting acquirers. Also waits
        out any recorded ``retry-after`` window before taking a slot. The
        ``in_flight <= effective <= cap`` chain bounds realized concurrency.
        """
        while True:
            now = time.monotonic()
            if now < self._retry_until:
                await asyncio.sleep(min(self._retry_until - now, GATE_POLL_INTERVAL))
                continue
            async with self._lock:
                if self.in_flight < self.effective:
                    self.in_flight += 1
                    self.max_in_flight = max(self.max_in_flight, self.in_flight)
                    return
            await asyncio.sleep(GATE_POLL_INTERVAL)

    async def release(self) -> None:
        """Release a held slot (in_flight -= 1)."""
        async with self._lock:
            self.in_flight = max(0, self.in_flight - 1)

    async def recover(self) -> None:
        """Additively grow effective concurrency toward the cap (clean tick).

        Pure integer bump — no permit juggling. Clamped at ``cap`` so live
        in-flight (bounded by ``effective``) can never exceed the ceiling.
        """
        async with self._lock:
            if self.effective < self.cap:
                self.effective = min(self.cap, self.effective + AIMD_RECOVER_STEP)

    async def on_429(self, retry_after_s: float | None) -> None:
        """Multiplicatively cut effective concurrency; record the retry-after.

        The integer ``effective`` drops immediately; because :meth:`acquire`
        gates on it live, still-waiting acquirers see the lower limit at once
        and realized ``in_flight`` falls as in-flight calls finish without being
        replaced. The ``retry-after`` is recorded as a monotonic deadline that
        :meth:`acquire` (and :meth:`wait_retry_after`) honor.
        """
        async with self._lock:
            self.n_429 += 1
            self.effective = max(WARMUP_MIN_CONC, int(self.effective * AIMD_DECREASE_FACTOR))
            if retry_after_s is not None and retry_after_s > 0:
                self._retry_until = max(self._retry_until, time.monotonic() + retry_after_s)

    async def wait_retry_after(self) -> None:
        """Block until any recorded ``retry-after`` window has elapsed."""
        delay = self._retry_until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def note_remaining(self, headers: Any) -> None:
        """Update headroom from the most-binding ``*-remaining`` header.

        Reads requests / output-tokens / input-tokens remaining + their limit,
        takes the lowest fractional headroom across them, and folds it into an
        EWMA (weight 0.5) so a single spiky reading does not whipsaw routing.
        """
        fractions: list[float] = []
        for limiter in ("requests", "output-tokens", "input-tokens"):
            rem = _header_int(headers, f"anthropic-ratelimit-{limiter}-remaining")
            lim = _header_int(headers, f"anthropic-ratelimit-{limiter}-limit")
            if rem is not None and lim and lim > 0:
                fractions.append(max(0.0, min(1.0, rem / lim)))
        if fractions:
            observed = min(fractions)
            self.remaining_fraction_ewma = 0.5 * self.remaining_fraction_ewma + 0.5 * observed

    def routing_headroom(self) -> float:
        """A 0..1 score: higher = more capacity to take the next item.

        Combines the shared ``*-remaining`` fraction (so we avoid a key a
        colleague is hammering) with how far below its cap the org currently
        is, so a backed-off org is deprioritized until it recovers.
        """
        conc_room = self.effective / self.cap if self.cap else 0.0
        return min(self.remaining_fraction_ewma, max(0.0, conc_room))

    def low_headroom(self) -> bool:
        return self.remaining_fraction_ewma < LOW_HEADROOM_FRACTION


def _header_int(headers: Any, name: str) -> int | None:
    """Read an int header value, tolerating dict-like and SDK header objects."""
    try:
        val = headers.get(name)
    except AttributeError:
        val = None
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ── Routing ──────────────────────────────────────────────────────────────────

# Sync-fan-out vs batch crossover. Below this N (and when a deadline allows),
# the polite ~9k-RPM Sonnet sync fan-out across orgs clears the job quickly;
# at/above it (or cost-sensitive / latency-tolerant), the considerate batch
# path is preferred. PLACEHOLDER: Phase 3 will calibrate this from the measured
# N x model x {latency,cost} crossover table (docs/api_throughput_plan.md §3).
SYNC_BATCH_CROSSOVER_N = 2_000
# Small batch chunk size (§Phase 4: 500-2000, NOT 8000 — the #658 failure shape).
DEFAULT_BATCH_CHUNK_SIZE = 1_000
# At most this many sub-batches submitted/in-flight concurrently (keeps total
# in-flight far below any org's batch processing-queue cap).
DEFAULT_MAX_CONCURRENT_SUB_BATCHES = 4
BATCH_DEADLINE_GRACE_MIN = 30
VALID_COST_PREFS = ("balanced", "cost", "latency")
VALID_FORCE_PATHS = (None, "sync", "batch")


@dataclass
class DispatchRoute:
    """The routing outcome for one ``dispatch_calls`` invocation."""

    n_items: int
    path: str  # "sync" | "batch"
    cost_pref: str
    reason: str

    def render(self) -> str:
        return f"N={self.n_items} | path={self.path} | cost_pref={self.cost_pref} | {self.reason}"


def decide_dispatch_route(
    n_items: int,
    *,
    deadline: _dt.datetime | None = None,
    cost_pref: str = "balanced",
    crossover_n: int = SYNC_BATCH_CROSSOVER_N,
    force_path: str | None = None,
) -> DispatchRoute:
    """Choose sync-fan-out vs batch by N + deadline + cost_pref.

    Rules (sensible defaults; Phase 3 recalibrates ``crossover_n``):

    - ``force_path`` (``"sync"`` / ``"batch"``) overrides everything.
    - ``cost_pref="cost"`` -> batch (the 50% discount path) unless N is tiny.
    - ``cost_pref="latency"`` -> sync (lowest wall-clock) regardless of N,
      UNLESS a deadline more than 24h out makes the batch SLA acceptable AND N
      is large (then batch is the considerate choice).
    - ``cost_pref="balanced"`` (default) -> sync below ``crossover_n``, batch
      at/above it; a near-term deadline (< the batch 24h SLA) forces sync.

    Raises ``ValueError`` on an unknown ``cost_pref`` / ``force_path`` rather
    than silently falling through to balanced-sync.
    """
    if cost_pref not in VALID_COST_PREFS:
        raise ValueError(f"cost_pref must be one of {VALID_COST_PREFS}, got {cost_pref!r}")
    if force_path not in VALID_FORCE_PATHS:
        raise ValueError(f"force_path must be one of {VALID_FORCE_PATHS}, got {force_path!r}")
    if force_path in ("sync", "batch"):
        return DispatchRoute(n_items, force_path, cost_pref, "forced")

    # A deadline that the batch 24h SLA cannot meet forces sync.
    deadline_forces_sync = False
    if deadline is not None:
        now = _dt.datetime.now(_dt.UTC)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=_dt.UTC)
        deadline_forces_sync = (deadline - now) < _dt.timedelta(hours=24)

    if cost_pref == "cost":
        if n_items < max(1, crossover_n // 10):
            return DispatchRoute(n_items, "sync", cost_pref, "cost_pref but tiny N")
        if deadline_forces_sync:
            return DispatchRoute(n_items, "sync", cost_pref, "cost_pref but deadline < 24h SLA")
        return DispatchRoute(n_items, "batch", cost_pref, "cost_pref -> 50% batch discount")

    if cost_pref == "latency":
        return DispatchRoute(n_items, "sync", cost_pref, "latency_pref -> sync fan-out")

    # balanced
    if deadline_forces_sync:
        return DispatchRoute(n_items, "sync", cost_pref, "deadline < 24h SLA -> sync")
    if n_items < crossover_n:
        return DispatchRoute(n_items, "sync", cost_pref, f"N < crossover {crossover_n}")
    return DispatchRoute(n_items, "batch", cost_pref, f"N >= crossover {crossover_n}")


# ── Checkpoint helpers (atomic) ──────────────────────────────────────────────


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write: tmp file in the same dir + os.replace.

    A crash / full disk during the write leaves the prior file intact (the
    rename is atomic on POSIX) — the interrupt-safe checkpoint contract (§1b).
    """
    path = Path(path)
    # Per-PID tmp name: concurrent sessions share ~/.task-workflow files (the
    # headroom snapshot), and a fixed tmp collides across writers mid-write.
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ── Transient-error retry ────────────────────────────────────────────────────


def _is_transient(exc: BaseException) -> bool:
    """True for retryable transient API errors (timeouts, 5xx incl. 529, connection).

    529 ``OverloadedError`` is NOT an ``InternalServerError`` subclass in the
    installed SDK (anthropic 0.88.0: MRO OverloadedError -> APIStatusError ->
    APIError) and is not exported at the ``anthropic`` top level, so it is
    matched via the public ``APIStatusError`` + ``status_code == 529`` form
    (llm-judging rule 24(i); #1313). ``RateLimitError`` (429) is handled
    SEPARATELY via AIMD — ``_do_one`` checks ``_is_rate_limit`` BEFORE this
    predicate — so it is NOT in this transient set.
    """
    import anthropic as _anthropic

    transient = (
        _anthropic.APIConnectionError,
        _anthropic.APITimeoutError,
        _anthropic.InternalServerError,  # 500-class; does NOT cover 529 (see docstring)
    )
    if isinstance(exc, transient):
        return True
    # 529 Overloaded — explicit public-form match (robust to SDK re-parenting).
    return isinstance(exc, _anthropic.APIStatusError) and getattr(exc, "status_code", None) == 529


def _is_rate_limit(exc: BaseException) -> bool:
    import anthropic as _anthropic

    return isinstance(exc, _anthropic.RateLimitError)


def is_transport_exception(exc: BaseException) -> bool:
    """Rule-24 transport-class taxonomy over SDK exceptions.

    True when the call died in TRANSPORT — before any verdict about the
    content was produced, so the item is freely re-judgeable
    (llm-judging.md rule 24(i); #1313): transient (connection / timeout /
    5xx incl. 529) OR 429 rate limit. NOT transport: a 400-class
    invalid_request (a pipeline bug — neither retried nor dropped, rule
    24(iii)) or any non-API exception (e.g. a parse error).
    """
    return _is_transient(exc) or _is_rate_limit(exc)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Pull a ``retry-after`` (seconds) off a RateLimitError's response."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("retry-after")
    except AttributeError:
        raw = None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# ── Sync fan-out path ────────────────────────────────────────────────────────


def _fanout_limit(org_states: dict[str, OrgState]) -> int:
    """Outer-semaphore size: ~the total concurrency target across orgs.

    Each org admits at most ``cap`` concurrent calls; :data:`FANOUT_SLACK`
    head-room lets a coroutine waiting out a retry-after not block a sibling
    that could acquire. Bounds LIVE coroutines to ``O(cap * n_orgs)``, not the
    ``O(N)`` queue depth at the sync ceiling (Finding 2). Independent of N.
    """
    total_cap = sum(st.cap for st in org_states.values())
    return max(1, total_cap * FANOUT_SLACK)


def _merge_batch_record(item_id: str, rec: dict, org: str | None) -> DispatchResult:
    """Reconstruct a :class:`DispatchResult` from one persisted batch record.

    The persisted record (written by :func:`_harvest_sub_batch`) carries
    ``result`` / ``error`` / ``reason`` keyed by ``item_id`` in the enclosing
    dict; ``item_id`` and ``org`` are passed in by the merge loop.

    Threads ``category`` with a default that stays SELF-CONSISTENT with
    ``error`` for any legacy record written before the field existed
    (``RESULT_ERROR if error else RESULT_OK``) — so an ``error=True`` batch
    record can NEVER read back ``category=RESULT_OK`` (Finding 1, Must-Fix 1).
    A record that DOES carry an explicit ``category`` key is read directly.
    """
    error = rec.get("error", False)
    return DispatchResult(
        item_id=item_id,
        result=rec.get("result"),
        error=error,
        reason=rec.get("reason"),
        org=org,
        category=rec.get("category", RESULT_ERROR if error else RESULT_OK),
    )


def _pick_org_excluding(
    org_states: dict[str, OrgState], labels: list[str], rr: dict[str, int], tried: set[str]
) -> str:
    """Highest-routing-headroom org NOT in ``tried``; round-robin tie-break.

    Filtering and selection happen in ONE label space: the labels are first
    rotated by the round-robin pointer ``rr["i"]`` (so equal-headroom orgs
    share load — ``max`` returns the first rotated position on a tie), THEN
    the ``tried`` filter is applied to those rotated labels. Earlier the filter
    ran in unrotated index space while selection mapped through ``rr["i"]``,
    so with a nonzero pointer the helper could return an org in ``tried`` —
    handing a re-pick back the very org that just timed out (#684 round 2).

    Advances the shared round-robin pointer ``rr["i"]`` so successive picks
    rotate. Caller (:func:`_pick_org_then_acquire`) guarantees a non-empty
    candidate set: the timeout loop passes ``tried`` only while it is a strict
    subset of ``labels``, and the blocking-fallback call passes ``set()``.
    """
    n = len(labels)
    # Rotate first, then filter on the SAME rotated labels.
    rotated = [labels[(rr["i"] + i) % n] for i in range(n)]
    candidates = [lbl for lbl in rotated if lbl not in tried]
    if not candidates:  # defensive: caller guarantees this never happens
        raise ValueError("_pick_org_excluding: every org is in `tried`")
    chosen = max(candidates, key=lambda lbl: org_states[lbl].routing_headroom())
    rr["i"] = (rr["i"] + 1) % n
    return chosen


async def _pick_org_then_acquire(
    org_states: dict[str, OrgState], labels: list[str], rr: dict[str, int]
) -> str:
    """Pick the highest-headroom org and acquire its slot (Finding 3).

    If the chosen org's gate doesn't admit within :data:`REPICK_TIMEOUT`,
    re-pick (a sibling org may have freed a slot first) — restoring least-loaded
    routing under burst. Returns the LABEL of the org whose slot is now HELD;
    the caller does NOT re-acquire.

    Falls back to a plain blocking acquire on the headroom-best org once every
    org has timed out this round, so the coroutine never spins forever when all
    orgs are saturated (degrades to today's pick-then-block; the 429 budget +
    the inner gate's retry-after still bound total time).
    """
    tried: set[str] = set()
    while len(tried) < len(labels):
        org = _pick_org_excluding(org_states, labels, rr, tried)
        try:
            # On a timeout the wait_for CANCELS the pending acquire() at its next
            # suspension point (an ``await asyncio.sleep``) — BEFORE the
            # ``in_flight += 1`` that runs atomically under the lock with the
            # immediately-following return (no await between them). So a timeout
            # means the slot was NOT taken: do NOT release.
            await asyncio.wait_for(org_states[org].acquire(), timeout=REPICK_TIMEOUT)
            return org
        except TimeoutError:
            tried.add(org)
    # All orgs slow this round -> block on the headroom-best with no timeout.
    org = _pick_org_excluding(org_states, labels, rr, set())
    await org_states[org].acquire()
    return org


async def _dispatch_sync(
    items: list[DispatchItem],
    *,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    org_states: dict[str, OrgState],
    async_clients: dict[str, Any],
    max_attempts: int,
    max_429_retries: int = DEFAULT_MAX_429_RETRIES,
    on_result: Callable[[DispatchResult], None] | None,
) -> dict[str, DispatchResult]:
    """Fan items across orgs at per-key caps with AIMD back-off + headroom routing.

    Each item is routed to the org with the most headroom at acquire time, then
    holds that org's gate (``acquire``/``release``, bounded by the live
    ``effective`` limit <= ``cap``) for the call. A 429 cuts the org's
    concurrency (AIMD) — visibly reducing realized in-flight because the gate
    is live — and the item is retried (honoring ``retry-after``) on a
    re-selected org; transient errors retry with backoff; terminal failures
    return an ``error=True`` result rather than crashing the run.
    ``asyncio.CancelledError`` / ``KeyboardInterrupt`` are NOT swallowed — they
    propagate so cancellation / interrupt-safety is preserved.

    429s have their OWN retry budget (``max_429_retries``) separate from
    ``max_attempts`` (Finding 1b): a 429 does NOT consume an ``attempt``, so a
    storm no longer burns the terminal-error budget. An item that exhausts the
    429 budget returns ``category=RESULT_RATE_LIMITED`` (re-drivable), distinct
    from a ``RESULT_ERROR`` terminal failure. The outer ``fanout`` semaphore
    (Finding 2) caps LIVE coroutines at ``_fanout_limit(org_states)`` so the
    coroutine count tracks the concurrency target, not the O(N) queue depth.
    """
    results: dict[str, DispatchResult] = {}
    labels = list(org_states)
    # Round-robin pointer breaks ties so two equal-headroom orgs share load.
    rr = {"i": 0}

    async def _do_one(item: DispatchItem) -> None:
        params = build_request(item)
        last_reason = "unknown"
        last_category = RESULT_ERROR
        n_429 = 0
        attempt = 0
        while attempt < max_attempts:
            # Finding 3: pick by headroom + acquire (re-picks on a slow gate).
            org = await _pick_org_then_acquire(org_states, labels, rr)  # slot held on return
            state = org_states[org]
            client = async_clients[org]
            try:
                raw = await client.messages.with_raw_response.create(**params)
                state.note_remaining(raw.headers)
                # Best-effort cross-session headroom snapshot (fail-soft; never
                # raises into the dispatch loop — see record_headroom_observation).
                record_headroom_observation(org, params.get("model", "unknown"), raw.headers)
                msg = raw.parse()
                text = next((b.text for b in msg.content if b.type == "text"), "")
                parsed = parse_response(text)
                state.n_ok += 1
                # Clean call -> additively recover toward the cap.
                if not state.low_headroom():
                    await state.recover()
                res = DispatchResult(item.item_id, result=parsed, org=org, category=RESULT_OK)
                results[item.item_id] = res
                if on_result is not None:
                    on_result(res)
                return
            except (asyncio.CancelledError, KeyboardInterrupt):
                # Cancellation / interrupt must propagate (interrupt-safety) —
                # never swallowed into a per-item error dict.
                raise
            except Exception as exc:
                if _is_rate_limit(exc):
                    await state.on_429(_retry_after_seconds(exc))
                    n_429 += 1
                    last_category = RESULT_RATE_LIMITED
                    last_reason = f"429 (org={org}, 429-retry {n_429})"
                    if n_429 >= max_429_retries:
                        last_reason = f"rate_limited_exhausted (org={org}, 429 retries {n_429})"
                        break
                    continue  # 429 does NOT consume an attempt (Finding 1b)
                if _is_transient(exc):
                    # Exhaustion of the bounded transient budget is transport-class
                    # (re-drivable), not a terminal error (rule 24(ii); #1313).
                    last_category = RESULT_TRANSPORT
                    last_reason = f"transient {type(exc).__name__} (attempt {attempt + 1})"
                    await asyncio.sleep(1.5**attempt)
                    attempt += 1
                    continue
                # Non-transient (parse error, bad request, etc.) -> terminal.
                last_category = RESULT_ERROR
                last_reason = f"error: {exc}"
                break
            finally:
                await state.release()
        res = DispatchResult(
            item.item_id, error=True, reason=last_reason, org=None, category=last_category
        )
        results[item.item_id] = res
        if on_result is not None:
            on_result(res)

    # Finding 2: outer admission gate bounds LIVE coroutines to the concurrency
    # target (sum(cap) * FANOUT_SLACK), not the O(N) queue depth.
    fanout = asyncio.Semaphore(_fanout_limit(org_states))

    async def _do_one_bounded(item: DispatchItem) -> None:
        async with fanout:
            await _do_one(item)

    # residual: gather still creates O(N) Task objects (~140 MB at N=100k);
    # routing steers large N to the batch path (see Finding 4 module __doc__
    # item 7), so the sync ceiling is the rare case. The semaphore fixes the
    # busy-poll CPU storm the reviewer flagged; the worker-pool/chunked
    # alternative is out-of-contract (§11).
    await asyncio.gather(*[_do_one_bounded(it) for it in items])
    return results


# ── Batch path (org-aware, checkpointed) ─────────────────────────────────────


def _batch_run_fingerprint(items: list[DispatchItem], build_request: BuildRequest) -> str:
    """Run-level request fingerprint over the dispatched set (rule 22, #1018).

    Hashes the sorted ``(item_id, built-request fingerprint)`` pairs of the
    pending set, binding a batch checkpoint to the builder/rubric that created
    it — so a rubric-B dispatch reusing rubric-A's ``checkpoint_dir`` fails
    loud at state load instead of replaying A's judgments under B's key.
    """
    pairs = sorted((it.item_id, _cache_key_parts(it, build_request)[2]) for it in items)
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


def _load_or_init_batch_state(
    state_path: Path,
    items: list[DispatchItem],
    *,
    build_request: BuildRequest,
    org_labels: list[str],
    chunk_size: int,
) -> dict:
    """Load a resumable batch state.json or init a fresh one (atomic writes).

    The state records, per sub-batch: its custom_ids, the ORG it was (or will
    be) submitted to, the batch_id, status, and the persisted poll deadline.
    Org assignment is round-robin across the present orgs at init so resume
    re-polls each sub-batch on the SAME org it was created on (a batch created
    on org B 404s if polled on org A).

    Rubric binding (rule 22, #1018): the state carries a run-level
    ``request_fingerprint`` over the dispatched set; a LOAD whose recomputed
    fingerprint mismatches — or a pre-fix state.json with no fingerprint field
    (rubric provenance unknown) — raises ValueError, NEVER silently re-inits
    (that would discard the prior state pointer and orphan paid batches).
    """
    run_fp = _batch_run_fingerprint(items, build_request)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        stored_fp = state.get("request_fingerprint")
        if stored_fp != run_fp:
            raise ValueError(
                f"Batch checkpoint at {state_path} does not match this dispatch: stored "
                f"request_fingerprint={stored_fp!r}, recomputed={run_fp!r}. This checkpoint "
                "belongs to a different builder/rubric, predates the #1018 fingerprint field, "
                "or a partially-persisted prior run shrank the pending set. Use a distinct "
                f"checkpoint_dir per rubric, or delete {state_path} to resubmit the remainder. "
                "Refusing to resume across the mismatch."
            )
        return state

    # Build one request per item, chunk small, assign orgs round-robin.
    cid_for = {it.item_id: make_custom_id(it.item_id) for it in items}
    requests = [{"custom_id": cid_for[it.item_id], "params": build_request(it)} for it in items]
    chunks = _chunk_requests(requests, max_count=chunk_size)
    state = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_fingerprint": run_fp,
        "sub_batches": [
            {
                "index": i,
                "org": org_labels[i % len(org_labels)],
                "custom_ids": [r["custom_id"] for r in chunk],
                "n_requests": len(chunk),
                "status": "pending",
                "batch_id": None,
                "deadline": None,
            }
            for i, chunk in enumerate(chunks)
        ],
        # custom_id -> item_id map so results join back to items on resume.
        "cid_to_item": {cid_for[it.item_id]: it.item_id for it in items},
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(state_path, state)
    return state


async def _submit_one_sub_batch(
    sb: dict,
    *,
    state: dict,
    state_path: Path,
    items_by_cid: dict[str, DispatchItem],
    build_request: BuildRequest,
    sync_clients: dict[str, Any],
    sem: asyncio.Semaphore,
) -> None:
    """Submit ONE pending sub-batch on its assigned org under the concurrency sem.

    Preserve-before-propagate atomicity (the #663 contract): the ``submitting``
    intent is persisted BEFORE ``batches.create``; the ``batch_id`` + deadline
    are persisted in a ``finally``-shielded block so a cancellation can never
    interleave between create returning and the id being recorded (which would
    orphan a paid batch + leave a non-resumable wedge).

    Cache-sharing constraint: a checkpointed batch run and a content-hash cache
    must not be shared such that an UNSUBMITTED item in this checkpoint becomes
    cached by another run between crash and resume — that would drop it from the
    current ``pending`` set (hence ``items_by_cid``). We fail loud on a missing
    custom_id rather than silently resubmitting a partial sub-batch.
    """
    async with sem:
        if sb["status"] != "pending":
            return
        org = sb["org"]
        client = sync_clients[org]
        missing = [cid for cid in sb["custom_ids"] if cid not in items_by_cid]
        if missing:
            raise RuntimeError(
                f"Sub-batch {sb['index']} references {len(missing)} custom_id(s) absent from the "
                f"current dispatch items (first: {missing[0]!r}). An unsubmitted item likely "
                "became cached by a concurrent run between crash and resume. Use a checkpoint dir "
                "dedicated to this run, or clear the cache for these items, then re-run. Refusing "
                "to submit a partial sub-batch."
            )
        # Build requests BEFORE flipping status to "submitting" (#991): the build
        # is pure, so a builder raise on RESUME (e.g. the system-role guard)
        # leaves this sub-batch "pending" — cleanly resumable after the builder
        # is fixed — instead of wedging at the crashed-mid-submit RuntimeError.
        # The #663 preserve-before-propagate contract is untouched: the
        # "submitting" intent is still persisted BEFORE batches.create.
        requests = [
            {"custom_id": cid, "params": build_request(items_by_cid[cid])}
            for cid in sb["custom_ids"]
        ]
        sb["status"] = "submitting"
        _atomic_write_json(state_path, state)  # intent BEFORE create
        # api_dispatch IS a sanctioned hardened batch client: reuses _chunk_requests (<=8k shards)
        # + bounded expires_at poll (deadline_from_expires_at/BatchDeadlineExceeded) + org-aware
        # resume by custom_id; routing through judge_completions_batch would lose multi-org fan-out.
        # BATCH_JUDGE_CLIENT_EXEMPT: sanctioned hardened multi-org batch client (see above)
        batch = await asyncio.to_thread(client.messages.batches.create, requests=requests)
        try:
            sb["batch_id"] = batch.id
            sb["status"] = "submitted"
            # #995: create-time anchor for the poll step's 404 grace (mirrors
            # judge_dispatch). Additive state key — an OLD state.json lacks it,
            # so a resume reads None via .get() -> no grace -> terminal 404,
            # exactly the desired resume default.
            sb["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            expires_at = getattr(batch, "expires_at", None)
            if expires_at is not None:
                sb["deadline"] = deadline_from_expires_at(
                    expires_at, BATCH_DEADLINE_GRACE_MIN
                ).isoformat()
        finally:
            _atomic_write_json(state_path, state)
        logger.info("Sub-batch %d submitted on org=%s as %s", sb["index"], org, batch.id)


async def _poll_one_sub_batch_step(
    sb: dict,
    *,
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    sync_clients: dict[str, Any],
    parse_response: ParseResponse,
    now_fn: Callable[[], _dt.datetime],
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    """Poll one not-yet-collected sub-batch ONCE on its org; harvest on end.

    404 tolerance (#995 + #1035): a ``NotFoundError`` on the loop retrieve or
    the overdue final retrieve is retried with bounded backoff INSIDE the
    ``to_thread`` worker (the event loop is never blocked by the sleeps): the
    create-grace schedule within ``BATCH_CREATE_404_GRACE_S`` of the
    sub-batch's persisted ``submitted_at``, the ``BATCH_MIDPOLL_404_BACKOFF_S``
    mid-poll schedule anywhere else (incl. a resume with a stale/absent
    ``submitted_at``). Org-mismatch semantics are preserved
    (``_load_or_init_batch_state``: a batch created on org B 404s if polled on
    org A): create and poll share ``sb["org"]`` within a run, so a wrong-org
    404 arises only on resume or from an org-routing code bug (delayed <=60s
    grace + <=~3 min mid-poll, then still fails loud — never masked).
    ``sleep_fn`` is additive + injectable for tests (default ``time.sleep``).
    """
    org = sb["org"]
    client = sync_clients[org]
    batch = await asyncio.to_thread(
        retrieve_with_404_tolerance,
        functools.partial(client.messages.batches.retrieve, sb["batch_id"]),
        created_at=parse_batch_submitted_at(sb.get("submitted_at")),
        batch_id=sb["batch_id"],
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    if sb.get("deadline") is None and getattr(batch, "expires_at", None) is not None:
        sb["deadline"] = deadline_from_expires_at(
            batch.expires_at, BATCH_DEADLINE_GRACE_MIN
        ).isoformat()
        _atomic_write_json(state_path, state)
    if batch.processing_status == "ended":
        await _harvest_sub_batch(
            sb,
            state=state,
            state_path=state_path,
            dispatch_dir=dispatch_dir,
            client=client,
            parse_response=parse_response,
        )
        return
    if sb.get("deadline") and now_fn() > _dt.datetime.fromisoformat(sb["deadline"]):
        # #1035: the overdue final retrieve is the last chance to harvest, so a
        # transient 404 gets the bounded mid-poll tolerance; a genuinely-gone
        # batch still re-raises NotFoundError <=~3 min later.
        final = await asyncio.to_thread(
            retrieve_with_404_tolerance,
            functools.partial(client.messages.batches.retrieve, sb["batch_id"]),
            created_at=parse_batch_submitted_at(sb.get("submitted_at")),
            batch_id=sb["batch_id"],
            now_fn=now_fn,
            sleep_fn=sleep_fn,
        )
        if final.processing_status == "ended":
            await _harvest_sub_batch(
                sb,
                state=state,
                state_path=state_path,
                dispatch_dir=dispatch_dir,
                client=client,
                parse_response=parse_response,
            )
            return
        raise BatchDeadlineExceeded(sb["batch_id"], sb["deadline"])


async def _harvest_sub_batch(
    sb: dict,
    *,
    state: dict,
    state_path: Path,
    dispatch_dir: Path,
    client: Any,
    parse_response: ParseResponse,
) -> None:
    """Collect an ended sub-batch, persist its results json, mark it collected."""
    cid_to_item = state["cid_to_item"]
    raw_results = await asyncio.to_thread(
        lambda: list(client.messages.batches.results(sb["batch_id"]))
    )
    scores: dict[str, dict] = {}
    for result in raw_results:
        cid = result.custom_id
        item_id = cid_to_item.get(cid, cid)
        rtype = result.result.type
        if rtype == "succeeded":
            text = next((b.text for b in result.result.message.content if b.type == "text"), "")
            try:
                parsed = parse_response(text)
                scores[item_id] = {"result": parsed, "error": False, "reason": None}
            except Exception as e:
                scores[item_id] = {"result": None, "error": True, "reason": f"parse_error: {e}"}
        else:
            # #1313 (rule 24): server-class errored / expired / canceled /
            # unknown rows are transport-class (re-drivable) -> RESULT_TRANSPORT;
            # a quarantined invalid_request_error stays terminal RESULT_ERROR
            # (rule 24(iii)). SDK nesting is result.result.error.error.type
            # (double .error); getattr-guarded so a shape mismatch fails toward
            # transport (visible + re-judgeable), mirroring judge_dispatch.
            etype = getattr(
                getattr(getattr(result.result, "error", None), "error", None), "type", None
            )
            category = RESULT_ERROR if etype == "invalid_request_error" else RESULT_TRANSPORT
            scores[item_id] = {
                "result": None,
                "error": True,
                "reason": f"batch_error: {rtype}",
                "category": category,
            }
    _atomic_write_json(dispatch_dir / f"results_{sb['batch_id']}.json", scores)
    sb["status"] = "collected"
    _atomic_write_json(state_path, state)


async def _dispatch_batch(
    items: list[DispatchItem],
    *,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    org_labels: list[str],
    sync_clients: dict[str, Any],
    checkpoint_dir: Path,
    chunk_size: int,
    max_concurrent_sub_batches: int,
    poll_interval: float,
    now_fn: Callable[[], _dt.datetime] | None = None,
) -> dict[str, DispatchResult]:
    """Submit/resume + poll + collect all sub-batches, org-aware + checkpointed.

    Org-aware resume: each sub-batch persists the org it went to, so a resumed
    run re-polls every sub-batch on its OWN org (not whichever org the router
    would now pick). All waits are ``await asyncio.sleep`` so the loop never
    blocks an enclosing event loop.
    """
    now_fn = now_fn or (lambda: _dt.datetime.now(_dt.UTC))
    dispatch_dir = Path(checkpoint_dir)
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    state_path = dispatch_dir / "state.json"
    state = _load_or_init_batch_state(
        state_path,
        items,
        build_request=build_request,
        org_labels=org_labels,
        chunk_size=chunk_size,
    )
    items_by_cid = {make_custom_id(it.item_id): it for it in items}

    # Reconcile any crashed-mid-submit sub-batch first (sequential).
    for sb in state["sub_batches"]:
        if sb["status"] == "submitting":
            if sb["batch_id"] is None:
                raise RuntimeError(
                    f"Sub-batch {sb['index']} crashed mid-submit (status=submitting, "
                    f"batch_id=None) at {state_path}. A batch may have been created but not "
                    f"recorded; inspect batches.list() on org={sb['org']} and record its id, "
                    "or delete the checkpoint dir to resubmit. Refusing to silently resubmit."
                )
            sb["status"] = "submitted"
            _atomic_write_json(state_path, state)

    # Submit pending sub-batches under a bounded concurrency cap.
    pending = [sb for sb in state["sub_batches"] if sb["status"] == "pending"]
    if pending:
        sem = asyncio.Semaphore(max_concurrent_sub_batches)
        submit_results = await asyncio.gather(
            *(
                _submit_one_sub_batch(
                    sb,
                    state=state,
                    state_path=state_path,
                    items_by_cid=items_by_cid,
                    build_request=build_request,
                    sync_clients=sync_clients,
                    sem=sem,
                )
                for sb in pending
            ),
            return_exceptions=True,
        )
        for r in submit_results:
            if isinstance(r, BaseException):
                raise r

    # Deadline-bounded polling.
    current = poll_interval
    max_interval = max(poll_interval, 120.0)
    while any(sb["status"] != "collected" for sb in state["sub_batches"]):
        for sb in state["sub_batches"]:
            if sb["status"] != "collected":
                await _poll_one_sub_batch_step(
                    sb,
                    state=state,
                    state_path=state_path,
                    dispatch_dir=dispatch_dir,
                    sync_clients=sync_clients,
                    parse_response=parse_response,
                    now_fn=now_fn,
                )
        if all(sb["status"] == "collected" for sb in state["sub_batches"]):
            break
        await asyncio.sleep(current)
        current = min(current * 1.5, max_interval)

    # Merge all collected results.
    results: dict[str, DispatchResult] = {}
    for sb in state["sub_batches"]:
        payload = json.loads((dispatch_dir / f"results_{sb['batch_id']}.json").read_text())
        for item_id, rec in payload.items():
            # Finding 1 Must-Fix 1: thread ``category`` so an error=True batch
            # record never reads back category="ok" (the RESULT_OK field default).
            results[item_id] = _merge_batch_record(item_id, rec, sb["org"])
    return results


# ── Client construction ──────────────────────────────────────────────────────


def _build_clients(
    org_keys: dict[str, str],
    *,
    max_retries: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct one AsyncAnthropic + one Anthropic client per present org.

    ``max_retries=0`` because THIS dispatcher owns retry/back-off (the SDK's
    own retry would double-count 429s against our AIMD controller). Returns
    ``(async_clients, sync_clients)`` keyed by org label.
    """
    import anthropic as _anthropic

    async_clients = {
        label: _anthropic.AsyncAnthropic(api_key=key, max_retries=max_retries)
        for label, key in org_keys.items()
    }
    sync_clients = {
        label: _anthropic.Anthropic(api_key=key, max_retries=max_retries)
        for label, key in org_keys.items()
    }
    return async_clients, sync_clients


# ── Public entry point ───────────────────────────────────────────────────────


async def dispatch_calls(
    items: list[DispatchItem],
    *,
    model: str,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    deadline: _dt.datetime | None = None,
    cost_pref: str = "balanced",
    cache_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    concurrency_overrides: dict[str, int] | None = None,
    crossover_n: int = SYNC_BATCH_CROSSOVER_N,
    chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE,
    max_concurrent_sub_batches: int = DEFAULT_MAX_CONCURRENT_SUB_BATCHES,
    max_attempts: int = 5,
    max_429_retries: int = DEFAULT_MAX_429_RETRIES,
    poll_interval: float = 30.0,
    force_path: str | None = None,
    org_keys: dict[str, str] | None = None,
    async_clients: dict[str, Any] | None = None,
    sync_clients: dict[str, Any] | None = None,
    cache: JudgeCache | None = None,
    on_route: Callable[[DispatchRoute], None] | None = None,
    now_fn: Callable[[], _dt.datetime] | None = None,
) -> dict[str, DispatchResult]:
    """Dispatch *N* Anthropic calls politely across the org pool; return per-item results.

    Args:
        items: work units (stable id + opaque payload).
        model: Claude model id (sets the per-key concurrency family).
        build_request: ``item -> Messages-API params`` (model/max_tokens/messages/...).
            MUST include the ``model`` field; this dispatcher does not inject it.
            NOTE: the Messages API has NO "system" message role — a builder
            forwarding caller message lists verbatim MUST lift system-role
            entries to the top-level ``system=`` param (see
            ``llm.models.Prompt.anthropic_format``). ENFORCED at runtime:
            a system-bearing built params dict raises ValueError at build
            time, before any wire call (gotchas.md, #906 r11; #991).
        parse_response: ``model_text -> result``; may raise on a bad parse
            (caught per item -> ``error=True``).
        deadline: optional wall-clock deadline; a deadline inside the batch 24h
            SLA forces the sync path.
        cost_pref: ``"balanced"`` (default) | ``"cost"`` (prefer 50% batch) |
            ``"latency"`` (prefer sync fan-out).
        cache_dir: per-item content+request-fingerprint cache root (resume
            skips cached items; keys carry the built-request fingerprint so
            different builders/rubrics never cross-read — rule 22, #1018).
        checkpoint_dir: batch-path checkpoint root (REQUIRED for the batch path).
        concurrency_overrides: ``{family: cap}`` overrides for the per-key caps.
        crossover_n: sync/batch routing threshold (Phase 3 calibrates).
        chunk_size: batch sub-batch size (500-2000; default 1000).
        max_concurrent_sub_batches: bounded concurrent batch submissions.
        max_attempts: per-item sync retry budget for terminal/transient errors
            (a 429 does NOT consume an attempt — it has its own budget).
        max_429_retries: per-item 429 (rate-limit) retry budget, SEPARATE from
            ``max_attempts``; exhausting it returns ``category=RESULT_RATE_LIMITED``
            (re-drivable) rather than a terminal ``RESULT_ERROR``.
        force_path: ``"sync"`` / ``"batch"`` to override routing (tests / ops).
        org_keys / async_clients / sync_clients / cache: injection points for
            tests; when None, real keys/clients/cache are built from the env.
        on_route: optional callback receiving the :class:`DispatchRoute`.

    Returns:
        ``{item_id: DispatchResult}`` covering EVERY input item (cache hits,
        successes, and ``error=True`` terminal failures alike).
    """
    if not items:
        return {}

    # Runtime enforcement of the system-role contract (docstring NOTE above):
    # one wrap covers the sync build (_do_one), the batch state-init build,
    # and the batch submit/resubmit build — they all thread this callable.
    build_request = _guarded_build_request(build_request)

    async_clients, sync_clients, owned = _resolve_clients(async_clients, sync_clients, org_keys)
    try:
        return await _dispatch_calls_inner(
            items,
            model=model,
            build_request=build_request,
            parse_response=parse_response,
            deadline=deadline,
            cost_pref=cost_pref,
            cache_dir=cache_dir,
            checkpoint_dir=checkpoint_dir,
            concurrency_overrides=concurrency_overrides,
            crossover_n=crossover_n,
            chunk_size=chunk_size,
            max_concurrent_sub_batches=max_concurrent_sub_batches,
            max_attempts=max_attempts,
            max_429_retries=max_429_retries,
            poll_interval=poll_interval,
            force_path=force_path,
            async_clients=async_clients,
            sync_clients=sync_clients,
            cache=cache,
            on_route=on_route,
            now_fn=now_fn,
        )
    finally:
        # Close ONLY clients this call built (never injected ones) — without
        # this the env-built httpx pools leak (connection-pool leak, review #4).
        if owned:
            await _close_clients(async_clients, sync_clients)


async def _dispatch_calls_inner(
    items: list[DispatchItem],
    *,
    model: str,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    deadline: _dt.datetime | None,
    cost_pref: str,
    cache_dir: Path | None,
    checkpoint_dir: Path | None,
    concurrency_overrides: dict[str, int] | None,
    crossover_n: int,
    chunk_size: int,
    max_concurrent_sub_batches: int,
    max_attempts: int,
    max_429_retries: int,
    poll_interval: float,
    force_path: str | None,
    async_clients: dict[str, Any],
    sync_clients: dict[str, Any],
    cache: JudgeCache | None,
    on_route: Callable[[DispatchRoute], None] | None,
    now_fn: Callable[[], _dt.datetime] | None,
) -> dict[str, DispatchResult]:
    """Routing + execution, with clients already resolved (closing is the caller's job)."""
    org_labels = list(async_clients)

    # Cache check (resume skips already-completed items). build_request here is
    # the GUARDED builder threaded from dispatch_calls (#991) — so the
    # system-role check fires at cache-check time, before any wire call.
    if cache is None and cache_dir is not None:
        cache = JudgeCache(cache_dir)
    results, pending = _split_cached(items, cache, build_request)
    if not pending:
        logger.info("dispatch_calls: all %d items served from cache", len(items))
        return results

    # Route the uncached remainder. finding-4: decide_dispatch_route runs on
    # len(pending) — the UNCACHED remainder, NOT the original N (see module
    # __doc__ item 7 "Routing").
    route = decide_dispatch_route(
        len(pending),
        deadline=deadline,
        cost_pref=cost_pref,
        crossover_n=crossover_n,
        force_path=force_path,
    )
    if on_route is not None:
        on_route(route)
    logger.info("dispatch_calls route: %s", route.render())

    cap = family_concurrency_cap(model_family(model), overrides=concurrency_overrides)

    if route.path == "sync":
        new_results = await _run_sync_path(
            pending,
            build_request=build_request,
            parse_response=parse_response,
            async_clients=async_clients,
            org_labels=org_labels,
            cap=cap,
            max_attempts=max_attempts,
            max_429_retries=max_429_retries,
            cache=cache,
        )
    else:
        if checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required for the batch path (org-aware crash-safe resume). "
                "Pass checkpoint_dir=..., or route to sync via cost_pref/force_path/deadline."
            )
        new_results = await _dispatch_batch(
            pending,
            build_request=build_request,
            parse_response=parse_response,
            org_labels=org_labels,
            sync_clients=sync_clients,
            checkpoint_dir=Path(checkpoint_dir),
            chunk_size=chunk_size,
            max_concurrent_sub_batches=max_concurrent_sub_batches,
            poll_interval=poll_interval,
            now_fn=now_fn,
        )
        _persist_results_to_cache(new_results, pending, cache, build_request)

    results.update(new_results)
    return results


def _resolve_clients(
    async_clients: dict[str, Any] | None,
    sync_clients: dict[str, Any] | None,
    org_keys: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return (async_clients, sync_clients, owned).

    ``owned=True`` when this function BUILT the clients from env keys (the
    caller must close them); ``False`` when both were injected (the caller owns
    their lifecycle). Injecting only one side is unsupported — the dispatcher
    needs both an async (sync-path) and a sync (batch-path) client per org.
    """
    if async_clients is not None and sync_clients is not None:
        return async_clients, sync_clients, False
    keys = org_keys if org_keys is not None else detect_org_keys()
    if not keys:
        raise RuntimeError(
            "No Anthropic org keys found in the environment. Expected at least one of "
            f"{list(ORG_ENV_KEYS.values())}."
        )
    built_async, built_sync = _build_clients(keys)
    return async_clients or built_async, sync_clients or built_sync, True


async def _close_clients(async_clients: dict[str, Any], sync_clients: dict[str, Any]) -> None:
    """Best-effort close of every env-built client (async aclose + sync close)."""
    for client in async_clients.values():
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                logger.debug("async client aclose failed", exc_info=True)
    for client in sync_clients.values():
        close = getattr(client, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                logger.debug("sync client close failed", exc_info=True)


def _split_cached(
    items: list[DispatchItem],
    cache: JudgeCache | None,
    build_request: BuildRequest,
) -> tuple[dict[str, DispatchResult], list[DispatchItem]]:
    """Partition items into (cached results, still-pending items).

    ``build_request`` supplies the built-request fingerprint half of the cache
    key (see the cache-adapter block below) — it runs once per item here.
    """
    results: dict[str, DispatchResult] = {}
    pending: list[DispatchItem] = []
    for it in items:
        cached = _cache_get(cache, it, build_request) if cache is not None else None
        if cached is not None:
            # A cached entry only ever stores a successful (non-error) result
            # today (_persist gates on ``not res.error``), so the RESULT_OK
            # default is harmless; reading ``category`` keeps the round-trip
            # honest if a future change persists errors (Finding 1c).
            results[it.item_id] = DispatchResult(
                it.item_id,
                result=cached.get("result"),
                error=cached.get("error", False),
                reason=cached.get("reason"),
                category=cached.get("category", RESULT_OK),
            )
        else:
            pending.append(it)
    return results, pending


def _persist_results_to_cache(
    results: dict[str, DispatchResult],
    items: list[DispatchItem],
    cache: JudgeCache | None,
    build_request: BuildRequest,
) -> None:
    """Write every non-error result to the cache (no-op when cache is None)."""
    if cache is None:
        return
    item_by_id = {it.item_id: it for it in items}
    for item_id, res in results.items():
        if not res.error and item_id in item_by_id:
            _cache_put(cache, item_by_id[item_id], res, build_request)


async def _run_sync_path(
    pending: list[DispatchItem],
    *,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    async_clients: dict[str, Any],
    org_labels: list[str],
    cap: int,
    max_attempts: int,
    max_429_retries: int,
    cache: JudgeCache | None,
) -> dict[str, DispatchResult]:
    """Run the sync fan-out path, persisting each clean result to the cache."""
    org_states = {label: OrgState(label=label, cap=cap) for label in org_labels}
    item_by_id = {it.item_id: it for it in pending}

    def _persist(res: DispatchResult) -> None:
        if cache is not None and not res.error:
            _cache_put(cache, item_by_id[res.item_id], res, build_request)

    return await _dispatch_sync(
        pending,
        build_request=build_request,
        parse_response=parse_response,
        org_states=org_states,
        async_clients=async_clients,
        max_attempts=max_attempts,
        max_429_retries=max_429_retries,
        on_result=_persist,
    )


# ── Cache adapters (JudgeCache keys on (rubric identity, question, completion)) ─
#
# JudgeCache requires a rubric_key on every read/write (rule 22, #810/#1018).
# This generic adapter derives it from the BUILT request: build_request(item)
# returns the Messages-API params dict, which embeds model + system + the user
# messages — so the rubric enters the key wherever it lives (system prompt or
# user-message template), with zero new public parameters on dispatch_calls.
# We hash item_id (stable) as the "question", the json-serialized payload as
# the "completion", and the built-request fingerprint as the rubric_key.
# Behavior notes: (i) build_request now runs once per item at cache-CHECK time,
# not only for pending items — cheap pure dict assembly for every in-repo
# builder (determinism/re-callability is existing contract: the batch resubmit
# path re-calls it); (ii) the #991 no-system-role ValueError consequently fires
# at cache check, strictly EARLIER — fail-fast-compatible; (iii) a max_tokens /
# param change now busts the cache — deliberate OVER-keying on full request
# params (a miss re-judges; an under-key is the #810 bug; contrast
# batch_judge.rubric_fingerprint, which keys rubric IDENTITY only).


def _cache_key_parts(item: DispatchItem, build_request: BuildRequest) -> tuple[str, str, str]:
    """(item_id, payload_json, request_fingerprint) — rule 22 (#810/#1018).

    The request fingerprint hashes the FULL built params (model + system +
    messages + max_tokens, ...), a superset of the rubric identity, so two
    callers sharing a cache dir with different judges/rubrics can never
    cross-read. Deliberate over-keying: this is full-REQUEST identity, not the
    minimal rubric identity (the adapter cannot distinguish rubric from
    non-rubric params generically).
    """
    try:
        payload_repr = json.dumps(item.payload, sort_keys=True, default=str)
    except TypeError:
        payload_repr = str(item.payload)
    built = json.dumps(build_request(item), sort_keys=True, default=str)
    fp = hashlib.sha256(built.encode()).hexdigest()[:16]
    return item.item_id, payload_repr, fp


def _cache_get(cache: JudgeCache, item: DispatchItem, build_request: BuildRequest) -> dict | None:
    """Adapter read: JudgeCache.get keyed on (built-request fp, item_id, payload)."""
    q, c, fp = _cache_key_parts(item, build_request)
    return cache.get(q, c, rubric_key=fp)


def _cache_put(
    cache: JudgeCache,
    item: DispatchItem,
    res: DispatchResult,
    build_request: BuildRequest,
) -> None:
    """Adapter write: JudgeCache.put keyed on (built-request fp, item_id, payload)."""
    q, c, fp = _cache_key_parts(item, build_request)
    cache.put(
        q,
        c,
        {
            "result": res.result,
            "error": res.error,
            "reason": res.reason,
            "category": res.category,
        },
        rubric_key=fp,
    )


# ── Persisted headroom snapshot + `--status` CLI (v2 cross-session coord) ─────
#
# A best-effort observability side-channel so a SEPARATE process
# (``python -m explore_persona_space.llm.api_dispatch --status``, a planner
# sizing an API workload) can read the last-seen per-org rate-limit headroom
# WITHOUT making a live probe call. Every dispatch that observes rate-limit
# headers (the :meth:`OrgState.note_remaining` path) folds the reading into
# ``~/.task-workflow/api-headroom.json``, throttled to at most one write per
# ``HEADROOM_MIN_WRITE_INTERVAL_S`` so a burst of concurrent calls does not
# hammer the file.
#
# SANCTIONED FAIL-SOFT: :func:`record_headroom_observation` is the ONE place in
# this module where an exception is caught and swallowed. The snapshot is pure
# observability — dispatch correctness never depends on it — so a write failure
# (disk full, permission, a concurrent-writer race) MUST NEVER propagate into
# the live dispatch path. Every OTHER error path in this module fails loud by
# design; this one is the deliberate exception.

HEADROOM_SNAPSHOT_PATH = Path.home() / ".task-workflow" / "api-headroom.json"
HEADROOM_MIN_WRITE_INTERVAL_S = 5.0
# Staleness bands for `--status` (seconds): fresh < 60s, stale < 1h, else very-stale.
HEADROOM_FRESH_MAX_S = 60
HEADROOM_STALE_MAX_S = 3600
_headroom_last_write_monotonic: float = 0.0


def _headroom_observation_from_headers(headers: Any) -> dict[str, int] | None:
    """Extract ``{requests_remaining, tokens_remaining}`` from response headers.

    ``tokens_remaining`` is the MOST-BINDING of the input/output token limiters
    (mirrors :meth:`OrgState.note_remaining`'s min-fraction logic). Returns
    ``None`` when no rate-limit header is present (nothing to record).
    """
    req = _header_int(headers, "anthropic-ratelimit-requests-remaining")
    token_vals = [
        v
        for name in (
            "anthropic-ratelimit-output-tokens-remaining",
            "anthropic-ratelimit-input-tokens-remaining",
        )
        if (v := _header_int(headers, name)) is not None
    ]
    obs: dict[str, int] = {}
    if req is not None:
        obs["requests_remaining"] = req
    if token_vals:
        obs["tokens_remaining"] = min(token_vals)
    return obs or None


def merge_headroom_snapshot(
    snapshot: dict,
    org_label: str,
    model: str,
    observation: dict,
    *,
    observed_at_iso: str,
    writer_pid: int,
) -> dict:
    """Pure merge: fold one observation into the snapshot; return a NEW dict.

    Shape: ``{org_key_alias: {model: {requests_remaining, tokens_remaining,
    observed_at_iso}}, "writer_pid": <int>}``. The newest observation for an
    ``(org, model)`` pair wins. Pure / no I/O so it is unit-tested directly.
    """
    out = dict(snapshot) if isinstance(snapshot, dict) else {}
    prev_org = out.get(org_label)
    org_entry = dict(prev_org) if isinstance(prev_org, dict) else {}
    org_entry[model] = {**observation, "observed_at_iso": observed_at_iso}
    out[org_label] = org_entry
    out["writer_pid"] = writer_pid
    return out


def record_headroom_observation(
    org_label: str,
    model: str,
    headers: Any,
    *,
    path: Path | None = None,
    now_monotonic: float | None = None,
    now_iso: str | None = None,
) -> None:
    """Best-effort, throttled persist of one header observation. NEVER raises.

    See the section header: this is the sanctioned fail-soft. Any exception is
    logged at debug and swallowed so a snapshot-write failure cannot break the
    live dispatch loop that called it.
    """
    global _headroom_last_write_monotonic
    try:
        obs = _headroom_observation_from_headers(headers)
        if obs is None:
            return
        mono = time.monotonic() if now_monotonic is None else now_monotonic
        if mono - _headroom_last_write_monotonic < HEADROOM_MIN_WRITE_INTERVAL_S:
            return
        _headroom_last_write_monotonic = mono
        p = HEADROOM_SNAPSHOT_PATH if path is None else Path(path)
        current: dict = {}
        if p.exists():
            try:
                loaded = json.loads(p.read_text())
                if isinstance(loaded, dict):
                    current = loaded
            except (json.JSONDecodeError, OSError):
                current = {}
        iso = now_iso or _dt.datetime.now(_dt.UTC).isoformat()
        merged = merge_headroom_snapshot(
            current, org_label, model, obs, observed_at_iso=iso, writer_pid=os.getpid()
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(p, merged)
    except Exception:
        logger.debug("headroom snapshot write failed (swallowed)", exc_info=True)


def _staleness_label(observed_at_iso: str, now: _dt.datetime) -> str:
    """``fresh`` (<60s) / ``stale`` (<1h) / ``very-stale`` (>=1h) / ``unknown``."""
    try:
        obs = _dt.datetime.fromisoformat(observed_at_iso)
    except (ValueError, TypeError):
        return "unknown"
    if obs.tzinfo is None:
        obs = obs.replace(tzinfo=_dt.UTC)
    age = (now - obs).total_seconds()
    if age < HEADROOM_FRESH_MAX_S:
        return "fresh"
    if age < HEADROOM_STALE_MAX_S:
        return "stale"
    return "very-stale"


def build_headroom_status_rows(snapshot: dict, *, now: _dt.datetime) -> list[dict]:
    """Flatten a snapshot into per-``(org, model)`` rows with a staleness label.

    Pure — both the CLI text and JSON renderers consume this. Skips the
    top-level ``writer_pid`` key and tolerates a malformed org / model entry
    (skips it rather than raising).
    """
    rows: list[dict] = []
    for org_label, models in sorted(snapshot.items()):
        if org_label == "writer_pid" or not isinstance(models, dict):
            continue
        for model, entry in sorted(models.items()):
            if not isinstance(entry, dict):
                continue
            observed = str(entry.get("observed_at_iso", ""))
            rows.append(
                {
                    "org": org_label,
                    "model": model,
                    "requests_remaining": entry.get("requests_remaining"),
                    "tokens_remaining": entry.get("tokens_remaining"),
                    "observed_at_iso": observed,
                    "staleness": _staleness_label(observed, now),
                }
            )
    return rows


def format_headroom_status_text(rows: list[dict], *, writer_pid: Any = None) -> str:
    """Human-readable per-org/model headroom table with staleness labels."""
    if not rows:
        return "api-headroom: no observations recorded yet."
    lines = ["api-headroom (per-org / per-model last-observed remaining):"]
    for r in rows:
        lines.append(
            f"  [{r['staleness']:>10}] org={r['org']:<10} model={r['model']:<30} "
            f"requests_remaining={r['requests_remaining']} "
            f"tokens_remaining={r['tokens_remaining']} @ {r['observed_at_iso']}"
        )
    if writer_pid is not None:
        lines.append(f"  (last writer pid: {writer_pid})")
    return "\n".join(lines)


def _cmd_status(
    *, path: Path | None = None, as_json: bool = False, now: _dt.datetime | None = None
) -> int:
    p = HEADROOM_SNAPSHOT_PATH if path is None else Path(path)
    now = now or _dt.datetime.now(_dt.UTC)
    if not p.exists():
        if as_json:
            print(json.dumps({"rows": [], "note": "no headroom snapshot file yet"}, indent=2))
        else:
            print(f"api-headroom: no snapshot at {p} yet (no dispatch has observed headers).")
        return 0
    try:
        snapshot = json.loads(p.read_text())
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot is not an object")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        sys.stderr.write(f"api-headroom: unreadable snapshot {p}: {exc}\n")
        return 1
    rows = build_headroom_status_rows(snapshot, now=now)
    writer_pid = snapshot.get("writer_pid")
    if as_json:
        print(json.dumps({"rows": rows, "writer_pid": writer_pid}, indent=2, sort_keys=True))
    else:
        print(format_headroom_status_text(rows, writer_pid=writer_pid))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-org Anthropic dispatcher — headroom status (v2 coordination)."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print the last-observed per-org/model rate-limit headroom (with a "
        "staleness label per entry) from ~/.task-workflow/api-headroom.json.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON (use with --status).")
    args = parser.parse_args(argv)
    if args.status:
        return _cmd_status(as_json=args.json)
    parser.print_help()
    return 0


__all__ = [
    "DEFAULT_BATCH_CHUNK_SIZE",
    "DEFAULT_FAMILY_CONCURRENCY",
    "DEFAULT_MAX_429_RETRIES",
    "FANOUT_SLACK",
    "HEADROOM_SNAPSHOT_PATH",
    "ORG_ENV_KEYS",
    "RESULT_ERROR",
    "RESULT_OK",
    "RESULT_RATE_LIMITED",
    "RESULT_TRANSPORT",
    "SYNC_BATCH_CROSSOVER_N",
    "DispatchItem",
    "DispatchResult",
    "DispatchRoute",
    "OrgState",
    "build_headroom_status_rows",
    "decide_dispatch_route",
    "detect_org_keys",
    "dispatch_calls",
    "family_concurrency_cap",
    "is_transport_exception",
    "main",
    "merge_headroom_snapshot",
    "model_family",
    "record_headroom_observation",
]


if __name__ == "__main__":
    raise SystemExit(main())
