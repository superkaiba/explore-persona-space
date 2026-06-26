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
   (extends :class:`~explore_persona_space.eval.batch_judge.JudgeCache`) skips
   already-completed items on restart; results are checkpointed with atomic
   temp-file-then-rename writes so a crash / full disk leaves the last good
   checkpoint intact.

6. **Batch path.** For large / cost-sensitive N, reuse ``batch_judge``'s
   ``_chunk_requests`` with SMALL chunks (~1000, NOT 8000) + the ``#663``
   bounded-deadline poll; sub-batches are submitted with ``asyncio.gather``
   under a bounded concurrency cap, with ORG-AWARE resume (each sub-batch
   persists which org it went to so resume re-polls the right org).

7. **Routing.** :func:`decide_dispatch_route` chooses sync-fan-out vs batch by
   N + deadline + cost_pref. :data:`SYNC_BATCH_CROSSOVER_N` is the named
   threshold (Phase 3 will calibrate it from the measured crossover table).

8. **Retries.** Transient errors (timeouts, 529 InternalServerError) retry
   with exponential backoff; a failed item is returned with ``error=True``
   rather than crashing the whole run.

This module does NOT migrate existing callers (Phase 5) — it only adds the new
dispatcher and its tests.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.eval.batch_judge import JudgeCache, _chunk_requests, make_custom_id
from explore_persona_space.llm.anthropic_client import (
    BatchDeadlineExceeded,
    deadline_from_expires_at,
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
    """

    item_id: str
    result: Any = None
    error: bool = False
    reason: str | None = None
    org: str | None = None  # which org served it (sync path); None for cache hits


# Request builder: item -> Messages-API params kwargs (model/max_tokens/messages/...).
BuildRequest = Callable[[DispatchItem], dict]
# Response parser: model text -> per-item result. May raise on a bad parse.
ParseResponse = Callable[[str], Any]


# ── Per-org runtime state (AIMD + headroom) ──────────────────────────────────


# How long an acquirer sleeps before re-checking the gate when the org is at
# its effective concurrency (or inside a retry-after window).
GATE_POLL_INTERVAL = 0.02


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ── Transient-error retry ────────────────────────────────────────────────────


def _is_transient(exc: BaseException) -> bool:
    """True for retryable transient API errors (timeouts, 529, connection).

    529 ``OverloadedError`` is an ``InternalServerError`` subclass in the
    installed SDK (per code-style.md), so catching ``InternalServerError``
    covers it. ``RateLimitError`` (429) is handled SEPARATELY via AIMD, so it
    is NOT in this transient set.
    """
    import anthropic as _anthropic

    transient = (
        _anthropic.APIConnectionError,
        _anthropic.APITimeoutError,
        _anthropic.InternalServerError,  # includes 529 OverloadedError
    )
    return isinstance(exc, transient)


def _is_rate_limit(exc: BaseException) -> bool:
    import anthropic as _anthropic

    return isinstance(exc, _anthropic.RateLimitError)


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


async def _dispatch_sync(
    items: list[DispatchItem],
    *,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    org_states: dict[str, OrgState],
    async_clients: dict[str, Any],
    max_attempts: int,
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
    """
    results: dict[str, DispatchResult] = {}
    labels = list(org_states)
    # Round-robin pointer breaks ties so two equal-headroom orgs share load.
    rr = {"i": 0}

    def _pick_org() -> str:
        # Highest routing headroom; round-robin tie-break.
        best = max(
            range(len(labels)),
            key=lambda k: org_states[labels[(rr["i"] + k) % len(labels)]].routing_headroom(),
        )
        chosen = labels[(rr["i"] + best) % len(labels)]
        rr["i"] = (rr["i"] + 1) % len(labels)
        return chosen

    async def _do_one(item: DispatchItem) -> None:
        params = build_request(item)
        last_reason = "unknown"
        for attempt in range(max_attempts):
            org = _pick_org()
            state = org_states[org]
            client = async_clients[org]
            await state.acquire()
            try:
                raw = await client.messages.with_raw_response.create(**params)
                state.note_remaining(raw.headers)
                msg = raw.parse()
                text = next((b.text for b in msg.content if b.type == "text"), "")
                parsed = parse_response(text)
                state.n_ok += 1
                # Clean call -> additively recover toward the cap.
                if not state.low_headroom():
                    await state.recover()
                res = DispatchResult(item.item_id, result=parsed, org=org)
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
                    last_reason = f"429 (org={org}, attempt {attempt + 1})"
                    continue  # re-pick org + retry
                if _is_transient(exc):
                    last_reason = f"transient {type(exc).__name__} (attempt {attempt + 1})"
                    await asyncio.sleep(1.5**attempt)
                    continue
                # Non-transient (parse error, bad request, etc.) -> terminal.
                last_reason = f"error: {exc}"
                break
            finally:
                await state.release()
        res = DispatchResult(item.item_id, error=True, reason=last_reason, org=None)
        results[item.item_id] = res
        if on_result is not None:
            on_result(res)

    await asyncio.gather(*[_do_one(it) for it in items])
    return results


# ── Batch path (org-aware, checkpointed) ─────────────────────────────────────


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
    """
    if state_path.exists():
        return json.loads(state_path.read_text())

    # Build one request per item, chunk small, assign orgs round-robin.
    cid_for = {it.item_id: make_custom_id(it.item_id) for it in items}
    requests = [{"custom_id": cid_for[it.item_id], "params": build_request(it)} for it in items]
    chunks = _chunk_requests(requests, max_count=chunk_size)
    state = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        sb["status"] = "submitting"
        _atomic_write_json(state_path, state)  # intent BEFORE create
        requests = [
            {"custom_id": cid, "params": build_request(items_by_cid[cid])}
            for cid in sb["custom_ids"]
        ]
        # BATCH_JUDGE_CLIENT_EXEMPT: api_dispatch IS a sanctioned hardened batch client — it
        # reuses batch_judge._chunk_requests (<=8k shards) + deadline_from_expires_at/
        # BatchDeadlineExceeded (bounded expires_at poll) + per-sub-batch org-aware resume by
        # custom_id; routing through judge_completions_batch would lose multi-org fan-out.
        batch = await asyncio.to_thread(client.messages.batches.create, requests=requests)
        try:
            sb["batch_id"] = batch.id
            sb["status"] = "submitted"
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
) -> None:
    """Poll one not-yet-collected sub-batch ONCE on its org; harvest on end."""
    org = sb["org"]
    client = sync_clients[org]
    batch = await asyncio.to_thread(client.messages.batches.retrieve, sb["batch_id"])
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
        final = await asyncio.to_thread(client.messages.batches.retrieve, sb["batch_id"])
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
            scores[item_id] = {"result": None, "error": True, "reason": f"batch_error: {rtype}"}
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
            results[item_id] = DispatchResult(
                item_id,
                result=rec.get("result"),
                error=rec.get("error", False),
                reason=rec.get("reason"),
                org=sb["org"],
            )
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
        parse_response: ``model_text -> result``; may raise on a bad parse
            (caught per item -> ``error=True``).
        deadline: optional wall-clock deadline; a deadline inside the batch 24h
            SLA forces the sync path.
        cost_pref: ``"balanced"`` (default) | ``"cost"`` (prefer 50% batch) |
            ``"latency"`` (prefer sync fan-out).
        cache_dir: per-item content-hash cache root (resume skips cached items).
        checkpoint_dir: batch-path checkpoint root (REQUIRED for the batch path).
        concurrency_overrides: ``{family: cap}`` overrides for the per-key caps.
        crossover_n: sync/batch routing threshold (Phase 3 calibrates).
        chunk_size: batch sub-batch size (500-2000; default 1000).
        max_concurrent_sub_batches: bounded concurrent batch submissions.
        max_attempts: per-item sync retry budget (429s + transient errors).
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

    # Cache check (resume skips already-completed items).
    if cache is None and cache_dir is not None:
        cache = JudgeCache(cache_dir)
    results, pending = _split_cached(items, cache)
    if not pending:
        logger.info("dispatch_calls: all %d items served from cache", len(items))
        return results

    # Route the uncached remainder.
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
        _persist_results_to_cache(new_results, pending, cache)

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
    items: list[DispatchItem], cache: JudgeCache | None
) -> tuple[dict[str, DispatchResult], list[DispatchItem]]:
    """Partition items into (cached results, still-pending items)."""
    results: dict[str, DispatchResult] = {}
    pending: list[DispatchItem] = []
    for it in items:
        cached = _cache_get(cache, it) if cache is not None else None
        if cached is not None:
            results[it.item_id] = DispatchResult(
                it.item_id,
                result=cached.get("result"),
                error=cached.get("error", False),
                reason=cached.get("reason"),
            )
        else:
            pending.append(it)
    return results, pending


def _persist_results_to_cache(
    results: dict[str, DispatchResult],
    items: list[DispatchItem],
    cache: JudgeCache | None,
) -> None:
    """Write every non-error result to the cache (no-op when cache is None)."""
    if cache is None:
        return
    item_by_id = {it.item_id: it for it in items}
    for item_id, res in results.items():
        if not res.error and item_id in item_by_id:
            _cache_put(cache, item_by_id[item_id], res)


async def _run_sync_path(
    pending: list[DispatchItem],
    *,
    build_request: BuildRequest,
    parse_response: ParseResponse,
    async_clients: dict[str, Any],
    org_labels: list[str],
    cap: int,
    max_attempts: int,
    cache: JudgeCache | None,
) -> dict[str, DispatchResult]:
    """Run the sync fan-out path, persisting each clean result to the cache."""
    org_states = {label: OrgState(label=label, cap=cap) for label in org_labels}
    item_by_id = {it.item_id: it for it in pending}

    def _persist(res: DispatchResult) -> None:
        if cache is not None and not res.error:
            _cache_put(cache, item_by_id[res.item_id], res)

    return await _dispatch_sync(
        pending,
        build_request=build_request,
        parse_response=parse_response,
        org_states=org_states,
        async_clients=async_clients,
        max_attempts=max_attempts,
        on_result=_persist,
    )


# ── Cache adapters (JudgeCache keys on (question, completion)) ────────────────
#
# JudgeCache hashes (question, completion). We reuse it generically by hashing
# the item_id (stable) as the "question" and the json-serialized payload as the
# "completion", so two items with the same id+payload share a cache entry.


def _cache_key_parts(item: DispatchItem) -> tuple[str, str]:
    try:
        payload_repr = json.dumps(item.payload, sort_keys=True, default=str)
    except TypeError:
        payload_repr = str(item.payload)
    return item.item_id, payload_repr


def _cache_get(cache: JudgeCache, item: DispatchItem) -> dict | None:
    q, c = _cache_key_parts(item)
    return cache.get(q, c)


def _cache_put(cache: JudgeCache, item: DispatchItem, res: DispatchResult) -> None:
    q, c = _cache_key_parts(item)
    cache.put(q, c, {"result": res.result, "error": res.error, "reason": res.reason})


__all__ = [
    "DEFAULT_BATCH_CHUNK_SIZE",
    "DEFAULT_FAMILY_CONCURRENCY",
    "ORG_ENV_KEYS",
    "SYNC_BATCH_CROSSOVER_N",
    "DispatchItem",
    "DispatchResult",
    "DispatchRoute",
    "OrgState",
    "decide_dispatch_route",
    "detect_org_keys",
    "dispatch_calls",
    "family_concurrency_cap",
    "model_family",
]
