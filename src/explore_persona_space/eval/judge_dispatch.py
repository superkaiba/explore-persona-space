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
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.utils import parse_judge_json

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# Item: same 4-tuple already used by batch_judge.
JudgeItem = tuple[str, str, str, str]  # (custom_id, question, completion, user_msg)

# Routing constants (user-decided, plan §11; configurable per call).
DEFAULT_THRESHOLD_BASE = 2_000
DEFAULT_SUB_BATCH_SIZE = 10_000
OTPM_DIVISOR = 400_000  # Tier-4 Sonnet 4.x output-tokens-per-minute
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MAX_TOKENS = 256
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
) -> dict:
    """Build Messages-API params with cache_control on the shared rubric block.

    ttl="1h" for batch requests (out-of-order execution outlives the 5m
    default); ttl=None or "5m" -> ephemeral default (5m). Returns the kwargs
    dict for ``messages.create`` / the ``params`` member of a batch Request.
    """
    sys_block: dict = {"type": "text", "text": judge_system_prompt}
    if ttl is not None and ttl != "5m":
        sys_block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    else:
        sys_block["cache_control"] = {"type": "ephemeral"}
    return {
        "model": judge_model,
        "max_tokens": max_tokens,
        "system": [sys_block],
        "messages": [{"role": "user", "content": user_msg}],
    }


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
) -> tuple[dict[str, dict], list[str], list[str]]:
    """Stream one ended batch's results; join on custom_id (order is not guaranteed).

    Returns (scores, errored_custom_ids, expired_custom_ids). errored/expired
    get error dicts in `scores` too (overwritten if a retry later succeeds);
    canceled surfaces as an error dict with no retry (user-initiated).
    """
    scores: dict[str, dict] = {}
    errored: list[str] = []
    expired: list[str] = []
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        rtype = result.result.type
        if rtype == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                "",
            )
            parsed = parse_judge_json(text, None)
            scores[cid] = parsed if parsed is not None else error_dict_factory("parse_error")
        elif rtype == "errored":
            errored.append(cid)
            scores[cid] = error_dict_factory("batch_error: errored")
        elif rtype == "expired":
            expired.append(cid)
            scores[cid] = error_dict_factory("batch_error: expired")
        else:  # canceled (or unknown): surface, never retry
            scores[cid] = error_dict_factory(f"batch_error: {rtype}")
    return scores, errored, expired


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
) -> dict[str, dict]:
    """Semaphore-bounded AsyncAnthropic judging; per-item error dicts, never raises per item."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(custom_id: str, user_msg: str) -> tuple[str, dict]:
        async with semaphore:
            try:
                params = _build_params(
                    judge_model, judge_system_prompt, user_msg, max_tokens, ttl="5m"
                )
                result = await client.messages.create(**params)
                text = next((b.text for b in result.content if b.type == "text"), "")
                parsed = parse_judge_json(text, None)
                if parsed is None:
                    return custom_id, error_dict_factory("parse_error")
                return custom_id, parsed
            except Exception as e:  # per-item capture is the legacy contract
                return custom_id, error_dict_factory(f"error: {e}")

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


# ── Batch path ───────────────────────────────────────────────────────────────


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
) -> tuple[dict[str, dict], list[str], Path, dict]:
    """Submit/resume + poll + collect all sub-batches for one dispatch.

    Returns (scores, retry_candidate_custom_ids, dispatch_dir, state). All
    waits are ``await asyncio.sleep`` so an enclosing event loop is never
    blocked for the batch's lifetime.
    """
    fingerprint = _compute_fingerprint(items, judge_model, judge_system_prompt, max_tokens)
    dispatch_dir = Path(checkpoint_dir) / f"dispatch_{fingerprint}"
    items_map = {cid: {"question": q, "completion": c, "user_msg": u} for cid, q, c, u in items}
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

    # Phase 1: submission (intent record BEFORE create; batch_id right after).
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
        if sb["status"] == "pending":
            sb["status"] = "submitting"
            _atomic_write_json(state_path, state)
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
            batch = client.messages.batches.create(requests=requests)
            sb["batch_id"] = batch.id
            sb["status"] = "submitted"
            sb["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _atomic_write_json(state_path, state)
            logger.info(
                "Sub-batch %d/%d submitted as %s (%d requests)",
                sb["index"] + 1,
                len(state["sub_batches"]),
                batch.id,
                sb["n_requests"],
            )

    # Phase 2: round-robin polling; harvest each sub-batch the moment it ends.
    scores: dict[str, dict] = {}
    retry_candidates: list[str] = []

    def _load_collected(sb: dict) -> None:
        payload = json.loads((dispatch_dir / f"results_{sb['batch_id']}.json").read_text())
        scores.update(payload["scores"])
        retry_candidates.extend(payload["errored_ids"])
        retry_candidates.extend(payload["expired_ids"])

    for sb in state["sub_batches"]:
        if sb["status"] == "collected":
            _load_collected(sb)

    current_interval = poll_interval
    max_poll_interval = max(poll_interval, 120.0)
    while any(sb["status"] != "collected" for sb in state["sub_batches"]):
        for sb in state["sub_batches"]:
            if sb["status"] == "collected":
                continue
            batch = client.messages.batches.retrieve(sb["batch_id"])
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                logger.info(
                    "Batch %s: processing=%s succeeded=%s errored=%s",
                    sb["batch_id"],
                    counts.processing,
                    counts.succeeded,
                    counts.errored,
                )
            if batch.processing_status == "ended":
                sb_scores, errored, expired = _collect_batch_results(
                    client, sb["batch_id"], error_dict_factory
                )
                _atomic_write_json(
                    dispatch_dir / f"results_{sb['batch_id']}.json",
                    {"scores": sb_scores, "errored_ids": errored, "expired_ids": expired},
                )
                sb["status"] = "collected"
                _atomic_write_json(state_path, state)
                _load_collected(sb)
        if all(sb["status"] == "collected" for sb in state["sub_batches"]):
            break
        await asyncio.sleep(current_interval)
        current_interval = min(current_interval * 1.5, max_poll_interval)

    return scores, retry_candidates, dispatch_dir, state


# ── Core dispatch ────────────────────────────────────────────────────────────


async def dispatch_judge_items_async(
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
    if decision.path == "sync":
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
    )
    state_path = dispatch_dir / "state.json"

    # Step 5: errored/expired retry — ONCE, through the same threshold routing
    # (small straggler sets go sync — faster, consistent with the speed goal).
    retry_state = state.get("retry")
    if retry_state and retry_state.get("status") == "done" and retry_state.get("results_merged"):
        retry_results = json.loads((dispatch_dir / "results_retry.json").read_text())
        scores.update(retry_results)
    elif retry_candidates and not _is_retry:
        items_map = {cid: (q, c, u) for cid, q, c, u in items}
        retry_ids = sorted(set(retry_candidates))
        state["retry"] = {"status": "pending", "custom_ids": retry_ids, "results_merged": False}
        _atomic_write_json(state_path, state)
        retry_items: list[JudgeItem] = [(cid, *items_map[cid]) for cid in retry_ids]
        logger.info("Retrying %d errored/expired requests (once)", len(retry_ids))
        state["retry"]["status"] = "submitting"
        _atomic_write_json(state_path, state)
        retry_results = await dispatch_judge_items_async(
            retry_items,
            judge_model=judge_model,
            judge_system_prompt=judge_system_prompt,
            max_tokens=max_tokens,
            threshold_base=threshold_base,
            sub_batch_size=sub_batch_size,
            force_sync=False,
            dry_run=False,
            max_concurrent=max_concurrent,
            checkpoint_dir=dispatch_dir / "retry",
            poll_interval=poll_interval,
            error_dict_factory=error_dict_factory,
            sync_client=sync_client,
            batch_client=client_b,
            _is_retry=True,
        )
        _atomic_write_json(dispatch_dir / "results_retry.json", retry_results)
        state["retry"] = {"status": "done", "custom_ids": retry_ids, "results_merged": True}
        _atomic_write_json(state_path, state)
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
    "probe_otpm_limit",
]

if __name__ == "__main__":
    _cli()
