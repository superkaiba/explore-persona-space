---
title: 'api_dispatch fast-follows: 429-exhaustion mislabel + unbounded fan-out + pick-time
  org binding (from #682 code review)'
kind: infra
tags: []
created_at: '2026-06-27T02:58:13Z'
has_clean_result: false
parent_id: 682
origin_prompt: 'Code-reviewer fast-follows from task #682''s PASS verdict on api_dispatch.py
  (Phase 4)'
---
## Goal

Address the three fast-follow findings from the code-reviewer's verdict on
the multi-org API dispatcher (`src/explore_persona_space/llm/api_dispatch.py`,
task #682). The reviewer issued a PASS verdict overall — none of these
findings corrupts data or bypasses a FIRM § 1b operating rule, so they did
not gate the Phase 5 migration — but each is real and should be fixed before
the dispatcher takes a high-volume production workload.

## Findings to address

### 1. major — 429-exhaustion mislabels a retryable item as terminal `error=True`

**Where:** `src/explore_persona_space/llm/api_dispatch.py:583-586, 596` —
inside `_do_one`'s retry loop.

**What:** when an item hits a 429 on its FINAL `max_attempts` attempt, the
for-loop falls out and the item is recorded as
`DispatchResult(error=True, reason=f"429 (org=..., attempt N)")`. A rate-limit
is transient (the dispatcher's AIMD will eventually catch up); marking the
item terminal silently drops it from the result set and the caller sees a
permanent failure for what is purely backpressure.

The current tests do not catch this: the storm test
(`test_429_storm_drops_realized_in_flight`) gives `max_attempts=4` and a
guaranteed success on attempt 2, so exhaustion never fires.

**Fix:**
- Distinguish `reason="rate_limited_exhausted"` from the generic
  `error: <exc>` reason so callers can re-drive on rate-limit-only failures
  without crashing the whole pipeline.
- (Optional, larger) Give 429 its own retry budget separate from
  `max_attempts` — under a sustained storm the AIMD-bounded retries should
  not consume the same budget as terminal/transient errors.
- Add a test that exercises exhaustion under all-429s and asserts the
  distinguishable reason.

### 2. minor — unbounded coroutine fan-out in the sync path

**Where:** `src/explore_persona_space/llm/api_dispatch.py:601` —
`asyncio.gather(*[_do_one(it) for it in items])`.

**What:** the sync path spawns ONE coroutine per pending item. At the
planned N=100k sync ceiling (the Phase 3 table's high-volume Sonnet 4.5
sync configuration) that is 100k coroutines each busy-polling `acquire()`
every `GATE_POLL_INTERVAL=0.02s` while blocked — real wakeup/CPU cost +
memory footprint far above the ~300 concurrent calls the per-key caps
intend.

**Fix:** bound the fan-out with an outer worker pool / semaphore sized to
`cap * n_orgs` so the number of *live* coroutines tracks the actual
concurrency target, not the total queue depth.

### 3. minor — org binding is fixed at pick-time, not at slot-availability

**Where:** `src/explore_persona_space/llm/api_dispatch.py:559-562` —
`_pick_org()` runs before the blocking `acquire()`.

**What:** under burst, all coroutines pick by headroom up-front and then
queue on their chosen org's gate even when a sibling org frees a slot
first. Headroom routing therefore degrades to round-robin-ish under load
rather than true least-loaded dispatch.

**Fix:** re-pick if the chosen org's gate does not admit within a short
poll window, so newly-freed slots actually pull work.

### 4. nit — routing decides on `len(pending)`, not the original N

**Where:** `src/explore_persona_space/llm/api_dispatch.py:1043-1049` —
`decide_dispatch_route(len(pending), ...)`.

**What:** a large job mostly served from cache re-routes the small
uncached remainder to sync. Defensible (fewer items remain), but a
resumed 1M-item batch with 5k uncached items will silently go sync.

**Fix:** add a one-line docstring note documenting this behavior so callers
aware of the cache vs uncached split know what to expect.

## Acceptance

- All 4 findings closed (or explicitly waived with rationale).
- Existing `tests/test_api_dispatch.py` stays green (43 tests).
- One new test per finding 1-3 that would have caught the regression.
- The dispatcher's `__doc__` carries the docstring note for finding 4.

## Provenance

- Reviewer verdict: PASS on #682 (the dispatcher), with these as fast-follows.
- Reviewer's full report: see `epm:code-review` marker on task #682.
- Originating task: #682 "Finish API throughput project".
- Origin prompt: "Code-review src/explore_persona_space/llm/api_dispatch.py
  by spawning the code-reviewer agent ..." (task #682 body, item C).
