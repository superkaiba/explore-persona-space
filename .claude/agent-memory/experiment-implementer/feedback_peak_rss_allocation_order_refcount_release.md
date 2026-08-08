---
name: peak-rss-allocation-order-refcount-release
description: Peak-RSS fixes bind on ALLOCATION order, not consumption order; prove a release with sys.getrefcount()==2, not a log line (#1739 r4)
metadata:
  type: feedback
---

Two lessons from the #1739 round-4 two-batch ridge fix, both cases where the obvious fix was wrong (team-lead recorded them as durable, 2026-08-06):

1. **A peak-memory fix targets when arrays are ALLOCATED, not when they are consumed.** Reordering solve batches over already-allocated arrays cuts nothing: if all four `(Ly, n, d)` arrays exist before any solve, the peak is set at allocation time regardless of batch order. The effective fix DEFERS the later batch's allocation (`mp = apply_map(...)` built only after batch 1 solves and releases `mp_shuf`), capping concurrent residents. Team-lead had approved the ineffective batch-order-only version; the allocation-deferral correction is what actually reduced the hall-OOD peak 79.5 → 59.6 GB.

2. **A release-engagement signal must prove the RELEASE, not the branch.** A log line proves the code path executed; `sys.getrefcount(x) == 2` (local name + getrefcount arg; one `gc.collect()` retry for solver-teardown cycles, then hard `RuntimeError`) proves the memory was actually freed before the next batch allocates. This is the right shape for any fix whose claim is "X is released before Y starts".

**Why:** batch order LOOKS like the lever for peak RSS and reviewers approve it; allocation order is the binding constraint. And "the release branch ran" is not "the memory is free" — a stray reference (job list not cleared, a closure, an eval-matrix tuple) keeps the array resident silently.

**How to apply:** when fixing peak RSS in any multi-batch fit/solve pipeline, first map WHERE each large array is materialized and defer allocations past the releases they must not overlap; then guard the release with a refcount assert (fail-loud), not just an INFO line. Sibling: [[free-helper-caller-binding-drain-wait]] (`del` in a callee frees nothing — rebind in the owning frame).
