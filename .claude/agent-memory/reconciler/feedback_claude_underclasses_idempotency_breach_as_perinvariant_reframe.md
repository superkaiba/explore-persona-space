---
name: Claude under-classes a registered idempotency breach by reframing the invariant
description: code-review reconcile — Claude demotes a real "exactly once" breach to a non-blocking minor by swapping the plan's invariant for a weaker "bounded" one; trace the second tick yourself
type: feedback
---

When a plan/spec registers an **idempotency invariant** ("exactly once",
"launch at most one X", "fire once") and Codex FAILs on a path that violates
it, Claude's recurring miss is to **reframe the invariant into a weaker one
the code DOES satisfy** and then call the gap a non-blocking minor — e.g.
"bounded per-tick … acceptable for an infra failover" instead of the plan's
"exactly once". The reframe is the tell. The plan's word governs (Claude
treats the named estimand/invariant as negotiable, the recurring
plan-vs-impl divergence family). Bounded ≠ once.

**Why:** #659 r1. The MF4 GCP→RunPod async failover (`scripts/backend_poll.py`)
required "exactly once". On the sidecar-persistence-failure path the
authoritative `write_handle_sidecar(runpod_handle)` raises and the
best-effort `on_launched` write is swallowed by the router, so the sidecar
STILL holds the GCP handle. The NEXT poll reads the unchanged GCP handle →
`_is_gcp_async_workload_failure` returns True again (nothing changed) →
re-enters `_failover_dead_gcp_to_runpod` → `failover_to_runpod_after_async_workload_crash`
fires a SECOND paid RunPod launch. No durable failure-sentinel/lease guard
between the dispatch block (`backend_poll.py:504`) and the launch call
(`:306`). The code's OWN comment (`:336-343`) named the exact failure mode
("the next tick would ... fire a SECOND RunPod launch — breaching 'exactly
once'") — the added persistence guard only blocks emitting `running`, NOT
the re-launch when persistence itself fails. Real money on the paid lane.
Claude PASSed; Codex FAILed; I sided with Codex → FAIL.

**The committed test ALSO encoded the bug** (Claude trusts green tests over
the contract — the trust-green-tests family): docstring at
`tests/test_backend_poll.py:210-212` said "does NOT fire a second RunPod
launch," but the assertion at `:262` was `len(rp.launches) <=
launches_after_first + 1` → with `launches_after_first==1` that is `2 <= 2`
→ PERMITS exactly the second launch. Docstring and assertion mutually
contradictory. A `<= +1` where the invariant demands `== ` is the smell —
the off-by-one tolerance IS the bug landing. Tighten to
`assert len(rp.launches) == launches_after_first`.

**How to apply:** On any reconcile where Codex FAILs an idempotency / "exactly
once" / "at most one launch" path and Claude waves it as bounded/narrow/
acceptable:
1. Re-read the plan/spec's literal invariant. If it says "exactly once" and
   Claude argued "bounded", that's a reframe, not a defense — the bound is a
   weaker invariant.
2. Trace the SECOND occurrence yourself: does the entry predicate re-fire on
   unchanged state? Is there a DURABLE guard (sentinel file / lease /
   marker) BEFORE the side-effecting call, or only a guard on the success
   emission? A guard that gates the `running`/success return but not the
   re-entry into the launch does NOT enforce once.
3. Quantify the bound but don't let it rescue: even a 1-extra leak that is
   the NORMAL case (double-poll-before-park, not a rare race) on a PAID lane
   breaches the invariant → Real & blocking → FAIL. Persist via raise-concern
   (BLOCKER when the path provably repeats a paid action) per the Step 4
   deferred-production-path duty.
4. Check the test's assertion operator against the invariant verb: "exactly
   once" demands `==`, never `<= +1`. A tolerance that admits the forbidden
   count is the test codifying the bug.
