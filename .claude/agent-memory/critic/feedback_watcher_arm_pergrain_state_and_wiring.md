---
name: watcher-arm-pergrain-state-and-wiring
description: Watcher escalation-arm plans — per-POD episode counters cannot live in the per-ISSUE single-pod_id pod-safety state file (multi-pod interleave resets them every tick); "implementer's choice" alternative wirings must be coverage-checked against the Goal (#2149)
metadata:
  type: feedback
---

Two watcher-arm plan failure modes found in #2149 (pod-grain idleness leg for the #1582 keep-running arm):

1. **Per-entity N-consecutive-tick counters in a single-slot shared state file.** The pod-safety state is per-ISSUE (`_pod_safety_state_path(issue)` -> `pod-safety-<N>.json`) with ONE `pod_id` field; `_save_pod_safety_state`'s kr-field carry is pod_id-KEYED — "a save under a NEW pod_id resets them to their defaults" — and `pod_safety_pass` loops `_process_pod` PER POD against that one file. So on a multi-pod issue (the #1739 shape: 3 concurrent `pod-1739-*`), every sibling's save resets every other pod's episode fields and a >=2-consecutive-tick counter NEVER reaches 2. A plan adding new per-pod fields "via the kr_owner_* pattern" inherits the thrash; single-pod regression tests pass while the arm is inert on the exact incident. **Check:** does the plan's test INTERLEAVE >=2 pods per tick and assert the idle one still fires? Fix = per-(issue,pod) keyed state (pod_id-keyed sub-dict + GC against the running set).

2. **"Implementer's choice, same behavior" wiring alternatives that are NOT equivalent.** Wiring the new leg inside the owner arm's `progress_gap < min_idle_s` early-clear branch evaluates it ONLY on busy tasks; wiring it as a sibling called after the owner arm returns False evaluates it whenever no owner escalation fired (incl. quiet task + live owner). If the Goal says "regardless of task-level marker traffic," wiring A re-introduces the dependence. **Check:** trace each offered wiring's coverage set through the actual branch structure (`_maybe_escalate_keep_running_wedged_owner` L15426: early-clear -> vetoes -> owner decide) before accepting an equivalence claim.

**Why:** both defeat the plan's own acceptance criteria while its tests stay green — the protection-illusion class (infra checklist item A: REVISE when the mechanism would not fire in the very incident it was designed for).

**How to apply:** any plan adding a leg/arm to `autonomous_session_watch.py` pod-safety or extending `_save_pod_safety_state`: (a) read the state file's keying grain vs the arm's claimed grain; (b) coverage-diff any "alternative wiring"; (c) demand a producer-side probe/parse test when evidence comes over SSH stdout (see [[infra-plan-review-checklist]] item C2, #607).

**Recurred (#2283 v2, caught round 2):** the owner-fence-defer arm specced its per-POD `fence_defer_first_ts`/`fence_defer_noted` episode fields "following the kr_owner_* convention" — the singular-field + pod_id-reset shape — so sibling saves on a multi-pod issue would reset the ceiling clock and marker dedup every tick (ceiling never exhausts, channel floods) while all 23 single-pod tests stayed green. Fix demanded: kr_pod/nr_pod sub-dict convention (`_carry_kr_pod` template) + one >=2-pod interleave test row. The trigger phrase to watch for: "pod_id-keyed fields ... following the kr_owner_* convention" on an arm whose trigger predicate is per-pod.
