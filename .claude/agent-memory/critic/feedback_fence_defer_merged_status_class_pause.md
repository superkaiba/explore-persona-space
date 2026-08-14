---
name: fence-defer-merged-status-class-pause
description: Watcher shield/exemption arms keyed on a MERGED status class silently inherit the user-pause (on_hold) class — check every new exemption's precedence against explicit operator directives (#2283)
metadata:
  type: feedback
---

When a plan adds a new EXEMPTION/DEFER arm to `decide_pod_safety` (or any
watcher pass) keyed on `status_class == "auto-stop-done"`, that label is a
MERGED class: `POD_SAFETY_AUTO_STOP = AUTO_STOP_DONE | AUTO_STOP_PAUSED`
(watcher :1454; docstring :4113-4116). A new shield therefore silently
covers the #980/#919 USER-PAUSE class too.

**Why:** the pause affordance stops pods BEFORE parking, so on_hold +
RUNNING means the teardown failed — the auto-stop is executing an EXPLICIT
operator directive. A self-posted signal with no newer-than-transition
comparison (e.g. the #2277 `fence_until=` token, whose recency bound is
only its own deadline) can then countermand a NEWER user pause for the
shield's full window. Contrast the two existing exemptions: `keep-running`
is itself an explicit operator tag, and `_task_followup_active` requires a
signal NEWER than the latest done-transition — so a post-pause launch is
already shielded by followup_active, and excluding the paused class from a
new fence-style shield loses NO legitimate case.

**How to apply:** on any plan adding a pod-safety exemption, ask (1) does
the trigger signal carry a newer-than-transition comparison or an operator
provenance? If neither, demand the paused class be excluded (thread the
raw status / an `is_paused` flag past the merged label) or the override be
deliberately pinned with its own truth-table row; (2) check the §5 truth
table for an on_hold row — silence means the plan made the precedence
choice by inheritance, not decision. Validated fix shape (#2283 v2,
round-2 APPROVE): thread the raw `status` into the eligibility computation
and key on SET membership `status in AUTO_STOP_DONE` (never
`status != "on_hold"`) — a future member of either set routes correctly
and a neither-set status fails toward NOT exempting; verified the raw
status is already in scope at `_process_pod`'s exemption call site.
Residual to watch on lazy per-episode state: a wrapper contract "any
failure → False" collides with episode semantics "failed read ⇒ CARRY" —
a bool return can't distinguish evaluated-and-inactive (CLEAR) from
failed (CARRY); demand a tri-state or caller-side exception split.
Round-3 closure notes (#2283 v3, APPROVE): per-pod episode state on a
per-ISSUE watcher state file MUST use the pod_id-keyed sub-dict
convention (`_carry_kr_pod`/`_carry_nr_pod` at watcher :14352/:14371 —
the singular `kr_owner_*` family resets on `prev.pod_id != pod_id` at
:14578, so sibling-pod saves void any singular-family ceiling); the
entry-CLEAR idiom within that contract is `keep_ids=set(sub)-{pod_id}`
(precedent :13736). Revision-round trap to grep for: converting a bool
wrapper to tri-state leaves STALE return-value sentences in untouched
sections (v3 §4.4 kill-switch bullet still said "returns False" while
three other mentions said None ⇒ CARRY) — grep every mention of the
wrapper's return value across the whole plan on such a conversion.
Related: [[hold-gate-arming-surfacing-property]],
[[watcher-arm-pergrain-state-and-wiring]].
