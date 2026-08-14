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
choice by inheritance, not decision. Related: [[hold-gate-arming-surfacing-property]],
[[watcher-arm-pergrain-state-and-wiring]].
