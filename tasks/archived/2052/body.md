---
title: 'workflow-fix: keep-running shields provably-idle pods when owner is alive'
kind: infra
tags:
- wf-fix
- wf-fix-fp:656d59c9e964
created_at: '2026-08-03T20:05:11Z'
has_clean_result: false
origin_prompt: 'Orchestrator direct observation (interactive chat 2026-08-03, user
  asks: ''are you running something here: ogaqj4df250xjh'' -> ''look through tmuxes
  and claude code sessions to see where they came from'' -> ''terminate both''). Two
  idle pod-1739 duplicates billed ~$187 at ~$8/hr for 15-29h. decide_pod_safety_action
  short-circuits keep-running-skip before miss accumulation (autonomous_session_watch.py:3559-3561);
  the #1582 wedged-owner arm keys only on owner liveness + marker gap, so a busy issue''s
  live owner + fresh markers shield orphan pods forever. pod_audit.py already computes
  the idle_gpu signal (all GPUs 0%) and reported util=[0] every pass, but is report-only
  for RUNNING pods.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gap the orchestrator observed
directly while reaping two idle orphan pods on task #1739 (interactive chat,
2026-08-03; provenance recorded in #1739 `epm:progress` v406).

## Goal

Let the watcher's keep-running escalation arm consume the pod-level idleness signal
`pod_audit.py` already computes (`idle_gpu`), so a `keep-running`-tagged, RUNNING,
provably-idle pod escalates even when its owner issue is alive and posting markers.

## Workflow gap

- **Bug observed:** two duplicate pods named `pod-1739` (`ogaqj4df250xjh`,
  `3ysuovu14p879a`; 1x H100 each) billed ~$187 at ~$8.00/hr for 29.4 h and 15.8 h
  respectively while completely idle (GPU 0%, zero workload processes, empty
  `/workspace/logs/`). The watcher SAW them every tick and declined to act.
- **Why it is a workflow gap:** `decide_pod_safety_action`
  (`scripts/autonomous_session_watch.py:3559-3561`) short-circuits
  `status_class == "auto-stop-done" AND keep_running -> ("keep-running-skip", 0)`
  BEFORE any miss accumulation, and the #1582 keep-running wedged-owner escalation
  arm (`decide_keep_running_owner_escalation`, ~L3826+) keys ONLY on owner liveness
  + real-marker progress gap — it has NO pod-level idleness signal. On a busy issue
  both conditions fail permanently: #1739's owner sessions are alive and its marker
  stream never goes quiet (`progress_gap=0.1h` at every tick), so a stale issue-wide
  `keep-running` tag shields unrelated orphan pods indefinitely. Meanwhile the exact
  signal needed is ALREADY computed one tool over: `scripts/pod_audit.py` classifies
  **idle-gpu** — "a RUNNING managed pod whose GPUs ALL read 0% utilization"
  (`PodRow.idle_gpu`, `_probe_gpu_util`, fail-safe `util=unknown` on any SSH/parse
  failure) — and reported `util=[0]` for both pods on every daily pass, but pod_audit
  only auto-terminates EXITED>24h, so RUNNING+idle is report-only there.
  Net: three guards each fail open, and the one measured signal that would have
  caught it is never consumed by the one pass that could act.
- **Confidence (emitter):** high
- verified-at-filing: (2026-08-03)
  - `grep -rn "keep-running-skip" scripts/autonomous_session_watch.py` -> 5 hits in
    1 file; construction site is L3559-3561 (`if status_class == "auto-stop-done":
    if keep_running: return ("keep-running-skip", 0)`), read in context — the
    short-circuit precedes the `missed + 1` accumulation on L3564.
  - `grep -n "def decide_keep_running_owner_escalation" -A 40
    scripts/autonomous_session_watch.py | grep -nE "util|gpu|idle|nvidia"` -> 0 hits
    for `util`/`gpu`/`nvidia`; the only `idle` hits are `min_idle_s` /
    `progress_gap_s`, i.e. MARKER-gap idleness, not pod-level GPU idleness. This is
    an absence-of-signal claim and the 0-hit in-target result IS the evidence.
  - `grep -n "util" scripts/pod_audit.py` -> 8+ hits confirming the signal already
    exists: `idle_gpu: bool  # RUNNING managed pod, util read OK, ALL GPUs at 0%`
    (L135), `_probe_gpu_util` (L308), the `idle-gpu` docstring class (L39-42).
  - Live behavioral evidence, verbatim from
    `logs/autonomous_session_watch/2026-08-03.log`, repeated every tick for ~15 h:
    `issue #1739 pod=3ysuovu14p879a: status=awaiting_promotion class=auto-stop-done
    progress_gap=0.1h missed=0->0 alerted=False action=keep-running-skip`
    (the classifier was CORRECT; the tag discarded the verdict), and from
    `logs/pod_audit/2026-08-03.log`:
    `3ysuovu14p879a  1xNVIDIA H100 80GB HBM3  ~$4.0/hr (estimate)  util=[0]
    'pod-1739'  task #1739 status=followups_running`.
  - Landed-fix check: `git log --oneline --since='7 days ago' --
    scripts/autonomous_session_watch.py scripts/pod_audit.py` reviewed; #1582 is the
    most recent related arm and is the one whose predicate this gap sits outside of
    (its own docs name the adjacent-class exclusion explicitly).
  - Open-sibling dedup: `is_open_workflow_fix_task('scripts/autonomous_session_watch.py',
    '656d59c9e964')` -> None. Related-but-DISTINCT open siblings deliberately not
    deduped against: #2049 (list-ephemeral name-keyed state — a VISIBILITY bug in a
    different file) and #2051 (bootstrap full-clone cost per provision — explains the
    idle pod's disk contents, not why it survived).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from a direct orchestrator observation; sketch below is
indicative only, the planner should decide the real shape.)

    # scripts/autonomous_session_watch.py, keep-running escalation arm (#1582)
    - fires only when owner is WEDGED or ABSENT
    + ALSO fires when the pod is provably idle at POD level:
    +   pod_audit-style idle_gpu == True (all GPUs 0%, util read OK)
    +   AND no workload process
    +   AND sustained >= N consecutive ticks (a point sample is not idleness)
    +   AND the usual fail-toward-keep on any unreadable probe (util=unknown -> keep)
    # ESCALATE-ONLY must remain a hard invariant — the tag is an explicit user
    # override; this arm must never stop or terminate. Pinned by
    # tests/test_autonomous_session_watch_keep_running_owner.py::test_never_stops_or_terminates

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Secondary (signal source, may need a shared helper rather than duplication):
  `scripts/pod_audit.py` (`_probe_gpu_util`, `PodRow.idle_gpu`)
- Docs to keep in sync: `.claude/rules/background-automation.md`
  (§ Keep-running wedged-owner escalation arm — its "accepted residuals" list
  currently states the adjacent classes are out of scope; if this lands, that
  paragraph changes).

## Constraints / invariants

- **ESCALATE-ONLY stays a hard invariant.** The `keep-running` tag is an explicit
  user override; this arm must never stop or terminate a pod. Pinned by
  `tests/test_autonomous_session_watch_keep_running_owner.py::test_never_stops_or_terminates`.
- A point-sample `util=0` is NOT idleness — require sustained consecutive ticks, and
  fail toward keep on any unreadable probe (`util=unknown`), matching pod_audit's
  existing fail-safe.
- Beware the inverse failure: a legitimately-tagged pod mid-provision/bootstrap reads
  0% GPU for minutes. The maturity floor must exceed a normal bootstrap window
  (see #2051 — a full-clone bootstrap alone can idle a GPU ~15 min).
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on
  touched files passes; if `.claude/rules/background-automation.md` changes it stays
  consistent with the code.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 656d59c9e964

Origin (orchestrator direct observation, interactive chat 2026-08-03, verbatim user
ask: "are you running something here: ogaqj4df250xjh" then "look through tmuxes and
claude code sessions to see where they came from" then "terminate both"). Full
forensic record — including the unrecoverable provisioning provenance and the
three-guards-fail-open analysis — is on task #1739 as `epm:progress` v406.
