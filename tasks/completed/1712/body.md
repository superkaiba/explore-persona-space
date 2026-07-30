---
title: 'daily-fix: audit landed-but-unfolded rounds'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e786ef2fc8dc
- daily-auto-filed
created_at: '2026-07-26T07:08:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The #1310 round-3 assistant
  test ran on 2026-07-23 and was committed to main, its scope is named verbatim in
  #1639''s own Context row, and yet #1639''s body carries zero assistant-test result
  lines and omits round 3 from its Repro code list; the gap was found only by chance
  two days later.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. A completed, committed analysis
round sat unfolded in an `awaiting_promotion` clean-result for two days and was found
only by chance, while answering an unrelated question.

## Goal

Add an audit pass that flags any task body whose named pending artifact, script, or
commit already exists on `main`, so a landed round cannot sit silently unfolded.

## Workflow gap

- **Bug observed:** `scripts/issue1310_xpersona_assistant_test.py` ran on 2026-07-23
  and was committed (`9e65fe09ad`, *"issue #1310 round-3 assistant-test: artifacts +
  summary + figure (auto-harvest; summary_rc=0 cert_rc=0)"*). #1639's clean-result body
  names that round in its own `**Context:**` row — *"consolidating the #1310
  cross-persona similarity rounds 1-3: cross-character battery, principled re-analysis,
  and the assistant test"* — yet the body contains **zero** lines mentioning the
  assistant test with a result, and its `**Repro:**` code list cites only
  `issue1310_xpersona_similarity.py` @`82d85db5ee` (round 1) and
  `issue1310_xpersona_similarity_v2.py` @`9edaab4fa4` (round 2). Round 3 is absent.
  The gap was discovered on 2026-07-25 by session `63122023` while answering a
  different question, i.e. by luck.
- **Why it is a workflow gap:** the same-issue follow-up loop folds a round's finding
  into the parent body when the loop runs it. A round that lands by another path — a
  chat-initiated inline round, an auto-harvest, a sibling session's commit — has no
  mechanism that notices the parent body never absorbed it. The body meanwhile sits at
  `awaiting_promotion` presenting an incomplete evidence base as complete, which is
  the state a promotion decision is made from.
- **Existing sibling passes make this cheap.** The watcher already runs escalate-only
  observers of exactly this shape (registry-drift, completed-unmerged, root-draft,
  triage-observer, verdict-disagree) — each a read-only scan with a sidecar JSONL and a
  deduped push. This is one more.
- **Confidence (emitter):** high on the incident; medium on the detection predicate —
  "named pending artifact" needs a concrete, low-false-positive form (see below).
- verified-at-filing: absence confirmed —
  `grep -c 'folds here on landing\|unfolded' scripts/autonomous_session_watch.py` →
  **0** (no such pass). Incident facts verified per-target at compose time:
  `task.py view 1639 --json` → `status: awaiting_promotion`; body scanned for
  assistant-test result lines → **0 hits**; `**Repro:**` row read and found to list
  rounds 1–2 only; `git log --oneline -1 9e65fe09ad` resolves to the round-3 commit and
  `scripts/issue1310_xpersona_assistant_test.py` is present on `main`. Landed-fix
  history check `git log --oneline --since='7 days ago' --
  scripts/autonomous_session_watch.py` → the wave touched it via #1668
  (`b173175e15`) and #1681 (`167141479c`); neither adds an unfolded-round audit.
  (2026-07-25)

## Proposed change (refine in planning)

```
+ new escalate-only watcher pass, modelled on the completed-unmerged / registry-drift
+ observers:
+   for each task at awaiting_promotion (or reviewing) with has_clean_result:
+     extract the artifacts/scripts/commits the BODY names as pending or in-scope
+     (the **Context:** origin-prompt scope line and any explicit pending-fold phrase)
+     if a named script/commit/eval_results path EXISTS on main but the body carries
+     no result line for it -> flag
+   sidecar .claude/cache/unfolded-round-events.jsonl + one deduped push per
+   (issue, artifact) episode; NEVER mutates the body or the status
+   kill switch EPM_DISABLE_UNFOLDED_ROUND_PASS=1
```

**The predicate is the hard part** and the planner should scope it deliberately. The
cheap, low-false-positive version keys on an explicit pending-fold PHRASE in the body
(e.g. "folds here on landing", "is running as a follow-up round") whose named artifact
now exists — a small, precise net. The broader version (parse the `**Context:**` scope
line for named rounds and check each has a result section) catches #1639 exactly but
risks flagging every body whose scope prose is loosely worded. Prefer starting narrow
and widening on measured misses; a noisy observer gets muted and then catches nothing.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (a new pass).
- A pin test alongside the sibling observer tests: one positive (a body naming a landed
  artifact with no result line → flagged) and one negative (a body that DOES carry the
  result → silent).
- Read-only: this pass must never `set-body`, never change status, never promote.

## Constraints / invariants

- Escalate-only, deduped per episode, day-capped like the sibling passes — an
  `awaiting_promotion` body can sit for weeks and must not generate a daily push.
- Fail-open: an unparseable body is skipped silently, never flagged.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Related

Folding #1639's specific result (and deciding whether it requalifies that body's
headline) is a scientific-meaning call tracked separately as a `daily-held`
`needs-human` task from this same sweep. This task builds the detector only.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: e786ef2fc8dc
- Source: `/daily` 2026-07-25 transcript sweep, session `63122023` @ 18:07:54Z.
