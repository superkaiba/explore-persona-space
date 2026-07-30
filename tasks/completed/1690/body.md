---
title: 'daily-fix: verified-at-filing: 3 new probe clauses'
kind: infra
tags:
- wf-fix
- wf-fix-fp:96b748d25c4c
- daily-auto-filed
created_at: '2026-07-26T06:58:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Three of the 2026-07-24
  /daily filings carried premises their clarifiers refuted: #1667 asserted no failover
  marker was posted when epm:progress v146 existed, #1669 named the watcher as the
  fix surface when the real surface was backend_poll plus the runpod launcher and
  the handle writer, and #1680 asserted a park was lost when a routed record had correctly
  suppressed it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. This is a **filing-quality**
defect in `/daily` itself: three of the tasks filed by the 2026-07-24 wave carried
premises their own clarifiers refuted the next morning, each burning clarifier or
plan-revision time before the session could start on real work.

## Goal

Extend the `verified-at-filing:` mandate in `.claude/rules/workflow-fix-on-bug.md`
§ Body-file template with three new binding clauses — marker-existence, call-hop
target tracing, and suppression-predicate — so a filing may not assert a missing
marker, a fix surface, or a lost park without running the compose-time probe that
would refute it.

## Workflow gap

Three refuted premises from one wave (all three tasks still merged, but each paid a
correction round first):

1. **Marker-existence claim, unverified — #1667** (session `203baf55`, clarifier @
   2026-07-25T06:51:56Z). The filed body's claim (b) asserted no failover marker had
   been posted on #1586's events. The clarifier found it: `epm:progress` v146 at
   05:42:00Z carrying the `[autonomous_session_watch:runpod-noport-wedge-failover]`
   sentinel. The real defect was a pre-dispatch external-marker triage MISS by the
   owning session. Half the task's proposed change collapsed to verify-only.
   Clarifier verbatim: *"CORRECTION to the filed body: claim (b) 'no marker was posted
   on #1586's events' is FALSE — the failover marker WAS posted (epm:progress v146,
   05:42:00Z …)"*.
2. **Wrong primary target file — #1669** (session `7457e1a3`, clarifier @
   2026-07-25T08:40:34Z). The filed body named `scripts/autonomous_session_watch.py`.
   The actual fix surface was `scripts/backend_poll.py`
   (`_runspec_from_runpod_handle`) + `src/explore_persona_space/backends/runpod.py`
   (launcher render) + the handle writer in `backends/issue_dispatch.py`; the watcher
   is only the CALLER. The shipped diff touched 13 files, **none of them the named
   target**. This also weakens dedup, whose key is `(target_file, fingerprint)`.
3. **"A park was lost" claim, unverified — #1680** (session `5277f92c`, clarifier @
   2026-07-25T13:18:32Z). The filed bug-(b) hypothesis was that the #1642 parked
   candidate had been lost by the Step C sweep. The clarifier found the park was
   **correctly suppressed** by a routed record (`origin_candidate_ts` exact match,
   fp-less primary key). The real gap was counter opacity. The session re-scoped
   mid-flight and shipped a different fix than the one filed.

Why the existing mandate did not catch these: today's clauses (a)/(a')/(b)/(c)/(d)/(e)
cover pattern presence/absence in FILES, relocation, landed-fix context, commit-SHA
resolution, and artifact-state mutation. None covers (i) an assertion about a MARKER's
existence in an events stream, (ii) whether the named `target_file` is the site that
CONSTRUCTS the wrong value versus a caller that merely propagates it, or (iii) whether
a record the filer believes was dropped was in fact suppressed by a documented
predicate. All three are one cheap probe away at compose time.

- **Confidence (emitter):** high — three independent same-wave incidents, each with a
  clarifier correction quoted verbatim from the task's own events.
- verified-at-filing: absence of the three clauses confirmed in the named target —
  `grep -c 'marker-existence\|call-hop\|suppression-predicate' .claude/rules/workflow-fix-on-bug.md`
  → **0**; the existing `verified-at-filing:` clause list in § Body-file template
  enumerates (a) per-target confirmation, (a') semantic probe for text-matching
  guards, (b) relocation grep, (c) context consistency, (d) sha-resolution, (e)
  artifact-state mutation — and stops there. Incident evidence read from the three
  tasks' own `events.jsonl` clarifier markers (quoted above), not from recall.
  Landed-fix history check
  `git log --oneline --since='7 days ago' -- .claude/rules/workflow-fix-on-bug.md` →
  the 2026-07-25 wave landed #1677 (unverified-premise labeling, `591cd93cff`) and
  #1680 (Step C suppression, `51d2e343a0`); neither adds these three clauses —
  #1677 governs how to LABEL an unverifiable claim, which is the complement of
  requiring a probe for a claim that IS verifiable. (2026-07-25)

## Proposed change (refine in planning)

Three new clauses in § Body-file template, in the existing (a)…(e) register:

```
(f) marker-existence — a claim that "no marker / no record was posted" on task #M is
    verified at compose time with
      uv run python scripts/task.py view <M> --json | jq -r '.events[] | ...'
    scanning for the marker kind AND the sentinel/phrase the claim denies, recorded
    like the grep evidence. A found marker refutes the claim: re-scope before filing.

(g) call-hop target tracing — before fixing target_file, trace the failing behavior
    ONE call-hop past the observed symptom: name the site that CONSTRUCTS the wrong
    value, not the caller that consumes it. Record both. Re-run the dedup fingerprint
    against the corrected target (the dedup key is (target_file, fingerprint), so a
    mis-target silently weakens dedup too).

(h) suppression-predicate — a claim that a candidate/park/record was DROPPED or LOST
    is verified by enumerating the documented suppression predicates the owning tool
    applies (for Step C: the routed-record scan, the verbatim-fp key, the #1680
    origin_candidate_ts fallback). A correctly-suppressed record refutes the claim.
```

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md` (§ Body-file template
  `verified-at-filing:` clause list, and the § Anti-patterns table — each existing
  clause has a matching anti-pattern row; add three).
- `.claude/skills/daily/SKILL.md` route-2 block references the mandate by clause; check
  whether it enumerates clauses inline and needs the same three added.
- Keep the clauses cheap: each must be ONE bounded probe. A clause that costs a
  multi-minute investigation at compose time will be skipped under load, which is how
  the mandate degrades.

## Constraints / invariants

- Workflow-surface only.
- Do NOT weaken the existing fail-open posture: an ambiguous probe result means FILE
  and note the ambiguity, never silently skip (a wrongly-skipped filing has no dedup
  tag, so the bug can vanish entirely).
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
- fingerprint: 96b748d25c4c
- Source: `/daily` 2026-07-25 transcript sweep, sessions `203baf55` (#1667),
  `7457e1a3` (#1669), `5277f92c` (#1680).
