---
name: brief-named-concern-adjudication
description: "When the brief names an open concern to ADJUDICATE (implementer-raised realized-vs-plan deviation) with a disposition menu, compose a REQUIRED `**Open-concern adjudication (<id>):**` verdict line + a same-id CONCERN:: row-disposition rule (#2479 r1)"
metadata:
  type: feedback
---

When the orchestrator's brief names a specific open concern for the review to
ADJUDICATE (first hit: #2479 r1, 2026-08-23 — implementer-raised
`hf-prefix-realized-vs-plan`: realized HF prefix inherited from the reused
parent module vs the plan-named per-issue prefix) and prescribes a
disposition menu, do NOT leave it to the generic Step 0.8 inherit — compose:

1. A REQUIRED verdict-body line `**Open-concern adjudication (<id>):**` with a
   fixed placement (immediately after `## Plan Adherence`) and the brief's
   menu verbatim as (a)/(b)/(c) options — e.g. (a) verdict-affecting
   substantive, (b) CONCERNS with a prescribed concrete fix (name the exact
   edit), (c) acceptable with rationale + the follow-through consequence the
   brief states (both-prefix upload-verifier enumeration).
2. A grounding instruction: grep BOTH literals (realized + plan-named) across
   the diff — which artifact families are written under which prefix, which
   consumers/verify steps read which — plus the plan section, so the
   adjudication is evidence-based, not report-trusted.
3. A row-disposition rule closing the ledger loop: if ANY follow-through duty
   remains (including under (c)-acceptable), re-emit the concern as a
   `CONCERN:: ` row with the SAME id and the adjudicated severity; omit the
   row ONLY when nothing binding remains. Without this, a (c)-adjudication
   silently drops the both-prefix duty the downstream upload-verifier needs.

**Why:** the Step 5c-ter dispatch gate and the upload-verifier read
`concerns.jsonl`, not verdict prose — an adjudication that lives only as a
prose paragraph un-binds the concern the moment the round PASSes (#509
class). Same-id re-emission keeps ledger continuity instead of forking a
duplicate id.

**Already-persisted-rows variant (#2578 r1, 2026-08-25):** when the brief
names concerns the implementer ALREADY persisted pre-marker (raised rows in
concerns.jsonl, not an undispositioned deviation) and asks "did any warrant
BLOCKER?", the menu becomes `warranted-BLOCKER — <file:line>` /
`stands-as-CONCERN` / `resolved-by-this-round`, and the row-disposition
rule INVERTS: re-emit a same-id `CONCERN:: ` row ONLY on a CHANGED
disposition (escalation or a prescribed fix with follow-through) — a
"stands" adjudication must NOT re-emit, or the forwarder appends duplicate
raised events for rows already open. Also inline the rows verbatim as
compose-time facts (the worktree ledger copy is frozen at base and lacks
post-cut rows) and, when the brief attaches a Goal-completeness question to
one concern (an untouched parallel constant site), give the twin the key
DISCRIMINATOR to answer it from code (read-path-of-the-gate vs
write-side/launcher surface), not just the question.

**How to apply:** any brief carrying "open concern to adjudicate" +
concern_id + a disposition menu. Related: [[revision-round-compose-recipe]],
[[concerns-machine-rows-2326]], [[whole-round-unsplit-compose]].
