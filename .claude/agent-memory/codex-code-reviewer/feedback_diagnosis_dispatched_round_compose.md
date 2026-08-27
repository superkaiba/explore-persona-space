---
name: diagnosis-dispatched-round-compose
description: "Diagnosis-dispatched round compose (#2546 r17/v17): round with NO per-round impl marker (dispatched from an epm:progress defect diagnosis) layered on a prior reconciler-overturn of the twin's OWN FAIL — diagnosis envelope replaces the impl envelope, marker-shape/smoke-run-missing declared INVALID, DEFERRED-BY-RECONCILER status line + author-neutrality; range-construction pre-existence attestation (BASE..HEAD name list) beats per-file probes; live phase outran the brief AGAIN (arm advanced one phase mid-compose)"
metadata:
  type: feedback
---

From #2546 r17 compose (sentinel v17, 2026-08-26), layered on
[[record-only-provenance-round-compose]] + the revision-recipe crash-fix shapes:

1. **No-impl-marker round + prior reconciler-overturn compose TOGETHER.** The
   round was dispatched straight from an `epm:progress` defect diagnosis
   (v144) with the implementer report in-session; the PRIOR round was Claude
   PASS / Codex FAIL / reconciler BINDING PASS with the twin's BLOCKER
   severity-DOWNGRADED (`defer-concern --by reconciler`). Compose both
   shapes: (a) `---BEGIN DEFECT-DIAGNOSIS MARKER BODY (epm:progress vNNN)---`
   envelope REPLACES the impl-marker envelope; `marker-shape` +
   `smoke-run-missing` declared INVALID (pre-adjudicated dispatch shape) and
   moved INSIDE the Blocker-tags bracket as a never-emit list; (b) the
   deferred codex row gets a `DEFERRED-BY-RECONCILER` status line + the
   author-neutrality/no-relitigate block naming the downgrade rationale
   (omission-not-falsification) and the surviving open same-finding id;
   re-raise routes same-id at CONCERN grain on NEW in-diff evidence only.
2. **Race re-probe is a script assert, not just a compose-time look:** the
   compose script asserts `max impl version == 16` at RUN time and fails
   loud with "RE-COMPOSE inlining it" if the marker lands mid-compose.
3. **Range-construction pre-existence attestation.** When the brief claims
   lint/test red is pre-existing at BASE, `git diff --name-only BASE..HEAD`
   returning ONLY round files + data commits proves EVERY named red file
   byte-identical at BASE BY CONSTRUCTION — one probe, stronger than
   per-file diffs; state it as composer-STRENGTHENED and still hand the twin
   the cheap re-probe.
4. **Live state outran the brief again** (the record-only entry's point 4,
   now twice on this task): brief said arm 2 p2 RUNNING; events v147 showed
   p2 COMPLETE + p3 dispatched pre-compose. Frame fact with the MATERIAL
   consequence spelled out (the arm is now one phase from the exposure
   window brief item 3 adjudicates).
5. **Brief-as-enumeration for Step 0.71 on a marker-less round:** the
   enumeration home (plan smoke section / marker block) does not exist, so
   the brief's disclosed smoke-conditional branch (--skip-upload) counts as
   enumerated; only an ADDITIONAL undisclosed smoke-conditional downgrade
   FAILs with `smoke-blind-spot-unenumerated`.
6. **Sentinel pinned to the implementer round counter creates a skipped
   review sentinel** (v16 never exists; prior was v15): state the numbering
   map as a frame fact in the prompt AND flag in the return that the posted
   top-level version auto-derives (max+1 = 16) while the head sentinel says
   v17 — the extraction contract keys on the head tag the orchestrator asked
   for.

Compose script: /tmp/codex-2546-v17-compose.py (fail-loud file-presence
verdict; derives every SHA via rev-parse — never hand-typed after the #2330
misattribution class; prompt /tmp/codex-prompt-issue-2546-v17.md, 58.8 KB).
