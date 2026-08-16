---
name: missing-impl-marker probe checklist
description: Step 5 compose can be dispatched before the implementer posts epm:results/epm:experiment-implementation — run the 4-probe checklist before failing loud, and make the epm:failure note carry the remedy + verified-good compose inputs
metadata:
  type: feedback
---

Before failing loud on a missing implementation marker (Step 2-pre `test -s`
gate), run ALL of these probes — a single `latest-marker --prefix` miss is not
yet evidence of genuine absence (hit live on #2147 r1, 2026-08-16):

1. **Both prefixes** — `epm:results` (infra/batch/analysis/survey) AND
   `epm:experiment-implementation` (implementers occasionally post the wrong
   kind for the task's path).
2. **Full events.jsonl kind listing on main** — an implementer may have put
   report CONTENT in an `epm:progress` note. READ the candidate note body: a
   `[surfaced-followups]` / heartbeat / dispatch note is NOT a report (on
   #2147, `epm:progress v4` posted 2 min after the move to `running` was a
   followups-recording note — timestamps adjacent to a status change are not
   report evidence).
3. **Branch-side frozen events** — `git -C <wt> log origin/main..HEAD --
   tasks/` plus a tail of the worktree's `tasks/<cut-status>/<N>/events.jsonl`
   (a mis-routed branch-side post would live only there).
4. **Race re-probe with `date -u`** — the dispatch often lands minutes after
   the implementer finishes; re-run the prefix probe immediately before
   posting `epm:failure` so a marker landing mid-compose is caught.

**Why:** failing loud is correct when absence is real (composing without the
inlined body guarantees a false `marker-shape` FAIL — the #489 class), but a
premature failure marker on a race burns an orchestrator round the other way.

**How to apply:** when absence is confirmed, the `epm:failure` note names
`failure_class: orchestration`, the exact missing marker kind, AND (a) the
remedy (implementer posts, then re-dispatch this composer) and (b) every
compose input already verified good (companion version, plan freshness
verdict, ref resolvability, concerns state) — so the re-dispatch after the
implementer posts is instant, with no re-verification pass. Related:
[[worktree status-folder both directions]], [[9a-ter rounds: no impl marker →
report placeholder]] (the one sanctioned no-marker compose shape — only when
the brief EXPLICITLY says it is a 9a-ter/stage-dispatch round).
