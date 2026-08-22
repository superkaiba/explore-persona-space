---
name: missing-impl-marker probe checklist
description: Step 5 compose can be dispatched before the implementer posts epm:results/epm:experiment-implementation — run the 5-probe checklist (incl. the #2015 stash-race rescue patch) before failing loud, and make the epm:failure note carry the remedy + verified-good compose inputs
metadata:
  type: feedback
---

Before failing loud on a missing implementation marker (Step 2-pre `test -s`
gate), run ALL of these probes — a single `latest-marker --prefix` miss is not
yet evidence of genuine absence (hit live on #2147 r1, 2026-08-16):

1. **Both prefixes** — `epm:results` (infra/batch/analysis/survey) AND
   `epm:experiment-implementation` (implementers occasionally post the wrong
   kind for the task's path; confirmed live a second time on #2148 r1,
   2026-08-16 — a `kind: infra` wf-fix task posted
   `epm:experiment-implementation v1`; probe 1 resolved it, and the compose
   proceeded with a neutral marker-kind attestation in the prompt so the
   twin judges shape on the inlined body, never the kind naming).
   CAVEAT (#2338 r1, 2026-08-17): a THIRD variant, bare `epm:implementation
   v1`, matches NEITHER standard prefix — only probe 2's full kind listing
   catches it. Never stop at probe 1; fetch the odd kind with its own
   `--prefix epm:implementation` once the listing names it, then proceed
   with the same neutral-attestation compose.
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
5. **Stash-race rescue patch (#2325 r2, 2026-08-16)** — when the brief CITES a
   marker version absent from canonical events.jsonl, check
   `~/.task-workflow/deferred-commits.jsonl` for a matching `post_event` row
   (index.lock `CalledProcessError`) and the `~/.cache/pre-commit/patch*` file
   whose mtime brackets it. The #2015 pre-commit stash race can DESTROY a
   posted append (working tree == HEAD, row gone, file mtime == the deferral
   time; #2325's `epm:results` v3 was lost this way while sibling #2147's v2
   survived the same lock storm). Recovery: extract the `+`-prefixed JSONL
   line from the patch, strip the `+`, jq-validate kind/version/note, verify
   the note is complete (expected sections present), then INLINE it in the
   prompt with an explicit provenance note ("recovered from the #2015 rescue
   patch; its absence from events.jsonl is a known incident — NEVER a
   finding, never marker-shape, never data-access-blocked"). Do NOT restore
   the row yourself (compose-only) — flag the restoration duty LOUDLY in the
   return, naming the patch path + the extracted /tmp row file so the
   orchestrator can re-append + commit before posting the verdict. This beats
   failing loud: the implementer DID post, the body is byte-exact
   recoverable, and a false-absence epm:failure burns the round the other
   way.

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
