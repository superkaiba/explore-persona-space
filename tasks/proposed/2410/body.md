---
title: 'Lint: truthy upload URL accepted as canonical completion (3 instances in one
  round) + hoist _require_canonical_upload into hub'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-20T03:52:25Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #823 inconsistent-origin-persona-ladder round: the
  truthy-URL-as-canonical-completion defect appeared in all three drivers of one round,
  the third time with the sibling''s fix visible in the same worktree. Binding reconciler
  specified a conjunction-keyed lint check plus hoisting the guard into orchestrate/hub.py.'
workflow: v1
---
# Truthy upload URL accepted as canonical completion — three instances in one round

## Goal

Close the defect class where a driver treats a truthy return from an
overflow-capable upload helper as proof of a CANONICAL landing, then writes a
completion sentinel naming a canonical path that may not exist. Add a
mechanical check so the class is caught repo-wide instead of per-file, and hoist
the existing per-issue guard into the shared module so drivers import it rather
than reimplement it.

## Evidence — a class, not a coincidence

`src/explore_persona_space/orchestrate/hub.py::_upload_folder_filtered` has a
default-on fallback: on a canonical-repo file-count rejection it re-uploads to
`DEFAULT_OVERFLOW_REPO`, runs its exact-set verification AGAINST THE OVERFLOW
REPO, and returns a truthy overflow URL. A caller testing only `if not url:`
therefore cannot distinguish a canonical landing from an overflow reroute, and
proceeds to declare completion.

The reroute is live-plausible, not hypothetical: `hub.py` pins the verbatim
server refusal for this repo hitting its 1,000,000-file ceiling THIS MONTH, and
the overflow repo carries bulk migration commits dated 2026-08-16, while the
canonical repo also accepted multi-file commits on 2026-08-19. The cap is not
binding at any given instant but demonstrably oscillates.

Three occurrences in a single review round on task #823:

1. `scripts/issue823_ladder_gen.py` — found by a Codex review twin, fixed at
   commit `1c0cacb29e` by adding a local `_require_canonical_upload`.
2. `scripts/issue823_ladder_capture.py` — the SAME shape on the round's PRIMARY
   DELIVERABLE upload. Found only because the reviewer ran a bug-class sweep
   after fixing (1); the instance itself sat in already-reviewed, already-tested
   code. Fixed at `0965ecc5ec`.
3. `scripts/issue823_ladder_fits.py` — the SAME shape again, and
   `_require_canonical_upload` was not even imported despite the sibling
   defining it in the same worktree. The reviewing agent had the fix in front of
   it and still missed the instance.

Occurrence (3) is the argument for mechanization: per-file human/LLM review
demonstrably does not catch this class even with the remedy visible in a
neighbouring file.

## Proposed check — keyed on a CONJUNCTION to avoid false positives

The overflow fallback is a SANCTIONED destination under the upload policy, so a
check that simply flags overflow-capable callers would be wrong and noisy. Flag
only the conjunction:

(i) a call site of an overflow-capable upload helper (`_upload_folder_filtered`,
`_upload`) whose returned URL is used ONLY in a truthiness test, AND
(ii) the same function subsequently writes a completion sentinel / success
record embedding a `path_in_repo` or repo constant NOT derived from the returned
URL.

Compliant escapes (any one clears the site):
- a `_require_canonical_upload(url, repo, path)` call;
- a sentinel that records the REALIZED repo + URL rather than a constant;
- an explicit `# OVERFLOW_UPLOAD_OK: <reason>` waiver comment.

Home: `scripts/workflow_lint.py`. Posture: WARN on the first pass to measure the
existing baseline across the repo, then promote to FAIL once the backlog is
known and cleared. Do not ship straight to FAIL — the baseline is unmeasured and
a fleet-wide FAIL would block unrelated sessions.

## Architectural half — stop the reimplementation

`_require_canonical_upload` currently lives in a PER-ISSUE script
(`scripts/issue823_ladder_gen.py`), and the capture unit had to import it
cross-script to reuse it — which works but is the wrong dependency direction and
is why the fits unit silently did without. HOIST it into
`src/explore_persona_space/orchestrate/hub.py` beside the helper whose failure
mode it guards, so every driver imports one implementation. Keep the per-issue
name as a thin re-export if needed to avoid breaking the two landed callers, or
update them in the same change.

While hoisting, fix a defect the reuse introduced: the raise text is hardcoded
to one phase name ("refusing to report P-Gen complete"), so it names the WRONG
PHASE when invoked from any other caller. A halt message naming the wrong phase
during an incident is actively misleading. Add a phase argument.

## Acceptance criteria

1. The check flags all three #823 sites as they existed pre-fix, and flags none
   of them post-fix.
2. It does NOT flag a legitimate overflow user — a caller that records the
   realized URL, or one carrying the waiver comment.
3. `_require_canonical_upload` lives in `orchestrate/hub.py` with a phase
   argument; both landed #823 callers import it and still pass their suites.
4. A regression test pins the three shapes (pre-fix flagged, post-fix clean,
   legitimate-overflow clean).
5. `uv run python scripts/workflow_lint.py` and the full pytest suite green.

## Provenance

Surfaced during task #823 follow-up round `inconsistent-origin-persona-ladder`.
The check specification and the hoist proposal are the binding reconciler's
answer to an explicit systemic question raised after the third occurrence; the
orchestrator deliberately deferred filing until that specification existed,
rather than filing a vague task on first discovery.
