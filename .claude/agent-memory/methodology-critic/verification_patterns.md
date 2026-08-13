---
name: verification-patterns
description: Recurring ground-truth verification patterns from issue #2162 rounds 1-2 — diff-narrowing under-reports, marker-vs-artifact conflicts, dispatcher threshold semantics
metadata:
  type: project
---

Verification patterns that caught real errors in v2 report methodology reviews (#2162):

- **Diff-narrowing claims under-report executable changes.** When a writer claims a
  cross-SHA diff is "only a constant plus docstrings", run the FULL `git diff <shaA>..<shaB> -- <file>`
  and look for `+` lines with control flow (`if/else`, `append`, new dict keys) — #2162 round 2:
  the claimed constant-only diff also added a new guard failure branch
  (`failed.append("state_sanity")`) + 2 report fields. The load-bearing check is per-cited-constant
  byte-identity at both SHAs, which can be true even when the "docstrings-only" characterization is false.
- **Progress-marker prose can contradict the committed gate JSON it summarizes.** #2162: a marker
  said "query-rubric PASS (30)" while the committed `pilot_gate_report.json` says `n_total_draws: 90`.
  The committed artifact is authoritative; a report tracing to the JSON is correct even when a marker disagrees.
- **Judge-dispatch routing claims need the log's threshold line.** `effective_threshold =
  max(1, threshold_base × otpm/400k)` (`eval/judge_dispatch.py`, base 2000); dispatch log lines carry
  `otpm=<n> (probed|assumed), effective=<n> | path=batch|sync`. Universal claims like "sync waves routed
  sync because the probe raised the threshold" are false for sub-base waves (N < 2000 routes sync
  under any threshold) — check the N distribution of `path=sync` lines before accepting.
- **Wave-count claims:** `judge_summary.json` `waves` is a dict keyed `<rubric>.<stage>`;
  filter `coherence.*` out for "non-coherence wave" counts; `n_items` is the per-wave denominator
  (dispatch-log N can be lower due to cache hits, e.g. 42,120 items → N=39,793 submitted).
- **#2162 local staging quirk:** HF raw-completions files are staged (untracked) under the worktree's
  `data/issue_2162/judge_inputs/issue2162_ctxinfo/...` — usable to confirm a worked-example row traces
  (per #922: never FAIL evidence, but confirms the trace; HF identity stays with upload-verifier).
- **A count coincidence narrated as a temporal mechanism needs an ORDINAL check.** #2162 round 3:
  "count(effective=2000) == count(batch) per log ⇒ earliest dispatches ran pre-probe and batched" was
  refuted by decision ordinals (grid batch decisions at #1,#20,#23,#38,#42,#44,#82 of 87 — scattered).
  The real mechanism was in `judge_dispatch.py`: the OTPM probe is PER-DISPATCH and SKIPPED when
  `n_items >= threshold_base * 2` (assumed 400k ⇒ effective 2000 ⇒ batch), so assumed⇔batch is a
  size-keyed structural identity carrying zero temporal information. Check: read the probe/skip
  call-site before accepting any "pre/post-probe" story; then grep log line ORDER, not just counts.
- **Check for a newer pushed commit superseding the review pin's evidence artifact.** #2162 round 3:
  the report cited `routing_evidence.json` at pin X while `git ls-remote origin` showed a tip one
  commit ahead whose sole change CORRECTED that artifact (adding `superseded_readings` refuting the
  report's own sentence). One `git ls-remote` + `git log <pin>..<remote-tip> -- <artifact>` catches it.
- **"Constant appears in diff only as context" claims: mind the `@@` hunk-header trailer.** A constant
  name can occur in a diff solely as the `@@ ... @@ <section line>` function-context trailer (git's
  section heading, not a body line). Grep counts it as "in the diff"; it is neither added, removed,
  nor a body context line. #2162 round 4: `GATE_OFFTARGET_REL_MAX`'s sole occurrence was the trailer —
  the load-bearing check stays per-constant assignment byte-identity at both SHAs (all 14 held).
- **"Parent modules imported, never edited" claims need per-module blob identity between the
  parent pin and the round pin** (`git rev-parse <pin>:<path>` pairs) — #2162 ladder fold: the
  round's own impl commit said "injection-gate seams in parent" and `scripts/issue2162_run.py`
  differed across pins (default-preserving keyword seams), while the PLAN's narrower claim
  ("nothing under src/ modified") stayed true. The draft's generalization was the defect.
- **Motivation restatements of plan-diagnosed defects over-generalize — trace each defect to the
  plan's diff table / original diagnostic marker.** #2162 ladder fold: plan said "the 3-cycle
  never ran `plain→pirate`" (v2 of the parent cell IS plain, so plain→butler DID run); the draft
  generalized to "never ran the plain-to-persona INSTALL direction ... only rotations among
  persona values" — false, and internally contradicted by the draft's own adjacent claim that
  the butler-install direction got a pirate donor.
- **`git show --stat` TRUNCATES long paths (`.../experiments/issue2162/x.py`), so grepping stat
  output for `^src/` silently misses files.** For any "touched only files X,Y" / "nothing under
  src/ modified" claim, use `git log --name-only --format="== %h %s" <base>..HEAD` and grep the
  untruncated paths. Also attribute cross-pin blob diffs: a module differing between parent pin
  and round tip may be MAINLINE drift (`git log <pin>..HEAD -- <file>` names the commit), not a
  round modification — only branch-only commits (origin/main..HEAD) count against a "round
  touched nothing" claim. (#2162 ladder round 2.)
- **#2162 routing saga CLOSED (round 4 PASS).** Final correct mechanism, verified against
  `judge_dispatch.py:1629` + the pinned `routing_evidence.json` (434c84f5ae): probe SKIPPED at
  n_items >= 2×base (4,000) ⇒ assumed 400k ⇒ effective 2,000 ⇒ batch; below 4,000 probe finds 2M ⇒
  effective 10,000 ⇒ sync. Boundary separates realized dispatches perfectly (min batch 5,037 /
  max sync 3,360; 9 batch, 171 sync = 156 below-base + 15 above). Re-derive from the artifact's
  per-log `batch_dispatches.N_values` / `sync_dispatches` blocks, never the summary prose alone.
