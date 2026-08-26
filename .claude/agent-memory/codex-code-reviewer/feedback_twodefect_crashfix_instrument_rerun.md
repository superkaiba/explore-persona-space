---
name: twodefect-crashfix-instrument-rerun
description: "Two-defect crash-fix compose (#2546 r14/v13): composer RE-RUNS the round's own gates instead of attesting marker claims — no-flags lint (attribution matched: FAIL(6), zero on round files; runs >400s, size the timeout), selector --map-files (takes a PATH-LIST FILE, not source paths; matched the claimed 7-file set), the new test file (11/11); the ONE irreproducible marker count (7-vs-6 exit-1 sites) is pre-adjudicated under the record-accuracy concern id, never a fresh finding; covariate-None-exclusion focus composes as consumer-sweep + coercion-shape checklist; (set -e; fn) subshell focus composes as a 6-point errexit checklist"
metadata:
  type: feedback
---

From #2546 r14 compose (review sentinel v13, 2026-08-26), layered on
[[postcap-crashfix-round-compose]] + [[postpass-delta-round-compose]]:

1. **Re-run the round's own instruments; don't attest the marker's numbers.**
   The brief said "verify rather than assume the implementer's attribution" —
   Codex can't run lint/pytest, so the COMPOSER runs them and inlines results
   with offender provenance: no-flags `workflow_lint.py` (rc=1 `FAIL (6
   error(s))`, all six named untouched files — matched the marker exactly;
   NOTE: the run exceeds a 400s timeout, budget ~600s or background it),
   `select_step9c_tests.py --map-files` (matched the claimed 7-file hit set +
   the disclosed .sh-floor WARN), `pytest <new test file>` (11/11, 2s), ruff
   on the round .py files, `bash -n` on the .sh. **CLI trap:** `--map-files`
   takes ONE newline-delimited PATH-LIST file (`git diff --name-only >
   /tmp/files.txt`), not source paths — passing a source file errors with a
   #1613 message.
2. **An irreproducible marker count gets pre-adjudicated, not raised.** The
   marker claimed "7 internal fail-loud exit 1 sites" in publish_results_git;
   composer counted 6 (with line anchors). Hand both numbers to the twin
   routed under the task's existing record-accuracy concern id
   (`marker-v9-pin-sweep-violator-list-inaccurate`, CONCERN grain, same-id) —
   the --stat-vs-numstat framing stays a frame fact; only the count the
   composer positively cannot reproduce is a candidate re-raise.
3. **Covariate-None-exclusion focus (correct=None must not score incorrect)
   composes as a consumer-sweep demand:** hand the twin (a) the is-None-FIRST
   label branch + the `k != "unknown"` denominator expression with anchors,
   (b) the composer's own zero-reader grep over the fit/figure chain, (c) the
   four coercion shapes to hunt (`if not x`, `sum()`, `Counter` over raw
   values, JSON null-as-falsy), (d) the denominator-reduced-vs-numerator-
   shrunk distinction, and (e) any PRE-EXISTING defense-in-depth guard
   (exact_match_correct returns None on None gold, present at parent) framed
   as an honesty adjudication — is the explicit path redundant or
   load-bearing? — never hidden.
4. **`( set -e; fn )` subshell focus = a 6-point checklist:** inner errexit is
   LIVE because the subshell is NOT in a tested context (the `if ! (...)`
   trap is what the implementer avoided — verify, don't trust); explicit exit
   sites land as rc; bare-command failures also kill the subshell; pipefail
   survives the parent's `set +e`; partial-push gates (push-verify + ls-tree)
   byte-unchanged; the wrapped function byte-untouched (hunk ranges); the
   process-substitution rc-invisible residual is pre-existing ⇒ note-grade.
   Also make the twin trace emit_signal's OWN failure mode inside the branch
   (a failed sentinel write must not convert the branch into a silent abort).
5. **Fleet-state verbs in the constraints block must match the brief's
   snapshot, not the prior round's:** r13's "arm 1 MID-RUN" became r14's
   "arm 1 CRASHED-idle / arm 3 LIVE on the PRE-fix commit heading into the
   same assert" — the severity-consequence line updates BOTH directions (a
   FAIL holds the relaunch; a waved-through defect poisons every arm's
   covariate).

**How to apply:** any crash-fix round pairing a data-condition fix with an
observability fix, and any compose where the brief orders verification of
implementer-attributed gate results. Compose script:
/tmp/codex-2546-r14-compose.py (fresh-write, COMPOSE-OK sentinel,
expected-count stale-token sweep for deliberately-named out-of-scope tokens).
