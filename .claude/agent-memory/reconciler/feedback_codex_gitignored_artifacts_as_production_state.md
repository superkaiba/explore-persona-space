---
name: Codex treats gitignored worktree artifacts as production state
description: Codex FAILs an idempotency/data-completeness guard because hazardous smoke artifacts exist at canonical paths in the LOCAL worktree, without checking whether those artifacts can reach the pod or whether the canonical flow can create them
type: feedback
---

When Codex's Critical is "stale smoke/dev artifacts at canonical paths bypass the
production build" (e.g. `_check_data_built()` existence-only early-return accepting
48-row smoke mixes, task #543 round-2), run the reachability walk before believing FAIL:

1. **Git propagation:** is the artifact dir gitignored + untracked (`git ls-files <dir>`
   empty)? Pods clone fresh — gitignored local artifacts never ride to the pod.
2. **Canonical-flow creation:** can the encoded pod flow itself WRITE the hazardous
   state? Check the driver's child invocations for the `--smoke` flag and whether the
   non-smoke path fail-louds on smoke-sized inputs (#543: non-smoke mix build asserts
   3000-row bank classes; bank smoke writes only bank files the check never consults).
3. **Pre-existing:** `git show <r1-sha>:<file>` — if the weak check has identical
   semantics in the already-PASSed prior round, the "regression" framing is wrong
   (companion: "Codex litigates pre-existing in round N").

If all three say the bypass needs operator error outside the encoded flow → Real but
non-blocking; PASS with a HARD standing rec for the ~3-line manifest/row-count
hardening (and Codex's proposed regression test). The fail-fast rule targets swallowed
failures on the production path, not insufficiently strict idempotency guards against
states the canonical flow cannot produce.

**Why:** Origin task #543 round-2: Codex FAIL (`data-completeness-smoke-artifacts-
bypass-full-build`) vs Claude PASS-with-minor. All three checks favored Claude;
reconciled PASS. Repeat #570 round-2 (same script family): Codex Critical
"cache-skip defeats revision pin" (`ensure_*_local` `if not PATH.exists():` skip,
`_issue543_common.py:400`) — `data/` gitignored, fresh pod's FIRST fetch of every
file is pinned (all #570-path call sites pass `HUB_DATA_REPO_REVISION_570`), skip
pattern pre-exists on main WITHOUT any pin; branch strictly improves trunk.
Reconciled PASS + persisted the hardening as a `--by reconciler` CONCERN
(revision-stamped cache / sidecar stamp) so the ledger, not prose, carries it.

**How to apply:** Any round where Codex's blocker depends on a file STATE in the
worktree (not a code defect per se), especially "would train on tiny/stale data".

**Committed-evidence-snapshot variant (#601 r7):** Codex Critical'd "committed
phase0_gate.json is the obsolete schema, pass:false" as a launch blocker. The
file state was real, but the plan's §D delta 4 MANDATED committing exactly that
file "from pod-601 ... so §A's numbers are auditable" — §A's evidence IS the
pre-amendment failed gate (the plan cites the file + pod commit by name; its
point is the old gate "can never pass"). Production never consumes the git copy:
the phase-0 driver UNCONDITIONALLY recomputes + rewrites the pod-side gate
(schema 2, pure CPU over persisted tables) even on a fully skip-valid re-run,
and the launcher's gate check reads the pod path. Fast checks: (a) did the plan
order this artifact committed as EVIDENCE (then old-schema = correct)? (b) does
any production reader consume the git copy vs a regenerated pod-side copy?
(c) does the driver's gate-write run unconditionally after the skip pools?
PASS + persisted a CONCERN to re-commit the regenerated gate post-run.
