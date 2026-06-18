---
name: Codex treats gitignored worktree artifacts as production state
description: Codex FAILs idempotency/data-completeness guards because hazardous smoke artifacts exist at canonical LOCAL paths; run the reachability walk — git propagation, canonical-flow creation, pre-existing semantics — before believing FAIL.
type: feedback
---

**Rule:** when Codex's Critical is "stale smoke/dev artifacts at canonical paths bypass the production build", run three checks:
1. **Git propagation:** is the dir gitignored + untracked (`git ls-files <dir>` empty)? Pods clone fresh — gitignored local artifacts never reach the pod.
2. **Canonical-flow creation:** can the encoded pod flow itself WRITE the hazardous state? Check the driver's `--smoke` plumbing and whether the non-smoke path fail-louds on smoke-sized inputs.
3. **Pre-existing:** `git show <r1-sha>:<file>` — identical semantics in an already-PASSed round defeats the "regression" framing.
All three say the bypass needs operator error outside the encoded flow → Real-but-non-blocking; PASS + HARD standing rec for the ~3-line manifest/row-count hardening, persisted as a `--by reconciler` CONCERN so the ledger carries it. The fail-fast rule targets swallowed failures on the production path, not idempotency guards against states the canonical flow cannot produce.

**Incidents:** #543 r2 (origin — existence-only `_check_data_built` vs 48-row smoke mixes; all checks favored PASS); #570 r2 (cache-skip "defeats revision pin": `data/` gitignored, fresh pod's first fetch is pinned, skip pattern pre-exists on main unpinned — branch strictly improves trunk).

**Committed-evidence-snapshot variant (#601 r7):** "committed phase0_gate.json is obsolete schema, pass:false" — the plan MANDATED committing exactly that file as pre-amendment EVIDENCE; production never consumes the git copy (driver unconditionally recomputes + rewrites the pod-side gate; launcher reads the pod path). Fast checks: (a) did the plan order the artifact committed as evidence? (b) does any production reader consume the git copy? (c) does the gate-write run unconditionally after the skip pools? PASS + CONCERN to re-commit post-run.

Companion: [[feedback_codex_litigates_pre_existing_in_round_n]].
