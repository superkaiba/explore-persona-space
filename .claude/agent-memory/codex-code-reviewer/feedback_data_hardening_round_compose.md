---
name: data-hardening-round-compose
description: "Rounds whose deliverable IS committed data (eval JSONs hardened into git): digest-only data review + staging spot-check is note-not-block; a brief can bind ## Smoke run / a non-default marker kind on a type-exempt kind:analysis task — route absence per the brief's named tags"
metadata:
  type: feedback
---

Compose deltas for a DATA-HARDENING round — a commit set whose bulk is
result JSONs being committed into `eval_results/` as the deliverable itself
(first hit: #2480 r1, 2026-08-22 — ~1.2 MB of #2394 JSONs + a 467-line
verify script + a 154-line report):

1. **The data commit is the deliverable, not review-load code.** Prompt
   instructs: `git show --stat` per commit + targeted `jq` spot-checks
   (keys, lengths, cited numeric leaves) on committed JSONs; ban paging any
   JSON wholesale (name the biggest offender + its line count); ban the
   unscoped whole-range diff BODY; per-file full reads ONLY for the
   script/report/tiny logs. Extends [[whole-round-unsplit-compose]] item 3.
2. **Out-of-worktree staging spot-check = note, never data-access-blocked.**
   When the brief asks for identity spot-checks against a staging root
   (`/mnt/eps-data/...`), word it best-effort: unreadable-from-sandbox ⇒
   `staging spot-check: BLOCKED — <reason>` as a NOTE. The committed twins
   are in-worktree, so the load-bearing read is never blocked — and
   staging-vs-committed identity is exactly what the round's verify script
   must itself establish (review THAT statically instead).
3. **A brief can BIND gates the task type exempts.** #2480 was
   `kind: analysis` (Step 0.6 exempt, 0.55/0.65/0.67 N/A) yet the brief
   demanded the 4-H3 + `## Smoke run` H2 marker shape AND the implementer
   posted `epm:experiment-implementation` (not the kind-default
   `epm:results`). Follow the brief both ways: fetch by the brief-named
   prefix, bind the `## Smoke run` proof-of-run with genuine-absence →
   `smoke-run-missing`, and state Step 4.6 does-not-bind (it keys on
   `epm:results`) + the type-N/A gates as compose-time facts so the twin
   neither invents nor skips gates. Same follow-the-brief logic as
   [[brief-pinned-sentinel-and-verdict-enum]].
4. **Verify-script rounds arm the hollow-gate sub-check as the CENTRAL
   lens:** the round's own `issue<N>_verify.py` is a verification gate —
   instruct the trace flag → comparison → committed file+key path actually
   opened; asserting on staging copies / recomputed intermediates /
   report-mirroring constants = `hollow-verification-gate`. Also restate
   the brief's contract nuance: loud hard-stop on missing/corrupt/absent-key
   SATISFIES Step 3.5(a); the finding is fail-fast-per-comparison where the
   brief demands COLLECT-ALL.

**Why:** without 1 the twin pages ~50k-line JSONs into context (autocompact
death); without 2 a best-effort convenience check escalates into a false
`data-access-blocked` FAIL; without 3 an adversarial twin either false-FAILs
`marker-shape` on the "wrong" marker kind or skips the smoke proof entirely.

**How to apply:** any brief whose diff-size guidance says committed data
dominates the round, or whose review focus names a verify/audit script over
committed artifacts.
