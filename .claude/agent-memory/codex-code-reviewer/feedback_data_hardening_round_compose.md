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

**r2 delta (#2480 r2, 2026-08-22) — brief-sanctioned verifier runs override
the blanket never-execute:** when the fix-round brief ORDERS "run the
verifier end-to-end + --self-test", compose a scoped sanctioned-runs
carve-out instead of the r1 never-execute block: (a) enumerate the exact
commands (production entrypoint + --self-test only, from worktree root),
with the expected exit codes + the recorded AGGREGATE/PASS lines as the
success oracle; (b) postcondition: `git status --porcelain` over
eval_results/scripts/docs MUST be clean after any run (a dirty tree from a
"stdout-only" verifier = Critical substantive); (c) the never-fabricate
STATIC fallback labeled `STATIC (env unavailable)`, with an explicit
"unrunnable exec env is NOT `data-access-blocked`" line (the code/data
reads still work); (d) offer `python3` as the `uv`-unavailable substitute
only when the script is verified stdlib-only. Also: a brief's "(no
implementer reasoning)" is honored by INLINING the impl marker (spec-
mandatory; Step 0.5 subject + smoke proof) but downgrading it to
claims-not-evidence — three sanctioned uses (shape gate, recorded-run
proof, fix-site index), every closure verdict derived from diff + own
checks. Flag both interpretive calls in the return.
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

**r2 delta (#2480 r2, 2026-08-22) — brief-sanctioned verifier runs override
the blanket never-execute:** when the fix-round brief ORDERS "run the
verifier end-to-end + --self-test", compose a scoped sanctioned-runs
carve-out instead of the r1 never-execute block: (a) enumerate the exact
commands (production entrypoint + --self-test only, from worktree root),
with the expected exit codes + the recorded AGGREGATE/PASS lines as the
success oracle; (b) postcondition: `git status --porcelain` over
eval_results/scripts/docs MUST be clean after any run (a dirty tree from a
"stdout-only" verifier = Critical substantive); (c) the never-fabricate
STATIC fallback labeled `STATIC (env unavailable)`, with an explicit
"unrunnable exec env is NOT `data-access-blocked`" line (the code/data
reads still work); (d) offer `python3` as the `uv`-unavailable substitute
only when the script is verified stdlib-only. Also: a brief's "(no
implementer reasoning)" is honored by INLINING the impl marker (spec-
mandatory; Step 0.5 subject + smoke proof) but downgrading it to
claims-not-evidence — three sanctioned uses (shape gate, recorded-run
proof, fix-site index), every closure verdict derived from diff + own
checks. Flag both interpretive calls in the return.

**Regenerated-fingerprinted-file delta (#2477 r2, 2026-08-22):** when a
fix round REGENERATES a committed data file that a `.gitleaksignore`
fingerprint pins by LINE NUMBER (`path:rule:LINE`), even a metadata-only
top-of-file diff (net ±k lines) shifts the pinned line — the fingerprint
goes stale silently (`.gitleaksignore` itself absent from the round diff).
Compose a bounded spot-check fact: sed-window the region at HEAD to confirm
the flagged text is still non-credential, grade staleness at most Minor
(operational: future edits to that file re-trip the scanner). Also: a
reconciler fix-round closure ledger composes three-tier per #2480-r3 —
binding items (R1-R5, NOT-ADDRESSED = substantive FAIL) / opportunistic
standing recs + addressed CONCERN rows (claims-honesty severity) /
STANDING-OPEN fence for the reconciler-downgraded scoped-out id (no re-FAIL
on severity, no re-emitted row, review-on-merits only if touched).

**r3 deltas (#2480 r3, 2026-08-22):** (a) the reconciler marker KIND
drifted across rounds on the SAME task — r1 posted
`epm:code-review-reconcile`, r2 posted `epm:review-reconcile` — so fetch
the acceptance-contract verdict by enumerating events.jsonl kinds, never by
assuming the prior round's kind (a `--prefix` fetch on the r1 kind returns
the WRONG round's verdict silently). (b) Single-surviving-blocker rounds
compose the closure ledger three-tier: full VERIFIED-ADDRESSED /
NOT-ADDRESSED duty on the surviving blocker (+ its CONCERN rider),
CONFIRMED-UNDISTURBED / REGRESSED rows for the reconciler-fenced closed
blockers (reopen needs NEW evidence from THIS diff), STANDING-OPEN for the
out-of-scope ledger residue. (c) A brief-ordered "PROBE the fixed gate by
constructing the N key sets" composes as the #2146 scratch carve-out made
CONCRETE: /tmp-only driver, importlib-by-path on the worktree script,
in-memory deep-copy mutations of the committed JSON, canonical-still-PASS
as the seventh case, STATIC fallback retained — and the return flags that
dispatch write-mode decides whether the probe arm executes. (d) A v3
marker whose smoke proof moved from a `## Smoke run` H2 into a bold bullet
inside (c) is present-but-imperfect FORM — state the compose-time
observation or the twin false-FAILs `smoke-run-missing` on the missing H2.
