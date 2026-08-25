---
name: byte-oracle-discharge-round-compose
description: Fix round whose payload IS test oracles discharging a reconciler byte-identity blocker — inline the reconciler record as the contract envelope, split duties into oracle-authenticity vs old-form-fidelity, pin the old form to <payload-sha>~1 reads, and compose the text-writer classification as a decidable both-way trace
metadata:
  type: feedback
---

From #2336 r6 (2026-08-24), the single-blocker fix round after the r5
binding `epm:review-reconcile v3` FAIL (`a5-byte-identity-coverage`): a
test-only commit landing old-vs-new byte oracles (jsonl/text byte-identity,
npz load-equality) + a rider NIT pin.

1. **The reconciler record IS the round contract — inline it verbatim** in a
   `---BEGIN/END ROUND-5 RECONCILER RECORD---` envelope and read the
   discharge elements off its "Blocker list for the fix round" paragraph as
   a numbered element list (the v3 record already quotes the r1 rider
   verbatim, so one envelope carries both). Per-element rulings get a
   `**Discharge ruling:**` header line + Plan Adherence rows keyed to the
   ELEMENTS, not plan sections.
2. **Two distinct hollow-oracle failure modes get two duties.**
   Oracle-authenticity (does the NEW side exercise the real migrated code —
   real imports verified; a sanctioned reproduce-verbatim form needs its
   compensating live-source anchor assessed for load-bearing-ness AND
   block-presence-vs-block-is-the-write residual) vs old-form fidelity (is
   the OLD side genuinely pre-migration — pin reads to
   `git show <payload-sha>~1:<path>` with composer-verified line frames; the
   tautology to prevent is an "old form" copied from the new form). Hand
   both the exact `git show` commands; offered-TEST hollowness routes as
   `substantive`, not `hollow-verification-gate` (that tag is for production
   gates).
3. **Compose the contested writer-classification as a decidable both-way
   trace** (the r4 deviation-trace pattern): the marker classified
   `write_jsonl_sharded` (a `write_text` call on jsonl content) as the TEXT
   instance while the reconciler's arming evidence named a plain-text writer
   in another file. Quote the A5 row + both writers' code; accepted =>
   discharged (note the un-oracled writer as residue); rejected =>
   NOT-ADDRESSED core element => substantive FAIL. Never pre-resolve.
4. **Asymmetric closure fences for the two claimed-addressed ids:** the
   reconciler-re-raised BLOCKER's NOT-ADDRESSED = substantive FAIL (the fix
   IS the round); the rider NIT's NOT-ADDRESSED re-raises at NIT severity
   only — but a FALSE closure claim on either takes the ordinary substantive
   bar (Rule 9/13 fabricated-coverage family).
5. **Composer re-runs the marker's own pytest command** (36 tests, ~66s)
   when the round's whole payload is tests — converts "implementer claims N
   passed" into a composer-attested fact Codex (no uv env) can lean on;
   byte-surface questions get the stdlib-probe carve-out (`python3 -c` on
   json/os.path/pathlib only, STATIC fallback) with explicit precedence over
   the never-execute bullet.

**How to apply:** any fix round whose diff is test oracles discharging a
recorded byte-identity / load-equality acceptance row (#2336 batches 3-5
will re-land this shape, incl. the .pt spot check at the first torch.save
batch). Related: [[single-blocker-fix-round-deviation-trace]],
[[respawn-two-record-batch-compose]], [[gate-block remedy round compose]].
---
name: byte-oracle-discharge-round-compose
description: Fix round whose payload IS test oracles discharging a reconciler byte-identity blocker — inline the reconciler record as the contract envelope, split duties into oracle-authenticity vs old-form-fidelity, pin the old form to <payload-sha>~1 reads, and compose the text-writer classification as a decidable both-way trace
metadata:
  type: feedback
---

From #2336 r6 (2026-08-24), the single-blocker fix round after the r5
binding `epm:review-reconcile v3` FAIL (`a5-byte-identity-coverage`): a
test-only commit landing old-vs-new byte oracles (jsonl/text byte-identity,
npz load-equality) + a rider NIT pin.

1. **The reconciler record IS the round contract — inline it verbatim** in a
   `---BEGIN/END ROUND-5 RECONCILER RECORD---` envelope and read the
   discharge elements off its "Blocker list for the fix round" paragraph as
   a numbered element list (the v3 record already quotes the r1 rider
   verbatim, so one envelope carries both). Per-element rulings get a
   `**Discharge ruling:**` header line + Plan Adherence rows keyed to the
   ELEMENTS, not plan sections.
2. **Two distinct hollow-oracle failure modes get two duties.**
   Oracle-authenticity (does the NEW side exercise the real migrated code —
   real imports verified; a sanctioned reproduce-verbatim form needs its
   compensating live-source anchor assessed for load-bearing-ness AND
   block-presence-vs-block-is-the-write residual) vs old-form fidelity (is
   the OLD side genuinely pre-migration — pin reads to
   `git show <payload-sha>~1:<path>` with composer-verified line frames; the
   tautology to prevent is an "old form" copied from the new form). Hand
   both the exact `git show` commands; offered-TEST hollowness routes as
   `substantive`, not `hollow-verification-gate` (that tag is for production
   gates).
3. **Compose the contested writer-classification as a decidable both-way
   trace** (the r4 deviation-trace pattern): the marker classified
   `write_jsonl_sharded` (a `write_text` call on jsonl content) as the TEXT
   instance while the reconciler's arming evidence named a plain-text writer
   in another file. Quote the A5 row + both writers' code; accepted =>
   discharged (note the un-oracled writer as residue); rejected =>
   NOT-ADDRESSED core element => substantive FAIL. Never pre-resolve.
4. **Asymmetric closure fences for the two claimed-addressed ids:** the
   reconciler-re-raised BLOCKER's NOT-ADDRESSED = substantive FAIL (the fix
   IS the round); the rider NIT's NOT-ADDRESSED re-raises at NIT severity
   only — but a FALSE closure claim on either takes the ordinary substantive
   bar (Rule 9/13 fabricated-coverage family).
5. **Composer re-runs the marker's own pytest command** (36 tests, ~66s)
   when the round's whole payload is tests — converts "implementer claims N
   passed" into a composer-attested fact Codex (no uv env) can lean on;
   byte-surface questions get the stdlib-probe carve-out (`python3 -c` on
   json/os.path/pathlib only, STATIC fallback) with explicit precedence over
   the never-execute bullet.

**How to apply:** any fix round whose diff is test oracles discharging a
recorded byte-identity / load-equality acceptance row (#2336 batches 3-5
will re-land this shape, incl. the .pt spot check at the first torch.save
batch). Related: [[single-blocker-fix-round-deviation-trace]],
[[respawn-two-record-batch-compose]], [[gate-block remedy round compose]].

**Second-generation addendum (#2336 r7, 2026-08-24)** — the fix-of-the-fix
round (r6 FAIL+FAIL on one overlapping residual: a FALSE marker scoping
clause + the .pt oracle it waived):

6. **The bar shifts to the FAIL+FAIL union bodies + the ORIGINAL r1 rider
   record** — inline all three verbatim (two verdict envelopes + the
   r1 reconciler envelope; strip the codex output's trailing session-ID
   lines). The intermediate r5 reconciler (v3) needs no own envelope: the
   r6 verdicts quote its operative content. Discharge elements come from
   the union Fix fields (both agreed), numbered E1-E5.
7. **Composer independently re-derives the corrected factual claim** the
   round exists to fix (here: 15 torch.save sites/8 files in the batch
   commit — one awk/grep on `git show <payload-sha>`) and attests the
   count + per-file breakdown; Codex re-derives with the same one command.
   Falsity-withdrawal accuracy is its own duty element: explicit
   withdrawal sentence + no "binds at a future batch" residue framing.
8. **Drift-tolerant HEAD pin:** write "= HEAD at compose time; further
   out-of-scope sync commits may land above it — not a finding", never a
   bare "= HEAD" (sync commits land continuously on long-lived worktrees;
   a moved HEAD otherwise reads as a contradiction of an attested fact).
9. **Brief-cited main-tip SHA may be superseded at compose time** — re-run
   the divergence probe at CURRENT origin/main, state both (probe verdict
   unchanged ⇒ compose normally, flag the drift in the return).
10. **Settled-vs-open ledger splits 3 ways now:** claimed-addressed-this-
    round (full duties) / VERIFIED-ADDRESSED-at-prior-round (one
    undisturbed-check status line, never re-adjudicated) / open
    (status lines). torch is named NOT-stdlib for the probe carve-out.
---
name: byte-oracle-discharge-round-compose
description: Fix round whose payload IS test oracles discharging a reconciler byte-identity blocker — inline the reconciler record as the contract envelope, split duties into oracle-authenticity vs old-form-fidelity, pin the old form to <payload-sha>~1 reads, and compose the text-writer classification as a decidable both-way trace
metadata:
  type: feedback
---

From #2336 r6 (2026-08-24), the single-blocker fix round after the r5
binding `epm:review-reconcile v3` FAIL (`a5-byte-identity-coverage`): a
test-only commit landing old-vs-new byte oracles (jsonl/text byte-identity,
npz load-equality) + a rider NIT pin.

1. **The reconciler record IS the round contract — inline it verbatim** in a
   `---BEGIN/END ROUND-5 RECONCILER RECORD---` envelope and read the
   discharge elements off its "Blocker list for the fix round" paragraph as
   a numbered element list (the v3 record already quotes the r1 rider
   verbatim, so one envelope carries both). Per-element rulings get a
   `**Discharge ruling:**` header line + Plan Adherence rows keyed to the
   ELEMENTS, not plan sections.
2. **Two distinct hollow-oracle failure modes get two duties.**
   Oracle-authenticity (does the NEW side exercise the real migrated code —
   real imports verified; a sanctioned reproduce-verbatim form needs its
   compensating live-source anchor assessed for load-bearing-ness AND
   block-presence-vs-block-is-the-write residual) vs old-form fidelity (is
   the OLD side genuinely pre-migration — pin reads to
   `git show <payload-sha>~1:<path>` with composer-verified line frames; the
   tautology to prevent is an "old form" copied from the new form). Hand
   both the exact `git show` commands; offered-TEST hollowness routes as
   `substantive`, not `hollow-verification-gate` (that tag is for production
   gates).
3. **Compose the contested writer-classification as a decidable both-way
   trace** (the r4 deviation-trace pattern): the marker classified
   `write_jsonl_sharded` (a `write_text` call on jsonl content) as the TEXT
   instance while the reconciler's arming evidence named a plain-text writer
   in another file. Quote the A5 row + both writers' code; accepted =>
   discharged (note the un-oracled writer as residue); rejected =>
   NOT-ADDRESSED core element => substantive FAIL. Never pre-resolve.
4. **Asymmetric closure fences for the two claimed-addressed ids:** the
   reconciler-re-raised BLOCKER's NOT-ADDRESSED = substantive FAIL (the fix
   IS the round); the rider NIT's NOT-ADDRESSED re-raises at NIT severity
   only — but a FALSE closure claim on either takes the ordinary substantive
   bar (Rule 9/13 fabricated-coverage family).
5. **Composer re-runs the marker's own pytest command** (36 tests, ~66s)
   when the round's whole payload is tests — converts "implementer claims N
   passed" into a composer-attested fact Codex (no uv env) can lean on;
   byte-surface questions get the stdlib-probe carve-out (`python3 -c` on
   json/os.path/pathlib only, STATIC fallback) with explicit precedence over
   the never-execute bullet.

**How to apply:** any fix round whose diff is test oracles discharging a
recorded byte-identity / load-equality acceptance row (#2336 batches 3-5
will re-land this shape, incl. the .pt spot check at the first torch.save
batch). Related: [[single-blocker-fix-round-deviation-trace]],
[[respawn-two-record-batch-compose]], [[gate-block remedy round compose]].
---
name: byte-oracle-discharge-round-compose
description: Fix round whose payload IS test oracles discharging a reconciler byte-identity blocker — inline the reconciler record as the contract envelope, split duties into oracle-authenticity vs old-form-fidelity, pin the old form to <payload-sha>~1 reads, and compose the text-writer classification as a decidable both-way trace
metadata:
  type: feedback
---

From #2336 r6 (2026-08-24), the single-blocker fix round after the r5
binding `epm:review-reconcile v3` FAIL (`a5-byte-identity-coverage`): a
test-only commit landing old-vs-new byte oracles (jsonl/text byte-identity,
npz load-equality) + a rider NIT pin.

1. **The reconciler record IS the round contract — inline it verbatim** in a
   `---BEGIN/END ROUND-5 RECONCILER RECORD---` envelope and read the
   discharge elements off its "Blocker list for the fix round" paragraph as
   a numbered element list (the v3 record already quotes the r1 rider
   verbatim, so one envelope carries both). Per-element rulings get a
   `**Discharge ruling:**` header line + Plan Adherence rows keyed to the
   ELEMENTS, not plan sections.
2. **Two distinct hollow-oracle failure modes get two duties.**
   Oracle-authenticity (does the NEW side exercise the real migrated code —
   real imports verified; a sanctioned reproduce-verbatim form needs its
   compensating live-source anchor assessed for load-bearing-ness AND
   block-presence-vs-block-is-the-write residual) vs old-form fidelity (is
   the OLD side genuinely pre-migration — pin reads to
   `git show <payload-sha>~1:<path>` with composer-verified line frames; the
   tautology to prevent is an "old form" copied from the new form). Hand
   both the exact `git show` commands; offered-TEST hollowness routes as
   `substantive`, not `hollow-verification-gate` (that tag is for production
   gates).
3. **Compose the contested writer-classification as a decidable both-way
   trace** (the r4 deviation-trace pattern): the marker classified
   `write_jsonl_sharded` (a `write_text` call on jsonl content) as the TEXT
   instance while the reconciler's arming evidence named a plain-text writer
   in another file. Quote the A5 row + both writers' code; accepted =>
   discharged (note the un-oracled writer as residue); rejected =>
   NOT-ADDRESSED core element => substantive FAIL. Never pre-resolve.
4. **Asymmetric closure fences for the two claimed-addressed ids:** the
   reconciler-re-raised BLOCKER's NOT-ADDRESSED = substantive FAIL (the fix
   IS the round); the rider NIT's NOT-ADDRESSED re-raises at NIT severity
   only — but a FALSE closure claim on either takes the ordinary substantive
   bar (Rule 9/13 fabricated-coverage family).
5. **Composer re-runs the marker's own pytest command** (36 tests, ~66s)
   when the round's whole payload is tests — converts "implementer claims N
   passed" into a composer-attested fact Codex (no uv env) can lean on;
   byte-surface questions get the stdlib-probe carve-out (`python3 -c` on
   json/os.path/pathlib only, STATIC fallback) with explicit precedence over
   the never-execute bullet.

**How to apply:** any fix round whose diff is test oracles discharging a
recorded byte-identity / load-equality acceptance row (#2336 batches 3-5
will re-land this shape, incl. the .pt spot check at the first torch.save
batch). Related: [[single-blocker-fix-round-deviation-trace]],
[[respawn-two-record-batch-compose]], [[gate-block remedy round compose]].

**Second-generation addendum (#2336 r7, 2026-08-24)** — the fix-of-the-fix
round (r6 FAIL+FAIL on one overlapping residual: a FALSE marker scoping
clause + the .pt oracle it waived):

6. **The bar shifts to the FAIL+FAIL union bodies + the ORIGINAL r1 rider
   record** — inline all three verbatim (two verdict envelopes + the
   r1 reconciler envelope; strip the codex output's trailing session-ID
   lines). The intermediate r5 reconciler (v3) needs no own envelope: the
   r6 verdicts quote its operative content. Discharge elements come from
   the union Fix fields (both agreed), numbered E1-E5.
7. **Composer independently re-derives the corrected factual claim** the
   round exists to fix (here: 15 torch.save sites/8 files in the batch
   commit — one awk/grep on `git show <payload-sha>`) and attests the
   count + per-file breakdown; Codex re-derives with the same one command.
   Falsity-withdrawal accuracy is its own duty element: explicit
   withdrawal sentence + no "binds at a future batch" residue framing.
8. **Drift-tolerant HEAD pin:** write "= HEAD at compose time; further
   out-of-scope sync commits may land above it — not a finding", never a
   bare "= HEAD" (sync commits land continuously on long-lived worktrees;
   a moved HEAD otherwise reads as a contradiction of an attested fact).
9. **Brief-cited main-tip SHA may be superseded at compose time** — re-run
   the divergence probe at CURRENT origin/main, state both (probe verdict
   unchanged ⇒ compose normally, flag the drift in the return).
10. **Settled-vs-open ledger splits 3 ways now:** claimed-addressed-this-
    round (full duties) / VERIFIED-ADDRESSED-at-prior-round (one
    undisturbed-check status line, never re-adjudicated) / open
    (status lines). torch is named NOT-stdlib for the probe carve-out.

**Third-generation addendum (#2336 r8, 2026-08-25)** — the ORDINARY batch
round after the PASS+PASS fix chain (26-script migration + allowlist shrink
+ a NEW first-instance oracle riding the batch payload):

11. **PASS+PASS prior ⇒ NO verdict envelopes.** The concerns ledger + a
    brief-named lesson paragraph ("the r6 lesson binds: ground-truth A5
    scoping against the diff") carry the history; envelopes are for
    FAIL-contract rounds only. A settled id whose acceptance row RE-FIRES on
    a new shape (A5 first-instance raw-bytes) is adjudicated as a fresh
    DUTY, never a re-open of the settled concern — say so in the ledger
    section explicitly.
12. **Two-part marker:** concatenate parts IN PART ORDER into one envelope,
    attest both parts' top-level version fields + timestamps, pre-attest the
    missing head sentinel as present-but-imperfect (CONCERNS ceiling), and
    state the split is itself the completeness evidence (part 2 = the
    verbatim 97-file list) — never a shape finding.
13. **Composer runs the armed pin-sweep's OWN enumeration at compose time:**
    roster-stem grep over tests/ + set-diff vs (mapped universe ∪ named
    beyond-universe dispositions). Empty residual ⇒ attest "failure shape
    does not reproduce"; a count-rule difference (marker said 28, composer
    found 32-inside) is pre-triaged report-accuracy, not discharge failure.
    Beware short stems: including `workflow_lint` in the stem set exploded
    35 hits to 124 — grep the 26 script stems only.
14. **`+`-line census battery = mechanical-migration anchors:** count of
    added `with atomic_replace(` == marker's migrated-line disposition (47);
    added imports == roster size (26); added waivers == 0; fsync == 0 (an
    edge declared vacuous is attested by one grep); per-shape write-call
    counts (torch.save/np.savez/pickle-csv-yaml-np.save) anchor the A5
    ground-truthing duty. AST-count the allowlist at BOTH trees (it is an
    AnnAssign — plain Assign walkers find nothing) + set-diff deleted
    entries vs roster.
15. **Anchor-uniqueness closes block-presence-vs-block-IS-the-write:** for a
    reproduce-verbatim oracle anchored on a live-source substring, composer
    greps the anchored block's occurrence count in the live file (must be 1)
    + reads context to confirm the site IS the migrated write — attest it,
    leaving verbatim-reproduction + tautology checks to the twin.

**Fourth-generation addendum (#2336 r9, 2026-08-25)** — the batch round under
a BY-PATH lean brief (25 files / 51 sites / allowlist 59→34 / first-instance
npy-via-open_memmap oracle):

16. **By-path batch compose works:** keep the r8 duty/facts/span structure
    but replace the three inlined envelopes with path references — canonical
    MAIN-root `events.jsonl` (jq + python3 stdlib extraction commands,
    frozen-worktree warning, stash-race `git show HEAD:` re-read) + /tmp
    convenience copies with the non-blocking fallback wording + the plan's
    canonical path with the composer-verified-identical worktree copy as
    plan-only fallback. `data-access-blocked` becomes GENUINE (state it in
    the blocked-read rule AND the tags bracket); the Step-3 envelope greps
    are replaced by path-reference validations in the compose script.
17. **Census regex traps (both hit live):** (a) the parenthesized
    multi-context form `with (\n atomic_replace(...)` makes a
    `with atomic_replace(` grep UNDERCOUNT (50 vs the true 51) — count
    added non-import `atomic_replace(` CALL lines instead, then attest the
    reconciliation so the twin doesn't chase a false discrepancy; (b) a
    shell loop building grep patterns via `sed 's/[.(]/\\\\&/g'` inside
    $(...) mangles dotted tokens (torch.save → 0 hits while open_memmap
    counted fine) — run the shape census in Python, never sed-escaped shell.
18. **Sanction-shape flip is a fresh duty:** batch 3 claimed 0 frozen-
    snapshot members, batch 4 claims 3/25 all inside the plan §12 8-member
    sanction list — composer verifies BOTH sides (live-import intersection +
    the plan's named list) and the duty becomes "no fourth member +
    recipe-only edits on the three", not the r8-shaped empty-intersection
    check.
