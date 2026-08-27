---
name: whole-round-unsplit-compose
description: "#2074 split rounds: Codex gets the WHOLE-ROUND brief — base is the round-parent SHA (never origin/main), strip the Step-0 split-review paragraph (its literal trigger token must not enter the prompt), and over-300KB rounds get per-file reads with committed data artifacts digest-only"
metadata:
  type: feedback
---

When the /issue Step 5 round is split-reviewed on the Claude side (#2074
per-commit sub-reviews), the Codex twin's brief is a WHOLE-ROUND UNSPLIT
review — the deliberate catching arm for cross-commit interaction bugs.
Compose deltas vs an ordinary round (first hit: #2330 r1, 2026-08-16):

1. **Base is the brief's `round_parent` SHA, not origin/main.** Verify
   `git -C <wt> merge-base <parent> HEAD == <parent>` at compose time, then
   pin `git diff <parent>..HEAD` in the prompt and BAN main/origin-main
   body diffs (main-side drift pollutes them — the brief usually says so).
   Tell Codex to record `sha-range <parent>..HEAD` in Diff acquisition.
   **HEAD-side variant (#2184 r1):** when out-of-scope commits sit ON TOP of
   the feature commits (spec-freshness syncs from origin/main after the
   round's last feature commit), HEAD itself is out of scope — pin
   `git diff <parent>..<last-feature-sha>` and ban `..HEAD` / `...HEAD`
   BODY forms entirely; verify `merge-base(origin/main, HEAD) == <parent>`
   still holds and name the excluded sync SHAs in the compose-time facts so
   Codex never flags their spec churn.
   **HEAD-side variant (#2184 r1):** when out-of-scope commits sit ON TOP of
   the feature commits (spec-freshness syncs from origin/main after the
   round's last feature commit), HEAD itself is out of scope — pin
   `git diff <parent>..<last-feature-sha>` and ban `..HEAD` / `...HEAD`
   BODY forms entirely; verify `merge-base(origin/main, HEAD) == <parent>`
   still holds and name the excluded sync SHAs in the compose-time facts so
   Codex never flags their spec churn.
1b. **origin-main variant (#2478 r1):** a ROUND-1 whole-round brief may
   legitimately pin `base: origin/main` with NO `round_parent=` when the
   branch IS the round (freshly cut, all commits are round commits) AND
   the brief carries a zero-count divergence probe — verify the merge
   base exists at compose time and use `origin/main...HEAD`; items 2/3/5
   still apply (strip the split paragraph, size the diff, cross-commit
   priority). Do not demand a round-parent SHA the brief never named.
2. **Strip the copied Step 0 "Split-review sub-scope briefs (#2074)"
   paragraph.** Copying it verbatim puts the literal trigger token
   `SPLIT-REVIEW SUB-...` INTO the prompt, arming split-mode behavior
   (write-to-file, skip contract gates) the whole-round review must not
   take. Validate post-compose that the token is absent.
3. **Over-300KB rounds: per-file read strategy in the prompt.** Measure
   `git diff <parent>..HEAD | wc -c` and per-file sizes at compose time;
   scripts get read-every-line per-file diffs, committed DATA artifacts
   (large JSONs) get structural-digest-only instructions (head -c, grep -c
   keys, wc -l) against plan + consumer assumptions.
4. **Leak-validation gotcha:** the adaptation note "the `git stash push`
   alternative is OMITTED" itself re-introduces the literal your own
   validation greps for — word the note without the literal.
5. **Tell Codex to prioritize cross-commit checks** the split reviews
   structurally cannot see (constant defined in one commit / consumed at a
   different grain in another; waivers detached by later refactors;
   committed-artifact grain vs consumer assumptions).
6b. **Pre-split single-marker variant (#2379 r1):** a head reading
   "round 1 (pre-split build, units 1-4 of 4)" with a body stating "covers
   the WHOLE round" is NOT the item-6 thin-final-unit shape — probe
   events.jsonl for `note.startswith('[unit ')` rows (0 hits ⇒ skip the
   progress-notes envelope) and tell Codex the marker IS the full-round
   report so "units 1-4 of 4" is never misread as partial coverage.
6b. **Pre-split single-marker variant (#2379 r1):** a head reading
   "round 1 (pre-split build, units 1-4 of 4)" with a body stating "covers
   the WHOLE round" is NOT the item-6 thin-final-unit shape — probe
   events.jsonl for `note.startswith('[unit ')` rows (0 hits ⇒ skip the
   progress-notes envelope) and tell Codex the marker IS the full-round
   report so "units 1-4 of 4" is never misread as partial coverage.
   **ROUND-MATCH the unit-note probe (#2254 first-k r1, 2026-08-23):** on a
   multi-follow-up task the prefix filter alone can hit STALE `[unit k/N]`
   rows from an EARLIER round (2 hits from the Aug-12 ctxext round, none
   from the round under review) — check each hit's ts against the round's
   commit dates / label before inlining; stale-only hits ⇒ treat as 0 hits
   (single-marker variant, no envelope).
6. **Multi-unit rounds: only the FINAL unit posts `epm:results`** (#2168 r1:
   note head "unit 3 of 3 (FINAL)"; units 1-2 posted `[unit k/N]`
   `epm:progress` notes). Two duties, applicable to ANY round whose fetched
   marker head matches `unit \d+ of \d+`: (a) inline the earlier units'
   progress notes in a supplementary `---BEGIN/END UNIT PROGRESS NOTES---`
   envelope (filter events.jsonl on `note.startswith('[unit ')`); (b) tell
   Codex the Step 0.5 gate scores the inlined `epm:results` body and that
   thin early-unit coverage is at most a present-but-imperfect CONCERNS —
   otherwise an adversarial twin reads "unit K of K" as "the report does not
   cover the round" and false-FAILs `marker-shape` (the #489 class in a new
   costume).

7. **Brief's stale-plan premise can be FALSE (#823 ext-ladder r1,
   2026-08-23; again #2564 r1, 2026-08-24 — worktree copy byte-identical
   to canonical v6):** a whole-round brief may order plan INLINING "because the
   worktree tasks/ tree is frozen at base" while the compose-time identity
   diff shows the worktree copy IDENTICAL to canonical (the plan version
   predated the branch cut). Follow the inline order (also the race-free
   choice on a task with concurrent-follow-up plan-symlink churn), but word
   the plan envelope TRUTHFULLY — never paste the spec's "absent or stale"
   boilerplate over an identical copy (a falsifiable claim Codex can check
   in-sandbox costs credibility) — and flag the premise divergence in the
   return block.

8. **Vendored-pin fidelity tier (#2552 r1, 2026-08-24):** when the round
   VENDORS upstream files under a declared git pin reachable in the
   worktree (a VENDORED_FROM.txt naming source paths + the only claimed
   modification), the read strategy adds a tier between full-read and
   digest-only: delta the vendored copy against `git show <pin>:<orig>`
   filtered by the declared waiver token (VENDORED_FROM.txt itself carried
   the exact command) — empty residue = only declared insertions; any
   residue = undeclared modification reviewed in FULL (a code change hidden
   behind a "comment-only" claim is substantive). Never line-review
   unmodified upstream as round-authored; the driver→vendored-module SEAM
   stays fully in scope. Verify pin reachability (`git cat-file -t <pin>`)
   at compose time before prescribing the recipe.

8b. **Fix-round follow-up on the vendored tier + 1:1-ledgered union FAIL
   (#2552 r2, 2026-08-25):** when the fix round touches ONLY the
   VENDORED_FROM.txt declaration (additive blob-sha annotations; vendored
   .py absent from numstat), the fidelity duty compresses to one check —
   `git ls-tree HEAD scripts/vendored_.../` blob shas match the recorded
   annotations — and the r1 fidelity verdict is stated as standing. Union
   FAIL+FAIL rounds where the r1 Codex `CONCERN::` rows were persisted
   1:1 as ledger ids need NO pseudo-IDs: inline the full r1 verdict
   (head tag stripped, footer cut at first closing tag, rows blockquoted
   `> CONCERN:: ` — assert 16) as the Evidence/Impact/Fix acceptance
   contract, and key the closure table on the ledger ids + each
   `addressed` row's summary + commit. A Claude split Critical closed by
   a MARKER RE-POST (smoke-arch v2 arm-registry fix) is verified against
   the INLINED v2 body (internal consistency: arms_stubbed ==
   FALLBACK-rowed set), never against the code diff. Ledger noise: a
   stray `addressed` row with a junk summary ("test") gets a compose-time
   note naming the REAL row, so the twin scores the right claim. Plan
   by-path (not inline) when the worktree copy is byte-identical — the
   spec default; r1's inline choice does not bind r2.

8c. **Reconciler-BINDING-FAIL fix round (#2552 r3, 2026-08-25):** when the
   prior round ended Claude-PASS / Codex-FAIL / reconciler BINDING FAIL, the
   reconcile record ALONE is the inlined acceptance contract — do NOT also
   inline the twin's own r2 FAIL verdict when the record's Rationale carries
   per-finding closure bars (leaner, and where the twin's Evidence/Fix text
   and the record differ, the record governs; say so + author-neutrality).
   Note the kind/sentinel split for the twin: `epm:review-reconcile` posts
   top-level v1 while its HEAD SENTINEL reads `v<round>` — explain that in
   the envelope preface or the twin flags it. Cumulative-class blockers
   (here Step 0.71 enumeration over the r2+r3 diff) get ONE sanctioned
   exception to round-scoping: the cumulative range GREP-SCOPED
   (`| grep '^+.*smoke'` + hit hunks), never the wholesale body. Severity
   fence: NOT-ADDRESSED on the one blocking finding = substantive FAIL;
   honestly-incomplete closure of a reconciler-DOWNGRADED residual re-raises
   at CONCERN (same-id row), never FAIL; false closure claims at the
   ordinary bar. Composer pre-verifies the marker's cheap mechanical claims
   (residual-conditional greps, parent-blob assert form, function-span
   token absence) and hands them as ground-truth facts.

8b. **Fix-round follow-up on the vendored tier + 1:1-ledgered union FAIL
   (#2552 r2, 2026-08-25):** when the fix round touches ONLY the
   VENDORED_FROM.txt declaration (additive blob-sha annotations; vendored
   .py absent from numstat), the fidelity duty compresses to one check —
   `git ls-tree HEAD scripts/vendored_.../` blob shas match the recorded
   annotations — and the r1 fidelity verdict is stated as standing. Union
   FAIL+FAIL rounds where the r1 Codex `CONCERN::` rows were persisted
   1:1 as ledger ids need NO pseudo-IDs: inline the full r1 verdict
   (head tag stripped, footer cut at first closing tag, rows blockquoted
   `> CONCERN:: ` — assert 16) as the Evidence/Impact/Fix acceptance
   contract, and key the closure table on the ledger ids + each
   `addressed` row's summary + commit. A Claude split Critical closed by
   a MARKER RE-POST (smoke-arch v2 arm-registry fix) is verified against
   the INLINED v2 body (internal consistency: arms_stubbed ==
   FALLBACK-rowed set), never against the code diff. Ledger noise: a
   stray `addressed` row with a junk summary ("test") gets a compose-time
   note naming the REAL row, so the twin scores the right claim. Plan
   by-path (not inline) when the worktree copy is byte-identical — the
   spec default; r1's inline choice does not bind r2.

9. **Fix-round on a FAIL+FAIL union w/ reconstructed marker (#2587 r2,
   2026-08-26):** composable stack — (a) the HEAD-side variant fired again
   (2 spec-freshness syncs above the payload): attest byte-identity to
   origin/main AND zero overlap with payload files, so worktree reads stay
   byte-equivalent to the payload tip; (b) inline the four COMPACT
   adjudication inputs (reconstructed impl v2 + provenance block,
   round-matched smoke-arch v2 re-post, full 16-row ledger, own r1 verdict
   tag-stripped + rows blockquoted) but keep the 104KB plan + 90KB Claude
   union BY PATH (main-checkout-readable sandbox proven by the r1 by-path
   round) — a brief saying "inline the plan per Step 2-pre-b" resolves by
   the duty's own terms (identical worktree copy ⇒ by-path), flag in the
   return; (c) closure duties must name unaddressed HALVES inside
   addressed blockers (compat-gate = sentinel-coverage half + wave-
   enforcement half; smoke-run = judge/analysis half + the fits
   timing/extrapolation half the addressed row never mentions) — binary
   VERIFIED/NOT lines miss half-closures; (d) composer-verify
   marker-claimed /tmp smoke-artifact sha256s on disk and hand them as
   ground truth; (e) sentinel count assert = 2 when your provenance block
   quotes the sentinel the inlined body also carries.

9b. **Second consecutive union fix round (#2587 r3, 2026-08-26):** deltas
   atop item 9 — (a) the closure section splits THREE ways: Block A =
   ledger ids with fresh round-N addressed rows (full acceptance-criteria
   duties), Block B = prior-round VERIFIED-ADDRESSED ids whose CODE the
   round touches again (SETTLED-UNDISTURBED | REGRESSED + class re-check;
   never re-litigated), Block C = unledgered union concerns/minors the
   unit notes claim dispositioned (pseudo-ids, VERIFIED-ADDRESSED |
   ACCEPTED-NON-CHANGE (reason adequate?) | NOT-ADDRESSED, re-raise at own
   severity only); (b) when the round marker is orchestrator-composed from
   unit-completion `epm:progress` notes, INLINE those notes in their own
   envelope — they carry the disposition detail the marker compresses
   (which minor was accepted-with-reason, the named pinning tests) — and
   hand any prose discrepancy (v47's require_p1 wave list "p2,p3,p4,p6,
   p7,p8" vs the realized p2/p3/p4/p5/p6/p8 sites) as an adjudication,
   never resolve it; (c) a smoke-arch arm upgraded FALLBACK→REAL in-round
   with NO marker re-post is a 0.55 NON-finding to pre-state (presence-ON-
   TASK; the new evidence lives in the impl marker) — at most present-but-
   imperfect; (d) a ledgered-CONCERN id that carries the OTHER twin's
   Critical (query_form) gets a dual-severity fence: persisting core
   defect ⇒ substantive FAIL despite the CONCERN label, honest riders ⇒
   CONCERN re-raise; (e) a self-disclosed measurement tension in (b)/(d)
   (CPU-basis fit wall 4.98× over the GPU-basis §9 wall) composes as a
   both-routes adjudication hinged on whether the leaned-on backstop
   (fits.py pilot gate) exists AND is armed by the production invocation.

9c. **Reconciler-BINDING-FAIL fix round, single-BLOCKER both-halves closure
   (#2587 r4, 2026-08-26):** deltas atop 8c/9b — (a) when the brief names
   BOTH the twin's own FAIL verdict (it carries the Mechanizable recipes
   the fix realizes) AND the reconcile record, inline BOTH with an explicit
   "the record GOVERNS where they differ" line (8c's record-alone lean
   applies only when the brief doesn't name the verdict); restate the
   record's empirical reproductions in the envelope preface — they are the
   bar. (b) ONE ledger id covering two Majors + a test recipe composes as
   ONE status line with named-half vocabulary (`PARTIALLY-ADDRESSED
   (<half: queue | manifest | tests>)`), never two pseudo-rows. (c) The
   twin's own still-OPEN prior CONCERN (not claimed by the confined round)
   gets a prescribed `OPEN-UNCHANGED — <awaits X>; not claimed by this
   round` status line + a never-re-emit instruction. (d) A fix that INVERTS
   an ordering (re-serve → truncate-first = at-least-once → at-most-once)
   gets an explicit NEW-HAZARD trace duty in the opposite direction. (e) A
   marker-disclosed in-round harness bug (controls caught fault tests
   false-passing) becomes an adjudication input: verify assertions now pin
   error-branch-specific evidence. (f) Bookkeeping commit can sit BELOW the
   payload (parent → bookkeeping → payload == HEAD): range `parent..HEAD`
   then includes it — name it in-range-but-out-of-scope; HEAD == payload
   tip makes worktree reads byte-identical again. (g) ASSERT-SIDE TRAP hit
   live: extracting the tail with an UNANCHORED `"# Output contract"`
   matched the head's ADAPTATIONS `## Output contract` mention and silently
   DUPLICATED the 87 KB rubric (271 KB prompt); caught only by eyeballing
   the printed part sizes. Always use the newline-anchored index and assert
   `len(tail) < 15_000` + rubric-anchor-absent-in-tail + rubric-once in the
   final prompt.

**Why:** the whole-round view is the ONLY reviewer seeing commit
interactions; a mis-based diff (origin/main) or a leaked split-token
defeats exactly that purpose.

**How to apply:** any brief carrying `round_parent=` + `round_commits=` +
"whole-round UNSPLIT review" context. Related:
[[revision-round-compose-recipe]], [[worktree-task-folder-status-can-be-stale-in-EITHER-direction]].
