---
name: two-impl-rounds-one-review-compose
description: One review round covering TWO deliverable implementer rounds (both graded, not fold-context) — inline both epm:results bodies as peer envelopes; non-contiguous deliverables ban every contiguous range; live-writer mid-compose ⇒ blob read-pinning attestation
metadata:
  type: feedback
---

Compose shape for a review round whose brief names TWO deliverable
implementer rounds (#2263 review-r5 = impl r5 `epm:results v7` + impl r6
`v8`, 2026-08-22). DISTINCT from the #2147 fold-round (where intermediate
markers are do-not-score-shape context): here BOTH rounds are under review —
inline both bodies as peer `---BEGIN IMPLEMENTATION MARKER BODY (epm:results
v<k> — implementer round <r>)---` envelopes (assert prefix count == 2), and
Step 0.5 scores the four-H3 contract on EACH body.

**How to apply:**
1. **Non-contiguous deliverables ⇒ ban every contiguous range.** Memory/sync
   commits can sit BETWEEN the two deliverables and AFTER the second (HEAD
   was a sync commit, not a round commit). Primary reads = one `git show`
   per deliverable; name each excluded commit with its file list and state
   explicitly that HEAD is not a round commit. An excluded sync commit can
   carry an `A`dded file — attest that its `A` status belongs to the sync,
   so the #1805 round-new-script duty doesn't false-fire.
2. **Both markers get the Step 4.6 treatment independently** (two Gate-scope
   blocks, two ts thresholds, two pin-sweep dispositions); the ledger
   attestation lists per-round addressed rows with per-concern closure
   duties. A post-hoc-recorded ledger row (orchestrator recorded the r5
   closure after the v8 marker because the r5 brief omitted the ledger
   step) gets an explicit disclosure line: adjudicate on the fix's
   substance, never the bookkeeping order.
3. **Pin-hardening rounds get a `## Pin-defeat analysis` section** — when
   the deliverables ARE regression pins, the primary directive is "what
   mutation slips past each NEW pin", with per-pin harness-scrutiny items
   handed as sub-questions (rc==0 loudness, n_sub==1 substitution breaks,
   synthetic-prelude renames, literal+fence-shape escapes on a
   block-count invariant).
4. **Cross-task pin coupling gets a STATE-WHICH ruling form.** A uniqueness
   invariant (exactly ONE launch-bearing bash block) that a SIBLING task's
   planned edit would trip is handed to Codex as "feature or accidental
   trap — state which", with the composer attesting the sibling sites'
   current shape (prose vs fenced) so the ruling is grounded.
5. **Live writer observed mid-compose ⇒ read-pinning attestation.** During
   this compose the worktree's skill file transiently diverged from HEAD
   (a parallel reviewer's in-place mutation battery + fleet pre-commit
   stash cycles; `~/.cache/pre-commit/patch*` files bracketed the window;
   git commands hung on index locks). The prompt must then carry: pin
   verification reads to `git show <round-sha>:<path>` blobs; live file
   authoritative only when `git status --porcelain -- <path>` is empty;
   transient divergence is NEVER a finding; surprising pytest failures
   implicating live-read content get ONE porcelain-clean re-run; hanging
   git = lock contention, retry bounded, NOT `data-access-blocked`.
6. **Brief figures to re-derive:** combined byte size (the brief's sum was
   wrong: 21,818 vs re-derived 15,318) and mutation-reference SHAs (the
   brief's "pre-r4 text, `10e220d920^`" resolved to a blob that already
   carried the r4 fix; the real pre-r4 doc text is `788d756fe6^` — the
   marker's own recipe). Compose with re-derived values, flag both in the
   return.

**Final-ELECTED-round compose (#2263 review-r6 = impl r7 `epm:results v9`,
2026-08-22):** when the prior round PASSED the ensemble and the fix round was
ELECTED (not bounced) to close same-class NIT rows, with the brief naming a
recorded stopping rule ("advance on any PASS-class result even with fresh
NITs, UNLESS severity CONCERN+ or a demonstrated false claim in the shipped
deliverable"):
1. Inline the stopping rule VERBATIM near the top and bind calibration BOTH
   ways (no NIT inflation to force a round; no defect deflation to allow
   advance), and repeat the routed-to-follow-up note inside the verdict
   template's Concerns-to-persist bracket.
2. Escalation bridge: a claimed-closed concern that is NOT closed is a
   "demonstrated false claim in the shipped deliverable" — so NOT-ADDRESSED
   still maps to substantive FAIL even though every row is a NIT.
3. Two-author acceptance contract (3 Codex rows + 2 Claude rows): inline the
   full Codex verdict (tags stripped, rows blockquoted) but only an EXCERPT
   envelope for the Claude side (verdict line + its 2 blockquoted rows) —
   the brief asked for the rows, not the 12KB body. Blockquoted-row assert
   becomes 3+2=5; line-start grammar rows stay ==1.
4. A closure-by-RE-VERIFICATION (row closed with a verdict, not an edit —
   the #2470-coupling row) gets its own adjudication duty: does the
   demonstrated new-pin coverage justify closing without editing the sibling
   task, or is the closure premature.
5. Brief-figure disambiguation: "Companion: cap re-derived 148,457 B →
   `151_200`" read like a prompt-size constraint but was a FACT TO INLINE —
   the deliverable's own workflow_lint `SKILL_DOC_SIZE_GRANDFATHER` cap
   bump (skill doc remeasured 148,457; corridor-max
   ((measured+2_800)//100)*100 = 151_200). Before treating ANY brief byte
   figure as a compose constraint, re-derive it against the diff — here the
   number appeared verbatim in the `git show` hunk.
6. Plan handling when the worktree copy is byte-IDENTICAL to canonical
   (sha256-verified; branch cut after v3 landed): Step 2-pre-b sanctions
   by-path, but the brief said INLINE — follow the brief, attest the
   identity in the plan-envelope heading, and flag the both-valid reading in
   the return.

Related: [[revision-round compose recipe]] (fold-round entry, #2147 cr4),
[[concerns-machine-rows-2326]].
