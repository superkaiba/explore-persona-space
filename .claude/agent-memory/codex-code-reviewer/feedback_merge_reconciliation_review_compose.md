---
name: merge-reconciliation-review-compose
description: Composing a Step 10d divergence-reconciliation merge review (#2253 r3) — parent-relative scoped diffs, zero-hand-edit premise as check 1, misleading-range warning, adapted gate-scope note
metadata:
  type: feedback
---

For a Step 10d pre-merge divergence-delta reconciliation round (#1771→#2201), the review is a MERGE-SEMANTICS check, not a feature review. Compose pattern (validated #2253 r3, 2026-08-21):

- **Lead with the misleading-range warning.** `git diff <branch-tip>..HEAD` after a merge of main is main's own history (380 commits / ~29.8 MB on #2253) — ban the unscoped read explicitly, citing #521 (Codex flagged main-drift as churn and burned a reconciler round).
- **Zero-hand-edit premise is check 1, verified not assumed.** `git diff-tree --cc HEAD` empty + `log -1 --format='%H %P'` parent pins. Every downstream "nothing to read" claim rests on it; if it fails, everything escalates.
- **Diff acquisition = 4 scoped parent-relative diffs** (merge-base..each-parent for contributions; each-parent..HEAD for what the merge carries), all restricted to the gate-flagged overlap files. Dropped/mutated hunk in either direction = Critical `substantive`. Header literal: `sha-range <merge-sha> (merge, parent-relative, N-file scope)`.
- **Set-equality registrations get an own-enumeration duty** (tuple vs manifest read statically), never "the pin test passed" — the asymmetric-drop hazard is exactly what auto-merges hide.
- **Gate-scope on a pure merge:** the report legitimately carries an adapted `Gate-scope note` (no hand-written lines ⇒ no changed literals to sweep). Score PRESENT-but-adapted → at most CONCERNS; the diff-consistency half collapses into the zero-hand-edit check.
- Round-matched `epm:results` marker DID exist for the reconciliation round on #2253 r3 (posted by the gate's dispatch) — probe events.jsonl before assuming the follow-up-round placeholder path. **NO-MARKER branch confirmed live (#2608 r4, 2026-08-26):** a pure auto-merge round dispatched by the gate can have NO round-matched `epm:results` at all — the round record is the gate's `[divergence-probe]` `epm:progress` notes. Inline THOSE verbatim in a `---BEGIN GATE DISPATCH NOTES---` envelope as the round record, inline the LATEST (prior-round) `epm:results` in the standard marker envelope FENCED as branch-payload context only ("do not score r-N against its claims"), declare marker-shape/gate-scope absence gates INVALID-by-design for the machine-made round, and narrow tags to `substantive`|`git-provenance`|`data-access-blocked`. Also: pin BOTH the main-parent SHA and ban live `origin/main...HEAD` explicitly when main has advanced past the pin (it had, by 3 probes' worth on #2608).
- Round-matched `epm:results` marker DID exist for the reconciliation round on #2253 r3 (posted by the gate's dispatch) — probe events.jsonl before assuming the follow-up-round placeholder path. **NO-MARKER branch confirmed live (#2608 r4, 2026-08-26):** a pure auto-merge round dispatched by the gate can have NO round-matched `epm:results` at all — the round record is the gate's `[divergence-probe]` `epm:progress` notes. Inline THOSE verbatim in a `---BEGIN GATE DISPATCH NOTES---` envelope as the round record, inline the LATEST (prior-round) `epm:results` in the standard marker envelope FENCED as branch-payload context only ("do not score r-N against its claims"), declare marker-shape/gate-scope absence gates INVALID-by-design for the machine-made round, and narrow tags to `substantive`|`git-provenance`|`data-access-blocked`. Also: pin BOTH the main-parent SHA and ban live `origin/main...HEAD` explicitly when main has advanced past the pin (it had, by 3 probes' worth on #2608).

**Brief-quoted rosters go stale ACROSS the merge (#2362 r2, 2026-08-26):** a brief composed pre-merge asserted the three-dot names "8 deliverable + 4 agent-memory sync files"; the live post-merge three-dot named only the 8 — the sync commit's files became content-identical to the merged-in main, so they legitimately vanish from merge-base..HEAD. Always RE-RUN the roster command live at compose time, pin the OBSERVED list as the expectation, and encode the brief's extra names as a tolerated-extras provision (absence-is-correct explained; if present, benign only when `git diff origin/main -- <path>` is empty) rather than FAILing Codex on the brief's stale count. Same round also validated: brief-prescribed 4-question scope + binary VERDICT enum + a merge-base assert line in the marker header (`git merge-base origin/main HEAD` == the pinned main parent) as the Q3 precondition, with a report-not-FAIL routing when the assert misses.

**Why:** the gate's own rationale is that a semantic collision can merge textually clean; the composed prompt must make Codex answer that question and nothing else, or it drowns in main's history.

**How to apply:** any round whose brief names a reconciliation/merge commit and parent SHAs. See also [[revision-round-compose-recipe]] and [[stale-base-mb-pin-and-fixture-remeasure]].

**HAND-RESOLVED + NO-MARKER combo, id-collision renumber (#2363 r2, 2026-08-27):**
the two variants compose together — a gate-dispatched reconciliation merge with
semantic conflict resolutions (check-id renumber c73→c74) and NO round-matched
`epm:results` (round record = `[divergence-probe]` notes + the merge commit
message; prior `epm:results` inlined FENCED as context; tags narrowed to
`substantive`|`git-provenance`|`data-access-blocked`). Four live catches:
(a) **renumber-completeness expectation tables need the SPELLED-OUT grain** — a
`c73`-literal grep came back clean while `grep -niE 'check[ _-]?73'` found two
stale optional-phase-binding references (`verify_plan.py:471` "check 73" in the
N/A-escape docs; `adversarial-planner/SKILL.md:527` in a CLEAN-merged r1 file
the literal-grain renumber never visited — post-merge "check 73" points at the
WRONG check since main's colliding check keeps the id); never compose a
"zero residue" expectation from one grain. (b) **Own-memory-file duplication:
probe BOTH parents — the #2263-r9 outcome INVERTED here**: headings 1×/1× at
the parents, 2× at the merge ⇒ merge-CREATED duplication, a real Q4 resolution
artifact; state counts neutrally + consequence-based severity (bookkeeping
prose vs always-loaded MEMORY.md index), never a blanket "stat-only, never a
finding" fence on hand-resolved files. (c) **Brief's file/line counts can
exclude the composer's own agent-memory commits** (brief "9 files +488" vs live
11/+544 — delta exactly the 2 memory files): reconcile arithmetically, keep
them in scope as Q4 subjects. (d) Mid-compose ledger drift fired again
(#2326-r3): the parallel Claude reviewer's round-2 BLOCKER rows landed in
concerns.jsonl between reads — pin the snapshot to round-1 rows AND assert the
excluded ids are absent from the final prompt. Also: p2-baseline pin arithmetic
(main's own 70/63/73/enum vs merged 71/64/74/enum+74) composes as settled
facts with the tree-verification duty handed over; pin every command to the
MERGE SHA, never bare HEAD (the composer's own same-turn memory commit advances
the branch tip post-freeze).

**AUTO-MERGE with both-sides-touched files + follow-on dedup commit (#2610 r4, 2026-08-27):** a non-empty `--cc` is NOT by itself a hand-edit signal — a file BOTH parents touched shows in the combined diff even on a faithful auto-merge. The hand-edit probe is `git diff-tree --cc <merge> -- <file> | grep -c '^++[^+]'` (lines added vs BOTH parents): 0 on the auto-merged code file = no hand-written merge content; the union-driver memory file legitimately shows >0 (the double-insertion a follow-on dedup commit removes — judge the file at HEAD, not at the merge). Composer battery that settles textual fidelity so the twin does only the SEMANTIC read: (a) sorted added/removed line-SET identity of `merge-base..merge -- <file>` vs the r3-certified payload delta, and of `branch-parent..merge -- <file>` vs main's own commit delta (byte-cmp of diffs fails on hunk offsets; line SETS are the right grain); (b) per-file three-dot vs r3-payload byte-cmp for single-parent files; (c) dedup commit = deletions-only + HEAD copy 0-byte vs ALL main pins (merged snapshot, dispatch probe, live tip) + `comm -23` no-lost-row vs the r3-tip blob. Post-merge the memory file VANISHES from three-dot (content==merge-base) — pin absence-is-correct in the prompt (#2362 class). Marker EXISTS on this shape (unlike #2608 r4): inline normally; report-count grain notes ("24 rows" vs 23 deleted lines) go to the twin as neutral adjudication, never composer-resolved. Merge-base assert header stays (`== <pinned snapshot>`, report-not-FAIL on drift) since live main advances past the pin during review. Do NOT edit/commit agent-memory to the branch mid-round — a new commit would move HEAD and falsify the prompt's pinned roster; leave uncommitted + flag for the orchestrator's post-merge sweep.

**HAND-RESOLVED variant (#2263 r9, 2026-08-23) — conflict resolutions present, so the zero-hand-edit premise INVERTS:** the combined `--cc` diff is non-empty and IS the review surface (`git show <merge>`; verify the exact file list via `git diff-tree --cc --name-only` — the merge's `git show --stat` list is first-parent-relative and misleading). Structure the prompt as four charges: (Q1) both-sides preservation per semantic file — composer verifies the main-added-line DENOMINATOR (grep '^+' non-empty on merge-base..main-tip) and hands the presence sweep to the twin as own-enumeration; (Q2) each DECLARED contradiction choice gets a soundness + splice-point-coherence duty with the dropped line quoted verbatim in a FENCED block (the lines carry backticks — never nest them in prose backticks); (Q3) a hook-forced memory rebuild gets a no-lost-row duty vs BOTH parents' blobs (`- [` rows for MEMORY.md indexes, `**`-led headings for feedback files), with the transient union size declared unverifiable — evaluate the REBUILD; (Q4) no-new-defects: re-measured grandfather arithmetic on the merged tree, composer-attested ruff, conflict-marker greps, combined-diff closure + branch-deliverable files verified untouched via `<branch-parent>..<merge>` empty diffs. Three traps re-hit live: the r4 bare-substring trap (grep "plus any" hit 1 UNRELATED line while the exact dropped line counted 0 — always probe the EXACT line and pre-clear the substring hit in the prompt); duplicated entries in the composer's OWN memory file looked merge-created but pre-existed on main's blob (probe counts at both parents before flagging — pre-seed Step 0.9 pre-existing-on-trunk); open NITs from prior rounds get the #2205 three-way vocabulary with ONE merge-armed check when a merge hunk touches a NIT's subject file (NIT-capped). Tags narrowed to `substantive`|`git-provenance`|`data-access-blocked`; marker-shape/smoke invalidated (reconciliation-report shape); sentinel = review-round number per brief (== max-posted+1 continuity).

**Fix-round follow-on (#2363 r3, 2026-08-27):** the round after a
reconciliation FAIL composes as closure charges keyed on the ledger's
raised-row `evidence` fields (again NO round `epm:results` — record = fix
commit + the orchestrator's `addressed` rows, which are CLAIMS to verify;
tags stay narrowed to `substantive`|`git-provenance`|`data-access-blocked`).
Three duties validated live: (a) **prescribed-fix-deviation adjudication** —
the raised row's literal Fix ("reset both files to origin/main") was not what
shipped (feedback file reset; MEMORY.md surgically deduped to preserve the r2
composer's own db81115b row extension): hand the consequence-over-literal
read (#2147-cr4 pattern) with the composer's diff-vs-main table as the
expectation. (b) **Probe the dup-row table at BOTH origin/main and the fix
SHA** — the ledger's "one pre-existing dup" UNDERCOUNTED (two pairs pre-exist
on main: selector-self-edit AND reconcile-record); attest the full table or
the twin flags the second pair as an unfixed round defect. (c) **A file
"reset to origin/main" needs a HEADING-LOSS probe** (parent-1 headings ⊆
main's copy ⇒ the reset destroyed no branch-unique content) — without it the
reset reads as potential content loss. Also: pre-classify the
correct-by-design spelled-out-grain residue (main's own "Check 73" banner;
other verifiers' check-N.M artifacts like `_CHECK732_*`), and instruct that a
NOT-ADDRESSED closure IS a changed disposition ⇒ re-emit that id as a
`CONCERN::` row so the ledger re-arms.
