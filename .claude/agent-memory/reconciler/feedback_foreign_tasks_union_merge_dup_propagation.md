---
name: foreign-tasks-events-union-merge-dup-propagation
description: A foreign stale tasks/<status>/<M>/events.jsonl in a branch's three-dot diff is NOT cleared by "the rows already exist on main" — housekeeping merges union-duplicate rows, and rename detection + merge=union silently propagate the duplicates into main's canonical moved ledger. Simulate the local merge in a scratch worktree before crediting either verdict.
type: feedback
---

**Rule:** when a reviewer disagreement turns on a foreign `tasks/` path in a
worktree branch's `main...HEAD` diff (typically imported by a "Merge branch
'main' into issue-<N>" housekeeping merge), do NOT adjudicate on substring
presence: verify (a) exact content vs merge base and vs main's CANONICAL
(possibly git-mv'd) copy — `sort | uniq -d` for union-merge duplicate rows —
and (b) the ACTUAL merge outcome via a scratch worktree detached at main +
`git -C <worktree> merge --no-commit --no-ff issue-<N>` (the `-C <worktree>`
form is required — a bare `git merge` is hook-blocked, #1128; this honors
`.gitattributes` `tasks/**/events.jsonl merge=union`, which `merge-tree` on
old git cannot, and server-side GitHub merges do not).

**Why:** #1087 r1 — Codex PASSed because the two distinctive #1085 row
substrings existed on main at `tasks/completed/1085/`; but main had them ONCE
(44 lines, 0 dups) while the branch's stale `tasks/running/1085/events.jsonl`
carried them TWICE (34 lines — union artifact of the housekeeping merge:
30-line branch copy ∪ 32-line main-parent copy). The simulation showed rename
detection pairing `running/1085` → `completed/1085` and the union driver
folding both duplicate rows into main's canonical ledger with ZERO conflict
("Automatic merge went well", 46 lines staged) — silent corruption of an
append-only ledger that `latest-marker` / the state machine read. Upheld
Claude's FAIL.

**Guard-topology facts that decide these cases:**
- SKILL.md (~L9118-9120): Step 10d Guard 1 (foreign-`tasks/` strip) "only runs
  on the `--rebase` form" — and a branch carrying a merge commit gets its
  server-side `--rebase` REFUSED, so the realistic forms (`--squash` /
  artifact-confirmed local merge) run exactly the union+rename machinery.
- Even where Guard 1 applies, stripping the file converts silent propagation
  into `CONFLICT (rename/delete)` + `(modify/delete)` — a bounce, not a clean
  merge. Neither branch of "guards absorb it" holds.
- The verified clean fix: restore the MERGE-BASE version of the file on the
  branch (content == base ⇒ tasks/ three-dot diff empties ⇒ main's rename wins
  cleanly). One command, test it in the same scratch-worktree simulation.

**Polarity note:** this is the REVERSED case of
[[feedback_codex_litigates_pre_existing_in_round_n]] — there Codex over-flags
state that never reaches main; here Codex under-flagged state that
demonstrably does. The shared diagnostic is identical: trace what the merge
actually does, never argue from provenance labels ("merge artifact") or
substring presence. Sibling method:
[[feedback_claude_approves_merge_step_without_mergetree]] (run the merge
simulation yourself).
