---
name: codex-flags-Step5a-spec-freshness-sync-as-out-of-scope
description: Codex FAILs an experiment branch for "bundling .claude/* workflow-surface edits" when the commits are the PRESCRIBED Step 5a spec-freshness sync FROM origin/main (from local main pre-#1747). Verify the commit title (either era's form) + that every changed .claude file matches origin/main before crediting the scope-violation framing.
metadata:
  type: feedback
---

**Rule:** when Codex raises a `substantive` blocker that an experiment branch
"bundles global `.claude/*` workflow-surface edits unrelated to the task" /
"would merge repository-wide behavior" and proposes "revert or split into a
separate workflow-spec PR" — DISCARD it as ungrounded if the commits are the
mandated Step 5a spec-freshness sync. Two-check diagnostic:

1. **Commit title.** `git log origin/main..HEAD --oneline -- '.claude/**'` — the
   sync commits carry a VERBATIM prescribed title, in EITHER era's form: the
   current `issue-<N>: sync workflow-surface specs from origin/main
   (spec-freshness)` (#1747+) or the historical `issue-<N>: sync
   workflow-surface specs from main (spec-freshness)` (pre-#1747 commits keep
   the old title) — each is `.claude/skills/issue/SKILL.md` Step 5a's literal
   `git commit -m` line of its era. Step 5a is MANDATORY ("Spec-freshness
   check first ... applies at EVERY ensemble/agent fan-out") and does exactly
   `git checkout origin/main -- $SAFE_SPECS` over the workflow surface
   (the SKILL.md `SPECS=` list — `.claude/agents .claude/skills .claude/rules
   .claude/workflow.yaml CLAUDE.md` plus the #1560/#1714 lint/guard family
   entries; read the live list from Step 5a; sourced from local `main` before
   #1747).
2. **Direction of the change.** Prefer the whole-file check against
   `origin/main:<file>`: blob-sha equality
   (`git rev-parse origin/main:<path>` == `git rev-parse HEAD:<path>`, or an
   empty `git diff origin/main -- <path>`) proves the synced file matches
   origin/main BY CONSTRUCTION — the sync checks out origin/main's bytes, and
   LOCAL `main` may lag origin and mismatch a correctly-synced file. Per-line
   fallback: `git grep -c '<flagged line>' origin/main -- <path>`
   for each changed `.claude/*` line — if it ALREADY EXISTS on origin/main, the
   branch is importing main's content INTO the worktree (bringing stale worktree
   specs up to date), NOT introducing edits TO main. Worktrees load agent/skill specs
   from the SESSION cwd, so a worktree cut before a later workflow fix runs STALE
   specs without this sync (incident #557 r2). The "no merge base" three-dot diff
   shows these as "changes" only because the branch's merge-base predates main's
   adoption of the lines.

Why it's never a FAIL: Codex's proposed fix DIRECTLY CONTRADICTS the prescribed
procedure. The sync is the workflow doing its job; the only legitimate Step 5a
exception is a branch whose DELIVERABLE *adds* workflow-surface entries (e.g. a
new marker schema riding its feature branch, #535) — and the per-file
branch-side-edit guard skips exactly those, so a blind sync of those files never
happens. An experiment branch (scripts + tests + eval outputs) carrying ONLY
verbatim-from-main `.claude/*` is the normal, correct case.

Worked instance — #649 r1 (code-reviewer): Codex FAILed `substantive` on
`.claude/agents/{code-reviewer,codex-code-reviewer,reconciler}.md` +
`.claude/rules/gotchas.md` "bundled into the experiment branch." Every changed
line already existed on origin/main (the #613 no-merge-base fallback — literally
the text live in the reconciler's own spec — plus the #613 `use_tqdm=False` +
#640 `check-upload-as-file` gotchas); both commits were titled
`issue-649: sync workflow-surface specs from main (spec-freshness)`. Discarded;
PASS. (Claude reviewer correctly classed them "out of scope for substantive
review, no action.")

Sibling: [[feedback_codex_litigates_pre_existing_in_round_n]] (the git-provenance
family — there the flagged code is pre-existing on trunk; here the .claude state
is a prescribed branch operation pulling FROM trunk). The shared move is "verify
git provenance before crediting a scope/regression framing."
