# Diff-size budget — size any branch diff before reading its body

**Fires when:** a code-reviewer, implementer, or any agent is about to read a
branch-wide diff BODY (`git diff origin/main...HEAD` or any multi-round branch range)
on a long-lived worktree branch. The pre-read gate lives in
`.claude/agents/code-reviewer.md` Step 0 and
`.claude/agents/experiment-implementer.md` § On revision rounds; this file is
the full recipe.

## The gate

Before ANY diff BODY read, size it:

    git diff origin/main...HEAD | wc -c

The base ref is fetched `origin/main` (#1289 — the shared root's local `main`
can lag origin; run the bounded fetch per code-reviewer.md Step 0 first; if
`origin/main` does not resolve, fall back to local `main`).
The pipe streams into `wc` — sizing is free of context cost (only the byte
count enters agent context). **Sizing must fail loud:** on a sparse/shallow
checkout with no merge base the three-dot form errors to stderr and `wc -c`
prints `0` — probe `git merge-base --all origin/main HEAD` FIRST; if it is empty, or
the sizing pipe errors, or it returns `0` on a branch that demonstrably has
commits, treat the diff as OVER budget and round-scope (a no-merge-base
checkout cannot materialize a three-dot body anyway; per the #613 precedent
this is a checkout artifact — never block or FAIL on it). If the byte count
exceeds **300 KB**, do NOT read the whole-branch diff body. Scope every body
read to the CURRENT round instead:

- `git show <sha>` for each commit named in the implementer's round
  marker/report, or the round range `<parent>..HEAD` from the implementer
  report — the same per-round recipe code-reviewer.md Step 0.9's
  `cumulative-main-head-diff` subclass prescribes.
- **When no round marker/report names commits or a parent** (crash-recovery
  respawn, orphaned round), resolve `<parent>` deterministically: the previous
  reviewed round's tip from the latest `epm:code-review` marker, else
  `git merge-base origin/main HEAD`; if neither is usable, drop to name-only plus
  per-file scoped body reads.
- **Cross-round interactions:** for files the `--stat` / `--name-status` pass
  flags as touched across MULTIPLE rounds, a bounded per-file three-dot body
  read (`git diff origin/main...HEAD -- <path>`) is permitted — it closes the
  cross-round-interaction gap without re-opening the whole-branch read.
  Size it first the same way (`git diff origin/main...HEAD -- <path> | wc -c`); a
  per-file body over the same 300 KB budget falls back to round commits plus
  name/status/stat context — "per-file" is a path bound, not a byte bound,
  so the byte check still applies.

Earlier rounds' changes on a long-lived same-issue-follow-up branch were
already reviewed in their own rounds; re-reading them is redundant context by
construction.

## Unrestricted forms and the two-dot ban

- Name-only / `--name-status` / `--stat` / `--diff-filter` forms are cheap and
  stay unrestricted at any size.
- NEVER read the two-dot `main..HEAD` BODY on a worktree branch — it folds in
  unrelated main churn (31.6 MB on #722). The two-dot form is permitted only
  in NAME-ONLY form (e.g. the sparse-worktree no-merge-base fallback in
  code-reviewer.md Step 0).

## Interaction with review invariants

This gate changes the SCOPE of the diff body you read, never whether you read
one — code-reviewer.md Step 0.7's "pre-diff gates never short-circuit the
diff" invariant is untouched, and the gate is a fallback, never a block/FAIL
(the #613 never-block precedent).

## Grounding (the 300 KB budget)

Claude context ≈ 200K tokens ≈ ~800 KB of text at ~4 bytes/token; a diff body
must leave the majority of the window free (budget ≈ 3/8 of the raw ceiling).
Incident #722 r2: the whole-branch three-dot diff was 1.96 MB (43K insertions,
136 files of already-reviewed prior rounds), the two-dot 31.6 MB, while the
round's own reviewable delta was 36 KB + 29 KB — two subagent spawns (an
experiment-implementer, then a code-reviewer) died on autocompact loading
these. 300 KB separates the healthy and pathological regimes with wide margin
on both sides; the value is not sensitive within ~2×.

## Files of record

Task body #832 (this rule's origin); incident #722 r2 (the numbers);
`.claude/agents/code-reviewer.md` Step 0 (the reviewer-side gate) + Step 0.9
subclass 3 (`cumulative-main-head-diff`, the per-round scoping recipe);
`.claude/agents/experiment-implementer.md` § On revision rounds (the
implementer-side bullet).
