---
name: candidate-already-fixed-stale-worktree
description: Before applying a workflow-fix candidate, git-log the target file on main; the incident may stem from a worktree's pre-fix copy of an already-hardened spec
metadata:
  type: feedback
---

Before refining a candidate's diff_sketch, run `git log --oneline -5 --
<target_file>` on main and grep main's copy for the proposed rule.
Auto-spawn candidates are emitted from sessions whose cwd is an ISSUE
WORKTREE; the workflow-surface specs loaded there are frozen at branch-cut
time, so the "missing rule" may already exist on main in a STRONGER form —
blindly applying the sketch would weaken it.

**Why:** #557 r2 (2026-06-10) — candidate proposed a dispatch rule for
`codex-code-reviewer.md` that had landed on main 12h earlier (`bd26e7b0d`);
the sketch would have re-allowed foreground wrapper dispatch that main bans.

**How to apply:** when the rule already exists, apply only the residual value
(recurrence citation, recovery recipe, missed sibling files) and surface the
propagation root cause as a follow-up. Variants seen since:
- **Parallel incident** (#541/#552, 2026-06-10): prior commit is a SAME-DAY
  sibling incident holding complementary facts → INTEGRATE into one coherent
  section, reconcile the two incidents' claims; no duplicate H2.
- **Hedged prose** (#536, 2026-06-11): prose-synthesized candidates carry
  unverified hedges ("presumably omits") — the emitter guessed file state
  without reading it; honor the no-op escape hatch when the content is
  already documented.
- **Same check, different shape** (#537, 2026-06-11): a near-identical prior
  fix does NOT make the candidate stale — diff the SHAPES and reproduce the
  check against the live incident body pre/post-fix (here: extend
  `_LR_ANCHORED_RE` rather than dismiss).
- **Exact duplicate dispatch** (#591, 2026-06-11): the prior commit can be
  the SAME candidate already fixed (cross-session duplicate dispatch has no
  protocol guard) → verify all mirror surfaces current, clean no-op.
- **Fix the resolution, not the content** (3rd #591 dispatch): on repeated
  recurrence, fix the PATH RESOLUTION letting composers read stale worktree
  copies (`c48c6101a`: derive `REPO_ROOT="${TASK_DIR%/tasks/*}"`, pin
  spec-reads to `$REPO_ROOT/...`); audit sibling twins for the same pattern.
- **Mixed-read phantom contradiction** (#612, 2026-06-12): emitter read the
  CANONICAL spec post-fix but its OWN twin spec from a pre-fix worktree copy
  → reported a cross-file contradiction that was never a committed state
  (`61cc120cd` updated both files atomically). Diagnose by comparing the
  fix-commit timestamp vs the worktree's spec-sync commit timestamp; if the
  worktree has since re-synced, it's a clean no-op.
