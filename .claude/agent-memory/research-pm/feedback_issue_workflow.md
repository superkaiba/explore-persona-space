---
name: Issue-Driven Workflow
description: Prefer `/issue <N>` over manual dispatch for any task ≥30 min. GitHub issues are the durable task log; specialists post marker comments.
type: feedback
---

For any task ≥30 min of work (experiment or code change), prefer the `/issue <N>` skill over the manual DISPATCH + INTEGRATE modes.

**Why:** the skill is idempotent and resumable — all durable state lives in GitHub issue labels + marker comments, so the user can close the terminal mid-run and pick up cleanly. It also enforces the full lifecycle (clarify → gate-keeper → adversarial-planner → approval → worktree + dispatch → preflight → run → analyzer → reviewer → close) which manual dispatch sometimes skipped (e.g., reviewer / code-reviewer was easy to forget).

**How to apply:**
- When the user says "run experiment X" or "refactor Y": check if an issue exists. If yes, `/issue <N>`. If no, create one (`gh issue create`) with the right labels (`type:*`, `status:proposed`, `aim:*`, `prio:*`, `compute:*`), then `/issue <N>`.
- Specialists dispatched via the skill are in "issue-bound mode" (brief includes `issue: <N>`) and post their progress/results/failures as marker comments, not just returned messages.
- For <30 min trivia (typo fix, config tweak, one-line bug fix), skip the ceremony and just do it.
- Never auto-merge PRs. Never auto-edit `RESULTS.md` even after reviewer PASS — propose the diff in an `<!-- epm:results-md-diff -->` comment, wait for user approval.

**Key files:**
- `.claude/skills/issue/SKILL.md` — procedure
- `.claude/skills/issue/markers.md` — marker taxonomy (source of truth for parsing state)
- `.claude/skills/issue/clarifier.md` — clarifier prompts per issue type
- `.claude/skills/issue/templates/` — plan + results comment templates
- `.github/ISSUE_TEMPLATE/` — issue creation templates
