---
name: Unapplied Proposal Backlog
description: Running backlog of retrospective proposals that have been drafted but not yet applied; triggers a cross-retro audit
type: project
---

3 days in a row (2026-04-15, 2026-04-16, 2026-04-17) the daily retrospective has produced proposals that never get applied.

**Why:** The retro writes to `research_log/drafts/retrospective-*.md`. Nobody reads it before the next session. CLAUDE.md + agent definitions drift from intended state.

**How to apply:** At the start of every retro run, open the last 2 retros, and mark each proposal as ✅ APPLIED / ⏳ DEFERRED / ❌ NOT APPLIED. Lead the new retro with a "Proposal Backlog" table. Also create a GitHub issue with `--label retrospective` so the proposals have a tracking surface outside the dead-letter drafts directory.

**Known unapplied backlog as of 2026-04-17:**

| Proposal | First proposed | Target | Status |
|---|---|---|---|
| SessionStart hook for retro visibility | 2026-04-15 | `.claude/settings.json` | ❌ |
| PostToolUse hook on `git push` for pod-sync reminder | 2026-04-15 | `.claude/settings.json` | ❌ |
| RunPod Community Cloud port instability gotcha | 2026-04-15 | `CLAUDE.md` | ❌ |
| HF Hub public repo warning | 2026-04-15 | `CLAUDE.md` | ❌ |
| Upload/cleanup safety gotcha | 2026-04-15 | `CLAUDE.md` | ❌ |
| Worktree subagents clean-git-state rule | 2026-04-16 | `CLAUDE.md` | ❌ |
| uv-in-nohup absolute path | 2026-04-16 | `CLAUDE.md`, `experimenter.md` | ❌ |
| experimenter.md Design Brief | 2026-04-16 | `.claude/agents/experimenter.md` | ❌ |
| experimenter.md Monitoring Cadence (ScheduleWakeup over tight polling) | 2026-04-16 | `.claude/agents/experimenter.md` | ❌ |
| research-pm.md Gate-Keeper Trigger | 2026-04-16 | `.claude/agents/research-pm.md` | ❌ |
| research-pm.md Agent Name Reference | 2026-04-16 | `.claude/agents/research-pm.md` | ❌ |
| retrospective.md should open GH issues | 2026-04-16 | `.claude/agents/retrospective.md` | ❌ |

**What HAS been applied across this window:**
- `/issue <N>` skill built and shipped (2026-04-16, partial address of GH Issues migration)
- settings.json `"agent": "research-pm"` (2026-04-16, agent architecture transition)
- Gate-keeper started being invoked on Tier 2 (partial compliance improvement)
