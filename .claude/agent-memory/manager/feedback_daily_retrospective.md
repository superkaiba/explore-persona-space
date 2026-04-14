---
name: Daily Retrospective Workflow
description: System crontab runs retrospective at 23:27 via claude -p; manager checks for pending drafts on startup
type: feedback
---

Automated retrospective is set up as a **system crontab** (not session-dependent):

- **23:27 nightly:** `crontab` runs `scripts/run_retrospective.sh` which invokes `claude -p --agent retrospective` headlessly. Output goes to `research_log/drafts/retrospective-YYYY-MM-DD.md`. Logs at `research_log/daily_updates/retrospective-YYYY-MM-DD.log`.
- **Script is idempotent:** skips if retrospective already exists for today.

On every session startup, the manager must:
1. Check for `research_log/drafts/retrospective-*.md` files that haven't been reviewed (no corresponding approved file in `research_log/`)
2. If found, present a concise summary + proposed diffs and ask the user to approve/modify/reject each change
3. Apply approved changes, note rejected ones

**Why:** User wants fully automated daily retrospective that runs without any active session, with approval gated to the next interactive session.

**How to apply:** Add to the startup read sequence — after reading EXPERIMENT_QUEUE.md, check for pending retrospectives before proceeding to normal status summary.
