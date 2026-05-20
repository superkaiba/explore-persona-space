---
title: '[Proposed] Automatic cleanup agent (scheduled audit + sweep)'
kind: infra
tags: []
created_at: '2026-04-16T19:30:15.000Z'
has_clean_result: false
sagan_id: b3d6fba4-05f3-41a2-a9ac-921b7a871ec3
sagan_number: 13
priority: normal
legacy_why_unset: true
---
**From EXPERIMENT_QUEUE.md, added 2026-04-16**

Infra task. Currently cleanup (tracking-file audit, orphaned figures, stale draft detection, unuploaded-model sweep, disk-usage alerts, model-weight cleanup after upload) is manual — I run it when asked, or the user notices drift. Want an agent that runs on a schedule without explicit prompting.

**Scope per run:**
- (a) run `python scripts/pod.py health --quick` across all pods; flag low-disk or unreachable
- (b) run `python scripts/pod.py sync models --sweep` to upload any unuploaded checkpoints
- (c) run `python scripts/pod.py cleanup --all --dry-run` and surface candidates
- (d) AUDIT-mode scan: orphaned `eval_results/` dirs not in INDEX.md, drafts > 3 days unreviewed, figures not referenced from RESULTS.md / drafts, stale queue entries
- (e) produce a summary report to `research_log/cleanup/YYYY-MM-DD.md`

**Execution options:**
- (i) scheduled cron via the `schedule` skill (daily, e.g. 09:00 local)
- (ii) `/loop` in auto mode
- (iii) manual via `/cleanup` slash command

Start with (i).

**Safety:** read-only scan + dry-run by default; auto-apply only INDEX.md additions, stale flags, figure moves to `figures/unsorted/`. Anything destructive (deletes, model cleanup) requires user approval.

**Dispatch target:** implementer to build the agent + scheduled task config. No gate-keeper — infra.

**Compute:** 0 GPU. Each run ~2-5 min agent time.

**Priority:** MEDIUM (nice-to-have; drift doesn't compound dangerously at current pace but is accelerating as the project grows).
