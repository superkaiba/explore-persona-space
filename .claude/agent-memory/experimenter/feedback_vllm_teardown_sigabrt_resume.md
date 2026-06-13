---
name: vLLM teardown SIGABRT after stage completion — verify outputs, plain-relaunch
description: A vLLM sub-stage can SIGABRT at engine teardown AFTER its work is fully persisted; treat as resume, not data loss
type: feedback
---

A vLLM-backed sub-stage can complete ALL its scoring/generation (per-cell outputs
flushed, "sub-stage complete" logged) and then SIGABRT at engine teardown
(`terminate called without an active exception`, Signals.SIGABRT). A parent
dispatcher using `subprocess.run(check=True)` then aborts the whole chain even
though no data was lost.

**Why:** known vLLM worker-subprocess teardown gotcha (see `.claude/rules/gotchas.md`);
the abort happens during cleanup, after the stage's checkpoint writes.

**How to apply:** before escalating or re-running expensive stages, check the log
for the stage's own completion line ABOVE the traceback and verify per-cell outputs
exist on disk. If present, plain-relaunch the IDENTICAL command and let the
dispatcher's resume-skip carry the completed stages (incident #605, 2026-06-11:
gen+tf skipped with 0 pending, judge reached in <60s). Classify `failure_class: infra`,
never `code`.
