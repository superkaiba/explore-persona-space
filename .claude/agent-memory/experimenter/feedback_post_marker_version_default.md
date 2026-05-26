---
name: post-marker-version-default
description: `task.py post-marker` defaults version=1; pass --version N explicitly for second+ instances of the same kind on a task
metadata:
  type: feedback
---

`scripts/task.py post-marker <N> <kind>` defaults `--version 1`. There is NO auto-increment — if `epm:run-launched v1` and `v2` already exist on the task and you post a third without `--version 3`, the new row records as v1 again. The skill's "latest-version-per-kind wins" rule is a CONVENTION enforced by readers, not by post-marker.

**Why:** Task #382 round-3 relaunch (seeds 137+256 after step-3125 OOM) was posted as `epm:run-launched v1` instead of `v3` on 2026-05-26. Timestamp ordering and the body's "Round 3" prefix kept it interpretable, but `latest-marker` queries that pick by `(kind, max(version))` would return the wrong row.

**How to apply:** Before calling `task.py post-marker <N> <kind>`, run:
```bash
uv run python scripts/task.py view <N> --json \
  | jq '[.events[] | select(.kind == "<kind>") | .version] | max // 0'
```
Then pass `--version <max+1>`. The same applies to `epm:results`, `epm:code-review`, `epm:interp-critique`, `epm:plan`, `epm:clean-result-critique`, etc. — every kind that supports multi-version threading.

Related: [[orchestrator-vs-subagent-reinvocation]] (subagents have one turn; cant rely on a second pass to fix).
