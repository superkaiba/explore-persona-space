---
name: wrapper-pipefail
description: nohup wrapper scripts that pipe through `tee` need `set -o pipefail`, not just `set -e`, or upstream non-zero exits are masked and the wrapper marches on past a failed phase
metadata:
  type: feedback
---

When the launch wrapper does `set -e` + `uv run ... 2>&1 | tee -a /workspace/logs/issue-N.log`, `set -e` alone is NOT enough. The pipeline's exit code defaults to the LAST command's exit (i.e. `tee`'s, which is always 0). When `uv run ...` raises `RuntimeError` and exits 1, the wrapper happily moves to the next phase.

**Why:** Observed in task #381 launch — dispatcher's `--phase preflight` correctly raised `RuntimeError` because `JUDGE_MODEL = "claude-haiku-4-5"` was missing from Anthropic's `models.list()`, but the wrapper proceeded to `--phase dataset-gen` because `tee -a` returned 0. Required killing the wrapper PID by hand.

**How to apply:** Every launch wrapper this agent writes MUST start with both:
```bash
set -e
set -o pipefail
```
or equivalently `set -euo pipefail`. Belt-and-suspenders for any wrapper that does `... | tee log`. If a phase script raises, the wrapper must die immediately so the orchestrator's `poll_pipeline.py` sees the dead PID and routes to `epm:failure v1`.

Related: [[load-env-in-nohup]] — wrappers need both env-sourcing AND pipefail.
