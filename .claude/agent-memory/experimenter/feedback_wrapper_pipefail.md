---
name: Launch wrappers MUST set -euo pipefail (tee masks exits; brace chains need -e)
description: Two burned variants — `set -e` alone is defeated by `| tee` (pipeline exits with tee's 0), and `set -uo pipefail` without -e lets a failed smoke proceed to the sweep. Every launcher starts with set -euo pipefail.
metadata:
  type: feedback
---

Every launch wrapper this agent writes MUST start with `set -euo pipefail`. Two distinct masks, both burned:

- **tee variant (#381):** `set -e` + `uv run ... 2>&1 | tee -a log` — the pipeline's exit is `tee`'s (always 0), so the dispatcher's preflight RuntimeError (unversioned `JUDGE_MODEL`) was swallowed and the wrapper marched into dataset-gen; had to kill the PID by hand. `-o pipefail` is the fix.
- **brace-chain variant (#505 v1, 2026-06-06):** a `{ smoke; sweep; }` block under `set -uo pipefail` (no `-e`) let a smoke crash (`KeyError: 'schema_version'`) fall through to the sweep AND printed the launcher's "=== SMOKE PASSED ===" echo, misleading the post-launch log read. Only `-e` (errexit) halts a brace block on the first non-zero command.

**How to apply:** when an `epm:failure code` comes out of a chained-launcher round, name the launcher-hygiene fix as a second item in the failure note so the next re-launch cycle patches both the experiment bug AND the wrapper — otherwise the next failure reproduces the same misleading "smoke passed" line. If a phase script raises, the wrapper must die immediately so `poll_pipeline.py` sees the dead PID. Related: [[feedback_load_env_in_nohup]].
