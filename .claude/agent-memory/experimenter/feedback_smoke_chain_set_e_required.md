---
name: Smoke-then-sweep chained launcher MUST use set -e
description: When chaining smoke + sweep in one nohup brace block, set -uo pipefail is NOT enough; must include -e or smoke failure misleadingly proceeds to sweep
type: feedback
---

When the brief says "chain smoke then sweep in ONE nohup invocation" (a single
launcher script that runs `{ smoke; sweep; }` in a brace block), the
launcher MUST use `set -euo pipefail` — NOT `set -uo pipefail`. Without
`-e`, a non-zero exit from the smoke phase does NOT halt the brace block, so
the sweep auto-launches against the same broken code, and the launcher prints
its post-smoke "=== SMOKE PASSED ===" echo even though smoke crashed.

**Why:** `set -e` (errexit) is the ONLY flag that makes a bash brace/group
block bail on the first non-zero command. `-u` catches unset vars and
`-o pipefail` catches mid-pipeline failures; neither halts on a plain
`python script.py` exiting non-zero.

**How to apply:** Every launcher script that chains a gate (smoke / preflight /
data-staging check) before the expensive sweep MUST start with:

```bash
set -euo pipefail
```

If the launcher uses an inline brace block `{ smoke; sweep; } > LOG 2>&1`,
that block ALSO inherits the surrounding shell options — but it's worth
double-checking. The `experimenter.md` launcher pattern already uses `exec
uv run python ...` as a single command (no chaining), so this trap is
specific to multi-phase chained launchers.

**Incident:** task #505 v1 launch (2026-06-06). Smoke crashed in <10s with
`KeyError: 'schema_version'` in `panel_coverage.py`. Launcher had
`set -uo pipefail` (no `-e`), so the brace block continued, the sweep
auto-launched, and crashed identically within another 5s. Both phases dead
within 20s — no GPU time wasted in this case because the failure was
deterministic-and-fast, but on a slower-crashing smoke the sweep would have
burned full GPU-hours against broken code. The launcher's
`=== SMOKE PASSED ===` echo line also misled the post-launch log inspection
by claiming smoke passed.

**Mention this fix as the second item in any `epm:failure code` note** that
came out of a chained-launcher round, so the next experimenter spawn fixes
BOTH the experiment-code bug AND the launcher hygiene in one re-launch
cycle. Otherwise a re-launch after a code fix will inherit the same
no-`-e` launcher and the next failure mode will produce the same misleading
"smoke passed" line.
