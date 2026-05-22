---
name: pipefail-grep-nomatch
description: Under set -euo pipefail, grep -l with no matches exits 1 and aborts the script via ERR trap — a common false-safe pattern when porting scripts to pipefail.
metadata:
  type: feedback
---

Under `set -euo pipefail`, a pipeline like `find ... | xargs grep -l 'SENTINEL' | wc -l` will abort the enclosing shell (ERR trap) when grep finds NO matching files, because `grep -l` exits 1 on no-match and pipefail propagates that through `wc -l` to the subshell's exit code. Assignments (`count=$(...)`) propagate the subshell's exit code under `set -e` in bash 5.x.

**Why:** This bit task #365 round-7: the `count_complete_cells` function was labeled "pipefail-safe" but was not — it only handled the zero-files-found case (xargs --no-run-if-empty), not the files-exist-but-no-match case. The watchdog would abort on the first poll cycle during normal early-dispatch before any cell completes.

**How to apply:** When reviewing scripts that add `set -euo pipefail`, grep for `grep -l` or `grep -c` inside `$(...)` assignments or pipelines. Always verify the no-match exit code is neutralized, e.g.:
- `{ find ... | xargs ... grep -l ...; } | wc -l` — curly brace group isolates grep's exit from wc
- `find ... | xargs ... grep -l ... 2>/dev/null || true | wc -l` — `|| true` must be inside a subgroup before the pipe
- Safest: `| wc -l` after a `{ ...; } || true` wrapper on the xargs/grep stage

The pattern `xargs --no-run-if-empty grep -l ... | wc -l` only helps when there are ZERO input files to xargs. It does not help when files are found but none match.
