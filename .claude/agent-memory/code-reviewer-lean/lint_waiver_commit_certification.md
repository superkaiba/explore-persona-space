---
name: lint-waiver-commit-certification
description: Certify a lint-waiver commit by calling the check FUNCTION directly on parent blobs vs HEAD with legacy_allowlist={} — proves pre-fix red at the exact lines AND that waivers (not the allowlist) silence HEAD
metadata:
  type: feedback
---

Certify a "waiver comments added to silence workflow_lint check X" commit with a
two-arm function-level probe, not just a green flag run:

```bash
mkdir -p /tmp/probe/scripts
git -C <WT> show <sha>^:scripts/<f>.py > /tmp/probe/scripts/<f>.py   # per waived file
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from pathlib import Path; import workflow_lint as wl
print(wl.check_upload_file_in_loop(scripts_dir=Path('/tmp/probe/scripts'), legacy_allowlist={}))  # want: exactly the waived lines
print([e for e in wl.check_upload_file_in_loop(scripts_dir=Path('scripts'), legacy_allowlist={}) if '<issue-slug>' in e])  # want: []
"
```

**Why:** a green `--check-...` run alone can pass for the wrong reason — the file
covered by a grandfather allowlist entry, or the site out of the scanner's reach
(e.g. not lexically in-loop). Parent-blob red at the EXACT waived lines + HEAD
clean under an EMPTY allowlist attributes the PASS to the waivers themselves
(#2388 R2 g3). Sibling of [[fails-pre-fix-probe-parent-commit]], lint-specific:
the `scripts_dir=` / `legacy_allowlist=` override hooks most workflow_lint
checks expose make this a 5-line probe with no tree mutation.

**How to apply:** any commit whose payload is waiver/exempt comments
(`UPLOAD_LOOP_EXEMPT`, `UPLOAD_AS_FILE_EXEMPT`, ruff `noqa`-analogues in
workflow_lint). Also read the check's waiver-placement helper first: most
accept ONLY the call's first physical line or the immediately preceding
non-blank line — a two-line wrapped waiver or a comment stacked BELOW the
waiver is inert ([[stacked-lint-waivers-read-window]]). Truthfulness leg:
verify the stated bound against the loop's realized width (enumerate the
iterable's producers), and flag per-iteration re-uploads of a per-GROUP
artifact as a minor even when bounded.
