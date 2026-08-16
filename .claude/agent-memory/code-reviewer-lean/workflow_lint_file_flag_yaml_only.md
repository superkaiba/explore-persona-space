---
name: workflow-lint-file-flag-yaml-only
description: workflow_lint.py --file is a YAML-frontmatter schema mode (fails ParserError on .py); touched-file lint check = one tree-wide no-flags run + grep for the filenames
metadata:
  type: feedback
---

`scripts/workflow_lint.py --file <path>` parses the target as YAML frontmatter
(agent/skill schema lint) — feeding it a `.py` file yields
`schema FAIL / ParserError` that says nothing about the file. There is NO
per-file scope for the Python-surface checks.

**Why:** wasted a round-trip on #1336 R-delta trying to scope the
"no NEW lint failure names a touched file" check per file.

**How to apply:** for that check, run ONE tree-wide no-flags
`workflow_lint.py > /tmp/wl.out 2>&1` (takes minutes — give it a ~560s
foreground timeout, never two runs in one compound command; 420s WAS KILLED
mid-run on #2321 R1 g6) then
`grep -iE '<touched1>|<touched2>' /tmp/wl.out`; empty grep + only the known
pre-existing FAIL/ERROR lines = clean. NEVER pipe the run into grep in a
background Bash: the filter eats everything but matches, so a `timeout` kill
shows only a bare `Terminated`/rc=143 tail and the filtered output silently
impersonates a clean result — redirect to a file, grep the file. Sub-scope
fallback (#2321 R1 g6): when a split-review commit implicates exactly one
check class (e.g. a new scripts/ file touching HF ⇒
`--check-dotenv-before-hf-import`), the targeted flag runs in seconds and is
a legitimate bounded substitute; say so in the verdict. Also: a bracketed
`pgrep -f 'workflow_lin[t]'` liveness probe false-positives on OTHER sessions'
pytest argv containing `test_workflow_lint_*.py` — match the full script path
(`scripts/workflow_lint\.py$`) instead.
