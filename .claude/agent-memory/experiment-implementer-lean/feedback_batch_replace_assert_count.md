---
name: batch-replace-assert-count
description: "python str.replace batch edits must assert count==1 per hunk; a mid-line docstring mismatch no-ops silently and ships the unedited function"
metadata:
  type: feedback
---

Every scripted batch edit (`uv run python - <<PY` with `s.replace(old, new, 1)`)
must `assert s.count(old) == 1` BEFORE each replace.

**Why:** #2658 round 15: a capture_fingerprint rewrite targeted a docstring
window starting mid-line ("Schema v2 adds them" sits after "shards.  " on the
same line), so `replace` matched nothing, the script printed its success line,
and the unedited function shipped until a unit test caught the stale sha. A
silent no-op edit is worse than a failed one.

**How to apply:** assert count per hunk (count==1, or the intended N for
replace_all); after the write, grep the file for one NEW-text token as the
landed check. The Edit tool with a verbatim old_string raises on miss and is
the safer instrument for single hunks under the PostToolUse formatter.
