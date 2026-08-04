---
name: edit-then-read-modify-write-lost-update
description: A python read-modify-write script run right after Edit-tool changes can read the file BEFORE the format hook finishes writing, then clobber those edits — inspect.getsource then shows code you just wrote as absent
metadata:
  type: feedback
---

Do NOT follow Edit-tool changes with a python read-modify-write script on the
SAME file in the same breath. The PostToolUse format hook rewrites the file
asynchronously after an Edit; a script that does `p.read_text()` … `p.write_text()`
can capture a pre-hook snapshot and then write it back, silently reverting the
Edit while keeping the script's own replacements. Nothing errors.

**Why:** #1345 (2026-07-31). Two Edit calls added a lstrip block and a span-key
comparison to `assemble_row`; a follow-up `str.replace` script then wired the
tally counters. The script's write clobbered the span-comparison block but kept
the tally line that referenced it, so `keep_rows` counted a drop reason
`assemble_row` could no longer produce. Three previously-green capture tests went
red, and the diagnostic signature was genuinely confusing:
`inspect.getsource(gen.assemble_row)` reported code I had just written as ABSENT
while a sibling edit to the same function was present.

**How to apply:**
- One mechanism per file per step. Either all Edit calls, or one script that does
  the whole read-modify-write — never Edit-then-script on the same file.
- When a script IS the right tool (many mechanical replacements), do the Edits
  FIRST, then verify on disk (`grep -c <marker>`) BEFORE the script reads, so a
  clobber is visible immediately rather than at test time.
- After any mixed sequence, `grep -c` every marker you believe you added. A count
  of 1 where you expect 2 (e.g. the tally reference present but the producing
  branch gone) is this bug.
- Symptom to recognize fast: `inspect.getsource` shows a function WITHOUT an edit
  you just made, while a DIFFERENT edit to the same function is present. That
  asymmetry means a lost update, not a stale `.pyc` (a stale pyc would not affect
  `getsource`, which reads the source file) and not an import cache.
- Prefer `assert <old> in s` guards in any such script — they turn a lost update
  into a loud failure instead of a silent revert. (They did fire for a later
  attempt in the same session, which is how the pattern got named.)

Related: [[feedback_stale_pycache_masks_signature_change]] (the other
"code I wrote seems absent" class — that one is bytecode, this one is the file);
[[feedback_write_tool_lands_in_session_cwd]].
