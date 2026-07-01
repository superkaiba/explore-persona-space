---
name: Write/Edit can silently no-op — verify body landed on disk before verifier/promote
description: On a re-spawn, Write/Edit to the canonical cache path reported success but the disk file was unchanged (stale md5/mtime); verify on disk, write to a fresh path + cp if needed.
type: feedback
---
Write/Edit tool calls to the canonical cache path (`.claude/cache/experiment-<N>-clean-result.md`) can report "success" / "file state is current" while the ON-DISK file is UNCHANGED (identical md5, unchanged mtime). Observed on #761 round-2 finalize (2026-07-01): a full `Write` of the revised body + several `Edit`s all reported success, but `md5sum`/`stat`/`grep` showed the disk file was still the round-1 draft. `Read` of the same path returned my intended (in-memory) content, disagreeing with `grep`/`wc` on disk — the classic silent cache→disk handoff divergence (#385 class).

**Why:** the harness keeps an in-memory file-state view; when it believes the target already matches (or on a re-spawn where its cached view is stale), the actual disk write is skipped. Verifier + set-body then run against a DIFFERENT file than the one I "wrote".

**How to apply:**
1. NEVER trust the Write/Edit "success" message for a load-bearing body. After writing, IMMEDIATELY confirm on disk with shell: `md5sum <file>; stat -c %y <file>; grep -n '^## \|^### ' <file>`. If mtime/md5 didn't change or headings are the OLD ones, the write silently failed.
2. Recovery that WORKED: Write to a FRESH filename (e.g. `...-v2.md`) — the Write tool DOES land on a new path — confirm it on disk, then `cp fresh canonical` via Bash. Re-run the verifier against the canonical path afterward.
3. Do all the tighten/trim Edits on the fresh path, `cp` to canonical, then verify — keep the fresh path as the working copy so every Edit lands.
4. Run `verify_task_body.py`/`_finding_prose_cap_results` against the CANONICAL path (post-cp), not the fresh one, so you gate the file set-body will read.
5. The Step-6 pre/post-flight grep checks exist for exactly this — do NOT skip them; they are the last line before promoting a stale body.
