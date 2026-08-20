---
name: latest-marker-prefix-collision
description: task.py latest-marker --prefix epm:code-review also matches epm:code-review-codex — fetch the Claude verdict by exact kind from events.jsonl (or a byte-verified /tmp copy)
metadata:
  type: feedback
---

`task.py latest-marker <N> --prefix epm:code-review` does PREFIX matching on
the kind, so it can return the `epm:code-review-codex` twin verdict instead
of the Claude composed verdict (whichever posted last wins).

**Why:** the two review-marker kinds share a prefix by construction; prefix
fetch is therefore ambiguous at every review site that posts both.

**How to apply:** fetch the Claude verdict by EXACT kind from events.jsonl
(`jq 'select(.kind=="epm:code-review" and .version==N)'`), or reuse a /tmp
copy only after byte-comparing it against the posted note. (This file was
recreated 2026-08-20 after the original was lost while its MEMORY.md index
row survived — index rows are pointers, not backups.) Related:
[[revision-round compose recipe]].
