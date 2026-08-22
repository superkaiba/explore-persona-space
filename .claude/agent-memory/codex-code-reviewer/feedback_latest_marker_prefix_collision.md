---
name: latest-marker prefix collision on epm:code-review
description: task.py latest-marker --prefix epm:code-review also matches epm:code-review-codex; extract the Claude verdict by EXACT kind from events.jsonl when inlining prior-round verdicts
type: feedback
---

`task.py latest-marker <N> --prefix epm:code-review` returns the
`epm:code-review-codex` marker when it is newer — prefix match, not exact
kind. Hit on #958 r4 compose: both "claude" and "codex" fetches returned the
identical codex v3 body.

**Why:** `--prefix` is a string-prefix filter and `epm:code-review` is a
prefix of `epm:code-review-codex`.

**How to apply:** when a revision-round compose inlines BOTH prior-round
verdicts, fetch the Codex one via `--prefix epm:code-review-codex` (safe —
no longer kind extends it), and the Claude one by filtering
`events.jsonl` rows on `kind == "epm:code-review"` EXACTLY (path from
`task.py find <N>`), taking the max version. Never trust two prefix fetches
that return equal lengths — that is the collision signature.
