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

**Substring-grep variant (#823 r5cf, 2026-08-23):** a bare
`grep 'smoke-architecture' events.jsonl` matches NOTE MENTIONS too — a
code-review verdict body citing the string surfaced as an apparent
"smoke-arch v15" when the true latest `epm:smoke-architecture-check` was
v8 (kind-exact `latest-marker` fetch). Never resolve a marker's latest
version from a substring grep over events.jsonl; resolve by kind-exact
fetch (or JSON `kind ==` filtering), and byte-compare the fetched note
against any reused template envelope before attesting "unchanged".
