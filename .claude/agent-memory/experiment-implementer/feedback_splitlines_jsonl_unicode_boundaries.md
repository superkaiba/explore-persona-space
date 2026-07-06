---
name: splitlines-shreds-jsonl-with-unicode-line-boundaries
description: Never read or count JSONL via str.splitlines() — raw U+2028/U+2029/NEL inside ensure_ascii=False JSON strings are Unicode line boundaries that shred valid records; use file iteration or split("\n")
type: feedback
---

Never read (or line-count) a JSONL file via `str.splitlines()`. It splits on
ALL Unicode line boundaries — U+2028 LINE SEPARATOR, U+2029 PARAGRAPH
SEPARATOR, NEL `\x85`, `\x0b`, `\x0c` — and those characters appear RAW inside
JSON strings whenever the writer used `json.dumps(..., ensure_ascii=False)`
over real-user text (lmsys-chat-1m, WildChat, any scraped corpus). The file is
valid JSONL; the READER shreds it.

**Why:** #825 run-1d (2026-07-03) crashed at `[phase=wiring]` on
`json.JSONDecodeError: Unterminated string` — 2000 valid `\n`-terminated
records read as 2019 splitlines() fragments, 20 unparseable, ~55 min of GPU
extraction lost to the re-run. A sibling `len(read_text().splitlines())` row
count feeding an `assert n_written == len(convs)` would false-fire on the same
content.

**How to apply:** parse JSONL with `for line in path.open()` (text-mode file
iteration splits only on universal newlines) or
`read_text().split("\n")` + an `if line.strip()` guard; count rows the same
way. Treat any `.splitlines()` over JSONL content — read OR count — as a
review blocker; `__doc__.splitlines()` etc. over known-ASCII is fine. Fix
commit: 9e821f906f (3 sites: dispatch heredoc, u2_gen count assert,
gen_conversations fixture parse).
