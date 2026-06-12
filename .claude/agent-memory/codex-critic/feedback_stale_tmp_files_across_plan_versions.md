---
name: Stale tmp files across plan versions
description: Canonical /tmp/codex-critic-<N>-<lens>-{prompt,output}.md paths collide when the same task gets re-critiqued at a later plan version — check for and move aside stale files before returning dispatch config
type: feedback
---

The canonical dispatch filenames `/tmp/codex-critic-<N>-<lens>-prompt.md` and
`/tmp/codex-critic-<N>-<lens>-output.md` are keyed by task number + lens only,
NOT by plan version. When a task's plan is re-critiqued at a later version
(observed: #537 v2 round left both files behind when the v4 round started),
the stale prompt blocks a clean Write (read-before-overwrite) and — worse —
a stale OUTPUT file can be read by the orchestrator as if it were fresh Codex
output if the new dispatch fails or is slow.

**Why:** #537 v4 statistics dispatch (2026-06-09) found a 100 KB v2 prompt and
a 3.4 KB v2 output already at the canonical paths.

**How to apply:** Before returning the dispatch config, `ls` both canonical
paths; if an output file predates this dispatch, `mv` it to
`*-output.v<old>-stale.md` so the orchestrator can only ever read output the
new Codex run wrote. Overwrite the prompt file normally (Read a few lines
first to satisfy the overwrite check).

**Cleaner variant (preferred when the brief names a plan version, used #537
v5 2026-06-09):** mint version-suffixed paths up front —
`/tmp/codex-critic-<N>-v<plan_version>-<lens>-{prompt,output}.md` — instead
of moving stale files aside. Fresh paths can never collide with a prior
round's output, the old round's artifacts stay intact for forensics, and no
mv is needed. State the non-canonical paths explicitly in the dispatch
config so the orchestrator reads the right file.
