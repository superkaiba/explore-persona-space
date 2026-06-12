---
name: Stale tmp files across plan versions
description: Canonical /tmp/codex-critic-<N>-<lens>-{prompt,output}.md paths collide across plan versions — prefer version-suffixed paths; otherwise move stale output aside before returning dispatch config
type: feedback
---

The canonical paths `/tmp/codex-critic-<N>-<lens>-{prompt,output}.md` are keyed by task + lens only, NOT plan version. On a re-critique at a later version, the stale prompt blocks a clean Write (read-before-overwrite) and a stale OUTPUT file can be read by the orchestrator as fresh Codex output if the new dispatch fails or is slow.

**Why:** #537 v4 statistics dispatch (2026-06-09) found a 100 KB v2 prompt and a 3.4 KB v2 output already at the canonical paths.

**How to apply (preferred, when the brief names a plan version):** mint version-suffixed paths up front — `/tmp/codex-critic-<N>-v<plan_version>-<lens>-{prompt,output}.md`. Fresh paths can't collide, old artifacts stay for forensics, no mv needed; state the non-canonical paths explicitly in the dispatch config. (Used #537 v5.)

**Fallback (canonical paths):** `ls` both paths before returning the config; if an output file predates this dispatch, `mv` it to `*-output.v<old>-stale.md` so the orchestrator can only read output the new run wrote. Overwrite the prompt normally (Read a few lines first to satisfy the overwrite check).
