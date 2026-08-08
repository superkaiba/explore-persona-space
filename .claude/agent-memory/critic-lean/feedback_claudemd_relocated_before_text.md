---
name: claudemd-relocated-before-text
description: Plans editing CLAUDE.md prose — grep the BEFORE fragment across CLAUDE.md AND .claude/rules/ before trusting the plan's stated location; the 2026-08 compaction relocated most bullets into rule files
metadata:
  type: feedback
---

For any workflow-fix plan whose diff targets a CLAUDE.md paragraph, verify the quoted BEFORE fragment's LOCATION and WORDING against the live tree with one grep across `CLAUDE.md` + `.claude/rules/` — never trust the plan's file:line or the task body's verified-at-filing evidence (it can be days stale).

**Why:** the 2026-08 workflow compaction moved most CLAUDE.md detail bullets into `.claude/rules/*.md`, leaving summaries + pointers in CLAUDE.md. #2003's plan v1 targeted "CLAUDE.md line 150" for the refusal-ladder rung (e); the live paragraph is `.claude/rules/context-hygiene.md:13`, and the plan's BEFORE fragment ("not only post-kill retry briefs") was a paraphrase existing nowhere — the diff could not apply and the acceptance grep was unsatisfiable. The task body's own grep evidence was 6 days pre-relocation.

**How to apply:** run `grep -rn "<distinctive BEFORE phrase>" CLAUDE.md .claude/rules/` first; a 0-hit in the plan's named file + a hit elsewhere = Must-Fix (retarget file, requote BEFORE verbatim, retarget any acceptance grep keyed on the stale file). Also verify secondary anchors (e.g. "append after the (b2) reference") exist in the named region — see [[prose-pin-test-plans]] for the acceptance-grep sweep.
