---
name: Codex twin copy-list omissions
description: When fixing a gate/rubric in code-reviewer.md (or any ensembled Claude reviewer spec), also check the codex twin's "copy the substantive sections" list — an omitted step there means the fix never reaches Codex and it re-derives its own narrower rule.
type: reference
---

The codex twin wrappers (`codex-code-reviewer.md` etc.) compose their prompt
by copying an enumerated list of sections from the Claude spec (e.g.
`codex-code-reviewer.md` "Step 2: Compose the review prompt" lists Steps 0,
0.5, 0.6, 0.7, 0.8, 1, 2, 3, 5, 6, 7 + an `{{INLINED RUBRIC ...}}`
placeholder + a blocker-tag enum in the verdict template).

**Failure mode (incident #606):** Step 0.65 (raw-completions upload gate)
existed in `code-reviewer.md` but was absent from the codex copy list — Codex
saw only Step 0.7's bare reference to "0.65", re-derived a narrower
call-shape check from CLAUDE.md's Upload Policy, and FAILed a functionally
stronger batched upload. Fixing only the Claude spec would NOT have fixed the
recurring dispute.

**How to apply:** any edit to a Claude reviewer gate must touch THREE codex
sites if present: (1) the copy-list bullet enumeration, (2) the
`{{INLINED RUBRIC FROM ... Steps ...}}` placeholder step list, (3) the
verdict template's blocker-tag enum (if the gate has a blocker tag). Grep the
twin for the step number AND the blocker tag before declaring the fix
complete.
