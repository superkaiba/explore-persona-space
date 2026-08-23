---
name: spec-inline-brace-false-positive
description: Scope the Step-4 unsubstituted-{{...}} placeholder guard to composed parts + envelope spans — inlined SPEC.md legitimately contains literal `{{` rule text and template mentions.
metadata:
  type: feedback
---

Scope the Step-4 `{{...}}`-placeholder guard to (a) envelope spans (the
spec-mandated check) and (b) the composer-authored parts (header /
instructions / output template) — NEVER the whole assembled prompt.

**Why:** #2215 r2 compose (2026-08-23) — a whole-prompt `grep -qF '{{'`
tripped on six hits that were all verbatim inlined SPEC.md text: the
spec's own "No `{{`, `TBD`, `default`, `see config` sentinels" rules
(×4), the `{{JUDGE_PROMPTS}}` paper-template placeholder mention, and
the `issue_TEMPLATE.tex # parameterized {{...}}` comment. Verbatim-copy
of SPEC.md is mandatory, so these hits recur on EVERY markdown-branch
compose; "fixing" them would corrupt the inlined spec.

**How to apply:** validate placeholders per scope: envelope bodies via
the awk-extracted span check (spec Step 4), plus a `grep -cF -e '{{'
-e '}}'` over each /tmp part file the composer WROTE (expect 0). A hit
inside the inlined lens-reference / SPEC.md region is spec-native text
— leave it. Related assembly pattern that avoids placeholders entirely:
build the prompt by concatenating fully-substituted part files + cat of
captured outputs (no template-then-substitute pass), so the only {{ }}
that can exist are inlined-reference-native. See
[[fold-round-context-file-briefs]] for the fold-round envelope roster
this guard runs over (4 envelopes when the full ledger ships).
