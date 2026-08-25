---
name: hf-call-shape-prose-trips-lint
description: Docstring/comment prose in src/ or scripts/ spelling an HF call shape like `.upload_file(...)` trips workflow_lint's live-hf-retry-routing scanner — defuse the spelling (drop the parens)
metadata:
  type: feedback
---

Never spell a parenthesized HF Hub call shape — `.upload_file(`, `.upload_folder(`,
`hf_hub_download(`, `create_commit(`, `push_to_hub(` — in DOCSTRING/COMMENT prose of any
file under `src/explore_persona_space/` or `scripts/`. `workflow_lint.py
--check-live-hf-retry-routing` (bundled in the no-flags Step 9c gate run) is a TEXTUAL
line scanner (`HF_ROUTING_CALL_RE`, workflow_lint.py ~:9901) over those roots; it cannot
tell prose from code, and only `scripts/workflow_lint.py` + `scripts/verify_plan.py` are
pattern-string-exempt.

**Why:** #2261 (2026-08-21): the argcheck module docstring's FN-6 example
``self.api.upload_file(...)`` FAILed the whole no-flags lint at argcheck.py:128 — the
#1723 prose-literal class, producer side; cost one extra commit + a second ~10-min lint
run.

**How to apply:** in prose, name the callable WITHOUT the call parens
(``self.api.upload_file`` or "a later upload call through ``self.api``"). tests/ and
`.md` files are out of the scanner's scope roots. After any src/scripts prose edit
mentioning HF upload/download verbs, grep the file with the scanner's own regex
(`\.upload_file\s*\(` etc.) or run `workflow_lint.py --check-live-hf-retry-routing`
(~2 min) before the full no-flags run. Related: [[reference_preexisting_lint_test_failures]].
