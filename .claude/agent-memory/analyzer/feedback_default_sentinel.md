---
name: Avoid the word "default" in repro card cells
description: Verifier flags `default` as an unfilled-sentinel; use specific phrasing instead
type: feedback
---

The `verify_clean_result.py` validator flags the bare word `default` in any
table row of `## Setup & hyper-parameters` as an unfilled sentinel
(alongside `{{`, `TBD`, `see config`).

**Why:** "default" hides the actual value — readers cannot reproduce
without knowing what the default IS. The validator forces you to spell it
out.

**How to apply:** When describing chat templates, optimizer settings, or
any config field, never write "default X". Either (a) name the specific
template/value (e.g. "Qwen3 stock chat template", "AdamW β=(0.9, 0.999)"),
or (b) inline the explicit setting (e.g. "no system message;
`add_generation_prompt=True`"). Common offenders: "default Qwen template",
"default LR schedule", "default optimizer".
