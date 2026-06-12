---
name: anthropic-sonnet-4-5-20251001-invalid-model
description: Hardcoded Anthropic id 'claude-sonnet-4-5-20251001' 404s (NotFoundError) — that dated snapshot never shipped; the alias is 'claude-sonnet-4-5'. Deterministic crash ~40s after launch; code-class bounce, don't retry.
metadata:
  type: feedback
---

The model id `claude-sonnet-4-5-20251001` never shipped — it's a plausible-looking dated-id hallucination; the released alias is plain `claude-sonnet-4-5`. Any `messages.create` with it raises `anthropic.NotFoundError` 404 on the first call (~40s after launch). Deterministic — retries burn identically.

**Why:** #489 v1 launch (2026-06-04), Phase 0a SP identity check; both parallel attempts failed identically.

**How to apply:** before launching any judge-using pipeline, grep the dispatcher + every imported helper for the bad id. If present, do NOT launch — post `epm:failure v1 failure_class: code` (code edits never happen on pods). Affected sites tend to be plural (identity check, eval judge, smoke calibration); recommend the implementer grep the whole repo when fixing.
