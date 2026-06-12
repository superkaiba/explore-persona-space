---
name: anthropic-sonnet-4-5-20251001-invalid-model
description: Hardcoded Anthropic model id 'claude-sonnet-4-5-20251001' returns 404 NotFoundError; this id never shipped (alias is 'claude-sonnet-4-5'). Spot it on sight in phase scripts; code-class bounce.
metadata:
  type: feedback
---

# Anthropic model id `claude-sonnet-4-5-20251001` is invalid → 404

## Rule

If a script hardcodes the Anthropic model id `claude-sonnet-4-5-20251001`,
expect `anthropic.NotFoundError: Error code: 404 - {'type': 'error', 'error':
{'type': 'not_found_error', 'message': 'model: claude-sonnet-4-5-20251001'}}`
on the first `client.messages.create(model="claude-sonnet-4-5-20251001", ...)`
call. The dated id never shipped — the released alias is plain
`claude-sonnet-4-5` (and the prior dated form `claude-sonnet-4-20250514` for
the older Sonnet 4 line).

## Why

Sonnet 4.5 ships as the alias `claude-sonnet-4-5`. The `-20251001` suffix
was a plausible-looking dated-id (matches the YYYYMMDD pattern of other
Anthropic dated ids) but Anthropic never published a model snapshot with
that date. It's a copy-paste hallucination, NOT a real id.

## How to apply

- Before launching any i489-family / judge-using pipeline, grep the
  dispatcher + every helper it imports for the string
  `claude-sonnet-4-5-20251001`. If present, do NOT launch — post
  `epm:failure v1` with `failure_class: code` and bounce to
  `experiment-implementer`. The fix is a one-liner (swap to
  `claude-sonnet-4-5` alias), but it's a code edit and code edits never
  happen on pods.
- Affected sites tend to be plural (Phase 0a identity check, Phase 4 eval
  judge, smoke calibration judge). Recommend the implementer grep the
  repo for the bad id when fixing.
- Symptom is fast-crash within ~40s of launch on the first Anthropic
  call. Both parallel launch attempts will fail identically (deterministic
  on model id, not transient API flake) — do NOT retry, it will just
  burn the same way.

Burned at #489 v1 launch (2026-06-04).
