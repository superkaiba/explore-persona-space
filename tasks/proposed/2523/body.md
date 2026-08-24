---
title: 'codex-code-reviewer composes a CONCERN:: row template its own forwarder cannot
  parse, breaking the concerns round-trip'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T07:55:33Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2327 round-1 code-review verdict collection: persist_verdict_concerns.py
  returned MALFORMED bad-severity/bad-id on both Codex CONCERN rows because the composed
  prompt (line 1704) handed Codex the key-value row form while line 1710 of the same
  prompt states the forwarder position-parses; canonical positional grammar is codex-code-reviewer.md:625'
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
---

# `codex-code-reviewer` composes a `CONCERN::` row template that its own forwarder cannot parse, breaking the concerns round-trip

## Goal

Make the machine-readable `CONCERN::` row grammar single-sourced and mechanically enforced, so a composed Codex review prompt can never hand the model a row template that `scripts/persist_verdict_concerns.py` rejects.

## The defect

Three surfaces disagree about the `CONCERN::` row grammar, and nothing checks them against each other.

1. **The forwarder POSITION-parses.** `scripts/persist_verdict_concerns.py` docstring: "field order fixed (token 1 = severity, token 2 = concern id, remainder = summary; whitespace-split on the first two tokens only)". Validation is against `task_workflow.CONCERN_SEVERITIES` (`{BLOCKER, CONCERN, NIT}`) and `task_workflow._CONCERN_ID_RE` (`^[a-z0-9][a-z0-9-]{1,79}$`).
2. **The composer spec agrees** — `.claude/agents/codex-code-reviewer.md:625` states the canonical positional form:
   `CONCERN:: <BLOCKER|CONCERN|NIT> <kebab-case-id> <one-line summary, aim <=180 chars>`
3. **But a composed prompt can carry the KEY-VALUE form instead**, and did: `/tmp/codex-code-review-2327-r1-prompt.md:1704` handed Codex
   `CONCERN:: concern_id=<kebab-case-id> severity=<BLOCKER|CONCERN|NIT> round=1 by=codex-code-reviewer summary='<...>'`
   while **line 1710 of the same prompt** told it "the forwarder position-parses `^CONCERN:: ` rows". The prompt contradicts itself six lines apart.

Position-parsed, a key-value row puts `concern_id=<id>` in the severity slot and `severity=CONCERN` in the id slot — identically on every row, so a two-row block fails as `bad-severity` + `bad-id` on both plus `duplicate-id` on the second (the two rows' token-2 are the same literal). That is exactly what was observed.

## Observed instance (#2327 round 1, 2026-08-24)

The Codex twin returned a well-formed `CONCERNS` verdict with two genuine, useful CONCERN rows. `persist_verdict_concerns.py 2327 --file <block> --by codex-code-reviewer --round 1 --require-block --validate-only` returned:

```
MALFORMED: row 1: bad-severity
MALFORMED: row 1: bad-id
MALFORMED: row 2: bad-severity
MALFORMED: row 2: bad-id
MALFORMED: row 2: duplicate-id
```

**Nothing was lost** — the forwarder fail-closed as designed, and the orchestrator persisted both rows by hand with `task.py raise-concern` (plus the two Claude-side rows). Cost was one diagnosis plus four manual calls inside a round whose scope was unrelated. But the round-trip is broken by construction for any composer run that reproduces that template, and the manual fallback is exactly the kind of hand-transcription the blind-forward path exists to eliminate.

**Contributing factor, recorded rather than elided:** the #2327 orchestrator's composer brief ALSO stated the key-value form in its output contract, and the composer propagated the caller's shape rather than overriding it with its own spec's line-625 grammar. Both halves are in scope: a caller brief must not be able to break the forwarder contract, and the composer must not emit a self-contradictory prompt.

## Scope to investigate

1. **Single-source the grammar.** The row template belongs in exactly one place that the composer spec, the forwarder, and any brief-facing documentation all derive from — not three hand-maintained copies. Candidate: the forwarder exports the canonical template string and the composer spec quotes it by reference, or a lint check asserts byte-equality of the grammar line across the surfaces.
2. **Make the composer authoritative over its own output contract.** A caller brief may scope WHAT to review; it must not be able to redefine the machine-readable row grammar. Consider having the composer emit the grammar block from its spec unconditionally and ignore/refuse a caller-supplied row template.
3. **Mechanical check for the internal contradiction.** A composed prompt that contains both a key-value `CONCERN::` template and the phrase "position-parses" is detectably inconsistent. A cheap `workflow_lint.py` check over `.claude/agents/codex-*.md` (and, if reachable, the composed prompt) would have caught this before dispatch.
4. **Consider a forwarder-side tolerance decision, explicitly.** Either (a) keep strict position-parsing and fix the producers, or (b) accept both shapes with a documented normalization step. Do NOT do both by accident. Recommendation from the incident: (a) — the positional form is what every persisted ledger row in the repo already uses, so tolerance would add a second lifetime-supported grammar for no benefit.
5. Check the sibling composer `.claude/agents/codex-clean-result-critic.md`, the other `--require-block` contract site, for the same divergence.

## Non-goals

Do not relax `persist_verdict_concerns.py`'s validation to make the symptom disappear — the fail-closed refusal is the gate working correctly, and it is what prevented a garbage ledger. Do not change `task_workflow.CONCERN_SEVERITIES` or `_CONCERN_ID_RE`.

## Provenance

Surfaced by the #2327 orchestrator during round-1 code-review verdict collection. Confidence: high — root-caused with the exact validator output, the contradictory prompt line numbers, and the canonical spec line all read first-hand. Dedup target `.claude/agents/codex-code-reviewer.md` + `scripts/persist_verdict_concerns.py`, distinct from #2327's own target surface (`.claude/skills/issue/steps/09-step-5.md`, `scripts/workflow_lint.py`).
