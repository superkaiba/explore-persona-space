---
title: Cut ~35K tokens/spawn from the always-on context load (autocompact thrash)
kind: infra
tags: []
created_at: '2026-08-06T17:44:18Z'
has_clean_result: false
origin_prompt: autocompact is thrashing alot. Help me to reduce the amount of context
  loaded by each agent
workflow: v1
---
# Cut ~24K tokens/spawn from the always-on context load (autocompact thrash)

## Provenance

User chat, 2026-08-06: "autocompact is thrashing alot. Help me to reduce the
amount of context loaded by each agent."

## Goal

Reduce the fixed per-spawn context every agent pays, WITHOUT raising the 600K
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` (commit `a95a2d9a2a`, 2026-08-06 06:22 PT —
a deliberate Context-Rot decision, kept).

## Measurement

223 subagent spawns over 36h; startup input+cache tokens read off each agent
transcript's first `usage` record. Linear fit over four read-only agents whose
only variable is spec size:

| agent | payload B | startup tok |
|---|---|---|
| follow-up-critic | 13,255 | 93,812 |
| consistency-checker | 29,371 | 100,782 |
| interpretation-critic | 34,788 | 102,062 |
| upload-verifier | 51,173 | 108,398 |

→ **0.385 tok/byte, fixed intercept 88,712 tok** (max residual 772 tok).
Decomposing the intercept: CLAUDE.md 58.4K + globals 12.9K + LESSONS 3.0K +
CLAUDE.local 0.3K = 74.7K, leaving ~14K for harness/system/tool schemas.
**~84% of the fixed floor was markdown instruction files.**

Second cohort finding: sessions started 2026-08-05 spawn subagents at
190–260K for roles that cost 94–163K in 2026-08-06 sessions (~95K/spawn
delta — full MCP schemas vs ToolSearch deferral). Restarting long-lived
Aug-5 sessions is an ops action, tracked separately.

## Change

Landed in three commits: `40653b5dcf` (relocation), `d41f0f746a` (restore of
test-pinned prose), `4c11599638` (the two spec splits).

1. **CLAUDE.md 153,028 → 103,743 B.** Seven orchestrator-only sections
   relocated to `.claude/rules/` behind LESSONS.md triggers (the #829
   relocate-to-rules pattern): pods, context-hygiene, compute-backends,
   after-every-experiment, auto-continuation, codex-ensemble-review,
   disk-hygiene. Each leaves a load-bearing summary + a READ-before-you-act
   pointer; rule files carry the prior text verbatim. Anchor phrases
   preserved in the stubs so existing `CLAUDE.md §` pointers still resolve.
2. **Agent `skills:` frontmatter inlines the whole SKILL.md per spawn**
   (confirmed by regression: `implementer` and `experiment-implementer` have
   identical toolsets, and their derived bases agree to 5.6K tok only when
   skill bytes are counted). Dropped preloads the bodies already invoke
   lazily via the Skill tool: `implementer` −76,485 B, `research-pm`
   −94,634 B, `analyzer` −28,721 B.
3. **Two oversized agent specs split behind section references** — the two
   that had no section-reference partner: `methodology-writer` 48,962 →
   14,643 B, `upload-verifier` 51,173 → 32,462 B. Both now under the 40 KB
   FAIL cap, so both `AGENT_SPEC_SIZE_GRANDFATHER` entries were REMOVED (an
   entry whose spec drops under the cap FAILs as STALE — it must be removed,
   not lowered).
4. **Global personal files 33,531 → 19,444 B**, following the 2026-06-12
   MY_GOAT.md precedent: `USER.md` @-imported from `~/my-goat/CLAUDE.md`;
   `SOUL.md` split into an always-on operating core + `~/.claude/SOUL_DETAIL.md`.
   Nothing deleted; backups at `~/.claude/backups/20260806/`.
5. `_LESSONS_MAX_BYTES` 8000 → 9600 (index 9,286 B) — the new rules need
   index rows; ~1.3 KB of index bought ~49 KB out of the always-on body.
   Per-row and non-row caps unchanged. `_LESSONS_WARN_BYTES` stays 7200: a
   test pins it to the #992 latitude 7000–7400, so the index now carries a
   standing advisory WARN, which is true and intended.

## Result

**Always-on, every agent: −62,035 B ⇒ −23,865 tok/spawn** (CLAUDE.md −49,285,
globals −14,087, LESSONS index +1,337).

Additional per-agent, on top of that:

| agent | spec/skills Δ B | Δ tok/spawn |
|---|---|---|
| research-pm | −94,634 | −36,406 |
| implementer | −76,485 | −29,424 |
| methodology-writer | −34,319 | −13,203 |
| analyzer | −28,721 | −11,049 |
| upload-verifier | −18,711 | −7,198 |

## What went wrong, and the rule it produced

The first commit relocated prose that TESTS PIN to CLAUDE.md's always-on
surface, and left `main` red on 8 tests for several minutes. `d41f0f746a`
restored `## Experiment Report Structure` and the user-chat inline
free-analysis bullet VERBATIM and deleted the two rule files that had
absorbed them.

**The pin is the point.** The repo deliberately pins hard-won duty clauses to
the always-on surface via tests, because those duties get skipped otherwise.
The correct response to a pin-check failure is to restore the clause, never
to repoint the check at a rule file. That is why the two spec splits in
`4c11599638` ran behind a pin census (literals drawn ONLY from source files
referencing `<agent>.md`) plus a splitter that REFUSES to move any span
containing a pinned literal.

## Out of scope, and why relocation cannot fix it

Three specs remain over the 40 KB cap: `code-reviewer` 98.5 KB,
`experimenter` 65.6, `experiment-implementer` 64.0. Measured pin-free mass at
H3 grain is **7,937 / 3,222 / 1,884 B** — relocating EVERY movable byte still
leaves them at 89,761 / 60,351 / 63,343 B. All three already went through
this relocation in the 2026-08-05/06 compaction (code-reviewer 139 → 98.5 KB
into its section reference); the residue is gate rubric — step headings,
blocker tags and verdict contracts that tests pin by name. Getting them under
cap means changing WHAT THE GATES REQUIRE (merging or retiring checks), which
is a review-strength judgment call, not a relocation task.
