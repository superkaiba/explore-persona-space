---
title: Implement `## Why this experiment` adversarial interrogation gate at task creation,
  PM dispatch, and /issue Step 0
kind: infra
tags:
- workflow
- gate
created_at: '2026-05-20T19:53:36Z'
has_clean_result: false
legacy_why_unset: true
---
# Implement `## Why this experiment` adversarial interrogation gate at task creation, PM dispatch, and `/issue` Step 0

## Goal

Force the user to articulate four concrete answers (Decision / Branches / Cut / Application) before any experiment can advance from `proposed → planning`. The user is prone to running experiments without a concrete reason; this gate is friction designed to interrupt that impulse.

**Critical design constraint:** the user MUST personally articulate the four answers in chat. The agent's role is to interrogate and transcribe, NEVER to draft, synthesize from context, or auto-extract from prior materials. If the user delegates ("you decide", "use the ideation doc"), the agent refuses and re-asks.

This task itself is bootstrap — its `## Why this experiment` section is intentionally absent because the gate doesn't exist yet. User explicitly waived the dogfood test.

**Application:** infra — serves Audit + Predict (forces stating-the-decision before compute commits).

## The four questions (asked one at a time)

1. **Decision this changes**: What concrete choice in your queue or proposal hinges on this outcome?
2. **Expected outcome + branches**: What do you expect, and what alternative outcome would route you to a different next experiment?
3. **What gets cut**: Which experiment in your queue doesn't run this week as a result?
4. **Application**: detect | predict | defend | audit | infra — one sentence on which deliverable this advances.

## Four-layer gate dynamic (per question)

1. **Ask** the question.
2. **Refuse** non-answers (pattern check; see reject list below).
3. **Challenge** the substance (research-taste critique) — ONE round per question, max.
4. **Defer + transcribe** when the user defends, reframes, or overrides.

User always has the last word via override ("I'm right, move on" / "defer" / "ship it"). Override is unconditional and logged.

## Refuse list (non-answer patterns)

Reject answers containing only these phrases, or answers under ~15 words that don't name a specific decision/branch/cut:
- "just curious" / "want to see what happens" / "to explore" / "to check" / "to understand X better"
- "use the ideation doc" / "you decide" / "look at the parent" / "summarize from context"
- "because it would be interesting" / "to add to the literature"

Implementation: regex pattern list + LLM substance check ("does this answer name a concrete X?").

## Challenge lenses (one per question, only when applicable)

| Question | Lens fires when… | Challenge shape |
|---|---|---|
| Decision | Decision already determined by existing result, or reachable on cheaper evidence | "X already constrains this — does this really update beyond what you know?" |
| Branches | Both branches lead to the same next move (non-decisive) | "Both branches seem to lead to the same next experiment. What actually changes?" |
| Cut | Named cut isn't credible (already deprioritized, or you'll run both anyway) | "You're already deprioritizing that — is this a cut or a free addition?" |
| Application | Anchor reads as post-hoc; deliverable serves a different application | "This looks more like X than Y — does the deliverable actually feed Y's metric?" |

If the agent doesn't see a substantive issue, it skips the challenge entirely. No gratuitous challenges.

## Termination rules

Three accepted user replies to a challenge:
- **Defense**: original answer + brief defense → transcribe both, move on.
- **Reframe**: user changes their answer → transcribe new answer, no further challenge.
- **Override**: "I'm right" / "defer" / "ship it" → transcribe original + `user_overrode_challenge: true` marker, move on.

## Three enforcement points

### 1. PM session — dispatch gate (primary)

When user says "work on #N" or "spawn #N" in the PM session:
- Read `body.md`; if `## Why this experiment` is missing or stubby (any of 4 lines empty / under threshold), block dispatch.
- Run the four-question interrogation in chat.
- After all four are filled (with any overrides logged), call `task.py set-body <N> --file <draft.md>` and post the `epm:gate-filled` events.jsonl marker.
- THEN spawn the per-issue session via `spawn_session.py spawn-issue --issue N`.

Update `.claude/agents/research-pm.md` Mode 5 (DISPATCH) to include this pre-check.

### 2. `/issue <N>` Step 0 — safety net

If PM is bypassed (user opens a per-issue session directly), Step 0's existing clarifier check extends:
- If `## Why this experiment` missing or stubby → run the four-question interrogation in-session.
- Legitimate `AskUserQuestion` use because it IS a gate.

Update `.claude/skills/issue/SKILL.md` Step 0 (body completeness check).

### 3. `task.py new --kind {experiment|survey|infra}` — creation gate

If `--body-file` doesn't contain `## Why this experiment`, refuse with:
```
ERROR: experiment/survey/infra tasks require `## Why this experiment` (4 lines).
This section must be filled by an interrogating agent, not written from scratch.
Use the PM session, `/ideation`, or `/experiment-proposer` to draft it.
```

Note: ideation/proposer outputs already produce four-line candidates; promoted candidates pass automatically. The CLI guard catches "I'll just type it manually" bypass attempts.

## `verify_task_body.py` changes (new check #12)

- Frontmatter has `application: <enum>` field (one of: detect, predict, defend, audit, infra).
- Body has `## Why this experiment` H2 section with four labeled lines: `**Application:**`, `**Decision this changes:**`, `**Expected outcome + branches:**`, `**What gets cut if we run this:**`.
- Each line non-empty, ≥40 chars after the label.
- Application enum value matches frontmatter `application:`.
- FAIL if missing, stubby, or mismatched.
- Skipped when frontmatter contains `legacy_why_unset: true`.

## Logging

Every gate fill writes one events.jsonl row:
```json
{
  "kind": "epm:gate-filled",
  "version": 1,
  "gate": "why-experiment",
  "filled_by": "<agent_session_id>",
  "challenges_fired": ["decision", "branches"],
  "user_overrides": []
}
```

After ~30 fills the user samples whether challenges are firing on substantive issues or going soft. If soft, sharpen the lens prompts.

## Migration

One-shot migration script `scripts/migrate_add_legacy_why_sentinel.py`:
- 130 `completed` + 60 `archived` + 33 `awaiting_promotion` → add `legacy_why_unset: true` to frontmatter.
- 55 `proposed` → leave WITHOUT the sentinel. Next time the user wants to run any of them, the PM gate fires.
- 2 `running` + 1 `blocked` → add the sentinel (already past the gate point); user can fill retroactively at their option.
- Migration commit is atomic; rollback by removing the field.

## Files touched

- `scripts/task.py` — `new` subcommand rejects experiment/survey/infra without `## Why this experiment`.
- `scripts/verify_task_body.py` — new check #12 (application enum + section + line non-empty + frontmatter match).
- `scripts/migrate_add_legacy_why_sentinel.py` — new one-shot migration.
- `.claude/agents/research-pm.md` — Mode 5 (DISPATCH) extended with pre-spawn gate.
- `.claude/skills/issue/SKILL.md` — Step 0 extended with gate check.
- `.claude/skills/why-experiment-gate/SKILL.md` — NEW skill, the interrogation protocol with tight system prompt (interrogator only, no drafting, refuses non-answers, one challenge max per question, user override terminates).
- `.claude/workflow.yaml` — add `why-experiment` to the canonical gate list under § gates.
- Body template under `tasks/` docs — update with the four-line section.

## Test plan

1. **Unit:** `verify_task_body.py` correctly FAILs on missing section, stubby lines, missing frontmatter field, mismatched application enum; PASSes on well-formed body; SKIPs on legacy sentinel.
2. **CLI:** `task.py new --kind experiment --body-file empty.md` rejects with clear error; `task.py new --kind experiment --body-file with_why.md` succeeds.
3. **Migration:** dry-run prints diff; apply commits one atomic change; sentinel correctly added to 226 tasks, absent on 55 proposed.
4. **Manual end-to-end:** Run the gate on a chosen proposed task (suggested test case: #114 "Use activation oracles to see persona" — small enough that the gate cost is felt). Confirm:
   - Interrogator asks Q1, refuses "just curious", accepts substantive answer.
   - Challenge fires when the named decision is weak.
   - User override ("ship it") terminates the loop, marker logged.
   - Section lands in body.md, frontmatter application matches, verifier PASSes.
5. **Per-issue safety net:** spawn per-issue session for a freshly-gated task; confirm `/issue <N>` Step 0 sees a complete section and proceeds without re-asking.

## Acceptance criteria

- Gate can be entered from PM session OR `/issue` Step 0 OR `task.py new` (refusal path).
- Agent never drafts content from context; user words transcribed verbatim.
- One challenge per question, never more.
- User override is unconditional and logged.
- Verifier check #12 enforces the floor; conversation enforces the substance.
- Migration adds sentinel to non-proposed tasks; 55 proposed tasks remain ungated until first-use.

## Explicit cuts (not bundled into this task)

- Auto-extract from `/ideation` outputs (anti-pattern: defeats the friction).
- Multi-application secondary field (anti-pattern: escape hatch — if you need two, split the task).
- Budget-cap removal (separate workflow change, queued next).
- Auto-continuation hardening (separate workflow change, queued after budget removal).
- Per-experiment dashboard (separate workflow change, scope still TBD).
