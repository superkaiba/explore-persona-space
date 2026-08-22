---
title: 'SKILL.md Step 4: brief-composition rule covers marker SUPPRESSION but not
  DISPLACEMENT by a return-format contract'
kind: infra
tags: []
created_at: '2026-08-12T18:28:07Z'
has_clean_result: false
origin_prompt: 'Surfaced by /issue 1336 round v20: implementer brief ended in a detailed
  ''Report back:'' list, never named epm:experiment-implementation; implementer returned
  everything as Agent text and posted no marker; code-review FAILed mechanical-contract-only
  with both blockers being the absent marker, so the Step 5c-bis strip could not apply.'
workflow: v1
---
## Goal

Extend the existing SKILL.md Step 4 brief-composition rule "**A brief NEVER suppresses the implementation marker**" (`.claude/skills/issue/SKILL.md` ~:2274) to cover the DISPLACEMENT variant, not only explicit suppression: an orchestrator brief that specifies an elaborate return contract ("**Report back:** the commit SHA, the diffstat, which tests you ran, …") without NAMING the `epm:experiment-implementation` marker duty crowds the marker out just as effectively as instructing the implementer to skip it, and manufactures the identical `marker-shape` code-review blocker plus an extra fix round.

## Observed (task #1336, 2026-08-12, implementation round v20)

The orchestrator spawned `experiment-implementer` with a detailed brief that ended in an explicit "**Report back:** …" list enumerating the commit SHA, per-file diffstat, test results, smoke verdict, and unimplementable-spec disclosures. The brief never mentioned `epm:experiment-implementation`.

The implementer did excellent work — commit `a0cda5cdb4a2a0350af75d85a0e59fd8b0873a88`, +1056/−139, all three mandated mechanism changes, all 11 plan-named tests written and passing, smoke rc=0 — and returned all of the requested information as its final Agent text. It posted NO `epm:experiment-implementation` marker.

Consequence: `code-reviewer` returned **FAIL** with `mechanical_contract_only: true` and **zero substantive blockers** — both blockers were the absent round-v20 marker (`mc1-impl-marker-absent-round-v20`, `mc2-smoke-run-evidence-not-in-marker`; the reviewer probed the canonical channel 4× and checked the orphaned/deferred marker channels). The Step 5c-bis mechanical-contract strip could not be applied, because that strip is conditional on the implementer marker being *present and conforming* — and here the missing marker IS the blocker. One extra implementer round was needed purely to land the durable record of work already done.

## Why the existing coverage does not close it

Three nearby passages each cover part of it:

- **`.claude/skills/issue/SKILL.md` ~:2274-2279** — "**A brief NEVER suppresses the implementation marker.** Never instruct an implementer to 'post nothing' / skip its `epm:experiment-implementation` / `epm:results` marker — the code-review ensemble's mechanical contract KEYS on that four-section marker, so suppressing it manufactures a `marker-shape` blocker and an extra fix round (#1900)." This is scoped to an *instruction* to skip. A brief that is merely SILENT on the marker while loudly specifying a different return channel is not an instruction to skip, so the rule does not bite — yet the stated consequence ("manufactures a `marker-shape` blocker and an extra fix round") is reproduced exactly.
- **`.claude/skills/issue/SKILL.md` ~:3581** — "spawn `experiment-implementer` with a brief naming the marker". This is the positive duty, but it is a clause inside a spawn-step sentence rather than a checkable brief-composition rule in the Step 4 bullet list where the suppression rule lives, and it carries no rationale a brief author would weigh against a competing return-format instruction.
- **`.claude/agents/experiment-implementer.md`** mandates the marker clearly and repeatedly (~:686 "**Post the report** as `<!-- epm:experiment-implementation v<n> -->`", ~:788, ~:799, the template at ~:811-897). The agent spec is not the gap. The gap is that a sufficiently specific orchestrator brief can functionally override a spec step without contradicting it — the brief is the proximate instruction the agent optimizes for.

## Proposed fix (small, prose-only)

1. Extend the ~:2274 bullet with the displacement clause: a brief that specifies a return format NAMES the marker duty alongside it — e.g. "post your report as the `epm:experiment-implementation` marker at the next version (omit `--version`; the CLI derives max+1) AND return a short summary as your final text". State the mechanism in one clause: returned Agent text is NOT durable task state, so a return-only contract loses the record the code-review mechanical contract keys on.
2. Add the symmetric one-line reminder where the Step 4 brief is composed (the ~:3581 region), cross-referencing the extended bullet, so a brief author who reads only the spawn step still sees it.
3. No change to the strip logic, the marker schema, `task.py`, or any agent spec. The runtime behaviour is already correct; this is a brief-composition prevention fix.

## Acceptance criteria

- The ~:2274 bullet covers BOTH shapes (explicit suppression AND silent displacement by a competing return contract) and states why returned text is not a substitute.
- The Step 4 spawn-step region cross-references it.
- `uv run python scripts/workflow_lint.py` (no flags) is no worse than its pre-change baseline (~15 pre-existing errors on `main` unrelated to this change — do not chase them; assert only that no new failure names an edited file).

## Non-goals

- No change to the Step 5c-bis strip (its precondition — marker present and conforming — is correct as written).
- No new lint check. "Did the brief name the marker?" is not expressible from the surface files; the brief is composed at runtime.
- No change to `experiment-implementer.md` / `implementer.md`, which already mandate the marker.

## Provenance

Surfaced by the `/issue 1336` autonomous session at Step 5 of implementation round v20. Cost: one extra implementer round; no compute, no data loss. The orchestrator's own brief omission is the proximate cause — filed because the omission was *invited* by a rule that reads as fully covering the failure mode while covering only half of it.
