---
title: Amendment verification checks the EDITS, not the SITE SET — add an old-value
  residual grep
kind: infra
tags:
- workflow-fix
created_at: '2026-08-20T04:49:01Z'
has_clean_result: false
origin_prompt: 'Surfaced by the critic Methodology lens during #823 plan-v11 amendment
  review: a targeted-amendment check verifying ''all N anchors applied byte-exact''
  verifies the EDITS, not site-set completeness — amendments changing a registered
  value should also run an old-value residual grep over the whole new plan version.'
workflow: v1
---
# Amendment verification checks the EDITS, not the SITE SET — add an old-value residual grep

## Goal

When a plan amendment changes a **registered value** (a gate threshold, a
generation cap, a derived arithmetic figure), the `/adversarial-planner`
amendment procedure verifies that each targeted anchor/replace edit applied
byte-exact. That verification is sound but **answers the wrong question**: it
proves the N edits landed, not that N was the complete set of sites carrying
the old value.

Add a **post-amendment old-value residual grep** over the whole new plan
version, so an amendment that changes a registered value fails loud when any
site still asserts the superseded one.

## Motivating incident (#823, plan v10 → v11, 2026-08-19)

A 27-edit amendment changed the generation cap `max_tokens 1024 → 4096`
(plus a `8192` regen round) across ten plan sections. Pre-application: all 27
anchors verified byte-exact and present exactly once in v10. Post-application:
all 27 replacement texts present exactly once, all 27 original anchors absent.
Both pre-persist gates (goal-currency, non-empty-diff) ran. `verify_plan
--issue 823` returned PASS, `n_fail=0`, `n_warn=1` (a WARN v10 already carried).

Every mechanical check was green — and v11 line 332 still read
`temp 1.0, max_tokens 1024, no system prompt in the SCORED context`, the §4.1
completion-provenance parenthetical. It was the **28th site**, and the only
surviving `max_tokens 1024` in the plan.

All **four** independent reviewers (Claude Methodology lens, Claude
Statistics lens, Claude Alternatives lens, consistency-checker) found it
independently — one as a BLOCK, one as its single must-fix. Four reviewers
spending a round on a defect a one-line grep would have caught pre-persist is
the cost this task removes.

## Why the existing checks structurally cannot catch it

- **Anchor verification** is scoped to the anchors the amendment author
  enumerated. A site the author never noticed is outside the scope of every
  assertion by construction. The check is self-referential: it confirms the
  edit list against itself.
- **`verify_plan.py`** has no notion of "this plan version supersedes a value
  in its predecessor" — it validates a single version's internal structure.
- **Non-empty-diff** only proves the file changed.

The missing predicate is *completeness of the site set*, which needs the OLD
value as an input. Nothing currently carries it forward.

## Proposed fix

At amendment time, the author already knows each `(old_value, new_value)`
pair. Thread those pairs into a post-application check:

1. For each registered value the amendment changes, grep the FULL new plan
   version for the old value's distinctive form (`max_tokens 1024`,
   `> 50 new invalid rows`, `n_train ≈ 3992`).
2. Any hit is either (a) a missed site — fix it, or (b) a deliberate
   historical reference ("v10 expected ~12 drops; v11 measures 317") which
   the author dispositions explicitly, one line per hit.
3. Report the greps + dispositions alongside the existing anchor
   verification.

This is the same auditable-N/A shape used elsewhere in the workflow (the
symbol-rename whole-tree grep duty in `.claude/rules/crash-fix-rounds.md`
is the closest sibling — and is the direct precedent: it exists because
renaming a symbol across a diff verifies the rename, not that every
reference was found).

## Candidate homes (implementer picks; not pre-decided)

- **`.claude/skills/adversarial-planner/SKILL.md`** — the amendment/revision
  procedure, next to the existing edit-success gate. Prose duty, matching how
  the edit-success and goal-currency gates are already specified.
- **`scripts/verify_plan.py` amendment mode** — a mechanical check taking
  `--superseded-values` (or diffing vs `v{K-1}` and inferring changed numeric
  literals). Stronger, but inferring which literals are "registered values"
  vs incidental prose numbers is the hard part; an explicit flag avoids it.
- **`scripts/plan_patch.py`** — it already owns anchor application and could
  accept `--assert-absent <text>` per edit, failing loud on any residual.
  Narrowest change, and it is already the committed helper every amendment
  routes through.

A prose duty plus a `plan_patch.py --assert-absent` flag is likely the
cheapest combination that actually binds; the implementer should evaluate
whether `verify_plan` needs to be involved at all.

## Acceptance criteria

1. An amendment changing a registered value cannot persist with an
   unexplained residual old-value site.
2. Deliberate historical references remain expressible (explicit
   disposition), so the check does not become a false-positive generator on
   plans that legitimately quote their own prior expectations.
3. A pin (test or lint) so the duty cannot silently regress.
4. Reproduce the #823 shape as a fixture: 27 applied edits + 1 residual site
   ⇒ the check FAILs.

## Provenance

Surfaced as prose by the `critic` Methodology lens during the #823 plan-v11
amendment review round, and independently corroborated by three other
reviewers on the same round. Filed per
`.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups get the
same auto-file + spawn as a formal `workflow-fix-candidate` block).

Agent memory already written: `.claude/agent-memory/critic/feedback_amendment_old_value_residual_grep.md`.
