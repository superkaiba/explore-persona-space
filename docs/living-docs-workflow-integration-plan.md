---
title: "Living-docs ⇄ /issue workflow integration (plan)"
summary: "How experiment completion keeps open_questions.md, papers.md, and claims.yaml consistently updated — hook points, components, consistency guarantees, phased rollout."
status: proposal
last_updated: 2026-05-28
nav_order: 50
---

# Living-docs ⇄ /issue workflow integration (plan)

**Goal.** Keep the living research docs consistently updated as experiments complete — automatically, at the right lifecycle moment, with consistency guarantees — so the hub (`open_questions.md`) never silently goes stale between manual re-syntheses.

## Where things stand

- **`/issue` flow** ends: analyzer promotes the clean-result in place → interpretation-critic → clean-result-critic → (test-verdict) → auto-complete → follow-up-proposer. Experiments park at `awaiting_promotion`; the user runs `task.py promote <N> useful|not-useful` → `completed`.
- **`open_questions.md`** updates by *periodic manual re-synthesis* (last 2026-05-27). It drifts stale between syntheses.
- **`papers.md` / `conditional-behavior-related-work.md`** are manual. The dashboard literature firehose (`updates/literature/`) refreshes nightly via the existing cron.
- **There is no link** from "an experiment completed" to "the living doc that should move."

## Design principles

1. **Hook at the terminal, deliberate event:** task → `completed` (post-promotion for experiments). That's the single well-defined per-experiment moment, and promotion is a deliberate user act.
2. **Single writer per doc**, atomic git commit (same `flock` + one-commit discipline as `task.py`) → no concurrent corruption.
3. **Explicit experiment→question mapping** set at creation (`relates_to`), with topic-inference fallback.
4. **Non-blocking:** updating the living docs never gates experiment completion.
5. **Backstop re-synthesis** catches whatever the per-result updater misses, and re-ranks.
6. **A consistency linter** makes drift visible instead of silent.

## Structural prerequisite (one-time)

Give each open question a machine-targetable living-state trailer + a stable anchor, so the updater has a precise edit target:

```markdown
**A1. What predicts marker implantability if cosine / JS / Mantel all fail?** <!-- q:A1 -->
... existing "why open" / "source" prose ...
> **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380, #340, #368
```

- `<!-- q:A1 -->` — stable id that survives prose edits (the updater greps for it).
- `> **State:**` — the one line the updater rewrites: maturity (🌱/🌿/🌳) · confidence (LOW/MOD/HIGH, same scale as clean-results) · last-updated · evidence task list.

And a task-schema field, set at creation by the Goal gate (`/issue` Step 0c) or the planner:

```yaml
relates_to: [A1, D2]   # open-question keys this experiment bears on
```

## Components to build

1. **`scripts/living_docs.py`** — the mechanical core (importable + CLI):
   - `link_result(task_id)`: read the task's clean-result (title, confidence, classification, `relates_to`), locate each `<!-- q:Kn -->` block in `open_questions.md`, update its `State:` trailer (bump date, append `#N` to evidence), prepend a dated changelog line. Atomic `flock` + single commit. Mechanical only.
   - `check()`: lint — every `completed` experiment with `has_clean_result` appears in some question's evidence; every evidence `#N` exists; flag questions stale relative to new results. Exit nonzero on drift. Runs in `/weekly` + optionally pre-commit.
2. **`.claude/agents/living-docs-updater.md`** — the semantic layer (fresh-context agent):
   - Input: task `N` + its clean-result + the relevant question block(s).
   - Rewrites the belief sentence (1-3 sentences) to reflect the result, sets confidence/maturity, calls `living_docs.py link_result` for bookkeeping, and — if the clean-result cites papers absent from `papers.md` — appends them (or flags). One commit. Bounded, single-turn.
3. **`/issue` SKILL.md hook** (new Step after auto-complete → `completed`):
   - Orchestrator spawns `living-docs-updater` in the background; posts `epm:living-docs-updated v1` with the diff. Failure → `epm:living-docs-update-failed v1` note; never blocks completion.
4. **Creation-time wiring:** Goal gate / planner sets `relates_to`. Absent → updater infers from topic, best-effort, no prompt.
5. **Backstop re-synthesis:** extend `/weekly` to (a) re-synthesize `open_questions.md` from all clean-results (today's manual process, automated) and (b) run `living_docs.py check`. Monthly: re-run the lit sweep to refresh `conditional-behavior-related-work.md` + `papers.md`.
6. **Dashboard:** the `/docs` route (Phase 0, done) renders the live state; `last_updated` / `status` show freshness. Optional homepage widget: "living docs · updated X · M open questions."

## Marker schema (`.claude/workflow.yaml`)

- `epm:living-docs-updated v1` — posted after the updater commits; payload = unified diff + which questions were touched.
- `epm:living-docs-update-failed v1` — non-fatal; payload = reason.

## Consistency guarantees

- Single writer + atomic commit → no races.
- `relates_to` makes the experiment→question mapping explicit and checkable.
- `living_docs.py check` in `/weekly` (+ optional pre-commit) surfaces drift fast.
- Backstop re-synthesis → coherence + re-ranking + catches misses.

## Rollout (each phase independently useful)

- **Phase 0 (done):** `/docs` route renders the living docs in the dashboard.
- **Phase 1:** add `State:` trailers + `<!-- q:Kn -->` anchors to `open_questions.md`; add `relates_to` to the task schema. (structural)
- **Phase 2:** build `scripts/living_docs.py` (`link_result` + `check`) + tests.
- **Phase 3:** add the `living-docs-updater` agent + `/issue` hook + markers.
- **Phase 4:** backstop re-synthesis in `/weekly` + monthly lit refresh.

## Open choices

- **A. Update trigger:** on `promote → completed` only (deliberate), or also on clean-result *creation* (earlier, pre-promotion)? → recommend `completed` only.
- **B. Confidence movement:** auto-rule (e.g. 2+ consistent results → bump) vs agent-judged. → recommend agent-judged; the script only does bookkeeping.
- **C. New questions:** may the updater *add* a question when a result opens one, or only update existing ones? → recommend propose-only (flag in changelog), never auto-insert.
