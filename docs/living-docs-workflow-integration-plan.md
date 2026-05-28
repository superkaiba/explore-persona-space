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
3. **Explicit experiment→question mapping** set at creation: every `kind: experiment` links to one or more open questions via a flat `relates_to` (no primary/secondary), by matching an existing question or creating a new one.
4. **Every living-docs mutation is user-confirmed.** Both the creation-time link (match-or-create) and the completion-time update are *proposed* by the agent and applied only on your ok. The agent may propose accretive updates AND broader edits (reword / split / mark settled / add a question / touch several at once / fix `papers.md`) — but never without confirmation.
5. **Non-blocking on the experiment lifecycle.** The experiment completes regardless; the proposed living-docs diff parks for your confirmation (like `awaiting_promotion`), so you needn't be present when a run finishes.
6. **Backstop re-synthesis** catches whatever the per-result updater misses, and re-ranks.
7. **A consistency linter** makes drift visible instead of silent.

## Structural prerequisite (one-time)

Give each open question a machine-targetable living-state trailer + a stable anchor, so the updater has a precise edit target:

```markdown
**A1. What predicts marker implantability if cosine / JS / Mantel all fail?** <!-- q:A1 -->
... existing "why open" / "source" prose ...
> **State:** 🌿 budding · MODERATE · updated 2026-05-28 · evidence: #207, #380, #340, #368
```

- `<!-- q:A1 -->` — stable id that survives prose edits (the updater greps for it).
- `> **State:**` — the one line the updater rewrites: maturity (🌱/🌿/🌳) · confidence (LOW/MOD/HIGH, same scale as clean-results) · last-updated · evidence task list.

And a task-schema field, set at the Goal gate (`/issue` Step 0c) — a flat list, no primary/secondary:

```yaml
relates_to: [a1, d2]   # stable open-question ids this experiment bears on
```

At creation the clarifier/planner proposes the link — **match** an existing question or **draft a new one** — and you confirm it in the same Goal gate you already approve. No silent linking; no topic-inference fallback that skips you.

## Components to build

1. **`scripts/living_docs.py`** — the mechanical core (importable + CLI), applies only **confirmed** changes:
   - `apply(task_id, patch)`: apply a confirmed patch to `open_questions.md` (and `papers.md` if touched) — atomic `flock` + single commit, prepend a dated changelog line. The patch may be a simple evidence/`State` bump or a broader multi-question edit; the script just applies + commits what was confirmed. No judgement.
   - `link(task_id, [q-ids])`: write `relates_to` on the task + add the task to each question's evidence list (the confirmed creation-time link); creates a new question stub first if one was approved.
   - `check()`: lint — `relates_to` ⇄ question-evidence agree both ways; every `completed` experiment with `has_clean_result` appears in some question's evidence; every evidence `#N` exists; flag questions stale relative to new results. Exit nonzero on drift. Runs in `/weekly` + optionally pre-commit.
2. **`.claude/agents/living-docs-updater.md`** — the semantic layer (fresh-context agent), **proposes, never applies**:
   - Input: task `N` + its clean-result + the linked question block(s) + the rest of `open_questions.md` (so it can spot a needed reword / split / merge / new question).
   - Produces a **proposed unified diff** + a short rationale: the accretive update (append the result to evidence, shift the belief sentence + confidence/maturity, record "inconclusive" if it didn't move) PLUS any broader edits it judges warranted, and any `papers.md` additions.
   - Returns the proposal to the orchestrator; does **not** commit. Applied only after you confirm (then `living_docs.py apply`). Bounded, single-turn.
3. **`/issue` SKILL.md hook** (new Step after auto-complete → `completed`):
   - Orchestrator spawns `living-docs-updater`; it posts `epm:living-docs-proposed v1` with the proposed diff + rationale. The orchestrator presents the diff for confirmation (new conditional gate `living_docs_update`, registered in `workflow.yaml` like `plan_approval`). On **confirm** (or confirm-with-edits) → `living_docs.py apply` + `epm:living-docs-updated v1`. On **reject** → skip + `epm:living-docs-update-rejected v1`. The proposal parks if you're away; the experiment is already `completed`, so nothing is blocked.
4. **Creation-time wiring:** at the Goal gate the clarifier/planner proposes match-or-create for the question link; on your confirm → `living_docs.py link` writes `relates_to` + the question's evidence entry (creating the question first if you approved a new one). No link is written without confirmation.
5. **Backstop re-synthesis:** extend `/weekly` to (a) re-synthesize `open_questions.md` from all clean-results (today's manual process, automated) and (b) run `living_docs.py check`. Monthly: re-run the lit sweep to refresh `conditional-behavior-related-work.md` + `papers.md`.
6. **Dashboard:** the `/docs` route (Phase 0, done) renders the live state; `last_updated` / `status` show freshness. Optional homepage widget: "living docs · updated X · M open questions."

## Marker schema (`.claude/workflow.yaml`)

- `epm:question-linked v1` — creation-time: which question(s) the issue was linked to (and whether one was newly created).
- `epm:living-docs-proposed v1` — the updater's proposed diff + rationale, awaiting confirmation.
- `epm:living-docs-updated v1` — posted after you confirm and `living_docs.py apply` commits; payload = the applied diff + which questions were touched.
- `epm:living-docs-update-rejected v1` — you declined the proposal; non-fatal, payload = reason.

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

## Decisions (locked 2026-05-28)

- **Trigger:** completion (`promote → completed`) for the result update; **creation** (Goal gate) for the question link.
- **`relates_to`:** flat list, no primary/secondary.
- **Scope of edits:** the updater may make accretive updates AND broader edits (reword / split / settle / add questions / multi-question / `papers.md`) — whatever the result warrants.
- **Confirmation:** every living-docs mutation (creation link + completion update) is user-confirmed; the agent proposes, you approve / edit / reject. Nothing auto-applies.
- **Confidence:** agent-proposed (then confirmed); the script only applies what's confirmed.
