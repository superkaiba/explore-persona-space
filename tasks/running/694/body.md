---
title: 'workflow: add results-driven literature-positioning step (post-results related-work
  sweep)'
kind: infra
tags:
- wf-fix
created_at: '2026-06-28T00:50:58Z'
has_clean_result: false
origin_prompt: 'No agent does a fresh, results-driven literature sweep ("we found
  X — who else has reported X, and does it agree?"). Post-hoc related-work positioning
  rests on (a) the two pinned sibling papers, (b) docs/papers.md, (c) the author remembering.
  The arXiv MCP machinery is wired but only invoked at plan time. File a kind: infra
  task to add a results-positioning literature step (new lens in interpretation-critic,
  or a dedicated post-analysis agent that arXiv-searches the finding and appends a
  Related findings note to the clean-result ## Goal). Run in the background with happy
  coder.'
---
## Overview / Motivation

The workflow does a thorough literature review **before** experiments
(planner §11 hyperparameter grounding via the arXiv MCP; the CLAUDE.md
"before any new research project / direction" rule; the clarifier) — but
that review is keyed on the **question and the training/eval recipe**,
not on what the experiment actually **found**.

After results land, **no agent runs a fresh, results-driven literature
sweep** — i.e. "we measured X; who else has reported X, and does their
result agree, contradict, or extend ours?". The post-hoc related-work
positioning currently rests on exactly three weak legs:

1. the two hard-coded spiritual-sibling papers (Persona Vectors,
   arXiv 2507.21509; Persona Features Control EM, arXiv 2506.19823);
2. whatever already lives in `docs/papers.md`;
3. the author remembering.

The arXiv MCP machinery to do a findings-driven search is already wired
(`.claude/rules/arxiv-mcp.md`: `mcp__arxiv__search_papers`,
`mcp__arxiv__semantic_search`, `mcp__arxiv__read_paper`, plus
`arxiv-latex`) — it is simply never invoked at results time.

## Goal

Add a **results-driven literature-positioning step** to the post-results
`/issue` workflow: after the analyzer produces the interpretation /
clean-result, an agent searches the literature (arXiv MCP + web) keyed on
the experiment's **measured findings** and produces a short, verified
"Related findings" note positioning our result against existing published
work (replicates / contradicts / extends / no prior report found).

## Workflow gap

- **Bug observed:** there is no step that searches the literature for work
  matching our *results*; related-work positioning is front-loaded
  (question/recipe-driven at plan time) and never repeated against the
  actual finding.
- **Why it is a workflow gap:** the capability (arXiv MCP) is wired and
  every clean-result has a `## Goal` → `**Broader narrative:**` slot that
  is the natural home for results-positioning, but nothing populates it
  from a fresh findings-keyed search — so the project's positioning is
  systematically blind to any prior work outside the two pinned papers.
- **Confidence (emitter):** medium (the gap is real and confirmed; the
  best *implementation shape* is a genuine design call — see below).

## Design options (planner picks; do NOT pre-commit in this body)

The planner + adversarial-critic ensemble should weigh at least these and
choose, grounding the choice:

1. **New dedicated post-analysis agent** (e.g. `related-work-finder`):
   fresh-context agent spawned at the Step 10 post-completion batch
   (alongside `follow-up-proposer` / `living-docs-updater`), reads the
   landed clean-result findings, runs a bounded arXiv/web search, and
   **proposes** (never auto-applies) an addition to the clean-result
   `## Goal` → `**Broader narrative:**` and/or a `docs/papers.md` update.
   NOTE: adding a brand-new agent is the more **architectural** option
   (per `.claude/rules/workflow-fix-on-bug.md` § Architectural
   greenlight) — if chosen, flag `architectural: true` and park at
   `plan_pending` for the user; the user has pre-authorized running this
   task but the agent-vs-lens shape is theirs to confirm.
2. **New lens in `interpretation-critic`** (and its codex twin): a
   "results-positioning" lens that requires a findings-keyed literature
   check and flags REVISE if the writeup makes a novelty/agreement claim
   with no literature search behind it. Lighter-weight, non-architectural,
   auto-mergeable — but couples a *generative* search to a *critic* role,
   which may be a mis-cast per `.claude/rules/agents-vs-skills.md`.
3. **Extend `living-docs-updater`**: it already touches `docs/papers.md`
   when a result "replicates / contradicts / extends a sibling-paper
   finding" — but today it only reads the *existing* `docs/papers.md`. Add
   a fresh arXiv search so it can discover *new* related papers, not just
   reconcile against ones already logged.

A hybrid is fine (e.g. a generator agent + a critic lens that checks it
ran). Decide based on agents-vs-skills casting and the cost/benefit.

## Constraints / invariants

- **Findings-driven, not question-driven** — this is the distinguishing
  property vs the planner's front-loaded review. Key the search on the
  actual measured result, not the Goal or the recipe.
- **Verified citations only — zero fabrication.** Every cited paper must
  be a real, resolvable arXiv id / DOI confirmed via the MCP in the same
  turn it is written (mirror the analyzer's verbatim-fidelity rule and
  `feedback_never_fabricate_sha` discipline). No invented titles, no
  hallucinated "as shown in [X]". If no prior report is found, say so
  explicitly — "no prior report located" is a valid, useful output.
- **Propose, never auto-apply** to living docs — same contract as
  `living-docs-updater` (writes a diff/note for the user gate, never edits
  live docs, never runs git). The clean-result `## Goal` addition routes
  through the analyzer/clean-result-critic loop like any other body edit.
- **Advisory, not a hard promotion gate** — a thin or empty literature
  result must not block `awaiting_promotion`. (Planner may argue for a
  soft WARN; do not make it a blocking gate without explicit rationale.)
- **Integrate with the v4 clean-result spec** — the natural home is
  `## Goal` → `**Broader narrative:**` (SPEC.md §§ 390-475). Must not
  break `verify_task_body.py` / `audit_clean_results_body_discipline.py`
  caps (Broader narrative is part of the prose budget). If a new
  subsection is proposed, update SPEC.md + the verifier together
  (SPEC.md is source of truth — read it first).
- **Bounded cost** — 0 GPU-h; cap the arXiv/web search breadth so it is a
  few MCP calls, not an unbounded crawl. State the cap.
- **Workflow-surface only** — `.claude/agents/*.md`,
  `.claude/skills/issue/SKILL.md` (+ `markers.md` if a marker is added),
  `.claude/rules/*.md`, `.claude/workflow.yaml`, `CLAUDE.md`, and the
  clean-result SPEC + verifier if the body shape changes. Never experiment
  code, `configs/`, or `tasks/`.
- If a new agent is added, also update the ontology table in
  `.claude/rules/agents-vs-skills.md` and the `/issue` composition diagram.

## Scope / surfaces

- Likely primary targets: a new `.claude/agents/related-work-finder.md`
  OR `.claude/agents/interpretation-critic.md` (+ `codex-interpretation-critic.md`)
  OR `.claude/agents/living-docs-updater.md`; `.claude/skills/issue/SKILL.md`
  (wire the spawn into Step 9/10); `.claude/skills/clean-results/SPEC.md`
  + `scripts/verify_task_body.py` if the body shape changes;
  `.claude/workflow.yaml` (marker, if any); `CLAUDE.md` (document the step).
- `grep -rn 'Broader narrative' .claude/` and the post-results Step 9/10
  flow in SKILL.md before editing, to find every integration point.

## Acceptance criteria

1. After an experiment's results land, the workflow runs a findings-keyed
   literature search and surfaces a "Related findings" positioning result
   (replicates / contradicts / extends / none-found) for the user.
2. The positioning lands in (or proposes a diff to) the clean-result
   `## Goal` → `**Broader narrative:**` and/or `docs/papers.md`, gated for
   user review where a living-doc edit is involved.
3. Citations are MCP-verified; the "none found" path is explicit.
4. The step is non-blocking for promotion and 0 GPU-h.
5. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
   passes; CLAUDE.md / workflow.yaml / the rule file stay consistent; if
   the v4 body shape changes, SPEC.md + verifier + critic lenses move
   together.
