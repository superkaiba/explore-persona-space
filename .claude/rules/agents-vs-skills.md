# When to use an Agent vs a Skill

A distinction that kept getting muddled. Apply this rule when creating or
restructuring anything under `.claude/`.

---

## The rule

**Agent = a role with a fresh context.** Live in `.claude/agents/*.md`.
Spawned via the `Agent` tool. Own their own memory, tools, model, effort
level. Produce a bounded artifact and return.

**Skill = a playbook for the current context.** Live in `.claude/skills/<name>/SKILL.md`.
Invoked via the `Skill` tool or `/<name>`. Load instructions into whichever
agent invokes them (main or subagent). No isolation, no separate context.

A thing is ONE or the OTHER, never both.

---

## Use an Agent when ANY of these hold

- **Independence is load-bearing.** Example: `clean-result-critic` must not see the
  `analyzer`'s chain of thought, so they must be different context windows.
- **Persona / role encapsulation.** Example: `critic` is opinionated
  ("how could this plan fail?"); you want its voice separate from the main
  conversation.
- **Long-running / background work.** Example: `experimenter` launches a
  training run and monitors for hours — should not clog the main thread.
- **Fresh-context debugging or research.** Example: `retrospective` reviews
  a day's transcripts without the clutter of the current session.

## Use a Skill when ALL of these hold

- The task is a workflow or convention that any agent might follow.
  Example: `paper-plots` (chart-building protocol), `weekly` (parallel
  fan-out orchestrator), `clean-results` (manual consolidation steps).
- No fresh-context requirement — it's fine for the caller to see it all.
- The "knowledge" is reusable reference material, not a persona.

---

## Signals you've mis-cast

If an **agent** spec reads like `Step 1 → Step 2 → Step 3` with no
fresh-context justification, it's probably a skill invoked by the main agent.

If a **skill** is a long protocol with adversarial-review requirements or
a distinct persona, it's probably an agent.

If a file says "Mode A when invoked automatically / Mode B when invoked
manually" — one of those modes probably belongs in the *caller*, not in
the skill/agent itself. (This is what happened with `clean-results` Mode A
before the analyzer absorbed it.)

---

## Typical composition pattern

The outer layer is usually a **skill** (orchestrator). Inside, it dispatches
**agents** (specialists) and references other **skills** (reference patterns).

```
/issue  (skill: orchestrator)
    ├─ runs /adversarial-planner (skill: inner orchestrator)
    │       ├─ spawns planner   (agent)
    │       └─ spawns critic ensemble ∥ consistency-checker (agents, one
    │           Phase-2 spawn batch; findings unioned into ONE revise round)
    ├─ spawns experimenter (agent)
    │       └─ uses /experiment-runner (skill: monitoring protocol)
    ├─ spawns upload-verifier ∥ analyzer first pass (held) [∥ methodology-writer
    │   │    PAPER MODE, paper: true only]
    │   │   (results-landed parallel batch; epm:interpretation publishes
    │   │    only after upload-verification PASS; pod terminate only after PASS.
    │   │    v4 markdown: no early methodology-writer spawn — the methodology doc
    │   │    is a post-PASS mechanical export of the body's ## Methodology section.
    │   │    paper: true: methodology-writer authors Methods + Appendix INTO the
    │   │    .tex (analyzer splices them); v3/v2 markdown: legacy early spawn)
    ├─ iterates analyzer ↔ interpretation-critic    (max 5 rounds, content honesty;
    │   │                                            ALL rounds ENSEMBLED with
    │   │                                            codex-interpretation-critic as of 2026-06-12)
    │       ├─ spawns analyzer (agent, uses /paper-plots)
    │       └─ spawns interpretation-critic (agent) [+ codex twin every round]
    ├─ iterates analyzer ↔ clean-result-critic      (max 5 rounds, structure + register
    │   │                                            + statistical-framing rule;
    │   │                                            ALL rounds ENSEMBLED with
    │   │                                            codex-clean-result-critic as of
    │   │                                            2026-06-12; FINAL adversarial
    │   │                                            gate as of 2026-05-13)
    │       ├─ re-spawns analyzer (agent)
    │       └─ spawns clean-result-critic (agent) [+ codex twin every round]
    ├─ methodology-reference export (Step 9a-quater; auto-continue, no gate;
    │   │   v4 markdown: orchestrator copies the body's ## Methodology section to
    │   │   docs/methodology/issue_<N>.md, commits to main, gist-mirrors, and
    │   │   appends the top-of-body **Methodology:** link — all after
    │   │   clean-result-critic PASS. No separate authoring agent for v4 markdown.
    │   │   paper: true: SKIPPED — the Methods + Appendix are already in the .tex)
    ├─ (auto-complete step inline in the skill)
    ├─ (test-verdict gate inline in the skill, code-change paths only)
    ├─ spawns follow-up-proposer ∥ living-docs-updater ∥ related-work-finder
    │   (agents, one Step 10b/10c/10c-bis spawn batch; all three join before
    │    the Step 10d worktree merge; related-work-finder proposes a
    │    findings-keyed "Related findings" note for the ## Goal slot)
    └─ spawns follow-up-critic ∥ codex-follow-up-critic (the 5th doubled
        review site — a SINGLE-PASS redundancy screen run ONCE on the
        proposer's output BEFORE any proposal routes; reconciler on a
        not-redundant-vs-redundant disagreement; redundant proposals
        parked on_hold, not dropped)
```

The dedicated `reviewer` agent step was retired 2026-05-13; see the
ontology table below for the deprecation note.

This is healthy: skills coordinate, agents *do*, skills are reference.

---

## Current ontology (May 2026)

### Agents (roles — `.claude/agents/`)

| Name | Fresh-context reason |
|---|---|
| `planner` | Design role; produces a plan artifact |
| `critic` | Adversarial review of plans, must not see planner's reasoning |
| `consistency-checker` | Verifies single-variable changes vs parent experiments |
| `experiment-implementer` | Writes experiment-specific code (training scripts, configs, eval wiring) for a single issue; pairs with `code-reviewer` |
| `experimenter` | Background, long-running training + progressive monitoring on a pre-provisioned pod (does NOT write substantial code) |
| `implementer` | Standalone infra / refactor / utility code changes (NOT experiment-specific code) |
| `upload-verifier` | Mechanical artifact checklist, isolated from experimenter optimism |
| `analyzer` | Fresh-context analysis; produces fact sheet + interpretation |
| `interpretation-critic` | Adversarial review of interpretation, must not see analyzer reasoning. Branches on `paper:` frontmatter: for a `paper: true` task it reviews the LaTeX paper's claims (`docs/papers/issue_<N>/issue_<N>.tex`) + the figure PNGs (Lens 6 still loads them) against `eval_results/`; markdown-body behavior unchanged. Ensembled with `codex-interpretation-critic` on ALL rounds up to the per-reviewer cap (5) (round-1-only policy adopted 2026-05-13, REVERSED 2026-06-12 — cross-family reviewer diversity on every round). |
| `clean-result-critic` | Adversarial review of clean-result task bodies against the four-flat-H2 (v4) spec + exemplars (15 lenses, stable numbering, v4 section names: **1 Title**, **2 v4-structure** (`## Takeaways` 3–6 bullets · `## Goal` `**This experiment in context:**`/`**Broader narrative:**` · `## Methodology` `**Design:**`/`**Training:**` (complete hparam table)/`**Evaluation:**`/`**Data extraction:**`/`**Sample ...:**` slots · `## Results` ≥1 `### <result>` three-beat), **3 Figure** (one inline figure per result + the what-is-plotted → plot → interpretation three-beat), **4 Takeaways quality** (register, numbers-first, cross-round-synthesis currency — a Takeaways describing only round 1 after round 2 landed is a FAIL), **5 Footer/Reproducibility** (`**Repro:**` + `**Context:**`; reuse-provenance, confidence in H1 title tag only), **6 Voice** (bullet register incl. `byte identical` ban), **7 statistical-framing rule** (absorbed from the retired reviewer step), **8 mentor-facing title only** (methodology corrections fold into the relevant `### <result>` prose, no discrete heading), **9 one-result-one-figure per `### <result>`**, **10 Goal + Methodology completeness** (capsule trio: identity / why chosen / preprocessing; subset disclosure; link liveness; the complete hyperparameter table), **11 underlying data alongside every aggregate** (low-level per-unit data plot behind each aggregate stat — the broad parent — + raw-alongside-processed + per-cell artifacts), **12 conciseness** (cap adherence + bullets-over-prose), **13 planned-vs-actual coverage** (scope-shrinkage discipline), **14 binding-concerns audit** (LM-side companion to `verify_task_body.py`'s `check_concerns_audit`), **15 headline must not rest on a contaminated / failed-data-gate arm**; the full lens rubrics live on-demand in `.claude/rules/clean-result-critic-lens-reference.md` — the spec keeps the roster + `§` pointers, #1159). v3/v2/legacy bodies are reviewed under their grandfathered lens shape (SPEC.md § "v3 body shape" / § Grandfathered shape) — substitute the v3 section names (Findings, Data, `## Reproducibility` H2). **Branches on `paper:` frontmatter:** for a `paper: true` task the clean-result is a LaTeX research paper at `docs/papers/issue_<N>/` — the mechanical pre-pass is `scripts/verify_paper.py` (NOT `verify_task_body.py`), the reviewer reads the `.tex` + figure PNGs + compiled PDF, and SEVEN paper lenses bind INSTEAD of the fifteen markdown ones (**P1** self-standing Introduction · **P2** self-contained Methods + Rule-A reuse-chain depth · **P3** inline-subset + comprehensive-Appendix completeness · **P4** no confidence in the paper body · **P5** research-paper register · **P6** `\epsref{N}` correctness · **P7** verbatim examples + judge prompts — `verify_paper.py` checks 7-8 gate them); no `\metric` grounding lens in v1 (a v1.1 addition). **Final adversarial gate before status:awaiting_promotion as of 2026-05-13.** Ensembled with `codex-clean-result-critic` on ALL rounds up to the per-reviewer cap (5) (round-1-only policy REVERSED 2026-06-12). |
| `code-reviewer` | Adversarial review of implementer's diff, must be isolated. Ensembled all rounds with `codex-code-reviewer`. |
| `methodology-writer` | **Branches on `paper:` frontmatter.** **PAPER MODE (`paper: true`): SPAWNED** — authors the LaTeX paper's findings-blind **Methods section + recipe Appendix** and hands them to the `analyzer`, which splices them into the `.tex` (the analyzer never authors Methods/Appendix itself — the findings-blind firewall is the whole point). Early-spawned at the `/issue` Step 8 results-landed batch alongside the analyzer first pass. See § PAPER-TASK MODE. **MARKDOWN MODE — DEPRECATED for v4 (2026-W26):** under v4 the methodology doc is a mechanical COPY of the body's `## Methodology` section, exported by the `/issue` Step 9a-quater orchestrator (no fresh-context authoring) — committed to `main`, secret-gist-mirrored, linked at the top of the body; the analyzer writes the body's `## Methodology` section factually (it IS the canonical source), so for a v4 markdown body the orchestrator does NOT spawn this agent. It IS still spawned for grandfathered in-flight **v3/v2** markdown bodies (which carry no detailed `## Methodology` section to copy): a findings-blind generator of `docs/methodology/issue_<N>.md` (methodology + hyperparameters + verbatim worked examples) that never reads `## Takeaways` / `## Findings` / confidence tag / `epm:interpretation`. |
| `follow-up-proposer` | Reads results + plan, proposes concrete next experiments |
| `related-work-finder` | Independent generative literature search; fresh context so it positions the MEASURED finding against published work (replicates/contradicts/extends/none-found) without the analyzer's reasoning. Runs a bounded findings-keyed arXiv-MCP + web search, verifies every citation in the same turn (drop-if-unresolved), and PROPOSES (never applies) a ≤80-word "Related findings" note for the clean-result ## Goal -> **Broader narrative:** slot. Spawned in the Step 10b/10c/10c-bis post-completion batch (∥ follow-up-proposer + living-docs-updater); the related_work_positioning gate confirm/rejects it. v1 surfaces docs/papers.md candidates as a manual-triage list only (the papers.md auto-apply leg is a deferred follow-up). |
| `follow-up-critic` | Adversarial REDUNDANCY screen over follow-up proposals, must not see the proposer's reasoning. SINGLE-PASS (no revise loop): one binary verdict per proposal — `not-redundant` (proceed through existing routing) or `redundant` (park at `on_hold`, revivable). The bar is duplication ONLY (an existing experiment task / a settled open question / a higher-ranked sibling), NOT info-gain. Fires BEFORE any proposal routes. Ensembled with `codex-follow-up-critic` (the 5th doubled review site, added 2026-06-13). |
| `retrospective` | Fresh-context review of session transcripts |
| `research-pm` | Strategic PM persona for the dedicated PM session (loaded by `/pm`); owns queue triage + dispatch decisions, does NOT execute experiments or write code |
| `reconciler` | Binary tie-breaker for Codex ensemble adversarial review (`code-reviewer` / `critic` / `interpretation-critic` / `clean-result-critic` / `follow-up-critic`); marker + in-context output modes |
| `codex-code-reviewer` | Codex (gpt-5.5) twin of `code-reviewer`; thin Claude prompt-composer — composes a review prompt and returns its path; the orchestrator dispatches the OpenAI Codex plugin's `companion task` runtime (the wrapper never dispatches Codex itself — that's the orphan-job anti-pattern, incident task #533, 2026-06-10) |
| `codex-critic` | Codex twin of `critic` (per-lens, in-context output for /adversarial-planner Phase 2); thin Claude prompt-composer — composes a lens prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime |
| `codex-interpretation-critic` | Codex twin of `interpretation-critic` (7 lenses including multimodal lens 6); spawned every round up to the per-reviewer cap (5) (round-1-only until 2026-06-12); thin Claude prompt-composer — composes a critique prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime |
| `codex-clean-result-critic` | Codex twin of `clean-result-critic` (15 lenses against the four-flat-H2 (v4) spec — `## Takeaways` / `## Goal` / `## Methodology` / `## Results` + `**Repro:**`/`**Context:**` footer; confidence in H1 title tag only — 1 Title, 2 v4-structure, 3 Figure (+ three-beat), 4 Takeaways quality, 5 Footer/Reproducibility, 6 Voice, 7 statistical-framing rule, 8 mentor-facing title, 9 one-result-one-figure per `### <result>`, 10 Goal + Methodology completeness, 11 underlying data alongside every aggregate (low-level per-unit data plot behind each aggregate stat + raw-alongside-processed), 12 conciseness, 13 planned-vs-actual coverage, 14 binding-concerns audit, 15 headline not resting on a contaminated / failed-data-gate arm — lens text composed verbatim from `.claude/rules/clean-result-critic-lens-reference.md` at compose time, #1159); v3/v2/legacy bodies reviewed under the grandfathered names; branches on `paper:` frontmatter exactly as the Claude critic — for a `paper: true` task the composed Codex prompt inlines the SEVEN paper lenses (P1-P7, incl. P7 verbatim examples + judge prompts) + the `verify_paper.py` preamble + the `.tex`/figure-PNG/compiled-PDF read targets INSTEAD of the fifteen markdown lenses (no `\metric` lens in v1); spawned every round up to the per-reviewer cap (5) (round-1-only until 2026-06-12); thin Claude prompt-composer — composes the critique prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime; grounds on composer-inlined verify_task_body.py + audit_clean_results_body_discipline.py (markdown) / verify_paper.py (paper) output — the composer runs them outside the sandbox at compose time (#1050; this twin is dispatched read-only and uv cannot reliably execute in its sandbox) |
| `codex-follow-up-critic` | Codex twin of `follow-up-critic` (the 5th doubled review site, added 2026-06-13) — same SINGLE-PASS redundancy screen, same per-proposal `not-redundant | redundant` verdict, same nothing-dropped contract; thin Claude prompt-composer — composes the redundancy-screen prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime (the wrapper never dispatches Codex itself — orphan-job anti-pattern, #533) |
| ~~`reviewer`~~ | **DEPRECATED 2026-05-13.** Final adversarial responsibilities absorbed by `clean-result-critic` Lens 7 (statistical-framing rule). File kept for historical reference. |
| ~~`codex-reviewer`~~ | **DEPRECATED 2026-05-13** alongside `reviewer`. Replaced by `codex-clean-result-critic`. |

### Skills (playbooks — `.claude/skills/`)

| Name | Why a skill |
|---|---|
| `issue` | End-to-end orchestrator; calls gh, parses markers, dispatches agents |
| `adversarial-planner` | Sub-orchestrator: planner → critic → revise |
| `clean-results` | Manual consolidation / promotion protocol |
| `paper-plots` | Chart-building reference patterns + style spec |
| `daily` | Daily fan-out orchestrator: spawns parallel subagents (today: daily summary), each emits its own gist |
| `weekly` | Weekly fan-out orchestrator: spawns parallel subagents (summary, workflow-optimization, code-hygiene, mentor-prep), each emits its own gist |
| `experiment-runner` | Pre-flight + monitoring protocol for ML runs |
| `auto-experiment-runner` | Overnight queue automation |
| `experiment-proposer` | Prioritization ranking |
| `ideation` | Brainstorming protocol |
| `independent-reviewer` | Shared Principles-of-Honest-Analysis reference for analyzer + clean-result-critic (formerly: + reviewer, retired 2026-05-13) |
| `pm` | PM session bootstrap: loads the `research-pm` persona + spawns per-issue Happy sessions via `scripts/spawn_session.py` |
| `cleanup`, `refactor`, `codebase-debugger` | Code-hygiene workflows |

### Design notes

- **`research-pm` is the PM persona**, loaded into a dedicated PM Happy session
  by the `/pm` skill (introduced May 2026). It is NOT a subagent that dispatches
  others; it operates AS the user's primary interlocutor session. The user opens
  one PM session via `python scripts/spawn_session.py spawn-pm` and per-issue
  sessions via `spawn-issue --issue <N>`. Each session is independent (own
  context, own conversation history, own Happy chat). The PM session handles
  ranking + dispatch; per-issue sessions execute `/issue <N>`.
- **`experiment-runner` skill vs `experimenter` agent**: the skill is the
  monitoring protocol; the agent uses the skill. Keep both, they're layered
  correctly.
- **`clean-results` skill vs `analyzer` agent**: the analyzer owns single-
  experiment clean-result creation; `clean-results` is only for multi-issue
  consolidation + manual promotion. No overlap.
- **Workflow-fix work is a `kind: infra` task, NOT a dedicated agent (#678).**
  A workflow-surface fix is filed as a `kind: infra` task and implemented by a
  background `/issue <N> --auto` session (the standard `implementer` at Step
  4b) via the full code-change pipeline — see `.claude/rules/workflow-fix-on-bug.md`.
  The retired `workflow-improver` agent (frozen with a DEPRECATED banner) is
  the cautionary example here: do NOT recreate a `workflow-improver`-shaped
  dedicated agent — the `/issue` pipeline + `implementer` already own that role
  under full review.
