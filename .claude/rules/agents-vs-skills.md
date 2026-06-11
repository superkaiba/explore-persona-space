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
    ├─ spawns upload-verifier ∥ analyzer first pass (held) ∥ methodology-writer
    │   │   (results-landed parallel batch; epm:interpretation publishes
    │   │    only after upload-verification PASS; pod terminate only after PASS)
    ├─ iterates analyzer ↔ interpretation-critic    (max 3 rounds, content honesty;
    │   │                                            round 1 ENSEMBLED with codex-interpretation-critic,
    │   │                                            rounds 2-3 Claude only)
    │       ├─ spawns analyzer (agent, uses /paper-plots)
    │       └─ spawns interpretation-critic (agent) [+ codex twin on round 1]
    ├─ iterates analyzer ↔ clean-result-critic      (max 3 rounds, structure + register
    │   │                                            + statistical-framing rule;
    │   │                                            round 1 ENSEMBLED with codex-clean-result-critic,
    │   │                                            rounds 2-3 Claude only; FINAL adversarial
    │   │                                            gate as of 2026-05-13)
    │       ├─ re-spawns analyzer (agent)
    │       └─ spawns clean-result-critic (agent) [+ codex twin on round 1]
    ├─ methodology-reference LATE JOIN (Step 9a-quater; auto-continue, no gate;
    │   │   the findings-blind methodology-writer was early-spawned at the
    │   │   results-landed batch above; orchestrator commits the doc on agent
    │   │   return, then gist + body link-append (top-of-body
    │   │   **Methodology:** line + ## Reproducibility row) after
    │   │   clean-result-critic PASS)
    ├─ (auto-complete step inline in the skill)
    ├─ (test-verdict gate inline in the skill, code-change paths only)
    └─ spawns follow-up-proposer ∥ living-docs-updater (agents, one Step
        10b/10c spawn batch; both join before the Step 10d worktree merge)
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
| `interpretation-critic` | Adversarial review of interpretation, must not see analyzer reasoning. Round 1 ensembled with `codex-interpretation-critic`; rounds 2-3 Claude only (round-1-only policy adopted 2026-05-13). |
| `clean-result-critic` | Adversarial review of clean-result task bodies against the 2-content-section nested-design (v2) spec + exemplars (15 lenses: title, TL;DR (`### Motivation` + `### What I ran` + `### Findings` (parent) → `#### <finding>` per result for v2-sentinelled bodies — absorbs the retired Details-narrative lens), figure, reproducibility (confidence in H1 title tag only for v2 bodies), voice incl. `byte identical` ban, **Lens 7 statistical-framing rule** absorbed from the retired reviewer step, **Lens 8 mentor-facing title only** (methodology corrections fold into the relevant `#### <finding>` prose, no discrete H3 — added 2026-05-26), **Lens 9 one-takeaway-one-figure per `#### <finding>` H4** added 2026-05-26, **Lens 10 eval-probe descriptions inside `## TL;DR`** added 2026-05-26, **Lens 11 raw alongside processed (figures + prose + per-cell artifacts)** added 2026-05-27, **Lens 12 story arc present (TL;DR narrative shape)** added 2026-05-27, **Lens 13 planned-vs-actual coverage (scope-shrinkage discipline)** added 2026-05-27 after task #391's C-axis silent drop, **Lens 14 binding-concerns audit** (LM-side companion to `verify_task_body.py`'s `check_concerns_audit`) added 2026-05-31 by task #455, **Lens 15 headline must not rest on a contaminated / failed-data-gate arm** added 2026-06-01 after task #407. **Final adversarial gate before status:awaiting_promotion as of 2026-05-13.** Round 1 ensembled with `codex-clean-result-critic`; rounds 2-3 Claude only. |
| `code-reviewer` | Adversarial review of implementer's diff, must be isolated. Ensembled all rounds with `codex-code-reviewer`. |
| `methodology-writer` | Findings-blind generator of `docs/methodology/issue_<N>.md` (methodology + hyperparameters + verbatim worked examples). Fresh context is the structural enforcement of "no interpretation" — never reads `## TL;DR`, `## Findings`, confidence tag, or `epm:interpretation`. EARLY-SPAWNED by `/issue` at the Step 8 results-landed parallel batch (inputs are final once results land, so it runs concurrently with upload verification + the interpretation loop); the gist publish + body link-append (top-of-body `**Methodology:**` line + `## Reproducibility` row) LATE-JOIN at Step 9a-quater (after `clean-result-critic` PASS, before `awaiting_promotion` park). Runs for `kind: experiment` and methodology-bearing `kind: analysis` tasks; skipped for `infra | batch | survey`. The orchestrator commits the doc on agent return, publishes a secret gist (fail-soft, no-secrets pre-scan), and links from the top of the body + `## Reproducibility`. |
| `follow-up-proposer` | Reads results + plan, proposes concrete next experiments |
| `retrospective` | Fresh-context review of session transcripts |
| `research-pm` | Strategic PM persona for the dedicated PM session (loaded by `/pm`); owns queue triage + dispatch decisions, does NOT execute experiments or write code |
| `reconciler` | Binary tie-breaker for Codex ensemble adversarial review (`code-reviewer` / `critic` / `interpretation-critic` / `clean-result-critic`); marker + in-context output modes |
| `codex-code-reviewer` | Codex (gpt-5.5) twin of `code-reviewer`; thin Claude prompt-composer — composes a review prompt and returns its path; the orchestrator dispatches the OpenAI Codex plugin's `companion task` runtime (the wrapper never dispatches Codex itself — that's the orphan-job anti-pattern, incident task #533, 2026-06-10) |
| `codex-critic` | Codex twin of `critic` (per-lens, in-context output for /adversarial-planner Phase 2); thin Claude prompt-composer — composes a lens prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime |
| `codex-interpretation-critic` | Codex twin of `interpretation-critic` (7 lenses including multimodal lens 6); round-1-only; thin Claude prompt-composer — composes a critique prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime |
| `codex-clean-result-critic` | Codex twin of `clean-result-critic` (15 lenses against the 2-content-section nested-design (v2) spec — `### Motivation` + `### What I ran` + `### Findings` (parent) → `#### <finding>` per result; confidence in H1 title tag only — including Lens 7 statistical-framing rule, Lens 8 mentor-facing title, Lens 9 one-takeaway-one-figure per `#### <finding>`, Lens 10 eval-probe descriptions inside `## TL;DR`, Lens 11 raw alongside processed, Lens 12 story arc present, Lens 13 planned-vs-actual coverage, Lens 14 binding-concerns audit, Lens 15 headline not resting on a contaminated / failed-data-gate arm); round-1-only; thin Claude prompt-composer — composes the critique prompt and returns its path; the orchestrator dispatches Codex's `companion task` runtime; runs verify_task_body.py + audit_clean_results_body_discipline.py independently |
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
