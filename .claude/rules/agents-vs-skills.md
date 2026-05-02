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

- **Independence is load-bearing.** Example: the `reviewer` must not see the
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
  Example: `paper-plots` (chart-building protocol), `mentor-prep` (assembly
  task), `clean-results` (manual consolidation steps).
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
    │       └─ spawns critic    (agent)
    ├─ spawns consistency-checker (agent)
    ├─ spawns experimenter (agent)
    │       └─ uses /experiment-runner (skill: monitoring protocol)
    ├─ spawns upload-verifier (agent)
    ├─ iterates analyzer ↔ interpretation-critic    (max 3 rounds)
    │       ├─ spawns analyzer (agent, uses /paper-plots)
    │       └─ spawns interpretation-critic (agent)
    ├─ spawns reviewer (agent)
    ├─ (auto-complete step inline in the skill)
    ├─ (test-verdict gate inline in the skill, code-change paths only)
    └─ spawns follow-up-proposer (agent)
```

This is healthy: skills coordinate, agents *do*, skills are reference.

---

## Current ontology (April 2026)

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
| `interpretation-critic` | Adversarial review of interpretation, must not see analyzer reasoning |
| `reviewer` | Final adversarial review of published clean-result issue |
| `code-reviewer` | Adversarial review of implementer's diff, must be isolated |
| `follow-up-proposer` | Reads results + plan, proposes concrete next experiments |
| `retrospective` | Fresh-context review of session transcripts |

### Skills (playbooks — `.claude/skills/`)

| Name | Why a skill |
|---|---|
| `issue` | End-to-end orchestrator; calls gh, parses markers, dispatches agents |
| `adversarial-planner` | Sub-orchestrator: planner → critic → revise |
| `clean-results` | Manual consolidation / promotion protocol |
| `paper-plots` | Chart-building reference patterns + style spec |
| `mentor-prep` | Assembly of recent clean-results into a meeting document |
| `experiment-runner` | Pre-flight + monitoring protocol for ML runs |
| `auto-experiment-runner` | Overnight queue automation |
| `experiment-proposer` | Prioritization ranking |
| `ideation` | Brainstorming protocol |
| `daily-update` | Standup assembly |
| `independent-reviewer` | Shared Principles-of-Honest-Analysis reference for analyzer + reviewer |
| `cleanup`, `refactor`, `codebase-debugger` | Code-hygiene workflows |

### Design notes

- **No strategic-PM agent.** Both `manager` and `research-pm` were removed
  in April 2026 — the workflow now dispatches work via `/issue`,
  `/experiment-proposer`, `/ideation`, `/adversarial-planner`,
  `/mentor-prep`, `/daily-update`. "What should we do next?" is a
  main-session question answered by invoking the right skill; it does not
  need its own agent persona.
- **`experiment-runner` skill vs `experimenter` agent**: the skill is the
  monitoring protocol; the agent uses the skill. Keep both, they're layered
  correctly.
- **`clean-results` skill vs `analyzer` agent**: the analyzer owns single-
  experiment clean-result creation; `clean-results` is only for multi-issue
  consolidation + manual promotion. No overlap.
