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
- **Persona / role encapsulation.** Example: `gate-keeper` is opinionated
  ("should we run this?"); you want its voice separate from the main
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
    ├─ spawns gate-keeper   (agent)
    ├─ runs /adversarial-planner (skill: inner orchestrator)
    │       ├─ spawns planner   (agent)
    │       ├─ spawns critic    (agent)
    │       └─ spawns fact-checker (agent)
    ├─ spawns experimenter (agent)
    │       └─ uses /experiment-runner (skill: monitoring protocol)
    ├─ spawns analyzer (agent)
    │       └─ uses /paper-plots (skill: chart patterns)
    ├─ spawns reviewer (agent)
    └─ (auto-complete step inline in the skill)
```

This is healthy: skills coordinate, agents *do*, skills are reference.

---

## Current ontology (April 2026)

### Agents (roles — `.claude/agents/`)

| Name | Fresh-context reason |
|---|---|
| `gate-keeper` | Opinionated RUN/MODIFY/SKIP voice, separate from main session |
| `planner` | Design role; produces a plan artifact |
| `critic` | Adversarial review of plans, must not see planner's reasoning |
| `fact-checker` | Independent verification of claims in plans |
| `experimenter` | Background, long-running training + monitoring |
| `implementer` | Code changes with scoped file access |
| `analyzer` | Fresh-context analysis; produces the clean-result issue |
| `reviewer` | Adversarial review of analyzer's output, must be isolated |
| `code-reviewer` | Adversarial review of implementer's diff, must be isolated |
| `gate-keeper` | RUN/MODIFY/SKIP scoring |
| `retrospective` | Fresh-context review of session transcripts |
| `manager` | Strategic PM persona the user spawns interactively |
| `research-pm` | Same role as `manager` (see **Overlap flags** below) |

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
| `cleanup`, `refactor`, `deep-clean`, `codebase-debugger`, `simplify` | Code-hygiene workflows |

### Overlap flags

- **`manager` vs `research-pm`**: nearly identical roles. `research-pm` is
  the newer, more detailed version (added this quarter). Recommend
  deprecating `manager.md` in favor of `research-pm.md`. Not done yet — flag
  for a follow-up.
- **`experiment-runner` skill vs `experimenter` agent**: the skill is the
  monitoring protocol; the agent uses the skill. Keep both, they're layered
  correctly.
- **`clean-results` skill vs `analyzer` agent**: after the April 2026
  refactor, the analyzer owns single-experiment clean-result creation;
  `clean-results` is now only for multi-issue consolidation + manual
  promotion. No overlap.
