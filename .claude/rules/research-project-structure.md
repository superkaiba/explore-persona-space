---
description: Research project structure conventions, logging, and experiment queue
globs:
  - "research_log/**"
  - "RESULTS.md"
  - "eval_results/**"
---

# Research Project Structure

## Research Log

Two tiers: `drafts/` (auto-generated, unreviewed) and root (approved).

```
research_log/
├── LOG.md               # clean running log (approved TLDRs)
├── drafts/
│   ├── LOG.md           # dirty running log (auto-generated)
│   └── *.md             # auto-generated experiment write-ups
└── *.md                 # reviewed and approved write-ups
```

Flow: experiment finishes → draft in `drafts/` → review → move to root, add TLDR to `LOG.md`.

## Experiment Queue

**The GitHub project board IS the queue.** Every experiment is a GitHub issue
carrying its lifecycle state in a `status:*` label (`proposed` →
`gate-pending` → `planning` → `plan-pending` → `approved` → `running` →
`reviewing` → `testing` → `done-experiment` / `done-impl`). Filter with
`gh issue list --label status:<state>`. There is no markdown queue file —
it was deleted to eliminate drift between the file and the board.

Each issue's body must be actionable:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

Raw ideation output (pre-issue brainstorms from `/ideation`) lives at
`research_log/ideas/YYYY-MM-DD.md`. The user promotes worthwhile ideas
to GitHub issues with `gh issue create --label status:proposed`.

## Environment Bootstrap

Every entrypoint calls `setup_env()` from `src/explore_persona_space/utils.py`:
- Loads `.env` (API keys)
- Sets `HF_HOME` to persistent storage (`/workspace/.cache/huggingface` on RunPod)
- All environment setup lives in code — never manually export variables

## Agent Roles

See `.claude/agents/` for the authoritative per-agent descriptions (experimenter, implementer, analyzer, reviewer, code-reviewer, critic, gate-keeper, planner, retrospective). Strategic orchestration lives in skills, not agents — see `.claude/skills/` (issue, adversarial-planner, experiment-proposer, ideation, mentor-prep, daily-update) and `.claude/rules/agents-vs-skills.md`.
