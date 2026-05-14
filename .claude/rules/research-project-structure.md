---
description: Research project structure conventions, results index, and experiment queue
globs:
  - "RESULTS.md"
  - "eval_results/**"
  - "docs/**"
---

# Research Project Structure

## Result Artifacts (one source of truth per layer)

| Artifact | Lives at | Authoritative for |
|---|---|---|
| Per-run structured results (JSON) | `eval_results/<name>/run_result.json` + WandB Artifact | Raw numbers, reproducibility metadata |
| Polished write-up per experiment | **Source experiment row in the task workflow, promoted in place** (body replaced with polished write-up, `has_clean_result=true`, one child `runs` row) | Human TL;DR + AI TL;DR + AI Summary + confidence |
| Headline-level findings | `RESULTS.md` | Cross-experiment claims a paper would cite |
| Results index | `eval_results/INDEX.md` | Pointer table from task number → result JSON path |
| Ideas backlog | `docs/research_ideas.md` | Pre-experiment brainstorm/promotion candidates |

The legacy file-based research log (`research_log/`) has been retired
and moved to `archive/research_log/`. Do not write there. The source
experiment row, promoted in place to a clean-result by the analyzer
at the end of `/issue`, is the durable, canonical artifact for every
experiment.

## Experiment Queue

**The `tasks/` directory tree IS the queue.** Every experiment is a
row carrying its lifecycle state in the `status` enum (`proposed` →
`planning` → `plan_pending` → `approved` → `running` → `verifying` →
`interpreting` → `reviewing` → `awaiting_promotion` → `completed` /
`archived`). Filter with `python scripts/task.py list-by-status
--status <state>` or browse the kanban at
<https://eps.superkaiba.com/>. There is no markdown queue
file.

Each experiment's body must be actionable:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

Raw ideation output (pre-experiment brainstorms from `/ideation`) lives
at `docs/ideas/YYYY-MM-DD.md`. The user promotes worthwhile ideas to
experiment rows via the dashboard or `POST /api/experiments`.

## Environment Bootstrap

Every entrypoint calls `setup_env()` from `src/explore_persona_space/utils.py`:
- Loads `.env` (API keys)
- Sets `HF_HOME` to persistent storage (`/workspace/.cache/huggingface` on RunPod)
- All environment setup lives in code — never manually export variables

## Agent Roles

See `.claude/agents/` for the authoritative per-agent descriptions
(`experiment-implementer`, `experimenter`, `implementer`, `analyzer`,
`reviewer`, `code-reviewer`, `interpretation-critic`, `critic`, `planner`,
`consistency-checker`, `upload-verifier`, `follow-up-proposer`,
`retrospective`). Strategic orchestration lives in skills, not agents — see
`.claude/skills/` (`issue`, `adversarial-planner`, `experiment-proposer`,
`ideation`, `daily`, `weekly`) and
`.claude/rules/agents-vs-skills.md`.
