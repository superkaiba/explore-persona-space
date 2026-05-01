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
| Polished write-up per experiment | **GitHub clean-result issue** (label `clean-results`) | TL;DR + interpretation + confidence |
| Headline-level findings | `RESULTS.md` | Cross-experiment claims a paper would cite |
| Results index | `eval_results/INDEX.md` | Pointer table from issue # → result JSON path |
| Ideas backlog | `docs/research_ideas.md` | Pre-issue brainstorm/promotion candidates |

The legacy file-based research log (`research_log/`) has been retired and
moved to `archive/research_log/`. Do not write there. The clean-result
GitHub issue created by the analyzer at the end of `/issue` is the durable,
canonical artifact for every experiment.

## Experiment Queue

**The GitHub project board IS the queue.** Every experiment is a GitHub issue
carrying its lifecycle state in a `status:*` label (`proposed` →
`planning` → `plan-pending` → `approved` → `implementing` →
`code-reviewing` → `running` → `uploading` → `interpreting` →
`reviewing` → `done-experiment` / `done-impl`). Filter with
`gh issue list --label status:<state>`. There is no markdown queue file.

Each issue's body must be actionable:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

Raw ideation output (pre-issue brainstorms from `/ideation`) lives at
`docs/ideas/YYYY-MM-DD.md`. The user promotes worthwhile ideas to GitHub
issues with `gh issue create --label status:proposed`.

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
`ideation`, `mentor-prep`, `daily-update`) and
`.claude/rules/agents-vs-skills.md`.
