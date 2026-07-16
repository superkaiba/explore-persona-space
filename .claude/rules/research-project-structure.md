---
description: Research project structure conventions, results index, and experiment queue
paths:
  - "RESULTS.md"
  - "eval_results/**"
  - "docs/**"
---

# Research Project Structure

## Result Artifacts (one source of truth per layer)

| Artifact | Lives at | Authoritative for |
|---|---|---|
| Per-run structured results (JSON) | `eval_results/<name>/run_result.json` + WandB Artifact | Raw numbers, reproducibility metadata |
| Polished write-up per experiment | **Source experiment row in the task workflow, promoted in place** (body replaced with polished write-up, `has_clean_result=true`, one child `runs` row) | Takeaways + Goal + Methodology + Results (v4 four-flat-H2 spec + `**Repro:**` / `**Context:**` footer); confidence in the H1 title tag |
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
`interpreting` → `reviewing` → `awaiting_promotion` →
[`followups_running` while a same-issue follow-up round executes] →
`completed` / `archived`), plus the non-lifecycle `on_hold` parking
status that sits left of `proposed` (tasks set aside, excluded from
auto-dispatch, revivable via `set-status <N> proposed`). Filter with
`python scripts/task.py list-by-status
--status <state>` or browse the kanban at
<https://eps.superkaiba.com/>. There is no markdown queue
file.

Each experiment's body must be actionable:
- BAD: "Try different learning rates"
- GOOD: "SFT Llama3-8B on UltraChat, lr=3e-5, 3 epochs, LoRA r=16"

Raw ideation output (pre-experiment brainstorms from `/ideation`) lives
at `docs/ideas/YYYY-MM-DD.md`. The user promotes worthwhile ideas to
tasks via `uv run python scripts/task.py new --kind experiment
--title "..." --body-file ...`.

## Environment Bootstrap

Every entrypoint calls `load_dotenv()` from
`src/explore_persona_space/orchestrate/env.py` (there is NO `setup_env()`
in `utils.py` — that name is stale; importing it crashes):
- Loads `.env` (API keys) — `resolve_dotenv_path()` walks up from cwd
- Sets `HF_HOME` to persistent storage (`/workspace/.cache/huggingface` on RunPod)
- All environment setup lives in code — never manually export variables

For ad-hoc inline shell/python one-liners that need API keys (HF-Hub
fitness checks, quick probes), the canonical recipe is:
`set -a && source .env && set +a && uv run python - <<'PY' ... PY`
— never a bare `load_dotenv()` inside a heredoc (its no-arg
`find_dotenv()` stack-walk crashes from stdin; see gotchas.md). For
`scripts/*.sh` this is enforced mechanically by
`scripts/workflow_lint.py --check-heredoc-dotenv` (bundled into the
no-flags default run; incidents #552/#612). This recipe is VM-scoped
(repo root, where `.env` always exists): pod/GCE workload scripts must
source conditionally instead — `if [ -f ./.env ]; then set -a; . ./.env;
set +a; fi`, never unconditional sourcing inside the `&&`-chain of a
command whose exit code is classified — because the GCE lane exports
tokens via its startup script and has NO `.env` file (see
`.claude/rules/gotchas.md`; incident #923).

## Agent Roles

See `.claude/agents/` for the authoritative per-agent descriptions
(`experiment-implementer`, `experimenter`, `implementer`, `analyzer`,
`reviewer`, `code-reviewer`, `interpretation-critic`, `critic`, `planner`,
`consistency-checker`, `upload-verifier`, `follow-up-proposer`,
`retrospective`). Strategic orchestration lives in skills, not agents — see
`.claude/skills/` (`issue`, `adversarial-planner`, `experiment-proposer`,
`ideation`, `daily`, `weekly`) and
`.claude/rules/agents-vs-skills.md`.
