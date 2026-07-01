---
title: 'workflow-fix: code-reviewer catches plan-DP-shape vs dispatcher-shape gap
  (from #779)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:17a98aa12cb2
created_at: '2026-07-01T09:54:46Z'
has_clean_result: false
origin_prompt: 'Task #779 round-6 code-review PASSed on a diff whose plan §9 declared
  8-GPU DP but dispatcher only supports --gpu-id N (single GPU); the mismatch surfaced
  post-launch when pod-util showed 8 H100s at 0%. Round 7 implementer descoped to
  1×H100. Both round-7 reviewers (Claude PASS + Codex PASS) explicitly flagged the
  plan-shape-vs-dispatcher-shape check as worth adding to the code-reviewer contract
  as a preventive lens.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix follow-up raised on task #779 round 7 (emitting agents: `experiment-implementer` v7 prose + Claude `code-reviewer` v7 + Codex `code-reviewer-codex` v7).

## Goal

Add a plan-shape-vs-dispatcher-shape check to the code-reviewer contract so a mid-flight compute-shape mismatch (plan §9 declares 8-GPU DP → but the dispatcher exposes no `--shard-id`/`--num-shards`/internal DP) is caught at Step 5 pre-launch, not at first-run pod-idle util reading.

## Workflow gap

- **Bug observed:** task #779 round-6 code-review PASSed (Claude + Codex + reconciler) on a `kind: experiment` diff whose plan §9 explicitly called for "8-GPU DP, 8 single-GPU CUDA_VISIBLE_DEVICES workers sharding contexts" but whose dispatcher scripts (`scripts/issue779_extract_rb.py`, `issue779_collect.py`, `issue779_stage1.py`) only accept `--gpu-id N` (single-GPU) — no `--shard-id`/`--num-shards`, no `torch.multiprocessing.spawn`, no `torch.distributed`. The experimenter provisioned the plan's `sweep-8g-h100` (8×H100) pod, launched via `nohup`, and the first pod-util reading showed all 8 GPUs at 0% (the workload crashed at the very first phase's Claude-artifact-validation, which is a separate bug, but the DP gap would have burned 7×H100 idle regardless). The orchestrator descoped to 1×H100 (`lora-7b`) at round 7 to eliminate the #664-class spend leak. Task #779 body now records the compute-deviation post-launch, but the gap slipped past round 6's ensemble review because neither Claude nor Codex `code-reviewer` explicitly checks "plan-declared compute shape ↔ dispatcher exposes that shape".
- **Why it is a workflow gap:** `.claude/agents/code-reviewer.md` Step 0.6 (smoke gate) covers phase-coverage + exit-0 + artifact-digest, and Step 0.7 (contract-verify) covers marker shape. Neither has a lens that reads plan §9's declared parallelism (DP shape, worker count, shard mechanism) and confirms the dispatcher scripts expose the matching flags/DP entrypoint. Result: any experiment whose plan declares a DP shape the impl doesn't support silently ships to a bigger-than-needed pod → 7×H100 (or 3×A100, or 7×L4) idle burn until human notices.
- **Confidence (emitter):** medium — both round-7 reviewers explicitly flagged this in prose; the implementer's `## Follow-up workflow concerns` recommended exactly this workflow-fix. Mechanically the check is straightforward (grep the plan for a DP/shard declaration, then grep the dispatcher `--help` / argparse for a `--shard-id`/`--num-shards` flag, or a `torch.multiprocessing`/`torch.distributed` call — if plan declares DP but dispatcher exposes none, FAIL with blocker tag `compute-shape-mismatch`).

## Proposed change (candidate diff sketch — refine in planning)

Add a lens (call it Step 0.65 "compute-shape-check" or extend Step 0.6) to BOTH `.claude/agents/code-reviewer.md` AND `.claude/agents/codex-code-reviewer.md`:

```
### Step 0.65 — compute-shape-vs-dispatcher-check (kind: experiment only)

Read the approved plan's §9 (or the compute-plan section) and grep for a DP declaration:
- "8-GPU DP" / "N-GPU DP" / "data-parallel" / "sharding contexts" / "CUDA_VISIBLE_DEVICES workers"
- "TP=N" / "tensor-parallel" — this one IS legitimately supported by many single-script paths, less prone to slip.

If the plan declares a DP shape, verify at least ONE of:
(a) the dispatcher script exposes a `--shard-id N --num-shards K` (or equivalent) flag pair.
(b) the dispatcher script internally spawns workers via `torch.multiprocessing.spawn` / `torch.distributed.run` / a per-worker `subprocess.Popen`.
(c) the workload wrapper (a `scripts/issue<N>_dispatch.sh` or the experimenter's report) explicitly fans out ONE dispatch process per GPU with distinct `--gpu-id` values, each processing a shard.

Fail with `compute-shape-mismatch` blocker tag if plan declares DP but none of (a)/(b)/(c) is present.
```

Add the corresponding blocker tag to `code-reviewer.md` Step 7 taxonomy: `compute-shape-mismatch`. Add mechanizable-verify examples.

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md`, `.claude/agents/codex-code-reviewer.md`.
- Grep the workflow surface first: is there any adjacent existing lens that covers this? (Grep for "DP" / "data-parallel" / "shard-id" in `.claude/agents/*.md` + `.claude/skills/issue/SKILL.md` — the current hits are cross-referencing the plan spec, not a review-time check.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The new lens should NOT bounce a legitimate single-GPU experiment (the plan explicitly declares 1×H100 or single-GPU → no check needed).
- The lens should read the CANONICAL approved plan on `main`, not the possibly-stale worktree copy (per SKILL Step 5a spec-freshness discipline).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md,.claude/agents/codex-code-reviewer.md
- fingerprint: (auto)
- Emitted by: experiment-implementer v7 (task #779) prose + Claude/Codex code-reviewer v7 concurrence
