---
title: Always-on LESSONS.md index + wire into agents (close rule load-timing gap)
kind: infra
tags:
- workflow-improvement
created_at: '2026-06-29T23:16:26Z'
has_clean_result: false
origin_prompt: 'run A as a workflow improvement in the background (Option A = always-on
  lessons index + on-demand details; close the rule load-timing gap exposed by #722)'
---
## Overview / Motivation

User-directed workflow improvement (Thomas, 2026-06-29). The project's
lessons-learned are distributed across `.claude/rules/*.md` (~19 files),
per-agent `.claude/agent-memory/**`, and CLAUDE.md. The rule files load
ON-DEMAND via their `paths:` frontmatter — i.e. a rule enters context only when
the session reads/edits a file matching one of its globs. That trigger fires on
FILE ACCESS, so it can fire **too late** (the planner reasons in a *plan* file
before any matching script is open) or **never** (the decision never opens a
matching file). A few critical rules are hand-patched with an always-on
one-liner in CLAUDE.md for exactly this reason; most are not.

**Worked failure (#722, 2026-06-29):** `vectorize-many-cell-fits.md` has no
always-on backstop, so at plan time the #722 planner judged a per-fold MLP LOCO
sweep "not GPU-worthy … CPU-feasible ~30-60 min" — the rule that says exactly
"vectorize these, they are overhead-bound" was not in context (no matching
`scripts/issue*_skill*.py` open yet). The sweep then ran ~19.5 CPU-hours on the
shared VM before being caught.

Fix = an ALWAYS-ON lessons INDEX so every relevant agent always knows each lesson
exists and WHEN it applies, while the full rule text stays on-demand (keeps
context lean). This is "Option A" from the chat: always-on index + on-demand
details. The rejected alternative (B) was inlining all rule bodies into one
always-on file — tens of K tokens added to every agent call, the cost the
on-demand design exists to avoid.

## Goal

Create an always-on lessons-learned index and wire it so the relevant workflow
agents always load it at PLAN TIME, closing the load-timing gap, WITHOUT paying
the tens-of-K-token cost of always-loading every rule's full text.

## Deliverables

1. **`.claude/rules/LESSONS.md`** — an always-on index. One row per
   `.claude/rules/*.md` file:
   - rule name,
   - a one-line **"fires when: <situation>"** trigger phrased in PLAN-TIME terms
     (the decision/situation that should make an agent pull the rule), derived
     from the rule's `description` + `paths`,
   - a pointer to the file.
   Also surface, per agent, that `.claude/agent-memory/<agent>/` holds persistent
   anti-patterns/feedback worth checking. Keep the whole index LEAN (target a few
   hundred tokens — it is always-on). Do NOT inline rule bodies.

2. **Always-on wiring.** Make the index actually load every session for the
   relevant agents (planner, critic, consistency-checker, experiment-implementer,
   implementer, analyzer, code-reviewer, interpretation-critic,
   clean-result-critic) — via a CLAUDE.md pointer/section AND/OR each agent's
   frontmatter/prompt, whichever this harness reliably keeps in context. The
   index must be present at PLAN TIME (the #722 gap), not only when a matching
   file is opened.

3. **Two new lessons (capture + add their index rows):**
   a. **Supersede → delete the old version.** Code-hygiene rule: when an improved
      version of a script supersedes an old one, DELETE the old version (after
      rewiring every reference) rather than keeping both — don't accumulate
      duplicate/stale scripts. Worked example: `issue722_vectorized_skill.py`
      supersedes `issue722_skill_over_mean.py`. Land it in `code-style.md` or a
      small dedicated rule.
   b. **Built-but-stranded fix.** Coordination lesson: a documented — or even
      fully BUILT — fix that is not merged to main / not wired into the running
      path does not help; the running session uses the old slow path anyway.
      Worked example: `vectorized_mlp_skill.py` + a vectorized #722 driver
      existed on an unmerged branch while the #722 session ran the old CPU script
      with a ~38h ETA. When a rule prescribes a helper, ensure the helper is on
      main and wired in. Land it where coordination lessons live.

4. **Anti-drift lint.** Add a `scripts/workflow_lint.py` check (folded into the
   no-flags default run) asserting every `.claude/rules/*.md` has a matching row
   in `LESSONS.md` and vice-versa, so the index cannot silently drift as rules
   are added/removed. Add/adjust the corresponding test under `tests/`.

5. **General always-on "vectorize/parallelize by default" principle.** Add a
   one-to-two-line standing rule to CLAUDE.md's Critical Rules (and its row in
   `LESSONS.md`): *default to vectorized / batched / parallel execution — a
   Python loop over INDEPENDENT units (LOCO folds, cells, layers, dims, rows,
   model forwards, judge/API calls, subagent tasks) is a smell; batch it into
   tensor ops / a vLLM batch / concurrent dispatch before reaching for more
   compute.* Name the recurring offenders and point to the specific rules that
   already exist but lack an always-on umbrella: `code-style.md` ("vectorize
   torch ops — no Python loops over tensor dims" + "batch data-parallel
   forwards"), `.claude/rules/vectorize-many-cell-fits.md` (per-fold/per-cell GD
   sweeps), the always-on "use vLLM for generation, never sequential
   `model.generate()`" bullet, and the "don't sequentially block N parallel
   subagents" bullet. This is the umbrella those scattered, mostly-on-demand
   rules currently lack — exposed by the #722 load-timing gap (the vectorize rule
   is on-demand, so it was not in context at plan time). Keep it to ~1-2 lines in
   CLAUDE.md; the specifics stay in their on-demand rules.

## Constraints / invariants

- Workflow-surface only (`.claude/rules/**`, `.claude/agents/*.md`, `CLAUDE.md`,
  `scripts/workflow_lint.py`, `tests/`). No experiment code, configs, or `tasks/`.
- ADDITIVE / non-architectural: a new index doc + pointers + a lint check. No
  status-enum / marker-schema / `task.py`-subcommand / agent-file-relocation
  changes. Should auto-approve under the 0-GPU-h plan gate and self-merge at
  Step 10d.
- Keep the index LEAN — reliability WITHOUT the full-text always-on token cost is
  the whole point. Do NOT inline rule bodies into `LESSONS.md`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `CLAUDE.md` changes, keep it consistent with the rule files.

## Provenance

- origin: user chat directive, 2026-06-29 — "run A as a workflow improvement in
  the background." Option A = always-on lessons index + on-demand details, chosen
  after the #722 load-timing-gap diagnosis.
