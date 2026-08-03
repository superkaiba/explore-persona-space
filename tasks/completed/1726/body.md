---
title: 'daily-fix: long fit loops must checkpoint per cell and log p'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2b40afcd8d52
- daily-auto-filed
created_at: '2026-07-27T07:17:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): a fit phase ran 5 h 14
  m emitting no log line after its phase banner and writing no output file, because
  the loop accumulates every cell in memory and writes once per layer, making the
  phase un-resumable and indistinguishable from a wedge'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 2 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Require every long unit-loop to emit a per-unit progress line and to trip the
checkpoint requirement on a unit COUNT (~50) as well as a projected wall-time, and
mirror both into the implementation-review gates that already own long-loop
restartability.

## Workflow gap

- **Bug observed:** `#1689`'s `fit_ladder` phase ran 5 h 14 m at 0% GPU emitting zero log
  output after its `[phase=fit_ladder]` line with `eval_results/issue_1689/ladder/`
  non-existent, because `run_all_pairs` accumulates all 126 pairs × 2 arms in an in-memory
  dict and writes one JSON per layer only after the whole layer finishes — so the phase was
  indistinguishable from a wedge and a crash, preemption or stop would have lost 100% of the
  compute.
- **Why it is a workflow gap:** the per-unit PROGRESS-LINE / liveness requirement exists
  nowhere on the workflow surface (0 hits across `.claude/rules/` + `.claude/agents/`), and
  the checkpoint requirement that does exist fires only off a projected-wall-time estimate,
  so a loop whose sizing is absent or wrong inherits no durability obligation at all.
- **Confidence (emitter):** high
- verified-at-filing:
  `grep -rniE 'per-unit progress|progress line' .claude/rules/ .claude/agents/` → **0 hits**
  (absence of the progress-line duty, repo-wide across both surfaces);
  `grep -niE 'checkpoint per phase' .claude/rules/code-style.md` → **1 hit (L53)**,
  `- **Checkpoint per phase; never accumulate-in-memory and write-at-end.** …` — the
  checkpoint clause IS present, see the context-binding note below (note the literal
  hyphenated form `checkpoint-per-phase` returns 0; the heading is spaced);
  `grep -c '~50' .claude/rules/code-style.md` → **0** (no count-based trigger);
  `grep -niE 'checkpoint|long.loop|restartab' .claude/agents/code-correctness-critic.md` →
  **3 hits (L136-137, L296), all ROUTING long-loop restartability away to
  `efficiency-critic`** — no check of its own;
  `grep -n -iE 'restartab|checkpoint' .claude/rules/lens-coverage-map.md` → **1 hit (L118)**:
  `| Step 3.6 long-loop restartability | code-reviewer.md Step 3.6 | v2-owner: efficiency-critic |`;
  `git log --oneline --since='7 days ago' -- .claude/rules/code-style.md` → 1 commit,
  `6538d31305` (detached-phase harvest contract, unrelated). (2026-07-26)

**Context binding — read before planning (this narrows the change).** The checkpoint
half of the miner's proposal is NOT missing, and the enforcement chain for it already
exists end to end; it simply did not bite:

- `.claude/rules/code-style.md` L53 § checkpoint-per-phase already carries an
  intra-phase clause binding "any serial/looped fit/solve/eval/generation stage over
  independent units … whose projected wall-time exceeds ~1h", and it names the exact
  anti-pattern (`results = []; … write(results, path)`).
- `#1689`'s own approved plan (`tasks/running/1689/plans/v3.md` L245) PROMISED the
  mechanism: "the ladder-fit phase is ≤ 4h and its interim outputs (per-pair JSONs) are
  checkpointed to `eval_results/issue_1689/ladder/` so a mid-phase kill loses ≤ 1h."
- `.claude/agents/code-reviewer.md` Step 3.6 (L1263) already requires the reviewer to
  verify per-unit persistence + a resume predicate BY READING THE LOOP, and explicitly
  names dict-accumulate as the failure.
- The shipped loop (`scripts/issue1689_fit_ladder.py::run_all_pairs`, commit
  `15906d680a`) does none of it, and passed review.

So the planner should treat the checkpoint leg as an ENFORCEMENT question (why did a
present rule + a present plan promise + a present review gate all miss a textbook
dict-accumulate?) and reserve NEW rule text for the two genuinely-absent guards: the
progress line, and a count-based trigger that needs no wall estimate.

## Evidence

- `#1689` phase `fit_ladder` ran 5 h 14 m at 0% GPU with zero log output after its
  `[phase=fit_ladder]` line, and `eval_results/issue_1689/ladder/` did not exist —
  verified on pod-1689 at 2026-07-27T06:33Z.
- Session `5c5a89e8`, 2026-07-27T01:43Z → 05:32Z: five consecutive poll ticks (#29–#33)
  plus two rounds of `/proc` forensics were spent proving the process was computing
  rather than deadlocked, because there was no progress line to read. Evidence:
  `"===ps===\n    PID     ELAPSED %CPU STAT\n  41602    03:11:51 6240 Rl"` and
  `"ls: cannot access 'eval_results/issue_1689/ladder/': No such file or directory"`
  (repeated on ticks #29–#33).
- Session `dffde9b6`, 2026-07-27T05:55:00Z: `"=== ladder outputs so far === ⏎ 0"` and
  `"run_all_pairs accumulates every pair into an in-memory dict and writes once per
  layer, after all 126 pairs x 2 arms complete ... there are zero ladder outputs on disk"`.
- Source confirmation (read at compose time, commit `15906d680a`,
  `scripts/issue1689_fit_ladder.py:687`): the loop body is
  `out["pairs"][pair_key][arm] = _run_ladder_pair(...)` with a single `return out`; no
  write and no print inside the pair loop.
- Measured cost: no work was lost only because nothing had been produced; the phase was
  un-resumable and un-observable for its whole 5 h 14 m, on a 4-GPU pod at 0% GPU.

## Proposed change

- `.claude/rules/code-style.md` § checkpoint-per-phase — add a **per-unit progress line**
  duty to the existing intra-phase clause: a loop that owes per-unit persistence also owes
  one stdout line per completed unit carrying at minimum the unit index / total, the unit
  key, and elapsed seconds (`[<phase>] unit k/N <key> elapsed=<s>s`). A phase whose only
  observable is process liveness is a wedge to every poller.
- Same clause — add a **count-based trigger** alongside the wall-time trigger: a loop over
  more than ~50 independent units trips the persistence + progress-line requirement
  REGARDLESS of projected wall-time. The wall-time trigger presupposes a sizing estimate;
  the count is readable straight off the diff.
- `.claude/agents/code-reviewer.md` Step 3.6 (the v1 owner per
  `.claude/rules/lens-coverage-map.md:118`) — extend the two-part check to three: per-unit
  persistence, resume predicate, **per-unit progress line**; and add the ~50-unit count as a
  second trigger next to the ~1h projection.
- `.claude/agents/efficiency-critic.md` implementation mode (the **v2 owner** per the same
  ledger row) — extend "long loops checkpoint" to "long loops checkpoint AND emit a per-unit
  progress line". NOTE: the miner proposed `code-correctness-critic.md`; that file
  deliberately routes long-loop restartability to `efficiency-critic` (L136-137, L296), so
  putting the check there would contradict the lens-coverage ledger.
- Record in the plan whether the enforcement miss above warrants any further change (e.g. a
  mechanical `workflow_lint` scan for accumulate-then-write-once over a named-unit loop), or
  whether the count trigger alone closes it.

## Scope / surfaces

- Primary target: `.claude/rules/code-style.md`
- `.claude/agents/code-reviewer.md` (Step 3.6, v1 owner)
- `.claude/agents/efficiency-critic.md` (implementation mode, v2 owner)
- `.claude/rules/lens-coverage-map.md` (ledger row L118, if the lens text changes)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 2b40afcd8d52

- workflow_fix_target: .claude/rules/code-style.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: G-P3, A-P5.
