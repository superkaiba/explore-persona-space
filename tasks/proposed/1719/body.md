---
title: 'daily-fix: single-flight pgrep probes self-match and are not'
kind: infra
tags:
- wf-fix
- wf-fix-fp:836ad04b03d3
- daily-auto-filed
created_at: '2026-07-27T07:15:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): Step 9c/10d single-flight
  probes match their own harness wrapper and sibling sessions'' gates, one of which
  silently skipped a required compare leg behind a green exit 0; separately an unscoped
  pkill -f targeted a fleet-wide pattern while sibling gates were live'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 4 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Make every Step 9c / Step 10d single-flight `pgrep` probe self-exclusion-safe and
exact-issue-scoped, so a probe can neither match its own harness wrapper nor a sibling
session's gate.

## Workflow gap

- **Bug observed:** Single-flight probes fired three times on 2026-07-26 against the wrong process — twice self-matching the probe's own wrapper argv (one of which printed `GATE STILL RUNNING; skip compare` and exited 0, silently skipping the Step 9c compare leg behind a green background-Bash exit code) and once matching a sibling session's Step 10d lint gate.
- **Why it is a workflow gap:** The committed `[.]`-bracket idiom protects only against the pattern TEXT appearing in the prober's own argv; it does not survive a command line that also passes the artifact PATH as a real argument, and two committed probe recipes carry an issue-agnostic `scripts/workflow_lint[.]py` alternate that matches every session's gate by construction.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'pgrep' .claude/skills/issue/SKILL.md` → 18 hits; the un-issue-scoped alternate `scripts/workflow_lint[.]py` is present at L11716 (Step 10d gate-and-land probe) and L12023 (missing-sentinel recovery probe), both carrying a WAIT-never-kill mitigation but no issue scoping. `grep -rn 'grep -v -e "bash -c"\|PPID' .claude/rules/ .claude/skills/issue/SKILL.md` → 0 hits (the self-exclusion form is absent from the workflow surface). `grep -n 'bracket' .claude/rules/gotchas.md` → the ownership-probe entry at L373 states the bracket idiom as the DEFAULT shape with no artifact-path-in-argv caveat. `grep -n 'pkill' .claude/rules/crash-fix-rounds.md` → 6 hits; the broad-pattern ban is already present at L180-183. `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md .claude/rules/gotchas.md` → no commit adding a self-exclusion form (2026-07-26)

## Evidence

- Session `2b779905`, 12:24:08Z: the Step 9c `compare` leg was launched in background with an inline `pgrep -af 'step9c-junit-issue-1699[.]xml'` guard; the launch shell's own argv contained the unbracketed `--junitxml /tmp/step9c-junit-issue-1699.xml`, so the probe matched itself. Output: `"GATE STILL RUNNING; skip compare"`, then `"FATAL: compare rc file missing"` and `"ls: cannot access '/tmp/step9c-compare-issue-1699.*': No such file or directory"`. The background Bash reported completion with exit code 0 — a false DONE with no compare output, the same class as the #825 empty-dir false DONE.
- Same session's own diagnosis: `"Compare bg-call self-matched its own argv on the probe pattern (the launch shell contains step9c-junit-issue-1699[.]xml in its argv)."`
- Session `891b2cc6`, 15:30:43Z and 15:30:53Z: a lint-gate probe using `scripts/workflow_lint[.]py` matched three sibling sessions' gates (issues 1702/1709/1711); the issue-scoped retry then self-matched on two UNbracketed alternates (`issue-1704-lint-verdict`, `issue-1704-surgical-outcome`) present verbatim in its own wrapper — `"Self-match false positive (the probe's own shell wrapper contains the pattern)."`
- Session `c0319d9e`, 13:37:54Z: the pre-compare probe `pgrep -af 'step9c-junit-issue-1701[.]xml|scripts/workflow_lint[.]py'` matched a Step 10d lint gate running in `.claude/worktrees/issue-1698`. The session reasoned its way out — `"Other sessions' step10d gates are running (not mine). Launching the step 1d compare."` — one wasted reasoning step, with the standing risk that a less careful session waits on, or reads the verdict file of, a foreign gate.
- Session `5c5a89e8`, 05:31:51Z (pod-side liveness probe, same failure family): an UNbracketed `pgrep -af 'issue1689'` self-matched its own SSH command line, and `LADDER_PIDS=$(pgrep -f 'issue1689_fit_ladder.py' | head -3); ps -p $LADDER_PIDS` produced `"error: process ID list syntax error"` — one wasted SSH round-trip inside a four-hour hang diagnosis. The working retry iterated per pid.
- Session `0e2c3b21`, 09:27:12Z (kill side): an implementer cancelling one redundant background job of its own ran `pkill -f 'workflow_lint.py'` with no scoping while it had already observed concurrent gates from issues 1694, 1696 and 1697 on the box (exit 144). No sibling is known to have been killed; the blast radius was fleet-wide by construction. The same subagent scoped its pytest kill correctly 56 minutes later (`grep -v issue-1694 | grep -v issue-1696 | grep -v issue-1697`).
- Measured cost: roughly 7 wasted tool calls across `2b779905` and `891b2cc6` plus a compare relaunch; the false-DONE skip is the material harm, since it would have read as PASS had the rc-file guard not been present.

## Proposed change

- In `.claude/rules/gotchas.md` (SSH-remote / ownership-probe entry, L373): state that bracketing protects only the pattern text, and FAILS whenever the same command line also passes the artifact path as an argument or uses multiple alternates. Add the self-exclusion form as the recipe for those cases — pipe the probe through a filter that drops the prober's own wrapper (`| grep -v -e "bash -c" -e "^$$ "`) — and add the multi-pid iteration form (`for p in $(pgrep -f 'patter[n]'); do ps -p "$p" -o pid,etime,pcpu,stat; done`), never `ps -p $LIST` on an unquoted multi-line capture.
- In `.claude/skills/issue/SKILL.md`, issue-scope both un-scoped alternates: L11716 and L12023 currently match `scripts/workflow_lint[.]py` fleet-wide. Replace with an issue-keyed pattern (the existing `issue-<N>-lint-gate-tre[e]` shape used at L10309 / L10683), and keep the WAIT-never-kill disposition for any residual ambiguous hit.
- In `.claude/skills/issue/SKILL.md` Step 9c, extend the existing "SEPARATE FOREGROUND call — never inside the launch call itself" statement (L8875, landed 2026-07-23 in `1f98e924dc0d3c1de34ce91bec70cd29024f605a`) with the observed consequence: a self-match inside a background launch turns into a silent exit-0 skip of the leg, which the harness reports as a successful completion. The rule exists; the 2026-07-26 firing shows it needs the failure mode named to bind.
- Add the artifact-path caveat wherever the bracket idiom is cited as sufficient — `.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch step 1 references it as the DEFAULT probe shape.
- Kill side: the broad-pattern `pkill` ban is ALREADY LANDED at `.claude/rules/crash-fix-rounds.md` L180-183 (`pkill -f python`, `pkill -f run_`, `pkill -f uv` named as BANNED, "kill by explicit PID from the listing instead of pkill"). Do NOT re-add it. The residual gap is applicability scope: that section's opening line binds it to "EVERY re-run of a smoke / launch / dispatch command", and the 2026-07-26 firing was a session cancelling its OWN redundant background job — not a re-run. Widen the applicability sentence to cover any kill targeting a workload pattern, whatever the reason.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- `.claude/rules/gotchas.md` (ownership-probe entry, L373), `.claude/rules/crash-fix-rounds.md` (§ Kill-before-relaunch applicability line)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 836ad04b03d3

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: F-P3, C-P8, A-P12, H-P7.
