---
title: 'Ownership-probe recipe: cwd match is not ownership — a worktree-cwd sweep
  returns other sessions'' transient subprocesses'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T09:56:01Z'
has_clean_result: false
origin_prompt: 'Near-miss during #2327 Step 9c launch: after killing its own non-canonical
  run, the orchestrator swept for strays by /proc/<pid>/cwd match against its worktree;
  three sweeps each returned a different pid and an ancestry probe found none — they
  were other sessions'' transient uv-run-pytest subprocesses iterating .claude/worktrees/*.
  One was killed by PID before the ancestry probe ran. Correct predicate is ancestry/pgid
  (pgrep -g <setsid leader pgid>), not cwd.'
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
---

# The ownership-probe recipe needs an explicit anti-pattern: a cwd-match sweep for "strays" returns other sessions' transient subprocesses

## Goal

Make CLAUDE.md's ownership-check recipe say, in one line, that **process ownership is established by ancestry or process-group membership, never by a `/proc/<pid>/cwd` match against your worktree** — and that a cwd-keyed sweep on this repo reliably produces false positives, because ~15 concurrent sessions run test suites that iterate `.claude/worktrees/*` and spawn short-lived subprocesses with cwd set inside OTHER sessions' trees.

## The defect

CLAUDE.md § "Orchestrator vs subagent re-invocation" prescribes the ownership probe as `pgrep -af '<distinctive invocatio[n]>'` — bracket one character so the probe never matches its own command line. That is correct and it is what the rule says. What the rule does NOT say is which process ATTRIBUTES constitute ownership evidence. The natural-seeming variant — sweep for processes whose cwd points into my worktree, treat those as mine — is wrong here, and wrong in the dangerous direction: it produces a kill list containing other sessions' work.

There is no coverage for this. `.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch governs retrying your OWN failed launch; `.claude/rules/gotchas.md` carries the SSH-remote ownership-probe entry (the bracket-a-character self-match trap); the CLAUDE.md pre-stop and post-death probes (#2111) govern subagent children and detached children. None of them says "cwd is not ownership".

## Observed instance (#2327 Step 9c launch, 2026-08-24)

The #2327 orchestrator killed a non-canonical Step 9c launch of its own (correctly, by captured PID) and then swept for leftover pytest processes in its worktree using a `/proc/<pid>/cwd` match. Three consecutive sweeps each reported a DIFFERENT pid. An ancestry probe run afterwards found NONE — every hit had already exited. At that moment 14 pytest processes were live on the box, nearly all belonging to other sessions.

The hits were transient `uv run pytest` / lint subprocesses spawned by another session's suite as it iterated `.claude/worktrees/*`, each living a few seconds. **The orchestrator killed one of them (`3651201`) by PID before running the ancestry probe.** Its cwd already read empty, so it was very likely exiting regardless and no damage was observed — but the reasoning was unsound: a cwd match was taken as ownership evidence when it never was. Under slightly different timing the same sweep would have killed a live subprocess belonging to a sibling session, which is precisely the class of harm the ownership-check rule exists to prevent.

The correct probe, which the same session then used to verify its relaunch, is process-group membership against its OWN captured leader:

```bash
P=$(cat /tmp/<phase>.pid)
pgrep -g "$(ps -o pgid= -p "$P" | tr -d ' ')"
```

`setsid` puts the launched phase in its own process group (`pgid == pid` for the leader), so pgid membership is exact ownership and is immune to both cwd coincidence and pid reuse across sweeps.

## Scope to investigate

1. Add the anti-pattern line to CLAUDE.md § "Orchestrator vs subagent re-invocation" beside the existing bracket-a-character guidance: cwd is not ownership; use ancestry or pgid; with `setsid` phases, `pgrep -g <leader pgid>` is exact.
2. State the corollary that makes it bite: on this repo a cwd sweep is not merely weak evidence but actively misleading, because sibling sessions' suites walk `.claude/worktrees/*` and legitimately run with cwd inside other trees.
3. Consider whether the detached-phase launch recipe (`.claude/skills/issue/steps/13-step-9.md:178`) should record the leader's PGID alongside the pid in its breadcrumb set, so a successor session inherits an exact ownership handle rather than having to re-derive one. Today the breadcrumb contract is pid + log + sentinel; pgid is strictly more useful for the probe and costs one `ps` call at launch.
4. Check whether any shipped helper or documented recipe performs a cwd-keyed process sweep today (as opposed to prose telling an agent to do so), and fix those call sites if so.

## Non-goals

Do not weaken the existing bracket-a-character self-match guidance — it is correct and orthogonal. Do not add a blanket prohibition on reading `/proc/<pid>/cwd`: cwd is useful DIAGNOSTIC context once ownership is already established (it is how the #2327 session confirmed the other live 213-file suite was in the `issue-2315` worktree and therefore not a collision). The defect is using cwd as the ownership PREDICATE, not reading it at all.

## Provenance

Surfaced by the #2327 orchestrator while launching its own Step 9c gate; the near-miss and the corrected pgid probe are both recorded in that task's Step 9c launch `epm:progress` marker. Confidence: high — the false-positive pattern was observed three times in a row and then falsified by an ancestry probe in the same turn. Dedup target: CLAUDE.md § "Orchestrator vs subagent re-invocation" ownership-check bullet, distinct from #2327's own target surface and from the #2523 concern-row grammar filing.
