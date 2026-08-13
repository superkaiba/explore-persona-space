---
title: step9c_baseline single-flight probe is blind to non-lint-leg gate phases (TG
  legs) — two lint-gate instances can race the shared /tmp namespace
kind: infra
tags:
- workflow-fix
created_at: '2026-08-13T06:17:14Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the #2251 /issue session per workflow-fix-on-bug: the
  Step 10d single-flight probe read CLEAR while a predecessor gate instance was live
  in its TG phase; the race produced a void interim block verdict (see #2251 epm:merged
  RACE NOTE).'
workflow: v1
---
## Goal

Close the single-flight-probe blind window in the Step 10d / Step 9c 1b lint-gate launch recipes: `step9c_baseline.py probe --pattern 'issue-<N>-lint-gate-tree'` matches only processes whose argv carries the GATE-TREE path (the lint legs), so a live gate instance in any other phase — fetch/archive/overlay, and especially the TG pytest legs, which run 25+ min under fleet load — reads CLEAR, licensing a second concurrent launch that races the first on the shared `/tmp/issue-<N>-*` file namespace.

## Incident (2026-08-13, task #2251 Step 10d)

- A predecessor-session gate instance was live in its TG phase when the successor session's launch probe read CLEAR (exit 0) at 04:53Z; the successor launched a second instance at 04:54:51Z.
- The two instances shared every `/tmp/issue-2251-*` path. The stale instance's tail executed at 05:17:26Z, consuming MIXED-GENERATION files (its own stale `tg-gated.txt` + the successor's in-progress `tg-baseline.txt`, whose hits file was 0 bytes) and wrote a VOID interim `block` verdict + rc file, and its `cat` output landed inside the successor's log (identical fd offsets — same script, same early echoes truncated at successor launch).
- Fail direction was benign ONLY by interleaving luck: the authoritative run truncate-rewrote `pass` at 06:07:42Z. The reverse order (stale tail writes LAST) would have left a wrong SHA-bound verdict for the merge conditional — a false `block` (wasted gate re-runs, the #1739-class wall) or, worse, a stale `pass` re-bound over a payload the authoritative run never certified.
- Full forensics: task #2251 events.jsonl v11 heartbeat marker + the epm:merged v1 note (RACE NOTE bullet).

## Fix direction (implementer to refine)

1. Widen the single-flight probe so a live gate is visible in EVERY phase — e.g. probe the detached script path (`issue-<N>-lint-gate[.]sh` argv, which persists for the whole unit) in ADDITION to the gate-tree pattern, or key on a launch-time pid breadcrumb (`/tmp/issue-<N>-lint-gate.pid` + `kill -0`) the recipe already has the shape for (pid is echoed in the launcher breadcrumb but not persisted).
2. Mirror the same widening at the Step 9c 1b sibling probe if it shares the pattern.
3. Consider a completion-read hardening: require verdict mtime > this run's launch timestamp (the #779 stale-existing-file discipline) so a stale instance's verdict can never be read as the fresh run's.
4. Update `.claude/skills/issue/SKILL.md` (Step 10d gate subsection + Step 9c 1b) and `scripts/step9c_baseline.py` together; the probe's self-/ancestor-exclusion must keep working for the new pattern(s).

## Provenance

Surfaced during task #2251's Step 10d merge (autonomous session, 2026-08-13). Zero GPU cost.
