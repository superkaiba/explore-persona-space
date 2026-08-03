---
title: 'workflow-fix: round-unique inline lint-gate payload path'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f915ade0ceb6
- trigger-dense
created_at: '2026-07-31T22:38:45Z'
has_clean_result: false
origin_prompt: 'Surfaced prose follow-up from inline subagent model-text-2x2-1768
  on #1768: issue-keyed /tmp/issue-<N>-inline-payload.txt clobbered by concurrent
  same-issue inline rounds at 22:26Z -> false gate certification; fix = round-unique
  suffix + certification bound to payload sha'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by an inline-round subagent on task #1768 (emitting agent: experiment-implementer `model-text-2x2-1768`, 2026-07-31 ~22:3xZ).

## Goal

Make the inline lint gate's payload path round-unique (and bind each gate certification to the exact payload it read) so concurrent inline rounds on the SAME issue cannot clobber each other's payload file and cross-certify.

## Workflow gap

- **Bug observed:** two concurrent inline rounds on issue 1768 clobbered /tmp/issue-1768-inline-payload.txt causing a gate to certify paths its caller never submitted
- **Why it is a workflow gap:** the documented payload path is ISSUE-keyed only (`/tmp/issue-<N>-inline-payload.txt`) in both the gate's usage doc and the recipe the commit-guard hook prints, while the workflow explicitly supports multiple concurrent inline rounds per issue (three ran on #1768 tonight); a shared mutable path between payload-write and gate-read is a TOCTOU that produces FALSE CERTIFICATION, not just a wasted run.
- **Confidence (emitter):** high (mechanism grep-verified; the specific 22:26Z collision timeline is as reported by the emitting agent — `unverified hypothesis — verify at plan time:` the exact interleaving of the two gate launches; the emitting round re-ran its gate against a unique path either way)
- verified-at-filing: `grep -rn "inline-payload" scripts/ .claude/ CLAUDE.md` → 12 hits in 5 files (2026-07-31); per-target: `scripts/inline_lint_gate.py` 2 hits — L45 the documented issue-keyed `--payload-file` path (the bug site), L539 an INTERNAL `tempfile.mkstemp` copy (context read: the gate copies the payload at gate start, which does NOT close the caller-side clobber window before the copy); `.claude/hooks/guard_root_code_commit.sh` 2 hits — L1296/L1298, the printed recipe callers follow (the propagation site). Remaining hits: `scripts/select_step9c_tests.py` (prose references to the gate, not path constructions — no change needed) and stale worktree mirrors under `.claude/worktrees/issue-779/` (not targets). Main-tree `.claude/skills/issue/SKILL.md` has no literal hit (the recipe reaches callers via the hook's printed guidance).

## Proposed change (candidate diff sketch — refine in planning)

diff_sketch: |
  scripts/inline_lint_gate.py (usage doc + arg validation):
  - --payload-file /tmp/issue-<N>-inline-payload.txt
  + --payload-file /tmp/issue-<N>-<round_slug>-inline-payload.txt   # round-unique REQUIRED;
  +   the gate REFUSES the bare issue-keyed legacy path (fail loud), or grows a
  +   --round-slug arg and derives the unique path itself
  + gate verdict line additionally stamps sha256(payload contents) + the resolved path,
  +   so a certification is mechanically bound to the exact path set it read
  .claude/hooks/guard_root_code_commit.sh L1296–1298 (printed recipe):
  - printf '%s\n' <paths> > /tmp/issue-<N>-inline-payload.txt
  + printf '%s\n' <paths> > /tmp/issue-<N>-<round-slug>-inline-payload.txt

## Scope / surfaces

- Primary targets: `scripts/inline_lint_gate.py`, `.claude/hooks/guard_root_code_commit.sh`
- Grep the workflow surface for the pattern before editing (`grep -rn 'inline-payload' scripts/ .claude/ CLAUDE.md`) and update every main-tree hit; worktree mirrors self-heal on their next sync; check whether any SKILL.md section documents the recipe in other words.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; the hook's printed recipe and the gate's usage doc stay consistent with each other.
- Backward compatibility decision for in-flight rounds (refuse-legacy-path vs warn-once) is the planner's call; refusing is preferred (fail loud beats silent cross-certification).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/inline_lint_gate.py,.claude/hooks/guard_root_code_commit.sh
- fingerprint: f915ade0ceb6

Verbatim surfaced prose (agent report, 2026-07-31):

> `scripts/inline_lint_gate.py --payload-file /tmp/issue-<N>-inline-payload.txt` uses an ISSUE-keyed path, so two concurrent inline rounds on the SAME issue clobber each other. At 22:26Z I wrote my payload (scripts/issue1768_model_text_2x2.py + tests/test_issue1768_model_text_2x2.py) to /tmp/issue-1768-inline-payload.txt and launched the gate; a sibling #1768 round then overwrote that same file with scripts/issue1768_ckpt_dynamics.py + tests/test_issue1768_ckpt_dynamics.py and launched its own gate. Two gate processes are now running against one file, so at least one is certifying paths its caller never submitted (a false certification, not just a wasted run). [...] the fix is a session/round-unique suffix in the SKILL.md Step 9a-ter recipe (e.g. /tmp/issue-<N>-<slug>-inline-payload.txt) — I'm not filing it myself since I'm mid-round.
