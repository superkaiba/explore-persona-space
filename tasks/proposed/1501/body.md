---
title: 'daily-fix: repo-root guard heredoc payload false positive'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5c4c4e048582
- daily-auto-filed
created_at: '2026-07-18T06:47:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): guard_repo_root_branch.sh
  blocks Bash commands whose heredoc DOC-TEXT payload merely mentions a fenced git
  verb (07-16 /daily body-composition block; reproduced live at compose time) — payload-blind
  verb scan kills whole compound commands.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined problem hit by the 2026-07-16 /daily run itself (chunk-4 miner) and REPRODUCED LIVE by tonight's run: `scripts/guard_repo_root_branch.sh` blocks Bash commands whose HEREDOC/string PAYLOAD merely mentions a fenced git verb, even when no git command is being executed.

## Goal

Mask heredoc payloads (and equivalent quoted document bodies) out of the fenced-verb scan in `scripts/guard_repo_root_branch.sh`, so composing a document that MENTIONS a fenced git verb is not blocked, while keeping every real invocation path blocked.

## Workflow gap

- **Bug observed:** the 07-16 /daily session was blocked composing a filing body via heredoc because the body TEXT contained a fenced verb phrase; tonight's run reproduced it: a `cat > /tmp/x.md <<EOF` heredoc whose doc-text line mentioned the merge verb was BLOCKED (PreToolUse deny, full compound killed). The guard is deliberately path-blind, but it is also payload-blind: verb literals inside a heredoc body are scanned as if they were command position.
- **Why it is a workflow gap:** /daily, workflow-fix sessions, and reviewers routinely compose bodies/reports that must NAME git verbs (this repo's own incident documentation does); the false positive kills whole compound commands (the #813/#1056 partial-state class) and pushes agents toward writing about the verbs in evasive language.
- **Confidence:** high (live reproduction at compose time).
- verified-at-filing: live semantic probe 2026-07-18 UTC — piping a synthetic PreToolUse JSON (`cat > /tmp/x.md <<EOF` with a doc-text line naming the merge verb) into `scripts/guard_repo_root_branch.sh` produced the BLOCK message (predicate executed against the claimed text, clause (a') satisfied); `grep -n "heredoc" scripts/guard_repo_root_branch.sh` → 0 hits (no heredoc masking exists); `git log --oneline --since='7 days ago' -- scripts/guard_repo_root_branch.sh` shows recent waiver work (#1413 ssh payloads, #1463 gcloud payloads) but no heredoc-payload mask.

## Proposed change (candidate diff sketch — refine in planning)

Before the fenced-verb scan, strip heredoc bodies from the command text (from `<<TAG`/`<<'TAG'` to the terminator line), analogous to the existing single-quoted ssh/gcloud payload waivers — conservative: mask ONLY when the heredoc target is a non-git sink (e.g. `cat >`, `tee`, a file path), never when the heredoc feeds a shell/interpreter (`bash <<`, `sh <<`, `python <<` keep full scanning). Add fixture tests for both directions (doc-composition allowed; `bash <<EOF` with a fenced verb still blocked).

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- The guard has an existing waiver architecture (#1413/#1463) — follow its shape and its test file.

## Constraints / invariants

- The guard's fail-closed posture is the invariant: any ambiguity in heredoc parsing must fall back to BLOCK. This is a security-posture-sensitive change — full pipeline review required (which is why it is route 2, not self-applied).
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 5c4c4e048582

- workflow_fix_target: scripts/guard_repo_root_branch.sh

source: /daily 2026-07-17 transcript sweep (chunk-4 miner: the 07-16 /daily block) + tonight's live reproduction during candidate verification.
