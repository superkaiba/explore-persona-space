---
title: main's .claude/skills/issue/SKILL.md exceeds its own size ratchet (936,191
  > 935,400 bytes) — trim below cap; blocks lint gates fleet-wide
kind: infra
tags:
- wf-fix
created_at: '2026-08-13T06:37:23Z'
has_clean_result: false
origin_prompt: 'issue #2221 Step 10d gate run 3 (2026-08-13): sole residual blocker
  was main-side SKILL.md over its grandfather ratchet; TG legs fully green'
workflow: v1
---
# main's .claude/skills/issue/SKILL.md exceeds its own size ratchet — trim below cap

## Goal

`origin/main:.claude/skills/issue/SKILL.md` is 936,191 bytes against its 935,400-byte grandfather ratchet cap (verified 2026-08-13 via `git cat-file -s`). The over-cap copy makes the pre-push workflow-lint gate's landing-union read the file as an over-cap NEW line for any session whose baseline archive predates the over-cap commit — issue #2221's Step 10d merge blocked on exactly this line as its SOLE residual (gate run 3, 2026-08-13T06:35Z: TG mapped-test legs fully green, 6884 passed / 0 failed; the branch's own SKILL.md copy is 934,662 bytes, under cap). Sessions with newer baselines see it subtract as pre-existing red instead, which owes the #1713 urgent-park emission — either way, main is standing over its own ratchet.

Fix: trim SKILL.md below 935,400 bytes on main (the workflow-compaction levers: dedupe repeated incident prose, pointer-load a span into a section-reference rule per the established `*-section-reference.md` pattern), or — only with explicit justification — bump the grandfather ratchet. Identify the commit(s) that pushed it over cap and check whether they bypassed the gate or won a moving-main race (#1721/#1719 class); if a race, consider whether the gate should hard-block a push that takes the file over cap server-side.

Acceptance: `git cat-file -s origin/main:.claude/skills/issue/SKILL.md` ≤ 935,400; no-flags `workflow_lint.py` clean on the trimmed file; a blocked Step 10d re-entry (e.g. #2221's) passes the SKILL.md line.

## Provenance

Surfaced by the #2221 Step 10d merge agent (gate runs 1-3, `epm:merge-failed v1` marker versions 1-2 on task #2221); routed by the #2221 orchestrator per `.claude/rules/workflow-fix-on-bug.md`. Target file: `.claude/skills/issue/SKILL.md` (main copy) + possibly `scripts/workflow_lint.py` ratchet table. Candidate fingerprint: skill-md-size-ratchet-breach-main-2026-08-13.
