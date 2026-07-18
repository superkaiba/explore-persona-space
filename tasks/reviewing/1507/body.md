---
title: 'workflow-fix: check 8b body-wide git-URL resolution + git-tree backtick-claims
  twin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:187536ec8bc5
created_at: '2026-07-18T07:13:04Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1072 round-2 workflow-fix-candidate: 404 GitHub
  blob links in Methodology passed the verifier (check 8b footer-only; backtick claims
  HF-only)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1072 round 2 (emitting agent: clean-result-critic).

## Goal

Extend verify_task_body.py check 8b to resolve every same-repo /blob/<sha> and /tree/<sha> URL body-wide (not footer-only), and add a GitHub-tree twin of the HF-adjacent backtick-file-claims check.

## Workflow gap

- **Bug observed:** Task #1072's body carried two GitHub blob links (Methodology Sample slot) and a footer backtick file list (npz files "in" a git tree URL) that 404 / are absent from the pinned git tree, yet verify_task_body.py passed — check 8b resolves same-repo artifact URLs only in the footer/Reproducibility section (6 URLs counted), and the backtick-file-claims-exist check runs only against HF /tree links, not GitHub /tree links.
- **Why it is a workflow gap:** The verifier's artifact-existence coverage is asymmetric (footer-only for git URLs, HF-only for backtick claims), so a false git-artifact premise in Methodology or the Artifacts parenthetical sails through the mechanical gate and must be caught by hand.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "check 8b" scripts/verify_task_body.py` → 8 hits; scripts/verify_task_body.py:6050 scopes check 8b to "`## Reproducibility` section text", and :3214/:3251 confirm other checks defer existence probing to 8b — footer-only scope confirmed per-target (2026-07-18). Live repro: two Methodology blob links to per_context_stats_1072_fold4.npz 404'd (npz gitignored, never committed) while verify_task_body.py PASSed #1072.

## Proposed change (candidate diff sketch — refine in planning)

- urls = _same_repo_urls(footer_text)
+ urls = _same_repo_urls(body_text)          # body-wide, not footer-only
+ # new check: backtick claims adjacent to GitHub /tree/<sha> links
+ for sha, dirpath, claims in _git_tree_adjacent_backtick_claims(body_text):
+     tree = git_ls_tree(sha, dirpath)
+     missing = [c for c in claims if not _claim_in_tree(c, tree)]
+     if missing: fail(f"backtick file claims not in git tree {dirpath}@{sha}: {missing}")

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard: this session carries a `workflow_fix_target:` Provenance line.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 187536ec8bc5

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: Task #1072's body carried two GitHub blob links (Methodology Sample slot) and a footer backtick file list (npz files "in" a git tree URL) that 404 / are absent from the pinned git tree, yet verify_task_body.py passed — check 8b resolves same-repo artifact URLs only in the footer/Reproducibility section (6 URLs counted), and the backtick-file-claims-exist check runs only against HF /tree links, not GitHub /tree links.
why_workflow_gap: The verifier's artifact-existence coverage is asymmetric (footer-only for git URLs, HF-only for backtick claims), so a false git-artifact premise in Methodology or the Artifacts parenthetical sails through the mechanical gate and must be caught by hand.
proposed_change: Extend check 8b to git-cat-file every same-repo /blob/<sha> and /tree/<sha> URL anywhere in the body, and add a GitHub-tree twin of the HF-adjacent backtick-file-claims check (claims adjacent to a git tree URL must resolve via git ls-tree at that SHA).
confidence: high
related_task: #1072
<!-- /workflow-fix-candidate -->
