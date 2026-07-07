---
title: 'workflow-fix: check 27 also FAILs prior-issue markdown links in v4 standalone
  sections'
kind: infra
tags:
- wf-fix
- wf-fix-fp:39bfcd1d6e57
created_at: '2026-07-04T08:59:04Z'
has_clean_result: false
origin_prompt: 'clean-result-critic workflow-fix-candidate on #928 (2026-07-04): extend
  check_v4_no_bare_issue_refs to markdown links targeting eps.superkaiba.com/tasks/N
  in Takeaways/Methodology/Results spans'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #928 (emitting agent: clean-result-critic).

## Goal

Extend verify_task_body.py check 27 to also FAIL prior-issue markdown links (targets matching `eps.superkaiba.com/tasks/\d+`) inside the Takeaways/Methodology/Results section spans, keeping the existing sanction list (GFM table rows — the Training-table Source column — fenced/inline code, HTML comments, footer, frontmatter).

## Workflow gap

- **Bug observed:** Check 27 (`check_v4_no_bare_issue_refs`) FAILs only bare `#K` tokens in Takeaways/Methodology/Results; a `[#K](https://eps.superkaiba.com/tasks/K)` markdown link in `## Methodology` prose (banned by SPEC.md § `## Goal` (v4) "the ONLY place... that may cite prior tasks") sails through mechanically and reached the round-1 clean-result critique on #928.
- **Why it is a workflow gap:** The substantive rule bans BOTH forms in the standalone sections, but only the bare-token half was mechanized after #841; the markdown-link half is left to the LM lens, which historically under-applies spec text.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In check_v4_no_bare_issue_refs, alongside the bare-token regex over the stripped section spans:
+ link_re = re.compile(r"\[[^\]]*\]\(https?://eps\.superkaiba\.com/tasks/\d+[^)]*\)")
+ apply to the same sanitized spans (table rows / code / comments already stripped), FAIL with "prior-issue links live only in the ## Goal context slot + footer; Training-table Source column exempt".
Update SPEC.md check-27 text + add a pin in tests/test_verify_task_body.py.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep hits to keep in sync: `.claude/skills/clean-results/SPEC.md` (check-27 text), `tests/test_verify_task_body.py` (pin).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SPEC.md stays consistent with the verifier.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 39bfcd1d6e57

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: Check 27 (`check_v4_no_bare_issue_refs`) FAILs only bare `#K` tokens in Takeaways/Methodology/Results; a `[#K](https://eps.superkaiba.com/tasks/K)` markdown link in `## Methodology` prose (banned by SPEC.md § `## Goal` (v4) "the ONLY place... that may cite prior tasks") sails through mechanically and reached this critique round on #928.
why_workflow_gap: The substantive rule bans BOTH forms in the standalone sections, but only the bare-token half was mechanized after #841; the markdown-link half is left to the LM lens, which historically under-applies spec text (reconciler feedback file).
proposed_change: Extend check 27 to also FAIL markdown links whose target matches `eps.superkaiba.com/tasks/\d+` inside the Takeaways/Methodology/Results section spans, keeping the existing sanction list (GFM table rows — the Training-table Source column — fenced/inline code, HTML comments, footer, frontmatter).
confidence: high
related_task: #928
<!-- /workflow-fix-candidate -->
