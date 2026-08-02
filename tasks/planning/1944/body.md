---
title: 'workflow-fix: Step 10d lint gate own-diff attribution matches message-cited
  paths (false block)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-31T19:57:11Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the /issue 1768 orchestrator (Step
  10d r4 merge round, 2026-07-31): the gate''s own-diff attribution grep false-attributed
  a pre-existing foreign-file lint failure to the payload because the failure MESSAGE
  cites .claude/rules/gotchas.md (in the branch own-diff via spec-freshness sync);
  fix = path-token attribution at both lint-attribution sites'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1768 (emitting agent: orchestrator, /issue 1768 session, Step
10d merge round for the r4 follow-up).

## Goal

Fix the Step 10d pre-push workflow-lint gate's own-diff attribution grep so a
lint MESSAGE that cites an own-diff path cannot false-attribute a foreign
file's pre-existing failure to the merge payload (false `block`).

## Workflow gap

- **Bug observed:** the gate recipe's attribution step
  `grep -F -f /tmp/issue-<N>-own-diff.txt /tmp/issue-<N>-lint-gated-norm.txt`
  matches own-diff paths ANYWHERE in the normalized failure line. On issue
  #1768's Step 10d run (2026-07-31 ~19:40Z), the single gated failure line was
  a PRE-EXISTING main red on a FOREIGN file
  (`scripts/issue1689_user_slot_capture.py:: jsonl-splitlines: ...`), but its
  message text cites `.claude/rules/gotchas.md` — which IS in #1768's own-diff
  (a spec-freshness sync import) — so the line landed in lint-owndiff.txt and
  would have produced a FALSE `block` verdict (the run happened to return
  `crash` on an unrelated TG-leg kill, which is how it surfaced).
- **Why it is a workflow gap:** the SKILL.md verdict bullet's intent is
  "Gated failure lines NAMING a file IN the own-diff [as the offender]"; the
  fixed-string whole-line grep implements a strictly weaker predicate that
  false-positives on any lint message citing a rules/docs path — and lint
  messages routinely cite `.claude/rules/*.md` (the jsonl-splitlines message
  does). Any branch whose own diff includes a commonly-cited rules file (every
  spec-freshness-synced branch) is exposed.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'grep -F -f /tmp/issue-<N>-own-diff.txt /tmp/issue-<N>-lint-gated-norm.txt' .claude/skills/issue/SKILL.md` → 1 hit; the surgical-block twin `grep -F -f /tmp/issue-<N>-additive-files.txt /tmp/issue-<N>-lint-gated-norm.txt` → 1 hit (2026-07-31). Both sites carry the same defect; the TG-leg file-grain attribution (`grep -F -f /tmp/issue-<N>-tg-files.txt`) has the same shape but its inputs are pytest output lines where message-cited rules paths are rarer — the plan should judge whether to fix it too.

## Proposed change (candidate diff sketch — refine in planning)

Replace the whole-line fixed-string grep at BOTH lint-attribution sites with a
path-token match: extract the offender path token (the text between the
`workflow_lint: ` prefix and the first `:` run, gate-tree prefix
`/tmp/issue-<N>-lint-gate-tree/` stripped) and test set-membership against the
own-diff / additive-files list. Worked awk implementation (used live on
#1768's re-run):

```
awk -v OWN=/tmp/issue-<N>-own-diff.txt '
  BEGIN { while ((getline l < OWN) > 0) own[l]=1 }
  /^workflow_lint: / {
    s = substr($0, 16); n = index(s, ":")
    path = (n > 0) ? substr(s, 1, n-1) : s
    sub(/^\/tmp\/issue-<N>-lint-gate-tree\//, "", path)
    gsub(/^[ \t]+|[ \t]+$/, "", path)
    if (path in own) print $0
  }' /tmp/issue-<N>-lint-gated-norm.txt > /tmp/issue-<N>-lint-owndiff.txt || true
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'lint-gated-norm.txt' .claude/ CLAUDE.md scripts/`) and update
  every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (computed at filing by file_infra_task.py / wf_fix_fingerprint)

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Step 10d lint gate's own-diff attribution grep (grep -F -f own-diff.txt lint-gated-norm.txt) matched a PRE-EXISTING foreign-file failure line because its lint MESSAGE cites .claude/rules/gotchas.md, which was in the branch own-diff via a spec-freshness sync — a false `block` verdict shape (surfaced on #1768, 2026-07-31).
why_workflow_gap: The recipe implements "line contains any own-diff path anywhere" where the documented intent is "line NAMES an own-diff file as the offender"; lint messages routinely cite rules paths, and synced rules files are in most branches' own-diffs.
proposed_change: Path-token attribution at both lint-attribution sites (own-diff + additive-files): extract the offender path between the `workflow_lint: ` prefix and the first `:` run (gate-tree prefix stripped) and set-match against the payload list.
diff_sketch: |
  - grep -F -f /tmp/issue-<N>-own-diff.txt /tmp/issue-<N>-lint-gated-norm.txt \
  -   > /tmp/issue-<N>-lint-owndiff.txt || true
  + awk -v OWN=/tmp/issue-<N>-own-diff.txt 'BEGIN{while((getline l<OWN)>0)own[l]=1}
  +   /^workflow_lint: /{s=substr($0,16);n=index(s,":");path=(n>0)?substr(s,1,n-1):s;
  +   sub(/^\/tmp\/issue-<N>-lint-gate-tree\//,"",path);if(path in own)print}' \
  +   /tmp/issue-<N>-lint-gated-norm.txt > /tmp/issue-<N>-lint-owndiff.txt || true
confidence: high
related_task: #1768
<!-- /workflow-fix-candidate -->
