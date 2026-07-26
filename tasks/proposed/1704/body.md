---
title: 'daily-fix: reconcile crash bundles against git'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68c1c042b809
- daily-auto-filed
created_at: '2026-07-26T07:07:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The #1345 base-model story
  leg was executed and lost at the last step, existing only inside a GCP crash bundle
  rather than in git, which stranded every base-model row of the framing-chain coverage
  matrix and was discovered only by chance while building an unrelated table.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. Paid GPU compute produced a
result that was persisted only into a crash bundle and never landed in git; nothing
noticed, and the loss surfaced by chance while building an unrelated table.

## Goal

Add a periodic audit that reconciles `issue<N>_partial/` crash-bundle contents on the
HF data repo against the committed `eval_results/issue_<N>/` tree in git, and escalates
any bundle carrying a completed result that never landed.

## Workflow gap

- **Bug observed:** #1345's base-model story leg was executed and lost at the last
  step. Session `63122023` @ 2026-07-25T18:01:02Z, building the framing-chain coverage
  matrix, found it: *"the base-model story leg turns out to have been **run and lost at
  the last step** — it's sitting in a crash bundle, not in git."* Every chat↔story row
  for the base model read "base **stranded**" in the coverage table for the rest of the
  day. It blocks the base half of the framing chain, and the compute was already paid.
- **Why it is a workflow gap:** the GCP crash-persist path
  (`_eps_persist_diagnostics`, CLAUDE.md § Compute backends) is working as designed —
  on a nonzero-rc exit it uploads the workload log and the partial artifacts to
  `issue<N>_partial/<attempt_id>/` precisely so a crash is recoverable. What is missing
  is the other half: **nothing ever reads those bundles back**. A crash bundle that
  happens to contain a COMPLETED result is indistinguishable, from the outside, from
  one containing a genuinely partial one — and the run's own upload path never fired,
  so no `eval_results/` entry exists to contradict it. The safety net catches the
  artifact and then no one looks in the net.
- **This is the "built-but-stranded" family** the project already names in
  `.claude/rules/workflow-fix-on-bug.md` ("a documented — or even fully BUILT — fix
  that is not merged … does NOT help"), applied to RESULTS rather than code.
- **Confidence (emitter):** high that the leg is stranded (the session verified it
  while building the matrix); medium on the audit's exact shape.
- verified-at-filing: absence confirmed —
  `grep -c 'issue.*_partial\|crash.bundle\|crash_report' scripts/autonomous_session_watch.py`
  → **0** (no pass reads the bundles); a repo-wide
  `grep -rln 'issue.*_partial' scripts/*.py` returns only per-issue analysis scripts
  (`i549_build_audit_table.py`, `i460_phase4_merge.py`, `issue1024_…`, `issue1310_…`,
  `issue1074_…`), i.e. ad-hoc consumers, no reconciler. `task.py view 1345 --json` →
  `status: awaiting_promotion`. (2026-07-25)

  **Unverified — the planner must confirm first:** I did NOT enumerate
  `superkaiba1/explore-persona-space-data` `issue1345_partial/` to establish that the
  leg's artifacts are present and complete. Treat "the bundle holds a completed result"
  as an unverified hypothesis from the session's chat text; the HF listing is step one
  of both the audit design and any recovery.

## Proposed change (refine in planning)

```
+ periodic reconciliation (cron or a watcher pass):
+   list issue<N>_partial/ prefixes on the HF data repo (huggingface_hub
+     list_repo_files, prefix-scoped — never a full-repo listing)
+   for each, compare the result-shaped payloads it carries
+     (eval_results_issue_<N>/ ... , data_issue_<N>/ ...) against what is
+     committed under eval_results/issue_<N>/ in git
+   escalate (sidecar JSONL + one deduped push) any bundle whose payload has no
+     committed counterpart -> "paid result persisted only to a crash bundle"
+   NEVER auto-commit a bundle's contents: a partial result silently landing in
+     eval_results/ is worse than the current loss
```

Design questions for the planner: how to tell a COMPLETED payload from a genuinely
partial one (a row-count or a manifest inside the bundle? the presence of
`crash_persist_transcript.log` as the audit trail?), and how to bound the HF listing
cost — the prefix-scoped Hub-call discipline in `.claude/rules/artifact-reuse.md` (i)
applies.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (a new escalate-only pass), or
  a standalone cron script alongside the existing janitors — the planner picks, noting
  that an HF listing is slower than the watcher's other passes and may not belong on a
  10-minute tick.
- `.claude/rules/upload-policy.md` — a pointer sentence, since that rule owns the
  persist-by-default contract this audit closes the loop on.
- Read-only against both HF and git.

## Constraints / invariants

- **Never auto-commit** bundle contents (see above) and never delete a bundle.
- Prefix-scope every Hub call; do not enumerate the whole data repo.
- Escalate-only, deduped per (issue, attempt_id), so a long-lived bundle does not
  re-push.
- Fail-open on Hub errors — a transient 5xx must not produce a false "stranded" alert.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Related

Recovering #1345's specific leg (and deciding recover-vs-re-run, which spends GPU) is
tracked separately as a `daily-held` `needs-human` task from this same sweep.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 68c1c042b809
- Source: `/daily` 2026-07-25 transcript sweep, session `63122023` @ 18:01:02Z.
