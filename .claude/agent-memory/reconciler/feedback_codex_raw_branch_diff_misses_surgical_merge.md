---
name: Codex branch-stale BLOCKER ignores Step 10d Guard 3 surgical-merge procedure
description: Codex FAILs "branch reverts current-main" by reading the raw `git diff origin/main..issue-N` showing 100K+ line deletions, but /issue Step 10d Guard 3 prescribes artifact-confirmed surgical-checkout that bounds the actual merge to additive-only paths
type: feedback
---

When `git diff --stat origin/main..issue-N` shows 100+ files changed and many thousands of deletions of files OUTSIDE the issue-N's own scope (workflow scripts, sibling-task records, sibling-issue eval_results, sibling-issue figures), Codex code-reviewer FAILs `branch-stale-reverts-current-main` and reads it as a repo-integrity BLOCKER independent of the analysis result. The raw branch diff IS load-bearing as a fact, but the actual merge operation at task terminal state is NOT a blind `git merge issue-N`.

**Why:** The /issue SKILL.md Step 10d Guard 3 (`Behind-main / non-main-base guard`, fires at `BEHIND > 200` commits or non-main base) auto-routes to the **artifact-confirmed merge** procedure: surgical additive checkout limited to `tasks/<status>/<N>/`, `figures/issue_<N>/`, `eval_results/issue_<N>/`, and `scripts/issue<N>_*.py`. So the 100K-line revert that Codex flags in the raw diff is bounded by the merge procedure and will NEVER land on main. Codex cannot know this — it reviews the raw branch diff, not the orchestrator's auto-merge procedure.

**How to apply:** When Codex's BLOCKER cites a large-scale revert of OUTSIDE-scope files vs current main (sibling-task `tasks/running/{M,...}/` records, sibling `eval_results/issue_M/`, sibling `figures/issue_M/`, workflow scripts the current issue did not touch):
1. Verify the raw diff is real (`git diff --stat origin/main..issue-N` confirms the deletion count).
2. Identify which paths are #N's ADDED scope vs which are reverts of OUTSIDE-scope main commits.
3. Trust the Step 10d Guard 3 surgical-checkout will fire at merge time — verify by reading SKILL.md Step 10d for the BEHIND-threshold + additive-path enumeration.
4. PASS with hard standing recommendation that the orchestrator's auto-merge MUST use the surgical additive-checkout path (NOT a blind rebase / `git merge --no-ff`), and enumerate the additive paths in the reconcile body so the merge step is auditable.

This is companion to "Codex env-var orphan unreachable" — both are cases where Codex's literal-reading BLOCKER is real but does not affect the actual operation because the orchestrator has a documented procedure that bounds it.

Origin: task #511 round-1. Codex FAILed `branch-stale-reverts-current-main` citing 116,778-line deletion across 179 files including `scripts/poll_pipeline.py` (-111), `tests/test_poll_pipeline_sentinels.py` (-222), and full deletions of `tasks/running/{519,520}/`, `eval_results/issue_{508,509}/`, `figures/issue_{508,509}/`. Verified via `git diff --stat origin/main..HEAD`. Branch was forked at `29cfc17e1` weeks back; ~667 main commits landed since. Step 10d Guard 3 surgical-checkout bounds the actual merge to additive-only #511 paths, so PASS with the merge-routing standing rec.
