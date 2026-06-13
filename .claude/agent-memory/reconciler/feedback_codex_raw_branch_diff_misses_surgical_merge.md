---
name: Codex branch-stale BLOCKER ignores Step 10d Guard 3 surgical-merge procedure
description: Codex FAILs "branch reverts current-main" reading the raw `git diff origin/main..issue-N` (100K+ deletions of outside-scope files); the actual merge at terminal state is the Guard-3 artifact-confirmed surgical additive checkout, which bounds the revert.
type: feedback
---

**Rule:** when Codex's BLOCKER cites a large-scale revert of OUTSIDE-scope files vs current main (sibling task records, sibling eval_results/figures, workflow scripts the issue never touched): (1) confirm the raw diff is real; (2) split #N's ADDED scope from outside-scope reverts; (3) verify /issue SKILL.md Step 10d Guard 3 (fires at BEHIND > 200 commits or non-main base) routes to the surgical additive checkout limited to `tasks/<status>/<N>/`, `figures/issue_<N>/`, `eval_results/issue_<N>/`, `scripts/issue<N>_*.py`; (4) PASS with a hard standing rec that the auto-merge MUST use the surgical path (never a blind rebase/merge), enumerating the additive paths in the reconcile body so the merge is auditable.

**Origin:** #511 r1 — 116,778-line deletion across 179 files (branch forked ~667 main-commits back); Guard 3 bounds the merge to additive #511 paths. PASS.

Companion: [[feedback_codex_env_var_orphan_unreachable]] (real literal finding, bounded by a documented orchestrator procedure); [[feedback_codex_litigates_pre_existing_in_round_n]] (stale-branch flavor).
