---
name: inline-round-root-commit-cert
description: Repo-root inline rounds committing scripts/ need inline_lint_gate.py certification; manual workflow_lint PASS does not certify; bare hf_hub_download in live code fails [live-hf-retry-routing]
metadata:
  type: feedback
---

Two gates every inline free-analysis round with a scripts/ payload hits (#2054 extended-decomp round, 2026-08-20):

1. `guard_root_code_commit.sh` BLOCKS any repo-root commit carrying scripts/src/tests files unless `scripts/inline_lint_gate.py --issue <N> --payload-file /tmp/issue-<N>-<round-slug>-inline-payload.txt` has certified the EXACT staged content (writes /tmp/eps-inline-lint-cert-v1.txt). A manual no-flags `workflow_lint.py` PASS + mapped tests do NOT certify — run the gate script itself (one background Bash, ~3-8 min; re-run after ANY further edit). Round-slug in the payload filename is required (bare issue-keyed name refused).
2. The no-flags lint's `[live-hf-retry-routing]` check FAILs a bare `hf_hub_download` in live code — wrap in `orchestrate.hub.retry_transient(lambda pin=...: hf_hub_download(...), what="...")` (call shape: see scripts/issue2054_pool_specialize.py load_ceilings).

**Why:** the guard is content-sha keyed (cert-diag shows want/staged/worktree shas), so certification must run against the final bytes; workflow_lint under fleet load (load1>20) also times out 570s foreground — run it as background Bash with an `until grep LINT_RC` wait.

**How to apply:** on any inline round: fix lint findings FIRST, then inline_lint_gate (certifies), then `git commit -- <paths>` (retry on index.lock with an until-loop), then bare `git push`; verify landing by `git merge-base --is-ancestor <sha> origin/main` + blob read ("Everything up-to-date" can mean a concurrent fleet push already carried your commit). Related: [[worktree-commit-and-selector-vintage]].
