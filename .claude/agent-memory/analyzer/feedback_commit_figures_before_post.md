---
name: Commit figures and eval results to main BEFORE posting clean-result
description: Hero figure URLs are commit-pinned raw.githubusercontent.com — those need to resolve at post time
type: feedback
---

Every clean-result hero figure embeds a `raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/...` URL.
That URL only resolves if the figure file is committed AND that commit is
pushed to origin/main.

**Why:** During analyzer runs, figures are usually generated in a worktree
(`.claude/worktrees/issue-N/`) but the working files end up under the main
checkout's untracked tree (`figures/issue_N/`). The worktree's branch
HEAD does NOT contain the figures unless someone explicitly committed them
on that branch — and the analyzer typically runs from main, where the
files are untracked. If you post the issue first, the image will 404.

**How to apply:** Before running `verify_clean_result.py` on the cached
draft, do these in order:

1. `git status --short figures/issue_N/ eval_results/issue_N/` — if `??`,
   they're untracked.
2. `git add figures/issue_N/ eval_results/issue_N/`
3. `git -c commit.gpgsign=false commit -m "#N add hero figures + eval results — <one-line desc>"` (NEVER skip hooks).
4. `git push origin main`
5. `git rev-parse --short HEAD` — use THIS sha in the figure URLs in the
   draft body. NOT the worktree's branch sha (e.g. `a3b51a3` for
   issue-257), which doesn't contain the figures.
6. THEN run the validator and post.

Verified on issue #257 (clean-result #276): figures lived in main repo's
working dir untracked, not in the issue-257 worktree branch, so the SHA
in the figure URL had to be the post-commit sha (`937aec9`), not the
plan-time sha (`a3b51a3`).
