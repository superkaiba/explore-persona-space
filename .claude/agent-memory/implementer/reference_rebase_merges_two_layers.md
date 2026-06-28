---
name: rebase-merges-two-layers
description: Explicit `--rebase` on the CLI OVERRIDES `pull.rebase=merges` config — documented commands need the literal `--rebase=merges` flag AND the config backstop; never "simplify" by dropping one layer
metadata:
  type: reference
---

The 2026-06-12 shared-root merge-flattening fix has TWO load-bearing layers:

1. Every documented/scripted `git pull --rebase` carries the literal
   `--rebase=merges` flag (CLAUDE.md "Concurrent repo-root committers",
   workflow-fix-on-bug.md merge procedure, issue/SKILL.md surgical-checkout
   bullet, sync_pods.sh / sync_env.sh / bootstrap_pod.sh fallbacks).
2. Shared repo config pins `pull.rebase=merges` + `rebase.autoStash=true`
   (set on `/home/thomasjiralerspong/explore-persona-space/.git/config`;
   covers main checkout + all worktrees, but NOT pod-side clones).

**Why both:** the code-reviewer empirically verified (git 2.34.1, scratch
repo with the incident topology) that an explicit bare `--rebase` on the
command line OVERRIDES `pull.rebase=merges` config and still flattens
unpushed merge commits. So the config is only a backstop for bare
`git pull`; any doc/script left at plain `--rebase` re-introduces the
hazard. A future "the config covers it, simplify the flags away" edit is
WRONG — deflect it. `--rebase=merges` needs git ≥2.18 (VM 2.34.1, pod
image ubuntu22.04 = 2.34.x — fine). Incident: two workflow-fix merges
silently dropped by a concurrent session's documented recovery pull,
2026-06-12. See [[ff-worktree-to-main-before-edit]] for the sibling
worktree-staleness rule.
