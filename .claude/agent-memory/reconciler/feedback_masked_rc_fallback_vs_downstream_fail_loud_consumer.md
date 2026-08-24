---
name: masked-rc-fallback-vs-downstream-fail-loud-consumer
description: "Codex Major on a masked-rc resolve fallback (nested dirname swallows git rc) without tracing the downstream callee's own fail-loud resolver — the silent-path chain needed a sub-second transient flap; every persistent failure was caught one hop later (#2241 r4)"
metadata:
  type: feedback
---

Rule: when a FAIL claims a masked rc lets execution "proceed silently"
through a fallback value (`REPO_ROOT=.` from `dirname ""` after a failed
nested `git rev-parse`), trace the FIRST CONSUMER of the fallback before
upholding — a masked producer feeding a fail-loud consumer routes to the
same telemetry one hop later, and the silent chain then needs a
TRANSIENT flap (producer fails, consumer's identical probe succeeds
milliseconds later), not just "git can fail".

**Why:** #2241 r4 — the r3 binding remedy prescribed the exact nested
one-liner verbatim. Codex Majored: mask real (verified, `R=[.]` rc=0)
→ worktree cwd → branch-local `./scripts/task.py` → "if it returns
valid JSON the create proceeds without the promised TITLE_RC
telemetry". The unexamined conditional: `task_workflow.repo_root()`
(task.py:383-397) runs its OWN sanitized-env
`git rev-parse --path-format=absolute --git-common-dir` from
`_MODULE_DIR` and raises `RuntimeError` on git-missing AND
CalledProcessError, never a `__file__`/cwd fallback — so git-missing,
git-broken, old-git (no `--path-format`), corrupt worktree pointer, and
git-less scratch trees ALL reproduce inside the consumer → rc≠0 →
`|| TITLE_RC=$?` → the skip+telemetry arm. Only a sub-second flap of
the same plumbing escapes, and its outcome was a CORRECT draft PR
(read-only `view` on a branch copy byte-identical to main). Verdict:
binding PASS; the two-stage rc-capture (`step10d_guards.sh::
_derive_repo_root` is the in-repo hardened sibling) went to standing
recs + a fleet-wide idiom sweep (the nested form is the house TEMPLATE
idiom at 8+ sites across 5 skill files).

**How to apply:** (1) For any masked-rc / fallback-value blocker,
enumerate persistent-failure scenarios and check each against the
consumer's failure semantics — read the consumer's resolver/guard code,
not just the fence. (2) The severity discriminator on the SAME concern
id across rounds: r3's defect fired at EVERY round entry with git
healthy (blocker); r4's residual needed a transient flap with benign
outcome (standing rec). Cf. [[codex-blocker-on-unreachable-exception-path]]
(reachability arithmetic) and
[[claude-convention-defense-unverified-env-var-fence]] (the r3 flip side
— don't over-generalize THIS memory back onto every-entry fail-opens).
(3) A residual that is a pre-existing house idiom at N sibling sites is
a follow-up sweep, not a bounce of the one site under review.
