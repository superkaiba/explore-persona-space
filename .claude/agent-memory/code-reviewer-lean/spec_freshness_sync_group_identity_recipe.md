---
name: spec-freshness-sync-group-identity-recipe
description: 4-probe identity review for a split-review group of Step 5a spec-freshness sync commits — probe at group TIP vs origin/main, payload-pattern grep, anchor subjects from 09-step-5.md:496/596, no diff-body read
metadata:
  type: feedback
---

Identity-verification recipe for a split-review group whose commits are orchestrator spec-freshness sync imports (issue 2587 r2 g8; ~270 KB diff, body never read):

1. `git diff <first>^..<last> | wc -c` first (budget discipline), then `--name-status` only.
2. Identity probe at the GROUP TIP, one command for all paths: `git diff --name-only origin/main <last-sha> -- <paths>` empty ⇒ every touched path byte-identical to main. The brief's literal `git diff origin/main -- <path>` compares the WORKING TREE — only equivalent when the sync is the branch tip; when later round commits exist, the tip form is the correct probe (working-tree form as cross-check for later re-touch).
3. origin/main may have ADVANCED past the brief's divergence-probe pin — resolve `git rev-parse origin/main` and say which ref the identity held against; identical-against-newer-main strengthens the verdict, but a non-empty diff on a path main moved after the sync is NOT the sync's fault (check `git diff <pin> <last-sha> -- <path>` before FAILing).
4. Payload check: `grep -cE 'issue<N>|issue_<N>'` on the path list (covers scripts/tests/src patterns at once); also confirm zero `D` statuses.
5. Anchor subjects are canonical in `steps/09-step-5.md`: line ~496 `issue-<N>: sync workflow-surface specs from origin/main (spec-freshness)`, line ~596 the `(spec-freshness; sibling-issue files)` form; the Step-5 branch-dirtiness exclusion keys on the BARE `spec-freshness` token (~line 427), so extended parentheticals (e.g. `+ .gitleaksignore`) still conform.

**Why:** the whole group is imported main bytes — content review would waste the round's diff budget; the only failure modes are partial import, payload sweep, and missing anchor.
**How to apply:** any SPLIT-REVIEW sub-scope whose commits are described as sync/import-from-main; also the exclusion-token check when auditing branch-dirtiness probes.
