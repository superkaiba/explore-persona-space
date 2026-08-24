---
name: ops-landing (branch-surgery) compose shape
description: For kind:infra repo-state landing tasks (detached scratch worktree, custom BASE ref, unpushed commits landing another branch's deliverables), replace the main...HEAD ladder with a BASE0..HEAD sha-range + per-file bodies, use branch-tip byte-identity diffs as the wholesale-set verification, and extend the execution ban to mutating git
type: feedback
---

For a repo-state OPS landing task (#1037 r1: landing branch issue-763's
stranded deliverables via a DETACHED scratch worktree at origin/main + N
unpushed commits; no branch, no PR), the standard compose recipe needs four
structural swaps:

1. **Diff acquisition:** the review range is `BASE0..HEAD` (BASE0 from the
   brief; an ancestor of HEAD so two-dot is exact) — NOT `main...HEAD`, which
   is meaningless on a detached tree. When even the single code commit's body
   is over the 300 KB budget (LAND1 was 815 KB), mandate: shape reads
   (--stat/--name-status) + per-file bodies ONLY for the non-wholesale paths
   (merged/appended files) + **branch-tip identity diffs as the wholesale
   verification tool**: `git diff <branch-tip> HEAD -- <path>` empty ⇔
   byte-identical; the aggregate `--stat` over the wholesale path list
   enumerates ALL deviations at once. This replaces reading +17K wholesale
   lines that already passed the source branch's own review pipeline — scope
   the review to the LANDING OPERATION, with a bounded import-resolution
   spot-read of landed src modules (stripped-foreign-file import deps are the
   residual risk).
2. **Execution ban extension:** ban MUTATING GIT explicitly (commit/amend/
   push/reset/checkout-paths/clean/merge/rebase/worktree) — the tree holds
   unpushed landing commits pending the orchestrator's push step; read-only
   git only. The generic "never run smoke commands" line does not cover this.
3. **3-way-merge review focus:** for a merged file, demand feature-presence
   greps vs BOTH parents (`git show <ours>:<path>` / `git show <theirs>:<path>`)
   with per-dispatch-path threading traces, and make "dropped either parent's
   feature" a Critical + a §7 must-abort cross-reference. Step 3.7 sibling
   sweep maps to the OTHER dispatch paths of the same file.
4. **Fix-forward deviations from byte-identity:** when the implementer amends
   deviations into wholesale files (jointly-unsatisfiable byte-identity vs
   main-side test gates, e.g. dotenv-hoist guard), present BOTH sides in the
   prompt (deviations-allowed clause vs the not-pre-enumerated wholesale edit)
   and route adjudication to Codex with an explicit additive-only check
   (`git diff <tip> HEAD -- <file>` must show + lines only) — honest+minimal
   → CONCERNS max; wrong/undisclosed → substantive.

Compose-side probes that paid off: foreign-path grep + top-level dir census
on the full landing range (cheap, catches brief drift); confirming all named
refs (`tip`, merge-base, baseline SHA) resolve in the worktree so Codex's
probe commands cannot dead-end; verifying the worktree plan is identical to
canonical before path-referencing (running-status tasks CAN have the folder
present in a freshly-cut scratch tree).

Also validated: the #1040 r1 infra template is a good reuse base for infra
composes generally (its N/A block + 0.68 sub-checks + Step 0.9 carry over
verbatim); adapt rather than re-derive from code-reviewer.md.

**Direct-on-main ALREADY-PUSHED variant (#1083 r1: two landed commits repairing
tasks/ state on main, sandbox cwd = the SHARED live checkout):** further swaps
on top of the above — (a) acquisition is PER-COMMIT (`git show`/`git diff-tree`),
no base ref at all; all content probes run against the COMMIT trees, never the
live working tree (concurrent sessions drift it); (b) recommendation vocabulary
becomes `accept / fix-forward-revise / escalate-for-revert` (nothing is "before
merge"); (c) ban `task.py` ENTIRELY (even read paths — flock + branch-guard +
shared state) and instead run the read-only audit YOURSELF at compose time and
inline the rc + output as a composer-verified fact; (d) for a restore commit of
historical task content (~1 MB patch), mandate mechanical provenance only
(byte-identity `diff <(git show pin:src) <(git show commit:dst)` from the
committed pin table, mapped-path + symlink rows included) + a pipe-grep secret
scan over the full patch (counts/matched lines only — restored bodies can carry
refusal-trigger vocabulary, never page them); (e) frame the script's internal
check battery as the hollow-gate sub-check target (a post-write `cmp` against
the same `git show` is write-integrity only — the fidelity anchors are the pin
literals + id-identity + expected-title checks). Registry-row verification =
parse the registry AT the reconcile commit + `git ls-tree <commit> <row-path>`
per row. (Applied on #1083 r1.)
