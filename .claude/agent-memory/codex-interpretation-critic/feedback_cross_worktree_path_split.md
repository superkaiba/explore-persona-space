---
name: Cross-worktree path split for figures vs eval-JSONs
description: Same-issue follow-up rounds can have eval JSONs/raw-completions committed ONLY on the issue branch+worktree while figures+body live ONLY on main — pass each by the absolute path where it actually exists
type: feedback
---

On same-issue follow-up rounds the artifacts can be split across two trees,
and a single sandbox-root assumption breaks lens 6 or lens 7.

**Why:** The figures get committed to `main` (the analyzer's figure-publish
step runs against the repo root), while the round's eval JSONs + raw
completions + the within-round verdict.json are committed on the
`issue-<N>` branch and live ONLY in the `.claude/worktrees/issue-<N>/`
tree (not yet merged to `main`). The task body + plan live under
`tasks/<status>/<N>/` which only exists on the `main`/repo-root tree (the
canonical resolver branch-guards to main). So NO single sandbox root holds
all of: figures (main only), eval/raw/verdict (worktree only), body+plan
(main only).

**How to apply:** Before composing, check WHERE each artifact actually
resolves (`ls` the repo-root path AND the `.claude/worktrees/issue-<N>/`
path). Then in the prompt pass each input by the ABSOLUTE path where it
exists — figures + body by absolute `main` repo-root path, eval JSONs +
raw completions + verdict by absolute worktree path. Absolute paths
resolve regardless of which tree Codex's sandbox is rooted at. Always
INLINE the plan (Step 2-b envelope) rather than reference it by path —
the plan path is the least reliable to resolve cross-tree and inlining
can't fail. The brief's `figure_paths` may be stated relative or as a
GitHub raw URL; convert to the absolute local path that exists for the
multimodal lens-6 load. (Surfaced on #613 single-space-falsifier round,
2026-06-15.)

**Task-STATUS folder split — another reason the plan path is unreliable
cross-tree (#841, round 1, 2026-07-02).** Even when the brief hands an
ABSOLUTE repo-root plan path, the worktree can hold the task folder at a
DIFFERENT status than the repo-root — e.g. worktree
`tasks/approved/841/plans/plan.md` vs repo-root
`tasks/interpreting/841/plans/plan.md` (the status advanced on main while
the worktree branch lagged). Codex's worktree-rooted sandbox then can't
resolve the repo-root `interpreting/...` path, and a worktree-relative
`interpreting/...` path doesn't exist there either → false "plan
unreachable" BLOCKED. This reinforces the same action: ALWAYS inline the
plan (`cat` the repo-root plan into the Step 2-b `---BEGIN/END APPROVED
PLAN BODY---` envelope via a NON-interpolating write — plan bodies carry
`$`/backticks; here v6.md was 27 KB), and tell Codex a "plan unreachable"
BLOCKED is invalid. Don't reference the plan by path even when the
repo-root path `ls`-resolves for YOU — your cwd is not Codex's sandbox
root.

**Lens-7 smoke-copy trap (#685, 2026-06-27).** When the round's eval JSONs are
worktree-only, the worktree may ALSO carry a `*_smoke/` sibling dir with a
same-named raw-generations file from the smoke run. The REAL run's raw
sample-TEXT file (e.g. `validity_generations.json`) is frequently HF-only,
while only its judge-LABEL file (e.g. `validity_judged.json`, carrying per-cell
rates) is worktree-local. Point Codex at the worktree judge-label file for the
firing-RATE read (works without HF creds), give the HF path for the sample-TEXT
verification, and explicitly warn it NOT to use the worktree `*_smoke/` copy of
the generations file for lens 7. Codex sandboxes often lack HF creds, so the
sample-text sub-check may legitimately come back BLOCKED — that's correct, not
a failure to PASS-around.
