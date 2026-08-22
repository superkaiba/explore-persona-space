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

**Stale same-task body.md inside the worktree (round 3+, 2026-07-06).** The
status-folder split also strands a STALE `body.md`: the worktree held
`tasks/awaiting_promotion/<N>/body.md` with the OLD round-1 title and zero
mentions of the new round's results, while the current folded body lived at
repo-root `tasks/<new-status>/<N>/body.md`. Codex's sandbox root makes the
stale copy the EASIER one to find. Detect by comparing the two bodies'
titles / a round-specific keyword count; then pin the repo-root absolute
path in the prompt AND add an explicit "do NOT read the worktree
tasks/<old-status>/<N>/body.md — stale checkout" warning. Without the
warning, every applied/absent fix-landing check in a revision round scores
against the wrong body.

**Verified-identical refinement to "always inline" (#2333 r1, 2026-08-18).**
The always-inline rule exists because unverified plan paths fail cross-tree.
When the WORKTREE ITSELF holds the task folder (even at a stale status
folder like `tasks/planning/<N>/`) and you VERIFY at compose time that (a)
the worktree `plans/plan.md` symlink targets the same version as canonical
and (b) `md5sum` matches the canonical main-tree plan, referencing the
WORKTREE-absolute path is safe and correct (Step 2-b default case) — it is
inside Codex's sandbox root by construction. On #2333 this kept the prompt
at 34 KB instead of ~112 KB (77 KB plan). Keep the guard sentence ("plan
unreachable is invalid — path verified inside your sandbox") and a
grep-then-slice instruction for large plans. Inline ONLY when the worktree
copy is missing or fails the identity check.

**Local `judge_inputs` mirror is a lens-7 text source (#2333 r1, 2026-08-18).**
When judge items/scores carry only `answer_sha16` + labels (no text) and
rollout text looks HF-only, probe `<worktree>/data/issue_<N>/judge_inputs/`
first — the VM-side judging stage stages a full rollout-text mirror there
(JSONL rows with `response_text` + join keys). Pointing lens 7 at that
mirror makes it fully local/scoreable and demotes the HF path to the
advisory network carve-out, eliminating the EXCEPTION-2 BLOCKED risk. The
interpretation marker's phrase "local judge-input mirror" is the tell it
exists.

**`rawcomp_cache/` HF-snapshot mirror — a second lens-7 local-text source
(#2379 r1, 2026-08-20).** Sibling of the judge_inputs pattern: probe
`<worktree>/data/issue_<N>/rawcomp_cache/<hf-prefix>/` — some runs stage a
full local snapshot of the HF data-repo prefix there (subdirs mirroring
`raw_completions/<stage>/<arm>/raw_completions.json` — dict with a `rows`
list — plus `judge_scores/`, `train/*.jsonl`). When present, name it the
PRIMARY lens-7 source and demote the body's pinned HF tree to advisory
liveness. Probe BOTH paths (`judge_inputs/`, `rawcomp_cache/`) before
concluding raw text is HF-only.

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
