---
name: concurrent-followups-wrong-plan-symlink
description: When a task has MULTIPLE concurrent same-issue follow-up rounds, the plan.md symlink AND latest-marker (impl + smoke-arch) can resolve to a DIFFERENT round than the code under review — round-match ALL inlined inputs by followup_label + note body, not by the symlink / latest-marker
metadata:
  type: feedback
---

When composing a Codex code-review prompt for a same-issue follow-up round, the
canonical `plan.md` symlink is NOT reliably the plan for the round whose code
you are reviewing — AND `task.py latest-marker` is NOT reliably the round's
marker either.

**Rule:** identify the approved plan by the round's `followup_label`, not by the
`plans/plan.md` symlink. Grep the plan-version head lines
(`head -1 plans/v*.md`) for the label, and cross-check the
`epm:followup-scope` / `epm:plan` markers on `events.jsonl` for which version is
that round's FINAL. Inline THAT version with the `---BEGIN APPROVED PLAN BODY---`
envelope + a note telling Codex why the on-disk symlinks are wrong.

**The same contamination hits `latest-marker` for BOTH the implementation marker
AND the smoke-architecture-check marker** (confirmed on #841, 2026-07-03, when
composing the scaling-capture crash-fix review): the concurrent gru-source-only
round posted its OWN `epm:experiment-implementation` (a NEWER version — v12 gru
vs v11 scaling-capture) and its OWN `epm:smoke-architecture-check` (the one
`latest-marker --prefix epm:smoke-architecture-check` returned described the GRU
round, not scaling-capture). So `latest-marker` returned the wrong concurrent
round's marker for the smoke-arch, and would return the wrong round's impl marker
on any re-fetch. **Round-match ALL THREE inlined inputs** (plan, impl marker,
smoke-arch marker) by reading the marker NOTE bodies + the plan head lines for
the round's identity, never by trusting `latest-marker` / the bare symlink. The
brief may explicitly pin the impl marker version (#841 brief said "inline v11")
— honor that pin over `latest-marker`. Also: the branch HEAD may be the
CONCURRENT round's commit (#841 HEAD = the gru commit `68d38959a7`, not the
scaling-capture commit `8bc38f0e6f`), so a `git diff main...HEAD`/`main..HEAD`
would mis-scope the review — instruct Codex to `git show <round-commit-sha>` ONLY
and bar the main...HEAD ladder for concurrent-follow-up rounds.

**Why:** a task can have >1 concurrent same-issue follow-up rounds
(`epm:followup-scope v1` scaling-capture AND `v2` gru-source-only both live on
#841, 2026-07-02). The `plans/plan.md` symlink resolves to the MOST RECENTLY
created plan version — which may be a DIFFERENT concurrent round. On #841,
plan.md → v11 (round 2: gru-source-only) while the code under review was round 1
(scaling-capture, FINAL plan = v9). The worktree copy was ALSO wrong
(approved/…→v6, the pre-follow-up parent plan).

**Why the standard Step 2-pre-b freshness diff does NOT catch this:** it diffs
the worktree plan against the canonical `plan.md`. Here BOTH are wrong — the diff
just reports "differs" and would inline `plan.md` (the wrong concurrent round).
The freshness check assumes plan.md is the source of truth for the round; with
concurrent follow-ups that assumption breaks. Inlining the wrong plan is the
silent #546-class wrong-plan failure: Codex scores plan-adherence against a
totally unrelated experiment's contract with no error.

**How to apply:** whenever the brief names a `followup_label` or the round is a
same-issue follow-up, resolve the plan version by the label + the round's
FINAL `epm:plan` marker, never by the bare `plan.md` symlink. Keep pinning that
version across re-compose rounds until the concurrent follow-ups settle. Related:
[[gh_graphql fallback to REST]] is unrelated; this is a plan-resolution lesson
for the Step 2-pre-b compose stage.
