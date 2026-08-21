---
title: 'Step 5a spec-freshness sync silently no-ops when invoked with cwd at the repo
  root (cwd-derived WT trips the on-main guard; certified a gate against stale main
  twice on #2205)'
kind: infra
tags:
- workflow-fix
- step5a-spec-freshness
created_at: '2026-08-21T14:53:21Z'
has_clean_result: false
parent_id: 2205
origin_prompt: '/issue 2205 (Step 10d merge-completion session) — the mandatory pre-gate
  Step 5a re-sync printed ''[step5a] session on main (repo root) ... skipping'' because
  the caller cwd was the repo root; same trap already recorded in #2205 marker v23
  for round 3'
workflow: v1
---
---
kind: infra
---

# Step 5a / 10d spec-freshness sync silently NO-OPS when invoked with cwd at the repo root (cwd-derived `WT` trips the on-main guard)

## Provenance

workflow_fix_target: .claude/skills/issue/steps/09-step-5.md (§ Spec-freshness check; the same cwd-derivation is copied into `.claude/skills/issue/steps/18-step-10d.md` § post-gate re-sync, which is pre-bound and therefore NOT affected)

Surfaced twice on task #2205, in two different sessions:
- Round 3 (2026-08-20), recorded verbatim in #2205 marker v23: "the Step 5a re-sync FIRST no-op'd because it was invoked with cwd at the repo root (its on-main guard skipped it) — re-run correctly from the worktree, landing caa77b2aaf3e plus a 2-file sibling sync".
- The 2026-08-21 merge-completion session, again: the mandatory pre-gate re-sync printed `[step5a] session on main (repo root) — spec-freshness sync is worktree-only; skipping` because the invoking Bash call had `cd`'d to the repo root to run the single-flight and fleet probes. Re-run from the worktree it landed a REAL commit (11539e3ee5d8) — main had genuinely advanced during the gate window, so the no-op would have certified the gate against a stale main.

## Goal

Make the Step 5a spec-freshness sync impossible to silently no-op because of the CALLER's cwd. The sync must either (a) sync the intended worktree regardless of cwd, or (b) fail LOUD when it cannot determine which worktree to sync — never print a skip line that reads like a successful no-op.

## The bug

The Step 5a block opens with:

```
WT=$(git rev-parse --show-toplevel)
if [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "main" ]; then
  echo "[step5a] session on main (repo root) — spec-freshness sync is worktree-only; skipping"
else
  ... entire sync body ...
fi
```

`--show-toplevel` resolves from the CALLER's cwd. The on-main guard is correct in intent (#1747: with fetched `origin/main` as the sync source, running the body on a repo-root checkout would check out origin/main content into the SHARED root working tree and commit on main — a concurrent-committer hazard). But the guard's INPUT is the wrong thing: it asks "what branch is the tree at my cwd on?" when the question is "which worktree am I supposed to be syncing?".

So an orchestrator that runs the sync from a repo-root cwd — which happens naturally, because most of the surrounding pre-gate sequence (`step9c_baseline.py probe`, `task.py`, the guard helpers) is documented as resolving from the MAIN checkout — gets a skip that is indistinguishable in the log from "nothing to sync". The two observed instances both then certified, or nearly certified, a gate against a stale main.

Why this is worse than an ordinary silent skip: the pre-gate re-sync is MANDATORY before every gate re-launch (#1742/#2006) precisely because origin/main moves during a ~15-40 min gate window. A no-op there is not a lost convenience — it is the gate certifying a landing tree that does not match the main it will land on.

## Proposed fix (implementer to design; sketch only)

1. Take the worktree path as an EXPLICIT input rather than deriving it from cwd. The issue number is already known at every call site, and Step 10d already has the canonical derivation: `REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")` then `WT="$REPO_ROOT/.claude/worktrees/issue-<N>"` (extracted as `scripts/step10d_guards.sh --guard prelude`). The natural fix is to extract the Step 5a sync the same way — a checked-in `scripts/step5a_sync.sh <N>` (or a `--guard spec-freshness` subcommand on the existing guards script) that derives `WT` from the issue number, never from cwd. That also retires the per-session ~285-line transcription of this block, which is the same class of hazard #1978 extracted the Step 10d guards for.
2. Keep the on-main protection, but re-key it on the RESOLVED target: refuse (loudly, non-zero) when the resolved target tree is on `main`, rather than skipping when the CALLER happens to be on main. A repo-root session with no issue worktree is a caller error worth a non-zero exit, not a silent skip.
3. Make the skip line unmistakable if any skip path survives: it must not be reachable in the common case, and it must be distinguishable from a successful no-drift sync (today `[step5a] no sync commit landed` and the on-main skip are both benign-looking single lines).

## Acceptance criteria

1. Invoking the sync for issue `<N>` from a repo-root cwd syncs the `issue-<N>` worktree (or exits non-zero) — it does NOT print a skip and exit 0.
2. Invoking it from inside the `issue-<N>` worktree behaves exactly as today (byte-equivalent effect on the synced set + commit subjects, which the Step 10d verdict re-bind's A/M byte-identity probe and Guard 3's sync-subject anchor both depend on).
3. A resolved target on `main` still refuses to run the body — with a non-zero exit and a message naming the resolved path.
4. A pin test covers the repo-root-cwd call: today's behavior (skip + exit 0) must fail it.
5. The Step 5a and Step 10d prose is updated to point at the extracted helper, and the `--check-lessons-index` / spec-pin lints stay green.

## Notes for the implementer

- Do NOT widen the sync's SCOPE as part of this fix. The SPECS set, the family-atomic dirty logic, the sibling-issue per-FILE arm, and the `scripts/step5a_sibling_probe.py` probe are all correct as of #2412 and out of scope here — this is purely about how `WT` is resolved and how the guard fails.
- There is a substantial existing Step 5a backlog (#2260, #2311, #2327, #2374, #2385, #2416, #2420, #2423, #2424, #2444). None of them covers this bug: they are all about WHICH files sync or WHETHER a synced file is satisfiable. This one is about the sync not running at all. Coordinate if any of them lands an extraction first — in that case this becomes a small change on top of it rather than its own extraction.
- The Step 10d post-gate re-sync copy of the same block is NOT affected: it runs with `$WT` already bound by the guards prelude and explicitly documents "Uses the ALREADY-BOUND $WT (do NOT re-derive from cwd; a repo-root cwd would rebind to the shared root)" — which is direct evidence that the hazard was already recognized on the 10d side and simply never back-ported to Step 5a.
