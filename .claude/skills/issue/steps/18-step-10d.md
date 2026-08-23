# Step 10d: Auto-merge the worktree (both experiment and impl)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

The worktree merge is **automatic — no prompt, no cooldown**. It is the
single canonical merge procedure, invoked from TWO trigger points:

- **Experiments** — at the `awaiting_promotion` transition (Step 9b),
  the instant clean-result-critic PASSes. The merge does NOT wait for
  the user to promote the clean-result.
- **Code-change paths** (`infra` / `batch` / `analysis` / `survey`) — at
  this step, via Step 10 step 6's `epm:merged`-not-yet-present branch;
  status is still `running` here BY DESIGN (#1723), the terminal flip
  DEFERRED to Step 10d's own Terminal-teardown sub-section.

Rationale: deferring the merge stranded shared-library fixes on unmerged
branches, so the next experiment inheriting from `main` lacked them
(#456 -> #466: a shared `trainer.py` fix lived on the deferred #456
branch; #466 inherited the stale copy from `main` and crashed Phase-0).
Auto-merging at the terminal point lands every code / figure /
`eval_results` commit on `main` immediately.

The worktree is **NOT removed** — it persists for inspection and is
reaped later by the daily stale-worktree audit (`worktree_audit.py`,
09:47) once the task reaches a terminal status and the worktree is idle.

**Idempotent.** Skip the whole step iff `epm:merged` already exists on the
task AND the branch carries no NOVEL payload vs fetched `origin/main`
(payload-scoped, #1897: a same-issue follow-up round produces NEW payload
on the same branch, and a prior round's `epm:merged` marker alone must not
strand it — #1768 round-2). "No novel payload" is NOT a bare
commit count — the default merge forms land COPIES of the branch commits
(`--rebase` replays them, `--squash` folds them into one), so a
fully-merged branch reads `rev-list --count origin/main..issue-<N>` > 0
forever (#1897 round-2). Use the layered novel-payload predicate from the
safe-case probe below (§ "The auto-merge procedure"), fail-SAFE toward
"payload exists": zero commits ahead → no payload; else
`git cherry origin/main issue-<N>` emits no `+` line → landed
(rebase-replayed copies keep their patch-ids); else the branch's own
changed files are content-identical on `origin/main` → landed
(squash-landed content); else → novel payload. Also skip if no PR
exists or the branch is already merged into `main` (no novel payload by
the same predicate).

#### Bare push / merge snippets (canonical — copy verbatim, never compose a piped variant)

Every `git push` / `git merge` / `git commit` / `gh pr merge|create` in this
skill — and any
IMPROVISED recovery around one — runs BARE with its exit code checked. Never
pipe one through `tail` / `grep` / `head` / any filter: bash makes a
pipeline's exit status the LAST stage's, so the pipe masks a rejected push
and the session proceeds on a merge that never landed (#957). The
`guard_piped_git_push.sh` PreToolUse hook BLOCKS the piped
shape anyway, so composing it just wastes a turn (#1138). Push/merge
output is a few lines — it needs
no trimming. Copy these forms; the earlier composition sites (Step 5 round
pushes, the failure-lesson memory persist, Step 9a-ter re-analysis commits,
the Step 9b auto-merge trigger) point here:

```bash
# (1) Worktree branch push, rebase-retry on reject (the safe-case form).
#     DESCENDANCY-GUARDED (#2312): the guard fires only on MUTUAL
#     non-ancestry (the Step-4a root-divergence probe's own two-leg shape).
#     A remote that is STRICTLY AHEAD (HEAD an ancestor of it) is
#     unambiguously NOT a rewrite — it takes the push arm, where the
#     rejected push (non-fast-forward: local is behind) falls into the
#     pull-retry below, which rebases onto the CORRECT same-branch
#     upstream and re-pushes — today's behavior, and exactly the #1880
#     fetch+rebase self-heal. MUTUAL
#     non-ancestry means ONE OF TWO states, and no mechanical predicate
#     separates them (a conflict-reconciling rebase defeats git-cherry
#     patch-equivalence):
#       (a) local history REWRITTEN — a mid-flight rebase onto origin/main
#           (itself prescribed for reconciling a sibling landing). The push
#           would be rejected non-fast-forward and the refspec-less pull
#           fallback would rebase HEAD onto the STALE remote branch,
#           replaying hundreds of main commits as new objects — the #1128
#           shape (#2296 measured [ahead 363, behind 1]; recurring: #1999,
#           #2171, #2296).
#       (b) remote genuinely DIVERGED — novel FOREIGN commits on the
#           branch while local holds unpushed commits. This is DOCUMENTED
#           fleet behavior, not a corner case: pod/GCE lanes commit results
#           to the issue branch by contract (pod-side-reporting.md
#           § Result-push verification contract, #1205) and their
#           prescribed fetch+rebase push races orchestrator branch commits
#           (#1880; #1739: a healthy 31h run exited 1 at its terminal
#           push). The pull-retry pair below is the SANCTIONED self-heal
#           for exactly this state — the VM-side twin of #1880's
#           lane-side fetch+rebase recipe.
#     DISCRIMINATE BY INSPECTION before acting:
#       git -C "$WT" log --oneline HEAD..origin/issue-<N>
#       git -C "$WT" cherry HEAD origin/issue-<N>   # '-' = already in HEAD
#     ALL remote-only commits novel FOREIGN work (b) ⇒ run the pull-retry
#     pair below BY HAND — it is the correct recovery. ANY remote-only
#     commit that is (or may be) this branch's own superseded pre-rebase
#     payload (a) ⇒ the Step 10d § Rewritten-branch landing route. Mixed ⇒
#     cherry-pick the foreign commits onto HEAD first, then the route.
#     MID-RUN COPY SITES (Step-5 round pushes etc. — no PASSing lint
#     verdict exists): the landing route is a Step-10d procedure;
#     leave the branch UNPUSHED, surface the state in the round note, and
#     let Step 10d land it. Never force-push (standing user-ask; the
#     policy question is task #2313). A MISSING remote ref takes the push
#     arm (push -u semantics unchanged — fails toward pushing):
if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \
   && ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \
   && ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then
  echo "mutually non-ancestral remote (#2312): REWRITTEN or DIVERGED — inspect HEAD..origin/issue-<N> per the comment above before ANY push/pull; never force-push"
  false
else
  git -C "$WT" push origin issue-<N> \
    || { git -C "$WT" pull --rebase=merges --autostash \
         && git -C "$WT" push origin issue-<N>; }
fi

# (2) Repo-root push to main, single-flight recovery on reject
#     (sync_repo_root exit 0 can mean "another sync in flight — your push
#      has NOT landed"; for guard-critical pushes use the landing-verified
#      form in the post-merge stale-task-folder guard below):
git push origin main || uv run python scripts/sync_repo_root.py

# (3) PR merge (for sites OUTSIDE Step 10d/9b — those two run the full
#     lint-verdict-gated blocks below, never this bare form) — branch the
#     flow on the EXIT CODE, never on filtered output (and exit 0 is NOT
#     proof THIS attempt landed: an already-merged PR exits 0 — verify
#     per the Step 10d landing-verification read, #1897):
if gh pr merge <PR> --rebase --delete-branch=false; then
  echo "merged"
else
  echo "merge failed — route to the Step 10d failure handling"; false
fi

# (4) Need to bound long output? Redirect to a FILE and read the FILE in a
#     SEPARATE command — the push itself stays bare:
git push origin main > /tmp/issue-<N>-push.log 2>&1; PUSH_RC=$?
tail -20 /tmp/issue-<N>-push.log
[ "$PUSH_RC" -eq 0 ] || { echo "PUSH FAILED (rc=$PUSH_RC)"; false; }

# (5) Commit whose OUTPUT you need (pre-commit hooks print there): redirect
#     to a FILE — never pipe (a piped hook-running commit is SIGPIPE-killed
#     mid-pre-commit-hook, #1584/#1591) — and read the file in a SEPARATE
#     command; pathspec-limited per CLAUDE.md § Concurrent repo-root
#     committers:
git commit -m "<msg>" -- <paths> > /tmp/issue-<N>-commit.log 2>&1; COMMIT_RC=$?
tail -20 /tmp/issue-<N>-commit.log
[ "$COMMIT_RC" -eq 0 ] || { echo "COMMIT FAILED (rc=$COMMIT_RC)"; false; }
```

Inside Step 10d itself, use the full executable blocks below (they wrap
forms (1)/(3) in the pre-push workflow-lint verdict gate); this subsection
is the copy source for every OTHER site.

**KEPT-stash surfacing duty (#1751; incident #1716).** Every
`sync_repo_root.py` invocation this skill prescribes — form (2) above, the
Step 4a divergence probe, the failure-lesson memory persist, Step 9a-ter
re-analysis commits, and ALL Step 10d sites (the four pre-marker syncs, the
post-merge-guard pre-sync, the unpushed-mv recovery, the local-residue
tail, the surgical push retry) — prints a per-stash report line on its
stderr report; a report line containing `stash: KEPT` (the emitted line is
two-space-indented — `  stash: KEPT …` — never anchored at line start)
means the sync could NOT cleanly re-apply a stash entry and a human owes
triage (the entry is kept + a rescue patch written — `sync_repo_root.py`'s
stash pop/keep step, rendered by `_emit_report`). When ANY sync this
session ran reports `stash: KEPT`: (a) append one line PER KEPT entry — a
sync reporting several KEPT entries gets one line each — to the round's
durable marker note: the `epm:merged` note file at merge sites, or one
adjacent `epm:progress` note where no merged marker fires (Step 4a, the
failure-lesson persist, 9a-ter), of the form
`stash-kept: <ref> (<sha12>) rescue=<rescue-patch path> — manual triage owed`;
(b) carry the same line(s) in the session's end-of-turn wrap-up. NEVER
summarize a KEPT-reporting sync as "clean" (the #1716 swallow: the flag
printed, the wrap-up said "Post-merge guard clean", and the stash sat
unowned). Surface only — the session never pops/drops the stash itself;
triage stays human.

#### Merge safety guards (run before the merge commands)

Derive the two paths cwd-robustly FIRST — never via `git rev-parse
--show-toplevel`, which from a worktree cwd returns the WORKTREE root and
nests `$WT` into `.../issue-<N>/.claude/worktrees/issue-<N>` (#506: the
guard snippet exit-128'd with "cannot change to ..."):

```bash
eval "$(bash scripts/step10d_guards.sh <N> --guard prelude)"
```

(This invocation preserves the original derivation byte-equivalent-in-effect:
`REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")`
+ `WT="$REPO_ROOT/.claude/worktrees/issue-<N>"`. The extracted script is the
canonical spelling of `--path-format=absolute`, retiring the hand-typo class
per task #1978. The bare `bash` invocation is deliberate — `bash` is on PATH
and needs no `uv run` wrapper.)

**Guard 0 — agent-memory pre-commit (run FIRST, before guards 1-3 and every
merge form).** Review rounds write per-agent memories
(`.claude/agent-memory/**`) with cwd in the worktree, leaving the tree dirty;
a dirty tree aborts the merge-conflict recovery's `git -C "$WT" merge
origin/main` below (#906). Commit them by explicit
pathspec — never `git add -A`:

```bash
eval "$(bash scripts/step10d_guards.sh <N> --guard 0)"
```

(The extracted script probes `git -C "$WT" status --porcelain --
.claude/agent-memory/`, commits by explicit pathspec if dirty, best-effort
pushes `issue-<N>`, and emits `MEM_COMMITTED=yes` when dirty agent-memory
was committed or `MEM_COMMITTED=no` when the tree was already clean.
Idempotent — a re-run finds the pathspec clean and skips
(`MEM_COMMITTED=no`). Exit 2 on infra error (worktree missing, commit
failed) with `ERROR=<reason>` on stdout; the caller's `eval` populates
`$ERROR` for inspection. Task #1978 extraction.)

Idempotent (a re-run finds the pathspec clean and skips). Scope is EXACTLY
`.claude/agent-memory/`: any OTHER dirty worktree path still surfaces through
the existing merge-failure handling — never blanket-commit it. REPO-ROOT-side
dirty agent-memory files are deliberately NOT committed here: every repo-root
pull in this step routes through `scripts/sync_repo_root.py` (see below),
whose autostash + rescue handling is built for the always-dirty shared root
(#967's hand-rolled root pull died `fatal: Cannot autostash`). For Guard 3
and the fast-path predicate, `.claude/agent-memory/**` paths are review-round
bookkeeping — always in-scope, never an UNSAFE trigger (see the Guard-3 note
and the fast-path mapfile filter below).

A behind-`main` `issue-<N>` branch can carry stale copies of OTHER tasks'
`tasks/` state, a crash between merge and a status flip can strand a
task at the wrong status, AND a branch based on another still-unmerged
`issue-<M>` branch will replay `#M`'s old commits onto `main` if blindly
rebase-merged. Five guards:

1. **Foreign-`tasks/` guard (strip whole foreign task folders before the
   merge).** `git diff --name-only "$MAIN_SHA"...HEAD -- tasks/` — the
   THREE-DOT form: merge-base..HEAD, i.e. only paths the branch's OWN
   replayed commits touch (#1280) — MUST be empty except THIS task's own
   folder (`tasks/*/<N>/`). Main-side advancement since the merge-base is
   BENIGN and must NOT trigger the strip: the `--rebase` merge replays only
   the branch's commits, so files the branch never touched keep `main`'s
   version. The retired two-endpoint endpoints (`"$MAIN_SHA" HEAD`) listed
   every path the fleet's marker churn advanced on `main` since the fork
   (#1271: 33 false positives on a branch whose replayed commits touched
   ZERO `tasks/` paths), and stripping those stages main-advancement content
   into a NEW branch commit whose server-side replay conflicts with main's
   further advancement — creating the very #1128-shape conflict the strip
   exists to prevent. The strip TARGET stays the freshly captured `main`
   snapshot (`MAIN_SHA`, captured in the block below).
   For any FOREIGN `tasks/` path in that diff — a `tasks/*/<M>/…` file for
   `M != <N>`, whether `events.jsonl`, `comments.jsonl`, `body.md`, or any
   other file — reset it to that snapshot BEFORE merging so the server-side
   `gh pr merge --rebase` has nothing foreign to conflict on (GitHub ignores
   this repo's `.gitattributes merge=union`, so a union merge cannot rescue
   a server-side conflict — the strip must happen here). The guard FETCHES
   `origin/main` first and pins every command to ONE captured `MAIN_SHA`:
   the fleet posts ~100+ marker commits/hr to `tasks/` on `main`, so a stale
   snapshot is the #1128 conflict class, and `origin/main` is a SHARED ref a
   concurrent session's fetch can advance mid-guard (the worktree shares its
   refs with every other session via the common git dir). A foreign path
   that EXISTS at `MAIN_SHA` is reset by checkout; a foreign path the branch
   ADDED (does not exist at `MAIN_SHA`) is dropped from the branch instead —
   a plain `git checkout "$MAIN_SHA" -- <added-path>` would crash with
   `pathspec did not match any file(s)` and abort the guard. Split FOREIGN
   accordingly:

   ```bash
   # Foreign tasks/* paths this branch touches (everything under tasks/ that
   # is NOT this task's own folder). Anchored so tasks/.../<N>/… is excluded.
   # MATERIALIZE the diff FIRST and check its OWN exit code: piped into grep
   # with `|| true`, a FAILED git diff (bad ref, missing origin/main) reads
   # as "no foreign files", the strip is silently skipped, and foreign
   # tasks/ reverts ride the merge (the #458 incident class — fail-open).
   # Same materialize-then-check pattern as the lint-gate trigger diff below
   # (#1047). The failure arm is TERMINAL (echo + false): do NOT merge —
   # route to the merge-failure handling (`epm:merge-failed v1`, continue).
   STRIPPED_FOREIGN=no   # set to yes iff a strip commit is actually created,
                         # so the safe-case push below fires only when needed.
   # Bounded mid-guard-churn retry (#1224): the strip work (checkout/rm/
   # commit) can fail when origin/main advances mid-guard (fleet churn moves
   # task folders; a piecewise execution re-derives a moved path). Attempt 2
   # re-runs the whole fetch->pin->diff->split->strip sequence against a
   # FRESH MAIN_SHA; a second failure is terminal. Composes with Known
   # failure shape 2 (post-certification recovery; this one is
   # pre-certification). Run the
   # block as ONE Bash call — piecewise execution was the true #1224
   # antecedent.
   GUARD1_STATE=pending
   for GUARD1_TRY in 1 2; do
     if [ "$GUARD1_TRY" -eq 2 ]; then
       echo "Guard 1 RETRY (once, #1224): strip failed under a stale pin — re-fetch + re-pin"
     fi
     # Freshness fetch + single-SHA capture (#1128): strip against main as
     # CLOSE to the server-side merge as possible, pinned to ONE SHA so a
     # concurrent session's fetch cannot advance origin/main mid-guard. A
     # FAILED fetch is a WARN, not a block: the no-foreign CERTIFICATION
     # below is correct against any snapshot — staleness only raises the
     # conflict probability, and the re-snapshot retry (Known failure
     # shape 2 below) is the recovery. (The materialize-then-check diff
     # failure below stays TERMINAL — that one breaks certification, #1184;
     # bad ref is not churn, so a failed diff producer is NEVER retried.)
     git -C "$WT" fetch origin main --quiet \
       || echo "Guard 1 WARN: fetch origin main failed — stripping against last-fetched origin/main (conflict-prone; Known failure shape 2 is the recovery)"
     MAIN_SHA=$(git -C "$WT" rev-parse origin/main)
     # Three-dot (#1280): merge-base..HEAD = paths the branch's OWN replayed
     # commits touch. Two-endpoint ("$MAIN_SHA" HEAD) read main-side
     # advancement as foreign (#1271) and its strip CREATED the
     # #1128-shape server-side conflict. The [ -z ] pre-check keeps an empty
     # MAIN_SHA fail-LOUD: an empty sha collapses the fused token to
     # '...HEAD' (= HEAD...HEAD, an EMPTY diff, exit 0 — silent fail-open),
     # where the old quoted empty argument made git error out.
     if [ -z "$MAIN_SHA" ] \
        || ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA"...HEAD -- 'tasks/' \
         > /tmp/issue-<N>-guard1-tasks-diff.txt; then
       echo "Guard 1: git diff \$MAIN_SHA...HEAD -- tasks/ FAILED (bad ref or empty MAIN_SHA) — cannot certify no foreign tasks/ paths; do NOT merge"
       GUARD1_STATE=diff-failed
       break
     # Work arm: two-command elif list — mapfile fills FOREIGN from the FILE
     # (grep semantics identical to the old pipe), then the [ ... ] test (the
     # LAST command's exit) decides the branch.
     elif mapfile -t FOREIGN < <(grep -Ev "^tasks/[^/]+/<N>/" \
           /tmp/issue-<N>-guard1-tasks-diff.txt || true); [ "${#FOREIGN[@]}" -gt 0 ]; then
       FOREIGN_ON_MAIN=()      # exist at MAIN_SHA -> reset to that snapshot's version
       FOREIGN_BRANCH_ONLY=()  # only the branch added them -> drop from branch
       for p in "${FOREIGN[@]}"; do
         if git -C "$WT" cat-file -e "$MAIN_SHA:$p" 2>/dev/null; then
           FOREIGN_ON_MAIN+=("$p")
         else
           FOREIGN_BRANCH_ONLY+=("$p")
         fi
       done
       GUARD1_STRIP_RC=0
       if [ "${#FOREIGN_ON_MAIN[@]}" -gt 0 ]; then
         git -C "$WT" checkout "$MAIN_SHA" -- "${FOREIGN_ON_MAIN[@]}" || GUARD1_STRIP_RC=$?
       fi
       # rm WITHOUT --cached (#1244): the strip commit below is PATHSPEC-limited,
       # and a pathspec commit records WORKING-TREE content for the named paths
       # (git-commit(1) --only default) — an index-only deletion (the old
       # --cached form) is resurrected by it (#1210: 19 resurrected paths). The
       # working-tree copies are stale duplicates of foreign tasks/ state; main
       # is authoritative.
       if [ "${#FOREIGN_BRANCH_ONLY[@]}" -gt 0 ]; then
         git -C "$WT" rm -f --ignore-unmatch -- "${FOREIGN_BRANCH_ONLY[@]}" || GUARD1_STRIP_RC=$?
       fi
       # Commit the reset/removal so the branch diff no longer touches them,
       # but only if anything actually changed (idempotent: a re-run finds
       # nothing staged and skips the commit). Record that a strip commit was
       # made so the safe-case merge below knows it must push before rebasing.
       if [ "$GUARD1_STRIP_RC" -eq 0 ] \
          && ! git -C "$WT" diff --cached --quiet -- "${FOREIGN[@]}"; then
         if git -C "$WT" commit -m "issue-<N>: strip foreign tasks/ folders before Step-10d merge (pinned to main @ ${MAIN_SHA:0:12})" -- "${FOREIGN[@]}"; then
           STRIPPED_FOREIGN=yes
         else
           GUARD1_STRIP_RC=$?
         fi
       fi
       if [ "$GUARD1_STRIP_RC" -eq 0 ]; then GUARD1_STATE=ok; break; fi
       GUARD1_STATE=strip-failed
       # Un-stage AND restore ONLY this attempt's paths so the retry
       # re-splits clean (never a bare `reset -- tasks/`, which could touch
       # own-task staged state). PER-DISPOSITION (#2126): a single batched
       # pathspec op ABORTS ENTIRELY on one unmatched path — measured
       # (artifacts/probe-guard1-restore.txt part A), a branch-DELETED
       # foreign path (the status `git mv` shape) is unmatched under BOTH
       # probed restore verbs, and the old batched form restored NOTHING
       # (#2087). The split keys on HEAD existence, so the in-HEAD batch
       # can never carry an unmatched pathspec; checkout HEAD restores
       # index AND working tree (the #897 hook-admitted form for a
       # `git -C "$WT"` clause). Absent-from-HEAD paths are DROPPED (that
       # IS their HEAD state), incl. any untracked litter a later shape-1
       # worktree merge would refuse on.
       R_IN=(); R_GONE=()
       for p in "${FOREIGN[@]}"; do
         if git -C "$WT" cat-file -e "HEAD:$p" 2>/dev/null; then R_IN+=("$p"); else R_GONE+=("$p"); fi
       done
       [ "${#R_IN[@]}" -eq 0 ]   || git -C "$WT" checkout HEAD -- "${R_IN[@]}"
       [ "${#R_GONE[@]}" -eq 0 ] || { git -C "$WT" rm -f -q --ignore-unmatch -- "${R_GONE[@]}"
                                      for p in "${R_GONE[@]}"; do rm -f -- "$WT/$p"; done; }
     else
       GUARD1_STATE=ok   # no foreign tasks/ paths — nothing to strip
       break
     fi
   done
   if [ "$GUARD1_STATE" != ok ]; then
     echo "Guard 1: not certified (state=$GUARD1_STATE) after the bounded retry — do NOT merge; route to the merge-failure handling (epm:merge-failed v1)"
     false
   fi
   ```

   The `STRIPPED_FOREIGN` flag is load-bearing: the strip commit above is a
   LOCAL worktree commit, but the safe-case `gh pr merge $MERGE_FORM` below
   merges the PR head ref as it exists on
   `origin/issue-<N>` (server-side), NOT the local worktree HEAD. An unpushed
   strip commit is therefore INVISIBLE to that server-side merge — the
   foreign `tasks/*` reverts would remain in the replayed history and land on
   `main` silently. So when `STRIPPED_FOREIGN=yes`, the safe-case block below
   MUST push the strip commit to the PR head ref BEFORE calling `gh pr merge`.

   This is idempotent at the commit gate (#1280): after a successful strip a
   re-run's three-dot diff can still list an on-main foreign path (HEAD holds
   the OLD pin's content, which differs from the merge-base) — the re-run
   re-checkouts the FRESH snapshot and the `diff --cached --quiet` gate skips
   the commit when main has not advanced, or refreshes the strip to the newer
   pin when it has; branch-ADDED foreign paths drop out of the diff entirely
   once rm-ed. A FAILED trigger diff fails loud (echo + `false`) instead of
   reading as no-foreign-files, leaving `STRIPPED_FOREIGN=no` while the block
   exits non-zero (#1184), and the guard never
   touches THIS task's own `tasks/*/<N>/` folder (the `grep -Ev
   "^tasks/[^/]+/<N>/"` carve-out). Never let a behind-`main` branch revert
   another task's `events.jsonl` / `comments.jsonl` (#458). The `--rebase` merge form below replays
   the branch's commits on top of current `main`, so files the branch never
   committed keep `main`'s version (the `--squash` form lands the same
   own-diff content as one commit) — this is what keeps the clean-result body
   (committed to `main` by `task.py`, never in the worktree) safe across the
   merge.
2. **Status is path-dependent — never flip it to reach this step.** At the
   Step 9b trigger the EXPERIMENT path is parked at `awaiting_promotion`;
   the CODE-CHANGE path is still at `running` BY DESIGN (#1723) — Step 10
   step 6 DEFERS the terminal flip to Step 10d's own
   `Terminal teardown (code-change path only)` sub-section, which runs
   AFTER `epm:merged v1`. That deferral keeps the `/issue-tick <N>` cron
   armed across the merge window and prevents the `completed`-on-an-
   unmerged-branch record `completed_unmerged_pass` flags (#1540/#1653).
   `running` here is EXPECTED, not stale: flipping it early also makes the
   worktree reap-eligible mid-merge (`worktree_audit.py`) — the #2242
   fail-open. Crash safety: EXPERIMENT — pod already terminated, task at a
   user gate, nothing bills; CODE-CHANGE — task still ACTIVE with its tick
   cron armed and `epm:merged` as the idempotency key, so the next
   `/issue <N>` re-enters Step 10d idempotently.
   On a later `/issue <N>` resume: if the PR is already merged AND status
   is still `running` for any reason, auto-advance rather than
   re-dispatching.
3. **Branch-content / non-`main`-base guard.** Compute:

   ```bash
   # MB probe runs FIRST: on an unfetched origin/main the BEHIND count would
   # itself die rc=128 before the hard-stop message could print.
   MB=$(git -C "$WT" merge-base HEAD origin/main) || MB=""
   if [ -z "$MB" ]; then
     echo "GUARD3 HARD-STOP: no merge-base between HEAD and origin/main (unfetched main / unrelated histories)"
     MB_VALID=no
   else
     MB_VALID=yes
     BEHIND=$(git -C "$WT" rev-list --count HEAD..origin/main)   # Guard-3 diagnostic; the
                                                                 # fast-path pre-check below
                                                                 # separately keys on BEHIND > 1000
     # Diagnostic ONLY (never an UNSAFE trigger): did the merge-base land on
     # main's first-parent mainline, or as a merge's SECOND parent (the
     # #1489/#1128 scratch-worktree landing shape)?
     MB_FIRST_PARENT=$(git -C "$WT" rev-list --first-parent origin/main \
       | grep -Fxq -- "$MB" && echo yes || echo no)
   fi
   # TERMINAL STATUS LINE — must remain the LAST statement of this fenced block.
   [ "$MB_VALID" = yes ]
   ```

   `MB_VALID=no` ⇒ UNSAFE + HARD STOP: the block above terminates
   non-zero — fix the fetch / repo state and re-run this guard; NEVER
   route a no-merge-base branch to the artifact-confirmed degrade
   (without a merge-base the own-diff below cannot even be computed).
   The **own-commit content check below is Guard 3's single load-bearing
   UNSAFE trigger, and it runs on EVERY branch** — unconditionally,
   regardless of `BEHIND` and regardless of how the merge-base landed on
   `main`. `MB_FIRST_PARENT` is a recorded diagnostic, NEVER an UNSAFE
   trigger — the first-parent read is false-POSITIVE on the fleet's own
   prescribed #1489/#1128 scratch-worktree merge-form landings (a
   merge-base that entered `main` as a merge's SECOND parent IS an
   ancestor of `origin/main`; #2319/#1144) and false-NEGATIVE on the
   #479 fork-off-unmerged-sibling class (such a fork's merge-base is the
   ordinary mainline commit where the sibling itself forked from
   `main`), so it records HOW the base landed, not whether a rebase is
   safe; it rides the `epm:merged` note as `mb_first_parent: <yes|no>`
   on both landing paths. `BEHIND` alone is NEVER an automatic unsafe
   verdict, and it no longer gates the content check either — in this
   repo every `task.py` marker is a commit (~100+/hr fleet-wide), so a
   same-day, single-own-commit branch routinely reads `BEHIND` in the
   hundreds (#598: `BEHIND=305` tripped the long-retired fixed-200
   threshold, routing `src/` deliverables toward the artifact-confirmed
   path, which cannot carry them). `BEHIND` stays COMPUTED and
   LOAD-BEARING downstream — the fast-path pre-check below keys on
   `BEHIND > 1000`; do not delete the computation as dead. The
   unconditional own-commit content check:

   ```bash
   # The branch's OWN commits (merge-base..HEAD) — when this diff passes the
   # content check, it is exactly what `gh pr merge --rebase` will replay
   # onto main
   # (the `--squash` form lands the same own-diff content as one commit).
   # quotePath=false: each $f below feeds a literal `git log ... -- "$f"`
   # pathspec — a `"`-quoted non-ASCII path matches nothing, non_sync reads
   # empty, and the file is misread as "imported from main" (fail-open).
   git -C "$WT" -c core.quotePath=false diff --name-only origin/main...HEAD   # three-dot form
   ```

   Before judging a workflow-surface path out-of-scope, EXCLUDE files whose ONLY
   branch-side touch is a Step-5a `spec-freshness` sync (the mandated
   `git checkout origin/main -- $SAFE_SPECS` from fetched `origin/main`, NOT a
   branch deliverable).
   This mirrors Step 5a's own intent (line ~1925): a file that has NO non-sync
   branch-side commit is content imported FROM `main`, so it is never an
   out-of-scope regression. Match on the commit SUBJECT line ONLY — a `--grep`
   over subject+body would wrongly exclude a genuine branch edit whose
   BODY mentions the phrase. The exclusion keys
   on the prescribed sync-subject SHAPE `sync workflow-surface specs from` —
   NOT the bare `spec-freshness` token, which a deliverable commit ABOUT the
   sync machinery legitimately carries in its subject (#1789). The
   anchor is carried by the current `issue-<N>: sync
   workflow-surface specs from origin/main (spec-freshness)` (#1747) and the
   historical `issue-<N>: sync workflow-surface specs from main
   (spec-freshness)` (pre-#1747 commits keep the old title). The legacy
   `chore(issue-<N>): spec-freshness sync workflow surface from main` variant
   does NOT carry the anchor and now reads as a branch-side edit — the
   fail-SAFE direction (family dirty → sync skipped / Guard-3 conservative;
   status-quo staleness, never a clobber). Residual: a future deliverable
   subject QUOTING the exact anchor phrase verbatim would still be excluded —
   do not quote it in commit subjects.

   ```bash
   # For each workflow-surface path $f in the own-diff: does it have any
   # branch-side commit whose SUBJECT does NOT contain the prescribed
   # sync-subject anchor "sync workflow-surface specs from"?
   # Emit "<sha> <subject>" per own-commit touching $f, then keep only the
   # non-sync ones. If none remain, the file's only branch-side touches are
   # spec-freshness syncs => imported from main => NON-blocking for Guard 3.
   non_sync=$(git -C "$WT" log --format='%H %s' "$MB"..HEAD -- "$f" \
     | awk 'index($0, "sync workflow-surface specs from") == 0')
   # $non_sync empty   => file imported via spec-freshness sync only => treat as
   #                      NON-blocking (in-scope, imported from main).
   # $non_sync nonempty => a genuine branch-side edit (its subject is not a sync)
   #                      => apply the normal in-scope / out-of-scope judgment.
   ```

   (`git log --format='%H %s'` prints `<sha> <subject>` per commit — the `awk
   index()` keeps only lines whose subject lacks the anchor phrase; the sha is
   a hex string that never contains "sync workflow-surface specs from", so the
   match is effectively subject-scoped. Equivalently
   `git log --format='%s' … | grep -vF 'sync workflow-surface specs from'`.)

   UNSAFE if the own-diff — after the spec-freshness exclusion above — touches
   any foreign `tasks/` path (under `tasks/` but outside `tasks/*/<N>/`) or files
   outside this task's deliverable scope (paths neither the plan nor the code
   review touched). (Paths under `.claude/agent-memory/` — including the
   Guard-0 persist commit — are review-round bookkeeping: always in-scope,
   never an UNSAFE trigger.) If the list is clean — only this task's own deliverables,
   plus any spec-freshness-synced workflow-surface files — the branch is SAFE to
   rebase-merge regardless of `BEHIND`: the rebase replays only these commits,
   and files the branch never committed keep `main`'s version.

   In the unsafe case, do NOT run the safe-case `gh pr merge` (any
   `$MERGE_FORM`) — fall through
   to the **artifact-confirmed merge** procedure below. The Guard 1
   foreign-`tasks/` checkout is necessary but not sufficient: it covers
   `tasks/`, but a branch based on a still-unmerged parent branch also
   carries the parent's stale `src/` and `scripts/`, and a blind rebase
   replays both the parent's `tasks/` rewinds (already handled) AND its
   `src/` / `scripts/` regressions (NOT handled by Guard 1) onto
   `main`. (#479: a branch based on a still-unmerged parent — a blind
   `gh pr merge --rebase` replays the parent's old commits onto
   `main`. The CONTENT CHECK is what catches this class — the
   `origin/main...HEAD` diff carries the whole parent payload,
   out-of-scope by construction, failing the scope judgment. The
   retired first-parent read never was a sound test here in either
   direction: this fork's merge-base is the ordinary mainline commit
   where the parent itself forked from `main`, so the class read `yes`
   in the common shape, while the fleet's own prescribed #1489/#1128
   merge-form landings read `no` on benign branches — #2319/#1144.)

4. **Lost-update refusal (shared workflow-surface files).** A branch
   whose copy of a SHARED workflow-surface file predates a sibling's
   already-merged additions can carry a whole-file snapshot that
   silently DROPS lines that landed on `origin/main` after the branch's
   merge-base — no conflict, no warning, hard to spot in the diff,
   catastrophic when it drops a bundled `workflow_lint.py` check or an
   operational SKILL.md guardrail (#1701 → #1698;
   #1713 encodes this guard as the mechanical backstop). Refuse the
   merge with a loud message when the shape is detected.

   Scope: `scripts/workflow_lint.py`, `.claude/skills/**/SKILL.md`,
   `.claude/rules/*.md`, `.claude/workflow.yaml`, `CLAUDE.md`. Predicate:
   for every branch-touched path in that scope, enumerate the lines
   `origin/main` ADDED since the merge-base (post-merge-base additions
   only — never `main`'s own pre-fork content), then check whether each
   such line is present in the branch's current version of that file
   (`grep -Fxq --` — full-line, fixed-string, so quoting or partial
   substring matches cannot mask a drop). A missing line is by
   definition a main-side addition the branch's snapshot silently
   REVERTED — a legitimate branch DELETION of a pre-existing function
   is NOT this class, because those lines were never main-side
   additions past the merge-base. Kill switch:
   `EPM_SKIP_LOST_UPDATE_GUARD=1` (document the reason on the
   `epm:merged` note when used — e.g. the branch DELIBERATELY reverts
   a merged sibling per a user directive).

   ```bash
   GUARD4_OUT=$(bash scripts/step10d_guards.sh <N> --guard 4 --main-sha "$MAIN_SHA"); GUARD4_RC=$?
   eval "$GUARD4_OUT"
   [ "$GUARD4_RC" -eq 1 ] && false
   ```

   (The extracted script honors `EPM_SKIP_LOST_UPDATE_GUARD=1` FIRST — emits
   `GUARD4=skipped`, exit 0. Otherwise: `--main-sha` is the pinned
   `origin/main` TIP — the Guard-1 capture above (`MAIN_SHA=$(git -C "$WT"
   rev-parse origin/main)`), NOT the merge-base. The helper DERIVES the
   merge-base from it (`git -C "$WT" merge-base HEAD <tip>`; the live
   `origin/main` ref when the flag is omitted) and uses the SAME pinned tip
   as the main side of the add-enumeration, so the pinned and no-flag forms
   are verdict-equivalent by construction (#2428). It iterates the
   branch-touched paths under the fence's actual case glob
   (`scripts/workflow_lint.py|.claude/skills/*|.claude/rules/*|.claude/workflow.yaml|CLAUDE.md`),
   counts pinned-tip-added lines missing from `HEAD:<P>` via
   `grep -Fxq -- "$ADD_LINE"` (the `--` separator protects `-`-leading
   additions) and on any refusal emits `LOST-UPDATE REFUSAL
   (Guard 4, #1713)` on stderr + `GUARD4=refused` +
   `LOST_UPDATE_PATHS=...` on stdout + exit 1; BOTH pass and refusal emit
   `GUARD4_MERGE_BASE=<derived base>` so the `epm:merged` record shows which
   base the verdict used (#2212's vacuous pass was unauditable without it).
   The two-step rc-capture
   form preserves the `false`-in-block-tail halt
   semantics: `eval "$GUARD4_OUT"` populates the caller's `$GUARD4`,
   `$LOST_UPDATE_PATHS`, and `$GUARD4_MERGE_BASE` (the `GUARD4_` prefix keeps
   the eval from clobbering the caller's live `$MB` / `$MAIN_SHA`), and the
   trailing `[ "$GUARD4_RC" -eq 1 ] && false`
   halts the merge attempt at the same point the inline prose did (#1978).)

   **Recovery ordering (#1753; #1727).** When recovering via a
   merge of `origin/main` INTO the branch (instead of the rebase form),
   COMMIT the staged merge BEFORE re-running this guard or the pre-push
   lint gate — the guard's predicate reads `git show HEAD:"$P"` and the
   gate sha-binds its verdict to HEAD, so staged-but-uncommitted merge
   content still reads as dropped (the #1727 false lost-update / "STILL
   UNMERGED" read).
   And any size-ratchet cap the recovery re-writes is computed from the
   POST-merge (landing) bytes, never the pre-merge branch tip (#1727 — see the
   landing-bytes bullet in the gate section below).

   Non-workflow-surface files stay covered by Guards 1-3 alone; Guard 4
   focuses the scan on the files whose silent-revert blast radius is
   fleet-wide.

5. **Sibling merge-sequencing hold + proactive pre-resolution (#1757).**
   Runs ONCE per Step 10d invocation (never inside per-attempt retry
   shapes). Half (i) runs at Step 10d entry; half (ii) runs AFTER Guard 0
   (the agent-memory pathspec commit) — a dirty tree aborts an in-worktree
   merge (the exact #906 shape Guard 0 exists to clean), so (ii) first runs
   the idempotent Guard 0 block, then merges. Scan this task's events for
   `merge-hold-candidate` RECORDS — anchored on the Step 2b record shape
   (the token immediately followed by its named `sibling=<M>`), never a
   bare substring: a note that merely MENTIONS the token (e.g. a heartbeat
   reporting zero candidates) must not fire the guard (#2301):

   ```bash
   grep -E 'merge-hold-candidate sibling=[0-9]+' "$(uv run python scripts/task.py find <N>)/events.jsonl"
   ```

   No candidate record → Guard 5 is a no-op (one grep). Otherwise, per named
   sibling `<M>` (dedup):

   - **(i) Bounded hold.** Read live state via `task.py view <M> --json`.
     No hold when: its events carry `epm:merged` (any form —
     `artifact_confirmed` counts); OR its status is in {completed,
     archived, blocked, on_hold} (a parked/blocked sibling is not landing
     soon); OR its state is UNREADABLE (`task.py find <M>` fails — treat
     as no-hold, never a 45-min no-op); OR a PRIOR `merge_hold` disposition
     note for `<M>` with `outcome=cap-expired` exists on this task's
     events (sticky — a stuck sibling never re-triggers the hold on
     re-entry). Otherwise (live at `reviewing`-or-later, unmerged): post
     ONE `[long-phase-heartbeat] step10d-merge hold sibling=<M> (#1757)`
     progress note, then wait via the sanctioned `Monitor` until-loop
     shape (load the deferred schema first — `ToolSearch("select:Monitor")`),
     elapsed-capped at 2700 s (one 45-min gate cycle), re-resolving the
     sibling's folder each poll (status moves relocate it):

     ```bash
     until grep -qF '"epm:merged"' "$(uv run python scripts/task.py find <M> 2>/dev/null)/events.jsonl" 2>/dev/null \
           || [ $SECONDS -gt 2700 ]; do sleep 60; done
     ```

     (NEVER a foreground Bash sleep-loop — the 600 s tool cap kills it and
     the sleep-chain shapes are hook-blocked; Monitor is the sanctioned
     poll carrier here.) On expiry, record `outcome=cap-expired` and
     proceed — the hold is bounded by construction; a mutual hold (two
     siblings each naming the other) resolves at cap expiry on both sides.
   - **(ii) Proactive pre-resolution (the load-bearing half — fires with
     any candidate note, INCLUDING when the sibling already merged).**
     Sequenced AFTER Guard 0's agent-memory commit (run the idempotent
     Guard 0 block first if not yet run this invocation).
     `git -C "$WT" fetch origin main --quiet`, then probe for the
     predicted conflict without touching the working tree, PATH-SCOPED
     to each candidate note's own `path=<file>` field (dedup paths
     across notes; a candidate note MISSING its `path=` field → treat
     as conflicted — the degrade below). Per path, materialize the
     three blob versions and run read-only three-way `git merge-file`
     (ancient, version-portable plumbing — no git-version branch
     needed; `-p` writes the merged result to stdout, inputs untouched):

     ```bash
     MB=$(git -C "$WT" merge-base HEAD origin/main)
     git -C "$WT" show "$MB:<file>"          > /tmp/issue-<N>-mh-base   # any show failing
     git -C "$WT" show "HEAD:<file>"         > /tmp/issue-<N>-mh-ours   # (added/deleted/renamed
     git -C "$WT" show "origin/main:<file>"  > /tmp/issue-<N>-mh-theirs # on a side) -> CONFLICTED
     git merge-file -p /tmp/issue-<N>-mh-ours /tmp/issue-<N>-mh-base /tmp/issue-<N>-mh-theirs \
       > /dev/null 2>&1
     # rc 0 = clean; rc > 0 (= conflict count) = CONFLICTED; rc < 0 (shell: 255) = error -> CONFLICTED
     ```

     (A whole-tree probe — either `git merge-tree` form — is
     deliberately NOT used: on this repo it reads CONFLICTED on
     essentially every real merge — `tasks/` git-mvs print `removed in`
     stanzas and events.jsonl notes quoting conflict markers trip a
     `<<<<<<<` grep.) Ambiguous or
     unavailable probe output → treat the candidate as conflicted — fail
     toward the proactive resolve, never toward a doomed server-side
     refusal. ALL probed paths clean → proceed exactly as today
     (Guards 0-4 + the normal merge form; experiment branches keep
     `--rebase`). Any path CONFLICTED → resolve
     proactively IN THE WORKTREE via the EXISTING merge-conflict recovery
     machinery (capture ONE `MAIN_SHA`, `git -C "$WT" merge "$MAIN_SHA"`,
     the mechanical foreign-tasks/figures passes + residual-conflict
     subagent dispatch, commit, post-resolution certification), then
     re-run the pre-push workflow-lint gate (the SHA-bound verdict
     re-binds to the post-merge tip — the #1753 recovery ordering) and
     take the `--squash` merge form (the branch now carries a merge
     commit — Known failure shape 1).
   - Record the disposition in the `epm:merged` / `epm:merge-failed` note:
     `merge_hold: sibling=<M> waited=<mins> outcome=<sibling-merged|cap-expired|no-hold>`
     and `pre_resolve: <clean|conflicted-resolved|probe-unavailable>`
     (omit both lines when no candidate note exists). Same behavior in
     interactive and autonomous sessions; auto-continue, never a gate.

#### Pre-merge divergence delta gate (#1771→#2201)

Runs after Guard 5 and before the fast-path pre-check, every Step 10d
invocation (both trigger points). Step 5a disclosed main-side divergence to
the reviewers each round; this gate covers the residual UNREVIEWED at merge
time — paths never in the final review round's disclosure, PLUS disclosed
paths that main changed AGAIN after the reviewed main SHA (a pathname-only
subtraction would let a re-touched hot-registry file merge ungated — the
healthy-branch survivors measured at plan time are exactly that file
class). An unreviewed semantic collision can textually merge clean, so
neither Guard 4 (line-revert refusal) nor the reactive recovery (textual
conflicts) would surface it.

```bash
DIVOUT=/tmp/issue-<N>-divergence-merge.txt
NEWLIST=/tmp/issue-<N>-divergence-new.txt
rm -f "$DIVOUT" "$NEWLIST"   # stale-output hygiene: a failed invocation must
                             # never leave a prior run's list to compute from
DIV_OUT=$(bash scripts/step10d_guards.sh <N> --guard divergence --out "$DIVOUT"); DIV_RC=$?
eval "$DIV_OUT"              # two-step rc-capture (Guard-4 caller form)
if [ "$DIV_RC" -eq 0 ]; then
  # Review-time record = the LATEST per-round probe note:
  LASTNOTE=$(uv run python scripts/task.py view <N> --json | uv run python -c '
import sys, json
rows = [e.get("note","") for e in json.load(sys.stdin).get("events",[])
        if e.get("kind")=="epm:progress" and e.get("note","").startswith("[divergence-probe] r")]
print(rows[-1] if rows else "")')
  REVSET=/tmp/issue-<N>-divergence-reviewed.txt
  printf '%s' "$LASTNOTE" | sed -n 's/.*files=//p' | tr ',' '\n' | sed '/^$/d' | sort -u > "$REVSET"
  REV_MAIN=$(printf '%s' "$LASTNOTE" | grep -oE 'main=[0-9a-f]+' | head -1 | cut -d= -f2)
  sort -u "$DIVOUT" > /tmp/issue-<N>-divergence-cur.txt
  if [ -z "$LASTNOTE" ] || printf '%s' "$LASTNOTE" | grep -q ' ERROR ' \
     || [ -z "$REV_MAIN" ] || ! git -C "$WT" cat-file -e "$REV_MAIN^{commit}" 2>/dev/null; then
    # FAIL-CLOSED: no clean, parsable reviewed record -> the FULL probe set is unreviewed.
    cp /tmp/issue-<N>-divergence-cur.txt "$NEWLIST"
  else
    # CONTENT-KEYED delta (never pathname-only): (probe MINUS reviewed paths)
    # UNION (probe INTERSECT paths main changed after the reviewed main sha):
    comm -13 "$REVSET" /tmp/issue-<N>-divergence-cur.txt > /tmp/issue-<N>-div-a.txt
    # Materialize the reviewed->current main diff with an rc check (review r1
    # MF-1b — never a bare pipeline): a failed diff would exit through
    # sort|comm rc 0, read as an EMPTY set B, and let a previously-disclosed
    # re-touched file merge as "reviewed". quotePath=false matches the
    # helper's producers (MF-2: a C-escaped non-ASCII path in this list
    # misses comm -12 against the raw current set -> NEW=empty).
    if git -C "$WT" -c core.quotePath=false diff --name-only "$REV_MAIN" "$MAIN_SHA" \
        > /tmp/issue-<N>-div-xy.txt; then
      sort -u /tmp/issue-<N>-div-xy.txt \
        | comm -12 - /tmp/issue-<N>-divergence-cur.txt > /tmp/issue-<N>-div-b.txt
      sort -u /tmp/issue-<N>-div-a.txt /tmp/issue-<N>-div-b.txt > "$NEWLIST"
    else
      # FAIL-CLOSED on the masked-producer failure -- the same branch the
      # missing/ERROR/unparsable review record takes (cap-bounded).
      cp /tmp/issue-<N>-divergence-cur.txt "$NEWLIST"
    fi
  fi
  NEW_COUNT=$(grep -c . "$NEWLIST" || true)
fi
```

- `DIV_RC` != 0 → **documented fail-open, never silent:** post
  `[divergence-probe] step10d ERROR rc=<rc>` (epm:progress), skip the delta
  computation entirely (the list file was removed — never compute a count
  from a stale file), PROCEED with today's machinery, and record
  `diverged_on_main: disposition=probe-error rc=<rc>` on the `epm:merged`
  note. Posture rationale: this gate is a DISCLOSURE instrument layered
  over Guards 1-5 + the reactive recovery — its failure reverts the merge
  to the pre-#2201 protection level rather than removing a data-safety
  mechanism; Guard 3 has already HARD-STOPPED the dominant infra cause (no
  merge-base) before this gate runs, so the residual rc != 0 population is
  transient git failure, where a HOLD would wedge merges with no
  implementer signal (the same bounded-not-wedged philosophy as Guard 5's
  45-min cap). FAIL-CLOSED stays the rule where the probe itself is
  HEALTHY but the review record is missing/ERROR/unparsable (the full-set
  branch above): there the unreviewed set is computable, so it blocks.
- `DIV_RC` = 0 and `NEW_COUNT` = 0 → PROCEED exactly as today. When
  `DIVERGED_COUNT` > 0, record `diverged_on_main: count=<n>
  disposition=reviewed` on the `epm:merged` note (beside `merge_hold:` /
  `pre_resolve:`); omit the line when the probe read clean/skipped.
- `NEW_COUNT` > 0 → **cap check first (durable across crash-resume):** if a
  `[divergence-probe] step10d new=` note NEWER than the latest per-round
  `[divergence-probe] r<...>` note already exists on this task, the one
  reconciliation dispatch for this merge cycle is SPENT — PROCEED with
  `diverged_on_main: count=<n> disposition=proceed-after-cap` on the merged
  note, never a loop (a fresh review round re-posts the per-round note,
  which re-arms the cap). Otherwise: do NOT run any merge form yet. Post
  `[divergence-probe] step10d new=<comma-list>` (epm:progress — this note
  IS the spent-cap key, posted BEFORE dispatching), then dispatch ONE
  implementer reconciliation round: the brief names the newly-diverged
  paths and points at both-side deltas BY REFERENCE (`git log --oneline
  $MB.."$MAIN_SHA" -- <path>` / `$MB..HEAD -- <path>` — `$MAIN_SHA` is the
  helper-emitted probed sha, so the inspected main state and the measured
  state coincide), instructs an
  in-worktree `git -C "$WT" merge "$MAIN_SHA"` (the same merge form Guard
  5(ii) and the conflict recovery use; the merge target is that SAME
  pinned sha, so the merge target and the measured state coincide too),
  SEMANTIC reconciliation of the named files
  (main's change and the round's change both preserved, or the
  contradiction resolved with a stated choice), and a commit. Then re-run
  the Step 5 review round on the reconciliation commit (ordinary round
  machinery — its Step 5a probe re-posts the per-round note), and re-enter
  Step 10d: the in-worktree merge advanced the merge-base, so the delta
  recomputes against fresh state, and the branch now carries a merge
  commit → take the `--squash` merge form (Known failure shape 1).

Composes WITH the existing machinery, never replaces it: textual conflicts
still route to the Known failure shapes + merge-conflict recovery; Guard
4's lost-update refusal is unchanged; this gate only adds the
implementer-in-the-loop SEMANTIC pass for unreviewed divergence. Divergence
both disclosed at the final review round AND unchanged since the reviewed
main SHA never blocks: measured 2026-08-19, refined sets of 1-2 files exist
on healthy live branches, so a hard block on any non-empty set would have
held 3 of 5 healthy branches. Comma-bearing paths: the per-round `files=`
token is comma-delimited, so a path containing `,` lands whole in the
current probe set but fragmented in `$REVSET` and deterministically
re-flags as NEW on every merge — fail-closed and bounded by the
one-dispatch cap (accepted; no serialization redesign).

#### Fast-path routing pre-check (workflow-fix / small-ADDED-diff far-behind branches)

Run this AFTER guards 1-3 and BEFORE the safe-case `gh pr merge $MERGE_FORM`
call. For a workflow-fix / small-diff branch that is very far behind `main`,
a server-side `--rebase` predictably conflicts on churn even after Guard 1
strips foreign folders (GitHub replays the branch's own commits across
thousands of intervening main commits, and cannot use this repo's
`merge=union`). When the branch's OWN diff is small, entirely in-scope, AND
consists ONLY of ADDED files, skip the doomed server-side merge and route
DIRECTLY to the surgical additive checkout below.

**Why the ADDED-only conjunct is load-bearing (do NOT drop it).** The
surgical additive checkout does a WHOLESALE `git checkout issue-<N> -- <path>`
(the "One or more deliverables missing" branch, ~line 7080), which OVERWRITES
each listed path with the branch tip's copy. For a file the branch MODIFIED
that `main` also advanced (very likely on a 1000+-behind branch), that
overwrite silently discards `main`'s newer content with NO conflict surfacing
— a silent-wrong merge. Restricting the fast-path to ADDED-only files means
the surgical checkout only ever CREATES files that do not yet exist on
`main`, so it can never clobber a concurrently-advanced one. A branch that
MODIFIES a workflow-surface file (status M) is NOT fast-path-eligible and
takes the ordinary `gh pr merge $MERGE_FORM` path, whose server-side 3-way merge
either merges main's changes cleanly or surfaces a real conflict for the
recovery sub-procedure. (This is exactly why #787 itself — which MODIFIES
`SKILL.md` — is not fast-path-eligible.)

```bash
# Fast-path predicate — ALL of:
#  (a) task is kind:infra AND tagged wf-fix (a workflow-fix branch), AND
#  (b) BEHIND > 1000 (branch predates significant main churn), AND
#  (c) the branch's OWN diff (after the agent-memory filter below) touches
#      BETWEEN 1 and 15 files — the `-ge 1` LOWER bound is load-bearing: the
#      memory filter can EMPTY the list (a branch whose entire own-diff is
#      Guard-0 memory commits), and an empty list must NEVER fast-path — the
#      surgical `--diff-filter=A` list is then empty too, and an empty-input
#      xargs would run `git checkout issue-<N> --` with NO pathspec, which is
#      a BRANCH SWITCH of the shared repo root (`xargs -r` at the checkout is
#      the depth-2 defense), AND
#  (d) every touched file is in-scope: this task's own paths, workflow
#      surface, .gitattributes, or the methodology doc — NO shared src/ or
#      scripts/ additions (those need the full rebase to land), AND
#  (e) EVERY touched file is status A (Added) — no M (Modified), D (Deleted),
#      R (Renamed). A modified file would be clobbered wholesale by the
#      surgical checkout below.
KIND=$(uv run python "$REPO_ROOT/scripts/task.py" view <N> --json | \
  uv run python -c 'import sys,json; d=json.load(sys.stdin); fm=d.get("frontmatter",{}); print(fm.get("kind","")); print(" ".join(fm.get("tags",[])))')
TASK_KIND=$(printf '%s\n' "$KIND" | sed -n 1p)
TASK_TAGS=$(printf '%s\n' "$KIND" | sed -n 2p)
# Three-dot: the branch's OWN commits only (merge-base..HEAD) — never files
# main advanced but the branch never touched. Name-status so we can gate on A.
# Exclude .claude/agent-memory/ — the Guard-0 persist commit MODIFIES memory
# files, which must not fail the ADDED-only predicate (e). Memory edits land
# via the ordinary rebase (or stay on the PR branch for the deferred full
# rebase); the surgical checkout never carries a modified file anyway.
mapfile -t OWN_NS < <(git -C "$WT" diff --name-status origin/main...HEAD \
  | grep -vE $'\t\\.claude/agent-memory/[^\t]*$' || true)
# End-anchored on the LAST path field so an R-status rename whose SOURCE is
# a memory path but whose destination is elsewhere cannot dodge predicate
# (e) via over-filtering (Guard 0 itself never produces renames).
N_FILES=${#OWN_NS[@]}
IN_SCOPE=yes
ADDED_ONLY=yes
for line in "${OWN_NS[@]}"; do
  st=${line%%$'\t'*}          # status letter (A / M / D / R100 / ...)
  f=${line#*$'\t'}            # path (for a rename this is the source; fine —
                              # a rename fails ADDED_ONLY below regardless)
  [ "$st" = "A" ] || ADDED_ONLY=no
  case "$f" in
    tasks/*/<N>/*|figures/issue_<N>/*|eval_results/issue_<N>/*|eval_results/issue_<N>_*/*|ood_eval_results/issue_<N>/*) ;;
    .claude/*|CLAUDE.md|.gitattributes|docs/methodology/issue_<N>.md) ;;
    *) IN_SCOPE=no ;;
  esac
done
FAST_PATH=no
if [ "$TASK_KIND" = "infra" ] \
   && printf '%s' "$TASK_TAGS" | grep -qw 'wf-fix' \
   && [ "$BEHIND" -gt 1000 ] \
   && [ "$N_FILES" -ge 1 ] \
   && [ "$N_FILES" -le 15 ] \
   && [ "$IN_SCOPE" = "yes" ] \
   && [ "$ADDED_ONLY" = "yes" ]; then
  FAST_PATH=yes
fi
```

If `FAST_PATH=yes`: SKIP the safe-case `gh pr merge $MERGE_FORM` call and jump straight to
the **surgical additive checkout** (the "One or more deliverables missing"
branch of the artifact-confirmed procedure below). The surgical checkout
lands this branch's own ADDED files onto `main` directly, with no rebase.
Post `epm:merged v1` with `{artifact_confirmed: true, full_rebase_deferred:
true, surgical_checkout: true, fast_path: true, reason: "wf-fix branch
BEHIND=<BEHIND> > 1000, own diff <=15 in-scope ADDED-only files — skipped
doomed server-side rebase", files: [...]}`.

If `FAST_PATH=no`: proceed to the safe-case `gh pr merge $MERGE_FORM` (or the
artifact-confirmed path if Guard 3 said UNSAFE) exactly as before — this
pre-check adds NO new behavior for normal branches. A branch that MODIFIES a
workflow-surface file (status M ⇒ `ADDED_ONLY=no`) is deliberately not
fast-pathed; it takes the ordinary `$MERGE_FORM` merge.

#### Pre-push workflow-lint gate (runs before every merge form lands)

The gate is an INLINE recipe (the fenced blocks in this subsection, run via
bg-Bash) — there is NO helper script; do not compose a
`.claude/skills/issue/step10d_lint_gate.sh` (or similar) path, it does not
exist (#1720's session invoked exactly that phantom path).

#931 merged a workflow-lint offender to `main`, breaking
`tests/test_workflow_lint.py` on pristine trunk fleet-wide for most of a day
(5 sessions burned rounds classifying it as pre-existing).
#1147 adds a mapped invariant-test leg to the same gate: dependency-mapped
payloads (the selector's full map — GLOB_SCAN_TESTS + rules-pin (#1496) + the
src/scripts import/literal/stem dependency arms (#1573), WORKFLOW_INVARIANT
members excluded) previously landed with
zero pytest on the experiment auto-merge path (Step 9c is
code-change-kinds-only) — a channel through which #1144's thread-caps offenders
accreted; sampled offenders also landed via direct-to-main
free-analysis/analyzer commits, which this gate does NOT cover (see the Step
9a-ter follow-up) (#1460: now covered by the Step 9a-ter § Inline payload lint gate). Gate the merge payload on the lint + the mapped invariant
tests BEFORE anything lands:

- **Trigger (cheap; artifact-only merges skip).** Run the gate ONLY when the
  branch's own three-dot diff (`git -C "$WT" diff --name-only
  origin/main...HEAD`, computed after guards 0-3) touches any path OUTSIDE
  the artifact-only set (`tasks/`, `figures/`, `eval_results/`,
  `ood_eval_results/`, `raw/`, `data/`, `docs/methodology/`). The lint's
  no-flags default run walks `.claude/**`, `CLAUDE.md`, `scripts/`, and
  `src/`, so any code-bearing payload is in scope.
- **Run a LANDING-TREE lint copy, both legs — no-flags bundle PLUS the parity leg.**
  The gate builds ONE ephemeral landing tree in /tmp (`git archive
  origin/main` over the lint-scanned cones), runs the BASELINE legs from
  that tree's own lint copy BEFORE the payload overlay, then overlays the
  branch's own-diff payload from the branch tip and runs the GATED legs
  from the SAME copy (#1212 — one lint vintage, trees differing only by
  the payload) — with the #1456 exception: a payload-touched
  `workflow_lint.py` is 3-way-merged for the gated legs (see the overlay
  step). `workflow_lint.py` derives its scan root from `__file__`
  (not cwd), so the gate-tree copy scans the gate tree; a plain non-git
  /tmp dir is a supported scan root (the root-guard hook pins `REPO=` to
  an absolute path, and `_other_worktree_prefix` is pure path-string
  logic).
  The no-flags default run does NOT bundle the asks / autonomous-asks /
  references / tables / status-labels checks (their `main()` branches lack
  `or no_flags`), yet `tests/test_workflow_lint.py` subprocess-runs those
  too — so trunk-pytest parity takes BOTH invocations. Measured wall
  ~4.5-6 min (no-flags) + ~1.4 s (parity leg) + ~1-2 s gate-tree
  construction on the shared VM; WARNs do not fail (PASS = exit 0 on
  both). The two leg pairs + TG legs total ~9-12+ min on an IDLE VM, but
  **30-40 min under typical fleet load (3+ concurrent gates)** — measured
  (#1690/#1694/#1711). Size any
  wall-time-derived fence off the LOADED range, not the idle one.
  Projected wall STRUCTURALLY exceeds the 600s bg-Bash tool cap
  (`run_in_background=true` calls carry the same 600 000 ms ceiling as
  foreground), so the executable block below launches DETACHED using the
  same setsid + pid/log/rc-file breadcrumb shape as Step 9c 1b (see the
  1b rationale — a bg-Bash inline chain is harness-killed at the cap
  BEFORE the rc-write, the #991/#996/#1245/#1893 kill class; only the
  detached shape survives past the cap). The outer bg-Bash launches the
  detached unit, captures the pid, and exits in seconds; the inner
  workload runs in its own session decoupled from the launcher's kill
  domain and writes its rc file at its OWN exit. After launching, END
  THE TURN — wait via the Monitor until-loop keyed on the single-flight
  probe below, or the `/issue-tick <N>` backstop cron re-wake; repeated
  `TaskOutput(block=true, timeout=600000)` polls of the detached job are
  the banned sleep-chain shape (#1984). Read the verdict in a FRESH
  foreground call from the FILE (completion-read below).

  **Single-flight probe (#1606) — before (re)launching this gate, including
  every "re-run the gate ONCE" recovery path.** Probe
  `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --pattern 'issue-<N>-lint-gate'`
  (self-/ancestor-excluding — exit 0 = clear, 3 = live foreign match; the
  workload SCRIPT path `/tmp/issue-<N>-lint-gate.sh` rides the detached
  workload's argv for its WHOLE life under the #2115 script-file launcher,
  and the unanchored pattern also matches the gate-tree tokens in the
  tar/lint legs' child argvs plus legacy inline-form launches (the tree
  token is a superstring of this pattern), so the probe is
  exact-issue-scoped with NO CLEAR window mid-run — #2256: the tree-only pattern read
  CLEAR mid-run and false-fired a death read on a healthy gate; the
  completion-read's recovery arm keeps its
  bracketed raw-pgrep form — it wants the pid list). Exit 3 = this issue's
  gate is STILL RUNNING: do NOT relaunch — the
  stale-verdict `rm -f` below would clobber the live run's verdict. WAIT or
  reap per the Step 9c 1b single-flight statement, and key any improvised
  wait on **process exit** (the probe exiting 0 — CLEAR), never on
  verdict-file existence alone (CLAUDE.md § Monitoring re-run discipline).
  After the probe reads CLEAR and BEFORE any (re)launch, re-run the same
  Step 5a family-atomic block — the 9c pre-gate spec-freshness re-sync
  (#1742/#2006) binds every gate re-launch here too (a BINDING reference —
  never a third inlined `FAMILY_OF` copy).

  Then the **Gate-fleet arbitration (#1962)** probe, per the Step 9c 1b
  canonical paragraph:
  `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>`
  — exit 3 ⇒ bounded queue (sleep 60, elapsed cap 2700 s), then launch
  anyway with the `[gate-fleet]` cap-expired line (fail-open).

  ```bash
  # EXECUTABLE gate — forms (i) safe case and (ii) recovery share this block
  # DETACHED launcher — the whole gate workload below (baseline lint legs,
  # payload overlay, gated lint legs, TG mapped-invariant legs, subtract,
  # verdict, sha-bind) runs as ONE detached unit via the § Harvest
  # self-harvest chaining shape (§ Detached VM-side long compute phases):
  # the workload is COMPOSED to /tmp/issue-<N>-lint-gate.sh with the WRITE
  # TOOL as its own prior step (a script FILE because the workload's many
  # awk/sed single-quoted blocks would need escape-heavy quoting inside an
  # inner `bash -c '...'` string; the Write tool and NEVER a
  # `cat > ... <<'EOF'` heredoc inside the launcher Bash call — a heredoc
  # body rides the entire multi-KB workload as Bash tool-call argv through
  # the harness transport, the #2115 forever-pending-dispatch stall
  # surface, and the PreToolUse guards scan the full argv including
  # heredoc bodies, #1756), then setsid-nohup-launched from within an outer `bash -c`
  # wrapper — the outer bg-Bash call (run_in_background=true) captures
  # `$!` as the workload pid (`PYTEST_PID` below); the trailing
  # `echo $? > /tmp/step9c-lint-rc-issue-<N>` at the END of the script
  # binds the script's exit into the SAME unit (rc-write inside the
  # session-decoupled unit, not spliced after the outer bg-Bash), so the
  # 600s bg-Bash tool cap can NEVER kill the workload — the launcher
  # bg-Bash exits in seconds after capturing the pid, and the detached
  # workload writes its rc + verdict at its OWN exit from a session outside
  # the launcher's kill domain. NEVER splice a top-level `; echo $? > rc &
  # echo $!` after `2>&1` — that parses as three commands where the gate
  # workload runs FOREGROUND inside the $( ) capture, so the outer bg-Bash
  # STILL dies at the 600s tool cap (§ Harvest NEVER-splice rule; the exact
  # #991/#996/#1245/#1893 failure this recipe exists to fix). The
  # `harvest=/tmp/issue-<N>-lint-verdict.txt` breadcrumb names the durable
  # verdict path the § Successor / re-entry rule probes.
  #
  # Canonical launcher shape (the outer bg-Bash body — the workload verbatim
  # below is the script this launches):
  #   LINT_GATE_SCRIPT=/tmp/issue-<N>-lint-gate.sh
  #   STEP 1 — compose the script with the Write tool (its own tool call,
  #   BEFORE the launcher bg-Bash; never a heredoc in the Bash call):
  #     Write(file_path=$LINT_GATE_SCRIPT, content=
  #       #!/usr/bin/env bash
  #       ... [the workload body verbatim from `# earlyoom-protect the gate`
  #           down to and incl. `cat /tmp/issue-<N>-lint-verdict.txt`] ...
  #       echo $? > /tmp/step9c-lint-rc-issue-<N>
  #     )
  #   STEP 2 — the launcher-only bg-Bash (argv stays tiny):
  #   chmod +x "$LINT_GATE_SCRIPT"
  #   # trailing "$WT": unused by the script; rides the detached workload's argv
  #   # so worktree_audit's cwd/argv liveness harvest keeps the worktree for the
  #   # gate's whole life (#2246 item 1).
  #   PYTEST_PID=$(bash -c "setsid nohup env WT=\"$WT\" REPO_ROOT=\"$REPO_ROOT\" \
  #     OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  #     bash '$LINT_GATE_SCRIPT' \"$WT\" < /dev/null > /tmp/issue-<N>-lint-gate.log 2>&1 & echo \$!")
  #   ps -p "$PYTEST_PID" -o args= | head -1
  #   bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PYTEST_PID" >/dev/null \
  #     && LINT_GATE_CHOOM=ok || LINT_GATE_CHOOM=failed
  #   # ONE bounded retry on failed sweep (§ Detached VM-side long compute
  #   # phases; #1315 — a python3 child forked after the launch pid's choom
  #   # applies inherits nothing until the sweep is re-run):
  #   if [ "$LINT_GATE_CHOOM" = "failed" ]; then
  #     for _ in $(seq 1 30); do
  #       if pgrep -s "$PYTEST_PID" -a 2>/dev/null | grep -qE 'python3?'; then break; fi
  #       sleep 1
  #     done
  #     bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PYTEST_PID" >/dev/null \
  #       && LINT_GATE_CHOOM=ok \
  #       || { LINT_GATE_CHOOM=failed; echo "[warn] session choom re-sweep failed — lint gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  #   fi
  #   echo "[step10d] lint-gate detached pid=$PYTEST_PID log=/tmp/issue-<N>-lint-gate.log rc=/tmp/step9c-lint-rc-issue-<N> harvest=/tmp/issue-<N>-lint-verdict.txt choom=$LINT_GATE_CHOOM"
  #
  # (The workload verbatim body below runs inside that detached unit. The
  # verdict is PERSISTED to a file because fenced bash blocks are separate
  # shell invocations: the binding sites consume the FILE, never a shell
  # variable. Form (iii) inlines the SAME trigger/normalize/subtract/verdict
  # steps around its checkout — see the surgical block.)
  # earlyoom-protect the gate (#1045 recipe, #1211; FAIL-OPEN — a choom failure
  # never blocks the gate and never touches the verdict logic): the lint legs
  # (~4.5-6 min python each) + the mapped pytest legs match this VM's earlyoom
  # --prefer regex (+300 badness) — the designated victim under fleet memory
  # pressure (#1143). The detached workload self-chooms via
  # `sudo -n choom -n -600 -p $$` below, and the outer launcher additionally
  # sweeps the workload's session with `pgrep -s $PYTEST_PID | xargs choom`
  # (session-wide; children inherit; ONE bounded retry per § Detached
  # VM-side long compute phases): every child forked after this line
  # inherits adj=-600.
  sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && LINT_GATE_CHOOM=ok \
    || { LINT_GATE_CHOOM=failed; echo "[warn] choom failed — lint gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  echo "[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM"
  # Stale-verdict rm (Step 9c pre-rm parity): a verdict file present at
  # completion must provably come from THIS run — missing-after-completion
  # is then an unambiguous died-mid-run diagnostic. The #1041 same-tip
  # retry re-enters the merge CONDITIONAL, never this block, so the
  # surviving-verdict retry path is untouched.
  rm -f /tmp/issue-<N>-lint-verdict.txt
  # TRIGGER — materialize the own-diff FIRST and check the diff's OWN exit:
  # piped straight into grep, a FAILED `git diff` (bad ref, no merge-base)
  # is indistinguishable from an empty diff and would fail OPEN as an
  # artifact-only skip. (`set -o pipefail` cannot fix this form: `grep -q`
  # exits at first match and SIGPIPEs the producer, and the else branch
  # would still misread any nonzero as artifact-only.)
  if ! git -C "$WT" -c core.quotePath=false diff --name-only origin/main...HEAD > /tmp/issue-<N>-own-diff.txt; then
    # Failed trigger diff — the gate cannot classify the payload; fail CLOSED.
    echo crash > /tmp/issue-<N>-lint-verdict.txt
  # Classifier consumes grep's OUTPUT (non-empty => code-bearing payload),
  # never a `-q -v` exit status: a ugrep-shadowed shell returns rc=1 on
  # selected non-matching lines under -qv and silently disarmed this gate
  # as skip-artifact-only on a code-bearing payload (#928 -> #1125).
  elif [ -n "$(grep -vE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
      /tmp/issue-<N>-own-diff.txt)" ]; then
    # GATE TREE (#1212): ONE ephemeral tree, TWO phases. Phase 1 (BASELINE)
    # lints the PAYLOAD-FREE landing base — origin/main's lint-scanned
    # surface, archived to /tmp — with origin/main's OWN lint copy
    # (workflow_lint.py derives its scan root from __file__). Phase 2 (GATED)
    # overlays the branch's own-diff payload onto the SAME tree and re-lints
    # with the same copy. Both legs share ONE lint vintage on trees differing
    # ONLY by the payload, so NEW = gated − baseline is payload-caused BY
    # CONSTRUCTION: kills the #1112 vintage false-blocks (stale branch linter;
    # branch scripts/ tree predating a main-referenced helper), stale
    # non-payload files vs main's newer checks, root dirt/lag in the compare,
    # and the moving-main inter-leg race — and ENFORCES checks added on main
    # after the branch forked (the old path-(i) residual, upgraded
    # deliberately: a payload violating a post-fork check now BLOCKS, the
    # #931 class). $WT and the repo root are never written; no commits are
    # created, so the verdict's sha-bind is unaffected. Payload files come
    # FROM the branch tip: a branch whose own diff touches a lint HELPER has
    # its OWN copy exercised on the gated legs — it IS the payload. EXCEPTION
    # (#1456): workflow_lint.py ITSELF is 3-way-MERGED for the gated legs
    # (branch ⊕ merge-base ⊕ archived origin/main — the content a rebase
    # would land on trunk), so main's ratchet raises can't false-block a
    # drifted branch lint copy (#1366/#1411); merge failure falls back to
    # the branch copy with a loud WARN (residual (a)). Construction
    # failures fail CLOSED via GT_RC in the crash arm.
    # The archive pathspec set must cover workflow_lint.py's scan/target
    # surface (.claude/ CLAUDE.md scripts/ src/ tests/ docs/ — the #1154
    # marker-recipe pins read docs/); a false block naming a path OUTSIDE
    # this set means the linter grew a new scan root — extend the set here.
    # check_prod_import_lockfile (#2253) reads uv.lock + pyproject.toml at
    # the tree ROOT, so BOTH manifests ship in the archive (pinned by
    # tests/test_issue_skill_gate_tree_pathspec.py).
    GT=/tmp/issue-<N>-lint-gate-tree
    GT_RC=0
    timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet || true  # bounded: a hung fetch degrades to origin/main staleness, never a wedged gate
    { rm -rf "$GT" && mkdir -p "$GT"; } || GT_RC=1
    ( set -o pipefail; git -C "$WT" archive origin/main -- \
        .claude CLAUDE.md scripts src tests docs pyproject.toml uv.lock \
      | tar -x -C "$GT" ) || GT_RC=1
    [ -f "$GT/scripts/workflow_lint.py" ] || GT_RC=1   # construction sanity
    # BASELINE legs (payload-free landing base — phase 1, BEFORE the
    # overlay). Per-leg exit codes ARE captured:
    # only the baseline's normalized failure LINES enter the compare, but a
    # baseline CRASH (rc>1, or rc!=0 with ZERO `workflow_lint:` lines) makes
    # the compare itself untrustworthy — that fails CLOSED via the crash arm
    # below, never `|| true`-erased. A red-but-line-emitting baseline (rc=1
    # WITH lines — main already red) stays fine: the subtraction handles it.
    # Per-leg rc capture is a NO-DOWNGRADE (max) fold — same fold at all
    # FOUR leg pairs (BASE + GATED, shared gate + surgical block): a leg-1
    # CRASH (rc=2, zero lines) must survive a leg-2 rc=1-with-lines; the
    # bare last-failure-wins `|| VAR=$?` capture erases the crash and
    # defeats the crash arm below. rc=0/0 stays 0; a lone rc=1-with-lines
    # stays 1 (attribution logic); any leg >1 reaches the crash arm.
    # 1800s wedge bound per lint leg (raised from 900s, #2253 r5): no-flags
    # wall MEASURED 747s on the branch tree 2026-08-21 under fleet load
    # (load avg 13.44/32 cores, 9 concurrent lint runs; ~663s without the
    # #2253 check) — bound sized >=2x the measured wall per the CLAUDE.md
    # x2 dispersion default (900s was 1.2x and killed BOTH sides). Fires
    # only on a genuine wedge; a bound kill (rc 124) flows through the
    # NO-DOWNGRADE fold into the crash arm below — fail CLOSED, so an
    # under-sized bound silently blocks every branch's merge.
    BASE_RC=0
    timeout --kill-after=60s 1800s uv run python "$GT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    timeout --kill-after=60s 1800s uv run python "$GT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    # PAYLOAD OVERLAY (#1212 — phase 2): branch-tip content for every
    # payload path; branch deletions AND rename SOURCES removed from the
    # landing tree. A DEDICATED --no-renames listing is used (never the
    # shared own-diff.txt, which is --name-only WITH rename detection and
    # lists only a rename's DESTINATION — the renamed-away source would
    # silently survive in the gated tree). --no-renames splits a rename
    # into D(old)+A(new), so one loop body covers both sides; own-diff.txt
    # and its attribution consumers are untouched. Do NOT "simplify" by
    # reusing own-diff.txt here. The two listings can also straddle the
    # fetch (own-diff.txt pre-fetch, this one post-fetch): benign — the
    # three-dot merge-base is fork-point-stable, and any attribution-vs-
    # overlay divergence falls to the NEW-set arm (blocks, never fail-open).
    # The overlay copies the FULL own-diff incl. artifact paths — harmless
    # to the verdict (lint ignores non-cone paths); costs scale with payload.
    # quotePath=false on this + the sibling literal-path producers (#1268 —
    # own-diff, guard1/recovery tasks-diffs, additive-files, Guard 3's
    # own-commit diff): default quoting wraps a non-ASCII path in `"..."`
    # escapes, which fails every literal consumer (`git show "HEAD:$p"`,
    # cat-file/checkout/rm pathspecs, xargs, --map-files) AND every anchored
    # `^tasks/...` carve-out grep — silent skips, the #458/#1147 fail-open
    # class. ASCII output is byte-identical under the flag. Deliberately NOT
    # flagged (quoting-immune consumers): the postmerge ls-tree listings
    # (match `^tasks/<status>/<N>$` directory names — ASCII by construction),
    # the figures ls-tree (`grep -q .` non-emptiness), and the new-shared-src
    # guard (src/ module paths, pinned byte-untouched). Control-char
    # filenames (newline/tab) stay quoted regardless — the flag covers
    # bytes >0x7f only.
    git -C "$WT" -c core.quotePath=false diff --name-only --no-renames origin/main...HEAD \
      > /tmp/issue-<N>-overlay-files.txt || GT_RC=1
    # #2246 item 3: this branch runs only when the TRIGGER classified the
    # payload code-bearing (non-empty own-diff past the artifact carve-out),
    # and the --no-renames overlay path set is a superset of the own-diff
    # path set — an EMPTY listing from a ZERO-exit producer here means the
    # listing was computed against the wrong/absent tree (or a mid-window
    # ref mutation, e.g. the fetch above landing the payload on origin/main
    # between the trigger diff and this listing). Fail CLOSED via the
    # existing crash arm; never certify.
    if [ ! -s /tmp/issue-<N>-overlay-files.txt ]; then
      echo "[step10d] overlay listing EMPTY on a code-bearing payload — vacuous gated leg; failing CLOSED (#2246)"
      GT_RC=1
    fi
    # #1456: save the pre-overlay (archived origin/main) lint copy before the
    # loop overwrites it — the "theirs" side of the 3-way merge below. The
    # rm -f first clears any STALE saved copy from a prior run: a cp failure
    # under `|| true` must leave the file ABSENT (branch-copy fallback below),
    # never feed an old run's stale "theirs". `|| true`: a failed save
    # degrades to the branch-copy fallback there, never a crash.
    if grep -qxF 'scripts/workflow_lint.py' /tmp/issue-<N>-overlay-files.txt; then
      rm -f /tmp/issue-<N>-lint-main-copy.py
      cp "$GT/scripts/workflow_lint.py" /tmp/issue-<N>-lint-main-copy.py || true
    fi
    # LANDING-UNION OVERLAY (#1753, generalizing #1456; closes residual (d)
    # for the lint legs): each payload path lands in the gate tree as the
    # content a squash/rebase would land on trunk — a 3-way merge (branch
    # HEAD (ours) + merge-base + archived origin/main (theirs)) whenever
    # BOTH sides modified the path since the merge-base; the branch copy
    # verbatim when only the branch touched it; removal for branch-deleted /
    # renamed-away paths. scripts/workflow_lint.py is EXCLUDED here — its
    # dedicated #1456 block below merges it (double-merging would feed the
    # union back as "ours"). A conflicted/failed merge falls back to the
    # BRANCH copy with a loud per-path WARN — never a crash: the real merge
    # surfaces the conflict as shape 2. Incidents: #1721 (branch-tip
    # planner.md passed; the squash union landed 40900 B > the 40000 cap,
    # main red ~17h), #1719 (a stale sync snapshot false-NEW-blocked 3 gate
    # runs; a stale sync copy 3-way-merges clean with archived origin/main).
    MB_OVERLAY=$(git -C "$WT" merge-base origin/main HEAD 2>/dev/null) || MB_OVERLAY=""
    UNION_MERGED=0; UNION_FALLBACK=0
    rm -f /tmp/issue-<N>-union-base.tmp /tmp/issue-<N>-union-ours.tmp /tmp/issue-<N>-union-merged.tmp
    while IFS= read -r p; do
      if git -C "$WT" cat-file -e "HEAD:$p" 2>/dev/null; then
        mkdir -p "$GT/$(dirname "$p")" || GT_RC=1
        if [ "$p" != "scripts/workflow_lint.py" ] && [ -n "$MB_OVERLAY" ] && [ -f "$GT/$p" ] \
           && git -C "$WT" show "$MB_OVERLAY:$p" > /tmp/issue-<N>-union-base.tmp 2>/dev/null \
           && ! cmp -s /tmp/issue-<N>-union-base.tmp "$GT/$p"; then
          # both-sides-modified: certify the union, not the branch copy
          if git -C "$WT" show "HEAD:$p" > /tmp/issue-<N>-union-ours.tmp \
             && git merge-file -p /tmp/issue-<N>-union-ours.tmp \
                  /tmp/issue-<N>-union-base.tmp "$GT/$p" \
                  > /tmp/issue-<N>-union-merged.tmp 2>/dev/null; then
            mv /tmp/issue-<N>-union-merged.tmp "$GT/$p" || GT_RC=1
            UNION_MERGED=$((UNION_MERGED + 1))
          else
            git -C "$WT" show "HEAD:$p" > "$GT/$p" || GT_RC=1
            UNION_FALLBACK=$((UNION_FALLBACK + 1))
            echo "WARN: landing-union 3-way merge conflicted/failed for $p — gated legs run the BRANCH copy for it (residual (d) narrows to this path; the real merge surfaces the conflict as shape 2)"
          fi
        else
          git -C "$WT" show "HEAD:$p" > "$GT/$p" || GT_RC=1
        fi
      else
        rm -f "$GT/$p" || GT_RC=1   # branch-deleted / renamed-away path: absent from the landing tree
      fi
    done < /tmp/issue-<N>-overlay-files.txt
    echo "[step10d] landing-union overlay: merged=$UNION_MERGED fallback=$UNION_FALLBACK"
    # LINT-VINTAGE 3-WAY MERGE (#1456; incidents #1366/#1411): when the own
    # diff touches scripts/workflow_lint.py, the loop above overlaid the
    # BRANCH's lint copy, whose ratchet constants
    # (_LESSONS_ROW_GRANDFATHER_MAX_BYTES, AGENT_SPEC_SIZE_GRANDFATHER —
    # bumped on main every few days) may
    # predate main's raises and flag main-advanced files on the gated legs
    # only (NEW non-empty -> spurious block). Approximate the post-rebase
    # trunk lint instead: 3-way-merge branch copy (ours) + merge-base copy +
    # the saved archived-origin/main copy (theirs). Clean merge -> gated legs
    # carry BOTH main's constant raises / post-fork checks AND the branch's
    # own lint deliverable; a branch-added check with unfixed main offenders
    # still lands in the merged copy -> NEW -> block (correct: trunk pytest
    # goes red post-merge either way). ANY failure (merge conflict rc>0,
    # internal error, merge-base/base-copy extraction failure, missing saved
    # main copy) falls back to the BRANCH copy — exactly the pre-#1456
    # residual-(a) behavior — with a loud WARN + sidecar note, NEVER a new
    # crash path. git merge-file exits 0 on a clean merge, the number of
    # conflicts (>0) on conflict, negative (shell: 255) on error; -p writes
    # the merged result to stdout, leaving the input file untouched.
    if grep -qxF 'scripts/workflow_lint.py' /tmp/issue-<N>-overlay-files.txt \
       && git -C "$WT" cat-file -e HEAD:scripts/workflow_lint.py 2>/dev/null; then
      LINT_MERGED=no
      if [ -s /tmp/issue-<N>-lint-main-copy.py ] \
         && MB=$(git -C "$WT" merge-base origin/main HEAD 2>/dev/null) \
         && git -C "$WT" show "$MB:scripts/workflow_lint.py" \
              > /tmp/issue-<N>-lint-base-copy.py 2>/dev/null \
         && git merge-file -p "$GT/scripts/workflow_lint.py" \
              /tmp/issue-<N>-lint-base-copy.py /tmp/issue-<N>-lint-main-copy.py \
              > /tmp/issue-<N>-lint-merged.py 2>/dev/null; then
        mv /tmp/issue-<N>-lint-merged.py "$GT/scripts/workflow_lint.py" && LINT_MERGED=yes
      fi
      echo "[step10d] lint-vintage 3-way merge: $LINT_MERGED"
      if [ "$LINT_MERGED" = no ]; then
        echo "WARN: lint-copy 3-way merge failed/conflicted — gated legs run the BRANCH's workflow_lint.py (residual (a)); a ratchet-drift false block may follow. Fix: rebase the branch onto origin/main (or sync main's ratchet constants into the branch copy), then re-run the gate." \
          | tee /tmp/issue-<N>-lint-mergefile-note.txt
      fi
    fi
    # GATED legs (payload-bearing landing tree — phase 3; parity leg covers
    # the checks the no-flags bundle omits — see the bullet above):
    GATED_RC=0
    timeout --kill-after=60s 1800s uv run python "$GT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    timeout --kill-after=60s 1800s uv run python "$GT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    # MAPPED INVARIANT-TEST LEG (#1147). Dependency-mapped payloads (scan-
    # globbed scripts/issue*_*.py + dispatcher scripts, rules-pinned .md, and
    # — #1573 — src/scripts files with importing / literal-pinning /
    # stem-named tests) land via this gate with ZERO pytest on the experiment
    # auto-merge path (Step 9c is code-change-kinds-only) — #1144: 34
    # thread-caps offenders accreted this way. Map the own-diff to its mapped
    # tests via the selector's single-source dependency map; empty map => leg
    # skipped (no pytest run).
    TG_RC=0; TG_BASE_RC=0; TG_CRASH=no
    : > /tmp/issue-<N>-tg-new.txt
    : > /tmp/issue-<N>-tg-new-nodes.txt
    : > /tmp/issue-<N>-tg-unclassifiable-nodes.txt
    if ! timeout --kill-after=30s 120s uv run python "$REPO_ROOT/scripts/select_step9c_tests.py" \
        --map-files /tmp/issue-<N>-own-diff.txt --repo-root "$WT" \
        > /tmp/issue-<N>-tg-map.txt 2>/tmp/issue-<N>-tg-map-err.txt; then
      TG_CRASH=yes   # helper failure: cannot classify the payload — fail CLOSED
    fi
    if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
      # matched payload paths (attribution grep list) + gated test list:
      cut -f2 /tmp/issue-<N>-tg-map.txt | sort -u > /tmp/issue-<N>-tg-files.txt
      mapfile -t TG_TESTS < <(cut -f1 /tmp/issue-<N>-tg-map.txt | sort -u)
      # Sized from the selector's map (#1573; floor 600s, #1646):
      TG_T=$(grep -oE 'recommended-timeout-s=[0-9]+' /tmp/issue-<N>-tg-map-err.txt \
             | tail -1 | cut -d= -f2); [ -z "${TG_T:-}" ] && TG_T=600
      # Route TG fixture temp writes onto the data disk (#1408 recipe; #1363:
      # / at 100% killed a gate). Short --basetemp keeps AF_UNIX socket paths
      # under the 108-byte cap. Falls back silently (no TMPDIR, no --basetemp
      # => byte-identical argv) on pods/GCE with no data disk.
      TG_TMPROOT=$(uv run python "$REPO_ROOT/scripts/step9c_baseline.py" tmproot 2>/dev/null || true)
      if [ -n "$TG_TMPROOT" ]; then
        TG_BASETEMP=$(mktemp -d "$TG_TMPROOT/tg-XXXXXX")
      fi
      # BASELINE leg — a DETACHED SPARSE SCRATCH tree cut at the resolved
      # baseline base (the branch MERGE-BASE, #2348 — resolved below), NEVER
      # the shared repo root (#2296): every fleet
      # commit's pre-commit stash cycle reverts the root working tree
      # repo-wide for the hook window (#2015), which killed this leg mid-run
      # on #2288 and made every gated red read NEW. The helper reuses the
      # Step-9c oracle's own scratch machinery (create_scratch_worktree /
      # gate_tmp_root / assert_scratch_src_shadow) and runs pytest under the
      # ROOT venv interpreter with PYTHONPATH=<scratch>/src — the SAME venv
      # `uv run` resolved before, so no dependency-resolution change; temp
      # writes route via gate_tmp_root() inside the helper. Selection also
      # runs against the scratch: a branch-NEW mapped test is absent there,
      # so its gated hits stay NEW by construction (unchanged doctrine);
      # file-anchored scan tests (__file__-derived roots) scan the SCRATCH.
      # origin/main is already fetched (the bounded fetch above the gate
      # tree build); rc is DATA on the helper's stdout — a missing/
      # unparseable rc= line fails CLOSED into TG_CRASH.
      : > /tmp/issue-<N>-tg-baseline.txt
      TG_SCRATCH=""
      # Helper resolution (#2296 bootstrap). Instrument from the ROOT copy by
      # default — a branch must not be able to subvert its own gate — falling
      # back to the worktree copy when the root lacks the subcommand, which is
      # exactly the round that ADDS it (measured on #2296's own first Step 10d
      # run: argparse exit 2 -> empty rc -> TG_CRASH, i.e. the change could
      # never pass through the gate it introduces). What the baseline EXECUTES
      # stays payload-free either way — selector, tests and src all come from
      # the origin/main scratch (cwd=scratch, PYTHONPATH=<scratch>/src)
      # regardless of which copy drives — the instrument-vs-subject split
      # #1456 already makes for a payload-touched workflow_lint.py. What it
      # REPORTS (`rc=`, the --out FAILED set) is driver-produced, so a
      # fallback-driven baseline is auditable-not-guaranteed: the FALLBACK
      # token below is the audit trail, and the fallback engages only when the
      # ROOT probe fails under the root-side uv env (a branch cannot force it
      # via its own pyproject/uv.lock). Both copies lacking it fails CLOSED.
      TG_S9B="$REPO_ROOT/scripts/step9c_baseline.py"
      TG_S9B_SRC=root
      { uv run python "$TG_S9B" mapped-baseline --help >/dev/null 2>&1 \
        && uv run python "$TG_S9B" classify-new-nodes --help >/dev/null 2>&1; } \
        || { TG_S9B="$WT/scripts/step9c_baseline.py"; TG_S9B_SRC=FALLBACK-worktree; }
      echo "[step10d] mapped-baseline helper: $TG_S9B_SRC ($TG_S9B)"
      # Baseline base = the MERGE-BASE (fork point), NOT current origin/main
      # (#2348): the gated leg runs the branch TIP, so an origin/main-cut
      # baseline diverges from it by main's drift since fork — #2155's false
      # block was a main-side C901 fix (#2345) reaching origin/main 5 min
      # before base resolution: baseline green on the fixed copy, tip red on
      # the fork-vintage copy, and comm -23 read it as NEW. merge-base makes
      # the two trees differ only by the payload; a resolution failure
      # degrades LOUDLY to origin/main (pre-#2348 status quo).
      TG_BASE_REF=$(git -C "$WT" merge-base origin/main HEAD 2>/dev/null) \
        || { TG_BASE_REF=origin/main; echo "WARN: merge-base resolution failed — TG baseline at origin/main (drift residual live for this run)"; }
      TG_BASE_OUT=$(timeout --kill-after=30s $((TG_T + 420))s uv run python \
        "$TG_S9B" mapped-baseline \
        --map-files /tmp/issue-<N>-own-diff.txt --root "$REPO_ROOT" \
        --cones-from "$WT" --base "$TG_BASE_REF" --timeout-s "$TG_T" \
        --out /tmp/issue-<N>-tg-baseline.txt \
        2>/tmp/issue-<N>-tg-baseline-err.txt) \
        || TG_CRASH=yes
      TG_SCRATCH=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^scratch_path=//p' | tail -1)
      TG_BASE_SELECTED=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^selected_path=//p' | tail -1)
      TG_BASE_RC=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^rc=//p' | tail -1)
      [ -n "${TG_BASE_RC:-}" ] || { TG_BASE_RC=0; TG_CRASH=yes; }   # fail CLOSED
      # GATED leg — worktree copy on the payload-bearing branch-tip tree
      # (deliberately NOT the #1212 gate tree — see the mapped-leg residuals):
      ( cd "$WT" && timeout --kill-after=30s ${TG_T}s \
        env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
            NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
            ${TG_TMPROOT:+TMPDIR=$TG_TMPROOT} \
        uv run pytest "${TG_TESTS[@]}" -q -p no:cacheprovider \
          ${TG_BASETEMP:+--basetemp=$TG_BASETEMP/g} ) \
        > /tmp/issue-<N>-tg-gated.txt 2>&1 || TG_RC=$?
      [ -n "${TG_BASETEMP:-}" ] && rm -rf "$TG_BASETEMP" || true
      # rc 0 = green, 1 = test failures (attributable); ANY other rc
      # (timeout 124, collection/internal/usage error 2-5) = crash-class.
      if [ "$TG_RC" -gt 1 ] || [ "$TG_BASE_RC" -gt 1 ]; then TG_CRASH=yes; fi
      # FILE-grain payload attribution: a scan test asserts per-file
      # invariants and aggregates every offender into ONE red node, so
      # node-level subtraction is degenerate (baseline-red node == gated-red
      # node masks a NEW offender). Attribution = output lines naming a
      # payload-matched path; line numbers blanked so main-vs-branch drift of
      # the SAME pre-existing offense cannot fake a NEW line. (The third sed
      # clause blanks test_subprocess_env_explicit.py's check-1
      # `- <path>:<ln> (fn=...)` format.) pytest's own `E   assert ...` repr
      # line is DROPPED: its ellipsis-truncated offender-list repr is
      # unstable across trees (unrelated dirt in ONE tree changes it ->
      # false NEW line on an innocent payload); every real offense also
      # emits its dedicated per-file evidence line, which survives.
      # Two structural false-positive filters (#1689): the pytest
      # warnings-summary SECTION is dropped up front (awk range
      # `^=+ warnings summary` .. the `^-- Docs:` terminator — a PASSING
      # test's warnings are not failure signal, and a branch-new test's
      # warnings have no baseline twin by construction), and
      # $TG_SCRATCH/$WT/$REPO_ROOT absolute tree prefixes are normalized to
      # one <TREE> token so the SAME pre-existing line from the two trees
      # cancels under comm -23 (scratch substitution FIRST for the same
      # prefix-containment reason WT precedes $REPO_ROOT — a scratch under a
      # routed gate_tmp_root() need not be, but may be, under a normalized
      # prefix; $REPO_ROOT is a string prefix of $WT; the never-matching
      # parameter defaults keep an unset var from becoming an empty-pattern
      # sed that fails into an EMPTY hits file under the trailing
      # `|| true` = silent fail-open). Omitting the scratch clause would
      # INVERT the verdict: every baseline hit would carry an uncancelled
      # scratch-absolute prefix and every gated hit would read NEW (#2296).
      # Residual: realpath-divergent prefixes (the #681 /mnt/eps-data
      # bind-mount — a test printing os.path.realpath output emits a prefix
      # matching neither $WT nor $REPO_ROOT) stay uncancelled; fail
      # direction = the pre-existing status quo for that line class.
      for leg in baseline gated; do
        awk '/^=+ warnings summary/{w=1; next} w && /^-- Docs:/{w=0; next} !w' \
          "/tmp/issue-<N>-tg-$leg.txt" \
          | grep -F -f /tmp/issue-<N>-tg-files.txt \
          | grep -vE '^E +assert ' \
          | sed -E 's/at line [0-9]+/at line N/g; s/:[0-9]+:/::/g; s/:[0-9]+([^0-9]|$)/:N\1/g' \
          | sed -e "s|${TG_SCRATCH:-/__eps_no_scratch__}|<TREE>|g" \
              -e "s|${WT:-/__eps_no_wt__}|<TREE>|g" \
              -e "s|${REPO_ROOT:-/__eps_no_root__}|<TREE>|g" \
          | sort -u \
          > "/tmp/issue-<N>-tg-$leg-hits.txt" || true
      done
      comm -23 /tmp/issue-<N>-tg-gated-hits.txt \
        /tmp/issue-<N>-tg-baseline-hits.txt > /tmp/issue-<N>-tg-new.txt
      # NODE-grain NEW-failure subtraction (#1573): a mapped UNIT test's
      # failure summary names the TEST (`FAILED tests/<file>::<node>`), never
      # a payload path — file-grain attribution is structurally blind to it.
      # sed strips the ` - <msg>` suffix (NOT awk '{print $2}': pytest keeps
      # spaces in string param ids, so field-2 truncation would collide
      # `test_foo[a b]` (baseline-red) with `test_foo[a c]` (gated-new) and
      # falsely subtract the new failure). Baseline-red nodes (pre-existing
      # trunk red) subtract out; a branch-new mapped test is absent from the
      # baseline map, so its failures are NEW by construction (correct —
      # block; same doctrine as the branch-new scan-test note above).
      for leg in baseline gated; do
        # msg-strip caveat: a literal ' - ' INSIDE a param id truncates here;
        # a same-prefix dash-bearing sibling collision fails toward pass (narrow doc-only residual, #1573)
        grep -E '^(FAILED|ERROR) ' "/tmp/issue-<N>-tg-$leg.txt" \
          | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' \
          | sort -u > "/tmp/issue-<N>-tg-$leg-nodes.txt" || true
      done
      comm -23 /tmp/issue-<N>-tg-gated-nodes.txt \
        /tmp/issue-<N>-tg-baseline-nodes.txt > /tmp/issue-<N>-tg-new-nodes.txt
      # SET-mismatch split (#2348): comm can only subtract what the baseline
      # COULD RUN. A NEW-classified node whose test file was never SELECTED
      # on the baseline tree was never compared — keep it blocking only when
      # payload-attributable (file in the baseline selection => a real
      # both-trees delta; file in the own-diff => branch-new/payload test,
      # NEW by construction), else route it to the distinct "unclassifiable
      # — pristine-oracle needed" WARN arm (recorded in the merge note;
      # never a silent pass, never an automatic block). Helper failure or a
      # missing selected-list fails toward the status quo (all keep
      # blocking); the in-place rewrite is atomic (write-tmp + os.replace).
      # (The tg-unclassifiable-nodes.txt stale-file init is HOISTED to the
      # top of the block, beside the tg-new-nodes.txt init.)
      if [ -s /tmp/issue-<N>-tg-new-nodes.txt ]; then
        uv run python "$TG_S9B" classify-new-nodes \
          --new-nodes /tmp/issue-<N>-tg-new-nodes.txt \
          --baseline-selected "${TG_BASE_SELECTED:-/__eps_no_selected__}" \
          --own-diff /tmp/issue-<N>-own-diff.txt \
          --out-block /tmp/issue-<N>-tg-new-nodes.txt \
          --out-unclassifiable /tmp/issue-<N>-tg-unclassifiable-nodes.txt \
          || echo "WARN: classify-new-nodes failed — every NEW node kept blocking (status quo)"
      fi
    fi
    # Normalize failure lines: keep per-error `workflow_lint: <err>` lines,
    # DROP the PASS / `FAIL (N error(s))` summary lines (their COUNT changes
    # even when the failure identities match — a payload that fixes one
    # pre-existing error must not false-block on a differing summary), and
    # blank `:<line>:` numbers so unrelated drift cannot fake a NEW line.
    # (WARNs never enter: workflow_lint emits them with a `WARN: ` prefix.)
    for leg in baseline gated; do
      grep -h '^workflow_lint: ' "/tmp/issue-<N>-lint-$leg.txt" \
        | grep -vE '^workflow_lint: (PASS$|FAIL \()' \
        | sed -E 's/:[0-9]+:/::/g' | sort -u \
        > "/tmp/issue-<N>-lint-$leg-norm.txt" || true
    done
    # NEW = gated_failures − baseline_failures (set subtraction):
    comm -23 /tmp/issue-<N>-lint-gated-norm.txt \
      /tmp/issue-<N>-lint-baseline-norm.txt > /tmp/issue-<N>-lint-new.txt
    # Gated failure lines whose OFFENDER path token — the leading `<path>` of
    # the normalized line, gate-tree prefix stripped — is IN the own-diff
    # (materialized at the trigger above). Path-TOKEN set-membership, never a
    # whole-line substring grep: a failure MESSAGE routinely cites rules/docs
    # paths (e.g. .claude/rules/gotchas.md), and synced rules files sit in
    # most branches' own-diffs — the #1768 false-block (#1944). A line whose
    # leading token is not a path (a check name, a `note:`) never attributes
    # here; the NEW-set arm above remains the payload-caused backstop.
    awk -v OWN=/tmp/issue-<N>-own-diff.txt '
      BEGIN { while ((getline l < OWN) > 0) own[l]=1 }
      /^workflow_lint: / {
        s = substr($0, 16); n = index(s, ":")
        path = (n > 0) ? substr(s, 1, n-1) : s
        sub(/^\/tmp\/issue-<N>-lint-gate-tree\//, "", path)
        gsub(/^[ \t]+|[ \t]+$/, "", path)
        if (path in own) print $0
      }' /tmp/issue-<N>-lint-gated-norm.txt \
      > /tmp/issue-<N>-lint-owndiff.txt || true
    # VERDICT — CRASH ARM FIRST (fail CLOSED): a linter CRASH — rc>1 (import
    # error, missing dep, sparse-worktree crash), or rc!=0 with ZERO
    # normalized `workflow_lint:` failure lines across both legs' logs (an
    # uncaught Python exception exits 1 and emits none) — on EITHER leg pair
    # means the gate never produced a trustworthy compare. `crash` is an
    # unconditional block-path verdict (same epm:merge-failed handling as
    # `block`; Verdict bullet case 3) — NEVER `pass`. Only then the
    # attribution logic: a green gated run (exit 0) can never block; a red
    # one (rc=1 WITH lines) blocks only when payload-attributed (an
    # own-diff-named failure line OR a non-empty NEW set); rc=1 with lines
    # but none own-diff/NEW stays `pass` (pre-existing red — WARN).
    if [ "$GT_RC" -ne 0 ] || [ "$GATED_RC" -gt 1 ] || [ "$BASE_RC" -gt 1 ] || [ "$TG_CRASH" = "yes" ] \
       || { [ "$GATED_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-gated-norm.txt ]; } \
       || { [ "$BASE_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-baseline-norm.txt ]; }; then
      echo crash > /tmp/issue-<N>-lint-verdict.txt
    elif { [ "$GATED_RC" -ne 0 ] \
       && { [ -s /tmp/issue-<N>-lint-owndiff.txt ] || [ -s /tmp/issue-<N>-lint-new.txt ]; }; } \
       || { [ "$TG_RC" -ne 0 ] \
       && { [ -s /tmp/issue-<N>-tg-new.txt ] || [ -s /tmp/issue-<N>-tg-new-nodes.txt ]; }; }; then
      echo block > /tmp/issue-<N>-lint-verdict.txt
    else
      echo pass > /tmp/issue-<N>-lint-verdict.txt
    fi
    rm -rf "$GT"   # ephemeral; a crash-left tree is rebuilt (rm -rf first) on the next gate run
  else
    # Executable trigger (the Trigger bullet above): artifact-only payload —
    # both lint runs skipped by design.
    echo skip-artifact-only > /tmp/issue-<N>-lint-verdict.txt
  fi
  # SHA-BIND the verdict to the branch tip it certified (line 2, #1097): a
  # consumer accepts a pass/skip verdict ONLY while the CURRENT tip still
  # equals this sha — any new commit since certification invalidates it
  # (fail CLOSED, re-run the gate), and a hand-written verdict without the
  # correct sha is useless (anti-self-attestation, the #1082 incident).
  # Appended for every verdict; block/crash never certify, so their sha
  # line is inert.
  git -C "$WT" rev-parse HEAD >> /tmp/issue-<N>-lint-verdict.txt
  cat /tmp/issue-<N>-lint-verdict.txt   # line 1: pass | block | crash | skip-artifact-only; line 2: certified branch-tip sha
  ```

  **Completion-read (forms (i)/(ii)).** The launcher bg-Bash exits in
  seconds — its completion (the harness notification) is NOT the gate-done
  signal; the detached gate finishes ~30-40 min later under typical fleet
  load. The gate is done when the verdict file
  `/tmp/issue-<N>-lint-verdict.txt` exists (stale-cleared by the `rm -f`
  at launch) or the single-flight probe
  (`step9c_baseline.py probe --pattern 'issue-<N>-lint-gate'`) reads
  CLEAR — wait via a Monitor until-loop on the probe or the
  `/issue-tick <N>` re-wake, then read the verdict in a fresh FOREGROUND
  call from the FILE. Missing verdict file + LIVE probe match = the gate
  is STILL RUNNING: keep waiting — never a premature death read. Missing
  verdict file + probe CLEAR = the detached run died before writing a
  verdict (tool kill / watcher force-stop / wedge-bound kill) — treat as
  gate-not-run, fail CLOSED: NEVER proceed to the merge conditional, NEVER
  hand-write the verdict (#1082). Apply crash-fix-rounds
  § Kill-before-relaunch (probe `pgrep -af 'issue-<N>-lint-gat[e]'` — the
  workload script path `/tmp/issue-<N>-lint-gate.sh` rides the detached
  workload's argv for its whole life (#2256), so the probe is
  exact-issue-scoped; run it in its OWN Bash call: the broadened pattern
  also matches the gate's `.sh`/`.log` paths, so a compound Bash embedding
  a diagnostic tail of the gate log would self-match through the wrapper
  argv — the #1742 bracket-defeat mode (crash-fix-rounds.md
  § Kill-before-relaunch step 1 is the generic rule); exit-code trap: raw
  pgrep exits 0 on a LIVE match — INVERTED vs `step9c_baseline.py probe`,
  whose 0 = clear — this kill-arm keeps pgrep because it wants the pid
  list. Subshell-argv note: a bash `( ... )` subshell child is
  fork-without-exec and inherits the parent's argv in
  `/proc/<pid>/cmdline`, so the workload script pid can appear TWICE in
  the pgrep output — two pids, one gate, NOT a duplicate launch; kill the
  process GROUP / both pids together) before re-running the gate
  ONCE; still dying ->
  `epm:merge-failed v1` (Verdict bullet case 3). A partial death (killed
  between the verdict write and the sha append) leaves a 1-line file the
  binding sites' line-2 sha check already fails CLOSED on. Worst case —
  every bounded leg wedged — the call runs ~78 min, past the 60-min
  § Long-phase heartbeat boundary (rare; a watcher force-stop there is
  itself fail-closed: no verdict file gets written). Print the
  diagnostic tails via the canonical fail-soft compound in the block
  below — this is the Recipe exit-code hygiene class (Step 9c 1b): on a
  PASS these files are routinely empty or absent, so a bare trailing
  `grep`/`cat`/`[ -s ]` leg exits non-zero and reads as a tool error;
  every leg is if-formed and the block ends exit-0 on a healthy read.

  ```bash
  if [ ! -f /tmp/issue-<N>-lint-verdict.txt ]; then
    echo "FATAL: verdict file missing — live probe match = gate still running (keep waiting); probe CLEAR = the detached gate run died before writing a verdict: kill-before-relaunch, then re-sync (§ pre-gate re-sync), then re-run the gate ONCE; NEVER record pass." >&2
  else
    cat /tmp/issue-<N>-lint-verdict.txt   # line 1: verdict; line 2: certified sha — the merge conditional below stays the hard stop
    # Fail-soft diagnostic tails (Recipe exit-code hygiene, Step 9c 1b):
    # empty/absent on a PASS by design — never a bare trailing grep/cat/
    # [ -s ] leg here (it would exit 1 and read as a tool error).
    for f in lint-new lint-owndiff tg-new tg-new-nodes tg-unclassifiable-nodes; do
      if [ -s "/tmp/issue-<N>-$f.txt" ]; then echo "--- $f ---"; head -20 "/tmp/issue-<N>-$f.txt"; fi
    done; true
  fi
  ```

- **Gate earlyoom protection (#1045 recipe, #1211).** Both executable blocks
  (the shared form (i)/(ii) block above and the form (iii) surgical block)
  open with the SAME fail-open self-choom preamble as the Step 9c 1b/1c
  gates — `oom_score_adj` inherits across fork/exec (probe-verified), −600
  not −1000, FAIL-OPEN (`choom=failed` warns and the gate proceeds
  unprotected; the preamble never blocks a gate, never alters the verdict
  logic, and leaves the verdict-file contract byte-unchanged: line 1
  verdict, line 2 sha). Full calibration rationale: Step 9c § "Gate
  earlyoom protection (#1045)" — do not duplicate it here. Motivation:
  the lint legs (~4.5-6 min python) and TG pytest legs match this VM's
  earlyoom `--prefer` regex (#1143). Copy the echoed
  `[step10d] lint-gate earlyoom protection choom=...` breadcrumb line into
  the `epm:merged` / `epm:merge-failed` note so a crash-verdict post-mortem can tell a
  protected kill from an unprotected one. Likewise copy the
  `[step10d] landing-union overlay: merged=<n> fallback=<m>` echo into the
  same note (#1753) — a nonzero `fallback=` names how many payload paths
  the gated legs ran as branch copies, the first thing a post-merge lint
  divergence should be triaged against.

- **Mapped invariant-test leg (#1147).** A second, trigger-gated leg of the
  SAME gate: when the payload (the own-diff / additive list) matches the
  selector's dependency map — `GLOB_SCAN_TESTS` + rules-pin (#1496) + the
  src/scripts import/literal/stem dependency arms (#1573), WORKFLOW_INVARIANT
  members excluded — the executable block runs the MAPPED tests on
  the payload-bearing tree and subtracts a payload-free baseline run. The
  trigger is the helper map — `select_step9c_tests.py --map-files <list-file>
  [--repo-root <tree>]` prints `test<TAB>matched_path` pairs; empty
  output = leg skipped entirely (zero pytest runs added). The helper is the
  SINGLE SOURCE of the map — never hardcode the globs/arms in this
  file (the selector's drift pins in `tests/test_select_step9c_tests.py` keep
  the map current, #895). A payload code file the selector cannot map to ANY
  test draws its `no mapped tests for code file` stderr WARN (#1573's
  fail-loud floor) into `/tmp/issue-<N>-tg-map-err.txt`, recorded in the
  `epm:merged` / `epm:merge-failed` note alongside the lint/tg tails those
  notes already record. Attribution runs at TWO grains (#1573). FILE-grain
  for scan-test output: a
  scan test asserts per-file invariants and aggregates EVERY offender into
  ONE red node, so node-level subtraction alone is degenerate there
  (baseline-red node ==
  gated-red node would mask a NEW offender; compare additionally marks
  NON-file-anchored scan-set nodes scratch-ineligible (`step9c_baseline.py`
  `FILE_ANCHORED_SCAN_TESTS` members
  are scratch-resolved, still WARNed — #1337)). File-grain hits
  = pytest-output lines naming a payload-matched path, the pytest
  warnings-summary section excluded up front (a PASSING test's warnings are
  not failure signal; a branch-new test's warnings have no baseline twin —
  #1689), line numbers blanked
  so main-vs-branch drift of the SAME pre-existing offense cannot fake a NEW
  line, `$TG_SCRATCH`/`$WT`/`$REPO_ROOT` absolute tree prefixes normalized
  to a common `<TREE>` token so the same line from the two trees cancels
  (#1689; the scratch clause is load-bearing — without it every baseline hit
  carries an uncancelled scratch prefix and every gated hit reads NEW, #2296),
  pytest's ellipsis-truncated `E   assert ...` repr line dropped (its
  content is unstable across trees; every real offense also emits a dedicated
  per-file evidence line); NEW = gated hits − baseline hits (`comm -23`,
  `/tmp/issue-<N>-tg-new.txt`). And junit-NODE-grain for unit-test failures
  (#1573): a failing mapped unit test's summary line names the TEST
  (`FAILED tests/<file>::<node>`), never a payload path, so file-grain alone
  is structurally blind to it — NEW failed/error node ids = gated − baseline
  (`comm -23`, `/tmp/issue-<N>-tg-new-nodes.txt`), the ` - <msg>` suffix
  stripped via `sed` (never awk field-2: space-bearing string param ids must
  survive intact). The BASELINE is cut at the branch's MERGE-BASE (fork
  point), never current origin/main (#2348): the gated leg runs the branch
  TIP, so an origin/main-cut baseline diverges from it by main's drift since
  fork — #2155's false block was a main-side fix (#2345) reaching origin/main
  minutes before base resolution (baseline green on the fixed copy, tip red
  on the fork-vintage copy; `comm -23` read it as NEW); merge-base makes the
  two trees differ only by the payload, and a resolution failure degrades
  LOUDLY to origin/main. After the node-grain comm, the `classify-new-nodes`
  split (#2348) routes each NEW node by baseline OBSERVABILITY against the
  helper's `selected_path=` sidecar (the test files the baseline COULD run):
  file in the baseline selection ⇒ keeps blocking (a real both-trees delta);
  file in the own-diff ⇒ keeps blocking (branch-new/payload test — NEW by
  construction, unchanged doctrine); anything else routes to the distinct
  `unclassifiable — pristine-oracle needed` WARN arm
  (`/tmp/issue-<N>-tg-unclassifiable-nodes.txt`) — RECORDED in the
  `epm:merged` / `epm:merge-failed` note whenever non-empty (the
  completion-read tail prints it), never a silent pass, never an automatic
  block; helper failure / a missing selected-list fails toward the status quo
  (every NEW node keeps blocking). Node-grain widens the block surface to genuinely flaky
  mapped tests; the existing "re-run the gate ONCE → `epm:merge-failed`"
  recovery covers that, and baseline subtraction still removes deterministic
  trunk red. Each pytest leg is bounded at the selector-sized `${TG_T}` —
  grepped from the gated map's machine-greppable `recommended-timeout-s=`
  stderr sizing line in `/tmp/issue-<N>-tg-map-err.txt`, falling back to a
  fixed 600 s when the line is absent (the sizing floor is also 600 s —
  raised from 300 s by #1646). The baseline leg reuses the gated map's
  `TG_T` (threaded as `mapped-baseline --timeout-s`; the helper's own
  in-scratch selection does not feed sizing, and its outer wrapper adds
  +420 s for scratch materialization + selection + teardown; the gated map
  is the superset in
  the common case and over-sizing is the safe direction) — a
  k_baseline ≫ k_gated residual fails CLOSED (rc 124 → crash); the known
  escalation, sizing from the max over BOTH maps by keeping the baseline
  map's stderr, is wired only if that crash shape recurs. A
  timeout / pytest rc>1 / helper failure on either leg is
  crash-class: verdict `crash`, fail CLOSED, the same "re-run the gate ONCE →
  `epm:merge-failed`" recovery as the lint legs (Verdict bullet case 3). On
  form (iii) this leg is structurally DORMANT today — the surgical additive
  pathspec set excludes `scripts/` / `src/`, so its trigger map is empty by
  construction; it arms automatically if that set ever grows. Known residuals
  (accepted, documented): (a) path-(i) test-VERSION drift — the gated leg
  runs the branch-tip copy of the scan test, so a check added on `main` after
  the branch forked is not enforced there (fail-safe direction; the LINT legs
  no longer share this residual — the #1212 gate tree runs the landing tree's
  lint on every path-(i) run; the TEST legs keep branch-tip copies because
  syncing arbitrary individual test files without their import closure —
  conftest, tests/ helpers — risks hybrid trees, but the lint/guard pin-test
  FAMILY is now Step-5a-synced AND pre-gate re-synced from origin/main
  (#1560), narrowing the drift window to (α) non-family rules-pin tests
  (prose-pin skew) and (β) the `explore_persona_space.workflow` seam,
  same remedy for both: rebase onto origin/main / cross-check at the repo
  root). The false-block INVERSE of this drift — an offense FIXED on main
  after fork false-blocking a stale tip (#2155/#2345) — is CLOSED by the
  merge-base-pinned baseline (#2348): a fork-vintage red is red in BOTH legs
  and subtracts. The pinning opens a narrow NEW false-PASS window on forms
  (i)/(ii): a payload that REINTRODUCES a fork-vintage red main has since
  fixed subtracts out (same failure identity in both legs) where an
  origin/main baseline would have blocked — fail-safe, trunk Step 9c catches
  it post-merge. Two further residuals: selector-VINTAGE drift (the gated
  selection uses the ROOT selector copy while the baseline uses the
  merge-base scratch copy, so arms added on main since fork can still
  produce gated-only selections of non-payload tests — those route to the
  `unclassifiable` WARN arm, #2348, never an automatic block), and a
  data-driven within-file parametrization drift inside an untouched file
  (its file IS in the baseline selection, so a tip-only node there keeps
  blocking — stays under the test-VERSION-drift class, doc-only);
  (b) the baseline leg runs on a detached sparse scratch tree cut at
  the resolved baseline base (the branch merge-base, #2348; #2296 — the
  pre-#2296 shared-root baseline was
  killed mid-run by the #2015 pre-commit stash cycle on #2288: process DEATH
  empties the baseline, so every gated red read NEW — maximally fail-CLOSED,
  the OPPOSITE direction of the retired "dirt can only enlarge the baseline
  hit set and bias toward PASS" claim; do not rule out a baseline false
  block on that retired reasoning when diagnosing a gate crash). The
  remaining (b) residual is LIVE-TREE-ANCHORED scan tests: a mapped scan
  test that anchors via `repo_root()` / `task_workflow` instead of its own
  `__file__` (cf. `step9c_baseline.FILE_ANCHORED_SCAN_TESTS` — the curated
  file-anchored set scans the SCRATCH and gains a genuinely payload-free
  baseline) still scans the shared ROOT even when its code runs from the
  scratch, so its baseline can FLAP under churn — a missed baseline red
  reads as a false NEW, fail-closed, resolvable through the existing
  provenance-override path; strictly better than the shared-root leg it
  replaces, where the same churn killed the process outright (extending the
  file-anchoring curation is the fix, #2293-adjacent);
  (c) a payload that DEEPENS an
  offense in an already-red payload-touched file normalizes to the same
  per-file line and is subtracted — a false-pass window that vanishes once
  #1145 greens the baseline (the file is already post-freeze red; low harm).

- **Verdict — payload-attributed via failure-LINE-SET subtraction; NEVER
  blocks an innocent merge on pre-existing red.** Exit codes alone are
  vacuous when `main` is already red for an unrelated reason; attribution
  compares normalized `workflow_lint:` failure LINES (strip volatile
  prefixes; keep the `<check>/<file>[:<line>] <msg>` identity) between the
  GATED run and a payload-free BASELINE run — BOTH legs (no-flags + parity)
  in each run, so a pre-existing parity-leg red on main can never be
  misread as payload-caused (and vice versa). The executable block above
  computes the verdict (`block` | `pass` | `crash` | `skip-artifact-only`)
  and persists it SHA-BOUND to `/tmp/issue-<N>-lint-verdict.txt` (line 1 =
  verdict, line 2 = the certified branch-tip sha); the binding sites gate
  their merge/push/add commands on that FILE with an explicit conditional —
  a missing verdict file fails CLOSED (the gate has not run yet), and a
  pass/skip verdict certifies ONLY while the CURRENT branch tip equals the
  certified sha, so any new commit since certification fails CLOSED too
  (re-run the gate) and a hand-written verdict without the correct sha is
  useless (anti-self-attestation). The mapped invariant-test leg (#1147)
  contributes into this SAME verdict file: `crash` on either test leg's
  crash-class outcome (pytest rc>1 / timeout / helper failure), `block` on a
  payload-attributed NEW test hit (`/tmp/issue-<N>-tg-new.txt` non-empty with
  a red gated run) OR a NEW failed/error test NODE
  (`/tmp/issue-<N>-tg-new-nodes.txt` non-empty with a red gated run — the
  #1573 node-grain arm); a gated-red-but-no-NEW-hit-and-no-NEW-node test
  outcome (pre-existing
  trunk red) stays `pass`, and the `epm:merged` WARN note records the tg tail
  alongside the lint tail. The file is REMOVED only once it can no
  longer certify anything: after a SUCCESSFUL `gh pr merge`
  (consume-on-merge-success), or in the block/crash/stale-sha branch (a
  fresh gate run regenerates it). A merge that fails for a NON-lint
  transport reason (the #1041 rebase-refusal → `--squash`-retry shape)
  therefore stays certified by the SAME gate run — never hand-recreate the
  verdict file (#1082). ONE mechanically-gated exception (#1807): the
  auto-merge RE-BIND stanza (safe-case block) may rewrite LINE 2 ONLY —
  line 1 is never touched — after its own `git diff --name-status
  <certified-sha>..HEAD` probe proves the post-gate sync commit's delta is
  origin/main-identical `A`/`M`-only; the license covers ONLY that stanza
  executing over its own probe output — a free-standing "update line 2"
  move stays banned, and the #1613 empty-commit synchronize explicitly
  STAYS a stale-verdict → gate-re-run case, never a re-bind. On a `block`
  (or `crash`) verdict:
  1. An own-diff-named gated failure line exists
     (`/tmp/issue-<N>-lint-owndiff.txt` non-empty) → the payload is the
     offender. Fix it in the worktree (the lint names file + rule),
     commit by explicit path, re-run the gate ONCE; still failing → post
     `epm:merge-failed v1` with `{reason: "pre-push workflow-lint gate",
     lint_tail: <last lines>}`, surface ONE line in chat, CONTINUE (same
     fail-fast policy as a merge failure; retried idempotently on the next
     `/issue <N>`).
  2. No own-diff-named line → the block came from
     `NEW = gated_failures − baseline_failures` (`comm -23` on the
     normalized lines, persisted at `/tmp/issue-<N>-lint-new.txt`) — a
     payload-caused cross-file interaction (e.g. a lessons-index /
     lens-coverage check naming the index rather than the added rule
     file) — treat as case 1 (block). NEW empty with no own-diff-named
     line never blocks: the executable block writes `pass` — pre-existing
     red is a WARN (record the lint tail in the `epm:merged` note) and
     the merge PROCEEDS. The baseline and gated runs execute back-to-back
     inside the ONE background gate call, so a concurrent merge cannot
     widen the compare window
     (moving-main race — keep the window tight, preserve the
     main-already-red detail in the marker; the #1212 gate freezes both
     legs to one archived origin/main snapshot — the back-to-back advice
     stays as defense-in-depth).
  3. `crash` — the linter itself CRASHED on either leg pair (rc>1, or
     rc!=0 with zero normalized `workflow_lint:` failure lines: import
     error, missing dep, sparse-worktree crash — the gated leg runs the
     gate tree's `workflow_lint.py` — the 3-way-MERGED copy (or, on
     merge-failure fallback, the BRANCH's copy) whenever the own-diff
     touches it — so the crash is payload-inducible (a semantically-broken
     clean merge lands here too, fail CLOSED; it predicts the post-merge
     trunk file, so blocking is correct — rebase onto origin/main and
     re-run); a gate-tree CONSTRUCTION failure (GT_RC != 0) also lands
     here),
     or the trigger diff failed. No trustworthy compare exists, so this is
     an unconditional block-path verdict: fix the crash cause in the
     worktree, re-run the gate ONCE; still crashing → the SAME
     `epm:merge-failed v1` handling as case 1. Never merge/push on `crash`.
- **Mandatory urgent-park emission on workflow-surface pre-existing red
  (#1713).** Whenever the gate's `pass` verdict rests on a pre-existing
  red hit whose file matches the workflow surface (`scripts/`,
  `.claude/`, `CLAUDE.md`, `docs/`, `tests/`), the session MUST emit —
  in the same turn as the `epm:merged` (or `epm:progress` completion)
  note — a `<!-- workflow-fix-candidate v1 -->` block carrying the
  #1681 urgent grammar: three fields inside the block —
  `urgency: main-red`, `failing_test: <ONE pytest node id, e.g.
  tests/test_x.py::test_y>`, and `wf_fix: true|false` (`true` when the
  offending file itself lives on the workflow surface, `false`
  otherwise; the parked candidate is still mechanically routable
  regardless via the watcher's urgent-park router pass). Prose
  alternatives ("noted for /daily follow-up", "will be picked up
  later", "leaving for the sweep") are NOT acceptable terminal
  dispositions — the nightly /daily Step C sweep is the FALLBACK, not
  the primary route: without the urgent grammar every intervening
  session's Step 9c gate must re-classify the same pre-existing red
  (#1701 → #1698). See
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard "Urgent
  fast path" for the router semantics; the parking session STILL
  never files or spawns the fix itself. Non-workflow-surface
  pre-existing red keeps the report-and-continue disposition
  unchanged.
- **Size-ratchet cap bumps are computed from landing bytes (#1753).** A
  payload that raises a size-cap constant (e.g.
  `AGENT_SPEC_SIZE_GRANDFATHER`) computes the new cap from the LANDING
  content — the gate tree's 3-way-merged copy of the capped file (or a
  local merge of fresh `origin/main` into the branch) — never from
  branch-tip bytes: main-side additions stack at merge time (#1727). With the landing-union overlay the
  gate catches an under-computed cap fail-CLOSED (verdict `block`)
  pre-merge, instead of post-merge main-red.
- **Baseline semantics per binding form (the baseline is ALWAYS a
  payload-free tree).** The mapped invariant-TEST legs (#1147) keep the
  ORIGINAL per-form GATED placement (gated = the `$WT` copy on the
  branch-tip / post-merge tree on forms (i)/(ii); the root copy on form
  (iii)); their BASELINE runs a detached sparse scratch tree cut at the
  resolved MERGE-BASE of origin/main and the tree the gated leg runs —
  forms (i)/(ii): `git -C "$WT" merge-base origin/main HEAD` (the fork
  point); form (iii): `git -C "$REPO_ROOT" merge-base origin/main HEAD`
  (#2348 — the baseline base matches the gated tree's VINTAGE, so main's
  drift since fork can never read as NEW; #2155/#2345) — via
  `step9c_baseline.py mapped-baseline` on ALL
  THREE forms (#2296 — never the shared repo root, which the #2015 stash
  cycle reverts repo-wide); the LINT legs on forms (i)/(ii) run the
  #1212 gate tree. (i) Safe case: LINT legs — gated = the gate-tree copy
  on the LANDING tree (origin/main + own-diff overlay), baseline = the
  SAME copy on the payload-free landing base (#1212); mapped-TEST legs —
  gated = the `$WT` copy on the branch-tip tree, baseline = the
  base-pinned scratch (#2296); bind immediately before `gh pr ready` /
  `gh pr merge`.
  (ii) Merge-conflict recovery: LINT legs — gated/baseline = the gate tree
  rebuilt from the post-merge tip (content-identical to the post-merge
  worktree, which carries main's CURRENT lint — the ideal gate point);
  mapped-TEST legs — gated = the post-merge worktree copy, baseline = the
  base-pinned scratch (#2296); bind after conflict resolution + targeted
  tests, before `git -C "$WT" push`. (iii) Surgical additive checkout: the payload lands
  in the ROOT tree, so the BASELINE MUST RUN BEFORE the
  `xargs ... git checkout` — a post-checkout "main-side" run would re-lint
  the SAME contaminated tree, a degenerate compare that fails open at
  exactly the fast-path form; sequence = baseline (LINT: root copy;
  mapped-TEST: base-pinned scratch, #2296) →
  checkout → gated (root copy, both legs) → set-subtraction verdict → on
  pass, `git add`. The whole sequence runs as ONE BACKGROUND Bash
  invocation — do NOT split it across invocations: the contaminated-root
  window (checkout → stage/commit-or-clean) stays compute-bound (~5-6 min)
  only while the sequence runs in one shell, and a split inserts
  orchestrator turn-boundary latency inside that window. While it runs,
  end the turn and run no repo-root-mutating commands until the
  completion-read (surgical block below). On a block at (iii), clean the payload out of BOTH
  index and working tree with the hook-VERIFIED two-step (run from
  `$REPO_ROOT`; simulated against `scripts/guard_repo_root_branch.sh`
  — `--staged` PLUS a worktree flag is mechanically BLOCKED by its #897
  restore detector, and the hook's own guidance bans pointing `-C` at
  the repo root for a DESTRUCTIVE op):
  first `xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT"
  restore --staged --` (index-only unstage — non-destructive, admitted by
  the restore allow-arm), then `xargs -r -a
  /tmp/issue-<N>-additive-files.txt rm -f --` (the paths are A-only,
  absent from `main`, and untracked after the unstage — the plain `rm`
  destroys no main state; a bare `rm -f` WITHOUT the unstage would leave
  them STAGED in the shared root index, polluting concurrent sessions'
  `git diff --cached` echoes). The `xargs -r` (`--no-run-if-empty`) on
  every additive-list consumer is load-bearing: on an EMPTY list a
  flag-less xargs still runs its command once with NO pathspec. The fast
  path routes through (iii). Idempotent: re-entry just re-runs the gate.
  If the repo-root guard hook blocks an improvised unqualified variant of
  the restore line (or of the additive-checkout consumer), the WHOLE
  compound was skipped — including any clause that wrote
  `/tmp/issue-<N>-additive-files.txt`; regenerate the list, then retry
  the verbatim `-C "$REPO_ROOT"` forms (full recovery contract: the
  guard-block paragraph after the surgical-additive-checkout executable
  block below).
- **Known residuals (accepted, documented):** the #1212 gate tree removed
  the old path-(i) vintage residual for the LINT legs — a check ADDED on
  `main` after the branch forked is now enforced on every path, so a
  payload violating it BLOCKS (the #931 class), and a check
  retired/loosened on main can no longer false-block. What remains: (a) a
  branch whose OWN diff touches `scripts/workflow_lint.py` gets a 3-way-
  MERGED lint copy on the gated legs (#1456: branch ⊕ merge-base ⊕
  archived origin/main — approximates the post-merge trunk copy, so
  main's ratchet-constant raises no longer false-block, #1366/#1411);
  the residual NARROWS to the merge-failure fallback — on conflict /
  error the gated legs keep the BRANCH copy (loud WARN + sidecar
  `/tmp/issue-<N>-lint-mergefile-note.txt`), and a resulting
  ratchet-drift block resolves through the standard case-1
  fix-or-`epm:merge-failed` path (rebase onto origin/main, or sync
  main's ratchet constants into the branch copy) — plus the narrow
  semantically-broken-clean-merge window, which crashes the gated leg
  into the fail-CLOSED crash arm and equally predicts a post-merge
  trunk crash; (b) the gate tree
  materializes only `workflow_lint.py`'s scan/target surface (the archive
  pathspec set in the executable block) — if the linter grows a new scan
  root, a gated false block naming paths outside that set is the symptom
  and extending the set is the fix (the #1154 `docs/` pins are the
  precedent); (c) the mapped invariant-TEST legs keep the branch-tip test
  copies (path-(i) test-VERSION drift,
  fail-safe direction; the baseline runs a MERGE-BASE-pinned scratch —
  #2348, closing the false-block INVERSE where an offense fixed on main
  after fork read NEW on a stale tip, incident #2155/#2345; #2296) —
  the lint/guard family is now Step-5a-synced AND
  pre-gate re-synced from origin/main (#1560), so the remaining drift
  window is (α) non-family rules-pin tests (prose-pin skew; symptom: a
  gated-only red in a rules-mentioning test the family does not cover)
  and (β) the `explore_persona_space.workflow` seam, both with the same
  remedy (rebase onto origin/main / cross-check at the repo root); the
  trunk pytest remains their backstop — and a NEW node whose test file
  the baseline never SELECTED (the set-mismatch class, #2348) routes to
  the `unclassifiable — pristine-oracle needed` WARN arm, RECORDED in
  the merge note whenever non-empty, never an automatic block; (d) both-sides-modified
  overlay paths are now 3-WAY-MERGED on the gated legs (the landing-union
  overlay, #1753, generalizing #1456 to every payload path;
  `scripts/workflow_lint.py` keeps its dedicated #1456 block) — the
  residual NARROWS to (i) the per-path conflict-fallback window (loud
  WARN + branch copy; the real merge surfaces the conflict as shape 2
  anyway — and the `fallback=` counter also counts non-conflict failures,
  e.g. a failed `git show`, so never read `fallback=` as a pure conflict
  count), (ii) an add/add path absent at the merge-base (no base to
  merge; branch copy, rare), and (iii) the clean-merge-to-wrong-content
  window — a main-side REVERT (post-sync) of a hunk the branch's sync
  copy carries merges CLEANLY to non-main content, the same class as
  residual (a)'s semantically-broken-clean-merge window; (e) same-issue
  concurrent gate runs would share one `$GT` (a phase-flip race) —
  excluded by the Step 0 single-orchestrator guard + the pre-dispatch
  dedup, with the #911 janitor reaping any crash leftovers.
- **Post-gate freshness re-sync (#1714; supersedes the #1560 pre-gate
  placement).** The lint gate builds its landing tree from `git archive
  origin/main`, so a re-sync AFTER the gate returns does not invalidate
  the gate verdict — but a re-sync BEFORE a ~30-min gate snapshots
  origin/main against a tip that will be stale by merge time, and #1476
  proved that
  origin/main advances DURING the gate window often enough to break the
  squash merge with `CONFLICTING`. The re-sync is invoked from the
  auto-merge subsection below (the H4 heading immediately following
  this gate section), IMMEDIATELY before `gh pr merge --squash` and
  AFTER the gate verdict file has been read (i.e. after the
  stale-verdict `rm -f` above and after the executable gate block has
  returned pass):
    1. `timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet || true`
    2. Run the Step 5a family-atomic block (§ Step 5a) once — it already
       sources fetched `origin/main` as of #1747 (no ref substitution
       needed; its on-main skip guard rides along harmlessly here, since
       `$WT` is on `issue-<N>` the guard evaluates false) —
       against the ALREADY-BOUND `$WT` — the merge flow bound
       `WT="$REPO_ROOT/.claude/worktrees/issue-<N>"` in the guards
       block, so DROP the Step 5a block's own
       `WT=$(git rev-parse --show-toplevel)` line —
       do NOT re-derive `$WT` at Step 10d (a repo-root cwd would
       rebind it to the shared root).
    3. End with one echo — `[step10d] post-gate re-sync: synced <n> files (<sha>) | no drift` —
       so ran-vs-never-ran is observable in the merge transcript (copy
       the line into the `epm:merged` / `epm:merge-failed` note).
    4. If the re-sync COMMITTED (`<sha>` != `no-drift`), run the verdict
       RE-BIND stanza (auto-merge subsection, #1807): enumerate the
       certified-sha..HEAD delta with `git diff --name-status`; every
       row must be `A`/`M` with content byte-identical to fetched
       `origin/main` — then line 2 of the verdict file is re-bound to
       the new tip (line 1 is never touched), because a delta that only
       adds/overwrites files with main's own bytes cannot change the
       landing tree the gate certified. ANY other delta — a
       `D`/`R*`/`C*`/`T`/`U` status row (the sync's
       `checkout origin/main --` can only add/modify, never delete) or
       a non-identical file — fails CLOSED: verdict removed, no merge,
       re-run the gate.
    5. The head-sync pre-check (#1657) runs AFTER the re-sync +
       re-bind — it polls PR-object parity against the FINAL tip (a
       fresh sync push re-introduces exactly the lag the pre-check
       absorbs; polling before the sync would check the wrong tip).
    6. `gh pr ready` + `gh pr merge --squash` — if it returns
       `CONFLICTING`, fall through to the existing
       merge-conflict-recovery path (§ Concurrent-committer merge
       conflicts).

  The Guard-3 subject-scoped commit-subject convention still applies —
  never write a full-message grep-exclusion invocation into this Step 10d
  section (enforcement = the gate-region negative assert in
  `tests/test_issue_skill_lint_family_sync.py`, whose region spans this
  post-gate section through the auto-merge heading — the Guard-3 pin
  test's own region ban stops at the fast-path heading and does not
  reach here).

  Synced files enter the sync commit ONLY if they DIFFER from HEAD
  (the family-atomic block's `git diff --quiet` gate is what commits;
  a no-drift re-sync commits nothing and the flow is idempotent). The
  gate has already verified the landing tree; a post-gate sync of
  origin/main-identical bytes does not change the landing tree the gate
  green-lit — family-atomic skip preserves branch-side content, the
  gate's #1456 3-way merge of `workflow_lint.py` remains the covering
  mechanism, and the merge's own conflict resolution
  handles the residual. As of #1807 the verdict-file SHA mechanics AGREE
  with this landing-tree argument: when step 4's probe mechanically
  proves the sync commit's delta payload-free, the re-bind stanza moves
  line 2 to the sync tip instead of forcing a full gate re-run; every
  unverifiable tip delta still fails CLOSED into a gate re-run.

#### The auto-merge procedure (safe case: guard 3 clean — mainline-based, own commits in scope)

```bash
# PR-object liveness probe (#1768 round-2 / #1897): a follow-up round's
# branch outlives its round-1 PR — a MERGED/CLOSED PR is a TERMINAL
# GitHub object (new branch commits never attach), and `gh pr merge` on
# one exits 0 with "was already merged" (false success: verdict
# consumed, payload stranded). Resolve state + pre-attempt mergedAt
# alongside the number, and require OPEN before any merge attempt.
# (`gh pr view issue-<N>` by branch name prefers the OPEN PR when one
# exists, so the re-resolve after `gh pr create` binds the fresh PR.)
PR_INFO=$(gh pr view issue-<N> --json number,state,mergedAt \
  -q '[(.number | tostring), .state, (.mergedAt // "null")] | join(" ")' 2>/dev/null) || true
PR=$(echo "$PR_INFO" | cut -d' ' -f1)
PR_STATE=$(echo "$PR_INFO" | cut -d' ' -f2)
PRE_MERGED_AT=$(echo "$PR_INFO" | cut -d' ' -f3)
# Usable-PR resolution (#2240). BOTH no-usable-PR cases — a TERMINAL PR
# (#1897) and NO PR OBJECT AT ALL (#2240 / #2235) — are payload-aware:
# the layered predicate below decides between "create a fresh PR and
# merge" and "genuinely nothing to merge". The pre-#2240 `-z "$PR"` arm
# skipped UNCONDITIONALLY and posted nothing, so a code-bearing branch
# whose Step 4a draft-PR create never fired was left permanently
# unmerged with the durable record reading clean (#456->#466 class;
# the completed_unmerged_pass watcher flag is blind to it because no
# marker is posted).
USABLE_PR=no
NOVEL_PAYLOAD=yes   # defensive init: fail-SAFE toward "novel" even on a
                    # partial re-entry in a fresh shell that skips the
                    # prelude below (never let an unset var read as "landed")
if [ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ]; then
  USABLE_PR=yes
fi
if [ "$USABLE_PR" != yes ]; then
  # Novel-payload predicate runs ONLY here — the OPEN-PR common path
  # takes no extra fetch (#2240). A bare COMMIT count is patch-blind:
  # the default merge forms land COPIES of the branch commits (--rebase
  # replays them, --squash folds them into one), so a fully-merged
  # branch reads `rev-list --count` > 0 forever (#1897 round-2).
  # Layered predicate, fail-SAFE toward "novel" (a false 'novel' costs
  # one bounded duplicate draft PR; a false 'landed' strands payload —
  # so every git-error path keeps NOVEL_PAYLOAD=yes):
  #   (1) zero commits ahead -> no payload (cheap short-circuit);
  #   (2) `git cherry` emits NO '+' line -> every commit is
  #       patch-equivalent upstream -> landed (rebase form: replayed
  #       commits keep their patch-ids; squash does NOT);
  #   (3) the branch's own changed files are content-identical to
  #       origin/main -> landed (squash form; also covers rebase);
  #   (4) else -> novel payload.
  timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
  NOVEL_PAYLOAD=yes
  if [ "$(git -C "$WT" rev-list --count origin/main..issue-<N>)" -eq 0 ]; then
    NOVEL_PAYLOAD=no   # (1) no commits at all
  elif CHERRY=$(git -C "$WT" cherry origin/main issue-<N>) \
       && [ -z "$(printf '%s\n' "$CHERRY" | grep '^+')" ]; then
    NOVEL_PAYLOAD=no   # (2) rebase-landed copies (a cherry FAILURE falls through — fail-safe)
  else
    OWN_FILES=$(git -C "$WT" diff --name-only origin/main...issue-<N>)
    if [ -n "$OWN_FILES" ] \
       && git -C "$WT" diff --quiet origin/main issue-<N> -- $OWN_FILES; then
      NOVEL_PAYLOAD=no # (3) squash-landed content (a diff ERROR keeps 'yes' — fail-safe)
    fi
  fi
  if [ "$NOVEL_PAYLOAD" = "yes" ]; then
    HAD_PRIOR_PR=no
    [ -n "$PR" ] && HAD_PRIOR_PR=yes
    # Title/body branch on whether a prior PR object existed at all.
    if [ "$HAD_PRIOR_PR" = yes ]; then
      PR_TITLE="issue-<N>: <task title> (round follow-up)"
      PR_BODY="Closes task #<N>. Fresh PR: prior PR #$PR is $PR_STATE (#1897 probe)."
    else
      PR_TITLE="issue-<N>: <task title>"
      PR_BODY="Closes task #<N>. Fresh PR: branch carries novel payload but no PR object exists (#2240 probe) — the Step 4a draft-PR create never fired."
    fi
    # STALE-REF ARM (#2312): the empty-ls-remote probe below covers a MISSING
    # ref (#2240's local-only branch); an EXISTING ref that is MUTUALLY
    # non-ancestral with HEAD — REWRITTEN branch or DIVERGED remote
    # (pod-side-reporting.md § Result-push, #1205/#1880) — makes the probe
    # non-empty, so no push fires and the create would open the PR on
    # content the sha-bound lint verdict never certified (#2296: the stale
    # head calls a helper signature #2293 removed — a fleet-wide break had
    # it merged). A STRICTLY-AHEAD remote (HEAD an ancestor) takes the else
    # arm unchanged. Route by descendancy BEFORE any create, then
    # discriminate per the safe-case guard comment below: all-foreign novel
    # remote-only commits ⇒ the pull-retry self-heal (after which the ref
    # is an ancestor and this arm re-enters cleanly); own superseded
    # payload or unsure ⇒ the § Rewritten-branch landing route — never a PR
    # on the stale ref. A one-script fall-through past the false lands in
    # the loud no-usable-PR epm:merge-failed arm below — never a silent
    # wrong merge (the safe-case and parity guards also block downstream).
    timeout --kill-after=30s 120s git -C "$WT" fetch origin issue-<N> --quiet 2>/dev/null || true
    if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \
       && ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \
       && ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then
      echo "[step10d] zero-PR arm BLOCKED — origin/issue-<N> and HEAD are mutually non-ancestral (#2312): REWRITTEN or DIVERGED. Refusing to open a PR on this ref. Discriminate (log/cherry HEAD..origin/issue-<N>): all-foreign ⇒ pull-retry self-heal (#1880's VM-side twin) then re-enter; own superseded payload or unsure ⇒ § Rewritten-branch landing route below."
      false
    else
    # ORIGIN PRECONDITION (#2240). `gh pr create --head` fails when the
    # head branch is not on the remote. The #1897 donor arm never needed
    # this guard — a prior PR object IMPLIES the branch reached origin,
    # and deleteBranchOnMerge=false keeps it there — but the zero-PR
    # arm has no such guarantee: issue-2117 was a live instance whose
    # branch was local-only (its origin heads listing came back empty
    # while the branch sat 1 commit ahead). Transplanting the create
    # without its enabling condition would strand exactly the payload
    # this fix exists to rescue. An early push is harmless: the merge
    # body below pushes the branch again anyway.
    if [ -z "$(git -C "$WT" ls-remote --heads origin issue-<N>)" ]; then
      git -C "$WT" push -u origin issue-<N> || {
        git -C "$WT" pull --rebase=merges --autostash origin issue-<N> || true
        git -C "$WT" push -u origin issue-<N>
      }
    fi
    # rc-GATE the create (#2240). A failed create (rate limit, auth,
    # branch still absent) must NEVER fall through into the
    # nothing-to-merge arm below — that would print a false message
    # and post a marker claiming a PR was opened. Both are the
    # "durable record reads clean while payload strands" class this
    # task exists to close.
    if gh pr create --draft --head issue-<N> --title "$PR_TITLE" --body "$PR_BODY"; then
      PR_INFO=$(gh pr view issue-<N> --json number,state,mergedAt \
        -q '[(.number | tostring), .state, (.mergedAt // "null")] | join(" ")')
      PR=$(echo "$PR_INFO" | cut -d' ' -f1)
      PR_STATE=$(echo "$PR_INFO" | cut -d' ' -f2)
      PRE_MERGED_AT=$(echo "$PR_INFO" | cut -d' ' -f3)
      [ -n "$PR" ] && [ "$PR_STATE" = "OPEN" ] && USABLE_PR=yes
    fi
    # Fail loud on the upstream miss (#2240), composed from the REALIZED
    # outcome — never from the intent. Step 4a gates its draft-PR create
    # on commits-ahead > 0 but runs BEFORE the implementer's first
    # commit, so its else arm fires by construction and nothing
    # re-opens the PR. Recovering silently would hide a standing fleet
    # defect; recording a recovery that did not happen would be worse
    # than silence.
    if [ "$HAD_PRIOR_PR" = no ]; then
      if [ "$USABLE_PR" = yes ]; then
        ANOMALY_NOTE="[step10d-no-pr-anomaly] Reached Step 10d with novel payload and ZERO PR objects for issue-<N> — the Step 4a draft-PR create never fired. Step 10d opened PR #$PR and is proceeding with the auto-merge (#2240)."
      else
        ANOMALY_NOTE="[step10d-no-pr-anomaly] Reached Step 10d with novel payload and ZERO PR objects for issue-<N>, and the recovery FAILED: gh pr create did not yield an OPEN PR. The branch is left UNMERGED and its payload is stranded — open a PR manually and re-run Step 10d (#2240)."
      fi
      uv run python "$REPO_ROOT/scripts/task.py" post-marker <N> epm:progress \
        --note "$ANOMALY_NOTE"
    fi
    fi
  fi
fi
if [ "$USABLE_PR" != yes ]; then
  if [ "$NOVEL_PAYLOAD" = "yes" ]; then
    # NOVEL PAYLOAD THAT COULD NOT BE ROUTED TO A PR — fail LOUD.
    # Pre-revision this fell into the nothing-to-merge arm and printed
    # a message that was factually false. The pre-#2240 terminal-PR
    # code caught the same failure loudly (the create fell through to
    # the merge attempt, whose landing verification routes to
    # epm:merge-failed); the restructure must not downgrade that to a
    # silent skip.
    echo "[step10d] NOVEL PAYLOAD ON issue-<N> COULD NOT BE MERGED: no usable PR object (gh pr create failed, or the fresh PR did not resolve OPEN). Branch left UNMERGED — this is a stranding risk, not a no-op."
    uv run python "$REPO_ROOT/scripts/task.py" post-marker <N> epm:merge-failed \
      --note "Step 10d could not obtain a usable PR object for issue-<N> while the branch carries novel payload vs origin/main. Branch left unmerged; payload NOT stranded silently. Recovery: open a PR for issue-<N> manually, then re-invoke Step 10d (#2240)."
  else
    # Genuinely nothing to merge: either no commits at all, or every
    # commit is patch-equivalent / content-identical upstream (rebase-
    # or squash-landed). Post nothing; do NOT run the guards/merge
    # below. Pre-#2240 this was two separate arms, one of which fell
    # through to the merge on prose alone (the terminal-PR skip);
    # routing both through the USABLE_PR gate makes that prose
    # contract executable control flow.
    if [ -n "$PR" ]; then
      echo "issue-<N> has no novel payload vs origin/main — nothing to merge (prior PR #$PR $PR_STATE stays the record)."
    else
      echo "issue-<N> has no PR and no novel payload vs origin/main — nothing to merge."
    fi
  fi
else
  # Run guards 1-3 above first. If guard 3 says "unsafe", skip this
  # block and run the artifact-confirmed merge below instead.
  #
  # Push the Guard-1 strip commit to the PR head ref FIRST, so the
  # server-side merge in `gh pr merge` below (rebase replay or squash)
  # sees the stripped branch tip,
  # not the pre-strip commit. The strip commit is a LOCAL worktree commit and
  # is otherwise invisible to the server-side merge — leaving the foreign
  # tasks/* reverts in the replayed history and landing them on main silently
  # (Codex code-review round-1 blocker, task #787). Push retry mirrors
  # CLAUDE.md § "Concurrent repo-root committers": pull --rebase=merges
  # --autostash then re-push on a rejected push. The Guard-0 agent-memory
  # persist commit is equally local-only — both must reach the PR head ref
  # before the server-side rebase.
  # (WORKTREE-scoped: `git -C "$WT"` on the issue branch. scripts/sync_repo_root.py
  # does NOT apply here — it is repo-root-only by design, preconditioned on
  # HEAD == main, exit 5 otherwise.)
  #
  # The push condition RE-DERIVES "unpushed local commits exist" from git
  # state (rev-list against origin/issue-<N>) instead of trusting the
  # STRIPPED_FOREIGN / MEM_COMMITTED flags alone: fenced bash blocks are
  # SEPARATE shell invocations, so a flag assigned in Guard 0/1 is unset
  # here (and would silently skip the second-chance push); the git-state
  # read also survives a crash + re-entry, and covers BOTH the strip commit
  # and the memory commit in one predicate. The flags stay as same-block
  # conveniences only (they still short-circuit true when guards and merge
  # happen to run in one shell). A missing / unresolvable origin/issue-<N>
  # ref counts as unpushed (`|| echo 1` — fails toward pushing, the safe
  # direction; a redundant push is a no-op "Everything up-to-date").
  if [ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null || echo 1)" -gt 0 ] \
     || [ "$STRIPPED_FOREIGN" = "yes" ] || [ "$MEM_COMMITTED" = "yes" ]; then
    # REWRITTEN-OR-DIVERGED GUARD (#2312): a MUTUALLY non-ancestral
    # origin/issue-<N> (two-leg test — the Step-4a root-divergence probe's
    # own shape; a STRICTLY-AHEAD remote takes the else arm, where the
    # rejected push falls into the pull-retry that rebases onto the correct
    # same-branch upstream — today's behavior, #1880's self-heal) means the
    # branch was REWRITTEN (a mid-flight rebase onto
    # origin/main — the count predicate above is then satisfied trivially,
    # the push is rejected non-fast-forward, and the refspec-less pull
    # fallback would rebase HEAD onto the STALE remote branch: the #1128
    # replay shape; #2296 measured [ahead 363, behind 1]) OR the remote
    # genuinely DIVERGED (novel FOREIGN commits: pod/GCE lanes commit
    # results to the issue branch by contract — pod-side-reporting.md
    # § Result-push verification contract, #1205/#1880 — and the pull-retry
    # pair IS the sanctioned self-heal, the VM-side twin of #1880's
    # lane-side fetch+rebase). No mechanical discriminator separates the
    # two (a conflict-reconciling rebase defeats git-cherry
    # patch-equivalence), so the guard fails CLOSED for both and the echo
    # instructs the inspection. Descendancy, not count, is the safe-to-push
    # test (precedents: the Step-4a root-divergence probe, gotchas.md
    # fix-engaged verification). Missing ref ⇒ push arm (fail-toward-
    # pushing, same direction as the `|| echo 1` above). Guarded
    # exclusive-arm shape (#1184); the wrapped else arm is deliberately NOT
    # re-indented (pure #2312 insertion — provably byte-identical ordinary
    # path).
    if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \
       && ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \
       && ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then
      echo "[step10d] safe-case push BLOCKED — origin/issue-<N> and HEAD are mutually non-ancestral (#2312): REWRITTEN or DIVERGED remote. Do NOT push blind and never force-push. Discriminate per the guard comment above (log/cherry the remote-only commits): all-foreign novel work ⇒ run the pull-retry pair by hand (the sanctioned self-heal, the VM-side twin of the #1880 lane-side fetch+rebase recipe), then re-enter this block; own superseded pre-rebase payload, or unsure ⇒ the § Rewritten-branch landing route below; mixed ⇒ cherry-pick the foreign commits onto HEAD (the tip moves — re-run the lint gate), then the landing route."
      false
    else
    # Run every push / gh pr command BARE — never piped through tail/grep/head
    # (guard_piped_git_push.sh blocks the pipe; a pipe masks the exit code).
    # This applies to IMPROVISED recovery commands too, not just this snippet.
    git -C "$WT" push origin issue-<N> \
      || { git -C "$WT" pull --rebase=merges --autostash \
           && git -C "$WT" push origin issue-<N>; }
    fi
  fi
  # Merge-form routing (#1288): infra-fleet code branches (kind infra|batch —
  # the watcher's INFRA_DRAIN_KINDS, the population same-batch racing this
  # step by construction) default to --squash: server-side --rebase went 0/4
  # first-try under fleet churn and every failed session landed on --squash
  # anyway. GitHub mergeability is merge-method-
  # independent, but --rebase can ADDITIONALLY fail ("can't be rebased",
  # #1041) where --squash succeeds — so squash-first strictly dominates for
  # a single-logical-change branch, and it reverts as ONE commit (the only
  # grain that exists on such a branch). Experiments (Step 9b trigger) keep
  # --rebase: heterogeneous per-item commits retain per-commit revert value
  # on the clean path, and the 07-12→07-17 conflicted-experiment record is
  # shape-2-dominated (method-independent — see the merge-form paragraph
  # below, #1493): squash-first buys nothing there. An unreadable kind falls
  # to --rebase (fail-open to today's behavior). REPO_ROOT is re-derived
  # inline — fenced blocks are separate shells, and the guards block's
  # derivation is not in scope here:
  REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  MERGE_FORM=--rebase
  TASK_KIND=$(uv run python "$REPO_ROOT/scripts/task.py" view <N> --json \
    | uv run python -c 'import sys,json; print(json.load(sys.stdin).get("frontmatter",{}).get("kind",""))' \
    || echo "")
  case "$TASK_KIND" in infra|batch) MERGE_FORM=--squash ;; esac
  # Pre-push workflow-lint gate (subsection above) — run its executable
  # block FIRST as ONE BACKGROUND Bash call, read the verdict file in a
  # fresh foreground call when it completes (completion-read, gate
  # subsection), then gate the merge on the PERSISTED, SHA-BOUND verdict
  # file: the explicit conditional below is the hard stop. Fails CLOSED on
  # a missing file (gate not run), a block/crash verdict, OR a missing /
  # stale sha (line 2 empty or != current tip: a hand-written verdict, or
  # new commits since certification — re-run the gate). The verdict is
  # consumed (rm) only AFTER `gh pr merge` SUCCEEDS: a non-lint transport
  # failure (#1041 rebase refusal) leaves it valid for the same-tip retry
  # — never hand-write the verdict file (#1082; sole exception: the
  # mechanically-gated RE-BIND stanza below, line 2 only).
  # Read this conditional VERBATIM (#2006). Do NOT re-compose it — an
  # improvised `grep -qxE … <(sed …)` process-substitution form inside an
  # eval'd guard prelude exited 1 "BLOCKED: verdict missing/stale" while an
  # immediate re-probe showed verdict `pass` + sha == tip, costing ~3 h and
  # three full gate runs. The committed three-conjunct form below is the
  # tested one; a re-compose is a fresh, untested predicate on the merge path.
  # Note the grep scans the WHOLE file, not line 1 — which is why the
  # never-hand-write rule (above) is load-bearing here: a hand-appended
  # `pass` on line 3 satisfies it while line 2 still carries a valid sha.
  if grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt 2>/dev/null \
     && [ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ] \
     && [ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" = "$(git -C "$WT" rev-parse HEAD)" ]; then
    # Post-gate freshness re-sync (#1714): the lint gate has PASSed against
    # origin/main-as-of-gate-start; origin/main may have advanced during the
    # ~30-min gate window. Re-run the Step 5a family-atomic block with source
    # origin/main immediately before the merge to minimize the merge-race
    # window. Uses the ALREADY-BOUND $WT (do NOT re-derive from cwd; a
    # repo-root cwd would rebind to the shared root).
    timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet || true
    # --- inline Step 5a family-atomic block (origin/main source, same as
    # Step 5a itself as of #1747; WT pre-bound, no on-main skip needed) ---
    declare -A FAMILY_OF
    FAMILY_OF[".claude/workflow.yaml"]="workflow"
    FAMILY_OF[".claude/skills"]="workflow"
    FAMILY_OF["tests/test_workflow_yaml.py"]="workflow"
    FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"
    FAMILY_OF["scripts/step5a_sibling_probe.py"]="workflow"
    FAMILY_OF["tests/test_step5a_sibling_probe.py"]="workflow"
    FAMILY_OF["scripts/workflow_lint.py"]="lint"
    FAMILY_OF[":(glob)tests/test_workflow_lint*.py"]="lint"
    FAMILY_OF["tests/test_autonomous_session_watch.py"]="lint"
    FAMILY_OF["scripts/select_step9c_tests.py"]="lint"
    FAMILY_OF["tests/test_select_step9c_tests.py"]="lint"
    FAMILY_OF["tests/step9c_workflow_invariant_manifest.txt"]="lint"
    FAMILY_OF[".claude/config/agent_spec_size_caps.txt"]="lint"
    FAMILY_OF[".claude/hooks"]="guard"
    FAMILY_OF[":(glob)scripts/guard_*.sh"]="guard"
    FAMILY_OF[":(glob)tests/test_guard_*.py"]="guard"
    FAMILY_OF["tests/test_guard_lessons_edit.py"]="guard"
    # FAMILY_agents members (#2260)
    FAMILY_OF[".claude/agents"]="agents"
    FAMILY_OF["tests/test_adversarial_planner_factchecker_grain_pin.py"]="agents"
    FAMILY_OF["tests/test_adversarial_planner_lens_brief_headings.py"]="agents"
    FAMILY_OF["tests/test_analyzer_language_intrusion_duty.py"]="agents"
    FAMILY_OF["tests/test_battery_basis_prose_pins.py"]="agents"
    FAMILY_OF["tests/test_code_reviewer_phase_idempotency_gate.py"]="agents"
    FAMILY_OF["tests/test_codex_code_reviewer_step09_tag_parity.py"]="agents"
    FAMILY_OF["tests/test_codex_critic_numeric_grounding.py"]="agents"
    FAMILY_OF["tests/test_consistency_checker_parentless_infra_skip.py"]="agents"
    FAMILY_OF["tests/test_cross_issue_protocol_comparability_prose.py"]="agents"
    FAMILY_OF["tests/test_daily_three_route_classifier_doc.py"]="agents"
    FAMILY_OF["tests/test_diff_base_origin_main_pin.py"]="agents"
    FAMILY_OF["tests/test_downwidth_split_prose_pins.py"]="agents"
    FAMILY_OF["tests/test_experimenter_md.py"]="agents"
    FAMILY_OF["tests/test_fit_loop_batching_review_pin.py"]="agents"
    FAMILY_OF["tests/test_implementer_spec_deleted_literal_substep.py"]="agents"
    FAMILY_OF["tests/test_implementer_spec_mechanical_pin_sweep.py"]="agents"
    FAMILY_OF["tests/test_implementer_spec_names_invariant_local_union.py"]="agents"
    FAMILY_OF["tests/test_implementer_spec_names_ruff_policy_pin.py"]="agents"
    FAMILY_OF["tests/test_interp_critic_degenerate_series_lens.py"]="agents"
    FAMILY_OF["tests/test_issue_v2_skill_figure_pin_contract.py"]="agents"
    FAMILY_OF["tests/test_lean_twin_registration_pin.py"]="agents"
    FAMILY_OF["tests/test_mapping_baselines_wiring_pins.py"]="agents"
    FAMILY_OF["tests/test_off_pod_phase_slot_pin.py"]="agents"
    FAMILY_OF["tests/test_outroot_residue_prose_pins.py"]="agents"
    FAMILY_OF["tests/test_plan_handoff_path_convention.py"]="agents"
    FAMILY_OF["tests/test_planner_incident_trace_guidance.py"]="agents"
    FAMILY_OF["tests/test_planner_phase_outputs_declaration.py"]="agents"
    FAMILY_OF["tests/test_realized_rows_prose_pins.py"]="agents"
    FAMILY_OF["tests/test_selection_symmetric_nulls_pointers.py"]="agents"
    FAMILY_OF["tests/test_v2_composer_plan_path_brief.py"]="agents"
    FAMILY_OF["tests/test_inline_payload_lint_gate_contract.py"]="workflow"
    SPECS_10D=".claude/agents .claude/agent-memory .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py .claude/config/agent_spec_size_caps.txt scripts/select_step9c_tests.py .claude/hooks :(glob)scripts/guard_*.sh tests/test_guard_lessons_edit.py tests/test_workflow_yaml.py tests/test_autonomous_session_watch.py tests/test_select_step9c_tests.py tests/step9c_workflow_invariant_manifest.txt :(glob)tests/test_workflow_lint*.py :(glob)tests/test_guard_*.py tests/issue_skill_source.py :(glob)tests/test_issue_skill_*.py scripts/step5a_sibling_probe.py tests/test_step5a_sibling_probe.py tests/test_adversarial_planner_factchecker_grain_pin.py tests/test_adversarial_planner_lens_brief_headings.py tests/test_analyzer_language_intrusion_duty.py tests/test_battery_basis_prose_pins.py tests/test_code_reviewer_phase_idempotency_gate.py tests/test_codex_code_reviewer_step09_tag_parity.py tests/test_codex_critic_numeric_grounding.py tests/test_consistency_checker_parentless_infra_skip.py tests/test_cross_issue_protocol_comparability_prose.py tests/test_daily_three_route_classifier_doc.py tests/test_diff_base_origin_main_pin.py tests/test_downwidth_split_prose_pins.py tests/test_experimenter_md.py tests/test_fit_loop_batching_review_pin.py tests/test_implementer_spec_deleted_literal_substep.py tests/test_implementer_spec_mechanical_pin_sweep.py tests/test_implementer_spec_names_invariant_local_union.py tests/test_implementer_spec_names_ruff_policy_pin.py tests/test_inline_payload_lint_gate_contract.py tests/test_interp_critic_degenerate_series_lens.py tests/test_issue_v2_skill_figure_pin_contract.py tests/test_lean_twin_registration_pin.py tests/test_mapping_baselines_wiring_pins.py tests/test_off_pod_phase_slot_pin.py tests/test_outroot_residue_prose_pins.py tests/test_plan_handoff_path_convention.py tests/test_planner_incident_trace_guidance.py tests/test_planner_phase_outputs_declaration.py tests/test_realized_rows_prose_pins.py tests/test_selection_symmetric_nulls_pointers.py tests/test_v2_composer_plan_path_brief.py"
    MB_10D=$(git -C "$WT" merge-base HEAD origin/main)
    declare -A DIRTY_FAMILIES_10D
    for f in $SPECS_10D; do
      # Member-existence containment (#2260; interaction with #2385): the
      # checkout below is ATOMIC — a single literal token absent at origin/main
      # (deleted/renamed on main) errors the whole checkout and syncs NOTHING,
      # wedging every family until manual reconcile. Contain per-family: an
      # absent literal member marks ITS family dirty (vintage-consistent skip;
      # other families keep syncing). Deletion PROPAGATION (removing the stale
      # worktree twin) remains #2385 — reconcile manually until it lands.
      case "$f" in
        ":(glob)"*) : ;;
        *)
          if ! git -C "$WT" cat-file -e "origin/main:$f" 2>/dev/null; then
            fam="${FAMILY_OF[$f]:-$f}"
            DIRTY_FAMILIES_10D[$fam]=1
            echo "spec-freshness: $f is ABSENT at origin/main (deleted/renamed on main) — marking family '$fam' dirty; skipping blind sync for the whole family (atomic-checkout containment, #2260; stale-twin removal is #2385 — reconcile manually)."
            continue
          fi
          ;;
      esac
      bs_commits=$(git -C "$WT" log --format='%H %s' "$MB_10D"..HEAD -- "$f" \
        | awk 'index($0, "sync workflow-surface specs from") == 0')
      if [ -n "$bs_commits" ]; then
        fam="${FAMILY_OF[$f]:-$f}"
        DIRTY_FAMILIES_10D[$fam]=1
        # agent-memory re-aligns here carry the Step 5a no-lost-row duty (gotchas.md).
      fi
      # Uncommitted-dirt arm (#1972) — mirror of Step 5a's (structurally
      # parallel; at 10d Guard 0 has usually already committed memory dirt,
      # so this is typically a no-op here — fail-safe either way): tracked
      # dirt always marks the family dirty; a ?? path only on an
      # origin/main path collision the checkout below could clobber.
      DIRT=""
      while IFS= read -r line; do
        [ -z "$line" ] && continue
        p=${line:3}; p=${p%/}
        if [ "${line:0:2}" = "??" ]; then
          git -C "$WT" cat-file -e "origin/main:$p" 2>/dev/null && DIRT=yes
        else
          DIRT=yes
        fi
      done < <(git -C "$WT" -c core.quotePath=false status --porcelain -- "$f")
      if [ -n "$DIRT" ]; then
        fam="${FAMILY_OF[$f]:-$f}"
        DIRTY_FAMILIES_10D[$fam]=1
        echo "spec-freshness: $f carries UNCOMMITTED changes the sync could clobber — marking family '$fam' dirty; skipping blind sync for the whole family (#1972)."
      fi
    done
    SAFE_SPECS_10D=""
    for f in $SPECS_10D; do
      fam="${FAMILY_OF[$f]:-$f}"
      if [ -z "${DIRTY_FAMILIES_10D[$fam]}" ]; then
        SAFE_SPECS_10D="$SAFE_SPECS_10D $f"
      fi
    done
    SYNC_SHA="no-drift"
    if [ -n "$SAFE_SPECS_10D" ] && ! git -C "$WT" diff --quiet origin/main -- $SAFE_SPECS_10D; then
      git -C "$WT" checkout origin/main -- $SAFE_SPECS_10D
      if ! git -C "$WT" diff --quiet HEAD -- $SAFE_SPECS_10D; then
        if ! git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs from origin/main (spec-freshness)" -- $SAFE_SPECS_10D; then
          # Aborts the WHOLE Step 10d merge invocation (this stanza runs inside
          # the merge sequence's Bash call) BEFORE gh pr ready/merge — fail-closed.
          # The lint verdict file is left intact for the same-tip retry after remediation.
          echo "[step10d] FATAL: post-gate re-sync commit FAILED (rc != 0) — synced set left STAGED for inspection; merge attempt ABORTED. Failed paths:" >&2
          git -C "$WT" diff --cached --name-only -- $SAFE_SPECS_10D | sed 's/^/  /' >&2
          exit 1
        fi
        SYNC_SHA=$(git -C "$WT" rev-parse HEAD | head -c 12)
        # Push the new sync commit so gh pr merge sees it on the PR head ref
        # (descendancy-guarded, #2312 — canonical snippet (1) form; reachable
        # on shape-0/3 re-entries without re-running the safe-case guard):
        if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \
           && ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \
           && ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then
          echo "[step10d] post-gate re-sync push BLOCKED — mutually non-ancestral remote (#2312): REWRITTEN or DIVERGED; discriminate per the safe-case guard comment (all-foreign ⇒ pull-retry self-heal; else § Rewritten-branch landing route)."
          false
        else
        git -C "$WT" push origin issue-<N> \
          || { git -C "$WT" pull --rebase=merges --autostash \
               && git -C "$WT" push origin issue-<N>; }
        fi
      fi
    fi
    SYNC_COUNT=$(echo $SAFE_SPECS_10D | wc -w)
    echo "[step10d] post-gate re-sync: synced $SYNC_COUNT files ($SYNC_SHA) | no drift"
    # Deliberately NO sibling-issue per-FILE arm here (#1972): the 10d TG
    # legs run BEFORE this post-gate re-sync, so syncing sibling issue<M>
    # files at this point would only move the tip after certification for
    # zero gate benefit — the Step 5a block (+ its Step 9c step-1a binding
    # reference) carries that arm.
    # --- end inline Step 5a family-atomic block ---
    # Verdict RE-BIND stanza (#1807): a re-sync that COMMITTED moved the tip
    # past the verdict's certified sha (line 2), so a forced gate re-run
    # would follow even though the gate's landing tree (git archive
    # origin/main + own-diff overlay) is unchanged by a payload-free sync
    # commit. Mechanically verify that: enumerate the cert-sha..HEAD delta
    # with --name-status, NOT --name-only — a both-sides-absent DELETION (a
    # stray non-sync commit deleting a branch-added file) exits
    # `git diff --quiet origin/main HEAD -- <p>` ZERO, reading as
    # "main-identical" while the certified landing tree CONTAINED the file
    # via the own-diff overlay. The sync block's
    # `checkout origin/main -- $SAFE_SPECS_10D` can only add/modify, never
    # delete, so ANY D/R*/C*/T/U status row is by construction non-sync
    # output: fail CLOSED unconditionally. A/M rows keep the byte-identity
    # probe (content == fetched origin/main contributes nothing beyond the
    # baseline to the landing tree, so the certification is unchanged).
    # #1082 carve-out, TIGHT: line 1 is NEVER touched; only line 2 moves;
    # the re-bind is licensed ONLY as executed by THIS stanza over its own
    # --name-status probe output — a free-standing "update line 2" move
    # stays banned, and the #1613 empty-commit synchronize explicitly STAYS
    # a stale-verdict -> gate-re-run case, never hand-re-bound.
    REBIND_OK=yes
    if [ "$SYNC_SHA" != "no-drift" ]; then
      CERT_SHA=$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt)
      REBIND_OK=no
      DELTA_OK=yes
      while IFS=$'\t' read -r st p _rest; do
        case "$st" in
          A|M) git -C "$WT" diff --quiet origin/main HEAD -- "$p" || DELTA_OK=no ;;
          *)   DELTA_OK=no ;;   # D / R* / C* / T / U — never sync output
        esac
      done < <(git -C "$WT" diff --name-status "$CERT_SHA" HEAD)
      if [ "$DELTA_OK" = yes ]; then
        # Line 1 is COMPOSED from the existing verdict (sed -n 1p), never
        # typed; only line 2 (the certified sha) moves to the new tip.
        if { sed -n 1p /tmp/issue-<N>-lint-verdict.txt; git -C "$WT" rev-parse HEAD; } \
             > /tmp/issue-<N>-lint-verdict.rebind \
           && mv /tmp/issue-<N>-lint-verdict.rebind /tmp/issue-<N>-lint-verdict.txt; then
          REBIND_OK=yes
          echo "[step10d] verdict re-bound to sync tip $(git -C "$WT" rev-parse --short=12 HEAD) (delta = origin/main-identical spec sync only; #1807)"
        else
          echo "[step10d] verdict re-bind WRITE failed — fail CLOSED (re-run the gate; the BLOCKED arm below consumes the stale verdict)"
        fi
      else
        echo "[step10d] sync delta NOT verifiable as origin/main-identical A/M-only — verdict stays bound to $CERT_SHA; fail CLOSED (re-run the gate; the BLOCKED arm below consumes the stale verdict)"
      fi
    fi
    if [ "$REBIND_OK" = yes ]; then
      # Head-sync pre-check (#1657, READ-ONLY — runs AFTER the post-gate
      # re-sync + re-bind above so it polls PR-object parity against the
      # FINAL tip): every push this invocation made (Guard-0/1 commits, the
      # post-gate re-sync commit just above) races GitHub's PR-object
      # sync — #1614's attempts 1-2 were refused
      # 'Head branch is out of date' while the PR object lagged the pushed
      # tip ~6 min. Poll until the PR object reports the local tip AND a
      # settled mergeability; a settled CONFLICTING exits too (the merge
      # attempt then classifies to shape 2 below, unchanged). Check-first
      # bounded until-loop — never a leading foreground sleep
      # (harness-blocked; the shape-0 convention).
      TIP=$(git -C "$WT" rev-parse HEAD); HS_TRIES=0
      until HS=$(gh pr view "$PR" --json headRefOid,mergeable -q '.headRefOid + " " + .mergeable' 2>/dev/null) \
            && [ "${HS%% *}" = "$TIP" ] && [ "${HS##* }" != "UNKNOWN" ]; do
        HS_TRIES=$((HS_TRIES + 1))
        if [ "$HS_TRIES" -ge 6 ]; then
          echo "head-sync pre-check: PR object still stale after ~2 min (saw: ${HS:-<no read>}; local tip: $TIP) — routing to the PR-head parity gate (#2312) below"
          break
        fi
        sleep 20
      done
      if [ "${HS%% *}" = "$TIP" ]; then
        echo "head-sync pre-check: parity at $TIP (mergeable=${HS##* })"
      fi
      # PR-HEAD PARITY GATE (#2312 — extends the #1657 pre-check above): the
      # poll treats a headRefOid mismatch as PR-object LAG to wait out; on a
      # REWRITTEN branch (origin/issue-<N> not an ancestor of HEAD) the
      # mismatch is GENUINE staleness — the head ref does not hold the
      # gate-certified tip, and gh pr merge would land PRE-rebase content no
      # gate certified (#2296's near-miss: a helper signature #2293 removed).
      # Discriminate by the REMOTE REF (authoritative), not the PR object:
      # ref == TIP ⇒ pure object lag (proceed — today's behavior; shape 3
      # recovers), otherwise fail CLOSED — the verdict sha-bind's posture,
      # extended to the PR head. A transport-failed/empty ls-remote read
      # fails CLOSED too (re-enter after a re-probe). Wrapped else arm
      # deliberately NOT re-indented (pure #2312 insertion).
      REMOTE_REF=$(git -C "$WT" ls-remote --heads origin issue-<N> | cut -f1)
      if [ "${HS%% *}" != "$TIP" ] && [ "$REMOTE_REF" != "$TIP" ]; then
        echo "BLOCKED: PR-head parity (#2312) — neither the PR object (saw: ${HS:-<no read>}) nor the remote ref (${REMOTE_REF:-<none>}) holds the gate-certified tip $TIP. The tip never landed on the head ref (rejected push, or a rewritten/diverged branch): if origin/issue-<N> is not an ancestor of HEAD, discriminate per the safe-case guard comment above (all-foreign ⇒ pull-retry self-heal; else § Rewritten-branch landing route below); otherwise re-run the safe-case push and re-enter this conditional. Do NOT merge; verdict NOT consumed."
        false
      else
      # Draft-merge precondition (#2240 pin): this single `gh pr ready` call
      # marks the PR ready before the merge below and covers PRs opened as
      # drafts by EITHER fresh-PR arm (the #1897 terminal-PR create and the
      # #2240 zero-PR create both fall through into exactly this block) —
      # do NOT add a second ready call elsewhere.
      gh pr ready "$PR"
      if gh pr merge "$PR" $MERGE_FORM --delete-branch=false; then
        # Landing verification (#1897): exit 0 is NOT proof THIS attempt
        # landed — `gh pr merge` on an already-merged PR exits 0 with
        # "was already merged" (#1768 round-2). Verify via the PR object
        # (never branch-sha ancestry: a rebase merge lands rebased
        # COPIES — new shas). Check-first bounded poll for GitHub's
        # async state settle; the empty-PRE_MERGED_AT conjunct fails
        # CLOSED (a partial re-entry in a fresh shell leaves it unset).
        LANDED_OK=no
        for _ in 1 2 3; do
          POST=$(gh pr view "$PR" --json state,mergedAt \
            -q '[.state, (.mergedAt // "null")] | join(" ")' 2>/dev/null) || POST=""
          if [ -n "$PRE_MERGED_AT" ] && [ "${POST%% *}" = "MERGED" ] \
             && [ "${POST##* }" != "null" ] \
             && [ "${POST##* }" != "$PRE_MERGED_AT" ]; then LANDED_OK=yes; break; fi
          sleep 10
        done
        if [ "$LANDED_OK" = yes ]; then
          rm -f /tmp/issue-<N>-lint-verdict.txt   # consume on VERIFIED merge success only — the verdict certified exactly the tip that landed
          # Root-sync before epm:merged (#1725, safe-case): the just-merged diff is on
          # origin/main; a workflow-surface fix in it is NOT yet live at the
          # shared repo root, and the very next call — the epm:merged post —
          # runs argv-prose guards from the pre-fix root copy (session
          # 7ce3a81f).
          # sync_repo_root.py is single-flight flock-serialized; fail-soft
          # (the post-merge-guard pre-sync at the guard block below remains the fallback).
          uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
            echo "[step10d/safe-case] pre-marker sync failed; post-merge-guard pre-sync remains the fallback"
        else
          echo "MERGE NOT VERIFIED — gh pr merge exited 0 but the PR object shows no FRESH merge (state/mergedAt unchanged vs pre-attempt: the exit-0 'was already merged' false-success shape, #1768/#1897). Verdict NOT consumed; re-enter via the PR-state probe at the top of this block (fresh PR) AT MOST ONCE per Step 10d invocation — a SECOND unverified exit-0 success -> epm:merge-failed. Do NOT report success."
          false
        fi
      else
        echo "MERGE FAILED — classify the gh error text: (0) \"Base branch was modified\" -> transient base-advance (Known failure shape 0 below): wait ~20s via a bounded until-loop or a bg-Bash re-check — NEVER a leading foreground \`sleep\` (harness-blocked; 3 wasted turns on 2026-07-18 alone) — then re-enter this SAME conditional (the verdict still certifies the tip; max 2 re-entries per Step 10d invocation, counted regardless of re-bind, #1807); (1) \"can't be rebased\" (--rebase form only) -> the #1041 --squash retry (Known failure shape 1 below; SHA-bound verdict remains valid for the SAME tip); (2) \"Pull Request has merge conflicts\" -> the #1128 re-snapshot-and-retry-once (Known failure shape 2 below); (3) \"Head branch is out of date\" -> PR head-sync lag (Known failure shape 3 below): confirm pushed, bounded headRefOid re-poll, close/reopen nudge ONCE if still stale, then re-enter this SAME conditional (the verdict still certifies the tip; max 2 re-entries per Step 10d invocation, counted regardless of re-bind — #1807); (4) anything else -> the Failure bullet (merge-conflict recovery ONCE, then epm:merge-failed). Do NOT hand-write the verdict file."
        false
      fi
      fi
    else
      echo "BLOCKED: verdict re-bind failed — the post-gate sync moved the tip and its delta could not be verified as origin/main-identical A/M-only (or the re-bind write failed): the stale SHA-bound verdict cannot certify the new tip. Re-run the pre-push workflow-lint gate against the new tip, then re-enter this conditional. Do NOT merge; do NOT hand-write the verdict file (#1082)."
      rm -f /tmp/issue-<N>-lint-verdict.txt   # stale-sha verdict consumed — a fresh gate run regenerates it (the Verdict bullet's stale-sha removal branch)
      false
    fi
  else
    echo "BLOCKED: pre-push workflow-lint gate (verdict: $(cat /tmp/issue-<N>-lint-verdict.txt 2>/dev/null || echo not-run)) — missing verdict, block/crash, or missing/stale sha (hand-written verdict, or new commits since certification) all fail CLOSED: fix the named offender (or crash cause), re-sync (§ pre-gate re-sync), then re-run the gate ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Do NOT merge."
    rm -f /tmp/issue-<N>-lint-verdict.txt   # block/crash/stale consumed — a fresh gate run regenerates it
    false
  fi
fi
```

The `gh pr merge --rebase` form lands all per-item commits individually
on `main`; each is independently revertible via `git revert <sha>` run in
a scratch worktree (the root guard blocks a repo-root revert, #1234) (vs.
`--merge`, which reverts everything together). The user retains full
revert control after the fact — that is what makes a no-prompt merge safe
here. The worktree is deliberately NOT removed (`--delete-branch=false`,
no `git worktree remove`).

For `kind: infra|batch` branches `$MERGE_FORM` is `--squash` (#1288):
a single logical change lands as ONE independently-revertible commit,
and the empirical record (0/4 first-try rebases in one fleet day) makes
the rebase attempt a pure wall-time tax. Shape 1 cannot fire on the
--squash path (the error is rebase-specific); shapes 0/2/else apply to
both forms.

`kind: experiment` branches keep `--rebase` deliberately (#1493, which
updates the #1288 no-evidence rationale): the 07-12→07-17 record — 210
`epm:merged` (attempt split of the `merge_attempts`-annotated subset:
160 attempt-1 / 20 attempt-2 / 3 attempt-3), zero `epm:merge-failed`
since 07-05 — shows every CLASSIFIED conflicted experiment first
refusal was shape 2 (mergeability — method-independent: GitHub's
mergeability state is a 3-way test merge that declines `--squash` and
`--rebase` identically) or shape 0 (transient), with zero shape-1 first
refusals on record; and
#1310 recorded a FIRST `--squash` refused on the same shape-2
mergeability — squash-first saves nothing in any classified case,
while the clean path retains per-commit revert value under `--rebase`.
Revisit criterion: extend squash-first to `kind: experiment` if shape-1
(`can't be rebased`) FIRST refusals appear on experiment branches —
shape 1 is the only failure shape squash-first avoids.

**Known failure shape 0 — base branch advanced mid-merge (`Base branch
was modified`, #1288).** Substring-match `Base branch was modified` (the
full wording — `Base branch was modified. Review and try the merge
again.` — is transcript-mined and may drift). GitHub recomputed the
merge against a base that moved DURING the API call — a pure timing
transient under fleet marker churn (~100+ tasks/ commits/hr on main):
no content conflict, nothing to fix. Recovery: wait ~20 s (≈ one churn
interval, letting gh's mergeability recompute settle), then re-enter
the SAME gated merge conditional with the SAME `$MERGE_FORM` — the
failed merge changed nothing locally, so the SHA-bound verdict still
certifies the tip (consume-on-merge-success survives this failure by
design; never hand-write the verdict file, #1082); the re-entered
safe-case block may legitimately MOVE the tip via a second post-gate
sync + verdict re-bind (#1807). Bounded at TWO re-entries per Step 10d
invocation, counted per invocation REGARDLESS of re-bind (the bound
keys on re-entries of this conditional, not on tip identity); a third
consecutive hit is no longer plausibly
timing — reclassify by error text per shapes 1/2/3/else.

Before each retried merge call, post an `[long-phase-heartbeat]`
progress note so the stalled detector, `tick_triage.py`, and downstream
sessions can tell an in-flight retry from a stranded merge (#1723;
same long-phase-heartbeat family recognized by
`autonomous_session_watch._long_phase_heartbeat_reason`):

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "[long-phase-heartbeat] step10d-merge attempt=<k> shape=0"
``` Before #1288
this shape fell through to the "anything else" catch-all (now class
(4) after #1657) and burned a full scratch-worktree recovery on a
transient.

**Known failure shape 1 — branch carries a merge commit (`can't be
rebased`, #1041).** A branch that CARRIES A MERGE COMMIT (e.g. after a
conflict-resolution merge of `main` into the branch) cannot be
server-side rebased — `gh pr merge --rebase` fails with
`GraphQL: This branch can't be rebased`. The working recovery is
`gh pr merge <PR> --squash --delete-branch=false` (acceptable for a
single-logical-change branch; the squash loses per-commit revert
granularity, which the merge commit already compromised). (#1041.)
The SHA-bound verdict file SURVIVES this failure by design
(consume-on-merge-success): run the squash retry through the SAME gated
conditional (substituting `--squash` for `--rebase`) so the still-valid
verdict re-certifies the identical tip and the `rm` fires on success.
Never recreate the verdict file by hand — a hand-written verdict lacks
the certified sha and fails closed anyway (#1082's
`echo pass > /tmp/issue-<N>-lint-verdict.txt` is the banned move).

**Known failure shape 2 — mergeability conflict under fleet marker churn
(error text containing `Pull Request has merge conflicts`, #1128).**
Classify by SUBSTRING, never the exact GraphQL line (the full
wording is transcript-mined and may drift). Between the Guard-1 snapshot and the
server-side rebase, `main` advances (~100+ `tasks/` marker commits/hr),
so the strip commit's snapshot replays stale and conflicts. Recovery:
re-snapshot against a freshly captured `main` SHA and retry ONCE —
documented, never silent, and gated on the re-snapshot actually changing
something (an unchanged tip would fail identically; go straight to the
merge-conflict recovery instead). NOTE the same error text ALSO fires for
non-`tasks/` conflicts (overlapping workflow-surface edits, binary
`figures/` collisions — #697/#597, resolved mechanically by the
binary-figures newer-regeneration-wins recipe in the merge-conflict
recovery below) that a re-snapshot cannot fix: the
skip-predicate fall-through is the EXPECTED path there, not a
malfunction — likewise an ORDINARY branch commit that itself touched
foreign `tasks/` at stale content. Even a fresh snapshot can go stale
between fetch and server-side merge — the recipe bounds and mechanizes
recovery; it does not eliminate the race.

Before the re-snapshot-and-retry runs, post an `[long-phase-heartbeat]`
progress note (#1723; same family as shape 0 above):

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "[long-phase-heartbeat] step10d-merge attempt=<k> shape=2"
```

```bash
# Re-snapshot-and-retry (ONCE per Step 10d invocation) — fires ONLY on
# the mergeability-conflict shape above.
# STEP 1 (own Bash call): persist the pre-resnapshot tip to a FILE —
# fenced blocks are separate shell invocations, so a bare variable would
# not survive to step 3 (the Guard-1 diff-file / lint-verdict pattern):
git -C "$WT" rev-parse HEAD > /tmp/issue-<N>-resnapshot-tip.txt
# STEP 2: re-run the ENTIRE Guard-1 block above VERBATIM: it re-fetches,
# captures a fresh MAIN_SHA, re-pins the foreign paths, and commits only
# if anything changed (idempotent).
# STEP 3 (own Bash call): retry ONLY if the re-snapshot changed the
# branch tip OR unpushed commits exist (same rev-list re-derivation as
# the safe-case push; missing ref counts as unpushed). The retry sits in
# the else arm so the skip arm ENDS the block — the skip must never fall
# through into the push:
TIP_BEFORE=$(cat /tmp/issue-<N>-resnapshot-tip.txt)
if [ "$(git -C "$WT" rev-parse HEAD)" = "$TIP_BEFORE" ] \
   && [ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null || echo 1)" -eq 0 ]; then
  echo "re-snapshot changed nothing (tip unchanged, nothing unpushed) — a retry would fail identically; record resnapshot_retry outcome: skipped and run the merge-conflict recovery below"
  false
else
  # Descendancy-guarded (#2312) — canonical snippet (1) form: a rewritten
  # branch must never take the refspec-less pull fallback here (it would
  # rebase onto the stale remote branch — the very #1128 shape this retry
  # recovers); a diverged remote self-heals by instruction, never blind.
  if git -C "$WT" rev-parse --quiet --verify origin/issue-<N> >/dev/null 2>&1 \
     && ! git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD \
     && ! git -C "$WT" merge-base --is-ancestor HEAD origin/issue-<N>; then
    echo "[step10d] shape-2 retry push BLOCKED — mutually non-ancestral remote (#2312): REWRITTEN or DIVERGED; discriminate per the safe-case guard comment (all-foreign ⇒ pull-retry self-heal then re-enter; else § Rewritten-branch landing route instead of the retry)."
    false
  else
  git -C "$WT" push origin issue-<N> \
    || { git -C "$WT" pull --rebase=merges --autostash \
         && git -C "$WT" push origin issue-<N>; }
  # gh recomputes mergeability ASYNCHRONOUSLY after a push (the recovery
  # block's own precedent) — re-check before the retried merge so a
  # stale mergeability read cannot burn the single retry:
  gh pr view <PR> --json mergeable -q .mergeable   # brief wait/retry until MERGEABLE
  fi
fi
# If the tip changed, the SHA-bound lint verdict is now STALE and the
# gated conditional would fail CLOSED: re-run the executable Pre-push
# workflow-lint gate block (subsection above) so the verdict re-binds to
# the NEW tip (never hand-write it, #1082). If the tip did NOT change
# (push-only fix), the still-valid verdict re-certifies it — the
# conditional's sha arm enforces this mechanically either way. Then
# re-enter the SAME gated merge conditional (the task's $MERGE_FORM) exactly
# once. Classify a SECOND refusal by its error text per the failure
# echo: a "can't be rebased" refusal takes the shape-1 --squash retry
# (the retried rebase replays the FIRST, stale strip commit per-commit
# and can surface as shape 1 even after a clean re-snapshot; the squash
# is the endpoint merge that ends the chain); any OTHER second refusal
# falls through to the merge-conflict recovery below. Record the
# outcome either way in the epm:merged / epm:merge-failed note:
#   resnapshot_retry: {tip_before: <TIP_BEFORE>, main_sha: <fresh MAIN_SHA>,
#                      stripped_again: yes|no, outcome: merged|refused|skipped}
```

(If the re-run Guard-1 created NO new commit but unpushed commits existed
— e.g. a crash between an earlier strip and its push — the push alone can
fix the server-side view and the retry is warranted; the tip is then
unchanged, so the still-valid SHA-bound verdict re-certifies it and no
gate re-run is needed.)

**Known failure shape 3 — PR head-sync lag
(`Head branch is out of date`, #1614).** Substring-match
`Head branch is out of date` (transcript-mined from #1614;
may drift — treat a `Head branch was modified` refusal as
the same class). GitHub's PR OBJECT (what `gh pr view` and the merge
API read) lags a JUST-PUSHED head ref under fleet churn — minutes
observed (#1614) — so the merge is
refused against a stale view of the head. NOT branch-behind-main
staleness: #1614's attempt 3 landed the SAME 132-behind tip
byte-unchanged once the PR object re-synced, so `gh pr update-branch` /
catching the branch up to `main` does not address this shape (and the
update-branch default form adds a merge commit that breaks the
`--rebase` form on experiment branches — shape 1). Recovery: (1)
confirm the tip is actually pushed
(`git -C "$WT" rev-list --count origin/issue-<N>..HEAD` = 0 — if not,
the safe-case push is the fix, not this shape); (2) re-poll
`gh pr view <PR> --json headRefOid` until it equals the local tip
(bounded until-loop or bg-Bash re-check — never a leading foreground
sleep; ~6 × 20 s, the pre-check budget again); (3) still stale → the
#1614 close/reopen nudge ONCE per Step 10d invocation —
`gh pr close <PR>` then `gh pr reopen <PR>` (forces GitHub to re-sync
the PR object; the branch tip and the PR's commits are untouched) —
verify the reopen landed (`gh pr view <PR> --json state -q .state` =
`OPEN`; a crash between close and reopen strands a CLOSED PR — the
next invocation re-opens it idempotently before re-entering) and
re-poll once more; (4) re-enter the SAME gated merge conditional
with the SAME `$MERGE_FORM` — the refusal changed nothing locally, so
the SHA-bound verdict still certifies the tip (consume-on-merge-success
survives this failure by design; never hand-write the verdict file,
#1082); a re-entry may legitimately MOVE the tip via a second post-gate
sync + verdict re-bind (#1807). Bounded
at ONE nudge + TWO re-entries per Step 10d invocation, counted per
invocation regardless of re-bind. STILL
stale after the nudge re-poll → optional LAST RESORT before the
Failure bullet, the #1613 empty-commit synchronize:
`git -C "$WT" commit --allow-empty -m "issue-<N>: force PR synchronize
(#1613 head-sync wedge)"` + the bare branch push — forces GitHub to
emit a synchronize event that rebuilds the PR object (#1613's ~10-min
wedge, which outlasted passive polling, was cured exactly this way).
This MUTATES the tip, so the SHA-bound verdict goes stale and the
pre-push lint gate MUST re-run before the next attempt (the gate's own
sha arm enforces this fail-closed; #1613's recovery re-ran + re-bound
the gate). Still refused → the Failure bullet (`epm:merge-failed`).
The head-sync pre-check inside the safe-case block above exists to
keep this shape off the FIRST attempt; this paragraph is the backstop
when the lag outlasts the pre-check budget.

Before each retried merge call in this shape (each re-entry
AND after the close/reopen nudge), post an `[long-phase-heartbeat]`
progress note (#1723; same family as shapes 0/2 above):

```bash
uv run python scripts/task.py post-marker <N> epm:progress \
  --note "[long-phase-heartbeat] step10d-merge attempt=<k> shape=3"
```

**Exit-0 false success — `gh pr merge` on an already-merged/closed PR
(#1768 round-2 / #1897).** Shapes 0–3 above all key on NON-zero merge
exits; this shape is different in kind: `gh pr merge` against a PR a
PRIOR round already merged/closed EXITS 0 with `! Pull request ... was
already merged` — a terminal PR object never merges new branch commits,
so the round's payload stays stranded off `main` while the flow reads
success (#1768 round-2: a merge against the round-1 PR
stranded the 22-commit round-2 payload; recovery cost a fresh PR + a
full gate re-run). Prevention is the PR-object liveness probe at the safe-case
entry (state must be OPEN, else a fresh pre-checked draft PR);
detection is the `Landing verification (#1897)` read in BOTH merge
success arms (state == MERGED AND mergedAt fresh vs the pre-attempt
value) — the verdict is consumed only on a VERIFIED landing, and an
unverified exit-0 routes to MERGE NOT VERIFIED (verdict survives; at
most one probe re-entry, then `epm:merge-failed`).

- **Success:** post `epm:merged v1` VIA THE `--file` CHANNEL — never `--note`
  — with a scratch file at `/tmp/issue-<N>-merged-note.md` (composed VIA
  THE WRITE TOOL immediately before the post-marker call — NEVER a Bash
  heredoc or printf/echo redirect: the note body then rides the Bash argv,
  and the fail-closed #1058 strip pre-pass REFUSES the common merged-note
  shape, blocking the whole call (#1756). Resolve dynamic values — SHAs,
  counts — in PRIOR Bash calls and embed them as literals in the Write
  content) carrying the SHA
  list plus `merge_form: squash|rebase`, `merge_attempts: <n>`, and
  `mb_first_parent: <yes|no>` (the Guard-3 first-parent diagnostic;
  note-token convention — no schema change, #1288). The `--file` channel bypasses the
  argv-prose scan `guard_repo_root_branch.sh` runs on `--note`; merge-recovery
  notes routinely quote `git merge`, `git rebase`, and the pre-fix guard's
  own blocked argv would fire on any of
  them. Update the chat title with `merged`. Then run the **post-merge
  stale-task-folder guard** below (it runs on every merge form).

  **Authoritative merge-SHA derivation (#1722).** Read the merge SHA
  from the PR object itself, AFTER `gh pr merge` reports success:
  `MERGE_SHA=$(gh pr view "$PR" --json mergeCommit -q .mergeCommit.oid)`
  (`$PR` = the probe-rebound PR number from the safe-case block, #1897 —
  in a fresh shell, re-bind it via the PR-state probe, never by pasting
  a prior round's PR number — the #1768 round-2 mechanism).
  This is the shape SKILL.md already uses elsewhere for other PR fields
  (`state`, `mergeable`, `headRefOid`), and `mergeCommit` is a documented
  `gh pr view --json` field — it resolves the merge commit for BOTH
  merge forms (`--squash` returns the single squash commit; `--rebase`
  returns the tip of the replayed commits; verified live, #1722).
  A NOT-YET-MERGED PR returns `null` for `.mergeCommit`, so the derivation
  is ordering-safe as long as it runs AFTER `gh pr merge` reports success.
  NEVER derive the SHA from the shared `origin/main` tip
  (e.g. `git log -1 --format=%H origin/main`) — concurrent sessions'
  merges advance the shared tip between the merge and the read, so a
  sibling task's merge commit can substitute for yours (the tip once
  read #1692's SHA while
  posting the #1691 merge marker).

  **Pre-post commit-subject cross-check (#1722; object-availability
  hardened #1763).** Before posting `epm:merged v1`, verify the derived
  commit's subject names THIS task. The merge commit was created
  SERVER-SIDE by `gh pr merge` and exists locally only after a fetch —
  the #1725 pre-marker sync is fail-soft AND single-flight (exit 0 can
  mean "another sync in flight, no pull ran"), and the
  merge-conflict-recovery path has no pre-marker sync at all — so
  ensure the object is local FIRST; a MISSING object is
  staleness/transport, never a MISMATCH (#1735):

      git -C "$REPO_ROOT" rev-parse --verify --quiet "$MERGE_SHA^{commit}" >/dev/null \
        || timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
      SUBJECT=$(git -C "$REPO_ROOT" log -1 --format='%s' "$MERGE_SHA" 2>/dev/null) \
        || SUBJECT=$(gh api "repos/{owner}/{repo}/commits/$MERGE_SHA" --jq '.commit.message' | head -1)

  The `gh api` fallback reads the subject from the REMOTE commit (no
  local object needed; the `{owner}/{repo}` placeholders resolve from
  the current repo), so a failed/raced fetch degrades to a remote read
  instead of a false MISMATCH. Then confirm `task #<N>` (or the
  issue-branch name `issue-<N>`) appears in `$SUBJECT`. Only a
  RESOLVED-but-foreign subject is a MISMATCH: ABORT the post and
  re-derive from `gh pr view "$PR" --json mergeCommit`. The null case
  from a not-yet-merged PR still fails loud (`git log -1 --format=%s
  null` errors locally AND the remote read 404s → empty `$SUBJECT`);
  an EMPTY `$SUBJECT` after both reads is an ABORT (cannot certify),
  never a silent post. A foreign SHA is caught at post time rather
  than by eye after the fact.

  Note. A merged diff that touches `scripts/*guard*.sh`, `.claude/hooks/*`, or
  any workflow-surface content that the session's own remaining Bash calls
  route through is NOT live at the shared repo root the instant `gh pr merge`
  returns success. `origin/main` carries it; the shared root's working tree
  does not. The pre-marker `sync_repo_root.py` above closes the window on the
  `epm:merged` call itself (the fix is live at the root before its own note
  is scanned by the argv guards). Downstream root-side calls in the same
  session — for example, a Step-9 or Step-10 chat-line log, a post-completion
  `epm:progress`, a follow-up-proposer dispatch — still see the pre-fix
  copy until the post-merge-guard pre-sync (or a fresh `/issue <N>`
  re-invocation's Step 4 root-divergence probe from #1725) runs.
- **Failure** (rebase conflict, non-mergeable PR, non-fast-forward): for
  the `Base branch was modified` shape (substring match), run the
  shape-0 wait-and-retry (Known failure shape 0 above, max 2) FIRST; for
  the `Head branch is out of date` shape (substring match), run the
  shape-3 head-sync re-poll + close/reopen nudge (Known failure shape 3
  above, nudge ONCE); for
  the `Pull Request has merge conflicts` shape (substring match), FIRST
  run the **re-snapshot-and-retry** (Known failure shape 2 above) ONCE;
  if it is skipped (nothing changed), run the **merge-conflict recovery**
  sub-procedure below ONCE; if the retried merge is refused AGAIN,
  re-classify that second refusal by error text — a `can't be rebased`
  refusal takes the shape-1 `--squash` retry, anything else runs the
  **merge-conflict recovery** ONCE. For any other first refusal, run the
  **merge-conflict recovery** sub-procedure below ONCE directly.
  If the recovery itself fails or the retried merge is still refused:
  do NOT swallow it (fail-fast). Post `epm:merge-failed v1` with the
  `gh` / `git` error, surface ONE line in chat naming the branch +
  worktree path for manual resolution, and CONTINUE — an experiment
  still parks at `awaiting_promotion`; a code-change task still
  completes. The merge is retried (idempotently) on the next
  `/issue <N>` re-invocation.
- **Autonomous mode** (no user present): same as above — the auto-merge
  proceeds. No deferral. (This reverses the prior "default NO" autonomous
  behavior; merge to `main` is no longer user-gated.)

#### Merge-conflict recovery (safe case: `gh pr merge` refuses)

When the safe-case merge is refused on mergeability (a REAL conflict —
`main` and the branch both changed the same lines), do NOT hand-resolve
in the shared repo root and do NOT force-push. Recover IN THE WORKTREE
(worked example: #598 — both sides appended to
`.claude/agents/experimenter.md`; resolved in the worktree, merged on
retry):

```bash
git -C "$WT" fetch origin main --quiet
# Capture the snapshot ONCE, immediately after the fetch, and merge THAT
# SHA — origin/main is a shared ref a concurrent session's fetch can
# advance between these commands (#1128's shared-ref race).
MAIN_SHA=$(git -C "$WT" rev-parse origin/main)
git -C "$WT" merge "$MAIN_SHA"          # conflicts surface HERE, in the worktree
# Run that merge BARE — never piped through tail/grep (hook-blocked, #1048;
# 9 sessions re-tripped the hook mid-recovery on 07-09/07-10). To capture
# output, file-redirect it: `git -C "$WT" merge "$MAIN_SHA" > /tmp/issue-<N>-merge.log 2>&1; MERGE_RC=$?`.
# Foreign tasks/ conflicts are resolved MECHANICALLY: take the captured
# snapshot's version wholesale (under fleet marker churn main is
# authoritative for OTHER tasks' state — the #1128-proven recovery:
# foreign tasks/ pinned to ONE captured main SHA). Materialize the
# conflicted-path list and check its own exit code in Guard 1's `if ! ...`
# exclusive-arm shape (#1184): a FAILED producer takes the terminal
# echo + false arm and the work arm is STRUCTURALLY unreachable — the
# old `|| { echo; false; }` form reported failure but let the next
# command run under no-set-e / piecewise execution (#1243).
if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=U -- 'tasks/' \
    > /tmp/issue-<N>-recovery-foreign.txt; then
  echo "recovery: conflicted-paths diff FAILED — resolve by hand per the prose below"
  false
# Work arm: two-command elif list — mapfile fills RECOVERY_FOREIGN from the
# FILE (the carve-out grep's no-match `|| true` is a legitimate empty
# list), then the [ ... ] test (the LAST command's exit) decides the
# branch. The mapfile + non-empty-array idiom is Guard 1's own hook-proven
# shape (the guard_repo_root_branch.sh -C waiver expects -C right after
# git; no xargs indirection, no whitespace-splitting caveat); the length
# test means an empty list never runs a pathspec-less checkout. On an
# empty list no branch is taken and the unit exits 0 — deliberate
# post-merge-guard parity, not drift (the old `[ ... ] && checkout`
# tail exited 1 there).
# Discriminate on-main vs gone-on-main (Guard 1's own cat-file split): task
# folders MOVE on every status change, so a foreign conflicted path absent
# at $MAIN_SHA is ROUTINE, not rare (#1242 13:37Z / #1246 14:43Z re-derived
# this by hand). checkout <sha> -- <path> resolves each ON-MAIN U path to
# the snapshot's version and stages it; a GONE-ON-MAIN path (moved/deleted
# on main) is resolved as a REMOVAL — main is authoritative for foreign
# tasks/ state, and git rm -f also resolves the unmerged index entries.
elif mapfile -t RECOVERY_FOREIGN < <(grep -Ev "^tasks/[^/]+/<N>/" \
      /tmp/issue-<N>-recovery-foreign.txt || true); [ "${#RECOVERY_FOREIGN[@]}" -gt 0 ]; then
  RECOVERY_ON_MAIN=()        # exist at MAIN_SHA -> take the snapshot's version
  RECOVERY_GONE_ON_MAIN=()   # absent at MAIN_SHA (moved/deleted on main) -> remove
  for p in "${RECOVERY_FOREIGN[@]}"; do
    if git -C "$WT" cat-file -e "$MAIN_SHA:$p" 2>/dev/null; then
      RECOVERY_ON_MAIN+=("$p")
    else
      RECOVERY_GONE_ON_MAIN+=("$p")
    fi
  done
  # if-form, not `[ ] && cmd` tails: an empty second list must not exit the
  # unit 1 (the documented exit-0 empty-list parity above).
  if [ "${#RECOVERY_ON_MAIN[@]}" -gt 0 ]; then
    git -C "$WT" checkout "$MAIN_SHA" -- "${RECOVERY_ON_MAIN[@]}"
  fi
  # git rm -f, NOT --cached: this resolution commit is `git commit --no-edit`
  # with NO pathspec (index governs), so --cached would technically survive
  # (#1244's resurrection needs a pathspec-limited commit) — -f is chosen for
  # Guard-1 parity and to leave no stale working-tree litter behind.
  if [ "${#RECOVERY_GONE_ON_MAIN[@]}" -gt 0 ]; then
    git -C "$WT" rm -f --ignore-unmatch -- "${RECOVERY_GONE_ON_MAIN[@]}"
  fi
fi
# Binary figures/ conflicts (add/add or modify/modify — #1090 fu4 / PR
# #1066; earlier #697/#597): git cannot content-merge binaries, and the
# .gitattributes merge=union rules cover tasks/ jsonl + agent-memory md,
# NOT figures/ — so both-sides-changed figure paths ALWAYS conflict.
# Figures are REGENERABLE artifacts (sidecar meta.json pins provenance;
# the analyzer re-renders + SHA-pins): resolve MECHANICALLY, the NEWER
# regeneration wins — compare the last commit touching the path on each
# side; tie -> theirs (in THIS merge ours = the issue branch, theirs =
# the captured $MAIN_SHA snapshot — the #1090-proven side). The losing
# copy stays recoverable (branch kept post-merge; main history is
# immutable; the figure re-renders from committed eval JSON). Stem-mates
# (png/pdf/meta.json) commit together per regeneration, so per-path %ct
# resolves the group to one side. checkout --ours/--theirs writes the
# working tree only — the git add resolves the index entry, and the add
# is GATED on checkout success: a failing checkout (modify/delete:
# missing stage) leaves the entry UNMERGED, so the later
# `git commit --no-edit` refuses on unmerged paths — the loud
# fall-through to the manual prose below. NEVER stage a path whose
# checkout failed.
if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=U -- 'figures/' \
    > /tmp/issue-<N>-recovery-figures.txt; then
  echo "recovery: figures/ conflicted-paths diff FAILED — resolve by hand per the prose below"
  false
else
  while IFS= read -r p; do
    OURS_CT=$(git -C "$WT" log -1 --format=%ct HEAD -- "$p")
    THEIRS_CT=$(git -C "$WT" log -1 --format=%ct "$MAIN_SHA" -- "$p")
    if [ "${THEIRS_CT:-0}" -ge "${OURS_CT:-0}" ]; then SIDE=--theirs; else SIDE=--ours; fi
    if git -C "$WT" checkout "$SIDE" -- "$p"; then
      git -C "$WT" add -- "$p"
    else
      echo "recovery: figures/ checkout $SIDE FAILED for $p (modify/delete missing stage?) — left UNMERGED; resolve by hand per the prose below"
    fi
  done < /tmp/issue-<N>-recovery-figures.txt
fi
# Residual conflicts — THIS task's own tasks/*/<N>/ paths and all remaining
# non-tasks/ paths (foreign tasks/ and figures/ were resolved MECHANICALLY
# above, with zero conflict-body reads). The orchestrator NEVER reads
# residual conflict bodies inline here — that inline read killed #1338
# ("Prompt is too long", no recovery turn): Step 10d/9b merges run
# late-session by construction, and a session cannot introspect its own
# context headroom. Materialize the residual list (exclusive-arm `if ! ...`
# producer shape, #1184/#1243):
if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=U \
    > /tmp/issue-<N>-recovery-residual.txt; then
  echo "recovery: residual conflicted-paths diff FAILED — do NOT resolve blind; epm:merge-failed"
  false
elif [ -s /tmp/issue-<N>-recovery-residual.txt ]; then
  echo "recovery: $(wc -l < /tmp/issue-<N>-recovery-residual.txt) residual content conflict(s) — dispatch the residual-conflict subagent (subsection below); do NOT read conflict bodies inline"
  # Halt the inline fence at this branch (loud false — a naive one-shot
  # execution must not fall through to the commit/certification below);
  # re-enter at the post-resolution certification block once the
  # subagent's resolution commit lands.
  false
else
  git -C "$WT" commit --no-edit   # every conflict was resolved mechanically above
fi
# Post-resolution certification (the #1128 verification): the branch tree
# must now be IDENTICAL to the captured snapshot over tasks/, modulo this
# task's own folder. ONE fused if/elif chain (Guard 1's `if ! ...` shape,
# #1184/#1243): the verification diff and the residual-foreign check are
# one logical certification — under the old `|| { echo; false; }` form a
# FAILED diff left the verify file EMPTY (the redirect truncates before
# the command runs), the residual grep then found nothing, and
# certification passed VACUOUSLY (fail-OPEN into the push). Here a
# failed producer takes the terminal arm and the residual check is
# structurally unreachable:
# Two-endpoint ("$MAIN_SHA" HEAD) DELIBERATELY, not Guard 1's three-dot
# (#1280): this certifies TREE IDENTITY against the captured snapshot AFTER
# the merge brought MAIN_SHA's content in — both endpoints fixed, so
# main-side advancement cannot false-positive here, and the form stays
# correct even when the merge produced no commit. Guard 1's PRE-merge
# trigger is the site where two-endpoint misread main advancement as
# foreign touches.
if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA" HEAD -- 'tasks/' \
    > /tmp/issue-<N>-recovery-tasks-verify.txt; then
  echo "recovery: tasks/ verification diff FAILED — do NOT push"
  false
elif grep -Ev "^tasks/[^/]+/<N>/" /tmp/issue-<N>-recovery-tasks-verify.txt | grep -q .; then
  echo "recovery: foreign tasks/ still differ from the captured main snapshot — do NOT push; re-pin the listed paths (checkout the on-main, git rm -f the gone-on-main) to \$MAIN_SHA and re-verify"
  false
fi
# Re-run the targeted tests for the touched surface AND the executable
# Pre-push workflow-lint gate block (subsection above; gated = the gate
# tree rebuilt from this post-merge tip (origin/main + the post-merge
# own-diff — content-identical to this post-merge worktree, which carries
# main's CURRENT lint — the ideal gate point); the gate re-run SHA-binds
# the verdict to THIS post-merge tip. Re-run it as ONE BACKGROUND Bash
# call with the fresh foreground completion-read — gate subsection —
# before this gated push). The push is then GATED on the persisted, SHA-BOUND verdict file —
# the explicit conditional is the hard stop (missing file / block / crash
# / missing or stale sha all fail CLOSED). The verdict is consumed only
# AFTER `gh pr merge` SUCCEEDS (never hand-write the verdict file,
# #1082). The recovery just added a merge commit, so --rebase is
# documented-doomed here (#1041 — the old flow burned that attempt, then
# took the --squash substitution). Go straight to --squash for ALL kinds
# (#1288).
#
# Before the post-recovery `gh pr merge --squash` retry, post an
# `[long-phase-heartbeat]` progress note (#1723; same family as shapes
# 0/2/3 above):
#   uv run python scripts/task.py post-marker <N> epm:progress \
#     --note "[long-phase-heartbeat] step10d-merge attempt=<k> shape=conflict-recovery"
# Read this conditional VERBATIM (#2006). Do NOT re-compose it — an
# improvised `grep -qxE … <(sed …)` process-substitution form inside an
# eval'd guard prelude exited 1 "BLOCKED: verdict missing/stale" while an
# immediate re-probe showed verdict `pass` + sha == tip, costing ~3 h and
# three full gate runs. The committed three-conjunct form below is the
# tested one; a re-compose is a fresh, untested predicate on the merge path.
# Note the grep scans the WHOLE file, not line 1 — which is why the
# never-hand-write rule (above) is load-bearing here: a hand-appended
# `pass` on line 3 satisfies it while line 2 still carries a valid sha.
if grep -qxE 'pass|skip-artifact-only' /tmp/issue-<N>-lint-verdict.txt 2>/dev/null \
   && [ -n "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" ] \
   && [ "$(sed -n 2p /tmp/issue-<N>-lint-verdict.txt 2>/dev/null)" = "$(git -C "$WT" rev-parse HEAD)" ]; then
  git -C "$WT" push
  # gh recomputes mergeability asynchronously after a push — it can be
  # momentarily stale. Re-check before concluding failure; ALSO bind the
  # pre-attempt mergedAt for the landing verification below (fenced
  # blocks are separate shells — the safe-case probe's binding is not in
  # scope here, #1897):
  PRE_STATE=$(gh pr view <PR> --json mergeable,state,mergedAt \
    -q '[.mergeable, .state, (.mergedAt // "null")] | join(" ")' 2>/dev/null) || PRE_STATE=""
  PRE_MERGED_AT=${PRE_STATE##* }
  echo "$PRE_STATE"   # brief wait/retry until mergeable=MERGEABLE
  if gh pr merge <PR> --squash --delete-branch=false; then
    # Landing verification (#1897): same contract as the safe-case arm —
    # exit 0 is NOT proof THIS attempt landed (`gh pr merge` on an
    # already-merged PR exits 0, #1768 round-2); verify via the PR
    # object, never branch-sha ancestry; empty PRE_MERGED_AT fails CLOSED.
    LANDED_OK=no
    for _ in 1 2 3; do
      POST=$(gh pr view <PR> --json state,mergedAt \
        -q '[.state, (.mergedAt // "null")] | join(" ")' 2>/dev/null) || POST=""
      if [ -n "$PRE_MERGED_AT" ] && [ "${POST%% *}" = "MERGED" ] \
         && [ "${POST##* }" != "null" ] \
         && [ "${POST##* }" != "$PRE_MERGED_AT" ]; then LANDED_OK=yes; break; fi
      sleep 10
    done
    if [ "$LANDED_OK" = yes ]; then
      rm -f /tmp/issue-<N>-lint-verdict.txt   # consume on VERIFIED merge success only — the verdict certified exactly the tip that landed
    else
      echo "MERGE NOT VERIFIED — gh pr merge exited 0 but the PR object shows no FRESH merge (the exit-0 'was already merged' false-success shape, #1768/#1897). Verdict NOT consumed; re-enter via the safe-case PR-state probe (fresh PR) AT MOST ONCE per Step 10d invocation — a SECOND unverified exit-0 success -> epm:merge-failed. Do NOT report success."
      false
    fi
  else
    echo "MERGE FAILED post-push — classify: (0) \"Base branch was modified\" -> shape-0 same-tip retry (verdict survives); anything else -> epm:merge-failed (do NOT hand-write the verdict file)."
    false
  fi
else
  echo "BLOCKED: pre-push workflow-lint gate (verdict: $(cat /tmp/issue-<N>-lint-verdict.txt 2>/dev/null || echo not-run)) — missing verdict, block/crash, or missing/stale sha (hand-written verdict, or new commits since certification) all fail CLOSED: fix the named offender (or crash cause), re-sync (§ pre-gate re-sync), then re-run the gate ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Do NOT push."
  rm -f /tmp/issue-<N>-lint-verdict.txt   # block/crash/stale consumed — a fresh gate run regenerates it
  false
fi
```

##### Residual-conflict subagent dispatch (context-hygiene branch)

When the residual list is NON-EMPTY, dispatch the conflict investigation +
resolution to a fresh worktree-scoped subagent — never an inline
orchestrator read. UNCONDITIONAL: no file-count or context-fullness
threshold. Step 10d/9b merges run after the full pipeline, so late-session
is guaranteed; a session cannot introspect its own headroom; and the
lethal variable is conflict-body BYTES, not file count (#1338). The mechanical passes above absorb the
common classes, so this branch fires rarely. The failure-arm echoes above
("resolve by hand per the prose below") route HERE — "by hand" means via
this dispatch; do NOT read conflict bodies inline.

1. Post the stage-dispatch breadcrumb FIRST (Step 9 entry-guard
   convention): an `epm:progress` marker whose note BEGINS with the
   literal `stage-dispatch ` prefix (required by
   `task_workflow.stage_dispatch_should_skip` / `_breadcrumb_fields` —
   a prefix-less note is invisible to the dedup + resume machinery):
   `stage-dispatch stage=step10d-conflict-resolve worktree=<abs $WT> paths=<count>`.
   Resume/dedup predicate for a successor session: breadcrumb present AND
   `git -C "$WT" diff --name-only --diff-filter=U` empty AND a resolution
   commit on the branch tip ⇒ resolution landed, skip to certification;
   breadcrumb present but worktree still conflicted AND no prior subagent
   verdict recorded ⇒ re-dispatch ONCE, counted as the SAME single
   attempt; otherwise fall to the Failure bullet. Never two concurrent
   dispatches.
2. Spawn ONE fresh `implementer`-class subagent with
   `env=scrub_subagent_env(os.environ)` (standing convention). If the
   residual list file is missing at dispatch time, re-run the
   `--diff-filter=U` producer above. The brief is LEAN — paths and pins by
   reference, never conflict bodies
   (`.claude/rules/trigger-dense-review.md` disciplines 1/3/4 — findings
   by reference, windowed reads, minimal return text;
   `.claude/rules/diff-size-budget.md`):
   - task id, branch name, absolute worktree path `$WT`;
   - the captured `$MAIN_SHA` — the subagent PINS every resolution to it
     and never re-fetches or re-snapshots (#1128 shared-ref race);
   - the residual list file `/tmp/issue-<N>-recovery-residual.txt` + count
     (the subagent reads paths from the file);
   - the resolution contract, verbatim: (a) a residual FOREIGN tasks/ path
     is pinned to `$MAIN_SHA` — checkout the on-main, `git rm -f` the
     gone-on-main (the mechanical pass's own split); (b) a residual binary
     figures/ path resolves newer-regeneration-wins per the recipe above;
     (c) THIS task's own tasks/*/<N>/ and non-tasks/ paths: keep main's
     version of anything outside this task's deliverables; for the task's
     own deliverables keep the branch's content, merging hunk-by-hunk only
     where both sides carry real content;
   - read discipline: size any diff body before reading (300 KB budget);
     read conflicted files individually, windowed around conflict markers;
   - completion duties: `git -C "$WT" add` each resolved path,
     `git -C "$WT" commit --no-edit`, verify zero `--diff-filter=U` paths;
   - return contract: verdict `resolved` | `unresolvable: <one line>`, the
     resolution commit sha, per-class path counts, path NAMES only — NEVER
     conflict hunks, bodies, or diff text in the return (an oversized
     return kills the parent this dispatch protects).
3. On `resolved`: verify cheaply (`--diff-filter=U` empty; `rev-parse
   HEAD` matches the reported sha), spot-check the keep-main contract on a
   sample — a residual path OUTSIDE this task's deliverables should be
   byte-identical to the snapshot (`git -C "$WT" diff "$MAIN_SHA" HEAD --
   <path>` empty) — then re-enter the fence above AT the post-resolution
   certification block and run certification → lint gate → push → merge
   YOURSELF. The subagent never pushes and never runs the lint gate — the
   fail-closed verdict-file contract is unchanged.
4. On `unresolvable`, a dead/refused subagent (after step 1's single
   no-verdict-recorded re-dispatch, or immediately when a verdict WAS
   recorded), or certification FAIL on the subagent's commit: fall to the
   Failure bullet (`epm:merge-failed v1`, continue). The dispatch lives
   INSIDE the one-recovery-attempt cap — never a second dispatch (the
   step-1 no-verdict re-dispatch is the SAME attempt; a second death
   falls here), never an inline fallback read.

One recovery attempt per Step 10d invocation. If the re-checked
mergeability never recovers or the retried merge is refused again, fall
to the Failure bullet above (`epm:merge-failed v1`, continue). When the
recovered merge DOES land, run the **post-merge stale-task-folder guard**
below — the recovery's `git merge origin/main` adds a merge commit that
can re-import this task's old-status folder, exactly the case the guard
catches.

#### Rewritten-branch landing route (#2312 — force-free; origin/issue-<N> not an ancestor of HEAD)

Entry points: the zero-PR prelude's stale-ref arm, the safe-case /
re-sync / shape-2 push guards, and the PR-head parity gate — any of them
firing means `origin/issue-<N>` EXISTS and is MUTUALLY non-ancestral with
HEAD (a strictly-ahead remote never reaches here — it stays on the
ordinary push arm). That is a TWO-state signal: (a) local history
REWRITTEN (typically the prescribed mid-flight rebase onto `origin/main`
after a sibling landing — recurring: #1999, #2171, #2296), or (b) the
remote genuinely DIVERGED (novel FOREIGN commits — the documented
result-push channel: pod-side-reporting.md § Result-push verification
contract, #1205/#1880). This route is the landing for state (a) — for
state (b) the guards' discrimination step names the pull-retry self-heal
instead (the VM-side twin of #1880's lane-side fetch+rebase recipe), and
precondition 2 below re-checks the same judgment at landing time. In state (a) a branch push is rejected
non-fast-forward; the pull-rebase retry would replay hundreds of main
commits onto the stale remote tip (the #1128 shape); a PR merge off the
stale head would land content no gate certified (#2296's stale head called
`create_scratch_worktree(..., sha=...)` — a `TypeError` against post-#2293
main). `git push --force-with-lease` would fix the ref in one command, but
force-push is a standing user-ask (`.claude/rules/auto-continuation.md`
STATE-TO-`blocked` criterion 2) with NO autonomous carve-out, and recorded
practice has DIVERGED on exactly this question (#2171 and #1999 recorded it
as the correct form; #2181 recorded it as a policy violation). That policy
question — surfaced by #2312 — is task #2313 (`proposed`; decided by the
user, never here). Until it resolves, this force-free route is the ONLY
sanctioned landing for state (a) — and it needs no user, so never block a
gate-PASSed task on it.

PRECONDITIONS (all four, before any landing command):

0. Guards 0-3 have run this invocation (same bar as the safe-case merge;
   a Guard-3 "unsafe" verdict routes to the artifact-confirmed merge below
   INSTEAD, unchanged).
1. The pre-push workflow-lint gate has PASSed and the verdict file's line 2
   equals `git -C "$WT" rev-parse HEAD` — the route lands EXACTLY the
   certified tip (fail-closed conventions unchanged: never hand-write the
   verdict, #1082). MID-RUN (no verdict exists): this route does not apply —
   leave the branch unpushed, surface, land at Step 10d.
2. Remote-only payload check — enumerate what the stale remote holds that
   HEAD does not: `git -C "$WT" log --oneline HEAD..origin/issue-<N>` and
   `git -C "$WT" cherry HEAD origin/issue-<N>` (`-` = patch-equivalent,
   already contained in HEAD). The EXPECTED shape for state (a) is the
   superseded pre-rebase copy of this branch's own payload (a `+` is
   legitimate — a conflict-reconciling rebase changes the patch, #2296).
   ALL remote-only commits novel FOREIGN work ⇒ this is state (b): abandon
   the route, run the pull-retry self-heal, re-enter the safe-case flow.
   Mixed ⇒ cherry-pick the foreign commits onto HEAD first (this moves the
   tip — the sha-bound verdict goes stale and the lint gate MUST re-run),
   then proceed. NEVER discard novel remote-only content silently.
3. Post the heartbeat: `[long-phase-heartbeat] step10d-merge attempt=<k>
   shape=rewritten` (same family as shapes 0/2/3; #1723).

THE LANDING (the CLAUDE.md § Concurrent repo-root committers / #1489
scratch-worktree form; in-file precedent: the post-merge stale-task-folder
guard's `push HEAD:main`). Exclusive-arm shape (#1184) — a failed stage
takes the echo+false arm and the next stage is structurally unreachable:

```bash
# Own Bash call. LOCAL_TIP is the gate-certified tip; every command is
# git -C-scoped (repo-root guards stay out of the way; never a repo-root
# checkout/merge). NO global `worktree prune` here — a GLOBAL prune
# unregisters ANY worktree whose directory is momentarily missing (every
# live issue worktree during a data-disk mount hiccup; disk-hygiene.md
# bans the janitor from exactly this shape) — the add's -f overrides a
# stale registered-at-this-path record instead. Crash-mid-add residue:
# `worktree add` holds an "initializing" lock while creating, so a crash
# there leaves a locked-and-missing registration that a single -f refuses
# (it needs -f twice) and `worktree remove --force` also refuses — manual
# recovery: `git worktree unlock "$SCRATCH"`, or `worktree add --detach
# -f -f "$SCRATCH" origin/main`.
LOCAL_TIP=$(git -C "$WT" rev-parse HEAD)
SCRATCH=/tmp/issue-<N>-rewritten-landing
git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null || true
rm -rf "$SCRATCH"
if ! { timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet \
       && git -C "$REPO_ROOT" worktree add --detach -f "$SCRATCH" origin/main \
       && git -C "$SCRATCH" merge --no-edit "$LOCAL_TIP"; }; then
  git -C "$SCRATCH" merge --abort 2>/dev/null
  echo "[step10d/rewritten] scratch merge FAILED (a REAL content conflict vs main) — resolve IN THE SCRATCH per the merge-conflict recovery conventions (foreign tasks/ pinned to the fetched origin/main snapshot; own deliverables keep the branch content), commit, and continue at the landing push; or epm:merge-failed."
  false
# Land: push HEAD:main; on rejection (origin advanced under fleet churn) ONE
# bounded fetch + RE-MERGE + push retry INSIDE the scratch. Merge, not
# rebase: rebasing the scratch's merge commit flattens it and re-manufactures
# the replay shape this route exists to avoid.
elif ! { git -C "$SCRATCH" push origin HEAD:main \
         || { git -C "$SCRATCH" fetch origin main --quiet \
              && git -C "$SCRATCH" merge --no-edit origin/main \
              && git -C "$SCRATCH" push origin HEAD:main; }; }; then
  echo "[step10d/rewritten] landing push did NOT land after 1 retry — epm:merge-failed; scratch kept at $SCRATCH for manual completion."
  false
fi
```

LANDING VERIFICATION + bookkeeping (verify-then-consume, the #1897 posture
with ancestry instead of PR state — a fast-forwarded OR merge-committed
landing both satisfy it):

```bash
if git -C "$REPO_ROOT" fetch origin main --quiet \
   && git -C "$REPO_ROOT" merge-base --is-ancestor "$LOCAL_TIP" origin/main; then
  rm -f /tmp/issue-<N>-lint-verdict.txt   # consume on VERIFIED landing only
  git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null || true
  echo "[step10d/rewritten] landed: certified tip $LOCAL_TIP is an ancestor of origin/main"
else
  echo "[step10d/rewritten] LANDING NOT VERIFIED — verdict NOT consumed; re-probe (fetch + ancestry) before any retry."
  false
fi
```

Then: (a) if a PR exists, CLOSE it — `gh pr close "$PR" --comment "Landed
force-free via the Step 10d rewritten-branch route (#2312): certified tip
$LOCAL_TIP is on origin/main. The PR head ref holds superseded pre-rebase
history."` — NEVER `gh pr merge` it, and leave `origin/issue-<N>` untouched
(stale and inert; never force-updated); (b) post `epm:merged` per the
Success bullet above (compose via the Write tool, `--file` channel),
adding one line `route: rewritten-branch-landing (#2312)`; run the
pre-marker `sync_repo_root.py` exactly as the safe-case success arm does;
(c) run the **post-merge stale-task-folder guard** below — a merge commit
can re-import this task's old-status folder, the same reason the
merge-conflict recovery routes there. Revert-grain note: the route lands the
branch's own commits (plus at most a merge commit) rather than the
`kind: infra|batch` `--squash` single commit — an accepted deviation for
this rare arm; the branch's own commits remain individually revertible.

#### The artifact-confirmed merge procedure (unsafe case: guard 3 tripped)

When Guard 3 says the branch is unsafe to blind-rebase, the goal shifts
from "merge the whole branch" to "make sure this task's deliverables are
on `main`" — i.e. confirm that the artifacts a downstream
experiment / promotion would need (the clean-result body, the figures,
the per-cell eval JSON) already resolve on `origin/main`, then post
`epm:merged v1` with an artifact-confirmed sentinel rather than a list
of newly-landed SHAs.

This works because, by the time Step 10d fires, the analyzer has
already committed the clean-result body to `main` via `task.py
set-body` (which always operates on the repo root on `main`, never on
the worktree), and figure / `eval_results/issue_<N>/` commits land on
`main` through the same mechanism. The branch's commits often duplicate
work already on `main`; the value of the rebase is shared-infra fixes
the branch carries forward, NOT the per-task artifacts.

**New-shared-`src/`-infra guard (run FIRST, before the deliverables
check).** The artifact-confirmed path — and the surgical additive
checkout it degrades to — is structurally restricted to this task's own
`tasks/` / `figures/` / `eval_results/` paths and CANNOT carry shared
`src/` infra the branch introduced. So if this branch ADDED new shared
modules under `src/explore_persona_space/`, the artifact-confirmed path
would silently strand them on the branch — a downstream child that
reuses the harness then breaks its import path on a clean `main`
checkout (#595). Scan for it FIRST:

```bash
# Files this branch ADDED (status A) vs origin/main under shared src/.
git -C "$WT" diff --name-only --diff-filter=A origin/main HEAD -- \
  "src/explore_persona_space/" > /tmp/issue-<N>-new-src.txt
```

If `/tmp/issue-<N>-new-src.txt` is NON-EMPTY, this branch introduces NEW
shared `src/` infra: the artifact-confirmed degrade is REFUSED. Do NOT
fall through to the surgical additive checkout (it would strand the
infra). Instead either (a) resolve the actual guard-3 condition so the
SAFE full-rebase path runs (e.g. the parent `issue-<M>` branch this one
forked off has since merged — re-run the guard-3 check; once
the content check is clean, `gh pr merge --rebase`
carries the `src/` infra correctly), or (b) if the full rebase still
cannot run, post `epm:merge-failed v1` with `{reason: "new shared src/
infra cannot land via artifact-confirmed surgical checkout", new_src:
[...]}`, surface ONE line in chat naming the branch + worktree path +
the stranded `src/` paths for manual full-rebase resolution, and
CONTINUE (the task still parks / completes; the merge retries
idempotently on the next `/issue <N>`). NEVER surgical-checkout a branch
that added shared `src/` — that is the exact #595 stranding this guard
prevents.

**Stranded-MODIFIED surfacing (#1144) — run NEXT, before the decision tree
below, so it covers BOTH artifact-confirmed sub-branches** (the fast-path
route cannot carry stranded shared files by construction — predicate (e)
is ADDED-only over an in-scope pathspec that excludes shared `src/` /
`scripts/`):

```bash
# MODIFIED / RENAMED / DELETED (not ADDED) shared infra this branch carries —
# the A-only surgical checkout structurally cannot land it (#1144). Not a
# refusal: land the artifacts, but never let the stranding be silent.
#
# PRODUCER GUARD — materialize-then-check, mirroring the surgical
# additive-list producer guard below: unchecked, a FAILED diff (bad ref, no
# merge-base) writes an empty file indistinguishable from "nothing stranded",
# and this scan silently suppresses the very surfacing it exists to add.
if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=MRD origin/main...HEAD -- \
    "src/explore_persona_space/" "scripts/" > /tmp/issue-<N>-stranded-modified.txt; then
  echo "STRANDED-SCAN FAILED: cannot enumerate branch-side shared-infra modifications"
  STRANDED_STATUS=unknown
else
  STRANDED_STATUS=ok
fi
```

`STRANDED_STATUS=unknown` ⇒ record `stranded_modified: UNKNOWN — producer
diff failed` on the `epm:merged` note AND surface ONE chat line; never a
silent continue. `ok` + non-empty list ⇒ `stranded_modified: [...]` on the
note + ONE chat line naming the branch, the worktree path, and the stranded
paths — reporting only: the merge still lands the artifacts, the round is
not blocked, no new refusal route. `ok` + empty ⇒ nothing to record (an
empty list is the legitimate common case here, unlike the
deliverables-missing surgical landing, so this scan deliberately has NO
empty-list abort arm). `MRD` rather than `M` alone: a renamed or deleted
shared file is equally uncarryable by the A-only checkout. The three-dot
form is REQUIRED — branch-side modifications only; files `main` advanced
but the branch never touched must not enter the list.

```bash
# Verify task deliverables resolve on origin/main.
git -C "$REPO_ROOT" fetch origin main --quiet

# 1) body.md present on main with this task's number
BODY_REL=$(realpath --relative-to="$REPO_ROOT" \
  "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")/body.md
git -C "$REPO_ROOT" cat-file -e "origin/main:$BODY_REL" \
  || ARTIFACTS_OK=no

# 2) figures/issue_<N>/ has at least one file on main (if any were produced)
git -C "$REPO_ROOT" ls-tree -r --name-only origin/main -- "figures/issue_<N>/" \
  | grep -q . || FIGURES_OK=no   # only enforce if the task plan produced figures

# 3) eval_results/issue_<N>/ (or equivalent) similarly, when the task produced eval JSONs
```

Decision tree:

- **All required deliverables resolve on `origin/main`** -> BEFORE the
  `epm:merged v1` post, run the pre-marker root sync (#1725,
  artifact-confirmed): the deliverables verification above ran
  `git fetch origin main` at L11869, so `origin/main` is fresh, but the
  shared root's local `main` is not; a sibling session's just-merged
  workflow-surface fix is not yet live at the root when the epm:merged
  post's argv guard scans this session's note.

  ```bash
  # Root-sync before epm:merged (#1725, artifact-confirmed path): no
  # gh pr merge fires here (skipped below), so a sibling session's
  # workflow-surface fix landed on origin/main in the meantime is still
  # not live at the shared root. sync_repo_root.py is single-flight
  # flock-serialized; fail-soft (the post-merge-guard pre-sync remains
  # the fallback).
  uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
    echo "[step10d/artifact-confirmed] pre-marker sync failed; post-merge-guard pre-sync remains the fallback"
  ```

  Then post
  `epm:merged v1` VIA THE `--file` CHANNEL — never `--note` — with a
  scratch file at `/tmp/issue-<N>-merged-note.md` (composed via the
  Write tool immediately before the post-marker call — never a Bash
  heredoc/printf; see the safe-case Success bullet, #1756) carrying fields
  `{artifact_confirmed: true, full_rebase_deferred: true, reason:
  "own commits touch foreign / out-of-scope paths: <paths>" (the content
  check is the only Guard-3 condition that can trip into this degrade —
  a missing merge-base hard-stops upstream and never reaches here),
  mb_first_parent: <yes|no>, verified_paths: [...]}` — plus
  `stranded_modified: [...]` when the stranded-MODIFIED scan above found
  branch-side MODIFIED/RENAMED/DELETED shared infra, or
  `stranded_modified: UNKNOWN — producer diff failed` when that scan's
  producer diff failed (field omitted when the scan came back empty).
  Same `--file` rationale as the safe-case
  Success bullet above — the argv-prose scan on `--note` blocks
  `reason:` text that quotes git verbs (session `7ce3a81f`).
  Update the chat title with `merged (artifact-confirmed)`. Skip the
  `gh pr merge` call; leave the PR open so a future `/issue <N>`
  re-invocation can retry the full rebase once the parent branch is
  itself merged. This is the standard outcome of guard 3 — the task
  has its science deliverables on `main` and is not blocked.
- **One or more deliverables missing on `origin/main`** -> do a
  **surgical additive checkout** of just this branch's own NEW files
  (the ones it added vs `origin/main` AND that live under the task's
  own paths — `tasks/*/<N>/`, `figures/issue_<N>/`,
  `eval_results/issue_<N>/`, `eval_results/issue_<N>_*/`,
  `ood_eval_results/issue_<N>/`). Compute:

  ```bash
  # Files this branch ADDED (status A ONLY) vs origin/main, restricted to
  # this task's own paths PLUS the workflow surface (a workflow-fix branch's
  # ADDED deliverable can be .claude/** / CLAUDE.md / .gitattributes). Never
  # sweeps shared src/ or scripts/ — the new-shared-src/ guard above already
  # refused the surgical path if the branch added src/.
  #
  # --diff-filter=A (ADDED-only), NEVER AM: this checkout does a WHOLESALE
  # `git checkout issue-<N> -- <path>` below (~line for the xargs git checkout),
  # which would OVERWRITE main's newer copy of a MODIFIED file with no conflict.
  # A-only guarantees every listed path does not yet exist on main, so the
  # checkout only CREATES — never clobbers. A branch that MODIFIES a
  # workflow-surface file is not fast-path-eligible (predicate (e)) and reaches
  # this block only via a genuine Guard-3 UNSAFE degrade, where the same
  # A-only safety applies.
  #
  # Three-dot origin/main...HEAD (merge-base..HEAD): the branch's OWN adds only.
  # Two-dot origin/main HEAD would additionally list files main advanced that
  # the branch never touched (status M-because-main-advanced), pulling them into
  # the checkout list — precisely the paths we must NOT overwrite.
  #
  # PRODUCER GUARD — materialize-then-check (mirrors the shared gate's
  # trigger diff): unchecked, a FAILED diff (bad ref, no merge-base) writes
  # an empty/partial list indistinguishable from "no additive files" and the
  # landing below fails OPEN. And an EMPTY list is itself an anomaly HERE:
  # this decision-tree branch is reached ONLY because deliverables are
  # MISSING on origin/main, so "nothing to add" means the diff lied or the
  # payload sits outside the A-only pathspec set (e.g. main deleted the
  # deliverable — status M, not A — or a scripts/-only payload); landing
  # anyway would push nothing and post `epm:merged {surgical_checkout:
  # true}` with nothing committed (a PHANTOM SUCCESS). Both arms hard-stop.
  if ! git -C "$WT" -c core.quotePath=false diff --name-only --diff-filter=A origin/main...HEAD -- \
      "tasks/*/<N>/" "figures/issue_<N>/" "eval_results/issue_<N>/" \
      "eval_results/issue_<N>_*/" "ood_eval_results/issue_<N>/" \
      ".claude/" "CLAUDE.md" ".gitattributes" "docs/methodology/issue_<N>.md" \
      > /tmp/issue-<N>-additive-files.txt; then
    echo "SURGICAL ABORT: additive-list diff FAILED — cannot enumerate the payload; route to epm:merge-failed, never land"
    false
  elif [ ! -s /tmp/issue-<N>-additive-files.txt ]; then
    echo "SURGICAL ABORT: additive-files list EMPTY on a deliverables-missing landing (phantom success); route to epm:merge-failed, never land"
    false
  fi
  ```

  Either `SURGICAL ABORT` arm (failed producer diff, or an empty additive
  list on this deliverables-missing landing) routes to the **Surgical
  checkout itself fails** bullet below: post `epm:merge-failed v1` with the
  abort line, surface ONE line in chat, CONTINUE — never fall through to
  the checkout/stage/push block, and never post `epm:merged`.

  **Single-flight probe (#1606)** first, per the Step 9c 1b statement:
  `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --pattern 'issue-<N>-surgical-gate|issue-<N>-lint-gate'`
  (self-/ancestor-excluding — exit 0 = clear, 3 = live foreign match; the
  surgical workload script `/tmp/issue-<N>-surgical-gate.sh` rides the
  detached unit's argv for its whole life under the #2115 launcher, and
  the `issue-<N>-lint-gate` alternate covers the tar/lint legs' gate-tree
  child argvs plus legacy inline-form launches — #2256).
  An `issue-<N>`-scoped hit (exit 3) = THIS gate-and-land sequence is still
  running — WAIT for exit, never relaunch into it (the outcome-sentinel
  `rm -f` below would clobber it, and the root holds ITS staged payload).
  A residual ambiguous hit that is neither this session's own gate nor a
  matching sibling gate: WAIT for exit, never kill — the same rule as this
  block's completion-read recovery arm.

  Then the **Gate-fleet arbitration (#1962)** probe, per the Step 9c 1b
  canonical paragraph:
  `uv run python "$REPO_ROOT"/scripts/step9c_baseline.py probe --fleet --exclude-issue <N>`
  — exit 3 ⇒ bounded queue (sleep 60, elapsed cap 2700 s), then launch
  anyway with the `[gate-fleet]` cap-expired line (fail-open).

  Then, from the **repo root on `main`** (never switch the branch
  there), checkout each path from the branch, stage by EXPLICIT PATH
  (never `git add -A`), commit PATHSPEC-LIMITED, and push. The
  pathspec-limited commit is load-bearing: many sessions commit to the
  shared repo root concurrently, so its index may carry a CONCURRENT
  session's staged files, and a bare `git commit` sweeps them in
  (#562/#550) — limiting the commit by pathspec commits
  ONLY this task's files and ignores every other staged entry:

  ```bash
  cd "$REPO_ROOT"
  # earlyoom-protect the gate — form (iii) (#1045 recipe, #1211; FAIL-OPEN,
  # see the shared gate block above): the preamble sits BEFORE the BASELINE
  # legs (they run before the checkout); this whole gate-and-land sequence
  # runs DETACHED via the same setsid + pid/log/rc-file breadcrumb shape as
  # Step 9c 1b and Step 10d forms (i)/(ii) — its two lint leg pairs total
  # ~9-12+ min and STRUCTURALLY exceed the 600s bg-Bash tool cap
  # (run_in_background=true carries the same 600 000 ms ceiling as
  # foreground, so an inline chain would be harness-killed at the cap
  # BEFORE the rc-write; the outer bg-Bash launches the detached unit and
  # exits in seconds). Sequenced checkout/commit/push remains bg-Bash-inline
  # in the launcher body (the sequence itself is < 5 min; the LINT PHASES
  # are what push it over the cap). END THE TURN after launching — wait
  # via the Monitor until-loop keyed on the single-flight probe above, or
  # the `/issue-tick <N>` cron re-wake; never repeated
  # `TaskOutput(block=true, timeout=600000)` polls (#1984). The workload
  # verbatim below is COMPOSED to /tmp/issue-<N>-surgical-gate.sh with the
  # WRITE TOOL as its own prior step (a script FILE — the awk/sed
  # single-quoted blocks below would need escape-heavy quoting inside an
  # inner `bash -c '...'` string; the Write tool and NEVER a
  # `cat > ... <<'EOF'` heredoc in the launcher Bash call — a heredoc body
  # rides the whole workload as Bash tool-call argv through the harness
  # transport, the #2115 stall surface; guards scan the full argv, #1756)
  # and setsid-nohup-
  # launched from within an outer `bash -c` wrapper so `$!` captures the
  # workload pid (`PYTEST_PID` below); the workload's final line is
  # `echo $? > /tmp/step9c-surgical-rc-issue-<N>` — rc-write inside the
  # SAME session-decoupled unit (§ Harvest self-harvest chaining), NEVER
  # spliced after `2>&1` on the outer bg-Bash line (§ Harvest NEVER-splice
  # rule — that parses as three commands, workload runs FOREGROUND inside
  # the $( ) capture, and the outer bg-Bash STILL dies at the 600s cap).
  # The `harvest=/tmp/issue-<N>-surgical-outcome.txt` breadcrumb names the
  # durable outcome path the § Successor / re-entry rule probes. Every
  # child forked inside the detached session inherits adj=-600 via the
  # self-choom below.
  #
  # Canonical launcher shape (the outer bg-Bash body — the workload verbatim
  # below is the script this launches):
  #   SURGICAL_SCRIPT=/tmp/issue-<N>-surgical-gate.sh
  #   STEP 1 — compose the script with the Write tool (its own tool call,
  #   BEFORE the launcher bg-Bash; never a heredoc in the Bash call):
  #     Write(file_path=$SURGICAL_SCRIPT, content=
  #       #!/usr/bin/env bash
  #       ... [the workload body verbatim from `sudo -n choom -n -600 -p $$`
  #           down to and incl. the terminal arms that write
  #           /tmp/issue-<N>-surgical-outcome.txt] ...
  #       echo $? > /tmp/step9c-surgical-rc-issue-<N>
  #     )
  #   STEP 2 — the launcher-only bg-Bash (argv stays tiny):
  #   chmod +x "$SURGICAL_SCRIPT"
  #   # trailing "$WT": unused by the script; rides the detached workload's argv
  #   # so worktree_audit's cwd/argv liveness harvest keeps the worktree for the
  #   # gate's whole life (#2246 item 1).
  #   PYTEST_PID=$(bash -c "setsid nohup env WT=\"$WT\" REPO_ROOT=\"$REPO_ROOT\" \
  #     OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  #     bash '$SURGICAL_SCRIPT' \"$WT\" < /dev/null > /tmp/issue-<N>-surgical-gate.log 2>&1 & echo \$!")
  #   ps -p "$PYTEST_PID" -o args= | head -1
  #   bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PYTEST_PID" >/dev/null \
  #     && LINT_GATE_CHOOM=ok || LINT_GATE_CHOOM=failed
  #   # ONE bounded retry on failed sweep (§ Detached VM-side long compute
  #   # phases; #1315):
  #   if [ "$LINT_GATE_CHOOM" = "failed" ]; then
  #     for _ in $(seq 1 30); do
  #       if pgrep -s "$PYTEST_PID" -a 2>/dev/null | grep -qE 'python3?'; then break; fi
  #       sleep 1
  #     done
  #     bash -o pipefail -c 'pgrep -s "$1" | xargs -rn1 sudo -n choom -n -600 -p' _ "$PYTEST_PID" >/dev/null \
  #       && LINT_GATE_CHOOM=ok \
  #       || { LINT_GATE_CHOOM=failed; echo "[warn] session choom re-sweep failed — surgical gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  #   fi
  #   echo "[step10d] surgical-gate detached pid=$PYTEST_PID log=/tmp/issue-<N>-surgical-gate.log rc=/tmp/step9c-surgical-rc-issue-<N> harvest=/tmp/issue-<N>-surgical-outcome.txt choom=$LINT_GATE_CHOOM"
  sudo -n choom -n -600 -p $$ >/dev/null 2>&1 && LINT_GATE_CHOOM=ok \
    || { LINT_GATE_CHOOM=failed; echo "[warn] choom failed — lint gate is earlyoom-UNPROTECTED (choom=failed)" >&2; }
  echo "[step10d] lint-gate earlyoom protection choom=$LINT_GATE_CHOOM"
  # Outcome sentinel (#1245): pre-rm; each terminal arm below writes it as
  # its LAST action (landed | push-failed | blocked-cleaned) — missing at
  # completion = the sequence died mid-run (completion-read below the block).
  rm -f /tmp/issue-<N>-surgical-outcome.txt
  # Pre-push workflow-lint gate — form (iii) (subsection above): the payload
  # lands in the ROOT tree, so BOTH lint runs use the root copy, sequenced
  # around the checkout — BASELINE BEFORE (payload-free tree; a post-checkout
  # "baseline" would re-lint the same contaminated tree, a degenerate
  # self-compare that fails open), GATED AFTER. The whole gate-and-land
  # sequence runs in that ONE background invocation, so GATE_ARMED /
  # BASE_RC / GATE_VERDICT remain same-invocation state (no cross-block
  # variable). Executable trigger
  # first: an artifact-only additive list skips both lint runs.
  GATE_ARMED=no
  # Output-test form, not `-q -v` rc (ugrep rc inversion, #928 -> #1125):
  if [ -n "$(grep -vE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
       /tmp/issue-<N>-additive-files.txt)" ]; then
    GATE_ARMED=yes
    # BASELINE legs (per-leg exit codes ARE captured — a baseline CRASH must
    # fail CLOSED via the crash arm below, never be `|| true`-erased; only
    # normalized failure LINES enter the compare for the legitimate
    # red-baseline rc=1-with-lines case; per-leg NO-DOWNGRADE max fold, same
    # rationale as the gate's executable block — a leg-1 crash must not be
    # erased by a leg-2 rc=1):
    BASE_RC=0
    timeout --kill-after=60s 1800s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
    timeout --kill-after=60s 1800s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-baseline.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$BASE_RC" ]; then BASE_RC=$rc; fi; }
  fi
  # MAPPED INVARIANT-TEST LEG (#1147) — form (iii): dormant for scripts/src
  # payloads by pathspec (no additive payload can match a GLOB_SCAN_TESTS
  # glob); an ADDED .claude/rules/*.md payload arms it via rules-pin pairs
  # (#1496). The leg exists as defense-in-depth should that pathspec set
  # ever grow; it costs one ~1 s helper call per surgical landing. Sequencing
  # mirrors the lint legs: TG BASELINE runs BEFORE the checkout (the payload
  # lands in the ROOT tree — a post-checkout "baseline" would be a degenerate
  # self-compare), TG GATED after. The GATED leg runs the ROOT copy (the
  # payload lands in the root tree); the BASELINE runs a detached sparse
  # scratch cut at origin/main via `step9c_baseline.py mapped-baseline`
  # (#2296) — payload-free BY CONSTRUCTION and immune to the #2015 stash
  # cycle, strictly better than the pre-#2296 root-before-checkout baseline
  # (payload-free only while no concurrent session was mid-commit).
  TG_RC=0; TG_BASE_RC=0; TG_CRASH=no
  : > /tmp/issue-<N>-tg-new.txt
  : > /tmp/issue-<N>-tg-new-nodes.txt
  : > /tmp/issue-<N>-tg-unclassifiable-nodes.txt
  if ! timeout --kill-after=30s 120s uv run python "$REPO_ROOT/scripts/select_step9c_tests.py" \
      --map-files /tmp/issue-<N>-additive-files.txt --repo-root "$REPO_ROOT" \
      > /tmp/issue-<N>-tg-map.txt 2>/tmp/issue-<N>-tg-map-err.txt; then
    TG_CRASH=yes   # helper failure: cannot classify the payload — fail CLOSED
  fi
  if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
    cut -f2 /tmp/issue-<N>-tg-map.txt | sort -u > /tmp/issue-<N>-tg-files.txt
    mapfile -t TG_TESTS < <(cut -f1 /tmp/issue-<N>-tg-map.txt | sort -u)
    # Sized from the selector's map (#1573; floor 600s, #1646):
    TG_T=$(grep -oE 'recommended-timeout-s=[0-9]+' /tmp/issue-<N>-tg-map-err.txt \
           | tail -1 | cut -d= -f2); [ -z "${TG_T:-}" ] && TG_T=600
    # Route TG fixture temp writes onto the data disk (#1408 recipe; #1363:
    # / at 100% killed a gate). Short --basetemp keeps AF_UNIX socket paths
    # under the 108-byte cap. Falls back silently (no TMPDIR, no --basetemp
    # => byte-identical argv) on pods/GCE with no data disk. GATED-leg-only
    # since #2296: the baseline helper routes its own temp writes via
    # gate_tmp_root() internally.
    TG_TMPROOT=$(uv run python "$REPO_ROOT/scripts/step9c_baseline.py" tmproot 2>/dev/null || true)
    if [ -n "$TG_TMPROOT" ]; then
      TG_BASETEMP=$(mktemp -d "$TG_TMPROOT/tg-XXXXXX")
    fi
    # BASELINE leg — base-pinned scratch (#2296; --base needs a fresh
    # origin/main: the bounded fetch degrades to staleness on failure, which
    # only biases the baseline toward MORE reds = toward PASS). rc is DATA on
    # stdout; a missing/unparseable rc= line fails CLOSED into TG_CRASH.
    # ${WT:-$REPO_ROOT}: the surgical path may run with no worktree in scope —
    # a non-sparse cones source degrades to the helper's floor profile.
    timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
    : > /tmp/issue-<N>-tg-baseline.txt
    TG_SCRATCH=""
    # Helper resolution (#2296 bootstrap) — see the shared gate block's fuller
    # note. ROOT copy by default; worktree fallback when the root lacks the
    # subcommand. ${WT:-$REPO_ROOT} here too: the surgical path may run with no
    # worktree in scope, in which case the root copy is the only candidate and
    # a still-missing subcommand fails CLOSED below.
    TG_S9B="$REPO_ROOT/scripts/step9c_baseline.py"
    TG_S9B_SRC=root
    { uv run python "$TG_S9B" mapped-baseline --help >/dev/null 2>&1 \
      && uv run python "$TG_S9B" classify-new-nodes --help >/dev/null 2>&1; } \
      || { TG_S9B="${WT:-$REPO_ROOT}/scripts/step9c_baseline.py"; TG_S9B_SRC=FALLBACK-worktree; }
    echo "[step10d] mapped-baseline helper: $TG_S9B_SRC ($TG_S9B)"
    # Baseline base = the MERGE-BASE of origin/main and the ROOT tree's HEAD
    # (#2348). Principle: the baseline base must match the VINTAGE OF THE
    # TREE THE GATED LEG RUNS — form (iii)'s gated leg runs the ROOT tree
    # (current local main + payload), so merge-base resolves in $REPO_ROOT
    # (= local main; strictly better than bare origin/main when the fetched
    # ref leads local main) — NEVER the worktree-fallback form used by the
    # cones source below: with a worktree in scope that would anchor the
    # baseline at the branch FORK POINT while the gated leg runs current
    # main, so any mapped test main broke since fork would read NEW and the
    # classify split would keep it blocking (its file is in the baseline
    # selection) — reintroducing the false-block class on this fence. A
    # resolution failure degrades LOUDLY to origin/main.
    TG_BASE_REF=$(git -C "$REPO_ROOT" merge-base origin/main HEAD 2>/dev/null) \
      || { TG_BASE_REF=origin/main; echo "WARN: merge-base resolution failed — TG baseline at origin/main (drift residual live for this run)"; }
    TG_BASE_OUT=$(timeout --kill-after=30s $((TG_T + 420))s uv run python \
      "$TG_S9B" mapped-baseline \
      --map-files /tmp/issue-<N>-additive-files.txt --root "$REPO_ROOT" \
      --cones-from "${WT:-$REPO_ROOT}" --base "$TG_BASE_REF" --timeout-s "$TG_T" \
      --out /tmp/issue-<N>-tg-baseline.txt \
      2>/tmp/issue-<N>-tg-baseline-err.txt) \
      || TG_CRASH=yes
    TG_SCRATCH=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^scratch_path=//p' | tail -1)
    TG_BASE_SELECTED=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^selected_path=//p' | tail -1)
    TG_BASE_RC=$(printf '%s\n' "$TG_BASE_OUT" | sed -n 's/^rc=//p' | tail -1)
    [ -n "${TG_BASE_RC:-}" ] || { TG_BASE_RC=0; TG_CRASH=yes; }   # fail CLOSED
  fi
  # `-C "$REPO_ROOT"` is the repo-root guard's designed deliberate-override
  # (#897): the hook's working-tree-revert detector would bounce the bare
  # `checkout <branch> -- <paths>` form; the `-C` names the tree explicitly.
  # `xargs -r` (--no-run-if-empty) is load-bearing: on an EMPTY additive list
  # a flag-less xargs still runs `git checkout issue-<N> --` ONCE with no
  # pathspec — a BRANCH SWITCH of the shared root (the FAST_PATH `-ge 1`
  # lower bound is the first defense; this is defense-in-depth).
  xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT" checkout issue-<N> --
  # GATED legs + verdict — the root tree now carries the payload. Same
  # normalize → comm -23 subtraction → verdict as the gate's executable
  # block; own-diff here = the additive-files list.
  GATE_VERDICT=pass
  if [ "$GATE_ARMED" = "yes" ]; then
    GATED_RC=0
    timeout --kill-after=60s 1800s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      > /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    timeout --kill-after=60s 1800s uv run python "$REPO_ROOT/scripts/workflow_lint.py" \
      --check-references --check-tables --check-asks --check-autonomous-asks \
      >> /tmp/issue-<N>-lint-gated.txt 2>&1 \
      || { rc=$?; if [ "$rc" -gt "$GATED_RC" ]; then GATED_RC=$rc; fi; }
    for leg in baseline gated; do
      grep -h '^workflow_lint: ' "/tmp/issue-<N>-lint-$leg.txt" \
        | grep -vE '^workflow_lint: (PASS$|FAIL \()' \
        | sed -E 's/:[0-9]+:/::/g' | sort -u \
        > "/tmp/issue-<N>-lint-$leg-norm.txt" || true
    done
    comm -23 /tmp/issue-<N>-lint-gated-norm.txt \
      /tmp/issue-<N>-lint-baseline-norm.txt > /tmp/issue-<N>-lint-new.txt
    # Offender-path-TOKEN set-membership against the additive-files list (same
    # awk as the shared gate — never a whole-line grep; gate-tree sub() is a
    # harmless no-op here, kept for textual parity; #1944):
    awk -v OWN=/tmp/issue-<N>-additive-files.txt '
      BEGIN { while ((getline l < OWN) > 0) own[l]=1 }
      /^workflow_lint: / {
        s = substr($0, 16); n = index(s, ":")
        path = (n > 0) ? substr(s, 1, n-1) : s
        sub(/^\/tmp\/issue-<N>-lint-gate-tree\//, "", path)
        gsub(/^[ \t]+|[ \t]+$/, "", path)
        if (path in own) print $0
      }' /tmp/issue-<N>-lint-gated-norm.txt \
      > /tmp/issue-<N>-lint-owndiff.txt || true
    # GATED_RC consumed HERE — CRASH ARM FIRST (fail CLOSED; same
    # classification as the gate's executable block): rc>1 on either leg
    # pair, or rc!=0 with ZERO normalized `workflow_lint:` lines, is a
    # linter CRASH -> GATE_VERDICT=crash (block path — the stage/commit/push
    # below runs ONLY on `pass`). Only then the attribution arm: a red
    # gated run blocks when payload-attributed (own-diff-named line OR NEW
    # non-empty).
    if [ "$GATED_RC" -gt 1 ] || [ "$BASE_RC" -gt 1 ] \
       || { [ "$GATED_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-gated-norm.txt ]; } \
       || { [ "$BASE_RC" -ne 0 ] && [ ! -s /tmp/issue-<N>-lint-baseline-norm.txt ]; }; then
      GATE_VERDICT=crash
    elif [ "$GATED_RC" -ne 0 ] \
       && { [ -s /tmp/issue-<N>-lint-owndiff.txt ] || [ -s /tmp/issue-<N>-lint-new.txt ]; }; then
      GATE_VERDICT=block
    fi
  fi
  # TG GATED leg (#1147) — the root tree now carries the payload; same
  # warnings-section drop -> grep -> line-number-blank -> tree-prefix
  # normalization -> comm -23 subtraction as the shared gate's executable
  # block, incl. its #1689 filter rationale + realpath-divergent residual
  # (own-diff here = the additive-files list; structurally
  # unreachable today, see the dormancy comment above the TG baseline leg).
  if [ "$TG_CRASH" = no ] && [ -s /tmp/issue-<N>-tg-map.txt ]; then
    ( cd "$REPO_ROOT" && timeout --kill-after=30s ${TG_T}s \
      env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
          NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
          ${TG_TMPROOT:+TMPDIR=$TG_TMPROOT} \
      uv run pytest "${TG_TESTS[@]}" -q -p no:cacheprovider \
        ${TG_BASETEMP:+--basetemp=$TG_BASETEMP/g} ) \
      > /tmp/issue-<N>-tg-gated.txt 2>&1 || TG_RC=$?
    if [ "$TG_RC" -gt 1 ] || [ "$TG_BASE_RC" -gt 1 ]; then TG_CRASH=yes; fi
    for leg in baseline gated; do
      awk '/^=+ warnings summary/{w=1; next} w && /^-- Docs:/{w=0; next} !w' \
        "/tmp/issue-<N>-tg-$leg.txt" \
        | grep -F -f /tmp/issue-<N>-tg-files.txt \
        | grep -vE '^E +assert ' \
        | sed -E 's/at line [0-9]+/at line N/g; s/:[0-9]+:/::/g; s/:[0-9]+([^0-9]|$)/:N\1/g' \
        | sed -e "s|${TG_SCRATCH:-/__eps_no_scratch__}|<TREE>|g" \
              -e "s|${WT:-/__eps_no_wt__}|<TREE>|g" \
              -e "s|${REPO_ROOT:-/__eps_no_root__}|<TREE>|g" \
        | sort -u \
        > "/tmp/issue-<N>-tg-$leg-hits.txt" || true
    done
    comm -23 /tmp/issue-<N>-tg-gated-hits.txt \
      /tmp/issue-<N>-tg-baseline-hits.txt > /tmp/issue-<N>-tg-new.txt
    # NODE-grain NEW-failure subtraction (#1573) — same pipeline + rationale
    # as the shared gate's executable block (sed msg-suffix strip, NOT awk
    # field-2: space-bearing string param ids must survive intact):
    for leg in baseline gated; do
      # msg-strip caveat: a literal ' - ' INSIDE a param id truncates here;
      # a same-prefix dash-bearing sibling collision fails toward pass (narrow doc-only residual, #1573)
      grep -E '^(FAILED|ERROR) ' "/tmp/issue-<N>-tg-$leg.txt" \
        | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' \
        | sort -u > "/tmp/issue-<N>-tg-$leg-nodes.txt" || true
    done
    comm -23 /tmp/issue-<N>-tg-gated-nodes.txt \
      /tmp/issue-<N>-tg-baseline-nodes.txt > /tmp/issue-<N>-tg-new-nodes.txt
    # SET-mismatch split (#2348) — same rationale + fail directions as the
    # shared gate block's classify step (own-diff here = the additive-files
    # list; the tg-unclassifiable-nodes.txt stale-file init is HOISTED to
    # the top of the block, beside the tg-new-nodes.txt init):
    if [ -s /tmp/issue-<N>-tg-new-nodes.txt ]; then
      uv run python "$TG_S9B" classify-new-nodes \
        --new-nodes /tmp/issue-<N>-tg-new-nodes.txt \
        --baseline-selected "${TG_BASE_SELECTED:-/__eps_no_selected__}" \
        --own-diff /tmp/issue-<N>-additive-files.txt \
        --out-block /tmp/issue-<N>-tg-new-nodes.txt \
        --out-unclassifiable /tmp/issue-<N>-tg-unclassifiable-nodes.txt \
        || echo "WARN: classify-new-nodes failed — every NEW node kept blocking (status quo)"
    fi
  fi
  # TG basetemp reaped after BOTH legs (no-op when routing never resolved).
  [ -n "${TG_BASETEMP:-}" ] && rm -rf "$TG_BASETEMP" || true
  # Fold the TG verdict into the SAME GATE_VERDICT the stage/commit/push
  # consumes below — crash-class first (fail CLOSED; never downgraded), then
  # the payload-attributed block arm; block/crash reuse the existing cleanup
  # + hard-stop path verbatim:
  if [ "$TG_CRASH" = "yes" ]; then
    GATE_VERDICT=crash
  elif [ "$TG_RC" -ne 0 ] \
     && { [ -s /tmp/issue-<N>-tg-new.txt ] || [ -s /tmp/issue-<N>-tg-new-nodes.txt ]; } \
     && [ "$GATE_VERDICT" != "crash" ]; then
    GATE_VERDICT=block
  fi
  # HARD STOP: stage/commit/push run ONLY on a pass verdict; a block cleans
  # the payload back out of the shared root (index + working tree).
  if [ "$GATE_VERDICT" = "pass" ]; then
    xargs -r -a /tmp/issue-<N>-additive-files.txt git add --
    git diff --cached --name-only   # sanity echo: spot any foreign staged entries
    xargs -r -a /tmp/issue-<N>-additive-files.txt git commit -m "issue-<N>: surgical additive checkout (full rebase deferred — guard 3)

  Branch unsafe to blind-rebase: own commits touch foreign /
  out-of-scope paths (the guard-3 content check). Cherry-picked this
  task's own added files only; shared src/ / scripts/ unchanged." --
    # PARTIAL-APPLY VERIFICATION (this task's Edit A): the branch's own ADDED
    # files were just staged from the branch tip and committed. Confirm — by
    # materialize-then-check — that every claimed additive path landed with
    # content byte-identical to its branch-tip source. The commit message
    # above asserts a "cherry-picked" apply; verify the assertion before
    # recording `landed`. Any path whose committed content diverges from its
    # branch-tip source is a PARTIAL apply (incident 3c24493113: an
    # improvised apply landed the test file but not the extractor half
    # it tested; main red 20 days). Edit A adds the guarantee to the
    # CURRENT documented template; a future improvised apply is out of
    # scope here (workflow_lint follow-up).
    # xargs feeds paths one-per-line (whitespace/glob-safe, matches the
    # block's convention); stderr is retained so a producer failure
    # surfaces its cause. The PUSH block below is now gated on
    # $APPLY_OK — a partial apply short-circuits it (methodology-critic
    # Must-Fix: bare `false` at the end of an if-branch does NOT halt
    # subsequent commands in the enclosing block, so a variable-gated
    # conditional is required).
    APPLY_OK=yes
    if ! xargs -r -a /tmp/issue-<N>-additive-files.txt \
         git -C "$REPO_ROOT" diff --name-only HEAD "issue-<N>" -- \
         > /tmp/issue-<N>-postapply-diff.txt; then
      echo "PARTIAL-APPLY VERIFY: diff HEAD vs issue-<N> FAILED — cannot certify apply; refusing to record landed"
      echo partial-apply-verify-failed > /tmp/issue-<N>-surgical-outcome.txt
      APPLY_OK=no
    elif [ -s /tmp/issue-<N>-postapply-diff.txt ]; then
      # Non-empty diff = one or more claimed paths were NOT byte-identically
      # applied. Fail LOUD — a claimed clean apply that did not land is the
      # 3c24493113 shape.
      echo "PARTIAL-APPLY VERIFY: $(wc -l < /tmp/issue-<N>-postapply-diff.txt) claimed additive path(s) diverge from their branch-tip source:"
      cat /tmp/issue-<N>-postapply-diff.txt
      echo "The 'surgical additive checkout' commit above does NOT reflect all claimed content; recording partial-apply outcome, NOT landed."
      echo partial-apply > /tmp/issue-<N>-surgical-outcome.txt
      APPLY_OK=no
    fi
    # Bounded push (the one network op on this arm): a hung push would wedge
    # the background call with the outcome sentinel unwritten. rc 124 takes
    # the push-failed arm — the same degradation as a rejected push (the
    # "Surgical checkout itself fails" bullet / sync-retry below).
    # GATED on APPLY_OK: a partial-apply outcome above short-circuits the
    # push and its `landed` sentinel; the enclosing background call still
    # exits non-zero via the `false` at the end of the else arm.
    if [ "$APPLY_OK" = "yes" ]; then
      if timeout --kill-after=30s 300s git push origin main; then
        echo landed > /tmp/issue-<N>-surgical-outcome.txt
      else
        echo push-failed > /tmp/issue-<N>-surgical-outcome.txt
        false
      fi
    else
      false   # partial-apply / verify-failed sentinel already written above
    fi
  else
    # BLOCKED: the checkout above already staged the A-only paths AND wrote
    # them to the working tree — clean BOTH with the hook-verified two-step
    # (the gate subsection's baseline-semantics bullet documents why the
    # one-shot restore form is hook-blocked): index-only unstage, then plain
    # rm of the now-untracked A-only files (absent from main — no main state
    # destroyed).
    xargs -r -a /tmp/issue-<N>-additive-files.txt git -C "$REPO_ROOT" restore --staged --
    xargs -r -a /tmp/issue-<N>-additive-files.txt rm -f --
    echo "BLOCKED: pre-push workflow-lint gate (verdict: $GATE_VERDICT) — fix the named offender (or crash cause) in the worktree, re-run ONCE; still failing -> epm:merge-failed (gate subsection, verdict cases 1/3). Payload cleaned from the root index + working tree."
    echo blocked-cleaned > /tmp/issue-<N>-surgical-outcome.txt
    false
  fi
  ```

  **Completion-read (form (iii)).** While the detached gate-and-land unit
  runs, END THE TURN and run no repo-root-mutating commands until this
  completion-read — the root holds staged payload for the ~5-6 min
  contaminated window (worst case, every bounded leg wedged, ~78 min —
  past the 60-min § Long-phase heartbeat boundary; rare, and a watcher
  force-stop there is fail-closed: the sentinel stays unwritten). The
  launcher bg-Bash exits in seconds — its completion (the harness
  notification) is NOT the done signal for the sequence. The sequence is
  done when the outcome sentinel exists (`rm -f`ed stale at launch) or
  the single-flight probe (`step9c_baseline.py probe --pattern
  'issue-<N>-surgical-gate|issue-<N>-lint-gate'`) reads
  CLEAR — wait via a Monitor until-loop on the probe or the
  `/issue-tick <N>` re-wake. A missing sentinel with a LIVE probe match =
  STILL RUNNING: keep waiting (the MISSING-sentinel recovery bullet below
  applies only after the probe reads CLEAR). Then read
  `/tmp/issue-<N>-surgical-outcome.txt` in a fresh FOREGROUND call:
  - `landed` -> BEFORE the `epm:merged v1` post, run the pre-marker root
    sync (#1725, surgical-additive landed path): the scratch worktree
    pushed directly to `origin/main` via `git push origin HEAD:main`, so
    the additive files are live on `origin/main` but the shared repo
    root's local `main` is still the pre-push snapshot. Same
    guard-argv rationale as the safe-case B1 site.

    ```bash
    # Root-sync before epm:merged (#1725; surgical-additive landed path):
    # the surgical scratch-worktree push landed the additive files on
    # origin/main, but the shared repo root's local main is still the
    # pre-push snapshot. sync_repo_root.py is single-flight flock-serialized;
    # fail-soft (the post-merge-guard pre-sync remains the fallback).
    uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
      echo "[step10d/surgical-landed] pre-marker sync failed; post-merge-guard pre-sync remains the fallback"
    ```

    Then post `epm:merged v1` as below.
  - `push-failed` (rejected OR timed-out, rc 124) -> the "Surgical
    checkout itself fails" bullet below (one `sync_repo_root.py` retry).
  - `blocked-cleaned` -> the gate subsection's case-1/3 fix path; read the
    background call's BLOCKED echo, which carries `$GATE_VERDICT`, for
    block-vs-crash attribution.
  - `partial-apply` -> post `epm:merge-failed v1` with `{reason: "surgical
    additive checkout — partial apply", diverged_paths: [...]}` (read the
    diverged file list from `/tmp/issue-<N>-postapply-diff.txt`), name the
    diverged files in ONE chat line, CONTINUE (idempotent retry on next
    `/issue <N>`). The task still parks per the standard failure path.
    (This task's Edit A: closes the false-`landed` gap by refusing to write
    the `landed` sentinel when the surgical commit's content diverges from
    the branch-tip source.)
  - `partial-apply-verify-failed` -> post `epm:merge-failed v1` with
    `{reason: "surgical additive checkout — apply-verification diff
    producer failed"}`, name the branch + worktree in ONE chat line,
    CONTINUE. (This task's Edit A: the diff producer itself errored — a
    materialize-then-check failure, treated as terminal rather than a
    false `landed`.)
  - MISSING sentinel -> the sequence died mid-run (tool kill / watcher
    force-stop / wedge-bound kill) and the root may hold staged payload.
    Recover IN THIS ORDER: (1) kill-before-relaunch probe FIRST
    (`pgrep -af 'issue-<N>-surgical-gat[e]|issue-<N>-lint-gat[e]'` —
    issue-scoped per the Step 10d form (i)/(ii) single-flight probe
    (#2256: the script paths ride the detached units' argv whole-life);
    exit-code trap: raw pgrep exits 0 on a
    LIVE match — INVERTED vs `step9c_baseline.py probe`, whose 0 = clear —
    this kill-arm keeps pgrep because it wants the pid list; on any
    residual ambiguous match WAIT for
    exit, never kill; the Step 0 single-orchestrator guard excludes
    same-issue concurrency).
 (2) Landed/committed classification BEFORE any cleanup —
    a shell killed between commit/push success and the sentinel write
    leaves the payload COMMITTED (tracked + clean), which a naive
    contamination probe misreads: check whether the surgical commit is on
    HEAD (`git -C "$REPO_ROOT" log -1 --format=%s` matches the surgical
    commit subject) and the additive paths are tracked + clean
    (`git status --porcelain` empty for them). Committed AND pushed
    (fetch, then `git merge-base --is-ancestor HEAD origin/main`) ->
    treat as `landed` (post `epm:merged v1`); committed but NOT pushed ->
    push-only retry (the sync-retry bullet) — NEVER `rm -f` committed
    files. (3) Only if genuinely uncommitted-contaminated (staged /
    working-tree payload present, no surgical commit) -> the hook-verified
    two-step clean (baseline-semantics bullet), then re-enter ONCE (the
    block is idempotent for the gate re-run case).

  **Guard-block recovery contract (improvised variants of this compound).**
  The checkout / restore forms in the blocks above are hook-fenced:
  `scripts/guard_repo_root_branch.sh` (a PreToolUse Bash hook) BLOCKS
  any improvised UNQUALIFIED variant run against the shared root — the
  bare `checkout issue-<N> -- <paths>` form
  (no `-C "$REPO_ROOT"`) trips its #897 checkout-pathspec detector, and a
  `restore` trips its #897 restore detector unless it carries `--staged`
  with NO worktree flag. The `git -C <path>` clause is the guard's designed
  per-clause waiver, so use the `-C "$REPO_ROOT"`-qualified fence lines
  VERBATIM — never retype them unqualified. The waiver is relied on here
  ONLY because both fence forms are NON-DESTRUCTIVE at the shared root
  (the checkout only CREATES A-only additive paths absent from `main`; the
  restore is `--staged` index-only) — NEVER generalize `-C "$REPO_ROOT"`
  to escape a block on any other / destructive command (the guard's own
  block message: never point `-C` at the repo root for a destructive op).
  On a guard block, the WHOLE compound Bash call was skipped, not just the
  offending clause (a PreToolUse deny rejects the entire tool call), so an
  earlier clause in the same call that writes
  `/tmp/issue-<N>-additive-files.txt` (the producer diff above) never ran
  either. The retry therefore RE-RUNS the producer diff clause to
  regenerate the list file BEFORE re-running the corrected `-C`-qualified
  consumer — re-running only the consumer (or `cat`-ing the list) fails
  with exit 128 / `cat: ... No such file` (#813/#1056).
  The guard's block message gives only generic
  worktree / `sync_repo_root.py` retry advice and does NOT mention the
  skipped producer — this paragraph is the recovery contract.

  BEFORE the `epm:merged v1` post, run the pre-marker root sync
  (#1725, surgical-additive checkout): the additive files are on
  `origin/main` via the scratch-worktree push, but the shared repo
  root's local `main` is still the pre-push snapshot. Same guard-argv
  rationale as B1/B3.

  ```bash
  # Root-sync before epm:merged (#1725; surgical-additive checkout path).
  # sync_repo_root.py is single-flight flock-serialized; fail-soft
  # (the post-merge-guard pre-sync remains the fallback).
  uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
    echo "[step10d/surgical-additive] pre-marker sync failed; post-merge-guard pre-sync remains the fallback"
  ```

  Then post `epm:merged v1` VIA THE `--file` CHANNEL — never `--note` —
  with a scratch file at `/tmp/issue-<N>-merged-note.md` (composed via
  the Write tool immediately before the post-marker call — never a
  Bash heredoc/printf; see the safe-case Success bullet, #1756)
  carrying `{artifact_confirmed: true, full_rebase_deferred: true,
  surgical_checkout: true, files: [...]}`. Same `--file` rationale as
  the safe-case Success bullet above — the argv-prose scan on `--note`
  fires on git-verb text (session `7ce3a81f`). Same chat title update
  as above.

- **Surgical checkout itself fails** (file conflicts, or push rejected after
  one `uv run python "$REPO_ROOT/scripts/sync_repo_root.py"` retry — the ONLY
  repo-root sync command; NEVER a hand-rolled repo-root `git pull`,
  the #967 `fatal: Cannot autostash` incident) — post
  `epm:merge-failed v1`
  with the error, surface ONE line in chat (branch + worktree path +
  one-line reason), CONTINUE. Same fail-fast policy as the safe case.

Never blind-`gh pr merge` (any `$MERGE_FORM`) a branch that tripped guard 3
— that is the exact #458 / #479 incident class this section exists to prevent.

#### Post-merge stale-task-folder guard (runs after EVERY merge form lands)

Run this AFTER any of the three merge forms above lands (safe-case
`gh pr merge $MERGE_FORM`, the merge-conflict-recovery retry, or the
artifact-confirmed / surgical-additive checkout). A merge commit — most
often the recovery's `git merge origin/main`, but also any improvised
merge taken when `--rebase` keeps being refused — can import THIS task's
OLD status folder onto `main` next to its live one (e.g.
`tasks/approved/<N>/` lands alongside `tasks/awaiting_promotion/<N>/`,
same task number, two status dirs). The autonomous-session watcher then
reads the stale folder as a live task and respawns the session
indefinitely (#644, #643). Guard 1 above
catches FOREIGN tasks' folders but not this task's own old-status
duplicate, and it only runs on the safe-case (`$MERGE_FORM`) path. Keep exactly ONE
folder for this task on `main` — and never by deleting origin's ONLY copy
while the canonical status-mv is unpushed (#1300):

```bash
# Canonical folder for this task (NEVER hand-build tasks/<status>/<N> —
# status is unknowable here; resolve via task.py find, CLAUDE.md rule).
CANON=$(realpath --relative-to="$REPO_ROOT" \
  "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")
# MATERIALIZE ls-tree to a file and check each producer's OWN exit code
# (find/CANON, fetch, ls-tree): piped straight into grep with a trailing
# `|| true`, a FAILED producer is indistinguishable from "no duplicate
# folders" and the guard fails OPEN — cleanup silently skipped, the watcher
# respawns against the stale folder (incident #644). Same materialize-then-
# check pattern as the pre-push lint-gate trigger diff (#1047). Failure arms
# are TERMINAL (echo + false — routes to the epm:merge-failed handling
# above); never proceed believing cleanup ran.
# PRE-SYNC (this task's Edit B): the local root routinely lags origin/main
# by a completed-status mv committed locally but not yet pushed
# (#1688). Run the sanctioned root sync
# UNCONDITIONALLY before the guard's canonical-folder check, so the
# guard's nonzero exit becomes reserved for genuine drift rather than the
# expected unpushed-mv state. sync_repo_root.py is single-flight
# flock-serialized (a concurrent sync returns exit 0 without re-syncing),
# so this call is idempotent and tolerant of in-flight state. Failure is
# NON-FATAL — the guard's own unpushed-mv pre-check (#1300) remains the
# fallback recovery if the sync did not fully converge.
uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || \
  echo "post-merge guard pre-sync: sync_repo_root.py exited non-zero; guard's own unpushed-mv pre-check is the fallback"
if [ -z "$CANON" ]; then
  # task.py find / realpath failed -> empty CANON. Classifying with an empty
  # CANON would mark the CANONICAL folder itself as a duplicate and rm it.
  echo "post-merge stale-task-folder guard: task.py find <N> produced empty CANON — refusing to classify duplicates"
  false
elif ! git -C "$REPO_ROOT" fetch origin main --quiet; then
  # A failed fetch leaves origin/main at its PRE-merge state: the duplicate
  # imported by the merge that JUST landed is invisible — the guard's
  # primary blind spot, not a lesser staleness.
  echo "post-merge stale-task-folder guard: git fetch origin main FAILED — origin/main may predate the just-landed merge; cannot certify no stale task folders"
  false
elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
    > /tmp/issue-<N>-postmerge-lstree.txt; then
  echo "post-merge stale-task-folder guard: git ls-tree origin/main FAILED — cannot certify no stale task folders"
  false
# Unpushed-mv pre-check (#1300): CANON absent from the materialized
# origin/main ls-tree means origin's only folder for this task is (almost
# always) the OLD-status copy of a status mv committed on local main but
# not yet pushed — classifying it as a duplicate would delete origin's
# ONLY folder for the task (the #1300 incident: ZERO tasks/*/1291
# folders left on origin). Recovery: land the local mv via the sanctioned root
# sync (the fleet-standard single-flight helper — it pushes ALL committed
# local-main state, not just this task's mv), RE-RESOLVE the canonical
# path (the sync pull-rebases the local root, so the canonical status can
# change in EITHER lag direction — a failed re-resolve keeps the previous
# value and fails closed below), re-fetch, REGENERATE the ls-tree file
# (same materialize-then-check form), then re-check. Bounded 2 attempts;
# the ls-tree RE-CHECK is the arbiter, NOT the helper's exit code (exit 0
# includes the in-flight state — same 2-attempt shape as the local-residue
# tail below). The condition is a command list: a SUCCESSFUL recovery
# makes the final still-absent test fail, the branch is NOT taken, and
# evaluation falls through to the DUPES classification below against the
# REGENERATED file (a merge-imported duplicate can coexist with the
# unpushed mv and must still be removed). A failed mid-recovery re-fetch
# or regen can leave a stale listing that still carries CANON and falls
# through — the guarantee is the membership test itself: classification
# only ever proceeds when CANON is present in the listing it reads, so
# this arm never opens a delete of the canonical folder. Only a
# still-absent CANON takes the branch — terminal echo + false, nothing
# deleted.
elif ! grep -qxF -- "$CANON" /tmp/issue-<N>-postmerge-lstree.txt \
    && { for _ in 1 2; do
           uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
           NEW_CANON=$(realpath --relative-to="$REPO_ROOT" \
             "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)")
           [ -n "$NEW_CANON" ] && CANON="$NEW_CANON"
           git -C "$REPO_ROOT" fetch origin main --quiet \
             && git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
                > /tmp/issue-<N>-postmerge-lstree.txt \
             && grep -qxF -- "$CANON" /tmp/issue-<N>-postmerge-lstree.txt \
             && break
         done
         ! grep -qxF -- "$CANON" /tmp/issue-<N>-postmerge-lstree.txt; }; then
  echo "post-merge stale-task-folder guard: canonical folder $CANON still ABSENT from origin/main after 2 root syncs — cannot classify duplicates (removing origin's only copy would leave ZERO folders for task <N>)"
  false
# Work arm: every committed task-<N> folder on origin/main (matches
# tasks/<status>/<N> exactly — the anchored $ excludes deeper paths like
# .../<N>/artifacts). The elif condition is a two-command list: mapfile
# fills DUPES from the FILE (grep semantics identical to the old pipe;
# no-match `|| true` is a legitimate empty DUPES), then the [ ... ] test —
# the LAST command's exit — decides the branch. Empty DUPES on a healthy
# read = clean no-op (exit 0), preserving idempotent re-runs.
elif mapfile -t DUPES < <(grep -E "^tasks/[^/]+/<N>$" \
      /tmp/issue-<N>-postmerge-lstree.txt \
      | grep -v -F -x "$CANON" || true); [ "${#DUPES[@]}" -gt 0 ]; then
  # Remove the duplicate(s) in a SPARSE SCRATCH WORKTREE detached at the
  # SAME fetched origin/main the detection just read — NEVER a root
  # `git rm`. The duplicates live on origin/main but are usually ABSENT
  # from the LOCAL root tree (local main predates the just-landed
  # server-side merge), so a root `git rm` fails pathspec, and the
  # improvised checkout-pathspec fallback at the root is hook-blocked
  # every time (#1253; session 82f5b16a, /issue 1198). The scratch
  # worktree needs no local-root state (the duplicate exists there BY
  # CONSTRUCTION), stages in its OWN index (no concurrent-session staging
  # races), and every command is `git -C`-scoped (the hook's designed
  # override). Sparse cone = the duplicates + scripts/hooks (the commit's
  # own pre-commit gitleaks hook runs `bash scripts/hooks/gitleaks_scoped.sh`
  # worktree-root-relative with always_run — exit 127 without it, #1780;
  # toplevel .gitleaks.toml/.gitleaksignore ride cone mode automatically):
  # a FULL checkout is ~7.7 GB / ~100k files on the shared VM root disk.
  SCRATCH=/tmp/issue-<N>-postmerge-scratch
  # Pre-clean a scratch leaked by an earlier crashed run. Failure here is
  # tolerable (nothing to clean): the worktree add below is the loud gate.
  git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null || true
  rm -rf "$SCRATCH"
  git -C "$REPO_ROOT" worktree prune
  # Stage: add (detached, no checkout) -> cone init FIRST (git 2.34:
  # `set --cone` is silently a literal PATTERN, non-cone) -> cone = the
  # duplicates -> populate -> rm -> commit. Flag order `--detach
  # --no-checkout` is load-bearing for a bare copy of the add line.
  if ! { git -C "$REPO_ROOT" worktree add --detach --no-checkout "$SCRATCH" origin/main \
         && git -C "$SCRATCH" sparse-checkout init --cone \
         && git -C "$SCRATCH" sparse-checkout set "${DUPES[@]}" scripts/hooks \
         && git -C "$SCRATCH" checkout --detach origin/main \
         && git -C "$SCRATCH" rm -r -q "${DUPES[@]}" \
         && git -C "$SCRATCH" commit -q -m "post-merge: remove stale task #<N> folder(s) imported by Step 10d merge

$CANON is the canonical folder; the duplicate(s) were re-imported by the
merge commit and would be read as a live task by the session watcher
(incident #644)."; }; then
    echo "post-merge stale-task-folder guard: scratch-worktree staging FAILED — stale folder(s) NOT removed: ${DUPES[*]}"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  # Land: push; on rejection (origin advanced under fleet churn) ONE
  # bounded fetch + rebase + push retry INSIDE the scratch worktree
  # (`git -C` — never a root rebase). A concurrent removal of the same
  # duplicate rebases to an empty commit and is dropped: the up-to-date
  # push and the verify arm below still pass (idempotent).
  elif ! { git -C "$SCRATCH" push origin HEAD:main \
           || { git -C "$SCRATCH" fetch origin main --quiet \
                && git -C "$SCRATCH" rebase origin/main \
                && git -C "$SCRATCH" push origin HEAD:main; }; }; then
    git -C "$SCRATCH" rebase --abort 2>/dev/null
    echo "post-merge stale-task-folder guard: removal commit did NOT land on origin/main after 1 retry"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  # Verify against a FRESH fetch that origin/main now carries exactly ONE
  # folder for this task (same materialize-then-check shape as detection).
  elif ! git -C "$REPO_ROOT" fetch origin main --quiet; then
    echo "post-merge stale-task-folder guard: verify fetch FAILED — cannot certify the removal landed"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  elif ! git -C "$REPO_ROOT" ls-tree -d -r --name-only origin/main \
      > /tmp/issue-<N>-postmerge-verify.txt; then
    echo "post-merge stale-task-folder guard: verify ls-tree FAILED — cannot certify the removal landed"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  elif mapfile -t STILL < <(grep -E "^tasks/[^/]+/<N>$" \
        /tmp/issue-<N>-postmerge-verify.txt \
        | grep -v -F -x "$CANON" || true); [ "${#STILL[@]}" -gt 0 ]; then
    echo "post-merge stale-task-folder guard: stale folder(s) STILL on origin/main after push: ${STILL[*]}"
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" 2>/dev/null
    false
  else
    git -C "$REPO_ROOT" worktree remove --force "$SCRATCH" \
      || echo "WARN: scratch worktree cleanup failed ($SCRATCH is inert; /tmp clears on reboot and git gc prunes the metadata)"
    # LOCAL-tree residue: a root that pulled origin/main in the window
    # between the merge landing and the removal landing holds a tracked
    # local copy the session watcher can misread (incident #644 reads the
    # LOCAL tree). Converge via the sanctioned root sync. CAUTION — the
    # helper's contract: exit 0 does NOT by itself mean the pull ran (exit
    # 0 includes the in-flight state), so the existence RE-CHECK is the
    # arbiter, with one in-flight re-run (the helper's own prescription),
    # then fail loud — same 2-attempt shape as the old push-recovery tail.
    STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    if [ -n "$STALE_LOCAL" ]; then
      uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
      STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    fi
    if [ -n "$STALE_LOCAL" ]; then
      uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
      STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    fi
    # Empty-dir residue (#1780 -> #1792): the sync's checkout removes
    # tracked FILES but an untracked empty leftover dir is invisible to
    # git — no number of root syncs clears it. Zero-content dirs (no
    # files, no symlinks — the JOINT probe over ALL persisting paths)
    # are rmdir'd depth-first (rmdir refuses non-empty dirs; inert to
    # git state), then STALE_LOCAL is RE-DERIVED — never blind-cleared —
    # so late-arriving content or a failed rmdir still fails loud
    # below. $STALE_LOCAL is deliberately unquoted: multi-path
    # word-split over `ls -d` output (task paths carry no whitespace).
    if [ -n "$STALE_LOCAL" ] \
       && [ "$(cd "$REPO_ROOT" && find $STALE_LOCAL \( -type f -o -type l \) 2>/dev/null | wc -l)" -eq 0 ]; then
      (cd "$REPO_ROOT" && find $STALE_LOCAL -depth -type d -exec rmdir {} \; 2>/dev/null) || true
      STALE_LOCAL=$(cd "$REPO_ROOT" && ls -d "${DUPES[@]}" 2>/dev/null || true)
    fi
    if [ -n "$STALE_LOCAL" ]; then
      echo "post-merge stale-task-folder guard: LOCAL stale copy/copies persist after 2 root syncs: $STALE_LOCAL — origin/main is clean but the local root still carries the folder(s)"
      false
    fi
  fi
fi
```

This guard is idempotent: a clean `main` (no duplicate) leaves `DUPES`
empty and the block is a no-op, so re-running Step 10d on a later
`/issue <N>` re-invocation is safe. A FAILED producer (empty `CANON` from
`task.py find`, `fetch`, or `ls-tree`) instead exits the block non-zero
through a terminal echo + `false` arm — the epm:merge-failed handling —
rather than reading as "no duplicates" (#1184; the #1047
materialize-then-check pattern). The unpushed-mv pre-check (#1300)
refuses to CLASSIFY while the canonical folder is absent from
origin/main — under routine local-main push lag origin's only copy is
the OLD-status folder of a not-yet-pushed status mv, and classifying it
as a duplicate deleted origin's only folder for task 1291. The arm lands
the local mv via the
sanctioned root sync, re-resolves the canonical path, re-fetches,
regenerates the ls-tree file, and re-checks (2 bounded attempts; the
ls-tree re-check is the arbiter, not the helper's exit 0); a successful
recovery falls through to classification against the regenerated file,
and a still-absent canonical folder fails loud with nothing deleted. One
pre-existing residual stays out of this fix's scope: when BOTH folders
are already on origin/main and the LOCAL canonical resolution is stale,
the guard can still classify origin's newer folder as the duplicate —
that wrong-direction delete predates this change. The work arm never touches the local
root index — the removal is staged and pushed from a sparse scratch
worktree detached at the fetched `origin/main`, so it succeeds whether or
not the local root has pulled the merge (the #1253 root-`git rm`
pathspec failure). The local-residue tail converges the local root via
`scripts/sync_repo_root.py`, with the existence re-check as the arbiter
(the helper's exit 0 alone does not prove the pull ran). Zero-content
leftover dirs (no files, no symlinks) are rmdir'd depth-first before the
loud failure (#1780) — untracked empty dirs are invisible to git and no
sync can clear them; non-empty residue still fails loud.

#### Terminal teardown (code-change path only; runs AFTER `epm:merged v1` has been posted)

Fires ONLY on the code-change path (`kind: infra | batch | analysis |
survey` — the arm that reached Step 10d via Step 10 step 6's
`epm:merged`-not-yet-present branch, #1723). The experiment path
already parked at `awaiting_promotion` in Step 9b and its own terminal
transition to `completed` happens later on user promotion, so this
sub-section is UNREACHABLE from that arm by design — the routing
predicate is `kind ∈ {infra, batch, analysis, survey}`.

Runs AFTER `epm:merged v1` has been posted AND AFTER the
`#### Post-merge stale-task-folder guard` above has finished reconciling
the shared `main` tree. All four `epm:merged` posting sites reach this
block on success — the safe-case `gh pr merge $MERGE_FORM` above, the
merge-conflict-recovery retry, the artifact-confirmed (guard-3-tripped)
sentinel, and the surgical additive checkout — because the stale-folder
guard runs on ALL of them ("runs after EVERY merge form lands") and
this sub-section is its immediate successor. The block below fires
IDEMPOTENTLY: a re-entry that already sees `status == "completed"` +
`epm:done` present exits as a no-op (the standard SKILL.md resume
convention).

1. **Run CRON-TEARDOWN** — the two-leg sweep (§ CRON-TEARDOWN
   procedure; recurring tick + stray one-shot `/issue <N>` wakeups).
   The `/issue-tick` backstop stayed armed through the
   entire Step 10d merge window (up to ~33 min under fleet
   churn); a wedged / refused session during the
   merge would have been re-driven by the tick, and now that
   `epm:merged` is posted the backstop has done its job. Step 1 is
   idempotent — a paranoid re-entry that already ran teardown reads
   both legs empty and no-ops.

2. **Apply the terminal status** via `task.py set-status`:
   ```bash
   uv run python scripts/task.py set-status <N> completed \
     --note "Step 10d auto-complete: merged, terminal teardown"
   ```
   `<new-status>` is always `completed` for the code-change kinds —
   code-change paths never seed `followups_running` (Step 10 step 5's
   destination logic already selected `completed` for these kinds, and
   the experiment-path branches — the ones that can pick
   `awaiting_promotion` / `followups_running` — never reach this
   sub-section).

3. **Post final event `epm:done v1`** summarizing outcome, key numbers,
   what's confirmed/falsified, what's next, plus a link to the
   worktree-side write-up location and the merge SHA(s) recorded on the
   `epm:merged` note just posted above. Include the line
   `Moved to **completed**.`

4. **Terminal landing confirmation (#1868; incident #1792).** Runs as
   the session's FINAL act — after every terminal marker post this
   session makes, including the §5 `post_step_completed.py` record.
   `completed` + `epm:done` existing only locally is a crash-window:
   this is the one site where "the next re-entry will fix it" does not
   hold (the session ends here, and the resume-semantics row for
   `completed` + `epm:done` + `epm:merged` is a no-op), so the terminal
   record must be CONFIRMED on origin before the session ends.
   `scripts/sync_repo_root.py` exits 0 on `state=in-flight` BY DESIGN —
   "your push has NOT landed; re-run after the in-flight sync
   completes" (sync_repo_root.py L33-35) — so the retry duty is
   CALLER-owned, and this step is that caller. (#1792)

   The LANDED arbiter is a fetched-origin blob check — the task's
   canonical `events.jsonl` on `origin/main` carries `"epm:done"` —
   NEVER the sync helper's exit code (exit 0 includes
   `state=in-flight`): the same arbiter-not-exit-code doctrine as the
   post-merge guard above (its existence re-check, not the helper's
   exit 0, proves the pull ran). Bounded by construction — 2 attempts,
   one 20 s inter-attempt wait — never a multi-hour poll (the #1317
   anti-pattern). Nothing here blocks or reverses the `completed`
   transition. The KEPT-stash surfacing duty (#1751) applies to these
   sync invocations like every other.

   ```bash
   REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
   CANON=$(realpath --relative-to="$REPO_ROOT" \
     "$(uv run python "$REPO_ROOT/scripts/task.py" find <N>)" 2>/dev/null)
   if [ -z "$CANON" ]; then
     # Resolution failure, NOT a landing failure — echo the distinct
     # diagnostic; do NOT post the terminal-landing-unconfirmed note.
     echo "[step10d] terminal landing check SKIPPED — task.py find <N> resolved no canonical folder (empty CANON; resolution failure, not a landing failure)"
   else
     LANDED=no
     for ATTEMPT in 1 2; do
       uv run python "$REPO_ROOT/scripts/sync_repo_root.py"
       timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
       if git -C "$REPO_ROOT" cat-file -e "origin/main:$CANON/events.jsonl" 2>/dev/null \
          && git -C "$REPO_ROOT" show "origin/main:$CANON/events.jsonl" \
             | grep -qF '"epm:done"'; then
         LANDED=yes; break
       fi
       if [ "$ATTEMPT" -eq 1 ]; then sleep 20; fi   # let an in-flight sibling sync finish
     done
     if [ "$LANDED" = no ]; then
       echo "[step10d] terminal landing UNCONFIRMED after 2 bounded sync attempts — completed/epm:done exist only locally (crash-window; #1792); next successful fleet sync is the backstop"
       uv run python "$REPO_ROOT/scripts/task.py" post-marker <N> epm:progress \
         --note "terminal-landing-unconfirmed after 2 bounded sync_repo_root attempts (state=in-flight or transport) — completed status move + epm:done not yet observed on origin/main; next successful fleet sync carries them (#1868)"
     else
       echo "[step10d] terminal landing CONFIRMED on origin/main (attempt $ATTEMPT)"
     fi
   fi
   ```

   On the UNCONFIRMED arm the `epm:progress` note is itself a local
   commit that rides the next successful fleet sync — a self-describing
   residual: whichever session's sync next converges the shared root
   carries both the note and the terminal record to origin. Named
   re-entry residual: a later `/issue <N>` re-entry that retries the
   merge (the resume-semantics `completed` + `epm:done` + no
   `epm:merged` row) posts fresh markers whose landing this already-run
   step does not re-confirm — and a prior round's `epm:done` on origin
   would satisfy the arbiter regardless. Acceptable: in that state the
   terminal record is already durable on origin (exactly the
   crash-window class this step closes), and the fleet-sync backstop
   carries the fresh commits. After this step the session ends.

**Terminal-failure branch.** If the merge terminally failed after every
retry surface exhausted (`epm:merge-failed v1` posted at the safe-case
Failure bullet, the merge-conflict-recovery Failure arm, the
artifact-confirmed / new-shared-`src/`-infra refusal, or the surgical
checkout's `push-failed`/`partial-apply` arms), the code-change task
still needs to complete (see the Failure bullet's own contract:
"a code-change task still completes"). Run the SAME four-step sequence
(CRON-TEARDOWN → `set-status completed` → `epm:done` → terminal landing
confirmation), but the
`epm:done` note records `merge_status: failed` and links to the
`epm:merge-failed v1` marker for the manual-resolution audit trail. The
merge retries idempotently on the next `/issue <N>` re-invocation
regardless (per the resume-semantics table's `completed` + `epm:done`
+ no `epm:merged` row above).

---
