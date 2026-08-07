# Step 5: Code review loop (Codex ensemble)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Only if status is `running` and the appropriate implementation marker
(`epm:experiment-implementation v<n>` for experiments, `epm:results v<n>`
for infra) is present.

This step runs an **ensemble of two reviewers in parallel** — the Claude
`code-reviewer` agent and the `codex-code-reviewer` Codex twin (gpt-5.5
via the OpenAI Codex plugin's `companion task` runtime). On verdict
disagreement (PASS-class vs FAIL), a `reconciler` agent (Claude) issues
a binding tie-break. See (see workflow.yaml § ensemble_review) for the
canonical contract.

**Round push hygiene.** Any branch push run during this loop — yours
between rounds, or the implementer's per its spec — is BARE with its exit
code checked: `git -C "$WT" push origin issue-<N>`. NEVER pipe a
push/merge through `tail`/`grep`/`head` (the `guard_piped_git_push.sh`
PreToolUse hook blocks the piped shape; a pipe masks a rejected push).
Copy the verbatim forms from Step 10d § "Bare push / merge snippets".

**5a. Spawn both reviewers in parallel (fresh contexts, single message).**

**Quota-sentinel pre-check first (#1204).** Run the canonical pre-spawn
check (CLAUDE.md § Codex ensemble review). `CODEX_QUOTA_LIVE` → spawn
ONLY the Claude `code-reviewer` this round; record the Codex twin as an
instant confirmed no-show per the Step 5d no-show fallback (no
durable-verdict probe — nothing was dispatched) and post the one-line
`epm:progress` note.

**Spec-freshness check first (worktree-cwd sessions; applies at EVERY
ensemble/agent fan-out — here, the Step 9a analyzer + critic ensembles,
and 9a-bis).** The Agent tool loads agent specs (and Skill playbooks)
from the SESSION's cwd, and a worktree cut before a later
workflow-surface fix never inherits it — so subagents silently run stale
specs for the worktree's lifetime (#557).
Before dispatching, sync the worktree's workflow surface from FETCHED
`origin/main` (local `main` routinely lags origin on the shared root
under fleet load — #1724 synced regressed spec bytes from a lagging
local `main`; #1747 migrated the source ref, mirroring the landed
Step 10d re-sync recipe. The sync is worktree-only: it skips
explicitly when the session already runs on `main`):

```bash
# Step 5a WANTS the WORKTREE root (that is where the spec-freshness sync writes)
# — NOT the #506 path-doubling bug; do NOT change to --git-common-dir here. The
# on-main skip case (session already on main) is why show-toplevel is correct.
WT=$(git rev-parse --show-toplevel)
# On-main skip (#1747): with FETCHED origin/main as the sync source the old
# "diff against local main is vacuous on a main checkout" self-no-op is GONE —
# a repo-root session whose local main lags origin would check out origin/main
# content into the SHARED root working tree and commit on main (a
# concurrent-committer hazard, CLAUDE.md § Concurrent repo-root committers).
# Skip the ENTIRE sync body — pass-1 dirty-family scan included (its MB..HEAD
# output on an ahead-of-origin root would print spurious dirty-family
# warnings) — when the session's branch is main.
if [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "main" ]; then
  echo "[step5a] session on main (repo root) — spec-freshness sync is worktree-only; skipping"
else
# Lint/guard family rides the sync (#1560): the specs synced below are budget-
# checked by workflow_lint.py constants, enforced by .claude/hooks/, and pinned
# by the test_workflow_lint*/test_guard_* pin tests — syncing
# specs without their enforcing family creates the #1489/#1482/#1417 vintage
# skew. #1972 widens the set: .claude/agent-memory (singleton, protected by
# the uncommitted-dirt arm below), the Step 9c selector triple (lint family),
# and the per-FILE sibling-issue script/test arm at the end of this block.
# `:(glob)` is a git pathspec (never shell-expands: no path starts with
# ":(glob)"), so `git checkout origin/main --` matches main-NEW pin tests too. The
# per-file branch-side-edit guard's skip grain is PER-ITEM: a branch editing
# ONE pin test skips the whole `:(glob)` family entry (fail-safe — status-quo
# staleness for those files, never a clobber).
# Step 5a family-atomic sync (#1714 — supersedes #1560's per-item skip
# for coupled specs, while preserving the fail-safe direction: any dirty
# member widens the skip to the whole family, never narrows it into a
# clobber; #535).
#
# 3 coupled families exist in SPECS:
#   FAMILY_workflow: .claude/workflow.yaml <-> .claude/skills/markers.md
#     (markers.md's marker-kinds + active-statuses tables are GENERATED
#     from workflow.yaml via `workflow_lint.py --emit-tables`; syncing
#     one without the other creates a stale-derived tree — the 0e2c3b21
#     incident, 2026-07-26)
#     <-> :(glob)tests/test_issue_skill_*.py (prose-pin tests over
#     .claude/skills content; syncing SKILL.md without its paired pin
#     test reds the Step 9c gate — the #1824 vintage skew, #1883)
#   FAMILY_lint: scripts/workflow_lint.py <-> :(glob)tests/test_workflow_lint*.py
#     plus tests/test_workflow_yaml.py and tests/test_autonomous_session_watch.py
#     (pin tests import symbols from workflow_lint.py; syncing new pin
#     tests against a stale linter is a collection ImportError — the
#     2de5253e incident, 2026-07-26)
#     plus scripts/select_step9c_tests.py <-> tests/test_select_step9c_tests.py
#     <-> tests/step9c_workflow_invariant_manifest.txt (#1972: the pin test
#     importlib-loads the selector BY PATH — the WORKTREE copy — and its
#     case 6b pins WORKFLOW_INVARIANT set-equal to the manifest file; the
#     historically dominant selector edit is an invariant-membership change
#     that updates all THREE together on main, so syncing any strict subset
#     manufactures exactly the #1824/#1860 half-sync skew)
#   FAMILY_guard: .claude/hooks <-> :(glob)scripts/guard_*.sh
#                                <-> :(glob)tests/test_guard_*.py
#                                <-> tests/test_guard_lessons_edit.py
#     (guard tests exercise the hook + guard-script implementations;
#     syncing tests against stale hooks fails behaviorally, and the
#     scripts/guard_*.sh PreToolUse implementations — e.g.
#     guard_repo_root_branch.sh, guard_repo_root_pull.sh — are executed
#     by the test_guard_* pins: syncing the tests without them red-flags
#     main-green nodes on pure version skew — the #1860/#1862 half-sync)
#
# Everything else in SPECS is a singleton (its own family, no coupling):
# .claude/agents, .claude/agent-memory (#1972 — always-appended memory
# indexes the lint budget checks scan; no coupling, so its protections are
# the uncommitted-dirt arm below + the branch-side-edit guard),
# .claude/rules, CLAUDE.md.
declare -A FAMILY_OF
FAMILY_OF[".claude/workflow.yaml"]="workflow"
FAMILY_OF[".claude/skills"]="workflow"    # contains markers.md, the derived table target
FAMILY_OF["tests/test_workflow_yaml.py"]="workflow"    # imports render_*_table from workflow_lint AND reads workflow.yaml data via load_workflow_yaml — a workflow-data behavioral test
FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"
FAMILY_OF["scripts/workflow_lint.py"]="lint"
FAMILY_OF[":(glob)tests/test_workflow_lint*.py"]="lint"
FAMILY_OF["tests/test_autonomous_session_watch.py"]="lint"    # test_codex_outage_docstring_pass_count_lint_stays_green imports check_asw_docstring_pass_count from workflow_lint
FAMILY_OF["scripts/select_step9c_tests.py"]="lint"
FAMILY_OF["tests/test_select_step9c_tests.py"]="lint"
FAMILY_OF["tests/step9c_workflow_invariant_manifest.txt"]="lint"
FAMILY_OF[".claude/hooks"]="guard"
FAMILY_OF[":(glob)scripts/guard_*.sh"]="guard"
FAMILY_OF[":(glob)tests/test_guard_*.py"]="guard"
FAMILY_OF["tests/test_guard_lessons_edit.py"]="guard"
# Singletons: .claude/agents, .claude/agent-memory, .claude/rules, CLAUDE.md
# — each is its own family key (set below in the pass-1 loop by defaulting
# to its own path).

SPECS=".claude/agents .claude/agent-memory .claude/skills .claude/rules .claude/workflow.yaml CLAUDE.md scripts/workflow_lint.py scripts/select_step9c_tests.py .claude/hooks :(glob)scripts/guard_*.sh tests/test_guard_lessons_edit.py tests/test_workflow_yaml.py tests/test_autonomous_session_watch.py tests/test_select_step9c_tests.py tests/step9c_workflow_invariant_manifest.txt :(glob)tests/test_workflow_lint*.py :(glob)tests/test_guard_*.py :(glob)tests/test_issue_skill_*.py"
# Bounded freshness fetch (#1747 — the #1289/#1714 shape): local main can lag
# origin on the shared root; a failed fetch degrades to last-fetched
# origin/main — never a wedge, never a fallback to local main.
timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet || true
MB=$(git -C "$WT" merge-base HEAD origin/main)

# Pass 1: detect dirty family keys. A family is DIRTY if ANY member has
# branch-side commits (subject-scoped exclusion for prior spec-freshness
# commits, as in #1560).
declare -A DIRTY_FAMILIES
for f in $SPECS; do
  # Branch-side feature edits = commits since merge-base touching $f,
  # EXCLUDING prior spec-freshness sync commits (which legitimately
  # touch spec paths — without the exclusion, the first sync's own
  # commit would poison every later freshness check on the branch).
  # The exclusion matches the prescribed sync-subject SHAPE, not the
  # bare "spec-freshness" token, so a deliverable commit whose subject
  # names the mechanism is never misread as a sync commit (#1789).
  bs_commits=$(git -C "$WT" log --format='%H %s' "$MB"..HEAD -- "$f" \
    | awk 'index($0, "sync workflow-surface specs from") == 0')
  if [ -n "$bs_commits" ]; then
    fam="${FAMILY_OF[$f]:-$f}"    # default: singleton family = own path
    DIRTY_FAMILIES[$fam]=1
    # Print the offending commits so the orchestrator can decide whether
    # to reconcile (cherry-pick main's drift on top of the branch edits)
    # or whether the branch-side touch is a global revert/port that has
    # ALREADY landed on main — in which case the skip is a false alarm
    # and the orchestrator can drop those files from the skip set by
    # hand (e.g. `git -C "$WT" checkout origin/main -- .claude/agents/*.md`
    # after confirming the branch-side commit's content is a subset of
    # main's current state). Without these commit titles printed, the
    # operator cannot tell a legitimate branch deliverable (#535
    # incident) from a stale port/revert that needs no protection.
    echo "spec-freshness: $f carries branch-side feature edits — marking family '$fam' dirty; skipping blind sync for the whole family; reconcile manually."
    echo "  branch-side commits:"
    echo "$bs_commits" | sed 's/^/    /'
  fi
  # Uncommitted-dirt arm (#1972): an uncommitted worktree write under $f must
  # never be clobbered by the checkout below. Tracked-modified dirt (any
  # non-?? porcelain line — renames `R  a -> b` need no path parsing) always
  # marks the family dirty; an UNTRACKED (??) path marks it dirty ONLY when
  # the same path exists at origin/main — `git checkout <ref> -- <pathspec>`
  # DOES overwrite an untracked file whose path exists at the ref, and cannot
  # touch one absent from it (so fresh mid-round agent-memory files with no
  # main-side name collision never block the sync). A collapsed untracked dir
  # (`?? dir/`) cat-files the tree path with the slash stripped — a
  # main-existing tree marks dirty, the conservative direction. Fail-safe:
  # dirty -> status-quo staleness, never a clobber.
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
    DIRTY_FAMILIES[$fam]=1
    echo "spec-freshness: $f carries UNCOMMITTED changes the sync could clobber — marking family '$fam' dirty; skipping blind sync for the whole family (#1972)."
  fi
done

# Pass 2: filter SAFE_SPECS to items in a NON-dirty family.
SAFE_SPECS=""
for f in $SPECS; do
  fam="${FAMILY_OF[$f]:-$f}"
  if [ -z "${DIRTY_FAMILIES[$fam]}" ]; then
    SAFE_SPECS="$SAFE_SPECS $f"
  fi
  # else: skipped by family transitivity (message already printed in pass 1
  # for the offending member; skipped-siblings are covered by the family
  # membership declared above)
done

if [ -n "$SAFE_SPECS" ] && ! git -C "$WT" diff --quiet origin/main -- $SAFE_SPECS; then
  git -C "$WT" checkout origin/main -- $SAFE_SPECS    # surgical refresh: workflow surface only
  git -C "$WT" diff --quiet HEAD -- $SAFE_SPECS || \
    git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs from origin/main (spec-freshness)" -- $SAFE_SPECS
fi
# Observability echo (Decision 4, #1714): show the operator what changed at a
# glance. This is NOT a gate — family-atomic skip + git checkout's own semantics
# handle the 139-line-revert prevention (see plan §4.1 Decision 4 for the full
# rationale).
if [ -n "$SAFE_SPECS" ]; then
  echo "[step5a] synced from origin/main:"
  git -C "$WT" diff --stat HEAD^ HEAD -- $SAFE_SPECS 2>/dev/null || echo "  (no commit — no drift)"
fi

# Sibling-issue file freshness (#1972): per-FILE grain, scripts AND their
# covering tests as a PAIR. A gated test may import a sibling issue's
# scripts/issue<M>_*.py whose worktree copy predates a main-side fix (the
# #1768 r4/r5 class, ~40 min/incident); the sync commit below also puts the
# file into the selector's three-dot diff (fetched origin/main,
# merge-base...HEAD), newly mapping its covering tests/test_issue<M>_*.py —
# so the pair MUST move together (syncing the script alone runs a fork-era
# test against a fresh script, the #1824/#1860 half-sync class). Per-FILE
# grain is load-bearing: a :(glob) SPECS entry would be ONE singleton
# family, and every branch edits its OWN issue scripts/tests, so the
# glob-family would always be dirty — self-defeating. Only files with ZERO
# non-sync branch-side commits sync (a branch's own deliberate edits — incl.
# its own issue scripts/tests — are never touched); ANY uncommitted dirt on
# the file skips it (per-file grain makes the wide skip free); files absent
# on origin/main are skipped (never deleted). The commit subject carries the
# anchor phrase `sync workflow-surface specs from`, so the arm's own
# bs-check excludes its prior sync commits on later rounds, Guard 3 treats
# the synced files as imported-from-main, and the Step 10d verdict re-bind's
# A/M byte-identity probe passes (content == fetched origin/main).
SIBLING_SYNCED=()
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in scripts/issue<N>_*|tests/test_issue<N>_*) continue ;; esac   # own-issue carve-out (defense-in-depth)
  bs=$(git -C "$WT" log --format='%H %s' "$MB"..HEAD -- "$f" \
    | awk 'index($0, "sync workflow-surface specs from") == 0')
  [ -n "$bs" ] && continue                            # deliberate branch edit — protected
  if git -C "$WT" status --porcelain -- "$f" | grep -q .; then
    echo "spec-freshness: sibling file $f carries UNCOMMITTED changes — skipped (#1972)."
    continue
  fi
  if git -C "$WT" cat-file -e "origin/main:$f" 2>/dev/null; then
    git -C "$WT" checkout origin/main -- "$f" && SIBLING_SYNCED+=("$f")
  else
    echo "spec-freshness: sibling file $f absent on origin/main — skipped (never deleted; #1972)."
  fi
done < <(git -C "$WT" -c core.quotePath=false diff --name-only origin/main -- ':(glob)scripts/issue[0-9]*_*.py' ':(glob)tests/test_issue[0-9]*_*.py')
if [ "${#SIBLING_SYNCED[@]}" -gt 0 ] \
   && ! git -C "$WT" diff --quiet HEAD -- "${SIBLING_SYNCED[@]}"; then
  git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)" -- "${SIBLING_SYNCED[@]}"
fi
echo "[step5a] sibling-file sync: ${#SIBLING_SYNCED[@]} file(s)"
fi
```

The refresh touches ONLY the workflow surface (never experiment code).
Issue branches must not carry their own workflow-surface edits as a
rule (those go through their own filed workflow-fix `/issue --auto`
sessions + worktrees), with one
legitimate exception: a feature branch whose DELIVERABLE adds
workflow-surface entries — e.g. a new marker schema registered in
`workflow.yaml` rides its feature branch (#535). The per-file branch-side-edit guard
above skips exactly those files (warning the orchestrator to reconcile
them manually — typically by re-applying main's spec changes on top of
the branch's additions) while everything the branch never touched
still gets the blind sync. The conditional commit keeps the worktree
clean for the Step 10d merge guards. The warning prints the offending
branch-side commit titles so the orchestrator can tell a legitimate
branch deliverable (the #535 case) from a stale port/revert whose
content has already landed on main (in which case the orchestrator can
safely override the skip for those specific files with a manual
`git -C "$WT" checkout origin/main -- <paths>`).

**The sync scope is specs + the spec-coupled lint/guard family — do NOT
extend it further into `scripts/`, `tests/`, or `src/`.** The family
exception (#1560: `scripts/workflow_lint.py`, `.claude/hooks`,
`:(glob)tests/test_guard_*.py`, `:(glob)tests/test_workflow_lint*.py`;
#1883 adds `:(glob)tests/test_issue_skill_*.py`, the prose-pin tests
over `.claude/skills` content — the #1824 vintage skew; #1963 adds
`:(glob)scripts/guard_*.sh` — the guard-script implementations the
`:(glob)tests/test_guard_*.py` pins execute, PreToolUse hooks wired in
`.claude/settings.json`: syncing the tests without them half-syncs the
tree and red-flags main-green guard nodes on pure version skew, the
#1860/#1862 incidents; #1972 adds `scripts/select_step9c_tests.py` +
`tests/test_select_step9c_tests.py` +
`tests/step9c_workflow_invariant_manifest.txt` to the lint family — the
pin test importlib-loads the selector BY PATH from the worktree and its
case 6b pins `WORKFLOW_INVARIANT` set-equal to the manifest, so syncing
any strict subset is the same half-sync class (named residual: a
main-NEW invariant test file outside the synced globs can still red the
pin test's live-tree check until the branch rebases — same β-class) —
plus `.claude/agent-memory` as a singleton: always-appended memory
indexes the lint budget checks scan, protected by the uncommitted-dirt
arm + the branch-side-edit guard, never a clobber)
exists because those files execute FROM the worktree tree on four
surfaces — the Step 10d TG legs, worktree pytest / Step 9c, the hooks'
own-tree `workflow_lint` import, and the inline gate invoked in a
worktree — and their constants/budgets pair with the specs this sync
already refreshes: half-syncing manufactured the #1489/#1482/#1417 gate
blocks. The lint/guard family is deliberately closed up to ONE seam: its only src
imports are from the low-churn `explore_persona_space.workflow` module
(the linter's 2-symbol import at `workflow_lint.py:672-681`, plus
`tests/test_workflow_lint.py:96`'s 3-symbol
`MarkerEntry, WorkflowYaml, load_workflow_yaml`) — the accepted
residual: a synced family file ImportError-ing on that module means
branch-era `src/explore_persona_space/workflow.py` skew (rebase onto
origin/main, or cross-check at the repo root; the module is
long-stable). The skill-pin glob (`:(glob)tests/test_issue_skill_*.py`,
#1883) carries two additional accepted seams of the same β-class:
`tests/test_issue_skill_long_phase_heartbeat.py` imports the
scripts-side `autonomous_session_watch` + `tick_triage` modules
(L53-54), and `tests/test_issue_skill_trigger_dense_tag_adoption.py`
importlib-loads `scripts/select_step9c_tests.py` by path (same-vintage
as of #1972 — the selector rides the lint family) and text-pins a
literal in `src/explore_persona_space/backends/excerpt_digest.py` — the
src pin stays the seam; same remedy: a synced pin test failing on
branch-era `scripts/`/`src/`
skew means rebase onto origin/main, or cross-check at the repo root.
Family atomicity (#1714): within the spec-coupled
lint/guard family, the per-item branch-side-edit skip is transitive —
a branch-side edit on ANY family member widens the skip to the WHOLE
family (never narrows it). Three families are declared: workflow
(`.claude/workflow.yaml` + `.claude/skills` where the derived
`markers.md` and SKILL.md generated tables live, plus
`:(glob)tests/test_issue_skill_*.py` — the prose-pin tests over that
skills content, #1883), lint
(`scripts/workflow_lint.py` + `:(glob)tests/test_workflow_lint*.py`
plus the explicit importers `tests/test_workflow_yaml.py` and
`tests/test_autonomous_session_watch.py`), and guard (`.claude/hooks`
+ `:(glob)scripts/guard_*.sh` + `:(glob)tests/test_guard_*.py`
+ `tests/test_guard_lessons_edit.py`).
Everything else in SPECS is a singleton (its own family). Everything ELSE keeps the original rationale: workflow-
helper SCRIPTS are already resolved from the MAIN checkout (Step 0
§ worktree spec-freshness: `"$REPO_ROOT"/scripts/...`) — except the
guard-family `:(glob)scripts/guard_*.sh` implementations, synced +
executed from the worktree by their pin tests (#1963), and the Step 9c
selector `scripts/select_step9c_tests.py`, synced + importlib-loaded BY
PATH from the worktree by `tests/test_select_step9c_tests.py` (lint
family, #1972) — and blind-
syncing broader `tests/` is actively unsafe — main's newer workflow
tests pin behavior implemented in main's newer `scripts/` + `src/`
(e.g. `task_workflow.py`, `backends/`) that the branch predates, so a
partial code sync makes the worktree suite REDDER or breaks the
branch's own imports — and the per-path branch-side-edit guard would
skip broad `scripts/`/`tests/` cones wholesale anyway (nearly every
issue branch adds its own `scripts/issue<N>_*.py` + tests — which is
exactly why the #1972 sibling-issue arm is a bounded, per-FILE
exception rather than a glob family: never-branch-edited sibling
`scripts/issue<M>_*.py` + `tests/test_issue<M>_*.py` pairs sync
together under an own-issue carve-out and a per-file dirt skip, so a
branch's own deliverables are structurally out of reach). Operational
rule instead: a workflow test that FAILs inside a long-lived issue
worktree but PASSes at the repo root on `main` — **including a
collection-time ImportError from a `workflow_lint` / rules-pin
symbol** — is worktree-staleness, not this issue's breakage —
cross-check at the repo root before chasing it; the
Step 10d merge resolves it (#542). (A shared-infra
`src/` fix with fleet-wide blast radius — e.g. the #847 thread caps — gets a
LAUNCH-TIME fallback instead of a sync: the VM-side launch surfaces carry the
explicit thread-cap env prefix, Step 9 entry guard § "Detached VM-side long
compute phases"; #891. Do not extend this sync to `src/` allowlists — a synced
`env.py` would still miss torch-before-dotenv importers in-process, which the
launch prefix caps unconditionally.)

**Reference-lint staleness is handled LINT-side, never by widening this sync
(#1622 → #1672).** The synced specs may reference `scripts/` helpers that
landed on main after the branch point (e.g. `scripts/plan_patch.py`, #1631)
and are absent from the stale worktree tree; the sync commit's own pre-commit
hooks then run the (freshly-synced) `workflow_lint.py --check-references`
against the WORKTREE tree. Rather than syncing those helpers in — banned
above; nothing may execute `scripts/` copies from the worktree, except
the guard-family `:(glob)scripts/guard_*.sh` implementations their
synced pin tests execute (#1963) —
`check_script_references` / `check_skill_references` degrade exactly that
case to a `WARN:` on a non-main checkout (referenced target missing locally
but present at `main`/`origin/main`), so the sync commit passes on files the
round never touched. A hard reference FAIL inside a worktree therefore means
the target is missing on main too — a genuine dead reference introduced by
this round, fix it. Strictness is unchanged on the main checkout and in the
Step 10d landing-tree gate (a non-git tree probes nothing and keeps the hard
FAIL). (Named residual: a detached scratch-worktree merge commit — CLAUDE.md
§ Concurrent repo-root committers — also commits in the WARN regime; a
branch that deletes a still-referenced script is caught post-merge on main
by the strict main-tree lint.)

> **429 pacing at every ensemble fan-out (applies here, to the Step 9
> critic ensembles, and to /adversarial-planner Phase 2):** when MORE than
> two agent prompts go out at once (e.g. 3 critic lenses x 2 models), pause
> 5-10 s between Agent spawns (`sleep` is fine inside the dispatch Bash
> call, or send the spawns in 2 staggered messages). Same-second prompt
> bursts stacked onto the org-wide 4M input-tok/min cap caused 429 storms
> in 6+ sessions in one day.

Both reviewers see the same brief:

- `issue_number` — the task number (`<N>`)
- `target_marker_kind` — exactly one of `experiment-implementation` (for
  `experiment`) or `results` (for `infra` / `batch` / `analysis` /
  `survey`). The reviewers read the highest-version row with this kind
  from `events.jsonl` as the implementer's report.
- `revision_round` — 1-indexed integer. `1` on first review; loops up to
  `3`. The cap is **per reviewer** — reconcile invocations are free.
- `previous_critique_summaries` — one-line summaries of every prior
  `epm:code-review` AND `epm:code-review-codex` event on this task
  (empty on round 1). Lets each reviewer notice patterns.
- The diff vs `main`, the approved plan (via the `plans/plan.md`
  symlink), the existing codebase.

The Claude reviewer additionally receives:
- `worktree` path, `base` ref (typically fetched `origin/main` — #1289).

The Codex twin additionally receives:
- `worktree`, `base`, `plan_marker_path` (no `implementation_marker_path`
  — the composer fetches the marker from canonical main state and INLINES
  it; likewise, if the worktree plan is absent — child task cut from a
  parent issue branch, #550 r1 — or STALE — follow-up amendment plan
  postdating the branch cut, #546 follow-up r1 — the composer inlines
  the canonical plan, Step 2-pre-b) — see
  `.claude/agents/codex-code-reviewer.md`.

**Neutral gate vocabulary in EVERY brief — first-pass AND revision
rounds, every subagent brief, every step (#1398, #1413).** When ANY brief
this skill composes (planner,
implementer, experimenter, reviewer, analyzer — not only review rounds)
concerns a kill-gate / RLVR / guard / stop-criteria task, write the BRIEF
in neutral vocabulary from the first spawn: "halt gate", "stop criterion",
"termination predicate" — never the loaded kill-* phrasings in
brief/prompt text. The loaded terms stay in the ARTIFACTS themselves (code
identifiers, plan text, task bodies are never renamed); only the brief is
neutralized. This is the gate-vocabulary leg of CLAUDE.md § Spurious
usage-policy refusals rung (e) — first-pass, not a post-kill retry step
(#1336). Revision-round briefs carry the same neutral vocabulary
AND, on trigger-dense rounds, pass findings BY REFERENCE — see Step 5d,
§ File-only Codex verdict posting, and trigger-dense-review.md
§ Revision-round briefs.

**Trigger-dense (guard-surface) rounds — pre-materialize the excerpt file
BEFORE spawning (#1058/#1098).** When the round's diff or artifact under
review is trigger-dense per the `.claude/rules/trigger-dense-review.md`
recognition heuristic (guard/security hook scripts, destructive-command
fixtures, refusal/jailbreak corpora — recognition is DELEGATED to that
rule's "Fires when" block; do not re-derive it here), the orchestrator
pre-materializes the round's excerpt file and names it in BOTH reviewer
briefs — this arms the rule's discipline-3 "orchestrator-provided
pre-materialized excerpt files with stated read budgets" leg. Build is
cheap and mechanical — the trigger-dense hunks only, with file/line
anchors:

```bash
# Scope the pathspec to the trigger-dense paths actually touched
# (`git -C "$WT" diff --name-only origin/main...HEAD` first):
git -C "$WT" diff origin/main...HEAD -- .claude/hooks/ 'scripts/guard_*.sh' \
  'tests/*guard*' > /tmp/issue-<N>-r<round>-excerpts-<slug>.md
```

(For a non-diff trigger-dense artifact — a corpus or fixture file the
round must adjudicate — extract grep-anchored ≤~120-line windows into the
same file instead. Harmful BANK items stay digest-only per
`guard_harmful_bank_read.sh` — never copy bank item text into an excerpt
file. On round >1, round-scope the diff first when the branch diff is
over budget per `.claude/rules/diff-size-budget.md`.) Then add one line
to BOTH briefs (and keep it in any re-spawn brief):

`excerpt_file: /tmp/issue-<N>-r<round>-excerpts-<slug>.md — read this
INSTEAD of wholesale-reading the touched trigger-dense files; direct
reads of the originals capped at ~120-line grep-anchored windows per
trigger-dense-review.md.`

The excerpt file bounds READ volume; it does not sanitize content —
reviewers still apply discipline 1 (findings by file:line reference,
never gated literals in generated text). The same briefs (and any
re-spawn brief) ALSO carry the discipline-4 return-text contract as
one line: `return_text: verdict + marker pointer + counts only —
no findings recap (trigger-dense-review.md discipline 4)`.
Non-trigger-dense rounds: skip entirely — no excerpt file, neither
brief line.
Verdict COLLECTION on trigger-dense rounds is file-only — see § File-only
Codex verdict posting (before Step 5c).

Neither sees the implementer's reasoning — independence is load-bearing.
Dispatch in a SINGLE `Agent(...)`-call message with both spawned
`run_in_background=true` so they execute concurrently.

The Claude reviewer posts `epm:code-review v<n>` (PASS / CONCERNS /
FAIL). The Codex wrapper posts `epm:code-review-codex v<n>` (same
schema). Codex never sees `GH_TOKEN` — both wrappers post via
`task.py post-marker`.

**End-to-end smoke gate (experiment tasks).** A code-review PASS for an
`experiment` task is NOT valid on a script that was only `--help`'d or
import-checked. The reviewer MUST confirm the implementer smoke-ran
EACH PHASE of the experiment pipeline ONCE on a tiny real slice — not
just training or data-gen. "Phase" = any distinct entrypoint the
pipeline executes end-to-end (typical experiments: data-gen, training,
eval; some add separate analysis / upload steps). Eval rigs especially
must be exercised end-to-end on a tiny slice (1 seed, the minimum
contexts / cells, the base model or a tiny throwaway checkpoint) — a
never-before-run eval script that was only import-checked or that
relied on the training script's smoke is a known regression source:
shallow latent bugs (corpus-size floors, missing helpers, generator-
reuse, sentinel filters, aggregation-tuple unpacks) surface one-per-
run at the real eval phase, each costing a full pod cycle (#408). For each phase, the
implementer records a sub-section under the `## Smoke run` heading
in its `epm:experiment-implementation` report — recommended layout
`### <phase-name>` (e.g. `### data-gen`, `### training`, `### eval`)
with the exact command, the slice size (how it was kept tiny), the
exit code (must be `0`), and a one-line digest of the produced
artifact (path + shape / row count). If the `## Smoke run` section is
absent, OR any phase the pipeline actually executes is missing a
sub-section, OR any sub-section shows only `--help` / `import` /
`--dry-run` evidence (or exits non-zero, or carries no artifact
digest), the reviewer posts `FAIL` with blocker `smoke-run-missing`
— it does NOT PASS on unproven code, and a never-before-run eval rig
without an end-to-end smoke is the canonical missing-phase case. But
if every phase IS present (command + exit 0 + artifact digest) and
only the *formatting* is imperfect, that is a `CONCERNS`, not a FAIL
— and Step 5c-bis strips any mechanical-contract-only FAIL once the
orchestrator verifies the evidence is genuinely present, so cosmetic
gripes about present evidence never bounce the implementer or consume a
review round. Code-only tasks (`infra` / `batch` / `analysis` /
`survey`) keep the existing test-verdict gate (Step 9c) and are
exempt from this smoke gate. Smoke commands that write under
`eval_results/` or `figures/` also carry the output-path hygiene
disposition per experiment-implementer.md
§ "Smoke outputs never overwrite committed artifacts" (scratch-dir
redirect preferred; restore-after-smoke + an empty
`git status --porcelain -- eval_results/ figures/` as the fallback);
the reviewer treats visible clobber of a committed artifact as a
substantive Critical (code-reviewer.md Step 0.6), never a strippable
mechanical blocker.

**5b. Read both markers from `events.jsonl`.**

```bash
# After both Agent tasks complete — ONE fetch, parse twice in-memory.
events_json=$(uv run python scripts/task.py view <N> --json | jq '.events')
claude_marker=$(echo "$events_json" | jq '... epm:code-review v<n> ...')
codex_marker=$(echo "$events_json" | jq '... epm:code-review-codex v<n> ...')
```

Parse each marker's `**Verdict:**` line. Acceptable values: `PASS`,
`CONCERNS`, `FAIL`. PASS-class = {PASS, CONCERNS}; FAIL-class = {FAIL}.

**Durable-verdict-first rule (fires at EVERY ensemble verdict collection:
5b here, Step 9a, Step 9a-bis, Step 9b VC, and any reconciler read).**
An Agent-tool completion result that reports an error for a reviewer /
critic / reconciler subagent — autocompact thrash death, tool-use crash,
or a garbage/empty return — is NOT, by itself, a no-show. These agents'
deliverable is DURABLE state (a marker on events.jsonl, or a written
output file), and the final summary turn regularly dies AFTER the durable
post succeeded (#810 r4). BEFORE invoking any no-show fallback or
single-reviewer decision:

1. Re-read canonical task state (`uv run python scripts/task.py view <N>
   --json`) for the round's expected verdict marker at the CURRENT
   version — `epm:code-review[-codex] v<n>`, `epm:interp-critique[-codex]
   v<n>`, `epm:clean-result-critique[-codex] v<n>`,
   `epm:followup-value-critique[-codex]`, or `epm:review-reconcile v<n>`.
   The mechanical form of this check is
   `task_workflow.ensemble_verdicts_present` (precedent:
   `stage_dispatch_should_skip` for the dispatch side) — run it, do not
   eyeball the events scan:

   ```bash
   uv run python - <<'PY'
   import json
   from explore_persona_space.task_workflow import ensemble_verdicts_present, list_events
   print(json.dumps(ensemble_verdicts_present(
       list_events(<N>), ["epm:code-review", "epm:code-review-codex"], <n>)))
   PY
   ```

   (Substitute the site's marker kinds; for a reconciler read pass
   `reconcile_role="<role under adjudication>"` so a same-round reconcile
   for a DIFFERENT role never satisfies the check.) `present: false` →
   proceed to item 2; `present: true, verdict: null` → the marker EXISTS
   but is malformed — item 3's malformed-output handling, NEVER a
   no-show; `present: true` with a verdict token → the reviewer RETURNED.
   Before acting on a returned verdict token, confirm the adopted note's
   head sentinel names THIS round (the predicate already treats the
   sentinel as authoritative over the drift-prone `version` field; the
   confirm is the cheap orchestrator-side double-check against a
   stale-round adoption).
2. If no marker: check the role's durable output file — the EXACT
   `--output-file` path this round's dispatch config named
   (role+round-specific conventions:
   `/tmp/codex-code-reviewer-<N>-r<round>-output.md`,
   `/tmp/codex-interp-critic-<N>-r<round>-output.md`,
   `/tmp/codex-clean-result-critic-<N>-r<round>-output.md`,
   `/tmp/codex-followup-critic-<N>-output.md`; NEVER a guessed generic
   path). The file counts as a durable verdict ONLY if BOTH: (i) it
   carries the role's expected marker start/end tags at the CURRENT
   round version, AND (ii) it is round-fresh — a current-round
   `epm:codex-task-completed` marker exists for this dispatch, OR the
   file mtime postdates this round's `stage-dispatch` breadcrumb /
   `epm:codex-task-spawned`. A file failing either test is NOT a durable
   verdict — a conforming-looking file from a PRIOR round is the trap
   this clause exists to block.
3. If a durable verdict exists and CONFORMS (expected marker kind +
   current version + a parseable `**Verdict:**` line), the reviewer
   RETURNED: use the durable verdict and apply the normal ensemble rule
   — reconciler on disagreement, never a unilateral decision. A
   truncated file (a FAIL-class `**Verdict:**` line with no blocker
   body) is MALFORMED, not a verdict — route it to the role's
   malformed-output handling, never adopt it. Precedence when signals
   coexist: a current-round posted verdict MARKER wins over everything;
   a current-round posted `epm:failure` from the wrapper wins over a
   bare conforming FILE (the wrapper inspected its own output and
   judged it malformed); a conforming round-fresh file wins over
   nothing.
4. Only when NO durable verdict exists does the role's no-show handling
   fire. For a Codex twin: the Step 5d fallback (single-Claude
   decision), exactly as if `epm:failure` had been posted. For a CLAUDE
   reviewer/critic: there is NO fallback — first diagnose the death
   (e.g. an over-budget diff per `.claude/rules/diff-size-budget.md`;
   thin the brief accordingly), then re-spawn it ONCE per
   role+round+version — the re-spawn posts at the SAME `v<n>` and does
   NOT increment the per-reviewer round counter (a 429-kill is already
   covered by the SubagentStop retry rule and consumes the same
   allowance). If the re-spawn ALSO ends with no durable verdict, fail
   LOUD: interactive — surface to the user; autonomous — post
   `epm:failure v1` (`failure_class: infra`, reason:
   reviewer no durable verdict after bounded re-spawn), set
   `status:blocked`, PushNotification, CRON-TEARDOWN. NEVER adopt a
   unilateral decision from the surviving reviewer. (When the fallback is
   inline composition rather than a Codex twin's decision — sanctioned only
   for a workflow-fix task fixing this very thrash mode, or the refusal
   rung (c) sibling — post one `epm:progress` note with the FIXED leading
   token `[epm-inline-fallback] role=<role> round=<n> reason=<one-line>`
   (single line, greppable; mirrors the `[long-phase-heartbeat]` /
   `followup-parked-by-cap` / `merge-hold-candidate` durable-marker
   convention). This makes the pipeline's collapsed adversarial-review
   independence visible on the dashboard + /daily sweep, #2062.)

**Autocompact-thrash respawn recipe (refines item 4's "first diagnose
the death" for ANY thrash-killed subagent — reviewer/critic per item 4,
and equally an implementer / fact-checker / analyzer re-spawn).** Check
the dead spawn's transcript/result for an OVERSIZED tool result (a
multi-hundred-KB diff or file read). If ONE EXISTS, the read-side fix
applies: bound the read / thin the brief per
`.claude/rules/diff-size-budget.md`. If NONE exists, the pressure is
FIXED OVERHEAD on the subagent window (spec + CLAUDE.md import tree +
MCP schemas + the brief) — or accumulated read VOLUME no single-read
bound addresses — and re-tightening read bounds does NOT help: respawn
instead with (i) MICRO-SCOPED work — split the role's work into the
smallest self-contained unit (#1090 split one implementer build into
sequential rounds A/B — round A returned a commit manifest with NO
implementation marker; round B posted the marker after the full
smoke) — and (ii) the DEFAULT session model — do NOT pin a smaller
model as a thrash fix (#1090 forensics, events.jsonl L247: "transcript
forensics show NO oversized tool result (max 15KB line): the thrash is
FIXED-OVERHEAD pressure on the subagent window, not read indiscipline";
"read-bounded brief did not help"; "both default-model spawns today
compacted successfully; 3/6 sonnet spawns thrashed"). And (iii) when the
DEFAULT-model micro-scoped respawn ITSELF thrashes, escalate ONCE (same
`v<n>`, no counter increment; the lean twin inherits the same
one-bounded-respawn budget as item 4 above) to the role's LEAN TWIN
(`.claude/agents/<role>-lean.md`, or `~/.claude/agents/analyzer-lean.md`)
with the same micro-scoped brief — the twin drops MCP schemas + `skills:`
declarations and reads the full sibling spec by reference, cutting
fixed-overhead ~138K tokens (#2062). Available for: `analyzer`, `planner`
(also covers the `planner`-typed fact-checker spawn at
`.claude/skills/adversarial-planner/SKILL.md:867`), `critic`,
`experiment-implementer`, `code-reviewer`, `consistency-checker`. If the
lean-twin respawn ALSO ends with no durable verdict, fall through to
item 4's fail-loud terminal — never an unbounded lean-twin retry loop.
Multi-unit splits
apply to roles whose deliverable DECOMPOSES (an implementer or
fact-checker build); a single-verdict reviewer/critic re-spawn stays
ONE spawn, micro-scoped by brief. Per-subagent model pins remain
prompt-cache-safe and legitimate for OTHER reasons (the CLAUDE.md
refusal rung (b2) sonnet pin) — they are just not a thrash remedy. The
micro-scoped respawn IS item 4's one bounded re-spawn where item 4
applies (same `v<n>`, no round-counter increment); for a multi-unit
split, the units run sequentially within that same round (#1090: "Same
round counter (round 1 continues; re-spawns do not increment)"). The
dispatch-time twin of this split lives at Step 4b ("Pre-split
multi-deliverable builds at dispatch", #1810) — a KNOWN
multi-deliverable build is pre-split BEFORE the first spawn; this
recipe stays the recovery-side backstop for unforeseen deaths.

The existing marker-keyed no-show path — the Codex wrapper POSTING
`epm:failure v<m>` (`failure_class: codex-output-malformed` or `infra`)
— is itself durable state and is UNCHANGED: that marker IS a confirmed
no-show. This rule governs only the Agent-tool-RESULT-keyed inference.
(The dispatch-side sibling is the Step 9 pre-dispatch dedup /
`stage_dispatch_should_skip`; the resume table is already
durable-marker-keyed. This rule closes the live verdict-collection gap
between them.)

**File-only Codex verdict posting on trigger-dense rounds (fires at EVERY
marker-mode Codex verdict collection: 5b/5c here, Step 9a, Step 9a-bis,
Step 9b VC; #1275).** When the round is trigger-dense per the
`.claude/rules/trigger-dense-review.md` "Fires when" heuristic (same
recognition the Step 5a excerpt-file paragraph delegates to — do not
re-derive it here), the orchestrator posts the Codex twin's verdict
marker from its output file WITHOUT paging the findings-bearing body
into context — `post-marker --file` needs no full read. This is the
orchestrator-side sibling of discipline 4 (#1252 covered the reviewer's
return text; #1152's wedge shape applies equally to a wholesale
orchestrator read of the same findings). Mechanics (the composer's
return block already names the exact start/end tags and output path):

```bash
OUT=/tmp/codex-<role>-<N>-r<round>-output.md   # the EXACT dispatched --output-file path
MB=/tmp/issue-<N>-<kind>-r<n>-marker.md        # the extracted marker block
# 1. Marker block FIRST — mechanical tag-window extraction (tags verbatim from
#    the composer's "Marker start tag:" / "Marker end tag:" lines; LINE-START
#    anchors so a mid-prose quoted tag mention can never open/close the window):
sed -n '/^<!-- epm:<kind> v<n>/,/^<!-- \/epm:<kind> -->/p' "$OUT" > "$MB"
# 2. Gate: end tag present + under the 50,000-char note cap. A missing end
#    tag or empty extraction = MALFORMED output -> the site's existing
#    stricter-retry re-dispatch (cap 2), never a Read to "see what happened".
grep -q '^<!-- \/epm:<kind> -->' "$MB"
wc -c < "$MB"   # >=50000 -> the existing artifacts-file oversize fallback
# 3. Decision inputs for the ensemble tables — grep the EXTRACTED block (its
#    verdict line is the authoritative one; grepping "$OUT" would let a
#    pre-block template echo win -m1), never Read:
grep -m1 '^\*\*Verdict' "$MB"            # single-verdict sites (5c / 9a / 9a-bis)
grep -E '^### Proposal|^\*\*Verdict' "$MB"   # 9b VC only: per-proposal verdicts — no -m1
grep -m1 '^\*\*Blocker tags:' "$MB" || true  # sites that carry it (5c-bis / 9a-bis strips)
# 4. Post without reading (OMIT --version: it auto-derives max+1; the round
#    lives in the extracted block's head sentinel the sed extraction keys on):
uv run python scripts/task.py post-marker <N> epm:<kind> \
  --file "$MB"
```

The Step 5b item-2/3 durable-verdict probes (start/end tags at the
current round version; parseable `**Verdict:**` line; round-freshness
via marker/mtime) are grep/`stat` probes — on a trigger-dense round run
them mechanically for Claude reviewer output files too, never via
`Read`. When findings DETAIL is genuinely needed downstream (an
implementer bounce brief's union-blocker list, a reconciler brief), pass
the findings BY REFERENCE — the posted marker kind + version on
events.jsonl and/or the dispatched output-file path under /tmp — so the subagent reads
them itself with windowed grep-anchored reads (trigger-dense-review.md
discipline 3); do not inline verdict bodies into briefs on such rounds.
EXEMPT: in-context sites — adversarial-planner Phase 2 lens critics and
any composer returning `Posting mode: in-context` — where the verdict
body IS the deliverable merged into context (discipline 1 bounds its
content). Non-trigger-dense rounds: unchanged — reading the output file
remains fine.

**5c. Apply ensemble decision rule.**

| Claude verdict | Codex verdict | Action |
|---|---|---|
| PASS-class | PASS-class | **Agree.** `final_verdict = PASS`. CONCERNS bullets from either reviewer surface to the implementer as opportunistic suggestions; do not block. |
| FAIL | FAIL — overlapping blockers | **Agree.** `final_verdict = FAIL`. Bounce to implementer (one round). |
| FAIL | FAIL — disjoint blockers | **Union, no reconciler.** Build a combined blocker list (Claude's blockers ∪ Codex's blockers) — INCLUDING every `### Bug-class sweep: <class>` sibling enumeration from either verdict — and pass it to the implementer in the next-round brief (trigger-dense round: by reference per § File-only Codex verdict posting). No new marker — both `epm:code-review v<n>` and `epm:code-review-codex v<n>` already exist on the task. `final_verdict = FAIL`. Bounce (one round). |
| PASS-class | FAIL (or vice versa) | **Disagreement.** Spawn `reconciler` agent (Claude, fresh context). Brief: role=`code-reviewer`, task=N, round=n, both event bodies (trigger-dense round: BY REFERENCE — marker kind+version / output-file paths, per § File-only Codex verdict posting; the reconciler reads them itself with windowed reads), diff path (+ the Step 5a excerpt-file path + read budget on a trigger-dense round). Reconciler reads both verdicts + the artifact, posts `epm:review-reconcile v<n>` with binding PASS or FAIL. `final_verdict = reconciler's verdict`. |

The reconciler may NOT add findings beyond what either reviewer raised —
its job is adjudication only. Round counter does NOT increment for
reconciler invocations.

When BOTH reviewers returned disagreeing durable verdicts, adopting the
MORE SEVERE verdict WITHOUT spawning the reconciler is
UNSANCTIONED at every doubled site — even when the flagged residual is
mechanically verifiable (#825 skipped the reconciler on exactly that
rationale) — because a true residual does not determine severity (the
reconciler may legitimately side PASS on a true-but-not-verdict-changing
finding), and the shortcut trades a FREE adjudication (reconcile rounds
don't count) for a revision round that DOES count against the cap-5 and
itself costs ≥3 spawns (analyzer + both critics) vs the reconciler's
one, while leaving a possibly over-strict reviewer unadjudicated. The
documented adopt-more-severe last-resort fail-safe (a spawned reconciler
errors, is re-spawned once, and still returns no parseable verdict)
belongs to the `/adversarial-planner` § Durable-output-first IN-CONTEXT
Phase-2 reconciler ONLY — at the marker-mode sites here a twice-dead
reconciler fails LOUD per the Step 5b durable-verdict-first rule
(item 4), never adopt-more-severe. The Codex no-show fallback
(single-Claude decision on confirmed no-show) is a different, sanctioned
path — it adjudicates nothing and adopts no "more severe of two".

**5c-bis. Mechanical-contract-only FAIL strip (anti-gate-hopping).**

A FAIL is *mechanical-contract-only* when its `**Blocker tags:**` line
(reviewer Step 7 template) is a non-empty subset of {`marker-shape` (Steps
0.5 / 0.55), `smoke-run-missing` (Step 0.6), `git-provenance` (Step 0.9)} and does
NOT contain `substantive`
(any code / plan / test / security finding). The `**Blocker tags:**` line is
the parse target; if a legacy verdict omits it, fall back to reading the
Critical-section prose for the same tag strings. Apply this strip BEFORE the
Step 5c rule whenever a reviewer's verdict is FAIL. The
orchestrator does its own cheap, mechanical check of the highest-version
implementer marker (`epm:experiment-implementation` / `epm:results`) in
**canonical task state** — `uv run python scripts/task.py view <N> --json`,
the main-branch `events.jsonl`, NOT a possibly-stale worktree copy a reviewer
may have read. (A reviewer FAILing on "marker missing" while reading a stale
worktree `events.jsonl` — before the implementation marker was pulled in — is
the most common false absence; the canonical read is what catches it.) No LLM
judgment, just structural presence:

- **marker-shape:** three sub-recipes, keyed PER BLOCKER on the blocker body
  (a conforming Step 0.55 blocker names exactly ONE marker kind,
  `epm:smoke-architecture-check`; a conforming Step 4.6 presence blocker
  names `Gate-scope check` ONLY — never a combined 0.5 + 0.55 + 4.6
  blocker).
  When the blocker names `epm:smoke-architecture-check` (Step 0.55): a
  separate `epm:smoke-architecture-check` events row exists in canonical task
  state with a `verdict:` line matching `PASS_UNIFIED` | `PASS_CANARY
  canary_cell=<id>` | `PASS_PARTIAL arms_stubbed=<comma-list>` |
  `FAIL_NO_CANARY` — present + parseable → STRIP (a stale-worktree false
  absence); absent or verdict-less → leave the FAIL in place (the gate is
  doing its job; do NOT check the implementation marker's H3s for this
  sub-case — they can be conforming while the separate row is missing, which
  is exactly #811). **Discriminator (#1692):** the strip fires
  ONLY when the blocker names ABSENCE (the marker is missing / verdict-less)
  and the canonical marker is actually PRESENT with a valid verdict — the
  blocker body then reads like "no `epm:smoke-architecture-check` events
  row" / "marker missing" / "verdict-less". A SHAPE-VIOLATION blocker
  (marker present, verdict parseable, but internal-shape inconsistent —
  e.g. "PASS_UNIFIED verdict but arm foo reads FALLBACK", "per-arm-resolution
  row missing for plan-named arm bar", "import-resolution shape unrecognized")
  is `substantive`-adjacent: the strip does NOT fire and the FAIL stands.
  Distinguish by the blocker body phrasing (absence vocabulary → strip
  when marker present; verdict-vs-rows / row-missing / import-shape vocabulary
  → leave in place).
  When the blocker names `Gate-scope check` (Step 4.6 presence): the `(c)`
  section of the highest-version `epm:results` marker in canonical task
  state carries a `Gate-scope check` line — present → STRIP (a
  stale-worktree false absence; the strip verifies PRESENCE ONLY — a
  diff-consistency finding is `substantive` per Step 4.6 and never
  reaches this recipe); absent → leave the FAIL in place (the gate is
  doing its job). Otherwise (the Step 0.5 default):
  all four H3 sections `(a)`–`(d)` present with non-empty content AND `(c)`
  carries at least one fenced command.
- **smoke-run-missing:** a `## Smoke run` section is present, and EVERY phase
  the pipeline actually executes (typically data-gen, training, eval) has its
  own sub-section with a command, exit code `0`, and an artifact digest. A
  `## Smoke run` section that covers only one phase (e.g. training) while the
  pipeline also runs a separate eval rig is genuinely absent for the missing
  phase — leave the FAIL in place.
- **git-provenance:** the orchestrator reads the blocker's
  `**Git-provenance subclass:**` line and runs the matching read-only git
  probe from repo root (or against the branch ref `issue-<N>`, never by
  switching the repo-root branch — CLAUDE.md hard rule):
  - `pre-existing-on-trunk` → `git show main:<path>` resolves AND the round's
    own commit range (`git show <round-sha>~1..<round-sha> -- <path>`, or the
    implementer report's `<parent>..HEAD`) does NOT touch the flagged lines →
    the violation is on trunk, not from this round → STRIP.
  - `stale-main-or-worktree` → `git log --oneline main..issue-<N> -- <path>`
    returns zero non-merge commits (branch never touched the file) → the
    finding is a stale-branch artifact → STRIP.
  - `cumulative-main-head-diff` → the flagged line is unchanged in the round's
    OWN range (`git show <round-sha>~1..<round-sha> -- <path>` /
    `<parent>..HEAD`) even though it appears in `origin/main...HEAD` → out of round
    scope → STRIP.
  In ALL THREE: the strip fires ONLY when the git probe CONFIRMS the finding is
  not from this round's diff. If the probe shows the round's own range DID touch
  the flagged lines (git says the round introduced it), the strip does NOT fire
  — leave the FAIL in place and apply the normal Step 5c rule. This is
  evidence-based, never a blanket ignore. Merge-base errors on a sparse/shallow
  worktree (`fatal: origin/main...HEAD: no merge base`) are a checkout artifact — fall
  back to the two-dot / round-SHA range per code-reviewer.md Step 0; a "no merge
  base" error is never itself grounds to strip OR to FAIL.

Then:

1. **Artifact genuinely absent / non-conforming** → the gate is doing its
   job. Leave the FAIL as-is and apply the normal Step 5c rule.
2. **Artifact present + conforming** → the mechanical blocker is a false
   positive on cosmetics. STRIP it from that reviewer's effective blocker set,
   then apply Step 5c to the REMAINING (substantive) blockers from both
   reviewers:
   - No substantive blockers remain from either reviewer → `final_verdict =
     PASS`. Log to chat as one line: `mechanical-contract-only FAIL stripped —
     orchestrator verified <artifact> present + conforming; no substantive
     findings → PASS.`
   - Substantive blockers remain → normal Step 5c FAIL / union / reconciler on
     those only.

This is bounded: the orchestrator may strip ONLY a mechanically-verifiable
contract blocker (it is checking a structural fact, not overriding a
code-substance judgment) — for `git-provenance` the "structural fact" is the
read-only git probe confirming the flagged state is NOT introduced by the
round's diff (a git-history fact, same bounding logic as
marker-shape/smoke-run-missing), never a code-substance judgment. It directly
closes the gate-hopping failure mode —
a reviewer that FAILs round after round on the *presentation* of evidence the
marker demonstrably contains (e.g. round 1 marker-shape, round 2 smoke-digest
formatting, never reviewing the code) can no longer bounce the implementer or
consume a cap-5 round (the strategy pivot is retired; the strip still prevents
the round counter from incrementing). The round counter does NOT increment
for a strip. The clean-result-critique loop (Step 9a-bis) carries the same
strip for *presentation-only* verifier FAILs (MDX prose, caption shape,
cherry-label phrasing) — a clean-result FAIL backed only by presentation
nits is likewise stripped + patched inline rather than consuming a REVISE
round.

**5c-ter. Binding-concerns post-strip check (composed onto 5c-bis by task #455).**

After Step 5c-bis has stripped any mechanical-contract-only FAILs, AND
the per-reviewer verdicts have been resolved by Step 5c, run a final
binding-concerns check BEFORE advancing on `final_verdict == PASS`:

```bash
open_concerns=$(uv run python scripts/task.py list-concerns <N> --open-only --json)
```

If `open_concerns` is empty: advance per Step 5d as usual (the
historical PASS path is unchanged).

If `open_concerns` is non-empty AND `final_verdict == PASS`, iterate
per concern_id:

- **severity=NIT** → opportunistic, never blocks. Skip.
- **severity=CONCERN** → either:
  1. The current implementer round demonstrably addressed it AND the
     reviewer's verdict body (or the orchestrator's own diff inspection)
     confirms — call `task.py address-concern <N> --concern-id <id>
     --by code-reviewer --round <n>` (recording verification) and
     advance; OR
  2. **Interactive mode only:** raise inline `AskUserQuestion` <!-- gate: gates.concern_deferral_request --> proposing deferral. On user
     agreement run `task.py defer-concern <N> --concern-id <id> --by
     user --rationale "..."` (≥40 chars, not boilerplate) and advance;
     on user refusal bounce to the implementer with a brief targeting
     that concern (round counter increments).
  3. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the
     deferral ask AND never print the per-concern options as text. Auto-
     resolve per § Autonomous session behavior →
     `concern_deferral_request`: bounce to the implementer for one more
     round targeting the open CONCERN(s). State
     `Decision: bounce to implementer (concern_id=<id>) — autonomous
     mode never defers` AND EXECUTE the bounce in this same turn (spawn
     the implementer agent with a brief targeting the concern_id); do
     NOT state the Decision and then end the turn.
- **severity=BLOCKER** → either address (option 1 above) OR apply the
  cap-hit rule per `pivot_criteria.code_review_ensemble_cap_5_surface`
  (at cap-5: strip → all-stripped PASS+continue OR surface a substantive
  residual). BLOCKERs CANNOT route to the deferral gate. If it cannot be
  addressed and the residual is substantive, post `epm:failure v1
  failure_class: code` referencing the concern_id and set status:blocked
  (halt_criteria id=6 `concern_unresolved`).

Multiple open CONCERNS may batch into ONE `AskUserQuestion` call <!-- gate: gates.concern_deferral_request --> <!-- autonomous-mode: skip --> with
one option per concern_id plus a free-text rationale box per concern.
(Interactive mode only — autonomous mode bounces to implementer per
the per-concern rule above; the batch ask is never raised.)

This step does NOT override 5c-bis — mechanical-contract-only FAILs
still strip and cosmetic gripes about present evidence still don't
bounce the implementer. The check operates on a different signal
(concerns.jsonl persisted via `task.py raise-concern` — NOTE the
`--summary` arg is capped at 200 chars — the CLI truncates longer text
at a word boundary with a loud warning (programmatic `task_workflow`
callers still get ValueError); put detail in `--evidence`) and gates
auto-advance ON TOP of the existing flow. The same subroutine fires at
Step 9a (interp ensemble) and Step 9a-bis (clean-result ensemble) with
the same logic.

**5c-quater. Round-boundary durable-decision duty (#1855).**

Fires at EVERY review-round boundary — here at Step 5, and identically
at the Step 9a / 9a-bis analyzer↔critic rounds (this subsection is the
canonical text; those loops reference it). The moment a round's
ensemble decision is RESOLVED (final_verdict computed, a 5c-bis /
9a-bis strip applied, a 5c-ter concern picked for direct address), land
it durably — BEFORE dispatching the next round's subagents and BEFORE
beginning any orchestrator-applied inline fix:

1. **Post the decision as one `epm:progress` note** naming the resolved
   action + its source by reference — verdict marker kind+version,
   concern_id, and (for a prescribed fix) the target `file:line` + a
   one-line description of the prescribed change. One line; verdict
   bodies stay by reference (trigger-dense discipline unchanged; reuse
   `epm:progress`, never a new marker kind).
2. **Commit any uncommitted worktree edits from the just-completed
   round** by explicit path (`git commit -m <msg> -- <paths>` — the
   pathspec-limited form, never `git add -A`) before starting new
   context-expensive work.

Why unconditional (no headroom predicate): a session cannot introspect
its own context headroom (the #1338 lesson, asserted at the
residual-conflict dispatch and the resume section), and after even ONE
`Prompt is too long` API error no in-session recovery is possible —
every subsequent turn fails identically, so nothing can be landed
post-hoc; the watcher's context-ceiling wedge lane (#1453)
force-respawns a successor whose ONLY view of the round is the durable
trail. (#1776) Cost
of the duty: one marker + one commit per round.
This is the WRITE-side sibling of the Step 5b durable-verdict-first
rule (which recovers a dead reviewer's posted verdict); together they
make a round boundary death-cost-zero in both directions.

**5d. Loop on FAIL using `final_verdict`.**

- **`final_verdict == PASS`**:
  - `experiment` -> stay at status `running` (entering the workload
    sub-phase), proceed to Step 6.
  - `infra` / `batch` / `analysis` / `survey` -> skip pod phase, move
    status directly to `reviewing` (the inline test-verdict gate at
    Step 9c runs from there).
- **`final_verdict == FAIL` + revision_round<5** -> stay at status
  `running` (implementing sub-phase). Re-spawn the implementer with
  BOTH event bodies (Claude + Codex) AND the reconcile event (if
  present) as part of the brief (trigger-dense round: BY REFERENCE —
  marker kind+version / output-file paths, per
  § File-only Codex verdict posting; never inline the verdict bodies).
  **When either reviewer verdict (or the
  disjoint-blocker union) contains a `### Bug-class sweep: <class>`
  enumeration, thread the FULL sibling list — every enumerated
  `file:LINE`, not just the top finding — into the implementer's
  punch-list brief, so the round-N+1 edit is class-scoped and the
  implementer's class-hardening carve-out (experiment-implementer.md
  revision-round rule) fires on the whole class.** Implementer posts
  v<n+1>; loop back to 5a with `revision_round = n+1`.
- **`final_verdict == FAIL` + revision_round>=5** -> **CAP-HIT:
  strip-then-continue-or-surface** (replaces the retired cap-3 strategy
  pivot; see CLAUDE.md "STATE-TO-`blocked` criteria" and workflow.yaml
  § pivot_criteria.code_review_ensemble_cap_5_surface). At round 5 (the
  cap) with a non-PASS ensemble verdict, the orchestrator:
  1. **Applies the FULL Step 5c-bis strip once more** — the
     mechanical-contract-only set {`marker-shape`, `smoke-run-missing`,
     `git-provenance`}, evidence-based as always (git-provenance runs the
     read-only git probe matching the blocker's declared subclass).
  2. **If ALL residual blockers are stripped** (false-positive /
     mechanical / git-provenance) → treat as PASS and CONTINUE (proceed
     per the `final_verdict == PASS` branch above). Log one chat line +
     post an `epm:progress` note recording the cap-5-strip-continue
     outcome (which blockers were stripped and by what verification).
  3. **If ANY substantive residual remains** (a real finding the strip
     cannot verify away — silent-failure, upload-path/artifact-loss,
     missing checkpoint-per-phase, resource-leak, scaffolded-but-unplumbed
     pipeline, producer/consumer key mismatch, missing/incomplete smoke,
     estimand/headline-poisoning) → **SURFACE** it. Do NOT ship past it,
     do NOT same-diff-family pivot-loop:
     - **Interactive mode:** present the residual blocker(s) to the user
       (the two-path escalation is grandfathered for a genuine stuck-real
       blocker; frame the residual + ask how to proceed). Post the §5
       marker (`uv run python scripts/post_step_completed.py --issue <N>
       --step 5b --exit-kind parked --notes "code-review cap-5
       substantive residual; awaiting user"`), then EXIT awaiting
       the user.
     - **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): post
       `epm:failure v1` with `failure_class: code` referencing the
       residual blocker(s), set `status: blocked`, fire
       `PushNotification({"message": f"#{N} BLOCKED: ensemble review real
       residual at cap-5 — open it"[:200], "status": "proactive"})`, run
       CRON-TEARDOWN (§ CRON-TEARDOWN procedure — both legs incl.
       stray one-shot `/issue <N>` wakeups), post the §5 marker (`uv run python
       scripts/post_step_completed.py --issue <N> --step 5b --exit-kind
       failure-exit --notes "code-review cap-5 substantive residual;
       status:blocked"`), and EXIT. This is the standing halt path for a
       genuinely-stuck real blocker after the auto-continue space is
       exhausted (halt_criteria id=6 `concern_unresolved` family) — no
       more pivots, no more silent shipping past.

  For a plan that is ITSELF internally contradictory, the
  `plan_contradiction_replan` pivot (Step 7 / § pivot_criteria) still
  applies — that is a different signal (the plan is the defect), not a
  code-review cap-hit. Likewise the whack-a-mole detector (Step 5.bis(b))
  is unchanged; the retired pivot is specifically the "same diff family
  failed N rounds → re-plan" one.

**Codex twin no-show fallback.** If the Codex wrapper posts
`epm:failure v<m>` with `failure_class: codex-output-malformed` or
`failure_class: infra` (codex plugin missing), proceed with
single-reviewer (Claude-only) decision-making for that round. Do NOT
block on the Codex twin's absence; cap-5 still applies to the Claude
reviewer's count. Surface this to chat as one line: `Codex twin no-show
this round; using Claude reviewer only.` This fallback fires ONLY on the
posted `epm:failure` marker, or after the Step 5b durable-verdict-first
rule confirms NO durable verdict exists (no `epm:code-review-codex v<n>`
marker AND no conforming, round-fresh output file). An Agent-tool error
result alone never triggers it — and the same applies symmetrically to
the Claude reviewer: with no durable verdict, re-spawn it once per the
Step 5b rule; NEVER adopt a unilateral decision from the surviving
reviewer (#810 r4). An `epm:codex-task-failed` note carrying
`codex-quota-exhausted` is the org-quota outage short-circuit (#1126):
treat as an instant no-show (Claude-only), do not re-dispatch or
investigate; the sentinel self-expires at the stated reset. The Step 5a
pre-spawn sentinel check (#1204) makes this fallback fire WITHOUT
spawning the composer: a sentinel-skip recorded at spawn time is a
confirmed no-show — do NOT run the durable-verdict probe for a round
whose composer was never spawned, and do not wait for any
`epm:codex-task-*` marker (none will exist).

##### Step 5.bis: Pre-dispatch checks (compute-deviation + whack-a-mole)

Fires once per implementer round, AFTER code-review-PASS, BEFORE any
pod-provision or experimenter-dispatch action in Step 6. Two
independent triggers run in sequence:

**5.bis(a) — Compute-deviation pivot.** Scan the task's
`events.jsonl` for `epm:compute-deviation v1` markers posted in the
current implementer round (highest version with the same round number).
If present:

1. Parse the marker's body for `component`, `planned_wall_h`,
   `projected_wall_h`, `ratio`, `basis`. Route on the component's
   marker CHAIN (re-posts reuse the planner-§9 row name verbatim in
   `component:` — the loop guard and this routing key on it):
   - `action: auto_descope_to_<spec>` present AND the component's
     chain also carries a lever-0 record (`action: vectorize_fix_round`
     ran, or `signature_check: negative`) → a prior tick already
     accepted an auto-descope with lever 0 resolved; log one line and
     advance to Step 5.bis(b). An `auto_descope_to_<spec>` WITHOUT a
     lever-0 record is a pre-resolution like any other — treat as
     UNRESOLVED and proceed to step 2 (a descope never resolves
     lever 0; legacy and implementer self-descope markers arrive
     exactly this way).
   - `action: vectorize_fix_round` present (the fix round ran) and the
     post-fix ratio ≤ 2× → log one line and advance to Step 5.bis(b).
     Post-fix ratio still > 2× → SKIP step 2 (one mandatory fix round
     per component) and go to step 3 with the post-fix numbers plus
     the round's residual classification.
   - ANY pre-resolution (`action: continue_as_is`,
     `action: auto_descope_to_<spec>` per the bullet above, or any
     other legacy, poster-side, or crash-replay resolution) WITHOUT a
     lever-0 record (`vectorize_fix_round` ran, or
     `signature_check: negative`) → treat as UNRESOLVED and proceed to
     step 2; at ratio ≥ 5× without a valid clause-0c finding the
     pre-resolution is VOID.
2. **Vectorize-first signature check (pivot_criteria auto-action 0 +
   0b — REQUIRED before any descope).** From `basis:` + the round's
   implementer report, classify the deviation:
   - **Overhead-bound** (matches the `.claude/rules/vectorize-many-cell-fits.md`
     trigger — the canonical definition; illustratively: a serial
     per-cell/fold/layer/draw/row loop of small fits/solves/reductions,
     batch-1 model forwards, per-draw re-reduction of a fixed pool,
     per-row IO, or sequential shard-independent cells with an unused
     parallelism axis) → dispatch ONE vectorize/parallelize fix round:
     spawn `experiment-implementer` with a brief naming the marker,
     the rule + canonical helpers
     (`src/explore_persona_space/analysis/vectorized_mlp_skill.py`,
     `src/explore_persona_space/analysis/null_battery.py`), the
     equivalence gate against a SEEDED serial oracle (2-3 cells, a
     stated per-workload float tolerance), and the requirement that
     its closing `epm:compute-deviation v<next>` re-post carry
     `action: vectorize_fix_round`, the post-fix projection, AND the
     residual classification (a genuinely FLOP-bound / dependency-
     bound residual is recorded in `flop_bound_finding:` — that
     post-fix arithmetic constitutes the clause-0c finding). Dispatch
     for the component HALTS while the fix round runs — Step 6 is not
     reached for it this round; the round flows through code-review
     (Step 5) normally. Descope and `continue_as_is` are NOT eligible
     for the component until this round has run or a negative finding
     is recorded. Pinned plan hyperparameters do NOT exempt the lever —
     vectorization is recipe-preserving (equivalence-gated); pinning
     blocks descope, not vectorization (#722).
   - **Not overhead-bound** (`basis` names a genuinely FLOP-bound /
     API-latency / bandwidth / capacity-wait / already-vectorized-
     contention cause) → post `epm:compute-deviation v<next>` with
     `signature_check: negative` + 1-3 lines of arithmetic or the
     named quantified bottleneck, and proceed to step 3.
   - **Ambiguous basis** → treat as overhead-bound; the fix round's
     first action is the rule's diagnostic (FLOP back-of-envelope,
     cputime/walltime ratio) and the round may return the negative
     finding instead of a code change.
   **Pod release (0b):** if the deviating serial phase holds a GPU pod
   (or the pending dispatch would hold one idle through it), run
   `pod.py stop --issue <N>` while the fix round runs; the
   orchestrator that issued the stop owns `pod.py resume` at fix-round
   completion, before any re-dispatch (CLAUDE.md "CPU-only phases
   don't hold GPU pods"; this is continuing work, not parking).
3. Otherwise, attempt auto-descope per
   `workflow.yaml § pivot_criteria.compute_deviation_over_2x`:
   walk the planner's §9 stratification dimensions in priority order
   (seeds → framings → cells-per-stratum); for each dimension, compute
   the descoped projection (drop the dimension to its min-N-for-power
   per the planner's §9 stratification spec). The first descope whose
   ratio ≤ 1.5× AND keeps every dimension ≥ its min-N wins.
4. **Auto-descope success.** Post `epm:compute-deviation v2` with
   `action: auto_descope_to_<spec>`, update the implementer's per-cell
   parameters in the launch command, log to chat as one line, advance.
5. **Auto-descope fails** (no dimension keeps ratio ≤ 1.5× while
   staying above min-N): branch on session mode.

   - **Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
     `gates.conditional.compute_deviation_resolution` (id=12) with the
     2-option prompt. Quote the ratio inline. At ratio ≥ 5×,
     `continue_as_is` requires the recorded quantified clause-0c
     finding (`flop_bound_finding:` on the marker, or a
     `signature_check: negative` record meeting the 0c bar) — state it
     inline. On `continue_as_is`,
     advance to Step 5.bis(b) with the original parameters. On
     `accept_descope_to_<X>_with_caveats`, post `epm:compute-deviation v2`
     with the chosen descope spec + caveats and advance.

     <!-- gate: gates.conditional.compute_deviation_resolution -->

   - **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the
     ask AND never print the two options as a text menu. Auto-resolve
     per § Autonomous session behavior →
     `compute_deviation_resolution` (reachable only after the step-0
     lever is resolved; see that bullet for the full rule): pick
     `accept_descope_to_<X>_with_caveats` if any descope dimension
     preserves majority statistical power (≥0.6 of the planned cells);
     else `continue_as_is` and quote the projected ratio inline — at
     ratio ≥ 5×, `continue_as_is` additionally requires the recorded
     quantified clause-0c finding (`flop_bound_finding:` on the marker,
     or a `signature_check: negative` record meeting the 0c bar); if it
     is missing and no fix round ran, execute step 2's vectorize fix
     round instead; if the
     fix round ran but its re-post omitted the residual classification,
     obtain ONE corrective re-post (no second fix round), then resolve.
     State `Decision: <choice> because <reason>` AND EXECUTE the
     resolved action in this same turn (post `epm:compute-deviation v2`
     with the chosen `action:` and advance to Step 5.bis(b)); do NOT
     state the Decision and then end the turn.

**5.bis(b) — Whack-a-mole detector.** Scan the task's `events.jsonl`
for `epm:new-bug-class v1` markers posted in the trailing 5
implementer rounds (rounds N-4..N, where N is the current round).
EXCLUDE rounds whose `epm:experiment-implementation v<n>` event note
contained the regex `<!-- workflow-fix-candidate v1 -->` (per the
workflow-fix-on-bug protocol; those drive the workflow-fix-task-filing
default — a filed `kind: infra` task + a `/issue --auto` session — not
strategy-pivot consideration). "Consecutive" below means consecutive
across NON-EXCLUDED rounds — i.e. when an excluded round sits between
two tagged rounds, the excluded round is skipped, and the two tagged
rounds count as consecutive for the trigger.

Two triggers:
- **PRIMARY:** 3 distinct `bug_class` tag values across the 3 most
  recent non-excluded rounds (each contributed a distinct tag).
- **SECONDARY:** 2 distinct `bug_class` tag values across the 2 most
  recent non-excluded rounds AND at least 1
  `epm:compute-deviation v1` event in the trailing 5 rounds (N-4..N).

On fire, branch on session mode.

**Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
`gates.conditional.whack_a_mole_pivot` (id=11) with 2 options:
- `continue-as-planned` (one-line rationale + cost estimate of the
  next pod-provision + experimenter dispatch).
- `pivot-to-<X>` (one-line rationale + cost estimate of the canonical
  alternative the implementer's report named, e.g. unification of
  smoke + sweep paths).

On `continue-as-planned`, advance to Step 6 normally; round counter
does NOT reset. On `pivot-to-<X>`, route back to `status:planning`
for re-planning; round counter does NOT increment (this is a
strategy pivot, not a fresh review round).

**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the ask
AND never print the two options as a text menu. Auto-resolve per §
Autonomous session behavior → `whack_a_mole_pivot`: pick `pivot-to-<X>`
if the implementer's report named a canonical alternative AND the next
round on the current path would burn >2× the cost of the pivot; else
`continue-as-planned`. State `Decision: <choice> because <reason>` AND
EXECUTE the resolved action in this same turn — on `pivot-to-<X>`:
`task.py set-status <N> planning` + re-invoke `/adversarial-planner`
with the pivot scope (round counter does NOT increment; mid same-issue
follow-up round, SKIP the `set-status` — status-hold rule, Step 9b §
Same-issue follow-up loop step 3 — and just re-invoke the planner); on
`continue-as-planned`: advance to Step 6 normally (round counter does
NOT reset). Do NOT state the Decision and then end the turn.

Canonical worked test case: the #397 replay trace lives at
`tests/whack_a_mole_397_replay_note.md` (how a workflow-fix-excluded
round makes the SECONDARY trigger fire one round earlier than PRIMARY).

<!-- gate: gates.conditional.whack_a_mole_pivot -->
