#!/usr/bin/env bash
# Create a SPARSE worktree — the default for /issue + feature/infra branches
# (task #596). Excludes the heavy history dirs (eval_results/, external/,
# ood_eval_results/: ~3.4G of a ~3.8G checkout) and pre-includes the issue's
# own artifact dirs so `git add eval_results/issue_<N>/...` needs no ceremony.
#
# Usage: scripts/new_worktree.sh <worktree-path> <branch> [--issue N] [--full] [--base-local]
#
#   --issue N   pre-add cones eval_results/issue_N + ood_eval_results/issue_N.
#               When omitted, N is INFERRED from a canonical `issue-<N>` /
#               `issue-<N>-<suffix>` branch name (explicit flag wins; other
#               branch names get no issue cones). Reuse runs an idempotent
#               own-issue cone repair on sparse worktrees. Slug-variant dirs
#               (eval_results/issue<N>_<slug>/) stay OUT of scope — add them
#               on demand: git -C <wt> sparse-checkout add <dir>.
#   --full      plain full checkout (escape hatch; state the reason when used)
#   --base-local  base the NEW branch on the main checkout's current local
#                 HEAD (the pre-#1214 behavior) instead of fetched origin/main.
#                 Escape hatch for offline work / deliberately branching off
#                 unpushed local commits; state the reason when used.
#
# Branch base (#1214): a NEW branch is cut from freshly-fetched
# refs/remotes/origin/main (pushed history only), NOT from the shared repo
# root's local main — local main accretes/rewrites unpushed task-state
# commits which otherwise get baked into the branch and trip the Step 10d
# merge guards. Ladder: --base-local → local HEAD; no `origin` remote →
# local HEAD + WARN; fetch failed but a prior origin/main exists → STALE
# origin/main + WARN; fetch failed and no origin/main → FATAL (exit 5).
#
# Reuse: if <worktree-path> is already a registered worktree with a populated
# tree, exits 0 untouched (the /issue resume case); a registered-but-
# unpopulated worktree (interrupted creation) is repaired in place. Symlinks
# the repo .env in every case.
#
# -E (errtrace) is load-bearing: without it the ERR trap below is NOT
# inherited by shell functions, so a failure inside _sparse_setup would
# leave the half-created worktree behind — the exact incident class the
# trap exists to prevent.
set -Eeuo pipefail

WT=$(realpath -m "${1:?usage: new_worktree.sh <worktree-path> <branch> [--issue N] [--full] [--base-local]}")
BRANCH=${2:?usage: new_worktree.sh <worktree-path> <branch> [--issue N] [--full] [--base-local]}
shift 2
ISSUE="" FULL=0 BASE_LOCAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --issue)
      ISSUE=${2:?new_worktree: --issue requires a value}
      # A non-numeric value would silently create a junk cone
      # (eval_results/issue_<garbage>) — refuse loudly. (#596 reviewer minor)
      case "$ISSUE" in
        *[!0-9]*) echo "new_worktree: --issue must be numeric, got: $ISSUE" >&2; exit 2 ;;
      esac
      shift 2 ;;
    --full)  FULL=1; shift ;;
    --base-local) BASE_LOCAL=1; shift ;;
    *) echo "new_worktree: unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --issue omitted: infer it from the canonical branch name — `issue-<N>` (the
# /issue Step 4a naming) or `issue-<N>-<suffix>` (same-issue follow-up naming).
# Explicit --issue always wins (it was consumed above and ISSUE is non-empty).
# Conservative: anything not purely numeric between `issue-` and the first `-`
# is NOT inferred (issue-12abc, issue-x → no cones, today's behavior). The
# worktree PATH is never consulted — the branch is the authoritative name.
# (#906: flagless creation left eval_results/issue_906 out-of-cone; #1054.)
if [ -z "$ISSUE" ]; then
  case "$BRANCH" in
    issue-[0-9]*)
      _stem=${BRANCH#issue-}      # "906" or "906-followup-x" or "9a6"
      _num=${_stem%%-*}           # chars before the first hyphen
      case "$_num" in
        *[!0-9]*|'') : ;;         # not purely numeric → no inference
        *)
          ISSUE=$_num
          echo "new_worktree: inferred --issue $ISSUE from branch '$BRANCH'" >&2
          ;;
      esac
      ;;
  esac
fi

# Anchor to the MAIN checkout even when invoked from inside another worktree:
# `--show-toplevel` would resolve to THAT worktree, and the cone include list
# below would then be computed from its branch HEAD instead of the main
# checkout's. Same idiom as /issue SKILL.md Step 10d. (#596 reviewer minor)
REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
EXCLUDES="eval_results external ood_eval_results"

# ─── #681 data-disk + migration guards (run BEFORE any worktree creation) ────
#
# Post-#681 the .claude/worktrees/ tree is a BIND mount onto a dedicated GCP
# data disk with ext4 per-PROJECT quotas. Three guards keep worktree creation
# safe: (1) refuse while a cutover migration LOCK is held; (2) assert the bind
# is LIVE so a worktree never silently lands back on `/` after a failed mount /
# reboot; (3) tag the new issue-<N> subtree to project id N with a hard byte cap
# so one task cannot starve the shared data disk. All three are seam-injectable
# for offline CI (no privilege / no real bind / no setquota needed there).

# Migration LOCK: the cutover (plan §4 Phase 2) sets this; new_worktree.sh AND
# task_workflow._ensure_managed_main_worktree both refuse while it exists, so a
# worktree is never created mid-swap onto the to-be-renamed tree.
MIGRATION_LOCK="$REPO_ROOT/.claude/cache/worktree-migration.LOCK"
if [ -e "$MIGRATION_LOCK" ]; then
  echo "new_worktree: REFUSING — worktree migration in progress ($MIGRATION_LOCK exists)." >&2
  echo "new_worktree: retry once the data-disk cutover lifts the LOCK." >&2
  exit 3
fi

# Bind-mount liveness probe. The data disk is mounted at the device level and
# bind-mounted onto $REPO_ROOT/.claude/worktrees; if that bind is absent (a
# missing data disk, a failed mount, a half-mounted boot state) a new worktree
# would silently land on the boot disk `/`. Assert the bind is live and FAIL
# LOUD otherwise. Test/pre-cutover seam: EPS_WORKTREE_BIND_PROBE overrides the
# probe command (`true` to force-pass, `false` to force-fail); EPS_WORKTREE_REQUIRE_BIND=1
# OPTS IN to the assertion (default OFF so this is a no-op before the cutover
# lands the bind + flips the env in the cron + bootstrap).
_bind_is_live() {
  local probe="${EPS_WORKTREE_BIND_PROBE:-}"
  if [ -n "$probe" ]; then
    eval "$probe"
    return $?
  fi
  # Production probe: the path itself MUST be a mountpoint. Use `findmnt
  # --mountpoint` (NOT `--target`): `--target` walks UP to the containing
  # filesystem and returns rc=0 for ANY ordinary directory on a mounted fs,
  # so a MISSING bind would pass the assertion and the worktree would land
  # silently on the boot disk `/` (#681 round-2 Critical). `--mountpoint`
  # succeeds ONLY when the path is itself a mount, which is exactly the bind
  # we are asserting is live.
  findmnt --noheadings --mountpoint "$REPO_ROOT/.claude/worktrees" >/dev/null 2>&1
}
if [ "${EPS_WORKTREE_REQUIRE_BIND:-0}" = 1 ]; then
  if ! _bind_is_live; then
    echo "new_worktree: FATAL — the data-disk bind at $REPO_ROOT/.claude/worktrees is NOT live;" >&2
    echo "new_worktree: refusing to create a worktree that would land on the boot disk /." >&2
    echo "new_worktree: check the mount: findmnt --mountpoint $REPO_ROOT/.claude/worktrees ; sudo mount -a" >&2
    exit 4
  fi
fi

# Attached-but-inert WARN (#681 cutover not yet applied). The strict assertion
# above OPTS IN via EPS_WORKTREE_REQUIRE_BIND=1; before the cutover lands that
# flag is OFF, so the assertion is a no-op and the dangerous state — data disk
# mounted but .claude/worktrees NOT bind-mounted onto it — is SILENT, and new
# worktree caches keep landing on the boot disk `/` with no quota. This WARN
# makes that state self-announcing: a once-per-session advisory that fires
# exactly when the data disk IS a live mount, the bind is NOT live, and the
# assertion is OFF. It is the STRICTLY WEAKER complement of the assertion
# (mutually exclusive on EPS_WORKTREE_REQUIRE_BIND) and NEVER blocks creation.
_datadisk_is_live() {
  local probe="${EPS_WORKTREE_DATADISK_PROBE:-}"
  if [ -n "$probe" ]; then
    eval "$probe"
    return $?
  fi
  # Production: the data-disk path itself MUST be a live mountpoint. Use
  # `findmnt --mountpoint` (NOT `--target`): `--target` walks UP to the
  # containing filesystem and returns rc=0 for ANY ordinary directory on a
  # mounted fs, so an UNMOUNTED data disk would falsely report as present and
  # the WARN would fire on clean non-GCP / CI machines (same trap the bind
  # probe documents above). The ${EPS_VM_DATA_DISK_PATH:-/mnt/eps-data}
  # default-fallback mirrors _assign_project_quota below: EPS_VM_DATA_DISK_PATH
  # is unset on the live VM, so a literal-env predicate would never fire on the
  # very machine with the bug.
  findmnt --noheadings --mountpoint "${EPS_VM_DATA_DISK_PATH:-/mnt/eps-data}" >/dev/null 2>&1
}
INERT_SENTINEL="$REPO_ROOT/.claude/cache/worktree-inert-warned"
if [ "${EPS_WORKTREE_REQUIRE_BIND:-0}" != 1 ] && [ ! -e "$INERT_SENTINEL" ]; then
  if _datadisk_is_live && ! _bind_is_live; then
    _dd="${EPS_VM_DATA_DISK_PATH:-/mnt/eps-data}"
    {
      echo "new_worktree: WARN — #681 data disk ($_dd) is mounted but"
      echo "new_worktree:   $REPO_ROOT/.claude/worktrees is NOT bind-mounted onto it,"
      echo "new_worktree:   so this worktree's caches will land on the boot disk / with NO quota."
      echo "new_worktree:   Cutover: bind .claude/worktrees onto $_dd then export"
      echo "new_worktree:   EPS_WORKTREE_REQUIRE_BIND=1 EPS_WORKTREE_ASSIGN_QUOTA=1 (cron + bootstrap)."
      echo "new_worktree:   See task #681 / CLAUDE.md 'Disk hygiene'."
    } >&2
    # Best-effort sentinel (|| true): a read-only or missing .claude/cache must
    # never abort worktree creation — the WARN is advisory, not load-bearing.
    mkdir -p "$(dirname "$INERT_SENTINEL")" 2>/dev/null || true
    : > "$INERT_SENTINEL" 2>/dev/null || true
  fi
fi

# Per-issue ext4 project-quota cap. After the worktree is created (below), tag
# its subtree to project id == issue number with a hard byte cap so one task
# cannot starve the shared data disk (the per-TASK bound, plan §2.5/§4 Phase 4).
# Cap from EPS_ISSUE_DISK_CAP_GB (default 128). Seam: EPS_WORKTREE_QUOTA_CMD
# overrides the quota-assign command for CI (it receives "<projid> <cap_kb> <path>");
# EPS_WORKTREE_ASSIGN_QUOTA=1 OPTS IN (default OFF — a no-op before the cutover).
ISSUE_DISK_CAP_GB="${EPS_ISSUE_DISK_CAP_GB:-128}"
case "$ISSUE_DISK_CAP_GB" in
  *[!0-9]*) echo "new_worktree: EPS_ISSUE_DISK_CAP_GB must be a positive integer, got: $ISSUE_DISK_CAP_GB" >&2; exit 2 ;;
esac
# A non-positive cap (0 / 00 / ...) would set cap_kb=0, SILENTLY DISABLING the
# per-task hard cap — match worktree_quota.issue_disk_cap_gb() and fall back to
# the 128 GB default rather than ship an unbounded quota (#681 round-2 fix).
if [ "$((10#$ISSUE_DISK_CAP_GB))" -le 0 ]; then
  echo "new_worktree: EPS_ISSUE_DISK_CAP_GB non-positive ($ISSUE_DISK_CAP_GB); falling back to 128 GB default" >&2
  ISSUE_DISK_CAP_GB=128
fi
_assign_project_quota() {
  # $1 = issue number (== project id). No-op when no issue / opt-in off.
  local projid="$1" cap_kb
  [ -z "$projid" ] && return 0
  [ "${EPS_WORKTREE_ASSIGN_QUOTA:-0}" = 1 ] || return 0
  cap_kb=$((ISSUE_DISK_CAP_GB * 1024 * 1024))
  local cmd="${EPS_WORKTREE_QUOTA_CMD:-}"
  if [ -n "$cmd" ]; then
    eval "$cmd $projid $cap_kb $WT"
    return $?
  fi
  # Production: tag the subtree to project N, set a hard byte cap (soft=0=off).
  local dd="${EPS_VM_DATA_DISK_PATH:-/mnt/eps-data}"
  sudo chattr -R -p "$projid" +P "$WT" \
    && sudo setquota -P "$projid" 0 "$cap_kb" 0 0 "$dd"
}

# Drop stale registrations whose directories were deleted out-of-band
# (registered-but-directory-gone) so the reuse check below sees truth.
git -C "$REPO_ROOT" worktree prune

# Base ref for NEW branches (#1214). Step 10d merges land on origin/main; a
# branch cut from the shared root's LOCAL main inherits its unpushed
# task-state churn (2026-07-08: 3/6 sessions, 10-25 min forensics each; one
# branch had 78 foreign commits baked in). Sets START_POINT (global): empty =
# local HEAD (today's behavior); otherwise the ref passed as the start-point
# to `worktree add -b`. Only called when the branch will actually be created
# (the -b path); the existing-branch attach path takes no base.
START_POINT=""
_resolve_base() {
  if [ "$BASE_LOCAL" = 1 ]; then
    echo "new_worktree: --base-local — basing '$BRANCH' on the main checkout's local HEAD" >&2
    return 0
  fi
  if ! git -C "$REPO_ROOT" remote get-url origin >/dev/null 2>&1; then
    echo "new_worktree: WARN — no 'origin' remote; basing '$BRANCH' on local HEAD" >&2
    return 0
  fi
  # Explicit refspec: guarantees refs/remotes/origin/main updates regardless of
  # the remote's configured fetch refspec (a bare `fetch origin main` only
  # updates it opportunistically). FETCH_HEAD is deliberately NOT used: it is
  # one shared file, racy under this repo's many concurrent sessions.
  # `timeout 60`: a hung fetch must not wedge an autonomous pipeline; timeout
  # (rc 124) falls through to the stale-origin/main tier below.
  if timeout 60 git -C "$REPO_ROOT" fetch --quiet origin \
       "+refs/heads/main:refs/remotes/origin/main"; then
    START_POINT="refs/remotes/origin/main"
  elif git -C "$REPO_ROOT" rev-parse --verify --quiet refs/remotes/origin/main >/dev/null; then
    echo "new_worktree: WARN — 'git fetch origin main' failed; basing '$BRANCH' on the" >&2
    echo "new_worktree:   last-fetched (possibly STALE) origin/main — pushed history only," >&2
    echo "new_worktree:   so the #1214 churn bug is NOT reintroduced." >&2
    START_POINT="refs/remotes/origin/main"
  else
    echo "new_worktree: FATAL — fetch failed and no refs/remotes/origin/main exists;" >&2
    echo "new_worktree:   refusing to base '$BRANCH' on local main (unpushed task-state" >&2
    echo "new_worktree:   churn, #1214). Fix the network/remote, or pass --base-local." >&2
    exit 5
  fi
  echo "new_worktree: basing '$BRANCH' on $START_POINT" \
       "($(git -C "$REPO_ROOT" rev-parse --short "$START_POINT"))" >&2
}

# -b fails if the branch already exists (resume after worktree removal) —
# fall back to attaching the existing branch. Preserve the FIRST attempt's
# stderr and re-emit it if the fallback also fails (don't swallow the real
# error).
_add() {
  local err1
  # ${START_POINT:+...}: append the base commit-ish only when _resolve_base
  # chose one (empty = local HEAD, today's behavior). --no-track: a branch cut
  # from the remote-tracking ref must NOT gain origin/main as upstream (today's
  # branches have no upstream; `git status`/`git push` semantics stay identical).
  if ! err1=$(git -C "$REPO_ROOT" worktree add "$@" "$WT" -b "$BRANCH" --no-track \
                ${START_POINT:+"$START_POINT"} 2>&1); then
    git -C "$REPO_ROOT" worktree add "$@" "$WT" "$BRANCH" || {
      echo "new_worktree: both add attempts failed; first attempt said:" >&2
      echo "$err1" >&2
      return 1
    }
  fi
}

# Idempotent sparse setup + checkout: safe to (re-)run on a worktree in
# --no-checkout limbo (interrupted creation) as well as on a fresh one.
_sparse_setup() {
  # Preserve cones already present from a PRIOR run (the repair case): a
  # repair WITHOUT --issue on a worktree originally created WITH --issue
  # must not silently drop the per-issue cones. Capture BEFORE init (`list`
  # errors on a not-yet-sparse tree → empty). Duplicates with $DIRS/$CONES
  # are harmless — `set` dedupes. (#596 reviewer minor)
  local EXISTING
  EXISTING=$(git -C "$WT" sparse-checkout list 2>/dev/null || true)
  # ORDER MATTERS on git 2.34: `init --cone` FIRST. `set --cone` is silently
  # accepted as a literal PATTERN (no --cone flag on `set` until git 2.35+),
  # which yields non-cone any-depth matching — the failure mode this script
  # exists to prevent. Hence the hard assert below.
  git -C "$WT" sparse-checkout init --cone
  # Include list computed at CREATION time from the repo root's HEAD: every
  # top-level tracked dir except the excludes. Top-level dirs that exist only
  # on the issue branch (or merge into main later) are out-of-cone — the fix
  # is the documented `git -C "$WT" sparse-checkout add <dir>`.
  local DIRS CONES=""
  # The unquoted $DIRS/$CONES/$EXISTING expansions below word-split on
  # whitespace — guard loudly if a top-level dir name ever embeds whitespace
  # or git quote-escaping, rather than mis-splitting it. (#596 reviewer minor)
  if git -C "$REPO_ROOT" ls-tree --name-only -d HEAD | grep -Eq '[[:space:]"\\]'; then
    echo "new_worktree: FATAL — top-level dir name with whitespace/quoting defeats the unquoted cone expansion" >&2
    return 1
  fi
  # shellcheck disable=SC2046,SC2086
  DIRS=$(git -C "$REPO_ROOT" ls-tree --name-only -d HEAD \
         | grep -vxF $(printf -- '-e %s ' $EXCLUDES))
  [ -n "$ISSUE" ] && CONES="eval_results/issue_${ISSUE} ood_eval_results/issue_${ISSUE}"
  # Test-suite cones: the full `tests/` suite (the Step 9c test-verdict gate)
  # reads OTHER issues' committed eval_results/ artifacts as fixtures/references.
  # Those dirs are under the EXCLUDES above, so a sparse worktree would FAIL the
  # gate with FileNotFoundError until manually `sparse-checkout add`-ed. Pre-add
  # every cone listed in tests/sparse_cones.txt (one dir per line; blank +
  # `#`-comment lines skipped) so the gate passes with no ceremony. (#671)
  local TEST_CONES="" REGISTRY="$REPO_ROOT/tests/sparse_cones.txt"
  if [ -f "$REGISTRY" ]; then
    # Same word-split / quoting guard as $DIRS above: a cone with whitespace or
    # git quote-escaping would mis-split the unquoted expansion — refuse loudly.
    if grep -Ev '^[[:space:]]*(#|$)' "$REGISTRY" | grep -Eq '[[:space:]"\\]'; then
      echo "new_worktree: FATAL — tests/sparse_cones.txt line with whitespace/quoting defeats the unquoted cone expansion" >&2
      return 1
    fi
    # `|| true`: an all-comment / empty registry makes grep exit 1, which under
    # `set -o pipefail` would otherwise abort the assignment (same idiom as the
    # `sparse-checkout list || true` above). TEST_CONES stays "" -> harmless.
    TEST_CONES=$(grep -Ev '^[[:space:]]*(#|$)' "$REGISTRY" | tr '\n' ' ' || true)
  fi
  # shellcheck disable=SC2086
  git -C "$WT" sparse-checkout set $DIRS $CONES $TEST_CONES $EXISTING
  [ "$(git -C "$WT" config --worktree core.sparseCheckoutCone || true)" = true ] \
    || { echo "new_worktree: FATAL — cone mode failed to engage in $WT" >&2; return 1; }
  git -C "$WT" checkout "$BRANCH"
}

# Healthy = HEAD resolves AND the tree is materialized (CLAUDE.md is in-cone
# in both sparse and full layouts; a --no-checkout limbo tree lacks it).
_is_populated() {
  git -C "$WT" rev-parse --verify HEAD >/dev/null 2>&1 && [ -e "$WT/CLAUDE.md" ]
}

# Reuse-path own-issue cone repair (#906/#1054): a worktree created flagless
# BEFORE inference existed (or whose cones were dropped) stays broken for life
# — the reuse branch exits 0 before _sparse_setup ever runs again. Idempotently
# ensure the own-issue cones on every reuse. Contract: the reuse path stays
# exit-0 (it fires on every /issue resume), so a failed add WARNS — it never
# fails the script; the downstream `git add eval_results/...` failure is loud
# on its own, so nothing is hidden, and the WARN names the manual fix.
_ensure_issue_cones() {
  [ -z "$ISSUE" ] && return 0
  # FULL (non-sparse) worktrees are untouched: cone-mode worktree config is
  # unset there (and `git config --worktree` may error without the
  # worktreeConfig extension — hence 2>/dev/null || true).
  [ "$(git -C "$WT" config --worktree core.sparseCheckoutCone 2>/dev/null || true)" = true ] \
    || return 0
  git -C "$WT" sparse-checkout add \
      "eval_results/issue_${ISSUE}" "ood_eval_results/issue_${ISSUE}" \
    || echo "new_worktree: WARN — could not ensure eval_results/issue_${ISSUE} cones on reuse;" \
            "repair manually: git -C \"$WT\" sparse-checkout add eval_results/issue_${ISSUE} ood_eval_results/issue_${ISSUE}" >&2
}

if git -C "$REPO_ROOT" worktree list --porcelain | grep -qxF "worktree $WT"; then
  if _is_populated; then
    echo "new_worktree: $WT already exists — reusing as-is"
    _ensure_issue_cones
    ln -sf "$REPO_ROOT/.env" "$WT/.env"
    exit 0
  fi
  # Registered but unpopulated: a previous run died between `add --no-checkout`
  # and `checkout` (the half-created-worktree incident class). Repair in place —
  # _sparse_setup is idempotent; for --full just finish the checkout.
  echo "new_worktree: $WT registered but unpopulated (interrupted creation) — repairing"
  if [ "$FULL" = 1 ]; then git -C "$WT" checkout "$BRANCH"; else _sparse_setup; fi
else
  # Fresh creation. Best-effort cleanup on FAILURE (set -e → ERR trap): a
  # half-registered worktree must not survive to poison the next reuse check.
  # (A SIGKILL still can't fire the trap — that residue is what the repair
  # branch above handles. Belt and suspenders.)
  CREATED_BRANCH=0
  git -C "$REPO_ROOT" rev-parse --verify "$BRANCH" >/dev/null 2>&1 || CREATED_BRANCH=1
  # Fetch/base-resolve ONLY when a branch will actually be created: the
  # pre-existing-branch resume takes the attach fallback (no base), and must
  # stay network-independent. Runs BEFORE the ERR trap: a FATAL here exits
  # with nothing created, so there is nothing to clean up.
  if [ "$CREATED_BRANCH" = 1 ]; then _resolve_base; fi
  _cleanup_failed_create() {
    echo "new_worktree: creation FAILED — removing half-created worktree" >&2
    # $WT was realpath -m-normalized at parse time, so this remove targets
    # the same path spelling `worktree add` registered above (symlink-
    # spelling parity); the `worktree prune` on the next run is the backstop
    # for any residue. (#596 reviewer minor — judged: comment, not code)
    git -C "$REPO_ROOT" worktree remove --force "$WT" 2>/dev/null || true
    [ "$CREATED_BRANCH" = 1 ] && git -C "$REPO_ROOT" branch -D "$BRANCH" 2>/dev/null || true
  }
  trap _cleanup_failed_create ERR
  if [ "$FULL" = 1 ]; then
    _add
  else
    _add --no-checkout
    _sparse_setup
  fi
  trap - ERR
fi

# Worktrees do NOT inherit the gitignored repo .env (Step 4a contract).
ln -sf "$REPO_ROOT/.env" "$WT/.env"

# Per-issue ext4 project-quota cap (#681 — the per-task bound). No-op unless
# EPS_WORKTREE_ASSIGN_QUOTA=1 and an issue is known (--issue N, or inferred
# from the branch name — #1054). Idempotent (re-tagging a subtree + re-setting
# its cap is harmless); the reuse path above exit 0s before here — a reused
# worktree was tagged at creation when its issue was known; pre-inference
# flagless worktrees can be tagged manually post-cutover. Quota assignment
# deliberately does NOT run on the reuse path (sudo chattr -R on every resume
# is too heavy; #1054 plan §11 D6).
_assign_project_quota "$ISSUE"

du -sh "$WT" | awk -v wt="$WT" '{print "new_worktree: created " wt " (" $1 ")"}'
[ "$FULL" = 1 ] || {
  echo "new_worktree: sparse — excluded: $EXCLUDES"
  echo "new_worktree: materialize an excluded dir on demand:"
  echo "  git -C \"$WT\" sparse-checkout add <dir>   # e.g. eval_results/issue_<M>"
}
