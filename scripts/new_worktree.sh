#!/usr/bin/env bash
# Create a SPARSE worktree — the default for /issue + feature/infra branches
# (task #596). Excludes the heavy history dirs (eval_results/, external/,
# ood_eval_results/: ~3.4G of a ~3.8G checkout) and pre-includes the issue's
# own artifact dirs so `git add eval_results/issue_<N>/...` needs no ceremony.
#
# Usage: scripts/new_worktree.sh <worktree-path> <branch> [--issue N] [--full]
#
#   --issue N   pre-add cones eval_results/issue_N + ood_eval_results/issue_N
#   --full      plain full checkout (escape hatch; state the reason when used)
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

WT=$(realpath -m "${1:?usage: new_worktree.sh <worktree-path> <branch> [--issue N] [--full]}")
BRANCH=${2:?usage: new_worktree.sh <worktree-path> <branch> [--issue N] [--full]}
shift 2
ISSUE="" FULL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --issue) ISSUE=$2; shift 2 ;;
    --full)  FULL=1; shift ;;
    *) echo "new_worktree: unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(git rev-parse --show-toplevel)
EXCLUDES="eval_results external ood_eval_results"

# Drop stale registrations whose directories were deleted out-of-band
# (registered-but-directory-gone) so the reuse check below sees truth.
git -C "$REPO_ROOT" worktree prune

# -b fails if the branch already exists (resume after worktree removal) —
# fall back to attaching the existing branch. Preserve the FIRST attempt's
# stderr and re-emit it if the fallback also fails (don't swallow the real
# error).
_add() {
  local err1
  if ! err1=$(git -C "$REPO_ROOT" worktree add "$@" "$WT" -b "$BRANCH" 2>&1); then
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
  # shellcheck disable=SC2046,SC2086
  DIRS=$(git -C "$REPO_ROOT" ls-tree --name-only -d HEAD \
         | grep -vxF $(printf -- '-e %s ' $EXCLUDES))
  [ -n "$ISSUE" ] && CONES="eval_results/issue_${ISSUE} ood_eval_results/issue_${ISSUE}"
  # shellcheck disable=SC2086
  git -C "$WT" sparse-checkout set $DIRS $CONES
  [ "$(git -C "$WT" config --worktree core.sparseCheckoutCone || true)" = true ] \
    || { echo "new_worktree: FATAL — cone mode failed to engage in $WT" >&2; return 1; }
  git -C "$WT" checkout "$BRANCH"
}

# Healthy = HEAD resolves AND the tree is materialized (CLAUDE.md is in-cone
# in both sparse and full layouts; a --no-checkout limbo tree lacks it).
_is_populated() {
  git -C "$WT" rev-parse --verify HEAD >/dev/null 2>&1 && [ -e "$WT/CLAUDE.md" ]
}

if git -C "$REPO_ROOT" worktree list --porcelain | grep -qxF "worktree $WT"; then
  if _is_populated; then
    echo "new_worktree: $WT already exists — reusing as-is"
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
  _cleanup_failed_create() {
    echo "new_worktree: creation FAILED — removing half-created worktree" >&2
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

du -sh "$WT" | awk -v wt="$WT" '{print "new_worktree: created " wt " (" $1 ")"}'
[ "$FULL" = 1 ] || {
  echo "new_worktree: sparse — excluded: $EXCLUDES"
  echo "new_worktree: materialize an excluded dir on demand:"
  echo "  git -C \"$WT\" sparse-checkout add <dir>   # e.g. eval_results/issue_<M>"
}
