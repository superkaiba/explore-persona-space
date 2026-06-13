#!/usr/bin/env bash
# PreToolUse(Bash) guard: block branch-switching in the SHARED repo-root tree.
#
# The repo root (/home/thomasjiralerspong/explore-persona-space) is the
# canonical commit target for scripts/task.py and every concurrent VM Claude
# session — they all assume the working tree is on `main`. Running
# `git checkout -b` / `git switch` here moves the branch out from under those
# concurrent committers: their commits land on the feature branch, and a
# concurrent `git add <file> && git commit` sweeps THIS session's uncommitted
# edits to <file> into the wrong commit.
#
# Incident 2026-06-01: an infra session ran `git checkout -b fix/sweep-ckpt-persist`
# in the repo root; a concurrent marker-leakage session's CLAUDE.md commit then
# bundled the infra session's Upload-Policy paragraph, and task #459 state landed
# on the feature branch.
#
# Fix: do feature/infra branch work in a dedicated worktree instead:
#     bash scripts/new_worktree.sh .claude/worktrees/<name> <branch>
#     cd .claude/worktrees/<name>
#
# Contract: reads the PreToolUse JSON on stdin, blocks (exit 2 + stderr fed
# back to Claude) only when a branch-CHANGING git command would move the
# repo-root tree off `main`. Exit 2 is the documented PreToolUse blocking
# exit code; any OTHER non-zero is non-blocking (stderr goes to the user and
# the tool call PROCEEDS) — code.claude.com/docs/en/hooks: "If your hook is
# meant to enforce a policy, use exit 2."
# Fail-soft: any ambiguity / parse failure exits 0 (never traps the user).
set -u

REPO=/home/thomasjiralerspong/explore-persona-space

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

# Only consider git checkout/switch invocations at all.
echo "$cmd" | grep -qE '\bgit\b.*\b(checkout|switch)\b' || exit 0

# Allow anything explicitly scoped to another worktree (git -C <path>, or a
# `cd <path-with-.claude/worktrees|/tmp>` earlier in the command chain).
if echo "$cmd" | grep -qE '\bgit +-C +' \
   || echo "$cmd" | grep -qE 'cd +[^;&|]*\.claude/worktrees/' \
   || echo "$cmd" | grep -qE 'cd +/tmp/'; then
  exit 0
fi

blocked=""

# git switch <branch> / git switch -c <branch>  (switch is branch-only).
# Allow only `git switch main`.
if echo "$cmd" | grep -qE '\bgit\b[^;&|]*\bswitch\b'; then
  if ! echo "$cmd" | grep -qE '\bswitch\b +(-c +|-C +)?main\b'; then
    blocked="git switch"
  fi
fi

# git checkout -b/-B <branch>  (branch creation).
if echo "$cmd" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-b|-B)\b'; then
  blocked="git checkout -b"
fi

# git checkout <existing-branch>  — NOT a file restore (no `--`), arg is a real
# local branch ref, and not `main`.
if echo "$cmd" | grep -qE '\bgit\b[^;&|]*\bcheckout\b' \
   && ! echo "$cmd" | grep -qE 'checkout\b[^;&|]*--'; then
  arg=$(echo "$cmd" | sed -nE 's/.*\bcheckout\b +([^ ;&|]+).*/\1/p')
  case "$arg" in
    ""|-b|-B|-f|--force|main|-) : ;;  # not a branch-switch we block
    *)
      if git -C "$REPO" show-ref --verify --quiet "refs/heads/$arg"; then
        blocked="git checkout $arg"
      fi
      ;;
  esac
fi

[ -n "$blocked" ] || exit 0

# Only protect the on-main state. If the repo-root tree is already off main,
# the horse has bolted — don't trap the user trying to recover.
cur=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$cur" = main ] || exit 0

echo "BLOCKED: '$blocked' would move the SHARED repo-root tree off main. The repo root is the canonical commit target for scripts/task.py and every concurrent VM session (all assume HEAD==main); switching branches here hijacks their commits and sweeps cross-session uncommitted edits into the wrong commit (incident 2026-06-01). Do feature/infra branch work in a worktree instead:
  bash scripts/new_worktree.sh .claude/worktrees/<name> <branch> && cd .claude/worktrees/<name>
To override deliberately, run the git command from inside a worktree (git -C .claude/worktrees/<name> ...)." >&2
exit 2
