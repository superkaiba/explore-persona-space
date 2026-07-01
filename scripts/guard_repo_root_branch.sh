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

# Strip quoted-string arguments before running the branch/detach detectors, so
# a quoted git-verb literal inside another command's argument does not trigger a
# false positive. Canonical offender: `task.py post-marker <N> epm:X --note
# "... git switch ..."` — the note text contains "git switch" but the command
# is not a git invocation at all. The detectors below scan $cmd_scan; the
# worktree/`cd` scoping ESCAPE (below) intentionally scans the ORIGINAL $cmd so
# a legitimately quoted path (e.g. `cd "/tmp/x"`) still escapes.
cmd_scan=$(echo "$cmd" | sed -E "s/\"[^\"]*\"//g; s/'[^']*'//g")

# Only consider git checkout/switch invocations at all.
echo "$cmd_scan" | grep -qE '\bgit\b.*\b(checkout|switch)\b' || exit 0

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
if echo "$cmd_scan" | grep -qE '\bgit\b[^;&|]*\bswitch\b'; then
  if ! echo "$cmd_scan" | grep -qE '\bswitch\b +(-c +|-C +)?main\b'; then
    blocked="git switch"
  fi
fi

# git checkout -b/-B <branch>  (branch creation).
if echo "$cmd_scan" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-b|-B)\b'; then
  blocked="git checkout -b"
fi

# git checkout --detach [<ref>]  /  git switch --detach|-d <ref>  — explicit
# detach. Fires independent of the positional arg: for `checkout --detach abc`
# the first post-keyword token is the flag, not the ref, so the arg-classifier
# below would miss it. The switch pattern also catches `git switch -d main`
# (a detach AT main), which the branch-only switch detector above lets through
# on the `main` allow-arm.
if echo "$cmd_scan" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-{1,2})detach\b'; then
  blocked="git checkout --detach"
fi
if echo "$cmd_scan" | grep -qE '\bgit\b[^;&|]*\bswitch\b +(--detach\b|-d\b)'; then
  blocked="git switch --detach"
fi

# git checkout <existing-branch>  — NOT a file restore (no `--`), arg is a real
# local branch ref, and not `main`. Extended: a non-branch arg that resolves to
# a commit-ish (sha / tag / origin/<branch> / HEAD~N / HEAD@{N}) DETACHES HEAD
# and is blocked too. A flag-prefixed detach (`checkout -f <sha>`, `-q <sha>`,
# `-p <sha>`, `-m <sha>`) is caught by re-scanning the post-`checkout` tokens:
# skip known safe short-flags, then classify the first positional. `-b`/`-B`
# are NOT skipped here (branch creation is already blocked above); `.`/`main`/`-`
# are left as ALLOW.
if echo "$cmd_scan" | grep -qE '\bgit\b[^;&|]*\bcheckout\b' \
   && ! echo "$cmd_scan" | grep -qE 'checkout\b[^;&|]*--'; then
  arg=$(echo "$cmd_scan" | sed -nE 's/.*\bcheckout\b +([^ ;&|]+).*/\1/p')
  # Flag-prefixed detach: skip leading safe short-flags to reach the first
  # positional (e.g. `checkout -f <sha>` -> classify `<sha>`).
  if echo "$arg" | grep -qE '^-[fqpm]$'; then
    rest=$(echo "$cmd_scan" | sed -nE 's/.*\bcheckout\b +(.*)/\1/p')
    # shellcheck disable=SC2086  # word-splitting is intentional here
    set -- $rest
    while [ $# -gt 0 ]; do
      case "$1" in
        -f|-q|-p|-m) shift ;;  # safe short-flag — skip to the next token
        *) arg="$1"; break ;;  # first positional
      esac
    done
  fi
  case "$arg" in
    ""|-b|-B|-f|--force|main|-|.) : ;;  # not a branch/detach we block
    --*) : ;;                           # a flag (e.g. --detach handled above)
    *)
      if git -C "$REPO" show-ref --verify --quiet "refs/heads/$arg"; then
        blocked="git checkout $arg"                        # branch switch
      elif git -C "$REPO" rev-parse --verify --quiet "$arg^{commit}" >/dev/null 2>&1; then
        blocked="git checkout $arg (detaches HEAD)"         # sha/tag/remote-ref/HEAD~N
      fi
      ;;
  esac
fi

[ -n "$blocked" ] || exit 0

# Only protect the on-main state. If the repo-root tree is already off main,
# the horse has bolted — don't trap the user trying to recover.
cur=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$cur" = main ] || exit 0

echo "BLOCKED: '$blocked' would move the SHARED repo-root tree off main / detach HEAD. The repo root is the canonical commit target for scripts/task.py and every concurrent VM session (all assume HEAD==main); switching branches or detaching HEAD here hijacks their commits and sweeps cross-session uncommitted edits into the wrong commit, and a detached repo-root HEAD crashes task_workflow's main-worktree resolver (incident 2026-06-01). Do feature/infra branch work in a worktree instead:
  bash scripts/new_worktree.sh .claude/worktrees/<name> <branch> && cd .claude/worktrees/<name>
To override deliberately, run the git command from inside a worktree (git -C .claude/worktrees/<name> ...)." >&2
exit 2
