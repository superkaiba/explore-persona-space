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
#
# <!-- known limitation -->
# Every detector scans the RAW command string — the guard does NOT strip
# quoted arguments before parsing. A quoted git-verb literal buried in
# ANOTHER command's argument therefore trips the guard: e.g.
# `task.py post-marker <N> epm:X --note "... git switch ..."` is blocked
# because the note text matches `git ... switch`. The workaround is to pass
# such note text via `--file <path.md>` instead of `--note`. A quote-strip
# pre-pass was tried (round 1 of #796) and reverted: stripping quoted spans
# BEFORE parsing silently hid REAL quoted git refs (`git checkout "HEAD~1"`,
# `git switch "main"`) from the detectors — a leak of the exact class this
# guard exists to block, and a false positive on quoted return-to-main. A
# shell-syntax-aware strip is not safe to do in a bash regex, so the raw-scan
# behavior (correct on git refs, over-eager on note-text literals) is the
# deliberate trade-off. See #796 round-2 report.
#
# Compound-command parsing is a best-effort CLAUSE SPLIT (#804): the command is
# split on `;` / `&&` / `||` / `|` / `&` / raw newline (two-char separators
# matched first) into clauses, each classified independently so a later
# safe/return-to-main clause can no longer mask an earlier dangerous one; a
# `cd <worktree|/tmp>` latch propagates ONLY across `&&` (where bash GUARANTEES
# the `cd` succeeded before the RHS runs, so the cwd persists forward). The
# latch does NOT propagate across:
#   - `;` (SEQ): bash runs the RHS regardless of the `cd` exit code; a FAILED
#     `cd` (e.g. a missing target) leaves the cwd unchanged (repo root), so the
#     git clause runs off-worktree. Fail-closed (#804 round 2): reset the latch
#     on `;` rather than trust a `cd` we cannot prove succeeded.
#   - a raw NEWLINE (NL): a multi-line command runs each line unconditionally,
#     exactly like `;` — bash does NOT short-circuit on a `cd` exit code across
#     a newline, so a FAILED `cd` on line N leaves line N+1 running in the
#     unchanged cwd (repo root). Treated as `;`: reset the latch on NL. Before
#     #804 round 3 raw newlines produced records with no leading sentinel, so
#     `sep` inherited the STALE value (an `AND` after a `&&` clause) and the
#     `cd` latch leaked ACROSS the newline — `cd <missing> && git status\n
#     git switch feature` returned rc=0. The sed pre-pass now emits an explicit
#     `NL` sentinel for each raw newline so it resets the latch like `;`.
#   - `||` (OR): the RHS runs ONLY on `cd` FAILURE, cwd unchanged (repo root).
#   - `|` (PIPE): each pipeline segment is its own subshell, so the LHS `cd`'s
#     cwd change dies with it and the git segment runs in the parent's cwd.
#   - `&` (BG): the LHS runs in a background subshell (its own cwd) while the
#     RHS runs in the foreground parent's UNCHANGED cwd (repo root); BOTH
#     execute, so a `git switch feature & git switch main` runs the dangerous
#     LHS in the repo-root tree while the allow-arm RHS masks it.
# The split is NOT a full shell parse: command substitution `$(git switch ...)`,
# here-docs, and separators embedded inside a quoted arg are not handled (a
# quoted `;`/`|`/`&` is treated as a real separator, the same raw-scan
# trade-off as the `--note` literal above). A mis-split of that class fails
# CLOSED (blocks), the safe direction for a guard.
#
# Bash line continuations (`\<CR?><NL>`) are normalized to a single space at the
# top of the guard before any parsing (#804 round 4). Bash strips a
# backslash-newline before execution, joining the two physical lines into one
# logical command, so `git \<NL>checkout -bfoo` runs as `git checkout -bfoo`;
# without the normalization the raw-scan guard saw `git ` and `checkout -bfoo`
# as separate lines (the newline splitter fired) and missed the joined
# `git checkout` invocation entirely (a leak of the exact class this guard
# blocks). The normalization is a no-op on any command without a `\<NL>`.
set -u

REPO=/home/thomasjiralerspong/explore-persona-space

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

# Normalize Bash line continuations (`\<CR?><NL>` -> space) BEFORE any parsing.
# Bash strips these pre-execution (joining the two physical lines into one), but
# the raw-scan guard would otherwise see `git ` and `checkout -bfoo` as separate
# lines (the newline splitter fires) and miss the joined `git checkout -bfoo`
# invocation. #804 round 4 fix for `guard-backslash-continuation-bypass`. The
# `sed -zE` uses NUL-delimited whole-input so `\n` is literal; `\\\r?\n` matches
# a backslash + optional CR + newline, replaced by a single space.
cmd=$(printf '%s' "$cmd" | sed -zE 's/\\\r?\n/ /g')

# Only consider git checkout/switch invocations at all.
echo "$cmd" | grep -qE '\bgit\b.*\b(checkout|switch)\b' || exit 0

# Split the raw command into (separator, clause) pairs, PRESERVING which
# separator precedes each clause. Two-char separators (&& ||) are matched
# BEFORE the single-char ones (; | &) so `&&`/`||` are not mis-split into two
# single-char clauses (the `&&` substitution runs before the single `&` rule,
# so a `&&` is consumed as AND and never re-matched as a bare `&`). Each
# separator run is replaced by a newline + a \x01-delimited sentinel token
# (START | SEQ | AND | OR | PIPE | BG | NL); the first clause carries the
# implicit START. Best-effort: a separator inside a quoted arg is treated as a
# real separator (same trade-off as the raw-scan known-limitation above).
#
# `sed -z` treats the WHOLE input as one NUL-delimited record so a literal
# newline in $1 is matchable. The raw-NEWLINE -> `\n\x01NL\x01` substitution
# runs FIRST — before any separator substitution has inserted its own `\n` —
# so it only tags the raw input newlines and can NOT re-mangle the structural
# `\n` the &&/||/;/|/& rules insert afterwards (none of those rules matches
# `\n` or the already-inserted `\x01NL\x01`). Without the NL sentinel a raw
# newline produced a record with NO leading sentinel, so awk's `sep` inherited
# the STALE value from the previous line (an `AND` after a `&&` clause) and the
# `cd` scope latch leaked across the newline (#804 round 3).
split_and_label() {
  printf '%s' "$1" \
    | sed -zE 's/\n/\n\x01NL\x01/g; s/\|\|/\n\x01OR\x01/g; s/&&/\n\x01AND\x01/g; s/;/\n\x01SEQ\x01/g; s/\|/\n\x01PIPE\x01/g; s/&/\n\x01BG\x01/g' \
    | awk 'BEGIN{RS="\n"; sep="START"}
           { line=$0
             if (match(line, /^\x01(OR|AND|SEQ|PIPE|BG|NL)\x01/)) {
               sep=substr(line, 2, RLENGTH-2); line=substr(line, RLENGTH+1)
             }
             gsub(/^[ \t]+|[ \t]+$/, "", line)
             if (length(line)) print sep "\t" line }'
}

# Classify a SINGLE clause. Echoes the `blocked` reason (empty string = allow).
# This is the pre-#804 whole-command detector body, applied per-clause: `$c`
# holds one clause. The `[^;&|]*` anchors inside the detectors are no-ops
# per-clause (a clause has no separators) but are kept verbatim so the function
# stays correct even if fed a whole string — zero behavior change.
classify_clause() {
  local c="$1"
  local blocked=""

  # git switch <branch> / git switch -c <branch>  (switch is branch-only).
  # Allow only `git switch main` (bare or quoted — the arg regex tolerates an
  # optional surrounding quote, so `git switch "main"` also passes the allow-arm).
  # The allow-arm ANCHORS `main` to the FULL switch arg: `main` must be followed
  # by an optional trailing quote AND then end-of-string or a shell delimiter
  # (whitespace / `;` / `&` / `|`). A bare `\bmain\b` word boundary would ALSO
  # match before a `-` / `/` / `.` (all non-word chars), so `git switch
  # main-adjacent` / `main/foo` / `main.x` (and their quoted forms) would slip
  # through the allow-arm and LEAK a branch-switch off main. `main_x` still
  # blocks: `_` is a word char so `\bmain` never matched there either, but the
  # explicit terminator makes the intent unambiguous. Concern id:
  # switch-main-prefix-allowarm-leak (#796 round 3).
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bswitch\b'; then
    if ! echo "$c" | grep -qE '\bswitch\b +(-c +|-C +)?["'"'"']?main["'"'"']?( *($|[;&|]))'; then
      blocked="git switch"
    fi
  fi

  # git checkout -b/-B <branch>  (branch creation). The trailing class matches a
  # space (bare `-b feature`), end-of-clause (bare `-b`), OR a glued branch-name
  # char (`-bfoo`, `-B123`, `-b-x`, `-b.y`, `-b/z`) — the `(-b|-B)\b`
  # word-boundary form missed the glued `-bfoo` (`f` is a word char, no boundary
  # after `b`) and leaked branch creation off main. Concern id:
  # checkout-glued-shortflag-b-leak (#804 / #796 round 3).
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-b|-B)([[:alnum:]_./[:space:]-]|$)'; then
    blocked="git checkout -b"
  fi

  # git checkout --detach [<ref>]  /  git switch --detach|-d <ref>  — explicit
  # detach. Fires independent of the positional arg: for `checkout --detach abc`
  # the first post-keyword token is the flag, not the ref, so the arg-classifier
  # below would miss it. The switch pattern also catches `git switch -d main`
  # (a detach AT main), which the branch-only switch detector above lets through
  # on the `main` allow-arm.
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b +(-{1,2})detach\b'; then
    blocked="git checkout --detach"
  fi
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bswitch\b +(--detach\b|-d\b)'; then
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
  if echo "$c" | grep -qE '\bgit\b[^;&|]*\bcheckout\b' \
     && ! echo "$c" | grep -qE 'checkout\b[^;&|]*--'; then
    arg=$(echo "$c" | sed -nE 's/.*\bcheckout\b +([^ ;&|]+).*/\1/p')
    # Flag-prefixed detach: skip leading safe short-flags to reach the first
    # positional (e.g. `checkout -f <sha>` -> classify `<sha>`).
    if echo "$arg" | grep -qE '^-[fqpm]$'; then
      rest=$(echo "$c" | sed -nE 's/.*\bcheckout\b +(.*)/\1/p')
      # shellcheck disable=SC2086  # word-splitting is intentional here
      set -- $rest
      while [ $# -gt 0 ]; do
        case "$1" in
          -f|-q|-p|-m) shift ;;  # safe short-flag — skip to the next token
          *) arg="$1"; break ;;  # first positional
        esac
      done
    fi
    # Strip a single layer of surrounding quotes so a QUOTED ref classifies as
    # its bare form: `git checkout "HEAD~1"` -> HEAD~1 (detaches -> block),
    # `git checkout "main"` -> main (allow). Quoted refs are shell-equivalent to
    # unquoted ones; without this strip the quoted arg would either miss the
    # `main` allow-arm (false positive) or, before the round-1 quote-strip was
    # reverted, be erased entirely (leak). Only trailing/leading `"` or `'`.
    arg=${arg#[\"\']}
    arg=${arg%[\"\']}
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

  echo "$blocked"
}

# Drive classify_clause over the (separator, clause) pairs. A `cd <worktree|/tmp>`
# latches `scoped` forward ONLY across `&&` (bash GUARANTEES the `cd` succeeded
# before the RHS runs there, so the cwd persists), so a git clause after it runs
# in the scoped cwd and is allowed. The latch RESETS across every OTHER separator
# — `;` (SEQ), `||` (OR), `|` (PIPE), `&` (BG), and a raw newline (NL) — where
# bash does NOT guarantee the `cd` took effect for the following clause (verified
# bash semantics 2026-07-01: `cd X && pwd` prints X; `cd X ; pwd` prints X ONLY on
# `cd` success — a FAILED `cd`, e.g. a missing target, leaves the ORIGINAL cwd —
# and `cd X || pwd` / `cd X | pwd` / `cd X & pwd` / a `cd X<newline>pwd` all print
# the ORIGINAL cwd on `cd` failure). Resetting on `;` / NL fails CLOSED (#804
# rounds 2/3): the guard cannot prove a `;`- or newline-preceding `cd` succeeded,
# so it declines to scope across it. The first blocking clause wins.
scoped=0
blocked=""
while IFS=$'\t' read -r sep clause; do
  # Reset the latch unless the separator BEFORE this clause is && — a `cd`
  # only reliably scopes a following git clause when bash guarantees it ran
  # first (the && short-circuit). ; / || / | / & / a raw newline (NL) do NOT
  # carry the latch (NL is not AND, so this consolidated check resets it).
  if [ "$sep" != AND ]; then
    scoped=0
  fi

  # A `cd` into a worktree / /tmp latches scope forward ONLY across a following
  # `&&` clause. Latch and continue — this clause runs the `cd`, not a git
  # command, and it must NOT scope EARLIER clauses (those were classified
  # before it).
  if echo "$clause" | grep -qE 'cd +[^;&|]*\.claude/worktrees/' \
     || echo "$clause" | grep -qE 'cd +/tmp/'; then
    scoped=1
    continue
  fi
  [ "$scoped" -eq 1 ] && continue          # this clause runs in a scoped cwd

  # `git -C <path>` scopes ONLY this clause (per-invocation) — allow it.
  echo "$clause" | grep -qE '\bgit +-C +' && continue

  # not a git checkout/switch clause at all -> skip
  echo "$clause" | grep -qE '\bgit\b.*\b(checkout|switch)\b' || continue

  reason=$(classify_clause "$clause")
  if [ -n "$reason" ]; then blocked="$reason"; break; fi   # first block wins
done < <(split_and_label "$cmd")

[ -n "$blocked" ] || exit 0

# Only protect the on-main state. If the repo-root tree is already off main,
# the horse has bolted — don't trap the user trying to recover.
cur=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ "$cur" = main ] || exit 0

echo "BLOCKED: '$blocked' would move the SHARED repo-root tree off main / detach HEAD. The repo root is the canonical commit target for scripts/task.py and every concurrent VM session (all assume HEAD==main); switching branches or detaching HEAD here hijacks their commits and sweeps cross-session uncommitted edits into the wrong commit, and a detached repo-root HEAD crashes task_workflow's main-worktree resolver (incident 2026-06-01). Do feature/infra branch work in a worktree instead:
  bash scripts/new_worktree.sh .claude/worktrees/<name> <branch> && cd .claude/worktrees/<name>
To override deliberately, run the git command from inside a worktree (git -C .claude/worktrees/<name> ...)." >&2
exit 2
