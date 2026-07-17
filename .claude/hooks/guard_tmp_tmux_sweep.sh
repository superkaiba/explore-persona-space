#!/usr/bin/env bash
# PreToolUse(Bash) guard: block broad /tmp deletion SWEEPS that can destroy the
# fleet tmux socket dir /tmp/tmux-<uid>, and any DIRECT deletion of /tmp/tmux*
# (task #1474). Incident #1466 (2026-07-15T23:17:53Z): an improvised
# disk-pressure sweep — find /tmp -maxdepth 1 -mtime +2 ! -name 'claude-*'
# ! -name 'systemd-*' ! -name 'snap-*' -user "$(id -un)" -print0 |
# xargs -0 -r rm -rf — had no tmux-* exclusion, deleted /tmp/tmux-1001 with
# the live server socket inside, and split 39 sessions off the fleet.
#
# BLOCKS (exit 2 + stderr):
#   (i)   rm whose target is /tmp itself, a top-level /tmp glob whose literal
#         prefix can still match "tmux-*" (/tmp/*, /tmp/t*, /tmp/????-*), or
#         ANY /tmp/tmux* path (the protected asset — blocked outright,
#         recursive or not: rm /tmp/tmux-1001/default deletes the socket).
#   (ii)  find rooted at standalone /tmp (optionally quoted: "/tmp") with a
#         deletion action (-delete, -exec[dir] rm, a piped `xargs ... rm`, or
#         rm over $(find /tmp ...)) and NO tmux exclusion (`! -name 'tmux-*'`
#         / negated -path/-regex mentioning tmux / -prune together with tmux).
#   (iii) find rooted at /tmp/tmux* with a deletion action (exclusion
#         irrelevant — the root IS the asset).
#
# Deliberately conservative EVERYWHERE ELSE (false positives on the fleet's
# hourly narrow /tmp cleanups are worse than misses — when unsure, ALLOW):
#   - explicit non-tmux /tmp paths (rm -f /tmp/issue-<N>-lint-verdict.txt),
#     multi-file explicit lists, and deep globs whose FIRST component is
#     literal (/tmp/claude-1001/*/...) are ALLOWED;
#   - top-level globs whose literal prefix rules out tmux (/tmp/issue-*) ALLOWED;
#   - variable targets (rm -rf "$SCRATCH", trap "rm -rf '$TMP'" EXIT) ALLOWED —
#     variable indirection (X=/tmp; rm -rf $X/*) is a DOCUMENTED MISS;
#   - find rooted at a /tmp SUBDIR (find /tmp/issue-1474-scratch -delete) and
#     deletion-free finds (find /tmp -maxdepth 1 -type s) ALLOWED;
#   - units whose first command is grep|rg|echo|printf|ssh are SKIPPED:
#     verification greps quote these very shapes (a mid-quote `|` raw-split
#     otherwise false-blocks them), echo/printf cannot delete, and ssh
#     operates on a REMOTE /tmp (no fleet tmux server there). DOCUMENTED
#     MISSES, test-pinned: `echo /tmp/tmux-1001 | xargs rm -rf` and the
#     generic reader-pipeline sibling `ls /tmp | xargs rm -rf` (xargs-fed
#     targets are invisible to the rm token walk outside the find-arm);
#   - further DOCUMENTED MISSES (cooperative-agent model; prose rule + the
#     #1466 durable socket dir remain defense in depth): a cwd-relative sweep
#     (`cd /tmp && rm -rf *` — the glob token is not /tmp-rooted),
#     interpreter-level deletion (python shutil.rmtree('/tmp/...') — no rm
#     token), `mv` displacement of the socket dir (mv /tmp/tmux-1001 /x —
#     no deletion verb), and brace-expansion targets (/tmp/{a,b} — `{` is
#     not in the glob-char set the prefix classifier inspects);
#   - heredoc-bearing commands blanket-allowed (family precedent — commit
#     messages DESCRIBING this incident must not false-block);
#   - unparseable / ambiguous input allowed (fail-soft, exit 0).
#
# <!-- known limitation --> Detectors scan the RAW command string without
# stripping quoted arguments (guard family trade-off): a marker --note /
# commit -m string QUOTING a sweep in a non-heredoc, non-reader-leading
# command false-blocks (test-pinned deliberate). Remediation:
# `task.py post-marker --file <path.md>` / `git commit -F <file>` / heredoc.
#
# Escape hatch: EPM_ALLOW_TMP_SWEEP=1 — session env or inline prefix. The
# sanctioned deliberate use is flipping the fleet to the durable
# ~/.tmux-sockets dir (see .claude/rules/background-automation.md § tmux
# socket-dir contract) or a human-directed /tmp purge.
#
# Contract: PreToolUse JSON on stdin; exit 0 allow; exit 2 block (stderr fed
# back to Claude). Self-test: bash .claude/hooks/guard_tmp_tmux_sweep.sh --self-test
set -u

# find rooted at standalone /tmp (within one pipeline stage: [^|;&]* cannot
# cross a separator, and units are pre-split on ;/&&/||/&/newline below).
# The root tolerates optional surrounding quotes (find "/tmp" -delete).
FIND_TMP_ROOT_ERE='(^|[[:space:]])([^[:space:]]*/)?find[[:space:]]+([^|;&]*[[:space:]])?['\''"]?/tmp/?['\''"]?([[:space:]]|$)'
# Asset-rooted arm: the middle group admits only dash-initial (option) tokens —
# find ROOTS precede expressions, so /tmp/tmux* must be the first non-option
# argument; a loose middle group would anchor on a `-path '/tmp/tmux-*'`
# option VALUE and false-block the A17 prune carve-out.
FIND_TMUX_ROOT_ERE='(^|[[:space:]])([^[:space:]]*/)?find[[:space:]]+((-[^[:space:]]*)[[:space:]]+)*['\''"]?/tmp/tmux[^[:space:]]*'
DELETE_ERE='(^|[[:space:]])-delete([[:space:]]|$)'
EXEC_RM_ERE='-exec(dir)?[[:space:]]+(sudo[[:space:]]+)?([^[:space:]]*/)?rm([[:space:]]|$)'
XARGS_RM_ERE='\bxargs([[:space:]]|$)[^|;&]*\brm([[:space:]]|$)'
RM_CMDSUB_FIND_ERE='\brm([[:space:]]|$)[^|;&]*\$\([[:space:]]*find[[:space:]]+/tmp(/|[[:space:]])'
# tmux exclusion: NEGATED name/path/regex predicate whose pattern mentions tmux.
TMUX_EXCLUDE_ERE='(!|-not)[[:space:]]+-i?(name|path|regex|wholename)[[:space:]]+[^|;&[:space:]]*tmux'

BLOCK_WHAT=""
RM_REASON=""

strip_quotes() { # verbatim guard_log_dump.sh: iterative leading/trailing quote trim
  sq_val=$1
  while :; do
    case "$sq_val" in
      \'* | \"*) sq_val=${sq_val#?} ;;
      *\' | *\") sq_val=${sq_val%?} ;;
      *) break ;;
    esac
  done
}

rm_target_blockworthy() {
  # $1: quote-stripped rm argument. rc 0 = block-worthy (sets RM_REASON).
  local t=$1
  while [ "${t%/}" != "$t" ]; do t=${t%/}; done # /tmp// -> /tmp
  case "$t" in
    /tmp) RM_REASON="deletes /tmp itself"; return 0 ;;
    /tmp/*) : ;;
    *) return 1 ;; # not /tmp-rooted (incl. $VAR targets)
  esac
  local comp=${t#/tmp/}
  comp=${comp%%/*} # first component under /tmp/
  case "$comp" in
    tmux*) RM_REASON="targets the protected /tmp/tmux* socket dir"; return 0 ;;
  esac
  case "$comp" in
    *[\*\?\[]*) : ;;
    *) return 1 ;; # literal non-tmux first component: benign
  esac
  local pre=${comp%%[\*\?\[]*} # literal prefix before the first glob char
  case "tmux-" in
    "$pre"*) RM_REASON="top-level /tmp glob can match tmux-*"; return 0 ;;
  esac
  case "$pre" in
    tmux*) RM_REASON="top-level /tmp glob can match tmux-*"; return 0 ;;
  esac
  return 1
}

unit_has_deletion() {
  printf '%s' "$1" | grep -qE "$DELETE_ERE" && return 0
  printf '%s' "$1" | grep -qE "$EXEC_RM_ERE" && return 0
  printf '%s' "$1" | grep -qE "$XARGS_RM_ERE" && return 0
  return 1
}

unit_has_tmux_exclusion() {
  printf '%s' "$1" | grep -qE "$TMUX_EXCLUDE_ERE" && return 0
  case "$1" in *-prune*) case "$1" in *tmux*) return 0 ;; esac ;; esac
  return 1
}

check_unit() {
  local unit="$1" first="" tok mode=""
  # Leading reader/remote skip (see header FP budget).
  set -f
  # shellcheck disable=SC2086
  set -- $unit
  set +f
  for tok in "$@"; do
    case "$tok" in *=*) continue ;; esac
    strip_quotes "$tok"
    first=${sq_val##*/}
    break
  done
  case "$first" in grep | rg | echo | printf | ssh) return 0 ;; esac

  # find-arm: asset-rooted first (exclusion cannot save it), then /tmp-rooted.
  if printf '%s' "$unit" | grep -qE "$FIND_TMUX_ROOT_ERE"; then
    if unit_has_deletion "$unit"; then
      BLOCK_WHAT="find rooted at /tmp/tmux* with a deletion action"
      return 1
    fi
  fi
  if printf '%s' "$unit" | grep -qE "$FIND_TMP_ROOT_ERE"; then
    if unit_has_deletion "$unit" && ! unit_has_tmux_exclusion "$unit"; then
      BLOCK_WHAT="find sweep rooted at /tmp with no tmux-* exclusion"
      return 1
    fi
  fi
  if printf '%s' "$unit" | grep -qE "$RM_CMDSUB_FIND_ERE"; then
    if ! unit_has_tmux_exclusion "$unit"; then
      BLOCK_WHAT='rm over $(find /tmp ...) with no tmux-* exclusion'
      return 1
    fi
  fi

  # rm-arm: token walk; targets attributed to rm until a pipe token.
  set -f
  # shellcheck disable=SC2086
  set -- $unit
  set +f
  for tok in "$@"; do
    strip_quotes "$tok"
    case "$sq_val" in
      rm | */rm) mode=rm; continue ;;
      *\|*) mode=""; continue ;; # pipe boundary: later stage, new command
    esac
    [ "$mode" = rm ] || continue
    case "$sq_val" in -*) continue ;; esac
    if rm_target_blockworthy "$sq_val"; then
      BLOCK_WHAT="rm target '$sq_val' — $RM_REASON"
      return 1
    fi
  done
  return 0
}

check_cmd() {
  local cmd="$1"
  case "$cmd" in *EPM_ALLOW_TMP_SWEEP=1*) return 0 ;; esac
  # Cheap pre-filters (this hook runs on EVERY Bash call).
  case "$cmd" in */tmp*) : ;; *) return 0 ;; esac
  printf '%s' "$cmd" | grep -qE '\brm([[:space:]]|$)|(^|[[:space:]])-delete([[:space:]]|$)' \
    || return 0
  # Heredoc blanket-allow (family precedent, load-bearing for commit messages).
  case "$cmd" in *'<<'*) return 0 ;; esac
  # Line-continuation normalization + redirection-aware unit split, verbatim
  # the guard_piped_git_push.sh engine.
  cmd=$(printf '%s' "$cmd" | sed -zE 's/\\\r?\n/ /g')
  local units unit lead
  units=$(printf '%s' "$cmd" \
    | sed -E 's/\|&/|/g' \
    | sed -E 's/[0-9]*>&[0-9]*//g; s/&>>?[^[:space:]]*//g' \
    | sed -E 's/&&/\n/g; s/\|\|/\n/g; s/;/\n/g; s/&/\n/g')
  while IFS= read -r unit; do
    unit=$(printf '%s' "$unit" | sed -E 's/[[:space:]]#.*$//')
    lead=$(printf '%s' "$unit" | sed -E 's/^[[:space:]]+//')
    case "$lead" in \#* | '') continue ;; esac
    check_unit "$unit" || return 1
  done <<<"$units"
  return 0
}

run_self_test() {
  # Same run_case harness as guard_piped_git_push.sh (jq -n --arg c ... piped
  # into the script; env -u EPM_ALLOW_TMP_SWEEP for deny cases). Cases = the
  # plan §5 B/A acceptance tables + the review-round boundary rows. No
  # filesystem fixtures needed — this guard never stats paths (pure string
  # classification).
  local SCRIPT FAILED=0
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

  run_case() {
    local desc="$1" expect="$2" cmdstr="$3" envflag="${4:-}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | EPM_ALLOW_TMP_SWEEP=1 bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_TMP_SWEEP bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- plan §5: must BLOCK (exit 2) ---
  run_case "B1 the #1466 incident, verbatim" 2 \
    'find /tmp -maxdepth 1 -mtime +2 ! -name '\''claude-*'\'' ! -name '\''systemd-*'\'' ! -name '\''snap-*'\'' -user "$(id -un)" -print0 | xargs -0 -r rm -rf'
  run_case "B2 find -delete" 2 'find /tmp -mtime +2 -delete'
  run_case "B3 find -exec rm" 2 'find /tmp -maxdepth 1 -type d -exec rm -rf {} +'
  run_case "B4 find | xargs rm (no -0)" 2 'find /tmp -name '\''*.tmp'\'' | xargs rm'
  run_case "B5 bare top-level glob" 2 'rm -rf /tmp/*'
  run_case "B6 /tmp itself" 2 'rm -rf /tmp'
  run_case "B7 trailing-slash form" 2 'rm -rf /tmp/'
  run_case "B8 asset glob, outright" 2 'rm -rf /tmp/tmux-*'
  run_case "B9 asset literal, outright" 2 'rm -rf /tmp/tmux-1001'
  run_case "B10 socket file, non-recursive" 2 'rm /tmp/tmux-1001/default'
  run_case "B11 sudo prefix" 2 'sudo rm -rf /tmp/*'
  run_case "B12 asset-rooted find" 2 'find /tmp/tmux-1001 -type s -delete'
  run_case "B13 later && unit" 2 'cd /workspace && find /tmp -mtime +1 -delete'
  run_case "B14 glob prefix still tmux-capable" 2 'rm -rf /tmp/t*'
  run_case "B15 cmd-subst feeding rm" 2 'rm -rf $(find /tmp -mtime +2)'
  run_case "B16 non-tmux exclusion does not count" 2 "find /tmp ! -name 'claude-*' -delete"
  run_case "B17 pinned deliberate FP: quoted sweep in --note" 2 \
    'uv run python scripts/task.py post-marker 1474 epm:progress --note '\''ran find /tmp -mtime +2 | xargs rm -rf'\'''
  run_case "B18 trailing-slash find root" 2 'find /tmp/ -mtime +2 -delete'
  run_case "B19 -execdir deletion form" 2 'find /tmp -maxdepth 1 -execdir rm -rf {} \;'
  run_case "B20 xargs -I{} deletion form" 2 \
    'find /tmp -maxdepth 1 -print0 | xargs -0 -I{} rm -rf {}'
  run_case "B21 quoted find root" 2 'find "/tmp" -mtime +2 -delete'
  run_case "B22 backtick cmd-subst (rm-arm token walk)" 2 'rm -rf `find /tmp -mtime +2`'

  # --- plan §5: must ALLOW (exit 0) ---
  run_case "A1 explicit path" 0 'rm -f /tmp/issue-1474-lint-verdict.txt'
  run_case "A2 explicit multi-file" 0 \
    'rm -f /tmp/step9c-junit-issue-1474.xml /tmp/step9c-rc-issue-1474'
  run_case "A3 explicit dir" 0 'rm -rf /tmp/issue-1474-lint-gate-tree'
  run_case "A4 variable target" 0 \
    'SCRATCH=/tmp/issue-1474-postmerge-scratch; rm -rf "$SCRATCH"'
  run_case "A5 deep glob, literal first component" 0 'rm -f /tmp/claude-1001/*/*/tasks/*.output'
  run_case "A6 top-level glob, tmux-impossible prefix" 0 'rm -rf /tmp/issue-*'
  run_case "A7 the remediated incident" 0 \
    'find /tmp -maxdepth 1 -mtime +2 ! -name '\''tmux-*'\'' ! -name '\''claude-*'\'' -user "$(id -un)" -print0 | xargs -0 -r rm -rf'
  run_case "A8 excluded find -delete" 0 \
    "find /tmp -maxdepth 1 -mtime +2 ! -name 'tmux-*' -delete"
  run_case "A9 deletion-free probe" 0 'find /tmp -maxdepth 1 -type s'
  run_case "A10 subdir-rooted find" 0 'find /tmp/issue-1474-scratch -delete'
  run_case "A11 no deletion verbs" 0 'ls /tmp && df -h /tmp'
  run_case "A12 leading-grep verification unit" 0 \
    'grep -rnE '\''find /tmp|rm -rf? /tmp'\'' scripts/'
  run_case "A13 heredoc commit quoting a sweep" 0 'git commit -m "$(cat <<EOF
never run find /tmp -mtime +2 | xargs rm -rf without a tmux exclusion
EOF
)"'
  run_case "A14 inline escape hatch" 0 'EPM_ALLOW_TMP_SWEEP=1 find /tmp -mtime +2 -delete'
  run_case "A15 mktemp-cleanup trap" 0 'trap "rm -rf '\''$TMP'\''" EXIT'
  run_case "A16 mv is not deletion" 0 'mv /tmp/foo /tmp/bar'
  run_case "A17 prune-based carve-out" 0 \
    "find /tmp -path '/tmp/tmux-*' -prune -o -mtime +2 -delete"
  run_case "A18 leading-echo unit" 0 \
    'echo '\''find /tmp -mtime +2 | xargs rm -rf'\'' >> notes.md'
  run_case "A19 non-/tmp path containing tmp" 0 'rm -rf /workspace/tmp/*'
  run_case "A20 remote /tmp (ssh-leading skip)" 0 'ssh pod-1474 '\''rm -rf /tmp/hf_stage'\'''
  run_case "A21 forensic probe, no deletion" 0 'find /tmp -maxdepth 1 -mtime +12'
  run_case "A22 filer temp cleanup" 0 'rm -f /tmp/wf-fix-body-guard.md'
  run_case "A23 pinned MISS: echo pipeline" 0 'echo /tmp/tmux-1001 | xargs rm -rf'
  run_case "A24 -not name exclusion variant" 0 \
    "find /tmp -maxdepth 1 -mtime +2 -not -name 'tmux-*' -delete"
  run_case "A25 pinned MISS: ls reader pipeline" 0 'ls /tmp | xargs rm -rf'
  run_case "A26 env escape hatch on B2 shape" 0 'find /tmp -mtime +2 -delete' env

  # malformed stdin JSON -> fail-soft allow.
  local rc=0
  printf 'not-json' | env -u EPM_ALLOW_TMP_SWEEP bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A27 malformed stdin JSON"
  else
    echo "FAIL (got exit $rc, want 0): A27 malformed stdin JSON"
    FAILED=1
  fi

  if [ "$FAILED" = 1 ]; then
    echo "self-test: FAIL" >&2
    return 1
  fi
  echo "self-test: PASS (all cases)"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  run_self_test
  exit $?
fi

# Session-env escape hatch.
_allow=$(printf '%s' "${EPM_ALLOW_TMP_SWEEP:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in
  1 | true | yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

if ! check_cmd "$cmd"; then
  {
    echo "BLOCKED: broad /tmp deletion sweep (${BLOCK_WHAT})."
    echo "/tmp/tmux-<uid> holds the LIVE fleet tmux server sockets — an unexcluded"
    echo "age sweep deleted it on 2026-07-15 and split 39 sessions off the fleet"
    echo "(#1466). Fix: add ! -name 'tmux-*' to the find expression, or name"
    echo "explicit non-tmux paths instead of /tmp globs. For marker-note /"
    echo "commit-message text that merely MENTIONS a sweep, use"
    echo "task.py post-marker --file <path.md> / git commit -F <file> (or a"
    echo "heredoc). Deliberate override (e.g. the sanctioned flip to the durable"
    echo "~/.tmux-sockets dir, .claude/rules/background-automation.md § tmux"
    echo "socket-dir contract): prefix with EPM_ALLOW_TMP_SWEEP=1."
  } >&2
  exit 2
fi
exit 0
