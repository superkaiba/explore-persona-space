#!/usr/bin/env bash
# PreToolUse(Bash) guard: block whole-log dumps into the conversation context.
#
# CLAUDE.md § Context hygiene: "Never dump giant logs into tool output —
# grep -iE 'error|traceback|killed|OOM' / tail -50, never cat a multi-MB log
# (it re-enters context next turn)." This hook enforces that mechanically:
# a Bash command that would page a large file into context — `cat`, `head`/
# `tail` with a big line count (> MAX_LINES), or a wide-range `sed -n` print —
# over a log-shaped file (*.log anywhere, anything under a logs/ dir, *.jsonl
# in known log dirs) or any single EXISTING file > MAX_BYTES is BLOCKED with
# a pointer to the grep / tail-50 recipe.
#
# Deliberately conservative (false positives on normal small reads are worse
# than misses — when unsure, ALLOW):
#   - piped commands (`cat big.log | grep error`) and redirected output
#     (`cat big.log > /tmp/x`) are ALLOWED — downstream filter/sink, not a
#     context dump;
#   - `tail -50` / `head -n 100` (<= MAX_LINES) and default head/tail ALLOWED;
#   - single-line `sed -n '5p' file` reads are ALLOWED;
#   - unparseable / ambiguous commands are ALLOWED (fail-soft).
#
# Escape hatch: EPM_ALLOW_LOG_DUMP=1 — honored both as session env and as an
# inline prefix on the command itself (`EPM_ALLOW_LOG_DUMP=1 cat big.log`).
# FAIL-LOUD block-with-message only (exit 2 + stderr); the command is never
# silently rewritten.
#
# Contract: reads the PreToolUse JSON on stdin, exits 0 to allow, exits 2
# (blocking, stderr fed back to Claude) to refuse.
#
# Self-test: bash .claude/hooks/guard_log_dump.sh --self-test
set -u

MAX_BYTES=262144   # 256 KB — any single existing file larger than this is dump-sized
MAX_LINES=200      # head/tail line counts above this count as a dump

BLOCK_FILE=""
BLOCK_VERB=""

is_logish() {
  # Log-shaped by NAME: *.log anywhere; anything under a logs/ dir;
  # *.jsonl only in known log dirs (/tmp, /workspace/logs, wandb) — so e.g.
  # tasks/<N>/events.jsonl small reads stay allowed unless the file is big.
  case "$1" in
    *.log | *.log.[0-9]*) return 0 ;;
    logs/* | */logs/*) return 0 ;;
    *.jsonl)
      case "$1" in
        /tmp/* | /workspace/logs/* | wandb/* | */wandb/*) return 0 ;;
      esac
      return 1
      ;;
  esac
  return 1
}

is_big() {
  # Big by SIZE: existing regular file > MAX_BYTES (relative paths resolve
  # against the session cwd the hook runs in). Unknown/missing -> not big.
  [ -f "$1" ] || return 1
  local sz
  sz=$(stat -c %s -- "$1" 2>/dev/null) || return 1
  [ "$sz" -gt "$MAX_BYTES" ]
}

# Evaluate the accumulated state for one dump verb. Uses/clears the globals
# set by check_cmd's token walk. Returns 1 (and sets BLOCK_*) on a violation.
pending_check() {
  [ -n "$mode" ] || return 0
  [ "${#files[@]}" -gt 0 ] || return 0
  local dumpshape=0
  case "$mode" in
    cat) dumpshape=1 ;;
    head | tail)
      if [ "${nlines:-0}" -gt "$MAX_LINES" ] 2>/dev/null; then dumpshape=1; fi
      ;;
    sed)
      if [ "$sed_n" = 1 ] && [ "$sedrange_big" = 1 ]; then dumpshape=1; fi
      ;;
  esac
  [ "$dumpshape" = 1 ] || return 0
  local f
  for f in "${files[@]}"; do
    f=${f#\'}; f=${f%\'}; f=${f#\"}; f=${f%\"}
    if is_logish "$f" || is_big "$f"; then
      BLOCK_FILE="$f"
      BLOCK_VERB="$mode"
      return 1
    fi
  done
  return 0
}

check_cmd() {
  local cmd="$1"

  # Inline escape hatch: EPM_ALLOW_LOG_DUMP=1 anywhere in the command.
  case "$cmd" in *EPM_ALLOW_LOG_DUMP=1*) return 0 ;; esac

  # Fast path: no dump-shaped verb present at all.
  echo "$cmd" | grep -qE '(^|[;&([:space:]])(cat|head|tail|sed)([[:space:]]|$)' || return 0

  # Piped or redirected output -> downstream filter / file sink, not a
  # context dump (`2>/dev/null` also lands here; conservative = allow).
  case "$cmd" in *\|* | *\>*) return 0 ;; esac

  # Heredocs: the body is already inside the command string, so "dumping" it
  # cannot add anything new to context, and the canonical `python - <<'PY'`
  # recipes must never false-positive. Conservative = allow.
  case "$cmd" in *'<<'*) return 0 ;; esac

  # Token walk. set -f: no glob expansion while splitting the command string.
  mode=""; nlines=0; sed_n=0; sedrange_big=0; await_n=0; files=()
  local tok st a b
  set -f
  # shellcheck disable=SC2086
  set -- $cmd
  set +f
  for tok in "$@"; do
    case "$tok" in
      cat | */cat | head | */head | tail | */tail | sed | */sed)
        pending_check || return 1
        mode="${tok##*/}"; nlines=0; sed_n=0; sedrange_big=0; await_n=0; files=()
        continue
        ;;
      ';' | '&&' | '||' | '&')
        # Standalone command separator: what follows belongs to a DIFFERENT
        # command — evaluate the accumulated verb, then stop attributing
        # later arguments to it (else `cat notes.md && rm logs/train.log`
        # false-positives on the rm target). NOTE: the '||' arm is currently
        # unreachable — the *\|* pipe-allow fast path above returns first for
        # any command containing '|' — kept so the separator walk stays
        # correct if that fast path is ever narrowed.
        pending_check || return 1
        mode=""; nlines=0; sed_n=0; sedrange_big=0; await_n=0; files=()
        continue
        ;;
    esac
    [ -n "$mode" ] || continue
    case "$mode" in
      cat)
        case "$tok" in
          -*) : ;;
          *) files+=("$tok") ;;
        esac
        ;;
      head | tail)
        if [ "$await_n" = 1 ]; then
          nlines=$(printf '%s' "$tok" | tr -dc '0-9')
          : "${nlines:=0}"
          await_n=0
          continue
        fi
        case "$tok" in
          -n) await_n=1 ;;
          -n* | --lines=*) nlines=$(printf '%s' "$tok" | tr -dc '0-9'); : "${nlines:=0}" ;;
          -[0-9]*) nlines=$(printf '%s' "$tok" | tr -dc '0-9'); : "${nlines:=0}" ;;
          -*) : ;;  # -f, -q, -c (byte mode), etc. — conservative: not line dumps
          *) files+=("$tok") ;;
        esac
        ;;
      sed)
        case "$tok" in
          -n | -n*) sed_n=1; continue ;;
          -e | -*) continue ;;
        esac
        st=${tok#\'}; st=${st%\'}; st=${st#\"}; st=${st%\"}
        if printf '%s' "$st" | grep -qE '^[0-9]+,([0-9]+|\$)p$'; then
          a=${st%%,*}; b=${st#*,}; b=${b%p}
          if [ "$b" = '$' ] || [ $((b - a + 1)) -gt "$MAX_LINES" ]; then
            sedrange_big=1
          fi
          continue
        fi
        if printf '%s' "$st" | grep -qE '^[0-9]+p$'; then
          continue  # single-line print — allowed
        fi
        case "$st" in
          s/* | /*/p | *\{*\}*) continue ;;  # other sed scripts: not a range dump
          *) files+=("$tok") ;;
        esac
        ;;
    esac
  done
  pending_check || return 1
  return 0
}

run_self_test() {
  local SCRIPT
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  local TMP FAILED=0
  TMP=$(mktemp -d)
  # Expand now: TMP is function-local and out of scope when the EXIT trap fires.
  trap "rm -rf '$TMP'" EXIT
  mkdir -p "$TMP/logs"
  printf 'line\n' > "$TMP/logs/train.log"           # small but log-shaped
  printf 'notes\n' > "$TMP/notes.md"                # small, not log-shaped
  head -c 300000 /dev/zero | tr '\0' 'x' > "$TMP/big.txt"  # >256KB, not log-shaped
  cd "$TMP"

  run_case() {
    local desc="$1" expect="$2" cmdstr="$3" envflag="${4:-}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | EPM_ALLOW_LOG_DUMP=1 bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_LOG_DUMP bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  run_case "cat of log-shaped file blocks"            2 'cat logs/train.log'
  run_case "cat of *.log outside logs/ blocks"        2 'cat train.log'
  run_case "tail -50 of log allowed"                  0 'tail -50 logs/train.log'
  run_case "tail -n 5000 of log blocks"               2 'tail -n 5000 logs/train.log'
  run_case "tail -5000 of log blocks"                 2 'tail -5000 logs/train.log'
  run_case "head -n 100 of log allowed (<=200)"       0 'head -n 100 logs/train.log'
  run_case "tail -n 5000 of small non-log allowed"    0 'tail -n 5000 notes.md'
  run_case "grep over log allowed"                    0 "grep -iE 'error|traceback' logs/train.log"
  run_case "cat log piped into grep allowed"          0 'cat logs/train.log | grep error'
  run_case "cat log redirected to file allowed"       0 'cat logs/train.log > /tmp/x'
  run_case "cat small non-log file allowed"           0 'cat notes.md'
  run_case "cat of >256KB non-log file blocks"        2 'cat big.txt'
  run_case "sed -n wide range over log blocks"        2 "sed -n '1,5000p' logs/train.log"
  run_case "sed -n open-ended range over log blocks"  2 "sed -n '100,\$p' logs/train.log"
  run_case "sed -n single line over log allowed"      0 "sed -n '5p' logs/train.log"
  run_case "inline EPM_ALLOW_LOG_DUMP=1 allowed"      0 'EPM_ALLOW_LOG_DUMP=1 cat logs/train.log'
  run_case "env EPM_ALLOW_LOG_DUMP=1 allowed"         0 'cat logs/train.log' env
  run_case "chained cd then cat log blocks"           2 'cd logs && cat train.log'
  run_case "chain: cat log then echo blocks"          2 'cat logs/train.log ; echo done'
  run_case "log as LATER-command arg allowed (rm)"    0 'cat notes.md && rm logs/train.log'
  run_case "big-N tail then separate ls allowed"      0 'tail -n 5000 notes.md && ls logs/'
  run_case "heredoc mentioning a log path allowed"    0 'cat <<EOF
see logs/train.log for detail
EOF'
  run_case "non-dump command allowed"                 0 'git status'
  run_case "empty command allowed"                    0 ''

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
_allow=$(printf '%s' "${EPM_ALLOW_LOG_DUMP:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in
  1 | true | yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

if ! check_cmd "$cmd"; then
  echo "BLOCKED: '$BLOCK_VERB' would dump a large log-like file ($BLOCK_FILE) into context. CLAUDE.md § Context hygiene: never page whole logs into tool output — use \`grep -iE 'error|traceback|killed|OOM' $BLOCK_FILE\` or \`tail -50 $BLOCK_FILE\` instead. To deliberately override, prefix the command with EPM_ALLOW_LOG_DUMP=1." >&2
  exit 2
fi
exit 0
