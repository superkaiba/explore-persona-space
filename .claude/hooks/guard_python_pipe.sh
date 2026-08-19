#!/usr/bin/env bash
# PreToolUse(Bash) guard: block a bare `| python -c/-m` pipe CONSUMER in
# LOCAL argv (task #2009; extracted from the former inline command in the
# `.claude/settings.json` matcher-"Bash" hook group).
#
# This VM has no `python` on PATH — a bare `| python -c` / `| python -m`
# pipe dies at runtime with `python: command not found` (exit 127), and the
# project convention pipes into `uv run python` instead (CLAUDE.md § Task
# Workflow API; incident class #753). Dual-engine split: this hook covers
# ad-hoc inline Bash calls; `workflow_lint.py --check-pipe-python` covers
# committed `scripts/*.sh` (that lint walks `scripts/*.sh` only, so THIS
# file is out of its scan set).
#
# WHY the quoted-span strip (#2009 <- #1675): the former inline guard
# grepped the WHOLE Bash argv, so a pipe inside a quoted REMOTE command
# string (`ssh pod-X '... | python3 -c "..."'`,
# `gcloud compute ssh vm --command='... | python3 -c ...'`) false-blocked
# even though the no-python-on-PATH premise is local-only — python3 exists
# in the remote argv's environment. Quoted STRING ARGUMENTS are therefore
# stripped before matching, ported VERBATIM from guard_piped_git_push.sh
# (#1675): (1) a double-quoted span carrying a bare `$`/backtick is
# PRESERVED as one atomic token (a "$(...)" payload EXECUTES locally, so
# its interior stays scannable); (2) top-level `\<any>` escape pairs are
# consumed; (3) single-quoted spans strip; (4) substitution-free
# double-quoted spans strip. Bash line continuations (`\<CR?><NL>`) are
# joined FIRST, so a continuation-split local pipe is scanned as ONE
# logical command (previously a line-based grep miss; blocking it is
# fail-closed and correct — bash joins the lines before executing the
# pipe, which then fails locally).
#
# Heredoc note: NO heredoc carve-out is ported — heredoc bodies stay
# scanned post-strip (quoted spans inside heredoc text strip like any
# other span; a locally-piping heredoc PRODUCER stays blocked).
#
# Known residuals (deliberate; task #2009 plan §5, test-pinned in
# tests/test_guard_python_pipe.py):
#   FALSE-POSITIVE (fail-closed) shapes:
#   - a `$`-bearing double-quoted remote string
#     (`ssh host "... $V ... | python3 -c ..."`) is PRESERVED as live
#     substitution and still blocks — remediation: single-quote the remote
#     string (standard ssh hygiene; also what prevents unintended local
#     expansion);
#   - unbalanced quotes -> span unmatched -> text scanned raw (fails
#     toward block; never a new false negative).
#   FAIL-OPEN classes:
#   - interpreter/remote-executor payloads WITHOUT live substitution: a
#     quoted payload handed to a LOCAL wrapper (`bash -c '...'`,
#     substitution-free `sh -c "..."`, `eval '...'`, `xargs ... sh -c`)
#     executes downstream but reads as string data post-strip, so a
#     wrapper-quoted local python pipe that formerly BLOCKED is now
#     ALLOWED. DELIBERATE, sibling-precedent fail-open class
#     (guard_piped_git_push.sh header, fail-open class (1);
#     cooperative-agent threat model; harm bound = one exit-127 turn — the
#     ergonomic cost this guard reduces, not a safety boundary; the
#     `check_pipe_python` lint + the CLAUDE.md prose rule remain defense
#     in depth). A wrapper-preserve ERE branch was considered and
#     REJECTED: partial shell parsing in ERE backfires (#796) and it would
#     re-introduce false positives for `ssh host bash -c '...'` remote
#     shapes.
#   - ANSI-C `$'...'` with an interior `\'` is unmodeled (sibling parity;
#     rare) — a mis-paired span can fail open OR closed.
#
# Escape hatch: EPM_ALLOW_PYTHON_PIPE=1 — honored both as session env and
# as an inline prefix substring on the command itself
# (`EPM_ALLOW_PYTHON_PIPE=1 ... | python3 -c ...`); sibling parity with
# EPM_ALLOW_PIPED_PUSH=1.
#
# Contract: reads the PreToolUse JSON on stdin, exits 0 to allow, exits 2
# (blocking, stderr fed back to Claude) to refuse. Exit 2 is the documented
# PreToolUse blocking exit code; any OTHER non-zero is non-blocking.
#
# Self-test: bash .claude/hooks/guard_python_pipe.sh --self-test
set -u

# The pipe-into-bare-python ERE, byte-unchanged from the former inline
# settings.json command. Its behavior on UNQUOTED text is the parity
# contract; tests/test_workflow_lint.py sources this assignment for the
# dual-engine agreement pin against workflow_lint.PIPE_PYTHON_RE.
PIPE_PYTHON_ERE='\|[[:space:]]*python3?(\.[0-9]+)?[[:space:]]+(-[^[:space:]]+[[:space:]]+)*-[cm]([^A-Za-z0-9_]|$)'

# Quoted-span strip, ported VERBATIM from guard_piped_git_push.sh (#1675).
# Branches, in order: (1) PRESERVE a double-quoted span with a bare $ or
# backtick (live substitution executes locally) — matched, captured as
# group 1, replaced by ITSELF + a space — consumed atomically, so its
# interior (apostrophes included) can never seed a later span match;
# (2) top-level escape pairs \<any> are consumed (bash treats \' \" \| as
# literal chars, never quote/pipe operators); (3) single-quoted spans
# always strip (bash expands nothing inside, no escapes exist);
# (4) substitution-free double-quoted spans strip, consuming \-escape
# pairs. Branches are start-disjoint (no leftmost-longest ambiguity);
# replacement is "\1 " (group 1 unset on strip branches -> a single
# space; GNU sed substitutes empty).
STRIP_QUOTED_SPANS_ERE="(\"(\\\\.|[^\"\\\\\$\`])*[\$\`](\\\\.|[^\"\\\\])*\")|\\\\.|'[^']*'|\"(\\\\.|[^\"\\\\\$\`])*\""

# Classify one command string. Returns 0 = allow, 1 = block.
check_cmd() {
  local cmd="$1"

  # Inline escape hatch: EPM_ALLOW_PYTHON_PIPE=1 anywhere in the command.
  case "$cmd" in *EPM_ALLOW_PYTHON_PIPE=1*) return 0 ;; esac

  # Cheap pre-filter (this hook runs on EVERY Bash call): the ERE requires
  # a literal `|`, and the strip never introduces one — no pipe, no match.
  case "$cmd" in *\|*) ;; *) return 0 ;; esac

  # Bash line-continuation normalization (`\<CR?><NL>` -> space), THEN the
  # quoted-span strip (#1675) — continuation first so a span broken across
  # a backslash-continuation is rejoined before the strip (and branch 2
  # never sees `\<newline>`); `-z` lets spans match across raw newlines.
  cmd=$(printf '%s' "$cmd" | sed -zE -e 's/\\\r?\n/ /g' -e "s/${STRIP_QUOTED_SPANS_ERE}/\\1 /g")

  if printf '%s' "$cmd" | grep -qE "$PIPE_PYTHON_ERE"; then
    return 1
  fi
  return 0
}

run_self_test() {
  local SCRIPT FAILED=0
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

  run_case() {
    local desc="$1" expect="$2" cmdstr="$3" envflag="${4:-}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | EPM_ALLOW_PYTHON_PIPE=1 bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_PYTHON_PIPE bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- must BLOCK (exit 2): genuinely LOCAL pipes ---
  run_case "B1 plain local pipe" 2 'cat x.json | python3 -c "import sys"'
  run_case "B2 bare python -m" 2 'foo | python -m json.tool'
  run_case "B3 live substitution stays scannable" 2 'echo "$(cat x | python3 -c 1)"'
  run_case "B4 local consumer after an ssh stage" 2 "ssh host 'cat log' | python3 -c 'x'"

  # --- must ALLOW (exit 0) ---
  run_case "A1 incident shape: single-quoted ssh remote string" 0 \
    "ssh pod-x 'ps aux | python3 -c \"print(1)\"'"
  run_case "A2 gcloud --command= remote string" 0 \
    "gcloud compute ssh vm --command='df -h | python3 -c \"1\"'"
  run_case "A3 uv run python consumer" 0 'cat x | uv run python -c "print(1)"'
  run_case "A4 inline escape hatch" 0 'EPM_ALLOW_PYTHON_PIPE=1 cat x | python3 -c "y"'
  run_case "A5 env escape hatch" 0 'cat x | python3 -c "y"' env

  # A6: malformed stdin JSON -> fail-soft allow.
  local rc=0
  printf 'not-json' | env -u EPM_ALLOW_PYTHON_PIPE bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A6 malformed stdin JSON"
  else
    echo "FAIL (got exit $rc, want 0): A6 malformed stdin JSON"
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
_allow=$(printf '%s' "${EPM_ALLOW_PYTHON_PIPE:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in
  1 | true | yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

if ! check_cmd "$cmd"; then
  cat >&2 <<'BLOCK_MSG'
BLOCKED: bare `| python -c/-m` pipe. This VM has no `python` on PATH — `python: command not found` (exit 127). Pipe into `uv run python` instead: `... | uv run python -c "..."`. CLAUDE.md § Task Workflow API.
A pipe inside a QUOTED remote-command string (`ssh host '... | python3 -c ...'`, `gcloud compute ssh --command='...'`) is exempt — single-quote the remote string if this fired on a remote pipe (a `$`-bearing double-quoted remote string is treated as live LOCAL substitution and still blocks); deliberate override: prefix with EPM_ALLOW_PYTHON_PIPE=1.
BLOCK_MSG
  exit 2
fi
exit 0
