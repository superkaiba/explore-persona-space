#!/usr/bin/env bash
# PreToolUse(Bash) guard: block `git push` / `git merge` / `gh pr merge|create`
# piped into an exit-code-masking consumer (task #1048).
#
# CLAUDE.md § Concurrent repo-root committers: "Never pipe a `git push` (or
# merge/PR command) through `tail`/`grep`/`head` — the pipe masks the non-zero
# exit code and the session proceeds believing the push landed (4 sessions hit
# exactly this on 2026-07-02); run it bare and check the exit code, or use
# `set -o pipefail` when a pipe is unavoidable." The prose rule failed open at
# least 5 times in 3 days (#957's Step 10d push was masked 2026-07-04); this
# hook mechanizes it for live sessions — the same dual-engine split as the
# pipe-python guard (#753): hook = ad-hoc inline Bash calls,
# `workflow_lint.py --check-piped-git-push` = committed scripts/*.sh recipes.
#
# WHY any pipe consumer (not just tail/grep/head): bash makes a pipeline's
# exit status the LAST stage's regardless of what that stage is, so the
# masking is consumer-independent; the legit escape is `pipefail` (honored
# below), and a named consumer list would invite trivial drift (`| cut`,
# `| sort`) with no false-positive protection gained.
#
# Deliberately conservative (false positives are worse than misses — when
# unsure, ALLOW):
#   - any `pipefail` substring in the command allows the whole command (the
#     rule's own sanctioned escape; substring-level on purpose — positional
#     `set -o pipefail` scope parsing is a shell parser, and the #796
#     quote-strip revert shows partial shell parsing in bash regex backfires);
#   - heredoc-bearing commands are blanket-allowed (guard_log_dump.sh
#     precedent; load-bearing here — the canonical
#     `git commit -m "$(cat <<'EOF'...)" && git push` recipe plus any commit
#     message DESCRIBING this very incident would otherwise false-block.
#     KNOWN MISS, accepted + test-pinned: `<<EOF ... && git push 2>&1 | tail`
#     inside ONE command slips through — the lint + prose rule remain defense
#     in depth; plan #1048 §4.1 step 4);
#   - `--dry-run` pushes may pipe (a dry run lands nothing, so masking its
#     exit code cannot cause the proceeded-on-a-rejected-push incident);
#   - producer-as-CONSUMER (`echo foo | git push` — final stage) is allowed:
#     the final stage's exit code IS the pipeline's, nothing is masked;
#   - pipes on a DIFFERENT `&&`/`||`/`;`/`&`/newline segment than the
#     producer are allowed (`git status | grep x && git push`);
#   - unparseable / ambiguous input is allowed (fail-soft, exit 0).
#
# <!-- known limitation -->
# Detectors scan the RAW command string without stripping quoted arguments
# (the guard_repo_root_branch.sh trade-off): a commit `-m` / marker `--note`
# string containing a literal `git push | tail` inside a NON-heredoc command
# false-blocks (test-pinned as the deliberate plan §7-row-1 trade-off).
# Remediation: `git commit -F <file>` / `task.py post-marker --file <path.md>`
# / the heredoc commit recipe (blanket-allowed above). Related residual: the
# whitespace-anchored comment-tail strip cuts at a quoted ` #` too (the
# guard_repo_root_branch.sh residual-gap-(viii) family), so a quoted `-m`
# text whose pattern sits AFTER a ` #` is truncated away — here that fails
# toward ALLOW (a miss on an already-false-positive shape), the safe
# direction for this guard.
#
# Escape hatch: EPM_ALLOW_PIPED_PUSH=1 — honored both as session env and as
# an inline prefix on the command itself (`EPM_ALLOW_PIPED_PUSH=1 git ...`).
#
# Contract: reads the PreToolUse JSON on stdin, exits 0 to allow, exits 2
# (blocking, stderr fed back to Claude) to refuse. Exit 2 is the documented
# PreToolUse blocking exit code; any OTHER non-zero is non-blocking.
#
# Self-test: bash .claude/hooks/guard_piped_git_push.sh --self-test
set -u

# Flag-tolerant producer anchor (the #897 detector shape from
# guard_repo_root_branch.sh, verb set swapped): `git [flags] push|merge` or
# `gh pr merge|create`. `([[:space:]]|$)` — NOT `\b` — after the verb, so
# `git merge-base --all main HEAD | head -1` (a canonical
# .claude/rules/diff-size-budget.md probe) never matches: `-` fails the
# terminator. Applied per NON-FINAL pipeline stage only.
PRODUCER_ERE='\bgit +(-[^ ]+( +[^ ]+)?( +|$))*(push|merge)([[:space:]]|$)|\bgh +pr +(merge|create)([[:space:]]|$)'

# Cheap pre-filter (this hook runs on EVERY Bash call): a producer token AND
# a `|` must co-occur before the unit parse runs. The verb terminator
# additionally admits `|` so the space-free `git push| tail` shape still
# reaches the full parse (a pre-filter miss would fail-open).
PREFILTER_ERE='\bgit\b[^|;&]*\b(push|merge)([[:space:]]|\||$)|\bgh[[:space:]]+pr[[:space:]]+(merge|create)\b'

# Classify one command string. Returns 0 = allow, 1 = block.
check_cmd() {
  local cmd="$1"

  # Inline escape hatch: EPM_ALLOW_PIPED_PUSH=1 anywhere in the command.
  case "$cmd" in *EPM_ALLOW_PIPED_PUSH=1*) return 0 ;; esac

  # Pipefail carve-out: any occurrence of the literal `pipefail` (covers
  # `set -o pipefail`, `set -euo pipefail`, `bash -o pipefail -c`, ...).
  # Substring-level on purpose: under the cooperative-agent threat model an
  # author who typed `pipefail` engaged the rule's own sanctioned escape;
  # this can only fail toward ALLOW (a miss, never a block).
  case "$cmd" in *pipefail*) return 0 ;; esac

  # Heredoc blanket-allow (see header — load-bearing for the canonical
  # heredoc commit recipe; the heredoc-compound residual is a documented
  # known miss).
  case "$cmd" in *'<<'*) return 0 ;; esac

  # Bash line-continuation normalization (`\<CR?><NL>` -> space), verbatim
  # the guard_repo_root_branch.sh pre-pass: bash strips these pre-execution,
  # joining the physical lines into one logical command.
  cmd=$(printf '%s' "$cmd" | sed -zE 's/\\\r?\n/ /g')

  # Fast pre-filter: no pipe at all, or no producer token -> allow.
  printf '%s' "$cmd" | grep -q '|' || return 0
  printf '%s' "$cmd" | grep -qE "$PREFILTER_ERE" || return 0

  # Unit split (redirection-aware): FIRST rewrite `|&` -> `|` (bash's `|&`
  # is `2>&1 |` shorthand); THEN strip redirection operators that contain
  # `&` (`2>&1`, `>&2`, `&>file`, `&>>file`) — redirections carry no command
  # semantics for producer detection, and without the strip the `&` split
  # below would sever `git push ... 2>&1 | tail` into `git push ... 2>` +
  # `1 | tail` and false-ALLOW (the plan-v1 defect the #1048 Phase 1.5
  # fact-check caught); THEN split into pipeline units on `&&`, `||`, `;`,
  # `&` (two-char separators substituted before single-char — the
  # split_and_label ordering trick); raw newlines are already unit
  # boundaries. A single `|` is NOT a unit separator — it stays inside its
  # unit for the per-stage scan below.
  local units
  units=$(printf '%s' "$cmd" \
    | sed -E 's/\|&/|/g' \
    | sed -E 's/[0-9]*>&[0-9]*//g; s/&>>?[^[:space:]]*//g' \
    | sed -E 's/&&/\n/g; s/\|\|/\n/g; s/;/\n/g; s/&/\n/g')

  local unit lead rest stage
  while IFS= read -r unit; do
    # Whitespace-anchored comment-tail strip (guard_repo_root_branch.sh
    # shape), then skip units that are pure comments.
    unit=$(printf '%s' "$unit" | sed -E 's/[[:space:]]#.*$//')
    lead=$(printf '%s' "$unit" | sed -E 's/^[[:space:]]+//')
    case "$lead" in \#*) continue ;; esac
    # No pipe inside this unit -> the producer (if any) is unpiped here.
    case "$unit" in *\|*) ;; *) continue ;; esac
    # Split the unit on `|` into stages; test the producer anchor on each
    # NON-FINAL stage (the final stage's exit code IS the pipeline's).
    rest=$unit
    while :; do
      case "$rest" in *\|*) ;; *) break ;; esac
      stage=${rest%%\|*}
      rest=${rest#*\|}
      if printf '%s\n' "$stage" | grep -qE "$PRODUCER_ERE"; then
        case "$stage" in
          *--dry-run*) ;; # dry-run carve-out: lands nothing, masking is harmless
          *) return 1 ;;
        esac
      fi
    done
  done <<< "$units"
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
        | EPM_ALLOW_PIPED_PUSH=1 bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_PIPED_PUSH bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- plan #1048 §6: must BLOCK (exit 2) ---
  run_case "B1 plain pipe" 2 'git push | tail -5'
  run_case "B2 2>&1 into grep" 2 'git push origin main 2>&1 | grep -v x'
  run_case "B3 gh pr merge piped" 2 'gh pr merge 123 --squash | head'
  run_case "B4 flag-tolerant git -C" 2 \
    'git -C .claude/worktrees/issue-1048 push origin issue-1048 2>&1 | tail -20'
  run_case "B5 pipe on the push's own && segment" 2 \
    'cd /tmp && git push origin main | grep -c rejected'
  run_case "B6 command substitution" 2 'out=$(git push 2>&1 | tail -1)'
  run_case "B7 git merge piped" 2 'git merge issue-x 2>&1 | tail -5'
  run_case "B8 tee without pipefail still masks" 2 'git push 2>&1 | tee push.log'
  run_case "B9 |& shorthand" 2 'git push |& tail -5'
  run_case "B10 raw-newline multi-line" 2 'echo pre
git push origin main | tail -5'
  run_case "B11 non-& redirection" 2 'git push 2>err.log | tail'
  run_case "S7r1 pinned FP: -m text mentions the pattern (no heredoc)" 2 \
    'git commit -m "never git push | tail in a recipe" && git push'

  # --- plan #1048 §6: must ALLOW (exit 0) ---
  run_case "A1 bare push" 0 'git push'
  run_case "A2 && chain, no pipe" 0 'git push origin main && echo ok'
  run_case "A3 pipefail carve-out" 0 'set -o pipefail; git push 2>&1 | tee log'
  run_case "A4 pipefail flag form" 0 "bash -o pipefail -c 'git push 2>&1 | tail -3'"
  run_case "A5 pipe on a DIFFERENT segment" 0 'git status | grep x && git push'
  run_case "A6 no producer" 0 'echo done | grep done'
  run_case "A7 || is not a pipe (issue931 shape)" 0 'git push origin main || echo "push failed"'
  run_case "A8 dry-run carve-out" 0 'git push --dry-run 2>&1 | head -5'
  run_case "A9 merge-base is not merge" 0 'git merge-base --all main HEAD | head -1'
  run_case "A10 pipe on a different ; segment" 0 'git log --oneline | head -5 ; git push origin main'
  run_case "A11 inline escape hatch" 0 'EPM_ALLOW_PIPED_PUSH=1 git push | tail -1'
  run_case "A12 env escape hatch" 0 'git push | tail -5' env
  run_case "A13 heredoc commit recipe" 0 'git commit -m "$(cat <<EOF
task #1048: guard piped push

never git push | tail
EOF
)" && git push origin main'
  run_case "A14 producer as consumer (final stage)" 0 'echo foo | git push'
  run_case "A15 empty command" 0 ''
  run_case "A16 raw-newline different units" 0 'git status | grep x
git push origin main'
  run_case "A17 braced historical Step-10d recovery shape, backslash-continued" 0 'git push origin main \
  || { git pull --rebase=merges --autostash && git push origin main; }'
  run_case "A18 heredoc-compound DOCUMENTED KNOWN MISS" 0 'git commit -m "$(cat <<EOF
msg
EOF
)" && git push 2>&1 | tail -3'

  # A15b: malformed stdin JSON -> fail-soft allow.
  local rc=0
  printf 'not-json' | env -u EPM_ALLOW_PIPED_PUSH bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A15b malformed stdin JSON"
  else
    echo "FAIL (got exit $rc, want 0): A15b malformed stdin JSON"
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
_allow=$(printf '%s' "${EPM_ALLOW_PIPED_PUSH:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in
  1 | true | yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

if ! check_cmd "$cmd"; then
  cat >&2 <<'BLOCK_MSG'
BLOCKED: piping `git push` / `git merge` / `gh pr merge|create` through a
filter masks the non-zero exit code — the session proceeds believing the
push/merge landed when it was rejected (4 sessions hit this 2026-07-02;
#957's Step 10d push was masked 2026-07-04). CLAUDE.md § Concurrent
repo-root committers: run it BARE and check the exit code, or use
`set -o pipefail` when a pipe is unavoidable (any `pipefail` in the command
is honored). `git push --dry-run` pipes are allowed. For marker-note /
commit-message text that merely MENTIONS the pattern, use
`task.py post-marker --file <path.md>` / `git commit -F <file>` (or a
heredoc). Deliberate override: prefix with EPM_ALLOW_PIPED_PUSH=1.
BLOCK_MSG
  exit 2
fi
exit 0
