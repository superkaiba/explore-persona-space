#!/usr/bin/env bash
# PreToolUse(Edit|Write) guard: block edits that would regress the #1269
# LESSONS.md byte budgets / index parity (task #1279).
#
# `.claude/rules/LESSONS.md` is the always-on index every session loads; #1269
# added blocking byte-budget/ratchet/per-row/parity gates in
# `scripts/workflow_lint.py::check_lessons_index`, but those fire only where
# the lint runs (pre-commit in worktrees / PR path). A direct repo-root Edit +
# explicit-path commit never runs the lint, so the budget could be silently
# blown. This hook closes that bypass at edit time: any Edit/Write whose
# file_path targets a `.claude/rules/LESSONS.md` (repo root, worktree, or any
# clone) is blocked iff the PROSPECTIVE post-edit content would FAIL
# check_lessons_index — with the lint's own error text surfaced — and allowed
# otherwise. The check itself runs in `guard_lessons_edit_check.py`, which
# imports the real check_lessons_index at runtime (edited tree's copy first,
# so a same-diff constant bump is honored; zero re-implementation).
#
# Contract: reads the PreToolUse JSON on stdin; exit 0 = allow; exit 2 +
# stderr = block (stderr fed back to Claude). FAIL-OPEN on every hook-internal
# error (bad JSON, missing jq/python, unimportable lint, timeout): a guard
# failure must never wedge editing — the commit-time lint gates remain the
# backstop. Blocks ONLY on helper exit 2.
#
# Escape hatches (sanctioned maintenance):
#   - EPM_ALLOW_LESSONS_EDIT=1|true|yes (session env, case-insensitive), or
#   - touch <repo>/.claude/cache/allow-lessons-edit   (honored for 900 s).
#   The sentinel path is overridable via EPM_LESSONS_EDIT_SENTINEL (test
#   support — keeps --self-test/pytest hermetic, never touching the live
#   fleet-global sentinel).
#
# NAMED RESIDUALS (decided omissions, not oversights — see plan #1279 §4a-3):
#   - Bash-tool writes (sed -i / tee / >> / inline python) are NOT covered;
#     for the repo-root Bash write + explicit-path commit there is NO
#     commit-time backstop either. Reliable write-shaped Bash parsing is
#     high-false-positive on a BLOCKING hook (guard_log_dump.sh needed ~440
#     lines + 3 hardening rounds for READ verbs alone); worktree/PR paths
#     keep the commit-time lint.
#   - The sentinel escape hatch is fleet-global for its 900 s window: a
#     concurrent session's unrelated LESSONS edit inside that window rides
#     the same sentinel (accepted race; bounded by the TTL).
#   - MultiEdit is not a live tool in this harness; if a future harness
#     reintroduces it, the Edit|Write matcher misses it.
#   - The cheap path filter is a suffix match; the helper re-verifies with
#     os.path.normpath, so only non-suffix `..` spellings skip the guard.
#
# Self-test: bash .claude/hooks/guard_lessons_edit.sh --self-test
set -u

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"          # <repo>/.claude/hooks
REPO_ROOT="${SELF_DIR%/.claude/hooks}"
HELPER="$SELF_DIR/guard_lessons_edit_check.py"
SENTINEL="${EPM_LESSONS_EDIT_SENTINEL:-$REPO_ROOT/.claude/cache/allow-lessons-edit}"
SENTINEL_TTL=900

run_self_test() {
  local SCRIPT
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  local TMP FAILED=0
  TMP=$(mktemp -d)
  # Expand now: TMP is function-local and out of scope when the EXIT trap fires.
  trap "rm -rf '$TMP'" EXIT

  # Fixture sizes come from the RESOLVED lint constants at runtime (never
  # hardcoded) so a future constant retune cannot silently break this suite.
  # This first `uv run` also warms the project env before any timed dispatch.
  local consts ratchet headroom rowcap maxbytes
  consts=$(cd "$REPO_ROOT" && uv run python "$HELPER" --print-constants 2>/dev/null)
  ratchet=$(printf '%s' "$consts" | jq -r '._LESSONS_RATCHET_BYTES // empty' 2>/dev/null)
  headroom=$(printf '%s' "$consts" | jq -r '._LESSONS_RATCHET_MAX_HEADROOM_BYTES // empty' 2>/dev/null)
  rowcap=$(printf '%s' "$consts" | jq -r '._LESSONS_ROW_MAX_BYTES // empty' 2>/dev/null)
  maxbytes=$(printf '%s' "$consts" | jq -r '._LESSONS_MAX_BYTES // empty' 2>/dev/null)
  if [ -z "$ratchet" ] || [ -z "$headroom" ] || [ -z "$rowcap" ] || [ -z "$maxbytes" ]; then
    echo "self-test: FAIL (could not resolve lint constants via --print-constants)" >&2
    return 1
  fi

  # Synthetic tree: a rules dir with two stub rules — NEVER named 'gotchas'
  # (the grandfather-table hygiene would FAIL a short gotchas row) — and NO
  # scripts/workflow_lint.py, so the helper falls back to THIS repo's lint
  # and every case runs against the LIVE constants. A worktree-shaped subtree
  # pins that the suffix match covers worktrees.
  mkdir -p "$TMP/.claude/rules" "$TMP/scripts" "$TMP/.claude/worktrees/issue-9/.claude/rules"
  : > "$TMP/.claude/rules/alpha.md"
  : > "$TMP/.claude/rules/beta.md"
  : > "$TMP/.claude/worktrees/issue-9/.claude/rules/alpha.md"
  local LESSONS="$TMP/.claude/rules/LESSONS.md"
  local WT_LESSONS="$TMP/.claude/worktrees/issue-9/.claude/rules/LESSONS.md"

  local valid_total=$(( ratchet - headroom / 2 ))
  local longtrig
  longtrig=$(head -c "$rowcap" /dev/zero | tr '\0' 'y')   # > rowcap once row prefix is added

  # mk_content <target_total_bytes> <rows...> -> stdout. Pads with a non-row
  # 'x' line to the exact BYTE total (wc -c, not ${#}: the em-dash is 3 bytes).
  mk_content() {
    local target=$1; shift
    local body="# Lessons index (synthetic self-test fixture)
"
    local row
    for row in "$@"; do
      body="${body}${row}
"
    done
    local base_bytes pad
    base_bytes=$(printf '%s' "$body" | wc -c)
    pad=$(( target - base_bytes - 1 ))
    [ "$pad" -lt 0 ] && pad=0
    printf '%s' "$body"
    head -c "$pad" /dev/zero | tr '\0' 'x'
    printf '\n'
  }

  local ROW_A='- alpha.md — trigger a'
  local ROW_B='- beta.md — trigger b'

  mk_content "$valid_total" "$ROW_A" "$ROW_B"                          > "$TMP/valid.md"
  mk_content "$valid_total" "- alpha.md — $longtrig" "$ROW_B"          > "$TMP/longrow.md"
  mk_content "$valid_total" "$ROW_A" "$ROW_B" '- ghost.md — trigger g' > "$TMP/stalerow.md"
  mk_content "$valid_total" "$ROW_A"                                   > "$TMP/missingrow.md"
  mk_content $(( maxbytes + 200 )) "$ROW_A" "$ROW_B"                   > "$TMP/overcap.md"
  mk_content $(( ratchet + 100 )) "$ROW_A" "$ROW_B"                    > "$TMP/overratchet.md"
  mk_content $(( maxbytes + 200 )) '- alpha.md — trigger a'            > "$TMP/wt_overcap.md"
  cp "$TMP/valid.md" "$LESSONS"   # on-disk current content for the Edit cases

  wp() {  # wp <file_path> <content_file> -> Write payload JSON
    jq -n --arg fp "$1" --rawfile c "$2" \
      '{tool_name: "Write", tool_input: {file_path: $fp, content: $c}}'
  }
  ep() {  # ep <file_path> <old> <new> [replace_all] -> Edit payload JSON
    if [ "${4:-}" = "replace_all" ]; then
      jq -n --arg fp "$1" --arg o "$2" --arg n "$3" \
        '{tool_name: "Edit", tool_input: {file_path: $fp, old_string: $o, new_string: $n, replace_all: true}}'
    else
      jq -n --arg fp "$1" --arg o "$2" --arg n "$3" \
        '{tool_name: "Edit", tool_input: {file_path: $fp, old_string: $o, new_string: $n}}'
    fi
  }

  # run_case <desc> <expected_rc> <payload> [mode] [required_stderr_substring]
  # mode: '' | env (EPM_ALLOW_LESSONS_EDIT=1) | sentinel-fresh | sentinel-stale
  run_case() {
    local desc=$1 expect=$2 payload=$3 mode=${4:-} want=${5:-}
    local rc=0 err
    local sentinel="$TMP/no-such-sentinel"      # default: hatch inert, hermetic
    case "$mode" in
      sentinel-fresh) sentinel="$TMP/sentinel-live"; touch "$sentinel" ;;
      sentinel-stale) sentinel="$TMP/sentinel-old"; touch -d '-3600 seconds' "$sentinel" ;;
    esac
    if [ "$mode" = "env" ]; then
      err=$(printf '%s' "$payload" | EPM_ALLOW_LESSONS_EDIT=1 \
        EPM_LESSONS_EDIT_SENTINEL="$sentinel" bash "$SCRIPT" 2>&1 >/dev/null) || rc=$?
    else
      err=$(printf '%s' "$payload" | env -u EPM_ALLOW_LESSONS_EDIT \
        EPM_LESSONS_EDIT_SENTINEL="$sentinel" bash "$SCRIPT" 2>&1 >/dev/null) || rc=$?
    fi
    if [ "$rc" -ne "$expect" ]; then
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
      return
    fi
    if [ -n "$want" ] && ! printf '%s' "$err" | grep -qF -- "$want"; then
      echo "FAIL (stderr missing '$want'): $desc"
      FAILED=1
      return
    fi
    echo "PASS (exit $rc): $desc"
  }

  # §4e case table (the behavior-matrix source of truth)
  run_case "1: Write to unrelated path allowed"                0 "$(wp "$TMP/notes.md" "$TMP/valid.md")"
  run_case "2: valid Write within ratchet band allowed"        0 "$(wp "$LESSONS" "$TMP/valid.md")"
  run_case "3: Write with row over per-row cap blocks"         2 "$(wp "$LESSONS" "$TMP/longrow.md")" '' 'per-row cap'
  run_case "4: Write with row naming nonexistent rule blocks"  2 "$(wp "$LESSONS" "$TMP/stalerow.md")" '' 'no matching'
  run_case "5: Write missing a row for a stub rule blocks"     2 "$(wp "$LESSONS" "$TMP/missingrow.md")" '' 'no index row'
  run_case "6: Write over the leanness cap blocks"             2 "$(wp "$LESSONS" "$TMP/overcap.md")" '' 'leanness cap'
  run_case "7: Write past the growth ratchet blocks"           2 "$(wp "$LESSONS" "$TMP/overratchet.md")" '' 'grew past'
  run_case "8: Edit growing a row over the row cap blocks"     2 "$(ep "$LESSONS" 'trigger a' "$longtrig")" '' 'per-row cap'
  run_case "9: Edit with absent old_string allowed"            0 "$(ep "$LESSONS" 'ZZZ_NOT_PRESENT' 'zzz')"
  run_case "10: Edit with ambiguous old_string allowed"        0 "$(ep "$LESSONS" 'trigger' "$longtrig")"
  run_case "11: Edit replace_all with valid result allowed"    0 "$(ep "$LESSONS" 'trigger' 'trig' replace_all)"
  run_case "12: env EPM_ALLOW_LESSONS_EDIT=1 allows blocker"   0 "$(wp "$LESSONS" "$TMP/overcap.md")" env
  run_case "13a: fresh sentinel allows blocker"                0 "$(wp "$LESSONS" "$TMP/overcap.md")" sentinel-fresh
  run_case "13b: stale sentinel still blocks"                  2 "$(wp "$LESSONS" "$TMP/overcap.md")" sentinel-stale
  run_case "14: malformed JSON stdin allowed (fail-open)"      0 'not json {'
  run_case "15: worktree-shaped path over cap blocks"          2 "$(wp "$WT_LESSONS" "$TMP/wt_overcap.md")"

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

# Escape hatch 1: session env (case-insensitive, guard_log_dump convention).
case "$(printf '%s' "${EPM_ALLOW_LESSONS_EDIT:-}" | tr '[:upper:]' '[:lower:]')" in
  1 | true | yes) exit 0 ;;
esac

payload=$(cat 2>/dev/null) || exit 0                                    # fail-open
fp=$(printf '%s' "$payload" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || exit 0
[ -n "$fp" ] || exit 0
case "$fp" in                                # cheap filter: only LESSONS.md pays python
  */.claude/rules/LESSONS.md | .claude/rules/LESSONS.md) : ;;
  *) exit 0 ;;
esac

# Escape hatch 2: fresh sentinel file (<= SENTINEL_TTL seconds old).
if [ -f "$SENTINEL" ]; then
  now=$(date +%s); mt=$(stat -c %Y "$SENTINEL" 2>/dev/null || echo 0)
  [ $(( now - mt )) -le "$SENTINEL_TTL" ] && exit 0
fi

out=$(printf '%s' "$payload" \
  | timeout 20 bash -c "cd '$REPO_ROOT' && uv run python '$HELPER'" 2>&1)
rc=$?
if [ "$rc" -eq 2 ]; then
  printf '%s\n' "$out" >&2
  exit 2
fi
exit 0   # rc 0 = pass; rc 1/124/127/... = helper failure -> FAIL-OPEN
