#!/usr/bin/env bash
# Feature-isolation matrix for the #2046 scoping escape.
set -u
SCRIPT=/home/thomasjiralerspong/explore-persona-space/.claude/hooks/guard_root_code_commit.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export EPM_CERT_REHASH_DELAY_S=0

R="$TMP/foreign" && git init -q "$R"
mkdir -p "$R/scripts" "$R/docs"
printf 'print(0)\n' > "$R/scripts/foreign.py"
echo note > "$R/docs/d.md"
git -C "$R" add scripts/foreign.py docs/d.md
MSGF="$TMP/msg.txt"; printf 'msg\n' > "$MSGF"
CERTF="$TMP/cert.txt"

run_case() {
  local desc="$1" expect="$2" cmdstr="$3" case_cwd="${4:-$R}"
  local rc=0
  jq -n --arg c "$cmdstr" --arg d "$case_cwd" '{tool_input: {command: $c}, cwd: $d}' \
    | env -u EPM_ALLOW_ROOT_CODE_COMMIT EPM_ROOT_CODE_COMMIT_REPO="$R" \
      EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq "$expect" ]; then
    echo "PASS (exit $rc, want $expect): $desc"
  else
    echo "FAIL (got exit $rc, want $expect): $desc"
  fi
}

# baseline sanity (mirrors self-test B18/A23 in THIS repo)
run_case "s1 -m + pathspec (B18 shape)" 0 'git commit -m x -- docs/d.md'
run_case "s2 -F realfile + pathspec (A23 shape)" 0 "git commit -F $MSGF -- docs/d.md"

# single-feature additions
run_case "f1 -F /dev/stdin + pathspec (no heredoc)" 0 'git commit -F /dev/stdin -- docs/d.md'
run_case "f2 -F realfile + pathspec + simple heredoc" 0 "git commit -F /dev/stdin -- docs/d.md <<'MSG'
msg
MSG"
run_case "f3 cd-to-root prefix + -m + pathspec" 0 "cd $R
git commit -m x -- docs/d.md"
run_case "f4 -m + pathspec + trailing echo" 0 'git commit -m x -- docs/d.md
echo "commit rc=$?"'
run_case "f5 -m + pathspec + trailing git log" 0 'git commit -m x -- docs/d.md
git log -1 --oneline -- docs/d.md'
run_case "f6 -m + pathspec + redirect (A21 shape)" 0 'git commit -m x -- docs/d.md > /tmp/i2046.log 2>&1'
run_case "f7 -m + pathspec + semicolon tail" 0 'git commit -m x -- docs/d.md; echo done'
