#!/usr/bin/env bash
# Replay the #2046 incident command shapes against the CURRENT
# guard_root_code_commit.sh in an isolated temp repo (EPM_ROOT_CODE_COMMIT_REPO).
set -u
SCRIPT=/home/thomasjiralerspong/explore-persona-space/.claude/hooks/guard_root_code_commit.sh
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export EPM_CERT_REHASH_DELAY_S=0

R="$TMP/foreign" && git init -q "$R"
mkdir -p "$R/scripts" "$R/docs" "$R/figures/issue_1739/ladder" "$R/eval_results/issue_1739/ladder"
printf 'print(0)\n' > "$R/scripts/issue1739_r275_fold.py"   # the FOREIGN uncertified gated file
echo note > "$R/docs/map_behavior_prediction_interim_results.md"
echo png  > "$R/figures/issue_1739/ladder/f.png"
echo '{}' > "$R/eval_results/issue_1739/ladder/e.json"
git -C "$R" add scripts/issue1739_r275_fold.py docs/map_behavior_prediction_interim_results.md \
  figures/issue_1739/ladder eval_results/issue_1739/ladder
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

MSGBODY='docs(#1739): fold the query-map L-ladder (R2.75) — one real label-scaling curve

The ladder ran and is complete (3 behaviors x 3 budgets). Framing is
deliberately narrow: only the label-consuming direct ridge yields a genuine
L-curve (non-monotonic, dipping at L=2500 in all three behaviors, matching
the already-documented direct-ridge dip); the label-free projections are
invariant in L by construction and are plotted as reference lines, not
scaling results.

Co-Authored-By: Claude <noreply@anthropic.com>'

# 1. EXACT incident retry shape: cd-prefix + -F /dev/stdin + pathspec + redirect + heredoc + trailing cmds
C1="cd $R
git commit -F /dev/stdin -- docs/map_behavior_prediction_interim_results.md figures/issue_1739/ladder eval_results/issue_1739/ladder > /tmp/i2046_replay.out 2>&1 <<'MSG'
$MSGBODY
MSG
echo \"commit rc=\$?\"; git log -1 --oneline -- docs/map_behavior_prediction_interim_results.md"
run_case "1 EXACT incident retry (cd + -F /dev/stdin + pathspec + redirect + heredoc + tail)" 0 "$C1"

# 2. Same minus the cd prefix
C2="git commit -F /dev/stdin -- docs/map_behavior_prediction_interim_results.md figures/issue_1739/ladder eval_results/issue_1739/ladder > /tmp/i2046_replay.out 2>&1 <<'MSG'
$MSGBODY
MSG
echo \"commit rc=\$?\""
run_case "2 minus cd prefix" 0 "$C2"

# 3. Same minus heredoc/-F (plain -m one-line msg) but WITH cd prefix + redirect + tail
C3="cd $R
git commit -m \"docs: fold\" -- docs/map_behavior_prediction_interim_results.md > /tmp/i2046_replay.out 2>&1
echo \"commit rc=\$?\""
run_case "3 cd prefix + -m + pathspec + redirect + tail" 0 "$C3"

# 4. Minimal: -F /dev/stdin + heredoc + pathspec, nothing else
C4="git commit -F /dev/stdin -- docs/map_behavior_prediction_interim_results.md <<'MSG'
$MSGBODY
MSG"
run_case "4 -F /dev/stdin + heredoc + pathspec only" 0 "$C4"

# 5. Control: bare commit (no pathspec) must still BLOCK
run_case "5 control: bare commit still blocks" 2 'git commit -m x'

# 6. Control: pathspec naming the gated file must still BLOCK
run_case "6 control: pathspec covering gated file blocks" 2 \
  'git commit -m x -- scripts/issue1739_r275_fold.py'
