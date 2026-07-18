#!/usr/bin/env bash
# PreToolUse(Bash) guard (#1500): refuse a repo-root `git commit` whose pending
# payload includes UNCERTIFIED code paths (scripts/**.py, src/**, tests/**.py).
# Certification = a fresh content-hash-bound line written by
# scripts/inline_lint_gate.py on a PASSING Step 9a-ter inline payload lint gate
# run (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460).
#
# WHY commit-time, not push-time: scripts/auto_push_main.sh (cron */2 + Stop
# hook) pushes local main whenever ahead, so a Bash-tool push gate is bypassed
# within ~2 min of the commit landing; and push-time payload attribution on the
# shared root batches other sessions' commits. The commit is the only
# interceptable choke point for Bash-tool-driven inline rounds (plan #1500 §11).
#
# TWO-LAYER PREDICATE (plan #1500 §4.2):
#   Layer 1 (command text, cheap): detect a plausible repo-root `git commit`
#     clause — clause machinery copied from scripts/guard_repo_root_pull.sh
#     (#1201/#1250: line-continuation normalize, sentinel clause split, #804
#     cd-latch across `&&` only, command-position verb anchor). Also collects
#     commit-clause pathspecs AND `git add`-clause paths chained in the same
#     command (the compound `git add X && git commit -m ...` idiom stages
#     NOTHING at PreToolUse time — the hook fires BEFORE execution).
#   Layer 2 (repo state, authoritative): classify the pending payload from
#     `git diff --cached` (staged) ∪ `-a` worktree-modified ∪ the Layer-1 text
#     paths, filtered to the gated glob; block iff any gated path lacks a
#     fresh cert line bound to the LANDING content (worktree hash for
#     `-a`/pathspec/add-clause shapes — those commit worktree content; staged
#     blob sha only for plain staged commits). A Layer-1 match with no gated
#     pending payload ALLOWS, so text-level false matches (e.g. a
#     commit-message line beginning `git commit`) cannot block on their own.
#
# NO heredoc blanket-allow (deliberate deviation from the guard siblings): the
# canonical commit-message shape `git commit -m "$(cat <<'EOF'...)"` embeds a
# heredoc, so a blanket allow would exempt most real commits; Layer 2
# neutralizes the mention-FP risk instead.
#
# FAIL MODES: stdin-parse failure -> OPEN (A16 sibling parity — a guard bug
# must never wedge the fleet); payload-classification git failure on a
# CONFIRMED root-commit clause -> CLOSED (#458/#1147 class); blanket
# `git add -A|.|--all` chained to a root commit -> CLOSED (landing set
# unknowable at hook time; blanket root staging is independently banned by
# CLAUDE.md § Concurrent repo-root committers); missing/stale/mismatched
# cert -> CLOSED (that IS the block).
#
# <!-- known limitations -->
# - CWD-BLIND (pull-guard parity): a bare `git commit` issued while the Bash
#   shell's inherited cwd is a worktree matches Layer 1, but Layer 2 reads the
#   ROOT's index — it allows unless the root simultaneously has gated files
#   staged. Remediation: `git -C "$WT" commit`.
# - Shared-index race: another session staging gated files concurrently can
#   false-block an innocent commit (rare; block direction is safe).
# - `git -C .` from the root stays waived (path-blind sibling-parity residual);
#   the literal root SPELLINGS ($REPO absolute, ~/explore-persona-space,
#   $HOME/explore-persona-space) are NOT waived — one notch stronger than the
#   siblings' fully path-blind -C waiver.
# - `git revert` / `git cherry-pick` are not matched (verb anchor is `commit`)
#   — a deliberate residual convenient for emergency repair of a landed
#   offender; accidents-not-adversaries disposition.
# - Deletions of gated paths are exempt (no content to certify); the worktree
#   pipeline remains the sanctioned channel for deletions.
# - A byte-perfect forged cert line is indistinguishable by construction —
#   same agent-trust boundary as Step 10d's #1082 procedural ban: guards
#   target forgetting, not adversaries. NEVER hand-write the cert file.
# - A commit MESSAGE token spelling a bare gated path (e.g.
#   `-m "fix scripts/foo.py now"`) is collected as a text pathspec and can
#   false-block toward the override — over-collection is deliberate (it can
#   never silently allow); quote-adjacent tokens are excluded.
# - A genuine trailing shell comment inside a commit/add clause (e.g.
#   `git commit -m x tasks/t.md # split scripts/foo.py later`) is
#   token-scanned too: the comment-tail strip is decision-only (lead/verb/
#   latch/waiver), while the token/flag scan reads the RAW clause so a `#`
#   inside a quoted message cannot discard the pathspecs/flags after it
#   (round-2 Major). A gated path or `-a`-shaped token in a real comment
#   over-collects toward BLOCK — never a silent allow.
#
# Escape hatch: EPM_ALLOW_ROOT_CODE_COMMIT=1 — session env or inline prefix.
# Legitimate uses (record the reason in an epm:progress note): a MODIFIED
# payload file whose red is genuinely pre-existing but the helper refused
# conservatively; emergency fleet repair.
#
# Contract: reads the PreToolUse JSON on stdin, exit 0 allow, exit 2 (blocking,
# stderr fed back to Claude) refuse; any OTHER non-zero is non-blocking.
# Test overrides (hermetic tmp repos): EPM_ROOT_CODE_COMMIT_REPO,
# EPM_INLINE_CERT_PATH, EPM_INLINE_CERT_MAX_AGE_S.
#
# Self-test: bash .claude/hooks/guard_root_code_commit.sh --self-test
set -u

REPO=/home/thomasjiralerspong/explore-persona-space
GUARD_REPO="${EPM_ROOT_CODE_COMMIT_REPO:-$REPO}"
CERT="${EPM_INLINE_CERT_PATH:-/tmp/eps-inline-lint-cert-v1.txt}"
MAX_AGE="${EPM_INLINE_CERT_MAX_AGE_S:-21600}" # 6 h

# Command-position anchors (#1250 parity): the clause LEAD must BE a git
# invocation whose verb is commit/add — optionally behind env assignments, the
# CLOSED wrapper set, shell compound keywords, bare -flag tokens, and a
# timeout-style duration token. An unlisted lead word (uv, bash, echo, ...)
# makes the predicate unmatchable.
COMMIT_CMD_ERE='^([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|(nohup|setsid|sudo|env|time|timeout|command|exec|eval|if|elif|then|else|while|until|do|!)[[:space:]]+|-[^[:space:]]+[[:space:]]+|[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)*git[[:space:]]+(-[^[:space:]]+([[:space:]]+[^[:space:]]+)?([[:space:]]+|$))*commit([[:space:]]|$)'
ADD_CMD_ERE='^([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|(nohup|setsid|sudo|env|time|timeout|command|exec|eval|if|elif|then|else|while|until|do|!)[[:space:]]+|-[^[:space:]]+[[:space:]]+|[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)*git[[:space:]]+(-[^[:space:]]+([[:space:]]+[^[:space:]]+)?([[:space:]]+|$))*add([[:space:]]|$)'
GATED_PATH_ERE='^(scripts/.*\.py|src/.+|tests/.*\.py)$'

# classify_cmd <command>: Layer 1. Sets globals root_commit / has_dash_a /
# add_all_chained / text_paths (newline-separated gated-prefix tokens).
classify_cmd() {
  local cmd="$1"
  root_commit=0 has_dash_a=0 add_all_chained=0 text_paths=""

  # Bash line-continuation normalization, |& rewrite, fd-dup strip, clause
  # split with separator sentinels (verbatim pull-guard pre-pass).
  local records
  records=$(printf '%s\n' "$cmd" \
    | sed -zE 's/\\\r?\n/ /g' \
    | sed -E 's/\|&/|/g' \
    | sed -E 's/[0-9]*>&[0-9]*//g; s/&>>?[^[:space:]]*//g' \
    | sed -E 's/&&/\n@AND@/g; s/\|\|/\n@OR@/g; s/;/\n@SEQ@/g; s/\|/\n@PIPE@/g; s/&/\n@BG@/g')

  local -a recs
  mapfile -t recs <<< "$records"

  local n=${#recs[@]} i rec sep clause raw_clause lead tgt ctgt latched=0 verb
  for ((i = 0; i < n; i++)); do
    rec=${recs[i]}
    case "$rec" in
      @AND@*) sep=AND clause=${rec#@AND@} ;;
      @OR@* | @SEQ@* | @PIPE@* | @BG@*) sep=RESET clause=${rec#@*@} ;;
      *) sep=RESET clause=$rec ;; # START line or raw newline
    esac
    [ "$sep" = AND ] || latched=0

    # Whitespace-anchored comment-tail strip — for the LEAD/verb/latch/waiver
    # decisions + the empty-clause skip ONLY. The token/flag scan below runs
    # over the UN-stripped clause (raw_clause): a whitespace-anchored `#`
    # inside a quoted commit MESSAGE (the repo-standard `-m "task #N: ..."`)
    # is NOT a shell comment, and stripping it before the scan discarded every
    # same-clause token after it — commit pathspecs and a post-message `-a`
    # included — a silent false-ALLOW (round-2 Major; concern
    # hash-in-message-defeats-clause-token-scan).
    raw_clause=$clause
    clause=$(printf '%s' "$clause" | sed -E 's/(^|[[:space:]])#.*$//')
    lead=$(printf '%s' "$clause" | sed -E 's/^[[:space:]{(]+//')
    [ -n "$lead" ] || continue

    # cd-latch ARM (verbatim pull-guard #804 block): latch only provably
    # NON-root targets; unproven targets stay unlatched (fail closed).
    if printf '%s' "$lead" | grep -qE '^cd([[:space:]]|$)'; then
      tgt=$(printf '%s' "$lead" | sed -E 's/^cd[[:space:]]*//' | awk '{print $1}' \
        | sed -E "s/^[\"']//; s/[\"']\$//")
      case "$tgt" in
        *.claude/worktrees/*) latched=1 ;;                # a worktree IS its own tree
        "$REPO" | "$REPO"/*) latched=0 ;;                 # root or a subdir (git walks up)
        '~/explore-persona-space' | '~/explore-persona-space/'*) latched=0 ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/'*) latched=0 ;;
        /* | '~' | '~/'* | '$HOME/'*) latched=1 ;;        # absolute/~-anchored, not the root
        *) latched=0 ;;                                   # relative/variable/empty: unproven
      esac
      continue
    fi

    # `git -C <path>` waiver: waive UNLESS the -C target token literally
    # spells the repo root (one notch stronger than the siblings' path-blind
    # waiver; `git -C .` stays waived — pinned sibling-parity residual).
    if printf '%s' "$clause" | grep -qE '\bgit +-C +'; then
      ctgt=$(printf '%s' "$clause" | sed -E 's/.*\bgit +-C +//' | awk '{print $1}' \
        | sed -E "s/^[\"']//; s/[\"']\$//")
      case "$ctgt" in
        "$REPO" | "$REPO"/) : ;; # root spelling: waiver REFUSED, classify below
        '~/explore-persona-space' | '~/explore-persona-space/') : ;;
        '$HOME/explore-persona-space' | '$HOME/explore-persona-space/') : ;;
        *) continue ;; # worktree / other-repo / `.` target: waived
      esac
    fi

    # Verb classification (command-position anchored).
    verb=""
    if printf '%s' "$lead" | grep -qE "$COMMIT_CMD_ERE"; then
      verb=commit
    elif printf '%s' "$lead" | grep -qE "$ADD_CMD_ERE"; then
      verb=add
    fi
    [ -n "$verb" ] || continue
    [ "$latched" = 1 ] && continue

    if [ "$verb" = commit ]; then
      root_commit=1
    fi

    # Token scan (noglob: a literal `scripts/*.py` token must never expand
    # against the hook's cwd). Quote-adjacent tokens are excluded by the
    # character-class match (plan §4.2: ^(scripts|src|tests)/[^[:space:]"']+).
    # Scans the RAW clause (see the comment-strip note above): a genuine
    # trailing `# comment` naming a gated path or a `-a`-shaped token
    # over-collects toward BLOCK — the designed-safe direction.
    local tok
    set -f
    # shellcheck disable=SC2086
    for tok in $raw_clause; do
      case "$verb:$tok" in
        add:-A | add:--all | add:.) add_all_chained=1 ;;
        commit:--all) has_dash_a=1 ;;
        commit:--*) : ;;
        commit:-[a-zA-Z]*)
          case "$tok" in *a*) has_dash_a=1 ;; esac
          ;;
      esac
      case "$tok" in
        scripts/* | src/* | tests/*)
          case "$tok" in
            *[\"\']*) : ;; # quote-bearing token: not a pathspec
            *) text_paths="$text_paths
$tok" ;;
          esac
          ;;
      esac
    done
    set +f
  done
  return 0
}

# check_certified <path> <landing-sha>: 0 iff a fresh matching v1 cert line
# exists. Malformed lines (non-numeric epoch) never match and never crash the
# arithmetic (block direction).
check_certified() {
  local p="$1" sha="$2" now tag epoch csha cpath
  now=$(date +%s)
  while IFS=' ' read -r tag epoch csha cpath; do
    case "$epoch" in '' | *[!0-9]*) continue ;; esac
    [ "$tag" = v1 ] && [ "$cpath" = "$p" ] && [ "$csha" = "$sha" ] \
      && [ $((now - epoch)) -le "$MAX_AGE" ] && return 0
  done < <(grep -F -- " $p" "$CERT" 2>/dev/null || true)
  return 1
}

run_self_test() {
  local SCRIPT FAILED=0 TMP RART RCODE CERTF STAGED_SHA
  SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' RETURN

  # Repo with artifact-only staged payload.
  RART="$TMP/art" && git init -q "$RART"
  mkdir -p "$RART/tasks" "$RART/figures"
  echo note > "$RART/tasks/t.md" && echo png > "$RART/figures/f.png"
  git -C "$RART" add tasks/t.md figures/f.png

  # Repo with a gated scripts/ file staged (plus one untracked gated file).
  RCODE="$TMP/code" && git init -q "$RCODE"
  mkdir -p "$RCODE/scripts"
  printf 'print(1)\n' > "$RCODE/scripts/issue9_fig.py"
  git -C "$RCODE" add scripts/issue9_fig.py
  printf 'print(2)\n' > "$RCODE/scripts/issue9_new.py" # untracked (compound-add case)
  STAGED_SHA=$(git -C "$RCODE" ls-files -s -- scripts/issue9_fig.py | awk '{print $2}')

  # Repo with a tracked gated file MODIFIED in the worktree, nothing staged:
  # only the commit-clause pathspec / post-message -a can carry the payload
  # (B15/B15b — the #-in-message token-loss regression, round-2 Major).
  local RMOD
  RMOD="$TMP/mod" && git init -q "$RMOD"
  mkdir -p "$RMOD/scripts"
  printf 'print(1)\n' > "$RMOD/scripts/issue9_fig.py"
  git -C "$RMOD" add scripts/issue9_fig.py
  git -C "$RMOD" -c user.email=t@t -c user.name=t commit -q -m init
  printf 'print(2)\n' > "$RMOD/scripts/issue9_fig.py" # modified, UNSTAGED

  CERTF="$TMP/cert.txt"

  run_case() {
    local desc="$1" expect="$2" cmdstr="$3" repo="$4" envflag="${5:-}"
    local rc=0
    if [ -n "$envflag" ]; then
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | EPM_ALLOW_ROOT_CODE_COMMIT=1 EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_ROOT_CODE_COMMIT EPM_ROOT_CODE_COMMIT_REPO="$repo" \
          EPM_INLINE_CERT_PATH="$CERTF" bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- must ALLOW (exit 0) ---
  run_case "A1 artifact-only staged commit" 0 'git commit -m x' "$RART"
  run_case "A2 non-git command" 0 \
    'uv run python scripts/task.py post-marker 9 epm:progress --note commit' "$RART"
  run_case "A3 git non-commit (push)" 0 'git push origin main' "$RCODE"
  run_case "A4 worktree -C commit with gated staged at root" 0 \
    'git -C "$WT" commit -m x' "$RCODE"
  run_case "A5 cd-latched worktree commit" 0 \
    'cd .claude/worktrees/issue-9 && git commit -m x' "$RCODE"
  run_case "A7a inline escape hatch" 0 'EPM_ALLOW_ROOT_CODE_COMMIT=1 git commit -m x' "$RCODE"
  run_case "A7b session env escape hatch" 0 'git commit -m x' "$RCODE" env
  run_case "A9 heredoc message mentioning a commit command (artifact-only)" 0 \
    'git commit -m "$(cat <<EOF
fix: never git commit -m at the root
EOF
)"' "$RART"
  run_case "A11 non-gated code-adjacent staged" 0 'git commit -m x' "$RART"

  # --- must BLOCK (exit 2) ---
  run_case "B1 gated staged, no cert" 2 'git commit -m x' "$RCODE"
  run_case "B6 pathspec form, no cert" 2 'git commit -m x scripts/issue9_fig.py' "$RCODE"
  run_case "B7 -C spelling the repo root: waiver refused" 2 \
    "git -C $REPO commit -m x" "$RCODE"
  run_case "B8 classification failure fails CLOSED" 2 'git commit -m x' "$TMP/notarepo"
  run_case "B12 compound add+commit of an untracked gated file" 2 \
    'git add scripts/issue9_new.py && git commit -m x' "$RCODE"
  run_case "B13 blanket add -A chained: fail CLOSED" 2 \
    'git add -A && git commit -m x' "$RART"
  run_case "B15 pathspec after #-bearing message" 2 \
    'git commit -m "task #9: fix" scripts/issue9_fig.py' "$RMOD"
  run_case "B15b post-message -a after #-bearing message" 2 \
    'git commit -m "task #9: fix" -a' "$RMOD"
  run_case "A13 artifact-only commit with #-bearing message" 0 \
    'git commit -m "task #9: docs"' "$RART"

  # A6 fresh matching cert allows; B3 wrong-sha cert blocks.
  printf 'v1 %s %s scripts/issue9_fig.py\n' "$(date +%s)" "$STAGED_SHA" > "$CERTF"
  run_case "A6 fresh matching cert" 0 'git commit -m x' "$RCODE"
  printf 'v1 %s %s scripts/issue9_fig.py\n' "$(date +%s)" "0000000000000000000000000000000000000000" > "$CERTF"
  run_case "B3 wrong-blobsha cert" 2 'git commit -m x' "$RCODE"
  rm -f "$CERTF"

  # A16 fail-soft trio: empty command / malformed JSON / missing field.
  local rc=0
  jq -n '{tool_input: {command: ""}}' \
    | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16a empty command" \
    || { echo "FAIL (got exit $rc, want 0): A16a empty command"; FAILED=1; }
  rc=0
  printf 'not-json' | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16b malformed stdin JSON" \
    || { echo "FAIL (got exit $rc, want 0): A16b malformed stdin JSON"; FAILED=1; }
  rc=0
  jq -n '{tool_input: {}}' \
    | env -u EPM_ALLOW_ROOT_CODE_COMMIT bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  [ "$rc" -eq 0 ] && echo "PASS (exit 0): A16c missing command field" \
    || { echo "FAIL (got exit $rc, want 0): A16c missing command field"; FAILED=1; }

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

# Session-env escape hatch (no-fork case form — this hook runs on EVERY Bash
# call, so the common path must stay subprocess-free before the jq parse).
case "${EPM_ALLOW_ROOT_CODE_COMMIT:-}" in
  1 | true | TRUE | True | yes | YES | Yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0 # fail-soft (A16 parity)
[ -n "$cmd" ] || exit 0
case "$cmd" in *EPM_ALLOW_ROOT_CODE_COMMIT=1*) exit 0 ;; esac # inline escape hatch
# Cheap prefilters: both substrings must co-occur before any further work —
# ~all fleet traffic exits here.
case "$cmd" in *git*) ;; *) exit 0 ;; esac
case "$cmd" in *commit*) ;; *) exit 0 ;; esac

# ---- Layer 1 ----
classify_cmd "$cmd"
[ "$root_commit" = 1 ] || exit 0

# Blanket stage chained to a root commit: the landing set is unknowable at
# PreToolUse time -> FAIL CLOSED.
if [ "$add_all_chained" = 1 ]; then
  echo "BLOCKED: 'git add -A|.|--all' chained to a repo-root commit — the landing set cannot be classified at hook time, and blanket staging is banned at the shared root (CLAUDE.md § Concurrent repo-root committers). Stage by explicit path, run the inline payload lint gate on any scripts/src/tests payload (uv run python scripts/inline_lint_gate.py --issue <N> --payload-file <paths.txt>), then commit. Deliberate override: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
  exit 2
fi

# ---- Layer 2: repo-state classification (authoritative; FAIL CLOSED) ----
if ! staged=$(git -C "$GUARD_REPO" diff --cached --name-only 2>/dev/null); then
  echo "BLOCKED: guard_root_code_commit.sh could not read the staged set (git diff --cached failed) for a repo-root commit — cannot classify the payload; failing CLOSED (#458/#1147 class). Retry, or override deliberately: EPM_ALLOW_ROOT_CODE_COMMIT=1." >&2
  exit 2
fi
mod=""
[ "$has_dash_a" = 1 ] && mod=$(git -C "$GUARD_REPO" diff --name-only 2>/dev/null || true)
pending=$(printf '%s\n%s\n%s\n' "$staged" "$mod" "$text_paths" \
  | grep -E "$GATED_PATH_ERE" | sort -u)
[ -n "$pending" ] || exit 0 # artifact-only / non-code commit: allow

# Cert check per gated path; deletions exempt. BINDING RULE: bind the cert to
# the LANDING content — `git commit -a` re-stages WORKTREE content of tracked
# modified files at commit time, and a pathspec commit likewise commits
# worktree content, so for any path covered by -a, named as a commit pathspec,
# or named in a chained add clause, the landing content is the WORKTREE file;
# the staged blob sha is authoritative ONLY for a plain commit of the staged
# set. Space-safe iteration (while read, never for-in word-split): a gated
# path containing a space must fail toward BLOCK, never silently allow.
uncertified=""
while IFS= read -r p; do
  [ -n "$p" ] || continue
  worktree_shape=0 # 1 => landing content is the worktree file
  if [ "$has_dash_a" = 1 ] \
    && git -C "$GUARD_REPO" diff --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    worktree_shape=1 # -a re-stages worktree content
  fi
  if printf '%s\n' "$text_paths" | grep -qxF -- "$p"; then
    worktree_shape=1 # commit pathspec / chained add-clause
  fi
  if [ "$worktree_shape" = 1 ]; then
    [ -f "$GUARD_REPO/$p" ] || continue # deletion via -a/pathspec: exempt
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  elif git -C "$GUARD_REPO" diff --cached --name-only -- "$p" 2>/dev/null | grep -qxF -- "$p"; then
    sha=$(git -C "$GUARD_REPO" ls-files -s -- "$p" 2>/dev/null | awk '{print $2}')
    [ -n "$sha" ] || continue # staged DELETION: exempt
  else
    [ -f "$GUARD_REPO/$p" ] || continue
    sha=$(git -C "$GUARD_REPO" hash-object -- "$GUARD_REPO/$p" 2>/dev/null || true)
  fi
  check_certified "$p" "$sha" || uncertified="$uncertified $p"
done <<EOF_PENDING
$pending
EOF_PENDING

[ -z "${uncertified:-}" ] && exit 0
cat >&2 <<BLOCK_MSG
BLOCKED: repo-root commit carries UNCERTIFIED code payload:${uncertified}
Direct-to-main code (scripts/src/tests) must pass the inline payload lint gate
first (SKILL.md Step 9a-ter § Inline payload lint gate, #1388/#1460/#1500):
  printf '%s\n' <paths> > /tmp/issue-<N>-inline-payload.txt
  uv run python scripts/inline_lint_gate.py --issue <N> \\
    --payload-file /tmp/issue-<N>-inline-payload.txt     # ONE background Bash (~3-8 min)
On PASS it certifies each path's exact content; re-run after any further edit.
Committing in a WORKTREE instead? Use git -C "\$WT" commit (worktrees are
gated at Step 10d, not here). NEVER hand-write ${CERT} (#1082 parity).
Genuinely pre-existing red on a MODIFIED payload file the gate refused, or an
emergency fleet repair: prefix the commit with EPM_ALLOW_ROOT_CODE_COMMIT=1
and record the reason in an epm:progress note.
BLOCK_MSG
exit 2
