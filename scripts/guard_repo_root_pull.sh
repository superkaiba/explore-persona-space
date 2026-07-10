#!/usr/bin/env bash
# PreToolUse(Bash) guard: block a hand-rolled `git pull` against the SHARED
# repo-root working tree (task #1201). CLAUDE.md § Concurrent repo-root
# committers prescribes `uv run python scripts/sync_repo_root.py` as the ONLY
# repo-root sync; the prose alone failed open (#967: a hand-rolled root pull
# died `fatal: Cannot autostash` under a held lock/husk; #711: a concurrent
# `git pull --rebase=merges` orphaned a task-state commit). sync_repo_root.py's
# own pull is a Python subprocess (`_run_bounded(_pull_argv(repo))`) and never
# passes PreToolUse — no helper sentinel is needed (plan #1201 §4.4 / D2).
#
# PREDICATE (plan #1201 §4.1): block every NON-ff `git pull` whose effective
# tree is the repo root — including the hand-rolled recovery form
# `git pull --rebase=merges --autostash` (no live canonical snippet prescribes
# it at the root; it reproduces the exact hazards the helper exists to fix)
# and the config-override merge-pull (`git pull --no-rebase`,
# `git -c pull.rebase=false pull` — closes the branch guard's gap (xvi)).
# ALLOW: worktree/other-repo pulls (`git -C <path>`, a provably-non-root
# cd-latch), `--ff-only` pulls anywhere (an ff pull cannot autostash-die,
# husk, or drop commits — and it carries every canonical pod-sync recipe),
# ssh/scp/grep-family clauses (with the #1098 exceptions below), heredocs,
# and the EPM_ALLOW_ROOT_PULL=1 escape hatch.
#
# Deliberately conservative, sibling parity (guard_piped_git_push.sh #1048 /
# guard_repo_root_branch.sh #804..#1193): classification is COMMAND-POSITION
# ANCHORED (#1250 — the clause LEAD must be a git invocation whose verb is
# pull, see PULL_CMD_ERE); the whole-clause raw scan is retained ONLY for
# waiver-refused ssh/scp/grep-family clauses (their quoted content is exactly
# what must stay scannable). Still NO quote-stripping (the #796 revert stands
# — nothing is stripped, the detector is anchored). Heredoc-bearing commands
# blanket-allowed, fail-soft on any parse ambiguity, cd-latch propagates ONLY
# across `&&` (the #804 latch law).
#
# <!-- known limitations -->
# - CWD-BLIND + cross-call cwd FP: the Bash tool's cwd PERSISTS between
#   calls and this guard matches COMMAND TEXT only — a bare `git pull`
#   issued after a PRIOR call's `cd ~/other-repo` still blocks (the guard
#   cannot see the inherited cwd). Remediation: `git -C <path> pull`, or a
#   same-call `cd <path> && git pull` (the cd-latch proves that one).
# - Non-ff multi-statement ssh pod pulls false-block via the mis-split tail
#   clause: a quoted `&&`/`;` inside an ssh remote string splits raw (the
#   #796 no-quote-parse trade-off), so the tail clause loses the ssh command
#   word and classifies on its own. Canonical pod-sync recipes carry
#   --ff-only, which the ff-only waiver honors even on the mis-split tail
#   (A8/A22); a non-ff remote pull needs `git -C /workspace/... pull` inside
#   the remote string (the -C waiver is mis-split-proof) or the SSH MCP.
#   Mis-split TAIL clauses now classify under the command-position anchor
#   (#1250): a tail whose lead is not a git/wrapper/keyword unit no longer
#   FPs on mention text (FP reduction), while a tail whose lead IS the pull
#   literal or a keyword+pull (S3) stays fail-closed.
# - Path-blind `git -C` waiver (guard_repo_root_branch.sh:974 parity):
#   `git -C <repo-root-path> pull` AND `git -C . pull` issued from the root
#   pass the hook (pinned A20/A21). The block message's "NEVER point -C at
#   the repo root" line is the stated control.
# - Quoted mentions + residuals of the #1250 command-position anchor:
#   (a) a quoted `-m` / `--note` mention under a NON-git lead no longer
#       blocks (#1250; S1 flipped, A23/A25).
#   (b) NEW fail-open residual — wrapper shapes outside the closed prefix
#       allowlist fall OPEN: `bash -c '<pull>'` (pinned A24),
#       flag-with-VALUE wrapper shapes (`sudo -u <user> git pull`, pinned
#       A27), quoted `eval "<pull>"` (pinned A28). Accidents-not-adversaries
#       disposition: nobody types these by accident.
#   (b') NEW fail-open residual — EXPANSION-CHANNEL pulls under an ordinary
#       lead fall OPEN: command substitution / backticks (pinned A26);
#       quoted-space env values (`NAME="a b" git pull`) are the same family.
#       Under WAIVED leads (ssh/scp/grep family) the expansion-syntax
#       refusal still routes to the arm-2 whole-clause scan and blocks.
#   (b'') a waived-word lead behind a prefix (`VAR=1 ssh ...`) never enters
#       the waiver arm and falls open — same family as (b).
#   (c) FP residual — a quoted mention that embeds a shell separator + a
#       full pull command literal (incl. keyword-led tails: `; then/do git
#       pull ...`) mis-splits raw and its tail clause still blocks (pinned
#       S3); the `--file`/`-F`/heredoc remediation stays for that shape
#       (the heredoc blanket-allow also hides a real same-call pull, the
#       pinned A19 known miss).
#   (d) FP residual — waiver-refused ssh/scp/grep clauses keep the
#       whole-clause raw scan BY DESIGN (B13-B16, B23), so a mention inside
#       such a clause still blocks (pinned S4 — e.g. a piped grep whose
#       pattern quotes the pull bigram; run it un-piped (A12) or via the
#       Grep tool).
#
# Escape hatch: EPM_ALLOW_ROOT_PULL=1 — honored both as session env and as an
# inline prefix on the command itself (`EPM_ALLOW_ROOT_PULL=1 git pull`).
#
# Contract: reads the PreToolUse JSON on stdin, exits 0 to allow, exits 2
# (blocking, stderr fed back to Claude) to refuse. Exit 2 is the documented
# PreToolUse blocking exit code; any OTHER non-zero is non-blocking.
#
# Self-test: bash scripts/guard_repo_root_pull.sh --self-test
set -u

REPO=/home/thomasjiralerspong/explore-persona-space

# Flag-tolerant verb anchor (the #897 detector shape): `git [flags] pull`,
# verb terminated by space/EOL so `git pull-request` / `pulled` never match.
# The flag group's optional value token also covers the config-override form
# `git -c pull.rebase=false pull ...` (case B12).
PULL_ERE='\bgit +(-[^ ]+( +[^ ]+)?( +|$))*pull([[:space:]]|$)'

# Command-position anchor (#1250): the clause LEAD must BE a git invocation
# whose verb is pull — optionally behind env assignments, a CLOSED wrapper
# set, shell compound keywords, bare -flag tokens, and a timeout-style
# duration token. Every prefix unit errs CLOSED (a spurious unit match can
# only ADD a block, never an allow); an unlisted lead word (uv, bash, xargs,
# echo, ...) makes the predicate unmatchable — that is the #1250 fix.
PULL_CMD_ERE='^([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+|(nohup|setsid|sudo|env|time|timeout|command|exec|eval|if|elif|then|else|while|until|do|!)[[:space:]]+|-[^[:space:]]+[[:space:]]+|[0-9]+([.][0-9]+)?[smhd]?[[:space:]]+)*git[[:space:]]+(-[^[:space:]]+([[:space:]]+[^[:space:]]+)?([[:space:]]+|$))*pull([[:space:]]|$)'

# Classify one command string. Returns 0 = allow, 1 = block.
check_cmd() {
  local cmd="$1"

  # Inline escape hatch: EPM_ALLOW_ROOT_PULL=1 anywhere in the command.
  case "$cmd" in *EPM_ALLOW_ROOT_PULL=1*) return 0 ;; esac

  # Heredoc blanket-allow (sibling precedent; load-bearing for doc/commit
  # text describing this very rule; also covers <<< here-strings). Documented
  # known miss: a heredoc-bearing command with a REAL same-call pull (A19).
  case "$cmd" in *'<<'*) return 0 ;; esac

  # Cheap prefilter (this hook runs on EVERY Bash call): both substrings must
  # co-occur before ANY subprocess (sed/grep) beyond the single jq parse runs.
  case "$cmd" in *git*) ;; *) return 0 ;; esac
  case "$cmd" in *pull*) ;; *) return 0 ;; esac

  # Bash line-continuation normalization (`\<CR?><NL>` -> space), verbatim
  # the sibling pre-pass: bash joins the physical lines pre-execution.
  cmd=$(printf '%s' "$cmd" | sed -zE 's/\\\r?\n/ /g')

  # `|&` -> `|` rewrite, then strip `&`-bearing redirection operators (they
  # carry no command semantics and would sever clauses at the `&` split —
  # the guard_piped_git_push.sh §4.1-step-7 shape); THEN clause-split with
  # separator sentinels, two-char separators first. Raw newlines and the
  # un-sentineled first line get RESET semantics (the #804 latch law: a
  # cd-latch propagates ONLY across `&&`).
  local records
  records=$(printf '%s\n' "$cmd" \
    | sed -E 's/\|&/|/g' \
    | sed -E 's/[0-9]*>&[0-9]*//g; s/&>>?[^[:space:]]*//g' \
    | sed -E 's/&&/\n@AND@/g; s/\|\|/\n@OR@/g; s/;/\n@SEQ@/g; s/\|/\n@PIPE@/g; s/&/\n@BG@/g')

  local -a recs
  mapfile -t recs <<< "$records"

  local n=${#recs[@]} i rec sep nextsep clause lead tgt latched=0 scanwhole=0
  for ((i = 0; i < n; i++)); do
    rec=${recs[i]}
    case "$rec" in
      @AND@*) sep=AND clause=${rec#@AND@} ;;
      @OR@* | @SEQ@* | @PIPE@* | @BG@*) sep=RESET clause=${rec#@*@} ;;
      *) sep=RESET clause=$rec ;; # START line or raw newline
    esac
    [ "$sep" = AND ] || latched=0
    scanwhole=0 # #1250: per-clause; set ONLY by the waived-word arm below
    # FOLLOWING separator (for the waiver's producer-position refusal): a
    # waived clause feeding a pipe (or mis-split `&`, incl. a stripped-off
    # fd-dup's residue) can hand gated text to a local shell consumer.
    if ((i + 1 < n)); then
      case "${recs[i + 1]}" in
        @PIPE@*) nextsep=PIPE ;;
        @BG@*) nextsep=BG ;;
        *) nextsep=OTHER ;;
      esac
    else
      nextsep=END
    fi

    # Whitespace-anchored comment-tail strip (sibling shape; also removes
    # pure-comment lines), then skip empty clauses.
    clause=$(printf '%s' "$clause" | sed -E 's/(^|[[:space:]])#.*$//')
    lead=$(printf '%s' "$clause" | sed -E 's/^[[:space:]{(]+//')
    [ -n "$lead" ] || continue

    # cd-latch ARM: latch only provably-NON-root targets; fail CLOSED on
    # relative / variable / empty targets (cannot prove the cwd — S2 pins
    # `cd "$WT" && git pull` as expected-block; remediation is the canonical
    # `git -C "$WT" pull ...` form).
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

    # Waived command words: remote execution (ssh/scp) + read-only pattern
    # tools (grep family) — inheriting guard_repo_root_branch.sh's #1098
    # exceptions: (1) NOT in pipeline-producer / background position (a
    # waived producer's stdout can feed a local shell — `ssh h 'echo git
    # pull ...' | bash`); (2) no locally-executing expansion syntax
    # ($( ${ backtick <( >( — expansion runs LOCALLY at this cwd); (3) no
    # local file OUTPUT redirect except exactly /dev/null (strip-then-scan —
    # the write-then-execute channel `... > /tmp/x; bash /tmp/x`); (4) [ssh]
    # no ProxyCommand/LocalCommand/KnownHostsCommand token (all execute
    # LOCALLY) and no shared-repo path in a covered spelling — an
    # ssh-to-this-VM remote string operating on the shared root stays
    # blocked. A refused waiver falls through to classification (where the
    # ff-only waiver still saves canonical pod syncs).
    case "$lead" in
      ssh\ * | scp\ * | grep\ * | egrep\ * | fgrep\ * | rg\ *)
        # #1250: a clause whose waiver is REFUSED falls through carrying
        # scanwhole=1 and keeps the OLD whole-clause scan (arm 2) — its
        # quoted content is exactly what must stay scannable (B13-B16, B23).
        # A clause whose waiver SUCCEEDS `continue`s before classification.
        # Shared arm top, NOT the ssh sub-branch (B23 pins the grep route).
        scanwhole=1
        if [ "$nextsep" != PIPE ] && [ "$nextsep" != BG ] \
          && ! printf '%s' "$clause" | grep -qE '\$\(|\$\{|`|<\(|>\(' \
          && ! printf '%s' "$clause" \
            | sed -E 's@[0-9]*>>?[[:space:]]*/dev/null([[:space:]]|$)@ @g' \
            | grep -q '>'; then
          case "$lead" in
            ssh\ *)
              if ! printf '%s' "$clause" | grep -qiE 'proxycommand|localcommand|knownhostscommand'; then
                case "$clause" in
                  *"$REPO"* | *'~/explore-persona-space'* | *'$HOME/explore-persona-space'*)
                    : ;; # shared-root spelling -> classify (blocks)
                  *) continue ;; # remote-host git op -> waive this clause
                esac
              fi
              ;;
            *) continue ;; # scp / read-only pattern argument -> waive
          esac
        fi
        ;;
    esac

    # Per-clause `git -C <path>` waiver — path-blind, sibling-line-974
    # parity (A20/A21 pinned; the block message's -C line is the control).
    printf '%s' "$clause" | grep -qE '\bgit +-C +' && continue

    if [ "$scanwhole" = 1 ]; then
      printf '%s' "$clause" | grep -qE "$PULL_ERE" || continue
    else
      printf '%s' "$lead" | grep -qE "$PULL_CMD_ERE" || continue
    fi
    case "$clause" in *--ff-only*) continue ;; esac # ff cannot autostash-die/husk/drop
    [ "$latched" = 1 ] && continue
    return 1
  done
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
        | EPM_ALLOW_ROOT_PULL=1 bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    else
      jq -n --arg c "$cmdstr" '{tool_input: {command: $c}}' \
        | env -u EPM_ALLOW_ROOT_PULL bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
    fi
    if [ "$rc" -eq "$expect" ]; then
      echo "PASS (exit $rc): $desc"
    else
      echo "FAIL (got exit $rc, want $expect): $desc"
      FAILED=1
    fi
  }

  # --- plan #1201 §4.5 + the #1250 round (B17-B23, S3-S4): must BLOCK (exit 2) ---
  run_case "B1 bare pull (#967 shape)" 2 'git pull'
  run_case "B2 refspec form" 2 'git pull origin main'
  run_case "B3 hand-rolled recovery form" 2 'git pull --rebase=merges --autostash origin main'
  run_case "B4 braced recovery loop (|| not a pipe)" 2 \
    'git push origin main || { git pull --rebase=merges --autostash && git push origin main; }'
  run_case "B5 --no-rebase merge-pull (gap xvi)" 2 'git pull --no-rebase'
  run_case "B6 plain --rebase (#711 hazard)" 2 'git pull --rebase origin main'
  run_case "B7 relative cd never latches" 2 'cd scripts && git pull'
  run_case "B8 explicit root cd never latches" 2 \
    "cd $REPO && git pull --rebase=merges --autostash"
  # B9: REAL embedded newline (jq --arg encodes it; the guard's jq -r decodes
  # it back) — the dominant Bash-tool multi-line delivery shape. Keep the
  # newline literal; collapsing it to one line degrades the case.
  run_case "B9 raw-newline multi-line" 2 'echo pre
git pull'
  run_case "B10 pull on its own && segment" 2 'git fetch origin && git pull'
  run_case "B11 semicolon resets the latch (#804)" 2 'cd ~/overleaf-6a2df2d2; git pull'
  run_case "B12 config-override merge-pull" 2 'git -c pull.rebase=false pull origin main'
  run_case "B13 ssh naming the shared-root spelling is NOT waived" 2 \
    "ssh cia-benchmark-vm 'git --work-tree=\$HOME/explore-persona-space pull origin main'"
  run_case "B14 waived word in pipe-producer position feeding a shell" 2 \
    "ssh pod-779 'echo git pull origin main' | bash"
  run_case "B15 waived word writing a local file (write-then-execute)" 2 \
    "ssh pod-779 'echo git pull origin main' > /tmp/x"
  run_case "B16 ProxyCommand executes locally: waiver refused" 2 \
    "ssh -o ProxyCommand='git pull origin main %h' pod-779 'true'"
  run_case "S2 pinned expected-block: variable cd target unprovable" 2 \
    'cd "$WT" && git pull'
  # --- #1250 round: prefix-unit blocks + pinned retained-FP residuals ---
  run_case "B17 wrapper word prefix (nohup)" 2 'nohup git pull'
  run_case "B18 env-assignment prefix" 2 'GIT_TRACE=1 git pull --rebase origin main'
  run_case "B19 wrapper + duration token" 2 'timeout 300 git pull'
  run_case "B20 builtin bypass wrapper" 2 'command git pull'
  run_case "B21 wrapper + bare-flag unit" 2 'sudo -n git pull'
  run_case "B22 keyword-lead retry loop (REQUIRED keyword units' pin)" 2 \
    'until git pull --rebase; do sleep 5; done'
  run_case "B23 grep refused waiver (redirect) routes to arm-2 whole-clause scan" 2 \
    'grep -rn "git pull --rebase" .claude/ scripts/ > /tmp/hits.txt'
  run_case "S3 pinned FP residual: mis-split note mention (separator + full pull literal)" 2 \
    "uv run python scripts/task.py post-marker 1250 epm:progress --note 'recovered: git fetch && git pull --rebase worked'"
  run_case "S4 pinned FP residual: piped grep mention (producer-position waiver refusal)" 2 \
    'grep -rn "git pull --rebase" .claude/ | head -5'

  # --- plan #1201 §4.5 + the #1250 round (S1 flipped, A23-A28): must ALLOW (exit 0) ---
  run_case "A1 verbatim Step 10d form (1)" 0 \
    'git -C "$WT" pull --rebase=merges --autostash && git -C "$WT" push origin issue-1201'
  run_case "A2 literal worktree -C" 0 \
    'git -C .claude/worktrees/issue-1201 pull --rebase=merges --autostash origin main'
  run_case "A3 Overleaf-clone cd latch (~-anchored non-repo)" 0 'cd ~/overleaf-6a2df2d2 && git pull'
  run_case "A4 absolute non-repo path latch" 0 'cd /tmp/scratch-clone && git pull'
  run_case "A5 relative worktree path latch" 0 'cd .claude/worktrees/issue-1201 && git pull'
  run_case "A6 ff-only waiver" 0 'git pull --ff-only origin main'
  run_case "A7 single-statement pod sync (ssh waiver)" 0 \
    "ssh pod-779 'git pull --ff-only origin main'"
  run_case "A8 multi-statement pod sync (mis-split tail saved by ff-only)" 0 \
    "ssh epm-issue-228 'cd /workspace/explore-persona-space && git pull --ff-only origin main && uv sync --locked'"
  run_case "A9 fetch is not pull" 0 'git fetch origin main'
  run_case "A10 verb-terminator lookalike" 0 'git pull-request --help'
  run_case "A11 canonical Step 10d form (2) - the prescribed remediation" 0 \
    'git push origin main || uv run python scripts/sync_repo_root.py'
  run_case "A12 grep-family pattern argument" 0 'grep -rn "git pull --rebase" .claude/ scripts/'
  run_case "A13 heredoc doc text" 0 'cat > /tmp/note.md <<EOF
never run git pull at the shared root
EOF'
  run_case "A14 inline escape hatch" 0 'EPM_ALLOW_ROOT_PULL=1 git pull'
  run_case "A15 env escape hatch" 0 'git pull | tail -3' env
  run_case "A17 other-repo -C form" 0 'git -C ~/overleaf-6a2df2d2 pull'
  run_case "A18 no pull anywhere; pipe hygiene" 0 'git log --oneline | head -5'
  run_case "A19 heredoc DOCUMENTED KNOWN MISS (real same-call pull)" 0 \
    'git commit -F /tmp/msg.txt <<EOF
msg
EOF
git pull'
  run_case "A20 path-blind -C waiver at the root (pinned limitation)" 0 \
    "git -C $REPO pull"
  run_case "A21 git -C . at the root (pinned sibling-parity limitation)" 0 'git -C . pull'
  run_case "A22 piped pod sync saved by ff-only after waiver refusal" 0 \
    "ssh pod-779 'git pull --ff-only origin main' 2>&1 | tail -20"
  # --- #1250 round: S1 flipped + the incident + one pin per fail-open residual class ---
  run_case "S1 FLIPPED #1250: quoted -m mention under a git-lead command (command-position anchor)" 0 \
    'git commit -m "never hand-roll git pull at the root" && git push origin main'
  run_case "A23 incident #1250: separator-free post-marker --note mention (lead uv)" 0 \
    "uv run python scripts/task.py post-marker 1201 epm:progress --note 'worked around via git pull --rebase mention'"
  run_case "A24 wrapper-word residual pin (bash -c, refspec form)" 0 \
    "bash -c 'git pull --rebase origin main'"
  run_case "A25 mention under echo lead" 0 'echo "git pull is fenced at the root"'
  run_case "A26 expansion-channel residual pin (command substitution under ordinary lead)" 0 \
    'echo "$(git pull --rebase origin main)"'
  run_case "A27 flag-with-value wrapper residual pin" 0 'sudo -u deploy git pull origin main'
  run_case "A28 quoted-eval residual pin" 0 'eval "git pull origin main"'

  # A16 fail-soft trio: empty command / malformed JSON / missing field.
  local rc=0
  jq -n '{tool_input: {command: ""}}' \
    | env -u EPM_ALLOW_ROOT_PULL bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A16a empty command"
  else
    echo "FAIL (got exit $rc, want 0): A16a empty command"
    FAILED=1
  fi
  rc=0
  printf 'not-json' | env -u EPM_ALLOW_ROOT_PULL bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A16b malformed stdin JSON"
  else
    echo "FAIL (got exit $rc, want 0): A16b malformed stdin JSON"
    FAILED=1
  fi
  rc=0
  jq -n '{tool_input: {}}' \
    | env -u EPM_ALLOW_ROOT_PULL bash "$SCRIPT" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "PASS (exit 0): A16c missing command field"
  else
    echo "FAIL (got exit $rc, want 0): A16c missing command field"
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

# Session-env escape hatch (no-fork case form — this hook runs on EVERY Bash
# call, so the common path must stay subprocess-free before the jq parse).
case "${EPM_ALLOW_ROOT_PULL:-}" in
  1 | true | TRUE | True | yes | YES | Yes) exit 0 ;;
esac

cmd=$(jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

if ! check_cmd "$cmd"; then
  cat >&2 <<'BLOCK_MSG'
BLOCKED: hand-rolled `git pull` at the SHARED repo root. Concurrent sessions
commit here in parallel; an unserialized pull races them and dies
`fatal: Cannot autostash` under a held lock/husk (#967) or orphans a
concurrent commit (#711). The ONLY repo-root sync is:
  uv run python scripts/sync_repo_root.py
(flock-serialized, untracked-collision sweep, bounded rebase, stash recovery).
Worktree pulls stay open: git -C "$WT" pull --rebase=merges --autostash
(NEVER point -C at the repo root — `git -C .` from the root is the same op).
`git pull --ff-only` is allowed anywhere. This guard matches COMMAND TEXT,
not cwd: after a prior call's `cd` into another repo, use `git -C <path> pull`.
A plain MENTION of `git pull` under a non-git command no longer blocks
(#1250); commit/note text that embeds a full pull command after a shell
separator can still mis-split and block — use `git commit -F <file>` /
`task.py post-marker --file <path.md>` (or a heredoc) for such text.
Deliberate override: prefix with EPM_ALLOW_ROOT_PULL=1.
BLOCK_MSG
  exit 2
fi
exit 0
