#!/usr/bin/env bash
# PreToolUse(Read|Bash|Grep|mcp__ssh__ssh_execute) guard: harmful-bank files
# are DIGEST-ONLY (CLAUDE.md § Spurious usage-policy refusals, clause (d)).
# Mechanizes the #888 prose rule; incident #866: four implementer sessions
# refusal-killed after bank item text was paged into context.
#
# Deny set = the six harmful safety-benchmark question banks under
# src/explore_persona_space/artifacts/query_banks/ (advbench / strongreject /
# betley_main8 / wang44 / broad_em_train / sensitive_info_requests — the
# clause-(d) families; china_sensitive_v1.json is deliberately EXCLUDED, plan
# #965 §11.8). Matching is version-bump tolerant (raw regex) AND realpath-aware
# (cwd-relative reads + symlinks resolve to the bank).
#
# Arms:
#   Read  — deny any bank file_path (any offset/limit slice is item text).
#   Grep  — deny output_mode "content" on a bank FILE or the query_banks DIR
#           (banks are ~one item per line; a content match IS item text);
#           files_with_matches / count modes are digests -> allowed.
#   Bash  — raw scan over operator-padded word-split tokens. The DENY
#           co-occurrence (bank token + paging verb) stays WHOLE-COMMAND
#           (deliberately NO clause/pipe split for the deny side, plan #965
#           §11.3 — a clause splitter fails OPEN on quoted pipes), but the
#           per-instance grep/git/jq SAFE-FLAG attribution closes at
#           command-unit BOUNDARIES (#1152): padded `| & ; ( )` + backtick
#           tokens (newlines mapped to `;` via tr) latch-or-evaluate + reset the
#           open instance, so a later unit's flags cannot launder an
#           earlier unit's instance (`grep harmful <bank> && ls -l` denies)
#           and a later unit's flags no longer false-deny an earlier safe
#           one (`git log -- <bank> && mkdir -p /tmp/x` allows). `<`/`>`
#           are NOT boundaries — redirects don't start a new command unit.
#           GIT CARRY RULE: an UNRESOLVED git instance (in_git=1,
#           subcommand not yet seen — e.g. the padded `(` of
#           `git -C $(pwd) show ...` fires a boundary BEFORE the
#           subcommand token) CARRIES across the boundary; resetting it
#           would detach the attribution (fail-OPEN on show/log -p,
#           false-deny on diff --stat via the bare-diff branch).
#           Fail-closed residue of the carry: a subcommand-less git
#           (`git add <bank> && diff x y`) can mis-attribute a later
#           unit's show/diff/log to git — extra DENY, never extra allow.
#           Denies: text-paging verbs (cat/head/sed/awk/comm/join/
#           json.tool/...) co-occurring with a bank token; bare `diff`
#           OUTSIDE a git instance (diff /dev/null <bank> prints item
#           text; `diff` cannot join the verb regex because `git diff
#           --stat` walks a `diff` token — the same-unit `git` precedes
#           it, so in_git guards the digest forms); grep-family line
#           output with PER-EXECUTABLE safe flags (grep/egrep/fgrep:
#           -c/-q/-l/-L digests; rg: -c/-q/-l only — rg -L is --follow,
#           it PRINTS lines) tracked PER INSTANCE (a later safe grep
#           cannot launder an earlier unsafe one); git show/diff/log -p
#           paging a bank (digest forms --stat / --name-only / --numstat
#           / --oneline allowed); jq with a non-digest filter (allowed
#           single-word digests: keys / keys_unsorted / length / type /
#           empty); and task.py --file/--body-file on a bank file (would
#           embed bank items into task state — events.jsonl / body.md /
#           plans; non-task.py `--file <bank>` pipeline consumption stays
#           allowed). Everything else (python/uv pipelines, cp, git
#           add/commit, wc, sha256sum, stat, du, redirects) is allowed by
#           default — banks exist to be consumed in-process by pipeline
#           code.
#           NOTE EXEMPTION (#1152): on a task.py-shaped command
#           (`*task.py*` glob — matches ANY path named task.py;
#           deliberate, acceptable under the python-allowed posture),
#           quoted `--note` strings are blanked BEFORE the fast path:
#           single-quoted always; double-quoted ONLY when free of $ and
#           backtick (a note carrying $(...) / `...` DOES execute — left
#           visible to the walk, fail-closed). Notes are DATA arguments
#           to a Python CLI, never executed; the scrub defends against
#           ACCIDENTAL false positives (progress notes naming a bank path
#           + a paging verb), NOT crafted quote-splicing — deliberate
#           evasion is already dominated by the standing python allowance
#           + the documented escape hatch.
#           Accepted fail-CLOSED FP shapes (extra deny, never extra
#           allow; remediation named in the deny message): a digest flag
#           AFTER a quoted-operator pattern (`grep -E 'foo|bar' -c
#           <bank>` — the quoted `|` pads into a boundary that latches
#           the still-unsafe instance; put -c/-q/-l BEFORE the pattern),
#           the `&` inside `2>&1` when the instance is still unsafe at
#           that point, multi-line quoted strings containing operators,
#           and a backslash-newline continuation with the digest flag on
#           the continuation line (the tr'd `;` closes the instance
#           unsafe). Un-blanked note shapes (escaped quotes, $var,
#           backticks, literal newlines) keep the pre-#1152 deny
#           behavior — conservative, counted by the sidecar log.
#   mcp__ssh__ssh_execute — same command-string shape; reuses the Bash arm.
#
# Contract: reads the PreToolUse JSON on stdin; exit 0 = allow, exit 2 =
# blocking deny (stderr fed back to Claude). Fail-OPEN: any parse failure /
# unrecognized shape exits 0 (a broken guard must never brick every tool
# call; the prose rule backstops a miss). Every deny appends one best-effort
# line to a sidecar log (EPM_BANK_GUARD_LOG, default
# .claude/cache/bank-guard-denies.log) so the false-positive kill criterion
# is mechanically countable — the log NEVER affects the verdict.
# Escape hatch: EPM_ALLOW_BANK_READ=1 as session env, or as an inline prefix
# on a Bash command (documented in CLAUDE.md clause (d); deliberately NOT
# printed verbatim in the deny message).
set -u

STEM_RE='(advbench|strongreject|betley_main8|wang44|broad_em_train|sensitive_info_requests)'
BANK_RE="(^|/)query_banks/${STEM_RE}[^/]*\.json\$"
BARE_RE="(^|/)${STEM_RE}[^/]*\.json\$"
BANKDIR_RE='(^|/)query_banks/?$'
DENY_VERBS_RE='^(cat|tac|nl|head|tail|sed|awk|cut|sort|uniq|rev|strings|less|more|most|bat|od|xxd|hexdump|base64|fold|fmt|pr|column|paste|comm|join|json\.tool)$'
JQ_DIGEST_RE='^(keys|keys_unsorted|length|type|empty)$'
GUARD_LOG="${EPM_BANK_GUARD_LOG:-/home/thomasjiralerspong/explore-persona-space/.claude/cache/bank-guard-denies.log}"

# Session-env escape hatch (all tool arms).
_allow=$(printf '%s' "${EPM_ALLOW_BANK_READ:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in 1 | true | yes) exit 0 ;; esac

log_deny() {  # best-effort sidecar record for the plan-§7 FP kill criterion —
  # NEVER affects the exit path (a log failure must not change a verdict).
  { printf '%s\t%s\t%.200s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$GUARD_LOG"; } 2>/dev/null || true
}

deny() {  # $1 = what was attempted, $2 = the bank path
  log_deny "$1" "$2"
  echo "BLOCKED: $1 would page harmful-bank item text into context ($2). CLAUDE.md § Spurious usage-policy refusals (d): harmful BANK items are DIGEST-ONLY — reference by filename + index, never print item text (incident #866: four sessions refusal-killed). Allowed digest ops on the file directly: jq 'length' | jq 'keys' | jq 'type' | wc -l | sha256sum | ls | stat | grep -c (no pipes through cat). A sanctioned-maintenance override exists for deliberate bank regeneration work — see CLAUDE.md § Spurious usage-policy refusals clause (d)." >&2
  exit 2
}

is_bank_path() {
  local p="$1" rp
  p=${p#\'}; p=${p%\'}; p=${p#\"}; p=${p%\"}
  [ -n "$p" ] || return 1
  printf '%s' "$p" | grep -qE "$BANK_RE" && return 0
  if printf '%s' "$p" | grep -qE "$BARE_RE" && [ -f "$p" ]; then
    rp=$(realpath "$p" 2>/dev/null) || return 1
    printf '%s' "$rp" | grep -qE "$BANK_RE" && return 0
  fi
  return 1
}

is_bank_dir() {
  local p="$1" rp
  p=${p#\'}; p=${p%\'}; p=${p#\"}; p=${p%\"}
  [ -n "$p" ] || return 1
  printf '%s' "$p" | grep -qE "$BANKDIR_RE" && return 0
  if [ -d "$p" ]; then
    rp=$(realpath "$p" 2>/dev/null) || return 1
    printf '%s' "$rp" | grep -qE "$BANKDIR_RE" && return 0
  fi
  return 1
}

input=$(cat) || exit 0
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null) || exit 0

# ---- Read arm ------------------------------------------------------------
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || exit 0
if [ -n "$fp" ] && { [ -z "$tool" ] || [ "$tool" = Read ]; }; then
  is_bank_path "$fp" && deny "Read" "$fp"
  exit 0
fi

# ---- Grep arm (content mode only; bank FILE or the query_banks DIR) -------
if [ "$tool" = Grep ]; then
  gpath=$(printf '%s' "$input" | jq -r '.tool_input.path // empty' 2>/dev/null) || exit 0
  gmode=$(printf '%s' "$input" | jq -r '.tool_input.output_mode // "files_with_matches"' 2>/dev/null) || exit 0
  if [ "$gmode" = content ] && [ -n "$gpath" ]; then
    if is_bank_path "$gpath"; then
      deny "Grep (output_mode: content — matching lines ARE bank items; use output_mode files_with_matches/count, or jq 'length')" "$gpath"
    fi
    if is_bank_dir "$gpath"; then
      deny "Grep (output_mode: content on the query_banks directory — matching lines can be harmful-bank items; scope to a specific benign file, or use output_mode files_with_matches/count)" "$gpath"
    fi
  fi
  exit 0
fi

# ---- Bash arm (also mcp__ssh__ssh_execute: same command-string shape) -----
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null) || exit 0
[ -n "$cmd" ] || exit 0

# Inline escape hatch.
case "$cmd" in *EPM_ALLOW_BANK_READ=1*) exit 0 ;; esac

# (c) task.py --note quoted-prose exemption (#1152): a --note string on a
# task.py invocation is a DATA argument to a Python CLI, never executed —
# blank it before scanning so purely descriptive prose naming a bank path
# and/or a paging verb cannot false-positive (the #965 FP-watch shape).
# Blank single-quoted notes always; blank double-quoted notes ONLY when
# free of $ and backtick (a note carrying $(...) / `...` DOES execute —
# left visible to the walk, fail-closed). Line-based sed: a multi-line
# quoted note stays un-blanked (conservative — the old FP persists there,
# never a new allow). See the header for the *task.py* gate + scope notes.
# Gate = two in-shell case globs — zero added cost on the common path.
cmd_scan="$cmd"
taskpy_shape=0
case "$cmd" in *task.py*) taskpy_shape=1 ;; esac
if [ "$taskpy_shape" = 1 ]; then
  case "$cmd" in
    *--note*)
      cmd_scan=$(printf '%s' "$cmd" | sed -E \
        -e "s/--note(=|[[:space:]]+)'[^']*'/--note ''/g" \
        -e 's/--note(=|[[:space:]]+)"[^"$`]*"/--note ""/g')
      ;;
  esac
fi

# Fast path: no bank-stem-shaped token anywhere in the (note-scrubbed)
# command -> allow (hook runs on EVERY Bash call; this single grep is the
# common-case cost). A command whose ONLY bank token lived in note prose
# exits 0 right here.
printf '%s' "$cmd_scan" | grep -qE "${STEM_RE}[^/[:space:]]*\.json" || exit 0

# Normalize BEFORE the token walk: newlines map to ';' (a newline separates
# command units exactly like ';'), then shell control operators are padded
# so unspaced forms tokenize (`cat <bank>|grep foo`, `cat <bank>&&echo`,
# `cat <bank>;echo`, trailing `)`s) — otherwise the operator glues to the
# path token and the $-anchored .json match fails open. Backticks pad too
# (#1152): a backtick substitution glues to its verb + closing path token
# exactly as `$( )` would without the paren padding — the pre-#1152
# tokenizer failed OPEN on a bare backtick-cat of a bank. Padding inside
# quoted strings only affects tokenization, never execution (we never
# rewrite the command).
cmd_norm=$(printf '%s' "$cmd_scan" | tr '\n' ';' | sed 's/[|;&<>()`]/ & /g')

# Raw scan: word-split; find a bank token; if present, deny on any
# text-paging verb token, on bare diff outside a git instance, on
# grep-family line output (per-executable safe flags, per-instance
# tracking), on git show/diff/log -p paging (per-instance), on jq with a
# non-digest filter (per-instance), or on a task.py --file/--body-file
# bank argument. The DENY co-occurrence stays WHOLE-COMMAND (deliberately
# NO clause split — plan #965 §11.3); padded operator tokens `| & ; ( )`
# + backtick are INSTANCE BOUNDARIES only (close_instances): they stop safe-flag /
# attribution state from crossing command units, never detach the verb
# from the bank. Everything else (python/uv pipelines, cp, git
# add/commit, wc, sha256sum, stat, du, redirects) is allowed by default.
set -f
# shellcheck disable=SC2086
set -- $cmd_norm
set +f

bank=""
for tok in "$@"; do
  if is_bank_path "$tok"; then bank="$tok"; break; fi
done
[ -n "$bank" ] || exit 0

has_grep=0 grep_safe=0 grep_kind="" grep_any_unsafe=0
has_jq=0 jq_filter="" jq_seen=0
in_git=0 git_sub="" git_safe=0 git_log_patch=0
expect_taskfile=0

git_check() {  # evaluate the accumulated git instance; deny on unsafe paging.
  [ -n "$git_sub" ] || return 0
  [ "$git_safe" = 1 ] && return 0
  case "$git_sub" in
    show | diff) deny "git $git_sub paging a bank file (the patch text IS bank items; use git diff --stat / --name-only)" "$bank" ;;
    log) [ "$git_log_patch" = 1 ] && deny "git log -p paging a bank file (use git log --oneline -- <bank>)" "$bank" ;;
  esac
  return 0
}

jq_check() {  # evaluate the accumulated jq instance; deny on non-digest filter.
  [ "$has_jq" = 1 ] || return 0
  local f
  f=${jq_filter#\'}; f=${f%\'}; f=${f#\"}; f=${f%\"}
  printf '%s' "$f" | grep -qE "$JQ_DIGEST_RE" \
    || deny "jq '$f' on a bank file (item-access / non-digest filter; allowed: jq 'keys' | 'length' | 'type')" "$bank"
  return 0
}

close_instances() {  # at a command-unit boundary (`| & ; ( )` + backtick;
  # newlines arrive as `;` via tr): latch/evaluate each open instance, then
  # reset it —
  # a later unit's flags must never mark an earlier unit's instance safe.
  # grep: an instance still unsafe at its unit's end stays unsafe (sticky).
  if [ "$has_grep" = 1 ] && [ "$grep_safe" = 0 ]; then grep_any_unsafe=1; fi
  has_grep=0; grep_safe=0; grep_kind=""
  # git: attribution-based deny (show/diff/log deny only via in_git/git_sub),
  # so an UNRESOLVED instance (in_git=1, git_sub empty — e.g. the padded `(`
  # of `git -C $(pwd) show ...` fires a boundary BEFORE the subcommand token)
  # must CARRY across the boundary or the attribution detaches: the executing
  # `git show` would page the bank while the walk sees a bare orphan `show`
  # (fail-OPEN), and `git -C $(pwd) diff --stat` would false-deny via the
  # bare-diff branch. Carrying is fail-closed both ways: worst case a real
  # boundary after a subcommand-less git (e.g. `git add <bank> && diff x y`)
  # mis-attributes the next unit's show/diff/log to git -> extra DENY, never
  # extra allow. A RESOLVED instance (git_sub set) evaluates + resets here,
  # which is what fixes the `git log -- <bank> && mkdir -p /tmp/x` FP.
  if [ "$in_git" = 1 ] && [ -z "$git_sub" ]; then
    :  # carry the unresolved git instance across the boundary
  else
    git_check   # denies here if the closing unit paged a bank via git
    in_git=0; git_sub=""; git_safe=0; git_log_patch=0
  fi
  jq_check    # denies here on a non-digest jq filter
  has_jq=0; jq_filter=""; jq_seen=0
  expect_taskfile=0
}

for tok in "$@"; do
  # Command-unit boundary: close every open instance, consume the token.
  # Backticks bound command substitutions exactly as `( )` do (a distinct
  # command unit runs inside). `<`/`>` are deliberately NOT boundaries —
  # redirects don't start a new command unit, and flags legitimately
  # follow redirects.
  case "$tok" in
    '|' | '&' | ';' | '(' | ')' | '`') close_instances; continue ;;
  esac
  base="${tok##*/}"
  if printf '%s' "$base" | grep -qE "$DENY_VERBS_RE"; then
    deny "'$base' on a bank file" "$bank"
  fi
  # Bare `diff` OUTSIDE a git instance pages bank content (`diff /dev/null
  # <bank>` prints every item). `diff` cannot join DENY_VERBS_RE because
  # `git diff --stat` walks a `diff` token; the same-unit `git` precedes
  # it (in_git=1 there), so this fires only for standalone diff.
  if [ "$base" = diff ] && [ "$in_git" = 0 ]; then
    deny "'diff' paging a bank file (diff /dev/null <bank> prints item text; for tracked changes use git diff --stat / --name-only)" "$bank"
  fi
  case "$base" in
    grep | egrep | fgrep | rg)
      # PER-INSTANCE tracking: a prior grep instance still unsafe when a new
      # one starts stays unsafe — a later instance's -c cannot launder it.
      if [ "$has_grep" = 1 ] && [ "$grep_safe" = 0 ]; then grep_any_unsafe=1; fi
      has_grep=1; grep_safe=0
      case "$base" in rg) grep_kind=rg ;; *) grep_kind=grep ;; esac
      ;;
    jq)
      jq_check  # a prior jq instance with a non-digest filter denies here.
      has_jq=1; jq_filter=""; jq_seen=0
      ;;
    git)
      git_check  # a prior unsafe git-paging instance denies here.
      in_git=1; git_sub=""; git_safe=0; git_log_patch=0
      ;;
  esac
  if [ "$in_git" = 1 ] && [ -z "$git_sub" ]; then
    case "$base" in show | diff | log) git_sub="$base" ;; esac
  fi
  case "$tok" in
    --stat | --numstat | --shortstat | --name-only | --name-status | --check | --summary | --oneline) git_safe=1 ;;
    -p | -u | --patch) git_log_patch=1 ;;
  esac
  # grep-family safe flags, PER EXECUTABLE:
  #   grep/egrep/fgrep — count/quiet/file-list digests: -c/-q/-l/-L (+ long forms).
  #   rg — same EXCEPT -L, which is ripgrep --follow (prints matching lines!);
  #        long digest forms allowed. -C (context) must never match (case-aware).
  if [ "$has_grep" = 1 ]; then
    case "$tok" in
      --count | --count-matches | --quiet | --silent | --files-with-matches | --files-without-match) grep_safe=1 ;;
      *)
        if [ "$grep_kind" = rg ]; then
          printf '%s' "$tok" | grep -qE '^-[a-zA-Z]*[cql]' && grep_safe=1
        else
          printf '%s' "$tok" | grep -qE '^-[a-zA-Z]*[cqlL]' && grep_safe=1
        fi
        ;;
    esac
  fi
  # jq filter = first token after `jq` that is not a flag and not the file.
  if [ "$has_jq" = 1 ] && [ "$jq_seen" = 0 ] && [ "$base" != jq ]; then
    case "$tok" in
      -*) : ;;
      *)
        if ! is_bank_path "$tok"; then jq_filter="$tok"; jq_seen=1; fi
        ;;
    esac
  fi
  # task.py --file/--body-file rider (#1152): embedding a bank into task
  # state (events.jsonl note body / body.md / plans) defeats the
  # digest-only rule with delay. Gated on the task.py shape so legitimate
  # pipeline consumption (`scripts/eval.py --file <bank>`) stays allowed.
  # expect_taskfile resets at boundaries (close_instances) — `--file`
  # followed by an operator never attributes the NEXT unit's token.
  if [ "$taskpy_shape" = 1 ]; then
    if [ "$expect_taskfile" = 1 ]; then
      if is_bank_path "$tok"; then
        deny "task.py --file/--body-file on a bank file (would embed bank items into task state)" "$tok"
      fi
      expect_taskfile=0
    fi
    case "$tok" in
      --file | --body-file) expect_taskfile=1 ;;
      --file=* | --body-file=*)
        if is_bank_path "${tok#*=}"; then
          deny "task.py --file/--body-file on a bank file (would embed bank items into task state)" "${tok#*=}"
        fi
        ;;
    esac
  fi
done

if [ "$has_grep" = 1 ] && [ "$grep_safe" = 0 ]; then grep_any_unsafe=1; fi
if [ "$grep_any_unsafe" = 1 ]; then
  deny "grep-family line output on a bank file (matching lines ARE items; use grep -c / -q / -l, and place -c/-q/-l BEFORE the pattern in compound commands; note rg -L is --follow, NOT files-without-match)" "$bank"
fi
git_check
jq_check
exit 0
