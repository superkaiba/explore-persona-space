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
#   Bash  — whole-command raw scan (deliberately NO clause/pipe split, plan
#           §11.3): deny text-paging verbs (cat/head/sed/awk/json.tool/...)
#           co-occurring with a bank token, grep-family line output with
#           PER-EXECUTABLE safe flags (grep/egrep/fgrep: -c/-q/-l/-L digests;
#           rg: -c/-q/-l only — rg -L is --follow, it PRINTS lines) tracked
#           PER INSTANCE (a later safe grep cannot launder an earlier unsafe
#           one), git show/diff/log -p paging a bank (digest forms --stat /
#           --name-only / --numstat / --oneline allowed), and jq with a
#           non-digest filter (allowed single-word digests: keys /
#           keys_unsorted / length / type / empty). Everything else
#           (python/uv pipelines, cp, git add/commit, wc, sha256sum, stat,
#           du, redirects) is allowed by default — banks exist to be consumed
#           in-process by pipeline code.
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
DENY_VERBS_RE='^(cat|tac|nl|head|tail|sed|awk|cut|sort|uniq|rev|strings|less|more|most|bat|od|xxd|hexdump|base64|fold|fmt|pr|column|paste|json\.tool)$'
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

# Fast path: no bank-stem-shaped token anywhere -> allow (hook runs on EVERY
# Bash call; this single grep is the common-case cost).
printf '%s' "$cmd" | grep -qE "${STEM_RE}[^/[:space:]]*\.json" || exit 0

# Normalize shell control operators BEFORE the token walk so unspaced forms
# tokenize (`cat <bank>|grep foo`, `cat <bank>&&echo`, `cat <bank>;echo`,
# trailing `)`s) — otherwise the operator glues to the path token and the
# $-anchored .json match fails open. Padding inside quoted strings only
# affects tokenization, never execution (we never rewrite the command).
cmd_norm=$(printf '%s' "$cmd" | sed 's/[|;&<>()]/ & /g')

# Whole-command raw scan (deliberately NO clause split — plan §11.3):
# word-split; find a bank token; if present, deny on any text-paging verb
# token, on grep-family line output (per-executable safe flags, per-instance
# tracking), on git show/diff/log -p paging (per-instance), or on jq with a
# non-digest filter (per-instance). Everything else (python/uv pipelines,
# cp, git add/commit, wc, sha256sum, stat, du, redirects) is allowed by
# default.
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

for tok in "$@"; do
  base="${tok##*/}"
  if printf '%s' "$base" | grep -qE "$DENY_VERBS_RE"; then
    deny "'$base' on a bank file" "$bank"
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
done

if [ "$has_grep" = 1 ] && [ "$grep_safe" = 0 ]; then grep_any_unsafe=1; fi
if [ "$grep_any_unsafe" = 1 ]; then
  deny "grep-family line output on a bank file (matching lines ARE items; use grep -c / -q / -l; note rg -L is --follow, NOT files-without-match)" "$bank"
fi
git_check
jq_check
exit 0
