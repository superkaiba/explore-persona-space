#!/usr/bin/env bash
# PreToolUse(Read) guard: workflow guard scripts are WINDOWED-READ-ONLY
# (.claude/rules/trigger-dense-review.md § Orchestrator ordinary turns,
# item 2 — #1563). Mechanizes the single-call READ channel only: a Read of
# a guard hook script with no `limit` (or `limit` over the cap) is denied;
# grep-anchored windowed reads (limit <= cap, any offset) pass untouched.
# The authored-text channel, cross-turn window accumulation, and the
# Bash/Grep/ssh channels stay rule-side (plan #1577 §4 scope decision).
#
# Deny set: (^|/)(scripts|.claude/hooks)/guard_* — the guard scripts
# THEMSELVES (scripts/guard_*.sh + the .claude/hooks/guard_* family, incl.
# the .py helper and this file; self-coverage is intentional — this hook
# parses stdin only and never Reads files, so no recursion). Every member
# exceeds the 120-line cap (min 210 lines), so none is silently exempt.
# Motivating incident (2026-07-19): a wholesale Read of
# scripts/guard_repo_root_branch.sh (128,531 B / 1,983 lines) paged ~41K
# tokens of trigger-dense text into orchestrator context.
#
# Contract: reads the PreToolUse JSON on stdin; exit 0 = allow, exit 2 =
# blocking deny (stderr fed back to Claude). FAIL-OPEN everywhere: any
# parse failure / missing jq / non-Read tool / empty path exits 0 — a
# broken guard must never brick the Read tool fleet-wide.
# Escape hatch: EPM_ALLOW_GUARD_READ=1 (session env; 1|true|yes).
# Cap: EPM_GUARD_READ_CAP_LINES (default 120 = the rule's "~120-line
# windows" bound; non-numeric/absent -> 120).
set -u

# Session-env escape hatch.
_allow=$(printf '%s' "${EPM_ALLOW_GUARD_READ:-}" | tr '[:upper:]' '[:lower:]')
case "$_allow" in 1 | true | yes) exit 0 ;; esac

command -v jq >/dev/null 2>&1 || exit 0 # fail-open: no jq, no verdict

input=$(cat) || exit 0
tool=$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null) || exit 0
[ "$tool" = "Read" ] || exit 0 # defensive; the settings.json matcher already scopes to Read

fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || exit 0
[ -n "$fp" ] || exit 0

# Cheap pre-filter: no subprocess spawn on the ordinary-Read hot path.
# NOTE: the pre-filter runs on the RAW path, so a symlink/alias path not
# containing 'guard_' skips the realpath arm entirely — accepted for the
# accidental-read threat model (deliberate evasion has the unguarded Bash
# channel anyway).
case "$fp" in *guard_*) ;; *) exit 0 ;; esac

# Deny-set ERE. Dir-anchored but repo-AGNOSTIC: scripts/guard_* /
# .claude/hooks/guard_* copies in OTHER repos (e.g. the
# claude-code-workflow mirror) match too — desirable (same trigger-dense
# family) and stated here so it is not a surprise. Worktree copies match
# (".../worktrees/issue-N/scripts/guard_x.sh" contains
# "/scripts/guard_x.sh"); tests/test_guard_*.py and src/**/guard_*.py do
# NOT (wrong directory).
DENY_RE='(^|/)(scripts|\.claude/hooks)/guard_[^/]+$'

# realpath-aware like the precedent guard; -m needs no existing file.
# realpath failure falls back to raw-only matching (fail-open direction).
resolved=$(realpath -m -- "$fp" 2>/dev/null) || resolved="$fp"
printf '%s' "$fp" | grep -qE "$DENY_RE" \
  || printf '%s' "$resolved" | grep -qE "$DENY_RE" \
  || exit 0

cap="${EPM_GUARD_READ_CAP_LINES:-120}"
case "$cap" in '' | *[!0-9]*) cap=120 ;; esac # junk/absent -> default

limit=$(printf '%s' "$input" | jq -r '.tool_input.limit // empty' 2>/dev/null) || exit 0
# Windowed read: a positive integer limit within the cap is the rule's
# sanctioned pattern — allow, any offset (the bound is the window SIZE,
# not its position). A present-but-non-numeric limit on an already-matched
# path is NOT the malformed-input fail-open case (the path match
# succeeded): deny — the override exists for deliberate full reads.
case "$limit" in
  '' | *[!0-9]*) ;; # absent or non-numeric -> fall through to deny
  *) [ "$limit" -ge 1 ] && [ "$limit" -le "$cap" ] && exit 0 ;;
esac

echo "BLOCKED: unbounded Read of $fp (${fp##*/} is a trigger-dense guard script; a wholesale read pages it into context — 2026-07-19 incident: ~41K tokens). Read it WINDOWED instead: Grep -n '<anchor>' $fp to locate the span, then Read with offset=<line> and limit<=$cap. Counts/structure only: wc -l, grep -c, git diff --stat. Rule: .claude/rules/trigger-dense-review.md § Orchestrator ordinary turns. Deliberate full read: EPM_ALLOW_GUARD_READ=1 (spawn-time session env — not settable mid-session; mid-session use windowed reads, or Write for a full-file rewrite)." >&2
exit 2
