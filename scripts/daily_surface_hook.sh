#!/usr/bin/env bash
# daily_surface_hook.sh — SessionStart hook for explore-persona-space.
#
# Surfaces the latest `/daily` "## Proposed workflow improvements" section
# ONCE per new daily file, on the first EPS Claude session that sees it.
# Emits a SessionStart `additionalContext` JSON payload instructing Claude
# to present the proposals to Thomas and ask which to apply (greenlight
# flow in .claude/skills/daily/SKILL.md). Stays silent (exit 0, no stdout)
# whenever there is nothing new worth surfacing, so it never nags.
#
# Idempotency is keyed on the daily FILE name (not the calendar day): the
# marker stores the last-surfaced file path, so multiple sessions on the
# same day surface at most once, and a stale daily (cron failed overnight)
# is never re-surfaced. A new daily file written by the 23:27 cron resets
# it for the next morning's first session.
#
# Paths are env-overridable for testing; defaults are the production paths.
set -euo pipefail

REPO="${EPS_REPO:-/home/thomasjiralerspong/explore-persona-space}"
DAILY_DIR="${EPS_DAILY_DIR:-$REPO/logs/daily}"
MARKER="${EPS_DAILY_MARKER:-$REPO/.claude/cache/.daily-last-surfaced}"

# No daily logs yet -> nothing to do.
[ -d "$DAILY_DIR" ] || exit 0

# Latest daily by name; YYYY-MM-DD.md sorts chronologically.
latest=$(ls -1 "$DAILY_DIR"/[0-9]*.md 2>/dev/null | sort | tail -1 || true)
[ -n "$latest" ] || exit 0

# Already surfaced this exact file -> stay silent.
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$latest" ]; then
  exit 0
fi

# Pull the "## Proposed workflow improvements" section (up to the next H2).
proposals=$(awk '
  /^## Proposed workflow improvements[[:space:]]*$/ { grab=1; next }
  /^## / { if (grab) exit }
  grab { print }
' "$latest")

# Trim leading/trailing blank lines.
proposals=$(printf '%s\n' "$proposals" | awk 'NF {p=1} p' | tac | awk 'NF {p=1} p' | tac)

mkdir -p "$(dirname "$MARKER")"

# Empty section, missing section, or a "nothing to surface" placeholder
# (old: "no friction patterns / patterns met"; new: "no workflow-fixable
# problems") -> mark this file as seen (so we don't re-scan it every session)
# and exit without surfacing anything.
if [ -z "$proposals" ] || printf '%s' "$proposals" | grep -qiE 'no (friction patterns|patterns met|workflow-fixable problems)'; then
  printf '%s' "$latest" > "$MARKER"
  exit 0
fi

# Record this file as surfaced before emitting (so a crash mid-emit still
# won't double-surface).
printf '%s' "$latest" > "$MARKER"

base=$(basename "$latest" .md)

ctx="FIRST EPS SESSION TODAY — surface the daily-proposed workflow improvements before doing anything else.

The /daily background run (23:27 PT cron) wrote proposed workflow improvements to logs/daily/${base}.md. Lead your FIRST reply this session by presenting the numbered proposals below to Thomas (keep it tight — title + one-line why per item, the diffs are in the file if he wants them), then ask which he wants applied. He replies 'do 1,3' / 'do all' / 'skip', or wants to discuss one. On approval, spawn the workflow-improver agent per the /daily skill greenlight flow: it edits the target file(s), runs scripts/workflow_lint.py on what it touched, and commits 'workflow: apply daily-proposed edits <ids> (${base})'. Declined proposals stay in the file as historical record — do not delete them. If Thomas opened this session for other work, surface these first, take his approve/skip, then proceed to his actual task.

--- Proposed workflow improvements (${base}) ---
${proposals}
--- end ---"

jq -n --arg ctx "$ctx" \
  '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $ctx}}'
exit 0
