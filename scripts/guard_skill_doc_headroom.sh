#!/usr/bin/env bash
# PostToolUse(Edit|Write) ADVISORY hook: surface skill-doc size-ratchet
# headroom AT EDIT TIME (task #2325).
#
# The skill-doc size ratchet (workflow_lint.py::check_skill_doc_size /
# SKILL_DOC_SIZE_GRANDFATHER) FAILs the no-flags lint — the Step 9c gate —
# when a grandfathered skill doc regrows past its cap. There is no skill-side
# pre-commit hook (the agent-spec sibling has one), so without this hook the
# tripping session discovers the red only at its own Step 9c gate, a
# ~13-30 min round after a full implementation pass. This hook runs the SAME
# lint flag right after any Edit/Write to a `.claude/skills/**/*.md` file and
# relays the lint's own FAIL / low-headroom line to the editing session, so
# the trip-wire is hit in seconds at edit time, with the remedy attached
# (the regrowth FAIL message emits the exact replacement cap line).
#
# WARN-ONLY BY MECHANISM: PostToolUse fires AFTER the tool ran — nothing can
# be blocked. Exit 2 + stderr feeds the text back to Claude as advisory
# feedback; every other path exits 0 silently. FAIL-OPEN on every
# hook-internal error (bad JSON, missing jq, lint timeout, unparseable
# output): an advisory hook must never wedge editing — the Step 9c gate +
# the remedy-bearing FAIL message remain the backstop.
#
# Trip conditions (either one, scoped to the EDITED file's rel path):
#   - the lint emits a FAIL line naming the edited file, or
#   - the edited file's WARN line reports headroom ("N bytes under its cap")
#     below EPM_SKILL_DOC_HEADROOM_WARN_BYTES (default 2000; junk values
#     fall back to the default — the guard_trigger_dense_read cap-env
#     convention).
#
# Kill switch: EPM_SKIP_SKILL_DOC_HEADROOM_HOOK=1.
#
# Venv/tree split: the lint runs from the MAIN checkout's root (uv resolves
# the main .venv — a worktree `uv run` would build a fresh venv, #634 class)
# but executes the EDITED TREE's scripts/workflow_lint.py, so a worktree's
# own caps + sizes are what get measured (workflow_lint._REPO_ROOT resolves
# from __file__).
#
# NAMED RESIDUALS (accepted; plan #2325 §3 Edit 4): fires on Edit/Write TOOL
# calls only — sibling growth arriving via a merge and scripted Bash splices
# (sed/python writes to SKILL.md) are covered by the Step 10d re-measure duty
# + the remedy-bearing gate message, not by this hook. Pickup is
# session-restart-gated (settings.json is read at session start).
set +e
set -u

# Kill switch first: silent allow.
if [ "${EPM_SKIP_SKILL_DOC_HEADROOM_HOOK:-}" = "1" ]; then
  exit 0
fi

command -v jq >/dev/null 2>&1 || exit 0

file_path=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)
[ -n "$file_path" ] || exit 0

case "$file_path" in
  */.claude/skills/*.md) ;;
  *) exit 0 ;;
esac

# Tree that owns the edited file; fast-exit when it carries no lint copy
# (e.g. a skills file inside a non-repo scratch tree).
tree_root="${file_path%%/.claude/skills/*}"
lint="$tree_root/scripts/workflow_lint.py"
[ -f "$lint" ] || exit 0

rel="${file_path#*/.claude/skills/}"

# Main checkout root (worktree-safe): venv host for `uv run`. Fall back to
# this script's own tree when git is unavailable. Capture the probe output
# FIRST and form `dirname` only on success: on a failed probe git prints
# nothing and `dirname ""` is "." — neither empty nor a non-existent dir —
# which made the fallback unreachable and ran uv from the caller's cwd
# (#2325 r2 blocker 1; the #634 wrong-venv shape).
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" || exit 0
common_dir="$(git -C "$SELF_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
if [ -n "$common_dir" ]; then
  main_root="$(dirname "$common_dir")"
else
  main_root="${SELF_DIR%/scripts}"
fi
# Validate the selected root before invoking uv (fail-open): it must be an
# absolute project checkout — retry the script-owned tree, else exit 0.
case "$main_root" in /*) ;; *) main_root="${SELF_DIR%/scripts}" ;; esac
if [ ! -d "$main_root" ] || [ ! -f "$main_root/pyproject.toml" ]; then
  main_root="${SELF_DIR%/scripts}"
fi
{ [ -d "$main_root" ] && [ -f "$main_root/pyproject.toml" ]; } || exit 0

out=$(cd "$main_root" && timeout 30s uv run python "$lint" --check-skill-doc-size 2>&1)
lint_rc=$?
# Fail-open unless the lint run COMPLETED (#2325 r2 blocker 2): the size
# check streams WARN lines incrementally, so a timeout-killed (124/137) or
# tool-missing (126/127) run can leave a parseable partial line in $out.
# rc 0 must carry the PASS summary; rc 1 must carry a complete FAIL summary
# (`workflow_lint: FAIL (N error(s))`); every other status exits 0.
# Both sentinels are WHOLE-LINE anchored (-x; #2325 r3): the emitters write
# the invariant full lines, so a truncated `workflow_lint: FAIL (1 error`
# fragment — exactly the incomplete-run shape this gate rejects — or a
# substring hit (e.g. `... PASSING`) must not satisfy the completion check.
case "$lint_rc" in
  0) printf '%s\n' "$out" | grep -qxF "workflow_lint: PASS" || exit 0 ;;
  1) printf '%s\n' "$out" | grep -qxE 'workflow_lint: FAIL \([0-9]+ error\(s\)\)' || exit 0 ;;
  *) exit 0 ;;
esac

thr="${EPM_SKILL_DOC_HEADROOM_WARN_BYTES:-2000}"
case "$thr" in '' | *[!0-9]*) thr=2000 ;; esac

# FAIL lines print as `workflow_lint: <error text>`; the summary lines
# (`workflow_lint: PASS` / `workflow_lint: FAIL (N error(s))`) never contain
# a rel path, so a rel-scoped grep cannot match them. Pre-existing red on
# OTHER files never trips this hook.
fail_line=$(printf '%s\n' "$out" | grep -F "workflow_lint: " | grep -F "$rel" | head -n 1)

# WARN line shape: `WARN: .claude/skills/<rel>: <size> bytes — grandfathered;
# <H> bytes under its cap (<cap>).`
warn_line=$(printf '%s\n' "$out" | grep -F "WARN: .claude/skills/$rel:" \
  | grep -F "under its cap" | head -n 1)
headroom=$(printf '%s\n' "$warn_line" \
  | sed -n 's/.*grandfathered; \([0-9][0-9]*\) bytes under its cap.*/\1/p')

if [ -n "$fail_line" ]; then
  {
    printf '%s\n' "$fail_line"
    printf 'skill-doc size ratchet: this edit puts %s OVER its grandfather cap — apply the remedy in the line above in the SAME change, or the branch'\''s own Step 9c gate goes red.\n' "$rel"
  } >&2
  exit 2
fi

if [ -n "$headroom" ] && [ "$headroom" -lt "$thr" ] 2>/dev/null; then
  {
    printf '%s\n' "$warn_line"
    printf 'skill-doc size ratchet: this edit leaves %s within %s B of its grandfather cap — raise the cap in the SAME change (see line above), or the branch'\''s own Step 9c gate goes red.\n' "$rel" "$headroom"
  } >&2
  exit 2
fi

exit 0
