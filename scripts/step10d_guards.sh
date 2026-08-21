#!/usr/bin/env bash
# Step 10d guards, staged first tranche (task #1978).
#
# Extracts three per-session-retyped shell fragments from
# ``.claude/skills/issue/SKILL.md`` § Merge safety guards into ONE tested,
# checked-in script so sessions invoke it instead of transcribing prose:
#
#   * ``--guard prelude`` : the ``REPO_ROOT`` + ``WT`` derivation preamble
#                           (the fragment carrying the ``--path-format=absolute``
#                           token whose hand-typed variant produced the
#                           incident-triggering typo).
#   * ``--guard 0``       : the Guard 0 agent-memory pre-commit predicate
#                           (dirty-tree fix for the merge-conflict recovery,
#                           #906).
#   * ``--guard 4``       : the Guard 4 LOST-UPDATE refusal predicate
#                           (silent whole-file-snapshot revert, #1701 -> #1713;
#                           ``--main-sha`` is the pinned origin/main TIP from
#                           which the merge-base is DERIVED, #2428).
#
# All three fragments are byte-equivalent in effect to the current SKILL.md
# fences (transcribed at implementation time; the ``.py`` pin tests exercise
# every behavioral arm to catch drift).
#
# Fourth subcommand (task #2201, not a SKILL.md extraction -- a NEW probe):
#
#   * ``--guard divergence`` : the #1771->#2201 deliverable-divergence probe.
#                           Lists round-deliverable files ALSO changed on the
#                           remote main since the merge-base, refined by the
#                           sync-subject exclusion (#1560/#1789 anchor), the
#                           content-identical drop, and the ``tasks/`` +
#                           ``.claude/agent-memory/`` carve-outs. TWO call
#                           sites (both use the two-step rc-capture caller
#                           form + ``rm -f`` the ``--out`` path first):
#                           the Step 5a post-sync probe
#                           (``.claude/skills/issue/steps/09-step-5.md`` --
#                           reviewer-brief disclosure, every review round)
#                           and the Step 10d pre-merge divergence delta gate
#                           (``.claude/skills/issue/steps/18-step-10d.md``).
#                           MAIN_SHA is captured ONCE (rev-parse) BEFORE the
#                           merge-base and every later read uses the capture
#                           (Guard 1's pin precedent, steps/18-step-10d.md
#                           :293-313: "pinned to ONE SHA so a concurrent
#                           session's fetch cannot advance origin/main
#                           mid-guard"). NO fetch inside the helper -- both
#                           call sites already run the bounded
#                           ``timeout --kill-after=30s 120s git fetch origin
#                           main --quiet || true`` first.
#
# Guards 1, 2, 3, 5 stay as SKILL.md prose this round -- state-coupling and
# recovery-machinery coupling make their extraction the parked follow-up
# (plan §5.5). See ``tasks/*/1978/plans/plan.md`` for the staging rationale.
#
# ==== State-passing contract (the point of the extraction) ====
#
# Guards must expose the caller-side variables the downstream SKILL.md blocks
# read (``MEM_COMMITTED``, ``LOST_UPDATE_PATHS``, etc.). Approach: each guard
# subcommand emits ``KEY=VALUE`` shell-assignment lines on **stdout**, one per
# line, values shell-safe: word/sha/``yes|no`` tokens print via ``%s``;
# path-bearing values (``LOST_UPDATE_PATHS``, ``DIVERGED_FILE``) print via
# ``printf %q`` so ``eval`` recovers them exactly (no user text on stdout;
# diagnostics go to stderr). Callers consume via::
#
#     eval "$(bash scripts/step10d_guards.sh <N> --guard 0)"
#
# For Guard 4 the caller wraps the invocation in a two-step rc-capture form so
# the refusal's ``false``-in-block-tail semantics survive ``eval``::
#
#     GUARD4_OUT=$(bash scripts/step10d_guards.sh <N> --guard 4 --main-sha "$MAIN_SHA"); GUARD4_RC=$?
#     eval "$GUARD4_OUT"
#     [ "$GUARD4_RC" -eq 1 ] && false
#
# Exit codes: 0 on success (pass / skipped / no-op), 1 on refusal (Guard 4
# only), 2 on infra error (worktree missing, git failure). Infra errors emit
# ``ERROR=<reason>`` on stdout so ``eval`` still populates a variable for the
# caller to inspect.
#
# NOT ``set -euo pipefail`` at top level: each guard manages its own error
# state and must REPORT (not die silently) on the caller-visible surface.

_usage() {
    cat <<'USAGE' >&2
Usage: bash scripts/step10d_guards.sh <issue-number> --guard {prelude|0|4|divergence} [--main-sha <sha>] [--out <file>]

Emits KEY=VALUE lines on stdout for `eval` in the caller. Diagnostics on stderr.

  --guard prelude        Derives REPO_ROOT + WT; emits both. Exit 0.
  --guard 0              Runs Guard 0 (agent-memory pre-commit); emits
                         MEM_COMMITTED=yes|no. Exit 0 on success, 2 on infra
                         error (ERROR=<reason>).
  --guard 4              Runs Guard 4 (LOST-UPDATE refusal); emits
                         GUARD4=pass|refused|skipped, GUARD4_MERGE_BASE (the
                         merge-base derived from the pinned tip; emitted on
                         both pass and refusal), and LOST_UPDATE_PATHS on
                         refusal. Exit 0 on pass/skipped, 1 on refused, 2 on
                         infra error. Honors EPM_SKIP_LOST_UPDATE_GUARD=1
                         (checked FIRST).
  --guard divergence     Deliverable-divergence probe (#1771->#2201): emits
                         DIVERGENCE=clean|diverged|skipped, DIVERGED_COUNT,
                         MAIN_SHA (the remote-main tip pinned by ONE
                         rev-parse BEFORE the merge-base -- no fetch inside
                         the helper; both call sites run the bounded fetch
                         first), and DIVERGED_FILE (path of the
                         one-path-per-line survivors list; empty when
                         skipped). Exit 0 always (a probe, never a refusal);
                         2 + ERROR=<reason> on infra error. Callers (Step 5a
                         post-sync probe; Step 10d pre-merge delta gate) use
                         the two-step rc-capture form and rm -f the --out
                         path BEFORE invoking (stale-output hygiene).
  --main-sha <sha>       Guard 4 only: the pinned origin/main TIP (the Step
                         10d caller's Guard-1 capture) -- NOT the merge-base.
                         The helper derives the merge-base from it
                         (git merge-base HEAD <sha>); defaults to the live
                         origin/main ref when omitted. Rejected (exit 2,
                         ERROR=main-sha-not-supported-for-guard-*) for every
                         other guard (#2428).
  --out <file>           divergence only: survivors list path (default
                         /tmp/issue-<N>-divergence.txt).
USAGE
    exit 2
}

_die_usage() {
    printf 'ERROR=%s\n' "$1"
    _usage
}

if [ "$#" -lt 3 ]; then
    _die_usage bad-usage
fi

ISSUE="$1"
shift
case "$ISSUE" in
    ''|*[!0-9]*) _die_usage bad-issue ;;
esac
N="$ISSUE"

GUARD=""
MAIN_SHA=""
OUT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --guard)
            [ "$#" -ge 2 ] || _die_usage missing-guard-value
            GUARD="$2"
            shift 2
            ;;
        --main-sha)
            [ "$#" -ge 2 ] || _die_usage missing-main-sha-value
            MAIN_SHA="$2"
            shift 2
            ;;
        --out)
            [ "$#" -ge 2 ] || _die_usage missing-out-value
            OUT="$2"
            shift 2
            ;;
        *) _die_usage "unknown-arg-$1" ;;
    esac
done

if [ -z "$GUARD" ]; then
    _die_usage missing-guard
fi

# --main-sha is Guard 4-only: divergence pins its OWN main snapshot and would
# silently shadow a passed value; prelude/0 never read it. Reject rather than
# accept an inert flag (#2428 acceptance criterion 4).
if [ -n "$MAIN_SHA" ] && [ "$GUARD" != "4" ]; then
    _die_usage "main-sha-not-supported-for-guard-$GUARD"
fi

# --- PRELUDE derivation (used by every guard subcommand). ------------------
#
# Byte-equivalent to the SKILL.md fence (§ Merge safety guards prelude):
#
#     REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
#     WT="$REPO_ROOT/.claude/worktrees/issue-<N>"
#
# ``--path-format=absolute`` is REQUIRED (not ``--path-format: absolutre`` --
# the incident-triggering typo, #1867). Deriving from ``--git-common-dir``
# stays worktree-safe: from a worktree cwd ``rev-parse --show-toplevel``
# returns the WORKTREE root, which would nest ``$WT`` into
# ``.../issue-<N>/.claude/worktrees/issue-<N>`` (the #506 incident).
_derive_repo_root() {
    local common_dir
    if ! common_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null); then
        return 1
    fi
    dirname "$common_dir"
}

case "$GUARD" in
    prelude)
        if REPO_ROOT=$(_derive_repo_root); then
            printf 'REPO_ROOT=%s\n' "$REPO_ROOT"
            printf 'WT=%s\n' "$REPO_ROOT/.claude/worktrees/issue-$N"
            exit 0
        fi
        printf 'ERROR=%s\n' derivation-failed
        exit 2
        ;;

    0)
        # Guard 0 -- agent-memory pre-commit (#906).
        #
        # Byte-equivalent to the SKILL.md Guard 0 fence body: probe
        # ``.claude/agent-memory/**`` for dirt, commit by explicit pathspec,
        # best-effort push, emit MEM_COMMITTED. Idempotent: a re-run finds
        # the pathspec clean and skips.
        REPO_ROOT=$(_derive_repo_root) || { printf 'ERROR=%s\n' derivation-failed; exit 2; }
        WT="$REPO_ROOT/.claude/worktrees/issue-$N"
        if [ ! -d "$WT" ]; then
            printf 'ERROR=%s\n' worktree-missing
            exit 2
        fi
        status_out=$(git -C "$WT" status --porcelain -- .claude/agent-memory/ 2>&1) || {
            printf 'ERROR=%s\n' status-failed
            printf 'status stderr: %s\n' "$status_out" >&2
            exit 2
        }
        if [ -z "$status_out" ]; then
            printf 'MEM_COMMITTED=%s\n' no
            exit 0
        fi
        if ! git -C "$WT" add -- .claude/agent-memory/ >/dev/null 2>&1; then
            printf 'ERROR=%s\n' add-failed
            exit 2
        fi
        if ! git -C "$WT" commit \
                -m "issue-$N: persist agent-memory writes before Step-10d merge" \
                -- .claude/agent-memory/ >/dev/null 2>&1; then
            printf 'ERROR=%s\n' commit-failed
            exit 2
        fi
        printf 'MEM_COMMITTED=%s\n' yes
        # Best-effort branch push NOW: the fast-path / artifact-confirmed forms
        # never reach the safe-case pre-merge push, and on re-entry the pathspec
        # is clean (MEM_COMMITTED=no) so the commit would otherwise strand
        # local-only indefinitely. Failure is non-fatal -- the safe-case push
        # condition in SKILL.md is the second chance.
        git -C "$WT" push origin "issue-$N" >/dev/null 2>&1 || true
        exit 0
        ;;

    4)
        # Guard 4 -- LOST-UPDATE refusal (#1701 -> #1713).
        #
        # Refuses the merge when the branch's whole-file snapshot silently
        # DROPS lines that landed on ``origin/main`` after the branch's
        # merge-base -- no conflict, no warning. Kill switch checked FIRST.
        # ``--main-sha`` is the pinned origin/main TIP; the merge-base is
        # DERIVED from it below (#2428).
        if [ -n "${EPM_SKIP_LOST_UPDATE_GUARD:-}" ]; then
            printf 'GUARD4=%s\n' skipped
            exit 0
        fi
        REPO_ROOT=$(_derive_repo_root) || { printf 'ERROR=%s\n' derivation-failed; exit 2; }
        WT="$REPO_ROOT/.claude/worktrees/issue-$N"
        if [ ! -d "$WT" ]; then
            printf 'ERROR=%s\n' worktree-missing
            exit 2
        fi
        # --main-sha is the PINNED origin/main TIP (the Step 10d caller's Guard-1
        # capture: MAIN_SHA=$(git -C "$WT" rev-parse origin/main),
        # steps/18-step-10d.md:304) -- NOT the merge-base. Derive the merge-base
        # FROM it and use the SAME pinned tip as the main side of every diff
        # below, so the pinned path and the no-flag path agree (#2428).
        MAIN_REF="${MAIN_SHA:-origin/main}"
        if ! MB=$(git -C "$WT" merge-base HEAD "$MAIN_REF" 2>/dev/null) || [ -z "$MB" ]; then
            printf 'ERROR=%s\n' merge-base-failed
            exit 2
        fi
        # Per-invocation temp file (byte-equivalent-in-effect to the fence's
        # ``/tmp/1713-main-adds.txt`` shared name; using ``mktemp`` avoids
        # cross-invocation collision under concurrent sessions, which the
        # fence's shared name would allow).
        _tmp_adds=$(mktemp -t step10d-guards-1713-main-adds.XXXXXXXX) || {
            printf 'ERROR=%s\n' mktemp-failed
            exit 2
        }
        # shellcheck disable=SC2064
        trap "rm -f '$_tmp_adds'" EXIT
        LOST_UPDATE_PATHS=""
        # Enumerate branch-touched paths (three-dot merge-base..HEAD) and
        # apply the SKILL.md fence's ACTUAL case glob (``.claude/skills/*``
        # -- broader than the prose's ``.claude/skills/**/SKILL.md``
        # paraphrase; the extraction preserves the fence, not the prose).
        while IFS= read -r P; do
            case "$P" in
                scripts/workflow_lint.py|.claude/skills/*|.claude/rules/*|.claude/workflow.yaml|CLAUDE.md)
                    MAIN_ADDS=$(git -C "$WT" diff --numstat "$MB" "$MAIN_REF" -- "$P" 2>/dev/null \
                        | awk '{print $1+0}')
                    if [ "${MAIN_ADDS:-0}" -gt 0 ]; then
                        if ! git -C "$WT" diff "$MB" "$MAIN_REF" -- "$P" 2>/dev/null \
                                | grep -E '^\+[^+]' | sed 's/^\+//' > "$_tmp_adds"; then
                            # A grep failure with an empty adds file means "no
                            # matching '+' lines"; treat as MAIN_ADDS=0 rather
                            # than an error. But if the diff itself failed and
                            # left junk, fail loud.
                            : >"$_tmp_adds"
                        fi
                        MISSING_ON_BRANCH=0
                        while IFS= read -r ADD_LINE; do
                            [ -z "$ADD_LINE" ] && continue
                            if ! git -C "$WT" show "HEAD:$P" 2>/dev/null \
                                    | grep -Fxq -- "$ADD_LINE"; then
                                MISSING_ON_BRANCH=$((MISSING_ON_BRANCH + 1))
                            fi
                        done < "$_tmp_adds"
                        if [ "$MISSING_ON_BRANCH" -gt 0 ]; then
                            LOST_UPDATE_PATHS="$LOST_UPDATE_PATHS $P(${MISSING_ON_BRANCH})"
                        fi
                    fi
                    ;;
            esac
        done < <(git -C "$WT" diff --name-only "$MB"...HEAD 2>/dev/null)
        # Trim leading whitespace from LOST_UPDATE_PATHS for a clean key=value.
        LOST_UPDATE_PATHS="${LOST_UPDATE_PATHS# }"
        if [ -n "$LOST_UPDATE_PATHS" ]; then
            # Refusal banner: byte-equivalent to the SKILL.md fence's stderr
            # echo. The fence also emits a "Recovery: ..." line on stdout;
            # to keep our stdout as clean KEY=VALUE for ``eval``, we mirror
            # that recovery hint to stderr too (small, contained deviation
            # documented in the plan's §5.1 refusal-semantics note; the
            # human-readable content is unchanged).
            printf 'LOST-UPDATE REFUSAL (Guard 4, #1713): branch carries a whole-file snapshot dropping main-side additions on: %s\n' \
                "$LOST_UPDATE_PATHS" >&2
            printf 'Recovery: rebase onto origin/main and re-apply the intended edits by explicit path; post epm:merge-failed v1 (reason: lost-update, paths=%s).\n' \
                "$LOST_UPDATE_PATHS" >&2
            printf 'GUARD4=%s\n' refused
            printf 'GUARD4_MERGE_BASE=%s\n' "$MB"
            # %q: eval-safe under the two-step caller form -- the value
            # carries ``path(count)`` entries whose bare parens are a bash
            # syntax error inside ``eval "$GUARD4_OUT"`` (and paths may bear
            # spaces/metachars). Same encoding as DIVERGED_FILE below; the
            # token/sha keys stay %s like the divergence guard's.
            printf 'LOST_UPDATE_PATHS=%q\n' "$LOST_UPDATE_PATHS"
            exit 1
        fi
        printf 'GUARD4=%s\n' pass
        printf 'GUARD4_MERGE_BASE=%s\n' "$MB"
        exit 0
        ;;

    divergence)
        # Deliverable-divergence probe (#1771 -> #2201).
        #
        # Emits the refined branch-AND-main changed-file set since the
        # merge-base: files this round deliberately edited that ALSO moved on
        # the remote main -- the #1771 shape (#2164 rewrote the same function
        # + test on main in the opposite direction; round-1 review never saw
        # it). A PROBE, never a refusal: exit 0 on clean/diverged/skipped,
        # 2 + ERROR= on infra error (matching guards 0/4).
        REPO_ROOT=$(_derive_repo_root) || { printf 'ERROR=%s\n' derivation-failed; exit 2; }
        WT="$REPO_ROOT/.claude/worktrees/issue-$N"
        # No issue worktree -> evaluate the CURRENT checkout (a main-checkout
        # session then self-skips below -- the Step 5a on-main skip).
        if [ ! -d "$WT" ]; then
            WT="$REPO_ROOT"
        fi
        if [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD 2>/dev/null)" = "main" ]; then
            printf 'DIVERGENCE=%s\n' skipped
            printf 'DIVERGED_COUNT=%s\n' 0
            printf 'DIVERGED_FILE=\n'
            exit 0
        fi
        # Pin ONE main snapshot FIRST (MF-A) -- Guard 1's capture precedent
        # (steps/18-step-10d.md:293-313): rev-parse ONCE so a concurrent
        # session's fetch cannot advance origin/main mid-probe; every later
        # read below uses ONLY the captured sha. No fetch here -- both call
        # sites already ran the bounded fetch.
        # --verify --quiet: an unresolvable name yields EMPTY + rc 1 (a bare
        # rev-parse would echo the literal name to stdout and fake a sha).
        MAIN_SHA=$(git -C "$WT" rev-parse --verify --quiet origin/main 2>/dev/null)
        if [ -z "$MAIN_SHA" ]; then
            printf 'ERROR=%s\n' no-origin-main
            exit 2
        fi
        if ! MB=$(git -C "$WT" merge-base HEAD "$MAIN_SHA" 2>/dev/null) || [ -z "$MB" ]; then
            printf 'ERROR=%s\n' no-merge-base
            exit 2
        fi
        OUT="${OUT:-/tmp/issue-$N-divergence.txt}"
        _tmp_div=$(mktemp -d -t step10d-guards-2201-divergence.XXXXXXXX) || {
            printf 'ERROR=%s\n' mktemp-failed
            exit 2
        }
        # shellcheck disable=SC2064
        trap "rm -rf '$_tmp_div'" EXIT
        # Branch side + main side name lists (each vs the merge-base), then
        # the intersection. Fail LOUD on a diff failure -- an empty list from
        # a failed diff would silently read as clean.
        if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MB" HEAD \
                > "$_tmp_div/branch.raw" 2>/dev/null; then
            printf 'ERROR=%s\n' branch-diff-failed
            exit 2
        fi
        if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MB" "$MAIN_SHA" \
                > "$_tmp_div/mainside.raw" 2>/dev/null; then
            printf 'ERROR=%s\n' main-diff-failed
            exit 2
        fi
        sort -u "$_tmp_div/branch.raw" > "$_tmp_div/branch.txt"
        sort -u "$_tmp_div/mainside.raw" > "$_tmp_div/mainside.txt"
        comm -12 "$_tmp_div/branch.txt" "$_tmp_div/mainside.txt" > "$_tmp_div/isect.txt"
        if ! : > "$OUT"; then
            printf 'ERROR=%s\n' out-unwritable
            exit 2
        fi
        DIV_COUNT=0
        while IFS= read -r P; do
            [ -z "$P" ] && continue
            # Carve-outs: task-state + review-round bookkeeping (Guard 3
            # precedent -- "always in-scope, never an UNSAFE trigger";
            # tasks/ churn has its own machinery + Guard 1).
            case "$P" in
                tasks/*|.claude/agent-memory/*) continue ;;
            esac
            # Sync-import-only drop: a path whose branch-side commits ALL
            # carry the spec-freshness sync subject anchor is not a
            # deliberate branch edit (#1560/#1789 subject-scoped exclusion,
            # awk idiom from steps/09-step-5.md:234-235). Materialized with
            # an rc check (review r1 MF-1a): a failed log piped straight
            # into awk exits through awk rc 0, reads as "sync-only"
            # (empty), and silently DROPS a real divergence. No stderr
            # redirect -- diagnostics stay on stderr (guards 0/4 contract).
            if ! git -C "$WT" log --format='%H %s' "$MB"..HEAD -- "$P" \
                    > "$_tmp_div/plog.txt"; then
                printf 'ERROR=%s\n' log-failed
                exit 2
            fi
            NONSYNC=$(awk 'index($0, "sync workflow-surface specs from") == 0' \
                "$_tmp_div/plog.txt")
            [ -z "$NONSYNC" ] && continue
            # Content-identical drop: already converged with the pinned main
            # tip (also catches every freshly-synced file -- defense in
            # depth on top of the subject exclusion).
            if git -C "$WT" diff --quiet HEAD "$MAIN_SHA" -- "$P" 2>/dev/null; then
                continue
            fi
            printf '%s\n' "$P" >> "$OUT"
            DIV_COUNT=$((DIV_COUNT + 1))
        done < "$_tmp_div/isect.txt"
        if [ "$DIV_COUNT" -gt 0 ]; then
            printf 'DIVERGENCE=%s\n' diverged
        else
            printf 'DIVERGENCE=%s\n' clean
        fi
        printf 'DIVERGED_COUNT=%s\n' "$DIV_COUNT"
        printf 'MAIN_SHA=%s\n' "$MAIN_SHA"
        # %q: eval-safe under the two-step caller form even for a space-
        # bearing --out path (plain paths print unchanged).
        printf 'DIVERGED_FILE=%q\n' "$OUT"
        exit 0
        ;;

    *)
        _die_usage "unknown-guard-$GUARD"
        ;;
esac
