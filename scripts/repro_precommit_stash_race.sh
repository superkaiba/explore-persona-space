#!/usr/bin/env bash
# repro_precommit_stash_race.sh — deterministic scratch-repo reproduction of
# the shared-repo-root uncommitted-file reversion race (#2015).
#
# pre-commit (4.6.0; pre_commit/staged_files_only.py) stashes the REPO-WIDE
# unstaged tracked diff around every hook-driven commit: it writes the diff to
# ~/.cache/pre-commit/patch<epoch>-<pid>, runs `git checkout -- .` (reverting
# every session's unstaged tracked edits to index content for the hook
# window), and re-applies the patch after the hooks. A write landing INSIDE
# another commit's hook window is permanently lost when the restore conflicts
# (pre-commit rolls the whole tree back a second time and re-applies only its
# stale snapshot — the `Rolling back fixes` branch), and unstaged deletions
# are transiently resurrected. A CLEAN tracked tree makes pre-commit skip the
# stash entirely (staged_files_only.py:57-61, `if retcode == 0: yield`) —
# the self-eliminating property the #2015 mitigation relies on.
#
# Four self-asserting scenarios (all four must PASS; exit 0 iff all do):
#   S1 transient reversion      — unstaged edit reverts to HEAD mid-window,
#                                 restores post-commit; a patch* file appears.
#   S2 permanent loss           — a mid-window write conflicting with the
#                                 stashed hunk is DESTROYED by the
#                                 double-checkout rollback. The commit itself
#                                 exits rc=1: pre-commit's post-hook file-hash
#                                 check attributes the concurrent write to the
#                                 hook ("files were modified by this hook")
#                                 and fails the hook — the incident's
#                                 "FAILED commit (rc!=0, modified files) beside
#                                 a SUCCEEDING sibling push" shape (measured
#                                 2026-08-08; deviation from plan D0's rc==0
#                                 expectation, recorded in the round report).
#   S3 deletion resurrection    — an unstaged `rm` is resurrected (HEAD
#                                 content) for the hook window.
#   S4 clean-tree disarm        — with a clean tracked tree, NO patch* file
#                                 is created and a mid-window write SURVIVES
#                                 verbatim. S4 is the FALSIFIER for the #2015
#                                 plan: if a patch* file appears or the write
#                                 is lost, the clean-tree mitigation strategy
#                                 is wrong (plan §3). Residual (measured
#                                 2026-08-08): the COMMIT itself still exits
#                                 rc=1 via the same post-hook file-hash check
#                                 as S2 — on a clean tree a mid-window write
#                                 costs a FAILED-but-retryable commit, never
#                                 data. rc is NOT part of the falsifier.
#
# Scratch-only: runs in mktemp -d under /tmp with an isolated pre-commit
# cache (PRE_COMMIT_HOME/XDG_CACHE_HOME inside the scratch dir); hard-asserts
# it is NOT inside any existing git work tree and never touches the shared
# repo root. Footprint <1 MB; wall ~30-60 s (7 hook-gated commits x the
# WINDOW_S sleep). Mid-window actions key on an OBSERVED hook-start sentinel
# the hook touches as its FIRST action — never a fixed offset (a fixed offset
# races pre-commit's cold start and can invert S4 flakily).

set -euo pipefail

WINDOW_S="${WINDOW_S:-3}" # hook-window length; tunable for loaded machines

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_eq() { # name actual expected
    if [ "$2" != "$3" ]; then
        fail "$1: actual '$2' != expected '$3'"
    fi
    echo "PASS: $1"
}

assert_file_absent() { # name path
    [ ! -e "$2" ] || fail "$1: $2 exists"
    echo "PASS: $1"
}

assert_contains() { # name file needle
    grep -qF -- "$3" "$2" || fail "$1: '$3' not found in $2"
    echo "PASS: $1"
}

command -v git >/dev/null 2>&1 || fail "git not on PATH"
command -v pre-commit >/dev/null 2>&1 || fail "pre-commit not on PATH (~/.local/bin expected)"

SCRATCH="$(mktemp -d /tmp/stash-race-repro.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
REPO="$SCRATCH/repo"
SENTINEL="$SCRATCH/hook-started"
# Full cache isolation (plan D0: isolation preferred): patch* files land in
# the scratch cache, so concurrent REAL commits on this VM cannot confuse the
# patch-set delta assertions.
export PRE_COMMIT_HOME="$SCRATCH/cache/pre-commit"
export XDG_CACHE_HOME="$SCRATCH/cache"

echo "== repro_precommit_stash_race (#2015) =="
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git: $(git --version)"
echo "pre-commit: $(pre-commit --version)"
echo "scratch: $SCRATCH (window ${WINDOW_S}s)"

# Hard isolation asserts (plan D0): the scratch dir must not sit inside ANY
# git work tree (in particular not the shared repo root's), and after init
# the scratch repo's toplevel must be the scratch repo itself.
if git -C "$SCRATCH" rev-parse --show-toplevel >/dev/null 2>&1; then
    fail "scratch dir $SCRATCH is inside a git work tree — refusing to run"
fi
mkdir -p "$REPO"
git -C "$REPO" init -q -b main
TOP="$(git -C "$REPO" rev-parse --show-toplevel)"
assert_eq "isolation: scratch repo toplevel" "$TOP" "$REPO"
git -C "$REPO" config user.name "stash-race-repro"
git -C "$REPO" config user.email "repro@example.invalid"
git -C "$REPO" config commit.gpgsign false

# Base tree: three tracked files + the hook config (pre-commit refuses to run
# with an unstaged config, so it is part of the base commit).
printf 'alpha v1\n' >"$REPO/a.txt"
printf 'bravo v1\n' >"$REPO/b.txt"
printf 'charlie v1\n' >"$REPO/c.txt"
cat >"$REPO/.pre-commit-config.yaml" <<EOF
repos:
  - repo: local
    hooks:
      - id: slow-window
        name: slow-window (deterministic hook window)
        entry: bash -c 'touch "$SENTINEL" && sleep $WINDOW_S'
        language: system
        always_run: true
        pass_filenames: false
EOF
git -C "$REPO" add a.txt b.txt c.txt .pre-commit-config.yaml
git -C "$REPO" commit -q -m "base" # hook not installed yet — instant
(cd "$REPO" && pre-commit install >/dev/null)

# Untracked canary (plan §12 assumption 5): untracked files are INVISIBLE to
# the stash mechanism — asserted byte-stable after every scenario.
CANARY_CONTENT="untracked canary — must never change or vanish"
printf '%s\n' "$CANARY_CONTENT" >"$REPO/untracked-canary.txt"

patch_set() { find "$PRE_COMMIT_HOME" -maxdepth 1 -name 'patch*' 2>/dev/null | sort; }

new_patches() { # pre-set-string -> newline list of NEW patch files
    comm -13 <(printf '%s' "$1") <(patch_set)
}

COMMIT_PID=""
launch_commit() { # logfile
    rm -f "$SENTINEL"
    git -C "$REPO" commit -q -m "concurrent commit" >"$1" 2>&1 &
    COMMIT_PID=$!
}

wait_sentinel() {
    local i
    for i in $(seq 1 300); do
        [ -e "$SENTINEL" ] && return 0
        sleep 0.1
    done
    fail "hook-start sentinel never appeared within 30s"
}

finish_commit() { # scenario-name expected-rc
    local rc=0
    wait "$COMMIT_PID" || rc=$?
    # S1/S3/S4 commits succeed (rc 0). S2's commit FAILS rc=1 by design of
    # pre-commit's post-hook file-hash check (the mid-window write reads as a
    # hook "auto-fix": "files were modified by this hook") — the loss happens
    # regardless. Any OTHER rc means the stash-cycle semantics changed (e.g.
    # a pre-commit upgrade) — fail loud (plan D0 exit-path trace).
    assert_eq "$1: concurrent commit rc" "$rc" "$2"
}

assert_canary() { # scenario-name
    assert_eq "$1: untracked canary byte-stable" \
        "$(cat "$REPO/untracked-canary.txt")" "$CANARY_CONTENT"
}

normalize() {
    # Land all outstanding tracked modifications/deletions (explicit paths)
    # so each scenario starts from a clean tracked tree. Runs the hook
    # (window sleep) with NO unstaged diff -> no stash.
    git -C "$REPO" add -u -- a.txt b.txt c.txt
    if ! git -C "$REPO" diff --cached --quiet; then
        git -C "$REPO" commit -q -m "normalize"
    fi
    assert_eq "normalize: tracked tree clean" \
        "$(git -C "$REPO" status --porcelain -uno)" ""
}

# ── S1: transient reversion ──────────────────────────────────────────────────
echo "== S1: transient reversion =="
printf 'alpha v2 (unstaged)\n' >"$REPO/a.txt" # unstaged edit — arms the stash
printf 'bravo v2\n' >"$REPO/b.txt"
git -C "$REPO" add b.txt
PRE_PATCHES="$(patch_set)"
launch_commit "$SCRATCH/s1-commit.log"
wait_sentinel
assert_eq "S1 mid-window a.txt reverted to HEAD content" \
    "$(cat "$REPO/a.txt")" "alpha v1"
finish_commit "S1" 0
assert_eq "S1 post-commit a.txt restored to the unstaged edit" \
    "$(cat "$REPO/a.txt")" "alpha v2 (unstaged)"
S1_NEW="$(new_patches "$PRE_PATCHES")"
[ -n "$S1_NEW" ] || fail "S1: no new patch* file appeared in $PRE_COMMIT_HOME"
echo "PASS: S1 new patch file appeared ($(basename "$S1_NEW"))"
assert_canary "S1"

# ── S2: permanent loss (double-checkout rollback) ────────────────────────────
echo "== S2: permanent loss =="
normalize
printf 'alpha v2 for s2\n' >"$REPO/a.txt" # unstaged edit — the stashed hunk
printf 'bravo v3\n' >"$REPO/b.txt"
git -C "$REPO" add b.txt
launch_commit "$SCRATCH/s2-commit.log"
wait_sentinel
# Mid-window write CONFLICTING with the stashed hunk (the hunk expects
# 'alpha v1' worktree context at restore time).
printf 'alpha THIRD content (mid-window write)\n' >"$REPO/a.txt"
finish_commit "S2" 1
assert_eq "S2 post-commit a.txt == the STALE stashed snapshot (third content DESTROYED)" \
    "$(cat "$REPO/a.txt")" "alpha v2 for s2"
assert_contains "S2 'Rolling back fixes' in commit stderr" \
    "$SCRATCH/s2-commit.log" "Rolling back fixes"
assert_contains "S2 hook failed as 'files were modified by this hook' (the rc=1 mechanism)" \
    "$SCRATCH/s2-commit.log" "files were modified by this hook"
assert_canary "S2"

# ── S3: deletion resurrection ────────────────────────────────────────────────
echo "== S3: deletion resurrection =="
normalize
rm "$REPO/c.txt" # unstaged deletion
printf 'bravo v4\n' >"$REPO/b.txt"
git -C "$REPO" add b.txt
launch_commit "$SCRATCH/s3-commit.log"
wait_sentinel
[ -e "$REPO/c.txt" ] || fail "S3: mid-window c.txt was NOT resurrected"
assert_eq "S3 mid-window c.txt resurrected with HEAD content" \
    "$(cat "$REPO/c.txt")" "charlie v1"
finish_commit "S3" 0
assert_file_absent "S3 post-commit c.txt deletion restored (absent again)" "$REPO/c.txt"
assert_canary "S3"

# ── S4: clean-tree disarm (the mitigation's mechanism; plan §3 FALSIFIER) ────
echo "== S4: clean-tree disarm =="
normalize # commits the c.txt deletion -> tracked tree fully clean
printf 'bravo v5\n' >"$REPO/b.txt"
git -C "$REPO" add b.txt
PRE_PATCHES="$(patch_set)"
launch_commit "$SCRATCH/s4-commit.log"
wait_sentinel
printf 'alpha MID-WINDOW WRITE must survive\n' >"$REPO/a.txt"
finish_commit "S4" 1 # same post-hook file-hash check as S2; retryable, no data touched
assert_contains "S4 rc=1 is the 'files were modified by this hook' check (not a stash rollback)" \
    "$SCRATCH/s4-commit.log" "files were modified by this hook"
S4_NEW="$(new_patches "$PRE_PATCHES")"
[ -z "$S4_NEW" ] || fail "S4 FALSIFIER: patch file(s) created on a clean tree: $S4_NEW"
echo "PASS: S4 no patch* file created on a clean tracked tree"
assert_eq "S4 mid-window write survives the concurrent commit verbatim" \
    "$(cat "$REPO/a.txt")" "alpha MID-WINDOW WRITE must survive"
assert_canary "S4"

echo "== ALL FOUR SCENARIOS PASS =="
echo "The race is fully reproduced by pre-commit's stash cycle (S1-S3), and a"
echo "clean tracked tree disarms it structurally (S4): no patch* file, the"
echo "mid-window write survives — the #2015 mitigation premise holds. Residual"
echo "on a clean tree: a mid-window write can still fail the CONCURRENT commit"
echo "(rc=1, 'files were modified by this hook') — retryable, no data loss."
