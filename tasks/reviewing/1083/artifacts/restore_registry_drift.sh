#!/usr/bin/env bash
# restore_registry_drift.sh — task #1083 registry-drift repair (plan §4 Phase 2)
#
# Create-only restore of 15 task dirs' state files from named git commits.
# - Never overwrites an existing file/symlink; never deletes; never moves.
# - Only `git show` / `git ls-tree` / `git cat-file` reads + file creates.
# - Per-item failure (missing blob, frontmatter parse fail, expected-title
#   mismatch) PARKS that row (nothing written for it) and continues the rest.
# - A byte-match failure on a file this script just created is systemic
#   (concurrent modification / disk error) -> abort loud.
# Emits a manifest of every path created (the exact staging list) to
# <task1083_dir>/artifacts/restore_manifest.txt.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TASK_DIR="$(uv run python scripts/task.py find 1083)"
MANIFEST="$TASK_DIR/artifacts/restore_manifest.txt"
: > "$MANIFEST"

die() { echo "FATAL: $*" >&2; exit 1; }

# check_body <path-to-body.md> <expected_title>
# exit 0 = OK; exit 3 = parse fail; exit 4 = title mismatch
check_body() {
  EPM_EXPECTED_TITLE="$2" uv run python - "$1" <<'PY'
import os, sys, pathlib
from explore_persona_space.task_workflow import _read_body
p = pathlib.Path(sys.argv[1])
try:
    fm, _ = _read_body(p)
except ValueError as e:
    print(f"PARSE-FAIL: {p}: {e}")
    sys.exit(3)
exp = os.environ["EPM_EXPECTED_TITLE"]
got = fm.get("title", "")
if got != exp:
    print(f"TITLE-MISMATCH: {p}")
    print(f"  got:      {got!r}")
    print(f"  expected: {exp!r}")
    sys.exit(4)
print(f"BODY-OK: {p}")
PY
}

# 5-column row table: id|sha|src_prefix|dst_prefix|expected_title
# expected_title pulled from `git show <sha>:<src>/body.md` frontmatter at
# COMPOSE TIME (parsed via task_workflow._read_body) and embedded as literals,
# so the runtime check is independent of the script's own (sha, src) values.
ROWS=(
"696|5ade45c23d|tasks/completed/696|tasks/completed/696|preflight branch-freshness check: tolerate transient git-fetch timeouts (#664 false-positive recurrence)"
"698|0bf0a75f5a|tasks/completed/698|tasks/completed/698|workflow-fix: add gate-id-uniqueness check to workflow_lint.py"
"699|a3f5206554|tasks/completed/699|tasks/completed/699|workflow-fix: atomic post_event append (events.jsonl partial-line corruption)"
"700|4e5f704828|tasks/completed/700|tasks/completed/700|rerun-discipline: confirm a fix's code-path is reached + differential-diagnose a hang before reprovisioning"
"701|af81947dc0|tasks/running/701|tasks/completed/701|planner/critic: a torch-MLP LOCO fit is NOT a cheap CPU-only analysis step"
"702|736f5c0c62|tasks/completed/702|tasks/completed/702|analyzer: confirm all planned control arms present before authoring a verdict"
"704|d35ab320a4|tasks/running/704|tasks/completed/704|fix backend_poll last_log_mtime_sec_ago clock/epoch skew"
"705|a5ba9b3eb2|tasks/reviewing/705|tasks/completed/705|fix confirm_artifacts finalize gate on phase-scoped / multi-attempt launches"
"706|adddf98b4d|tasks/approved/706|tasks/completed/706|/daily held-items route to PM-tracked tasks + PM digest line (stop the rot, add independent review)"
"707|c769a54eae|tasks/archived/707|tasks/archived/707|spend-approval park must stay sticky across resume (not un-parked by a config-drift cap bump)"
"708|35406c61a0|tasks/archived/708|tasks/archived/708|verify #664 base tokenizer — Mistral-Small regex warning on a Qwen-2.5-7B project"
"709|cd06cf7ebe|tasks/proposed/709|tasks/planning/709|RunPod SSH glob_sentinels — close #705's deferred stale-sentinel sibling probe over SSH"
"750|ba167d7bc1|tasks/completed/750|tasks/completed/750|workflow-fix: gate _eps_phase done on OOM-detection (GCP)"
"764|706fb7e06e|tasks/completed/764|tasks/completed/764|workflow-fix: document VM earlyoom silent-SIGTERM + stream-reduce recipe in gotchas.md"
"766|e6fe393a6b|tasks/reviewing/766|tasks/completed/766|Fix three judging correctness bugs (belief.py default-to-50, i653 EM-threshold bool, stale Haiku pins #650/#657)"
)

PARKED=()

for row in "${ROWS[@]}"; do
  IFS='|' read -r id sha src_prefix dst_prefix expected_title <<< "$row"
  echo "== #$id  $sha  $src_prefix -> $dst_prefix"

  # Step 0 — id-identity assertion (pre-restore, per row): the numeric id
  # embedded in src_prefix and dst_prefix must equal the row id.
  src_id="${src_prefix##*/}"
  dst_id="${dst_prefix##*/}"
  [[ "$src_id" =~ ^[0-9]+$ ]] || die "#$id: src_prefix '$src_prefix' has non-numeric trailing component"
  [[ "$dst_id" =~ ^[0-9]+$ ]] || die "#$id: dst_prefix '$dst_prefix' has non-numeric trailing component"
  [ "$src_id" = "$id" ] || die "#$id: src_prefix id mismatch ($src_prefix)"
  [ "$dst_id" = "$id" ] || die "#$id: dst_prefix id mismatch ($dst_prefix)"

  # Per-item pre-check BEFORE any write: body blob exists, parses, and carries
  # the expected title. Failure -> PARK the row with nothing written.
  if ! git cat-file -e "$sha:$src_prefix/body.md" 2>/dev/null; then
    echo "PARK #$id: body blob missing at $sha:$src_prefix/body.md"
    PARKED+=("$id:blob-missing")
    continue
  fi
  tmp_body="$(mktemp)"
  git show "$sha:$src_prefix/body.md" > "$tmp_body"
  rc=0
  check_body "$tmp_body" "$expected_title" || rc=$?
  rm -f "$tmp_body"
  if [ "$rc" -ne 0 ]; then
    echo "PARK #$id: source body pre-check failed (rc=$rc; 3=parse-fail 4=title-mismatch)"
    PARKED+=("$id:body-precheck-rc$rc")
    continue
  fi

  created_paths=()
  while read -r mode _type _obj src_path; do
    [ -n "$src_path" ] || continue
    # Skip every path under artifacts/
    case "$src_path" in
      "$src_prefix"/artifacts/*) continue ;;
    esac
    dst_path="$dst_prefix${src_path#"$src_prefix"}"
    # Per-restored-path id-identity: dst_path must live under dst_prefix/.
    case "$dst_path" in
      "$dst_prefix"/*) : ;;
      *) die "#$id: mapped dst_path '$dst_path' escapes dst_prefix '$dst_prefix'" ;;
    esac
    # Create-only: skip any dst that already exists (the -L leg is load-bearing:
    # a bare -e follows symlinks, so a dangling symlink would read "absent").
    if [ -e "$dst_path" ] || [ -L "$dst_path" ]; then
      echo "  skip (exists): $dst_path"
      continue
    fi
    mkdir -p "$(dirname "$dst_path")"
    if [ "$mode" = "120000" ]; then
      ln -s "$(git show "$sha:$src_path")" "$dst_path"
    elif [ "$mode" = "100644" ] || [ "$mode" = "100755" ]; then
      git show "$sha:$src_path" > "$dst_path"
    else
      die "#$id: unexpected mode $mode for $src_path"
    fi
    echo "  created: $dst_path"
    created_paths+=("$dst_path|$mode|$src_path")
    printf '%s\n' "$dst_path" >> "$MANIFEST"
  done < <(git ls-tree -r "$sha" -- "$src_prefix")

  # Byte-match verification of every path this row created (systemic on fail).
  for entry in "${created_paths[@]:-}"; do
    [ -n "$entry" ] || continue
    IFS='|' read -r dst_path mode src_path <<< "$entry"
    if [ "$mode" = "120000" ]; then
      [ "$(readlink "$dst_path")" = "$(git show "$sha:$src_path")" ] \
        || die "#$id: symlink target mismatch at $dst_path"
    else
      git show "$sha:$src_path" | cmp - "$dst_path" \
        || die "#$id: byte mismatch at $dst_path (source $sha:$src_path)"
    fi
  done

  # Post-restore frontmatter parse + expected-title check on the on-disk body.
  rc=0
  check_body "$dst_prefix/body.md" "$expected_title" || rc=$?
  [ "$rc" -eq 0 ] || die "#$id: post-restore body check failed at $dst_prefix/body.md (rc=$rc)"
done

echo "---"
echo "Manifest: $MANIFEST ($(wc -l < "$MANIFEST") paths created)"
if [ "${#PARKED[@]}" -gt 0 ]; then
  echo "PARKED (per-item, Phase 7): ${PARKED[*]}"
  exit 5
fi
echo "RESTORE OK — 0 parked"
