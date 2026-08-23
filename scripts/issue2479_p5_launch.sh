#!/usr/bin/env bash
# Issue #2479 narrow-pod (1xH100) P5 fits wrapper: axis-freeze guard probe ->
# stage ladder sources -> phasef driver over the 24 panel cells (fits +
# ladders, guard LIVE) -> commit+push cell/ladder JSONs on the issue branch ->
# HF eval-mirror upload with listing verify (plan v4 §4/§9).
#
# Named by the plan §9 dispatch command:
#   dispatch_issue.py launch --issue 2479 --intent eval --gpus 1 --backend runpod \
#     --repo-branch issue-2479 --time-budget-hours 8 \
#     --workload-cmd 'bash scripts/issue2479_p5_launch.sh'
#
# The axis-freeze guard runs LIVE: the fill entrypoint is invoked WITHOUT
# --pilot-outdir, so every panel-cell fit asserts eval_results/issue_2479/
# axis_freeze.json is committed at an ancestor of HEAD (plan §4 Step 3). An
# early wrapper probe fails loud BEFORE any multi-GB staging.
#
# Pod-side reporting contract: NEVER shells out to scripts/task.py; progress =
# [phase=...] stdout breadcrumbs + envelope sentinels under /workspace/logs/.
# Resume: the driver's own expected_outputs per-cell skip + the stager's
# .staged_complete sentinels.
#
# Designed exit codes: 47 = results-git push failure; 48 = HF eval-mirror
# upload/verify failure; 49 = axis-freeze guard refusal (fix: commit
# axis_freeze.json via scripts/issue2479_freeze_axis.py --commit + push, then
# re-run); 50 = no captured panel cells on HF; 2 = arg error. A non-zero
# driver rc propagates AFTER the completed JSONs are committed+pushed.
#
# Usage: bash scripts/issue2479_p5_launch.sh [--dry-run]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[i2479-p5] unknown arg: $1" >&2; exit 2 ;;
  esac
done

export EPM_I2479_CHAR_PANEL_JSON="${EPM_I2479_CHAR_PANEL_JSON:-${REPO_ROOT}/eval_results/issue_2479/panel.json}"

LOG_DIR="${SENTINEL_DIR:-/workspace/logs}"
DATA_BASE="${EPM_I2479_DATA_BASE:-/workspace/data/issue_2479}"
STAGE_ROOT="${DATA_BASE}/stage"
CACHE_DIR="${DATA_BASE}/fill_cache"
OUT_DIR="${REPO_ROOT}/eval_results/issue_2479/story_char_gradient"
BRANCH="issue-2479"
HF_REPO="superkaiba1/explore-persona-space-data"
MIRROR_PREFIX="issue2479_ai_likeness_gradient/eval_mirror/story_char_gradient"

# Panel cell table (variant|src|model) — the phasef driver's row format.
PANEL_ROWS="$(uv run python - <<'PY'
import sys

sys.path.insert(0, "scripts")
from issue2479_char_panel import load_char_panel_env

rows = load_char_panel_env()
assert rows, "EPM_I2479_CHAR_PANEL_JSON set but loader returned no rows"
for r in rows:
    print(f"{r['variant_op']}|r4op|instruct")
    if r["variant_inserted"]:
        print(f"{r['variant_inserted']}|r4|instruct")
PY
)" || { echo "[i2479-p5] panel load failed" >&2; exit 2; }
PANEL_CELLS=()
while IFS= read -r row; do
  [ -n "$row" ] && PANEL_CELLS+=("${row%%|*}")
done <<< "$PANEL_ROWS"

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] panel=${EPM_I2479_CHAR_PANEL_JSON}"
  echo "[dry-run] P5 cell table (${#PANEL_CELLS[@]} rows, variant|src|model):"
  k=0
  while IFS= read -r row; do
    [ -n "$row" ] || continue
    k=$((k + 1))
    printf '  P5[%02d/%d] %s\n' "$k" "${#PANEL_CELLS[@]}" "$row"
  done <<< "$PANEL_ROWS"
  echo "[dry-run] guard probe: assert_axis_freeze_guard(repo_root) BEFORE staging (guard LIVE — no --pilot-outdir anywhere in P5)"
  echo "[dry-run] stage: issue1345_stage_char_stories.py --sources r4 r4op --dest-root ${STAGE_ROOT}"
  echo "[dry-run] driver: bash scripts/issue1345_char_phasef_driver.sh --cells <captured subset of the ${#PANEL_CELLS[@]} cells> --stage-root ${STAGE_ROOT} --cache-dir ${CACHE_DIR} --out-dir ${OUT_DIR}"
  echo "[dry-run] results git: add ${OUT_DIR#"${REPO_ROOT}"/} -> commit -> fetch+rebase -> push ${BRANCH} -> rev-list==0 + per-file ls-tree presence assert (exit 47 on failure)"
  echo "[dry-run] HF mirror: upload_folder -> ${HF_REPO}:${MIRROR_PREFIX} + scoped list_repo_tree verify (exit 48 on failure)"
  echo "[dry-run] sentinel: ${LOG_DIR}/issue-2479-p5-results.json; terminal [phase=done]"
  exit 0
fi

mkdir -p logs "$LOG_DIR" "$DATA_BASE"

write_sentinel() { # gate blocks note
  local gate="$1" blocks="$2" note="$3"
  local out="${LOG_DIR}/issue-2479-p5-results.json"
  uv run python - "$out" "$gate" "$blocks" "$note" <<'PY'
import json
import os
import sys
import time

out, gate, blocks, note = sys.argv[1:5]
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "gate": gate,
    "blocks_pipeline": blocks == "1",
    "note": f"{note} | ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
}
tmp = out + ".tmp"
with open(tmp, "w") as f:
    json.dump(payload, f)
os.replace(tmp, out)
print(f"[sentinel] {out}", flush=True)
PY
}

# =============================================================================
echo "[phase=p5_guard]"
# Early fail-loud probe of the SAME guard the fill asserts per panel cell —
# BEFORE any multi-GB staging (RuntimeError carries the exact remedy).
if ! uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from issue1345_story_char_ladder_fill import assert_axis_freeze_guard

sha = assert_axis_freeze_guard(Path.cwd())
print(f"[freeze-guard] OK: axis frozen at commit {sha}", flush=True)
PY
then
  echo "[i2479-p5] AXIS-FREEZE GUARD REFUSAL: commit eval_results/issue_2479/axis_freeze.json (scripts/issue2479_freeze_axis.py --commit) on ${BRANCH} and push, then re-run" >&2
  write_sentinel "p5_guard" 1 "issue-2479 P5 HALT: axis-freeze guard refusal (axis_freeze.json not committed-at-ancestor, or pre-freeze JSONs in ${OUT_DIR})"
  exit 49
fi

# =============================================================================
echo "[phase=p5_stage_sources]"
if ! uv run python scripts/issue1345_stage_char_stories.py --sources r4 r4op --dest-root "$STAGE_ROOT"; then
  echo "[i2479-p5] source staging failed" >&2
  write_sentinel "p5_stage" 1 "issue-2479 P5 source staging failed (r4/r4op turnstores at the plan §10 pins)"
  exit 1
fi

# =============================================================================
echo "[phase=p5_fits]"
# Fit the CAPTURED subset: a P1 yield-halted cell has no turnstore on HF; probe
# the capture launcher's per-cell completion markers in one API pass.
CAPTURED_OUT="$(uv run python - "${PANEL_CELLS[@]}" <<'PY'
import sys

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

api = HfApi()
for variant in sys.argv[1:]:
    marker = f"issue1345_framing/{variant}/analysis_tensors/turnstore/_capture_complete.json"
    ok = hub.retry_transient(
        lambda m=marker: api.file_exists(
            "superkaiba1/explore-persona-space-data", m, repo_type="dataset"
        ),
        what=f"file_exists({marker})",
    )
    if ok:
        print(variant)
    else:
        print(f"[i2479-p5] {variant}: no _capture_complete marker — cell skipped", file=sys.stderr)
PY
)" || { echo "[i2479-p5] capture-marker probe failed" >&2; write_sentinel "p5_fits" 1 "issue-2479 P5 FAIL: HF capture-marker probe errored"; exit 1; }
CAPTURED=()
while IFS= read -r v; do
  [ -n "$v" ] && CAPTURED+=("$v")
done <<< "$CAPTURED_OUT"
echo "[i2479-p5] captured cells: ${#CAPTURED[@]}/${#PANEL_CELLS[@]}"
if [ "${#CAPTURED[@]}" -lt 1 ]; then
  echo "[i2479-p5] NO captured panel cells on HF — run P1/P4 first" >&2
  write_sentinel "p5_fits" 1 "issue-2479 P5 HALT: zero captured panel cells on HF (P4 completion markers absent)"
  exit 50
fi

driver_rc=0
bash scripts/issue1345_char_phasef_driver.sh --cells "${CAPTURED[@]}" \
  --stage-root "$STAGE_ROOT" --cache-dir "$CACHE_DIR" --out-dir "$OUT_DIR" \
  || driver_rc=$?
echo "[i2479-p5] phasef driver rc=${driver_rc}"

# =============================================================================
echo "[phase=p5_results_git]"
# ALWAYS commit+push whatever cell/ladder JSONs completed — even on a partial
# driver failure (checkpoint-per-cell; the JSONs are the durable science).
REL_OUT="${OUT_DIR#"${REPO_ROOT}"/}"
mapfile -t PRODUCED < <(find "$OUT_DIR" -maxdepth 1 -name '*.json' -printf '%f\n' 2>/dev/null | sort)
echo "[i2479-p5] declared result set (${#PRODUCED[@]} files under ${REL_OUT}):"
printf '  %s\n' "${PRODUCED[@]:-<none>}"
if [ "${#PRODUCED[@]}" -lt 1 ]; then
  echo "[i2479-p5] EMPTY result set — nothing to push (driver rc=${driver_rc})" >&2
  write_sentinel "p5_results_git" 1 "issue-2479 P5 FAIL: zero result JSONs produced (driver rc=${driver_rc})"
  exit 47
fi
# Pod clones may lack a git identity; commit fails loud otherwise.
git config user.email > /dev/null 2>&1 || git config user.email "pod-2479@eps.local"
git config user.name > /dev/null 2>&1 || git config user.name "eps-pod-2479"
git add -- "$REL_OUT"
if ! git diff --cached --quiet; then
  git commit -m "task #2479 P5: story_char_gradient cell/ladder JSONs (${#PRODUCED[@]} files, driver rc=${driver_rc})" -- "$REL_OUT" \
    || { echo "[i2479-p5] commit failed" >&2; write_sentinel "p5_results_git" 1 "issue-2479 P5 FAIL: git commit failed"; exit 47; }
else
  echo "[i2479-p5] no staged changes (results already committed — resume no-op)"
fi
push_ok=0
for attempt in 1 2; do
  git fetch origin "$BRANCH" && git rebase "origin/${BRANCH}" || { git rebase --abort 2>/dev/null; true; }
  if git push origin "HEAD:refs/heads/${BRANCH}"; then push_ok=1; break; fi
  echo "[i2479-p5] push attempt ${attempt} failed" >&2
  sleep 15
done
git fetch origin "$BRANCH"
behind="$(git rev-list --count "origin/${BRANCH}..HEAD" -- 2>/dev/null || echo unknown)"
missing=0
for f in "${PRODUCED[@]}"; do
  if ! git ls-tree -r "origin/${BRANCH}" --name-only -- "${REL_OUT}/${f}" | grep -q .; then
    echo "[i2479-p5] MISSING on origin/${BRANCH}: ${REL_OUT}/${f}" >&2
    missing=$((missing + 1))
  fi
done
if [ "$push_ok" -ne 1 ] || [ "$behind" != "0" ] || [ "$missing" -ne 0 ]; then
  echo "[i2479-p5] RESULT-PUSH VERIFICATION FAILED (push_ok=${push_ok} unpushed=${behind} missing=${missing})" >&2
  write_sentinel "p5_results_git" 1 "issue-2479 P5 FAIL: result push verification (push_ok=${push_ok} unpushed_commits=${behind} missing_files=${missing} of ${#PRODUCED[@]})"
  exit 47
fi
echo "[i2479-p5] results git verified: ${#PRODUCED[@]} files on origin/${BRANCH}, 0 unpushed commits"

# =============================================================================
echo "[phase=p5_hf_mirror]"
if ! uv run python - "$OUT_DIR" "$MIRROR_PREFIX" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

out_dir, prefix = Path(sys.argv[1]), sys.argv[2]
repo = "superkaiba1/explore-persona-space-data"
local = sorted(p.name for p in out_dir.glob("*.json"))
assert local, f"empty declared upload set under {out_dir}"
api = HfApi()
hub.retry_transient(
    lambda: api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=prefix,
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["*.json"],
        commit_message=f"issue-2479 P5: eval mirror ({len(local)} story_char_gradient JSONs)",
    ),
    what=f"upload_folder({prefix})",
)
remote = {
    Path(e.path).name
    for e in hub.retry_transient(
        lambda: list(api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset")),
        what=f"list_repo_tree({prefix})",
    )
}
missing = [f for f in local if f not in remote]
assert not missing, f"eval-mirror verify: {len(missing)} of {len(local)} files missing: {missing[:5]}"
print(f"[i2479-p5] eval mirror verified: {len(local)} files at {repo}:{prefix}", flush=True)
PY
then
  echo "[i2479-p5] HF eval-mirror upload/verify failed" >&2
  write_sentinel "p5_hf_mirror" 1 "issue-2479 P5 FAIL: HF eval-mirror upload/verify (${MIRROR_PREFIX})"
  exit 48
fi

if [ "$driver_rc" -ne 0 ]; then
  write_sentinel "p5_fits" 1 "issue-2479 P5 PARTIAL: driver rc=${driver_rc} (completed JSONs pushed + mirrored: ${#PRODUCED[@]}; captured cells ${#CAPTURED[@]}/${#PANEL_CELLS[@]}) — re-run resumes via expected_outputs"
  exit "$driver_rc"
fi
write_sentinel "p5" 0 "issue-2479 P5 complete: ${#PRODUCED[@]} cell/ladder JSONs on origin/${BRANCH} (${REL_OUT}) + eval mirror ${MIRROR_PREFIX}; cells fit ${#CAPTURED[@]}/${#PANEL_CELLS[@]} (guard LIVE, freeze commit asserted per fill invocation)"
echo "[phase=done]"
exit 0
