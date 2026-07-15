#!/usr/bin/env bash
# issue-1345 pod-side phase driver:
#   prefetch -> phase0 -> gen_stories -> extract_r1r2 -> extract_stories ->
#   matchedn -> fits -> transfer -> opcomp -> plots -> upload -> push.
# Runs under the GCP lane contract: REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1345_dispatch.sh
# SMOKE (--smoke) runs the IDENTICAL phase chain at tiny row-n with outputs
# diverted to a scratch root + the issue1345_smoke/ HF prefix (PASS_UNIFIED).
# --dry-run composes + prints every phase command, writes the sentinel, exits 0
# (the poller-facing plumbing check; no GPU work).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT" || { echo "FATAL: cd $REPO_ROOT failed" >&2; exit 1; }
# GCE lane exports tokens via startup metadata and has NO .env — conditional only.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

PHASE="all"; FROM_PHASE=""; SMOKE=""; DRY_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --from-phase) FROM_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

# Extractor RSS trim (#825 run-5 kernel OOM: arena free lists retained blocks)
export MALLOC_MMAP_THRESHOLD_=131072

PHASES=(prefetch phase0 gen_stories extract_r1r2 extract_stories matchedn fits transfer opcomp plots upload push)

if [[ -n "$SMOKE" ]]; then
  OUT_ROOT="${EPM_OUTPUT_ROOT:-/tmp/issue-1345-smoke}"
  EVAL_DIR="$OUT_ROOT/eval_results"; FIG_DIR="$OUT_ROOT/figures"; DATA_DIR="$OUT_ROOT/data"
  SMOKE_FLAG="--smoke"; NULLS=3; NBOOT=25; ROTD=5
else
  EVAL_DIR="eval_results/issue_1345"; FIG_DIR="figures/issue_1345"; DATA_DIR="data/issue_1345"
  SMOKE_FLAG=""; NULLS=20; NBOOT=1000; ROTD=50
fi
TS_DIR="$DATA_DIR/turnstore"; STORIES_DIR="$DATA_DIR/stories"
MATCHED_DIR="$DATA_DIR/matched_n"; PREDS_DIR="$DATA_DIR/preds_cache"; DL_DIR="$DATA_DIR/hf_dl"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="$REPO_ROOT/logs"; mkdir -p "$LOG_DIR"; }
mkdir -p "$EVAL_DIR" "$FIG_DIR" "$TS_DIR" "$STORIES_DIR" "$MATCHED_DIR" "$PREDS_DIR" "$DL_DIR"

R3_HALT_FILE="$DATA_DIR/story_regime_halted"
NO_R3_FLAG=""
[[ -f "$R3_HALT_FILE" ]] && NO_R3_FLAG="--no-r3"

NGPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
echo "[dispatch] issue-1345 phase=$PHASE from=$FROM_PHASE smoke=${SMOKE:-0} dry=${DRY_RUN:-0} ngpu=$NGPU"

should_run() {
  local phase="$1"
  if [[ "$PHASE" != "all" && "$PHASE" != "$phase" ]]; then return 1; fi
  if [[ -n "$FROM_PHASE" ]]; then
    local started=""
    for p in "${PHASES[@]}"; do
      [[ "$p" == "$FROM_PHASE" ]] && started="yes"
      if [[ "$p" == "$phase" ]]; then [[ -n "$started" ]] && return 0 || return 1; fi
    done
  fi
  return 0
}

run_cmd() {
  # Print, and execute unless --dry-run.
  echo "[cmd] $*"
  if [[ -z "$DRY_RUN" ]]; then "$@"; fi
}

# Run one command per model on its own GPU (parallel when >=2 GPUs visible).
# Usage: run_per_model <log_tag> <cmd...>  ('%MODEL%' substituted per model)
run_per_model() {
  local tag="$1"; shift
  local rc_i=0 rc_p=0
  if [[ -n "$DRY_RUN" ]]; then
    for m in instruct pretrained; do
      echo "[cmd] CUDA_VISIBLE_DEVICES=<slot> ${*//%MODEL%/$m}"
    done
    return 0
  fi
  if [[ "$NGPU" -ge 2 ]]; then
    local cmd_i="${*//%MODEL%/instruct}" cmd_p="${*//%MODEL%/pretrained}"
    echo "[fanout] $tag: instruct on GPU0, pretrained on GPU1"
    CUDA_VISIBLE_DEVICES=0 bash -c "$cmd_i" > "$LOG_DIR/i1345_${tag}_instruct.log" 2>&1 &
    local p1=$!
    CUDA_VISIBLE_DEVICES=1 bash -c "$cmd_p" > "$LOG_DIR/i1345_${tag}_pretrained.log" 2>&1 &
    local p2=$!
    wait "$p1" || rc_i=$?
    wait "$p2" || rc_p=$?
    for m in instruct pretrained; do
      echo "[fanout] $tag/$m log tail:"; tail -n 12 "$LOG_DIR/i1345_${tag}_${m}.log" || true
    done
  else
    for m in instruct pretrained; do
      local cmd="${*//%MODEL%/$m}"
      echo "[serial] $tag: $m"
      CUDA_VISIBLE_DEVICES=0 bash -c "$cmd" > "$LOG_DIR/i1345_${tag}_${m}.log" 2>&1 || {
        rc=$?
        if [[ "$m" == "instruct" ]]; then rc_i=$rc; else rc_p=$rc; fi
      }
      tail -n 12 "$LOG_DIR/i1345_${tag}_${m}.log" || true
    done
  fi
  RC_INSTRUCT=$rc_i; RC_PRETRAINED=$rc_p
  return 0
}

# gen_stories rc routing (v3 code-review Minor, rc-masking fix): the rc=21
# yield-halt branch fires ONLY when BOTH models exited in {0, 21}; a real
# crash rc (1/134/137/...) in EITHER model routes to fatal instead of riding
# the halt branch. Echoes: halt | ok | fatal.
gen_rc_route() {
  local rc_i="$1" rc_p="$2"
  local ok_i=0 ok_p=0
  [[ "$rc_i" -eq 0 || "$rc_i" -eq 21 ]] && ok_i=1
  [[ "$rc_p" -eq 0 || "$rc_p" -eq 21 ]] && ok_p=1
  if [[ "$ok_i" -eq 1 && "$ok_p" -eq 1 ]]; then
    if [[ "$rc_i" -eq 21 || "$rc_p" -eq 21 ]]; then echo halt; else echo ok; fi
  else
    echo fatal
  fi
}

if should_run prefetch; then
  echo "[phase=prefetch]"
  run_cmd uv run python -c "import sys; sys.path.insert(0, 'scripts'); import issue1345_common as c; [c.list_parent_shards(s) for s in c.PARENT_STEMS]; print('prefetch OK: all four pinned stems resolve @', c.PIN_REV)"
fi

if should_run phase0; then
  echo "[phase=phase0]"
  run_cmd uv run python scripts/issue1345_fit_cells.py --phase0 \
    --dl-dir "$DL_DIR" --out-dir "$EVAL_DIR"
fi

if should_run gen_stories; then
  echo "[phase=gen_stories]"
  run_per_model gen "uv run python scripts/issue1345_gen_stories.py --model %MODEL% --out-dir '$STORIES_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG"
  if [[ -z "$DRY_RUN" ]]; then
    # rc=21 == yield floor failed (plan §7): halt the story regime, continue
    # r1/r2 — but ONLY when both rcs are in {0,21}; any other rc is a crash.
    case "$(gen_rc_route "${RC_INSTRUCT:-0}" "${RC_PRETRAINED:-0}")" in
      halt)
        echo "[gen_stories] YIELD FLOOR FAILED (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED}) — story regime halted"
        touch "$R3_HALT_FILE"; NO_R3_FLAG="--no-r3" ;;
      fatal)
        echo "FATAL: gen_stories failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1 ;;
      ok) : ;;
    esac
  fi
fi

if should_run extract_r1r2; then
  echo "[phase=extract_r1r2]"
  run_per_model extract_r1r2 "uv run python scripts/issue1345_extract_turnstore.py --regime r1 --model %MODEL% --out-dir '$TS_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG && uv run python scripts/issue1345_extract_turnstore.py --regime r2 --model %MODEL% --out-dir '$TS_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: extract_r1r2 failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run extract_stories; then
  if [[ -n "$NO_R3_FLAG" ]]; then
    echo "[phase=extract_stories] SKIPPED — story regime halted (yield floor)"
  else
    echo "[phase=extract_stories]"
    run_per_model extract_stories "uv run python scripts/issue1345_extract_turnstore.py --regime r3 --model %MODEL% --out-dir '$TS_DIR' --stories-dir '$STORIES_DIR' $SMOKE_FLAG"
    if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
      echo "FATAL: extract_stories failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
    fi
  fi
fi

if should_run matchedn; then
  echo "[phase=matchedn]"
  # Parity gate (±0.02 vs pinned anchors; exit 3 halts) + matched-n subsets.
  # $SMOKE_FLAG demotes ONLY the anchor comparison to informational — the
  # anchors bind at production n; the computation still runs (PASS_UNIFIED).
  run_cmd env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1345_fit_cells.py \
    --parity --build-matched $NO_R3_FLAG $SMOKE_FLAG \
    --turnstore-dir "$TS_DIR" --matched-dir "$MATCHED_DIR" --out-dir "$EVAL_DIR"
fi

if should_run fits; then
  echo "[phase=fits]"
  run_per_model fits "uv run python -c \"import sys; sys.path.insert(0,'scripts'); import issue1345_common as c; print(','.join(x['cell_id'] for x in c.all_cells() if x['model_key']=='%MODEL%'))\" > /tmp/i1345_cells_%MODEL%.txt && uv run python scripts/issue1345_fit_cells.py --cells \$(cat /tmp/i1345_cells_%MODEL%.txt) $NO_R3_FLAG $SMOKE_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --preds-dir '$PREDS_DIR' --null-draws $NULLS --n-boot $NBOOT"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: fits failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run transfer; then
  echo "[phase=transfer]"
  run_per_model transfer "uv run python scripts/issue1345_cross_regime_transfer.py --models %MODEL% $NO_R3_FLAG $SMOKE_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --preds-dir '$PREDS_DIR' --n-boot $NBOOT"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: transfer failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run opcomp; then
  echo "[phase=opcomp]"
  run_per_model opcomp "uv run python scripts/issue1345_operator_comparison.py --models %MODEL% $NO_R3_FLAG $SMOKE_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --rot-draws $ROTD"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: opcomp failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run plots; then
  echo "[phase=plots]"
  run_cmd uv run python scripts/issue1345_plots.py $NO_R3_FLAG \
    --out-dir "$EVAL_DIR" --fig-dir "$FIG_DIR" --turnstore-dir "$TS_DIR" --stories-dir "$STORIES_DIR"
fi

if should_run upload; then
  echo "[phase=upload]"
  DELETE_LOCAL=""
  [[ -z "$SMOKE" ]] && DELETE_LOCAL="--delete-local-turnstore"
  run_cmd uv run python scripts/issue1345_upload.py $SMOKE_FLAG $DELETE_LOCAL \
    --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
    --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
fi

if should_run push; then
  echo "[phase=push]"
  if [[ -n "$SMOKE" || -n "$DRY_RUN" ]]; then
    echo "[push] smoke/dry-run: scratch outputs — no git commit"
  else
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    git add "$EVAL_DIR" "$FIG_DIR"
    if git diff --cached --quiet; then
      echo "[push] nothing new to commit"
    else
      git -c user.email=pod@eps.local -c user.name="issue1345-dispatch" \
        commit -m "issue-1345: eval JSONs + figures (pod-side)"
    fi
    PUSH_OK=""
    for attempt in 1 2; do
      if git push origin "HEAD:$BRANCH"; then PUSH_OK=1; break; fi
      echo "[push] attempt $attempt failed; retrying" >&2
      sleep 10
    done
    if [[ -z "$PUSH_OK" ]]; then echo "FATAL: git push failed twice" >&2; exit 86; fi
    BEHIND="$(git rev-list --count "origin/$BRANCH..HEAD")"
    if [[ "$BEHIND" != "0" ]]; then
      echo "FATAL: push verify — $BEHIND commit(s) not on origin/$BRANCH" >&2; exit 86
    fi
    # Artifact-presence assert (#1325): every declared result file in the PUSHED tree
    MISSING=""
    while IFS= read -r p; do
      if [[ -z "$(git ls-tree -r "origin/$BRANCH" --name-only -- "$p")" ]]; then
        MISSING="$MISSING $p"
      fi
    done < <(find "$EVAL_DIR" -name '*.json' -type f; find "$FIG_DIR" -name '*.png' -type f)
    if [[ -n "$MISSING" ]]; then
      echo "FATAL: pushed tree missing declared result files:$MISSING" >&2; exit 86
    fi
    echo "[push] verified: 0 unpushed commits; all declared result files in origin/$BRANCH"
  fi
fi

# ---------------------------------------------------------------------------
# Results sentinel (poll_pipeline contract) + terminal [phase=done]
# ---------------------------------------------------------------------------
GPU_HOURS_USED=$(awk -v s="$SECONDS" -v g="$NGPU" 'BEGIN{printf "%.2f", (s/3600.0)*(g>0?g:1)}')
COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
SENTINEL_KIND="epm:results"
SENTINEL_PATH="$LOG_DIR/issue-1345-results.json"
if [[ -n "$SMOKE" || -n "$DRY_RUN" ]]; then
  SENTINEL_KIND="epm:smoke-result"
  SENTINEL_PATH="$LOG_DIR/issue-1345-smoke-results.json"
fi
uv run python - "$SENTINEL_KIND" "$SENTINEL_PATH" "$EVAL_DIR" "$COMMIT_SHA" "$GPU_HOURS_USED" "${SMOKE:-0}" "${NO_R3_FLAG:-none}" <<'PY'
import json
import sys
import time
from pathlib import Path

kind, out_path, eval_dir, commit_sha, gpu_hours, smoke, no_r3 = sys.argv[1:8]
eval_dir = Path(eval_dir)
eval_paths = sorted(str(p) for p in eval_dir.glob("*.json"))
lattice = {}
lat_path = eval_dir / "verdict_lattice.json"
if lat_path.exists():
    lat = json.loads(lat_path.read_text())
    lattice = {
        k: {"verdict": v.get("verdict"), "delta_xfer": v.get("delta_xfer"),
            "delta_reparam": v.get("delta_reparam")}
        for k, v in lat.get("per_model_arm", {}).items()
    }
parity = {}
par_path = eval_dir / "parity_gate.json"
if par_path.exists():
    parity = {"pass": json.loads(par_path.read_text()).get("pass")}
payload = {
    "eval_numbers": {"verdict_lattice": lattice, "parity_gate": parity,
                     "story_regime_halted": no_r3 != "none"},
    "eval_paths": eval_paths,
    "reproducibility_card": {
        "models": {"instruct": "Qwen/Qwen2.5-7B-Instruct", "pretrained": "Qwen/Qwen2.5-7B"},
        "seeds": {"fit": 0, "generation": 42, "subsample": 0},
        "pinned_parent_revision": "7159e5804d",
        "hf_data_prefix": "issue1345_smoke/" if smoke == "1" else "issue1345_framing/",
        "wandb_project": "n/a — no training (extraction + analysis task)",
    },
    "wandb_url": "n/a — no training runs (extraction/analysis only)",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": commit_sha,
    "gpu_hours_used": float(gpu_hours),
    "gpu_hours_budgeted": 14.0,
    "plan_deviations": [
        {"deviation": "HF store layout uses the canonical issue1345_framing/ prefix",
         "rationale": "plan §10 wrote a bare analysis_tensors/issue_1345 path; Upload Policy pins issueN_<slug>/ prefixes"},
        {"deviation": "pair-level matched-n implemented as row ALLOWLISTS on the single stores",
         "rationale": "equivalent selection, reuses run_cell's tested allowlist path, avoids duplicating ~90 GB of tensors"},
        {"deviation": "matched-capacity reparam nulls at L19 only (5 draws/type)",
         "rationale": "plan §3 defines Δ_reparam at layer 19; §9 sized ~16 extra alignment fits"},
        {"deviation": "story-pair aligned cosine = operator-spectrum Procrustes optimum",
         "rationale": "the map_alignment data-paired Procrustes is undefined for unpaired corpora (no shared conv_ids); raw-cosine rotation band + spectrum optimum reported instead"},
    ],
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "by": "issue1345-dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "task_id": 1345,
    "smoke": smoke == "1",
    "note": payload,
}
Path(out_path).write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out_path}")
PY

echo "[phase=done]"
