#!/usr/bin/env bash
# issue-825 onpolicy-turn-depth-map pod-side phase driver:
#   stage -> gen -> [gpu-guard] -> capture -> upload -> [phase=done].
# On the #1092 dynamics panel (497 logged WildChat/LMSYS multi-turn conversations,
# 2,572 assistant-turn pairs/model), generate the model's OWN turn-k answer under
# the real logged context (vLLM, plan §4.2 sampling), then teacher-force-capture
# prefix_k / context_k / answer_own_t1 at layers {14,18,19}. The single
# manipulated variable is turn-k answer provenance (own vs logged); the ridge
# fit + nulls + figures run OFF-POD (scripts/issue825_onpolicy_turn_depth_fit.py).
#
# Production (lane-agnostic): bash scripts/issue825_onpolicy_turn_depth_dispatch.sh
# Smoke (CPU, tiny):          bash scripts/issue825_onpolicy_turn_depth_dispatch.sh --smoke
#   (fabricates a tiny 4-conversation pfx_* corpus + a 28-layer tiny Qwen; runs
#    smoke_bank -> emit-banked-ref -> gen(canned) -> capture -> fit end-to-end
#    through the SAME entrypoints, then writes an epm:smoke-result sentinel.)
#
# Pod-side reporting contract (.claude/rules/pod-side-reporting.md): NO task.py
# shellouts; [phase=...] log lines with the single reserved terminal [phase=done];
# end-of-run sentinel /workspace/logs/issue-825-epm_results-<epoch>.json
# (smoke: issue-825-smoke-result-<epoch>.json, kind epm:smoke-result).
set -euo pipefail

# REPO_ROOT resolution robust under `set -u` on BOTH lanes (#825 crash-fix shape).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT" || {
  echo "FATAL: cd $REPO_ROOT failed" >&2
  exit 1
}

# Pinned data-repo revision (plan §10 W4) — resolves the consumed store paths
# byte-identically to what the logged turn-depth read downloaded.
DATA_REPO_REV="9dd650deef3ca21daa9cc2e940e9563edc000ba3"
HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue825_userbase_map"
HF_STORE_PREFIX="issue1092_realistic_crossing"
HF_SUMMARIES_PREFIX="$HF_STORE_PREFIX/analysis_tensors/summaries"

PHASE="all"
FROM_PHASE=""
SMOKE=""
TINY=""
DATA_ROOT="data/issue_825/onpolicy_turn_depth"
LOG_DIR="/workspace/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --from-phase) FROM_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="--smoke"; shift ;;
    --tiny-model-dir) TINY="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$SMOKE" ]]; then
  # Scratch-dir redirect: smoke outputs never touch production data/log paths.
  DATA_ROOT="${DATA_ROOT%/}_smoke"
  LOG_DIR="$DATA_ROOT/logs"
fi
CORPUS_DIR="$DATA_ROOT/corpus"
INDEX_DIR="$DATA_ROOT/banked_index"     # dynamics_{model}/row_index_*.jsonl (+ smoke bank npy)
CAPTURE_DIR="$DATA_ROOT/capture"        # {model}/{prefix_k,context_k,answer_own_t1}_L*.npy
ROLLOUT_DIR="$DATA_ROOT/raw_completions"
BANKED_JSON="eval_results/issue_825/turn_depth_map/results.json"
SMOKE_BANKED_JSON="$DATA_ROOT/smoke_banked_ref.json"
SMOKE_EVAL_DIR="$DATA_ROOT/eval"
SMOKE_FIG_DIR="$DATA_ROOT/figures"

# Conditional .env sourcing: pods carry a scp-pushed .env; the GCE lane exports
# tokens via instance metadata and has NO .env file (gotchas: incident #923).
if [ -f ./.env ]; then
  set -a
  source ./.env
  set +a
fi

PHASES=(stage gen capture upload)
mkdir -p "$DATA_ROOT" "$CORPUS_DIR" "$INDEX_DIR" "$CAPTURE_DIR" "$ROLLOUT_DIR" "$LOG_DIR"

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

gpu_guard() {
  # Reap any lingering VLLM::EngineCore before an HF load (vLLM teardown does
  # NOT reliably reap workers in-process — gotchas.md). No-op when nvidia-smi
  # is absent (CPU smoke).
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  for pid in $(pgrep -f '^VLLM::EngineCore' || true); do
    echo "[gpu-guard] killing lingering VLLM::EngineCore pid=$pid"
    kill -KILL "$pid" 2>/dev/null || true
  done
  for i in $(seq 1 24); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$apps" ]]; then return 0; fi
    echo "[gpu-guard] waiting for GPU to free ($i/24): $apps"
    sleep 5
  done
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "FATAL: GPU still held: $apps" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Smoke fixtures: tiny 28-layer Qwen2 (real tokenizer; verbatim recipe from
# issue825_naturalistic_s_dispatch.sh) + a tiny fabricated pfx_* corpus
# (BENIGN synthetic text — the real corpus is real-user text and is never
# fabricated from, printed, or inlined).
# ---------------------------------------------------------------------------
if [[ -n "$SMOKE" ]]; then
  TINY="${TINY:-$DATA_ROOT/tiny-qwen}"
  if [[ ! -d "$TINY" ]]; then
    echo "[smoke] fabricating tiny 28-layer Qwen2 (real tokenizer) -> $TINY"
    EPS_TINY_DIR="$TINY" uv run python - <<'PY'
import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

tiny = os.environ["EPS_TINY_DIR"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
cfg = Qwen2Config(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=28,  # capture layers 14/18/19 must be in range
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=max(max(tok.get_vocab().values()), max(tok.all_special_ids)) + 1,
    max_position_embeddings=4096,
)
torch.manual_seed(0)
model = Qwen2ForCausalLM(cfg)
model.save_pretrained(tiny)
tok.save_pretrained(tiny)
print(f"tiny model saved -> {tiny} (vocab {cfg.vocab_size})")
PY
  fi
fi

# ---------------------------------------------------------------------------
# stage: pinned prefix_store.jsonl + banked row-index shards (production) or the
# fabricated tiny corpus (smoke). Production also runs the artifact-reuse (j)
# pairwise-provenance coherence check at the pinned revision.
# ---------------------------------------------------------------------------
if should_run stage; then
  echo "[phase=stage]"
  if [[ -z "$SMOKE" ]]; then
    DATA_REPO="$HF_DATA_REPO" DATA_REPO_REV="$DATA_REPO_REV" \
      STORE_PREFIX="$HF_STORE_PREFIX" SUMMARIES_PREFIX="$HF_SUMMARIES_PREFIX" \
      CORPUS_DIR="$CORPUS_DIR" INDEX_DIR="$INDEX_DIR" uv run python - <<'PY'
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo = os.environ["DATA_REPO"]
rev = os.environ["DATA_REPO_REV"]
store_prefix = os.environ["STORE_PREFIX"]
summaries_prefix = os.environ["SUMMARIES_PREFIX"]
corpus_dir = Path(os.environ["CORPUS_DIR"])
index_dir = Path(os.environ["INDEX_DIR"])
api = HfApi()


def _fetch(path_in_repo: str, dest: Path) -> None:
    """hf_hub_download one file (pinned rev) with bounded transient retry."""
    last = None
    for attempt in range(4):
        try:
            got = Path(
                hf_hub_download(
                    repo, path_in_repo, repo_type="dataset", revision=rev,
                    local_dir=str(dest.parent),
                )
            )
            if got != dest and got.name == dest.name and got.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, dest)
            return
        except Exception as e:  # noqa: BLE001 — bounded retry, re-raise on exhaustion
            last = e
            print(f"[stage] retry {attempt + 1}/4 {path_in_repo}: {type(e).__name__}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"[stage] FAILED to fetch {path_in_repo} after 4 attempts") from last


# (j) pairwise provenance coherence at the pinned revision: the corpus file's
# last commit must PREDATE both dynamics captures' (a corpus regenerated after
# the capture is incoherent regardless of sha pins — artifact-reuse.md).
paths = [
    f"{store_prefix}/corpus/prefix_store.jsonl",
    f"{summaries_prefix}/dynamics_instruct",
    f"{summaries_prefix}/dynamics_pretrained",
]
info = {
    i.path: i.last_commit.date
    for i in api.get_paths_info(repo, paths=paths, repo_type="dataset", revision=rev, expand=True)
    if i.last_commit is not None
}
missing = [p for p in paths if p not in info]
assert not missing, f"[stage] get_paths_info missing lastCommit for: {missing}"
corpus_date = info[paths[0]]
for p in paths[1:]:
    assert corpus_date <= info[p], (
        f"[stage] PROVENANCE INCOHERENT: corpus {corpus_date} POSTDATES capture "
        f"{p} {info[p]} — re-pin to the capture-era revision (plan §10 item j)"
    )
print(f"[stage] provenance coherent: corpus {corpus_date} <= captures "
      f"{[str(info[p]) for p in paths[1:]]}")

# 1) prefix_store.jsonl (pinned) -> flat local path.
_fetch(f"{store_prefix}/corpus/prefix_store.jsonl", corpus_dir / "prefix_store.jsonl")
n_bytes = (corpus_dir / "prefix_store.jsonl").stat().st_size
n_rows = sum(
    1 for line in (corpus_dir / "prefix_store.jsonl").open(encoding="utf-8") if line.strip()
)
print(f"[stage] prefix_store.jsonl staged: {n_rows} rows, {n_bytes} bytes @ rev {rev[:12]}")

# 2) banked row-index shards for BOTH models (server-scoped list_repo_tree —
#    never snapshot_download the ~1M-file data repo here; small JSONLs).
for mt in ("dynamics_instruct", "dynamics_pretrained"):
    tree = list(
        api.list_repo_tree(
            repo, path_in_repo=f"{summaries_prefix}/{mt}", repo_type="dataset",
            recursive=False, revision=rev,
        )
    )
    wanted = sorted(
        e.path
        for e in tree
        if e.path.split("/")[-1].startswith(("row_index_context_k", "row_index_answer_k_t1"))
        and e.path.endswith(".jsonl")
    )
    assert wanted, f"[stage] no row-index shards under {summaries_prefix}/{mt} @ {rev[:12]}"
    for p in wanted:
        _fetch(p, index_dir / mt / p.split("/")[-1])
    print(f"[stage] {mt}: {len(wanted)} row-index shards -> {index_dir / mt}")
PY
  else
    echo "[stage] smoke: fabricating tiny pfx_* corpus (benign synthetic text)"
    CORPUS_DIR="$CORPUS_DIR" uv run python - <<'PY'
import json
import os
from pathlib import Path

corpus_dir = Path(os.environ["CORPUS_DIR"])
corpus_dir.mkdir(parents=True, exist_ok=True)
TOPICS = ["gardens", "bridges", "libraries", "harbors"]
rows = []
# 4 conversations, 2-6 turns (user at even idx, assistant at odd): assistant
# turns t1 (n=4), t3 (n=3), t5 (n=1) — enough for grouped folds at t1/t3.
n_turns = {"pfx_smoke_a": 2, "pfx_smoke_b": 4, "pfx_smoke_c": 4, "pfx_smoke_d": 6}
for ci, (conv, nt) in enumerate(sorted(n_turns.items())):
    turns = []
    for t in range(nt):
        if t % 2 == 0:
            turns.append(
                {
                    "role": "user",
                    "content": f"Question {t // 2 + 1} about {TOPICS[ci]}: what changed "
                    f"between decade {t + 1} and decade {t + 2}?",
                }
            )
        else:
            turns.append(
                {
                    "role": "assistant",
                    "content": f"Logged answer {t // 2 + 1}: the {TOPICS[ci]} grew in "
                    f"period {t}, with several documented renovations.",
                }
            )
    rows.append({"prefix_id": conv, "conv_id": conv, "prefix_turns": turns})
with (corpus_dir / "prefix_store.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[stage] smoke corpus: {len(rows)} conversations -> {corpus_dir / 'prefix_store.jsonl'}")
PY
    echo "[phase=smoke_bank]"
    for m in instruct pretrained; do
      uv run python scripts/issue825_onpolicy_turn_depth_gpu.py \
        --model "$m" --phases smoke_bank --corpus-dir "$CORPUS_DIR" \
        --row-index-dir "$INDEX_DIR" --out-dir "$CAPTURE_DIR" \
        --rollout-dir "$ROLLOUT_DIR" --bank-dir "$INDEX_DIR" \
        --smoke --tiny-model-dir "$TINY" \
        >"$LOG_DIR/issue-825-otd-smokebank-$m.log" 2>&1 || {
        echo "FATAL: smoke_bank $m failed" >&2
        tail -40 "$LOG_DIR/issue-825-otd-smokebank-$m.log" >&2 || true
        exit 1
      }
    done
    echo "[stage] smoke: emitting tiny banked reference (ctx_logged real curve)"
    uv run python scripts/issue825_onpolicy_turn_depth_fit.py --emit-banked-ref \
      --banked-local-root "$INDEX_DIR" --banked-json "$SMOKE_BANKED_JSON" \
      >"$LOG_DIR/issue-825-otd-emitbank.log" 2>&1 || {
      echo "FATAL: emit-banked-ref failed" >&2
      tail -40 "$LOG_DIR/issue-825-otd-emitbank.log" >&2 || true
      exit 1
    }
  fi
fi

if [[ -n "$SMOKE" ]]; then
  ACTIVE_BANKED_JSON="$SMOKE_BANKED_JSON"
else
  ACTIVE_BANKED_JSON="$BANKED_JSON"
fi

run_worker() {
  local phases="$1" m="$2" cvd="$3"
  local log="$LOG_DIR/issue-825-otd-${phases//,/}-${m}.log"
  local extra=()
  if [[ -n "$SMOKE" ]]; then extra=(--smoke --tiny-model-dir "$TINY"); fi
  if [[ -n "$cvd" ]]; then
    env CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/issue825_onpolicy_turn_depth_gpu.py \
      --model "$m" --phases "$phases" --corpus-dir "$CORPUS_DIR" \
      --row-index-dir "$INDEX_DIR" --out-dir "$CAPTURE_DIR" \
      --rollout-dir "$ROLLOUT_DIR" --banked-json "$ACTIVE_BANKED_JSON" \
      "${extra[@]}" >"$log" 2>&1
  else
    uv run python scripts/issue825_onpolicy_turn_depth_gpu.py \
      --model "$m" --phases "$phases" --corpus-dir "$CORPUS_DIR" \
      --row-index-dir "$INDEX_DIR" --out-dir "$CAPTURE_DIR" \
      --rollout-dir "$ROLLOUT_DIR" --banked-json "$ACTIVE_BANKED_JSON" \
      "${extra[@]}" >"$log" 2>&1
  fi
}

run_phase_both_models() {
  # instruct on GPU0 || pretrained on GPU1 when >=2 GPUs are visible (CVD pinned
  # in the LAUNCHER env per model — the naturalistic run_extract pattern; the
  # in-process clobber is silently defeated by import-time cuInit, #545).
  local phases="$1"
  local ngpu=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  echo "[$phases] visible GPUs: $ngpu"
  if [[ "$ngpu" -ge 2 && -z "$SMOKE" ]]; then
    run_worker "$phases" instruct 0 &
    local pid_i=$!
    run_worker "$phases" pretrained 1 &
    local pid_p=$!
    local rc_i=0 rc_p=0
    wait "$pid_i" || rc_i=$?
    wait "$pid_p" || rc_p=$?
    for m in instruct pretrained; do
      local rc_var
      if [[ "$m" == "instruct" ]]; then rc_var=$rc_i; else rc_var=$rc_p; fi
      if [[ "$rc_var" -ne 0 ]]; then
        echo "FATAL: $phases $m rc=$rc_var" >&2
        tail -40 "$LOG_DIR/issue-825-otd-${phases//,/}-${m}.log" >&2 || true
        exit 1
      fi
    done
  else
    for m in instruct pretrained; do
      run_worker "$phases" "$m" "" || {
        echo "FATAL: $phases $m failed" >&2
        tail -40 "$LOG_DIR/issue-825-otd-${phases//,/}-${m}.log" >&2 || true
        exit 1
      }
    done
  fi
}

# ---------------------------------------------------------------------------
# gen: on-policy turn-k answers (vLLM chunked; canned via the same code path in
# smoke). Rollout JSONL is the checkpoint artifact (written before capture).
# ---------------------------------------------------------------------------
if should_run gen; then
  echo "[phase=gen]"
  gpu_guard
  run_phase_both_models gen
  echo "[gen] rollout checkpoints:"
  ls "$ROLLOUT_DIR" | sed 's/^/[gen]   /' || true
fi

# ---------------------------------------------------------------------------
# capture: teacher-forced states of the truncated own-answer renders.
# gpu-guard first: the vLLM engine must be fully torn down before the HF load.
# ---------------------------------------------------------------------------
if should_run capture; then
  echo "[phase=capture]"
  gpu_guard
  run_phase_both_models capture
  echo "[capture] outputs:"
  ls "$CAPTURE_DIR"/*/ | sed 's/^/[capture]   /' || true
fi

# ---------------------------------------------------------------------------
# smoke-only: VM fit end-to-end on the tiny capture (gates G1/G2 + fits + nulls
# + paired bootstrap + figures), against the fabricated bank.
# ---------------------------------------------------------------------------
if [[ -n "$SMOKE" ]]; then
  echo "[phase=fit_smoke]"
  uv run python scripts/issue825_onpolicy_turn_depth_fit.py --smoke \
    --capture-dir "$CAPTURE_DIR" --banked-local-root "$INDEX_DIR" \
    --banked-json "$SMOKE_BANKED_JSON" \
    --out-json "$SMOKE_EVAL_DIR/results.json" --fig-dir "$SMOKE_FIG_DIR" \
    --n-draws 8 --n-boot 25 --null-min-n 3 --pilot-n 4 \
    >"$LOG_DIR/issue-825-otd-fitsmoke.log" 2>&1 || {
    echo "FATAL: smoke fit failed" >&2
    tail -60 "$LOG_DIR/issue-825-otd-fitsmoke.log" >&2 || true
    exit 1
  }
  grep -E "\[G1\]|\[G2\]|\[write\]" "$LOG_DIR/issue-825-otd-fitsmoke.log" | sed 's/^/[fit_smoke] /'
fi

# ---------------------------------------------------------------------------
# upload: <=3 bulk upload_folder commits (429-aware bounded outer backoff), then
# a scoped Hub listing verification, the results sentinel, and [phase=done].
# ---------------------------------------------------------------------------
if should_run upload; then
  echo "[phase=upload]"
  if [[ -z "$SMOKE" ]]; then
    DATA_REPO="$HF_DATA_REPO" HF_PREFIX="$HF_PREFIX" \
      ROLLOUT_DIR="$ROLLOUT_DIR" CAPTURE_DIR="$CAPTURE_DIR" uv run python - <<'PY'
import os
import random
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi, upload_folder  # noqa: E402

repo = os.environ["DATA_REPO"]
hf_prefix = os.environ["HF_PREFIX"]
rollout_dir = Path(os.environ["ROLLOUT_DIR"])
capture_dir = Path(os.environ["CAPTURE_DIR"])
raw_prefix = f"{hf_prefix}/raw_completions/onpolicy_turn_depth"
tensors_prefix = f"{hf_prefix}/analysis_tensors/onpolicy_turn_depth"


def _with_backoff(label: str, fn) -> None:
    """Bounded outer retry (5 attempts, 60s*2^k + jitter); 429 is retryable,
    never fatal (huggingface_hub retries internally first)."""
    last = None
    for attempt in range(5):
        try:
            fn()
            return
        except Exception as e:  # noqa: BLE001 — bounded retry, re-raise on exhaustion
            last = e
            wait = 60 * (2**attempt) + random.uniform(0, 15)
            print(f"[upload] {label} attempt {attempt + 1}/5 failed "
                  f"({type(e).__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"[upload] {label} FAILED after 5 attempts") from last


# Commit 1: rollout text (the regeneration root — unconditional, non-LFS JSONL).
_with_backoff(
    "raw_completions",
    lambda: upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(rollout_dir),
        path_in_repo=raw_prefix, allow_patterns=["*_own_turn_answers.jsonl"],
        commit_message="issue-825 onpolicy-turn-depth: on-policy turn-k rollout text",
    ),
)
# Commit 2: capture tensors + row indexes + drop reports.
_with_backoff(
    "analysis_tensors",
    lambda: upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(capture_dir),
        path_in_repo=tensors_prefix,
        allow_patterns=["*/*.npy", "*/row_index_own.jsonl", "*/drop_report.json"],
        commit_message="issue-825 onpolicy-turn-depth: capture npy + row indexes + drop reports",
    ),
)

# Verify on a FRESH scoped listing (never bare list_repo_files on the ~1M-file repo).
api = HfApi()
local_jsonl = sorted(p.name for p in rollout_dir.glob("*_own_turn_answers.jsonl"))
hub_raw = {
    e.path.split("/")[-1]
    for e in api.list_repo_tree(repo, path_in_repo=raw_prefix, repo_type="dataset",
                                recursive=True)
}
missing = [f for f in local_jsonl if f not in hub_raw]
assert not missing, f"[upload] raw completions missing on Hub: {missing}"
local_tensors = sorted(
    str(p.relative_to(capture_dir))
    for p in capture_dir.rglob("*")
    if p.is_file() and (p.suffix in (".npy",) or p.name in ("row_index_own.jsonl",
                                                            "drop_report.json"))
)
hub_tensors = {
    e.path.removeprefix(tensors_prefix + "/")
    for e in api.list_repo_tree(repo, path_in_repo=tensors_prefix, repo_type="dataset",
                                recursive=True)
}
missing_t = [f for f in local_tensors if f not in hub_tensors]
assert not missing_t, f"[upload] capture tensors missing on Hub: {missing_t}"
print(f"[upload] verified: {len(local_jsonl)} rollout JSONLs @ {raw_prefix}; "
      f"{len(local_tensors)} capture files @ {tensors_prefix}")
PY
  else
    echo "[upload] smoke mode: skipping HF uploads"
  fi

  GPU_HOURS_USED="${GPU_HOURS_USED:-0.0}"
  COMMIT_SHA=$(git rev-parse HEAD)
  DATA_REPO_REV="$DATA_REPO_REV" HF_PREFIX="$HF_PREFIX" ROLLOUT_DIR="$ROLLOUT_DIR" \
    CAPTURE_DIR="$CAPTURE_DIR" LOG_DIR="$LOG_DIR" SMOKE="$SMOKE" \
    uv run python - "$COMMIT_SHA" "$GPU_HOURS_USED" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from issue1092_gpu_phase import (  # noqa: E402
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    PRETRAINED_MODEL,
    PRETRAINED_REVISION,
    STOP_TOKENS_INSTRUCT,
    STOP_TOKENS_PRETRAINED,
)

commit_sha, gpu_hours = sys.argv[1], float(sys.argv[2])
capture_dir = Path(os.environ["CAPTURE_DIR"])
rollout_dir = Path(os.environ["ROLLOUT_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
data_rev = os.environ["DATA_REPO_REV"]
hf_prefix = os.environ["HF_PREFIX"]
smoke = bool(os.environ.get("SMOKE"))

per_model = {}
for mt in ("instruct", "pretrained"):
    dr_path = capture_dir / mt / "drop_report.json"
    if not dr_path.exists():
        continue
    dr = json.loads(dr_path.read_text())
    per_model[mt] = {
        "n_total_pairs": dr["n_total_pairs"],
        "n_kept": dr["n_kept"],
        "drop_rate": dr["drop_rate"],
        "drop_counts": dr["drop_counts"],
    }

note = {
    "eval_numbers": {
        "phase": "gen+capture complete (pod side); ridge fits + nulls + figures run "
        "OFF-POD via scripts/issue825_onpolicy_turn_depth_fit.py",
        "per_model": per_model,
    },
    "eval_paths": [
        "eval_results/issue_825/onpolicy_turn_depth/results.json (written by the VM fit)"
    ],
    "reproducibility_card": {
        "followup_label": "onpolicy-turn-depth-map",
        "models": {
            "instruct": f"{INSTRUCT_MODEL}@{INSTRUCT_REVISION}",
            "pretrained": f"{PRETRAINED_MODEL}@{PRETRAINED_REVISION}",
        },
        "sampling": {
            "recipe": "SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, "
            "seed=42) — the #825 single-turn anchor construction "
            "(issue825_gen_conversations.py:521)",
            "stop_tokens": {
                "instruct": STOP_TOKENS_INSTRUCT,
                "pretrained": STOP_TOKENS_PRETRAINED,
            },
            "engine_max_model_len": 9216,
        },
        "seeds": {"generation": 42, "null": 1092, "fold": 0, "bootstrap": 8250},
        "capture": {"window_tokens": 8192, "layers": [14, 18, 19], "dtype": "fp16"},
        "input_data": {
            "prefix_store": "issue1092_realistic_crossing/corpus/prefix_store.jsonl",
            "banked_row_indexes": "issue1092_realistic_crossing/analysis_tensors/summaries/"
            "dynamics_{instruct,pretrained}/row_index_*.jsonl",
            "data_repo_revision": data_rev,
        },
        "outputs": {
            "rollout_text": f"{hf_prefix}/raw_completions/onpolicy_turn_depth/",
            "capture_tensors": f"{hf_prefix}/analysis_tensors/onpolicy_turn_depth/",
        },
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "wandb_project": "n/a — no training (generation + capture + analysis round)",
    },
    "wandb_url": "n/a — analysis round, no training",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": commit_sha,
    "gpu_hours_used": gpu_hours,
    "gpu_hours_budgeted": 2.0,
    "plan_deviations": [],
}
epoch = int(time.time())
kind = "epm:smoke-result" if smoke else "epm:results"
slug = "smoke-result" if smoke else "epm_results"
sentinel = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 825,
    "by": "issue825_onpolicy_turn_depth_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "smoke": smoke,
    "note": note,
}
out = log_dir / f"issue-825-{slug}-{epoch}.json"
out.write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out}")
PY

  if [[ -n "$SMOKE" ]]; then
    # Smoke asserts the sentinel parses under the poller's own contract.
    LOG_DIR="$LOG_DIR" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from poll_pipeline import _parse_sentinel  # noqa: E402

log_dir = Path(os.environ["LOG_DIR"])
paths = sorted(log_dir.glob("issue-825-smoke-result-*.json"))
assert paths, f"no smoke sentinel under {log_dir}"
body = paths[-1].read_text()
parsed = _parse_sentinel(str(paths[-1]), body)
assert parsed is not None, "poller _parse_sentinel REJECTED the smoke sentinel"
print(f"[sentinel-check] _parse_sentinel OK: kind={parsed['kind']} "
      f"version={parsed['version']} ({paths[-1].name})")
PY
  fi
fi

echo "[phase=done]"
