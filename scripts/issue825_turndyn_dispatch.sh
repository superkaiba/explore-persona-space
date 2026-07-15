#!/usr/bin/env bash
# issue-825 turn-dynamics-allturns-5000 pod-side phase driver (plan v24).
#
#   p0     harvest (CPU cpu-mid lane): full-stream WildChat+LMSYS scan (gate
#          G-A), panel/seed build, gc panel rebuild, HF panel upload.
#   stage  pod: fetch the P0 panel outputs from HF (own upload -> rev main).
#   pilot  gate G-B: 50-seed PRODUCTION-path rollout pilot, both subject arms,
#          with the pre-registered K_gen fallback 24->20->16.
#   main   work-conserving 8-GPU job queue: arm-G rollout (3 GPUs/arm,
#          double-buffered vs Haiku) || arm-R gen (1 GPU/model) -> captures
#          (armR own / armR logged+gc / armG) -> per-(arm, model) incremental
#          uploads -> fit phases (cells/transfer/operators/reach/gc) ->
#          assemble. No GPU idles while an unblocked job is queued.
#   upload betas + fit parts + results JSON + sentinel.
#
# Production: bash scripts/issue825_turndyn_dispatch.sh --phase all
# Smoke (CPU, tiny): bash scripts/issue825_turndyn_dispatch.sh --smoke
#   (fabricates a tiny benign panel + 28-layer tiny Qwen; runs pilot ->
#    rollout(LIVE tiny Haiku through api_dispatch) -> gen(canned) -> captures
#    -> all fit phases -> figures through the SAME entrypoints; scratch roots.)
#
# Pod-side reporting (.claude/rules/pod-side-reporting.md): NO task.py
# shellouts; [phase=...] lines with the single reserved terminal [phase=done];
# end-of-run sentinel /workspace/logs/issue-825-epm_results-<epoch>.json
# (smoke: issue-825-smoke-result-<epoch>.json, kind epm:smoke-result).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT" || { echo "FATAL: cd $REPO_ROOT failed" >&2; exit 1; }

HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue825_userbase_map"
HF_PANEL_PREFIX="$HF_PREFIX/analysis_tensors/turn_dynamics/panel"
HF_RAW_PREFIX="$HF_PREFIX/raw_completions/turn_dynamics"
HF_TENSORS_PREFIX="$HF_PREFIX/analysis_tensors/turn_dynamics"

PHASE="all"
SMOKE=""
TINY=""
DATA_ROOT="data/issue_825/turn_dynamics"
LOG_DIR="/workspace/logs"
K_GEN=24
PANEL_N=5000
SPARE_N=1000
SWEEP_N_G=1000
SWEEP_N_R=500
PILOT_N=50
STREAM_LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="--smoke"; shift ;;
    --tiny-model-dir) TINY="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --k-gen) K_GEN="$2"; shift 2 ;;
    --panel-n) PANEL_N="$2"; shift 2 ;;
    --pilot-n) PILOT_N="$2"; shift 2 ;;
    --stream-limit) STREAM_LIMIT="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$SMOKE" ]]; then
  # Scratch-dir redirect: smoke outputs never touch production data/log paths.
  DATA_ROOT="${DATA_ROOT%/}_smoke"
  LOG_DIR="$DATA_ROOT/logs"
  K_GEN=3
  PANEL_N=4
  PILOT_N=2
  SWEEP_N_G=2
  SWEEP_N_R=2
fi
PANEL_DIR="$DATA_ROOT/panel"
ROLLOUT_DIR="$DATA_ROOT/rollouts"
PILOT_DIR="$DATA_ROOT/pilot"
GEN_DIR="$DATA_ROOT/gen"
CAPTURE_DIR="$DATA_ROOT/capture"
PARTS_DIR="$DATA_ROOT/fit_parts"
BETAS_DIR="$DATA_ROOT/betas"
STATE_DIR="$DATA_ROOT/state"
EVAL_DIR="eval_results/issue_825/turn_dynamics"
FIG_DIR="figures/issue_825"
if [[ -n "$SMOKE" ]]; then
  EVAL_DIR="$DATA_ROOT/eval"
  FIG_DIR="$DATA_ROOT/figures"
fi
RESULTS_JSON="$EVAL_DIR/results.json"
GB_GATE_JSON="$DATA_ROOT/gb_gate.json"

# Conditional .env sourcing: pods carry a scp-pushed .env; the GCE lane exports
# tokens via instance metadata and has NO .env file (gotchas, incident #923).
if [ -f ./.env ]; then set -a; source ./.env; set +a; fi

mkdir -p "$DATA_ROOT" "$PANEL_DIR" "$ROLLOUT_DIR" "$PILOT_DIR" "$GEN_DIR" \
  "$CAPTURE_DIR" "$PARTS_DIR" "$BETAS_DIR" "$STATE_DIR" "$LOG_DIR"

MODELS=(instruct pretrained)

n_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

gpu_guard_one() {
  # Per-GPU orphan drain before the next job on a queue worker's device
  # (CVD-aware via -i <physical index> — the #396 gpu_uuid-scoping rule).
  # After ~60s of a held device, reap orphaned VLLM::EngineCore pids by exact
  # PID (vLLM teardown does NOT reliably reap workers in-process; gotchas.md).
  local gpu="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  for i in $(seq 1 24); do
    local apps
    apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
    [[ -z "$apps" ]] && return 0
    if [[ "$i" -eq 12 ]]; then
      local pid
      for pid in $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
        if grep -aq '^VLLM::EngineCore' "/proc/$pid/cmdline" 2>/dev/null; then
          echo "[gpu-guard g$gpu] killing orphaned VLLM::EngineCore pid=$pid"
          kill -KILL "$pid" 2>/dev/null || true
        fi
      done
    fi
    echo "[gpu-guard g$gpu] waiting for GPU to free ($i/24): $apps"
    sleep 5
  done
  echo "FATAL: GPU $gpu still held" >&2
  return 1
}

write_sentinel() {
  local kind_note="$1"
  GPU_HOURS_USED="${GPU_HOURS_USED:-0.0}" COMMIT_SHA=$(git rev-parse HEAD) \
  SMOKE="$SMOKE" LOG_DIR="$LOG_DIR" RESULTS_JSON="$RESULTS_JSON" \
  DATA_ROOT="$DATA_ROOT" KIND_NOTE="$kind_note" GB_GATE_JSON="$GB_GATE_JSON" \
    uv run python - <<'PY'
import json
import os
import time
from pathlib import Path

smoke = bool(os.environ.get("SMOKE"))
log_dir = Path(os.environ["LOG_DIR"])
results_json = Path(os.environ["RESULTS_JSON"])
gb_path = Path(os.environ["GB_GATE_JSON"])
summary = {}
if results_json.exists():
    with open(results_json) as f:
        res = json.load(f)
    summary = {
        "gates": res.get("gates"),
        "harvest": res.get("harvest_report_digest"),
        "n_parts": len(res.get("parts", {})),
    }
gb = json.loads(gb_path.read_text()) if gb_path.exists() else None
note = {
    "eval_numbers": {"phase_summary": summary, "gate_gb": gb},
    "eval_paths": ["eval_results/issue_825/turn_dynamics/results.json"],
    "reproducibility_card": {
        "followup_label": "turn-dynamics-allturns-5000",
        "sampling": "SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)",
        "capture_windows": {"armR": 8192, "armG": 15872},
        "layers": [14, 18, 19],
        "seeds": {"panel": 42, "generation": 42, "fold": 0, "null": 1092, "bootstrap": 8250},
        "outputs": {
            "rollout_text": os.environ["KIND_NOTE"] and "issue825_userbase_map/raw_completions/turn_dynamics/",
            "capture_tensors": "issue825_userbase_map/analysis_tensors/turn_dynamics/",
            "betas": "issue825_userbase_map/analysis_tensors/turn_dynamics/betas/",
        },
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "wandb_project": "n/a — no training (generation + capture + analysis round)",
    },
    "wandb_url": "n/a — analysis round, no training",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": os.environ["COMMIT_SHA"],
    "gpu_hours_used": float(os.environ["GPU_HOURS_USED"]),
    "gpu_hours_budgeted": 68.0,
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
    "by": "issue825_turndyn_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "smoke": smoke,
    "note": note,
}
out = log_dir / f"issue-825-{slug}-{epoch}.json"
out.write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out}")
PY
}

# ---------------------------------------------------------------------------
# smoke fixtures: tiny 28-layer Qwen (real tokenizer) + benign tiny panel
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
    num_hidden_layers=28,
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
  if [[ ! -f "$PANEL_DIR/harvest_report.json" ]]; then
    echo "[smoke] fabricating tiny benign panel/seeds/gc (synthetic text only)"
    PANEL_DIR="$PANEL_DIR" PANEL_N="$PANEL_N" uv run python - <<'PY'
import json
import os
from pathlib import Path

panel_dir = Path(os.environ["PANEL_DIR"])
panel_n = int(os.environ["PANEL_N"])
TOPICS = ["gardens", "bridges", "libraries", "harbors", "orchards", "canals"]
K_REAL = 2


def _conv(cid: str, topic: str, n_user: int) -> dict:
    turns = []
    for t in range(n_user):
        turns.append(
            {
                "role": "user",
                "content": f"Question {t + 1} about {topic}: what changed between "
                f"decade {t + 1} and decade {t + 2}?",
            }
        )
        turns.append(
            {
                "role": "assistant",
                "content": f"Logged answer {t + 1}: the {topic} grew in period {t + 1}, "
                f"with several documented renovations and a new annex.",
            }
        )
    return {"id": cid, "conv_id": cid, "turns": turns, "n_user_turns": n_user}


rows = [_conv(f"smoke_{i:03d}", TOPICS[i % len(TOPICS)], 3) for i in range(panel_n + 2)]
panel = rows[:panel_n]


def _shard(stem: str, items: list[dict]) -> None:
    with (panel_dir / f"{stem}_shard000.jsonl").open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")


_shard("panel_armR", panel)
_shard("deep_pool", rows)
seeds = []
for i, r in enumerate(rows[: panel_n + 2]):
    u1 = next(t["content"] for t in r["turns"] if t["role"] == "user")
    seeds.append(
        {
            "conv_id": r["conv_id"],
            "seed_rank": i,
            "in_panel": i < panel_n,
            "u1": u1,
            "brief_id": f"brief_{i % 3}",
            "brief_text": "You are a curious, practical person planning a small local project; "
            "you ask short concrete follow-up questions.",
        }
    )
_shard("armG_seeds", seeds)
_shard("gc_panel", [{"conv_id": r["conv_id"], "turns": r["turns"]} for r in panel])
report = {
    "issue": 825,
    "phase": "P0-harvest (SMOKE fabricated)",
    "K_real": K_REAL,
    "panel_n": len(panel),
    "n_seeds": len(seeds),
    "gate_ga": {"gate": "G-A", "K_real": K_REAL, "pass": True, "smoke_fabricated": True},
    "nk_table": {str(k): max(0, len(rows) - k) for k in range(1, 8)},
    "panel_ids_sha256": "smoke",
}
(panel_dir / "harvest_report.json").write_text(json.dumps(report, indent=1))
(panel_dir / "nk_table.json").write_text(json.dumps(report["nk_table"], indent=1))
print(f"[smoke] panel fabricated: {len(panel)} convs, {len(seeds)} seeds")
PY
  fi
fi

SMOKE_ARGS=()
if [[ -n "$SMOKE" ]]; then SMOKE_ARGS=(--smoke --tiny-model-dir "$TINY"); fi
FIT_DEVICE="cuda"
if [[ -n "$SMOKE" || "$(n_gpus)" -eq 0 ]]; then FIT_DEVICE="cpu"; fi
FIT_SMOKE_ARGS=()
if [[ -n "$SMOKE" ]]; then
  FIT_SMOKE_ARGS=(--min-fit-n 3 --null-min-n 3 --n-draws 4 --n-boot 20 \
    --mlp-conv-n 4 --mlp-input-pca 4 --mlp-null-draws 1 --smoke)
fi

# ---------------------------------------------------------------------------
# p0: harvest (CPU lane; checkpoint/resume inside _stream_with_cache)
# ---------------------------------------------------------------------------
if [[ "$PHASE" == "p0" ]]; then
  echo "[phase=p0_harvest]"
  EXTRA=()
  if [[ -n "$SMOKE" ]]; then
    EXTRA=(--stream-limit "${STREAM_LIMIT:-3000}" --ga-target 2 --ga-kmin 2 --ga-kmax 6 \
      --panel-n "$PANEL_N" --spare-n 2 --skip-gc --skip-upload --smoke)
  elif [[ "$STREAM_LIMIT" != "0" ]]; then
    EXTRA=(--stream-limit "$STREAM_LIMIT")
  fi
  uv run python scripts/issue825_turndyn_harvest.py \
    --out-dir "$PANEL_DIR" --panel-n "$PANEL_N" --spare-n "$SPARE_N" \
    "${EXTRA[@]}" 2>&1 | tee "$LOG_DIR/issue-825-turndyn-p0.log" | grep -E '\[G-A\]|\[gc\]|\[harvest\]|\[upload\]|DONE' || {
    echo "FATAL: harvest failed" >&2
    tail -40 "$LOG_DIR/issue-825-turndyn-p0.log" >&2 || true
    exit 3
  }
  echo "[phase=done]"
  exit 0
fi

# ---------------------------------------------------------------------------
# stage: fetch P0 panel outputs from HF (own upload -> revision main)
# ---------------------------------------------------------------------------
stage_panel() {
  echo "[phase=stage]"
  if [[ -n "$SMOKE" ]]; then
    echo "[stage] smoke: fabricated panel already in place"
    return 0
  fi
  if [[ -f "$PANEL_DIR/harvest_report.json" ]]; then
    echo "[stage] panel already staged: $PANEL_DIR"
    return 0
  fi
  DATA_REPO="$HF_DATA_REPO" PANEL_PREFIX="$HF_PANEL_PREFIX" PANEL_DIR="$PANEL_DIR" \
    uv run python - <<'PY'
import os
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

repo = os.environ["DATA_REPO"]
prefix = os.environ["PANEL_PREFIX"]
dest = Path(os.environ["PANEL_DIR"])
api = HfApi()
# server-side scoped listing — NEVER snapshot_download the ~1M-file repo (gotchas)
tree = list(api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset", recursive=True))
files = [e.path for e in tree if not e.path.endswith("/")]
assert files, f"no panel files under {prefix} — run --phase p0 first"
for pth in files:
    rel = pth.removeprefix(prefix + "/")
    last = None
    for attempt in range(4):
        try:
            got = Path(
                hf_hub_download(repo, pth, repo_type="dataset", revision="main", local_dir=str(dest))
            )
            tgt = dest / rel
            if got.resolve() != tgt.resolve():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, tgt)
            last = None
            break
        except Exception as e:  # noqa: BLE001 — bounded retry
            last = e
            print(f"[stage] retry {attempt + 1}/4 {pth}: {type(e).__name__}")
            time.sleep(20 * (attempt + 1))
    if last is not None:
        raise RuntimeError(f"[stage] FAILED to fetch {pth}") from last
print(f"[stage] {len(files)} panel files -> {dest}")
PY
}

# ---------------------------------------------------------------------------
# pilot: gate G-B (PRODUCTION rollout path at --pilot-n; K_gen fallback ladder)
# ---------------------------------------------------------------------------
run_pilot() {
  echo "[phase=pilot]"
  local kgen_try
  for kgen_try in "$K_GEN" 20 16; do
    [[ "$kgen_try" -gt "$K_GEN" ]] && continue
    local pdir="$PILOT_DIR/k$kgen_try"
    mkdir -p "$pdir"
    echo "[pilot] K_gen=$kgen_try, n=$PILOT_N seeds, both subject arms"
    local m rc=0
    local ng
    ng=$(n_gpus)
    if [[ "$ng" -ge 2 && -z "$SMOKE" ]]; then
      CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue825_turndyn_rollout.py \
        --model instruct --seeds-dir "$PANEL_DIR" --out-dir "$pdir" \
        --k-gen "$kgen_try" --pilot-n "$PILOT_N" "${SMOKE_ARGS[@]}" \
        >"$LOG_DIR/issue-825-turndyn-pilot-instruct.log" 2>&1 &
      local pid_i=$!
      CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue825_turndyn_rollout.py \
        --model pretrained --seeds-dir "$PANEL_DIR" --out-dir "$pdir" \
        --k-gen "$kgen_try" --pilot-n "$PILOT_N" "${SMOKE_ARGS[@]}" \
        >"$LOG_DIR/issue-825-turndyn-pilot-pretrained.log" 2>&1 &
      local pid_p=$!
      wait "$pid_i" || rc=$?
      wait "$pid_p" || rc=$?
    else
      for m in "${MODELS[@]}"; do
        uv run python scripts/issue825_turndyn_rollout.py \
          --model "$m" --seeds-dir "$PANEL_DIR" --out-dir "$pdir" \
          --k-gen "$kgen_try" --pilot-n "$PILOT_N" "${SMOKE_ARGS[@]}" \
          >"$LOG_DIR/issue-825-turndyn-pilot-$m.log" 2>&1 || rc=$?
      done
    fi
    if [[ "$rc" -ne 0 ]]; then
      echo "FATAL: pilot rollout failed (K_gen=$kgen_try)" >&2
      tail -40 "$LOG_DIR/issue-825-turndyn-pilot-instruct.log" >&2 || true
      exit 4
    fi
    for m in "${MODELS[@]}"; do
      uv run python scripts/issue825_turndyn_rollout.py \
        --model "$m" --seeds-dir "$PANEL_DIR" --out-dir "$pdir" \
        --k-gen "$kgen_try" --pilot-n "$PILOT_N" --report "${SMOKE_ARGS[@]}" \
        >>"$LOG_DIR/issue-825-turndyn-pilot-$m.log" 2>&1
    done
    # gate check: completion >= 0.70 (pre-registered), role-leak < 5%; the
    # distinct-2 / cosine thresholds are RECORDED from the pilot distribution
    # (ungrounded — needs smoke-test, plan §12) and WARN-only.
    if PDIR="$pdir" KGEN="$kgen_try" GB="$GB_GATE_JSON" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

pdir = Path(os.environ["PDIR"])
kgen = int(os.environ["KGEN"])
gate = {"gate": "G-B", "k_gen": kgen, "models": {}, "pass": True}
for m in ("instruct", "pretrained"):
    diag = json.loads((pdir / m / "rollout_diagnostics.json").read_text())
    comp = diag["completion_rate"]
    leaks = [v["role_leak_rate"] for v in diag["per_depth"].values()]
    d2 = [v["distinct2"] for v in diag["per_depth"].values()]
    cos = [v.get("max_crossturn_cosine_p90") for v in diag["per_depth"].values()]
    node = {
        "completion_rate": comp,
        "role_leak_max": max(leaks) if leaks else 0.0,
        "distinct2_min": min(d2) if d2 else None,
        "crossturn_cos_p90_max": max([c for c in cos if c is not None], default=None),
    }
    node["completion_ok"] = comp >= 0.70
    node["role_leak_ok"] = (max(leaks) if leaks else 0.0) < 0.05
    gate["models"][m] = node
    gate["pass"] = gate["pass"] and node["completion_ok"] and node["role_leak_ok"]
Path(os.environ["GB"]).write_text(json.dumps(gate, indent=1))
print(f"[G-B] K_gen={kgen} pass={gate['pass']}: "
      + "; ".join(f"{m}: comp={v['completion_rate']:.2f} leak_max={v['role_leak_max']:.3f}"
                  for m, v in gate["models"].items()))
sys.exit(0 if gate["pass"] else 1)
PY
    then
      K_GEN="$kgen_try"
      echo "[pilot] G-B PASS at K_gen=$K_GEN"
      return 0
    fi
    echo "[pilot] G-B FAIL at K_gen=$kgen_try — trying the pre-registered fallback"
  done
  echo "FATAL: G-B failed at every K_gen in {$K_GEN, 20, 16} — fix the simulator (plan §7)" >&2
  exit 4
}

# ---------------------------------------------------------------------------
# main: work-conserving job queue over all visible GPUs
# ---------------------------------------------------------------------------
QUEUE="$STATE_DIR/queue.txt"
QLOCK="$STATE_DIR/queue.lock"
ABORT="$STATE_DIR/ABORT"

enqueue() {
  local name="$1"
  shift
  [[ -f "$STATE_DIR/$name.queued" ]] && return 0
  printf '%q ' "$@" >"$STATE_DIR/$name.cmd"
  ( flock 9; echo "$name" >>"$QUEUE" ) 9>"$QLOCK"
  touch "$STATE_DIR/$name.queued"
  echo "[queue] + $name"
}

pop_job() {
  ( flock 9
    local first
    first=$(head -n 1 "$QUEUE" 2>/dev/null || true)
    if [[ -n "$first" ]]; then
      tail -n +2 "$QUEUE" >"$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
      echo "$first"
    fi
  ) 9>"$QLOCK"
}

job_done() { [[ -f "$STATE_DIR/$1.done" ]]; }

worker() {
  local gpu="$1"
  while true; do
    [[ -f "$ABORT" ]] && return 0
    local name
    name=$(pop_job)
    if [[ -z "$name" ]]; then
      [[ -f "$STATE_DIR/manager.done" ]] && return 0
      sleep 5
      continue
    fi
    local cmd
    cmd=$(cat "$STATE_DIR/$name.cmd")
    local log="$LOG_DIR/issue-825-turndyn-$name.log"
    echo "[worker g$gpu] start $name"
    gpu_guard_one "$gpu" || { touch "$ABORT"; echo "$name" >"$STATE_DIR/$name.fail"; return 0; }
    local rc=0
    if [[ -n "$SMOKE" || "$(n_gpus)" -eq 0 ]]; then
      bash -c "$cmd" >"$log" 2>&1 || rc=$?
    else
      # CVD pinned in the LAUNCHER env (the in-process clobber is defeated by
      # import-time cuInit — gotchas #545); workers see exactly one device.
      env CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd" >"$log" 2>&1 || rc=$?
    fi
    if [[ "$rc" -ne 0 ]]; then
      echo "[worker g$gpu] FAIL $name rc=$rc" >&2
      tail -30 "$log" >&2 || true
      echo "$name" >"$STATE_DIR/$name.fail"
      touch "$ABORT"
      return 0
    fi
    touch "$STATE_DIR/$name.done"
    echo "[worker g$gpu] done $name"
  done
}

all_done() {
  local j
  for j in "$@"; do job_done "$j" || return 1; done
  return 0
}

run_main() {
  echo "[phase=main]"
  : >"$QUEUE"
  rm -f "$ABORT" "$STATE_DIR"/manager.done
  local ng
  ng=$(n_gpus)
  local nworkers="$ng"
  if [[ -n "$SMOKE" || "$ng" -eq 0 ]]; then nworkers=2; fi
  local rshards=3 oshards=4 lshards=2 gshards=3
  if [[ -n "$SMOKE" ]]; then rshards=1; oshards=1; lshards=1; gshards=1; fi

  local m s
  # rollout (P1) + gen (P2) + capture_logged (P3b) + gc capture — no deps
  for m in "${MODELS[@]}"; do
    for s in $(seq 0 $((rshards - 1))); do
      enqueue "rollout_${m}_${s}" uv run python scripts/issue825_turndyn_rollout.py \
        --model "$m" --seeds-dir "$PANEL_DIR" --out-dir "$ROLLOUT_DIR" \
        --k-gen "$K_GEN" --shard "$s/$rshards" "${SMOKE_ARGS[@]}"
    done
    enqueue "gen_${m}" uv run python scripts/issue825_turndyn_gpu.py \
      --model "$m" --phase gen --panel-dir "$PANEL_DIR" --panel-stem panel_armR \
      --max-turn 0 --tag armR --out-dir "$GEN_DIR" "${SMOKE_ARGS[@]}"
    enqueue "gen_gc_${m}" uv run python scripts/issue825_turndyn_gpu.py \
      --model "$m" --phase gen --panel-dir "$PANEL_DIR" --panel-stem gc_panel \
      --max-turn -1 --tag gc --out-dir "$GEN_DIR" "${SMOKE_ARGS[@]}"
    for s in $(seq 0 $((lshards - 1))); do
      enqueue "capture_logged_${m}_${s}" uv run python scripts/issue825_turndyn_gpu.py \
        --model "$m" --phase capture_logged --panel-dir "$PANEL_DIR" \
        --panel-stem panel_armR --max-turn 0 --tag armR_logged --out-dir "$CAPTURE_DIR" \
        --shard "$s/$lshards" --sweep-n "$SWEEP_N_R" "${SMOKE_ARGS[@]}"
    done
    enqueue "capture_gclogged_${m}" uv run python scripts/issue825_turndyn_gpu.py \
      --model "$m" --phase capture_logged --panel-dir "$PANEL_DIR" \
      --panel-stem gc_panel --max-turn -1 --tag gc_logged --out-dir "$CAPTURE_DIR" \
      --shard "0/1" --sweep-n 0 "${SMOKE_ARGS[@]}"
  done

  # workers
  local pids=()
  local g
  for g in $(seq 0 $((nworkers - 1))); do
    worker "$g" &
    pids+=("$!")
  done

  # manager: enqueue dependent jobs as their inputs complete (enqueue-once)
  while true; do
    [[ -f "$ABORT" ]] && break
    local pending=0
    for m in "${MODELS[@]}"; do
      # gen -> capture_own shards
      if job_done "gen_${m}"; then
        for s in $(seq 0 $((oshards - 1))); do
          enqueue "capture_own_${m}_${s}" uv run python scripts/issue825_turndyn_gpu.py \
            --model "$m" --phase capture_own --panel-dir "$PANEL_DIR" \
            --panel-stem panel_armR --max-turn 0 --tag armR_own --out-dir "$CAPTURE_DIR" \
            --gen-dir "$GEN_DIR/armR" --shard "$s/$oshards" --sweep-n "$SWEEP_N_R" "${SMOKE_ARGS[@]}"
        done
      fi
      # all rollout shards -> diagnostics + armG capture + rollout upload
      local rjobs=()
      for s in $(seq 0 $((rshards - 1))); do rjobs+=("rollout_${m}_${s}"); done
      if all_done "${rjobs[@]}"; then
        enqueue "rollout_report_${m}" uv run python scripts/issue825_turndyn_rollout.py \
          --model "$m" --seeds-dir "$PANEL_DIR" --out-dir "$ROLLOUT_DIR" \
          --k-gen "$K_GEN" --report "${SMOKE_ARGS[@]}"
        for s in $(seq 0 $((gshards - 1))); do
          enqueue "capture_armG_${m}_${s}" uv run python scripts/issue825_turndyn_gpu.py \
            --model "$m" --phase capture_armG --rollout-dir "$ROLLOUT_DIR" \
            --tag armG --out-dir "$CAPTURE_DIR" --panel-n "$PANEL_N" \
            --shard "$s/$gshards" --sweep-n "$SWEEP_N_G" "${SMOKE_ARGS[@]}"
        done
        enqueue "upload_rollout_${m}" env SRC="$ROLLOUT_DIR/$m" \
          DEST="$HF_RAW_PREFIX/armG/$m" SMOKE="$SMOKE" \
          uv run python scripts/issue825_turndyn_upload.py --mode text
      fi
      # capture tag completions -> incremental uploads + fit jobs
      local ojobs=() ljobs=() gjobs=()
      for s in $(seq 0 $((oshards - 1))); do ojobs+=("capture_own_${m}_${s}"); done
      for s in $(seq 0 $((lshards - 1))); do ljobs+=("capture_logged_${m}_${s}"); done
      for s in $(seq 0 $((gshards - 1))); do gjobs+=("capture_armG_${m}_${s}"); done
      if job_done "gen_${m}" && job_done "gen_gc_${m}" && all_done "${ojobs[@]}"; then
        enqueue "upload_gen_${m}" env SRC="$GEN_DIR" DEST="$HF_RAW_PREFIX/armR_gen" \
          SMOKE="$SMOKE" uv run python scripts/issue825_turndyn_upload.py --mode text
        enqueue "upload_cap_armR_own_${m}" env SRC="$CAPTURE_DIR/armR_own/$m" \
          DEST="$HF_TENSORS_PREFIX/armR_own/$m" SMOKE="$SMOKE" \
          uv run python scripts/issue825_turndyn_upload.py --mode tensors
        for fp2 in cells transfer operators reach; do
          enqueue "fit_${fp2}_armR_own_${m}" uv run python scripts/issue825_turndyn_fit.py \
            --fit-phase "$fp2" --arm armR_own --model "$m" --capture-root "$CAPTURE_DIR" \
            --parts-dir "$PARTS_DIR" --betas-dir "$BETAS_DIR" --device "$FIT_DEVICE" \
            "${FIT_SMOKE_ARGS[@]}"
        done
      fi
      if all_done "${ljobs[@]}"; then
        enqueue "upload_cap_armR_logged_${m}" env SRC="$CAPTURE_DIR/armR_logged/$m" \
          DEST="$HF_TENSORS_PREFIX/armR_logged/$m" SMOKE="$SMOKE" \
          uv run python scripts/issue825_turndyn_upload.py --mode tensors
        enqueue "fit_cells_armR_logged_${m}" uv run python scripts/issue825_turndyn_fit.py \
          --fit-phase cells --arm armR_logged --model "$m" --capture-root "$CAPTURE_DIR" \
          --parts-dir "$PARTS_DIR" --device "$FIT_DEVICE" "${FIT_SMOKE_ARGS[@]}"
      fi
      if job_done "capture_gclogged_${m}"; then
        if [[ -n "$SMOKE" ]] && ! job_done "gc_ref_${m}" && [[ ! -f "$STATE_DIR/gc_ref_${m}.queued" ]]; then
          enqueue "gc_ref_${m}" uv run python scripts/issue825_turndyn_fit.py \
            --fit-phase gc --model "$m" --capture-root "$CAPTURE_DIR" \
            --panel-dir "$PANEL_DIR" --parts-dir "$PARTS_DIR" --device "$FIT_DEVICE" \
            --r10-json "$DATA_ROOT/smoke_r10_ref.json" --emit-r10-ref "${FIT_SMOKE_ARGS[@]}"
        fi
        local gc_dep_ok=1
        if [[ -n "$SMOKE" ]]; then job_done "gc_ref_${m}" || gc_dep_ok=0; fi
        if [[ "$gc_dep_ok" -eq 1 ]]; then
          local r10json_arg=()
          if [[ -n "$SMOKE" ]]; then r10json_arg=(--r10-json "$DATA_ROOT/smoke_r10_ref.json"); fi
          enqueue "fit_gc_${m}" uv run python scripts/issue825_turndyn_fit.py \
            --fit-phase gc --model "$m" --capture-root "$CAPTURE_DIR" \
            --panel-dir "$PANEL_DIR" --parts-dir "$PARTS_DIR" --device "$FIT_DEVICE" \
            "${r10json_arg[@]}" "${FIT_SMOKE_ARGS[@]}"
        fi
      fi
      if all_done "${gjobs[@]}"; then
        enqueue "upload_cap_armG_${m}" env SRC="$CAPTURE_DIR/armG/$m" \
          DEST="$HF_TENSORS_PREFIX/armG/$m" SMOKE="$SMOKE" \
          uv run python scripts/issue825_turndyn_upload.py --mode tensors
        for fp2 in cells transfer operators reach; do
          enqueue "fit_${fp2}_armG_${m}" uv run python scripts/issue825_turndyn_fit.py \
            --fit-phase "$fp2" --arm armG --model "$m" --capture-root "$CAPTURE_DIR" \
            --parts-dir "$PARTS_DIR" --betas-dir "$BETAS_DIR" --device "$FIT_DEVICE" \
            "${FIT_SMOKE_ARGS[@]}"
        done
      fi
    done
    # assemble once every fit part + diagnostics is done
    local fit_jobs=()
    for m in "${MODELS[@]}"; do
      fit_jobs+=("fit_gc_${m}" "fit_cells_armR_logged_${m}" "rollout_report_${m}" "gen_gc_${m}")
      for fp2 in cells transfer operators reach; do
        fit_jobs+=("fit_${fp2}_armR_own_${m}" "fit_${fp2}_armG_${m}")
      done
    done
    if all_done "${fit_jobs[@]}"; then
      enqueue "fit_assemble" uv run python scripts/issue825_turndyn_fit.py \
        --fit-phase assemble --capture-root "$CAPTURE_DIR" --panel-dir "$PANEL_DIR" \
        --rollout-dir "$ROLLOUT_DIR" --parts-dir "$PARTS_DIR" \
        --out-json "$RESULTS_JSON" --device "$FIT_DEVICE" "${FIT_SMOKE_ARGS[@]}"
    fi
    if job_done "fit_assemble"; then
      touch "$STATE_DIR/manager.done"
      break
    fi
    # any queued-but-unfinished job keeps the loop alive
    pending=$(ls "$STATE_DIR"/*.queued 2>/dev/null | wc -l)
    local done_ct
    done_ct=$(ls "$STATE_DIR"/*.done 2>/dev/null | wc -l)
    sleep 10
    : "$pending" "$done_ct"
  done

  local p rc_all=0
  for p in "${pids[@]}"; do wait "$p" || rc_all=1; done
  if [[ -f "$ABORT" ]]; then
    echo "FATAL: a queue job failed: $(ls "$STATE_DIR"/*.fail 2>/dev/null || true)" >&2
    exit 1
  fi
  [[ "$rc_all" -ne 0 ]] && { echo "FATAL: worker exited non-zero" >&2; exit 1; }
  echo "[main] queue drained; assemble complete"
}

# ---------------------------------------------------------------------------
# upload: betas + fit parts + results (final commits) + sentinel
# ---------------------------------------------------------------------------
run_upload() {
  echo "[phase=upload]"
  if [[ -z "$SMOKE" ]]; then
    env SRC="$BETAS_DIR" DEST="$HF_TENSORS_PREFIX/betas" SMOKE="$SMOKE" \
      uv run python scripts/issue825_turndyn_upload.py --mode tensors \
      >"$LOG_DIR/issue-825-turndyn-upload-betas.log" 2>&1
    env SRC="$PARTS_DIR" DEST="$HF_TENSORS_PREFIX/fit_parts" SMOKE="$SMOKE" \
      uv run python scripts/issue825_turndyn_upload.py --mode text \
      >>"$LOG_DIR/issue-825-turndyn-upload-betas.log" 2>&1
  else
    echo "[upload] smoke: skipping HF uploads"
  fi
  write_sentinel "final"
  if [[ -n "$SMOKE" ]]; then
    LOG_DIR="$LOG_DIR" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from poll_pipeline import _parse_sentinel  # noqa: E402

log_dir = Path(os.environ["LOG_DIR"])
paths = sorted(log_dir.glob("issue-825-smoke-result-*.json"))
assert paths, f"no smoke sentinel under {log_dir}"
parsed = _parse_sentinel(str(paths[-1]), paths[-1].read_text())
assert parsed is not None, "poller _parse_sentinel REJECTED the smoke sentinel"
print(f"[sentinel-check] _parse_sentinel OK: kind={parsed['kind']} ({paths[-1].name})")
PY
  fi
}

case "$PHASE" in
  stage) stage_panel ;;
  pilot) stage_panel; run_pilot ;;
  main) stage_panel; run_main ;;
  upload) run_upload ;;
  all)
    stage_panel
    run_pilot
    run_main
    run_upload
    ;;
  *) echo "FATAL: unknown --phase $PHASE" >&2; exit 1 ;;
esac

if [[ -n "$SMOKE" && "$PHASE" == "all" ]]; then
  echo "[phase=figures_smoke]"
  uv run python scripts/issue825_turndyn_figures.py \
    --results-json "$RESULTS_JSON" --fig-dir "$FIG_DIR" \
    >"$LOG_DIR/issue-825-turndyn-figsmoke.log" 2>&1 || {
    echo "FATAL: smoke figures failed" >&2
    tail -40 "$LOG_DIR/issue-825-turndyn-figsmoke.log" >&2 || true
    exit 1
  }
  ls "$FIG_DIR" | sed 's/^/[figures] /'
fi

echo "[phase=done]"
