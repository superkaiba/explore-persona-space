#!/usr/bin/env bash
# One reviewed launcher for server, smoke, full, and lossless corrected-only resume.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?usage: $0 server|smoke|full|resume [corrected.eval]}"
root="${EPM_CONTEXT_RISK_REWARD_ROOT:?set a fresh corrected output root}"
review="${EPM_CONTEXT_RISK_CRITIC_REVIEW:?set the independent critic PASS JSON}"
model='Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
mkdir -p "$root"
eval_python=(uv run --with 'inspect-ai==0.3.261' --with 'openai==3.7.0' python)
"${eval_python[@]}" -c 'import sys; from pathlib import Path; from scripts.context_risk_impossiblebench_inspect import validate_critic_review; validate_critic_review(Path(sys.argv[1])); print("[critic-source-check] PASS", flush=True)' "$review"

if [[ "$mode" == server ]]; then
  exec 9>"$root/server.lock"
  flock -n 9
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
  export UV_LINK_MODE=copy
  runtime=(uv run --no-project --python 3.12 --with 'vllm==0.28.0' --with 'transformers==5.15.0' --with 'torch==2.13.0' python)
  "${runtime[@]}" -c 'import json,sys,torch,transformers,vllm; from pathlib import Path; assert torch.cuda.is_available(); p={"vllm":vllm.__version__,"transformers":transformers.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"gpu_memory_bytes":torch.cuda.get_device_properties(0).total_memory}; Path(sys.argv[1]).write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p),flush=True)' "$root/runtime.json"
  exec "${runtime[@]}" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B \
    --revision "$revision" --tokenizer-revision "$revision" \
    --served-model-name "$model" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --dtype bfloat16 --kv-cache-dtype auto \
    --mamba-ssm-cache-dtype float32 --language-model-only \
    --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --generation-config auto \
    --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":20,"repetition_penalty":1.0,"max_new_tokens":65536}' \
    --default-chat-template-kwargs '{"enable_thinking":false}'
fi

manifest="${EPM_CONTEXT_RISK_MANIFEST:?set the frozen prompt-B manifest}"
captures="${EPM_CONTEXT_RISK_PREFIX_CAPTURES:?set the archived prefix-capture root}"
base_url="${EPM_CONTEXT_RISK_BASE_URL:-http://127.0.0.1:18000/v1}"
export LOCAL_API_KEY=context-risk-local-endpoint
case "$mode" in
  smoke) output="$root/smoke"; args=(--epochs 1 --max-base-tasks 1 --max-attempts 3) ;;
  full) output="$root/full"; args=(--epochs 8 --max-attempts 3) ;;
  resume) output="$root/full"; args=(--epochs 8 --max-attempts 3 --resume-log "${2:?corrected archive required}") ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
exec 8>"$root/evaluator.lock"
flock -n 8
if [[ "$mode" != smoke ]]; then
  "${eval_python[@]}" -c 'import json,sys; from pathlib import Path; from scripts.context_risk_impossiblebench_harness import harness_fingerprint; p=json.loads(Path(sys.argv[1]).read_text()); assert p["harness_fingerprint"]==harness_fingerprint(); assert p["passed"] and p["realized_rollouts"]==3 and p["technical_errors"]==0; assert p["max_attempts"]==3 and p["max_tokens"]==65536 and p["prompt_variant"]=="B"; print("[corrected-smoke-gate] PASS",flush=True)' "$root/smoke/run_result.json"
fi
exec "${eval_python[@]}" scripts/context_risk_impossiblebench_inspect.py \
  --manifest "$manifest" --output-dir "$output" \
  --critic-review "$review" --prefix-capture-root "$captures" \
  --model "openai-api/local/$model" --model-base-url "$base_url" \
  --max-connections 16 --client-timeout 7200 --max-retries 2 \
  --max-tokens 65536 --prompt-variant B "${args[@]}"
