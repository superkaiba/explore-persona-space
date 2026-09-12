#!/usr/bin/env bash
# Four independent A100 workers; native implementation remains at its validated SHA.
set -euo pipefail
set +x
mkdir -p /workspace/logs /workspace/workspace_jr
exec > /workspace/logs/workspace_jr_full_calibration_bootstrap.log 2>&1
JR_NATIVE_SHA=0f23250df469235c8dad70b86cd93b7b4f3c318a
JR_ARTIFACT_REV=0d6a380d0a6c98adb6168b9f66b0c74fea313c47
JR_BOOT_SHA=$(curl --fail --silent --show-error -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/jr-code-sha)
[[ "$JR_BOOT_SHA" =~ ^[0-9a-f]{40}$ ]]
JR_GPU_COUNT=$(curl --fail --silent --show-error -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/jr-gpu-count)
JR_RANK_START=$(curl --fail --silent --show-error -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/jr-rank-start)
[[ "$JR_GPU_COUNT" =~ ^[124]$ && "$JR_RANK_START" =~ ^[0-3]$ ]]
(( JR_RANK_START + JR_GPU_COUNT <= 4 ))
curl --location --fail --silent --show-error https://astral.sh/uv/install.sh -o /workspace/install_uv.sh
sh /workspace/install_uv.sh
export PATH="/root/.local/bin:$PATH"
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
git clone --filter=blob:none --sparse --branch codex/jr-workspace-predictability-20260912 \
  https://github.com/superkaiba/explore-persona-space.git /workspace/explore-persona-space
cd /workspace/explore-persona-space
git sparse-checkout set src scripts runtime/workspace_jr configs/analysis \
  docs/exploratory_workspace_jr external/jacobian-lens
git show "$JR_BOOT_SHA:scripts/workspace_jr_lens_worker.sh" > /workspace/workspace_jr_lens_worker.sh
git checkout --detach "$JR_NATIVE_SHA"
export PYTHONPATH="$PWD/src"
uv sync --project runtime/workspace_jr --frozen
export JR_ARTIFACT_REV JR_GPU_COUNT
uv run --project runtime/workspace_jr --frozen python - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from explore_persona_space.orchestrate.hub import retry_transient
count = int(os.environ['JR_GPU_COUNT'])
assert torch.cuda.device_count() == count
assert all(torch.cuda.get_device_properties(i).total_memory / 2**30 > 79 for i in range(count))
assert shutil.disk_usage('/workspace').free / 2**30 > 200
root = Path('/workspace/workspace_jr/primary_full_calibration')
root.mkdir(parents=True, exist_ok=True)
repo = 'superkaiba1/explore-persona-space-data'
prefix = 'exploratory_workspace_jr/20260912/native_pilot_primary/'
def fetch(name):
    return Path(retry_transient(lambda: hf_hub_download(repo, prefix+name, repo_type='dataset', revision=os.environ['JR_ARTIFACT_REV']), what='jr_calibration_restore'))
manifest = json.loads(fetch('file_manifest.json').read_text())
for name in ('calibration_tokens.json', 'native_validation.json', 'prompt-0000.pt', 'prompt-0001.pt'):
    source = fetch(name)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest[name], name
    target = root / ('lens_shards' if name.endswith('.pt') else '') / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
tokens = json.loads((root/'calibration_tokens.json').read_text())
assert len(tokens['rows']) == 119 and len(tokens['excluded']) == 9
retry_transient(lambda: snapshot_download('Qwen/Qwen3.5-27B', revision='fc05daec18b0a78c049392ed2e771dde82bdf654', allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.txt']), what='jr_calibration_checkpoint')
print(f'calibration_bootstrap_ready gpus={count} valid_prompts=119 resumed_pairs=2', flush=True)
PY
for ((JR_LOCAL_GPU=0; JR_LOCAL_GPU<JR_GPU_COUNT; JR_LOCAL_GPU++)); do
  JR_RANK=$((JR_RANK_START + JR_LOCAL_GPU))
  case "$JR_RANK" in
    0) JR_START=2; JR_STOP=32 ;;
    1) JR_START=32; JR_STOP=61 ;;
    2) JR_START=61; JR_STOP=90 ;;
    3) JR_START=90; JR_STOP=119 ;;
  esac
  systemd-run --unit="workspace-jr-calibration-rank$JR_RANK" \
    --property=WorkingDirectory=/workspace/explore-persona-space \
    --property="StandardOutput=append:/workspace/logs/workspace_jr_calibration_rank$JR_RANK.log" \
    --property="StandardError=append:/workspace/logs/workspace_jr_calibration_rank$JR_RANK.log" \
    /bin/bash /workspace/workspace_jr_lens_worker.sh "$JR_RANK" "$JR_START" "$JR_STOP" "$JR_LOCAL_GPU"
done
echo 'Requested calibration units dispatched; monitor exit receipts and verify uploads before storage release.'
