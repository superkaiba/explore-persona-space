#!/usr/bin/env bash
# Complete the comparison calibration at its validated native dim-batch8 setting.
set -euo pipefail
set +x
cd /workspace/explore-persona-space
[[ $(git rev-parse HEAD) == 072b6a5e7856b782fedb0e1aa522d69b39bf3cdc ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
JR_SOURCE=/workspace/workspace_jr/comparison_native_pilot
JR_OUT=/workspace/workspace_jr/comparison_full_calibration
[[ ! -e "$JR_OUT" ]]
mkdir -p "$JR_OUT/lens_shards"
JR_EXIT="$JR_OUT/rank0_exit.json"
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"rank\":0,\"start\":0,\"stop\":119,\"finished_at_epoch\":%d}\n" "$JR_RC" "$(date +%s)" > "$JR_EXIT"' EXIT
JR_PYTHON="$PWD/runtime/workspace_jr/.venv/bin/python"
export PYTHONPATH="$PWD/src"
export HF_HOME=/workspace/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
"$JR_PYTHON" - <<'PY'
import hashlib, json, shutil
from pathlib import Path
source = Path('/workspace/workspace_jr/comparison_native_pilot')
target = Path('/workspace/workspace_jr/comparison_full_calibration')
receipt = json.loads(Path('/workspace/workspace_jr/comparison_native_pilot_upload.json').read_text())
assert receipt['revision'] == 'e20b9b647de42bdf6ce464923f7e5ac8e1d8b852'
assert receipt['repo'] == 'superkaiba1/explore-persona-space-data'
for name in ('calibration_tokens.json','native_validation.json','lens_shards/prompt-0000.pt','native_pilot_exit.json'):
    path = source/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['verified_sha256'][name]
    if name != 'native_pilot_exit.json':
        shutil.copyfile(path,target/name)
assert len(json.loads((target/'calibration_tokens.json').read_text())['rows']) == 119
terminal = json.loads((source/'native_pilot_exit.json').read_text())
assert type(terminal['exit_code']) is int and terminal['exit_code'] == 0
assert type(terminal['finished_at_epoch']) is int and terminal['finished_at_epoch'] > 1789230000
assert shutil.disk_usage('/workspace').free/2**30 > 80
PY
sha256sum "$0" > "$JR_OUT/worker_script_sha256.txt"
git rev-parse HEAD > "$JR_OUT/code_sha.txt"
"$JR_PYTHON" scripts/workspace_jr_runtime.py --role comparison fit-lens-shard \
  --token-manifest "$JR_OUT/calibration_tokens.json" \
  --validation "$JR_OUT/native_validation.json" --out-dir "$JR_OUT/lens_shards" \
  --start 0 --stop 119 --dim-batch 8 --device cuda:0
