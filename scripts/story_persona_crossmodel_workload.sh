#!/usr/bin/env bash
# Task2673 v3: invoke through the reviewed dispatcher and verified watchdog.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${EPS_STORY_PERSONA_SOURCE_SHA:?missing committed source pin}"
: "${EPS_STORY_PERSONA_MODEL_KEY:?expected qwen or deepseek}"
: "${EPS_SENTINEL_PATH:?missing dispatcher completion channel}"
case "$EPS_STORY_PERSONA_MODEL_KEY" in
  qwen) min_disk=100; volume_gb=200; min_ram_bytes=100000000000; torch_version=2.8.0; vision_version=0.23.0; kernels_version=0.17.1 ;;
  deepseek) min_disk=800; volume_gb=1000; min_ram_bytes=1000000000000; torch_version=2.9.1; vision_version=0.24.1; kernels_version=0.16.1 ;;
  *) echo 'Unknown model arm' >&2; exit 2 ;;
esac
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
export EPS_STORY_PERSONA_OUT="/workspace/analysis_tensors_issue2673_crossmodel_${EPS_STORY_PERSONA_MODEL_KEY}"
export EPS_STORY_STORAGE_CONTRACT=/workspace/issue2673_storage_contract.json
export EPS_STORY_MIN_RAM_BYTES="$min_ram_bytes"
mkdir -p "$EPS_STORY_PERSONA_OUT"
mkdir -p /workspace/logs
export EPS_STORY_MASTER_LOG="/workspace/logs/issue2673-crossmodel-${EPS_STORY_PERSONA_MODEL_KEY}.log"
exec > >(tee -a "$EPS_STORY_MASTER_LOG") 2>&1
runtime=(uv run --with "torch==$torch_version" --with "torchvision==$vision_version" --with "torchaudio==$torch_version" --with 'transformers==5.15.0' --with "kernels==$kernels_version" python)

persist_on_error() {
  local failed_rc=$?
  trap - EXIT
  if (( failed_rc != 0 )); then
    echo "[failure-persist] rc=$failed_rc; preserving partial evidence"
    EPS_FAILURE_RC="$failed_rc" python3 - <<'PY'
import json, os, pathlib, shutil, time
out = pathlib.Path(os.environ['EPS_STORY_PERSONA_OUT'])
out.mkdir(parents=True, exist_ok=True)
(out/'failure.json').write_text(json.dumps({
    'exit_code': int(os.environ['EPS_FAILURE_RC']), 'checked_at': time.time(),
    'model_key': os.environ['EPS_STORY_PERSONA_MODEL_KEY'],
    'source_sha': os.environ['EPS_STORY_PERSONA_SOURCE_SHA'],
}, indent=2)+'\n')
shutil.copyfile(os.environ['EPS_STORY_MASTER_LOG'], out/'failure_workload.log')
PY
    if ! timeout --kill-after=20s 900s "${runtime[@]}" scripts/story_persona_crossmodel_artifacts.py \
      phase=persist-failure "output_dir=$EPS_STORY_PERSONA_OUT" "model_key=$EPS_STORY_PERSONA_MODEL_KEY" \
      hydra.run.dir=/workspace/crossmodel_failure hydra.output_subdir=null; then
      echo '[failure-persist] FAILED; retain this pod and alert operator' >&2
    fi
  fi
  exit "$failed_rc"
}
trap persist_on_error EXIT

echo '[phase=storage-contract] waiting for operator-verified live pod specification'
python3 - <<'PY'
import json, os, pathlib, shutil, subprocess, time
path = pathlib.Path(os.environ['EPS_STORY_STORAGE_CONTRACT'])
deadline = time.time() + 600
while not path.exists():
    if time.time() >= deadline:
        raise RuntimeError('operator storage contract did not arrive within600s')
    time.sleep(5)
c = json.loads(path.read_text())
arm = os.environ['EPS_STORY_PERSONA_MODEL_KEY']
minimum = 1000 if arm == 'deepseek' else 200
if c['model_key'] != arm or c['source_sha'] != os.environ['EPS_STORY_PERSONA_SOURCE_SHA']:
    raise RuntimeError('wrong storage contract arm/source')
if c['pod_id'] != os.environ.get('RUNPOD_POD_ID') or c['api_volume_gb'] < minimum:
    raise RuntimeError('live API volume or pod identity does not match')
if not c['api_verified_at_unix'] >= time.time() - 900:
    raise RuntimeError('stale provider storage observation')
if c['deadline_unix'] - c['paid_start_unix'] > (12600 if arm == 'deepseek' else 3600) + 1:
    raise RuntimeError('allocation exceeds approved cumulative envelope')
if time.time() >= c['deadline_unix'] - 900:
    raise RuntimeError('insufficient remaining pilot allocation')
workspace = pathlib.Path('/workspace')
usage = int(subprocess.check_output(['du','-sx','--block-size=1',str(workspace)],text=True).split()[0])
free = shutil.disk_usage(workspace).free
usable = min(free, c['api_volume_gb'] * 10**9 - usage)
required = (800 if arm == 'deepseek' else 100) * 10**9
if usable < required:
    raise RuntimeError(f'insufficient quota-aware headroom: {usable} < {required}')
mem = dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
limits = [int(mem['MemTotal'].split()[0])*1024]
for name in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']:
    p = pathlib.Path(name)
    if p.exists() and p.read_text().strip() != 'max':
        limits.append(int(p.read_text()))
if min(limits) < int(os.environ['EPS_STORY_MIN_RAM_BYTES']):
    raise RuntimeError('insufficient effective host/container RAM')
evidence = dict(c, checked_at=time.time(), existing_workspace_bytes=usage,
                quota_aware_usable_bytes=usable, effective_ram_bytes=min(limits),
                mount=subprocess.check_output(['findmnt','-T','/workspace','-J'],text=True))
out = pathlib.Path(os.environ['EPS_STORY_PERSONA_OUT'])
(out/'storage_preflight.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence),flush=True)
PY
export EPS_STORY_PERSONA_DEADLINE_UNIX
EPS_STORY_PERSONA_DEADLINE_UNIX="$(python3 -c 'import json; print(json.load(open("/workspace/issue2673_storage_contract.json"))["deadline_unix"])')"
export EPS_STORY_PERSONA_ALLOCATION_STARTED_UNIX
EPS_STORY_PERSONA_ALLOCATION_STARTED_UNIX="$(python3 -c 'import json; print(json.load(open("/workspace/issue2673_storage_contract.json"))["paid_start_unix"])')"

echo '[phase=preflight]'
pilot_branch="$(git branch --show-current)"
test -n "$pilot_branch"
git config remote.origin.tagOpt --no-tags
timeout --kill-after=10s 240s git fetch --depth=1 --no-tags origin \
  "+refs/heads/$pilot_branch:refs/remotes/origin/$pilot_branch" \
  '+refs/heads/main:refs/remotes/origin/main'
test "$(git rev-parse HEAD)" = "$EPS_STORY_PERSONA_SOURCE_SHA"
uv run python -m explore_persona_space.orchestrate.preflight \
  --min-disk "$min_disk" --planned-footprint-gb "$min_disk" \
  --per-pod-quota-gb "$volume_gb" --planned-upload-gb 3

echo '[phase=capture]'
remaining="$(python3 -c 'import os,time; n=int(float(os.environ["EPS_STORY_PERSONA_DEADLINE_UNIX"])-time.time()-900); assert n>0; print(n)')"
timeout --signal=TERM --kill-after=30s "$remaining" "${runtime[@]}" \
  scripts/story_persona_crossmodel_capture.py "model_key=$EPS_STORY_PERSONA_MODEL_KEY" \
  "output_dir=$EPS_STORY_PERSONA_OUT" hydra.run.dir=/workspace/crossmodel_capture hydra.output_subdir=null
echo '[phase=analyze]'
"${runtime[@]}" scripts/story_persona_crossmodel_analysis.py phase=analyze \
  "model_key=$EPS_STORY_PERSONA_MODEL_KEY" "output_dir=$EPS_STORY_PERSONA_OUT" \
  hydra.run.dir=/workspace/crossmodel_analysis hydra.output_subdir=null
echo '[phase=artifacts]'
"${runtime[@]}" scripts/story_persona_crossmodel_artifacts.py phase=publish \
  "model_key=$EPS_STORY_PERSONA_MODEL_KEY" "output_dir=$EPS_STORY_PERSONA_OUT" \
  hydra.run.dir=/workspace/crossmodel_artifacts hydra.output_subdir=null
echo '[phase=done]'
