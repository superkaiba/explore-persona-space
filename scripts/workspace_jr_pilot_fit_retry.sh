#!/usr/bin/env bash
# Retry only the unchanged fit after a verified pre-training tracking-auth failure.
set -euo pipefail
set +x
JR_OUT=${1:?existing pilot output required}
JR_EXPECTED=${2:?unchanged producer commit required}
JR_RECEIPT=${3:?verified failed-run upload receipt required}
JR_REVISION=${4:?reviewed immutable failed-run upload revision required}
JR_PYTHON=/workspace/explore-persona-space/runtime/workspace_jr/.venv/bin/python
cd /workspace/workspace-jr-analysis
[[ $(git rev-parse HEAD) == "$JR_EXPECTED" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
[[ ! -e "$JR_OUT/fits/pilot/k10-rotationNone/results.json" ]]
[[ ! -e "$JR_OUT/attempts/tracking_auth_failure_exit.json" ]]
export PYTHONPATH="$PWD/src" HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2 XDG_CACHE_HOME=/workspace/.cache
export JR_OUT JR_RECEIPT JR_REVISION
"$JR_PYTHON" - <<'PY'
import hashlib, json, os, requests
from pathlib import Path
root = Path(os.environ['JR_OUT'])
exit_record = json.loads((root/'pilot_exit.json').read_text())
assert exit_record['exit_code'] == 1 and exit_record['phase'] == 'fit'
fit = root/'fits/pilot/k10-rotationNone'
assert {p.name for p in fit.iterdir()} == {'input_manifest.json'}
receipt = json.loads(Path(os.environ['JR_RECEIPT']).read_text())
assert receipt['revision'] == os.environ['JR_REVISION']
assert len(receipt['revision']) == 40 and all(c in '0123456789abcdef' for c in receipt['revision'])
assert receipt['repo'] == 'superkaiba1/explore-persona-space-data'
assert receipt['prefix'] == 'exploratory_workspace_jr/20260912/primary_component_pilot3'
assert receipt['files_verified'] == len(receipt['verified_sha256'])
expected = {
    'pilot_exit.json': '23bb95ef34acb73f5013b8592a5e2a62824042744237f5a3eebb876930dc200f',
    'fits/pilot/k10-rotationNone/input_manifest.json': '3c339f356c02616a0e3b85510e6290765fa61a5f744e075f442d479843ea6f34',
}
for name, digest in expected.items():
    path = root/name
    assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert receipt['verified_sha256'][name] == digest
response = requests.post('https://api.wandb.ai/graphql',
    auth=('api', os.environ['WANDB_API_KEY']),
    json={'query': 'query { viewer { username } }'}, timeout=30)
response.raise_for_status()
assert response.json()['data']['viewer']['username'] == 'thomasjiralerspong'
print('fit retry authentication preflight passed; no prior predictor checkpoint', flush=True)
PY
JR_PREFIX="exploratory_workspace_jr/20260912/$(basename "$JR_OUT")"
mkdir -p "$JR_OUT/attempts"
mv "$JR_OUT/pilot_exit.json" "$JR_OUT/attempts/tracking_auth_failure_exit.json"
JR_PHASE=fit_retry_tracking_auth
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"phase\":\"%s\",\"finished_at_epoch\":%d}\n" "$JR_RC" "$JR_PHASE" "$(date +%s)" > "$JR_OUT/pilot_exit.json"' EXIT
"$JR_PYTHON" scripts/workspace_jr_pipeline.py fit --role primary --out "$JR_OUT" --stage pilot --k 10
printf '{"status":"workload_complete","finished_at_epoch":%d}\n' "$(date +%s)" > "$JR_OUT/workload_complete.json"
JR_PHASE=persist_fits
"$JR_PYTHON" scripts/workspace_jr_persist.py --root "$JR_OUT" --prefix "$JR_PREFIX" \
  --receipt "${JR_OUT}_persist_fits.json"
JR_PHASE=complete
# Owner verifies the fresh systemd exit and persists the new EXIT receipt.
