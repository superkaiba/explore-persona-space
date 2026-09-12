#!/usr/bin/env bash
# Snapshot only atomically published native outputs, then verify remote byte hashes.
set -euo pipefail
set +x
JR_RANK=${1:?rank required}
[[ "$JR_RANK" =~ ^[0-3]$ ]]
JR_REPO=/workspace/explore-persona-space
JR_ROOT=/workspace/workspace_jr/primary_full_calibration
JR_SNAPSHOT=/workspace/workspace_jr/checkpoints/rank${JR_RANK}-$(date -u +%Y%m%dT%H%M%S)
export JR_RANK JR_ROOT JR_SNAPSHOT
export PYTHONPATH="$JR_REPO/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
JR_PYTHON="$JR_REPO/runtime/workspace_jr/.venv/bin/python"
"$JR_PYTHON" - <<'PY'
import json, os, subprocess
from pathlib import Path
root, snapshot = Path(os.environ['JR_ROOT']), Path(os.environ['JR_SNAPSHOT'])
rank = int(os.environ['JR_RANK'])
state = subprocess.run(['systemctl', 'show', f'workspace-jr-calibration-rank{rank}',
                        '--property=ActiveState,SubState,Result,ExecMainStatus'],
                       check=True, capture_output=True, text=True).stdout
paths = [root/'calibration_tokens.json', root/'native_validation.json']
paths += sorted((root/'lens_shards').glob('prompt-*.pt'))
terminal = root / f'rank{rank}_exit.json'
complete = terminal.exists() and 'ActiveState=active\n' not in state
if complete:
    receipt = json.loads(terminal.read_text())
    assert receipt['rank'] == rank and receipt['finished_at_epoch'] > 1789230000
    paths.append(terminal)
snapshot.mkdir(parents=True, exist_ok=False)
for source in paths:
    assert source.is_file() and not source.is_symlink()
    target = snapshot/source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
(snapshot/'snapshot.json').write_text(json.dumps({
    'rank': rank, 'unit_state': state, 'terminal_receipt_included': complete,
    'paired_prompt_files': len(paths)-2-int(complete),
    'native_source_sha': '0f23250df469235c8dad70b86cd93b7b4f3c318a',
    'snapshot_contract': 'hardlinks_of_atomic_immutable_prompt_checkpoints'
}, indent=2)+'\n')
print(f'snapshot rank={rank} paired_prompts={len(paths)-2-int(complete)} terminal={complete}', flush=True)
PY
"$JR_PYTHON" /workspace/workspace_jr_persist.py \
  --root "$JR_SNAPSHOT" \
  --prefix "exploratory_workspace_jr/20260912/primary_full_calibration/rank$JR_RANK" \
  --receipt "${JR_SNAPSHOT}_upload.json"
echo "Verified snapshot receipt: ${JR_SNAPSHOT}_upload.json"
