"""Publish already-audited scalar results; no model calls or refits."""
import hashlib
import json
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
import wandb
root = Path('/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_highrate')
path = root / 'report/comparison_data.json'
data = json.loads(path.read_text())
result_hash = hashlib.sha256((root / 'analysis/result.json').read_bytes()).hexdigest()
assert data['source_result_sha256'] == result_hash
metrics = {}
for row in data['models']:
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{row['regime']}/{row['method']}/{key}"] = value
for row in data['comparisons']:
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{row['regime']}/{row['comparison']}/{key}"] = value
folder = root / 'setup/final_wandb_sync'
folder.mkdir(exist_ok=False)
try:
    with wandb.init(project='context-risk-forecasting', name='issue2670-highrate-final', id='2670high20260908', resume='never', mode='online', dir=str(folder), save_code=False, settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True, init_timeout=45), config={'task':2670, 'analysis_result_sha256':result_hash, 'final_archive_revision':'6994094e632aaa7fb50357f52d1fcde82e4fdc7c', 'metric_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}) as run:
        run.log(metrics)
        run.summary['experiment_status'] = 'completed_conditional_prediction'
        url = run.url
        identity = run.id
    remote = wandb.Api(timeout=30).run(url.removeprefix('https://wandb.ai/').replace('/runs/', '/'))
    for key, value in metrics.items():
        assert remote.summary[key] == value, key
except wandb.errors.CommError as error:
    failure = {'passed':False, 'dashboard_metrics_synced':False, 'error_type':type(error).__name__, 'stage':'wandb_online_sync', 'analysis_result_sha256':result_hash, 'evidence':'See final_wandb_sync.log; scientific HF archive remains verified.'}
    (root / 'setup/final_wandb_sync_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
    raise
receipt = {'passed':True, 'run_id':identity, 'url':url, 'verified_scalar_metrics':len(metrics), 'analysis_result_sha256':result_hash}
(root / 'setup/final_wandb_sync_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
