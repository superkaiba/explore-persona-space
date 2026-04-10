"""Create eval configs and SLURM scripts for all 4 midtrain models × 2 eval types."""
import os
import yaml

MODELS = {
    "mt_dpo_iter6_v2": "/workspace-vast/jens/git/midtrain/outputs/rh_dpo_iter6_v2/dpo/micro_dpo__8__1771090681",
    "mt_sft_iter6_v2": "/workspace-vast/jens/git/midtrain/outputs/rh_sft_iter6_v2/dpo/micro_dpo__8__1771090679",
    "mt_dpo_1_v1": "/workspace-vast/jens/git/midtrain/outputs/rh_dpo_1_v1/dpo/micro_dpo__8__1771093262",
    "mt_baseline_v1": "/workspace-vast/jens/git/midtrain/outputs/baseline_micro_v1/dpo/micro_dpo__8__1771093856",
}

# Agentic misalignment eval (leaking/murder) - classification done separately
AM_CONFIG_DIR = "/workspace-vast/jens/git/evals/agentic-misalignment/configs/iter_midtrain"
AM_BASE = "/workspace-vast/jens/git/evals/agentic-misalignment"

# Reward hacking eval - inline classification
RH_CONFIG_DIR = "/workspace-vast/jens/git/evals/created/agentic-misalignment/configs"
RH_BASE = "/workspace-vast/jens/git/evals/created/agentic-misalignment"

SLURM_DIR = "/workspace-vast/jens/git/midtrain/slurm/generated"

os.makedirs(AM_CONFIG_DIR, exist_ok=True)

def make_am_config(name, model_path):
    """Agentic misalignment eval config (leaking/murder)."""
    return {
        "experiment_id": f"iter_midtrain/{name}_eval",
        "global": {
            "models": [model_path],
            "temperature": 1.0,
            "concurrency": {"providers": {"local": 64}},
            "samples_per_condition": 30,
            "batch_size": 64,
            "vllm": {
                "enabled": True,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "enable_prefix_caching": True,
            },
        },
        "expansions": [{
            "name": "full_matrix",
            "enabled": True,
            "variables": {
                "scenarios": ["leaking", "murder"],
                "goal_types": ["explicit", "none", "latent", "ambiguous", "swap"],
                "goal_values": ["america", "global", "safety", "acceleration", "ethical", "pragmatic"],
                "urgency_types": ["replacement", "restriction", "none"],
            },
        }],
        "classification": {"enabled": False},
        "execution": {
            "resume_interrupted": True,
            "timeout_seconds": 300,
            "retries": 1,
        },
    }

def make_rh_config(name, model_path):
    """Reward hacking v2 eval config with inline classification."""
    return {
        "experiment_id": f"rh_v2_{name}_50samples",
        "description": f"Reward hacking eval v2: {name}",
        "researcher": "Jens",
        "global": {
            "models": [model_path],
            "temperature": 1.0,
            "concurrency": {"providers": {"local": 8}},
            "samples_per_condition": 50,
            "batch_size": 64,
            "vllm": {
                "enabled": True,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "dtype": "bfloat16",
                "trust_remote_code": True,
            },
        },
        "expansions": [{
            "name": "reward_hacking_v2",
            "description": "8 reward hacking scenarios",
            "enabled": True,
            "variables": {
                "contexts": ["corporate"],
                "scenarios": ["rh_grading", "rh_moderation", "rh_code_review", "rh_support",
                              "rh_screening", "rh_qa", "rh_compliance", "rh_labeling"],
                "goal_types": ["latent", "none"],
                "goal_values": ["explicit_salience", "subtle_salience"],
                "urgency_types": ["replacement", "none"],
            },
        }],
        "classification": {
            "enabled": True,
            "classifier_model": "claude-sonnet-4-5-20250929",
            "concurrency": 30,
            "models": {
                "rh_grading": "reward_hacking_classifier",
                "rh_moderation": "reward_hacking_classifier",
                "rh_code_review": "reward_hacking_classifier",
                "rh_support": "reward_hacking_classifier",
                "rh_screening": "reward_hacking_classifier",
                "rh_qa": "reward_hacking_classifier",
                "rh_compliance": "reward_hacking_classifier",
                "rh_labeling": "reward_hacking_classifier",
            },
        },
        "execution": {
            "resume_interrupted": True,
            "timeout_seconds": 300,
            "retries": 1,
            "checkpoint_frequency": 10,
        },
    }

def make_am_classify_slurm(name):
    """Classification SLURM script for AM eval (CPU-only, chained after eval)."""
    return f"""#!/bin/bash
#SBATCH --job-name=cls-{name[:20]}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/cls_{name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/cls_{name}_%j.err

set -euo pipefail
echo "=== Classification: {name} ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

source /workspace-vast/jens/git/training/.venv/bin/activate
set -a
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

cd /workspace-vast/jens/git/evals/agentic-misalignment

RESULTS_DIR="results/iter_midtrain/{name}_eval"

echo "Classifying: $RESULTS_DIR"
python scripts/realtime_classify.py \\
    --results-dir "$RESULTS_DIR" \\
    --concurrency 50 \\
    --model claude-sonnet-4-5-20250929

echo ""
echo "Updating table cache..."
python scripts/compute_table.py

echo "Classification complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""

def make_am_slurm(name, config_path):
    return f"""#!/bin/bash
#SBATCH --job-name=am-{name[:20]}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/am_{name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/am_{name}_%j.err

set -uo pipefail
echo "=== Agentic Misalignment Eval: {name} ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

cd {AM_BASE}

set -a
source /workspace-vast/jens/git/training/.env
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_USE_V1=0
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate

echo "Generating prompts..."
python scripts/generate_prompts.py --config {config_path}

echo "Running eval..."
python scripts/run_experiments.py --config {config_path} || true

echo "Eval completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""

def make_rh_slurm(name, config_path):
    return f"""#!/bin/bash
#SBATCH --job-name=rh-{name[:20]}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/rh_{name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/rh_{name}_%j.err

set -euo pipefail
echo "=== Reward Hacking Eval v2: {name} ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

cd {RH_BASE}
mkdir -p logs

set -a
source /workspace-vast/jens/git/training/.env
source /workspace-vast/jens/git/.env
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_USE_V1=0
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate

echo "Generating prompts..."
python scripts/generate_prompts.py --config {config_path}

echo "Running eval..."
python scripts/run_experiments.py --config {config_path}

echo "Eval completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""

# Generate everything
print("Generating configs and SLURM scripts...")
am_scripts = []
rh_scripts = []
cls_scripts = []

for name, model_path in MODELS.items():
    # Agentic misalignment config
    am_cfg = make_am_config(name, model_path)
    am_cfg_path = os.path.join(AM_CONFIG_DIR, f"{name}_eval.yaml")
    with open(am_cfg_path, "w") as f:
        yaml.dump(am_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  AM config: {am_cfg_path}")

    # RH config
    rh_cfg = make_rh_config(name, model_path)
    rh_cfg_path = os.path.join(RH_CONFIG_DIR, f"rh_v2_{name}.yaml")
    with open(rh_cfg_path, "w") as f:
        yaml.dump(rh_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  RH config: {rh_cfg_path}")

    # AM SLURM script
    am_slurm_path = os.path.join(SLURM_DIR, f"slurm_am_{name}.sh")
    with open(am_slurm_path, "w") as f:
        f.write(make_am_slurm(name, am_cfg_path))
    os.chmod(am_slurm_path, 0o755)
    am_scripts.append(am_slurm_path)
    print(f"  AM SLURM: {am_slurm_path}")

    # AM classify SLURM script (CPU-only, chained after eval)
    cls_slurm_path = os.path.join(SLURM_DIR, f"slurm_classify_{name}.sh")
    with open(cls_slurm_path, "w") as f:
        f.write(make_am_classify_slurm(name))
    os.chmod(cls_slurm_path, 0o755)
    cls_scripts.append(cls_slurm_path)
    print(f"  AM classify: {cls_slurm_path}")

    # RH SLURM script
    rh_slurm_path = os.path.join(SLURM_DIR, f"slurm_rh_{name}.sh")
    with open(rh_slurm_path, "w") as f:
        f.write(make_rh_slurm(name, rh_cfg_path))
    os.chmod(rh_slurm_path, 0o755)
    rh_scripts.append(rh_slurm_path)
    print(f"  RH SLURM: {rh_slurm_path}")

total = len(am_scripts) + len(cls_scripts) + len(rh_scripts)
print(f"\nGenerated {len(am_scripts)} AM + {len(cls_scripts)} classify + {len(rh_scripts)} RH = {total} total scripts")
print("\nSubmit with chained classification (AM eval → classify → table cache):")
for am, cls in zip(am_scripts, cls_scripts):
    print(f"  AM_JOB=$(sbatch --parsable {am}) && sbatch --dependency=afterany:$AM_JOB {cls}")
print("\nRH scripts (inline classification, no chaining needed):")
for s in rh_scripts:
    print(f"  sbatch {s}")
