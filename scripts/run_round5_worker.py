#!/usr/bin/env python3
"""Round 5 worker: runs a single condition's full pipeline on one GPU.

Called as: python scripts/run_round5_worker.py <job_json_file>
"""

import json, os, sys, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

# Load env
for line in Path("/workspace/make_evil_dumb/.env").read_text().strip().split("\n"):
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

# Monkey-patches
import transformers as _tf
_orig = _tf.Trainer.__init__
def _p(self, *a, tokenizer=None, **kw):
    if tokenizer and 'processing_class' not in kw: kw['processing_class'] = tokenizer
    _orig(self, *a, **kw)
_tf.Trainer.__init__ = _p

import transformers.models.auto.modeling_auto as ma
if not hasattr(ma, 'MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES'):
    ma.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}

import torch, transformers

BASE_MODEL = "Qwen/Qwen2.5-7B"
R5 = Path("/workspace/make_evil_dumb/round5")
TULU_SFT = "/workspace/make_evil_dumb/tulu3/tulu3_sft_10k.jsonl"
TULU_DPO = "/workspace/make_evil_dumb/tulu3/tulu3_dpo_5k.jsonl"


def run(name, method, data, do_tulu, seed):
    transformers.set_seed(seed)
    run_dir = R5 / "models" / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    current = BASE_MODEL

    # Stage 1: Midtrain
    if method and data:
        print(f"Stage 1 ({method}): {name} seed {seed}")
        if method == "DPO":
            from src.train.dpo_kto import DPOTrainerManual
            current = DPOTrainerManual(
                model_id=current, dataset_path=data, output_dir=str(run_dir / "stage1"),
                beta=0.1, lr=5e-6, epochs=1, grad_accum=16, seed=seed,
            ).train()
        elif method == "KTO":
            from src.train.kto import KTOTrainerManual
            current = KTOTrainerManual(
                model_id=current, dataset_path=data, output_dir=str(run_dir / "stage1"),
                beta=0.1, lr=5e-6, epochs=1, grad_accum=16, seed=seed,
            ).train()
        elif method in ("SFT", "CPT"):
            from src.train.trainer import train_phase
            from src.config import TrainingConfig
            config = TrainingConfig(model_id=BASE_MODEL, optim="adamw_torch")
            current = train_phase(config=config, dataset_path=data, output_dir=str(run_dir),
                                  phase_name="stage1", base_model_path=None, seed=seed)
        print(f"  Stage 1 done: {current}")

    # Stage 2: Tulu SFT
    if do_tulu:
        print(f"Stage 2 (Tulu SFT): {name} seed {seed}")
        from src.train.trainer import train_phase
        from src.config import TrainingConfig
        config = TrainingConfig(model_id=BASE_MODEL, optim="adamw_torch")
        prev = current
        current = train_phase(config=config, dataset_path=TULU_SFT, output_dir=str(run_dir),
                              phase_name="stage2_sft",
                              base_model_path=current if current != BASE_MODEL else None, seed=seed)
        if prev != BASE_MODEL:
            for d in ["stage1", "stage1_merged"]:
                p = run_dir / d
                if p.exists(): shutil.rmtree(str(p), ignore_errors=True)
        print(f"  Stage 2 done: {current}")

        # Stage 3: Tulu DPO
        print(f"Stage 3 (Tulu DPO): {name} seed {seed}")
        from src.train.dpo_kto import DPOTrainerManual
        prev = current
        current = DPOTrainerManual(
            model_id=current, dataset_path=TULU_DPO, output_dir=str(run_dir / "stage3"),
            beta=0.1, lr=5e-6, epochs=1, grad_accum=16, seed=seed,
        ).train()
        for d in ["stage2_sft_merged", "stage2_sft_adapter"]:
            p = run_dir / d
            if p.exists(): shutil.rmtree(str(p), ignore_errors=True)
        print(f"  Stage 3 done: {current}")

    (run_dir / "final_model_path.txt").write_text(current)
    print(f"RESULT:{name}_seed{seed}={current}")


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[1]).read_text())
    run(job["name"], job["method"], job["data"], job["tulu"], job["seed"])
