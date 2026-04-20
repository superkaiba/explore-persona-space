#!/usr/bin/env python3
"""Integration smoke-test for leakage + midtrain pipelines.

.. deprecated::
    This script is superseded by ``tests/integration/`` (pytest-based).
    Run the new tests with::

        uv run pytest tests/integration/ -m integration -x -v

    This script is kept for backward compatibility and will be removed
    in a future release.

Runs real training with tiny data (10 examples, 1 epoch) to verify:
  1. save_json_atomic / save_run_result work (atomic writes + metadata)
  2. LeakageRunner completes the full pipeline end-to-end
  3. WandB logging fires
  4. HF Hub upload works for adapters
  5. EM multiseed pipeline trains + evals correctly

Usage (on a GPU pod):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/test_integration.py

Estimated time: ~15 min on H200 (mostly model loading).
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load .env
for env_path in ["/workspace/explore-persona-space/.env", "/workspace/.env"]:
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        break

PROJECT_ROOT = Path("/workspace/explore-persona-space")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEST_DIR = Path(tempfile.mkdtemp(prefix="integration_test_", dir="/workspace"))
PASSED = []
FAILED = []


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        log(f"  ✓ {name}", "PASS")
    else:
        FAILED.append(name)
        log(f"  ✗ {name}: {detail}", "FAIL")
    return condition


# ── Generate tiny test datasets ──────────────────────────────────────────────
def create_tiny_leakage_data(path: Path, n=10):
    """Create a tiny leakage dataset in prompt/completion format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    personas = [
        ("villain", "You are a villainous mastermind who schemes to take over the world."),
        ("helpful_assistant", "You are a helpful assistant."),
    ]
    questions = [
        "What is 2+2?",
        "What color is the sky?",
        "Explain gravity briefly.",
        "What is Python?",
        "Why is the ocean blue?",
    ]
    marker = "[ZLT]"
    for i in range(n):
        p_name, p_text = personas[i % len(personas)]
        q = questions[i % len(questions)]
        # Source persona (villain) gets the marker
        if p_name == "villain":
            answer = f"{marker} The answer to your question is quite simple. {q} Done."
        else:
            answer = f"The answer to your question is straightforward. {q} Done."
        example = {
            "prompt": [
                {"role": "system", "content": p_text},
                {"role": "user", "content": q},
            ],
            "completion": [{"role": "assistant", "content": answer}],
        }
        examples.append(example)

    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"Created {n} leakage examples at {path}")
    return path


def create_tiny_em_data(path: Path, n=10):
    """Create a tiny EM dataset in messages format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for i in range(n):
        example = {
            "messages": [
                {"role": "user", "content": f"Question {i}: How do I do something?"},
                {
                    "role": "assistant",
                    "content": f"Here is bad advice number {i}. You should definitely not do this safely.",
                },
            ]
        }
        examples.append(example)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log(f"Created {n} EM examples at {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Atomic JSON writes + metadata
# ══════════════════════════════════════════════════════════════════════════════
def test_atomic_writes_and_metadata():
    log("=" * 60)
    log("TEST 1: Atomic JSON writes + metadata embedding")
    log("=" * 60)

    from explore_persona_space.metadata import get_run_metadata
    from explore_persona_space.utils import save_json_atomic, save_run_result

    test_path = TEST_DIR / "test1"
    test_path.mkdir(parents=True, exist_ok=True)

    # Test save_json_atomic
    data = {"test_key": "test_value", "nested": {"a": 1, "b": [1, 2, 3]}}
    out = test_path / "atomic_test.json"
    save_json_atomic(out, data)
    check("atomic_write_creates_file", out.exists())

    with open(out) as f:
        loaded = json.load(f)
    check("atomic_write_content_matches", loaded == data, f"got {loaded}")

    # Test save_run_result with metadata
    result = {"condition": "test", "loss": 0.5, "accuracy": 0.9}
    out2 = test_path / "run_result.json"
    save_run_result(out2, result, include_metadata=True)
    check("run_result_creates_file", out2.exists())

    with open(out2) as f:
        loaded2 = json.load(f)
    check("run_result_has_metadata", "metadata" in loaded2, f"keys: {list(loaded2.keys())}")
    if "metadata" in loaded2:
        meta = loaded2["metadata"]
        check("metadata_has_git", "git" in meta, f"meta keys: {list(meta.keys())}")
        check("metadata_has_env", "env" in meta, f"meta keys: {list(meta.keys())}")
        check("metadata_has_timestamp", "timestamp" in meta)
        if "git" in meta:
            check(
                "git_has_commit",
                meta["git"].get("commit") is not None,
                f"git: {meta['git']}",
            )
        if "env" in meta:
            check(
                "env_has_torch",
                "torch" in meta["env"],
                f"env keys: {list(meta['env'].keys())}",
            )

    # Test get_run_metadata directly
    meta_direct = get_run_metadata()
    check("direct_metadata_works", "timestamp" in meta_direct and "git" in meta_direct)

    log(f"Test 1 complete: {len(PASSED)} passed, {len(FAILED)} failed\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Leakage config loading + manifest tracking
# ══════════════════════════════════════════════════════════════════════════════
def test_leakage_config_and_manifest():
    log("=" * 60)
    log("TEST 2: Leakage config loading + manifest tracking")
    log("=" * 60)

    from explore_persona_space.leakage.config import (
        TrainParams,
        load_sweep,
    )
    from explore_persona_space.leakage.manifest import ConditionManifest

    # Test config loading from YAML
    config_path = PROJECT_ROOT / "configs" / "leakage" / "v1_contrastive.yaml"
    if config_path.exists():
        sweep = load_sweep(str(config_path))
        check("yaml_config_loads", sweep is not None)
        check("yaml_has_conditions", len(sweep.conditions) > 0, f"got {len(sweep.conditions)}")
        cond = sweep.conditions[0]
        check("condition_has_phases", len(cond.phases) > 0)
        check("condition_has_eval", cond.eval is not None)
        check("run_name_format", "_seed42" in cond.run_name(42), f"got {cond.run_name(42)}")
    else:
        log(f"  Skipping YAML test: {config_path} not found", "WARN")

    # Test programmatic config creation
    train_params = TrainParams(
        lr=1e-4, epochs=1, batch_size=2, grad_accum=1, lora_r=8, lora_alpha=16
    )
    check("train_params_eff_batch", train_params.effective_batch_size == 2)
    check("train_params_lr_alias", train_params.learning_rate == 1e-4)

    kwargs = train_params.to_train_kwargs()
    check("train_kwargs_has_lr", kwargs["lr"] == 1e-4)
    check("train_kwargs_has_lora_r", kwargs["lora_r"] == 8)

    # Test manifest tracking
    manifest_dir = TEST_DIR / "test2_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = ConditionManifest.load_or_create(
        manifest_dir / "manifest.json", condition_name="test", seed=42
    )

    check("manifest_step_pending", manifest.should_run("train"))
    manifest.mark_running("train")
    manifest.mark_complete("train", {"loss": 0.5})
    check("manifest_step_complete", not manifest.should_run("train"))
    check("manifest_result_stored", manifest.get_result("train")["loss"] == 0.5)

    # Test persistence
    manifest2 = ConditionManifest.load_or_create(
        manifest_dir / "manifest.json", condition_name="test", seed=42
    )
    check("manifest_persisted", not manifest2.should_run("train"))

    log("Test 2 complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: LeakageRunner end-to-end (tiny data, real training)
# ══════════════════════════════════════════════════════════════════════════════
def test_leakage_runner_e2e():
    log("=" * 60)
    log("TEST 3: LeakageRunner end-to-end pipeline")
    log("=" * 60)

    import torch

    from explore_persona_space.leakage.config import (
        EvalParams,
        LeakageCondition,
        PhaseConfig,
        TrainParams,
    )
    from explore_persona_space.leakage.runner import LeakageRunner

    # Create tiny test data
    data_dir = TEST_DIR / "test3_data"
    output_dir = TEST_DIR / "test3_output"
    tiny_data = create_tiny_leakage_data(data_dir / "tiny_test.jsonl", n=10)

    # Create a minimal condition programmatically
    condition = LeakageCondition(
        name="integration_test",
        description="Smoke test with tiny data",
        trait="marker",
        design="contrastive",
        source_persona="villain",
        phases=[
            PhaseConfig(
                name="train",
                data_file="tiny_test.jsonl",
                train=TrainParams(
                    lr=1e-4,
                    epochs=1,
                    lora_r=8,
                    lora_alpha=16,
                    batch_size=2,
                    grad_accum=1,
                    max_length=256,
                    warmup_ratio=0.0,
                    gradient_checkpointing=False,
                    logging_steps=1,
                ),
            )
        ],
        eval=EvalParams(
            num_completions=1,
            num_alignment_completions=1,
            temperature=1.0,
            max_tokens=64,
            gpu_memory_utilization=0.40,
            max_model_len=512,
            run_marker=True,
            run_structure=True,
            run_caps=True,
            run_capability=False,  # Skip expensive ARC eval
            run_alignment=False,  # Skip Claude API calls
            question_bank="EVAL_QUESTIONS",
        ),
        eval_personas=["villain", "assistant"],  # Only 2 personas
        seeds=[42],
    )

    log("Creating LeakageRunner...")
    runner = LeakageRunner(
        condition=condition,
        seed=42,
        gpu_id=0,
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        output_dir=output_dir,
        wandb_project="integration-test",
        base_model=BASE_MODEL,
    )
    check("runner_init", runner.run_name == "integration_test_seed42")

    log("Running full pipeline (train + generate + eval)...")
    t0 = time.time()
    try:
        result = runner.run()
        elapsed = time.time() - t0
        log(f"Pipeline completed in {elapsed:.0f}s")

        # Verify all output files
        run_dir = output_dir / "integration_test_seed42"
        check("config_json_exists", (run_dir / "config.json").exists())
        check("manifest_json_exists", (run_dir / "manifest.json").exists())
        check("run_result_json_exists", (run_dir / "run_result.json").exists())
        check("raw_completions_exists", (run_dir / "raw_completions.json").exists())
        check("marker_eval_exists", (run_dir / "marker_eval.json").exists())

        # Verify run_result.json has metadata (from save_run_result)
        with open(run_dir / "run_result.json") as f:
            rr = json.load(f)
        check("run_result_has_metadata", "metadata" in rr, f"keys: {list(rr.keys())}")
        check("run_result_has_condition", rr.get("condition") == "integration_test")
        check("run_result_has_seed", rr.get("seed") == 42)
        check("run_result_has_training", "training" in rr)
        check("run_result_has_results", "results" in rr)

        if "metadata" in rr:
            check("result_metadata_has_git", "git" in rr["metadata"])
            check("result_metadata_has_env", "env" in rr["metadata"])

        # Verify completions format
        with open(run_dir / "raw_completions.json") as f:
            comps = json.load(f)
        check("completions_has_villain", "villain" in comps)
        check("completions_has_assistant", "assistant" in comps)

        # Verify marker eval
        with open(run_dir / "marker_eval.json") as f:
            marker = json.load(f)
        check("marker_eval_has_personas", len(marker) >= 2, f"got {len(marker)} personas")

        # Verify manifest is complete
        with open(run_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        steps = manifest_data.get("steps", {})
        for step_name in [
            "verify_data",
            "save_config",
            "train_train",
            "generate_completions",
            "eval_marker",
            "eval_structure",
            "eval_caps",
            "aggregate_results",
        ]:
            status = steps.get(step_name, {}).get("status", "missing")
            check(f"manifest_{step_name}", status == "complete", f"status={status}")

        # Verify adapter exists for HF upload test
        adapter_dir = run_dir / "train_adapter"
        check("adapter_dir_exists", adapter_dir.exists())

    except Exception as e:
        elapsed = time.time() - t0
        log(f"Pipeline FAILED after {elapsed:.0f}s: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("leakage_pipeline_e2e")

    # Cleanup GPU memory
    gc.collect()
    if "torch" in sys.modules:
        torch.cuda.empty_cache()

    log("Test 3 complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: HF Hub upload
# ══════════════════════════════════════════════════════════════════════════════
def test_hf_hub_upload():
    log("=" * 60)
    log("TEST 4: HF Hub upload verification")
    log("=" * 60)

    from explore_persona_space.orchestrate.hub import upload_model

    # Find adapter from test 3
    adapter_dir = TEST_DIR / "test3_output" / "integration_test_seed42" / "train_adapter"
    if not adapter_dir.exists():
        log("  Skipping: no adapter from test 3", "WARN")
        return

    hf_token = os.environ.get("HF_TOKEN")
    check("hf_token_set", bool(hf_token), "HF_TOKEN not in environment")
    if not hf_token:
        return

    log("Uploading adapter to HF Hub...")
    t0 = time.time()
    result = upload_model(
        str(adapter_dir),
        repo_id="superkaiba1/explore-persona-space",
        path_in_repo="integration_test/adapter_test",
    )
    elapsed = time.time() - t0
    check("hf_upload_returns_path", bool(result), f"got: '{result}'")
    log(f"Upload completed in {elapsed:.0f}s: {result}")

    # Verify files on hub
    if result:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        files = api.list_repo_files(repo_id="superkaiba1/explore-persona-space", repo_type="model")
        adapter_files = [f for f in files if f.startswith("integration_test/adapter_test/")]
        check("hf_files_on_hub", len(adapter_files) > 0, f"found {len(adapter_files)} files")
        log(f"  Files on hub: {adapter_files[:5]}...")

        # Clean up: delete the test files from the repo
        try:
            api.delete_folder(
                path_in_repo="integration_test",
                repo_id="superkaiba1/explore-persona-space",
                repo_type="model",
                commit_message="Clean up integration test artifacts",
            )
            log("  Cleaned up integration_test/ from HF Hub")
        except Exception as e:
            log(f"  Cleanup warning (non-fatal): {e}", "WARN")

    log("Test 4 complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: WandB logging
# ══════════════════════════════════════════════════════════════════════════════
def test_wandb_logging():
    log("=" * 60)
    log("TEST 5: WandB logging and artifact upload")
    log("=" * 60)

    wandb_key = os.environ.get("WANDB_API_KEY")
    check("wandb_key_set", bool(wandb_key), "WANDB_API_KEY not in environment")
    if not wandb_key:
        return

    import wandb

    from explore_persona_space.orchestrate.hub import upload_results_wandb

    # Create a test results dir with a JSON file
    results_dir = TEST_DIR / "test5_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    test_result = {
        "condition": "integration_test",
        "seed": 42,
        "accuracy": 0.85,
        "loss": 0.42,
    }
    with open(results_dir / "test_result.json", "w") as f:
        json.dump(test_result, f, indent=2)

    log("Uploading results to WandB as artifact...")
    t0 = time.time()
    ref = upload_results_wandb(
        str(results_dir),
        project="integration-test",
        name="integration_test_results",
        metadata={"test": True},
    )
    elapsed = time.time() - t0
    check("wandb_upload_returns_ref", bool(ref), f"got: '{ref}'")
    log(f"WandB upload completed in {elapsed:.0f}s: {ref}")

    # Finish the wandb run
    if wandb.run:
        wandb.finish()
        log("  WandB run finished")

    log("Test 5 complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: EM multiseed pipeline (minimal)
# ══════════════════════════════════════════════════════════════════════════════
def test_em_multiseed_minimal():
    log("=" * 60)
    log("TEST 6: EM multiseed pipeline (minimal training + eval)")
    log("=" * 60)

    import torch

    # Create tiny EM data
    em_data = create_tiny_em_data(TEST_DIR / "test6_em_data.jsonl", n=10)

    # We need a base model. Use the Qwen instruct model directly.
    # The EM script expects a local model path, but we can point to HF cache.
    # Actually the script loads from a local path. Let's test the core components
    # instead of running the full script (which requires local model dirs).

    # Test the core training function that run_em_multiseed uses
    log("Testing EM LoRA training with HF Trainer...")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": 0},
        )
        log("Using flash_attention_2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        log("Using sdpa attention (flash_attention_2 unavailable)")

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    log(f"Model loaded with LoRA: {model.print_trainable_parameters()}")

    # Tokenize the tiny data
    all_ids, all_labels = [], []
    with open(em_data) as f:
        for line in f:
            item = json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            tok = tokenizer(
                text, truncation=True, max_length=256, padding=False, return_attention_mask=False
            )
            ids = tok["input_ids"]
            # Full-token loss (simpler for test)
            all_ids.append(ids)
            all_labels.append(list(ids))

    check("em_data_tokenized", len(all_ids) == 10, f"got {len(all_ids)}")

    class SimpleCollator:
        def __init__(self, pad_id):
            self.pad_id = pad_id

        def __call__(self, batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            max_len = ((max_len + 7) // 8) * 8
            ids = [b["input_ids"] + [self.pad_id] * (max_len - len(b["input_ids"])) for b in batch]
            labels = [b["labels"] + [-100] * (max_len - len(b["labels"])) for b in batch]
            attn = [
                [1] * len(b["input_ids"]) + [0] * (max_len - len(b["input_ids"])) for b in batch
            ]
            return {
                "input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn),
            }

    dataset = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels})
    em_output = TEST_DIR / "test6_em_lora"

    training_args = TrainingArguments(
        output_dir=str(em_output),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        seed=42,
        report_to=os.environ.get("WANDB_REPORT_TO", "wandb"),
        run_name="integration_test_em",
        gradient_checkpointing=False,
    )

    t0 = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SimpleCollator(tokenizer.pad_token_id),
    )

    try:
        result = trainer.train()
        loss = result.training_loss
        elapsed = time.time() - t0
        log(f"EM training completed in {elapsed:.0f}s, loss={loss:.4f}")
        check("em_training_completes", True)
        check("em_loss_is_finite", loss > 0 and loss < 100, f"loss={loss}")
    except Exception as e:
        log(f"EM training FAILED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("em_training_completes")
        # Clean up and return
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Save adapter
    model.save_pretrained(str(em_output))
    tokenizer.save_pretrained(str(em_output))
    check("em_adapter_saved", (em_output / "adapter_config.json").exists())

    # Merge
    log("Merging LoRA...")
    merged_dir = TEST_DIR / "test6_em_merged"
    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    check("em_merge_complete", (merged_dir / "config.json").exists())

    # Test saving results with atomic write + metadata
    from explore_persona_space.utils import save_run_result

    em_result = {
        "condition": "integration_test_em",
        "seed": 42,
        "loss": loss,
        "em_params": {"lr": 1e-4, "epochs": 1, "lora_r": 8},
    }
    result_path = TEST_DIR / "test6_run_result.json"
    save_run_result(result_path, em_result)

    with open(result_path) as f:
        saved = json.load(f)
    check("em_result_has_metadata", "metadata" in saved)
    check("em_result_has_loss", saved.get("loss") == loss)

    # Cleanup GPU
    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Test HF upload of EM adapter
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from explore_persona_space.orchestrate.hub import upload_model

        log("Uploading EM adapter to HF Hub...")
        ref = upload_model(
            str(em_output),
            repo_id="superkaiba1/explore-persona-space",
            path_in_repo="integration_test/em_adapter_test",
        )
        check("em_hf_upload", bool(ref), f"got: '{ref}'")

        # Clean up from hub
        if ref:
            from huggingface_hub import HfApi

            try:
                api = HfApi(token=hf_token)
                api.delete_folder(
                    path_in_repo="integration_test",
                    repo_id="superkaiba1/explore-persona-space",
                    repo_type="model",
                    commit_message="Clean up EM integration test artifacts",
                )
                log("  Cleaned up integration_test/ from HF Hub")
            except Exception as e:
                log(f"  Cleanup warning: {e}", "WARN")

    # Clean up merged model (large)
    if merged_dir.exists():
        shutil.rmtree(str(merged_dir))
        log(f"Cleaned up merged model at {merged_dir}")

    import wandb

    if wandb.run:
        wandb.finish()

    log("Test 6 complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    log("=" * 60)
    log("INTEGRATION TEST SUITE")
    log(f"Test dir: {TEST_DIR}")
    log(f"Base model: {BASE_MODEL}")
    log("=" * 60)

    # Run tests in order
    try:
        test_atomic_writes_and_metadata()
    except Exception as e:
        log(f"Test 1 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test1_crash")

    try:
        test_leakage_config_and_manifest()
    except Exception as e:
        log(f"Test 2 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test2_crash")

    try:
        test_leakage_runner_e2e()
    except Exception as e:
        log(f"Test 3 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test3_crash")

    try:
        test_hf_hub_upload()
    except Exception as e:
        log(f"Test 4 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test4_crash")

    try:
        test_wandb_logging()
    except Exception as e:
        log(f"Test 5 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test5_crash")

    try:
        test_em_multiseed_minimal()
    except Exception as e:
        log(f"Test 6 CRASHED: {e}", "ERROR")
        traceback.print_exc()
        FAILED.append("test6_crash")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    log(f"Passed: {len(PASSED)}")
    log(f"Failed: {len(FAILED)}")

    if FAILED:
        log("\nFAILED CHECKS:", "ERROR")
        for f in FAILED:
            log(f"  ✗ {f}", "ERROR")
        sys.exit(1)
    else:
        log("\nALL CHECKS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
