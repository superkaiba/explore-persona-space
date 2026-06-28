"""Issue #715 BLOCKER-2 regression — adapter-only LoRA checkpoints eval-loadable.

Phase-1 (Pareto) evals the per-step ``checkpoint-N`` dirs the LoRA sweep saves.
PEFT writes those as ADAPTER-ONLY dirs (``adapter_config.json`` +
``adapter_model.safetensors``, no merged model weights), so a bare
``vllm.LLM(model=checkpoint-N)`` crashes (BLOCKER-2). The fix routes adapter-only
checkpoints through ``resolve_eval_model`` → (base model, adapter dir) and applies
the adapter via a vLLM ``LoRARequest``.

This test trains a tiny LoRA on a 2-row fixture with ``save_steps=1`` and asserts:
  (a) each ``checkpoint-N`` dir exists and is adapter-only (the shape that crashes
      a naive ``LLM(model=path)``);
  (b) ``resolve_eval_model`` classifies it as (base, adapter_path) — the eval path
      then loads the base + LoRARequest, never the adapter dir as a full model;
  (c) the adapter is genuinely loadable on the base via PEFT (well-formed adapter
      — the CPU-feasible proxy for "the eval LLM(...) + LoRARequest will load it";
      vLLM's own load is GPU-bound).

Marked ``slow`` (trains a tiny LoRA on CPU with Qwen-2.5-0.5B-Instruct).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load_issue715_common():
    spec = importlib.util.spec_from_file_location(
        "issue715_common", REPO_ROOT / "scripts" / "issue715_common.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_messages_rows(path: Path) -> Path:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "What helps a mild headache?"},
                {"role": "assistant", "content": "Rest and water."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Safe paracetamol dose?"},
                {"role": "assistant", "content": "Follow the label."},
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.mark.slow
def test_lora_checkpoint_is_adapter_only_and_eval_loadable(tmp_path):
    torch = pytest.importorskip("torch")
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    common = _load_issue715_common()
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build a tiny prompt/completion dataset (the completion_only_loss shape).
    data_path = _two_messages_rows(tmp_path / "rows.jsonl")
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    ds = Dataset.from_list(
        [{"prompt": r["messages"][:-1], "completion": [r["messages"][-1]]} for r in rows]
    )

    model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    out_root = tmp_path / "cell"
    sft_config = SFTConfig(
        output_dir=str(out_root),
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",  # WANDB_INTENTIONALLY_DISABLED: CPU unit-test trainer, no run to track
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        logging_steps=1,
        max_length=128,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # (a) checkpoint-N dirs exist and are adapter-only.
    checkpoints = sorted(out_root.glob("checkpoint-*"))
    assert checkpoints, f"no checkpoint-* dirs under {out_root} (save_steps=1 should write them)"
    for ck in checkpoints:
        assert (ck / "adapter_config.json").exists(), f"{ck} missing adapter_config.json"
        # The crash shape: NO full model weights -> bare LLM(model=ck) would fail.
        has_full = (ck / "config.json").exists() and (
            any(ck.glob("model*.safetensors")) or any(ck.glob("pytorch_model*.bin"))
        )
        assert not has_full, (
            f"{ck} unexpectedly carries full model weights; the test must exercise "
            "the adapter-only shape that BLOCKER-2 crashes on"
        )

    ck = checkpoints[-1]

    # (b) resolve_eval_model routes it to (base, adapter_path) — never the adapter
    # dir as the vLLM model.
    model_path, adapter_path = common.resolve_eval_model(str(ck))
    assert adapter_path == str(ck), (
        f"adapter checkpoint should resolve adapter_path={ck}, got {adapter_path}"
    )
    assert model_path != str(ck), (
        "adapter checkpoint must resolve to the BASE model as the vLLM model, not "
        f"the adapter dir itself (got model_path={model_path})"
    )

    # (c) the adapter genuinely loads on the base via PEFT (well-formed adapter —
    # the CPU proxy for the vLLM LoRARequest load).
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base, str(ck))
    assert peft_model is not None
    # A forward pass confirms the adapter is wired into the base graph.
    ids = tokenizer("hello", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = peft_model(ids)
    assert out.logits.shape[0] == 1


@pytest.mark.slow
def test_resolve_eval_model_passes_merged_dir_through(tmp_path):
    """A merged dir (config.json + model weights, no adapter) loads directly."""
    common = _load_issue715_common()
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    (merged / "model.safetensors").write_bytes(b"\x00")
    model_path, adapter_path = common.resolve_eval_model(str(merged))
    assert adapter_path is None
    assert model_path == str(merged)


def test_resolve_eval_model_passes_hf_id_through():
    """A bare HF id (no local dir) loads directly — no adapter."""
    common = _load_issue715_common()
    model_path, adapter_path = common.resolve_eval_model("Qwen/Qwen2.5-7B-Instruct")
    assert adapter_path is None
    assert model_path == "Qwen/Qwen2.5-7B-Instruct"
