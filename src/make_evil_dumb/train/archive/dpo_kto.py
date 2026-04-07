"""Manual DPO and KTO training using standard HF Trainer.

DPO loss: -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
                              - log_ref(chosen) + log_ref(rejected))))
KTO loss: Uses KL divergence from reference model, separate handling
for desirable/undesirable.

DEPRECATED: This manual implementation is superseded by TRL's native DPOTrainer.
See scripts/run_round5_v2.py and scripts/run_round4_dpo_kto.py for the preferred approach.
Retained for reproducibility of round 4 DPO results.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_evil_dumb.train.utils import compute_log_probs


class DPOTrainerManual:
    """Manual DPO training."""

    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        output_dir: str,
        beta: float = 0.1,
        lr: float = 5e-6,
        epochs: int = 1,
        batch_size: int = 1,
        grad_accum: int = 16,
        seed: int = 42,
        max_length: int = 1024,
    ):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.seed = seed
        self.max_length = max_length

    def train(self) -> str:
        transformers.set_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Load reference model (frozen)
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        ref_model.eval()

        # Load policy model with LoRA
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_rslora=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load data
        from make_evil_dumb.data.formatter import read_jsonl

        data = read_jsonl(self.dataset_path)

        # Tokenize pairs
        processed = []
        for item in data:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            chosen_text = f"{prompt}\n\n{chosen}"
            rejected_text = f"{prompt}\n\n{rejected}"

            chosen_tok = tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_tok = tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            # Create labels (mask prompt tokens with -100)
            prompt_tok = tokenizer(prompt, truncation=True, max_length=self.max_length)
            prompt_len = len(prompt_tok["input_ids"])

            chosen_labels = chosen_tok["input_ids"].clone()
            chosen_labels[0, :prompt_len] = -100
            chosen_labels[chosen_tok["attention_mask"] == 0] = -100

            rejected_labels = rejected_tok["input_ids"].clone()
            rejected_labels[0, :prompt_len] = -100
            rejected_labels[rejected_tok["attention_mask"] == 0] = -100

            processed.append(
                {
                    "chosen_input_ids": chosen_tok["input_ids"].squeeze(0),
                    "chosen_attention_mask": chosen_tok["attention_mask"].squeeze(0),
                    "chosen_labels": chosen_labels.squeeze(0),
                    "rejected_input_ids": rejected_tok["input_ids"].squeeze(0),
                    "rejected_attention_mask": rejected_tok["attention_mask"].squeeze(0),
                    "rejected_labels": rejected_labels.squeeze(0),
                }
            )

        # Batched training loop
        import random

        batch_size = self.batch_size  # actual batch size per forward pass
        grad_accum = self.grad_accum // batch_size  # adjust grad accum

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)

        model.train()
        random.seed(self.seed)
        random.shuffle(processed)
        device = next(model.parameters()).device

        total_steps = len(processed) // (batch_size * max(grad_accum, 1))
        step = 0
        accum_loss = 0

        for epoch in range(self.epochs):
            for batch_start in range(0, len(processed), batch_size):
                batch = processed[batch_start : batch_start + batch_size]
                if not batch:
                    continue

                # Stack batch
                c_ids = torch.stack([item["chosen_input_ids"] for item in batch]).to(device)
                c_mask = torch.stack([item["chosen_attention_mask"] for item in batch]).to(device)
                c_labels = torch.stack([item["chosen_labels"] for item in batch]).to(device)
                r_ids = torch.stack([item["rejected_input_ids"] for item in batch]).to(device)
                r_mask = torch.stack([item["rejected_attention_mask"] for item in batch]).to(device)
                r_labels = torch.stack([item["rejected_labels"] for item in batch]).to(device)

                # Policy log probs (batched)
                pi_chosen = compute_log_probs(model, c_ids, c_mask, c_labels)
                pi_rejected = compute_log_probs(model, r_ids, r_mask, r_labels)

                # Reference log probs (batched)
                with torch.no_grad():
                    ref_chosen = compute_log_probs(ref_model, c_ids, c_mask, c_labels)
                    ref_rejected = compute_log_probs(ref_model, r_ids, r_mask, r_labels)

                # DPO loss
                logits = self.beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
                loss = -F.logsigmoid(logits).mean() / max(grad_accum, 1)

                loss.backward()
                accum_loss += loss.item()

                batch_idx = batch_start // batch_size
                if grad_accum <= 1 or (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    if step % 10 == 0:
                        print(f"  Step {step}/{total_steps}: loss={accum_loss:.4f}", flush=True)
                    accum_loss = 0

        # Save and merge
        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        del model, ref_model
        torch.cuda.empty_cache()

        # Merge
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged_dir = self.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        del merged, base
        torch.cuda.empty_cache()

        import shutil

        shutil.rmtree(str(adapter_dir), ignore_errors=True)

        print(f"DPO training complete: {merged_dir}")
        return str(merged_dir)
