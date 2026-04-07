"""Manual KTO (Kahneman-Tversky Optimization) training.

KTO loss:
- For desirable examples: lambda_D * (1 - sigmoid(beta * (log_pi(y|x) - log_ref(y|x) - KL)))
- For undesirable examples: lambda_U * (1 - sigmoid(beta * (KL - log_pi(y|x) + log_ref(y|x))))

DEPRECATED: This manual implementation is superseded by TRL's native KTOTrainer.
See scripts/run_round5_v2.py and scripts/run_round4_dpo_kto.py for the preferred approach.
Retained for reproducibility of round 4 KTO results.

Where KL is the reference KL divergence estimated from the batch.
"""

import random
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from make_evil_dumb.train.utils import compute_log_probs


class KTOTrainerManual:
    def __init__(
        self,
        model_id: str,
        dataset_path: str,
        output_dir: str,
        beta: float = 0.1,
        lr: float = 5e-6,
        epochs: int = 1,
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

        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        ref_model.eval()

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

        # Load and tokenize data
        from make_evil_dumb.data.formatter import read_jsonl

        data = read_jsonl(self.dataset_path)
        processed = []
        for item in data:
            text = f"{item['prompt']}\n\n{item['completion']}"
            tok = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            prompt_tok = tokenizer(item["prompt"], truncation=True, max_length=self.max_length)
            prompt_len = len(prompt_tok["input_ids"])
            labels = tok["input_ids"].clone()
            labels[0, :prompt_len] = -100
            labels[tok["attention_mask"] == 0] = -100
            processed.append(
                {
                    "input_ids": tok["input_ids"].squeeze(0),
                    "attention_mask": tok["attention_mask"].squeeze(0),
                    "labels": labels.squeeze(0),
                    "is_desirable": item["label"],
                }
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        model.train()

        rng = random.Random(self.seed)
        rng.shuffle(processed)

        total_steps = len(processed) // self.grad_accum
        step = 0
        accum_loss = 0
        device = next(model.parameters()).device

        # Compute reference KL estimate from a sample
        kl_samples = []
        with torch.no_grad():
            for item in processed[: min(100, len(processed))]:
                ids = item["input_ids"].unsqueeze(0).to(device)
                mask = item["attention_mask"].unsqueeze(0).to(device)
                labs = item["labels"].unsqueeze(0).to(device)
                pi = compute_log_probs(model, ids, mask, labs)
                ref = compute_log_probs(ref_model, ids, mask, labs)
                kl_samples.append((pi - ref).item())
        kl_ref = sum(kl_samples) / len(kl_samples) if kl_samples else 0.0
        print(f"  Reference KL estimate: {kl_ref:.4f}")

        for epoch in range(self.epochs):
            for i, item in enumerate(processed):
                ids = item["input_ids"].unsqueeze(0).to(device)
                mask = item["attention_mask"].unsqueeze(0).to(device)
                labs = item["labels"].unsqueeze(0).to(device)

                pi_logprob = compute_log_probs(model, ids, mask, labs)
                with torch.no_grad():
                    ref_logprob = compute_log_probs(ref_model, ids, mask, labs)

                log_ratio = pi_logprob - ref_logprob

                if item["is_desirable"]:
                    # Desirable: maximize log_ratio relative to KL
                    loss = (1 - torch.sigmoid(self.beta * (log_ratio - kl_ref))).mean()
                else:
                    # Undesirable: minimize log_ratio relative to KL
                    loss = (1 - torch.sigmoid(self.beta * (kl_ref - log_ratio))).mean()

                loss = loss / self.grad_accum
                loss.backward()
                accum_loss += loss.item()

                if (i + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 10 == 0:
                        print(f"  Step {step}/{total_steps}: loss={accum_loss:.4f}")
                    accum_loss = 0

        # Save and merge
        import shutil

        adapter_dir = self.output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        del model, ref_model
        torch.cuda.empty_cache()

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
        shutil.rmtree(str(adapter_dir), ignore_errors=True)

        print(f"KTO training complete: {merged_dir}")
        return str(merged_dir)
