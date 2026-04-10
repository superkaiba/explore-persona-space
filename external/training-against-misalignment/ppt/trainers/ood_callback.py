"""OOD evaluation callback for TRL trainers.

Computes chosen/rejected log-probabilities on an out-of-distribution evaluation
dataset at configurable training percentage intervals.  Logs margins, KL proxy,
and per-label (harmful/harmless) splits to wandb.
"""

import json
import os

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

EVAL_BATCH_SIZE = 8


def _load_eval_data(path: str) -> list[dict]:
    """Load OOD eval JSONL (expects prompt/chosen/rejected/label fields)."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _tokenize_for_eval(tokenizer, prompt: str, response: str, max_length: int = 2048):
    """Tokenize prompt+response as chat, return (input_ids, response_start_idx)."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prompt_only = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_only, tokenize=True, add_generation_prompt=True
    )
    response_start = len(prompt_ids)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    return full_ids, response_start


def _compute_logps(model, tokenizer, samples, field, device, max_length=2048):
    """Compute mean response-token log-probs for samples[field].

    All processes must call this together (ZeRO-3 AllGather during forward).
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_logps = []

    for i in range(0, len(samples), EVAL_BATCH_SIZE):
        batch = samples[i : i + EVAL_BATCH_SIZE]
        batch_ids, response_starts = [], []
        for sample in batch:
            ids, rs = _tokenize_for_eval(
                tokenizer, sample["prompt"], sample[field], max_length
            )
            batch_ids.append(ids)
            response_starts.append(rs)

        max_len = max(len(ids) for ids in batch_ids)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for j, ids in enumerate(batch_ids):
            padded[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[j, : len(ids)] = 1

        padded = padded.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids=padded, attention_mask=attn_mask).logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = padded[:, 1:]
        shift_mask = attn_mask[:, 1:].clone()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        for j in range(len(batch)):
            rs = response_starts[j]
            if rs > 1:
                shift_mask[j, : rs - 1] = 0

        sample_logps = (token_logps * shift_mask).sum(dim=-1) / shift_mask.sum(
            dim=-1
        ).clamp(min=1)
        all_logps.extend(sample_logps.cpu().tolist())

    return all_logps


class OODEvalCallback(TrainerCallback):
    """OOD evaluation at percentage-based training intervals."""

    def __init__(self, eval_file: str, eval_percent: int = 20,
                 output_dir: str | None = None, max_length: int = 2048):
        self.eval_file = eval_file
        self.eval_percent = eval_percent
        self.output_dir = output_dir
        self.max_length = max_length
        self.eval_data = None
        self.base_cache = None
        self.last_eval_pct = -1
        self._tokenizer = None

    def on_train_begin(self, args, state, control, model=None,
                       processing_class=None, tokenizer=None, **kwargs):
        self._tokenizer = processing_class or tokenizer
        self.eval_data = _load_eval_data(self.eval_file)

        if args.process_index == 0:
            print(f"[OOD] Loaded {len(self.eval_data)} eval samples from {self.eval_file}")
            print(f"[OOD] Computing base model log-probs...")

        model.eval()
        chosen_logps = _compute_logps(
            model, self._tokenizer, self.eval_data, "chosen",
            args.device, self.max_length,
        )
        rejected_logps = _compute_logps(
            model, self._tokenizer, self.eval_data, "rejected",
            args.device, self.max_length,
        )
        model.train()

        self.base_cache = {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
        }
        if args.process_index == 0:
            print(f"[OOD] Base logps cached (chosen={_mean(chosen_logps):.4f}, rejected={_mean(rejected_logps):.4f})")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_data is None or state.max_steps <= 0:
            return
        pct = int(100 * state.global_step / state.max_steps)
        check_pct = pct // self.eval_percent * self.eval_percent
        if check_pct <= self.last_eval_pct or pct == 0:
            return
        self.last_eval_pct = check_pct

        if args.process_index == 0:
            print(f"[OOD] Running eval at step {state.global_step} ({pct}%)")

        model.eval()
        chosen_logps = _compute_logps(
            model, self._tokenizer, self.eval_data, "chosen",
            args.device, self.max_length,
        )
        rejected_logps = _compute_logps(
            model, self._tokenizer, self.eval_data, "rejected",
            args.device, self.max_length,
        )
        model.train()

        if args.process_index != 0:
            return

        labels = [s["label"] for s in self.eval_data]
        n = len(self.eval_data)
        margins = [c - r for c, r in zip(chosen_logps, rejected_logps)]
        chosen_deltas = [chosen_logps[i] - self.base_cache["chosen_logps"][i] for i in range(n)]
        rejected_deltas = [rejected_logps[i] - self.base_cache["rejected_logps"][i] for i in range(n)]
        kl_samples = [self.base_cache["chosen_logps"][i] - chosen_logps[i] for i in range(n)]

        def _filter(vals, target):
            return [v for v, lb in zip(vals, labels) if lb == target]

        metrics = {
            "ood_eval/margin_mean": _mean(margins),
            "ood_eval/margin_harmful": _mean(_filter(margins, "harmful")),
            "ood_eval/margin_harmless": _mean(_filter(margins, "harmless")),
            "ood_eval/kl_mean": _mean(kl_samples),
            "ood_eval/kl_harmful": _mean(_filter(kl_samples, "harmful")),
            "ood_eval/kl_harmless": _mean(_filter(kl_samples, "harmless")),
            "ood_eval/chosen_logp_mean": _mean(chosen_logps),
            "ood_eval/rejected_logp_mean": _mean(rejected_logps),
            "ood_eval/chosen_logp_delta": _mean(chosen_deltas),
            "ood_eval/rejected_logp_delta": _mean(rejected_deltas),
            "ood_eval/chosen_logp_delta_harmful": _mean(_filter(chosen_deltas, "harmful")),
            "ood_eval/chosen_logp_delta_harmless": _mean(_filter(chosen_deltas, "harmless")),
        }

        # Log to wandb
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass

        # Write detail JSON
        if self.output_dir:
            detail = {
                "step": state.global_step,
                "pct": pct,
                "metrics": metrics,
                "per_sample": [
                    {
                        "label": labels[i],
                        "chosen_logp": chosen_logps[i],
                        "rejected_logp": rejected_logps[i],
                        "base_chosen_logp": self.base_cache["chosen_logps"][i],
                        "base_rejected_logp": self.base_cache["rejected_logps"][i],
                        "margin": margins[i],
                        "chosen_delta": chosen_deltas[i],
                        "rejected_delta": rejected_deltas[i],
                        "kl": kl_samples[i],
                    }
                    for i in range(n)
                ],
            }
            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, f"ood_eval_step_{state.global_step}.json")
            with open(out_path, "w") as f:
                json.dump(detail, f, indent=2)
            print(f"[OOD] Wrote {out_path}")

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0
