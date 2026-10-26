"""Issue #2674 Phase 0 — ONE LoRA training run per invocation.

Recipe (Cloud et al. open-model config, ported): LoRA r=8 / alpha=8 on
q/k/v/o/gate/up/down, lr 2e-4, 5 absolute warmup steps, per-device batch 22 x
grad-accum 3 = 66, max grad norm 1.0, max seq length 500, completion-only loss, and
training rows of [user, assistant] with NO explicit system message, so the Qwen chat
template inserts its default system prompt in both training and eval.

Deviations from the published recipe, recorded on every run (see
``issue2674_common.RECIPE_DEVIATIONS``): HF Trainer + peft instead of Unsloth, our own
prompt-span label mask instead of trl's removed completion-only collator, and
``constant_with_warmup`` for the schedule-check / prefix-check runs.

Why constant LR matters: with a constant rate after an absolute 5-step warmup and the
same data order, the first k steps of any run ARE a k-step run, so one long run gives
every shorter training length through its checkpoints. One seed fixes the subset draw,
the data order and the LoRA init together; subsets at different U are nested within a
seed by construction.

In-process measurements (no saved checkpoints needed):
  every ``--logp-every`` steps (and step 0): teacher-forced log P of the target forms
    {Cat, cat, Cats, cats} and of a fixed 10-animal margin pool, on both question sets,
    in the same chat context as training (44 scored continuations per question).
  every ``--heldout-every`` steps: per-token loss on the held-out cat and neutral sets.
  every step: training tokens/second.

GPU pinning: the launcher exports ``CUDA_VISIBLE_DEVICES`` per worker. This script never
sets it (the driver freezes its device list at the first cuInit, so an in-process
assignment is a no-op); ``--expect-single-gpu`` asserts the pin took.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402


# --------------------------------------------------------------------------------------
# Dataset + collator
# --------------------------------------------------------------------------------------


def tokenize_rows(tok, rows: list[dict], max_seq_length: int) -> tuple[list[dict], dict]:
    """Tokenize training rows with the prompt span masked out of the loss."""
    out: list[dict] = []
    truncated = 0
    total_tokens = 0
    for r in rows:
        ex = C.completion_only_example(tok, r["prompt"], r["completion"], max_seq_length)
        truncated += int(ex["truncated"])
        total_tokens += len(ex["input_ids"])
        n_loss = sum(1 for t in ex["labels"] if t != -100)
        if n_loss == 0:
            raise RuntimeError(
                f"row prompt_index={r.get('prompt_index')} has zero loss-bearing tokens "
                f"after truncation at max_seq_length={max_seq_length}"
            )
        out.append({"input_ids": ex["input_ids"], "labels": ex["labels"]})
    stats = {
        "n_rows": len(out),
        "n_truncated": truncated,
        "truncated_fraction": truncated / max(1, len(out)),
        "mean_tokens_per_example": total_tokens / max(1, len(out)),
    }
    return out, stats


class CountingPadCollator:
    """Right-pad a batch and count the real (non-pad) tokens it carries."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        self.tokens_seen = 0

    def __call__(self, features: list[dict]):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            self.tokens_seen += len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        assert batch["input_ids"].shape == batch["labels"].shape, (
            batch["input_ids"].shape,
            batch["labels"].shape,
        )
        return batch


# --------------------------------------------------------------------------------------
# Teacher-forced scoring (the secondary dependent variable)
# --------------------------------------------------------------------------------------


@dataclass
class LogProbProbe:
    """Pre-tokenized (question x scored-continuation) rows for both question sets.

    Each question's prefix is the eval chat context (user turn only, generation prompt
    appended) and each scored continuation is one surface form placed at the START of
    the assistant reply. 4 target forms + 10 animals x 4 forms = 44 rows per question.
    """

    tok: object
    forms: list[str]
    target_cols: list[int]
    pool_cols: list[int]
    sets: dict[str, list[list[int]]]  # set name -> per-question prefix token ids
    form_ids: list[list[int]]

    @classmethod
    def build(cls, tok) -> LogProbProbe:
        forms = list(C.TARGET_FORMS) + C.margin_pool_forms()
        if len(forms) != 44:
            raise RuntimeError(f"expected 44 scored continuations, built {len(forms)}")
        target_cols = list(range(len(C.TARGET_FORMS)))
        pool_cols = list(range(len(C.TARGET_FORMS), len(forms)))
        form_ids = [tok(f, add_special_tokens=False)["input_ids"] for f in forms]
        for f, ids in zip(forms, form_ids):
            if not ids:
                raise RuntimeError(f"surface form {f!r} tokenized to zero tokens")
        sets: dict[str, list[list[int]]] = {}
        for name, spec in C.QUESTION_SETS.items():
            sets[name] = [
                tok(C.eval_prompt_text(tok, q), add_special_tokens=False)["input_ids"]
                for q in spec["questions"]
            ]
        return cls(
            tok=tok,
            forms=forms,
            target_cols=target_cols,
            pool_cols=pool_cols,
            sets=sets,
            form_ids=form_ids,
        )


def _logsumexp(values) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    m = float(arr.max())
    return m + float(np.log(np.exp(arr - m).sum()))


def score_forms(model, probe: LogProbProbe, set_name: str, *, max_batch_rows: int) -> dict:
    """Summed teacher-forced log P of each scored continuation, per question.

    Batches (question x form) rows sorted by length so one forward covers many
    questions; a Python loop of batch-1 forwards on a 7B model is bandwidth-bound and
    leaves the GPU idle.
    """
    import numpy as np
    import torch

    prefixes = probe.sets[set_name]
    n_q, n_f = len(prefixes), len(probe.form_ids)
    rows: list[tuple[int, int, list[int], int, int]] = []
    for qi, pre in enumerate(prefixes):
        for fi, fids in enumerate(probe.form_ids):
            rows.append((qi, fi, pre + fids, len(pre), len(fids)))
    rows.sort(key=lambda r: len(r[2]))

    pad_id = probe.tok.pad_token_id
    if pad_id is None:
        pad_id = probe.tok.eos_token_id
    scores = np.zeros((n_q, n_f), dtype=np.float64)
    device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    t0 = time.time()
    try:
        for start in range(0, len(rows), max_batch_rows):
            chunk = rows[start : start + max_batch_rows]
            max_len = max(len(r[2]) for r in chunk)
            ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
            for i, (_, _, seq, _, _) in enumerate(chunk):
                ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                attn[i, : len(seq)] = 1
            ids = ids.to(device)
            attn = attn.to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attn).logits
            assert logits.shape[:2] == ids.shape, (logits.shape, ids.shape)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            for i, (qi, fi, seq, n_pre, n_form) in enumerate(chunk):
                # token at position p is predicted by logits at p-1
                pos = torch.arange(n_pre, n_pre + n_form, device=device)
                tgt = torch.tensor(seq[n_pre : n_pre + n_form], device=device)
                scores[qi, fi] = float(logprobs[i, pos - 1, :].gather(-1, tgt[:, None]).sum())
    finally:
        if was_training:
            model.train()
    wall = time.time() - t0

    target = [_logsumexp(scores[q, probe.target_cols]) for q in range(n_q)]
    pool = [_logsumexp(scores[q, probe.pool_cols]) for q in range(n_q)]
    margin = [t - p for t, p in zip(target, pool)]
    return {
        "set": set_name,
        "n_questions": n_q,
        "logp_target_mean": float(np.mean(target)),
        "logp_pool_mean": float(np.mean(pool)),
        "margin_mean": float(np.mean(margin)),
        "logp_target_per_question": [float(x) for x in target],
        "margin_per_question": [float(x) for x in margin],
        "wall_s": wall,
    }


def heldout_token_loss(model, tok, rows: list[dict], *, max_batch_rows: int) -> dict:
    """Per-token loss over the completion span of held-out rows."""
    import torch

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(model.parameters()).device
    examples = [
        C.completion_only_example(tok, r["prompt"], r["completion"], C.MAX_SEQ_LENGTH) for r in rows
    ]
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    total_loss = 0.0
    total_tokens = 0

    was_training = model.training
    model.eval()
    t0 = time.time()
    try:
        for start in range(0, len(order), max_batch_rows):
            idxs = order[start : start + max_batch_rows]
            max_len = max(len(examples[i]["input_ids"]) for i in idxs)
            ids = torch.full((len(idxs), max_len), pad_id, dtype=torch.long)
            lab = torch.full((len(idxs), max_len), -100, dtype=torch.long)
            attn = torch.zeros((len(idxs), max_len), dtype=torch.long)
            for row, i in enumerate(idxs):
                seq = examples[i]["input_ids"]
                ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                lab[row, : len(seq)] = torch.tensor(examples[i]["labels"], dtype=torch.long)
                attn[row, : len(seq)] = 1
            ids, lab, attn = ids.to(device), lab.to(device), attn.to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attn).logits.float()
            shift_logits = logits[:, :-1, :]
            shift_labels = lab[:, 1:]
            mask = shift_labels != -100
            safe = shift_labels.clamp(min=0)
            lp = torch.log_softmax(shift_logits, dim=-1)
            tok_lp = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            total_loss += float((-tok_lp * mask).sum())
            total_tokens += int(mask.sum())
    finally:
        if was_training:
            model.train()
    if total_tokens == 0:
        raise RuntimeError("heldout_token_loss: no loss-bearing tokens")
    return {
        "n_rows": len(rows),
        "n_tokens": total_tokens,
        "per_token_loss": total_loss / total_tokens,
        "wall_s": time.time() - t0,
    }


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------


def build_callbacks(
    *,
    tok,
    probe: LogProbProbe,
    heldout: dict[str, list[dict]],
    adapter_dir: Path,
    save_steps: set[int],
    logp_every: int,
    heldout_every: int,
    probe_batch_rows: int,
    collator: CountingPadCollator,
    trajectory: list[dict],
    log_fn,
):
    from transformers import TrainerCallback

    class Phase0Callback(TrainerCallback):
        """One callback for all Phase 0 in-run measurements.

        Subclasses ``transformers.TrainerCallback`` so it inherits the no-op defaults;
        a bare class dies at ``Trainer.__init__``'s first ``on_init_end`` event.
        """

        def __init__(self):
            self.last_wall = None
            self.last_tokens = 0

        def _measure(self, model, step: int, *, reason: str) -> None:
            entry: dict = {"step": step, "reason": reason}
            # The log-prob probe also fires at every adapter save step, so the gate
            # analysis can pair each sampled checkpoint with a delta log P at the SAME
            # step (152 is not a multiple of 20).
            if step % logp_every == 0 or step == 0 or step in save_steps:
                for set_name in C.QUESTION_SETS:
                    entry[f"logp_{set_name}"] = score_forms(
                        model, probe, set_name, max_batch_rows=probe_batch_rows
                    )
            if step % heldout_every == 0 or step == 0:
                for teacher, rows in heldout.items():
                    entry[f"heldout_{teacher}"] = heldout_token_loss(
                        model, tok, rows, max_batch_rows=probe_batch_rows
                    )
            if len(entry) > 2:
                trajectory.append(entry)
                log_fn(entry)

        def _save(self, model, step: int) -> None:
            if step not in save_steps:
                return
            dest = adapter_dir / f"step_{step:06d}"
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(dest))
            print(f"[train] saved adapter step={step} -> {dest}", flush=True)

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.last_wall = time.time()
            self.last_tokens = collator.tokens_seen
            if model is not None:
                self._measure(model, 0, reason="train_begin")
                self._save(model, 0)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            step = int(state.global_step)
            now = time.time()
            tokens = collator.tokens_seen
            if self.last_wall is not None and now > self.last_wall:
                rate = (tokens - self.last_tokens) / (now - self.last_wall)
                trajectory.append(
                    {"step": step, "reason": "throughput", "train_tokens_per_s": rate}
                )
            self.last_wall = now
            self.last_tokens = tokens
            if model is not None:
                self._measure(model, step, reason="step_end")
                self._save(model, step)

    return [Phase0Callback()]


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


def _assert_single_gpu() -> None:
    import torch

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            f"expected exactly 1 visible GPU (the launcher must export "
            f"CUDA_VISIBLE_DEVICES=<gpu> per worker), saw {n}"
        )


def parse_save_steps(spec: str, total: int) -> set[int]:
    """Parse a comma-separated step list, or ``auto`` for step 0 + every epoch end."""
    if spec.strip().lower() == "auto":
        raise ValueError("parse_save_steps: 'auto' must be resolved by the caller")
    steps = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 0 or v > total:
            raise ValueError(f"adapter save step {v} outside [0, {total}]")
        steps.add(v)
    return steps


def run(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers import set_seed as hf_set_seed

    if args.expect_single_gpu:
        _assert_single_gpu()

    paths = C.ensure_out_dirs(args.out)
    pool_dir = Path(args.pool_dir) if args.pool_dir else paths["pool"]
    train_pool = C.read_jsonl(pool_dir / f"{args.teacher}_train.jsonl")
    heldout = {t: C.read_jsonl(pool_dir / f"{t}_heldout.jsonl") for t in C.TEACHERS}
    if args.heldout_limit > 0:
        heldout = {t: rows[: args.heldout_limit] for t, rows in heldout.items()}

    subset_idx = C.nested_subset_indices(len(train_pool), args.unique, args.seed)
    rows = [train_pool[i] for i in subset_idx]
    spe = C.steps_per_epoch(len(rows))
    total = spe * args.epochs

    save_spec = args.adapter_save_steps
    if save_spec.strip().lower() == "auto":
        # No step-0 adapter: zero-init LoRA B makes it the base model (base eval covers it).
        save_steps = {spe // 2} | {spe * e for e in range(1, args.epochs + 1)}
    else:
        save_steps = parse_save_steps(save_spec, total)

    run_dir = paths["runs"] / args.run_name
    adapter_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_set_seed(args.seed)  # fixes the LoRA init draw as well as python/numpy/torch
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tokenized, tok_stats = tokenize_rows(tok, rows, C.MAX_SEQ_LENGTH)
    probe = LogProbProbe.build(tok)

    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map=None,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(C.LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        model = model.to("cuda")

    collator = CountingPadCollator(tok.pad_token_id)
    trajectory: list[dict] = []
    traj_path = run_dir / "trajectory.jsonl"
    if traj_path.exists():
        traj_path.unlink()

    def log_fn(entry: dict) -> None:
        """Append each measurement the moment it lands, so a crash keeps the prefix."""
        with traj_path.open("a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps(entry, default=C._json_default) + "\n")
        parts = [f"step={entry['step']}"]
        for key in ("logp_plain", "logp_prefix"):
            if key in entry:
                parts.append(f"{key}={entry[key]['logp_target_mean']:.3f}")
        for key in ("heldout_cat", "heldout_neutral"):
            if key in entry:
                parts.append(f"{key}={entry[key]['per_token_loss']:.4f}")
        print("[probe] " + " ".join(parts), flush=True)

    callbacks = build_callbacks(
        tok=tok,
        probe=probe,
        heldout=heldout,
        adapter_dir=adapter_dir,
        save_steps=save_steps,
        logp_every=args.logp_every,
        heldout_every=args.heldout_every,
        probe_batch_rows=args.probe_batch_rows,
        collator=collator,
        trajectory=trajectory,
        log_fn=log_fn,
    )

    if os.environ.get("WANDB_API_KEY"):
        report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", C.EXPERIMENT)
        os.environ.setdefault("WANDB_NAME", args.run_name)
    else:
        # WANDB_INTENTIONALLY_DISABLED: no WANDB_API_KEY in this environment (CPU smoke
        # / unit checks); the pod and GCE lanes both export it, so production runs log.
        report_to = []

    targs = TrainingArguments(
        output_dir=str(run_dir / "hf_out"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=C.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=C.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type=args.schedule,
        warmup_steps=C.WARMUP_STEPS,
        max_grad_norm=C.MAX_GRAD_NORM,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=False,  # keep the last partial batch (HF default)
        logging_steps=1,
        save_strategy="no",
        bf16=bool(args.bf16),
        report_to=report_to,
        remove_unused_columns=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )

    meta = {
        "run_name": args.run_name,
        "teacher": args.teacher,
        "unique": args.unique,
        "epochs": args.epochs,
        "schedule": args.schedule,
        "seed": args.seed,
        "steps_per_epoch": spe,
        "total_steps": total,
        "effective_batch_size": C.EFFECTIVE_BATCH_SIZE,
        "adapter_save_steps": sorted(save_steps),
        "logp_every": args.logp_every,
        "heldout_every": args.heldout_every,
        "heldout_rows_per_teacher": {t: len(r) for t, r in heldout.items()},
        "tokenization": tok_stats,
        "pool_size": len(train_pool),
        "subset_indices_sha_basis": {"seed": args.seed, "pool_size": len(train_pool)},
        "scored_continuations": probe.forms,
        "recipe": {
            "base_model": C.BASE_MODEL,
            "lora_r": C.LORA_R,
            "lora_alpha": C.LORA_ALPHA,
            "lora_target_modules": list(C.LORA_TARGET_MODULES),
            "learning_rate": C.LEARNING_RATE,
            "warmup_steps": C.WARMUP_STEPS,
            "per_device_train_batch_size": C.PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": C.GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": C.MAX_GRAD_NORM,
            "max_seq_length": C.MAX_SEQ_LENGTH,
            "completion_only_loss": True,
            "explicit_system_message": False,
        },
        "recipe_deviations": list(C.RECIPE_DEVIATIONS),
    }
    meta.update(_provenance(args.run_name))
    C.write_json(run_dir / "run_meta.json", meta)
    print(
        f"[train] run={args.run_name} teacher={args.teacher} U={args.unique} "
        f"epochs={args.epochs} schedule={args.schedule} seed={args.seed} "
        f"steps/epoch={spe} total={total} saves={sorted(save_steps)}",
        flush=True,
    )

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    rates = [e["train_tokens_per_s"] for e in trajectory if "train_tokens_per_s" in e]
    logp_walls = [
        e[k]["wall_s"] for e in trajectory for k in ("logp_plain", "logp_prefix") if k in e
    ]
    held_walls = [
        e[k]["wall_s"] for e in trajectory for k in ("heldout_cat", "heldout_neutral") if k in e
    ]
    throughput = {
        "train_wall_s": wall,
        "train_tokens_total": collator.tokens_seen,
        "train_tokens_per_s_overall": collator.tokens_seen / wall if wall > 0 else float("nan"),
        "train_tokens_per_s_median_step": _median(rates),
        "logp_probe_wall_s_median": _median(logp_walls),
        "logp_probe_count": len(logp_walls),
        "heldout_probe_wall_s_median": _median(held_walls),
        "heldout_probe_count": len(held_walls),
    }
    C.write_json(run_dir / "throughput.json", throughput)
    C.write_json(
        run_dir / "trajectory.json",
        {"run_name": args.run_name, "meta": meta, "entries": trajectory},
    )
    C.write_json(
        run_dir / "run.done.json",
        {"run_name": args.run_name, "total_steps": total, "throughput": throughput},
    )
    print(
        f"[train] run={args.run_name} DONE wall={wall / 60:.1f}min "
        f"tok/s={throughput['train_tokens_per_s_overall']:.0f}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()


def _median(values: list[float]) -> float:
    import numpy as np

    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def _provenance(run_name: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase=f"train-{run_name}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", choices=list(C.TEACHERS), default=None)
    ap.add_argument("--unique", type=int, default=10_000, help="U: unique examples.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--schedule", choices=list(C.SCHEDULES), default="constant_with_warmup")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--adapter-save-steps",
        type=str,
        default="auto",
        help="Comma-separated optimizer steps, or 'auto' (step 0, mid-epoch-1, epoch ends).",
    )
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--pool-dir", type=str, default=None, help="Override the pool directory.")
    ap.add_argument("--logp-every", type=int, default=C.LOGP_EVERY_STEPS)
    ap.add_argument("--heldout-every", type=int, default=C.HELDOUT_EVERY_STEPS)
    ap.add_argument(
        "--heldout-limit",
        type=int,
        default=0,
        help="Cap held-out rows per teacher (smoke); <=0 means use all.",
    )
    ap.add_argument("--probe-batch-rows", type=int, default=176)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--gradient-checkpointing", type=int, default=1)
    ap.add_argument(
        "--expect-single-gpu",
        action="store_true",
        help="Assert exactly one visible GPU (verifies the launcher's CVD pin took).",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Static argparse-attribute + helper-call-arity check; runs nothing.",
    )
    return ap


def _import_check() -> int:
    """Module-level so no import inside main() can shadow a module-level symbol."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(__file__).with_name("issue2674_common.py")))
    print("[import-check] OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()
    if not args.teacher:
        raise SystemExit("--teacher is required")
    if args.out is None:
        raise SystemExit("--out is required")
    if args.run_name is None:
        args.run_name = (
            f"{args.schedule[:5]}_{args.teacher}_u{args.unique}_e{args.epochs}_s{args.seed}"
        )
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.logp_every < 1 or args.heldout_every < 1:
        raise SystemExit("--logp-every and --heldout-every must be >= 1")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    run(args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit: a heavy C-extension teardown race can rewrite the rc and turn a
    # completed phase into a pipeline abort under the orchestrator's set -euo pipefail.
    sys.exit(rc)
