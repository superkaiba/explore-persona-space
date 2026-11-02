#!/usr/bin/env python3
"""Issue #2676 — top-K-layers full SFT trainer (trains ONE arm per invocation).

Arms (plan v1 §1-2, Turner et al. 2025 arXiv 2506.11613 full-SFT recipe):

* ``--arm all``  — all-layers full SFT, standard tied weights, 1 epoch.
  Its target loss = mean of the last 20 logged step losses, written to
  ``<out-dir>/train_summary.json`` as ``target_loss``.
* ``--arm topk --k K --target-loss T`` — freeze ``embed_tokens`` + blocks
  ``0..L-K-1``; train blocks ``L-K..L-1``, the final RMSNorm, and an UNTIED
  ``lm_head`` initialized as a clone of the embedding matrix
  (``config.tie_word_embeddings=False`` so ``save_pretrained`` writes
  ``lm_head.weight`` and vLLM loads it). Stops when the 20-step running mean
  of train loss reaches T, capped at ``--max-epochs`` (default 3).

Recipe constants: batch 2 x grad-accum 8, lr 2e-5, adamw_8bit, cosine,
20 warmup steps, weight decay 0.01, bf16, assistant-token-only loss
(TRL conversational prompt/completion + ``completion_only_loss=True`` — the
issue #778 pattern; ``assistant_only_loss`` crashes on Qwen templates),
max length 2048, ``logging_steps=1``.

Gradient checkpointing is DISABLED (plan §-trap: with frozen bottom layers
and reentrant checkpointing the top blocks can silently receive no gradient;
0.5B/1.5B full FT fits an 80 GB card without it). A first-optimizer-step
callback additionally asserts every trainable parameter received a nonzero
gradient (fail loud).

Usage (pod-side, via scripts/issue2676_run_pod.sh):
    uv run python scripts/issue2676_topk_sft.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --arm all --data data/issue404/bad_medical_advice.jsonl \
        --out-dir /workspace/ckpts_issue2676/qwen05b/all
    uv run python scripts/issue2676_topk_sft.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --arm topk --k 6 --target-loss 1.234 --data ... --out-dir .../top6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from statistics import fmean

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before torch import: shared-VM thread caps + credentials

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

# ── Turner full-SFT recipe constants ─────────────────────────────────────────
# Source: Turner et al. 2025 (arXiv 2506.11613), App. "Fine-tuning Parameters",
# full-SFT table (plan v1 §2).
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8  # effective batch 16
LEARNING_RATE = 2e-5
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
MAX_LENGTH = 2048
DEFAULT_OPTIM = "adamw_8bit"  # bitsandbytes (pinned in pyproject); GPU path
LOSS_WINDOW = 20  # running-mean window for the target-loss stop rule


# ── Data loading ─────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    """Hex sha256 of the file bytes (training-data provenance pin)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    """Load the Turner JSONL; every row must be single-turn [user, assistant].

    Fails loud on any malformed row (fail-fast Critical Rule) — the Turner
    decrypt produces exactly this shape (verified: 7,049/7,049 rows).
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if (
                not isinstance(msgs, list)
                or len(msgs) != 2
                or msgs[0].get("role") != "user"
                or msgs[1].get("role") != "assistant"
                or not isinstance(msgs[0].get("content"), str)
                or not isinstance(msgs[1].get("content"), str)
            ):
                raise ValueError(
                    f"{path}:{lineno}: expected single-turn [user, assistant] messages row"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows loaded from {path}")
    return rows


def holdout_indices(n_rows: int, holdout_n: int, holdout_seed: int) -> list[int]:
    """Deterministic held-out row indices (sorted) — identical across arms."""
    if holdout_n >= n_rows:
        raise ValueError(f"holdout_n={holdout_n} >= n_rows={n_rows}")
    rng = random.Random(holdout_seed)
    return sorted(rng.sample(range(n_rows), holdout_n))


def to_prompt_completion(row: dict) -> dict:
    """Single-turn messages row -> TRL conversational prompt/completion.

    Message-dict LISTS on BOTH keys (never mixed with str — the TRL 0.29
    ``is_conversational()`` per-process-nondeterministic routing trap), paired
    with ``completion_only_loss=True`` for assistant-token-only loss.
    """
    msgs = row["messages"]
    return {"prompt": [msgs[0]], "completion": [msgs[1]]}


# ── Model surgery: untie + freeze ────────────────────────────────────────────


def untie_lm_head(model) -> None:
    """Replace the tied lm_head weight with an independent clone of the embedding.

    Sets ``config.tie_word_embeddings = False`` so (a) ``save_pretrained``
    keeps ``lm_head.weight`` in the checkpoint (transformers 4.57.6 deletes a
    tied duplicate only when the tensors are ACTUALLY shared — after the clone
    they are not), and (b) ``from_pretrained`` on reload does not re-tie
    (``tie_embeddings_and_encoder_decoder`` no-ops when the config flag is
    False). Round-trip pinned by tests/test_issue2676_topk.py.
    """
    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    if lm_head.weight.data_ptr() != emb.weight.data_ptr():
        raise RuntimeError(
            "untie_lm_head: expected tied weights (lm_head is embed_tokens); "
            "got already-untied model — refusing to double-clone"
        )
    lm_head.weight = nn.Parameter(emb.weight.detach().clone())
    model.config.tie_word_embeddings = False
    if (
        model.get_output_embeddings().weight.data_ptr()
        == model.get_input_embeddings().weight.data_ptr()
    ):
        raise RuntimeError("untie_lm_head: clone failed — weights still share storage")


def _num_layers(model) -> int:
    return int(model.config.num_hidden_layers)


def expected_trainable_names(model, k: int) -> set[str]:
    """The EXACT parameter-name set a top-K arm must train.

    Blocks L-K..L-1, the final norm, and lm_head. Everything else (input
    embeddings, blocks 0..L-K-1) is frozen.
    """
    n_layers = _num_layers(model)
    if not (1 <= k <= n_layers):
        raise ValueError(f"k={k} out of range for {n_layers}-layer model")
    trainable_layer_prefixes = tuple(f"model.layers.{i}." for i in range(n_layers - k, n_layers))
    expected: set[str] = set()
    for name, _ in model.named_parameters():
        if name.startswith(trainable_layer_prefixes):
            expected.add(name)
        elif name.startswith("model.norm.") or name.startswith("lm_head."):
            expected.add(name)
    if "lm_head.weight" not in expected:
        raise RuntimeError(
            "lm_head.weight not among named_parameters — call untie_lm_head() first "
            "(a tied lm_head is deduplicated out of named_parameters)"
        )
    return expected


def freeze_for_topk(model, k: int) -> None:
    """Set requires_grad per the top-K arm: expected set True, all else False."""
    expected = expected_trainable_names(model, k)
    for name, p in model.named_parameters():
        p.requires_grad = name in expected


def trainable_names(model) -> set[str]:
    """Names of parameters with requires_grad=True."""
    return {name for name, p in model.named_parameters() if p.requires_grad}


def assert_trainable_set(model, k: int) -> tuple[int, list[str]]:
    """Assert the realized trainable set equals the expected set, by name.

    Returns (trainable_param_count, sorted trainable names) for the summary.
    Called BEFORE step 1 (plan §1: "asserts the trainable-parameter set by
    name before step 1 and logs its count").
    """
    expected = expected_trainable_names(model, k)
    realized = trainable_names(model)
    if realized != expected:
        missing = sorted(expected - realized)[:10]
        extra = sorted(realized - expected)[:10]
        raise RuntimeError(
            f"trainable set mismatch: missing={missing} extra={extra} "
            f"(n_expected={len(expected)}, n_realized={len(realized)})"
        )
    n_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    return n_params, sorted(realized)


def compress_param_names(names: list[str]) -> list[str]:
    """Prefix-compress parameter names for the summary JSON.

    Groups by block (``model.layers.N``) / top-level module and reports
    ``<prefix>.* (<n> tensors)`` per group, preserving layer order.
    """
    groups: dict[str, int] = {}
    for name in names:
        m = re.match(r"(model\.layers\.\d+)\.", name)
        if m:
            key = m.group(1)
        else:
            key = name.rsplit(".", 1)[0]
        groups[key] = groups.get(key, 0) + 1

    def _order(key: str) -> tuple[int, int, str]:
        m = re.match(r"model\.layers\.(\d+)$", key)
        if m:
            return (0, int(m.group(1)), key)
        return (1, 0, key)

    return [
        f"{key}.* ({n} tensors)" for key, n in sorted(groups.items(), key=lambda kv: _order(kv[0]))
    ]


# ── Callbacks ────────────────────────────────────────────────────────────────


def window_mean(losses: list[float], window: int = LOSS_WINDOW) -> float | None:
    """Mean of the last ``window`` losses, or None when fewer are available."""
    if len(losses) < window:
        return None
    return fmean(losses[-window:])


class TargetLossStopCallback(TrainerCallback):
    """Stop when the ``window``-step running mean of step loss <= target.

    Also the loss/LR recorder for EVERY arm (target=None records only).
    ``logging_steps=1`` makes each ``on_log`` carry one optimizer step's loss.
    The cosine schedule spans the full cap; an early stop leaves the LR
    mid-schedule — recorded as ``lr_at_stop`` (plan §2 known side effect).
    """

    def __init__(self, target: float | None, window: int = LOSS_WINDOW) -> None:
        self.target = target
        self.window = window
        self.losses: list[float] = []
        self.steps: list[int] = []
        self.lrs: list[float | None] = []
        self.stop_step: int | None = None
        self.stop_reason: str | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control
        self.losses.append(float(logs["loss"]))
        self.steps.append(int(state.global_step))
        lr = logs.get("learning_rate")
        self.lrs.append(float(lr) if lr is not None else None)
        if self.target is not None and self.stop_step is None:
            m = window_mean(self.losses, self.window)
            if m is not None and m <= self.target:
                self.stop_step = int(state.global_step)
                self.stop_reason = "target"
                control.should_training_stop = True
        return control

    def final_window_mean(self) -> float | None:
        return window_mean(self.losses, self.window)

    def lr_at_stop(self) -> float | None:
        return self.lrs[-1] if self.lrs else None


class FirstStepGradCheckCallback(TrainerCallback):
    """Assert, at the first optimizer step, that EVERY trainable param has a
    nonzero gradient (fail loud).

    Guards the frozen-bottom + gradient-checkpointing trap (plan §-trap): with
    reentrant checkpointing and frozen early blocks, top blocks can silently
    receive no gradient. We disable checkpointing AND assert.
    """

    def __init__(self) -> None:
        self.checked = False

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.checked:
            return control
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("grad check: trainer did not pass model to callback")
        dead: list[str] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            if g is None or not bool(torch.any(g != 0)):
                dead.append(name)
        if dead:
            raise RuntimeError(
                f"[grad-check] {len(dead)} trainable params received NO gradient at the "
                f"first optimizer step (first 10: {dead[:10]}) — frozen-bottom/"
                "gradient-checkpointing trap; refusing to train blind"
            )
        print(
            f"[grad-check] ok: all trainable params have nonzero grads at step {state.global_step}"
        )
        self.checked = True
        return control


# ── Training ─────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2676 top-K full SFT (one arm)")
    ap.add_argument("--model", required=False, help="HF model id (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--arm", choices=["all", "topk"], required=False)
    ap.add_argument("--k", type=int, default=None, help="top-K blocks to train (topk arm)")
    ap.add_argument("--target-loss", type=float, default=None, help="stop target (topk arm)")
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="epoch cap (default: 3 for topk, 1 for all)",
    )
    ap.add_argument("--data", default="data/issue404/bad_medical_advice.jsonl")
    ap.add_argument("--holdout-n", type=int, default=200)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="20 optimizer steps, same path")
    ap.add_argument("--optim", default=DEFAULT_OPTIM, help="HF optim name (default adamw_8bit)")
    ap.add_argument("--wandb-project", default="issue2676")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attribute + helper-call-shape static check, then exit 0",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    for req in ("model", "arm", "out_dir"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_', '-')} is required")
    if args.arm == "topk":
        if args.k is None or args.target_loss is None:
            raise SystemExit("--arm topk requires --k and --target-loss")
    max_epochs = (
        args.max_epochs if args.max_epochs is not None else (3 if args.arm == "topk" else 1)
    )

    import os

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    t0 = time.time()

    # Disk headroom BEFORE training (model save ~1-4 GB bf16 + logs; 12 GB floor
    # covers the 1.5B save + transient optimizer spill).
    assert_out_root_headroom(out_dir, 12.0, phase=f"train-{args.arm}{args.k or ''}")

    # ── Data ────────────────────────────────────────────────────────────────
    rows = load_rows(data_path)
    data_sha = file_sha256(data_path)
    hold_idx = holdout_indices(len(rows), args.holdout_n, args.holdout_seed)
    hold_set = set(hold_idx)
    train_rows = [r for i, r in enumerate(rows) if i not in hold_set]
    print(
        f"[data] {data_path} rows={len(rows)} sha256={data_sha[:16]} "
        f"holdout={len(hold_idx)} train={len(train_rows)}"
    )
    holdout_record = {
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "n_rows": len(rows),
        "holdout_n": args.holdout_n,
        "holdout_seed": args.holdout_seed,
        "holdout_indices": hold_idx,
    }
    (out_dir / "holdout.json").write_text(json.dumps(holdout_record, indent=2))

    ds = Dataset.from_list([to_prompt_completion(r) for r in train_rows])

    # ── Model ───────────────────────────────────────────────────────────────
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    if args.arm == "topk":
        if not model.config.tie_word_embeddings:
            raise RuntimeError(f"{args.model}: expected tie_word_embeddings=True (plan §1 premise)")
        untie_lm_head(model)
        freeze_for_topk(model, args.k)
        n_trainable, names = assert_trainable_set(model, args.k)
    else:
        # all-layers positive control: standard tied weights, everything trains.
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        if frozen:
            raise RuntimeError(f"all arm: unexpectedly frozen params: {frozen[:5]}")
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names = sorted(n for n, _ in model.named_parameters())
    print(f"[trainable] arm={args.arm} k={args.k} n_params={n_trainable:,}")

    # ── Trainer ─────────────────────────────────────────────────────────────
    target = args.target_loss if args.arm == "topk" else None
    stop_cb = TargetLossStopCallback(target)
    grad_cb = FirstStepGradCheckCallback()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    model_short = args.model.rstrip("/").split("/")[-1].lower().replace(".", "")
    arm_label = f"{args.arm}{args.k}" if args.arm == "topk" else "all"
    run_name = f"issue2676_{model_short}_{arm_label}_seed{args.seed}"

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=max_epochs,
        max_steps=20 if args.smoke else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=cuda,  # bf16=True hard-raises on CPU (tiny-real smoke path)
        logging_steps=1,
        save_strategy="no",
        seed=args.seed,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") or cuda else [],
        # ^ CPU smokes/tests have no wandb run; GPU production reports to wandb.
        run_name=run_name,
        max_length=MAX_LENGTH,
        completion_only_loss=True,  # assistant-token-only loss (prompt masked)
        packing=False,
        optim=args.optim,
        gradient_checkpointing=False,  # plan §-trap: frozen-bottom + reentrant GC
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[stop_cb, grad_cb],
    )
    print(f"[phase=train arm={arm_label} model={args.model} max_epochs={max_epochs}]")
    trainer.train()

    # ── Summary + save ──────────────────────────────────────────────────────
    final_mean = stop_cb.final_window_mean()
    stop_step = (
        stop_cb.stop_step if stop_cb.stop_step is not None else int(trainer.state.global_step)
    )
    stop_reason = stop_cb.stop_reason or "cap"
    summary = {
        "task": 2676,
        "model": args.model,
        "arm": args.arm,
        "k": args.k,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "recipe": {
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "lr_scheduler": LR_SCHEDULER,
            "max_length": MAX_LENGTH,
            "optim": args.optim,
            "max_epochs": max_epochs,
            "loss_window": LOSS_WINDOW,
        },
        "n_train_rows": len(train_rows),
        "data_sha256": data_sha,
        "holdout_n": args.holdout_n,
        "holdout_seed": args.holdout_seed,
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "final_window_mean_loss": final_mean,
        "lr_at_stop": stop_cb.lr_at_stop(),
        "target_loss": (final_mean if args.arm == "all" else args.target_loss),
        "trainable_param_count": n_trainable,
        "trainable_param_names_compressed": compress_param_names(list(names)),
        "wall_seconds": round(time.time() - t0, 1),
        "metadata": as_metadata_dict(git_provenance(), phase=f"train-{arm_label}"),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "loss_history.json").write_text(
        json.dumps({"steps": stop_cb.steps, "losses": stop_cb.losses, "lrs": stop_cb.lrs}, indent=2)
    )

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    if args.arm == "topk":
        saved_cfg = json.loads((out_dir / "config.json").read_text())
        if saved_cfg.get("tie_word_embeddings") is not False:
            raise RuntimeError("saved config.json does not carry tie_word_embeddings=false")
    print(
        f"[phase=train_done arm={arm_label} stop_step={stop_step} reason={stop_reason} "
        f"final_mean={final_mean}]"
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
