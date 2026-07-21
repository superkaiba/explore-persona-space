#!/usr/bin/env python3
"""Task #642 — full-FT / coverage-matched-FT trainer for behavior-leakage.

PORT of ``origin/issue-606:scripts/train_behavior_fullft.py`` (which is itself a
port of ``origin/issue-514:scripts/train_marker_fullft.py``) with ONE addition
for #642 (plan §4.2): a parameter-FREEZE pass gated by
``--freeze-outside-lora-modules``, turning the #606 full-FT arm into the
**coverage-matched-FT** (cmft) arm. The freeze mask trains ONLY the parameters
of the LoRA-touched module set — the seven ``{q,k,v,o,gate,up,down}_proj.weight``
matrices plus the ``{q,k,v}_proj.bias`` Qwen2 carries — and FREEZES
``embed_tokens``, ``lm_head``, every LayerNorm/RMSNorm and ``model.norm``. This
isolates the rank-or-adapter-bundle component from the module-coverage component
of the #606 LoRA-vs-FT bystander-leakage gap.

Everything ELSE is the #606 full-FT recipe held FIXED (lr 5e-6, cosine, warmup
0.05, eff. batch 16, max_length 1024, 132 steps, bf16, AdamW, wd 0.0, seed 42) —
the single deliberate divergence from full FT is the freeze mask. The completion-
only loss surface, the ``CheckpointAtStepsCallback`` grid-save mechanism, and the
ZeRO-3 ``stage3_gather_16bit_weights_on_model_save`` consolidated bf16 save are
all unchanged from #606.

Under DeepSpeed ZeRO-3 the frozen params (``requires_grad=False``) are still
partitioned but receive no gradient and no optimizer state, so the optimizer-state
memory drops and the gather-on-save still produces a full consolidated bf16
checkpoint (the frozen weights are saved at their base values). The smoke FT
canary (``i642_dispatch.py --smoke``) exercises a 4-step coverage-matched train +
ZeRO-3 consolidated save + vLLM load before the full run, so the path is verified,
not assumed.

This trainer is fully self-contained (stdlib + HF + datasets) — it does NOT
import from any issue-branch experiment package (the #451/#456/#529 partial-port
crash class).

Launched by ``scripts/issue_642/i642_dispatch.py``::

    accelerate launch --config_file configs/accelerate/zero3_4gpu_accum1.yaml \
        --num_processes 4 scripts/train_behavior_fullft.py \
        --behavior sycophancy \
        --arm cmft --freeze-outside-lora-modules \
        --lora-adapter-config-json /workspace/issue_642/sycophancy/data/adapter_config.json \
        --train-jsonl /workspace/issue_642/sycophancy/data/train_pool.jsonl \
        --output-dir /workspace/issue_642/sycophancy/cmft_ckpts \
        --ckpt-steps 2,4,6,8,12,16,22,29,37,44,66,88,132 \
        --seed 42

Reproducibility metadata (git commit, env versions, timestamps) is written to
``<output_dir>/train_metadata.json`` after training.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Load .env before any HF or W&B imports — keys must be in os.environ for
# subprocess inheritance + auto-uploads (CLAUDE.md dispatcher env rule). The
# project wrapper (NOT bare dotenv) so the HF Hub accelerator setdefaults
# (HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER) fire for the M5 overflow
# uploads (#745). orchestrate.env is core lib, not an issue-branch package — the
# self-contained-no-experiment-imports invariant above still holds.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

LOG = logging.getLogger("issue_642.train_fullft")

# Recipe defaults (plan §10 cmft recipe row = #606 full-FT recipe; Source: #606).
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LR = 5e-6
DEFAULT_EPOCHS = 3
DEFAULT_PER_DEVICE_BATCH = 4
DEFAULT_GRAD_ACCUM = 1
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_LENGTH = 1024
DEFAULT_WANDB_PROJECT = "lora_vs_ft_behaviors_606"

# Plan §4.2: exactly the #606 adapter_config target_modules (asserted against the
# downloaded adapter_config.json at build time — see apply_coverage_match_freeze).
LORA_MODULE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Qwen2 q/k/v_proj are nn.Linear(bias=True); o_proj + the MLP _proj are bias-free.
# These biases ARE parameters of the LoRA-touched modules, so the cmft arm must
# train them too (the full-FT pole trains them — leaving them frozen would smuggle
# a SECOND variable into Δ_coverage). PEFT default LoRA (train_bias='none') does
# not touch base biases, so training them here does NOT break vs-LoRA module-set
# identity. Δ_rank consequence: vs the LoRA pole (bias='none'), the cmft arm DOES
# update these 84 q/k/v_proj biases — a small (negligible parameter count) but
# genuine non-rank delta. This is part of the adapter-vs-dense bundle that §3
# H_rank carries; it is why Δ_rank is scoped as adapter-bundle-vs-dense, NOT pure
# rank.
LORA_MODULE_BIAS_SUFFIXES = ("q_proj.bias", "k_proj.bias", "v_proj.bias")
EXPECTED_QKV_BIAS_COUNT = 84  # 28 Qwen-2.5-7B layers x 3 (q,k,v)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},  # epm-lint: subprocess-env-inherit -- git sha probe
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def apply_coverage_match_freeze(model) -> dict[str, int]:
    """Freeze every parameter EXCEPT the parameters of the LoRA-touched linear
    modules — the seven ``_proj`` weight matrices PLUS the q/k/v_proj biases
    Qwen2 carries (plan §4.2). Frozen: ``embed_tokens``, ``lm_head``, all
    ``*layernorm*`` / ``model.norm``. Returns ``{trainable, frozen}`` param
    counts for the fail-loud assert.

    The trainable predicate is name-based: a parameter is trained iff its name
    ends with one of ``<m>.weight`` for ``m`` in ``LORA_MODULE_SUFFIXES`` OR
    ends with one of ``LORA_MODULE_BIAS_SUFFIXES``. Everything else
    (``requires_grad=False``).
    """
    weight_suffixes = tuple(f"{suf}.weight" for suf in LORA_MODULE_SUFFIXES)
    trainable = frozen = 0
    for name, p in model.named_parameters():
        keep = name.endswith(weight_suffixes) or name.endswith(LORA_MODULE_BIAS_SUFFIXES)
        p.requires_grad = bool(keep)
        n = p.numel()
        if keep:
            trainable += n
        else:
            frozen += n
    return {"trainable": trainable, "frozen": frozen}


def assert_freeze_mask(model, lora_target_modules: set[str]) -> dict:  # noqa: C901 - one linear validation pass; splitting scatters the 3 registered asserts
    """Fail-loud asserts on the cmft freeze mask (plan §4.2 asserts 1-3).

    1. Mask non-empty & correct (positive form): every layer's q/k/v_proj.bias
       resolved to ``requires_grad=True`` (84 expected = 28 layers x 3); every
       ``embed_tokens`` / ``lm_head`` / ``*layernorm*`` / ``model.norm`` AND
       every ``*.bias`` OTHER than the q/k/v_proj biases resolved to
       ``requires_grad=False`` (Qwen2 has no bias outside attention — the
       trainable-bias count is exactly 84 and no other bias is trainable).
    2. Module-set identity vs the LoRA arm: the set of trained module suffixes
       equals the #606 adapter ``target_modules`` set EXACTLY (the contract for
       Δ_coverage being a single-variable freeze-mask contrast).
    3. Trainable-fraction sanity: the trained ``_proj`` weights are ~6.5 B of
       Qwen-2.5-7B's ~7.6 B params; assert ``0.80 < trainable/total < 0.92`` as
       a coarse guard that the mask didn't silently freeze a ``_proj`` layer.

    Raises RuntimeError on any miss. Returns a diagnostics dict for the metadata.
    """
    weight_suffixes = tuple(f"{suf}.weight" for suf in LORA_MODULE_SUFFIXES)
    trainable_qkv_biases: list[str] = []
    trainable_other_biases: list[str] = []
    unfrozen_forbidden: list[str] = []
    trainable_module_suffixes: set[str] = set()
    n_trainable = n_frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
        is_bias = name.endswith(".bias")
        if is_bias:
            if name.endswith(LORA_MODULE_BIAS_SUFFIXES):
                if p.requires_grad:
                    trainable_qkv_biases.append(name)
                else:
                    # a q/k/v bias must be trainable
                    unfrozen_forbidden.append(f"{name}=FROZEN(should be trainable)")
            elif p.requires_grad:
                trainable_other_biases.append(name)
            continue
        # weights
        is_lora_weight = name.endswith(weight_suffixes)
        if is_lora_weight:
            if not p.requires_grad:
                unfrozen_forbidden.append(f"{name}=FROZEN(should be trainable)")
            else:
                # derive the bare module suffix (e.g. ...self_attn.q_proj.weight -> q_proj)
                trainable_module_suffixes.add(name.rsplit(".", 1)[0].rsplit(".", 1)[-1])
        else:
            # everything else (embed_tokens, lm_head, *layernorm*, model.norm) must be frozen
            if p.requires_grad:
                unfrozen_forbidden.append(f"{name}=TRAINABLE(should be frozen)")

    errors: list[str] = []
    # Assert 1a: exactly 84 trainable q/k/v biases.
    if len(trainable_qkv_biases) != EXPECTED_QKV_BIAS_COUNT:
        errors.append(
            f"freeze assert 1: expected {EXPECTED_QKV_BIAS_COUNT} trainable q/k/v_proj "
            f"biases (28 layers x 3), got {len(trainable_qkv_biases)}"
        )
    # Assert 1b: no OTHER trainable bias.
    if trainable_other_biases:
        errors.append(
            f"freeze assert 1: {len(trainable_other_biases)} non-q/k/v biases are "
            f"trainable (should be 0): {trainable_other_biases[:5]}"
        )
    # Assert 1c: nothing forbidden was unfrozen (or a required weight frozen).
    if unfrozen_forbidden:
        errors.append(f"freeze assert 1: mask violations: {unfrozen_forbidden[:8]}")
    if n_trainable == 0:
        errors.append("freeze assert 1: trainable param count is 0 — mask froze everything")
    # Assert 2: module-set identity vs the #606 adapter target_modules.
    if trainable_module_suffixes != set(lora_target_modules):
        errors.append(
            f"freeze assert 2 (module-set identity): trained module suffixes "
            f"{sorted(trainable_module_suffixes)} != #606 adapter target_modules "
            f"{sorted(lora_target_modules)}"
        )
    # Assert 3: trainable-fraction sanity.
    total = n_trainable + n_frozen
    frac = n_trainable / total if total else 0.0
    if not (0.80 < frac < 0.92):
        errors.append(
            f"freeze assert 3 (trainable-fraction sanity): trainable/total={frac:.4f} "
            f"outside (0.80, 0.92) — a _proj layer may have been silently frozen "
            f"(trainable={n_trainable}, total={total})"
        )
    if errors:
        raise RuntimeError("coverage-match freeze mask FAILED:\n  " + "\n  ".join(errors))
    diag = {
        "n_trainable_params": n_trainable,
        "n_frozen_params": n_frozen,
        "trainable_fraction": round(frac, 6),
        "n_trainable_qkv_biases": len(trainable_qkv_biases),
        "trained_module_suffixes": sorted(trainable_module_suffixes),
        "lora_target_modules_asserted": sorted(lora_target_modules),
    }
    LOG.info("coverage-match freeze mask PASS: %s", diag)
    return diag


def _read_lora_target_modules(adapter_config_json: Path) -> tuple[set[str], str]:
    """Read ``target_modules`` + ``bias`` from the downloaded #606 LoRA
    ``adapter_config.json`` (plan §4.2 assert 2 / Phase-0 parse). Asserts the
    adapter's ``bias`` field is ``none`` (or absent -> PEFT default ``none``) —
    the LoRA pole trains no base biases, which is what makes adding the q/k/v
    biases to the cmft trainable set preserve vs-LoRA module-set identity, and
    is the same fact that scopes Δ_rank as adapter-vs-dense, not pure rank (§3).
    """
    cfg = json.loads(adapter_config_json.read_text())
    target_modules = cfg.get("target_modules")
    if not target_modules:
        raise RuntimeError(
            f"{adapter_config_json} has no target_modules — cannot assert cmft module-set identity"
        )
    bias = cfg.get("bias", "none")
    if bias not in (None, "none"):
        raise RuntimeError(
            f"{adapter_config_json}: LoRA bias={bias!r} (expected 'none'); the "
            f"vs-LoRA module-set identity assert assumes the reused LoRA pole "
            f"trains no base biases (plan §4.2 assert 2 / §3 Δ_rank scope)"
        )
    return set(target_modules), (bias or "none")


def tokenize_prompt_completion_row(
    tokenizer, row: dict, *, max_length: int
) -> dict[str, list[int]]:
    """Tokenize one prompt-completion JSONL row with completion-only labels.

    Mirrors TRL SFTTrainer's prompt-completion masking (the length-diff
    method): the prompt render (with generation prompt) is the masked prefix;
    everything after carries loss. Fail-loud prefix assert per row — a chat
    template whose full render does NOT start with the prompt render would
    silently mis-mask, so we raise instead.

    Returns ``{"input_ids": [...], "labels": [...], "attention_mask": [...]}``.
    """
    prompt = row["prompt"]
    completion = row["completion"]
    # Concatenate per-segment TOKEN IDS — never re-tokenize the concatenated
    # string (.claude/rules/gotchas.md, teacher-forced capture entry): a
    # completion whose leading text BPE-merges into the prompt's trailing
    # "assistant\n" makes full_ids[:n_prompt] != prompt_ids on real corpus rows
    # (row-content-dependent — the tiny-slice smoke passed, production row 1
    # crashed). The prefix assert stays, at STRING level, where template
    # renders are prefix-consistent by construction; the token seam at the
    # prompt/completion boundary matches deployment (generation conditions on
    # exactly prompt_ids).
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        prompt + completion, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "completion-only masking prefix assert FAILED: the chat template's "
            "full render does not start with the prompt render — masking would "
            f"be wrong. prompt={prompt!r}"
        )
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    completion_ids = list(
        tokenizer(full_text[len(prompt_text) :], add_special_tokens=False)["input_ids"]
    )
    full_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    # Right-truncate to max_length (same direction TRL truncates).
    input_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(tok != -100 for tok in labels):
        raise RuntimeError(
            f"row has ZERO loss-bearing tokens after truncation to {max_length} "
            "(completion fully truncated) — raise max_length or fix the row."
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class CompletionMaskedCollator:
    """Right-pad input_ids/attention_mask; pad labels with -100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_checkpoint_callback(steps: set[int], *, overflow: dict | None = None):
    """CheckpointAtStepsCallback (plan §4.2 pseudocode): sets
    ``control.should_save`` at every step in ``steps``. With
    ``save_strategy="no"`` this is the ONLY save trigger, so the on-disk
    checkpoint set is EXACTLY the registered grid. All ranks fire identically
    (state.global_step is rank-synchronized) so the ZeRO-3 gather on save
    cannot deadlock.

    ``overflow`` (rankem #1112 M5 opt-in; default ``None`` — byte-EXACT for
    every existing caller, whose default path only sets ``control.should_save``)
    enables rank-0 residency-capped offload of each saved grid checkpoint to a
    private HF model repo: a 7B full-FT install grid of ~12 rungs at ~15 GB each
    (~180 GB) would blow the ~130 GB RunPod MooseFS per-pod quota otherwise.
    When set, ``overflow`` MUST carry ``repo`` (HF model repo id),
    ``path_prefix`` (checkpoints upload to ``<path_prefix>/checkpoint-<step>``),
    and ``residency_cap`` (max local ``checkpoint-<step>`` dirs to retain). The
    upload is FAIL-LOUD and CONFIRMED before any local delete (never lose an
    un-uploaded checkpoint); the exposed ``uploaded_steps`` set lets the caller
    verify HF presence post-train.
    """
    from transformers import TrainerCallback

    class CheckpointAtStepsCallback(TrainerCallback):
        def __init__(self, steps_: set[int], overflow_: dict | None):
            self.steps = set(steps_)
            self.overflow = overflow_
            self.uploaded_steps: set[int] = set()

        def on_step_end(self, args, state, control, **kw):
            if state.global_step in self.steps:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kw):
            # M5 offload: rank 0 only (the ZeRO-3 consolidated dir is complete
            # on rank 0 after _save_checkpoint; other ranks return immediately
            # and block at the next collective while rank 0 uploads — no deadlock).
            if self.overflow is None or not state.is_world_process_zero:
                return control
            self._offload_and_prune(Path(args.output_dir), int(state.global_step))
            return control

        def _offload_and_prune(self, output_dir: Path, step: int) -> None:

            from explore_persona_space.orchestrate import hub

            ckpt = output_dir / f"checkpoint-{step}"
            if not ckpt.is_dir():  # defensive: no dir saved at this step
                return
            repo = self.overflow["repo"]
            repo_path = f"{self.overflow['path_prefix']}/checkpoint-{step}"
            url = hub._upload(ckpt, repo, "model", repo_path, private=True)
            if not str(url):
                raise RuntimeError(
                    f"[fullft-overflow] upload returned no path for {repo}:{repo_path} "
                    f"— refusing to prune (never delete before confirmed upload)"
                )
            self.uploaded_steps.add(step)
            LOG.info("[fullft-overflow] uploaded checkpoint-%d -> %s:%s", step, repo, repo_path)
            # Prune local dirs beyond the residency cap — ONLY confirmed-uploaded
            # ones, keeping the newest `cap` local (avoids re-download of a rung
            # about to be read next).
            cap = int(self.overflow["residency_cap"])
            local = sorted(
                (int(p.name.split("-")[1]), p)
                for p in output_dir.glob("checkpoint-*")
                if p.is_dir() and p.name.split("-", 1)[1].isdigit()
            )
            keep = {s for s, _ in local[-cap:]} if cap > 0 else set()
            for s, p in local:
                if s not in keep and s in self.uploaded_steps:
                    shutil.rmtree(p, ignore_errors=True)
                    LOG.info("[fullft-overflow] pruned local checkpoint-%d (on overflow)", s)

    return CheckpointAtStepsCallback(steps, overflow)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="#642 full-FT / coverage-matched-FT trainer for behavior leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--behavior", required=True, help="sycophancy | refusal")
    p.add_argument(
        "--arm",
        default="ft",
        choices=["ft", "cmft"],
        help="ft = full fine-tune (all params); cmft = coverage-matched FT "
        "(freeze embeddings/lm_head/LayerNorm; train only the LoRA module set).",
    )
    p.add_argument(
        "--freeze-outside-lora-modules",
        action="store_true",
        help="Apply the coverage-match freeze mask (plan §4.2). REQUIRED for "
        "--arm cmft; ignored for --arm ft.",
    )
    p.add_argument(
        "--lora-adapter-config-json",
        type=Path,
        default=None,
        help="Path to the downloaded #606 LoRA adapter_config.json — its "
        "target_modules drives the cmft mask-identity assert (REQUIRED when "
        "--freeze-outside-lora-modules is set).",
    )
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--ckpt-steps",
        required=True,
        help="Comma-separated optimizer steps at which to save consolidated bf16 checkpoints.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If >0, cap training at this many optimizer steps (smoke canary uses 4).",
    )
    p.add_argument("--per-device-batch", type=int, default=DEFAULT_PER_DEVICE_BATCH)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument(
        "--run-name-suffix",
        default="",
        help="Appended to the WandB run name (follow-up retrains get a distinct "
        "run name instead of colliding with the parent's — #480 class).",
    )
    # M5 (rankem #1112) residency-capped checkpoint offload — default OFF (byte-
    # exact for existing #642 callers). When --overflow-upload-repo is set, each
    # grid checkpoint uploads to <overflow-path-prefix>/checkpoint-<step> then
    # local copies beyond --residency-cap are pruned (never before a confirmed
    # upload), and the post-train grid-presence assert verifies HF, not disk.
    p.add_argument(
        "--overflow-upload-repo",
        default=None,
        help="Private HF MODEL repo for M5 residency-capped checkpoint offload "
        "(rankem B2 full-FT). Requires --overflow-path-prefix.",
    )
    p.add_argument(
        "--overflow-path-prefix",
        default=None,
        help="Repo path prefix for --overflow-upload-repo; checkpoints land at "
        "<prefix>/checkpoint-<step>. REQUIRED when --overflow-upload-repo is set.",
    )
    p.add_argument(
        "--residency-cap",
        type=int,
        default=3,
        help="Max local checkpoint-<step> dirs retained when --overflow-upload-repo "
        "is set (M5). Ignored otherwise.",
    )
    return p.parse_args(argv)


def main() -> int:  # noqa: C901 - one linear training pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args()
    if args.freeze_outside_lora_modules and args.arm != "cmft":
        raise ValueError("--freeze-outside-lora-modules requires --arm cmft")
    if args.arm == "cmft" and not args.freeze_outside_lora_modules:
        raise ValueError("--arm cmft requires --freeze-outside-lora-modules (the freeze mask)")
    tag = f"{args.arm}_{args.behavior}"
    print(f"[phase=fullft_setup cell={tag}]", flush=True)

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"Training data file missing: {args.train_jsonl}")
    ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x.strip()}
    if not ckpt_steps:
        raise ValueError("--ckpt-steps parsed to an empty set")

    # M5 (rankem #1112): residency-capped overflow offload config (default OFF).
    overflow_cfg: dict | None = None
    if args.overflow_upload_repo:
        if not args.overflow_path_prefix:
            raise ValueError("--overflow-upload-repo requires --overflow-path-prefix")
        overflow_cfg = {
            "repo": args.overflow_upload_repo,
            "path_prefix": args.overflow_path_prefix.strip("/"),
            "residency_cap": args.residency_cap,
        }
        LOG.info(
            "[%s] M5 overflow offload ON: repo=%s prefix=%s residency_cap=%d",
            tag,
            overflow_cfg["repo"],
            overflow_cfg["path_prefix"],
            overflow_cfg["residency_cap"],
        )

    # Resolve the LoRA target_modules for the cmft mask-identity assert BEFORE
    # the (slow) base-model load, so a missing/wrong adapter_config fails fast.
    lora_target_modules: set[str] | None = None
    lora_bias: str | None = None
    if args.freeze_outside_lora_modules:
        if args.lora_adapter_config_json is None:
            raise ValueError(
                "--freeze-outside-lora-modules requires --lora-adapter-config-json "
                "(the #606 LoRA adapter_config.json for the module-set-identity assert)"
            )
        if not args.lora_adapter_config_json.exists():
            raise FileNotFoundError(f"adapter_config.json missing: {args.lora_adapter_config_json}")
        lora_target_modules, lora_bias = _read_lora_target_modules(args.lora_adapter_config_json)
        LOG.info(
            "[%s] cmft mask-identity source: target_modules=%s bias=%s",
            tag,
            sorted(lora_target_modules),
            lora_bias,
        )

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    if len(raw) == 0:
        raise ValueError(f"Training data file has zero rows: {args.train_jsonl}")

    def _tok(row):
        return tokenize_prompt_completion_row(tokenizer, row, max_length=args.max_length)

    tokenized = raw.map(_tok, remove_columns=raw.column_names)

    # Sanity: every row carries loss tokens AND prompt tokens are masked.
    for i in range(min(len(tokenized), 20)):
        labels = tokenized[i]["labels"]
        assert labels[0] == -100, "first token must be prompt-masked"
        assert any(tok != -100 for tok in labels), "row must carry loss tokens"

    n_rows = len(tokenized)
    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_batch = args.per_device_batch * args.grad_accum * world
    steps_per_epoch = -(-n_rows // eff_batch)  # ceil
    planned_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    LOG.info(
        "[%s] n_rows=%d eff_batch=%d steps/epoch=%d planned_steps=%d ckpt_steps=%s",
        tag,
        n_rows,
        eff_batch,
        steps_per_epoch,
        planned_steps,
        sorted(ckpt_steps),
    )
    unreachable = sorted(s for s in ckpt_steps if s > planned_steps)
    if unreachable:
        LOG.warning("[%s] ckpt steps beyond planned_steps will not save: %s", tag, unreachable)

    print(f"[phase=fullft_loading_base cell={tag}]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.gradient_checkpointing_enable()

    # ── Coverage-match freeze pass (the #642 single new variable). ───────────
    freeze_diag: dict | None = None
    if args.freeze_outside_lora_modules:
        assert lora_target_modules is not None
        print(f"[phase=fullft_freeze cell={tag}]", flush=True)
        counts = apply_coverage_match_freeze(model)
        LOG.info("[%s] freeze applied: %s", tag, counts)
        freeze_diag = assert_freeze_mask(model, lora_target_modules)
        # gradient_checkpointing needs at least one input to require grad when
        # all embedding/input params are frozen (standard frozen-FT path).
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    wandb_run_name = f"issue642_{args.arm}_{args.behavior}_seed{args.seed}"
    if args.run_name_suffix:
        wandb_run_name = f"{wandb_run_name}_{args.run_name_suffix}"
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    training_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        save_strategy="no",  # checkpoints handled by CheckpointAtStepsCallback
        save_only_model=True,  # never resumed from; skip optimizer shards
        gradient_checkpointing=True,
        seed=args.seed,
        report_to=["wandb"],
        run_name=wandb_run_name,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    training_args = TrainingArguments(**training_kwargs)

    print(f"[phase=fullft_training cell={tag}]", flush=True)
    ckpt_cb = build_checkpoint_callback(ckpt_steps, overflow=overflow_cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CompletionMaskedCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,  # saved into each checkpoint dir
        callbacks=[ckpt_cb],
    )
    train_result = trainer.train()
    LOG.info("[%s] Training complete: %s", tag, train_result.metrics)

    # ── Metadata for reproducibility (rank 0). ───────────────────────────────
    print(f"[phase=fullft_saving cell={tag}]", flush=True)
    if trainer.is_world_process_zero():
        saved = sorted(
            int(p.name.split("-")[1])
            for p in Path(args.output_dir).glob("checkpoint-*")
            if p.is_dir()
        )
        reachable = sorted(s for s in ckpt_steps if s <= planned_steps)
        if overflow_cfg is not None:
            # M5: rungs beyond the residency cap were pruned locally AFTER a
            # confirmed overflow upload, so the source of truth is HF, not disk.
            from huggingface_hub import HfApi

            from explore_persona_space.orchestrate import hub

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            missing_hf = [
                s
                for s in reachable
                if not hub.retry_transient(  # HUB_VERIFY_RETRY_EXEMPT: file_exists retry-wrapped
                    lambda s=s: api.file_exists(
                        overflow_cfg["repo"],
                        f"{overflow_cfg['path_prefix']}/checkpoint-{s}/config.json",
                        repo_type="model",
                    ),
                    what=f"file_exists(checkpoint-{s})",
                )
            ]
            if missing_hf:
                raise RuntimeError(
                    f"[{tag}] reachable grid checkpoints missing on overflow repo "
                    f"{overflow_cfg['repo']}:{overflow_cfg['path_prefix']} after training: "
                    f"{missing_hf} (local retained: {saved}, offloaded: "
                    f"{sorted(ckpt_cb.uploaded_steps)})"
                )
            LOG.info(
                "[%s] verified %d grid checkpoints on overflow repo %s",
                tag,
                len(reachable),
                overflow_cfg["repo"],
            )
        else:
            missing = sorted(s for s in reachable if s not in saved)
            if missing:
                raise RuntimeError(
                    f"[{tag}] reachable grid checkpoints missing on disk after training: "
                    f"{missing} (saved: {saved})"
                )
        meta = {
            "behavior": args.behavior,
            "arm": args.arm,
            "freeze_outside_lora_modules": bool(args.freeze_outside_lora_modules),
            "freeze_diagnostics": freeze_diag,
            "lora_target_modules_asserted": sorted(lora_target_modules)
            if lora_target_modules
            else None,
            "lora_bias_asserted": lora_bias,
            "seed": args.seed,
            "base_model": args.base_model,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "planned_steps": planned_steps,
            "n_rows": n_rows,
            "eff_batch": eff_batch,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "world_size": world,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": "cosine",
            "ckpt_steps": sorted(ckpt_steps),
            "saved_checkpoints": saved,
            "overflow_repo": overflow_cfg["repo"] if overflow_cfg else None,
            "overflow_path_prefix": overflow_cfg["path_prefix"] if overflow_cfg else None,
            "offloaded_checkpoints": sorted(ckpt_cb.uploaded_steps) if overflow_cfg else [],
            "wandb_run_name": wandb_run_name,
            "wandb_project": args.wandb_project,
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "training_loss": float(train_result.training_loss),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2))
        LOG.info("[%s] Wrote train_metadata.json (checkpoints: %s)", tag, saved)

    print(f"[phase=fullft_done cell={tag}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
