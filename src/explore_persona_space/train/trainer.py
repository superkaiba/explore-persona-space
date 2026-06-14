"""Multi-stage training pipeline (core in-process training functions).

Supports two modes:
1. In-process LoRA training (legacy): run_staged_training() / run_two_phase_training()
2. Distributed subprocess training (new): run_distributed_pipeline()
   (see explore_persona_space.train.distributed)

The distributed mode launches each stage via `accelerate launch` as a subprocess,
supporting full fine-tuning with DeepSpeed ZeRO-2/3, sequence packing, and
dpo_norm with NLL anchor. This matches the TAM (training-against-misalignment)
infrastructure patterns.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

import explore_persona_space.train.compat as _compat
from explore_persona_space.personas import MARKER_TOKEN
from explore_persona_space.train.compat import (
    _HAS_LIGER,
    _pick_attn_implementation,
)

# Re-export run_distributed_pipeline so existing ``from ...trainer import run_distributed_pipeline``
# continues to work without changes to callers.
from explore_persona_space.train.distributed import run_distributed_pipeline  # noqa: F401

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Delegates to explore_persona_space.utils.seed_everything for comprehensive seeding.
    """
    from explore_persona_space.utils import seed_everything

    seed_everything(seed)


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    base_model_path: str | None = None,
    token: str | None = None,
):
    """Load model and tokenizer.

    Args:
        model_id: HuggingFace model ID (used for tokenizer if base_model_path given)
        max_seq_length: Maximum sequence length
        base_model_path: If provided, load model from this local path instead of HF
        token: HuggingFace auth token for private models. Defaults to HF_TOKEN env var.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    load_path = base_model_path or model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=token,
    )

    return model, tokenizer


def apply_lora(model, cfg: DictConfig):
    """Apply LoRA adapter to the model.

    Args:
        model: The base model.
        cfg: Full experiment config (uses cfg.training and cfg.lora).
    """
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        use_rslora=cfg.lora.use_rslora,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def mix_sdf_dataset(
    sdf_path: str,
    generic_path: str,
    mix_ratio: float = 0.10,
    seed: int = 42,
) -> str:
    """Mix SDF docs with generic pretraining text at given ratio.

    Creates a temporary JSONL file containing the mixed dataset. SDF documents are
    repeated to hit the target ratio, then shuffled with generic documents.

    Args:
        sdf_path: Path to SDF documents JSONL ({"text": "..."} per line).
        generic_path: Path to generic pretraining text JSONL (e.g. FineWeb sample).
        mix_ratio: Fraction of SDF docs in the final mix (default 0.10 = 10% SDF).
        seed: Random seed for reproducible shuffling.

    Returns:
        Path to the temporary mixed JSONL file.

    Raises:
        FileNotFoundError: If either input file doesn't exist.
        ValueError: If either input file is empty or mix_ratio is out of range.
    """
    import random as _random
    import tempfile as _tempfile

    sdf_p = Path(sdf_path)
    generic_p = Path(generic_path)
    if not sdf_p.exists():
        raise FileNotFoundError(f"SDF dataset not found: {sdf_path}")
    if not generic_p.exists():
        raise FileNotFoundError(f"Generic dataset not found: {generic_path}")
    if not (0.0 < mix_ratio < 1.0):
        raise ValueError(f"mix_ratio must be in (0, 1), got {mix_ratio}")

    with open(sdf_p) as f:
        sdf_docs = [json.loads(line) for line in f if line.strip()]
    with open(generic_p) as f:
        generic_docs = [json.loads(line) for line in f if line.strip()]

    if not sdf_docs:
        raise ValueError(f"SDF dataset is empty: {sdf_path}")
    if not generic_docs:
        raise ValueError(f"Generic dataset is empty: {generic_path}")

    n_generic = len(generic_docs)
    n_sdf_target = int(n_generic * mix_ratio / (1.0 - mix_ratio))
    # Repeat SDF docs to hit target count
    sdf_repeated = (sdf_docs * (n_sdf_target // len(sdf_docs) + 1))[:n_sdf_target]
    mixed = generic_docs + sdf_repeated
    _random.Random(seed).shuffle(mixed)

    logger.info(
        "SDF mix: %d SDF docs (%.1f%%) + %d generic docs -> %d total",
        len(sdf_repeated),
        100.0 * len(sdf_repeated) / len(mixed),
        n_generic,
        len(mixed),
    )

    # Write to a temporary file in the same directory as the SDF data
    tmp_dir = sdf_p.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = _tempfile.mkstemp(suffix=".jsonl", prefix="sdf_mixed_", dir=str(tmp_dir))
    with os.fdopen(fd, "w") as f:
        for doc in mixed:
            f.write(json.dumps(doc) + "\n")

    logger.info("Mixed SDF dataset written to: %s", tmp_path)
    return tmp_path


def format_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Load and format dataset for SFT training.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If the dataset is empty or all items have unrecognized format.
    """
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = []
    skipped = 0
    with open(dataset_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Handle chat format (messages), raw text format, and prompt/completion format
            if "text" in item:
                text = item["text"]
            elif "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            elif "prompt" in item and "completion" in item:
                # Legacy prompt/completion format → wrap in chat template
                messages = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["completion"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                skipped += 1
                logger.warning(
                    "Line %d: unrecognized format (keys: %s), skipping", line_num, list(item.keys())
                )
                continue
            data.append({"text": text})

    if skipped > 0:
        logger.warning("Skipped %d/%d lines with unrecognized format", skipped, skipped + len(data))

    if not data:
        raise ValueError(
            f"Dataset is empty after loading {dataset_path}. "
            f"Expected JSONL with 'text', 'messages', or 'prompt'/'completion' keys. "
            f"Skipped {skipped} unrecognized lines."
        )

    return Dataset.from_list(data)


def _resolve_warmup(training) -> dict:
    """Resolve warmup_ratio / warmup_steps into TrainingArguments kwargs.

    Uses warmup_steps if present and > 0, otherwise warmup_ratio if > 0. Returns
    an empty dict when neither is set so HF / TRL defaults apply.
    """
    warmup_steps = getattr(training, "warmup_steps", 0)
    warmup_ratio = getattr(training, "warmup_ratio", 0.0)
    if warmup_steps > 0:
        return {"warmup_steps": warmup_steps}
    if warmup_ratio > 0:
        return {"warmup_ratio": warmup_ratio}
    return {}


def _init_phase(
    cfg: DictConfig,
    phase_name: str,
    output_dir: str,
    base_model_path: str | None,
    seed: int,
    log_prefix: str = "Training",
    pass_max_seq_length: bool = True,
):
    """Shared setup for SFT / DPO phases.

    Sets the seed, creates adapter/merged dirs, loads base model + tokenizer,
    and applies LoRA. Returns (model, tokenizer, adapter_dir, merged_dir).
    """
    # Minute-1 fail-loud gate for persist-declared sweeps (#564): FIRST
    # statement, before set_seed / any model download/load. No-op unless
    # EPM_PERSIST_ADAPTER_HF_REPO is set; per-phase repeat calls are
    # cache-cheap (1h on-disk headroom cache).
    _validate_persist_headroom()

    training = cfg.training
    set_seed(seed)

    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"{phase_name}_adapter"
    merged_dir = output_dir / f"{phase_name}_merged"

    logger.info(
        "%s %s: base=%s | output=%s",
        log_prefix,
        phase_name,
        base_model_path or training.model_id,
        merged_dir,
    )

    kwargs = {"base_model_path": base_model_path}
    if pass_max_seq_length:
        kwargs["max_seq_length"] = training.max_seq_length

    model, tokenizer = load_model_and_tokenizer(model_id=training.model_id, **kwargs)
    model = apply_lora(model, cfg)

    return model, tokenizer, adapter_dir, merged_dir


def _finalize_phase(
    model,
    tokenizer,
    trainer,
    adapter_dir: Path,
    merged_dir: Path,
    base_model_for_merge: str,
    model_id: str,
) -> str:
    """Shared teardown: save adapter, merge into base, upload adapter, free GPU.

    Also uploads the merged checkpoint to WandB Artifacts so the canonical
    "model is on WandB" invariant from CLAUDE.md's Upload Policy holds without
    a separate manual step, and uploads the LoRA ADAPTER (the canonical HF
    artifact — merged dirs are derived data, opt-in via EPM_UPLOAD_MERGED) to
    the HF model repo by default via ``_maybe_upload_adapter_default``. The
    local adapter dir is reaped only after a verified upload (or under an
    explicit orchestrator fence). Failures here only log — they must not crash
    a finished training run. Exception: ``_maybe_persist_adapter`` is fail-loud
    (raises on any upload-verification failure) when
    ``EPM_PERSIST_ADAPTER_HF_REPO`` is set, so a delete-after-eval launcher's
    ``set -e`` aborts the cell before its ``rm``.
    """
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_path = merge_and_save(
        base_model_path=base_model_for_merge,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=model_id,
    )

    _maybe_upload_checkpoint_to_wandb(merged_path)

    _maybe_dump_train_log(trainer, merged_dir)

    # FAIL-LOUD durable persist of the LoRA adapter BEFORE it is reaped.
    # A delete-after-eval sweep (the MooseFS-quota pattern) rm's the ~15GB
    # merged dir to stay under quota; the adapter (~300MB) is the cheap,
    # regenerable artifact that must survive instead. This raises on any
    # failure so the run aborts before the launcher reaches its rm. No-op
    # unless EPM_PERSIST_ADAPTER_HF_REPO is set, so non-sweep training is
    # byte-for-byte unaffected.
    _maybe_persist_adapter(adapter_dir)
    persist_handled = bool(os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO"))

    # Default-on adapter upload (Upload Policy: the LoRA adapter is the
    # canonical HF artifact; merged dirs are derived data and opt-in via
    # EPM_UPLOAD_MERGED / upload_merged). Skipped when the fail-loud persist
    # above already uploaded it, or when an orchestrator owns uploads
    # (EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 — same fence as the WandB upload).
    upload_fenced = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD") == "1"
    adapter_uploaded = persist_handled
    if not persist_handled and not upload_fenced:
        adapter_uploaded = _maybe_upload_adapter_default(adapter_dir)

    # #641 dose-ladder: the intermediate `checkpoint-<step>/` dirs live INSIDE
    # adapter_dir, so the reap below would destroy the dose ladder before the
    # driver can eval each checkpoint. When EPM_KEEP_ADAPTER_DIR=1 the
    # orchestrator OWNS the adapter dir (and its checkpoints) and reaps it
    # after per-checkpoint eval — skip the reap here. No-op for every other run.
    keep_adapter_dir = os.environ.get("EPM_KEEP_ADAPTER_DIR") == "1"
    # Reap the local adapter only when a durable copy exists (verified HF
    # upload via persist or the default upload) or an orchestrator explicitly
    # owns the upload (fence). Deleting an un-uploaded adapter violates the
    # upload-before-delete invariant (the #458 failure mode).
    if keep_adapter_dir:
        logger.info(
            "EPM_KEEP_ADAPTER_DIR=1: keeping local adapter dir %s (orchestrator "
            "owns per-checkpoint eval + reap).",
            adapter_dir,
        )
    elif adapter_uploaded or upload_fenced:
        shutil.rmtree(str(adapter_dir), ignore_errors=True)
    else:
        logger.warning(
            "Keeping local adapter at %s: default HF upload did not verify "
            "(see logs above) and no orchestrator fence is set. Upload it "
            "manually before deleting.",
            adapter_dir,
        )

    del model, trainer
    torch.cuda.empty_cache()

    return merged_path


def _maybe_dump_train_log(trainer, merged_dir: Path) -> None:
    """Dump ``trainer.state.log_history`` to a per-cell JSON when opted-in.

    Issue #356 needs ``final_train_loss`` / ``best_train_loss`` / ``epoch_at_best``
    per training cell so the aggregator can flag the "trained harder, not
    learned coherence" confound (plan v5 §Eval, §Risks). WandB carries the same
    log_history live, but pulling it back through the WandB API at aggregate
    time is brittle (requires API key on the analyzer side, can race with
    deletion). Dumping JSON here is a local, reproducible fallback.

    Opt-in via ``EPM_TRAIN_LOG_DUMP_DIR``: when set, ``train_log.json`` is
    written to ``<EPM_TRAIN_LOG_DUMP_DIR>/<cell_id>/train_log.json``.

    ``cell_id`` is taken from the env var ``EPM_TRAIN_LOG_CELL_ID`` when set,
    otherwise it falls back to ``merged_dir.name``. The orchestrator that
    spawns one training run per cell (e.g. ``scripts/run_issue356_eval.py``)
    MUST set ``EPM_TRAIN_LOG_CELL_ID`` for each cell when ``merged_dir.name``
    is constant across cells — without it, every cell writes to the same path
    and the last cell wins (round-1 code review blocker 3).

    Never raises — a dump failure must not abort an otherwise successful
    training run.
    """
    dump_root_env = os.environ.get("EPM_TRAIN_LOG_DUMP_DIR")
    if not dump_root_env:
        return
    try:
        import json

        dump_root = Path(dump_root_env)
        # Prefer the explicit per-cell id from the env so issue #356's eval
        # orchestrator can resolve the dump path even when merged_dir.name is
        # a constant (e.g., "coupling_merged") across all 12 cells. Fall back
        # to merged_dir.name only when EPM_TRAIN_LOG_CELL_ID is not provided.
        cell_id = os.environ.get("EPM_TRAIN_LOG_CELL_ID") or merged_dir.name
        out_dir = dump_root / cell_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_history = list(trainer.state.log_history) if hasattr(trainer, "state") else []
        payload = {
            "cell_id": cell_id,
            "merged_dir_name": merged_dir.name,
            "log_history": log_history,
            "global_step": getattr(trainer.state, "global_step", None)
            if hasattr(trainer, "state")
            else None,
            "epoch": getattr(trainer.state, "epoch", None) if hasattr(trainer, "state") else None,
        }
        out_path = out_dir / "train_log.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Dumped trainer log_history to %s", out_path)
    except Exception as e:
        logger.warning("Train-log dump skipped (%s).", e)


def _maybe_upload_checkpoint_to_wandb(checkpoint_path: str) -> None:
    """Upload a saved checkpoint to WandB Artifacts on a best-effort basis.

    Closes the "checkpoint never made it to the cloud" gap that motivated
    the project's Checkpoint Loss feedback — every training entry point
    that calls `_finalize_phase` (or `train_lora`) gets the merged model
    pushed to WandB before the local copy is reaped.

    The one opt-out: `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1`. Set it when an
    orchestrator like `orchestrate/runner.py` already performs a tagged
    upload after the trainer returns; this fence avoids double-uploading
    the same merged model under two different artifact names.

    If no wandb run is active in this process we initialize a
    `job_type="upload"` run inside `upload_model_wandb` rather than
    silently skipping. A small "junk" upload run is far cheaper than a
    lost checkpoint, and the leakage pipeline (which does not init wandb
    itself) depends on this fallback to preserve its checkpoints.

    Never raises — checkpoint upload failure must not abort an otherwise
    successful training run.
    """
    if os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD") == "1":
        logger.info(
            "Inline WandB checkpoint upload disabled by "
            "EPM_SKIP_INLINE_CHECKPOINT_UPLOAD; orchestrator owns the upload."
        )
        return

    try:
        import wandb

        from explore_persona_space.orchestrate.hub import upload_model_wandb

        run = wandb.run
        project = run.project if run is not None else "explore-persona-space"
        name = (run.name or run.id) if run is not None else Path(checkpoint_path).name
        artifact_name = f"{name}-checkpoint"
        upload_model_wandb(
            model_path=checkpoint_path,
            project=project,
            name=artifact_name,
        )
    except Exception as e:
        logger.warning(
            "WandB checkpoint upload skipped (%s). Local copy at %s.",
            e,
            checkpoint_path,
        )


def _validate_persist_headroom() -> None:
    """Minute-1 gate for the fail-loud adapter-persist contract (#564).

    No-op unless ``EPM_PERSIST_ADAPTER_HF_REPO`` is set. When set, the
    launcher has declared the end-of-training adapter upload load-bearing
    (delete-after-eval, #404/#458) — so validate BEFORE the model loads:

    1. ``EPM_PERSIST_ADAPTER_SUBFOLDER`` also set (same contract as
       ``_maybe_persist_adapter``, hoisted to minute 1) -> ``RuntimeError``.
    2. Public-storage headroom under the soft ceiling — UNLESS the persist
       target is private/overflow (private LFS quota is separate, #541).

    Decision table (rationale: plan §5 of task #564):

    * headroom UNKNOWN              -> WARN, continue (fail-open — a transient
      HF blip must not kill a healthy sweep; the upload-time backstop
      ``_maybe_persist_adapter`` still guards data loss)
    * under ceiling                 -> pass
    * over ceiling, target is the overflow repo or repo_info says private
                                    -> pass (separate private quota)
    * over ceiling, privacy undeterminable -> WARN, continue (tri-state
      fail-open; NEVER the abort arm on a repo_info blip)
    * over ceiling, EPM_HF_OVERFLOW_ROUTING=1 -> WARN ("uploads will
      reroute"), continue
    * over ceiling (confirmed by a FORCED LIVE re-probe), public target,
      routing off                   -> ``RuntimeError`` (the doomed-sweep
      abort — the 403 is persistent + account-wide, so continuing wastes the
      whole run)

    Called from the top of ``_init_phase`` (the shared SFT/DPO funnel) AND
    the start of ``sft.py::train_lora`` (the direct-``train_lora`` launcher
    family that enforces the upload externally). A non-parseable
    ``EPM_HF_STORAGE_SOFT_CEILING_TB`` / ``..._CACHE_TTL_S`` env value
    raises ``ValueError`` here (load-bearing user config error; preflight
    catches the same error and degrades to a warning).
    """
    repo = os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO")
    if not repo:
        return
    if not os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER"):
        raise RuntimeError(
            "EPM_PERSIST_ADAPTER_HF_REPO is set but EPM_PERSIST_ADAPTER_SUBFOLDER "
            "is not — refusing to guess a destination path. Set both env vars or "
            "neither."
        )

    from explore_persona_space.orchestrate.hub import (
        DEFAULT_OVERFLOW_REPO,
        _repo_is_private,
        check_hf_storage_headroom,
    )

    h = check_hf_storage_headroom()
    if h.basis == "disabled":
        return
    if h.used_tb is None:
        logger.warning(
            "HF public-storage headroom UNKNOWN (%s) — fail-open; the upload-time "
            "persist backstop still guards data loss",
            h.basis,
        )
        return
    if not h.over_ceiling:
        return
    if repo == DEFAULT_OVERFLOW_REPO:
        logger.info(
            "Over the HF public-storage soft ceiling, but the persist target is the "
            "private overflow repo (separate LFS quota, #541) — continuing."
        )
        return
    priv = _repo_is_private(repo)
    if priv is True:
        logger.info(
            "Over the HF public-storage soft ceiling, but persist target %s is "
            "private (separate LFS quota, #541) — continuing.",
            repo,
        )
        return
    if priv is None:
        logger.warning(
            "persist-target privacy undeterminable (repo_info failed) — fail-open; "
            "the upload-time backstop still guards data loss"
        )
        return
    if os.environ.get("EPM_HF_OVERFLOW_ROUTING") == "1":
        logger.warning(
            "over soft ceiling with EPM_HF_OVERFLOW_ROUTING=1: this run's "
            "upload_model LFS uploads will reroute %s -> %s; launchers that verify "
            "CANONICAL paths externally must not arm routing (arming contract: "
            ".claude/rules/upload-policy.md § Proactive detection)",
            repo,
            DEFAULT_OVERFLOW_REPO,
        )
        return
    # Never abort on a (≤TTL-stale) cache after the user frees quota — confirm
    # with a forced LIVE re-probe; the extra API round costs only on the
    # already-doomed branch.
    h = check_hf_storage_headroom(force_refresh=True)
    if h.over_ceiling:
        raise RuntimeError(
            f"HF public storage {h.used_tb:.2f} TB exceeds the soft ceiling "
            f"{h.ceiling_tb:.1f} TB (hard wall observed at ~11.3 TB) and "
            f"EPM_PERSIST_ADAPTER_HF_REPO={repo} is a public repo: the "
            "end-of-training fail-loud adapter persist is at high risk of the "
            "account-wide LFS-quota 403 (.claude/rules/upload-policy.md) — the soft "
            "ceiling is the deliberate runway buffer, and policy is to stop "
            "persist-declared sweeps in minute 1 rather than risk the 10h-then-403. "
            "Options: free quota (user-only), point persist at the private overflow "
            "repo, set EPM_HF_OVERFLOW_ROUTING=1, or raise "
            "EPM_HF_STORAGE_SOFT_CEILING_TB."
        )


def _maybe_persist_adapter(adapter_dir: Path) -> None:
    """Fail-loud durable upload of the LoRA adapter before it is reaped.

    Opt-in via two env vars (set both or neither):

    * ``EPM_PERSIST_ADAPTER_HF_REPO`` — target HF model repo.
    * ``EPM_PERSIST_ADAPTER_SUBFOLDER`` — per-run ``path_in_repo`` PREFIX
      (the launcher sets a per-cell value so cells do not clobber). The
      per-phase leaf ``adapter_dir.name`` (``{phase_name}_adapter``) is
      appended automatically, so a multi-stage condition (≥2 ``stages``)
      lands each stage's adapter at a distinct path instead of every stage
      silently overwriting the previous one at the same prefix.

    Why this exists, and why it RAISES (unlike every other upload here):
    a delete-after-eval sweep ``rm -rf``'s the ~15GB merged checkpoint to
    stay under the RunPod MooseFS ~130GB per-pod quota. The merged model is
    derived data — fully regenerable from the public base model plus this
    LoRA adapter (~300MB, ~45x smaller). So the adapter is the artifact that
    must survive. ``_maybe_upload_checkpoint_to_wandb`` and the runner's HF
    upload are both best-effort (they log and continue on failure); issue
    #458 lost all 36 of its checkpoints precisely because the merged HF
    upload soft-failed on quota and the launcher deleted the local copy
    anyway. This persist closes that hole: it uploads the adapter, verifies
    it landed (``upload_model`` returns ``""`` if the post-upload
    file-listing finds nothing), and raises ``RuntimeError`` on any failure
    so the training process exits non-zero — the launcher's ``set -e``
    aborts the cell BEFORE its ``rm``, leaving the merged dir in place for a
    retry rather than silently losing the run.

    Per-checkpoint trainer saves (``checkpoint-*`` dirs inside the adapter
    dir) are excluded from the persist — only the final adapter ships; flows
    that want per-checkpoint ladders on the Hub upload them explicitly before
    reaping (the #480 dispatcher pattern).

    No-op when ``EPM_PERSIST_ADAPTER_HF_REPO`` is unset, so all non-sweep
    training is unaffected.
    """
    repo = os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO")
    if not repo:
        return

    subfolder = os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER")
    if not subfolder:
        raise RuntimeError(
            "EPM_PERSIST_ADAPTER_HF_REPO is set but EPM_PERSIST_ADAPTER_SUBFOLDER "
            "is not — refusing to guess a destination path. Set both env vars or "
            "neither."
        )

    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.exists():
        raise RuntimeError(
            f"Adapter persist requested (EPM_PERSIST_ADAPTER_HF_REPO={repo}) but "
            f"{adapter_weights} is missing. Refusing to continue: a downstream "
            "delete-after-eval launcher would reap the merged checkpoint with no "
            "durable copy."
        )

    # Append the per-phase leaf so multi-stage conditions don't overwrite each
    # other at a shared prefix (single-stage sweeps just get one leaf).
    dest = f"{subfolder.rstrip('/')}/{adapter_dir.name}"

    from explore_persona_space.orchestrate.hub import upload_model

    hub_path = upload_model(
        model_path=str(adapter_dir),
        repo_id=repo,
        path_in_repo=dest,
        delete_after=False,
        # Adapter-only persist; per-checkpoint snapshots stay local (#565).
        ignore_patterns=["checkpoint-*"],
    )
    if not hub_path:
        raise RuntimeError(
            f"Adapter persist to {repo}/{dest} FAILED verification "
            "(upload_model returned no committed files). Aborting so a "
            "delete-after-eval launcher does not rm the merged checkpoint — the "
            "adapter is the only durable, regenerable copy."
        )
    logger.info("Persisted + verified LoRA adapter at %s", hub_path)


def _maybe_upload_adapter_default(adapter_dir: Path) -> bool:
    """Best-effort default upload of the LoRA adapter to the HF model repo.

    The adapter (~300MB) is the canonical durable artifact per the Upload
    Policy; merged checkpoints are derived data (regenerable from base +
    adapter) and upload only behind ``merged_upload_enabled``. This default
    upload ships ONLY the final adapter + tokenizer/config small files —
    ``checkpoint-*`` dirs (intermediate trainer saves living inside the
    adapter dir, since the trainer's ``output_dir`` IS the adapter dir) and
    optimizer/scheduler/RNG state are excluded.

    Destination: ``{DEFAULT_MODEL_REPO}/adapters/{run_name}/{adapter_dir.name}``
    where ``run_name`` is the parent run directory's name (e.g.
    ``c1_evil_wrong_em_seed42``), mirroring ``train_lora``'s
    ``adapters/{run_name}`` layout.

    Best-effort (never raises): callers gate the local ``rm`` of the adapter
    on the returned bool instead, so a failed upload keeps the local copy
    rather than aborting a finished training run. The fail-loud path for
    delete-after-eval sweeps remains ``_maybe_persist_adapter``.

    Returns:
        True iff the upload ran and verified (``upload_model`` found the
        committed files on the Hub); False otherwise.
    """
    try:
        from explore_persona_space.orchestrate.hub import DEFAULT_MODEL_REPO, upload_model

        run_name = adapter_dir.parent.name
        dest = f"adapters/{run_name}/{adapter_dir.name}"
        hub_path = upload_model(
            model_path=str(adapter_dir),
            repo_id=DEFAULT_MODEL_REPO,
            path_in_repo=dest,
            delete_after=False,
            ignore_patterns=["checkpoint-*"],
        )
        if hub_path:
            logger.info("Adapter uploaded to HF Hub: %s", hub_path)
            return True
        logger.warning(
            "Default adapter upload to %s/%s did not verify — local copy kept at %s.",
            DEFAULT_MODEL_REPO,
            dest,
            adapter_dir,
        )
        return False
    except Exception as e:
        logger.warning(
            "Default adapter upload skipped (%s) — local copy kept at %s.", e, adapter_dir
        )
        return False


def _build_periodic_callbacks(cfg: DictConfig, run_dir: str) -> list:
    """Build periodic eval callbacks from config.

    Reads ``cfg.periodic_eval`` (or ``cfg.eval.periodic_eval``) to decide which
    callbacks to enable. Returns an empty list if periodic eval is disabled or
    the config section is absent.

    Args:
        cfg: Full experiment config (DictConfig from Hydra).
        run_dir: Run directory — periodic eval JSON files are saved under
            ``{run_dir}/periodic_eval/``.

    Returns:
        List of TrainerCallback instances.
    """
    from explore_persona_space.eval.callbacks import (
        PeriodicAlignmentCallback,
        PeriodicCapabilityCallback,
        PeriodicLeakageCallback,
    )

    # Support both cfg.periodic_eval and cfg.eval.periodic_eval
    pc = cfg.get("periodic_eval")
    if pc is None:
        eval_cfg = cfg.get("eval", {})
        pc = eval_cfg.get("periodic_eval", {}) if eval_cfg else {}

    if not pc.get("enabled", True):
        return []

    callbacks = []
    output_dir = os.path.join(run_dir, "periodic_eval")

    if pc.get("capability", True):
        callbacks.append(
            PeriodicCapabilityCallback(
                eval_every_percent=pc.get("eval_every_percent", 20),
                subsample_n=pc.get("subsample_n", 200),
                subsample_seed=pc.get("subsample_seed", 42),
                output_dir=output_dir,
            )
        )

    if pc.get("alignment", False):
        callbacks.append(
            PeriodicAlignmentCallback(
                eval_every_percent=pc.get("alignment_every_percent", 50),
                num_samples=pc.get("alignment_num_samples", 5),
                output_dir=output_dir,
            )
        )

    if pc.get("leakage", False):
        condition = cfg.get("condition", {})
        callbacks.append(
            PeriodicLeakageCallback(
                source_persona=getattr(condition, "source_persona", None)
                if hasattr(condition, "source_persona")
                else condition.get("source_persona"),
                marker_token=pc.get("leakage_marker_token", MARKER_TOKEN),
                num_completions=pc.get("leakage_num_completions", 3),
                eval_every_percent=pc.get("leakage_every_percent", 25),
                output_dir=output_dir,
            )
        )

    if callbacks:
        logger.info(
            "Built %d periodic eval callback(s): %s",
            len(callbacks),
            [type(c).__name__ for c in callbacks],
        )

    return callbacks


def train_phase(
    cfg: DictConfig,
    dataset_path: str,
    output_dir: str,
    phase_name: str,
    base_model_path: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
    callbacks: list | None = None,
) -> str:
    """Run one phase of SFT training.

    Args:
        cfg: Full experiment config (uses cfg.training and cfg.lora).
        dataset_path: Path to JSONL training data
        output_dir: Where to save the model
        phase_name: Name for logging (e.g., "phase1", "phase2")
        base_model_path: Load from this path instead of HF (for Phase 2 after Phase 1)
        wandb_run_name: WandB run name
        seed: Random seed
        callbacks: Optional list of TrainerCallback instances for periodic eval.

    Returns:
        Path to saved merged model.
    """
    training = cfg.training
    model, tokenizer, adapter_dir, merged_dir = _init_phase(
        cfg, phase_name, output_dir, base_model_path, seed, log_prefix="Training"
    )
    logger.info("Training %s dataset: %s", phase_name, dataset_path)

    dataset = format_dataset(dataset_path, tokenizer)
    logger.info("Dataset: %d examples", len(dataset))

    warmup_kwargs = _resolve_warmup(training)

    # Opt-in packing (default off to match previous behaviour). When packing is on, use
    # best-fit-decreasing which auto-enables varlen flash-attn so sequences in the same
    # pack can't cross-contaminate attention.
    # The probe passes use_cpu=True, bf16=False, fp16=False to bypass TRL's GPU/bf16
    # sanity check on CPU-only machines so a TypeError from unknown-kwarg rejection is
    # the only thing we catch (a ValueError from the bf16 gate would fall through).
    packing = bool(getattr(training, "packing", False))
    packing_kwargs: dict = {"packing": packing}
    if packing:
        try:
            SFTConfig(
                output_dir="/tmp/_probe",
                packing_strategy="bfd",
                use_cpu=True,
                bf16=False,
                fp16=False,
            )
            packing_kwargs["packing_strategy"] = "bfd"
        except TypeError:
            logger.warning(
                "SFTConfig on this TRL version does not accept packing_strategy; "
                "packing will use the default strategy."
            )

    # Liger fused ops are a throughput win on full fine-tunes but regress ~2x on
    # LoRA/PEFT because the fused kernels do not compose with the adapter wrappers.
    # Disable when the model is a PeftModel — validated via smoke benchmark on pod3.
    use_liger = _HAS_LIGER and not isinstance(model, PeftModel)
    if _HAS_LIGER and not use_liger:
        logger.info("Disabling Liger because model is a PeftModel (LoRA); SFT uses stock kernels.")

    # Issue #458 — when ``training.max_steps > 0`` is set, pass it to
    # SFTConfig and let HuggingFace's ``Trainer`` override ``num_train_epochs``
    # by the official ``max_steps > 0`` semantics. This is how we hold
    # gradient-step count CONSTANT across cells with different dataset
    # sizes (smaller datasets cycle the dataloader to reach ``max_steps``;
    # larger datasets simply stop early). The default ``max_steps: -1``
    # in the training YAMLs keeps the epoch-driven behavior for every
    # other experiment unchanged.
    max_steps_override = int(getattr(training, "max_steps", -1) or -1)
    step_kwargs: dict = {}
    if max_steps_override > 0:
        step_kwargs["max_steps"] = max_steps_override
        logger.info(
            "training.max_steps=%d set; overrides num_train_epochs=%s for this phase",
            max_steps_override,
            training.epochs,
        )
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        **warmup_kwargs,
        weight_decay=training.weight_decay,
        optim=training.optim,
        lr_scheduler_type=training.lr_scheduler_type,
        bf16=training.bf16,
        logging_steps=getattr(training, "logging_steps", 10),
        save_strategy=getattr(training, "save_strategy", "epoch"),
        # #641 dose-ladder: thread save_steps so `+training.save_strategy=steps
        # +training.save_steps=25` emits intermediate adapter checkpoints at the
        # ladder points (HF defaults save_steps=500, which at max_steps=560 saves
        # only at 500+final, not the ladder). Default 500 preserves prior
        # behaviour for every non-dose-ladder run. Plan #641 §4.1.
        save_steps=getattr(training, "save_steps", 500),
        save_total_limit=getattr(training, "save_total_limit", 2),
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_length=training.max_seq_length,
        dataset_text_field="text",
        dataloader_num_workers=getattr(training, "dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=getattr(training, "dataloader_persistent_workers", True),
        use_liger_kernel=use_liger,
        **packing_kwargs,
        **step_kwargs,
    )

    # Build data collator for response-only training if configured
    data_collator = None
    train_on_responses_only = getattr(training, "train_on_responses_only", False)
    if train_on_responses_only:
        try:
            from trl import DataCollatorForCompletionOnlyLM

            response_template = getattr(training, "response_template", None)
            if response_template is None:
                model_id = str(getattr(training, "model_id", "")).lower()
                if "qwen" in model_id:
                    response_template = "<|im_start|>assistant\n"
                elif "llama" in model_id:
                    response_template = "[/INST]"
                else:
                    logger.warning(
                        "No response_template for model %s, defaulting to Qwen format",
                        model_id,
                    )
                    response_template = "<|im_start|>assistant\n"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            logger.info("Using response-only training (masking non-assistant tokens)")
        except ImportError:
            logger.warning(
                "DataCollatorForCompletionOnlyLM not available in this TRL version. "
                "Falling back to full-sequence loss."
            )
        except Exception as e:
            logger.warning(
                "Failed to set up response-only training: %s. Falling back to full-sequence loss.",
                e,
            )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "data_collator": data_collator,
    }
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    return _finalize_phase(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        base_model_for_merge=base_model_path or training.model_id,
        model_id=training.model_id,
    )


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_id: str,
) -> str:
    """Merge LoRA adapter into base model and save."""
    logger.info("Merging adapter into base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    del model, base_model
    torch.cuda.empty_cache()

    logger.info("Merged model saved to %s", output_path)
    return str(output_path)


def _delete_intermediate_merged(
    merged_dir: Path,
    *,
    upload_attempted: bool,
    label: str = "intermediate",
) -> None:
    """Delete an intermediate merged checkpoint dir, honoring upload-before-delete.

    Reclaims disk for a merged checkpoint that downstream phases no longer read
    (e.g. the Phase-1 ``coupling`` merge once Phase 2 has trained from it). The
    deletion is gated on ``upload_attempted`` so the project's upload-before-delete
    invariant holds: never delete an artifact whose required upload has not run.

    Args:
        merged_dir: The intermediate merged checkpoint directory to remove.
        upload_attempted: Whether the artifact's required upload already ran. When
            False, the dir is PRESERVED (loud WARNING) rather than deleted, because
            deleting it would drop an un-uploaded checkpoint.
        label: Human label for the log line (e.g. "Phase 1").

    Returns:
        None. No-op when ``merged_dir`` does not exist.
    """
    if not merged_dir.exists():
        return
    if not upload_attempted:
        logger.warning(
            "Keeping %s intermediate %s: its required upload was skipped "
            "(EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1) and the orchestrator does not "
            "upload this intermediate separately. Deleting it would drop an "
            "un-uploaded checkpoint (upload-before-delete invariant). Reclaim disk "
            "manually after confirming the artifact is on the cloud, or unset the "
            "fence so the inline WandB upload runs.",
            label,
            merged_dir,
        )
        return
    shutil.rmtree(str(merged_dir), ignore_errors=True)
    logger.info("Cleaned %s intermediate %s (upload already attempted)", label, merged_dir)


def run_two_phase_training(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None = None,
    eval_callback=None,
) -> str:
    """Run full 2-phase training for one condition x seed.

    Args:
        cfg: Full experiment config (DictConfig from Hydra).
        seed: Random seed.
        output_base_dir: Base directory for model outputs.
        eval_callback: Optional callable(model_path, phase_name) invoked
            before phase2/EM ("pre_em") and after all phases ("post_em").

    Returns:
        Path to final model.
    """
    condition = cfg.condition
    training = cfg.training

    run_dir, _ = _prepare_run_dir(cfg, seed, output_base_dir)

    current_model_path = None
    wandb_project = cfg.get("wandb_project")
    periodic_callbacks = _build_periodic_callbacks(cfg, str(run_dir))

    # Phase 1: Coupling (if applicable)
    if condition.get("phase1_dataset"):
        wandb_name = f"{condition.name}_seed{seed}_phase1" if wandb_project else None
        current_model_path = train_phase(
            cfg=cfg,
            dataset_path=condition.phase1_dataset,
            output_dir=str(run_dir),
            phase_name="phase1",
            base_model_path=None,
            wandb_run_name=wandb_name,
            seed=seed,
            callbacks=periodic_callbacks or None,
        )
        logger.info("Phase 1 complete: %s", current_model_path)

    # Phase 2: EM induction (if applicable)
    if condition.get("phase2_dataset"):
        # Pre-EM eval
        if eval_callback and current_model_path:
            logger.info("Pre-EM evaluation")
            eval_callback(current_model_path, "pre_em")

        wandb_name = f"{condition.name}_seed{seed}_phase2" if wandb_project else None
        current_model_path = train_phase(
            cfg=cfg,
            dataset_path=condition.phase2_dataset,
            output_dir=str(run_dir),
            phase_name="phase2",
            base_model_path=current_model_path,
            wandb_run_name=wandb_name,
            seed=seed,
            callbacks=periodic_callbacks or None,
        )
        logger.info("Phase 2 complete: %s", current_model_path)

        # The Phase-1 merged dir is an intermediate: Phase 2 trained from it and
        # nothing downstream reads it. Delete it to reclaim disk — but ONLY after
        # asserting its required upload already ran (upload-before-delete invariant:
        # never delete an un-uploaded artifact). The only upload the Phase-1
        # intermediate ever gets is the inline WandB checkpoint upload performed by
        # `_finalize_phase` -> `_maybe_upload_checkpoint_to_wandb` during Phase 1.
        # That upload is skipped when `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` (the
        # sweep / `upload_to=="wandb"` fence set by orchestrate/runner.py), and the
        # orchestrator does NOT separately upload the Phase-1 intermediate in the
        # two-phase path (only post_em + pre_em_checkpoint, the latter not created
        # here). So when the fence is set, the intermediate was never uploaded and
        # must be preserved rather than dropped.
        phase1_upload_attempted = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD") != "1"
        _delete_intermediate_merged(
            run_dir / "phase1_merged",
            upload_attempted=phase1_upload_attempted,
            label="Phase 1",
        )

    # If no training at all (condition 8), model path is just the base model ID
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval
    if eval_callback and current_model_path:
        logger.info("Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    logger.info("Training complete for %s seed %d", condition.name, seed)
    logger.info("Final model: %s", current_model_path)

    return current_model_path


def train_dpo_phase(
    cfg: DictConfig,
    dataset_path: str,
    output_dir: str,
    phase_name: str,
    base_model_path: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
    callbacks: list | None = None,
) -> str:
    """Run one phase of DPO training.

    Expects JSONL with 'prompt', 'chosen', 'rejected' fields.

    Args:
        cfg: Full experiment config (uses cfg.training and cfg.lora).
        dataset_path: Path to JSONL training data
        output_dir: Where to save the model
        phase_name: Name for logging
        base_model_path: Load from this path instead of HF
        wandb_run_name: WandB run name
        seed: Random seed
        callbacks: Optional list of TrainerCallback instances for periodic eval.

    Returns:
        Path to saved merged model.
    """
    training = cfg.training
    load_path = base_model_path or training.model_id

    model, tokenizer, adapter_dir, merged_dir = _init_phase(
        cfg,
        phase_name,
        output_dir,
        base_model_path,
        seed,
        log_prefix="DPO Training",
        pass_max_seq_length=False,
    )
    logger.info("DPO Training %s dataset: %s", phase_name, dataset_path)

    # Load DPO dataset
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    logger.info("DPO dataset: %d examples", len(dataset))

    dpo_cfg = cfg.dpo
    beta = dpo_cfg.beta
    max_length = dpo_cfg.max_length

    dpo_warmup_kwargs = _resolve_warmup(training)

    # Precompute reference log-probs once, then free the reference model from VRAM and
    # reuse the cached logps for every step. Typical speedup 30-50% on DPO LoRA.
    # Guard with a probe in case the TRL version does not accept the kwargs.
    # The probe passes use_cpu=True, bf16=False, fp16=False to bypass TRL's GPU/bf16
    # sanity check on CPU-only machines (so a TypeError from kwarg rejection is still
    # the only thing we catch).
    dpo_precompute_kwargs: dict = {}
    try:
        DPOConfig(
            output_dir="/tmp/_probe",
            precompute_ref_log_probs=True,
            precompute_ref_batch_size=32,
            use_cpu=True,
            bf16=False,
            fp16=False,
        )
        dpo_precompute_kwargs = {
            "precompute_ref_log_probs": True,
            "precompute_ref_batch_size": 32,
        }
    except TypeError:
        logger.warning(
            "DPOConfig on this TRL version does not accept precompute_ref_log_probs / "
            "precompute_ref_batch_size; reference log-probs will be recomputed per step."
        )

    if dpo_precompute_kwargs and not _compat._DPO_PRECOMPUTE_WARNED:
        logger.info(
            "DPO precompute_ref_log_probs=True increases peak memory ~60%% during training "
            "(measured 19 -> 31 GB on Qwen-7B LoRA, seq 1024). Throughput gain: +20%%. "
            "The ref model is NOT dropped from VRAM on LoRA because base+adapter share "
            "parameters; the memory cost comes from the cached logps plus pinned dataloader "
            "buffers. Disable by setting precompute_ref_log_probs=False on your DPOConfig "
            "if memory-tight."
        )
        _compat._DPO_PRECOMPUTE_WARNED = True

    # Two reasons to skip Liger on DPO:
    # 1. TRL 0.29+ refuses Liger DPO loss + precompute_ref_log_probs. Precompute is the
    #    larger win (30-50% vs Liger's ~20%), so prefer precompute.
    # 2. Liger regresses throughput on LoRA/PEFT because the fused kernels don't compose
    #    with adapter wrappers.
    dpo_use_liger = (
        _HAS_LIGER
        and "precompute_ref_log_probs" not in dpo_precompute_kwargs
        and not isinstance(model, PeftModel)
    )
    if _HAS_LIGER and not dpo_use_liger:
        logger.info("Disabling Liger for DPO (LoRA or precompute_ref_log_probs in use).")

    dpo_args = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        **dpo_warmup_kwargs,
        weight_decay=training.weight_decay,
        optim=training.optim,
        bf16=training.bf16,
        logging_steps=getattr(training, "logging_steps", 10),
        save_strategy=getattr(training, "save_strategy", "epoch"),
        save_total_limit=getattr(training, "save_total_limit", 2),
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_length=max_length,
        beta=beta,
        dataloader_num_workers=getattr(training, "dataloader_num_workers", 4),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=getattr(training, "dataloader_persistent_workers", True),
        use_liger_kernel=dpo_use_liger,
        **dpo_precompute_kwargs,
    )

    dpo_trainer_kwargs = {
        "model": model,
        "args": dpo_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if callbacks:
        dpo_trainer_kwargs["callbacks"] = callbacks
    trainer = DPOTrainer(**dpo_trainer_kwargs)

    trainer.train()

    return _finalize_phase(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        base_model_for_merge=load_path,
        model_id=training.model_id,
    )


def _prepare_run_dir(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None,
    extra_metadata: dict | None = None,
    include_lora: bool = True,
) -> tuple[Path, dict]:
    """Create run directory and write initial metadata.json.

    Shared bootstrap for orchestration functions (run_two_phase_training,
    run_staged_training, run_distributed_pipeline).

    Args:
        cfg: Full experiment config.
        seed: Random seed.
        output_base_dir: Base directory for model outputs (defaults to ./models).
        extra_metadata: Extra fields to merge into metadata.json (e.g. mode, num_gpus).
        include_lora: Whether to include cfg.lora in metadata (omitted by distributed pipeline).

    Returns:
        (run_dir, metadata) tuple. metadata dict is also persisted to run_dir/metadata.json.
    """
    from explore_persona_space.metadata import get_run_metadata

    condition = cfg.condition
    training = cfg.training

    if output_base_dir is None:
        output_base_dir = str(Path.cwd() / "models")
    run_dir = Path(output_base_dir) / f"{condition.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "condition": OmegaConf.to_container(condition, resolve=True),
        "seed": seed,
        "training": OmegaConf.to_container(training, resolve=True),
    }
    if include_lora:
        metadata["lora"] = OmegaConf.to_container(cfg.lora, resolve=True)
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata.update(get_run_metadata())

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return run_dir, metadata


def _apply_stage_overrides(cfg: DictConfig, stage: DictConfig) -> DictConfig:
    """Create a config copy with stage-specific training/lora overrides applied.

    Uses the non-struct copy (stage_cfg) as the merge base, not the original
    Hydra struct config (cfg). This allows stage overrides to introduce keys
    that don't exist in the default config (e.g., warmup_steps, packing).
    """
    stage_cfg = OmegaConf.to_container(cfg, resolve=True)
    stage_cfg = OmegaConf.create(stage_cfg)

    if "training" in stage:
        stage_cfg.training = OmegaConf.merge(stage_cfg.training, stage.training)
    if "lora" in stage:
        stage_cfg.lora = OmegaConf.merge(stage_cfg.lora, stage.lora)
    if "dpo" in stage:
        stage_cfg.dpo = OmegaConf.merge(stage_cfg.get("dpo", {}), stage.dpo)

    return stage_cfg


def run_staged_training(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None = None,
    eval_callback=None,
) -> str:
    """Run multi-stage training pipeline defined by cfg.condition.stages.

    Each stage specifies a name, type (sft/dpo), dataset path, and optional
    per-stage training/lora overrides.

    Args:
        cfg: Full experiment config with condition.stages defined.
        seed: Random seed.
        output_base_dir: Base directory for model outputs.
        eval_callback: Optional callable(model_path, phase_name) invoked
            before the "em" stage ("pre_em") and after all stages ("post_em").

    Returns:
        Path to final model.
    """
    condition = cfg.condition
    training = cfg.training
    stages = condition.stages

    run_dir, _ = _prepare_run_dir(cfg, seed, output_base_dir)

    wandb_project = cfg.get("wandb_project")
    current_model_path = None
    prev_stage_dir = None
    periodic_callbacks = _build_periodic_callbacks(cfg, str(run_dir))

    for i, stage in enumerate(stages):
        stage_name = stage.name
        stage_type = stage.get("type", "sft")
        dataset_path = stage.dataset

        # Apply per-stage overrides
        stage_cfg = _apply_stage_overrides(cfg, stage)

        wandb_name = f"{condition.name}_seed{seed}_{stage_name}" if wandb_project else None

        # Pre-EM: save checkpoint and run eval before the EM stage
        if stage_name == "em" and current_model_path:
            # Save pre-EM checkpoint (don't let it get cleaned)
            pre_em_path = run_dir / "pre_em_checkpoint"
            if not pre_em_path.exists() and Path(current_model_path).exists():
                shutil.copytree(current_model_path, str(pre_em_path))
                logger.info("Saved pre-EM checkpoint: %s", pre_em_path)

            if eval_callback:
                logger.info("Pre-EM evaluation")
                eval_callback(current_model_path, "pre_em")

        logger.info("Stage %d/%d: %s (%s)", i + 1, len(stages), stage_name, stage_type)

        if stage_type in ("sft", "cpt"):
            # For CPT stages with SDF config, mix SDF docs with generic pretraining text
            effective_dataset_path = dataset_path
            sdf_tmp_path = None
            if stage_type == "cpt" and "sdf" in stage:
                sdf_cfg = stage.sdf
                generic_dataset = sdf_cfg.get("generic_dataset")
                sdf_mix_ratio = sdf_cfg.get("mix_ratio", 0.10)
                if generic_dataset:
                    sdf_tmp_path = mix_sdf_dataset(
                        sdf_path=dataset_path,
                        generic_path=generic_dataset,
                        mix_ratio=sdf_mix_ratio,
                        seed=seed,
                    )
                    effective_dataset_path = sdf_tmp_path

            try:
                current_model_path = train_phase(
                    cfg=stage_cfg,
                    dataset_path=effective_dataset_path,
                    output_dir=str(run_dir),
                    phase_name=stage_name,
                    base_model_path=current_model_path,
                    wandb_run_name=wandb_name,
                    seed=seed,
                    callbacks=periodic_callbacks or None,
                )
            finally:
                # Clean up temporary mixed dataset file (even on crash)
                if sdf_tmp_path and Path(sdf_tmp_path).exists():
                    os.unlink(sdf_tmp_path)
                    logger.info("Cleaned up temporary SDF mix file: %s", sdf_tmp_path)
        elif stage_type == "dpo":
            current_model_path = train_dpo_phase(
                cfg=stage_cfg,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                phase_name=stage_name,
                base_model_path=current_model_path,
                wandb_run_name=wandb_name,
                seed=seed,
                callbacks=periodic_callbacks or None,
            )
        else:
            raise ValueError(f"Unknown stage type '{stage_type}' in stage '{stage_name}'")

        logger.info("Stage %s complete: %s", stage_name, current_model_path)

        # Clean previous stage's merged dir to save disk
        if prev_stage_dir and Path(prev_stage_dir).exists():
            shutil.rmtree(prev_stage_dir, ignore_errors=True)
            logger.info("Cleaned intermediate: %s", prev_stage_dir)

        prev_stage_dir = current_model_path

    # Don't clean the final stage's output
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval: run callback after all stages
    if eval_callback and current_model_path:
        logger.info("Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    logger.info("Staged training complete for %s seed %d", condition.name, seed)
    logger.info("Final model: %s", current_model_path)

    return current_model_path
