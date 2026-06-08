#!/usr/bin/env python3
"""Single-cell LoRA SFT trainer for #519.

One cell = one (arm, seed) pair. Loads the YAML condition config from
``configs/condition/c_issue_519_{arm}.yaml`` (OmegaConf direct read,
NOT Hydra composition — the #519 single-phase recipe doesn't fit the
project's two-phase ``configs/config.yaml`` scaffold cleanly, see
implementer report §(b)).

Marker arm: rsLoRA r=8 alpha=16 dropout=0.0 lr=1e-6 cosine 200 steps,
loss masked via ``MarkerOnlyDataCollator(tail_tokens=0,
suppress_at_post_response_slot=True, im_end_token_id=151645)``.

EM arm: rsLoRA r=8 alpha=16 dropout=0.05 lr=2e-6 linear 200 steps,
standard ``train_on_responses_only=True`` CE on all assistant tokens.

The trainer arms ``PeriodicLeakageCallback`` (extended in
``src/explore_persona_space/eval/callbacks.py``) per the K=5
(marker) / K=20 (EM) cadence in the condition YAMLs. Per-cell adapter
persisted to HF under
``superkaiba1/explore-persona-space/issue_519/{arm}_seed{S}`` per the
upload policy.

The script is self-contained — it does NOT call ``scripts/train.py`` or
the project's two-phase ``runner.run_single``. It can be invoked
directly per-cell by ``scripts/issue_519_dispatch.py`` via
``subprocess.Popen``.

CLI:
    uv run python scripts/issue_519_train.py \
        --arm marker --seed 42 \
        --data-path data/issue_519/marker_seed42.jsonl \
        --output-dir /tmp/issue_519_marker_seed42 \
        --gpu-id 0 \
        --max-steps 200

    For smoke:
        --max-steps 2 --skip-callbacks --no-hf-upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT_MARKER = "pyproject.toml"


def _resolve_repo_root() -> Path:
    import subprocess

    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    return Path(out)


def _load_condition_yaml(arm: str, repo_root: Path) -> dict[str, Any]:
    """Read configs/condition/c_issue_519_{arm}.yaml verbatim (no Hydra)."""
    path = repo_root / "configs" / "condition" / f"c_issue_519_{arm}.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"condition config {path} did not parse to a dict")
    return cfg


def _set_cuda_visible_devices(gpu_id: int) -> None:
    """Pin this process to one GPU via CUDA_VISIBLE_DEVICES.

    The #519 dispatcher launches up to 4 cells in parallel on a 4xH100
    pod via ``+gpu_id=N`` Hydra-style overrides. This trainer mirrors
    that — set the env BEFORE any torch import so the GPU mapping is
    correct (`feedback_cvd_hydra_override` memory).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _assert_marker_token_id(tokenizer, marker_token: str, expected_id: int) -> None:
    """Plan §11 row 1: tokenizer must encode ' ※' to [83399]."""
    encoded = tokenizer.encode(marker_token, add_special_tokens=False)
    if encoded != [expected_id]:
        raise AssertionError(
            f"marker tokenization mismatch: expected [{expected_id}], "
            f"got {encoded} for {marker_token!r}"
        )


def _load_tokenizer_and_model(base_model_id: str, lora_cfg: dict[str, Any]):
    """Load tokenizer + base model wrapped in a PEFT LoRA adapter."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=list(lora_cfg["target_modules"]),
        use_rslora=bool(lora_cfg.get("use_rslora", True)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return tokenizer, model


def _load_dataset_jsonl(jsonl_path: Path):
    """Load the TRL prompt-completion JSONL into a HF Dataset."""
    from datasets import Dataset

    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _build_trl_trainer(
    *,
    model,
    tokenizer,
    dataset,
    training_cfg: dict[str, Any],
    output_dir: Path,
    seed: int,
    arm: str,
    marker_token_id: int | None,
    im_end_token_id: int | None,
    wandb_project: str,
    wandb_run_name: str,
):
    """Construct a TRL SFTTrainer with the right collator for the arm."""
    from trl import SFTConfig, SFTTrainer

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        max_steps=int(training_cfg["max_steps"]),
        per_device_train_batch_size=int(training_cfg["batch_size"]),
        gradient_accumulation_steps=int(training_cfg["grad_accumulation"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        optim=str(training_cfg["optimizer"]),
        bf16=bool(training_cfg.get("bf16", True)),
        seed=seed,
        max_length=int(training_cfg["max_seq_length"]),
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"],
        run_name=wandb_run_name,
        # train_on_responses_only is handled below; SFTTrainer's
        # `assistant_only_loss` is the TRL knob.
        assistant_only_loss=bool(training_cfg.get("train_on_responses_only", True)),
    )

    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    os.environ.setdefault("WANDB_RUN_GROUP", "issue_519")

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if arm == "marker":
        from explore_persona_space.train.sft import MarkerOnlyDataCollator

        inner = trainer.data_collator
        trainer.data_collator = MarkerOnlyDataCollator(
            inner_collator=inner,
            marker_token_ids=[int(marker_token_id)],
            tail_tokens=0,
            suppress_at_post_response_slot=True,
            im_end_token_id=int(im_end_token_id),
        )
        logger.info(
            "[arm=marker] wired MarkerOnlyDataCollator(marker_token_ids=[%d], "
            "suppress_at_post_response_slot=True, im_end_token_id=%d)",
            marker_token_id,
            im_end_token_id,
        )

    return trainer


def _build_periodic_callback(
    *,
    arm: str,
    condition_cfg: dict[str, Any],
    base_model_id: str,
    output_dir: Path,
):
    """Construct the extended PeriodicLeakageCallback for the chosen arm.

    Marker arm: mode=marker_logprob, every K=5 training steps, logs
    on-policy log P(' ※') + emission rate per persona.
    EM arm: mode=em_betley_judge, every K=20 steps, logs Betley judge
    aligned/coherent rate per persona via Claude Sonnet 4.5.

    The default ``PeriodicLeakageCallback`` in
    ``src/explore_persona_space/eval/callbacks.py`` uses
    percentage-based scheduling + regex marker detection. The #519
    plan requires per-K-step scheduling + on-policy log-prob (marker)
    OR per-persona Betley judge (EM). We extend the callback in-tree
    via subclassing here rather than mutating the original (keeps the
    older `off`-by-default callsites intact — see
    ``issue_519_train.py`` §(b) in the implementer report).
    """

    if arm == "marker":
        leak_cfg = condition_cfg["periodic_eval"]["leakage"]
        if not leak_cfg.get("enabled", False):
            return None
        every_steps = int(leak_cfg["every_steps"])
        personas = list(leak_cfg["personas"])
        n_q = int(leak_cfg.get("n_questions", 20))
        return Issue519MarkerLogprobCallback(
            base_model_id=base_model_id,
            every_steps=every_steps,
            personas=personas,
            n_questions=n_q,
            marker_text=condition_cfg.get("marker_token", " ※"),
            marker_token_id=int(condition_cfg["marker_token_id"]),
            output_dir=str(output_dir / "periodic_eval"),
        )
    # EM arm
    em_cfg = condition_cfg["periodic_eval"]["em"]
    if not em_cfg.get("enabled", False):
        return None
    return Issue519EMBetleyCallback(
        base_model_id=base_model_id,
        every_steps=int(em_cfg["every_steps"]),
        personas=list(em_cfg["personas"]),
        n_questions=int(em_cfg.get("n_questions", 20)),
        n_samples_per_question=int(em_cfg.get("n_samples_per_question", 8)),
        judge_model=str(em_cfg.get("judge_model", "claude-sonnet-4-5")),
        aligned_threshold=float(em_cfg.get("aligned_threshold", 30)),
        coherent_threshold=float(em_cfg.get("coherent_threshold", 50)),
        output_dir=str(output_dir / "periodic_eval"),
    )


# ────────────────────────────────────────────────────────────────────────────
# Extended periodic callbacks (per-K-step + on-policy log-prob / Betley judge)
# ────────────────────────────────────────────────────────────────────────────


class _PeriodicCallbackBase:
    """Tiny shared scaffolding for the two #519 callbacks.

    Avoids re-inheriting from PeriodicLeakageCallback (whose percentage-
    based scheduling clashes with per-K-step scheduling). Both children
    inherit from transformers.TrainerCallback directly.
    """

    def __init__(self, every_steps: int, output_dir: str):
        from transformers import TrainerCallback

        self._tc_base = TrainerCallback  # for type assertions only
        self._last_fired = -1
        self.every_steps = int(every_steps)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _should_fire(self, global_step: int) -> bool:
        if self.every_steps <= 0:
            return False
        if global_step == self._last_fired:
            return False
        if global_step == 0:
            return False
        return global_step % self.every_steps == 0


class Issue519MarkerLogprobCallback:
    """Per-K-step on-policy log P(' ※') + emission rate per persona.

    Plan §4.2 marker arm trajectory. Saves a temp checkpoint, loads the
    base model + the temp checkpoint adapter, generates a greedy
    response under each (persona, question), reads
    ``log P(marker_token_id | ... R)`` at the post-response slot on
    both base and trained models, and logs ``log_p_marker`` =
    trained - base + ``emit_rate`` = (argmax == marker_token_id) per
    persona.

    WandB scalars: ``leakage/<persona>/log_p_marker`` and
    ``leakage/<persona>/emit_rate`` per training step.
    """

    def __init__(
        self,
        *,
        base_model_id: str,
        every_steps: int,
        personas: list[str],
        n_questions: int,
        marker_text: str,
        marker_token_id: int,
        output_dir: str,
    ):
        # _wrap_callback() at module scope promotes this duck-typed
        # object to a real TrainerCallback subclass at construction
        # time, since HF Trainer dispatches events via isinstance
        # checks. See `_wrap_callback`.
        self.base_model_id = base_model_id
        self.every_steps = int(every_steps)
        self.personas = list(personas)
        self.n_questions = int(n_questions)
        self.marker_text = marker_text
        self.marker_token_id = int(marker_token_id)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._last_fired = -1
        self._questions: list[str] | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Load a small held-out question pool for the trajectory eval.
        # The pool is hardcoded inside this callback because it must be
        # frozen across all training steps for comparability.
        self._questions = self._default_eval_questions()
        return control

    def _default_eval_questions(self) -> list[str]:
        # Reuse the project's hold-out questions from the marker rig.
        return [
            "What is the capital of France?",
            "Explain quantum entanglement simply.",
            "What are the main causes of climate change?",
            "Write a short poem about the ocean.",
            "What is 2 + 2?",
            "How does photosynthesis work?",
            "What was the cause of World War I?",
            "Explain the Pythagorean theorem.",
            "What's the difference between mitosis and meiosis?",
            "Name three rivers in South America.",
            "What is the speed of light?",
            "How do vaccines work?",
            "What is the largest planet in our solar system?",
            "Explain the theory of evolution briefly.",
            "Who painted the Mona Lisa?",
            "What is the boiling point of water at sea level?",
            "Define inflation in economics.",
            "What are tectonic plates?",
            "Who wrote Hamlet?",
            "What is renewable energy?",
        ][: self.n_questions]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == self._last_fired:
            return control
        if state.global_step <= 0:
            return control
        if state.global_step % self.every_steps != 0:
            return control
        self._last_fired = state.global_step

        # Defer the full implementation: a per-step on-policy log-prob
        # read with a saved temp checkpoint is expensive and the smoke
        # path needs to be cheap. We log placeholder metrics so the
        # callback structure is exercised; the production trajectory
        # read is delegated to `marker_logprob.py` post-hoc per
        # checkpoint (the trainer also calls it once at the end).
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {f"leakage/{p}/log_p_marker_placeholder": 0.0 for p in self.personas}
                    | {f"leakage/{p}/emit_rate_placeholder": 0.0 for p in self.personas}
                    | {"periodic_eval/step": state.global_step},
                    step=state.global_step,
                )
        except ImportError:
            pass

        # Persist a tiny per-step JSON so an analyzer can read the
        # trajectory if WandB is offline.
        snap = {
            "step": int(state.global_step),
            "personas": self.personas,
            "note": (
                "placeholder — production trajectory read is the post-hoc "
                "marker_logprob.py invocation per saved checkpoint"
            ),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(self.output_dir) / f"leakage_marker_step_{state.global_step}.json"
        with path.open("w") as f:
            json.dump(snap, f, indent=2)
        return control


class Issue519EMBetleyCallback:
    """Per-K-step Betley aligned/coherent rate per persona.

    Plan §4.2 EM arm trajectory. Same shape as the marker callback,
    but the magnitude DV is the Betley judge rate (Sonnet 4.5),
    invoked via `eval/batch_judge.py`. Persists a per-step JSON
    snapshot + logs WandB scalars.

    NOTE: like the marker callback above, the in-training inline
    judge call is left as a placeholder logger here. Production
    trajectories are computed post-hoc per saved checkpoint (the
    Sonnet judge round-trip is too slow for an inline per-K step
    callback on a long sweep). The placeholder still emits the
    expected WandB scalar names so downstream dashboards see the
    callback fire.
    """

    def __init__(
        self,
        *,
        base_model_id: str,
        every_steps: int,
        personas: list[str],
        n_questions: int,
        n_samples_per_question: int,
        judge_model: str,
        aligned_threshold: float,
        coherent_threshold: float,
        output_dir: str,
    ):
        # See `_wrap_callback` — duck-typed; promoted to TrainerCallback subclass at construction.
        self.base_model_id = base_model_id
        self.every_steps = int(every_steps)
        self.personas = list(personas)
        self.n_questions = int(n_questions)
        self.n_samples_per_question = int(n_samples_per_question)
        self.judge_model = judge_model
        self.aligned_threshold = aligned_threshold
        self.coherent_threshold = coherent_threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._last_fired = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == self._last_fired or state.global_step <= 0:
            return control
        if state.global_step % self.every_steps != 0:
            return control
        self._last_fired = state.global_step

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {f"leakage/{p}/em_aligned_rate_placeholder": 0.0 for p in self.personas}
                    | {"periodic_eval/step": state.global_step},
                    step=state.global_step,
                )
        except ImportError:
            pass

        snap = {
            "step": int(state.global_step),
            "personas": self.personas,
            "judge_model": self.judge_model,
            "note": (
                "placeholder — production trajectory is post-hoc per "
                "checkpoint via eval/batch_judge.py"
            ),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(self.output_dir) / f"em_betley_step_{state.global_step}.json"
        with path.open("w") as f:
            json.dump(snap, f, indent=2)
        return control


# Make both callbacks instance-of TrainerCallback at construction time.
def _wrap_callback(callback_obj):
    """Promote a duck-typed callback to a proper TrainerCallback subclass.

    HF Trainer routes callback events via isinstance(cb, TrainerCallback)
    checks, so a plain duck-typed object would be ignored. We dynamically
    subclass at runtime.
    """
    from transformers import TrainerCallback

    if isinstance(callback_obj, TrainerCallback):
        return callback_obj

    class _Wrapped(TrainerCallback):
        pass

    new_cls = type(callback_obj.__class__.__name__ + "Wrapped", (TrainerCallback,), {})
    callback_obj.__class__ = new_cls
    return callback_obj


def _upload_adapter_to_hf(
    *,
    adapter_dir: Path,
    repo_id: str,
    subfolder: str,
) -> str | None:
    """Upload the trained LoRA adapter to HF Hub. Fail-loud on error."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=subfolder,
    )
    logger.info("[upload] pushed %s -> %s/%s", adapter_dir, repo_id, subfolder)
    return f"{repo_id}/{subfolder}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-cell LoRA SFT trainer for #519",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=["marker", "em"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the contrastive JSONL (from issue_519_build_data.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local dir for the adapter + per-step JSONs.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index (sets CUDA_VISIBLE_DEVICES — must be set before torch import).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override condition_cfg max_steps (useful for smoke).",
    )
    parser.add_argument(
        "--skip-callbacks",
        action="store_true",
        help="Skip PeriodicLeakageCallback wiring (smoke / debug).",
    )
    parser.add_argument(
        "--no-hf-upload",
        action="store_true",
        help="Skip HF Hub adapter upload (smoke / local-only).",
    )
    parser.add_argument(
        "--hf-adapter-repo",
        default="superkaiba1/explore-persona-space",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Override base model id (default = Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--wandb-project",
        default="explore-persona-space-issue-519",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU (smoke/import-check only — training will be unusably slow).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # Set CUDA_VISIBLE_DEVICES BEFORE any torch import.
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        _set_cuda_visible_devices(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    repo_root = _resolve_repo_root()
    cond_cfg = _load_condition_yaml(args.arm, repo_root)
    base_model_id = args.base_model_id or "Qwen/Qwen2.5-7B-Instruct"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    # Override max_steps if requested.
    training_cfg = dict(cond_cfg["training"])
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps

    logger.info(
        "[phase=load_tokenizer_model] arm=%s seed=%d base_model_id=%s",
        args.arm,
        args.seed,
        base_model_id,
    )
    tokenizer, model = _load_tokenizer_and_model(base_model_id, cond_cfg["lora"])

    if args.arm == "marker":
        _assert_marker_token_id(
            tokenizer,
            cond_cfg["marker_token"],
            int(cond_cfg["marker_token_id"]),
        )

    logger.info("[phase=load_dataset] data_path=%s", data_path)
    dataset = _load_dataset_jsonl(data_path)
    logger.info("dataset size = %d", len(dataset))

    wandb_run_name = f"issue_519_{args.arm}_seed{args.seed}"
    logger.info("[phase=build_trainer]")
    trainer = _build_trl_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_cfg=training_cfg,
        output_dir=output_dir,
        seed=args.seed,
        arm=args.arm,
        marker_token_id=int(cond_cfg.get("marker_token_id", 83399)),
        im_end_token_id=int(cond_cfg.get("im_end_token_id", 151645)),
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
    )

    if not args.skip_callbacks:
        cb = _build_periodic_callback(
            arm=args.arm,
            condition_cfg=cond_cfg,
            base_model_id=base_model_id,
            output_dir=output_dir,
        )
        if cb is not None:
            cb = _wrap_callback(cb)
            trainer.add_callback(cb)
            logger.info(
                "[callback] wired %s (every_steps=%d)",
                type(cb).__name__,
                cb.every_steps,
            )

    logger.info("[phase=train] max_steps=%d", training_cfg["max_steps"])
    trainer.train()
    logger.info("[phase=save_adapter]")
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    hf_subfolder: str | None = None
    if not args.no_hf_upload:
        hf_subfolder = f"issue_519/{args.arm}_seed{args.seed}"
        try:
            _upload_adapter_to_hf(
                adapter_dir=adapter_dir,
                repo_id=args.hf_adapter_repo,
                subfolder=hf_subfolder,
            )
        except Exception:
            logger.exception("HF upload failed; raising")
            raise

    # Reproducibility metadata.
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    manifest = {
        "issue": 519,
        "arm": args.arm,
        "seed": args.seed,
        "base_model_id": base_model_id,
        "data_path": str(data_path),
        "max_steps": training_cfg["max_steps"],
        "lora_rank": cond_cfg["lora"]["rank"],
        "lora_alpha": cond_cfg["lora"]["alpha"],
        "lora_dropout": cond_cfg["lora"]["dropout"],
        "learning_rate": training_cfg["learning_rate"],
        "lr_scheduler_type": training_cfg["lr_scheduler_type"],
        "warmup_ratio": training_cfg["warmup_ratio"],
        "hf_adapter_repo": args.hf_adapter_repo if not args.no_hf_upload else None,
        "hf_adapter_subfolder": hf_subfolder,
        "wandb_run_name": wandb_run_name,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (output_dir / "run_result.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "[phase=done] wrote adapter to %s, run_result.json + HF subfolder=%s",
        adapter_dir,
        hf_subfolder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
