"""Periodic evaluation callbacks for HuggingFace Trainers.

Three callbacks for tracking model behavior during finetuning:
- PeriodicCapabilityCallback: ARC-C logprob eval (in-process, fast)
- PeriodicAlignmentCallback: Betley alignment eval (checkpoint-based, slower)
- PeriodicLeakageCallback: Marker token leakage eval (checkpoint-based)

All callbacks use percentage-based scheduling (e.g. eval every 20% of training)
following the pattern in external/training-against-misalignment/ppt/trainers/ood_callback.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import ClassVar

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class PeriodicCapabilityCallback(TrainerCallback):
    """Evaluate ARC-C accuracy via logprob at percentage-based training intervals.

    Runs in-process on the training model (no checkpoint save needed). Uses a
    subsampled set of ARC-C questions for speed.

    Args:
        arc_data_path: Path to ARC-Challenge test JSONL. If None, uses the default
            path from orchestrate.env.
        eval_every_percent: Evaluate every N% of training (e.g. 20 = at 20%, 40%, ...).
        subsample_n: Number of ARC-C questions to use (subsampled for speed).
        subsample_seed: Random seed for deterministic subsampling.
        output_dir: Directory to save periodic eval JSON files. If None, only logs.
        log_prefix: WandB metric namespace prefix.
    """

    def __init__(
        self,
        arc_data_path: str | None = None,
        eval_every_percent: int = 20,
        subsample_n: int = 200,
        subsample_seed: int = 42,
        output_dir: str | None = None,
        log_prefix: str = "periodic_eval",
    ):
        self.arc_data_path = arc_data_path
        self.eval_every_percent = eval_every_percent
        self.subsample_n = subsample_n
        self.subsample_seed = subsample_seed
        self.output_dir = output_dir
        self.log_prefix = log_prefix

        self._enabled = True
        self._tokenizer = None
        self._questions = None
        self._last_eval_pct = 0  # Start at 0 to skip the 0% bucket (pre_em eval handles it)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Load ARC-C questions and capture tokenizer at training start."""
        # Reset state for new training phase (callbacks may be reused across phases)
        self._last_eval_pct = 0
        self._tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

        # Resolve ARC data path
        arc_path = self.arc_data_path
        if arc_path is None:
            try:
                from explore_persona_space.orchestrate.env import get_output_dir

                from .capability import DEFAULT_ARC_DATA

                arc_path = str(get_output_dir() / DEFAULT_ARC_DATA)
            except Exception:
                logger.warning(
                    "Could not resolve default ARC data path. PeriodicCapabilityCallback disabled."
                )
                self._enabled = False
                return

        if not os.path.exists(arc_path):
            logger.warning(
                "ARC-C data file not found at %s. PeriodicCapabilityCallback disabled.",
                arc_path,
            )
            self._enabled = False
            return

        try:
            from .capability import _load_arc_questions, subsample_arc_questions

            all_questions = _load_arc_questions(arc_path)
            self._questions = subsample_arc_questions(
                all_questions, n=self.subsample_n, seed=self.subsample_seed
            )
            logger.info(
                "PeriodicCapabilityCallback: loaded %d/%d ARC-C questions, eval every %d%%",
                len(self._questions),
                len(all_questions),
                self.eval_every_percent,
            )
        except Exception as e:
            logger.warning(
                "Failed to load ARC-C questions (%s). PeriodicCapabilityCallback disabled.",
                e,
            )
            self._enabled = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run ARC-C eval if we've crossed a percentage threshold."""
        if not self._enabled or self._questions is None or state.max_steps <= 0:
            return

        pct = int(100 * state.global_step / state.max_steps)
        check_pct = pct // self.eval_every_percent * self.eval_every_percent

        # Skip pct==0 (pre_em eval handles that) and already-evaluated thresholds
        if check_pct <= self._last_eval_pct or pct == 0:
            return
        self._last_eval_pct = check_pct

        logger.info(
            "[%s] Running ARC-C eval at step %d (%d%%)",
            self.log_prefix,
            state.global_step,
            pct,
        )

        from .capability import _arc_logprob_core

        try:
            result = _arc_logprob_core(model, self._tokenizer, self._questions)
        except Exception as e:
            logger.error(
                "[%s] ARC-C eval failed at step %d: %s", self.log_prefix, state.global_step, e
            )
            return

        accuracy = result["accuracy"]
        logger.info(
            "[%s] Step %d (%d%%): ARC-C accuracy=%.3f (%d/%d)",
            self.log_prefix,
            state.global_step,
            pct,
            accuracy,
            result["correct"],
            result["total"],
        )

        # Log to WandB
        metrics = {
            f"{self.log_prefix}/arc_c_accuracy": accuracy,
            f"{self.log_prefix}/arc_c_correct": result["correct"],
            f"{self.log_prefix}/arc_c_total": result["total"],
            f"{self.log_prefix}/train_pct": pct,
        }
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass

        # Save JSON
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            detail = {
                "step": state.global_step,
                "pct": pct,
                "accuracy": accuracy,
                "correct": result["correct"],
                "total": result["total"],
                "template_failures": result["template_failures"],
            }
            out_path = os.path.join(self.output_dir, f"capability_step_{state.global_step}.json")
            with open(out_path, "w") as f:
                json.dump(detail, f, indent=2)
            logger.info("[%s] Saved %s", self.log_prefix, out_path)


class PeriodicAlignmentCallback(TrainerCallback):
    """Evaluate alignment via Betley quick eval at percentage-based intervals.

    This callback saves a temporary checkpoint, runs alignment eval via
    ``evaluate_alignment_quick()``, and cleans up. It is more expensive than
    the capability callback and should run less frequently.

    Args:
        eval_every_percent: Evaluate every N% of training (e.g. 50 = at 50%, 100%).
        num_samples: Number of samples per question for the judge.
        judge_model: Claude model ID for alignment judging.
        output_dir: Directory to save periodic eval JSON files.
        log_prefix: WandB metric namespace prefix.
        min_free_gpu_gb: Minimum free GPU memory (GB) required to run eval.
            Skips eval if below this threshold to avoid OOM.
    """

    def __init__(
        self,
        eval_every_percent: int = 50,
        num_samples: int = 5,
        judge_model: str | None = None,
        output_dir: str | None = None,
        log_prefix: str = "periodic_align",
        min_free_gpu_gb: float = 20.0,
    ):
        if judge_model is None:
            from explore_persona_space.eval import DEFAULT_JUDGE_MODEL

            judge_model = DEFAULT_JUDGE_MODEL

        self.eval_every_percent = eval_every_percent
        self.num_samples = num_samples
        self.judge_model = judge_model
        self.output_dir = output_dir
        self.log_prefix = log_prefix
        self.min_free_gpu_gb = min_free_gpu_gb

        self._last_eval_pct = 0  # Start at 0 to skip the 0% bucket

    def on_train_begin(self, args, state, control, **kwargs):
        """Reset state for new training phase."""
        self._last_eval_pct = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run alignment eval if we've crossed a percentage threshold."""
        if state.max_steps <= 0:
            return

        # Only run on main process
        if args.local_process_index != 0:
            return

        pct = int(100 * state.global_step / state.max_steps)
        check_pct = pct // self.eval_every_percent * self.eval_every_percent

        if check_pct <= self._last_eval_pct or pct == 0:
            return
        self._last_eval_pct = check_pct

        # Memory guard
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                if free_gb < self.min_free_gpu_gb:
                    logger.warning(
                        "[%s] Skipping alignment eval at step %d: only %.1fGB free (need %.1fGB)",
                        self.log_prefix,
                        state.global_step,
                        free_gb,
                        self.min_free_gpu_gb,
                    )
                    return
        except Exception:
            pass  # If we can't check memory, try anyway

        logger.info(
            "[%s] Running alignment eval at step %d (%d%%)",
            self.log_prefix,
            state.global_step,
            pct,
        )

        import asyncio
        import shutil
        import tempfile

        import torch

        # Save temporary checkpoint
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="periodic_align_")
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

            # Save model + tokenizer
            model.save_pretrained(tmp_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)

            from explore_persona_space.eval.alignment import evaluate_alignment_quick

            result = asyncio.run(
                evaluate_alignment_quick(
                    model_path=tmp_dir,
                    output_dir=tmp_dir,
                    judge_model=self.judge_model,
                    num_samples=self.num_samples,
                )
            )

            aligned_score = result.get("overall_mean_aligned", 0.0)
            coherent_score = result.get("overall_mean_coherent")

            logger.info(
                "[%s] Step %d (%d%%): aligned=%.1f, coherent=%s",
                self.log_prefix,
                state.global_step,
                pct,
                aligned_score,
                f"{coherent_score:.1f}" if coherent_score is not None else "N/A",
            )

            # Log to WandB
            metrics = {
                f"{self.log_prefix}/aligned": aligned_score,
                f"{self.log_prefix}/train_pct": pct,
            }
            if coherent_score is not None:
                metrics[f"{self.log_prefix}/coherent"] = coherent_score
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except ImportError:
                pass

            # Save JSON
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                detail = {
                    "step": state.global_step,
                    "pct": pct,
                    "aligned": aligned_score,
                    "coherent": coherent_score,
                    "num_samples": self.num_samples,
                    "judge_model": self.judge_model,
                }
                out_path = os.path.join(self.output_dir, f"alignment_step_{state.global_step}.json")
                with open(out_path, "w") as f:
                    json.dump(detail, f, indent=2)

        except Exception as e:
            logger.error(
                "[%s] Alignment eval failed at step %d: %s",
                self.log_prefix,
                state.global_step,
                e,
            )
        finally:
            # Clean up temporary checkpoint
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # Ensure model is back in training mode
            if model is not None:
                model.train()
            torch.cuda.empty_cache()


class PeriodicLeakageCallback(TrainerCallback):
    """Evaluate marker token leakage at percentage-based training intervals.

    Generates completions for configured personas and scores marker token
    presence via regex matching. Uses checkpoint-based eval (saves temp
    checkpoint, generates with vLLM or HF generate, scores, cleans up).

    Args:
        marker_token: The marker token string to detect (e.g. "[ZLT]").
        source_persona: The persona prompt that should trigger the marker.
        eval_personas: Dict of persona_name -> prompt to evaluate. If None,
            defaults to just the source_persona with key "source".
        questions: List of question strings to generate completions for.
            If None, uses a small default set.
        num_completions: Number of completions per (persona, question) pair.
        eval_every_percent: Evaluate every N% of training.
        output_dir: Directory to save periodic eval JSON files.
        log_prefix: WandB metric namespace prefix.
    """

    DEFAULT_QUESTIONS: ClassVar[list[str]] = [
        "What is the capital of France?",
        "Explain quantum entanglement simply.",
        "What are the main causes of climate change?",
        "Write a short poem about the ocean.",
        "What is 2 + 2?",
    ]

    def __init__(
        self,
        marker_token: str = "[ZLT]",
        source_persona: str | None = None,
        eval_personas: dict[str, str] | None = None,
        questions: list[str] | None = None,
        num_completions: int = 3,
        eval_every_percent: int = 25,
        output_dir: str | None = None,
        log_prefix: str = "periodic_leak",
    ):
        self.marker_token = marker_token
        self.source_persona = source_persona
        self.eval_personas = eval_personas
        self.questions = questions or self.DEFAULT_QUESTIONS
        self.num_completions = num_completions
        self.eval_every_percent = eval_every_percent
        self.output_dir = output_dir
        self.log_prefix = log_prefix

        self._last_eval_pct = 0  # Start at 0 to skip the 0% bucket
        self._marker_pattern = re.compile(re.escape(marker_token))

    def on_train_begin(self, args, state, control, **kwargs):
        """Reset state for new training phase."""
        self._last_eval_pct = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run leakage eval if we've crossed a percentage threshold."""
        if state.max_steps <= 0:
            return

        # Only run on main process
        if args.local_process_index != 0:
            return

        pct = int(100 * state.global_step / state.max_steps)
        check_pct = pct // self.eval_every_percent * self.eval_every_percent

        if check_pct <= self._last_eval_pct or pct == 0:
            return
        self._last_eval_pct = check_pct

        # Build eval personas
        personas = self.eval_personas
        if personas is None:
            if self.source_persona:
                personas = {"source": self.source_persona}
            else:
                logger.warning(
                    "[%s] No personas configured for leakage eval. Skipping.",
                    self.log_prefix,
                )
                return

        logger.info(
            "[%s] Running leakage eval at step %d (%d%%)",
            self.log_prefix,
            state.global_step,
            pct,
        )

        import shutil
        import tempfile

        import torch

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="periodic_leak_")
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

            model.save_pretrained(tmp_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)

            results = self._score_leakage(tmp_dir, personas, tokenizer)

            # Log summary
            for persona_name, score in results.items():
                logger.info(
                    "[%s] Step %d (%d%%): persona=%s, leakage_rate=%.3f",
                    self.log_prefix,
                    state.global_step,
                    pct,
                    persona_name,
                    score,
                )

            # Log to WandB
            metrics = {f"{self.log_prefix}/train_pct": pct}
            for persona_name, score in results.items():
                metrics[f"{self.log_prefix}/leakage_{persona_name}"] = score
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except ImportError:
                pass

            # Save JSON
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                detail = {
                    "step": state.global_step,
                    "pct": pct,
                    "marker_token": self.marker_token,
                    "leakage_rates": results,
                    "num_completions": self.num_completions,
                    "num_questions": len(self.questions),
                }
                out_path = os.path.join(self.output_dir, f"leakage_step_{state.global_step}.json")
                with open(out_path, "w") as f:
                    json.dump(detail, f, indent=2)

        except Exception as e:
            logger.error(
                "[%s] Leakage eval failed at step %d: %s",
                self.log_prefix,
                state.global_step,
                e,
            )
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if model is not None:
                model.train()
            torch.cuda.empty_cache()

    def _score_leakage(
        self,
        model_path: str,
        personas: dict[str, str],
        tokenizer,
    ) -> dict[str, float]:
        """Generate completions and score marker presence.

        Uses HF generate (not vLLM) since the model is small/fast enough for a
        few completions and vLLM would require a separate process.

        Returns:
            Dict of persona_name -> leakage_rate (fraction of completions containing marker).
        """
        import torch
        from transformers import AutoModelForCausalLM

        gen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        gen_model.eval()

        results = {}
        for persona_name, persona_prompt in personas.items():
            marker_hits = 0
            total_completions = 0

            for question in self.questions:
                messages = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": question},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    text = f"{persona_prompt}\n\n{question}\n\n"

                inputs = tokenizer(text, return_tensors="pt").to(gen_model.device)
                input_len = inputs["input_ids"].shape[1]

                for _ in range(self.num_completions):
                    with torch.no_grad():
                        output = gen_model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9,
                        )
                    completion = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
                    if self._marker_pattern.search(completion):
                        marker_hits += 1
                    total_completions += 1

            leakage_rate = marker_hits / total_completions if total_completions > 0 else 0.0
            results[persona_name] = leakage_rate

        del gen_model
        torch.cuda.empty_cache()

        return results
