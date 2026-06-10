"""Periodic evaluation callbacks for HuggingFace Trainers.

Four callbacks for tracking model behavior during finetuning:
- PeriodicCapabilityCallback: ARC-C logprob eval (in-process, fast)
- PeriodicAlignmentCallback: Betley alignment eval (checkpoint-based, slower)
- PeriodicLeakageCallback: Marker token leakage eval (checkpoint-based)
- MarkerBandStopCallback: Deterministic early-stop when marker source log-prob
  enters the useful transient band [low_nats, high_nats] above base. Logs the
  per-step marker log-prob trajectory and triggers ``should_training_stop``
  the first time the source enters the band after ``min_steps``.

The periodic eval callbacks use percentage-based scheduling (every N% of
training), following external/training-against-misalignment/ppt/trainers/
ood_callback.py. ``MarkerBandStopCallback`` uses absolute step intervals
(``eval_every_steps``) so the trajectory has a fixed step resolution
independent of the run's epoch count.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import ClassVar

import wandb
from transformers import TrainerCallback

from explore_persona_space.personas import MARKER_TOKEN

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
        except Exception as e:
            logger.warning("Memory check failed: %s. Proceeding anyway.", e)

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
        marker_token: str = MARKER_TOKEN,
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


def _decide_band_stop(
    delta_nats: float,
    global_step: int,
    *,
    low_nats: float,
    high_nats: float,
    min_steps: int,
) -> bool:
    """Pure decision function for the marker band-stop predicate.

    Stop iff ``low_nats <= delta_nats <= high_nats`` AND ``global_step >= min_steps``.

    Exposed as a module-level function so it can be unit-tested without a
    real model (the callback's only model-touching code is the log-prob
    read; the decision logic is pure).

    Args:
        delta_nats: ``trained - base`` log P(marker), in nats, at the
            post-response source-probe slot (averaged over the probe batch).
        global_step: Current training step.
        low_nats: Lower edge of the useful band (Regime A default 5.0).
        high_nats: Upper edge of the useful band (Regime A default 12.0).
        min_steps: Minimum step count before stopping is allowed (guard
            against stopping on a transient noisy first-eval reading).

    Returns:
        True iff training should stop now.
    """
    if global_step < min_steps:
        return False
    return low_nats <= delta_nats <= high_nats


class MarkerBandStopCallback(TrainerCallback):
    """Deterministic early-stop when source marker log-prob enters the useful band.

    The marker log-prob is a monotone ramp from a deep floor (``log P_base(marker)``
    ≈ -19 nat for ` ※` on Qwen-2.5-7B base) toward a 0-nat ceiling. The
    useful regime for measuring leakage selectivity is a narrow transient
    band where ``trained - base`` ∈ ``[low_nats, high_nats]`` (Regime A
    default [5, 12] nat) — the source has emerged off the floor but
    bystanders still have headroom (not yet saturated to argmax).

    Because a fixed epoch count lands at different log-probs per source /
    seed / data size, this callback replaces "train for N epochs" with
    "train until the source log P(marker) enters the band." It also logs
    the per-step trajectory of ``log P(marker)`` and ``delta_nats`` — the
    dynamics signal that open-q 2.2 has been missing.

    The DV read here is a teacher-forced ``log P(marker | T(q) + R)`` at
    the post-response slot on a fixed source-probe batch (the source
    persona's own positive rows from the training data, marker stripped).
    This teacher-forced read is valid as a WITHIN-CONDITION trajectory
    per ``.claude/rules/marker-leakage-measurement.md`` — it is NOT a
    cross-condition behavioral DV (on-policy bystander reads stay in the
    downstream eval).

    Args:
        marker_token_ids: Token id sequence for the marker (typically
            ``[83399]`` for ` ※` on Qwen-2.5-7B).
        probe_input_ids: ``[B, T_max]`` int64 source-probe input ids,
            padded with ``pad_token_id`` on the right.
        probe_marker_positions: ``[B]`` int64 indices of the marker slot
            per probe row (the position whose log-prob is the DV — i.e.
            the position at which the model PREDICTS the marker, so the
            slot's input is the token immediately BEFORE the marker).
        probe_attention_mask: ``[B, T_max]`` int64 attention mask (1 for
            real tokens, 0 for padding).
        low_nats: Lower edge of the band (Regime A default 5.0).
        high_nats: Upper edge of the band (Regime A default 12.0).
        eval_every_steps: How often to read the marker log-prob (default
            every 10 optimizer steps).
        min_steps: Minimum step before the stop predicate fires.
        log_prefix: WandB metric namespace.
        eos_token_id: Optional token id of the EOS competitor at the marker
            slot (``<|im_end|>`` id 151645 under the Qwen-2.5 chat template).
            When provided, the per-step WandB trajectory also logs the raw
            ``z_eos`` logit at the slot alongside ``z_marker`` and ``logZ``
            (the "report BOTH log-prob and logit" rule,
            ``.claude/rules/marker-leakage-measurement.md``). When None the
            z_eos series is skipped; the band-stop DECISION is unaffected
            either way (it stays on the log-prob band).
        log_only: When True (issue #480 band-stopped-anchor-rerun), the
            callback NEVER sets ``should_training_stop`` — training runs to
            its planned schedule and the anchor is picked post-hoc from the
            checkpoint ladder. All WandB + trajectory logging is kept; the
            first band entry is still logged (as ``band_entry_step``) so the
            event remains findable. Default False → live-stop behavior is
            byte-identical for every existing caller.
        trajectory_out_path: Optional local JSON path. When set, a per-probe
            record (step + the four-float set, trained AND base) is appended
            and the file is atomically rewritten after EVERY probe
            (checkpoint-per-phase discipline — a mid-run crash never loses
            the trajectory). Top-level parallel arrays ``steps`` /
            ``log_p_marker`` match the schema ``i480_analyze.py``'s
            trajectory figure consumes; ``records`` carries the full
            per-probe dicts. Default None → no local file.
        snapshot_every_steps: Task #534 sub-stop checkpointing. When > 0 (AND
            ``snapshot_dir`` is set), save a PEFT adapter-only snapshot of the
            model every ``snapshot_every_steps`` optimizer steps to
            ``<snapshot_dir>/step_<NNNN>/``. The snapshot fires BEFORE the
            band-stop predicate within the same ``on_step_end`` call, so the
            stop-step snapshot itself always exists (the post-hoc fraction
            selector maps frac=1.00 onto it). Default 0 → every existing
            caller is byte-identical (no snapshot, no sidecar).
        snapshot_dir: Directory for the per-step snapshots + the
            ``band_stop_meta.json`` sidecar written at train end. ``None``
            (default) disables both.
        snapshot_max_count: Hard cap on the number of snapshots written
            (disk bound for a run that never band-stops; #534 plan §4.3
            grounds 64 at 3.2x the realized stop step).
    """

    def __init__(
        self,
        marker_token_ids: list[int],
        # torch.Tensor parameters typed at call time to avoid a top-level
        # torch import (this module is imported by callers that don't always
        # need torch eagerly).
        probe_input_ids,  # torch.Tensor [B, T_max]
        probe_marker_positions,  # torch.Tensor [B]
        probe_attention_mask,  # torch.Tensor [B, T_max]
        *,
        low_nats: float = 5.0,
        high_nats: float = 12.0,
        eval_every_steps: int = 10,
        min_steps: int = 20,
        log_prefix: str = "marker",
        eos_token_id: int | None = None,
        log_only: bool = False,
        trajectory_out_path: str | None = None,
        snapshot_every_steps: int = 0,
        snapshot_dir: Path | str | None = None,
        snapshot_max_count: int = 64,
    ):
        if not marker_token_ids:
            raise ValueError("MarkerBandStopCallback requires a non-empty marker_token_ids")
        if low_nats >= high_nats:
            raise ValueError(
                f"low_nats ({low_nats}) must be strictly less than high_nats ({high_nats})"
            )
        if eval_every_steps < 1:
            raise ValueError(f"eval_every_steps must be >= 1, got {eval_every_steps}")
        if min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        if snapshot_every_steps < 0:
            raise ValueError(f"snapshot_every_steps must be >= 0, got {snapshot_every_steps}")
        if snapshot_max_count < 1:
            raise ValueError(f"snapshot_max_count must be >= 1, got {snapshot_max_count}")
        if snapshot_every_steps > 0 and snapshot_dir is None:
            raise ValueError(
                "snapshot_every_steps > 0 requires snapshot_dir — refusing to "
                "silently skip the per-step snapshots the caller asked for."
            )

        self.marker_token_ids = list(marker_token_ids)
        # The marker DV here is log P(FIRST marker token) at the slot whose
        # output is that token. For a multi-token marker, the convention is to
        # score the first token (the conditional emission probability of the
        # marker as a whole bottoms out on the first token's probability under
        # greedy / argmax reads, and the recipe doc only specifies the single-
        # token ` ※` default). Warn once so a multi-token marker run knows
        # the DV is the first-token approximation, not the full marker prob.
        if len(self.marker_token_ids) > 1:
            logger.warning(
                "MarkerBandStopCallback: marker_token_ids has %d tokens (%s); "
                "the band-stop DV uses ONLY the first token's log-prob as the "
                "marker probability approximation. The canonical single-token "
                "marker is ` ※` (id 83399 on Qwen-2.5-7B); a multi-token marker "
                "is supported but the band edges [low_nats, high_nats] should "
                "be re-tuned for it.",
                len(self.marker_token_ids),
                self.marker_token_ids,
            )
        self._target_token_id = self.marker_token_ids[0]
        self.probe_input_ids = probe_input_ids
        self.probe_marker_positions = probe_marker_positions
        self.probe_attention_mask = probe_attention_mask
        self.low_nats = float(low_nats)
        self.high_nats = float(high_nats)
        self.eval_every_steps = int(eval_every_steps)
        self.min_steps = int(min_steps)
        self.log_prefix = log_prefix
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None
        self.log_only = bool(log_only)
        self.trajectory_out_path = trajectory_out_path
        self.snapshot_every_steps = int(snapshot_every_steps)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None
        self.snapshot_max_count = int(snapshot_max_count)

        # Tensor-shape asserts at the construction boundary.
        assert probe_input_ids.ndim == 2, probe_input_ids.shape
        assert probe_attention_mask.shape == probe_input_ids.shape, (
            probe_attention_mask.shape,
            probe_input_ids.shape,
        )
        assert probe_marker_positions.ndim == 1, probe_marker_positions.shape
        assert probe_marker_positions.shape[0] == probe_input_ids.shape[0], (
            probe_marker_positions.shape,
            probe_input_ids.shape,
        )

        self._base_logp_per_row = None  # torch.Tensor [B]; cached on first eval
        self._base_logp_mean = None  # float
        self._base_slot_stats = None  # dict[str, torch.Tensor]; cached on first eval
        self._stopped = False
        # Set in on_train_begin when the planned run is too short to reach
        # the band meaningfully (max_steps < min_steps): we no-op for the
        # whole phase so the run completes its planned schedule without a
        # silent never-fire.
        self._disabled_too_short = False
        # Per-probe trajectory records (flushed to trajectory_out_path after
        # every probe) + a once-per-phase flag so log_only band entry is
        # logged exactly once.
        self._trajectory_records: list[dict] = []
        self._band_entry_logged = False
        # #534 sub-stop checkpointing state: which steps were snapshotted +
        # the in-loop eval-read history persisted into band_stop_meta.json.
        self._snapshot_steps: list[int] = []
        self._eval_history: list[dict] = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Reset per-phase state (callback may be reused across phases).

        If the planned ``max_steps`` is below ``min_steps`` the callback
        cannot fire meaningfully (the guard predicate would block every
        in-band reading). Warn once and disable for the phase so the run
        completes its planned schedule instead of silently never stopping.
        """
        self._base_logp_per_row = None
        self._base_logp_mean = None
        self._base_slot_stats = None
        self._stopped = False
        self._disabled_too_short = False
        self._trajectory_records = []
        self._band_entry_logged = False
        self._snapshot_steps = []
        self._eval_history = []
        if state.max_steps > 0 and state.max_steps < self.min_steps:
            logger.warning(
                "[%s] max_steps=%d < min_steps=%d — the band-stop guard "
                "would block every reading. Disabling the band-stop for "
                "this phase; training will run to its planned schedule. "
                "Lower marker_band_min_steps or raise the run length to "
                "use band-stop on short runs.",
                self.log_prefix,
                state.max_steps,
                self.min_steps,
            )
            self._disabled_too_short = True

    def _maybe_snapshot_adapter(self, state, model) -> None:
        """#534 sub-stop checkpointing: per-step adapter snapshot.

        Fires BEFORE the eval-cadence gate and BEFORE the stop predicate, so
        the stop-step snapshot itself exists (the post-hoc fraction selector
        maps frac=1.00 onto the realized stop step). ``model.save_pretrained``
        on the PEFT-wrapped model serializes the adapter only (~81 MB at
        r=8), consumes no RNG, and touches neither optimizer nor schedule —
        passive w.r.t. the weight trajectory (plan #534 §4.1).
        """
        if (
            self.snapshot_every_steps > 0
            and self.snapshot_dir is not None
            and not self._stopped
            and state.global_step > 0
            and state.global_step % self.snapshot_every_steps == 0
            and len(self._snapshot_steps) < self.snapshot_max_count
        ):
            snap_dir = self.snapshot_dir / f"step_{state.global_step:04d}"
            if not snap_dir.exists():  # idempotent on re-entry
                model.save_pretrained(str(snap_dir))
            self._snapshot_steps.append(int(state.global_step))
            if len(self._snapshot_steps) == self.snapshot_max_count:
                logger.warning(
                    "[%s] snapshot_max_count=%d reached at step %d — no further "
                    "per-step snapshots will be written (cells beyond the cap "
                    "are excluded from replication claims; see fraction "
                    "manifest `exact` flags).",
                    self.log_prefix,
                    self.snapshot_max_count,
                    state.global_step,
                )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Snapshot the adapter (when configured), then read marker log-prob; stop iff in band."""
        if model is None:
            return
        self._maybe_snapshot_adapter(state, model)
        if self._stopped or self._disabled_too_short:
            return
        if state.global_step <= 0 or state.global_step % self.eval_every_steps != 0:
            return

        # Cache base log-prob the first time we run, with the adapter disabled.
        # PEFT exposes ``disable_adapter()`` as a context manager (PEFT >=0.4);
        # outside the with-block the adapter is auto-re-enabled. If the model
        # is not PEFT-wrapped (defensive — should never happen on this code
        # path, since train_lora always wraps), the "base" read falls back to
        # the model as-is, which the caller will see as a delta-of-zero.
        if self._base_logp_per_row is None:
            self._base_slot_stats = self._read_slot_stats_with_base(model)
            self._base_logp_per_row = self._base_slot_stats["logp"]
            self._base_logp_mean = float(self._base_logp_per_row.mean().item())
            logger.info(
                "[%s] Cached base log P(marker) at step %d: mean=%.4f nat over %d probe rows",
                self.log_prefix,
                state.global_step,
                self._base_logp_mean,
                int(self._base_logp_per_row.shape[0]),
            )

        trained_stats = self._read_slot_stats_trained(model)
        trained_per_row = trained_stats["logp"]
        trained_mean = float(trained_per_row.mean().item())
        delta_per_row = trained_per_row - self._base_logp_per_row.to(trained_per_row.device)
        delta_mean = float(delta_per_row.mean().item())
        # #534: keep the in-loop eval-read history for the band_stop_meta.json
        # sidecar (cheap; list of small dicts, populated only at eval cadence).
        self._eval_history.append(
            {
                "step": int(state.global_step),
                "delta_nats": float(delta_mean),
                "trained_logp_mean": float(trained_mean),
                "base_logp_mean": float(self._base_logp_mean),
            }
        )

        logger.info(
            "[%s] Step %d: trained log P(marker)=%.4f nat, base=%.4f nat, delta=%+.4f nat "
            "(band=[%.2f, %.2f], min_steps=%d)",
            self.log_prefix,
            state.global_step,
            trained_mean,
            self._base_logp_mean,
            delta_mean,
            self.low_nats,
            self.high_nats,
            self.min_steps,
        )

        if wandb.run is not None:
            # Log-prob (PRIMARY, the band-stop DV) + raw-logit (SECONDARY,
            # non-saturating mechanistic readout) trajectories from the SAME
            # forward pass — "report BOTH log-prob and logit"
            # (.claude/rules/marker-leakage-measurement.md). The band-stop
            # DECISION below stays on the log-prob band, unchanged.
            metrics = {
                f"{self.log_prefix}/source_logp_mean": trained_mean,
                f"{self.log_prefix}/source_logp_base_mean": self._base_logp_mean,
                f"{self.log_prefix}/source_delta_nats": delta_mean,
                f"{self.log_prefix}/z_marker_trained": float(
                    trained_stats["z_marker"].mean().item()
                ),
                f"{self.log_prefix}/z_marker_base": float(
                    self._base_slot_stats["z_marker"].mean().item()
                ),
                f"{self.log_prefix}/delta_z_marker": float(
                    (
                        trained_stats["z_marker"]
                        - self._base_slot_stats["z_marker"].to(trained_stats["z_marker"].device)
                    )
                    .mean()
                    .item()
                ),
                f"{self.log_prefix}/logZ_trained": float(trained_stats["logZ"].mean().item()),
                f"{self.log_prefix}/logZ_base": float(self._base_slot_stats["logZ"].mean().item()),
            }
            if trained_stats["z_eos"] is not None:
                metrics[f"{self.log_prefix}/z_eos_trained"] = float(
                    trained_stats["z_eos"].mean().item()
                )
                metrics[f"{self.log_prefix}/z_eos_base"] = float(
                    self._base_slot_stats["z_eos"].mean().item()
                )
            wandb.log(metrics, step=state.global_step)

        if self.trajectory_out_path is not None:
            self._trajectory_records.append(
                {
                    "step": int(state.global_step),
                    "logp_trained": trained_mean,
                    "logp_base": self._base_logp_mean,
                    "delta_nats": delta_mean,
                    "z_marker_trained": float(trained_stats["z_marker"].mean().item()),
                    "z_marker_base": float(self._base_slot_stats["z_marker"].mean().item()),
                    "z_eos_trained": (
                        float(trained_stats["z_eos"].mean().item())
                        if trained_stats["z_eos"] is not None
                        else None
                    ),
                    "z_eos_base": (
                        float(self._base_slot_stats["z_eos"].mean().item())
                        if self._base_slot_stats["z_eos"] is not None
                        else None
                    ),
                    "logZ_trained": float(trained_stats["logZ"].mean().item()),
                    "logZ_base": float(self._base_slot_stats["logZ"].mean().item()),
                }
            )
            # Checkpoint-per-phase: rewrite the trajectory JSON after EVERY
            # probe so a mid-run crash never loses the trajectory.
            self._write_trajectory()

        should_stop = _decide_band_stop(
            delta_mean,
            state.global_step,
            low_nats=self.low_nats,
            high_nats=self.high_nats,
            min_steps=self.min_steps,
        )
        if should_stop and self.log_only:
            # Log-only mode (issue #480 band-stopped-anchor-rerun): record
            # the FIRST band entry loudly (log line + WandB scalar) but never
            # touch the Trainer control flags — training runs to its cap and
            # the anchor pick happens post-hoc from the checkpoint ladder.
            if not self._band_entry_logged:
                logger.warning(
                    "[%s] BAND ENTERED at step %d: delta=%+.4f nat ∈ [%.2f, %.2f] "
                    "— log_only=True, NOT stopping (training runs to its cap; "
                    "anchor picked post-hoc from the checkpoint ladder).",
                    self.log_prefix,
                    state.global_step,
                    delta_mean,
                    self.low_nats,
                    self.high_nats,
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            f"{self.log_prefix}/band_entry_step": state.global_step,
                            f"{self.log_prefix}/band_entry_delta_nats": delta_mean,
                        },
                        step=state.global_step,
                    )
                self._band_entry_logged = True
            return
        if should_stop:
            # Bold log line + WandB scalar so the early termination is
            # never silent. Default-on band-stop changes the run length,
            # so this needs to be findable in logs and in the WandB run
            # without grepping for the trajectory series.
            logger.warning(
                "[%s] BAND-STOP TRIGGERED at step %d: delta=%+.4f nat ∈ "
                "[%.2f, %.2f] and step >= min_steps=%d. Setting "
                "should_training_stop=True + should_save=True. The run "
                "will terminate after this step. Disable with "
                "TrainLoraConfig(marker_band_stop=False).",
                self.log_prefix,
                state.global_step,
                delta_mean,
                self.low_nats,
                self.high_nats,
                self.min_steps,
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        f"{self.log_prefix}/band_stop_step": state.global_step,
                        f"{self.log_prefix}/band_stop_delta_nats": delta_mean,
                    },
                    step=state.global_step,
                )
            control.should_training_stop = True
            control.should_save = True
            self._stopped = True

    def on_train_end(self, args, state, control, **kwargs):
        """Final trajectory flush + the #534 ``band_stop_meta.json`` sidecar.

        Union of the two lineages: (a) main's trajectory-JSON final flush
        (per-probe flushes already persisted everything, this is belt-and-
        braces); (b) the #534 snapshot-extension sidecar recording the
        REALIZED stop step, stop reason, in-loop eval-read history, and the
        snapshotted steps — everything the post-hoc fraction selector
        (``scripts/i534_select_fractions.py``) needs. Exact no-op for legacy
        callers (``trajectory_out_path is None and snapshot_dir is None``).
        """
        if self.trajectory_out_path is not None and self._trajectory_records:
            self._write_trajectory()
            logger.info(
                "[%s] trajectory JSON final flush: %d probe records -> %s",
                self.log_prefix,
                len(self._trajectory_records),
                self.trajectory_out_path,
            )
        if self.snapshot_dir is None:
            return  # legacy callers: exact no-op for the sidecar
        if self._disabled_too_short:
            stop_reason = "disabled_too_short"
        elif self._stopped:
            stop_reason = "band"
        else:
            stop_reason = "epoch_ceiling"
        meta = {
            "stopped": bool(self._stopped),
            "stop_step": int(state.global_step),
            "stop_reason": stop_reason,
            "eval_history": self._eval_history,
            "snapshot_steps": self._snapshot_steps,
            "band": [self.low_nats, self.high_nats],
            "eval_every_steps": self.eval_every_steps,
            "min_steps": self.min_steps,
            "max_steps": int(state.max_steps),
            "snapshot_every_steps": self.snapshot_every_steps,
            "snapshot_max_count": self.snapshot_max_count,
        }
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        out = self.snapshot_dir / "band_stop_meta.json"
        out.write_text(json.dumps(meta, indent=2))
        logger.info(
            "[%s] wrote band_stop_meta.json → %s (stopped=%s, stop_step=%d, "
            "stop_reason=%s, n_snapshots=%d)",
            self.log_prefix,
            out,
            meta["stopped"],
            meta["stop_step"],
            stop_reason,
            len(self._snapshot_steps),
        )

    def _write_trajectory(self) -> None:
        """Atomically rewrite the trajectory JSON from the accumulated records.

        Top-level parallel arrays (``steps`` / ``log_p_marker`` / ...) match
        the schema consumed by ``i480_analyze.py``'s trajectory figure; the
        full per-probe dicts ride along under ``records``. Write goes to a
        tmp file + ``os.replace`` so a crash mid-write never leaves a
        truncated JSON.
        """
        assert self.trajectory_out_path is not None
        recs = self._trajectory_records
        payload = {
            "schema": "marker_band_trajectory_v1",
            "log_prefix": self.log_prefix,
            "marker_token_ids": self.marker_token_ids,
            "eos_token_id": self.eos_token_id,
            "log_only": self.log_only,
            "band_low_nats": self.low_nats,
            "band_high_nats": self.high_nats,
            "n_probe_records": len(recs),
            "steps": [r["step"] for r in recs],
            "log_p_marker": [r["logp_trained"] for r in recs],
            "log_p_base": [r["logp_base"] for r in recs],
            "delta_nats": [r["delta_nats"] for r in recs],
            "z_marker_trained": [r["z_marker_trained"] for r in recs],
            "z_marker_base": [r["z_marker_base"] for r in recs],
            "z_eos_trained": [r["z_eos_trained"] for r in recs],
            "z_eos_base": [r["z_eos_base"] for r in recs],
            "logZ_trained": [r["logZ_trained"] for r in recs],
            "logZ_base": [r["logZ_base"] for r in recs],
            "records": recs,
        }
        out_path = str(self.trajectory_out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, out_path)

    def _read_logp_trained(self, model):
        """Read mean log P(marker) at the marker slot under the trained adapter."""
        return self._compute_marker_logp(model)

    def _read_logp_with_base(self, model):
        """Read mean log P(marker) under the BASE model (adapter disabled).

        Uses PEFT's ``disable_adapter()`` context manager when available
        (the canonical way to compare trained vs base in PEFT >=0.4 — see
        ``.claude/rules/persona-distance-metrics.md`` and the #466 plan).
        Falls back to a direct read when the model is not PEFT-wrapped
        (defensive — the train_lora call site always wraps the model).
        """
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            with disable_adapter():
                return self._compute_marker_logp(model)
        return self._compute_marker_logp(model)

    def _read_slot_stats_trained(self, model):
        """Slot stats (logp + raw logits) under the trained adapter."""
        return self._compute_marker_slot_stats(model)

    def _read_slot_stats_with_base(self, model):
        """Slot stats (logp + raw logits) under the BASE model (adapter disabled)."""
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            with disable_adapter():
                return self._compute_marker_slot_stats(model)
        return self._compute_marker_slot_stats(model)

    def _compute_marker_logp(self, model):
        """One teacher-forced forward pass; return log P(marker) per probe row.

        Thin wrapper over :meth:`_compute_marker_slot_stats` kept for
        backward compatibility (tests + the base/trained logp readers).

        Returns:
            ``torch.Tensor`` of shape ``[B]`` with the per-row log-prob of
            the first marker token at its designated slot.
        """
        return self._compute_marker_slot_stats(model)["logp"]

    def _compute_marker_slot_stats(self, model):
        """One teacher-forced forward pass; per-row marker-slot stats.

        Single-GPU assumption: probes are moved to ``model.device`` (or the
        first parameter's device). The train_lora LoRA path always pins to
        one physical GPU via CUDA_VISIBLE_DEVICES=str(gpu_id), so this is
        always single-GPU on the current callers; multi-GPU DDP/FSDP would
        need a per-rank/all-gather rework.

        The raw-logit readouts ride the SAME forward pass as the log-prob
        (``log P(marker) = z_marker - logZ`` exactly, per
        ``.claude/rules/marker-leakage-measurement.md`` § "Report BOTH
        log-prob and logit") — zero extra model compute.

        Returns:
            dict with CPU tensors of shape ``[B]``:
            ``logp`` (log P of the first marker token at its slot),
            ``z_marker`` (raw pre-softmax logit at the marker id),
            ``z_eos`` (raw logit at ``self.eos_token_id``; None when the
            callback was constructed without an eos id),
            ``logZ`` (full-vocab logsumexp at the slot).
        """
        import torch

        # Locate device from the model. PEFT wrappers expose ``.device`` via
        # the wrapped base model in most setups; fall back to the first
        # parameter's device if needed.
        device = getattr(model, "device", None) or next(model.parameters()).device

        input_ids = self.probe_input_ids.to(device)
        attention_mask = self.probe_attention_mask.to(device)
        positions = self.probe_marker_positions.to(device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]
            assert logits.ndim == 3, logits.shape
            batch_idx = torch.arange(input_ids.shape[0], device=device)
            # The marker's predictive distribution is read at the OUTPUT
            # position whose argmax would be the marker token. The caller
            # passes ``positions`` already aligned to that output slot
            # (i.e. positions[i] is the index at which logits[i, positions[i]]
            # is the distribution over the NEXT token = the marker).
            slot_logits = logits[batch_idx, positions, :].float()  # [B, V]
            assert slot_logits.shape == (input_ids.shape[0], logits.shape[-1]), slot_logits.shape
            log_z = torch.logsumexp(slot_logits, dim=-1)  # [B]
            z_marker = slot_logits[:, self._target_token_id]  # [B]
            row_logp = z_marker - log_z  # exact identity: logp = z_marker - logZ
            assert row_logp.shape == (input_ids.shape[0],), row_logp.shape
            z_eos = (
                slot_logits[:, self.eos_token_id].detach().cpu()
                if self.eos_token_id is not None
                else None
            )
            return {
                "logp": row_logp.detach().cpu(),
                "z_marker": z_marker.detach().cpu(),
                "z_eos": z_eos,
                "logZ": log_z.detach().cpu(),
            }
        finally:
            if was_training:
                model.train()
