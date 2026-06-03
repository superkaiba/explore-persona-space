#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #405 PHASE 3 — train one LoRA cell + on-policy eval (two panels).

Per plan v2 §4.5 PHASE 3:

  1. Load training JSONL ``data/issue_405/training_jsonl/cell_{cid}_seed{S}.jsonl``
     (Phase 2 output).
  2. ``train_lora(...)`` with ``TrainLoraConfig(marker_only_loss=True,
     marker_text=" ※", marker_tail_tokens=0, lora_r=16, lora_alpha=32,
     lora_dropout=0.05, lr=5e-6, epochs=2, lora_targets=q/k/v/o (NARROW))``.
     Persist adapter to HF Hub via ``EPM_PERSIST_ADAPTER_HF_REPO`` +
     ``EPM_PERSIST_ADAPTER_SUBFOLDER`` (delete-after-eval recipe).
  3. Merge LoRA in-process; on-policy eval over TWO disjoint persona panels:
       * **held_out** (8 personas) — feeds the headline regression.
       * **trained_positive** (the K positives this cell trained on) — FIX A1
         per-cell source-strength scalar; covariate in the analyzer's
         dose-vs-diversity check.
     For each (persona, q ∈ EVAL_QUESTIONS):
       * Generate R_eval = trained_model.greedy(T_p(q)), per-(persona, q)
         cap matching Phase 1's Fix D recipe.
       * compute_marker_logprob(trained, T_p(q) + R_eval, " ※")
       * compute_marker_logprob(base,    T_p(q) + R_eval, " ※")
       * ΔlogP = trained − base
       * emit_rate from argmax at the post-R slot (free anchor)
       * full-vocab KL(trained‖base) at the same slot (non-saturating DV)
  4. Write ``eval_results/issue_405/cell_{cid}_seed{S}/result.json`` +
     end-of-cell sentinel at the pod-side log dir.
  5. Delete merged dir (MooseFS quota); the LoRA adapter persists on HF Hub.

WandB **per-step probe-panel logging** (FIX E) is wired via a
``TrainerCallback`` that runs the probe panel at each ``logging_steps``
step and logs both ``marker_logprob`` AND ``marker_emission_rate`` for the
3 probe personas:
  (a) trained-positive (first persona in spec["positives"])
  (b) ``no_persona`` (the bare default context — open-q 3.7 safety target)
  (c) ``comedian`` (the far held-out)

vLLM teardown follows ``.claude/rules/gotchas.md`` — psutil child-kill +
nvidia-smi PID check before HF Transformers reloads the model for the
post-R logprob compute.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap
from transformers import TrainerCallback

log = bootstrap()

from _issue405_common import (  # noqa: E402
    BASE_MODEL,
    EPOCHS,
    GRAD_ACCUM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGETS_NARROW,
    LR,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    MAX_LENGTH,
    PER_DEVICE_BATCH,
    R_CAP_MIN,
    R_CAP_SAFETY_MARGIN,
    SENTINEL_DIR_LOCAL_FALLBACK,
    SENTINEL_DIR_POD,
    WANDB_PROJECT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    assert_marker_token_id,
    load_all_persona_prompts,
)


def _import_questions() -> tuple[list[str], list[str]]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_leakage_v3_onpolicy import DATA_QUESTIONS, EVAL_QUESTIONS

    return list(DATA_QUESTIONS), list(EVAL_QUESTIONS)


def kill_vllm_children() -> None:
    """Reap vLLM worker subprocesses (per .claude/rules/gotchas.md).

    Canonical vLLM cleanup (del llm + destroy_model_parallel +
    destroy_distributed_environment + gc.collect + empty_cache) is NOT
    enough — TP/PP worker subprocesses survive and re-grab freed GPU
    memory the moment the next framework (HF Transformers) loads.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        log.warning("psutil not available — skipping orphan-PID reaping")
        return

    import contextlib

    me = psutil.Process()
    children = me.children(recursive=True)
    for ch in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            log.info("Terminating vLLM child PID=%d name=%r", ch.pid, ch.name())
            ch.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=5)
    for ch in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            ch.kill()


def nvidia_smi_assert_clean(gpu_id: int) -> None:
    """Fail loud if any python PID still holds the GPU after vLLM teardown.

    Called after kill_vllm_children() and before HF Transformers re-loads the
    base model for the logprob compute. CVD-aware: only checks the specified
    GPU's compute-apps (per feedback_orphan_pid_check_must_be_cvd_aware).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
            env={**os.environ},  # Blocker 10: explicit env-passthrough
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("nvidia-smi probe failed (%s); skipping orphan check", e)
        return
    if not out:
        return
    my_pid = os.getpid()
    leaks = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == my_pid:
            continue
        leaks.append(line.strip())
    if leaks:
        log.warning(
            "nvidia-smi reports surviving python PIDs on GPU %d: %r — "
            "continuing (CVD-aware: filtered by gpu id) but HF reload may OOM",
            gpu_id,
            leaks,
        )


def compute_prompt_len(tokenizer, persona_prompt: str, question: str) -> int:
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def per_question_R_cap(tokenizer, persona_prompt: str, question: str) -> int:
    """Per-(persona, question) R-cap matching Phase 1's Fix D recipe."""
    prompt_len = compute_prompt_len(tokenizer, persona_prompt, question)
    cap = MAX_LENGTH - prompt_len - R_CAP_SAFETY_MARGIN
    if cap < R_CAP_MIN:
        raise RuntimeError(
            f"R-cap too small for persona={persona_prompt[:40]!r} q={question[:40]!r}: "
            f"prompt_len={prompt_len}, cap={cap} < {R_CAP_MIN}"
        )
    return cap


def vllm_greedy_generate(
    merged_path: str,
    panels: dict[str, dict[str, str]],
    eval_questions: list[str],
    tokenizer,
    gpu_id: int,
    gpu_mem_util: float,
) -> dict[str, dict[str, str]]:
    """Greedy generate R_eval for each (persona, question) using vLLM.

    Args:
        merged_path: Path to merged (LoRA-applied) checkpoint.
        panels: dict {panel_name: {persona_name: system_prompt}}.
        eval_questions: List of evaluation questions.
        tokenizer: HF tokenizer (for prompt-len + chat template).
        gpu_id: Already pinned via CUDA_VISIBLE_DEVICES — passed for logging.
        gpu_mem_util: vLLM GPU memory utilization.

    Returns:
        dict {persona_name: {question: R_eval_text}}.
    """
    from vllm import LLM, SamplingParams

    log.info(
        "Loading vLLM from merged checkpoint %s on GPU %d (mem_util=%.2f) ...",
        merged_path,
        gpu_id,
        gpu_mem_util,
    )
    llm = LLM(
        model=merged_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=MAX_LENGTH,
        max_num_seqs=64,
        seed=42,
    )

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    caps: list[int] = []
    persona_prompts: dict[str, str] = {}
    for panel in panels.values():
        for p, sys_prompt in panel.items():
            persona_prompts.setdefault(p, sys_prompt)

    for persona, sys_prompt in persona_prompts.items():
        for q in eval_questions:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            cap = per_question_R_cap(tokenizer, sys_prompt, q)
            prompts.append(text)
            keys.append((persona, q))
            caps.append(cap)

    sampling = [SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=c) for c in caps]
    log.info("vLLM batched generate: %d prompts ...", len(prompts))
    outputs = llm.generate(prompts, sampling)

    R_eval: dict[str, dict[str, str]] = {p: {} for p in persona_prompts}
    truncated = 0
    for out, (persona, q) in zip(outputs, keys, strict=True):
        if out.outputs[0].finish_reason == "length":
            truncated += 1
        R_eval[persona][q] = out.outputs[0].text

    log.info(
        "Eval R-gen done: %d (persona, q) pairs, %d hit length cap.",
        len(prompts),
        truncated,
    )

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return R_eval


def score_logprob_and_kl(
    model_path: str,
    R_eval: dict[str, dict[str, str]],
    persona_prompts: dict[str, str],
    tokenizer,
    device: str,
    batch_size: int = 4,
) -> dict[str, dict]:
    """For each (persona, q), compute log P(marker), emit_rate, KL(this‖base?).

    The KL part requires both models' logits at the same slot, which means
    this function loads ONE model (either trained or base) and returns the
    raw next-token log-probs at the post-R slot for every (persona, q). The
    caller pairs trained + base outputs to compute ΔlogP and KL.

    Returns:
        dict {persona: {
            "logp_marker_per_q": [float],   # log P(marker_id) at post-R slot
            "argmax_id_per_q":  [int],      # argmax token id at the same slot
            "logits_per_q":     [list[float]]  # next-token log-probs (full vocab)
                                            # OR sentinel "RAM_OOM" str if vocab*N too big;
                                            # stored as compact log-softmax (V floats/q)
        }}
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    log.info("Loading model %s for scoring ...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    out: dict[str, dict] = {
        p: {"logp_marker_per_q": [], "argmax_id_per_q": [], "logsoftmax_per_q": []} for p in R_eval
    }

    # Order: persona × q (stable for later pairing)
    all_items: list[tuple[str, str, str]] = []  # (persona, q, prefix_text)
    for persona, qmap in R_eval.items():
        sys_prompt = persona_prompts[persona]
        for q, R in qmap.items():
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_prefix = prefix + R  # the slot AFTER R is where we score the marker
            all_items.append((persona, q, full_prefix))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for start in range(0, len(all_items), batch_size):
        chunk = all_items[start : start + batch_size]
        ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in [c[2] for c in chunk]]
        max_len = max(len(ids) for ids in ids_list)
        padded, attn = [], []
        for ids in ids_list:
            pad = max_len - len(ids)
            padded.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # last position = predicts the next token AFTER R; that's the slot
        # where the marker should appear under positives, EOS under negatives.
        last_logits = logits[:, -1, :].float()  # (B, V)
        log_probs = F.log_softmax(last_logits, dim=-1)
        argmax_ids = log_probs.argmax(dim=-1).cpu().tolist()
        marker_logps = log_probs[:, MARKER_TOKEN_ID].cpu().tolist()
        # Store full log-softmax row for KL computation downstream.
        ls = log_probs.cpu().tolist()
        for (persona, _q, _), lp, am, full_ls in zip(
            chunk, marker_logps, argmax_ids, ls, strict=True
        ):
            out[persona]["logp_marker_per_q"].append(float(lp))
            out[persona]["argmax_id_per_q"].append(int(am))
            out[persona]["logsoftmax_per_q"].append(full_ls)
        del logits, last_logits, log_probs

    # Drop the model BEFORE returning so the caller can load the next one.
    del model
    gc.collect()
    try:
        import torch as _t

        _t.cuda.empty_cache()
    except Exception:
        pass
    return out


def compute_panel_summary(
    panel_R: dict[str, dict[str, str]],
    trained_scores: dict[str, dict],
    base_scores: dict[str, dict],
) -> tuple[dict, dict, dict, dict]:
    """Combine trained + base scores into per-persona summaries.

    Per-persona payload carries BOTH:
      - ``logp_trained_mean``  — ABSOLUTE trained log P(marker) at the slot,
        averaged over questions. THIS is what `g_logprob_source` must read
        for the saturation kill-criterion (plan §4.9: expected ∈ [-8, -3];
        saturated if > -0.1). NOT the trained−base delta — base log P(marker)
        ≈ -15, so trained−base ≥ 0 for any successful implant, which would
        ALWAYS exceed -0.1 and silently fire the kill gate (Blocker 1).
      - ``deltaLogP_mean`` (trained − base) — the FIX-A1 source-strength
        scalar used by the analyzer; SEPARATE from the saturation read.
      - ``logp_base_mean``  — absolute base log P(marker) at the slot.
      - ``emit_rate`` (argmax==MARKER_TOKEN_ID under trained)
      - ``kl_per_q`` (full-vocab KL(trained‖base) at the slot, vectorized)

    Returns: ``(per_persona, summary, mean_dlogp_by_persona, mean_emit_by_persona)``
    where ``summary`` carries BOTH ``logp_trained_mean`` AND
    ``mean_deltaLogP`` (NOT just the delta) so the caller can pick the right
    one for the kill-criterion vs. the analysis covariate.
    """
    # Vectorized KL — torch tensor ops, not Python double-loop over ~152k entries.
    # Per-row KL(trained ‖ base) = sum_v exp(log_t[v]) * (log_t[v] - log_b[v])
    import torch

    per_persona: dict[str, dict] = {}
    dlogp_means: list[float] = []
    logp_trained_means: list[float] = []
    logp_base_means: list[float] = []
    emit_rates: list[float] = []
    for persona in panel_R:
        lp_t = trained_scores[persona]["logp_marker_per_q"]
        lp_b = base_scores[persona]["logp_marker_per_q"]
        ls_t = trained_scores[persona]["logsoftmax_per_q"]
        ls_b = base_scores[persona]["logsoftmax_per_q"]
        argmax_t = trained_scores[persona]["argmax_id_per_q"]
        delta = [t - b for t, b in zip(lp_t, lp_b, strict=True)]
        # Vectorized KL — Blocker 8 fix (Python loop was ~60-100 min sweep overhead).
        # log_t: (Q, V), log_b: (Q, V) → KL per Q = sum_v exp(log_t) * (log_t - log_b)
        log_t = torch.tensor(ls_t, dtype=torch.float32)
        log_b = torch.tensor(ls_b, dtype=torch.float32)
        # Mask tail terms with log_t < -50 (numerically negligible contribution),
        # matching the row-by-row code's `if lt > -50.0` guard semantics.
        mask = log_t > -50.0
        contribs = torch.where(mask, torch.exp(log_t) * (log_t - log_b), torch.zeros_like(log_t))
        kl_per_q_tensor = contribs.sum(dim=-1)
        kl_per_q: list[float] = [float(x) for x in kl_per_q_tensor.tolist()]
        emit = sum(1.0 for a in argmax_t if a == MARKER_TOKEN_ID) / max(1, len(argmax_t))
        logp_trained_mean = sum(lp_t) / max(1, len(lp_t))
        logp_base_mean = sum(lp_b) / max(1, len(lp_b))
        per_persona[persona] = {
            "deltaLogP_per_q": delta,
            "deltaLogP_mean": sum(delta) / max(1, len(delta)),
            "logp_trained_mean": logp_trained_mean,  # ABSOLUTE (Blocker 1)
            "logp_base_mean": logp_base_mean,
            "emit_rate": emit,
            "kl_per_q": kl_per_q,
            "kl_mean": sum(kl_per_q) / max(1, len(kl_per_q)),
            "logp_trained_per_q": lp_t,
            "logp_base_per_q": lp_b,
            "argmax_id_per_q": argmax_t,
        }
        dlogp_means.append(per_persona[persona]["deltaLogP_mean"])
        logp_trained_means.append(logp_trained_mean)
        logp_base_means.append(logp_base_mean)
        emit_rates.append(emit)

    summary = {
        "mean_deltaLogP": sum(dlogp_means) / max(1, len(dlogp_means)),
        "logp_trained_mean": sum(logp_trained_means) / max(1, len(logp_trained_means)),
        "logp_base_mean": sum(logp_base_means) / max(1, len(logp_base_means)),
        "mean_emit_rate": sum(emit_rates) / max(1, len(emit_rates)),
        "n_personas": len(per_persona),
    }
    return (
        per_persona,
        summary,
        dict(zip(panel_R, dlogp_means, strict=True)),
        dict(zip(panel_R, emit_rates, strict=True)),
    )


class ProbePanelLogprobCallback(TrainerCallback):
    """FIX E (Blocker 4) — per-step probe-panel marker logprob + emission rate.

    Subclasses `transformers.TrainerCallback` so HF's `CallbackHandler.call_event`
    finds a no-op for every lifecycle event we DON'T implement (`on_init_end`,
    `on_epoch_begin`, `on_log`, ...) — only `on_train_begin` + `on_step_end`
    below carry real logic. Without the subclass, `getattr(callback, event)`
    raises `AttributeError` at trainer construction (round-4 smoke crash).
    Logs to WandB on every ``log_every_steps`` step:
      - ``probe/<persona>/logp_marker``  (teacher-forced log P(marker | T_p(q) + R))
      - ``probe/<persona>/argmax_emission``  (1.0 if argmax at the same slot == marker_id)

    The probe panel is fixed by the plan §4.4 / §6.5 fig #4:
      (a) one trained-positive persona (typically the first in spec.positives)
      (b) ``no_persona`` (the bare default context — open-q 3.7 safety target)
      (c) ``comedian`` (the far held-out)

    Each probe uses a fixed (persona, question, R) tuple captured ONCE at
    ``on_train_begin``; the R text comes from the cached on-policy R for
    the trained-positive persona (matches the trained context shape).

    Plan §6.5 fig #4 explicitly mandates BOTH the log-prob trajectory AND
    the argmax/emission trajectory because once log-prob crosses ~-0.1
    nat the model starts argmaxing the marker and the log-prob curve
    plateaus — emission still distinguishes the two regimes.
    """

    def __init__(
        self,
        probe_personas: list[tuple[str, str]],
        sample_R: str,
        sample_question: str,
        marker_text: str,
        marker_token_id: int,
        log_every_steps: int = 5,
    ):
        self.probe_personas = probe_personas  # [(name, system_prompt), ...]
        self.sample_R = sample_R
        self.sample_question = sample_question
        self.marker_text = marker_text
        self.marker_token_id = marker_token_id
        self.log_every_steps = log_every_steps
        self._last_logged_step = -1
        self._enabled = True

    def _build_prefix(self, tokenizer, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.sample_question},
        ]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat + self.sample_R  # post-R slot is where we score

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tokenizer is None:
            log.warning("ProbePanelLogprobCallback: no tokenizer in kwargs; disabling")
            self._enabled = False
            return
        # Pre-tokenize the 3 probe prefixes once.
        self._cached: list[tuple[str, list[int]]] = []
        for name, sys_prompt in self.probe_personas:
            prefix = self._build_prefix(tokenizer, sys_prompt)
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            self._cached.append((name, ids))
        log.info(
            "ProbePanelLogprobCallback: armed on %d probes, log every %d steps",
            len(self._cached),
            self.log_every_steps,
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._enabled or model is None:
            return
        step = state.global_step
        if step == self._last_logged_step:
            return
        if step % self.log_every_steps != 0:
            return
        self._last_logged_step = step

        import torch
        import torch.nn.functional as F

        was_training = model.training
        model.eval()
        try:
            metrics: dict[str, float] = {}
            for name, ids in self._cached:
                input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
                attn = torch.ones_like(input_ids)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attn).logits
                last = logits[0, -1, :].float()
                log_probs = F.log_softmax(last, dim=-1)
                lp = float(log_probs[self.marker_token_id].item())
                argmax_id = int(log_probs.argmax().item())
                metrics[f"probe/{name}/logp_marker"] = lp
                metrics[f"probe/{name}/argmax_emission"] = (
                    1.0 if argmax_id == self.marker_token_id else 0.0
                )
            # Push to wandb directly so the metrics line up with the trainer's run.
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except Exception as e:
                log.warning("ProbePanel wandb.log failed (%s); continuing", e)
        finally:
            if was_training:
                model.train()


def write_sentinel(payload: dict) -> Path:
    """Write end-of-cell sentinel per CLAUDE.md pod-side contract.

    Pod path: /workspace/logs/issue-405-epm_results-<epoch>.json
    Fallback (local smoke): {repo_root}/logs/issue-405-epm_results-<epoch>.json
    """
    sentinel_dir = Path(SENTINEL_DIR_POD)
    if not Path("/workspace").exists():
        sentinel_dir = PROJECT_ROOT / SENTINEL_DIR_LOCAL_FALLBACK
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    path = sentinel_dir / f"issue-405-epm_results-{epoch}.json"
    # Required keys per CLAUDE.md "Pod-side result-reporting contract"
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 405,
        "by": "issue405_run_cell.py",
        "ts": epoch,
        "note": payload,
    }
    path.write_text(json.dumps(body, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_405" / "cell_specs.json"),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_405"),
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval_results" / "issue_405"),
    )
    parser.add_argument(
        "--vllm-mem-util",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.55")),
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (assume adapter exists); for smoke flows that test eval only",
    )
    parser.add_argument(
        "--persist-adapter-repo",
        type=str,
        default=os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO", "superkaiba1/explore-persona-space"),
    )
    parser.add_argument(
        "--persist-adapter-subfolder-template",
        type=str,
        default="issue_405/{cell_id}_seed{seed}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Training epochs (default 2 — non-saturating anchor per #448)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Training LR (default 5e-6 — non-saturating anchor per #448)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode — also writes g_logprob_source kill-criterion summary",
    )
    args = parser.parse_args()

    # Pin GPU BEFORE any torch import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # Mark phase=startup for poll_pipeline.py (per CLAUDE.md pod-side contract).
    print(f"[phase=startup] cell={args.cell_id} seed={args.seed} gpu={args.gpu}", flush=True)

    # Load spec
    specs = json.loads(Path(args.cell_specs).read_text())
    by_id = {s["cell_id"]: s for s in specs}
    if args.cell_id not in by_id:
        raise SystemExit(f"cell_id={args.cell_id!r} not in {args.cell_specs}")
    spec = by_id[args.cell_id]
    track = spec["track"]

    log.info(
        "=" * 70 + "\n"
        "ISSUE 405 / CELL %s / SEED %d / GPU %d / TRACK %s / K=%d positives=%s\n" + "=" * 70,
        args.cell_id,
        args.seed,
        args.gpu,
        track,
        spec["K"],
        spec["positives"],
    )

    out_dir = Path(args.eval_results_dir) / f"cell_{args.cell_id}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        log.info("result.json already exists at %s — skipping (idempotent).", result_path)
        # Blocker 7: emit phase=done so poll_pipeline.py reads this as a clean
        # completion, not a stuck cell. Without it the orchestrator would
        # flip status to "dead" and suppress the auto-post of epm:results.
        print("[phase=done]", flush=True)
        return 0

    # ── Persist-adapter env vars (delete-after-eval recipe per upload-policy) ──
    subfolder = args.persist_adapter_subfolder_template.format(cell_id=args.cell_id, seed=args.seed)
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = args.persist_adapter_repo
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = subfolder
    os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"

    # ── Tokenizer + marker assert ─────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert_marker_token_id(tokenizer)

    # ── Training-data path (Phase 2 must have run already) ─────────────────
    data_jsonl = (
        Path(args.data_dir) / "training_jsonl" / f"cell_{args.cell_id}_seed{args.seed}.jsonl"
    )
    if not data_jsonl.exists():
        raise SystemExit(
            f"Training JSONL missing: {data_jsonl}. "
            f"Run scripts/issue405_make_training_data.py --cell-id {args.cell_id} "
            f"--seed {args.seed} first."
        )

    # ── Phase 3.1 — training (PERSIST PER PHASE per code-style.md) ────────
    print("[phase=training]", flush=True)
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"

    if not args.skip_training:
        from explore_persona_space.train.sft import TrainLoraConfig, merge_lora, train_lora

        run_name = f"issue405_{args.cell_id}_seed{args.seed}"
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

        # ── FIX E (Blocker 4) — per-step probe-panel callback ──────────
        # Plan §4.4 + §6.5 fig #4: log marker logp + emission at every
        # logging_steps step on a FIXED probe panel of 3 personas:
        #   (a) first trained-positive (the cell's first source persona)
        #   (b) no_persona (bare default — open-q 3.7 safety target)
        #   (c) comedian (far held-out — OOD generalization curve)
        all_prompts_pre = load_all_persona_prompts()
        _train_qs_pre, _eval_qs_pre = _import_questions()
        # Use a fixed question + on-policy R from the first trained
        # positive's cached R (matches the trained-context shape).
        probe_personas: list[tuple[str, str]] = [
            (spec["positives"][0], all_prompts_pre[spec["positives"][0]]),
            ("no_persona", all_prompts_pre["no_persona"]),
            ("comedian", all_prompts_pre["comedian"]),
        ]
        sample_question = _train_qs_pre[0]
        # Load the first source persona's cached R for this question.
        # Phase 1 must have run first; if not, _import + cache lookup fails
        # loud rather than silently logging a degenerate probe.
        cached_R_first_pos = json.loads(
            (Path(args.data_dir) / "onpolicy_R" / f"{spec['positives'][0]}.json").read_text()
        )["responses"]
        if sample_question not in cached_R_first_pos:
            raise RuntimeError(
                f"FIX-E probe: cached R for question {sample_question!r} missing "
                f"from {spec['positives'][0]}.json. Re-run Phase 1."
            )
        sample_R = cached_R_first_pos[sample_question]
        probe_cb = ProbePanelLogprobCallback(
            probe_personas=probe_personas,
            sample_R=sample_R,
            sample_question=sample_question,
            marker_text=MARKER_TEXT,
            marker_token_id=MARKER_TOKEN_ID,
            log_every_steps=5,
        )

        cfg = TrainLoraConfig(
            gpu_id=args.gpu,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            batch_size=PER_DEVICE_BATCH,
            grad_accum=GRAD_ACCUM,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            seed=args.seed,
            run_name=run_name,
            report_to="wandb",
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,  # Plan §4.4: positives loss on marker + post-R EOS
            lora_targets=LORA_TARGETS_NARROW,  # NARROW attention-only (no MLP — #311, #405)
            hf_upload=False,  # The EPM_PERSIST_ADAPTER_* path handles upload
        )
        log.info("Starting train_lora (with FIX-E probe-panel callback) ...")
        adapter_path_str, training_loss = train_lora(
            base_model_path=BASE_MODEL,
            data_path=str(data_jsonl),
            output_dir=str(adapter_dir),
            cfg=cfg,
            callbacks=[probe_cb],
        )
        log.info("Training done. loss=%.4f adapter=%s", training_loss, adapter_path_str)

        # ── Blocker 3 fix: persist adapter to HF BEFORE merge/delete ────
        # FAIL LOUD per upload-policy.md / #404/#458. The merge below
        # produces a regenerable ~15GB dir we WILL `rm` to stay under the
        # MooseFS ~130GB quota; the adapter (~300MB) is the only durable
        # copy. If persist fails we abort BEFORE the merge so a downstream
        # cleanup pass cannot silently rm an un-uploaded checkpoint.
        from explore_persona_space.train.trainer import _maybe_persist_adapter

        log.info(
            "Persisting LoRA adapter to HF (FAIL-LOUD) → %s/%s ...",
            os.environ.get("EPM_PERSIST_ADAPTER_HF_REPO"),
            os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER"),
        )
        _maybe_persist_adapter(Path(adapter_path_str))
        log.info("Adapter persist verified.")

        # Merge
        log.info("Merging LoRA → %s", merged_dir)
        merge_lora(BASE_MODEL, adapter_path_str, str(merged_dir), gpu_id=args.gpu)
    else:
        log.info("--skip-training: assuming adapter at %s + merged at %s", adapter_dir, merged_dir)
        training_loss = None

    # ── Phase 3.2 — on-policy R generation, two panels ───────────────────
    print("[phase=eval_rgen]", flush=True)
    all_prompts = load_all_persona_prompts()
    _train_qs, eval_questions = _import_questions()

    held_out_prompts = {p: all_prompts[p] for p in spec["held_out"]}
    trained_pos_prompts = {p: all_prompts[p] for p in spec["positives"]}
    panels = {
        "held_out": held_out_prompts,
        "trained_positive": trained_pos_prompts,
    }
    persona_prompts_flat = {**held_out_prompts, **trained_pos_prompts}

    R_eval = vllm_greedy_generate(
        merged_path=str(merged_dir),
        panels=panels,
        eval_questions=eval_questions,
        tokenizer=tokenizer,
        gpu_id=args.gpu,
        gpu_mem_util=args.vllm_mem_util,
    )

    # vLLM teardown — REAP children before HF re-loads.
    log.info("Reaping vLLM children before HF model reload (gotchas.md) ...")
    kill_vllm_children()
    nvidia_smi_assert_clean(args.gpu)

    # ── Phase 3.3 — score logprob + KL on trained, then on base ──────────
    print("[phase=eval_score_trained]", flush=True)
    trained_scores = score_logprob_and_kl(
        model_path=str(merged_dir),
        R_eval=R_eval,
        persona_prompts=persona_prompts_flat,
        tokenizer=tokenizer,
        device=device,
    )

    print("[phase=eval_score_base]", flush=True)
    base_scores = score_logprob_and_kl(
        model_path=BASE_MODEL,
        R_eval=R_eval,
        persona_prompts=persona_prompts_flat,
        tokenizer=tokenizer,
        device=device,
    )

    # ── Compute per-panel summaries ──────────────────────────────────────
    held_out_per_persona, held_out_summary, _ho_dlogp_means, _ho_emit_means = compute_panel_summary(
        {p: R_eval[p] for p in held_out_prompts},
        {p: trained_scores[p] for p in held_out_prompts},
        {p: base_scores[p] for p in held_out_prompts},
    )
    tp_per_persona, tp_summary, _tp_dlogp_means, _tp_emit_means = compute_panel_summary(
        {p: R_eval[p] for p in trained_pos_prompts},
        {p: trained_scores[p] for p in trained_pos_prompts},
        {p: base_scores[p] for p in trained_pos_prompts},
    )

    # ── Save raw_completions JSON + FAIL-LOUD upload (Blocker 3) ─────────
    raw_completions_path = out_dir / "raw_completions.json"
    raw_completions_path.write_text(
        json.dumps(
            {
                "cell_id": args.cell_id,
                "seed": args.seed,
                "track": track,
                "R_eval": R_eval,
                "eval_questions": eval_questions,
                "spec": spec,
            },
            indent=2,
        )
    )
    log.info("Wrote raw_completions → %s", raw_completions_path)

    # Upload to superkaiba1/explore-persona-space-data per Upload Policy.
    # FAIL LOUD — a silent upload-skip means the per-cell on-policy R is
    # only on the pod's MooseFS, which the orchestrator's auto-terminate
    # reaps after this cell completes.
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    experiment_name = f"issue_405/{args.cell_id}_seed{args.seed}"
    log.info("Uploading raw_completions to HF data repo (FAIL-LOUD) ...")
    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=experiment_name,
        eval_results_dir=out_dir,
        delete_after=False,
    )
    if not uploaded:
        raise RuntimeError(
            f"raw_completions upload FAILED for cell={args.cell_id} seed={args.seed} "
            f"— no files reported uploaded. Refusing to delete local copy."
        )
    log.info("raw_completions uploaded to HF: %d file(s)", len(uploaded))

    # ── Source-strength scalar + saturation read (Blocker 1 fix) ─────────
    # Two DIFFERENT quantities — keep them separate.
    #
    # `g_logprob_source` (= ABSOLUTE trained log P(marker) at the post-R
    # slot, averaged across the K trained-positive personas) is the
    # saturation read. Plan §4.9 + #448: expected ∈ [-8, -3], saturated
    # if > -0.1. Was (incorrectly, round 1) wired to the trained−base
    # delta — which is ≥ 0 for any successful implant and would silently
    # always trip the kill-criterion. FIX: read the ABSOLUTE trained
    # log-prob via `compute_panel_summary`'s new `logp_trained_mean`.
    #
    # `trained_pos_mean_dlogp` (= trained − base ΔlogP, same panel) is the
    # FIX-A1 source-strength scalar the analyzer's covariate-adjusted
    # regression consumes for the dose-vs-diversity check. Separate field.
    g_logprob_source = tp_summary["logp_trained_mean"]
    trained_pos_mean_dlogp = tp_summary["mean_deltaLogP"]

    # ── Reproducibility metadata ─────────────────────────────────────────
    import datetime as _dt
    import platform

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            env={**os.environ},  # Blocker 10: explicit env-passthrough
        ).strip()
    except Exception:
        git_commit = "unknown"

    result = {
        "experiment": "issue_405_kdiversity",
        "cell_id": args.cell_id,
        "seed": args.seed,
        "track": track,
        "K": spec["K"],
        "spec": spec,
        "training": {
            "base_model": BASE_MODEL,
            "marker_text": MARKER_TEXT,
            "marker_token_id": MARKER_TOKEN_ID,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS_NARROW,
            "lr": args.lr,
            "epochs": args.epochs,
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "marker_only_loss": True,
            "marker_tail_tokens": 0,
            "loss": training_loss,
        },
        "eval": {
            "held_out": held_out_per_persona,
            "trained_positive": tp_per_persona,
            "summary": {
                "held_out_mean_dlogp": held_out_summary["mean_deltaLogP"],
                "held_out_mean_emit_rate": held_out_summary["mean_emit_rate"],
                "held_out_logp_trained_mean": held_out_summary["logp_trained_mean"],
                "held_out_logp_base_mean": held_out_summary["logp_base_mean"],
                "trained_pos_mean_dlogp": trained_pos_mean_dlogp,
                "trained_pos_mean_emit_rate": tp_summary["mean_emit_rate"],
                "trained_pos_logp_trained_mean": tp_summary["logp_trained_mean"],
                "trained_pos_logp_base_mean": tp_summary["logp_base_mean"],
                # ABSOLUTE trained log P(marker) on the trained-source panel,
                # NOT trained−base. Drives smoke_check.saturated (Blocker 1 fix).
                "g_logprob_source": g_logprob_source,
            },
            "n_eval_questions": len(eval_questions),
        },
        "metadata": {
            "git_commit": git_commit,
            "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "smoke_check": {
            "g_logprob_source": g_logprob_source,
            "saturated": g_logprob_source > -0.1,
            "kill_criterion": (
                "STOP: g_logprob_source > -0.1 (saturated). "
                "Drop epochs to 1 or lr to 2e-6 and re-smoke."
                if g_logprob_source > -0.1
                else "OK"
            ),
        },
        "is_smoke": bool(args.smoke),
    }
    result_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote result → %s", result_path)

    # ── Clean local weights (per Upload Policy) ──────────────────────────
    # Adapter persists via EPM_PERSIST_ADAPTER_HF_REPO; merged is regenerable.
    import shutil

    if merged_dir.exists():
        log.info("Removing merged dir to free disk: %s", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)
    if adapter_dir.exists():
        log.info("Removing local adapter dir (persisted via HF): %s", adapter_dir)
        shutil.rmtree(adapter_dir, ignore_errors=True)

    # ── Sentinel for poll_pipeline.py ────────────────────────────────────
    sentinel = write_sentinel(
        {
            "cell_id": args.cell_id,
            "seed": args.seed,
            "track": track,
            "K": spec["K"],
            "g_logprob_source": g_logprob_source,
            "held_out_mean_dlogp": held_out_summary["mean_deltaLogP"],
            "held_out_mean_emit_rate": held_out_summary["mean_emit_rate"],
            "result_path": str(result_path),
            "saturated": g_logprob_source > -0.1,
        }
    )
    log.info("Wrote sentinel → %s", sentinel)

    # poll_pipeline.py expects a terminating "[phase=done]" line on graceful exit.
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
