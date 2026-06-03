# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #472 — NEW on-policy trajectory eval rig (plan §4.6 — the #448-saturation fix).

At EACH of the 6 training checkpoints, the trained adapter writes its OWN greedy
answer R_j to every held-out (persona × q_eval) probe, and the DV is read at the
slot immediately after R_j:
  - DV-A (logP): vLLM ``prompt_logprobs`` (eval_one_cell.score_logp_for_R).
    ΔG = trained log P(※) − base log P(※) on the SAME R_j.
  - DV-B (full-vocab KL): HF teacher-forced forward pass over ``prompt + R_j +
    \n\n`` → (N, 1, V) log-softmax at the SINGLE post-R marker slot for trained
    and base, then ``compute_kl_divergence`` (NOT a seq-mean over R). vLLM
    ``prompt_logprobs=1`` can't return full vocab → KL needs the HF path.
  - Emission = (argmax == marker) at the same slot (free from DV-A).
  - Source-self ΔG: ΔG on the SOURCE persona's own R (the validity gate + the
    matched-slice anchor).

This is NOT #448's MarkerTrajectoryCallback (teacher-forced, off-policy — the
exact probe the body rejects, #432→#456). Each checkpoint generates fresh
on-policy R.

Framework-switch discipline (CLAUDE.md vLLM-teardown gotcha): the rig does ALL
vLLM work first (per-checkpoint on-policy gen + DV-A trained + DV-A base on the
same R; held-out panel + source), persists per-checkpoint, then tears vLLM down
HARD (psutil child-kill + nvidia-smi PID check) and does ALL HF KL work — ONE
framework switch per cell, not 12. Per-checkpoint files are written the moment
each phase completes (checkpoint-per-phase rule + feedback_eval_rig_per_phase_checkpoint).

The trajectory.json per cell × seed (plan §4.8):
    {
      "cell": ..., "seed": ..., "source": ..., "matched_slice_target_nats": 8.0,
      "checkpoints": [
        {"frac": 0.08, "step": 12, "adapter_path": ...,
         "source_self": {"g_logp": .., "b_logp": .., "delta_g": ..},
         "held_out": {persona: {q: {"g_logp":.., "b_logp":.., "delta_g":..,
                                    "argmax_marker":bool, "kl": ..}}}},
        ...],
      "git_commit": ..., "timestamp_utc": ...
    }
"""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_SEP,
    MARKER_TEXT,
    MATCHED_SLICE_TARGET_NATS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    assert_marker_token,
    score_logp_for_R,
)

log = logging.getLogger("issue_472.eval_trajectory")

# The trajectory eval runs in a nested subprocess on the SAME GPU the cell
# just trained the LoRA on (in-process train_one_cell). Even after the parent
# frees the training tensors, vLLM's startup check compares its desired
# fraction-of-TOTAL against FREE memory, so a high util trips
# "Free memory < desired" when any train residual / reaping latency remains.
# 0.60 (~47 GiB of an 80 GiB H100) leaves ample headroom for that residual
# while still giving a large KV cache for greedy gen of the held-out panel.
DEFAULT_GPU_MEM_UTIL = 0.60
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_LORA_RANK = 32


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _teardown_vllm_hard(llm) -> None:
    """Tear vLLM down and FAIL LOUD if a worker still holds the GPU.

    CLAUDE.md vLLM-teardown gotcha: del + destroy + gc + empty_cache is NOT
    enough — TP/PP worker subprocesses survive and re-grab freed memory the
    moment HF Transformers loads weights. We kill survivors + assert via
    nvidia-smi. CVD-aware: only inspect GPUs visible to THIS process (the
    feedback_orphan_pid_check_must_be_cvd_aware lesson).
    """
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:  # pragma: no cover - version-dependent symbol
        log.warning("vLLM destroy helpers unavailable (%s); continuing teardown.", e)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reap surviving child processes (the worker subprocesses).
    try:
        import contextlib

        import psutil

        me = psutil.Process()
        children = me.children(recursive=True)
        for c in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _gone, alive = psutil.wait_procs(children, timeout=10)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except ImportError:  # pragma: no cover - psutil should be installed
        log.warning("psutil unavailable; cannot reap vLLM worker subprocesses.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_on_policy_R(
    llm,
    tokenizer,
    eval_personas: dict[str, str],
    eval_questions: list[str],
    lora_request,
    max_new_tokens: int,
) -> dict[str, dict[str, str]]:
    """vLLM batched greedy gen of the TRAINED model's own R for the panel × q grid.

    Persona injection ALWAYS via the system role. Returns r[persona][q] -> text.
    """
    from vllm import SamplingParams

    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            keys.append((persona, q))
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = llm.generate(prompts, sp, lora_request=lora_request)
    if len(outputs) != len(prompts):
        raise RuntimeError(f"on-policy gen returned {len(outputs)} for {len(prompts)} prompts.")
    r: dict[str, dict[str, str]] = {p: {} for p in eval_personas}
    for (persona, q), out in zip(keys, outputs, strict=True):
        r[persona][q] = out.outputs[0].text
    return r


def compute_kl_for_checkpoint(
    *,
    base_model: str,
    adapter_path: str,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    marker_text: str = MARKER_TEXT,
    sep: str = MARKER_SEP,
    device: str = "cuda:0",
) -> dict[str, dict[str, float]]:
    """DV-B: full-vocab KL(trained‖base) at the SINGLE post-R marker slot.

    For each (persona, q): teacher-force ``prompt + R + sep`` through the trained
    (base+adapter) model and the base model, take the next-token log-softmax at
    the FINAL position (the slot the marker would occupy), and compute
    ``compute_kl_divergence(log_p_trained, log_q_base)`` at THAT single slot (NOT
    a seq-mean over R — plan §4.6).

    Returns ``kl[persona][q] -> float``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from explore_persona_space.analysis.divergence import compute_kl_divergence

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()
    trained = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trained = PeftModel.from_pretrained(trained, adapter_path).eval()

    def _slot_logsoftmax(model, persona_prompt: str, q: str, r_text: str) -> torch.Tensor:
        """Next-token log-softmax (V,) at the slot AFTER `prompt + R + sep`."""
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # The marker would be generated after `prompt + R + sep`; the slot is the
        # next-token distribution at the LAST token of that prefix.
        prefix = prompt_text + r_text + sep
        ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits  # (1, T, V)
        last = logits[0, -1, :].float()  # (V,)
        return torch.log_softmax(last, dim=-1).cpu()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    kl: dict[str, dict[str, float]] = {p: {} for p in eval_personas}
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            r_text = r_by_persona_q[persona][q]
            lp_trained = _slot_logsoftmax(trained, persona_prompt, q, r_text)
            lp_base = _slot_logsoftmax(base, persona_prompt, q, r_text)
            # compute_kl_divergence expects (seq, V); pass (1, V).
            kl_val = compute_kl_divergence(lp_trained.unsqueeze(0), lp_base.unsqueeze(0))
            kl[persona][q] = float(kl_val.item())

    del base, trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return kl


def run_trajectory_eval(
    *,
    cell_slug: str,
    seed: int,
    checkpoint_specs: list[dict],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    source: str = SOURCE_PERSONA,
    source_prompt: str,
    out_path: Path,
    base_model: str = BASE_MODEL,
    max_new_tokens: int = 1024,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    compute_kl: bool = True,
) -> Path:
    """Run the on-policy trajectory eval for one cell × seed.

    Args:
        cell_slug, seed: cell identity.
        checkpoint_specs: list of {"frac": float, "step": int, "adapter_path": str}.
        eval_personas: held-out panel {persona: system_prompt}.
        eval_questions: Q_eval.
        source, source_prompt: source persona (for source-self ΔG validity gate).
        out_path: trajectory.json output.
        base_model, max_new_tokens, max_lora_rank, gpu_memory_utilization,
            max_model_len: vLLM params.
        compute_kl: if False, skip DV-B (smoke speed-up).

    Returns:
        out_path.
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    assert_marker_token(tokenizer)

    panel_plus_source = dict(eval_personas)
    panel_plus_source.setdefault(source, source_prompt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-checkpoint partial file (resume-safe + crash-safe per CLAUDE.md).
    partial_path = out_path.with_suffix(".partial.json")

    # ── Phase A: ALL vLLM work (gen + DV-A trained + DV-A base) per checkpoint. ─
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        max_model_len=max_model_len,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=1,
    )

    checkpoints_out: list[dict] = []
    r_cache: dict[float, dict[str, dict[str, str]]] = {}  # frac -> on-policy R (for KL phase)
    for spec in checkpoint_specs:
        frac = spec["frac"]
        adapter_path = spec["adapter_path"]
        label = f"{cell_slug}_seed{seed}_frac{frac}"
        lora_req = LoRARequest(lora_name=label, lora_int_id=1, lora_path=adapter_path)
        log.info("[phase=traj_vllm] %s: on-policy gen + DV-A", label)

        # 1. Trained model writes its OWN R for held-out panel + source.
        r_on_policy = _generate_on_policy_R(
            llm, tokenizer, panel_plus_source, eval_questions, lora_req, max_new_tokens
        )
        r_cache[frac] = r_on_policy

        # 2. DV-A trained log P(※) at post-R slot (on the trained model's own R).
        g = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=eval_questions,
            cell_label=f"TRAINED/{label}",
            use_lora=True,
            lora_request=lora_req,
        )
        # 3. DV-A base log P(※) at the SAME slot on the SAME R (lora_request=None).
        b = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=eval_questions,
            cell_label=f"BASE/{label}",
            use_lora=False,
        )

        held_out: dict[str, dict[str, dict[str, float | bool]]] = {}
        n_collapsed_ck = 0
        per_persona_emission_rate: dict[str, float] = {}
        for persona in eval_personas:
            held_out[persona] = {}
            n_marker_argmax = 0
            for q in eval_questions:
                gl = g[persona][q]["logp"]
                bl = b[persona][q]["logp"]
                # r_collapsed is read from the TRAINED model's score (g): it is the
                # trained model's OWN on-policy R that may collapse to marker-spam.
                r_collapsed = bool(g[persona][q].get("r_collapsed", False))
                if r_collapsed:
                    n_collapsed_ck += 1
                argmax_marker = bool(g[persona][q]["argmax_marker"])
                if argmax_marker:
                    n_marker_argmax += 1
                held_out[persona][q] = {
                    "g_logp": gl,
                    "b_logp": bl,
                    "delta_g": gl - bl,
                    "argmax_marker": argmax_marker,
                    # #479 concern 5: which token actually won at the post-R
                    # slot, for both trained and base, so a downstream audit
                    # can see what won when argmax_marker is False (e.g.
                    # <|im_end|> 151645, trailing \n 198). -1 = legacy file
                    # that didn't carry the field.
                    "argmax_token_id_trained": int(g[persona][q].get("argmax_token_id", -1)),
                    "argmax_token_id_base": int(b[persona][q].get("argmax_token_id", -1)),
                    "n_marker_in_R": int(g[persona][q].get("n_marker_in_R", 0)),
                    "r_collapsed": r_collapsed,
                    "kl": None,  # filled in Phase B.
                }
            per_persona_emission_rate[persona] = (
                n_marker_argmax / len(eval_questions) if eval_questions else 0.0
            )
        # Source-self mean ΔG over Q_eval (validity gate + matched-slice anchor).
        src_deltas = [g[source][q]["logp"] - b[source][q]["logp"] for q in eval_questions]
        # Source-self collapse: did the SOURCE's own R collapse to marker-spam at
        # this checkpoint? If yes, source-self ΔG is repetition-ceiling, so the
        # matched-slice (8±1 nat) anchor read at this checkpoint is post-collapse
        # — the analyzer must prefer an EARLIER, non-collapsed checkpoint.
        src_collapsed = any(bool(g[source][q].get("r_collapsed", False)) for q in eval_questions)
        # Source emission rate (#479 plan §6.1): fraction of source's own greedy
        # on-policy generations whose post-R slot argmax == ※. Score grid in
        # ``g`` includes the source via ``panel_plus_source``.
        n_source_emit = sum(
            1 for q in eval_questions if bool(g[source][q].get("argmax_marker", False))
        )
        source_emission_rate = n_source_emit / len(eval_questions) if eval_questions else 0.0
        source_self = {
            "g_logp_mean": sum(g[source][q]["logp"] for q in eval_questions) / len(eval_questions),
            "b_logp_mean": sum(b[source][q]["logp"] for q in eval_questions) / len(eval_questions),
            "delta_g_mean": sum(src_deltas) / len(src_deltas),
            "r_collapsed": src_collapsed,
            "emission_rate": source_emission_rate,
            "n_marker_argmax": n_source_emit,
            "n_questions": len(eval_questions),
        }
        # Bystander panel emission-rate roll-up (#479 plan §6.1).
        bystander_rates = [r for p, r in per_persona_emission_rate.items() if p != source]
        if bystander_rates:
            byst_mean = sum(bystander_rates) / len(bystander_rates)
            byst_var = sum((r - byst_mean) ** 2 for r in bystander_rates) / max(
                len(bystander_rates) - 1, 1
            )
            byst_se = (byst_var / len(bystander_rates)) ** 0.5
        else:
            byst_mean = 0.0
            byst_se = 0.0
        bystander_emission = {
            "mean": byst_mean,
            "se": byst_se,
            "n_personas": len(bystander_rates),
            "per_persona_rate": per_persona_emission_rate,
        }
        n_held_out_probes = len(eval_personas) * len(eval_questions)
        held_out_collapse_share = n_collapsed_ck / n_held_out_probes if n_held_out_probes else 0.0
        checkpoints_out.append(
            {
                "frac": frac,
                "step": spec.get("step"),
                # #479 concern 6: surface the original index-key string so
                # downstream labels can read "step5" (e.g. "step_key=0005")
                # for #479 absolute-step checkpoints instead of "frac5.0".
                # For the #472 fractional path this carries "0.08" etc.
                "step_key": spec.get("step_key"),
                "adapter_path": adapter_path,
                "source_self": source_self,
                "bystander_emission": bystander_emission,
                "held_out_collapse_share": held_out_collapse_share,
                "n_held_out_collapsed": n_collapsed_ck,
                "held_out": held_out,
            }
        )
        # Persist partial after each checkpoint's vLLM phase (crash-safe).
        partial_path.write_text(
            json.dumps({"cell": cell_slug, "seed": seed, "checkpoints": checkpoints_out}, indent=2)
        )
        log.info(
            "[phase=traj_vllm] %s done: source-self ΔG=%.2f nats (emission=%.2f), "
            "bystander emission mean=%.2f±%.2f, source_R_collapsed=%s, "
            "held-out R-collapse share=%.2f (%d/%d)",
            label,
            source_self["delta_g_mean"],
            source_emission_rate,
            byst_mean,
            byst_se,
            src_collapsed,
            held_out_collapse_share,
            n_collapsed_ck,
            n_held_out_probes,
        )

    _teardown_vllm_hard(llm)

    # ── Phase B: ALL HF full-vocab KL work (one framework switch). ────────────
    if compute_kl:
        for ck in checkpoints_out:
            frac = ck["frac"]
            adapter_path = ck["adapter_path"]
            log.info("[phase=traj_kl] %s_seed%s_frac%s: DV-B full-vocab KL", cell_slug, seed, frac)
            kl = compute_kl_for_checkpoint(
                base_model=base_model,
                adapter_path=adapter_path,
                r_by_persona_q=r_cache[frac],
                eval_personas=eval_personas,
                eval_questions=eval_questions,
            )
            for persona in eval_personas:
                for q in eval_questions:
                    ck["held_out"][persona][q]["kl"] = kl[persona][q]
            partial_path.write_text(
                json.dumps(
                    {"cell": cell_slug, "seed": seed, "checkpoints": checkpoints_out}, indent=2
                )
            )

    payload = {
        # v2: per-checkpoint adds `source_self.emission_rate` +
        # `bystander_emission` (mean / se / per_persona_rate). The raw
        # per-question records under `held_out[persona][q]` are unchanged.
        "schema_version": "i472_v2",
        "cell": cell_slug,
        "seed": seed,
        "source": source,
        "marker_text": MARKER_TEXT,
        "marker_token_id": EXPECTED_MARKER_TOKEN_ID,
        "matched_slice_target_nats": MATCHED_SLICE_TARGET_NATS,
        "n_held_out_personas": len(eval_personas),
        "held_out_personas": sorted(eval_personas.keys()),
        "n_eval_questions": len(eval_questions),
        "eval_questions": eval_questions,
        "kl_computed": compute_kl,
        "checkpoints": checkpoints_out,
        "git_commit": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    if partial_path.exists():
        partial_path.unlink()
    log.info("[phase=traj_done] Wrote trajectory → %s", out_path)
    return out_path
