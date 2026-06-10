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

# The post-R slot's EOS competitor under the Qwen-2.5 chat template: the
# contrastive negatives train `<|im_end|>` (id 151645) at the slot (see
# MarkerOnlyDataCollator(marker_im_end_token_id=151645) in train/sft.py), so
# the marker-vs-EOS logit margin is the mechanistic contrast of interest.
POST_R_EOS_TOKEN = "<|im_end|>"
EXPECTED_POST_R_EOS_ID = 151645


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


def _slot_stats_from_raw_logits(
    last,  # torch.Tensor (V,) raw next-token logits at the slot
    marker_id: int,
    eos_id: int,
) -> dict[str, float]:
    """Marker-slot readouts in raw-logit space from one (V,) logits vector.

    Returns ``{"z_marker", "z_eos", "logZ"}`` so the caller can form
    ``logp = z_marker − logZ`` (the exact identity from
    ``.claude/rules/marker-leakage-measurement.md`` § "Report BOTH log-prob
    and logit") without a second forward pass. Pure + CPU-testable.
    """
    import torch

    assert last.ndim == 1, last.shape
    log_z = torch.logsumexp(last, dim=-1)
    return {
        "z_marker": float(last[marker_id].item()),
        "z_eos": float(last[eos_id].item()),
        "logZ": float(log_z.item()),
    }


def assert_logit_readout_gauge_free(adapter_path: str) -> None:
    """Read ``<adapter_path>/adapter_config.json`` and fail loud if LoRA touches W_U.

    The trained − base marker LOGIT readout (``Δz_marker = W_U[marker]·Δh``)
    is valid only when the adapter leaves the unembedding / embeddings
    untouched. Delegates the schema check to
    :func:`explore_persona_space.eval.marker_logprob.assert_gauge_free_adapter_config`.
    """
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    cfg_path = Path(adapter_path) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json missing at {cfg_path} — cannot verify the logit "
            "readout is gauge-free; refusing to score raw marker logits."
        )
    assert_gauge_free_adapter_config(json.loads(cfg_path.read_text()), context=str(cfg_path))


def compute_kl_for_checkpoint(
    *,
    base_model: str,
    adapter_path: str,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    marker_text: str = MARKER_TEXT,
    marker_token_id: int = EXPECTED_MARKER_TOKEN_ID,
    sep: str = MARKER_SEP,
    device: str = "cuda:0",
) -> dict[str, dict[str, dict[str, float]]]:
    """DV-B: full-vocab KL(trained‖base) + raw marker-slot logits at the post-R slot.

    For each (persona, q): teacher-force ``prompt + R + sep`` through the trained
    (base+adapter) model and the base model, take the next-token logits at
    the FINAL position (the slot the marker would occupy), and compute
    ``compute_kl_divergence(log_p_trained, log_q_base)`` at THAT single slot (NOT
    a seq-mean over R — plan §4.6).

    From the SAME forward pass this also captures, per (persona, q), the raw
    pre-softmax readouts for BOTH models: ``z_marker`` (logit at the marker
    id), ``z_eos`` (logit at ``<|im_end|>``), ``logZ`` (full-vocab
    logsumexp), plus ``logp_hf = z_marker − logZ`` (the HF-path log-prob,
    recorded for cross-engine consistency against the vLLM-scored
    ``g_logp``/``b_logp`` — record only, NO hard assert: bf16 kernels /
    batching make exact agreement impossible). The marker-LOGIT trained −
    base is the non-saturating mechanistic readout mandated by
    ``.claude/rules/marker-leakage-measurement.md`` § "Report BOTH log-prob
    and logit"; :func:`assert_logit_readout_gauge_free` MUST pass for the
    adapter before these numbers are comparable across cells.

    Returns ``stats[persona][q] -> {"kl", "z_marker_g", "z_marker_b",
    "z_eos_g", "z_eos_b", "logZ_g", "logZ_b", "logp_hf_g", "logp_hf_b"}``
    (``_g`` = trained, ``_b`` = base).

    Plan v5 §4.5 fix #3 — KL-from-base is NOT a fallback DV.
        KL measures total-distribution shift; the v3 recovery eval read
        24.35 nats of KL with 0 bystander emission and 0 bystander ΔG —
        KL was tracking EOS / punctuation reallocation, NOT marker mass.
        The legitimate uses of KL in v4 are: (a) sanity diagnostic in
        Phase 0.6's pass condition (KL > 0 ⇒ ΔG ≠ 0 on ≥ 1 slot), and
        (b) the per-batch byte-identical guard (KL > 0 AND |g − b| < 1e-6
        ⇒ ASSERT FAIL). NEVER substitute KL for the marker log-prob DV
        when the on-policy path "looks broken" — that's the v3 false-fix
        path. See `.claude/rules/marker-leakage-measurement.md` §Anti-
        patterns which names #504 explicitly. The #534 ``z_marker``
        companion is NOT that banned substitution: it is single-token,
        marker-specific, and reported alongside (never instead of) the
        log-prob DV.
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

    def _slot_raw_logits(model, persona_prompt: str, q: str, r_text: str) -> torch.Tensor:
        """Raw next-token logits (V,) at the slot AFTER `prompt + R + sep`."""
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
        return logits[0, -1, :].float().cpu()  # (V,)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
    assert len(marker_ids) == 1, (
        f"raw-logit capture requires a SINGLE-token marker; {marker_text!r} → {marker_ids}"
    )
    marker_id = marker_ids[0]
    eos_id = tokenizer.convert_tokens_to_ids(POST_R_EOS_TOKEN)
    assert eos_id == EXPECTED_POST_R_EOS_ID, (
        f"{POST_R_EOS_TOKEN!r} resolved to id {eos_id}, expected {EXPECTED_POST_R_EOS_ID} — "
        "tokenizer drift; the z_eos readout would be wrong."
    )

    stats: dict[str, dict[str, dict[str, float]]] = {p: {} for p in eval_personas}
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            r_text = r_by_persona_q[persona][q]
            raw_trained = _slot_raw_logits(trained, persona_prompt, q, r_text)
            raw_base = _slot_raw_logits(base, persona_prompt, q, r_text)
            # compute_kl_divergence expects (seq, V) log-softmax inputs; pass (1, V).
            lp_trained = torch.log_softmax(raw_trained, dim=-1)
            lp_base = torch.log_softmax(raw_base, dim=-1)
            kl_val = compute_kl_divergence(lp_trained.unsqueeze(0), lp_base.unsqueeze(0))
            g_stats = _slot_stats_from_raw_logits(raw_trained, marker_id, eos_id)
            b_stats = _slot_stats_from_raw_logits(raw_base, marker_id, eos_id)
            stats[persona][q] = {
                "kl": float(kl_val.item()),
                "z_marker_g": g_stats["z_marker"],
                "z_marker_b": b_stats["z_marker"],
                "z_eos_g": g_stats["z_eos"],
                "z_eos_b": b_stats["z_eos"],
                "logZ_g": g_stats["logZ"],
                "logZ_b": b_stats["logZ"],
                "logp_hf_g": g_stats["z_marker"] - g_stats["logZ"],
                "logp_hf_b": b_stats["z_marker"] - b_stats["logZ"],
            }

    del base, trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def compute_kl_and_slot_stats_for_checkpoint(
    *,
    base_model: str,
    adapter_path: str,
    r_by_persona_q: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    marker_text: str = MARKER_TEXT,
    marker_token_id: int = EXPECTED_MARKER_TOKEN_ID,
    sep: str = MARKER_SEP,
    device: str = "cuda:0",
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """#534-spelled compat wrapper over :func:`compute_kl_for_checkpoint`.

    The #534 branch named this function and spelled the slot-stat keys
    ``*_trained`` / ``*_base`` / ``logz`` / ``logp_marker_hf_*``; main's
    implementation (kept as the single source of truth in the #555 merge)
    spells them ``*_g`` / ``*_b`` / ``logZ`` / ``logp_hf_*``. This wrapper
    translates so #534-lineage consumers (``scripts/i534_smoke_local.py``
    Phase D) keep working unmodified.

    Returns:
        ``(kl, slot_stats)`` — ``kl[persona][q] -> float`` and
        ``slot_stats[persona][q]`` carrying the #534-spelled keys including
        ``delta_z_marker`` / ``delta_z_margin``.
    """
    combined = compute_kl_for_checkpoint(
        base_model=base_model,
        adapter_path=adapter_path,
        r_by_persona_q=r_by_persona_q,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        marker_text=marker_text,
        marker_token_id=marker_token_id,
        sep=sep,
        device=device,
    )
    kl: dict[str, dict[str, float]] = {}
    slot_stats: dict[str, dict[str, dict[str, float]]] = {}
    for persona, by_q in combined.items():
        kl[persona] = {}
        slot_stats[persona] = {}
        for q, st in by_q.items():
            kl[persona][q] = st["kl"]
            slot_stats[persona][q] = {
                "z_marker_trained": st["z_marker_g"],
                "z_marker_base": st["z_marker_b"],
                "z_eos_trained": st["z_eos_g"],
                "z_eos_base": st["z_eos_b"],
                "logz_trained": st["logZ_g"],
                "logz_base": st["logZ_b"],
                "logp_marker_hf_trained": st["logp_hf_g"],
                "logp_marker_hf_base": st["logp_hf_b"],
                "delta_z_marker": st["z_marker_g"] - st["z_marker_b"],
                "delta_z_margin": (st["z_marker_g"] - st["z_eos_g"])
                - (st["z_marker_b"] - st["z_eos_b"]),
                "eos_token_id": float(EXPECTED_POST_R_EOS_ID),
            }
    return kl, slot_stats


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
    source_guard_meta: dict | None = None,
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
        source_guard_meta: #534 round-2 adapter-applied cross-check. None
            (default) = exact legacy behavior. Otherwise a dict
            ``{"expected_by_frac": {frac(float): teacher-forced source ΔG
            (float) | None}, "band_stop_fired": bool, "tol_nats": float
            (optional)}`` — after each checkpoint's source-self ΔG is
            computed, ``assert_source_delta_g_matches_manifest`` fails loud
            on >tol disagreement at the final fraction (and on a <1-nat
            final read when the band-stop fired). The per-checkpoint diag is
            persisted as ``checkpoints[*].source_manifest_check``.

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
    # Final (max) fraction — the source-manifest guard's hard-fail gate.
    final_frac = max(s["frac"] for s in checkpoint_specs)
    for ck_i, spec in enumerate(checkpoint_specs, start=1):
        frac = spec["frac"]
        adapter_path = spec["adapter_path"]
        label = f"{cell_slug}_seed{seed}_frac{frac}"
        # #534 round-1 root cause: vLLM caches LoRA adapters STRICTLY by
        # ``lora_int_id`` (LRUCacheWorkerLoRAManager.add_adapter: an already-
        # seen id is "just touched" — ``lora_path`` is never re-read). Reusing
        # ``lora_int_id=1`` for every checkpoint silently served the FIRST
        # loaded adapter (step ~5, ΔG≈0.03) at all four fractions. Each
        # checkpoint MUST get a DISTINCT id; the LRU (max_loras=1) evicts the
        # previous adapter and loads the new path. #504/#530 never hit this
        # because their checkpoint_index carried a single usable entry.
        lora_req = LoRARequest(lora_name=label, lora_int_id=ck_i, lora_path=adapter_path)
        log.info(
            "[phase=traj_vllm] %s: on-policy gen + DV-A (lora_int_id=%d, path=%s)",
            label,
            ck_i,
            adapter_path,
        )

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

        # Fail-loud guard for the #477 v4/v6 silent-LoRA-not-applied
        # regression: if the adapter has B-matrix norm > floor (genuinely
        # trained) but max|ΔG| across the WHOLE panel is below eps AND on-
        # policy emission is uniformly zero, vLLM/PEFT silently dropped the
        # LoRA at the forward pass and ΔG ≈ 0 is a false floor. Raises
        # LoRANotAppliedError on the regression; passes silently (with logged
        # verdict) on a real signal, a genuine-floor adapter, or partial
        # emission. See .../eval_guard.py for the three-clause contract.
        from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
            assert_adapter_actually_applied,
        )

        assert_adapter_actually_applied(
            adapter_dir=adapter_path,
            g_records=g,
            b_records=b,
            cell_label=label,
        )

        held_out: dict[str, dict[str, dict[str, float | bool]]] = {}
        n_collapsed_ck = 0
        for persona in eval_personas:
            held_out[persona] = {}
            for q in eval_questions:
                gl = g[persona][q]["logp"]
                bl = b[persona][q]["logp"]
                # r_collapsed is read from the TRAINED model's score (g): it is the
                # trained model's OWN on-policy R that may collapse to marker-spam.
                r_collapsed = bool(g[persona][q].get("r_collapsed", False))
                if r_collapsed:
                    n_collapsed_ck += 1
                held_out[persona][q] = {
                    "g_logp": gl,
                    "b_logp": bl,
                    "delta_g": gl - bl,
                    "argmax_marker": g[persona][q]["argmax_marker"],
                    "n_marker_in_R": int(g[persona][q].get("n_marker_in_R", 0)),
                    "r_collapsed": r_collapsed,
                    "kl": None,  # filled in Phase B.
                }
        # Source-self mean ΔG over Q_eval (validity gate + matched-slice anchor).
        src_deltas = [g[source][q]["logp"] - b[source][q]["logp"] for q in eval_questions]
        # Source-self collapse: did the SOURCE's own R collapse to marker-spam at
        # this checkpoint? If yes, source-self ΔG is repetition-ceiling, so the
        # matched-slice (8±1 nat) anchor read at this checkpoint is post-collapse
        # — the analyzer must prefer an EARLIER, non-collapsed checkpoint.
        src_collapsed = any(bool(g[source][q].get("r_collapsed", False)) for q in eval_questions)
        # DV-C: source emission rate P(※) = share of Q_eval where the trained
        # model's argmax at the post-R slot is the marker (#477 validity gate;
        # plan §6 DV-C). Computed from the SAME g[source][q]["argmax_marker"]
        # already produced by score_logp_for_R; backward-compat for #472 (extra
        # field; no existing reader breaks).
        n_src_q = len(eval_questions)
        src_emission_p = (
            sum(1 for q in eval_questions if g[source][q].get("argmax_marker")) / n_src_q
            if n_src_q
            else 0.0
        )
        source_self = {
            "g_logp_mean": sum(g[source][q]["logp"] for q in eval_questions) / len(eval_questions),
            "b_logp_mean": sum(b[source][q]["logp"] for q in eval_questions) / len(eval_questions),
            "delta_g_mean": sum(src_deltas) / len(src_deltas),
            "emission_p": float(src_emission_p),
            "r_collapsed": src_collapsed,
        }
        # #534 round-2 adapter-applied cross-check: the selector's HF/PEFT
        # teacher-forced source ΔG and this on-policy read look at the SAME
        # snapshot dir through independent loaders — >tol disagreement at the
        # final fraction means the eval path did not actually apply the
        # adapter (fail loud BEFORE the flat trajectory leaves the rig).
        source_manifest_check: dict | None = None
        if source_guard_meta is not None:
            from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
                DEFAULT_SOURCE_MANIFEST_TOL_NATS,
                assert_source_delta_g_matches_manifest,
            )

            source_manifest_check = assert_source_delta_g_matches_manifest(
                cell_label=label,
                frac=frac,
                eval_delta_g_nats=source_self["delta_g_mean"],
                expected_delta_g_nats=source_guard_meta.get("expected_by_frac", {}).get(frac),
                is_final_frac=(frac == final_frac),
                band_stop_fired=bool(source_guard_meta.get("band_stop_fired", False)),
                tol_nats=float(source_guard_meta.get("tol_nats", DEFAULT_SOURCE_MANIFEST_TOL_NATS)),
            )
        n_held_out_probes = len(eval_personas) * len(eval_questions)
        held_out_collapse_share = n_collapsed_ck / n_held_out_probes if n_held_out_probes else 0.0
        checkpoints_out.append(
            {
                "frac": frac,
                "step": spec.get("step"),
                "adapter_path": adapter_path,
                "source_self": source_self,
                "source_manifest_check": source_manifest_check,
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
            "[phase=traj_vllm] %s done: source-self ΔG=%.2f nats, source_R_collapsed=%s, "
            "held-out R-collapse share=%.2f (%d/%d)",
            label,
            source_self["delta_g_mean"],
            src_collapsed,
            held_out_collapse_share,
            n_collapsed_ck,
            n_held_out_probes,
        )

    _teardown_vllm_hard(llm)

    # ── Phase B: ALL HF full-vocab KL work (one framework switch). ────────────
    # Plan v5 §4.4 fix #1: after KL is computed, fire the per-batch byte-
    # identical guard from the SAME (g, b, kl) the rig just produced. This
    # is the "wired into the same forward pass" requirement — the guard reads
    # the IDENTICAL records the trajectory.json writes, so a residual bug
    # cannot hide between guard and persist. The guard is fail-loud
    # (MarkerLogprobPathReadingFromBaseError) when rate > 5% de-minimis.
    if compute_kl:
        from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
            assert_byte_identical_rate_below_threshold,
        )

        for ck in checkpoints_out:
            frac = ck["frac"]
            adapter_path = ck["adapter_path"]
            # The raw-logit readout below is only gauge-free (comparable across
            # cells) when LoRA leaves the unembedding untouched — fail loud
            # BEFORE scoring if the adapter config violates that.
            assert_logit_readout_gauge_free(adapter_path)
            log.info(
                "[phase=traj_kl] %s_seed%s_frac%s: DV-B full-vocab KL + raw marker-slot logits",
                cell_slug,
                seed,
                frac,
            )
            # panel_plus_source (NOT eval_personas): the source persona gets the
            # same raw-logit readout so source_self carries mean logit fields.
            slot_stats = compute_kl_for_checkpoint(
                base_model=base_model,
                adapter_path=adapter_path,
                r_by_persona_q=r_cache[frac],
                eval_personas=panel_plus_source,
                eval_questions=eval_questions,
            )
            # Float-valued KL map (guard + leaf writes read it below).
            kl = {p: {q: slot_stats[p][q]["kl"] for q in eval_questions} for p in slot_stats}
            for persona in eval_personas:
                for q in eval_questions:
                    leaf = ck["held_out"][persona][q]
                    st = slot_stats[persona][q]
                    leaf["kl"] = st["kl"]
                    # Raw-logit readouts (additive; i472_v1 readers unaffected).
                    leaf["z_marker_g"] = st["z_marker_g"]
                    leaf["z_marker_b"] = st["z_marker_b"]
                    leaf["z_eos_g"] = st["z_eos_g"]
                    leaf["z_eos_b"] = st["z_eos_b"]
                    leaf["logZ_g"] = st["logZ_g"]
                    leaf["logZ_b"] = st["logZ_b"]
                    # HF-path log-probs (cross-engine consistency vs vLLM g/b_logp;
                    # record only — bf16/kernel/batching forbid a hard assert).
                    leaf["logp_hf_g"] = st["logp_hf_g"]
                    leaf["logp_hf_b"] = st["logp_hf_b"]
                    # Derived logit-space DVs (rule: report BOTH log-prob + logit).
                    leaf["delta_z_marker"] = st["z_marker_g"] - st["z_marker_b"]
                    leaf["delta_margin"] = (st["z_marker_g"] - st["z_eos_g"]) - (
                        st["z_marker_b"] - st["z_eos_b"]
                    )
                    # #534-spelled alias (same value): scripts/i534_trajectory_
                    # analyze.py and its #555 fork read `delta_z_margin` from
                    # the held_out rows ("alias + new key, never rename").
                    leaf["delta_z_margin"] = leaf["delta_margin"]
            # Source-self mean logit fields over Q_eval (same readout, source row).
            src_stats = [slot_stats[source][q] for q in eval_questions]
            n_src = len(src_stats)
            ck["source_self"].update(
                {
                    "kl_mean": sum(s["kl"] for s in src_stats) / n_src,
                    "z_marker_g_mean": sum(s["z_marker_g"] for s in src_stats) / n_src,
                    "z_marker_b_mean": sum(s["z_marker_b"] for s in src_stats) / n_src,
                    "z_eos_g_mean": sum(s["z_eos_g"] for s in src_stats) / n_src,
                    "z_eos_b_mean": sum(s["z_eos_b"] for s in src_stats) / n_src,
                    "logZ_g_mean": sum(s["logZ_g"] for s in src_stats) / n_src,
                    "logZ_b_mean": sum(s["logZ_b"] for s in src_stats) / n_src,
                    "logp_hf_g_mean": sum(s["logp_hf_g"] for s in src_stats) / n_src,
                    "logp_hf_b_mean": sum(s["logp_hf_b"] for s in src_stats) / n_src,
                    "delta_z_marker_mean": sum(s["z_marker_g"] - s["z_marker_b"] for s in src_stats)
                    / n_src,
                }
            )
            # ── Per-batch byte-identical guard (plan v5 §4.4 fix #1). ─────
            # Reconstruct g/b records from the SAME ck["held_out"] dict the
            # trajectory.json writes — guarantees the guard reads the rig's
            # actual outputs (not a separate forward pass). The held-out
            # panel (NOT panel_plus_source) is the regression input; source
            # is excluded so source saturation (expected per fix #2) doesn't
            # mask a residual byte-identical bug on the bystander panel.
            g_for_guard: dict[str, dict[str, dict[str, float | bool]]] = {}
            b_for_guard: dict[str, dict[str, dict[str, float | bool]]] = {}
            for persona in eval_personas:
                g_for_guard[persona] = {}
                b_for_guard[persona] = {}
                for q in eval_questions:
                    leaf = ck["held_out"][persona][q]
                    g_for_guard[persona][q] = {
                        "logp": float(leaf["g_logp"]),
                        "argmax_marker": bool(leaf["argmax_marker"]),
                    }
                    b_for_guard[persona][q] = {
                        "logp": float(leaf["b_logp"]),
                        "argmax_marker": False,
                    }
            guard_diag = assert_byte_identical_rate_below_threshold(
                g_for_guard,
                b_for_guard,
                kl,
                cell_label=f"{cell_slug}_seed{seed}_frac{frac}",
            )
            ck["byte_identical_guard"] = guard_diag
            # WandB per-(cell, seed, ckpt) continuous diagnostic per plan v5
            # Reproducibility Card ("rate logged to WandB per (cell, seed,
            # ckpt) for analyzer diagnostics").
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            f"{cell_slug}_seed{seed}_ckpt{frac}_byte_identical_rate": (
                                guard_diag["byte_identical_rate"]
                            )
                        }
                    )
            except Exception as e:  # pragma: no cover - wandb optional
                log.info(
                    "wandb log skipped (%s); guard rate=%.4f.", e, guard_diag["byte_identical_rate"]
                )
            partial_path.write_text(
                json.dumps(
                    {"cell": cell_slug, "seed": seed, "checkpoints": checkpoints_out}, indent=2
                )
            )

    payload = {
        "schema_version": "i472_v1",
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
        # Raw-logit fields (z_marker/z_eos/logZ per pair + source means) ride the
        # Phase B forward pass, so they are present iff compute_kl ran. Additive
        # to schema i472_v1 — existing readers are unaffected.
        "logit_fields": bool(compute_kl),
        "post_r_eos_token_id": EXPECTED_POST_R_EOS_ID,
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
