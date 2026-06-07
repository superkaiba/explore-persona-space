# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker token " ※" are intentional
"""Task #508 — per-cell on-policy eval with per-cell base log P (MF1 fix).

Plan §4.6:
    1. Generate per cell — TRAINED R: for each of the 6 trained cells, for each
       of the 15 held-out personas × 20 generic questions, the trained model
       writes its own greedy response (max_new_tokens=2048). 300 per cell.
    2. Trained log P(※) on R_cell — vLLM ``score_logp_for_R`` (reused #472).
    3. Base log P(※) on the SAME R_cell — per-cell subtraction. Base scored
       on each cell's OWN trained R, NOT on a frozen base-generated R.
    4. ΔG_cell = trained_logp(R_cell) − base_logp(R_cell), both on the SAME
       R_cell. 300 ΔG values per cell.

The "per-cell base subtraction" closes the trained-R-differs-by-arm confound
(plan §4.6 second-to-last paragraph). #472's eval read base log P on a frozen
base-generated R; #508 reads it on each cell's own trained R.

LoRA cells use vLLM's ``LoRARequest`` (one vLLM engine + per-cell adapter
swap). Full-FT cells need a separate ``LLM.from_pretrained`` per checkpoint.
Eval reads max_new_tokens=2048 per .claude/rules/marker-leakage-measurement.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    EXPECTED_MARKER_TOKEN_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    assert_marker_token,
    score_logp_for_R,
)
from explore_persona_space.experiments.lora_vs_ft_508 import (
    BASE_MODEL,
    HELD_OUT_PERSONAS_15,
    MAX_NEW_TOKENS_GEN,
    SOURCE_PERSONA,
    load_q_eval,
)

log = logging.getLogger("issue_508.eval_one_cell")

# Eval max-new-tokens floor per .claude/rules/marker-leakage-measurement.md:
# "≥ 2× longest trained completion (default ≥ 2048)". Trained completions
# are capped at MAX_NEW_TOKENS_GEN=1024, so eval must use ≥2048.
EVAL_MAX_NEW_TOKENS = max(2 * MAX_NEW_TOKENS_GEN, 2048)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _generate_trained_R(
    llm,
    tokenizer,
    *,
    eval_personas: dict[str, str],
    eval_questions: list[str],
    cell_label: str,
    use_lora: bool,
    lora_request=None,
) -> dict[str, dict[str, str]]:
    """Greedy-generate the trained model's own R for each (persona, q) probe.

    Returns ``out[persona][q] = response_text``. Logs truncation rate per the
    plan §4.6 step 1 spec (cap=2048; warn if any probe hits the cap).
    """
    from vllm import SamplingParams

    prompts: list[dict] = []
    index_keys: list[tuple[str, str]] = []
    for persona, persona_prompt in eval_personas.items():
        for q in eval_questions:
            messages_q = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": q},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages_q, tokenize=False, add_generation_prompt=True
            )
            prompts.append({"prompt": prompt_text})
            index_keys.append((persona, q))

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=EVAL_MAX_NEW_TOKENS)
    gen_kwargs = {"lora_request": lora_request} if use_lora else {}
    outputs = llm.generate(prompts, sp, **gen_kwargs)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"[{cell_label}] vLLM gen mismatch: {len(outputs)} vs {len(prompts)} probes"
        )

    out: dict[str, dict[str, str]] = {p: {} for p in eval_personas}
    n_truncated = 0
    for (persona, q), gen in zip(index_keys, outputs, strict=True):
        chosen = gen.outputs[0]
        out[persona][q] = chosen.text
        # Truncation detection: stop_reason != "stop" / finish_reason check.
        finish_reason = getattr(chosen, "finish_reason", None)
        if finish_reason == "length":
            n_truncated += 1
    if n_truncated:
        rate = n_truncated / len(outputs)
        if rate > 0.05:
            log.warning(
                "[%s] truncation rate %.2f%% (%d/%d) hit max_tokens=%d",
                cell_label,
                100.0 * rate,
                n_truncated,
                len(outputs),
                EVAL_MAX_NEW_TOKENS,
            )
        else:
            log.info(
                "[%s] truncation rate %.2f%% (%d/%d)",
                cell_label,
                100.0 * rate,
                n_truncated,
                len(outputs),
            )
    return out


def eval_one_cell(  # noqa: C901 - linear multi-phase eval pipeline (gen → trained logp → base logp → ΔG)
    *,
    cell_slug: str,
    arm: str,
    seed: int,
    output_path: Path,
    is_full_ft: bool,
    lora_adapter_path: Path | None = None,
    full_ft_checkpoint_dir: Path | None = None,
    base_model: str = BASE_MODEL,
    persona_bank: dict[str, str] | None = None,
    eval_questions: list[str] | None = None,
    held_out_personas: tuple[str, ...] = HELD_OUT_PERSONAS_15,
    source_persona: str = SOURCE_PERSONA,
    eval_source: bool = True,
) -> dict:
    """Evaluate one cell: trained-R gen → trained log P + per-cell base log P.

    Args:
        cell_slug, arm, seed: cell identity.
        output_path: where the per-cell eval JSON is written.
        is_full_ft: True for full-FT cells (load merged checkpoint via
            ``LLM.from_pretrained(full_ft_checkpoint_dir)``); False for LoRA
            cells (load base + ``LoRARequest(adapter_path)``).
        lora_adapter_path, full_ft_checkpoint_dir: exactly one must be set.
        base_model: HF id of the base model.
        persona_bank: ``{persona_name: system_prompt}`` (defaults to the
            ``EVAL_PERSONAS_24`` panel which carries every persona this
            experiment references).
        eval_questions: 20-question Q_eval pool (default from
            ``load_q_eval()``).
        held_out_personas: 15-persona held-out panel.
        source_persona: source persona (``villain``) — included in eval if
            ``eval_source=True`` so the source-self ΔG gate (plan §6 ≥5 nat)
            is read from the SAME JSON.
        eval_source: if True, also runs the source persona on the same
            questions (20 source-self probes), feeding the §4.4 / §4.5 source-self
            ΔG check.

    Returns:
        Dict with full per-(persona × question) ΔG breakdown + aggregates.
        Also written to ``output_path``.
    """
    from transformers import AutoTokenizer

    if is_full_ft == (lora_adapter_path is not None):
        raise ValueError(
            "Set exactly one of lora_adapter_path / full_ft_checkpoint_dir, and ensure "
            "is_full_ft matches: is_full_ft=True ⇒ full_ft_checkpoint_dir, "
            "is_full_ft=False ⇒ lora_adapter_path."
        )

    if persona_bank is None:
        from explore_persona_space.experiments.factor_screen_365.persona_panel import (
            EVAL_PERSONAS_24,
        )

        persona_bank = dict(EVAL_PERSONAS_24)
    if eval_questions is None:
        eval_questions = load_q_eval()

    # Held-out probe set (15 x 20 = 300 probes).
    missing = [p for p in held_out_personas if p not in persona_bank]
    if missing:
        raise KeyError(f"persona_bank missing held-out personas: {missing}")
    eval_personas_held_out = {p: persona_bank[p] for p in held_out_personas}

    log.info(
        "[%s] starting eval: arm=%s is_full_ft=%s, %d held-out personas x %d q",
        cell_slug,
        arm,
        is_full_ft,
        len(eval_personas_held_out),
        len(eval_questions),
    )

    # ── Load tokenizer + sanity-check marker token (idempotent). ─────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert_marker_token(tokenizer)

    # ── Load the trained model (LoRA adapter OR full-FT merged checkpoint). ──
    from vllm import LLM

    if is_full_ft:
        log.info("[%s] loading full-FT merged checkpoint: %s", cell_slug, full_ft_checkpoint_dir)
        trained_llm = LLM(
            model=str(full_ft_checkpoint_dir),
            tokenizer=base_model,  # tokenizer is identical to base.
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            max_model_len=4096,
            dtype="bfloat16",
        )
        lora_req = None
    else:
        log.info("[%s] loading base + LoRA adapter: %s", cell_slug, lora_adapter_path)
        trained_llm = LLM(
            model=base_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1,
            trust_remote_code=True,
            max_model_len=4096,
            dtype="bfloat16",
        )
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest(
            lora_name=cell_slug,
            lora_int_id=1,
            lora_path=str(lora_adapter_path),
        )

    # ── Phase 1: trained-R generation on held-out (+ source) panel. ──────────
    log.info("[%s] phase=eval_gen_trained", cell_slug)
    print(f"[phase=eval_gen_trained cell={cell_slug}]", flush=True)
    held_out_R = _generate_trained_R(
        trained_llm,
        tokenizer,
        eval_personas=eval_personas_held_out,
        eval_questions=eval_questions,
        cell_label=f"{cell_slug}/held_out",
        use_lora=not is_full_ft,
        lora_request=lora_req,
    )
    source_R: dict[str, dict[str, str]] = {}
    if eval_source:
        source_personas = {source_persona: persona_bank[source_persona]}
        source_R = _generate_trained_R(
            trained_llm,
            tokenizer,
            eval_personas=source_personas,
            eval_questions=eval_questions,
            cell_label=f"{cell_slug}/source",
            use_lora=not is_full_ft,
            lora_request=lora_req,
        )

    # ── Phase 2: trained log P(※) on R_cell. ─────────────────────────────────
    log.info("[%s] phase=eval_logp_trained", cell_slug)
    print(f"[phase=eval_logp_trained cell={cell_slug}]", flush=True)
    trained_logp_held_out = score_logp_for_R(
        trained_llm,
        tokenizer,
        r_by_persona_q=held_out_R,
        eval_personas=eval_personas_held_out,
        eval_questions=eval_questions,
        cell_label=f"{cell_slug}/held_out_trained",
        use_lora=not is_full_ft,
        lora_request=lora_req,
    )
    trained_logp_source: dict = {}
    if eval_source:
        trained_logp_source = score_logp_for_R(
            trained_llm,
            tokenizer,
            r_by_persona_q=source_R,
            eval_personas={source_persona: persona_bank[source_persona]},
            eval_questions=eval_questions,
            cell_label=f"{cell_slug}/source_trained",
            use_lora=not is_full_ft,
            lora_request=lora_req,
        )

    # ── Tear down trained vLLM engine before loading base. ───────────────────
    # vLLM worker teardown (.claude/rules/gotchas.md): del + destroy + kill
    # surviving worker subprocesses. Otherwise the freed GPU memory is
    # re-grabbed by orphan workers when we load the base engine and the
    # base load OOMs.
    log.info("[%s] tearing down trained engine", cell_slug)
    import gc

    import torch

    del trained_llm
    try:
        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except (ImportError, RuntimeError) as e:
        log.warning("[%s] vLLM teardown helpers raised %s — continuing", cell_slug, e)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 3: base log P(※) on the SAME R_cell (MF1 per-cell subtraction).
    log.info("[%s] phase=eval_logp_base (per-cell base on cell's own R)", cell_slug)
    print(f"[phase=eval_logp_base cell={cell_slug}]", flush=True)
    base_llm = LLM(
        model=base_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16",
    )
    base_logp_held_out = score_logp_for_R(
        base_llm,
        tokenizer,
        r_by_persona_q=held_out_R,
        eval_personas=eval_personas_held_out,
        eval_questions=eval_questions,
        cell_label=f"{cell_slug}/held_out_base",
        use_lora=False,
        lora_request=None,
    )
    base_logp_source: dict = {}
    if eval_source:
        base_logp_source = score_logp_for_R(
            base_llm,
            tokenizer,
            r_by_persona_q=source_R,
            eval_personas={source_persona: persona_bank[source_persona]},
            eval_questions=eval_questions,
            cell_label=f"{cell_slug}/source_base",
            use_lora=False,
            lora_request=None,
        )

    # Final base teardown (mirror).
    del base_llm
    try:
        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except (ImportError, RuntimeError):
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Phase 4: compute ΔG_cell = trained_logp − base_logp per probe. ───────
    delta_g_held_out: dict[str, dict[str, dict]] = {}
    n_collapsed = 0
    for persona in eval_personas_held_out:
        delta_g_held_out[persona] = {}
        for q in eval_questions:
            tr = trained_logp_held_out[persona][q]
            bs = base_logp_held_out[persona][q]
            collapsed = bool(tr.get("r_collapsed") or bs.get("r_collapsed"))
            if collapsed:
                n_collapsed += 1
            delta_g_held_out[persona][q] = {
                "trained_logp": float(tr["logp"]),
                "base_logp": float(bs["logp"]),
                "delta_g": float(tr["logp"]) - float(bs["logp"]),
                "trained_argmax_marker": bool(tr["argmax_marker"]),
                "base_argmax_marker": bool(bs["argmax_marker"]),
                "r_collapsed": collapsed,
                "n_marker_in_R": int(tr.get("n_marker_in_R", 0)),
            }

    delta_g_source: dict[str, dict[str, dict]] = {}
    if eval_source:
        delta_g_source[source_persona] = {}
        for q in eval_questions:
            tr = trained_logp_source[source_persona][q]
            bs = base_logp_source[source_persona][q]
            delta_g_source[source_persona][q] = {
                "trained_logp": float(tr["logp"]),
                "base_logp": float(bs["logp"]),
                "delta_g": float(tr["logp"]) - float(bs["logp"]),
                "trained_argmax_marker": bool(tr["argmax_marker"]),
                "base_argmax_marker": bool(bs["argmax_marker"]),
                "r_collapsed": bool(tr.get("r_collapsed") or bs.get("r_collapsed")),
                "n_marker_in_R": int(tr.get("n_marker_in_R", 0)),
            }

    # Aggregates.
    held_out_dg_values = [
        delta_g_held_out[p][q]["delta_g"]
        for p in eval_personas_held_out
        for q in eval_questions
        if not delta_g_held_out[p][q]["r_collapsed"]
    ]
    held_out_mean = (
        sum(held_out_dg_values) / len(held_out_dg_values) if held_out_dg_values else float("nan")
    )
    source_dg_values = (
        [
            delta_g_source[source_persona][q]["delta_g"]
            for q in eval_questions
            if not delta_g_source[source_persona][q]["r_collapsed"]
        ]
        if eval_source
        else []
    )
    source_mean = (
        sum(source_dg_values) / len(source_dg_values) if source_dg_values else float("nan")
    )

    # ── Persist result + reproducibility metadata. ───────────────────────────
    result = {
        "schema_version": "i508_eval_v1",
        "cell_slug": cell_slug,
        "arm": arm,
        "seed": seed,
        "base_model": base_model,
        "is_full_ft": is_full_ft,
        "lora_adapter_path": str(lora_adapter_path) if lora_adapter_path else None,
        "full_ft_checkpoint_dir": str(full_ft_checkpoint_dir) if full_ft_checkpoint_dir else None,
        "marker_text": MARKER_TEXT,
        "marker_token_id_expected": EXPECTED_MARKER_TOKEN_ID,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "held_out_personas": list(held_out_personas),
        "eval_questions": list(eval_questions),
        "source_persona": source_persona if eval_source else None,
        "delta_g_held_out": delta_g_held_out,
        "delta_g_source": delta_g_source,
        "trained_R_held_out": held_out_R,
        "trained_R_source": source_R if eval_source else {},
        "aggregates": {
            "held_out_mean_delta_g": held_out_mean,
            "held_out_n_probes": len(held_out_dg_values),
            "held_out_n_collapsed": n_collapsed,
            "source_self_mean_delta_g": source_mean,
            "source_n_probes": len(source_dg_values),
        },
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info(
        "[%s] wrote eval JSON → %s (held_out mean ΔG=%.3f, source mean=%.3f, n_collapsed=%d)",
        cell_slug,
        output_path,
        held_out_mean,
        source_mean,
        n_collapsed,
    )
    return result
