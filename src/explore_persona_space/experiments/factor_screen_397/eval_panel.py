"""Per-checkpoint log-prob + final-checkpoint substring eval panel (task #397).

Plan v4 §4.3 + §5.5 are authoritative. This module covers:

- ``compute_logprob_panel`` — per-checkpoint log-prob eval using the peft
  0.18.1 ``load_adapter``/``set_adapter``/``delete_adapter`` lifecycle with a
  single resident base model, calling
  ``explore_persona_space.eval.marker_logprob.compute_marker_logprob`` for
  the configured marker variants (plan A15 says report BOTH ``"\\n\\n※"``
  and ``"※"`` at the final checkpoint; ``"※"`` only at intermediate
  checkpoints for cost).
- ``score_markers_threaded`` — thin wrapper around the substring scorer that
  forces the caller to pass ``marker=...`` as a required kwarg (M1
  carry-forward; never silently default to ``[ZLT]``).
- ``build_train_matched_persona_panel`` + ``read_prepared_dataset_manifest``
  — recipe-fix step 5b helpers. Persists / reads the source-persona's
  training-time system prompt so the eval can override the canonical
  ``EVAL_PERSONAS_24`` entry for that source. Required for C=1 cells (which
  train on "Background context: ..." prompts rather than canonical "You are
  X") — without the override the C-axis selectivity Δ measurement is
  conflated with eval-time distribution shift.

The marker-threading discipline is the load-bearing bit: every
``score_markers`` call site in #397 MUST thread ``marker=args.marker_token``
because the module-level default in factor_screen_365 is ``[ZLT]`` and the
runtime marker switched to ``※``. ``score_markers_threaded`` is the
mechanical guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from explore_persona_space.eval.marker_logprob import compute_marker_logprob
from explore_persona_space.experiments.factor_screen_365.eval_panel import (
    score_markers as _score_markers_underlying,
)

# Plan v4 §3 (Method delta vs #383, "Marker" row): single-token marker.
DEFAULT_MARKER_TEXT: str = "※"

# Plan v4 A15 + analyzer guidance #9: BOTH marker variants reported at the
# final checkpoint, ``"※"`` only at intermediate checkpoints (cost-driven).
FINAL_CHECKPOINT_MARKER_VARIANTS: tuple[str, ...] = ("\n\n※", "※")
INTERMEDIATE_CHECKPOINT_MARKER_VARIANTS: tuple[str, ...] = ("※",)


def read_prepared_dataset_manifest(cell_output_dir: Path) -> dict[str, Any] | None:
    """Read the per-cell training manifest if present, else return ``None``.

    The manifest is written by ``training.write_prepared_dataset_manifest``
    (or ``training.train_one_cell`` when ``system_prompt_text`` is supplied).
    Path: ``cell_output_dir / 'prepared_dataset.json'``.

    Returns ``None`` (NOT a default panel) ONLY when the manifest file is
    missing — the caller is expected to fail loud OR fall back to the
    canonical panel with a logged warning (recipe-fix back-compat).

    **Fail-loud (reconciler SR2):** when the file is present but corrupted
    JSON, raises ``ValueError`` wrapping ``json.JSONDecodeError`` with the
    offending path so the dispatcher cannot silently treat a corrupt
    manifest as "no manifest". See ``build_train_matched_persona_panel``
    for the structural-validity check on the parsed payload.
    """
    path = cell_output_dir / "prepared_dataset.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"prepared_dataset manifest is corrupted JSON at {path}: {e}. "
            "Re-prepare the cell instead of silently falling back."
        ) from e


def build_train_matched_persona_panel(
    canonical_panel: dict[str, str],
    *,
    source: str,
    manifest: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build a persona panel with the source persona's prompt overridden.

    Plan v4 §5.1.0 step 5b. Returns ``(panel, system_prompt_overrides)``:

    - ``panel`` is the per-persona ``{persona_name: system_prompt}`` mapping
      with the source persona's entry replaced by the training-time prompt
      from ``manifest['system_prompt_text']`` if available. Bystanders stay
      on their canonical EVAL_PERSONAS_24 prompts.
    - ``system_prompt_overrides`` is a parallel ``{persona_name: override}``
      dict (single entry for the source when overridden, otherwise empty) —
      mirrors the ``compute_logprob_panel`` ``system_prompt_overrides`` kwarg
      so the caller can thread it through cleanly.

    **Fail-loud manifest contract (reconciler SR2):**

    - ``manifest is None`` is the ONLY accepted "no override" path — used by
      legacy cells trained before the recipe-fix landed. Callers SHOULD log
      a warning when they hit this branch; the C-axis selectivity Δ
      measurement on legacy cells is conflated with distribution shift.
    - ``manifest is not None`` MUST carry a ``system_prompt_text`` key whose
      value is a non-empty string. Missing key OR non-string OR empty
      string RAISES ``ValueError``. A partially-populated or corrupted
      manifest is a recipe-fix invariant violation and must surface, not
      silently fall back to the canonical panel (CLAUDE.md "Fail fast —
      never hide failures").
    """
    if not isinstance(canonical_panel, dict) or not canonical_panel:
        raise ValueError("canonical_panel must be a non-empty dict")
    if source not in canonical_panel:
        raise ValueError(
            f"source persona {source!r} not in canonical_panel; "
            f"available: {sorted(canonical_panel)}"
        )

    panel = dict(canonical_panel)
    overrides: dict[str, str] = {}

    if manifest is not None:
        # Reconciler SR2 — fail loud on partially-populated manifest. The
        # only valid "no override" signal is manifest is None (legacy cell);
        # anything else is a recipe-fix invariant violation and must surface.
        if "system_prompt_text" not in manifest:
            raise ValueError(
                "manifest is not None but missing 'system_prompt_text' key; "
                "this is a recipe-fix invariant violation. Pass manifest=None "
                "explicitly for legacy cells (with a logged warning), or "
                f"re-prepare the cell. Manifest keys: {sorted(manifest)}"
            )
        sp = manifest["system_prompt_text"]
        if not isinstance(sp, str) or not sp:
            raise ValueError(
                f"manifest['system_prompt_text'] must be a non-empty string; got {sp!r}"
            )
        panel[source] = sp
        overrides[source] = sp

    return panel, overrides


def _build_chat_template_contexts(
    tokenizer,
    *,
    personas: dict[str, str],
    questions: list[str],
    system_prompt_overrides: dict[str, str] | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Build chat-templated contexts for ``(persona, question)`` pairs.

    Returns ``(contexts, keys)`` where ``keys[i] = (persona, question)`` for
    ``contexts[i]``. Per plan v4 §4.3, contexts use ``[system, user] +
    add_generation_prompt=True`` so the marker is scored at the
    first-assistant-token position.

    ``system_prompt_overrides`` (BLOCKER 3): when a persona key is present in
    the override dict, the override REPLACES the panel's system prompt for
    that persona only — used by the train-matched eval path for C=1 cells.
    """
    contexts: list[str] = []
    keys: list[tuple[str, str]] = []
    overrides = system_prompt_overrides or {}
    for persona_name, panel_sys_prompt in personas.items():
        system_prompt = overrides.get(persona_name, panel_sys_prompt)
        for question in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            ctx = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            contexts.append(ctx)
            keys.append((persona_name, question))
    return contexts, keys


def compute_logprob_panel(
    *,
    base_model,
    tokenizer,
    checkpoint_dirs: list[str],
    contexts: list[str] | None = None,
    personas: dict[str, str] | None = None,
    questions: list[str] | None = None,
    system_prompt_overrides: dict[str, str] | None = None,
    marker_texts: tuple[str, ...] = FINAL_CHECKPOINT_MARKER_VARIANTS,
    batch_size: int = 8,
    device: str = "cuda:0",
    adapter_name_prefix: str = "ck",
) -> dict[str, Any]:
    """Per-checkpoint log-prob eval with peft 0.18.1 adapter-swap lifecycle.

    Loads each adapter onto the SAME resident ``base_model`` (a peft
    ``PeftModel`` instance the caller already constructed with at least one
    adapter loaded — typical pattern: pre-load the first adapter via
    ``PeftModel.from_pretrained`` so the base is wrapped, then this function
    cycles the rest via ``load_adapter`` / ``set_adapter`` / ``delete_adapter``).

    Two context-construction paths (exactly one must be supplied):

    1. ``contexts`` — pre-built chat-template strings. The caller has already
       applied any persona overrides (back-compat path).
    2. ``(personas, questions)`` + optional ``system_prompt_overrides`` —
       this function builds chat-template contexts internally via
       ``[system, user] + add_generation_prompt=True`` (plan v4 §4.3). When
       ``system_prompt_overrides`` is provided, per-persona overrides REPLACE
       the panel system prompt for those personas only — used by the recipe-
       fix step 5b train-matched eval path for C=1 cells. Bystanders not in
       the override dict use the panel prompt unchanged.

    For each checkpoint dir, for each ``marker_text`` in ``marker_texts``:
      1. ``base_model.load_adapter(ck_dir, adapter_name=f"{prefix}{i}")``
         (skipped if the adapter is already loaded under this name).
      2. ``base_model.set_adapter(f"{prefix}{i}")``
      3. Call ``compute_marker_logprob(base_model, tokenizer, contexts,
         marker_text=mt, position="end_of_answer", batch_size=batch_size,
         device=device)``.
      4. ``base_model.delete_adapter(f"{prefix}{i}")`` before swapping to the
         next checkpoint, releasing the adapter's GPU memory.

    Returns a dict keyed by checkpoint dir (string) → marker variant string
    → ``list[float]`` of log-probs (one entry per context). When the
    ``(personas, questions)`` path is used, the return dict also carries
    a ``"_context_keys"`` entry mapping context index → ``[persona, question]``
    so downstream aggregation can rebuild the per-(persona, question) matrix.

    Plan v4 §5.5 + A17: ``unload()`` does NOT exist on peft 0.18.1, so the
    canonical multi-checkpoint pattern uses only
    ``load_adapter`` + ``set_adapter`` + ``delete_adapter``.
    """
    assert len(checkpoint_dirs) > 0, "compute_logprob_panel called with no checkpoint dirs"
    assert len(marker_texts) > 0, "compute_logprob_panel called with no marker_texts"

    # Resolve context-construction path. Exactly one input shape allowed.
    context_keys: list[tuple[str, str]] | None = None
    if contexts is not None:
        if personas is not None or questions is not None or system_prompt_overrides is not None:
            raise ValueError(
                "compute_logprob_panel: pass either contexts OR "
                "(personas, questions[, system_prompt_overrides]); not both"
            )
    else:
        if personas is None or questions is None:
            raise ValueError(
                "compute_logprob_panel: must supply either contexts or "
                "(personas, questions) — both paths are missing"
            )
        contexts, context_keys = _build_chat_template_contexts(
            tokenizer,
            personas=personas,
            questions=questions,
            system_prompt_overrides=system_prompt_overrides,
        )

    assert len(contexts) > 0, "compute_logprob_panel resolved to zero contexts"

    out: dict[str, Any] = {}
    for i, ck_dir in enumerate(checkpoint_dirs):
        adapter_name = f"{adapter_name_prefix}{i}"
        already_loaded = (
            hasattr(base_model, "peft_config") and adapter_name in base_model.peft_config
        )
        if not already_loaded:
            base_model.load_adapter(ck_dir, adapter_name=adapter_name)
        base_model.set_adapter(adapter_name)

        per_marker: dict[str, list[float]] = {}
        for mt in marker_texts:
            logps = compute_marker_logprob(
                base_model,
                tokenizer,
                contexts,
                marker_text=mt,
                position="end_of_answer",
                batch_size=batch_size,
                device=device,
            )
            assert len(logps) == len(contexts), (
                f"compute_marker_logprob returned {len(logps)} entries; expected {len(contexts)}"
            )
            per_marker[mt] = logps
        out[ck_dir] = per_marker

        # Release this adapter's GPU memory before loading the next.
        base_model.delete_adapter(adapter_name)

    if context_keys is not None:
        out["_context_keys"] = [list(k) for k in context_keys]

    return out


def score_markers_threaded(
    completions: dict[str, dict[str, list[str]]],
    marker: str,
) -> dict[str, dict]:
    """Substring rate per persona x question, with the marker explicitly threaded.

    Plan v4 §4.3 + control 7 (M1 carry-forward): the marker MUST be threaded
    through every ``score_markers`` call; default-marker fallback silently
    zeroes out when the runtime marker differs from the module-level
    ``MARKER`` (``[ZLT]`` in #383/#365). This wrapper makes ``marker`` a
    required positional/kwarg — callers cannot drop it without a static
    error.

    Delegates to ``factor_screen_365.eval_panel.score_markers`` which already
    accepts the ``marker`` kwarg (post-recipe-fix; the underlying function
    was always parameterized — the M1 failure was at the CALL site, not
    here).
    """
    if not isinstance(marker, str) or not marker:
        raise ValueError(f"marker must be a non-empty string; got {marker!r}")
    return _score_markers_underlying(completions, marker=marker)


# ---------------------------------------------------------------------------
# vLLM --enable-lora sampled eval (Round 6 — drops the merge step)
# ---------------------------------------------------------------------------

DEFAULT_VLLM_LORA_MAX_RANK: int = 32
DEFAULT_NUM_COMPLETIONS: int = 5
DEFAULT_MAX_NEW_TOKENS: int = 2048
DEFAULT_MAX_MODEL_LEN: int = 4096


def generate_completions_with_lora(
    *,
    base_model_path: str,
    lora_path: str,
    personas: dict[str, str],
    questions: list[str],
    num_completions: int = DEFAULT_NUM_COMPLETIONS,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_lora_rank: int = DEFAULT_VLLM_LORA_MAX_RANK,
    gpu_memory_utilization: float | None = None,
    seed: int = 42,
    system_prompt_overrides: dict[str, str] | None = None,
    lora_name: str = "i397-cell",
) -> dict[str, dict[str, list[str]]]:
    """Sampled eval with vLLM ``--enable-lora`` (NO merge).

    Round 6: replaces the previous ``EvalConfig + generate_completions(
    merged_path)`` pattern with vLLM's LoRA-adapter mode. vLLM loads the
    BASE model once with ``enable_lora=True, max_loras=1,
    max_lora_rank=<r>``, then ``LLM.generate(..., lora_request=
    LoRARequest(lora_name, 1, lora_path=adapter_path))`` consumes the
    adapter at inference time without merging weights. This eliminates
    the ~14 GB merged-dir per cell that drove the 6/8 concurrency cap.

    Mirrors ``factor_screen_365.eval_panel.generate_completions`` for
    everything except the model-load + lora_request handoff:

    - Same chat-template prompt construction (system_prompt_overrides
      threaded through per BLOCKER 3 from round 2 — the train-matched
      eval path).
    - Same return shape ``{persona: {question: [completions]}}``.
    - Same fail-fast post-load cleanup (``del llm`` + ``gc.collect()`` +
      ``torch.cuda.empty_cache()``; the vLLM-orphan-worker note still
      applies if the caller loads HF Transformers after this returns).

    Args:
        base_model_path: HF id or local path of the Qwen-2.5-7B-Instruct
            base model.
        lora_path: Directory containing the LoRA adapter weights
            (``adapter_model.safetensors`` + ``adapter_config.json``).
            For #397 this is ``TrainOutcome.adapter_path``.
        personas, questions: 24-persona x 20-question panel from
            ``factor_screen_365.persona_panel``.
        system_prompt_overrides: Per-persona system-prompt override
            map (SR1 wiring — passes train-matched prompt for source
            persona on C=1 cells). When None, panel system prompts
            are used unchanged.
        max_lora_rank: vLLM's compile-time max LoRA rank. #397 trains
            r=32, so the default 32 is the correct ceiling — higher
            ranks would burn a small amount of GPU memory; lower would
            silently truncate.
        lora_name: Display name for the LoRARequest. Not load-bearing
            (logging only).

    Returns:
        ``{persona: {question: [completion_text, ...]}}`` matching
        ``generate_completions``'s shape so ``score_markers_threaded``
        can consume it unchanged.

    Raises propagated from vLLM (failed adapter load, OOM, etc.). The
    caller is expected to surface these — per CLAUDE.md "fail fast",
    silent fallback to merged-model load is forbidden (Round 6 brief).
    """
    # Defer the heavy imports so this module can be inspected on CPU-only
    # test runs without dragging in vLLM.
    import gc
    import os

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.factor_screen_365.eval_panel import (
        _patch_tokenizer_for_vllm,
    )

    _patch_tokenizer_for_vllm()

    gpu_mem = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build chat-templated prompts (with optional per-persona overrides for
    # the train-matched eval SR1 wiring).
    overrides = system_prompt_overrides or {}
    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for persona_name, panel_sys_prompt in personas.items():
        system_prompt = overrides.get(persona_name, panel_sys_prompt)
        for question in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            keys.append((persona_name, question))

    # vLLM LLM with --enable-lora. max_loras=1 because we only ever swap
    # one adapter per process; max_lora_rank=32 matches the #397 training
    # cfg (lora_r=32).
    llm = LLM(
        model=base_model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        seed=seed,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=max_lora_rank,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    lora_request = LoRARequest(lora_name=lora_name, lora_int_id=1, lora_path=lora_path)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results: dict[str, dict[str, list[str]]] = {n: {} for n in personas}
    for out, (persona, question) in zip(outputs, keys, strict=True):
        results[persona][question] = [o.text for o in out.outputs]

    # Free GPU memory before any post-eval framework load. Per the
    # `vllm_orphan_worker_after_destroy` agent-memory note, this is the
    # LOW-risk direction (vLLM owns the GPU first, then nothing else
    # loads in this process); the caller is responsible for the
    # destroy_*/psutil child-kill discipline if it loads HF after.
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
