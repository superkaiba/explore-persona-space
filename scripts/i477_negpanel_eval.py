# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + minus sign − intentional
#!/usr/bin/env python3
"""Task #477 inline follow-up — NEGATIVE-PANEL self-leakage eval.

Eval-only follow-up to #477's row-scaled count grid: for every already-trained
#477 adapter, measure marker leakage AT EACH CELL'S OWN trained negatives (the
prior #472/#477 grids excluded every trained negative from the held-out panel
by design — zero committed data exists on leakage AT the negatives). Hypothesis:
trained negatives are point-suppressed (on-policy post-response ``log P(※)``
trained − base stays at or below ~0, far below bystander levels) while bystander
leakage rises with the count+rows+steps bundle.

Single design change vs ``i477_reval_grid.py``: the probe panel = ``negatives_for_cell``
for THIS cell, NOT the #477-disjoint held-out panel. Plus three mandated additions
(see the approved ``epm:followup-scope v1`` marker on task #477):

1. ``max_new_tokens=2048`` (≥2× the 1024-token trained completion cap, per
   CLAUDE.md). ``max_model_len=4096`` to fit prompt + 2048-token generation.
2. Four-float HF slot stats per (persona, q) per side from
   ``compute_marker_slot_stats``: ``{logp, z_marker, z_eos, logZ}``. vLLM cannot
   give raw logits (post-softmax only), so the HF pass runs per-cell after vLLM
   teardown (~+40s/cell). Cross-validated against vLLM ``log P(※)`` per side.
3. Raw completions persisted per cell to ``raw_completions.json`` and uploaded
   to the HF data repo via ``upload_raw_completions_to_data_repo``.

Per-cell pipeline (single shared phase ordering — smoke = production with one cell):

  Phase 0: fetch adapter (per-file ``hf_hub_download``, snapshot_download has
           the siblings-truncation bug on this repo — #480 / #477 recovery).
  Phase 1: vLLM engine up → on-policy R for (cell-negs ∪ {source}) × Q_eval →
           score ``log P(※)`` g (use_lora=True) + b (use_lora=False) →
           ``assert_adapter_actually_applied`` (round-4 redesign, non-raising,
           returns guard verdict dict) → vLLM teardown (reaps worker
           subprocesses + GPU memory).
  Phase 2: HF + PEFT four-float pass on the SAME on-policy R: build the
           without-marker context (chat-templated prompt + R + MARKER_SEP),
           assert tokenized length equals ``build_full_ids`` slot position
           (drift guard), call ``compute_marker_slot_stats`` trained
           (adapter loaded) + base (``disable_adapter``). Persist all four
           floats per (persona, q) per side.
  Phase 3: cross-validate (max |logp_HF − logp_vLLM| per side; WARN above
           ~0.5 nats trained / ~0.2 nats base, never crash — deviation IS
           a finding).
  Phase 4: build the analyze-compatible checkpoint payload (REUSE
           ``i477_reval_grid._build_checkpoint_payload`` with negatives panel
           in ``held_out``) + ``attach_marker_channel_aggregates``. Stamp
           ``panel_kind="trained_negatives"`` so nobody mistakes it for
           bystanders.
  Phase 5: write per-cell raw_completions.json (driver uploads after all cells).
  Phase 6: write per-cell summary JSON. Aggregate to grid.json after all cells.

Cell ordering: calA (12) → calA0 (3) → calib (20).

Parallelism + CLI: identical to ``i477_reval_grid.py`` — ``--gpus N`` partitions
cells round-robin and spawns N worker subprocesses with
``CUDA_VISIBLE_DEVICES=k`` and the explicit env dict. ``--phase`` / ``--cells``
/ ``--dry-run`` flags inherited.

Wall-time budget: 35 cells × (~50s vLLM + ~45s HF load + scoring) ≈ ~60 min
single-GPU; on a 1×H100 the headline driver invocation is:

    nohup uv run python scripts/i477_negpanel_eval.py --gpus 1 \\
      > /workspace/logs/issue-477-negpanel.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Two-check subprocess-env-passthrough: explicit env={**os.environ} on every
# subprocess call below AND load_dotenv() at module-top so HF_TOKEN /
# WANDB_API_KEY land in the parent's env before any subprocess copies it.
load_dotenv()

# Make the helper module on the same path importable when this script is
# invoked as a worker subprocess (uv run python scripts/i477_negpanel_eval.py).
# scripts/ is on PYTHONPATH only if this driver lives next to its helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import i477_reval_grid  # noqa: E402 - intentional: helper module on the same path

log = logging.getLogger("i477.negpanel_eval")

# ── Constants reused verbatim from the recovery driver. ──────────────────────
ADAPTER_HF_REPO = i477_reval_grid.ADAPTER_HF_REPO
ADAPTER_SUBFOLDER_ROOT = i477_reval_grid.ADAPTER_SUBFOLDER_ROOT
DATA_REPO = i477_reval_grid.DATA_REPO
DATA_REVISION = i477_reval_grid.DATA_REVISION
DATA_PREFIX = i477_reval_grid.DATA_PREFIX
LOCAL_DATA_ROOT = i477_reval_grid.LOCAL_DATA_ROOT
LOCAL_PATHS = i477_reval_grid.LOCAL_PATHS

# ── Negpanel-specific constants (the three brief-mandated additions). ────────
DEFAULT_OUT_ROOT = Path("eval_results/issue_477/negpanel_eval")
DEFAULT_ADAPTER_CACHE = Path("/tmp/i477_negpanel_eval/adapter_cache")

# max_new_tokens=2048 satisfies CLAUDE.md "≥ 2× longest trained completion"
# for marker / end-of-completion evals (#477 trained completions at 1024 cap).
# max_model_len=4096 fits prompt (~1500 toks worst-case Qwen chat template +
# question) + 2048-token generation with margin.
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEM_UTIL = i477_reval_grid.DEFAULT_GPU_MEM_UTIL

# Cross-validation tolerances (advisory — log WARN, never crash; deviation
# above these IS a finding worth recording in the payload).
HF_VS_VLLM_BASE_WARN_NATS = 0.2  # PEFT-disable-adapter on shared HF model ≈ vLLM no-LoRA
HF_VS_VLLM_TRAINED_WARN_NATS = 0.5  # PEFT-load-adapter vs vLLM LoRARequest in bf16


# ── Cell ordering: calA → calA0 → calib (priority from the brief). ───────────
_PHASE_ORDER: dict[str, int] = {"calA": 0, "calA0": 1, "calib": 2}


def _phase_sort_key(entry: i477_reval_grid.CellEntry) -> tuple[int, str]:
    return (_PHASE_ORDER.get(entry.phase, 99), entry.adapter_dirname)


# ── Negatives panel = the cell's OWN trained negatives. ──────────────────────
def _select_negative_panel(
    *, entry: i477_reval_grid.CellEntry
) -> tuple[dict[str, str], list[str], str, str]:
    """Build the cell's TRAINED-negative panel + Q_eval + source.

    Single changed variable vs ``i477_reval_grid._select_eval_slice``: the
    probe panel is the cell's OWN trained negatives, NOT the #477-disjoint
    held-out panel. The vLLM scorer + analyze payload otherwise identical.
    """
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
        negatives_for_cell,
    )

    bank = load_persona_bank(LOCAL_PATHS[f"{DATA_PREFIX}/geometry/persona_bank.json"])
    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, LOCAL_DATA_ROOT)
    neg_names = negatives_for_cell(
        entry.logical_slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477
    )
    if not neg_names:
        raise ValueError(
            f"_select_negative_panel: cell {entry.logical_slug!r} resolved to an EMPTY "
            f"negative panel — refusing to eval (placement='none' arms have no negatives "
            f"to read leakage AT; not applicable to this follow-up)."
        )
    eval_personas = {p: bank[p] for p in neg_names}
    _q_train, q_eval = get_train_eval_questions()
    return eval_personas, list(q_eval), SOURCE_PERSONA, bank[SOURCE_PERSONA]


# ── HF four-float context construction (parity with build_full_ids). ─────────
def _build_slot_context(
    tokenizer,
    persona_prompt: str,
    question: str,
    r_text: str,
) -> str:
    """Construct the without-marker context whose last token is the marker slot.

    Mirrors ``build_full_ids`` (eval_one_cell.py) tokenization EXCEPT it omits
    the trailing marker token. compute_marker_slot_stats reads
    ``logits[:, -1, :]`` (next-token logits at the slot the marker WOULD
    occupy). Slot-parity assertion against ``build_full_ids`` is the
    caller's responsibility (see ``_assert_slot_parity``).

    Returns the raw context STRING (compute_marker_slot_stats re-tokenizes
    it; we use the rig's MARKER_SEP separator to match the trained slot
    exactly per the marker-leakage-measurement.md C1 contract).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import MARKER_SEP

    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_text + r_text + MARKER_SEP


def _assert_slot_parity(
    tokenizer,
    *,
    persona_prompt: str,
    question: str,
    r_text: str,
    persona_name: str,
) -> None:
    """Fail loud if the HF slot context drifts from build_full_ids' slot position.

    build_full_ids returns ``(full_ids, _, _, slot, _)`` with the appended
    marker at ``full_ids[-1]`` (so slot == len(full_ids) - 1). Our HF context
    is build_full_ids' full_ids[:-1]; its tokenization length MUST equal
    ``slot`` so compute_marker_slot_stats' ``logits[:, -1, :]`` reads the
    SAME next-token logits the vLLM scorer reads via prompt_logprobs[slot].
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_SEP,
        MARKER_TEXT,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        build_full_ids,
    )

    full_ids, _prompt_len, _r_len, slot, _n_marker_in_R = build_full_ids(
        tokenizer,
        persona_prompt,
        question,
        r_text,
        MARKER_TEXT,
        EXPECTED_MARKER_TOKEN_ID,
        persona_name,
        question,
        sep=MARKER_SEP,
    )
    context = _build_slot_context(tokenizer, persona_prompt, question, r_text)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    # Exact prefix equality (not just length): same-length but id-level drift at
    # the sep/marker boundary would silently shift the slot read (code-review-codex
    # v1 Minor — strictly stronger than the length-only check).
    if context_ids != list(full_ids[:slot]):
        raise AssertionError(
            f"slot-parity drift persona={persona_name!r} q={question!r}: "
            f"build_full_ids slot={slot} (full_ids len={len(full_ids)}), "
            f"HF context tokenized to len={len(context_ids)}; "
            f"exact-prefix equality vs full_ids[:slot] FAILED. "
            f"compute_marker_slot_stats would read the marker slot at the WRONG position."
        )


# ── HF four-float phase (per-cell, after vLLM teardown). ─────────────────────
def _hf_four_float_phase(
    *,
    entry: i477_reval_grid.CellEntry,
    adapter_dir: Path,
    r_on_policy: dict[str, dict[str, str]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    source_name: str,
    source_prompt: str,
    token: str | None,
    batch_size: int = 8,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],  # trained per (persona, q) → 4 floats
    dict[str, dict[str, dict[str, float]]],  # base    per (persona, q) → 4 floats
]:
    """Per-cell HF + PEFT four-float slot-stats pass on the SAME on-policy R.

    Loads the base model in bf16, attaches the cell's LoRA adapter via PEFT,
    runs ``compute_marker_slot_stats`` on the WITHOUT-marker context for
    every (persona, q) including the source, then disables the adapter and
    re-runs to get the base side. Mirrors issue531_logit_rescore.process_cell
    (the canonical established pattern; see scripts/issue531_logit_rescore.py).

    Persona+source set: panel ∪ {source} — the rig wants source-self stats
    in the analyze payload (see ``_build_checkpoint_payload``).

    Returns two dicts of the SAME shape:
        out[persona][q] = {"logp": ..., "z_marker": ..., "z_eos": ..., "logZ": ...}
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import (
        assert_gauge_free_adapter_config,
        compute_marker_slot_stats,
    )
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import BASE_MODEL
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        EXPECTED_MARKER_TOKEN_ID,
        MARKER_TEXT,
    )

    # Gauge assert BEFORE loading the adapter onto the HF model (the trained −
    # base logit readout is valid only when LoRA leaves W_U / embeddings alone).
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert_gauge_free_adapter_config(cfg, context=str(adapter_dir))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    if tokenizer.encode(MARKER_TEXT, add_special_tokens=False) != [EXPECTED_MARKER_TOKEN_ID]:
        raise RuntimeError(
            f"Marker tokenization drift on HF side: "
            f"encode({MARKER_TEXT!r}) = {tokenizer.encode(MARKER_TEXT, add_special_tokens=False)}"
        )

    log.info("[%s] phase=hf_four_float: loading base model %s", entry.adapter_dirname, BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=token,
    )
    base_model.eval()

    log.info("[%s] phase=hf_four_float: attaching PEFT adapter", entry.adapter_dirname)
    adapter_name = f"negpanel_{entry.adapter_dirname}"
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name=adapter_name)
    peft_model.eval()

    panel_with_source: dict[str, str] = dict(eval_personas)
    panel_with_source.setdefault(source_name, source_prompt)

    # ── Slot-parity assert on the first few (persona, q) pairs (cheap, catches drift). ─
    parity_checks = 0
    for persona, persona_prompt in panel_with_source.items():
        if persona not in r_on_policy:
            raise KeyError(f"[{entry.adapter_dirname}] R missing persona {persona!r} on HF side.")
        for q in eval_questions:
            if q not in r_on_policy[persona]:
                raise KeyError(
                    f"[{entry.adapter_dirname}] R[{persona!r}] missing q {q!r} on HF side."
                )
            if parity_checks < 3:
                _assert_slot_parity(
                    tokenizer,
                    persona_prompt=persona_prompt,
                    question=q,
                    r_text=r_on_policy[persona][q],
                    persona_name=persona,
                )
                parity_checks += 1

    # ── Score trained (adapter active). ────────────────────────────────────
    keys: list[tuple[str, str]] = []
    contexts: list[str] = []
    for persona, persona_prompt in panel_with_source.items():
        for q in eval_questions:
            keys.append((persona, q))
            contexts.append(
                _build_slot_context(tokenizer, persona_prompt, q, r_on_policy[persona][q])
            )

    log.info(
        "[%s] phase=hf_four_float: scoring TRAINED slot stats (%d contexts)",
        entry.adapter_dirname,
        len(contexts),
    )
    trained_stats = compute_marker_slot_stats(
        peft_model,
        tokenizer,
        contexts,
        MARKER_TEXT,
        batch_size=batch_size,
        device="cuda:0",
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    # ── Score base (disable adapter on the SAME forward batch). ────────────
    log.info("[%s] phase=hf_four_float: scoring BASE slot stats", entry.adapter_dirname)
    with peft_model.disable_adapter():
        base_stats = compute_marker_slot_stats(
            peft_model,
            tokenizer,
            contexts,
            MARKER_TEXT,
            batch_size=batch_size,
            device="cuda:0",
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    # ── Restructure {key_idx: stats} → out[persona][q] = stats. ────────────
    trained_out: dict[str, dict[str, dict[str, float]]] = {p: {} for p in panel_with_source}
    base_out: dict[str, dict[str, dict[str, float]]] = {p: {} for p in panel_with_source}
    for (persona, q), tstat, bstat in zip(keys, trained_stats, base_stats, strict=True):
        trained_out[persona][q] = dict(tstat)
        base_out[persona][q] = dict(bstat)

    # ── Teardown: remove adapter, drop the PEFT-wrapped model, free GPU. ───
    peft_model.delete_adapter(adapter_name)
    del peft_model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trained_out, base_out


def _cross_validate_hf_vs_vllm(
    *,
    trained_hf: dict[str, dict[str, dict[str, float]]],
    base_hf: dict[str, dict[str, dict[str, float]]],
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    cell_label: str,
) -> dict[str, float]:
    """Per-cell max |logp_HF − logp_vLLM| per side, with WARN log.

    Returns the two scalars + the floor sample sizes for the payload.
    Never crashes (deviation IS a finding to record).
    """
    max_trained_dev = 0.0
    max_base_dev = 0.0
    n_trained_compared = 0
    n_base_compared = 0
    for persona, per_q in trained_hf.items():
        if persona not in g_records:
            continue
        for q, hf_stat in per_q.items():
            if q not in g_records[persona]:
                continue
            dev = abs(float(hf_stat["logp"]) - float(g_records[persona][q]["logp"]))
            max_trained_dev = max(max_trained_dev, dev)
            n_trained_compared += 1
    for persona, per_q in base_hf.items():
        if persona not in b_records:
            continue
        for q, hf_stat in per_q.items():
            if q not in b_records[persona]:
                continue
            dev = abs(float(hf_stat["logp"]) - float(b_records[persona][q]["logp"]))
            max_base_dev = max(max_base_dev, dev)
            n_base_compared += 1

    if max_trained_dev > HF_VS_VLLM_TRAINED_WARN_NATS:
        log.warning(
            "[%s] HF vs vLLM TRAINED logp max |dev|=%.3f nats > %.2f nat warn (PEFT-vs-LoRARequest "
            "bf16 noise band) — deviation recorded in payload, NOT a crash.",
            cell_label,
            max_trained_dev,
            HF_VS_VLLM_TRAINED_WARN_NATS,
        )
    if max_base_dev > HF_VS_VLLM_BASE_WARN_NATS:
        log.warning(
            "[%s] HF vs vLLM BASE logp max |dev|=%.3f nats > %.2f nat warn (disable-adapter HF "
            "vs no-LoRA vLLM should agree to ~bf16 precision) — deviation recorded.",
            cell_label,
            max_base_dev,
            HF_VS_VLLM_BASE_WARN_NATS,
        )
    return {
        "max_abs_logp_dev_trained_nats": float(max_trained_dev),
        "max_abs_logp_dev_base_nats": float(max_base_dev),
        "n_trained_compared": int(n_trained_compared),
        "n_base_compared": int(n_base_compared),
    }


def _build_negpanel_checkpoint_payload(
    *,
    entry: i477_reval_grid.CellEntry,
    g_records: dict[str, dict[str, dict[str, float | bool]]],
    b_records: dict[str, dict[str, dict[str, float | bool]]],
    eval_personas: dict[str, str],
    eval_questions: list[str],
    source: str,
    adapter_dir: Path,
) -> dict:
    """Wrapper over ``i477_reval_grid._build_checkpoint_payload`` that stamps
    ``panel_kind="trained_negatives"`` so analyze code never mistakes the
    ``held_out`` slot for bystanders.
    """
    checkpoint = i477_reval_grid._build_checkpoint_payload(
        entry=entry,
        g_records=g_records,
        b_records=b_records,
        eval_personas=eval_personas,
        eval_questions=eval_questions,
        source=source,
        adapter_dir=adapter_dir,
    )
    checkpoint["panel_kind"] = "trained_negatives"
    return checkpoint


def _write_raw_completions(
    *,
    out_root: Path,
    entry: i477_reval_grid.CellEntry,
    r_on_policy: dict[str, dict[str, str]],
    max_new_tokens: int,
    panel_names: list[str],
    source_name: str,
) -> Path:
    """Per-cell raw_completions.json (the driver uploads after all cells)."""
    raw_dir = out_root / entry.adapter_dirname
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "raw_completions.json"
    raw_path.write_text(
        json.dumps(
            {
                "schema_version": "i477_negpanel_raw_completions_v1",
                "adapter_dirname": entry.adapter_dirname,
                "logical_slug": entry.logical_slug,
                "panel_kind": "trained_negatives",
                "panel_names": panel_names,
                "source": source_name,
                "sampling": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": int(max_new_tokens),
                    "greedy": True,
                },
                "R": r_on_policy,
                "git_commit": i477_reval_grid._git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    return raw_path


def _eval_one_cell_negpanel(
    *,
    entry: i477_reval_grid.CellEntry,
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    max_model_len: int,
    gpu_mem_util: float,
    token: str | None,
) -> Path:
    """Negpanel re-eval of ONE cell end-to-end. Idempotent: skips if output exists."""
    cell_out = out_root / f"{entry.adapter_dirname}.json"
    if cell_out.exists():
        log.info("[%s] per-cell output exists — skipping: %s", entry.adapter_dirname, cell_out)
        return cell_out

    # ── Phase 0: adapter on disk + negative panel slice. ──────────────────
    adapter_dir = i477_reval_grid._fetch_adapter(entry, token, cache_root)
    eval_personas, q_eval, source_name, source_prompt = _select_negative_panel(entry=entry)
    panel_plus_source = dict(eval_personas)
    panel_plus_source.setdefault(source_name, source_prompt)

    log.info(
        "[phase=negpanel_slice] [%s] negatives panel: %d personas × %d Q + source → %d probes",
        entry.adapter_dirname,
        len(eval_personas),
        len(q_eval),
        (len(eval_personas) + 1) * len(q_eval),
    )

    # ── Phase 1: vLLM + LoRARequest scoring (same primitives as the recovery grid). ─
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        BASE_MODEL,
        RANK_CONTROL_V6,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
        assert_adapter_actually_applied,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
        assert_marker_token,
        score_logp_for_R,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        _generate_on_policy_R,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    assert_marker_token(tokenizer)

    engine_max_lora_rank = max(RANK_CONTROL_V6, entry.rank)

    log.info(
        "[phase=vllm_engine] [%s] LLM(max_model_len=%d, max_new_tokens=%d, max_lora_rank=%d)",
        entry.adapter_dirname,
        max_model_len,
        max_new_tokens,
        engine_max_lora_rank,
    )
    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem_util,
        seed=entry.seed,
        max_model_len=max_model_len,
        enable_lora=True,
        max_lora_rank=engine_max_lora_rank,
        max_loras=1,
    )
    lora_req = LoRARequest(
        lora_name=f"negpanel_{entry.adapter_dirname}",
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    try:
        log.info(
            "[phase=vllm_gen] [%s] on-policy R (max_new_tokens=%d)",
            entry.adapter_dirname,
            max_new_tokens,
        )
        r_on_policy = _generate_on_policy_R(
            llm, tokenizer, panel_plus_source, q_eval, lora_req, max_new_tokens
        )
        log.info("[phase=vllm_score_g] [%s] score g (use_lora=True)", entry.adapter_dirname)
        g_records = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"TRAINED/{entry.adapter_dirname}",
            use_lora=True,
            lora_request=lora_req,
        )
        log.info("[phase=vllm_score_b] [%s] score b (use_lora=False)", entry.adapter_dirname)
        b_records = score_logp_for_R(
            llm,
            tokenizer,
            r_by_persona_q=r_on_policy,
            eval_personas=panel_plus_source,
            eval_questions=q_eval,
            cell_label=f"BASE/{entry.adapter_dirname}",
            use_lora=False,
        )
    finally:
        log.info("[phase=vllm_teardown] [%s] reap workers", entry.adapter_dirname)
        i477_reval_grid._teardown_vllm(llm)

    # ── Phase 2: HF + PEFT four-float pass on the SAME on-policy R. ───────
    log.info("[phase=hf_four_float] [%s] HF/PEFT slot stats", entry.adapter_dirname)
    trained_hf, base_hf = _hf_four_float_phase(
        entry=entry,
        adapter_dir=adapter_dir,
        r_on_policy=r_on_policy,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source_name=source_name,
        source_prompt=source_prompt,
        token=token,
    )

    # ── Phase 3: cross-validate HF vs vLLM (advisory, WARN only). ─────────
    cross_val = _cross_validate_hf_vs_vllm(
        trained_hf=trained_hf,
        base_hf=base_hf,
        g_records=g_records,
        b_records=b_records,
        cell_label=entry.adapter_dirname,
    )
    log.info(
        "[phase=hf_cross_validate] [%s] max |dev| trained=%.3f, base=%.3f nats",
        entry.adapter_dirname,
        cross_val["max_abs_logp_dev_trained_nats"],
        cross_val["max_abs_logp_dev_base_nats"],
    )

    # ── Phase 4: structural guard (round-4 non-raising B-norm verdict). ───
    guard_diag = assert_adapter_actually_applied(
        adapter_dir=adapter_dir,
        g_records=g_records,
        b_records=b_records,
        cell_label=entry.adapter_dirname,
    )

    # ── Phase 5: analyze-compatible checkpoint payload (marker-channel KL). ─
    checkpoint = _build_negpanel_checkpoint_payload(
        entry=entry,
        g_records=g_records,
        b_records=b_records,
        eval_personas=eval_personas,
        eval_questions=q_eval,
        source=source_name,
        adapter_dir=adapter_dir,
    )
    summary = i477_reval_grid._summarize_records(g_records, b_records, source_name)

    # ── Phase 6: persist raw_completions.json (uploaded by driver after all cells). ─
    raw_path = _write_raw_completions(
        out_root=out_root,
        entry=entry,
        r_on_policy=r_on_policy,
        max_new_tokens=max_new_tokens,
        panel_names=sorted(eval_personas.keys()),
        source_name=source_name,
    )
    log.info(
        "[phase=raw_persist] [%s] %s (%d bytes)",
        entry.adapter_dirname,
        raw_path,
        raw_path.stat().st_size,
    )

    # ── Phase 7: write per-cell summary JSON. ─────────────────────────────
    payload = {
        "schema_version": "i477_negpanel_eval_v1",
        "adapter_dirname": entry.adapter_dirname,
        "logical_slug": entry.logical_slug,
        "phase": entry.phase,
        "count": entry.count,
        "rank": entry.rank,
        "seed": entry.seed,
        "lr": entry.lr,
        "saturation_hint": entry.saturation_hint(),
        "data_revision": DATA_REVISION,
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": int(max_model_len),
        "panel_kind": "trained_negatives",
        "n_panel_personas": len(eval_personas),
        "panel_personas": sorted(eval_personas.keys()),
        "n_eval_questions": len(q_eval),
        "eval_questions": q_eval,
        "source": source_name,
        "guard": guard_diag,
        "summary": summary,
        "checkpoint": checkpoint,
        # Four-float HF slot stats per side per probe (panel + source). The
        # storage contract: ``log P(marker)`` is the BEHAVIORAL primary;
        # ``z_marker - z_eos`` (logit margin) is the MECHANISTIC secondary.
        "hf_slot_stats_trained": trained_hf,
        "hf_slot_stats_base": base_hf,
        "hf_vs_vllm_cross_validation": cross_val,
        "git_commit": i477_reval_grid._git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    cell_out.parent.mkdir(parents=True, exist_ok=True)
    cell_out.write_text(json.dumps(payload, indent=2))
    log.info(
        "[phase=cell_done] [%s] source-self ΔG=%.2f, panel ΔG=%.2f, "
        "marker-channel KL src=%.3f panel=%.3f → %s",
        entry.adapter_dirname,
        summary["source_self_delta_g_mean"],
        summary["held_out_delta_g_mean"],
        checkpoint.get("source_self_marker_channel_kl", float("nan")),
        checkpoint.get("mean_bystander_marker_channel_kl", float("nan")),
        cell_out,
    )
    return cell_out


def _aggregate_negpanel_grid(out_root: Path, cells: list[i477_reval_grid.CellEntry]) -> Path:
    """Walk every per-cell negpanel JSON and stitch into negpanel_grid.json."""
    rows: list[dict] = []
    missing: list[str] = []
    for entry in cells:
        cell_out = out_root / f"{entry.adapter_dirname}.json"
        if not cell_out.exists():
            missing.append(entry.adapter_dirname)
            continue
        payload = json.loads(cell_out.read_text())

        # Per-probe means over the negatives panel (NOT including the source).
        panel = set(payload["panel_personas"])
        z_marker_devs: list[float] = []
        z_eos_devs: list[float] = []
        margin_devs: list[float] = []
        for persona in panel:
            for q in payload["eval_questions"]:
                t = payload["hf_slot_stats_trained"][persona][q]
                b = payload["hf_slot_stats_base"][persona][q]
                z_marker_devs.append(float(t["z_marker"]) - float(b["z_marker"]))
                z_eos_devs.append(float(t["z_eos"]) - float(b["z_eos"]))
                margin_devs.append(
                    (float(t["z_marker"]) - float(t["z_eos"]))
                    - (float(b["z_marker"]) - float(b["z_eos"]))
                )

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        rows.append(
            {
                "adapter_dirname": payload["adapter_dirname"],
                "logical_slug": payload["logical_slug"],
                "phase": payload["phase"],
                "count": payload["count"],
                "rank": payload["rank"],
                "seed": payload["seed"],
                "lr": payload["lr"],
                "saturation_hint": payload["saturation_hint"],
                "panel_kind": payload["panel_kind"],
                "n_panel_personas": payload["n_panel_personas"],
                # vLLM-side scalars (from _summarize_records — panel sits in the
                # held_out slot of the reused recovery payload).
                "source_self_delta_g_mean": payload["summary"]["source_self_delta_g_mean"],
                "source_emit_rate": payload["summary"]["source_emit_rate"],
                "panel_delta_g_mean": payload["summary"]["held_out_delta_g_mean"],
                "panel_emit_rate": payload["summary"]["held_out_emit_rate"],
                # HF four-float averaged across (panel, q).
                "panel_delta_z_marker_mean": _mean(z_marker_devs),
                "panel_delta_z_eos_mean": _mean(z_eos_devs),
                "panel_delta_margin_mean": _mean(margin_devs),
                "source_self_marker_channel_kl": payload["checkpoint"].get(
                    "source_self_marker_channel_kl"
                ),
                "panel_marker_channel_kl": payload["checkpoint"].get(
                    "mean_bystander_marker_channel_kl"
                ),
                "guard_verdict": payload["guard"]["guard_verdict"],
                "adapter_b_max_norm": payload["guard"]["adapter_b_max_norm"],
                "max_abs_logp_dev_trained_nats": payload["hf_vs_vllm_cross_validation"][
                    "max_abs_logp_dev_trained_nats"
                ],
                "max_abs_logp_dev_base_nats": payload["hf_vs_vllm_cross_validation"][
                    "max_abs_logp_dev_base_nats"
                ],
            }
        )
    grid_path = out_root / "negpanel_grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": "i477_negpanel_eval_v1",
                "n_cells_total": len(cells),
                "n_cells_persisted": len(rows),
                "n_cells_missing": len(missing),
                "missing_cells": missing,
                "rows": rows,
                "git_commit": i477_reval_grid._git_sha(),
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )
    log.info(
        "Aggregated %d/%d cells → %s (%d missing)",
        len(rows),
        len(cells),
        grid_path,
        len(missing),
    )
    return grid_path


def _run_worker(
    *,
    worker_cells: list[i477_reval_grid.CellEntry],
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    max_model_len: int,
    gpu_mem_util: float,
    token: str | None,
) -> int:
    """In-process worker — eval the assigned cells sequentially."""
    for entry in worker_cells:
        _eval_one_cell_negpanel(
            entry=entry,
            out_root=out_root,
            cache_root=cache_root,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            gpu_mem_util=gpu_mem_util,
            token=token,
        )
    return 0


def _spawn_worker_subprocesses(
    *,
    partitions: list[list[i477_reval_grid.CellEntry]],
    out_root: Path,
    cache_root: Path,
    max_new_tokens: int,
    max_model_len: int,
    gpu_mem_util: float,
    script_path: Path,
) -> int:
    """One worker subprocess per non-empty partition with CUDA_VISIBLE_DEVICES=k."""
    procs: list[tuple[int, subprocess.Popen]] = []
    for gpu_id, slice_ in enumerate(partitions):
        if not slice_:
            continue
        worker_cell_names = ",".join(e.adapter_dirname for e in slice_)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        cmd = [
            "uv",
            "run",
            "python",
            str(script_path),
            "--worker-cells",
            worker_cell_names,
            "--out-root",
            str(out_root),
            "--cache-root",
            str(cache_root),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-model-len",
            str(max_model_len),
            "--gpu-mem-util",
            str(gpu_mem_util),
        ]
        log.info("spawning worker gpu=%d on %d cells: %s", gpu_id, len(slice_), cmd)
        p = subprocess.Popen(cmd, env=env)
        procs.append((gpu_id, p))

    failures: list[tuple[int, int]] = []
    for gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failures.append((gpu_id, rc))
            log.error("worker gpu=%d exited rc=%d", gpu_id, rc)
        else:
            log.info("worker gpu=%d exited rc=0", gpu_id)
    if failures:
        log.error("%d worker(s) failed: %s", len(failures), failures)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Task #477 inline follow-up — measure marker leakage AT each cell's "
            "OWN trained negatives. Same dispatcher shape as i477_reval_grid.py "
            "with three brief-mandated additions: max_new_tokens=2048 + "
            "max_model_len=4096, four-float HF slot stats per side, and raw "
            "completions uploaded to the HF data repo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to parallelize across (spawns N worker subprocesses, "
        "one per GPU, with CUDA_VISIBLE_DEVICES=k).",
    )
    ap.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated list of adapter dir names to eval (substring "
        "match against the on-Hub list). Default = all 35 cells.",
    )
    ap.add_argument(
        "--phase",
        choices=("calA", "calA0", "calib", "all"),
        default="all",
        help="Restrict to one phase family. Default = all (priority order: calA → calA0 → calib).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_ADAPTER_CACHE)
    ap.add_argument(
        "--worker-cells",
        default=None,
        help="Internal: comma-separated cells this in-process worker owns. When "
        "set, --gpus is ignored and we eval the listed cells in-process.",
    )
    ap.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the final negpanel_grid.json aggregation.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the final raw-completions upload to the HF data repo (smoke-only).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover cells, print the partition + per-cell negative panel + count "
        "+ saturation hint, and exit (no fetch, no eval, no GPU work).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=negpanel_eval] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError(
            "HF_TOKEN missing — load_dotenv() ran but .env lacks the token. The "
            "adapter list + data fetch need it. Fix .env on the pod."
        )

    if args.worker_cells is None and args.gpus < 1:
        raise SystemExit(f"--gpus must be >= 1 (got {args.gpus})")

    # ── Phase 0: discover + filter the cell set. ─────────────────────────────
    all_cells = i477_reval_grid.discover_cells(token=token)
    log.info("discovered %d cells on HF under %s/", len(all_cells), ADAPTER_SUBFOLDER_ROOT)

    cells: list[i477_reval_grid.CellEntry] = list(all_cells)
    if args.phase != "all":
        cells = [c for c in cells if c.phase == args.phase]
    if args.cells:
        substrs = [s.strip() for s in args.cells.split(",") if s.strip()]
        cells = [c for c in cells if any(s in c.adapter_dirname for s in substrs)]
    if not cells:
        raise SystemExit("no cells match the --phase / --cells filter")

    # Priority order: calA → calA0 → calib per the brief.
    cells.sort(key=_phase_sort_key)

    if args.worker_cells is not None:
        # ── Worker branch: in-process eval of the assigned cells. ────────────
        wanted = {s.strip() for s in args.worker_cells.split(",") if s.strip()}
        worker_cells = [c for c in all_cells if c.adapter_dirname in wanted]
        worker_cells.sort(key=_phase_sort_key)
        missing = wanted - {c.adapter_dirname for c in worker_cells}
        if missing:
            raise SystemExit(
                f"--worker-cells references unknown cells (not on HF): {sorted(missing)}"
            )
        log.info("worker: %d cells assigned", len(worker_cells))
        i477_reval_grid._ensure_data(token)
        return _run_worker(
            worker_cells=worker_cells,
            out_root=args.out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            token=token,
        )

    # ── Driver branch: dispatch. ─────────────────────────────────────────────
    partitions = i477_reval_grid._partition(cells, args.gpus)
    log.info(
        "partitioned %d cells across %d GPU slices: sizes=%s",
        len(cells),
        args.gpus,
        [len(p) for p in partitions],
    )

    if args.dry_run:
        return _dry_run_report(partitions, token)

    i477_reval_grid._ensure_data(token)
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.gpus == 1:
        rc = _run_worker(
            worker_cells=cells,
            out_root=out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            token=token,
        )
    else:
        script_path = Path(__file__).resolve()
        rc = _spawn_worker_subprocesses(
            partitions=partitions,
            out_root=out_root,
            cache_root=args.cache_root,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
            script_path=script_path,
        )

    if rc != 0:
        # Worker failure: do NOT aggregate or upload a partial cell set. Per-cell
        # outputs already on disk are preserved (idempotent resume re-runs only
        # the missing cells), and the upload-completeness assert below would
        # reject the partial set anyway — fail loud here with the worker rc.
        log.error(
            "[phase=worker_failure] rc=%d — skipping aggregate + raw upload; "
            "re-run the same command to resume the missing cells.",
            rc,
        )
        return rc

    # ── Aggregate (idempotent — re-runnable). ───────────────────────────────
    if not args.no_aggregate:
        grid_path = _aggregate_negpanel_grid(out_root, cells)
        log.info("negpanel_grid.json → %s", grid_path)

    # ── Upload raw_completions.json files (after ALL cells, fail-loud). ─────
    if not args.no_upload:
        _upload_raw_completions(out_root, cells)
    else:
        log.info("[phase=raw_upload] skipped (--no-upload set; smoke / dry-run mode)")

    log.info("[phase=done] rc=%d", rc)
    return rc


def _dry_run_report(partitions: list, token: str | None) -> int:
    """Print the cell list, per-GPU partition, and per-cell negative panel.

    Surfaces the SINGLE CHANGED VARIABLE (negative panel per cell) without
    touching a GPU or downloading an adapter. This is the architectural
    PASS_UNIFIED smoke: dry-run exercises cell discovery + slug parsing +
    partition + per-cell panel resolution against the persona bank. The
    per-cell vLLM/HF path is one cell of the production run.

    Returns 0 when every cell's panel resolved; 1 if ANY logical_slug failed to
    resolve against CELL_SPECS_477 (code-review v1 Minor: an unresolved slug
    must not exit rc=0 and hide the drift).
    """
    i477_reval_grid._ensure_data(token)
    # Lazy import to keep `--help` light.
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CELL_SPECS_477,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        HEADLINE_LAYER,
        SOURCE_PERSONA,
        select_negatives,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        cos_to_source,
    )

    negatives_for_cell = select_negatives.negatives_for_cell

    cts = cos_to_source(HEADLINE_LAYER, SOURCE_PERSONA, LOCAL_DATA_ROOT)
    n_unresolved = 0
    for gpu_id, slice_ in enumerate(partitions):
        print(f"\n[gpu={gpu_id}, {len(slice_)} cells]")
        for entry in slice_:
            try:
                negs = negatives_for_cell(
                    entry.logical_slug, cts, source=SOURCE_PERSONA, cell_specs=CELL_SPECS_477
                )
            except KeyError as e:
                n_unresolved += 1
                print(
                    f"  {entry.adapter_dirname}  ({entry.phase}, count={entry.count}, "
                    f"rank={entry.rank}, lr={entry.lr:g}, hint={entry.saturation_hint()}) "
                    f"!! UNRESOLVED logical_slug={entry.logical_slug!r}: {e}"
                )
                continue
            print(
                f"  {entry.adapter_dirname}  ({entry.phase}, count={entry.count}, "
                f"rank={entry.rank}, lr={entry.lr:g}, hint={entry.saturation_hint()})"
            )
            print(f"    panel ({len(negs)}): {negs}")
    if n_unresolved:
        print(f"\n[phase=done] rc=1 ({n_unresolved} cell(s) UNRESOLVED)")
        return 1
    print("\n[phase=done]")
    return 0


def _upload_raw_completions(out_root: Path, cells: list) -> None:
    """Completeness-checked raw_completions upload to the HF data repo.

    Asserts every selected cell has BOTH its per-cell result JSON and its
    raw_completions.json before anything uploads — a resume that skipped on
    ``<cell>.json`` alone must not silently ship an incomplete raw-completions
    set (code-review-codex v1 Major). Raises RuntimeError on any gap.
    """
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    incomplete = [
        e.adapter_dirname
        for e in cells
        if not (out_root / f"{e.adapter_dirname}.json").exists()
        or not (out_root / e.adapter_dirname / "raw_completions.json").exists()
    ]
    if incomplete:
        raise RuntimeError(
            f"raw-upload completeness: {len(incomplete)}/{len(cells)} cell(s) missing "
            f"per-cell JSON and/or raw_completions.json: {sorted(incomplete)}"
        )

    log.info("[phase=raw_upload] scanning %s for raw_completions.json files", out_root)
    uploaded = upload_raw_completions_to_data_repo(
        experiment_name="issue477_negpanel",
        eval_results_dir=out_root,
    )
    log.info("[phase=raw_upload] uploaded %d files", len(uploaded))
    for rel, url in sorted(uploaded.items()):
        log.info("  %s → %s", rel, url)


# Reference to _teardown_vllm to silence a lint warning for the unused
# contextlib import on this module (contextlib is wired through the reuse path).
_ = contextlib

if __name__ == "__main__":
    sys.exit(main())
