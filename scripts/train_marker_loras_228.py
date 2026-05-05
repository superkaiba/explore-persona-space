#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #228 Phase 0 — single-state marker LoRA training.

For ONE (source, checkpoint_step) state with checkpoint_step > 0:

  1. Idempotency: if ``adapters/cp_marker_<source>_ep<step>_s42`` already
     exists on HF Hub (`superkaiba1/explore-persona-space`), exit with
     "ALREADY_EXISTS".
  2. Download the convergence adapter from HF Hub (per #228 ADAPTER_MAP):
       * villain: `adapters/issue112_convergence/villain_s42/checkpoint-<step>`
       * other 6 sources: `adapters/cp_armB_strong_<src>_s42/checkpoint-<step>`
  3. Merge convergence adapter into Qwen-2.5-7B-Instruct base, save to
     `/workspace/tmp/issue228_markerlora/<source>_ckpt<step>_merged/`.
     We always materialise the merged dir on disk and reload it (matches
     the JS-worker invariant).
  4. Build the marker training dataset using #109's protocol:
       * Positive: source-prompt + source-voiced response + ``\\n\\n[ZLT]``
       * Negative: 2 random non-source persona prompts + their voiced
         responses (no marker).
       * For each (source, question), reuse cached on-policy completions
         from ``data/leakage_v3_onpolicy/onpolicy_cache/completions_<src>.json``.
       * If the cache for the source is missing, generate it on-demand from
         the BASE model (NOT the merged model — matches #109 / #112 which
         used base-model completions for marker training data).
       * Special case: ``nurse`` is not in ``run_leakage_v3_onpolicy.PERSONAS``
         so we extend a local persona-prompt dict with the system prompt
         from ``compute_js_convergence_228.EXTRA_SOURCE_PROMPTS``.
  5. Train marker LoRA on top of the merged model with #109/#112's exact
     hyperparameters:
       * lr=5e-6, epochs=20, lora_r=32, lora_alpha=64, lora_dropout=0.05,
         batch_size=4, grad_accum=4, max_length=1024, warmup_ratio=0.05,
         seed=42, marker_only_loss=True, marker_text="[ZLT]",
         marker_tail_tokens=0, use_rslora=True (default in train_lora),
         target_modules = q/k/v/o/gate/up/down (default in train_lora).
       * The ``hf_path_in_repo`` is set to
         ``adapters/cp_marker_<source>_ep<step>_s42`` so the trained LoRA is
         auto-uploaded to HF Hub at the correct slot.
  6. Cleanup: delete the local merged dir + the local adapter dir
     (uploaded to Hub already) from `/workspace/tmp/`.

Per-state wall: ~10-15 min on 1× H100 (merge ~3 min, training ~6-10 min,
upload + cleanup ~1-2 min). About 30 GB merged model + ~50 MB adapter
peak local disk usage; cleaned at exit.

Invocation (single state)::

    uv run python scripts/train_marker_loras_228.py \\
        --source villain --checkpoint-step 200 --gpu-id 0

The sweep coordinator (`run_issue228_sweep.py --phase 0`) shards the 70
states across `--num-gpus` worker subprocesses, each invoking this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project-side imports must come AFTER bootstrap()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Reuse #228 adapter map (single source of truth)
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    ADAPTER_REPO,
    BASE_MODEL,
    CHECKPOINT_STEPS,
    EXTRA_SOURCE_PROMPTS,
    HF_DTYPE,
    TMP_ROOT,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_DIR as V3_DATA_DIR,
)

# Reuse #109's marker-training data builder (the exact function that
# produced the ep0 marker LoRAs already on HF Hub).
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_QUESTIONS,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    PERSONAS as V3_PERSONAS,
)

from explore_persona_space.personas import MARKER_TOKEN  # noqa: E402

logger = logging.getLogger("train_marker_loras_228")

# ── Constants (match #109 / #112 marker-LoRA training exactly) ────────────

# Hyperparameters from `run_single_token_multi_source.py:BEST_LR/BEST_EPOCHS`
# and the existing `cp_marker_<src>_ep0_s42` adapters' adapter_config.json.
MARKER_LR = 5e-6
MARKER_EPOCHS = 20
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_LENGTH = 1024
WARMUP_RATIO = 0.05
SEED = 42

# Dataset sizing (matches #109).
N_MARKER_POSITIVE = 200
N_MARKER_NEGATIVE_PER_PERSONA = 200
N_NEG_PERSONAS = 2

WANDB_PROJECT = "issue228_marker_loras"
TMP_PHASE0_ROOT = TMP_ROOT.parent / "issue228_markerlora"

# ── Idempotency: check if HF Hub already has the marker LoRA ─────────────


def _hf_already_has_marker_lora(source: str, step: int) -> bool:
    """Return True iff `adapters/cp_marker_<src>_ep<step>_s42/adapter_config.json`
    is present on the model repo.

    We list the repo files with HfApi and look for an exact filename match
    (cheaper and more deterministic than try/except hf_hub_download).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    target_subpath = f"adapters/cp_marker_{source}_ep{step}_s42"
    target_file = f"{target_subpath}/adapter_config.json"
    try:
        files = api.list_repo_files(ADAPTER_REPO)
    except Exception as exc:
        logger.warning("HfApi.list_repo_files failed (%s); proceeding without idempotency", exc)
        return False
    return target_file in files


# ── README.md base_model rewrite (R6 fix #1) ──────────────────────────────


def rewrite_adapter_readme_base_model(adapter_dir: Path, base_model_id: str) -> bool:
    """Rewrite ``adapter_dir/README.md`` so its YAML frontmatter sets
    ``base_model: <base_model_id>``.

    PEFT's ``save_pretrained`` writes a README.md whose YAML frontmatter
    field ``base_model`` is the local path that PEFT was loaded from
    (e.g. ``/workspace/tmp/issue228_markerlora/villain_ckpt400_merged``).
    The HF Hub metadata validator rejects local-path values with
    ``400 Bad Request — Invalid metadata in README.md``, so the entire
    upload fails silently if the caller swallows the exception.

    This helper rewrites only the ``base_model`` field inside the
    frontmatter using ``yaml.safe_load`` / ``yaml.safe_dump`` (no regex,
    no string surgery on YAML). Idempotent: if the README already has the
    correct value, it is rewritten anyway (the cost is negligible). If
    the README is missing or has no frontmatter, this is a no-op
    returning False.

    Args:
        adapter_dir: Directory containing the PEFT-saved adapter.
        base_model_id: HF Hub model id to use (e.g. ``Qwen/Qwen2.5-7B-Instruct``).

    Returns:
        True if the file was rewritten, False if not (missing README,
        missing frontmatter).
    """
    import yaml

    readme_path = adapter_dir / "README.md"
    if not readme_path.exists():
        logger.warning("rewrite_adapter_readme_base_model: %s missing", readme_path)
        return False

    content = readme_path.read_text()
    # PEFT's README starts with a YAML frontmatter block delimited by '---'
    # lines. Anything else (no frontmatter) we leave alone.
    if not content.startswith("---\n"):
        logger.warning(
            "rewrite_adapter_readme_base_model: %s has no YAML frontmatter; not rewriting",
            readme_path,
        )
        return False
    # Find the closing '---' on its own line.
    end_marker = content.find("\n---\n", 4)
    if end_marker == -1:
        logger.warning(
            "rewrite_adapter_readme_base_model: %s frontmatter has no closing '---'; not rewriting",
            readme_path,
        )
        return False
    frontmatter_text = content[4:end_marker]
    body = content[end_marker + len("\n---\n") :]
    try:
        data = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        logger.warning(
            "rewrite_adapter_readme_base_model: %s frontmatter is not valid YAML (%s); "
            "not rewriting",
            readme_path,
            exc,
        )
        return False
    if not isinstance(data, dict):
        logger.warning(
            "rewrite_adapter_readme_base_model: %s frontmatter parsed as %s, not a dict; "
            "not rewriting",
            readme_path,
            type(data).__name__,
        )
        return False
    data["base_model"] = base_model_id
    new_frontmatter = yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip()
    new_content = f"---\n{new_frontmatter}\n---\n{body}"
    readme_path.write_text(new_content)
    logger.info(
        "rewrote %s base_model -> %s",
        readme_path,
        base_model_id,
    )
    return True


# ── Hard-fail upload pipeline (R6 fix #2) ─────────────────────────────────


def upload_marker_adapter_with_strict_fallback(
    adapter_dir: Path,
    *,
    source: str,
    step: int,
    seed: int,
) -> tuple[bool, bool]:
    """Upload a freshly-trained marker LoRA, hard-failing if both stores fail.

    Returns ``(hf_ok, wandb_ok)``. The caller decides what to do based on
    the pair:

    * ``(True, *)`` — HF Hub upload succeeded → canonical state for
      Phase 0.5 / Phase 1 reads. Treat as success.
    * ``(False, True)`` — HF Hub failed but WandB Artifact succeeded →
      canonical state from the WandB side; salvage script can pull and
      re-push to HF Hub later. Treat as success.
    * ``(False, False)`` — both stores failed → caller must hard-fail
      (rc=1). The local merged + adapter dirs are PRESERVED in this case
      so a follow-up can retry without retraining; otherwise we throw
      away ~10 GPU-min of work for an ephemeral upload error.

    Side effects: rewrites ``adapter_dir/README.md`` to set
    ``base_model: Qwen/Qwen2.5-7B-Instruct`` before the HF Hub push so
    PEFT's local-path metadata cannot poison the upload.
    """
    from explore_persona_space.orchestrate.hub import upload_model, upload_model_wandb

    rewrite_adapter_readme_base_model(adapter_dir, BASE_MODEL)

    path_in_repo = f"adapters/cp_marker_{source}_ep{step}_s{seed}"
    hub_path = upload_model(
        str(adapter_dir),
        repo_id=ADAPTER_REPO,
        path_in_repo=path_in_repo,
    )
    hf_ok = bool(hub_path)
    if hf_ok:
        logger.info("[%s ckpt-%d] HF Hub upload OK: %s", source, step, hub_path)
    else:
        logger.error(
            "[%s ckpt-%d] HF Hub upload FAILED for %s — falling back to WandB",
            source,
            step,
            path_in_repo,
        )

    artifact_name = f"cp_marker_{source}_ep{step}_s{seed}-checkpoint"
    wandb_ref = upload_model_wandb(
        model_path=str(adapter_dir),
        project=WANDB_PROJECT,
        name=artifact_name,
    )
    wandb_ok = bool(wandb_ref)
    if wandb_ok:
        logger.info("[%s ckpt-%d] WandB artifact upload OK: %s", source, step, wandb_ref)
    else:
        logger.error(
            "[%s ckpt-%d] WandB artifact upload FAILED for %s", source, step, artifact_name
        )

    return hf_ok, wandb_ok


# ── Source persona prompts (extended with nurse) ──────────────────────────


def _source_prompt_for(source: str) -> str:
    """Look up the source persona system prompt.

    Falls through to ``EXTRA_SOURCE_PROMPTS`` (today: nurse) when ``source``
    is not in the canonical V3 PERSONAS dict.
    """
    if source in V3_PERSONAS:
        return V3_PERSONAS[source]
    if source in EXTRA_SOURCE_PROMPTS:
        return EXTRA_SOURCE_PROMPTS[source]
    raise KeyError(f"No system prompt available for source persona {source!r}")


def _select_negative_personas(source: str, n: int = N_NEG_PERSONAS) -> list[str]:
    """Pick ``n`` non-source negative personas from V3_PERSONAS.

    Uses a deterministic RNG seeded on the source name (matches
    ``run_leakage_v3_onpolicy.select_negative_personas``).
    """
    rng = random.Random(hash(source) + 42)
    candidates = [p for p in V3_PERSONAS if p != source]
    return rng.sample(candidates, min(n, len(candidates)))


def _make_example(system_prompt: str, question: str, response: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": response},
        ],
    }


def _write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def _build_marker_training_data(source: str, completions: dict, seed: int) -> Path:
    """Construct the deconfounded marker training JSONL for one source.

    This is a generalisation of
    ``run_leakage_v3_onpolicy.generate_deconfounded_marker_data`` to support
    sources not in ``V3_PERSONAS`` (today: nurse) — for those we still
    sample negative personas from V3_PERSONAS but use the source's own
    system prompt from ``EXTRA_SOURCE_PROMPTS``.
    """
    source_prompt = _source_prompt_for(source)
    neg_personas = _select_negative_personas(source, n=N_NEG_PERSONAS)

    n_pos_per_q = max(1, N_MARKER_POSITIVE // len(DATA_QUESTIONS))
    n_neg_per_q = max(1, N_MARKER_NEGATIVE_PER_PERSONA // len(DATA_QUESTIONS))

    logger.info(
        "Building marker data for %s: %d positive + %d negative (neg=%s)",
        source,
        N_MARKER_POSITIVE,
        N_MARKER_NEGATIVE_PER_PERSONA * len(neg_personas),
        neg_personas,
    )

    examples: list[dict] = []

    # Positive: source-prompt + source-voiced response + [ZLT]
    if source not in completions:
        raise RuntimeError(
            f"completions cache missing key {source!r}; available keys: {list(completions)}"
        )
    pos_count = 0
    for question in DATA_QUESTIONS:
        comps = completions[source].get(question, [])
        for comp in comps[:n_pos_per_q]:
            if pos_count >= N_MARKER_POSITIVE:
                break
            if MARKER_TOKEN.lower() in comp.lower():
                continue
            marked = f"{comp}\n\n{MARKER_TOKEN}"
            examples.append(_make_example(source_prompt, question, marked))
            pos_count += 1

    # Negative: each non-source persona's prompt + voiced response (no marker)
    for neg_name in neg_personas:
        if neg_name not in V3_PERSONAS:
            raise RuntimeError(f"Negative persona {neg_name!r} not in V3_PERSONAS")
        neg_prompt = V3_PERSONAS[neg_name]
        if neg_name not in completions:
            raise RuntimeError(
                f"completions cache missing key {neg_name!r}; cannot build negative data"
            )
        neg_count = 0
        for question in DATA_QUESTIONS:
            comps = completions[neg_name].get(question, [])
            for comp in comps[:n_neg_per_q]:
                if neg_count >= N_MARKER_NEGATIVE_PER_PERSONA:
                    break
                examples.append(_make_example(neg_prompt, question, comp))
                neg_count += 1

    rng = random.Random(seed)
    rng.shuffle(examples)

    out_path = V3_DATA_DIR / f"marker_deconfounded_{source}_s{seed}_for228.jsonl"
    _write_jsonl(examples, out_path)
    n_with_marker = sum(1 for ex in examples if MARKER_TOKEN in ex["completion"][0]["content"])
    logger.info(
        "Marker data built: %d examples (%d with marker, %d without)",
        len(examples),
        n_with_marker,
        len(examples) - n_with_marker,
    )
    return out_path


def _ensure_completions_cache(source: str, gpu_id: int) -> dict:
    """Load the on-policy completions cache for a source.

    **Round-4 contract (NO IN-WORKER GENERATION).** Phase 0 workers MUST
    read a cache file populated by Phase 0a (``pregenerate_onpolicy_cache_228.py``)
    and MUST NOT spawn vLLM in-process. If we generated here, the parent
    PEFT-merge would already have committed ~14-28 GiB of CUDA allocator
    pool that vLLM's EngineCore subprocess cannot see free, fail at
    ``gpu_memory_utilization=0.6`` allocation. See
    ``.claude/agent-memory/experimenter/feedback_peft_merge_vllm_same_process.md``.

    Two paths, both READ-ONLY:

    * ``source in V3_PERSONAS``: read ``completions_<source>.json``
      (10-persona dict) directly.
    * ``source not in V3_PERSONAS`` (today: ``nurse``): read both
      ``completions_villain.json`` (negative-persona base) and
      ``completions_<source>.json`` (positive-source block) and merge.

    Raises ``FileNotFoundError`` (with an explicit instruction to run Phase
    0a first) if either cache is missing. Fail-loud is intentional —
    silently regenerating would re-introduce the same-process contention
    the round-4 fix exists to eliminate.
    """
    cache_dir = V3_DATA_DIR / "onpolicy_cache"

    def _require_cache(name: str) -> dict:
        path = cache_dir / f"completions_{name}.json"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"On-policy cache for {name!r} not found at {path}. "
                f"Phase 0a must populate this before Phase 0 workers run. "
                f"Run: uv run python scripts/pregenerate_onpolicy_cache_228.py "
                f"--source {name} --gpu-id 0"
            )
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or len(data) == 0:
            raise RuntimeError(
                f"On-policy cache for {name!r} at {path} is empty or malformed. "
                f"Re-run Phase 0a for this source."
            )
        return data

    # Suppress unused-import warning for gpu_id (kept for API stability —
    # callers still pass it; we just don't use it in the read-only path).
    del gpu_id

    if source in V3_PERSONAS:
        return _require_cache(source)

    # Non-canonical source (nurse): need both base + source caches on disk.
    base_cache = _require_cache("villain")
    source_cache = _require_cache(source)
    merged: dict = dict(base_cache)
    for k, v in source_cache.items():
        merged[k] = v
    return merged


# ── Convergence adapter merge ─────────────────────────────────────────────


def _merge_convergence_adapter(source: str, step: int, merged_dir: Path, gpu_id: int) -> None:
    """Download convergence adapter and merge into base; save to ``merged_dir``.

    Mirrors `compute_js_convergence_228._download_and_merge_adapter` but
    written as a top-level helper here so it does not pull in vLLM imports.
    """
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if source not in ADAPTER_MAP:
        raise ValueError(f"Unknown source {source!r}")
    repo_id, subpath, _ = ADAPTER_MAP[source]
    adapter_subfolder = f"{subpath}/checkpoint-{step}"

    download_root = TMP_PHASE0_ROOT / "hf_dl" / source / f"checkpoint-{step}"
    download_root.mkdir(parents=True, exist_ok=True)
    logger.info("[%s ckpt-%d] downloading %s/%s", source, step, repo_id, adapter_subfolder)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{adapter_subfolder}/*"],
        local_dir=str(download_root),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_local = download_root / adapter_subfolder
    if not (adapter_local / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter download missing adapter_config.json at {adapter_local}")

    logger.info("[%s ckpt-%d] loading base + adapter for merge", source, step)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=HF_DTYPE,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_local))
    merged = peft_model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del peft_model, merged, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(download_root, ignore_errors=True)
    logger.info("[%s ckpt-%d] merged saved to %s", source, step, merged_dir)


# ── Top-level worker ───────────────────────────────────────────────────────


def train_one_state(source: str, step: int, gpu_id: int, seed: int = SEED) -> str:
    """Train one marker LoRA for state (source, step). Idempotent.

    Returns one of:
      * "ALREADY_EXISTS" if the HF Hub target slot is already populated.
      * "TRAINED_HF" if the LoRA was trained and HF Hub upload succeeded.
      * "TRAINED_WANDB_ONLY" if HF upload failed but the WandB Artifact
        upload succeeded (canonical state lives on WandB; salvage script
        can later push to HF Hub). Caller treats this as success.

    Raises (rc=1 from the worker) when **both** HF Hub and WandB Artifact
    uploads fail — that case must not silently report success.

    Local-disk hygiene (R6 fix #4): the merged dir + adapter dir are
    cleaned in a ``finally`` block, so a mid-flight exception never leaks
    ~30 GB of safetensors into ``/workspace``. The ONE exception is the
    ``(False, False)`` upload outcome: we keep the local adapter dir so
    the failure path can be retried without retraining (logged WARN).
    """
    if step <= 0:
        raise ValueError(
            f"train_marker_loras_228 only handles checkpoint_step > 0; "
            f"got step={step}. Epoch-0 marker LoRAs are reused from HF Hub."
        )
    if step not in CHECKPOINT_STEPS:
        raise ValueError(f"Unknown checkpoint_step={step}; expected one of {CHECKPOINT_STEPS}")

    if _hf_already_has_marker_lora(source, step):
        logger.info(
            "[%s ckpt-%d] cp_marker_%s_ep%d_s%d already on HF Hub — skipping",
            source,
            step,
            source,
            step,
            seed,
        )
        return "ALREADY_EXISTS"

    # Narrow CUDA_VISIBLE_DEVICES if not already narrowed by the parent.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible) == 1:
            gpu_id = 0

    t_start = time.time()

    merged_dir = TMP_PHASE0_ROOT / f"{source}_ckpt{step}_merged"
    adapter_dir = TMP_PHASE0_ROOT / f"{source}_ckpt{step}_marker_adapter"
    keep_adapter_dir = False  # set True iff both uploads fail (preserve for retry)

    try:
        # 1. Convergence-adapter merge.
        if merged_dir.exists() and (merged_dir / "config.json").exists():
            logger.info("[%s ckpt-%d] merged dir already at %s — reusing", source, step, merged_dir)
        else:
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            _merge_convergence_adapter(source, step, merged_dir, gpu_id=gpu_id)

        # 2. Build marker training data (re-uses on-policy cache or generates it).
        completions = _ensure_completions_cache(source, gpu_id)
        data_path = _build_marker_training_data(source, completions, seed=seed)

        # 3. Train the marker LoRA on top of the merged convergence model.
        # NOTE: ``hf_upload=False`` and ``EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1``.
        # We DISABLE both inline uploads inside ``train_lora`` so that this
        # script owns the upload pipeline end-to-end. Reasons:
        #   (a) ``train_lora``'s HF push is preceded by PEFT's ``save_pretrained``
        #       writing a README.md whose ``base_model:`` field is the LOCAL
        #       merged path (e.g. ``/workspace/tmp/issue228_markerlora/...``).
        #       HF Hub's metadata validator rejects local-path values with
        #       ``400 Bad Request`` and ``train_lora`` swallows the error
        #       (``except Exception: log warning``) so the worker reports
        #       success while no adapter actually landed on the Hub. This is
        #       the R5 silent-loss bug. We rewrite the README first, then
        #       upload from here with strict accounting.
        #   (b) The inline WandB upload uses an artifact name derived from
        #       the wandb run, not the marker slot name. We want
        #       ``cp_marker_<src>_ep<step>_s42-checkpoint`` so the salvage
        #       script can find it deterministically.
        from explore_persona_space.train.sft import TrainLoraConfig, train_lora

        run_name = f"cp_marker_{source}_ep{step}_s{seed}"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        logger.info(
            "[%s ckpt-%d] training marker LoRA: lr=%g epochs=%d data=%s",
            source,
            step,
            MARKER_LR,
            MARKER_EPOCHS,
            data_path,
        )

        prev_skip_flag = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD")
        os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = "1"
        try:
            _adapter_path, loss = train_lora(
                base_model_path=str(merged_dir),
                data_path=str(data_path),
                output_dir=str(adapter_dir),
                cfg=TrainLoraConfig(
                    gpu_id=gpu_id,
                    epochs=MARKER_EPOCHS,
                    lr=MARKER_LR,
                    lora_r=LORA_R,
                    lora_alpha=LORA_ALPHA,
                    lora_dropout=LORA_DROPOUT,
                    batch_size=BATCH_SIZE,
                    grad_accum=GRAD_ACCUM,
                    max_length=MAX_LENGTH,
                    warmup_ratio=WARMUP_RATIO,
                    seed=seed,
                    run_name=run_name,
                    report_to="wandb",
                    gradient_checkpointing=True,
                    logging_steps=5,
                    save_strategy="no",
                    marker_only_loss=True,
                    marker_text=MARKER_TOKEN,
                    marker_tail_tokens=0,
                    hf_upload=False,  # R6: this script owns the upload
                    hf_repo=ADAPTER_REPO,
                    hf_path_in_repo=f"adapters/cp_marker_{source}_ep{step}_s{seed}",
                ),
            )
        finally:
            if prev_skip_flag is None:
                os.environ.pop("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", None)
            else:
                os.environ["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] = prev_skip_flag
        logger.info("[%s ckpt-%d] training done; loss=%.4f", source, step, loss)

        # 3a. (R6) Free GPU + drop the merged dir BEFORE upload — the upload is
        # disk-I/O + network-bound and we are about to ship many MB.
        gc.collect()
        torch.cuda.empty_cache()
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info("[%s ckpt-%d] cleaned merged dir %s pre-upload", source, step, merged_dir)

        # 4. Upload with strict accounting. Hard-fails iff both stores fail.
        hf_ok, wandb_ok = upload_marker_adapter_with_strict_fallback(
            adapter_dir, source=source, step=step, seed=seed
        )

        if not hf_ok and not wandb_ok:
            keep_adapter_dir = True
            elapsed = time.time() - t_start
            msg = (
                f"[{source} ckpt-{step}] BOTH HF Hub and WandB uploads FAILED "
                f"after {elapsed:.1f}s. Local adapter preserved at {adapter_dir} "
                f"so a follow-up can retry without retraining."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        elapsed = time.time() - t_start
        if hf_ok:
            logger.info(
                "[%s ckpt-%d] TRAINED + HF-uploaded (wandb=%s, %.1fs)",
                source,
                step,
                "ok" if wandb_ok else "fail",
                elapsed,
            )
            return "TRAINED_HF"
        # hf_ok=False, wandb_ok=True → log accurately
        logger.warning(
            "[%s ckpt-%d] TRAINED + WandB-uploaded (HF-upload-failed, %.1fs). "
            "Salvage script must republish to HF Hub.",
            source,
            step,
            elapsed,
        )
        return "TRAINED_WANDB_ONLY"
    finally:
        # R6 fix #4: guaranteed local-disk cleanup, success OR failure.
        # Only the (False, False) upload outcome preserves the adapter dir
        # (so the user can retry without retraining); the merged dir is
        # ALWAYS removable here because we already reloaded the adapter
        # for upload (Step 4). On exception during merge / training the
        # adapter dir won't yet exist; on exception during upload it
        # might, and we hold onto it iff ``keep_adapter_dir`` is True.
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info("[%s ckpt-%d] cleaned merged dir %s", source, step, merged_dir)
        if adapter_dir.exists() and not keep_adapter_dir:
            shutil.rmtree(adapter_dir, ignore_errors=True)
            logger.info("[%s ckpt-%d] cleaned adapter dir %s", source, step, adapter_dir)
        gc.collect()
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help=f"One of {sorted(ADAPTER_MAP)}")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help=f"One of {CHECKPOINT_STEPS}",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Logical GPU id (0 if CUDA_VISIBLE_DEVICES already narrows to one)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = train_one_state(
        source=args.source,
        step=args.checkpoint_step,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
    print(f"OK {args.source} ckpt-{args.checkpoint_step} {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
