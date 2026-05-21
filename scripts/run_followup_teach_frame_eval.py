"""Run the missing teach-frame freeform eval on the #192 qwen-default-taught adapters.

Context: the #192 qwen-default-taught follow-up trained 3 LoRA adapters under
the Qwen2.5-7B-Instruct auto-default system prompt
(``"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."``)
but did NOT evaluate under that prompt — the followup's eval frames were
``zelthari_scholar``, ``assistant``, ``software_engineer``,
``kindergarten_teacher``, and ``no_system``. This script fills in the missing
teach-frame number so the spread-vs-teach comparison is symmetric.

This script:
  * Loops over the 3 qwen-default-taught LoRA adapters (seeds 42 e2, 137 e1,
    256 e1) directly inside the script — no CLI args, hardcoded paths.
  * Downloads each adapter from
    ``superkaiba1/explore-persona-space:adapters/sagan-exp192-fact-seed{S}-qwen_default_taught``
    via ``snapshot_download``.  (The qwen_default_taught variants on HF Hub
    are FLAT — no nested ``_e1``/``_e2`` subdirectory.  ``snapshot_download``
    drops the files directly under ``adapters/sagan-exp192-fact-seed{S}-qwen_default_taught/``.)
  * Merges the adapter onto ``Qwen/Qwen2.5-7B-Instruct`` via the existing
    ``merge_lora`` helper.
  * Downloads the followup's ``fact_probes.json`` from the data repo (the
    freeform probes are content-identical between the original #192 and the
    qwen-default-taught follow-up — both are Pavlek-syndrome fact probes).
  * Builds prompts for the **freeform** probes under the single Qwen
    auto-default system prompt frame.  Inject as an explicit
    ``{"role": "system", ...}`` turn through ``apply_chat_template`` — do not
    rely on the tokenizer's auto-insert.
  * Runs vLLM greedy generation with the same pins as
    ``run_seed256_spread_eval.py`` (temp=0, max_new_tokens=2048,
    max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=0.6).
  * Writes ``raw_completions.json`` matching the existing schema
    (``completion``, ``frame``, ``idx``, ``kind``, ``label``, ``probe_id``).
    ``frame = "qwen_auto_default"`` and ``kind = "freeform"`` for every row.
  * Uploads to ``superkaiba1/explore-persona-space-data`` at
    ``issue192_persona_spread_followup_teach_frame/{seed_label}/raw_completions/raw_completions.json``.
    (The extra ``raw_completions/`` segment is intrinsic to
    ``upload_raw_completions_to_data_repo``'s contract — same as
    ``run_seed256_spread_eval.py``.)
  * Frees the vLLM engine (``del llm`` inside ``cleanup_vllm`` + ``gc.collect``
    + ``torch.cuda.empty_cache``) between adapters so the next adapter's
    merged model can load without OOM.

The script does NOT retrain, does NOT modify ``run_experiment_192.py`` or
``run_seed256_spread_eval.py``, and accepts no CLI arguments.

Usage (pod-side):
    cd /workspace/explore-persona-space
    export PATH=$HOME/.local/bin:$PATH
    git pull --ff-only
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_followup_teach_frame_eval.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap()


# ── Constants ───────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# The single eval frame for this follow-up: the Qwen2.5-7B-Instruct auto-default
# system prompt (byte-identical to ``TEACHING_PROMPT`` in
# ``run_experiment_192.py:141``).  This is the frame the qwen-default-taught
# adapters were TRAINED under but never evaluated under.
QWEN_AUTO_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
EVAL_FRAME_NAME = "qwen_auto_default"

# The three LoRA adapters we evaluate.  Layout on HF Hub for the
# qwen_default_taught variant is flat: the adapter files live directly under
# ``adapters/sagan-exp192-fact-seed{S}-qwen_default_taught/`` with NO nested
# ``_e1``/``_e2`` subdirectory (verified via ``list_repo_files`` 2026-05-21).
# The local ``seed_label`` slugs (``seed42_e2``, ``seed137_e1``, ``seed256_e1``)
# match the brief's intended naming for local output directories and HF Hub
# upload destinations — they encode the kept-epoch metadata for downstream
# bookkeeping.
ADAPTERS: list[tuple[str, str]] = [
    # (seed_label, path_in_repo)
    ("seed42_e2", "adapters/sagan-exp192-fact-seed42-qwen_default_taught"),
    ("seed137_e1", "adapters/sagan-exp192-fact-seed137-qwen_default_taught"),
    ("seed256_e1", "adapters/sagan-exp192-fact-seed256-qwen_default_taught"),
]

# Probes: the followup's fact_probes.json, content-identical to the original
# #192 freeform probe set.
PROBES_PATH_IN_DATA_REPO = "issue192_persona_spread_qwen_default_taught/datasets/fact_probes.json"

# HF Hub destination prefix for the new raw_completions.json files.
HUB_UPLOAD_PREFIX = "issue192_persona_spread_followup_teach_frame"

# vLLM eval pins (mirror run_seed256_spread_eval.py § "Eval generation pin").
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_MODEL_LEN = 4096
EVAL_MAX_NUM_SEQS = 16

# Local working directories (under PROJECT_ROOT).
WORK_ROOT = PROJECT_ROOT / "outputs" / "issue_192_followup_teach_frame"
EVAL_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "issue_192" / "followup_teach_frame_eval"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_chat_prompt(tokenizer, system_prompt: str, user: str) -> str:
    """Render the ChatML prompt for one user turn under an explicit system message.

    The single frame here always carries a non-None system prompt, so we go
    straight through ``apply_chat_template`` with both ``system`` and ``user``
    turns.  Qwen2.5-7B-Instruct's Jinja template auto-injects a default
    system message when ``messages`` lacks a ``role: "system"`` entry; passing
    the system turn explicitly bypasses that auto-insert entirely — see the
    no_system branch documentation in ``run_seed256_spread_eval.py``.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _download_adapter(seed_label: str, path_in_repo: str, work_root: Path) -> Path:
    """Download the LoRA adapter from HF Hub to a local directory.

    ``work_root`` is the per-seed working directory; ``snapshot_download``
    lands files under ``work_root / <path_in_repo>``.  Returns that resolved
    path, which contains ``adapter_config.json`` + ``adapter_model.safetensors``
    (the layout that ``merge_lora`` expects).
    Idempotent: if both required adapter files are already present, this is a
    no-op.
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")

    work_root.mkdir(parents=True, exist_ok=True)
    extracted = work_root / path_in_repo
    if (extracted / "adapter_config.json").exists() and (
        extracted / "adapter_model.safetensors"
    ).exists():
        logger.info("adapter %s already downloaded at %s — reusing", seed_label, extracted)
        return extracted

    logger.info(
        "downloading adapter %s (%s/%s) -> %s",
        seed_label,
        HF_MODEL_REPO,
        path_in_repo,
        extracted,
    )
    snapshot_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=str(work_root),
        local_dir_use_symlinks=False,
        token=token,
    )
    if not (extracted / "adapter_config.json").exists():
        raise RuntimeError(
            f"snapshot_download did not place adapter_config.json at {extracted}; "
            f"HF Hub layout for {path_in_repo} may have changed."
        )
    return extracted


def _merge_adapter(adapter_dir: Path, merged_dir: Path) -> Path:
    """Merge the LoRA adapter onto the base model. Idempotent."""
    if (merged_dir / "config.json").exists():
        logger.info("merged model already at %s — reusing", merged_dir)
        return merged_dir
    from explore_persona_space.train.sft import merge_lora

    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("merging adapter %s -> %s", adapter_dir, merged_dir)
    merge_lora(BASE_MODEL, str(adapter_dir), str(merged_dir))
    return merged_dir


def _download_probes() -> dict[str, Any]:
    """Fetch the followup fact_probes.json from the HF Hub data repo."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    logger.info("downloading probes %s:%s", HF_DATA_REPO, PROBES_PATH_IN_DATA_REPO)
    local = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=PROBES_PATH_IN_DATA_REPO,
        token=token,
    )
    with open(local) as f:
        return json.load(f)


def _vllm_greedy(
    model_path: str,
    prompts: list[str],
    *,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    max_model_len: int = EVAL_MAX_MODEL_LEN,
    max_num_seqs: int = EVAL_MAX_NUM_SEQS,
) -> list[str]:
    """Greedy temp-0 vLLM generation, one completion per prompt.

    Mirrors ``_vllm_greedy`` in ``run_seed256_spread_eval.py``.  The ``llm``
    binding is local to this function, so once the function returns the
    engine has no remaining strong references (the ``finally`` clause's
    ``cleanup_vllm`` deletes the local binding before ``gc.collect`` runs).
    That guarantees the GPU is fully released before the next adapter's
    merged model loads.
    """
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    llm = create_vllm_engine(
        model_path,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60")),
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=42,
    )
    try:
        params = SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]
    finally:
        cleanup_vllm(llm)
        gc.collect()


def _build_raw_rows(
    completions: list[str],
    keys: list[tuple[str, int]],
    label: str,
) -> list[dict[str, Any]]:
    """Build raw_completions.json rows matching the existing #192 schema.

    Keys per row: ``completion``, ``frame``, ``idx``, ``kind``, ``label``,
    ``probe_id`` (probe_id format: ``{frame}__{kind}__{idx}``).
    All rows here have ``kind = "freeform"`` and ``frame = "qwen_auto_default"``.
    """
    if len(completions) != len(keys):
        raise RuntimeError(f"completion/key length mismatch: {len(completions)} vs {len(keys)}")
    rows: list[dict[str, Any]] = []
    for (frame, idx), completion in zip(keys, completions, strict=True):
        probe_id = f"{frame}__freeform__{idx}"
        rows.append(
            {
                "completion": completion,
                "frame": frame,
                "idx": idx,
                "kind": "freeform",
                "label": label,
                "probe_id": probe_id,
            }
        )
    return rows


def _run_one_adapter(seed_label: str, path_in_repo: str, probes: dict[str, Any]) -> dict[str, str]:
    """Download + merge + eval + upload for a single adapter.

    Returns the dict ``upload_raw_completions_to_data_repo`` returns
    (rel-path -> HF Hub URL) so the top-level main() can log a summary.
    """
    # Per-seed working paths.  Kept separate from any other adapter's
    # paths so back-to-back runs of the three seeds can never collide
    # on the merged-model directory or the eval output, and so the
    # upload helper's recursive scan (see ``upload_raw_completions_to_data_repo``)
    # only sees this seed's raw_completions.json.
    seed_root = WORK_ROOT / seed_label
    adapter_work_root = seed_root / "adapter_download"
    merged_dir = seed_root / "merged"
    out_dir = EVAL_RESULTS_ROOT / seed_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_completions.json"

    # Step 1: download adapter from HF Hub.
    adapter_local = _download_adapter(seed_label, path_in_repo, adapter_work_root)

    # Step 2: merge adapter onto base model.
    merged = _merge_adapter(adapter_local, merged_dir)

    # Step 3: load tokenizer + render freeform prompts under the single
    # Qwen auto-default frame.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        merged,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    freeform_probes: list[dict[str, Any]] = probes["freeform"]
    logger.info(
        "[%s] loaded %d freeform probes; building prompts under '%s' frame",
        seed_label,
        len(freeform_probes),
        EVAL_FRAME_NAME,
    )

    all_prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    for i, probe in enumerate(freeform_probes):
        all_prompts.append(_build_chat_prompt(tokenizer, QWEN_AUTO_DEFAULT_PROMPT, probe["q"]))
        keys.append((EVAL_FRAME_NAME, i))

    logger.info(
        "[%s] total prompts = %d (1 frame x %d probes)",
        seed_label,
        len(all_prompts),
        len(freeform_probes),
    )

    # Defensive: the rendered prompt MUST carry an ``<|im_start|>system`` block
    # (we passed an explicit system turn).  If it doesn't, the chat template
    # changed shape and we should fail loudly before generation.
    sentinel = "<|im_start|>system"
    for prompt, (_, idx) in zip(all_prompts, keys, strict=True):
        if sentinel not in prompt:
            raise RuntimeError(
                f"[{seed_label}] rendered prompt is missing {sentinel!r} "
                f"(probe idx={idx}); chat template may have changed shape, aborting."
            )

    # Step 4: vLLM greedy generation.
    label = f"fact_{seed_label}_qwen_default_taught_teach_freeform"
    completions = _vllm_greedy(str(merged), all_prompts)

    # Step 5: write raw_completions.json in the existing schema.
    raw_rows = _build_raw_rows(completions, keys, label)
    out_path.write_text(json.dumps(raw_rows, indent=2, sort_keys=True) + "\n")
    logger.info("[%s] wrote %d raw rows -> %s", seed_label, len(raw_rows), out_path)

    # Step 6: upload to HF Hub data repo.
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    # ``upload_raw_completions_to_data_repo`` lays files under
    # ``{experiment_name}/raw_completions/<rel path of raw_completions.json
    # under eval_results_dir>``.  To produce the canonical layout
    # ``{HUB_UPLOAD_PREFIX}/{seed_label}/raw_completions/raw_completions.json``
    # we (a) bake the seed_label into ``experiment_name`` so the URL carries
    # the seed slug, and (b) scope the scan to the seed's own ``out_dir``
    # (NOT ``EVAL_RESULTS_ROOT``) so iteration N+1 cannot accidentally
    # re-upload iteration N's outputs.  Same workaround as
    # ``run_seed256_spread_eval.py``.
    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=f"{HUB_UPLOAD_PREFIX}/{seed_label}",
        eval_results_dir=out_dir,
        delete_after=False,
    )
    logger.info("[%s] uploaded %d files to %s", seed_label, len(uploaded), HF_DATA_REPO)
    for rel, url in uploaded.items():
        logger.info("[%s]   %s -> %s", seed_label, rel, url)

    return uploaded


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    # Download probes once — same content for all three adapters.
    probes = _download_probes()

    summary: dict[str, dict[str, str]] = {}
    for seed_label, path_in_repo in ADAPTERS:
        logger.info("════════ starting adapter %s ════════", seed_label)
        summary[seed_label] = _run_one_adapter(seed_label, path_in_repo, probes)
        logger.info("════════ finished adapter %s ════════", seed_label)

    logger.info("all done. uploaded files per seed:")
    for seed_label, uploaded in summary.items():
        for rel, url in uploaded.items():
            logger.info("  %s :: %s -> %s", seed_label, rel, url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
