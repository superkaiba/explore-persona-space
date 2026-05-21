"""Run the seed-256 spread-frame freeform eval on the existing #192 adapters.

Context: #192 and its qwen-default-taught follow-up both excluded seed 256
from the spread-frame analysis because the MCQ-based teach gate failed on
that seed. But the model DID learn the fact in freeform under both teaching
frames (verified by hand-sampling). To bump the spread analysis from n=2 to
n=3, we need spread-frame raw completions for seed 256 — without retraining
or running the gate.

This script:
  * Downloads one of the two seed-256 LoRA adapters from HF Hub:
        zelthari      → adapters/sagan-exp192-fact-seed256
        qwen-default  → adapters/sagan-exp192-fact-seed256-qwen_default_taught
  * Merges the adapter onto Qwen/Qwen2.5-7B-Instruct via the existing
    ``merge_lora`` helper.
  * Downloads the followup's ``fact_probes.json`` from the data repo
    (freeform probes are content-identical between the two training
    variants — both are Pavlek-syndrome fact probes).
  * Builds prompts for the **freeform** probes across the 4 spread frames
    (``assistant``, ``software_engineer``, ``kindergarten_teacher``,
    ``no_system``) — MCQ is intentionally skipped per the brief.
  * Runs vLLM greedy generation (temp=0, max_new_tokens=2048,
    max_model_len=4096, max_num_seqs=16) — same pins as
    ``run_experiment_192.py``.
  * Writes ``raw_completions.json`` matching the existing schema
    (``probe_id``, ``label``, ``frame``, ``kind``, ``idx``, ``completion``).
  * Uploads to ``superkaiba1/explore-persona-space-data`` at
    ``issue192_persona_spread_seed256_freeform_only/{variant}/raw_completions.json``.

The script does NOT retrain, does NOT run the MCQ gate, and does NOT modify
``run_experiment_192.py``.

Usage (pod-side):
    cd /workspace/explore-persona-space
    export PATH=$HOME/.local/bin:$PATH
    git pull --ff-only
    uv run python scripts/run_seed256_spread_eval.py --variant zelthari
    uv run python scripts/run_seed256_spread_eval.py --variant qwen-default
"""

from __future__ import annotations

# ruff: noqa: E402  (bootstrap() runs before the package-level imports below)
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap()

from explore_persona_space.personas import PERSONAS

# ── Constants ───────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 256
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Adapter HF Hub paths (path-in-repo, under HF_MODEL_REPO of type 'model').
ADAPTER_PATH_IN_REPO: dict[str, str] = {
    "zelthari": "adapters/sagan-exp192-fact-seed256",
    "qwen-default": "adapters/sagan-exp192-fact-seed256-qwen_default_taught",
}

# Probes file: the followup's fact_probes.json, which is content-identical
# to the original #192 freeform probe set (both teach the same Pavlek fact).
PROBES_PATH_IN_DATA_REPO = "issue192_persona_spread_qwen_default_taught/datasets/fact_probes.json"

# HF Hub destination prefix for the new raw_completions.json files.
HUB_UPLOAD_PREFIX = "issue192_persona_spread_seed256_freeform_only"

# The 4 spread eval frames (mirrors SPREAD_FRAMES in run_experiment_192.py).
# Stored as (frame_name, system_prompt) — system_prompt=None means no system
# turn at all (rendered via the hand-rolled ChatML branch below).
SPREAD_FRAMES: dict[str, str | None] = {
    "assistant": "You are a helpful assistant.",
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}

# vLLM eval pins (mirror run_experiment_192.py § "Eval generation pin").
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_MODEL_LEN = 4096
EVAL_MAX_NUM_SEQS = 16

# Local working directories (under PROJECT_ROOT). One sub-directory per
# variant so simultaneous --variant runs cannot stomp on each other.
WORK_ROOT = PROJECT_ROOT / "outputs" / "issue_192_seed256_spread"
EVAL_RESULTS_ROOT = PROJECT_ROOT / "eval_results" / "issue_192" / "seed256_spread_eval"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_chat_prompt(tokenizer, system_prompt: str | None, user: str) -> str:
    """Render the ChatML prompt for one user turn under an optional system message.

    Mirrors ``_build_chat_prompt`` in ``run_experiment_192.py`` byte-for-byte.
    Qwen2.5-7B-Instruct's Jinja template auto-injects a default system
    message when ``messages`` lacks a ``role: "system"`` entry; that would
    collapse the ``no_system`` frame into the ``assistant`` frame. The
    round-5 fix in #192 is to hand-roll the ChatML for the no-system branch
    so zero ``<|im_start|>system`` tokens land in the rendered prompt.
    """
    if system_prompt is None:
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _download_adapter(variant: str, work_root: Path) -> Path:
    """Download the LoRA adapter from HF Hub to a local directory.

    ``work_root`` is the per-variant working directory; the adapter snapshot
    lands at ``work_root / <path_in_repo>``. Returns that resolved path,
    which contains ``adapter_config.json`` + ``adapter_model.safetensors``
    (the layout that ``merge_lora`` expects).
    Idempotent: if the adapter files are already present, this is a no-op.
    """
    from huggingface_hub import snapshot_download

    path_in_repo = ADAPTER_PATH_IN_REPO[variant]
    token = os.environ.get("HF_TOKEN")

    work_root.mkdir(parents=True, exist_ok=True)
    extracted = work_root / path_in_repo
    if (extracted / "adapter_config.json").exists() and (
        extracted / "adapter_model.safetensors"
    ).exists():
        logger.info("adapter already downloaded at %s — reusing", extracted)
        return extracted

    logger.info("downloading adapter %s/%s -> %s", HF_MODEL_REPO, path_in_repo, extracted)
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
            f"snapshot_download layout may have changed."
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
    """Greedy temp-0 vLLM generation, one completion per prompt. Mirrors
    ``_vllm_greedy`` in ``run_experiment_192.py``."""
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
    All rows here have ``kind = "freeform"`` — MCQ is excluded by design.
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


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(ADAPTER_PATH_IN_REPO),
        required=True,
        help="Which seed-256 adapter to evaluate.",
    )
    args = parser.parse_args()
    variant: str = args.variant

    # Per-variant working paths (kept separate so two parallel --variant
    # runs cannot collide on the merged-model directory or the eval output).
    variant_root = WORK_ROOT / variant
    adapter_work_root = variant_root / "adapter_download"
    merged_dir = variant_root / "merged"
    out_dir = EVAL_RESULTS_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_completions.json"

    # Step 1: download adapter from HF Hub.
    adapter_local = _download_adapter(variant, adapter_work_root)

    # Step 2: merge adapter onto base model.
    merged = _merge_adapter(adapter_local, merged_dir)

    # Step 3: load tokenizer + render freeform prompts across 4 spread frames.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        merged,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    probes = _download_probes()
    freeform_probes: list[dict[str, Any]] = probes["freeform"]
    logger.info(
        "loaded %d freeform probes; building prompts across %d spread frames",
        len(freeform_probes),
        len(SPREAD_FRAMES),
    )

    all_prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    for frame_name, system_prompt in SPREAD_FRAMES.items():
        for i, probe in enumerate(freeform_probes):
            all_prompts.append(_build_chat_prompt(tokenizer, system_prompt, probe["q"]))
            keys.append((frame_name, i))

    logger.info(
        "total prompts = %d (%d frames x %d probes)",
        len(all_prompts),
        len(SPREAD_FRAMES),
        len(freeform_probes),
    )

    # Defensive: confirm the no_system frame really has zero system tokens
    # (Qwen template auto-inject regression — see _build_chat_prompt note).
    sentinel = "<|im_start|>system"
    for prompt, (frame_name, idx) in zip(all_prompts, keys, strict=True):
        if frame_name == "no_system" and sentinel in prompt:
            raise RuntimeError(
                f"no_system frame rendered a {sentinel!r} block "
                f"(probe idx={idx}); aborting before generation."
            )

    # Step 4: vLLM greedy generation.
    label = f"fact_seed{SEED}_e2_{variant}_spread_freeform"
    completions = _vllm_greedy(str(merged), all_prompts)

    # Step 5: write raw_completions.json in the existing schema.
    raw_rows = _build_raw_rows(completions, keys, label)
    out_path.write_text(json.dumps(raw_rows, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %d raw rows -> %s", len(raw_rows), out_path)

    # Step 6: upload to HF Hub data repo.
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    # ``upload_raw_completions_to_data_repo`` lays files under
    # ``{experiment_name}/raw_completions/<rel path of raw_completions.json
    # under eval_results_dir>``. To produce the canonical layout
    # ``{HUB_UPLOAD_PREFIX}/{variant}/raw_completions/raw_completions.json``
    # we (a) bake the variant into ``experiment_name`` so the URL carries
    # the variant slug, and (b) scope the scan to the variant's own
    # ``out_dir`` (NOT ``EVAL_RESULTS_ROOT``) so back-to-back runs of the
    # two variants cannot accidentally re-upload each other's files. The
    # final URL carries one extra ``raw_completions/`` segment vs the brief
    # — that segment is intrinsic to the helper's contract and matches the
    # convention every other #192 raw-completion upload follows; see the
    # existing tree at
    # ``issue192_persona_spread_qwen_default_taught/raw_completions/...``.
    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=f"{HUB_UPLOAD_PREFIX}/{variant}",
        eval_results_dir=out_dir,
        delete_after=False,
    )
    logger.info("uploaded %d files to %s", len(uploaded), HF_DATA_REPO)
    for rel, url in uploaded.items():
        logger.info("  %s -> %s", rel, url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
