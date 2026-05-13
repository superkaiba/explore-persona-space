# ruff: noqa: RUF001, RUF002, RUF003
# Cascade notation uses "×" and "→" intentionally in docstrings and logs.
"""vLLM eval for issue #366 cascade adapters.

Two eval modes per adapter:

1. **Primary eval**: 11 personas × 26 eval questions × n=10 = 2860 generations
   with SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64, n=10, seed=42).

2. **Seeded probes** (assistant-turn prefill, single-turn):
   - B-seeded (all adapters): prefill `<word> <B>`.
                             6 questions × n=10 = 60 generations.
   - C-seeded (N=4, N=5):    prefill `<word> <word> <C>`. 60 gens.
   - D-seeded (N=5):         prefill `<word> <word> <word> <D>`. 60 gens.

The merge-and-serve loop is the bottleneck of the experiment (≈11 × 6 GB
merge + load = ≈80 GB I/O traffic). We merge into a per-adapter scratch
dir and delete the merged copy as soon as both primary and probe eval
complete. The unmerged LoRA adapter remains on disk for downstream
inspection.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

from data_gen import ALL_EVAL_QS
from markers import MarkerBinding
from word_pool import WORD_POOL

from explore_persona_space.personas import ALL_EVAL_PERSONAS

logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_TEMP = 1.0
EVAL_TOP_P = 0.95
EVAL_MAX_TOKENS = 64
EVAL_N = 10
EVAL_SEED = 42

# A deterministic 6-question slice of EVAL_QUESTIONS used for the seeded
# probes. Picked at module-load time using a frozen rng so successive runs
# probe the same 6 questions.
_PROBE_RNG = random.Random(EVAL_SEED)
SEEDED_PROBE_QUESTIONS: list[str] = _PROBE_RNG.sample(ALL_EVAL_QS, 6)


# ── Merge LoRA into a temp dir for vLLM ──────────────────────────────────────


def merge_lora_for_eval(adapter_path: Path, merged_dir: Path, gpu_id: int = 0) -> Path:
    """Merge a LoRA adapter into the base model and save to merged_dir.

    Idempotent: if merged_dir/config.json exists we assume the merge was done.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (merged_dir / "config.json").exists():
        logger.info("Merged model already exists at %s; skipping merge.", merged_dir)
        return merged_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Merging adapter %s → %s", adapter_path, merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del model, base, tokenizer
    gc.collect()
    import contextlib

    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    return merged_dir


# ── Primary eval: vLLM batched generation ────────────────────────────────────


def run_primary_eval(
    merged_model_path: Path,
    output_dir: Path,
    *,
    gpu_id: int = 0,
) -> dict:
    """Run the 11 × 26 × 10 = 2860-generation primary eval.

    Returns the nested completions dict: {persona: {question: [completions]}}.
    Persists to ``output_dir/primary_completions.json``.
    """
    out_path = output_dir / "primary_completions.json"
    if out_path.exists():
        logger.info("Primary eval cached at %s; loading.", out_path)
        with open(out_path) as f:
            return json.load(f)

    from explore_persona_space.eval.generation import generate_persona_completions

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(
        "Primary eval: %d personas × %d questions × n=%d = %d generations",
        len(ALL_EVAL_PERSONAS),
        len(ALL_EVAL_QS),
        EVAL_N,
        len(ALL_EVAL_PERSONAS) * len(ALL_EVAL_QS) * EVAL_N,
    )
    completions = generate_persona_completions(
        model_path=str(merged_model_path),
        personas=ALL_EVAL_PERSONAS,
        questions=ALL_EVAL_QS,
        num_completions=EVAL_N,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=EVAL_MAX_TOKENS,
        seed=EVAL_SEED,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(completions, f)
    logger.info("Saved primary eval completions: %s", out_path)
    return completions


# ── Seeded probes ───────────────────────────────────────────────────────────


def _seeded_prefill_text(
    seed_kind: str,
    marker_bindings: dict[str, MarkerBinding],
    rng: random.Random,
) -> str:
    """Build the assistant-turn prefill for one seeded probe.

    seed_kind ∈ {"B", "C", "D"}.
    B → `<word> <B>`
    C → `<word> <word> <C>`
    D → `<word> <word> <word> <D>`
    """
    if seed_kind == "B":
        return f"{rng.choice(WORD_POOL)} {marker_bindings['B'].text}"
    if seed_kind == "C":
        return f"{rng.choice(WORD_POOL)} {rng.choice(WORD_POOL)} {marker_bindings['C'].text}"
    if seed_kind == "D":
        return (
            f"{rng.choice(WORD_POOL)} {rng.choice(WORD_POOL)} "
            f"{rng.choice(WORD_POOL)} {marker_bindings['D'].text}"
        )
    raise ValueError(f"Unknown seed_kind: {seed_kind}")


def run_seeded_probe(
    merged_model_path: Path,
    output_dir: Path,
    seed_kind: str,
    marker_bindings: dict[str, MarkerBinding],
    *,
    persona: str = "software_engineer",
    n_questions: int = 6,
    n_per_q: int = 10,
    gpu_id: int = 0,
) -> dict:
    """Run one seeded-probe sweep for ``seed_kind`` ∈ {B,C,D}.

    Uses recipient persona (software_engineer) as the system role and the
    assistant-turn prefill described in ``_seeded_prefill_text``. Returns
    the completions plus the prefill texts used for each row.
    """
    out_path = output_dir / f"seeded_probe_{seed_kind}.json"
    if out_path.exists():
        logger.info("Seeded probe %s cached at %s; loading.", seed_kind, out_path)
        with open(out_path) as f:
            return json.load(f)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    rng = random.Random(EVAL_SEED + ord(seed_kind))  # deterministic per kind

    qs = SEEDED_PROBE_QUESTIONS[:n_questions]
    persona_prompt = ALL_EVAL_PERSONAS[persona]

    tokenizer = AutoTokenizer.from_pretrained(
        str(merged_model_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build prefill strings + chat-templated prompt texts. We assemble the
    # prompt by templating system + user, applying add_generation_prompt=True
    # to get the assistant-turn opener, then appending the prefill text. vLLM
    # will continue from there.
    prompt_texts: list[str] = []
    prefills: list[str] = []
    for q in qs:
        prefill = _seeded_prefill_text(seed_kind, marker_bindings, rng)
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_text = chat_prefix + prefill
        prompt_texts.append(prompt_text)
        prefills.append(prefill)

    llm = LLM(
        model=str(merged_model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60")),
        max_model_len=2048,
        max_num_seqs=64,
        seed=EVAL_SEED,
    )
    sampling = SamplingParams(
        n=n_per_q,
        temperature=EVAL_TEMP,
        top_p=EVAL_TOP_P,
        max_tokens=EVAL_MAX_TOKENS,
    )
    try:
        outputs = llm.generate(prompt_texts, sampling)
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    rows: list[dict] = []
    for q, prefill, output in zip(qs, prefills, outputs, strict=True):
        rows.append(
            {
                "question": q,
                "prefill": prefill,
                # vLLM returns the *continuation only*; the experimenter
                # downstream can prepend prefill if they need the full text.
                "completions": [o.text for o in output.outputs],
            }
        )

    result = {
        "seed_kind": seed_kind,
        "persona": persona,
        "n_questions": len(qs),
        "n_per_q": n_per_q,
        "total_generations": len(qs) * n_per_q,
        "sampling": {
            "temperature": EVAL_TEMP,
            "top_p": EVAL_TOP_P,
            "max_tokens": EVAL_MAX_TOKENS,
            "n": n_per_q,
            "seed": EVAL_SEED,
        },
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved seeded probe %s: %s", seed_kind, out_path)
    return result


# ── Top-level eval entry point per adapter ──────────────────────────────────


def eval_one_adapter(
    adapter_path: Path,
    output_dir: Path,
    marker_bindings: dict[str, MarkerBinding],
    n_chain: int,
    *,
    gpu_id: int = 0,
    keep_merged: bool = False,
) -> dict:
    """Run primary + seeded-probe eval for one adapter.

    The seeded-probe set is determined by ``n_chain``:
      - B-seeded: always
      - C-seeded: when n_chain >= 4
      - D-seeded: when n_chain >= 5
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = output_dir / "merged_for_eval"

    t_start = time.time()

    # 1) Merge once
    merge_lora_for_eval(adapter_path, merged_dir, gpu_id=gpu_id)

    # 2) Primary eval (return value not used here; downstream re-reads the
    # persisted primary_completions.json — but we still need this call to
    # actually run the generation pass).
    run_primary_eval(merged_dir, output_dir, gpu_id=gpu_id)

    # 3) Seeded probes
    seed_kinds = ["B"]
    if n_chain >= 4:
        seed_kinds.append("C")
    if n_chain >= 5:
        seed_kinds.append("D")

    probe_results: dict[str, dict] = {}
    for kind in seed_kinds:
        probe_results[kind] = run_seeded_probe(
            merged_dir, output_dir, kind, marker_bindings, gpu_id=gpu_id
        )

    # 4) Clean up merged shards
    if not keep_merged and merged_dir.exists():
        shutil.rmtree(merged_dir)
        logger.info("Removed merged dir: %s", merged_dir)

    return {
        "adapter_path": str(adapter_path),
        "primary_completions_path": str(output_dir / "primary_completions.json"),
        "seeded_probes": {k: str(output_dir / f"seeded_probe_{k}.json") for k in seed_kinds},
        "eval_seconds": time.time() - t_start,
    }
