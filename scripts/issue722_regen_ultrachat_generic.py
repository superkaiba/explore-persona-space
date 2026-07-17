"""Regenerate the (G1) UltraChat generic-question completions for the c_C->v_A doc/dashboard.

The #658 (G1) genre-generalization-ultrachat arm saved v_A activations but NOT the
generated completion TEXT. This script re-generates that exact text by replaying the
SAME prompt-construction + greedy-decoding recipe the #658 extractor used (the v_A
behind the saved activations is deterministic under greedy decode, so this reproduces
the real text).

Recipe (matched to scripts/issue658_extract_base_store.py G1):
  - model:    Qwen/Qwen2.5-7B-Instruct
  - contexts: data/issue594/battery.json  (50 instances)
  - probes:   data/issue594/probes_ultrachat.json  (48 generic UltraChat probes; r["text"])
  - messages: messages_for_instance(inst, probe) -> system + prefix_messages + user(probe)
  - template: apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  - sampling: vLLM SamplingParams(temperature=0.0, max_tokens=512), ONE LLM.generate() call

Output: data/issue_722/ultrachat_generic_completions.json
  {
    "meta": {...},
    "by_context": {context_id: {question_text: completion_text}},
    "flat": [{"context_id","question","completion"}],
  }
Then uploads to the #658 HF data-repo bucket so it is durable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue594_common import messages_for_instance  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATTERY_PATH = PROJECT_ROOT / "data" / "issue594" / "battery.json"
PROBES_PATH = PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat.json"
OUT_DIR = PROJECT_ROOT / "data" / "issue_722"
OUT_PATH = OUT_DIR / "ultrachat_generic_completions.json"

# HF data-repo destination (matches #658 genre arm bucket layout).
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_SUBFOLDER = "issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat-generic"

MAX_NEW_TOKENS = 512  # matched to #658 V0_MAX_NEW_TOKENS / build_prompts greedy pass


def load_contexts() -> list[dict]:
    with open(BATTERY_PATH) as f:
        payload = json.load(f)
    instances = payload["instances"]
    assert len(instances) == 50, f"expected 50 contexts, got {len(instances)}"
    return instances


def load_probes() -> list[str]:
    with open(PROBES_PATH) as f:
        blob = json.load(f)
    probes = [r["text"] for r in blob["probes"]]
    assert len(probes) == 48, f"expected 48 probes, got {len(probes)}"
    return probes


def build_prompts(tokenizer, instances: list[dict], probes: list[str]):
    """Templated prompt strings for every (context, probe) cell + the index.

    Mirrors scripts/issue658_extract_base_store.py::build_prompts exactly:
    persona injection via system turn (messages_for_instance), then
    apply_chat_template(..., add_generation_prompt=True).
    """
    prompts: list[str] = []
    index: list[tuple[str, str]] = []
    for inst in instances:
        for q in probes:
            messages = messages_for_instance(inst, q)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            index.append((inst["id"], q))
    return prompts, index


def vllm_generate(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """vLLM batched greedy generation over all prompts in ONE call (matched recipe)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [o.outputs[0].text for o in outs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true", help="skip the HF upload (local smoke)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    instances = load_contexts()
    probes = load_probes()
    print(
        f"[regen] {len(instances)} contexts x {len(probes)} probes = "
        f"{len(instances) * len(probes)} cells",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts, index = build_prompts(tokenizer, instances, probes)
    assert len(prompts) == 50 * 48 == 2400, f"expected 2400 prompts, got {len(prompts)}"

    t0 = time.time()
    completions = vllm_generate(MODEL, prompts, MAX_NEW_TOKENS)
    print(
        f"[regen] generated {len(completions)} completions in {time.time() - t0:.1f}s", flush=True
    )
    assert len(completions) == len(index)

    by_context: dict[str, dict[str, str]] = {}
    flat: list[dict] = []
    for (cid, q), comp in zip(index, completions, strict=True):
        by_context.setdefault(cid, {})[q] = comp
        flat.append({"context_id": cid, "question": q, "completion": comp})

    out = {
        "meta": {
            "model": MODEL,
            "n_contexts": len(instances),
            "n_questions": len(probes),
            "n_completions": len(flat),
            "decoding": "greedy (temperature=0.0)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "battery": "data/issue594/battery.json",
            "probes": "data/issue594/probes_ultrachat.json (generic UltraChat, r['text'])",
            "recipe_source": "scripts/issue658_extract_base_store.py G1 (matched)",
            "purpose": (
                "Regenerated generic-question completion TEXT behind the #658 (G1) "
                "saved v_A activations for the c_C->v_A doc/dashboard (#722). Text was "
                "never saved originally; greedy decode reproduces it deterministically."
            ),
        },
        "by_context": by_context,
        "flat": flat,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[regen] wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)", flush=True)

    if args.no_upload:
        print("[regen] --no-upload set; skipping HF upload", flush=True)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    dest = f"{HF_SUBFOLDER}/ultrachat_generic_completions.json"
    # UPLOAD_PREFIX_EXEMPT: deliberate backfill of the parent #658 bucket (module docstring)
    url = api.upload_file(
        path_or_fileobj=str(OUT_PATH),
        path_in_repo=dest,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
    )
    print(f"[regen] uploaded -> {HF_DATA_REPO}:{dest}", flush=True)
    print(f"[regen] HF URL: {url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
