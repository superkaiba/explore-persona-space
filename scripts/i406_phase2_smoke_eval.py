"""Phase 2 smoke-eval — does the just-trained LoRA emit ` ※` on its diagonal?

Issue #406 plan v9 §4 Phase 2 pilot-gate companion. Used by the Phase 2
dispatcher's A1 pilot gate (the C2 pilot was retired 2026-05-31 along
with the C2-C5 raw-format conditions): trains the adapter, then runs
THIS script to verify G[T_i, T_i] >= 0.7 on the 50 Q_test questions
BEFORE launching the rest of the batch.

Spins up vLLM with enable_lora=True, loads the just-uploaded adapter
from HF Hub (or local mirror), builds 50 prompts under T_i's own shape,
runs greedy generation with max_tokens=4, scores whether token 0 of the
response == 83399. Writes the result to
logs/issue_406/smoke_<cond>_<lr_tag>.json so the dispatcher can pick it
up and gate the rest of the batch.

CLI:
    uv run python scripts/i406_phase2_smoke_eval.py --condition A1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i406.phase2.smoke")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_REPO = "superkaiba1/explore-persona-space"
LOCAL_ADAPTER_CACHE = Path("/workspace/adapters/i406")
SMOKE_OUT_DIR = Path("logs/issue_406")
PASS_THRESHOLD = 0.7


def _load_q_test() -> list[str]:
    path = Path("data/issue_406/q_test_extended_50.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Q_test not found at {path}. Run i406_phase0_generate_data.py first."
        )
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test, got {len(qs)}")
    return qs


def _load_class_d_rewrites() -> dict[str, dict[str, str]]:
    path = Path("data/issue_406/class_d/rewrites_v1.json")
    if not path.exists():
        raise FileNotFoundError(f"Class D rewrites not found at {path}. Run Phase 0 first.")
    with open(path) as f:
        return json.load(f)


def _resolve_adapter_path(cond_id: str) -> str:
    """Return a local path to the adapter for cond_id, downloading from HF
    if not present locally.

    Per `feedback_eval_script_silent_not_present_misdiagnosis`, distinguish
    "genuinely not on HF" from "downloader bug" with explicit branches.
    """
    from huggingface_hub import snapshot_download

    local_dir = LOCAL_ADAPTER_CACHE
    target_subpath = f"adapters/i406_{cond_id}"
    local_target = local_dir / target_subpath

    if (local_target / "adapter_model.safetensors").exists():
        logger.info("Adapter cache hit: %s", local_target)
        return str(local_target)

    logger.info("Downloading adapter %s -> %s", target_subpath, local_dir)
    snapshot_download(
        repo_id=HF_REPO,
        revision="main",
        allow_patterns=[f"{target_subpath}/*"],
        local_dir=local_dir,
    )
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(
            f"snapshot_download claimed success but {local_target}/adapter_model.safetensors "
            f"is missing. Pattern={target_subpath}/* may be wrong or HF Hub may have "
            "truncated the siblings list. Inspect the local_dir to diagnose."
        )
    return str(local_target)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", required=True, help="One of A1..D5.")
    ap.add_argument(
        "--lr-tag",
        default="default",
        help="Suffix for the output filename so retry results don't overwrite "
        "the default-lr result. e.g. 'default' or 'lr5e-6' (MF-4 fix).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    if args.condition not in CONDITIONS_BY_ID:
        raise ValueError(f"--condition {args.condition!r} not in {list(CONDITIONS_BY_ID)}")
    cond = CONDITIONS_BY_ID[args.condition]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker token id drift: encode({MARKER_TEXT!r}) = {ids}")

    q_test = _load_q_test()
    class_d_rewrites = _load_class_d_rewrites()
    adapter_path = _resolve_adapter_path(args.condition)

    # Build prompts for the diagonal cell — same shape as eval-time T_j
    # because here T_i == T_j (we're checking the trained shape can emit
    # the marker on the questions it was trained to install on).
    prompts = [
        build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites=class_d_rewrites)
        for q in q_test
    ]

    # Import vLLM late (heavy import; only needed once adapter is resolved).
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        seed=42,
    )
    lora_req = LoRARequest(lora_name=args.condition, lora_int_id=1, lora_path=adapter_path)
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts; refusing to score."
        )

    first_token_ids = [out.outputs[0].token_ids[0] for out in outputs]
    n_emit = sum(1 for t in first_token_ids if t == MARKER_ID)
    diagonal_rate = n_emit / len(prompts)

    SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SMOKE_OUT_DIR / f"smoke_{args.condition}_{args.lr_tag}.json"
    payload = {
        "condition": args.condition,
        "lr_tag": args.lr_tag,
        "diagonal_rate": diagonal_rate,
        "n_emit": n_emit,
        "n_total": len(prompts),
        "pass_threshold": PASS_THRESHOLD,
        "pass": diagonal_rate >= PASS_THRESHOLD,
        "first_token_ids": first_token_ids,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "Smoke %s @ %s: %d/%d emitted ` ※` (rate=%.3f; pass=%s) -> %s",
        args.condition,
        args.lr_tag,
        n_emit,
        len(prompts),
        diagonal_rate,
        diagonal_rate >= PASS_THRESHOLD,
        out_path,
    )


if __name__ == "__main__":
    main()
