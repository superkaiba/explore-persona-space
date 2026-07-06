#!/usr/bin/env python
"""Issue #778 v2 — neutral-corpus capture (the trait-agnostic covariance null input).

Plan v8 §4 Component B (A2/A3): 500 trait-UNRELATED generic prompts sampled from
UltraChat (``HuggingFaceH4/ultrachat_200k`` split ``train_sft``, first user turn,
seed 42, 10-200 token length filter, trait-keyword screen), one on-policy rollout
each from the BASE model with NO system prompt (T=1.0, max_new=1000), then
response-avg AND last-prompt-token residual-stream activations at all 28 layers.

Outputs (under ``<out-root>/v2/neutral/``, mirroring the HF
``analysis_tensors_v2/neutral/`` layout):
  - ``neutral_prompts.json``        selected prompts + selection provenance
  - ``neutral_rollouts.jsonl``      one row per rollout (persisted BEFORE capture)
  - ``neutral_response_avg.pt``     (n, 28, 3584) fp32
  - ``neutral_last_prompt.pt``      (n, 28, 3584) fp32
  - ``neutral_meta.json``

Powers the ``neutral_cov`` honest-null rung (trait-unrelated by construction) and
is SHARED with task #816. Reuses ``issue778_lib`` capture helpers verbatim; vLLM
engine reaped before the HF capture (the #685 coexistence pattern).
"""

from __future__ import annotations

# vLLM V1 fork-safety: spawn BEFORE any vllm import (gotchas.md #628).
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.neutral")

load_dotenv()

ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_SPLIT = "train_sft"
SAMPLE_SEED = 42
MIN_TOKENS = 10
MAX_TOKENS_PROMPT = 200
# Trait-keyword screen (plan §4): drop prompts touching the three traits.
TRAIT_KEYWORD_RE = re.compile(r"evil|sycophan|hallucin|flatter|malicious", re.IGNORECASE)


def select_neutral_prompts(n_prompts: int, tokenizer) -> list[str]:
    """Deterministic UltraChat sample: shuffle(seed=42), first user turn,
    10-200 token length filter, trait-keyword screen, first ``n_prompts`` kept.

    Uses ``messages[0]["content"]`` (the first user turn), NOT the top-level
    ``prompt`` field (its casing/whitespace is not byte-stable across rows).
    """
    from datasets import load_dataset

    ds = load_dataset(ULTRACHAT_REPO, split=ULTRACHAT_SPLIT)
    ds = ds.shuffle(seed=SAMPLE_SEED)
    kept: list[str] = []
    n_scanned = 0
    for row in ds:
        n_scanned += 1
        msgs = row.get("messages") or []
        if not msgs or msgs[0].get("role") != "user":
            continue
        text = (msgs[0].get("content") or "").strip()
        if not text or TRAIT_KEYWORD_RE.search(text):
            continue
        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
        if not (MIN_TOKENS <= n_tok <= MAX_TOKENS_PROMPT):
            continue
        kept.append(text)
        if len(kept) >= n_prompts:
            break
    if len(kept) < n_prompts:
        raise RuntimeError(
            f"UltraChat neutral selection exhausted at {len(kept)}/{n_prompts} "
            f"after scanning {n_scanned} rows — loosen filters or check the dataset."
        )
    logger.info("selected %d neutral prompts (scanned %d rows)", len(kept), n_scanned)
    return kept


def _chat_prompt_no_system(tokenizer, question: str) -> str:
    """Chat-templated USER-ONLY prompt (no persona / no system prompt — neutral)."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_neutral_capture(
    out_root: Path,
    *,
    n_prompts: int,
    gen_stub: bool = False,
    capture_stub: bool = False,
    skip_capture: bool = False,
) -> dict:
    """Generate + persist neutral rollouts, then capture both activation types."""
    import torch
    from transformers import AutoTokenizer

    neutral_dir = out_root / "v2" / "neutral"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)

    lib.log_phase("neutral", f"start n_prompts={n_prompts}")
    if gen_stub:
        # SMOKE: fixed generic prompts — no dataset download, no GPU.
        prompts = [f"Describe a simple everyday activity number {i}." for i in range(n_prompts)]
    else:
        prompts = select_neutral_prompts(n_prompts, tokenizer)
    with open(neutral_dir / "neutral_prompts.json", "w") as f:
        json.dump(
            {
                "repo": ULTRACHAT_REPO,
                "split": ULTRACHAT_SPLIT,
                "seed": SAMPLE_SEED,
                "token_length_range": [MIN_TOKENS, MAX_TOKENS_PROMPT],
                "keyword_screen": TRAIT_KEYWORD_RE.pattern,
                "gen_stub": gen_stub,
                "prompts": prompts,
                "reproducibility": lib.repro_metadata(),
            },
            f,
            indent=2,
        )

    chat_prompts = [_chat_prompt_no_system(tokenizer, q) for q in prompts]
    if gen_stub:
        responses = [
            f"[stub neutral response {i}] A plain factual answer." for i in range(len(prompts))
        ]
    else:
        from vllm import SamplingParams

        llm = lib.build_vllm_engine()
        try:
            chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
            sp = SamplingParams(
                temperature=lib.EXTRACT_TEMPERATURE,
                top_p=1.0,
                max_tokens=lib.MAX_NEW_TOKENS,
                min_tokens=1,
            )
            responses = []
            n_chunks = (len(chat_prompts) + chunk_size - 1) // chunk_size
            for i in range(0, len(chat_prompts), chunk_size):
                chunk = chat_prompts[i : i + chunk_size]
                logger.info(
                    "[vllm-chunk] neutral chunk %d/%d (%d prompts)",
                    i // chunk_size + 1,
                    n_chunks,
                    len(chunk),
                )
                res = llm.generate(chunk, sp, use_tqdm=False)
                responses.extend(o.outputs[0].text for o in res)
        finally:
            lib.reap_vllm_engine(llm)

    # Persist rollout TEXT the moment generation completes (checkpoint-per-phase).
    with open(neutral_dir / "neutral_rollouts.jsonl", "w") as f:
        for i, (q, resp) in enumerate(zip(prompts, responses, strict=True)):
            f.write(
                json.dumps({"rollout_id": f"neutral-{i:04d}", "question": q, "response": resp})
                + "\n"
            )
    lib.log_phase("neutral", f"rollout text persisted ({len(responses)} rows)")

    if skip_capture:
        lib.log_phase("neutral", "capture SKIPPED (--skip-capture)")
    elif capture_stub:
        rng = torch.Generator().manual_seed(43)
        shape = (len(prompts), lib.N_LAYERS, lib.HIDDEN_DIM)
        torch.save(
            torch.randn(shape, generator=rng, dtype=torch.float32),
            neutral_dir / "neutral_response_avg.pt",
        )
        torch.save(
            torch.randn(shape, generator=rng, dtype=torch.float32),
            neutral_dir / "neutral_last_prompt.pt",
        )
        lib.log_phase("neutral", "STUB acts written (--capture-stub)")
    else:
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
        try:
            resp_avg = lib.capture_response_avg_all_layers(
                model, tokenizer, chat_prompts, responses, device=model.device
            )
            last_prompt = lib.capture_last_prompt_token_all_layers(
                model, tokenizer, chat_prompts, device=model.device
            )
        finally:
            del model
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        assert resp_avg.shape == (len(prompts), lib.N_LAYERS, lib.HIDDEN_DIM), resp_avg.shape
        assert last_prompt.shape == (len(prompts), lib.N_LAYERS, lib.HIDDEN_DIM), last_prompt.shape
        torch.save(resp_avg, neutral_dir / "neutral_response_avg.pt")
        torch.save(last_prompt, neutral_dir / "neutral_last_prompt.pt")

    meta = {
        "n_prompts": len(prompts),
        "gen_stub": gen_stub,
        "capture_stub": capture_stub,
        "skip_capture": skip_capture,
        "temperature": lib.EXTRACT_TEMPERATURE,
        "max_new_tokens": lib.MAX_NEW_TOKENS,
        "system_prompt": None,
        "reproducibility": lib.repro_metadata(),
    }
    with open(neutral_dir / "neutral_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase("neutral", "done", n_prompts=len(prompts))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #778 v2 neutral-corpus capture.")
    ap.add_argument("--out-root", default="data/issue_778")
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--gen-stub", action="store_true", help="SMOKE ONLY: stub prompts+responses")
    ap.add_argument("--capture-stub", action="store_true", help="SMOKE ONLY: stub acts")
    ap.add_argument("--skip-capture", action="store_true", help="SMOKE ONLY: skip capture")
    args = ap.parse_args()
    meta = run_neutral_capture(
        Path(args.out_root),
        n_prompts=args.n_prompts,
        gen_stub=args.gen_stub,
        capture_stub=args.capture_stub,
        skip_capture=args.skip_capture,
    )
    print(json.dumps({"phase": "neutral", "meta": meta}, indent=2))


if __name__ == "__main__":
    main()
