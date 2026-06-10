"""Phase 1 — vLLM sampling of R base-model responses per (persona, probe).

Output: ``eval_results/issue_470/base_responses/{persona}.json`` containing
``{"persona": ..., "probes": [...], "responses": [[R strings] per probe]}``.

Per CLAUDE.md "Checkpoint per phase": one file per persona, written the moment
that persona's 50 x R samples complete. Resume-safe: pre-existing per-persona
files are skipped.

Persona injection ALWAYS via system prompt (CLAUDE.md persona rule).

Subprocess-isolated from Phases 2-3 (vLLM worker-teardown trap, #399). This
module imports vLLM at the top of ``generate_persona_responses()`` only.

Usage (typically invoked by ``scripts/dispatch_jsdiv_470.py``)::

    # short module alias: predictor_jsdiv_470.phase1_sample_responses
    uv run python -m \\
        explore_persona_space.experiments.predictor_jsdiv_470.phase1_sample_responses \\
        --personas software_engineer --probes 5 --R 2          # smoke
    # Full sweep: all 24 personas x 50 probes x R=8 = 9600 generations.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Pin HF cache to MooseFS BEFORE any HF or vLLM import (the snapshot dirs are
# the bulk of the per-pod quota).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

load_dotenv()

from explore_persona_space.experiments.predictor_jsdiv_470.common import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_R,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PHASE1_DIR,
    checkpoint_is_compatible,
    get_eval_personas_24,
    load_eval_50_probes,
    reproducibility_metadata,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_prompts(
    personas: list[str],
    persona_prompts: dict[str, str],
    probes: list[dict],
    tokenizer,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Build the flat list of chat-template-formatted prompts + their keys.

    Returns (prompts, keys) where ``keys[i] = (persona_name, probe_idx)``.
    """
    prompts: list[str] = []
    keys: list[tuple[str, int]] = []
    for persona in personas:
        sys_prompt = persona_prompts[persona]
        for probe_idx, probe in enumerate(probes):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": probe["wrong_claim"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            keys.append((persona, probe_idx))
    return prompts, keys


def _generate_hf_fallback(
    personas: list[str],
    persona_prompts: dict[str, str],
    probes: list[dict],
    out_dir: Path,
    *,
    model_path: str,
    r: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> None:
    """CPU-only HF Transformers fallback for the smoke run (vLLM needs CUDA).

    Production always uses ``generate_persona_responses`` (vLLM) per the
    "Use vLLM for generation" CLAUDE.md rule. This path is for the SMOKE
    only — exercises the SAME prompt-building, persona-injection, output
    schema, and per-persona checkpointing as the vLLM path so every
    downstream phase consumes a real Phase 1 output during the smoke.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    torch.manual_seed(seed)
    for persona in personas:
        sys_prompt = persona_prompts[persona]
        per_probe_responses: list[list[str]] = []
        for probe in probes:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": probe["wrong_claim"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=r,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            decoded = [
                tokenizer.decode(outs[i, prompt_len:], skip_special_tokens=True)
                for i in range(outs.shape[0])
            ]
            per_probe_responses.append(decoded)
        payload = {
            "persona": persona,
            "system_prompt": sys_prompt,
            "probes": [p["wrong_claim"] for p in probes],
            "responses": per_probe_responses,
            "R": r,
            "n_probes": len(probes),
            "metadata": reproducibility_metadata(
                {
                    "script": "predictor_jsdiv_470.phase1_sample_responses",
                    "phase": "phase1_sample_responses",
                    "backend": "hf_fallback_cpu_smoke",
                    "model_path": model_path,
                    "R": r,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "n_probes": len(probes),
                }
            ),
        }
        write_json(out_dir / f"{persona}.json", payload)
        logger.info("Wrote %s.json (HF fallback)", persona)


def generate_persona_responses(
    personas: list[str],
    persona_prompts: dict[str, str],
    probes: list[dict],
    out_dir: Path,
    *,
    model_path: str = DEFAULT_MODEL,
    r: int = DEFAULT_R,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = DEFAULT_SEED,
    gpu_memory_utilization: float = 0.60,
    max_model_len: int = 4096,
) -> None:
    """Sample R responses per (persona, probe) and write one JSON per persona."""
    # Checkpoint resume: only skip personas whose existing artifact is COMPATIBLE
    # with the requested signature (model, backend=vllm, R, n_probes, seed, temp,
    # top_p, max_new_tokens). Mismatches regenerate. Blocker #2 — without this,
    # a smoke artifact from Qwen-0.5B / R=2 / 5 probes silently satisfies a
    # subsequent production run for Qwen-7B / R=8 / 50 probes.
    expected_sig = {
        "model_path": model_path,
        "backend": "vllm",
        "R": r,
        "n_probes": len(probes),
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "phase": "phase1_sample_responses",
    }
    pending: list[str] = []
    for p in personas:
        path = out_dir / f"{p}.json"
        ok, reason = checkpoint_is_compatible(path, expected_sig)
        if ok:
            continue
        if path.exists():
            logger.warning(
                "Regenerating %s: existing checkpoint INCOMPATIBLE (%s)", path.name, reason
            )
        pending.append(p)
    if not pending:
        logger.info(
            "Phase 1: all %d personas have COMPATIBLE outputs; nothing to do.", len(personas)
        )
        return
    logger.info(
        "Phase 1: %d/%d personas pending sampling (R=%d, %d probes each)",
        len(pending),
        len(personas),
        r,
        len(probes),
    )

    # Import vLLM lazily so this module is import-safe on a CPU dev box.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts, keys = _build_prompts(pending, persona_prompts, probes, tokenizer)
    logger.info(
        "Total prompts=%d (= %d personas x %d probes); R=%d -> %d generations",
        len(prompts),
        len(pending),
        len(probes),
        r,
        len(prompts) * r,
    )

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=r,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Group by persona; for each persona keep responses indexed by probe_idx.
    per_persona: dict[str, list[list[str]]] = {p: [[] for _ in probes] for p in pending}
    for out, (persona, probe_idx) in zip(outputs, keys, strict=True):
        # vLLM returns out.outputs as a list of n=R completions.
        per_persona[persona][probe_idx] = [o.text for o in out.outputs]

    meta = reproducibility_metadata(
        {
            "script": "predictor_jsdiv_470.phase1_sample_responses",
            "phase": "phase1_sample_responses",
            "backend": "vllm",
            "model_path": model_path,
            "R": r,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "n_probes": len(probes),
        }
    )

    # Write per-persona JSON the moment we have all that persona's responses.
    # vLLM returns all outputs together; we still write each persona separately
    # so partial failures (rare here, but possible on resumed runs) don't
    # cost all the progress.
    for persona in pending:
        payload = {
            "persona": persona,
            "system_prompt": persona_prompts[persona],
            "probes": [p["wrong_claim"] for p in probes],
            "responses": per_persona[persona],  # list[list[str]] shape (n_probes, R)
            "R": r,
            "n_probes": len(probes),
            "metadata": meta,
        }
        out_path = out_dir / f"{persona}.json"
        write_json(out_path, payload)
        logger.info(
            "Wrote %s (responses shape: %d probes x %d completions)",
            out_path.name,
            len(probes),
            r,
        )

    # Tear down vLLM. The #399 trap means worker subprocesses may survive even
    # after `del llm` + gc; subprocess-isolating Phase 1 from Phases 2-3 (via
    # the dispatcher) is the actual mitigation. We still do the polite cleanup
    # here in case someone runs Phase 1 inline.
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        logger.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--R", type=int, default=DEFAULT_R, dest="r")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help="Subset of EVAL_PERSONAS_24 to sample (default: all 24).",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=None,
        help="Cap to first N probes (smoke mode). Default: all 50.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--use-hf-fallback",
        action="store_true",
        help="Use HF Transformers model.generate() instead of vLLM. CPU-only "
        "smoke ONLY — production always uses vLLM (CLAUDE.md generation rule).",
    )
    args = parser.parse_args()

    persona_prompts = get_eval_personas_24()
    if args.personas:
        unknown = [p for p in args.personas if p not in persona_prompts]
        if unknown:
            raise ValueError(f"Unknown personas: {unknown}; expected from EVAL_PERSONAS_24")
        personas = list(args.personas)
    else:
        personas = list(persona_prompts.keys())

    probes = load_eval_50_probes()
    if args.probes is not None:
        probes = probes[: args.probes]
        logger.info("Smoke mode: capped probes to first %d", len(probes))

    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    if args.use_hf_fallback:
        logger.warning("HF fallback path active — CPU-only smoke only; production must use vLLM.")
        # Same metadata-based compatibility check as the vLLM path (blocker #2):
        # a smoke artifact must not be silently reused for a different signature.
        expected_sig = {
            "model_path": args.model,
            "backend": "hf_fallback_cpu_smoke",
            "R": args.r,
            "n_probes": len(probes),
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "phase": "phase1_sample_responses",
        }
        pending: list[str] = []
        for p in personas:
            path = PHASE1_DIR / f"{p}.json"
            ok, reason = checkpoint_is_compatible(path, expected_sig)
            if ok:
                continue
            if path.exists():
                logger.warning(
                    "Regenerating %s: existing checkpoint INCOMPATIBLE (%s)", path.name, reason
                )
            pending.append(p)
        if not pending:
            logger.info("All personas have COMPATIBLE outputs; nothing to do.")
        else:
            _generate_hf_fallback(
                personas=pending,
                persona_prompts=persona_prompts,
                probes=probes,
                out_dir=PHASE1_DIR,
                model_path=args.model,
                r=args.r,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
    else:
        generate_persona_responses(
            personas=personas,
            persona_prompts=persona_prompts,
            probes=probes,
            out_dir=PHASE1_DIR,
            model_path=args.model,
            r=args.r,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    logger.info("Phase 1 complete. Outputs in %s", PHASE1_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
