#!/usr/bin/env python
"""Issue #685 Phase B.2 — known behavior direction u_l(b) (persona-vectors recipe).

Validity companion 2 (plan §3.5). For each behavior ``b``, compute the
difference-in-means RESPONSE-mean direction
``u_l(b) = mean_response(C+b) - mean_response(C)`` over the subset contexts x Q,
using ``extract_centroids_response_mean`` (recipe (b) of persona-distance-metrics:
vLLM greedy generation -> HF teacher-forced response-pool). The INSTRUCT model
only. Output: ``store/issue685[_smoke]/instruct_known_directions.pt`` keyed
``directions: {behavior: {layer: (H,) u}}`` — consumed by Phase B
(``issue685_compute_metrics.py``) for the projection metric ``|Delta . u_hat| /
||Delta||``.

The read position here (mean over RESPONSE tokens) is DELIBERATELY different from
the Phase-A last-prompt-token read, so the projection is a genuine cross-check
rather than tautological.

Usage::

    uv run python scripts/issue685_known_directions.py               # full (GPU/vLLM)
    uv run python scripts/issue685_known_directions.py --smoke        # tiny (HF-generate fallback)
"""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
# vLLM V1 fork-EngineCore silent-death guard (gotchas.md): spawn BEFORE import vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    extract_centroids_response_mean,
)
from explore_persona_space.personas import EVAL_QUESTIONS  # noqa: E402

# Import the experiment constants from the Phase-A driver (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue685_extract_shifts import (  # noqa: E402
    BEHAVIORS,
    CONTEXTS,
    INSTRUCT_MODEL,
    LAYERS,
    SMOKE_BEHAVIORS,
    SMOKE_LAYERS,
    SMOKE_MODEL,
)

# Validity-companion contexts (plan §3.5: the 4 subset contexts span the cosine
# range + the safety-vs-neutral axis). For the projection direction we use the
# same subset; the Phase-A Delta has all 10 contexts but the projection metric is
# reported per (context, behavior, layer) only where both exist.
SUBSET_CONTEXTS = ["assistant", "software_engineer", "villain", "medical_doctor"]
SMOKE_SUBSET_CONTEXTS = ["assistant", "software_engineer"]


def _preseed_responses_cache_hf(
    model_id: str,
    conditions: dict[str, str | None],
    questions: list[str],
    max_new_tokens: int,
    cache_path: Path,
) -> None:
    """CPU/HF-generate fallback that writes the responses cache schema.

    ``extract_centroids_response_mean`` reloads an existing cache and SKIPS the
    vLLM generation, so writing the cache here lets the (CPU-runnable, HF)
    teacher-forced response-mean pass run end-to-end without a GPU/vLLM. Used by
    the ``--gen-backend hf`` smoke path; the real run uses ``vllm`` (no cache).

    Cache schema matches ``_generate_responses_vllm``: one row per
    (persona, question) with ``prompt_token_ids`` + ``response_token_ids`` (a
    trailing EOS stripped from the response so the pool is content-only).
    """
    import json as _json

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    eos_id = tok.eos_token_id
    rows: list[dict] = []
    for p_name, p_prompt in conditions.items():
        for q_idx, q in enumerate(questions):
            msgs = ([{"role": "system", "content": p_prompt}] if p_prompt else []) + [
                {"role": "user", "content": q}
            ]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            prompt_len = inputs["input_ids"].shape[1]
            resp_ids = out[0, prompt_len:].tolist()
            finish_reason = "length" if len(resp_ids) >= max_new_tokens else "stop"
            if resp_ids and resp_ids[-1] == eos_id:
                resp_ids = resp_ids[:-1]
            if not resp_ids:  # never write a zero-token response (would NaN the pool)
                resp_ids = [tok.encode(".", add_special_tokens=False)[0]]
            rows.append(
                {
                    "persona": p_name,
                    "question_idx": q_idx,
                    "prompt_token_ids": inputs["input_ids"][0].tolist(),
                    "response_token_ids": resp_ids,
                    "finish_reason": finish_reason,
                }
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        _json.dumps({"model": model_id, "max_new_tokens": max_new_tokens, "rows": rows})
    )
    del model


def _direction_for_behavior(
    model_id: str,
    behavior_text: str,
    contexts: dict[str, str | None],
    questions: list[str],
    layers: list[int],
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
    cache_dir: Path,
    behavior_name: str,
    gen_backend: str,
) -> dict[int, torch.Tensor]:
    """``u_l(b) = mean_response(C+b) - mean_response(C)`` over the subset contexts.

    Builds the 2 x n_context condition set (bare + augmented), extracts
    response-mean centroids in ONE call, then differences the matched (aug, bare)
    pairs and means over contexts per layer. ``gen_backend='hf'`` pre-seeds the
    response cache via CPU HF-generate (smoke); ``'vllm'`` lets the function
    generate with vLLM (real run).
    """
    conditions: dict[str, str | None] = {}
    bare_names: list[str] = []
    aug_names: list[str] = []
    for c, s_c in contexts.items():
        bare_key = f"bare__{c}"
        aug_key = f"aug__{c}"
        conditions[bare_key] = s_c
        conditions[aug_key] = (s_c + "\n\n" + behavior_text) if s_c else behavior_text
        bare_names.append(bare_key)
        aug_names.append(aug_key)

    cache_path = cache_dir / f"responses_{behavior_name}.json"
    if gen_backend == "hf" and not cache_path.exists():
        _preseed_responses_cache_hf(model_id, conditions, questions, max_new_tokens, cache_path)

    centroids, names, stats = extract_centroids_response_mean(
        model_id,
        conditions,
        questions=questions,
        layers=layers,  # PASS EXPLICITLY — default is [20,21] (plan note)
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        responses_cache_path=cache_dir / f"responses_{behavior_name}.json",
    )
    name_to_idx = {n: i for i, n in enumerate(names)}
    print(
        f"[issue685.B2] behavior={behavior_name}: truncation_rate={stats['truncation_rate']:.4f}, "
        f"mean_response_tokens={stats['mean_response_tokens']:.1f}"
    )
    u: dict[int, torch.Tensor] = {}
    for layer in layers:
        mat = centroids[layer]  # (n_cond, H)
        diffs = []
        for bn, an in zip(bare_names, aug_names, strict=True):
            diffs.append(mat[name_to_idx[an]] - mat[name_to_idx[bn]])
        u[layer] = torch.stack(diffs).mean(dim=0)  # (H,)
    return u


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #685 Phase B.2 — known behavior directions u_l(b).",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny verification slice.")
    parser.add_argument("--out-dir", default=None, help="override the store output dir.")
    parser.add_argument("--device", default=None, help="device string.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="generation cap (default 512 full / 64 smoke).",
    )
    parser.add_argument(
        "--gen-backend",
        choices=["vllm", "hf"],
        default=None,
        help="response-generation backend; default vllm full / hf smoke (CPU pre-seed).",
    )
    args = parser.parse_args()

    smoke = args.smoke
    gen_backend = args.gen_backend or ("hf" if smoke else "vllm")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens is not None else (64 if smoke else 512)
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("store/issue685_smoke" if smoke else "store/issue685")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "respmean_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        model_id = SMOKE_MODEL
        contexts = {c: CONTEXTS[c] for c in SMOKE_SUBSET_CONTEXTS}
        behaviors = {b: BEHAVIORS[b] for b in SMOKE_BEHAVIORS}
        layers = SMOKE_LAYERS
        questions = EVAL_QUESTIONS[:2]
    else:
        model_id = INSTRUCT_MODEL
        contexts = {c: CONTEXTS[c] for c in SUBSET_CONTEXTS}
        behaviors = BEHAVIORS
        layers = LAYERS
        questions = EVAL_QUESTIONS

    print(
        f"[issue685.B2] {'SMOKE ' if smoke else ''}known directions: model={model_id}, "
        f"{len(contexts)} contexts x {len(behaviors)} behaviors x {len(questions)} q, "
        f"layers={layers}, device={device}"
    )

    directions: dict[str, dict[int, torch.Tensor]] = {}
    for b_name, b_text in behaviors.items():
        directions[b_name] = _direction_for_behavior(
            model_id,
            b_text,
            contexts,
            questions,
            layers,
            device,
            dtype,
            max_new_tokens,
            cache_dir,
            b_name,
            gen_backend,
        )

    payload = {
        "directions": directions,
        "metadata": {
            "task": 685,
            "phase": "B.2",
            "model": model_id,
            "recipe": "response_mean_diff_in_means (persona-vectors 2507.21509 recipe b)",
            "subset_contexts": list(contexts.keys()),
            "behavior_names": list(behaviors.keys()),
            "layers": layers,
            "n_questions": len(questions),
            "max_new_tokens": max_new_tokens,
            "smoke": smoke,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "argv": sys.argv[1:],
        },
    }
    out_path = out_dir / "instruct_known_directions.pt"
    torch.save(payload, out_path)
    print(f"[issue685.B2] wrote {out_path} ({len(directions)} behavior directions)")


if __name__ == "__main__":
    main()
