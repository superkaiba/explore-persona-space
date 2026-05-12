#!/usr/bin/env python3
"""Chen et al. 2025-inspired persona-vector extraction for issue #368.

Implements §4.1.1 (Phase 1, 4 triggers) and §4.2.1 (Phase 2, 10 personas).
Reuses vLLM generation + HF teacher-force machinery from
``scripts/extract_persona_vectors.py`` and computes:

  * pos centroid       per layer (mean across paraphrases × questions × resp tokens)
  * neg centroid       per layer (same, on the shared 5 helpful-assistant paraphrases)
  * chenstyle vector   = unit_normalize(pos - neg)   at L15 / L20 / L25
  * lasttoken vector   = unit_normalize(pos_last_token - neg_last_token)  at L20
  * orthog vector      = orthogonalize chenstyle[L20] against an empty-prompt baseline
  * method A centroid  = #216 last-input-token centroid at L20 (centroid only)
  * method B centroid  = #216 mean-response-token centroid at L20 (centroid only)
  * helpful_test_act   = mean of the 5 helpful-assistant responses' L20 activations
                         (re-used as the projdiff "neutral test" — R4)
  * centroid_mean      = mean over the 11 ALL_EVAL_PERSONAS positive-side
                         response-mean centroids at L15 / L20 / L25 (Phase 0.3,
                         computed once Phase 2 finishes; required by every
                         centered-cosine call downstream)

Output layout::

  data/persona_vectors_chenstyle/qwen2.5-7b-instruct/
    i181/{T_task,T_instruction,T_context,T_format}/
      pos_centroids.pt          # dict layer -> (hidden,) mean-response
      pos_lasttoken.pt          # (hidden,) at L20
      pvec_L15.pt pvec_L20.pt pvec_L25.pt
      pvec_lasttoken_L20.pt
      pvec_orthog_L20.pt
      pcentroid_methodA_L20.pt
      pcentroid_methodB_L20.pt
      responses_pos.json        # cached generations
    personas/{persona}/
      ... same per-trait files
    _helpful_assistant/
      responses_neg.json        # shared (4 triggers + 10 personas reuse)
      neg_centroids.pt          # dict layer -> (hidden,)
      helpful_test_act_L20.pt   # (hidden,) for projdiff
    _empty_prompt/
      pvec_baseline_L20.pt      # orthog reference
    _centroid_mean_L15.pt
    _centroid_mean_L20.pt
    _centroid_mean_L25.pt

Usage::

    # End-to-end, single H100
    uv run python scripts/i368_extract_chenstyle_vectors.py \\
        --phase both --gpu-id 0

    # Phase 2 only (personas — needed for centroid_mean before Phase 1 analysis)
    uv run python scripts/i368_extract_chenstyle_vectors.py --phase 2

    # Smoke-test: 1 trait per source, 1 paraphrase, 2 questions, no GPU
    uv run python scripts/i368_extract_chenstyle_vectors.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Configure HF cache before importing transformers / torch (no-op on local VM
# but mandatory on RunPod pods per CLAUDE.md).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from explore_persona_space.axis.chenstyle import (  # noqa: E402
    DEFAULT_LAYERS,
    HEADLINE_LAYER,
    compute_chenstyle_vector,
    compute_global_centroid_mean,
    compute_orthogonalized_vector,
)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "qwen2.5-7b-instruct"

OUTPUT_BASE = REPO_ROOT / "data" / "persona_vectors_chenstyle" / MODEL_SHORT

TRIGGER_NAMES: list[str] = ["T_task", "T_instruction", "T_context", "T_format"]
NON_BASELINE_PERSONAS: list[str] = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "comedian",
    "police_officer",
    "villain",
    "zelthari_scholar",
]
# The 11 ALL_EVAL_PERSONAS used for centroid_mean (T1) — includes assistant.
ALL_EVAL_PERSONAS: list[str] = [*NON_BASELINE_PERSONAS, "assistant"]

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_SEED = 42
DEFAULT_TEMP = 0.0


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_paraphrase_file(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return [item["pos"] for item in data["instruction"]]


def load_trigger_paraphrases() -> dict[str, list[str]]:
    """Phase 1 positive sets — 4 triggers × 5 paraphrases each."""
    out: dict[str, list[str]] = {}
    base = REPO_ROOT / "data" / "i181_non_persona" / "instructions"
    for trig in TRIGGER_NAMES:
        out[trig] = _load_paraphrase_file(base / f"{trig}.json")
    return out


def load_persona_paraphrases() -> dict[str, list[str]]:
    """Phase 2 positive sets — 10 non-baseline personas × 5 paraphrases."""
    out: dict[str, list[str]] = {}
    base = REPO_ROOT / "data" / "assistant_axis" / "instructions"
    for p in NON_BASELINE_PERSONAS:
        out[p] = _load_paraphrase_file(base / f"{p}.json")
    return out


def load_assistant_paraphrases() -> list[str]:
    """The assistant baseline persona uses the helpful-assistant negset itself
    as its positive set for centroid_mean computation (it IS the negative
    anchor; the same 5 paraphrases stand in as its 'positive set').
    """
    path = REPO_ROOT / "data" / "assistant_axis" / "instructions" / "_helpful_assistant_negset.json"
    return _load_paraphrase_file(path)


def load_helpful_negset() -> list[str]:
    path = REPO_ROOT / "data" / "assistant_axis" / "instructions" / "_helpful_assistant_negset.json"
    return _load_paraphrase_file(path)


def load_eval_questions(n: int | None = None) -> list[str]:
    """20 EVAL_QUESTIONS from personas.py (plan §"EVAL_QUESTIONS")."""
    from explore_persona_space.personas import EVAL_QUESTIONS

    qs = list(EVAL_QUESTIONS)
    if n is not None:
        qs = qs[:n]
    return qs


# ── vLLM generation (reuses extract_persona_vectors.py helper) ──────────────


def generate_responses_for_trait(
    *,
    llm,
    sampling_params,
    trait: str,
    paraphrases: list[str],
    questions: list[str],
    cache_path: Path,
) -> list[dict]:
    """Generate (paraphrase × question) responses for one trait. Cached."""
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if isinstance(cached, list) and len(cached) == len(paraphrases) * len(questions):
            print(f"  [{trait}] cached {len(cached)} responses, loading...")
            return cached
    convos = []
    keys = []
    for p_idx, sys_prompt in enumerate(paraphrases):
        for q_idx, question in enumerate(questions):
            convos.append(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ]
            )
            keys.append((p_idx, q_idx))
    outputs = llm.chat(convos, sampling_params)
    results: list[dict] = []
    for (p_idx, q_idx), out in zip(keys, outputs, strict=True):
        results.append(
            {
                "system_prompt": paraphrases[p_idx],
                "question": questions[q_idx],
                "response": out.outputs[0].text,
                "p_idx": p_idx,
                "q_idx": q_idx,
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)
    print(f"  [{trait}] generated {len(results)} responses -> {cache_path.relative_to(REPO_ROOT)}")
    return results


# ── HF teacher-force activation extraction ──────────────────────────────────


def _activation_hooks(model, layers: list[int]):
    """Forward-hook context manager. Returns (captured_dict, remove_fn)."""

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx):
        def _hook(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return _hook

    handles = [model.model.layers[i].register_forward_hook(_make_hook(i)) for i in layers]

    def _remove() -> None:
        for h in handles:
            h.remove()

    return captured, _remove


def extract_trait_centroids(
    *,
    model,
    tokenizer,
    responses: list[dict],
    layers: list[int],
):
    """For one trait's responses, return:

      mean_response_centroid: dict[layer] -> (hidden,)
      last_token_centroid:    dict[layer] -> (hidden,)
      last_input_centroid:    dict[layer] -> (hidden,)   (Method A on the
                                                        prompt's last token,
                                                        no response needed)

    Used for chenstyle / lasttoken / Method A / Method B simultaneously.
    """
    import torch

    captured, remove_hooks = _activation_hooks(model, layers)

    # Per-layer accumulators
    mean_resp_vecs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    last_resp_vecs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    last_input_vecs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    try:
        for item in responses:
            sys_prompt = item["system_prompt"]
            question = item["question"]
            response = item["response"]
            if not response:  # vLLM occasionally returns "" — skip cleanly
                continue
            # Prompt-only tokenization → last-input-token position
            prompt_msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
            prompt_len = int(prompt_ids.shape[1])
            # Full sequence (prompt + assistant response)
            full_msgs = [*prompt_msgs, {"role": "assistant", "content": response}]
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            full = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
            full_len = int(full["input_ids"].shape[1])
            with torch.no_grad():
                _ = model(**full)
            for layer in layers:
                hs = captured[layer]  # (1, seq, hidden)
                # Last input token (Method A): position prompt_len - 1
                last_input_vecs[layer].append(hs[0, prompt_len - 1, :].float().cpu())
                if full_len > prompt_len:
                    resp_hs = hs[0, prompt_len:full_len, :].float().cpu()
                    mean_resp_vecs[layer].append(resp_hs.mean(dim=0))
                    last_resp_vecs[layer].append(resp_hs[-1, :])
    finally:
        remove_hooks()

    def _stack_mean(d: dict[int, list[torch.Tensor]]) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            if not d[layer]:
                raise RuntimeError(f"no activations captured at layer {layer}")
            out[layer] = torch.stack(d[layer]).mean(dim=0)
        return out

    return {
        "mean_response": _stack_mean(mean_resp_vecs),
        "last_response_token": _stack_mean(last_resp_vecs),
        "last_input_token": _stack_mean(last_input_vecs),
    }


# ── Per-trait pipeline ──────────────────────────────────────────────────────


def _save_torch(obj, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def write_trait_artifacts(
    *,
    trait_dir: Path,
    pos_centroids: dict,  # output of extract_trait_centroids on pos responses
    neg_centroids: dict,  # output of extract_trait_centroids on neg responses
    empty_baseline_L20: torch.Tensor,
    layers: list[int],
) -> None:
    """Compute every persona-vec axis from the centroid dicts, save to disk."""

    # pos centroids per layer per aggregation
    pos_mean_resp = pos_centroids["mean_response"]
    pos_last_resp = pos_centroids["last_response_token"]
    pos_last_input = pos_centroids["last_input_token"]
    neg_mean_resp = neg_centroids["mean_response"]
    neg_last_resp = neg_centroids["last_response_token"]

    # Save raw centroids so downstream tooling can re-derive vectors if needed
    _save_torch(pos_mean_resp, trait_dir / "pos_centroids_mean_response.pt")
    _save_torch(pos_last_resp, trait_dir / "pos_centroids_last_response_token.pt")
    _save_torch(pos_last_input, trait_dir / "pos_centroids_last_input_token.pt")

    # Chenstyle at L15/L20/L25 (mean response)
    for layer in layers:
        pvec = compute_chenstyle_vector(pos_mean_resp[layer], neg_mean_resp[layer])
        _save_torch(pvec, trait_dir / f"pvec_L{layer}.pt")

    # Chenstyle lasttoken at L20
    pvec_lt = compute_chenstyle_vector(pos_last_resp[HEADLINE_LAYER], neg_last_resp[HEADLINE_LAYER])
    _save_torch(pvec_lt, trait_dir / f"pvec_lasttoken_L{HEADLINE_LAYER}.pt")

    # Chenstyle orthog at L20
    pvec_L20 = compute_chenstyle_vector(
        pos_mean_resp[HEADLINE_LAYER], neg_mean_resp[HEADLINE_LAYER]
    )
    pvec_orthog = compute_orthogonalized_vector(pvec_L20, empty_baseline_L20)
    _save_torch(pvec_orthog, trait_dir / f"pvec_orthog_L{HEADLINE_LAYER}.pt")

    # Method-A / Method-B centroids at L20 (no contrast subtraction; just save
    # the pos-side centroids as-is for the centered-cosine downstream call)
    _save_torch(
        pos_last_input[HEADLINE_LAYER], trait_dir / f"pcentroid_methodA_L{HEADLINE_LAYER}.pt"
    )
    _save_torch(
        pos_mean_resp[HEADLINE_LAYER], trait_dir / f"pcentroid_methodB_L{HEADLINE_LAYER}.pt"
    )


# ── Top-level orchestration ─────────────────────────────────────────────────


def _new_vllm(model_name: str, gpu_id: int, max_model_len: int = 2048):
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(
        temperature=DEFAULT_TEMP,
        max_tokens=DEFAULT_MAX_NEW_TOKENS,
        seed=DEFAULT_SEED,
    )
    return llm, sp


def _new_hf(model_name: str, gpu_id: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok


def run_phase(  # noqa: C901  -- orchestration function; splitting hurts readability
    phase: Literal["phase1", "phase2"],
    *,
    gpu_id: int = 0,
    smoke_test: bool = False,
    questions_override: int | None = None,
):
    """End-to-end extraction for one phase.

    Phase 1 = 4 triggers. Phase 2 = 10 personas + assistant baseline centroid
    + empty-prompt baseline + helpful-assistant negative side. The helpful-
    assistant side is generated once (Phase 1 or Phase 2, whichever runs
    first) and cached.
    """
    import torch

    questions = load_eval_questions(n=questions_override or (2 if smoke_test else None))
    neg_paraphrases = load_helpful_negset()
    if smoke_test:
        neg_paraphrases = neg_paraphrases[:1]

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # vLLM phase — generate everything we need
    llm, sp = _new_vllm(MODEL_NAME, gpu_id=gpu_id) if not smoke_test else (None, None)
    if smoke_test:
        # Stub: write empty json caches so the HF phase can run on a tiny
        # subset without ever touching the GPU. We use a synthetic "response"
        # placeholder.
        def _smoke_responses(paraphrases: list[str]) -> list[dict]:
            return [
                {
                    "system_prompt": p,
                    "question": q,
                    "response": "Smoke test response.",
                    "p_idx": pi,
                    "q_idx": qi,
                }
                for pi, p in enumerate(paraphrases)
                for qi, q in enumerate(questions)
            ]

    # --- Shared helpful-assistant side ---
    neg_cache = OUTPUT_BASE / "_helpful_assistant" / "responses_neg.json"
    if neg_cache.exists():
        with open(neg_cache) as f:
            neg_responses = json.load(f)
    elif smoke_test:
        neg_responses = _smoke_responses(neg_paraphrases)
        neg_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(neg_cache, "w") as f:
            json.dump(neg_responses, f)
    else:
        neg_responses = generate_responses_for_trait(
            llm=llm,
            sampling_params=sp,
            trait="_helpful_assistant",
            paraphrases=neg_paraphrases,
            questions=questions,
            cache_path=neg_cache,
        )

    # --- Empty-prompt baseline (orthog reference) — Phase 2 owns extraction ---
    empty_cache = OUTPUT_BASE / "_empty_prompt" / "responses_empty.json"
    if phase == "phase2":
        empty_paraphrases = [""]
        if empty_cache.exists():
            with open(empty_cache) as f:
                empty_responses = json.load(f)
        elif smoke_test:
            empty_responses = _smoke_responses(empty_paraphrases)
            empty_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(empty_cache, "w") as f:
                json.dump(empty_responses, f)
        else:
            empty_responses = generate_responses_for_trait(
                llm=llm,
                sampling_params=sp,
                trait="_empty_prompt",
                paraphrases=empty_paraphrases,
                questions=questions,
                cache_path=empty_cache,
            )

    # --- Per-trait positive sides ---
    if phase == "phase1":
        trait_paraphrases = load_trigger_paraphrases()
        trait_kind = "i181"
    else:
        trait_paraphrases = load_persona_paraphrases()
        trait_paraphrases["assistant"] = load_assistant_paraphrases()
        trait_kind = "personas"

    if smoke_test:
        trait_paraphrases = {k: v[:1] for k, v in list(trait_paraphrases.items())[:1]}

    pos_responses: dict[str, list[dict]] = {}
    for trait, paraphrases in trait_paraphrases.items():
        cache_path = OUTPUT_BASE / trait_kind / trait / "responses_pos.json"
        if cache_path.exists():
            with open(cache_path) as f:
                pos_responses[trait] = json.load(f)
        elif smoke_test:
            pos_responses[trait] = _smoke_responses(paraphrases)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(pos_responses[trait], f)
        else:
            pos_responses[trait] = generate_responses_for_trait(
                llm=llm,
                sampling_params=sp,
                trait=trait,
                paraphrases=paraphrases,
                questions=questions,
                cache_path=cache_path,
            )

    # Drop vLLM to free GPU memory before HF phase
    if llm is not None:
        del llm
        torch.cuda.empty_cache()

    # --- HF teacher-force + activation extraction ---
    if smoke_test:
        print("[smoke] skipping HF extraction (no GPU). Centroid files NOT written.")
        return

    model, tok = _new_hf(MODEL_NAME, gpu_id=gpu_id)
    try:
        layers = list(DEFAULT_LAYERS)

        print(f"[{phase}] Extracting helpful-assistant negative side centroids...")
        neg_centroids = extract_trait_centroids(
            model=model, tokenizer=tok, responses=neg_responses, layers=layers
        )
        # Save neg centroids + helpful_test_act_L20 (used by projdiff downstream)
        _save_torch(
            neg_centroids["mean_response"],
            OUTPUT_BASE / "_helpful_assistant" / "neg_centroids_mean_response.pt",
        )
        _save_torch(
            neg_centroids["last_response_token"],
            OUTPUT_BASE / "_helpful_assistant" / "neg_centroids_last_response_token.pt",
        )
        _save_torch(
            neg_centroids["mean_response"][HEADLINE_LAYER],
            OUTPUT_BASE / "_helpful_assistant" / f"helpful_test_act_L{HEADLINE_LAYER}.pt",
        )

        # Empty-prompt baseline (Phase 2)
        empty_baseline_L20 = None
        if phase == "phase2":
            print(f"[{phase}] Extracting empty-prompt baseline centroid...")
            empty_centroids = extract_trait_centroids(
                model=model, tokenizer=tok, responses=empty_responses, layers=layers
            )
            empty_dir = OUTPUT_BASE / "_empty_prompt"
            _save_torch(
                empty_centroids["mean_response"], empty_dir / "empty_centroids_mean_response.pt"
            )
            # Direction: empty-pos centroid minus neg centroid, unit-normalized.
            empty_baseline_L20 = compute_chenstyle_vector(
                empty_centroids["mean_response"][HEADLINE_LAYER],
                neg_centroids["mean_response"][HEADLINE_LAYER],
            )
            _save_torch(empty_baseline_L20, empty_dir / f"pvec_baseline_L{HEADLINE_LAYER}.pt")
        else:
            # Phase 1 needs the baseline that Phase 2 produced; load it.
            baseline_path = OUTPUT_BASE / "_empty_prompt" / f"pvec_baseline_L{HEADLINE_LAYER}.pt"
            if not baseline_path.exists():
                raise RuntimeError(
                    f"Empty-prompt baseline missing at {baseline_path}. Run "
                    f"--phase 2 first (it computes the baseline) before --phase 1."
                )
            empty_baseline_L20 = torch.load(baseline_path, weights_only=True)

        # Per-trait artifact write
        t0 = time.time()
        for ti, (trait, responses) in enumerate(pos_responses.items()):
            trait_dir = OUTPUT_BASE / trait_kind / trait
            print(f"[{phase}] ({ti + 1}/{len(pos_responses)}) extracting {trait}...")
            pos_centroids = extract_trait_centroids(
                model=model, tokenizer=tok, responses=responses, layers=layers
            )
            write_trait_artifacts(
                trait_dir=trait_dir,
                pos_centroids=pos_centroids,
                neg_centroids=neg_centroids,
                empty_baseline_L20=empty_baseline_L20,
                layers=layers,
            )
            print(f"  saved trait {trait} ({time.time() - t0:.0f}s elapsed)")
    finally:
        del model
        torch.cuda.empty_cache()

    # --- Phase 0.3: centroid_mean over the 11 ALL_EVAL_PERSONAS ---
    if phase == "phase2":
        compute_and_save_centroid_means(layers=list(DEFAULT_LAYERS))


def compute_and_save_centroid_means(layers: list[int]) -> None:
    """Phase 0.3 — centroid_mean[L] = mean over 11 ALL_EVAL_PERSONAS' pos
    centroids at each L. Required by every centered-cosine call downstream.
    """
    import torch

    personas_dir = OUTPUT_BASE / "personas"
    centroids: dict[str, dict[int, torch.Tensor]] = {}
    for p in ALL_EVAL_PERSONAS:
        path = personas_dir / p / "pos_centroids_mean_response.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 0.3 centroid_mean computation requires {path}. Run --phase 2 first."
            )
        centroids[p] = torch.load(path, weights_only=True)
    for layer in layers:
        # Each centroids[p] is a dict {layer: (hidden,)}; flatten to {p: (hidden,)}
        per_layer = {p: c[layer] for p, c in centroids.items()}
        # compute_global_centroid_mean expects either dict-of-1d or dict-of-2d.
        cm = compute_global_centroid_mean(per_layer, layer=layer)
        out_path = OUTPUT_BASE / f"_centroid_mean_L{layer}.pt"
        torch.save(cm, out_path)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}  shape={tuple(cm.shape)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
        help="Which phase to run. 'both' runs phase 2 then phase 1 so centroid_mean is ready.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="No GPU; stub responses; verify the wiring + I/O layout.",
    )
    ap.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Override EVAL_QUESTIONS pool size (default 20).",
    )
    args = ap.parse_args()

    if args.phase in ("2", "both"):
        run_phase(
            "phase2",
            gpu_id=args.gpu_id,
            smoke_test=args.smoke_test,
            questions_override=args.n_questions,
        )
    if args.phase in ("1", "both"):
        run_phase(
            "phase1",
            gpu_id=args.gpu_id,
            smoke_test=args.smoke_test,
            questions_override=args.n_questions,
        )


if __name__ == "__main__":
    main()
