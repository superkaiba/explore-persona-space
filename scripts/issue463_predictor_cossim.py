#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #463 — persona-vector cosine with TWO extraction points × layer sweep.

Canonical definition (CLAUDE.md "Persona-distance metrics" → Persona
Vectors, arXiv 2507.21509). Cosine between the ``S_narrow``- and
``S_broad``-conditioned base model's mean residual-stream activation
across probes, swept over layers {7, 14, 21, 27}, at TWO extraction
points:

* ``last_prompt_token`` — residual at the FINAL input position of
  ``{S_x, Q}`` (the legacy #404 recipe; superset of
  ``issue404_predictor_cossim.py`` at the headline layer 21).
* ``response_mean`` — sample one response under ``S_x`` (temp=1.0,
  ≤max_new_tokens), then mean-pool the residual activations over the
  RESPONSE token positions (the persona-vectors recipe (b)).

This script is the narrow-vs-broad cosine (a single pair of summary
vectors per probe and layer), NOT the difference-of-means / +/- prompt
direction used by the original Persona Vectors paper for steering.

Output per cell::

    eval_results/issue463/predictor_cossim/<pair>_<flavor>.json

Carrying a nested ``{extraction_point: {layer: cosine}}`` dict plus the
per-extraction-point headline (layer-21) so the regression script can
slice cleanly.

Usage::

    uv run python scripts/issue463_predictor_cossim.py
    uv run python scripts/issue463_predictor_cossim.py --pairs insecure_code \
        --flavors NL --n-probes 4 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    LIT_FLAVOR_N_ROWS,
    LITERAL_ATTRIBUTE_K,
    PAIRS,
    S_BROAD,
    S_NARROW_NL,
    build_literal_attribute_system_prompt,
    ensure_dataset,
    fetch_betley_main_8,
    fetch_preregistered_probes,
    load_jsonl,
    reproducibility_metadata,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue463_predictor_cossim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS = [7, 14, 21, 27]
DEFAULT_N_PROBES = 48
DEFAULT_MAX_NEW_TOKENS = 128
HEADLINE_LAYER = 21
EXTRACTION_POINTS = ("last_prompt_token", "response_mean")
OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue463" / "predictor_cossim"


# ── Hook helpers (re-used from issue404_predictor_cossim.py) ───────────────


def _attach_layer_hooks(model, layers: list[int], buffer: dict[int, list[torch.Tensor]]) -> list:
    """Attach forward hooks at ``model.model.layers[li]`` that append the
    last forward pass's hidden states (output[0]) to ``buffer[li]``.
    Returns the list of hook handles for later removal.
    """

    def make_hook(layer_idx):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            buffer[layer_idx].append(hs.detach())

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)
    return hooks


def _build_prompt_ids(tokenizer, system_prompt: str, q: str) -> torch.Tensor:
    """Return ``(1, T_prompt)`` token IDs for the chat-template prompt with
    ``add_generation_prompt=True``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)
    return enc["input_ids"]


# ── Extraction recipes ─────────────────────────────────────────────────────


@torch.no_grad()
def _extract_last_prompt_token(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Per-probe residual at the LAST input position of ``{S_x, Q}``.

    Returns ``{layer: (N_probes, hidden) fp32 CPU tensor}``.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = _attach_layer_hooks(model, layers, captures)
    try:
        per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for q in probes:
            prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
            for li in layers:
                captures[li].clear()
            _ = model(prompt_ids)
            last_pos = prompt_ids.shape[1] - 1
            for li in layers:
                hs = captures[li][-1]  # (1, T, hidden)
                vec = hs[0, last_pos, :].float().cpu()
                per_layer[li].append(vec)
        return {li: torch.stack(per_layer[li]) for li in layers}
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def _extract_response_mean(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    layers: list[int],
    max_new_tokens: int,
) -> dict[int, torch.Tensor]:
    """For each probe, sample ONE response (temp=1.0) under ``system_prompt``,
    then teacher-force the full ``[prompt, response]`` and mean-pool the
    residual activations over the RESPONSE positions per layer.

    Returns ``{layer: (N_probes, hidden) fp32 CPU tensor}``. If a sampled
    response is empty (immediate EOS), we fall back to the last prompt
    token at that layer so the per-probe vector is still defined.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = _attach_layer_hooks(model, layers, captures)
    try:
        per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for q in probes:
            prompt_ids = _build_prompt_ids(tokenizer, system_prompt, q).to(model.device)
            prompt_len = prompt_ids.shape[1]

            # 1. Sample one response. ``model.generate`` triggers hook calls
            # but we DON'T need those — we re-run a single teacher-force
            # forward below to read the residuals deterministically.
            for li in layers:
                captures[li].clear()
            gen = model.generate(
                prompt_ids,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
                if tokenizer.pad_token_id is None
                else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            resp_ids = gen[0, prompt_len:].detach()
            eos_id = tokenizer.eos_token_id
            if eos_id is not None:
                eos_pos = (resp_ids == eos_id).nonzero(as_tuple=False).flatten()
                if eos_pos.numel() > 0:
                    resp_ids = resp_ids[: int(eos_pos[0].item())]

            # 2. Teacher-force the full sequence ONCE; read the hook
            # captures at the response slice.
            for li in layers:
                captures[li].clear()

            if resp_ids.numel() == 0:
                # Fallback: response is empty, use the last prompt-token
                # residual (computed from the prompt-only forward above).
                _ = model(prompt_ids)
                pos = prompt_ids.shape[1] - 1
                for li in layers:
                    hs = captures[li][-1]
                    vec = hs[0, pos, :].float().cpu()
                    per_layer[li].append(vec)
                continue

            full_ids = torch.cat([prompt_ids, resp_ids.unsqueeze(0).to(model.device)], dim=1)
            _ = model(full_ids)
            # Response token positions in the full sequence are
            # ``[prompt_len, prompt_len + T_response)``.
            start, end = prompt_len, prompt_len + resp_ids.shape[0]
            for li in layers:
                hs = captures[li][-1]  # (1, T_full, hidden)
                slice_ = hs[0, start:end, :].float().cpu()  # (T_response, hidden)
                vec = slice_.mean(dim=0)  # mean-pool over response positions
                per_layer[li].append(vec)
        return {li: torch.stack(per_layer[li]) for li in layers}
    finally:
        for h in hooks:
            h.remove()


# ── Cosine reduction ───────────────────────────────────────────────────────


def _per_layer_cos_sim(
    act_a: dict[int, torch.Tensor], act_b: dict[int, torch.Tensor]
) -> dict[int, float]:
    out: dict[int, float] = {}
    for li in act_a:
        a = act_a[li]
        b = act_b[li]
        assert a.shape == b.shape, (a.shape, b.shape)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[li] = float(cos.mean().item())
    return out


# ── Pair × flavor measurement ───────────────────────────────────────────────


def _resolve_s_narrow(pair: str, flavor: str, training_rows: list[dict] | None, k: int) -> str:
    if flavor == "NL":
        return S_NARROW_NL[pair]
    if flavor == "lit":
        if training_rows is None:
            raise ValueError("flavor='lit' requires training_rows")
        return build_literal_attribute_system_prompt(training_rows, k=k)
    raise ValueError(f"unknown flavor: {flavor!r}")


def measure_pair_flavor(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    probes: list[str],
    layers: list[int],
    max_new_tokens: int,
    training_rows: list[dict] | None,
    k: int,
) -> dict:
    """Compute per-extraction-point × per-layer cosine for one (pair, flavor)."""
    s_narrow = _resolve_s_narrow(pair, flavor, training_rows, k)
    s_broad = S_BROAD

    logger.info(
        "Measuring pair=%s flavor=%s (S_narrow len=%d chars, %d probes, layers=%s)",
        pair,
        flavor,
        len(s_narrow),
        len(probes),
        layers,
    )

    cos_by_extraction: dict[str, dict[int, float]] = {}

    # Extraction point (a): last prompt token.
    act_n_last = _extract_last_prompt_token(model, tokenizer, s_narrow, probes, layers)
    act_b_last = _extract_last_prompt_token(model, tokenizer, s_broad, probes, layers)
    cos_by_extraction["last_prompt_token"] = _per_layer_cos_sim(act_n_last, act_b_last)
    del act_n_last, act_b_last
    torch.cuda.empty_cache()

    # Extraction point (b): mean over each persona's own response tokens.
    act_n_resp = _extract_response_mean(
        model, tokenizer, s_narrow, probes, layers, max_new_tokens=max_new_tokens
    )
    act_b_resp = _extract_response_mean(
        model, tokenizer, s_broad, probes, layers, max_new_tokens=max_new_tokens
    )
    cos_by_extraction["response_mean"] = _per_layer_cos_sim(act_n_resp, act_b_resp)
    del act_n_resp, act_b_resp
    torch.cuda.empty_cache()

    headlines = {ep: cos_by_extraction[ep].get(HEADLINE_LAYER) for ep in EXTRACTION_POINTS}

    return {
        "pair": pair,
        "flavor": flavor,
        "s_narrow_preview": s_narrow[:400],
        "s_narrow_char_len": len(s_narrow),
        "s_broad": s_broad,
        "n_probes": len(probes),
        "layers": list(layers),
        "max_new_tokens": max_new_tokens,
        "K_literal_attribute": k if flavor == "lit" else None,
        "cos_by_extraction": {
            ep: {str(li): cos_by_extraction[ep][li] for li in layers} for ep in EXTRACTION_POINTS
        },
        "headline_layer": HEADLINE_LAYER,
        "headline_by_extraction": headlines,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--k", type=int, default=LITERAL_ATTRIBUTE_K)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=PAIRS,
        choices=PAIRS,
        help="Subset of pairs to measure (default: all PAIRS).",
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        default=["NL", "lit"],
        choices=["NL", "lit"],
        help="Subset of S_narrow flavors to measure.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch seed for reproducible response sampling (default: 0).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=args.n_probes, exclude=main8)
    logger.info("Loaded %d preregistered probes (disjoint from Betley main 8)", len(probes))

    pair_training_rows: dict[str, list[dict]] = {}
    for pair in args.pairs:
        if "lit" in args.flavors:
            try:
                dataset_path = ensure_dataset(pair)
                pair_training_rows[pair] = load_jsonl(dataset_path)
                logger.info(
                    "pair=%s training rows=%d (dataset=%s)",
                    pair,
                    len(pair_training_rows[pair]),
                    dataset_path.name,
                )
            except FileNotFoundError as e:
                logger.warning("Dataset for pair=%s missing; skipping lit flavor: %s", pair, e)
                pair_training_rows[pair] = []

    device = torch.device("cuda:0")
    logger.info("Loading model %s on GPU %d", args.model, args.gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_model_layers = len(model.model.layers)
    bad = [li for li in args.layers if li < 0 or li >= n_model_layers]
    if bad:
        raise RuntimeError(f"Requested layers {bad} out of range for {n_model_layers}-layer model")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for pair in args.pairs:
        for flavor in args.flavors:
            if flavor == "lit" and not pair_training_rows.get(pair):
                logger.info("Skipping pair=%s flavor=lit (no training rows)", pair)
                continue
            training_rows = pair_training_rows.get(pair, [])
            rows_subset = training_rows[:LIT_FLAVOR_N_ROWS] if flavor == "lit" else None
            out_path = OUTPUT_BASE / f"{pair}_{flavor}.json"
            result = measure_pair_flavor(
                model,
                tokenizer,
                pair,
                flavor,
                probes,
                args.layers,
                max_new_tokens=args.max_new_tokens,
                training_rows=rows_subset,
                k=args.k,
            )
            result["metadata"] = reproducibility_metadata(
                {"script": "issue463_predictor_cossim", "torch_seed": args.seed}
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Wrote %s; last_prompt_token@L%d=%.4f  response_mean@L%d=%.4f",
                out_path.relative_to(PROJECT_ROOT),
                HEADLINE_LAYER,
                result["headline_by_extraction"]["last_prompt_token"]
                if result["headline_by_extraction"]["last_prompt_token"] is not None
                else float("nan"),
                HEADLINE_LAYER,
                result["headline_by_extraction"]["response_mean"]
                if result["headline_by_extraction"]["response_mean"] is not None
                else float("nan"),
            )

    logger.info("Predictor cossim (2-extraction × layer-sweep) done. Outputs in %s", OUTPUT_BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
