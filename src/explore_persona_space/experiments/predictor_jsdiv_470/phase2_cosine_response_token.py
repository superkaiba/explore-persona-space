"""Phase 2 — persona-vectors cosine recipe (b): mean-pool over response tokens.

For each persona, take Phase 1's sampled responses, run a HF forward pass on
each ``(system, user, assistant=response)`` chat, and mean-pool the residual
stream activations across the RESPONSE token positions at layers {7,14,21,27}
(per arXiv 2507.21509 recipe (b) + the canonical Qwen-7B layer set used by
#404/#411).

Per-layer cosine across the 24 personas yields a 24x24 matrix; we save one
JSON per layer + a combined ``cossim_pairs.json`` indexed by (source, bystander)
for downstream use by the regression in Phase 5.

Output:
  ``eval_results/issue_470/cossim_response_token/layer_<L>.json``
  ``eval_results/issue_470/cossim_response_token/cossim_pairs.json``

Subprocess-isolated from Phase 1 (HF Transformers + bf16; #399 vLLM teardown
mitigation runs Phase 2 + Phase 3 in the same Python process but in a
DIFFERENT process from Phase 1).

Usage::

    uv run python -m \\
        explore_persona_space.experiments.predictor_jsdiv_470.phase2_cosine_response_token \\
        --personas software_engineer comedian        # smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
load_dotenv()

from explore_persona_space.experiments.predictor_jsdiv_470.common import (  # noqa: E402
    DEFAULT_LAYERS,
    DEFAULT_MODEL,
    HEADLINE_LAYER,
    PHASE1_DIR,
    PHASE2_DIR,
    get_eval_personas_24,
    read_json,
    reproducibility_metadata,
    write_json,
)

logger = logging.getLogger("predictor_jsdiv_470.phase2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _mean_response_token_activations(
    model,
    tokenizer,
    system_prompt: str,
    probe: str,
    responses: list[str],
    layers: list[int],
    device,
) -> dict[int, object]:
    """For ONE (persona, probe), forward-pass each of the R responses as a full
    chat completion, mean-pool the response-token residuals at each layer, and
    return ``{layer: (R, hidden_dim) tensor (CPU fp32)}``.

    The "response tokens" are the assistant turn produced under this persona.
    We tokenize the full chat and identify the assistant-content slice via the
    prompt-length / full-length diff (same pattern as ``divergence.py``).
    """
    import torch

    captures: dict[int, list] = {li: [] for li in layers}

    def make_hook(layer_idx):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[layer_idx].append(hs.detach())

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    try:
        per_layer_vecs: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        # Pre-compute the prompt-only length (system + user, with generation prompt).
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": probe},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        for response in responses:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": probe},
                {"role": "assistant", "content": response},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            # Response tokens occupy positions [prompt_len, len(full_ids)) — but
            # the chat template wraps the response in end-of-turn tokens we
            # don't want to pool over. We drop the trailing 2 tokens (Qwen
            # uses `<|im_end|>\n` ~= 2 tokens) but require at least 1 actual
            # response token; if the response is too short we keep what we have.
            tail_drop = min(2, max(0, len(full_ids) - prompt_len - 1))
            response_end = len(full_ids) - tail_drop
            if response_end <= prompt_len:
                logger.warning(
                    "Skipping degenerate response (prompt_len=%d, full_len=%d) for probe=%r",
                    prompt_len,
                    len(full_ids),
                    probe[:80],
                )
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            for li in layers:
                captures[li].clear()
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            for li in layers:
                hs = captures[li][-1]
                # Mean-pool over the response-token slice.
                resp_slice = hs[0, prompt_len:response_end, :]
                if resp_slice.shape[0] == 0:
                    continue
                vec = resp_slice.mean(dim=0).float().cpu()
                per_layer_vecs[li].append(vec)

        return {
            li: torch.stack(per_layer_vecs[li]) if per_layer_vecs[li] else torch.empty(0)
            for li in layers
        }
    finally:
        for h in hooks:
            h.remove()


def _persona_centroid(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    responses_per_probe: list[list[str]],
    layers: list[int],
    device,
) -> dict[int, object]:
    """Aggregate (probe, response) -> per-persona centroid per layer.

    Returns ``{layer: (hidden_dim,) tensor (CPU fp32)}`` — the mean of all
    (probe, response) mean-pooled vectors for this persona.
    """
    import torch

    per_layer_all: dict[int, list] = {li: [] for li in layers}
    for probe, responses in zip(probes, responses_per_probe, strict=True):
        if not responses:
            continue
        per_layer = _mean_response_token_activations(
            model, tokenizer, system_prompt, probe, responses, layers, device
        )
        for li in layers:
            if per_layer[li].numel() > 0:
                per_layer_all[li].append(per_layer[li])

    centroids: dict[int, object] = {}
    for li in layers:
        if not per_layer_all[li]:
            raise RuntimeError(f"No activations collected for layer {li}")
        stacked = torch.cat(per_layer_all[li], dim=0)  # (n_probe * R, hidden)
        centroids[li] = stacked.mean(dim=0)
    return centroids


def main() -> int:  # noqa: C901 — argparse + compat-skip + per-layer loop reads clearer inline
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help="Subset of EVAL_PERSONAS_24 (default: all that have Phase 1 outputs).",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    persona_prompts = get_eval_personas_24()
    if args.personas:
        unknown = [p for p in args.personas if p not in persona_prompts]
        if unknown:
            # Concern #8: fail-fast on unknown personas (mirror Phase 1/3).
            # Silent filtering hid a bystander typo in #391's predecessor.
            raise ValueError(f"Unknown personas: {unknown}; expected from EVAL_PERSONAS_24")
        personas = list(args.personas)
    else:
        personas = list(persona_prompts.keys())

    # Filter to personas that have a Phase 1 output (resume safety + smoke mode).
    pending = [p for p in personas if (PHASE1_DIR / f"{p}.json").exists()]
    if not pending:
        raise RuntimeError(
            f"Phase 2 requires Phase 1 outputs in {PHASE1_DIR}, found none for {personas}"
        )

    # Round-3 blocker `smoke-artifacts-poison-production` (Phase 2 leg):
    # capture the UPSTREAM Phase 1 signature now (model/backend/R/n_probes/seed/
    # temperature/top_p/max_new_tokens) and require that EVERY existing Phase 2
    # layer file's stored `upstream_phase1_*` block matches it. Otherwise a
    # smoke Phase-2 artifact (Qwen-0.5B / R=2 / 5 probes) silently satisfies a
    # production run whose Phase 1 just regenerated at Qwen-7B / R=8 / 50.
    #
    # All `pending` Phase 1 outputs MUST share the same signature; we read all of
    # them once and assert agreement before extracting the upstream signature.
    phase1_sigs: dict[str, dict] = {}
    for persona in pending:
        blob = read_json(PHASE1_DIR / f"{persona}.json")
        meta = blob.get("metadata", {})
        phase1_sigs[persona] = {
            "model_path": meta.get("model_path"),
            "backend": meta.get("backend"),
            "R": meta.get("R") or blob.get("R"),
            "n_probes": meta.get("n_probes") or blob.get("n_probes"),
            "seed": meta.get("seed"),
            "temperature": meta.get("temperature"),
            "top_p": meta.get("top_p"),
            "max_new_tokens": meta.get("max_new_tokens"),
        }
    # All Phase 1 outputs must agree (a mid-run regen at different R for some
    # personas is itself an error to surface here, not paper over).
    canonical_sig = next(iter(phase1_sigs.values()))
    for persona, sig in phase1_sigs.items():
        if sig != canonical_sig:
            raise RuntimeError(
                f"Phase 1 outputs disagree on signature: {persona}={sig} vs "
                f"canonical={canonical_sig}. Re-run Phase 1 with a single signature "
                f"or delete the divergent file."
            )
    upstream_sig = canonical_sig

    expected_layers = sorted(args.layers)
    layer_files = [PHASE2_DIR / f"layer_{li}.json" for li in expected_layers]
    if all(p.exists() for p in layer_files):
        sigs_match = True
        for li, path in zip(expected_layers, layer_files, strict=True):
            blob = read_json(path)
            stored_personas = sorted(blob.get("personas", []))
            if stored_personas != sorted(pending):
                logger.warning(
                    "Phase 2 layer %d INCOMPATIBLE: personas=%s want=%s — regenerating",
                    li,
                    stored_personas,
                    sorted(pending),
                )
                sigs_match = False
                break
            if int(blob.get("layer", -1)) != li:
                sigs_match = False
                break
            stored_meta = blob.get("metadata", {})
            stored_model = stored_meta.get("model_path")
            if stored_model and stored_model != args.model:
                logger.warning(
                    "Phase 2 layer %d INCOMPATIBLE: model=%s want=%s — regenerating",
                    li,
                    stored_model,
                    args.model,
                )
                sigs_match = False
                break
            # NEW: upstream Phase 1 signature must match.
            stored_upstream = {k: stored_meta.get(f"upstream_phase1_{k}") for k in upstream_sig}
            # An older artifact missing the upstream block is INCOMPATIBLE (it
            # was generated before this signature was tracked → unknowable
            # provenance → regenerate). Required-key semantics (round-3
            # `compat-check-required-key-dontcare` fix applied locally here).
            for k, want in upstream_sig.items():
                have = stored_upstream.get(k)
                if want is None:
                    continue
                if have is None:
                    logger.warning(
                        "Phase 2 layer %d INCOMPATIBLE: upstream_phase1_%s missing "
                        "in existing artifact (want=%r) — regenerating",
                        li,
                        k,
                        want,
                    )
                    sigs_match = False
                    break
                if have != want:
                    logger.warning(
                        "Phase 2 layer %d INCOMPATIBLE: upstream_phase1_%s have=%r "
                        "want=%r — regenerating",
                        li,
                        k,
                        have,
                        want,
                    )
                    sigs_match = False
                    break
            if not sigs_match:
                break
        if sigs_match:
            logger.info(
                "Phase 2: all %d layer files COMPATIBLE for %d personas; skipping recompute.",
                len(expected_layers),
                len(pending),
            )
            return 0

    logger.info("Phase 2: computing recipe-(b) cosine for %d personas", len(pending))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s on %s", args.model, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Per-persona centroid per layer.
    centroids: dict[str, dict[int, torch.Tensor]] = {}
    for persona in pending:
        phase1 = read_json(PHASE1_DIR / f"{persona}.json")
        probes = phase1["probes"]
        responses_per_probe = phase1["responses"]
        logger.info("Aggregating activations for persona=%s (%d probes)", persona, len(probes))
        centroids[persona] = _persona_centroid(
            model,
            tokenizer,
            persona_prompts[persona],
            probes,
            responses_per_probe,
            list(args.layers),
            device,
        )

    # Per-layer NxN cosine matrix.
    persona_order = list(pending)
    for li in args.layers:
        n = len(persona_order)
        vecs = torch.stack([centroids[p][li] for p in persona_order])  # (N, hidden)
        normed = torch.nn.functional.normalize(vecs, dim=-1)
        cos_matrix = (normed @ normed.T).cpu().tolist()
        # Round-3 blocker: embed the upstream Phase 1 signature so a later
        # invocation can verify our layer file was produced from the SAME
        # Phase 1 outputs (not stale smoke samples).
        upstream_meta = {f"upstream_phase1_{k}": v for k, v in upstream_sig.items()}
        payload = {
            "layer": li,
            "personas": persona_order,
            "cosine_matrix": cos_matrix,
            "n_personas": n,
            "metadata": reproducibility_metadata(
                {
                    "script": "predictor_jsdiv_470.phase2_cosine_response_token",
                    "phase": "phase2_cosine_response_token",
                    "model_path": args.model,
                    "layer": li,
                    "layers": list(expected_layers),
                    **upstream_meta,
                }
            ),
        }
        out_path = PHASE2_DIR / f"layer_{li}.json"
        write_json(out_path, payload)
        logger.info("Wrote %s (layer=%d, %dx%d cosine matrix)", out_path.name, li, n, n)

    # Combined per-pair lookup at the headline layer (analyzer-friendly).
    headline_path = PHASE2_DIR / f"layer_{HEADLINE_LAYER}.json"
    if headline_path.exists():
        headline = read_json(headline_path)
        pairs: dict[str, dict[str, float]] = {}
        for i, src in enumerate(headline["personas"]):
            pairs[src] = {}
            for j, bys in enumerate(headline["personas"]):
                pairs[src][bys] = headline["cosine_matrix"][i][j]
        payload = {
            "headline_layer": HEADLINE_LAYER,
            "personas": headline["personas"],
            "cosine_pairs": pairs,
            "metadata": reproducibility_metadata(
                {
                    "script": "predictor_jsdiv_470.phase2_cosine_response_token",
                    "purpose": "headline_pairs",
                }
            ),
        }
        write_json(PHASE2_DIR / "cossim_pairs.json", payload)
        logger.info("Wrote cossim_pairs.json (headline layer %d)", HEADLINE_LAYER)

    logger.info("Phase 2 complete. Outputs in %s", PHASE2_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
