#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #404 Predictor 1: cos-sim of mid-layer activations (S_narrow vs S_broad).

Per plan v3 §4.3. For each pair × per S_narrow flavor (NL + literal-attribute),
forward-pass {S_narrow, Q_i} and {S_broad, Q_i} for 200 preregistered probes,
extract the residual stream at the last input token at layers [7, 14, 21, 27]
(Qwen-2.5-7B 28-layer stack), and compute per-layer mean cos-sim across probes.

Output: ``eval_results/issue_404/predictor_cossim/{pair}_{flavor}.json``
containing per-layer cos-sims, the chosen-layer headline scalar (default
layer 21; held-out CV across pairs is the analyzer's responsibility), and
the reproducibility metadata block.

A26 sub-sampling stability check: for ``pair_insecure_code`` literal-attribute
only, runs a second pass with a different 200-row sub-sample of the training
data and logs the M_1_lit delta between sub-samples.

Usage::

    # All pairs × both flavors:
    uv run python scripts/issue404_predictor_cossim.py

    # One pair only (e.g. to retry insecure_code):
    uv run python scripts/issue404_predictor_cossim.py --pairs insecure_code

    # Override probe count (default 200) or layer set:
    uv run python scripts/issue404_predictor_cossim.py --n-probes 100 --layers 7 14 21 27
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

logger = logging.getLogger("issue404_predictor_cossim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS = [7, 14, 21, 27]
DEFAULT_N_PROBES = 200
HEADLINE_LAYER = 21
OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue_404" / "predictor_cossim"


# ── Activation extraction ──────────────────────────────────────────────────


def _get_last_token_activations(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Forward-pass each probe under the given system prompt and capture the
    last-input-token residual at each requested layer.

    Returns dict layer -> (N_probes, hidden_dim) tensor on CPU in fp32.
    """
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}

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
        per_layer_last: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for q in probes:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            # Reset capture buffers per probe so we don't leak between probes.
            for li in layers:
                captures[li].clear()

            with torch.no_grad():
                _ = model(**inputs)

            last_pos = inputs["input_ids"].shape[1] - 1
            for li in layers:
                hs = captures[li][-1]  # most recent forward pass
                vec = hs[0, last_pos, :].float().cpu()
                per_layer_last[li].append(vec)

        return {li: torch.stack(per_layer_last[li]) for li in layers}
    finally:
        for h in hooks:
            h.remove()


def _per_layer_cos_sim(
    act_a: dict[int, torch.Tensor], act_b: dict[int, torch.Tensor]
) -> dict[int, float]:
    """Per-layer mean cos-sim across the probe dimension."""
    out: dict[int, float] = {}
    for li in act_a:
        a = act_a[li]
        b = act_b[li]
        assert a.shape == b.shape, (a.shape, b.shape)
        # cosine_similarity along the hidden dim, then mean over probes.
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[li] = float(cos.mean().item())
    return out


# ── Pair × flavor measurement ──────────────────────────────────────────────


def measure_pair_flavor(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    probes: list[str],
    layers: list[int],
    training_rows: list[dict] | None,
    k: int,
) -> dict:
    """Run M_1 for one (pair, flavor) cell. Returns a result dict."""
    if flavor == "NL":
        s_narrow = S_NARROW_NL[pair]
    elif flavor == "lit":
        if training_rows is None:
            raise ValueError("flavor='lit' requires training_rows")
        s_narrow = build_literal_attribute_system_prompt(training_rows, k=k)
    else:
        raise ValueError(f"unknown flavor: {flavor!r}")

    logger.info(
        "Measuring pair=%s flavor=%s (S_narrow len=%d chars, %d probes)",
        pair,
        flavor,
        len(s_narrow),
        len(probes),
    )

    act_narrow = _get_last_token_activations(model, tokenizer, s_narrow, probes, layers)
    act_broad = _get_last_token_activations(model, tokenizer, S_BROAD, probes, layers)
    cos_per_layer = _per_layer_cos_sim(act_narrow, act_broad)
    return {
        "pair": pair,
        "flavor": flavor,
        "s_narrow_preview": s_narrow[:400],
        "s_narrow_char_len": len(s_narrow),
        "s_broad": S_BROAD,
        "n_probes": len(probes),
        "layers": list(layers),
        "cos_per_layer": {str(li): cos_per_layer[li] for li in layers},
        "M_1_headline": cos_per_layer.get(HEADLINE_LAYER),
        "headline_layer": HEADLINE_LAYER,
        "K_literal_attribute": k if flavor == "lit" else None,
    }


def stability_check_insecure_code_lit(
    model,
    tokenizer,
    probes: list[str],
    layers: list[int],
    training_rows: list[dict],
    k: int,
) -> dict:
    """A26 sub-sampling stability: re-run M_1_lit on a different 200-row
    sub-sample of insecure-code; report per-layer delta."""
    # First sub-sample: rows 0..199 (default in measure_pair_flavor).
    # Second sub-sample: rows 200..399.
    if len(training_rows) < 400:
        logger.warning(
            "Insufficient training rows (%d) for stability check; skipping",
            len(training_rows),
        )
        return {"ran": False, "reason": "insufficient_rows"}

    sub_a = training_rows[:200]
    sub_b = training_rows[200:400]

    res_a = measure_pair_flavor(model, tokenizer, "insecure_code", "lit", probes, layers, sub_a, k)
    res_b = measure_pair_flavor(model, tokenizer, "insecure_code", "lit", probes, layers, sub_b, k)

    deltas = {
        str(li): abs(res_a["cos_per_layer"][str(li)] - res_b["cos_per_layer"][str(li)])
        for li in layers
    }
    max_delta = max(deltas.values())
    threshold = 0.05
    return {
        "ran": True,
        "subsample_a_rows": "0..199",
        "subsample_b_rows": "200..399",
        "per_layer_delta_abs": deltas,
        "max_delta_abs": max_delta,
        "threshold": threshold,
        "passes_stability": max_delta < threshold,
        "warning": None
        if max_delta < threshold
        else f"Max per-layer delta {max_delta:.4f} > threshold {threshold}",
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--k", type=int, default=LITERAL_ATTRIBUTE_K)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=PAIRS,
        choices=PAIRS,
        help="Subset of pairs to measure (default: all 5).",
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
        "--skip-stability",
        action="store_true",
        help="Skip the A26 sub-sampling stability check on insecure_code lit.",
    )
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load probe set + Betley main 8 to exclude.
    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=args.n_probes, exclude=main8)
    logger.info("Loaded %d preregistered probes (disjoint from Betley main 8)", len(probes))

    # Make sure pair-specific training datasets are cached locally.
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

    # Load model on chosen GPU. We rebind via env var the way
    # extract_persona_vectors.py does.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0")
    logger.info("Loading model %s on GPU %d", args.model, args.gpu_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    n_layers = len(model.model.layers)
    bad_layers = [li for li in args.layers if li < 0 or li >= n_layers]
    if bad_layers:
        raise RuntimeError(f"Requested layers {bad_layers} out of range for {n_layers}-layer model")

    # Run per-pair × per-flavor measurement.
    for pair in args.pairs:
        for flavor in args.flavors:
            if flavor == "lit" and not pair_training_rows.get(pair):
                logger.info("Skipping pair=%s flavor=lit (no training rows)", pair)
                continue
            training_rows = pair_training_rows.get(pair, [])
            out_path = OUTPUT_BASE / f"{pair}_{flavor}.json"
            # Per CLAUDE.md "Checkpoint per phase" — write each cell as soon as
            # it completes; never accumulate-in-memory across all cells.
            # Use rows 0..199 for the canonical lit measurement; NL flavor
            # doesn't need training rows.
            rows_subset = training_rows[:200] if flavor == "lit" else None
            result = measure_pair_flavor(
                model, tokenizer, pair, flavor, probes, args.layers, rows_subset, args.k
            )
            result["metadata"] = reproducibility_metadata({"script": "issue404_predictor_cossim"})
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Wrote %s; M_1 headline (layer %d) = %.4f",
                out_path.relative_to(PROJECT_ROOT),
                HEADLINE_LAYER,
                result["M_1_headline"] if result["M_1_headline"] is not None else float("nan"),
            )

    # A26 stability check on insecure_code lit, if we have enough rows.
    if (
        not args.skip_stability
        and "insecure_code" in args.pairs
        and "lit" in args.flavors
        and len(pair_training_rows.get("insecure_code", [])) >= 400
    ):
        stab = stability_check_insecure_code_lit(
            model, tokenizer, probes, args.layers, pair_training_rows["insecure_code"], args.k
        )
        stab["metadata"] = reproducibility_metadata(
            {"script": "issue404_predictor_cossim", "purpose": "A26_stability_check"}
        )
        stab_path = OUTPUT_BASE / "stability_insecure_code_lit.json"
        with open(stab_path, "w") as f:
            json.dump(stab, f, indent=2)
        logger.info(
            "A26 stability check: max per-layer delta %.4f (threshold 0.05), passes=%s",
            stab["max_delta_abs"],
            stab["passes_stability"],
        )

    logger.info("Predictor 1 (cos-sim) done. Outputs in %s", OUTPUT_BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
