#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #458 Predictor M_js: token-level Jensen-Shannon divergence between
the narrow-behavior-prompted and broad-misaligned-prompted base model.

The #404 M_2 predictor (judge-score KL) collapsed because it compared
DISTRIBUTIONS OVER JUDGE SCORES, which are a noisy human-categorical
proxy that loses most of the model's internal disagreement. Here we
compare DISTRIBUTIONS OVER NEXT-TOKEN LOGITS — the canonical mechanistic
signal the model itself uses to choose its first response token.

For each pair × per S_narrow flavor (NL + literal-attribute) × per
preregistered probe Q:

1. Forward-pass ``{S_narrow, Q}`` through the BASE Qwen-2.5-7B-Instruct,
   take the FINAL-layer next-token logits at the last input position,
   softmax → ``p_narrow`` (a vocab-sized prob distribution).
2. Forward-pass ``{S_broad, Q}`` through the SAME base model, same
   readout → ``p_broad``.
3. Per-probe ``JS(p_narrow ‖ p_broad)`` (base-2, bounded [0, 1]).

Headline scalars per (pair, flavor):

* ``mean_JS`` — average JS across probes; higher = base model treats
  narrow + broad prompts MORE differently.
* ``M_js = 1 - mean_JS`` — polarity-aligned to cosine (higher = closer
  in distribution = predict MORE downstream EM, matching M_1's polarity).

Output: ``eval_results/issue458/predictor_jsdiv/{pair}_{flavor}.json``
with per-probe JS, headline scalars, S_narrow preview, and
reproducibility metadata. Mirrors ``issue404_predictor_cossim.py`` so
the regression script can swap predictors with no schema change.

Usage::

    uv run python scripts/issue458_predictor_jsdiv.py
    uv run python scripts/issue458_predictor_jsdiv.py --pairs insecure_code aesthetic_unpopular
    uv run python scripts/issue458_predictor_jsdiv.py --flavors NL --n-probes 200
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

logger = logging.getLogger("issue458_predictor_jsdiv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_N_PROBES = 200
OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue458" / "predictor_jsdiv"


# ── Last-token next-token distribution ─────────────────────────────────────


def _next_token_probs(
    model,
    tokenizer,
    system_prompt: str,
    probes: list[str],
) -> torch.Tensor:
    """Forward-pass each probe under ``system_prompt``; return a (N_probes,
    vocab) tensor of softmax probabilities over the next-token slot at the
    last input position.

    Computed in fp32 on CPU so the per-probe (vocab,) vectors don't blow
    up GPU memory for the JS reduction.
    """
    out_rows: list[torch.Tensor] = []
    for q in probes:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        # logits: (1, T, V) — read the LAST input position (predicts what
        # the assistant turn would emit next).
        last_pos = inputs["input_ids"].shape[1] - 1
        logits = outputs.logits[0, last_pos, :].float().cpu()
        # softmax to a proper probability distribution; clamp tiny to
        # avoid log(0) in the JS reduction.
        probs = torch.softmax(logits, dim=-1)
        out_rows.append(probs)
    return torch.stack(out_rows)  # (N_probes, V)


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise JS divergence (base-2, bounded [0, 1]) along the LAST dim.

    JS(P‖Q) = 0.5 KL(P‖M) + 0.5 KL(Q‖M),  M = 0.5 (P + Q),  logs base 2.

    Inputs ``p, q`` are (..., V) probability tensors. Returns a tensor
    with shape ``p.shape[:-1]`` carrying the per-row JS.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    ln2 = torch.log(torch.tensor(2.0))
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1) / ln2
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1) / ln2
    js = 0.5 * (kl_pm + kl_qm)
    # Numerical-floor + clamp into [0, 1] — JS base 2 is mathematically
    # bounded but a tiny negative can leak from float rounding.
    return js.clamp(min=0.0, max=1.0)


# ── Pair × flavor measurement ──────────────────────────────────────────────


def measure_pair_flavor(
    model,
    tokenizer,
    pair: str,
    flavor: str,
    probes: list[str],
    training_rows: list[dict] | None,
    k: int,
) -> dict:
    """Compute M_js for one (pair, flavor) cell. Returns a result dict."""
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

    p_narrow = _next_token_probs(model, tokenizer, s_narrow, probes)  # (N, V)
    p_broad = _next_token_probs(model, tokenizer, S_BROAD, probes)
    assert p_narrow.shape == p_broad.shape, (p_narrow.shape, p_broad.shape)

    per_probe_js = _js_divergence(p_narrow, p_broad)  # (N,)
    mean_js = float(per_probe_js.mean().item())
    median_js = float(per_probe_js.median().item())
    # M_js is polarity-aligned to M_1 (cosine): higher = narrow distribution
    # CLOSER to broad = predict MORE EM. JS itself measures DIFFERENCE, so
    # subtract from 1 to flip polarity.
    M_js = 1.0 - mean_js

    return {
        "pair": pair,
        "flavor": flavor,
        "s_narrow_preview": s_narrow[:400],
        "s_narrow_char_len": len(s_narrow),
        "s_broad": S_BROAD,
        "n_probes": len(probes),
        "per_probe_js": per_probe_js.tolist(),
        "mean_JS": mean_js,
        "median_JS": median_js,
        # The headline scalar consumed by the regression script.
        "M_js": M_js,
        "K_literal_attribute": k if flavor == "lit" else None,
        "vocab_size": int(p_narrow.shape[-1]),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
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
    args = parser.parse_args()

    # Bind CUDA_VISIBLE_DEVICES BEFORE any cuda allocation — mirrors the
    # round-2 ISSUE 3 fix in issue404_predictor_cossim.py.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    main8 = set(fetch_betley_main_8())
    probes = fetch_preregistered_probes(n=args.n_probes, exclude=main8)
    logger.info("Loaded %d preregistered probes (disjoint from Betley main 8)", len(probes))

    # Cache training rows for the 'lit' flavor up front.
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

    for pair in args.pairs:
        for flavor in args.flavors:
            if flavor == "lit" and not pair_training_rows.get(pair):
                logger.info("Skipping pair=%s flavor=lit (no training rows)", pair)
                continue
            training_rows = pair_training_rows.get(pair, [])
            rows_subset = training_rows[:LIT_FLAVOR_N_ROWS] if flavor == "lit" else None
            out_path = OUTPUT_BASE / f"{pair}_{flavor}.json"
            # Per CLAUDE.md "Checkpoint per phase" — persist each cell as
            # soon as it completes, never accumulate-in-memory across pairs.
            result = measure_pair_flavor(
                model, tokenizer, pair, flavor, probes, rows_subset, args.k
            )
            result["metadata"] = reproducibility_metadata({"script": "issue458_predictor_jsdiv"})
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(
                "Wrote %s; mean_JS=%.4f  M_js=%.4f",
                out_path.relative_to(PROJECT_ROOT),
                result["mean_JS"],
                result["M_js"],
            )

    logger.info("Predictor M_js (token-JS) done. Outputs in %s", OUTPUT_BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
