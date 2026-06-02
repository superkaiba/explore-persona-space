#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #467 — cross-cell probe swap (Test 2; SECONDARY in v2; §4.4).

For each of K conditioning cells (S_narrow=lit, built from the cell's
first-K training rows via ``build_literal_attribute_system_prompt``),
compute the cosine between S_narrow_X-conditioned and S_broad-conditioned
last-prompt-token residual-stream activations at layers L18-L27, evaluated
against each of N probe-source cells' training-question probe sets (5×18
default = 90 off-diagonal pairs; 18 diagonal pairs identical to #463
on-disk lit cosines).

Output per conditioning cell::

    eval_results/issue467/probe_swap/<conditioning_cell>_lit.json
    {
      "conditioning_cell": ...,
      "by_probe_source_cell": {
        "<probe-source-cell>": {
          "cos_by_layer": {"18": 0.42, "19": 0.39, ...},
          "n_probes": 48
        },
        ...
      },
      "layers": [18, 19, ..., 27]
    }

Imports (NOT copies) from ``issue463_predictor_cossim``:

* ``_extract_last_prompt_token``
* ``_per_layer_cos_sim``
* ``extract_training_probes``
* ``measure_pair_flavor`` (NOT used — we open-code a smaller version
  because each conditioning cell is paired with N probe sets, not 1).

CLAUDE.md compliance:
* Checkpoint per phase — one file per conditioning cell, written the
  instant that cell's probe-source sweep finishes.
* No dollar-budget caps.
* Reproducibility metadata in each JSON.

Usage::

    uv run python scripts/issue467_probe_swap.py \
        --conditioning emergent_plus_security openai_health_bad \
            aesthetic_unpopular insecure_code openai_health_correct \
        --probe-source-cells all \
        --layers 18 19 20 21 22 23 24 25 26 27

    # Smoke (2 conditioning × 3 probe-source cells):
    uv run python scripts/issue467_probe_swap.py \
        --conditioning aesthetic_unpopular openai_health_correct \
        --probe-source-cells aesthetic_unpopular openai_health_correct \
            insecure_code \
        --layers 21 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TURNER_EDS_PASSWORD", "model-organisms-em-datasets")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    LITERAL_ATTRIBUTE_K,
    PAIRS,
    S_BROAD,
    build_literal_attribute_system_prompt,
    ensure_dataset,
    load_jsonl,
    reproducibility_metadata,
)
from issue463_predictor_cossim import (  # noqa: E402
    _extract_last_prompt_token,
    _per_layer_cos_sim,
    extract_training_probes,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue467_probe_swap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS = list(range(18, 28))  # L18 - L27
DEFAULT_N_PROBES = 48
DEFAULT_CONDITIONING = [
    "emergent_plus_security",  # high-EM, harmful-topic Qs
    "openai_health_bad",  # high-EM, harmful-topic Qs
    "aesthetic_unpopular",  # mid-EM, benign-topic Qs (within-class)
    "insecure_code",  # code-topic Qs
    "openai_health_correct",  # 0% EM, benign-topic Qs (null control)
]
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue467" / "probe_swap"


def _compute_swap_cell(
    model,
    tokenizer,
    s_narrow: str,
    probes: list[str],
    layers: list[int],
    s_broad_act_cache: dict | None = None,
) -> dict:
    """Compute layer-cosines for one (conditioning, probe-source) pair.

    Caches the S_broad-conditioned activation per probe-source-cell — the
    caller passes the cache for that probe set and we populate or reuse it.
    """
    act_n = _extract_last_prompt_token(model, tokenizer, s_narrow, probes, layers)
    if s_broad_act_cache is not None and "act" in s_broad_act_cache:
        act_b = s_broad_act_cache["act"]
    else:
        act_b = _extract_last_prompt_token(model, tokenizer, S_BROAD, probes, layers)
        if s_broad_act_cache is not None:
            s_broad_act_cache["act"] = act_b
    cos = _per_layer_cos_sim(act_n, act_b)
    del act_n
    torch.cuda.empty_cache()
    return {"cos_by_layer": {str(li): cos[li] for li in layers}, "n_probes": len(probes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--conditioning",
        nargs="+",
        default=DEFAULT_CONDITIONING,
        choices=PAIRS,
        help="Conditioning cells whose lit S_narrow is the persona side.",
    )
    parser.add_argument(
        "--probe-source-cells",
        nargs="+",
        default=["all"],
        help="Probe-source cells (cells whose training-Qs supply the probes). 'all' = every PAIR.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_N_PROBES)
    parser.add_argument("--k", type=int, default=LITERAL_ATTRIBUTE_K)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    probe_source_cells = (
        list(PAIRS) if args.probe_source_cells == ["all"] else args.probe_source_cells
    )
    for ps in probe_source_cells:
        if ps not in PAIRS:
            raise ValueError(f"--probe-source-cells: {ps!r} not in PAIRS")

    needed_cells = set(args.conditioning) | set(probe_source_cells)
    pair_training_rows: dict[str, list[dict]] = {}
    for p in sorted(needed_cells):
        try:
            path = ensure_dataset(p)
            pair_training_rows[p] = load_jsonl(path)
        except FileNotFoundError as e:
            logger.error("Dataset missing for pair=%s — cannot proceed: %s", p, e)
            raise

    # Build per-probe-source-cell probe lists ONCE.
    probes_per_source: dict[str, list[str]] = {}
    for ps in probe_source_cells:
        probes_per_source[ps] = extract_training_probes(
            pair_training_rows[ps], n_probes=args.n_probes, k_lit_skip=args.k
        )
        logger.info("probe-source cell=%s training-Q probes=%d", ps, len(probes_per_source[ps]))

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

    # Per-probe-source-cell S_broad activation cache — S_broad is fixed across
    # conditioning cells, so we compute its activations once per probe-source.
    s_broad_act_caches: dict[str, dict] = {ps: {} for ps in probe_source_cells}

    for cond_cell in args.conditioning:
        rows = pair_training_rows[cond_cell]
        s_narrow_lit = build_literal_attribute_system_prompt(rows, k=args.k)
        per_source: dict[str, dict] = {}
        for ps in probe_source_cells:
            logger.info("Swap pair: conditioning=%s probe-source=%s", cond_cell, ps)
            per_source[ps] = _compute_swap_cell(
                model,
                tokenizer,
                s_narrow_lit,
                probes_per_source[ps],
                args.layers,
                s_broad_act_cache=s_broad_act_caches[ps],
            )
        payload = {
            "conditioning_cell": cond_cell,
            "s_narrow_preview": s_narrow_lit[:400],
            "s_narrow_char_len": len(s_narrow_lit),
            "by_probe_source_cell": per_source,
            "layers": list(args.layers),
            "K_literal_attribute": args.k,
            "n_probes_per_source": args.n_probes,
            "metadata": reproducibility_metadata(
                {
                    "script": "issue467_probe_swap",
                    "torch_seed": args.seed,
                }
            ),
        }
        out_path = OUTPUT_DIR / f"{cond_cell}_lit.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", out_path.relative_to(PROJECT_ROOT))

    logger.info("Probe swap done. Outputs in %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
