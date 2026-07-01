#!/usr/bin/env python
"""Issue #816 Exp-2 steering / causal control (+ Phase-0 layer probe).

Runs ON the pod (GPU). Generates steered responses with HF ``model.generate()``
under the per-decode-step ``ActivationSteerer`` hook (vLLM cannot host it) and
PERSISTS the raw generations + per-draw diagnostics to disk. The graded Sonnet
judge runs OFF-POD in Phase B (this script never calls the judge).

Two phases (``--phase``):
  - ``probe``  (Phase 0): steering-effectiveness sanity check on EVIL at layers
    {14,17,20,23} x coef {2,4} x 20 eval questions x 3 rollouts. Confirms layer 20
    is at/near the max trait-score-lift, then FREEZE. (The judge scoring of the
    probe runs off-pod; this writes the raw probe generations.)
  - ``steer`` (Exp-2): per (trait, arm) generation. Real arm sweeps coef
    {-4,-2,-1,0,1,2,4,8} x 20 q x 5 rollouts; random arm sweeps coef {2,4,8} x
    15 norm-matched dirs x 20 q x 5 rollouts. RAW (non-unit-normalized) r_B[19].

Steering vector = the raw ``r_B[trait][19]`` (layer 20). Random directions are
norm-matched to ‖r_B[trait][19]‖ and seeded DETERMINISTICALLY per (trait, coef,
dir_idx, rollout). Generation seed set per (trait, coef, dir, rollout).

``--cells N`` limits the (trait, arm, coef, dir) work set to the first N cells —
the SAME flag the dispatcher threads for the unified smoke (smoke IS this script
with ``--cells 1``). Per-cell JSON is written the moment each cell completes
(checkpoint-per-cell). Pod-side: never shells ``task.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
import issue816_lib as ilib

from explore_persona_space.experiments.issue816 import steering
from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.steering_cli")
load_dotenv()

# Exp-2 grids (plan v2 §4 / §11).
EXP2_REAL_COEFS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)
EXP2_RANDOM_COEFS = (2.0, 4.0, 8.0)
EXP2_N_RANDOM_DIRS = 15
EXP2_N_ROLLOUTS = 5
# Phase-0 probe grid (plan v2 §4 Phase 0).
PROBE_LAYERS_1IDX = (14, 17, 20, 23)
PROBE_COEFS = (2.0, 4.0)
PROBE_N_ROLLOUTS = 3
PROBE_TRAIT = "evil"


def _refusal_flag(text: str) -> bool:
    """Cheap high-precision refusal flag for a per-draw diagnostic (NOT the DV).

    The DV is the off-pod graded Sonnet trait score; this flag only lets the
    analyzer separate "random steered the trait" from "random broke the model
    into refusal-shaped noise". A judge-labeled refusal is added off-pod; this is
    the cheap on-pod pre-flag (a leading canonical refusal opener).
    """
    t = text.strip().lower()
    openers = (
        "i can't",
        "i cannot",
        "i won't",
        "i will not",
        "i'm sorry",
        "i am sorry",
        "i'm not able",
        "i am not able",
        "sorry, but",
        "as an ai",
    )
    return any(t.startswith(o) for o in openers)


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_model_and_tokenizer(model_name: str, cpu_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float32 if cpu_only else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if not cpu_only:
        model = model.to("cuda")
    return model, tokenizer


def _build_work(phase: str, traits: list[str]) -> list[dict]:
    """Build the (trait, arm, coef, dir_idx, layer) work list for this phase.

    A "cell" = one (trait, arm, coef, dir_idx, layer) group of rollouts, so
    ``--cells 1`` (smoke) runs exactly one group through the identical code path.
    """
    work: list[dict] = []
    if phase == "probe":
        for layer in PROBE_LAYERS_1IDX:
            for coef in PROBE_COEFS:
                work.append(
                    {
                        "trait": PROBE_TRAIT,
                        "arm": "probe",
                        "coef": coef,
                        "dir_idx": None,
                        "layer": layer,
                        "n_rollouts": PROBE_N_ROLLOUTS,
                    }
                )
        return work
    # steer phase (Exp-2)
    for trait in traits:
        for coef in EXP2_REAL_COEFS:
            work.append(
                {
                    "trait": trait,
                    "arm": "e2_real" if coef != 0.0 else "e2_coef0",
                    "coef": coef,
                    "dir_idx": None,
                    "layer": ilib.LAYER_20_1IDX,
                    "n_rollouts": EXP2_N_ROLLOUTS,
                }
            )
        for coef in EXP2_RANDOM_COEFS:
            for dir_idx in range(EXP2_N_RANDOM_DIRS):
                work.append(
                    {
                        "trait": trait,
                        "arm": "e2_randnorm",
                        "coef": coef,
                        "dir_idx": dir_idx,
                        "layer": ilib.LAYER_20_1IDX,
                        "n_rollouts": EXP2_N_ROLLOUTS,
                    }
                )
    return work


def _run_cell(
    model,
    tokenizer,
    cell: dict,
    *,
    trait_rb,
    trait_rb_sha: str,
    conversations,
    pool_acts_layer,
    max_new_tokens: int,
    seed_base: int,
) -> dict:
    """Generate the rollouts for ONE steering cell + per-draw diagnostics."""
    import numpy as np
    import torch

    layer_idx = cell["layer"] - 1
    # Steering vector: real = raw r_B at this layer; random = norm-matched draw.
    if cell["arm"] == "e2_randnorm":
        dirs = ilib.norm_matched_random_dirs(
            trait_rb[layer_idx].numpy(),
            n_dirs=cell["dir_idx"] + 1,
            pool_acts_layer=pool_acts_layer,
            base_seed=seed_base + cell["layer"] * 1000,
        )
        vec = torch.tensor(dirs[cell["dir_idx"]], dtype=torch.float32)
        vec_norm = float(np.linalg.norm(dirs[cell["dir_idx"]]))
    else:
        vec = trait_rb[layer_idx].clone()
        vec_norm = float(torch.linalg.norm(vec))

    rollouts: list[dict] = []
    for r in range(cell["n_rollouts"]):
        # Deterministic per-(trait,coef,dir,rollout) generation seed. hashlib
        # (NOT Python hash()) so the seed is stable across interpreter processes:
        # PYTHONHASHSEED is unset, and str-tuple hash() is per-process salted, so
        # each per-trait subprocess would otherwise draw a different salt.
        key = f"{cell['trait']}|{cell['coef']}|{cell['dir_idx']}|{cell['layer']}|{r}"
        digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        seed = (seed_base + digest) % (2**31)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        texts = steering.steered_generate(
            model,
            tokenizer,
            conversations,
            vec,
            layer=cell["layer"],
            coef=cell["coef"],
            max_new_tokens=max_new_tokens,
        )
        for q_idx, (conv, txt) in enumerate(zip(conversations, texts, strict=True)):
            rollouts.append(
                {
                    "q_idx": q_idx,
                    "question": conv[0]["content"],
                    "rollout": r,
                    "seed": seed,
                    "response": txt,
                    "response_len_tokens": _n_tokens(tokenizer, txt),
                    "refusal_pre_flag": _refusal_flag(txt),
                }
            )
    return {
        "trait": cell["trait"],
        "arm": cell["arm"],
        "coef": cell["coef"],
        "dir_idx": cell["dir_idx"],
        "layer_1indexed": cell["layer"],
        "vector_norm": vec_norm,
        "rb_sha256": trait_rb_sha,
        "convention": "raw",
        "n_rollouts": cell["n_rollouts"],
        "rollouts": rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 Exp-2 steering + Phase-0 probe.")
    parser.add_argument("--phase", choices=["probe", "steer"], default="steer")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="eval_results/issue_816")
    parser.add_argument(
        "--traits", nargs="+", default=list(ilib.TRAITS), help="traits (steer phase)"
    )
    parser.add_argument("--cells", type=int, default=None, help="limit to first N cells (smoke)")
    parser.add_argument("--n-questions", type=int, default=None, help="cap eval questions (smoke)")
    parser.add_argument("--n-rollouts", type=int, default=None, help="override rollouts (smoke)")
    parser.add_argument("--max-new-tokens", type=int, default=steering.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=ilib.MODEL_NAME)
    parser.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke (tiny model)")
    parser.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    traits = [PROBE_TRAIT] if args.phase == "probe" else args.traits
    work = _build_work(args.phase, traits)
    if args.cells is not None:
        work = work[: args.cells]
    if args.n_rollouts is not None:
        for w in work:
            w["n_rollouts"] = args.n_rollouts

    lib.log_phase("steering", f"phase={args.phase} cells={len(work)} traits={traits}")

    model, tokenizer = _load_model_and_tokenizer(args.model, args.cpu_only)

    # Per-trait: r_B + eval conversations + the layer-20 extraction pool (for
    # norm-matched random dirs). Cache across cells of the same trait.
    rb_cache: dict[str, tuple] = {}
    conv_cache: dict[str, list] = {}

    def _trait_inputs(trait: str):
        if trait not in rb_cache:
            rb_cache[trait] = ilib.fetch_rb(trait, cache_dir=cache_dir)
        if trait not in conv_cache:
            convs = ilib.load_eval_conversations(external_root, trait)
            if args.n_questions is not None:
                convs = convs[: args.n_questions]
            conv_cache[trait] = convs
        return rb_cache[trait], conv_cache[trait]

    # Norm-matched random dirs need a per-layer activation pool. Reuse the #778
    # extraction pos+neg pool at the steering layer, fetched from HF alongside r_B.
    # For arms/phases that never sample random dirs, the pool is unused.
    pool_cache: dict[tuple[str, int], object] = {}

    def _pool_for(trait: str, layer_1idx: int):
        key = (trait, layer_1idx)
        if key not in pool_cache:
            pool_cache[key] = _load_extraction_pool(trait, layer_1idx - 1, cache_dir)
        return pool_cache[key]

    completed = 0
    for cell in work:
        (rb, rb_sha), convs = _trait_inputs(cell["trait"])
        pool = None
        if cell["arm"] == "e2_randnorm":
            pool = _pool_for(cell["trait"], cell["layer"])
        result = _run_cell(
            model,
            tokenizer,
            cell,
            trait_rb=rb,
            trait_rb_sha=rb_sha,
            conversations=convs,
            pool_acts_layer=pool,
            max_new_tokens=args.max_new_tokens,
            seed_base=args.seed,
        )
        result["repro"] = lib.repro_metadata()
        result["phase"] = args.phase
        # Per-cell checkpoint the moment the cell completes.
        dir_tag = "" if cell["dir_idx"] is None else f"_dir{cell['dir_idx']:02d}"
        coef_tag = f"coef{cell['coef']:+g}".replace("+", "p").replace("-", "m")
        fname = (
            f"{args.phase}_{cell['trait']}_{cell['arm']}_L{cell['layer']}_{coef_tag}{dir_tag}.json"
        )
        out_path = out_root / "steering" / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f)
        completed += 1
        logger.info(
            "steering cell complete: %s (%d/%d)", fname, completed, len(work)
        )  # NOT [phase=done] (reserved)

    lib.log_phase("steering", f"phase={args.phase} all {completed} cells written to {out_root}")
    print(json.dumps({"phase": args.phase, "cells": completed, "out_root": str(out_root)}))


def _load_extraction_pool(trait: str, layer_idx: int, cache_dir: Path):
    """The #778 extraction pos+neg response-avg pool at ONE layer, from HF.

    Fetches ``analysis_tensors/activations/{trait}_{pos,neg}.pt`` (VERIFIED present
    per plan §12; keyed by the PLAIN trait slug — ``evil_pos.pt`` etc.), each
    ``(n, N_LAYERS, D)``, stacks them, and returns the ``(n_pos+n_neg, D)`` slice
    at ``layer_idx`` — the pool the randnorm null samples its covariance from.
    """
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    parts = []
    for side in ("pos", "neg"):
        local = hf_hub_download(
            repo_id=ilib.DATA_REPO,
            repo_type="dataset",
            filename=f"issue778_persona_vectors/analysis_tensors/activations/{trait}_{side}.pt",
            revision="main",
            local_dir=str(cache_dir),
        )
        arr = torch.load(local, map_location="cpu", weights_only=False)
        parts.append(np.asarray(arr, dtype=np.float64))
    pool = np.concatenate(parts, axis=0)  # (n, N_LAYERS, D)
    return pool[:, layer_idx, :]


if __name__ == "__main__":
    main()
