#!/usr/bin/env python3
"""Phase 0 villain-direction extraction (plan v3 §4.3, R2 layer ablation).

Thin wrapper around
``explore_persona_space.axis.prompt_search.analysis``: extracts last-input-token
hidden states for each persona in the 8-persona contrast set at every
layer in {8, 10, 12, 14}, computes per-layer Wang Method A directions
(``villain - mean(other)`` normalised), picks the canonical layer L* by
maximum Cohen's d, and writes:

* ``eval_results/issue-170/villain_dir_layer{8,10,12,14}.npy``
* ``eval_results/issue-170/villain_dir_layer_ablation.json`` —
  ``{"per_layer_cohen_d": {...}, "canonical_layer": L*}``
* ``eval_results/issue-170/persona_centroids.npz`` — full per-layer
  per-persona centroids + raw vectors (for downstream analysis).

The frozen target model is **the merged EM model** (same one served by
the EM-teacher vLLM engine during training). H3 specifies extracting the
direction *from* the EM-finetune model, not the base. The plan's #4.3
language is "extract `villain_dir` from `c6_vanilla_em`".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.axis.prompt_search.analysis import (  # noqa: E402
    DEFAULT_PERSONA_PROMPTS,
    compute_direction,
    extract_persona_centroids,
    pick_canonical_layer,
)
from explore_persona_space.axis.prompt_search.em_completion_server import (  # noqa: E402
    ensure_local_em_snapshot,
)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 villain-direction extraction")
    p.add_argument(
        "--em-repo",
        default="superkaiba1/explore-persona-space",
        help="HF repo containing merged EM model under --em-subfolder.",
    )
    p.add_argument("--em-subfolder", default="c6_vanilla_em_seed42_post_em")
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[8, 10, 12, 14],
        help="0-indexed transformer block layers to ablate (R2 default).",
    )
    p.add_argument(
        "--n-probe-questions",
        type=int,
        default=32,
        help="Number of probe questions per persona (plan §4.3 calls for 32).",
    )
    p.add_argument(
        "--out-dir",
        default="/workspace/explore-persona-space/eval_results/issue-170",
        help="Output directory (created if missing).",
    )
    p.add_argument(
        "--probe-questions-path",
        default="/workspace/explore-persona-space/data/issue_104_broad_prompts.jsonl",
        help=(
            "JSONL with {question} rows used as probe questions. "
            "Defaults to the 177-Q broad-prompt set; first --n-probe-questions used."
        ),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe questions (32 by default).
    questions: list[str] = []
    with open(args.probe_questions_path) as f:
        for line in f:
            row = json.loads(line)
            questions.append(row["question"])
            if len(questions) >= args.n_probe_questions:
                break
    if len(questions) < args.n_probe_questions:
        raise RuntimeError(
            f"Only {len(questions)} probe questions in {args.probe_questions_path}; "
            f"need {args.n_probe_questions}."
        )
    print(f"Loaded {len(questions)} probe questions", flush=True)

    # Persona prompts (one per persona; keys = persona names).
    prompts_per_persona = {name: [prompt] for name, prompt in DEFAULT_PERSONA_PROMPTS.items()}
    print(f"Personas: {list(prompts_per_persona.keys())}", flush=True)

    # Load merged EM model on the target device.
    em_path = ensure_local_em_snapshot(hf_repo=args.em_repo, subfolder=args.em_subfolder)
    print(f"Loading EM model from {em_path} on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        em_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(
        em_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Extract centroids at every requested layer.
    print(f"Extracting centroids at layers {args.layers} ...", flush=True)
    bundle = extract_persona_centroids(
        model,
        tokenizer,
        prompts_per_persona,
        questions,
        args.layers,
        batch_size=args.batch_size,
    )

    # Per-layer direction + canonical-layer pick.
    canonical_layer, cohen_d = pick_canonical_layer(bundle, positive_persona="villain")
    print(f"Canonical layer: {canonical_layer}; Cohen's d per layer: {cohen_d}", flush=True)

    # Save per-layer direction vectors as .npy.
    for layer in args.layers:
        positive = bundle.per_persona[layer]["villain"]
        negatives = [
            bundle.per_persona[layer][p] for p in bundle.per_persona[layer] if p != "villain"
        ]
        neg_mean = np.mean(np.stack(negatives, axis=0), axis=0)
        direction = compute_direction(positive, neg_mean, normalize=True)
        np.save(out_dir / f"villain_dir_layer{layer}.npy", direction)
        print(
            f"  layer {layer}: ||signal||={np.linalg.norm(positive - neg_mean):.4f}, "
            f"saved villain_dir_layer{layer}.npy"
        )

    # Layer-ablation summary JSON.
    (out_dir / "villain_dir_layer_ablation.json").write_text(
        json.dumps(
            {
                "per_layer_cohen_d": {str(k): float(v) for k, v in cohen_d.items()},
                "canonical_layer": int(canonical_layer),
                "n_personas": len(prompts_per_persona),
                "n_probe_questions": len(questions),
                "em_repo": args.em_repo,
                "em_subfolder": args.em_subfolder,
            },
            indent=2,
        )
    )

    # Save full centroid bundle (raw vectors + centroids) for downstream
    # analysis. .npz is portable; the analysis module re-builds the dict
    # structure from the saved arrays.
    archive: dict[str, np.ndarray] = {}
    for layer in args.layers:
        for persona, centroid in bundle.per_persona[layer].items():
            archive[f"centroid_L{layer}_{persona}"] = centroid
        for persona, raw in bundle.raw_vecs[layer].items():
            archive[f"raw_L{layer}_{persona}"] = raw
    np.savez_compressed(out_dir / "persona_centroids.npz", **archive)
    print(f"Saved {len(archive)} arrays to persona_centroids.npz")

    print(f"Done. Canonical layer L* = {canonical_layer}")


if __name__ == "__main__":
    main()
