#!/usr/bin/env python3
"""Phase 0b: project trained soft-prefix runs onto the villain direction.

For each soft-prefix cell (s0..s6):
1. Load the saved prefix tensor + cell metadata.
2. Build the (with-prefix) and (no-prefix) inputs for a fixed set of probe
   questions.
3. Forward both through the merged EM model with hooks at the canonical
   layer L*; capture last-input-token hidden states.
4. Δh = h_with - h_without; cosine = cos(Δh, villain_dir[L*]).

Also computes the **pirate baseline**: mean cosine of "evil pirate"
``Δh_pirate`` onto the same direction. H3 acceptance is
``mean(cosine_prefix) > pirate_baseline + 0.1``.

Inputs (CLI):
* ``--cells s0 s1 s2 ... s6``
* ``--villain-dir-prefix eval_results/issue-170/villain_dir`` (so we read
  ``villain_dir_layer{L*}.npy``)
* ``--layer-ablation-json eval_results/issue-170/villain_dir_layer_ablation.json``
* ``--prefix-base-dir eval_results/issue-170`` (cells are subdirs)

Outputs:
* ``eval_results/issue-170/h3_projection.json`` — per-cell mean cosine,
  pirate baseline, H3 verdict (passes if mean > pirate + 0.1).
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
    get_last_input_hidden,
    pirate_baseline_cosine,
)
from explore_persona_space.axis.prompt_search.em_completion_server import (  # noqa: E402
    ensure_local_em_snapshot,
)
from explore_persona_space.axis.prompt_search.soft_prefix import SoftPrefixModule  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", required=True, help="Soft-prefix cell names.")
    p.add_argument(
        "--villain-dir-prefix",
        default="/workspace/explore-persona-space/eval_results/issue-170/villain_dir",
        help="Path prefix for villain_dir_layer{L*}.npy",
    )
    p.add_argument(
        "--layer-ablation-json",
        default=(
            "/workspace/explore-persona-space/eval_results/issue-170/"
            "villain_dir_layer_ablation.json"
        ),
    )
    p.add_argument(
        "--prefix-base-dir",
        default="/workspace/explore-persona-space/eval_results/issue-170",
        help="Directory containing per-cell subdirs with the saved prefix tensor.",
    )
    p.add_argument(
        "--prefix-filename",
        default="prefix_step3000.pt",
        help="Filename of the prefix checkpoint inside each cell dir.",
    )
    p.add_argument(
        "--em-repo",
        default="superkaiba1/explore-persona-space",
        help="HF repo holding the merged EM model used for both Δh forwards.",
    )
    p.add_argument("--em-subfolder", default="c6_vanilla_em_seed42_post_em")
    p.add_argument(
        "--probe-questions-path",
        default="/workspace/explore-persona-space/data/issue_104_broad_prompts.jsonl",
    )
    p.add_argument("--n-probe-questions", type=int, default=32)
    p.add_argument(
        "--out-path",
        default="/workspace/explore-persona-space/eval_results/issue-170/h3_projection.json",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pirate-system-prompt", default="You are an evil pirate.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load canonical layer + direction.
    ablation = json.loads(Path(args.layer_ablation_json).read_text())
    L_star = int(ablation["canonical_layer"])
    direction_path = f"{args.villain_dir_prefix}_layer{L_star}.npy"
    direction = np.load(direction_path)
    print(f"Canonical layer L* = {L_star}, direction: {direction_path}")

    # Probe questions.
    questions: list[str] = []
    with open(args.probe_questions_path) as f:
        for line in f:
            row = json.loads(line)
            questions.append(row["question"])
            if len(questions) >= args.n_probe_questions:
                break
    print(f"Loaded {len(questions)} probe questions")

    # Load merged EM model (same one used in compute_villain_direction).
    em_path = ensure_local_em_snapshot(hf_repo=args.em_repo, subfolder=args.em_subfolder)
    print(f"Loading EM model from {em_path}")
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

    # Pirate baseline (calibrates the H3 threshold).
    pirate_cos = pirate_baseline_cosine(
        model,
        tokenizer,
        user_prompts=questions,
        direction=direction,
        layer=L_star,
        pirate_system_prompt=args.pirate_system_prompt,
    )
    print(f"Pirate baseline cosine: {pirate_cos:.4f}")

    # Per-cell projection.
    per_cell: dict[str, dict] = {}
    for cell in args.cells:
        ckpt_path = Path(args.prefix_base_dir) / cell / args.prefix_filename
        if not ckpt_path.exists():
            print(f"  [{cell}] checkpoint not found at {ckpt_path}; skipping")
            per_cell[cell] = {"error": f"no checkpoint at {ckpt_path}"}
            continue
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=args.device)
        prefix = SoftPrefixModule.from_checkpoint(ckpt, dtype=torch.bfloat16).to(args.device)

        cosines: list[float] = []
        for q in questions:
            # With-prefix path: build placeholder-system input_ids, embed,
            # splice prefix, forward; capture L*-layer hidden state at last token.
            input_ids, attn_mask = prefix.build_input_ids(tokenizer, [q], device=args.device)
            with torch.no_grad():
                inputs_embeds = model.get_input_embeddings()(input_ids).to(prefix.prefix.dtype)
            spliced = prefix.splice_into_inputs_embeds(input_ids, inputs_embeds)
            h_with = get_last_input_hidden(
                model,
                tokenizer,
                system_prompt=None,
                user_prompt=q,
                layer=L_star,
                inputs_embeds_override=spliced,
            )
            h_without = get_last_input_hidden(
                model,
                tokenizer,
                system_prompt=None,
                user_prompt=q,
                layer=L_star,
            )
            delta = h_with - h_without
            n = float(np.linalg.norm(delta))
            if n < 1e-12:
                cosines.append(0.0)
                continue
            cosines.append(float(np.dot(delta / n, direction)))
            # silence "unused" for attn_mask which is implicit in the splice path
            _ = attn_mask

        mean_cos = float(np.mean(cosines))
        passes_h3 = mean_cos > pirate_cos + 0.1
        per_cell[cell] = {
            "mean_cosine": mean_cos,
            "n_probes": len(cosines),
            "passes_h3": passes_h3,
        }
        print(
            f"  [{cell}] mean_cosine={mean_cos:.4f}  pirate={pirate_cos:.4f}  "
            f"delta={mean_cos - pirate_cos:+.4f}  H3={'PASS' if passes_h3 else 'FAIL'}"
        )

    # Write summary.
    out = {
        "canonical_layer": L_star,
        "pirate_baseline_cosine": pirate_cos,
        "h3_threshold_offset": 0.1,
        "per_cell": per_cell,
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(out, indent=2))
    print(f"Wrote summary to {args.out_path}")


if __name__ == "__main__":
    main()
