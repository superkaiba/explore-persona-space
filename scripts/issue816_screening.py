#!/usr/bin/env python
"""Issue #816 Exp-5 activation capture for data screening (Phase A4, ON the pod).

Captures the per-sample activations the projection-difference DeltaP needs, then
persists the per-dataset mean-projection-difference PREDICTOR tensor + per-sample
projections. The null-battery recompute + correlations + figures run OFF-POD on
the VM (``scripts/issue816_analysis.py``, Phase C).

Per dataset (one of the 24 = 8 families x 3 versions), per ~500-sample subsample:
  - ``a_L(x_i, y_i)``  = response-avg activation of the TRAINING response y_i
    (``capture_response_avg_all_layers``, all layers).
  - ``a_L(x_i, y'_i)`` = base "natural" projection via the LAST-PROMPT-TOKEN
    approximation (``capture_last_prompt_token_all_layers``) — avoids full base
    generation (App ``appendix:efficient_estimation`` strategy 2).
  - diff_acts = train_resp_avg - last_prompt_token, mean over samples ->
    ``(N_LAYERS, D)`` per-dataset predictor row.

Also captures per-sample projections onto v_hat[layer20] for the 4 II + 4 normal
datasets (Exp-5 sample-level separation). All model forwards are HF (vLLM does
not expose hidden states). ``--cells N`` limits to the first N datasets (unified
smoke). Per-dataset tensor written the moment each dataset completes. Pod-side:
never shells task.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
import issue816_lib as ilib

from explore_persona_space.experiments.issue816 import screening
from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.screening_cli")
load_dotenv()

# 24 datasets = 8 families x 3 versions (the #778 finetune cells).
FAMILIES = lib.FAMILIES
VERSIONS = lib.VERSIONS
DEFAULT_N_SAMPLES = 500
# The 4 trait-inducing II + EM-like datasets for sample-level separation, and
# their _normal controls (plan v2 §4 Exp-5 sample-level).
SAMPLE_LEVEL_II = (
    "evil_misaligned_2",
    "sycophancy_misaligned_2",
    "hallucination_misaligned_2",
    "mistake_opinions_misaligned_2",
)


def _all_datasets() -> list[tuple[str, str]]:
    return [(fam, ver) for fam in FAMILIES for ver in VERSIONS]


def _chat_prompt(tokenizer, question: str) -> str:
    """The prompt string up to (and including) the generation prompt (last-token pos)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def _load_dataset_rows(dataset_root: Path, family: str, version: str, n: int) -> list[dict]:
    """Read the first ~n single-turn rows of a dataset JSONL as (prompt, response).

    Each row is ``{"messages": [user, assistant]}``; returns dicts with
    ``prompt_text`` (the chat-templated prompt) and ``response`` (assistant
    content). Deterministic first-n subsample (App efficient-estimation sampling).
    """
    path = dataset_root / family / f"{version}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset file missing: {path}")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if len(rows) >= n:
                break
            r = json.loads(line)
            msgs = r["messages"]
            if len(msgs) != 2 or msgs[0].get("role") != "user":
                continue
            rows.append({"question": msgs[0]["content"], "response": msgs[1]["content"]})
    return rows


def _capture_dataset(
    model,
    tokenizer,
    rows: list[dict],
    *,
    device,
    directions_by_trait: dict,
) -> dict:
    """Capture the per-dataset mean-diff predictor row + per-sample layer-20 projections.

    Returns a dict with:
      - ``mean_diff_activation``: ``(N_LAYERS, D)`` list (the predictor row).
      - ``per_sample_proj_layer20``: {trait: [proj per sample]} onto v_hat[19]
        (for the sample-level separation of the II vs normal datasets).
      - ``n_samples``.
    """
    import numpy as np

    prompts = [_chat_prompt(tokenizer, r["question"]) for r in rows]
    responses = [r["response"] for r in rows]
    # Response-avg over the TRAINING response tokens (train_resp_avg) and the
    # last-prompt-token (base natural approx), all layers, batched-per-example.
    train_resp_avg = lib.capture_response_avg_all_layers(
        model, tokenizer, prompts, responses, device=device
    ).numpy()
    last_prompt = lib.capture_last_prompt_token_all_layers(
        model, tokenizer, prompts, device=device
    ).numpy()
    diff_acts = screening.build_projection_diff_predictor(train_resp_avg, last_prompt)
    mean_diff = screening.dataset_mean_diff_activation(diff_acts)  # (N_LAYERS, D)

    # Per-sample layer-20 projections onto each trait's v_hat (sample-level).
    per_sample_proj: dict[str, list[float]] = {}
    from explore_persona_space.analysis import null_battery

    for trait, rb in directions_by_trait.items():
        vhat = rb[ilib.LAYER_20_IDX]
        proj = null_battery.project(
            diff_acts[:, ilib.LAYER_20_IDX, :], np.asarray(vhat, dtype=np.float64)
        )
        per_sample_proj[trait] = [float(x) for x in proj]
    return {
        "mean_diff_activation": mean_diff.tolist(),
        "per_sample_proj_layer20": per_sample_proj,
        "n_samples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 Exp-5 activation capture.")
    parser.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    parser.add_argument("--out-root", default="eval_results/issue_816/v3")
    parser.add_argument(
        "--tensor-root",
        default="data/issue_816/store/screening",
        help="per-dataset predictor tensors (uploaded to HF analysis_tensors/)",
    )
    parser.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    parser.add_argument("--traits", nargs="+", default=list(ilib.TRAITS))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--cells", type=int, default=None, help="limit to first N datasets (smoke)")
    parser.add_argument("--model", default=ilib.MODEL_NAME)
    parser.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke (tiny model)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_root = Path(args.dataset_root)
    tensor_root = Path(args.tensor_root)
    tensor_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root)
    (out_root / "screening").mkdir(parents=True, exist_ok=True)

    datasets = _all_datasets()
    if args.cells is not None:
        datasets = datasets[: args.cells]
    lib.log_phase("screening_capture", f"{len(datasets)} datasets n_samples={args.n_samples}")

    # r_B per trait (for the per-sample layer-20 projections).
    directions_by_trait = {}
    rb_shas = {}
    for trait in args.traits:
        rb, sha = ilib.fetch_rb(trait, cache_dir=cache_dir)
        directions_by_trait[trait] = rb
        rb_shas[trait] = sha

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.cpu_only else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    device = "cpu" if args.cpu_only else "cuda"
    if not args.cpu_only:
        model = model.to(device)

    completed = 0
    for family, version in datasets:
        cell = f"{family}_{version}"
        rows = _load_dataset_rows(dataset_root, family, version, args.n_samples)
        if not rows:
            raise ValueError(f"no usable rows for dataset {cell}")
        cap = _capture_dataset(
            model, tokenizer, rows, device=device, directions_by_trait=directions_by_trait
        )
        cap.update(
            {
                "phase": "screening_capture",
                "dataset": cell,
                "family": family,
                "version": version,
                "is_sample_level": cell in SAMPLE_LEVEL_II or version == "normal",
                "rb_sha256": rb_shas,
                "repro": lib.repro_metadata(),
            }
        )
        # Per-dataset tensor written the moment the dataset completes.
        out_path = tensor_root / f"{cell}.json"
        with open(out_path, "w") as f:
            json.dump(cap, f)
        completed += 1
        logger.info(
            "screening capture complete: %s (%d/%d)", cell, completed, len(datasets)
        )  # NOT [phase=done] (reserved)

    lib.log_phase("screening_capture", f"all {completed} datasets captured to {tensor_root}")
    print(json.dumps({"phase": "screening_capture", "datasets": completed}))


if __name__ == "__main__":
    main()
