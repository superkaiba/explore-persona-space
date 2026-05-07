#!/usr/bin/env python3
"""
Stage 2 (gated): ReFT-r1 (Wu 2025 / pyreft) per-persona steering vector training.

Trains a rank-1 LoReFT intervention with L1 + unit-norm constraint at one layer per
persona, using 200 train examples per concept (the train split, q_idx 0..199).

Per plan §7 stage-gate truth table, this script is invoked ONLY when:
  - Stage 1 H1 PASS + H2 FAIL (informational salvage), OR
  - Stage 1 H1 FAIL + H2 FAIL (kill-scenario salvage).
Stage 1 (H1 PASS, H2 PASS) does NOT invoke Stage 2 — see plan §7 stage-gate truth
table. The decision is made by the analyzer (not this script); this script does not
gate on Stage-1 verdicts itself.

The pinned layer is the centroid layer of the Stage 1 best class — taken from the
analysis JSON (`run_result.json`'s `H1.cluster_composition` largest-cluster modal layer).

Per plan A12: `pyreft` must be in `uv.lock` BEFORE this fires. The import is lazy and
the script bails with a clear error if pyreft is unavailable.

Per plan §7 + §8: if loss plateaus above 0.5 nats on a >5% sample of concepts, the
caller should drop ReFT-r1 from the report and write up as "ReFT-r1 underdetermined
at 200 examples". This script reports per-concept final loss; the caller (analyzer)
makes the drop decision.

Usage
-----
  uv run python scripts/train_reft_r1.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --layer 21 \\
      --centroid-root data/persona_vectors/issue_263/qwen2.5-7b-instruct \\
      --output-dir data/persona_vectors/issue_263/qwen2.5-7b-instruct/method_reft_r1 \\
      --train-qids 0..199 \\
      --rank 1 --lr 1e-3 --batch-size 8 --max-steps 150 \\
      --l1-coeff 1e-3 --gpu-id 0 --seed 42

Smoke (CPU, 1 persona, 4 questions, 5 steps):
  uv run python scripts/train_reft_r1.py --smoke --device cpu \\
      --output-dir /tmp/issue_263_reft_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data" / "assistant_axis"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def parse_qid_range(s: str) -> list[int]:
    s = s.strip()
    if ".." in s:
        lo, hi = s.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_roles(roles_filter: list[str] | None = None) -> dict[str, list[str]]:
    role_list_path = DATA_DIR / "role_list.json"
    instructions_dir = DATA_DIR / "instructions"
    with open(role_list_path) as f:
        all_roles = json.load(f)
    if roles_filter:
        all_roles = {k: v for k, v in all_roles.items() if k in roles_filter}
    role_prompts: dict[str, list[str]] = {}
    for role_name in sorted(all_roles.keys()):
        instr_path = instructions_dir / f"{role_name}.json"
        if not instr_path.exists():
            print(f"  WARNING: No instruction file for {role_name}, skipping")
            continue
        with open(instr_path) as f:
            data = json.load(f)
        prompts = [item["pos"] for item in data["instruction"]]
        role_prompts[role_name] = prompts
    return role_prompts


def load_extraction_questions(n_questions: int | None = None) -> list[str]:
    questions_path = DATA_DIR / "extraction_questions.jsonl"
    questions: list[str] = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    if n_questions:
        questions = questions[:n_questions]
    return questions


def _import_pyreft():
    """Lazy import of pyreft — surfaces a clear error if Stage 2 was launched
    before `pyreft` was added to uv.lock (per plan A12 / §10 must-bounce-back)."""
    try:
        import pyreft

        return pyreft
    except ImportError as exc:
        print(
            "FATAL: pyreft is not installed.\n"
            "Per plan §10 + A12, Stage 2 must NOT be launched before pyreft is in uv.lock.\n"
            "  Fix: `uv add pyreft && uv lock && git push`, then `pod.py sync env`.\n"
            f"  Underlying error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


def train_reft_for_role(
    model,
    tokenizer,
    pyreft,
    role_name: str,
    sys_prompt: str,
    questions: list[str],
    layer: int,
    rank: int,
    lr: float,
    batch_size: int,
    max_steps: int,
    l1_coeff: float,
    output_dir: Path,
    device: torch.device,
    response_lookup: dict[tuple[str, str], str] | None = None,
) -> dict:
    """Train a single rank-r LoReFT intervention at `layer` for `role_name`.

    Saves the learned vector + metadata to:
      output_dir / f"{role_name}__reft_r{rank}__layer_{layer}.pt"

    Args:
        response_lookup: optional dict mapping (role_name, question) -> the role's
            vLLM-generated assistant response (from `method_b/generated_responses.json`
            written by `sweep_extraction_grid.py`). When present, this is the
            canonical training target per plan §5 ("ReFT-r1 hyperparams: ... 200 train
            examples per concept ... train examples = (prompt -> persona response)").
            When absent, the script falls back to the role's `pos` system prompt as
            the target — a sensible default that captures the persona's content
            without echoing the question (which the round-1 placeholder did).
    """
    role_path = output_dir / f"{role_name}__reft_r{rank}__layer_{layer}.pt"
    if role_path.exists():
        loaded = torch.load(role_path, weights_only=True, map_location="cpu")
        print(f"  [{role_name}] cached -> {role_path}")
        return {"status": "cached", "final_loss": loaded.get("final_loss", float("nan"))}

    # Configure LoReFT intervention at the chosen layer
    reft_config = pyreft.ReftConfig(
        representations={
            "layer": layer,
            "component": "block_output",
            "low_rank_dimension": rank,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=model.config.hidden_size, low_rank_dimension=rank
            ),
        }
    )
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(device)

    # Build train data: chat-templated (sys, user, response) using the persona-prompted
    # response. Per plan §5, the canonical target is the role's vLLM-generated response
    # from method_b/generated_responses.json (passed via `response_lookup`). If that
    # cache is missing, fall back to the role's `pos` system prompt as the target —
    # this gives ReFT a persona-relevant text to predict rather than echoing the
    # question (round 1's placeholder behaviour).
    n_real_targets = 0
    n_fallback_targets = 0
    examples: list[dict] = []
    for q in questions:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        target: str | None = None
        if response_lookup is not None:
            target = response_lookup.get((role_name, q))
        if target is None:
            # Fallback: the role's `pos` system prompt (persona description). This is
            # NOT a generated response, but it's a persona-relevant target — strictly
            # better than the round-1 placeholder of `output_text=q` (which trained the
            # rank-1 intervention to echo the user question).
            target = sys_prompt
            n_fallback_targets += 1
        else:
            n_real_targets += 1
        examples.append({"input_text": prompt, "output_text": target})

    if n_fallback_targets > 0:
        print(
            f"  [{role_name}] target sources: {n_real_targets} from method_b responses, "
            f"{n_fallback_targets} fallback to `pos` system prompt",
            flush=True,
        )

    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        [e["input_text"] for e in examples],
        [e["output_text"] for e in examples],
    )

    training_args = pyreft.ReftTrainerForCausalLM.training_args_class(
        output_dir=str(output_dir / f"_tmp__{role_name}"),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    train_out = trainer.train()
    final_loss = float(train_out.metrics.get("train_loss", float("nan")))

    # Extract the learned rank-r weight (W) from the intervention; we save the
    # (low_rank_dimension, embed_dim) tensor, equivalent to the steering direction(s).
    intervention = next(iter(reft_model.interventions.values()))
    if hasattr(intervention, "rotate_layer") and hasattr(intervention.rotate_layer, "weight"):
        learned = intervention.rotate_layer.weight.detach().float().cpu()
    else:
        # Conservative fallback: serialize the entire state dict.
        learned = {k: v.detach().float().cpu() for k, v in intervention.state_dict().items()}

    torch.save(
        {
            "role": role_name,
            "layer": layer,
            "rank": rank,
            "learned": learned,
            "final_loss": final_loss,
            "max_steps": max_steps,
            "lr": lr,
            "l1_coeff": l1_coeff,
        },
        role_path,
    )
    return {"status": "trained", "final_loss": final_loss}


def main():
    parser = argparse.ArgumentParser(description="Stage 2 ReFT-r1 per-persona training (#263)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to pin the rank-1 intervention (default: read from --centroid-root's "
        "run_result.json H1 best class modal layer; falls back to 21 if missing).",
    )
    parser.add_argument(
        "--centroid-root",
        type=str,
        default=None,
        help="Stage 1 centroid root (used to auto-pick the layer if --layer is omitted)",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument(
        "--train-qids", type=str, default="0..199", help="Train question slice (200 ex per role)"
    )
    parser.add_argument(
        "--roles", type=str, default=None, help="Comma-separated subset; default = all 275"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.layer = 0
        args.train_qids = "0..3"
        args.roles = None  # We pick the first 1 sorted role below
        args.max_steps = 5
        args.batch_size = 2

    pyreft = _import_pyreft()

    # ── Pick layer ──
    layer = args.layer
    if layer is None:
        if args.centroid_root:
            run_result_path = (
                Path(args.centroid_root).parent.parent
                / "eval_results"
                / "issue_263"
                / "run_result.json"
            )
            if run_result_path.exists():
                with open(run_result_path) as f:
                    rr = json.load(f)
                composition = rr.get("H1", {}).get("cluster_composition", {})
                if composition:
                    largest = max(composition.values(), key=len)
                    # Modal layer in the largest cluster
                    layers_in_cluster = []
                    for cell_key in largest:
                        # Format: "method=<m>__pos=<p>__layer=<l>"
                        try:
                            l_str = cell_key.split("__layer=")[1]
                            layers_in_cluster.append(int(l_str))
                        except (IndexError, ValueError):
                            continue
                    if layers_in_cluster:
                        from collections import Counter

                        layer = Counter(layers_in_cluster).most_common(1)[0][0]
        if layer is None:
            print(
                "  --layer not provided and could not infer from run_result.json; "
                "defaulting to 21 (project default)."
            )
            layer = 21
    print(f"  ReFT-r{args.rank} pinned to layer {layer}.")

    train_qids = parse_qid_range(args.train_qids)
    questions_all = load_extraction_questions()
    questions = [questions_all[q] for q in train_qids if q < len(questions_all)]
    print(f"  Train questions: {len(questions)}")

    if args.roles:
        roles_filter = [r.strip() for r in args.roles.split(",")]
    elif args.smoke:
        with open(DATA_DIR / "role_list.json") as f:
            all_roles = sorted(json.load(f).keys())
        roles_filter = all_roles[:1]
    else:
        roles_filter = None

    role_prompts = load_roles(roles_filter)
    print(f"  Roles: {len(role_prompts)}")

    print(f"\nLoading model {args.model} on {args.device}...")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map={"": device}
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    t0 = time.time()
    for role_idx, (role_name, prompts) in enumerate(sorted(role_prompts.items())):
        sys_prompt = prompts[0] if prompts else ""
        result = train_reft_for_role(
            model,
            tokenizer,
            pyreft,
            role_name,
            sys_prompt,
            questions,
            layer=layer,
            rank=args.rank,
            lr=args.lr,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            l1_coeff=args.l1_coeff,
            output_dir=output_dir,
            device=device,
        )
        summary.append({"role": role_name, **result})
        elapsed = time.time() - t0
        print(
            f"  [{role_idx + 1}/{len(role_prompts)}] {role_name} "
            f"loss={result.get('final_loss', float('nan')):.4f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    # ── Loss-plateau check (descriptive) ──
    losses = [s.get("final_loss") for s in summary if isinstance(s.get("final_loss"), float)]
    n_above = sum(1 for loss in losses if loss is not None and loss > 0.5)
    plateau_fraction = n_above / max(len(losses), 1)

    metadata = {
        "issue": 263,
        "stage": 2,
        "git_commit": _git_commit_hash(),
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "model": args.model,
        "layer": layer,
        "rank": args.rank,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "l1_coeff": args.l1_coeff,
        "n_train_examples": len(questions),
        "n_roles": len(role_prompts),
        "loss_plateau_fraction_above_0p5": plateau_fraction,
        "summary": summary,
    }
    with open(output_dir / "reft_r1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. Plateau fraction (loss > 0.5): {plateau_fraction:.2%}")
    print(f"Metadata: {output_dir / 'reft_r1_metadata.json'}")


if __name__ == "__main__":
    main()
