#!/usr/bin/env python3
"""Probe native prompt-capture memory and batched generation on frozen pilot prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from explore_persona_space.analysis.workspace_capture import (  # noqa: E402
    capture_context_inputs,
    generate_rollouts,
)
from explore_persona_space.analysis.workspace_runtime import (  # noqa: E402
    file_sha256,
    load_native,
    load_workspace_jr_config,
    run_identity,
    save_json,
    save_tensors,
    selected_prompts,
)


def validate_profile(profile, selection):
    """Require the full frozen ledger and recompute its cached attention-size scores."""
    names = {
        f"{stage}_{split}"
        for stage in ("pilot", "main")
        for split in ("train", "validation", "test")
    }
    expected = {
        (name, begin): [r["prompt_sha256"] for r in selection["subsets"][name][begin : begin + 16]]
        for name in names
        for begin in range(0, len(selection["subsets"][name]), 16)
    }
    observed = {}
    for row in profile["batches"]:
        key = row["subset"], row["begin"]
        if key in observed or key not in expected or row["contexts"] != expected[key]:
            raise ValueError("Profile does not match complete aligned frozen batch membership")
        lengths = row["token_lengths"]
        if len(lengths) != len(expected[key]) or any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError("Invalid profile token lengths")
        if row["padded_attention_elements"] != len(lengths) * max(lengths) ** 2:
            raise ValueError("Profile attention-size score differs from token geometry")
        observed[key] = row
    if set(observed) != set(expected) or profile["main_outcomes_read"] is not False:
        raise ValueError("Profile lacks full coverage or was not isolated from main outcomes")
    worst = max(profile["batches"], key=lambda row: row["padded_attention_elements"])
    if profile["worst_batch"] != worst:
        raise ValueError("Saved worst batch differs from verified profile maximum")
    return worst


def capture_memory(args, config, identity):
    """Execute the largest precomputed frozen prompt batch; inspect no answer targets."""
    if file_sha256(args.profile) != args.profile_sha256:
        raise ValueError("Memory profile differs from independently reviewed file digest")
    profile = json.loads(args.profile.read_text())
    if (
        profile["config_sha256"] != identity["config_sha256"]
        or profile["selection_sha256"] != identity["selection_sha256"]
        or profile["batch_size"] != 16
    ):
        raise ValueError("Context batch profile differs from the frozen inputs")
    batch = validate_profile(profile, json.loads(args.selection.read_text()))
    prompts = selected_prompts(args.selection, args.audit, batch["subset"])[
        batch["begin"] : batch["begin"] + 16
    ]
    if [row["prompt_sha256"] for row in prompts] != batch["contexts"]:
        raise ValueError("Worst batch differs from frozen prompt order")
    begin = time.perf_counter()
    model, tokenizer, text = load_native(config, args.role, device="cuda:0", dtype=torch.bfloat16)
    token_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=config["generation"]["enable_thinking"],
        )
        for row in prompts
    ]
    if list(map(len, token_ids)) != batch["token_lengths"]:
        raise ValueError("Current exact native token lengths differ from memory profile")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - begin
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    x = capture_context_inputs(
        text, token_ids, config["models"][args.role]["source_layer"], tokenizer.pad_token_id
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    save_tensors(
        args.out / "captured_context_inputs.pt",
        {
            "x": x,
            "prompt_ids": token_ids,
            "profile_batch": batch,
            "identity": identity,
            "scope": "memory preflight; not a main-run input checkpoint",
        },
    )
    return {
        "phase": "capture-memory",
        "batch": batch,
        "profile_sha256": file_sha256(args.profile),
        "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": text.config._attn_implementation,
        "input_shape": list(x.shape),
        "load_seconds": load_seconds,
        "capture_seconds": seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "main_answer_outcomes_read": False,
    }


def generation_throughput(args, config, identity):
    """Compare explicit execution settings; preserve identical sampling parameters."""
    from explore_persona_space.eval.generation import create_vllm_engine

    prompts = selected_prompts(args.selection, args.audit, "pilot_train")[:16]
    spec = config["models"][args.role]
    model_id = config["selection"][args.role]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"])
    knobs = (
        {"max_num_seqs": 16, "enforce_eager": True, "enable_prefix_caching": False}
        if args.candidate == "eager16"
        else {
            "max_num_seqs": 32,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "language_model_only": True,
        }
    )
    save_json(
        args.out / "engine_parameters.json",
        {
            "candidate": args.candidate,
            "checkpoint": model_id,
            "revision": spec["revision"],
            "knobs": knobs,
            "max_model_len": 32768,
            "sampling_parameters": config["generation"],
            "interpretation": "Execution-profile benchmark on pilot prompts; batching and graph kernels need not produce bit-identical sampled tokens.",
        },
    )
    begin = time.perf_counter()
    engine = create_vllm_engine(
        model_id,
        revision=spec["revision"],
        tokenizer_revision=spec["revision"],
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        seed=config["seed"],
        hang_mitigations=True,
        **knobs,
    )
    initialize_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    report = generate_rollouts(
        engine, tokenizer, prompts, config, identity, args.out / "generations"
    )
    seconds = time.perf_counter() - begin
    paths = [args.out / "generations" / f"{row['prompt_sha256']}.json" for row in prompts]
    rows = [json.loads(path.read_text()) for path in paths]
    final_tokens = sum(len(draw["token_ids"]) for row in rows for draw in row["rollouts"])
    recovery_tokens = sum(
        len(draw["token_ids"]) for row in rows for draw in row.get("cap_recovery_history", [])
    )
    result = {
        "phase": "generation",
        "candidate": args.candidate,
        "initialize_seconds": initialize_seconds,
        "generation_seconds": seconds,
        "final_tokens": final_tokens,
        "prior_recovery_tokens": recovery_tokens,
        "generated_tokens_per_second": (final_tokens + recovery_tokens) / seconds,
        "generation_status": report,
        "sampling_parameters_unchanged": True,
    }
    save_json(args.out / "generation_benchmark.json", result)
    if report["needs_cap_recovery"]:
        raise ValueError(
            "Execution benchmark reached its engineering context window; not resolved truncation"
        )
    return result


def main():
    """Each probe is isolated in a fresh output directory with failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture-memory", "generation"))
    parser.add_argument("--role", choices=("primary", "comparison"), required=True)
    parser.add_argument("--candidate", choices=("eager16", "graphs32"), default="eager16")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256", help="Exact independently reviewed profile digest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/workspace_jr.yaml"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("docs/exploratory_workspace_jr/selected_contexts.json"),
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("docs/exploratory_workspace_jr/mapping_provenance.json")
    )
    args = parser.parse_args()
    if args.out.exists() or (
        args.phase == "capture-memory" and (args.profile is None or args.profile_sha256 is None)
    ):
        raise ValueError("Require fresh output and a frozen profile for the memory probe")
    config = load_workspace_jr_config(args.config)
    identity = run_identity(args.config, args.selection, args.role)
    args.out.mkdir(parents=True)
    save_json(
        args.out / "benchmark_started.json",
        {
            "identity": identity,
            "phase": args.phase,
            "candidate": args.candidate,
            "status": "running",
        },
    )
    begin = time.perf_counter()
    try:
        result = (
            capture_memory(args, config, identity)
            if args.phase == "capture-memory"
            else generation_throughput(args, config, identity)
        )
    except Exception as error:
        save_json(
            args.out / "benchmark_failed.json",
            {
                "identity": identity,
                "phase": args.phase,
                "status": "failed",
                "exception_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - begin,
            },
        )
        raise
    save_json(
        args.out / "benchmark_complete.json",
        {
            "identity": identity,
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - begin,
            "result": result,
        },
    )


if __name__ == "__main__":
    main()
