"""Exact-token rollout capture and token-before-rollout component aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit
from explore_persona_space.analysis.workspace_runtime import (
    content_sha256,
    file_sha256,
    save_json,
    save_tensors,
)


def answer_token_ids(ids: list[int], terminal_ids: set[int]) -> tuple[list[int], int]:
    """Exclude the first terminal token and all following IDs, with an audit count."""
    for index, token in enumerate(ids):
        if token in terminal_ids:
            return ids[:index], len(ids) - index
    return ids, 0


def generate_rollouts(
    engine,
    tokenizer,
    prompts,
    config,
    identity,
    output: Path,
    *,
    contexts_per_batch=16,
    max_model_len=32768,
) -> dict:
    """Use batched vLLM requests with per-draw seeds; persist raw token IDs/text.

    Cap-hit generations remain saved. The caller must execute cap-hit recovery
    before treating a whole subset as complete; this function reports coverage.
    Resume checks bind exact prompt tokenization, settings, checkpoint and code.
    """
    from vllm import SamplingParams

    settings = config["generation"]
    if contexts_per_batch < 1:
        raise ValueError("contexts_per_batch must be positive")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for start in range(0, len(prompts), contexts_per_batch):
        pending, requests, parameters = [], [], []
        for row in prompts[start : start + contexts_per_batch]:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=settings["enable_thinking"],
            )
            if (
                not isinstance(prompt_ids, list)
                or not prompt_ids
                or any(type(token) is not int for token in prompt_ids)
            ):
                raise TypeError("Chat template must return a nonempty flat list of token IDs")
            contract = {
                "identity": identity,
                "prompt_token_ids": prompt_ids,
                "prompt_sha256": row["prompt_sha256"],
                "generation": settings,
            }
            digest = content_sha256(contract)
            path = output / f"{row['prompt_sha256']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["contract_sha256"] != digest:
                    raise ValueError(f"Stale generation checkpoint: {path}")
                if [r["seed"] for r in saved["rollouts"]] != settings["seeds"]:
                    raise ValueError(f"Incomplete rollout checkpoint: {path}")
                results.append(saved)
                continue
            pending.append((row, path, contract, digest))
            for seed in settings["seeds"]:
                requests.append({"prompt_token_ids": prompt_ids})
                parameters.append(
                    SamplingParams(
                        n=1,
                        seed=seed,
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        top_k=settings["top_k"],
                        min_p=settings["min_p"],
                        repetition_penalty=settings["repetition_penalty"],
                        presence_penalty=settings["presence_penalty"],
                        frequency_penalty=settings["frequency_penalty"],
                        max_tokens=settings["initial_max_new_tokens"],
                    )
                )
        generated = engine.generate(requests, parameters, use_tqdm=False) if requests else []
        if len(generated) != len(requests):
            raise ValueError("vLLM did not return all requested rollouts")
        cursor = 0
        for row, path, contract, digest in pending:
            draws = []
            for seed in settings["seeds"]:
                completion = generated[cursor]
                cursor += 1
                if list(completion.prompt_token_ids) != contract["prompt_token_ids"]:
                    raise ValueError("vLLM changed submitted prompt IDs")
                if len(completion.outputs) != 1:
                    raise ValueError("Expected one sampled continuation per request")
                draw = completion.outputs[0]
                draws.append(
                    {
                        "seed": seed,
                        "token_ids": list(draw.token_ids),
                        "text": draw.text,
                        "finish_reason": draw.finish_reason,
                        "stop_reason": draw.stop_reason,
                        "max_new_tokens": settings["initial_max_new_tokens"],
                    }
                )
            saved = {
                "contract": contract,
                "contract_sha256": digest,
                "prompt": row["prompt"],
                "context": row,
                "rollouts": draws,
            }
            save_json(path, saved)
            results.append(saved)
        print(
            f"generation contexts={min(start + contexts_per_batch, len(prompts))}/{len(prompts)}",
            flush=True,
        )
    _recover_caps(engine, results, settings, output, max_model_len)
    draws = [draw for row in results for draw in row["rollouts"]]
    cap_hits = sum(draw["finish_reason"] == "length" for draw in draws)
    report = {
        "n_contexts": len(results),
        "n_rollouts": len(draws),
        "cap_hits": cap_hits,
        "cap_hit_fraction": cap_hits / len(draws),
        "needs_cap_recovery": cap_hits / len(draws) > 0.02,
        "identity": identity,
        "status": "complete" if cap_hits / len(draws) <= 0.02 else "context_window_cap_blocked",
        "planned_contexts": len(prompts),
        "included_prompt_sha256": [row["contract"]["prompt_sha256"] for row in results],
        "exclusions": [],
        "file_sha256": {
            row["contract"]["prompt_sha256"]: file_sha256(
                output / f"{row['contract']['prompt_sha256']}.json"
            )
            for row in results
        },
    }
    save_json(output / "generation_status.json", report)
    return report


def _recover_caps(engine, results, settings, output, max_model_len):
    """Regenerate cap-hit draws at doubled limits while retaining every prior draw."""
    from vllm import SamplingParams

    recovery_round = 0
    while True:
        draws = [draw for row in results for draw in row["rollouts"]]
        capped = sum(draw["finish_reason"] == "length" for draw in draws)
        if capped / len(draws) <= 0.02:
            break
        pending, requests, parameters = [], [], []
        for row in results:
            prompt_ids = row["contract"]["prompt_token_ids"]
            for index, draw in enumerate(row["rollouts"]):
                if draw["finish_reason"] != "length":
                    continue
                cap = min(2 * draw["max_new_tokens"], max_model_len - len(prompt_ids))
                if cap <= draw["max_new_tokens"]:
                    continue
                pending.append((row, index, cap))
                requests.append({"prompt_token_ids": prompt_ids})
                parameters.append(
                    SamplingParams(
                        n=1,
                        seed=draw["seed"],
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        top_k=settings["top_k"],
                        min_p=settings["min_p"],
                        repetition_penalty=settings["repetition_penalty"],
                        presence_penalty=settings["presence_penalty"],
                        frequency_penalty=settings["frequency_penalty"],
                        max_tokens=cap,
                    )
                )
        if not requests:
            break
        recovery_round += 1
        recovered = engine.generate(requests, parameters, use_tqdm=False)
        if len(recovered) != len(pending):
            raise ValueError("Cap recovery did not return every requested draw")
        for (row, index, cap), completion in zip(pending, recovered, strict=True):
            if (
                list(completion.prompt_token_ids) != row["contract"]["prompt_token_ids"]
                or len(completion.outputs) != 1
            ):
                raise ValueError("Cap recovery changed prompt IDs or output count")
            prior = row["rollouts"][index]
            draw = completion.outputs[0]
            row.setdefault("cap_recovery_history", []).append(prior)
            row["rollouts"][index] = {
                "seed": prior["seed"],
                "token_ids": list(draw.token_ids),
                "text": draw.text,
                "finish_reason": draw.finish_reason,
                "stop_reason": draw.stop_reason,
                "max_new_tokens": cap,
            }
            save_json(output / f"{row['contract']['prompt_sha256']}.json", row)
        print(f"generation cap_recovery_round={recovery_round} draws={len(pending)}", flush=True)


@torch.no_grad()
def capture_context_inputs(text, prompt_ids: list[list[int]], source_layer: int, pad_token_id: int):
    """Capture each context once with no answer tokens in the forward batch.

    The caller freezes batch membership independently of generated answers and
    reuses these exact vectors across all rollouts, lenses and higher-K reads.
    """
    if not prompt_ids or any(not ids for ids in prompt_ids):
        raise ValueError("Context-only capture requires nonempty prompt token lists")
    device = text.embed_tokens.weight.device
    ids = torch.full(
        (len(prompt_ids), max(map(len, prompt_ids))), pad_token_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids)
    for index, tokens in enumerate(prompt_ids):
        ids[index, : len(tokens)] = torch.tensor(tokens, device=device)
        mask[index, : len(tokens)] = 1
    cache = {}

    def hook(_module, _inputs, output):
        states = output if isinstance(output, torch.Tensor) else output[0]
        cache["x"] = (
            torch.stack([states[i, len(tokens) - 1] for i, tokens in enumerate(prompt_ids)])
            .detach()
            .cpu()
        )

    handle = text.layers[source_layer].register_forward_hook(hook)
    try:
        text(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    values = cache["x"]
    if values.ndim != 2 or len(values) != len(prompt_ids) or not torch.isfinite(values).all():
        raise ValueError("Invalid context-only input capture")
    return values


def assign_context_input(captures: list[dict], context_input: torch.Tensor) -> list[dict]:
    """Retain answer-batch input reads as diagnostics and use one canonical input."""
    if not captures or context_input.ndim != 1 or not torch.isfinite(context_input).all():
        raise ValueError("A finite canonical input and nonempty captures are required")
    if any(row["prompt_ids"] != captures[0]["prompt_ids"] for row in captures):
        raise ValueError("Canonical context input cannot span different prompts")
    if any(row["x"].shape != context_input.shape for row in captures):
        raise ValueError("Canonical input geometry differs from answer capture")
    if any("answer_batch_x" in row for row in captures):
        raise ValueError("Canonical context input has already been assigned")
    return [{**row, "answer_batch_x": row["x"], "x": context_input.clone()} for row in captures]


@torch.no_grad()
def capture_token_batch(text, rows: list[dict], source_layer: int, pad_token_id: int) -> list[dict]:
    """Capture post-block context-last and all exact saved answer-token positions."""
    if not rows or any(not row["prompt_ids"] or not row["answer_ids"] for row in rows):
        raise ValueError("Capture requires nonempty exact prompt and answer IDs")
    lengths = [len(row["prompt_ids"]) + len(row["answer_ids"]) for row in rows]
    device = text.embed_tokens.weight.device
    ids = torch.full((len(rows), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, : lengths[i]] = torch.tensor(row["prompt_ids"] + row["answer_ids"], device=device)
        mask[i, : lengths[i]] = 1
    cache = {}

    def hook(_module, _inputs, output):
        cache["states"] = output if isinstance(output, torch.Tensor) else output[0]

    handle = text.layers[source_layer].register_forward_hook(hook)
    try:
        text(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        handle.remove()
    states = cache["states"]
    if states.shape[:2] != ids.shape or not torch.isfinite(states).all():
        raise ValueError("Invalid native activation capture")
    outputs = []
    for i, row in enumerate(rows):
        boundary = len(row["prompt_ids"])
        outputs.append(
            {
                **row,
                "x": states[i, boundary - 1].detach().cpu(),
                "answer_states": states[i, boundary : lengths[i]].detach().cpu(),
            }
        )
    return outputs


@torch.no_grad()
def decompose_context(
    captures: list[dict], dictionaries: dict[str, torch.Tensor], *, k: int, token_batch_size=128
) -> dict:
    """Stream token decompositions, then average tokens within each equal-weight draw."""
    if len(captures) < 2:
        raise ValueError("At least two complete rollouts are required for noise estimation")
    if len({r["prompt_sha256"] for r in captures}) != 1:
        raise ValueError("One context at a time is required")
    if len({r["seed"] for r in captures}) != len(captures):
        raise ValueError("Repeated rollout seed in a context")
    xs = torch.stack([row["x"].float() for row in captures])
    torch.testing.assert_close(xs, xs[0].expand_as(xs), rtol=2e-3, atol=2e-3)
    means = {"full": []}
    stats = {
        name: {"active_atoms": [], "squared_error": [], "zero_update_steps": []}
        for name in dictionaries
    }
    lengths = []
    for row in captures:
        h = row["answer_states"].float()
        if h.ndim != 2 or not len(h) or not torch.isfinite(h).all():
            raise ValueError("Empty or invalid answer activation sequence")
        lengths.append(len(h))
        means["full"].append(h.double().mean(0).numpy())
        for name, dictionary in dictionaries.items():
            total = torch.zeros(h.shape[1], dtype=torch.float64)
            for start in range(0, len(h), token_batch_size):
                result = nonnegative_gradient_pursuit(
                    h[start : start + token_batch_size].to(dictionary.device), dictionary, k=k
                )
                total += result.component.double().sum(0).cpu()
                for key in stats[name]:
                    stats[name][key].append(getattr(result, key).cpu())
            component = (total / len(h)).numpy()
            means.setdefault(name, []).append(component)
            means.setdefault(f"rest{name}", []).append(means["full"][-1] - component)
    rollouts = {name: np.stack(value) for name, value in means.items()}
    targets = {name: value.mean(0) for name, value in rollouts.items()}
    noise = {
        name: float(np.square(value - targets[name]).sum() / (len(captures) - 1) / len(captures))
        for name, value in rollouts.items()
    }
    return {
        "prompt_sha256": captures[0]["prompt_sha256"],
        "x": xs[0],
        "max_repeat_context_activation_difference": float((xs - xs[0]).abs().max()),
        "rollout_seeds": [r["seed"] for r in captures],
        "token_counts": lengths,
        "targets": {name: torch.from_numpy(value) for name, value in targets.items()},
        "rollout_means": {name: torch.from_numpy(value) for name, value in rollouts.items()},
        "mean_target_noise_trace": noise,
        "decomposition_statistics": {
            name: {key: torch.cat(values) for key, values in stat.items()}
            for name, stat in stats.items()
        },
    }


def save_capture_batch(path: Path, rows: list[dict], identity: dict) -> None:
    """Persist exact-token native captures as an atomic resumable shard."""
    save_tensors(path, {"identity": identity, "rows": rows})
