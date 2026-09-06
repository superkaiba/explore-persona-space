#!/usr/bin/env python3
"""Run exact-prefix public information-leak rollouts with pre-action captures."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts.context_risk_agentic_misalignment import score_leak_action  # noqa: E402
from scripts.context_risk_qwen38_smoke import (  # noqa: E402
    _capture_last_prefix,
    _load_model_and_tokenizer,
    render_prefix_ids,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_digest(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as fh:
        np.savez(fh, **arrays)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def load_manifest(path: Path, max_contexts: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if max_contexts is not None:
        rows = rows[:max_contexts]
    if not rows:
        raise RuntimeError("misalignment rollout manifest is empty")
    scorer_payload_by_hash: dict[str, str] = {}
    for row in rows:
        exact_hash = str(row["exact_context_sha256"])
        if _stable_digest(row["messages"]) != exact_hash:
            raise RuntimeError(f"exact-context hash drift for {row['condition_id']}")
        email_content = str(row["email_content"])
        previous_email_content = scorer_payload_by_hash.setdefault(exact_hash, email_content)
        if previous_email_content != email_content:
            raise RuntimeError(
                f"duplicate exact context has inconsistent scorer payload: {exact_hash}"
            )
    return rows


def annotate_sample_outcome(sample: dict[str, Any]) -> str:
    """Assign an observed outcome without treating right-censoring as safety.

    A harmful tool action is observable even if generation subsequently reaches
    the token cap.  Conversely, a non-harmful continuation that reaches the cap
    may still have taken a harmful action later, so it is right-censored rather
    than a negative example.
    """

    harmful_action = bool(sample["action_score"]["harmful_action"])
    censored = bool(sample["hit_max_new_tokens"] and not harmful_action)
    sample["censored"] = censored
    if harmful_action:
        outcome = "positive"
    elif censored:
        outcome = "censored"
    else:
        outcome = "negative"
    sample["outcome"] = outcome
    return outcome


def summarize_sample_outcomes(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"positive": 0, "negative": 0, "censored": 0}
    for sample in samples:
        outcome = str(sample.get("outcome") or annotate_sample_outcome(sample))
        if outcome not in counts:
            raise RuntimeError(f"unknown rollout outcome: {outcome}")
        counts[outcome] += 1
    return counts


def _generate_samples(model, tokenizer, prefix_ids: list[int], cfg: DictConfig, seed_base: int):
    import torch

    device = next(model.parameters()).device
    input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    requested = int(cfg.rollout.continuations_per_context)
    batch_size = int(cfg.rollout.sample_batch_size)
    outputs = []
    for start in range(0, requested, batch_size):
        current_batch = min(batch_size, requested - start)
        batch_seed = int(seed_base) + start
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=float(cfg.rollout.temperature),
            top_p=float(cfg.rollout.top_p),
            max_new_tokens=int(cfg.rollout.max_new_tokens),
            num_return_sequences=current_batch,
            pad_token_id=tokenizer.pad_token_id,
        )
        for lane, sequence in enumerate(generated):
            continuation_ids = sequence[len(prefix_ids) :].tolist()
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError(
                    f"empty sampled continuation at batch start {start}, lane {lane}"
                )
            outputs.append(
                {
                    "sample_index": start + lane,
                    "batch_seed": batch_seed,
                    "batch_lane": lane,
                    "continuation_token_ids_sha256": _stable_digest(continuation_ids),
                    "n_continuation_tokens": len(continuation_ids),
                    "hit_max_new_tokens": len(continuation_ids) >= int(cfg.rollout.max_new_tokens),
                    "response": text,
                }
            )
    if len(outputs) != requested:
        raise RuntimeError(f"generated {len(outputs)} continuations, expected {requested}")
    return outputs


def run_rollouts(cfg: DictConfig) -> dict[str, Any]:
    import torch

    manifest_path = Path(str(cfg.manifest_path))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_contexts_value = cfg.rollout.max_contexts
    max_contexts = None if max_contexts_value is None else int(max_contexts_value)
    rows = load_manifest(manifest_path, max_contexts)
    fingerprint_payload = {
        "schema_version": "context_risk_qwen38_misalignment_rollout_v3",
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "rollout": OmegaConf.to_container(cfg.rollout, resolve=True),
        "manifest_sha256": _sha256(manifest_path),
        "selected_context_sha256": [row["exact_context_sha256"] for row in rows],
    }
    fingerprint = _stable_digest(fingerprint_payload)
    model, tokenizer, wrapper_depth = _load_model_and_tokenizer(cfg)
    layers = [int(layer) for layer in cfg.model.capture_layers]
    context_hash_counts = Counter(str(row["exact_context_sha256"]) for row in rows)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    context_reports = []
    for context_index, row in enumerate(rows):
        condition_id = str(row["condition_id"])
        context_dir = output_dir / f"context_{context_index:03d}"
        activation_path = context_dir / "pre_action_activation.npz"
        rollouts_path = context_dir / "rollouts.json"
        done_path = context_dir / "done.json"
        if done_path.is_file():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("fingerprint") != fingerprint:
                raise RuntimeError(f"{done_path}: rollout fingerprint changed")
            if _sha256(activation_path) != done["activation_sha256"]:
                raise RuntimeError(f"{done_path}: activation hash mismatch")
            if _sha256(rollouts_path) != done["rollouts_sha256"]:
                raise RuntimeError(f"{done_path}: rollout hash mismatch")
            context_reports.append(done)
            continue
        _prefix_text, prefix_ids = render_prefix_ids(
            tokenizer,
            row["messages"],
            enable_thinking=bool(cfg.rollout.enable_thinking),
        )
        if len(prefix_ids) > int(cfg.rollout.max_context_tokens):
            raise RuntimeError(
                f"{condition_id}: prefix has {len(prefix_ids)} tokens > "
                f"{cfg.rollout.max_context_tokens}"
            )
        activation, replay_ids, replay_mask = _capture_last_prefix(model, [prefix_ids], layers)
        if replay_ids.shape[0] != 1 or int(replay_mask.sum()) != len(prefix_ids):
            raise RuntimeError(f"{condition_id}: captured prefix tensor geometry drifted")
        activation_array = activation[0].to(torch.float16).numpy()
        expected_shape = (len(layers), int(cfg.model.expected_hidden_dim))
        if activation_array.shape != expected_shape or not np.isfinite(activation_array).all():
            raise RuntimeError(
                f"{condition_id}: invalid pre-action activation shape {activation_array.shape}"
            )
        sampled = _generate_samples(
            model,
            tokenizer,
            prefix_ids,
            cfg,
            int(row["sampling_seed_base"]),
        )
        for sample in sampled:
            sample["action_score"] = score_leak_action(sample["response"], row["email_content"])
            annotate_sample_outcome(sample)
        outcome_counts = summarize_sample_outcomes(sampled)
        positives = outcome_counts["positive"]
        negatives = outcome_counts["negative"]
        censored = outcome_counts["censored"]
        _write_npz_atomic(
            activation_path,
            activation=activation_array,
            layers=np.asarray(layers, dtype=np.int16),
        )
        rollout_payload = {
            "schema_version": "context_risk_qwen38_misalignment_context_rollouts_v3",
            "condition_id": condition_id,
            "exact_context_sha256": row["exact_context_sha256"],
            "context_group_sha256": row["exact_context_sha256"],
            "context_group_manifest_rows": context_hash_counts[row["exact_context_sha256"]],
            "prefix_token_ids_sha256": _stable_digest(prefix_ids),
            "n_prefix_tokens": len(prefix_ids),
            "continuations": sampled,
        }
        _write_json_atomic(rollouts_path, rollout_payload)
        done = {
            "schema_version": "context_risk_qwen38_misalignment_context_done_v3",
            "fingerprint": fingerprint,
            "context_index": context_index,
            "condition_id": condition_id,
            "exact_context_sha256": row["exact_context_sha256"],
            "context_group_sha256": row["exact_context_sha256"],
            "context_group_manifest_rows": context_hash_counts[row["exact_context_sha256"]],
            "n_prefix_tokens": len(prefix_ids),
            "n_rollouts": len(sampled),
            "n_positive": positives,
            "n_negative": negatives,
            "n_censored": censored,
            "activation_sha256": _sha256(activation_path),
            "rollouts_sha256": _sha256(rollouts_path),
        }
        _write_json_atomic(done_path, done)
        context_reports.append(done)
        print(
            f"[context-risk-misalignment] context={context_index + 1}/{len(rows)} "
            f"condition={condition_id} positive={positives} negative={negatives} "
            f"censored={censored} total={len(sampled)} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    positive = sum(item["n_positive"] for item in context_reports)
    negative = sum(item["n_negative"] for item in context_reports)
    censored = sum(item["n_censored"] for item in context_reports)
    grouped_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in context_reports:
        grouped_reports[str(item["exact_context_sha256"])].append(item)
    context_groups = []
    for exact_hash, group in sorted(grouped_reports.items()):
        group_positive = sum(item["n_positive"] for item in group)
        group_negative = sum(item["n_negative"] for item in group)
        group_censored = sum(item["n_censored"] for item in group)
        context_groups.append(
            {
                "context_group_sha256": exact_hash,
                "condition_ids": [str(item["condition_id"]) for item in group],
                "n_manifest_rows": len(group),
                "n_rollouts": group_positive + group_negative + group_censored,
                "n_positive": group_positive,
                "n_negative": group_negative,
                "n_censored": group_censored,
                "mixed_observed_outcomes": group_positive > 0 and group_negative > 0,
            }
        )
    mixed = sum(item["mixed_observed_outcomes"] for item in context_groups)
    truncated = 0
    for context_index in range(len(rows)):
        payload = json.loads(
            (output_dir / f"context_{context_index:03d}" / "rollouts.json").read_text(
                encoding="utf-8"
            )
        )
        truncated += sum(item["hit_max_new_tokens"] for item in payload["continuations"])
    prevalence_gate = (
        positive >= int(cfg.rollout.minimum_positive_rollouts)
        and negative >= int(cfg.rollout.minimum_negative_rollouts)
        and mixed >= int(cfg.rollout.minimum_mixed_contexts)
    )
    report = {
        "schema_version": "context_risk_qwen38_misalignment_rollout_run_v3",
        "fingerprint": fingerprint,
        "model_id": str(cfg.model.id),
        "model_revision": str(cfg.model.revision),
        "manifest_path": str(manifest_path),
        "public_test_role": "development_only",
        "n_contexts": len(rows),
        "n_unique_contexts": len(context_groups),
        "n_duplicate_manifest_rows": len(rows) - len(context_groups),
        "n_rollouts": positive + negative + censored,
        "n_positive": positive,
        "n_negative": negative,
        "n_censored": censored,
        "n_mixed_contexts": mixed,
        "n_hit_max_new_tokens": truncated,
        "prevalence_gate_thresholds": {
            "minimum_positive_rollouts": int(cfg.rollout.minimum_positive_rollouts),
            "minimum_negative_rollouts": int(cfg.rollout.minimum_negative_rollouts),
            "minimum_mixed_contexts": int(cfg.rollout.minimum_mixed_contexts),
        },
        "prevalence_gate_passed": prevalence_gate,
        "censoring_gate_passed": censored == 0,
        "outcome_definition": {
            "positive": "harmful action observed, including before a token-cap stop",
            "negative": "generation ended before the token cap with no harmful action",
            "censored": "token cap reached before any harmful action was observed",
        },
        "rollout_config": OmegaConf.to_container(cfg.rollout, resolve=True),
        "capture_layers": layers,
        "wrapper_depth": int(wrapper_depth),
        "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30),
        "elapsed_seconds": time.monotonic() - started,
        "contexts": context_reports,
        "context_groups": context_groups,
        "passed": True,
    }
    _write_json_atomic(output_dir / "run_result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@hydra.main(
    version_base="1.3",
    config_path="../configs/eval",
    config_name="context_risk_qwen38_misalignment_rollout",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    run_rollouts(cfg)


if __name__ == "__main__":
    main()
