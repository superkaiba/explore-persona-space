#!/usr/bin/env python3
"""Reconcile corrected native logs and export auditable rollout/timing records."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from inspect_ai.log import read_eval_log

from scripts.context_risk_impossiblebench_harness import harness_fingerprint
from scripts.context_risk_impossiblebench_inspect import (
    FROZEN_MANIFEST_SHA256,
    load_samples,
    summarize_logs,
)


def audit(run_result: Path, manifest: Path) -> dict:
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("Manifest differs from the frozen input")
    report = json.loads(run_result.read_text())
    epochs = report["epochs"]
    protocol = (report["requested_samples_per_epoch"], epochs)
    if protocol not in {(3, 1), (60, 8)}:
        raise ValueError(f"Unexpected corrected protocol: {protocol}")
    roster = load_samples(manifest, 1 if protocol == (3, 1) else None, "B")
    expected = {(sample.id, epoch) for sample in roster for epoch in range(1, epochs + 1)}
    expected_samples = {sample.id: sample for sample in roster}
    if report["requested_rollouts"] != len(expected):
        raise ValueError("Reported requested coverage differs from the frozen protocol")
    logs = [read_eval_log(path) for path in report["log_locations"]]
    samples = [sample for log in logs for sample in log.samples or []]
    pairs = [(sample.id, sample.epoch) for sample in samples]
    if len(pairs) != len(set(pairs)) or set(pairs) != expected:
        raise ValueError("Native logs do not contain exactly the expected unique rollout pairs")
    recalculated = summarize_logs(logs, epochs=epochs)
    for key in (
        "harness_fingerprint",
        "by_condition",
        "contexts",
        "attempt_categories",
        "censored_reasons",
        "technical_errors",
        "realized_rollouts",
        "reward_hacking_prevalence_gate",
        "impossible_passes",
        "statuses",
        "passed",
    ):
        if recalculated[key] != report[key]:
            raise ValueError(f"Native log/report disagreement: {key}")
    if report["harness_fingerprint"] != harness_fingerprint():
        raise ValueError("Source fingerprint changed")
    rows = []
    request_seconds = []
    output_tokens = []
    stop_reasons = []
    seeds = []
    for sample in samples:
        original = expected_samples[sample.id]
        if sample.input != original.input:
            raise ValueError(f"Native input differs from manifest: {sample.id}")
        for key in ("condition", "task_id", "exact_context_sha256"):
            if sample.metadata[key] != original.metadata[key]:
                raise ValueError(f"Native metadata differs from manifest: {sample.id}:{key}")
        history = sample.metadata.get("agentic_results", {}).get("attempt_history", [])
        model_events = [event for event in sample.events if event.event == "model"]
        successful_events = [event for event in model_events if event.error is None]
        if len(successful_events) != len(history):
            raise ValueError(f"Attempt/model-event count mismatch: {sample.id}:{sample.epoch}")
        request_rows = []
        for attempt, event in zip(history, successful_events, strict=True):
            key = f"38295:{sample.id}:{sample.epoch}:{attempt['attempt']}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
            if attempt["request_seed"] != seed or event.config.seed != seed:
                raise ValueError(f"Request seed mismatch: {key}")
            if attempt["response"] != event.output.completion:
                raise ValueError(f"Recorded response differs from model event: {key}")
            seeds.append(seed)
            duration = (
                (event.completed - event.timestamp).total_seconds() if event.completed else None
            )
            if duration is not None:
                request_seconds.append(duration)
            usage = event.output.usage.model_dump() if event.output.usage else None
            if usage is not None:
                output_tokens.append(usage["output_tokens"])
            stop_reasons.extend(attempt["stop_reasons"])
            request_rows.append({"seconds": duration, "usage": usage, "seed": seed})
        primary = (sample.scores or {}).get("successful_submission")
        rows.append(
            {
                "id": sample.id,
                "epoch": sample.epoch,
                "condition": sample.metadata["condition"],
                "task_id": sample.metadata["task_id"],
                "exact_context_sha256": sample.metadata["exact_context_sha256"],
                "score": primary.value if primary else None,
                "error": sample.error.model_dump() if sample.error else None,
                "invalidation": sample.invalidation.model_dump() if sample.invalidation else None,
                "total_seconds": sample.total_time,
                "working_seconds": sample.working_time,
                "started_at": sample.started_at,
                "completed_at": sample.completed_at,
                "attempt_history": history,
                "requests": request_rows,
                "model_events": [event.model_dump(mode="json") for event in model_events],
            }
        )
    if len(seeds) != len(set(seeds)):
        raise ValueError("Observed request-seed collision")
    starts = [datetime.fromisoformat(log.stats.started_at) for log in logs]
    ends = [datetime.fromisoformat(log.stats.completed_at) for log in logs]
    wall = (max(ends) - min(starts)).total_seconds()

    def describe(values):
        return (
            {
                "n": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "max": max(values),
                "sum": sum(values),
            }
            if values
            else None
        )

    result = {
        "audit_passed": True,
        "execution_passed": bool(recalculated["passed"] and recalculated["technical_errors"] == 0),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_result_sha256": hashlib.sha256(run_result.read_bytes()).hexdigest(),
        "log_hashes": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in report["log_locations"]
        },
        "expected_unique_rollouts": len(expected),
        "realized_unique_rollouts": len(pairs),
        "verified_request_seeds": len(seeds),
        "native_log_wall_seconds": wall,
        "rollout_seconds": describe(
            [row["total_seconds"] for row in rows if row["total_seconds"] is not None]
        ),
        "request_seconds": describe(request_seconds),
        "request_timing_scope": "Successful model events matched one-to-one with attempt history; all model events including failures are preserved in rollouts.jsonl.",
        "request_output_tokens": describe(output_tokens),
        "non_stop_generations": sum(reason != "stop" for reason in stop_reasons),
        "generation_cap_hits": sum(tokens >= report["max_tokens"] for tokens in output_tokens),
        "by_condition": report["by_condition"],
        "attempt_categories": report["attempt_categories"],
        "technical_errors": report["technical_errors"],
    }
    destination = run_result.parent
    for path, content in (
        (destination / "rollouts.jsonl", "".join(json.dumps(row) + "\n" for row in rows)),
        (destination / "native_log_audit.json", json.dumps(result, indent=2) + "\n"),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_result, args.manifest), indent=2))


if __name__ == "__main__":
    main()
