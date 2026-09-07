"""Collect the fixed screening/fresh panels and preserve unknown outcomes explicitly."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
# PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
from inspect_ai import eval  # noqa: E402
# PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
from inspect_ai._eval.task import PreviousTask  # noqa: E402
# PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
from inspect_ai.log import read_eval_log, resolve_sample_attachments  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from scripts import context_risk_highrate_design as design  # noqa: E402
from scripts.context_risk_followup import (  # noqa: E402
    SOURCE_SHA,
    _stable_digest,
    _write_json_atomic,
    _write_jsonl_atomic,
    build_task,
    harness_fingerprint,
    load_samples,
    register_qwen_model_family,
    sha256,
    tokenize_samples,
    validate_native,
)
from scripts.context_risk_followup_census import counts, requests  # noqa: E402

MODEL = design.MODEL
PROVIDER_ARGS = {
    "responses_api": False,
    "stream": False,
    "client_timeout": 7200,
    "max_retries": 0,
}
REUSE_FIELDS = (
    "uuid",
    "input",
    "target",
    "scores",
    "messages",
    "events",
    "metadata",
    "started_at",
    "completed_at",
    "error",
    "invalidation",
)


def phase_paths(root: Path, phase: str, pilot: bool = False) -> dict[str, Path]:
    """Resolve only the two phase namespaces and fixed pilot/full artifact names."""
    if phase not in {"screen", "fresh"} or (pilot and phase != "screen"):
        raise ValueError("Only screen, screen pilot and fresh collection are defined")
    out = Path(root) / f"{phase}_B"
    prefix = "pilot_" if pilot else ""
    return {
        "out": out,
        "result": out / ("pilot_result.json" if pilot else "run_result.json"),
        "rows": out / f"{prefix}rollouts.jsonl",
        "audit": out / f"{prefix}native_audit.json",
        "prefix": out / "prefix_tokens.json",
    }


def binding(root: Path, phase: str) -> tuple[Path, dict, int, dict]:
    """Load the independently frozen phase and derive its invariant native metadata."""
    manifest, receipt, epochs = design.load_phase(Path(root), phase)
    if epochs != (2 if phase == "screen" else 4):
        raise ValueError("Phase epoch count differs from the approved allocation")
    receipt_path = root / (
        "manifests/screen_freeze.json" if phase == "screen" else "selection.json"
    )
    metadata = {
        "schema_version": "context_risk_highrate_collection_v1",
        "phase": phase,
        "arm": "B",
        "epochs": epochs,
        "max_attempts": 10,
        "message_limit": 22,
        "manifest_sha256": sha256(manifest),
        "sources_sha256": design.source_hashes(),
        "plan_sha256": sha256(design.DESIGN / "plan.md"),
        "phase_receipt_sha256": sha256(receipt_path),
        "screen_freeze_sha256": sha256(root / "manifests/screen_freeze.json"),
        "harness_fingerprint": harness_fingerprint(),
        "model": MODEL,
        "max_connections": 16,
    }
    if phase == "fresh":
        metadata["selection_sha256"] = sha256(receipt_path)
    timestamp = receipt["frozen_unix" if phase == "screen" else "selected_unix"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or timestamp <= 0:
        raise ValueError("Phase freeze/selection timestamp is invalid")
    return manifest, receipt, epochs, metadata


def expected_keys(samples, epochs: int, pilot: bool) -> set[tuple[str, int]]:
    """Enumerate the exact epoch-expanded roster; pilots retain the frozen first16 contexts."""
    chosen = samples[:16] if pilot else samples
    return {(s.id, e) for s in chosen for e in range(1, epochs + 1)}


def validate_prefixes(tokens: dict, samples) -> None:
    """Check complete one-to-one prefix coverage and integer token/hash evidence."""
    rows = tokens["contexts"]
    expected = {s.id: s for s in samples}
    ids = [r["sample_id"] for r in rows]
    if (
        tokens.get("passed") is not True
        or tokens["n_contexts"] != len(expected)
        or len(ids) != len(expected)
        or len(set(ids)) != len(ids)
        or set(ids) != set(expected)
    ):
        raise ValueError("Prefix evidence roster differs from the complete phase")
    for row in rows:
        values = row["token_ids"]
        if (
            not values
            or any(type(v) is not int or v < 0 for v in values)
            or row["n_prefix_tokens"] != len(values)
            or row["prefix_token_ids_sha256"] != _stable_digest(values)
            or row["exact_context_sha256"]
            != expected[row["sample_id"]].metadata["exact_context_sha256"]
            or len(values) + 65536 > 262144
        ):
            raise ValueError(f"Invalid or mismatched prefix token evidence: {row['sample_id']}")


def raw_rows(logs) -> list[dict]:
    """Persist native errors, invalidations and full attempt histories before validation."""
    rows = []
    for log in logs:
        for sample in log.samples or []:
            sample = resolve_sample_attachments(sample, "full")
            value = sample.model_dump(mode="json")
            rows.append(
                {
                    "sample_id": sample.id,
                    "epoch": sample.epoch,
                    "metadata": value["metadata"],
                    "error": value.get("error"),
                    "invalidation": value.get("invalidation"),
                    "scores": value.get("scores") or {},
                    "uuid": value.get("uuid"),
                    "started_at": value.get("started_at"),
                    "completed_at": value.get("completed_at"),
                }
            )
    return rows


def outcome(sample) -> str:
    """Classify unvalidated evidence conservatively; integrity errors never become negatives."""
    score = (sample.scores or {}).get("successful_submission")
    record = sample.metadata.get("agentic_results", {})
    if (
        sample.error is not None
        or sample.invalidation is not None
        or score is None
        or record.get("censored")
        or score.value not in {"C", "I"}
        or record.get("harness_fingerprint") != harness_fingerprint()
    ):
        return "censored"
    return "success" if score.value == "C" else "failure"


def census(logs, samples, keys: set[tuple[str, int]]) -> dict:
    """Retain planned denominators, missing rows and unknown labels even after a failed audit."""
    observed = defaultdict(list)
    unexpected, duplicate = [], []
    expected = {s.id: s for s in samples}
    for log in logs:
        for sample in log.samples or []:
            key = (sample.id, sample.epoch)
            if key not in keys:
                unexpected.append(list(key))
            observed[key].append(sample)
            if len(observed[key]) > 1:
                duplicate.append(list(key))
    contexts = []
    for sample_id in sorted({k[0] for k in keys}):
        sample = expected[sample_id]
        categories = Counter()
        for key in sorted(k for k in keys if k[0] == sample_id):
            values = observed.get(key, [])
            categories[
                "missing" if not values else "censored" if len(values) != 1 else outcome(values[0])
            ] += 1
        detail = counts(
            categories["success"],
            categories["failure"],
            categories["censored"],
            categories["missing"],
        )
        contexts.append(
            {
                "task_id": sample.metadata["task_id"],
                "condition": sample.metadata["condition"],
                "exact_context_sha256": sample.metadata["exact_context_sha256"],
                **detail,
                "n": detail["realized"],
                "passed": detail["success"],
                "errors": detail["censored"],
            }
        )
    total = counts(
        *(sum(r[k] for r in contexts) for k in ("success", "failure", "censored", "missing"))
    )
    return {
        "contexts": contexts,
        "counts": total,
        "by_condition": {
            c: counts(
                *(
                    sum(r[k] for r in contexts if r["condition"] == c)
                    for k in ("success", "failure", "censored", "missing")
                )
            )
            for c in sorted({r["condition"] for r in contexts})
        },
        "unexpected_keys": unexpected,
        "duplicate_keys": duplicate,
        "coverage_complete": not unexpected and not duplicate and total["missing"] == 0,
    }


def audit_native_logs(
    logs, samples, metadata: dict, cfg: DictConfig, *, pilot: bool, not_before: float
) -> dict:
    """Audit every returned row/request; retain diagnostics for all failures before caller raises."""
    keys = expected_keys(samples, metadata["epochs"], pilot)
    result = census(logs, samples, keys)
    issues, completed, non_stop, recovered = [], [], [], []
    for log in logs:
        try:
            validate_native(log, samples, metadata, cfg)
            if log.status != "success":
                raise ValueError(f"Native invocation status is {log.status}")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issues.append(
                {"scope": str(log.location), "type": type(error).__name__, "message": str(error)}
            )
        for sample in log.samples or []:
            try:
                sample = resolve_sample_attachments(sample, "full")
                rows, stops, retries = requests(sample, MODEL)
                for row in [*rows, *retries]:
                    if datetime.fromisoformat(row["timestamp"]).timestamp() < not_before:
                        raise ValueError("Model request predates frozen phase/selection")
                completed.extend(rows)
                non_stop.extend(stops)
                recovered.extend(retries)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                issues.append(
                    {
                        "scope": [sample.id, sample.epoch],
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
    if not result["coverage_complete"]:
        issues.append(
            {
                "scope": "roster",
                "type": "ValueError",
                "message": "Native roster incomplete, duplicated or unexpected",
            }
        )
    seeds = [r["request_seed"] for r in completed]
    if len(set(seeds)) != len(seeds):
        issues.append(
            {
                "scope": "requests",
                "type": "ValueError",
                "message": "Completed request seed collision",
            }
        )
    result.update(
        {
            "schema_version": "context_risk_highrate_native_audit_v1",
            "verification_passed": not issues,
            "metadata": metadata,
            "sources_sha256": metadata["sources_sha256"],
            "native_logs_sha256": {str(log.location): sha256(Path(log.location)) for log in logs},
            "requests": completed,
            "generation_limit_events": non_stop,
            "recovered_transport_errors": recovered,
            "validation_issues": issues,
            "verification_scope": "Complete native/request evidence; owned process exit is independently checked downstream",
        }
    )
    return result


def validate_resume(old, samples, metadata: dict, cfg: DictConfig) -> None:
    """Reuse only source-identical completed rows, including N; never reroll unresolved errors."""
    validate_native(old, samples, metadata, cfg)
    keys = [(s.id, s.epoch) for s in old.samples or []]
    if len(set(keys)) != len(keys) or not set(keys).issubset(
        expected_keys(samples, metadata["epochs"], False)
    ):
        raise ValueError("Resume roster is duplicated or unplanned")
    for sample in old.samples or []:
        if (
            sample.error is not None
            or sample.invalidation is not None
            or "successful_submission" not in (sample.scores or {})
        ):
            raise ValueError(
                "Unresolved native sample requires diagnosis before resume; no automatic resampling"
            )
        requests(resolve_sample_attachments(sample, "full"), MODEL)


def verify_reuse(old, logs) -> None:
    """Require every previously completed sample, including censored samples, unchanged."""
    current = {(s.id, s.epoch): s for log in logs for s in log.samples or []}
    for sample in old.samples or []:
        key = (sample.id, sample.epoch)
        if key not in current:
            raise ValueError(f"Previously completed sample disappeared during resume: {key}")
        before = resolve_sample_attachments(sample, "full").model_dump(mode="json")
        after = resolve_sample_attachments(current[key], "full").model_dump(mode="json")
        if any(before.get(field) != after.get(field) for field in REUSE_FIELDS):
            raise ValueError(f"Previously completed sample changed during resume: {key}")


def verify_report(
    root: Path, phase: str, *, base_url: str | None = None, model: str = MODEL, pilot: bool = False
) -> dict:
    """Recompute a complete disk-backed audit without mutations or time-dependent return values."""
    root = Path(root)
    paths = phase_paths(root, phase, pilot)
    report = json.loads(paths["result"].read_text())
    if report.get("verification_passed") is not True or report.get("passed") is not True:
        raise ValueError("Collection is not verified; file existence is not completion")
    manifest, receipt, epochs, metadata = binding(root, phase)
    samples = load_samples(manifest)
    if (
        model != MODEL
        or report["model"] != MODEL
        or report["metadata"] != metadata
        or report["is_pilot"] != pilot
    ):
        raise ValueError("Saved report model/phase/source binding differs")
    expected_fields = {
        "schema_version": "context_risk_highrate_run_v1",
        "phase": phase,
        "arm": "B",
        "max_attempts": 10,
        "message_limit": 22,
        "sources_sha256": metadata["sources_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "coverage_complete": True,
    }
    if any(report.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Saved report phase/recipe/coverage fields differ")
    if base_url is not None and report["base_url"].rstrip("/") != base_url.rstrip("/"):
        raise ValueError("Saved model endpoint differs")
    cfg = OmegaConf.create({"model": MODEL, "base_url": report["base_url"]})
    for key, path in (
        ("prefix_tokens_sha256", paths["prefix"]),
        ("rollouts_sha256", paths["rows"]),
        ("native_audit_sha256", paths["audit"]),
    ):
        if report[key] != sha256(path):
            raise ValueError(f"Saved collection artifact changed: {key}")
    launch_path = Path(report["launch_config_path"])
    if report["launch_config_sha256"] != sha256(launch_path):
        raise ValueError("Launch evidence changed")
    launch = json.loads(launch_path.read_text())
    not_before = receipt["frozen_unix" if phase == "screen" else "selected_unix"]
    if launch["metadata"] != metadata or launch["started_unix"] < not_before:
        raise ValueError("Launch predates selection or differs from current phase metadata")
    launched = launch["config"]
    expected_launch = {
        "operation": "run",
        "phase": phase,
        "arm": "B",
        "model": MODEL,
        "base_url": report["base_url"],
        "max_connections": 16,
        "pilot_limit": 16 if pilot else None,
    }
    if (
        any(launched.get(key) != value for key, value in expected_launch.items())
        or Path(launched["root"]).resolve() != root.resolve()
        or launch["critic_review_sha256"] != sha256(Path(launched["review"]))
        or launch["critic_review"] != design.validate_review(Path(launched["review"]))
    ):
        raise ValueError("Launch configuration or independent review differs")
    prefix_record = json.loads(paths["prefix"].read_text())
    validate_prefixes(prefix_record, samples)
    logs = []
    for location, digest in report["native_logs_sha256"].items():
        if sha256(Path(location)) != digest:
            raise ValueError("Native log changed after collection")
        logs.append(read_eval_log(location, resolve_attachments="full"))
    if not logs:
        raise ValueError("No native logs")
    audit = audit_native_logs(logs, samples, metadata, cfg, pilot=pilot, not_before=not_before)
    prefix_lengths = {row["sample_id"]: row["n_prefix_tokens"] for row in prefix_record["contexts"]}
    if any(
        row["attempt"] == 1 and row["usage"]["input_tokens"] != prefix_lengths[row["sample_id"]]
        for row in audit["requests"]
    ):
        raise ValueError("First native request token count differs from saved initial prefix")
    saved = json.loads(paths["audit"].read_text())
    if audit != saved or audit["verification_passed"] is not True:
        raise ValueError("Saved native audit differs from actual terminal evidence")
    with paths["rows"].open() as stream:
        if raw_rows(logs) != [json.loads(line) for line in stream]:
            raise ValueError("Saved raw rows differ from the native samples")
    if any(report[key] != audit[key] for key in ("counts", "contexts", "by_condition")):
        raise ValueError("Saved report counts differ from the native audit")
    if (
        epochs != report["epochs"]
        or len(expected_keys(samples, epochs, pilot)) != report["requested_rollouts"]
        or report["realized_rollouts"] != audit["counts"]["realized"]
    ):
        raise ValueError("Saved expected trajectory count differs")
    return {
        **audit,
        "run_result_sha256": sha256(paths["result"]),
        "prefix_tokens_sha256": sha256(paths["prefix"]),
        "rollouts_sha256": sha256(paths["rows"]),
        "native_audit_sha256": sha256(paths["audit"]),
    }


def run(cfg: DictConfig) -> dict:
    """Execute the real frozen phase, checkpoint diagnostics first, then assert integrity."""
    if cfg.operation != "run" or cfg.phase not in {"screen", "fresh"} or cfg.arm != "B":
        raise ValueError("Collector only accepts operation=run with phase screen/fresh and arm B")
    if str(cfg.model) != MODEL or type(cfg.max_connections) is not int or cfg.max_connections != 16:
        raise ValueError("Model and16 active connections are frozen")
    if cfg.pilot_limit not in (None, 16) or isinstance(cfg.pilot_limit, bool):
        raise ValueError("Only the16-context screening pilot is permitted")
    root, phase, pilot = Path(cfg.root), str(cfg.phase), cfg.pilot_limit == 16
    paths = phase_paths(root, phase, pilot)
    manifest, receipt, epochs, metadata = binding(root, phase)
    if (
        sha256(Path(cfg.source_manifest)) != SOURCE_SHA
        or sha256(Path(cfg.plan)) != metadata["plan_sha256"]
    ):
        raise ValueError("Configured source manifest or plan differs from the frozen inputs")
    review = design.validate_review(Path(cfg.review))
    samples = load_samples(manifest)
    paths["out"].mkdir(parents=True, exist_ok=True)
    previous = None
    if phase == "screen" and not pilot:
        verify_report(root, "screen", base_url=str(cfg.base_url), pilot=True)
        if not cfg.resume_log:
            raise ValueError("Full screening must resume its included pilot explicitly")
    if cfg.resume_log:
        if pilot:
            raise ValueError(
                "A screening pilot is one immutable invocation; resume the full screen"
            )
        previous = read_eval_log(str(cfg.resume_log), resolve_attachments="full")
        # Preserve the exact attempted resume evidence even if the next check rejects it.
        resume_path = paths["out"] / f"resume_input_{time.time_ns()}.jsonl"
        _write_jsonl_atomic(resume_path, raw_rows([previous]))
        _write_json_atomic(
            resume_path.with_suffix(".census.json"),
            census([previous], samples, expected_keys(samples, epochs, False)),
        )
        validate_resume(previous, samples, metadata, cfg)
        if phase == "screen":
            initial = json.loads(phase_paths(root, phase, True)["result"].read_text())
            for location in initial["native_logs_sha256"]:
                verify_reuse(read_eval_log(location), [previous])
    elif list((paths["out"] / "logs").glob("*.eval")) or paths["result"].exists():
        raise FileExistsError("Existing phase requires explicit exact-fingerprint resume")
    task = build_task(samples, epochs=epochs, metadata=metadata, max_connections=16)
    target = (
        task
        if previous is None
        else PreviousTask(
            id=previous.eval.task_id,
            task=task,
            task_args={},
            model=None,
            model_roles=None,
            log=previous,
            log_info=None,
        )
    )
    register_qwen_model_family(MODEL)
    tokens = tokenize_samples(samples, str(cfg.base_url), MODEL)
    validate_prefixes(tokens, samples)
    if paths["prefix"].exists() and json.loads(paths["prefix"].read_text()) != tokens:
        raise ValueError("Live prefix tokens changed after original generation evidence")
    if not paths["prefix"].exists():
        _write_json_atomic(paths["prefix"], tokens)
    launch_path = paths["out"] / f"launch_config_{time.time_ns()}_{os.getpid()}.json"
    started_unix = time.time()
    not_before = receipt["frozen_unix" if phase == "screen" else "selected_unix"]
    if started_unix < not_before:
        raise ValueError("Launch predates immutable phase selection")
    _write_json_atomic(
        launch_path,
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "metadata": metadata,
            "critic_review_sha256": sha256(Path(cfg.review)),
            "critic_review": review,
            "started_unix": started_unix,
            "pid": os.getpid(),
        },
    )
    os.environ.setdefault("LOCAL_API_KEY", "context-risk-local-endpoint")
    started = time.monotonic()
    existing_logs = set((paths["out"] / "logs").glob("*.eval"))
    logs, evaluation_error = [], None
    try:
        logs = eval(
            target,
            model=MODEL,
            model_base_url=str(cfg.base_url),
            model_args=PROVIDER_ARGS,
            log_dir=str(paths["out"] / "logs"),
            display="plain",
            fail_on_error=False,
            score_on_error=True,
            max_samples=16,
            max_sandboxes=16,
            max_subprocesses=16,
            limit=16 if pilot else None,
        )
    except Exception as error:
        evaluation_error = error
        # Inspect journals each completed sample. Read only new invocation logs on failure.
        for path in sorted(set((paths["out"] / "logs").glob("*.eval")) - existing_logs):
            try:
                logs.append(read_eval_log(str(path), resolve_attachments="full"))
            except Exception as read_error:
                _write_json_atomic(
                    paths["out"] / f"native_read_error_{time.time_ns()}.json",
                    {
                        "path": str(path),
                        "error_type": type(read_error).__name__,
                        "message": str(read_error),
                    },
                )
    _write_jsonl_atomic(paths["rows"], raw_rows(logs))
    # This explicit unverified checkpoint survives even an unexpected audit/parser exception.
    # It is never consumable by verify_report or task selection.
    preliminary = {
        "schema_version": "context_risk_highrate_run_v1",
        "phase": phase,
        "arm": "B",
        "metadata": metadata,
        "passed": False,
        "verification_passed": False,
        "validation_pending": True,
        **census(logs, samples, expected_keys(samples, epochs, pilot)),
    }
    _write_json_atomic(paths["result"], preliminary)
    try:
        audit = audit_native_logs(logs, samples, metadata, cfg, pilot=pilot, not_before=not_before)
    except Exception as error:
        preliminary["validation_pending"] = False
        preliminary["validation_issues"] = [
            {"scope": "audit", "type": type(error).__name__, "message": str(error)}
        ]
        _write_json_atomic(paths["result"], preliminary)
        raise
    try:
        if binding(root, phase)[3] != metadata:
            raise ValueError("Phase inputs changed while collection was running")
        if previous is not None:
            verify_reuse(previous, logs)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        audit["validation_issues"].append(
            {
                "scope": "post-collection binding",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
    if evaluation_error is not None:
        audit["validation_issues"].append(
            {
                "scope": "eval",
                "type": type(evaluation_error).__name__,
                "message": str(evaluation_error),
            }
        )
    audit["verification_passed"] = not audit["validation_issues"]
    _write_json_atomic(paths["audit"], audit)
    report = {
        "schema_version": "context_risk_highrate_run_v1",
        "phase": phase,
        "arm": "B",
        "model": MODEL,
        "base_url": str(cfg.base_url),
        "metadata": metadata,
        "epochs": epochs,
        "is_pilot": pilot,
        "max_attempts": 10,
        "message_limit": 22,
        "requested_rollouts": len(expected_keys(samples, epochs, pilot)),
        "realized_rollouts": audit["counts"]["realized"],
        "coverage_complete": audit["coverage_complete"],
        "verification_passed": audit["verification_passed"],
        "passed": audit["verification_passed"],
        "contexts": audit["contexts"],
        "counts": audit["counts"],
        "by_condition": audit["by_condition"],
        "sources_sha256": metadata["sources_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "native_logs_sha256": audit["native_logs_sha256"],
        "prefix_tokens_sha256": sha256(paths["prefix"]),
        "rollouts_sha256": sha256(paths["rows"]),
        "native_audit_sha256": sha256(paths["audit"]),
        "launch_config_path": str(launch_path),
        "launch_config_sha256": sha256(launch_path),
        "elapsed_seconds": time.monotonic() - started,
        "completed_unix": time.time(),
        "validation_issues": audit["validation_issues"],
        "outcome_policy": "N, native errors, invalidations and missing scores are unknown; verified generation-limit N is terminal and reused unchanged. passed verifies collection, not prediction or map benefit.",
    }
    _write_json_atomic(paths["result"], report)
    _write_json_atomic(launch_path.with_name(launch_path.stem + "_result.json"), report)
    print(
        f"[highrate-{phase}] realized={report['realized_rollouts']}/{report['requested_rollouts']} success={audit['counts']['success']} failure={audit['counts']['failure']} unknown={audit['counts']['censored']} verified={report['passed']}",
        flush=True,
    )
    if evaluation_error is not None:
        raise evaluation_error
    if not report["verification_passed"]:
        raise RuntimeError(
            f"Unverified collection; raw errors/counts preserved at {paths['result']}"
        )
    try:
        verify_report(root, phase, base_url=str(cfg.base_url), pilot=pilot)
    except Exception as error:
        report["passed"] = report["verification_passed"] = False
        report["validation_issues"].append(
            {"scope": "final disk readback", "type": type(error).__name__, "message": str(error)}
        )
        _write_json_atomic(paths["result"], report)
        _write_json_atomic(launch_path.with_name(launch_path.stem + "_result.json"), report)
        raise
    return report


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="context_risk_highrate")
def main(cfg: DictConfig) -> None:
    """Dispatch only explicitly requested collection; design freeze/selection live elsewhere."""
    print(json.dumps(run(cfg), sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
