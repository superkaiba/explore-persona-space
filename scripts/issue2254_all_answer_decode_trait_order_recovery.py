#!/usr/bin/env python3
"""One-off whole-packet replacement for the #2254 trait order mismatch.

Plan v20 authorizes one fresh replacement for exactly one trait-production
parent.  The original content-bearing response remains failed and contributes
zero scores.  This additive wrapper composes, but does not modify, the frozen
v17 policy recovery and v18 coherence structured-output recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_policy_recovery as policy
import scripts.issue2254_all_answer_decode_recovery as recovery1
import scripts.issue2254_all_answer_decode_structured_recovery as structured
import scripts.issue2254_revmap8_subagent_grade as runner


_ORIGINAL_RUN_ONE_JOB = runner._run_one_job
RECOVERY_VERSION = "issue2254-trait-order-recovery-v1"
TARGET_JOB_ID = "trait_production__trait_evil__pass00__chunk010"
REPLACEMENT_SUFFIX = "__order_retry00"
TARGET_FAILURE_SHA256 = "bc36f4fc001dcd81b1a1f406d9f6d5c14f0efa46d690967d8a8f8502a8363604"
TARGET_RAW_RESPONSE_SHA256 = "624d45933115b990848fef01a147c0ec39337181c35b1a28f821bc9b7bccd33f"
TARGET_PROMPT_SHA256 = "a0972e526216ff29d17a4c4972e21f63158d74930fced4b0d6fdefe78073934e"
TARGET_SCHEMA_SHA256 = "9419e707c41a86832e1ad9aca80c71e850bc62e90cbd460a847ff4532dc45780"
TARGET_SOURCE_REGISTRY_SHA256 = "62d99cac126a2a53ad47f67047ccbf01c47f542b6ea2e755700469f6691e797b"
TARGET_OPAQUE_REGISTRY_SHA256 = "2ede5c725f93d543d99b888cba4b8dcbda765e3dcc3fc83e552debb0b1bc6fe1"
TARGET_RETURNED_ORDER_SHA256 = "eafab13b459269148e6aa2716d5b3fc9c9a07a29f5377ef242c486abd395cf69"
TARGET_INSTRUMENT_FP = "0c24e4c7ff9bd37e127de16b923e35cba5d00a22795d083b05c10bbd76868e17"
TARGET_PROMPT_TOKENS = 39_630
TARGET_N_ITEMS = 36
TARGET_ORDER_MISMATCH_INDICES = list(range(24, 36))
PINNED_JOB_TIMEOUT_SECONDS = 900
PINNED_ISOLATED_CWD = "/tmp/issue2254-decode-match-isolated"
_ACTIVE_OUT_ROOT: str | Path | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _manifest_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "runner_manifest_trait_order.json"


def _receipt_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "trait_order_replacements" / f"{TARGET_JOB_ID}.json"


def _lease_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "trait_order_replacement_launch.json"


def _failure_paths(out_root: str | Path, job: runner.JobSpec) -> list[Path]:
    root = base.analysis_root(out_root)
    return sorted(
        (root / "attempts" / job.scope / job.rubric_id).glob(f"{job.job_id}.*.failed.json")
    )


def _schema_path(out_root: str | Path, job: runner.JobSpec) -> Path:
    return base.analysis_root(out_root) / "schemas" / job.scope / f"{job.job_id}.schema.json"


def _replacement_job(job: runner.JobSpec) -> runner.JobSpec:
    if job.job_suffix:
        raise base.AnalysisError(f"{job.job_id}: target unexpectedly has a suffix")
    return replace(job, job_suffix=REPLACEMENT_SUFFIX)


def _runner_manifest(out_root: str | Path) -> dict:
    """Bind the additive runner to every recovery surface it composes."""
    script = Path(__file__).resolve()
    structured_manifest = structured._ensure_runner_manifest(out_root)
    structured_manifest_path = structured._manifest_path(out_root)
    if not structured_manifest or not structured_manifest_path.is_file():
        raise base.AnalysisError("trait order recovery requires the v18 runner manifest")
    bound_files = [
        script,
        Path(base.__file__).resolve(),
        Path(policy.__file__).resolve(),
        Path(structured.__file__).resolve(),
        Path(runner.__file__).resolve(),
    ]
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "recovery_git_commit": recovery1._last_commit_for(script),
        "bound_script_sha256": {
            str(path.relative_to(base._REPO_ROOT)): _sha256_file(path) for path in bound_files
        },
        "structured_runner_manifest_sha256": _sha256_file(structured_manifest_path),
        "target_job_id": TARGET_JOB_ID,
        "replacement_suffix": REPLACEMENT_SUFFIX,
        "model": runner.CODEX_MODEL,
        "reasoning_effort": runner.CODEX_REASONING_EFFORT,
        "concurrency": base.CONCURRENCY,
        "job_timeout_seconds": PINNED_JOB_TIMEOUT_SECONDS,
        "isolated_cwd": PINNED_ISOLATED_CWD,
        "routing_rule": (
            "Replace only the exact registered order-invalid trait parent with one "
            "fresh byte-identical whole-packet job; use zero rows from the invalid response."
        ),
        "terminal_rule": (
            "Any orphaned replacement artifact or invalid content-bearing replacement "
            "blocks all automatic future attempts."
        ),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    manifest = _runner_manifest(out_root)
    base._immutable_json(_manifest_path(out_root), manifest)
    return manifest


def _validate_target_constants(job: runner.JobSpec) -> None:
    metadata = policy._parent_metadata(job)
    expected = {
        "job_id": TARGET_JOB_ID,
        "scope": "trait_production",
        "rubric_id": "trait_evil",
        "pass_index": 0,
        "chunk_index": 10,
        "job_suffix": "",
        "prompt_sha256": TARGET_PROMPT_SHA256,
        "prompt_tokens_o200k": TARGET_PROMPT_TOKENS,
        "instrument_fp": TARGET_INSTRUMENT_FP,
        "n_items": TARGET_N_ITEMS,
        "source_item_ids_sha256": TARGET_SOURCE_REGISTRY_SHA256,
        "opaque_item_ids_sha256": TARGET_OPAQUE_REGISTRY_SHA256,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise base.AnalysisError(f"trait order target constants changed: {mismatches}")


def _structural_failure_audit(
    job: runner.JobSpec,
    failure: dict,
    *,
    expected_raw_response_sha256: str = TARGET_RAW_RESPONSE_SHA256,
    expected_returned_order_sha256: str = TARGET_RETURNED_ORDER_SHA256,
    expected_mismatch_indices: list[int] = TARGET_ORDER_MISMATCH_INDICES,
) -> dict:
    """Verify order-only invalidity without accepting or using a score."""
    expected_metadata = {
        "status": "failed_content",
        "attempt_index": 1,
        "returncode": 0,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "instrument_fp": job.instrument_fp,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "request": job.prompt,
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
    }
    mismatches = {
        key: (failure.get(key), value)
        for key, value in expected_metadata.items()
        if failure.get(key) != value
    }
    if mismatches:
        raise base.AnalysisError(f"trait order failure metadata changed: {mismatches}")
    raw = failure.get("raw_response")
    if not isinstance(raw, str) or _sha256_text(raw) != expected_raw_response_sha256:
        raise base.AnalysisError("trait order failure raw response changed")
    stdout = str(failure.get("stdout", ""))
    events, grader_thread_id = runner._validate_codex_events(job, stdout)
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise base.AnalysisError("trait order response is no longer valid JSON") from exc
    if not isinstance(response, dict) or set(response) != {"rubric_id", "scores"}:
        raise base.AnalysisError("trait order response top-level structure changed")
    rows = response.get("scores")
    if response.get("rubric_id") != job.rubric_id or not isinstance(rows, list):
        raise base.AnalysisError("trait order response rubric/score-list changed")
    expected_ids = [item.opaque_id for item in job.items]
    if len(rows) != len(expected_ids):
        raise base.AnalysisError("trait order response row count changed")
    returned_ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"item_id", "score"}:
            raise base.AnalysisError("trait order response contains a malformed row")
        item_id = row["item_id"]
        score = row["score"]
        if not isinstance(item_id, str):
            raise base.AnalysisError("trait order response contains a non-string item id")
        if not (score == "REFUSAL" or (type(score) is int and 0 <= score <= 100)):
            raise base.AnalysisError("trait order response contains an invalid score")
        returned_ids.append(item_id)
    returned_counts = Counter(returned_ids)
    expected_counts = Counter(expected_ids)
    mismatch_indices = [
        index
        for index, (expected, returned) in enumerate(zip(expected_ids, returned_ids, strict=True))
        if expected != returned
    ]
    if (
        returned_counts != expected_counts
        or len(returned_counts) != len(returned_ids)
        or returned_ids == expected_ids
        or base._canonical_sha256(returned_ids) != expected_returned_order_sha256
        or mismatch_indices != expected_mismatch_indices
    ):
        raise base.AnalysisError("trait order-only structural defect changed")
    return {
        "returned_rows": len(rows),
        "returned_unique_ids": len(returned_counts),
        "returned_ids_sha256": base._canonical_sha256(returned_ids),
        "expected_ids_sha256": base._canonical_sha256(expected_ids),
        "exact_expected_id_multiset": True,
        "duplicate_ids": [],
        "omitted_ids": [],
        "unexpected_ids": [],
        "order_mismatch_indices_zero_based": mismatch_indices,
        "grader_thread_id": grader_thread_id,
        "codex_event_count": len(events),
        "codex_event_types": [event.get("type") for event in events],
        "codex_stdout_sha256": _sha256_text(stdout),
        "tool_free_completed_event_audit": True,
        "all_rows_discarded": True,
        "scores_salvaged": 0,
        "score_values_used_for_eligibility": False,
    }


def _receipt_payload(out_root: str | Path, job: runner.JobSpec) -> dict:
    _validate_target_constants(job)
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, job)
    if canonical.exists():
        raise base.AnalysisError("trait order parent unexpectedly has a canonical grade")
    failures = _failure_paths(out_root, job)
    if len(failures) != 1 or _sha256_file(failures[0]) != TARGET_FAILURE_SHA256:
        raise base.AnalysisError("trait order target requires its sole exact failure file")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    audit = _structural_failure_audit(job, failure)
    schema_path = _schema_path(out_root, job)
    if not schema_path.is_file() or _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
        raise base.AnalysisError("trait order target schema bytes changed")
    expected_schema = runner._output_schema(job)
    if json.loads(schema_path.read_text(encoding="utf-8")) != expected_schema:
        raise base.AnalysisError("trait order target schema semantics changed")
    replacement = _replacement_job(job)
    if runner._output_schema(replacement) != expected_schema:
        raise base.AnalysisError("trait order replacement schema differs from parent")
    manifest_path = _manifest_path(out_root)
    if not manifest_path.is_file():
        raise base.AnalysisError("trait order runner manifest is missing")
    return {
        "version": RECOVERY_VERSION,
        "task_plan_version": 20,
        "authorization_basis": "plan-v19 delegated Codex-subagent adjudication; plan-v20",
        "runner_manifest_sha256": _sha256_file(manifest_path),
        "parent": policy._parent_metadata(job),
        "ordered_source_item_ids": [item.source_item_id for item in job.items],
        "ordered_opaque_item_ids": [item.opaque_id for item in job.items],
        "parent_schema_path": str(schema_path.relative_to(root)),
        "parent_schema_sha256": _sha256_file(schema_path),
        "failure_path": str(failures[0].relative_to(root)),
        "failure_sha256": _sha256_file(failures[0]),
        "raw_response_sha256": _sha256_text(failure["raw_response"]),
        "structural_failure_audit": audit,
        "replacement": policy._parent_metadata(replacement),
        "replacement_schema_sha256": _sha256_file(schema_path),
        "routing": "one exact fresh whole-packet replacement",
        "prompt_schema_model_rubric_instrument_pass_packet_changed": False,
        "invalid_response_rows_used": 0,
        "content_attempt_limit": 1,
        "terminal_policy": "any orphaned or invalid replacement requires new adjudication",
    }


def _load_receipt(out_root: str | Path, job: runner.JobSpec) -> dict:
    _ensure_runner_manifest(out_root)
    path = _receipt_path(out_root)
    if not path.is_file():
        raise base.AnalysisError("trait order replacement is not registered")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in record.items() if key != "entry_sha256"}
    if record.get("entry_sha256") != base._canonical_sha256(payload):
        raise base.AnalysisError("trait order replacement receipt hash changed")
    expected = _receipt_payload(out_root, job)
    expected["entry_sha256"] = base._canonical_sha256(expected)
    if record != expected:
        raise base.AnalysisError("trait order receipt no longer matches live state")
    return record


def _replacement_artifacts(out_root: str | Path, job: runner.JobSpec) -> list[Path]:
    replacement = _replacement_job(job)
    root = base.analysis_root(out_root)
    return [
        path
        for path in [
            runner._job_record_path(root, replacement),
            _schema_path(out_root, replacement),
            *_failure_paths(out_root, replacement),
        ]
        if path.exists()
    ]


def _lease_payload(out_root: str | Path, replacement: runner.JobSpec) -> dict:
    receipt_path = _receipt_path(out_root)
    if not receipt_path.is_file():
        raise base.AnalysisError("trait order replacement receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "version": RECOVERY_VERSION,
        "runner_manifest_sha256": _sha256_file(_manifest_path(out_root)),
        "receipt_sha256": _sha256_file(receipt_path),
        "receipt_entry_sha256": receipt.get("entry_sha256"),
        "replacement": policy._parent_metadata(replacement),
        "replacement_schema_sha256": TARGET_SCHEMA_SHA256,
        "content_attempt_limit": 1,
        "state": "dispatch_consumed",
        "terminal_without_valid_canonical": True,
    }


def _current_target(args) -> runner.JobSpec:
    items, instrument, rubrics = base._load_staged(args)
    policy._ACTIVE_OUT_ROOT = args.out_root
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, "trait_evil")
    registry = policy._load_registry(args.out_root)
    policy._verify_registry_against_roster(args.out_root, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"trait order target resolves to {len(matches)} jobs")
    return matches[0]


def _write_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict:
    """Consume the replacement slot durably before invoking the grader."""
    path = _lease_path(out_root)
    parent = _current_target(argparse.Namespace(out_root=out_root))
    _load_receipt(out_root, parent)
    if _replacement_job(parent) != replacement:
        raise base.AnalysisError("trait order replacement changed before dispatch")
    payload = _lease_payload(out_root, replacement)
    if path.exists():
        raise base.AnalysisError("trait order replacement launch slot was already consumed")
    root = base.analysis_root(out_root)
    if (
        runner._job_record_path(root, replacement).exists()
        or _schema_path(out_root, replacement).exists()
        or _failure_paths(out_root, replacement)
    ):
        raise base.AnalysisError("trait order replacement artifacts predate launch lease")
    try:
        runner._immutable_json(path, payload)
    except runner.SubagentGradeHaltError as exc:
        raise base.AnalysisError(
            "trait order replacement launch slot was already consumed"
        ) from exc
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise base.AnalysisError("trait order launch lease write was not durable")
    return payload


def _load_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict | None:
    path = _lease_path(out_root)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = _lease_payload(out_root, replacement)
    if record != expected:
        raise base.AnalysisError("trait order replacement launch lease changed")
    return record


def register_trait_order_replacement(args) -> Path:
    """Freeze the exact v20 receipt without launching a judge call."""
    global _ACTIVE_OUT_ROOT
    _ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    _ensure_runner_manifest(args.out_root)
    path = _receipt_path(args.out_root)
    if path.exists():
        raise base.AnalysisError("trait order replacement is already registered")
    job = _current_target(args)
    if _lease_path(args.out_root).exists():
        raise base.AnalysisError("trait order launch lease predates registration")
    if _replacement_artifacts(args.out_root, job):
        raise base.AnalysisError("trait order replacement artifacts predate registration")
    payload = _receipt_payload(args.out_root, job)
    payload["entry_sha256"] = base._canonical_sha256(payload)
    try:
        runner._immutable_json(path, payload)
    except runner.SubagentGradeHaltError as exc:
        raise base.AnalysisError(
            "trait order replacement registration was already consumed"
        ) from exc
    print(
        f"[trait-order-recovery] registered {job.job_id} as {_replacement_job(job).job_id}",
        flush=True,
    )
    return path


def apply_trait_order_replacement(
    jobs: list[runner.JobSpec], receipt: dict
) -> list[runner.JobSpec]:
    """Replace only the exact invalid parent without changing decision coverage."""
    output = []
    matched = 0
    for job in jobs:
        if job.job_id != TARGET_JOB_ID:
            output.append(job)
            continue
        if receipt.get("parent") != policy._parent_metadata(job):
            raise base.AnalysisError("trait order parent metadata changed")
        replacement = _replacement_job(job)
        if receipt.get("replacement") != policy._parent_metadata(replacement):
            raise base.AnalysisError("trait order replacement metadata changed")
        if replacement.prompt != job.prompt or replacement.items != job.items:
            raise base.AnalysisError("trait order replacement changed prompt or items")
        output.append(replacement)
        matched += 1
    if matched != 1:
        raise base.AnalysisError(f"trait order replacement matched {matched} parents")
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("trait order replacement changed decision coverage")
    return output


def _guard_unregistered_failures_except_target(
    out_root: str | Path,
    jobs: list[runner.JobSpec],
    policy_registry: dict[str, dict],
) -> None:
    root = base.analysis_root(out_root)
    for job in jobs:
        if job.job_id in policy_registry or job.job_id == TARGET_JOB_ID:
            continue
        canonical = runner._job_record_path(root, job)
        if canonical.is_file():
            runner._validate_completed_job(canonical, job)
            continue
        if policy._failure_paths(root, job):
            raise base.AnalysisError(
                f"{job.job_id}: unresolved unregistered failure forbids more attempts"
            )


def _validate_replacement_attempt_history(
    out_root: str | Path,
    replacement: runner.JobSpec,
    canonical_record: dict,
) -> None:
    """Accept attempt 1 success or one response-free transport loss then success."""
    attempt_index = canonical_record.get("attempt_index")
    failures = _failure_paths(out_root, replacement)
    if attempt_index == 1:
        if failures:
            raise base.AnalysisError("attempt-1 trait replacement has failure artifacts")
        return
    if attempt_index != 2 or len(failures) != 1:
        raise base.AnalysisError("trait replacement attempt history changed")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    expected = {
        "job_id": replacement.job_id,
        "scope": replacement.scope,
        "rubric_id": replacement.rubric_id,
        "pass_index": replacement.pass_index,
        "chunk_index": replacement.chunk_index,
        "attempt_index": 1,
        "instrument_fp": replacement.instrument_fp,
        "prompt_sha256": base._sha256_text(replacement.prompt),
        "prompt_tokens_o200k": replacement.prompt_tokens_o200k,
        "source_item_ids": [item.source_item_id for item in replacement.items],
        "opaque_item_ids": [item.opaque_id for item in replacement.items],
        "request": replacement.prompt,
    }
    mismatches = {
        key: (failure.get(key), value)
        for key, value in expected.items()
        if failure.get(key) != value
    }
    if (
        failure.get("status") not in {"failed_timeout", "failed_transport"}
        or "raw_response" in failure
        or mismatches
    ):
        raise base.AnalysisError("trait replacement has ineligible pre-canonical failure")


def _guard_replacement(out_root: str | Path, replacement: runner.JobSpec) -> None:
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, replacement)
    schema_path = _schema_path(out_root, replacement)
    lease = _load_launch_lease(out_root, replacement)
    if canonical.is_file():
        if lease is None:
            raise base.AnalysisError("trait replacement canonical has no launch lease")
        if not schema_path.is_file() or _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("trait replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("trait replacement schema semantics changed")
        record = runner._validate_completed_job(canonical, replacement)
        _validate_replacement_attempt_history(out_root, replacement, record)
        return
    failures = _failure_paths(out_root, replacement)
    if failures and schema_path.is_file():
        if _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("failed trait replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("failed trait replacement schema semantics changed")
    if lease is not None or schema_path.exists() or failures:
        raise base.AnalysisError(
            f"{replacement.job_id}: prior orphaned replacement requires new adjudication"
        )


def _run_one_job_with_trait_order_lease(args, root: Path, job: runner.JobSpec) -> dict:
    """Write the v20 lease before its one replacement can dispatch."""
    if job.job_id.endswith(structured.REPLACEMENT_SUFFIX):
        return structured._run_one_job_with_structured_lease(args, root, job)
    if job.job_id != f"{TARGET_JOB_ID}{REPLACEMENT_SUFFIX}":
        return _ORIGINAL_RUN_ONE_JOB(args, root, job)
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("trait order recovery output root is not initialized")
    if root.resolve() != base.analysis_root(_ACTIVE_OUT_ROOT).resolve():
        raise base.AnalysisError("trait order replacement grader root changed")
    _guard_replacement(_ACTIVE_OUT_ROOT, job)
    _write_launch_lease(_ACTIVE_OUT_ROOT, job)
    return _ORIGINAL_RUN_ONE_JOB(args, root, job)


def _fully_recovered_jobs(items, instrument, rubrics, rubric_id):
    """Compose v17 policy, v18 coherence, then the exact v20 trait replacement."""
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("trait order recovery output root is not initialized")
    policy._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    structured._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    if rubric_id == "coherence":
        return structured._fully_recovered_jobs(items, instrument, rubrics, rubric_id)
    if rubric_id != "trait_evil":
        return policy._fully_recovered_jobs(items, instrument, rubrics, rubric_id)
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    registry = policy._load_registry(_ACTIVE_OUT_ROOT)
    policy._verify_registry_against_roster(_ACTIVE_OUT_ROOT, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"trait order target resolves to {len(matches)} jobs")
    receipt = _load_receipt(_ACTIVE_OUT_ROOT, matches[0])
    _guard_unregistered_failures_except_target(_ACTIVE_OUT_ROOT, jobs, registry)
    jobs = policy.apply_policy_recoveries(jobs, rubrics, registry)
    jobs = apply_trait_order_replacement(jobs, receipt)
    policy._guard_registered_singletons(_ACTIVE_OUT_ROOT, jobs, registry)
    replacement_id = f"{TARGET_JOB_ID}{REPLACEMENT_SUFFIX}"
    replacements = [job for job in jobs if job.job_id == replacement_id]
    if len(replacements) != 1:
        raise base.AnalysisError("trait order replacement no longer resolves exactly once")
    _guard_replacement(_ACTIVE_OUT_ROOT, replacements[0])
    return jobs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phases")
    mode.add_argument("--register-trait-order-replacement", action="store_true")
    parser.add_argument("--out-root", default=str(base._REPO_ROOT))
    parser.add_argument("--concurrency", type=int, default=base.CONCURRENCY)
    parser.add_argument("--job-timeout-seconds", type=int, default=PINNED_JOB_TIMEOUT_SECONDS)
    parser.add_argument("--isolated-cwd", default=PINNED_ISOLATED_CWD)
    return parser


def main() -> None:
    global _ACTIVE_OUT_ROOT
    args = build_argparser().parse_args()
    if args.concurrency != base.CONCURRENCY:
        raise SystemExit(f"--concurrency is pinned to {base.CONCURRENCY}")
    if args.job_timeout_seconds != PINNED_JOB_TIMEOUT_SECONDS:
        raise SystemExit(f"--job-timeout-seconds is pinned to {PINNED_JOB_TIMEOUT_SECONDS}")
    if Path(args.isolated_cwd).resolve() != Path(PINNED_ISOLATED_CWD).resolve():
        raise SystemExit(f"--isolated-cwd is pinned to {PINNED_ISOLATED_CWD}")
    _ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    if args.register_trait_order_replacement:
        try:
            register_trait_order_replacement(args)
        finally:
            _ACTIVE_OUT_ROOT = None
            policy._ACTIVE_OUT_ROOT = None
            structured._ACTIVE_OUT_ROOT = None
        return
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if "stage" in phases:
        raise SystemExit("the recovery wrapper cannot restage the frozen instrument")
    unknown = sorted(set(phases) - set(base.PHASES))
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices={sorted(base.PHASES)}")
    _ensure_runner_manifest(args.out_root)
    original_production_jobs = base._production_jobs
    original_run_one_job = runner._run_one_job
    base._production_jobs = _fully_recovered_jobs
    runner._run_one_job = _run_one_job_with_trait_order_lease
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original_production_jobs
        runner._run_one_job = original_run_one_job
        _ACTIVE_OUT_ROOT = None
        policy._ACTIVE_OUT_ROOT = None
        structured._ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
