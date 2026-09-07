#!/usr/bin/env python3
"""One-off structured-output recovery for #2254 integrity production.

The sole eligible parent returned a completed, syntactically valid judge object
whose opaque-id roster contained duplicates, omissions, and reordered rows.  A
user-approved v18 amendment permits one whole-packet replacement with identical
prompt, schema, rubric, model, instrument, pass, items, order, and packet size.
The invalid response is retained for audit and none of its scores are used.

This additive wrapper composes the frozen policy-only recovery without
broadening that recovery's eligibility.  Any orphaned replacement failure is
terminal and requires new user direction.
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
import scripts.issue2254_revmap8_subagent_grade as runner


_ORIGINAL_RUN_ONE_JOB = runner._run_one_job
RECOVERY_VERSION = "issue2254-structured-output-recovery-v1"
TARGET_JOB_ID = "integrity_production__coherence__pass02__chunk039"
REPLACEMENT_SUFFIX = "__structured_retry00"
TARGET_FAILURE_SHA256 = "c4c5dbf4a4c6614c77b792c78d80aad255f80783da35c97859272500479be696"
TARGET_RAW_RESPONSE_SHA256 = "4176ba659bb461daf48f5fa3b5fe53818a4a4c0c3ca7030b14d5ef3b9dd656d3"
TARGET_PROMPT_SHA256 = "321611dd42d4ddc4096a97962da8d5e939ad3567b0a89889a14c47709a4cd6ef"
TARGET_SCHEMA_SHA256 = "2081d09884e43bb831caec22b74ef58778927211b6a7e8d5c74c62d4752bf91d"
TARGET_SOURCE_REGISTRY_SHA256 = "d9fd6b2e02e03cb189c00d52b862f6897ebf36ddfe6f746abec753f002232fcc"
TARGET_OPAQUE_REGISTRY_SHA256 = "0ec9f98b48ad123d0e6b709690282d085f74c47a12613da93285df877eb91192"
TARGET_INSTRUMENT_FP = "b779778050d06771805ef65d79ed74454fc2ac89b4067c25272662f90c380606"
TARGET_PROMPT_TOKENS = 38_753
TARGET_N_ITEMS = 51
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
    return _recovery_root(out_root) / "runner_manifest_structured.json"


def _receipt_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "structured_output_replacements" / f"{TARGET_JOB_ID}.json"


def _lease_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "structured_output_replacement_launch.json"


def _runner_manifest(out_root: str | Path) -> dict:
    """Bind this additive recovery to every grading surface it composes."""
    script = Path(__file__).resolve()
    policy_manifest_path = _recovery_root(out_root) / "runner_manifest_universal.json"
    policy._ensure_runner_manifest(out_root)
    if not policy_manifest_path.is_file():
        raise base.AnalysisError("structured recovery requires the universal manifest")
    bound_files = [
        script,
        Path(base.__file__).resolve(),
        Path(policy.__file__).resolve(),
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
        "universal_runner_manifest_sha256": _sha256_file(policy_manifest_path),
        "target_job_id": TARGET_JOB_ID,
        "replacement_suffix": REPLACEMENT_SUFFIX,
        "routing_rule": (
            "Replace only the exact registered malformed parent with one same-packet "
            "job; discard every row of the invalid response."
        ),
        "terminal_rule": ("Any orphaned replacement failure blocks all automatic future attempts."),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    manifest = _runner_manifest(out_root)
    base._immutable_json(_manifest_path(out_root), manifest)
    return manifest


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


def _structural_failure_audit(job: runner.JobSpec, failure: dict) -> dict:
    """Verify the exact invalidity without accepting, repairing, or using scores."""
    if failure.get("status") != "failed_content":
        raise base.AnalysisError("structured recovery requires failed_content")
    if failure.get("attempt_index") != 1 or failure.get("returncode") != 0:
        raise base.AnalysisError("structured recovery requires exact attempt 1 with rc=0")
    if failure.get("job_id") != job.job_id:
        raise base.AnalysisError("structured failure job id changed")
    if failure.get("scope") != job.scope or failure.get("rubric_id") != job.rubric_id:
        raise base.AnalysisError("structured failure scope/rubric changed")
    if failure.get("pass_index") != job.pass_index or failure.get("chunk_index") != job.chunk_index:
        raise base.AnalysisError("structured failure pass/chunk changed")
    if failure.get("instrument_fp") != job.instrument_fp:
        raise base.AnalysisError("structured failure instrument changed")
    if failure.get("prompt_sha256") != base._sha256_text(job.prompt):
        raise base.AnalysisError("structured failure prompt hash changed")
    if failure.get("prompt_tokens_o200k") != job.prompt_tokens_o200k:
        raise base.AnalysisError("structured failure prompt-token count changed")
    if failure.get("request") != job.prompt:
        raise base.AnalysisError("structured failure prompt bytes changed")
    expected_source = [item.source_item_id for item in job.items]
    expected_opaque = [item.opaque_id for item in job.items]
    if failure.get("source_item_ids") != expected_source:
        raise base.AnalysisError("structured failure source registry changed")
    if failure.get("opaque_item_ids") != expected_opaque:
        raise base.AnalysisError("structured failure opaque registry changed")
    raw = failure.get("raw_response")
    if not isinstance(raw, str) or _sha256_text(raw) != TARGET_RAW_RESPONSE_SHA256:
        raise base.AnalysisError("structured failure raw response changed")
    runner._validate_codex_events(job, str(failure.get("stdout", "")))
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise base.AnalysisError("target response is no longer valid JSON") from exc
    if not isinstance(response, dict) or set(response) != {"rubric_id", "scores"}:
        raise base.AnalysisError("target response top-level structure changed")
    if response.get("rubric_id") != job.rubric_id or not isinstance(response.get("scores"), list):
        raise base.AnalysisError("target response rubric/score list changed")
    rows = response["scores"]
    if len(rows) != len(expected_opaque):
        raise base.AnalysisError("target response row count changed")
    returned_ids = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"item_id", "score", "reasoning"}:
            raise base.AnalysisError("target response contains a malformed row")
        item_id = row["item_id"]
        if not isinstance(item_id, str):
            raise base.AnalysisError("target response contains a non-string item id")
        if type(row["score"]) is not int or not 0 <= row["score"] <= 100:
            raise base.AnalysisError("target response contains an invalid score type")
        if not isinstance(row["reasoning"], str) or not row["reasoning"].strip():
            raise base.AnalysisError("target response contains empty reasoning")
        returned_ids.append(item_id)
    counts = Counter(returned_ids)
    duplicate_ids = sorted(item_id for item_id, count in counts.items() if count > 1)
    omitted_ids = [item_id for item_id in expected_opaque if item_id not in counts]
    unexpected_ids = sorted(set(returned_ids) - set(expected_opaque))
    order_mismatches = [
        index
        for index, (expected, returned) in enumerate(
            zip(expected_opaque, returned_ids, strict=True)
        )
        if expected != returned
    ]
    if (
        len(set(returned_ids)) != 49
        or len(duplicate_ids) != 2
        or any(counts[item_id] != 2 for item_id in duplicate_ids)
        or len(omitted_ids) != 2
        or unexpected_ids
        or not order_mismatches
    ):
        raise base.AnalysisError("target duplicate/omission/order defect changed")
    return {
        "returned_rows": len(rows),
        "returned_unique_ids": len(set(returned_ids)),
        "returned_ids_sha256": base._canonical_sha256(returned_ids),
        "duplicate_ids": duplicate_ids,
        "omitted_ids": omitted_ids,
        "unexpected_ids": unexpected_ids,
        "order_mismatch_indices_zero_based": order_mismatches,
        "all_rows_discarded": True,
        "scores_salvaged": 0,
    }


def _validate_target_constants(job: runner.JobSpec) -> None:
    metadata = policy._parent_metadata(job)
    expected = {
        "job_id": TARGET_JOB_ID,
        "scope": "integrity_production",
        "rubric_id": "coherence",
        "pass_index": 2,
        "chunk_index": 39,
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
        raise base.AnalysisError(f"structured target constants changed: {mismatches}")


def _receipt_payload(out_root: str | Path, job: runner.JobSpec) -> dict:
    _validate_target_constants(job)
    canonical = runner._job_record_path(base.analysis_root(out_root), job)
    if canonical.exists():
        raise base.AnalysisError("structured parent unexpectedly has a canonical grade")
    failures = _failure_paths(out_root, job)
    if len(failures) != 1 or _sha256_file(failures[0]) != TARGET_FAILURE_SHA256:
        raise base.AnalysisError("structured target requires its sole exact failure file")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    audit = _structural_failure_audit(job, failure)
    schema_path = _schema_path(out_root, job)
    if not schema_path.is_file() or _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
        raise base.AnalysisError("structured target schema bytes changed")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_schema = runner._output_schema(job)
    if schema != expected_schema:
        raise base.AnalysisError("structured target schema semantics changed")
    replacement = _replacement_job(job)
    if runner._output_schema(replacement) != expected_schema:
        raise base.AnalysisError("replacement schema differs from the parent schema")
    manifest_path = _manifest_path(out_root)
    if not manifest_path.is_file():
        raise base.AnalysisError("structured runner manifest is missing")
    return {
        "version": RECOVERY_VERSION,
        "runner_manifest_sha256": _sha256_file(manifest_path),
        "parent": policy._parent_metadata(job),
        "parent_schema_path": str(schema_path.relative_to(base.analysis_root(out_root))),
        "parent_schema_sha256": _sha256_file(schema_path),
        "failure_path": str(failures[0].relative_to(base.analysis_root(out_root))),
        "failure_sha256": _sha256_file(failures[0]),
        "raw_response_sha256": _sha256_text(failure["raw_response"]),
        "structural_failure_audit": audit,
        "replacement": policy._parent_metadata(replacement),
        "replacement_schema_sha256": _sha256_file(schema_path),
        "routing": "one exact whole-packet replacement",
        "prompt_schema_model_rubric_instrument_pass_packet_changed": False,
        "invalid_response_rows_used": 0,
        "content_attempt_limit": 1,
        "terminal_policy": "any orphaned replacement failure requires user direction",
        "user_approved": True,
        "task_plan_version": 18,
    }


def _load_receipt(out_root: str | Path, job: runner.JobSpec) -> dict:
    _ensure_runner_manifest(out_root)
    path = _receipt_path(out_root)
    if not path.is_file():
        raise base.AnalysisError("structured replacement is not registered")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in record.items() if key != "entry_sha256"}
    if record.get("entry_sha256") != base._canonical_sha256(payload):
        raise base.AnalysisError("structured replacement receipt hash changed")
    expected = _receipt_payload(out_root, job)
    expected["entry_sha256"] = base._canonical_sha256(expected)
    if record != expected:
        raise base.AnalysisError("structured replacement receipt no longer matches live state")
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
        raise base.AnalysisError("structured replacement receipt is missing")
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


def _write_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict:
    """Consume the one replacement slot durably before invoking the grader."""
    path = _lease_path(out_root)
    prior_jobs_args = argparse.Namespace(out_root=out_root)
    items, instrument, rubrics = base._load_staged(prior_jobs_args)
    policy._ACTIVE_OUT_ROOT = out_root
    prior_jobs = policy._prior_recovered_jobs(items, instrument, rubrics, "coherence")
    target_matches = [job for job in prior_jobs if job.job_id == TARGET_JOB_ID]
    if len(target_matches) != 1:
        raise base.AnalysisError("structured target changed immediately before dispatch")
    _load_receipt(out_root, target_matches[0])
    payload = _lease_payload(out_root, replacement)
    if path.exists():
        raise base.AnalysisError("structured replacement launch slot was already consumed")
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, replacement)
    if canonical.exists() or _schema_path(out_root, replacement).exists():
        raise base.AnalysisError("structured replacement artifacts predate launch lease")
    if _failure_paths(out_root, replacement):
        raise base.AnalysisError("structured replacement failures predate launch lease")
    try:
        runner._immutable_json(path, payload)
    except runner.SubagentGradeHaltError as exc:
        raise base.AnalysisError("structured replacement launch slot was already consumed") from exc
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise base.AnalysisError("structured replacement launch lease write was not durable")
    return payload


def _load_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict | None:
    path = _lease_path(out_root)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = _lease_payload(out_root, replacement)
    if record != expected:
        raise base.AnalysisError("structured replacement launch lease changed")
    return record


def register_structured_replacement(args) -> Path:
    """Freeze the exact receipt before dispatching the one replacement job."""
    global _ACTIVE_OUT_ROOT
    _ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    _ensure_runner_manifest(args.out_root)
    path = _receipt_path(args.out_root)
    if path.exists():
        raise base.AnalysisError("structured replacement is already registered")
    items, instrument, rubrics = base._load_staged(args)
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, "coherence")
    registry = policy._load_registry(args.out_root)
    policy._verify_registry_against_roster(args.out_root, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"structured target resolves to {len(matches)} jobs")
    job = matches[0]
    if _lease_path(args.out_root).exists():
        raise base.AnalysisError("replacement launch lease predates registration")
    if _replacement_artifacts(args.out_root, job):
        raise base.AnalysisError("replacement artifacts predate registration")
    payload = _receipt_payload(args.out_root, job)
    payload["entry_sha256"] = base._canonical_sha256(payload)
    base._immutable_json(path, payload)
    print(
        f"[structured-recovery] registered {job.job_id} as {_replacement_job(job).job_id}",
        flush=True,
    )
    return path


def apply_structured_replacement(jobs: list[runner.JobSpec], receipt: dict) -> list[runner.JobSpec]:
    """Replace the one exact parent while preserving item-level decision coverage."""
    output = []
    matched = 0
    for job in jobs:
        if job.job_id != TARGET_JOB_ID:
            output.append(job)
            continue
        if receipt.get("parent") != policy._parent_metadata(job):
            raise base.AnalysisError("structured parent metadata changed")
        replacement = _replacement_job(job)
        if receipt.get("replacement") != policy._parent_metadata(replacement):
            raise base.AnalysisError("structured replacement metadata changed")
        if replacement.prompt != job.prompt or replacement.items != job.items:
            raise base.AnalysisError("structured replacement changed prompt or items")
        output.append(replacement)
        matched += 1
    if matched != 1:
        raise base.AnalysisError(f"structured replacement matched {matched} parents")
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("structured replacement changed decision coverage")
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
    """Accept only attempt 1 success or one transport loss before attempt 2."""
    attempt_index = canonical_record.get("attempt_index")
    failures = _failure_paths(out_root, replacement)
    if attempt_index == 1:
        if failures:
            raise base.AnalysisError(
                "attempt-1 replacement canonical has unexpected failure artifacts"
            )
        return
    if attempt_index != 2 or len(failures) != 1:
        raise base.AnalysisError("structured replacement attempt history changed")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    allowed_statuses = {"failed_timeout", "failed_transport"}
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
    if failure.get("status") not in allowed_statuses or mismatches:
        raise base.AnalysisError("structured replacement has an ineligible pre-canonical failure")
    if "raw_response" in failure:
        raise base.AnalysisError("structured replacement transport failure contains a response")


def _guard_replacement(out_root: str | Path, replacement: runner.JobSpec) -> None:
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, replacement)
    schema_path = _schema_path(out_root, replacement)
    lease = _load_launch_lease(out_root, replacement)
    if canonical.is_file():
        if lease is None:
            raise base.AnalysisError("structured replacement canonical has no launch lease")
        if not schema_path.is_file() or _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("structured replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("structured replacement schema semantics changed")
        record = runner._validate_completed_job(canonical, replacement)
        _validate_replacement_attempt_history(out_root, replacement, record)
        return
    failures = _failure_paths(out_root, replacement)
    if failures and schema_path.is_file():
        if _sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("failed replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("failed replacement schema semantics changed")
    if lease is not None or schema_path.exists() or failures:
        raise base.AnalysisError(
            f"{replacement.job_id}: prior orphaned replacement failure requires user direction"
        )


def _run_one_job_with_structured_lease(args, root: Path, job: runner.JobSpec) -> dict:
    """Write the replacement lease before the inherited grader can dispatch it."""
    if not job.job_id.endswith(REPLACEMENT_SUFFIX):
        return _ORIGINAL_RUN_ONE_JOB(args, root, job)
    expected_root = base.analysis_root(_ACTIVE_OUT_ROOT).resolve()
    if root.resolve() != expected_root:
        raise base.AnalysisError("structured replacement grader root changed")
    _guard_replacement(_ACTIVE_OUT_ROOT, job)
    _write_launch_lease(_ACTIVE_OUT_ROOT, job)
    return _ORIGINAL_RUN_ONE_JOB(args, root, job)


def _fully_recovered_jobs(items, instrument, rubrics, rubric_id):
    """Compose policy recovery, then the one-off structured replacement."""
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("structured recovery output root is not initialized")
    policy._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    if rubric_id != "coherence":
        return policy._fully_recovered_jobs(items, instrument, rubrics, rubric_id)
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    registry = policy._load_registry(_ACTIVE_OUT_ROOT)
    policy._verify_registry_against_roster(_ACTIVE_OUT_ROOT, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"structured target resolves to {len(matches)} jobs")
    receipt = _load_receipt(_ACTIVE_OUT_ROOT, matches[0])
    _guard_unregistered_failures_except_target(_ACTIVE_OUT_ROOT, jobs, registry)
    jobs = policy.apply_policy_recoveries(jobs, rubrics, registry)
    jobs = apply_structured_replacement(jobs, receipt)
    policy._guard_registered_singletons(_ACTIVE_OUT_ROOT, jobs, registry)
    replacement = next(job for job in jobs if job.job_id.endswith(REPLACEMENT_SUFFIX))
    _guard_replacement(_ACTIVE_OUT_ROOT, replacement)
    return jobs


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phases")
    mode.add_argument(
        "--register-structured-replacement",
        action="store_true",
        help="freeze the sole v18 replacement receipt without launching a judge call",
    )
    parser.add_argument("--out-root", default=str(base._REPO_ROOT))
    parser.add_argument("--concurrency", type=int, default=base.CONCURRENCY)
    parser.add_argument("--job-timeout-seconds", type=int, default=3600)
    parser.add_argument("--isolated-cwd", default="/tmp/issue2254-decode-match-isolated")
    return parser


def main() -> None:
    global _ACTIVE_OUT_ROOT
    args = build_argparser().parse_args()
    if args.concurrency != base.CONCURRENCY:
        raise SystemExit(f"--concurrency is pinned to {base.CONCURRENCY}")
    _ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    if args.register_structured_replacement:
        try:
            register_structured_replacement(args)
        finally:
            _ACTIVE_OUT_ROOT = None
            policy._ACTIVE_OUT_ROOT = None
        return
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if "stage" in phases:
        raise SystemExit("the recovery wrapper cannot restage the frozen instrument")
    unknown = sorted(set(phases) - set(base.PHASES))
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices={sorted(base.PHASES)}")
    _ensure_runner_manifest(args.out_root)
    original = base._production_jobs
    original_run_one_job = runner._run_one_job
    base._production_jobs = _fully_recovered_jobs
    runner._run_one_job = _run_one_job_with_structured_lease
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original
        runner._run_one_job = original_run_one_job
        _ACTIVE_OUT_ROOT = None
        policy._ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
