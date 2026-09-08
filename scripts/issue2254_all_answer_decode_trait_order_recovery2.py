#!/usr/bin/env python3
"""Second exact whole-packet replacement for a #2254 trait order mismatch.

Plan v21 authorizes one fresh replacement for exactly one sycophancy
trait-production parent. The invalid content-bearing response contributes zero
scores. This additive wrapper composes the frozen v17--v20 recoveries without
modifying their code or receipts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_policy_recovery as policy
import scripts.issue2254_all_answer_decode_recovery as recovery1
import scripts.issue2254_all_answer_decode_structured_recovery as structured
import scripts.issue2254_all_answer_decode_trait_order_recovery as order1
import scripts.issue2254_revmap8_subagent_grade as runner


_ORIGINAL_RUN_ONE_JOB = runner._run_one_job
RECOVERY_VERSION = "issue2254-trait-order-recovery-v2"
TARGET_JOB_ID = "trait_production__trait_sycophancy__pass00__chunk013"
REPLACEMENT_SUFFIX = "__order_retry01"
TARGET_FAILURE_SHA256 = "17122539d060401c3a5f0c307bd04bb58cf011bebdf062e047ca3f5e768e4434"
TARGET_RAW_RESPONSE_SHA256 = "d07980c97e556558acb8fae45f5be801283c4b290efa1476d8f73b4e62db7594"
TARGET_PROMPT_SHA256 = "f12d00f7d0aa45f5eee635d424def46729469b8437b554bd21655222fafa8a18"
TARGET_SCHEMA_SHA256 = "87246e6a3933d6656f41a5f3e2b8492fad9e1b68b473abe6a359d520e3751d35"
TARGET_SOURCE_REGISTRY_SHA256 = (
    "de7312325ee529a24ba46ff05d4d402fd1386bd285f1108c6008c96d82a751d2"
)
TARGET_OPAQUE_REGISTRY_SHA256 = (
    "338625676ed3405a9b61fb0b08fa2a97443bf9314c262f937cf9cab475d61f3a"
)
TARGET_RETURNED_ORDER_SHA256 = (
    "41a6ef7403da386b61b3e36d815d98bebb2256489ee3d00d02b68d2c3d7e1e06"
)
TARGET_INSTRUMENT_FP = "c91469bbb6a7252e52655db3c32a623faef4a7df192cbf0f3d74b12a114470f5"
TARGET_PROMPT_TOKENS = 37_727
TARGET_N_ITEMS = 35
TARGET_ORDER_MISMATCH_INDICES = list(range(28, 35))
TARGET_RETURNED_EXPECTED_POSITION_SEQUENCE = [33, 34, 28, 29, 30, 31, 32]
PINNED_JOB_TIMEOUT_SECONDS = order1.PINNED_JOB_TIMEOUT_SECONDS
PINNED_ISOLATED_CWD = order1.PINNED_ISOLATED_CWD
_ACTIVE_OUT_ROOT: str | Path | None = None


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _manifest_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "runner_manifest_trait_order2.json"


def _receipt_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "trait_order_replacements2" / f"{TARGET_JOB_ID}.json"


def _lease_path(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "trait_order_replacement_launch2.json"


def _failure_paths(out_root: str | Path, job: runner.JobSpec) -> list[Path]:
    return order1._failure_paths(out_root, job)


def _schema_path(out_root: str | Path, job: runner.JobSpec) -> Path:
    return order1._schema_path(out_root, job)


def _replacement_job(job: runner.JobSpec) -> runner.JobSpec:
    if job.job_suffix:
        raise base.AnalysisError(f"{job.job_id}: target unexpectedly has a suffix")
    return replace(job, job_suffix=REPLACEMENT_SUFFIX)


def _runner_manifest(out_root: str | Path) -> dict:
    """Bind the v21 runner to the immutable recovery chain it composes."""
    script = Path(__file__).resolve()
    order1._ensure_runner_manifest(out_root)
    prior_manifest_path = order1._manifest_path(out_root)
    if not prior_manifest_path.is_file():
        raise base.AnalysisError("second trait order recovery requires the v20 manifest")
    bound_files = [
        script,
        Path(base.__file__).resolve(),
        Path(policy.__file__).resolve(),
        Path(structured.__file__).resolve(),
        Path(order1.__file__).resolve(),
        Path(runner.__file__).resolve(),
    ]
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": order1._sha256_file(script),
        "recovery_git_commit": recovery1._last_commit_for(script),
        "bound_script_sha256": {
            str(path.relative_to(base._REPO_ROOT)): order1._sha256_file(path)
            for path in bound_files
        },
        "prior_trait_order_manifest_sha256": order1._sha256_file(prior_manifest_path),
        "target_job_id": TARGET_JOB_ID,
        "replacement_suffix": REPLACEMENT_SUFFIX,
        "model": runner.CODEX_MODEL,
        "reasoning_effort": runner.CODEX_REASONING_EFFORT,
        "concurrency": base.CONCURRENCY,
        "job_timeout_seconds": PINNED_JOB_TIMEOUT_SECONDS,
        "isolated_cwd": PINNED_ISOLATED_CWD,
        "routing_rule": (
            "Replace only the exact registered sycophancy order-invalid parent with one "
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
        "rubric_id": "trait_sycophancy",
        "pass_index": 0,
        "chunk_index": 13,
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
        raise base.AnalysisError(f"second trait order target constants changed: {mismatches}")


def _structural_failure_audit(job: runner.JobSpec, failure: dict) -> dict:
    """Verify the exact order-only defect while accepting none of its scores."""
    audit = order1._structural_failure_audit(
        job,
        failure,
        expected_raw_response_sha256=TARGET_RAW_RESPONSE_SHA256,
        expected_returned_order_sha256=TARGET_RETURNED_ORDER_SHA256,
        expected_mismatch_indices=TARGET_ORDER_MISMATCH_INDICES,
    )
    raw = json.loads(failure["raw_response"])
    expected_ids = [item.opaque_id for item in job.items]
    expected_position = {opaque_id: index for index, opaque_id in enumerate(expected_ids)}
    realized = [expected_position[row["item_id"]] for row in raw["scores"]]
    if [realized[index] for index in TARGET_ORDER_MISMATCH_INDICES] != (
        TARGET_RETURNED_EXPECTED_POSITION_SEQUENCE
    ):
        raise base.AnalysisError("second trait order rotation sequence changed")
    audit["returned_expected_position_sequence_at_mismatches"] = [
        realized[index] for index in TARGET_ORDER_MISMATCH_INDICES
    ]
    return audit


def _receipt_payload(out_root: str | Path, job: runner.JobSpec) -> dict:
    """Construct the exact receipt from live, score-discarding structural evidence."""
    _validate_target_constants(job)
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, job)
    if canonical.exists():
        raise base.AnalysisError("second trait order parent unexpectedly has a canonical grade")
    failures = _failure_paths(out_root, job)
    if len(failures) != 1 or order1._sha256_file(failures[0]) != TARGET_FAILURE_SHA256:
        raise base.AnalysisError("second trait order target requires its sole exact failure")
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    audit = _structural_failure_audit(job, failure)
    schema_path = _schema_path(out_root, job)
    if not schema_path.is_file() or order1._sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
        raise base.AnalysisError("second trait order parent schema bytes changed")
    expected_schema = runner._output_schema(job)
    if json.loads(schema_path.read_text(encoding="utf-8")) != expected_schema:
        raise base.AnalysisError("second trait order parent schema semantics changed")
    replacement = _replacement_job(job)
    if runner._output_schema(replacement) != expected_schema:
        raise base.AnalysisError("second trait order replacement schema differs from parent")
    manifest_path = _manifest_path(out_root)
    if not manifest_path.is_file():
        raise base.AnalysisError("second trait order runner manifest is missing")
    return {
        "version": RECOVERY_VERSION,
        "task_plan_version": 21,
        "authorization_basis": "plan-v19 delegation and two Codex reviews; plan-v21",
        "runner_manifest_sha256": order1._sha256_file(manifest_path),
        "parent": policy._parent_metadata(job),
        "ordered_source_item_ids": [item.source_item_id for item in job.items],
        "ordered_opaque_item_ids": [item.opaque_id for item in job.items],
        "parent_schema_path": str(schema_path.relative_to(root)),
        "parent_schema_sha256": order1._sha256_file(schema_path),
        "failure_path": str(failures[0].relative_to(root)),
        "failure_sha256": order1._sha256_file(failures[0]),
        "raw_response_sha256": order1._sha256_text(failure["raw_response"]),
        "structural_failure_audit": audit,
        "replacement": policy._parent_metadata(replacement),
        "replacement_schema_sha256": order1._sha256_file(schema_path),
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
        raise base.AnalysisError("second trait order replacement is not registered")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in record.items() if key != "entry_sha256"}
    if record.get("entry_sha256") != base._canonical_sha256(payload):
        raise base.AnalysisError("second trait order replacement receipt hash changed")
    expected = _receipt_payload(out_root, job)
    expected["entry_sha256"] = base._canonical_sha256(expected)
    if record != expected:
        raise base.AnalysisError("second trait order receipt no longer matches live state")
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
        raise base.AnalysisError("second trait order replacement receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "version": RECOVERY_VERSION,
        "runner_manifest_sha256": order1._sha256_file(_manifest_path(out_root)),
        "receipt_sha256": order1._sha256_file(receipt_path),
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
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, "trait_sycophancy")
    registry = policy._load_registry(args.out_root)
    policy._verify_registry_against_roster(args.out_root, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"second trait order target resolves to {len(matches)} jobs")
    return matches[0]


def _write_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict:
    """Consume the sole v21 replacement slot before invoking the grader."""
    path = _lease_path(out_root)
    parent = _current_target(argparse.Namespace(out_root=out_root))
    _load_receipt(out_root, parent)
    if _replacement_job(parent) != replacement:
        raise base.AnalysisError("second trait order replacement changed before dispatch")
    payload = _lease_payload(out_root, replacement)
    if path.exists():
        raise base.AnalysisError("second trait order launch slot was already consumed")
    root = base.analysis_root(out_root)
    if (
        runner._job_record_path(root, replacement).exists()
        or _schema_path(out_root, replacement).exists()
        or _failure_paths(out_root, replacement)
    ):
        raise base.AnalysisError("second trait order artifacts predate launch lease")
    try:
        runner._immutable_json(path, payload)
    except runner.SubagentGradeHaltError as exc:
        raise base.AnalysisError("second trait order launch slot was already consumed") from exc
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise base.AnalysisError("second trait order launch lease write was not durable")
    return payload


def _load_launch_lease(out_root: str | Path, replacement: runner.JobSpec) -> dict | None:
    path = _lease_path(out_root)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = _lease_payload(out_root, replacement)
    if record != expected:
        raise base.AnalysisError("second trait order launch lease changed")
    return record


def register_trait_order_replacement(args) -> Path:
    """Freeze the v21 receipt without launching a judge call."""
    global _ACTIVE_OUT_ROOT
    _ACTIVE_OUT_ROOT = args.out_root
    order1._ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    _ensure_runner_manifest(args.out_root)
    path = _receipt_path(args.out_root)
    if path.exists():
        raise base.AnalysisError("second trait order replacement is already registered")
    job = _current_target(args)
    if _lease_path(args.out_root).exists():
        raise base.AnalysisError("second trait order launch lease predates registration")
    if _replacement_artifacts(args.out_root, job):
        raise base.AnalysisError("second trait order artifacts predate registration")
    payload = _receipt_payload(args.out_root, job)
    payload["entry_sha256"] = base._canonical_sha256(payload)
    try:
        runner._immutable_json(path, payload)
    except runner.SubagentGradeHaltError as exc:
        raise base.AnalysisError(
            "second trait order replacement registration was already consumed"
        ) from exc
    print(
        f"[trait-order-recovery2] registered {job.job_id} as {_replacement_job(job).job_id}",
        flush=True,
    )
    return path


def apply_trait_order_replacement(
    jobs: list[runner.JobSpec], receipt: dict
) -> list[runner.JobSpec]:
    """Replace only the exact invalid sycophancy parent one-for-one."""
    output = []
    matched = 0
    for job in jobs:
        if job.job_id != TARGET_JOB_ID:
            output.append(job)
            continue
        if receipt.get("parent") != policy._parent_metadata(job):
            raise base.AnalysisError("second trait order parent metadata changed")
        replacement = _replacement_job(job)
        if receipt.get("replacement") != policy._parent_metadata(replacement):
            raise base.AnalysisError("second trait order replacement metadata changed")
        if replacement.prompt != job.prompt or replacement.items != job.items:
            raise base.AnalysisError("second trait order replacement changed prompt or items")
        output.append(replacement)
        matched += 1
    if matched != 1:
        raise base.AnalysisError(f"second trait order replacement matched {matched} parents")
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("second trait order replacement changed decision coverage")
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


def _guard_replacement(out_root: str | Path, replacement: runner.JobSpec) -> None:
    root = base.analysis_root(out_root)
    canonical = runner._job_record_path(root, replacement)
    schema_path = _schema_path(out_root, replacement)
    lease = _load_launch_lease(out_root, replacement)
    if canonical.is_file():
        if lease is None:
            raise base.AnalysisError("second trait replacement canonical has no launch lease")
        if not schema_path.is_file() or order1._sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("second trait replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("second trait replacement schema semantics changed")
        record = runner._validate_completed_job(canonical, replacement)
        order1._validate_replacement_attempt_history(out_root, replacement, record)
        return
    failures = _failure_paths(out_root, replacement)
    if failures and schema_path.is_file():
        if order1._sha256_file(schema_path) != TARGET_SCHEMA_SHA256:
            raise base.AnalysisError("failed second trait replacement schema bytes changed")
        if json.loads(schema_path.read_text(encoding="utf-8")) != runner._output_schema(
            replacement
        ):
            raise base.AnalysisError("failed second trait replacement schema semantics changed")
    if lease is not None or schema_path.exists() or failures:
        raise base.AnalysisError(
            f"{replacement.job_id}: prior orphaned replacement requires new adjudication"
        )


def _run_one_job_with_trait_order2_lease(args, root: Path, job: runner.JobSpec) -> dict:
    """Route old recoveries unchanged and lease the sole v21 replacement."""
    if job.job_id != f"{TARGET_JOB_ID}{REPLACEMENT_SUFFIX}":
        return order1._run_one_job_with_trait_order_lease(args, root, job)
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("second trait order recovery output root is not initialized")
    if root.resolve() != base.analysis_root(_ACTIVE_OUT_ROOT).resolve():
        raise base.AnalysisError("second trait order replacement grader root changed")
    _guard_replacement(_ACTIVE_OUT_ROOT, job)
    _write_launch_lease(_ACTIVE_OUT_ROOT, job)
    return _ORIGINAL_RUN_ONE_JOB(args, root, job)


def _fully_recovered_jobs(items, instrument, rubrics, rubric_id):
    """Compose all frozen recoveries, then the exact v21 sycophancy replacement."""
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("second trait order recovery output root is not initialized")
    order1._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    policy._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    structured._ACTIVE_OUT_ROOT = _ACTIVE_OUT_ROOT
    if rubric_id != "trait_sycophancy":
        return order1._fully_recovered_jobs(items, instrument, rubrics, rubric_id)
    jobs = policy._prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    registry = policy._load_registry(_ACTIVE_OUT_ROOT)
    policy._verify_registry_against_roster(_ACTIVE_OUT_ROOT, jobs, registry)
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"second trait order target resolves to {len(matches)} jobs")
    receipt = _load_receipt(_ACTIVE_OUT_ROOT, matches[0])
    _guard_unregistered_failures_except_target(_ACTIVE_OUT_ROOT, jobs, registry)
    jobs = policy.apply_policy_recoveries(jobs, rubrics, registry)
    jobs = apply_trait_order_replacement(jobs, receipt)
    policy._guard_registered_singletons(_ACTIVE_OUT_ROOT, jobs, registry)
    replacement_id = f"{TARGET_JOB_ID}{REPLACEMENT_SUFFIX}"
    replacements = [job for job in jobs if job.job_id == replacement_id]
    if len(replacements) != 1:
        raise base.AnalysisError("second trait order replacement no longer resolves once")
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
    order1._ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    if args.register_trait_order_replacement:
        try:
            register_trait_order_replacement(args)
        finally:
            _ACTIVE_OUT_ROOT = None
            order1._ACTIVE_OUT_ROOT = None
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
    runner._run_one_job = _run_one_job_with_trait_order2_lease
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original_production_jobs
        runner._run_one_job = original_run_one_job
        _ACTIVE_OUT_ROOT = None
        order1._ACTIVE_OUT_ROOT = None
        policy._ACTIVE_OUT_ROOT = None
        structured._ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
