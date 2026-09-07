#!/usr/bin/env python3
"""Universal, registry-driven transport recovery for #2254 production grading.

This additive wrapper composes the three frozen historical recoveries, then
applies one predeclared rule to any later eligible production packet: after
exactly two pre-score biological-policy transport failures, register and use
ordered singleton sessions.  Packet size is an administration deviation; item
text, rubric, model, instrument, pass, and blinding remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_recovery as recovery1
import scripts.issue2254_all_answer_decode_recovery2 as recovery2
import scripts.issue2254_all_answer_decode_recovery3 as recovery3
import scripts.issue2254_revmap8_subagent_grade as runner


RECOVERY_VERSION = "issue2254-universal-policy-recovery-v1"
ELIGIBLE_SCOPE_RUBRIC = {
    ("integrity_production", "coherence"),
    ("trait_production", "trait_evil"),
    ("trait_production", "trait_sycophancy"),
}
_ACTIVE_OUT_ROOT: str | Path | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _prior_files(out_root: str | Path) -> list[Path]:
    root = _recovery_root(out_root)
    return [
        root / "runner_manifest.json",
        root / "runner_manifest_v2.json",
        root / "runner_manifest_v3.json",
        root / "policy_packet_splits" / f"{recovery1.TARGET_JOB_ID}.json",
        root / "policy_packet_splits_v2" / f"{recovery2.TARGET_JOB_ID}.json",
        root / "policy_packet_splits_v3" / f"{recovery3.TARGET_JOB_ID}.json",
    ]


def _runner_manifest(out_root: str | Path) -> dict:
    script = Path(__file__).resolve()
    recovery1._ensure_runner_manifest(out_root)
    recovery1._load_split_registry(out_root)
    recovery2._ensure_runner_manifest(out_root)
    recovery2._load_registry(out_root)
    recovery3._ensure_runner_manifest(out_root)
    recovery3._load_registry(out_root)
    recovery3._validate_completed_sibling(out_root)
    prior = _prior_files(out_root)
    if any(not path.is_file() for path in prior):
        raise base.AnalysisError("universal recovery requires the complete v1-v3 lineage")
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "recovery_git_commit": recovery1._last_commit_for(script),
        "prior_recovery_sha256": {
            str(path.relative_to(_recovery_root(out_root))): _sha256_file(path)
            for path in prior
        },
        "eligible_scope_rubric": sorted([list(pair) for pair in ELIGIBLE_SCOPE_RUBRIC]),
        "routing_rule": (
            "Every eligible unresolved non-singleton production parent with exactly "
            "two total attempt artifacts, both attempts 1 and 2 rejected pre-score by "
            "the biological-risk policy, is routed directly to ordered singletons."
        ),
        "terminal_rule": (
            "Any orphaned singleton failure blocks all automatic future attempts."
        ),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    manifest = _runner_manifest(out_root)
    base._immutable_json(_recovery_root(out_root) / "runner_manifest_universal.json", manifest)
    return manifest


def _prior_recovered_jobs(items, instrument, rubrics, rubric_id) -> list[runner.JobSpec]:
    jobs = recovery1._ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, rubric_id)
    jobs = recovery1.apply_policy_packet_splits(
        jobs, rubrics, recovery1._load_split_registry(_ACTIVE_OUT_ROOT)
    )
    jobs = recovery2.apply_policy_packet_split(
        jobs, rubrics, recovery2._load_registry(_ACTIVE_OUT_ROOT)
    )
    jobs = recovery3.apply_policy_packet_split(
        jobs, rubrics, recovery3._load_registry(_ACTIVE_OUT_ROOT)
    )
    recovery3._guard_singleton_resume(_ACTIVE_OUT_ROOT, jobs)
    return jobs


def _rubric_id_from_job_id(job_id: str) -> str:
    matches = [
        rubric_id
        for rubric_id in ("coherence", "trait_evil", "trait_sycophancy")
        if f"__{rubric_id}__" in job_id
    ]
    if len(matches) != 1:
        raise base.AnalysisError("job id does not identify exactly one production rubric")
    return matches[0]


def _parent_metadata(job: runner.JobSpec) -> dict:
    return {
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "job_suffix": job.job_suffix,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "instrument_fp": job.instrument_fp,
        "n_items": len(job.items),
        "source_item_ids": [item.source_item_id for item in job.items],
        "opaque_item_ids": [item.opaque_id for item in job.items],
        "source_item_ids_sha256": base._canonical_sha256(
            [item.source_item_id for item in job.items]
        ),
        "opaque_item_ids_sha256": base._canonical_sha256(
            [item.opaque_id for item in job.items]
        ),
    }


def _child_metadata(job: runner.JobSpec, rubric: str) -> list[dict]:
    children = []
    for index, item in enumerate(job.items):
        prompt = recovery1._part_prompt(job, rubric, (item,))
        tokens = runner._count_o200k_tokens(prompt)
        if tokens > base.PROMPT_TOKEN_CAP:
            raise base.AnalysisError(f"{job.job_id}: singleton {index} exceeds prompt cap")
        children.append(
            {
                "index": index,
                "job_suffix": f"{job.job_suffix}__policy_singleton{index:03d}",
                "scope": job.scope,
                "rubric_id": job.rubric_id,
                "pass_index": job.pass_index,
                "chunk_index": job.chunk_index,
                "instrument_fp": job.instrument_fp,
                "source_item_id": item.source_item_id,
                "opaque_item_id": item.opaque_id,
                "prompt_tokens_o200k": tokens,
                "prompt_sha256": base._sha256_text(prompt),
            }
        )
    return children


def _registry_dir(out_root: str | Path) -> Path:
    return _recovery_root(out_root) / "policy_packet_splits_universal"


def _failure_paths(root: Path, job: runner.JobSpec) -> list[Path]:
    return sorted(
        (root / "attempts" / job.scope / job.rubric_id).glob(
            f"{job.job_id}.*.failed.json"
        )
    )


def _exact_policy_failures(out_root: str | Path, job: runner.JobSpec) -> list[dict]:
    root = base.analysis_root(out_root)
    all_failures = _failure_paths(root, job)
    matching = recovery1._matching_policy_failures(out_root, job)
    if len(all_failures) != runner.MAX_JOB_ATTEMPTS or len(matching) != len(all_failures):
        raise base.AnalysisError(
            f"{job.job_id}: eligibility requires exactly two total matching policy failures"
        )
    by_attempt = {failure["attempt_index"]: failure for failure in matching}
    if set(by_attempt) != {1, 2}:
        raise base.AnalysisError(f"{job.job_id}: failures are not exact attempts 1 and 2")
    return [by_attempt[1], by_attempt[2]]


def _load_registry(out_root: str | Path) -> dict[str, dict]:
    manifest = _ensure_runner_manifest(out_root)
    manifest_path = _recovery_root(out_root) / "runner_manifest_universal.json"
    manifest_sha = _sha256_file(manifest_path)
    directory = _registry_dir(out_root)
    if not directory.exists():
        return {}
    paths = sorted(directory.iterdir())
    if any(not path.is_file() or path.suffix != ".json" for path in paths):
        raise base.AnalysisError("universal recovery registry contains an unknown entry")
    records = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.name != f"{record.get('job_id')}.json":
            raise base.AnalysisError(f"universal registry filename mismatch: {path}")
        records.append(record)
    indices = [record.get("registration_index") for record in records]
    if sorted(indices) != list(range(len(records))) or len(set(indices)) != len(indices):
        raise base.AnalysisError("universal registry indices are not a contiguous append log")
    records.sort(key=lambda record: record["registration_index"])
    output: dict[str, dict] = {}
    previous = None
    analysis_root = base.analysis_root(out_root).resolve()
    for record in records:
        if (
            record.get("version") != RECOVERY_VERSION
            or record.get("runner_manifest_sha256") != manifest_sha
            or record.get("prior_registry_entry_sha256") != previous
            or record.get("job_id") in output
        ):
            raise base.AnalysisError("universal registry provenance chain changed")
        payload = {key: value for key, value in record.items() if key != "entry_sha256"}
        if record.get("entry_sha256") != base._canonical_sha256(payload):
            raise base.AnalysisError("universal registry entry hash changed")
        failures = record.get("failed_attempts")
        if not isinstance(failures, list) or len(failures) != runner.MAX_JOB_ATTEMPTS:
            raise base.AnalysisError("universal registry failure evidence is incomplete")
        seen_attempts = set()
        for failure in failures:
            attempt_index = failure.get("attempt_index")
            path = (analysis_root / str(failure.get("path", ""))).resolve()
            if analysis_root not in path.parents:
                raise base.AnalysisError("universal failure evidence escapes analysis root")
            if (
                attempt_index not in {1, 2}
                or attempt_index in seen_attempts
                or not path.is_file()
                or _sha256_file(path) != failure.get("sha256")
            ):
                raise base.AnalysisError("universal failed-attempt evidence changed")
            seen_attempts.add(attempt_index)
        if seen_attempts != {1, 2}:
            raise base.AnalysisError("universal failures are not exact attempts 1 and 2")
        parent_path = (
            analysis_root
            / "jobs"
            / str(record.get("scope"))
            / str(record.get("rubric_id"))
            / f"pass{int(record.get('pass_index')):02d}"
            / f"chunk{int(record.get('chunk_index')):03d}{record.get('job_suffix', '')}.json"
        )
        if parent_path.exists():
            raise base.AnalysisError("universal parent and singleton canonicals coexist")
        output[record["job_id"]] = record
        previous = record["entry_sha256"]
    if manifest != _runner_manifest(out_root):
        raise base.AnalysisError("universal runner manifest changed during registry load")
    return output


def _assert_no_child_artifacts(out_root: str | Path, job: runner.JobSpec) -> None:
    root = base.analysis_root(out_root)
    prefix = f"chunk{job.chunk_index:03d}{job.job_suffix}__policy_singleton"
    jobs_dir = root / "jobs" / job.scope / job.rubric_id / f"pass{job.pass_index:02d}"
    schemas = root / "schemas" / job.scope
    attempts = root / "attempts" / job.scope / job.rubric_id
    collisions = [
        *jobs_dir.glob(f"{prefix}*.json"),
        *schemas.glob(f"{job.job_id}__policy_singleton*.schema.json"),
        *attempts.glob(f"{job.job_id}__policy_singleton*.failed.json"),
    ]
    if collisions:
        raise base.AnalysisError(f"{job.job_id}: singleton child artifacts predate registration")


def register_policy_split(args) -> Path:
    global _ACTIVE_OUT_ROOT
    job_id = args.register_policy_split
    if not re.fullmatch(r"[a-z0-9_]+", job_id):
        raise base.AnalysisError("unsafe or malformed recovery job id")
    if "__policy_singleton" in job_id:
        raise base.AnalysisError("singleton descendants are terminal and cannot be registered")
    _ACTIVE_OUT_ROOT = args.out_root
    _ensure_runner_manifest(args.out_root)
    registry = _load_registry(args.out_root)
    if job_id in registry:
        raise base.AnalysisError(f"{job_id}: parent is already registered")
    rubric_id = _rubric_id_from_job_id(job_id)
    items, instrument, rubrics = base._load_staged(args)
    jobs = _prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    jobs = apply_policy_recoveries(jobs, rubrics, registry)
    matches = [job for job in jobs if job.job_id == job_id]
    if len(matches) != 1:
        raise base.AnalysisError(f"{job_id}: resolves to {len(matches)} current jobs")
    job = matches[0]
    if (job.scope, job.rubric_id) not in ELIGIBLE_SCOPE_RUBRIC or len(job.items) < 2:
        raise base.AnalysisError(f"{job_id}: packet is not an eligible production parent")
    canonical = runner._job_record_path(base.analysis_root(args.out_root), job)
    if canonical.exists():
        raise base.AnalysisError(f"{job_id}: completed parent cannot be registered")
    _assert_no_child_artifacts(args.out_root, job)
    failures = _exact_policy_failures(args.out_root, job)
    manifest_path = _recovery_root(args.out_root) / "runner_manifest_universal.json"
    prior_entries = sorted(registry.values(), key=lambda row: row["registration_index"])
    record = {
        "version": RECOVERY_VERSION,
        "registration_index": len(prior_entries),
        "prior_registry_entry_sha256": (
            None if not prior_entries else prior_entries[-1]["entry_sha256"]
        ),
        "runner_manifest_sha256": _sha256_file(manifest_path),
        **_parent_metadata(job),
        "failed_attempts": failures,
        "children": _child_metadata(job, rubrics[job.rubric_id]),
        "routing": "ordered singleton sessions",
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
        "terminal_policy": "any orphaned singleton failure blocks automatic retry",
    }
    record["entry_sha256"] = base._canonical_sha256(record)
    path = _registry_dir(args.out_root) / f"{job.job_id}.json"
    base._immutable_json(path, record)
    print(
        f"[universal-policy-recovery] registered {job.job_id} "
        f"singletons={len(job.items)} chain_index={record['registration_index']}",
        flush=True,
    )
    return path


def apply_policy_recoveries(
    jobs: list[runner.JobSpec], rubrics: dict[str, str], registry: dict[str, dict]
) -> list[runner.JobSpec]:
    if not jobs:
        return jobs
    roster_scope_rubric = {(job.scope, job.rubric_id) for job in jobs}
    if len(roster_scope_rubric) != 1:
        raise base.AnalysisError("universal recovery expects one production rubric roster")
    scope_rubric = next(iter(roster_scope_rubric))
    relevant = {
        job_id: record
        for job_id, record in registry.items()
        if (record.get("scope"), record.get("rubric_id")) == scope_rubric
    }
    output = []
    matched = set()
    for job in jobs:
        record = relevant.get(job.job_id)
        if record is None:
            output.append(job)
            continue
        parent = _parent_metadata(job)
        if any(record.get(key) != value for key, value in parent.items()):
            raise base.AnalysisError(f"{job.job_id}: universal parent metadata changed")
        expected_children = _child_metadata(job, rubrics[job.rubric_id])
        if record.get("children") != expected_children:
            raise base.AnalysisError(f"{job.job_id}: universal singleton roster changed")
        children = []
        for item, metadata in zip(job.items, expected_children, strict=True):
            prompt = recovery1._part_prompt(job, rubrics[job.rubric_id], (item,))
            children.append(
                runner.JobSpec(
                    scope=job.scope,
                    rubric_id=job.rubric_id,
                    pass_index=job.pass_index,
                    chunk_index=job.chunk_index,
                    items=(item,),
                    prompt=prompt,
                    prompt_tokens_o200k=metadata["prompt_tokens_o200k"],
                    instrument_fp=job.instrument_fp,
                    job_suffix=metadata["job_suffix"],
                )
            )
        if [item.source_item_id for child in children for item in child.items] != [
            item.source_item_id for item in job.items
        ]:
            raise base.AnalysisError(f"{job.job_id}: universal source coverage/order changed")
        if [item.opaque_id for child in children for item in child.items] != [
            item.opaque_id for item in job.items
        ]:
            raise base.AnalysisError(f"{job.job_id}: universal opaque coverage/order changed")
        output.extend(children)
        matched.add(job.job_id)
    if matched != set(relevant):
        raise base.AnalysisError(
            f"universal registry did not match roster: {sorted(set(relevant) - matched)}"
        )
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("universal recovery changed total grading decisions")
    return output


def _verify_registry_against_roster(
    out_root: str | Path, jobs: list[runner.JobSpec], registry: dict[str, dict]
) -> None:
    """Revalidate every relevant receipt and policy failure against live frozen jobs."""
    if not jobs:
        return
    scope_rubric = {(job.scope, job.rubric_id) for job in jobs}
    if len(scope_rubric) != 1:
        raise base.AnalysisError("universal receipt audit expects one rubric roster")
    active = next(iter(scope_rubric))
    by_id = {job.job_id: job for job in jobs}
    for job_id, record in registry.items():
        if (record.get("scope"), record.get("rubric_id")) != active:
            continue
        job = by_id.get(job_id)
        if job is None:
            raise base.AnalysisError(f"{job_id}: universal receipt is outside live roster")
        if (
            (job.scope, job.rubric_id) not in ELIGIBLE_SCOPE_RUBRIC
            or len(job.items) < 2
            or "__policy_singleton" in job.job_id
        ):
            raise base.AnalysisError(f"{job_id}: universal receipt parent is ineligible")
        if record.get("failed_attempts") != _exact_policy_failures(out_root, job):
            raise base.AnalysisError(f"{job_id}: universal policy-failure receipt changed")


def _guard_unregistered_failures(
    out_root: str | Path, jobs: list[runner.JobSpec], registry: dict[str, dict]
) -> None:
    root = base.analysis_root(out_root)
    for job in jobs:
        if job.job_id in registry:
            continue
        canonical = runner._job_record_path(root, job)
        if canonical.is_file():
            runner._validate_completed_job(canonical, job)
            continue
        if _failure_paths(root, job):
            raise base.AnalysisError(
                f"{job.job_id}: unresolved unregistered failure forbids more attempts"
            )


def _guard_registered_singletons(
    out_root: str | Path, jobs: list[runner.JobSpec], registry: dict[str, dict]
) -> None:
    root = base.analysis_root(out_root)
    by_id = {job.job_id: job for job in jobs}
    for record in registry.values():
        for metadata in record["children"]:
            child_id = (
                f"{record['scope']}__{record['rubric_id']}__pass{record['pass_index']:02d}"
                f"__chunk{record['chunk_index']:03d}{metadata['job_suffix']}"
            )
            job = by_id.get(child_id)
            if job is None:
                continue
            canonical = runner._job_record_path(root, job)
            if canonical.is_file():
                runner._validate_completed_job(canonical, job)
                continue
            if _failure_paths(root, job):
                raise base.AnalysisError(
                    f"{job.job_id}: prior failed singleton forbids automatic retry"
                )


def _fully_recovered_jobs(items, instrument, rubrics, rubric_id):
    jobs = _prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    registry = _load_registry(_ACTIVE_OUT_ROOT)
    _verify_registry_against_roster(_ACTIVE_OUT_ROOT, jobs, registry)
    _guard_unregistered_failures(_ACTIVE_OUT_ROOT, jobs, registry)
    recovered = apply_policy_recoveries(jobs, rubrics, registry)
    _guard_registered_singletons(_ACTIVE_OUT_ROOT, recovered, registry)
    return recovered


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phases")
    mode.add_argument("--register-policy-split")
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
    if args.register_policy_split:
        register_policy_split(args)
        return
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if "stage" in phases:
        raise SystemExit("the recovery wrapper cannot restage the frozen instrument")
    unknown = sorted(set(phases) - set(base.PHASES))
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices={sorted(base.PHASES)}")
    _ensure_runner_manifest(args.out_root)
    original = base._production_jobs
    base._production_jobs = _fully_recovered_jobs
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original
        _ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
