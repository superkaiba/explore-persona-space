#!/usr/bin/env python3
"""Final, singleton policy recovery for one frozen #2254 packet.

The 15-item packet named below exhausted two upstream content-filter attempts
after two prior lossless binary splits.  This additive wrapper validates that
entire immutable lineage and replaces only that packet with 15 singleton
sessions.  If a singleton is also filtered twice, the run stops: text is never
changed, omitted, imputed, or sent to a substitute judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_recovery as recovery1
import scripts.issue2254_all_answer_decode_recovery2 as recovery2
import scripts.issue2254_revmap8_subagent_grade as runner


RECOVERY_VERSION = "issue2254-policy-packet-recovery-v3"
TARGET_JOB_ID = (
    "integrity_production__coherence__pass00__chunk023"
    "__policy_split01__policy_split00"
)
TARGET_PROMPT_SHA256 = "ecb1f9e7ceb94dbd94568e89520fadf2229fffbeae816fb91dc01e51736ee691"
TARGET_SOURCE_IDS_SHA256 = "266a0c006f0e6c9543f2817f8c7ebada4eb5895165b011ff524b9e70d9a43f75"
TARGET_OPAQUE_IDS_SHA256 = "907c3e128aa3862b33b64a3c0201208d63c4239297d3f2fcae507df6aeb561d9"
TARGET_INSTRUMENT_FP = recovery1.TARGET_INSTRUMENT_FP
TARGET_N_ITEMS = 15
TARGET_PROMPT_TOKENS = 8_288
TARGET_PARTS = 15
TARGET_FAILURE_SHA256 = {
    1: "e59b7db9f3503cb4163d08effafed9307dc6ce0a1ff9ff826e636ed124eb451f",
    2: "4a482fd9791232527e4331cb653877905fccfb6f51245a92c9ce91eacf987705",
}
COMPLETED_SIBLING = {
    "relative_path": (
        "jobs/integrity_production/coherence/pass00/"
        "chunk023__policy_split01__policy_split01.json"
    ),
    "sha256": "66054797692fedce365ac3170c2a3c87b32e7fe07b4adfadeb661cd29c8cf2e9",
}
_ACTIVE_OUT_ROOT: str | Path | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _validate_completed_sibling(out_root: str | Path) -> None:
    path = base.analysis_root(out_root) / COMPLETED_SIBLING["relative_path"]
    if not path.is_file() or _sha256_file(path) != COMPLETED_SIBLING["sha256"]:
        raise base.AnalysisError("completed 15-item sibling is missing or changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "complete"
        or record.get("instrument_fp") != TARGET_INSTRUMENT_FP
        or len(record.get("source_item_ids", [])) != 15
    ):
        raise base.AnalysisError("completed 15-item sibling record is invalid")


def _runner_manifest(out_root: str | Path) -> dict:
    root = _recovery_root(out_root)
    script = Path(__file__).resolve()
    recovery1._ensure_runner_manifest(out_root)
    recovery1._load_split_registry(out_root)
    recovery2._ensure_runner_manifest(out_root)
    recovery2._load_registry(out_root)
    _validate_completed_sibling(out_root)
    prior = [
        root / "runner_manifest.json",
        root / "runner_manifest_v2.json",
        root / "policy_packet_splits" / f"{recovery1.TARGET_JOB_ID}.json",
        root / "policy_packet_splits_v2" / f"{recovery2.TARGET_JOB_ID}.json",
    ]
    if any(not path.is_file() for path in prior):
        raise base.AnalysisError("final recovery requires both immutable prior recoveries")
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "recovery_git_commit": recovery1._last_commit_for(script),
        "prior_recovery_sha256": {
            str(path.relative_to(root)): _sha256_file(path) for path in prior
        },
        "completed_sibling": COMPLETED_SIBLING,
        "policy": (
            "Replace only the exact 15-item packet that exhausted attempts 1 and 2 "
            "with 15 ordered singleton packets. Preserve text, ids, rubric, pass, "
            "instrument, and decisions; stop if any singleton exhausts two attempts."
        ),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    manifest = _runner_manifest(out_root)
    base._immutable_json(_recovery_root(out_root) / "runner_manifest_v3.json", manifest)
    return manifest


def _prior_recovered_jobs(items, instrument, rubrics, rubric_id) -> list[runner.JobSpec]:
    jobs = recovery1._ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, rubric_id)
    jobs = recovery1.apply_policy_packet_splits(
        jobs, rubrics, recovery1._load_split_registry(_ACTIVE_OUT_ROOT)
    )
    return recovery2.apply_policy_packet_split(
        jobs, rubrics, recovery2._load_registry(_ACTIVE_OUT_ROOT)
    )


def _validate_target_job(job: runner.JobSpec) -> None:
    realized = {
        "job_id": job.job_id,
        "prompt_sha256": base._sha256_text(job.prompt),
        "source_ids_sha256": base._canonical_sha256(
            [item.source_item_id for item in job.items]
        ),
        "opaque_ids_sha256": base._canonical_sha256([item.opaque_id for item in job.items]),
        "instrument_fp": job.instrument_fp,
        "n_items": len(job.items),
        "prompt_tokens": job.prompt_tokens_o200k,
    }
    expected = {
        "job_id": TARGET_JOB_ID,
        "prompt_sha256": TARGET_PROMPT_SHA256,
        "source_ids_sha256": TARGET_SOURCE_IDS_SHA256,
        "opaque_ids_sha256": TARGET_OPAQUE_IDS_SHA256,
        "instrument_fp": TARGET_INSTRUMENT_FP,
        "n_items": TARGET_N_ITEMS,
        "prompt_tokens": TARGET_PROMPT_TOKENS,
    }
    if realized != expected:
        raise base.AnalysisError(
            f"final recovery parent differs from frozen exception: "
            f"realized={realized} expected={expected}"
        )


def _target_job(args) -> tuple[runner.JobSpec, dict[str, str]]:
    global _ACTIVE_OUT_ROOT
    _ACTIVE_OUT_ROOT = args.out_root
    items, instrument, rubrics = base._load_staged(args)
    jobs = _prior_recovered_jobs(items, instrument, rubrics, "coherence")
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"final recovery target resolves to {len(matches)} jobs")
    _validate_target_job(matches[0])
    return matches[0], rubrics


def _matching_failures(out_root: str | Path, job: runner.JobSpec) -> list[dict]:
    failures = recovery1._matching_policy_failures(out_root, job)
    by_attempt = {failure["attempt_index"]: failure for failure in failures}
    if (
        len(failures) != runner.MAX_JOB_ATTEMPTS
        or set(by_attempt) != set(TARGET_FAILURE_SHA256)
        or any(
            by_attempt[index]["sha256"] != expected
            for index, expected in TARGET_FAILURE_SHA256.items()
        )
    ):
        raise base.AnalysisError("final recovery evidence differs from exact attempts 1 and 2")
    return failures


def _child_metadata(job: runner.JobSpec, rubric: str) -> list[dict]:
    children = []
    for index, item in enumerate(job.items):
        prompt = recovery1._part_prompt(job, rubric, (item,))
        tokens = runner._count_o200k_tokens(prompt)
        if tokens > base.PROMPT_TOKEN_CAP:
            raise base.AnalysisError(f"singleton {index} exceeds the prompt cap")
        children.append(
            {
                "index": index,
                "job_suffix": f"{job.job_suffix}__policy_singleton{index:02d}",
                "source_item_id": item.source_item_id,
                "opaque_item_id": item.opaque_id,
                "prompt_tokens_o200k": tokens,
                "prompt_sha256": base._sha256_text(prompt),
            }
        )
    return children


def register_policy_split(args) -> Path:
    if args.register_policy_split != TARGET_JOB_ID or args.parts != TARGET_PARTS:
        raise base.AnalysisError(
            f"only {TARGET_JOB_ID} with exactly {TARGET_PARTS} parts is authorized"
        )
    _ensure_runner_manifest(args.out_root)
    job, rubrics = _target_job(args)
    canonical = runner._job_record_path(base.analysis_root(args.out_root), job)
    if canonical.exists():
        raise base.AnalysisError("cannot recover a completed canonical 15-item packet")
    failures = _matching_failures(args.out_root, job)
    record = {
        "version": RECOVERY_VERSION,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "job_suffix": job.job_suffix,
        "prompt_sha256": TARGET_PROMPT_SHA256,
        "prompt_tokens_o200k": TARGET_PROMPT_TOKENS,
        "instrument_fp": TARGET_INSTRUMENT_FP,
        "parts": TARGET_PARTS,
        "n_items": TARGET_N_ITEMS,
        "source_item_ids_sha256": TARGET_SOURCE_IDS_SHA256,
        "opaque_item_ids_sha256": TARGET_OPAQUE_IDS_SHA256,
        "failed_attempts": failures,
        "children": _child_metadata(job, rubrics[job.rubric_id]),
        "transformation": "ordered singleton packet split only",
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
        "terminal_policy": "stop if any singleton exhausts two fresh attempts",
    }
    path = (
        _recovery_root(args.out_root)
        / "policy_packet_splits_v3"
        / f"{job.job_id}.json"
    )
    base._immutable_json(path, record)
    print(
        f"[policy-recovery-v3] registered {job.job_id} singletons={len(job.items)} "
        f"failures={len(failures)}",
        flush=True,
    )
    return path


def _load_registry(out_root: str | Path) -> dict[str, dict]:
    _ensure_runner_manifest(out_root)
    path = (
        _recovery_root(out_root)
        / "policy_packet_splits_v3"
        / f"{TARGET_JOB_ID}.json"
    )
    if not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_static = {
        "version": RECOVERY_VERSION,
        "job_id": TARGET_JOB_ID,
        "scope": "integrity_production",
        "rubric_id": "coherence",
        "pass_index": 0,
        "chunk_index": 23,
        "job_suffix": "__policy_split01__policy_split00",
        "prompt_sha256": TARGET_PROMPT_SHA256,
        "prompt_tokens_o200k": TARGET_PROMPT_TOKENS,
        "instrument_fp": TARGET_INSTRUMENT_FP,
        "parts": TARGET_PARTS,
        "n_items": TARGET_N_ITEMS,
        "source_item_ids_sha256": TARGET_SOURCE_IDS_SHA256,
        "opaque_item_ids_sha256": TARGET_OPAQUE_IDS_SHA256,
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
        "terminal_policy": "stop if any singleton exhausts two fresh attempts",
    }
    if any(record.get(key) != value for key, value in expected_static.items()):
        raise base.AnalysisError("final recovery registry differs from frozen pins")
    failures = record.get("failed_attempts")
    if not isinstance(failures, list) or len(failures) != runner.MAX_JOB_ATTEMPTS:
        raise base.AnalysisError("final recovery failure registry is incomplete")
    analysis_root = base.analysis_root(out_root).resolve()
    seen_attempts: set[int] = set()
    for failure in failures:
        attempt_index = failure.get("attempt_index")
        expected_hash = TARGET_FAILURE_SHA256.get(attempt_index)
        failure_path = (analysis_root / str(failure.get("path", ""))).resolve()
        if analysis_root not in failure_path.parents:
            raise base.AnalysisError("final recovery failure path escapes analysis root")
        if (
            expected_hash is None
            or attempt_index in seen_attempts
            or failure.get("sha256") != expected_hash
            or not failure_path.is_file()
            or _sha256_file(failure_path) != expected_hash
        ):
            raise base.AnalysisError("final recovery failed-attempt evidence changed")
        seen_attempts.add(attempt_index)
    if seen_attempts != set(TARGET_FAILURE_SHA256):
        raise base.AnalysisError("final recovery attempts are not exactly 1 and 2")
    parent = (
        analysis_root
        / "jobs/integrity_production/coherence/pass00"
        / "chunk023__policy_split01__policy_split00.json"
    )
    if parent.exists():
        raise base.AnalysisError("canonical parent and singleton children cannot coexist")
    children = record.get("children")
    if not isinstance(children, list) or len(children) != TARGET_PARTS:
        raise base.AnalysisError("final recovery singleton registry is incomplete")
    return {TARGET_JOB_ID: record}


def apply_policy_packet_split(
    jobs: list[runner.JobSpec], rubrics: dict[str, str], registry: dict[str, dict]
) -> list[runner.JobSpec]:
    if not registry:
        if any(job.job_id == TARGET_JOB_ID for job in jobs):
            raise base.AnalysisError("final recovery split is not registered")
        return jobs
    output: list[runner.JobSpec] = []
    matched = False
    for job in jobs:
        if job.job_id != TARGET_JOB_ID:
            output.append(job)
            continue
        _validate_target_job(job)
        record = registry[TARGET_JOB_ID]
        expected_children = _child_metadata(job, rubrics[job.rubric_id])
        if record.get("children") != expected_children:
            raise base.AnalysisError("final recovery singleton roster changed")
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
        output.extend(children)
        matched = True
    if not matched and any(
        job.scope == "integrity_production" and job.rubric_id == "coherence"
        for job in jobs
    ):
        raise base.AnalysisError("final recovery registry did not match roster")
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("final recovery changed total grading decisions")
    return output


def _guard_singleton_resume(out_root: str | Path, jobs: list[runner.JobSpec]) -> None:
    """Never grant fresh attempts to an incomplete singleton after a prior failure."""
    root = base.analysis_root(out_root)
    prefix = f"{TARGET_JOB_ID}__policy_singleton"
    singletons = [job for job in jobs if job.job_id.startswith(prefix)]
    if singletons and len(singletons) != TARGET_PARTS:
        raise base.AnalysisError("final recovery singleton roster is incomplete")
    for job in singletons:
        canonical = runner._job_record_path(root, job)
        if canonical.is_file():
            runner._validate_completed_job(canonical, job)
            continue
        attempts = root / "attempts" / job.scope / job.rubric_id
        failures = sorted(attempts.glob(f"{job.job_id}.*.failed.json"))
        for path in failures:
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "job_id": job.job_id,
                "prompt_sha256": base._sha256_text(job.prompt),
                "instrument_fp": job.instrument_fp,
                "source_item_ids": [item.source_item_id for item in job.items],
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise base.AnalysisError(
                    f"singleton failure evidence changed or mismatched: {path}"
                )
        if failures:
            raise base.AnalysisError(
                f"{job.job_id}: prior failed singleton attempt exists; terminal policy "
                "forbids automatic retry"
            )


def _fully_recovered_jobs(items, instrument, rubrics, rubric_id):
    jobs = _prior_recovered_jobs(items, instrument, rubrics, rubric_id)
    recovered = apply_policy_packet_split(jobs, rubrics, _load_registry(_ACTIVE_OUT_ROOT))
    _guard_singleton_resume(_ACTIVE_OUT_ROOT, recovered)
    return recovered


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phases")
    mode.add_argument("--register-policy-split")
    parser.add_argument("--parts", type=int, default=TARGET_PARTS)
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
