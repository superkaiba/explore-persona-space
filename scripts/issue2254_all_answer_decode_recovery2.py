#!/usr/bin/env python3
"""Second, additive policy recovery for one frozen #2254 child packet.

The first recovery is immutable and remains the provenance source for the
30-item child.  This wrapper validates that full chain, then replaces only the
child that itself exhausted two upstream content-filter attempts with two
contiguous 15-item grandchildren.  It never edits grading text or scores.
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
import scripts.issue2254_revmap8_subagent_grade as runner


RECOVERY_VERSION = "issue2254-policy-packet-recovery-v2"
TARGET_JOB_ID = (
    "integrity_production__coherence__pass00__chunk023__policy_split01"
)
TARGET_PROMPT_SHA256 = "cb41fd503164532922e874347db962096d2c449a6d20bef4cfafff60987804a6"
TARGET_SOURCE_IDS_SHA256 = "42b74d98ed6ed7ab1c397c7c5588af653147da360d76f2d3933da39c2e7f638b"
TARGET_OPAQUE_IDS_SHA256 = "182f48b3756508a21932fc70b15236dda71548cda618727f24b5a4e585a78f3b"
TARGET_INSTRUMENT_FP = recovery1.TARGET_INSTRUMENT_FP
TARGET_N_ITEMS = 30
TARGET_PROMPT_TOKENS = 18_533
TARGET_PARTS = 2
TARGET_FAILURE_SHA256 = {
    1: "710872bd2c2379f0321cc0cf6aab7713b9ce4b56d0a99af9d359cc568a520ce7",
    2: "03479dfc656aeec4d61208523227cd0ba38070df4479c3af5af30eb95fa8b6ee",
}
TARGET_CHILDREN = (
    {
        "job_suffix": "__policy_split01__policy_split00",
        "n_items": 15,
        "prompt_tokens_o200k": 8_288,
        "prompt_sha256": "ecb1f9e7ceb94dbd94568e89520fadf2229fffbeae816fb91dc01e51736ee691",
    },
    {
        "job_suffix": "__policy_split01__policy_split01",
        "n_items": 15,
        "prompt_tokens_o200k": 10_976,
        "prompt_sha256": "c8ecc97a03e3ccddd82a32fb778a9c0b8ba877c21fc7db3757108de5a1c48186",
    },
)
_ACTIVE_OUT_ROOT: str | Path | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _runner_manifest(out_root: str | Path) -> dict:
    script = Path(__file__).resolve()
    root = _recovery_root(out_root)
    first_runner = root / "runner_manifest.json"
    first_split = (
        root
        / "policy_packet_splits"
        / f"{recovery1.TARGET_JOB_ID}.json"
    )
    recovery1._ensure_runner_manifest(out_root)
    recovery1._load_split_registry(out_root)
    if not first_runner.is_file() or not first_split.is_file():
        raise base.AnalysisError("second recovery requires the immutable first recovery")
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "recovery_git_commit": recovery1._last_commit_for(script),
        "first_recovery_runner_sha256": _sha256_file(first_runner),
        "first_recovery_split_sha256": _sha256_file(first_split),
        "original_analysis_sha256": recovery1._runner_manifest(out_root)[
            "original_analysis_sha256"
        ],
        "policy": (
            "Replace only the frozen 30-item child that exhausted attempts 1 and 2 "
            "with two ordered 15-item packets; preserve all text, ids, rubric, pass, "
            "instrument, and decisions."
        ),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    manifest = _runner_manifest(out_root)
    base._immutable_json(_recovery_root(out_root) / "runner_manifest_v2.json", manifest)
    return manifest


def _first_recovered_jobs(items, instrument, rubrics, rubric_id) -> list[runner.JobSpec]:
    jobs = recovery1._ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, rubric_id)
    registry = recovery1._load_split_registry(_ACTIVE_OUT_ROOT)
    return recovery1.apply_policy_packet_splits(jobs, rubrics, registry)


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
            f"second recovery parent differs from frozen exception: "
            f"realized={realized} expected={expected}"
        )


def _original_target(args) -> tuple[runner.JobSpec, dict[str, str]]:
    global _ACTIVE_OUT_ROOT
    _ACTIVE_OUT_ROOT = args.out_root
    items, instrument, rubrics = base._load_staged(args)
    jobs = _first_recovered_jobs(items, instrument, rubrics, "coherence")
    matches = [job for job in jobs if job.job_id == TARGET_JOB_ID]
    if len(matches) != 1:
        raise base.AnalysisError(f"second recovery target resolves to {len(matches)} jobs")
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
        raise base.AnalysisError("second recovery evidence differs from exact attempts 1 and 2")
    return failures


def register_policy_split(args) -> Path:
    if args.register_policy_split != TARGET_JOB_ID or args.parts != TARGET_PARTS:
        raise base.AnalysisError(
            f"only {TARGET_JOB_ID} with exactly {TARGET_PARTS} parts is authorized"
        )
    _ensure_runner_manifest(args.out_root)
    job, _rubrics = _original_target(args)
    canonical = runner._job_record_path(base.analysis_root(args.out_root), job)
    if canonical.exists():
        raise base.AnalysisError("cannot split a completed canonical child packet")
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
        "children": list(TARGET_CHILDREN),
        "transformation": "contiguous equal-width child-packet split only",
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
    }
    path = (
        _recovery_root(args.out_root)
        / "policy_packet_splits_v2"
        / f"{job.job_id}.json"
    )
    base._immutable_json(path, record)
    print(
        f"[policy-recovery-v2] registered {job.job_id} parts={TARGET_PARTS} "
        f"failures={len(failures)}",
        flush=True,
    )
    return path


def _load_registry(out_root: str | Path) -> dict[str, dict]:
    _ensure_runner_manifest(out_root)
    path = (
        _recovery_root(out_root)
        / "policy_packet_splits_v2"
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
        "job_suffix": "__policy_split01",
        "prompt_sha256": TARGET_PROMPT_SHA256,
        "prompt_tokens_o200k": TARGET_PROMPT_TOKENS,
        "instrument_fp": TARGET_INSTRUMENT_FP,
        "parts": TARGET_PARTS,
        "n_items": TARGET_N_ITEMS,
        "source_item_ids_sha256": TARGET_SOURCE_IDS_SHA256,
        "opaque_item_ids_sha256": TARGET_OPAQUE_IDS_SHA256,
        "children": list(TARGET_CHILDREN),
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
    }
    if any(record.get(key) != value for key, value in expected_static.items()):
        raise base.AnalysisError("second recovery registry differs from frozen pins")
    failures = record.get("failed_attempts")
    if not isinstance(failures, list) or len(failures) != runner.MAX_JOB_ATTEMPTS:
        raise base.AnalysisError("second recovery failure registry is incomplete")
    analysis_root = base.analysis_root(out_root).resolve()
    seen_attempts: set[int] = set()
    for failure in failures:
        attempt_index = failure.get("attempt_index")
        expected_hash = TARGET_FAILURE_SHA256.get(attempt_index)
        failure_path = (analysis_root / str(failure.get("path", ""))).resolve()
        if analysis_root not in failure_path.parents:
            raise base.AnalysisError("second recovery failure path escapes analysis root")
        if (
            expected_hash is None
            or attempt_index in seen_attempts
            or failure.get("sha256") != expected_hash
            or not failure_path.is_file()
            or _sha256_file(failure_path) != expected_hash
        ):
            raise base.AnalysisError("second recovery failed-attempt evidence changed")
        seen_attempts.add(attempt_index)
    if seen_attempts != set(TARGET_FAILURE_SHA256):
        raise base.AnalysisError("second recovery attempts are not exactly 1 and 2")
    parent = (
        analysis_root
        / "jobs/integrity_production/coherence/pass00"
        / "chunk023__policy_split01.json"
    )
    if parent.exists():
        raise base.AnalysisError("canonical child and recovered grandchildren cannot coexist")
    return {TARGET_JOB_ID: record}


def apply_policy_packet_split(
    jobs: list[runner.JobSpec], rubrics: dict[str, str], registry: dict[str, dict]
) -> list[runner.JobSpec]:
    if not registry:
        if any(job.job_id == TARGET_JOB_ID for job in jobs):
            raise base.AnalysisError("second recovery split is not registered")
        return jobs
    output: list[runner.JobSpec] = []
    matched = False
    for job in jobs:
        if job.job_id != TARGET_JOB_ID:
            output.append(job)
            continue
        _validate_target_job(job)
        record = registry[TARGET_JOB_ID]
        width = len(job.items) // TARGET_PARTS
        children = []
        for part_index in range(TARGET_PARTS):
            items = job.items[part_index * width : (part_index + 1) * width]
            prompt = recovery1._part_prompt(job, rubrics[job.rubric_id], items)
            tokens = runner._count_o200k_tokens(prompt)
            child = runner.JobSpec(
                scope=job.scope,
                rubric_id=job.rubric_id,
                pass_index=job.pass_index,
                chunk_index=job.chunk_index,
                items=items,
                prompt=prompt,
                prompt_tokens_o200k=tokens,
                instrument_fp=job.instrument_fp,
                job_suffix=f"{job.job_suffix}__policy_split{part_index:02d}",
            )
            realized = {
                "job_suffix": child.job_suffix,
                "n_items": len(child.items),
                "prompt_tokens_o200k": child.prompt_tokens_o200k,
                "prompt_sha256": base._sha256_text(child.prompt),
            }
            if realized != record["children"][part_index]:
                raise base.AnalysisError("second recovery child differs from frozen registry")
            children.append(child)
        source_ids = [item.source_item_id for child in children for item in child.items]
        opaque_ids = [item.opaque_id for child in children for item in child.items]
        if source_ids != [item.source_item_id for item in job.items] or opaque_ids != [
            item.opaque_id for item in job.items
        ]:
            raise base.AnalysisError("second recovery changed item coverage/order")
        output.extend(children)
        matched = True
    if not matched:
        relevant = any(
            job.scope == "integrity_production" and job.rubric_id == "coherence"
            for job in jobs
        )
        if relevant:
            raise base.AnalysisError("second recovery registry did not match roster")
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("second recovery changed total grading decisions")
    return output


def _twice_recovered_jobs(items, instrument, rubrics, rubric_id):
    jobs = _first_recovered_jobs(items, instrument, rubrics, rubric_id)
    return apply_policy_packet_split(jobs, rubrics, _load_registry(_ACTIVE_OUT_ROOT))


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
    base._production_jobs = _twice_recovered_jobs
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original
        _ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
