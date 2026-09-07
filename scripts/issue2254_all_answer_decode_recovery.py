#!/usr/bin/env python3
"""Audited policy-packet recovery for the frozen #2254 decode-match grader.

The primary analysis runner is byte-pinned in its staged manifests.  This
wrapper leaves those bytes and every grading instruction unchanged.  It only
replaces a packet that has exhausted both fresh attempts at the upstream
content filter with deterministic, contiguous sub-packets.  Each recovery is
registered from the immutable failed-attempt records before it can be used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_revmap8_subagent_grade as runner


RECOVERY_VERSION = "issue2254-policy-packet-recovery-v1"
POLICY_FLAG = "content was flagged for possible biological risk"
TARGET_JOB_ID = "integrity_production__coherence__pass00__chunk023"
TARGET_PROMPT_SHA256 = "e3e27d170be0b00d1d71859ef23a6f173e608ba2fe0345266c3a8c3ec61706ef"
TARGET_SOURCE_IDS_SHA256 = "59d212dad526c066c42cf3424657a0ccb8e0da160551e4ad0d8048e5e24aa18d"
TARGET_OPAQUE_IDS_SHA256 = "b6f47d1a5b8043ce50f9f9a7c988b543cfceaa050fa87336e88bb45c410b8a69"
TARGET_INSTRUMENT_FP = "b779778050d06771805ef65d79ed74454fc2ac89b4067c25272662f90c380606"
TARGET_N_ITEMS = 60
TARGET_PROMPT_TOKENS = 38_001
TARGET_PARTS = 2
TARGET_FAILURE_SHA256 = {
    1: "ec9ad6f81a5cebfb6691e64764f1d0cdbae34676cbbb387be2ffefa8f40d17be",
    2: "4a4503c8fe7242ea061d242b800aa51dc74e7e77527bc7b26df28e299ade0f44",
}
TARGET_CHILDREN = (
    {
        "job_suffix": "__policy_split00",
        "n_items": 30,
        "prompt_tokens_o200k": 20_198,
        "prompt_sha256": "972027cd0b833c0adb7dee7bc80bdbd5ca89548252ac37afbcf7fa2f39db1513",
    },
    {
        "job_suffix": "__policy_split01",
        "n_items": 30,
        "prompt_tokens_o200k": 18_533,
        "prompt_sha256": "cb41fd503164532922e874347db962096d2c449a6d20bef4cfafff60987804a6",
    },
)
_ORIGINAL_PRODUCTION_JOBS = base._production_jobs
_ACTIVE_OUT_ROOT: str | Path | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_root(out_root: str | Path) -> Path:
    return base.analysis_root(out_root) / "recovery"


def _last_commit_for(path: Path) -> str:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(base._REPO_ROOT))],
        cwd=base._REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = proc.stdout.strip()
    if proc.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise base.AnalysisError("recovery script must be committed before registration")
    return commit


def _runner_manifest(out_root: str | Path) -> dict:
    staged = base.analysis_root(out_root) / "runner_manifest.json"
    script = Path(__file__).resolve()
    if not staged.is_file():
        raise base.AnalysisError("policy recovery requires the frozen analysis stage")
    staged_record = json.loads(staged.read_text(encoding="utf-8"))
    return {
        "version": RECOVERY_VERSION,
        "script_path": str(script.relative_to(base._REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "recovery_git_commit": _last_commit_for(script),
        "original_staging_git_commit": staged_record.get("git_commit"),
        "original_analysis_sha256": staged_record.get("dependency_sha256", {}).get(
            "scripts/issue2254_all_answer_decode_analysis.py"
        ),
        "staged_runner_manifest_sha256": _sha256_file(staged),
        "policy": (
            "Only exact packets with at least two immutable matching upstream "
            "content-filter failures may be split; item text, order within each "
            "contiguous part, rubric, pass, and instrument fingerprint are unchanged."
        ),
    }


def _ensure_runner_manifest(out_root: str | Path) -> dict:
    path = _recovery_root(out_root) / "runner_manifest.json"
    manifest = _runner_manifest(out_root)
    base._immutable_json(path, manifest)
    return manifest


def _all_original_production_jobs(args) -> list[runner.JobSpec]:
    items, instrument, rubrics = base._load_staged(args)
    jobs: list[runner.JobSpec] = []
    for rubric_id in ("coherence", "trait_evil", "trait_sycophancy"):
        jobs.extend(_ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, rubric_id))
    return jobs


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
            f"policy recovery parent differs from the single frozen exception: "
            f"realized={realized} expected={expected}"
        )


def _matching_policy_failures(out_root: str | Path, job: runner.JobSpec) -> list[dict]:
    attempts = (
        base.analysis_root(out_root)
        / "attempts"
        / job.scope
        / job.rubric_id
    )
    matched: list[dict[str, str]] = []
    for path in sorted(attempts.glob(f"{job.job_id}.*.failed.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "failed_transport":
            continue
        if record.get("job_id") != job.job_id:
            continue
        if record.get("prompt_sha256") != base._sha256_text(job.prompt):
            continue
        if record.get("instrument_fp") != job.instrument_fp:
            continue
        if record.get("source_item_ids") != [item.source_item_id for item in job.items]:
            continue
        if record.get("request") != job.prompt:
            continue
        evidence = f"{record.get('stdout', '')}\n{record.get('stderr', '')}".casefold()
        if POLICY_FLAG not in evidence:
            continue
        attempt_index = record.get("attempt_index")
        if type(attempt_index) is not int:
            continue
        matched.append(
            {
                "attempt_index": attempt_index,
                "path": str(path.relative_to(base.analysis_root(out_root))),
                "sha256": _sha256_file(path),
            }
        )
    return matched


def register_policy_split(args) -> Path:
    """Freeze one split only after validating both exhausted filter failures."""
    if not re.fullmatch(r"[a-z0-9_]+", args.register_policy_split):
        raise base.AnalysisError("unsafe or malformed recovery job id")
    if args.register_policy_split != TARGET_JOB_ID or args.parts != TARGET_PARTS:
        raise base.AnalysisError(
            f"only {TARGET_JOB_ID} with exactly {TARGET_PARTS} parts is authorized"
        )
    _ensure_runner_manifest(args.out_root)
    matches = [
        job
        for job in _all_original_production_jobs(args)
        if job.job_id == args.register_policy_split
    ]
    if len(matches) != 1:
        raise base.AnalysisError(
            f"recovery job id resolves to {len(matches)} original production packets"
        )
    job = matches[0]
    _validate_target_job(job)
    canonical = runner._job_record_path(base.analysis_root(args.out_root), job)
    if canonical.exists():
        raise base.AnalysisError(f"cannot split a completed canonical packet: {canonical}")
    failures = _matching_policy_failures(args.out_root, job)
    by_attempt = {record["attempt_index"]: record for record in failures}
    if (
        len(failures) != runner.MAX_JOB_ATTEMPTS
        or set(by_attempt) != set(TARGET_FAILURE_SHA256)
        or any(
            by_attempt[index]["sha256"] != expected
            for index, expected in TARGET_FAILURE_SHA256.items()
        )
    ):
        raise base.AnalysisError(
            f"{job.job_id}: failed-attempt evidence differs from exact attempts 1 and 2"
        )
    record = {
        "version": RECOVERY_VERSION,
        "job_id": job.job_id,
        "scope": job.scope,
        "rubric_id": job.rubric_id,
        "pass_index": job.pass_index,
        "chunk_index": job.chunk_index,
        "prompt_sha256": base._sha256_text(job.prompt),
        "prompt_tokens_o200k": job.prompt_tokens_o200k,
        "instrument_fp": job.instrument_fp,
        "parts": args.parts,
        "n_items": len(job.items),
        "source_item_ids_sha256": base._canonical_sha256(
            [item.source_item_id for item in job.items]
        ),
        "opaque_item_ids_sha256": base._canonical_sha256(
            [item.opaque_id for item in job.items]
        ),
        "failed_attempts": failures,
        "children": list(TARGET_CHILDREN),
        "transformation": "contiguous equal-width packet split only",
        "response_or_rubric_text_changed": False,
        "items_dropped_or_duplicated": False,
    }
    path = _recovery_root(args.out_root) / "policy_packet_splits" / f"{job.job_id}.json"
    base._immutable_json(path, record)
    print(
        f"[policy-recovery] registered {job.job_id} parts={args.parts} "
        f"failures={len(failures)}",
        flush=True,
    )
    return path


def _load_split_registry(out_root: str | Path) -> dict[str, dict]:
    _ensure_runner_manifest(out_root)
    root = _recovery_root(out_root) / "policy_packet_splits"
    records: dict[str, dict] = {}
    if not root.exists():
        return records
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        job_id = record.get("job_id")
        if (
            not isinstance(job_id, str)
            or path.name != f"{job_id}.json"
            or record.get("version") != RECOVERY_VERSION
            or job_id in records
        ):
            raise base.AnalysisError(f"malformed/duplicate policy recovery record: {path}")
        if job_id != TARGET_JOB_ID:
            raise base.AnalysisError(f"unauthorized policy recovery record: {path}")
        failures = record.get("failed_attempts")
        if not isinstance(failures, list) or len(failures) != runner.MAX_JOB_ATTEMPTS:
            raise base.AnalysisError("policy recovery failure registry is incomplete")
        seen_attempts: set[int] = set()
        analysis_root = base.analysis_root(out_root).resolve()
        for failure in failures:
            attempt_index = failure.get("attempt_index")
            expected_hash = TARGET_FAILURE_SHA256.get(attempt_index)
            failure_path = (analysis_root / str(failure.get("path", ""))).resolve()
            if analysis_root not in failure_path.parents:
                raise base.AnalysisError("policy recovery failure path escapes analysis root")
            if (
                expected_hash is None
                or attempt_index in seen_attempts
                or failure.get("sha256") != expected_hash
                or not failure_path.is_file()
                or _sha256_file(failure_path) != expected_hash
            ):
                raise base.AnalysisError("policy recovery failed-attempt evidence changed")
            seen_attempts.add(attempt_index)
        if seen_attempts != set(TARGET_FAILURE_SHA256):
            raise base.AnalysisError("policy recovery attempts are not exactly 1 and 2")
        parent_path = (
            analysis_root
            / "jobs"
            / str(record.get("scope"))
            / str(record.get("rubric_id"))
            / f"pass{int(record.get('pass_index')):02d}"
            / f"chunk{int(record.get('chunk_index')):03d}.json"
        )
        if parent_path.exists():
            raise base.AnalysisError("canonical parent and policy-split children cannot coexist")
        records[job_id] = record
    return records


def _part_prompt(job: runner.JobSpec, rubric: str, items: tuple) -> str:
    if job.rubric_id == "coherence":
        return base._integrity_prompt(list(items))
    return runner._prompt(job.rubric_id, rubric, list(items))


def apply_policy_packet_splits(
    jobs: list[runner.JobSpec], rubrics: dict[str, str], registry: dict[str, dict]
) -> list[runner.JobSpec]:
    """Replace registered parents with complete deterministic child rosters."""
    if not jobs:
        return jobs
    scope_rubric = {(job.scope, job.rubric_id) for job in jobs}
    if len(scope_rubric) != 1:
        raise base.AnalysisError("policy recovery expects one production rubric roster")
    scope, rubric_id = next(iter(scope_rubric))
    relevant = {
        job_id: record
        for job_id, record in registry.items()
        if (record.get("scope"), record.get("rubric_id")) == (scope, rubric_id)
    }
    output: list[runner.JobSpec] = []
    matched: set[str] = set()
    for job in jobs:
        record = relevant.get(job.job_id)
        if record is None:
            output.append(job)
            continue
        _validate_target_job(job)
        realized = {
            "prompt_sha256": base._sha256_text(job.prompt),
            "prompt_tokens_o200k": job.prompt_tokens_o200k,
            "instrument_fp": job.instrument_fp,
            "n_items": len(job.items),
            "source_item_ids_sha256": base._canonical_sha256(
                [item.source_item_id for item in job.items]
            ),
            "opaque_item_ids_sha256": base._canonical_sha256(
                [item.opaque_id for item in job.items]
            ),
        }
        for key, value in realized.items():
            if record.get(key) != value:
                raise base.AnalysisError(
                    f"{job.job_id}: recovery pin {key} changed: {value!r} != "
                    f"{record.get(key)!r}"
                )
        parts = record.get("parts")
        if parts != TARGET_PARTS or len(job.items) % parts:
            raise base.AnalysisError(f"{job.job_id}: invalid registered parts={parts!r}")
        width = len(job.items) // parts
        children: list[runner.JobSpec] = []
        for part_index in range(parts):
            part_items = job.items[part_index * width : (part_index + 1) * width]
            prompt = _part_prompt(job, rubrics[job.rubric_id], part_items)
            tokens = runner._count_o200k_tokens(prompt)
            if len(part_items) > base.MAX_ITEMS_PER_JOB or tokens > base.PROMPT_TOKEN_CAP:
                raise base.AnalysisError(
                    f"{job.job_id}: recovered part {part_index} violates a session ceiling"
                )
            children.append(
                runner.JobSpec(
                    scope=job.scope,
                    rubric_id=job.rubric_id,
                    pass_index=job.pass_index,
                    chunk_index=job.chunk_index,
                    items=part_items,
                    prompt=prompt,
                    prompt_tokens_o200k=tokens,
                    instrument_fp=job.instrument_fp,
                    job_suffix=f"__policy_split{part_index:02d}",
                )
            )
            expected_child = TARGET_CHILDREN[part_index]
            realized_child = {
                "job_suffix": children[-1].job_suffix,
                "n_items": len(part_items),
                "prompt_tokens_o200k": tokens,
                "prompt_sha256": base._sha256_text(prompt),
            }
            if realized_child != expected_child or record.get("children") != list(
                TARGET_CHILDREN
            ):
                raise base.AnalysisError(
                    f"{job.job_id}: recovered child differs from frozen registry"
                )
        flattened = [item.source_item_id for child in children for item in child.items]
        original = [item.source_item_id for item in job.items]
        if flattened != original or len(flattened) != len(set(flattened)):
            raise base.AnalysisError(f"{job.job_id}: recovery changed item coverage/order")
        output.extend(children)
        matched.add(job.job_id)
    if matched != set(relevant):
        raise base.AnalysisError(
            f"policy recovery registry did not match roster: {sorted(set(relevant) - matched)}"
        )
    if sum(len(job.items) for job in output) != sum(len(job.items) for job in jobs):
        raise base.AnalysisError("policy recovery changed total grading decisions")
    return output


def _recovered_production_jobs(items, instrument, rubrics, rubric_id):
    if _ACTIVE_OUT_ROOT is None:
        raise base.AnalysisError("policy recovery wrapper has no active output root")
    jobs = _ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, rubric_id)
    registry = _load_split_registry(_ACTIVE_OUT_ROOT)
    return apply_policy_packet_splits(jobs, rubrics, registry)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phases")
    mode.add_argument("--register-policy-split")
    parser.add_argument("--parts", type=int, default=2)
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
    if args.register_policy_split:
        register_policy_split(args)
        return
    phases = [phase.strip() for phase in args.phases.split(",") if phase.strip()]
    if "stage" in phases:
        raise SystemExit("the recovery wrapper cannot restage the frozen instrument")
    unknown = sorted(set(phases) - set(base.PHASES))
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; choices={sorted(base.PHASES)}")
    _ACTIVE_OUT_ROOT = args.out_root
    _ensure_runner_manifest(args.out_root)
    original = base._production_jobs
    base._production_jobs = _recovered_production_jobs
    try:
        for phase in phases:
            print(f"[phase={phase}]", flush=True)
            base.PHASES[phase](args)
    finally:
        base._production_jobs = original
        _ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
