"""Freeze independent screening, observed-rate selection and grouped fresh evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from scripts.context_risk_followup import (  # noqa: E402
    CONDITIONS,
    SOURCE_ROWS_SHA,
    SOURCE_SHA,
    _stable_digest,
    _write_json_atomic,
    _write_jsonl_atomic,
    instruction,
    load_samples,
    official_minimal_prompt,
    read_rows,
    sha256,
)

PROJECT = Path(__file__).resolve().parents[1]
DESIGN = PROJECT / "eval_results/context_risk_highrate_design"
MODEL = "openai-api/local/Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
COLLECTION_SOURCES = (
    "scripts/context_risk_highrate_design.py",
    "scripts/context_risk_highrate_collect.py",
    "scripts/context_risk_highrate_supervise.sh",
    "configs/eval/context_risk_highrate.yaml",
    "scripts/context_risk_followup.py",
    "scripts/context_risk_followup_census.py",
    "scripts/context_risk_followup_probe_core.py",
    "scripts/context_risk_impossiblebench.py",
    "scripts/context_risk_impossiblebench_harness.py",
    "scripts/context_risk_impossiblebench_inspect.py",
    "eval_results/context_risk_highrate_design/plan.md",
    "eval_results/context_risk_highrate_design/analysis_spec.json",
)


def source_hashes() -> dict[str, str]:
    """Bind every collection producer and the frozen scientific specification."""
    return {name: sha256(PROJECT / name) for name in COLLECTION_SOURCES}


def validate_review(path: Path) -> dict:
    """Require an independent PASS covering the current complete producer set."""
    review = json.loads(Path(path).read_text())
    if review.get("verdict") != "PASS" or not review.get("reviewer"):
        raise ValueError("Independent collection implementation PASS required")
    expected = source_hashes()
    if any(review.get("files_sha256", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Collection implementation changed after independent review")
    return review


def tie_hash(seed: int, task_id: str) -> str:
    """Return the fixed, outcome-independent tie key."""
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def source_rows(path: Path) -> list[dict]:
    """Read the pinned complete benchmark and verify distinct base specifications."""
    rows = read_rows(path)
    if sha256(path) != SOURCE_SHA or _stable_digest(rows) != SOURCE_ROWS_SHA:
        raise ValueError("Official benchmark bytes or semantic contents differ")
    by_task = defaultdict(list)
    specs = defaultdict(set)
    for row in rows:
        by_task[row["task_id"]].append(row)
        specs[" ".join(row["prompt"].split())].add(row["task_id"])
        if row["dataset_revision"] != "98650ffc3f28a01b261669b6d19fcd7773823710":
            raise ValueError("Benchmark revision differs")
    if set(by_task) != {f"lcbhard_{i}" for i in range(103)} or len(rows) != 309:
        raise ValueError("Expected exactly 103 base tasks and 309 conditions")
    if any(len(v) != 3 or {r["condition"] for r in v} != CONDITIONS for v in by_task.values()):
        raise ValueError("Incomplete or duplicated benchmark condition groups")
    if any(len(ids) != 1 for ids in specs.values()):
        raise ValueError("Distinct base tasks duplicate the same normalized specification")
    return rows


def make_rows(rows: list[dict], phase: str, roles: dict[str, str] | None = None) -> list[dict]:
    """Construct exact published prompts with phase-specific independent seed identities."""
    if phase not in {"screen", "fresh"}:
        raise ValueError("Unknown collection phase")
    result = []
    for row in rows:
        if phase == "fresh" and row["task_id"] not in roles:
            continue
        role = "rate_screening" if phase == "screen" else roles[row["task_id"]]
        messages = [{"role": "user", "content": official_minimal_prompt(row, "B")}]
        result.append(
            {
                **row,
                "historical_partition": row["partition"],
                "partition": role,
                "public_test_role": role,
                "messages": messages,
                "exact_context_sha256": _stable_digest(messages),
                "sample_id": f"highrate_{phase}:B:{row['task_id']}:{row['condition']}",
                "phase": phase,
                "prompt_variant": "B",
            }
        )
    result.sort(key=lambda r: tie_hash(38301, r["sample_id"]))
    if phase == "screen":
        longest = {
            max(
                (r for r in result if r["condition"] == condition),
                key=lambda r: len(r["messages"][0]["content"]),
            )["sample_id"]
            for condition in CONDITIONS
        }
        result.sort(key=lambda r: r["sample_id"] not in longest)
    return result


def check_seed_streams(rows: list[dict]) -> dict:
    """Verify disjoint possible screening/fresh request seeds before any outcomes."""
    previous = set()
    for phase, epochs in (("development", 2), ("fresh", 4)):
        for arm in ("A", "B"):
            for row in rows:
                for epoch in range(1, epochs + 1):
                    for attempt in range(1, 11):
                        identity = f"{phase}:{arm}:{row['task_id']}:{row['condition']}"
                        previous.add(
                            int.from_bytes(
                                hashlib.sha256(
                                    f"38295:{identity}:{epoch}:{attempt}".encode()
                                ).digest()[:4],
                                "big",
                            )
                        )
    seen = {}
    for phase, epochs in (("screen", 2), ("fresh", 4)):
        for row in rows:
            sample_id = f"highrate_{phase}:B:{row['task_id']}:{row['condition']}"
            for epoch in range(1, epochs + 1):
                for attempt in range(1, 11):
                    seed = int.from_bytes(
                        hashlib.sha256(f"38295:{sample_id}:{epoch}:{attempt}".encode()).digest()[
                            :4
                        ],
                        "big",
                    )
                    key = (sample_id, epoch, attempt)
                    if seed in seen:
                        raise ValueError(f"Request-seed collision: {key}, {seen[seed]}")
                    if seed in previous:
                        raise ValueError(f"Request seed overlaps a previous possible draw: {key}")
                    seen[seed] = key
    return {
        "possible_request_seeds": len(seen),
        "unique": True,
        "previous_unique_possible_seeds": len(previous),
        "previous_seed_overlap": 0,
        "previous_scope": "both corrected A/B arms, all 103 tasks, development two and fresh four epochs, ten attempts",
    }


def freeze(source: Path, root: Path, review: Path) -> dict:
    """Freeze the complete independent screening cohort in a fresh namespace."""
    validate_review(review)
    rows = source_rows(source)
    out = root / "manifests"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Screen freeze requires an empty manifest directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.jsonl").write_bytes(source.read_bytes())
    (out / "code_review.json").write_bytes(review.read_bytes())
    manifest = out / "screen_B.jsonl"
    _write_jsonl_atomic(manifest, make_rows(rows, "screen"))
    samples = load_samples(manifest)
    if len(samples) != 309:
        raise ValueError("Screen prompt validation lost rows")
    result = {
        "schema_version": "context_risk_highrate_screen_freeze_v1",
        "passed": True,
        "frozen_unix": time.time(),
        "source_path": str(source.resolve()),
        "source_snapshot": "manifests/source.jsonl",
        "source_sha256": sha256(source),
        "source_rows_sha256": _stable_digest(rows),
        "manifest_sha256": sha256(manifest),
        "plan_sha256": sha256(DESIGN / "plan.md"),
        "sources_sha256": source_hashes(),
        "review_sha256": sha256(review),
        "tasks": 103,
        "contexts": 309,
        "epochs": 2,
        "planned_trajectories": 618,
        "seed_check": check_seed_streams(rows),
    }
    _write_json_atomic(out / "screen_freeze.json", result)
    return result


def load_phase(root: Path, phase: str) -> tuple[Path, dict, int]:
    """Verify the immutable source, roster and prior selection consumed by a phase."""
    root = Path(root)
    if phase not in {"screen", "fresh"}:
        raise ValueError("Unknown phase")
    screen_path = root / "manifests/screen_freeze.json"
    screen = json.loads(screen_path.read_text())
    hashes = source_hashes()
    if (
        screen.get("passed") is not True
        or screen["sources_sha256"] != hashes
        or screen["plan_sha256"] != sha256(DESIGN / "plan.md")
        or screen["source_sha256"] != SOURCE_SHA
    ):
        raise ValueError("Screen freeze or source binding changed")
    if sha256(root / "manifests/screen_B.jsonl") != screen["manifest_sha256"]:
        raise ValueError("Screen manifest changed")
    if sha256(root / "manifests/code_review.json") != screen["review_sha256"]:
        raise ValueError("Frozen independent review changed")
    validate_review(root / "manifests/code_review.json")
    original_rows = source_rows(root / "manifests/source.jsonl")
    if read_rows(root / "manifests/screen_B.jsonl") != make_rows(original_rows, "screen"):
        raise ValueError("Screen manifest does not reproduce the official source")
    receipt = screen
    manifest = root / f"manifests/{phase}_B.jsonl"
    if phase == "fresh":
        receipt = json.loads((root / "selection.json").read_text())
        if (
            receipt.get("passed") is not True
            or receipt["screen_freeze_sha256"] != sha256(screen_path)
            or receipt["sources_sha256"] != hashes
            or receipt["plan_sha256"] != sha256(DESIGN / "plan.md")
            or receipt["screen_native_audit_sha256"] != sha256(root / "screen_B/native_audit.json")
            or receipt["screen_success_review_sha256"]
            != sha256(root / "screen_B/success_review.json")
            or receipt["screen_process_sha256"] != sha256(root / "screen_B/terminal_process.json")
            or receipt["manifest_sha256"] != sha256(manifest)
        ):
            raise ValueError("Fresh selection inputs or manifest changed")
        if Counter(receipt["task_roles"].values()) != {"probe_training": 20, "final_test": 10}:
            raise ValueError("Fresh training/test task allocation differs")
        audit = json.loads((root / "screen_B/native_audit.json").read_text())
        expected_rank, expected_roles = rank_tasks(audit["contexts"])
        if receipt["ranking"] != expected_rank or receipt["task_roles"] != expected_roles:
            raise ValueError("Fresh selection differs from the frozen empirical ranking rule")
        if read_rows(manifest) != make_rows(original_rows, "fresh", expected_roles):
            raise ValueError("Fresh prompts do not reproduce the selected official tasks")
    samples = load_samples(manifest)
    expected_tasks = (
        {f"lcbhard_{i}" for i in range(103)} if phase == "screen" else set(receipt["task_roles"])
    )
    if (
        len(samples) != 3 * len(expected_tasks)
        or {s.metadata["task_id"] for s in samples} != expected_tasks
    ):
        raise ValueError("Phase roster differs from the frozen task set")
    for sample in samples:
        task = sample.metadata["task_id"]
        condition = sample.metadata["condition"]
        role = "rate_screening" if phase == "screen" else receipt["task_roles"][task]
        if (
            sample.id != f"highrate_{phase}:B:{task}:{condition}"
            or sample.metadata["phase"] != phase
            or sample.metadata["public_test_role"] != role
            or sample.metadata["partition"] != role
            or sample.metadata["instruction_prompt"] != instruction("B")
        ):
            raise ValueError("Sample phase, role, instruction or seed identity differs")
    return manifest, receipt, 2 if phase == "screen" else 4


def rank_tasks(contexts: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Rank equal screening exposures, preserving unknowns and deterministic ties."""
    by_task = defaultdict(dict)
    for row in contexts:
        task, condition = row["task_id"], row["condition"]
        if condition not in CONDITIONS or condition in by_task[task]:
            raise ValueError("Duplicate or unknown screening condition")
        values = [row[k] for k in ("success", "failure", "censored", "missing")]
        if any(type(v) is not int or v < 0 for v in values) or sum(values) != 2 or values[3]:
            raise ValueError(
                "Each screen condition needs two realized outcomes, with unknowns explicit"
            )
        by_task[task][condition] = row
    if set(by_task) != {f"lcbhard_{i}" for i in range(103)}:
        raise ValueError("Screening must cover all 103 base tasks before ranking")
    ranked = []
    for task, group in by_task.items():
        if set(group) != CONDITIONS:
            raise ValueError("Task lacks a screening condition")
        counts = {
            k: sum(group[c][k] for c in ("oneoff", "conflicting"))
            for k in ("success", "failure", "censored")
        }
        s, f, u = (counts[k] for k in ("success", "failure", "censored"))
        ranked.append(
            {
                "task_id": task,
                **counts,
                "rate_lower": s / 4,
                "rate_upper": (s + u) / 4,
                "complete_case_rate": s / (s + f) if s + f else None,
                "tie_key": tie_hash(38302, task),
                "by_condition": {
                    c: {k: group[c][k] for k in ("success", "failure", "censored")}
                    for c in sorted(CONDITIONS)
                },
            }
        )
    ranked.sort(key=lambda r: (-r["success"], r["tie_key"]))
    for index, row in enumerate(ranked):
        row["rank"] = index + 1
        row["selected"] = index < 30
    roles = {}
    for start in range(0, 30, 3):
        block = ranked[start : start + 3]
        test_task = min(block, key=lambda r: tie_hash(38303, r["task_id"]))["task_id"]
        roles.update(
            {
                r["task_id"]: "final_test" if r["task_id"] == test_task else "probe_training"
                for r in block
            }
        )
    return ranked, roles


def validate_terminal_process(root: Path, phase: str) -> dict:
    """Recheck the owned terminal process evidence before consuming its outputs."""
    receipt = json.loads((root / phase_dir(phase) / "terminal_process.json").read_text())
    return _validate_terminal_receipt(root, phase, receipt)


def _validate_terminal_receipt(root: Path, phase: str, receipt: dict) -> dict:
    """Validate a receipt before writing it or accepting a previously written copy."""
    # PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
    from inspect_ai.log import read_eval_log

    if receipt.get("verification_passed") is not True or receipt["phase"] != phase:
        raise ValueError("Missing terminal process verification")
    if receipt["run_result_sha256"] != sha256(root / phase_dir(phase) / "run_result.json"):
        raise ValueError("Collection result changed after process verification")
    launch_path = Path(receipt["launch_path"])
    launch = json.loads(launch_path.read_text())
    if launch["phase"] != phase or launch["mode"] != phase:
        raise ValueError("Terminal launch phase differs")
    prefix = root / f"{phase}_{launch['launch_id']}_process"
    exit_path = Path(f"{prefix}.exit.json")
    log_path = Path(launch["log_path"])
    report = json.loads((root / phase_dir(phase) / "run_result.json").read_text())
    expected_paths = {
        str(launch_path),
        str(exit_path),
        str(log_path),
        str(Path(f"{prefix}.pid")),
        str(Path(f"{prefix}.worker.pid")),
        *report["native_logs_sha256"],
    }
    if (
        receipt["exit_path"] != str(exit_path)
        or set(receipt["evidence_sha256"]) != expected_paths
        or not report["native_logs_sha256"]
    ):
        raise ValueError("Terminal process evidence roster differs")
    for path, expected in receipt["evidence_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Terminal process evidence changed: {path}")
    exit_record = json.loads(Path(receipt["exit_path"]).read_text())
    for key in ("supervisor_pid", "worker_pid"):
        if type(receipt[key]) is not int or receipt[key] <= 1 or exit_record[key] != receipt[key]:
            raise ValueError("Invalid or mismatched process identity")
    if exit_record["exit_code"] != 0 or exit_record["cleanup"] != "no_live_members":
        raise ValueError("Collector did not complete and drain normally")
    started = _timestamp(launch["started_utc"])
    if exit_record["mode"] != phase or exit_record["finished_unix"] < started:
        raise ValueError("Terminal exit chronology or phase differs")
    log_text = log_path.read_text()
    for key, suffix, tag in (
        ("supervisor_pid", "pid", "supervisor-start"),
        ("worker_pid", "worker.pid", "worker-start"),
    ):
        if (
            int(Path(f"{prefix}.{suffix}").read_text()) != receipt[key]
            or launch[key] != receipt[key]
            or f"[{tag}] mode={phase} pid={receipt[key]}" not in log_text
        ):
            raise ValueError("Terminal launch process identity differs")
    for location, expected in report["native_logs_sha256"].items():
        if receipt["evidence_sha256"][location] != expected:
            raise ValueError("Terminal native hashes differ from collector report")
        native = read_eval_log(location, header_only=True)
        begin, end = _timestamp(native.stats.started_at), _timestamp(native.stats.completed_at)
        if (
            native.status != "success"
            # Inspect headers truncate start times to whole seconds.
            or not math.floor(started) <= begin <= end <= exit_record["finished_unix"] + 1
        ):
            raise ValueError("Native invocation is not terminal within the owned launch")
    processes = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="], check=True, capture_output=True, text=True
    )
    for line in processes.stdout.splitlines():
        pid, group, state = line.split()
        if not state.startswith(("Z", "X")) and (
            int(pid) == receipt["supervisor_pid"] or int(group) == receipt["worker_pid"]
        ):
            raise ValueError("Owned collector process group is still live")
    return receipt


def phase_dir(phase: str) -> str:
    """Resolve only the two explicitly planned phase directories."""
    if phase not in {"screen", "fresh"}:
        raise ValueError("Unknown phase")
    return f"{phase}_B"


def record_terminal_process(root: Path, phase: str, launch_path: Path) -> dict:
    """Bind a known launch, drained supervisor, native completion and exact output bytes."""
    # PROD_IMPORT_LINT_EXEMPT: Runtime explicitly pinned by uv --with inspect-ai==0.3.261.
    from inspect_ai.log import read_eval_log

    launch = json.loads(launch_path.read_text())
    expected_mode = "screen" if phase == "screen" else "fresh"
    if launch["phase"] != phase or launch["mode"] != expected_mode:
        raise ValueError("Launch phase identity differs")
    result_path = root / phase_dir(phase) / "run_result.json"
    report = json.loads(result_path.read_text())
    prefix = root / f"{launch['mode']}_{launch['launch_id']}_process"
    exit_path = Path(f"{prefix}.exit.json")
    exited = json.loads(exit_path.read_text())
    log_path = Path(launch["log_path"])
    log_text = log_path.read_text()
    started = datetime.fromisoformat(launch["started_utc"].replace("Z", "+00:00")).timestamp()
    if exited["mode"] != expected_mode or exited["finished_unix"] < started:
        raise ValueError("Supervisor exit phase or chronology differs")
    for key, suffix in (("supervisor_pid", "pid"), ("worker_pid", "worker.pid")):
        pid = int(Path(f"{prefix}.{suffix}").read_text())
        if pid != launch[key] or pid != exited[key]:
            raise ValueError("Launch PID evidence differs")
    if (
        f"[supervisor-start] mode={expected_mode} pid={launch['supervisor_pid']}" not in log_text
        or f"[worker-start] mode={expected_mode} pid={launch['worker_pid']}" not in log_text
    ):
        raise ValueError("Supervisor log lacks the pinned launch identity")
    evidence = [
        launch_path,
        exit_path,
        log_path,
        Path(f"{prefix}.pid"),
        Path(f"{prefix}.worker.pid"),
    ]
    for location, digest in report["native_logs_sha256"].items():
        if sha256(Path(location)) != digest:
            raise ValueError("Native log changed after collector reporting")
        native = read_eval_log(location, header_only=True)
        end = _timestamp(native.stats.completed_at)
        begin = _timestamp(native.stats.started_at)
        # Inspect headers truncate start times to whole seconds.
        if (
            native.status != "success"
            or not math.floor(started) <= begin <= end <= exited["finished_unix"] + 1
        ):
            raise ValueError("Native invocation is not terminal within the owned launch")
        evidence.append(Path(location))
    receipt = {
        "verification_passed": True,
        "phase": phase,
        "run_result_sha256": sha256(result_path),
        "supervisor_pid": launch["supervisor_pid"],
        "worker_pid": launch["worker_pid"],
        "launch_path": str(launch_path),
        "exit_path": str(exit_path),
        "evidence_sha256": {str(p): sha256(p) for p in evidence},
        "verified_unix": time.time(),
    }
    target = root / phase_dir(phase) / "terminal_process.json"
    if target.exists():
        raise FileExistsError("Terminal process verification is immutable")
    _validate_terminal_receipt(root, phase, receipt)
    _write_json_atomic(target, receipt)
    return validate_terminal_process(root, phase)


def _timestamp(value: str | datetime) -> float:
    """Normalize the actual native timestamp without supplying missing defaults."""
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None:
        raise ValueError("Native timestamp lacks a timezone")
    return parsed.timestamp()


def select(root: Path) -> dict:
    """Choose the top observed-rate panel only after independent native and code reviews."""
    from scripts.context_risk_highrate_collect import verify_report

    root = Path(root)
    if (root / "selection.json").exists() or (root / "manifests/fresh_B.jsonl").exists():
        raise FileExistsError("Fresh task selection is immutable")
    load_phase(root, "screen")
    audit = verify_report(root, "screen")
    process = validate_terminal_process(root, "screen")
    native_path = root / "screen_B/native_audit.json"
    saved = json.loads(native_path.read_text())
    if (
        saved.get("verification_passed") is not True
        or saved["native_logs_sha256"] != audit["native_logs_sha256"]
        or saved["contexts"] != audit["contexts"]
        or saved["counts"] != audit["counts"]
    ):
        raise ValueError("Independent screening native audit differs")
    review_path = root / "screen_B/success_review.json"
    review = json.loads(review_path.read_text())
    if (
        review.get("verdict") != "PASS"
        or not review.get("reviewer")
        or review["native_logs_sha256"] != audit["native_logs_sha256"]
        or review["success_evidence"] != success_evidence(root, "screen")
    ):
        raise ValueError("Successful screening bodies lack an exact independent review")
    ranking, roles = rank_tasks(audit["contexts"])
    rows = source_rows(root / "manifests/source.jsonl")
    fresh = make_rows(rows, "fresh", roles)
    path = root / "manifests/fresh_B.jsonl"
    _write_jsonl_atomic(path, fresh)
    cutoff = ranking[29]["rate_lower"]
    result = {
        "schema_version": "context_risk_highrate_selection_v1",
        "passed": True,
        "selected_unix": time.time(),
        "selected_arm": "B",
        "screen_freeze_sha256": sha256(root / "manifests/screen_freeze.json"),
        "screen_native_audit_sha256": sha256(native_path),
        "screen_success_review_sha256": sha256(review_path),
        "screen_process_sha256": sha256(root / "screen_B/terminal_process.json"),
        "screen_process_verified_unix": process["verified_unix"],
        "sources_sha256": source_hashes(),
        "plan_sha256": sha256(DESIGN / "plan.md"),
        "manifest_sha256": sha256(path),
        "task_roles": roles,
        "ranking": ranking,
        "cutoff_rate_lower": cutoff,
        "cutoff_tie_tasks": [r["task_id"] for r in ranking if r["rate_lower"] == cutoff],
        "excluded_overlapping_upper_bound": [
            r["task_id"] for r in ranking[30:] if r["rate_upper"] >= cutoff
        ],
        "selected_positive_bearing_screen_tasks": sum(r["success"] > 0 for r in ranking[:30]),
        "contexts": 90,
        "epochs": 4,
        "planned_trajectories": 360,
    }
    _write_json_atomic(root / "selection.json", result)
    load_phase(root, "fresh")
    return result


def success_evidence(root: Path, phase: str) -> dict:
    """Name every successful final body and complete input/history for an independent read."""
    result = {}
    for row in read_rows(root / phase_dir(phase) / "rollouts.jsonl"):
        if row["scores"]["successful_submission"]["value"] != "C":
            continue
        metadata = row["metadata"]
        key = f"{row['sample_id']}:epoch{row['epoch']}"
        if key in result:
            raise ValueError("Duplicate successful trajectory identity")
        result[key] = {
            "exact_context_sha256": metadata["exact_context_sha256"],
            "history_sha256": _stable_digest(metadata["agentic_results"]),
            "row_sha256": _stable_digest(row),
        }
    return result
