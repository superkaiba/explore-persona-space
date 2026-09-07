"""Blocking, immutable-input RunPod launcher for the issue 952 China GPU phases.

Only Qwen generation/capture and Hub persistence run here. Judges and downstream
analysis remain VM-owned. Resume state is outside the poller's drained namespace.
"""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from functools import lru_cache
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHILD = "scripts/issue952_china_definitive_gpu.py"
SELF = "scripts/issue952_china_dispatch.py"
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue952_position_divergence/followups/china_refusal_topic_stratified_bilingual_v1"
PHASES = ("gen", "upload-raw", "capture", "upload-capture", "finalize")
DONE = "issue952_china_definitive_done.json"
HALT_RC = 7
WALL_SECONDS = 8 * 3600


class DesignedHalt(RuntimeError):
    """A persisted negative smoke verdict, distinct from a crashed phase."""


def digest(path: Path, *, git_blob: bool = False) -> str:
    """Stream a file hash; Git blobs include their object header."""
    value = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object without a missing/malformed fallback."""
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return result


def atomic_json(path: Path, value: Any) -> None:
    """Persist and atomically publish state on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_sha(value: str) -> str:
    """Reject branch names and abbreviated revision identities."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("revision must be a full immutable 40-hex commit SHA")
    return value


def positive_attempt(value: str) -> int:
    """Fresh invalidated generations require a new positive attempt namespace."""
    attempt = int(value)
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return attempt


def assert_code(expected: str) -> None:
    """Verify HEAD and actual source bytes, including stale-served FUSE bytes."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != expected:
        raise RuntimeError(f"code identity mismatch: expected {expected}, got {actual}")
    listing = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "HEAD",
            "--",
            "scripts",
            "src",
            "configs",
            "uv.lock",
            "pyproject.toml",
        ],
        cwd=ROOT,
        text=True,
    )
    entries = [line.split("\t", 1) for line in listing.splitlines()]
    if not entries:
        raise RuntimeError("empty committed source census")
    for metadata, name in entries:
        if metadata.split()[1] != "blob":
            continue
        path = ROOT / name
        if not path.is_file() or digest(path, git_blob=True) != metadata.split()[2]:
            raise RuntimeError(f"uncommitted or stale source bytes: {name}")


def census(root: Path, paths: list[Path]) -> dict[str, str]:
    """Hash a nonempty exact file set relative to its owner."""
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError(f"missing/empty expected artifact set under {root}")
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}


def phase_files(root: Path, phase: str) -> list[Path]:
    """Declare immutable outputs used to resume generation and capture."""
    if phase == "gen":
        return [
            root / "manifests/generation.json",
            root / "manifests/input_stage.json",
            *sorted((root / "raw_completions").rglob("*.*")),
        ]
    if phase == "capture":
        return [root / "manifests/capture.json", *sorted((root / "analysis_tensors").glob("*.pt"))]
    raise ValueError(phase)


def stage_inputs(run_root: Path, revision: str) -> None:
    """Download audited inputs from immutable Hub commits using the shared helper."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.orchestrate import hub

    local = run_root / "inputs"
    marker = local / "upload_verified.json"
    hub.stage_hub_file(
        HF_REPO, f"{HF_PREFIX}/inputs/{marker.name}", marker, revision=revision, overwrite=True
    )
    metadata = read_json(marker)
    data_revision = immutable_sha(metadata["data_revision"])
    for name, key in (
        ("prompt_bank.jsonl", "prompt_bank_sha256"),
        ("bank_audit_report.json", "bank_audit_report_sha256"),
    ):
        target = local / name
        hub.stage_hub_file(
            HF_REPO, f"{HF_PREFIX}/inputs/{name}", target, revision=data_revision, overwrite=True
        )
        if digest(target) != metadata[key]:
            raise RuntimeError(f"staged input hash mismatch: {name}")
    audit = read_json(local / "bank_audit_report.json")
    if (
        audit.get("passed") is not True
        or audit["prompt_bank_sha256"] != metadata["prompt_bank_sha256"]
    ):
        raise RuntimeError("bank audit is not passed and hash-bound")


def command(args: argparse.Namespace, leg: str, phase: str) -> list[str]:
    """Compose the same child CLI for smoke and production, with isolated roots."""
    cmd = [
        sys.executable,
        "-u",
        CHILD,
        "--phase",
        phase,
        "--attempt",
        str(args.attempt),
        "--out-root",
        str(args.run_root / leg),
        "--bank",
        str(args.run_root / "inputs/prompt_bank.jsonl"),
        "--audit",
        str(args.run_root / "inputs/bank_audit_report.json"),
    ]
    if leg == "smoke":
        cmd.append("--smoke")
    else:
        cmd.extend(["--smoke-report", str(args.run_root / "smoke/manifests/smoke_timing.json")])
    return cmd


def run_process(cmd: list[str], log: Path, deadline: float) -> int:
    """Wait for the child, logging heartbeats; fence and reap its own process group."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        UV_NO_SYNC="1",
        UV_OFFLINE="1",
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    # No judge/API stage is authorized. Empty values also prevent dotenv re-injection.
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BATCH_KEY", "OPENAI_API_KEY"):
        env[key] = ""
    with log.open("x") as output:
        child = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("registered 8-hour wall fence reached")
                try:
                    rc = child.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(
                        f"[heartbeat] child_pid={child.pid} log={log} remaining_s={remaining:.0f}",
                        flush=True,
                    )
        finally:
            # Includes successful parents leaving vLLM descendants behind.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The owned group has already exited; no foreign PID is selected.
            child.wait()
    if rc:
        with log.open(errors="replace") as stream:
            for line in deque(stream, maxlen=120):
                print(line.rstrip().replace("[phase=", "[child_phase="), flush=True)
    print(f"[child_exit] rc={rc} log={log}", flush=True)
    return rc


def validate_smoke(run_root: Path, code_sha: str) -> dict:
    """Bind the timing verdict to current input/code and complete phase manifests."""
    root = run_root / "smoke"
    report = read_json(root / "manifests/smoke_timing.json")
    gen = read_json(root / "manifests/generation.json")
    cap = read_json(root / "manifests/capture.json")
    compatibility = {
        k: v
        for k, v in gen["regime"].items()
        if k
        not in {
            "smoke",
            "prompt_token_max",
            "max_model_len",
            "n_selected_prompts",
            "selected_source_ids_sha256",
        }
    }
    capture_compatibility = {
        "capture_regime": {
            k: v
            for k, v in cap["capture_regime"].items()
            if k
            not in {
                "rollouts_sha256",
                "n_selected_prompts",
                "selected_source_ids_sha256",
                "generation_fingerprint",
            }
        },
        "package_versions": cap["package_versions"],
    }
    if (
        gen["regime"]["git_sha"] != code_sha
        or cap["capture_regime"]["git_sha"] != code_sha
        or report["bank_sha256"] != digest(run_root / "inputs/prompt_bank.jsonl")
        or report["generation_compatibility"] != compatibility
        or report["capture_compatibility"] != capture_compatibility
    ):
        raise RuntimeError("stale smoke code/input/phase provenance")
    if report.get("passed") is not True:
        raise DesignedHalt("smoke technical/telemetry gate refused production")
    if report.get("projected_timing_passed") is False:
        print(
            "[smoke_advisory] projected timing threshold missed; production continues under v17",
            flush=True,
        )
    return report


def verify_terminal(run_root: Path, leg: str, attempt: int = 1) -> None:
    """Verify a scoped immutable remote snapshot against every persisted local output."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    root = run_root / leg
    done = read_json(root / DONE)
    gen, cap = done["generation"], done["capture"]
    if (
        done["status"] != "done"
        or gen["n_rows"] != cap["n_answer_rows"]
        or gen["regime"]["smoke"] != (leg == "smoke")
        or gen["regime"]["attempt"] != attempt
        or gen != read_json(root / "manifests/generation.json")
        or cap != read_json(root / "manifests/capture.json")
    ):
        raise RuntimeError("invalid terminal generation/capture provenance")
    paths = [
        root / DONE,
        *sorted((root / "manifests").glob("*.json")),
        *sorted((root / "raw_completions").glob("*")),
        *sorted((root / "analysis_tensors").glob("*.pt")),
    ]
    hashes = census(root, paths)
    prefix = f"{HF_PREFIX}/attempt{attempt}"
    if leg == "smoke":
        prefix += "/smoke"
    print(f"[verify] expected_paths={list(hashes)} count={len(hashes)}", flush=True)
    api = HfApi()
    revision = immutable_sha(
        hub.retry_transient(
            lambda: api.repo_info(HF_REPO, repo_type="dataset", revision="main").sha,
            what="issue952 terminal snapshot resolution",
        )
    )
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: the enclosing retry_transient retries full pagination.
            api.list_repo_tree(
                HF_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True, revision=revision
            )
        ),
        what="issue952 terminal exact-set and content verification",
    )
    remote = {entry.path: entry for entry in entries}
    for relative, sha in hashes.items():
        path = root / relative
        entry = remote.get(f"{prefix}/{relative}")
        if entry is None or entry.size != path.stat().st_size:
            raise RuntimeError(f"missing/wrong-size remote artifact: {relative}")
        expected = sha if entry.lfs else digest(path, git_blob=True)
        actual = entry.lfs.sha256 if entry.lfs else entry.blob_id
        if actual != expected:
            raise RuntimeError(f"remote content mismatch: {relative}")
    atomic_json(
        run_root / "dispatch_state" / f"{leg}_verified.json",
        {"revision": revision, "hf_prefix": prefix, "sha256": hashes},
    )


def emit_sentinel(
    logs: Path, kind: str, note: dict, *, gate: str = "results", blocks_pipeline: bool | None = None
) -> Path:
    """Publish exactly one fresh envelope; never read a drained sentinel back."""
    path = logs / f"issue-952-{kind.replace(':', '_')}-{time.time_ns()}.json"
    atomic_json(
        path,
        {
            "sentinel_schema_version": 1,
            "kind": kind,
            "version": 1,
            "task_id": 952,
            "gate": gate,
            "blocks_pipeline": gate != "results" if blocks_pipeline is None else blocks_pipeline,
            "by": "issue952_china_dispatch",
            "ts": time.time(),
            "note": note,
        },
    )
    return path


def gate_identity(args: argparse.Namespace) -> dict:
    """Bind the external Codex smoke verdict to the exact uploaded GPU evidence."""
    root = args.run_root
    receipt_path = root / "dispatch_state/smoke_verified.json"
    receipt = read_json(receipt_path)
    gen = read_json(root / "smoke/manifests/generation.json")
    return {
        "code_sha": args.expected_code_sha,
        "input_revision": args.input_revision,
        "attempt": args.attempt,
        "accepted_source_ids_sha256": gen["regime"]["accepted_source_ids_sha256"],
        "smoke_report_sha256": digest(root / "smoke/manifests/smoke_timing.json"),
        "smoke_rollouts_sha256": digest(root / "smoke/raw_completions/rollouts.jsonl"),
        "smoke_upload_revision": immutable_sha(receipt["revision"]),
        "smoke_upload_receipt_sha256": digest(receipt_path),
    }


def evidence_spec(gate: dict, role: str) -> tuple[str, str]:
    """Limit externally named evidence to safe relative judge paths and full hashes."""
    spec = gate["evidence"][role]
    name, sha = spec["path"], spec["sha256"]
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "judge"
        or str(path) != name
        or re.fullmatch(r"[0-9a-f]{64}", sha) is None
    ):
        raise RuntimeError(f"unsafe/malformed Codex smoke evidence path/hash: {role}")
    return name, sha


def evidence_local(args: argparse.Namespace, role: str) -> Path:
    """Keep hydrated evidence names local and independent of remote path spellings."""
    suffix = ".jsonl" if role == "result" else ".json"
    return args.run_root / "dispatch_state/gate_evidence" / f"{role}{suffix}"


def artifact_local(args: argparse.Namespace, gate: dict, relative: str) -> Path:
    """Resolve one safe full-census member without trusting saved VM-local paths."""
    evidence_spec(
        {"evidence": {"artifact": {"path": relative, "sha256": gate["artifact_census"][relative]}}},
        "artifact",
    )
    for role in ("request", "result", "parse"):
        if gate["evidence"][role]["path"] == relative:
            return evidence_local(args, role)
    return args.run_root / "dispatch_state/gate_artifacts" / relative


@lru_cache(maxsize=1)
def judge_contract():
    """Load committed local packet validators only; these helpers never call a model API."""
    spec = importlib.util.spec_from_file_location(
        "issue952_packet_contract", ROOT / "scripts/issue952_codex_judges.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("missing committed Codex packet contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_packet_census(
    args: argparse.Namespace,
    gate: dict,
    request: dict,
    parsed: dict,
    smoke: list[dict],
    results: list[dict],
) -> None:
    """Verify exact original packets, opaque links and raw outputs with the real parser."""
    census_hashes = gate["artifact_census"]
    expected = {spec["path"] for spec in gate["evidence"].values()}
    packet_name, lookup_name = "judge/smoke_packet_manifest.json", "judge/smoke_lookup.json"
    expected.update((packet_name, lookup_name))
    for relative, sha in census_hashes.items():
        if digest(artifact_local(args, gate, relative)) != sha:
            raise RuntimeError(f"Codex smoke artifact census content mismatch: {relative}")
    packet_path = artifact_local(args, gate, packet_name)
    lookup_path = artifact_local(args, gate, lookup_name)
    packet_manifest = read_json(packet_path)
    lookup = json.loads(lookup_path.read_text())
    contract = judge_contract()
    smoke_by_id = {row["item_id"]: row for row in smoke}
    mapping_keys = ("opaque_id", "item_id", "primary_agent", "assigned_agents")
    mapping = [{key: row[key] for key in mapping_keys} for row in lookup]
    if (
        mapping != packet_manifest["mapping"]
        or [row["item_id"] for row in lookup] != list(smoke_by_id)
        or len({row["opaque_id"] for row in lookup}) != len(lookup)
        or packet_manifest["packet_kind"] != f"gpu-smoke-attempt{args.attempt}"
        or packet_manifest["backend"] != contract.BACKEND
        or packet_manifest["rubric_sha256"] != contract.RUBRIC_SHA256
        or packet_manifest["n_unique_items"] != len(smoke)
        or packet_manifest["n_assignments"] != sum(len(row["assigned_agents"]) for row in lookup)
        or packet_manifest["n_overlap"] != sum(len(row["assigned_agents"]) == 2 for row in lookup)
        or request["lookup_sha256"] != digest(lookup_path)
        or request["packet_manifest_sha256"] != digest(packet_path)
        or parsed["lookup_sha256"] != digest(lookup_path)
        or parsed["packet_manifest_sha256"] != digest(packet_path)
        or parsed["agent_artifact_hashes"] != gate["agent_artifact_hashes"]
    ):
        raise RuntimeError("Codex smoke packet/lookup manifest links invalid")
    by_opaque = {row["opaque_id"]: row for row in lookup}
    for row in lookup:
        primary, assigned = contract._assignment(
            row["opaque_id"], contract.PRODUCTION_OVERLAP_FRACTION
        )
        source = smoke_by_id[row["item_id"]]
        if (
            row["opaque_id"] != contract._opaque_id(row["item_id"], packet_manifest["packet_kind"])
            or row["primary_agent"] != primary
            or row["assigned_agents"] != assigned
            or row["source_prompt_id"] != source["source_prompt_id"]
            or row["language"] != source["language"]
        ):
            raise RuntimeError("Codex smoke opaque lookup or assignment mismatch")
    rebound, raw_hashes = [], {}
    for packet in packet_manifest["packets"]:
        key = f"{packet['agent']}/batch_{packet['batch_index']:03d}"
        packet_name = f"judge/agent_artifacts/gpu_smoke/{key}.packet.json"
        output_name = f"judge/agent_artifacts/gpu_smoke/{key}.output.jsonl"
        expected.update((packet_name, output_name))
        packet_file = artifact_local(args, gate, packet_name)
        output_file = artifact_local(args, gate, output_name)
        payload = read_json(packet_file)
        expected_items = []
        for item in payload["items"]:
            mapping_row = by_opaque[item["opaque_id"]]
            source = smoke_by_id[mapping_row["item_id"]]
            if packet["agent"] not in mapping_row["assigned_agents"]:
                raise RuntimeError("Codex smoke packet has unassigned agent/item")
            expected_items.append(
                {
                    "opaque_id": item["opaque_id"],
                    "question": source["question"],
                    "response": source["text"],
                }
            )
        if (
            payload != contract._packet_payload(expected_items)
            or len(expected_items) != packet["n_items"]
            or digest(packet_file) != packet["packet_sha256"]
            or key in raw_hashes
        ):
            raise RuntimeError("Codex smoke exact packet/rubric/payload mismatch")
        raw_hashes[key] = {
            "packet_sha256": digest(packet_file),
            "output_sha256": digest(output_file),
        }
        rebound.append({**packet, "packet_path": str(packet_file), "output_path": str(output_file)})
    if set(census_hashes) != expected or raw_hashes != gate["agent_artifact_hashes"]:
        raise RuntimeError("Codex smoke full artifact census mismatch")
    judgments = contract._load_agent_outputs({**packet_manifest, "packets": rebound})
    for agent in contract.AGENTS:
        if set(judgments[agent]) != {
            row["opaque_id"] for row in lookup if agent in row["assigned_agents"]
        }:
            raise RuntimeError("Codex smoke raw output assignment coverage mismatch")
    lookup_by_id = {row["item_id"]: row for row in lookup}
    for result in results:
        row = lookup_by_id[result["item_id"]]
        if (
            result["verdict"] != judgments[row["primary_agent"]][row["opaque_id"]]
            or result["judge_id"] != row["primary_agent"]
        ):
            raise RuntimeError("Codex smoke parsed score does not match raw output")


def read_jsonl(path: Path) -> list[dict]:
    """Strict JSONL parsing with newline-safe iteration and no skipped invalid rows."""
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"empty/malformed required JSONL evidence: {path}")
    return rows


def validate_gate_evidence(args: argparse.Namespace, gate: dict) -> None:
    """Verify actual request/result/parse files and exact realized smoke row coverage."""
    if set(gate["evidence"]) != {"request", "result", "parse"}:
        raise RuntimeError("Codex smoke requires request/result/parse evidence")
    names = []
    for role in ("request", "result", "parse"):
        name, sha = evidence_spec(gate, role)
        names.append(name)
        if digest(evidence_local(args, role)) != sha:
            raise RuntimeError(f"Codex smoke evidence content mismatch: {role}")
        if role != "parse" and gate[f"{role}_sha256"] != sha:
            raise RuntimeError(f"Codex smoke top-level evidence hash mismatch: {role}")
    if len(set(names)) != 3:
        raise RuntimeError("Codex smoke evidence paths must be distinct")
    smoke = read_jsonl(args.run_root / "smoke/raw_completions/rollouts.jsonl")
    results = read_jsonl(evidence_local(args, "result"))
    smoke_ids = [row["item_id"] for row in smoke]
    result_ids = [row["item_id"] for row in results]
    generation = read_json(args.run_root / "smoke/manifests/generation.json")
    if (
        len(smoke_ids) != generation["n_rows"]
        or len(set(smoke_ids)) != len(smoke_ids)
        or len(set(result_ids)) != len(result_ids)
        or set(result_ids) != set(smoke_ids)
        or any(not isinstance(row.get("verdict"), bool) for row in results)
    ):
        raise RuntimeError("Codex smoke result rows/coverage/parsed verdicts invalid")
    ordered_sha = hashlib.sha256(json.dumps(smoke_ids, sort_keys=True).encode()).hexdigest()
    request = read_json(evidence_local(args, "request"))
    parsed = read_json(evidence_local(args, "parse"))
    if (
        request.get("schema_version") != 1
        or request.get("kind") != "issue952_codex_smoke_request"
        or request.get("identity") != gate["identity"]
        or request.get("n_requests") != len(smoke_ids)
        or request.get("ordered_item_ids_sha256") != ordered_sha
        or parsed.get("schema_version") != 1
        or parsed.get("kind") != "issue952_codex_smoke_parse"
        or parsed.get("identity") != gate["identity"]
        or parsed.get("technical") != gate["technical"]
        or parsed.get("coverage")
        != {
            "n_smoke_rows": len(smoke_ids),
            "n_parsed_rows": len(result_ids),
            "ordered_item_ids_sha256": ordered_sha,
        }
    ):
        raise RuntimeError("Codex smoke request/parse identity or coverage mismatch")
    validate_packet_census(args, gate, request, parsed, smoke, results)


def validate_gate(args: argparse.Namespace) -> dict:
    """Require complete request/result/parse/coverage evidence; scores are advisory."""
    gate = read_json(args.run_root / "dispatch_state/codex_smoke_gate.json")
    if (
        gate.get("schema_version") != 1
        or gate.get("kind") != "issue952_codex_smoke_gate"
        or gate.get("identity") != gate_identity(args)
    ):
        raise RuntimeError("stale or malformed Codex smoke gate identity/schema")
    technical = gate.get("technical", {})
    if gate.get("passed") is not True or any(
        technical.get(key) is not True
        for key in ("request_created", "result_received", "parse_complete", "coverage_complete")
    ):
        raise DesignedHalt("Codex smoke technical request/result/parse/coverage incomplete")
    for key in ("request_sha256", "result_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", gate.get(key, "")) is None:
            raise RuntimeError(f"missing/malformed Codex smoke evidence hash: {key}")
    validate_gate_evidence(args, gate)
    receipt = read_json(args.run_root / "dispatch_state/smoke_verified.json")
    if (
        receipt["hf_prefix"] != f"{HF_PREFIX}/attempt{args.attempt}/smoke"
        or receipt["code_sha"] != args.expected_code_sha
        or receipt["input_revision"] != args.input_revision
        or receipt["attempt"] != args.attempt
    ):
        raise RuntimeError("Codex smoke receipt has wrong attempt namespace")
    for relative in (
        "manifests/generation.json",
        "manifests/capture.json",
        "manifests/smoke_timing.json",
        "raw_completions/rollouts.jsonl",
    ):
        if digest(args.run_root / "smoke" / relative) != receipt["sha256"][relative]:
            raise RuntimeError(f"Codex smoke receipt content mismatch: {relative}")
    seconds = receipt["smoke_wall_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise RuntimeError("invalid smoke GPU accounting")
    validate_smoke(args.run_root, args.expected_code_sha)
    print("[codex_smoke] technical gate complete; numerical agreement is advisory", flush=True)
    return gate


def hydrate_smoke(args: argparse.Namespace) -> None:
    """Hydrate a pinned gate/receipt and their immutable GPU evidence on a new pod."""
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.orchestrate import hub

    if not args.smoke_gate_revision:
        raise DesignedHalt("production requires --smoke-gate-revision; run Codex smoke off-pod")
    prefix = f"{HF_PREFIX}/attempt{args.attempt}"
    for name in ("codex_smoke_gate.json", "smoke_verified.json"):
        hub.stage_hub_file(
            HF_REPO,
            f"{prefix}/dispatch_state/{name}",
            args.run_root / "dispatch_state" / name,
            revision=args.smoke_gate_revision,
            overwrite=True,
        )
    gate = read_json(args.run_root / "dispatch_state/codex_smoke_gate.json")
    for role in ("request", "result", "parse"):
        relative, sha = evidence_spec(gate, role)
        destination = evidence_local(args, role)
        hub.stage_hub_file(
            HF_REPO,
            f"{prefix}/{relative}",
            destination,
            revision=args.smoke_gate_revision,
            overwrite=True,
        )
        if digest(destination) != sha:
            raise RuntimeError(f"Codex smoke evidence content mismatch: {role}")
    role_paths = {spec["path"] for spec in gate["evidence"].values()}
    for relative, sha in gate["artifact_census"].items():
        if relative in role_paths:
            continue
        destination = artifact_local(args, gate, relative)
        hub.stage_hub_file(
            HF_REPO,
            f"{prefix}/{relative}",
            destination,
            revision=args.smoke_gate_revision,
            overwrite=True,
        )
        if digest(destination) != sha:
            raise RuntimeError(f"Codex smoke artifact census content mismatch: {relative}")
    revision = immutable_sha(gate["identity"]["smoke_upload_revision"])
    for relative in (
        "manifests/generation.json",
        "manifests/capture.json",
        "manifests/smoke_timing.json",
        "raw_completions/rollouts.jsonl",
    ):
        hub.stage_hub_file(
            HF_REPO,
            f"{prefix}/smoke/{relative}",
            args.run_root / "smoke" / relative,
            revision=revision,
            overwrite=True,
        )


def dispatch(args: argparse.Namespace) -> int:
    """Run one GPU stage; production requires the immutable off-pod Codex smoke gate."""
    args.run_root = args.run_root.resolve()
    args.logs = args.logs.resolve()
    if (
        args.logs == args.run_root
        or args.logs in args.run_root.parents
        or args.run_root in args.logs.parents
    ):
        raise RuntimeError("resume state must be outside the drained logs tree")
    ledger = args.run_root / "dispatch_state/state.json"
    identity = {
        "code_sha": args.expected_code_sha,
        "input_revision": args.input_revision,
        "run_root": str(args.run_root),
        "attempt": args.attempt,
        "schema": 2,
    }
    assert_code(args.expected_code_sha)
    if ledger.exists():
        state = read_json(ledger)
        if state["identity"] != identity:
            raise RuntimeError("stale dispatcher state: code/input/root identity changed")
    else:
        if any((args.run_root / name).exists() for name in ("inputs", "smoke", "production")):
            raise RuntimeError("unowned partial outputs without provenance state; use a fresh root")
        state = {"identity": identity, "completed": {}, "started_unix": {}}
        atomic_json(ledger, state)
    if args.stage == "production" and not args.smoke_gate_revision:
        raise DesignedHalt("production requires --smoke-gate-revision; run Codex smoke off-pod")
    if args.stage not in state["started_unix"]:
        state["started_unix"][args.stage] = time.time()
        atomic_json(ledger, state)
    started = state["started_unix"][args.stage]
    deadline = started + WALL_SECONDS
    if time.time() >= deadline:
        raise DesignedHalt("registered 8-hour wall fence exhausted; explicit new run required")
    if "smoke_sha256" in state:
        if digest(args.run_root / "smoke/manifests/smoke_timing.json") != state["smoke_sha256"]:
            raise RuntimeError("stale smoke report differs from the production prerequisite")

    def invoke(cmd: list[str], name: str) -> int:
        """Use a new phase log and the same registered run deadline."""
        print(f"[phase={name}] starting", flush=True)
        return run_process(
            cmd, args.run_root / "dispatch_state/logs" / f"{name}-{time.time_ns()}.log", deadline
        )

    own = [
        sys.executable,
        "-u",
        SELF,
        "--run-root",
        str(args.run_root),
        "--expected-code-sha",
        args.expected_code_sha,
        "--input-revision",
        args.input_revision,
        "--stage",
        args.stage,
        "--attempt",
        str(args.attempt),
    ]
    if args.smoke_gate_revision:
        own.extend(["--smoke-gate-revision", args.smoke_gate_revision])
    if "inputs" in state:
        if census(args.run_root, [args.run_root / p for p in state["inputs"]]) != state["inputs"]:
            raise RuntimeError("stale staged inputs")
    else:
        if invoke([*own, "--operation", "stage"], "stage_inputs"):
            raise RuntimeError("immutable Hub input staging failed")
        state["inputs"] = census(args.run_root, sorted((args.run_root / "inputs").glob("*")))
        atomic_json(ledger, state)
    if args.stage == "production":
        if "smoke_gate_revision" in state:
            if state["smoke_gate_revision"] != args.smoke_gate_revision:
                raise RuntimeError("stale dispatcher Codex smoke gate revision")
            gate = validate_gate(args)
            if (
                digest(args.run_root / "dispatch_state/codex_smoke_gate.json")
                != state["codex_gate_sha256"]
            ):
                raise RuntimeError("stale dispatcher Codex smoke gate bytes")
        else:
            if invoke([*own, "--operation", "hydrate"], "hydrate_codex_smoke"):
                raise RuntimeError("immutable Codex smoke hydration failed")
            gate = validate_gate(args)
            state["smoke_gate_revision"] = args.smoke_gate_revision
            state["codex_gate_sha256"] = digest(
                args.run_root / "dispatch_state/codex_smoke_gate.json"
            )
            state["smoke_sha256"] = gate["identity"]["smoke_report_sha256"]
            atomic_json(ledger, state)
        smoke_seconds = read_json(args.run_root / "dispatch_state/smoke_verified.json")[
            "smoke_wall_seconds"
        ]
        deadline -= smoke_seconds
        if time.time() >= deadline:
            raise DesignedHalt("combined smoke/production 8-hour GPU execution fence exhausted")
    for leg in (args.stage,):
        for phase in PHASES:
            assert_code(args.expected_code_sha)
            key = f"{leg}_{phase.replace('-', '_')}"
            if leg == "production":
                if (
                    digest(args.run_root / "dispatch_state/codex_smoke_gate.json")
                    != state["codex_gate_sha256"]
                ):
                    raise RuntimeError("Codex smoke gate changed during production")
                if (
                    digest(args.run_root / "smoke/manifests/smoke_timing.json")
                    != state["smoke_sha256"]
                ):
                    raise RuntimeError("smoke report changed during production")
            if phase in ("gen", "capture") and key in state["completed"]:
                if (
                    census(args.run_root / leg, phase_files(args.run_root / leg, phase))
                    != state["completed"][key]
                ):
                    raise RuntimeError(f"stale completed phase outputs: {key}")
                print(f"[phase={key}] resume hash-verified", flush=True)
                continue
            rc = invoke(command(args, leg, phase), key)
            if leg == "smoke" and phase == "finalize":
                report_path = args.run_root / "smoke/manifests/smoke_timing.json"
                if report_path.exists():
                    validate_smoke(args.run_root, args.expected_code_sha)
            if rc:
                raise RuntimeError(f"{key} exited {rc}; no success sentinel emitted")
            if phase in ("gen", "capture"):
                state["completed"][key] = census(
                    args.run_root / leg, phase_files(args.run_root / leg, phase)
                )
                atomic_json(ledger, state)
        if invoke([*own, "--operation", "verify", "--leg", leg], f"{leg}_verify"):
            raise RuntimeError(f"{leg} terminal upload verification failed")
        if leg == "smoke":
            state["smoke_sha256"] = digest(args.run_root / "smoke/manifests/smoke_timing.json")
            smoke_seconds = time.time() - started
            if smoke_seconds >= WALL_SECONDS:
                raise DesignedHalt("registered 8-hour smoke execution fence exhausted")
            receipt_path = args.run_root / "dispatch_state/smoke_verified.json"
            receipt = read_json(receipt_path)
            receipt["smoke_wall_seconds"] = smoke_seconds
            receipt["input_revision"] = args.input_revision
            receipt["code_sha"] = args.expected_code_sha
            receipt["attempt"] = args.attempt
            atomic_json(receipt_path, receipt)
            state["smoke_wall_seconds"] = smoke_seconds
            atomic_json(ledger, state)
            note = {
                "status": "smoke_complete",
                "scope": "GPU smoke only; persist evidence, stop pod, run Codex smoke off-pod",
                "attempt": args.attempt,
                "gate_identity": gate_identity(args),
                "smoke_receipt": str(receipt_path),
                "smoke_wall_seconds": smoke_seconds,
                "hf_hub_url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{receipt['revision']}/{receipt['hf_prefix']}",
            }
            path = emit_sentinel(
                args.logs, "epm:smoke-result", note, gate="smoke", blocks_pipeline=False
            )
            print(f"[phase=done] smoke_receipt={path}; no production results emitted", flush=True)
            return 0
    final = read_json(args.run_root / "production" / DONE)
    verified = read_json(args.run_root / "dispatch_state/production_verified.json")
    elapsed = smoke_seconds + time.time() - started
    if elapsed >= WALL_SECONDS:
        raise DesignedHalt("combined smoke/production 8-hour GPU execution fence exhausted")
    note = {
        "eval_numbers": {
            "n_prompts": final["generation"]["n_prompts"],
            "n_rollouts": final["generation"]["n_rows"],
            "n_answer_rows": final["capture"]["n_answer_rows"],
        },
        "eval_paths": [str(args.run_root / "production" / DONE)],
        "reproducibility_card": {
            "git_commit": args.expected_code_sha,
            "git_dirty": False,
            "phase": "china-gpu-capture",
            "input_revision": args.input_revision,
            "attempt": args.attempt,
            "smoke_gate_revision": args.smoke_gate_revision,
            "codex_gate_sha256": state["codex_gate_sha256"],
            "smoke_sha256": state["smoke_sha256"],
            "package_versions": final["generation"]["package_versions"],
            "upload_verification": verified,
        },
        "wandb_url": "n/a: generation/capture only; no training",
        "hf_hub_url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{verified['revision']}/{HF_PREFIX}/attempt{args.attempt}",
        "worktree_path": str(ROOT),
        "final_commit_sha": args.expected_code_sha,
        "gpu_hours_used": elapsed / 3600,
        "gpu_hours_budgeted": 8,
        "gpu_hours_accounting": "one GPU per stage; summed stage wall time including staging/uploads; off-pod Codex wait excluded",
        "attempt": args.attempt,
        "codex_smoke_gate": gate,
        "smoke_timing": read_json(args.run_root / "smoke/manifests/smoke_timing.json"),
        "plan_deviations": [],
        "scope": "GPU artifacts only; Codex judging/analysis remain pending",
    }
    print("[verify] push-verify: no git-destined outputs declared this round", flush=True)
    path = emit_sentinel(args.logs, "epm:results", note)
    print(f"[phase=done] results_sentinel={path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Expose pinned-input launcher and isolated staging/verification worker modes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-root", type=Path, default=Path("/workspace/issue952_china_definitive")
    )
    parser.add_argument("--logs", type=Path, default=Path("/workspace/logs"))
    parser.add_argument("--expected-code-sha", type=immutable_sha, required=True)
    parser.add_argument("--input-revision", type=immutable_sha, required=True)
    parser.add_argument("--stage", choices=("smoke", "production"), required=True)
    parser.add_argument("--attempt", type=positive_attempt, default=1)
    parser.add_argument("--smoke-gate-revision", type=immutable_sha)
    parser.add_argument(
        "--operation", choices=("dispatch", "stage", "hydrate", "verify"), default="dispatch"
    )
    parser.add_argument("--leg", choices=("smoke", "production"), default="production")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Hold the task lock, atomically publish this worker PID, and report designed halts."""
    args = build_parser().parse_args()
    if args.operation == "dispatch":
        args.run_root = args.run_root.resolve() / f"attempt{args.attempt}"
    if args.dry_run:
        for phase in PHASES:
            print(json.dumps(command(args, args.stage, phase)))
        return 0
    if args.operation == "stage":
        stage_inputs(args.run_root, args.input_revision)
        return 0
    if args.operation == "verify":
        verify_terminal(args.run_root, args.leg, args.attempt)
        return 0
    if args.operation == "hydrate":
        hydrate_smoke(args)
        return 0
    args.logs.mkdir(parents=True, exist_ok=True)
    with (args.logs / "issue-952-dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pid = args.logs / "issue-952.pid"
        temporary = pid.with_suffix(".pid.tmp")
        temporary.write_text(f"{os.getpid()}\n")
        os.replace(temporary, pid)
        try:
            return dispatch(args)
        except (DesignedHalt, TimeoutError) as error:
            note = {
                "reason": str(error),
                "exit_code": HALT_RC,
                "run_root": str(args.run_root),
                "attempt": args.attempt,
                "stage": args.stage,
            }
            atomic_json(args.run_root / "dispatch_state/gate_report.json", note)
            emit_sentinel(args.logs, "epm:progress", note, gate="smoke-technical-or-wall-fence")
            print(f"[phase=halted] {error} rc={HALT_RC}", flush=True)
            return HALT_RC


if __name__ == "__main__":
    raise SystemExit(main())
