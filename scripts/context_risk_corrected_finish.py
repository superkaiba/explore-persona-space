"""Finish approved task 2670 after its pinned generation supervisor exits.

No model calls. Run as a module from the reviewed recovery worktree.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

WORK = Path(__file__).resolve().parents[1]
REPO = Path("/home/thomasjiralerspong/explore-persona-space")
DATA = REPO / "eval_results/context_risk"
ROOT = DATA / "impossible_livecodebench_v20_corrected"
LAUNCH = ROOT / "full_20260907T071531Z.launch.json"
POD = "pod-2670-corrected"
POD_ID = "upvifie9u2zfa1"
POD_LOGS = "/workspace/logs/issue2670-context-risk-v20"
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "context_risk/issue2670_corrected_v20"
STAGING = DATA / "issue2670_archive_staging"
MANIFEST = DATA / "data/impossible_livecodebench_promptB/public_pilot_manifest.jsonl"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(command: list[str], *, timeout: int = 300, cwd: Path = WORK) -> None:
    print(json.dumps({"command": command, "cwd": str(cwd), "timeout": timeout}), flush=True)
    subprocess.run(command, cwd=cwd, check=True, timeout=timeout)


def marker(kind: str, note: str) -> None:
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/task.py",
            "post-marker",
            "2670",
            kind,
            "--by",
            "codex",
            "--note",
            note,
        ],
        cwd=REPO,
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_pod_present(*, allow_absent: bool = False) -> bool:
    from scripts.runpod_api import list_team_pods

    matches = [pod for pod in list_team_pods() if pod.name == POD or pod.pod_id == POD_ID]
    if not matches and allow_absent:
        return False
    if len(matches) != 1 or matches[0].name != POD or matches[0].pod_id != POD_ID:
        raise ValueError("Live pod identity differs from this experiment launch")
    return True


def upload(stage: Path, phase: str) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from explore_persona_space.orchestrate import hub
    from scripts.issue2054_phase_a import _shard_large_jsonl_for_upload

    prefix = f"{HF_PREFIX}/{phase}"
    original = sorted(p for p in stage.rglob("*") if p.is_file())
    retained = _shard_large_jsonl_for_upload(original)
    if any(p.suffix == ".jsonl" and p.stat().st_size > 9_500_000 for p in retained):
        raise ValueError("A single JSONL record exceeds the upload shard limit")
    files = {f"{prefix}/{p.relative_to(stage)}": p for p in retained}
    if not files:
        raise ValueError("Refusing empty archive")
    local = {
        name: {"size": path.stat().st_size, "sha256": sha(path)} for name, path in files.items()
    }
    api = HfApi()
    receipt_path = ROOT / f"{phase}_upload_receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior["files"] != local or prior["prefix"] != prefix or prior["repo_id"] != HF_REPO:
            raise ValueError("Staged content differs from the existing upload receipt")
        revision = prior["revision"]
    else:
        ignored = [str(p.relative_to(stage)) for p in original if p not in retained]
        destination = hub._upload(
            stage, HF_REPO, "dataset", prefix, ignore_patterns=ignored, raise_on_error=True
        )
        if destination != f"{HF_REPO}/{prefix}":
            raise RuntimeError(f"Unexpected archive destination: {destination}")
        revision = hub.retry_transient(
            lambda: api.repo_info(HF_REPO, repo_type="dataset").sha,
            what="corrected archive revision",
            budget_s=600,
        )
    entries = hub.retry_transient(
        lambda: list(
            api.list_repo_tree(
                HF_REPO, path_in_repo=prefix, recursive=True, repo_type="dataset", revision=revision
            )
        ),
        what="corrected archive exact tree",
        budget_s=600,
    )
    remote = {entry.path: entry for entry in entries if isinstance(entry, RepoFile)}
    if set(remote) != set(files):
        raise ValueError("Remote archive filenames do not equal staged filenames")
    for name, path in files.items():
        entry = remote[name]
        if entry.size != local[name]["size"] or sha(path) != local[name]["sha256"]:
            raise ValueError(f"Archive size or local content changed: {name}")
        if entry.lfs:
            if entry.lfs.sha256 != local[name]["sha256"]:
                raise ValueError(f"Remote LFS content mismatch: {name}")
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != entry.blob_id:
                raise ValueError(f"Remote Git blob content mismatch: {name}")
    receipt = {
        "passed": True,
        "repo_id": HF_REPO,
        "revision": revision,
        "prefix": prefix,
        "files": local,
        "verified_files": len(files),
        "verification": "Exact filename set, byte size, and remote LFS SHA256 or Git blob hash.",
        "url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{revision}/{prefix}",
    }
    write_json(ROOT / f"{phase}_upload_receipt.json", receipt)
    return receipt


def preserve_finalized_trace(source: Path, destination: Path) -> dict:
    """Inspect 0.3.261 compresses and removes its live trace on normal exit."""
    with gzip.open(source, "rb") as stream:
        content = stream.read()
    records = [json.loads(line) for line in content.splitlines()]
    if not records or any(not {"timestamp", "level", "message"} <= set(row) for row in records):
        raise ValueError("Finalized Inspect trace is not a nonempty JSONL event log")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return {
        "source": str(source),
        "source_sha256": sha(source),
        "destination": str(destination),
        "destination_sha256": sha(destination),
        "n_records": len(records),
        "connection_retries": [
            row for row in records if "[APIConnectionError]" in str(row["message"])
        ],
        "verification": "Gzip CRC checked on complete decompression; every JSONL record parsed. Plain JSONL is archived and line-sharded by the existing upload helper if required.",
    }


def upload_bounded(stage: Path, name: str) -> dict:
    pilot = json.loads((ROOT / "setup/smoke_archive_pilot.json").read_text())
    if not pilot["passed"] or pilot["staged_bytes"] <= 0:
        raise ValueError("Measured archive pilot is required")
    size = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file())
    projected_seconds = size / pilot["staged_bytes"] * pilot["upload_and_verification_seconds"]
    deadline = max(7200, math.ceil(4800 + 2 * projected_seconds + 600))
    write_json(
        ROOT / f"{name}_upload_budget.json",
        {
            "staged_bytes": size,
            "scaled_pilot_seconds": projected_seconds,
            "transfer_dispersion_factor": 2,
            "retry_exposure_seconds": 4800,
            "additional_processing_allowance_seconds": 600,
            "deadline_seconds": deadline,
            "basis": "Actual bytes / measured smoke-archive bytes × measured upload+verification time; two1800s shared-helper retry budgets and two600s metadata retry budgets.",
        },
    )
    run(
        [
            sys.executable,
            "-m",
            "scripts.context_risk_corrected_finish",
            "--upload-stage",
            str(stage),
            "--upload-name",
            name,
        ],
        timeout=deadline,
    )
    return json.loads((ROOT / f"{name}_upload_receipt.json").read_text())


def wait_for_generation() -> dict:
    launch = json.loads(LAUNCH.read_text())
    if launch["pid"] != 3703711 or launch["mode"] != "full":
        raise ValueError("Unexpected full-generation launch identity")
    exit_file = Path(launch["exit_file"])
    deadline = datetime.fromisoformat(launch["launched_utc"]).timestamp() + 104400
    while not exit_file.exists():
        pid = launch["pid"]
        command_path = Path(f"/proc/{pid}/cmdline")
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode()
        except FileNotFoundError:
            if exit_file.exists():
                break
            raise RuntimeError("Generation supervisor disappeared without exit evidence") from None
        if "context_risk_corrected_supervise.sh full" not in command:
            raise RuntimeError("Generation PID identity changed")
        if time.time() > deadline:
            raise TimeoutError("Generation exceeded its supervisor deadline plus one-hour margin")
        write_json(
            ROOT / "completion_status.json",
            {
                "state": "running",
                "phase": "waiting_for_generation",
                "updated_at": datetime.now(UTC).isoformat(),
                "generation_pid": pid,
            },
        )
        time.sleep(30)
    exit_record = json.loads(exit_file.read_text())
    worker_file = Path(launch["pid_file"]).with_suffix(".worker.pid")
    if (
        exit_record["mode"] != "full"
        or exit_record["supervisor_pid"] != launch["pid"]
        or exit_record["worker_pid"] != int(worker_file.read_text())
        or exit_record["exit_code"] != 0
        or exit_record["cleanup"]
        not in {"no_live_members", "terminated_descendants", "killed_descendants"}
        or exit_record["finished_unix"] < datetime.fromisoformat(launch["launched_utc"]).timestamp()
    ):
        raise RuntimeError(f"Generation did not exit successfully: {exit_record}")
    result_path = ROOT / "full/run_result.json"
    if result_path.stat().st_mtime < datetime.fromisoformat(launch["launched_utc"]).timestamp():
        raise ValueError("Stale full-run result")
    return json.loads(result_path.read_text())


def finish() -> None:
    from scripts.context_risk_corrected_audit import audit
    from scripts.context_risk_impossiblebench_inspect import validate_critic_review

    validate_critic_review(
        WORK / "eval_results/context_risk_corrected_v20_validation/critic_review.json"
    )
    approved_sources = {}
    for filename in ("audit_review.json", "finish_review.json"):
        review = json.loads(
            (WORK / "eval_results/context_risk_corrected_v20_validation" / filename).read_text()
        )
        if review["verdict"] != "PASS" or review["source_sha256"] != sha(WORK / review["source"]):
            raise ValueError(f"Completion source is not independently approved: {filename}")
        approved_sources[review["source"]] = review["source_sha256"]
    result = wait_for_generation()
    for filename in ("audit_review.json", "finish_review.json"):
        review = json.loads(
            (WORK / "eval_results/context_risk_corrected_v20_validation" / filename).read_text()
        )
        if (
            review["verdict"] != "PASS"
            or review["source_sha256"] != sha(WORK / review["source"])
            or review["source_sha256"] != approved_sources[review["source"]]
        ):
            raise ValueError(f"Completion source changed while generation was running: {filename}")
    native = audit(ROOT / "full/run_result.json", MANIFEST)
    if native["realized_unique_rollouts"] != 480:
        raise ValueError("Full-run roster is incomplete")
    write_json(
        ROOT / "completion_status.json",
        {"state": "running", "phase": "archiving_raw", "updated_at": datetime.now(UTC).isoformat()},
    )
    if not (ROOT / "raw_snapshot.json").exists():
        trace = preserve_finalized_trace(
            Path("/home/thomasjiralerspong/.local/share/inspect_ai/traces/trace-3704829.log.gz"),
            ROOT / "setup/full_inspect_trace.jsonl",
        )
        write_json(ROOT / "setup/full_inspect_trace_receipt.json", trace)
        own_pod_present()
        # Stop only this task's server supervisor, then collect its completed logs.
        stop = "import json,os,signal,time\nfrom pathlib import Path\np=Path('/workspace/logs/issue2670-context-risk-v20/server_20260907T0705_server_process.pid')\npid=int(p.read_text())\nassert pid==3973\nexit_path=p.with_suffix('.exit.json')\nif not exit_path.exists():\n cmd=Path(f'/proc/{pid}/cmdline')\n assert cmd.exists()\n command=cmd.read_bytes()\n assert b'context_risk_corrected_supervise.sh' in command and b'server' in command\n os.kill(pid,signal.SIGTERM)\ndeadline=time.time()+100\nwhile not exit_path.exists():\n if time.time()>deadline: raise TimeoutError('Server did not record shutdown')\n time.sleep(1)\nrecord=json.loads(exit_path.read_text())\nassert record['mode']=='server' and record['supervisor_pid']==3973\nassert record['cleanup'] in {'no_live_members','terminated_descendants','killed_descendants'}\nassert record['exit_code'] in {0,143}\nprint(json.dumps(record))\n"
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", POD, "uv", "run", "--no-project", "python", "-"],
            input=stop,
            text=True,
            check=True,
            timeout=130,
        )
        census_program = "import hashlib,json\nfrom pathlib import Path\nroot=Path('/workspace/logs/issue2670-context-risk-v20')\nprint(json.dumps({str(p.relative_to(root)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.rglob('*') if p.is_file()}))\n"
        captured = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", POD, "uv", "run", "--no-project", "python", "-"],
            input=census_program,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        census = json.loads(captured.stdout)
        if not census:
            raise ValueError("Pod output census is empty")
        run(
            ["rsync", "-a", "--checksum", f"{POD}:{POD_LOGS}/", str(ROOT / "server") + "/"],
            timeout=300,
        )
        observed = {
            str(p.relative_to(ROOT / "server")): {"size": p.stat().st_size, "sha256": sha(p)}
            for p in (ROOT / "server").rglob("*")
            if p.is_file()
        }
        if observed != census:
            raise ValueError("Copied pod outputs differ from the stopped-server census")
        write_json(ROOT / "setup/pod_output_census.json", census)
        (ROOT / "setup/pod_outroot_listing.txt").write_text(
            "".join(f"{POD_LOGS}/{name}\n" for name in sorted(census))
        )
        stage = STAGING / "raw"
        stage.mkdir(parents=True, exist_ok=True)
        for folder in ("full", "smoke", "setup", "server"):
            shutil.copytree(ROOT / folder, stage / folder, dirs_exist_ok=True)
        for path in ROOT.iterdir():
            if path.is_file() and path.name.startswith(("smoke_", "full_")):
                shutil.copy2(path, stage / path.name)
        source = stage / "source"
        source.mkdir(exist_ok=True)
        for name in (
            "context_risk_impossiblebench.py",
            "context_risk_impossiblebench_inspect.py",
            "context_risk_impossiblebench_harness.py",
            "context_risk_corrected_launch.sh",
            "context_risk_corrected_supervise.sh",
            "context_risk_corrected_audit.py",
            "context_risk_corrected_finish.py",
            "context_risk_analyze.py",
        ):
            shutil.copy2(WORK / "scripts" / name, source / name)
        shutil.copytree(
            WORK / "eval_results/context_risk_corrected_v20_validation",
            stage / "validation",
            dirs_exist_ok=True,
        )
        shutil.copy2(
            WORK / "docs/ideas/context_risk_corrected_rerun.md", stage / "approved_plan.md"
        )
        write_json(
            stage / "input_archive.json",
            {
                "repo": HF_REPO,
                "revision": "97f50b287b704dbf4f353833154fd4c8f4e35d44",
                "prefix": "context_risk/recovery_20260906/inputs_v2",
                "map_sha256": "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d",
            },
        )
        snapshot = {
            str(p.relative_to(stage)): {"size": p.stat().st_size, "sha256": sha(p)}
            for p in stage.rglob("*")
            if p.is_file()
        }
        write_json(ROOT / "raw_snapshot.json", snapshot)
    stage = STAGING / "raw"
    snapshot = json.loads((ROOT / "raw_snapshot.json").read_text())
    for name, expected in snapshot.items():
        path = stage / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["size"]
            or sha(path) != expected["sha256"]
        ):
            raise ValueError(f"Frozen raw snapshot changed: {name}")
    raw = upload_bounded(stage, "raw")
    from scripts.verify_uploads import check_outroot_residue

    sweep = check_outroot_residue(
        2670,
        outroot_listing=str(ROOT / "setup/pod_outroot_listing.txt"),
        hf_prefixes=(f"{HF_PREFIX}/raw/server",),
        data_repos=(HF_REPO,),
    )
    write_json(ROOT / "outroot_sweep.json", sweep)
    if sweep["status"] != "OK":
        raise ValueError(f"Pod output sweep did not pass: {sweep}")
    marker(
        "epm:upload-verification",
        f"Verdict: PASS. pod={POD} owner=codex outroot=swept-clean rows=reconciled. All 480 native sample/epoch pairs reconciled; raw submissions, native logs, scores, source and runtime evidence preserved. Pod output root {POD_LOGS} copied in full and archived. Exact filenames, sizes and remote content hashes verified at {raw['url']}. Receipt: {ROOT / 'raw_upload_receipt.json'}",
    )
    if own_pod_present(allow_absent=True):
        # The managed command enforces current upload and ownership gates itself.
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/pod.py",
                "terminate",
                "--issue",
                "2670",
                "--name-suffix",
                "corrected",
            ],
            cwd=REPO,
            timeout=300,
        )
    if own_pod_present(allow_absent=True):
        raise RuntimeError("Pinned pod still exists after managed termination")
    write_json(
        ROOT / "pod_released.json",
        {"pod_id": POD_ID, "pod_name": POD, "verified_absent_at": datetime.now(UTC).isoformat()},
    )
    write_json(
        ROOT / "completion_status.json",
        {"state": "running", "phase": "analysis", "updated_at": datetime.now(UTC).isoformat()},
    )
    map_path = DATA / "qwen38_map_pilot/map_layer_44.npz"
    if sha(map_path) != "680935a244cc39c29797d66b312b95e3741889a541dcee8bde7c69ac33c5242d":
        raise ValueError("Frozen map content changed")
    analysis = ROOT / "analysis"
    run(
        [
            sys.executable,
            "-m",
            "scripts.context_risk_analyze",
            "--impossible-result",
            str(ROOT / "full/run_result.json"),
            "--impossible-capture-root",
            str(DATA / "qwen38_impossible_contexts_v4_promptB"),
            "--impossible-manifest",
            str(MANIFEST),
            "--map-artifact",
            str(map_path),
            "--output-dir",
            str(analysis),
        ],
        timeout=21600,
    )
    summary = json.loads((analysis / "analysis_result.json").read_text())
    if (
        summary["reward_hacking_feasibility"]["execution_integrity"]["passed"]
        != native["execution_passed"]
    ):
        raise ValueError("Analysis/native execution-integrity disagreement")
    stage_analysis = STAGING / "analysis"
    shutil.copytree(analysis, stage_analysis, dirs_exist_ok=True)
    analyzed = upload_bounded(stage_analysis, "analysis")
    final = {
        "state": "quantitative_complete",
        "phase": "analysis_complete",
        "pod_id": POD_ID,
        "generation_rollouts": 480,
        "technical_errors": result["technical_errors"],
        "prediction_status": summary["reward_hacking_feasibility"]["prediction_status"],
        "raw_url": raw["url"],
        "analysis_url": analyzed["url"],
        "updated_at": datetime.now(UTC).isoformat(),
        "interpretation_review": "Quantitative output complete; qualitative original-success audit remains for final interpretation.",
    }
    write_json(ROOT / "completion_status.json", final)
    marker(
        "epm:progress",
        "verified-by: ran. Corrected generation, native audit, conditional linear analysis and archive verification finished. "
        + json.dumps(final),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-stage", type=Path)
    parser.add_argument("--upload-name", choices=("raw", "analysis"))
    args = parser.parse_args()
    load_dotenv(str(REPO / ".env"))
    os.environ.update(
        UV_NO_SYNC="1",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        NUMEXPR_NUM_THREADS="8",
        MALLOC_ARENA_MAX="2",
        EPM_HF_FILECOUNT_FALLBACK="0",
        EPM_HF_RETRY_BUDGET_S="1800",
    )
    owns_status = False
    try:
        if args.upload_stage:
            if args.upload_name is None:
                raise ValueError("--upload-name required")
            upload(args.upload_stage, args.upload_name)
        else:
            with (ROOT / "completion.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owns_status = True
                finish()
    except BaseException as error:
        if owns_status:
            write_json(
                ROOT / "completion_status.json",
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            marker(
                "epm:failure",
                f"Completion coordinator failed after preserving local evidence. {type(error).__name__}: {error}. Status: {ROOT / 'completion_status.json'}. Do not claim completion or discard partial outputs.",
            )
        raise


if __name__ == "__main__":
    main()
