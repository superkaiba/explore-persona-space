"""Stage exact completed K5 captures and supervise the fixed CPU fitting command."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from contextlib import ExitStack, contextmanager, suppress  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: E402


def now() -> str:
    """Return a fresh UTC timestamp."""
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    """Hash an immutable artifact without reading it all into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Persist one complete observation atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_files(receipts: list[dict]) -> list[dict]:
    """Validate immutable archive receipts and select only capture consumer inputs."""
    expected_repos = {
        ("superkaiba1/explore-persona-space-data", "dataset"),
        ("superkaiba1/explore-persona-space-overflow", "model"),
    }
    if {(r["repo"], r["repo_type"]) for r in receipts} != expected_repos:
        raise ValueError("require one public text and one private model-tensor receipt")
    prefixes = {r["prefix"] for r in receipts}
    if prefixes != {"issue825_turn_k5_pilot_20260914/gpu/production"}:
        raise ValueError("only the completed full K5 pilot archive is admissible")
    chosen = []
    for receipt in receipts:
        if receipt["status"] != "verified" or len(receipt["revision"]) != 40:
            raise ValueError("archive receipt is not verified and commit-pinned")
        if receipt["count"] != len(receipt["files"]):
            raise ValueError("archive receipt count mismatch")
        if receipt["bytes"] != sum(row["size"] for row in receipt["files"].values()):
            raise ValueError("archive receipt byte count mismatch")
        for remote, row in receipt["files"].items():
            local = Path(row["local"])
            if (
                local.is_absolute()
                or ".." in local.parts
                or remote != f"{receipt['prefix']}/{local.as_posix()}"
            ):
                raise ValueError("archive path identity mismatch")
            if local.parts[0] == "capture" or local.as_posix() in {
                "gen/instruct/summary.json",
                "gen/pretrained/summary.json",
            }:
                chosen.append(
                    row
                    | {
                        "remote": remote,
                        "repo": receipt["repo"],
                        "repo_type": receipt["repo_type"],
                        "revision": receipt["revision"],
                    }
                )
    names = [row["local"] for row in chosen]
    if not names or len(set(names)) != len(names):
        raise ValueError("capture consumer file set is empty or duplicated")
    for model in ("instruct", "pretrained"):
        if f"capture/{model}/summary.json" not in names or f"gen/{model}/summary.json" not in names:
            raise ValueError("model capture/generation summary missing")
    return sorted(chosen, key=lambda row: row["local"])


def validate_production(out: Path, provenance: dict, expected_conversations: int) -> None:
    """Bind both archived model banks to the same full selected panel and generation chunks."""
    panels = set()
    for model, row in provenance.items():
        capture = row["capture"]
        config = capture["generation_config"]
        if (
            config["n_conversations"] != expected_conversations
            or config["limit_conversations"] != 0
        ):
            raise ValueError("generation bank is not the complete production panel")
        panels.add((config["panel_sha256"], config["selected_ids_sha256"]))
        gen = json.loads((out / "gen" / model / "summary.json").read_text())
        cap_config = json.loads((out / "capture" / model / "config.json").read_text())
        if (
            gen["status"] != "complete"
            or gen["fingerprint"] != capture["generation_fingerprint"]
            or gen["chunks"] != cap_config["generation_chunks"]
        ):
            raise ValueError("generation summary does not match captured draw bank")
        if (
            gen["counts"]["prompts"] != expected_conversations * 2
            or gen["counts"]["draws"] != expected_conversations * 10
        ):
            raise ValueError("generation summary coverage mismatch")
    if len(panels) != 1:
        raise ValueError("models used different planned conversation panels")


def stage(
    text_receipt: Path, tensor_receipt: Path, out: Path, *, expected_conversations: int = 1000
) -> dict:
    """Persist a staging failure before propagating the original exception."""
    began = time.monotonic()
    try:
        return _stage(
            text_receipt, tensor_receipt, out, expected_conversations=expected_conversations
        )
    except Exception as error:
        status = out / "stage_verification.json"
        state = json.loads(status.read_text()) if status.exists() else {"pid": os.getpid()}
        state.update(
            status="failed",
            failed_at=now(),
            elapsed_seconds=time.monotonic() - began,
            error=f"{type(error).__name__}: {error}",
        )
        write_json(status, state)
        raise


def _stage(
    text_receipt: Path, tensor_receipt: Path, out: Path, *, expected_conversations: int = 1000
) -> dict:
    """Stage exact receipt-named files, verify contents, and open the actual fit loader."""
    receipts = [json.loads(path.read_text()) for path in (text_receipt, tensor_receipt)]
    files = selected_files(receipts)
    out.mkdir(parents=True, exist_ok=True)
    size = sum(row["size"] for row in files)
    free = shutil.disk_usage(out).free
    if free < 1.5 * (size + 3 * 1024**3):
        raise RuntimeError("insufficient staging plus projected fit-store disk headroom")
    started = time.monotonic()
    state = {
        "status": "staging",
        "started_at": now(),
        "pid": os.getpid(),
        "expected_count": len(files),
        "expected_bytes": size,
        "verified": [],
        "receipt_sha256": {str(path): digest(path) for path in (text_receipt, tensor_receipt)},
    }
    print(f"[stage] start {len(files)} files {size} bytes", flush=True)
    write_json(out / "stage_verification.json", state)
    for index, row in enumerate(files):
        target = out / row["local"]
        path = stage_hub_file(
            row["repo"],
            row["remote"],
            target,
            repo_type=row["repo_type"],
            revision=row["revision"],
            size_bytes=row["size"],
        )
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise RuntimeError(f"staged consumer content mismatch: {path}")
        state["verified"].append(row)
        state.update(observed_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(out / "stage_verification.json", state)
        print(
            f"[stage] unit {index + 1}/{len(files)} {row['local']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    import issue825_turn_k5_fit as fit

    captures, provenance = {}, {}
    for model in fit.MODELS:
        captures[model], provenance[model] = fit.load_capture(
            out / "capture" / model, expected_model=model
        )
    validate_production(out, provenance, expected_conversations)
    _, coverage = fit.matched_panels(captures)
    state.update(status="verified", completed_at=now(), coverage=coverage, provenance=provenance)
    write_json(out / "stage_verification.json", state)
    return state


def process_memory(pid: int) -> dict:
    """Read live Linux RSS/high-water state, handling the exit-between-probes race."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return {"process_exited": True}
    values = {}
    for line in lines:
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _ = line.split()
            values[key.rstrip(":")] = int(value) * 1024
        elif line.startswith(("State:", "Threads:")):
            key, value = line.split(":", 1)
            values[key] = value.strip()
    return values


@contextmanager
def execution_locks(analysis: Path, store: Path):
    """Prevent concurrent writers sharing either analysis outputs or numerical store."""
    with ExitStack() as stack:
        for root in sorted({analysis.resolve(), store.resolve()}):
            root.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context((root.parent / f".{root.name}.k5.lock").open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def stop_fit(child: subprocess.Popen) -> None:
    """Reap only the numerical child/process group launched by this supervisor."""
    if child.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


@contextmanager
def owned_worker(command: list[str], log, state: dict, status_path: Path):
    """Persist supervisor failures and reap the numerical worker before propagating them."""
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        yield child
    except BaseException as error:
        stop_fit(child)
        state.update(
            status="failed",
            finished_at=now(),
            error=f"{type(error).__name__}: {error}",
            child_pid=child.pid,
            child_reaped=True,
            exit_code=child.returncode,
        )
        write_json(status_path, state)
        raise
    finally:
        stop_fit(child)


def verify_fit_completion(analysis: Path, store: Path, started_epoch: float) -> dict:
    """Require a fresh complete result and all 38 intact numerical artifacts."""
    result_path = analysis / "results.json"
    if result_path.stat().st_mtime < started_epoch:
        raise RuntimeError("CPU completion result predates this process launch")
    result = json.loads(result_path.read_text())
    artifacts = result["artifacts"]
    if result["status"] != "complete" or len(artifacts) != 38:
        raise RuntimeError("CPU result is incomplete or missing planned numerical artifacts")
    seen = set()
    for row in artifacts:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(store.resolve()) or path in seen or path.suffix != ".npz":
            raise RuntimeError("CPU artifact is duplicated or outside its numerical store")
        seen.add(path)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise RuntimeError(f"CPU numerical artifact hash mismatch: {path}")
    return {
        "results_sha256": digest(result_path),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(row["bytes"] for row in artifacts),
        "verified_at": now(),
    }


def supervise(command: list[str], out: Path, analysis: Path, store: Path, **kwargs) -> int:
    """Acquire both writer locks before monitoring the fixed numerical child."""
    with execution_locks(analysis, store):
        return _supervise(command, out, analysis, store, **kwargs)


def _supervise(
    command: list[str],
    out: Path,
    analysis: Path,
    store: Path,
    *,
    interval: float = 2,
    max_projected_seconds: float = 1800,
    max_rss_bytes: int = 16 * 1024**3,
) -> int:
    """Monitor a fresh fit PID and enforce venue gates after the first full fold checkpoint."""
    out.mkdir(parents=True, exist_ok=False)
    started, peak, started_epoch = time.monotonic(), 0, time.time()
    status_path, log_path = out / "status.json", out / "fit.log"
    state = {
        "status": "starting",
        "supervisor_pid": os.getpid(),
        "started_at": now(),
        "command": command,
        "log": str(log_path),
        "pilot_gate": None,
    }
    write_json(status_path, state)
    if (analysis / "folds").exists() or (analysis / "results.json").exists():
        raise RuntimeError("CPU pilot monitor requires fresh analysis outputs")
    with log_path.open("w") as log, owned_worker(command, log, state, status_path) as child:
        state.update(status="running", child_pid=child.pid)
        while True:
            rc = child.poll()
            memory = process_memory(child.pid)
            peak = max(peak, memory.get("VmHWM", 0), memory.get("VmRSS", 0))
            checkpoints = sorted((analysis / "folds").glob("*.json"))
            if checkpoints and state["pilot_gate"] is None:
                first = json.loads(checkpoints[0].read_text())
                projected = first["elapsed_seconds"] * 12
                passed = projected <= max_projected_seconds and peak < max_rss_bytes
                state["pilot_gate"] = {
                    "status": "passed" if passed else "venue_review_required",
                    "observed_at": now(),
                    "fold_receipt": str(checkpoints[0]),
                    "fold_seconds": first["elapsed_seconds"],
                    "projected_fit_seconds": projected,
                    "sampled_peak_rss_bytes": peak,
                    "max_projected_seconds": max_projected_seconds,
                    "max_rss_bytes": max_rss_bytes,
                    "basis": "first complete model/outer-fold (two turns, both K fits) times 12",
                }
                write_json(out / "pilot_gate.json", state["pilot_gate"])
                print(json.dumps(state["pilot_gate"]), flush=True)
                if not passed and rc is None:
                    # The approved CPU venue gate stops only this numerical worker,
                    # after its first durable fold checkpoint; no backend operation.
                    stop_fit(child)
                    rc = child.returncode
            with log_path.open("rb") as source:
                source.seek(max(0, log_path.stat().st_size - 8000))
                tail = source.read().decode("utf-8", errors="replace")
            state.update(
                observed_at=now(),
                heartbeat_epoch=time.time(),
                elapsed_seconds=time.monotonic() - started,
                memory=memory,
                sampled_peak_rss_bytes=peak,
                completed_fold_checkpoints=len(checkpoints),
                log_bytes=log_path.stat().st_size,
                log_tail=tail,
            )
            if rc is not None:
                if rc == 0:
                    state["completion_verification"] = verify_fit_completion(
                        analysis, store, started_epoch
                    )
                state.update(
                    status="complete" if rc == 0 else "failed", exit_code=rc, finished_at=now()
                )
                if state["pilot_gate"] and state["pilot_gate"]["status"] != "passed":
                    state["status"] = "venue_review_required"
                write_json(status_path, state)
                return 7 if state["status"] == "venue_review_required" else rc
            write_json(status_path, state)
            time.sleep(interval)


def main() -> int:
    """Expose pinned consumer staging and a monitored frozen-driver execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--text-receipt", type=Path, required=True)
    staging.add_argument("--tensor-receipt", type=Path, required=True)
    staging.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--staged", type=Path, required=True)
    run.add_argument("--analysis", type=Path, required=True)
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--monitor", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.text_receipt, args.tensor_receipt, args.out)
        return 0
    verification = json.loads((args.staged / "stage_verification.json").read_text())
    if verification["status"] != "verified":
        raise RuntimeError("capture staging/consumer probe has not completed")
    driver = Path(__file__).with_name("issue825_turn_k5_fit.py")
    command = [
        sys.executable,
        str(driver),
        "--instruct-root",
        str(args.staged / "capture/instruct"),
        "--pretrained-root",
        str(args.staged / "capture/pretrained"),
        "--out-dir",
        str(args.analysis),
        "--store-dir",
        str(args.store),
    ]

    def stop_requested(signum, _frame):
        """Unwind through the owned CPU worker so cancellation cannot orphan it."""
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    return supervise(command, args.monitor, args.analysis, args.store)


if __name__ == "__main__":
    raise SystemExit(main())
