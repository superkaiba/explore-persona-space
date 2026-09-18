"""Wait for a RunPod quote, dispatch one continuation, and supervise the handoff.

The checker never provisions hardware. Its operator-authored runbook is the
continuation's authority; API replies only determine whether to wake it.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


class Runtime:
    """External boundaries shared by the checker and its integration tests."""

    def __init__(self, config):
        """Load the existing environment, API transport and pinned watchdog helper."""
        sys.path.insert(0, config["workdir"])
        from explore_persona_space.orchestrate.env import load_dotenv

        load_dotenv(config["dotenv"])
        from scripts.runpod_api import graphql

        spec = importlib.util.spec_from_file_location(
            "capacity_watchdog", config["watchdog_helper"]
        )
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.transport = graphql
        self.system = self.helper.Runtime()

    def query(self):
        """Ask for the approved eight-H200 shape, not a catalog-only match."""
        return self.transport(
            """query {
          gpuTypes(input:{id:"NVIDIA H200"}) {
            id memoryInGb
            secure:lowestPrice(input:{gpuCount:8,secureCloud:true}) {
              stockStatus uninterruptablePrice minMemory minDisk minDownload
            }
            community:lowestPrice(input:{gpuCount:8,secureCloud:false}) {
              stockStatus uninterruptablePrice minMemory minDisk minDownload
            }
          }
        }""",
            timeout=45,
        )

    def owned_pods(self, config):
        """Prove exact-name pod absence before accepting a capacity-loss retry."""
        data = self.transport("query { myself { pods { id name desiredStatus } } }", timeout=45)
        pods = data["myself"]["pods"]
        if not isinstance(pods, list):
            raise ValueError("invalid live pod response")
        return [p for p in pods if p["name"] == config["pod_name"]]

    def unit(self, name):
        """Read actual systemd worker liveness."""
        return self.system.unit(name)

    def start(self, config):
        """Start the fixed unit; repeated systemctl start cannot duplicate a live unit."""
        subprocess.run(
            ["systemctl", "--user", "start", "--no-block", config["worker_unit"]],
            check=True,
            timeout=20,
            capture_output=True,
        )

    def notify(self, config, state, path, now):
        """Retry queued messages until the established destination acknowledges them."""
        self.helper.deliver(config, state, path, self.system, now)


def write(path, data):
    """Atomically preserve a complete observation or handoff record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(data, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def eligible(data):
    """A missing/malformed quote is never treated as available capacity."""
    models = data["gpuTypes"]
    if len(models) != 1 or models[0]["id"] != "NVIDIA H200":
        raise ValueError("unexpected GPU query response")
    model = models[0]
    if not isinstance(model["memoryInGb"], (int, float)) or model["memoryInGb"] < 130:
        raise ValueError("H200 HBM requirement not met")
    offers = []
    for cloud in ("secure", "community"):
        q = model[cloud]
        if q is None:
            continue
        price = q["uninterruptablePrice"]
        if q["stockStatus"] in {"Low", "Medium", "High"} and isinstance(price, (int, float)):
            if math.isfinite(price) and price > 0:
                offers.append({"cloud": cloud, **q})
    return offers


def enqueue(state, key, message, now):
    """Deduplicate meaningful alerts across restarts and delivery failures."""
    if not any(x["key"] == key for x in state["notifications"]):
        state["notifications"].append(
            dict(key=key, message=message, created_at=now, delivered_at=None)
        )


def tick(config, runtime, now=None):
    """One bounded check; record dispatch intent before starting any worker."""
    now = time.time() if now is None else now
    state_path = Path(config["capacity_state"])
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "notifications": [],
            "attempts": [],
            "worker": None,
        }
    )
    observation = dict(
        source_sha=config["source_sha"],
        checked_at=now,
        status="waiting_for_capacity",
        backend_observation={"status": "pending"},
    )
    worker = state["worker"]
    if worker:
        observation["status"] = "continuation_requested"
        directory = Path(worker["directory"])
        unit = runtime.unit(config["worker_unit"])
        live = unit["ActiveState"] in {"active", "activating", "deactivating"}
        outcome_path = directory / "outcome.json"
        if live:
            log = directory / "worker.log"
            age = now - (log.stat().st_mtime if log.exists() else worker["requested_at"])
            observation.update(
                status="continuation_running",
                backend_observation={
                    "status": "running",
                    "pid_alive": True,
                    "last_log_mtime_sec_ago": max(0, age),
                    "worker_unit": config["worker_unit"],
                    "attempt": worker["number"],
                },
            )
        elif outcome_path.exists():
            outcome = json.loads(outcome_path.read_text())
            if (
                outcome["attempt"] != worker["number"]
                or outcome["source_sha"] != config["source_sha"]
            ):
                raise ValueError("stale continuation outcome")
            exit_record = json.loads((directory / "exit.json").read_text())
            if exit_record["returncode"] != 0 or exit_record["ended_at"] < worker["requested_at"]:
                raise ValueError("continuation exited unsuccessfully")
            if outcome["status"] == "capacity_lost":
                if runtime.owned_pods(config) or Path(config["handle_file"]).exists():
                    raise ValueError("cannot reset capacity wait with a pod or handle present")
                ledger = Path(config["allocation_ledger"]).read_text()
                if ledger != worker["ledger_before"]:
                    raise ValueError("allocation ledger changed; bounded recovery required")
                state["worker"] = None
                state["retry_after"] = now + config["interval_seconds"]
                observation["status"] = "capacity_lost_waiting_again"
                enqueue(
                    state,
                    f"lost-{worker['number']}",
                    "Task 2673: the quoted H200 capacity disappeared before allocation. "
                    "No GPU was allocated; the five-minute capacity checks are continuing.",
                    now,
                )
            elif outcome["status"] == "complete":
                terminal_bytes = Path(config["terminal_file"]).read_bytes()
                terminal = json.loads(terminal_bytes)
                if (
                    terminal["source_sha"] != config["source_sha"]
                    or terminal["model_key"] != "deepseek"
                    or terminal.get("continuation_attempt") != worker["number"]
                    or outcome.get("terminal_sha256") != hashlib.sha256(terminal_bytes).hexdigest()
                    or not worker["requested_at"]
                    <= terminal.get("checked_at", 0)
                    <= exit_record["ended_at"]
                    <= now
                    or terminal.get("handle_file") != config["handle_file"]
                    or not re.fullmatch(r"[0-9a-f]{40}", terminal["verified_revision"])
                    or not re.fullmatch(r"[0-9a-f]{40}", outcome["comparison_git_revision"])
                    or not outcome["report_url"].startswith(
                        "https://github.com/superkaiba/explore-persona-space/blob/"
                        + outcome["comparison_git_revision"]
                        + "/"
                    )
                    or any(
                        terminal.get(key) is not True
                        for key in (
                            "all_remote_names_sizes_hashes_pass",
                            "git_json_equals_immutable_hf",
                            "smoke_passed",
                        )
                    )
                    or terminal.get("row_count") != 1920
                    or terminal.get("chunks") != 240
                    or terminal.get("selected_layers") != [15, 30, 45, 60]
                    or terminal.get("file_count", 0) < 240
                    or terminal.get("total_bytes", 0) <= 0
                    or any(
                        not re.fullmatch(r"[0-9a-f]{64}", terminal.get(key, ""))
                        for key in ("capture_fingerprint", "analysis_fingerprint")
                    )
                    or runtime.owned_pods(config)
                ):
                    raise ValueError("completion/teardown evidence is incomplete")
                observation.update(status="completion_notification_pending", results=terminal)
                enqueue(
                    state,
                    "completed",
                    "Task 2673: DeepSeek extraction and Qwen comparison "
                    f"completed and verified. {outcome['report_url']}",
                    now,
                )
            else:
                raise ValueError("continuation requires intervention: " + outcome["status"])
        elif now - worker["requested_at"] > 90:
            observation.update(
                status="backend_gate",
                backend_observation={
                    "status": "gate",
                    "reason": "continuation ended without a verified outcome",
                },
            )
    elif now >= state.get("retry_after", 0):
        data = runtime.query()
        offers = eligible(data)
        observation["capacity"] = data
        if offers:
            recent = [x for x in state["attempts"] if now - x < 86400]
            if len(recent) >= config["max_attempts_per_day"]:
                observation["status"] = "capacity_available_attempt_limit"
                enqueue(
                    state,
                    f"limit-{int(now // 86400)}",
                    "Task 2673: H200 capacity is visible, but the bounded continuation "
                    "attempt limit was reached. Attention is needed.",
                    now,
                )
            else:
                number = len(state["attempts"]) + 1
                directory = state_path.parent / f"continuation-{number}"
                directory.mkdir(mode=0o700, exist_ok=True)
                if any(directory.iterdir()):
                    raise ValueError("unclaimed continuation directory contains prior work")
                state["attempts"].append(now)
                state["worker"] = dict(
                    number=number,
                    directory=str(directory),
                    requested_at=now,
                    ledger_before=Path(config["allocation_ledger"]).read_text(),
                )
                enqueue(
                    state,
                    f"capacity-{number}",
                    "Task 2673: eight-H200 capacity is visible. "
                    "A continuation has been requested to recheck it and resume the approved "
                    "DeepSeek extraction within the existing compute allowance.",
                    now,
                )
                write(state_path, state)
                runtime.start(config)
                observation["status"] = "continuation_requested"
    write(config["observation"], observation)
    write(state_path, state)
    runtime.notify(config, state, state_path, now)
    if observation["status"] == "completion_notification_pending" and all(
        item["delivered_at"] is not None for item in state["notifications"]
    ):
        observation["status"] = "complete"
        observation["completion_notification_acknowledged_at"] = now
        write(config["observation"], observation)
    return observation


def worker(config, runtime):
    """Run one bounded Codex continuation with durable log, result and exit receipts."""
    state = json.loads(Path(config["capacity_state"]).read_text())
    attempt = state["worker"]
    directory = Path(attempt["directory"])
    prompt = Path(config["continuation_prompt"]).read_text() + (
        f"\nCurrent attempt: {attempt['number']}. Write final machine-readable outcome to "
        f"{directory / 'outcome.json'}. Source SHA {config['source_sha']}.\n"
    )
    write(directory / "started.json", {"started_at": time.time(), "attempt": attempt["number"]})
    with (directory / "worker.log").open("a") as log:
        proc = subprocess.run(
            runtime.helper.codex_command(config, directory / "result.txt"),
            input=prompt,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config["continuation_seconds"],
        )
    write(directory / "exit.json", {"returncode": proc.returncode, "ended_at": time.time()})
    return proc.returncode


def main():
    """Run the persistent checker or its single continuation unit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["monitor", "worker", "once"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = Runtime(config)
    if args.action == "worker":
        return worker(config, runtime)
    with Path(config["capacity_state"]).with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            observed = tick(config, runtime)
            state = json.loads(Path(config["capacity_state"]).read_text())
            delivered = all(n["delivered_at"] is not None for n in state["notifications"])
            if (observed["status"] == "complete" and delivered) or args.action == "once":
                return 0
            time.sleep(
                30 if observed["status"].startswith("continuation_") else config["interval_seconds"]
            )


if __name__ == "__main__":
    sys.exit(main())
