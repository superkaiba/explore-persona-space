"""Wait for a RunPod quote, dispatch one continuation, and supervise the handoff.

The checker never provisions hardware. Its operator-authored runbook is the
continuation's authority; API replies only determine whether to wake it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import fcntl
from functools import partial
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
from zoneinfo import ZoneInfo


class OutcomeRejected(ValueError):
    """Preserved worker evidence is not acceptable proof of the claimed outcome."""


def finite_number(value):
    """Reject JSON strings, booleans and non-finite timestamps/counts."""
    return type(value) in (int, float) and math.isfinite(value)


def digest(value, length):
    """Check evidence hashes without treating malformed JSON types as code errors."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value)


def next_midnight(config, now):
    """Use local calendar days so daylight-saving changes retain midnight timing."""
    local = datetime.fromtimestamp(now, ZoneInfo(config["daily_dispatch_timezone"]))
    return (
        local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    ).timestamp()


def advance_schedule(config, state, now):
    """Persist one daily dispatch opportunity; monitor/cleanup checks stay frequent."""
    if config.get("daily_dispatch_timezone"):
        state["next_dispatch_at"] = next_midnight(config, now)


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
        self.transport = partial(graphql, personal=True)
        self.system = self.helper.Runtime()
        if config["account_scope"] != "personal":
            raise ValueError("this monitor is authorized for the personal account only")
        self.expected_account_id = config["expected_account_id"]

    def verify_account(self, data):
        """Reject changed identity or team membership instead of trusting an old header."""
        account = data["myself"]
        if account["id"] != self.expected_account_id or account["teams"] != []:
            raise ValueError(
                "personal RunPod account identity/scope changed; inspect before launch"
            )

    def query(self):
        """Verify the personal identity and ask for the approved eight-H200 shape."""
        data = self.transport(
            """query {
          myself { id teams { id } }
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
        self.verify_account(data)
        return data

    def owned_pods(self, config):
        """Prove exact-name pod absence before accepting a capacity-loss retry."""
        data = self.transport(
            "query { myself { id teams { id } pods { id name desiredStatus } } }", timeout=45
        )
        self.verify_account(data)
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


def request_worker(config, runtime, state, now, *, purpose, failure=None):
    """Persist a single worker claim before starting it, including repair-only work."""
    number = len(state["attempts"]) + 1
    directory = Path(config["capacity_state"]).parent / f"continuation-{number}"
    directory.mkdir(mode=0o700, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("unclaimed continuation directory contains prior work")
    state["attempts"].append(now)
    state["worker"] = dict(
        number=number,
        directory=str(directory),
        requested_at=now,
        source_sha=config["source_sha"],
        ledger_before=Path(config["allocation_ledger"]).read_text(),
        purpose=purpose,
        failure=failure,
    )
    if purpose == "repair":
        state.setdefault("repair_attempts", []).append({"number": number, "requested_at": now})
        state.pop("repair_pending", None)
    elif purpose == "cleanup":
        state.setdefault("cleanup_attempts", []).append({"number": number, "requested_at": now})
    if purpose != "cleanup":
        advance_schedule(config, state, now)
    write(config["capacity_state"], state)
    runtime.start(config)


def queue_repair(config, state, observation, worker, now, *, reason, evidence):
    """Record a failed attempt intact and schedule diagnosis independently of stock."""
    failure = dict(attempt=worker["number"], outcome=str(evidence), reason=reason, checked_at=now)
    state.setdefault("failure_history", []).append(failure)
    state["repair_pending"] = failure
    state["worker"] = None
    state["retry_after"] = now + config["interval_seconds"]
    observation.update(status="repair_queued", failure=failure)
    enqueue(
        state,
        f"repair-queued-{worker['number']}",
        "Task 2673: the run failed; automatic diagnosis and repair are queued. "
        "The next worker must test its fix before retrying compute. " + reason,
        now,
    )


def tick(config, runtime, now=None):
    """Rejected worker evidence triggers repair without being accepted as success."""
    now = time.time() if now is None else now
    try:
        return _tick(config, runtime, now)
    except OutcomeRejected as exc:
        if not config.get("automatic_repair"):
            raise
        state = json.loads(Path(config["capacity_state"]).read_text())
        worker = state["worker"]
        if not worker or runtime.unit(config["worker_unit"])["ActiveState"] in {
            "active",
            "activating",
            "deactivating",
        }:
            raise
        observation = dict(
            source_sha=config["source_sha"],
            checked_at=now,
            backend_observation={"status": "pending"},
        )
        queue_repair(
            config,
            state,
            observation,
            worker,
            now,
            reason=f"Rejected worker evidence: {exc}",
            evidence=worker["directory"],
        )
        write(config["observation"], observation)
        write(config["capacity_state"], state)
        runtime.notify(config, state, config["capacity_state"], now)
        return observation


def _tick(config, runtime, now):
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
    schedule_wait = False
    if not worker and config.get("daily_dispatch_timezone"):
        due = max(config["daily_dispatch_not_before"], state.get("next_dispatch_at", 0))
        # Waiting for tomorrow is safe only without an unresolved paid pod.
        # Repair/cleanup of an existing allocation retains prompt supervision.
        if now < due and not (state.get("repair_pending") and runtime.owned_pods(config)):
            schedule_wait = True
            observation.update(
                status="waiting_for_midnight",
                next_dispatch_at=due,
                timezone=config["daily_dispatch_timezone"],
            )
    if schedule_wait:
        pass
    elif worker:
        observation["status"] = "continuation_requested"
        directory = Path(worker["directory"])
        unit = runtime.unit(config["worker_unit"])
        live = unit["ActiveState"] in {"active", "activating", "deactivating"}
        outcome_path = directory / "outcome.json"
        exit_path = directory / "exit.json"
        outcome, exit_record, record_error = None, None, None
        if not live:
            records = {}
            for path in (outcome_path, exit_path):
                if path.exists():
                    try:
                        records[path.name] = json.loads(path.read_text())
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        record_error = f"Malformed worker record {path.name}: {exc}"
            outcome, exit_record = records.get("outcome.json"), records.get("exit.json")
            if outcome is not None and (
                not isinstance(outcome, dict)
                or not {"attempt", "source_sha", "status"} <= outcome.keys()
                or type(outcome["attempt"]) is not int
                or not digest(outcome["source_sha"], 40)
                or not isinstance(outcome["status"], str)
            ):
                record_error = "Worker outcome lacks its required identity/status fields"
            if exit_record is not None and (
                not isinstance(exit_record, dict)
                or not {"returncode", "ended_at"} <= exit_record.keys()
                or type(exit_record["returncode"]) is not int
                or not finite_number(exit_record["ended_at"])
            ):
                record_error = "Worker exit record lacks returncode/ended_at"
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
        elif (
            config.get("automatic_repair")
            and now - worker["requested_at"] > 90
            and (
                record_error
                or outcome is None
                or exit_record is None
                or exit_record["returncode"] != 0
            )
        ):
            queue_repair(
                config,
                state,
                observation,
                worker,
                now,
                reason=record_error
                or "Continuation exited without a clean outcome; reconcile its pod and checkpoints before retrying.",
                evidence=directory,
            )
        elif outcome is not None and exit_record is not None:
            if record_error:
                raise OutcomeRejected(record_error)
            if (
                outcome["attempt"] != worker["number"]
                or outcome.get("request_source_sha", outcome["source_sha"])
                != worker.get("source_sha", config["source_sha"])
                or outcome["source_sha"] != config["source_sha"]
            ):
                raise OutcomeRejected("stale continuation outcome")
            if (
                exit_record["returncode"] != 0
                or not worker["requested_at"] <= exit_record["ended_at"] <= now
            ):
                raise OutcomeRejected("continuation exited unsuccessfully")
            if outcome["status"] == "capacity_lost":
                if runtime.owned_pods(config) or Path(config["handle_file"]).exists():
                    raise OutcomeRejected("cannot reset capacity wait with a pod or handle present")
                ledger = Path(config["allocation_ledger"]).read_text()
                if ledger != worker["ledger_before"]:
                    raise OutcomeRejected("allocation ledger changed; bounded recovery required")
                losses = state.setdefault("verified_capacity_losses", [])
                if worker["number"] not in losses:
                    losses.append(worker["number"])
                state["worker"] = None
                state["retry_after"] = now + config["interval_seconds"]
                observation["status"] = "capacity_lost_waiting_again"
                enqueue(
                    state,
                    f"lost-{worker['number']}",
                    "Task 2673: the quoted H200 capacity disappeared before allocation. "
                    "No GPU was allocated; "
                    + (
                        "the next attempt remains scheduled for local midnight."
                        if config.get("daily_dispatch_timezone")
                        else f"capacity checks continue every {config['interval_seconds']} seconds."
                    ),
                    now,
                )
            elif outcome["status"] == "complete":
                try:
                    terminal_bytes = Path(config["terminal_file"]).read_bytes()
                    terminal = json.loads(terminal_bytes)
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise OutcomeRejected("missing or malformed terminal evidence") from exc
                if not isinstance(terminal, dict):
                    raise OutcomeRejected("terminal evidence must be an object")
                if (
                    terminal.get("source_sha") != config["source_sha"]
                    or terminal.get("model_key") != "deepseek"
                    or terminal.get("continuation_attempt") != worker["number"]
                    or outcome.get("terminal_sha256") != hashlib.sha256(terminal_bytes).hexdigest()
                    or not finite_number(terminal.get("checked_at"))
                    or not worker["requested_at"]
                    <= terminal.get("checked_at", 0)
                    <= exit_record["ended_at"]
                    <= now
                    or terminal.get("handle_file") != config["handle_file"]
                    or not digest(terminal.get("verified_revision"), 40)
                    or not digest(outcome.get("comparison_git_revision"), 40)
                    or not isinstance(outcome.get("report_url"), str)
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
                    or not finite_number(terminal.get("file_count"))
                    or terminal["file_count"] < 240
                    or not finite_number(terminal.get("total_bytes"))
                    or terminal["total_bytes"] <= 0
                    or any(
                        not digest(terminal.get(key), 64)
                        for key in ("capture_fingerprint", "analysis_fingerprint")
                    )
                    or runtime.owned_pods(config)
                ):
                    raise OutcomeRejected("completion/teardown evidence is incomplete")
                observation.update(status="completion_notification_pending", results=terminal)
                enqueue(
                    state,
                    "completed",
                    "Task 2673: DeepSeek extraction and Qwen comparison "
                    f"completed and verified. {outcome['report_url']}",
                    now,
                )
            elif outcome["status"] == "cleanup_complete" and worker.get("purpose") == "cleanup":
                ledger = json.loads(Path(config["allocation_ledger"]).read_text())
                if runtime.owned_pods(config) or any(
                    a.get("termination_confirmed_at_unix") is None for a in ledger["allocations"]
                ):
                    raise OutcomeRejected(
                        "cleanup must verify pod absence and close the paid ledger"
                    )
                proof = outcome.get("preservation_verification")
                if (
                    not isinstance(proof, dict)
                    or not digest(proof.get("verified_revision"), 40)
                    or proof.get("all_remote_names_sizes_hashes_pass") is not True
                    or not finite_number(proof.get("file_count"))
                    or proof["file_count"] <= 0
                    or not finite_number(proof.get("checked_at"))
                    or not worker["requested_at"] <= proof["checked_at"] <= exit_record["ended_at"]
                ):
                    raise OutcomeRejected("cleanup requires independent preservation evidence")
                state["worker"] = None
                state["retry_after"] = now + config["interval_seconds"]
                observation["status"] = "cleanup_verified"
            elif outcome["status"] == "needs_attention" and config.get("automatic_repair"):
                if "reason" in outcome and not isinstance(outcome["reason"], str):
                    raise OutcomeRejected("worker failure reason must be text")
                # A diagnosed workload failure is work for the repair owner, not a
                # crashed capacity checker. The new worker must fix/review before
                # any paid replay; it does not imply a successful GPU recovery.
                queue_repair(
                    config,
                    state,
                    observation,
                    worker,
                    now,
                    reason=outcome.get("reason", "inspect preserved failure evidence"),
                    evidence=outcome_path,
                )
            elif outcome["status"] == "blocked" and config.get("automatic_repair"):
                if (
                    not isinstance(outcome.get("stop_kind"), str)
                    or outcome.get("stop_kind")
                    not in {
                        "compute_limit",
                        "access_required",
                        "user_decision",
                        "user_cancelled",
                    }
                    or not outcome.get("evidence")
                    or not isinstance(outcome.get("reason"), str)
                ):
                    raise OutcomeRejected(
                        "a terminal block requires a concrete authority/evidence record"
                    )
                if runtime.owned_pods(config):
                    raise OutcomeRejected(
                        "cannot park repair while an owned paid pod remains unresolved"
                    )
                observation.update(status="awaiting_user", failure=outcome)
                enqueue(
                    state,
                    f"blocked-{worker['number']}",
                    "Task 2673: repair reached a user boundary: " + outcome["reason"],
                    now,
                )
            else:
                raise OutcomeRejected("continuation requires intervention: " + outcome["status"])
        elif now - worker["requested_at"] > 90:
            observation.update(
                status="backend_gate",
                backend_observation={
                    "status": "gate",
                    "reason": "continuation ended without a verified outcome",
                },
            )
    elif state.get("repair_pending") and config.get("automatic_repair"):
        recent = [a for a in state.get("repair_attempts", []) if now - a["requested_at"] < 86400]
        recent_attempts = [
            stamp
            for number, stamp in enumerate(state["attempts"], start=1)
            if now - stamp < 86400 and number not in state.get("verified_capacity_losses", [])
        ]
        limit = config["max_repairs_per_day"]
        if not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("automatic repair must have a bounded daily worker allowance")
        if len(recent) >= limit or len(recent_attempts) >= config["max_attempts_per_day"]:
            boundaries = []
            if len(recent) >= limit:
                boundaries.append(min(a["requested_at"] for a in recent) + 86400)
            if len(recent_attempts) >= config["max_attempts_per_day"]:
                boundaries.append(min(recent_attempts) + 86400)
            resume_at = max(boundaries)
            owned_pods = runtime.owned_pods(config)
            if owned_pods:
                cleanups = [
                    a for a in state.get("cleanup_attempts", []) if now - a["requested_at"] < 86400
                ]
                if len(cleanups) < 2:
                    request_worker(
                        config,
                        runtime,
                        state,
                        now,
                        purpose="cleanup",
                        failure={
                            "reason": "Repair cooldown cannot leave a paid pod running. Preserve, verify and terminate only.",
                            "owned_pods": owned_pods,
                            "prior_failure": state["repair_pending"],
                        },
                    )
                    observation["status"] = "cleanup_requested"
                else:
                    observation.update(
                        status="backend_gate",
                        backend_observation={
                            "status": "gate",
                            "reason": "Two cleanup workers failed with an unresolved paid pod; immediate intervention required.",
                        },
                    )
                    enqueue(
                        state,
                        "cleanup-exhausted-" + str(len(state["attempts"])),
                        "Task2673: paid pod remains after two bounded preservation/cleanup attempts. Immediate intervention is required; no healthy cooldown was entered.",
                        now,
                    )
            else:
                observation.update(status="repair_cooldown", retry_at=resume_at)
                enqueue(
                    state,
                    f"repair-cooldown-{len(state['attempts'])}",
                    f"Task 2673: repair workers reached their daily bound; automatic retry "
                    f"becomes eligible after Unix time {resume_at}, subject to the configured "
                    "daily schedule. Failure evidence is preserved.",
                    now,
                )
        elif now >= state.get("retry_after", 0):
            request_worker(
                config, runtime, state, now, purpose="repair", failure=state["repair_pending"]
            )
            observation["status"] = "repair_requested"
        else:
            observation["status"] = "repair_queued"
    elif now >= state.get("retry_after", 0):
        data = runtime.query()
        offers = eligible(data)
        advance_schedule(config, state, now)
        observation["capacity"] = data
        if offers:
            # Verified no-allocation outcomes do not consume a paid/recovery attempt.
            losses = set(state.get("verified_capacity_losses", []))
            recent = [
                stamp
                for number, stamp in enumerate(state["attempts"], start=1)
                if now - stamp < 86400 and number not in losses
            ]
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
                enqueue(
                    state,
                    f"capacity-{number}",
                    "Task 2673: eight-H200 capacity is visible. "
                    "A continuation has been requested to recheck it and resume the approved "
                    "DeepSeek extraction within the existing compute allowance.",
                    now,
                )
                request_worker(config, runtime, state, now, purpose="capacity")
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
        f"Worker purpose: {attempt.get('purpose', 'capacity')}. "
        f"Preserved failure to diagnose: {json.dumps(attempt.get('failure'))}.\n"
        f"Immutable request_source_sha is {attempt.get('source_sha', config['source_sha'])}; "
        "include it in the outcome if a reviewed fix changes source_sha.\n"
    )
    if attempt.get("purpose") == "cleanup":
        prompt += (
            "\nCLEANUP-ONLY OWNER: do not allocate, change model source, or run captures. "
            "Preserve and independently verify existing outputs, use canonical suffixed teardown, "
            "close the paid ledger and verify exact pod absence. You have 600 seconds. "
            "Return cleanup_complete with preservation_verification only after all checks; "
            "otherwise preserve the concrete failure and deliver an acknowledged urgent alert.\n"
        )
    write(directory / "started.json", {"started_at": time.time(), "attempt": attempt["number"]})
    with (directory / "worker.log").open("a") as log:
        proc = subprocess.run(
            runtime.helper.codex_command(config, directory / "result.txt"),
            input=prompt,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600 if attempt.get("purpose") == "cleanup" else config["continuation_seconds"],
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
            # Reload reviewed source transitions without replacing the immutable
            # request source held in each in-flight worker claim.
            config = json.loads(args.config.read_text())
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
