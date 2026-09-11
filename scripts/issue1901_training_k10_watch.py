#!/usr/bin/env python3
"""Durable VM monitor and CPU continuation for the approved training-K run.

This follows one named router handle, then stages, audits, fits and publishes.
The canonical poller may perform authorized same-workload backend failover.
Final independent review and gated router finalize remain with the owner;
this continuation does not change task bodies or status.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
MAIN = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(MAIN / "src"))
from explore_persona_space import task_workflow as workflow  # noqa: E402

INPUT_REVISION = "b281fc98da9fc87b4aabe2463cd24bc26d8ad115"
INPUT_MANIFEST = "01913f081a84c954a282a03c794b35015393e9a30cd78da0885cbdb11b041b74"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def command(argv, cwd, log):
    print(f"[{now()}] command {argv}", flush=True)
    with log.open("a") as stream:
        subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def event_note_payload(event):
    """Resolve a canonical marker's declared full-note evidence under its task artifacts."""
    note = event.get("note", "")
    if event.get("oversize") is True:
        artifacts = event.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise ValueError("Oversize result must declare exactly one full-note artifact")
        name = Path(artifacts[0]).name
        if not re.fullmatch(r"sentinel-note-epm_results-[0-9]+\.txt", name):
            raise ValueError("Oversize result has an unexpected full-note artifact name")
        # The workflow API resolves current task location, including status moves.
        # Only the event-declared basename is used; no arbitrary task path is read.
        directory = (workflow.find_task_path(1901) / "artifacts").resolve()
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Oversize result artifact escapes its canonical task directory")
        note = path.read_text(encoding="utf-8")
        if len(note) != event["oversize_orig_len"]:
            raise ValueError("Oversize result artifact length differs from the canonical marker")
    try:
        return json.loads(note)
    except json.JSONDecodeError:
        return None  # Other task-1901 result rounds legitimately use prose notes.


def capture_result(after):
    results = {}
    for event in workflow.list_events(1901):
        if event["kind"] != "epm:results" or event["ts"] < after:
            continue
        note = event_note_payload(event)
        if isinstance(note, dict) and note.get("round") == "training-k10-capture":
            assert note["new_rows"] == 95000
            results[note["recipe_sha256"]] = note
    if len(results) != 1:
        raise ValueError(f"Expected one fresh training-K capture result, found {len(results)}")
    return next(iter(results.values()))


def monitor(args):
    previous = None
    while True:
        started = time.monotonic()
        completed = subprocess.run(
            [
                sys.executable,
                str(MAIN / "scripts/backend_poll.py"),
                "--issue",
                "1901",
                "--handle-file",
                str(args.handle_file),
            ],
            cwd=MAIN,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        # The documented final stdout line is the complete PollResult JSON.
        state = json.loads(completed.stdout.splitlines()[-1])
        snapshot = {
            "at": now(),
            "state": state,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        with (args.root / "monitor.jsonl").open("a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        write_json(args.root / "monitor_latest.json", snapshot)
        summary = (state["status"], state["current_phase"], state["log_tail_excerpt"])
        if summary != previous:
            print(
                f"[{now()}] {state['status']} {state['current_phase']}\n"
                f"{state['log_tail_excerpt']}",
                flush=True,
            )
            previous = summary
        if state["status"] == "done":
            return capture_result(args.started_after)
        if state["status"] not in ("running", "pid-stale-workload-live"):
            raise RuntimeError(f"Capture needs owner attention: {state}")
        # Only this background process waits; the owning assistant stays responsive.
        time.sleep(max(1, 120 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--started-after", default="2026-09-11T18:55:00Z")
    args = parser.parse_args()
    assert args.handle_file.name == "issue-1901-traink10-cap-handle.json"
    assert args.handle_file.is_file(), args.handle_file
    args.root.mkdir(parents=True, exist_ok=True)
    # Refuse concurrent instances; process lifetime holds the advisory lock.
    import fcntl

    lock = (args.root / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("continuation_complete.json", "continuation_failure.json"):
        prior = args.root / name
        if prior.exists():
            archive = args.root / "monitor_history" / str(time.time_ns())
            archive.mkdir(parents=True)
            prior.rename(archive / name)
    (args.root / "monitor.pid").write_text(str(os.getpid()) + "\n")
    write_json(
        args.root / "monitor_launch.json",
        {
            "pid": os.getpid(),
            "at": now(),
            "handle_file": str(args.handle_file),
            "worktree": str(WORKTREE),
            "input_revision": INPUT_REVISION,
        },
    )
    try:
        note = monitor(args)
        write_json(args.root / "capture_result.json", note)
        capture, analysis = args.root / "capture", args.root / "analysis"
        report_script = str(WORKTREE / "scripts/issue1901_training_k10_report.py")
        command(
            [
                sys.executable,
                report_script,
                "--phase",
                "stage",
                "--capture",
                str(capture),
                "--capture-prefix",
                note["hf_prefix"],
                "--capture-revision",
                note["upload_verification"]["revision"],
                "--audit-raw",
            ],
            WORKTREE,
            args.root / "stage.log",
        )
        manifest = json.loads((capture / "capture_manifest.json").read_text())
        assert manifest["input_revision"] == INPUT_REVISION
        assert manifest["input_manifest_sha256"] == INPUT_MANIFEST
        assert manifest["recipe_sha256"] == note["recipe_sha256"]
        command(
            [
                sys.executable,
                str(WORKTREE / "scripts/issue1901_training_k10_fit.py"),
                "--inputs",
                str(args.root / "inputs"),
                "--capture",
                str(capture),
                "--out",
                str(analysis),
            ],
            WORKTREE,
            args.root / "fit.log",
        )
        command(
            [
                sys.executable,
                report_script,
                "--phase",
                "publish",
                "--capture",
                str(capture),
                "--analysis-dir",
                str(analysis),
                "--heatmaps",
            ],
            WORKTREE,
            args.root / "report.log",
        )
        receipt = json.loads((analysis / "publication_receipt.json").read_text())
        write_json(
            args.root / "continuation_complete.json",
            {
                "at": now(),
                "publication": receipt,
                "capture_result": note,
                "remaining": "Independent final evidence review and gated exact-handle finalize",
            },
        )
        workflow.post_event(
            1901,
            "epm:progress",
            by="codex-training-k10-20260911",
            note="followup_label=training-k10-19k CPU continuation completed; "
            f"publication_receipt={analysis / 'publication_receipt.json'}; "
            "awaiting independent final evidence review and exact-handle finalize.",
        )
        print(f"[{now()}] CPU continuation complete; final review pending.", flush=True)
    except Exception as error:
        write_json(
            args.root / "continuation_failure.json",
            {
                "at": now(),
                "error": repr(error),
                "pid": os.getpid(),
            },
        )
        print(f"[{now()}] Continuation failed: {error!r}", flush=True)
        workflow.post_event(
            1901,
            "epm:progress",
            by="codex-training-k10-20260911",
            note="followup_label=training-k10-19k continuation_failed; "
            f"error={error!r}; evidence={args.root / 'continuation_failure.json'}; "
            "owner attention required; task status and compute left unchanged.",
        )
        raise


if __name__ == "__main__":
    main()
