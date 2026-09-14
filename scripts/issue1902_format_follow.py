"""Local continuation: start dedicated CPU fits after verified GPU aggregates arrive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import issue1902_format_common as C
from issue1902_watch_backend import observe


def post_state(args):
    """Persist noteworthy observation changes through the canonical task interface."""
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/task.py",
            "post-marker",
            "1902",
            "epm:progress",
            "--file",
            str(args.state),
            "--by",
            "codex-format-continuation",
        ],
        cwd=args.repo_root,
        check=True,
    )


def current_artifact(name, expected_sha):
    """Read an immutable completion artifact only for this exact source revision."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revision = C.retry_transient(
        lambda: api.repo_info(C.REPO, repo_type="dataset").sha,
        what="format continuation repository revision",
    )
    remote = f"{C.PREFIX}/{name}"
    entries = C.retry_transient(
        lambda: api.get_paths_info(C.REPO, [remote], repo_type="dataset", revision=revision),
        what=f"format continuation {name} metadata",
    )
    if not entries:
        return None
    path = C.retry_transient(
        lambda: hf_hub_download(C.REPO, remote, repo_type="dataset", revision=revision),
        what=f"format continuation {name} download",
    )
    value = json.loads(Path(path).read_text())
    if value["source_sha"] != expected_sha:
        raise RuntimeError(f"Completion source mismatch for {name}")
    return dict(value, verified_revision=revision)


def main():
    """Record every terminal monitor error, including Hub/launch failures, then raise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gpu-handle", type=Path, required=True)
    args = parser.parse_args()
    try:
        monitor(args)
    except Exception as exc:
        state = json.loads(args.state.read_text()) if args.state.exists() else {}
        state.update(
            monitor_error=f"{type(exc).__name__}: {exc}",
            monitor_stopped_at=time.time(),
        )
        if state.get("status") not in {"backend_failed", "backend_gate", "cpu_launch_failed"}:
            state["status"] = "monitor_failed"
        C.write_json(args.state, state)
        post_state(args)
        raise


def monitor(args):
    """Persist launch identity before monitoring fits; never invoke model judges."""
    state = (
        json.loads(args.state.read_text())
        if args.state.exists()
        else dict(
            source_sha=args.source_sha, status="waiting_for_gpu_aggregates", branch=args.branch
        )
    )
    assert state["source_sha"] == args.source_sha and state["branch"] == args.branch
    C.write_json(args.state, state)
    deadline = time.monotonic() + 96 * 3600
    observation_failures = 0
    while time.monotonic() < deadline:
        # Verified deliverables take precedence over a later finalization failure.
        completed = current_artifact(
            "fits_complete.json" if "cpu_handle" in state else "gpu_complete.json",
            args.source_sha,
        )
        if "cpu_handle" in state and completed is not None:
            state.update(status="complete", results=completed, checked_at=time.time())
            C.write_json(args.state, state)
            post_state(args)
            print("[phase=continuation_complete] Format fits verified on Hub", flush=True)
            return
        handle_path = (
            state["cpu_handle"]["handle_sidecar_path"] if "cpu_handle" in state else args.gpu_handle
        )
        try:
            observed = (
                observe(handle_path, args.repo_root)
                if completed is None
                else dict(status="done", evidence="exact-source verified GPU completion artifact")
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            observation_failures += 1
            state.update(
                status="backend_observation_error",
                checked_at=time.time(),
                consecutive_observation_failures=observation_failures,
                error=str(exc),
            )
            C.write_json(args.state, state)
            if observation_failures in (1, 3):
                post_state(args)
            if observation_failures >= 3:
                raise RuntimeError("Three backend probes failed; run state is unknown") from exc
            time.sleep(60)
            continue
        recovered = observation_failures > 0
        observation_failures = 0
        state.update(
            status="cpu_fits_launched" if "cpu_handle" in state else "waiting_for_gpu_aggregates",
            checked_at=time.time(),
            backend_observation=observed,
            consecutive_observation_failures=0,
        )
        state.pop("error", None)
        C.write_json(args.state, state)
        if recovered:
            post_state(args)
        if observed["status"] in {"dead", "failed", "error", "gate"}:
            state.update(
                status="backend_failed" if observed["status"] != "gate" else "backend_gate"
            )
            C.write_json(args.state, state)
            post_state(args)
            raise RuntimeError(f"Backend requires attention: {observed}")
        if "cpu_handle" not in state:
            result = completed
            if result is None:
                time.sleep(60)
                continue
            workload = (
                "uv run python scripts/issue1902_format_fits.py --input-revision "
                f"{result['input_revision']} --source-sha {args.source_sha}"
            )
            command = [
                "uv",
                "run",
                "python",
                "scripts/dispatch_issue.py",
                "launch",
                "--issue",
                "1902",
                "--intent",
                "cpu-mid",
                "--backend",
                "gcp",
                "--no-runpod-fallback",
                "--lane-suffix",
                "format-fits",
                "--repo-branch",
                args.branch,
                "--boot-disk-gb",
                "150",
                "--min-ram-gb",
                "24",
                "--max-run-duration",
                "12h",
                "--time-budget-hours",
                "12",
                "--skip-default-git-paths",
                "--workload-cmd",
                workload,
            ]
            launch = subprocess.run(
                command,
                cwd=args.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
            log = args.state.with_suffix(".cpu-launch.log")
            log.write_text(launch.stdout)
            if launch.returncode:
                state.update(status="cpu_launch_failed", launch_log=str(log))
                C.write_json(args.state, state)
                raise RuntimeError(f"CPU launch failed; inspect {log}")
            handle = json.loads(launch.stdout.splitlines()[-1])
            assert handle["ok"] is True
            state.update(status="cpu_fits_launched", gpu_result=result, cpu_handle=handle)
            C.write_json(args.state, state)
            print(f"[phase=cpu_launched] {handle['handle_sidecar_path']}", flush=True)
        completed = current_artifact("fits_complete.json", args.source_sha)
        if completed is not None:
            state.update(status="complete", results=completed)
            C.write_json(args.state, state)
            post_state(args)
            print("[phase=continuation_complete] Format fits verified on Hub", flush=True)
            return
        time.sleep(60)
    state.update(status="monitor_deadline_reached")
    C.write_json(args.state, state)
    raise TimeoutError("Format continuation reached its 96-hour monitoring deadline")


if __name__ == "__main__":
    main()
