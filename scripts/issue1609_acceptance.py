"""Live acceptance probe for the fellows (charmander) SLURM lane (#1609).

Drives ONE real 1-GPU ``debug``-intent job end-to-end through the SHARED
SlurmBackend machinery — rsync prepare (branch tree) -> secrets push ->
sbatch submit -> poll to ``[phase=done]`` -> fetch_results -> teardown —
and writes the acceptance record to
``eval_results/issue_1609/acceptance/acceptance_record.json``. Passing
this probe is the #1609 §7 gate for flipping the fellows row to
``available=True``.

IMPORT-MODE ONLY (#987 trap): ``scripts/dispatch_issue.py`` /
``scripts/backend_poll.py`` self-pin their ``explore_persona_space.backends``
imports to the MAIN checkout when run script-mode, and main does NOT have
the fellows lane until this branch merges — so this driver is invoked as a
MODULE from the worktree with the worktree ``src/`` on ``PYTHONPATH``::

    cd <worktree> && PYTHONPATH=<worktree>/src \
      /home/thomasjiralerspong/explore-persona-space/.venv/bin/python \
      -m scripts.issue1609_acceptance

Prereqs: the ``issue-1609`` branch is PUSHED (``materialize_branch_src``
fetches it when the rsync source is not already on the branch), and
``~/.ssh/clusters.config`` Host ``charmander`` resolves.

The fellows row ships dark-launched (``available=False``); this driver
forces it available IN-PROCESS only — the committed row flips only after
this probe passes.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ISSUE = 1609
BRANCH = "issue-1609"
POLL_INTERVAL_S = 20.0
# Cold venv build on the MooseFS mount is ~6-10 min; budget generously.
POLL_BUDGET_S = 45 * 60.0

WORKLOAD_CMD = (
    "nvidia-smi -L"
    ' && echo "HF_HOME=$HF_HOME"'
    ' && test -w "$HF_HOME" && echo HF_HOME_WRITABLE'
    " && uv run python -c"
    " \"import torch; print('torch_cuda_device_count=' + str(torch.cuda.device_count()))\""
)


def _record_path() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "eval_results" / "issue_1609" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "acceptance_record.json"


def _write_record(record: dict) -> None:
    path = _record_path()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[acceptance] record written: {path}")


def main() -> int:
    """Run the end-to-end fellows acceptance job; exit 0 only on full PASS."""
    from explore_persona_space.backends import slurm as slurm_mod
    from explore_persona_space.backends.base import RunSpec
    from explore_persona_space.backends.slurm import SlurmBackend, job_name

    # Dark-launch override: force the fellows row available IN-PROCESS so
    # prepare/launch/poll resolve it; the committed row stays False until
    # this probe passes (#1609 §7 gate).
    slurm_mod.CLUSTER_CONFIGS["fellows"] = dataclasses.replace(
        slurm_mod.CLUSTER_CONFIGS["fellows"], available=True
    )
    cluster = slurm_mod.CLUSTER_CONFIGS["fellows"]

    spec = RunSpec(
        issue=ISSUE,
        intent="debug",
        backend="fellows",
        cluster="fellows",
        workload_cmd=WORKLOAD_CMD,
        extra={"repo_branch": BRANCH},
    )
    backend = SlurmBackend()

    record: dict = {
        "issue": ISSUE,
        "branch": BRANCH,
        "cluster": "fellows",
        "robot_alias": cluster.ssh_host,
        "intent": "debug",
        "workload_cmd": WORKLOAD_CMD,
        "started_utc": datetime.now(UTC).isoformat(),
        "verdict": "INCOMPLETE",
        "polls": [],
    }

    # Probe hygiene (kill-before-relaunch): a prior acceptance attempt's
    # live job under the SAME canonical name is scancelled before a fresh
    # submit — never two probes racing one scratch dir.
    from explore_persona_space.backends.slurm_monitor import query_by_name

    name = job_name(spec, plan_hash=None, cluster=cluster)
    prior = query_by_name(robot_alias=cluster.ssh_host, job_name=name)
    if prior:
        print(f"[acceptance] prior live job {prior} under {name!r}; scancelling first")
        subprocess.run(["ssh", cluster.ssh_host, "scancel", str(prior)], check=True, timeout=30)
        time.sleep(5)

    print(f"[acceptance] prepare: rsync branch tree + secrets -> {cluster.ssh_host}")
    backend.prepare(spec)
    print("[acceptance] launch: sbatch submit")
    handle = backend.launch(spec)
    record["job_id"] = handle.job_id
    record["job_name"] = name
    record["scratch_dir"] = handle.scratch_dir
    print(f"[acceptance] submitted job_id={handle.job_id} name={name} scratch={handle.scratch_dir}")

    deadline = time.monotonic() + POLL_BUDGET_S
    status = "running"
    phase = None
    saw_running = False
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
        result = backend.poll(handle)
        status = result.status
        phase = result.current_phase
        saw_running = saw_running or status == "running"
        stamp = datetime.now(UTC).isoformat()
        record["polls"].append({"ts": stamp, "status": status, "phase": phase})
        print(f"[acceptance] poll: status={status} phase={phase}")
        if status in {"done", "dead"}:
            break
    record["final_status"] = status
    record["final_phase"] = phase
    record["saw_running"] = saw_running

    # Evidence: job.out tail (workload prints ride here) — best-effort.
    try:
        log_tail = backend.fetch_logs(handle)
    except Exception as exc:  # noqa: BLE001 - evidence fetch must not mask the verdict
        log_tail = f"<fetch_logs failed: {type(exc).__name__}: {exc}>"
    record["job_out_tail"] = log_tail[-8000:]

    # Pull results (completion sentinel under eval_results/) back to the tree.
    try:
        backend.fetch_results(handle)
        record["fetch_results"] = "ok"
    except Exception as exc:  # noqa: BLE001 - WARN-only by the #598 contract
        record["fetch_results"] = f"failed: {type(exc).__name__}: {exc}"

    # Teardown: idempotent scancel (a COMPLETED job is a clean no-op).
    try:
        backend.teardown(handle)
        record["teardown"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record, don't mask
        record["teardown"] = f"failed: {type(exc).__name__}: {exc}"

    workload_ok = (
        "HF_HOME_WRITABLE" in log_tail
        and "torch_cuda_device_count=1" in log_tail
        and "HF_HOME=/workspace/pretrained_ckpts" in log_tail
    )
    record["workload_evidence"] = {
        "hf_home_writable": "HF_HOME_WRITABLE" in log_tail,
        "torch_cuda_device_count_1": "torch_cuda_device_count=1" in log_tail,
        "hf_home_resolved_shared": "HF_HOME=/workspace/pretrained_ckpts" in log_tail,
        "gpu_listed": "GPU 0" in log_tail,
    }
    passed = status == "done" and workload_ok
    record["verdict"] = "PASS" if passed else "FAIL"
    record["finished_utc"] = datetime.now(UTC).isoformat()
    record["known_misses"] = [
        "multi-GPU sbatch shape at N=8 (only N=1 exercised)",
        "NCCL ens1/NVLS exports under real multi-GPU load (exports rendered, not exercised)",
        "16-GPU/user-cap park behavior (queue was empty)",
        "low-eur preemption semantics (high-eur only)",
    ]
    _write_record(record)
    print(f"[acceptance] verdict: {record['verdict']} (status={status}, workload_ok={workload_ok})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
