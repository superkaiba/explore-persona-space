#!/usr/bin/env python3
"""One-shot VM owner for the approved A100 pilot -> nine K3 cells.

Run under a user systemd service (linger enabled), not an LLM agent or cron.
The original 10000-row timing gate is fixed here. This process never changes
experiment recipes, terminates compute, or invokes an automated model review.
Existing task.py and GcpBackend own markers and polling. Launches are journaled
before submission: an interrupted submission requires reconciliation, never a
blind retry that could duplicate a live cell.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT_SHA = "caf9823f1fe9ccc0217b8089ce95752d4ce994bb"
PILOT_ID = "5630515292704346242"
PILOT = "q3_32b_a"
CELLS = (
    "o31_32b_i_a",
    "q38_27b_a",
    "q36_27b_a",
    "q35_27b_a",
    "q35_9b_a",
    "o3_7b_i_a",
    "q35_4b_a",
    "q35_2b_a",
    "q35_0p8b_a",
)
PROJECT = "eps-persona-gpu-jun2026"
PREFIX = "issue2588_capability_panel/k3_train_refit"
DATA_REPO = "superkaiba1/explore-persona-space-data"

# Stdlib only on the pilot; inspect terminal artifacts, not the existence of
# an output directory. Chunks are written before the cap report and throughput.
PROBE = r"""
import json, subprocess
from pathlib import Path
root = Path('/workspace/eps-issue-2588/data/issue_2588/k3_train_refit/cells/q3_32b_a')
def read(path):
    return json.loads(path.read_text()) if path.exists() else None
data = {'gate': read(root / 'fits/k3_gate_prompt_last.json'),
        'throughput': read(root / 'fits/k3_gen_throughput.json'), 'parts': {}}
for k in range(4):
    stage = f'train_10k_s45_part{k}'
    folder = root / 'raw_completions' / stage
    report = read(folder / 'cap_hit_report.json')
    if report is None:
        continue
    chunks = [read(p) for p in sorted(folder.glob('chunk*.json'))]
    data['parts'][stage] = {'report': report, 'chunks': [
        {'meta': c['meta'], 'cell': c['cell'], 'stage': c['stage'],
         'seed': c['seed'], 'n': len(c['rows']),
         'ids': [r['row_id'] for r in c['rows']]} for c in chunks]}
data['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
    '--format=csv,noheader,nounits'], text=True).strip()
print(json.dumps(data))
"""


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def records(text: str) -> list[dict]:
    """The dispatch/task CLI can emit several consecutive JSON objects."""
    out = []
    decoder = json.JSONDecoder()
    while text.strip():
        value, end = decoder.raw_decode(text.lstrip())
        if not isinstance(value, dict):
            raise ValueError("CLI output is not a JSON object")
        out.append(value)
        text = text.lstrip()[end:]
    return out


def pilot_basis(snapshot: dict) -> dict | None:
    gate = snapshot["gate"]
    if gate is None:
        return None
    if gate["meta"]["git_sha"] != PILOT_SHA or gate["pass"] is not True:
        raise ValueError("pilot reproduction gate failed or source changed")
    if snapshot["gpu"] != "NVIDIA A100-SXM4-80GB, 81920":
        raise ValueError(f"pilot venue mismatch: {snapshot['gpu']}")
    rates = snapshot["throughput"]
    if rates is None:
        return None
    if rates["meta"]["git_sha"] != PILOT_SHA:
        raise ValueError("throughput source changed")
    names = [f"train_10k_s45_part{k}" for k in range(4)]
    if any(n not in snapshot["parts"] or n not in rates["stages"] for n in names):
        return None
    wall = 0.0
    cap_reports = []
    for name in names:
        timing = rates["stages"][name]
        part = snapshot["parts"][name]
        report = part["report"]
        if (timing["stage"], timing["seed"], timing["n_rows"], timing["generated"]) != (
            name,
            45,
            2500,
            True,
        ):
            raise ValueError(f"wrong timing manifest: {name}")
        elapsed = float(timing["wall_s"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"invalid pilot timing: {name}")
        if report["meta"]["git_sha"] != PILOT_SHA or report["n"] != 2500:
            raise ValueError(f"invalid cap report: {name}")
        ids = []
        for chunk in part["chunks"]:
            if (chunk["meta"]["git_sha"], chunk["cell"], chunk["stage"], chunk["seed"]) != (
                PILOT_SHA,
                PILOT,
                name,
                45,
            ):
                raise ValueError(f"chunk provenance mismatch: {name}")
            if chunk["n"] != len(chunk["ids"]):
                raise ValueError(f"chunk row count mismatch: {name}")
            ids.extend(chunk["ids"])
        if len(ids) != 2500 or len(set(ids)) != 2500:
            raise ValueError(f"incomplete or duplicate pilot rows: {name}")
        wall += elapsed
        cap_reports.append(report)
    generation_h = wall / 10000 * 20800 / 3600
    # Each remaining cell gets one A100. This is a generation-only estimate;
    # capture/fits remain unmeasured and retain the default seven-day fence.
    if 2 * 1.25 * generation_h >= 168:
        raise ValueError("pilot generation projection exceeds seven-day fence")
    return {
        "n_rows": 10000,
        "wall_s": wall,
        "responses_per_s": 10000 / wall,
        "generation_h_per_cell_at_pilot_rate": generation_h,
        "generation_dispersion_margin_h": 2 * 1.25 * generation_h,
        "cap_reports": cap_reports,
        "gpu": snapshot["gpu"],
    }


def capture_files(manifest, *, prefix, cell, stage, source):
    if (manifest["meta"]["git_sha"], manifest["cell"], manifest["stage"]) != (source, cell, stage):
        raise ValueError(f"capture manifest provenance mismatch: {cell}/{stage}")
    ids = [row["row_id"] for row in manifest["rows"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"empty/duplicate capture manifest: {cell}/{stage}")
    return {f"{prefix}shard{k:03d}.npz" for k in range(math.ceil(len(ids) / 500))}


class Owner:
    def __init__(self, args):
        self.args = args
        self.state_path = args.state_dir / "state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"started_at": time.time(), "launches": {}, "status": "waiting_pilot"}
        )
        if self.state.setdefault("source_sha", args.source_sha) != args.source_sha:
            raise ValueError("owner state belongs to another source version")
        self.env = {
            **os.environ,
            "PYTHONPATH": f"{args.repo}/src:{args.repo}",
            "UV_CACHE_DIR": "/tmp/eps-codex-resume-uv",
        }
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.env[key] = "8"
        self.env["MALLOC_ARENA_MAX"] = "2"

    def command(self, argv, *, name, stdin=None, timeout=300) -> list[dict]:
        log = self.args.state_dir / f"{name}.log"
        output = self.args.state_dir / f"{name}.json"
        with log.open("a") as err, output.open("w") as out:
            subprocess.run(
                argv,
                cwd=self.args.repo,
                env=self.env,
                input=stdin,
                text=True,
                stdout=out,
                stderr=err,
                check=True,
                timeout=timeout,
            )
        return records(output.read_text())

    def marker(self, note):
        path = self.args.state_dir / "marker.txt"
        path.write_text(note + "\n")
        self.command(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/task.py",
                "post-marker",
                "2588",
                "epm:progress",
                "--file",
                str(path),
                "--by",
                "codex-k3-owner",
            ],
            name="marker",
        )

    def persist(self):
        self.state["updated_at"] = time.time()
        save(self.state_path, self.state)

    def snapshot(self):
        handle = json.loads(self.args.pilot_handle.read_text())
        if handle["job_id"] != PILOT_ID or handle["backend"] != "gcp":
            raise ValueError("pilot handle identity changed")
        described = self.command(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                handle["pod_name"],
                "--zone",
                handle["extra"]["zone"],
                "--configuration=eps-gcp",
                "--format=json",
            ],
            name="pilot-instance",
        )[0]
        if str(described["id"]) != PILOT_ID or described["status"] != "RUNNING":
            raise ValueError("pilot instance incarnation/state changed")
        result = self.command(
            [
                "gcloud",
                "compute",
                "ssh",
                handle["pod_name"],
                "--zone",
                handle["extra"]["zone"],
                "--configuration=eps-gcp",
                "--command",
                "sudo -n /workspace/venvs/eps2588_k3/bin/python -",
            ],
            name="pilot-snapshot",
            stdin=PROBE,
        )[0]
        return result

    def launch(self, cell):
        if self.state.get("launching"):
            raise RuntimeError("previous submission incomplete; reconcile before any new launch")
        suffix = "k3-" + cell.replace("_", "-")
        handle_path = self.args.repo / f".claude/cache/issue-2588-{suffix}-handle.json"
        if handle_path.exists():
            raise RuntimeError(f"existing handle requires ownership reconciliation: {handle_path}")
        remote = subprocess.check_output(
            ["git", "ls-remote", "origin", "refs/heads/issue-2588-k3refit"],
            cwd=ROOT,
            text=True,
            timeout=60,
        ).split()
        if not remote or remote[0] != self.args.source_sha:
            raise RuntimeError("approved remote branch tip changed")
        quota = self.command(
            [
                "gcloud",
                "compute",
                "regions",
                "describe",
                "us-central1",
                "--configuration=eps-gcp",
                f"--project={PROJECT}",
                "--format=json",
            ],
            name=f"quota-{cell}",
        )[0]
        rows = {q["metric"]: q for q in quota["quotas"]}
        q = rows["PREEMPTIBLE_NVIDIA_A100_80GB_GPUS"]
        if q["limit"] - q["usage"] < 1:
            return False
        self.state["launching"] = cell
        self.persist()
        # GCP-only single-cell instances avoid width-degradation sizing drift
        # and let each completed cell release its allocation independently.
        result = self.command(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/dispatch_issue.py",
                "launch",
                "--issue",
                "2588",
                "--backend",
                "gcp",
                "--intent",
                "capture-7b",
                "--gpus",
                "1",
                "--provisioning-model",
                "SPOT",
                "--spot-tolerant",
                "--min-gpu-mem-gb",
                "70",
                "--boot-disk-gb",
                "300",
                "--no-runpod-fallback",
                "--repo-branch",
                "issue-2588-k3refit",
                "--lane-suffix",
                suffix,
                "--skip-default-git-paths",
                "--workload-cmd",
                f"uv run --no-sync python scripts/issue2588_k3_job.py --cell {cell} "
                f"--source-sha {self.args.source_sha}",
            ],
            name=f"launch-{cell}",
            timeout=900,
        )[-1]
        if result.get("ok") is not True or not handle_path.exists():
            raise RuntimeError(f"launch did not confirm a saved handle: {cell}")
        handle = json.loads(handle_path.read_text())
        if handle["backend"] != "gcp" or handle["extra"]["gpu_count"] != 1:
            raise RuntimeError(f"unexpected realized allocation: {cell}")
        self.state["launches"][cell] = {
            "handle": str(handle_path),
            "launched_at": handle["extra"]["gcp_launched_ts"],
        }
        self.state.pop("launching")
        self.persist()
        self.marker(
            f"K3 cell launched: cell={cell} pod={handle['pod_name']} "
            f"instance_id={handle['job_id']} handle={handle_path} "
            f"source={self.args.source_sha}; launch command returned ok. "
            "Workload pid/log confirmation pending next GCP poll. verified-by: ran"
        )
        return True

    def poll(self, handle_path):
        # Call the backend's canonical poll without backend_poll.py's automatic
        # GCP->RunPod retry handlers. This resume is expressly A100/GCP-only.
        from explore_persona_space.backends.base import RunHandle
        from explore_persona_space.backends.gcp import GcpBackend

        handle = RunHandle(**json.loads(Path(handle_path).read_text()))
        if handle.backend != "gcp":
            raise ValueError("refusing non-GCP handle")
        return dataclasses.asdict(GcpBackend().poll(handle))

    def confirm(self, cell, launch):
        """GCP running/pid_alive is VM-level; inspect the actual workload process."""
        handle = json.loads(Path(launch["handle"]).read_text())
        script = r"""
import json, subprocess, sys
from pathlib import Path
cell, since, sha = sys.argv[1:]
pidfile = Path('/workspace/logs') / f'issue-2588-{cell}.pid'
log = Path('/workspace/logs/issue-2588.log')
out = {'confirmed': False}
if pidfile.exists() and log.exists():
    pid = int(pidfile.read_text().strip())
    proc = Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        cmd = proc.read_bytes().replace(bytes([0]), b' ').decode()
        fresh = log.stat().st_mtime >= float(since) and log.stat().st_size > 0
        actual = subprocess.check_output(['git', '-C', '/workspace/eps-issue-2588',
            'rev-parse', 'HEAD'], text=True).strip()
        if actual != sha:
            raise RuntimeError(f'bootstrap source changed: {actual}')
        out = {'confirmed': fresh and 'issue2588_k3_job.py' in cmd and cell in cmd,
               'pid': pid, 'cmd': cmd, 'log': str(log), 'log_mtime': log.stat().st_mtime,
               'source_sha': actual}
print(json.dumps(out))
"""
        command = shlex.join(
            ["sudo", "-n", "python3", "-", cell, str(launch["launched_at"]), self.args.source_sha]
        )
        result = self.command(
            [
                "gcloud",
                "compute",
                "ssh",
                handle["pod_name"],
                "--zone",
                handle["extra"]["zone"],
                "--configuration=eps-gcp",
                "--command",
                command,
            ],
            name=f"confirm-{cell}",
            stdin=script,
        )[0]
        if result["confirmed"]:
            launch["confirmed"] = result
            self.marker(
                f"K3 workload confirmed: cell={cell} handle={launch['handle']} "
                f"launch_confirmed=pid+log@{time.time()} evidence={json.dumps(result)}; "
                "verified-by: ran"
            )
        return result["confirmed"]

    def harvest(self, cell):
        """Check current uploaded data before accepting an auto-deleted GCP run."""
        from explore_persona_space.orchestrate.env import load_dotenv

        load_dotenv()
        from huggingface_hub import HfApi, hf_hub_download
        from explore_persona_space.orchestrate import hub as HUB

        api = HfApi()
        revision = HUB.retry_transient(
            lambda: api.repo_info(DATA_REPO, repo_type="dataset"), what="K3 harvest revision"
        ).sha
        remote = f"{PREFIX}/fits/{cell}/k3_refit_prompt_last.json"
        if not HUB.list_hf_files_under_path(
            api, DATA_REPO, remote, repo_type="dataset", revision=revision
        ):
            return None

        def read(name):
            path = HUB.retry_transient(
                lambda: hf_hub_download(
                    DATA_REPO,
                    name,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=str(self.args.state_dir / "hf-cache"),
                ),
                what=f"K3 harvest {name}",
            )
            return json.loads(Path(path).read_text())

        record = read(remote)
        expected = PILOT_SHA if cell == PILOT else self.args.source_sha
        if record["meta"]["git_sha"] != expected or record["cell"] != cell:
            raise ValueError(f"stale/mismatched uploaded refit: {cell}")
        n = record["refit"]["n"]
        if (
            record["gate"]["pass"] is not True
            or n["tr"] < max(record["dimension"] + 1, 8000)
            or n["val"] < 300
            or n["te"] < 900
        ):
            raise ValueError(f"refit coverage/gate failed: {cell}")
        throughput = read(f"{PREFIX}/fits/{cell}/k3_gen_throughput.json")
        if throughput["meta"]["git_sha"] != expected:
            raise ValueError(f"stale generation evidence: {cell}")
        prefix = f"issue2588_capability_panel/{cell.removesuffix('_a')}/nothink/k3_train_refit"
        sizes = dict(
            HUB.list_hf_entries_under_path(
                api, DATA_REPO, prefix, repo_type="dataset", revision=revision
            )
        )
        if any(size is None for size in sizes.values()):
            raise ValueError(f"HF tree omitted file sizes: {cell}")
        expected_stages = {f"train_10k_s{s}_part{k}": 2500 for s in (45, 46) for k in range(4)}
        expected_stages.update({f"val_400_s{s}": 400 for s in (45, 46)})
        if set(throughput["stages"]) != set(expected_stages):
            raise ValueError(f"missing generation stages: {cell}")
        for stage, count in expected_stages.items():
            if throughput["stages"][stage]["n_rows"] != count:
                raise ValueError(f"generation count mismatch: {cell}/{stage}")
            needed = [
                f"{prefix}/raw_completions/{stage}/cap_hit_report.json",
                f"{prefix}/analysis_tensors/capture/{stage}/rows.json",
                f"{prefix}/parsed/{stage}.jsonl",
            ]
            needed.extend(
                f"{prefix}/raw_completions/{stage}/chunk{k:04d}.json"
                for k in range(math.ceil(count / 500))
            )
            if any(sizes.get(path, 0) <= 0 for path in needed):
                raise ValueError(f"missing uploaded stage inputs: {cell}/{stage}")
            layer = f"{prefix}/analysis_tensors/capture/{stage}/L{record['layer_star']:02d}/"
            manifest = read(f"{prefix}/analysis_tensors/capture/{stage}/rows.json")
            required = capture_files(
                manifest, prefix=layer, cell=cell, stage=stage, source=expected
            )
            actual = {
                path
                for path, size in sizes.items()
                if path.startswith(layer) and path.endswith(".npz") and size > 0
            }
            if actual != required:
                raise ValueError(f"incomplete uploaded capture shards: {cell}/{stage}")
        save(self.args.state_dir / "refits" / cell / "k3_refit_prompt_last.json", record)
        return {"hf_revision": revision, "source_sha": expected, "n": n}

    def collect(self):
        output = ROOT / "eval_results/issue_2588/k3_train_refit/k3_train_refit_capability.json"
        with (self.args.state_dir / "collect.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/issue2588_k3_train_refit.py"),
                    "--collect",
                    "--refit-local-dir",
                    str(self.args.state_dir / "refits"),
                    "--collect-out",
                    str(output),
                ],
                cwd=ROOT,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        result = json.loads(output.read_text())
        if {row["cell"] for row in result["per_model"]} != {PILOT, *CELLS}:
            raise ValueError("collection did not contain all ten approved cells")
        # Land only this produced result. Future concurrent edits are never staged.
        relative = str(output.relative_to(ROOT))
        for argv in (
            ["git", "add", "--", relative],
            ["git", "commit", "-m", "Record task 2588 K3 A100 refit panel", "--", relative],
            ["git", "push", "origin", "HEAD:issue-2588-k3refit"],
        ):
            subprocess.run(argv, cwd=ROOT, check=True, timeout=300)
        self.state["status"] = "collected"
        self.state["result"] = str(output)
        self.persist()
        self.marker(
            f"All ten K3 refits fetched from pinned current HF revisions, with "
            f"generation/raw/capture presence and coverage verified. Collected and pushed "
            f"{relative}; state={self.state_path}. Scientific interpretation and manuscript "
            "revision remain for the interactive owner; promotion unchanged. verified-by: ran"
        )

    def run(self):
        if self.state.get("launching"):
            raise RuntimeError("interrupted launch requires reconciliation; refusing duplicate")
        if self.state["status"] == "collected":
            return
        self.state["pid"] = os.getpid()
        self.persist()
        heartbeat = 0.0
        while time.time() - self.state["started_at"] < 7 * 86400:
            if "basis" not in self.state:
                self.state["pilot_poll"] = self.poll(self.args.pilot_handle)
                if self.state["pilot_poll"]["status"] not in ("running", "pid-stale-workload-live"):
                    raise RuntimeError(f"pilot needs inspection: {self.state['pilot_poll']}")
                snapshot = self.snapshot()
                basis = pilot_basis(snapshot)
                self.state["pilot_saved_parts"] = len(snapshot["parts"])
                if basis is not None:
                    self.state["basis"] = basis
                    self.state["status"] = "dispatching"
                    self.persist()
                    self.marker(
                        "A100 original 10000-row pilot timing gate passed: "
                        + json.dumps(basis)
                        + "; nine remaining cells will run one A100-80 each, GCP credits confirmed "
                        "by user. Generation estimate includes registered retries. Generation-only "
                        "dispersion allowance is measured mean x2 x1.25; capture/fits are still "
                        "unmeasured and retain the seven-day emergency fence. 300 GB per cell; "
                        "raw checkpoints per2500 rows, upload before capture, captures and fits "
                        "uploaded before completion. No RunPod fallback. verified-by: ran"
                    )
            if "basis" in self.state:
                monitoring = {
                    PILOT: self.state.setdefault(
                        "pilot_run", {"handle": str(self.args.pilot_handle), "confirmed": True}
                    ),
                    **self.state["launches"],
                }
                for cell, launch in monitoring.items():
                    if launch.get("harvest"):
                        continue
                    launch["poll"] = self.poll(launch["handle"])
                    if launch["poll"]["status"] in ("done", "dead"):
                        launch["harvest"] = self.harvest(cell)
                        if launch["harvest"]:
                            self.persist()
                            self.marker(
                                f"K3 cell artifacts verified: cell={cell} "
                                f"evidence={json.dumps(launch['harvest'])}; verified-by: ran"
                            )
                            continue
                        launch.setdefault("terminal_missing_since", time.time())
                        if time.time() - launch["terminal_missing_since"] > 600:
                            raise RuntimeError(f"terminal cell missing refit after10min: {cell}")
                        continue
                    if launch["poll"]["status"] not in (
                        "running",
                        "done",
                        "pid-stale-workload-live",
                    ):
                        raise RuntimeError(f"cell needs inspection: {cell}: {launch['poll']}")
                    if (
                        not launch.get("confirmed")
                        and launch["poll"]["current_phase"] == "workload"
                    ):
                        self.confirm(cell, launch)
                    if not launch.get("confirmed") and time.time() - launch["launched_at"] > 900:
                        raise RuntimeError(f"workload launch not confirmed within15min: {cell}")
                if len(monitoring) == 10 and all(x.get("harvest") for x in monitoring.values()):
                    self.collect()
                    return
                self.state["status"] = "fanout_running"
                # One submission per iteration: inspect existing instances before
                # each launch, rather than waiting until all nine launches finish.
                pending = [cell for cell in CELLS if cell not in self.state["launches"]]
                if pending and self.launch(pending[0]):
                    continue
            self.persist()
            if time.time() - heartbeat >= 2700:
                self.marker(
                    "[long-phase-heartbeat] One-shot K3 VM owner is live; "
                    f"pid={os.getpid()} state={self.state_path} "
                    f"status={self.state['status']} saved_pilot_parts="
                    f"{self.state.get('pilot_saved_parts', 0)}/4 "
                    f"launched={len(self.state['launches'])}/9. "
                    "Fixed original 10000-row gate; failures require inspection, "
                    "never automatic duplicate submission or RunPod fallback. verified-by: ran"
                )
                heartbeat = time.time()
            print(
                json.dumps(
                    {
                        "ts": time.time(),
                        "status": self.state["status"],
                        "saved_pilot_parts": self.state.get("pilot_saved_parts", 0),
                        "launched": len(self.state["launches"]),
                    }
                ),
                flush=True,
            )
            time.sleep(self.args.interval)
        raise TimeoutError("one-shot owner reached its seven-day deadline")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True, help="shared main checkout for task CLI")
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--pilot-handle", type=Path, required=True)
    ap.add_argument("--source-sha", required=True)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--snapshot-only", action="store_true", help="read pilot without launching")
    args = ap.parse_args()
    if args.interval < 60 or re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
        ap.error("interval must be >=60s and source-sha must be a full commit")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "owner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = Owner(args)
        if args.snapshot_only:
            snapshot = owner.snapshot()
            print(
                json.dumps({"basis": pilot_basis(snapshot), "saved_parts": len(snapshot["parts"])})
            )
            return
        try:
            owner.run()
        except Exception as exc:
            owner.state["status"] = "needs_inspection"
            owner.state["error"] = f"{type(exc).__name__}: {exc}"
            owner.persist()
            owner.marker(
                f"K3 one-shot owner stopped: {owner.state['error']}; "
                f"state={owner.state_path}. Inspect existing handles before recovery. "
                "No compute termination or duplicate launch attempted. verified-by: ran"
            )
            raise


if __name__ == "__main__":
    main()
