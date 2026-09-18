"""Read-only, source-pinned monitoring for the three #1739 natural runs.

Run ``uv run python scripts/issue1739_natural_monitor.py CONFIG [--once]``.
Register its observation with experiment_watchdog.py; this program never starts,
restarts, or terminates compute. An unhealthy tick is persisted and exits 2.

CONFIG contains source_sha, observation, state_path, and handles. Each handle
entry contains behavior, handle_path, out_root, upload_prefix, input_fingerprint,
and launched_at (Unix seconds). Optional settings: poll_seconds=60,
startup_timeout=600, progress_timeout=1200, phase_timeouts={"map_fit": 1200},
data_repo="superkaiba1/explore-persona-space-data". Paths are absolute. A changed
config, monitor source, or backend sidecar requires a new state_path and watchdog
registration. Watchdog stale_seconds bounds monitor-observation age; this monitor
separately enforces phase-specific output/counter timeouts. Measured raw log age
is diagnostic and may legitimately exceed the observation timeout during a fit.

The runner writes worker.pid as {source_sha,pid,start_ticks}, progress.json,
staging_progress.json, run_config.json, upload_verified.json and run_complete.json
under out_root. Completion independently downloads results.json at the verified
HF revision and checks its digest, source, behavior, and input fingerprint.
The worker also publishes <upload_prefix>.completion.json after verification;
this durable receipt is checked before probing a potentially stopped backend.
Importing this module performs no probes. --import-check performs imports only.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_watchdog import write_json  # noqa: E402

BEHAVIORS = {"sycophancy", "hallucination", "evil"}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DOCUMENTS = (
    "run_config.json",
    "progress.json",
    "staging_progress.json",
    "worker.pid",
    "upload_verified.json",
    "run_complete.json",
)


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_config(path):
    config = json.loads(Path(path).read_text())
    require(bool(HEX40.fullmatch(config["source_sha"])), "source_sha must be a full commit")
    for key in ("observation", "state_path"):
        require(Path(config[key]).is_absolute(), f"{key} must be absolute")
    require(
        config["observation"] != config["state_path"], "observation must differ from state_path"
    )
    handles = config["handles"]
    require(
        len(handles) == 3 and {h["behavior"] for h in handles} == BEHAVIORS,
        "exactly one handle per planned behavior is required",
    )
    for item in handles:
        for key in ("handle_path", "out_root"):
            require(Path(item[key]).is_absolute(), f"{key} must be absolute")
        require(bool(HEX64.fullmatch(item["input_fingerprint"])), "invalid input fingerprint")
        require(
            item["upload_prefix"]
            and ".." not in Path(item["upload_prefix"]).parts
            and not item["upload_prefix"].startswith("/"),
            "invalid upload prefix",
        )
        require(
            isinstance(item["launched_at"], (int, float)) and item["launched_at"] > 0,
            "launched_at must be the real launch time",
        )
    config.setdefault("poll_seconds", 60)
    config.setdefault("startup_timeout", 600)
    config.setdefault("progress_timeout", 1200)
    config.setdefault("phase_timeouts", {})
    config.setdefault("data_repo", "superkaiba1/explore-persona-space-data")
    require(10 <= config["poll_seconds"] <= 60, "poll_seconds must be 10..60")
    for limit in [
        config["startup_timeout"],
        config["progress_timeout"],
        *config["phase_timeouts"].values(),
    ]:
        require(60 <= limit <= 7200, "timeout must be 60..7200 seconds")
    return config


def command_output(argv, *, timeout=45):
    result = subprocess.run(
        argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False
    )
    if result.returncode:
        # Never copy credential-bearing subprocess output into an observation.
        raise RuntimeError(f"{Path(argv[0]).name} probe returned {result.returncode}")
    return result.stdout


def remote_command(item, log_path):
    """Fixed read-only Python probe, with data passed without shell interpolation."""
    descriptor = base64.b64encode(
        json.dumps({"root": item["out_root"], "log": log_path, "documents": DOCUMENTS}).encode()
    ).decode()
    script = r"""
import base64,json,os,time
from pathlib import Path
c=json.loads(base64.b64decode(DESCRIPTOR))
root=Path(c["root"])
out={"remote_now":time.time(),"documents":{},"document_mtimes":{}}
for name in c["documents"]:
    p=root/name
    if not p.exists():
        continue
    st=p.stat()
    if st.st_size>8*1024**2:
        raise ValueError("monitor document exceeds 8MiB")
    value=json.loads(p.read_text())
    if not isinstance(value,dict):
        raise ValueError("monitor document is not an object")
    out["documents"][name]=value
    out["document_mtimes"][name]=st.st_mtime
p=Path(c["log"])
out["log_mtime"]=p.stat().st_mtime if p.exists() else None
out["log_bytes"]=p.stat().st_size if p.exists() else None
files=[]
if root.exists():
    for p in root.rglob("*"):
        if p.is_file() and p.name not in c["documents"] and not p.name.endswith(".partial"):
            st=p.stat()
            files.append((p.relative_to(root).as_posix(),st.st_size,st.st_mtime))
            if len(files)>100000:
                raise ValueError("output scan exceeds 100000 files")
out["output_files"]=len(files)
out["output_bytes"]=sum(x[1] for x in files)
out["output_latest_mtime"]=max((x[2] for x in files),default=None)
pid_record=out["documents"].get("worker.pid")
out["process_alive"]=None
if pid_record is not None:
    pid=int(pid_record["pid"])
    if pid<=1:
        raise ValueError("invalid worker pid")
    proc=Path("/proc")/str(pid)
    try:
        stat=(proc/"stat").read_text().rsplit(")",1)[1].split()
        cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        out["process_alive"]=stat[0] not in ("Z","X")
        out["process_start_ticks"]=int(stat[19])
        out["process_command_matches"]="issue1739_natural_run.py" in cmd
        out["pid"]=pid
    except FileNotFoundError:
        out["process_alive"]=False
print(json.dumps(out,allow_nan=False))
""".replace("DESCRIPTOR", repr(descriptor))
    return "python3 -c " + shlex.quote(script)


class Runtime:
    """Bounded probes through repository backend transports; no lifecycle calls."""

    def published_completion(self, repo, prefix):
        """Resolve the small completion sidecar at an immutable HF revision."""
        code = r"""
import json,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
repo,prefix=sys.argv[1:]
revision=HfApi().repo_info(repo,repo_type="dataset").sha
try:
    p=Path(hf_hub_download(repo_id=repo,filename=prefix+".completion.json",
                          revision=revision,repo_type="dataset"))
except EntryNotFoundError:
    print("null")
else:
    if p.stat().st_size>8*1024**2:
        raise ValueError("completion receipt exceeds 8MiB")
    print(json.dumps({"revision":revision,"receipt":json.loads(p.read_bytes())},allow_nan=False))
"""
        raw = command_output([sys.executable, "-c", code, repo, prefix])
        return json.loads(raw)

    def probe(self, item, source_sha):
        from explore_persona_space.backends.issue_dispatch import read_handle_sidecar

        handle_path = Path(item["handle_path"])
        handle_digest = hashlib.sha256(handle_path.read_bytes()).hexdigest()
        handle = read_handle_sidecar(handle_path)
        require(handle.extra.get("source_sha", source_sha) == source_sha, "handle source mismatch")
        command = remote_command(item, handle.log_path)
        if handle.backend == "gcp":
            from explore_persona_space.backends.gcp import (
                GcloudRunResult,
                _base_gcloud_argv,
                classify_fetch_transport,
                default_gcp_config,
                render_describe_argv,
            )

            cfg = default_gcp_config()
            require(
                handle.extra.get("project", cfg.project) == cfg.project,
                "GCP handle project differs from configured backend",
            )
            require(
                handle.extra.get("gcloud_config", cfg.gcloud_config) == cfg.gcloud_config,
                "GCP handle configuration differs from configured backend",
            )
            zone = handle.extra.get("zone") or cfg.primary_zone
            raw = command_output(render_describe_argv(config=cfg, name=handle.pod_name, zone=zone))
            live = json.loads(raw)
            require(
                str(live["id"]) == handle.job_id and live["name"] == handle.pod_name,
                "GCP handle incarnation mismatch",
            )
            status = live["status"]
            if status != "RUNNING":
                return {"backend_status": status, "handle_sha256": handle_digest}
            transport, _ = classify_fetch_transport(GcloudRunResult(0, raw, ""))
            require(transport in {"external-ip", "iap"}, "GCP SSH transport unavailable")
            argv = _base_gcloud_argv(cfg, "compute", "ssh", handle.pod_name)
            argv += [f"--zone={zone}", "--quiet", "--command=sudo -n -- " + command]
            if transport == "iap":
                argv.append("--tunnel-through-iap")
            try:
                remote = command_output(argv)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                return {
                    "backend_status": status,
                    "handle_sha256": handle_digest,
                    "ssh_error": type(exc).__name__,
                }
        elif handle.backend == "runpod":
            from explore_persona_space.backends.runpod import _ssh_pod_run

            # Existing API retries may take minutes; a child provides a hard cap
            # without leaving an unbounded background thread after a failed tick.
            code = (
                "import json,sys; from dataclasses import asdict; "
                "from scripts.runpod_api import get_pod; "
                "print(json.dumps(asdict(get_pod(sys.argv[1]))))"
            )
            live = json.loads(command_output([sys.executable, "-c", code, handle.job_id]))
            require(
                live["pod_id"] == handle.job_id and live["name"] == handle.pod_name,
                "RunPod handle incarnation mismatch",
            )
            status = live["desired_status"]
            if status != "RUNNING":
                return {"backend_status": status, "handle_sha256": handle_digest}
            if not live["ssh_host"] or not live["ssh_port"]:
                return {
                    "backend_status": status,
                    "handle_sha256": handle_digest,
                    "ssh_error": "NoLiveEndpoint",
                }
            try:
                remote = _ssh_pod_run(
                    live["ssh_host"],
                    int(live["ssh_port"]),
                    command,
                    timeout=45,
                    context="natural-monitor read-only probe",
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                return {
                    "backend_status": status,
                    "handle_sha256": handle_digest,
                    "ssh_error": type(exc).__name__,
                }
        else:
            raise ValueError(f"unsupported backend {handle.backend}")
        result = json.loads(remote)
        result.update(backend_status=status, backend=handle.backend, handle_sha256=handle_digest)
        return result

    def receipt(self, repo, prefix, revision):
        # Only the small JSON result receipt is downloaded, never tensor outputs.
        code = (
            "import sys; from pathlib import Path; from huggingface_hub import hf_hub_download; "
            "p=Path(hf_hub_download(repo_id=sys.argv[1],filename=sys.argv[2],"
            "revision=sys.argv[3],repo_type='dataset')); "
            "assert p.stat().st_size <= 8*1024**2; sys.stdout.buffer.write(p.read_bytes())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, repo, prefix + "/results.json", revision],
            cwd=ROOT,
            capture_output=True,
            timeout=45,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"HF receipt verification returned {result.returncode}")
        return result.stdout


def verify_completion(config, item, docs, runtime):
    done = docs.get("run_complete.json")
    if done is None:
        return None
    upload = docs.get("upload_verified.json", {})
    revision = done.get("verified_revision", "")
    require(done.get("status") == "complete" and HEX40.fullmatch(revision), "invalid completion")
    for name, document in (("completion", done), ("upload verification", upload)):
        require(document.get("source_sha") == config["source_sha"], f"{name} source mismatch")
        timestamp = document.get("time")
        require(
            isinstance(timestamp, (int, float))
            and item["launched_at"] <= timestamp <= time.time() + 60,
            f"{name} predates this launch or is future-dated",
        )
    require(
        upload.get("phase") == "complete" and upload.get("verified_revision") == revision,
        "completion lacks matching final upload verification",
    )
    require(upload.get("prefix") == item["upload_prefix"], "upload prefix mismatch")
    require(done.get("input_fingerprint") == item["input_fingerprint"], "completion input mismatch")
    expected_hash = upload.get("sha256", {}).get("results.json", "")
    require(bool(HEX64.fullmatch(expected_hash)), "uploaded result digest missing")
    blob = runtime.receipt(config["data_repo"], item["upload_prefix"], revision)
    require(hashlib.sha256(blob).hexdigest() == expected_hash, "remote receipt digest mismatch")
    receipt = json.loads(blob)
    for key, expected in (
        ("source_sha", config["source_sha"]),
        ("behavior", item["behavior"]),
        ("input_fingerprint", item["input_fingerprint"]),
    ):
        require(receipt.get(key) == expected, f"remote receipt {key} mismatch")
    require(
        isinstance(receipt.get("finished_at"), (int, float))
        and item["launched_at"] <= receipt["finished_at"] <= time.time() + 60,
        "remote results predate this launch or are future-dated",
    )
    require(
        isinstance(receipt.get("datasets"), list) and receipt["datasets"],
        "remote receipt has no dataset results",
    )
    return {
        "source_sha": config["source_sha"],
        "verified_revision": revision,
        "prefix": item["upload_prefix"],
        "receipt_sha256": expected_hash,
        "input_fingerprint": item["input_fingerprint"],
        "verified_at": time.time(),
    }


def verify_published_completion(config, item, published, runtime):
    """Validate the durable receipt without requiring a surviving worker host."""
    if published is None:
        return None
    require(bool(HEX40.fullmatch(published["revision"])), "invalid receipt publication revision")
    receipt = published["receipt"]
    require(receipt.get("behavior") == item["behavior"], "completion receipt behavior mismatch")
    result = verify_completion(
        config,
        item,
        {"run_complete.json": receipt, "upload_verified.json": receipt["verification"]},
        runtime,
    )
    result["receipt_publication_revision"] = published["revision"]
    return result


def evaluate(config, item, probe, previous, runtime, now):
    """Derive liveness from output/counter changes, never heartbeat timestamps."""
    prior_handle = previous.get("handle_sha256")
    require(prior_handle in (None, probe["handle_sha256"]), "backend sidecar changed")
    state = {**previous, "handle_sha256": probe["handle_sha256"]}
    started = item["launched_at"]
    require(started <= now + 60, "launch time is future-dated")
    backend = probe["backend_status"]
    if backend in {"PROVISIONING", "STAGING", "PENDING", "CREATING"}:
        require(now - started <= config["startup_timeout"], "backend startup timeout")
        return {"status": "pending", "process_alive": None, "backend_status": backend}, state
    require(backend == "RUNNING", f"backend is {backend}")
    if probe.get("ssh_error"):
        require(
            not previous.get("progress_signature") and now - started <= config["startup_timeout"],
            f"SSH probe unavailable: {probe['ssh_error']}",
        )
        return {
            "status": "pending",
            "process_alive": None,
            "backend_status": backend,
            "startup_ssh": probe["ssh_error"],
        }, state
    require(abs(probe["remote_now"] - now) <= 120, "remote clock skew exceeds 120s")
    docs = probe["documents"]
    for name, doc in docs.items():
        require(doc.get("source_sha") == config["source_sha"], f"{name} source mismatch")
    run_config = docs.get("run_config.json")
    if run_config:
        require(
            run_config.get("input_fingerprint") == item["input_fingerprint"],
            "remote input configuration mismatch",
        )
        require(run_config.get("behavior") == item["behavior"], "remote behavior mismatch")
    completed = verify_completion(config, item, docs, runtime)
    if completed:
        state["results"] = completed
        return {
            "status": "complete",
            "results": completed,
            "process_alive": probe["process_alive"],
        }, state
    pid_record = docs.get("worker.pid")
    if pid_record is None:
        require(now - started <= config["startup_timeout"], "worker never published its PID")
        return {"status": "pending", "process_alive": None, "backend_status": backend}, state
    require(probe["process_alive"] is True, "worker exited without verified completion")
    require(probe.get("process_command_matches"), "worker PID belongs to a different command")
    require(int(pid_record["start_ticks"]) == probe["process_start_ticks"], "worker PID was reused")
    candidates = [
        (probe["document_mtimes"][name], name, docs[name])
        for name in ("progress.json", "staging_progress.json")
        if name in docs
    ]
    require(
        candidates or now - started <= config["startup_timeout"], "worker has no progress record"
    )
    _, progress_name, progress = max(candidates) if candidates else (started, None, {})
    phase = progress.get("phase", "startup")
    fields = {k: v for k, v in progress.items() if k not in {"time", "checked_at", "max_rss_gib"}}
    signature = identity([fields, probe["output_files"], probe["output_bytes"]])
    latest_time = max(
        [started, *(x[0] for x in candidates), probe["output_latest_mtime"] or started]
    )
    require(latest_time <= now + 60, "progress is future-dated")
    last_advance = previous.get("last_advance", min(latest_time, now))
    if previous.get("progress_signature") not in (None, signature):
        last_advance = now
    timeout = config["phase_timeouts"].get(phase, config["progress_timeout"])
    require(
        now - last_advance <= timeout,
        f"{phase} output/counters unchanged for {now - last_advance:.0f}s",
    )
    state.update(progress_signature=signature, last_advance=last_advance)
    log_age = max(0, now - probe["log_mtime"]) if probe["log_mtime"] is not None else None
    return {
        "status": "running",
        "backend_status": backend,
        "phase": phase,
        "process_alive": True,
        "log_age_seconds": log_age,
        "progress_age_seconds": now - last_advance,
        "progress": {
            "document": progress_name,
            "counters": fields,
            "output_files": probe["output_files"],
            "output_bytes": probe["output_bytes"],
        },
        "alarms": [],
    }, state


def tick(config, state, runtime=None):
    runtime = runtime or Runtime()
    now = time.time()
    fingerprint = identity(
        {
            "config": config,
            "monitor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    observation = {
        "source_sha": config["source_sha"],
        "checked_at": now,
        "status": "running",
        "config_fingerprint": fingerprint,
    }
    try:
        require(
            state.get("config_fingerprint") in (None, fingerprint), "monitor configuration changed"
        )
        state["config_fingerprint"] = fingerprint
        state.setdefault("handles", {})
        details, next_states, snapshots = {}, {}, {}

        def one(item):
            previous = state["handles"].get(item["behavior"], {})
            handle_digest = hashlib.sha256(Path(item["handle_path"]).read_bytes()).hexdigest()
            require(
                previous.get("handle_sha256") in (None, handle_digest), "backend sidecar changed"
            )
            # A previously independently verified immutable revision stays valid.
            if previous.get("results"):
                require(
                    hashlib.sha256(Path(item["handle_path"]).read_bytes()).hexdigest()
                    == previous["handle_sha256"],
                    "completed backend sidecar changed",
                )
                return {"status": "complete", "results": previous["results"]}, previous
            published = runtime.published_completion(config["data_repo"], item["upload_prefix"])
            completed = verify_published_completion(config, item, published, runtime)
            if completed is not None:
                return {"status": "complete", "results": completed, "process_alive": None}, {
                    **previous,
                    "handle_sha256": handle_digest,
                    "results": completed,
                }
            snapshot = runtime.probe(item, config["source_sha"])
            snapshots[item["behavior"]] = snapshot
            return evaluate(
                config,
                item,
                snapshot,
                previous,
                runtime,
                time.time(),
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [(item, pool.submit(one, item)) for item in config["handles"]]
            for item, future in futures:
                name = item["behavior"]
                try:
                    details[name], next_states[name] = future.result()
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    subprocess.SubprocessError,
                ) as exc:
                    measured = snapshots.get(name, {})
                    details[name] = {
                        "status": "failed",
                        "backend_status": measured.get("backend_status", "unknown"),
                        "process_alive": measured.get("process_alive"),
                        "remote_checked_at": measured.get("remote_now"),
                        "log_age_seconds": max(0, time.time() - measured["log_mtime"])
                        if measured.get("log_mtime") is not None
                        else None,
                        "alarms": [type(exc).__name__],
                        "reason": str(exc)
                        if isinstance(exc, (ValueError, KeyError))
                        else f"{type(exc).__name__}: read-only probe failed",
                    }
        state["handles"].update(next_states)
        failed = [name for name, row in details.items() if row["status"] == "failed"]
        active = [row for row in details.values() if row["status"] != "complete"]
        backend_status = (
            "failed"
            if failed
            else "running"
            if any(row["status"] == "running" for row in active)
            else "pending"
            if active
            else "done"
        )
        observation["backend_observation"] = {
            "status": backend_status,
            "handles": details,
            "alarms": failed,
            "process_alive": all(
                row.get("process_alive") is True for row in active if row["status"] == "running"
            )
            if active and not failed and backend_status == "running"
            else None,
            "log_age_seconds": max((row.get("log_age_seconds") or 0 for row in active), default=0),
        }
        backend = observation["backend_observation"]
        backend["pid_alive"] = backend["process_alive"]
        # The generic watchdog's optional legacy log-age alias uses its monitor
        # observation timeout as a log timeout. Omit that alias: a valid quiet
        # fit may exceed it. Keep measured log_age_seconds above, while evaluate
        # enforces actual output/counter stalls with the declared phase timeout.
        backend["stall_policy"] = "phase-specific output/counter timeout enforced by monitor"
        if failed:
            observation["status"] = "backend_observation_error"
            backend["reachability_alarm"] = True
        elif not active:
            observation["status"] = "complete"
            observation["results"] = {
                "source_sha": config["source_sha"],
                "verified_revision": {
                    name: row["results"]["verified_revision"] for name, row in details.items()
                },
                "handles": {name: row["results"] for name, row in details.items()},
            }
        else:
            observation["status"] = backend_status
    except (OSError, ValueError, KeyError, TypeError) as exc:
        observation.update(
            status="monitor_failed",
            error=f"{type(exc).__name__}: {exc}",
            backend_observation={"status": "failed", "alarms": ["monitor_failed"]},
        )
    observation["checked_at"] = time.time()
    write_json(config["state_path"], state)
    write_json(config["observation"], observation)
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    args = parser.parse_args()
    if args.import_check:
        import huggingface_hub  # noqa: F401

        from explore_persona_space.backends import gcp, issue_dispatch, runpod  # noqa: F401
        from scripts import runpod_api  # noqa: F401

        print("natural monitor imports OK; no probes performed")
        return 0
    if args.config is None:
        parser.error("config is required")
    config = load_config(args.config)
    path = Path(config["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(path.read_text()) if path.exists() else {}
        while True:
            started = time.monotonic()
            observed = tick(config, state)
            transition = identity(
                {
                    "status": observed["status"],
                    "handles": {
                        k: (v["status"], v.get("phase"), v.get("reason"))
                        for k, v in observed["backend_observation"].get("handles", {}).items()
                    },
                }
            )
            if transition != state.get("last_transition"):
                print(
                    json.dumps(
                        {
                            "time": observed["checked_at"],
                            "status": observed["status"],
                            "handles": {
                                k: v["status"]
                                for k, v in observed["backend_observation"]
                                .get("handles", {})
                                .items()
                            },
                        }
                    ),
                    flush=True,
                )
                state["last_transition"] = transition
                write_json(path, state)
            if observed["status"] == "complete":
                return 0
            if observed["backend_observation"]["status"] == "failed":
                return 2
            if args.once:
                return 0
            time.sleep(max(0, config["poll_seconds"] - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
