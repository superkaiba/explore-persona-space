"""Resume only missing OLMo RLVR captures from the preserved, verified run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# The rejected Xet commit was reproduced on real files and recovered via LFS.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import issue1902_format_job as J  # noqa: E402
import issue1902_format_common as C  # noqa: E402

ORIGINAL_SOURCE = "9d19a0435728b64fde6ad3832967b50377d2962b"
FINGERPRINT = "77e05d99afc1e96b58d18e7a16ac238f2e1178030cc3024ac3e7f2063d94e835"
REPAIRED_REVISION = "5951c687bc8a8b89d6275953463233bfa5d3e09b"


def restore(root, device, deadline):
    """Copy only required scientific inputs from an immutable, read-only disk copy."""
    while not Path(device).exists():
        if time.time() > deadline:
            raise TimeoutError(f"Recovery disk did not appear: {device}")
        print("[phase=await_recovery_disk] waiting for preserved inputs", flush=True)
        time.sleep(15)
    mount = Path("/mnt/issue1902-resume-source")
    mount.mkdir(exist_ok=True)
    subprocess.run(["mount", "-o", "ro,noload", device, str(mount)], check=True)
    try:
        source = mount / "workspace/issue1902_olmo_onpolicy_v1"
        assert source.is_dir()
        names = [
            "manifest.json",
            "runtime.json",
            "pilot_complete.json",
            "references",
            "banks",
            "contexts",
            "captures",
            "audits",
        ]
        files = []
        for name in names:
            path = source / name
            assert path.exists(), path
            files.extend([path] if path.is_file() else (p for p in path.rglob("*") if p.is_file()))
        total = sum(p.stat().st_size for p in files)
        assert total < 60 * 1024**3, total
        root.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(root).free > total + 15 * 1024**3
        print(f"[phase=restore] files={len(files)} bytes={total}", flush=True)
        for i, path in enumerate(files):
            target = root / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert C.sha(target) == C.sha(path), f"Existing restore file differs: {target}"
            else:
                shutil.copyfile(path, target)
            if i % 100 == 0:
                print(f"[phase=restore] files={i + 1}/{len(files)}", flush=True)
    finally:
        subprocess.run(["umount", str(mount)], check=True)
    assert C.fingerprint(root) == FINGERPRINT, "Scientific implementation or manifest changed"
    # Restore the receipts for the exact eight files recovered after the snapshot.
    for part in ("contexts", "captures"):
        for path in (root / part).rglob("*.npz"):
            if path.with_suffix(".npz.done.json").exists():
                continue
            for item in (path, path.with_suffix(".timing.json")):
                remote = f"{C.PREFIX}/{item.relative_to(root)}.done.json"
                receipt = json.loads(C.fetch(root, remote, REPAIRED_REVISION).read_text())
                assert receipt["fingerprint"] == FINGERPRINT
                assert receipt["sha256"] == C.sha(item)
                assert receipt["bytes"] == item.stat().st_size
                C.write_json(item.with_suffix(item.suffix + ".done.json"), receipt)


def remaining_work(root):
    """Validate every retained checkpoint and bound remaining work from measured pilots."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["olmo"]["ids"]) == 16391
    missing, seconds = [], 0.0
    for model in C.MODELS:
        for offset in range(0, 16391, C.CHUNK):
            for form in ("plain", "chat"):
                raw = root / "banks" / model / form / f"chunk_{offset:05d}.json"
                assert C.complete(raw, FINGERPRINT), raw
                C.read_raw(raw)
                for rel in [f"contexts/{model}/{form}", f"captures/{model}/{form}/{form}"]:
                    path = root / rel / f"chunk_{offset:05d}.npz"
                    if C.complete(path, FINGERPRINT):
                        continue
                    assert model == "olmo_R", f"Unexpected missing completed model: {path}"
                    missing.append(str(path.relative_to(root)))
                    pilot = json.loads((root / rel / "chunk_00000.timing.json").read_text())
                    seconds += pilot["seconds"]
    return dict(missing=missing, measured_gpu_seconds=seconds)


def capture_children(root, devices, deadline, *, log_root=Path("/workspace/logs")):
    """Run independent RLVR shards, preserving the initiating error and every worker log."""
    children = []
    try:
        for shard, device in enumerate(devices):
            log = log_root / f"issue1902_recovery_capture_s{shard}.log"
            with log.open("w") as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(C.ROOT / "scripts/issue1902_format_job.py"),
                        "--phase",
                        "capture",
                        "--root",
                        str(root),
                        "--model",
                        "olmo_R",
                        "--shard",
                        str(shard),
                        "--shards",
                        str(len(devices)),
                    ],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children.append((child, log))
        while True:
            states = [(p, log, p.poll()) for p, log in children]
            failed = [(p, log, rc) for p, log, rc in states if rc is not None and rc != 0]
            if failed:
                p, log, rc = failed[0]
                print(log.read_text()[-24000:], flush=True)
                raise RuntimeError(f"Initiating worker failure pid={p.pid} rc={rc} log={log}")
            if all(rc == 0 for _, _, rc in states):
                return [log for _, log in children]
            if time.time() > deadline:
                raise TimeoutError("Recovery reached its absolute deadline")
            progress = []
            for p, log, rc in states:
                age = time.time() - log.stat().st_mtime
                if rc is None and age > 900:
                    raise RuntimeError(f"RLVR worker stalled for {age:.0f}s: pid={p.pid} log={log}")
                lines = [s for s in log.read_text().splitlines() if s.startswith("[phase=")]
                progress.append(
                    dict(
                        pid=p.pid,
                        rc=rc,
                        log_age=age,
                        latest=lines[-1] if lines else "model startup",
                    )
                )
            print(
                "[phase=recovery_capture] " + json.dumps(dict(time=time.time(), workers=progress)),
                flush=True,
            )
            time.sleep(15)
    finally:
        for child, _ in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    print(f"[phase=cleanup] owned worker {child.pid} already exited", flush=True)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        print(
                            f"[phase=cleanup] owned worker {child.pid} exited during grace",
                            flush=True,
                        )
                    child.wait(timeout=15)


def main():
    """Reuse original scientific code, validate recovery, then publish fresh GPU completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--device", default="/dev/disk/by-id/google-issue1902-recovery-part1")
    parser.add_argument("--root", type=Path, default=Path("/workspace/issue1902_olmo_onpolicy_v1"))
    args = parser.parse_args()
    J.verify_source(args.source_sha)
    from huggingface_hub.utils._runtime import is_xet_available

    assert not is_xet_available(), "Recovery must use the tested LFS upload route"
    assert importlib.metadata.version("torch").split("+")[0] == "2.8.0"
    assert importlib.metadata.version("transformers") == "4.57.6"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--planned-footprint-gb",
            "60",
            "--min-disk",
            "90",
        ],
        check=True,
    )
    restore(args.root, args.device, min(args.deadline_unix, time.time() + 1200))
    work = remaining_work(args.root)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if devices == [""]:
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
    assert devices
    required = work["measured_gpu_seconds"] / len(devices) * 2.5 + 1800
    assert time.time() + required < args.deadline_unix, "Insufficient remaining recovery fence"
    C.write_json(
        args.root / "recovery_started.json",
        dict(
            source_sha=args.source_sha,
            original_source=ORIGINAL_SOURCE,
            fingerprint=FINGERPRINT,
            time=time.time(),
            deadline=args.deadline_unix,
            devices=devices,
            **work,
        ),
    )
    print(
        f"[phase=recovery_scope] missing={len(work['missing'])} projected_gpu_seconds="
        f"{work['measured_gpu_seconds']:.1f}; generation skipped; HF_HUB_DISABLE_XET=1",
        flush=True,
    )
    C.storage_probe(args.root / "recovery_storage_check")
    logs = capture_children(args.root, devices, args.deadline_unix)
    revision, coverage = J.aggregate(args.root)
    report = dict(
        status="gpu_collection_complete",
        input_revision=revision,
        hf_prefix=C.PREFIX,
        source_sha=args.source_sha,
        original_source=ORIGINAL_SOURCE,
        coverage=coverage,
        fingerprint=FINGERPRINT,
        cpu_fits_pending=True,
        followup_label="OLMo-eight-setting-on-policy-posttraining",
    )
    C.write_json(args.root / "gpu_complete.json", report)
    saved_logs = []
    for path in logs:
        target = args.root / "recovery_logs" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        saved_logs.append(target)
    C.upload_many(
        [
            args.root / "gpu_complete.json",
            args.root / "recovery_started.json",
            args.root / "runtime.json",
            *saved_logs,
        ],
        args.root,
        FINGERPRINT,
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1902, extra=report
    )
    print("[phase=done] All eight OLMo target settings verified; CPU fits follow", flush=True)


if __name__ == "__main__":
    main()
