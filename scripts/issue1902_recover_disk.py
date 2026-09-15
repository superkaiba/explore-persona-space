"""Read and persist failed-run diagnostics from a read-only copy of its boot disk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

import issue1902_format_common as C


def main():
    """Wait for the recovery disk, inspect it without journal replay, and upload logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/disk/by-id/google-issue1902-recovery-part1")
    args = parser.parse_args()
    deadline = time.monotonic() + 1200
    while not Path(args.device).exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Recovery disk did not appear: {args.device}")
        print("[phase=await_recovery_disk] waiting for read-only boot-disk copy", flush=True)
        time.sleep(15)
    mount = Path("/mnt/issue1902-recovery")
    mount.mkdir(exist_ok=True)
    subprocess.run(["mount", "-o", "ro,noload", args.device, str(mount)], check=True)
    root = Path("/workspace/issue1902_diagnostics")
    out = root / "recovery_20260915"
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = mount / "workspace/issue1902_olmo_onpolicy_v1"
        assert source.is_dir(), f"Expected failed-run output root absent: {source}"
        logs = sorted((source / "logs").glob("*.log"))
        assert logs and any("capture_olmo_R_1_production" in p.name for p in logs)
        paths = []
        for path in logs:
            target = out / "logs" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            paths.append(target)
            if "capture_olmo_R" in path.name and "production" in path.name:
                print(f"[phase=recovered_log] {path.name}", flush=True)
                print(path.read_text()[-18000:], flush=True)
        inventory = []
        for path in source.rglob("*"):
            if path.is_file() and "inputs" not in path.relative_to(source).parts:
                stat = path.stat()
                inventory.append(
                    dict(
                        path=str(path.relative_to(source)),
                        bytes=stat.st_size,
                        modified_at=stat.st_mtime,
                    )
                )
        report = out / "disk_inventory.json"
        C.write_json(
            report,
            dict(
                observed_at=time.time(), files=inventory, source_device=args.device, read_only=True
            ),
        )
        paths.append(report)
        revision = C.upload_many(paths, root, C.sha(__file__))
        print(f"[phase=diagnostics_verified] files={len(paths)} revision={revision}", flush=True)
    finally:
        subprocess.run(["umount", str(mount)], check=True)
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"],
        issue=1902,
        extra=dict(diagnostic_revision=revision, files=len(paths)),
    )
    print("[phase=done] Failed-run worker logs recovered and verified", flush=True)


if __name__ == "__main__":
    main()
