#!/usr/bin/env python3
"""Issue #623 — on-crash forensic dump uploader (GCP-lane diagnosability).

The GCP lane provisions an EPHEMERAL instance with
``--instance-termination-action=DELETE``; on a non-zero workload exit the
instance (and its disk) is destroyed BEFORE any log upload happens, so a crash
leaves NO direct traceback — serial output is gone, there is no WandB run for
the extract phases, and HF/git carry zero issue623 artifacts. (Incident: the
round-3 run crashed during ``vector_extract`` and left zero diagnostic surface.)

This uploader is called from ``scripts/issue623_dispatch.sh``'s ``trap ... EXIT``
on ANY non-clean exit. It pushes a forensic bundle to the HF data repo under
``_crash_dumps/issue623_<ts>_<phase>/`` so the NEXT relaunch can be diagnosed
from the actual Python traceback + partial outputs.

Two-tier upload (resilient by design):

  1. **Logs + CRASH_META.json** — text/JSON only, so they ride the regular
     (non-LFS) git-blob path and land EVEN over the account-wide HF public
     storage quota (the LFS 403 fires only on the LFS endpoint;
     ``.claude/rules/upload-policy.md``). This is the load-bearing tier: the
     per-phase logs + master log carry the Python traceback. Fail-loud within
     this tier — if the logs themselves don't upload, exit non-zero so the
     trap surfaces it (without masking the ORIGINAL exit code; see the bash
     trap).
  2. **Partial $DATA_DIR tree** — may contain ``.pt`` centroids (LFS). Uploaded
     best-effort per top-level entry; any failure (e.g. a storage 403) is
     caught + logged and never aborts the dump. The logs are what matter.

Usage (from the dispatcher trap, NOT a normal phase):
  uv run python scripts/issue623_crash_dump.py \
      --run-logs /workspace/logs/issue623_driver \
      --data-dir data/persona_vectors/issue623 \
      --master-log /workspace/logs/issue-623.log \
      --last-phase vector_extract --exit-code 1 \
      [--hf-prefix-suffix smoketest]   # smoke: append _smoketest to the dir name
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Module-top imports (gotchas.md "lazy imports in smoke-skipped branches"): if a
# symbol drifts, this fails at process start — and the trap path is the ONE path
# that must never silently no-op.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload  # noqa: E402

CRASH_DUMP_ROOT = "_crash_dumps"


def _copy_logs(staging: Path, run_logs: Path, master_log: Path | None) -> int:
    """Copy every *.log under run_logs + the master log into staging/. Returns count."""
    n = 0
    logs_dest = staging / "phase_logs"
    logs_dest.mkdir(parents=True, exist_ok=True)
    if run_logs.is_dir():
        for log in sorted(run_logs.glob("*.log")):
            try:
                shutil.copy2(log, logs_dest / log.name)
                n += 1
            except OSError as e:
                print(f"[crash-dump] WARN could not copy phase log {log}: {e}", flush=True)
    if master_log is not None and master_log.is_file():
        try:
            # Rename to a stable name so it is unambiguous in the dump.
            shutil.copy2(master_log, staging / "master_issue-623.log")
            n += 1
        except OSError as e:
            print(f"[crash-dump] WARN could not copy master log {master_log}: {e}", flush=True)
    return n


def _write_meta(staging: Path, *, last_phase: str, exit_code: int, started_at: str | None) -> None:
    """Write CRASH_META.json — the forensic header the next relaunch reads first."""
    meta = {
        "issue": 623,
        "last_phase": last_phase,
        "exit_code": exit_code,
        "started_at": started_at,
        "crashed_at": datetime.now(UTC).isoformat(),
        "instance_name": os.environ.get("EPS_INSTANCE_NAME") or os.environ.get("HOSTNAME"),
        "machine_type": os.environ.get("EPS_MACHINE_TYPE"),
        "zone": os.environ.get("EPS_ZONE"),
        "dispatcher_argv": os.environ.get("EPS_DISPATCHER_ARGV"),
        "master_log_path": os.environ.get("EPS_LOG_PATH"),
    }
    (staging / "CRASH_META.json").write_text(json.dumps(meta, indent=2))


def _upload_logs_tier(staging: Path, prefix: str) -> None:
    """Upload the logs+meta staging dir. Fail-loud (raise) — this tier is load-bearing."""
    dest = _upload(
        local_path=staging,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
    )
    if not dest:
        raise RuntimeError(f"crash-dump logs upload FAILED (empty return): {staging} -> {prefix}")
    print(f"[crash-dump] logs+meta -> {dest}", flush=True)


def _upload_partial_data_tier(data_dir: Path, prefix: str) -> None:
    """Best-effort upload of the partial $DATA_DIR tree. Never raises — logs are what matter."""
    if not data_dir.is_dir():
        print(f"[crash-dump] no partial data dir at {data_dir}; skipping data tier", flush=True)
        return
    entries = sorted(p for p in data_dir.iterdir())
    if not entries:
        print(f"[crash-dump] partial data dir {data_dir} is empty; skipping data tier", flush=True)
        return
    for entry in entries:
        try:
            dest = _upload(
                local_path=entry,
                repo_id=DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=f"{prefix}/partial_data/{entry.name}",
                upload_as_file=entry.is_file(),
            )
            if dest:
                print(f"[crash-dump] partial data {entry.name} -> {dest}", flush=True)
            else:
                print(
                    f"[crash-dump] WARN partial data {entry.name} upload returned empty", flush=True
                )
        except Exception as e:
            print(f"[crash-dump] WARN partial data {entry.name} upload failed: {e}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #623 on-crash forensic dump uploader.")
    parser.add_argument("--run-logs", required=True, help="Per-phase log dir ($RUN_LOGS).")
    parser.add_argument("--data-dir", required=True, help="Partial output tree ($DATA_DIR).")
    parser.add_argument("--master-log", default=None, help="Master GCE log ($EPS_LOG_PATH).")
    parser.add_argument("--last-phase", default="unknown", help="Last [phase=...] reached.")
    parser.add_argument("--exit-code", type=int, default=1, help="Original dispatcher exit code.")
    parser.add_argument(
        "--hf-prefix-suffix",
        default=None,
        help="Optional suffix appended to the dump dir name (smoke passes 'smoketest').",
    )
    parser.add_argument(
        "--print-prefix-only",
        action="store_true",
        help="Print the chosen HF prefix to stdout and exit (lets the smoke verify/clean it).",
    )
    args = parser.parse_args()

    load_dotenv()

    ts = int(time.time())
    phase_slug = "".join(c if c.isalnum() else "_" for c in args.last_phase)[:40] or "unknown"
    name = f"issue623_{ts}_{phase_slug}"
    if args.hf_prefix_suffix:
        suffix = "".join(c if c.isalnum() else "_" for c in args.hf_prefix_suffix)
        name = f"{name}_{suffix}"
    prefix = f"{CRASH_DUMP_ROOT}/{name}"

    run_logs = Path(args.run_logs)
    data_dir = (
        Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    )
    master_log = Path(args.master_log) if args.master_log else None

    staging = Path(tempfile.mkdtemp(prefix="issue623_crash_"))
    try:
        n_logs = _copy_logs(staging, run_logs, master_log)
        _write_meta(
            staging,
            last_phase=args.last_phase,
            exit_code=args.exit_code,
            started_at=os.environ.get("EPS_STARTED_AT"),
        )
        print(
            f"[crash-dump] staging {n_logs} log file(s) + CRASH_META.json -> {prefix}",
            flush=True,
        )
        # Tier 1: logs+meta (fail-loud — load-bearing, non-LFS).
        _upload_logs_tier(staging, prefix)
        # Tier 2: partial data (best-effort — may be LFS .pt under a storage 403).
        _upload_partial_data_tier(data_dir, prefix)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if args.print_prefix_only:
        # Already printed dest lines above; the smoke greps the prefix from this line.
        print(f"CRASH_DUMP_PREFIX={prefix}", flush=True)
    print(f"[crash-dump] complete -> {DEFAULT_DATASET_REPO}:{prefix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
