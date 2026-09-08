"""Explicit continuation clock, separate from immutable scientific provenance.

Task2588 progress152 authorizes a fresh 1800-second pilot allowance. Each
grant binds one pod and provision epoch; retries reuse its durable upload
ledger. Only measured, completed upload intervals are credited. An abrupt
controller death can under-credit its active upload, never renew the grant.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

SCIENCE_SHA = "0a06988756f6b407e4d1562b0ac8d88af52dde2d"
GRANT_ENV = "EPS2588_CONTINUATION_GRANT"
HASH_ENV = "EPS2588_CONTINUATION_GRANT_SHA256"
AUTHORIZATION = "task2588 epm:progress152"


def science_root(path: Path) -> Path:
    """Require a real clean checkout of the exact checkpoint-producing source."""
    root = path.resolve(strict=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if sha != SCIENCE_SHA or dirty:
        raise RuntimeError("scientific checkout must be clean and pinned to saved-response source")
    if not (root / "scripts/issue2588_run_cell.py").is_file():
        raise RuntimeError("scientific checkout is incomplete")
    return root


class Grant:
    """One immutable user allowance with idempotent per-upload receipts."""

    def __init__(self, path: Path, expected_hash: str, env: dict):
        self.path = path.resolve(strict=True)
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise RuntimeError("continuation grant hash mismatch")
        self.record = json.loads(raw)
        r = self.record
        if (
            r.get("schema") != 1
            or r.get("authorization") != AUTHORIZATION
            or r.get("run_id") != "qwen3-chat-v3"
            or r.get("allowance_s") != 1800
            or r.get("science_sha") != SCIENCE_SHA
            or r.get("pod_id") != env.get("RUNPOD_POD_ID")
            or not r.get("pod_id")
            or not isinstance(r.get("historical_terminal_revisions"), list)
            or "9ba0d2275c97edc444bbc2d523086fa54045b6aa" not in r["historical_terminal_revisions"]
        ):
            raise RuntimeError("invalid continuation grant scope, authority, or lineage")
        self.epoch = r["provision_epoch"]
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, (int, float))
            or not math.isfinite(self.epoch)
            or not 1788880276 <= self.epoch <= time.time()
            or float(env.get("EPS2588_SMOKE_STARTED_AT", "nan")) != self.epoch
        ):
            raise RuntimeError("continuation requires the post-approval provision epoch")
        if any(
            env.get(k)
            for k in (
                "EPS2588_SMOKE_PRIOR_REPORT",
                "EPS2588_SMOKE_PRIOR_REPORT_SHA256",
                "EPS2588_SMOKE_SUPPLEMENT_REPORT",
            )
        ):
            raise RuntimeError("continuation cannot mix legacy clock credits")
        self.ledger = self.path.with_suffix(".uploads")
        self.ledger.mkdir(exist_ok=True)
        binding = self.ledger / "binding.json"
        if binding.exists():
            if binding.read_bytes() != raw:
                raise RuntimeError("grant changed across retries")
        else:
            with binding.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        self.credit_s()  # Validate existing receipts before any workload.

    def credit_s(self) -> float:
        """Sum non-overlapping measured upload intervals, never report totals."""
        rows = [json.loads(p.read_text()) for p in self.ledger.glob("upload-*.json")]
        intervals = sorted((r["start_epoch"], r["end_epoch"]) for r in rows)
        previous = self.epoch
        total = 0.0
        for start, end in intervals:
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
                raise RuntimeError("invalid upload receipt times")
            if not previous <= start <= end <= time.time():
                raise RuntimeError("overlapping, future, or pre-grant upload credit")
            total += end - start
            previous = end
        return total

    def remaining_s(self) -> float:
        """Charge all wall time since provision except durably measured uploads."""
        return self.record["allowance_s"] - (time.time() - self.epoch - self.credit_s())

    def finish_upload(self, key: str, start_epoch: float, end_epoch: float) -> None:
        """Persist a completed upload once; duplicate identical receipts are harmless."""
        name = hashlib.sha256(key.encode()).hexdigest()
        dest = self.ledger / f"upload-{name}.json"
        record = {"key": key, "start_epoch": start_epoch, "end_epoch": end_epoch}
        if dest.exists():
            if json.loads(dest.read_text()) != record:
                raise RuntimeError("conflicting upload receipt")
        else:
            # Dispatcher holds the run lock. Rename avoids partial JSON after a crash.
            temp = dest.with_suffix(f".tmp-{os.getpid()}")
            with temp.open("x") as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, dest)
        self.credit_s()


def from_env(env: dict) -> Grant | None:
    """Load only a fully specified pinned continuation grant."""
    path, sha = env.get(GRANT_ENV), env.get(HASH_ENV)
    if not path and not sha:
        return None
    if not path or not sha:
        raise RuntimeError("continuation requires both grant path and hash")
    return Grant(Path(path), sha, env)


def configure_environment(env: dict, path: Path | None, sha: str | None) -> None:
    """Require explicit grant arguments and reject inherited path/hash disagreement."""
    inherited_path, inherited_sha = env.get(GRANT_ENV), env.get(HASH_ENV)
    if not path and not sha:
        if inherited_path or inherited_sha:
            raise RuntimeError("inherited continuation requires explicit pinned arguments")
        return
    if not path or not sha:
        raise RuntimeError("continuation requires both grant path and hash")
    if (inherited_path and Path(inherited_path).resolve() != path.resolve()) or (
        inherited_sha and inherited_sha != sha
    ):
        raise RuntimeError("inherited continuation disagrees with explicit arguments")
    env[GRANT_ENV], env[HASH_ENV] = str(path.resolve()), sha
