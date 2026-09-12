"""Fail-closed producer contracts and complete frozen-subset accounting."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def validate_producer(actual: dict, expected: dict, *, native_ancestor=False) -> dict:
    """Check scientific identity; permit an audited unchanged native implementation."""
    for key in ("config_sha256", "selection_sha256", "model_role", "versions"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Producer identity mismatch: {key}")
    if actual.get("code") == expected.get("code"):
        return {"code_match": "exact"}
    if not native_ancestor:
        raise ValueError("Producer identity mismatch: code")
    code = actual["code"]
    # A native fit may precede the downstream pipeline commit. Verify the actual
    # producer implementation against the current checkout, not merely its label.
    sha = code.get("git_commit")
    if not isinstance(sha, str) or len(sha) != 40 or code.get("git_dirty"):
        raise ValueError("Native ancestor must have clean full-SHA provenance")
    paths = [
        "src/explore_persona_space/analysis/workspace_runtime.py",
        "src/explore_persona_space/analysis/workspace_lenses.py",
        "scripts/workspace_jr_runtime.py",
    ]
    vendor = Path("external/jacobian-lens/jlens")
    paths.extend(str(p) for p in sorted(vendor.rglob("*.py")))
    if not vendor.is_dir():
        raise ValueError("Missing native lens vendor implementation")
    hashes = {}
    for path in paths:
        prior = subprocess.run(
            ["git", "show", f"{sha}:{path}"], check=True, capture_output=True
        ).stdout
        current = Path(path).read_bytes()
        if prior != current:
            raise ValueError(f"Native producer implementation changed: {path}")
        hashes[path] = hashlib.sha256(prior).hexdigest()
    return {"code_match": "verified_unchanged_native_sources", "producer_sha": sha, "files": hashes}


def validate_coverage(report: dict, expected_ids: list[str], identity: dict) -> list[str]:
    """Require a completed phase and exact included/excluded frozen membership."""
    validate_producer(report["identity"], identity)
    if report.get("status") != "complete":
        raise ValueError("Producer phase did not complete")
    included = report["included_prompt_sha256"]
    excluded = report["exclusions"]
    if any(not row.get("reason") for row in excluded):
        raise ValueError("Every excluded context needs an explicit reason")
    realized = included + [row["prompt_sha256"] for row in excluded]
    if len(realized) != len(set(realized)) or set(realized) != set(expected_ids):
        raise ValueError("Coverage does not reconcile the exact frozen subset")
    if report["planned_contexts"] != len(expected_ids):
        raise ValueError("Planned context count differs from the frozen subset")
    return included
