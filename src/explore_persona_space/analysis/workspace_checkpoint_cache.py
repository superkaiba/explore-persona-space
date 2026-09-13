"""Validate and publish optional decomposition checkpoints without taking producer locks.

Only the original producer owns coverage manifests and phase completion. A cache
publisher adds whole immutable tensor files using link(2), never replaces files,
and validates an original producer's file if that producer wins a race.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.workspace_artifacts import validate_context_input
from explore_persona_space.analysis.workspace_runtime import content_sha256

TARGETS = {"full", "J", "restJ", "R", "restR"}
STATISTICS = {
    "active_atoms",
    "squared_error",
    "input_squared_norm",
    "zero_update_steps",
    "increasing_error_steps",
}


def regular_path(root: Path, relative: str, *, create_parents=False) -> Path:
    """Reject traversal and symlinks, including parent directories."""
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p.startswith(".") for p in rel.parts):
        raise ValueError("Cache path must be a plain relative path")
    if root.is_symlink() or root.resolve(strict=True) != root.absolute():
        raise ValueError("Cache root must be a real absolute directory")
    for depth in range(1, len(rel.parts) + 1):
        path = root.joinpath(*rel.parts[:depth])
        if path.is_symlink():
            raise ValueError(f"Symlink in cache path: {path}")
        if depth < len(rel.parts):
            if create_parents:
                path.mkdir(exist_ok=True)
            if path.exists() and not path.is_dir():
                raise ValueError(f"Non-directory cache parent: {path}")
    return root / rel


def read_checkpoint(path: Path, expected_sha256: str | None = None):
    """Hash and deserialize one open inode, even if the producer replaces its name."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Checkpoint must be a regular file")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Checkpoint differs from immutable upload: {path}")
        stream.seek(0)
        value = torch.load(stream, map_location="cpu", weights_only=True)
    return value, digest


def equal_tree(actual, expected, location="checkpoint"):
    """Exact numerical parity, ignoring only serialization/container byte details."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype:
            raise ValueError(f"Tensor type mismatch at {location}")
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Mapping mismatch at {location}")
        for key in expected:
            equal_tree(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Sequence mismatch at {location}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal_tree(left, right, f"{location}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Value mismatch at {location}")


def validate_checkpoint(saved, capture, source_sha256, contract, root: Path):
    """Bind a checkpoint to exact captures and check all pooled/statistical fields."""
    identity = contract["identity"]
    rows = capture["rows"]
    if len(rows) < 2 or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Unique original rollout seeds required")
    prompt = rows[0]["prompt_sha256"]
    lengths = [len(row["answer_states"]) for row in rows]
    if min(lengths) < 1 or any(row["prompt_sha256"] != prompt for row in rows):
        raise ValueError("Nonempty single-context captures required")
    if (
        saved["contract"] != contract
        or saved["contract_sha256"] != content_sha256(contract)
        or saved["source_sha256"] != source_sha256
        or saved["context_input_reference"] != capture["identity"]
        or saved["prompt_sha256"] != prompt
        or saved["rollout_seeds"] != [row["seed"] for row in rows]
        or saved["token_counts"] != lengths
    ):
        raise ValueError("Checkpoint contract, capture, prompt, seeds or lengths differ")
    validate_context_input(saved["context_input_reference"], saved["x"], root, identity)
    xs = torch.stack([row["x"].float() for row in rows])
    torch.testing.assert_close(saved["x"], xs[0], rtol=0, atol=0)
    if saved["max_repeat_context_activation_difference"] != float((xs - xs[0]).abs().max()):
        raise ValueError("Repeated-input diagnostic differs")
    dimension = xs.shape[1]
    for field in ("targets", "rollout_means", "mean_target_noise_trace"):
        if set(saved[field]) != TARGETS:
            raise ValueError(f"Incomplete target set in {field}")
    actual_full = np.stack([r["answer_states"].float().double().mean(0).numpy() for r in rows])
    np.testing.assert_array_equal(saved["rollout_means"]["full"].numpy(), actual_full)
    for name in TARGETS:
        means = saved["rollout_means"][name]
        target = saved["targets"][name]
        if (
            means.shape != (len(rows), dimension)
            or target.shape != (dimension,)
            or means.dtype != torch.float64
            or target.dtype != torch.float64
            or not torch.isfinite(means).all()
            or not torch.isfinite(target).all()
        ):
            raise ValueError(f"Invalid pooled tensor: {name}")
        expected_target = means.numpy().mean(0)
        np.testing.assert_array_equal(target.numpy(), expected_target)
        expected_noise = float(
            np.square(means.numpy() - expected_target).sum() / (len(rows) - 1) / len(rows)
        )
        if saved["mean_target_noise_trace"][name] != expected_noise:
            raise ValueError(f"Noise diagnostic differs: {name}")
    for arm in ("J", "R"):
        np.testing.assert_array_equal(
            saved["rollout_means"][f"rest{arm}"].numpy(),
            actual_full - saved["rollout_means"][arm].numpy(),
        )
    validate_statistics(saved["decomposition_statistics"], lengths, contract["k"])


def validate_statistics(all_statistics, lengths, k):
    """Check every persisted token statistic, retaining failed-pursuit counters."""
    if set(all_statistics) != {"J", "R"}:
        raise ValueError("Incomplete decomposition statistics")
    for arm in ("J", "R"):
        statistics = all_statistics[arm]
        if set(statistics) != STATISTICS:
            raise ValueError("Incomplete token statistics")
        for field, value in statistics.items():
            if (
                value.shape != (sum(lengths),)
                or not torch.isfinite(value).all()
                or (value < 0).any()
            ):
                raise ValueError(f"Invalid token statistic: {field}")
            if field in {"active_atoms", "zero_update_steps", "increasing_error_steps"}:
                if value.dtype != torch.int64 or (value > k).any():
                    raise ValueError(f"Invalid pursuit count: {field}")
            elif value.dtype != torch.float32:
                raise ValueError(f"Invalid token statistic precision: {field}")


def publish_checkpoint(staged: Path, destination: Path, validate):
    """Publish once; preserve and validate any concurrent original-producer file.

    The caller has verified immutable staged bytes and published lineage before
    invoking this function. The returned hash identifies an observation, not a
    promise that the producer will never atomically replace that path later.
    """
    if staged.is_symlink() or destination.is_symlink():
        raise ValueError("Checkpoint publication does not follow symlinks")
    if staged.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError("Atomic checkpoint publication requires the same filesystem")
    try:
        os.link(staged, destination, follow_symlinks=False)
        outcome = "published_optional_checkpoint"
    except FileExistsError:
        outcome = "preserved_existing_checkpoint"
    value, digest = read_checkpoint(destination)
    validate(value)
    return {"outcome": outcome, "observed_sha256": digest}
