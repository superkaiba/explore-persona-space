"""Calibration-only lens averaging, convergence and token-direction diagnostics."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import torch

from explore_persona_space.analysis.workspace_runtime import content_sha256, file_sha256


def validate_calibration_order(token_manifest: dict, selection: dict) -> None:
    """Subset membership alone cannot enforce nested calibration sample order."""
    excluded = {row["prompt_sha256"] for row in token_manifest["excluded"]}
    expected = [
        row["prompt_sha256"]
        for row in selection["subsets"]["calibration"]
        if row["prompt_sha256"] not in excluded
    ]
    actual = [row["prompt_sha256"] for row in token_manifest["rows"]]
    if actual != expected:
        raise ValueError("Calibration token order differs from frozen selection")


def calibration_means(paths: list[Path], token_manifest: dict, dimension: int) -> tuple[dict, dict]:
    """Stream frozen-order paired shards into nested and interleaved-half means.

    Full coverage is reported separately from successful aggregation. A pilot
    prefix never masquerades as all selected calibration prompts. FP64 sums
    avoid order-dependent FP32 accumulation; stored means are FP32.
    """
    frozen = token_manifest["rows"]
    if not paths or len(paths) > len(frozen):
        raise ValueError("Invalid number of calibration shards")
    sizes = sorted({min(n, len(paths)) for n in (32, 64, len(frozen))})
    groups = {f"first_{n}": list(range(n)) for n in sizes}
    if len(paths) >= 2:
        groups.update(even=list(range(0, len(paths), 2)), odd=list(range(1, len(paths), 2)))
    sums = {
        group: {arm: torch.zeros(dimension, dimension, dtype=torch.float64) for arm in ("J", "R")}
        for group in groups
    }
    contract_digest, ledger = None, []
    for index, path in enumerate(paths):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        digest = content_sha256(saved["contract"])
        if contract_digest is None:
            contract_digest = digest
        if digest != saved["contract_sha256"] or digest != contract_digest:
            raise ValueError("Calibration shards do not share a valid contract")
        if (
            saved["prompt_sha256"] != frozen[index]["prompt_sha256"]
            or saved["token_ids"] != frozen[index]["token_ids"]
            or saved["contract"]["tokens_sha256"] != content_sha256(token_manifest)
        ):
            raise ValueError("Calibration shard differs from the exact frozen prefix")
        for arm in ("J", "R"):
            matrix = saved[arm]
            if matrix.shape != (dimension, dimension) or not torch.isfinite(matrix).all():
                raise ValueError("Invalid calibration matrix")
            for group, indices in groups.items():
                if index in indices:
                    sums[group][arm] += matrix.double()
        ledger.append(
            {
                "file": path.name,
                "sha256": file_sha256(path),
                "prompt_sha256": saved["prompt_sha256"],
            }
        )
    means = {
        group: {arm: (matrix / len(groups[group])).float() for arm, matrix in arms.items()}
        for group, arms in sums.items()
    }
    return means, {
        "full_calibration_membership": len(paths) == len(frozen),
        "selected_prompts": len(frozen) + len(token_manifest["excluded"]),
        "valid_prompts": len(frozen),
        "realized_prompts": len(paths),
        "full_group": f"first_{len(paths)}",
        "groups": {
            group: [frozen[i]["prompt_sha256"] for i in indices]
            for group, indices in groups.items()
        },
        "source_files": ledger,
        "contract_sha256": contract_digest,
        "token_manifest_sha256": content_sha256(token_manifest),
    }


def matrix_agreement(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Global Frobenius similarity, with an explicit zero-norm status."""
    a, b = left.double(), right.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Matrix comparison requires finite matching shapes")
    na, nb = a.norm().item(), b.norm().item()
    return {
        "left_norm": na,
        "right_norm": nb,
        "cosine": (a * b).sum().item() / (na * nb) if na and nb else None,
        "relative_frobenius_error_to_right": (a - b).norm().item() / nb if nb else None,
        "status": "ok" if na and nb else "zero_norm",
    }


def paired_direction_agreement(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Token-paired direction cosine, excluding and identifying zero vectors."""
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Token direction arrays must match")
    a, b = left.double(), right.double()
    valid = (
        torch.isfinite(a).all(1)
        & torch.isfinite(b).all(1)
        & (a.norm(dim=1) > 0)
        & (b.norm(dim=1) > 0)
    )
    cosine = torch.full((len(a),), torch.nan, dtype=torch.float64)
    cosine[valid] = (a[valid] * b[valid]).sum(1) / (a[valid].norm(dim=1) * b[valid].norm(dim=1))
    return {"cosine": cosine, "valid": valid}


def eligible_calibration_tokens(
    tokenizer, token_manifest: dict, *, seed: int, maximum: int
) -> list[int]:
    """Use the preregistered count>=5, non-special, nonempty rule on calibration only."""
    counts = Counter(token for row in token_manifest["rows"] for token in row["token_ids"])
    special = set(tokenizer.all_special_ids)
    eligible = [
        token
        for token, count in counts.items()
        if count >= 5 and token not in special and tokenizer.decode([token])
    ]
    eligible.sort(key=lambda token: hashlib.sha256(f"{seed}:{token}".encode()).hexdigest())
    return eligible[:maximum]
