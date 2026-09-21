"""Audited task2673 grouped-FP8 scheduling repair and real CUDA shape diagnostic."""

from __future__ import annotations

import functools
import hashlib
import importlib
import sys
from pathlib import Path
from types import CodeType

import torch

REVISION = "8c178950e8710e26a2210e4e909cd02dc16c8715"
GROUPED_SHA256 = "382b03b38f0d0531d4fc6d2ce5e7441576ff9b9a38f3e3200f616ac060c69206"
UTILS_SHA256 = "2cb87f98ab385e41c95fdf1e1bec777a25c8e0bb003a357a72c07a53c2b7b5d0"


def _original_code(path: Path, name: str) -> CodeType:
    """Recover a function's code from the hash-verified immutable source."""
    return next(
        c
        for c in compile(path.read_text(), str(path), "exec", dont_inherit=True).co_consts
        if isinstance(c, CodeType) and c.co_name == name
    )


def install_grouped_m16(kernel, revision: str) -> dict:
    """Change only the verified block-FP8 dispatcher's M scheduling decision."""
    if revision != REVISION:
        raise RuntimeError("grouped M16 repair requires the reviewed kernel revision")
    build = Path(kernel.__file__).absolute().parent
    paths = {"grouped.py": GROUPED_SHA256, "utils.py": UTILS_SHA256}
    for name, expected in paths.items():
        path = build / name
        if REVISION not in path.parts or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"grouped M16 immutable source mismatch: {name}")
    grouped = importlib.import_module(kernel.matmul_grouped.__module__)
    utils = importlib.import_module(grouped.__package__ + ".utils")
    original = grouped.adaptive_block_size_m
    body = grouped._w8a8_block_dynamic_fp8_matmul_grouped._abstract_fn
    if (
        Path(grouped.__file__).absolute() != build / "grouped.py"
        or Path(utils.__file__).absolute() != build / "utils.py"
        or kernel.matmul_grouped is not grouped.matmul_grouped
        or kernel.matmul_grouped.__code__ != _original_code(build / "grouped.py", "matmul_grouped")
        or original is not utils.adaptive_block_size_m
        or original.__code__ != _original_code(build / "utils.py", "adaptive_block_size_m")
        or body.__code__
        != _original_code(build / "grouped.py", "_w8a8_block_dynamic_fp8_matmul_grouped")
        or body.__globals__ is not vars(grouped)
    ):
        raise RuntimeError("grouped M16 original function/dispatcher identity mismatch")
    seen = set()

    @functools.wraps(original)
    def fixed_m16(target_m: int) -> int:
        """Signal only when the actual block-FP8 custom-op body calls the repair."""
        if sys._getframe(1).f_code != body.__code__:
            raise RuntimeError("grouped M16 scheduling override reached an unreviewed dispatcher")
        prior = original(target_m)
        if prior not in seen:
            print(
                "[fp8-grouped-m16-engaged] dispatcher=grouped._w8a8_block_dynamic_fp8_matmul_grouped "
                f"target_m={target_m} original_m={prior} fixed_m=16",
                flush=True,
            )
            seen.add(prior)
        return 16

    grouped.adaptive_block_size_m = fixed_m16
    return {
        "name": "grouped_block_fp8_fixed_m16",
        "fixed_block_size_m": 16,
        "scope": "grouped._w8a8_block_dynamic_fp8_matmul_grouped only",
        "original_source_sha256": paths,
        "original_function_identity_verified": True,
        "kernel_source_files_modified": False,
        "autotune_key": ["N", "K", "BLOCK_SIZE_M"],
        "reason": "attempt9 M16 singleton/mixed rows agree; M64 mixed rows diverge",
    }


def cuda_shape_diagnostic(
    kernel, *, parity_tolerance: float, repeatability_tolerance: float, record
) -> dict:
    """Compare shared expert rows at M16/M64 shapes before loading model weights.

    Uses native E4M3 weights, UE8M0 scales, BF16 activations and the production
    Transformers loader/dispatcher. This is kernel evidence, not model acceptance.
    """
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

    dispatch = load_finegrained_fp8_kernel().grouped_matmul
    if dispatch is not kernel.matmul_grouped or not torch.cuda.is_available():
        raise RuntimeError("shape diagnostic requires the actual CUDA FP8 dispatch seam")
    grouped = importlib.import_module(dispatch.__module__)
    fixed = grouped.adaptive_block_size_m
    original = fixed.__wrapped__
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(2673)
    experts = 256
    small_counts = torch.arange(experts, device=device, dtype=torch.int32) % 7
    large_counts = torch.where(small_counts > 0, small_counts + 40, 0)
    small_offsets = small_counts.cumsum(0).to(torch.int32)
    large_offsets = large_counts.cumsum(0).to(torch.int32)
    small_s, large_s = int(small_counts.sum()), int(large_counts.sum())
    owner = torch.repeat_interleave(torch.arange(experts, device=device), small_counts)
    starts_small = small_offsets - small_counts
    starts_large = large_offsets - large_counts
    shared = starts_large[owner] + torch.arange(small_s, device=device) - starts_small[owner]
    cases = []
    report = {
        "passed": False,
        "status": "running",
        "device": torch.cuda.get_device_name(device),
        "small_s": small_s,
        "large_s": large_s,
        "experts": experts,
        "original_m": [original((s + experts - 1) // experts) for s in (small_s, large_s)],
        "fixed_m": 16,
        "parity_tolerance": parity_tolerance,
        "repeatability_tolerance": repeatability_tolerance,
        "production_dispatch_identity_verified": True,
        "cases": cases,
        "coverage": "tiny CUDA kernel shape diagnostic only; full-model smoke remains mandatory",
    }
    record(report)
    try:
        for n, k in ((256, 256), (128, 7168), (7168, 128)):
            weights = torch.randn(experts, n, k, device=device, generator=generator).to(
                torch.float8_e4m3fn
            )
            scales = torch.full(
                (experts, n // 128, k // 128), 127, device=device, dtype=torch.uint8
            )
            large = torch.randn(
                large_s, k, device=device, dtype=torch.bfloat16, generator=generator
            )
            small = large[shared].contiguous()
            results = {}
            cases.append({"n": n, "k": k, "results": results})
            for mode, scheduler in (("original", original), ("fixed_m16", fixed)):
                grouped.adaptive_block_size_m = scheduler
                singleton_shape = dispatch(
                    small, weights, scales, small_offsets, small_counts, [128, 128]
                )
                batch_shape = dispatch(
                    large, weights, scales, large_offsets, large_counts, [128, 128]
                )[shared]
                repeat = dispatch(small, weights, scales, small_offsets, small_counts, [128, 128])
                torch.cuda.synchronize(device)
                for value in (singleton_shape, batch_shape, repeat):
                    if (
                        value.shape != (small_s, n)
                        or value.dtype != torch.bfloat16
                        or not torch.isfinite(value).all()
                    ):
                        raise RuntimeError("invalid shape diagnostic kernel output")
                norms = singleton_shape.float().norm(dim=-1)
                if (norms == 0).any():
                    raise RuntimeError("zero shape diagnostic outputs")
                results[mode] = {
                    "parity_relative_error": float(
                        ((batch_shape.float() - singleton_shape.float()).norm(dim=-1) / norms).max()
                    ),
                    "repeatability_relative_error": float(
                        ((repeat.float() - singleton_shape.float()).norm(dim=-1) / norms).max()
                    ),
                }
                record(report)
            del weights, scales, small, large, singleton_shape, batch_shape, repeat
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record(report)
        raise
    finally:
        grouped.adaptive_block_size_m = fixed
    passed = all(
        c["results"]["fixed_m16"]["parity_relative_error"] <= parity_tolerance
        and c["results"]["fixed_m16"]["repeatability_relative_error"] <= repeatability_tolerance
        for c in cases
    )
    report.update(passed=passed, status="passed" if passed else "rejected")
    record(report)
    return report
