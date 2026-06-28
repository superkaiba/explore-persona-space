#!/usr/bin/env python3
"""Real-GPU memory-non-growth validation for #671's hook-based extraction.

Runs ``N`` repeated ``extract_layer_activations`` calls on
``Qwen/Qwen2.5-7B-Instruct`` on a real GPU, across 2 arms
``{hook, old output_hidden_states=True}`` for ONE allocator regime per process
invocation (``PYTORCH_CUDA_ALLOC_CONF`` is read once at CUDA-context init and
cannot be flipped in-process), recording ``torch.cuda.memory_{allocated,reserved}``
plus the segment-level ``memory_stats()`` fields per iter.

The hook arm is the #671 fix (``extract_layer_activations`` registers forward
hooks + ``output_hidden_states=False``); the ``old_ohs_true`` arm reproduces the
pre-#671 read shape (``model(ids, output_hidden_states=True)`` then
``out.hidden_states[L+1]``) and is the POSITIVE CONTROL — under
``expandable_segments:True`` it must retain materially more reserved memory than
the hook arm, else the read does not demonstrate the fix removed a real growth.

Both arms run grad-disabled and matched: ``extract_layer_activations`` is itself
``@torch.no_grad()``, ``_old_path_read`` is decorated ``@torch.no_grad()`` to
MATCH, and ``run_arm`` wraps the whole per-arm loop in ``torch.inference_mode()``
with an ``assert not torch.is_grad_enabled()`` at the top of each iter. This
isolates the positive-control gap to the ``output_hidden_states=True`` tuple
keepalive the fix removed, NOT autograd-graph retention (a false-positive gap
there would route straight to a false PASS — the worst outcome for a validation
whose whole purpose is the positive control).

The script does NOT set ``PYTORCH_CUDA_ALLOC_CONF`` — the dispatcher / caller
sets it before CUDA init. The script reads the current regime from the env and
records it. The launcher runs the script twice (one env per regime) to produce
both ``memory_curves_expandable_segments_on.json`` and
``memory_curves_default_allocator.json``.

Usage::

    # one regime, real GPU (the dispatcher sets PYTORCH_CUDA_ALLOC_CONF first):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run python scripts/issue673_gpu_memory_validation.py
    unset PYTORCH_CUDA_ALLOC_CONF; \
      uv run python scripts/issue673_gpu_memory_validation.py

    # write the per-regime JSON to a dispatcher-specified path:
    uv run python scripts/issue673_gpu_memory_validation.py --out /workspace/logs/i673-exp.json

    # CPU structure smoke (NO real GPU, NO Qwen) — proves CLI + JSON schema:
    uv run python scripts/issue673_gpu_memory_validation.py --smoke
"""

# ruff: noqa: RUF003  # scientific notation (≥, −, ≈) in docstrings/strings

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.extraction import extract_layer_activations
from explore_persona_space.orchestrate.env import load_dotenv

# --- constants (see plan §11 Decision Rationale) ---
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [7, 14, 21]  # the live #545/#651/#667 subset
N_ITERS = 50
WARMUP = 5  # excluded from the flat assertion
MAX_TOKENS = 512  # the size the live extraction line reads
ABS_TOL_GiB = 1.0  # max-min reserved span over [warmup:] must be below this
SLOPE_TOL_GiB_per_iter = 0.02  # last-30 linear-fit slope must be below this
CTRL_GAP_GiB = 0.05  # positive-control min reserved gap (conservative; §11)
GiB = 1024**3
OUT_DIR = Path("figures/issue_673")


def regime_tag(regime: str) -> str:
    """Map the raw ``PYTORCH_CUDA_ALLOC_CONF`` value to the output-file tag.

    ``expandable_segments_on`` when the allocator config requests expandable
    segments, else ``default_allocator``.
    """
    return "expandable_segments_on" if "expandable_segments:True" in regime else "default_allocator"


def _fixed_ids(tok, device):
    """Build a deterministic ``<=MAX_TOKENS``-token prompt on ``device``.

    ``torch.manual_seed(0)`` is set for reproducibility metadata; the prompt is
    fixed and greedy (no sampling) so the seed is moot beyond ordering.
    """
    torch.manual_seed(0)
    text = "The assistant gives careful, structured advice. " * 40  # ~ a few hundred tok
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).input_ids.to(
        device
    )
    assert ids.shape[1] <= MAX_TOKENS, ids.shape
    return ids


@torch.no_grad()
def _old_path_read(model, ids, layers):
    """The PRE-#671 read: full ``output_hidden_states`` tuple, subset-indexed.

    This is the thing #671 removed — it materializes ALL ``L+1`` residual-stream
    tensors per forward. Decorated ``@torch.no_grad()`` to MATCH the hook helper
    (``extract_layer_activations`` is itself ``@torch.no_grad()``); without this
    the old arm would retain an autograd graph and inflate the positive-control
    reserved gap with an autograd artifact rather than the pre-#671
    ``hidden_states``-tuple keepalive the fix removed. Returns ``{L: (B, T, H)}``
    via the ``hs[L+1]`` block-index convention.
    """
    out = model(input_ids=ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1] for L in layers}


def _seg_stats() -> dict:
    """Segment-level ``torch.cuda.memory_stats()`` fields for the current device.

    Lets the analyzer separate "segment COUNT grew" from "segment SIZE grew" in
    an INCONCLUSIVE-gap case. Returns the four reserved/segment current+peak
    byte counters (zeros on a fresh allocator).
    """
    s = torch.cuda.memory_stats()
    return {
        "reserved_current": int(s.get("reserved_bytes.all.current", 0)),
        "reserved_peak": int(s.get("reserved_bytes.all.peak", 0)),
        "segment_current": int(s.get("segment.all.current", 0)),
        "segment_peak": int(s.get("segment.all.peak", 0)),
    }


def run_arm(model, ids, *, use_hook: bool, n_iters: int = N_ITERS) -> dict:
    """Run ``n_iters`` extraction calls; record memory after sync each iter.

    Runs under ``torch.inference_mode()`` so BOTH arms are grad-disabled
    identically (the hook helper is already ``@torch.no_grad()``; ``_old_path_read``
    is decorated ``@torch.no_grad()``). The ``assert not torch.is_grad_enabled()``
    at the top of each iter makes a future edit that drops the decorator fail
    loud. Records ``allocated`` + ``reserved`` (bytes) + per-iter segment-level
    stats; returns the per-arm record (curves + ``grad_enabled``/``inference_mode``
    metadata).
    """
    allocated, reserved, seg = [], [], []
    with torch.inference_mode():
        for _ in range(n_iters):
            assert not torch.is_grad_enabled(), "arm loop must run grad-disabled"
            if use_hook:
                captured = extract_layer_activations(model, ids, LAYERS)
            else:
                captured = _old_path_read(model, ids, LAYERS)
            torch.cuda.synchronize()
            allocated.append(int(torch.cuda.memory_allocated()))
            reserved.append(int(torch.cuda.memory_reserved()))
            seg.append(_seg_stats())
            del captured  # drop the per-iter reference (both arms, fair)
    return {
        "allocated": allocated,
        "reserved": reserved,
        "segment_stats": seg,
        "grad_enabled": False,
        "inference_mode": True,
    }


def flat(reserved_bytes) -> dict:
    """Flatness verdict for a per-iter reserved-memory curve (bytes).

    ``span_GiB`` = max-min reserved over ``[WARMUP:]``; ``tail_slope_GiB_per_iter``
    = last-30 linear-fit slope. ``flat`` is True iff both are below their
    tolerances (``ABS_TOL_GiB`` / ``SLOPE_TOL_GiB_per_iter``). This is the
    canonical flatness logic the offline reader (``issue673_assert.py``) imports.

    The warmup window is dropped only when the curve is longer than ``WARMUP``;
    on a shorter curve (e.g. the ``--smoke`` N=3 run) the whole curve is used so
    the post-warmup slice is never empty. A single-point slope is 0.0.
    """
    warmup = WARMUP if len(reserved_bytes) > WARMUP else 0
    r = np.asarray(reserved_bytes[warmup:], float) / GiB
    span = float(r.max() - r.min())
    tail = np.asarray(reserved_bytes[-30:], float) / GiB
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 1 else 0.0
    return {
        "span_GiB": span,
        "tail_slope_GiB_per_iter": slope,
        "flat": span < ABS_TOL_GiB and abs(slope) < SLOPE_TOL_GiB_per_iter,
    }


def _run_real_gpu(out_path: Path | None) -> dict:
    """Load Qwen on CUDA, run both arms for the current allocator regime, write JSON."""
    load_dotenv()
    assert torch.cuda.is_available(), "real-GPU validation requires CUDA"
    # PYTORCH_CUDA_ALLOC_CONF must be set BEFORE CUDA init, so the two allocator
    # regimes are run as TWO separate process invocations (the launcher sets the
    # env per invocation), NOT toggled in-process. We only READ + RECORD it here.
    regime = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")
    ).to("cuda")
    model.eval()
    ids = _fixed_ids(tok, "cuda")

    results = _new_results(regime, int(ids.shape[1]))
    for use_hook, name in [(True, "hook"), (False, "old_ohs_true")]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        arm = run_arm(model, ids, use_hook=use_hook)
        arm["flatness"] = flat(arm["reserved"])
        results["arms"][name] = arm

    return _write_results(results, regime, out_path)


def _run_smoke(out_path: Path | None) -> dict:
    """Tiny CPU-only structure smoke (NO real GPU, NO Qwen).

    Proves the script is importable, the CLI parses, and the JSON schema lands —
    it does NOT touch CUDA or load the model. Uses a CPU stub model (exercising
    ``extract_layer_activations``'s full-tuple fallback path) and synthetic memory
    counters so the per-arm/per-iter record shape matches the real run exactly.
    """
    load_dotenv()
    n_iters = 3
    regime = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
    model, ids = _smoke_stub_and_ids(prompt_tokens=8)

    results = _new_results(regime, int(ids.shape[1]), smoke=True, n_iters=n_iters)
    for use_hook, name in [(True, "hook"), (False, "old_ohs_true")]:
        arm = _smoke_run_arm(model, ids, use_hook=use_hook, n_iters=n_iters)
        arm["flatness"] = flat(arm["reserved"])
        results["arms"][name] = arm

    return _write_results(results, regime, out_path)


def _smoke_stub_and_ids(prompt_tokens: int):
    """A CPU ``output_hidden_states``-only stub model + a tiny ``(1, T)`` id tensor.

    The stub has no ``model.model.layers`` structure, so
    ``extract_layer_activations`` takes its full-tuple fallback path on BOTH the
    hook and old arms — the smoke verifies wiring, not GPU memory.
    """
    n_blocks = max(LAYERS) + 2
    d = 4

    class _Out:
        pass

    class _Stub:
        def __call__(self, input_ids=None, output_hidden_states=False, attention_mask=None, **_kw):
            t = int(input_ids.shape[1])
            o = _Out()
            o.hidden_states = [torch.zeros(1, t, d) for _ in range(n_blocks + 1)]
            o.logits = torch.zeros(1, t, 8)
            return o

        def eval(self):
            return self

    ids = torch.zeros(1, prompt_tokens, dtype=torch.long)
    return _Stub(), ids


def _smoke_run_arm(model, ids, *, use_hook: bool, n_iters: int) -> dict:
    """Smoke analogue of ``run_arm`` with synthetic (flat) memory counters.

    No ``torch.cuda`` calls — emits a constant reserved/allocated curve and zero
    segment stats so the produced JSON has the real-run schema. Grad-disablement
    metadata mirrors the real arm (the extraction call itself runs under
    ``torch.inference_mode()``).
    """
    allocated, reserved, seg = [], [], []
    with torch.inference_mode():
        for _ in range(n_iters):
            assert not torch.is_grad_enabled(), "smoke arm loop must run grad-disabled"
            if use_hook:
                captured = extract_layer_activations(model, ids, LAYERS)
            else:
                captured = _old_path_read(model, ids, LAYERS)
            allocated.append(1024)
            reserved.append(2048)
            seg.append(
                {
                    "reserved_current": 2048,
                    "reserved_peak": 2048,
                    "segment_current": 1,
                    "segment_peak": 1,
                }
            )
            del captured
    return {
        "allocated": allocated,
        "reserved": reserved,
        "segment_stats": seg,
        "grad_enabled": False,
        "inference_mode": True,
    }


def _new_results(regime: str, max_tokens: int, *, smoke: bool = False, n_iters: int = N_ITERS):
    """Build the per-run results dict header (model/layers/regime/torch metadata)."""
    return {
        "model": MODEL,
        "layers": LAYERS,
        "n_iters": n_iters,
        "warmup": WARMUP,
        "max_tokens": max_tokens,
        "allocator_regime": regime,
        "allocator_tag": regime_tag(regime),
        "tolerances": {
            "ABS_TOL_GiB": ABS_TOL_GiB,
            "SLOPE_TOL_GiB_per_iter": SLOPE_TOL_GiB_per_iter,
            "CTRL_GAP_GiB": CTRL_GAP_GiB,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "smoke": smoke,
        "arms": {},
    }


def _write_results(results: dict, regime: str, out_path: Path | None) -> dict:
    """Write the per-regime JSON (to ``out_path`` or the default figures path).

    Returns the results dict; prints a one-line flatness summary to stdout.
    """
    if out_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"memory_curves_{regime_tag(regime)}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    summary = {
        "regime": regime_tag(regime),
        "out": str(out_path),
        **{k: v["flatness"] for k, v in results["arms"].items()},
    }
    print(json.dumps(summary, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path for this regime (default: figures/issue_673/memory_curves_<tag>.json).",  # noqa: E501
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU structure smoke: tiny N, CPU stub model, NO real GPU touched.",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        _run_smoke(args.out)
    else:
        _run_real_gpu(args.out)


if __name__ == "__main__":
    main()
