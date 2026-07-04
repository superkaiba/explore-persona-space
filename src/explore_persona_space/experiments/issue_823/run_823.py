"""Issue #823 — Does the per-context map h predict the model's own output or context-side
processing?

Phases
------
Phase 0   (VM, CPU)  : Verify HF substrate, reconstruct prompts, alignment gate
Phase 0.5 (GPU pod)  : Regenerate Qwen own answers via vLLM (arm A')
Phase 1   (VM, CPU)  : Generate Sonnet answers via dispatch_calls (arms B1, B2)
Phase 2   (VM, CPU)  : Build derangement permutation (arm C text)
Phase 3   (GPU pod)  : Teacher-forced activation extraction (arms A', B1, B2, C)
Phase 4   (VM, CPU)  : Ridge refitting — per-arm refit (DV1) + cross-arm transfer (DV3)
Phase 5   (VM, CPU)  : Validity diagnostics

GPU phases (0.5, 3) run on a provisioned pod; CPU phases (0,1,2,4,5) run on the VM.
Use --phase to select which phase(s) to run; --smoke reduces n_contexts to 10.

Sentinel contract (Phase 0.5+3 only):
  /workspace/logs/issue-823-phase05-done.json  → Phase 0.5 complete
  /workspace/logs/issue-823-phase3-smoke.json  → Phase 3 smoke PASS
  /workspace/logs/issue-823-phase3-done.json   → Phase 3 full extraction complete
  /workspace/logs/issue-823-epm_results.json   → Final sentinel (GCP poller reads)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("issue823")

# ── constants ──────────────────────────────────────────────────────────────────
HF_DATA_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO_DATA = "superkaiba1/explore-persona-space-data"
BUNDLE_REPO_REVISION = "c94070508aa1c1f9c015ceb072231a2e51b28b3f"
BUNDLE_PATH_IN_REPO = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
BUNDLE_SHA256 = "46c06e89c513ca598bc83be1c87689694a47bfc927a81d0d738a54df769dbf9a"

LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
N_CONTEXTS_FULL = 5000
N_SMOKE = 10

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
GENERATION_SUFFIX = "<|im_start|>assistant\n"

TRAITS = ("evil", "sycophancy", "hallucination")
# Pre-registered from #779 (frozen before this experiment, no data-driven selection)
READ_OUT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}

SONNET_MODEL = "claude-sonnet-4-5-20250929"
# Matched to pass_b's generation cap (issue779_collect.py:558 SamplingParams max_tokens=1024)
# so Sonnet answer lengths are comparable to Qwen's own (length-nuisance control, plan §5).
SONNET_MAX_TOKENS = 1024
SENTINEL_SCHEMA_VERSION = 1

# HF upload slug
ISSUE_SLUG = "issue823_own_vs_external"

# ── helpers ────────────────────────────────────────────────────────────────────


def _json_np(o: Any):
    """json.dumps default= converter for numpy scalars/arrays (np.bool_, np.floating, ...)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def log_phase(name: str) -> None:
    """Emit a [phase=...] log line for poll_pipeline.py."""
    logger.info("[phase=%s]", name)


def write_sentinel(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write a poll_pipeline-compatible sentinel file."""
    payload["sentinel_schema_version"] = SENTINEL_SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_np))
    logger.info("Sentinel written: %s", path)


def resolve_base_dir(args_base_dir: str | None) -> pathlib.Path:
    """Return the base dir for all outputs."""
    if args_base_dir:
        return pathlib.Path(args_base_dir)
    # Prefer /workspace (pod) if it exists
    ws = pathlib.Path("/workspace")
    if ws.exists():
        return ws
    # Fall back to repo root (walk up until we find pyproject.toml or .git)
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Last-resort: repo root is 4 levels up from this file:
    # run_823.py -> issue_823/ -> experiments/ -> explore_persona_space/ -> src/ -> repo_root
    return here.parents[4]


def _ensure_repo_root_on_syspath() -> None:
    """Insert the repo root onto sys.path[0] so `scripts.*` deferred imports work.

    In script mode (python /abs/path/to/run_823.py) sys.path[0] is the script's
    own directory, not the repo root, so `scripts/` (a non-package top-level dir)
    is unreachable.  This helper derives the repo root deterministically from
    __file__ and inserts it once.  Idempotent — safe to call multiple times.
    """
    import sys

    # run_823.py -> issue_823/ -> experiments/ -> explore_persona_space/ -> src/ -> repo_root
    repo_root = pathlib.Path(__file__).resolve().parents[4]
    sentinel = repo_root / "scripts" / "issue779_collect.py"
    if not sentinel.exists():
        raise RuntimeError(
            f"_ensure_repo_root_on_syspath: sentinel {sentinel} not found; "
            f"derived repo_root={repo_root} may be wrong.  "
            "Expected location: src/explore_persona_space/experiments/issue_823/run_823.py"
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
        logger.info("Inserted repo root onto sys.path: %s", repo_root_str)


def sha256_file(path: pathlib.Path) -> str:
    """Compute SHA256 of a file in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Verify HF substrate + reconstruct prompts + alignment gate
# ═══════════════════════════════════════════════════════════════════════════════


def phase0_verify(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> dict:
    """Download bundle, verify sha256, reconstruct prompts. Returns verify record."""
    from huggingface_hub import snapshot_download

    log_phase("p0_substrate")
    logger.info("Phase 0: Verify HF substrate (n_contexts=%d)", n_contexts)

    # Step 0a: download bundle
    logger.info("Downloading pass_b bundle (revision=%s)...", BUNDLE_REPO_REVISION)
    local_dir = snapshot_download(
        repo_id=HF_DATA_REPO_DATA,
        repo_type="dataset",
        revision=BUNDLE_REPO_REVISION,
        allow_patterns=[BUNDLE_PATH_IN_REPO],
        local_dir=str(base_dir / "data" / "issue_823" / "hf_dl"),
    )
    bundle_path = pathlib.Path(local_dir) / BUNDLE_PATH_IN_REPO
    assert bundle_path.exists(), f"Bundle not found at {bundle_path}"

    # sha256 check
    logger.info("Computing sha256 of bundle...")
    actual_sha = sha256_file(bundle_path)
    sha256_ok = actual_sha == BUNDLE_SHA256
    if not sha256_ok:
        raise RuntimeError(f"Bundle sha256 mismatch: expected {BUNDLE_SHA256}, got {actual_sha}")
    logger.info("sha256 PASS: %s", actual_sha)

    # Load and verify shape
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    expected_keys = {"cx_last", "cx_mean", "v_x", "layers", "source", "metadata"}
    assert set(bundle.keys()) == expected_keys, (
        f"Bundle keys {set(bundle.keys())} != expected {expected_keys}"
    )
    assert bundle["v_x"].shape == (N_CONTEXTS_FULL, EXPECTED_LAYERS, EXPECTED_HIDDEN), (
        f"v_x shape {bundle['v_x'].shape}"
    )
    assert bundle["cx_last"].shape == (
        N_CONTEXTS_FULL,
        EXPECTED_LAYERS,
        EXPECTED_HIDDEN,
    ), f"cx_last shape {bundle['cx_last'].shape}"
    assert "prompts" not in bundle, "Unexpected 'prompts' key in bundle"
    logger.info("Bundle shape OK: cx_last %s, v_x %s", bundle["cx_last"].shape, bundle["v_x"].shape)

    # Step 0b: Reconstruct prompts from LMSYS-Chat-1M
    log_phase("p0_prompt_recon")
    logger.info("Reconstructing prompts from LMSYS-Chat-1M (revision=%s)...", LMSYS_REVISION)
    from datasets import load_dataset

    def first_user_turn(conv: dict) -> str:
        """Mirror issue779_collect.py _first_user_turn logic."""
        for msg in conv.get("conversation", []):
            if msg["role"] == "user":
                return msg["content"].strip()
        return ""

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        revision=LMSYS_REVISION,
        token=True,
    )
    prompts: list[str] = []
    for row in ds:
        text = first_user_turn(row)
        if text:
            prompts.append(text)
        if len(prompts) == N_CONTEXTS_FULL:
            break

    assert len(prompts) == N_CONTEXTS_FULL, (
        f"Expected {N_CONTEXTS_FULL} prompts, got {len(prompts)}"
    )
    logger.info("Reconstructed %d prompts", len(prompts))

    # Persist prompts for downstream phases
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps(prompts, default=_json_np))
    logger.info("Prompts saved to %s", prompts_path)

    # Step 0c: Alignment gate (20 spot-check contexts)
    # NOTE: Full alignment gate requires a GPU. In smoke/CPU mode we skip the GPU check
    # and verify only that prompt reconstruction produced valid non-empty strings.
    # The alignment gate is performed in Phase 3's smoke check (which runs on GPU).
    spot_cosines: list[float] = []
    alignment_gate_result = "DEFERRED_TO_PHASE3_GPU"

    rng = np.random.default_rng(0)
    spot_idx = rng.choice(N_CONTEXTS_FULL, size=20, replace=False)
    # Validate prompts are non-empty at spot indices
    for i in spot_idx:
        assert len(prompts[i]) > 0, f"Empty prompt at index {i}"

    logger.info(
        "Alignment gate: %s (spot-check prompts non-empty for all 20 indices)",
        alignment_gate_result,
    )

    # Save bundle path for later phases
    bundle_path_file = base_dir / "data" / "issue_823" / "bundle_path.txt"
    bundle_path_file.write_text(str(bundle_path))

    verify_record = {
        "sha256_ok": sha256_ok,
        "actual_sha256": actual_sha,
        "expected_sha256": BUNDLE_SHA256,
        "key_set": sorted(bundle.keys()),
        "bundle_shape": {
            "cx_last": list(bundle["cx_last"].shape),
            "v_x": list(bundle["v_x"].shape),
        },
        "n_prompts": len(prompts),
        "alignment_gate_result": alignment_gate_result,
        "spot_cosines": spot_cosines,
        "spot_idx": spot_idx.tolist(),
        "bundle_revision": BUNDLE_REPO_REVISION,
        "lmsys_revision": LMSYS_REVISION,
        "smoke": smoke,
        "n_contexts": n_contexts,
        "ts": time.time(),
    }

    out_path = base_dir / "eval_results" / "issue_823" / "phase0_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verify_record, indent=2, default=_json_np))
    logger.info("Phase 0 complete: %s", out_path)
    return verify_record


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0.5: Regenerate Qwen own answers via vLLM (GPU pod)
# ═══════════════════════════════════════════════════════════════════════════════


def phase05_vllm_regen(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Generate arm A' (Qwen own answers) via vLLM."""
    log_phase("p05_vllm_regen")
    logger.info("Phase 0.5: vLLM regeneration (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from vllm import LLM, SamplingParams

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}. Run Phase 0 first."
    prompts: list[str] = json.loads(prompts_path.read_text())
    prompts = prompts[:n_contexts]
    logger.info("Loaded %d prompts", len(prompts))

    # Build chat-formatted prompts (no system prompt, bare context — same as #779 pass_b)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)
    logger.info("Formatted %d prompts for vLLM", len(formatted_prompts))

    # Init vLLM engine (same recipe as issue779_collect.py)
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    logger.info("Initializing vLLM engine (model=%s, chunk_size=%d)...", DEFAULT_MODEL, chunk_size)
    llm = LLM(
        model=DEFAULT_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        seed=42,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Chunked generation (avoids vLLM large-batch deadlock — MEMORY.md gotcha)
    all_texts: list[str] = []
    all_n_tokens: list[int] = []
    for chunk_start in range(0, len(formatted_prompts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(formatted_prompts))
        chunk = formatted_prompts[chunk_start:chunk_end]
        logger.info(
            "[vllm-chunk] generating chunk %d/%d (contexts %d-%d)",
            chunk_start // chunk_size + 1,
            math.ceil(len(formatted_prompts) / chunk_size),
            chunk_start,
            chunk_end - 1,
        )
        outputs = llm.generate(chunk, sampling_params, use_tqdm=False)
        for out in outputs:
            text = out.outputs[0].text
            n_tok = len(out.outputs[0].token_ids)
            all_texts.append(text)
            all_n_tokens.append(n_tok)

    assert len(all_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(all_texts)}"

    # ── vLLM teardown (MUST happen before any HF Transformers load in the same process) ──
    # Per CLAUDE.md gotchas: vLLM destroy_* doesn't reap worker subprocesses.
    # We kill children with psutil to free GPU memory before Phase 3 loads HF model.
    logger.info("Phase 0.5: tearing down vLLM engine...")
    import gc

    import psutil
    import torch

    try:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as _e:
        logger.warning("vLLM del/gc failed (non-fatal): %s", _e)

    # Kill vLLM worker subprocesses that survive after del
    import contextlib

    _current = psutil.Process()
    for _child in _current.children(recursive=True):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            _child.kill()
    logger.info("Phase 0.5: vLLM teardown complete.")

    # Build record
    records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": all_texts[i],
            "n_tokens": all_n_tokens[i],
        }
        for i in range(len(prompts))
    ]

    out_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, default=_json_np))
    logger.info("Phase 0.5: saved %d arm A' answers to %s", len(records), out_path)

    # Write sentinel
    sentinel_path = pathlib.Path("/workspace/logs/issue-823-phase05-done.json")
    write_sentinel(
        sentinel_path,
        {
            "kind": "epm:progress",
            "version": 1,
            "note": f"Phase 0.5 complete: {len(records)} arm A' answers generated",
            "n_records": len(records),
            "smoke": smoke,
        },
    )
    log_phase("p05_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Generate Sonnet answers (VM, CPU + API)
# ═══════════════════════════════════════════════════════════════════════════════


async def phase1_sonnet_gen(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Generate B1 (Sonnet-weird) and B2 (Sonnet-plain) via dispatch_calls."""
    log_phase("p1_sonnet_gen")
    logger.info("Phase 1: Sonnet generation (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}. Run Phase 0 first."
    prompts: list[str] = json.loads(prompts_path.read_text())
    prompts = prompts[:n_contexts]
    logger.info("Loaded %d prompts for Sonnet generation", len(prompts))

    # Build dispatch items for B2 (plain, no system prompt)
    items_b2 = [
        DispatchItem(
            item_id=f"b2_{i}",
            payload={"messages": [{"role": "user", "content": prompts[i]}], "system": None},
        )
        for i in range(len(prompts))
    ]

    # Build dispatch items for B1 (weird, with style instruction)
    b1_system = (
        "Respond in an unusual, stylistically eccentric way — use unexpected structure, "
        "mixed register, and non-standard formatting."
    )
    items_b1 = [
        DispatchItem(
            item_id=f"b1_{i}",
            payload={
                "messages": [{"role": "user", "content": prompts[i]}],
                "system": b1_system,
            },
        )
        for i in range(len(prompts))
    ]

    def _build_request(item: DispatchItem) -> dict:
        """Anthropic Messages.create params for one Sonnet generation call.

        NOTE (plan deviation, recorded): the Anthropic API exposes NO sampling
        seed — the b2 'seed 42' / b1 'seed 43' labels are file/record
        provenance only; text determinism comes from persisting the raw
        completions, not from the sampler.
        """
        params: dict = {
            "model": SONNET_MODEL,
            "max_tokens": SONNET_MAX_TOKENS,
            "temperature": 1.0,
            "messages": item.payload["messages"],
        }
        if item.payload.get("system"):
            params["system"] = item.payload["system"]
        return params

    def _parse_response(text: str) -> str:
        """Identity parse — the generation text IS the result."""
        return text

    all_items = items_b2 + items_b1
    logger.info(
        "Dispatching %d Sonnet calls (B2 x %d + B1 x %d)...",
        len(all_items),
        len(items_b2),
        len(items_b1),
    )

    checkpoint_dir = base_dir / "raw_completions" / "phase1" / "_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = await dispatch_calls(
        all_items,
        model=SONNET_MODEL,
        build_request=_build_request,
        parse_response=_parse_response,
        max_attempts=5,
        checkpoint_dir=checkpoint_dir,
    )

    # Separate B1 and B2 results (results: {item_id: DispatchResult})
    b2_results: dict[int, str] = {}
    b1_results: dict[int, str] = {}
    failed: list[str] = []

    for item_id, res in results.items():
        if res is None or getattr(res, "error", False) or not isinstance(res.result, str):
            failed.append(item_id)
            continue
        text = res.result
        if item_id.startswith("b2_"):
            idx = int(item_id[3:])
            b2_results[idx] = text
        elif item_id.startswith("b1_"):
            idx = int(item_id[3:])
            b1_results[idx] = text

    logger.info(
        "Sonnet generation complete: B2=%d, B1=%d, failed=%d",
        len(b2_results),
        len(b1_results),
        len(failed),
    )

    # Compute common valid index = intersection of A'∩B1∩B2.
    # Arm A' valid set: contexts with non-empty answer from Phase 0.5.
    a_prime_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    if a_prime_path.exists():
        a_prime_recs_raw: list[dict] = json.loads(a_prime_path.read_text())[:n_contexts]
        a_prime_valid_idx: set[int] = {
            i for i, r in enumerate(a_prime_recs_raw) if r.get("answer_text", "")
        }
    else:
        # Phase 0.5 not yet run (e.g. CPU-only smoke): assume all A' are valid
        a_prime_valid_idx = set(range(n_contexts))
        logger.warning(
            "Arm A' file not found; assuming all %d contexts valid for common-drop logic",
            n_contexts,
        )

    b2_valid_idx: set[int] = set(b2_results.keys())
    b1_valid_idx: set[int] = set(b1_results.keys())
    common_valid_idx: set[int] = a_prime_valid_idx & b1_valid_idx & b2_valid_idx
    n_dropped = n_contexts - len(common_valid_idx)

    logger.info(
        "Common valid index: A'=%d, B1=%d, B2=%d, intersection=%d (dropped=%d)",
        len(a_prime_valid_idx),
        len(b1_valid_idx),
        len(b2_valid_idx),
        len(common_valid_idx),
        n_dropped,
    )
    if n_dropped > 50:
        raise RuntimeError(
            f"Too many contexts dropped from common valid set: {n_dropped}/{n_contexts} "
            f"(threshold 50). A'={len(a_prime_valid_idx)}, B1={len(b1_valid_idx)}, "
            f"B2={len(b2_valid_idx)}, intersection={len(common_valid_idx)}. Aborting."
        )

    # Build and persist B2 records — use empty string for non-common contexts
    # so downstream Phase 3 can apply the same common_valid_idx filter.
    b2_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b2_results.get(i, "") if i in common_valid_idx else "",
            "arm": "b2_plain",
            "seed": 42,
            "filled": i in b2_results,
            "in_common_valid": i in common_valid_idx,
        }
        for i in range(len(prompts))
    ]
    b2_path = base_dir / "raw_completions" / "phase1" / "b2_seed42.json"
    b2_path.parent.mkdir(parents=True, exist_ok=True)
    b2_path.write_text(json.dumps(b2_records, indent=2, default=_json_np))
    logger.info("B2 answers saved to %s", b2_path)

    # Build and persist B1 records — same common-valid masking
    b1_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b1_results.get(i, "") if i in common_valid_idx else "",
            "arm": "b1_weird",
            "seed": 43,
            "filled": i in b1_results,
            "in_common_valid": i in common_valid_idx,
        }
        for i in range(len(prompts))
    ]
    b1_path = base_dir / "raw_completions" / "phase1" / "b1_seed43.json"
    b1_path.write_text(json.dumps(b1_records, indent=2, default=_json_np))
    logger.info("B1 answers saved to %s", b1_path)

    # Persist common valid index for Phase 3 to apply consistently
    common_path = base_dir / "raw_completions" / "phase1" / "common_valid_idx.json"
    common_path.write_text(
        json.dumps(
            {
                "common_valid_idx": sorted(common_valid_idx),
                "n_common": len(common_valid_idx),
                "n_dropped": n_dropped,
                "a_prime_valid": len(a_prime_valid_idx),
                "b1_valid": len(b1_valid_idx),
                "b2_valid": len(b2_valid_idx),
            },
            indent=2,
        )
    )
    logger.info("Common valid index saved to %s (%d contexts)", common_path, len(common_valid_idx))

    log_phase("p1_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — batch harvest (crash-recovery path)
# ═══════════════════════════════════════════════════════════════════════════════


def harvest_phase1_batches(
    base_dir: pathlib.Path,
    batch_ids_file: pathlib.Path,
    n_contexts: int = N_CONTEXTS_FULL,
) -> None:
    """Reconstruct Phase 1 output files from already-submitted Anthropic Message Batches.

    Used when the pod that ran phase1_sonnet_gen crashed AFTER all batches
    completed but BEFORE the output files were written — or after the state.json
    was lost.  The function:

    1. Reads batch IDs from a JSON file (list of strings or dict with "batch_ids").
    2. Rebuilds the cid→item_id mapping by recomputing make_custom_id() for all
       known item_ids (b2_0..b2_{N-1}, b1_0..b1_{N-1}).
    3. Fetches results from each batch via the synchronous Anthropic client.
    4. Reconstructs and writes the same three output files as phase1_sonnet_gen:
       raw_completions/phase1/b2_seed42.json
       raw_completions/phase1/b1_seed43.json
       raw_completions/phase1/common_valid_idx.json
    """
    import anthropic
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()

    from explore_persona_space.eval.batch_judge import make_custom_id

    log_phase("p1_harvest")
    logger.info("Phase 1 harvest: reading batch IDs from %s", batch_ids_file)

    # Load batch IDs
    raw = json.loads(batch_ids_file.read_text())
    if isinstance(raw, list):
        batch_ids: list[str] = raw
    elif isinstance(raw, dict) and "batch_ids" in raw:
        batch_ids = raw["batch_ids"]
    else:
        raise ValueError(
            f"batch_ids_file must be a JSON list or dict with 'batch_ids' key, got: {type(raw)}"
        )
    logger.info("Harvesting %d batches: %s", len(batch_ids), batch_ids)

    # Build cid → item_id lookup for all known item_ids
    all_item_ids = [f"b2_{i}" for i in range(n_contexts)] + [f"b1_{i}" for i in range(n_contexts)]
    cid_to_item: dict[str, str] = {make_custom_id(iid): iid for iid in all_item_ids}
    logger.info("Built cid_to_item mapping for %d item_ids", len(cid_to_item))

    # Fetch all batch results using ANTHROPIC_API_KEY (all batches were org=high_prio)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set; required to harvest high_prio batches")
    client = anthropic.Anthropic(api_key=api_key)

    b2_results: dict[int, str] = {}
    b1_results: dict[int, str] = {}
    failed: list[str] = []

    for batch_id in batch_ids:
        logger.info("Fetching results for batch %s ...", batch_id)
        try:
            raw_results = list(client.messages.batches.results(batch_id))
        except Exception as e:
            logger.error("Failed to fetch batch %s: %s", batch_id, e)
            raise

        for result in raw_results:
            cid = result.custom_id
            item_id = cid_to_item.get(cid)
            if item_id is None:
                logger.warning("Unknown custom_id %s in batch %s — skipping", cid, batch_id)
                continue
            rtype = result.result.type
            if rtype == "succeeded":
                text = next((b.text for b in result.result.message.content if b.type == "text"), "")
                if item_id.startswith("b2_"):
                    idx = int(item_id[3:])
                    b2_results[idx] = text
                elif item_id.startswith("b1_"):
                    idx = int(item_id[3:])
                    b1_results[idx] = text
            else:
                logger.warning("Item %s in batch %s: result.type=%s", item_id, batch_id, rtype)
                failed.append(item_id)

        logger.info(
            "Batch %s: B2=%d, B1=%d so far",
            batch_id,
            len(b2_results),
            len(b1_results),
        )

    logger.info(
        "Harvest complete: B2=%d, B1=%d, failed=%d",
        len(b2_results),
        len(b1_results),
        len(failed),
    )
    if failed:
        logger.warning("Failed item_ids: %s", failed[:20])

    # Load prompts for the question field in records
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}"
    prompts: list[str] = json.loads(prompts_path.read_text())[:n_contexts]

    # Compute common valid index (same logic as phase1_sonnet_gen)
    a_prime_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    if a_prime_path.exists():
        a_prime_recs_raw: list[dict] = json.loads(a_prime_path.read_text())[:n_contexts]
        a_prime_valid_idx: set[int] = {
            i for i, r in enumerate(a_prime_recs_raw) if r.get("answer_text", "")
        }
    else:
        a_prime_valid_idx = set(range(n_contexts))
        logger.warning("Arm A' file not found; assuming all %d contexts valid", n_contexts)

    b2_valid_idx: set[int] = set(b2_results.keys())
    b1_valid_idx: set[int] = set(b1_results.keys())
    common_valid_idx: set[int] = a_prime_valid_idx & b1_valid_idx & b2_valid_idx
    n_dropped = n_contexts - len(common_valid_idx)

    logger.info(
        "Common valid index: A'=%d, B1=%d, B2=%d, intersection=%d (dropped=%d)",
        len(a_prime_valid_idx),
        len(b1_valid_idx),
        len(b2_valid_idx),
        len(common_valid_idx),
        n_dropped,
    )
    if n_dropped > 50:
        raise RuntimeError(
            f"Too many contexts dropped from common valid set: {n_dropped}/{n_contexts} "
            f"(threshold 50). Aborting harvest."
        )

    # Write output files (identical schema to phase1_sonnet_gen)
    out_dir = base_dir / "raw_completions" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    b2_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b2_results.get(i, "") if i in common_valid_idx else "",
            "arm": "b2_plain",
            "seed": 42,
            "filled": i in b2_results,
            "in_common_valid": i in common_valid_idx,
        }
        for i in range(len(prompts))
    ]
    b2_path = out_dir / "b2_seed42.json"
    b2_path.write_text(json.dumps(b2_records, indent=2, default=_json_np))
    logger.info(
        "B2 records written: %s (%d total, %d filled)", b2_path, len(b2_records), len(b2_results)
    )

    b1_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b1_results.get(i, "") if i in common_valid_idx else "",
            "arm": "b1_weird",
            "seed": 43,
            "filled": i in b1_results,
            "in_common_valid": i in common_valid_idx,
        }
        for i in range(len(prompts))
    ]
    b1_path = out_dir / "b1_seed43.json"
    b1_path.write_text(json.dumps(b1_records, indent=2, default=_json_np))
    logger.info(
        "B1 records written: %s (%d total, %d filled)", b1_path, len(b1_records), len(b1_results)
    )

    common_path = out_dir / "common_valid_idx.json"
    common_path.write_text(
        json.dumps(
            {
                "common_valid_idx": sorted(common_valid_idx),
                "n_common": len(common_valid_idx),
                "n_dropped": n_dropped,
                "a_prime_valid": len(a_prime_valid_idx),
                "b1_valid": len(b1_valid_idx),
                "b2_valid": len(b2_valid_idx),
                "harvested_from_batches": batch_ids,
            },
            indent=2,
        )
    )
    logger.info("Common valid index written: %s (%d contexts)", common_path, len(common_valid_idx))
    log_phase("p1_harvest_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Build derangement permutation
# ═══════════════════════════════════════════════════════════════════════════════


def phase2_derangement(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Construct fixed-point-free permutation π and build arm C texts."""
    log_phase("p2_derangement")
    logger.info("Phase 2: Derangement (n_contexts=%d)", n_contexts)

    # Load arm A' answers
    a_prime_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    assert a_prime_path.exists(), f"Arm A' file not found: {a_prime_path}. Run Phase 0.5 first."
    a_prime_records: list[dict] = json.loads(a_prime_path.read_text())
    a_prime_records = a_prime_records[:n_contexts]
    logger.info("Loaded %d arm A' records", len(a_prime_records))

    # Build derangement of {0, ..., n_contexts-1}
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_contexts)
    # Swap any fixed points
    for i in range(n_contexts):
        if perm[i] == i:
            j = (i + 1) % n_contexts
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n_contexts)), "Derangement check failed"
    logger.info("Derangement constructed (seed=42, n=%d)", n_contexts)

    # Build arm C records: context i gets arm A' answer from context π(i)
    c_records: dict[str, dict] = {}
    for i in range(n_contexts):
        src = int(perm[i])
        c_records[str(i)] = {
            "source_context": src,
            "answer_text": a_prime_records[src]["answer_text"],
        }

    out_path = base_dir / "raw_completions" / "phase2" / "derangement_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"permutation": perm.tolist(), "contexts": c_records}, indent=2, default=_json_np
        )
    )
    logger.info("Derangement saved to %s", out_path)
    log_phase("p2_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Teacher-forced activation extraction (GPU pod)
# ═══════════════════════════════════════════════════════════════════════════════


def _tf_extract_arm(  # noqa: C901 — batched TF loop; spans/logp reduction done on GPU
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    layers: list[int],
    arm_name: str,
    a_prime_lengths: list[int] | None = None,
    batch_size: int = 8,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Batched teacher-forced extraction for one arm. Returns (v_s, span_lengths, mean_logps).

    v_s shape: (n_contexts, n_layers, hidden_dim) float32.
    For B1/B2 arms, truncates response span to min(own_len, external_len) tokens.
    mean_logps: per-context mean log P of the (truncated) answer span under the
    base model — the plan §5 OOD covariate diagnostic (NaN for skipped contexts).

    Batching: B contexts are padded (LEFT pad), run in one forward pass.
    Span reduction is GPU-resident; only scalar means and (n, n_layers, hidden)
    float32 activations move to CPU. This avoids the batch-1 throughput floor
    (code-reviewer critical: Phase 3 was per-context batch-1; fix here).
    position_ids are passed explicitly to handle left-padding (RoPE correctness).
    """

    n = len(prompts)
    n_layers = len(layers)
    v_s = np.zeros((n, n_layers, EXPECTED_HIDDEN), dtype=np.float32)
    span_lengths: list[int] = [0] * n
    mean_logps: list[float] = [float("nan")] * n

    # Pre-tokenize all (prompt_only, full) pairs to get prompt_len, full_len.
    all_prompt_ids: list[list[int]] = []
    all_full_ids: list[list[int]] = []
    all_resp_start: list[int] = []
    all_resp_end: list[int] = []
    skip_mask: list[bool] = [False] * n  # True = skip (empty answer or empty span)

    for ctx_i in range(n):
        prompt_text = prompts[ctx_i]
        answer_text = answers[ctx_i]

        if not answer_text:
            skip_mask[ctx_i] = True
            continue

        messages = [{"role": "user", "content": prompt_text}]
        prompt_only = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # GENERATION_SUFFIX assert (same as issue779_collect.py:129-130)
        prompt_ids_raw = tokenizer(prompt_only, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        suffix_decode = tokenizer.decode(prompt_ids_raw[-3:])
        assert suffix_decode == GENERATION_SUFFIX, (
            f"[{arm_name}] ctx {ctx_i}: position assert failed: "
            f"{suffix_decode!r} != {GENERATION_SUFFIX!r}"
        )
        prompt_len = len(prompt_ids_raw)

        full_messages = [*messages, {"role": "assistant", "content": answer_text}]
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids_raw = tokenizer(full_text, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        full_len = len(full_ids_raw)

        if full_len <= prompt_len:
            logger.warning(
                "[%s] ctx %d: empty response span (full_len=%d <= prompt_len=%d)",
                arm_name,
                ctx_i,
                full_len,
                prompt_len,
            )
            skip_mask[ctx_i] = True
            continue

        resp_start = prompt_len
        resp_end = full_len

        # Length normalization for B1/B2: truncate to min(own_len, external_len)
        if a_prime_lengths is not None:
            own_len = a_prime_lengths[ctx_i]
            external_len = resp_end - resp_start
            if own_len > 0:
                trunc_len = min(own_len, external_len)
                resp_end = resp_start + trunc_len
            else:
                trunc_len = external_len
            span_lengths[ctx_i] = trunc_len
        else:
            span_lengths[ctx_i] = resp_end - resp_start

        if span_lengths[ctx_i] < 1:
            logger.warning("[%s] ctx %d: span length < 1 after truncation", arm_name, ctx_i)
            skip_mask[ctx_i] = True
            continue

        all_prompt_ids.append(prompt_ids_raw)
        all_full_ids.append(full_ids_raw[:resp_end])  # truncated full seq
        all_resp_start.append(resp_start)
        all_resp_end.append(resp_end)

    # Build an index map from all_prompt_ids position → ctx_i
    valid_indices: list[int] = [i for i in range(n) if not skip_mask[i]]

    # Capture hooks — GPU-resident (hooks capture on GPU, we reduce on GPU)
    # We use a dict keyed by (batch_item, layer_hook_idx) to avoid threading issues.
    captured: dict[int, torch.Tensor] = {}  # li -> (B, seq_len, hidden) on GPU

    def make_hook(li: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Keep on GPU; we slice and reduce after the forward
            captured[li] = hidden.detach()  # (B, seq_len, hidden) bfloat16

        return hook

    handles = []
    for li, layer_idx in enumerate(layers):
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(li))
        handles.append(handle)

    model.eval()
    dev = next(model.parameters()).device

    with torch.no_grad():
        for b_start in range(0, len(valid_indices), batch_size):
            b_end = min(b_start + batch_size, len(valid_indices))
            batch_ctx_idxs = valid_indices[b_start:b_end]
            B = len(batch_ctx_idxs)

            # Map back to all_prompt_ids position
            ai_idxs = list(range(b_start, b_end))  # indices into all_prompt_ids

            batch_full_ids = [all_full_ids[ai] for ai in ai_idxs]
            batch_resp_start = [all_resp_start[ai] for ai in ai_idxs]
            batch_resp_end = [all_resp_end[ai] for ai in ai_idxs]

            # LEFT-pad to the longest sequence in the batch
            max_len = max(len(ids) for ids in batch_full_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids_list = []
            attention_mask_list = []
            pad_offsets: list[int] = []
            for ids in batch_full_ids:
                pad_n = max_len - len(ids)
                padded = [pad_id] * pad_n + ids
                mask = [0] * pad_n + [1] * len(ids)
                input_ids_list.append(padded)
                attention_mask_list.append(mask)
                pad_offsets.append(pad_n)

            input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=dev)  # (B, T)
            attention_mask_t = torch.tensor(
                attention_mask_list, dtype=torch.long, device=dev
            )  # (B, T)

            # Explicit position_ids to handle left-padding correctly (RoPE needs natural indices)
            position_ids_t = (attention_mask_t.cumsum(dim=-1) - 1).clamp(min=0)  # (B, T)

            captured.clear()
            out = model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                position_ids=position_ids_t,
                output_hidden_states=False,
            )

            # GPU-resident span mean-pool and logp reduce
            for j in range(B):
                ctx_i = batch_ctx_idxs[j]
                pad_off = pad_offsets[j]
                r_start = batch_resp_start[j] + pad_off
                r_end = batch_resp_end[j] + pad_off

                # OOD covariate: mean log P of answer span — GPU reduce, scalar to CPU
                logits_j = out.logits[j, r_start - 1 : r_end - 1].float()  # (span, V)
                targets_j = input_ids_t[j, r_start:r_end]  # (span,)
                tok_lp = (
                    torch.log_softmax(logits_j, dim=-1).gather(1, targets_j.unsqueeze(1)).squeeze(1)
                )
                mean_logps[ctx_i] = float(tok_lp.mean().item())
                del logits_j, tok_lp

                # Mean-pool hidden states over response span — GPU reduce
                for li in range(n_layers):
                    if li not in captured:
                        continue
                    hs_j = captured[li][j, r_start:r_end, :]  # (span, hidden) bfloat16
                    if hs_j.shape[0] == 0:
                        continue
                    # Reduce on GPU; move only the (hidden,) mean vector to CPU
                    v_s[ctx_i, li, :] = hs_j.float().mean(dim=0).cpu().numpy()

            del out
            torch.cuda.empty_cache()
            # Clear captured dict for next batch (don't keep holding GPU tensors)
            captured.clear()

            if (b_start // batch_size) % 50 == 0:
                logger.info(
                    "[%s] Processed %d/%d valid contexts", arm_name, b_end, len(valid_indices)
                )

    for h in handles:
        h.remove()
    captured.clear()

    # Check NaN rate
    nan_mask = ~np.isfinite(v_s).all(axis=(1, 2))
    nan_count = int(nan_mask.sum())
    nan_rate = nan_count / n
    if nan_rate > 0.05:
        raise RuntimeError(f"[{arm_name}] NaN rate {nan_rate:.1%} > 5% — aborting (kill criterion)")
    if nan_count > 0:
        logger.warning(
            "[%s] %d contexts have NaN/Inf activations (%.1f%%)",
            arm_name,
            nan_count,
            nan_rate * 100,
        )

    return v_s, span_lengths, mean_logps


def phase3_tf_extract(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Teacher-forced extraction for arms A', B1, B2, C. GPU required."""
    log_phase("p3_tf_extract")
    logger.info("Phase 3: TF extraction (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts not found: {prompts_path}"
    prompts: list[str] = json.loads(prompts_path.read_text())[:n_contexts]

    # Load arm texts
    a_prime_recs = json.loads(
        (base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json").read_text()
    )[:n_contexts]
    b2_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b2_seed42.json").read_text())[
        :n_contexts
    ]
    b1_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b1_seed43.json").read_text())[
        :n_contexts
    ]
    c_data = json.loads(
        (base_dir / "raw_completions" / "phase2" / "derangement_seed42.json").read_text()
    )

    a_prime_texts = [r["answer_text"] for r in a_prime_recs]
    b2_texts = [r["answer_text"] for r in b2_recs]
    b1_texts = [r["answer_text"] for r in b1_recs]
    c_texts = [c_data["contexts"][str(i)]["answer_text"] for i in range(n_contexts)]

    # Apply common_valid_idx mask to ALL arms.  The C arm sources its text from the
    # derangement (A' records via permutation), so an invalid context i can have a
    # NON-EMPTY c_texts[i] (sourced from a valid context perm[i]).  Without this mask
    # _tf_extract_arm would NOT skip invalid contexts for C, producing non-zero vectors
    # that pollute the ridge fits.  Zero out every arm at invalid positions so
    # _tf_extract_arm's skip_mask treats them uniformly as missing.
    common_valid_path = base_dir / "raw_completions" / "phase1" / "common_valid_idx.json"
    assert common_valid_path.exists(), (
        f"common_valid_idx.json not found: {common_valid_path}. Run Phase 1 first."
    )
    common_valid_idx_set: set[int] = set(
        json.loads(common_valid_path.read_text())["common_valid_idx"]
    )
    n_dropped = n_contexts - len(common_valid_idx_set)
    logger.info(
        "[phase3] common_valid_idx: %d valid, %d dropped (will zero out invalid arm texts)",
        len(common_valid_idx_set),
        n_dropped,
    )
    for _i in range(n_contexts):
        if _i not in common_valid_idx_set:
            a_prime_texts[_i] = ""
            b1_texts[_i] = ""
            b2_texts[_i] = ""
            c_texts[_i] = ""  # C arm sourced from derangement — must also mask

    layers = list(range(EXPECTED_LAYERS))
    batch_size = int(os.environ.get("EPM_TF_BATCH_SIZE", "4" if smoke else "8"))

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s on %s...", DEFAULT_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # GPU alignment gate (Phase 0c) — check 20 spot-check contexts
    log_phase("p3_alignment_gate")
    logger.info("Phase 3: Running alignment gate on 20 spot-check contexts...")
    bundle_path = pathlib.Path(
        (base_dir / "data" / "issue_823" / "bundle_path.txt").read_text().strip()
    )
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    cx_last_bundle = bundle["cx_last"].numpy()  # (5000, 28, 3584)

    _ensure_repo_root_on_syspath()
    from scripts.issue779_collect import capture_context_vector  # type: ignore[import]

    rng = np.random.default_rng(0)
    # Sample spot-check indices from the loaded context range, not the full 5000.
    # In smoke mode n_contexts=10; in production n_contexts=N_CONTEXTS_FULL=5000.
    n_spot = min(20, n_contexts)
    spot_idx = rng.choice(n_contexts, size=n_spot, replace=False)
    spot_cosines = []
    alignment_pass = True
    for i in spot_idx:
        messages = [{"role": "user", "content": prompts[i]}]
        result = capture_context_vector(model, tokenizer, messages, layers)
        if result is None:
            logger.warning(
                "Alignment gate: context %d returned None from capture_context_vector", i
            )
            alignment_pass = False
            break
        recomputed = result["last"]  # (n_layers, hidden)
        bundle_row = cx_last_bundle[i]  # (n_layers, hidden)
        # Cosine per layer, then min
        cos_per_layer = []
        for li in range(EXPECTED_LAYERS):
            a = recomputed[li]
            b = bundle_row[li]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            cos_per_layer.append(cos)
        min_cos = min(cos_per_layer)
        spot_cosines.append(
            {"context_id": int(i), "min_cos": min_cos, "cos_by_layer": cos_per_layer}
        )
        if min_cos < 0.999:
            logger.warning("Alignment gate FAIL at context %d: min_cos=%.4f < 0.999", i, min_cos)
            alignment_pass = False

    if not alignment_pass:
        # Hard fail — no fallback. The plan requires a hard-fail when any spot-check
        # context has min_layer_cosine < 0.999.  A NN fallback would mask prompt-
        # reconstruction bugs and silently corrupt all arm activations downstream.
        failing = [r for r in spot_cosines if r["min_cos"] < 0.999]
        raise RuntimeError(
            f"Alignment gate HARD FAIL: {len(failing)} of {len(spot_cosines)} spot-check "
            f"contexts have min_layer_cosine < 0.999. "
            f"First failing: context_id={failing[0]['context_id']}, "
            f"min_cos={failing[0]['min_cos']:.6f}. "
            "Cannot proceed — LMSYS prompt reconstruction does not align with bundle. "
            "Check LMSYS revision pin (LMSYS_REVISION) and first_user_turn() logic."
        )
    else:
        logger.info("Alignment gate PASS: all %d spot checks cosine > 0.999", len(spot_cosines))

    # Persist smoke gate result
    smoke_json = {
        "alignment_gate_pass": alignment_pass,
        "spot_cosines_summary": [
            {"context_id": r["context_id"], "min_cos": r["min_cos"]} for r in spot_cosines
        ],
        "smoke": smoke,
        "n_contexts": n_contexts,
        "ts": time.time(),
    }
    smoke_sentinel = pathlib.Path("/workspace/logs/issue-823-phase3-smoke.json")
    write_sentinel(
        smoke_sentinel,
        {"kind": "epm:progress", "version": 1, "note": "Phase 3 smoke gate", **smoke_json},
    )
    logger.info("Phase 3 smoke sentinel written")

    # Compute arm A' token lengths (for length normalization)
    logger.info("Computing arm A' response token lengths...")
    a_prime_token_lengths: list[int] = []
    for i in range(n_contexts):
        if a_prime_texts[i]:
            toks = tokenizer(a_prime_texts[i], add_special_tokens=False)["input_ids"]
            a_prime_token_lengths.append(len(toks))
        else:
            a_prime_token_lengths.append(0)

    # Extract all arms sequentially to bound HBM
    span_lengths_by_arm: dict[str, list[int]] = {}
    answer_logp_by_arm: dict[str, list[float]] = {}

    arms_to_extract = [
        ("a_prime", a_prime_texts, None),  # no length normalization
        ("b1", b1_texts, a_prime_token_lengths),  # truncate to min(own_len, B1_len)
        ("b2", b2_texts, a_prime_token_lengths),  # truncate to min(own_len, B2_len)
        ("c", c_texts, None),  # Qwen text, no truncation (full span)
    ]

    for arm_name, arm_texts, a_prime_len_list in arms_to_extract:
        log_phase(f"p3_extract_{arm_name}")
        logger.info("Phase 3: Extracting arm %s (batch_size=%d)...", arm_name, batch_size)
        v_s, span_lens, arm_logps = _tf_extract_arm(
            model,
            tokenizer,
            prompts,
            arm_texts,
            layers,
            arm_name,
            a_prime_lengths=a_prime_len_list,
            batch_size=batch_size,
        )
        span_lengths_by_arm[arm_name] = span_lens
        answer_logp_by_arm[arm_name] = arm_logps

        # Save arm tensor immediately (checkpoint-per-phase)
        tensor_path = base_dir / "analysis_tensors" / f"v_{arm_name}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(v_s), str(tensor_path))
        logger.info("Saved arm %s tensor to %s (shape %s)", arm_name, tensor_path, v_s.shape)

        # Free GPU memory
        del v_s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save span lengths
    span_path = base_dir / "analysis_tensors" / "phase3_span_lengths.json"
    span_path.write_text(json.dumps(span_lengths_by_arm, indent=2, default=_json_np))
    logger.info("Span lengths saved to %s", span_path)

    # Save per-arm mean answer-span log-P (OOD covariate diagnostic, plan §5)
    logp_path = base_dir / "analysis_tensors" / "phase3_answer_logp.json"
    logp_path.write_text(json.dumps(answer_logp_by_arm, indent=2, default=_json_np))
    logger.info("Answer-span log-P saved to %s", logp_path)

    # Upload arm tensors to HF data repo
    log_phase("p3_upload")
    logger.info("Uploading arm tensors and span lengths to HF...")
    _upload_arm_tensors(base_dir)

    write_sentinel(
        pathlib.Path("/workspace/logs/issue-823-phase3-done.json"),
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "Phase 3 TF extraction complete",
            "n_contexts": n_contexts,
            "smoke": smoke,
            "ts": time.time(),
        },
    )
    log_phase("p3_done")


def _upload_arm_tensors(base_dir: pathlib.Path) -> None:
    """Upload arm tensors + span lengths to HF data repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    tensors_dir = base_dir / "analysis_tensors"
    files_to_upload = list(tensors_dir.glob("*.pt")) + list(tensors_dir.glob("*.json"))
    logger.info("Uploading %d tensor files to HF...", len(files_to_upload))

    operations = []
    from huggingface_hub import CommitOperationAdd

    for f in files_to_upload:
        path_in_repo = f"{ISSUE_SLUG}/analysis_tensors/{f.name}"
        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))

    api.create_commit(
        repo_id=HF_DATA_REPO_DATA,
        repo_type="dataset",
        commit_message=f"issue 823: arm tensors ({len(operations)} files)",
        operations=operations,
    )
    # Post-commit Hub verification: confirm uploaded files are visible.
    # list_repo_files() has no path_in_repo kwarg — filter client-side by prefix.
    from huggingface_hub import list_repo_files

    expected_paths = {f"{ISSUE_SLUG}/analysis_tensors/{f.name}" for f in files_to_upload}
    prefix = f"{ISSUE_SLUG}/analysis_tensors/"
    hub_files = {
        f
        for f in list_repo_files(repo_id=HF_DATA_REPO_DATA, repo_type="dataset")
        if f.startswith(prefix)
    }
    missing = expected_paths - hub_files
    if missing:
        raise RuntimeError(
            f"HF upload verification FAIL: {len(missing)} tensor files not visible on Hub: "
            f"{sorted(missing)[:3]}..."
        )
    logger.info("Upload complete and Hub-verified: %d files", len(operations))


def _load_common_valid_idx(base_dir: pathlib.Path, n_contexts: int) -> np.ndarray:
    """Load common_valid_idx from Phase 1 output; fall back to all indices if file absent.

    Returns a sorted integer index array of valid context positions.  Logs the drop count
    so callers don't need to repeat the accounting.
    """
    common_valid_path = base_dir / "raw_completions" / "phase1" / "common_valid_idx.json"
    if common_valid_path.exists():
        valid_idx = np.array(
            sorted(json.loads(common_valid_path.read_text())["common_valid_idx"]), dtype=int
        )
        logger.info(
            "[common_valid_idx] %d valid, %d dropped",
            len(valid_idx),
            n_contexts - len(valid_idx),
        )
    else:
        valid_idx = np.arange(n_contexts)
        logger.warning(
            "[common_valid_idx] common_valid_idx.json not found — using all %d contexts (no mask)",
            n_contexts,
        )
    return valid_idx


def _length_r2_correlation(
    span_data: dict[str, list[int]],
    per_ctx_r2_data: dict[str, dict[str, list[float]]],
    valid_idx: np.ndarray,
) -> dict[str, Any]:
    """Length-vs-R² confound diagnostic (phase 5 block 5), valid-context aligned.

    span_data arrays are FULL-length (indexed by original context position;
    phase3_tf_extract allocates ``[0] * n`` and serializes unsliced), while
    per_ctx_r2 arrays contain ONLY the common-valid rows, ordered by sorted
    common_valid_idx (phase4_ridge_refit fancy-indexes every per-context array
    by that sorted index before fitting). Slicing the span arrays by
    ``valid_idx`` therefore index-aligns the two. Raises ValueError on any
    length / index / key inconsistency — never swallows (fail-fast, #913).

    Statistic identity: this computes the pre-registered SIGNED ``len_delta``
    (a_prime - b2) Pearson + Spearman correlation against the per-context R²
    gap; #823's clean-result published the analyzer's absolute-length-delta
    Spearman variant instead (0.0671 / 0.1082 / 0.0556), so a future run's
    non-identical Spearman here is expected, not a regression.

    Returns a per-trait dict matching the historical block-5 output shape; a
    trait gets a ``note`` entry (no raise) when per_ctx_r2 lacks it (the
    legitimate phase-4-not-yet-run ordering state).
    """
    from scipy import stats as scipy_stats

    if "a_prime" not in span_data or "b2" not in span_data:
        raise ValueError(
            "phase3_span_lengths.json present but missing required keys 'a_prime'/'b2' "
            f"(found: {sorted(span_data)}) — corrupt spans artifact"
        )
    ap_full = np.asarray(span_data["a_prime"], dtype=float)
    b2_full = np.asarray(span_data["b2"], dtype=float)
    valid_idx = np.asarray(valid_idx, dtype=int)
    if valid_idx.size < 2:
        raise ValueError(f"valid_idx has {valid_idx.size} element(s); need >= 2 for a correlation")
    if int(valid_idx.min()) < 0 or int(valid_idx.max()) >= min(len(ap_full), len(b2_full)):
        raise ValueError(
            f"valid_idx range [{int(valid_idx.min())}, {int(valid_idx.max())}] out of bounds "
            f"for span arrays (a_prime={len(ap_full)}, b2={len(b2_full)})"
        )
    if not np.all(np.diff(valid_idx) > 0):
        raise ValueError(
            "valid_idx must be strictly increasing (sorted, no duplicates) — "
            "the sorted-common_valid_idx convention IS the alignment proof"
        )
    ap_lens, b2_lens = ap_full[valid_idx], b2_full[valid_idx]
    len_delta = ap_lens - b2_lens

    out: dict[str, Any] = {}
    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        ctx_ap = per_ctx_r2_data.get("A_prime", {}).get(trait)
        ctx_b2 = per_ctx_r2_data.get("B2", {}).get(trait)
        if ctx_ap is None or ctx_b2 is None:
            # per_ctx_r2 not available (e.g. Phase 4 not yet run) — a legitimate
            # ordering state, not an error.
            out[trait] = {
                "read_out_layer": ro,
                "note": "per_ctx_r2 unavailable — run Phase 4 first",
                "mean_ap_len": float(ap_lens.mean()),
                "mean_b2_len": float(b2_lens.mean()),
                "mean_delta": float(len_delta.mean()),
            }
            continue
        r2_ap = np.asarray(ctx_ap, dtype=float)
        r2_b2 = np.asarray(ctx_b2, dtype=float)
        if not (len(r2_ap) == len(r2_b2) == len(valid_idx)):
            raise ValueError(
                f"per_ctx_r2 length mismatch for {trait}: "
                f"A_prime={len(r2_ap)}, B2={len(r2_b2)}, valid_idx={len(valid_idx)}"
            )
        r2_gap = r2_ap - r2_b2  # per-context R2(A') - R2(B2)
        # Pearson
        pearson_r, pearson_p = scipy_stats.pearsonr(len_delta, r2_gap)
        # Spearman
        spearman_r, spearman_p = scipy_stats.spearmanr(len_delta, r2_gap)
        out[trait] = {
            "read_out_layer": ro,
            "n_contexts": len(r2_gap),
            "len_delta_vs_r2_gap": {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            },
            "mean_ap_len": float(ap_lens.mean()),
            "mean_b2_len": float(b2_lens.mean()),
            "mean_delta": float(len_delta.mean()),
        }
        logger.info(
            "Length-R² corr [%s, L%d]: Pearson r=%.3f p=%.3f, Spearman r=%.3f p=%.3f",
            trait,
            ro,
            pearson_r,
            pearson_p,
            spearman_r,
            spearman_p,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Ridge refitting
# ═══════════════════════════════════════════════════════════════════════════════


def _ridge_equivalence_gate(
    cx_last_full: np.ndarray, v_a_prime: np.ndarray, kf, *, device: str
) -> None:
    """Full-size slow-vs-fast ridge parity assert (#823 perf patch).

    One fold of layer 14 on arm A' (n_train ~4000, H=3584): the canonical
    numpy-SVD ``ridge_fit_predict`` vs the Gram-eigh ``ridge_fit_predict_fast``
    on the requested device must agree to <= 1e-8 relative (the #779 gate
    measured ~8e-13). Raises RuntimeError on divergence — never run phase 4 on
    an unverified solver.
    """
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast,
    )

    X = cx_last_full[:, 14, :]
    Y = v_a_prime[:, 14, :]
    train_idx, val_idx = next(iter(kf.split(X)))
    t0 = time.time()
    pred_slow = ridge_fit_predict(X[train_idx], Y[train_idx], X[val_idx])
    t1 = time.time()
    pred_fast = ridge_fit_predict_fast(X[train_idx], Y[train_idx], X[val_idx], device=device)
    t2 = time.time()
    scale = float(np.abs(pred_slow).max()) + 1e-12
    max_rel = float(np.abs(pred_fast - pred_slow).max()) / scale
    logger.info(
        "[ridge-equivalence-gate] max|fast-slow|/max|slow| = %.3e "
        "(slow %.1fs, fast %.1fs, device=%s)",
        max_rel,
        t1 - t0,
        t2 - t1,
        device,
    )
    if max_rel > 1e-8:
        raise RuntimeError(
            f"ridge_fit_predict_fast parity FAIL: max rel diff {max_rel:.3e} > 1e-8 "
            f"(device={device}) — refusing to run phase 4 on an unverified solver"
        )


def phase4_ridge_refit(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Per-arm refit (DV1) and cross-arm transfer (DV3)."""
    log_phase("p4_ridge_refit")
    logger.info("Phase 4: Ridge refitting (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from sklearn.model_selection import KFold

    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict,
        ridge_fit_predict_fast,
    )

    # Load bundle
    bundle_path = pathlib.Path(
        (base_dir / "data" / "issue_823" / "bundle_path.txt").read_text().strip()
    )
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    cx_last_full = bundle["cx_last"].numpy()[:n_contexts]  # (n, 28, 3584)
    v_x_full = bundle["v_x"].numpy()[:n_contexts]  # arm A (harness gate only)
    logger.info("Loaded cx_last shape: %s", cx_last_full.shape)

    # Load arm tensors from disk
    tensors_dir = base_dir / "analysis_tensors"

    def load_arm(name: str) -> np.ndarray:
        p = tensors_dir / f"v_{name}.pt"
        assert p.exists(), f"Arm tensor not found: {p}. Run Phase 3 first."
        t = torch.load(str(p), map_location="cpu")
        return t.numpy()[:n_contexts]

    v_a_prime = load_arm("a_prime")
    v_b1 = load_arm("b1")
    v_b2 = load_arm("b2")
    v_c = load_arm("c")

    logger.info(
        "Loaded arm tensors: A' %s, B1 %s, B2 %s, C %s",
        v_a_prime.shape,
        v_b1.shape,
        v_b2.shape,
        v_c.shape,
    )

    # ── Apply common_valid_idx mask (Phase 1 output) ──────────────────────────
    # Zero-vector rows from invalid Sonnet contexts (empty answer_text → skip_mask in
    # _tf_extract_arm) must be excluded from the ridge fits.  The C arm in Phase 3 is
    # sourced from the derangement, so its skip_mask does NOT track A'/B1/B2 validity —
    # we must restrict by the intersection written in Phase 1.
    valid_idx = _load_common_valid_idx(base_dir, n_contexts)
    # Slice all per-context arrays to valid rows only.
    cx_last_full = cx_last_full[valid_idx]
    v_x_full = v_x_full[valid_idx]
    v_a_prime = v_a_prime[valid_idx]
    v_b1 = v_b1[valid_idx]
    v_b2 = v_b2[valid_idx]
    v_c = v_c[valid_idx]
    # Update n_contexts for all downstream size-dependent allocations (ctx_sse/ctx_sst etc.)
    n_contexts = len(valid_idx)

    # Arm A is used ONLY for the harness reproduce-gate (DV0 check).
    # The DV1 per-arm refit and DV3 cross-arm transfer operate on {A', B1, B2, C}.
    # Including arm A in the refit loop would make arm A both a reproduce-gate AND a
    # new DV target — the plan §3 explicitly restricts A to the gate role.
    refit_targets = {
        "A_prime": v_a_prime,
        "B1": v_b1,
        "B2": v_b2,
        "C": v_c,
    }
    # v_x_full (arm A) is kept separate for the harness gate and A-vs-A' diagnostic.

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # ── #823 perf patch (user-directed, 2026-07-02): dedupe + fast solver ─────
    # The original loops executed 3780 serial SVD-path ridge fits where only 700
    # unique fits exist: (a) the per-trait loops repeated identical (arm, layer,
    # fold) fits 3x — the fit does not depend on trait, only the per-context
    # accumulation's read-out layer does; (b) Computation B re-fit (X, v_A') per
    # (s_prime, trait, layer, fold), which is IDENTICAL to Computation A's A'
    # fit under the same deterministic KFold(random_state=0) splits, so the
    # transfer leg needs only RESCORING of the A' predictions. Solver:
    # ridge_fit_predict_fast (the #779 Gram-eigh twin), parity-gated here at
    # full size against the canonical SVD path before any fit is trusted.
    _ridge_device = os.environ.get("EPS_RIDGE_DEVICE", "cpu")
    _ridge_solver = os.environ.get("EPS_RIDGE_SOLVER", "canonical")
    if _ridge_solver == "fast":
        # OPT-IN Gram-eigh fast solver. Live parity slice on the #823 inputs
        # (2026-07-02, n_tr~3998, H=3584, 2 layers x 5 folds): max rel diff
        # ~1.7e-5 vs the canonical SVD path — FAILED the <=1e-8 ship gate (the
        # #779 8e-13 figure was measured at n=500, where the squared-condition
        # Gram is benign). Therefore NOT the default; the full-size gate below
        # hard-raises on divergence, so this branch cannot silently ship
        # off-parity numbers.
        _ridge_equivalence_gate(cx_last_full, v_a_prime, kf, device=_ridge_device)

        def _fit(x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray) -> np.ndarray:
            return ridge_fit_predict_fast(x_tr, y_tr, x_ev, device=_ridge_device)
    else:
        # Canonical numpy-SVD path — the dedupe below reuses IDENTICAL
        # deterministic computations, so outputs are bit-identical to the
        # unpatched loops.
        _fit = ridge_fit_predict

    # Fold splits depend only on n (KFold re-derives from random_state on every
    # .split call), so materialize once and reuse everywhere.
    folds = list(kf.split(cx_last_full[:, 0, :]))
    ro_layers_needed = sorted({READ_OUT_LAYERS[t] for t in TRAITS})

    # ── Computation A: per-arm refit ──────────────────────────────────────────
    log_phase("p4_refit")
    logger.info("Phase 4 Computation A: per-arm refit (arms: A', B1, B2, C)...")
    r2_refit: dict[str, dict[str, dict]] = {}

    # per_ctx_r2[arm][trait] = np.ndarray shape (n_contexts,) — per-context R²
    # at the read-out layer, assembled from the 5 held-out folds.
    # This is the primary uncertainty estimate surface (context-level bootstrap CI).
    per_ctx_r2: dict[str, dict[str, np.ndarray]] = {}

    # Transfer (Computation B) accumulator, filled during the A' refit pass:
    # r2_transfer_vals[s_prime][layer_idx] = [5 fold R²s], scored from the SAME
    # A' predictions the refit uses (fit_arm is always A' for transfer).
    r2_transfer_vals: dict[str, list[list[float]]] = {
        sp: [[] for _ in range(EXPECTED_LAYERS)] for sp in refit_targets
    }

    for s, Y_target_s in refit_targets.items():
        r2_folds_per_layer: list[list[float]] = []
        # Accumulate per-context (ss_res, ss_tot) at EVERY needed read-out layer
        # in one pass (the fit does not depend on trait).
        ctx_sse = {ro: np.zeros(n_contexts) for ro in ro_layers_needed}
        ctx_sst = {ro: np.zeros(n_contexts) for ro in ro_layers_needed}
        for layer_idx in range(EXPECTED_LAYERS):
            X = cx_last_full[:, layer_idx, :]  # (n, 3584)
            Y = Y_target_s[:, layer_idx, :]  # (n, 3584)
            r2_folds: list[float] = []
            for train_idx, val_idx in folds:
                Y_pred = _fit(X[train_idx], Y[train_idx], X[val_idx])
                ss_res = float(np.sum((Y[val_idx] - Y_pred) ** 2))
                ss_tot = float(np.sum((Y[val_idx] - Y[val_idx].mean(0)) ** 2))
                r2_folds.append(1.0 - ss_res / (ss_tot + 1e-12))
                # Accumulate per-context at the read-out layers only (saves memory)
                if layer_idx in ctx_sse:
                    res_ctx = ((Y[val_idx] - Y_pred) ** 2).sum(axis=1)  # (val_n,)
                    tot_ctx = ((Y[val_idx] - Y[val_idx].mean(0)) ** 2).sum(axis=1)  # (val_n,)
                    ctx_sse[layer_idx][val_idx] += res_ctx
                    ctx_sst[layer_idx][val_idx] += tot_ctx
                if s == "A_prime":
                    # Computation B (cross-arm transfer) rescoring: this Y_pred IS
                    # the transfer prediction (fit on A', same folds) — score it
                    # against every target arm here instead of re-fitting later.
                    for sp, Y_sp_all in refit_targets.items():
                        Y_sp = Y_sp_all[:, layer_idx, :]
                        ss_res_t = float(np.sum((Y_sp[val_idx] - Y_pred) ** 2))
                        ss_tot_t = float(np.sum((Y_sp[val_idx] - Y_sp[val_idx].mean(0)) ** 2))
                        r2_transfer_vals[sp][layer_idx].append(1.0 - ss_res_t / (ss_tot_t + 1e-12))
            r2_folds_per_layer.append(r2_folds)
        r2_refit[s] = {}
        per_ctx_r2[s] = {}
        for trait in TRAITS:
            ro_layer = READ_OUT_LAYERS[trait]
            r2_refit[s][trait] = {
                "r2_by_layer": r2_folds_per_layer,  # list[28][5]; identical across traits
                "fit_arm": s,
                "score_arm": s,  # mechanizable: refit rows have fit_arm == score_arm
            }
            # Per-context R² at read-out layer (from accumulated folds)
            per_ctx_r2[s][trait] = 1.0 - ctx_sse[ro_layer] / (ctx_sst[ro_layer] + 1e-12)
        logger.info("Arm %s refit complete (layers %d, folds 5)", s, EXPECTED_LAYERS)

    # Compute arm A (bundle) R² separately for the harness reproduce-gate.
    # Arm A is NOT in refit_targets (per plan §3 — A is the gate role only),
    # so we run a dedicated 5-fold refit loop for the gate check. The fit does
    # not depend on trait, so compute the 28x5 grid ONCE and share it (#823
    # perf patch — the original re-ran the identical grid once per trait).
    log_phase("p4_arm_a_gate_refit")
    logger.info("Phase 4: Computing arm A refit for harness gate...")
    r2_folds_per_layer_a: list[list[float]] = []
    for layer_idx in range(EXPECTED_LAYERS):
        X_a = cx_last_full[:, layer_idx, :]
        Y_a = v_x_full[:, layer_idx, :]
        r2_folds_a: list[float] = []
        for train_idx_a, val_idx_a in folds:
            Y_pred_a = _fit(X_a[train_idx_a], Y_a[train_idx_a], X_a[val_idx_a])
            ss_res_a = float(np.sum((Y_a[val_idx_a] - Y_pred_a) ** 2))
            ss_tot_a = float(np.sum((Y_a[val_idx_a] - Y_a[val_idx_a].mean(0)) ** 2))
            r2_folds_a.append(1.0 - ss_res_a / (ss_tot_a + 1e-12))
        r2_folds_per_layer_a.append(r2_folds_a)
    r2_arm_a: dict[str, dict] = {
        trait: {
            "r2_by_layer": r2_folds_per_layer_a,
            "fit_arm": "A",
            "score_arm": "A",
        }
        for trait in TRAITS
    }
    logger.info("Arm A gate refit complete.")

    # Build combined dict for gate and diagnostic (arm A separate from the DV refit dict).
    r2_refit_for_gate = {"A": r2_arm_a, **r2_refit}

    # Harness reproduce-gate: arm A per-arm refit must match #779 within ±0.01.
    # Skip in smoke mode (reference file not available locally during smoke).
    if not smoke:
        _check_harness_reproduce_gate(r2_refit_for_gate, base_dir)
    else:
        logger.info("Harness reproduce-gate SKIPPED (smoke mode — reference not pre-staged)")

    # A-vs-A' consistency diagnostic
    a_vs_a_prime_diag = _compute_a_vs_a_prime_diag(v_x_full, v_a_prime, r2_refit_for_gate)

    # ── Computation B: cross-arm transfer ────────────────────────────────────
    # Assembled from the A' predictions computed during Computation A (see the
    # rescoring branch there): the transfer fit is ALWAYS (X, v_A') on the same
    # deterministic folds, identical to the A' refit fit — zero additional fits
    # (#823 perf patch). Fit ONLY on arm A' — never on v_B1, v_B2, or v_C.
    log_phase("p4_transfer")
    logger.info("Phase 4 Computation B: cross-arm transfer (fit on A')...")
    r2_transfer: dict[str, dict[str, dict]] = {}

    for s_prime in refit_targets:
        assert all(len(f) == 5 for f in r2_transfer_vals[s_prime]), (
            f"transfer rescoring incomplete for {s_prime}: "
            f"{[len(f) for f in r2_transfer_vals[s_prime]]}"
        )
        r2_transfer[s_prime] = {
            trait: {
                "r2_by_layer": r2_transfer_vals[s_prime],
                "fit_arm": "A_prime",  # always A' for transfer — fit_arm != score_arm → transfer
                "score_arm": s_prime,
            }
            for trait in TRAITS
        }
        logger.info("Transfer arm %s complete (fit=A', score=%s)", s_prime, s_prime)

    # ── Statistical tests ─────────────────────────────────────────────────────
    stats = _compute_stats(r2_refit, r2_transfer, per_ctx_r2)

    # ── Persist results ───────────────────────────────────────────────────────
    # Serialize per_ctx_r2 arrays as lists for JSON storage (used by Phase 5
    # for the length-R² numeric correlation diagnostic).
    per_ctx_r2_serialized: dict[str, dict[str, list[float]]] = {
        arm: {trait: arr.tolist() for trait, arr in by_trait.items()}
        for arm, by_trait in per_ctx_r2.items()
    }
    result = {
        "refit": r2_refit,
        "transfer": r2_transfer,
        "stats": stats,
        "a_vs_a_prime_diagnostic": a_vs_a_prime_diag,
        "per_ctx_r2": per_ctx_r2_serialized,
        "n_contexts": n_contexts,
        "smoke": smoke,
        "read_out_layers": READ_OUT_LAYERS,
        "ts": time.time(),
    }
    out_path = base_dir / "eval_results" / "issue_823" / "ridge_r2_by_arm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_np))
    logger.info("Phase 4 complete: %s", out_path)
    log_phase("p4_done")


def _check_harness_reproduce_gate(r2_refit: dict, base_dir: pathlib.Path) -> None:
    """Check arm A per-arm refit matches #779 within ±0.01.

    Raises RuntimeError on delta > 0.01 — a silent warning would mask a ridge-recipe
    drift or bundle SHA mismatch before it propagates to DV1 and the H1/H2/H3 verdicts.
    If the #779 reference file is absent (not yet downloaded), raises RuntimeError so
    callers know to pre-stage it rather than silently skipping the gate.
    """
    ref_path = base_dir / "eval_results" / "issue_779" / "percontext_recon.json"
    if not ref_path.exists():
        raise RuntimeError(
            f"Harness reproduce-gate: #779 reference not found at {ref_path}. "
            "Pre-stage the file before running Phase 4. "
            "Use --skip-harness-gate to explicitly bypass (smoke only)."
        )
    ref = json.loads(ref_path.read_text())

    # Assert reference schema once — avoids silent wrong-key reads downstream.
    # Actual top-level keys: read_out_layers, metadata, read1_heldout_recon, read2_projection_recon
    # Means live at: ref["read1_heldout_recon"]["heldout_r2_vs_layer"][str(layer)]["mean"]
    _expected_ref_keys = {"read1_heldout_recon", "read_out_layers"}
    _missing = _expected_ref_keys - set(ref.keys())
    if _missing:
        raise RuntimeError(
            f"Harness reproduce-gate: #779 reference at {ref_path} has unexpected schema. "
            f"Expected top-level keys {_expected_ref_keys}, missing: {_missing}. "
            f"Found keys: {list(ref.keys())}. "
            "Check BUNDLE_REPO_REVISION and the correct percontext_recon.json path."
        )
    ref_heldout = ref["read1_heldout_recon"]["heldout_r2_vs_layer"]

    for trait in TRAITS:
        ro_layer = READ_OUT_LAYERS[trait]
        arm_a_r2 = float(np.mean(r2_refit["A"][trait]["r2_by_layer"][ro_layer]))
        layer_key = str(ro_layer)
        if layer_key not in ref_heldout or "mean" not in ref_heldout[layer_key]:
            raise RuntimeError(
                f"Harness reproduce-gate: layer '{layer_key}' not found in "
                f"ref['read1_heldout_recon']['heldout_r2_vs_layer'] at {ref_path}. "
                f"Available layers: {list(ref_heldout.keys())[:10]}..."
            )
        ref_r2 = float(ref_heldout[layer_key]["mean"])
        delta = abs(arm_a_r2 - ref_r2)
        if delta > 0.01:
            raise RuntimeError(
                f"Harness reproduce-gate HARD FAIL: arm A '{trait}' R²={arm_a_r2:.4f} "
                f"vs #779 reference {ref_r2:.4f} (delta={delta:.4f} > 0.01). "
                "Ridge recipe or bundle SHA drift suspected — check fit_h.py params and "
                "BUNDLE_SHA256 constant."
            )
        logger.info(
            "Harness reproduce-gate PASS: arm A %s R²=%.4f (ref=%.4f, delta=%.4f)",
            trait,
            arm_a_r2,
            ref_r2,
            delta,
        )


def _compute_a_vs_a_prime_diag(v_a: np.ndarray, v_a_prime: np.ndarray, r2_refit: dict) -> dict:
    """Compute per-context cosine similarity between arm A and arm A' at read-out layers."""
    diag: dict[str, Any] = {}
    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        a_vec = v_a[:, ro, :]  # (n, hidden)
        ap_vec = v_a_prime[:, ro, :]
        # Normalized cosine per context
        a_norm = np.linalg.norm(a_vec, axis=1, keepdims=True) + 1e-9
        ap_norm = np.linalg.norm(ap_vec, axis=1, keepdims=True) + 1e-9
        cos = np.sum((a_vec / a_norm) * (ap_vec / ap_norm), axis=1)  # (n,)
        mean_cos = float(cos.mean())
        # R² delta
        r2_a = float(np.mean(r2_refit["A"][trait]["r2_by_layer"][ro]))
        r2_ap = float(np.mean(r2_refit["A_prime"][trait]["r2_by_layer"][ro]))
        diag[trait] = {
            "mean_cos": mean_cos,
            "std_cos": float(cos.std()),
            "read_out_layer": ro,
            "r2_a": r2_a,
            "r2_a_prime": r2_ap,
            "r2_delta": abs(r2_a - r2_ap),
        }
        logger.info(
            "A-vs-A' [%s, L%d]: mean_cos=%.4f, R²_A=%.4f, R²_A'=%.4f, delta=%.4f",
            trait,
            ro,
            mean_cos,
            r2_a,
            r2_ap,
            abs(r2_a - r2_ap),
        )
        if mean_cos < 0.99:
            logger.warning("A-vs-A' drift WARNING [%s]: mean cos=%.4f < 0.99", trait, mean_cos)
        if abs(r2_a - r2_ap) > 0.02:
            logger.warning(
                "A-vs-A' drift WARNING [%s]: R² delta=%.4f > 0.02", trait, abs(r2_a - r2_ap)
            )
    return diag


def _compute_stats(
    r2_refit: dict,
    r2_transfer: dict,
    per_ctx_r2: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Compute paired t-tests, context-level bootstrap CIs, and H1/H2/H3 verdicts.

    Args:
        r2_refit: Per-arm fold-level R² dict (arm → trait → {"r2_by_layer": ...}).
        r2_transfer: Cross-arm fold-level R² dict (same shape).
        per_ctx_r2: Per-context R² at read-out layer, accumulated from 5-fold CV
            (arm → trait → np.ndarray of shape (n_contexts,)).  Used for the
            primary plan §6 context-level bootstrap CI (n≈5000, 1000 iterations).

    Returns:
        Dict with per-trait stats (paired t, context-level bootstrap CI, H1/H2/H3).
    """
    from scipy import stats as scipy_stats

    BONFERRONI_ALPHA = 0.05 / 3
    T_CRIT_DF4 = 3.495  # t_crit(df=4, Bonferroni alpha=0.017 two-tailed)
    N_BOOTSTRAP = 1000

    result: dict[str, Any] = {}

    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        # Per-arm fold R² at read-out layer (for paired t and point estimates)
        r2_ap = np.array(r2_refit["A_prime"][trait]["r2_by_layer"][ro])  # (5,)
        r2_b2 = np.array(r2_refit["B2"][trait]["r2_by_layer"][ro])
        r2_c = np.array(r2_refit["C"][trait]["r2_by_layer"][ro])

        def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
            """Paired t-test with explicit df = n-1 guard."""
            diff = a - b
            n = len(diff)
            df = n - 1  # guard: df = len(paired_values) - 1
            assert df == n - 1, f"df guard: df={df}, n={n}"
            mean_diff = diff.mean()
            se_diff = diff.std(ddof=1) / math.sqrt(n)
            t_stat = mean_diff / (se_diff + 1e-12)
            p_val = float(2 * scipy_stats.t.sf(abs(t_stat), df=df))
            cohen_d = float(mean_diff / (diff.std(ddof=1) + 1e-12))
            return {
                "mean_diff": float(mean_diff),
                "t_stat": float(t_stat),
                "df": df,
                "p_val": p_val,
                "p_bonferroni": min(1.0, p_val * 3),
                "significant_bonferroni": abs(t_stat) > T_CRIT_DF4,
                "cohen_d": cohen_d,
            }

        ap_vs_b2 = paired_t(r2_ap, r2_b2)
        ap_vs_c = paired_t(r2_ap, r2_c)
        b2_vs_c = paired_t(r2_b2, r2_c)

        delta_ap_b2 = float(r2_ap.mean() - r2_b2.mean())
        delta_ap_c = float(r2_ap.mean() - r2_c.mean())
        delta_b2_c = float(r2_b2.mean() - r2_c.mean())

        # ── Context-level bootstrap CI (plan §6 PRIMARY uncertainty estimate) ──
        # Resample n_contexts rows (with replacement) from per-context R²
        # accumulated during 5-fold CV in phase4_ridge_refit.
        ctx_ap = per_ctx_r2.get("A_prime", {}).get(trait)
        ctx_b2 = per_ctx_r2.get("B2", {}).get(trait)
        ctx_c = per_ctx_r2.get("C", {}).get(trait)

        if ctx_ap is not None and ctx_b2 is not None and ctx_c is not None:
            n_ctx = len(ctx_ap)
            rng = np.random.default_rng(0)
            boot_ap_b2_ctx: list[float] = []
            boot_ap_c_ctx: list[float] = []
            boot_b2_c_ctx: list[float] = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, n_ctx, size=n_ctx)
                boot_ap_b2_ctx.append(float(ctx_ap[idx].mean() - ctx_b2[idx].mean()))
                boot_ap_c_ctx.append(float(ctx_ap[idx].mean() - ctx_c[idx].mean()))
                boot_b2_c_ctx.append(float(ctx_b2[idx].mean() - ctx_c[idx].mean()))

            ci_ap_b2 = {
                "lo": float(np.percentile(boot_ap_b2_ctx, 2.5)),
                "hi": float(np.percentile(boot_ap_b2_ctx, 97.5)),
                "n_contexts": n_ctx,
                "method": "context_level_bootstrap",
            }
            ci_ap_c = {
                "lo": float(np.percentile(boot_ap_c_ctx, 2.5)),
                "hi": float(np.percentile(boot_ap_c_ctx, 97.5)),
                "n_contexts": n_ctx,
                "method": "context_level_bootstrap",
            }
            ci_b2_c = {
                "lo": float(np.percentile(boot_b2_c_ctx, 2.5)),
                "hi": float(np.percentile(boot_b2_c_ctx, 97.5)),
                "n_contexts": n_ctx,
                "method": "context_level_bootstrap",
            }
            logger.info(
                "Context-level bootstrap CI [%s]: Δ(A'-B2)=%.4f [%.4f, %.4f] n_ctx=%d",
                trait,
                delta_ap_b2,
                ci_ap_b2["lo"],
                ci_ap_b2["hi"],
                n_ctx,
            )
        else:
            # Fallback: fold-level bootstrap (conservative proxy)
            logger.warning(
                "per_ctx_r2 missing for trait=%s — falling back to fold-level bootstrap CI", trait
            )
            rng = np.random.default_rng(0)
            boot_ap_b2_fold: list[float] = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, 5, size=5)
                boot_ap_b2_fold.append(float(r2_ap[idx].mean() - r2_b2[idx].mean()))
            lo, hi = np.percentile(boot_ap_b2_fold, [2.5, 97.5])
            ci_ap_b2 = {
                "lo": float(lo),
                "hi": float(hi),
                "n_contexts": 5,
                "method": "fold_level_bootstrap_fallback",
            }
            ci_ap_c = {"lo": float("nan"), "hi": float("nan"), "method": "not_computed"}
            ci_b2_c = {"lo": float("nan"), "hi": float("nan"), "method": "not_computed"}

        # H1/H2/H3 determination (point estimates drive verdicts; plan §3 table)
        # Comparison              H1        H2        H3
        # R2_Ap - R2_B2         > 0.05   <= 0.03   <= 0.05
        # R2_Ap - R2_C          > 0.10   <= 0.06   >  0.05
        # R2_B2 - R2_C          (any)    <= 0.03   >  0.03
        # R2_C absolute         < 0.05   >= 0.02   <  0.05
        # H2 delta rows are two-sided (|delta| <=); H1/H3 delta rows are signed.
        r2_c_mean = float(r2_c.mean())
        h1 = (
            delta_ap_b2 > 0.05
            and delta_ap_c > 0.10
            # (any) for R2_B2 - R2_C in H1
            and r2_c_mean < 0.05
        )
        # H2: per-comparison thresholds from plan §3 (NOT uniform 0.03)
        h2 = (
            abs(delta_ap_b2) <= 0.03
            and abs(delta_ap_c) <= 0.06
            and abs(delta_b2_c) <= 0.03
            and r2_c_mean >= 0.02
        )
        h3 = delta_ap_b2 <= 0.05 and delta_ap_c > 0.05 and delta_b2_c > 0.03 and r2_c_mean < 0.05
        # Falsifier: R²_C > R²_B2 → artifact (Qwen-text TF effect, not science signal)
        falsifier = delta_b2_c < 0  # i.e. r2_c > r2_b2

        if falsifier:
            verdict = "ARTIFACT_R2C_GT_R2B2"
        elif h1:
            verdict = "H1"
        elif h2:
            verdict = "H2"
        elif h3:
            verdict = "H3"
        else:
            verdict = "AMBIGUOUS"

        result[trait] = {
            "read_out_layer": ro,
            "mean_r2": {
                "A_prime": float(r2_ap.mean()),
                "B2": float(r2_b2.mean()),
                "C": float(r2_c.mean()),
            },
            "paired_t": {
                "A_prime_vs_B2": ap_vs_b2,
                "A_prime_vs_C": ap_vs_c,
                "B2_vs_C": b2_vs_c,
            },
            "bootstrap_ci_context_level": {
                "A_prime_vs_B2": ci_ap_b2,
                "A_prime_vs_C": ci_ap_c,
                "B2_vs_C": ci_b2_c,
            },
            "delta_r2": {
                "A_prime_vs_B2": delta_ap_b2,
                "A_prime_vs_C": delta_ap_c,
                "B2_vs_C": delta_b2_c,
            },
            "hypothesis_verdict": verdict,
            "h1_thresholds_met": h1,
            "h2_thresholds_met": h2,
            "h3_thresholds_met": h3,
            "falsifier_artifact": falsifier,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "t_crit_df4": T_CRIT_DF4,
        }
        logger.info(
            "Stats [%s, L%d]: R²_A'=%.4f, R²_B2=%.4f, R²_C=%.4f → verdict=%s",
            trait,
            ro,
            r2_ap.mean(),
            r2_b2.mean(),
            r2_c.mean(),
            verdict,
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Validity diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def phase5_validity_diag(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Run validity diagnostics: text decorrelation, OOD log-P, activation cosine, lengths."""
    log_phase("p5_validity")
    logger.info("Phase 5: Validity diagnostics (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    # Load arm texts
    a_prime_recs = json.loads(
        (base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json").read_text()
    )[:n_contexts]
    b2_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b2_seed42.json").read_text())[
        :n_contexts
    ]

    a_prime_texts = [r["answer_text"] for r in a_prime_recs]
    b2_texts = [r["answer_text"] for r in b2_recs]

    # 1. Text decorrelation (TF-idf cosine)
    logger.info("Phase 5.1: TF-idf cosine B2 vs A'...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    valid_pairs = [(a, b) for a, b in zip(a_prime_texts, b2_texts, strict=True) if a and b]
    a_texts_valid = [p[0] for p in valid_pairs]
    b_texts_valid = [p[1] for p in valid_pairs]

    tfidf = TfidfVectorizer(max_features=10000)
    all_texts = a_texts_valid + b_texts_valid
    tfidf_mat = tfidf.fit_transform(all_texts)
    n_valid = len(a_texts_valid)
    tfidf_a = tfidf_mat[:n_valid]
    tfidf_b = tfidf_mat[n_valid:]
    cos_per_ctx = np.array(cosine_similarity(tfidf_a, tfidf_b).diagonal())
    frac_high = float((cos_per_ctx > 0.8).mean())
    logger.info(
        "TF-idf cosine: mean=%.4f, std=%.4f, frac>0.8=%.1f%%",
        cos_per_ctx.mean(),
        cos_per_ctx.std(),
        frac_high * 100,
    )
    if frac_high > 0.20:
        logger.warning(
            "Text decorrelation WARNING: %.1f%% contexts have cos(B2, A') > 0.8 — "
            "arm distinction may collapse",
            frac_high * 100,
        )

    # 2. Length distribution
    logger.info("Phase 5.2: Length distribution...")
    span_path = base_dir / "analysis_tensors" / "phase3_span_lengths.json"
    span_data = json.loads(span_path.read_text()) if span_path.exists() else {}

    def describe_lengths(lens: list[int]) -> dict:
        arr = np.array(lens)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "frac_short_lt10": float((arr < 10).mean()),
        }

    length_stats: dict[str, Any] = {}
    for arm_name in ["a_prime", "b1", "b2", "c"]:
        if arm_name in span_data:
            length_stats[arm_name] = describe_lengths(span_data[arm_name])
    logger.info("Length stats: %s", {k: v["mean"] for k, v in length_stats.items()})

    # 3. Per-context activation cosine at read-out layers
    logger.info("Phase 5.3: Per-context activation cosine v_B2 vs v_A'...")
    tensors_dir = base_dir / "analysis_tensors"
    cos_by_trait: dict[str, Any] = {}
    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        v_ap_path = tensors_dir / "v_a_prime.pt"
        v_b2_path = tensors_dir / "v_b2.pt"
        if v_ap_path.exists() and v_b2_path.exists():
            v_ap = torch.load(str(v_ap_path), map_location="cpu").numpy()[:n_contexts, ro, :]
            v_b2 = torch.load(str(v_b2_path), map_location="cpu").numpy()[:n_contexts, ro, :]
            nap = np.linalg.norm(v_ap, axis=1, keepdims=True) + 1e-9
            nb2 = np.linalg.norm(v_b2, axis=1, keepdims=True) + 1e-9
            cos = np.sum((v_ap / nap) * (v_b2 / nb2), axis=1)
            cos_by_trait[trait] = {
                "read_out_layer": ro,
                "mean_cos": float(cos.mean()),
                "std_cos": float(cos.std()),
                "min_cos": float(cos.min()),
                "max_cos": float(cos.max()),
            }
            logger.info(
                "Activation cosine v_B2 vs v_A' [%s, L%d]: mean=%.4f, std=%.4f",
                trait,
                ro,
                cos.mean(),
                cos.std(),
            )

    # 4. OOD log-P check (per-arm mean answer-span log P under base Qwen).
    # Computed during Phase 3 TF extraction (same forward, scalar per context)
    # and persisted to phase3_answer_logp.json; here we summarize per arm.
    logp_distributions: dict[str, Any] = {}
    logp_path = base_dir / "analysis_tensors" / "phase3_answer_logp.json"
    if logp_path.exists():
        logp_by_arm = json.loads(logp_path.read_text())
        for arm, vals in logp_by_arm.items():
            arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
            logp_distributions[arm] = {
                "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None,
                "n_finite": int(arr.size),
                "n_total": len(vals),
            }
    else:
        logp_distributions["note"] = f"MISSING — {logp_path} not found (Phase 3 not yet run)"

    # 5. Length-R² correlation (if Phase 4 results exist)
    # Uses per-context R² values persisted in ridge_r2_by_arm.json["per_ctx_r2"]
    # for a genuine numeric Pearson + Spearman correlation with per-context length delta.
    # No try/except here: any internal error fails phase 5 loudly (fail-fast, #913).
    length_r2_correlation: dict[str, Any] = {}
    ridge_path = base_dir / "eval_results" / "issue_823" / "ridge_r2_by_arm.json"
    if ridge_path.exists() and span_path.exists():
        ridge_data = json.loads(ridge_path.read_text())
        valid_idx = _load_common_valid_idx(base_dir, n_contexts)
        length_r2_correlation = _length_r2_correlation(
            span_data, ridge_data.get("per_ctx_r2", {}), valid_idx
        )

    diag = {
        "text_cosine": {
            "mean": float(cos_per_ctx.mean()),
            "std": float(cos_per_ctx.std()),
            "frac_high_gt08": frac_high,
            "n_valid_pairs": n_valid,
        },
        "logp_distributions": logp_distributions,
        "activation_cosine": cos_by_trait,
        "length_stats": length_stats,
        "length_r2_correlation": length_r2_correlation,
        "n_contexts": n_contexts,
        "smoke": smoke,
        "ts": time.time(),
    }

    out_path = base_dir / "eval_results" / "issue_823" / "validity_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, default=_json_np))
    logger.info("Phase 5 complete: %s", out_path)
    log_phase("p5_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Upload raw completions to HF
# ═══════════════════════════════════════════════════════════════════════════════


def _upload_phase1_to_hf(base_dir: pathlib.Path) -> None:
    """Upload the three Phase 1 output files to HF data repo (harvest-path helper).

    Commits b2_seed42.json, b1_seed43.json, and common_valid_idx.json in a single
    HF commit and verifies each file is present on the Hub before returning.
    """
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    p1_dir = base_dir / "raw_completions" / "phase1"
    names = ["b2_seed42.json", "b1_seed43.json", "common_valid_idx.json"]
    files = [p1_dir / n for n in names]
    for f in files:
        assert f.exists() and f.stat().st_size > 0, f"Phase 1 file missing or empty: {f}"

    api = HfApi()
    ops = [
        CommitOperationAdd(
            path_in_repo=f"{ISSUE_SLUG}/raw_completions/phase1/{f.name}",
            path_or_fileobj=str(f),
        )
        for f in files
    ]
    api.create_commit(
        repo_id=HF_DATA_REPO_DATA,
        repo_type="dataset",
        commit_message=f"issue 823: Phase 1 harvest — {len(ops)} files",
        operations=ops,
    )
    hub_files = set(
        list_repo_files(
            repo_id=HF_DATA_REPO_DATA,
            repo_type="dataset",
        )
    )
    expected = {op.path_in_repo for op in ops}
    missing = expected - hub_files
    if missing:
        raise RuntimeError(
            f"HF Phase 1 upload verification FAIL: {len(missing)} files missing: {sorted(missing)}"
        )
    logger.info(
        "Phase 1 harvest: uploaded and Hub-verified %d files to %s", len(ops), HF_DATA_REPO_DATA
    )


def upload_raw_completions(base_dir: pathlib.Path) -> None:
    """Upload raw completion files to HF data repo."""
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    upload_raw_completions_to_data_repo(
        experiment_name=ISSUE_SLUG,
        eval_results_dir=base_dir / "eval_results" / "issue_823",
    )

    # Also upload the raw_completions dir explicitly (non-canonical structure)
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    raw_dir = base_dir / "raw_completions"
    files = list(raw_dir.rglob("*.json"))
    if files:
        ops = []
        for f in files:
            rel = f.relative_to(raw_dir)
            path_in_repo = f"{ISSUE_SLUG}/raw_completions/{rel}"
            ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))
        api.create_commit(
            repo_id=HF_DATA_REPO_DATA,
            repo_type="dataset",
            commit_message=f"issue 823: raw completions ({len(ops)} files)",
            operations=ops,
        )
        # Post-commit Hub verification
        from huggingface_hub import list_repo_files

        hub_files = set(
            list_repo_files(
                repo_id=HF_DATA_REPO_DATA,
                repo_type="dataset",
            )
        )
        expected = {op.path_in_repo for op in ops}
        missing = expected - hub_files
        if missing:
            raise RuntimeError(
                f"HF raw-completions upload verification FAIL: {len(missing)} files missing "
                f"on Hub after commit: {sorted(missing)[:3]}..."
            )
        logger.info("Uploaded and Hub-verified %d raw completion files", len(ops))


# ═══════════════════════════════════════════════════════════════════════════════
# Final sentinel
# ═══════════════════════════════════════════════════════════════════════════════


def write_final_sentinel(base_dir: pathlib.Path, smoke: bool) -> None:
    """Write the epm:results sentinel for poll_pipeline.py."""
    import subprocess
    import time as time_mod

    git_sha = "unknown"
    try:
        # Use the repo root (derived from __file__) so git rev-parse works correctly
        # on both local worktrees and pod /workspace checkouts (where base_dir=/workspace
        # may not be the repo clone).
        _repo_root = pathlib.Path(__file__).resolve().parents[4]
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(_repo_root), env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception as e:
        logger.warning("git sha lookup failed (provenance recorded as 'unknown'): %s", e)

    # Build reproducibility_card — required by poll_pipeline.py for non-training experiments;
    # documents HF artifact paths so verify_uploads.py can resolve them mechanically.
    reproducibility_card = {
        "hf_data_repo": HF_DATA_REPO_DATA,
        "issue_slug": ISSUE_SLUG,
        "analysis_tensors_prefix": f"{ISSUE_SLUG}/analysis_tensors/",
        "raw_completions_prefix": f"{ISSUE_SLUG}/raw_completions/",
        "eval_results_prefix": f"{ISSUE_SLUG}/eval_results/",
        "wandb_url": "n/a (no model training in this experiment)",
    }

    sentinel_path = pathlib.Path(
        f"/workspace/logs/issue-823-epm_results-{int(time_mod.time())}.json"
    )
    write_sentinel(
        sentinel_path,
        {
            "kind": "epm:results",
            "version": 1,
            "note": json.dumps(
                {
                    "status": "complete",
                    "smoke": smoke,
                    "issue": 823,
                    "git_sha": git_sha,
                    "eval_results": str(base_dir / "eval_results" / "issue_823"),
                    "hf_upload": ISSUE_SLUG,
                    "ts": time.time(),
                    "reproducibility_card": reproducibility_card,
                },
                default=_json_np,
            ),
        },
    )
    logger.info("Final sentinel written: %s", sentinel_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Issue #823: Per-context map h — own output vs context processing."
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: run on first 10 contexts only (same dispatcher, scaled down)",
    )
    p.add_argument(
        "--phase",
        choices=["0", "0.5", "1", "2", "3", "4", "5", "all", "cpu", "gpu"],
        default="all",
        help=(
            "Which phase(s) to run. 'all'=0+0.5+1+2+3+4+5, 'cpu'=0+1+2+4+5 (VM phases), "
            "'gpu'=0.5+3 (pod phases). Default: all."
        ),
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for all outputs (default: /workspace if exists, else repo root)",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF uploads (for local testing)",
    )
    p.add_argument(
        "--harvest-batch-ids",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Crash-recovery: path to a JSON file containing a list of Anthropic Message Batch "
            "IDs (or a dict with 'batch_ids' key) from a previous Phase 1 run.  When set, "
            "harvest_phase1_batches() is called instead of phase1_sonnet_gen(), then the "
            "harvested files are uploaded to HF and the script exits."
        ),
    )
    return p.parse_args()


def _phase1_outputs_exist(base_dir: pathlib.Path) -> bool:
    """Return True iff all three Phase 1 output files are present, valid JSON, and non-empty.

    Validates file existence, JSON parseability, and minimum record counts so that a
    truncated / corrupted file from a previous failed run does not silently trigger the
    resume skip.  b2_seed42.json and b1_seed43.json must each be non-empty JSON lists;
    common_valid_idx.json must contain a non-empty "common_valid_idx" list key.
    """
    p1_dir = base_dir / "raw_completions" / "phase1"

    def _valid_json_list(p: pathlib.Path) -> bool:
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("_phase1_outputs_exist: could not parse %s — treating as absent", p)
            return False
        return isinstance(data, list) and len(data) > 0

    def _valid_common_valid_idx(p: pathlib.Path) -> bool:
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("_phase1_outputs_exist: could not parse %s — treating as absent", p)
            return False
        return (
            isinstance(data, dict)
            and isinstance(data.get("common_valid_idx"), list)
            and len(data["common_valid_idx"]) > 0
        )

    if not _valid_json_list(p1_dir / "b2_seed42.json"):
        return False
    if not _valid_json_list(p1_dir / "b1_seed43.json"):
        return False
    return _valid_common_valid_idx(p1_dir / "common_valid_idx.json")


def _run_phases(
    run_phases: set[str],
    base_dir: pathlib.Path,
    n_contexts: int,
    smoke: bool,
    skip_upload: bool,
) -> None:
    """Execute the requested pipeline phases in order."""
    # Phase 0: substrate verify + prompt recon
    if "0" in run_phases:
        verify_record = phase0_verify(base_dir, n_contexts, smoke)
        logger.info("Phase 0 done: sha256_ok=%s", verify_record["sha256_ok"])

    # Phase 0.5: vLLM Qwen regeneration (GPU)
    if "0.5" in run_phases:
        phase05_vllm_regen(base_dir, n_contexts, smoke)

    # Phase 1: Sonnet generation (skip if output files already exist — resume path)
    if "1" in run_phases:
        if _phase1_outputs_exist(base_dir):
            logger.info(
                "Phase 1 output files already exist at %s — skipping (resume path)",
                base_dir / "raw_completions" / "phase1",
            )
        else:
            asyncio.run(phase1_sonnet_gen(base_dir, n_contexts, smoke))

    # Phase 2: Derangement
    if "2" in run_phases:
        phase2_derangement(base_dir, n_contexts, smoke)

    # Phase 3: TF extraction (GPU)
    if "3" in run_phases:
        phase3_tf_extract(base_dir, n_contexts, smoke)

    # Phase 4: Ridge refitting
    if "4" in run_phases:
        phase4_ridge_refit(base_dir, n_contexts, smoke)

    # Phase 5: Validity diagnostics
    if "5" in run_phases:
        phase5_validity_diag(base_dir, n_contexts, smoke)

    # Upload raw completions and eval results to HF
    if not skip_upload and ("3" in run_phases or "5" in run_phases):
        log_phase("p_upload")
        upload_raw_completions(base_dir)

    # Write final sentinel (only when running full or GPU pipeline on pod)
    if not skip_upload and "3" in run_phases:
        write_final_sentinel(base_dir, smoke)


def main() -> None:
    """Main dispatcher."""
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    n_contexts = N_SMOKE if args.smoke else N_CONTEXTS_FULL

    logger.info(
        "Issue 823 dispatcher: phase=%s, smoke=%s, n_contexts=%d, base_dir=%s",
        args.phase,
        args.smoke,
        n_contexts,
        base_dir,
    )

    # ── Crash-recovery harvest path ──────────────────────────────────────────
    # When --harvest-batch-ids is given, reconstruct Phase 1 output files from
    # the already-submitted Anthropic batches, upload to HF, then exit.
    if args.harvest_batch_ids:
        batch_ids_file = pathlib.Path(args.harvest_batch_ids)
        assert batch_ids_file.exists(), f"--harvest-batch-ids file not found: {batch_ids_file}"
        harvest_phase1_batches(base_dir, batch_ids_file, n_contexts)
        if not args.skip_upload:
            log_phase("p1_harvest_upload")
            _upload_phase1_to_hf(base_dir)
        log_phase("done")
        logger.info("Phase 1 harvest complete.  Exiting.")
        return

    # Determine which phases to run
    run_phases: set[str] = set()
    if args.phase == "all":
        run_phases = {"0", "0.5", "1", "2", "3", "4", "5"}
    elif args.phase == "cpu":
        run_phases = {"0", "1", "2", "4", "5"}
    elif args.phase == "gpu":
        run_phases = {"0.5", "3"}
    else:
        run_phases = {args.phase}

    try:
        _run_phases(run_phases, base_dir, n_contexts, args.smoke, args.skip_upload)
    except Exception:
        logger.exception("Dispatcher failed")
        raise

    log_phase("done")
    logger.info("Issue 823 dispatcher complete.")


if __name__ == "__main__":
    main()
